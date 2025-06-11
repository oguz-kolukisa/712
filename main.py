#!/usr/bin/env python
"""
train_blip2_llama_vqa_qformer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fine‑tunes **only the Q‑Former** of a custom BLIP‑2 model that uses a
LLaMA‑3.1 backend on the VQA‑v2 dataset with Hugging Face **TRL
SFTTrainer**.

Key features
------------
* 🏎️  **Multi‑GPU ready** ― use `torchrun` or `accelerate launch`.
* 🗄  **Two tuning modes** controlled by `--tuning_mode {full,lora}`
  * `full` (default): all Q‑Former weights are trainable (vision encoder
    + LLM stay frozen).
  * `lora`: inject LoRA adapters *inside* the Q‑Former and train **only
    those adapters** (massively reduces GPU memory).

Example runs
~~~~~~~~~~~~
Full Q‑Former fine‑tune on two GPUs:

    torchrun --standalone --nproc_per_node=2 train_blip2_llama_vqa_qformer.py \
        --llama_name /path/to/llama-3.1-8b \
        --tuning_mode full

LoRA‑only adapters inside the Q‑Former (tiny trainable footprint):

    torchrun --standalone --nproc_per_node=2 train_blip2_llama_vqa_qformer.py \
        --llama_name /path/to/llama-3.1-8b \
        --tuning_mode lora

Dependencies
~~~~~~~~~~~~
    pip install torch datasets transformers peft trl aiohttp
"""

import argparse
import os
from pathlib import Path
import aiohttp
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Blip2Config,
    Blip2ForConditionalGeneration,
    Blip2Processor,
)
from trl import SFTTrainer, SFTConfig
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model

# ───────────────────────────────────────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────────────────────────────────────


def freeze_everything(model: torch.nn.Module):
    """Set `requires_grad = False` for *all* parameters."""
    for p in model.parameters():
        p.requires_grad = False


def freeze_everything_but_qformer(model: Blip2ForConditionalGeneration):
    """Freeze parameters outside the Q‑Former (i.e. vision encoder + LLM)."""
    for name, param in model.named_parameters():
        param.requires_grad = "qformer" in name


def load_blip2_llama(blip2_opt_name: str, llama_name: str, device: torch.device):
    """Load BLIP‑2 vision+Q‑Former weights and swap the text model to LLaMA‑3.1."""
    # 1) Source BLIP‑2 (OPT) → vision + Q‑Former
    blip2_opt = Blip2ForConditionalGeneration.from_pretrained(blip2_opt_name)

    # 2) Target LLaMA‑3.1 (causal LM) + tokenizer
    llama_model = AutoModelForCausalLM.from_pretrained(llama_name)
    llama_tok = AutoTokenizer.from_pretrained(llama_name, use_fast=True)

    # 3) Compose new config whose text_config is taken from LLaMA
    new_cfg = Blip2Config.from_dict(blip2_opt.config.to_dict())
    new_cfg.text_config = llama_model.config

    # 4) Fresh BLIP‑2 shell with the merged config
    blip2_llama = Blip2ForConditionalGeneration(new_cfg)

    # 5) Weight transfer
    blip2_llama.vision_model.load_state_dict(blip2_opt.vision_model.state_dict())
    blip2_llama.qformer.load_state_dict(blip2_opt.qformer.state_dict())
    blip2_llama.language_model.load_state_dict(llama_model.state_dict())

    # 6) Processor: use original BLIP‑2 image processors + LLaMA tokenizer
    processor = Blip2Processor.from_pretrained(blip2_opt_name)
    processor.tokenizer = llama_tok
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    blip2_llama.to(device)
    return blip2_llama, processor


def vqa_collate_fn_factory(processor):
    """Return a collate function that prepares (image, question) → answer pairs."""

    def collate(batch):
        # Images & text fields
        images = [ex["image"].convert("RGB") for ex in batch]
        questions = [ex["question"].strip() for ex in batch]
        answers = [ex["multiple_choice_answer"].strip() for ex in batch]

        # Prompt template: "Question: …? Answer:"
        prompts, prompt_lens = [], []
        for q in questions:
            if not q.endswith("?"):
                q += "?"
            prompt = f"Question: {q} Answer:"
            prompts.append(prompt)
            prompt_lens.append(
                len(processor.tokenizer(prompt, add_special_tokens=False).input_ids)
            )

        full_texts = [f"{p} {a}" for p, a in zip(prompts, answers)]

        enc = processor(
            images=images,
            text=full_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Mask prompt & padding tokens in labels
        labels = enc["input_ids"].clone()
        pad_id = processor.tokenizer.pad_token_id
        for i, l in enumerate(prompt_lens):
            labels[i, :l] = -100
        labels[labels == pad_id] = -100
        enc["labels"] = labels

        return enc

    return collate


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_name", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HF model ID or local path to LLaMA‑3.1")
    parser.add_argument("--blip2_opt_name", default="Salesforce/blip2-opt-2.7b",
                        help="Base BLIP‑2 OPT checkpoint")
    parser.add_argument("--output_dir", default="./blip2-llama-vqa-checkpoints-qformer")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--tuning_mode", choices=["full", "lora"], default="lora",
                        help="Fine‑tune strategy for the Q‑Former")
    return parser.parse_args()


def main():
    args = parse_args()

    # Detect local rank in DDP / Accelerate; fall back to CUDA:0 or CPU
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 1) Load datasets (30‑min HF cache timeout for first download)
    train_ds = load_dataset(
        "HuggingFaceM4/VQAv2", split="train", trust_remote_code=True,
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}},
    )
    val_ds = load_dataset(
        "HuggingFaceM4/VQAv2", split="validation", trust_remote_code=True,
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}},
    )

    # 2) Build model & processor
    model, processor = load_blip2_llama(args.blip2_opt_name, args.llama_name, device)

    # 3) Choose tuning mode ------------------------------------------------------
    if args.tuning_mode == "full":
        freeze_everything_but_qformer(model)
    else:  # LoRA mode
        # Freeze everything first (including Q‑Former)
        freeze_everything(model)

        # LoRA configuration (feel free to adjust)
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",  # Q‑Former is transformer‑decoder‑like
            target_modules=[
                "q_proj", "k_proj", "v_proj",  # attention projections
                "out_proj",                      # attn output
                "fc1", "fc2",                  # MLP layers
            ],
        )

        # Inject adapters only into the Q‑Former
        model.qformer = get_peft_model(model.qformer, lora_cfg)
        model.qformer.print_trainable_parameters()  # sanity log

    # 4) Data collator
    collate_fn = vqa_collate_fn_factory(processor)

    # 5) SFT/Trainer config ------------------------------------------------------
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_steps=1000,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=5000,
        remove_unused_columns=False,
        dataset_text_field="question",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # 6) Trainer ---------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )

    # 7) Resume from last checkpoint if present ---------------------------------
    last_ckpt = None
    if os.path.isdir(args.output_dir):
        last_ckpt = get_last_checkpoint(args.output_dir)

    if last_ckpt is not None:
        print(f"↩️  Resuming from checkpoint {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        trainer.train()

    # 8) Save final artefacts ----------------------------------------------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    processor.tokenizer.save_pretrained(args.output_dir)

    if local_rank == 0:
        print(f"\n✅  Fine‑tuned Q‑Former saved to {args.output_dir}\n")


if __name__ == "__main__":
    main()
