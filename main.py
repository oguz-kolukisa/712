#!/usr/bin/env python
"""
train_blip2_llama_vqa_qformer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fine‑tunes **only the Q‑Former** of a custom BLIP‑2 model with a LLaMA‑3.1
backend on the VQA‑v2 dataset, using Hugging Face **TRL SFTTrainer**.

🏎️  **Multi‑GPU ready (2×GPU)**
    Launch with either `torchrun` (PyTorch DDP) or `accelerate launch`, e.g.:

        torchrun --standalone --nproc_per_node=2 train_blip2_llama_vqa_qformer.py \
            --llama_name /path/to/llama-3.1-7b

All non‑Q‑Former weights (vision encoder + language model) are frozen, so the
script uses very little GPU memory per device.
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
    TrainingArguments,
)
from trl import SFTTrainer

# ────────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────────

def freeze_everything_but_qformer(model: Blip2ForConditionalGeneration):
    """Freeze *all* parameters except those belonging to the Q‑Former."""
    for name, param in model.named_parameters():
        param.requires_grad = "qformer" in name  # train only Q‑Former


def load_blip2_llama(blip2_opt_name: str, llama_name: str, device: torch.device):
    """Load BLIP‑2 vision + Q‑Former weights and swap the text model to LLaMA‑3.1."""
    # 1) Source BLIP‑2 (OPT) → vision + Q‑Former
    blip2_opt = Blip2ForConditionalGeneration.from_pretrained(blip2_opt_name)

    # 2) Target LLaMA‑3.1 (causal LM) + tokenizer
    llama_model = AutoModelForCausalLM.from_pretrained(llama_name)
    llama_tok   = AutoTokenizer.from_pretrained(llama_name, use_fast=True)

    # 3) Compose new config whose text_config is LLaMA’s
    new_cfg             = Blip2Config.from_dict(blip2_opt.config.to_dict())
    new_cfg.text_config = llama_model.config

    # 4) Fresh BLIP‑2 shell with that config
    blip2_llama = Blip2ForConditionalGeneration(new_cfg)

    # 5) Weight transfer
    blip2_llama.vision_model.load_state_dict(blip2_opt.vision_model.state_dict())
    blip2_llama.qformer.load_state_dict(blip2_opt.qformer.state_dict())
    blip2_llama.language_model.load_state_dict(llama_model.state_dict())

    # 6) Processor with LLaMA tokenizer
    processor           = Blip2Processor.from_pretrained(blip2_opt_name)
    processor.tokenizer = llama_tok

    blip2_llama.to(device)
    return blip2_llama, processor


def vqa_collate_fn_factory(processor, device):
    """Return a collate function that prepares image + question → answer pairs."""

    def collate(batch):
        # Extract fields
        images    = [ex["image"].convert("RGB") for ex in batch]
        questions = [ex["question"].strip() for ex in batch]
        answers   = [ex["multiple_choice_answer"].strip() for ex in batch]

        # Build prompt: "Question: …? Answer:"
        prompts, prompt_lens = [], []
        for q in questions:
            if not q.endswith("?"):
                q += "?"
            prompt = f"Question: {q} Answer:"
            prompts.append(prompt)
            prompt_lens.append(
                len(processor.tokenizer(prompt, add_special_tokens=False).input_ids)
            )

        # Full text = prompt + ground‑truth answer
        full_texts = [f"{p} {a}" for p, a in zip(prompts, answers)]

        # Processor handles both modalities → tensors
        enc = processor(
            images=images,
            text=full_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Labels: mask prompt + padding tokens with −100
        labels = enc["input_ids"].clone()
        pad_id = processor.tokenizer.pad_token_id
        for i, l in enumerate(prompt_lens):
            labels[i, :l] = -100
        labels[labels == pad_id] = -100
        enc["labels"] = labels

        # Move everything to the correct device (per‑process GPU in DDP)
        return {k: v.to(device) for k, v in enc.items()}

    return collate


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_name", default="meta-llama/Llama-3.1-8B-Instruct", help="HF path or local dir of LLaMA‑3.1")
    parser.add_argument("--blip2_opt_name", default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--output_dir", default="./blip2-llama-vqa-checkpoints-qformer")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    # Detect local rank in DDP / Accelerate; default CUDA:0
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device     = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 1) Datasets
    train_ds = load_dataset("HuggingFaceM4/VQAv2", split="train", trust_remote_code=True,    storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
    val_ds   = load_dataset("HuggingFaceM4/VQAv2", split="validation", trust_remote_code=True,    storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})

    # 2) Model & processor
    model, processor = load_blip2_llama(args.blip2_opt_name, args.llama_name, device)

    # 3) Freeze everything except Q‑Former
    freeze_everything_but_qformer(model)

    

    # 4) Collator
    collate_fn = vqa_collate_fn_factory(processor, device)

    # 5) Training arguments (Accelerate handles multi‑GPU automatically)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        logging_steps=500,
        save_steps=1000,
        save_total_limit=2,
        eval_strategy="steps",
        remove_unused_columns=False,
    )

    # 6) SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,  # ensures tokenizer is saved
        dataset_kwargs={"skip_prepare_dataset": True},

    )

    # 7) Fine‑tune (only Q‑Former trainable)
    trainer.train()

    # 8) Save final artefacts
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    processor.tokenizer.save_pretrained(args.output_dir)
    if local_rank == 0:
        print(f"\n✅  Q‑Former‑only fine‑tuned model saved to {args.output_dir}\n")


if __name__ == "__main__":
    main()
