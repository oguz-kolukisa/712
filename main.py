#!/usr/bin/env python
"""
train_blip2_llama_vqa_qformer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fine-tune **only the Q-Former** of a custom BLIP-2 model with a LLaMA-3.1
backend on the VQA-v2 dataset using Hugging Face TRL’s SFTTrainer.

Key points
----------
* Dataset/loader rewritten to mirror the football-dataset example:
    - `VQADataset.__getitem__` returns *pixel_values* + a `"text"` string.
    - `collate_fn` stacks `pixel_values`, tokenises the `"text"`, and
      creates `input_ids / attention_mask / labels`.
* Everything else (LoRA, multi-GPU, checkpoint resume, etc.) is unchanged.
"""

# ───────────────────────────────────────────────────────────────────────────────
# Imports
# ───────────────────────────────────────────────────────────────────────────────
import argparse
import os
from pathlib import Path
import aiohttp
import torch
from torch.utils.data import Dataset
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
    """Disable gradient updates for every parameter."""
    for p in model.parameters():
        p.requires_grad = False


def freeze_everything_but_qformer(model: Blip2ForConditionalGeneration):
    """Leave only the Q-Former trainable."""
    for name, p in model.named_parameters():
        p.requires_grad = "qformer" in name


def load_blip2_llama(blip2_opt_name: str, llama_name: str, device: torch.device):
    """
    Build a BLIP-2 model whose vision & Q-Former come from the OPT
    checkpoint while the language model is swapped out for LLaMA-3.1.
    """
    # 1) Source checkpoints ----------------------------------------------------
    blip2_opt = Blip2ForConditionalGeneration.from_pretrained(blip2_opt_name)
    llama_model = AutoModelForCausalLM.from_pretrained(llama_name)
    llama_tok   = AutoTokenizer.from_pretrained(llama_name, use_fast=True)

    # 2) Merge configs ---------------------------------------------------------
    new_cfg = Blip2Config.from_dict(blip2_opt.config.to_dict())
    new_cfg.text_config = llama_model.config  # plug LLaMA text config

    # 3) Fresh shell & weight transfer ----------------------------------------
    model = Blip2ForConditionalGeneration(new_cfg)
    model.vision_model.load_state_dict(blip2_opt.vision_model.state_dict())
    model.qformer.load_state_dict(blip2_opt.qformer.state_dict())
    model.language_model.load_state_dict(llama_model.state_dict())

    # 4) Processor: keep BLIP-2 processors, swap tokenizer ---------------------
    processor = Blip2Processor.from_pretrained(blip2_opt_name)
    processor.tokenizer = llama_tok
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model.to(device)
    return model, processor

# ───────────────────────────────────────────────────────────────────────────────
# Dataset
# ───────────────────────────────────────────────────────────────────────────────
class VQADataset(Dataset):
    """
    *Exactly* mirrors the football-dataset style requested:

        encoding = processor(images=item["image"], ...)
        encoding["text"] = item["text"]
    """
    def __init__(self, hf_split, processor):
        self.dataset   = hf_split     # raw Hugging Face split
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Vision preprocessing (batch dim removed)
        encoding = self.processor(
            images=item["image"].convert("RGB"),
            padding="max_length",
            return_tensors="pt",
        )
        encoding = {k: v.squeeze() for k, v in encoding.items()}  # drop batch dim

        # Build the textual target in one string
        question = item["question"].strip()
        if not question.endswith("?"):
            question += "?"
        answer = item["multiple_choice_answer"].strip()

        # SINGLE field "text" (just like the caption example)
        encoding["text"] = f"Question: {question} Answer: {answer}"
        return encoding

# ───────────────────────────────────────────────────────────────────────────────
# Collate function
# ───────────────────────────────────────────────────────────────────────────────
def make_vqa_collate_fn(processor):
    """
    Stacks `pixel_values`, tokenises the `"text"` field, and builds labels
    identical to `input_ids` (causal-LM style).
    """
    def collate(batch):
        processed = {}

        # 1) Stack vision tensors ---------------------------------------------
        processed["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])

        # 2) Tokenise text -----------------------------------------------------
        text_inputs = processor.tokenizer(
            [b["text"] for b in batch],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        processed["input_ids"]      = text_inputs["input_ids"]
        processed["attention_mask"] = text_inputs["attention_mask"]

        # 3) Labels = input_ids (no masking this time) -------------------------
        processed["labels"] = text_inputs["input_ids"].clone()
        return processed

    return collate

# ───────────────────────────────────────────────────────────────────────────────
# Arg-parse & main
# ───────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama_name",
                    default="meta-llama/Llama-3.1-8B-Instruct",
                    help="HF model ID or local path to LLaMA-3.1")
    ap.add_argument("--blip2_opt_name",
                    default="Salesforce/blip2-opt-2.7b",
                    help="Base BLIP-2 OPT checkpoint")
    ap.add_argument("--output_dir",
                    default="./blip2-llama-vqa-checkpoints-qformer")
    ap.add_argument("--epochs",      type=int,   default=1)
    ap.add_argument("--batch_size",  type=int,   default=32)
    ap.add_argument("--lr",          type=float, default=2e-5)
    ap.add_argument("--tuning_mode", choices=["full", "lora"], default="lora",
                    help="Fine-tune strategy for the Q-Former")
    return ap.parse_args()


def main():
    args = parse_args()

    # Rank / device ------------------------------------------------------------
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}"
                          if torch.cuda.is_available() else "cpu")

    # 1) Raw splits ------------------------------------------------------------
    train_raw = load_dataset(
        "HuggingFaceM4/VQAv2",
        split="train",
        trust_remote_code=True,
        storage_options={"client_kwargs":
                         {"timeout": aiohttp.ClientTimeout(total=3600)}},
    )
    val_raw = load_dataset(
        "HuggingFaceM4/VQAv2",
        split="validation",
        trust_remote_code=True,
        storage_options={"client_kwargs":
                         {"timeout": aiohttp.ClientTimeout(total=3600)}},
    )

    # 2) Model & processor -----------------------------------------------------
    model, processor = load_blip2_llama(
        args.blip2_opt_name, args.llama_name, device
    )

    # 3) Tune mode -------------------------------------------------------------
    if args.tuning_mode == "full":
        freeze_everything_but_qformer(model)
    else:  # LoRA adapters inside Q-Former
        freeze_everything(model)
        lora_cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=[
                "query", "key", "value",  # attention projections

            ],
        )
        model.qformer = get_peft_model(model.qformer, lora_cfg)
        model.qformer.print_trainable_parameters()

    # 4) Dataset wrappers + collate -------------------------------------------
    train_ds = VQADataset(train_raw, processor)
    val_ds   = VQADataset(val_raw,   processor)
    collate_fn = make_vqa_collate_fn(processor)

    # 5) SFT config ------------------------------------------------------------
    sft_cfg = SFTConfig(
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
        dataset_text_field="text",          # irrelevant but required arg
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # 6) Trainer ---------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )

    # 7) Resume or train -------------------------------------------------------
    ckpt = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    trainer.train(resume_from_checkpoint=ckpt) if ckpt else trainer.train()

    # 8) Save ------------------------------------------------------------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    processor.tokenizer.save_pretrained(args.output_dir)

    if local_rank == 0:
        print(f"\n✅  Fine-tuned Q-Former saved to {args.output_dir}\n")


if __name__ == "__main__":
    main()
