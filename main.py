#!/usr/bin/env python
"""
Fine-tune only the Q-Former of a BLIP-2 + LLaMA-3.1 model on VQA-v2
(using TRL SFTTrainer).

Changes vs. previous draft
--------------------------
• Adds "<image>" to the LLaMA tokenizer and resizes token embeddings.
• Saves the *full* processor early.
• Dataset returns raw PIL images; collate uses processor(images=…, text=…).
• Masks prompt tokens in the labels (loss on answers only).
"""

# ───────────────────────────────────────────────────────────────────────────────
# Imports
# ───────────────────────────────────────────────────────────────────────────────
import argparse, os
from pathlib import Path
import aiohttp, torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Blip2Config, Blip2ForConditionalGeneration, Blip2Processor,
)
from trl import SFTTrainer, SFTConfig
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model

# ───────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ───────────────────────────────────────────────────────────────────────────────
def freeze_everything(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def freeze_everything_but_qformer(model: Blip2ForConditionalGeneration):
    for n, p in model.named_parameters():
        p.requires_grad = "qformer" in n


def load_blip2_llama(blip2_opt_name: str, llama_name: str, device):
    # 1) Load source checkpoints ------------------------------------------------
    blip2_opt  = Blip2ForConditionalGeneration.from_pretrained(blip2_opt_name)
    llama_lm   = AutoModelForCausalLM.from_pretrained(llama_name)
    llama_tok  = AutoTokenizer.from_pretrained(llama_name, use_fast=True)

    # 2) Ensure "<image>" token is present -------------------------------------
    if "<image>" not in llama_tok.get_vocab():
        llama_tok.add_special_tokens({"additional_special_tokens": ["<image>"]})
    image_tok_id = llama_tok.convert_tokens_to_ids("<image>")

    # 3) Merge configs ----------------------------------------------------------
    new_cfg = Blip2Config.from_dict(blip2_opt.config.to_dict())
    new_cfg.text_config    = llama_lm.config
    new_cfg.image_token_id = image_tok_id

    # 4) Fresh BLIP-2 shell & weight transfer ----------------------------------
    model = Blip2ForConditionalGeneration(new_cfg)
    model.vision_model.load_state_dict(blip2_opt.vision_model.state_dict())
    model.qformer.load_state_dict(blip2_opt.qformer.state_dict())
    model.language_model.load_state_dict(llama_lm.state_dict())

    # 5) Resize embeddings to fit new vocab ------------------------------------
    needed = len(llama_tok)
    if model.get_input_embeddings().weight.size(0) < needed:
        model.resize_token_embeddings(needed)

    # 6) Build processor (vision from BLIP-2, tokenizer = patched LLaMA) -------
    processor = Blip2Processor.from_pretrained(blip2_opt_name)
    processor.tokenizer = llama_tok
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model.to(device)
    return model, processor

# ───────────────────────────────────────────────────────────────────────────────
# Dataset (returns PIL image + prompt string)
# ───────────────────────────────────────────────────────────────────────────────
class VQADataset(Dataset):
    def __init__(self, hf_split):
        self.data = hf_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        q  = ex["question"].strip()
        if not q.endswith("?"):
            q += "?"
        prompt = f"Question: {q} Answer:"          # model should generate answer
        ans    = ex["multiple_choice_answer"].strip()
        return {
            "image":  ex["image"].convert("RGB"),
            "prompt": prompt,
            "answer": ans,
        }

# ───────────────────────────────────────────────────────────────────────────────
# Collate
# ───────────────────────────────────────────────────────────────────────────────
def make_collate_fn(proc):
    def collate(batch):
        # Use processor to inject <image> token and produce pixel_values
        enc = proc(
            images=[b["image"] for b in batch],
            text=[f'{b["prompt"]} {b["answer"]}' for b in batch],
            padding=True,
            return_tensors="pt",
        )

        # Build labels: mask prompt tokens
        labels = enc["input_ids"].clone()
        pad_id = proc.tokenizer.pad_token_id
        for i, (prompt, _) in enumerate(
            zip([b["prompt"] for b in batch], batch)
        ):
            n_prompt = len(proc.tokenizer(prompt, add_special_tokens=False).input_ids)
            labels[i, :n_prompt] = -100
        labels[labels == pad_id] = -100
        enc["labels"] = labels
        return enc
    return collate

# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llama_name",      default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--blip2_opt_name",  default="Salesforce/blip2-opt-2.7b")
    p.add_argument("--output_dir",      default="./blip2-llama-vqa-checkpoints-qformer")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float,  default=4e-5)
    p.add_argument("--tuning_mode", choices=["full", "lora"], default="lora")
    p.add_argument("--max_steps",   type=int,   default=-1,
                     help="Terminate training after this many update steps "
                          "(overrides --epochs when > 0)")
    return p.parse_args()


def main():
    args = parse_args()
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 1) HF splits -------------------------------------------------------------
    train_raw = load_dataset("HuggingFaceM4/VQAv2", split="train",
                             trust_remote_code=True,
                             storage_options={"client_kwargs":
                                 {"timeout": aiohttp.ClientTimeout(total=3600)}})
    val_raw   = load_dataset("HuggingFaceM4/VQAv2", split="validation",
                             trust_remote_code=True,
                             storage_options={"client_kwargs":
                                 {"timeout": aiohttp.ClientTimeout(total=3600)}})

    # 2) Model & processor -----------------------------------------------------
    model, processor = load_blip2_llama(args.blip2_opt_name, args.llama_name, device)

    # Save processor early so evaluation always finds it
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(args.output_dir)

    # 3) Tuning mode -----------------------------------------------------------
    if args.tuning_mode == "full":
        freeze_everything_but_qformer(model)
    else:
        freeze_everything(model)
        lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                              bias="none",
                              target_modules=[ "query", "key", "value"])
        model.qformer = get_peft_model(model.qformer, lora_cfg)
        model.qformer.print_trainable_parameters()

    # 4) Datasets + collate ----------------------------------------------------
    train_ds = VQADataset(train_raw)
    val_ds   = VQADataset(val_raw)
    collate  = make_collate_fn(processor)

    # 5) Trainer ---------------------------------------------------------------
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
        dataset_kwargs={"skip_prepare_dataset": True},
        max_steps=args.max_steps,
    )
    trainer = SFTTrainer(model=model, args=sft_cfg,
                         train_dataset=train_ds,
                         eval_dataset=val_ds,
                         data_collator=collate)

    ckpt = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    trainer.train(resume_from_checkpoint=ckpt) if ckpt else trainer.train()

    trainer.save_model(args.output_dir)
    if local_rank == 0:
        print(f"\n✅  Q-Former fine-tuned & saved to {args.output_dir}\n")


if __name__ == "__main__":
    main()
