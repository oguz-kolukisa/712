#!/usr/bin/env python
"""
Fine-tune BLIP-2 Q-Former on VQA-v2 (default 5 epochs, 1e-5 LR, 490 px images).
"""

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

# ───────────────────────── utils ────────────────────────────────────────────
def freeze_all(m):                [setattr(p, "requires_grad", False) for p in m.parameters()]
def freeze_but_qformer(m):        [setattr(p, "requires_grad", "qformer" in n) for n, p in m.named_parameters()]

def load_blip2_llama(blip_opt, llama, device, img_size):
    blip = Blip2ForConditionalGeneration.from_pretrained(blip_opt)
    lm   = AutoModelForCausalLM.from_pretrained(llama)
    tok  = AutoTokenizer.from_pretrained(llama, use_fast=True)

    if "<image>" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["<image>"]})
    img_tok = tok.convert_tokens_to_ids("<image>")

    cfg = Blip2Config.from_dict(blip.config.to_dict())
    cfg.text_config = lm.config
    cfg.image_token_id = img_tok

    model = Blip2ForConditionalGeneration(cfg)
    model.vision_model.load_state_dict(blip.vision_model.state_dict())
    model.qformer.load_state_dict(blip.qformer.state_dict())
    model.language_model.load_state_dict(lm.state_dict())

    if model.get_input_embeddings().num_embeddings < len(tok):
        model.resize_token_embeddings(len(tok))

    processor = Blip2Processor.from_pretrained(blip_opt)
    processor.tokenizer = tok
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.image_processor.size = {"height": img_size, "width": img_size}
    model.to(device)
    return model, processor

PROMPT = "Question: {} Answer:"

class VQADataset(Dataset):
    def __init__(self, split): self.d = split
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        ex = self.d[i]
        q  = ex["question"].strip()
        q += "" if q.endswith("?") else "?"
        return {"image": ex["image"].convert("RGB"),
                "prompt": PROMPT.format(q),
                "answer": ex["multiple_choice_answer"].strip()}

def make_collate(proc):
    pad = proc.tokenizer.pad_token_id
    def collate(batch):
        enc = proc(images=[b["image"] for b in batch],
                   text =[f"{b['prompt']} {b['answer']}" for b in batch],
                   padding=True, return_tensors="pt")
        lbl = enc["input_ids"].clone()
        for i, b in enumerate(batch):
            n = len(proc.tokenizer(b["prompt"], add_special_tokens=False).input_ids)
            lbl[i, :n] = -100
        lbl[lbl == pad] = -100
        enc["labels"] = lbl
        enc["interpolate_pos_encoding"] = True   # ← key fix
        return enc
    return collate

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--llama_name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--blip2_opt_name", default="Salesforce/blip2-opt-2.7b")
    p.add_argument("--output_dir", default="./blip2-llama-vqa-checkpoints-qformer")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--tuning_mode", choices=["full", "lora"], default="full")
    return p.parse_args()

def main():
    a = args()
    rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    train_raw = load_dataset("HuggingFaceM4/VQAv2", split="train", trust_remote_code=True,
                             storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}})
    val_raw   = load_dataset("HuggingFaceM4/VQAv2", split="validation", trust_remote_code=True,
                             storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}})

    model, proc = load_blip2_llama(a.blip2_opt_name, a.llama_name, device, img_size=490)
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    proc.save_pretrained(a.output_dir)

    if a.tuning_mode == "full": freeze_but_qformer(model)
    else:
        freeze_all(model)
        model.qformer = get_peft_model(model.qformer, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"]))

    collate = make_collate(proc)
    cfg = SFTConfig(output_dir=a.output_dir, per_device_train_batch_size=a.batch_size,
                    num_train_epochs=a.epochs, warmup_steps=a.warmup,
                    learning_rate=a.lr, weight_decay=a.weight_decay,
                    adam_beta1=0.9, adam_beta2=0.999, fp16=torch.cuda.is_available(),
                    logging_steps=50, save_steps=1000, save_total_limit=2,
                    eval_strategy="steps", eval_steps=5000, remove_unused_columns=False,
                    dataset_kwargs={"skip_prepare_dataset": True})

    trainer = SFTTrainer(model=model, args=cfg,
                         train_dataset=VQADataset(train_raw),
                         eval_dataset=VQADataset(val_raw),
                         data_collator=collate)

    ckpt = get_last_checkpoint(a.output_dir) if os.path.isdir(a.output_dir) else None
    trainer.train(resume_from_checkpoint=ckpt) if ckpt else trainer.train()
    trainer.save_model(a.output_dir)
    if rank == 0: print(f"\n✅  Saved to {a.output_dir}\n")

if __name__ == "__main__": main()
