#!/usr/bin/env python
"""
train_blip2_llama_vqa_qformer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fine-tune the BLIP-2 Q-Former on VQA-v2 with hand-picked defaults:

  • epochs ............. 5
  • warm-up steps ...... 1000
  • learning rate ...... 1e-5
  • batch size ......... 128
  • AdamW betas ........ (0.9, 0.999)
  • weight decay ....... 0.05
  • image resolution ... 490 px  (shortest edge)
  • prompt ............. "Question: {} Answer:"
"""

# ───────────────────── imports ──────────────────────────────────────────────
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

# ───────────────────── helper utils ─────────────────────────────────────────
def freeze_everything(model):                       # vision + LLM frozen
    for p in model.parameters(): p.requires_grad = False
def freeze_but_qformer(model):
    for n, p in model.named_parameters():
        p.requires_grad = "qformer" in n

def load_blip2_llama(blip_opt, llama, device, img_size):
    # 1) source checkpoints
    blip = Blip2ForConditionalGeneration.from_pretrained(blip_opt)
    llama_lm  = AutoModelForCausalLM.from_pretrained(llama)
    llama_tok = AutoTokenizer.from_pretrained(llama, use_fast=True)

    # 2) add "<image>" if missing
    if "<image>" not in llama_tok.get_vocab():
        llama_tok.add_special_tokens({"additional_special_tokens": ["<image>"]})
    image_tok_id = llama_tok.convert_tokens_to_ids("<image>")

    # 3) merged config
    cfg = Blip2Config.from_dict(blip.config.to_dict())
    cfg.text_config    = llama_lm.config
    cfg.image_token_id = image_tok_id

    model = Blip2ForConditionalGeneration(cfg)
    model.vision_model.load_state_dict(blip.vision_model.state_dict())
    model.qformer.load_state_dict(blip.qformer.state_dict())
    model.language_model.load_state_dict(llama_lm.state_dict())

    # 4) resize embeddings after adding <image>
    needed = len(llama_tok)
    if model.get_input_embeddings().num_embeddings < needed:
        model.resize_token_embeddings(needed)

    # 5) processor
    proc = Blip2Processor.from_pretrained(blip_opt)
    proc.tokenizer = llama_tok
    proc.tokenizer.pad_token = proc.tokenizer.eos_token
    # set target image size (shortest-edge)
    proc.image_processor.size["shortest_edge"] = img_size

    model.to(device)
    return model, proc

# ───────────────────── dataset & collate ────────────────────────────────────
PROMPT_TMPL = "Question: {} Answer:"

class VQADataset(Dataset):
    def __init__(self, hf_split): self.d = hf_split
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        ex = self.d[i]
        q  = ex["question"].strip()
        if not q.endswith("?"): q += "?"
        return {
            "image": ex["image"].convert("RGB"),
            "prompt": PROMPT_TMPL.format(q),
            "answer": ex["multiple_choice_answer"].strip(),
        }

def make_collate(proc):
    pad_id = proc.tokenizer.pad_token_id
    def collate(batch):
        images  = [b["image"]  for b in batch]
        texts   = [f"{b['prompt']} {b['answer']}" for b in batch]
        enc = proc(images=images, text=texts, padding=True, return_tensors="pt")

        # mask prompt tokens so loss computed only on answer
        labels = enc["input_ids"].clone()
        for i, b in enumerate(batch):
            n_prompt = len(proc.tokenizer(b["prompt"], add_special_tokens=False).input_ids)
            labels[i, :n_prompt] = -100
        labels[labels == pad_id] = -100
        enc["labels"] = labels
        return enc
    return collate

# ───────────────────── argument parser ──────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llama_name",     default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--blip2_opt_name", default="Salesforce/blip2-opt-2.7b")
    p.add_argument("--output_dir",     default="./blip2-llama-vqa-checkpoints-qformer")
    # defaults per request
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=1e-5)
    p.add_argument("--warmup",      type=int,   default=1000)
    p.add_argument("--weight_decay",type=float, default=0.05)
    p.add_argument("--tuning_mode", choices=["full", "lora"], default="full")
    return p.parse_args()

# ───────────────────── main ────────────────────────────────────────────────
def main():
    a = get_args()
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # splits
    train_raw = load_dataset("HuggingFaceM4/VQAv2", split="train", trust_remote_code=True,
                             storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}})
    val_raw   = load_dataset("HuggingFaceM4/VQAv2", split="validation", trust_remote_code=True,
                             storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}})

    # model & processor
    model, proc = load_blip2_llama(a.blip2_opt_name, a.llama_name, device, img_size=490)
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    proc.save_pretrained(a.output_dir)

    # freeze strategy
    if a.tuning_mode == "full": freeze_but_qformer(model)
    else:
        freeze_everything(model)
        model.qformer = get_peft_model(model.qformer, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"]))

    # datasets & collate
    train_ds, val_ds = VQADataset(train_raw), VQADataset(val_raw)
    collate_fn = make_collate(proc)

    # **layer-wise LR decay for ViT** stub — refine if needed
    vit_lw_decay = [0.95, 0.95, 0.9]  # applied outside TRL by custom optim if desired

    # trainer config
    cfg = SFTConfig(
        output_dir=a.output_dir,
        per_device_train_batch_size=a.batch_size,
        num_train_epochs=a.epochs,
        warmup_steps=a.warmup,
        learning_rate=a.lr,
        adam_beta1=0.9,
        adam_beta2=0.999,
        weight_decay=a.weight_decay,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_steps=1000,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=5000,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(model=model, args=cfg,
                         train_dataset=train_ds, eval_dataset=val_ds,
                         data_collator=collate_fn)

    ckpt = get_last_checkpoint(a.output_dir) if os.path.isdir(a.output_dir) else None
    trainer.train(resume_from_checkpoint=ckpt) if ckpt else trainer.train()

    trainer.save_model(a.output_dir)
    if local_rank == 0:
        print(f"\n✅  Fine-tuned Q-Former saved to {a.output_dir}\n")

if __name__ == "__main__":
    main()
