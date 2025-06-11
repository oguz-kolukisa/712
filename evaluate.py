#!/usr/bin/env python
"""
eval_blip2_llama_vqa_qformer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluates a BLIP-2 + LLaMA-3.1 checkpoint (Q-Former fine-tuned) on **VQA-v2**
and prints three accuracy scores:

    • Overall VQA accuracy
    • Yes/No accuracy
    • Number accuracy
      (Other is shown as an extra line for reference)

Run on a single GPU:

    python eval_blip2_llama_vqa_qformer.py \
        --ckpt_dir ./blip2-llama-vqa-checkpoints-qformer \
        --split validation

────────────────────────────────────────────────────────────────────────────────
Implementation notes
────────────────────────────────────────────────────────────────────────────────
* Uses the canonical VQA consensus metric:
      acc_i = min(# matching answers in GT, 3) / 3
  with the ten annotator answers provided by the dataset.
* Categorises questions with the dataset’s built-in **answer_type**
  field (“yes/no”, “number”, “other”).
* Prompts are identical to training:
      "Question: <question>? Answer:"
* Generates answers greedily with `max_new_tokens=5`.
"""

import argparse
import os
import json
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)

# ───────────────────────────────────────────────────────────────────────────────
# Collator (same as in training, but no ground-truth answer attached)
# ───────────────────────────────────────────────────────────────────────────────

def vqa_infer_collate_fn_factory(processor, device):
    def collate(batch):
        images    = [ex["image"].convert("RGB") for ex in batch]
        questions = [ex["question"].strip() for ex in batch]

        # Ensure every question ends with '?'
        prompts = [f"Question: {q if q.endswith('?') else q + '?'} Answer:" for q in questions]

        enc = processor(
            images=images,
            text=prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # Pass through extra fields needed for metric
        enc["answer_list"]  = [ex["answers"] for ex in batch]          # list of 10×str
        enc["answer_type"]  = [ex["answer_type"] for ex in batch]      # "yes/no", "number", "other"
        return enc
    return collate

# ───────────────────────────────────────────────────────────────────────────────
# VQA consensus metric
# ───────────────────────────────────────────────────────────────────────────────

def vqa_accuracy(pred: str, gts):
    """
    pred: str (lower-cased, stripped)
    gts : list[str] (10 annotator answers, lower-cased, stripped)
    """
    pred = pred.strip().lower()
    gts  = [a.strip().lower() for a in gts]
    matches = sum([pred == a for a in gts])
    return min(matches, 3) / 3.0    # ∈ {0, 0.33, 0.67, 1.0}

# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, help="Path to fine-tuned checkpoint dir")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--processor_name_or_path", default="Salesforce/blip2-flan-t5-xl", help="HF repo or local dir that holds the Blip2Processor")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Dataset ────────────────────────────────────────────────────────────────
    ds = load_dataset(
        "HuggingFaceM4/VQAv2",
        split=args.split,
        trust_remote_code=True,
        storage_options={"client_kwargs": {"timeout": 3600}},
    )

    # 2) Model + processor ──────────────────────────────────────────────────────
    model      = Blip2ForConditionalGeneration.from_pretrained(args.ckpt_dir).to(device).eval()
    processor = Blip2Processor.from_pretrained(args.processor_name_or_path)    
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 3) DataLoader ─────────────────────────────────────────────────────────────
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=os.cpu_count() // 2,
        collate_fn=vqa_infer_collate_fn_factory(processor, device),
    )

    # 4) Metric accumulation ───────────────────────────────────────────────────
    running = defaultdict(list)   # {"overall": [...], "yes/no": [...], ...}

    print(f"🚀 Evaluating {args.ckpt_dir} on VQA-v2 {args.split} ({len(ds):,} samples)…")
    for batch in dl:
        # a) Generation
        gen_ids = model.generate(
            **{k: v for k, v in batch.items() if k in ["input_ids", "attention_mask", "pixel_values"]},
            max_new_tokens=args.max_new_tokens,
        )
        preds = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        preds = [p.split("Answer:")[-1].strip().lower() for p in preds]  # keep text after prompt

        # b) Metric per sample
        for pred, gts, atype in zip(preds, batch["answer_list"], batch["answer_type"]):
            acc = vqa_accuracy(pred, [a["answer"] for a in gts])
            running["overall"].append(acc)
            running[atype.lower()].append(acc)

    # 5) Aggregate and print ────────────────────────────────────────────────────
    def mean(x): return sum(x) / len(x) if x else 0.0

    overall = mean(running["overall"]) * 100
    yesno   = mean(running["yes/no"]) * 100
    number  = mean(running["number"]) * 100
    other   = mean(running["other"]) * 100

    print("\n────────── VQA-v2 Accuracy ──────────")
    print(f"Overall : {overall:5.2f}%")
    print(f"Yes/No  : {yesno:5.2f}%")
    print(f"Number  : {number:5.2f}%")
    print(f"Other   : {other:5.2f}%")
    print("──────────────────────────────────────")

if __name__ == "__main__":
    main()
