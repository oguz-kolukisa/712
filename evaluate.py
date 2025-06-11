#!/usr/bin/env python
"""
evaluate_blip2_llama_vqa_qformer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluates a BLIP-2 + LLaMA-3.1 model whose **Q-Former** was fine-tuned on
VQA-v2 with the companion training script.

Usage
-----
python evaluate_blip2_llama_vqa_qformer.py \
    --model_dir  ./blip2-llama-vqa-checkpoints-qformer \
    --blip2_opt_name  Salesforce/blip2-opt-2.7b
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    Blip2Processor,
)

# ───────────────────────────────────────────────────────────────────────────────
# Utilities
# ───────────────────────────────────────────────────────────────────────────────


def build_processor(blip2_opt_name: str, ckpt_dir: Path) -> Blip2Processor:
    """
    Re-create the BLIP-2 processor using the *image* sub-processors from the
    original BLIP-2 checkpoint and the fine-tuned tokenizer from `ckpt_dir`.
    """
    proc = Blip2Processor.from_pretrained(blip2_opt_name)

    # Load the tokenizer *saved* by the training script
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    proc.tokenizer = tokenizer
    return proc


def collate_fn_factory(proc: Blip2Processor):
    """Make the same collator used for training (no answers in labels)."""

    def collate(batch):
        images = [ex["image"].convert("RGB") for ex in batch]
        questions = [ex["question"].strip() for ex in batch]

        prompts = []
        for q in questions:
            if not q.endswith("?"):
                q += "?"
            prompts.append(f"Question: {q} Answer:")

        enc = proc(
            images=images,
            text=prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return enc, prompts  # keep prompts for answer stripping

    return collate


def extract_answers(
    generated: torch.Tensor, prompts: List[str], tokenizer
) -> List[str]:
    """
    Decode generated sequences and cut off the prompt part.
    """
    txts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    preds = []
    for t, p in zip(txts, prompts):
        # Remove prompt (it should be the prefix)
        answer = t[len(p) :].strip()
        # VQA answers are lowercase + no trailing punctuation by convention
        preds.append(answer.lower().rstrip(" ."))
    return preds


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True,
                        help="Path to the fine-tuned checkpoint directory")
    parser.add_argument("--blip2_opt_name", default="Salesforce/blip2-opt-2.7b",
                        help="Original BLIP-2 OPT checkpoint used for training")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=5,
                        help="Max tokens to generate for the answer")
    parser.add_argument("--device", default=None,
                        help="Force device id, e.g. 'cuda:1' or 'cpu'")
    parser.add_argument("--save_results", default=None,
                        help="Optional JSONL file to dump per-sample results")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # 1) Model & processor ------------------------------------------------------
    print("🔄  Loading model…")
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else None
    ).to(device)
    model.eval()

    processor = build_processor(args.blip2_opt_name, Path(args.model_dir))
    collate_fn = collate_fn_factory(processor)

    # 2) Dataset ---------------------------------------------------------------
    val_ds = load_dataset(
        "HuggingFaceM4/VQAv2",
        split="validation",
        trust_remote_code=True,
    )

    dataloader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # 3) Inference loop --------------------------------------------------------
    num_correct = 0
    total = 0
    dumped = []  # optional per-sample logging

    for batch, prompts in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Generate answers
        out_ids = model.generate(
            **batch,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        preds = extract_answers(out_ids, prompts, processor.tokenizer)

        # Compare to ground-truth (exact multiple-choice answer)
        gt = [ex["multiple_choice_answer"].lower() for ex in
              val_ds.select(range(total, total + len(preds)))]  # fast slice

        for p, g in zip(preds, gt):
            if p == g:
                num_correct += 1

        if args.save_results:
            for p, g, pr in zip(prompts, gt, preds):
                dumped.append({"prompt": p, "pred": pr, "gt": g})

        total += len(preds)

        if total % 2048 == 0:
            print(f"…{total:6d}/{len(val_ds)} processed – "
                  f"running acc: {num_correct/total:.3%}")

    # 4) Final report ----------------------------------------------------------
    acc = num_correct / total
    print("\n" + "=" * 60)
    print(f"✅  Validation exact-match accuracy: {acc:.3%} "
          f"({num_correct}/{total})")
    print("=" * 60)

    # 5) Optional per-sample dump ---------------------------------------------
    if args.save_results:
        with open(args.save_results, "w") as fout:
            for obj in dumped:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"📝  Detailed results written to {args.save_results}")


if __name__ == "__main__":
    main()
