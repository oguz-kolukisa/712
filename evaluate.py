#!/usr/bin/env python
"""
evaluate_vqa.py
~~~~~~~~~~~~~~~
Compute VQA-v2 accuracy for a fine-tuned BLIP-2 + LLaMA model whose
checkpoints were saved with `trainer.save_model(...)`.

Example
-------
python evaluate_vqa.py \
    --model_path ./blip2-llama-vqa-checkpoints-qformer \
    --split validation
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)


# ───────────────────────────────────────────────────────────────────────────────
# Normalisation utilities (same as official VQA eval)
# ───────────────────────────────────────────────────────────────────────────────
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT    = re.compile(r"[^\w\s]", re.UNICODE)


def normalise_answer(ans: str) -> str:
    """Lower-case, strip punctuation/articles/extra spaces."""
    ans = ans.lower()
    ans = _PUNCT.sub(" ", ans)           # remove punctuation
    ans = _ARTICLES.sub(" ", ans)        # remove articles
    ans = re.sub(r"\s+", " ", ans)       # collapse whitespace
    return ans.strip()


def vqa_accuracy(pred: str, gt_answers) -> float:
    """
    `gt_answers` is the list of 10 ground-truth answers.
    Accuracy = min(matches / 3, 1)  (matches ∈ {0..10})
    """
    pred_norm = normalise_answer(pred)
    matches = sum(pred_norm == normalise_answer(a) for a in gt_answers)
    return min(matches / 3.0, 1.0)


# ───────────────────────────────────────────────────────────────────────────────
# Dataset & collate for *inference* (prompt only, no answers)
# ───────────────────────────────────────────────────────────────────────────────
class VQAEvalDataset(Dataset):
    """
    Each item provides:
      • pixel_values (vision encoding)
      • input_ids / attention_mask for the prompt "Question: … Answer:"
      • answers  (list of 10 ground-truth answers for scoring)
    """

    def __init__(self, hf_split, processor):
        self.data      = hf_split
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        # Prompt without the answer
        question = ex["question"].strip()
        if not question.endswith("?"):
            question += "?"
        prompt = f"Question: {question} Answer:"

        enc = self.processor(
            images=ex["image"].convert("RGB"),
            text=prompt,
            return_tensors="pt",
            padding="max_length",
        )
        enc = {k: v.squeeze() for k, v in enc.items()}  # drop batch dim

        # Ground-truth answers list (10 strings)
        gt_answers = (
            [d["answer"] for d in ex["answers"]]
            if isinstance(ex["answers"][0], dict)
            else ex["answers"]
        )
        enc["answers"] = gt_answers
        return enc


def collate_fn(batch):
    pixel_values   = torch.stack([b["pixel_values"] for b in batch])
    input_ids      = torch.stack([b["input_ids"]      for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    answers_list   = [b["answers"] for b in batch]
    return {
        "pixel_values":   pixel_values,
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "answers":        answers_list,
    }


# ───────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ───────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Path to fine-tuned checkpoint directory")
    p.add_argument("--split", default="validation",
                   choices=["train", "validation"],
                   help="VQA-v2 split to evaluate")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    # 1) Load model & processor (weights saved by SFTTrainer)
    print(f"🔄  Loading model from {args.model_path}")
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_path, device_map="auto"
    )
    processor = Blip2Processor.from_pretrained(args.model_path)

    # Ensure pad token is set (was done in training, but safeguard)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 2) Dataset / loader
    split = load_dataset(
        "HuggingFaceM4/VQAv2",
        split=args.split,
        trust_remote_code=True,
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}},
    )
    eval_ds  = VQAEvalDataset(split, processor)
    loader   = DataLoader(eval_ds,
                          batch_size=args.batch_size,
                          shuffle=False,
                          collate_fn=collate_fn,
                          num_workers=4)

    model.eval()
    device = torch.device(args.device)

    total_acc = 0.0
    n_samples = 0

    print(f"🚀  Evaluating on {args.split} split…")
    for step, batch in enumerate(loader):
        pixel_values   = batch["pixel_values"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        gen_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
        )

        decoded = processor.tokenizer.batch_decode(
            gen_ids, skip_special_tokens=True
        )

        # Strip the original prompt from each decoded string
        for pred_str, prompt_ids, gt in zip(
            decoded, input_ids, batch["answers"]
        ):
            prompt_len = (prompt_ids != processor.tokenizer.pad_token_id).sum().item()
            prompt_text = processor.tokenizer.decode(
                prompt_ids[:prompt_len], skip_special_tokens=True
            )
            # The generated answer is whatever comes *after* the prompt
            answer_pred = pred_str[len(prompt_text):].strip()
            acc = vqa_accuracy(answer_pred, gt)
            total_acc += acc
            n_samples += 1

        if (step + 1) % 100 == 0:
            print(f"Step {step+1:>4}: running accuracy = {total_acc/n_samples:.4f}")

    overall = total_acc / n_samples
    print(f"\n✅  {args.split} accuracy: {overall:.4%}  "
          f"({n_samples} questions)\n")


if __name__ == "__main__":
    main()
