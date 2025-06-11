#!/usr/bin/env python
"""
evaluate_vqa.py
~~~~~~~~~~~~~~~
Compute VQA-v2 accuracy for a fine-tuned BLIP-2 + LLaMA model.

Usage
-----
python evaluate_vqa.py \
    --model_path ./blip2-llama-vqa-checkpoints-qformer \
    --split validation \
    --batch_size 32 \
    --max_new_tokens 5
"""

# ───────────────────────────────────────────────────────────────────────────────
# Imports
# ───────────────────────────────────────────────────────────────────────────────
import argparse
import re
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    AutoTokenizer,
)
import aiohttp

# ───────────────────────────────────────────────────────────────────────────────
# VQA accuracy helpers
# ───────────────────────────────────────────────────────────────────────────────
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT    = re.compile(r"[^\w\s]", re.UNICODE)


def normalise(ans: str) -> str:
    ans = ans.lower()
    ans = _PUNCT.sub(" ", ans)
    ans = _ARTICLES.sub(" ", ans)
    return re.sub(r"\s+", " ", ans).strip()


def vqa_acc(pred: str, gts) -> float:
    pred = normalise(pred)
    matches = sum(pred == normalise(a) for a in gts)
    return min(matches / 3.0, 1.0)


# ───────────────────────────────────────────────────────────────────────────────
# Dataset: prompt *with* one <image> token
# ───────────────────────────────────────────────────────────────────────────────
class VQAEvalDataset(Dataset):
    def __init__(self, hf_split, processor, image_tok):
        self.data      = hf_split
        self.processor = processor
        self.img_tok   = image_tok

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        q = ex["question"].strip()
        if not q.endswith("?"):
            q += "?"
        prompt = f"{self.img_tok} Question: {q} Answer:"

        gt_answers = (
            [d["answer"] for d in ex["answers"]]
            if isinstance(ex["answers"][0], dict)
            else ex["answers"]
        )

        return {
            "image":   ex["image"].convert("RGB"),
            "prompt":  prompt,
            "answers": gt_answers,
        }


# ───────────────────────────────────────────────────────────────────────────────
# Arg-parse
# ───────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Checkpoint directory with fine-tuned weights + processor")
    p.add_argument("--split", default="validation",
                   choices=["train", "validation"], help="VQA-v2 split to score")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def main():
    args = parse_args()

    # 1) Model & processor ------------------------------------------------------
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_path, device_map="auto"
    )

    try:
        processor = Blip2Processor.from_pretrained(args.model_path)
    except OSError:
        # folder only has tokenizer → pull vision pre-processor from base model
        print("⚠️  No vision pre-processor in checkpoint; "
              "falling back to Salesforce/blip2-opt-2.7b defaults.")
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        processor.tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    image_tok_id  = model.config.image_token_id
    image_tok_str = processor.tokenizer.decode([image_tok_id]).strip().lstrip()

    # 2) Dataset & DataLoader ---------------------------------------------------
    split = load_dataset(
        "HuggingFaceM4/VQAv2",
        split=args.split,
        trust_remote_code=True,
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}},
    )
    eval_ds = VQAEvalDataset(split, processor, image_tok_str)

    def make_collate_fn(processor):
        def collate(batch):
            enc = processor(
                images=[b["image"] for b in batch],
                text=[b["prompt"] for b in batch],
                padding=True,
                return_tensors="pt",
            )
            enc["answers"] = [b["answers"] for b in batch]
            return enc
        return collate

    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(processor),
        num_workers=4,
    )

    # 3) Evaluation loop --------------------------------------------------------
    model.eval()
    device = torch.device(args.device)

    total_acc, n = 0.0, 0
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        gen_ids = model.generate(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=args.max_new_tokens,
        )

        decoded = processor.tokenizer.batch_decode(
            gen_ids, skip_special_tokens=True
        )

        # length of prompt in tokens → trim to isolate generated answer
        prompt_lens = (batch["input_ids"] != processor.tokenizer.pad_token_id).sum(dim=1)

        for i, full_pred in enumerate(decoded):
            prompt_text = processor.tokenizer.decode(
                batch["input_ids"][i, :prompt_lens[i]], skip_special_tokens=True
            )
            pred_ans = full_pred[len(prompt_text):].strip()
            total_acc += vqa_acc(pred_ans, batch["answers"][i])
            n += 1

        if (step + 1) % 100 == 0:
            print(f"Step {step+1:>4}: running acc = {total_acc / n:.4%}")

    print(f"\n✅  {args.split} accuracy: {total_acc / n:.4%} ({n} questions)\n")


if __name__ == "__main__":
    main()
