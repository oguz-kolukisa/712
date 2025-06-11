#!/usr/bin/env python
"""
evaluate_blip2_llama_vqa_qformer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Validation-set evaluation for a BLIP-2 + LLaMA-3.1 model whose Q-Former
was fine-tuned on VQA-v2.

Adds:
* tqdm progress bar
* explicit pad_token_id to silence HF warning
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Blip2ForConditionalGeneration, Blip2Processor
from tqdm.auto import tqdm   # ← progress bar
import random
# -----------------------------------------------------------------------------

def strip_prompt_to_question(prompt: str) -> str:
    """Remove the 'Question:' / 'Answer:' boiler-plate."""
    q = prompt
    if q.startswith("Question:"):
        q = q[len("Question:"):].lstrip()
    if q.endswith(" Answer:"):
        q = q[:-len(" Answer:")]
    return q.strip()

def build_processor(blip2_opt_name: str, ckpt_dir: Path) -> Blip2Processor:
    """Re-create processor; use saved tokenizer and set pad token."""
    proc = Blip2Processor.from_pretrained(blip2_opt_name)
    tok = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token        # add pad if missing
    proc.tokenizer = tok
    return proc


def collate_fn_factory(proc: Blip2Processor):
    def collate(batch):
        imgs = [ex["image"].convert("RGB") for ex in batch]
        qs   = [ex["question"].strip() for ex in batch]
        prompts = [f"Question: {q.rstrip('?')}? Answer:" for q in qs]

        enc = proc(images=imgs, text=prompts,
                   padding=True, truncation=True, return_tensors="pt")
        return enc, prompts
    return collate


def extract_answers(out_ids: torch.Tensor, prompts: List[str], tok) -> List[str]:
    txts = tok.batch_decode(out_ids, skip_special_tokens=True)
    preds = [(t[len(p):].strip()).lower().rstrip(" .") for t, p in zip(txts, prompts)]
    return preds


# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="Fine-tuned checkpoint")
    p.add_argument("--blip2_opt_name", default="Salesforce/blip2-opt-2.7b")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=5)
    p.add_argument("--device", default=None)
    p.add_argument("--save_results", default=None,
                   help="Dump predictions to JSONL")
    p.add_argument("--log_every", type=int, default=1000,
                   help="Print an example Q/A every N validation items")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    dev = torch.device(args.device) if args.device else \
          torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) model
    print("🔄 loading model…")
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None
    ).to(dev).eval()

    # 2) processor  (+ pad_token_id fix)
    proc = build_processor(args.blip2_opt_name, Path(args.model_dir))
    pad_id = proc.tokenizer.pad_token_id
    model.config.pad_token_id = pad_id
    model.generation_config.pad_token_id = pad_id   # stop warning

    collate_fn = collate_fn_factory(proc)

    # 3) data
    val_ds = load_dataset("HuggingFaceM4/VQAv2",
                           split="validation", trust_remote_code=True)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size,
                                         shuffle=False, collate_fn=collate_fn)

    # 4) loop with tqdm
    correct = total = 0
    offset  = 0
    dumped  = []
    pbar = tqdm(loader, desc="Evaluating", unit="batch")
    for batch, prompts in pbar:
        batch = {k: v.to(dev) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        out = model.generate(**batch,
                             max_new_tokens=args.max_new_tokens,
                             do_sample=False,
                             pad_token_id=pad_id)        # explicit

        preds = extract_answers(out, prompts, proc.tokenizer)
        gts   = [val_ds[i]["multiple_choice_answer"].lower()
                 for i in range(total, total + len(preds))]

        # ── accuracy bookkeeping ────────────────────────────────────────────
        correct += sum(p == g for p, g in zip(preds, gts))
        total   += len(preds)
        pbar.set_postfix(acc=f"{correct/total:.3%}")

        # ── log results if requested ────────────────────────────────────────
        if args.save_results:
            dumped.extend({"prompt": pr, "question": strip_prompt_to_question(pr),
                           "pred": p, "gt": g}
                          for pr, p, g in zip(prompts, preds, gts))

        # ── EXAMPLE LOGGING every N items ───────────────────────────────────
        while total >= next_log:
            # pick a random sample *inside the current batch* to display
            idx = random.randrange(len(prompts))
            print("\n" + "-"*50)
            print(f"📌 Example at {next_log} processed items:")
            print(f"Q: {strip_prompt_to_question(prompts[idx])}")
            print(f"GT: {gts[idx]}")
            print(f"PR: {preds[idx]}")
            print("-"*50 + "\n")
            next_log += args.log_every

    # 5) final
    acc = correct / total
    print(f"\n✅  Validation exact-match accuracy: {acc:.3%} "
          f"({correct}/{total})")

    if args.save_results:
        with open(args.save_results, "w") as f:
            for obj in dumped:
                f.write(json.dumps(obj) + "\n")
        print(f"📝 results written to {args.save_results}")


if __name__ == "__main__":
    main()
