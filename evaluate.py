#!/usr/bin/env python
"""
evaluate_vqa.py
───────────────
Compute the standard VQA-v2 accuracy for a BLIP-2 + LLaMA model that was
fine-tuned with the matching training script.

Key fix ➜ the collate function prepends *exactly*
`model.config.num_query_tokens` copies of `model.config.image_token_id`
to each `input_ids`, so `BLIP-2.generate()` never hits a shape-mismatch.
"""

# ───────────────────────────── imports ──────────────────────────────────────
import argparse, re, torch, aiohttp
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    AutoTokenizer,
)

# ───────────────── accuracy helpers ─────────────────────────────────────────
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT    = re.compile(r"[^\w\s]", re.UNICODE)
def _norm(txt: str) -> str:
    txt = _PUNCT.sub(" ", txt.lower())
    txt = _ARTICLES.sub(" ", txt)
    return re.sub(r"\s+", " ", txt).strip()
def vqa_acc(pred: str, gts) -> float:
    return min(sum(_norm(pred) == _norm(a) for a in gts) / 3.0, 1.0)

# ───────────────── dataset ──────────────────────────────────────────────────
class VQAEvalDS(Dataset):
    def __init__(self, split):
        self.d = split
    def __len__(self): return len(self.d)
    def __getitem__(self, idx):
        ex = self.d[idx]
        q  = ex["question"].strip()
        if not q.endswith("?"): q += "?"
        prompt = f"Question: {q} Answer:"
        gts = [d["answer"] for d in ex["answers"]] if isinstance(ex["answers"][0], dict) else ex["answers"]
        return {"image": ex["image"].convert("RGB"),
                "prompt": prompt,
                "answers": gts}

# ───────────────── arg-parse ────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Folder with fine-tuned model + processor")
    p.add_argument("--split", choices=["train", "validation"],
                   default="validation")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=5)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

# ───────────────── main ─────────────────────────────────────────────────────
@torch.no_grad()
def main():
    args = get_args()

    # 1) model & processor -----------------------------------------------------
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_path, device_map="auto"
    )
    try:
        processor = Blip2Processor.from_pretrained(args.model_path)
    except OSError:   # fallback if only tokenizer was saved (old ckpt)
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        processor.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, use_fast=True
        )

    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    pad_id = processor.tokenizer.pad_token_id

    # Make sure embedding matrix is big enough (older checkpoints)
    if model.get_input_embeddings().weight.size(0) < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))

    img_tok_id       = model.config.image_token_id
    num_query_tokens = model.config.num_query_tokens  # usually 32

    # 2) data ------------------------------------------------------------------
    split = load_dataset(
        "HuggingFaceM4/VQAv2",
        split=args.split,
        trust_remote_code=True,
        storage_options={"client_kwargs":
            {"timeout": aiohttp.ClientTimeout(total=3600)}},
    )
    ds = VQAEvalDS(split)

    # ── collate function that prepends N image tokens ─────────────────────────
    def collate(batch):
        # vision
        pix = processor.image_processor(
            images=[b["image"] for b in batch],
            return_tensors="pt"
        ).pixel_values

        # tokenise prompt (no special tokens)
        tok = processor.tokenizer(
            [b["prompt"] for b in batch],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt"
        )
        input_ids, attn = tok.input_ids, tok.attention_mask

        # prepend <image> token ID block
        B = input_ids.size(0)
        img_block  = torch.full(
            (B, num_query_tokens), img_tok_id, dtype=input_ids.dtype
        )
        ones_block = torch.ones(
            (B, num_query_tokens), dtype=attn.dtype
        )

        input_ids      = torch.cat([img_block, input_ids], dim=1)
        attention_mask = torch.cat([ones_block, attn], dim=1)

        # store the prompt length (incl. image tokens) for later trimming
        prompt_lens = attention_mask.sum(1)

        return {"pixel_values": pix,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "prompt_lens": prompt_lens,
                "answers": [b["answers"] for b in batch]}

    dl = DataLoader(ds, batch_size=args.batch_size,
                    shuffle=False, collate_fn=collate, num_workers=4)

    # 3) evaluation loop -------------------------------------------------------
    model.eval(); device = torch.device(args.device)
    total_acc, n = 0.0, 0

    for step, batch in enumerate(dl):
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

        for i, full in enumerate(decoded):
            prompt_tok_len = int(batch["prompt_lens"][i])
            prompt_text = processor.tokenizer.decode(
                gen_ids[i][:prompt_tok_len], skip_special_tokens=True
            )
            pred = full[len(prompt_text):].strip()
            total_acc += vqa_acc(pred, batch["answers"][i])
            n += 1

        if (step + 1) % 100 == 0:
            print(f"Step {step+1:>4}: running acc = {total_acc / n:.4%}")

    print(f"\n✅  {args.split} accuracy: {total_acc / n:.4%} ({n} questions)\n")

if __name__ == "__main__":
    main()
