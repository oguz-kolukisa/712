#!/usr/bin/env python
"""
evaluate_vqa.py
───────────────
• VQA-v2 accuracy for a fine-tuned BLIP-2 + LLaMA model.
• Progress bar via tqdm.
• Every --log_every examples prints question, ground-truth answers list,
  and model prediction.
"""

# ───────────────────────── imports ──────────────────────────────────────────
import argparse, re, torch, aiohttp
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import Blip2ForConditionalGeneration, Blip2Processor, AutoTokenizer

# ───────────────── accuracy helpers ─────────────────────────────────────────
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT    = re.compile(r"[^\w\s]", re.UNICODE)
def _norm(t): return re.sub(r"\s+", " ", _ARTICLES.sub(" ", _PUNCT.sub(" ", t.lower()))).strip()
def vqa_acc(pred, gts): return min(sum(_norm(pred) == _norm(a) for a in gts) / 3.0, 1.0)

# ───────────────── dataset ──────────────────────────────────────────────────
class VQAEvalDS(Dataset):
    def __init__(self, split):
        self.data = split
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        ex = self.data[i]
        q  = ex["question"].strip()
        if not q.endswith("?"): q += "?"
        prompt = f"Question: {q} Answer:"
        gts = [d["answer"] for d in ex["answers"]] if isinstance(ex["answers"][0], dict) else ex["answers"]
        return {"image":     ex["image"].convert("RGB"),
                "prompt":    prompt,
                "question":  q,
                "answers":   gts}

# ───────────────── arg-parse ────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--split", choices=["train", "validation"], default="validation")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=5)
    p.add_argument("--log_every", type=int, default=100,
                   help="Print Q / GT / prediction every N examples")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

# ───────────────── main ─────────────────────────────────────────────────────
@torch.no_grad()
def main():
    args = get_args()

    # 1) model + processor -----------------------------------------------------
    model = Blip2ForConditionalGeneration.from_pretrained(args.model_path, device_map="auto")
    try:
        processor = Blip2Processor.from_pretrained(args.model_path)
    except OSError:
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        processor.tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    pad_id = processor.tokenizer.pad_token_id

    if model.get_input_embeddings().weight.size(0) < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))

    img_tok_id       = model.config.image_token_id
    num_query_tokens = model.config.num_query_tokens

    # 2) data ------------------------------------------------------------------
    split = load_dataset("HuggingFaceM4/VQAv2", split=args.split, trust_remote_code=True,
                         storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}})
    ds = VQAEvalDS(split)

    def collate(batch):
        pix = processor.image_processor([b["image"] for b in batch], return_tensors="pt").pixel_values
        tok = processor.tokenizer([b["prompt"] for b in batch], add_special_tokens=False,
                                  padding=True, return_tensors="pt")
        input_ids, attn = tok.input_ids, tok.attention_mask

        B = input_ids.size(0)
        img_block  = torch.full((B, num_query_tokens), img_tok_id, dtype=input_ids.dtype)
        ones_block = torch.ones((B, num_query_tokens), dtype=attn.dtype)

        input_ids      = torch.cat([img_block,  input_ids], dim=1)
        attention_mask = torch.cat([ones_block, attn],      dim=1)
        prompt_len = attention_mask.sum(1)

        return {"pixel_values": pix,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "prompt_len": prompt_len,
                "questions": [b["question"] for b in batch],
                "answers":   [b["answers"]   for b in batch]}

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate, num_workers=4)

    # 3) loop with tqdm + periodic printing ------------------------------------
    model.eval(); dev = torch.device(args.device)
    total_acc = n = 0

    for batch in tqdm(dl, desc="Evaluating", unit="batch"):
        batch = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        gen = model.generate(pixel_values=batch["pixel_values"],
                             input_ids=batch["input_ids"],
                             attention_mask=batch["attention_mask"],
                             max_new_tokens=args.max_new_tokens)
        dec = processor.tokenizer.batch_decode(gen, skip_special_tokens=True)

        for i, full in enumerate(dec):
            prompt_txt = processor.tokenizer.decode(
                gen[i][: batch["prompt_len"][i]], skip_special_tokens=True)
            pred = full[len(prompt_txt):].strip()
            total_acc += vqa_acc(pred, batch["answers"][i]); n += 1

            # periodic console log --------------------------------------------
            if n % args.log_every == 0:
                print(f"\nQ: {batch['questions'][i]}\n"
                      f"GT: {batch['answers'][i]}\n"
                      f"PRED: {pred}\n")

    print(f"\n✅  {args.split} accuracy: {total_acc / n:.4%} ({n} questions)\n")

if __name__ == "__main__":
    main()
