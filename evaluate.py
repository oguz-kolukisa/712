#!/usr/bin/env python
"""
evaluate_vqa.py  —  NO more shape-mismatch errors
"""

import argparse, re, torch, aiohttp
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import Blip2ForConditionalGeneration, Blip2Processor, AutoTokenizer

# ─── Accuracy helpers ────────────────────────────────────────────────────────
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT    = re.compile(r"[^\w\s]", re.UNICODE)
def norm(t): return re.sub(r"\s+", " ", _ARTICLES.sub(" ", _PUNCT.sub(" ", t.lower()))).strip()
def vqa_acc(pred, gts): return min(sum(norm(pred) == norm(a) for a in gts) / 3.0, 1.0)

# ─── Dataset ─────────────────────────────────────────────────────────────────
class VQAEvalDS(Dataset):
    def __init__(self, split): self.data = split
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        ex = self.data[i]
        q  = ex["question"].strip()
        if not q.endswith("?"): q += "?"
        prompt = f"Question: {q} Answer:"
        gts = [d["answer"] for d in ex["answers"]] if isinstance(ex["answers"][0], dict) else ex["answers"]
        return {"image": ex["image"].convert("RGB"), "prompt": prompt, "answers": gts}

# ─── Arg-parse ───────────────────────────────────────────────────────────────
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--split", choices=["train", "validation"], default="validation")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

# ─── Main ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def main():
    a = parse()

    # 1) model & processor ------------------------------------------------------
    model = Blip2ForConditionalGeneration.from_pretrained(a.model_path, device_map="auto")
    try:
        processor = Blip2Processor.from_pretrained(a.model_path)
    except OSError:                               # (rare) tokenizer-only dir
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        processor.tokenizer = AutoTokenizer.from_pretrained(a.model_path, use_fast=True)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # enlarge embeddings if checkpoint predates fix
    if model.get_input_embeddings().weight.size(0) < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))

    # 2) data -------------------------------------------------------------------
    split = load_dataset("HuggingFaceM4/VQAv2", split=a.split, trust_remote_code=True,
                         storage_options={"client_kwargs":
                             {"timeout": aiohttp.ClientTimeout(total=3600)}})
    ds = VQAEvalDS(split)

    def collate(batch):
        # this call injects *num_query_tokens* image tokens automatically
        enc = processor(
            images=[b["image"] for b in batch],
            text=[b["prompt"] for b in batch],
            padding=True,
            return_tensors="pt",
        )
        enc["answers"] = [b["answers"] for b in batch]
        return enc

    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                    collate_fn=collate, num_workers=4)

    # 3) loop -------------------------------------------------------------------
    model.eval(); dev = torch.device(a.device)
    total, n = 0.0, 0
    pad_id = processor.tokenizer.pad_token_id

    for step, batch in enumerate(dl):
        batch = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        gen = model.generate(pixel_values=batch["pixel_values"],
                             input_ids=batch["input_ids"],
                             attention_mask=batch["attention_mask"],
                             max_new_tokens=a.max_new_tokens)
        dec = processor.tokenizer.batch_decode(gen, skip_special_tokens=True)
        prompt_lens = (batch["input_ids"] != pad_id).sum(1)

        for i, full in enumerate(dec):
            prompt_txt = processor.tokenizer.decode(
                gen[i][:prompt_lens[i]], skip_special_tokens=True)
            pred = full[len(prompt_txt):].strip()
            total += vqa_acc(pred, batch["answers"][i]); n += 1

        if (step + 1) % 100 == 0:
            print(f"Step {step+1:>4}: running acc = {total / n:.4%}")

    print(f"\n✅  {a.split} accuracy: {total / n:.4%} ({n} questions)\n")

if __name__ == "__main__":
    main()
