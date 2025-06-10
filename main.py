#!/usr/bin/env python
"""
train_blip2_llama_vqa.py
Fine-tunes a custom BLIP-2 model (vision encoder + Q-Former from Salesforce BLIP-2,
text model swapped to LLaMA-3.1) on the VQA-v2 dataset using Hugging Face TRL’s
SFTTrainer.

❱❱  Required arguments
    --llama_name       Local path or HF hub name of the LLaMA-3.1 checkpoint.
❱❱  Optional arguments
    --blip2_opt_name   Source BLIP-2 checkpoint for vision & Q-Former weights
                       (default: "Salesforce/blip2-opt-2.7b").
    --output_dir       Directory to save checkpoints (default shown below).
    --epochs           Number of fine-tuning epochs (default: 5).
    --batch_size       Per-device batch size (default: 4).
    --lr               Learning rate (default: 2e-5).
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Blip2Config,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    TrainingArguments,
)
from trl import SFTTrainer


# ────────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────────
def load_blip2_llama(blip2_opt_name: str, llama_name: str, device: torch.device):
    """Load BLIP-2 vision+Q-Former weights, replace text model with LLaMA-3.1."""
    # 1) Load source BLIP-2 (OPT) to grab vision encoder + Q-Former
    blip2_opt = Blip2ForConditionalGeneration.from_pretrained(blip2_opt_name)

    # 2) Load LLaMA-3.1 (causal LM) & tokenizer
    llama_model = AutoModelForCausalLM.from_pretrained(llama_name)
    llama_tok   = AutoTokenizer.from_pretrained(llama_name, use_fast=True)

    # 3) Build a new BLIP-2 config whose text_config is LLaMA’s
    new_cfg            = Blip2Config.from_dict(blip2_opt.config.to_dict())
    new_cfg.text_config = llama_model.config  # swap text config

    # 4) Instantiate fresh BLIP-2 model with that config
    blip2_llama = Blip2ForConditionalGeneration(new_cfg)

    # 5) Copy weights: vision encoder & Q-Former from OPT model, LLM from LLaMA
    blip2_llama.vision_model.load_state_dict(blip2_opt.vision_model.state_dict())
    blip2_llama.qformer.load_state_dict(blip2_opt.qformer.state_dict())
    blip2_llama.language_model.load_state_dict(llama_model.state_dict())

    # 6) Create processor; swap tokenizer to LLaMA’s
    processor               = Blip2Processor.from_pretrained(blip2_opt_name)
    processor.tokenizer     = llama_tok

    blip2_llama.to(device)
    return blip2_llama, processor


def vqa_collate_fn_factory(processor):
    """Return a data-collator that handles image + question → answer training."""
    def collate(batch):
        # Split fields
        images     = [ex["image"].convert("RGB") for ex in batch]
        questions  = [ex["question"].strip() for ex in batch]
        answers    = [ex["multiple_choice_answer"].strip() for ex in batch]

        # Build prompt “Question: …? Answer:” and track its token length
        prompts, prompt_lens = [], []
        for q in questions:
            if not q.endswith("?"):
                q += "?"
            prompt = f"Question: {q} Answer:"
            prompts.append(prompt)
            prompt_lens.append(
                len(processor.tokenizer(prompt, add_special_tokens=False).input_ids)
            )

        # Full text = prompt + ground-truth answer
        full_texts = [f"{p} {a}" for p, a in zip(prompts, answers)]

        # Image + text to tensors
        enc = processor(
            images=images,
            text=full_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Labels: mask prompt & padding tokens with −100
        labels = enc["input_ids"].clone()
        pad_id = processor.tokenizer.pad_token_id
        for i, l in enumerate(prompt_lens):
            labels[i, :l] = -100
        labels[labels == pad_id] = -100
        enc["labels"] = labels

        return {k: v.cuda() for k, v in enc.items()}

    return collate


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_name", required=True, help="HF path or local dir of LLaMA-3.1")
    parser.add_argument("--blip2_opt_name", default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--output_dir", default="./blip2-llama-vqa-checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Model & processor
    model, processor = load_blip2_llama(args.blip2_opt_name, args.llama_name, device)

    # 2) Datasets
    train_ds = load_dataset("HuggingFaceM4/VQAv2", split="train")
    val_ds   = load_dataset("HuggingFaceM4/VQAv2", split="validation")

    # 3) Collator
    collate_fn = vqa_collate_fn_factory(processor)

    # 4) Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_steps=1000,
        save_total_limit=2,
        evaluation_strategy="epoch",
        remove_unused_columns=False,             # custom collator
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # 5) SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,    # lets TRL save tokenizer on push
    )

    # 6) Fine-tune
    trainer.train()

    # 7) Save final model + tokenizer
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    processor.tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅  Fine-tuned model saved to {args.output_dir}\n")


if __name__ == "__main__":
    main()
