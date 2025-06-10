import argparse
from typing import Any, Dict

from datasets import load_dataset

from transformers import TrainingArguments, Trainer

from blip2_llama3_model import Blip2Llama3Model


def parse_args():
    parser = argparse.ArgumentParser(description="Train BLIP-2 with LLaMA v3.1 on VQA dataset")
    parser.add_argument("--blip2_model", default="Salesforce/blip2-flan-t5-xl", help="Base BLIP-2 vision model")
    parser.add_argument("--llm_model", default="meta-llama/Llama-3-8B-Instruct", help="LLaMA v3.1 language model")
    parser.add_argument("--dataset", default="HuggingFaceM4/VQAv2", help="VQA dataset to use")
    parser.add_argument("--output_dir", default="./blip2-llama3", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    model = Blip2Llama3Model(args.blip2_model, args.llm_model)

    # Load VQA dataset
    dataset = load_dataset(args.dataset, split="train")

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        proc = model.processor(images=example["image"], text=example["question"], return_tensors="pt")
        labels = model.tokenizer(example["answer"], return_tensors="pt").input_ids[0]
        return {
            "pixel_values": proc["pixel_values"][0],
            "input_ids": proc["input_ids"][0],
            "attention_mask": proc["attention_mask"][0],
            "labels": labels,
        }

    processed_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=model.tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
