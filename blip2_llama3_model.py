import torch
from torch import nn
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    AutoModelForCausalLM,
    AutoTokenizer,
    QFormerModel,
)
from typing import Any, Dict


class Blip2Llama3Model(nn.Module):
    """BLIP-2 model with LLaMA v3.1 language model using Q-Former."""

    def __init__(self, blip2_model: str, llm_model: str) -> None:
        super().__init__()
        self.processor = Blip2Processor.from_pretrained(blip2_model)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)

        llm = AutoModelForCausalLM.from_pretrained(llm_model)
        qformer = QFormerModel.from_pretrained(blip2_model)

        # Initialize BLIP-2 with custom language model and Q-Former
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            blip2_model,
            decoder_model=llm,
            text_config=llm.config,
        )
        self.model.qformer = qformer

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        return self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
