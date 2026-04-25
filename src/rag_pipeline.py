from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class RetrievedContext:
    documents: list[str]
    metadatas: list[dict[str, Any]]


def format_retrieved_context(results: dict) -> RetrievedContext:
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    return RetrievedContext(documents=documents, metadatas=metadatas)


def build_review_prompt(topic: str, context: RetrievedContext, mode: str = "literature_review") -> str:
    joined_context = "\n".join(
        f"- {meta.get('title', 'Untitled')}: {doc}"
        for doc, meta in zip(context.documents, context.metadatas)
    )
    instructions = {
        "literature_review": "Write a concise literature review with trends, methods, findings, and gaps.",
        "qa": "Answer the user question using only the provided context. Mention uncertainty if evidence is weak.",
        "gap_analysis": "Identify research gaps and open problems based on the provided papers.",
        "technical_explanation": "Explain the technical concept in simpler terms using the provided papers.",
    }
    return (
        f"Topic: {topic}\n"
        f"Task: {instructions.get(mode, instructions['literature_review'])}\n"
        f"Context:\n{joined_context}\n\n"
        "Cite paper titles inline where relevant."
    )


DEFAULT_GENERATION_MODEL = "google/flan-t5-base"
DEFAULT_LORA_ADAPTER_PATH = "models/flan_t5_lora"


def has_lora_adapter(adapter_path: str | Path = DEFAULT_LORA_ADAPTER_PATH) -> bool:
    adapter_path = Path(adapter_path)
    has_config = (adapter_path / "adapter_config.json").exists()
    has_weights = (adapter_path / "adapter_model.safetensors").exists() or (adapter_path / "adapter_model.bin").exists()
    return has_config and has_weights


def load_flan_t5_generator(
    model_name: str = DEFAULT_GENERATION_MODEL,
    lora_adapter_path: str | Path | None = DEFAULT_LORA_ADAPTER_PATH,
) -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    adapter_path = Path(lora_adapter_path) if lora_adapter_path else None
    tokenizer_source = adapter_path if adapter_path and adapter_path.exists() else model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if adapter_path and has_lora_adapter(adapter_path):
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

    model.eval()
    return tokenizer, model


class RagGenerator:
    def __init__(
        self,
        model_name: str = DEFAULT_GENERATION_MODEL,
        lora_adapter_path: str | Path | None = DEFAULT_LORA_ADAPTER_PATH,
    ) -> None:
        self.tokenizer, self.model = load_flan_t5_generator(
            model_name=model_name,
            lora_adapter_path=lora_adapter_path,
        )
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
