from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
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


class RagGenerator:
    def __init__(self, model_name: str = "google/flan-t5-base") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
