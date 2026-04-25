from __future__ import annotations

from pathlib import Path

import torch

from src.rag_pipeline import DEFAULT_GENERATION_MODEL, DEFAULT_LORA_ADAPTER_PATH, load_flan_t5_generator


def generate_with_saved_lora(
    prompt: str,
    model_name: str = DEFAULT_GENERATION_MODEL,
    adapter_path: str | Path = DEFAULT_LORA_ADAPTER_PATH,
    max_new_tokens: int = 160,
) -> str:
    tokenizer, model = load_flan_t5_generator(
        model_name=model_name,
        lora_adapter_path=adapter_path,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            do_sample=False,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


if __name__ == "__main__":
    sample_prompt = (
        "You are ScholarSynth AI, an academic research assistant. "
        "Write a concise answer grounded only in the given input.\n\n"
        "Task: technical_explanation\n"
        "Topic: retrieval augmented generation\n"
        "Instruction: Explain the main idea in simple terms.\n"
        "Input:\n"
        "Retrieval augmented generation combines search over external documents with text generation.\n\n"
        "Answer:"
    )
    print(generate_with_saved_lora(sample_prompt))
