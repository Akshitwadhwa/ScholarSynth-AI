from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from bert_score import score as bertscore

from src.evaluation import compute_bleu, compute_rouge, save_metrics
from src.rag_pipeline import RagGenerator, format_retrieved_context
from src.vector_store import ChromaVectorStore, load_embedding_model


MODE_BY_TASK = {
    "literature_review": "literature_review",
    "research_gap_analysis": "gap_analysis",
    "technical_explanation": "technical_explanation",
    "evidence_based_qa": "qa",
    "paper_summary": "technical_explanation",
    "comparative_analysis": "literature_review",
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_balanced_examples(records: list[dict[str, Any]], sample_size: int = 40) -> list[dict[str, Any]]:
    df = pd.DataFrame(records).reset_index(drop=True)
    if len(df) <= sample_size:
        return df.to_dict(orient="records")

    task_order = [
        "paper_summary",
        "technical_explanation",
        "evidence_based_qa",
        "literature_review",
        "research_gap_analysis",
        "comparative_analysis",
    ]
    selected_indices: list[int] = []
    per_task = max(sample_size // len(task_order), 1)

    for task in task_order:
        task_df = df[df["task"] == task]
        selected_indices.extend(task_df.head(per_task).index.tolist())

    if len(selected_indices) < sample_size:
        remaining = df[~df.index.isin(selected_indices)]
        selected_indices.extend(remaining.head(sample_size - len(selected_indices)).index.tolist())

    return df.loc[selected_indices[:sample_size]].to_dict(orient="records")


def build_plain_prompt(example: dict[str, Any]) -> str:
    return f"{example['instruction']}\n\n{example['input']}\n\nAnswer:"


def build_prompt_engineered_prompt(example: dict[str, Any]) -> str:
    task = example["task"]
    style_rules = {
        "paper_summary": "Start with the paper title, then summarize the abstract in 2-3 academic sentences.",
        "technical_explanation": "Start with 'In simple terms,' and explain the main idea for an early-stage researcher.",
        "evidence_based_qa": "Start with 'Based on the abstract,' and answer only the stated question.",
        "literature_review": "Start with 'Research on this topic shows' and synthesize the provided papers.",
        "research_gap_analysis": "Start with 'The retrieved papers suggest several research gaps' and list concise gaps.",
        "comparative_analysis": "Explain how the provided papers are related and where they differ.",
    }
    return (
        "You are an academic research assistant. Write a concise answer grounded only in the given input. "
        "Do not invent paper titles, metrics, citations, datasets, or results. Avoid repetition.\n\n"
        f"Task type: {example['task']}\n"
        f"Topic: {example['topic']}\n"
        f"Instruction: {example['instruction']}\n"
        f"Expected answer style: {style_rules.get(task, 'Write a concise academic answer.')}\n"
        f"Input:\n{example['input']}\n\n"
        "Answer in 2-5 clear sentences:"
    )


def build_rag_prompt(
    example: dict[str, Any],
    vector_store: ChromaVectorStore,
    embedding_model: Any,
    top_k: int = 4,
) -> tuple[str, dict]:
    query = f"{example['topic']} {example['instruction']} {example['input'][:300]}"
    retrieval_results = vector_store.semantic_search(query, embedding_model, top_k=top_k)
    context = format_retrieved_context(retrieval_results)
    evidence = "\n".join(
        f"[{idx}] {meta.get('title', 'Untitled')}: {doc}"
        for idx, (doc, meta) in enumerate(zip(context.documents, context.metadatas), start=1)
    )
    task = example["task"]
    style_rules = {
        "paper_summary": "Summarize the user paper first; use retrieved evidence only as supporting context.",
        "technical_explanation": "Explain the user paper or concept in simple terms; use retrieved evidence only to clarify.",
        "evidence_based_qa": "Answer the user question from the provided abstract first; use retrieved evidence only if it supports the same claim.",
        "literature_review": "Synthesize the retrieved papers into methods, findings, trends, and gaps.",
        "research_gap_analysis": "Identify research gaps supported by the retrieved papers.",
        "comparative_analysis": "Compare the user papers and retrieved evidence without inventing details.",
    }
    prompt = (
        "You are ScholarSynth AI, an academic research assistant. Use only the retrieved evidence and the user input. "
        "Keep the answer concise, evidence-aware, and non-repetitive. If evidence is weak, say that evidence is limited.\n\n"
        f"Task type: {task}\n"
        f"Topic: {example['topic']}\n"
        f"Instruction: {example['instruction']}\n"
        f"Expected answer style: {style_rules.get(task, 'Write a concise academic answer.')}\n"
        f"User input:\n{example['input'][:1200]}\n\n"
        f"Retrieved evidence:\n{evidence[:1800]}\n\n"
        "Answer in 2-5 clear sentences and refer to retrieved paper titles when useful:"
    )
    return prompt, retrieval_results


def clean_generation(text: str) -> str:
    text = " ".join((text or "").split())
    repeated_markers = ["a page, a page", "a document, a page", "a document, a answer"]
    if any(marker in text.lower() for marker in repeated_markers):
        text = text.split(", a page")[0].strip()
    return text


def run_baseline_evaluation(
    finetune_test_path: str | Path = "data/finetune_test.jsonl",
    chroma_dir: str | Path = "data/chroma",
    output_dir: str | Path = "outputs",
    sample_size: int = 40,
    model_name: str = "google/flan-t5-base",
    bertscore_model: str = "distilbert-base-uncased",
    bertscore_num_layers: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = select_balanced_examples(load_jsonl(finetune_test_path), sample_size=sample_size)
    embedding_model = load_embedding_model()
    vector_store = ChromaVectorStore(str(chroma_dir))
    generator = RagGenerator(model_name=model_name)

    generation_rows: list[dict[str, Any]] = []
    for example_id, example in enumerate(examples, start=1):
        print(f"Evaluating example {example_id}/{len(examples)}: {example['task']}")
        strategies = [
            ("pretrained", build_plain_prompt(example), None),
            ("prompt_engineered", build_prompt_engineered_prompt(example), None),
        ]
        rag_prompt, retrieval_results = build_rag_prompt(example, vector_store, embedding_model)
        strategies.append(("rag_system", rag_prompt, retrieval_results))

        for strategy, prompt, retrieval_payload in strategies:
            candidate = clean_generation(generator.generate(prompt, max_new_tokens=96))
            retrieved_titles = []
            if retrieval_payload is not None:
                retrieved_titles = [
                    meta.get("title", "") for meta in retrieval_payload.get("metadatas", [[]])[0]
                ]
            generation_rows.append(
                {
                    "example_id": example_id,
                    "strategy": strategy,
                    "task": example["task"],
                    "topic": example["topic"],
                    "instruction": example["instruction"],
                    "reference": example["output"],
                    "candidate": candidate,
                    "retrieved_titles": " | ".join(retrieved_titles),
                }
            )

    generations_df = pd.DataFrame(generation_rows)
    generations_path = output_dir / "baseline_40_generations.csv"
    generations_df.to_csv(generations_path, index=False)

    metric_rows = compute_metric_rows(
        generations_df,
        bertscore_model=bertscore_model,
        bertscore_num_layers=bertscore_num_layers,
    )

    metrics_df = save_metrics(metric_rows, output_dir / "baseline_40_metrics.csv")
    aggregate_df = (
        metrics_df.groupby("model")[["bleu", "rouge1", "rouge2", "rougeL", "bertscore_f1"]]
        .mean()
        .reset_index()
        .sort_values("rougeL", ascending=False)
    )
    aggregate_df.to_csv(output_dir / "baseline_40_comparison_table.csv", index=False)
    write_comparison_markdown(
        aggregate_df,
        metrics_df,
        generations_df,
        output_dir / "baseline_40_comparison.md",
        sample_size=sample_size,
    )
    return metrics_df, aggregate_df, generations_df


def compute_metric_rows(
    generations_df: pd.DataFrame,
    bertscore_model: str = "distilbert-base-uncased",
    bertscore_num_layers: int = 5,
) -> list[dict[str, Any]]:
    candidates = generations_df["candidate"].fillna("").tolist()
    references = generations_df["reference"].fillna("").tolist()
    _, _, bert_f1 = bertscore(
        candidates,
        references,
        lang="en",
        model_type=bertscore_model,
        num_layers=bertscore_num_layers,
        verbose=False,
    )

    metric_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(generations_df.itertuples(index=False)):
        scores = {
            "bleu": compute_bleu(row.reference, row.candidate),
            "bertscore_f1": float(bert_f1[idx]),
        }
        scores.update(compute_rouge(row.reference, row.candidate))
        scores.update(
            {
                "model": row.strategy,
                "example_id": row.example_id,
                "task": row.task,
                "topic": row.topic,
            }
        )
        metric_rows.append(scores)
    return metric_rows


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_comparison_markdown(
    aggregate_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    generations_df: pd.DataFrame,
    output_path: str | Path,
    sample_size: int,
) -> None:
    task_table = (
        metrics_df.groupby(["model", "task"])[["bleu", "rougeL", "bertscore_f1"]]
        .mean()
        .reset_index()
        .sort_values(["task", "model"])
    )
    lines = [
        "# Baseline Evaluation on 40 Examples",
        "",
        f"Evaluated examples: {sample_size}",
        "",
        "## Aggregate Comparison Table",
        dataframe_to_markdown_table(aggregate_df),
        "",
        "## Task-Level Metrics",
        dataframe_to_markdown_table(task_table),
        "",
        "## Qualitative Samples",
    ]

    for example_id in sorted(generations_df["example_id"].unique())[:5]:
        group = generations_df[generations_df["example_id"] == example_id]
        first = group.iloc[0]
        lines.extend(
            [
                "",
                f"### Example {example_id}: {first['task']}",
                f"Topic: {first['topic']}",
                "",
                "Reference:",
                first["reference"],
                "",
            ]
        )
        for row in group.itertuples(index=False):
            lines.append(f"#### {row.strategy}")
            lines.append(row.candidate)
            if row.retrieved_titles:
                lines.append("")
                lines.append(f"Retrieved titles: {row.retrieved_titles}")
            lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
