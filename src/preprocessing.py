from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def clean_text(text: str) -> str:
    text = text or ""
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def first_sentences(text: str, max_sentences: int = 2, max_words: int = 90) -> str:
    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(clean_text(text)) if sentence.strip()]
    selected = " ".join(sentences[:max_sentences]) if sentences else clean_text(text)
    words = selected.split()
    if len(words) > max_words:
        selected = " ".join(words[:max_words]).rstrip(",;:") + "."
    return selected


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
    words = clean_text(text).split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break
    return chunks


def preprocess_papers(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["title"] = df["title"].fillna("").map(clean_text)
    df["abstract"] = df["abstract"].fillna("").map(clean_text)
    df = df[df["abstract"].str.len() > 50]
    df = df.drop_duplicates(subset=["title", "abstract"]).reset_index(drop=True)
    df["clean_text"] = (df["title"] + ". " + df["abstract"]).map(clean_text)

    rows: list[dict] = []
    for _, row in df.iterrows():
        chunks = chunk_text(row["clean_text"])
        for idx, chunk in enumerate(chunks):
            row_dict = row.to_dict()
            row_dict["chunk_id"] = f"{row['paper_id']}_chunk_{idx}"
            row_dict["chunk_text"] = chunk
            rows.append(row_dict)
    return pd.DataFrame(rows)


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.75,
    val_ratio: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=random_state)
    relative_val_ratio = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(temp_df, test_size=(1 - relative_val_ratio), random_state=random_state)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_finetune_dataset(df: pd.DataFrame) -> list[dict]:
    examples: list[dict] = []
    clean_df = df.copy()
    clean_df["title"] = clean_df["title"].fillna("").map(clean_text)
    clean_df["abstract"] = clean_df["abstract"].fillna("").map(clean_text)
    clean_df = clean_df[clean_df["abstract"].str.len() > 50]
    clean_df = clean_df.drop_duplicates(subset=["title", "abstract"]).reset_index(drop=True)

    for topic, topic_df in clean_df.groupby("topic"):
        topic_df = topic_df.head(30).reset_index(drop=True)

        for row in topic_df.itertuples(index=False):
            short_summary = first_sentences(row.abstract, max_sentences=2, max_words=90)
            contribution = first_sentences(row.abstract, max_sentences=1, max_words=60)
            examples.extend(
                [
                    {
                        "task": "paper_summary",
                        "topic": topic,
                        "instruction": "Summarize the research paper abstract in 2-3 academic sentences.",
                        "input": f"Title: {row.title}\nAbstract: {row.abstract}",
                        "output": f"{row.title} studies {short_summary}",
                    },
                    {
                        "task": "technical_explanation",
                        "topic": topic,
                        "instruction": "Explain the paper's main idea in simple terms for an early-stage researcher.",
                        "input": f"Title: {row.title}\nAbstract: {row.abstract}",
                        "output": (
                            f"In simple terms, this paper is about {contribution} "
                            "The main value is that it helps readers understand a specific method, system, or evaluation "
                            "problem within the broader research area."
                        ),
                    },
                    {
                        "task": "evidence_based_qa",
                        "topic": topic,
                        "instruction": "Answer the question using only the given paper abstract.",
                        "input": (
                            f"Question: What problem does this paper address?\n"
                            f"Title: {row.title}\nAbstract: {row.abstract}"
                        ),
                        "output": f"Based on the abstract, the paper addresses this problem: {short_summary}",
                    },
                ]
            )

        group_size = 5
        for start in range(0, len(topic_df), group_size):
            group = topic_df.iloc[start : start + group_size]
            if len(group) < 3:
                continue

            paper_blocks = "\n".join(
                f"Paper {idx + 1}: {paper.title}. Abstract: {first_sentences(paper.abstract, max_sentences=2, max_words=80)}"
                for idx, paper in enumerate(group.itertuples(index=False))
            )
            titles = "; ".join(group["title"].head(4).tolist())
            review_output = (
                f"Research on {topic} shows a clear focus on methods represented by papers such as {titles}. "
                "Across these papers, the common pattern is the use of model-based techniques to improve retrieval, "
                "generation, explanation, or evaluation quality. The evidence suggests that stronger grounding, "
                "better evaluation, and domain adaptation are recurring themes in this area."
            )
            gap_output = (
                f"For {topic}, the retrieved papers suggest several research gaps: more reliable evaluation protocols, "
                "stronger evidence grounding, better handling of domain-specific terminology, and clearer comparison "
                "between retrieval-based, fine-tuned, and prompt-engineered systems. Future work should test these "
                "methods on larger and more diverse scientific corpora."
            )
            comparison_output = (
                f"The papers on {topic} are related because they address complementary parts of the same research area. "
                "Some papers emphasize model or system design, while others focus on evaluation, reliability, or "
                "domain-specific application. Together, they show that practical research assistants need both accurate "
                "retrieval and controlled generation."
            )
            examples.extend(
                [
                    {
                        "task": "literature_review",
                        "topic": topic,
                        "instruction": f"Generate a short literature review on {topic} using the provided papers.",
                        "input": paper_blocks,
                        "output": review_output,
                    },
                    {
                        "task": "research_gap_analysis",
                        "topic": topic,
                        "instruction": f"Identify research gaps for {topic} using the provided papers.",
                        "input": paper_blocks,
                        "output": gap_output,
                    },
                    {
                        "task": "comparative_analysis",
                        "topic": topic,
                        "instruction": "Compare the provided papers and explain how they relate to each other.",
                        "input": paper_blocks,
                        "output": comparison_output,
                    },
                ]
            )

    return examples


def split_records(
    records: list[dict],
    train_ratio: float = 0.75,
    val_ratio: float = 0.10,
    random_state: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    if not records:
        return [], [], []

    record_df = pd.DataFrame(records)
    train_df, temp_df = train_test_split(record_df, test_size=(1 - train_ratio), random_state=random_state)
    relative_val_ratio = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(temp_df, test_size=(1 - relative_val_ratio), random_state=random_state)
    return (
        train_df.to_dict(orient="records"),
        val_df.to_dict(orient="records"),
        test_df.to_dict(orient="records"),
    )


def save_jsonl(records: Iterable[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
