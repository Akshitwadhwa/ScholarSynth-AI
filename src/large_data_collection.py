from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.paper_search import collect_papers_for_topic
from src.preprocessing import (
    build_finetune_dataset,
    preprocess_papers,
    save_jsonl,
    split_dataset,
    split_records,
)
from src.topic_bank import LARGE_SCALE_TOPICS


DATA_DIR = Path("data")
RAW_CSV = DATA_DIR / "raw_papers.csv"
PROCESSED_CSV = DATA_DIR / "processed_papers.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"
FINETUNE_JSONL = DATA_DIR / "finetune_dataset.jsonl"
FINETUNE_TRAIN_JSONL = DATA_DIR / "finetune_train.jsonl"
FINETUNE_VAL_JSONL = DATA_DIR / "finetune_val.jsonl"
FINETUNE_TEST_JSONL = DATA_DIR / "finetune_test.jsonl"
CHECKPOINT_CSV = DATA_DIR / "raw_papers_checkpoint.csv"


def deduplicate_papers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ["paper_id", "title", "abstract"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)
    df = df[df["abstract"].str.len() > 50]
    df = df.drop_duplicates(subset=["paper_id"], keep="first")
    df = df.drop_duplicates(subset=["title", "abstract"], keep="first")
    return df.reset_index(drop=True)


def load_existing_raw(append_existing: bool) -> pd.DataFrame:
    if append_existing and RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)
    return pd.DataFrame()


def regenerate_derived_files(raw_df: pd.DataFrame) -> None:
    processed_df = preprocess_papers(raw_df)
    processed_df.to_csv(PROCESSED_CSV, index=False)

    train_df, val_df, test_df = split_dataset(processed_df)
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    finetune_records = build_finetune_dataset(raw_df)
    finetune_train, finetune_val, finetune_test = split_records(finetune_records)
    save_jsonl(finetune_records, FINETUNE_JSONL)
    save_jsonl(finetune_train, FINETUNE_TRAIN_JSONL)
    save_jsonl(finetune_val, FINETUNE_VAL_JSONL)
    save_jsonl(finetune_test, FINETUNE_TEST_JSONL)

    print("Regenerated derived files:")
    print(f"- processed chunks: {len(processed_df)}")
    print(f"- split rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(
        "- fine-tuning examples: "
        f"total={len(finetune_records)}, train={len(finetune_train)}, "
        f"val={len(finetune_val)}, test={len(finetune_test)}"
    )


def collect_large_dataset(
    target_papers: int,
    arxiv_limit: int,
    semantic_limit: int,
    sleep_seconds: float,
    semantic_api_key: str | None,
    append_existing: bool,
) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    collected_df = load_existing_raw(append_existing)
    collected_df = deduplicate_papers(collected_df) if not collected_df.empty else collected_df

    print(f"Starting with {len(collected_df)} unique papers.")
    for index, topic in enumerate(LARGE_SCALE_TOPICS, start=1):
        if len(collected_df) >= target_papers:
            break

        print(f"[{index}/{len(LARGE_SCALE_TOPICS)}] Collecting: {topic}")
        topic_df = collect_papers_for_topic(
            topic,
            arxiv_limit=arxiv_limit,
            semantic_limit=semantic_limit,
            semantic_api_key=semantic_api_key,
        )
        before_count = len(collected_df)
        collected_df = pd.concat([collected_df, topic_df], ignore_index=True)
        collected_df = deduplicate_papers(collected_df)
        added_count = len(collected_df) - before_count
        collected_df.to_csv(CHECKPOINT_CSV, index=False)
        print(f"  added={added_count}, total={len(collected_df)}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    collected_df = collected_df.head(target_papers).reset_index(drop=True)
    collected_df.to_csv(RAW_CSV, index=False)
    print(f"Saved {len(collected_df)} papers to {RAW_CSV}")
    return collected_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a large paper corpus for ScholarSynth AI.")
    parser.add_argument("--target-papers", type=int, default=5000)
    parser.add_argument("--arxiv-limit", type=int, default=120)
    parser.add_argument("--semantic-limit", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--semantic-api-key", default=None)
    parser.add_argument("--fresh", action="store_true", help="Ignore existing raw_papers.csv and start fresh.")
    parser.add_argument(
        "--skip-derived",
        action="store_true",
        help="Only collect raw papers; do not regenerate processed/split/fine-tuning files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_df = collect_large_dataset(
        target_papers=args.target_papers,
        arxiv_limit=args.arxiv_limit,
        semantic_limit=args.semantic_limit,
        sleep_seconds=args.sleep_seconds,
        semantic_api_key=args.semantic_api_key,
        append_existing=not args.fresh,
    )
    if not args.skip_derived:
        regenerate_derived_files(raw_df)


if __name__ == "__main__":
    main()
