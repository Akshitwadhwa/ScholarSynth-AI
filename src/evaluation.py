from __future__ import annotations

from pathlib import Path

import pandas as pd
from bert_score import score as bertscore
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer


def compute_bleu(reference: str, candidate: str) -> float:
    smoothie = SmoothingFunction().method1
    return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)


def compute_rouge(reference: str, candidate: str) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {metric: value.fmeasure for metric, value in scores.items()}


def compute_bertscore(reference: str, candidate: str) -> float:
    _, _, f1 = bertscore([candidate], [reference], lang="en", verbose=False)
    return float(f1.mean())


def evaluate_generation(reference: str, candidate: str) -> dict[str, float]:
    scores = {"bleu": compute_bleu(reference, candidate), "bertscore_f1": compute_bertscore(reference, candidate)}
    scores.update(compute_rouge(reference, candidate))
    return scores


def save_metrics(records: list[dict], output_path: str = "outputs/metrics.csv") -> pd.DataFrame:
    df = pd.DataFrame(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
