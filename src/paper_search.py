from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List

import arxiv
import pandas as pd
import requests


SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"


@dataclass
class PaperRecord:
    paper_id: str
    title: str
    abstract: str
    authors: str
    year: int | None
    venue: str
    url: str
    topic: str
    source: str


def _normalize_text(value: str | None) -> str:
    return " ".join((value or "").split())


def search_arxiv(topic: str, max_results: int = 20) -> List[PaperRecord]:
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results: List[PaperRecord] = []
    for result in search.results():
        results.append(
            PaperRecord(
                paper_id=result.entry_id.split("/")[-1],
                title=_normalize_text(result.title),
                abstract=_normalize_text(result.summary),
                authors=", ".join(author.name for author in result.authors),
                year=result.published.year if result.published else None,
                venue="arXiv",
                url=result.entry_id,
                topic=topic,
                source="arxiv",
            )
        )
    return results


def search_semantic_scholar(topic: str, limit: int = 20, api_key: str | None = None) -> List[PaperRecord]:
    params = {
        "query": topic,
        "limit": limit,
        "fields": "paperId,title,abstract,authors,year,venue,url",
    }
    headers = {"x-api-key": api_key} if api_key else {}
    response = requests.get(SEMANTIC_SCHOLAR_API, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    payload = response.json()

    results: List[PaperRecord] = []
    for item in payload.get("data", []):
        results.append(
            PaperRecord(
                paper_id=item.get("paperId", ""),
                title=_normalize_text(item.get("title")),
                abstract=_normalize_text(item.get("abstract")),
                authors=", ".join(author.get("name", "") for author in item.get("authors", [])),
                year=item.get("year"),
                venue=_normalize_text(item.get("venue")) or "Semantic Scholar",
                url=item.get("url", ""),
                topic=topic,
                source="semantic_scholar",
            )
        )
    return results


def merge_and_deduplicate(records: Iterable[PaperRecord]) -> pd.DataFrame:
    seen: set[tuple[str, str]] = set()
    rows = []
    for record in records:
        key = (record.paper_id, record.title.lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append(asdict(record))
    return pd.DataFrame(rows)


def collect_papers_for_topic(
    topic: str,
    arxiv_limit: int = 20,
    semantic_limit: int = 20,
    semantic_api_key: str | None = None,
) -> pd.DataFrame:
    records = []
    records.extend(search_arxiv(topic, max_results=arxiv_limit))
    try:
        records.extend(search_semantic_scholar(topic, limit=semantic_limit, api_key=semantic_api_key))
    except requests.RequestException:
        pass
    return merge_and_deduplicate(records)


def save_raw_papers(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False)
