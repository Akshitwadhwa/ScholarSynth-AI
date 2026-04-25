from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


class SQLiteMetadataStore:
    def __init__(self, db_path: str = "data/papers.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                authors TEXT,
                year INTEGER,
                venue TEXT,
                url TEXT,
                topic TEXT,
                source TEXT
            )
            """
        )
        self.connection.commit()

    def upsert_papers(self, papers_df: pd.DataFrame) -> None:
        records = papers_df[
            ["paper_id", "title", "abstract", "authors", "year", "venue", "url", "topic", "source"]
        ].drop_duplicates(subset=["paper_id"])
        rows = records.where(pd.notna(records), None).itertuples(index=False, name=None)
        self.connection.executemany(
            """
            INSERT INTO papers (paper_id, title, abstract, authors, year, venue, url, topic, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                title = excluded.title,
                abstract = excluded.abstract,
                authors = excluded.authors,
                year = excluded.year,
                venue = excluded.venue,
                url = excluded.url,
                topic = excluded.topic,
                source = excluded.source
            """,
            rows,
        )
        self.connection.commit()

    def fetch_all(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM papers", self.connection)


class ChromaVectorStore:
    def __init__(self, persist_dir: str = "data/chroma") -> None:
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="research_papers")

    def index_chunks(
        self,
        df: pd.DataFrame,
        embedding_model: SentenceTransformer,
        batch_size: int = 512,
    ) -> None:
        if df.empty:
            return
        for start in range(0, len(df), batch_size):
            batch_df = df.iloc[start : start + batch_size]
            ids = batch_df["chunk_id"].tolist()
            documents = batch_df["chunk_text"].tolist()
            embeddings = embedding_model.encode(documents).tolist()
            metadatas = batch_df[["paper_id", "title", "topic", "source", "url"]].to_dict(orient="records")
            self.collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def semantic_search(self, query: str, embedding_model: SentenceTransformer, top_k: int = 5) -> dict:
        query_embedding = embedding_model.encode([query]).tolist()
        return self.collection.query(query_embeddings=query_embedding, n_results=top_k)
