from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import chunk_text
from .config import RAGSettings
from .models import Chunk, Document, IndexBundle, RetrievalHit

INDEX_FILENAME = "rag_index.joblib"


def build_chunks(documents: Iterable[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        for chunk_index, chunk_text_value in enumerate(chunk_text(document.text, chunk_size, overlap), start=1):
            chunks.append(
                Chunk(
                    id=f"{document.id}::chunk-{chunk_index:03d}",
                    document_id=document.id,
                    document_title=document.title,
                    source=document.source,
                    text=chunk_text_value,
                    index=chunk_index,
                    metadata=document.metadata,
                )
            )
    return chunks


def build_index(documents: Iterable[Document], settings: RAGSettings) -> IndexBundle:
    chunks = build_chunks(documents, settings.chunk_size, settings.chunk_overlap)
    texts = [chunk.text for chunk in chunks] or ["placeholder index"]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_df=settings.max_df,
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)

    return IndexBundle(
        vectorizer=vectorizer,
        matrix=matrix,
        chunks=chunks,
        created_at=datetime.now(timezone.utc).isoformat(),
        config={
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "max_df": settings.max_df,
            "backend": "tfidf",
        },
    )


def search(index: IndexBundle, query: str, top_k: int = 3) -> list[RetrievalHit]:
    if not index.chunks:
        return []

    query_vector = index.vectorizer.transform([query])
    scores = cosine_similarity(query_vector, index.matrix)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    hits: list[RetrievalHit] = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        hits.append(RetrievalHit(chunk=index.chunks[idx], score=float(scores[idx])))
    return hits


def save_index(index: IndexBundle, store_dir: Path) -> Path:
    store_dir.mkdir(parents=True, exist_ok=True)
    target = store_dir / INDEX_FILENAME
    joblib.dump(index, target)
    return target


def load_index(store_dir: Path) -> IndexBundle | None:
    target = store_dir / INDEX_FILENAME
    if not target.exists():
        return None
    return joblib.load(target)
