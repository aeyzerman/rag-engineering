from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class Document:
    id: str
    title: str
    source: str
    text: str
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Chunk:
    id: str
    document_id: str
    document_title: str
    source: str
    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float


@dataclass(slots=True)
class IndexBundle:
    vectorizer: Any
    matrix: Any
    chunks: list[Chunk]
    created_at: str
    config: dict[str, Any]

    @property
    def vocabulary_size(self) -> int:
        return len(getattr(self.vectorizer, "vocabulary_", {}))
