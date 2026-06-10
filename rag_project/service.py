from __future__ import annotations

import json
from time import perf_counter
from pathlib import Path
from typing import Iterable

from .config import RAGSettings
from .documents import (
    document_from_bytes,
    document_from_path,
    documents_from_json,
    documents_to_json,
    load_default_documents,
)
from .generation import build_no_rag_prompt, build_prompt, generate_answer, generate_with_ollama
from .index import build_index, load_index, save_index, search
from .models import Document, IndexBundle, RetrievalHit

DOCUMENTS_FILENAME = "rag_documents.json"


class RAGService:
    def __init__(
        self,
        settings: RAGSettings | None = None,
        documents: list[Document] | None = None,
        index: IndexBundle | None = None,
    ) -> None:
        self.settings = settings or RAGSettings()
        self.documents = documents or []
        self.index = index

    @property
    def store_dir(self) -> Path:
        return self.settings.store_dir

    @classmethod
    def load(cls, settings: RAGSettings | None = None) -> "RAGService":
        resolved_settings = settings or RAGSettings()
        store_dir = resolved_settings.store_dir
        store_dir.mkdir(parents=True, exist_ok=True)

        documents_path = store_dir / DOCUMENTS_FILENAME
        if documents_path.exists():
            documents = documents_from_json(json.loads(documents_path.read_text(encoding="utf-8")))
        else:
            documents = load_default_documents(resolved_settings.docs_dir)

        index = load_index(store_dir)
        return cls(settings=resolved_settings, documents=documents, index=index)

    def persist(self) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        (self.store_dir / DOCUMENTS_FILENAME).write_text(
            json.dumps(documents_to_json(self.documents), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self.index is not None:
            save_index(self.index, self.store_dir)

    def reset_to_default_corpus(self) -> None:
        self.documents = load_default_documents(self.settings.docs_dir)
        self.rebuild_index()
        self.persist()

    def add_documents(self, documents: Iterable[Document], rebuild: bool = True) -> None:
        existing_ids = {document.id for document in self.documents}
        for document in documents:
            if document.id in existing_ids:
                continue
            self.documents.append(document)
            existing_ids.add(document.id)
        if rebuild:
            self.rebuild_index()
            self.persist()

    def ingest_paths(self, paths: Iterable[Path], rebuild: bool = True) -> None:
        documents: list[Document] = []
        for path in paths:
            if path.is_dir():
                for nested in sorted(path.rglob("*")):
                    if nested.is_file():
                        try:
                            documents.append(document_from_path(nested, base_dir=path))
                        except ValueError:
                            continue
            elif path.is_file():
                documents.append(document_from_path(path, base_dir=path.parent))

        self.add_documents(documents, rebuild=rebuild)

    def ingest_bytes(self, data: bytes, filename: str, title: str | None = None, rebuild: bool = True) -> None:
        self.add_documents([document_from_bytes(data, filename, title=title)], rebuild=rebuild)

    def ingest_text(self, title: str, text: str, source: str = "Manual input", rebuild: bool = True) -> None:
        document = Document(
            id=f"manual-{len(self.documents) + 1:04d}",
            title=title.strip() or "Untitled",
            source=source,
            text=text.strip(),
        )
        self.add_documents([document], rebuild=rebuild)

    def rebuild_index(self) -> IndexBundle:
        self.index = build_index(self.documents, self.settings)
        return self.index

    def ensure_index(self) -> IndexBundle:
        if self.index is None:
            self.index = self.rebuild_index()
            self.persist()
        return self.index

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        index = self.ensure_index()
        return search(index, query, top_k=top_k or self.settings.top_k)

    def answer(self, query: str, top_k: int | None = None, model: str | None = None) -> dict:
        started_at = perf_counter()
        retrieval_started_at = perf_counter()
        hits = self.retrieve(query, top_k=top_k)
        retrieval_seconds = perf_counter() - retrieval_started_at

        generation_started_at = perf_counter()
        answer = generate_answer(query, hits, model=model or self.settings.model)
        generation_seconds = perf_counter() - generation_started_at
        prompt = build_prompt(query, hits) if hits else build_no_rag_prompt(query)
        total_seconds = perf_counter() - started_at
        return {
            "query": query,
            "answer": answer,
            "prompt": prompt,
            "hits": hits,
            "model": model or self.settings.model,
            "timings": {
                "retrieval_seconds": retrieval_seconds,
                "generation_seconds": generation_seconds,
                "total_seconds": total_seconds,
            },
        }

    def compare_no_rag_vs_rag(self, query: str, model: str | None = None) -> dict:
        no_rag = generate_with_ollama(build_no_rag_prompt(query), model=model or self.settings.model)
        rag_result = self.answer(query, model=model)
        return {"no_rag": no_rag, "rag": rag_result}

    def stats(self) -> dict:
        index = self.index
        return {
            "documents": len(self.documents),
            "chunks": len(index.chunks) if index else 0,
            "vocabulary_size": index.vocabulary_size if index else 0,
            "model": self.settings.model,
            "store_dir": str(self.store_dir),
            "index_ready": index is not None,
        }


def load_service(settings: RAGSettings | None = None) -> RAGService:
    return RAGService.load(settings=settings)
