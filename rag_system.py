"""Backward-compatible entry point for the refactored RAG project."""

from __future__ import annotations

from engineering.rag_project.cli import main as cli_main
from engineering.rag_project.config import RAGSettings, default_settings
from engineering.rag_project.generation import build_no_rag_prompt, build_prompt, generate_answer, generate_with_ollama
from engineering.rag_project.service import RAGService, load_service

_DEFAULT_SERVICE = load_service(default_settings())


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    hits = _DEFAULT_SERVICE.retrieve(query, top_k=top_k)
    return [
        {
            "doc": {
                "id": hit.chunk.document_id,
                "source": hit.chunk.source,
                "title": hit.chunk.document_title,
                "text": hit.chunk.text,
            },
            "score": hit.score,
        }
        for hit in hits
    ]


def rag_pipeline(query: str, top_k: int = 3, verbose: bool = True) -> dict:
    result = _DEFAULT_SERVICE.answer(query, top_k=top_k)
    return {
        "query": query,
        "retrieved": [
            {"id": hit.chunk.document_id, "title": hit.chunk.document_title, "score": hit.score}
            for hit in result["hits"]
        ],
        "prompt": result["prompt"],
        "answer": result["answer"],
    }


def compare_no_rag_vs_rag(query: str) -> dict:
    return _DEFAULT_SERVICE.compare_no_rag_vs_rag(query)


if __name__ == "__main__":
    cli_main()
