from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import RAGSettings
from .service import RAGService


def _build_settings(args: argparse.Namespace) -> RAGSettings:
    return RAGSettings(
        store_dir=Path(args.store_dir),
        docs_dir=Path(args.docs_dir),
        model=args.model,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_df=args.max_df,
    )


def _print_stats(service: RAGService) -> None:
    stats = service.stats()
    print("RAG stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def cmd_index(args: argparse.Namespace) -> None:
    service = RAGService.load(_build_settings(args))
    if args.paths:
        service.ingest_paths([Path(path) for path in args.paths], rebuild=False)
        service.rebuild_index()
        service.persist()
    else:
        service.reset_to_default_corpus()
    _print_stats(service)


def cmd_query(args: argparse.Namespace) -> None:
    service = RAGService.load(_build_settings(args))
    result = service.answer(args.question, top_k=args.top_k, model=args.model)
    print(result["answer"])
    print("\nSources:")
    for hit in result["hits"]:
        print(f"  {hit.score:.3f} | {hit.chunk.source} | {hit.chunk.document_title}")


def cmd_stats(args: argparse.Namespace) -> None:
    service = RAGService.load(_build_settings(args))
    _print_stats(service)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-project", description="RAG project tools")
    parser.add_argument("--store-dir", default=str(RAGSettings().store_dir))
    parser.add_argument("--docs-dir", default=str(RAGSettings().docs_dir))
    parser.add_argument("--model", default=RAGSettings().model)
    parser.add_argument("--top-k", type=int, default=RAGSettings().top_k)
    parser.add_argument("--chunk-size", type=int, default=RAGSettings().chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=RAGSettings().chunk_overlap)
    parser.add_argument("--max-df", type=float, default=RAGSettings().max_df)

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build or update the local index")
    index_parser.add_argument("paths", nargs="*", help="Files or directories to ingest")
    index_parser.set_defaults(func=cmd_index)

    query_parser = subparsers.add_parser("query", help="Ask a question via the index and Ollama")
    query_parser.add_argument("question", help="Question to answer")
    query_parser.add_argument("--model", default=RAGSettings().model)
    query_parser.add_argument("--top-k", type=int, default=RAGSettings().top_k)
    query_parser.set_defaults(func=cmd_query)

    stats_parser = subparsers.add_parser("stats", help="Show index statistics")
    stats_parser.set_defaults(func=cmd_stats)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
