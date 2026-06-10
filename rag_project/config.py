from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
ENGINEERING_DIR = PACKAGE_DIR.parent
REPO_ROOT = ENGINEERING_DIR.parent
DEFAULT_DOCS_DIR = ENGINEERING_DIR / "docs"
DEFAULT_STORE_DIR = ENGINEERING_DIR / "rag_store"


@dataclass(slots=True)
class RAGSettings:
    store_dir: Path = DEFAULT_STORE_DIR
    docs_dir: Path = DEFAULT_DOCS_DIR
    model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    top_k: int = 3
    chunk_size: int = 120
    chunk_overlap: int = 24
    max_df: float = 0.9


def default_settings() -> RAGSettings:
    return RAGSettings()
