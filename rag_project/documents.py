from __future__ import annotations

import csv
import html
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Iterable

from .models import Document

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".rst"}
SUPPORTED_STRUCTURED_EXTENSIONS = {".json", ".csv", ".html", ".htm", ".xml"}
SUPPORTED_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_STRUCTURED_EXTENSIONS | {".pdf"}


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9а-яА-Я]+", "-", value.strip().lower()).strip("-")
    return cleaned or "document"


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("# ").strip()
        if stripped:
            return stripped[:120]
    return fallback


def _text_from_csv(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return ""

    lines: list[str] = []
    for row_number, row in enumerate(rows[:200], start=1):
        lines.append(f"Row {row_number}: " + " | ".join(cell.strip() for cell in row))
    return "\n".join(lines)


def _text_from_json(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        data = json.load(handle)
    return json.dumps(data, ensure_ascii=False, indent=2)


def _text_from_html(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        raw = handle.read()
    return html.unescape(re.sub(r"<[^>]+>", " ", raw))


def _text_from_xml(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        raw = handle.read()
    return html.unescape(re.sub(r"<[^>]+>", " ", raw))


def _text_from_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Для загрузки PDF установите пакет 'pypdf'.") from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def read_text_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".json":
        return _text_from_json(path)
    if suffix == ".csv":
        return _text_from_csv(path)
    if suffix in {".html", ".htm"}:
        return _text_from_html(path)
    if suffix == ".xml":
        return _text_from_xml(path)
    if suffix == ".pdf":
        return _text_from_pdf(path)
    raise ValueError(f"Неподдерживаемый формат файла: {path.name}")


def document_from_path(path: Path, base_dir: Path | None = None) -> Document:
    text = read_text_from_path(path)
    relative = str(path.relative_to(base_dir)) if base_dir and path.is_relative_to(base_dir) else str(path)
    title = _title_from_text(text, path.stem.replace("_", " ").strip())
    return Document(
        id=_slugify(relative),
        title=title,
        source=relative,
        text=text,
        path=str(path),
        metadata={"extension": path.suffix.lower()},
    )


def document_from_bytes(data: bytes, filename: str, title: str | None = None) -> Document:
    path = Path(filename)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Неподдерживаемый формат файла: {filename}")

    tmp = BytesIO(data)
    if suffix in SUPPORTED_TEXT_EXTENSIONS:
        text = tmp.read().decode("utf-8", errors="ignore")
    elif suffix == ".json":
        text = json.dumps(json.loads(tmp.read().decode("utf-8", errors="ignore")), ensure_ascii=False, indent=2)
    elif suffix == ".csv":
        rows = list(csv.reader(tmp.read().decode("utf-8", errors="ignore").splitlines()))
        text = "\n".join(
            f"Row {row_number}: " + " | ".join(cell.strip() for cell in row)
            for row_number, row in enumerate(rows[:200], start=1)
        )
    elif suffix in {".html", ".htm", ".xml"}:
        raw = tmp.read().decode("utf-8", errors="ignore")
        text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Для загрузки PDF установите пакет 'pypdf'.") from exc

        reader = PdfReader(BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {filename}")

    resolved_title = title or _title_from_text(text, path.stem.replace("_", " ").strip())
    return Document(
        id=_slugify(filename),
        title=resolved_title,
        source=filename,
        text=text,
        metadata={"extension": suffix},
    )


def documents_from_paths(paths: Iterable[Path]) -> list[Document]:
    documents: list[Document] = []
    for path in paths:
        if path.is_dir():
            for nested in sorted(path.rglob("*")):
                if nested.is_file() and nested.suffix.lower() in SUPPORTED_EXTENSIONS:
                    documents.append(document_from_path(nested, base_dir=path))
        elif path.is_file():
            documents.append(document_from_path(path, base_dir=path.parent))
    return documents


def load_default_documents(docs_dir: Path) -> list[Document]:
    if not docs_dir.exists():
        return []

    documents: list[Document] = []
    for path in sorted(docs_dir.glob("*.txt")):
        documents.append(document_from_path(path, base_dir=docs_dir))
    return documents


def documents_to_json(documents: list[Document]) -> list[dict]:
    return [
        {
            "id": document.id,
            "title": document.title,
            "source": document.source,
            "text": document.text,
            "path": document.path,
            "metadata": document.metadata,
        }
        for document in documents
    ]


def documents_from_json(payload: list[dict]) -> list[Document]:
    return [Document(**item) for item in payload]
