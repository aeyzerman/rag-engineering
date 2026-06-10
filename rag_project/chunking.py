from __future__ import annotations


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 24) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []

    words = cleaned.split(" ")
    if len(words) <= chunk_size:
        return [cleaned]

    step = max(chunk_size - overlap, 1)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks
