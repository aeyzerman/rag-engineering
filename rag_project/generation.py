from __future__ import annotations

import os

from .models import RetrievalHit


def build_prompt(query: str, hits: list[RetrievalHit]) -> str:
    context_text = "\n\n".join(
        f"[Источник: {hit.chunk.source} | {hit.chunk.document_title}]\n{hit.chunk.text}"
        for hit in hits
    )
    return (
        "Ты - учебный ассистент кафедры информационных технологий.\n"
        "Отвечай только на основе предоставленного контекста.\n"
        "Если ответа в контексте нет, честно скажи об этом.\n"
        "Пиши по-русски, кратко и по существу.\n\n"
        f"КОНТЕКСТ:\n{context_text}\n\n"
        f"ВОПРОС: {query}\n\n"
        "ОТВЕТ:"
    )


def build_no_rag_prompt(query: str) -> str:
    return (
        "Ты - учебный ассистент кафедры информационных технологий.\n"
        "Ответь кратко и по существу.\n"
        "Если не уверен, честно скажи об этом.\n\n"
        f"ВОПРОС: {query}\n\n"
        "ОТВЕТ:"
    )


def generate_with_ollama(prompt: str, model: str | None = None) -> str:
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError(
            "Не установлен пакет 'ollama'. Добавьте зависимость и запустите Ollama server."
        ) from exc

    resolved_model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    try:
        response = ollama.chat(
            model=resolved_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Модель '{resolved_model}' недоступна. Запустите Ollama и выполните `ollama pull {resolved_model}`."
        ) from exc

    return response["message"]["content"].strip()


def generate_answer(query: str, hits: list[RetrievalHit], model: str | None = None) -> str:
    prompt = build_prompt(query, hits) if hits else build_no_rag_prompt(query)
    return generate_with_ollama(prompt, model=model)
