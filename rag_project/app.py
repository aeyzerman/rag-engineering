from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "engineering.rag_project"

from .config import RAGSettings
from .generation import build_no_rag_prompt, generate_with_ollama
from .service import RAGService


def _load_service(settings: RAGSettings) -> RAGService:
    return RAGService.load(settings)


def _rerun() -> None:
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def main() -> None:
    st.set_page_config(page_title="RAG Studio", layout="wide")
    st.title("RAG Studio")

    state = st.session_state
    state.setdefault("rag_busy", False)
    state.setdefault("last_answer", None)
    state.setdefault("last_error", "")
    state.setdefault("last_query", "")

    with st.sidebar:
        st.header("Настройки")
        default = RAGSettings()
        store_dir = Path(st.text_input("Папка хранилища", str(default.store_dir)))
        docs_dir = Path(st.text_input("Базовый корпус", str(default.docs_dir)))
        model_options = ["qwen2.5:7b", "llama3.2:3b", "qwen2.5:14b", "llama3.2:1b", "custom"]
        chosen_model = st.selectbox("Ollama model", model_options, index=0)
        model = st.text_input("Custom model", default.model) if chosen_model == "custom" else chosen_model
        top_k = st.slider("Top-K", min_value=1, max_value=8, value=default.top_k)
        chunk_size = st.slider("Chunk size", min_value=40, max_value=320, value=default.chunk_size, step=10)
        chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=120, value=default.chunk_overlap, step=4)
        max_df = st.slider("TF-IDF max_df", min_value=0.5, max_value=1.0, value=default.max_df, step=0.05)
        settings = RAGSettings(
            store_dir=store_dir,
            docs_dir=docs_dir,
            model=model,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_df=max_df,
        )
        service = _load_service(settings)
        st.metric("Документы", len(service.documents))
        st.metric("Чанки", len(service.index.chunks) if service.index else 0)

        if st.button("Пересобрать индекс из текущего корпуса", use_container_width=True):
            service.rebuild_index()
            service.persist()
            st.success("Индекс пересобран")
            _rerun()

        if st.button("Сбросить на базовый корпус", use_container_width=True):
            service.reset_to_default_corpus()
            st.success("Корпус восстановлен из `engineering/docs`")
            _rerun()

    tab_chat, tab_ingest, tab_docs, tab_stats = st.tabs(["Чат", "Индексация", "Документы", "Статистика"])

    with tab_chat:
        question = st.text_area(
            "Вопрос",
            placeholder="Например: Что такое Delta Lake и зачем он нужен?",
            height=120,
            disabled=state["rag_busy"],
        )
        rag_enabled = st.checkbox("Использовать RAG-контекст", value=True, disabled=state["rag_busy"])
        compare = st.checkbox("Показать сравнение с ответом без RAG", value=False, disabled=state["rag_busy"])
        submit_disabled = state["rag_busy"]

        if st.button("Спросить", type="primary", disabled=submit_disabled):
            if not question.strip():
                st.warning("Введите вопрос.")
            elif compare:
                state["rag_busy"] = True
                try:
                    with st.status("Модель думает", expanded=True) as status:
                        status.write("Сначала получаю ответ без RAG, затем ответ с контекстом.")
                        compare_result = service.compare_no_rag_vs_rag(question, model=model)
                        status.update(label="Ответ получен", state="complete")
                    state["last_answer"] = {
                        "mode": "compare",
                        "data": compare_result,
                    }
                    state["last_query"] = question
                except Exception as exc:
                    state["last_error"] = str(exc)
                    st.error(state["last_error"])
                finally:
                    state["rag_busy"] = False
            else:
                state["rag_busy"] = True
                try:
                    with st.status("Модель думает", expanded=True) as status:
                        status.write("Ищу релевантный контекст и генерирую ответ.")
                        if rag_enabled:
                            result = service.answer(question, top_k=top_k, model=model)
                        else:
                            result = {
                                "answer": generate_with_ollama(build_no_rag_prompt(question), model=model),
                                "hits": [],
                                "prompt": "",
                                "timings": {
                                    "retrieval_seconds": 0.0,
                                    "generation_seconds": 0.0,
                                    "total_seconds": 0.0,
                                },
                            }
                        status.update(label="Ответ получен", state="complete")
                    state["last_answer"] = {
                        "mode": "rag" if rag_enabled else "no_rag",
                        "data": result,
                    }
                    state["last_query"] = question
                except Exception as exc:
                    state["last_error"] = str(exc)
                    st.error(state["last_error"])
                finally:
                    state["rag_busy"] = False

        if state["last_answer"] and state["last_query"]:
            payload = state["last_answer"]
            if payload["mode"] == "compare":
                compare_result = payload["data"]
                col_left, col_right = st.columns(2)
                with col_left:
                    st.subheader("Без RAG")
                    st.write(compare_result["no_rag"])
                with col_right:
                    st.subheader("С RAG")
                    st.write(compare_result["rag"]["answer"])
                    timings = compare_result["rag"].get("timings", {})
                    if timings:
                        st.caption(
                            f"Время: retrieval {timings.get('retrieval_seconds', 0):.2f}s, "
                            f"generation {timings.get('generation_seconds', 0):.2f}s, "
                            f"total {timings.get('total_seconds', 0):.2f}s"
                        )
                    with st.expander("Источники"):
                        for hit in compare_result["rag"]["hits"]:
                            st.write(f"{hit.score:.3f} | {hit.chunk.source} | {hit.chunk.document_title}")
            else:
                result = payload["data"]
                st.subheader("Ответ")
                st.write(result["answer"])
                timings = result.get("timings", {})
                if timings:
                    left, middle, right = st.columns(3)
                    left.metric("Retrieval", f"{timings.get('retrieval_seconds', 0):.2f}s")
                    middle.metric("Generation", f"{timings.get('generation_seconds', 0):.2f}s")
                    right.metric("Total", f"{timings.get('total_seconds', 0):.2f}s")
                    st.caption("Чат был заблокирован на время выполнения запроса.")
                if result["hits"]:
                    with st.expander("Источники и scores"):
                        for hit in result["hits"]:
                            st.write(f"{hit.score:.3f} | {hit.chunk.source} | {hit.chunk.document_title}")
                with st.expander("Prompt"):
                    st.code(result.get("prompt", ""), language="text")

    with tab_ingest:
        st.subheader("Добавить текст вручную")
        manual_title = st.text_input("Заголовок", placeholder="Например: Лекция по Spark")
        manual_source = st.text_input("Источник", value="Manual input")
        manual_text = st.text_area("Текст", height=200, placeholder="Вставьте контекст сюда")
        if st.button("Добавить текст и переиндексировать", use_container_width=True):
            if not manual_text.strip():
                st.warning("Текст пустой.")
            else:
                service.ingest_text(manual_title or "Untitled", manual_text, source=manual_source)
                service.persist()
                st.success("Текст добавлен и индекс пересобран")
                _rerun()

        st.divider()
        st.subheader("Загрузить файлы")
        uploads = st.file_uploader(
            "Поддерживаются txt, md, json, csv, xml, html, pdf",
            accept_multiple_files=True,
            type=["txt", "md", "rst", "json", "csv", "xml", "html", "htm", "pdf"],
        )
        if st.button("Добавить файлы и переиндексировать", use_container_width=True):
            if not uploads:
                st.warning("Выберите хотя бы один файл.")
            else:
                staged = []
                for upload in uploads:
                    staged.append((upload.read(), upload.name))
                for data, name in staged:
                    service.ingest_bytes(data, name, rebuild=False)
                service.rebuild_index()
                service.persist()
                st.success("Файлы добавлены и индекс пересобран")
                _rerun()

    with tab_docs:
        st.subheader("Текущий корпус")
        for document in service.documents:
            with st.expander(f"{document.title}  |  {document.source}", expanded=False):
                st.caption(f"ID: {document.id}")
                st.write(document.text[:4000] + ("..." if len(document.text) > 4000 else ""))

    with tab_stats:
        st.subheader("Состояние проекта")
        st.json(service.stats())
        if service.index:
            st.subheader("Конфигурация индекса")
            st.json(service.index.config)
            st.write(f"Vocabulary size: {service.index.vocabulary_size}")


if __name__ == "__main__":
    main()
