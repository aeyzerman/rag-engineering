# Engineering RAG
## Что тут есть

- `rag_project/app.py` - веб-интерфейс.
- `rag_project/cli.py` - команды для индексации и запросов.
- `rag_project/service.py` - основной сервис RAG.
- `docs/` - базовый корпус документов.
- `rag_store/` - локальное хранилище индекса и загруженного корпуса.

## Быстрый старт

Если ты из корня репы:

```bash
pip install -e .
```

Если используешь `uv`:

```bash
uv sync
```

Потом запусти Ollama и подтяни модель:

```bash
ollama serve &
ollama pull qwen2.5:7b
```

## Запуск UI

Из корня репозитория:

```bash
streamlit run engineering/rag_project/app.py
```