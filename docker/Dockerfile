FROM python:3.11-slim

WORKDIR /app
COPY lore_memory/ ./lore_memory/
COPY pyproject.toml ./

RUN pip install --no-cache-dir .

# No external dependencies — pure Python + SQLite (built into Python)
# No LLM required — hash embeddings for local, pluggable for production

ENV LORE_DATA_DIR=/data
ENV LORE_EMBEDDING_DIMS=64

VOLUME /data

HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "from lore_memory import Memory; print('ok')" || exit 1

EXPOSE 8000

CMD ["python", "-m", "lore_memory.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
