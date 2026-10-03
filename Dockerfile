FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/var/lib/rag-phy/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/var/lib/rag-phy/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system ragphy \
    && useradd --system --gid ragphy --home-dir /var/lib/rag-phy ragphy \
    && mkdir -p /var/lib/rag-phy/huggingface /app/data/chroma /app/reports/traces/design-runs \
    && chown -R ragphy:ragphy /var/lib/rag-phy /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY data/curated ./data/curated
COPY data/evaluation ./data/evaluation
COPY scripts ./scripts

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir '.[api,knowledge,embeddings]'

USER ragphy

EXPOSE 8000

CMD ["sh", "-c", "python scripts/ensure_curated_index.py && exec uvicorn rag_phy.api.main:app --host 0.0.0.0 --port 8000"]
