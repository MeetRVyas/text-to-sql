# Dockerfile
# Containerizes the FastAPI serving layer (api/main.py) for self-hosting.

FROM python:3.12-slim

WORKDIR /app

# build-essential: some ML wheels (e.g. bitsandbytes) need it at install time
# even when using prebuilt wheels for their own compiled extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the modules api.main actually imports at runtime — keeps the image
# free of training/evaluation-only code and its Spider-download tooling.
COPY api/ ./api/
COPY data/__init__.py ./data/__init__.py
COPY data/schema_extractor.py ./data/schema_extractor.py
COPY data/prompt_format.py ./data/prompt_format.py
COPY sample_db/ ./sample_db/

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENV MODEL_PATH=checkpoints/qlora-text2sql/final-adapter \
    BASE_MODEL=Qwen/Qwen2.5-Coder-3B-Instruct \
    DB_ROOT=data/spider/database \
    TABLES_JSON=data/spider/tables.json \
    MAX_NEW_TOKENS=256 \
    DEVICE=auto

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]