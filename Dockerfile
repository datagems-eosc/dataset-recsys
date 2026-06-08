FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-writer \
    libreoffice-impress \
    libreoffice-java-common \
    fonts-dejavu \
    fonts-liberation \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false && \
  poetry install --only main --no-interaction --no-ansi --no-root

COPY dataset_recsys/ ./dataset_recsys/
COPY recs_metrics/ ./recs_metrics/

EXPOSE 8000

CMD ["uvicorn", "dataset_recsys.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
