FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false && \
  poetry install --without dev --without docs --no-interaction --no-ansi --no-root

COPY dataset_recsys/ ./dataset_recsys/
COPY recs_metrics/ ./recs_metrics/

EXPOSE 8000

CMD ["uvicorn", "dataset_recsys.api.main:app", "--host", "0.0.0.0", "--port", "8000"]