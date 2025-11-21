FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false && \
  poetry install --without dev --without train --without docs --no-interaction --no-ansi --no-root

COPY src/ ./src/
COPY data/mathe/mathe_top20_recommendations.json ./data/mathe/mathe_top20_recommendations.json

EXPOSE 8000

CMD ["uvicorn", "src.services.dataset_recs_api:app", "--host", "0.0.0.0", "--port", "8000"]