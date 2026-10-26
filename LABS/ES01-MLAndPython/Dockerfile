FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml .
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
RUN pip install --no-cache-dir ".[dev]"

RUN mkdir -p /app/models

EXPOSE 8000 8501
