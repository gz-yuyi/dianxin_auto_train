# syntax=docker/dockerfile:1
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && rm -rf /root/.local

COPY pyproject.toml uv.lock .python-version ./

RUN uv sync --frozen --no-dev

COPY . .

RUN mkdir -p artifacts

EXPOSE 8000

CMD ["uv", "run", "python", "main.py", "api", "--host", "0.0.0.0", "--port", "8000"]
