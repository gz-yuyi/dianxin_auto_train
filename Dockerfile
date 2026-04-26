# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=hardlink \
    VIRTUAL_ENV="/app/.venv" \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && rm -rf /root/.local

COPY pyproject.toml uv.lock .python-version ./

# Use hardlinks when populating the venv to avoid doubling disk usage during installs.
RUN uv sync --frozen --no-dev --extra runtime

COPY . .

RUN mkdir -p artifacts

EXPOSE 8000

# Run with the prebuilt virtualenv directly so startup stays fully offline.
ENTRYPOINT ["/app/.venv/bin/python", "main.py"]
CMD ["api", "--host", "0.0.0.0", "--port", "8000"]
