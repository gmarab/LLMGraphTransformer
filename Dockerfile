FROM python:3.13-slim

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY server.py agent_falkor_load.py agent_falkor_qa.py agent_load.py agent_qa.py ./
COPY src/ ./src/

EXPOSE 8000

CMD ["uv", "run", "server.py"]
