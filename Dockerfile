FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    bash \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv first (before copying files)
RUN pip install --no-cache-dir uv

# Copy all necessary files
COPY pyproject.toml uv.lock* ./
COPY .python-version* ./
COPY README.md ./
COPY LICENSE ./
COPY agentic_nav/ ./agentic_nav/
COPY scripts/ ./scripts/
COPY graphs/ ./graphs/

# Use uv sync to install dependencies
RUN uv sync

EXPOSE 7860

# Set entrypoint
# Download and initialize the NeurIPS 2025 conference knowledge graph
RUN chmod +x /app/scripts/docker-entrypoint.sh
RUN chmod +x /app/scripts/import_neurips2025_kg.sh
ENTRYPOINT ["scripts/docker-entrypoint.sh"]

CMD ["uv", "run", "agentic_nav/frontend/browser_ui.py"]
