FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    bash \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (required for pnpm and building Gradio frontend)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install pnpm globally
RUN npm install -g pnpm

# Install uv first (before copying files)
RUN pip install --no-cache-dir uv

# Copy all necessary files
COPY pyproject.toml uv.lock* ./
COPY .python-version* ./
COPY README.md ./
COPY LICENSE ./
COPY llm_agents/ ./llm_agents/
COPY scripts/ ./scripts/
COPY graphs/ ./graphs/

RUN mkdir ./gradio
RUN git clone https://github.com/gradio-app/gradio.git gradio/

# Run the gradio preparation script (build frontend only, submodule already initialized)
RUN bash scripts/prepare_gradio.sh

# Use uv sync to install dependencies
RUN uv sync

EXPOSE 7860

# Set entrypoint
# Download and initialize the NeurIPS 2025 conference knowledge graph
RUN chmod +x /app/scripts/docker-entrypoint.sh
RUN chmod +x /app/scripts/import_neurips2025_kg.sh
ENTRYPOINT ["scripts/docker-entrypoint.sh"]

CMD ["uv", "run", "llm_agents/frontend/browser_ui.py"]
