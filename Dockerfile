FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* ./
COPY .python-version* ./
COPY README.md ./
COPY LICENSE ./

COPY llm_agents/ ./llm_agents/


RUN pip install --no-cache-dir uv
RUN uv pip install --system -e .

EXPOSE 7860

CMD ["llm-agents-web"]