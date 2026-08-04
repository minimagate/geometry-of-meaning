# ---------------------------------------------------------------------------
# Geometry of Meaning — Research Environment
#
# Freezes the full Python environment for reproducible experiments.
# Build:  docker build -t geometry-of-meaning .
# Run:    docker run --rm -it -v $(pwd)/runs:/app/runs geometry-of-meaning
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# System dependencies for sentence-transformers / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install -e ".[all]"

# Copy the rest of the project
COPY . .

# Pre-download a default embedding model (avoids download at first run)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Default command: print help
CMD ["python", "-c", "print('Geometry of Meaning ready. Mount runs/ and execute experiment scripts.')"]