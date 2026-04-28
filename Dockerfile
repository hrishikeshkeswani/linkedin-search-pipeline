FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for playwright + faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY linkedin_search/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install playwright browsers (needed for scraper)
RUN playwright install chromium --with-deps

# Copy application code
COPY linkedin_search/ ./linkedin_search/

WORKDIR /app/linkedin_search

# Pre-download the embedding model at build time so startup is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
