# Build on official lightweight python slim image
FROM python:3.9-slim

# Configure runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install base system compiling tools and validation helpers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first to optimize docker caching layer
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source directories and startup scripts
COPY src/ ./src/
COPY app.py .
COPY README.md .

# Initialize mount directories for persistent vector DB files and document storage
RUN mkdir -p data/chromadb data/embeddings_cache docs

# Expose standard Streamlit web port
EXPOSE 8501

# Healthcheck to verify the web interface responds to connection queries
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
