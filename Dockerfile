FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from citeconnect-backend directory
COPY citeconnect-backend/requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    torch>=1.9.0 \
    huggingface_hub==0.16.4 \
    transformers==4.21.3 \
    tokenizers==0.12.1 \
    sentence-transformers==2.2.2

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the contents of citeconnect-backend directly to /app (not as subdirectory)
COPY citeconnect-backend/ .

# Set model cache environment variables
ENV TRANSFORMERS_CACHE=/app/models
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV HF_HOME=/app/models

# Copy download_models.py from root
COPY download_models.py .

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs

# Run the download script
RUN python3 download_models.py && rm download_models.py

# Verify models directory
RUN ls -la /app/models/ && du -sh /app/models/

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Now the module path is simply app.main:app (not citeconnect-backend.app.main:app)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info", "--access-log"]