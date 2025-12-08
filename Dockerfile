FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from citeconnect-backend directory (assuming it's there)
COPY citeconnect-backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire citeconnect-backend directory
COPY citeconnect-backend/ ./citeconnect-backend/

# Copy download_models.py from root (make sure this file exists)
COPY download_models.py .

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs

# Set model cache environment variables
ENV TRANSFORMERS_CACHE=/app/models
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV HF_HOME=/app/models

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

# Run application - pointing to the correct module path
CMD ["uvicorn", "citeconnect-backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info", "--access-log"]