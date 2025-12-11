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

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the contents of citeconnect-backend directly to /app (not as subdirectory)
COPY citeconnect-backend/ .

# Ensure bias config directories exist (configs are copied from CI artifacts via deploy.yml)
# If configs don't exist, service will log warnings but continue to operate
RUN mkdir -p /app/bias_config && \
    echo "Bias mitigation configs will be loaded from:" && \
    echo "  - /app/bias_config/bias_mitigation_config.json (user-profile mitigation)" && \
    echo "  - /app/fairness_config.json (domain fairness)" && \
    (ls -la /app/bias_config/ 2>/dev/null || echo "  ℹ️  bias_config/ empty (will use runtime defaults)") && \
    (ls -la /app/fairness_config.json 2>/dev/null || echo "  ℹ️  fairness_config.json missing (will use runtime defaults)")

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