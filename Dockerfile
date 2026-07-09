FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create runtime directories
RUN mkdir -p data/raw/wikipedia data/raw/huggingface data/raw/paperswithcode \
             data/processed data/chromadb database logs reports experiments

# Environment
ENV PYTHONPATH=/app
ENV RAGOPS_ENV=production
ENV LOG_LEVEL=INFO

EXPOSE 8501

# Default: run scheduler (probe cycles + analysis + reports)
CMD ["python", "scheduler/main_scheduler.py"]
