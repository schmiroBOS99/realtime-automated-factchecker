# Dockerfile for German Fact-Checking Service
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY app/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/app/
COPY static/ /app/static/

# Create data directory for cache
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FACTCHECK_ENABLE_CROSSENCODER=0

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
