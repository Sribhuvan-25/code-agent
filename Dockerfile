# Multi-stage build for production optimization
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and add to docker group
RUN groupadd -g 999 docker || true
RUN useradd --create-home --shell /bin/bash -u 1000 -g docker app

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements-working.txt requirements.txt
RUN pip install -r requirements.txt

# Copy application code
COPY app/ ./app/

# Change ownership to app user
RUN chown -R app:docker /app

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 