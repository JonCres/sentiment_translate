# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set metadata
LABEL maintainer="AI Core Team"
LABEL description="Docker image for AI Core Voice of Customer"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY configs/ ./configs/
COPY conf/ ./conf/
COPY scripts/ ./scripts/
COPY pyproject.toml .
COPY setup.py .
COPY README.md .

# Install the project
RUN pip install --no-cache-dir -e .

# Create a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Default command (can be overridden by docker-compose or k8s)
CMD ["python", "-m", "prefect", "server", "start"]
