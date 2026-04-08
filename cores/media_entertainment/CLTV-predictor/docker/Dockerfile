# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set metadata
LABEL maintainer="AI Core Team"
LABEL description="Docker image for AI Core Template"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UV_SYSTEM_PYTHON=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project configuration
COPY pyproject.toml .
COPY README.md .

# Install dependencies using uv
# --system flag installs into the system python, avoiding the need for a venv in the container
RUN uv pip install --system -r pyproject.toml

# Copy project files
COPY src/ ./src/
COPY configs/ ./configs/
COPY conf/ ./conf/
COPY scripts/ ./scripts/
COPY setup.py .

# Install the project
RUN uv pip install --system -e .

# Create a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Default command (can be overridden by docker-compose or k8s)
CMD ["python", "-m", "prefect", "server", "start"]
