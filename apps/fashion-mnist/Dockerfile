# apps/fashion-mnist/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for CodeCarbon (sometimes needed for CPU tracking)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ src/

# Create directories for data and metrics so they can be mounted
RUN mkdir -p data/raw data/processed metrics

# Default command (can be overridden)
CMD ["python", "src/train.py"]