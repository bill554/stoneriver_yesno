# StoneRiver Prospect Processor
# Railway Deployment Container

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY processor.py .

# Create directory for checkpoint database
RUN mkdir -p /data

# Environment defaults (override in Railway)
ENV CHECKPOINT_FILE=/data/processor_checkpoint.db
ENV BATCH_SIZE=10
ENV SLEEP_BETWEEN_PROSPECTS=2.0
ENV SLEEP_BETWEEN_BATCHES=30.0
ENV MAX_RETRIES=3
ENV LOG_LEVEL=INFO

# Run in continuous mode by default
CMD ["python", "processor.py", "--continuous", "--interval", "1800"]
