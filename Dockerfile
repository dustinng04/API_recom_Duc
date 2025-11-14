FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
# libgomp is required for LightGBM
# cron is required for scheduled ETL jobs
# tzdata is required for timezone configuration
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    cron \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone to Asia/Ho_Chi_Minh (UTC+7)
# This affects cron schedule times
ENV TZ=Asia/Ho_Chi_Minh
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/models /app/data /app/app /app/scripts /var/log/etl

# Copy application code
COPY app/ ./app/
COPY data/ ./data/
COPY scripts/ ./scripts/

# Setup cron jobs
COPY scripts/crontab /etc/cron.d/etl-job
RUN chmod 0644 /etc/cron.d/etl-job
RUN crontab /etc/cron.d/etl-job

# Make scripts executable
RUN chmod +x /app/scripts/run_etl_and_train.sh
RUN chmod +x /app/scripts/train_now.sh

# Create log directory with proper permissions
RUN chmod 755 /var/log/etl

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
  CMD curl -f http://localhost:8001/health || exit 1

# Create startup script to run both cron and uvicorn
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Start cron daemon in background\n\
# Note: cron needs to run as root, but uvicorn will run as the default user\n\
if [ -f /etc/cron.d/etl-job ]; then\n\
    echo "Starting cron daemon..."\n\
    cron\n\
    echo "Cron daemon started"\n\
else\n\
    echo "Warning: crontab file not found"\n\
fi\n\
\n\
# Run uvicorn in foreground (this keeps container alive)\n\
echo "Starting uvicorn server..."\n\
exec uvicorn app.main:app --host 0.0.0.0 --port 8001\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run both cron and application
CMD ["/app/start.sh"]
