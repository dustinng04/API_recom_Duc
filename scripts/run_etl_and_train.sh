#!/bin/bash
# Run ETL job and training pipeline
# This script is designed to be run by cron

set -e  # Exit on error

cd /app

# Log file with timestamp
LOG_DIR="/var/log/etl"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/etl_and_train_${TIMESTAMP}.log"

echo "=========================================" >> "$LOG_FILE"
echo "ETL and Training Job Started: $(date)" >> "$LOG_FILE"
echo "=========================================" >> "$LOG_FILE"

# Step 1: Run ETL job
echo "[$(date)] Starting ETL job..." >> "$LOG_FILE"
python scripts/etl_job.py >> "$LOG_FILE" 2>&1
ETL_EXIT_CODE=$?

if [ $ETL_EXIT_CODE -eq 0 ]; then
    echo "[$(date)] ETL job completed successfully" >> "$LOG_FILE"
    
    # Step 2: Run training (only if ETL succeeded)
    echo "[$(date)] Starting training..." >> "$LOG_FILE"
    python scripts/train_reranker.py >> "$LOG_FILE" 2>&1
    TRAIN_EXIT_CODE=$?
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Training completed successfully" >> "$LOG_FILE"
        echo "=========================================" >> "$LOG_FILE"
        echo "ETL and Training Job Completed: $(date)" >> "$LOG_FILE"
        echo "=========================================" >> "$LOG_FILE"
        exit 0
    else
        echo "[$(date)] Training failed with exit code: $TRAIN_EXIT_CODE" >> "$LOG_FILE"
        echo "=========================================" >> "$LOG_FILE"
        echo "ETL and Training Job Failed: $(date)" >> "$LOG_FILE"
        echo "=========================================" >> "$LOG_FILE"
        exit $TRAIN_EXIT_CODE
    fi
else
    echo "[$(date)] ETL job failed with exit code: $ETL_EXIT_CODE" >> "$LOG_FILE"
    echo "[$(date)] Skipping training due to ETL failure" >> "$LOG_FILE"
    echo "=========================================" >> "$LOG_FILE"
    echo "ETL and Training Job Failed: $(date)" >> "$LOG_FILE"
    echo "=========================================" >> "$LOG_FILE"
    exit $ETL_EXIT_CODE
fi

