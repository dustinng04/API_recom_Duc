#!/bin/bash
# Quick script to trigger training immediately
# This script runs training with the latest training data

set -e  # Exit on error

cd /app

echo "========================================="
echo "Training Job Started: $(date)"
echo "========================================="

# Run training with latest data
echo "[$(date)] Starting training with latest data..."
python scripts/train_reranker.py

if [ $? -eq 0 ]; then
    echo "[$(date)] Training completed successfully!"
    echo "========================================="
    echo "Training Job Completed: $(date)"
    echo "========================================="
    exit 0
else
    echo "[$(date)] Training failed!"
    echo "========================================="
    echo "Training Job Failed: $(date)"
    echo "========================================="
    exit 1
fi

