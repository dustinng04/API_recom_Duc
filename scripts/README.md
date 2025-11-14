# Scripts Documentation

## merge_reranker.py

Merge reranker.pkl into recommender.pkl to have all models in one file.

### Usage

```bash
# Merge reranker.pkl into recommender.pkl (creates backup by default)
python scripts/merge_reranker.py

# Merge without creating backup
python scripts/merge_reranker.py --no-backup

# Specify custom paths
python scripts/merge_reranker.py --reranker models/reranker.pkl --recommender models/recommender.pkl
```

### What it does

1. Loads reranker.pkl
2. Loads recommender.pkl
3. Merges reranker model into recommender.pkl
4. Creates backup of recommender.pkl (optional)
5. Saves updated recommender.pkl

After merging, you can delete reranker.pkl as it's now included in recommender.pkl.

## train_reranker.py

Train LightGBMRanker model for re-ranking.

### Usage

```bash
# Train with latest training data
python scripts/train_reranker.py

# Train with specific data file
python scripts/train_reranker.py --data data/training/train_data_2025-11-07.csv

# Custom test size
python scripts/train_reranker.py --test-size 0.3
```

### What it does

1. Loads training data from CSV
2. Prepares features (os_score, rerank_score, price, rating, position)
3. Trains LightGBMRanker model
4. Saves model to models/reranker.pkl
5. Updates recommender.pkl with reranker model (if exists)

## etl_job.py

ETL job to collect training data from search logs and interaction logs.

### Usage

```bash
# Process data from 3 days ago (default: ETL_DAYS_LOOKBACK=3, ETL_NUM_DAYS=1)
python scripts/etl_job.py

# Process specific date
ETL_DATE=2025-11-08 python scripts/etl_job.py

# Process multiple days (e.g., 3 days)
ETL_NUM_DAYS=3 python scripts/etl_job.py

# Custom lookback days
ETL_DAYS_LOOKBACK=5 ETL_NUM_DAYS=2 python scripts/etl_job.py
```

### Environment Variables

- `ETL_DATE`: Base date in format `YYYY.MM.DD` or `YYYY-MM-DD` (default: today)
- `ETL_NUM_DAYS`: Number of days to process (default: 1)
- `ETL_DAYS_LOOKBACK`: Days to look back from base date (default: 3)
- `OS_HOST`: OpenSearch host URL
- `OS_USERNAME`: OpenSearch username
- `OS_PASSWORD`: OpenSearch password

### What it does

1. Extracts search logs from OpenSearch (uses `@timestamp` field)
2. Extracts interaction logs from OpenSearch (uses `timestamp` field)
3. Loads tutors data from `data/tutors_adjust.json`
4. Transforms and merges data
5. Outputs training data as CSV: `data/training/train_data_YYYY.MM.DD.csv`
6. Pushes training data to OpenSearch index: `train-data-raw-YYYY.MM.DD` (for monitoring)

### Output

- **CSV file**: `data/training/train_data_YYYY.MM.DD.csv` (primary storage)
- **OpenSearch index**: `train-data-raw-YYYY.MM.DD` (for monitoring/analytics)

## run_etl_and_train.sh

Shell script to run ETL job and training pipeline sequentially.

### Usage

```bash
# Run manually
bash scripts/run_etl_and_train.sh

# Or make it executable and run
chmod +x scripts/run_etl_and_train.sh
./scripts/run_etl_and_train.sh
```

### What it does

1. Runs ETL job (`scripts/etl_job.py`)
2. If ETL succeeds, runs training (`scripts/train_reranker.py`)
3. Logs everything to `/var/log/etl/etl_and_train_YYYYMMDD_HHMMSS.log`

### Logs

Logs are saved to `/var/log/etl/` with timestamp in filename:
- `etl_and_train_20251108_020000.log`

## Automated Scheduling (Cronjob)

The ETL and training pipeline can be automated using cron jobs in Docker.

### Setup

The cronjob is automatically configured when building the Docker image:

1. **Crontab file**: `scripts/crontab`
   - Runs daily at 2:00 AM
   - Executes `scripts/run_etl_and_train.sh`

2. **Dockerfile**: Automatically installs cron and sets up the cronjob

3. **Startup**: When container starts, both cron daemon and uvicorn server run

### Manual Cron Setup (if needed)

```bash
# Inside Docker container
crontab -e

# Add this line:
0 2 * * * root bash -c "cd /app && /app/scripts/run_etl_and_train.sh"
```

### Viewing Cron Logs

```bash
# View cron logs
docker exec -it tutor-recommendation-api tail -f /var/log/etl/etl_and_train_*.log

# List all log files
docker exec -it tutor-recommendation-api ls -lh /var/log/etl/
```

### Testing Cronjob

```bash
# Test the script manually
docker exec -it tutor-recommendation-api /app/scripts/run_etl_and_train.sh

# Check if cron is running
docker exec -it tutor-recommendation-api ps aux | grep cron

# View cron logs
docker exec -it tutor-recommendation-api tail -f /var/log/syslog | grep CRON
```

### Timezone Configuration

The Docker container is configured to use **Asia/Ho_Chi_Minh (UTC+7)** timezone. This means:
- Cron jobs run at 2:00 AM **Vietnam time** (not UTC)
- All timestamps in logs and model files use Vietnam timezone

To change timezone, modify the `TZ` environment variable in `Dockerfile`.

### Model Backup

When training completes, the system automatically:
1. **Backs up existing model** to `models/reranker_backup_YYYYMMDD_HHMMSS.pkl` before overwriting
2. **Saves new model** to `models/reranker.pkl`
3. **Updates recommender.pkl** with the new reranker model

This ensures you can rollback to previous models if needed.

### Train Immediately (Manual Trigger)

To trigger training immediately without waiting for cron:

```bash
# Option 1: Run training script directly
docker exec -it tutor-recommendation-api python scripts/train_reranker.py

# Option 2: Use the convenience script
docker exec -it tutor-recommendation-api /app/scripts/train_now.sh

# Option 3: Run full ETL + Training pipeline
docker exec -it tutor-recommendation-api /app/scripts/run_etl_and_train.sh
```

**Note**: Training uses the **latest** training data file in `data/training/` by default. To use a specific file:

```bash
docker exec -it tutor-recommendation-api python scripts/train_reranker.py --data data/training/train_data_2025.11.09.csv
```
