# ETL Job Scripts

## etl_job.py

ETL job for collecting training data from search logs and interaction logs.

## Installation

### 1. Install Dependencies

```bash
# From project root directory
pip install -r requirements.txt
```

Or install only ETL dependencies:
```bash
pip install pandas opensearch-py
```

### 2. Verify Installation

```bash
python -c "import pandas; import opensearchpy; print('All dependencies installed!')"
```

## Usage

### Basic Usage

```bash
# Set environment variables
export OS_HOST="https://your-opensearch-host.com"
export OS_USERNAME="your-username"
export OS_PASSWORD="your-password"

# Run ETL job for today's data
python scripts/etl_job.py

# Or specify a date
export ETL_DATE="2025-11-07"
python scripts/etl_job.py
```

### Environment Variables

- `OS_HOST`: OpenSearch host URL (https://... or http://...)
- `OS_USERNAME`: OpenSearch username
- `OS_PASSWORD`: OpenSearch password
- `ETL_DATE`: Date to process (format: YYYY-MM-DD). Defaults to today.

### Testing Without OpenSearch

For testing purposes, you can create a mock version or test with local data only:

```bash
# Test with local interaction logs and tutors data only
# (Requires modifying script to skip OpenSearch connection)
```

## Output

Training data CSV file saved to `data/training/train_data_YYYYMMDD.csv`

### Format

CSV columns:
- `userId`: User identifier
- `query`: Search query text
- `tutorId`: Tutor identifier
- `rerank_score`: Rerank score from search results
- `price`: Tutor price
- `rating`: Tutor rating
- `position`: Position in search results (1, 2, 3, ...)
- `label`: 1 if positive interaction (click/conversion/join/rating/wishlist), 0 otherwise

### Example Output

```csv
userId,query,tutorId,rerank_score,price,rating,position,label
a1f9429e-4364-44d3-8208-d1f5e0ac6739,"gia sư Tiếng Anh dạy Nâng cao",322,0.6426135,250000.0,4.5,1,1
a1f9429e-4364-44d3-8208-d1f5e0ac6739,"gia sư Tiếng Anh dạy Nâng cao",302,0.6266436,280000.0,4.2,2,0
```

## Notes

- Positive event types: `click`, `conversion`, `join`, `rating`, `wishlist`
- Tutors not found in `tutors_adjust.json` are skipped
- Interactions are aggregated by `sessionId` and `tutorId`
- Script must be run from project root directory (uses relative paths)

## train_reranker.py

Training script for LightGBMRanker model.

### Usage

```bash
# Train with latest training data
python scripts/train_reranker.py

# Train with specific data file
python scripts/train_reranker.py --data data/training/train_data_2025-11-07.csv

# Specify validation split
python scripts/train_reranker.py --test-size 0.2
```

### Output

Trained model saved to `models/reranker.pkl`

### Model Details

- Algorithm: LightGBMRanker (LambdaRank)
- Features: rerank_score, price, rating, position
- Objective: lambdarank
- Metric: NDCG@[1, 3, 5, 10]

## Troubleshooting

### Connection Issues

If you get connection errors:
1. Verify OpenSearch credentials are correct
2. Check network connectivity to OpenSearch host
3. Verify SSL certificates if using HTTPS

### Missing Data

If no data is extracted:
1. Check if search logs exist for the specified date
2. Verify index pattern `search-logs-*` is correct
3. Check OpenSearch query syntax

### Missing Tutors

If many tutors are skipped:
1. Verify `tutors_adjust.json` is up to date
2. Check tutorId format matches between search logs and tutors data

### Training Issues

If training fails:
1. Check training data format (required columns: rerank_score, price, rating, position, label, query)
2. Verify positive/negative label balance
3. Check if there are enough samples per query group
