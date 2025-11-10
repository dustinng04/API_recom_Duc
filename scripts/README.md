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
# Process today's data (default: 2025-11-07)
python scripts/etl_job.py

# Process specific date
ETL_DATE=2025-11-08 python scripts/etl_job.py
```

### What it does

1. Extracts search logs from OpenSearch
2. Loads interaction logs from local file
3. Loads tutors data
4. Transforms and merges data
5. Outputs training data as CSV
