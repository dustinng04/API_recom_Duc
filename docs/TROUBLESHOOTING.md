# Troubleshooting Guide

## NumPy Version Mismatch Error

### Error Message
```
ModuleNotFoundError: No module named 'numpy._core'
```

### Cause
This error occurs when a model was trained with NumPy 2.x but is being loaded in an environment with NumPy 1.x (or vice versa). NumPy 2.0+ changed internal structure from `numpy.core` to `numpy._core`.

### Solution

#### Option 1: Retrain Model with Correct NumPy Version (Recommended)

1. **Ensure NumPy version matches requirements.txt:**
   ```bash
   pip install numpy==1.26.4
   ```

2. **Retrain the recommender model:**
   ```bash
   # If you have data files
   python -c "from app.recommender import TutorRecommender; import json; ..."
   # Or use the /train endpoint
   ```

3. **Retrain the reranker model:**
   ```bash
   python scripts/train_reranker.py --data data/training/train_data_2025-11-07.csv
   ```

4. **Merge reranker into recommender:**
   ```bash
   python scripts/merge_reranker.py
   ```

#### Option 2: Update Requirements.txt to Match Training Environment

If you want to use NumPy 2.x (not recommended due to compatibility issues):

1. Update `requirements.txt`:
   ```txt
   numpy>=2.0.0
   ```

2. Update other packages that might not be compatible with NumPy 2.x

3. Rebuild Docker image:
   ```bash
   docker-compose build
   ```

### Prevention

Always use the same NumPy version for:
- Training models (local environment)
- Running models (Docker environment)

Check NumPy version:
```bash
python -c "import numpy; print(numpy.__version__)"
```

## Merging Reranker Model into Recommender

### Why Merge?
- Single file for all models (easier deployment)
- Avoid loading multiple files
- Better organization

### Steps

1. **Train reranker model** (if not already done):
   ```bash
   python scripts/train_reranker.py
   ```

2. **Merge reranker into recommender:**
   ```bash
   python scripts/merge_reranker.py
   ```

3. **Verify merge:**
   - Check logs: "Reranker model loaded successfully from recommender.pkl"
   - Test API: `/rerank-new` should use the model

4. **Optional: Delete reranker.pkl** (after verifying it works):
   ```bash
   rm models/reranker.pkl
   ```

### Troubleshooting Merge

If merge fails:
1. Check if both files exist:
   ```bash
   ls -la models/reranker.pkl models/recommender.pkl
   ```

2. Check file permissions

3. Check NumPy version compatibility (see above)

## Docker Build Issues

### NumPy Version Mismatch in Docker

1. **Rebuild Docker image:**
   ```bash
   docker-compose build --no-cache
   ```

2. **Ensure requirements.txt has correct NumPy version:**
   ```txt
   numpy==1.26.4
   ```

3. **Retrain models with matching NumPy version** (see above)

### Model Not Loading in Docker

1. **Check model files are copied:**
   ```bash
   docker-compose exec api ls -la /app/models/
   ```

2. **Check logs:**
   ```bash
   docker-compose logs api
   ```

3. **Verify NumPy version in container:**
   ```bash
   docker-compose exec api python -c "import numpy; print(numpy.__version__)"
   ```

## API Endpoints

### /rerank-new Not Using Model

If `/rerank-new` falls back to weighted combination:

1. **Check if reranker model is loaded:**
   - Check startup logs: "Reranker model loaded successfully"
   - Check if `recommender.reranker_model` is not None

2. **If model not loaded:**
   - Run `python scripts/merge_reranker.py` to merge
   - Restart API

3. **Check model file exists:**
   ```bash
   ls -la models/recommender.pkl
   ```

## Common Issues

### Model File Too Large

If model file is very large:
- Consider using model compression
- Remove unnecessary data from model (e.g., interactions_df if not needed)
- Use model quantization (if supported)

### Training Data Issues

If training fails:
1. Check data format:
   ```bash
   head data/training/train_data_*.csv
   ```

2. Check required columns: `os_score`, `rerank_score`, `price`, `rating`, `position`, `label`

3. Check data quality:
   ```bash
   python -c "import pandas as pd; df = pd.read_csv('data/training/train_data_2025-11-07.csv'); print(df.describe())"
   ```

## Getting Help

If you encounter issues not covered here:
1. Check logs: `docker-compose logs api`
2. Check NumPy version compatibility
3. Verify model files exist and are not corrupted
4. Check training data format and quality

