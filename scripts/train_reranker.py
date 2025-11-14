#!/usr/bin/env python3
"""
Train LightGBMRanker model for re-ranking tutors.

This script:
1. Loads training data from CSV
2. Prepares features and labels
3. Trains LightGBMRanker model
4. Saves model to models/reranker.pkl
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple
import argparse
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
TRAINING_DATA_DIR = Path('data/training')
MODEL_DIR = Path('models')
MODEL_PATH = MODEL_DIR / 'reranker.pkl'

# Features to use for training
FEATURE_COLUMNS = ['os_score', 'rerank_score', 'price', 'rating', 'position']
LABEL_COLUMN = 'label'
GROUP_COLUMN = 'query'  # Group by query for learning-to-rank


def load_training_data(data_path: Path) -> pd.DataFrame:
    """Load training data from CSV file."""
    logger.info(f"Loading training data from {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Check required columns
    required_cols = FEATURE_COLUMNS + [LABEL_COLUMN, GROUP_COLUMN]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features and labels for training.
    
    Returns:
        X: Feature matrix
        y: Label vector
        groups: Group sizes for learning-to-rank
    """
    logger.info("Preparing features...")
    
    initial_rows = len(df)
    logger.info(f"Initial data rows: {initial_rows}")
    
    # Remove rows with missing (NaN) query only
    # Empty string query is valid (could be default search or browse mode)
    missing_query_before = df[GROUP_COLUMN].isna().sum()
    if missing_query_before > 0:
        logger.warning(f"Found {missing_query_before} rows with missing (NaN) query. Removing them...")
        df = df.dropna(subset=[GROUP_COLUMN]).copy()
        logger.info(f"Removed {missing_query_before} rows with missing query. Remaining rows: {len(df)}")
    
    # Log empty query count (for information, but we keep them)
    empty_query_count = (df[GROUP_COLUMN] == '').sum()
    if empty_query_count > 0:
        logger.info(f"Found {empty_query_count} rows with empty query (will be grouped together)")
    
    # Handle duplicates: same (query, tutorId) pair may appear multiple times
    # Strategy: Keep the row with label=1 if any, otherwise keep the first one
    # This ensures we don't lose positive examples
    duplicates_before = df.duplicated(subset=[GROUP_COLUMN, 'tutorId']).sum()
    if duplicates_before > 0:
        logger.warning(f"Found {duplicates_before} duplicate (query, tutorId) pairs. Removing duplicates...")
        
        # Sort by label descending (1 before 0) to prioritize positive examples
        df = df.sort_values(by='label', ascending=False).reset_index(drop=True)
        
        # Remove duplicates, keeping first (which will be label=1 if available)
        df = df.drop_duplicates(subset=[GROUP_COLUMN, 'tutorId'], keep='first').reset_index(drop=True)
        
        duplicates_after = df.duplicated(subset=[GROUP_COLUMN, 'tutorId']).sum()
        rows_removed = initial_rows - len(df)
        logger.info(f"Removed {rows_removed} duplicate rows. Remaining rows: {len(df)}")
        logger.info(f"Remaining duplicates: {duplicates_after} (should be 0)")
    
    # IMPORTANT: Sort by query to ensure groups are contiguous
    # This is required for proper train/val split by groups
    df_sorted = df.sort_values(by=GROUP_COLUMN).reset_index(drop=True)
    
    # Extract features
    X = df_sorted[FEATURE_COLUMNS].values.astype(np.float32)
    y = df_sorted[LABEL_COLUMN].values.astype(np.int32)
    
    # Group by query for learning-to-rank
    # Each query represents a group (list of candidates)
    # After sorting, groups will be contiguous in the data
    groups = df_sorted.groupby(GROUP_COLUMN).size().values.astype(np.int32)
    
    # Verify that sum of groups equals total data size
    total_group_size = groups.sum()
    if total_group_size != len(X):
        raise ValueError(
            f"Group sizes don't match data size: {total_group_size} != {len(X)}. "
            f"This may indicate data inconsistency after duplicate removal."
        )
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Labels: {y.sum()} positive, {(y == 0).sum()} negative")
    logger.info(f"Number of groups (queries): {len(groups)}")
    logger.info(f"Average group size: {groups.mean():.2f}")
    logger.info(f"Total group size sum: {total_group_size} (matches data size: {len(X)})")
    
    return X, y, groups


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2
) -> Tuple[lgb.Booster, StandardScaler]:
    """
    Train LightGBMRanker model with feature scaling.
    
    Args:
        X: Feature matrix
        y: Label vector
        groups: Group sizes
        test_size: Fraction of data to use for validation
    
    Returns:
        Tuple of (Trained LightGBM model, StandardScaler fitted on training data)
    """
    logger.info("Training LightGBMRanker model...")
    
    # Split data into train and validation
    # We need to split by groups (queries), not individual rows
    n_groups = len(groups)
    n_train_groups = int(n_groups * (1 - test_size))
    
    # Ensure we have at least 1 group in each set
    if n_train_groups == 0:
        n_train_groups = 1
    if n_train_groups >= n_groups:
        n_train_groups = n_groups - 1
    
    # Calculate indices for train/val split
    cumsum_groups = np.cumsum(groups)
    train_end_idx = cumsum_groups[n_train_groups - 1]
    
    X_train = X[:train_end_idx]
    y_train = y[:train_end_idx]
    groups_train = groups[:n_train_groups].copy()
    
    X_val = X[train_end_idx:]
    y_val = y[train_end_idx:]
    groups_val = groups[n_train_groups:].copy()
    
    # Verify group sizes match data sizes
    train_group_sum = groups_train.sum()
    val_group_sum = groups_val.sum()
    
    logger.info(f"Train: {len(X_train)} samples in {len(groups_train)} groups (sum: {train_group_sum})")
    logger.info(f"Validation: {len(X_val)} samples in {len(groups_val)} groups (sum: {val_group_sum})")
    
    # Validate group sizes match data sizes
    if train_group_sum != len(X_train):
        logger.error(f"Train group sum mismatch: {train_group_sum} != {len(X_train)}")
        raise ValueError(f"Train group sizes don't match data size: {train_group_sum} != {len(X_train)}")
    
    if val_group_sum != len(X_val):
        logger.error(f"Validation group sum mismatch: {val_group_sum} != {len(X_val)}")
        raise ValueError(f"Validation group sizes don't match data size: {val_group_sum} != {len(X_val)}")
    
    # Fit scaler on training data only
    logger.info("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    logger.info("Feature scaling completed. Mean: {}, Std: {}".format(
        scaler.mean_.round(4), scaler.scale_.round(4)
    ))
    
    # LightGBM parameters for ranking
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5, 10],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 1,
        'num_threads': -1,
    }
    
    # Create datasets with scaled features
    train_data = lgb.Dataset(
        X_train_scaled,
        label=y_train,
        group=groups_train
    )
    
    val_data = lgb.Dataset(
        X_val_scaled,
        label=y_val,
        group=groups_val,
        reference=train_data
    )
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=10)
        ]
    )
    
    logger.info("Model training completed!")
    
    return model, scaler


def save_model(
    model: lgb.Booster,
    scaler: StandardScaler,
    model_path: Path,
    update_recommender: bool = True,
    backup_existing: bool = True
):
    """
    Save trained model and scaler to disk.
    If update_recommender is True, also update the main recommender model file.
    If backup_existing is True, creates a backup of existing model before overwriting.
    """
    logger.info(f"Saving model to {model_path}")
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Backup existing model if it exists
    if backup_existing and model_path.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = model_path.parent / f"{model_path.stem}_backup_{timestamp}.pkl"
        shutil.copy2(model_path, backup_path)
        logger.info(f"Backed up existing model to {backup_path}")
    
    # Save reranker model separately
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_columns': FEATURE_COLUMNS,
            'label_column': LABEL_COLUMN,
            'group_column': GROUP_COLUMN,
            'trained_at': datetime.now().isoformat()
        }, f)
    
    logger.info(f"Reranker model saved successfully to {model_path}")
    
    # Also update the main recommender model if it exists
    if update_recommender:
        recommender_path = MODEL_DIR / 'recommender.pkl'
        if recommender_path.exists():
            try:
                logger.info("Updating main recommender model with reranker...")
                with open(recommender_path, 'rb') as f:
                    recommender_data = pickle.load(f)
                
                # Add reranker model and scaler to recommender data
                recommender_data['reranker_model'] = model
                recommender_data['reranker_metadata'] = {
                    'scaler': scaler,
                    'feature_columns': FEATURE_COLUMNS,
                    'label_column': LABEL_COLUMN,
                    'group_column': GROUP_COLUMN
                }
                
                # Save updated recommender model
                with open(recommender_path, 'wb') as f:
                    pickle.dump(recommender_data, f)
                
                logger.info("Main recommender model updated with reranker successfully")
            except Exception as e:
                logger.warning(f"Failed to update main recommender model: {e}")
                logger.info("Reranker model saved separately. It will be loaded on next startup.")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train LightGBMRanker model')
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to training data CSV file. If not specified, uses latest file in data/training/'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data to use for validation (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    # Determine data path
    if args.data:
        data_path = Path(args.data)
    else:
        # Find latest training data file
        training_files = sorted(TRAINING_DATA_DIR.glob('train_data_*.csv'))
        if not training_files:
            raise FileNotFoundError(f"No training data files found in {TRAINING_DATA_DIR}")
        data_path = training_files[-1]
        logger.info(f"Using latest training data: {data_path}")
    
    try:
        # Load data
        df = load_training_data(data_path)
        
        # Prepare features
        X, y, groups = prepare_features(df)
        
        # Train model (returns model and scaler)
        model, scaler = train_model(X, y, groups, test_size=args.test_size)
        
        # Save model and scaler
        save_model(model, scaler, MODEL_PATH)
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {MODEL_PATH}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

