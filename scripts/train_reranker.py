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
from pathlib import Path
from typing import Optional
import argparse

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
FEATURE_COLUMNS = ['rerank_score', 'price', 'rating', 'position']
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
    
    # Extract features
    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df[LABEL_COLUMN].values.astype(np.int32)
    
    # Group by query for learning-to-rank
    # Each query represents a group (list of candidates)
    groups = df.groupby(GROUP_COLUMN).size().values.astype(np.int32)
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Labels: {y.sum()} positive, {(y == 0).sum()} negative")
    logger.info(f"Number of groups (queries): {len(groups)}")
    logger.info(f"Average group size: {groups.mean():.2f}")
    
    return X, y, groups


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2
) -> lgb.Booster:
    """
    Train LightGBMRanker model.
    
    Args:
        X: Feature matrix
        y: Label vector
        groups: Group sizes
        test_size: Fraction of data to use for validation
    
    Returns:
        Trained LightGBM model
    """
    logger.info("Training LightGBMRanker model...")
    
    # Split data into train and validation
    # We need to split by groups (queries), not individual rows
    n_groups = len(groups)
    n_train_groups = int(n_groups * (1 - test_size))
    
    # Calculate indices for train/val split
    cumsum_groups = np.cumsum(groups)
    train_end_idx = cumsum_groups[n_train_groups - 1]
    
    X_train = X[:train_end_idx]
    y_train = y[:train_end_idx]
    groups_train = groups[:n_train_groups]
    
    X_val = X[train_end_idx:]
    y_val = y[train_end_idx:]
    groups_val = groups[n_train_groups:]
    
    logger.info(f"Train: {len(X_train)} samples in {len(groups_train)} groups")
    logger.info(f"Validation: {len(X_val)} samples in {len(groups_val)} groups")
    
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
    
    # Create datasets
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        group=groups_train
    )
    
    val_data = lgb.Dataset(
        X_val,
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
    
    return model


def save_model(model: lgb.Booster, model_path: Path, update_recommender: bool = True):
    """
    Save trained model to disk.
    If update_recommender is True, also update the main recommender model file.
    """
    logger.info(f"Saving model to {model_path}")
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save reranker model separately
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_columns': FEATURE_COLUMNS,
            'label_column': LABEL_COLUMN,
            'group_column': GROUP_COLUMN
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
                
                # Add reranker model to recommender data
                recommender_data['reranker_model'] = model
                recommender_data['reranker_metadata'] = {
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
        
        # Train model
        model = train_model(X, y, groups, test_size=args.test_size)
        
        # Save model
        save_model(model, MODEL_PATH)
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {MODEL_PATH}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

