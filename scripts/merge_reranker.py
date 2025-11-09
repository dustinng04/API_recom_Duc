#!/usr/bin/env python3
"""
Script to merge reranker.pkl into recommender.pkl.

This script:
1. Loads reranker.pkl
2. Loads recommender.pkl
3. Merges reranker model into recommender.pkl
4. Saves updated recommender.pkl
"""

import pickle
import logging
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path('models')
RERANKER_PATH = MODEL_DIR / 'reranker.pkl'
RECOMMENDER_PATH = MODEL_DIR / 'recommender.pkl'


def merge_reranker_into_recommender(
    reranker_path: Path = RERANKER_PATH,
    recommender_path: Path = RECOMMENDER_PATH,
    backup: bool = True
) -> bool:
    """
    Merge reranker model into recommender.pkl.
    
    Args:
        reranker_path: Path to reranker.pkl
        recommender_path: Path to recommender.pkl
        backup: Whether to create backup of recommender.pkl
    
    Returns:
        True if successful, False otherwise
    """
    # Check if files exist
    if not reranker_path.exists():
        logger.error(f"Reranker model not found: {reranker_path}")
        return False
    
    if not recommender_path.exists():
        logger.error(f"Recommender model not found: {recommender_path}")
        return False
    
    try:
        # Load reranker model
        logger.info(f"Loading reranker model from {reranker_path}")
        with open(reranker_path, 'rb') as f:
            reranker_data = pickle.load(f)
        
        reranker_model = reranker_data.get('model')
        reranker_metadata = {
            'feature_columns': reranker_data.get('feature_columns'),
            'label_column': reranker_data.get('label_column'),
            'group_column': reranker_data.get('group_column')
        }
        
        if reranker_model is None:
            logger.error("Reranker model not found in reranker.pkl")
            return False
        
        logger.info("Reranker model loaded successfully")
        
        # Create backup if requested
        if backup:
            backup_path = recommender_path.with_suffix('.pkl.backup')
            logger.info(f"Creating backup: {backup_path}")
            import shutil
            shutil.copy2(recommender_path, backup_path)
            logger.info(f"Backup created: {backup_path}")
        
        # Load recommender model
        logger.info(f"Loading recommender model from {recommender_path}")
        with open(recommender_path, 'rb') as f:
            recommender_data = pickle.load(f)
        
        # Check if reranker already exists
        if 'reranker_model' in recommender_data and recommender_data['reranker_model'] is not None:
            logger.warning("Reranker model already exists in recommender.pkl. Overwriting...")
        
        # Merge reranker into recommender
        recommender_data['reranker_model'] = reranker_model
        recommender_data['reranker_metadata'] = reranker_metadata
        
        # Save updated recommender model
        logger.info(f"Saving updated recommender model to {recommender_path}")
        with open(recommender_path, 'wb') as f:
            pickle.dump(recommender_data, f)
        
        logger.info("Successfully merged reranker model into recommender.pkl")
        logger.info(f"Reranker metadata: {reranker_metadata}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to merge reranker model: {e}", exc_info=True)
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Merge reranker.pkl into recommender.pkl')
    parser.add_argument(
        '--reranker',
        type=str,
        default=str(RERANKER_PATH),
        help='Path to reranker.pkl (default: models/reranker.pkl)'
    )
    parser.add_argument(
        '--recommender',
        type=str,
        default=str(RECOMMENDER_PATH),
        help='Path to recommender.pkl (default: models/recommender.pkl)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup of recommender.pkl'
    )
    
    args = parser.parse_args()
    
    reranker_path = Path(args.reranker)
    recommender_path = Path(args.recommender)
    backup = not args.no_backup
    
    success = merge_reranker_into_recommender(
        reranker_path=reranker_path,
        recommender_path=recommender_path,
        backup=backup
    )
    
    if success:
        logger.info("Merge completed successfully!")
        logger.info("You can now delete reranker.pkl if you want (it's merged into recommender.pkl)")
    else:
        logger.error("Merge failed!")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

