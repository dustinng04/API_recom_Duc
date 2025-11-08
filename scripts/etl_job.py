#!/usr/bin/env python3
"""
ETL Job for training data collection from search logs and interaction logs.

This script:
1. Extracts search logs from OpenSearch (today's data)
2. Loads interaction logs from local file
3. Loads tutors data from local file
4. Transforms and merges data
5. Outputs training data as CSV
"""

from dotenv import load_dotenv
import os
import json
import ssl
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Set
from opensearchpy import OpenSearch
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
POSITIVE_EVENT_TYPES = ['click', 'conversion', 'join', 'rating', 'wishlist']
SEARCH_LOGS_INDEX = 'search-logs-*'
INTERACTION_LOGS_PATH = Path('data/interaction_logs.jsonl')
TUTORS_DATA_PATH = Path('data/tutors_adjust.json')
OUTPUT_DIR = Path('data/training')


def get_opensearch_client() -> OpenSearch:

    load_dotenv()
    print("DEBUG:", os.getenv("OS_HOST"), os.getenv("OS_USERNAME"))
    """Initialize OpenSearch client from environment variables."""
    host = os.getenv('OS_HOST')
    username = os.getenv('OS_USERNAME')
    password = os.getenv('OS_PASSWORD')
    
    if not all([host, username, password]):
        raise ValueError(
            "Missing OpenSearch credentials. Please set OS_HOST, OS_USERNAME, OS_PASSWORD"
        )
    
    # Parse host URL
    use_ssl = False
    port = 443
    
    # Remove protocol
    if host.startswith('https://'):
        use_ssl = True
        host = host.replace('https://', '')
    elif host.startswith('http://'):
        host = host.replace('http://', '')
        port = 80
    
    # Extract port if present
    if ':' in host:
        host, port_str = host.split(':', 1)
        try:
            port = int(port_str)
        except ValueError:
            logger.warning(f"Invalid port in OS_HOST: {port_str}, using default {port}")
    
    # Default to SSL if port is 443
    if port == 443 and not use_ssl:
        use_ssl = True
    
    # Build SSL context that trusts all certificates
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    
    client = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_auth=(username, password),
        use_ssl=use_ssl,
        ssl_context=context,
        verify_certs=False,
        ssl_show_warn=False
    )
    
    logger.info(f"Connected to OpenSearch: {host}:{port} (SSL: {use_ssl})")
    return client


def extract_search_logs(client: OpenSearch, date_str: str) -> List[Dict]:
    """
    Extract search logs from OpenSearch for a specific date.
    
    Args:
        client: OpenSearch client
        date_str: Date in format 'YYYY-MM-DD'
    
    Returns:
        List of search log documents
    """
    logger.info(f"Extracting search logs for date: {date_str}")
    
    # Query for search logs on the specified date
    # Note: Adjust timezone if needed (sample shows +07:00)
    query = {
        "query": {
            "range": {
                "@timestamp": {
                    "gte": f"{date_str}T00:00:00+07:00",
                    "lt": f"{date_str}T23:59:59.999999+07:00"
                }
            }
        },
        "size": 10000
    }
    
    all_logs = []
    scroll_size = 10000
    
    try:
        response = client.search(
            index=SEARCH_LOGS_INDEX,
            body=query,
            scroll='5m',
            size=scroll_size
        )
        
        scroll_id = response.get('_scroll_id')
        hits = response['hits']['hits']
        all_logs.extend([hit['_source'] for hit in hits])
        logger.info(f"Initial batch: {len(hits)} logs")
        
        # Continue scrolling if there are more results
        while len(hits) > 0:
            response = client.scroll(scroll_id=scroll_id, scroll='5m')
            scroll_id = response.get('_scroll_id')
            hits = response['hits']['hits']
            if hits:
                all_logs.extend([hit['_source'] for hit in hits])
                logger.info(f"Scroll batch: {len(hits)} logs")
        
        logger.info(f"Total search logs extracted: {len(all_logs)}")
        return all_logs
    
    except Exception as e:
        logger.error(f"Error extracting search logs: {e}")
        raise


def load_interaction_logs() -> pd.DataFrame:
    """Load interaction logs from local file."""
    logger.info(f"Loading interaction logs from {INTERACTION_LOGS_PATH}")
    
    if not INTERACTION_LOGS_PATH.exists():
        logger.warning(f"Interaction logs file not found: {INTERACTION_LOGS_PATH}")
        return pd.DataFrame()
    
    interactions = []
    with open(INTERACTION_LOGS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                interactions.append(json.loads(line))
    
    df = pd.DataFrame(interactions)
    logger.info(f"Loaded {len(df)} interactions")
    return df


def load_tutors_data() -> Dict[int, Dict]:
    """Load tutors data and create mapping {tutorId: {price, rating}}."""
    logger.info(f"Loading tutors data from {TUTORS_DATA_PATH}")
    
    if not TUTORS_DATA_PATH.exists():
        logger.warning(f"Tutors data file not found: {TUTORS_DATA_PATH}")
        return {}
    
    with open(TUTORS_DATA_PATH, 'r', encoding='utf-8') as f:
        tutors = json.load(f)
    
    tutors_map = {}
    for tutor in tutors:
        tutor_id = tutor.get('id')
        if tutor_id:
            tutors_map[tutor_id] = {
                'price': tutor.get('price', 0),
                'rating': tutor.get('rating', 0)
            }
    
    logger.info(f"Loaded {len(tutors_map)} tutors")
    return tutors_map


def expand_search_logs(search_logs: List[Dict]) -> pd.DataFrame:
    """
    Expand search logs: each result in results[] becomes one row.
    
    Returns:
        DataFrame with columns: sessionId, userId, query, tutorId, rerank_score, position
    """
    logger.info("Expanding search logs...")
    
    expanded_rows = []
    for log in search_logs:
        session_id = log.get('sessionId')
        user_id = log.get('userId')
        query = log.get('query', '')
        results = log.get('results', [])
        
        if not session_id or not user_id or not results:
            continue
        
        for result in results:
            tutor_id = result.get('tutorId')
            score = result.get('score')
            rank = result.get('rank')
            
            if tutor_id is not None and score is not None and rank is not None:
                # Convert tutorId to int if it's a string
                try:
                    tutor_id = int(tutor_id)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid tutorId: {tutor_id}, skipping")
                    continue
                
                expanded_rows.append({
                    'sessionId': session_id,
                    'userId': user_id,
                    'query': query,
                    'tutorId': tutor_id,
                    'rerank_score': float(score),
                    'position': int(rank)
                })
    
    df = pd.DataFrame(expanded_rows)
    logger.info(f"Expanded to {len(df)} rows from {len(search_logs)} search logs")
    return df


def aggregate_interactions(interactions_df: pd.DataFrame) -> Dict[str, Set[int]]:
    """
    Aggregate interactions by sessionId and tutorId.
    
    For each (sessionId, tutorId) pair, check if there's any positive interaction.
    
    Returns:
        Dict mapping sessionId -> Set of tutorIds with positive interactions
    """
    logger.info("Aggregating interactions...")
    
    # Filter for positive event types
    positive_interactions = interactions_df[
        interactions_df['eventType'].isin(POSITIVE_EVENT_TYPES)
    ].copy()
    
    # Group by sessionId and tutorId
    # If there's at least one positive interaction, mark as label=1
    session_tutor_map = defaultdict(set)
    
    for _, row in positive_interactions.iterrows():
        session_id = row.get('sessionId')
        tutor_id = row.get('tutorId')
        
        if session_id and tutor_id is not None:
            try:
                tutor_id = int(tutor_id)
                session_tutor_map[session_id].add(tutor_id)
            except (ValueError, TypeError):
                continue
    
    logger.info(f"Found {len(session_tutor_map)} sessions with positive interactions")
    return dict(session_tutor_map)


def merge_with_interactions(
    search_df: pd.DataFrame,
    interactions_map: Dict[str, Set[int]]
) -> pd.DataFrame:
    """
    Merge search logs with interactions to assign labels.
    
    Args:
        search_df: Expanded search logs DataFrame
        interactions_map: Dict mapping sessionId -> Set of tutorIds with positive interactions
    
    Returns:
        DataFrame with label column added
    """
    logger.info("Merging with interactions...")
    
    def assign_label(row):
        session_id = row['sessionId']
        tutor_id = row['tutorId']
        
        if session_id in interactions_map:
            if tutor_id in interactions_map[session_id]:
                return 1
        return 0
    
    search_df['label'] = search_df.apply(assign_label, axis=1)
    
    positive_count = search_df['label'].sum()
    logger.info(f"Assigned labels: {positive_count} positive, {len(search_df) - positive_count} negative")
    
    return search_df


def map_tutors_data(
    search_df: pd.DataFrame,
    tutors_map: Dict[int, Dict]
) -> pd.DataFrame:
    """
    Map tutors data (price, rating) to search DataFrame.
    
    Args:
        search_df: DataFrame with tutorId column
        tutors_map: Dict mapping tutorId -> {price, rating}
    
    Returns:
        DataFrame with price and rating columns added
    """
    logger.info("Mapping tutors data...")
    
    def get_tutor_info(tutor_id):
        return tutors_map.get(tutor_id, None)
    
    # Filter out tutors not found in tutors_map
    initial_count = len(search_df)
    search_df['tutor_info'] = search_df['tutorId'].apply(get_tutor_info)
    search_df = search_df[search_df['tutor_info'].notna()].copy()
    
    # Extract price and rating
    search_df['price'] = search_df['tutor_info'].apply(lambda x: x['price'])
    search_df['rating'] = search_df['tutor_info'].apply(lambda x: x['rating'])
    
    # Drop temporary column
    search_df = search_df.drop(columns=['tutor_info'])
    
    skipped_count = initial_count - len(search_df)
    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} rows with tutors not found in tutors data")
    
    logger.info(f"Mapped tutors data for {len(search_df)} rows")
    return search_df


def save_training_data(df: pd.DataFrame, date_str: str) -> Path:
    """
    Save training data to CSV file.
    
    Args:
        df: Training DataFrame
        date_str: Date string for filename
    
    Returns:
        Path to saved file
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    filename = f"train_data_{date_str}.csv"
    output_path = OUTPUT_DIR / filename
    
    # Select and order columns
    columns = ['userId', 'query', 'tutorId', 'rerank_score', 'price', 'rating', 'position', 'label']
    df_output = df[columns].copy()
    
    # Save to CSV
    df_output.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Saved training data to {output_path} ({len(df_output)} rows)")
    
    return output_path


def main():
    """Main ETL job execution."""
    logger.info("Starting ETL job...")
    
    # # Get date (today by default, or from environment variable)
    # date_str = os.getenv('ETL_DATE', datetime.now().strftime('%Y-%m-%d'))

     # Get date (default: 2025-11-07, or from environment variable, or today)
    date_str = os.getenv('ETL_DATE', '2025-11-07')
    logger.info(f"Processing data for date: {date_str}")
    
    try:
        # Step 1: Extract search logs from OpenSearch
        client = get_opensearch_client()
        search_logs = extract_search_logs(client, date_str)
        
        if not search_logs:
            logger.warning("No search logs found. Exiting.")
            return
        
        # Step 2: Load interaction logs
        interactions_df = load_interaction_logs()
        
        # Step 3: Load tutors data
        tutors_map = load_tutors_data()
        
        if not tutors_map:
            logger.error("No tutors data loaded. Cannot proceed.")
            return
        
        # Step 4: Expand search logs
        search_df = expand_search_logs(search_logs)
        
        if search_df.empty:
            logger.warning("No valid search logs after expansion. Exiting.")
            return
        
        # Step 5: Aggregate interactions
        interactions_map = aggregate_interactions(interactions_df)
        
        # Step 6: Merge with interactions to assign labels
        search_df = merge_with_interactions(search_df, interactions_map)
        
        # Step 7: Map tutors data
        search_df = map_tutors_data(search_df, tutors_map)
        
        if search_df.empty:
            logger.warning("No training data after mapping tutors. Exiting.")
            return
        
        # Step 8: Save training data
        output_path = save_training_data(search_df, date_str)
        
        logger.info("ETL job completed successfully!")
        logger.info(f"Output: {output_path}")
        logger.info(f"Total rows: {len(search_df)}")
        logger.info(f"Positive labels: {search_df['label'].sum()}")
        logger.info(f"Negative labels: {(search_df['label'] == 0).sum()}")
    
    except Exception as e:
        logger.error(f"ETL job failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

