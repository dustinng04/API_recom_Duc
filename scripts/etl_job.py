#!/usr/bin/env python3
"""
ETL Job for training data collection from search logs and interaction logs.

This script:
1. Extracts search logs from OpenSearch (multiple days, uses @timestamp)
2. Extracts interaction logs from OpenSearch (matching date range, uses timestamp)
3. Loads tutors data from local file
4. Transforms and merges data
5. Outputs training data as CSV (primary storage)
6. Pushes training data to OpenSearch index for monitoring (optional)
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
from opensearchpy.helpers import bulk
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
POSITIVE_EVENT_TYPES = ['click', 'conversion', 'join', 'rating', 'wishlist']
SEARCH_LOGS_INDEX_PATTERN = 'search-logs-*'
INTERACTION_LOGS_INDEX_PATTERN = 'interaction-logs-*'  # Assumed index pattern
TUTORS_DATA_PATH = Path('data/tutors_adjust.json')
OUTPUT_DIR = Path('data/training')


def get_opensearch_client() -> OpenSearch:
    """Initialize OpenSearch client from environment variables."""
    load_dotenv()
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
        date_str: Date in format 'YYYY.MM.DD' (e.g., '2025.11.07')
    
    Returns:
        List of search log documents
    """
    logger.info(f"Extracting search logs for date: {date_str}")
    
    # Convert date format from YYYY.MM.DD to YYYY-MM-DD for OpenSearch query
    # OpenSearch requires ISO 8601 format (with hyphens) for timestamp queries
    date_iso = date_str.replace('.', '-')
    
    # Build index name: search-logs-YYYY.MM.DD (with dots)
    index_name = f"search-logs-{date_str}"
    
    # Query for search logs on the specified date
    # Note: Adjust timezone if needed (sample shows +07:00)
    query = {
        "query": {
            "range": {
                "@timestamp": {
                    "gte": f"{date_iso}T00:00:00+07:00",
                    "lt": f"{date_iso}T23:59:59.999999+07:00"
                }
            }
        },
        "size": 10000
    }
    
    all_logs = []
    scroll_size = 10000
    
    try:
        response = client.search(
            index=index_name,
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
        
        logger.info(f"Total search logs extracted for {date_str}: {len(all_logs)}")
        return all_logs
    
    except Exception as e:
        logger.error(f"Error extracting search logs for {date_str}: {e}")
        # Return empty list instead of raising to allow processing other dates
        return []


def extract_search_logs_multi_days(client: OpenSearch, date_list: List[str]) -> List[Dict]:
    """
    Extract search logs from OpenSearch for multiple dates.
    
    Args:
        client: OpenSearch client
        date_list: List of dates in format 'YYYY.MM.DD'
    
    Returns:
        List of search log documents from all dates
    """
    logger.info(f"Extracting search logs for {len(date_list)} days: {date_list}")
    
    all_logs = []
    for date_str in date_list:
        logs = extract_search_logs(client, date_str)
        all_logs.extend(logs)
    
    logger.info(f"Total search logs extracted across all dates: {len(all_logs)}")
    return all_logs


def extract_interaction_logs(client: OpenSearch, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Extract interaction logs from OpenSearch for a date range.
    
    Args:
        client: OpenSearch client
        start_date: Start date in format 'YYYY-MM-DD' (ISO format)
        end_date: End date in format 'YYYY-MM-DD' (ISO format, exclusive)
    
    Returns:
        DataFrame with interaction logs
    """
    logger.info(f"Extracting interaction logs from {start_date} to {end_date}")
    
    # Query for interaction logs in the date range
    # Note: Using timestamp field (not @timestamp) for interaction logs
    # Adjust timezone if needed (sample shows +07:00)
    query = {
        "query": {
            "range": {
                "timestamp": {
                    "gte": f"{start_date}T00:00:00+07:00",
                    "lt": f"{end_date}T23:59:59.999999+07:00"
                }
            }
        },
        "size": 10000
    }
    
    all_logs = []
    scroll_size = 10000
    
    try:
        response = client.search(
            index=INTERACTION_LOGS_INDEX_PATTERN,
            body=query,
            scroll='5m',
            size=scroll_size
        )
        
        scroll_id = response.get('_scroll_id')
        hits = response['hits']['hits']
        all_logs.extend([hit['_source'] for hit in hits])
        logger.info(f"Initial batch: {len(hits)} interaction logs")
        
        # Continue scrolling if there are more results
        while len(hits) > 0:
            response = client.scroll(scroll_id=scroll_id, scroll='5m')
            scroll_id = response.get('_scroll_id')
            hits = response['hits']['hits']
            if hits:
                all_logs.extend([hit['_source'] for hit in hits])
                logger.info(f"Scroll batch: {len(hits)} interaction logs")
        
        df = pd.DataFrame(all_logs)
        logger.info(f"Total interaction logs extracted: {len(df)}")
        return df
    
    except Exception as e:
        logger.error(f"Error extracting interaction logs: {e}")
        logger.warning("Returning empty DataFrame. Check if interaction-logs-* index exists.")
        return pd.DataFrame()




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
        tutor_id = int(tutor_id)  # Convert to int

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
        DataFrame with columns: sessionId, userId, query, tutorId, os_score, rerank_score, position
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
            score = result.get('score')  # rerank_score (score sau khi rerank)
            os_score = result.get('os_score') or result.get('osScore') or 0.0  # os_score từ OpenSearch ban đầu
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
                    'os_score': float(os_score),
                    'rerank_score': float(score),
                    'position': int(rank)
                })
    
    df = pd.DataFrame(expanded_rows)
    df['tutorId'] = df['tutorId'].astype(int)
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
        
        try:
            tutor_id = int(tutor_id)
        except (ValueError, TypeError):
            return 0
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
    columns = ['userId', 'query', 'tutorId', 'os_score', 'rerank_score', 'price', 'rating', 'position', 'label']
    # Ensure os_score exists, if not create it with default 0.0
    if 'os_score' not in df.columns:
        df['os_score'] = 0.0
    df_output = df[columns].copy()
    
    # Save to CSV
    df_output.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Saved training data to {output_path} ({len(df_output)} rows)")
    
    return output_path


def save_training_data_to_opensearch(
    client: OpenSearch,
    df: pd.DataFrame,
    date_str: str
) -> None:
    """
    Push training data to OpenSearch index for monitoring and analytics.
    
    Args:
        client: OpenSearch client
        df: Training DataFrame
        date_str: Date string for index name (YYYY.MM.DD format)
    """
    logger.info(f"Pushing training data to OpenSearch index for date: {date_str}")
    
    index_name = f"train-data-raw-{date_str}"
    
    # Select and order columns (same as CSV output)
    columns = ['userId', 'query', 'tutorId', 'os_score', 'rerank_score', 'price', 'rating', 'position', 'label']
    df_output = df[columns].copy()
    
    # Convert DataFrame to list of dicts
    records = df_output.to_dict('records')
    
    # Prepare bulk actions
    actions = [
        {
            "_index": index_name,
            "_source": record
        }
        for record in records
    ]
    
    try:
        # Bulk insert
        success_count, failed_items = bulk(client, actions, raise_on_error=False)
        
        if failed_items:
            logger.warning(f"Failed to push {len(failed_items)} records to {index_name}")
            # Log first few failures for debugging
            for item in failed_items[:5]:
                logger.warning(f"Failed item: {item}")
        else:
            logger.info(f"Successfully pushed {success_count} records to {index_name}")
            
    except Exception as e:
        logger.error(f"Error pushing data to OpenSearch: {e}")
        logger.warning("Continuing without OpenSearch push. CSV file is still saved.")


def main():
    """Main ETL job execution."""
    logger.info("Starting ETL job...")
    
    # Get number of days to process (default: 1 day)
    num_days = int(os.getenv('ETL_NUM_DAYS', '1'))
    
    # Get number of days to look back (default: 3 days before base_date)
    days_lookback = int(os.getenv('ETL_DAYS_LOOKBACK', '3'))
    
    # Get base date (today by default, or from environment variable)
    base_date_str = os.getenv('ETL_DATE', None)
    if base_date_str:
        # Parse provided date
        try:
            base_date = datetime.strptime(base_date_str, '%Y.%m.%d')
        except ValueError:
            try:
                base_date = datetime.strptime(base_date_str, '%Y-%m-%d')
            except ValueError:
                logger.error(f"Invalid date format: {base_date_str}. Use YYYY.MM.DD or YYYY-MM-DD")
                return
    else:
        # Use today
        base_date = datetime.now()
    
    # Calculate target date: days_lookback days before base_date
    target_date = base_date - timedelta(days=days_lookback)
    
    # Generate list of dates to process (num_days going backwards from target_date)
    date_list = []
    for i in range(num_days):
        date = target_date - timedelta(days=i)
        date_str = date.strftime('%Y.%m.%d')
        date_list.append(date_str)
    
    logger.info(f"Base date: {base_date.strftime('%Y.%m.%d')}")
    logger.info(f"Target date (lookback {days_lookback} days): {target_date.strftime('%Y.%m.%d')}")
    logger.info(f"Processing data for {num_days} day(s): {date_list}")
    
    # Calculate date range for interaction logs (start = oldest, end = newest + 1 day)
    start_date_iso = (target_date - timedelta(days=num_days-1)).strftime('%Y-%m-%d')
    end_date_iso = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    try:
        # Step 1: Initialize OpenSearch client
        client = get_opensearch_client()
        
        # Step 2: Extract search logs from OpenSearch (multiple days)
        search_logs = extract_search_logs_multi_days(client, date_list)
        
        if not search_logs:
            logger.warning("No search logs found. Exiting.")
            return
        
        # Step 3: Extract interaction logs from OpenSearch (matching date range)
        interactions_df = extract_interaction_logs(client, start_date_iso, end_date_iso)
        
        if interactions_df.empty:
            logger.warning("No interaction logs found. Training data will have all negative labels.")
        
        # Step 4: Load tutors data
        tutors_map = load_tutors_data()
        
        if not tutors_map:
            logger.error("No tutors data loaded. Cannot proceed.")
            return
        
        # Step 5: Expand search logs
        search_df = expand_search_logs(search_logs)
        
        if search_df.empty:
            logger.warning("No valid search logs after expansion. Exiting.")
            return
        
        # Step 6: Aggregate interactions
        interactions_map = aggregate_interactions(interactions_df)
        
        # Step 7: Merge with interactions to assign labels
        search_df = merge_with_interactions(search_df, interactions_map)
        
        # Step 8: Map tutors data
        search_df = map_tutors_data(search_df, tutors_map)
        
        if search_df.empty:
            logger.warning("No training data after mapping tutors. Exiting.")
            return
        
        # Step 9: Save training data to CSV (use target date for filename)
        target_date_str = target_date.strftime('%Y.%m.%d')
        output_path = save_training_data(search_df, target_date_str)
        
        # Step 10: Push training data to OpenSearch for monitoring (optional, continues on error)
        try:
            save_training_data_to_opensearch(client, search_df, target_date_str)
        except Exception as e:
            logger.warning(f"Failed to push data to OpenSearch: {e}. CSV file is still saved.")
        
        logger.info("ETL job completed successfully!")
        logger.info(f"Output: {output_path}")
        logger.info(f"Total rows: {len(search_df)}")
        logger.info(f"Positive labels: {search_df['label'].sum()}")
        logger.info(f"Negative labels: {(search_df['label'] == 0).sum()}")
        logger.info(f"Date range processed: {date_list}")
    
    except Exception as e:
        logger.error(f"ETL job failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

