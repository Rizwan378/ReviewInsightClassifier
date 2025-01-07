import pandas as pd
import logging
import yaml
from dvc.api import DVCFileSystem
from typing import Optional, Dict
from pathlib import Path
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReviewDataIngestion:
    def __init__(self, config_path: str):
        """Initialize with a configuration file for review data ingestion."""
        self.config = self._load_config(config_path)
        self.s3_client = boto3.client('s3') if self.config.get('source') == 's3' else None
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config

    def load_data(self, file_path: str, version: Optional[str] = None) -> pd.DataFrame:
        """Load review data from local filesystem or S3 with versioning support."""
        try:
            if self.config.get('source') == 's3':
                data = self._load_from_s3(file_path)
            else:
                data = self._load_from_local(file_path, version)
            
            logger.info(f"Successfully loaded review data from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load review data: {str(e)}")
            raise

    def _load_from_local(self, file_path: str, version: Optional[str]) -> pd.DataFrame:
        """Load data from local filesystem with DVC versioning."""
        if version:
            fs = DVCFileSystem()
            local_path = Path(f"data/{Path(file_path).name}")
            fs.get(file_path, str(local_path), version=version)
            file_path = local_path
        return pd.read_csv(file_path)

    def _load_from_s3(self, file_path: str) -> pd.DataFrame:
        """Load data from S3 bucket."""
        bucket = self.config['s3_bucket']
        obj = self.s3_client.get_object(Bucket=bucket, Key=file_path)
        return pd.read_csv(obj['Body'])

    def validate_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate that required columns exist in review data."""
        required = ['review_text', 'rating']
        missing = [col for col in required if col not in data.columns]
        if missing:
            logger.error(f"Missing columns: {missing}")
            raise ValueError(f"Data missing required columns: {missing}")
        logger.info("All required columns validated")
        return data

    def extract_metadata(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract metadata like review length and word count."""
        data['review_length'] = data['review_text'].str.len()
        data['word_count'] = data['review_text'].str.split().str.len()
        data['has_emoji'] = data['review_text'].str.contains(r'[^\x00-\x7F]+')
        logger.info("Extracted metadata: length, word count, emoji presence")
        return data

    def filter_invalid_reviews(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove reviews with invalid or empty text."""
        original_len = len(data)
        data = data.dropna(subset=['review_text'])
        data = data[data['review_text'].str.strip() != '']
        logger.info(f"Filtered {original_len - len(data)} invalid reviews")
        return data.reset_index(drop=True)

    def normalize_ratings(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize rating values to 0-1 range."""
        if 'rating' in data.columns:
            max_rating = data['rating'].max()
            data['rating_normalized'] = data['rating'] / max_rating
            data['rating_normalized'] = data['rating_normalized'].clip(0, 1)
            logger.info(f"Normalized ratings to 0-1 range (max: {max_rating})")
        return data

    def log_data_stats(self, data: pd.DataFrame) -> None:
        """Log statistics about the review dataset."""
        stats = {
            'num_reviews': len(data),
            'avg_length': data['review_text'].str.len().mean(),
            'unique_words': len(set(' '.join(data['review_text']).split())),
            'rating_mean': data['rating'].mean() if 'rating' in data.columns else None
        }
        logger.info(f"Dataset stats: {stats}")

    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate review data for empty or invalid entries."""
        if data.empty:
            logger.error("Empty DataFrame provided")
            raise ValueError("DataFrame is empty")
        data = data.dropna(subset=['review_text'])
        data = data[data['review_text'].str.len() > 5]
        logger.info(f"Validated data, retained {len(data)} reviews")
        return data.reset_index(drop=True)

    def validate_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate that required columns exist in review data."""
        required = ['review_text', 'rating']
        missing = [col for col in required if col not in data.columns]
        if missing:
            logger.error(f"Missing columns: {missing}")
            raise ValueError(f"Data missing required columns: {missing}")
        logger.info("All required columns validated")
        return data

    def extract_metadata(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract metadata like review length and word count."""
        data['review_length'] = data['review_text'].str.len()
        data['word_count'] = data['review_text'].str.split().str.len()
        data['has_emoji'] = data['review_text'].str.contains(r'[^\x00-\x7F]+')
        logger.info("Extracted metadata: length, word count, emoji presence")
        return data

    def filter_invalid_reviews(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove reviews with invalid or empty text."""
        original_len = len(data)
        data = data.dropna(subset=['review_text'])
        data = data[data['review_text'].str.strip() != '']
        logger.info(f"Filtered {original_len - len(data)} invalid reviews")
        return data.reset_index(drop=True)

    def normalize_ratings(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize rating values to 0-1 range."""
        if 'rating' in data.columns:
            max_rating = data['rating'].max()
            data['rating_normalized'] = data['rating'] / max_rating
            data['rating_normalized'] = data['rating_normalized'].clip(0, 1)
            logger.info(f"Normalized ratings to 0-1 range (max: {max_rating})")
        return data
