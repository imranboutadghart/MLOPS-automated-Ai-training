"""
Data Ingestion Module
Handles loading data from various sources including local files, S3, and databases.
"""

import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class DataIngestion:
    """
    Data ingestion class for loading data from various sources.
    
    Supports:
    - Local CSV/Parquet files
    - S3 buckets (via boto3)
    - Database connections (PostgreSQL)
    """
    
    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize data ingestion with optional configuration."""
        self.config = config or {}
        self._setup_clients()
    
    def _setup_clients(self) -> None:
        """Set up clients for various data sources."""
        # S3 client setup
        if self.config.get("s3_enabled", False):
            import boto3
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
        else:
            self.s3_client = None
    
    def load_local(
        self,
        path: str | Path,
        file_format: str = "auto",
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Load data from local filesystem.
        
        Args:
            path: Path to data file or directory
            file_format: File format ('csv', 'parquet', 'json', 'auto')
            **kwargs: Additional arguments passed to pandas reader
            
        Returns:
            DataFrame with loaded data
        """
        path = Path(path)
        
        if file_format == "auto":
            file_format = path.suffix.lstrip(".")
        
        logger.info("Loading local data", path=str(path), format=file_format)
        
        readers = {
            "csv": pd.read_csv,
            "parquet": pd.read_parquet,
            "json": pd.read_json,
            "feather": pd.read_feather,
        }
        
        if file_format not in readers:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        if path.is_dir():
            # Load all files from directory
            dfs = []
            for file in path.glob(f"*.{file_format}"):
                dfs.append(readers[file_format](file, **kwargs))
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = readers[file_format](path, **kwargs)
        
        logger.info("Data loaded successfully", rows=len(df), columns=len(df.columns))
        return df
    
    def load_from_s3(
        self,
        bucket: str,
        key: str,
        file_format: str = "parquet",
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Load data from S3 bucket.
        
        Args:
            bucket: S3 bucket name
            key: Object key (path within bucket)
            file_format: File format
            **kwargs: Additional arguments passed to pandas reader
            
        Returns:
            DataFrame with loaded data
        """
        if self.s3_client is None:
            raise RuntimeError("S3 client not initialized. Enable s3_enabled in config.")
        
        import io
        
        logger.info("Loading data from S3", bucket=bucket, key=key)
        
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        data = response["Body"].read()
        
        if file_format == "parquet":
            df = pd.read_parquet(io.BytesIO(data), **kwargs)
        elif file_format == "csv":
            df = pd.read_csv(io.BytesIO(data), **kwargs)
        else:
            raise ValueError(f"Unsupported format for S3: {file_format}")
        
        logger.info("S3 data loaded successfully", rows=len(df), columns=len(df.columns))
        return df
    
    def load_from_database(
        self,
        query: str,
        connection_string: Optional[str] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Load data from database using SQL query.
        
        Args:
            query: SQL query to execute
            connection_string: Database connection string
            **kwargs: Additional arguments passed to pandas
            
        Returns:
            DataFrame with query results
        """
        from sqlalchemy import create_engine
        
        conn_str = connection_string or os.getenv("DATABASE_URL")
        if not conn_str:
            raise ValueError("No connection string provided")
        
        logger.info("Loading data from database")
        
        engine = create_engine(conn_str)
        df = pd.read_sql(query, engine, **kwargs)
        
        logger.info("Database query completed", rows=len(df))
        return df


def load_training_data(
    source: str,
    config: Optional[dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Convenience function to load training data.
    
    Args:
        source: Data source path or identifier
        config: Optional configuration
        
    Returns:
        DataFrame with training data
    """
    ingestion = DataIngestion(config)
    
    if source.startswith("s3://"):
        # Parse S3 URL
        parts = source[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return ingestion.load_from_s3(bucket, key)
    else:
        return ingestion.load_local(source)
