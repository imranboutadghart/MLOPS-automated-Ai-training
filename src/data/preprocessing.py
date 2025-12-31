"""
Data Preprocessing Module
Pandas-based preprocessing pipeline for feature engineering and data transformation.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = structlog.get_logger(__name__)


class DataPreprocessor:
    """
    Data preprocessing pipeline with support for:
    - Missing value handling
    - Feature scaling
    - Categorical encoding
    - Feature engineering
    - Train/validation/test splitting
    """
    
    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize preprocessor with configuration."""
        self.config = config or {}
        self.scalers: dict[str, StandardScaler] = {}
        self.encoders: dict[str, LabelEncoder] = {}
        self._fitted = False
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "mean",
        fill_value: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'fill', 'drop')
            fill_value: Value to use when strategy is 'fill'
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        
        logger.info("Handling missing values", strategy=strategy)
        
        if strategy == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            # Handle categorical columns only if they exist and have data
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    mode_val = df[col].mode()
                    fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "unknown"
                    df[col] = df[col].fillna(fill_val)
        elif strategy == "median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    mode_val = df[col].mode()
                    fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "unknown"
                    df[col] = df[col].fillna(fill_val)
        elif strategy == "mode":
            for col in df.columns:
                mode_val = df[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else (fill_value if fill_value is not None else 0)
                df[col] = df[col].fillna(fill_val)
        elif strategy == "fill":
            df = df.fillna(fill_value)
        elif strategy == "drop":
            df = df.dropna()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return df
    
    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: Optional[list[str]] = None,
        method: str = "label"
    ) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            columns: Columns to encode (None for auto-detect)
            method: Encoding method ('label', 'onehot')
            
        Returns:
            DataFrame with encoded categories
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        logger.info("Encoding categorical columns", columns=columns, method=method)
        
        for col in columns:
            if method == "label":
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[col] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.encoders[col].transform(df[col].astype(str))
            elif method == "onehot":
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
        
        return df
    
    def scale_features(
        self,
        df: pd.DataFrame,
        columns: Optional[list[str]] = None,
        method: str = "standard"
    ) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input DataFrame
            columns: Columns to scale (None for auto-detect)
            method: Scaling method ('standard', 'minmax')
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info("Scaling features", columns=columns, method=method)
        
        for col in columns:
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()
                df[col] = self.scalers[col].fit_transform(df[[col]])
            else:
                df[col] = self.scalers[col].transform(df[[col]])
        
        return df
    
    def create_features(
        self,
        df: pd.DataFrame,
        feature_config: Optional[dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Create new features based on existing ones.
        
        Args:
            df: Input DataFrame
            feature_config: Configuration for feature creation
            
        Returns:
            DataFrame with new features
        """
        df = df.copy()
        feature_config = feature_config or {}
        
        logger.info("Creating features")
        
        # Example feature engineering
        # Add interaction features
        if feature_config.get("interactions"):
            for col1, col2 in feature_config["interactions"]:
                if col1 in df.columns and col2 in df.columns:
                    df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
        
        # Add polynomial features
        if feature_config.get("polynomial"):
            for col in feature_config["polynomial"]:
                if col in df.columns:
                    df[f"{col}_squared"] = df[col] ** 2
        
        # Add date features if datetime columns exist
        date_cols = df.select_dtypes(include=["datetime64"]).columns
        for col in date_cols:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        
        return df
    
    def split_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state
        )
        
        logger.info(
            "Data split completed",
            train_size=len(X_train),
            val_size=len(X_val),
            test_size=len(X_test),
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def run_pipeline(
        self,
        input_path: str | Path,
        output_path: str | Path,
        config: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Run complete preprocessing pipeline.
        
        Args:
            input_path: Path to input data
            output_path: Path to save processed data
            config: Pipeline configuration
            
        Returns:
            Path to processed data
        """
        config = config or self.config
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting preprocessing pipeline", input=str(input_path))
        
        # Load data
        if input_path.suffix == ".parquet":
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)
        
        # Apply preprocessing steps
        df = self.handle_missing_values(
            df, 
            strategy=config.get("missing_strategy", "mean")
        )
        
        df = self.encode_categorical(
            df,
            method=config.get("encoding_method", "label")
        )
        
        df = self.scale_features(
            df,
            method=config.get("scaling_method", "standard")
        )
        
        if config.get("feature_config"):
            df = self.create_features(df, config["feature_config"])
        
        # Save processed data
        output_file = output_path / "processed_data.parquet"
        df.to_parquet(output_file, index=False)
        
        logger.info("Preprocessing completed", output=str(output_file))
        
        self._fitted = True
        return str(output_file)

    def run(self) -> dict[str, Any]:
        """
        Run the complete preprocessing pipeline including data loading, processing, splitting, and saving.
        
        Returns:
            Dictionary containing paths and statistics for the processed data.
        """
        config = self.config
        
        # Get paths from config
        data_config = config.get("data", {})
        raw_path = Path(data_config.get("raw_path", "/opt/airflow/data/raw"))
        train_dir = Path(data_config.get("train_path", "/opt/airflow/data/processed/training"))
        val_dir = Path(data_config.get("val_path", "/opt/airflow/data/processed/validation"))
        test_dir = Path(data_config.get("test_path", "/opt/airflow/data/processed/test"))
        
        # Ensure output directories exist
        for d in [train_dir, val_dir, test_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        logger.info("Starting preprocessing run", raw_path=str(raw_path))
        
        # Load data
        if raw_path.is_dir():
            dfs = []
            for file in raw_path.glob("*.parquet"):
                dfs.append(pd.read_parquet(file))
            for file in raw_path.glob("*.csv"):
                dfs.append(pd.read_csv(file))
            
            if not dfs:
                raise FileNotFoundError(f"No data files found in {raw_path}")
            df = pd.concat(dfs, ignore_index=True)
        elif raw_path.exists():
            if raw_path.suffix == ".parquet":
                df = pd.read_parquet(raw_path)
            else:
                df = pd.read_csv(raw_path)
        else:
            raise FileNotFoundError(f"Raw data path not found: {raw_path}")
            
        # Get sub-configs
        prep_config = data_config.get("preprocessing", {})
        root_prep_config = config.get("preprocessing", {})
        
        # Apply pipeline steps
        df = self.handle_missing_values(
            df, 
            strategy=prep_config.get("missing_strategy", "mean")
        )
        
        # Handle zeros in diabetes dataset (zeros are missing values for certain columns)
        # Columns where 0 is biologically impossible: Glucose, BloodPressure, SkinThickness, Insulin, BMI
        if "Glucose" in df.columns:
            logger.info("Detected diabetes dataset - handling zero values as missing")
            zero_as_missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
            for col in zero_as_missing_cols:
                if col in df.columns:
                    df[col] = df[col].replace(0, np.nan)
            # Fill missing values with median (more robust for medical data)
            df = self.handle_missing_values(df, strategy="median")
        
        df = self.encode_categorical(
            df,
            method=prep_config.get("encoding_method", "label")
        )
        
        df = self.scale_features(
            df,
            method=prep_config.get("scaling_method", "standard")
        )
        
        if "feature_config" in prep_config:
            df = self.create_features(df, prep_config["feature_config"])
            
        # Split data
        target_col = root_prep_config.get("target_column", "target")
        
        if target_col not in df.columns:
             if "target" in df.columns:
                 target_col = "target"
             else:
                 logger.warning("Target column not found, using last column", desired=target_col)
                 target_col = df.columns[-1]

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            df,
            target_column=target_col,
            test_size=root_prep_config.get("test_size", 0.15),
            val_size=root_prep_config.get("val_size", 0.15),
            random_state=root_prep_config.get("random_state", 42)
        )
        
        # Save split datasets
        train_file = train_dir / "train.parquet"
        val_file = val_dir / "val.parquet"
        test_file = test_dir / "test.parquet"
        
        pd.concat([X_train, y_train], axis=1).to_parquet(train_file, index=False)
        pd.concat([X_val, y_val], axis=1).to_parquet(val_file, index=False)
        pd.concat([X_test, y_test], axis=1).to_parquet(test_file, index=False)
        
        logger.info("Preprocessing run completed", 
                   train_samples=len(X_train),
                   val_samples=len(X_val))
        
        return {
            "train_path": str(train_file),
            "val_path": str(val_file),
            "test_path": str(test_file),
            "num_train_samples": len(X_train),
            "num_val_samples": len(X_val),
            "preprocessing_time": pd.Timestamp.now().isoformat()
        }
