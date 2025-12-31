"""
Data Validation Module
Quality checks and schema validation for training data.
"""

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import structlog
from pydantic import BaseModel, field_validator

logger = structlog.get_logger(__name__)


class DataSchema(BaseModel):
    """Schema definition for data validation."""
    
    required_columns: list[str] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    target_column: Optional[str] = None
    min_rows: int = 100
    max_null_ratio: float = 0.3
    
    @field_validator("max_null_ratio")
    @classmethod
    def validate_null_ratio(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("max_null_ratio must be between 0 and 1")
        return v


class ValidationResult(BaseModel):
    """Result of data validation."""
    
    is_valid: bool
    num_samples: int
    num_features: int
    features: list[str]
    errors: list[str] = []
    warnings: list[str] = []
    statistics: dict[str, Any] = {}


class DataValidator:
    """
    Data validation class for ensuring data quality before training.
    
    Performs checks for:
    - Schema compliance
    - Missing values
    - Data types
    - Value ranges
    - Duplicates
    """
    
    def __init__(self, schema: Optional[DataSchema] = None):
        """Initialize validator with optional schema."""
        self.schema = schema or DataSchema()
    
    def check_schema(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Check if DataFrame matches expected schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        
        # Check required columns
        missing_cols = set(self.schema.required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check numeric columns
        for col in self.schema.numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column {col} should be numeric but is {df[col].dtype}")
        
        # Check target column
        if self.schema.target_column and self.schema.target_column not in df.columns:
            errors.append(f"Target column '{self.schema.target_column}' not found")
        
        return len(errors) == 0, errors
    
    def check_null_values(self, df: pd.DataFrame) -> tuple[bool, list[str], list[str]]:
        """
        Check for excessive null values.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        null_ratios = df.isnull().mean()
        
        for col, ratio in null_ratios.items():
            if ratio > self.schema.max_null_ratio:
                errors.append(f"Column '{col}' has {ratio:.1%} null values (max: {self.schema.max_null_ratio:.1%})")
            elif ratio > 0.1:
                warnings.append(f"Column '{col}' has {ratio:.1%} null values")
        
        return len(errors) == 0, errors, warnings
    
    def check_data_quality(self, df: pd.DataFrame) -> tuple[bool, list[str], list[str]]:
        """
        Perform data quality checks.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check minimum rows
        if len(df) < self.schema.min_rows:
            errors.append(f"Insufficient data: {len(df)} rows (minimum: {self.schema.min_rows})")
        
        # Check for duplicates
        duplicate_ratio = df.duplicated().mean()
        if duplicate_ratio > 0.1:
            warnings.append(f"Data has {duplicate_ratio:.1%} duplicate rows")
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                warnings.append(f"Column '{col}' has constant value")
        
        # Check numeric column ranges
        for col in self.schema.numeric_columns:
            if col in df.columns:
                if df[col].std() == 0:
                    warnings.append(f"Column '{col}' has zero variance")
        
        return len(errors) == 0, errors, warnings
    
    def compute_statistics(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Compute dataset statistics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "null_counts": df.isnull().sum().to_dict(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
        
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            stats["numeric_summary"] = df[numeric_cols].describe().to_dict()
        
        # Categorical column statistics
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            stats["categorical_summary"] = {
                col: {
                    "unique_values": df[col].nunique(),
                    "top_values": df[col].value_counts().head(5).to_dict(),
                }
                for col in cat_cols
            }
        
        return stats
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Run all validation checks.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with all check results
        """
        logger.info("Starting data validation", rows=len(df), columns=len(df.columns))
        
        all_errors = []
        all_warnings = []
        
        # Schema check
        schema_valid, schema_errors = self.check_schema(df)
        all_errors.extend(schema_errors)
        
        # Null value check
        null_valid, null_errors, null_warnings = self.check_null_values(df)
        all_errors.extend(null_errors)
        all_warnings.extend(null_warnings)
        
        # Quality check
        quality_valid, quality_errors, quality_warnings = self.check_data_quality(df)
        all_errors.extend(quality_errors)
        all_warnings.extend(quality_warnings)
        
        # Compute statistics
        statistics = self.compute_statistics(df)
        
        is_valid = len(all_errors) == 0
        
        result = ValidationResult(
            is_valid=is_valid,
            num_samples=len(df),
            num_features=len(df.columns),
            features=df.columns.tolist(),
            errors=all_errors,
            warnings=all_warnings,
            statistics=statistics,
        )
        
        if is_valid:
            logger.info("Data validation passed", warnings=len(all_warnings))
        else:
            logger.error("Data validation failed", errors=all_errors)
        
        return result
    
    def validate_training_data(self, data_path: str | Path) -> dict[str, Any]:
        """
        Validate training data from file path.
        
        Args:
            data_path: Path to training data
            
        Returns:
            Dictionary with validation results
        """
        data_path = Path(data_path)
        
        # Load data
        if data_path.is_dir():
            # Load all files from directory
            dfs = []
            for file in data_path.glob("*.parquet"):
                dfs.append(pd.read_parquet(file))
            for file in data_path.glob("*.csv"):
                dfs.append(pd.read_csv(file))
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
            else:
                return {
                    "is_valid": False,
                    "errors": ["No data files found in directory"],
                    "num_samples": 0,
                    "features": [],
                }
        elif data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        elif data_path.suffix == ".csv":
            df = pd.read_csv(data_path)
        else:
            return {
                "is_valid": False,
                "errors": [f"Unsupported file format: {data_path.suffix}"],
                "num_samples": 0,
                "features": [],
            }
        
        result = self.validate(df)
        return result.model_dump()
