"""
Pipeline Integration Tests
Tests for the distributed training pipeline components.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDataIngestion:
    """Tests for data ingestion module."""
    
    def test_load_local_csv(self, tmp_path):
        """Test loading CSV file."""
        from data.ingestion import DataIngestion
        
        # Create test CSV
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0],
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        
        # Load data
        ingestion = DataIngestion()
        loaded_df = ingestion.load_local(csv_path)
        
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ["feature1", "feature2", "target"]
    
    def test_load_local_parquet(self, tmp_path):
        """Test loading Parquet file."""
        from data.ingestion import DataIngestion
        
        # Create test Parquet
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0],
        })
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path, index=False)
        
        # Load data
        ingestion = DataIngestion()
        loaded_df = ingestion.load_local(parquet_path)
        
        assert len(loaded_df) == 3


class TestDataPreprocessing:
    """Tests for data preprocessing module."""
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        from data.preprocessing import DataPreprocessor
        
        df = pd.DataFrame({
            "numeric": [1, 2, np.nan, 4],
            "category": ["a", None, "b", "a"],
        })
        
        preprocessor = DataPreprocessor()
        result = preprocessor.handle_missing_values(df, strategy="mean")
        
        assert result["numeric"].isna().sum() == 0
        assert result["category"].isna().sum() == 0
    
    def test_encode_categorical(self):
        """Test categorical encoding."""
        from data.preprocessing import DataPreprocessor
        
        df = pd.DataFrame({
            "category": ["a", "b", "c", "a"],
        })
        
        preprocessor = DataPreprocessor()
        result = preprocessor.encode_categorical(df)
        
        assert result["category"].dtype in [np.int32, np.int64]
    
    def test_scale_features(self):
        """Test feature scaling."""
        from data.preprocessing import DataPreprocessor
        
        df = pd.DataFrame({
            "feature": [1, 2, 3, 4, 5],
        })
        
        preprocessor = DataPreprocessor()
        result = preprocessor.scale_features(df)
        
        # StandardScaler should give mean ~0 and std ~1
        assert abs(result["feature"].mean()) < 0.01
        assert abs(result["feature"].std() - 1) < 0.01


class TestDataValidation:
    """Tests for data validation module."""
    
    def test_validation_success(self):
        """Test successful validation."""
        from data.validation import DataValidator, DataSchema
        
        df = pd.DataFrame({
            "feature1": np.random.randn(1000),
            "feature2": np.random.randn(1000),
            "target": np.random.randint(0, 2, 1000),
        })
        
        schema = DataSchema(
            required_columns=["feature1", "feature2", "target"],
            numeric_columns=["feature1", "feature2"],
            min_rows=100,
        )
        
        validator = DataValidator(schema=schema)
        result = validator.validate(df)
        
        assert result.is_valid
        assert result.num_samples == 1000
    
    def test_validation_failure(self):
        """Test validation failure."""
        from data.validation import DataValidator, DataSchema
        
        df = pd.DataFrame({
            "feature1": np.random.randn(50),  # Too few rows
        })
        
        schema = DataSchema(min_rows=100)
        
        validator = DataValidator(schema=schema)
        result = validator.validate(df)
        
        assert not result.is_valid
        assert len(result.errors) > 0


class TestModels:
    """Tests for model definitions."""
    
    def test_mlp_forward(self):
        """Test MLP forward pass."""
        from training.models import MLP
        
        model = MLP(
            input_size=10,
            hidden_sizes=[64, 32],
            output_size=2,
        )
        
        x = torch.randn(4, 10)
        output = model(x)
        
        assert output.shape == (4, 2)
    
    def test_model_factory(self):
        """Test model factory."""
        from training.models import ModelFactory
        
        model = ModelFactory.create(
            model_name="mlp",
            input_size=10,
            output_size=2,
        )
        
        assert model is not None
        x = torch.randn(4, 10)
        output = model(x)
        assert output.shape == (4, 2)


class TestTrainer:
    """Tests for trainer module."""
    
    def test_training_config(self):
        """Test training configuration."""
        from training.trainer import TrainingConfig
        
        config = TrainingConfig(
            epochs=5,
            batch_size=32,
            learning_rate=0.001,
        )
        
        assert config.epochs == 5
        assert config.batch_size == 32
        assert config.learning_rate == 0.001


class TestEvaluation:
    """Tests for evaluation module."""
    
    def test_compute_metrics(self):
        """Test metric computation."""
        from training.evaluation import ModelEvaluator
        
        evaluator = ModelEvaluator(num_classes=2)
        
        predictions = np.array([0, 1, 0, 1, 1])
        targets = np.array([0, 1, 0, 0, 1])
        
        metrics = evaluator.compute_metrics(predictions, targets)
        
        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert metrics["accuracy"] == 0.8


class TestModelPromotion:
    """Tests for model promotion module."""
    
    def test_threshold_check_pass(self):
        """Test threshold check passes."""
        from registry.model_promotion import ModelPromoter, PromotionThresholds
        
        thresholds = PromotionThresholds(
            min_accuracy=0.80,
            min_f1_score=0.75,
        )
        promoter = ModelPromoter(thresholds=thresholds)
        
        metrics = {"accuracy": 0.90, "f1_score": 0.85}
        passes, reason = promoter.check_thresholds(metrics)
        
        assert passes
    
    def test_threshold_check_fail(self):
        """Test threshold check fails."""
        from registry.model_promotion import ModelPromoter, PromotionThresholds
        
        thresholds = PromotionThresholds(
            min_accuracy=0.90,
            min_f1_score=0.85,
        )
        promoter = ModelPromoter(thresholds=thresholds)
        
        metrics = {"accuracy": 0.80, "f1_score": 0.75}
        passes, reason = promoter.check_thresholds(metrics)
        
        assert not passes


class TestCanaryDeployment:
    """Tests for canary deployment module."""
    
    def test_health_evaluation_pass(self):
        """Test health evaluation passes."""
        from deployment.canary import CanaryDeployment, CanaryConfig, HealthStatus
        
        config = CanaryConfig(
            success_threshold=0.95,
            latency_threshold_ms=100,
        )
        deployer = CanaryDeployment(config=config)
        
        status = HealthStatus(
            is_healthy=True,
            success_rate=0.99,
            avg_latency_ms=50,
            error_count=1,
            total_requests=100,
        )
        
        passes, reason = deployer.evaluate_health(status)
        assert passes
    
    def test_health_evaluation_fail(self):
        """Test health evaluation fails."""
        from deployment.canary import CanaryDeployment, CanaryConfig, HealthStatus
        
        config = CanaryConfig(
            success_threshold=0.99,
            latency_threshold_ms=50,
        )
        deployer = CanaryDeployment(config=config)
        
        status = HealthStatus(
            is_healthy=True,
            success_rate=0.90,
            avg_latency_ms=100,
            error_count=10,
            total_requests=100,
        )
        
        passes, reason = deployer.evaluate_health(status)
        assert not passes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
