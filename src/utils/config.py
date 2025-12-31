"""
Configuration Management Module

Provides configuration loading and validation.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import structlog

logger = structlog.get_logger(__name__)


class ModelConfig(BaseModel):
    """Model configuration."""
    name: str = "mlp"
    hidden_sizes: list[int] = [512, 256, 128]
    dropout: float = 0.3
    activation: str = "relu"
    use_batch_norm: bool = True


class TrainingConfig(BaseModel):
    """Training configuration."""
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    checkpoint_every_n_epochs: int = 1
    val_split: float = 0.1
    mixed_precision: str = "fp16"
    checkpoint_every: int = 1000
    eval_every: int = 500
    
    class Config:
        extra = "allow"


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""
    type: str = "cosine"
    warmup_steps: int = 100
    min_lr: float = 1e-6


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""
    raw_data_dir: str = "/opt/airflow/data/raw"
    output_dir: str = "/opt/airflow/data/processed"
    target_column: str = "target"
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
    remove_outliers: bool = False
    create_interactions: bool = False
    polynomial_degree: int = 1


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    task_type: str = "classification"
    batch_size: int = 256
    metrics: list[str] = ["accuracy", "f1_score", "precision", "recall"]


class MLflowConfig(BaseModel):
    """MLflow configuration."""
    tracking_uri: str = "http://mlflow:5000"
    experiment_name: str = "continuous-training"
    model_name: str = "ml-model"
    model_registry_name: str = "production_classifier"
    artifact_location: str = "s3://mlflow/artifacts"
    
    class Config:
        extra = "allow"


class PromotionConfig(BaseModel):
    """Model promotion configuration."""
    accuracy_threshold: float = 0.8
    f1_threshold: float = 0.75
    min_accuracy: float = 0.85
    min_f1_score: float = 0.80
    improvement_threshold: float = 0.01
    compare_with_production: bool = True
    minimum_improvement: float = 0.01
    
    class Config:
        extra = "allow"


class CanaryConfig(BaseModel):
    """Canary deployment configuration."""
    stages: list[int] = [1, 5, 25, 50, 100]
    stage_duration_minutes: int = 30
    health_threshold: float = 0.95
    error_rate_threshold: float = 0.05
    latency_threshold_ms: int = 500


class ShadowConfig(BaseModel):
    """Shadow deployment configuration."""
    duration_hours: int = 2
    sample_rate: float = 1.0
    prediction_diff_threshold: float = 0.1


class ServingConfig(BaseModel):
    """Model serving configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4


class DataPreprocessingConfig(BaseModel):
    """Data preprocessing sub-configuration."""
    missing_strategy: str = "mean"
    encoding_method: str = "label"
    scaling_method: str = "standard"


class DataConfig(BaseModel):
    """Data paths configuration."""
    raw_path: str = "/opt/airflow/data/raw"
    train_path: str = "/opt/airflow/data/processed/training"
    val_path: str = "/opt/airflow/data/processed/validation"
    test_path: str = "/opt/airflow/data/processed/test"
    preprocessing: DataPreprocessingConfig = Field(default_factory=DataPreprocessingConfig)


class OutputConfig(BaseModel):
    """Output paths configuration."""
    checkpoint_dir: str = "/app/checkpoints"
    log_dir: str = "/app/logs"


class Config(BaseSettings):
    """Main configuration class."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    canary: CanaryConfig = Field(default_factory=CanaryConfig)
    shadow: ShadowConfig = Field(default_factory=ShadowConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    class Config:
        env_prefix = "MLOPS_"
        env_nested_delimiter = "__"
        extra = "allow"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    if path is None:
        path = Path("/opt/airflow/configs/training_config.yaml")
    else:
        path = Path(path)
    
    if not path.exists():
        logger.warning("Config file not found, using defaults", path=str(path))
        return Config().model_dump()
    
    with open(path) as f:
        raw_config = yaml.safe_load(f) or {}
    
    logger.info("Loaded configuration", path=str(path))
    
    # Merge with defaults
    config = Config(**raw_config)
    return config.model_dump()


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary.
        path: Path to save to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info("Saved configuration", path=str(path))
