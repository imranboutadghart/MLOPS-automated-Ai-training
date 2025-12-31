"""
Model Evaluation Module
Metrics computation and model evaluation utilities.
"""

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import structlog

logger = structlog.get_logger(__name__)


class ModelEvaluator:
    """
    Model evaluation class with comprehensive metrics.
    
    Supports:
    - Classification metrics (accuracy, F1, precision, recall)
    - Multi-class and binary classification
    - Confusion matrix
    - ROC-AUC score
    """
    
    def __init__(
        self,
        config: dict = None,
        num_classes: int = 2,
        average: str = "weighted",
    ):
        """
        Initialize evaluator.
        
        Args:
            config: Optional training configuration dict
            num_classes: Number of classes
            average: Averaging strategy for multi-class metrics
        """
        if config is not None:
            self.num_classes = config.get("model", {}).get("output_size", num_classes)
            self.average = average
        else:
            self.num_classes = num_classes
            self.average = average
    
    def compute_metrics(
        self,
        predictions: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
        include_confusion_matrix: bool = False,
    ) -> dict[str, Any]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Model predictions (class indices or probabilities)
            targets: Ground truth labels
            include_confusion_matrix: Whether to include confusion matrix
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Handle probability predictions
        if predictions.ndim > 1:
            probs = predictions
            predictions = predictions.argmax(axis=1)
        else:
            probs = None
        
        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(targets, predictions),
            "f1_score": f1_score(targets, predictions, average=self.average, zero_division=0),
            "precision": precision_score(targets, predictions, average=self.average, zero_division=0),
            "recall": recall_score(targets, predictions, average=self.average, zero_division=0),
        }
        
        # Per-class metrics for multi-class
        if self.num_classes > 2:
            metrics["f1_per_class"] = f1_score(
                targets, predictions, average=None, zero_division=0
            ).tolist()
        
        # ROC-AUC for binary classification or if probabilities available
        if probs is not None:
            try:
                # Check if we have enough classes and valid probability shape
                unique_classes = len(np.unique(targets))
                if unique_classes < 2:
                    logger.warning("Skipping ROC-AUC: only one class present in targets")
                elif self.num_classes == 2 and probs.shape[1] >= 2:
                    metrics["roc_auc"] = roc_auc_score(targets, probs[:, 1])
                elif probs.shape[1] >= 2:
                    metrics["roc_auc"] = roc_auc_score(
                        targets, probs, multi_class="ovr", average=self.average
                    )
            except (ValueError, IndexError) as e:
                logger.warning("Skipping ROC-AUC calculation", error=str(e))
        
        # Confusion matrix
        if include_confusion_matrix:
            metrics["confusion_matrix"] = confusion_matrix(targets, predictions).tolist()
        
        logger.info("Metrics computed", **{k: v for k, v in metrics.items() if not isinstance(v, list)})
        return metrics
    
    def evaluate(
        self,
        model_path: str,
        data_path: str,
        metrics: Optional[list[str]] = None,
        device: str = "cuda",
    ) -> dict[str, float]:
        """
        Evaluate a saved model on data.
        
        Args:
            model_path: Path to saved model
            data_path: Path to evaluation data
            metrics: List of metrics to compute
            device: Device to use for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        import pandas as pd
        from pathlib import Path
        from .models import ModelFactory
        
        logger.info("Loading model for evaluation", model_path=model_path)
        
        # Load model checkpoint
        model_path = Path(model_path)
        model_state = torch.load(model_path / "model.pt", map_location=device)
        
        # Load model config (if available)
        config_path = model_path / "config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                model_config = json.load(f)
            model = ModelFactory.create(**model_config)
        else:
            # Infer from state dict
            input_size = list(model_state.values())[0].shape[1]
            output_size = list(model_state.values())[-1].shape[0]
            model = ModelFactory.create("mlp", input_size=input_size, output_size=output_size)
        
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()
        
        # Load evaluation data
        data_path = Path(data_path)
        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
        
        # Assume last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Run inference
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(device)
            outputs = model(inputs)
            predictions = torch.softmax(outputs, dim=-1)
        
        return self.compute_metrics(predictions, y, include_confusion_matrix=True)
    
    def evaluate_from_mlflow(
        self,
        model_run_name: str,
        test_data_path: str,
        device: str = "cpu",
    ) -> dict[str, float]:
        """
        Evaluate a model from MLflow run.
        
        Args:
            model_run_name: MLflow run name
            test_data_path: Path to test data
            device: Device to use for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        import mlflow
        import pandas as pd
        from pathlib import Path
        
        logger.info("Loading model from MLflow", run_name=model_run_name)
        
        # Get run by name
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("distributed_training")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = '{model_run_name}'",
                max_results=1,
            )
            if not runs:
                raise ValueError(f"No run found with name: {model_run_name}")
            run_id = runs[0].info.run_id
        else:
            raise ValueError("Experiment 'distributed_training' not found")
        
        # Load model from MLflow
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
        model = model.to(device)
        model.eval()
        
        # Load evaluation data
        data_path = Path(test_data_path)
        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
        
        # Assume last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Filter out invalid targets (NaN, -1, out of bounds)
        valid_mask = (y >= 0) & (y < self.num_classes) & (~pd.isna(y))
        if not valid_mask.all():
            logger.warning(
                "Found invalid targets, filtering them out",
                invalid_count=int((~valid_mask).sum()),
                total_count=len(y)
            )
            X = X[valid_mask]
            y = y[valid_mask]
        
        if len(y) == 0:
            raise ValueError("No valid samples found in test data after filtering")
        
        # Run inference
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(device)
            outputs = model(inputs)
            predictions_probs = torch.softmax(outputs, dim=-1)
        
        # Calculate loss
        criterion = nn.CrossEntropyLoss()
        targets_tensor = torch.tensor(y, dtype=torch.long).to(device)
        loss = criterion(outputs, targets_tensor).item()
        
        # Convert predictions to numpy and ensure integer targets
        predictions_np = predictions_probs.cpu().numpy()
        y = y.astype(np.int64)
        
        metrics = self.compute_metrics(predictions_np, y, include_confusion_matrix=False)
        metrics["loss"] = loss
        metrics["num_samples"] = len(y)
        
        logger.info("Evaluation completed", **{k: v for k, v in metrics.items() if not isinstance(v, list)})
        return metrics
    
    def generate_report(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        class_names: Optional[list[str]] = None,
    ) -> str:
        """
        Generate detailed classification report.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            class_names: Optional names for classes
            
        Returns:
            Classification report string
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        if predictions.ndim > 1:
            predictions = predictions.argmax(axis=1)
        
        return classification_report(targets, predictions, target_names=class_names)


def compute_training_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """
    Compute metrics during training.
    
    Args:
        outputs: Model outputs (logits)
        targets: Ground truth labels
        
    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        predictions = torch.argmax(outputs, dim=-1)
        accuracy = (predictions == targets).float().mean().item()
    
    return {"accuracy": accuracy}
