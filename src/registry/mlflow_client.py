"""
MLflow Client Module
Integration with MLflow for model versioning, tracking, and artifact storage.
"""

import os
from pathlib import Path
from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
import structlog

logger = structlog.get_logger(__name__)


class MLflowClientWrapper:
    """
    Wrapper for MLflow client with convenient methods for model management.
    
    Features:
    - Model registration with versioning
    - Metric and parameter logging
    - Artifact storage
    - Model stage transitions
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
    ):
        """
        Initialize MLflow client.
        
        Args:
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
        """
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self.registry_uri = registry_uri or self.tracking_uri
        
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
        logger.info("MLflow client initialized", tracking_uri=self.tracking_uri)
    
    def create_experiment(
        self,
        name: str,
        artifact_location: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Create or get experiment.
        
        Args:
            name: Experiment name
            artifact_location: Artifact storage location
            tags: Experiment tags
            
        Returns:
            Experiment ID
        """
        experiment = mlflow.get_experiment_by_name(name)
        
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name,
                artifact_location=artifact_location,
                tags=tags,
            )
            logger.info("Created experiment", name=name, id=experiment_id)
        else:
            experiment_id = experiment.experiment_id
            logger.info("Using existing experiment", name=name, id=experiment_id)
        
        return experiment_id
    
    def start_run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Start a new MLflow run.
        
        Args:
            experiment_name: Name of the experiment
            run_name: Optional run name
            tags: Run tags
            
        Returns:
            Run ID
        """
        mlflow.set_experiment(experiment_name)
        
        run = mlflow.start_run(run_name=run_name, tags=tags)
        run_id = run.info.run_id
        
        logger.info("Started MLflow run", run_id=run_id, experiment=experiment_name)
        return run_id
    
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to current run."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to current run."""
        # Convert non-string values to strings
        params = {k: str(v) if not isinstance(v, (str, int, float)) else v 
                  for k, v in params.items()}
        mlflow.log_params(params)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact file."""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
    ) -> Optional[ModelVersion]:
        """
        Log PyTorch model.
        
        Args:
            model: PyTorch model
            artifact_path: Path within artifacts
            registered_model_name: If provided, register model
            
        Returns:
            ModelVersion if registered, else None
        """
        result = mlflow.pytorch.log_model(
            model,
            artifact_path,
            registered_model_name=registered_model_name,
        )
        
        if registered_model_name:
            logger.info(
                "Model logged and registered",
                name=registered_model_name,
                version=result.registered_model_version,
            )
        
        return result
    
    def register_model(
        self,
        model_path: str,
        name: str,
        run_id: str,
        metrics: Optional[dict[str, float]] = None,
    ) -> str:
        """
        Register a model from a run.
        
        Args:
            model_path: Path to model artifacts
            name: Model name in registry
            run_id: Run ID containing the model
            metrics: Optional metrics to log
            
        Returns:
            Model version
        """
        model_uri = f"runs:/{run_id}/model"
        
        # Register model
        model_version = mlflow.register_model(model_uri, name)
        
        # Add metrics as version description
        if metrics:
            description = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.client.update_model_version(
                name=name,
                version=model_version.version,
                description=description,
            )
        
        logger.info(
            "Model registered",
            name=name,
            version=model_version.version,
        )
        
        return model_version.version
    
    def transition_model_stage(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing: bool = True,
    ) -> ModelVersion:
        """
        Transition model to a new stage.
        
        Args:
            name: Model name
            version: Model version
            stage: Target stage (Staging, Production, Archived)
            archive_existing: Archive models currently in target stage
            
        Returns:
            Updated ModelVersion
        """
        model_version = self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )
        
        logger.info(
            "Model stage transitioned",
            name=name,
            version=version,
            stage=stage,
        )
        
        return model_version
    
    def get_latest_version(
        self,
        name: str,
        stage: str = "Production",
    ) -> Optional[ModelVersion]:
        """
        Get latest model version in a stage.
        
        Args:
            name: Model name
            stage: Model stage
            
        Returns:
            ModelVersion or None
        """
        try:
            versions = self.client.get_latest_versions(name, stages=[stage])
            return versions[0] if versions else None
        except Exception as e:
            logger.warning("Failed to get model version", error=str(e))
            return None
    
    def load_model(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Any:
        """
        Load a model from registry.
        
        Args:
            name: Model name
            version: Specific version (optional)
            stage: Stage to load from (optional)
            
        Returns:
            Loaded model
        """
        if version:
            model_uri = f"models:/{name}/{version}"
        elif stage:
            model_uri = f"models:/{name}/{stage}"
        else:
            model_uri = f"models:/{name}/latest"
        
        model = mlflow.pytorch.load_model(model_uri)
        logger.info("Model loaded", name=name, uri=model_uri)
        
        return model
    
    def compare_models(
        self,
        name: str,
        version_a: str,
        version_b: str,
    ) -> dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            name: Model name
            version_a: First version
            version_b: Second version
            
        Returns:
            Comparison results
        """
        # Get model versions
        mv_a = self.client.get_model_version(name, version_a)
        mv_b = self.client.get_model_version(name, version_b)
        
        # Get runs
        run_a = self.client.get_run(mv_a.run_id)
        run_b = self.client.get_run(mv_b.run_id)
        
        # Compare metrics
        metrics_a = run_a.data.metrics
        metrics_b = run_b.data.metrics
        
        comparison = {
            "version_a": version_a,
            "version_b": version_b,
            "metrics_comparison": {},
        }
        
        all_metrics = set(metrics_a.keys()) | set(metrics_b.keys())
        for metric in all_metrics:
            val_a = metrics_a.get(metric, None)
            val_b = metrics_b.get(metric, None)
            if val_a is not None and val_b is not None:
                comparison["metrics_comparison"][metric] = {
                    "version_a": val_a,
                    "version_b": val_b,
                    "diff": val_b - val_a,
                    "improvement": val_b > val_a,
                }
        
        return comparison
    
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""
        mlflow.end_run(status=status)


# Convenience instance
_client: Optional[MLflowClientWrapper] = None


def get_mlflow_client() -> MLflowClientWrapper:
    """Get global MLflow client instance."""
    global _client
    if _client is None:
        _client = MLflowClientWrapper()
    return _client
