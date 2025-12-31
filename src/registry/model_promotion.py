"""
Model Promotion Module
Automatic model promotion logic with champion/challenger comparison.
"""

from dataclasses import dataclass
from typing import Any, Optional

import structlog

from .mlflow_client import get_mlflow_client, MLflowClientWrapper

logger = structlog.get_logger(__name__)


@dataclass
class PromotionThresholds:
    """Thresholds for model promotion."""
    
    min_accuracy: float = 0.85
    min_f1_score: float = 0.80
    min_precision: float = 0.75
    min_recall: float = 0.75
    max_latency_ms: float = 100.0
    improvement_threshold: float = 0.01  # 1% improvement required


@dataclass
class PromotionResult:
    """Result of promotion evaluation."""
    
    promoted: bool
    stage: Optional[str] = None
    reason: str = ""
    metrics: dict[str, float] = None
    previous_version: Optional[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class ModelPromoter:
    """
    Automatic model promotion with champion/challenger comparison.
    
    Promotion flow:
    1. New model → Staging (if meets thresholds)
    2. Staging → Production (if outperforms current production)
    
    Features:
    - Performance threshold validation
    - Champion/challenger comparison
    - Automatic stage transitions
    - Audit logging
    """
    
    def __init__(
        self,
        client: Optional[MLflowClientWrapper] = None,
        thresholds: Optional[PromotionThresholds] = None,
    ):
        """
        Initialize promoter.
        
        Args:
            client: MLflow client
            thresholds: Promotion thresholds
        """
        self.client = client or get_mlflow_client()
        self.thresholds = thresholds or PromotionThresholds()
    
    def check_thresholds(self, metrics: dict[str, float]) -> tuple[bool, str]:
        """
        Check if metrics meet promotion thresholds.
        
        Args:
            metrics: Model metrics
            
        Returns:
            Tuple of (passes, reason)
        """
        checks = []
        
        if "accuracy" in metrics:
            if metrics["accuracy"] < self.thresholds.min_accuracy:
                checks.append(
                    f"Accuracy {metrics['accuracy']:.3f} < {self.thresholds.min_accuracy}"
                )
        
        if "f1_score" in metrics:
            if metrics["f1_score"] < self.thresholds.min_f1_score:
                checks.append(
                    f"F1 Score {metrics['f1_score']:.3f} < {self.thresholds.min_f1_score}"
                )
        
        if "precision" in metrics:
            if metrics["precision"] < self.thresholds.min_precision:
                checks.append(
                    f"Precision {metrics['precision']:.3f} < {self.thresholds.min_precision}"
                )
        
        if "recall" in metrics:
            if metrics["recall"] < self.thresholds.min_recall:
                checks.append(
                    f"Recall {metrics['recall']:.3f} < {self.thresholds.min_recall}"
                )
        
        if checks:
            return False, "; ".join(checks)
        
        return True, "All thresholds met"
    
    def compare_with_production(
        self,
        model_name: str,
        candidate_metrics: dict[str, float],
    ) -> tuple[bool, str, Optional[str]]:
        """
        Compare candidate model with current production model.
        
        Args:
            model_name: Name of the model
            candidate_metrics: Metrics of candidate model
            
        Returns:
            Tuple of (is_better, reason, production_version)
        """
        # Get current production model
        production = self.client.get_latest_version(model_name, stage="Production")
        
        if production is None:
            return True, "No production model exists", None
        
        # Get production model metrics
        run = self.client.client.get_run(production.run_id)
        prod_metrics = run.data.metrics
        
        # Compare primary metric (accuracy)
        primary_metric = "accuracy"
        if primary_metric not in candidate_metrics or primary_metric not in prod_metrics:
            return False, "Cannot compare: missing primary metric", production.version
        
        candidate_value = candidate_metrics[primary_metric]
        prod_value = prod_metrics.get(f"final_{primary_metric}", prod_metrics.get(primary_metric, 0))
        
        improvement = candidate_value - prod_value
        
        if improvement >= self.thresholds.improvement_threshold:
            return (
                True,
                f"Improvement of {improvement:.3f} ({improvement/prod_value*100:.1f}%)",
                production.version,
            )
        
        return (
            False,
            f"No significant improvement: {candidate_value:.3f} vs {prod_value:.3f}",
            production.version,
        )
    
    def promote_to_staging(
        self,
        model_name: str,
        model_version: str,
        metrics: dict[str, float],
    ) -> PromotionResult:
        """
        Attempt to promote model to Staging.
        
        Args:
            model_name: Model name
            model_version: Model version
            metrics: Model metrics
            
        Returns:
            PromotionResult
        """
        # Check thresholds
        passes, reason = self.check_thresholds(metrics)
        
        if not passes:
            logger.info(
                "Model failed threshold check",
                model=model_name,
                version=model_version,
                reason=reason,
            )
            return PromotionResult(
                promoted=False,
                reason=f"Threshold check failed: {reason}",
                metrics=metrics,
            )
        
        # Promote to staging
        self.client.transition_model_stage(
            name=model_name,
            version=model_version,
            stage="Staging",
        )
        
        logger.info(
            "Model promoted to Staging",
            model=model_name,
            version=model_version,
        )
        
        return PromotionResult(
            promoted=True,
            stage="Staging",
            reason="Passed all threshold checks",
            metrics=metrics,
        )
    
    def promote_to_production(
        self,
        model_name: str,
        model_version: str,
        metrics: dict[str, float],
    ) -> PromotionResult:
        """
        Attempt to promote model to Production.
        
        Args:
            model_name: Model name
            model_version: Model version
            metrics: Model metrics
            
        Returns:
            PromotionResult
        """
        # Compare with production
        is_better, reason, prod_version = self.compare_with_production(
            model_name, metrics
        )
        
        if not is_better:
            logger.info(
                "Model not promoted to Production",
                model=model_name,
                version=model_version,
                reason=reason,
            )
            return PromotionResult(
                promoted=False,
                reason=reason,
                metrics=metrics,
                previous_version=prod_version,
            )
        
        # Promote to production
        self.client.transition_model_stage(
            name=model_name,
            version=model_version,
            stage="Production",
            archive_existing=True,
        )
        
        logger.info(
            "Model promoted to Production",
            model=model_name,
            version=model_version,
            previous=prod_version,
        )
        
        return PromotionResult(
            promoted=True,
            stage="Production",
            reason=reason,
            metrics=metrics,
            previous_version=prod_version,
        )
    
    def evaluate_and_promote(
        self,
        model_name: str,
        model_version: str,
        metrics: dict[str, float],
        target_stage: str = "auto",
    ) -> dict[str, Any]:
        """
        Evaluate model and promote if appropriate.
        
        Args:
            model_name: Model name
            model_version: Model version
            metrics: Model metrics
            target_stage: Target stage ('Staging', 'Production', or 'auto')
            
        Returns:
            Dictionary with promotion results
        """
        logger.info(
            "Evaluating model for promotion",
            model=model_name,
            version=model_version,
            target=target_stage,
        )
        
        if target_stage == "Production":
            result = self.promote_to_production(model_name, model_version, metrics)
        elif target_stage == "Staging":
            result = self.promote_to_staging(model_name, model_version, metrics)
        else:
            # Auto mode: try staging first, then production
            staging_result = self.promote_to_staging(model_name, model_version, metrics)
            
            if staging_result.promoted:
                prod_result = self.promote_to_production(model_name, model_version, metrics)
                result = prod_result if prod_result.promoted else staging_result
            else:
                result = staging_result
        
        return {
            "promoted": result.promoted,
            "stage": result.stage,
            "reason": result.reason,
            "metrics": result.metrics,
            "previous_version": result.previous_version,
        }


def auto_promote(
    model_name: str,
    model_version: str,
    metrics: dict[str, float],
) -> dict[str, Any]:
    """
    Convenience function for automatic promotion.
    
    Args:
        model_name: Model name
        model_version: Model version
        metrics: Model metrics
        
    Returns:
        Promotion result dictionary
    """
    promoter = ModelPromoter()
    return promoter.evaluate_and_promote(model_name, model_version, metrics)
