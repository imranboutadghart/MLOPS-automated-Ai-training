"""
Shadow Deployment Module

Implements shadow/dark launch deployments for risk-free testing.
"""

from typing import Any
from datetime import datetime
from collections import defaultdict

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class ShadowDeployment:
    """Manages shadow deployments for parallel inference and comparison."""

    def __init__(self, config: dict[str, Any]):
        """Initialize shadow deployment manager.
        
        Args:
            config: Deployment configuration.
        """
        self.config = config
        self.shadow_config = config.get("shadow", {})
        self.comparison_duration_hours = self.shadow_config.get("duration_hours", 2)
        self.sample_rate = self.shadow_config.get("sample_rate", 1.0)  # Percentage of requests to shadow
        self.prediction_diff_threshold = self.shadow_config.get("prediction_diff_threshold", 0.1)
        
        # Service endpoints
        self.deployment_service_url = config.get(
            "deployment_service_url", "http://localhost:8000"
        )
        
        # Active deployments and comparison data
        self._active_deployments: dict[str, dict[str, Any]] = {}
        self._comparison_data: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def initialize(
        self,
        model_name: str,
        model_version: str,
        deployment_id: str,
    ) -> dict[str, Any]:
        """Initialize a new shadow deployment.
        
        Args:
            model_name: Name of the model in the registry.
            model_version: Version to deploy as shadow.
            deployment_id: Unique deployment identifier.
            
        Returns:
            Deployment information.
        """
        logger.info(
            "Initializing shadow deployment",
            model_name=model_name,
            model_version=model_version,
            deployment_id=deployment_id,
        )
        
        shadow_endpoint = f"{self.deployment_service_url}/shadow/{deployment_id}"
        production_endpoint = f"{self.deployment_service_url}/production"
        
        deployment_info = {
            "deployment_id": deployment_id,
            "model_name": model_name,
            "model_version": model_version,
            "shadow_endpoint": shadow_endpoint,
            "production_endpoint": production_endpoint,
            "sample_rate": self.sample_rate,
            "status": "initialized",
            "created_at": datetime.now().isoformat(),
            "comparisons_collected": 0,
        }
        
        self._active_deployments[deployment_id] = deployment_info
        self._comparison_data[deployment_id] = []
        
        # In a real implementation, this would:
        # 1. Deploy shadow model alongside production
        # 2. Configure middleware to duplicate requests
        # 3. Set up comparison logging
        
        logger.info("Shadow deployment initialized", deployment_id=deployment_id)
        
        return deployment_info

    def log_comparison(
        self,
        deployment_id: str,
        request_id: str,
        production_prediction: Any,
        shadow_prediction: Any,
        production_latency_ms: float,
        shadow_latency_ms: float,
        input_features: dict[str, Any] | None = None,
    ) -> None:
        """Log a prediction comparison between production and shadow.
        
        Args:
            deployment_id: Deployment identifier.
            request_id: Unique request identifier.
            production_prediction: Prediction from production model.
            shadow_prediction: Prediction from shadow model.
            production_latency_ms: Production inference latency.
            shadow_latency_ms: Shadow inference latency.
            input_features: Optional input features for debugging.
        """
        if deployment_id not in self._active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        comparison = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "production_prediction": production_prediction,
            "shadow_prediction": shadow_prediction,
            "production_latency_ms": production_latency_ms,
            "shadow_latency_ms": shadow_latency_ms,
            "predictions_match": self._predictions_match(
                production_prediction, shadow_prediction
            ),
        }
        
        if input_features:
            comparison["input_features"] = input_features
        
        self._comparison_data[deployment_id].append(comparison)
        self._active_deployments[deployment_id]["comparisons_collected"] += 1

    def get_comparison_metrics(self, deployment_id: str) -> dict[str, Any]:
        """Get aggregated comparison metrics.
        
        Args:
            deployment_id: Deployment identifier.
            
        Returns:
            Comparison metrics.
        """
        if deployment_id not in self._active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        comparisons = self._comparison_data[deployment_id]
        
        if not comparisons:
            return {
                "deployment_id": deployment_id,
                "total_comparisons": 0,
                "message": "No comparisons collected yet",
            }
        
        # Calculate metrics
        total = len(comparisons)
        matches = sum(1 for c in comparisons if c["predictions_match"])
        
        prod_latencies = [c["production_latency_ms"] for c in comparisons]
        shadow_latencies = [c["shadow_latency_ms"] for c in comparisons]
        
        metrics = {
            "deployment_id": deployment_id,
            "total_comparisons": total,
            "prediction_match_rate": matches / total if total > 0 else 0,
            "prediction_mismatch_count": total - matches,
            "production_latency": {
                "p50": np.percentile(prod_latencies, 50),
                "p95": np.percentile(prod_latencies, 95),
                "p99": np.percentile(prod_latencies, 99),
                "mean": np.mean(prod_latencies),
            },
            "shadow_latency": {
                "p50": np.percentile(shadow_latencies, 50),
                "p95": np.percentile(shadow_latencies, 95),
                "p99": np.percentile(shadow_latencies, 99),
                "mean": np.mean(shadow_latencies),
            },
            "latency_improvement": (
                np.mean(prod_latencies) - np.mean(shadow_latencies)
            ) / np.mean(prod_latencies) if np.mean(prod_latencies) > 0 else 0,
            "collected_at": datetime.now().isoformat(),
        }
        
        return metrics

    def analyze_comparison(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Analyze comparison metrics and provide recommendations.
        
        Args:
            metrics: Comparison metrics from get_comparison_metrics.
            
        Returns:
            Analysis results with recommendation.
        """
        analysis = {
            "deployment_id": metrics.get("deployment_id"),
            "total_comparisons": metrics.get("total_comparisons", 0),
            "issues": [],
            "recommendation": "promote",
            "should_promote": True,
            "confidence": 1.0,
        }
        
        # Check prediction match rate
        match_rate = metrics.get("prediction_match_rate", 0)
        if match_rate < 1 - self.prediction_diff_threshold:
            analysis["issues"].append(
                f"Low prediction match rate: {match_rate:.2%} "
                f"(threshold: {1 - self.prediction_diff_threshold:.2%})"
            )
            analysis["confidence"] -= 0.3
        
        # Check latency
        shadow_p95 = metrics.get("shadow_latency", {}).get("p95", 0)
        prod_p95 = metrics.get("production_latency", {}).get("p95", 0)
        
        if shadow_p95 > prod_p95 * 1.2:  # 20% latency regression
            analysis["issues"].append(
                f"Latency regression: shadow P95 {shadow_p95:.1f}ms vs production {prod_p95:.1f}ms"
            )
            analysis["confidence"] -= 0.2
        
        # Check sample size
        if metrics.get("total_comparisons", 0) < 1000:
            analysis["issues"].append(
                f"Low sample size: {metrics.get('total_comparisons', 0)} comparisons"
            )
            analysis["confidence"] -= 0.1
        
        # Final recommendation
        if analysis["issues"]:
            if analysis["confidence"] < 0.5:
                analysis["recommendation"] = "do_not_promote"
                analysis["should_promote"] = False
            else:
                analysis["recommendation"] = "promote_with_caution"
        
        logger.info(
            "Shadow analysis complete",
            recommendation=analysis["recommendation"],
            confidence=analysis["confidence"],
            issues=analysis["issues"],
        )
        
        return analysis

    def stop(self, deployment_id: str) -> dict[str, Any]:
        """Stop a shadow deployment.
        
        Args:
            deployment_id: Deployment identifier.
            
        Returns:
            Stop status.
        """
        if deployment_id not in self._active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self._active_deployments[deployment_id]
        deployment["status"] = "stopped"
        deployment["stopped_at"] = datetime.now().isoformat()
        
        # In a real implementation, this would:
        # 1. Stop shadow inference
        # 2. Remove traffic mirroring
        # 3. Archive comparison logs
        
        logger.info(
            "Shadow deployment stopped",
            deployment_id=deployment_id,
            comparisons_collected=deployment["comparisons_collected"],
        )
        
        return {
            "deployment_id": deployment_id,
            "status": "stopped",
            "comparisons_collected": deployment["comparisons_collected"],
            "stopped_at": deployment["stopped_at"],
        }

    def _predictions_match(self, pred1: Any, pred2: Any) -> bool:
        """Check if two predictions match within threshold.
        
        Args:
            pred1: First prediction.
            pred2: Second prediction.
            
        Returns:
            True if predictions match.
        """
        if isinstance(pred1, (int, float)) and isinstance(pred2, (int, float)):
            return abs(pred1 - pred2) <= self.prediction_diff_threshold
        elif isinstance(pred1, (list, np.ndarray)) and isinstance(pred2, (list, np.ndarray)):
            arr1 = np.array(pred1)
            arr2 = np.array(pred2)
            if arr1.shape != arr2.shape:
                return False
            return np.allclose(arr1, arr2, atol=self.prediction_diff_threshold)
        else:
            return pred1 == pred2

    def get_mismatch_samples(
        self, deployment_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get sample mismatched predictions for debugging.
        
        Args:
            deployment_id: Deployment identifier.
            limit: Maximum number of samples to return.
            
        Returns:
            List of mismatch samples.
        """
        if deployment_id not in self._comparison_data:
            return []
        
        mismatches = [
            c for c in self._comparison_data[deployment_id]
            if not c["predictions_match"]
        ]
        
        return mismatches[:limit]
