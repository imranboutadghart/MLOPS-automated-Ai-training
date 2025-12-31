"""
Deployment DAG

Orchestrates canary and shadow deployment strategies.
Triggered automatically by the training pipeline on model promotion.
"""

from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import BranchPythonOperator
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email": ["mlops@example.com"],
    "email_on_failure": True,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

# Canary deployment stages (traffic percentages)
CANARY_STAGES = [1, 5, 25, 50, 100]
CANARY_STAGE_DURATION_MINUTES = 30

with DAG(
    dag_id="deployment_dag",
    default_args=default_args,
    description="Canary and Shadow Deployment Pipeline",
    schedule_interval=None,  # Triggered by training DAG
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["deployment", "canary", "shadow"],
    doc_md="""
    # Deployment DAG
    
    This DAG handles model deployment with two strategies:
    
    ## Canary Deployment
    - Gradual traffic shifting: 1% → 5% → 25% → 50% → 100%
    - Automatic rollback on metric degradation
    - Each stage runs for 30 minutes by default
    
    ## Shadow Deployment
    - Parallel inference without affecting production traffic
    - Comparison of predictions for analysis
    - No production impact
    
    ## Triggering
    This DAG is triggered by the training DAG when a model is promoted.
    Configuration is passed via `dag_run.conf`:
    - `model_version`: The MLflow model version to deploy
    - `deployment_strategy`: "canary" or "shadow" (default: canary)
    """,
) as dag:

    @task(task_id="get_deployment_config")
    def get_deployment_config(**context) -> dict[str, Any]:
        """Get deployment configuration from trigger."""
        dag_run = context["dag_run"]
        conf = dag_run.conf or {}
        
        model_version = conf.get("model_version")
        if not model_version:
            raise ValueError("model_version must be provided in dag_run.conf")
        
        strategy = conf.get("deployment_strategy", "canary")
        
        return {
            "model_version": model_version,
            "strategy": strategy,
            "model_name": Variable.get("model_name", default_var="ml-model"),
            "deployment_id": f"deploy_{context['ds_nodash']}_{model_version}",
        }

    def select_deployment_strategy(**context) -> str:
        """Branch to appropriate deployment strategy."""
        ti = context["ti"]
        config = ti.xcom_pull(task_ids="get_deployment_config")
        
        if config["strategy"] == "shadow":
            return "start_shadow_deployment"
        return "start_canary_deployment"

    branch_strategy = BranchPythonOperator(
        task_id="select_strategy",
        python_callable=select_deployment_strategy,
    )

    # ==================== CANARY DEPLOYMENT ====================

    @task(task_id="start_canary_deployment")
    def start_canary_deployment(**context) -> dict[str, Any]:
        """Initialize canary deployment."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        
        from deployment.canary import CanaryDeployment
        from utils.config import load_config
        
        ti = context["ti"]
        config = ti.xcom_pull(task_ids="get_deployment_config")
        
        deployment_config = load_config("/opt/airflow/configs/deployment_config.yaml")
        canary = CanaryDeployment(deployment_config)
        
        result = canary.initialize(
            model_name=config["model_name"],
            model_version=config["model_version"],
            deployment_id=config["deployment_id"],
        )
        
        return {
            "deployment_id": result["deployment_id"],
            "canary_endpoint": result["canary_endpoint"],
            "production_endpoint": result["production_endpoint"],
            "started_at": datetime.now().isoformat(),
        }

    @task(task_id="canary_stage_1")
    def canary_stage_1(deployment_info: dict[str, Any], **context) -> dict[str, Any]:
        """Canary Stage 1: 1% traffic."""
        import sys
        import time
        sys.path.insert(0, "/opt/airflow/src")
        
        from deployment.canary import CanaryDeployment
        from utils.config import load_config
        
        deployment_config = load_config("/opt/airflow/configs/deployment_config.yaml")
        canary = CanaryDeployment(deployment_config)
        
        # Shift to 1% traffic
        canary.shift_traffic(deployment_info["deployment_id"], percentage=1)
        
        # Monitor for stage duration
        time.sleep(CANARY_STAGE_DURATION_MINUTES * 60)
        
        # Check metrics
        metrics = canary.get_metrics(deployment_info["deployment_id"])
        
        if not canary.check_health(metrics):
            canary.rollback(deployment_info["deployment_id"])
            raise ValueError(f"Canary failed at 1%: {metrics}")
        
        return {"stage": 1, "percentage": 1, "metrics": metrics}

    @task(task_id="canary_stage_5")
    def canary_stage_5(stage_1_result: dict[str, Any], deployment_info: dict[str, Any], **context) -> dict[str, Any]:
        """Canary Stage 2: 5% traffic."""
        import sys
        import time
        sys.path.insert(0, "/opt/airflow/src")
        
        from deployment.canary import CanaryDeployment
        from utils.config import load_config
        
        deployment_config = load_config("/opt/airflow/configs/deployment_config.yaml")
        canary = CanaryDeployment(deployment_config)
        
        canary.shift_traffic(deployment_info["deployment_id"], percentage=5)
        time.sleep(CANARY_STAGE_DURATION_MINUTES * 60)
        
        metrics = canary.get_metrics(deployment_info["deployment_id"])
        
        if not canary.check_health(metrics):
            canary.rollback(deployment_info["deployment_id"])
            raise ValueError(f"Canary failed at 5%: {metrics}")
        
        return {"stage": 2, "percentage": 5, "metrics": metrics}

    @task(task_id="canary_stage_25")
    def canary_stage_25(stage_5_result: dict[str, Any], deployment_info: dict[str, Any], **context) -> dict[str, Any]:
        """Canary Stage 3: 25% traffic."""
        import sys
        import time
        sys.path.insert(0, "/opt/airflow/src")
        
        from deployment.canary import CanaryDeployment
        from utils.config import load_config
        
        deployment_config = load_config("/opt/airflow/configs/deployment_config.yaml")
        canary = CanaryDeployment(deployment_config)
        
        canary.shift_traffic(deployment_info["deployment_id"], percentage=25)
        time.sleep(CANARY_STAGE_DURATION_MINUTES * 60)
        
        metrics = canary.get_metrics(deployment_info["deployment_id"])
        
        if not canary.check_health(metrics):
            canary.rollback(deployment_info["deployment_id"])
            raise ValueError(f"Canary failed at 25%: {metrics}")
        
        return {"stage": 3, "percentage": 25, "metrics": metrics}

    @task(task_id="canary_stage_100")
    def canary_stage_100(stage_25_result: dict[str, Any], deployment_info: dict[str, Any], **context) -> dict[str, Any]:
        """Canary Stage 4: 100% traffic (full rollout)."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        
        from deployment.canary import CanaryDeployment
        from utils.config import load_config
        
        deployment_config = load_config("/opt/airflow/configs/deployment_config.yaml")
        canary = CanaryDeployment(deployment_config)
        
        canary.shift_traffic(deployment_info["deployment_id"], percentage=100)
        canary.finalize_deployment(deployment_info["deployment_id"])
        
        return {
            "stage": 4,
            "percentage": 100,
            "status": "completed",
            "finalized_at": datetime.now().isoformat(),
        }

    # ==================== SHADOW DEPLOYMENT ====================

    @task(task_id="start_shadow_deployment")
    def start_shadow_deployment(**context) -> dict[str, Any]:
        """Initialize shadow deployment."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        
        from deployment.shadow import ShadowDeployment
        from utils.config import load_config
        
        ti = context["ti"]
        config = ti.xcom_pull(task_ids="get_deployment_config")
        
        deployment_config = load_config("/opt/airflow/configs/deployment_config.yaml")
        shadow = ShadowDeployment(deployment_config)
        
        result = shadow.initialize(
            model_name=config["model_name"],
            model_version=config["model_version"],
            deployment_id=config["deployment_id"],
        )
        
        return {
            "deployment_id": result["deployment_id"],
            "shadow_endpoint": result["shadow_endpoint"],
            "production_endpoint": result["production_endpoint"],
            "started_at": datetime.now().isoformat(),
        }

    @task(task_id="run_shadow_comparison")
    def run_shadow_comparison(shadow_info: dict[str, Any], **context) -> dict[str, Any]:
        """Run shadow deployment for comparison period."""
        import sys
        import time
        sys.path.insert(0, "/opt/airflow/src")
        
        from deployment.shadow import ShadowDeployment
        from utils.config import load_config
        
        deployment_config = load_config("/opt/airflow/configs/deployment_config.yaml")
        shadow = ShadowDeployment(deployment_config)
        
        # Run shadow for configured duration (default 2 hours)
        shadow_duration = Variable.get("shadow_duration_hours", default_var=2)
        time.sleep(int(shadow_duration) * 3600)
        
        # Collect comparison metrics
        comparison = shadow.get_comparison_metrics(shadow_info["deployment_id"])
        
        return {
            "deployment_id": shadow_info["deployment_id"],
            "comparison_metrics": comparison,
            "completed_at": datetime.now().isoformat(),
        }

    @task(task_id="analyze_shadow_results")
    def analyze_shadow_results(comparison_result: dict[str, Any], **context) -> dict[str, Any]:
        """Analyze shadow deployment results."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        
        from deployment.shadow import ShadowDeployment
        from utils.config import load_config
        
        deployment_config = load_config("/opt/airflow/configs/deployment_config.yaml")
        shadow = ShadowDeployment(deployment_config)
        
        analysis = shadow.analyze_comparison(comparison_result["comparison_metrics"])
        
        # Stop shadow deployment
        shadow.stop(comparison_result["deployment_id"])
        
        return {
            "analysis": analysis,
            "recommendation": analysis["recommendation"],
            "promote_to_production": analysis["should_promote"],
        }

    # ==================== COMMON TASKS ====================

    @task(task_id="notify_deployment_complete", trigger_rule=TriggerRule.ONE_SUCCESS)
    def notify_deployment_complete(**context) -> None:
        """Send notification on deployment completion."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        
        from utils.monitoring import send_notification
        
        ti = context["ti"]
        config = ti.xcom_pull(task_ids="get_deployment_config")
        
        # Try to get results from either strategy
        canary_result = ti.xcom_pull(task_ids="canary_stage_100")
        shadow_result = ti.xcom_pull(task_ids="analyze_shadow_results")
        
        if canary_result:
            message = f"Canary deployment completed for model version {config['model_version']}"
        elif shadow_result:
            message = f"Shadow deployment analysis complete. Recommendation: {shadow_result['recommendation']}"
        else:
            message = f"Deployment completed for model version {config['model_version']}"
        
        send_notification(channel="slack", message=message, context=context)

    # Task dependencies
    deployment_config = get_deployment_config()
    deployment_config >> branch_strategy

    # Canary path
    canary_start = start_canary_deployment()
    stage_1 = canary_stage_1(canary_start)
    stage_5 = canary_stage_5(stage_1, canary_start)
    stage_25 = canary_stage_25(stage_5, canary_start)
    stage_100 = canary_stage_100(stage_25, canary_start)

    branch_strategy >> canary_start

    # Shadow path
    shadow_start = start_shadow_deployment()
    shadow_comparison = run_shadow_comparison(shadow_start)
    shadow_analysis = analyze_shadow_results(shadow_comparison)

    branch_strategy >> shadow_start

    # Common completion
    notification = notify_deployment_complete()
    stage_100 >> notification
    shadow_analysis >> notification
