"""
Continuous Training DAG

Main orchestration DAG for distributed model training with automatic scheduling.
Supports hourly, daily, and weekly retraining with HuggingFace Accelerate.
"""

from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.celery.sensors.celery_queue import CeleryQueueSensor
from airflow.utils.trigger_rule import TriggerRule

# Default arguments for the DAG
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email": ["mlops@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=6),
}

# Training schedules
SCHEDULES = {
    "hourly": "0 * * * *",      # Every hour
    "daily": "0 2 * * *",        # Daily at 2 AM
    "weekly": "0 3 * * 0",       # Weekly on Sunday at 3 AM
}


def get_schedule_interval() -> str:
    """Get schedule interval from Airflow variables or default to daily."""
    from airflow.models import Variable
    return Variable.get("training_schedule", default_var=SCHEDULES["daily"])


with DAG(
    dag_id="continuous_training_dag",
    default_args=default_args,
    description="Distributed Continuous Training Pipeline with HuggingFace Accelerate",
    schedule_interval=get_schedule_interval(),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["training", "ml", "distributed"],
    doc_md="""
    # Continuous Training DAG
    
    This DAG orchestrates the complete ML training pipeline:
    1. Data validation and preprocessing
    2. Distributed training with HuggingFace Accelerate
    3. Model evaluation
    4. Automatic registration to MLflow
    5. Triggering deployment pipeline on success
    
    ## Configuration
    Set the `training_schedule` Airflow variable to one of:
    - `0 * * * *` (hourly)
    - `0 2 * * *` (daily)
    - `0 3 * * 0` (weekly)
    """,
) as dag:

    @task(task_id="validate_data")
    def validate_data(**context) -> dict[str, Any]:
        """Validate input data quality before training."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        
        from data.validation import DataValidator, DataSchema
        from utils.config import load_config
        
        config = load_config("/opt/airflow/configs/training_config.yaml")
        
        # Initialize with custom schema from config
        validation_config = config.get("validation", {})
        schema = DataSchema(**validation_config)
        validator = DataValidator(schema=schema)
        
        validation_result = validator.validate_training_data(config["data"]["raw_path"])
        
        if not validation_result["is_valid"]:
            raise ValueError(f"Data validation failed: {validation_result['errors']}")
        
        return {
            "num_samples": validation_result["num_samples"],
            "features": validation_result["features"],
            "validation_time": datetime.now().isoformat(),
        }

    @task(task_id="preprocess_data")
    def preprocess_data(validation_result: dict[str, Any], **context) -> dict[str, Any]:
        """Preprocess data using Pandas pipeline."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        
        from data.preprocessing import DataPreprocessor
        from utils.config import load_config
        
        config = load_config("/opt/airflow/configs/training_config.yaml")
        pipeline = DataPreprocessor(config)
        
        result = pipeline.run()
        
        return {
            "train_path": result["train_path"],
            "val_path": result["val_path"],
            "test_path": result["test_path"],
            "preprocessing_time": datetime.now().isoformat(),
            "num_train_samples": result["num_train_samples"],
            "num_val_samples": result["num_val_samples"],
        }

    @task(task_id="run_distributed_training")
    def run_distributed_training(preprocessing_result: dict[str, Any], **context) -> dict[str, Any]:
        """Launch distributed training with HuggingFace Accelerate."""
        import subprocess
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        
        from utils.config import load_config
        import mlflow
        
        config = load_config("/opt/airflow/configs/training_config.yaml")
        accelerate_config = "/opt/airflow/configs/accelerate_config.yaml"
        
        # Generate unique run name
        run_name = f"training_{context['ds_nodash']}_{context['run_id'][:8]}"
        
        # Build accelerate command
        cmd = [
            "accelerate", "launch",
            "--config_file", accelerate_config,
            "-m", "training.distributed_train",
            "--config", "/opt/airflow/configs/training_config.yaml",
            "--train-data", preprocessing_result["train_path"],
            "--val-data", preprocessing_result["val_path"],
            "--run-name", run_name,
        ]
        
        # Execute distributed training
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/opt/airflow/src",
            env={
                **dict(__import__("os").environ),
                "PYTHONPATH": "/opt/airflow/src",
            },
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Training failed: {result.stderr}")
        
        # Parse training output for metrics
        return {
            "run_name": run_name,
            "training_completed": datetime.now().isoformat(),
            "stdout": result.stdout[-2000:],  # Last 2000 chars
        }

    @task(task_id="evaluate_model")
    def evaluate_model(training_result: dict[str, Any], preprocessing_result: dict[str, Any], **context) -> dict[str, Any]:
        """Evaluate trained model on test set."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        
        from training.evaluation import ModelEvaluator
        from utils.config import load_config
        import mlflow
        
        config = load_config("/opt/airflow/configs/training_config.yaml")
        
        # Get the latest model from MLflow
        mlflow.set_tracking_uri("http://mlflow:5000")
        
        evaluator = ModelEvaluator(config)
        metrics = evaluator.evaluate_from_mlflow(
            model_run_name=training_result["run_name"],
            test_data_path=preprocessing_result["test_path"],
            device="cpu",
        )
        
        return {
            "run_name": training_result["run_name"],
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "loss": metrics["loss"],
            "evaluation_time": datetime.now().isoformat(),
        }

    @task(task_id="register_model")
    def register_model(evaluation_result: dict[str, Any], **context) -> dict[str, Any]:
        """Register model to MLflow if it passes quality thresholds."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        
        from registry.model_promotion import ModelPromoter
        from utils.config import load_config
        import mlflow
        
        config = load_config("/opt/airflow/configs/training_config.yaml")
        
        # Get the model from MLflow using run name
        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.tracking.MlflowClient()
        
        # Find the run by name
        run_name = evaluation_result["run_name"]
        experiment = client.get_experiment_by_name("distributed_training")
        if not experiment:
            raise ValueError("Experiment 'distributed_training' not found")
            
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            max_results=1,
        )
        
        if not runs:
            raise ValueError(f"No run found with name: {run_name}")
        
        run = runs[0]
        run_id = run.info.run_id
        
        # Get model name and version from the run
        model_name = config.get("model", {}).get("name", "titanic_classifier")
        
        # Register the model if not already registered
        model_uri = f"runs:/{run_id}/model"
        registered_model = mlflow.register_model(model_uri, model_name)
        model_version = registered_model.version
        
        # Now evaluate and promote
        promoter = ModelPromoter()
        registration_result = promoter.evaluate_and_promote(
            model_name=model_name,
            model_version=model_version,
            metrics=evaluation_result,
        )
        
        return {
            "model_name": model_name,
            "model_version": model_version,
            "stage": registration_result["stage"],
            "promoted": registration_result["promoted"],
            "registration_time": datetime.now().isoformat(),
        }

    @task(task_id="notify_completion")
    def notify_completion(registration_result: dict[str, Any], **context) -> None:
        """Send notification on training completion."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        
        from utils.monitoring import send_notification
        
        message = f"""
        Training Pipeline Completed Successfully!
        
        Model: {registration_result['model_name']}
        Version: {registration_result['model_version']}
        Stage: {registration_result['stage']}
        Promoted: {registration_result['promoted']}
        Time: {registration_result['registration_time']}
        """
        
        send_notification(
            channel="slack",  # or "email"
            message=message,
            context=context,
        )

    # Trigger deployment if model was promoted
    trigger_deployment = TriggerDagRunOperator(
        task_id="trigger_deployment",
        trigger_dag_id="deployment_dag",
        conf={"model_version": "{{ ti.xcom_pull(task_ids='register_model')['model_version'] }}"},
        trigger_rule=TriggerRule.ALL_SUCCESS,
        wait_for_completion=False,
    )

    # Define task dependencies
    validation = validate_data()
    preprocessing = preprocess_data(validation)
    training = run_distributed_training(preprocessing)
    evaluation = evaluate_model(training, preprocessing)
    registration = register_model(evaluation)
    notification = notify_completion(registration)
    
    registration >> trigger_deployment
