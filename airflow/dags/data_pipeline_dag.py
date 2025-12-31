"""
Data Pipeline DAG

Orchestrates data ingestion, preprocessing, and validation.
Can be triggered independently or as part of the training pipeline.
"""

from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email": ["mlops@example.com"],
    "email_on_failure": True,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
    dag_id="data_pipeline_dag",
    default_args=default_args,
    description="Data ingestion, preprocessing, and validation pipeline",
    schedule_interval="0 1 * * *",  # Daily at 1 AM (before training)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["data", "preprocessing", "etl"],
) as dag:

    @task(task_id="ingest_data")
    def ingest_data(**context) -> dict[str, Any]:
        """Ingest data from configured sources."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        
        from data.ingestion import DataIngestion
        from utils.config import load_config
        
        config = load_config("/opt/airflow/configs/training_config.yaml")
        ingester = DataIngestion(config)
        
        result = ingester.ingest()
        
        return {
            "raw_data_path": result["output_path"],
            "num_records": result["num_records"],
            "sources": result["sources"],
            "ingestion_time": datetime.now().isoformat(),
        }

    @task(task_id="validate_raw_data")
    def validate_raw_data(ingestion_result: dict[str, Any], **context) -> dict[str, Any]:
        """Validate raw data quality."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        
        from data.validation import DataValidator
        from utils.config import load_config
        
        config = load_config("/opt/airflow/configs/training_config.yaml")
        validator = DataValidator(config)
        
        result = validator.validate_raw(ingestion_result["raw_data_path"])
        
        return {
            "is_valid": result["is_valid"],
            "quality_score": result["quality_score"],
            "issues": result.get("issues", []),
            "validation_time": datetime.now().isoformat(),
        }

    def check_data_quality(**context) -> str:
        """Branch based on data quality check."""
        ti = context["ti"]
        validation_result = ti.xcom_pull(task_ids="validate_raw_data")
        
        if validation_result["is_valid"] and validation_result["quality_score"] >= 0.8:
            return "preprocess_data"
        else:
            return "handle_quality_issues"

    branch_quality = BranchPythonOperator(
        task_id="branch_quality_check",
        python_callable=check_data_quality,
    )

    @task(task_id="handle_quality_issues")
    def handle_quality_issues(**context) -> None:
        """Handle data quality issues - log and alert."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        
        from utils.monitoring import send_alert
        
        ti = context["ti"]
        validation_result = ti.xcom_pull(task_ids="validate_raw_data")
        
        send_alert(
            level="warning",
            message=f"Data quality issues detected: {validation_result['issues']}",
            context=context,
        )

    @task(task_id="preprocess_data")
    def preprocess_data(**context) -> dict[str, Any]:
        """Run Pandas preprocessing pipeline."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        
        from data.preprocessing import DataPreprocessor
        from utils.config import load_config
        
        config = load_config("/opt/airflow/configs/training_config.yaml")
        
        ti = context["ti"]
        ingestion_result = ti.xcom_pull(task_ids="ingest_data")
        
        pipeline = DataPreprocessor(config)
        result = pipeline.run(input_path=ingestion_result["raw_data_path"])
        
        return {
            "train_path": result["train_path"],
            "val_path": result["val_path"],
            "test_path": result["test_path"],
            "num_train_samples": result["num_train_samples"],
            "num_val_samples": result["num_val_samples"],
            "num_test_samples": result["num_test_samples"],
            "preprocessing_time": datetime.now().isoformat(),
        }

    @task(task_id="update_feature_store")
    def update_feature_store(preprocessing_result: dict[str, Any], **context) -> dict[str, Any]:
        """Update feature store with new processed features."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        
        # Feature store update logic would go here
        # For now, we just log the update
        
        return {
            "feature_version": context["ds_nodash"],
            "updated_at": datetime.now().isoformat(),
            "train_samples": preprocessing_result["num_train_samples"],
        }

    @task(task_id="validate_processed_data", trigger_rule=TriggerRule.ALL_SUCCESS)
    def validate_processed_data(preprocessing_result: dict[str, Any], **context) -> dict[str, Any]:
        """Final validation of processed data."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        
        from data.validation import DataValidator
        from utils.config import load_config
        
        config = load_config("/opt/airflow/configs/training_config.yaml")
        validator = DataValidator(config)
        
        result = validator.validate_processed(
            train_path=preprocessing_result["train_path"],
            val_path=preprocessing_result["val_path"],
        )
        
        if not result["is_valid"]:
            raise ValueError(f"Processed data validation failed: {result['errors']}")
        
        return {
            "is_valid": result["is_valid"],
            "stats": result["stats"],
            "ready_for_training": True,
        }

    # Task dependencies
    ingestion = ingest_data()
    raw_validation = validate_raw_data(ingestion)
    
    raw_validation >> branch_quality
    branch_quality >> [preprocess_data.override(task_id="preprocess_data")(), handle_quality_issues()]
    
    preprocessing = preprocess_data()
    feature_update = update_feature_store(preprocessing)
    final_validation = validate_processed_data(preprocessing)
    
    preprocessing >> [feature_update, final_validation]
