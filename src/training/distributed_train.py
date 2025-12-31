"""
Distributed Training Script
Main entry point for HuggingFace Accelerate distributed training.

Usage:
    accelerate launch --config_file configs/accelerate_config.yaml \
        src/training/distributed_train.py --config configs/training_config.yaml
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml
import structlog

from training.models import ModelFactory
from training.trainer import AccelerateTrainer, TrainingConfig
from training.evaluation import ModelEvaluator

logger = structlog.get_logger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_dataloaders(
    data_path: str = None,
    train_path: str = None,
    val_path: str = None,
    batch_size: int = 32,
    val_split: float = 0.1,
) -> tuple[DataLoader, DataLoader, int, int]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_path: Path to processed data (for auto-split)
        train_path: Path to training data (pre-split)
        val_path: Path to validation data (pre-split)
        batch_size: Batch size
        val_split: Validation split ratio (if using data_path)
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, input_size, num_classes)
    """
    
    def load_dataset(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
        path = Path(path)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.is_dir():
            dfs = []
            for f in path.glob("*.parquet"):
                dfs.append(pd.read_parquet(f))
            for f in path.glob("*.csv"):
                dfs.append(pd.read_csv(f))
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_csv(path)
            
        X = df.iloc[:, :-1].values.astype("float32")
        y = df.iloc[:, -1].values.astype("int64")
        
        # Filter invalid targets
        valid_mask = y != -1
        if not valid_mask.all():
            logger.warning(f"Dropping {len(y) - valid_mask.sum()} rows with invalid target -1", path=str(path))
            X = X[valid_mask]
            y = y[valid_mask]
            
        return torch.tensor(X), torch.tensor(y)

    if train_path and val_path:
        # Load pre-split data
        X_train, y_train = load_dataset(train_path)
        X_val, y_val = load_dataset(val_path)
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        input_size = X_train.shape[1]
        num_classes = len(torch.unique(y_train))
        
    elif data_path:
        # Load and split
        X, y = load_dataset(data_path)
        
        # Split into train and validation
        n_val = int(len(X) * val_split)
        indices = torch.randperm(len(X))
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        train_dataset = TensorDataset(X[train_indices], y[train_indices])
        val_dataset = TensorDataset(X[val_indices], y[val_indices])
        
        input_size = X.shape[1]
        num_classes = len(torch.unique(y))
    else:
        raise ValueError("Must provide either data_path or (train_path and val_path)")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    logger.info(
        "Dataloaders created",
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
    )
    
    return train_loader, val_loader, input_size, num_classes


def setup_mlflow(config: dict[str, Any], run_name: str) -> str:
    """
    Set up MLflow tracking.
    
    Args:
        config: Training configuration
        run_name: Name for the run
        
    Returns:
        Run ID
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = config.get("experiment_name", "distributed_training")
    
    # Handle race condition in distributed training
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
    except Exception:
        # Experiment already exists (race condition), continue
        pass
    
    mlflow.set_experiment(experiment_name)
    
    mlflow.start_run(run_name=run_name)
    
    # Log configuration
    mlflow.log_params({
        "model_name": config["model"]["name"],
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
        "learning_rate": config["training"]["learning_rate"],
    })
    
    return mlflow.active_run().info.run_id


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Distributed Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to training data (overrides config)",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for MLflow run",
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with CLI arguments
    data_path = args.data_path or config.get("data_path")
    train_path = args.train_data or config.get("train_path")
    val_path = args.val_data or config.get("val_path")
    
    output_dir = args.output_dir or config.get("output_dir", "./checkpoints")
    run_name = args.run_name or f"training_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info("Starting distributed training", config=args.config)
    
    # Setup MLflow
    run_id = setup_mlflow(config, run_name)
    logger.info("MLflow run started", run_id=run_id)
    
    try:
        # Create dataloaders
        train_loader, val_loader, input_size, num_classes = create_dataloaders(
            data_path=data_path,
            train_path=train_path,
            val_path=val_path,
            batch_size=config["training"]["batch_size"],
            val_split=config["training"].get("val_split", 0.1),
        )
        
        # Create model
        model = ModelFactory.create(
            model_name=config["model"]["name"],
            input_size=input_size,
            output_size=num_classes,
            hidden_sizes=config["model"].get("hidden_sizes", [256, 128, 64]),
            dropout=config["model"].get("dropout", 0.3),
        )
        
        logger.info(
            "Model created",
            model=config["model"]["name"],
            input_size=input_size,
            output_size=num_classes,
            parameters=sum(p.numel() for p in model.parameters()),
        )
        
        # Create training config
        training_config = TrainingConfig(
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"],
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"].get("weight_decay", 0.01),
            warmup_steps=config["scheduler"].get("warmup_steps", 100),
            gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
            checkpoint_every=config["training"].get("checkpoint_every", 1000),
            eval_every=config["training"].get("eval_every", 500),
            output_dir=output_dir,
            mixed_precision=config["training"].get("mixed_precision", "fp16"),
            scheduler_type=config["scheduler"].get("type", "cosine"),
        )
        
        # Create trainer
        trainer = AccelerateTrainer(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=val_loader,
            config=training_config,
        )
        
        # Train
        results = trainer.train()
        
        # Log final metrics to MLflow
        final_metrics = results["final_metrics"]
        mlflow.log_metrics({
            "final_train_loss": final_metrics.get("train_loss", 0),
            "final_accuracy": final_metrics.get("accuracy", 0),
            "final_f1_score": final_metrics.get("f1_score", 0),
        })
        
        # Log model artifact
        mlflow.pytorch.log_model(
            trainer.accelerator.unwrap_model(trainer.model),
            "model",
            registered_model_name=config.get("model_registry_name"),
        )
        
        # Save results
        output = {
            "model_path": str(Path(output_dir) / "best_model"),
            "run_id": run_id,
            "metrics": final_metrics,
        }
        
        # Print JSON output for DAG parsing
        print(json.dumps(output))
        
        logger.info("Training completed", **output)
        
    except Exception as e:
        logger.error("Training failed", error=str(e))
        mlflow.log_param("status", "failed")
        mlflow.log_param("error", str(e))
        raise
    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()
