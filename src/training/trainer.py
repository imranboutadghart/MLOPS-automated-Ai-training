"""
HuggingFace Accelerate Trainer
Distributed Data Parallel training with mixed precision support.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    checkpoint_every: int = 1000
    eval_every: int = 500
    log_every: int = 100
    seed: int = 42
    output_dir: str = "./checkpoints"
    mixed_precision: str = "fp16"
    scheduler_type: str = "cosine"


class AccelerateTrainer:
    """
    Trainer using HuggingFace Accelerate for distributed training.
    
    Features:
    - Multi-GPU DDP training
    - Mixed precision (FP16/BF16)
    - Gradient accumulation
    - Automatic checkpointing
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        loss_fn: Optional[nn.Module] = None,
        metrics_fn: Optional[Callable] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            config: Training configuration
            loss_fn: Loss function
            metrics_fn: Function to compute metrics
        """
        self.config = config or TrainingConfig()
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.metrics_fn = metrics_fn
        
        # Set seed for reproducibility
        set_seed(self.config.seed)
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
            log_with="mlflow" if os.getenv("MLFLOW_TRACKING_URI") else None,
        )
        
        # Initialize optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler(
            len(train_dataloader) * self.config.epochs
        )
        
        # Prepare for distributed training
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            model, self.optimizer, train_dataloader, eval_dataloader, self.scheduler
        )
        
        # Training state
        self.global_step = 0
        self.best_metric = float("-inf")
        
        # Output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "Trainer initialized",
            distributed=self.accelerator.distributed_type,
            num_processes=self.accelerator.num_processes,
            mixed_precision=self.config.mixed_precision,
        )
    
    def _create_scheduler(self, total_steps: int):
        """Create learning rate scheduler with warmup."""
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )
        
        if self.config.scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - self.config.warmup_steps,
                eta_min=self.config.learning_rate * 0.01,
            )
        else:
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps - self.config.warmup_steps,
            )
        
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.config.warmup_steps],
        )
    
    def train_epoch(self, epoch: int) -> dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_local_main_process,
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model):
                # Forward pass
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.detach().float()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.log_every == 0:
                avg_loss = total_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})
                
                if self.accelerator.is_main_process:
                    self.accelerator.log(
                        {"train_loss": avg_loss, "learning_rate": lr},
                        step=self.global_step,
                    )
            
            # Evaluation
            if self.eval_dataloader and self.global_step % self.config.eval_every == 0:
                eval_metrics = self.evaluate()
                if self.accelerator.is_main_process:
                    self.accelerator.log(eval_metrics, step=self.global_step)
                self.model.train()
            
            # Checkpointing
            if self.global_step % self.config.checkpoint_every == 0:
                self.save_checkpoint(f"checkpoint-{self.global_step}")
        
        return {"train_loss": (total_loss / num_batches).item()}
    
    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """
        Evaluate model on eval dataloader.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch in self.eval_dataloader:
            inputs, targets = batch
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            
            total_loss += loss.detach().float()
            
            preds = torch.argmax(outputs, dim=-1)
            all_preds.append(self.accelerator.gather(preds))
            all_targets.append(self.accelerator.gather(targets))
        
        # Compute metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        metrics = {"eval_loss": (total_loss / len(self.eval_dataloader)).item()}
        
        if self.metrics_fn:
            metrics.update(self.metrics_fn(all_preds.cpu(), all_targets.cpu()))
        else:
            # Default accuracy
            accuracy = (all_preds == all_targets).float().mean().item()
            metrics["accuracy"] = accuracy
        
        logger.info("Evaluation completed", **metrics)
        return metrics
    
    def save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            checkpoint_dir = self.output_dir / name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped_model.state_dict(), checkpoint_dir / "model.pt")
            
            # Save optimizer and scheduler
            torch.save({
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
            }, checkpoint_dir / "training_state.pt")
            
            logger.info("Checkpoint saved", path=str(checkpoint_dir))
    
    def load_checkpoint(self, checkpoint_dir: str | Path) -> None:
        """Load model checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(torch.load(checkpoint_dir / "model.pt"))
        
        # Load training state
        training_state = torch.load(checkpoint_dir / "training_state.pt")
        self.optimizer.load_state_dict(training_state["optimizer"])
        self.scheduler.load_state_dict(training_state["scheduler"])
        self.global_step = training_state["global_step"]
        
        logger.info("Checkpoint loaded", path=str(checkpoint_dir), step=self.global_step)
    
    def train(self) -> dict[str, Any]:
        """
        Full training loop.
        
        Returns:
            Dictionary with training results
        """
        logger.info("Starting training", epochs=self.config.epochs)
        
        best_eval_metric = float("-inf")
        training_history = []
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_metrics = self.train_epoch(epoch)
            
            # Evaluate at end of epoch
            if self.eval_dataloader:
                eval_metrics = self.evaluate()
                epoch_metrics.update(eval_metrics)
                
                # Save best model
                current_metric = eval_metrics.get("accuracy", -eval_metrics.get("eval_loss", 0))
                if current_metric > best_eval_metric:
                    best_eval_metric = current_metric
                    self.save_checkpoint("best_model")
            
            training_history.append(epoch_metrics)
            
            if self.accelerator.is_main_process:
                logger.info(f"Epoch {epoch} completed", **epoch_metrics)
        
        # Save final model
        self.save_checkpoint("final_model")
        
        # End logging
        self.accelerator.end_training()
        
        return {
            "model_path": str(self.output_dir / "best_model"),
            "final_metrics": training_history[-1],
            "history": training_history,
        }
