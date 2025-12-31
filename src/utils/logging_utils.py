"""
Logging Utilities Module

Provides structured logging configuration.
"""

import sys
import logging
from typing import Any

import structlog


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure structured logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: Whether to output JSON formatted logs.
        log_file: Optional file path to write logs.
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )
    
    # Configure structlog
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name.
        
    Returns:
        Structlog bound logger.
    """
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding temporary logging context."""

    def __init__(self, **kwargs: Any):
        """Initialize with context key-value pairs."""
        self.context = kwargs
        self._token = None

    def __enter__(self) -> "LogContext":
        """Enter context and bind variables."""
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context and unbind variables."""
        if self._token:
            structlog.contextvars.unbind_contextvars(*self.context.keys())


class TrainingLogger:
    """Specialized logger for training metrics."""

    def __init__(self, experiment_name: str):
        """Initialize training logger.
        
        Args:
            experiment_name: Name of the experiment.
        """
        self.logger = structlog.get_logger("training")
        self.experiment_name = experiment_name
        self.step = 0
        self.epoch = 0

    def log_epoch_start(self, epoch: int) -> None:
        """Log the start of an epoch."""
        self.epoch = epoch
        self.logger.info("Epoch started", epoch=epoch, experiment=self.experiment_name)

    def log_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log the end of an epoch with metrics."""
        self.logger.info(
            "Epoch completed",
            epoch=epoch,
            experiment=self.experiment_name,
            **metrics,
        )

    def log_step(
        self,
        step: int,
        loss: float,
        learning_rate: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log a training step."""
        self.step = step
        self.logger.debug(
            "Training step",
            step=step,
            epoch=self.epoch,
            loss=loss,
            learning_rate=learning_rate,
            **kwargs,
        )

    def log_evaluation(self, metrics: dict[str, float]) -> None:
        """Log evaluation metrics."""
        self.logger.info(
            "Evaluation completed",
            epoch=self.epoch,
            experiment=self.experiment_name,
            **metrics,
        )

    def log_checkpoint(self, path: str) -> None:
        """Log checkpoint saved."""
        self.logger.info(
            "Checkpoint saved",
            path=path,
            epoch=self.epoch,
            step=self.step,
        )
