"""
Monitoring and Alerting Module

Provides notification and alert functionality for the ML pipeline.
"""

from typing import Any
from datetime import datetime
import os

import structlog

logger = structlog.get_logger(__name__)


def send_notification(
    channel: str,
    message: str,
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> bool:
    """Send a notification through the specified channel.
    
    Args:
        channel: Notification channel (slack, email, webhook).
        message: Notification message.
        context: Optional Airflow context.
        **kwargs: Additional channel-specific parameters.
        
    Returns:
        True if notification was sent successfully.
    """
    logger.info(
        "Sending notification",
        channel=channel,
        message_preview=message[:100] if len(message) > 100 else message,
    )
    
    if channel == "slack":
        return _send_slack_notification(message, **kwargs)
    elif channel == "email":
        return _send_email_notification(message, **kwargs)
    elif channel == "webhook":
        return _send_webhook_notification(message, **kwargs)
    else:
        logger.warning("Unknown notification channel", channel=channel)
        return False


def send_alert(
    level: str,
    message: str,
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> bool:
    """Send an alert for important events.
    
    Args:
        level: Alert level (info, warning, error, critical).
        message: Alert message.
        context: Optional Airflow context.
        **kwargs: Additional parameters.
        
    Returns:
        True if alert was sent successfully.
    """
    logger.log(
        level.upper(),
        "Alert triggered",
        alert_level=level,
        message=message,
    )
    
    # Format alert message
    timestamp = datetime.now().isoformat()
    formatted_message = f"[{level.upper()}] {timestamp}\n\n{message}"
    
    # Add context if available
    if context:
        dag_id = context.get("dag", {}).dag_id if hasattr(context.get("dag", {}), "dag_id") else "unknown"
        task_id = context.get("task", {}).task_id if hasattr(context.get("task", {}), "task_id") else "unknown"
        run_id = context.get("run_id", "unknown")
        
        formatted_message += f"\n\nDAG: {dag_id}\nTask: {task_id}\nRun: {run_id}"
    
    # Send to configured channels based on severity
    channels = _get_channels_for_level(level)
    
    success = True
    for channel in channels:
        if not send_notification(channel, formatted_message, context, **kwargs):
            success = False
    
    return success


def _send_slack_notification(message: str, **kwargs: Any) -> bool:
    """Send notification to Slack."""
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    
    if not webhook_url:
        logger.debug("Slack webhook not configured, skipping notification")
        return True  # Not a failure if not configured
    
    try:
        import httpx
        
        payload = {
            "text": message,
            "username": kwargs.get("username", "ML Pipeline"),
            "icon_emoji": kwargs.get("icon", ":robot_face:"),
        }
        
        if kwargs.get("channel"):
            payload["channel"] = kwargs["channel"]
        
        response = httpx.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        
        logger.debug("Slack notification sent successfully")
        return True
    except Exception as e:
        logger.error("Failed to send Slack notification", error=str(e))
        return False


def _send_email_notification(message: str, **kwargs: Any) -> bool:
    """Send notification via email."""
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    
    if not all([smtp_host, smtp_user, smtp_password]):
        logger.debug("Email not configured, skipping notification")
        return True
    
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        recipients = kwargs.get("recipients", [os.environ.get("ALERT_EMAIL", "mlops@example.com")])
        subject = kwargs.get("subject", "ML Pipeline Notification")
        
        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))
        
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, recipients, msg.as_string())
        
        logger.debug("Email notification sent successfully")
        return True
    except Exception as e:
        logger.error("Failed to send email notification", error=str(e))
        return False


def _send_webhook_notification(message: str, **kwargs: Any) -> bool:
    """Send notification to a custom webhook."""
    webhook_url = kwargs.get("webhook_url") or os.environ.get("ALERT_WEBHOOK_URL")
    
    if not webhook_url:
        logger.debug("Webhook not configured, skipping notification")
        return True
    
    try:
        import httpx
        
        payload = {
            "message": message,
            "timestamp": datetime.now().isoformat(),
            **kwargs.get("extra_fields", {}),
        }
        
        response = httpx.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        
        logger.debug("Webhook notification sent successfully")
        return True
    except Exception as e:
        logger.error("Failed to send webhook notification", error=str(e))
        return False


def _get_channels_for_level(level: str) -> list[str]:
    """Get notification channels based on alert level.
    
    Args:
        level: Alert level.
        
    Returns:
        List of channel names.
    """
    level = level.lower()
    
    # Configure channels per level
    channel_config = {
        "info": ["slack"],
        "warning": ["slack"],
        "error": ["slack", "email"],
        "critical": ["slack", "email", "webhook"],
    }
    
    return channel_config.get(level, ["slack"])


class MetricsCollector:
    """Collect and expose metrics for monitoring."""

    def __init__(self, prefix: str = "mlops"):
        """Initialize metrics collector.
        
        Args:
            prefix: Metric name prefix.
        """
        self.prefix = prefix
        self._metrics: dict[str, Any] = {}
        
        try:
            from prometheus_client import Counter, Gauge, Histogram
            
            self.training_runs = Counter(
                f"{prefix}_training_runs_total",
                "Total number of training runs",
                ["status"],
            )
            self.training_duration = Histogram(
                f"{prefix}_training_duration_seconds",
                "Training duration in seconds",
            )
            self.model_accuracy = Gauge(
                f"{prefix}_model_accuracy",
                "Current model accuracy",
                ["model_name", "model_version"],
            )
            self.deployment_status = Gauge(
                f"{prefix}_deployment_status",
                "Deployment status (1=active, 0=inactive)",
                ["deployment_id", "strategy"],
            )
            self._prometheus_available = True
        except ImportError:
            self._prometheus_available = False
            logger.debug("Prometheus client not available")

    def record_training_run(self, status: str, duration_seconds: float) -> None:
        """Record a training run.
        
        Args:
            status: Run status (success, failure).
            duration_seconds: Training duration.
        """
        if self._prometheus_available:
            self.training_runs.labels(status=status).inc()
            self.training_duration.observe(duration_seconds)

    def update_model_accuracy(
        self,
        model_name: str,
        model_version: str,
        accuracy: float,
    ) -> None:
        """Update model accuracy metric.
        
        Args:
            model_name: Model name.
            model_version: Model version.
            accuracy: Model accuracy.
        """
        if self._prometheus_available:
            self.model_accuracy.labels(
                model_name=model_name,
                model_version=model_version,
            ).set(accuracy)

    def update_deployment_status(
        self,
        deployment_id: str,
        strategy: str,
        active: bool,
    ) -> None:
        """Update deployment status metric.
        
        Args:
            deployment_id: Deployment identifier.
            strategy: Deployment strategy (canary, shadow).
            active: Whether deployment is active.
        """
        if self._prometheus_available:
            self.deployment_status.labels(
                deployment_id=deployment_id,
                strategy=strategy,
            ).set(1 if active else 0)
