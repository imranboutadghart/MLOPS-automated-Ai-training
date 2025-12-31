"""
PyTorch Model Definitions
Configurable neural network architectures with factory pattern.
"""

from typing import Any, Optional

import torch
import torch.nn as nn
import structlog

logger = structlog.get_logger(__name__)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron classifier.
    
    Args:
        input_size: Number of input features
        hidden_sizes: List of hidden layer sizes
        output_size: Number of output classes
        dropout: Dropout probability
        activation: Activation function ('relu', 'gelu', 'leaky_relu')
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        dropout: float = 0.3,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Build layers
        layers = []
        prev_size = input_size
        
        # Activation function
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "silu": nn.SiLU(),
        }
        act_fn = activations.get(activation, nn.ReLU())
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                act_fn,
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class CNNClassifier(nn.Module):
    """
    CNN-based classifier for tabular or sequential data.
    
    Args:
        input_size: Number of input features
        num_classes: Number of output classes
        num_channels: List of channel sizes for conv layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        num_channels: list[int] = [32, 64, 128],
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_size = input_size
        
        # Reshape input for 1D convolution
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, num_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(num_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(num_channels[0], num_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(num_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(num_channels[1], num_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(num_channels[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels[-1], 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Reshape: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for tabular data.
    
    Args:
        input_size: Number of input features
        num_classes: Number of output classes
        d_model: Transformer dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Project input
        x = self.input_proj(x)
        # Add sequence dimension: (batch, features) -> (batch, 1, d_model)
        x = x.unsqueeze(1)
        # Transformer
        x = self.transformer(x)
        # Pool and classify
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)
        return x


class ModelFactory:
    """
    Factory for creating models based on configuration.
    
    Example:
        >>> factory = ModelFactory()
        >>> model = factory.create("mlp", input_size=100, output_size=10)
    """
    
    _models: dict[str, type] = {
        "mlp": MLP,
        "classifier": MLP,  # alias
        "cnn": CNNClassifier,
        "transformer": TransformerClassifier,
    }
    
    @classmethod
    def register(cls, name: str, model_class: type) -> None:
        """Register a new model type."""
        cls._models[name] = model_class
    
    @classmethod
    def create(
        cls,
        model_name: str,
        input_size: int,
        output_size: int,
        **kwargs: Any
    ) -> nn.Module:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model type
            input_size: Number of input features
            output_size: Number of output classes
            **kwargs: Additional model-specific arguments
            
        Returns:
            PyTorch model instance
        """
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls._models.keys())}")
        
        model_class = cls._models[model_name]
        
        logger.info("Creating model", model=model_name, input_size=input_size, output_size=output_size)
        
        if model_name in ("mlp", "classifier"):
            return model_class(
                input_size=input_size,
                hidden_sizes=kwargs.get("hidden_sizes", [256, 128, 64]),
                output_size=output_size,
                dropout=kwargs.get("dropout", 0.3),
                activation=kwargs.get("activation", "relu"),
            )
        elif model_name == "cnn":
            return model_class(
                input_size=input_size,
                num_classes=output_size,
                num_channels=kwargs.get("num_channels", [32, 64, 128]),
                dropout=kwargs.get("dropout", 0.3),
            )
        elif model_name == "transformer":
            return model_class(
                input_size=input_size,
                num_classes=output_size,
                d_model=kwargs.get("d_model", 128),
                nhead=kwargs.get("nhead", 4),
                num_layers=kwargs.get("num_layers", 2),
                dropout=kwargs.get("dropout", 0.3),
            )
        
        return model_class(input_size=input_size, output_size=output_size, **kwargs)
    
    @classmethod
    def list_models(cls) -> list[str]:
        """List available model types."""
        return list(cls._models.keys())


def create_model(config: dict[str, Any]) -> nn.Module:
    """
    Convenience function to create model from config dict.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        PyTorch model instance
    """
    return ModelFactory.create(
        model_name=config.get("name", "mlp"),
        input_size=config["input_size"],
        output_size=config["output_size"],
        **config.get("kwargs", {}),
    )
