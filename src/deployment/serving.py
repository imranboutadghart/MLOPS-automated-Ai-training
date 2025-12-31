"""
Model Serving Module

FastAPI-based model serving with support for multiple deployment strategies.
"""

from typing import Any
from datetime import datetime
from contextlib import asynccontextmanager
import asyncio

import numpy as np
import torch
import torch.nn as nn
import structlog
import mlflow

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = structlog.get_logger(__name__)


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: list[float] = Field(..., description="Input features")
    request_id: str | None = Field(None, description="Optional request ID")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    request_id: str
    prediction: list[float]
    probabilities: list[float] | None = None
    model_version: str
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    model_version: str | None
    uptime_seconds: float


class ModelServer:
    """FastAPI-based model server."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the model server.
        
        Args:
            config: Server configuration.
        """
        self.config = config
        self.server_config = config.get("serving", {})
        self.host = self.server_config.get("host", "0.0.0.0")
        self.port = self.server_config.get("port", 8000)
        
        self.model: nn.Module | None = None
        self.model_version: str | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_time = datetime.now()
        
        # Shadow model for parallel inference
        self.shadow_model: nn.Module | None = None
        self.shadow_version: str | None = None
        
        if FASTAPI_AVAILABLE:
            self.app = self._create_app()
        else:
            self.app = None

    def _create_app(self) -> "FastAPI":
        """Create FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("Starting model server")
            self._load_production_model()
            yield
            # Shutdown
            logger.info("Shutting down model server")
        
        app = FastAPI(
            title="ML Model Server",
            description="Production ML model serving with canary and shadow deployments",
            version="1.0.0",
            lifespan=lifespan,
        )
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            return HealthResponse(
                status="healthy" if self.model is not None else "degraded",
                model_loaded=self.model is not None,
                model_version=self.model_version,
                uptime_seconds=(datetime.now() - self.start_time).total_seconds(),
            )
        
        @app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            return await self._predict(request)
        
        @app.post("/production", response_model=PredictionResponse)
        async def predict_production(request: PredictionRequest):
            return await self._predict(request, use_shadow=False)
        
        @app.post("/shadow/{deployment_id}", response_model=dict)
        async def predict_shadow(deployment_id: str, request: PredictionRequest):
            """Parallel inference with production and shadow models."""
            if self.shadow_model is None:
                raise HTTPException(status_code=404, detail="Shadow model not loaded")
            
            # Run both predictions in parallel
            prod_task = asyncio.create_task(self._predict(request, use_shadow=False))
            shadow_task = asyncio.create_task(self._predict(request, use_shadow=True))
            
            prod_result, shadow_result = await asyncio.gather(prod_task, shadow_task)
            
            return {
                "request_id": request.request_id or "auto",
                "production": prod_result.dict(),
                "shadow": shadow_result.dict(),
                "deployment_id": deployment_id,
            }
        
        @app.post("/canary/{deployment_id}", response_model=PredictionResponse)
        async def predict_canary(deployment_id: str, request: PredictionRequest):
            """Canary prediction endpoint."""
            # In a real implementation, traffic routing would be handled by a load balancer
            # This endpoint just serves the shadow/canary model
            return await self._predict(request, use_shadow=True)
        
        @app.post("/reload")
        async def reload_model(model_name: str, model_version: str):
            """Reload model from registry."""
            try:
                self._load_model(model_name, model_version, is_shadow=False)
                return {"status": "success", "model_version": model_version}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/load-shadow")
        async def load_shadow(model_name: str, model_version: str):
            """Load a shadow model for comparison."""
            try:
                self._load_model(model_name, model_version, is_shadow=True)
                return {"status": "success", "shadow_version": model_version}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        return app

    async def _predict(
        self,
        request: PredictionRequest,
        use_shadow: bool = False,
    ) -> PredictionResponse:
        """Make a prediction."""
        start_time = datetime.now()
        
        model = self.shadow_model if use_shadow else self.model
        version = self.shadow_version if use_shadow else self.model_version
        
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )
        
        try:
            # Prepare input
            features = torch.tensor(request.features, dtype=torch.float32).unsqueeze(0)
            features = features.to(self.device)
            
            # Inference
            model.eval()
            with torch.no_grad():
                output = model(features)
            
            # Process output
            if output.shape[-1] > 1:
                # Classification
                probabilities = torch.softmax(output, dim=-1).cpu().numpy().tolist()[0]
                prediction = output.argmax(dim=-1).cpu().numpy().tolist()
            else:
                # Regression
                probabilities = None
                prediction = output.cpu().numpy().tolist()[0]
            
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            return PredictionResponse(
                request_id=request.request_id or f"req_{datetime.now().timestamp()}",
                prediction=prediction if isinstance(prediction, list) else [prediction],
                probabilities=probabilities,
                model_version=version or "unknown",
                latency_ms=latency_ms,
                timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    def _load_production_model(self) -> None:
        """Load the production model from MLflow."""
        model_name = self.config.get("mlflow", {}).get("model_name", "ml-model")
        
        try:
            self._load_model(model_name, stage="Production", is_shadow=False)
        except Exception as e:
            logger.warning(
                "Failed to load production model, will serve without model",
                error=str(e),
            )

    def _load_model(
        self,
        model_name: str,
        model_version: str | None = None,
        stage: str | None = None,
        is_shadow: bool = False,
    ) -> None:
        """Load a model from MLflow.
        
        Args:
            model_name: Model name in registry.
            model_version: Specific version to load.
            stage: Model stage (Production, Staging, etc.).
            is_shadow: Whether to load as shadow model.
        """
        tracking_uri = self.config.get("mlflow", {}).get(
            "tracking_uri", "http://mlflow:5000"
        )
        mlflow.set_tracking_uri(tracking_uri)
        
        if model_version:
            model_uri = f"models:/{model_name}/{model_version}"
        elif stage:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            model_uri = f"models:/{model_name}/latest"
        
        logger.info("Loading model", uri=model_uri, is_shadow=is_shadow)
        
        model = mlflow.pytorch.load_model(model_uri)
        model = model.to(self.device)
        
        if is_shadow:
            self.shadow_model = model
            self.shadow_version = model_version or stage or "latest"
        else:
            self.model = model
            self.model_version = model_version or stage or "latest"
        
        logger.info(
            "Model loaded successfully",
            version=model_version or stage,
            is_shadow=is_shadow,
        )

    def run(self) -> None:
        """Run the model server."""
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI is not installed. Install with: pip install fastapi uvicorn")
        
        logger.info("Starting server", host=self.host, port=self.port)
        uvicorn.run(self.app, host=self.host, port=self.port)


def main():
    """Main entry point for the model server."""
    import yaml
    from pathlib import Path
    
    config_path = Path("/app/configs/deployment_config.yaml")
    
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    server = ModelServer(config)
    server.run()


if __name__ == "__main__":
    main()
