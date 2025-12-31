"""
Simple FastAPI Model Serving Application
"""
import os
from typing import List
from datetime import datetime
import time

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import mlflow
    import torch
    import numpy as np
    AVAILABLE = True
except ImportError:
    AVAILABLE = False
    print("Required packages not installed. Run: pip install fastapi uvicorn mlflow torch")

if AVAILABLE:
    app = FastAPI(title="ML Model Server", version="1.0.0")
    
    # Enable CORS for web interface
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Global model variable
    model = None
    model_version = None
    model_name = os.getenv("MODEL_NAME", "titanic_classifier")
    model_stage = os.getenv("MODEL_STAGE", "Production")
    
    
    class PredictionRequest(BaseModel):
        features: List[float]
    
    
    class PredictionResponse(BaseModel):
        prediction: int
        probabilities: List[float]
        model_version: str
        latency_ms: float
    
    
    @app.on_event("startup")
    async def load_model():
        """Load model on startup"""
        global model, model_version
        
        try:
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(mlflow_uri)
            
            # Load model from MLflow
            model_uri = f"models:/{model_name}/{model_stage}"
            print(f"Loading model from: {model_uri}")
            model = mlflow.pytorch.load_model(model_uri)
            model.eval()
            
            # Get version info
            client = mlflow.tracking.MlflowClient()
            try:
                versions = client.get_latest_versions(model_name, stages=[model_stage])
                if versions:
                    model_version = versions[0].version
                else:
                    model_version = "unknown"
            except:
                model_version = "unknown"
            
            print(f"Model loaded successfully: {model_name} v{model_version}")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    
    
    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {
            "status": "healthy" if model is not None else "unhealthy",
            "model_loaded": model is not None,
            "model_name": model_name,
            "model_version": model_version,
            "timestamp": datetime.now().isoformat()
        }
    
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """Make prediction"""
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start_time = time.time()
        
        try:
            # Convert input to tensor
            features = torch.tensor([request.features], dtype=torch.float32)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(features)
                probabilities = torch.softmax(outputs, dim=-1).squeeze().tolist()
                prediction = torch.argmax(outputs, dim=-1).item()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return PredictionResponse(
                prediction=int(prediction),
                probabilities=probabilities,
                model_version=str(model_version),
                latency_ms=round(latency_ms, 2)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "ML Model Server",
            "model": model_name,
            "version": model_version,
            "endpoints": {
                "health": "/health",
                "predict": "/predict (POST)",
                "docs": "/docs"
            }
        }
