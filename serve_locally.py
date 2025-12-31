"""
Simple standalone model serving script
Run: python serve_locally.py
"""
import os
# Only set defaults if not already set (allows Docker override)
os.environ.setdefault('MLFLOW_TRACKING_URI', 'http://localhost:5000')
os.environ.setdefault('AWS_ACCESS_KEY_ID', 'minioadmin')
os.environ.setdefault('AWS_SECRET_ACCESS_KEY', 'minioadmin')
os.environ.setdefault('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000')

from typing import List
from datetime import datetime
import time

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import mlflow
    import torch
    import uvicorn
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install fastapi uvicorn mlflow torch")
    exit(1)

app = FastAPI(title="Titanic Classifier API")

# Enable CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
model_version = "unknown"

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
        print("Loading model from MLflow...")
        model_uri = "models:/classifier/Production"
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()
        
        # Get version
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("classifier", stages=["Production"])
        if versions:
            model_version = versions[0].version
        
        print(f"✓ Model loaded: v{model_version}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Make sure MLflow is running and model is registered")

@app.get("/")
async def root():
    return {
        "service": "Titanic Classifier",
        "version": model_version,
        "status": "ready" if model else "not loaded"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "model_version": model_version,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
    
    try:
        # Convert to tensor
        features = torch.tensor([request.features], dtype=torch.float32)
        
        # Predict
        with torch.no_grad():
            outputs = model(features)
            probs = torch.softmax(outputs, dim=-1).squeeze()
            
            # Handle both single and multi-class outputs
            if probs.dim() == 0:  # Single value
                probs = [float(probs)]
            else:
                probs = probs.tolist()
                
            pred = torch.argmax(outputs, dim=-1).item()
        
        latency = (time.time() - start) * 1000
        
        return PredictionResponse(
            prediction=int(pred),
            probabilities=probs,
            model_version=model_version,
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("=" * 60)
    print("  Titanic Classifier - Local Serving")
    print("=" * 60)
    print()
    print("API Endpoints:")
    print("  • Health:  http://localhost:8000/health")
    print("  • Predict: http://localhost:8000/predict")
    print("  • Docs:    http://localhost:8000/docs")
    print()
    print("Starting server...")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
