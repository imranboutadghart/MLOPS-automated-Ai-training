#!/bin/bash
# Start Distributed Training Script
# Launches distributed training with HuggingFace Accelerate

set -e

# Configuration
CONFIG_FILE="${1:-configs/training_config.yaml}"
ACCELERATE_CONFIG="${2:-configs/accelerate_config.yaml}"
DATA_PATH="${3:-data/processed}"
OUTPUT_DIR="${4:-checkpoints}"
NUM_GPUS="${5:-auto}"

echo "=========================================="
echo "Starting Distributed Training"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Accelerate Config: $ACCELERATE_CONFIG"
echo "Data Path: $DATA_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo ""

# Check if CUDA is available
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null || {
    echo "WARNING: CUDA not available, training on CPU"
}

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export TOKENIZERS_PARALLELISM=false

# Determine number of processes
if [ "$NUM_GPUS" == "auto" ]; then
    NUM_GPUS=$(python3 -c "import torch; print(max(1, torch.cuda.device_count()))" 2>/dev/null || echo "1")
fi
echo "Using $NUM_GPUS GPU(s)"
echo ""

# Check MLflow
if [ -n "$MLFLOW_TRACKING_URI" ]; then
    echo "MLflow Tracking URI: $MLFLOW_TRACKING_URI"
else
    export MLFLOW_TRACKING_URI="http://localhost:5000"
    echo "MLflow Tracking URI (default): $MLFLOW_TRACKING_URI"
fi

echo ""
echo "Starting training..."
echo "=========================================="

# Launch distributed training
accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes "$NUM_GPUS" \
    src/training/distributed_train.py \
    --config "$CONFIG_FILE" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "View results in MLflow: $MLFLOW_TRACKING_URI"
