#!/bin/bash
# Environment Setup Script
# Sets up the distributed training environment

set -e

echo "=========================================="
echo "Distributed Training Pipeline Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if at least Python 3.10
if [[ "$(printf '%s\n' "3.10" "$python_version" | sort -V | head -n1)" != "3.10" ]]; then
    echo "ERROR: Python 3.10+ is required"
    exit 1
fi

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install project in editable mode
echo "Installing project..."
pip install -e .

# Setup Accelerate
echo "Configuring Accelerate..."
if [ ! -f "$HOME/.cache/huggingface/accelerate/default_config.yaml" ]; then
    cp configs/accelerate_config.yaml $HOME/.cache/huggingface/accelerate/default_config.yaml
fi

# Create data directories
echo "Creating data directories..."
mkdir -p data/training data/validation data/test
mkdir -p checkpoints logs

# Setup environment variables
echo "Setting up environment..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null || echo "PyTorch not installed or CUDA not available"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Start the stack: cd docker && docker-compose up -d"
echo "3. Access Airflow: http://localhost:8080"
echo "4. Access MLflow: http://localhost:5000"
echo ""
