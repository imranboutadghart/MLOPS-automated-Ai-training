# Distributed Continuous Training Pipeline

A production-style MLOps pipeline that automates model training, evaluation, and deployment using modern distributed computing and orchestration technologies.

## 🚀 Features

- **Automated Scheduling**: Configurable training schedules (hourly, daily, weekly)
- **Distributed Training**: Multi-GPU training with HuggingFace Accelerate
- **Model Registry**: MLflow integration for versioning and artifact management
- **Canary Deployments**: Gradual traffic shifting with automatic rollback
- **Shadow Deployments**: A/B testing without production impact
- **Full Observability**: Metrics, logging, and health monitoring

## 📋 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Orchestration Layer                          │
│  ┌─────────────┐   ┌───────────────┐   ┌──────────────────────┐ │
│  │   Airflow   │──▶│   Scheduler   │──▶│ Daily/Weekly/Hourly  │ │
│  └─────────────┘   └───────────────┘   └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Data Pipeline                              │
│  ┌────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │ Ingestion  │──▶│ Preprocessing │──▶│ Feature Store        │  │
│  └────────────┘   └──────────────┘   └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Training Layer                              │
│  ┌─────────────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │ HF Accelerate   │──▶│ PyTorch DDP  │──▶│ Multi-GPU       │  │
│  └─────────────────┘   └──────────────┘   └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Registry & Deployment                          │
│  ┌────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │  MLflow    │──▶│  Promotion   │──▶│ Canary/Shadow Deploy │  │
│  └────────────┘   └──────────────┘   └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | Apache Airflow 2.8+ |
| Deep Learning | PyTorch 2.1+ |
| Distributed Training | HuggingFace Accelerate |
| Data Processing | Pandas, NumPy |
| Model Registry | MLflow 2.10+ |
| Object Storage | MinIO |
| Containerization | Docker, Docker Compose |

## 📁 Project Structure

```
distributed-training-pipeline/
├── airflow/
│   ├── dags/                    # Airflow DAGs
│   │   ├── continuous_training_dag.py
│   │   ├── data_pipeline_dag.py
│   │   └── deployment_dag.py
│   ├── plugins/                 # Custom operators
│   └── config/                  # Airflow config
├── src/
│   ├── data/                    # Data ingestion & preprocessing
│   ├── training/                # Models, trainer, distributed training
│   ├── registry/                # MLflow client & model promotion
│   ├── deployment/              # Canary & shadow deployment
│   └── utils/                   # Config, logging, monitoring
├── configs/
│   ├── training_config.yaml     # Training hyperparameters
│   ├── accelerate_config.yaml   # DDP configuration
│   └── deployment_config.yaml   # Deployment settings
├── docker/
│   ├── Dockerfile.airflow       # Airflow image with ML deps
│   ├── Dockerfile.training      # GPU training image
│   └── docker-compose.yml       # Full stack
├── scripts/
│   ├── setup_environment.sh     # Environment setup
│   └── start_training.sh        # Launch training
├── tests/
│   └── test_pipeline.py         # Integration tests
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (optional, for GPU training)
- Python 3.10+

### 1. Clone and Setup

```bash
# Clone repository
git clone <repository-url>
cd distributed-training-pipeline

# Setup environment
chmod +x scripts/*.sh
./scripts/setup_environment.sh
```

### 2. Start the Stack

```bash
# Start all services
cd docker
docker-compose up -d

# Check status
docker-compose ps
```

### 3. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow UI | http://localhost:8080 | admin / admin |
| MLflow UI | http://localhost:5000 | - |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |

### 4. Trigger Training

```bash
# Via Airflow UI
# Go to http://localhost:8080 and trigger 'continuous_training_dag'
 
# Or via CLI
docker-compose exec airflow-scheduler airflow dags trigger continuous_training_dag
```

## 📊 Training Configuration

Edit `configs/training_config.yaml`:

```yaml
model:
  name: "classifier"
  hidden_sizes: [512, 256, 128]
  dropout: 0.3

training:
  epochs: 10
  batch_size: 64
  learning_rate: 0.001
  mixed_precision: "fp16"

scheduler:
  type: "cosine"
  warmup_steps: 100
```

## 🔄 Deployment Strategies

### Canary Deployment

Gradual traffic shifting: 1% → 5% → 25% → 50% → 100%

```yaml
canary:
  initial_weight: 0.01
  weight_steps: [0.05, 0.25, 0.50, 1.0]
  step_duration_seconds: 300
  rollback_on_failure: true
```

### Shadow Deployment

Parallel inference for A/B testing without production impact.

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

## 📈 Model Promotion Flow

1. **Training** → Model trained and evaluated
2. **Staging** → Model passes threshold checks (accuracy ≥ 0.85, F1 ≥ 0.80)
3. **Production** → Model outperforms current champion by ≥ 1%

## 🔧 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow server URL | http://localhost:5000 |
| `AWS_ACCESS_KEY_ID` | MinIO access key | minioadmin |
| `AWS_SECRET_ACCESS_KEY` | MinIO secret key | minioadmin |

## 📚 Documentation

- [Training Guide](docs/training.md)
- [Deployment Guide](docs/deployment.md)
- [API Reference](docs/api.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

