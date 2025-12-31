# Mermaid Architecture Diagrams for MLOps Pipeline Presentation

## 1. System Architecture Overview

```mermaid
graph TB
    subgraph "Client Layer"
        API[REST API Client]
    end
    
    subgraph "Serving Layer"
        FastAPI[FastAPI Server<br/>Port 8000]
    end
    
    subgraph "Orchestration Layer"
        Airflow-Web[Airflow Webserver<br/>Port 8081]
        Airflow-Sched[Airflow Scheduler]
        Airflow-Worker[Airflow Worker]
        Airflow-Trigger[Airflow Triggerer]
    end
    
    subgraph "ML Platform Layer"
        MLflow[MLflow Server<br/>Port 5000]
    end
    
    subgraph "Storage Layer"
        MinIO[MinIO<br/>S3 Storage<br/>Port 9000-9001]
        Postgres-AF[(PostgreSQL<br/>Airflow DB)]
        Postgres-ML[(PostgreSQL<br/>MLflow DB)]
        Redis[(Redis<br/>Task Queue)]
    end
    
    API --> FastAPI
    FastAPI --> MLflow
    FastAPI --> MinIO
    
    Airflow-Web --> Airflow-Sched
    Airflow-Sched --> Airflow-Worker
    Airflow-Sched --> Airflow-Trigger
    Airflow-Worker --> Redis
    Airflow-Worker --> MLflow
    Airflow-Worker --> MinIO
    
    Airflow-Sched --> Postgres-AF
    Airflow-Worker --> Postgres-AF
    
    MLflow --> Postgres-ML
    MLflow --> MinIO
    
    style API fill:#e1f5ff
    style FastAPI fill:#ffeb99
    style MLflow fill:#c3e88d
    style MinIO fill:#ff9999
    style Postgres-AF fill:#d4a5ff
    style Postgres-ML fill:#d4a5ff
    style Redis fill:#ffc299
```

**Usage:** Convert to PNG using [Mermaid Live Editor](https://mermaid.live/) or `mmdc` CLI tool.  
**Filename:** `architecture_overview.png`

---

## 2. Data Flow Pipeline

```mermaid
graph LR
    A[Raw Data<br/>CSV] --> B[1. Data Validation<br/>Schema Check<br/>Quality Assurance]
    B --> C[2. Preprocessing<br/>Pandas Pipeline<br/>Feature Engineering]
    C --> D[3. Training<br/>PyTorch + Accelerate<br/>Distributed GPU]
    D --> E[4. Evaluation<br/>Metrics Computation<br/>Performance Check]
    E --> F[5. Registration<br/>MLflow Registry<br/>Model Versioning]
    F --> G{Promotion<br/>Criteria<br/>Met?}
    G -->|Yes| H[6. Deployment<br/>FastAPI Serving<br/>Production]
    G -->|No| I[Archive<br/>Model]
    
    style A fill:#e1f5ff
    style B fill:#c3e88d
    style C fill:#ffeb99
    style D fill:#ff9999
    style E fill:#d4a5ff
    style F fill:#ffc299
    style G fill:#ffcccc
    style H fill:#90ee90
    style I fill:#d3d3d3
```

**Usage:** Convert to PNG for pipeline flow visualization.  
**Filename:** `pipeline_flow.png`

---

## 3. Docker Services Network

```mermaid
graph TB
    subgraph "Docker Network: mlops-network"
        AW[airflow-webserver:8081]
        AS[airflow-scheduler]
        AWK[airflow-worker]
        AT[airflow-triggerer]
        ML[mlflow:5000]
        MN[minio:9000-9001]
        MS[model-serving:8000]
        PA[(postgres-airflow)]
        PM[(postgres-mlflow)]
        RD[(redis:6379)]
    end
    
    AW -.-> AS
    AS --> AWK
    AS --> AT
    AWK --> RD
    AWK --> PA
    AWK --> ML
    
    ML --> PM
    ML --> MN
    
    MS --> ML
    MS --> MN
    
    AW --> PA
    
    style AW fill:#4a90e2
    style AS fill:#4a90e2
    style AWK fill:#4a90e2
    style AT fill:#4a90e2
    style ML fill:#50c878
    style MN fill:#ff6b6b
    style MS fill:#feca57
    style PA fill:#9b59b6
    style PM fill:#9b59b6
    style RD fill:#e74c3c
```

**Usage:** Convert to PNG for Docker services visualization.  
**Filename:** `docker_services.png`

---

## 4. Training Workflow (Detailed)

```mermaid
sequenceDiagram
    participant DAG as Airflow DAG
    participant DV as Data Validator
    participant DP as Data Preprocessor
    participant TR as Trainer (GPU)
    participant EV as Evaluator
    participant ML as MLflow Server
    participant PR as Promoter
    participant API as FastAPI Server
    
    DAG->>DV: Start Validation
    DV->>DV: Check Schema<br/>Validate Quality
    DV-->>DAG: ✓ Valid
    
    DAG->>DP: Start Preprocessing
    DP->>DP: Impute Missing<br/>Scale Features<br/>Train/Val/Test Split
    DP-->>DAG: ✓ Data Ready
    
    DAG->>TR: Start Training
    TR->>TR: Load Data<br/>Initialize Model<br/>Train (10 epochs)
    TR->>ML: Log Metrics<br/>Parameters
    TR-->>DAG: ✓ Training Complete
    
    DAG->>EV: Evaluate Model
    EV->>EV: Compute Metrics<br/>Accuracy, F1, ROC-AUC
    EV->>ML: Log Evaluation Results
    EV-->>DAG: ✓ Evaluation Complete
    
    DAG->>ML: Register Model
    ML-->>DAG: Model v3 Registered
    
    DAG->>PR: Check Promotion
    PR->>ML: Get Current Production
    PR->>PR: Compare Metrics<br/>Check Thresholds
    PR->>ML: Promote to Production
    
    ML->>API: Model Updated
    API-->>ML: Ready to Serve
```

**Usage:** Convert to PNG for training workflow sequence.  
**Filename:** `training_workflow.png`

---

## 5. Model Deployment Strategies

```mermaid
graph TB
    subgraph "Canary Deployment"
        NEW1[New Model v3]
        PROD1[Production Model v2]
        ROUTE1{Traffic Router}
        
        ROUTE1 -->|1%| NEW1
        ROUTE1 -->|99%| PROD1
        NEW1 --> MON1[Monitor Metrics]
        MON1 --> DEC1{Success?}
        DEC1 -->|Yes| INCR[Increase to 5%]
        DEC1 -->|No| ROLL1[Rollback]
        INCR --> ROUTE2{Traffic Router}
        ROUTE2 -->|5%| NEW1
        ROUTE2 -->|95%| PROD1
    end
    
    subgraph "Shadow Deployment"
        REQ[User Request]
        PROD2[Production Model]
        SHADOW[Shadow Model]
        
        REQ --> PROD2
        REQ -.->|Copy| SHADOW
        PROD2 --> RESP[Response to User]
        SHADOW --> LOG[Log & Compare]
        LOG -.-> METRICS[Metrics Analysis]
    end
    
    style NEW1 fill:#90ee90
    style PROD1 fill:#4a90e2
    style PROD2 fill:#4a90e2
    style SHADOW fill:#ffa500
    style MON1 fill:#ffeb99
    style LOG fill:#ffeb99
```

**Usage:** Convert to PNG for deployment strategies visualization.  
**Filename:** `deployment_strategies.png`

---

## How to Generate PNG Images

### Option 1: Mermaid Live Editor (Easiest)
1. Visit [https://mermaid.live/](https://mermaid.live/)
2. Paste each Mermaid code block
3. Click "Actions" → "PNG" to download

### Option 2: Mermaid CLI (Command Line)
```bash
npm install -g @mermaid-js/mermaid-cli

mmdc -i mermaid_charts.md -o architecture_overview.png -s 1
mmdc -i mermaid_charts.md -o pipeline_flow.png -s 2
mmdc -i mermaid_charts.md -o docker_services.png -s 3
mmdc -i mermaid_charts.md -o training_workflow.png -s 4
mmdc -i mermaid_charts.md -o deployment_strategies.png -s 5
```

### Option 3: VS Code Extension
1. Install "Markdown Preview Mermaid Support" extension
2. Open this file in VS Code
3. Right-click on diagram → "Export as PNG"

---

## Inserting into LaTeX Presentation

Once you have the PNG files, update the LaTeX placeholders:

```latex
\begin{center}
\includegraphics[width=0.8\textwidth]{architecture_overview.png}
\end{center}
```

Replace `\textit{[Architecture Diagram Placeholder]}` with the `\includegraphics` command.
