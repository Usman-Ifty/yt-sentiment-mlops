# YouTube Sentiment Analysis MLOps 🎥🧠

An end-to-end Machine Learning Operations (MLOps) pipeline for analyzing YouTube comments sentiment. This project fine-tunes a BERT model using a custom dataset and serves it via a Flask API deployed on AWS EC2. It features a custom Chrome Extension that dynamically displays sentiment badges on YouTube directly in your browser.

## 🚀 Phase 1: Project Initialization

This commit establishes the foundational skeleton for the MLOps pipeline, ensuring best practices for tracking and scale.

**Key Additions:**
*   **Standardized Directory Structure:** Created segmented directories for `data` (raw & processed), `src` (api, data, model), `notebooks`, `tests`, and GitHub Actions.
*   **Version Control:** Initialized Git for source code tracking.
*   **Data Version Control (DVC):** Initialized `dvc` to manage large datasets and models outside of Git.
*   **Dependency Management:** Added `requirements.txt` with essential libraries (`transformers`, `mlflow`, `dvc`, `flask`, `torch`).
*   **Environment Configuration:** Added a `.env` file securely.

## 🧹 Phase 2: Data Preprocessing & EDA

Integrated the Kaggle raw dataset and built out the NLP cleaning pipeline. 

**Key Additions:**
*   **Exploratory Data Analysis:** Built an interactive EDA script to explore string lengths, verify class imbalances, and map sentiment.
*   **HuggingFace Tokenization:** Leveraged `DistilBertTokenizer` to tokenize, pad, and truncate 17,000+ texts directly into PyTorch format `.pt`.
*   **DVC Pipelines (`dvc.yaml`):** Implemented an automated tracking and execution graph using `dvc repro`.
*   **Unit Testing:** Pytest suite for the data pipeline.

## 🤖 Phase 3: Fine-Tuning DistilBERT & MLflow Experiment Tracking

Finetuned a pre-trained `distilbert-base-uncased` model to classify YouTube comments into Negative, Neutral, and Positive.

**Key Additions:**
*   **Training Script:** Added `train.py` configured with PyTorch DataLoaders, weight decay, linear scheduling, and a customized CrossEntropy loss to handle class imbalances.
*   **Experiment Registry:** Used `MLflow` to dynamically log hyperparameters, step-loss, Validation Accuracy/F1 scores, and formally register the `best_model`.
*   **Production Promotion:** Added `promote.py` logic to evaluate newly trained models against current production thresholds (>0.80 F1) before updating `production_version.json`.
*   **Model Coverage:** Added robust verification tests (`test_model.py`) to assert successful loading and softmax probability structures.

## 🌐 Phase 4: Flask API Development & Dockerization

Built a production-ready REST API to serve the fine-tuned DistilBERT model and packaged it as a Docker container.

**Key Additions:**
*   **Flask REST API (`src/api/app.py`):** Exposes `GET /` (health check) and `POST /predict` endpoint accepting single `text` or batch `texts` returning sentiment label, confidence, and per-class scores.
*   **CORS Support:** Enabled Cross-Origin Resource Sharing (CORS) to allow the Chrome Extension to interact with the API.
*   **Dockerfile:** Multi-stage slim Python 3.10 image copying only the `best_model` weights and `label_map.json` — keeps the image lean (~1.5GB).
*   **`.dockerignore`:** Excludes `mlruns/`, raw data, venvs, and notebooks from the build context.

**How to Run locally:**
```bash
# Activation
.\venv_new\Scripts\Activate.ps1

# Run API
python src/api/app.py

# build and run with Docker
docker build -t yt-sentiment-api .
docker run -p 5000:5000 yt-sentiment-api

# Example Test (PowerShell)
Invoke-RestMethod -Method Post -Uri "http://localhost:5000/predict" `
    -ContentType "application/json" `
    -Body '{"text": "This video is absolutely amazing!"}'
```

### Next Steps
- **Phase 5:** AWS EC2 Deployment
- **Phase 6:** Chrome Extension Integration
- **Phase 7:** GitHub Actions CI/CD Pipeline

## Phase 5: AWS EC2 Deployment

Hosted the Flask Docker Container on an Amazon Web Services (AWS) EC2 `t3.micro` instance running Ubuntu 24.04. This provides a continuously available, public-facing endpoint required for the Chrome Extension to function in real-world scenarios.

**Deployment Architecture & Execution Details:**

*   **Instance Provisioning & Security Integration:** provisioned an AWS EC2 instance and securely managed access via an RSA `.pem` key pair. Strict file permission scoping (using `icacls`) was explicitly enforced on the Windows host to ensure SSH key confidentiality. AWS Security Groups were modified to expose inbound TCP traffic on Port 5000, establishing an open communication line to the Flask deployment.
*   **Decoupled Model Pipeline Transfer:** Due to GitHub's stringent file payload limits, large model artifacts like `model.safetensors` inherently bypass typical version control. Secure Copy Protocol (`scp -r`) was strictly utilized to recursively copy the 255MB model artifacts and configuration files from the local Windows environment directly to the target routing path (`~/yt-sentiment-mlops/models/bert/best_model`) on the AWS server in preparation for Docker encapsulation.
*   **Docker Containerization & Storage Optimization:** Source logic was obtained natively via `git clone`. The default 8GB Elastic Block Store (EBS) volume presented immediate storage limit errors ("No space left on device") during the Docker build stage due to the massive CUDA toolkit footprint associated with default PyTorch installations. This was critically resolved by patching the `Dockerfile` requirements sequentially, forcing `pip` to pull from the CPU-only PyTorch index (`--extra-index-url https://download.pytorch.org/whl/cpu`), drastically reducing image weight. Local dependencies unneeded by the app, such as `label_map.json`, were stripped from the Docker image layers using `sed`. 
*   **Live Production Output:** Following the lightweight Docker image generation, the container was run in detached mode (`docker run -d -p 5000:5000`), persistently mapping the host port to the internal Flask configuration. The `/predict` inference API correctly registers POST requests directly addressed to the assigned Public Elastic IPv4 Address, ensuring immediate scalability.
