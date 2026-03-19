# YouTube Sentiment Analysis MLOps 🎥🧠

An end-to-end Machine Learning Operations (MLOps) pipeline for analyzing YouTube comments sentiment. This project fine-tunes a BERT model using a custom dataset and serves it via a Flask API deployed on AWS EC2. It features a custom Chrome Extension that dynamically displays sentiment badges on YouTube directly in your browser.

## 🚀 Phase 1: Project Initialization

This commit establishes the foundational skeleton for the MLOps pipeline, ensuring best practices for tracking and scale.

**Key Additions:**
*   **Standardized Directory Structure:** Created segmented directories for `data` (raw & processed), `src` (api, data, model), `notebooks`, `tests`, and GitHub Actions.
*   **Version Control:** Initialized Git for source code tracking.
*   **Data Version Control (DVC):** Initialized `dvc` to manage large datasets and models outside of Git (via `.dvcignore` and `.gitignore`).
*   **Dependency Management:** Added `requirements.txt` with essential libraries (`transformers`, `mlflow`, `dvc`, `flask`, `torch`).
*   **Environment Configuration:** Added a `.env` file (ignored by Git) to securely manage MLflow tracking URIs and model parameters.

### Next Steps 
- **Phase 2:** Kaggle Data Acquisition & Exploratory Data Analysis (EDA)
- **Phase 3:** Fine-Tuning DistilBERT & MLflow Experiment Tracking
- **Phase 4:** Flask API Development & Dockerization
- **Phase 5:** AWS EC2 Deployment
- **Phase 6:** Chrome Extension Integration
- **Phase 7:** GitHub Actions CI/CD Pipeline
