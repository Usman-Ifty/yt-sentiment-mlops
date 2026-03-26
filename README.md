# Ifty YouTube Comment Sentiments - Full-Stack MLOps Pipeline 🚀🧠

A state-of-the-art Machine Learning Operations (MLOps) system designed to monitor, analyze, and gamify YouTube audience sentiment. This platform fine-tunes a **DistilBERT Transformers** model, scales it via **Docker**, hosts it on **AWS EC2**, and delivers real-time analytics through a premium **Chrome Extension**.

---

## 🏗️ The Full Technical Journey (Phase-by-Phase)

### Phase 1: Foundational Infrastructure
*   **Architectural Design:** Established a standardized MLOps directory structure to separate data engineering (`src/data`), model development (`src/model`), and API serving (`src/api`).
*   **Version Control:** Initialized **Git** for source tracking and **DVC (Data Version Control)** to manage multi-gigabyte model weights and datasets without overwhelming the repository.
*   **Dependency Strategy:** Configured isolated Python environments (`venv`) and managed package versions via `requirements.txt` specifically for cloud and local parity.

### Phase 2: Data Engineering & Text Cleaning
*   **Preprocessing:** Built a custom cleaning engine using Python's `regex` and `cleantext` libraries to strip emojis, URLs, and @mentions from 17,000+ raw records.
*   **Tokenization:** Leveraged the HuggingFace `DistilBertTokenizer` to convert raw text into highly optimized attention masks and input IDs for the GPU/CPU.
*   **Class Imbalance:** Handled skewed sentiment data (mostly positive) using custom weighting in the loss function to ensure the AI detects negative feedback accurately.

### Phase 3: Model Training & Fine-Tuning
*   **The Engine:** Fine-tuned `distilbert-base-uncased` for three labels (Positive, Neutral, Negative).
*   **GPU Integration:** Configured PyTorch logic to automatically detect CUDA for training but fallback to optimized CPU instructions for cost-effective cloud deployment.

### Phase 4: MLOps Experiment Tracking (MLflow)
*   **Logging:** Every training run was recorded via **MLflow**, tracking F1-Scores, Accuracy, and Learning Rate decay in real-time.
*   **Model Registry:** Built a promotion script (`promote.py`) that strictly evaluates model versions based on testing data performance before "promoting" them to the production registry.

### Phase 5: Cloud Deployment & Dockerization
*   **Docker Containerization:** Engineered a multi-stage Docker build. **CRITICAL FIX:** Solved AWS disk space errors by forcing the `pip` index to use a **CPU-only version of PyTorch**, reducing the image size by 3GB.
*   **AWS AWS Hosting:** Scaled the API to an **Amazon Web Services (AWS) EC2 t3.micro** instance. Managed security via RSA `.pem` keys and configured Security Group firewalls for Port 5000.
*   **Detached Serving:** Used `docker run -d` to ensure the API stays alive even after the deployment terminal is closed.

### Phase 6: The "Ifty Sentiments" Chrome Extension
*   **The UI:** Created a futuristic **Glassmorphism** interface using Vanilla JavaScript and CSS3.
*   **The Gamification (Vibe-Rank):** Developed an algorithm that ranks videos (S, A, B, C, D, or F) based on audience health.
*   **Dynamic Painting:** The extension injects logic directly into YouTube.com to paint every comment with a glowing sentiment border.

---

## 🛠️ Step-by-Step Installation (Baby Steps)

### Step 1: Downloading the Project
1.  Go to the [GitHub Repository](https://github.com/Usman-Ifty/yt-sentiment-mlops).
2.  Click the green **"Code"** button and select **"Download ZIP"**.
3.  Extract the folder onto your Desktop.

### Step 2: Activating the Extension in Chrome
1.  Open **Google Chrome** on your laptop.
2.  In the address bar, type `chrome://extensions/` and hit Enter.
3.  In the top-right corner, turn ON **"Developer mode"**.
4.  In the top-left corner, click the **"Load unpacked"** button.
5.  Select the **`chrome-extension`** folder from the project you extracted.
6.  *Success:* You will now see the colorful "Ifty" logo in your toolbar!

### Step 3: Using the AI Analytics
1.  Navigate to ANY YouTube video (e.g., a movie trailer or a news clip).
2.  **Scroll down** and wait for the comments to appear on the screen.
3.  Click the tiny puzzle-piece icon at the top of Chrome and pin **"Ifty Youtube Sentiments"**.
4.  Click the icon and hit the big button: **"RUN SENTIMENT AUDIT"**.
5.  Watch the AI rank the video (S, A, B, C, F) and paint the comments green/yellow/red!
---

**Developed by: [Usman Ifty](https://www.linkedin.com/in/usman-awan-a85877359/)**
**GitHub Repository:** [github.com/Usman-Ifty/yt-sentiment-mlops](https://github.com/Usman-Ifty/yt-sentiment-mlops)
