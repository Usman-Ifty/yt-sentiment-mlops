# Ifty YouTube Comment Sentiments - Industrial-Grade MLOps & AI Platform 🚀🧠⚡

An end-to-end, high-performance Machine Learning Operations (MLOps) system architected by **Usman Ifty & Mashood** to monitor, analyze, and gamify YouTube audience sentiment at scale. This project integrates cutting-edge Transformer models (**DistilBERT**), containerized deployment (**Docker**), and cloud-native infrastructure (**AWS EC2**) into a premium, real-time **Chrome Extension**.

---

## 🏗️ Deep-Dive Technical Journey (Full Execution Details)

### Phase 1: Foundational MLOps Architecture & Infrastructure
The project follows a modular, production-ready directory structure designed for scalability and collaborative maintenance.
*   **Directory Management:** Built a standardized layout including `src/` (core logic), `data/` (versioned storage), `mlruns/` (experiment data), and `chrome-extension/` (client-side).
*   **DVC (Data Version Control):** Implemented `dvc` to decouple high-volume model artifacts (255MB+ `.safetensors`) from the Git source tree. This prevents repository bloat while maintaining a rigorous audit trail of model iterations.
*   **Dependency Isolation:** Engineered isolated Python 3.10 environments (`venv`) to prevent version conflicts.

### Phase 2: The Evolutionary Journey (Model v1 vs. Model v2)
This project was not a "one-and-done" build. It involved a rigorous iterative process to solve real-world "Domain Drift."

#### 📉 Model v1: The Foundation
*   **Dataset:** 18,000+ Kaggle-labeled YouTube comments (mostly formal English).
*   **The Issue:** When tested on modern "Gen Z" creators like **IShowSpeed**, the model struggled. To the AI, "W Video" or "He's Cooked 💀" looked like noise or neutral text. It couldn't grasp the "Vibe."

#### 🚀 Model v2: The "Big Brain" Gen Z Upgrade
To fix this, we engineered a massively larger and more diverse intelligence pipeline:
*   **Hybrid Slang Dataset:** Merged 4 major sources (**Cardiff NLP, MTEB, Sentiment140, and local YouTube data**) to create a **80,000+ record Master Dataset**.
*   **Slang Normalization Layer:** Built a custom dictionary containing **60+ modern terms and emojis** (W, L, Goat, Mid, Fr, Rizz, etc.) that translates Gen Z slang into BERT-compatible English. 
*   **GPU Acceleration:** Shifted training to **Google Colab (T4 GPU)**, cutting training time from 90 minutes (local CPU) down to just **12 minutes** for 5 full epochs.

### Phase 3: Transformer Model Fine-Tuning (DistilBERT)
*   **Architecture Choice:** Selected `distilbert-base-uncased`—a distilled, 40% smaller version of BERT that retains 97% of its performance.
*   **Imbalance Correction:** Handled "Sentiment Skew" by calculating class weights for the `CrossEntropyLoss` function, ensuring the model treats Negative feedback with the same mathematical weight as Positive comments.

### Phase 4: Experiment Tracking & Model Registry (MLflow)
*   **Dynamic Logging:** Integrated **MLflow** to record every mathematical signal during training, including global step loss, validation F1-scores, and per-epoch accuracy.
*   **Run IDs:** Every training session (including the Gen Z Colab run) is given a unique 32-character ID (e.g., `1aa84b27b...`). This allows us to compare "v1" vs "v2" performance side-by-side.

### Phase 5: Production Deployment & Docker Optimization
*   **Docker Containerization:** Defined a multi-stage build in a `Dockerfile` based on `python:3.10-slim`.
*   **AWS EC2 Infrastructure:** Provisioned an **Ubuntu 24.04 (t3.micro)** instance.
*   **The "3GB Storage Hack":** Solved the common AWS `No space left on device` error by forcing `pip` to ignore massive CUDA binaries, using CPU-only PyTorch instead.

### Phase 6: "Ifty Sentiments" Chrome Extension v4.0
*   **Frontend Design:** Built an elite **Glassmorphism UI** using Backdrop-Blur (CSS3) for a dark-mode, premium feel.
*   **The Vibe-Rank Engine:** Developed a gamification algorithm that assigns a **Letter Grade (S, A, B, C, D, or F)** based on the ratio of Positive to Negative sentiment.
*   **Real-time DOM Painting:** Injects styling logic that adds [POSITIVE], [NEUTRAL], or [NEGATIVE] glow-badges directly into the YouTube comment section.

---

## 🔮 Future Roadmap (v5.0 Ideas)
*   **AI Reply Suggestions:** Use the model results to suggest a "Witty" or "Supportive" reply to a comment.
*   **Emoji Breakdown:** A visual cloud showing the most dominant emojis in the comment section (💀, 🔥, 👑).
*   **Real-time Vibe Ticker:** A live banner that updates the "Channel Health" as the user scrolls further down.

---

## 🛠️ Step-by-Step Installation Guide (Easy Baby Steps)

### Step 1: Downloading the Source
1.  Visit the [Official GitHub Repo](https://github.com/Usman-Ifty/yt-sentiment-mlops).
2.  Click the green **"Code"** button and choose **"Download ZIP"**.
3.  Extract the folder and move it to your Desktop.

### Step 2: Loading the Tool into Google Chrome
1.  Open Chrome and go to `chrome://extensions/`.
2.  Flip the switch for **"Developer mode"** to ON.
3.  Click **"Load unpacked"**.
4.  Select the subfolder named **`chrome-extension`**.

### Step 3: Running Your First Sentiment Scan
1.  Go to YouTube and open any video.
2.  **Scroll down** to load the comments.
3.  Click the **Ifty Sentiments** icon from your browser toolbar.
4.  Hit the big button: **"RUN SENTIMENT SCAN"**.
5.  Watch the ranking (S, A, B, C, F) pop up and your YouTube screen turn green/red!

---

**Developed by: [Usman Ifty](https://www.linkedin.com/in/usman-awan-a85877359/) & Mashood**
**Official Repository:** [github.com/Usman-Ifty/yt-sentiment-mlops](https://github.com/Usman-Ifty/yt-sentiment-mlops)

