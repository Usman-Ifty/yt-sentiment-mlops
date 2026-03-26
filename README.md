# Ifty YouTube Comment Sentiments 🎥🚀

A professional, end-to-end MLOps pipeline for analyzing YouTube comment sentiment in real-time. This system fine-tunes a **DistilBERT** Transformer model, packages it as a highly-optimized **Docker** container, and serves it via a public API on **AWS EC2**. It is seamlessly integrated into a sleek **Chrome Extension** for gamified audience analytics.

## 🌟 Key Features
*   **Vibe Rank (Gamified Analysis):** Assigns a Letter Grade (S, A, B, C, D, or F) to every YouTube video based on overall audience mood.
*   **Top 10 Keyword Cloud:** Automatically extracts and highlights the 10 most trending words from the comment section.
*   **Dynamic UI Highlighting:** Literally paints the YouTube page, adding [POSITIVE], [NEUTRAL], or [NEGATIVE] glow-badges to every comment.
*   **Public AWS API:** 24/7 inference server hosted on an Amazon EC2 t3.micro instance.

---

## 🛠️ The MLOps Pipeline (Technical Journey)

### Phase 1: Data Engineering & DVC
*   **Dataset:** Cleaned and tokenized over 17,000 raw YouTube comments from Kaggle.
*   **DVC (Data Version Control):** Integrated DVC to track large datasets and model weights securely without bloating Git history.

### Phase 2: AI Fine-Tuning with MLflow
*   **Model:** `distilbert-base-uncased` fine-tuned for 3-class sentiment classification (Negative, Neutral, Positive).
*   **Tracking:** Used **MLflow** to log hyperparameters, loss curves, and F1 scores. Only models with >0.85 F1 were promoted for deployment.

### Phase 3: Docker & AWS Deployment
*   **Storage Optimization:** Optimized the Docker image from 4GB to 1GB by forcing a CPU-only PyTorch index, solving AWS EBS disk space limitations.
*   **Cloud Hosting:** Deployed on Ubuntu 24.04 (AWS EC2) with custom Security Group rules for traffic routing and secure RSA key management.

### Phase 4: Full-Stack Chrome Integration
*   **Frontend:** Built a premium Glassmorphism UI using Vanilla JS and CSS3.
*   **Scraper:** Developed a high-speed DOM scanner that captures every comment loaded in the browser.

---

## 🚀 How to Install & Use

### 1. Server Side (AWS EC2)
1.  Ensure Docker is installed on your Linux server.
2.  Clone the repository and place the model weights in `models/bert/best_model/`.
3.  Run: `sudo docker build -t ifty-sentiment-api .`
4.  Run: `sudo docker run -d -p 5000:5000 ifty-sentiment-api`

### 2. Client Side (Chrome Extension)
1.  Open Chrome and navigate to `chrome://extensions/`.
2.  Enable **Developer Mode**.
3.  Click **Load Unpacked** and select the `chrome-extension/` folder.
4.  Open any YouTube video, scroll down to load comments, and hit **RUN SENTIMENT AUDIT**!

---

## 💰 Monetization Potential
*   **Influencer Brand Health Audit:** Charge YouTubers for a "Sentiment Health Score" report to help them avoid controversies.
*   **Pro Version SaaS:** Charge a monthly fee for unlimited comment analysis and PDF report exports.
*   **B2B Market Research:** Sell aggregated sentiment data on product reviews to corporate marketing firms.

---

**Developed by: [Usman Ifty](https://www.linkedin.com/in/usman-awan-a85877359/)**
**GitHub Repository:** [github.com/Usman-Ifty/yt-sentiment-mlops](https://github.com/Usman-Ifty/yt-sentiment-mlops)
