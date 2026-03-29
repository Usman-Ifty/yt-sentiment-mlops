"""
Flask API for YouTube Sentiment Analysis
Phase 4 – Serving the fine-tuned DistilBERT model
"""

import os
import json
import logging
from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR   = os.path.join(BASE_DIR, "models", "bert", "best_model")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "data", "processed", "label_map.json")
MAX_LEN     = 128
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
logger.info(f"Loading model from {MODEL_DIR} on {DEVICE} ...")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# Load label map {0: "negative", 1: "neutral", 2: "positive"}
if os.path.exists(LABEL_MAP_PATH):
    with open(LABEL_MAP_PATH) as f:
        raw = json.load(f)
    
    # Check if format has id2label nested (HuggingFace style)
    if "id2label" in raw:
        ID2LABEL = {str(k): v for k, v in raw["id2label"].items()}
    else:
        # Check if format is {"negative": 0} or {"0": "negative"}
        test_key = list(raw.keys())[0]
        if test_key.isdigit():
            ID2LABEL = {str(k): v for k, v in raw.items()}
        else:
            ID2LABEL = {str(v): k for k, v in raw.items()}
else:
    ID2LABEL = {"0": "negative", "1": "neutral", "2": "positive"}

logger.info(f"Model ready. Label map: {ID2LABEL}")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app) # Enable CORS for Chrome Extension support later


# ---------------------------------------------------------------------------
# Gen-Z Slang Normalization Logic (Ifty-Mashood Special)
# ---------------------------------------------------------------------------
SLANG_MAP = {
    "w": "great winning excellent",
    "l": "bad losing failure",
    "dead": "so funny laughing",
    "funny": "laughing great",
    "cooked": "bad lost over",
    "mid": "average neutral boring",
    "rizz": "charismatic excellent",
    "goat": "legendary greatest",
    "fire": "amazing awesome",
    "fr": "for real truly",
    "ngl": "not gonna lie truly",
    "no cap": "truthfully truly",
}

EMOJI_SENTIMENT = {
    "💀": " funny laughing ",
    "🔥🔥": " amazing amazing ",
    "🔥": " amazing fire ",
    "🐐": " legendary goat ",
    "👑": " king great ",
    "🙌": " amazing respect ",
    "😭": " laughing crying funny ",
    "💯": " perfect excellent ",
    "🤡": " bad clown ",
    "👎": " bad dislike ",
    "🤮": " gross bad ",
}

def normalize_slang(text: str) -> str:
    """Pre-process Gen-Z slang and emojis for the AI model."""
    text = text.lower()
    for emoji, replacement in EMOJI_SENTIMENT.items():
        text = text.replace(emoji, replacement)
    
    words = text.split()
    normalized = []
    for w in words:
        clean_w = "".join(char for char in w if char.isalnum())
        if clean_w in SLANG_MAP:
            normalized.append(SLANG_MAP[clean_w])
        else:
            normalized.append(w)
    return " ".join(normalized)


def predict_text(text: str) -> dict:
    """Run inference on a single text string."""
    # Step 1: Normalize the vibe
    vibe_text = normalize_slang(text)
    
    encoding = tokenizer(
        vibe_text,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    probs      = torch.softmax(logits, dim=1).squeeze().tolist()
    pred_id    = int(torch.argmax(logits, dim=1).item())
    pred_label = ID2LABEL.get(str(pred_id), str(pred_id))

    return {
        "label":      pred_label,
        "confidence": round(probs[pred_id], 4),
        "scores": {
            ID2LABEL.get(str(i), str(i)): round(p, 4)
            for i, p in enumerate(probs)
        },
    }


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok", 
        "model": "distilbert-yt-sentiment", 
        "device": str(DEVICE),
        "labels": list(ID2LABEL.values())
    })


@app.route("/predict", methods=["POST"])
    # Batch texts
    if "texts" in data:
        texts = data["texts"]
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"error": "texts must be a non-empty list"}), 400
        results = [{"input": str(t), **predict_text(str(t))} for t in texts]
        return jsonify({"predictions": results})

    # Single text fallback
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Provide 'text' or 'texts' in request body"}), 400
    
    result = predict_text(text)
    return jsonify({"input": text, **result})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
