import torch
import json
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

MODEL_PATH     = "models/bert/best_model"
TOKENIZER_PATH = "data/processed/tokenizer"
LABEL_MAP_PATH = "data/processed/label_map.json"

def load_model():
    model     = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH)
    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    return model, tokenizer, label_map

def predict(text: str, model, tokenizer, id2label: dict) -> dict:
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs     = torch.softmax(logits, dim=1).squeeze().tolist()
    label_idx = int(torch.argmax(logits, dim=1))
    return {
        "label"       : id2label[str(label_idx)],
        "confidence"  : round(probs[label_idx], 4),
        "probabilities": {
            id2label[str(i)]: round(p, 4) for i, p in enumerate(probs)
        },
    }

def test_positive_comment():
    model, tokenizer, lmap = load_model()
    result = predict(
        "apple pay is so convenient and easy to use",
        model, tokenizer, lmap["id2label"]
    )
    assert result["label"] == "positive"

def test_negative_comment():
    model, tokenizer, lmap = load_model()
    result = predict(
        "this is absolutely terrible and broken",
        model, tokenizer, lmap["id2label"]
    )
    assert result["label"] == "negative"

def test_output_has_probabilities():
    model, tokenizer, lmap = load_model()
    result = predict("it works fine", model, tokenizer, lmap["id2label"])
    assert "probabilities" in result
    assert abs(sum(result["probabilities"].values()) - 1.0) < 0.01
