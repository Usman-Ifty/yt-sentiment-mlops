import pandas as pd
import re
import os
import json
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv

load_dotenv()

# ── config ────────────────────────────────────────────────
RAW_PATH      = "data/raw/YoutubeCommentsDataSet.csv"
PROCESSED_DIR = "data/processed"
MODEL_CKPT    = "distilbert-base-uncased"
MAX_LEN       = 128

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ── helpers ───────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"[^a-z0-9\s\'\.,!?]", "", text)  # keep basic punctuation
    text = re.sub(r"\s+", " ", text)              # collapse whitespace
    return text

def encode_label(label: str) -> int:
    label = str(label).strip().lower()
    # dataset has 'positive','negative','neutral' but also variants
    for key in LABEL2ID:
        if key in label:
            return LABEL2ID[key]
    return -1   # unknown — will be dropped

# ── load ──────────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    # normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # keep only the two columns we need
    df = df[["comment", "sentiment"]].copy()

    # drop nulls
    df.dropna(inplace=True)

    # clean text
    df["comment"] = df["comment"].apply(clean_text)

    # drop empty strings after cleaning
    df = df[df["comment"].str.len() > 5]

    # encode labels
    df["label"] = df["sentiment"].apply(encode_label)
    df = df[df["label"] != -1]   # drop unknowns

    print(f"After cleaning: {len(df)} rows")
    print(df["label"].value_counts())

    return df

# ── split ─────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    # 80% train, 10% val, 10% test  — stratified to respect class imbalance
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df

# ── tokenise ──────────────────────────────────────────────
def tokenize_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    dataset = Dataset.from_pandas(
        df[["comment", "label"]].reset_index(drop=True)
    )

    def tokenize_fn(batch):
        return tokenizer(
            batch["comment"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    return dataset

# ── main ──────────────────────────────────────────────────
def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. load and clean
    df = load_and_clean(RAW_PATH)

    # 2. split
    train_df, val_df, test_df = split_data(df)

    # 3. tokenizer
    print(f"Loading tokenizer: {MODEL_CKPT}")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_CKPT)

    # 4. tokenise all splits
    dataset_dict = DatasetDict({
        "train": tokenize_dataset(train_df, tokenizer),
        "val":   tokenize_dataset(val_df,   tokenizer),
        "test":  tokenize_dataset(test_df,  tokenizer),
    })

    # 5. save to disk
    dataset_dict.save_to_disk(os.path.join(PROCESSED_DIR, "tokenized"))
    tokenizer.save_pretrained(os.path.join(PROCESSED_DIR, "tokenizer"))

    # 6. save label map for API use later
    with open(os.path.join(PROCESSED_DIR, "label_map.json"), "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)

    # 7. save class weights for handling imbalance during training
    total = len(df)
    class_counts = df["label"].value_counts().sort_index()
    weights = [total / (3 * class_counts[i]) for i in range(3)]

    with open(os.path.join(PROCESSED_DIR, "class_weights.json"), "w") as f:
        json.dump({"weights": weights}, f, indent=2)

    print("Class weights (neg, neu, pos):", [round(w, 3) for w in weights])
    print(f"\nDone! Saved to {PROCESSED_DIR}/")
    print("  tokenized/   ← HuggingFace DatasetDict")
    print("  tokenizer/   ← DistilBERT tokenizer files")
    print("  label_map.json")
    print("  class_weights.json")

if __name__ == "__main__":
    main()
