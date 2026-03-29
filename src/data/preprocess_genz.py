"""
Gen Z Preprocessor
===================
Reads the master_genz_sentiment.csv, applies slang normalization,
tokenizes for DistilBERT, and saves to data/processed/tokenized.

Run: python src/data/preprocess_genz.py
"""

import os
import json
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset, DatasetDict
from transformers import DistilBertTokenizer

# --------------------------------------------------------------------------
# Gen Z Slang Normalization Dictionary
# Extended to cover maximum internet/YouTube slang as of 2024-2025
# --------------------------------------------------------------------------
SLANG_MAP = {
    # Qualitative scores
    r'\bW\b': 'great win',
    r'\bL\b': 'bad loss',
    r'\bgoated\b': 'greatest of all time',
    r'\bgoat\b': 'greatest of all time',
    r'\bfire\b': 'amazing excellent',
    r'\blit\b': 'excellent exciting',
    r'\bbussin\b': 'delicious amazing',
    r'\bslay\b': 'impressive excellent',
    r'\bslaying\b': 'performing excellently',
    r'\bslay queen\b': 'excellent performance',
    r'\bmid\b': 'average boring mediocre',
    r'\btrash\b': 'terrible bad',
    r'\bwack\b': 'bad boring',
    r'\bbanger\b': 'excellent great song',
    r'\bslapped\b': 'excellent great',
    r'\brizz\b': 'charm charisma attractive',
    r'\bno rizz\b': 'unattractive awkward',

    # Affirmations / agreement
    r'\bfr\b': 'for real seriously',
    r'\bfr fr\b': 'seriously for real',
    r'\bno cap\b': 'honestly seriously',
    r'\bnocap\b': 'honestly seriously',
    r'\bcap\b': 'lie fake',
    r'\bfacts\b': 'true correct',
    r'\bngl\b': 'honestly',
    r'\bidk\b': 'i do not know',
    r'\bimo\b': 'in my opinion',
    r'\bimho\b': 'in my honest opinion',
    r'\bnpc\b': 'boring person no personality',
    r'\breal\b': 'genuine authentic',

    # Emotions / reactions
    r'\bdead\b': 'laughing so hard',
    r'\bdying\b': 'laughing so hard',
    r'\bi\'m dead\b': 'this is very funny',
    r'\bim dead\b': 'this is very funny',
    r'\bsending me\b': 'very funny',
    r'\bcrying\b': 'very funny hilarious',
    r'\bscreaming\b': 'very funny',
    r'\bbruh\b': 'wow unbelievable',
    r'\bbro\b': 'friend',
    r'\bsus\b': 'suspicious untrustworthy',
    r'\bawkward\b': 'uncomfortable',
    r'\bcringe\b': 'embarrassing awkward',
    r'\bvibes\b': 'feeling energy atmosphere',
    r'\bgood vibes\b': 'positive energy happy',
    r'\bbad vibes\b': 'negative energy uncomfortable',
    r'\bgaslit\b': 'manipulated deceived',
    r'\btoxic\b': 'harmful negative bad',
    r'\bsimping\b': 'admiring someone excessively',
    r'\bhype\b': 'excited enthusiastic',
    r'\bdaydreaming\b': 'imagining fantasizing',
    r'\bbody count\b': 'history record',
    r'\bglow up\b': 'improvement transformation',
    r'\bera\b': 'phase in life',
    r'\brent free\b': 'can not stop thinking about',

    # Slang intensifiers
    r'\bhits different\b': 'feels unique special',
    r'\bunderstand the assignment\b': 'did perfectly excellent job',
    r'\bit\'s giving\b': 'reminds me of feels like',
    r'\bits giving\b': 'reminds me of feels like',
    r'\bchef kiss\b': 'perfect excellent',
    r'\bperiod\b': 'end of discussion final',
    r'\bthat\'s it\b': 'that is correct perfect',
    r'\bthe way\b': 'how much i love notice',
    r'\bstayed\b': 'remained consistent',
    r'\bwe ate\b': 'we did really well',
    r'\bbody\b': 'defeated dominated',
    r'\bcooked\b': 'in trouble done for',
    r'\bbricked\b': 'broken failed',
    r'\bcleared\b': 'dominated excelled',
    r'\bstanding on business\b': 'serious committed',
    r'\btouched grass\b': 'went outside reality check',
    r'\bpressed\b': 'upset bothered',
    r'\brating\b': 'evaluation review',
    r'\bhating\b': 'criticizing jealously',
    r'\bhaters\b': 'jealous critics',
    r'\bclout\b': 'fame social influence',
    r'\bstan\b': 'devoted fan supporter',
    r'\bpoppin\b': 'popular exciting',
    r'\bboppin\b': 'excellent great music',

    # Emojis to text
    r'💀': 'laughing so hard very funny dead',
    r'🔥': 'amazing excellent fire',
    r'🙌': 'great celebrating praising',
    r'👑': 'king greatest royalty',
    r'😂': 'very funny hilarious',
    r'😭': 'crying sad funny',
    r'🤣': 'very funny laughing',
    r'❤️': 'love great',
    r'💯': 'totally agree perfect',
    r'👏': 'applauding great job',
    r'😤': 'angry frustrated',
    r'🤮': 'disgusting terrible',
    r'👎': 'bad dislike',
    r'👍': 'good like agreed',
    r'🥶': 'cold impressive',
    r'🫡': 'respect salute honor',
    r'🗣️': 'saying out loud',
    r'😮': 'surprised shocked',
    r'🤯': 'shocked mind blown',
    r'🫶': 'love support',
    r'💪': 'strong powerful',
    r'☠️': 'laughing very funny dead',
}

def normalize_slang(text):
    """Replace slang and emoji with their English equivalents."""
    for pattern, replacement in SLANG_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text.strip()

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
PROCESSED_DIR = "data/processed"
INPUT_CSV = "data/raw/master_genz_sentiment.csv"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Load & Normalize
# --------------------------------------------------------------------------
print(f"📂 Loading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
print(f"   Raw rows: {len(df):,}")

print("🔤 Normalizing Gen Z slang...")
df["comment_clean"] = df["comment"].astype(str).apply(normalize_slang)
df["label"] = df["sentiment"].map(LABEL2ID)
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)
df = df[df["comment_clean"].str.strip().str.len() > 3].reset_index(drop=True)
print(f"   Clean rows: {len(df):,}")

# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------
print("✂️  Splitting into train / val / test (80 / 10 / 10)...")
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])
print(f"   Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

# --------------------------------------------------------------------------
# Class Weights (handles imbalance)
# --------------------------------------------------------------------------
weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=train_df["label"].values)
print(f"   Class weights: {[round(w, 3) for w in weights]}")
with open(os.path.join(PROCESSED_DIR, "class_weights.json"), "w") as f:
    json.dump({"weights": weights.tolist()}, f, indent=2)

# --------------------------------------------------------------------------
# Tokenize
# --------------------------------------------------------------------------
print("🔢 Tokenizing for DistilBERT (max_length=128)...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_and_build(data_df):
    encoding = tokenizer(
        data_df["comment_clean"].tolist(),
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors=None,
    )
    return Dataset.from_dict({
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": data_df["label"].tolist(),
    })

tokenized_dataset = DatasetDict({
    "train": tokenize_and_build(train_df),
    "val":   tokenize_and_build(val_df),
    "test":  tokenize_and_build(test_df),
})

save_path = os.path.join(PROCESSED_DIR, "tokenized")
tokenized_dataset.save_to_disk(save_path)

# Save label map
with open(os.path.join(PROCESSED_DIR, "label_map.json"), "w") as f:
    json.dump({"id2label": {str(v): k for k, v in LABEL2ID.items()}, "label2id": {k: str(v) for k, v in LABEL2ID.items()}}, f, indent=2)

print(f"\n{'='*55}")
print(f"  PREPROCESSING COMPLETE")
print(f"  Tokenized data saved to: {save_path}")
print(f"  Next Step: python src/model/train.py")
print(f"{'='*55}")
