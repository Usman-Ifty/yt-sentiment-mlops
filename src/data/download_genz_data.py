"""
Gen Z Slang Dataset Builder
============================
Downloads and merges modern social media / Gen Z sentiment data
with the original YouTube training data.

Run: python src/data/download_genz_data.py
"""

import os
import re
import pandas as pd
from datasets import load_dataset

OUTPUT_PATH = "data/raw/master_genz_sentiment.csv"
ORIGINAL_YT = "data/raw/youtube_comments.csv"   # your original file — adjust name if different

os.makedirs("data/raw", exist_ok=True)

frames = []

# --------------------------------------------------------------------------
# 1. Cardiff NLP Tweet Sentiment (English) - high quality 3-class labels
#    Labels: 0=negative, 1=neutral, 2=positive
# --------------------------------------------------------------------------
print("📥 [1/4] Downloading Cardiff NLP Tweet Sentiment...")
try:
    ds = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "english")
    for split in ["train", "validation", "test"]:
        df = pd.DataFrame(ds[split])
        df = df.rename(columns={"text": "comment", "label": "sentiment"})
        df["sentiment"] = df["sentiment"].map({0: "negative", 1: "neutral", 2: "positive"})
        frames.append(df[["comment", "sentiment"]])
    print(f"   ✅ Cardiff NLP loaded: {sum(len(pd.DataFrame(ds[s])) for s in ['train','validation','test'])} rows")
except Exception as e:
    print(f"   ⚠️ Cardiff failed: {e}")


# --------------------------------------------------------------------------
# 2. MTEB Tweet Sentiment Extraction — very modern tweet slang
#    Labels: negative, neutral, positive
# --------------------------------------------------------------------------
print("📥 [2/4] Downloading MTEB Tweet Sentiment...")
try:
    ds2 = load_dataset("mteb/tweet_sentiment_extraction")
    for split in ds2.keys():
        df = pd.DataFrame(ds2[split])
        if "text" in df.columns and "label_text" in df.columns:
            df = df.rename(columns={"text": "comment", "label_text": "sentiment"})
            frames.append(df[["comment", "sentiment"]])
    total = sum(len(pd.DataFrame(ds2[s])) for s in ds2.keys())
    print(f"   ✅ MTEB loaded: {total} rows")
except Exception as e:
    print(f"   ⚠️ MTEB failed: {e}")


# --------------------------------------------------------------------------
# 3. Multiclass Sentiment Analysis Dataset — mixed social media sources
#    Labels: 0=negative, 1=neutral, 2=positive
# --------------------------------------------------------------------------
print("📥 [3/4] Downloading Multiclass Social Media Sentiment...")
try:
    ds3 = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset")
    for split in ds3.keys():
        df = pd.DataFrame(ds3[split])
        if "text" in df.columns and "label" in df.columns:
            df = df.rename(columns={"text": "comment", "label": "sentiment"})
            df["sentiment"] = df["sentiment"].map({0: "negative", 1: "neutral", 2: "positive"})
            frames.append(df[["comment", "sentiment"]])
    total = sum(len(pd.DataFrame(ds3[s])) for s in ds3.keys())
    print(f"   ✅ Multiclass dataset loaded: {total} rows")
except Exception as e:
    print(f"   ⚠️ Multiclass dataset failed: {e}")


# --------------------------------------------------------------------------
# 4. Your original YouTube dataset — if it exists
# --------------------------------------------------------------------------
print("📥 [4/4] Loading original YouTube comments...")
yt_found = False
for fname in os.listdir("data/raw"):
    if fname.endswith(".csv"):
        try:
            yt = pd.read_csv(f"data/raw/{fname}")
            # Try to detect comment and sentiment columns flexibly
            col_map = {}
            for c in yt.columns:
                if c.lower() in ["comment", "text", "review", "content", "comment_text"]:
                    col_map["comment"] = c
                if c.lower() in ["sentiment", "label", "class", "category"]:
                    col_map["sentiment"] = c
            if "comment" in col_map and "sentiment" in col_map:
                yt = yt.rename(columns={col_map["comment"]: "comment", col_map["sentiment"]: "sentiment"})
                yt["sentiment"] = yt["sentiment"].astype(str).str.lower().str.strip()
                yt = yt[yt["sentiment"].isin(["negative", "neutral", "positive"])]
                frames.append(yt[["comment", "sentiment"]])
                print(f"   ✅ YouTube ({fname}): {len(yt)} rows")
                yt_found = True
                break
        except Exception:
            continue

if not yt_found:
    print("   ⚠️ No YouTube CSV detected in data/raw — skipping")


# --------------------------------------------------------------------------
# Merge, Clean, Deduplicate
# --------------------------------------------------------------------------
print("\n🔧 Merging and cleaning all datasets...")

master = pd.concat(frames, ignore_index=True)

# Drop rows where comment is empty
master = master.dropna(subset=["comment", "sentiment"])
master["comment"] = master["comment"].astype(str).str.strip()
master = master[master["comment"].str.len() > 3]

# Normalize labels
master["sentiment"] = master["sentiment"].str.lower().str.strip()
master = master[master["sentiment"].isin(["negative", "neutral", "positive"])]

# Remove duplicates
master = master.drop_duplicates(subset=["comment"]).reset_index(drop=True)

# Save
master.to_csv(OUTPUT_PATH, index=False)

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
print(f"\n{'='*55}")
print(f"  MASTER DATASET BUILT SUCCESSFULLY")
print(f"  Total records    : {len(master):,}")
print(f"  Output path      : {OUTPUT_PATH}")
print(f"\n  Label distribution:")
for label, count in master["sentiment"].value_counts().items():
    pct = (count / len(master)) * 100
    bar = "#" * int(pct // 2)
    print(f"    {label:10s} | {bar:<25} {count:6,} ({pct:.1f}%)")
print(f"{'='*55}")
print("\nNext Step: python src/data/preprocess_genz.py")
