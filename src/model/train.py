import os
import json
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
from torch import nn
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from dotenv import load_dotenv

load_dotenv()

# ── config ────────────────────────────────────────────────
PROCESSED_DIR  = "data/processed"
MODEL_CKPT     = "distilbert-base-uncased"
MODEL_NAME     = os.getenv("MODEL_NAME", "yt-sentiment-bert")
OUTPUT_DIR     = "models/bert"

# Hyperparameters — logged to MLflow
CONFIG = {
    "model_checkpoint" : MODEL_CKPT,
    "max_len"          : 128,
    "batch_size"       : 32,
    "epochs"           : 4,
    "learning_rate"    : 2e-5,
    "warmup_ratio"     : 0.1,
    "weight_decay"     : 0.01,
    "num_labels"       : 3,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── data loaders ──────────────────────────────────────────
def get_dataloaders():
    dataset = load_from_disk(os.path.join(PROCESSED_DIR, "tokenized"))

    train_loader = DataLoader(
        dataset["train"],
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset["val"],
        batch_size=CONFIG["batch_size"] * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset["test"],
        batch_size=CONFIG["batch_size"] * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ── model ─────────────────────────────────────────────────
def build_model():
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_CKPT,
        num_labels=CONFIG["num_labels"],
    )
    model.to(DEVICE)
    return model


# ── weighted loss for class imbalance ─────────────────────
def get_loss_fn():
    with open(os.path.join(PROCESSED_DIR, "class_weights.json")) as f:
        weights = json.load(f)["weights"]
    weight_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    print(f"Class weights: {[round(w, 3) for w in weights]}")
    return nn.CrossEntropyLoss(weight=weight_tensor)


# ── evaluation helper ─────────────────────────────────────
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="weighted")

    return avg_loss, accuracy, f1, all_preds, all_labels


# ── training loop ─────────────────────────────────────────
def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders()
    model   = build_model()
    loss_fn = get_loss_fn()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    total_steps   = len(train_loader) * CONFIG["epochs"]
    warmup_steps  = int(total_steps * CONFIG["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── MLflow run ────────────────────────────────────────
    mlflow.set_experiment(MODEL_NAME)

    with mlflow.start_run() as run:
        print(f"\nMLflow Run ID: {run.info.run_id}")

        # log all hyperparameters at once
        mlflow.log_params(CONFIG)

        best_val_f1   = 0.0
        best_model_path = os.path.join(OUTPUT_DIR, "best_model")

        for epoch in range(CONFIG["epochs"]):
            # ── train one epoch ───────────────────────────
            model.train()
            total_train_loss = 0

            for step, batch in enumerate(train_loader):
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels         = batch["labels"].to(DEVICE)

                optimizer.zero_grad()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss    = loss_fn(outputs.logits, labels)

                loss.backward()

                # gradient clipping prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()

                if step % 100 == 0:
                    print(f"  Epoch {epoch+1} | Step {step}/{len(train_loader)} "
                          f"| Loss: {loss.item():.4f}")

            avg_train_loss = total_train_loss / len(train_loader)

            # ── validate ──────────────────────────────────
            val_loss, val_acc, val_f1, _, _ = evaluate(
                model, val_loader, loss_fn
            )

            print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
            print(f"  Train loss : {avg_train_loss:.4f}")
            print(f"  Val loss   : {val_loss:.4f}")
            print(f"  Val acc    : {val_acc:.4f}")
            print(f"  Val F1     : {val_f1:.4f}")

            # log metrics to MLflow — step = epoch number
            mlflow.log_metrics({
                "train_loss" : round(avg_train_loss, 4),
                "val_loss"   : round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
                "val_f1"     : round(val_f1, 4),
            }, step=epoch)

            # save best model based on val F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                model.save_pretrained(best_model_path)
                print(f"  Saved best model (F1={val_f1:.4f})")

        # ── final test evaluation ─────────────────────────
        print("\nEvaluating on test set...")
        best_model = DistilBertForSequenceClassification.from_pretrained(
            best_model_path
        ).to(DEVICE)

        test_loss, test_acc, test_f1, preds, labels = evaluate(
            best_model, test_loader, loss_fn
        )

        print(f"\nTest accuracy : {test_acc:.4f}")
        print(f"Test F1       : {test_f1:.4f}")

        label_names = ["negative", "neutral", "positive"]
        print("\nClassification report:")
        print(classification_report(labels, preds, target_names=label_names))

        # log final test metrics
        mlflow.log_metrics({
            "test_accuracy" : round(test_acc, 4),
            "test_f1"       : round(test_f1, 4),
            "test_loss"     : round(test_loss, 4),
        })

        # save classification report as artifact
        report = classification_report(
            labels, preds, target_names=label_names, output_dict=True
        )
        report_path = os.path.join(OUTPUT_DIR, "classification_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)

        # ── register model in MLflow ──────────────────────
        mlflow.pytorch.log_model(
            pytorch_model=best_model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        print(f"\nModel registered as '{MODEL_NAME}' in MLflow")
        print(f"Best Val F1  : {best_val_f1:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1      : {test_f1:.4f}")
        print(f"Run ID       : {run.info.run_id}")

        # save metrics file for DVC tracking
        os.makedirs("models/bert", exist_ok=True)
        with open("models/bert/metrics.json", "w") as f:
            json.dump({
                "test_accuracy" : round(test_acc, 4),
                "test_f1"       : round(test_f1, 4),
                "val_f1"        : round(best_val_f1, 4),
            }, f, indent=2)

        return run.info.run_id


if __name__ == "__main__":
    train()
