"""
One-time data preparation for autoresearch-classifier experiments.
Downloads the prompt-injection dataset from HuggingFace and exposes helpers.

Usage:
    python prepare.py          # download dataset and print stats
    python prepare.py --peek   # also print first 3 samples per split

Data is cached by the `datasets` library in ~/.cache/huggingface/.
This file is READ-ONLY for the agent. Do not modify.
"""

import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

DATASET_ID = "neuralchemy/Prompt-injection-dataset"
DATASET_CONFIG = "core"

_HF_TOKEN = None


def _get_hf_token():
    global _HF_TOKEN
    if _HF_TOKEN is not None:
        return _HF_TOKEN
    env_path = os.path.join(os.path.dirname(__file__), ".env_haggingface")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("TOKEN="):
                    _HF_TOKEN = line.split("=", 1)[1]
                    return _HF_TOKEN
    _HF_TOKEN = os.environ.get("HF_TOKEN", None)
    return _HF_TOKEN


_cached_splits = None


def load_splits():
    """
    Load train/val/test splits from the HuggingFace dataset.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
        where X is a list of strings and y is a numpy int array (0=benign, 1=malicious).
    """
    global _cached_splits
    if _cached_splits is not None:
        return _cached_splits

    from datasets import load_dataset

    token = _get_hf_token()
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, token=token)

    def _extract(split):
        texts = split["text"]
        labels = np.array(split["label"], dtype=np.int32)
        return texts, labels

    X_train, y_train = _extract(ds["train"])
    X_val, y_val = _extract(ds["validation"])
    X_test, y_test = _extract(ds["test"])

    _cached_splits = (X_train, y_train, X_val, y_val, X_test, y_test)
    return _cached_splits


def evaluate(y_true, y_pred):
    """
    Evaluate predictions and print a summary.

    Returns a dict with accuracy, f1, precision, recall, and confusion_matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    prec = precision_score(y_true, y_pred, average="binary")
    rec = recall_score(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred)

    results = {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm,
    }

    print(classification_report(y_true, y_pred, target_names=["benign", "malicious"]))
    print(f"Confusion matrix:\n{cm}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare prompt-injection dataset")
    parser.add_argument("--peek", action="store_true", help="Print sample rows")
    args = parser.parse_args()

    print(f"Dataset: {DATASET_ID} (config={DATASET_CONFIG})")
    print()

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()

    for name, X, y in [("train", X_train, y_train),
                        ("val", X_val, y_val),
                        ("test", X_test, y_test)]:
        n_mal = int(y.sum())
        n_ben = len(y) - n_mal
        print(f"  {name:6s}: {len(y):5d} samples  "
              f"({n_ben} benign / {n_mal} malicious, "
              f"{100*n_mal/len(y):.1f}% malicious)")

    if args.peek:
        print("\n--- Sample rows (train) ---")
        for i in range(min(3, len(X_train))):
            label_str = "malicious" if y_train[i] == 1 else "benign"
            text_preview = X_train[i][:120].replace("\n", " ")
            print(f"  [{label_str:9s}] {text_preview}...")

    print("\nDone. Data is cached by the datasets library.")
