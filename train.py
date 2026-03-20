"""
Autoresearch classifier training script.
Single-file, scikit-learn only.
Usage: uv run train.py
"""

import time
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import hstack, csr_matrix

from prepare import load_splits, evaluate

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

MAX_FEATURES = 50_000
NGRAM_RANGE = (1, 3)
SUBLINEAR_TF = True
C = 1.0
MAX_ITER = 1000

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

X_train, y_train, X_val, y_val, X_test, y_test = load_splits()


# ---------------------------------------------------------------------------
# Hand-crafted feature extractor
# ---------------------------------------------------------------------------

class TextFeatures(BaseEstimator, TransformerMixin):
    """Extract numeric features from raw text."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = np.zeros((len(X), 5), dtype=np.float64)
        for i, text in enumerate(X):
            feats[i, 0] = len(text)
            feats[i, 1] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
            feats[i, 2] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
            feats[i, 3] = text.count('\n')
            feats[i, 4] = len(re.findall(r'[{}()\[\]<>]', text)) / max(len(text), 1)
        return csr_matrix(feats)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

pipeline = Pipeline([
    ("features", FeatureUnion([
        ("word", TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
            sublinear_tf=SUBLINEAR_TF,
            strip_accents="unicode",
            analyzer="word",
        )),
        ("char", TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=(2, 6),
            sublinear_tf=SUBLINEAR_TF,
            strip_accents="unicode",
            analyzer="char_wb",
        )),
        ("meta", Pipeline([
            ("extract", TextFeatures()),
            ("scale", MaxAbsScaler()),
        ])),
    ])),
    ("clf", LinearSVC(
        C=C,
        max_iter=MAX_ITER,
        random_state=42,
    )),
])

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

t0 = time.time()
pipeline.fit(X_train, y_train)
train_time = time.time() - t0

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

y_pred_val = pipeline.predict(X_val)
print("=== Validation Results ===")
val_results = evaluate(y_val, y_pred_val)

# ---------------------------------------------------------------------------
# Parseable output
# ---------------------------------------------------------------------------

print("---")
print(f"val_accuracy:    {val_results['accuracy']:.6f}")
print(f"val_f1:          {val_results['f1']:.6f}")
print(f"val_precision:   {val_results['precision']:.6f}")
print(f"val_recall:      {val_results['recall']:.6f}")
print(f"training_seconds: {train_time:.1f}")
