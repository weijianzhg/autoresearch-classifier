"""
Autoresearch classifier training script.
Single-file, scikit-learn only.
Usage: python train.py
"""

import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from prepare import load_splits, evaluate

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

MAX_FEATURES = 50_000
NGRAM_RANGE = (1, 2)
SUBLINEAR_TF = True
C = 1.0
MAX_ITER = 1000

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

X_train, y_train, X_val, y_val, X_test, y_test = load_splits()

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=SUBLINEAR_TF,
        strip_accents="unicode",
        analyzer="word",
    )),
    ("clf", LogisticRegression(
        C=C,
        max_iter=MAX_ITER,
        solver="lbfgs",
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
