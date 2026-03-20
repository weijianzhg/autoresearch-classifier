# autoresearch-classifier

An autonomous experiment loop that trains classical ML classifiers to detect prompt injection and jailbreak attacks against LLMs.

Uses the [neuralchemy/Prompt-injection-dataset](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset) (`core` config: 4,391 train / 941 val / 942 test, ~60% malicious) and scikit-learn models.

## Why

Transformer-based guardrails are expensive and slow. A simple non-transformer classifier (logistic regression, SVM, etc.) can serve as a fast, predictable first line of defense against prompt injections. This project explores how far classical ML can go on this task.

## How it works

An LLM agent edits `train.py` in a loop, trying different models, features, and hyperparameters. After each run it checks whether validation accuracy improved — if yes, the commit stays; if not, it gets reverted. Results are logged to `results.tsv`.

```
prepare.py   — dataset loading + evaluation (read-only, do not modify)
train.py     — feature extraction + model pipeline (agent edits this)
program.md   — agent instructions for the experiment loop
analysis.ipynb — notebook for visualizing results.tsv
```

## Quick start

```bash
uv sync
uv run prepare.py        # download dataset, print stats
uv run train.py           # run baseline (TF-IDF + LogisticRegression)
```

## Running experiments

Read `program.md` for the full protocol. The short version:

```bash
git checkout -b autoresearch/<tag>
uv run train.py                        # baseline
# agent loop: edit train.py → commit → run → keep or revert
```

## Dataset

| Split | Samples | Benign | Malicious |
|-------|---------|--------|-----------|
| train | 4,391   | 1,741  | 2,650     |
| val   | 941     | 407    | 534       |
| test  | 942     | 390    | 552       |

Binary classification — 29 attack categories including 2025 techniques, zero data leakage (group-aware splitting).

## Baseline

TF-IDF (word unigrams+bigrams, 50k features) + LogisticRegression:

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9426 |
| F1        | 0.9514 |
| Precision | 0.9167 |
| Recall    | 0.9888 |
| Train time| 0.3s   |
