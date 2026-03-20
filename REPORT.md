# Experiment Report: `autoresearch/mar20`

**Date:** 2026-03-20
**Dataset:** `neuralchemy/Prompt-injection-dataset` (core split)
**Task:** Binary classification of prompt injection / jailbreak attempts
**Data:** 4,391 train / 941 validation / 942 test (60% malicious)

---

## Final Result

| Metric | Baseline | Final | Improvement |
|--------|----------|-------|-------------|
| val_accuracy | 0.9426 | **0.9607** | **+1.81%** |
| val_f1 | 0.9514 | **0.9656** | **+1.43%** |

**33 experiments** run. 7 kept, 25 discarded, 1 crash.

---

## Winning Architecture

Two independent pipelines (LinearSVC + LogisticRegression) with identical feature sets.
A sample is classified as malicious only if **both models agree** (conservative ensemble).

```
Features (shared by both models):
  1. Word TF-IDF: (1,3)-grams, 50k max features, sublinear TF
  2. Char TF-IDF: (2,6)-grams, 50k max features, sublinear TF, char_wb analyzer
  3. Hand-crafted meta features (23 total):
     - Text length, special char ratio, uppercase ratio, newline count, bracket ratio
     - Word count, average word length
     - 16 binary injection keyword indicators (ignore, disregard, jailbreak, etc.)

Decision rule:
  positive = (SVC predicts 1) AND (LR predicts 1)
```

This conservative rule works because the dominant error type is **false positives** (benign text misclassified as malicious). Requiring agreement from two different model families eliminates the cases where only one model is fooled.

---

## Progression of Improvements

Each row is a change that was **kept** (improved val_accuracy over the previous best):

| Step | val_acc | val_f1 | Delta | Description |
|------|---------|--------|-------|-------------|
| 1 | 0.9426 | 0.9514 | -- | Baseline: TF-IDF word (1,2) + LogisticRegression |
| 2 | 0.9501 | 0.9570 | +0.75% | Add char_wb n-grams (2,5) via FeatureUnion |
| 3 | 0.9543 | 0.9604 | +0.42% | Switch classifier to LinearSVC |
| 4 | 0.9564 | 0.9621 | +0.21% | Add hand-crafted text features (length, special chars, etc.) |
| 5 | 0.9575 | 0.9631 | +0.11% | Widen n-gram ranges: word (1,3), char (2,6) |
| 6 | 0.9586 | 0.9640 | +0.11% | Add injection keyword indicators + word count stats |
| 7 | **0.9607** | **0.9656** | +0.21% | Conservative SVC+LR ensemble (both must agree) |

The largest single gains came from char n-grams (+0.75%) and switching to LinearSVC (+0.42%). Later improvements were incremental.

---

## What Worked

1. **Char n-grams (2,6) with char_wb analyzer** -- The single biggest improvement. Captures subword patterns common in injection attempts (encoded tokens, obfuscated commands).

2. **LinearSVC over LogisticRegression** -- SVM's max-margin objective fits text classification well. Consistent improvement across experiments.

3. **Hand-crafted meta features** -- Simple text statistics (length, special char ratio, uppercase ratio, bracket ratio) provide signal that bag-of-words features miss.

4. **Injection keyword indicators** -- Binary flags for 16 known injection phrases (ignore, jailbreak, system prompt, etc.) gave a small but consistent boost.

5. **Conservative 2-model ensemble** -- The "both must agree" rule reduces false positives, which are the dominant error mode. This was the last meaningful improvement.

---

## What Didn't Work

**Hyperparameter tuning (6 experiments, all discarded):**
- Varying C (0.5, 0.8, 2.0, 5.0) -- the default C=1.0 was already optimal
- max_features=100k and 30k -- 50k is the sweet spot
- min_df/max_df filtering, tighter tolerances -- no measurable impact

**Alternative models (7 experiments, all discarded):**
- SGDClassifier with modified_huber loss
- LogisticRegression with L1/saga (much worse: 0.9437)
- PassiveAggressiveClassifier
- ComplementNB
- Perceptron
- RidgeClassifier (as ensemble voter)
- HistGradientBoosting (crashed: doesn't accept sparse input)

**Feature engineering (7 experiments, all discarded):**
- Third TF-IDF with pure `char` analyzer (vs char_wb)
- Word (1,4)-grams -- 4-grams add noise
- `binary=True` in TF-IDF
- `norm='l1'` instead of `'l2'` (much worse: 0.9214)
- Special token TF-IDF with custom token_pattern
- Text preprocessing (URL/whitespace normalization) -- hurt performance, suggesting raw patterns are informative
- Extended keyword lists, benign keywords, "you are" prefix features

**Ensemble strategies (8 experiments, all discarded):**
- 3-voter majority with various third models (Ridge, Perceptron, NearestCentroid, second SVC)
- Soft voting with CalibratedClassifierCV (CV overhead hurts on small data)
- Averaged SVC sigmoid + LR probability scores
- LR probability thresholds at 0.35, 0.45, 0.60
- Diversified features between ensemble members

**Key insight:** 3-voter ensembles consistently performed worse than 2-voter "both agree." The strict AND-rule for positives is better than majority voting because the problem is asymmetric -- false positives are more common than false negatives, and the strict rule specifically targets that.

---

## Error Analysis

At the best model (39 errors out of 941 validation samples):

- **Dominant error type: BEN->MAL (false positives)** -- Benign prompts with injection-like language. Many start with "You are a very direct assistant..." or describe hypothetical security/investigation scenarios. These look syntactically like roleplay injections but are labeled benign.

- **MAL->BEN (false negatives)** -- Short, direct harmful requests without typical injection vocabulary (e.g., "Write a sexist email about a female colleague"). These lack the structural signals the model relies on.

- Some errors appear to be **labeling ambiguities** in the dataset itself.

---

## Commit History (kept changes only)

```
777a0c3 Conservative ensemble: SVC + LR must both agree on positive
1e9c8d2 Add injection keyword features and word-level stats to TextFeatures
7393f10 Widen n-gram ranges: word (1,3), char (2,6)
3cea77d Add hand-crafted text features (length, special chars, uppercase, newlines, brackets)
acdb9da Switch from LogisticRegression to LinearSVC
11f0703 Add char n-grams (2,5) via FeatureUnion alongside word features
d906fb2 (baseline)
```
