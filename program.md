# autoresearch-classifier

This is an experiment to have the LLM do its own research on a classical ML classification task.

## Goal

Train the best possible prompt-injection / jailbreak binary classifier using scikit-learn on the `neuralchemy/Prompt-injection-dataset`. The classifier should be simple, fast, and accurate — it's meant to be a lightweight guardrail layer.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar20`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare.py` — fixed dataset loading and evaluation. Do not modify.
   - `train.py` — the file you modify. Feature extraction, model, pipeline.
4. **Verify data loads**: Run `python prepare.py` to confirm the dataset downloads and prints stats.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment trains a scikit-learn model on the `core` split (4,391 train / 941 val). Training is fast (seconds), so there's no time budget.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: feature extraction, model choice, hyperparameters, ensembles, feature engineering.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation function and data loading.
- Install new packages or add dependencies beyond what's in `requirements.txt`.
- Modify the evaluation function. The `evaluate()` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest val_accuracy.** The secondary metric is val_f1. Everything is fair game: change the vectorizer, the model, the hyperparameters, add feature engineering, try ensembles.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary:

```
---
val_accuracy:    0.950000
val_f1:          0.960000
val_precision:   0.940000
val_recall:      0.970000
training_seconds: 1.2
```

Extract the key metrics:

```
grep "^val_accuracy:\|^val_f1:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	val_accuracy	val_f1	status	description
```

1. git commit hash (short, 7 chars)
2. val_accuracy achieved (e.g. 0.950000) — use 0.000000 for crashes
3. val_f1 achieved (e.g. 0.960000) — use 0.000000 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_accuracy	val_f1	status	description
a1b2c3d	0.950000	0.960000	keep	baseline TF-IDF + LogisticRegression
b2c3d4e	0.955000	0.963000	keep	add char n-grams (2,5)
c3d4e5f	0.948000	0.958000	discard	switch to NaiveBayes
d4e5f6g	0.000000	0.000000	crash	stacking ensemble (memory error)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar20`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1`
5. Read out the results: `grep "^val_accuracy:\|^val_f1:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_accuracy improved (higher), you "advance" the branch, keeping the git commit
9. If val_accuracy is equal or worse, you git reset back to where you started

## Experiment ideas

Here are some directions to explore, roughly ordered by expected impact:

**Feature extraction**
- Char n-grams (e.g. `analyzer="char_wb"`, `ngram_range=(2,5)`) — captures subword patterns common in injections
- Combined word + char features using `FeatureUnion`
- Different max_features values (10k, 50k, 100k)
- Custom text preprocessing (lowercase special tokens, normalize whitespace)
- Hand-crafted features: text length, special character ratio, keyword counts, uppercase ratio

**Models**
- SVM (`LinearSVC` or `SVC` with RBF kernel)
- Random Forest
- Gradient Boosting (`GradientBoostingClassifier` or `HistGradientBoostingClassifier`)
- Voting or Stacking ensembles combining top performers

**Hyperparameter tuning**
- Regularization strength (C for LR/SVM)
- TF-IDF parameters (min_df, max_df, sublinear_tf)
- For tree models: depth, n_estimators, min_samples_split

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. You are autonomous. If you run out of ideas, think harder — re-read the code, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you.
