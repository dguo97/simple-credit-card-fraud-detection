# Simple Credit Card Fraud Detection

![CI](https://github.com/dguo97/simple-credit-card-fraud-detection/actions/workflows/ci.yml/badge.svg)

A small, self-contained fraud detector that combines unsupervised anomaly
detection with a supervised classifier:

1. An **Isolation Forest** scores how unusual each transaction looks.
2. That anomaly score is added as an extra feature to a **Random Forest**,
   which outputs the final fraud probability.

**It uses synthetic data throughout.** The transactions are generated on the
fly, so there is nothing to download, no dataset to set up, and it is easy to
type in any transaction and see how the model reacts.

## Quick start

```bash
git clone https://github.com/dguo97/simple-credit-card-fraud-detection.git
cd simple-credit-card-fraud-detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python fraud_detection.py
```

That starts the interactive checker: it asks for an amount, merchant category,
transaction type and whether it is a weekend, then prints a fraud probability
and a verdict. The first run trains the model (a few seconds) and saves it to
`fraud_model.joblib`; later runs load it.

## Try it

You can also pass a transaction directly:

```bash
python fraud_detection.py --amount 950 --category Travel --type Online --weekend
python fraud_detection.py --amount 12 --category Grocery --type In-Store --threshold 0.3
```

Some examples and what the model says:

| Transaction | Fraud probability | Verdict |
|-------------|-------------------|---------|
| $12.00, Grocery, In-Store | 0.0% | Normal |
| $45.50, Dining, In-Store | 0.0% | Normal |
| $150.00, Retail, Online | 2.7% | Normal |
| $8.00, Retail, Online | 25.7% | Normal (small online charges look a bit like "card testing") |
| $450.00, Electronics, Online, weekend | 99.5% | Fraud |
| $950.00, Travel, Online, weekend | about 100% | Fraud |

Options:

| Flag | Meaning |
|------|---------|
| `--amount`, `--category`, `--type`, `--weekend` | Transaction to score (give amount, category and type together) |
| `--threshold 0.3` | Probability above which a transaction is flagged (default 0.5). Lower it to catch more fraud at the cost of more false alarms |
| `--retrain` | Ignore any saved model and train a fresh one |
| `--model-path PATH` | Where to save/load the model |

Categories: Grocery, Electronics, Travel, Dining, Retail. Types: In-Store,
Online, ATM. Input is case-insensitive, and invalid input is rejected with a
clear message. Only load model files you created yourself, since joblib files
are pickles.

## Evaluate the model

```bash
python fraud_detection.py evaluate --plots
```

This generates a fresh synthetic dataset, makes a stratified 80/20 train/test
split, fits everything on the training split only, and prints a comparison table
and a classification report. `--plots` saves `confusion_matrix.png` and
`pr_curve.png`. `--n-samples` changes the dataset size (default 20,000).

### Results

20,000 synthetic transactions (3% fraud), threshold 0.5:

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|-------|-----------|--------|----|--------|---------|
| Isolation Forest only | 0.406 | 0.454 | 0.429 | 0.373 | 0.890 |
| Random Forest only | 0.328 | 0.664 | 0.439 | 0.597 | 0.927 |
| **Hybrid (IF score + RF)** | 0.446 | 0.588 | 0.507 | 0.618 | 0.953 |

That is a single split with 119 fraud cases in the test set, so treat small
differences with caution. Over five different seeds the hybrid had the higher
PR-AUC in four (mean 0.626 vs 0.603 for the plain Random Forest, standard
deviation about 0.015 for both). The anomaly score helps a little, not a lot.

PR-AUC is the headline metric: with only 3% fraud, accuracy is meaningless
because predicting "normal" every time already scores 97%.

## Use it from Python

```python
from fraud_detection import HybridFraudModel, generate_synthetic_data, score_transaction

X, y = generate_synthetic_data(20000)
model = HybridFraudModel().fit(X, y)
score_transaction(model, 950.0, "Travel", "Online", is_weekend=True)
```

## How it works

| Step | Detail |
|------|--------|
| Data | Synthetic fraud cases are drawn from their own distribution (larger amounts, more electronics/travel, mostly online, some small card-testing charges), so the classes overlap realistically rather than following a simple rule |
| Preprocessing | `RobustScaler` on the amount, one-hot encoding for category and type |
| Anomaly model | `IsolationForest`, with `contamination` set to the fraud rate of the training data |
| Anomaly feature | Computed **out-of-fold** (5-fold): each training row is scored by a forest that never saw it, so the classifier trains on scores that look like the ones it sees on new data |
| Classifier | `RandomForestClassifier(n_estimators=100, min_samples_leaf=3, class_weight="balanced_subsample")` |

## Project layout

```
fraud_detection.py       data generation, model, evaluation, CLI
tests/                   pytest suite
requirements.txt         runtime dependencies
requirements-dev.txt     test dependencies
.github/workflows/ci.yml runs the tests on Python 3.10 to 3.13
```

Run the tests with:

```bash
pip install -r requirements-dev.txt
pytest
```

## Limitations

- The model is trained on **synthetic** data, so its probabilities describe the
  simulation, not real-world fraud. It demonstrates the pipeline and is not a
  production fraud system.
- Real fraud data is anonymised, imbalanced and drifts over time. Testing on a
  real dataset with a time-based split would be the natural next step.

## License

MIT, Copyright (c) 2026 David Guo (see `LICENSE`).
