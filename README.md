# Credit Card Fraud Detection (Interactive Demo)

A small command-line tool that scores a credit card transaction for fraud risk
using a hybrid machine-learning approach:

1. An **Isolation Forest** learns what "normal" transactions look like and
   produces an anomaly score for each one.
2. That score is added as an extra feature to a **Random Forest classifier**,
   which outputs the final fraud probability.

You enter a transaction (amount, merchant category, type, weekend or not) and
the program prints a fraud probability and a verdict.

> **Note:** The model is trained on *synthetic* data generated at startup, and
> the fraud labels come from simple hand-written rules (very large amounts, or
> large online travel purchases, with some random noise). This is a
> demonstration of the pipeline, not a production fraud system, and it should
> not be used on real financial data without proper datasets and validation.

## Features

- Synthetic dataset of 5,000 transactions (reproducible via a fixed seed)
- Hybrid anomaly-score + classifier pipeline built with scikit-learn
- Input validation with friendly re-prompting
- Reusable `build_model()` and `predict()` functions for use in other code

## Installation

```bash
git clone https://github.com/<your-username>/fraud-detection.git
cd fraud-detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python fraud_detection.py
```

Example session:

```
1. Enter Transaction Amount (e.g., 45.50 or 1200.00): $1200
2. Available Merchant Categories: Grocery, Electronics, Travel, Dining, Retail
   Select or type a category from the list above: electronics
3. Available Transaction Types: In-Store, Online, ATM
   Select transaction type from the list above: online
4. Is this transaction taking place on a weekend? (y/n): n

📊 ANALYSIS RESULTS
• Calculated Fraud Probability: ...
🚨 VERDICT: High Risk / Potential Fraud Detected!
```

Using it from Python:

```python
from fraud_detection import build_model, predict

models = build_model()
prediction, probability = predict(models, 950.0, "Travel", "Online", 1)
```

## How it works

| Step | Detail |
|------|--------|
| Data | Exponential transaction amounts, random categories/types/weekend flag |
| Labels | Fraud if amount > $800, or Travel + Online + amount > $300 (70% of the time) |
| Features | Robust-scaled amount, label-encoded category and type, weekend flag |
| Anomaly model | `IsolationForest(contamination=0.03)` |
| Classifier | `RandomForestClassifier(n_estimators=100, class_weight="balanced")` |

## Ideas for improvement

- Train on a real dataset (e.g. the public Kaggle credit card fraud data)
- Add a train/test split and report precision, recall and PR-AUC
- Save and load the trained model with `joblib` instead of retraining each run
- Add unit tests and a command-line argument interface

## License

MIT (see `LICENSE`).
