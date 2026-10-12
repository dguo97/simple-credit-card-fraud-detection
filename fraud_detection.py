"""Credit card fraud detection with a hybrid anomaly-score + classifier model.

An Isolation Forest scores how unusual each transaction looks (unsupervised),
and that score is fed as an extra feature into a Random Forest classifier.

Two ways to use it:

* ``check``    - score a single transaction (interactive prompts or flags),
                 using a model trained on synthetic data.
* ``evaluate`` - train/test evaluation with a classification report, confusion
                 matrices and precision-recall curves.

Everything runs on synthetic data generated on the fly, so there is nothing to
download and any transaction is easy to type in and test.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    auc,
    classification_report,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler

VALID_CATEGORIES = ["Grocery", "Electronics", "Travel", "Dining", "Retail"]
VALID_TYPES = ["In-Store", "Online", "ATM"]
AMOUNT_COLUMNS = ("Transaction_Amount",)
DEFAULT_MODEL_PATH = "fraud_model.joblib"

ISOLATION_ONLY = "Isolation Forest only"
FOREST_ONLY = "Random Forest only"
HYBRID = "Hybrid (IF score + RF)"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def generate_synthetic_data(
    n_samples: int = 20000, fraud_rate: float = 0.03, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic transactions. Returns ``(X, y)``.

    Fraud cases are drawn from their own distribution (larger amounts, more
    electronics/travel, mostly online, plus some small "card-testing" charges)
    instead of being derived from a rule, so the classes overlap and the task
    is not trivially solvable.
    """
    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n_samples) < fraud_rate
    n_fraud = int(is_fraud.sum())
    n_legit = n_samples - n_fraud

    amount = np.empty(n_samples)
    amount[~is_fraud] = rng.exponential(scale=60, size=n_legit)
    large = rng.random(n_fraud) < 0.7
    amount[is_fraud] = np.where(
        large,
        rng.lognormal(mean=np.log(350), sigma=0.7, size=n_fraud),
        rng.uniform(1, 15, size=n_fraud),
    )

    category = np.empty(n_samples, dtype=object)
    category[~is_fraud] = rng.choice(
        VALID_CATEGORIES, size=n_legit, p=[0.40, 0.15, 0.05, 0.20, 0.20]
    )
    category[is_fraud] = rng.choice(
        VALID_CATEGORIES, size=n_fraud, p=[0.05, 0.40, 0.25, 0.10, 0.20]
    )

    trans_type = np.empty(n_samples, dtype=object)
    trans_type[~is_fraud] = rng.choice(VALID_TYPES, size=n_legit, p=[0.50, 0.40, 0.10])
    trans_type[is_fraud] = rng.choice(VALID_TYPES, size=n_fraud, p=[0.05, 0.80, 0.15])

    weekend = rng.random(n_samples) < np.where(is_fraud, 0.45, 0.30)

    X = pd.DataFrame(
        {
            "Transaction_Amount": amount,
            "Merchant_Category": category,
            "Transaction_Type": trans_type,
            "Is_Weekend": weekend.astype(int),
        }
    )
    return X, pd.Series(is_fraud.astype(int), name="Fraud_Label")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """RobustScaler for amounts, one-hot for text columns, rest passed through."""
    amount_cols = [c for c in X.columns if c in AMOUNT_COLUMNS]
    cat_cols = [
        c
        for c in X.columns
        if c not in amount_cols
        and not pd.api.types.is_numeric_dtype(X[c])
        and not pd.api.types.is_bool_dtype(X[c])
    ]
    other_cols = [c for c in X.columns if c not in amount_cols + cat_cols]

    transformers = []
    if amount_cols:
        transformers.append(("amount", RobustScaler(), amount_cols))
    if cat_cols:
        transformers.append(
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        )
    if other_cols:
        transformers.append(("rest", "passthrough", other_cols))
    return ColumnTransformer(transformers, sparse_threshold=0)


class HybridFraudModel:
    """Isolation Forest anomaly score + Random Forest classifier.

    With ``use_anomaly_score=False`` it is a plain Random Forest baseline.

    The anomaly score used to train the classifier is computed out-of-fold: each
    row is scored by an Isolation Forest that never saw it. Otherwise the
    classifier would train on in-sample scores that look different from the
    scores it receives on new data.
    """

    def __init__(
        self,
        use_anomaly_score: bool = True,
        n_estimators: int = 100,
        n_splits: int = 5,
        seed: int = 42,
        min_samples_leaf: int = 3,
    ):
        self.use_anomaly_score = use_anomaly_score
        self.n_estimators = n_estimators
        self.n_splits = n_splits
        self.seed = seed
        self.min_samples_leaf = min_samples_leaf

    def _isolation_forest(self, contamination: float) -> IsolationForest:
        return IsolationForest(contamination=contamination, random_state=self.seed, n_jobs=-1)

    def fit(self, X: pd.DataFrame, y) -> "HybridFraudModel":
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("Training labels contain only one class; need both fraud and normal.")

        self.preprocessor_ = make_preprocessor(X)
        Z = self.preprocessor_.fit_transform(X)

        if self.use_anomaly_score:
            # Expected share of anomalies = share of fraud in the training data.
            contamination = float(np.clip(y.mean(), 1e-4, 0.5))
            self.iso_ = self._isolation_forest(contamination).fit(Z)

            oof_scores = np.empty(len(Z))
            folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            for train_idx, val_idx in folds.split(Z):
                fold_iso = self._isolation_forest(contamination).fit(Z[train_idx])
                oof_scores[val_idx] = fold_iso.decision_function(Z[val_idx])
            Z = np.column_stack([Z, oof_scores])

        self.clf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ).fit(Z, y)
        return self

    def anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Isolation Forest decision scores (lower = more anomalous)."""
        if not self.use_anomaly_score:
            raise RuntimeError("This model was trained without an anomaly score.")
        return self.iso_.decision_function(self.preprocessor_.transform(X))

    def anomaly_flags(self, X: pd.DataFrame) -> np.ndarray:
        """1 where the Isolation Forest itself flags an anomaly, else 0."""
        if not self.use_anomaly_score:
            raise RuntimeError("This model was trained without an anomaly score.")
        return (self.iso_.predict(self.preprocessor_.transform(X)) == -1).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Fraud probability for each row."""
        Z = self.preprocessor_.transform(X)
        if self.use_anomaly_score:
            Z = np.column_stack([Z, self.iso_.decision_function(Z)])
        fraud_col = int(np.where(self.clf_.classes_ == 1)[0][0])
        return self.clf_.predict_proba(Z)[:, fraud_col]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def load_or_train(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    retrain: bool = False,
    n_samples: int = 20000,
    seed: int = 42,
) -> HybridFraudModel:
    """Load a saved model, or train one on synthetic data and save it.

    Only load model files you created yourself: joblib files are pickles.
    """
    path = Path(model_path)
    if path.exists() and not retrain:
        try:
            model = joblib.load(path)
            if isinstance(model, HybridFraudModel):
                return model
            print(f"{path} does not contain a fraud model; retraining.")
        except Exception as exc:  # e.g. saved with an incompatible scikit-learn
            print(f"Could not load {path} ({exc}); retraining.")

    print("Training model on synthetic data...")
    X, y = generate_synthetic_data(n_samples, seed=seed)
    model = HybridFraudModel(seed=seed).fit(X, y)
    joblib.dump(model, path)
    return model


# --------------------------------------------------------------------------
# Scoring a single transaction
# --------------------------------------------------------------------------


def normalize_choice(value: str, options: list[str], label: str) -> str:
    """Match ``value`` to one of ``options`` ignoring case and surrounding spaces."""
    lookup = {option.lower(): option for option in options}
    key = str(value).strip().lower()
    if key not in lookup:
        raise ValueError(f"Invalid {label} {value!r}. Please choose from: {', '.join(options)}")
    return lookup[key]


def score_transaction(
    model: HybridFraudModel,
    amount: float,
    category: str,
    trans_type: str,
    is_weekend: int | bool,
    threshold: float = 0.5,
) -> dict:
    """Score one transaction. Raises ValueError on invalid input."""
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("Amount must be a non-negative number.")
    category = normalize_choice(category, VALID_CATEGORIES, "category")
    trans_type = normalize_choice(trans_type, VALID_TYPES, "transaction type")

    row = pd.DataFrame(
        {
            "Transaction_Amount": [float(amount)],
            "Merchant_Category": [category],
            "Transaction_Type": [trans_type],
            "Is_Weekend": [int(bool(is_weekend))],
        }
    )
    probability = float(model.predict_proba(row)[0])
    return {
        "amount": float(amount),
        "category": category,
        "type": trans_type,
        "weekend": bool(is_weekend),
        "probability": probability,
        "is_fraud": probability >= threshold,
        "threshold": threshold,
    }


def print_result(result: dict) -> None:
    print("\n" + "=" * 50)
    print("📊 ANALYSIS RESULTS")
    print("=" * 50)
    print(f"• Amount: ${result['amount']:.2f}")
    print(f"• Category: {result['category']}")
    print(f"• Type: {result['type']}")
    print(f"• Weekend: {'Yes' if result['weekend'] else 'No'}")
    print(f"• Calculated Fraud Probability: {result['probability']:.2%}")
    print(f"• Decision threshold: {result['threshold']:.0%}")
    print("-" * 50)
    if result["is_fraud"]:
        print("🚨 VERDICT: High Risk / Potential Fraud Detected!")
    else:
        print("✅ VERDICT: Normal Transaction")
    print("=" * 50 + "\n")


def _parse_amount(raw: str) -> float:
    try:
        amount = float(raw)
    except ValueError:
        raise ValueError("Invalid input. Please enter a valid number.") from None
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("Amount must be a non-negative number. Try again.")
    return amount


def _parse_weekend(raw: str) -> int:
    answer = raw.strip().lower()
    if answer in ("y", "yes"):
        return 1
    if answer in ("n", "no"):
        return 0
    raise ValueError("Please type 'y' for yes or 'n' for no.")


def _ask(prompt: str, parse):
    """Keep asking until ``parse`` accepts the answer."""
    while True:
        try:
            return parse(input(prompt).strip())
        except ValueError as exc:
            print(f"⚠️ {exc}")


def prompt_user_transaction(model: HybridFraudModel, threshold: float = 0.5) -> dict:
    """Guide the user step by step, validating every answer."""
    print("=" * 50)
    print("💳 CREDIT CARD FRAUD DETECTION - INTERACTIVE CHECKER")
    print("=" * 50)
    print("Please enter the transaction details below when prompted.\n")

    amount = _ask("1. Enter Transaction Amount (e.g., 45.50 or 1200.00): $", _parse_amount)

    print(f"\n2. Available Merchant Categories: {', '.join(VALID_CATEGORIES)}")
    category = _ask(
        "Select or type a category from the list above: ",
        lambda raw: normalize_choice(raw, VALID_CATEGORIES, "category"),
    )

    print(f"\n3. Available Transaction Types: {', '.join(VALID_TYPES)}")
    trans_type = _ask(
        "Select transaction type from the list above: ",
        lambda raw: normalize_choice(raw, VALID_TYPES, "transaction type"),
    )

    is_weekend = _ask("\n4. Is this transaction taking place on a weekend? (y/n): ", _parse_weekend)

    result = score_transaction(model, amount, category, trans_type, is_weekend, threshold)
    print_result(result)
    return result


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def _summarise(y_true: np.ndarray, score: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, score)
    return {
        "score": score,
        "y_pred": y_pred,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(auc(rec_curve, prec_curve)),
        "roc_auc": float(roc_auc_score(y_true, score)),
    }


def evaluate(
    X: pd.DataFrame,
    y,
    threshold: float = 0.5,
    seed: int = 42,
    n_estimators: int = 100,
    test_size: float = 0.2,
) -> dict:
    """Stratified train/test evaluation of three detectors.

    * Isolation Forest only (unsupervised, flags its own top anomalies)
    * Random Forest only (supervised baseline)
    * Hybrid: Random Forest that also receives the Isolation Forest score

    Everything is fit on the training split only.
    """
    y = np.asarray(y).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    hybrid = HybridFraudModel(True, n_estimators=n_estimators, seed=seed).fit(X_train, y_train)
    baseline = HybridFraudModel(False, n_estimators=n_estimators, seed=seed).fit(X_train, y_train)

    results = {
        ISOLATION_ONLY: _summarise(
            y_test, -hybrid.anomaly_scores(X_test), hybrid.anomaly_flags(X_test)
        ),
    }
    proba = baseline.predict_proba(X_test)
    results[FOREST_ONLY] = _summarise(y_test, proba, (proba >= threshold).astype(int))
    proba = hybrid.predict_proba(X_test)
    results[HYBRID] = _summarise(y_test, proba, (proba >= threshold).astype(int))

    return {"y_test": y_test, "results": results, "threshold": threshold}


def print_report(evaluation: dict) -> None:
    y_test = evaluation["y_test"]
    results = evaluation["results"]
    threshold = evaluation["threshold"]

    print(
        f"\nTest set: {len(y_test):,} transactions, "
        f"{int(y_test.sum()):,} fraud ({y_test.mean():.3%})\n"
    )
    print(f"{'Model':<26}{'Precision':>10}{'Recall':>9}{'F1':>8}{'PR-AUC':>9}{'ROC-AUC':>9}")
    print("-" * 71)
    for name, r in results.items():
        print(
            f"{name:<26}{r['precision']:>10.3f}{r['recall']:>9.3f}{r['f1']:>8.3f}"
            f"{r['pr_auc']:>9.3f}{r['roc_auc']:>9.3f}"
        )
    print(f"\nPrecision/recall/F1 use a {threshold:.2f} probability threshold, except for")
    print("Isolation Forest only, which uses its own contamination-based cut-off.")

    print(f"\nClassification report ({HYBRID}, threshold {threshold:.2f}):")
    print(
        classification_report(
            y_test,
            results[HYBRID]["y_pred"],
            target_names=["Normal", "Fraud"],
            digits=3,
            zero_division=0,
        )
    )


def save_plots(evaluation: dict, out_dir: str | Path = ".") -> list[Path]:
    """Save confusion matrices and precision-recall curves as PNG files."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_test = evaluation["y_test"]
    results = evaluation["results"]

    fig, axes = plt.subplots(1, len(results), figsize=(4.5 * len(results), 4))
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            r["y_pred"],
            display_labels=["Normal", "Fraud"],
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(name)
    fig.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, r in results.items():
        precision, recall, _ = precision_recall_curve(y_test, r["score"])
        ax.plot(recall, precision, label=f"{name} (PR-AUC {r['pr_auc']:.3f})")
    ax.axhline(y_test.mean(), linestyle="--", color="grey", label="No skill")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    pr_path = out_dir / "pr_curve.png"
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)

    return [cm_path, pr_path]


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _threshold(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraud_detection.py",
        description="Hybrid Isolation Forest + Random Forest credit card fraud detection.",
        epilog="Running with no command is the same as 'check' (interactive prompts).",
    )
    sub = parser.add_subparsers(dest="command")

    check = sub.add_parser("check", help="score one transaction (prompts if no flags given)")
    check.add_argument("--amount", type=float, help="transaction amount in dollars")
    check.add_argument("--category", help=f"one of: {', '.join(VALID_CATEGORIES)}")
    check.add_argument("--type", dest="trans_type", help=f"one of: {', '.join(VALID_TYPES)}")
    check.add_argument("--weekend", action="store_true", help="transaction happens on a weekend")
    check.add_argument("--threshold", type=_threshold, default=0.5, help="fraud cut-off (default 0.5)")
    check.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="where to save/load the model")
    check.add_argument("--retrain", action="store_true", help="ignore any saved model and retrain")

    ev = sub.add_parser("evaluate", help="train/test evaluation with metrics and plots")
    ev.add_argument("--n-samples", type=int, default=20000, help="synthetic sample count")
    ev.add_argument("--threshold", type=_threshold, default=0.5, help="fraud cut-off (default 0.5)")
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--plots", action="store_true", help="save confusion matrix and PR curve PNGs")
    ev.add_argument("--out-dir", default=".", help="where to save plots")
    return parser


def _run_check(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    given = [args.amount is not None, args.category is not None, args.trans_type is not None]
    if any(given) and not all(given):
        parser.error("--amount, --category and --type must be given together")

    model = load_or_train(args.model_path, retrain=args.retrain)
    if all(given):
        try:
            result = score_transaction(
                model, args.amount, args.category, args.trans_type, args.weekend, args.threshold
            )
        except ValueError as exc:
            parser.error(str(exc))
        print_result(result)
    else:
        prompt_user_transaction(model, args.threshold)
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    X, y = generate_synthetic_data(args.n_samples, seed=args.seed)
    print(f"Generated {len(X):,} synthetic transactions, {int(y.sum()):,} fraud.")

    print("Training models...")
    evaluation = evaluate(X, y, threshold=args.threshold, seed=args.seed)
    print_report(evaluation)

    if args.plots:
        for path in save_plots(evaluation, args.out_dir):
            print(f"Saved {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"check", "evaluate", "-h", "--help"}:
        argv.insert(0, "check")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        return _run_evaluate(args)
    return _run_check(args, parser)


if __name__ == "__main__":
    # Re-import under the real module name so a saved model is pickled as
    # "fraud_detection.HybridFraudModel" rather than "__main__.HybridFraudModel",
    # which could not be loaded from anywhere else.
    import fraud_detection

    raise SystemExit(fraud_detection.main())
