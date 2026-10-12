import joblib
import numpy as np
import pandas as pd
import pytest

import fraud_detection as fd


@pytest.fixture(scope="module")
def data():
    return fd.generate_synthetic_data(4000, seed=0)


@pytest.fixture(scope="module")
def model(data):
    X, y = data
    return fd.HybridFraudModel(n_estimators=50, seed=0).fit(X, y)


def test_synthetic_data_has_expected_shape_and_fraud_rate(data):
    X, y = data
    assert len(X) == len(y) == 4000
    assert set(X.columns) == {
        "Transaction_Amount",
        "Merchant_Category",
        "Transaction_Type",
        "Is_Weekend",
    }
    assert 0.015 < y.mean() < 0.05


def test_fit_rejects_single_class_labels(data):
    X, _ = data
    with pytest.raises(ValueError, match="only one class"):
        fd.HybridFraudModel().fit(X, np.zeros(len(X), dtype=int))


def test_probabilities_are_valid(model, data):
    X, _ = data
    proba = model.predict_proba(X.head(50))
    assert proba.shape == (50,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_risky_transaction_scores_higher_than_safe_one(model):
    risky = fd.score_transaction(model, 600, "Electronics", "Online", True)
    safe = fd.score_transaction(model, 12, "Grocery", "In-Store", False)
    assert risky["probability"] > safe["probability"]
    assert risky["probability"] > 0.5 > safe["probability"]


def test_threshold_controls_verdict(model):
    assert fd.score_transaction(model, 12, "Grocery", "In-Store", 0, threshold=0.0)["is_fraud"]
    assert not fd.score_transaction(model, 12, "Grocery", "In-Store", 0, threshold=0.99)["is_fraud"]


def test_input_is_validated_and_case_insensitive(model):
    result = fd.score_transaction(model, 50, "travel", " ONLINE ", 0)
    assert (result["category"], result["type"]) == ("Travel", "Online")
    with pytest.raises(ValueError):
        fd.score_transaction(model, 50, "Furniture", "Online", 0)
    with pytest.raises(ValueError):
        fd.score_transaction(model, -5, "Travel", "Online", 0)
    with pytest.raises(ValueError):
        fd.score_transaction(model, float("nan"), "Travel", "Online", 0)


def test_save_and_load_roundtrip(model, data, tmp_path):
    X, _ = data
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(loaded.predict_proba(X.head(20)), model.predict_proba(X.head(20)))


def test_load_or_train_saves_then_reuses(tmp_path):
    path = tmp_path / "model.joblib"
    first = fd.load_or_train(path, n_samples=2000)
    assert path.exists()
    mtime = path.stat().st_mtime_ns
    second = fd.load_or_train(path, n_samples=2000)
    assert path.stat().st_mtime_ns == mtime
    assert isinstance(first, fd.HybridFraudModel) and isinstance(second, fd.HybridFraudModel)


def test_load_or_train_recovers_from_corrupt_file(tmp_path):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"not a model")
    assert isinstance(fd.load_or_train(path, n_samples=2000), fd.HybridFraudModel)


def test_evaluate_reports_sensible_metrics():
    X, y = fd.generate_synthetic_data(6000, seed=3)
    evaluation = fd.evaluate(X, y, n_estimators=50, seed=3)
    results = evaluation["results"]
    assert list(results) == [fd.ISOLATION_ONLY, fd.FOREST_ONLY, fd.HYBRID]
    for r in results.values():
        for key in ("precision", "recall", "f1", "pr_auc", "roc_auc"):
            assert 0.0 <= r[key] <= 1.0
    # A trained model must beat the "no skill" PR-AUC (= fraud rate).
    assert results[fd.HYBRID]["pr_auc"] > evaluation["y_test"].mean() * 3


def test_save_plots_writes_png_files(tmp_path):
    pytest.importorskip("matplotlib")
    X, y = fd.generate_synthetic_data(3000, seed=4)
    evaluation = fd.evaluate(X, y, n_estimators=30, seed=4)
    paths = fd.save_plots(evaluation, tmp_path)
    assert [p.name for p in paths] == ["confusion_matrix.png", "pr_curve.png"]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_interactive_prompt_reprompts_on_bad_input(model, monkeypatch, capsys):
    answers = iter(["abc", "-5", "950", "furniture", "electronics", "carrier pigeon", "online", "maybe", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    result = fd.prompt_user_transaction(model)
    out = capsys.readouterr().out
    assert out.count("⚠️") == 5
    assert "VERDICT" in out
    assert (result["amount"], result["category"], result["type"], result["weekend"]) == (
        950.0,
        "Electronics",
        "Online",
        True,
    )


def test_cli_one_shot_check(tmp_path, capsys):
    code = fd.main(
        [
            "--amount", "900",
            "--category", "electronics",
            "--type", "online",
            "--weekend",
            "--model-path", str(tmp_path / "m.joblib"),
        ]
    )
    assert code == 0
    assert "Calculated Fraud Probability" in capsys.readouterr().out


def test_cli_rejects_partial_flags(tmp_path):
    with pytest.raises(SystemExit):
        fd.main(["check", "--amount", "10", "--model-path", str(tmp_path / "m.joblib")])


def test_cli_evaluate_runs_on_synthetic_data(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    code = fd.main(["evaluate", "--n-samples", "3000", "--plots", "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert fd.HYBRID in out and "Classification report" in out
    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "pr_curve.png").exists()
