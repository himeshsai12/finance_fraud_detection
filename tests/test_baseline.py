import warnings

import pandas as pd

from fraud_detection.baseline import evaluate_predictions


def test_evaluation_reports_imbalanced_metrics() -> None:
    labels = pd.Series([0, 0, 0, 1, 1])
    probabilities = pd.Series([0.05, 0.20, 0.30, 0.80, 0.95])

    metrics = evaluate_predictions(labels, probabilities, fixed_precision=0.80)

    assert 0.0 <= metrics.pr_auc <= 1.0
    assert 0.0 <= metrics.fraud_f1 <= 1.0
    assert 0.0 <= metrics.recall_at_fixed_precision <= 1.0
    assert metrics.threshold_at_fixed_precision >= 0.0


def test_evaluation_handles_all_negative_labels_without_warning() -> None:
    labels = pd.Series([0, 0, 0, 0])
    probabilities = pd.Series([0.10, 0.25, 0.40, 0.60])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = evaluate_predictions(labels, probabilities, fixed_precision=0.80)

    assert len(caught) == 0
    assert metrics.pr_auc == 0.0
    assert metrics.fraud_f1 == 0.0
    assert metrics.recall_at_fixed_precision == 0.0
    assert metrics.threshold_at_fixed_precision == 1.0
