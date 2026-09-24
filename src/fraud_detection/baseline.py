"""Tabular XGBoost baseline and fraud-focused evaluation metrics."""

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from xgboost import XGBClassifier

from fraud_detection.data import TemporalSplit, build_tabular_features


@dataclass(frozen=True)
class Evaluation:
    """Metrics reported for an imbalanced fraud classifier."""

    pr_auc: float
    fraud_f1: float
    recall_at_fixed_precision: float
    threshold_at_fixed_precision: float

    def as_dict(self) -> dict[str, float]:
        return {
            "pr_auc": self.pr_auc,
            "fraud_f1": self.fraud_f1,
            "recall_at_fixed_precision": self.recall_at_fixed_precision,
            "threshold_at_fixed_precision": self.threshold_at_fixed_precision,
        }


def make_xgboost(scale_pos_weight: float) -> XGBClassifier:
    """Create a reproducible baseline with imbalance-aware weighting."""

    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
        n_jobs=4,
    )


def evaluate_predictions(
    labels: pd.Series,
    probabilities: pd.Series,
    fixed_precision: float = 0.80,
) -> Evaluation:
    """Calculate PR-AUC, fraud F1, and recall at a precision constraint."""

    if not 0 < fixed_precision <= 1:
        raise ValueError("fixed_precision must be in the interval (0, 1]")
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")

    if labels.empty:
        return Evaluation(
            pr_auc=0.0,
            fraud_f1=0.0,
            recall_at_fixed_precision=0.0,
            threshold_at_fixed_precision=1.0,
        )

    if labels.sum() == 0:
        return Evaluation(
            pr_auc=0.0,
            fraud_f1=0.0,
            recall_at_fixed_precision=0.0,
            threshold_at_fixed_precision=1.0,
        )

    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    eligible = precision[:-1] >= fixed_precision
    if eligible.any():
        eligible_recall = recall[:-1][eligible]
        best_index = eligible_recall.argmax()
        threshold = float(thresholds[eligible][best_index])
        constrained_recall = float(eligible_recall[best_index])
    else:
        threshold = 1.0
        constrained_recall = 0.0
    predictions = (probabilities >= threshold).astype("int64")
    return Evaluation(
        pr_auc=float(average_precision_score(labels, probabilities)),
        fraud_f1=float(f1_score(labels, predictions, zero_division=0)),
        recall_at_fixed_precision=constrained_recall,
        threshold_at_fixed_precision=threshold,
    )


def train_baseline(split: TemporalSplit) -> tuple[XGBClassifier, dict[str, Evaluation]]:
    """Train on the temporal training partition and evaluate future partitions."""

    combined = pd.concat([split.train, split.validation, split.test], ignore_index=True)
    all_features, all_labels = build_tabular_features(combined)
    train_end = len(split.train)
    validation_end = train_end + len(split.validation)
    train_features, train_labels = all_features.iloc[:train_end], all_labels.iloc[:train_end]
    validation_features = all_features.iloc[train_end:validation_end]
    validation_labels = all_labels.iloc[train_end:validation_end]
    test_features = all_features.iloc[validation_end:]
    test_labels = all_labels.iloc[validation_end:]
    validation_features = validation_features.reindex(
        columns=train_features.columns, fill_value=0.0
    )
    test_features = test_features.reindex(columns=train_features.columns, fill_value=0.0)

    positive_count = max(int(train_labels.sum()), 1)
    negative_count = max(len(train_labels) - positive_count, 1)
    model = make_xgboost(negative_count / positive_count)
    model.fit(train_features, train_labels)
    metrics = {
        "validation": evaluate_predictions(
            validation_labels, pd.Series(model.predict_proba(validation_features)[:, 1])
        ),
        "test": evaluate_predictions(
            test_labels, pd.Series(model.predict_proba(test_features)[:, 1])
        ),
    }
    return model, metrics
