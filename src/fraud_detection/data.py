"""PaySim loading, validation, temporal splitting, leakage-safe features, and downloads."""

import hashlib
from dataclasses import dataclass
from math import log1p
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

PAYSIM_COLUMNS = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
)


@dataclass(frozen=True)
class TemporalSplit:
    """A chronological train/validation/test split."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def download_paysim(
    url: str,
    destination: str | Path,
    expected_sha256: str | None = None,
) -> Path:
    """Download a PaySim CSV and confirm the file checksum when provided."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, str(target))

    if expected_sha256 is not None:
        actual_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256.lower():
            raise ValueError(
                f"Downloaded PaySim checksum mismatch: expected {expected_sha256.lower()}, "
                f"got {actual_sha256}"
            )

    validate_paysim(pd.read_csv(target))
    return target


def load_paysim(path: Path) -> pd.DataFrame:
    """Load and validate a PaySim CSV without changing row order."""

    frame = pd.read_csv(path)
    validate_paysim(frame)
    return frame


def validate_paysim(frame: pd.DataFrame) -> None:
    """Raise a useful error when the expected PaySim schema is absent."""

    missing = sorted(set(PAYSIM_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"PaySim data is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("PaySim data must contain at least one row")
    if frame["step"].isna().any() or frame["isFraud"].isna().any():
        raise ValueError("PaySim step and isFraud cannot contain null values")
    if not set(frame["isFraud"].unique()).issubset({0, 1}):
        raise ValueError("PaySim isFraud must contain only 0 and 1")


def temporal_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TemporalSplit:
    """Split rows by time, preserving ties at the same timestamp boundary."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave test data")

    ordered = frame.sort_values("step", kind="mergesort").reset_index(drop=True)
    train_cut = ordered["step"].quantile(train_fraction, interpolation="linear")
    validation_cut = ordered["step"].quantile(
        train_fraction + validation_fraction, interpolation="linear"
    )
    train = ordered[ordered["step"] <= train_cut]
    validation = ordered[(ordered["step"] > train_cut) & (ordered["step"] <= validation_cut)]
    test = ordered[ordered["step"] > validation_cut]

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Temporal split produced an empty partition")
    if not (train["step"].max() <= validation["step"].min() <= test["step"].min()):
        raise AssertionError("Temporal split is not ordered")
    return TemporalSplit(train=train, validation=validation, test=test)


def build_tabular_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build features using only current and prior events in chronological order.

    Historical entity features are calculated after sorting by time. The current
    transaction is excluded from its own history through cumulative subtraction.
    """

    validate_paysim(frame)
    ordered = frame.sort_values("step", kind="mergesort").reset_index(drop=True).copy()
    features = pd.DataFrame(index=ordered.index)
    features["step"] = ordered["step"].astype("float64")
    features["event_hour"] = (ordered["step"] % 24).astype("float64")
    features["event_day"] = (ordered["step"] // 24).astype("float64")
    features["amount"] = ordered["amount"].astype("float64")
    features["log_amount"] = features["amount"].clip(lower=0).fillna(0).map(log1p)
    features["origin_balance_delta"] = ordered["oldbalanceOrg"] - ordered["newbalanceOrig"]
    features["destination_balance_delta"] = ordered["newbalanceDest"] - ordered["oldbalanceDest"]
    features["balance_error"] = features["origin_balance_delta"] - ordered["amount"]

    for entity_column, prefix in (("nameOrig", "origin"), ("nameDest", "destination")):
        grouped_amount = ordered.groupby(entity_column, sort=False)["amount"]
        features[f"{prefix}_prior_count"] = grouped_amount.cumcount().astype("float64")
        features[f"{prefix}_prior_amount"] = (
            grouped_amount.cumsum() - ordered["amount"]
        ).astype("float64")

    type_features = pd.get_dummies(ordered["type"], prefix="type", dtype="float64")
    features = pd.concat([features, type_features], axis=1).fillna(0.0)
    target = ordered["isFraud"].astype("int64")
    return features, target
