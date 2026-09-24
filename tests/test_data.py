import hashlib

import pandas as pd
import pytest

from fraud_detection.data import (
    build_tabular_features,
    download_paysim,
    temporal_split,
    validate_paysim,
)


def sample_paysim() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "type": ["PAYMENT", "TRANSFER"] * 5,
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "nameOrig": ["a", "a", "b", "b", "a", "c", "c", "a", "b", "c"],
            "oldbalanceOrg": [100.0] * 10,
            "newbalanceOrig": [90.0] * 10,
            "nameDest": ["x", "y"] * 5,
            "oldbalanceDest": [0.0] * 10,
            "newbalanceDest": [10.0] * 10,
            "isFraud": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            "isFlaggedFraud": [0] * 10,
        }
    )


def test_temporal_split_has_ordered_nonempty_partitions() -> None:
    split = temporal_split(sample_paysim())

    assert split.train["step"].max() < split.validation["step"].min()
    assert split.validation["step"].max() < split.test["step"].min()
    assert len(split.train) + len(split.validation) + len(split.test) == 10


def test_prior_features_do_not_include_current_transaction() -> None:
    features, _ = build_tabular_features(sample_paysim())

    assert features.loc[0, "origin_prior_count"] == 0
    assert features.loc[1, "origin_prior_count"] == 1
    assert features.loc[1, "origin_prior_amount"] == 10.0


def test_schema_validation_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        validate_paysim(pd.DataFrame({"step": [1], "isFraud": [0]}))


def test_download_paysim_verifies_checksum(monkeypatch, tmp_path) -> None:
    payload = (
        b"step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,"
        b"oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n"
        b"1,PAYMENT,10,a,100,90,x,0,10,0,0\n"
    )
    checksum = hashlib.sha256(payload).hexdigest()

    def fake_urlretrieve(url: str, destination: str) -> tuple[str, None]:
        assert url == "https://example.com/paysim.csv"
        with open(destination, "wb") as handle:
            handle.write(payload)
        return destination, None

    monkeypatch.setattr("fraud_detection.data.urlretrieve", fake_urlretrieve)

    path = tmp_path / "paysim.csv"
    downloaded = download_paysim("https://example.com/paysim.csv", path, expected_sha256=checksum)

    assert downloaded == path
    assert path.exists()
    assert path.read_bytes() == payload
