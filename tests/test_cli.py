import json

import pandas as pd

from fraud_detection.cli import main


def test_cli_baseline_writes_metrics_file(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "step": list(range(1, 11)),
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
    data_path = tmp_path / "paysim.csv"
    output_path = tmp_path / "metrics.json"
    frame.to_csv(data_path, index=False)

    exit_code = main(["baseline", "--data-path", str(data_path), "--output", str(output_path)])

    assert exit_code == 0
    payload = json.loads(output_path.read_text())
    assert set(payload["validation"]).issuperset({"pr_auc", "fraud_f1"})
    assert set(payload["test"]).issuperset({"pr_auc", "fraud_f1"})


def test_cli_download_command_fetches_paysim(monkeypatch, tmp_path) -> None:
    payload = (
        b"step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,"
        b"oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n"
        b"1,PAYMENT,10,a,100,90,x,0,10,0,0\n"
    )
    sha256 = __import__("hashlib").sha256(payload).hexdigest()

    def fake_urlretrieve(url: str, destination: str) -> tuple[str, None]:
        with open(destination, "wb") as handle:
            handle.write(payload)
        return destination, None

    monkeypatch.setattr("fraud_detection.data.urlretrieve", fake_urlretrieve)

    destination = tmp_path / "raw" / "paysim.csv"
    exit_code = main([
        "download",
        "--url",
        "https://example.com/paysim.csv",
        "--sha256",
        sha256,
        "--destination",
        str(destination),
    ])

    assert exit_code == 0
    assert destination.exists()
    assert destination.read_bytes() == payload
