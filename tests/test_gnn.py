import json

import pandas as pd

from fraud_detection.cli import main
from fraud_detection.data import temporal_split
from fraud_detection.gnn import build_graph_dataset, train_graph_model


def sample_paysim() -> pd.DataFrame:
    return pd.DataFrame(
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


def test_build_graph_dataset_creates_node_and_edge_tensors() -> None:
    dataset = build_graph_dataset(sample_paysim(), neighbor_cap=2)

    assert dataset["x"].shape[0] >= 3
    assert dataset["edge_index"].shape[1] > 0
    assert dataset["y"].shape[0] == dataset["x"].shape[0]


def test_train_graph_model_returns_partition_metrics() -> None:
    split = temporal_split(sample_paysim())

    model, metrics = train_graph_model(split, epochs=2, hidden_dim=8)

    assert model is not None
    assert set(metrics) >= {"validation", "test"}
    assert all(0.0 <= metrics[name].pr_auc <= 1.0 for name in metrics)


def test_cli_gnn_command_writes_metrics(tmp_path) -> None:
    data_path = tmp_path / "paysim.csv"
    output_path = tmp_path / "gnn_metrics.json"
    sample_paysim().to_csv(data_path, index=False)

    exit_code = main(["gnn", "--data-path", str(data_path), "--output", str(output_path)])

    assert exit_code == 0
    payload = json.loads(output_path.read_text())
    assert set(payload["validation"]).issuperset({"pr_auc", "fraud_f1"})
    assert set(payload["test"]).issuperset({"pr_auc", "fraud_f1"})
