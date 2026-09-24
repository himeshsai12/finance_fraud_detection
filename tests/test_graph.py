import json

import pandas as pd

from fraud_detection.cli import main
from fraud_detection.graph import TemporalGraph, build_temporal_graph


def sample_paysim() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [1, 2, 3],
            "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
            "amount": [10.0, 20.0, 30.0],
            "nameOrig": ["a", "b", "a"],
            "oldbalanceOrg": [100.0, 200.0, 300.0],
            "newbalanceOrig": [90.0, 180.0, 270.0],
            "nameDest": ["x", "y", "x"],
            "oldbalanceDest": [0.0, 10.0, 20.0],
            "newbalanceDest": [10.0, 25.0, 50.0],
            "isFraud": [0, 1, 0],
            "isFlaggedFraud": [0, 0, 0],
        }
    )


def test_build_temporal_graph_keeps_temporal_edges() -> None:
    graph = build_temporal_graph(sample_paysim(), neighbor_cap=2)

    assert isinstance(graph, TemporalGraph)
    assert len(graph.edge_index) == 3
    assert len(graph.node_features) >= 4
    assert {"node_id", "node_type", "total_amount", "event_count"}.issubset(
        graph.node_features.columns
    )
    assert graph.node_features["event_count"].ge(1).all()


def test_cli_graph_command_writes_summary(tmp_path) -> None:
    data_path = tmp_path / "paysim.csv"
    output_path = tmp_path / "graph_summary.json"
    sample_paysim().to_csv(data_path, index=False)

    exit_code = main(["graph", "--data-path", str(data_path), "--output", str(output_path)])

    assert exit_code == 0
    payload = json.loads(output_path.read_text())
    assert payload["n_edges"] == 3
    assert payload["n_nodes"] >= 4
    assert payload["avg_degree"] > 0.0
