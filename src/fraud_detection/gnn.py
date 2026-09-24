"""Lightweight GraphSAGE-style model for temporal transaction graphs."""

from __future__ import annotations

import pandas as pd
import torch
from torch import nn

from fraud_detection.baseline import evaluate_predictions
from fraud_detection.data import TemporalSplit
from fraud_detection.graph import build_temporal_graph


class GraphSAGEModel(nn.Module):
    """A compact graph model that aggregates node neighborhoods before classification."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute node probabilities using a simple neighborhood aggregation step."""

        if x.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=x.device)

        num_nodes = x.size(0)
        if edge_index.numel() == 0:
            aggregated = x
        else:
            adjacency = torch.zeros(
                (num_nodes, num_nodes),
                dtype=x.dtype,
                device=x.device,
            )
            adjacency.index_put_(
                (edge_index[0], edge_index[1]),
                torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype),
                accumulate=True,
            )
            adjacency = adjacency + torch.eye(num_nodes, device=x.device, dtype=x.dtype)
            degrees = adjacency.sum(dim=1, keepdim=True).clamp_min(1)
            aggregated = adjacency / degrees @ x

        hidden = torch.relu(self.input_layer(aggregated))
        hidden = torch.relu(self.hidden_layer(hidden))
        logits = self.output_layer(hidden).squeeze(-1)
        return torch.sigmoid(logits)


def build_graph_dataset(
    frame: pd.DataFrame,
    neighbor_cap: int = 50,
) -> dict[str, torch.Tensor | list[str]]:
    """Convert a temporal graph into node features, edge indices, and labels."""

    graph = build_temporal_graph(frame, neighbor_cap=neighbor_cap)
    feature_columns = [
        column
        for column in ("total_amount", "event_count", "mean_amount", "fraud_count")
        if column in graph.node_features.columns
    ]
    if not feature_columns:
        feature_columns = list(graph.node_features.select_dtypes(include="number").columns)
    node_matrix = graph.node_features[feature_columns].fillna(0.0).to_numpy(dtype=float)
    labels = graph.node_features["fraud_count"].astype(float).to_numpy() > 0

    return {
        "x": torch.tensor(node_matrix, dtype=torch.float32),
        "edge_index": torch.tensor(graph.edge_index, dtype=torch.long).T,
        "y": torch.tensor(labels.astype(float), dtype=torch.float32),
        "node_names": graph.node_order,
    }


def train_graph_model(
    split: TemporalSplit,
    neighbor_cap: int = 50,
    epochs: int = 10,
    hidden_dim: int = 16,
) -> tuple[GraphSAGEModel, dict[str, object]]:
    """Train a compact GraphSAGE-style model on each temporal partition and score it."""

    train_dataset = build_graph_dataset(split.train, neighbor_cap=neighbor_cap)
    validation_dataset = build_graph_dataset(split.validation, neighbor_cap=neighbor_cap)
    test_dataset = build_graph_dataset(split.test, neighbor_cap=neighbor_cap)

    model = GraphSAGEModel(
        input_dim=int(train_dataset["x"].shape[1]),
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.BCELoss()

    for _ in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()
        probabilities = model(train_dataset["x"], train_dataset["edge_index"])
        loss = criterion(probabilities, train_dataset["y"])
        loss.backward()
        optimizer.step()

    metrics: dict[str, object] = {}
    for name, dataset in {
        "validation": validation_dataset,
        "test": test_dataset,
    }.items():
        model.eval()
        with torch.no_grad():
            probabilities = model(dataset["x"], dataset["edge_index"]).cpu().numpy()
        labels = dataset["y"].cpu().numpy()
        metrics[name] = evaluate_predictions(pd.Series(labels), pd.Series(probabilities))

    return model, metrics
