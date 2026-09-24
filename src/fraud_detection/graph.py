"""Temporal customer-merchant graph construction and summaries."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from fraud_detection.data import validate_paysim


@dataclass(frozen=True)
class TemporalGraph:
    """A lightweight graph structure for leakage-safe temporal modeling."""

    node_features: pd.DataFrame
    edge_index: list[tuple[int, int]]
    edge_features: pd.DataFrame
    node_order: list[str]


def build_temporal_graph(frame: pd.DataFrame, neighbor_cap: int = 50) -> TemporalGraph:
    """Build a temporal customer-merchant graph from a PaySim transaction stream.

    Nodes are the union of origin and destination entities. Each row creates a directed
    edge from the origin account to the destination account. The graph keeps all
    transactions in chronological order and enforces a per-node cap on recent neighbors
    when a downstream model consumes the adjacency list.
    """

    validate_paysim(frame)
    ordered = frame.sort_values("step", kind="mergesort").reset_index(drop=True).copy()
    if neighbor_cap < 1:
        raise ValueError("neighbor_cap must be at least 1")

    node_names = sorted(set(ordered["nameOrig"]).union(ordered["nameDest"]))
    node_lookup = {name: index for index, name in enumerate(node_names)}

    recent_neighbors: dict[str, list[tuple[int, int]]] = defaultdict(list)
    edge_rows: list[dict[str, object]] = []

    for _row_index, row in ordered.iterrows():
        src = str(row["nameOrig"])
        dst = str(row["nameDest"])
        source_index = node_lookup[src]
        target_index = node_lookup[dst]

        for node_name in (src, dst):
            recent = recent_neighbors[node_name]
            if len(recent) >= neighbor_cap:
                recent.pop(0)
            recent.append((source_index, target_index))

        edge_rows.append(
            {
                "source": source_index,
                "target": target_index,
                "step": float(row["step"]),
                "amount": float(row["amount"]),
                "is_fraud": int(row["isFraud"]),
                "transaction_type": str(row["type"]),
            }
        )

    node_rows: list[dict[str, object]] = []
    for node_name in node_names:
        node_index = node_lookup[node_name]
        node_transactions = ordered[
            (ordered["nameOrig"] == node_name) | (ordered["nameDest"] == node_name)
        ]
        node_type = "mixed"
        if node_name in set(ordered["nameOrig"]) and node_name in set(ordered["nameDest"]):
            node_type = "mixed"
        elif node_name in set(ordered["nameOrig"]):
            node_type = "origin"
        elif node_name in set(ordered["nameDest"]):
            node_type = "destination"

        node_rows.append(
            {
                "node_id": node_index,
                "node_name": node_name,
                "node_type": node_type,
                "total_amount": float(node_transactions["amount"].sum()),
                "event_count": int(len(node_transactions)),
                "mean_amount": float(node_transactions["amount"].mean())
                if not node_transactions.empty
                else 0.0,
                "fraud_count": int(node_transactions["isFraud"].sum()),
            }
        )

    node_features = pd.DataFrame(node_rows).sort_values("node_id").reset_index(drop=True)
    edge_features = pd.DataFrame(edge_rows)
    edge_index = [(int(row["source"]), int(row["target"])) for _, row in edge_features.iterrows()]

    return TemporalGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        node_order=node_names,
    )


def summarize_temporal_graph(graph: TemporalGraph) -> dict[str, float | int]:
    """Compute a compact summary for a temporal graph for CLI output or metrics."""

    if graph.node_features.empty:
        return {"n_nodes": 0, "n_edges": 0, "n_origins": 0, "n_destinations": 0, "avg_degree": 0.0}

    origin_count = int((graph.node_features["node_type"] == "origin").sum())
    destination_count = int((graph.node_features["node_type"] == "destination").sum())
    mixed_count = int((graph.node_features["node_type"] == "mixed").sum())
    n_edges = len(graph.edge_index)
    avg_degree = (2 * n_edges) / len(graph.node_features) if len(graph.node_features) else 0.0

    return {
        "n_nodes": int(len(graph.node_features)),
        "n_edges": n_edges,
        "n_origins": origin_count + mixed_count,
        "n_destinations": destination_count + mixed_count,
        "avg_degree": float(avg_degree),
    }
