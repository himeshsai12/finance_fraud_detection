"""Command-line entry points for training and local project configuration."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from fraud_detection.baseline import train_baseline
from fraud_detection.config import get_settings
from fraud_detection.data import download_paysim, load_paysim, temporal_split
from fraud_detection.gnn import train_graph_model
from fraud_detection.graph import build_temporal_graph, summarize_temporal_graph


def _print_config() -> None:
    """Print the active configuration used by local development."""

    settings = get_settings()
    print(f"{settings.project_name} {settings.random_seed=}")
    print(f"data_dir={settings.data_dir}")
    print(f"kafka={settings.kafka_bootstrap_servers}")


def _as_serializable_metrics(metrics: dict[str, object]) -> dict[str, dict[str, float]]:
    """Convert evaluation dataclasses to JSON-friendly dictionaries."""

    serializable: dict[str, dict[str, float]] = {}
    for name, metric in metrics.items():
        serializable[name] = metric.as_dict() if hasattr(metric, "as_dict") else dict(metric)
    return serializable


def _handle_baseline(args: argparse.Namespace) -> int:
    """Load a PaySim CSV, train the temporal baseline, and persist the metrics."""

    settings = get_settings()
    data_path = Path(args.data_path)
    download_url = args.download_url or settings.paysim_url
    expected_sha256 = args.expected_sha256 or settings.paysim_sha256

    if not data_path.exists() and download_url:
        if expected_sha256 is None:
            raise ValueError("An expected SHA-256 hash is required when downloading PaySim data")
        data_path = download_paysim(
            download_url,
            data_path,
            expected_sha256=expected_sha256,
        )
    elif not data_path.exists():
        raise FileNotFoundError(
            "PaySim data not found at "
            f"{data_path}. Provide --download-url and --expected-sha256 or "
            "configure FRAUD_PAYSIM_URL and FRAUD_PAYSIM_SHA256."
        )

    frame = load_paysim(data_path)
    split = temporal_split(frame)
    _, metrics = train_baseline(split)
    payload = _as_serializable_metrics(metrics)

    output_path = (
        Path(args.output)
        if args.output
        else settings.artifacts_dir / "baseline_metrics.json"
    )
    output_path = output_path if output_path.is_absolute() else Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _handle_download(args: argparse.Namespace) -> int:
    """Download a PaySim CSV to a target path and validate its checksum."""

    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    download_paysim(
        args.url,
        destination,
        expected_sha256=args.sha256,
    )
    print(f"Downloaded PaySim data to {destination}")
    return 0


def _handle_graph(args: argparse.Namespace) -> int:
    """Build a temporal customer-merchant graph summary from a PaySim CSV."""

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"PaySim data not found at {data_path}.")

    frame = load_paysim(data_path)
    neighbor_cap = (
        args.neighbor_cap if args.neighbor_cap is not None else get_settings().neighbor_cap
    )
    graph = build_temporal_graph(frame, neighbor_cap=neighbor_cap)
    payload = summarize_temporal_graph(graph)

    output_path = Path(args.output) if args.output else Path.cwd() / "graph_summary.json"
    output_path = output_path if output_path.is_absolute() else Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _handle_gnn(args: argparse.Namespace) -> int:
    """Train a lightweight GraphSAGE model on the temporal PaySim split."""

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"PaySim data not found at {data_path}.")

    frame = load_paysim(data_path)
    split = temporal_split(frame)
    _, metrics = train_graph_model(
        split,
        neighbor_cap=(
            args.neighbor_cap if args.neighbor_cap is not None else get_settings().neighbor_cap
        ),
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
    )
    payload = _as_serializable_metrics(metrics)

    output_path = Path(args.output) if args.output else Path.cwd() / "gnn_metrics.json"
    output_path = output_path if output_path.is_absolute() else Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run a local CLI command for config inspection or a baseline training job."""

    parser = argparse.ArgumentParser(description="Real-time fraud detection CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("config", help="Show active runtime settings")

    download_parser = subparsers.add_parser(
        "download",
        help="Download the PaySim CSV with a checksum verification",
    )
    download_parser.add_argument("--url", required=True, help="URL for the PaySim CSV")
    download_parser.add_argument("--sha256", required=True, help="Expected SHA-256 checksum")
    download_parser.add_argument(
        "--destination",
        required=True,
        help="Location to save the downloaded PaySim CSV",
    )

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Train the temporal XGBoost baseline on a PaySim CSV and save metrics",
    )
    baseline_parser.add_argument("--data-path", required=True, help="Path to a PaySim CSV file")
    baseline_parser.add_argument(
        "--download-url",
        default=None,
        help="Optional URL to download the PaySim CSV if the data path does not yet exist",
    )
    baseline_parser.add_argument(
        "--expected-sha256",
        default=None,
        help="Expected SHA-256 digest used to verify a downloaded PaySim CSV",
    )
    baseline_parser.add_argument(
        "--output",
        default=None,
        help="Where to save the baseline metrics JSON; defaults to artifacts/baseline_metrics.json",
    )

    graph_parser = subparsers.add_parser(
        "graph",
        help="Build a temporal customer-merchant graph summary and save it as JSON",
    )
    graph_parser.add_argument("--data-path", required=True, help="Path to a PaySim CSV file")
    graph_parser.add_argument(
        "--neighbor-cap",
        type=int,
        default=None,
        help="Maximum recent neighbors retained for each account when building the graph",
    )
    graph_parser.add_argument(
        "--output",
        default=None,
        help=(
            "Where to save the graph summary JSON; defaults to "
            "graph_summary.json in the working directory"
        ),
    )

    gnn_parser = subparsers.add_parser(
        "gnn",
        help="Train a lightweight GraphSAGE-style model on the temporal PaySim split",
    )
    gnn_parser.add_argument("--data-path", required=True, help="Path to a PaySim CSV file")
    gnn_parser.add_argument(
        "--neighbor-cap",
        type=int,
        default=None,
        help="Maximum recent neighbors retained for each account when building the graph",
    )
    gnn_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of optimization epochs for the GraphSAGE-style model",
    )
    gnn_parser.add_argument(
        "--hidden-dim",
        type=int,
        default=16,
        help="Hidden width for the GraphSAGE-style model",
    )
    gnn_parser.add_argument(
        "--output",
        default=None,
        help=(
            "Where to save the GNN metrics JSON; defaults to "
            "gnn_metrics.json in the working directory"
        ),
    )

    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        _print_config()
        return 0

    args = parser.parse_args(args_list)
    if args.command == "config":
        _print_config()
        return 0
    if args.command == "download":
        return _handle_download(args)
    if args.command == "baseline":
        return _handle_baseline(args)
    if args.command == "graph":
        return _handle_graph(args)
    if args.command == "gnn":
        return _handle_gnn(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
