"""Read-only MLflow experiment history for the Streamlit research dashboard."""
from __future__ import annotations

from pathlib import Path


def local_tracking_uri(database: str | Path = "mlflow.db") -> str:
    """Resolve an existing SQLite database below the current working directory."""
    root = Path.cwd().resolve()
    selected = Path(database).resolve()
    if not selected.is_relative_to(root):
        raise ValueError("Tracking database must be inside the project directory")
    if not selected.is_file() or selected.suffix.lower() != ".db":
        raise FileNotFoundError("Select an existing local .db MLflow tracking database")
    return "sqlite:///" + str(selected)


def read_experiment_history(database: str | Path, limit: int = 100):
    """Return real run metadata and per-step loss curves, without artifact downloads."""
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise RuntimeError('Install the optional tracking extra: pip install -e ".[tracking]"') from exc
    client = MlflowClient(tracking_uri=local_tracking_uri(database))
    experiments = client.search_experiments()
    ids = [exp.experiment_id for exp in experiments
           if exp.lifecycle_stage == "active"]
    if not ids:
        return [], {}
    runs = client.search_runs(ids, max_results=limit, order_by=["attributes.start_time DESC"])
    summaries = []
    curves = {}
    for run in runs:
        info = run.info
        summaries.append({
            "run_id": info.run_id,
            "experiment_id": info.experiment_id,
            "name": run.data.tags.get("mlflow.runName", ""),
            "status": info.status,
            "start_time_ms": info.start_time,
            **{f"param_{key}": value for key, value in run.data.params.items()},
            **{f"metric_{key}": value for key, value in run.data.metrics.items()},
        })
        history = []
        for key in ("train_mae", "val_mae", "lr"):
            for point in client.get_metric_history(info.run_id, key):
                history.append({"step": point.step, "metric": key, "value": point.value})
        curves[info.run_id] = history
    return summaries, curves
