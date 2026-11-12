"""Opt-in MLflow logging for research-only synthetic experiments.

No patient data, checkpoints or secrets are uploaded automatically. MLflow is
imported lazily, so the core platform runs without the tracking extra.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path


def _mlflow(uri: str):
    if not uri:
        raise ValueError("tracking URI must be nonempty")
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError('Install tracking with pip install -e ".[tracking]"') from exc
    mlflow.set_tracking_uri(uri)
    return mlflow


def _params(config):
    values = asdict(config) if is_dataclass(config) else dict(config)
    return {key: str(value) for key, value in values.items()
            if key not in {"resume", "checkpoint", "output", "mlflow_uri"}
            and value is not None}


def log_training(directory: str | Path, config, manifest: dict,
                 uri: str, experiment: str = "MedReconLab-CT") -> str:
    """Log epoch metrics and small text/image outputs; excludes model weights."""
    import csv

    mlflow = _mlflow(uri)
    mlflow.set_experiment(experiment)
    directory = Path(directory)
    with mlflow.start_run(run_name="ct-unet-train") as run:
        mlflow.log_params(_params(config))
        mlflow.set_tags({"medrecon.modality": "CT", "medrecon.dataset": "synthetic",
                         "medrecon.purpose": "research-only",
                         "medrecon.git_sha": str(manifest.get("git_commit") or "unknown")})
        with (directory / "history.csv").open(newline="") as file:
            for row in csv.DictReader(file):
                step = int(row["epoch"])
                for name in ("train_mae", "val_mae", "lr"):
                    mlflow.log_metric(name, float(row[name]), step=step)
        mlflow.log_metric("best_val_mae", float(manifest["best_val_mae"]))
        mlflow.log_metric("best_epoch", int(manifest["best_epoch"]))
        for name in ("history.csv", "loss_curve.png", "manifest.json"):
            mlflow.log_artifact(str(directory / name), artifact_path="training")
        return run.info.run_id


def log_ct_sweep(directory: str | Path, config, manifest: dict,
                 uri: str, experiment: str = "MedReconLab-CT") -> str:
    """Log aggregate quality metrics and CSV/PNG exports only."""
    import pandas as pd

    mlflow = _mlflow(uri)
    mlflow.set_experiment(experiment)
    directory = Path(directory)
    with mlflow.start_run(run_name="ct-acquisition-sweep") as run:
        mlflow.log_params(_params(config))
        mlflow.set_tags({"medrecon.modality": "CT", "medrecon.dataset": "synthetic",
                         "medrecon.purpose": "research-only",
                         "medrecon.git_sha": str(manifest.get("git_commit") or "unknown")})
        frame = pd.read_csv(directory / "summary.csv")
        for method, rows in frame.groupby("method"):
            for metric in ("psnr_mean", "ssim_mean", "rmse_mean"):
                mlflow.log_metric(f"{method}_{metric}", float(rows[metric].mean()))
        mlflow.log_metric("unique_cases", int(manifest["unique_cases"]))
        for name in ("cases.csv", "summary.csv", "paired_vs_fbp.csv",
                     "psnr_vs_projections.png", "quality_vs_runtime.png",
                     "manifest.json"):
            mlflow.log_artifact(str(directory / name), artifact_path="ct_sweep")
        return run.info.run_id
