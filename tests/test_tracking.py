"""End-to-end local MLflow tracking, with no checkpoint artifact uploads."""
from __future__ import annotations

import csv
import pandas as pd
import pytest

from medreconlab.tracking import log_ct_sweep, log_training

mlflow = pytest.importorskip("mlflow")


def test_real_mlflow_training_run(tmp_path):
    out = tmp_path / "train"
    out.mkdir()
    with (out / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_mae", "val_mae", "lr"])
        writer.writeheader()
        writer.writerow({"epoch": 1, "train_mae": .4, "val_mae": .3, "lr": .001})
        writer.writerow({"epoch": 2, "train_mae": .2, "val_mae": .15, "lr": .001})
    (out / "loss_curve.png").write_bytes(b"research placeholder for artifact API test")
    (out / "manifest.json").write_text('{"synthetic": true}')
    (out / "best.pt").write_bytes(b"must never be sent to MLflow")
    uri = "sqlite:///" + str(tmp_path / "tracking_train.db")
    run_id = log_training(out, {"size": 32, "seed": 42}, {
        "best_val_mae": .15, "best_epoch": 2, "git_commit": "test-sha"
    }, uri, "medrecon-test-train")
    mlflow.set_tracking_uri(uri)
    run = mlflow.get_run(run_id)
    assert run.data.metrics["val_mae"] == pytest.approx(.15)
    assert run.data.params["size"] == "32"
    artifacts = [x.path for x in mlflow.artifacts.list_artifacts(run_id=run_id, artifact_path="training")]
    assert "training/history.csv" in artifacts
    assert "training/best.pt" not in artifacts


def test_real_mlflow_sweep_run(tmp_path):
    out = tmp_path / "sweep"
    out.mkdir()
    pd.DataFrame({
        "method": ["fbp", "sart"], "psnr_mean": [12., 14.],
        "ssim_mean": [.3, .4], "rmse_mean": [.2, .1]
    }).to_csv(out / "summary.csv", index=False)
    for name in ("cases.csv", "paired_vs_fbp.csv", "psnr_vs_projections.png",
                 "quality_vs_runtime.png", "manifest.json"):
        (out / name).write_text("synthetic test data")
    uri = "sqlite:///" + str(tmp_path / "tracking_sweep.db")
    run_id = log_ct_sweep(out, {"projections": (30,)},
                          {"unique_cases": 2, "git_commit": "test-sha"}, uri,
                          "medrecon-test-sweep")
    mlflow.set_tracking_uri(uri)
    run = mlflow.get_run(run_id)
    assert run.data.metrics["fbp_psnr_mean"] == pytest.approx(12.)
    assert run.data.metrics["unique_cases"] == 2
    paths = [x.path for x in mlflow.artifacts.list_artifacts(run_id=run_id, artifact_path="ct_sweep")]
    assert "ct_sweep/summary.csv" in paths


def test_read_only_experiment_dashboard_history(tmp_path, monkeypatch):
    from medreconlab.experiment_view import read_experiment_history, local_tracking_uri

    monkeypatch.chdir(tmp_path)
    out = tmp_path / "training"
    out.mkdir()
    with (out / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_mae", "val_mae", "lr"])
        writer.writeheader()
        writer.writerow({"epoch": 1, "train_mae": .4, "val_mae": .3, "lr": .001})
        writer.writerow({"epoch": 2, "train_mae": .2, "val_mae": .15, "lr": .001})
    (out / "manifest.json").write_text('{"synthetic": true}')
    (out / "loss_curve.png").write_bytes(b"test image")
    database = tmp_path / "runs.db"
    uri = "sqlite:///" + str(database)
    run_id = log_training(out, {"seed": 42}, {
        "best_val_mae": .15, "best_epoch": 2}, uri, "medrecon-dashboard-test")
    assert local_tracking_uri("runs.db") == uri
    summaries, curves = read_experiment_history("runs.db")
    row = next(row for row in summaries if row["run_id"] == run_id)
    assert row["metric_best_val_mae"] == pytest.approx(.15)
    assert sorted(point["step"] for point in curves[run_id]
                  if point["metric"] == "val_mae") == [1, 2]


def test_experiment_dashboard_blocks_outside_files(tmp_path, monkeypatch):
    from medreconlab.experiment_view import local_tracking_uri, read_experiment_history

    inside = tmp_path / "project"
    inside.mkdir()
    outside = tmp_path / "outside.db"
    outside.write_text("not a database")
    monkeypatch.chdir(inside)
    with pytest.raises(ValueError, match="inside"):
        local_tracking_uri(outside)
    with pytest.raises(FileNotFoundError, match="existing"):
        read_experiment_history("missing.db")
    with pytest.raises(ValueError, match="limit"):
        read_experiment_history("missing.db", limit=0)
