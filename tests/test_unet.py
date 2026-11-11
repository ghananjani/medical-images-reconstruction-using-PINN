"""Functional U-Net tests. Installed only in the PyTorch CI job."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from medreconlab.ct_dataset import SyntheticCTPairs, ct_case
from medreconlab.train_ct import TrainConfig, load_ct_unet, train_ct_unet
from medreconlab.compare_ct import compare_ct
from medreconlab.unet import CTUNet


def test_unet_forward_backward():
    model = CTUNet(base_channels=4)
    x = torch.rand(2, 1, 32, 32)
    y = model(x)
    assert y.shape == x.shape
    assert 0 <= y.min() and y.max() <= 1
    y.square().mean().backward()
    assert model.output.weight.grad is not None
    assert torch.isfinite(model.output.weight.grad).all()


def test_unet_invalid_spatial_shape():
    model = CTUNet(4)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 1, 31, 32))


def test_synthetic_ct_pairs_deterministic_and_disjoint():
    train = SyntheticCTPairs(2, size=32, projections=20, seed_offset=100_042)
    val = SyntheticCTPairs(2, size=32, projections=20, seed_offset=200_042)
    x, y = train[0]
    assert x.shape == y.shape == (1, 32, 32)
    assert not torch.allclose(y, val[0][1])
    sample = ct_case(32, 20, 0, 100_042)
    np.testing.assert_allclose(y[0].numpy(), sample["reference"])


def test_unet_training_checkpoint_and_heldout_benchmark(tmp_path):
    torch.set_num_threads(2)
    cfg = TrainConfig(output=str(tmp_path / "training"), size=32,
                      train_cases=4, val_cases=2, test_cases=2,
                      projections=20, epochs=2, patience=2,
                      batch_size=2, channels=4, seed=17, device="cpu")
    report = train_ct_unet(cfg)
    assert report["epochs_completed"] == 2
    assert 1 <= report["best_epoch"] <= 2
    assert (tmp_path / "training/best.pt").is_file()
    assert (tmp_path / "training/history.csv").is_file()
    assert (tmp_path / "training/loss_curve.png").is_file()
    model, config = load_ct_unet(tmp_path / "training/best.pt")
    with torch.inference_mode():
        assert model(torch.zeros(1, 1, 32, 32)).shape == (1, 1, 32, 32)
    assert config["seed"] == 17
    result = compare_ct(tmp_path / "training/best.pt", tmp_path / "comparison", cases=2)
    assert set(result.method) == {"fbp", "sart", "unet"}
    assert len(result) == 6
    assert np.isfinite(result["ssim"]).all()
    assert (tmp_path / "comparison/summary.csv").is_file()
    assert (tmp_path / "comparison/comparison.png").is_file()


def test_unet_invalid_configuration(tmp_path):
    cfg = TrainConfig(output=str(tmp_path), size=30)
    with pytest.raises(ValueError):
        train_ct_unet(cfg)


def test_resume_matches_uninterrupted_training(tmp_path):
    """Epoch-specific sampling must make a two-stage run reproducible on CPU."""
    import csv
    from dataclasses import replace

    torch.set_num_threads(2)
    base = TrainConfig(size=32, projections=20, noise=0.0,
                       train_cases=4, val_cases=2, test_cases=2,
                       batch_size=2, channels=4, epochs=3, patience=4,
                       seed=9, device="cpu")
    full = replace(base, output=str(tmp_path / "full"))
    split = replace(base, output=str(tmp_path / "split"), epochs=1)
    train_ct_unet(full)
    train_ct_unet(split)
    last = tmp_path / "split/last.pt"
    assert last.is_file()
    continued = replace(base, output=str(tmp_path / "split"), resume=str(last))
    manifest = train_ct_unet(continued)
    assert manifest["epochs_completed"] == 3

    with (tmp_path / "full/history.csv").open() as f:
        uninterrupted = list(csv.DictReader(f))
    with (tmp_path / "split/history.csv").open() as f:
        resumed = list(csv.DictReader(f))
    assert len(uninterrupted) == len(resumed) == 3
    for a, b in zip(uninterrupted, resumed):
        assert a["epoch"] == b["epoch"]
        np.testing.assert_allclose(float(a["train_mae"]), float(b["train_mae"]),
                                   rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(float(a["val_mae"]), float(b["val_mae"]),
                                   rtol=1e-6, atol=1e-7)


def test_resume_rejects_incompatible_configuration(tmp_path):
    from dataclasses import replace
    cfg = TrainConfig(output=str(tmp_path), size=32, projections=20,
                      train_cases=2, val_cases=2, test_cases=2, epochs=1,
                      patience=2, batch_size=2, channels=4, device="cpu")
    train_ct_unet(cfg)
    with pytest.raises(ValueError, match="incompatible resume"):
        train_ct_unet(replace(cfg, epochs=2, projections=30,
                              resume=str(tmp_path / "last.pt")))


def test_checkpoint_inference_validates_geometry(tmp_path):
    from medreconlab.ct_inference import infer_ct_checkpoint
    cfg = TrainConfig(output=str(tmp_path), size=32, projections=20,
                      train_cases=2, val_cases=2, test_cases=2, epochs=1,
                      patience=2, batch_size=2, channels=4, device="cpu")
    train_ct_unet(cfg)
    case = ct_case(32, 20, cfg.noise, 300_000 + cfg.seed, cfg.angle_span)
    result = infer_ct_checkpoint(tmp_path / "best.pt", case["fbp"],
                                 case["sinogram"], case["theta"],
                                 reference=case["reference"])
    assert result["image"].shape == (32, 32)
    assert "psnr" in result["metrics"]
    assert np.isfinite(result["measurement_residual"])
    with pytest.raises(ValueError, match="projection count"):
        infer_ct_checkpoint(tmp_path / "best.pt", case["fbp"],
                            case["sinogram"][:, :-1], case["theta"][:-1],
                            reference=case["reference"])
