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
