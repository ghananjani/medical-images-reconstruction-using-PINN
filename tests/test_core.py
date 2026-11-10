import numpy as np
import pytest
from medreconlab.operators import phantom, fft2c, ifft2c, ct_measure, mri_measure
from medreconlab.metrics import evaluate
from medreconlab.pipeline import experiment


def test_fft_round_trip_complex():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
    np.testing.assert_allclose(ifft2c(fft2c(x)), x, atol=1e-12)


@pytest.mark.parametrize("modality", ["ct", "mri", "xray"])
def test_cpu_end_to_end(modality):
    outcome = experiment(modality=modality, size=32, noise=0.0)
    assert len(outcome["reconstructions"]) >= 2
    for row in outcome["reconstructions"].values():
        assert row["image"].shape == (32, 32)
        assert np.isfinite(row["image"]).all()
        assert np.isfinite(row["runtime_s"])
        assert np.isfinite(row["measurement_residual"])
        assert "psnr" in row["metrics"]


def test_ct_sinogram():
    measured, theta = ct_measure(phantom(32), projections=30)
    assert measured.shape[1] == len(theta) == 30


def test_mri_complex_mask():
    k, mask = mri_measure(phantom(32), acceleration=4)
    assert np.iscomplexobj(k)
    assert mask.dtype == bool
    assert np.all(k[~mask] == 0)


def test_missing_reference_skips_reference_metrics():
    assert evaluate(None, np.zeros((32, 32))) == {}


def test_mismatched_metric_shapes_rejected():
    with pytest.raises(ValueError):
        evaluate(np.zeros((32, 32)), np.zeros((16, 16)))


def test_ct_operator_output_matches_detector_width():
    image = phantom(32)
    measured, theta = ct_measure(image, projections=30)
    assert measured.shape == (32, len(theta))


def test_ct_sart_uses_native_geometry():
    from medreconlab.methods import reconstruct
    measured, theta = ct_measure(phantom(32), projections=30)
    image = reconstruct("ct", "sart", measured, theta=theta, size=32, iterations=2)
    assert image.shape == (32, 32)


def test_mri_complex_measurement_consistency_before_display():
    from medreconlab.methods import reconstruct, measurement_residual\n    from medreconlab.operators import fft2c
    reference = phantom(32)
    k, mask = mri_measure(reference, acceleration=4)
    for method in ("zero_filled", "iterative_dc"):
        image = reconstruct("mri", method, k, mask=mask, return_complex=True)
        np.testing.assert_allclose(fft2c(image)[mask], k[mask], atol=1e-10)
        assert measurement_residual("mri", image, k, mask=mask) < 1e-10


def test_torch_mri_physics_optional():
    torch = pytest.importorskip("torch")
    from medreconlab.physics import _torch_fft2c, _torch_ifft2c, mri_optimize
    k, mask = mri_measure(phantom(16), acceleration=4)
    tensor = torch.tensor(k)
    bmask = torch.tensor(mask)
    np.testing.assert_allclose(_torch_fft2c(_torch_ifft2c(tensor)).numpy(), k,
                               atol=1e-10)
    result, losses = mri_optimize(tensor, bmask, steps=3)
    assert len(losses) == 3
    assert torch.allclose(_torch_fft2c(result)[bmask], tensor[bmask], atol=1e-10)


def test_torch_ct_physics_optional():
    torch = pytest.importorskip("torch")
    from medreconlab.physics import torch_ct_forward, ct_optimize
    image = torch.tensor(phantom(16), dtype=torch.float32).view(1, 1, 16, 16)
    theta = torch.linspace(0, torch.pi, 6)
    measured = torch_ct_forward(image, theta)
    recon, losses = ct_optimize(measured, theta, 16, steps=2)
    assert recon.shape == image.shape
    assert len(losses) == 2
    assert np.isfinite(losses).all()
