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
