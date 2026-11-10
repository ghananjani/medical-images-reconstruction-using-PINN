"""Reference-based metrics for synthetic, aligned, unit-scale images."""
from __future__ import annotations
import numpy as np
from skimage.metrics import structural_similarity


def evaluate(reference: np.ndarray | None, estimate: np.ndarray) -> dict[str, float]:
    if reference is None:
        return {}
    a, b = np.asarray(reference, dtype=float), np.asarray(estimate, dtype=float)
    if a.shape != b.shape:
        raise ValueError("reference and estimate must have identical shapes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("metrics require finite values")
    diff = a - b
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    return {
        "mse": mse, "rmse": rmse, "mae": float(np.mean(np.abs(diff))),
        "nrmse": float(rmse / max(np.sqrt(np.mean(a ** 2)), 1e-12)),
        "psnr": float("inf") if mse == 0 else float(10 * np.log10(1 / mse)),
        "ssim": float(structural_similarity(a, b, data_range=1.0)),
    }
