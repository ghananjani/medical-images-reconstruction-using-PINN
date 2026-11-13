"""Opt-in inference with a locally trusted CT U-Net checkpoint."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .methods import measurement_residual
from .metrics import evaluate


def infer_ct_checkpoint(checkpoint: str | Path, fbp: np.ndarray, sinogram: np.ndarray,
                        theta: np.ndarray, reference: np.ndarray | None = None,
                        device: str = "cpu", expected_projections: int | None = None):
    """Run actual trained checkpoint on a matching reconstruction geometry.

    Checkpoint loading is only for files created locally by trusted code.
    """
    import torch

    from .train_ct import load_ct_unet

    x = np.asarray(fbp, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("expected a square 2D FBP image")
    model, cfg = load_ct_unet(checkpoint, device=device)
    if x.shape != (int(cfg["size"]), int(cfg["size"])):
        raise ValueError("checkpoint size does not match current CT acquisition")
    if len(theta) != int(cfg["projections"]):
        raise ValueError("checkpoint projection count does not match acquisition")
    if expected_projections is not None and len(theta) != expected_projections:
        raise ValueError("inconsistent sinogram projection count")
    if np.asarray(sinogram).shape != (x.shape[0], len(theta)):
        raise ValueError("invalid sinogram geometry")
    if not np.isclose(float(cfg["angle_span"]), float(theta[-1] + theta[1] - theta[0])):
        raise ValueError("checkpoint angle span does not match acquisition")
    tensor = torch.from_numpy(x[None, None].copy()).to(device)
    with torch.inference_mode():
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(tensor).squeeze().cpu().numpy()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        runtime = time.perf_counter() - start
    return {
        "image": output,
        "runtime_s": runtime,
        "measurement_residual": measurement_residual(
            "ct", output, np.asarray(sinogram), theta=theta),
        "metrics": evaluate(reference, output),
        "checkpoint": str(checkpoint),
    }
