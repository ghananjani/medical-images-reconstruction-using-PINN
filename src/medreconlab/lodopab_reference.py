"""Evaluate the published DIVαℓ LoDoPaB-CT FBP+U-Net reference model."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np

REFERENCE_REVISION = "6117cabc55d223f5b62ea43d4a40225270fb6756"
CHECKPOINT_SHA256 = "990b366a5053485093e2b9876f1503d92fcd786b44f8970ca6f5d9e4c1dfdf23"
REFERENCE_FBP_FREQUENCY = 0.641025641025641
UNET_FBP_FREQUENCY = 1.0


def load_reference(ray, checkpoint: Path):
    """Build the released architecture and verify its weights before loading."""
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError("DIVαℓ reference checkpoint SHA-256 mismatch")

    import torch
    from dival.reconstructors.fbpunet_reconstructor import FBPUNetReconstructor
    from odl.tomo import fbp_op

    reconstructor = FBPUNetReconstructor(
        ray, use_cuda=False,
        hyper_params={
            "scales": 5, "skip_channels": 4, "filter_type": "Hann",
            "frequency_scaling": UNET_FBP_FREQUENCY, "use_sigmoid": False,
            "normalize_by_opnorm": False,
        },
    )
    reconstructor.init_model()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reconstructor.model.load_state_dict(
        {key.removeprefix("module."): value for key, value in state.items()}, strict=True
    )
    reconstructor.model.eval()
    reference_fbp = fbp_op(ray, filter_type="Hann", frequency_scaling=REFERENCE_FBP_FREQUENCY)

    @torch.inference_mode()
    def reconstruct(measurement):
        return np.asarray(reconstructor.reconstruct(measurement))

    metadata = {
        "weights_revision": REFERENCE_REVISION,
        "weights_sha256": digest,
        "fbp_reference": {"filter": "Hann", "frequency_scaling": REFERENCE_FBP_FREQUENCY},
        "unet_input_fbp": {"filter": "Hann", "frequency_scaling": UNET_FBP_FREQUENCY},
        "unet_device": "cpu",
        "versions": {package: version(package) for package in ("dival", "torch")},
    }
    return {"fbp_reference": reference_fbp, "fbp_unet_reference": reconstruct}, metadata
