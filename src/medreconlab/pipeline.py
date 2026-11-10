"""Reproducible synthetic benchmark pipeline."""
from __future__ import annotations
from .operators import phantom, ct_measure, mri_measure, xray_measure, ifft2c
from .methods import run_methods
from .metrics import evaluate


def experiment(modality: str = "ct", size: int = 64, seed: int = 0,
               projections: int = 60, acceleration: int = 4, sigma: float = 1.5,
               noise: float = 0.01, angle_span: float = 180.0) -> dict:
    reference = phantom(size)
    if modality == "ct":
        measured, theta = ct_measure(reference, projections, angle_span, noise, seed)
        settings = {"theta": theta, "size": size}
        degraded = np.clip(__import__("skimage.transform", fromlist=["iradon"]).iradon(
            measured, theta=theta, filter_name=None, circle=False, output_size=size), 0, 1)
    elif modality == "mri":
        measured, mask = mri_measure(reference, acceleration, seed, noise)
        settings = {"mask": mask}
        degraded = np.abs(ifft2c(measured))
    elif modality == "xray":
        measured = xray_measure(reference, sigma, noise, seed)
        settings = {"sigma": sigma}
        degraded = measured
    else:
        raise ValueError("modality must be ct, mri or xray")
    reconstructions = run_methods(modality, measured, settings)
    for result in reconstructions.values():
        result["metrics"] = evaluate(reference, result["image"])
    return {"modality": modality, "reference": reference, "degraded": degraded,
            "measured": measured, "settings": settings,
            "reconstructions": reconstructions,
            "parameters": {"size": size, "seed": seed, "projections": projections,
                           "acceleration": acceleration, "sigma": sigma, "noise": noise,
                           "angle_span": angle_span}}
