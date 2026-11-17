"""Real baseline algorithms with modality-specific measurement consistency."""
from __future__ import annotations
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.restoration import richardson_lucy, wiener
from skimage.transform import iradon, iradon_sart, radon
from .operators import fft2c, ifft2c


METHODS = {"ct": ("fbp", "sart"), "mri": ("zero_filled", "iterative_dc"),
           "xray": ("wiener", "richardson_lucy")}


def reconstruct(modality: str, method: str, measured: np.ndarray,
                return_complex: bool = False, **settings) -> np.ndarray:
    if modality not in METHODS or method not in METHODS[modality]:
        raise ValueError(f"unsupported modality/method: {modality}/{method}")
    if modality == "ct":
        theta = np.asarray(settings["theta"])
        size = int(settings["size"])
        if method == "fbp":
            result = iradon(measured, theta=theta, filter_name="ramp",
                            circle=True, output_size=size)
        else:
            result = None
            for _ in range(int(settings.get("iterations", 5))):
                result = iradon_sart(measured, theta=theta, image=result, relaxation=0.1)
            # With circle=True, the detector width equals the image size.
    elif modality == "mri":
        mask = np.asarray(settings["mask"], dtype=bool)
        k = np.asarray(measured, dtype=np.complex128)
        if method == "zero_filled":
            result = ifft2c(k)
        else:
            # Iterative projected smoothing: an explicit, simple baseline, not compressed sensing.
            image = ifft2c(k)
            for _ in range(int(settings.get("iterations", 25))):
                prior = gaussian_filter(image.real, 0.65) + 1j * gaussian_filter(image.imag, 0.65)
                proposal = fft2c(prior)
                proposal[mask] = k[mask]  # exact acquired complex sample consistency
                image = ifft2c(proposal)
            result = image
        if return_complex:
            return np.asarray(result, dtype=np.complex128)
        result = np.abs(result)
    else:
        sigma = float(settings.get("sigma", 1.5))
        radius = max(3, int(np.ceil(4 * sigma)))
        ax = np.arange(-radius, radius + 1)
        xx, yy = np.meshgrid(ax, ax)
        psf = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
        psf /= psf.sum()
        if method == "wiener":
            result = wiener(measured, psf, balance=float(settings.get("balance", 0.01)),
                            clip=False)
        else:
            result = richardson_lucy(measured, psf,
                                    num_iter=int(settings.get("iterations", 20)), clip=False)
    return np.clip(np.asarray(result, dtype=float), 0, 1)


def measurement_residual(modality: str, reconstructed: np.ndarray,
                         measured: np.ndarray, **settings) -> float:
    if modality == "ct":
        predicted = radon(reconstructed, theta=np.asarray(settings["theta"]), circle=True)
    elif modality == "mri":
        mask = np.asarray(settings["mask"], dtype=bool)
        predicted = fft2c(np.asarray(reconstructed, dtype=np.complex128))[mask]
        measured = measured[mask]
    else:
        predicted = gaussian_filter(reconstructed, sigma=float(settings.get("sigma", 1.5)),
                                  mode="reflect")
    return float(np.linalg.norm(predicted - measured) /
                 max(np.linalg.norm(measured), 1e-12))


def run_methods(modality: str, measured: np.ndarray, settings: dict) -> dict:
    results = {}
    for method in METHODS[modality]:
        start = time.perf_counter()
        raw = reconstruct(modality, method, measured,
                          return_complex=(modality == "mri"), **settings)
        elapsed = time.perf_counter() - start
        image = np.abs(raw) if modality == "mri" else raw
        results[method] = {"image": image, "runtime_s": elapsed,
                           "measurement_residual": measurement_residual(
                               modality, raw, measured, **settings)}
    return results
