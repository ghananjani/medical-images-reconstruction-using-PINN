"""Synthetic modality-specific forward models (no patient data)."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize


def phantom(size: int = 64) -> np.ndarray:
    if not 16 <= size <= 1024:
        raise ValueError("size must be between 16 and 1024")
    return np.clip(resize(shepp_logan_phantom(), (size, size), anti_aliasing=True), 0, 1)


def ct_measure(image: np.ndarray, projections: int = 60, angle_span: float = 180.0,
               noise: float = 0.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if not 2 <= projections <= 720 or not 0 < angle_span <= 180 or noise < 0:
        raise ValueError("invalid CT acquisition")
    theta = np.linspace(0, angle_span, projections, endpoint=False)
    y = radon(image, theta=theta, circle=True)
    if noise:
        y = y + np.random.default_rng(seed).normal(0, noise, y.shape)
    return y.astype(np.float64), theta


def fft2c(image: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image), norm="ortho"))


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace), norm="ortho"))



def mri_forward(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Single-coil undersampled orthonormal Fourier forward operator."""
    if image.shape != mask.shape:
        raise ValueError("image and mask shapes must match")
    return np.where(np.asarray(mask, dtype=bool), fft2c(image), 0)


def mri_adjoint(kspace: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Hermitian adjoint of mri_forward under the complex inner product."""
    if kspace.shape != mask.shape:
        raise ValueError("kspace and mask shapes must match")
    return ifft2c(np.where(np.asarray(mask, dtype=bool), kspace, 0))


def mri_measure(image: np.ndarray, acceleration: int = 4, seed: int = 0,
                noise: float = 0.0, pattern: str = "equispaced") -> tuple[np.ndarray, np.ndarray]:
    if acceleration not in (2, 4, 8) or noise < 0:
        raise ValueError("invalid MRI acquisition")
    n = image.shape[1]
    if pattern == "equispaced":
        cols = np.arange(seed % acceleration, n, acceleration)
    elif pattern == "random":
        cols = np.random.default_rng(seed).choice(n, size=max(1, n // acceleration), replace=False)
    else:
        raise ValueError("unknown MRI mask")
    mask = np.zeros(image.shape, dtype=bool)
    mask[:, cols] = True
    # Preserve central low frequencies.
    center = n // 2
    mask[:, max(0, center - 2):min(n, center + 2)] = True
    k = fft2c(image.astype(np.complex128))
    if noise:
        rng = np.random.default_rng(seed + 9182)
        k = k + noise * (rng.normal(size=k.shape) + 1j * rng.normal(size=k.shape)) / np.sqrt(2)
    return np.where(mask, k, 0), mask


def xray_measure(image: np.ndarray, sigma: float = 1.5, noise: float = 0.02,
                 seed: int = 0) -> np.ndarray:
    if sigma <= 0 or noise < 0:
        raise ValueError("invalid X-ray degradation")
    blurred = gaussian_filter(image, sigma=sigma, mode="reflect")
    return np.clip(blurred + np.random.default_rng(seed).normal(0, noise, image.shape), 0, 1)
