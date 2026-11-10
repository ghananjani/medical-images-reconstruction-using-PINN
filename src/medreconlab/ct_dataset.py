"""Deterministic synthetic CT pairs with disjoint train/validation/test seeds."""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import affine_transform
from skimage.transform import iradon
from torch.utils.data import Dataset

from .operators import ct_measure, phantom


def ct_case(size: int, projections: int, noise: float, case_seed: int,
            angle_span: float = 180.0) -> dict:
    """Reference, sinogram, FBP for a unique seeded perturbed Shepp-Logan phantom.

    Affine perturbation preserves a circle-supported field of view. These are
    synthetic anatomy-like shapes, not independent human patients.
    """
    if size < 16 or size % 4:
        raise ValueError("CT size must be >=16 and divisible by four")
    rng = np.random.default_rng(case_seed)
    base = phantom(size)
    angle = rng.uniform(-0.3, 0.3)
    stretch = rng.uniform(0.87, 1.13, size=2)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    transform = rotation @ np.diag(stretch)
    center = np.array([(size - 1) / 2, (size - 1) / 2])
    offset = center - transform @ center + rng.uniform(-1.5, 1.5, size=2)
    reference = affine_transform(base, transform, offset=offset, order=1,
                                 mode="constant", cval=0.0)
    # Limit support to the circle assumed by the forward operator.
    yy, xx = np.ogrid[:size, :size]
    inside = (yy - size // 2) ** 2 + (xx - size // 2) ** 2 < (size // 2 - 1) ** 2
    reference = np.clip(reference * inside, 0, 1).astype(np.float32)
    measured, theta = ct_measure(reference, projections=projections,
                                 angle_span=angle_span, noise=noise,
                                 seed=case_seed + 173)
    fbp = iradon(measured, theta=theta, filter_name="ramp", circle=True,
                 output_size=size)
    fbp = np.clip(fbp, 0, 1).astype(np.float32)
    return {"reference": reference, "fbp": fbp,
            "sinogram": measured, "theta": theta, "seed": int(case_seed)}


class SyntheticCTPairs(Dataset):
    def __init__(self, n: int, size: int = 64, projections: int = 30,
                 noise: float = 0.01, seed_offset: int = 0, angle_span: float = 180.,
                 cache: bool = True):
        if n < 1:
            raise ValueError("n must be positive")
        self.n = n
        self.size = size
        self.projections = projections
        self.noise = noise
        self.seed_offset = seed_offset
        self.angle_span = angle_span
        self._cache = [self._case(i) for i in range(n)] if cache else None

    def _case(self, index: int):
        case = ct_case(self.size, self.projections, self.noise,
                       self.seed_offset + index, self.angle_span)
        return (torch.from_numpy(case["fbp"][None].copy()),
                torch.from_numpy(case["reference"][None].copy()))

    def __len__(self):
        return self.n

    def __getitem__(self, index: int):
        if not 0 <= index < self.n:
            raise IndexError(index)
        return self._cache[index] if self._cache is not None else self._case(index)
