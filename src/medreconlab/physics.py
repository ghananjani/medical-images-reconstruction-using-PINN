"""Differentiable physics-consistent optimization; not a PDE PINN.

CT training and assessment must use the same approximate torch forward model.
MRI measurements remain complex and acquired samples are projected exactly.
"""
from __future__ import annotations


def torch_ct_forward(image, angles):
    """Differentiable approximate parallel-beam sinogram, [B,N,A] output.

    Uses zero-padded affine sampling of N x N grid; differs numerically from
    scikit-image Radon, so do not compare measurements across the operators.
    """
    import torch
    import torch.nn.functional as F

    if image.ndim != 4 or image.shape[1] != 1 or image.shape[-1] != image.shape[-2]:
        raise ValueError("image must be [batch,1,N,N]")
    angles = torch.as_tensor(angles, dtype=image.dtype, device=image.device)
    projections = []
    for angle in angles.reshape(-1):
        c, s = torch.cos(angle), torch.sin(angle)
        mat = torch.stack([c, -s, torch.zeros_like(c), s, c, torch.zeros_like(c)])
        matrix = mat.reshape(1, 2, 3).expand(image.shape[0], -1, -1)
        grid = F.affine_grid(matrix, image.shape, align_corners=False)
        rotated = F.grid_sample(image, grid, align_corners=False, padding_mode="zeros")
        projections.append(rotated.sum(dim=-2).squeeze(1))
    if not projections:
        raise ValueError("at least one angle is required")
    return torch.stack(projections, dim=-1)


def _torch_fft2c(image):
    import torch
    return torch.fft.fftshift(torch.fft.fft2(
        torch.fft.ifftshift(image, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def _torch_ifft2c(kspace):
    import torch
    return torch.fft.fftshift(torch.fft.ifft2(
        torch.fft.ifftshift(kspace, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ct_optimize(measured, angles, size: int, steps: int = 100, lr: float = 0.05,
                tv_weight: float = 0.001, reference=None, supervised_weight: float = 0.0):
    """Fit image pixels with projection + TV losses, optional L1 supervision.

    The measurement must come from torch_ct_forward (radians), not scikit-image.
    """
    import torch

    if steps < 1 or size < 2 or lr <= 0 or tv_weight < 0 or supervised_weight < 0:
        raise ValueError("invalid CT optimization parameters")
    if supervised_weight and reference is None:
        raise ValueError("supervision requires reference")
    if tuple(measured.shape[:2]) != (1, size):
        raise ValueError("measured shape must be [1,N,num_angles]")
    x = torch.nn.Parameter(torch.zeros(
        (1, 1, size, size), device=measured.device, dtype=measured.dtype))
    optimizer = torch.optim.Adam([x], lr=lr)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        predicted = torch_ct_forward(x, angles)
        loss = (predicted - measured).square().mean()
        tv = ((x[..., 1:, :] - x[..., :-1, :]).abs().mean()
              + (x[..., :, 1:] - x[..., :, :-1]).abs().mean())
        loss = loss + tv_weight * tv
        if reference is not None and supervised_weight:
            loss = loss + supervised_weight * (x - reference).abs().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return x.detach().clamp(0, 1), losses


def mri_optimize(kspace, mask, steps: int = 100, lr: float = 0.05,
                 tv_weight: float = 0.001, reference=None, supervised_weight: float = 0.0):
    """Complex-valued single-coil image optimization with hard data projection.

    Inputs are centered k-space and Boolean mask with matching 2D shapes.
    The returned complex image exactly matches acquired k-space samples up to
    floating-point tolerance. This is a physics-informed pixel prior, not a PDE PINN.
    """
    import torch

    if kspace.ndim != 2 or mask.shape != kspace.shape or mask.dtype != torch.bool:
        raise ValueError("kspace and Boolean mask must be matching 2D arrays")
    if not mask.any().item() or steps < 1 or lr <= 0 or tv_weight < 0:
        raise ValueError("invalid MRI optimization settings")
    if supervised_weight < 0 or (supervised_weight and reference is None):
        raise ValueError("supervision requires a reference and nonnegative weight")
    x = torch.nn.Parameter(_torch_ifft2c(kspace).clone())
    optimizer = torch.optim.Adam([x], lr=lr)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        predicted = _torch_fft2c(x)
        residual = predicted[mask] - kspace[mask]
        loss = residual.abs().square().mean()
        tv = ((x[1:, :] - x[:-1, :]).abs().mean()
              + (x[:, 1:] - x[:, :-1]).abs().mean())
        loss = loss + tv_weight * tv
        if reference is not None and supervised_weight:
            loss = loss + supervised_weight * (x.abs() - reference).abs().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        final_k = _torch_fft2c(x)
        final_k[mask] = kspace[mask]
        final = _torch_ifft2c(final_k)
    return final.detach(), losses
