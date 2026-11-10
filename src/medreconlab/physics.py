"""Optional differentiable physics optimization; NOT a PDE PINN."""
from __future__ import annotations


def torch_ct_forward(image, angles):
    """Approximate parallel-beam Radon with differentiable torch grid_sample."""
    import torch
    import torch.nn.functional as F
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("image shape must be [batch,1,height,width]")
    projections = []
    for angle in angles:
        c, s = torch.cos(angle), torch.sin(angle)
        mat = torch.stack([c, -s, torch.zeros_like(c), s, c, torch.zeros_like(c)])
        mat = mat.reshape(1, 2, 3).expand(image.shape[0], -1, -1)
        grid = F.affine_grid(mat, image.shape, align_corners=False)
        rotated = F.grid_sample(image, grid, align_corners=False)
        projections.append(rotated.sum(dim=-2))
    return torch.stack(projections, dim=-1)


def ct_optimize(measured, angles, size: int, steps: int = 100, lr: float = 0.05):
    """Fit pixel parameters to the SAME differentiable torch operator as the measurement."""
    import torch
    x = torch.nn.Parameter(torch.zeros((1, 1, size, size),
                                       device=measured.device, dtype=measured.dtype))
    optimizer = torch.optim.Adam([x], lr=lr)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        pred = torch_ct_forward(x, angles)
        loss = (pred - measured).square().mean()
        loss = loss + 0.001 * ((x[..., 1:, :] - x[..., :-1, :]).abs().mean() +
                                (x[..., :, 1:] - x[..., :, :-1]).abs().mean())
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return x.detach().clamp(0, 1), losses


def mri_optimize(kspace, mask, steps: int = 100, lr: float = 0.05):
    """Optimize a complex image with exact acquired k-space penalty and TV prior."""
    import torch
    x = torch.nn.Parameter(torch.fft.ifft2(torch.fft.ifftshift(kspace), norm="ortho").clone())
    optimizer = torch.optim.Adam([x], lr=lr)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        k = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"))
        residual = k[mask] - kspace[mask]
        loss = residual.abs().square().mean()
        loss = loss + 0.001 * ((x[1:] - x[:-1]).abs().mean() +
                                (x[:, 1:] - x[:, :-1]).abs().mean())
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return x.detach(), losses
