"""Held-out CT U-Net versus FBP/SART, measured on the same sinograms."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .ct_dataset import ct_case
from .metrics import evaluate
from .methods import measurement_residual, reconstruct
from .train_ct import load_ct_unet


def compare_ct(checkpoint: str | Path, output: str | Path, cases: int = 8,
               device: str = "cpu") -> pd.DataFrame:
    model, config = load_ct_unet(checkpoint, device=device)
    if cases < 1:
        raise ValueError("cases must be positive")
    result_dir = Path(output)
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    figure_case = None
    for index in range(cases):
        seed = 300_000 + int(config["seed"]) + index
        sample = ct_case(int(config["size"]), int(config["projections"]),
                         float(config["noise"]), seed, float(config["angle_span"]))
        settings = {"theta": sample["theta"], "size": int(config["size"])}
        reference = sample["reference"]
        reconstruction = {}
        for method in ("fbp", "sart"):
            start = time.perf_counter()
            prediction = reconstruct("ct", method, sample["sinogram"], **settings)
            elapsed = time.perf_counter() - start
            reconstruction[method] = (prediction, elapsed)
        image = torch.from_numpy(sample["fbp"][None, None].copy()).to(device)
        with torch.inference_mode():
            start = time.perf_counter()
            prediction = model(image).squeeze().cpu().numpy()
            # Synchronize when timing CUDA inference.
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        reconstruction["unet"] = (prediction, elapsed)
        for method, (estimate, runtime) in reconstruction.items():
            rows.append({"seed": seed, "case": index, "method": method,
                         **evaluate(reference, estimate),
                         "runtime_s": runtime,
                         "measurement_residual": measurement_residual(
                             "ct", estimate, sample["sinogram"], **settings)})
        if index == 0:
            figure_case = (reference, reconstruction)

    data = pd.DataFrame(rows)
    data.to_csv(result_dir / "per_case.csv", index=False)
    data.groupby("method").agg(psnr_mean=("psnr", "mean"),
                               ssim_mean=("ssim", "mean"),
                               mae_mean=("mae", "mean"),
                               runtime_mean_s=("runtime_s", "mean"),
                               n=("case", "size")).to_csv(result_dir / "summary.csv")
    reference, images = figure_case
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    named = [("Reference", reference)] + [(m, images[m][0])
                                          for m in ("fbp", "sart", "unet")]
    for i, (name, image) in enumerate(named):
        axes[0, i].imshow(image, vmin=0, vmax=1, cmap="gray")
        axes[0, i].set_title(name)
        axes[1, i].imshow(abs(image - reference), vmin=0, vmax=1, cmap="magma")
        axes[1, i].set_title("Absolute error")
        for axis in axes[:, i]:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(result_dir / "comparison.png", dpi=180)
    plt.close(fig)
    (result_dir / "manifest.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "cases": cases,
        "seeds": [300_000 + int(config["seed"]) + i for i in range(cases)],
        "methods": ["fbp", "sart", "unet"],
        "note": "Synthetic held-out cases; no claim U-Net outperforms baselines."
    }, indent=2))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/ct_unet/comparison")
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = compare_ct(args.checkpoint, args.output, args.cases, args.device)
    print(result.groupby("method")[["psnr", "ssim", "runtime_s"]].mean())


if __name__ == "__main__":
    main()
