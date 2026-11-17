"""Seeded parameter sweeps: real computation, never hardcoded metric values."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from medreconlab.pipeline import experiment


def benchmark(modality: str, size: int = 64, seeds=(0, 1, 2)) -> pd.DataFrame:
    configs = {
        "ct": [dict(projections=p, angle_span=span, noise=noise)
               for p in (30, 60, 90, 180)
               for span in (120.0, 180.0)
               for noise in (0.0, 0.01)],
        "mri": [dict(acceleration=a, noise=noise)
                for a in (2, 4, 8) for noise in (0.0, 0.01)],
        "xray": [dict(sigma=s, noise=noise)
                 for s in (1.0, 2.0, 3.0) for noise in (0.0, 0.02)],
    }
    if modality not in configs:
        raise ValueError("invalid modality")
    rows = []
    for params in configs[modality]:
        for seed in seeds:
            outcome = experiment(modality=modality, size=size, seed=int(seed), **params)
            for method, data in outcome["reconstructions"].items():
                rows.append({"modality": modality, "method": method, "seed": seed,
                             **params, **data["metrics"],
                             "runtime_s": data["runtime_s"],
                             "measurement_residual": data["measurement_residual"]})
    return pd.DataFrame(rows)


def save_benchmark(frame: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "cases.csv", index=False)
    columns = [c for c in ("projections", "angle_span", "acceleration", "sigma", "noise")
               if c in frame.columns]
    groups = ["modality", "method", *columns]
    stats = frame.groupby(groups, dropna=False).agg(
        psnr_mean=("psnr", "mean"), psnr_sd=("psnr", "std"),
        ssim_mean=("ssim", "mean"), runtime_mean_s=("runtime_s", "mean"),
        n=("seed", "count")).reset_index()
    stats["psnr_95ci_halfwidth_normal"] = 1.96 * stats["psnr_sd"] / np.sqrt(stats["n"])
    stats.to_csv(output / "summary.csv", index=False)
    xcol = "projections" if "projections" in frame else (
        "acceleration" if "acceleration" in frame else "sigma")
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, subset in frame.groupby("method"):
        mean = subset.groupby(xcol)["psnr"].mean()
        ax.plot(mean.index, mean.values, marker="o", label=method)
    ax.set_xlabel(xcol.replace("_", " ").title())
    ax.set_ylabel("PSNR (dB), synthetic reference")
    ax.set_title("Reference-based reconstruction quality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "psnr_sweep.png", dpi=180)
    plt.close(fig)
    (output / "manifest.json").write_text(json.dumps(
        {"seeds": sorted(set(map(int, frame["seed"]))), "n_cases": len(frame),
         "note": "Normal-approximation confidence intervals; use caution for n<30."}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=("ct", "mri", "xray"), required=True)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--output", default="outputs/benchmark")
    args = p.parse_args()
    data = benchmark(args.modality, args.size, args.seeds)
    save_benchmark(data, Path(args.output))
    print(f"Computed {len(data)} real method-case rows in {args.output}")


if __name__ == "__main__":
    main()
