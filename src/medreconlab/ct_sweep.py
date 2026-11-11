"""CT acquisition ladder: matched synthetic cases, measured methods, paired uncertainty.

Research-only. Never extrapolate synthetic phantom results to clinical performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ct_dataset import ct_case
from .methods import measurement_residual, reconstruct
from .metrics import evaluate


@dataclass(frozen=True)
class CTSweepConfig:
    output: str = "outputs/ct_ladder"
    size: int = 64
    projections: tuple[int, ...] = (30, 60, 90, 180)
    angle_spans: tuple[float, ...] = (120.0, 180.0)
    noise_levels: tuple[float, ...] = (0.0, 0.01, 0.03)
    seeds: tuple[int, ...] = (300042, 300043, 300044)
    iterations: int = 5
    bootstrap_repeats: int = 1000
    bootstrap_seed: int = 11
    checkpoint: str | None = None
    device: str = "cpu"

    def validate(self):
        if self.size < 16 or self.size % 4:
            raise ValueError("size must be >=16 and divisible by four")
        if not all((self.projections, self.angle_spans, self.noise_levels, self.seeds)):
            raise ValueError("sweep axes and seeds cannot be empty")
        if any(p < 2 or p > 720 for p in self.projections):
            raise ValueError("CT projection counts must be between 2 and 720")
        if any(s <= 0 or s > 180 for s in self.angle_spans):
            raise ValueError("CT angle spans must be in (0,180]")
        if any(n < 0 or not np.isfinite(n) for n in self.noise_levels):
            raise ValueError("noise must be finite and nonnegative")
        if len(set(self.seeds)) != len(self.seeds) or any(s < 300000 for s in self.seeds):
            raise ValueError("use distinct held-out case seeds >=300000")
        if self.iterations < 1 or self.bootstrap_repeats < 1:
            raise ValueError("iterations and bootstrap repeats must be positive")


def _checkpoint_for_config(checkpoint, size, projections, span, noise, device):
    if checkpoint is None:
        return None
    from .train_ct import load_ct_unet

    model, config = load_ct_unet(checkpoint, device=device)
    fields = {"size": size, "projections": projections, "angle_span": span, "noise": noise}
    for key, expected in fields.items():
        if not np.isclose(float(config[key]), float(expected)):
            raise ValueError(
                f"checkpoint {key}={config[key]} mismatches sweep setting {expected}; "
                "train separate geometry-matched models or omit --checkpoint"
            )
    if int(config["seed"]) >= 100000:
        raise ValueError("invalid checkpoint training seed")
    return model


def sweep_ct(config: CTSweepConfig) -> pd.DataFrame:
    """Compare FBP, SART, and optional geometry-matched U-Net using identical data."""
    config.validate()
    import torch  # CT synthetic case uses torch; job is in the physics CPU environment.

    model = None
    if config.checkpoint is not None:
        # Refuse to silently present an out-of-domain model as in-domain.
        if len(config.projections) != 1 or len(config.angle_spans) != 1 or len(config.noise_levels) != 1:
            raise ValueError("U-Net requires a single matching acquisition configuration")
        model = _checkpoint_for_config(
            config.checkpoint, config.size, config.projections[0],
            config.angle_spans[0], config.noise_levels[0], config.device
        )
    rows = []
    for span in config.angle_spans:
        for noise in config.noise_levels:
            for projections in config.projections:
                for seed in config.seeds:
                    case = ct_case(config.size, projections, noise, seed, span)
                    reference = case["reference"]
                    settings = {"theta": case["theta"], "size": config.size}
                    for method in ("fbp", "sart"):
                        begin = time.perf_counter()
                        if method == "fbp":
                            image = case["fbp"]  # same FBP image used for U-Net input
                            # Reconstruction performed during case generation, not timed here.
                            # Recompute to time the baseline fairly.
                            image = reconstruct("ct", "fbp", case["sinogram"], **settings)
                        else:
                            image = reconstruct(
                                "ct", "sart", case["sinogram"],
                                iterations=config.iterations, **settings
                            )
                        runtime = time.perf_counter() - begin
                        rows.append({
                            "seed": seed, "projections": projections, "angle_span": span,
                            "noise": noise, "method": method, **evaluate(reference, image),
                            "runtime_s": runtime,
                            "measurement_residual": measurement_residual(
                                "ct", image, case["sinogram"], **settings
                            ),
                        })
                    if model is not None:
                        tensor = torch.from_numpy(case["fbp"][None, None].copy()).to(config.device)
                        with torch.inference_mode():
                            if config.device.startswith("cuda"):
                                torch.cuda.synchronize()
                            begin = time.perf_counter()
                            pred = model(tensor).squeeze().cpu().numpy()
                            if config.device.startswith("cuda"):
                                torch.cuda.synchronize()
                            runtime = time.perf_counter() - begin
                        rows.append({
                            "seed": seed, "projections": projections, "angle_span": span,
                            "noise": noise, "method": "unet", **evaluate(reference, pred),
                            "runtime_s": runtime, "measurement_residual": measurement_residual(
                                "ct", pred, case["sinogram"], **settings
                            ),
                        })
    return pd.DataFrame(rows)


def _percentile_ci(values, n_boot, rng):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return float("nan"), float("nan")
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    estimates = values[draws].mean(axis=1)
    return tuple(float(x) for x in np.percentile(estimates, [2.5, 97.5]))


def summarize_ct(frame: pd.DataFrame, repeats: int = 1000,
                 bootstrap_seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Percentile bootstrap over independent procedural seeds, paired to FBP."""
    rng = np.random.default_rng(bootstrap_seed)
    groups = ["angle_span", "noise", "projections"]
    required = set(groups + ["seed", "method", "psnr", "ssim", "runtime_s", "measurement_residual"])
    if not required.issubset(frame.columns):
        raise ValueError("missing columns for CT summary")
    # No duplicate method+case observations.
    if frame.duplicated(subset=groups + ["seed", "method"]).any():
        raise ValueError("duplicate method/seed/acquisition observations")
    summary = []
    paired = []
    for key, block in frame.groupby(groups, sort=True):
        setting = dict(zip(groups, key))
        baseline = block.loc[block.method == "fbp"].set_index("seed")
        for method, subset in block.groupby("method"):
            subset = subset.sort_values("seed")
            record = {**setting, "method": method, "n": len(subset)}
            for metric in ("psnr", "ssim", "rmse", "runtime_s", "measurement_residual"):
                vals = subset[metric].to_numpy(dtype=float)
                lo, hi = _percentile_ci(vals, repeats, rng)
                record.update({f"{metric}_mean": float(vals.mean()),
                               f"{metric}_ci_low": lo, f"{metric}_ci_high": hi})
            summary.append(record)
            if method != "fbp":
                current = subset.set_index("seed")
                common = baseline.index.intersection(current.index).sort_values()
                if len(common) != len(baseline) or len(common) != len(current):
                    raise ValueError("paired comparison requires identical case seeds")
                for metric in ("psnr", "ssim", "rmse"):
                    differences = (current.loc[common, metric]
                                   - baseline.loc[common, metric]).to_numpy()
                    lo, hi = _percentile_ci(differences, repeats, rng)
                    paired.append({
                        **setting, "method": method, "baseline": "fbp",
                        "metric": metric, "n": len(common),
                        "mean_delta": float(differences.mean()),
                        "ci_low": lo, "ci_high": hi,
                    })
    return pd.DataFrame(summary), pd.DataFrame(paired)


def export_ct(frame: pd.DataFrame, config: CTSweepConfig) -> dict:
    """Save measured CSV, uncertainty intervals, manuscript-style plots and manifest."""
    out = Path(config.output)
    out.mkdir(parents=True, exist_ok=True)
    summary, paired = summarize_ct(
        frame, repeats=config.bootstrap_repeats, bootstrap_seed=config.bootstrap_seed)
    frame.to_csv(out / "cases.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    paired.to_csv(out / "paired_vs_fbp.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (span, noise, method), block in summary.groupby(["angle_span", "noise", "method"]):
        block = block.sort_values("projections")
        ax.plot(block.projections, block.psnr_mean, marker="o",
                label=f"{method} | {span:g}° | noise {noise:g}")
    ax.set(xlabel="CT projection count", ylabel="Mean PSNR (dB)",
           title="CT reconstruction quality by acquisition and method")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "psnr_vs_projections.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, block in frame.groupby("method"):
        ax.scatter(block.runtime_s, block.psnr, label=method, alpha=0.65)
    ax.set(xlabel="Reconstruction runtime (s; excludes acquisition and FBP for U-Net)",
           ylabel="PSNR (dB)", title="Quality vs method-only runtime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "quality_vs_runtime.png", dpi=180)
    plt.close(fig)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {"config": asdict(config), "python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__, "git_commit": commit,
                "case_rows": len(frame), "unique_cases": len(
                    frame[["seed", "projections", "angle_span", "noise"]].drop_duplicates()),
                "cases_sha256": hashlib.sha256((out / "cases.csv").read_bytes()).hexdigest(),
                "ci_method": "percentile bootstrap over held-out procedural case seeds",
                "uncertainty_note": "For fewer than two seeds CI bounds are undefined.",
                "limitations": ("Synthetic affine Shepp-Logan variants only. "
                                "Timing excludes acquisition and U-Net FBP preprocessing; "
                                "no clinical claims or superiority implied.")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Held-out CT projection/noise benchmark")
    parser.add_argument("--output", default="outputs/ct_ladder")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--projections", nargs="+", type=int, default=[30, 60, 90, 180])
    parser.add_argument("--angle-spans", nargs="+", type=float, default=[120, 180])
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0, .01, .03])
    parser.add_argument("--seeds", nargs="+", type=int, default=[300042, 300043, 300044])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=11)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = CTSweepConfig(**{**vars(args),
                             "angle_spans": tuple(args.angle_spans),
                             "noise_levels": tuple(args.noise_levels),
                             "projections": tuple(args.projections),
                             "seeds": tuple(args.seeds)})
    manifest = export_ct(sweep_ct(config), config)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
