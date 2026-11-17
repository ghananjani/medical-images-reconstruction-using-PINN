"""Generate real benchmark artifacts (not illustrative mock data)."""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .pipeline import experiment


def export(result: dict, destination: str | Path) -> None:
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=True)
    items = [("reference", result["reference"]), ("degraded", result["degraded"])]
    items += [(name, r["image"]) for name, r in result["reconstructions"].items()]
    rows = []
    for name, value in items:
        plt.imsave(path / f"{name}.png", value, cmap="gray", vmin=0, vmax=1)
    fig, axes = plt.subplots(2, len(items), figsize=(4 * len(items), 7))
    for i, (name, image) in enumerate(items):
        axes[0, i].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(name)
        axes[1, i].imshow(np.abs(image - result["reference"]), cmap="magma", vmin=0, vmax=1)
        axes[1, i].set_title("Absolute error")
        for row in axes[:, i]:
            row.axis("off")
    fig.tight_layout()
    fig.savefig(path / "comparison.png", dpi=160)
    plt.close(fig)
    for method, info in result["reconstructions"].items():
        rows.append({"method": method, **info["metrics"], "runtime_s": info["runtime_s"],
                     "measurement_residual": info["measurement_residual"]})
    pd.DataFrame(rows).to_csv(path / "metrics.csv", index=False)
    metadata = {"modality": result["modality"], "parameters": result["parameters"],
                "python": platform.python_version(), "numpy": np.__version__,
                "methods": list(result["reconstructions"])}
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=("ct", "mri", "xray"), default="ct")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/ct")
    parser.add_argument("--projections", type=int, default=60)
    parser.add_argument("--acceleration", type=int, default=4, choices=(2, 4, 8))
    parser.add_argument("--noise", type=float, default=0.01)
    args = parser.parse_args()
    export(experiment(args.modality, args.size, args.seed, args.projections,
                      args.acceleration, noise=args.noise), args.output)
    print(f"Wrote measured benchmark artifacts to {args.output}")


if __name__ == "__main__":
    main()
