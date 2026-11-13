"""Read the public LoDoPaB-CT release and benchmark its native CT geometry."""
from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import evaluate

DATASET_DOI = "10.5281/zenodo.3384092"
IMAGE_SHAPE = (362, 362)
OBSERVATION_SHAPE = (1000, 513)
HALF_FIELD_M = 0.13
ROWS_PER_SHARD = 128
SPLIT_SIZES = {"train": 35820, "validation": 3522, "test": 3553}
PATIENT_RANGES = {"train": (0, 632), "validation": (632, 692), "test": (692, 752)}


def load_patient_ids(root: Path) -> dict[str, np.ndarray]:
    """Validate the published row counts and disjoint absolute patient ID ranges."""
    ids_by_split = {}
    for split, count in SPLIT_SIZES.items():
        path = root / f"patient_ids_rand_{split}.csv"
        ids = np.loadtxt(path, dtype=int, delimiter=",", ndmin=1)
        first, stop = PATIENT_RANGES[split]
        if ids.shape != (count,) or not np.array_equal(np.unique(ids), np.arange(first, stop)):
            raise ValueError(f"{path.name} does not match the official LoDoPaB-CT split")
        ids_by_split[split] = ids
    return ids_by_split


def _read_shard_row(path: Path, row: int, valid_rows: int, shape: tuple[int, int],
                    allow_padding: bool = False) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as shard:
        data = shard["data"]
        allowed_rows = {valid_rows, ROWS_PER_SHARD} if allow_padding else {valid_rows}
        if data.shape[1:] != shape or data.shape[0] not in allowed_rows:
            raise ValueError(f"{path.name} has unexpected data shape {data.shape}")
        array = np.asarray(data[row], dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{path.name}, row {row}, contains non-finite values")
    return array


def load_pair(root: Path, split: str, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Load a native, normalized post-log sinogram and its reference slice."""
    count = SPLIT_SIZES[split]
    if not 0 <= index < count:
        raise ValueError(f"{split} index must be between 0 and {count - 1}")
    shard, row = divmod(index, ROWS_PER_SHARD)
    valid_rows = min(ROWS_PER_SHARD, count - shard * ROWS_PER_SHARD)
    observation = _read_shard_row(
        root / f"observation_{split}_{shard:03d}.hdf5", row, valid_rows, OBSERVATION_SHAPE
    )
    reference = _read_shard_row(
        root / f"ground_truth_{split}_{shard:03d}.hdf5", row, valid_rows, IMAGE_SHAPE,
        allow_padding=True,
    )
    if reference.min() < 0 or reference.max() > 1:
        raise ValueError(f"{split} index {index} has a reference outside [0, 1]")
    return observation, reference


def make_ray_transform(backend: str = "astra_cpu"):
    """Use the same discretization and parallel-beam geometry as DIVαℓ."""
    import odl

    space = odl.uniform_discr(
        [-HALF_FIELD_M] * 2, [HALF_FIELD_M] * 2, IMAGE_SHAPE, dtype="float32"
    )
    geometry = odl.tomo.parallel_beam_geometry(
        space, num_angles=OBSERVATION_SHAPE[0], det_shape=(OBSERVATION_SHAPE[1],)
    )
    return odl.tomo.RayTransform(space, geometry, impl=backend)


def benchmark(root: Path, output: Path, split: str = "test",
              indices: list[int] | None = None, cgls_iterations: int = 0,
              backend: str = "astra_cpu") -> pd.DataFrame:
    """Compare native FBP and optional zero-initialized CGLS on matched slices."""
    import odl

    if cgls_iterations < 0:
        raise ValueError("CGLS iterations must be nonnegative")
    count = SPLIT_SIZES[split]
    selected = range(count) if indices is None else indices
    if not selected or len(set(selected)) != len(selected) or not all(
        0 <= index < count for index in selected
    ):
        raise ValueError(f"{split} indices must be unique and within [0, {count - 1}]")
    ids = load_patient_ids(root)[split]
    ray = make_ray_transform(backend)
    fbp = odl.tomo.fbp_op(ray, filter_type="Ram-Lak", frequency_scaling=1.0)

    def run_cgls(measurement):
        estimate = ray.domain.zero()
        odl.solvers.conjugate_gradient_normal(ray, estimate, measurement, niter=cgls_iterations)
        return estimate

    methods = {"fbp": fbp}
    if cgls_iterations:
        methods["cgls"] = run_cgls

    rows = []
    for index in selected:
        observation, reference = load_pair(root, split, index)
        measurement = ray.range.element(observation)
        data_range = float(np.ptp(reference))
        for name, reconstruct in methods.items():
            start = time.perf_counter()
            prediction = np.asarray(reconstruct(measurement))
            elapsed = time.perf_counter() - start
            residual = np.linalg.norm(np.asarray(ray(prediction)) - observation)
            rows.append({
                "split": split, "index": index, "patient_id": int(ids[index]), "method": name,
                **evaluate(reference, prediction, data_range=data_range),
                "relative_l2_residual": float(residual / max(np.linalg.norm(observation), 1e-12)),
                "reconstruction_s": elapsed,
            })

    frame = pd.DataFrame(rows)
    metric_names = ["psnr", "ssim", "mae", "rmse", "relative_l2_residual", "reconstruction_s"]
    per_patient = frame.groupby(["method", "patient_id"])[metric_names].mean()
    slice_means = frame.groupby("method")[metric_names].mean().add_suffix("_slice_mean")
    patient_means = per_patient.groupby("method").mean().add_suffix("_patient_mean")
    summary = slice_means.join(patient_means)
    summary["n_slices"] = frame.groupby("method")["index"].size()
    summary["n_patients"] = per_patient.groupby("method").size()

    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_slice.csv", index=False)
    per_patient.to_csv(output / "per_patient.csv")
    summary.to_csv(output / "summary.csv")
    (output / "manifest.json").write_text(json.dumps({
        "dataset_doi": DATASET_DOI,
        "split": split,
        "indices": list(selected),
        "image_shape": IMAGE_SHAPE,
        "sinogram_shape_angle_detector": OBSERVATION_SHAPE,
        "image_bounds_m": [-HALF_FIELD_M, HALF_FIELD_M],
        "detector_bounds_m": [float(ray.geometry.det_partition.min_pt[0]),
                              float(ray.geometry.det_partition.max_pt[0])],
        "angle_bounds_rad": [0, float(np.pi)],
        "backend": backend,
        "versions": {name: version(name) for name in (
            "medreconlab", "odl", "astra-toolbox", "numpy", "scikit-image", "h5py"
        )},
        "fbp_filter": "Ram-Lak",
        "fbp_frequency_scaling": 1.0,
        "cgls_iterations": cgls_iterations,
        "data_range": "per-reference max minus min",
        "ssim_window": 7,
        "timing": "reconstruction only, includes first-call overhead",
    }, indent=2) + "\n")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Extracted LoDoPaB-CT files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=SPLIT_SIZES, default="test")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--indices", nargs="+", type=int, help="Zero-based official slice indices")
    selection.add_argument("--all", action="store_true", help="Evaluate the entire split")
    parser.add_argument("--cgls-iterations", type=int, default=0, help="Zero runs FBP only")
    parser.add_argument("--backend", choices=("astra_cpu", "astra_cuda"), default="astra_cpu")
    args = parser.parse_args()
    frame = benchmark(args.root, args.output, args.split, args.indices, args.cgls_iterations,
                      args.backend)
    print(frame.groupby("method")[["psnr", "ssim", "reconstruction_s"]].mean())


if __name__ == "__main__":
    main()
