"""Read the public LoDoPaB-CT release and benchmark its native CT geometry."""
from __future__ import annotations

import argparse
import json
import math
import time
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .lodopab_progress import collect_slices
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


def _check_shard_layout(path: Path, data, valid_rows: int, shape: tuple[int, int],
                        allow_padding: bool) -> None:
    allowed_rows = {valid_rows, ROWS_PER_SHARD} if allow_padding else {valid_rows}
    if data.ndim != 3 or data.shape[1:] != shape or data.shape[0] not in allowed_rows:
        raise ValueError(f"{path.name} has unexpected data shape {data.shape}")


def _read_shard_row(path: Path, row: int, valid_rows: int, shape: tuple[int, int],
                    allow_padding: bool = False) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as shard:
        data = shard["data"]
        _check_shard_layout(path, data, valid_rows, shape, allow_padding)
        array = np.asarray(data[row], dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{path.name}, row {row}, contains non-finite values")
    return array


def check_data(root: Path, split: str, indices: list[int] | None = None) -> dict:
    """Report data coverage using patient mappings and shard metadata only."""
    import h5py

    selected = _selected_indices(split, indices)
    errors = []
    patients = None
    try:
        ids = load_patient_ids(root)[split]
        patients = len(set(ids[index] for index in selected))
    except (OSError, ValueError) as error:
        errors.append(f"Patient mappings: {error}")

    shards = sorted({index // ROWS_PER_SHARD for index in selected})
    usable = set()
    for number in shards:
        valid_rows = min(ROWS_PER_SHARD, SPLIT_SIZES[split] - number * ROWS_PER_SHARD)
        errors_before = len(errors)
        for kind, shape in (("observation", OBSERVATION_SHAPE), ("ground_truth", IMAGE_SHAPE)):
            path = root / f"{kind}_{split}_{number:03d}.hdf5"
            if not path.is_file():
                errors.append(f"Missing {path.name}")
                continue
            try:
                with h5py.File(path, "r") as shard:
                    data = shard.get("data")
                    if not isinstance(data, h5py.Dataset):
                        raise ValueError("expected a dataset named data")
                    _check_shard_layout(path, data, valid_rows, shape, kind == "ground_truth")
                    if data.dtype.kind != "f":
                        raise ValueError(f"expected floating point data, found {data.dtype}")
            except (OSError, ValueError) as error:
                errors.append(f"{path.name}: {error}")
        if len(errors) == errors_before:
            usable.add(number)

    return {
        "split": split,
        "scope": "full_split" if len(selected) == SPLIT_SIZES[split] else "subset",
        "requested_slices": len(selected),
        "slices_with_matching_shard_headers": sum(
            index // ROWS_PER_SHARD in usable for index in selected
        ),
        "requested_patients": patients,
        "required_shard_pairs": len(shards),
        "matching_shard_pairs": len(usable),
        "ready": not errors,
        "errors": errors,
    }


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


def _cgls_trace(ray, measurement, budgets):
    """Capture requested iterates from one solve, excluding snapshot time."""
    from odl.solvers import conjugate_gradient_normal

    estimate = ray.domain.zero()
    snapshots = {}
    iteration = 0
    snapshot_s = 0.0
    start = time.perf_counter()

    def capture(current):
        nonlocal iteration, snapshot_s
        iteration += 1
        if iteration in budgets:
            now = time.perf_counter()
            snapshots[iteration] = (np.asarray(current).copy(), now - start - snapshot_s)
            snapshot_s += time.perf_counter() - now

    conjugate_gradient_normal(ray, estimate, measurement, niter=budgets[-1], callback=capture)
    elapsed = time.perf_counter() - start - snapshot_s
    return [
        (f"cgls_{budget}", *snapshots.get(budget, (np.asarray(estimate), elapsed)))
        for budget in budgets
    ]


def _paired_patient_deltas(per_patient: pd.DataFrame) -> pd.DataFrame:
    """Estimate equal-patient mean differences using paired patient resampling."""
    rng = np.random.default_rng(11)
    measurements = ["psnr", "ssim", "relative_l2_residual"]
    fbp = per_patient.loc["fbp", measurements]
    draws = rng.integers(len(fbp), size=(2000, len(fbp)))
    rows = []
    for method in per_patient.index.get_level_values("method").unique().drop("fbp"):
        differences = per_patient.loc[method, measurements].reindex(fbp.index) - fbp
        for metric in measurements:
            values = differences[metric].to_numpy()
            bounds = (np.nan, np.nan)
            if len(values) > 1 and np.isfinite(values).all():
                bounds = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
            rows.append({
                "method": method, "baseline": "fbp", "metric": metric,
                "mean_delta": values.mean(), "ci95_low": bounds[0], "ci95_high": bounds[1],
                "n_patients": len(values),
            })
    return pd.DataFrame(rows)


def _patient_mapping_digest(ids: np.ndarray) -> str:
    return sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def load_frozen_selection(path: Path, validation_ids: np.ndarray, backend: str) -> dict:
    """Verify a complete validation selection before evaluating any test images."""
    raw = path.read_bytes()
    record = json.loads(raw)
    first, stop = PATIENT_RANGES["validation"]
    expected = {
        "dataset_doi": DATASET_DOI,
        "split": "validation",
        "scope": "full_split",
        "criterion": "psnr_patient_mean",
        "tie_break": "fewer iterations",
        "n_slices": SPLIT_SIZES["validation"],
        "n_patients": stop - first,
        "backend": backend,
        "patient_mapping_sha256": _patient_mapping_digest(validation_ids),
    }
    if not isinstance(record, dict):
        raise ValueError("Validation selection must be a JSON object")
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"Validation selection has incompatible {field}: expected {value}")
    chosen, candidates = record.get("selected_iterations"), record.get("candidates")
    if (
        type(chosen) is not int or not isinstance(candidates, list) or
        not all(type(budget) is int and budget > 0 for budget in candidates) or
        len(set(candidates)) != len(candidates) or chosen not in candidates
    ):
        raise ValueError("Validation selection has invalid iteration candidates or winner")
    score = record.get("selected_score")
    if type(score) not in (int, float) or not math.isfinite(score):
        raise ValueError("Validation selection must report a finite selected score")
    return {"sha256": sha256(raw).hexdigest(), "record": record}


def _selected_indices(split: str, indices: list[int] | None) -> range | list[int]:
    count = SPLIT_SIZES[split]
    selected = range(count) if indices is None else indices
    if not selected or len(set(selected)) != len(selected) or not all(
        0 <= index < count for index in selected
    ):
        raise ValueError(f"{split} indices must be unique and within [0, {count - 1}]")
    return selected


def benchmark(root: Path, output: Path, split: str = "test",
              indices: list[int] | None = None, cgls_iterations: int = 0,
              backend: str = "astra_cpu", *,
              validation_iterations: list[int] | None = None,
              reference_checkpoint: Path | None = None,
              resume: bool = False,
              frozen_selection: Path | None = None) -> pd.DataFrame:
    """Compare native FBP and optional zero-initialized CGLS on matched slices."""
    if cgls_iterations < 0:
        raise ValueError("CGLS iterations must be nonnegative")
    budgets = sorted(validation_iterations or [])
    if validation_iterations is not None:
        if split != "validation" or cgls_iterations:
            raise ValueError("The iteration sweep requires the validation split and its own budgets")
        if not budgets or budgets[0] < 1 or len(set(budgets)) != len(budgets):
            raise ValueError("Validation iteration budgets must be positive and unique")
    count = SPLIT_SIZES[split]
    selected = _selected_indices(split, indices)
    if frozen_selection is not None and (
        split != "test" or len(selected) != count or cgls_iterations or validation_iterations is not None
    ):
        raise ValueError("A frozen selection requires the complete test split and no other iteration setting")
    ids_by_split = load_patient_ids(root)
    ids = ids_by_split[split]
    selection_provenance = None
    if frozen_selection is not None:
        if frozen_selection.resolve() == (output / "selection.json").resolve():
            raise ValueError("Use a separate output directory for the held-out run")
        selection_provenance = load_frozen_selection(
            frozen_selection, ids_by_split["validation"], backend
        )
        cgls_iterations = selection_provenance["record"]["selected_iterations"]

    import odl

    ray = make_ray_transform(backend)
    fbp = odl.tomo.fbp_op(ray, filter_type="Ram-Lak", frequency_scaling=1.0)

    methods = {"fbp": fbp}
    reference_metadata = None
    if reference_checkpoint is not None:
        from .lodopab_reference import load_reference

        reference_methods, reference_metadata = load_reference(ray, reference_checkpoint)
        methods.update(reference_methods)

    manifest = {
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
        "validation_iterations": budgets,
        "reference_reconstructors": reference_metadata,
        "paired_bootstrap": {"unit": "patient", "resamples": 2000, "seed": 11},
        "data_range": "per-reference max minus min",
        "ssim_window": 7,
        "timing": "reconstruction only, includes first-call overhead",
    }

    complete_split = len(selected) == count
    manifest["scope"] = "full_split" if complete_split else "subset"
    manifest["n_slices"] = len(selected)
    manifest["n_patients"] = len(set(ids[index] for index in selected))
    manifest["patient_mapping_sha256"] = _patient_mapping_digest(ids)
    if selection_provenance is not None:
        manifest["validation_selection"] = selection_provenance
    checkpoint = output / "progress.sqlite"
    if resume:
        output.mkdir(parents=True, exist_ok=True)
        if not checkpoint.exists() and (output / "manifest.json").exists():
            raise ValueError("Existing output has no checkpoint. Use a new output directory.")
    elif checkpoint.exists():
        raise ValueError("This output has a checkpoint. Use --resume or a new output directory.")

    def evaluate_slice(index):
        rows = []
        observation, reference = load_pair(root, split, index)
        measurement = ray.range.element(observation)
        data_range = float(np.ptp(reference))
        predictions = []
        for name, reconstruct in methods.items():
            start = time.perf_counter()
            prediction = np.asarray(reconstruct(measurement))
            predictions.append((name, prediction, time.perf_counter() - start))
        if budgets:
            predictions.extend(_cgls_trace(ray, measurement, budgets))
        elif cgls_iterations:
            estimate = ray.domain.zero()
            start = time.perf_counter()
            odl.solvers.conjugate_gradient_normal(
                ray, estimate, measurement, niter=cgls_iterations
            )
            predictions.append(("cgls", np.asarray(estimate), time.perf_counter() - start))
        for name, prediction, elapsed in predictions:
            residual = np.linalg.norm(np.asarray(ray(prediction)) - observation)
            rows.append({
                "split": split, "index": index, "patient_id": int(ids[index]), "method": name,
                **evaluate(reference, prediction, data_range=data_range),
                "relative_l2_residual": float(residual / max(np.linalg.norm(observation), 1e-12)),
                "reconstruction_s": elapsed,
            })

        return rows

    configuration = {**manifest, "dataset_root": str(root.resolve()), "checkpoint_schema": 1}
    rows = collect_slices(selected, evaluate_slice, checkpoint if resume else None, configuration)
    frame = pd.DataFrame(rows)
    metric_names = ["psnr", "ssim", "mae", "rmse", "relative_l2_residual", "reconstruction_s"]
    per_patient = frame.groupby(["method", "patient_id"])[metric_names].mean()
    slice_means = frame.groupby("method")[metric_names].mean().add_suffix("_slice_mean")
    patient_means = per_patient.groupby("method").mean().add_suffix("_patient_mean")
    summary = slice_means.join(patient_means)
    summary["n_slices"] = frame.groupby("method")["index"].size()
    summary["n_patients"] = per_patient.groupby("method").size()

    output.mkdir(parents=True, exist_ok=True)
    if not budgets:
        (output / "selection.json").unlink(missing_ok=True)
    if not (budgets or cgls_iterations or reference_checkpoint is not None):
        (output / "paired_vs_fbp.csv").unlink(missing_ok=True)
    selection = None
    if budgets:
        criterion = "psnr_patient_mean"
        chosen = max(budgets, key=lambda budget: (summary.loc[f"cgls_{budget}", criterion], -budget))
        selection = {
            "dataset_doi": DATASET_DOI,
            "backend": backend,
            "patient_mapping_sha256": manifest["patient_mapping_sha256"],
            "split": split,
            "scope": manifest["scope"],
            "criterion": criterion,
            "tie_break": "fewer iterations",
            "candidates": budgets,
            "selected_iterations": chosen,
            "selected_score": float(summary.loc[f"cgls_{chosen}", criterion]),
            "n_slices": len(selected),
            "n_patients": int(summary.loc[f"cgls_{chosen}", "n_patients"]),
        }
    if budgets or cgls_iterations or reference_checkpoint is not None:
        _paired_patient_deltas(per_patient).to_csv(output / "paired_vs_fbp.csv", index=False)
    frame.to_csv(output / "per_slice.csv", index=False)
    per_patient.to_csv(output / "per_patient.csv")
    summary.to_csv(output / "summary.csv")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if selection is not None:
        temporary = output / "selection.json.tmp"
        temporary.write_text(json.dumps(selection, indent=2) + "\n")
        temporary.replace(output / "selection.json")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Extracted LoDoPaB-CT files")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=SPLIT_SIZES, default="test")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--indices", nargs="+", type=int, help="Zero-based official slice indices")
    selection.add_argument("--all", action="store_true", help="Evaluate the entire split")
    iteration_options = parser.add_mutually_exclusive_group()
    iteration_options.add_argument("--cgls-iterations", type=int, default=0, help="Zero runs FBP only")
    iteration_options.add_argument(
        "--validation-iterations", type=int, nargs="+",
        help="Sweep CGLS budgets on the validation split and select by patient-weighted PSNR",
    )
    iteration_options.add_argument(
        "--frozen-selection", type=Path,
        help="Use a full-validation selection.json for the complete held-out test split",
    )
    parser.add_argument("--backend", choices=("astra_cpu", "astra_cuda"), default="astra_cpu")
    parser.add_argument(
        "--reference-checkpoint", type=Path,
        help="Add DIVαℓ tuned Hann FBP and the published LoDoPaB FBP+U-Net",
    )
    parser.add_argument("--resume", action="store_true",
                        help="Save completed slices and reuse them on an identical rerun")
    parser.add_argument(
        "--check-data", action="store_true",
        help="Report patient mapping and HDF5 shard readiness without reconstructing",
    )
    args = parser.parse_args()
    if args.check_data:
        readiness = check_data(args.root, args.split, args.indices)
        print(json.dumps(readiness, indent=2))
        raise SystemExit(0 if readiness["ready"] else 1)
    if args.output is None:
        parser.error("--output is required for a benchmark run")
    frame = benchmark(
        args.root, args.output, args.split, args.indices, args.cgls_iterations, args.backend,
        validation_iterations=args.validation_iterations,
        reference_checkpoint=args.reference_checkpoint,
        resume=args.resume,
        frozen_selection=args.frozen_selection,
    )
    print(frame.groupby("method")[["psnr", "ssim", "reconstruction_s"]].mean())


if __name__ == "__main__":
    main()
