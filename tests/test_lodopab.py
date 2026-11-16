import json

import numpy as np
import pytest

from medreconlab import lodopab

h5py = pytest.importorskip("h5py")


def test_patient_split_metadata(tmp_path):
    for split, count in lodopab.SPLIT_SIZES.items():
        ids = np.arange(*lodopab.PATIENT_RANGES[split])
        np.savetxt(tmp_path / f"patient_ids_rand_{split}.csv", np.resize(ids, count), fmt="%d")
    assert len(np.unique(lodopab.load_patient_ids(tmp_path)["test"])) == 60

    path = tmp_path / "patient_ids_rand_test.csv"
    ids = np.loadtxt(path, dtype=int)
    ids[0] = 0
    np.savetxt(path, ids, fmt="%d")
    with pytest.raises(ValueError, match="official LoDoPaB-CT split"):
        lodopab.load_patient_ids(tmp_path)


def test_padded_final_shard_uses_official_indices(tmp_path):
    for kind, shape, rows, value in (
        ("observation", lodopab.OBSERVATION_SHAPE, 97, 0.03125),
        ("ground_truth", lodopab.IMAGE_SHAPE, 128, 0.25),
    ):
        with h5py.File(tmp_path / f"{kind}_test_027.hdf5", "w") as f:
            data = f.create_dataset("data", shape=(rows, *shape), dtype="f4", chunks=(1, *shape))
            data[96] = value
    obs, ref = lodopab.load_pair(tmp_path, "test", 3552)
    assert obs.shape == (1000, 513)
    assert ref.shape == (362, 362)
    assert np.all(obs == 0.03125)
    assert np.all(ref == 0.25)
    with pytest.raises(ValueError, match="3552"):
        lodopab.load_pair(tmp_path, "test", 3553)


def test_native_geometry_and_benchmark(tmp_path, monkeypatch):
    pytest.importorskip("odl")
    pytest.importorskip("astra")
    ray = lodopab.make_ray_transform()
    np.testing.assert_allclose(ray.geometry.det_partition.max_pt, [0.13 * np.sqrt(2)])
    angles = ray.geometry.motion_partition.grid.coord_vectors[0]
    np.testing.assert_allclose(angles[[0, -1]], np.pi * np.array([0.0005, 0.9995]))
    np.testing.assert_allclose(ray.domain.min_pt, [-0.13, -0.13])

    reference = np.zeros(lodopab.IMAGE_SHAPE, dtype=np.float32)
    reference[145:215, 185:240] = 0.5
    obs = np.asarray(ray(reference))
    patient_ids = np.full(lodopab.SPLIT_SIZES["test"], 692)
    monkeypatch.setattr(lodopab, "load_patient_ids", lambda _: {"test": patient_ids})
    monkeypatch.setattr(lodopab, "load_pair", lambda *_: (obs, reference))
    frame = lodopab.benchmark(tmp_path, tmp_path / "output", indices=[0], cgls_iterations=1)
    assert frame["method"].tolist() == ["fbp", "cgls"]
    assert frame["patient_id"].tolist() == [692, 692]
    np.testing.assert_allclose(frame["psnr"], 10 * np.log10(0.25 / frame["mse"]))
    assert frame.loc[0, "psnr"] > 25
    assert frame.loc[0, "relative_l2_residual"] < 0.1
    assert (tmp_path / "output" / "per_patient.csv").is_file()
    manifest = json.loads((tmp_path / "output" / "manifest.json").read_text())
    assert manifest["indices"] == [0]
    assert manifest["backend"] == "astra_cpu"


def test_validation_sweep_and_patient_pairing(tmp_path, monkeypatch):
    import pandas as pd

    pytest.importorskip("odl")
    pytest.importorskip("astra")
    ray = lodopab.make_ray_transform()
    reference = np.zeros(lodopab.IMAGE_SHAPE, dtype=np.float32)
    reference[145:215, 185:240] = 0.5
    observation = np.asarray(ray(reference))
    monkeypatch.setattr(lodopab, "load_patient_ids", lambda _: {"validation": [632, 633]})
    monkeypatch.setattr(lodopab, "load_pair", lambda *_: (observation, reference))

    with pytest.raises(ValueError, match="requires the validation split"):
        lodopab.benchmark(tmp_path, tmp_path, indices=[0], validation_iterations=[1, 2])

    output = tmp_path / "sweep"
    frame = lodopab.benchmark(
        tmp_path, output, split="validation", indices=[0, 1], validation_iterations=[2, 1],
    )
    assert list(frame["method"].unique()) == ["fbp", "cgls_1", "cgls_2"]
    fixed = lodopab.benchmark(
        tmp_path, tmp_path / "fixed", split="validation", indices=[0], cgls_iterations=2,
    )
    assert frame.loc[2, "psnr"] == pytest.approx(fixed.loc[1, "psnr"])
    selection = json.loads((output / "selection.json").read_text())
    candidate_scores = frame[frame["method"] != "fbp"].groupby("method").psnr.mean()
    assert selection["selected_iterations"] == int(candidate_scores.idxmax().split("_")[1])
    assert selection["n_patients"] == 2
    paired = pd.read_csv(output / "paired_vs_fbp.csv")
    assert (paired["n_patients"] == 2).all()
    np.testing.assert_allclose(paired["mean_delta"], paired["ci95_low"])
    np.testing.assert_allclose(paired["mean_delta"], paired["ci95_high"])


def test_patient_bootstrap_weights_each_patient_equally():
    import pandas as pd

    metrics = ["psnr", "ssim", "relative_l2_residual"]
    patients = pd.DataFrame(
        [[1, 1, 1], [8, 8, 8], [4, 4, 4], [7, 7, 7]],
        index=pd.MultiIndex.from_tuples(
            [("fbp", 1), ("fbp", 2), ("cgls", 1), ("cgls", 2)],
            names=["method", "patient_id"],
        ),
        columns=metrics,
    )
    result = lodopab._paired_patient_deltas(patients)
    np.testing.assert_allclose(result["mean_delta"], 1)
    np.testing.assert_allclose(result["ci95_low"], -1)
    np.testing.assert_allclose(result["ci95_high"], 3)


def test_reference_checkpoint_rejects_unverified_weights(tmp_path):
    from medreconlab.lodopab_reference import load_reference

    checkpoint = tmp_path / "reference.pt"
    checkpoint.write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_reference(None, checkpoint)


def test_reference_methods_enter_matched_benchmark(tmp_path, monkeypatch):
    from medreconlab import lodopab_reference

    pytest.importorskip("odl")
    pytest.importorskip("astra")
    ray = lodopab.make_ray_transform()
    reference = np.zeros(lodopab.IMAGE_SHAPE, dtype=np.float32)
    reference[145:215, 185:240] = 0.5
    obs = np.asarray(ray(reference))
    monkeypatch.setattr(lodopab, "load_patient_ids", lambda _: {"test": [692]})
    monkeypatch.setattr(lodopab, "load_pair", lambda *_: (obs, reference))

    def reconstruct(measurement):
        np.testing.assert_array_equal(np.asarray(measurement), obs)
        return reference * 0.9

    metadata = {"weights_sha256": "test fixture"}
    monkeypatch.setattr(lodopab_reference, "load_reference", lambda *_: (
        {"fbp_reference": reconstruct, "fbp_unet_reference": reconstruct}, metadata
    ))
    frame = lodopab.benchmark(
        tmp_path, tmp_path, indices=[0], reference_checkpoint=tmp_path / "reference.pt",
    )
    assert frame["method"].tolist() == ["fbp", "fbp_reference", "fbp_unet_reference"]
    assert (frame["patient_id"] == 692).all()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["reference_reconstructors"] == metadata
    assert (tmp_path / "paired_vs_fbp.csv").is_file()
