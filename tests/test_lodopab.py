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
