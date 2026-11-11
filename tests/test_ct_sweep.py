"""Scientific integrity tests for the CT projection/noise ladder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from medreconlab.ct_sweep import CTSweepConfig, export_ct, summarize_ct, sweep_ct


def test_ct_sweep_uses_matched_measurements_and_seeds(tmp_path):
    cfg = CTSweepConfig(
        output=str(tmp_path), size=32, projections=(16, 24),
        angle_spans=(120., 180.), noise_levels=(0., .01),
        seeds=(300101, 300102), iterations=1, bootstrap_repeats=50,
    )
    cases = sweep_ct(cfg)
    assert len(cases) == 2 * 2 * 2 * 2 * 2
    for _, block in cases.groupby(["projections", "angle_span", "noise", "seed"]):
        assert sorted(block.method.tolist()) == ["fbp", "sart"]
    assert np.isfinite(cases[["psnr", "ssim", "rmse", "runtime_s",
                              "measurement_residual"]].to_numpy()).all()
    manifest = export_ct(cases, cfg)
    assert manifest["unique_cases"] == 16
    for filename in ("cases.csv", "summary.csv", "paired_vs_fbp.csv",
                     "manifest.json", "psnr_vs_projections.png",
                     "quality_vs_runtime.png"):
        assert (tmp_path / filename).is_file()


def test_paired_bootstrap_delta_uses_same_case_seeds():
    rows = []
    for seed, base, delta in [(300101, 12., 1.), (300102, 16., 3.),
                              (300103, 14., 2.)]:
        for method, value in [("fbp", base), ("sart", base + delta)]:
            rows.append({"seed": seed, "angle_span": 180., "noise": 0.,
                         "projections": 30, "method": method, "psnr": value,
                         "ssim": .5, "rmse": .2, "runtime_s": .1,
                         "measurement_residual": .01})
    summary, deltas = summarize_ct(pd.DataFrame(rows), repeats=200,
                                   bootstrap_seed=7)
    assert len(summary) == 2
    psnr_delta = deltas.query("metric == 'psnr'").iloc[0]
    assert psnr_delta.mean_delta == pytest.approx(2.)
    assert psnr_delta.ci_low <= 2 <= psnr_delta.ci_high


def test_sweep_rejects_invalid_and_unmatched_configurations():
    with pytest.raises(ValueError, match="distinct"):
        CTSweepConfig(seeds=(300001, 300001)).validate()
    with pytest.raises(ValueError, match="single matching"):
        sweep_ct(CTSweepConfig(size=32, projections=(16, 24),
                              checkpoint="outputs/nonexistent/best.pt"))
    with pytest.raises(ValueError, match="duplicate"):
        rows = pd.DataFrame([{
            "seed": 300001, "angle_span": 180., "noise": 0.,
            "projections": 30, "method": "fbp", "psnr": 20.,
            "ssim": .5, "rmse": .1, "runtime_s": .1,
            "measurement_residual": .1}])
        summarize_ct(pd.concat([rows, rows], ignore_index=True))


def test_sweep_one_seed_has_undefined_intervals():
    rows = pd.DataFrame([
        {"seed": 300001, "angle_span": 180., "noise": 0.,
         "projections": 30, "method": "fbp", "psnr": 20.,
         "ssim": .5, "rmse": .1, "runtime_s": .1, "measurement_residual": .1}
    ])
    summary, paired = summarize_ct(rows, repeats=10)
    assert np.isnan(summary.iloc[0]["psnr_ci_low"])
    assert paired.empty
