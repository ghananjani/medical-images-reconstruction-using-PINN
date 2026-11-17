# MedReconLab

**Research-only, non-diagnostic** medical image reconstruction benchmarking platform. Synthetic demos for CT sparse-view tomography, Cartesian MRI undersampling, and 2D X-ray restoration. Never use outputs to make clinical decisions.

## Implemented in this milestone

- **CT:** Shepp-Logan phantom, Radon measurements, filtered backprojection (FBP), SART iterations on a circular reconstruction grid, limited-angle geometry, additive Gaussian sinogram noise; optional differentiable PyTorch Radon forward model with projection-consistent neural-field-style pixel optimization.
- **MRI:** complex-valued 2D Cartesian FFT measurements, seeded equispaced/random undersampling, zero-filled inverse FFT and iterative wavelet-free TV-style proximal reconstruction, exact acquired complex k-space projection before magnitude display; optional PyTorch data-consistency optimization.
- **X-ray:** explicit blur/noise degradation, Wiener deconvolution, Richardson-Lucy restoration.
- Reference-conditioned PSNR, SSIM, MSE, RMSE, MAE, NRMSE; measurement residuals; wall-clock runtime; reproducible CSV/PNG/JSON demo exports.
- FastAPI and Streamlit exploratory interfaces; CPU smoke-test workflow.
- **Not yet implemented:** PDE-based PINNs, diffusion/Transformer architectures, MLflow model registry, persistent background job queues, production-grade upload/DICOM support, multicoil MRI/CG-SENSE, downloadable research PDF, patient-level clinical dataset tooling. These are future phases, not working features.

## Quick start

```bash
git clone https://github.com/ghananjani/medical-images-reconstruction-using-PINN.git
cd medical-images-reconstruction-using-PINN
git checkout feat/medical-image-reconstruction
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
medrecon-demo --modality ct --size 64 --output outputs/ct
medrecon-demo --modality mri --size 64 --output outputs/mri
medrecon-demo --modality xray --size 64 --output outputs/xray
streamlit run dashboard/app.py
uvicorn medreconlab.api:app --reload
```

Optional torch physics optimization: `pip install -e ".[physics]"`.

## Scientific methodology

CT forward operator uses `skimage.transform.radon(circle=True)`; FBP uses `iradon(circle=True)`; SART uses `iradon_sart`. This shared geometry keeps detector and image widths equal, without rescaling the SART output. The circular field-of-view assumption is important: image content outside the inscribed disk is not modeled. Angles are degrees in [0, 180). Its PyTorch differentiable forward approximation uses affine grid sampling and a line integral sum, so the numerical model is not identical to scikit-image Radon; train and evaluate consistency with the same operator.

MRI FFT is orthonormal with centered k-space (`fftshift(fft2(ifftshift(x), norm="ortho"))`). An acquired sample mask operates on complex k-space; zero-fill and iterative methods reuse exactly the same samples. Physics residuals are evaluated using the complex-valued reconstruction **before** magnitude conversion and clipping; taking the magnitude first would destroy phase and invalidate a claim of exact measured-data consistency. Iterative MRI prior is Gaussian smoothing followed by hard complex k-space consistency: a heuristic, **not** formal compressed sensing. This first release supports single-coil only.

X-ray is 2D restoration of a radiograph under Gaussian blur and Gaussian noise. It is **not** 3D reconstruction from a single image.

Reference image metrics require ground truth and are not applied to clinical data without ground truth. Intensities are clipped to [0,1] for display/metrics and SSIM data_range=1. Runtime is process wall time and should not be treated as a hardware-normalized speed benchmark. Synthetic images are not real clinical outcomes.

## Reproducibility and scope

Each demo saves input, reference, reconstructions, residual maps, metrics CSV, a summary figure and metadata JSON with seed, acquisition parameters and package versions. Use identical measurements for method comparisons; run multiple seeds for statistical analyses. No restricted datasets are downloaded.

Optional datasets for future work: fastMRI (follow NYU access and use restrictions), NIH ChestX-ray14 and CheXpert (follow each dataset's license and agreements). Deidentify DICOM metadata and split by patient IDs when available. Do not add patient datasets to this repo.

API docs: `http://127.0.0.1:8000/docs`. Endpoints include health, modalities, methods, datasets, reconstruct, compare, experiments and reports. Current jobs are synchronous/in-memory; cancellation and persistence are **not implemented**.

## Physics optimizer checks

The optional functions `ct_optimize` and `mri_optimize` live in `medreconlab.physics`. They are physics-informed image-parameter optimization methods, **not** PDE-based physics-informed neural networks or trained U-Nets. CT optimization expects measurements from its own differentiable approximate projection operator, in radians. The scikit-image Radon model is not numerically interchangeable with the torch implementation. The MRI optimizer starts from centered complex k-space and projects acquired coefficients exactly at the end.

To run optional differentiable operator and gradient tests:

```bash
pip install -e '.[dev,physics]'
pytest -q tests/test_core.py -k torch
```

## Development

```bash
ruff check src tests dashboard
pytest -q
docker build -t medreconlab .
docker run --rm -p 8501:8501 medreconlab
```

## Phase 4: supervised CT U-Net training

A **working 2D PyTorch U-Net** learns to post-process filtered-backprojection (FBP)
images into synthetic CT references. This is **supervised reconstruction/restoration**,
not a PDE-based PINN. It uses a two-level encoder-decoder, skip connections,
group normalization and a sigmoid unit-range output.

```bash
pip install -e ".[dev,physics]"
medrecon-train-ct --output outputs/ct_unet --size 64 --projections 30 \
  --train-cases 128 --val-cases 32 --test-cases 32 \
  --epochs 30 --patience 6 --batch-size 8 --channels 16 --seed 42
medrecon-compare-ct --checkpoint outputs/ct_unet/best.pt \
  --output outputs/ct_unet/benchmark --cases 32
```

For an inexpensive local CPU smoke run:

```bash
medrecon-train-ct --output outputs/ct_unet_smoke --size 32 \
  --projections 20 --train-cases 4 --val-cases 2 --test-cases 2 \
  --epochs 2 --batch-size 2 --channels 4 --device cpu
medrecon-compare-ct --checkpoint outputs/ct_unet_smoke/best.pt \
  --output outputs/ct_unet_smoke/benchmark --cases 2
```

The trainer supports shuffled mini-batches, AdamW, L1 training/validation
loss, gradient clipping, automatic CUDA mixed precision when available,
a plateau LR scheduler, early stopping, and atomic best-validation and latest-state
checkpoints. It saves `best.pt`, `last.pt`, `history.csv`, `loss_curve.png`,
and `manifest.json` with seed, split seeds, package versions, and Git SHA.
Do not load untrusted checkpoint files.

The held-out benchmark measures FBP, SART and U-Net **against the same synthetic
sinogram and reference** for each test case, and exports `per_case.csv`,
`summary.csv`, `comparison.png`, and `manifest.json`. Only validation
loss selects the best checkpoint; the test seeds are reserved for the final
comparison. Train seeds start at 100000+seed, validation at 200000+seed, and
test at 300000+seed (for seed <100000). They are disjoint procedural variants
of **one phantom family**, not patient-level or independent anatomical data.

**Important limitations:** These preliminary synthetic results are not clinical
validation, and this U-Net is not assumed to outperform FBP/SART. Runtime
includes model-only inference for U-Net but full iterative reconstruction
for SART, and is not a normalized acquisition-to-reconstruction measurement.
The trained U-Net is opt-in in Streamlit's CT dashboard (not exposed by FastAPI).
Use a trusted **local** checkpoint within `outputs/`, and choose the same image
size and CT projection count used for training. The dashboard computes real
reference metrics and projection residuals. Do not load downloaded or untrusted
checkpoint binaries. Models trained on these simulations may generalize poorly.
CPU CI trains only a tiny smoke model; it is not a publication-grade benchmark.

## Continue an interrupted CT training run

The trainer records `last.pt` every epoch with model, optimizer, scheduler,
automatic-mixed-precision scaler, full loss history and early-stopping state.
Training shuffles examples using a per-epoch deterministic generator, so
continuing from `last.pt` can reproduce a CPU training schedule when the
software and hardware environment is unchanged.

```bash
# First train 3 epochs.
medrecon-train-ct --output outputs/ct_resume --size 32 --projections 20 \
  --train-cases 8 --val-cases 4 --test-cases 4 \
  --epochs 3 --patience 5 --channels 4 --device cpu

# Continue to 6 total epochs with the same data/model/optimizer settings.
medrecon-train-ct --output outputs/ct_resume --size 32 --projections 20 \
  --train-cases 8 --val-cases 4 --test-cases 4 \
  --epochs 6 --patience 5 --channels 4 --device cpu \
  --resume outputs/ct_resume/last.pt
```

Only the total epoch budget, output path, device, and resume path can differ
from the saved configuration. The existing `best.pt` must be in the same
directory. An already early-stopped run cannot be resumed. The test split is
not used to select checkpoints. No training checkpoints or patient data are
committed to the repository.

For interactive comparison, run `streamlit run dashboard/app.py`, select
CT, check **Use locally trained CT U-Net**, then enter your local checkpoint
(e.g. `outputs/ct_resume/best.pt`). The dashboard remains usable without
PyTorch if you leave this option unchecked.

## CT acquisition ladder and statistical analysis

A separate CLI runs controlled CT experiments with the **same measured sinogram
and reference for every method within each case**. Cases vary by projection
count, limited/full angle, noise and held-out procedural seed.

```bash
# Requires PyTorch because synthetic CT case generation uses the CT dataset module.
pip install -e ".[dev,physics]"
medrecon-ct-sweep --output outputs/ct_ladder \
  --size 64 --projections 30 60 90 180 --angle-spans 120 180 \
  --noise-levels 0 .01 .03 --seeds 300042 300043 300044 \
  --iterations 5 --bootstrap-repeats 1000
```

Creates `cases.csv`, `summary.csv`, `paired_vs_fbp.csv`,
`psnr_vs_projections.png`, `quality_vs_runtime.png`, and a manifest with
parameters, versions, Git commit and CSV SHA256. It computes PSNR, SSIM,
RMSE, reconstruction-only runtime and normalized sinogram residuals.
Percentile bootstrap confidence bounds resample held-out **procedural cases**
within each acquisition setting. Method improvement over FBP is calculated
**pairwise on the same case seeds**, rather than comparing unrelated averages.
With fewer than two case seeds, CI columns are undefined. The bootstrap is
descriptive and does not claim patient-level generalization.

**U-Net checkpoint policy:** The full acquisition ladder deliberately compares
FBP and SART without using an out-of-distribution U-Net. To include U-Net,
supply **one** acquisition setting and a trusted checkpoint trained at the
same size, projection count, angular span **and noise level**:

```bash
medrecon-ct-sweep --size 64 --projections 30 --angle-spans 180 \
  --noise-levels .01 --seeds 300042 300043 300044 \
  --checkpoint outputs/ct_unet/best.pt --output outputs/ct_unet/sweep
```

Incompatible checkpoints fail explicitly. Comparing U-Net across a full
projection ladder requires separately trained matching checkpoints (future
work), or an explicitly labelled out-of-distribution experiment. A single
phantom family and small numbers of procedural seeds are not clinical evidence.

**Runtime caution:** FBP and SART times cover reconstruction. U-Net's runtime
covers only network inference, **not** initial FBP preprocessing; therefore
method-only runtime plots are not end-to-end, equal-scope timing claims.

## Optional MLflow experiment tracking

Training and CT sweep scripts support **opt-in MLflow tracking**, disabled by
default. This logs only synthetic research metadata, scalar metrics and
generated CSV/PNG/JSON reports, **not** model checkpoint files, raw medical
images, patient data, or secrets. Use a trusted tracking server when sharing
runs outside your workstation.

```bash
pip install -e ".[dev,physics,tracking]"
# Local SQLite tracking store; no external server or patient dataset is required.
medrecon-train-ct --output outputs/mlflow_ct --size 32 --projections 20 \
  --train-cases 4 --val-cases 2 --test-cases 2 --epochs 2 \
  --channels 4 --device cpu \
  --mlflow-uri "sqlite:///$(pwd)/mlflow.db" \
  --mlflow-experiment MedReconLab-CT

medrecon-ct-sweep --output outputs/mlflow_sweep --size 32 \
  --projections 16 24 --angle-spans 120 180 --noise-levels 0 .01 \
  --seeds 300101 300102 --iterations 1 --bootstrap-repeats 100 \
  --mlflow-uri "sqlite:///$(pwd)/mlflow.db" \
  --mlflow-experiment MedReconLab-CT

mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db"
```

Open the MLflow UI at `http://127.0.0.1:5000`. The training run logs
per-epoch train/validation MAE and learning rate, best validation MAE and
selected epoch. The sweep run logs method-level mean PSNR, SSIM, RMSE,
number of synthetic cases and reproducible generated reports. The CLI
manifest includes the MLflow run ID when tracking is enabled. Failed remote
tracking requests are reported rather than silently ignored. This initial
integration is *experiment tracking only*, not model registry, remote job
scheduling or clinical dataset handling.

## Streamlit MLflow experiment viewer

After recording runs with `--mlflow-uri "sqlite:///$(pwd)/mlflow.db"`, start
`streamlit run dashboard/app.py` and open **Experiments**. Enter
`mlflow.db`, select **Load MLflow runs**, inspect real run parameters,
recorded validation losses, learning-rate curves and aggregate metrics,
or export the run summary as CSV. The viewer is **read-only** and supports
only local SQLite databases inside the current project directory; it does
not connect to arbitrary remote tracking services or download artifacts.
Install the optional tracking dependencies first: `pip install -e ".[tracking]"`.

For a responsible research-paper pathway, read
[`docs/research_protocol.md`](docs/research_protocol.md). The protocol
separates existing synthetic engineering results from external patient-level
validation required for stronger scientific claims.

## LoDoPaB-CT native-geometry baselines

The [pilot evidence](docs/lodopab_pilot_evidence.md) records the measured
validation and test subsets, comparator settings, and confidence intervals.

Download the observation and ground-truth ZIP files for your split and all
three `patient_ids_rand_*.csv` files from [LoDoPaB-CT on Zenodo](https://zenodo.org/records/3384092).
For the validation split, place the five downloads in `data/lodopab/downloads`
and run the commands below. The downloader requests bounded byte ranges, retries transient failures, and
resumes from saved bytes when rerun. Staging verifies the official checksums.

```bash
python scripts/download_lodopab.py --downloads data/lodopab/downloads
python scripts/stage_lodopab.py --downloads data/lodopab/downloads \
  --output data/lodopab --split validation
```

Staging verifies the release checksums and extracts the expected HDF5 files.
Both scripts accept `--split test` to prepare the held-out inputs. Keep space for the ZIP files and
their extracted contents. For a small pilot, the benchmark can also read
manually extracted shard pairs containing the selected indices.

```bash
pip install -e ".[lodopab]"
medrecon-lodopab --root data/lodopab --output outputs/lodopab_pilot \
  --split validation --indices 0 1 2 --cgls-iterations 5
```

The benchmark uses the published 1,000-angle, 513-detector geometry and the
released normalized post-log measurements. FBP uses a full-band Ram-Lak filter.
Optional CGLS starts at zero and uses the same ODL/ASTRA projector and adjoint.
PSNR and SSIM use the reference slice's intensity range. Output includes
`per_slice.csv`, `per_patient.csv`, `summary.csv` with both slice- and
patient-weighted averages, and `manifest.json` with the indices, geometry,
backend and library versions. Reconstruction timings include first-call
overhead and exclude file I/O and metric computation.

Before starting the full validation sweep, check the local input files:

```bash
medrecon-lodopab --root data/lodopab --split validation --all --check-data
```

This command checks the patient mappings and expected HDF5 dataset headers,
lists missing or malformed shards, and exits with status 1 when any are found.
Use the same command with `--split test` to check held-out inputs. The check
does not read image arrays, so it does not replace per-slice validation during
reconstruction.

Choose a CGLS iteration budget by sweeping the official validation split:

```bash
medrecon-lodopab --root data/lodopab --output outputs/lodopab_validation \
  --split validation --all --validation-iterations 1 2 3 5 8 12 16 24 32 --resume
```

With `--resume`, each completed slice is committed to a local
`progress.sqlite` file. After an interruption, repeat the same command.
A change to the selected indices, settings, software versions, or patient
mapping requires a new output directory. Keep the extracted source data
unchanged between runs. The final manifest and selection label coverage as
`full_split` or `subset`. Only a full official validation selection should be
used to freeze the paper's CGLS setting.

The sweep records intermediate reconstructions from a single CGLS solve.
`selection.json` records the budget with the highest patient-weighted mean
PSNR, using fewer iterations to break exact ties. `validation_sensitivity.csv` shows
which budget wins when each validation patient is left out in turn. This
diagnostic uses validation data and leaves the main selection rule unchanged.
Comparative runs export
`paired_vs_fbp.csv` with mean differences and 95% percentile bootstrap intervals
over paired patient averages. Use the completed validation selection directly
for final offline evaluation:

```bash
medrecon-lodopab --root data/lodopab --output outputs/lodopab_test \
  --split test --all --frozen-selection outputs/lodopab_validation/selection.json \
  --reference-checkpoint data/lodopab/reference.pt --resume
```

The frozen-selection option checks full validation coverage, the patient mapping,
backend and chosen candidate before reading any test images. The test manifest
records the selection and its SHA-256 digest. Use the same backend for validation
and test. Pilot runs can still set a fixed `--cgls-iterations` directly.
This workflow covers full-angle low-dose data. Sparse-view experiments require a separate
angular selection protocol and a corresponding forward operator.

### Published LoDoPaB-CT reference comparators

The optional reference path adds DIVαℓ's tuned Hann FBP and pretrained
FBP+U-Net. In an environment with the `lodopab` and `lodopab-reference`
extras available, download the [released checkpoint](https://media.githubusercontent.com/media/jleuschn/supp.dival/6117cabc55d223f5b62ea43d4a40225270fb6756/reference_params/lodopab/lodopab_fbpunet.pt)
to a local data directory and pass its path using `--reference-checkpoint`.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 medrecon-lodopab \
  --root data/lodopab --output outputs/lodopab_reference_pilot --split test \
  --indices 1664 1680 1696 1712 1728 1744 1760 1776 --cgls-iterations 8 \
  --reference-checkpoint data/lodopab/reference.pt
```

The command verifies the checkpoint SHA-256 and uses the published Hann
frequency scaling for each reference method. The benchmark manifest records
the weights revision, digest, filters and dependency versions. U-Net inference
runs on CPU and its timing includes its own FBP preprocessing.
`paired_vs_hann_fbp.csv` reports paired patient-level differences and intervals
against the published tuned Hann baseline, alongside the existing
`paired_vs_fbp.csv` comparisons against full-band Ram-Lak. Use
`--backend astra_cpu` for CPU-only latency comparisons.
