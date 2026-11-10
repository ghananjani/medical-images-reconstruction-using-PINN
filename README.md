# MedReconLab

**Research-only, non-diagnostic** medical image reconstruction benchmarking platform. Synthetic demos for CT sparse-view tomography, Cartesian MRI undersampling, and 2D X-ray restoration. Never use outputs to make clinical decisions.

## Implemented in this milestone

- **CT:** Shepp-Logan phantom, Radon measurements, filtered backprojection (FBP), SART iterations, limited-angle geometry, additive Gaussian sinogram noise; optional differentiable PyTorch Radon forward model with projection-consistent neural-field-style pixel optimization.
- **MRI:** complex-valued 2D Cartesian FFT measurements, seeded equispaced/random undersampling, zero-filled inverse FFT and iterative wavelet-free TV-style proximal reconstruction, exact acquired k-space projection; optional PyTorch data-consistency optimization.
- **X-ray:** explicit blur/noise degradation, Wiener deconvolution, Richardson-Lucy restoration.
- Reference-conditioned PSNR, SSIM, MSE, RMSE, MAE, NRMSE; measurement residuals; wall-clock runtime; reproducible CSV/PNG/JSON demo exports.
- FastAPI and Streamlit exploratory interfaces; CPU smoke-test workflow.
- **Not yet implemented:** supervised U-Net training, PDE-based PINNs, diffusion/Transformer architectures, MLflow registry, persistent background job queues, production-grade upload/DICOM support, multicoil MRI/CG-SENSE, downloadable research PDF, patient-level clinical dataset tooling. These are future phases, not working features.

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

CT forward operator uses `skimage.transform.radon`; FBP uses `iradon`; SART uses `iradon_sart`. Angles are degrees in [0, 180). Its PyTorch differentiable forward approximation uses affine grid sampling and a line integral sum, so the numerical model is not identical to scikit-image Radon; train and evaluate consistency with the same operator.

MRI FFT is orthonormal with centered k-space (`fftshift(fft2(ifftshift(x), norm="ortho"))`). An acquired sample mask operates on complex k-space; zero-fill and iterative methods reuse exactly the same samples. Iterative MRI prior is Gaussian smoothing followed by hard complex k-space consistency: a heuristic, **not** formal compressed sensing. This first release supports single-coil only.

X-ray is 2D restoration of a radiograph under Gaussian blur and Gaussian noise. It is **not** 3D reconstruction from a single image.

Reference image metrics require ground truth and are not applied to clinical data without ground truth. Intensities are clipped to [0,1] for display/metrics and SSIM data_range=1. Runtime is process wall time and should not be treated as a hardware-normalized speed benchmark. Synthetic images are not real clinical outcomes.

## Reproducibility and scope

Each demo saves input, reference, reconstructions, residual maps, metrics CSV, a summary figure and metadata JSON with seed, acquisition parameters and package versions. Use identical measurements for method comparisons; run multiple seeds for statistical analyses. No restricted datasets are downloaded.

Optional datasets for future work: fastMRI (follow NYU access and use restrictions), NIH ChestX-ray14 and CheXpert (follow each dataset's license and agreements). Deidentify DICOM metadata and split by patient IDs when available. Do not add patient datasets to this repo.

API docs: `http://127.0.0.1:8000/docs`. Endpoints include health, modalities, methods, datasets, reconstruct, compare, experiments and reports. Current jobs are synchronous/in-memory; cancellation and persistence are **not implemented**.

## Development

```bash
ruff check src tests dashboard
pytest -q
docker build -t medreconlab .
docker run --rm -p 8501:8501 medreconlab
```
