# MedReconLab research protocol (draft — no fabricated results)

## Candidate article

**MedReconLab: A Reproducible, Physics-Aware Benchmarking Framework for Sparse-View CT Reconstruction**

Alternative title after strong ablation results:
**Geometry-Consistent Evaluation of Classical, Supervised and Physics-Informed CT Reconstruction Under Sparse-View Acquisition**

This is a research *plan*, not evidence that either title represents a novel method.

## Research question

Under identical CT projection geometry, noise realizations, training data and held-out
cases, how do FBP, SART, supervised FBP-to-image U-Net, and genuinely physics-informed
reconstruction compare on quality, measurement residual, latency and compute cost?

The initial contribution is a controlled and reproducible evaluation **framework**.
Any proposed algorithmic novelty must be stated separately and experimentally demonstrated.
Do not present pixel optimization or a learned postprocessor as a PDE-based PINN.

## Baseline and advanced research matrix

| Category | Method | Status | Required publication evidence |
|---|---|---|---|
| Analytic | FBP | Implemented | Acquisition-matched reference |
| Iterative | SART | Implemented | Same sinogram, iterations, residual |
| Supervised | CT U-Net | Implemented, tiny CI training only | Full training, independent test sets |
| Physics-guided | Differentiable CT image optimization | Implemented on its own approximate operator | Operator agreement study, converged experiments |
| Regularization | TV / convex reconstruction | Not implemented | Real optimization baseline |
| Advanced | Unrolled/Transformer/Diffusion | Not implemented | Actual code, model weights and controlled testing |

## Dataset stages

1. **Synthetic smoke/ablation:** affine-perturbed Shepp–Logan phantoms. These are
   not independent human patients and do not justify clinical conclusions.
2. **External benchmark:** LoDoPaB-CT, using its specified split and patient IDs.
   It provides simulated low-dose measurements derived from CT patient scans.
   Keep the official acquisition operator, normalization and geometry intact
   instead of implying the existing scikit-image circle Radon pipeline is compatible.
3. **Optional second dataset:** AAPM Low Dose CT public image/projection resources,
   after verifying dataset access and permitted uses; preserve patient-level splits.

No dataset with usage restrictions is downloaded automatically or committed.

## Experimental design

- CT angular measurements: 30, 60, 90, 180 projections; 120° and 180° spans,
  three documented noise settings and multiple held-out cases.
- For each case/method use the same **measured sinogram** and reference.
- Train distinct U-Nets for each acquisition condition, or explicitly
  designate mismatched settings as out-of-distribution tests.
- Compare FBP, SART, regularized physics-based methods and U-Net.
- Report PSNR, SSIM, MAE/RMSE/NRMSE, sinogram residual, inference latency,
  training time, peak CPU/GPU memory, hardware and model parameter count.
- Timing must include FBP preprocessing for U-Net if claiming end-to-end speed.
- Use method-wise case pairing; report bootstrap CIs using patient-level
  resampling where patient IDs exist. Do not mistake a CI over phantom seeds
  for a CI over clinical subjects.
- Hold out a test set entirely from hyperparameter tuning and checkpoint selection.
- Compare against at least one *strong, modern* published baseline on the
  same established benchmark, with compatible licensing and citations.
- Ablate image prior, data consistency, projection density, dose/noise,
  and training geometry. Inspect artifacts and ROI behavior.

## Paper outline

1. Abstract — only after final measured evidence.
2. Introduction — clinical motivation and reproducibility challenge.
3. Related work — classical CT, iterative methods, learned reconstruction,
   physics-informed/unrolled models, and benchmark frameworks.
4. Forward model and acquisition geometries.
5. MedReconLab architecture and methods (distinguish implemented vs planned).
6. Datasets, data leakage prevention, hardware and training protocol.
7. Experiments and statistical methodology.
8. Results (actual tables and visualizations; no illustrative metrics).
9. Ablation, calibration, failure cases, threats to validity.
10. Reproducibility statement, ethics, code availability and limitations.

## Publication readiness gate

- [ ] At least one external medically derived CT benchmark legally acquired
- [ ] Official data split respected, patient-level independence checked
- [ ] Forward model and native geometry validated
- [ ] Strong iterative and learning-based baselines run under matching conditions
- [ ] Multi-seed training and test results with confidence intervals
- [ ] End-to-end inference and compute/memory measurements
- [ ] Representative error maps, failures and uncertainty analysis
- [ ] All metrics regenerated from saved raw case outputs
- [ ] Clear scientific contribution beyond a software integration
- [ ] Ethical review of data use and genuine attribution of prior work

Until then, a **software demonstration, reproducibility report or workshop
poster** is a more proportionate claim than a validated reconstruction breakthrough.

## Starting references

- Leuschner J. et al., *LoDoPaB-CT, a benchmark dataset for low-dose computed
  tomography reconstruction*, Scientific Data 8:109 (2021).
  https://doi.org/10.1038/s41597-021-00893-z
- LoDoPaB-CT data and documentation: https://zenodo.org/records/3384092
- AAPM Low Dose CT Grand Challenge resources:
  https://www.aapm.org/GrandChallenge/LowDoseCT/

Please check each source's access and redistribution restrictions before
incorporating any externally hosted dataset.
