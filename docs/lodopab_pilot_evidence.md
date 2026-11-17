# LoDoPaB-CT pilot evidence

The pilot used the full-angle, simulated low-dose measurements in LoDoPaB-CT
release 1.0.0 (DOI 10.5281/zenodo.3384092). It establishes the evaluation
workflow on subsets of the official validation and test splits. Full-cohort
validation and held-out results are pending.

## Validation subset

The validation indices were 0, 16, 32, 48, 64, 80, 96, 112, 1664, 1680, 1696,
1712, 1728, 1744, 1760, and 1776. The 16 slices came from 14 patients. The
sweep considered CGLS budgets of 1, 2, 3, 5, 8, 12, 16, 24, and 32 iterations.
It selected the highest mean PSNR after averaging within each patient, with
fewer iterations breaking a tie.

Eight iterations won at 30.88 dB and remained the winner in all 14
leave-one-patient-out reanalyses. The paper's iteration budget will come from
the full official validation split.

## Matched test subset

The test indices were 1664, 1680, 1696, 1712, 1728, 1744, 1760, and 1776,
covering eight slices from seven patients. All methods used the same
measurements. The CGLS budget was set from the validation pilot before
evaluating these test cases. The table reports equal-patient means.

| Method | PSNR (dB) | SSIM | PSNR difference vs. tuned Hann FBP (dB) |
| --- | ---: | ---: | ---: |
| Ram-Lak FBP | 23.01 | 0.366 | −7.17 |
| CGLS, 8 iterations | 29.70 | 0.709 | −0.48 |
| Published tuned Hann FBP | 30.18 | 0.742 | 0.00 |
| Published DIVαℓ FBP+U-Net | 34.21 | 0.821 | +4.03 |

For the PSNR difference against tuned Hann FBP, the paired patient-bootstrap
95% interval was −0.68 to −0.22 dB for CGLS and +2.83 to +5.15 dB for
FBP+U-Net. The intervals use 2,000 patient resamples with seed 11. The cohort
is small and the interval estimates are preliminary.

## Reproduction and full-cohort handoff

The run used ASTRA CPU and the published DIVαℓ checkpoint. The exact commands,
checkpoint link, archive staging steps, and frozen validation-to-test handoff
are in the [README](../README.md#lodopab-ct-native-geometry-baselines). The
[research protocol](research_protocol.md) describes the remaining evaluation
criteria. Archive the full validation `selection.json` with the sweep summary,
then use it for held-out evaluation and retain the resulting manifest and
paired patient summaries.
