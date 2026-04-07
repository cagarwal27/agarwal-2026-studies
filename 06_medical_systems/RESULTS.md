# Study 06: Medical Systems (Cancer Biology + Human Disease)

**Date:** 2026-04-06
**Scripts:** `structural_B_tumor_immune.py`, `structural_B_tumor_immune_2D.py`, `structural_B_tumor_immune_SDE_scan.py`, `structural_B_diabetes.py`, `structural_B_diabetes_2D_SDE.py`

## Question

Does B invariance hold for non-ecological medical systems? Does the 2D treatment resolve cases where 1D adiabatic reduction fails?

## Data

Two published BioModels with 0 free model parameters:
- Tumor-immune: Kuznetsov et al. 1994, BioModels BIOMD0000000762 (8 parameters)
- Diabetes: Topp et al. 2000, BioModels BIOMD0000000341 (9 parameters)

## Key Results

| System | Dimension | B | CV | In stability window? | Method |
|--------|-----------|------|------|---------------------|--------|
| Tumor-immune | 1D | 3.98 | -- | Yes | Adiabatic reduction (superseded) |
| Tumor-immune | 2D | 2.73 | 2.7% | Yes [1.8, 6.0] | QPot + SDE |
| Diabetes | 1D | FAILS | 80.4% | No | 1D MFPT |
| Diabetes | 2D | 5.54 | 5.2% | Yes [1.8, 6.0] | 2D SDE |

## Parameters and Sources

### Tumor-immune (Kuznetsov 1994)

All 8 parameters from Kuznetsov et al. 1994, Table 1:

| Parameter | Value | Description |
|-----------|-------|-------------|
| s | 13,000 | Effector cell source rate |
| p | 0.1245 | Proliferation rate |
| g | 2.019e7 | Half-saturation |
| m | 3.422e-10 | Killing rate |
| d | 0.0412 | Effector death rate |
| a | 0.18 | Tumor growth rate |
| b | 2.0e-9 | Reciprocal carrying capacity |
| n | 1.101e-7 | Tumor-induced inactivation |

Source: Kuznetsov et al. 1994, BioModels BIOMD0000000762. 0 free parameters.

### Diabetes (Topp 2000)

All 9 parameters from Topp et al. 2000:

| Parameter | Value | Description |
|-----------|-------|-------------|
| R0 | 864 | Insulin sensitivity |
| EG0 | 1.44 | Glucose effectiveness |
| SI | 0.72 | Insulin sensitivity |
| sigma_p | 43.2 | Beta-cell secretion |
| alpha | 20,000 | Insulin degradation |
| k | 432 | Glucose production |
| d0 | 0.06 | Beta-cell death |
| r1 | 0.84e-3 | Beta-cell replication |
| r2 | 2.4e-6 | Beta-cell death (glucose-dependent) |

Source: Topp et al. 2000, BioModels BIOMD0000000341. D_target=75 from DPP placebo arm.

### Diabetes 2D noise scales (structural_B_diabetes_2D_SDE.py)

| Parameter | Value | Notes |
|-----------|-------|-------|
| SIGMA_G | 10 | Described as "biologically motivated" -- **unsourced** |
| SIGMA_BETA | 0.01 | Described as "biologically motivated" -- **unsourced** |

These noise scales are not traced to a published source. This is the primary reason the diabetes 2D script is YELLOW.

## Script Details

### Tumor-Immune (Kuznetsov 1994, BioModels BIOMD0000000762)

| # | Script | What it does | Status |
|---|--------|-------------|--------|
| 1 | `structural_B_tumor_immune.py` | 1D adiabatic reduction. B=3.98 (1D, superseded by 2D result). All 8 params from Kuznetsov 1994 Table 1. MFPT=730 days (BCL1 dormancy). | GREEN |
| 2 | `structural_B_tumor_immune_2D.py` | 2D QPot + 2D SDE. B=2.73 +/- 2.7%. seed=42 (DE), seed=12345 (SDE). | GREEN |
| 3 | `structural_B_tumor_immune_SDE_scan.py` | Pure 2D SDE. B=2.73 +/- 2.7%, 500 trials, 10 scan points. seed=42. | GREEN |

### Diabetes (Topp 2000, BioModels BIOMD0000000341)

| # | Script | What it does | Status |
|---|--------|-------------|--------|
| 4 | `structural_B_diabetes.py` | 1D B FAILS (CV=80.4%). 29,670x timescale separation. All 9 params from Topp 2000. D_target=75 (DPP placebo arm). | GREEN |
| 5 | `structural_B_diabetes_2D_SDE.py` | 2D SDE reverses: B=5.54 +/- 5.2%. **Requires numba.** SIGMA_G=10, SIGMA_BETA=0.01. seed=42. | YELLOW |

## Interpretation

- Tumor-immune 1D result (B=3.98) is superseded by the correct 2D treatment (B=2.73).
- Diabetes 1D fails catastrophically (CV=80.4%) due to 29,670x timescale separation between glucose and beta-cell dynamics.
- Diabetes 2D reverses the failure: B=5.54 lands in the stability window, demonstrating that the 2D treatment captures dynamics that 1D misses.
- MFPT=730 days for tumor-immune matches BCL1 lymphoma dormancy timescale.
- Both systems confirm B invariance when treated in the correct dimensionality, extending the framework from ecology to cancer biology and metabolic disease.

## Conclusions

B invariance holds for both medical systems when analyzed in the appropriate dimensionality (2D). The diabetes result is particularly informative: the 1D catastrophic failure (CV=80.4%) followed by the 2D recovery (B=5.54, CV=5.2%) demonstrates that timescale separation can break adiabatic reduction, but the full 2D treatment restores B invariance. Both systems land in the stability window [1.8, 6.0] with 0 free model parameters.
