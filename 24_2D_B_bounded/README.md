# Study 24: B Boundedness Extends to 2D Systems

**Date:** 2026-04-06
**Claim:** 24.1 (Grade: GREEN)

## Purpose

Test whether the B = 2*DeltaPhi/sigma*^2 boundedness proof (established for 1D in Study 22) extends to 2D metastable systems. Three of the 13 systems in the stability window are irreducibly 2D (toggle switch, tumor-immune, diabetes) and are not covered by the 1D proof.

## Data Provenance

**Tumor-immune parameters:** Kuznetsov et al. 1994, Table 1 (BCL1 lymphoma model). MFPT target = 730 days (BCL1 dormancy).

**Toggle switch parameters:** Gardner et al. 2000, Hill coefficient n=2. CME spectral gap data from Study 04 / Step 9 (step9_toggle_epsilon.py, exact spectral gap of Chemical Master Equation transition matrix).

**Toggle CME data embedded in script:** D_CME values at (alpha, Omega) pairs from Study 04.

**B values for 2D systems (prior):** Toggle B=4.83 (Study 04), Tumor-immune B=2.73 (Study 06), Diabetes B=5.54 (structural_B_diabetes_2D_SDE.py).

## Replication

### Dependencies
```
Python 3.8+
pip install numpy scipy
```
No GPU, no numba, no Colab. Runs on any laptop.

### Parameters

**Tumor-immune (Kuznetsov et al. 1994, Table 1):**

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Tumor injection rate | s | 13,000 | cells/day |
| Immune stimulation | p | 0.1245 | day^-1 |
| Half-saturation | g | 2.019e7 | cells |
| Immune-tumor kill | m | 3.422e-10 | cells^-1 day^-1 |
| Immune death | d | 0.0412 | day^-1 |
| Tumor growth | a | 0.18 | day^-1 |
| Tumor carrying capacity | b | 2.0e-9 | cells^-1 |
| Tumor immune-kill | n | 1.101e-7 | cells^-1 day^-1 |
| MFPT target | -- | 730 | days (BCL1 dormancy) |

**Toggle switch (Gardner et al. 2000):**

| Parameter | Value | Note |
|-----------|-------|------|
| Hill coefficient n | 2 | Standard toggle |
| alpha values | 3, 5, 6, 8, 10 | Bistable range |
| CME data | See D_CME_DATA in script | From Study 04 / Step 9 (exact spectral gap) |

**SDE simulation parameters:**

| Parameter | Value | Note |
|-----------|-------|------|
| Trials per sigma | 500 | Vectorized numpy |
| dt | 0.05 | days |
| max_days | 40,000 | Censoring cutoff |
| sigma_base | [40k, 55k, 75k, 100k, 140k, 200k] | Chosen to span useful MFPT range |
| Scale factors c | [0.5, 1.0, 2.0, 5.0] | For scale invariance test |
| Random seed | 42 | Reproducible |

### Running

```bash
python3 ../scripts/study24_2D_B_bounded.py

# Or from study folder:
python3 24_2D_B_bounded/study24_2D_B_bounded.py
```

Expected runtime: ~10-20 minutes (SDE simulations dominate). Key output lines to verify:
- Test 1: "CV = X.XX%" with X < 2.0
- Test 2: "PASS: beta_0 variation = 0.115 < 1.5"
- Test 3: "PASS: mid-range beta_0 variation = 0.910 < 1.5"
- Final: "ALL TESTS PASS"

### Exact reproducibility

With seed=42, all numerical results should match to 4+ significant figures. The SDE results may vary slightly across numpy versions due to floating-point ordering differences, but the qualitative results (CV < 2%, beta_0 variation < 1.5) are robust.

## Files

| File | Description |
|------|-------------|
| `README.md` | This file (replication instructions) |
| `RESULTS.md` | Full results, interpretation, and conclusions |
| `study24_2D_B_bounded.py` | All 3 tests: scale invariance (tumor-immune SDE), Kramers-Langer prefactor (toggle + tumor-immune) |

## Relationship to other studies

- **Study 22 (General B bounded, 1D):** Study 24 extends from 1D to 2D. Same 3-step proof structure. Resolves Study 22 Limitation #1 ("Tested on 1D potentials only").
- **Study 02 (B invariance):** B constancy within each system's bistable range. Study 24 proves the prefactor that governs B's behavior is bounded in 2D.
- **Study 04 (Toggle switch):** Toggle B=4.83, CV=3.8% from CME data. Study 24 adds the Kramers-Langer prefactor analysis showing why B is stable across alpha.
- **Study 06 (Medical systems, tumor-immune):** Tumor-immune B=2.73 from 2D SDE. Study 24 adds scale invariance proof and prefactor bounds.
