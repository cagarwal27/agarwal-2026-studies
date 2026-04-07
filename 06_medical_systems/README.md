# Study 06: Medical Systems (Cancer Biology + Human Disease)

**Date:** 2026-04-06

## Purpose

Tests B invariance in two non-ecological medical systems (tumor-immune and diabetes), both using published BioModels with 0 free model parameters. Demonstrates that the 2D treatment is required when 1D adiabatic reduction fails.

## Data Provenance

### Tumor-immune
All 8 parameters from Kuznetsov et al. 1994, Table 1. BioModels BIOMD0000000762. 0 free parameters.

### Diabetes
All 9 parameters from Topp et al. 2000. BioModels BIOMD0000000341. D_target=75 from DPP placebo arm. Noise scales SIGMA_G=10 and SIGMA_BETA=0.01 in the 2D SDE script are unsourced.

**Provenance claims:** 10.1, 10.2

## Replication

### Requirements
- Python 3.8+
- numpy, scipy (scripts 1-4)
- numpy, scipy, numba (script 5)

```
pip install numpy scipy          # scripts 1-4
pip install numpy scipy numba    # script 5
```

No import dependencies between scripts. No external data files.

### To reproduce

Scripts can be run in any order. No dependencies between them.

```bash
cd 06_medical_systems/

# Tumor-immune
python3 structural_B_tumor_immune.py          # 1D reduction (~seconds, deterministic)
python3 structural_B_tumor_immune_2D.py       # 2D QPot + SDE (~minutes, seeded)
python3 structural_B_tumor_immune_SDE_scan.py # Pure 2D SDE (~minutes, 500 trials)

# Diabetes
python3 structural_B_diabetes.py              # 1D analysis (~seconds, deterministic)
python3 structural_B_diabetes_2D_SDE.py       # 2D SDE (~minutes, requires numba)
```

### Replicability grades

**Overall: YELLOW**

| Script | Grade | Notes |
|--------|-------|-------|
| `structural_B_tumor_immune.py` | GREEN | All params from Kuznetsov 1994 Table 1, deterministic |
| `structural_B_tumor_immune_2D.py` | GREEN | Seeded (42+12345), published params |
| `structural_B_tumor_immune_SDE_scan.py` | GREEN | Seeded (42), 500 trials, published params |
| `structural_B_diabetes.py` | GREEN | All params from Topp 2000, deterministic |
| `structural_B_diabetes_2D_SDE.py` | YELLOW | Requires numba; SIGMA_G and SIGMA_BETA unsourced |

Tumor-immune scripts are all GREEN: every parameter is traced to Kuznetsov 1994 Table 1 via BioModels, with 0 free parameters. The diabetes 2D script drags the overall grade to YELLOW because of the unsourced noise scales and the numba dependency.

## Files

| File | Contents |
|------|----------|
| `README.md` | This file -- provenance, replication, data sources |
| `RESULTS.md` | Findings, tables, interpretation |
| `structural_B_tumor_immune.py` | 1D adiabatic reduction (B=3.98, superseded by 2D) |
| `structural_B_tumor_immune_2D.py` | 2D QPot + 2D SDE (B=2.73) |
| `structural_B_tumor_immune_SDE_scan.py` | Pure 2D SDE (B=2.73, 500 trials) |
| `structural_B_diabetes.py` | 1D B analysis (FAILS, CV=80.4%) |
| `structural_B_diabetes_2D_SDE.py` | 2D SDE (B=5.54, requires numba) |
