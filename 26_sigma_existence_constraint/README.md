# Study 26: Sigma Existence Constraint

**Claim:** 26.1 (Grade: YELLOW)
**Date:** 2026-04-06

## Purpose

Test whether the stability window B in [1.8, 6.0] acts as an existence constraint on the physical noise intensity sigma_process, explaining why sigma* ~ sigma_process. Quantify how much of the sigma match is explained by the window alone.

## Data Provenance

All parameters from SYSTEMS.md tables. Zero free parameters.

| Parameter | Value | Source |
|-----------|-------|--------|
| B_LOWER | 1.8 | SYSTEMS.md stability window (kelp, 13 systems) |
| B_UPPER | 6.0 | SYSTEMS.md stability window (coral, 13 systems) |
| B values (13 systems) | 1.80 - 6.04 | SYSTEMS.md, 240 MFPT computations |
| sigma* values (5 systems) | 0.0170 - 11.75 | SYSTEMS.md noise-source mapping table |
| sigma_process values (5 systems) | 0.017 - 17.5 | SYSTEMS.md, independent environmental estimates |
| Random seed | 42 | Reproducible (not used -- pure algebra) |

## Replication

### Dependencies
```
Python 3.8+
pip install numpy
```
No scipy required -- this study is pure algebra.

### Running
```bash
# From repository root:
python3 26_sigma_existence_constraint/study26_sigma_existence_constraint.py

# Or from study folder:
python3 study26_sigma_existence_constraint.py
```

Expected output: 5 tests + summary, < 1 second (pure algebra). Key lines to verify:
- Test 1: Band width = 1.8257 for all 13 systems; 12/13 have sigma* inside
- Test 2: 4/5 observed sigmas inside; kelp outside (B_eff = 0.81)
- Test 3: Kelp room above = 0.000; Coral room below = 0.000
- Test 4: Prior eliminated = 93.5%; average constraint share = 95.2%
- Test 5: [1.0, 8.0] still eliminates 88.7%

## Files

| File | Description |
|------|-------------|
| `README.md` | This file (replication instructions) |
| `RESULTS.md` | Full results, interpretation, and conclusions |
| `study26_sigma_existence_constraint.py` | All 5 tests: allowed band, observed ratios, asymmetry, explanatory power, sensitivity |

## Relationship to other studies

- **Study 19 (B bounded, cusp):** Proved B is bounded for the cusp normal form. Study 26 uses the stability window as input to derive the sigma constraint.
- **Study 22 (general B bounded):** Extended B boundedness to 4 potential families. Study 26 takes the universal boundedness as established and asks what it implies for sigma.
- **Study 24 (2D B bounded):** Extended to 2D systems (toggle, tumor-immune, diabetes). Their B values are inputs to Study 26.
- **Noise-source mapping (Facts 42-44):** The sigma* ~ sigma_process match that Study 26 partially explains. Fact 42 establishes the match; Fact 43 derives sigma_env for 4/5 systems; Study 26 shows the stability window provides a necessary condition.
- **B invariance (Fact 77, 86):** B invariance (scale cancellation) is a separate, deeper explanation for sigma* = sigma_process. Study 26's existence constraint is complementary: it explains WHY sigma must be close, while B invariance explains HOW the scale is set.
