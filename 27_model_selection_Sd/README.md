# Study 27: Formal Model Selection for S(d)

**Date:** 2026-04-06

## Purpose

Formal AIC/BIC model comparison for the complexity scaling law S(d) = exp(gamma*d). Tests whether the exponential form is statistically preferred over alternatives, and quantifies extrapolation uncertainty to d ~ 150.

## Data Provenance

All data are P(bistable | d) measurements from Monte Carlo sampling of random 1D Hill-function ODEs: f(x) = a - b*x + sum_i [r_i * x^q_i / (x^q_i + h_i^q_i)], with d = 2 + 3*k parameters (k channels).

### Source 1: bridge_dimensional_scaling.py (Study 11)

- **d values:** 5, 8, 11, 14, 17, 20, 26, 32
- **P values:** 0.174, 0.285, 0.332, 0.320, 0.275, 0.211, 0.095, 0.024
- **MC samples:** Not recorded per-d but large (>10,000 bistable hits at each d)
- **Parameter distributions:** a ~ U(0.05, 0.80), b ~ U(0.2, 2.0), r_i ~ U(0.1, 2.0), q_i ~ U(2.0, 15.0), h_i ~ U(0.3, 2.0)
- **Location:** `11_cusp_bridge/`, hardcoded in `bridge_high_d_analysis.py` as D_PREV, P_PREV
- **Seed:** np.random.seed(42)

### Source 2: bridge_high_d_scaling.py (Study 11)

- **d values:** 38, 44, 50, 56, 62
- **P values:** 3.64e-3, 6.25e-4, 1.04e-4, 3.40e-5, 2.20e-5
- **MC samples:** 100K, 200K, 500K, 1M, 2M respectively
- **Bistable hits:** 364, 125, 52, 34, 44 respectively
- **Same parameter distributions and seed as Source 1**
- **Location:** `11_cusp_bridge/`, hardcoded in `bridge_high_d_analysis.py` as D_NEW, P_NEW

### Source 3: bridge_high_d_extension.py (Study 11)

- **d values:** 68, 74, 80
- **P values:** 8.67e-6, 4.40e-6, 1.50e-6
- **MC samples:** 3M, 5M, 10M respectively
- **Bistable hits:** ~26, ~22, ~15 respectively (estimated from P * N_samples)
- **Same parameter distributions and seed as Source 1**
- **Location:** `11_cusp_bridge/bridge_high_d_extension.py`
- **P values from:** Fact 73 in `../FACTS.md`

### What this study fits

Only the exponential tail regime: d >= 14 (13 of the 16 total points). The d = 5, 8, 11 points are in the peak regime where P(bistable) increases with d, not the decay tail.

## Important Note: Difference from Fact 73

Fact 73 reports fits on "8 points from d=14-80." These 8 points are d = {14, 17, 20, 26, 32, 68, 74, 80} — the Source 1 tail + Source 3, **skipping all of Source 2 (d=38-62)**. This is because bridge_high_d_extension.py combined the original low-d data directly with d=68-80, without incorporating the intermediate data from bridge_high_d_scaling.py.

This study uses all 13 points (Sources 1+2+3). The inclusion of Source 2 changes the curvature sign (see RESULTS.md Section 4).

## Replication

### Requirements
- Python 3.8+
- numpy, scipy (no other dependencies)

### To reproduce the model selection
```bash
python3 ../scripts/model_selection_Sd.py
```
Runtime: ~3 minutes (dominated by 10,000 bootstrap iterations with curve_fit).

Output matches `OUTPUT.txt`.

### To reproduce the underlying P(bistable) data
```bash
# Source 1 + Source 2 (d=5-62):
python3 11_cusp_bridge/bridge_high_d_analysis.py
# Runtime: 30-60 minutes (Phase 2 B computation)

# Source 3 (d=68-80):
python3 11_cusp_bridge/bridge_high_d_extension.py
# Runtime: several hours (10M samples at d=80)
```

All scripts use np.random.seed(42) for reproducibility.

## Files

| File | Contents |
|------|----------|
| `README.md` | This file — provenance, replication, data sources |
| `RESULTS.md` | Findings, tables, interpretation, sensitivity tests |
| `OUTPUT.txt` | Full script output (raw numbers) |
| `model_selection_Sd.py` | Analysis script (also in `../scripts/`) |
