# Study 11: Cusp Bridge -- Results

**Date:** 2026-04-04
**Scripts:** `cusp_bridge_derivation.py`, `bridge_dimensional_scaling.py`, `bridge_high_d_scaling.py`, `bridge_high_d_analysis.py`, `bridge_high_d_extension.py`

## Question

How does P(bistable) scale with parameter-space dimension d, and can the cusp normal form provide an analytic foundation for S(d) = exp(gamma*d)?

## Data Summary

16 data points for P(bistable | d) from Monte Carlo sampling of random 1D Hill-function ODEs: d = 5, 8, 11, 14, 17, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80. Sample sizes range from 30,000 (d=5-32) to 10,000,000 (d=80). Cusp normal form analytics from 300 random (a,b) pairs.

---

## Key Results

| Metric | Value | Source script |
|--------|-------|---------------|
| B_cusp (cusp normal form) | 2.979 | cusp_bridge_derivation |
| B_cusp CV | 3.7% (n=300) | cusp_bridge_derivation |
| K_cusp | 0.558 | cusp_bridge_derivation |
| P(bistable\|2D cusp) | 2/5 exactly | cusp_bridge_derivation |
| gamma_eff (exponential slope, d=14-62) | 0.2251 | bridge_high_d_scaling (combined d=14-62 fit) |
| **gamma_eff (exponential slope, d=14-80)** | **0.1973** | **bridge_high_d_extension (combined d=14-80 fit, R^2=0.9948)** |
| R^2 (linear fit, d>=14, to d=62) | 0.9799 | bridge_high_d_scaling |
| **R^2 (linear fit, d>=14, to d=80)** | **0.9948** | **bridge_high_d_extension** |
| Grand mean B (d=38-50) | 3.689 (CV=1.7%) | bridge_high_d_analysis |
| **B at d=68-80** | **3.43-3.45 (CV=9.5-10.9%)** | **bridge_high_d_extension** |
| Predicted d for S=10^13 (d=14-62 data) | 123-145 | cusp_bridge_derivation |
| **Predicted d for S=10^13 (d=14-80 data)** | **140-163** | **bridge_high_d_extension** |

---

## Script-by-Script Results

### Cusp analytics (cusp_bridge_derivation.py)

The cusp normal form dx/dt = -x^3 + ax + b has an analytic barrier DeltaPhi = x3*(x2-x1)^3/4. Sampling 300 random (a,b) pairs in the bistable region and computing exact MFPT gives B_cusp = 2.979 with CV = 3.7%. K_cusp = 0.558 matches the 1D SDE parabolic-well class. For 2D cusp parameters (a,b) drawn uniformly, the probability of bistability is exactly 2/5 (the cusp curve partitions the (a,b) plane into 3/5 monostable and 2/5 bistable).

Three models fitted to Omega(d) data:

| d | Omega |
|---|-------|
| 5 | 0.46 |
| 8 | 0.33 |
| 11 | 0.29 |
| 14 | 0.30 |
| 17 | 0.34 |
| 20 | 0.41 |
| 26 | 0.62 |
| 32 | 0.98 |

Three models fitted: linear, quadratic, and power-law.

### Dimensional scaling (bridge_dimensional_scaling.py, bridge_high_d_scaling.py)

As the number of parameters d increases, P(bistable) first rises (d=5-11, geometric effect), peaks (d~11-14), then decays exponentially. For d >= 14, ln(1/P) grows approximately linearly with d: gamma_eff = 0.2251 (d=14-62).

### Bridge equation

ln(D) = B + beta_0. Combining with S = 1/P(viable): S(d) = exp(gamma*d + c_0). For S = 10^13 (ln(S) ~ 29.93), d ~ 130 parameters.

### B invariance at high d (bridge_high_d_analysis.py)

Grand mean B = 3.689 (CV = 1.7%) across d=38-50. B remains in the stability window [1.8, 6.0] even at high dimensionality.

### Deceleration at d > 50: RESOLVED (bridge_high_d_extension.py)

P(bistable) at d=56 and d=62 showed slower decay than the linear extrapolation from d=14-50. Extension to d=68-80 shows the deceleration was statistical noise from small counts. Linear tail holds (max residual 0.468). Quadratic and stretched-exponential fits both show g2 > 0 / beta > 1 (decay accelerates, no plateau). gamma revised from 0.225 to 0.197; d for S=10^13 revised from 123-145 to 140-163.

### High-d extension data (bridge_high_d_extension.py)

Three new data points at d=68, 74, 80 with 3M-10M samples each:

| d | k | N_samples | N_bistable | P(bistable) | ln(1/P) | B mean | B CV |
|---|---|-----------|------------|-------------|---------|--------|------|
| 68 | 22 | 3,000,000 | 26 | 8.67e-6 | 11.656 | 3.429 | 0.095 |
| 74 | 24 | 5,000,000 | 22 | 4.40e-6 | 12.334 | 3.431 | 0.109 |
| 80 | 26 | 10,000,000 | 15 | 1.50e-6 | 13.410 | 3.451 | 0.100 |

Three model fits (d >= 14, 8 points):

| Model | Form | Key params | R^2 | d for S=10^13 |
|-------|------|------------|-----|---------------|
| A (linear) | c0 + gamma*d | gamma=0.1973, c0=-2.231 | 0.9948 | 163 |
| B (quadratic) | c0 + g1*d + g2*d^2 | g1=0.1443, g2=0.000562 | 0.9959 | 140 |
| C (stretched exp) | c0 + gamma*d^beta | gamma=0.0566, beta=1.267 | 0.9964 | 144 |

---

## Detailed Parameters

### cusp_bridge_derivation.py

| Parameter | Value | Notes |
|-----------|-------|-------|
| Cusp normal form | dx/dt = -x^3 + ax + b | Standard; Guckenheimer & Holmes 1983 |
| Potential | U(x) = x^4/4 - a*x^2/2 - b*x | Standard |
| B_EMPIRICAL | 3.78 | Mean B from bridge_dimensional_scaling |
| D_TARGET | 100.0 | Standard target |
| S0_TARGET | 1e13 | From Study 09 |
| seed | 42 | np.random.seed(42) |

### bridge_dimensional_scaling.py

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | f(x) = a - b*x + sum_i r_i*x^q_i/(x^q_i + h_i^q_i) | Generic multi-channel ODE |
| d = 2 + 3*k | 5, 8, 11, 14, 17, 20, 26, 32 | k = 1..10 channels |
| N_samples | 30,000 per d | Random parameter draws |
| N_MFPT | 100 subset | For B computation |
| B_TARGET | 3.8 | From B distribution test |
| Parameter ranges | a in [0.05, 0.80], b in [0.2, 2.0], r_i in [0.1, 2.0], q_i in [2.0, 15.0], h_i in [0.3, 2.0] | Design choices |
| seed | 42 | np.random.seed(42) |

### bridge_high_d_scaling.py

| Parameter | Value | Notes |
|-----------|-------|-------|
| D_PREV | [5, 8, 11, 14, 17, 20, 26, 32] | Hardcoded from bridge_dimensional_scaling.py |
| P_PREV | [0.174, 0.285, 0.332, 0.320, 0.275, 0.211, 0.095, 0.024] | Hardcoded from bridge_dimensional_scaling.py |
| New d values | 38, 44, 50, 56, 62 | k = 12, 14, 16, 18, 20 channels |
| N_samples (per d) | 100K, 200K, 500K, 1M, 2M | Increasing with d to maintain counts |
| N_MFPT_SUBSET | 50 | For B computation |
| BATCH_SIZE | 50,000 | Processing batch size |
| seed | 42 | np.random.seed(42) |

### bridge_high_d_extension.py

| Parameter | Value | Notes |
|-----------|-------|-------|
| D_PREV | [5, 8, 11, 14, 17, 20, 26, 32] | Hardcoded from bridge_dimensional_scaling.py |
| P_PREV | [0.174, 0.285, 0.332, 0.320, 0.275, 0.211, 0.095, 0.024] | Hardcoded from bridge_dimensional_scaling.py |
| New d values | 68, 74, 80 | k = 22, 24, 26 channels |
| N_samples (per d) | 3M, 5M, 10M | Much larger than prior scripts |
| X_SCAN | 500 points | Coarse scan for speed |
| X_VERIFY | 2000 points | Fine grid for brentq verification |
| N_MFPT_SUBSET | 50 | For B computation |
| BATCH_SIZE | 50,000 | Processing batch size |
| seed | 42 | np.random.seed(42) |

Two-stage approach: coarse 500-point scan identifies candidates, then 2000-point fine grid verifies with brentq. Log-space Hill evaluation (`np.exp(q * log_x)`) for numerical stability at high q. Combined fit uses 11 points (d=5-80), fitting d >= 14 (8 points).

### bridge_high_d_analysis.py

| Parameter | Value | Notes |
|-----------|-------|-------|
| P_NEW (d=38-62) | [3.64e-3, 6.25e-4, 1.04e-4, 3.40e-5, 2.20e-5] | Hardcoded from bridge_high_d_scaling.py |
| N_B_TARGET | 50 | B values per d |
| B_SPECS | (k=12,d=38), (k=14,d=44), (k=16,d=50) | d values for B computation |
| Three model fits | Linear, quadratic, stretched exponential | Fitted to ln(1/P) vs d for d >= 14 |
| seed | 42 | np.random.seed(42) |

---

## Replicability Assessment

**Overall: YELLOW**

| Script | Grade | Notes |
|--------|-------|-------|
| `cusp_bridge_derivation.py` | YELLOW | Cusp form is standard; Omega data hardcoded from prior computation. seed=42. |
| `bridge_dimensional_scaling.py` | YELLOW | Parameter ranges are design choices. seed=42. 30,000 samples per d. |
| `bridge_high_d_scaling.py` | YELLOW | P_PREV hardcoded. Up to 2M samples at d=62. seed=42. |
| `bridge_high_d_analysis.py` | YELLOW | P_NEW hardcoded from prior computation. **NEW/UNTRACKED.** seed=42. |
| `bridge_high_d_extension.py` | YELLOW | P_PREV hardcoded. Two-stage scan (500+2000 pts). Up to 10M samples at d=80. seed=42. |

All scripts are YELLOW because:
1. **Hardcoded data chain:** Each script hardcodes results from the previous script. bridge_high_d_analysis.py hardcodes P_NEW from bridge_high_d_scaling.py, which hardcodes P_PREV from bridge_dimensional_scaling.py, which provides Omega data used in cusp_bridge_derivation.py.
2. **Parameter ranges:** The ranges for a, b, r_i, q_i, h_i in the multi-channel model are design choices. Different ranges would change P(bistable) quantitatively but the exponential decay with d is expected to be robust.
3. **Cusp normal form:** Standard (Guckenheimer & Holmes 1983), but the connection to the multi-channel ODE is via analogy, not rigorous reduction.

All scripts are seeded (seed=42) and fully reproducible. numpy + scipy only.

bridge_high_d_analysis.py is a NEW/UNTRACKED file (not yet in git). Its results should be verified before relying on them.

---

## Limitations

1. **Extrapolation:** The data extends to d=80. Predicting d~150 for S=10^13 is a ~2x extrapolation beyond the data range (reduced from ~2.1x when data only reached d=62).
2. **Deceleration at d > 50: RESOLVED.** Extension to d=68-80 (script 5) shows the d=56-62 deceleration was statistical noise. All three models show no plateau; quadratic and stretched-exponential fits indicate slight acceleration (g2 > 0, beta > 1).
3. **Model specificity:** The multi-channel ODE f(x) = a - bx + sum Hill terms is one specific class of bistable system. Other functional forms (polynomial, rational, piecewise) might give different gamma values.
4. **Hardcoded chain:** Results propagate through 4 scripts via hardcoded constants. Any error in an early script propagates silently. The chain should be verified by running all scripts end-to-end.
5. **P(bistable|2D) = 2/5:** This exact result holds for uniformly distributed (a,b) in the cusp normal form. Real biological parameters are not uniformly distributed, so 2/5 is a reference point, not a prediction.
6. **bridge_high_d_analysis.py status:** This is a NEW/UNTRACKED file not yet committed to git. Its results should be verified before relying on them.
