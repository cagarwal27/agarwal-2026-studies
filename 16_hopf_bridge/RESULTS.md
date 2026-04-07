# Study 16: Hopf Bridge Scaling -- Results

**Date:** 2026-04-05
**Script:** `hopf_bridge_scaling.py`
**Provenance claims:** 16.1, 16.2

---

## Question

Does the cusp bridge result P(bistable) ~ exp(-gamma_cusp * d) generalize to Hopf bifurcations (limit cycles)?

**Status:** Measured gamma_Hopf = 0.172 from 4 decay-regime points (d=24-42). Exponential form confirmed. gamma is 13% below gamma_cusp = 0.197 -- same order but distinguishably different. Consistent with the "alpha pattern": exponential form universal, constant bifurcation-type-dependent.

---

## Model

```
dx/dt = a1 - b1*x + c1*y + sum_i [r1_i * x^q1_i / (x^q1_i + k1_i^q1_i)]
dy/dt = a2 - b2*y + c2*x + sum_i [r2_i * y^q2_i / (y^q2_i + k2_i^q2_i)]
```

This is a 2D analog of the cusp bridge's 1D multi-channel ODE. The cross-coupling terms (c1*y, c2*x) enable oscillation -- the 1D model cannot produce Hopf bifurcations.

### Detection pipeline

For each random parameter set:

1. **Find equilibria:** Vectorized Newton's method across all samples in batch. 16 initial conditions (4x4 grid in [0.15, 3.5]^2), 20 iterations. Pre-computed k^q for efficiency.

2. **Hopf check:** Analytical Jacobian at each converged equilibrium. Unstable spiral criterion: tr(J) > 0 AND discriminant < 0 (complex eigenvalues).

3. **Limit cycle verification:** RK45 integration (t=0 to 300) from perturbation near unstable equilibrium. Trajectory classified as stable limit cycle if:
   - Bounded (|x|, |y| < 100 throughout)
   - Periodic in last 50%: peak height CV < 0.20, inter-peak interval CV < 0.20
   - At least 5 peaks
   - Peak-to-trough amplitude > 0.01

### Detection rate calibration

Ran fsolve (16 initial conditions, ground truth) vs vectorized Newton on 2000 samples at each d:

| d | fsolve spirals | Newton spirals | ratio |
|---|---------------|----------------|-------|
| 12 | 8 | 4 | 0.50 |
| 18 | 13 | 12 | 0.92 |
| 24 | 23 | 17 | 0.74 |
| 30 | 27 | 19 | 0.70 |
| 36 | 28 | 23 | 0.82 |

Mean ratio: 0.74. Slope vs d: 0.007 (R^2 = 0.18 -- no meaningful trend). The d=12 outlier (0.50) is Poisson noise on 8 vs 4 counts. At d=18-36 the ratio is 0.70-0.92 with no systematic d-dependence.

**Conclusion:** Detection miss rate is approximately d-independent. Gamma is unbiased. Absolute P values are ~26% low.

---

## Key Results

| d | n_ch | N | N_spiral | N_lc | P(lc) | ln(1/P) |
|---|------|---|----------|------|-------|---------|
| 12 | 1 | 100,000 | 162 | 31 | 3.10e-4 | 8.079 |
| 18 | 2 | 250,000 | 1,409 | 164 | 6.56e-4 | 7.329 |
| 24 | 3 | 500,000 | 4,276 | 197 | 3.94e-4 | 7.839 |
| 30 | 4 | 1,000,000 | 9,317 | 143 | 1.43e-4 | 8.853 |
| 36 | 5 | 2,000,000 | 17,531 | 97 | 4.85e-5 | 9.934 |
| 42 | 6 | 5,000,000 | 38,338 | 91 | 1.82e-5 | 10.914 |

P(lc) peaks at d=18 then decays -- same qualitative pattern as cusp bridge (peak at d~14).

### Point-to-point gamma

| Interval | gamma | ratio to gamma_cusp |
|----------|-------|---------------------|
| d=12->18 | -0.125 | (P increasing) |
| d=18->24 | 0.085 | 0.43 |
| d=24->30 | 0.169 | 0.86 |
| d=30->36 | 0.180 | 0.91 |
| d=36->42 | 0.163 | 0.83 |

Local gamma rises from d=18-36 then drops at d=36-42. Stabilizes around 0.17, does not converge to 0.197.

### Linear fits

- d >= 18 (5 points): gamma = 0.154, R^2 = 0.987
- d >= 24 (4 points): gamma = 0.172, R^2 = 0.9997
- d >= 30 (3 points): gamma = 0.172, R^2 = 0.999

**Best estimate: gamma_Hopf = 0.172** (d >= 24, 4 decay-regime points).

For comparison: gamma_cusp = 0.197 (from Study 11, d=14-80, 8 fitting points, R^2 = 0.995). Ratio: 0.87 (13% difference).

### Spiral-to-limit-cycle conversion rate

| d | Conversion rate |
|---|----------------|
| 12 | 19.1% |
| 18 | 11.6% |
| 24 | 4.6% |
| 30 | 1.5% |
| 36 | 0.6% |
| 42 | 0.24% |

Unstable spirals increase monotonically with d; the fraction producing stable limit cycles drops. The concentration-of-measure effect acts on trajectory containment, not instability.

---

## Interpretation

### What the data shows

1. **P(limit cycle | d) decays exponentially at high d.** Confirmed for d=24-42 (4 points). The functional form is the same as the cusp bridge.

2. **gamma_Hopf = 0.172.** Same order as gamma_cusp = 0.197 but 13% lower. The local gamma stabilizes around 0.17 at d=30-42 -- the acceleration seen in the 5-point dataset (d=12-36) was an artifact of too few decay-regime points. With d=42 included, gamma does not converge to 0.197.

3. **Non-monotonic P(d) at low d.** P peaks at d=18 before decaying -- matches the cusp bridge pattern (peak at d~14). Hopf peak is at higher d because 2D oscillation requires more parameters to "turn on" than 1D bistability.

4. **Spiral abundance vs limit cycle rarity.** Unstable spirals increase monotonically with d (0.16% at d=12 to 0.77% at d=42). But the fraction converting to stable limit cycles drops from 19% to 0.24%. The concentration-of-measure effect acts on trajectory containment, not on the instability itself.

### What the data does not show

1. **Whether gamma_Hopf = gamma_cusp.** It doesn't. gamma_Hopf = 0.172, gamma_cusp = 0.197. The 13% difference is real -- the d=42 point confirms gamma stabilizes around 0.17, not trending toward 0.20. This is the "alpha pattern": exponential form universal, constant bifurcation-type-dependent.

2. **Model independence.** Both cusp and Hopf bridges use Hill-function ODEs. The same gamma could be a property of this model class rather than a universal geometric constant. Untested on polynomial or rational-function models.

3. **Detection-rate stability at d=42.** Calibration covers d=12-36. d=42 was not calibrated, but the flat trend (slope ~0) across d=12-36 suggests the miss rate is also stable at d=42.

---

## Replicability Assessment

**Overall: YELLOW**

| Aspect | Grade | Notes |
|--------|-------|-------|
| Model | YELLOW | 2D Hill-function ODE is a design choice. Different model classes untested. |
| Parameter ranges | YELLOW | Matched to cusp bridge ranges. Different ranges could shift P and potentially gamma. |
| Detection | YELLOW | Vectorized Newton finds ~74% of unstable spirals vs fsolve (mean across d=12-36). Calibration shows ratio is d-independent (slope=0.007/d, R^2=0.18), so gamma is unbiased. |
| Periodicity check | GREEN | Standard peak detection with CV thresholds. Conservative (may miss slow-convergence limit cycles). |
| Reproducibility | GREEN | Seeded (42), numpy+scipy only, self-contained. |

The YELLOW grade is primarily due to:
1. **Detection completeness:** Vectorized Newton finds ~74% of unstable spirals compared to per-sample fsolve (calibrated). The miss rate is d-independent (slope ~0), so gamma is unbiased. Absolute P values are ~26% low.
2. **Single model class:** Both cusp and Hopf bridges use Hill-function ODEs. The gamma similarity could be a property of this model class rather than a universal geometric constant.
3. **Decay regime coverage:** 4 points (d=24, 30, 36, 42) in the exponential decay regime. R^2 = 0.9997 from 4 points. Sufficient to establish gamma but fewer than the cusp bridge's 8 points.

---

## What would strengthen this

- Test a second model class (e.g., polynomial 2D ODE with random coefficients) to determine if gamma is model-dependent or geometry-dependent
- Extend to d=48-60 to confirm gamma stabilization
- Compute B analog (noise-induced amplitude death) to test whether B_Hopf falls in stability window [1.8, 6.0]

---

## Limitations

1. **4 points in decay regime.** The exponential decay is measured from d=24, 30, 36, 42. The cusp bridge used 8+ points (d=14-80). Sufficient to determine gamma ~ 0.172 but not to distinguish linear from quadratic from stretched-exponential functional forms.
2. **Single model class.** Both cusp and Hopf bridges use Hill-function ODEs. The gamma similarity could be a property of this model class rather than a universal geometric constant. Untested on polynomial or rational-function models.
3. **Parameter range dependence untested.** Widening or narrowing parameter ranges could shift both the absolute P and the slope. The cusp bridge's gamma was shown to be robust across ranges; this has not been verified for Hopf.
4. **No B analog.** The barrier analog for limit cycles (noise-induced amplitude death) was not computed. This would test whether B_Hopf falls in the stability window [1.8, 6.0], complementing Study 17's B_Bautin result.

---

## Relationship to other studies

- **Study 11 (Cusp bridge):** gamma_cusp = 0.197 from 1D Hill-function ODEs. Study 16 extends to 2D Hopf.
- **Study 17 (Bautin bridge):** Tests B analog for Bautin bifurcation.
- **Study 27 (Model selection):** Uses gamma values from Studies 11 and 16 in model comparison.

---

## Provenance

- Script: `16_hopf_bridge/hopf_bridge_scaling.py`
- Seed: np.random.seed(42)
- Dependencies: numpy + scipy only
- Runtime: ~75 min for d=12-36, ~132 min for d=42 (5M samples). Total ~210 min.
- Full output: `results_output.txt`
