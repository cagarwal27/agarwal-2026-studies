# Study 27: Formal Model Selection for S(d)

**Date:** 2026-04-06
**Script:** `../scripts/model_selection_Sd.py`
**Output:** `OUTPUT.txt`

## Question

Is S(d) = exp(gamma*d) formally preferred over competing models (power law, quadratic, stretched exponential)? How robust is the extrapolation to d ~ 150?

## Data

13 points in the exponential tail regime (d >= 14), from three Monte Carlo campaigns:

| Source | d range | Points | MC samples |
|--------|---------|--------|------------|
| bridge_dimensional_scaling.py | 14-32 | 5 | Large (>10K hits each) |
| bridge_high_d_scaling.py | 38-62 | 5 | 100K-2M (34-364 hits) |
| bridge_high_d_extension.py | 68-80 | 3 | 3M-10M (15-26 hits) |

## Models Tested

| Model | Formula | Params |
|-------|---------|--------|
| M1 (exponential) | ln(1/P) = c0 + gamma*d | 2 |
| M2 (power law) | ln(1/P) = c0 + alpha*ln(d) | 2 |
| M3 (quadratic) | ln(1/P) = c0 + g1*d + g2*d^2 | 3 |
| M4 (stretched exp) | ln(1/P) = c0 + gamma*d^beta | 3 |

## Key Results

### 1. Power law decisively rejected

| Criterion | dAIC (power vs exp) | Interpretation |
|-----------|-------|------|
| AIC | +12.63 | Decisive (>10) |
| BIC | +12.63 | Decisive (>10) |

Power law predicts log10(S) = 7.5 at d=150 (vs ~12-13 needed). Not viable.

### 2. Exponential is best parsimonious model

| Model | AICc | dAICc | Akaike weight | LOOCV RMSE |
|-------|------|-------|---------------|------------|
| M1 exponential | 33.09 | 0.00 | 50.0% | 0.723 |
| M3 quadratic | 33.78 | +0.69 | 35.5% | 0.704 |
| M4 stretched | 35.59 | +2.49 | 14.4% | 0.768 |
| M2 power law | 45.73 | +12.63 | 0.1% | 1.268 |

AICc prefers exponential (parsimony wins over marginal R^2 improvement of 3-param models).

### 3. Extrapolation to d = 150

| Model | log10(S) at d=150 | d for S=10^13 | Bootstrap 95% CI |
|-------|-------------------|---------------|------------------|
| M1 exponential | 12.3 | 158 | [150, 167] |
| M4 stretched | 11.0 | 181 | [150, 243] |
| M3 quadratic | 8.7 | infinity (turns over) | [142, 923] |
| M2 power law | 7.5 | 774 | [620, 958] |

### 4. Data discrepancy with Fact 73

**Critical finding:** Fact 73 fits used 8 points (d=14-32 + d=68-80), skipping d=38-62. This study uses all 13 points. The two analyses give opposite curvature:

| | 8-point (Fact 73) | 13-point (this study) |
|---|---|---|
| gamma | 0.197 | 0.203 |
| Quadratic g2 | +0.0006 (acceleration) | -0.0008 (deceleration) |
| Stretched beta | 1.27 (superlinear) | 0.78 (sublinear) |

The d=38-62 points sit above the line connecting the low-d and high-d endpoints, creating a mid-range bump. Whether this is real or Poisson noise (34-364 MC hits in that range) is unresolved.

**Residual pattern (13-point linear fit):**
- d=14-17: slightly positive
- d=20-32: negative (data below line)
- d=38-56: positive (data above line, up to +1.12)
- d=62-80: negative (data below line)

This S-shaped wave is systematic, suggesting the linear model misses some structure. But the structure could be a transition-zone artifact (tail of the P(bistable) peak at d~11).

### 5. Sensitivity tests

- Excluding d=56,62: gamma = 0.200, g2 still negative (-0.0006)
- Only d >= 26: gamma = 0.203, g2 = -0.0024 (deceleration stronger)
- Weighted by sqrt(N_hits): gamma = 0.158 (lower — high-d points downweighted)

## Conclusion

**Strong result:** The exponential form S(d) = exp(gamma*d) is formally validated. Power law is decisively rejected (dAIC > 12). The 3-parameter alternatives don't earn their extra parameter under AICc.

**Open question:** The curvature (acceleration vs deceleration) depends on whether d=38-62 data is included. Resolution: re-run d=38-62 with 10M MC samples each.

**Impact on framework:** The exponential form — the core universality claim — is confirmed. The specific gamma value and curvature question affect the extrapolation precision to d~150 but do not affect biological predictions (which use biology-specific gamma, not cusp bridge gamma).

## Recommended Follow-up

Re-run bridge_high_d_scaling.py at d=38, 44, 50, 56, 62 with 10M samples each (matching d=68-80 quality). This resolves the curvature ambiguity cheaply (~hours of computation).
