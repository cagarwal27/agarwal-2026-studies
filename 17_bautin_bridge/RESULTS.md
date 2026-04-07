# Study 17: Bautin Bridge — Results

## Question
Does B invariance and the stability window [1.8, 6.0] hold for Bautin (subcritical Hopf) bistability between a fixed point and a limit cycle? Does P(FP-LC | d) decay exponentially like cusp P(bistable | d)?

## Part 1: B from Bautin Normal Form

Amplitude equation: r' = mu*r + l1*r^3 - r^5

300 random (mu, l1) configurations in the bistable regime:

| Quantity | Value |
|----------|-------|
| B_Bautin | 4.143 (CV = 2.2%) |
| B_cusp (reference) | 2.979 (CV = 3.7%) |
| K_Bautin | 0.301 |
| Stability window | 100% of samples in [1.8, 6.0] |
| Barrier variation | 6,633x while B varies only 1.1x |

**B invariance confirmed for Bautin bistability.** B_Bautin > B_cusp (different geometry), but the invariance property holds with excellent precision.

## Part 2: P(FP-LC bistable | d) Dimensional Scaling

2D multi-channel Hill-function ODEs, d = 12-42:

| d  | P(FP-LC)   |
|----|------------|
| 12 | ~1.7e-3    |
| 18 | ~1.0e-3    |
| 24 | ~8.0e-4    |
| 30 | ~1.1e-3    |
| 36 | ~1.3e-3    |
| 42 | ~1.5e-3    |

gamma_Bautin = 0.021 (R^2 = 0.24). **P is essentially flat.** No exponential decay.

**Physical interpretation:** The Bautin (subcritical Hopf) requires tr(J) < 0 at the fixed point for FP stability. Degradation (negative self-terms) HELPS achieve tr(J) < 0. This is NOT a fight between degradation and activation. Contrast with Hopf limit cycles (Study 16) where degradation must be overcome for the limit cycle to survive — there, gamma = 0.172.

## Part 3: Multi-channel B via SDE escape

**Incomplete.** The SDE escape method fails for 2D FP-LC transitions. The escape from a fixed point to a limit cycle does not follow a 1D barrier-crossing path — the quasipotential landscape is inherently 2D. The ln(D) vs 1/sigma^2 Kramers relationship does not hold for this geometry.

## Comparison Table

| Source | gamma | R^2 | B | Decays? |
|--------|-------|-----|---|---------|
| Cusp (Study 11) | 0.197 | 0.995 | 2.979 | YES |
| Hopf (Study 16) | 0.172 | 0.9997 | -- | YES |
| Bautin (this study) | 0.021 | 0.24 | 4.143 | NO |

## Implications

1. **B invariance extends to FP-LC bistability.** The stability window is not cusp-specific. B_Bautin = 4.14 is inside [1.8, 6.0].
2. **Search scaling does NOT extend to Bautin.** FP-LC configurations are not exponentially rare with dimension. The exponential decay requires degradation OPPOSING the organized state.
3. **Consistent with degradation hypothesis (Fact 82):** gamma ~ 0.17-0.20 only when bounded feedback must overcome degradation. When degradation helps (Bautin) or is absent (polynomial), P is flat.

## Computational details
- Script: `bautin_bridge.py`
- Dependencies: numpy, scipy
- Part 1: 300 random configurations, analytical barrier, exact MFPT
- Part 2: 50k-2M samples per d value
- Part 3: Did not converge (methodological limitation)
