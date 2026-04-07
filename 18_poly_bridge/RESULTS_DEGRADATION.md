# Polynomial + Degradation Test: Does enforced degradation recover gamma ~ 0.20?

## Question
Study 18 showed P(limit cycle | d) is flat (~1.5%) for random polynomial ODEs. Study 16 showed gamma = 0.175 for Hill-function ODEs with degradation. Does the exponential decay come from (A) Hill-function saturation, or (B) degradation-vs-activation architecture?

## Model
2D polynomial ODE identical to Study 18, except the linear self-terms are forced negative:
```
dx/dt = -b1*x + [other a_{ij} x^i y^j]  -  0.01 * x * (x^2 + y^2)
dy/dt = -b2*y + [other b_{ij} x^i y^j]  -  0.01 * y * (x^2 + y^2)
```
b1, b2 ~ U[0.2, 2.0] (matches Study 16 degradation range). All other coefficients identical to Study 18. Same d = (p+1)(p+2) random parameters.

## Results

### P(limit cycle | d) table

| d  | p | N         | N_spiral    | N_lc   | P(lc)       | ln(1/P) |
|----|---|-----------|-------------|--------|-------------|---------|
| 12 | 2 | 100,000   | 5,003       | 1,092  | 1.092e-02   | 4.517   |
| 20 | 3 | 200,000   | 26,828      | 2,038  | 1.019e-02   | 4.586   |
| 30 | 4 | 500,000   | 106,521     | 4,512  | 9.024e-03   | 4.708   |
| 42 | 5 | 2,000,000 | 491,348     | 17,090 | 8.545e-03   | 4.762   |
| 56 | 6 | 5,000,000 | 1,482,055   | 39,590 | 7.918e-03   | 4.839   |

Total: 7,800,000 samples, 64,322 limit cycles detected.

### Fitting

**Linear fit:** ln(1/P) = 4.4483 + 0.007317 * d, R^2 = 0.964

gamma_poly_degrad = **0.0073**

### Comparison to all four references

| Source             | gamma  | R^2    | Model class                     |
|--------------------|--------|--------|---------------------------------|
| gamma_cusp         | 0.197  | 0.995  | 1D Hill (degradation + sat.)    |
| gamma_Hopf         | 0.175  | 0.9997 | 2D Hill (degradation + sat.)    |
| gamma_poly         | 0.010  | 0.498  | 2D polynomial (no degradation)  |
| gamma_poly_degrad  | 0.007  | 0.964  | 2D polynomial + degradation     |

### Key diagnostic: P at d=12 vs Study 18

| d  | P (poly+degrad) | P (poly, Study 18) | Ratio | Effect           |
|----|------------------|---------------------|-------|------------------|
| 12 | 1.092e-02        | 1.857e-02           | 0.588 | 41% suppression  |
| 20 | 1.019e-02        | 2.079e-02           | 0.490 | 51% suppression  |
| 30 | 9.024e-03        | 1.315e-02           | 0.686 | 31% suppression  |
| 42 | 8.545e-03        | 1.520e-02           | 0.562 | 44% suppression  |
| 56 | 7.918e-03        | 1.310e-02           | 0.604 | 40% suppression  |

Degradation suppresses P by a **uniform ~40-50%** across all d values. It shifts the probability DOWN by a constant factor but does not create d-dependent decay.

### Local gamma between consecutive points

| d range  | gamma_local |
|----------|-------------|
| 12 -> 20 | 0.0086      |
| 20 -> 30 | 0.0122      |
| 30 -> 42 | 0.0045      |
| 42 -> 56 | 0.0054      |

The local gammas are small and non-monotonic. No consistent exponential decay regime.

### Stretched exponent

beta = 0.289 (hitting sublinear regime). The "decay" is almost entirely a logarithmic drift, not exponential.

### Spiral rate comparison

Degradation dramatically reduces the spiral fraction compared to Study 18:

| d  | Spiral % (poly+degrad) | Spiral % (Study 18) |
|----|------------------------|----------------------|
| 12 | 5.0%                   | 23.4%                |
| 20 | 13.4%                  | 30.2%                |
| 30 | 21.3%                  | 36.4%                |
| 42 | 24.6%                  | 38.3%                |
| 56 | 29.6%                  | 42.2%                |

Enforcing negative self-terms makes tr(J) > 0 harder to achieve (since the diagonal contributes -b1, -b2 to the trace). This explains the uniform ~40-50% P suppression: fewer spirals -> fewer limit cycle candidates.

## Interpretation: DEGRADATION ALONE DOES NOT DRIVE DECAY

**Outcome: (A) confirmed.** gamma_poly_degrad = 0.007, which is:
- 96% below gamma_cusp (0.197)
- 96% below gamma_Hopf (0.175)
- Comparable to gamma_poly (0.010) from Study 18

Adding degradation structure to polynomial ODEs does NOT recover the exponential decay. The two effects of degradation are:

1. **Uniform P suppression (~40-50%):** Negative self-terms make unstable spirals harder to achieve, reducing the absolute probability of limit cycles at every d. But this is a constant multiplicative factor, not a d-dependent effect.

2. **No d-dependent decay:** P goes from ~1.1% (d=12) to ~0.8% (d=56), a total factor of 1.4x over 44 dimensions. In contrast, Hill-function models show P dropping by factors of 10^4 over the same d range (gamma ~ 0.18).

The exponential rarity of organized states requires **bounded nonlinearities** (Hill-function saturation), not just degradation architecture. The mechanism is:
- With saturation: each added parameter has bounded effect (output clipped to [0,1]). More parameters = exponentially more ways for the bounded feedback to fail to overcome degradation.
- Without saturation: polynomial terms x^i y^j can grow without bound. More high-degree terms create MORE opportunities for large nonlinear feedback to overcome degradation, partially compensating for the increased dimensional rarity.

## Implications for the framework

### What this confirms
- The exponential decay P ~ exp(-gamma*d) with gamma ~ 0.20 is specific to **bounded/saturating nonlinearities**
- Degradation is necessary but not sufficient for the bridge
- The architectural requirement is: **degradation + saturation** (not just degradation alone)

### What this narrows
- The bridge applies to systems with bounded activation opposing degradation: Hill functions, Michaelis-Menten, sigmoidal responses
- ALL biological regulatory networks have both properties (protein production saturates, proteins degrade), so biological scope is unchanged
- Polynomial ODEs without saturation are outside the bridge's scope, regardless of degradation structure

### Revised scope statement
The bridge equation ln(D) = B + beta_0 holds for systems where:
1. Degradation opposes activation (negative linear self-terms), AND
2. Activation is bounded (output saturates)

Both conditions are universal in biology (saturation kinetics + first-order degradation).

## Computational details
- Script: `poly_degradation_test.py` (modified from `poly_bridge_scaling.py`)
- Only change: x-equation's x-coefficient and y-equation's y-coefficient forced to -U[0.2, 2.0]
- Total samples: 7,800,000 (100k + 200k + 500k + 2M + 5M)
- Total limit cycles: 64,322 (excellent statistics at all d values)
- Total runtime: ~5.4 hours (54s + 216s + 1083s + 6444s + 11809s)
- Seed: 42
- Dependencies: numpy + scipy only
