# Study 18: Polynomial Bridge Scaling

## Question
Is gamma ~ 0.20 (the exponential decay rate of P(organized | d parameters)) a geometric universal, or an artifact of Hill-function saturation?

## Model
2D polynomial ODE with weak cubic damping (eps = 0.01):
```
dx/dt = sum_{i+j<=p} a_{ij} * x^i * y^j  -  0.01 * x * (x^2 + y^2)
dy/dt = sum_{i+j<=p} b_{ij} * x^i * y^j  -  0.01 * y * (x^2 + y^2)
```
Random parameters: d = (p+1)(p+2) coefficients. No Hill functions, no saturation.

## Results

| d  | p | N         | N_spiral | N_lc   | P(lc)       | ln(1/P) |
|----|---|-----------|----------|--------|-------------|---------|
| 12 | 2 | 100,000   | 23,406   | 1,857  | 1.857e-02   | 3.986   |
| 20 | 3 | 100,000   | 30,158   | 2,079  | 2.079e-02   | 3.873   |
| 30 | 4 | 200,000   | 72,815   | 2,629  | 1.315e-02   | 4.332   |
| 42 | 5 | 500,000   | 191,339  | 7,601  | 1.520e-02   | 4.186   |
| 56 | 6 | 1,000,000 | 421,736  | 13,095 | 1.310e-02   | 4.336   |

Total: 1,900,000 samples, 27,261 limit cycles detected.

## Fitting

**Linear (decay regime, d >= 20):**
- gamma_poly = 0.0099, R^2 = 0.498
- gamma_poly / gamma_cusp = 0.050 (95% below)

**Comparison table:**

| Source      | gamma  | R^2    | Model class       |
|-------------|--------|--------|-------------------|
| gamma_cusp  | 0.197  | 0.995  | 1D Hill function  |
| gamma_Hopf  | 0.175  | 0.9997 | 2D Hill function  |
| gamma_poly  | 0.010  | 0.498  | 2D polynomial     |

**Local gamma between consecutive points:**
- d=20 -> 30: 0.046
- d=30 -> 42: -0.012 (P increased)
- d=42 -> 56: 0.011

**Stretched exponent:** beta = 0.10 (hitting lower bound). The decay is essentially flat, not stretched-exponential.

## Interpretation: NO EXPONENTIAL DECAY FOR POLYNOMIALS

P(limit cycle | d) is approximately **constant at 1.3-2.1%** across d = 12 to 56. There is no exponential decay. The fitted gamma (0.010) has R^2 = 0.50, meaning it explains essentially no variance beyond the mean.

This is the **MODEL-CLASS-DEPENDENT** outcome: the exponential decay P ~ exp(-gamma * d) with gamma ~ 0.20 is specific to Hill-function (sigmoidal/saturating) parameterizations. It is NOT a universal geometric property of parameter spaces.

## Why polynomials differ from Hill functions

Three structural differences explain the flat P(lc):

1. **Spiral rate increases with d.** The fraction of samples with unstable spirals grows: 23% (d=12), 30% (d=20), 36% (d=30), 38% (d=42), 42% (d=56). More polynomial terms create more opportunities for complex eigenvalues. Hill functions don't have this property — their bounded outputs limit Jacobian flexibility.

2. **Polynomials are unbounded near equilibria.** Hill functions saturate at [0,1], so each added channel contributes at most O(1) to the dynamics. Polynomial terms x^i * y^j can contribute arbitrarily, creating richer local dynamics that partially offset the rarity effect.

3. **The cubic damping provides only global containment.** It doesn't affect dynamics near equilibria (contribution ~0.01 at |x|~1), so local bifurcation structure is entirely determined by the random polynomial. Hill models have intrinsic saturation that constrains local behavior.

## Implications for the framework

### What this narrows
- The bridge equation ln(D) = B + beta_0 and the search equation S ~ exp(gamma * d) are NOT universal across all parameterized ODE classes.
- gamma ~ 0.20 is a property of Hill/sigmoidal model classes, not of codimension-1 bifurcation geometry per se.

### What remains universal
- Studies 11, 16, 17 still show gamma_cusp ~ gamma_Hopf ~ gamma_Bautin ~ 0.18-0.20 within the Hill-function class. Universality across BIFURCATION TYPES (cusp, Hopf, Bautin) is intact.
- D = MFPT/tau and D = 1 threshold are model-independent (they don't depend on parameterization).
- B invariance is also model-independent (property of the potential landscape, not parameterization).

### Revised scope
The bridge connects persistence to rarity for systems with **bounded nonlinearities** (sigmoidal, saturating, Hill-type). This includes all biological regulatory networks (which universally use saturation kinetics), but excludes arbitrary polynomial dynamical systems. The framework's biological scope is unchanged; its mathematical scope is narrower than "all ODE model classes."

## Computational details
- Detection: vectorized Newton (16 init conditions, 20 iters) + Jacobian spiral check + vectorized batch RK4 (dt=0.1, T=300, save every 0.2)
- Coefficient scaling: degree-k terms drawn from U[-r_k, r_k] with r_k chosen so total contribution per degree is O(1) at |x|,|y| ~ 1
- Total runtime: ~2.7 hours (single core)
- Seed: 42
- Dependencies: numpy + scipy only
