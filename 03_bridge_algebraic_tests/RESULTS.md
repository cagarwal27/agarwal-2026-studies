# Study 03: Bridge Algebraic Tests -- Results

**Date:** 2026-04-04
**Scripts:** `03_bridge_algebraic_tests/*.py`

**Provenance claims:** Facts 12-17

## Question

What are the structural properties of the bridge identity ln(D) = B + beta_0? Is it algebraic? Does it decompose per-channel? How does K behave? Is epsilon structural or stochastic?

## Fact Summary

| Fact | Statement | Script | Result |
|------|-----------|--------|--------|
| 12 | K values resolved: K~0.55 (parabolic), 0.34-0.36 (anharmonic), ->1/2 (deep barrier) | `test3_k_universality.py`, `test3_k_deep_barrier.py` | Confirmed |
| 13 | Epsilon is structural, not stochastic (<8% shift across 10 noise levels) | `pathc_dynamic_epsilon.py` | Confirmed |
| 14 | Per-channel barrier decomposition fails (356% mean error, 9/260 correct) | `test1_barrier_epsilon.py` | Dead Hypothesis #2 |
| 15 | Growth exponent q creates the barrier; DeltaPhi saturates from zero at q_crit | `test5_barrier_scaling.py`, `test5_refinement.py` | Dead Hypothesis #3 |
| 16 | Bridge is NOT algebraic -- transcendental obstruction proven | `bridge_q3_symbolic.py` | Dead Hypothesis #4 |
| 17 | Hermite approximation connects barrier to boundary data (error <1% for Deltax<0.3) | `hermite_validation.py` | Confirmed |

---

## Key Results

### 1. Bridge NOT Algebraic (Dead Hypothesis #4)

**Script:** `bridge_q3_symbolic.py`

At q=3, DeltaPhi = [polynomial part] + [transcendental part (ln, arctan)]. Polynomial part is -350x DeltaPhi, transcendental part is +352x DeltaPhi. Barrier is 0.3% residual from massive cancellation. By Lindemann-Weierstrass theorem, sigma*^2 is irreducibly transcendental.

Lake params: b=0.8, r=1.0, h=1.0, K_CORR=1.12.

### 2. Hermite Approximation (Fact 17)

**Script:** `hermite_validation.py`

DeltaPhi ~ Deltax^2/12*(|lambda_eq|+|lambda_sad|). Error analysis:
- Double-well: 0.00% (exact)
- Lake near fold: 0.05%
- Lake q=8: 16.5% (far from fold)
- Savanna prediction: sigma*=0.01699 vs 0.017 (0.1%)

Valid for Deltax < 0.3 (error < 1%). K_CORR=1.12 framework-derived.

### 3. Per-Channel Barrier Decomposition FAILS (Dead Hypothesis #2)

**Script:** `test1_barrier_epsilon.py`

Tested: -ln(1/eps_i) should map to DeltaPhi_i. Only 9/260 calibrations (3.5%) show correct ratio. Mean error 356%, max error 4946%. The bridge is a duality, not a term-by-term decomposition. 2-channel lake model.

### 4. K Universality (Fact 12)

**Script:** `test3_k_universality.py`, `test3_k_deep_barrier.py`

K = D_exact / D_Kramers across models:
- K ~ 0.55 for parabolic wells (lake, double-well, savanna)
- K = 0.34-0.36 for anharmonic wells (kelp, tropical forest)
- K -> 1/2 analytically in deep-barrier limit (B up to 50)

References Berglund 2011 J Phys A. Grid resolution study included.

### 5. q Creates the Barrier (Dead Hypothesis #3)

**Script:** `test5_barrier_scaling.py`, `test5_refinement.py`

DeltaPhi(q) = 0.112*(1-exp(-0.096*(q-2.81)^1.33)), R^2=0.999.
- q < q_crit: search phase (no barrier)
- q > q_crit: metastable phase (barrier grows from zero)
- q_crit = 3.2 (analytic, found via bisection in refinement script)

Three-term decomposition falsified: D_gradient varies 6 OOM while D fixed. Critical scaling exponents computed.

### 6. Epsilon Structural, Not Stochastic (Fact 13)

**Script:** `pathc_dynamic_epsilon.py`

SDE simulation shows epsilon shifts <8% across CV=0.10-0.80 while D varies from 5 to 5x10^22. D_product/D_exact = 0.895 at exactly one sigma*, astronomically wrong at other sigma values. Optionally uses numba (fallback to pure numpy). No random seed (stochastic output).

---

## Dead Hypotheses Documented

| # | Hypothesis | Evidence | Script |
|---|-----------|----------|--------|
| 2 | Per-channel barrier: -ln(epsilon_i) = 2*DeltaPhi_i/sigma^2 | 356% mean error, 9/260 correct | `test1_barrier_epsilon.py` |
| 3 | Three-term decomposition D = D_channels x D_barrier x D_gradient | D_gradient varies 6 OOM while D fixed | `test5_barrier_scaling.py` |
| 4 | Algebraic proof of the bridge | Transcendental obstruction; Lindemann-Weierstrass | `bridge_q3_symbolic.py` |

---

## Replicability Assessment

**Overall: YELLOW**

- `bridge_q3_symbolic.py` requires **sympy** (not numpy/scipy only)
- `pathc_dynamic_epsilon.py` optionally uses **numba** (fallback to pure numpy), no random seed (stochastic)
- `hermite_validation.py` and `test1_barrier_epsilon.py` use K_CORR=1.12 (framework-derived constant)
- Other 4 scripts are GREEN (standard textbook models, numpy/scipy only)

All scripts self-contained. No local imports.

## Interpretation

The bridge identity ln(D) = B + beta_0 is a transcendental relationship that cannot be reduced to an algebraic expression -- this is proven, not conjectured. The Hermite approximation provides a practical closed-form estimate (error < 1% near the fold bifurcation) but cannot replace the exact integral.

The per-channel decomposition failure (Dead Hypothesis #2) is significant: it means the bridge operates at the whole-system level, not as a sum of independent channel contributions. Similarly, the failure of the three-term decomposition (Dead Hypothesis #3) confirms that D is not decomposable into independent multiplicative factors.

K universality shows the anharmonicity correction factor depends primarily on well shape (parabolic vs anharmonic, boundary vs interior) rather than system identity, converging to the analytically expected 1/2 in the deep-barrier limit.
