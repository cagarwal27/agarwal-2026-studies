# Study 04: Toggle Switch -- Results

**Date:** 2026-04-04
**Scripts:** `04_toggle_switch/*.py`

**Provenance claims:** 5.4, 9.2, Facts 8-9

## Question

Does the Kramers equation work for the Gardner et al. 2000 genetic toggle switch (a coupled, non-separable 2D system)? Does the product equation D = prod(1/epsilon_i) hold for coupled systems?

## Parameters

**Toggle switch model** (all scripts):
- Gardner et al. 2000 toggle switch
- Hill coefficient: n=2
- alpha values tested: 5, 6, 8, 10
- Omega values tested: 2, 3, 5
- Asymmetry: delta = 0..0.3

| Parameter | Value | Source |
|-----------|-------|--------|
| alpha | 10 (script 1), [5,6,8] (script 2) | Standard toggle switch; no paper citation for alpha=10 specifically |
| n (Hill) | 2 | Gardner et al. 2000 |
| Omega | [2,3,5] | Standard scan range |
| delta | 0..0.3 | Asymmetry sweep |

Tian-Burrage params are cited in step9_toggle_epsilon.py.

---

## Key Results

| Result | Value | Method |
|--------|-------|--------|
| K_CME | 1.0 (corrected from 2.0) | CME spectral method, 55 configs |
| Toggle shortcut accuracy | ~3% typical | Dense CME scan, 76 points |
| Epsilon product equation | FAILS for coupled systems | 9 definitions x 4 alpha values |

### 1. K Prefactor Correction (toggle_prefactor_fix.py)

K_CME=1.0 is correct for the 2D toggle; the earlier K=2.0 was an artifact of applying a 1D Kramers formula to a 2D system. Tested across 55 configurations: alpha=[5,6,8], Omega=[2,3,5], delta=[0..0.3]. Deterministic CME spectral method.

### 2. Toggle Shortcut (toggle_shortcut.py)

The toggle shortcut D(alpha,Omega) = exp(a + S*Omega) provides a closed-form approximation with R^2 > 0.99. Dense CME scan with 76 points. 7 candidate S(alpha) fits tested.

### 3. Product Equation Fails (step9_toggle_epsilon.py)

All 9 epsilon definitions fail across all 4 alpha values, confirming the product equation is invalid for coupled (non-separable) systems. This is a scope boundary: the product equation D = prod(1/epsilon_i) requires separable feedback channels (eps << 1 per channel), which the toggle switch does not have.

### 4. Kramers vs Gillespie (toggle_kramers_test.py)

Gillespie SSA comparison at alpha=10, n=2. Uses np.random.default_rng(seed=42). Kramers equation (with K_CME=1.0) matches SSA MFPT.

### 5. Unification Test (unification_test.py)

Tests product equation vs Kramers unification for toggle with asymmetry delta. 5 independent tests. Product equation fails; Kramers holds across asymmetry range.

---

## Replicability Assessment

**Overall: YELLOW**

| Script | Grade | Notes |
|--------|-------|-------|
| `toggle_kramers_test.py` | YELLOW | No paper citation for alpha=10 |
| `toggle_prefactor_fix.py` | YELLOW | Deterministic CME spectral method, self-contained |
| `toggle_shortcut.py` | YELLOW | Dense CME scan, deterministic |
| `unification_test.py` | YELLOW | 5 tests, self-contained |
| `step9_toggle_epsilon.py` | YELLOW | Tian-Burrage params cited but not individually sourced |

All scripts produce YELLOW because the toggle switch parameters are standard but not individually cited to a specific published table.

---

## Interpretation

The toggle switch provides a critical scope boundary for the framework:

1. **Kramers equation works** for the coupled 2D toggle switch with K_CME=1.0. The barrier-to-noise ratio B and the MFPT are well-defined and computable via the CME spectral method.

2. **Product equation fails** because the toggle switch has coupled, non-separable feedback channels. There is no meaningful way to decompose D into a product of independent 1/epsilon_i terms. This confirms the product equation requires separable (weakly coupled) channels with eps << 1.

3. **K=1.0 vs K=2.0:** The factor-of-2 correction arises from the 1D vs 2D Kramers prefactor. In 2D, the escape rate involves the ratio of eigenvalues at the saddle and minimum in both dimensions, which changes the prefactor relative to the 1D formula.

4. **The shortcut formula** D(alpha,Omega) = exp(a + S*Omega) with R^2 > 0.99 provides a practical closed-form for the toggle switch without running full CME calculations.
