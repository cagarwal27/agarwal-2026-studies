# Study 10: Architecture Scaling -- Results

**Date:** 2026-04-04
**Scripts:** `s0_derivation_architecture_scaling.py`, `alpha_2d_savanna_scaling.py`, `alpha_2d_toggle_scaling.py`, `alpha_2d_savanna_targeted.py`, `B_distribution_test.py`, `step13_cascade_and_barrier_distribution.py`, `step13b_savanna_cascade_and_random_arch.py`

## Question

Does f(k) = alpha^k hold for the fraction of random k-channel architectures that maintain bistability, and how does alpha depend on the model?

## Data Summary

Three published bistable models (1D lake, 2D toggle, 2D savanna) tested with random channel additions at k=1-6 (lake) or k=1-5 (savanna/toggle). 5000 trials per k per seed (lake, 3 seeds) or 2000 trials per k (savanna/toggle, 1 seed). Auxiliary scripts explore barrier distributions and cascades.

---

## Key Results

| Model | Dimension | alpha | R^2 | Dynamically active fraction | Grade |
|-------|-----------|-------|------|----------------------|-------|
| Lake (van Nes & Scheffer) | 1D | 0.373 | 0.997 | 1/1 (100%) | GREEN |
| Toggle (Gardner) | 2D | 0.503 | 0.9998 | 2/2 (100%) | GREEN |
| Savanna (Staver-Levin) | 2D | 0.844 | 0.984 | 1/2 (50%) | GREEN |
| Savanna, target T eq | 2D | 0.740 | -- | targeted | GREEN |
| Savanna, target G eq | 2D | 0.913 | -- | targeted | GREEN |

---

## Interpretation

- **Core finding:** f(k) = alpha^k holds across all tested models with R^2 > 0.98. alpha is model-dependent, ranging from 0.373 (1D lake) to 0.844 (2D savanna).
- **Dynamically active interpretation:** alpha correlates with the fraction of equations that are dynamically active for bistability. When all equations participate in the bistability mechanism (lake: 1/1, toggle: 2/2), alpha is lower (~0.37-0.50). When only one equation is dynamically active (savanna: omega sigmoid in T equation only), alpha is higher (~0.84) because channels on the non-mechanism equation rarely destroy bistability.
- **Targeting test (script 4):** Confirms the dynamically active interpretation. Forcing all channels onto the T equation (mechanism) drops alpha from 0.844 to 0.740. Forcing all channels onto the G equation (non-mechanism) raises alpha to 0.913.
- **Barrier distribution (scripts 5-7):** Among bistable random configurations, n <= 2 parameters in ~65% of bistable channels. Barrier heights follow approximately lognormal distributions.

---

## Detailed Parameters

### s0_derivation_architecture_scaling.py (Core)

Lake model parameters (van Nes & Scheffer 2007, Table 1):

| Parameter | Value | Source |
|-----------|-------|--------|
| a (P loading) | 0.326588 | van Nes & Scheffer 2007 |
| b (P loss rate) | 0.8 | van Nes & Scheffer 2007 |
| r (max recycling) | 1.0 | van Nes & Scheffer 2007 |
| q (Hill coefficient) | 8 | van Nes & Scheffer 2007 |
| h (half-saturation) | 1.0 | van Nes & Scheffer 2007 |
| x_clear | 0.409217 | Computed from model |
| x_saddle | 0.978 | Computed from model |
| x_turbid | 1.634 | Computed from model |

Channel generation parameters:

| Parameter | Value | Notes |
|-----------|-------|-------|
| eps_lo | 0.005 | Lower bound for channel epsilon |
| eps_hi | 0.30 | Upper bound for channel epsilon |
| eps_budget | 0.90 | Maximum total epsilon (reject if exceeded) |
| N_trials | 5000 | Per k value per seed |
| Seeds | 42, 137, 2718 | 3 independent seeds |
| k_values | 1, 2, 3, 4, 5, 6 | Channel counts tested |
| N_grid | 50000 | Root-finding grid resolution |

### alpha_2d_savanna_scaling.py

Staver-Levin parameters (Xu et al. 2021):

| Parameter | Value | Source |
|-----------|-------|--------|
| beta (herbivory) | 0.39 | Xu et al. 2021 |
| mu (grass colonization) | 0.2 | Xu et al. 2021 |
| nu (tree mortality) | 0.1 | Xu et al. 2021 |
| omega0 (fire, low grass) | 0.9 | Xu et al. 2021 |
| omega1 (fire, high grass) | 0.2 | Xu et al. 2021 |
| theta1 (sigmoid midpoint) | 0.4 | Xu et al. 2021 |
| ss1 (sigmoid steepness) | 0.01 | Xu et al. 2021 |

Known equilibria (from prior computation, verified):
- Savanna: G=0.5128, T=0.3248
- Forest: G=0.3134, T=0.6179
- Saddle: G=0.4155, T=0.4461

N_trials = 2000 per k. seed = 42.

### alpha_2d_toggle_scaling.py

Gardner toggle switch (Gardner et al., Nature 2000):

| Parameter | Value | Source |
|-----------|-------|--------|
| alpha_param | 8.0 | Gardner et al. 2000 (well inside bistable range) |
| n (Hill coefficient) | 2 | Gardner et al. 2000 |

Symmetric mutual repression: du/dt = alpha/(1+v^n) - u, dv/dt = alpha/(1+u^n) - v. Both equations dynamically active. N_trials = 2000, seed = 42.

### B_distribution_test.py

| Parameter | Value | Notes |
|-----------|-------|-------|
| N_samples | 50,000 | Per ensemble |
| N_MFPT_subset | 200 | For B computation |
| D_target | 100.0 | Standard target |
| Ensemble A | Random (a,q), fixed b=0.8, r=1.0, h=1.0 | a in [0.05, 0.80], q in [2.0, 20.0] |
| Ensemble B | Random multi-channel | Multiple Hill channels |
| Ensemble C | Fully random | All parameters drawn randomly |

seed = 42 (np.random.seed).

---

## Replicability Assessment

**Overall: GREEN (core results), YELLOW (auxiliary scripts)**

| Script | Grade | Notes |
|--------|-------|-------|
| `s0_derivation_architecture_scaling.py` | GREEN | Published lake model, 3 seeds, R^2=0.997 |
| `alpha_2d_savanna_scaling.py` | GREEN | Published savanna model, seeded (42) |
| `alpha_2d_toggle_scaling.py` | GREEN | Published toggle model, seeded (42) |
| `alpha_2d_savanna_targeted.py` | GREEN | Published savanna model, seeded (42) |
| `B_distribution_test.py` | YELLOW | Parameter ranges are design choices (eps_lo, eps_hi, eps_budget) |
| `step13_cascade_and_barrier_distribution.py` | YELLOW | Channel shapes (K1=0.5, K2=2.0, K3=1.0) are design choices |
| `step13b_savanna_cascade_and_random_arch.py` | YELLOW | Combines two models; QPot barrier from prior computation |

The core result (alpha=0.373, R^2=0.997) is robust: 3 independent seeds, 5000 trials each, published model parameters. The alpha value is model-dependent (0.37-0.91 across 3 models), which is correctly identified as a feature, not a bug.

YELLOW scripts are auxiliary: they explore barrier distributions and cascades where parameter ranges (eps bounds, channel shapes) are design choices that affect quantitative but not qualitative results.

---

## Limitations

1. **alpha is model-dependent:** alpha = 0.373 is specific to the 1D lake model. Other models give different alpha values (0.503 for toggle, 0.844 for savanna). There is no single universal alpha. The open question is whether alpha = 1/e has a deeper information-theoretic origin.
2. **Channel generation protocol:** The eps_budget = 0.90 cutoff and eps_lo/eps_hi bounds affect the absolute value of alpha. The exponential decay f(k) = alpha^k is robust to these choices, but alpha's value is not.
3. **Exponential vs. other decay:** R^2 > 0.98 for exponential fits, but the data spans only k=1-6 (lake) or k=1-5 (savanna/toggle). Power-law or stretched exponential fits are not ruled out at larger k.
4. **2D fixed-point finding:** The 2D scripts (savanna, toggle) use Newton's method from multiple initial conditions. False negatives (missing bistable configurations due to failed convergence) would bias alpha upward. This is mitigated by using perturbation grids around known fixed points + global grids.
5. **Dynamically active interpretation:** The correlation between alpha and dynamically active fraction is based on 3 models. More models would strengthen or refute this pattern.
