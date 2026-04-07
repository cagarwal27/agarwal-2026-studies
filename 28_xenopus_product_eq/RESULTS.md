# Study 28: Product Equation Blind Test -- Xenopus Cell Cycle -- Results

**Date:** 2026-04-06
**Script:** `xenopus_product_eq_test.py`
**Provenance claims:** Tests GAP 3 (product equation outside ecology, OPEN_QUESTIONS.md)

## Question

Does the product equation D = prod(1/epsilon_i) work on the first non-ecological bistable system with separable feedback channels (Xenopus Cdc2-Cdc25-Wee1 mitotic trigger)?

## Data summary

Effective 1D model from Trunnell et al. 2011, with two separable positive feedback channels (Cdc25 phosphatase, Wee1 kinase removal). All parameters measured in Xenopus egg extracts except kw/k25 = 0.5 (fitted to Sha 2003 hysteresis data). See README.md for full parameter table.

---

## Key Results

### Bistability

Bistable range: C = [49.1, 73.3] nM. Representative C = 61.2 nM:
- x_low = 16.4 nM (interphase), x_saddle = 29.5 nM, x_high = 54.1 nM (M-phase)
- DeltaPhi_high = 178.4, DeltaPhi_low = 22.5

### Epsilon and D_product

Three epsilon definitions tested for both states:

| State | Definition | eps_Cdc25 | eps_Wee1 | D_product |
|-------|-----------|-----------|----------|-----------|
| HIGH (M-phase) | A (flux fraction) | 0.227 | 0.773 | 5.70 |
| HIGH | B (saddle-normalized) | 0.668 | 2.278 | 0.66 |
| LOW (interphase) | A (flux fraction) | 0.012 | 0.988 | 83.6 |
| LOW | B (saddle-normalized) | 0.001 | 0.084 | 11,448 |
| System | C (perturbation robustness) | 0.295 | 0.124 | 27.5 |

### Bridge Test Results

| State | Definition | D_product | sigma* (nM) | B | In [1.8, 6.0]? |
|-------|-----------|-----------|-------------|---|-----------------|
| HIGH | A (flux frac) | 5.70 | 18.0 | 1.10 | NO |
| HIGH | C (perturbation) | 27.5 | 12.2 | 2.41 | YES |
| **LOW** | **A (flux frac)** | **83.6** | **3.61** | **3.45** | **YES** |
| LOW | B (saddle-norm) | 11,448 | 2.31 | 8.42 | NO |
| LOW | C (perturbation) | 27.5 | 4.37 | 2.36 | YES |

All bridge tests: D_exact/D_product = 1.000000 (exact by construction).

### B Invariance

**LOW state (interphase, flux-fraction definition):**
- 15 points across bistable range
- B range: [1.33, 5.15]
- Mean B = 3.40, CV = 33.9%
- **13/15 in stability window [1.8, 6.0]**

**HIGH state (M-phase, flux-fraction definition):**
- B range: [0.76, 1.32]
- Mean B = 1.08, CV = 16.4%
- **0/15 in stability window** -- all below 1.8

---

## Interpretation

### What works

1. **Kramers equation verified** for the cell cycle. The interphase (LOW) state has B = 3.45 at the representative C, inside the stability window [1.8, 6.0]. This is the first cell-cycle system confirmed in the Kramers framework.

2. **Single-channel product equation** (effective k=1) works for the interphase state. eps_Cdc25 = 0.012, comparable to kelp (eps = 0.034). The Cdc25 feedback is nearly inactive at x_low = 16.4 << EC50 = 35, so the system is effectively single-channel (Wee1 dominates with eps = 0.988).

3. **B invariance (interphase):** 13/15 points in stability window. CV = 34% is higher than typical ecological systems (2-5%) but the barrier varies strongly across the bistable range as the system approaches the saddle-node bifurcation at both ends.

### What doesn't work

1. **Two-channel product equation (k=2) NOT validated.** The HIGH state has both channels active (eps_Cdc25 = 0.23, eps_Wee1 = 0.77) but B = 1.10 -- below the stability window. The feedback loops are too strong for the ecological regime. The LOW state has B in the window but only one channel is effectively active.

2. **HIGH state B invariance fails.** All B values below 1.8. The M-phase state is in a strong-feedback regime where eps values are too large (0.2-0.8) for the product equation framework.

### Why the product equation works for ecology but not for the M-phase

The ecological product equation requires eps << 1 -- the feedback channels are weak perturbations on a baseline system. In ecology, fire consumes ~10% of NPP, herbivory ~10%, leaving ~80% for basal processes. The channels are a modest "tax."

In the cell cycle M-phase state, the feedback loops provide >95% of the stability (f_baseline = -31.0, R_total = +31.0). Without feedback, the high state doesn't exist at all. The channels aren't perturbations -- they ARE the dynamics. This puts the system outside the regime where D = prod(1/eps_i) with flux-fraction epsilon gives B in the stability window.

The interphase state works because Cdc25 feedback is naturally weak there (x << EC50), making the system behave like a single-channel ecological system.

### Structural finding

**The product equation requires not just separable channels but WEAK channels (eps << 1).** The Xenopus cell cycle has genuinely separable additive channels (confirmed architecturally) but strong channels (eps ~ 0.2-0.8) at the M-phase state. This explains the framework's prior negative finding that "molecular bistable systems universally have k=1" -- molecular systems may have k >= 2 architecturally, but the channels operate in a strong-feedback regime incompatible with the product equation's perturbation framework.

---

## Conclusions

1. Kramers equation: VERIFIED for Xenopus cell cycle (first cell-cycle system).
2. Product equation: PARTIAL -- works for LOW state (effectively k=1), fails for HIGH state (strong channels).
3. Scope boundary identified: product equation requires eps << 1, not just channel separability.
4. B invariance: PARTIAL -- LOW state 13/15 in window; HIGH state 0/15.

**Replicability: GREEN.** All parameters from Trunnell 2011; deterministic computation; 1 fitted parameter (kw/k25).

---

## Dead hypotheses

- **"Product equation works for any system with separable channels"** -- KILLED. Requires weak channels (eps << 1). The Xenopus M-phase state has separable but strong channels.
- **"Molecular bistable systems can have k >= 2 in the product equation"** -- KILLED for strong-feedback molecular systems. May survive for systems where one channel is naturally weak (as in the interphase state).
