# Study 26: Sigma Existence Constraint — Results

**Claim:** 26.1 (Grade: YELLOW)
**Date:** 2026-04-06
**Script:** `study26_sigma_existence_constraint.py`

## Question

Does the stability window B in [1.8, 6.0] act as an existence constraint on the physical noise intensity sigma_process, explaining why sigma* ~ sigma_process?

## Result

**Yes, partially.** The stability window constrains sigma_process/sigma* to a band of width sqrt(6.0/1.8) = 1.826, eliminating 93.5% of the prior log-space range (a factor of ~5,500 reduction). This is a necessary but not sufficient condition: it explains the order-of-magnitude match but not the remarkable precision of individual systems (e.g., lake at 2%).

---

## The argument

For a system to exhibit observable noise-driven bistability, its effective barrier-to-noise ratio B_eff = 2*DeltaPhi/sigma_process^2 must be in [1.8, 6.0]. Since B_structural = 2*DeltaPhi/sigma*^2:

```
B_eff = B_structural * (sigma*/sigma_process)^2
```

For B_eff in [1.8, 6.0]:
```
sigma_process/sigma* in [sqrt(B/6.0), sqrt(B/1.8)]
Band width = sqrt(6.0/1.8) = sqrt(3.333) = 1.826 (constant for all systems)
```

---

## Test 1: Allowed sigma band (13 systems)

| System | B | sig_min/s* | sig_max/s* | Band width | 1.0 inside? |
|--------|---|-----------|-----------|-----------|------------|
| Kelp | 1.80 | 0.5477 | 1.0000 | 1.8257 | YES (at boundary) |
| GBP ERM peg | 1.89 | 0.5612 | 1.0247 | 1.8257 | YES |
| Thai Baht peg | 1.90 | 0.5627 | 1.0274 | 1.8257 | YES |
| Tumor-immune | 2.73 | 0.6745 | 1.2315 | 1.8257 | YES |
| Peatland | 3.07 | 0.7153 | 1.3060 | 1.8257 | YES |
| Josephson junction | 3.26 | 0.7371 | 1.3458 | 1.8257 | YES |
| Magnetic nanoparticle | 3.41 | 0.7539 | 1.3764 | 1.8257 | YES |
| Savanna | 3.74 | 0.7895 | 1.4414 | 1.8257 | YES |
| Trop. forest | 4.00 | 0.8165 | 1.4907 | 1.8257 | YES |
| Lake | 4.25 | 0.8416 | 1.5366 | 1.8257 | YES |
| Toggle | 4.83 | 0.8972 | 1.6381 | 1.8257 | YES |
| Diabetes (2D) | 5.54 | 0.9609 | 1.7544 | 1.8257 | YES |
| Coral | 6.04 | 1.0033 | 1.8318 | 1.8257 | NO (at boundary) |

sigma* (ratio = 1.0) is inside or at the boundary for 12/13 systems. Coral (B = 6.04 > 6.0) has sigma* marginally below the lower bound (1.0033 vs 1.0).

## Test 2: Observed sigma ratios vs allowed band

| System | B | sig/s* | Band | Inside? | B_eff | ln(D_eff/D) | Grade |
|--------|---|--------|------|---------|-------|-------------|-------|
| Lake | 4.25 | 1.02 | [0.84, 1.54] | YES | 4.06 | -0.19 | A |
| Savanna | 3.74 | 1.06 | [0.79, 1.44] | YES | 3.33 | -0.41 | C |
| Trop. forest | 4.00 | 0.96 | [0.82, 1.49] | YES | 4.33 | +0.33 | C |
| Coral | 6.04 | 1.34 | [1.00, 1.83] | YES | 3.37 | -2.67 | C |
| Kelp | 1.80 | 1.49 | [0.55, 1.00] | **NO** | 0.81 | -0.99 | C |

4/5 observed systems fall inside the allowed band. Kelp (B = 1.80, at the lower window boundary) has an observed ratio of 1.49, exceeding the upper bound of 1.0. This means kelp's physical noise pushes B_eff = 0.81 below the stability window -- consistent with kelp forests being among the most transient bistable systems (D = 29).

At the actual physical noise, B_eff values for the 4 inside systems range from 3.33 to 4.33 -- all solidly within the stability window.

## Test 3: Asymmetry analysis

The band is centered on sigma/sigma* = 1.0 only at B = sqrt(1.8 * 6.0) = 3.286.

| System | B | Room below 1.0 | Room above 1.0 | Asymmetry | Most free direction |
|--------|---|---------------|---------------|-----------|-------------------|
| Kelp | 1.80 | 0.452 | 0.000 | 0 (no room above) | below sigma* |
| GBP ERM peg | 1.89 | 0.439 | 0.025 | 0.056 | below sigma* |
| Thai Baht peg | 1.90 | 0.437 | 0.027 | 0.063 | below sigma* |
| Tumor-immune | 2.73 | 0.326 | 0.232 | 0.711 | below sigma* |
| Peatland | 3.07 | 0.285 | 0.306 | 1.075 | ~symmetric |
| Josephson junction | 3.26 | 0.263 | 0.346 | 1.315 | ~symmetric |
| Magnetic nanoparticle | 3.41 | 0.246 | 0.376 | 1.529 | above sigma* |
| Savanna | 3.74 | 0.211 | 0.441 | 2.097 | above sigma* |
| Trop. forest | 4.00 | 0.184 | 0.491 | 2.674 | above sigma* |
| Lake | 4.25 | 0.158 | 0.537 | 3.388 | above sigma* |
| Toggle | 4.83 | 0.103 | 0.638 | 6.208 | above sigma* |
| Diabetes (2D) | 5.54 | 0.039 | 0.754 | 19.294 | above sigma* |
| Coral | 6.04 | 0.000 | 0.832 | inf (no room below) | above sigma* |

Key structural prediction: edge-of-window systems (kelp B = 1.80, coral B = 6.04) are the MOST constrained. Kelp cannot tolerate ANY increase in sigma above sigma*; coral cannot tolerate ANY decrease below sigma*. This forces sigma_process/sigma* -> 1.0 at the window edges.

## Test 4: Explanatory power

| Metric | Value |
|--------|-------|
| Prior range (log-space) | 4 OOM (factor 10,000) |
| Posterior (band width) | 0.261 OOM (factor 1.826) |
| Prior eliminated | 93.5% (factor ~5,477) |
| Average constraint share | 95.2% of explanatory work |

Per-system breakdown:

| System | Observed precision (OOM from 1.0) | Band OOM | Extra precision needed | Constraint share |
|--------|----------------------------------|----------|----------------------|-----------------|
| Lake | 0.010 | 0.261 | 0.252 | 93.7% |
| Savanna | 0.025 | 0.261 | 0.236 | 94.1% |
| Trop. forest | 0.017 | 0.261 | 0.244 | 93.9% |
| Coral | 0.126 | 0.261 | 0.135 | 96.5% |
| Kelp | 0.173 | 0.261 | 0.088 | 97.7% |

The constraint does the bulk of the work (93-98% per system), but the remaining precision (especially lake's 0.010 OOM) requires B invariance (Fact 77/86) and structural sigma derivation (Fact 43).

## Test 5: Sensitivity to window boundaries

| Window | Band width | Band OOM | Factor reduced | Prior eliminated |
|--------|-----------|----------|---------------|-----------------|
| [1.8, 6.0] (empirical) | 1.826 | 0.261 | 5,477 | 93.5% |
| [2.0, 5.5] (tighter) | 1.658 | 0.220 | 6,030 | 94.5% |
| [1.5, 7.0] (looser) | 2.160 | 0.335 | 4,629 | 91.6% |
| [1.0, 8.0] (very loose) | 2.828 | 0.452 | 3,536 | 88.7% |

Observed systems inside band: 4/5 at all four window definitions.

The constraint is robust: even the very loose [1.0, 8.0] window eliminates 88.7% of the prior. The tighter [2.0, 5.5] window (excluding kelp and coral edges) gives the strongest constraint but excludes the boundary systems by definition.

---

## Key numbers

- **Band width:** sqrt(6.0/1.8) = 1.8257 (constant, independent of B)
- **Prior elimination:** 93.5% of log-space range (factor ~5,477)
- **Systems with sigma* inside band:** 12/13 (coral marginally outside due to B = 6.04 > 6.0)
- **Observed sigma inside band:** 4/5 ecological systems (kelp outside: B_eff = 0.81)
- **B_symmetric:** sqrt(10.8) = 3.286 (where band is centered on sigma* = 1.0)
- **Average constraint share:** 95.2% of explanatory work across 5 systems

---

## Limitations

1. **Window boundaries are empirical, not derived.** The [1.8, 6.0] range comes from the 13 verified systems. A derivation from first principles (why not [1.5, 7.0]?) is an open question.
2. **The constraint is necessary, not sufficient.** The band width of 1.826 allows sigma_process to differ from sigma* by up to 83%. The lake's 2% match requires additional explanation beyond the existence constraint.
3. **Kelp sigma estimate is Grade C.** The independent sigma_process = 15-20 (midpoint 17.5) is from author estimates, not calibrated field data. The kelp violation (ratio = 1.49, outside band) may partly reflect measurement uncertainty.
4. **Potential circularity.** The stability window is defined by the same systems whose sigma values are being constrained. An independent determination of the window boundaries (e.g., from Kramers theory alone, as in Studies 22/24) would strengthen the argument.
5. **The argument assumes B_structural and B_eff use the same DeltaPhi.** If the potential landscape changes with noise intensity (noise-induced transitions to different attractors), the simple scaling B_eff = B * (sigma*/sigma_process)^2 breaks down.
