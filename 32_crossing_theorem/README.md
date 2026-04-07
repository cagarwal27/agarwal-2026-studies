# Study 32: Crossing Theorem (sigma* = sigma_process as Topological Necessity)

**Date:** 2026-04-06

## Purpose

Demonstrate that the empirical match sigma* = sigma_process is not an algebraic identity but a topological consequence of bistable dynamics plus observational selection. Route A (Study 31b, `route_a_eigenvalue_test.py`) showed Q(a) = sigma*(a) x |lambda_eq(a)| is NOT constant across the bistable range (CV = 42-106%), ruling out a global algebraic explanation. This study shows the match is instead guaranteed at the operating point by the intermediate value theorem (IVT).

## Background

For any bistable system with a fold (or transcritical) bifurcation:

- **sigma\*(a)** = the noise intensity that produces the observed D = MFPT/tau. Near the fold where the observed state disappears, the barrier DeltaPhi -> 0, so sigma* = sqrt(2 DeltaPhi / B) -> 0.
- **sigma_env(a)** = SD(forcing) / |lambda_eq(a)|. Near the same fold, |lambda_eq| -> 0, so sigma_env -> infinity.

Since sigma\* goes from positive (interior) to 0 (fold) and sigma_env goes from finite (interior) to infinity (fold), by the IVT they **must cross at least once**. At the crossing, sigma\* = sigma_env, which means B_physical = B_bridge, and B is bounded in [1.8, 6.0] by B invariance.

Systems exhibiting noise-driven transitions must operate near this crossing. Away from it: either noise overwhelms the barrier (too transient to observe as persistent) or the barrier overwhelms noise (no transitions observed).

For physics systems (JJ, nanoparticle), sigma_env = sqrt(2kT/E_scale) is CONSTANT (a horizontal line), and sigma\*(gamma) still sweeps from 0 to a maximum. The crossing is guaranteed for any positive sigma_thermal below max(sigma\*).

## Data Provenance

All model equations, parameters, and equilibrium-finding code are copied from existing verified scripts:

| System | Source Script | D_target | SD(forcing) | Operating Point |
|--------|-------------|----------|-------------|-----------------|
| Lake (q=8) | route_a_eigenvalue_test.py | 200 | 0.139 | a = 0.35 |
| Kelp | route_a_eigenvalue_test.py | 29.4 | 2.26 | p = 64 |
| Savanna | route_a_eigenvalue_test.py | 100 | 0.0021 | beta = 0.39 |
| Coral (Mumby) | route_a_eigenvalue_test.py | 1111.1 | 0.011 | g = 0.30 |
| Josephson junction | blind_test_josephson_junction.py | 100 | (thermal) | calibrated |
| Magnetic nanoparticle | blind_test_magnetic_nanoparticle.py | 100 | (thermal) | calibrated |

SD(forcing) values are from Fact 43 in `../FACTS.md`. Operating points are from `../SYSTEMS.md`.

Zero free parameters. All model equations from published literature (see SYSTEMS.md for citations).

## Method

### Part 1: Ecology Systems (4 systems)

At N=50 evenly spaced points across the bistable range of the bifurcation parameter:

1. Compute equilibria (x_eq, x_sad) and eigenvalue lambda_eq = |f'(x_eq)|.
2. Compute DeltaPhi (barrier height) via numerical quadrature.
3. Compute sigma\*(a) via bisection: find sigma where D_exact(sigma) = D_target, using the exact 1D Fokker-Planck MFPT integral (Gardiner formula, 80,000-point grid).
4. Compute sigma_env(a) = SD(forcing) / |lambda_eq(a)|.
5. Compute ratio r(a) = sigma\*(a) / sigma_env(a).

Then:
- Find the crossing point a_cross where r(a) = 1 (linear interpolation between grid points, then recompute sigma\* at the interpolated point for accurate B_cross).
- Report B at the crossing.
- Compute bandwidth: fraction of bistable range where r(a) is within [1/1.5, 1.5] and [1/2, 2].
- Check whether the known operating point falls within the bandwidth.

### Part 2: Physics Systems (JJ + nanoparticle)

At N=50 points across the bistable range, with all equilibria, barriers, and eigenvalues computed analytically:

1. Compute sigma\*(gamma) at D_target = 100 using the exact MFPT integral (200,000-point grid).
2. sigma_env = sigma_thermal (CONSTANT for a given temperature).
3. Sweep sigma_thermal and find crossing for each value.
4. Calibrate: find sigma_thermal such that B at the crossing matches the empirical B from SYSTEMS.md (JJ: 3.26, nanoparticle: 3.41).

### Part 3: IVT Verification

For each system, explicitly verify:
1. Near the fold: sigma\* -> 0 and |lambda_eq| -> 0 (so sigma_env -> infinity).
2. In the interior: sigma\* > 0 and sigma_env is finite.
3. Therefore r goes from >> 1 (interior) to 0 (fold), guaranteeing a crossing by IVT.

### Fold Locations

| System | Fold end | What happens |
|--------|----------|-------------|
| Lake | High a | Clear-lake state merges with saddle |
| Kelp | Low p (p -> 40) | U=0 state loses stability (transcritical) |
| Savanna | High beta | Savanna state merges with saddle |
| Coral | Low g (g -> g_lower) | M=0 state loses stability (transcritical) |
| Josephson junction | gamma -> 1 | Well disappears |
| Nanoparticle | h -> 1 | Shallow well disappears |

## Replication

### Requirements

```
Python 3.8+
pip install numpy scipy
```

No other dependencies (no matplotlib, no numba, no external data files).

### Run Command

```bash
cd 32_crossing_theorem/
python3 crossing_theorem_test.py
```

### Runtime

~3-4 minutes on a modern laptop (M-series Mac). The vectorized Savanna and Coral MFPT integrals dominate; physics systems are fast (analytic equilibria).

### Expected Output

The script prints tables for all 6 systems, then a summary. Key lines to verify:

```
=== SUMMARY ===
System                cross  B_cross  BW_1.5x    BW_2x  Op in band?
LAKE (q=8)           0.3103     4.81    52.0%    72.0%          YES
KELP                61.2755     2.25    12.0%    22.0%          YES
SAVANNA              0.3922     4.06    50.0%    80.0%          YES
CORAL                0.3156     6.00    20.0%    36.0%          YES
JOSEPHSON JUNCTION   0.6455     3.26    38.8%      --- (calibrated)
NANOPARTICLE         0.5161     3.41    38.8%      --- (calibrated)
```

And the interpretation block:
```
1. All ecology operating points inside 1.5x bandwidth: YES
2. B_cross in [1.8, 6.0] for all systems: YES
3. Typical 1.5x bandwidth: 35.3% (> 20%, generic not fine-tuned)
4. Physics bandwidth (38.8%) vs ecology (33.5%): comparable
```

Full raw output is preserved in `RESULTS.md`.

## Results

### 1. Crossing Exists for All 6 Systems

| System | Crossing Point | B at Crossing | In Stability Window? |
|--------|---------------|---------------|---------------------|
| Lake (q=8) | a = 0.310 | 4.81 | YES (1.8-6.0) |
| Kelp | p = 61.3 | 2.25 | YES |
| Savanna | beta = 0.392 | 4.06 | YES |
| Coral | g = 0.316 | 6.00 | YES |
| Josephson junction | gamma = 0.645 | 3.26 | YES |
| Nanoparticle | h = 0.516 | 3.41 | YES |

### 2. All Ecology Operating Points Inside Bandwidth

| System | Operating Point | Distance to Crossing | In 1.5x Band? | In 2x Band? |
|--------|----------------|---------------------|----------------|-------------|
| Lake | a = 0.35 | 10.9% of bistable width | YES | YES |
| Kelp | p = 64 | 5.7% | YES | YES |
| Savanna | beta = 0.39 | 2.0% | YES | YES |
| Coral | g = 0.30 | 7.3% | YES | YES |

### 3. Bandwidth Is Generic, Not Fine-Tuned

| System | 1.5x Bandwidth | 2x Bandwidth |
|--------|---------------|-------------|
| Lake | 52.0% | 72.0% |
| Kelp | 12.0% | 22.0% |
| Savanna | 50.0% | 80.0% |
| Coral | 20.0% | 36.0% |
| JJ | 38.8% | -- |
| Nanoparticle | 38.8% | -- |
| **Mean** | **35.3%** | |

The 1.5x bandwidth averages 35.3% of the bistable range. This is wide: a system does not need to be fine-tuned to sit near the crossing. Kelp is the narrowest (12%), consistent with its being the least stable system (lowest B).

### 4. IVT Conditions Verified

For all 6 systems, the script confirms:
- **Near the fold:** sigma\* < 1e-2 and sigma_env > 0.03 (ecology) or sigma\* < 0.06 (physics). Ratio -> 0.
- **In the interior:** ratio > 1.7 (ecology) or sigma\* = max at low gamma/h (physics).
- The ratio sweeps continuously from > 1 to ~0, guaranteeing at least one crossing.

### 5. Physics Systems: Crossing for Any Temperature

For JJ: crossing exists for any sigma_thermal in (0, 1.09). B at crossing = 3.26 +/- 0.04 regardless of where the crossing falls (B invariance, CV = 0.35%).

For nanoparticle: crossing exists for any sigma_thermal in (0, 0.77). B at crossing = 3.41 +/- 0.07 (B CV = 2.02%).

### 6. B Invariance Across the Crossing Sweep

JJ B CV = 0.35%. Nanoparticle B CV = 2.02%. B is effectively constant across the entire bistable range, confirming that the crossing value of B is not sensitive to the crossing location.

## Interpretation

The sigma\* = sigma_process match is a **topological consequence** of bistable dynamics plus observational selection:

1. **Existence (IVT):** For any bistable system, sigma\*(a) and sigma_env(a) must cross because sigma\* -> 0 and sigma_env -> infinity at the fold.
2. **Boundedness:** At the crossing, B_physical = B_bridge, which is bounded in [1.8, 6.0] by B invariance (Studies 19, 22, 24).
3. **Selection:** Systems exhibiting noise-driven transitions must operate near the crossing. Away from it, the system is either too transient (noise >> barrier) or too stable (no transitions observed).
4. **Genericity:** The bandwidth is wide (35% of bistable range on average), so operating near the crossing is the generic condition, not a fine-tuned one.

This resolves the Route A negative result: Q(a) is NOT constant, so there is no algebraic identity making sigma\* = sigma_env everywhere. But the crossing theorem shows the match is **guaranteed at the operating point** by topology, not algebra.
