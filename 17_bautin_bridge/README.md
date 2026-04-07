# Study 17: Bautin Bridge

**Date:** 2026-04-06

## Purpose

Test whether B invariance and the stability window [1.8, 6.0] hold for bistability between a **fixed point and a limit cycle** (Bautin / subcritical Hopf), not just between two fixed points (cusp). This is the persistence-side companion to Study 16 (Hopf bridge, search side). Combined with Study 16, determines whether the framework is genuinely universal across bifurcation types.

## Data Provenance

No external data. All results are from analytical computation (Part 1) and Monte Carlo sampling (Part 2-3) of ODE systems with internal parameters only.

| Parameter | Value | Source |
|-----------|-------|--------|
| B_cusp (reference) | 2.979, CV = 3.7% | Study 11, cusp_bridge_derivation.py |
| gamma_cusp (reference) | 0.197, R^2 = 0.995 | Study 11, bridge_dimensional_scaling.py |
| gamma_Hopf (reference) | 0.172, R^2 = 0.9997 | Study 16 |
| D_TARGET | 100 | Framework convention (sigma* defined at D=100) |
| Stability window | [1.8, 6.0] | SYSTEMS.md |
| Random seed | 42 | bautin_bridge.py |

### Part 1: Bautin normal form
Amplitude equation: `dr/dt = mu*r + l1*r^3 - r^5`. 300 random (mu, l1) pairs in the bistable regime (l1 > 0, mu in (-l1^2/4, 0)). Barrier computed analytically from the effective potential U(r) = -mu*r^2/2 - l1*r^4/4 + r^6/6. Exact MFPT via Gardiner integral formula (1D). sigma* found by bisection at D_TARGET = 100. B = 2*DeltaPhi/sigma*^2.

### Part 2: P(FP-LC bistable | d) dimensional scaling
2D multi-channel Hill-function ODE (same model class as Studies 11, 16):
```
dx/dt = a1 - b1*x + c1*y + sum_i [r1_i * x^q1_i / (x^q1_i + k1_i^q1_i)]
dy/dt = a2 - b2*y + c2*x + sum_i [r2_i * y^q2_i / (y^q2_i + k2_i^q2_i)]
```
Parameters: d = 6 + 6*n_channels. Parameter distributions: a ~ U(0.05, 2.0), b ~ U(0.2, 3.0), c ~ U(-1.5, 1.5), r ~ U(0.1, 2.0), q ~ U(2.0, 10.0), k ~ U(0.3, 2.0).

| n_channels | d  | N_samples   |
|------------|----|-------------|
| 1          | 12 | 50,000      |
| 2          | 18 | 100,000     |
| 3          | 24 | 200,000     |
| 4          | 30 | 500,000     |
| 5          | 36 | 1,000,000   |
| 6          | 42 | 2,000,000   |

Detection pipeline: vectorized grid evaluation -> Newton refinement -> Jacobian classification (filter for stable spirals near Hopf) -> ODE integration for limit cycle detection -> period consistency check.

### Part 3: B for multi-channel FP-LC configs via SDE escape
SDE escape method for 2D FP-LC transitions. Did not converge -- methodological limitation (escape from FP to LC does not follow 1D barrier-crossing path).

## Replication

### Requirements
- Python 3.8+
- numpy, scipy

### Commands
```bash
cd 17_bautin_bridge/
python3 bautin_bridge.py
```

### Runtime
- Part 1 (normal form): ~5 minutes
- Part 2 (dimensional scaling): several hours (scales with N_samples per d)
- Part 3 (SDE escape): did not converge

## Claims

- **17.1**: B_Bautin from normal form (mean, CV, stability window status)
- **17.2**: gamma_Bautin from P(FP-LC | d) scaling
- **17.3**: B values for multi-channel Bautin configs (incomplete -- SDE method fails for FP-LC geometry)

## Files

| File | Purpose |
|------|---------|
| `README.md` | This file -- provenance, replication, data sources |
| `RESULTS.md` | Findings, tables, interpretation |
| `bautin_bridge.py` | All three parts: normal form B, dimensional scaling, SDE escape |
| `insights.txt` | Session log from original computation |
