# Study 10: Architecture Scaling

**Date:** 2026-04-04

## Purpose

Tests f(k) = alpha^k (fraction of random k-channel architectures maintaining bistability). Shows alpha is model-dependent, not universal, and correlates with the fraction of dynamically active equations.

## Data Provenance

### Lake model (core, script 1)
- **van Nes & Scheffer 2007, Table 1:** a=0.326588, b=0.8, r=1.0, q=8, h=1.0

### Savanna model (scripts 2, 4)
- **Xu et al. 2021 (Staver-Levin):** beta=0.39, mu=0.2, nu=0.1, omega0=0.9, omega1=0.2, theta1=0.4, ss1=0.01
- Known equilibria from prior computation (verified): Savanna G=0.5128/T=0.3248, Forest G=0.3134/T=0.6179, Saddle G=0.4155/T=0.4461

### Toggle switch (script 3)
- **Gardner et al., Nature 2000:** alpha_param=8.0, n=2 (Hill coefficient). Symmetric mutual repression.

### Channel generation parameters (all scripts)
- eps_lo=0.005, eps_hi=0.30, eps_budget=0.90 (design choices, not from literature)

**Provenance claims:** 4.1-4.5, 11.4

## Replication

### Requirements
- Python 3.8+
- numpy, scipy

```
pip install numpy scipy
```

### To reproduce
Scripts can be run in any order. No dependencies between them.

```bash
cd 10_architecture_scaling

# Core results
python3 s0_derivation_architecture_scaling.py     # 1D lake, ~minutes (5000 trials x 3 seeds x 6 k values)
python3 alpha_2d_savanna_scaling.py                # 2D savanna, ~minutes (2000 trials x 5 k values)
python3 alpha_2d_toggle_scaling.py                 # 2D toggle, ~minutes (2000 trials x 5 k values)
python3 alpha_2d_savanna_targeted.py               # Targeting test, ~minutes (2000 trials x 5 k values)

# Auxiliary
python3 B_distribution_test.py                     # Barrier distributions, ~minutes (50,000 samples)
python3 step13_cascade_and_barrier_distribution.py  # Channel cascade, ~seconds
python3 step13b_savanna_cascade_and_random_arch.py  # Savanna cascade + random arch, ~minutes
```

All scripts use numpy + scipy only. No matplotlib. No external data files. No import dependencies between scripts. step13b uses scipy.linalg.solve_lyapunov for QPot computation.

## Files

| File | Contents |
|------|----------|
| `README.md` | This file -- provenance, replication, data sources |
| `RESULTS.md` | Findings, tables, interpretation, limitations |
| `s0_derivation_architecture_scaling.py` | Core result: 1D lake model, alpha=0.373, R^2=0.997 |
| `alpha_2d_savanna_scaling.py` | 2D Staver-Levin savanna, alpha=0.844 |
| `alpha_2d_toggle_scaling.py` | 2D Gardner toggle switch, alpha=0.503 |
| `alpha_2d_savanna_targeted.py` | Channel targeting experiment (T eq vs G eq) |
| `B_distribution_test.py` | Barrier/B distribution across random configs |
| `step13_cascade_and_barrier_distribution.py` | Sequential channel cascade on lake model |
| `step13b_savanna_cascade_and_random_arch.py` | Savanna cascade + random lake architectures |
