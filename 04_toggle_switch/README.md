# Study 04: Toggle Switch

**Date:** 2026-04-04

## Purpose

Tests the Kramers equation for the Gardner et al. 2000 genetic toggle switch -- a coupled, non-separable system. Establishes that the product equation fails for coupled systems while Kramers works with K_CME=1.0.

## Data Provenance

| Script | Parameters | Source |
|--------|-----------|--------|
| `toggle_kramers_test.py` | alpha=10, n=2 | Gardner et al. 2000 toggle switch; no paper citation for alpha=10 specifically |
| `toggle_prefactor_fix.py` | alpha=[5,6,8], Omega=[2,3,5], delta=[0..0.3] | Gardner et al. 2000; 55 configs via CME spectral method |
| `toggle_shortcut.py` | Dense CME scan, 76 points | Gardner et al. 2000 |
| `unification_test.py` | 5 asymmetry tests | Gardner et al. 2000 |
| `step9_toggle_epsilon.py` | 9 epsilon definitions x 4 alpha values | Gardner et al. 2000; Tian-Burrage params cited |

All scripts use the Gardner et al. 2000 toggle switch model with Hill coefficient n=2. Parameters (alpha, Omega, delta) are standard for this model but not individually cited to a specific published table.

**Provenance claims:** 5.4, 9.2, Facts 8-9

## Replication

### Requirements
- Python 3.8+
- numpy, scipy (no other dependencies)

### Commands

All scripts are independent. No dependencies between them. Run from the repository root:

```bash
python3 04_toggle_switch/toggle_kramers_test.py       # Gillespie SSA (~minutes, stochastic)
python3 04_toggle_switch/toggle_prefactor_fix.py      # CME spectral (~seconds, deterministic)
python3 04_toggle_switch/toggle_shortcut.py           # Dense CME scan (~seconds)
python3 04_toggle_switch/unification_test.py          # 5 tests (~seconds)
python3 04_toggle_switch/step9_toggle_epsilon.py      # 36 epsilon tests (~seconds)
```

Runtime: All scripts run in seconds except `toggle_kramers_test.py` (minutes, Gillespie SSA). Only `toggle_kramers_test.py` uses a random seed (np.random.default_rng(seed=42)).

## Files

| File | Contents |
|------|----------|
| `README.md` | This file -- provenance, replication, data sources |
| `RESULTS.md` | Findings, tables, interpretation |
| `toggle_kramers_test.py` | Gillespie SSA comparison (also in `../scripts/`) |
| `toggle_prefactor_fix.py` | K=2.0 to K=1.0 correction (also in `../scripts/`) |
| `toggle_shortcut.py` | D(alpha,Omega) shortcut derivation (also in `../scripts/`) |
| `unification_test.py` | Product eq vs Kramers unification (also in `../scripts/`) |
| `step9_toggle_epsilon.py` | 9 epsilon definitions test (also in `../scripts/`) |
