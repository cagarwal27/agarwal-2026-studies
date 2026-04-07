# Study 05: HIV Post-Treatment Control

**Date:** 2026-04-06

## Purpose

Tests the product equation cross-domain for HIV using the Conway-Perelson 2015 5D model (T, L, I, V, E). Computes eps_CTL, D_product, and Omega* via both SDE and Freidlin-Wentzell MAP methods.

## Data Provenance

All 18 model parameters from Conway & Perelson 2015 PNAS, SI Table S1. 5D state variables: T (target cells), L (latent), I (infected), V (virus), E (effector/CTL). The model is implemented in `conway_perelson_model.py` and duplicated inline in scripts 4-7.

**Provenance claims:** 5.1, Fact 44

## Replication

### Requirements
- Python 3.8+
- numpy, scipy, matplotlib

```
pip install numpy scipy matplotlib
```

`epsilon_duality_test.py` uses matplotlib to write PNG plots. This is the only matplotlib dependency in the study.

### Import dependencies

```
conway_perelson_model.py        (module, no main)
    |
    v
epsilon_duality_test.py         (imports conway_perelson_model)
    |                           (writes 3 PNG plots, uses matplotlib)
    v
omega_gap_closure.py            (imports conway_perelson_model AND epsilon_duality_test)
```

Scripts 4-7 duplicate the model inline and are fully self-contained.

### To reproduce

**Run order matters for scripts 1-3** due to import chain.

```bash
cd 05_hiv/

# Step 1: Model module (no output, defines parameters)
python3 conway_perelson_model.py

# Step 2: SDE scan (requires conway_perelson_model.py in same directory)
python3 epsilon_duality_test.py       # ~minutes, stochastic, writes 3 PNGs

# Step 3: Dense scan (requires both prior scripts in same directory)
python3 omega_gap_closure.py          # ~minutes, 500 trajectories

# Steps 4-7: Self-contained (any order)
python3 step13_hiv_structural_omega.py   # 5D MAP (~seconds, deterministic)
python3 noise_source_mapping_hiv.py      # CLE decomposition (~seconds)
python3 step13b_3d_reduced_map.py        # 3D/4D/5D MAP (~seconds)
python3 step13c_de_map.py               # DE optimization (~seconds)
```

### Replicability grades

**Overall: YELLOW**

| Script | Grade | Notes |
|--------|-------|-------|
| `conway_perelson_model.py` | GREEN | All 18 params from published SI Table S1 |
| `epsilon_duality_test.py` | GREEN | Seeded RNG (42+Omega), published params |
| `omega_gap_closure.py` | YELLOW | PREV_DATA hardcoded from prior runs |
| `step13_hiv_structural_omega.py` | YELLOW | Deterministic MAP but model duplicated inline |
| `noise_source_mapping_hiv.py` | GREEN | Self-contained noise decomposition |
| `step13b_3d_reduced_map.py` | YELLOW | Seeded (42+r*13) but model duplicated inline |
| `step13c_de_map.py` | YELLOW | Seeded (42) but model duplicated inline |

`omega_gap_closure.py` is YELLOW because it hardcodes PREV_DATA from prior `epsilon_duality_test.py` runs rather than computing them fresh. To fully replicate, run `epsilon_duality_test.py` first and verify the hardcoded values match.

## Files

| File | Contents |
|------|----------|
| `README.md` | This file -- provenance, replication, data sources |
| `RESULTS.md` | Findings, tables, interpretation |
| `conway_perelson_model.py` | Model definition module (all 18 params from SI Table S1) |
| `epsilon_duality_test.py` | SDE D(Omega) scan, finds Omega* where D_exact=D_product |
| `omega_gap_closure.py` | Dense scan Omega=2500-4500, 500 trajectories, bootstrap CI |
| `step13_hiv_structural_omega.py` | 5D Freidlin-Wentzell MAP |
| `noise_source_mapping_hiv.py` | CLE noise decomposition |
| `step13b_3d_reduced_map.py` | MAP in 3D/4D/5D reductions |
| `step13c_de_map.py` | Differential evolution MAP optimization |
