# Study 05: HIV Post-Treatment Control

**Date:** 2026-04-06
**Scripts:** `conway_perelson_model.py`, `epsilon_duality_test.py`, `omega_gap_closure.py`, `step13_hiv_structural_omega.py`, `noise_source_mapping_hiv.py`, `step13b_3d_reduced_map.py`, `step13c_de_map.py`

## Question

Does the product equation D = prod(1/epsilon_i) hold cross-domain for HIV? What is Omega* (the system size where D_exact = D_product), and do independent methods (SDE scan vs Freidlin-Wentzell MAP) agree?

## Data

Conway-Perelson 2015 PNAS 5D HIV model with all 18 parameters from SI Table S1. 5D state variables: T (target cells), L (latent), I (infected), V (virus), E (effector/CTL).

## Key Results

| Result | Value | Method |
|--------|-------|--------|
| Omega* (SDE) | 2296 mL | SDE D(Omega) scan, 500 trajectories |
| Omega* (structural) | 2293 mL | 5D Freidlin-Wentzell MAP |
| SDE vs structural match | 0.1% | Independent methods agree |
| eps_CTL | 0.077 | Product equation from 5D model |
| D_product | 13.0 | Product of 1/epsilon_i |
| Dominant noise source | Virus production | CLE noise decomposition |

## Parameters and Sources

**Conway-Perelson 2015 5D HIV model** (`conway_perelson_model.py`):

All 18 parameters from Conway & Perelson 2015 PNAS, SI Table S1:

| Parameter | Description | Value | Source |
|-----------|-------------|-------|--------|
| All 18 params | T, L, I, V, E dynamics | See conway_perelson_model.py | Conway-Perelson 2015 PNAS SI Table S1 |

5D state variables: T (target cells), L (latent), I (infected), V (virus), E (effector/CTL).

**Note:** Scripts 4-7 duplicate the model parameters inline rather than importing `conway_perelson_model.py`. The parameter values are identical but exist in multiple locations.

## Script Details

| # | Script | What it does | Status |
|---|--------|-------------|--------|
| 1 | `conway_perelson_model.py` | Model definition module. All 18 params from Conway-Perelson 2015 PNAS SI Table S1. | GREEN |
| 2 | `epsilon_duality_test.py` | SDE D(Omega) scan, finds Omega* where D_exact=D_product. IMPORTS conway_perelson_model. Uses np.random.default_rng(42+Omega). Writes 3 PNG plots. | GREEN |
| 3 | `omega_gap_closure.py` | Dense scan Omega=2500-4500. 500 trajectories. IMPORTS conway_perelson_model AND epsilon_duality_test. Omega*~2296. Bootstrap CI. | YELLOW |
| 4 | `step13_hiv_structural_omega.py` | 5D Freidlin-Wentzell MAP. Omega*_structural=2293 (0.1% match to SDE). Duplicates model inline (no import). Deterministic. | YELLOW |
| 5 | `noise_source_mapping_hiv.py` | CLE noise decomposition. Dominant noise source: virus production. Duplicates model inline. | GREEN |
| 6 | `step13b_3d_reduced_map.py` | MAP in 3D/4D/5D reductions. Duplicates model inline. np.random.seed(42+r*13). | YELLOW |
| 7 | `step13c_de_map.py` | Differential evolution MAP optimization. Duplicates model inline. seed=42. | YELLOW |

## Interpretation

- The product equation D = prod(1/epsilon_i) successfully applies to HIV, a domain entirely outside ecology. D_product = 13.0 with eps_CTL = 0.077 as the dominant sensitivity.
- Two independent methods (SDE stochastic simulation and deterministic Freidlin-Wentzell MAP) agree on Omega* to within 0.1% (2296 vs 2293 mL), providing strong cross-validation.
- CLE noise decomposition identifies virus production as the dominant noise source, consistent with the biological understanding that viral burst dynamics dominate stochastic fluctuations.

## Conclusions

The product equation holds cross-domain for HIV with zero free parameters (all 18 model parameters from published literature). The 0.1% agreement between SDE and structural MAP methods validates both computational approaches. This is the first test of the product equation outside ecology.
