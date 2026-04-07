# Study 03: Bridge Algebraic Tests

**Date:** 2026-04-04

## Purpose

Tests structural properties of the bridge identity ln(D) = B + beta_0: proves the bridge is transcendental (not algebraic), validates the Hermite approximation, shows per-channel barrier decomposition fails, establishes K universality, and shows epsilon is structural (not stochastic). Documents Dead Hypotheses #2, #3, #4.

## Data Provenance

| Script | System/Model | Source |
|--------|-------------|--------|
| `bridge_q3_symbolic.py` | Lake at q=3 | van Nes & Scheffer 2007; b=0.8, r=1.0, h=1.0, K_CORR=1.12 (framework) |
| `hermite_validation.py` | Lake (q=3..20), double-well, Schlogl | Textbook models; K_CORR=1.12 (framework) |
| `test1_barrier_epsilon.py` | 2-channel lake model | van Nes & Scheffer 2007; K_CORR=1.12 (framework) |
| `test3_k_universality.py` | Lake, double-well, Schlogl | Standard textbook models |
| `test3_k_deep_barrier.py` | Deep barrier limit (B up to 50) | Textbook; Berglund 2011 J Phys A |
| `test5_barrier_scaling.py` | Lake varying q | van Nes & Scheffer 2007 |
| `test5_refinement.py` | Lake near q_critical | van Nes & Scheffer 2007 |
| `pathc_dynamic_epsilon.py` | SDE simulation (lake-like) | Optionally uses numba; no random seed |

**Provenance claims:** Facts 12-17

## Replication

### Requirements
- Python 3.8+
- numpy, scipy
- sympy (required for `bridge_q3_symbolic.py` only)
- numba (optional for `pathc_dynamic_epsilon.py`; falls back to pure numpy)

### Commands

All scripts are independent. No import dependencies between scripts. Run from the repository root:

```bash
python3 03_bridge_algebraic_tests/bridge_q3_symbolic.py    # Symbolic analysis (requires sympy)
python3 03_bridge_algebraic_tests/hermite_validation.py    # Hermite approximation test
python3 03_bridge_algebraic_tests/test1_barrier_epsilon.py # Per-channel decomposition test
python3 03_bridge_algebraic_tests/test3_k_universality.py  # K universality across models
python3 03_bridge_algebraic_tests/test3_k_deep_barrier.py  # K at deep barriers
python3 03_bridge_algebraic_tests/test5_barrier_scaling.py  # q shapes barrier
python3 03_bridge_algebraic_tests/test5_refinement.py      # Fine-grained q near q_crit
python3 03_bridge_algebraic_tests/pathc_dynamic_epsilon.py  # SDE: epsilon structural (stochastic)
```

Runtime: All scripts run in seconds except `pathc_dynamic_epsilon.py` (minutes, SDE simulation). No random seeds except `pathc_dynamic_epsilon.py` which uses `np.random` without seeding (stochastic output).

## Files

| File | Contents |
|------|----------|
| `README.md` | This file -- provenance, replication, data sources |
| `RESULTS.md` | Findings, dead hypotheses, interpretation |
| `bridge_q3_symbolic.py` | Symbolic analysis at q=3 (also in `../scripts/`) |
| `hermite_validation.py` | Hermite approximation validation (also in `../scripts/`) |
| `test1_barrier_epsilon.py` | Per-channel barrier decomposition test (also in `../scripts/`) |
| `test3_k_universality.py` | K universality across models (also in `../scripts/`) |
| `test3_k_deep_barrier.py` | K at deep barriers (also in `../scripts/`) |
| `test5_barrier_scaling.py` | How q shapes the barrier (also in `../scripts/`) |
| `test5_refinement.py` | Fine-grained q near q_critical (also in `../scripts/`) |
| `pathc_dynamic_epsilon.py` | SDE: epsilon is structural (also in `../scripts/`) |
