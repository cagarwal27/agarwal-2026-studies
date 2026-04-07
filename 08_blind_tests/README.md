# Study 08: Blind Tests (New Domains)

**Date:** 2026-04-06

## Purpose

Two blind prediction tests in physics domains outside the framework's training set (superconducting electronics, nanomagnetism). Both have zero free parameters. These are the strongest tests of the framework.

## Data Provenance

### Josephson Junction
Overdamped RCSJ model: dphi/dt = gamma - sin(phi), V(phi) = -cos(phi) - gamma*phi. Stewart 1968, McCumber 1968. Normalized equation with 0 free parameters. Experimental references: Devoret et al. PRL 55, 1908 (1985); Martinis et al. PRB 35, 4682 (1987).

### Magnetic Nanoparticle
Stoner-Wohlfarth uniaxial model: V(theta) = sin^2(theta) - 2h*cos(theta). Stoner & Wohlfarth 1948, Neel-Brown overdamped dynamics. Normalized equation with 0 free parameters. Experimental reference: Wernsdorfer et al. PRL 78, 1791 (1997).

**Provenance claims:** 11.1, 11.2

## Replication

### Requirements
- Python 3.8+
- numpy, scipy

```
pip install numpy scipy
```

All scripts self-contained. No import dependencies between scripts. No external data files. No random seeds (deterministic, exact MFPT integrals on N=200,000 grids).

### To reproduce

Scripts can be run in any order. No dependencies between them.

```bash
cd 08_blind_tests/

python3 blind_test_josephson_junction.py      # ~seconds, deterministic, N=200,000 grid
python3 blind_test_magnetic_nanoparticle.py    # ~seconds, deterministic, N=200,000 grid
```

### Replicability grades

**Overall: GREEN**

| Script | Grade | Notes |
|--------|-------|-------|
| `blind_test_josephson_junction.py` | GREEN | 0 free params, deterministic (exact MFPT integrals), all structure analytic |
| `blind_test_magnetic_nanoparticle.py` | GREEN | 0 free params, deterministic (exact MFPT integrals), strongest asymmetry test |

Both scripts are deterministic. No random seeds are used (no stochastic sampling). All results follow from exact MFPT integrals evaluated on fine grids (N=200,000).

## Files

| File | Contents |
|------|----------|
| `README.md` | This file -- provenance, replication, data sources |
| `RESULTS.md` | Findings, tables, interpretation, limitations |
| `blind_test_josephson_junction.py` | Overdamped RCSJ Josephson junction (K=0.56) |
| `blind_test_magnetic_nanoparticle.py` | Stoner-Wohlfarth uniaxial nanoparticle (K=0.57) |
