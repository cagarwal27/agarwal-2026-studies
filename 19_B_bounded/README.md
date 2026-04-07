# Study 19: B Bounded Derivation

**Date:** 2026-04-05
**Provenance claim:** 19.1
**Grade:** GREEN

## Purpose

Derives why B = 2*DeltaPhi/sigma*^2 is confined to [1.8, 6.0] across 11 systems spanning 6 domains. Shows B is bounded by cusp scale invariance (structural constraint), and the specific range [1.8, 6.0] is set by the observed range of D values (selection). Neither alone is sufficient.

## Data Provenance

All parameters are internal to the cusp normal form. Zero free parameters. Zero external data.

| Parameter | Value | Source |
|-----------|-------|--------|
| Cusp normal form | dx/dt = -x^3 + a*x + b | Standard (Strogatz 1994) |
| K_cusp | 0.558 | From cusp_bridge_derivation.py (Study 11) |
| D_target | 100 (default), also scanned at 29, 50, 200, 500, 1111 | Ecological range |
| a range | 0.5 to 10.0 (scale invariance test) | Arbitrary (result is a-independent) |
| phi range | (0, pi/3) = (0, 1.047) | Exact: full bistable region |
| Random seed | 42 | Reproducible |

## Replication

### Requirements

```
Python 3.8+
pip install numpy scipy
```

No external data files. No matplotlib. Self-contained.

### Commands

```bash
python3 B_bounded_derivation.py
```

### Runtime

~30 seconds. 11 computation steps.

### Expected output

Key lines to verify:
- Step 2: "CONFIRMED: B varies by < 0.1% across a = 0.5 to 10.0 at each phi"
- Step 3: "B_exact range: [2.6xxx, 3.2xxx]"
- Step 7: "In [1.8, 6.0]: 200/200 = 100.0%"
- Final summary: "B in [2.609, 3.230]" and "Union ... B in [1.540, 5.659]"

### Script structure

| Step | What it computes |
|------|-----------------|
| 1 | phi parameterization and root structure |
| 2 | a-independence of B (scale invariance proof) |
| 3 | Full B(phi) curve via exact MFPT at 25 phi values |
| 4 | Kramers vs exact accuracy (interior vs near-fold) |
| 5 | B(phi) at D = 29, 50, 100, 200, 500, 1111 |
| 6 | Mapping 7 empirical systems to cusp phi |
| 7 | 200 random cusp instances (reproducing B_cusp = 2.979) |
| 8 | Kramers analytical B(phi) monotonicity and bounds |
| 9 | Sensitivity to D_target and K |
| 10 | Complete derivation summary |
| 11 | Stability window explanation |

## Files

| File | Description |
|------|-------------|
| `README.md` | This file (replication instructions) |
| `RESULTS.md` | Full results, interpretation, and provenance |
| `B_bounded_derivation.py` | Main script: 11-step B boundedness derivation |
