# Study 31: Protein Conformational Dynamics -- Stability Window Test

**Claim:** 31.1
**Date:** 2026-04-06

## Purpose

Test whether the stability window B in [1.8, 6.0] applies to protein folding/unfolding dynamics. Protein conformational switching is a textbook Kramers escape problem with exactly known noise (kT), independently measured barriers, and an enormous range of D values -- making it an ideal test domain for the universality of the stability window.

## Data Provenance

All kinetic rates from published literature (25 two-state proteins, 3 chevron datasets). Zero free parameters.

| Parameter | Value | Source |
|-----------|-------|--------|
| k_0 (Kramers speed limit) | 10^6 s^-1 | Kubelka et al. 2004 Curr Opin Struct Biol 14:76 |
| T | 298.15 K (25 C) | Standard conditions |
| B_LOWER, B_UPPER | 1.8, 6.0 | SYSTEMS.md stability window |
| Protein rates (25 systems) | See script | Published chevron analyses (cited per protein) |
| Chevron parameters (3 systems) | See script | Jackson & Fersht 1991; Matouschek et al. 1990; Scalley et al. 1997 |

Stability window bounds [1.8, 6.0] from ../SYSTEMS.md.

## Replication

### Requirements
```
Python 3.8+
pip install numpy
```
No scipy required -- this study uses only basic algebra.

### Commands
```bash
# From repository root:
python3 31_protein_stability_window/study31_protein_stability_window.py

# Or from study folder:
cd 31_protein_stability_window/
python3 study31_protein_stability_window.py
```

### Runtime
< 1 second (pure algebra).

### Expected output
5 tests + summary. Key lines to verify:
- Test 1: 3/25 in window, 22/25 above, B range [2.30, 23.72]
- Test 2: 0/8 proteins with D in [30, 1000] in window
- Test 3: beta_0 mean = 7.4, R^2 = 0.71
- Test 4: Delta_B up to 4.42 (105% of window width)
- Test 5: 3/25 at all k_0 <= 10^7; result robust to pre-exponential

## Files

| File | Purpose |
|------|---------|
| `README.md` | This file -- provenance, replication, data sources |
| `RESULTS.md` | Findings, tables, interpretation, scope boundary analysis |
| `study31_protein_stability_window.py` | Analysis script (pure algebra, no simulation) |
