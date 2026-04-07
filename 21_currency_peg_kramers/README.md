# Study 21: Currency Peg Kramers Test -- Black Wednesday + Thai Baht

**Date:** 2026-04-06

## Purpose

First calibrated Kramers test on human organizational systems. Two independent currency pegs (GBP 1992, THB 1997) -- different continents, decades, volatility regimes -- are tested for whether B falls in the stability window [1.8, 6.0] with zero effective free parameters.

## Data Provenance

### Black Wednesday (GBP/DEM, September 16, 1992)

| Data | Source |
|------|--------|
| ERM central rate (x_peg = 2.95 DEM/GBP) | Bank of England historical spot rates |
| Post-crisis equilibrium (x_float = 2.42 DEM/GBP) | Bank of England spot rates, late Oct 1992 |
| Peg duration (MFPT = 700 days) | Historical record (ERM entry Oct 1990, exit Sep 1992) |
| Relaxation time (tau = 25 days) | Post-crisis rate dynamics |
| FX implied volatility (sigma = 5.0% annualized) | BIS Conference Paper (pre-crisis options) |
| Interest rate data (10% -> 12% -> 15%) | Bank of England |
| Reserve spending (~GBP 27 billion) | Multiple sources |

### Thai Baht (THB/USD, July 1997)

| Data | Source |
|------|--------|
| Peg level (x_peg = 25.0 THB/USD) | Bank of Thailand |
| Post-float full (x_float = 40.0 THB/USD) | Mid-1998 stabilization |
| Post-float initial (x_float = 29.0 THB/USD) | Immediate post-float (2 weeks) |
| Peg duration (MFPT = 4,745 days, 13 years) | Historical record |
| Relaxation times (tau = 14d initial, 180d full) | Post-crisis rate dynamics |
| FX volatility (sigma = 1.2% annualized) | NYU V-Lab GARCH, World Bank annual data |
| BoT reserve data (USD 28B forward interventions) | Bank of Thailand |

### Cusp model reference
| Data | Source |
|------|--------|
| Chen & Chen (2022) cusp params (q=3.16, a=0.54) | J. Applied Statistics 48(13-15), PMC9041743 |

## Replication

### Requirements
- Python 3.8+
- numpy, scipy

### Run commands
```bash
python3 21_currency_peg_kramers/currency_peg_kramers.py
```
Runtime: < 1 minute (Kramers integral evaluation, no Monte Carlo).

### Expected output
- Calibrated cusp parameters for Black Wednesday and Thai Baht
- q-independence verification (CV = 0.0% across q = 1.5-4.0)
- B values for both crises
- B invariance scan (vary asymmetry a at fixed q)
- Stability window placement table
- Cross-crisis consistency check

## Files

| File | Contents |
|------|----------|
| `README.md` | This file -- provenance, replication, data sources |
| `RESULTS.md` | Findings, tables, interpretation, grade justification |
| `currency_peg_kramers.py` | Analysis script (also referenced in `../scripts/README.md`) |
