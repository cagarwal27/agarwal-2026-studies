# Study 23: Phage Lambda k* Validation

**Date:** 2026-04-06

## Purpose

Count regulatory interactions in a known bistable switch (phage lambda) and compare to the framework's complexity scaling predictions. Tests whether lambda's observed feedback loop count k ~ 5-8 matches the predicted k* ~ 5-10 for a system with d ~ 27-50 ODE parameters.

## Data Provenance

### Lambda regulatory network
| Data | Source |
|------|--------|
| Core decision circuit (CI, Cro, CII, CIII, N, Q) | Ptashne 2004, *A Genetic Switch* 3rd ed. CSHL Press |
| Promoter regulation, operator binding, cooperativity | Oppenheim et al. 2005, *Annu Rev Genet* 39:409-429 |
| Extended network interactions | Court et al. 2007, *J Bacteriol* 189:298-304 |
| Spontaneous induction rate (~10^-5 per cell per gen) | Little 2010, *J Bacteriol* 192:6064-6076 |

### Published ODE models
| Model | State vars | Params | Source |
|-------|-----------|--------|--------|
| Shea & Ackers 1985 | 2 | ~15 | PNAS 82:8506 |
| Reinitz & Vaisnys 1990 | 2 | ~10 | J Theor Biol 145:295 |
| Santillan & Mackey 2004 | 4 | ~27 | Biophys J 86:75 |
| Arkin et al. 1998 (stochastic) | ~25 | 50-100+ | Genetics 149:1633 |

### Framework parameters
| Parameter | Value | Source |
|-----------|-------|--------|
| gamma (cusp bridge) | 0.197 | Study 11 / Fact 73 |
| alpha (architecture scaling) | 0.373 | Study 11 |

## Replication

### Requirements
- Python 3.8+
- numpy, scipy

### Run commands
```bash
python3 23_lambda_k_count/study23_lambda_k_count.py
```
Runtime: < 1 minute (arithmetic computation on tabulated data, no Monte Carlo).

### Expected output
- Interaction count summary (core ~30, extended ~43)
- ODE parameter count summary (d ~ 27-50)
- Independent feedback loop identification (k ~ 5-8)
- Framework prediction comparison (predicted k* ~ 5-10 vs observed k ~ 5-8)
- Product equation consistency check (D ~ 10^4-10^5)

## Files

| File | Contents |
|------|----------|
| `README.md` | This file -- provenance, replication, data sources |
| `RESULTS.md` | Findings, tables, interpretation, assessment |
| `study23_lambda_k_count.py` | Analysis script (also referenced in `../scripts/README.md`) |
