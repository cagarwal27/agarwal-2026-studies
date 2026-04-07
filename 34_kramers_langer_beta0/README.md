# Study 34: Kramers-Langer beta_0 Prediction for 2D Systems

**Date:** 2026-04-07

## Purpose

Computes the theoretical Kramers-Langer prefactor beta_0 for three 2D systems using only published ODE parameters (Jacobians at equilibrium and saddle). This replaces the circular or semi-independent beta_0 extraction used in Figure 6 (data collapse) with a genuine theoretical prediction.

**Claim reference:** beta_0 = ln(2*pi / (lambda_u * tau * sqrt(det_eq / |det_sad|))) depends only on local curvature at equilibrium and saddle (Kramers-Langer 1969). If beta_0^(KL) matches beta_0^(sweep), the data collapse is a prediction, not circular.

## Data Provenance

| System | Model source | Parameters | Free params |
|--------|-------------|------------|-------------|
| Toggle switch | Gardner TS et al. 2000, Nature 403:339-342 | alpha (bifurcation), n=2 (Hill) | 0 |
| Tumor-immune | Kuznetsov VA et al. 1994, Bull Math Biol 56(2):295-321 | s (bifurcation), 7 published | 0 |
| Diabetes | Topp B et al. 2000, J Theor Biol 206:605-619 | d0 (bifurcation), 8 published | 0 |

BioModels accession: Tumor-immune BIOMD0000000762, Diabetes BIOMD0000000341.

## Replication

### Requirements

- Python 3.8+
- numpy, scipy (no other dependencies)

### Command

```bash
python3 Agarwal/studies/34_kramers_langer_beta0/study34_kramers_langer_beta0.py
```

Runtime: < 10 seconds. Deterministic (no random numbers, no SDE simulation).

## Files

| File | Contents |
|------|----------|
| `README.md` | This file -- provenance, replication, references |
| `RESULTS.md` | Findings, tables, interpretation |
| `study34_kramers_langer_beta0.py` | Self-contained computation script |

## References

- Gardner TS, Cantor CR, Collins JJ. 2000. Construction of a genetic toggle switch in Escherichia coli. Nature 403:339-342.
- Kuznetsov VA, Makalkin IA, Taylor MA, Perelson AS. 1994. Nonlinear dynamics of immunogenic tumors. Bull Math Biol 56(2):295-321.
- Topp B, Promislow K, deVries G, Miura RM, Finegood DT. 2000. A model of beta-cell mass, insulin, and glucose kinetics: pathways to diabetes. J Theor Biol 206:605-619.
- Kramers HA. 1940. Brownian motion in a field of force. Physica 7(4):284-304.
- Langer JS. 1969. Statistical theory of the decay of metastable states. Ann Phys 54(2):258-275.
