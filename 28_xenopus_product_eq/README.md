# Study 28: Product Equation Blind Test -- Xenopus Cell Cycle

**Date:** 2026-04-06
**Provenance claims:** Tests GAP 3 (product equation outside ecology, OPEN_QUESTIONS.md)

## Purpose

Test the product equation D = prod(1/epsilon_i) on the first non-ecological bistable system with separable feedback channels: the Xenopus Cdc2-Cdc25-Wee1 mitotic trigger. This is a blind test -- parameters are from published experiments, not fitted to the framework.

## Data Provenance

**Model:** Effective 1D model from Trunnell et al. 2011, *Molecular Cell* 43:550-560.

| Parameter | Value | Source | Measured? |
|-----------|-------|--------|-----------|
| n_Cdc25 | 11 | Trunnell 2011 Fig 3 | YES (egg extracts) |
| n_Wee1 | 3.5 | Trunnell 2011 Fig 5 | YES (egg extracts) |
| EC50_Cdc25 | 35 nM | Trunnell 2011 Fig 3 | YES |
| EC50_Wee1 | 30 nM | Trunnell 2011 Fig 5 | YES |
| bkgd_Cdc25 | 0.2 | Trunnell 2011 | YES |
| bkgd_Wee1 | 0.2 | Trunnell 2011 | YES |
| kw/k25 | 0.5 | Trunnell 2011 Eq 6 | Fitted to Sha 2003 hysteresis |
| k25 | 1.0 | Normalization | Sets timescale |

**Free parameters: 1** (kw/k25 ratio, constrained by experimental bistability data from Sha et al. 2003, PNAS 100:975-980 and Pomerening et al. 2003, Nature Cell Biol 5:346-351).

**Cross-reference:** Full multi-species model available as Novak-Tyson 1993 (BIOMD0000000107, BioModels database). The effective 1D model is the validated quasi-steady-state reduction.

## Model

The M-phase trigger has two positive feedback loops operating through independent enzymes on the same substrate (Cdk1 Tyr15):

- **Channel 1 (Cdc25):** Active Cdk1 phosphorylates Cdc25, activating it. Active Cdc25 dephosphorylates Cdk1 Tyr15 (positive feedback via phosphatase).
- **Channel 2 (Wee1):** Active Cdk1 phosphorylates Wee1, inactivating it. Reduced Wee1 activity means less Cdk1 Tyr15 phosphorylation (double-negative = positive feedback via kinase removal).

These channels are additive and separable in the effective 1D drift. Cdc25 and Wee1 do not directly regulate each other; both are regulated only by active Cdk1.

**ODE:**
```
dx/dt = k25*(C-x)*f_25(x) - kw*x*f_w(x)

x = active Cdk1 (nM), C = total cyclin-Cdk1 (nM, bifurcation parameter)
f_25(x) = 0.2 + x^11/(35^11 + x^11)        [Cdc25 fractional activity]
f_w(x)  = 0.2 + 30^3.5/(30^3.5 + x^3.5)    [Wee1 fractional activity]
```

## Replication

### Dependencies
```
Python 3.8+
pip install numpy scipy
```
numpy + scipy only. Self-contained. No import dependencies.

### Running
```bash
python3 28_xenopus_product_eq/xenopus_product_eq_test.py
```

Expected runtime: ~60 seconds, deterministic. Key output to verify:
- Bistable range: C = [49.1, 73.3] nM
- LOW state B = 3.45 at representative C = 61.2 nM (inside [1.8, 6.0])
- HIGH state B = 1.10 (below stability window)
- Bridge identity: D_exact/D_product = 1.000000

## Files

| File | Description |
|------|-------------|
| `README.md` | This file (replication instructions) |
| `RESULTS.md` | Full results, interpretation, and conclusions |
| `xenopus_product_eq_test.py` | Full blind test: bistability, epsilon (3 definitions), MFPT, bridge, B invariance |

## References

1. Trunnell NB, Poon AC, Kim SY, Ferrell JE Jr. (2011) Ultrasensitivity in the regulation of Cdc25C by Cdk1. *Mol Cell* 43:550-560.
2. Sha W, Moore J, Chen K, Lassaletta AD, Yi CS, Tyson JJ, Sible JC. (2003) Hysteresis drives cell-cycle transitions in Xenopus laevis egg extracts. *PNAS* 100:975-980.
3. Pomerening JR, Sontag ED, Ferrell JE Jr. (2003) Building a cell cycle oscillator: hysteresis and bistability in the activation of Cdc2. *Nature Cell Biol* 5:346-351.
4. Ferrell JE Jr. (2008) Feedback regulation of opposing enzymes generates robust, all-or-none bistable responses. *Curr Biol* 18:R244-R245.
5. Novak B, Tyson JJ. (1993) Numerical analysis of a comprehensive model of M-phase control in Xenopus oocyte extracts and intact embryos. *J Cell Sci* 106:1153-1168. BioModels: BIOMD0000000107.
