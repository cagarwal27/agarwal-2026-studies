# Study 20: Search-Persistence Unification Test -- Flight (v2)

**Date:** 2026-04-06

## Purpose

Test whether the search equation (search time S) and persistence equation (persistence D) can be verified on a single evolutionary innovation -- powered flight -- across 14 clades. The cusp bridge predicts that for noise-driven transitions in integrated (coupled-channel) systems, B = ln(D) - beta_0 should fall in the stability window [1.8, 6.0].

## Data Provenance

### Search side
- k values and log10(S) from Fact 70 / step12e (intermediate k computation)
- tau_search values from fossil record and molecular clock estimates

### Persistence side -- Vertebrates
| Data | Source |
|------|--------|
| Bird flight loss rates (r_loss = 11.70e-4/Myr) | Sayol et al. 2020, *Sci Adv* 6:eabb6095 |
| Bird diversification rates (r_spec = 0.16/Myr, PD ~ 81,200 lin-Myr) | Jetz et al. 2012, *Nature* 491:444 |
| Rail flightlessness (32/140 spp, r_spec ~ 0.4/Myr) | Kirchman 2012, *The Auk*; Gaspar et al. 2020 |
| Bat phylogenetic diversity (0/1287 flightless, PD = 7,549 lin-Myr) | Upham et al. 2019, *PLOS Biol* (MCC tree) |
| Pterosaur fossil diversity (0/200 spp, PD ~ 2,000 lin-Myr) | Longrich et al. 2018, *PLOS Biol* |

### Persistence side -- Insect orders
- Flightlessness fractions: Roff 1990, *Ecological Monographs* 60:389-421 (Table 8)
- Diversification rates: Condamine et al. 2016, *Scientific Reports* 6:19208
- Supplementary: Ikeda et al. 2012 (beetle ~10%), Bank et al. 2022 (phasmid 52.9%)

## Replication

### Requirements
- Python 3.8+
- numpy, scipy

### Run commands
```bash
python3 20_fire_tree_flight/fire_tree_flight.py
```
Runtime: < 1 minute (dimensional analysis on tabulated data, no Monte Carlo).

### Expected output
- Combined D table for 14 clades (sorted by D)
- Stability window membership for each clade
- Modularity gradient summary
- D = 1 threshold crossings
- Sensitivity analysis (tau_search, f uncertainty, bird D methods, bat Poisson bound)

## Files

| File | Contents |
|------|----------|
| `README.md` | This file -- provenance, replication, data sources |
| `RESULTS.md` | Findings, tables, interpretation, grade justification |
| `fire_tree_flight.py` | Analysis script (also referenced in `../scripts/README.md`) |
