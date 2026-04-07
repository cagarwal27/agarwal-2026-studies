# Study 01: Core Kramers Duality

**Date:** 2026-04-04

## Purpose

Tests the central claim D_product = D_Kramers for 6 ecological bistable systems: lake, kelp, coral, tropical forest, savanna, and peatland (two independent models). D = prod(1/epsilon_i) from feedback coupling efficiencies must equal D from the exact MFPT integral at a specific noise sigma*.

## Data Provenance

| System | Model source | Key parameters |
|--------|-------------|----------------|
| Lake | van Nes & Scheffer 2007, Table 1 | b=0.8, r=1.0, q=8, h=1.0, a=0.326588 |
| Kelp (Path A) | Arroyo-Esquivel et al. 2024, Table 1 | 17 published params (4D sea-star model) |
| Kelp (Path B) | Constructed 1D; Brey 2001, Ling et al. 2015 | r=0.4, K=668, U_SAD=71, h=FREE |
| Coral | Mumby et al. 2007, Nature 450:98-101 | a=0.1, gamma=0.8, r=1.0, d=0.44, g=0.30 |
| Tropical forest | Touboul, Staver & Levin 2018, Table 1 | alpha=0.6 (modified from 0.2), beta=1.0 (modified from 0.3); 10 other params published |
| Savanna | Staver-Levin / Xu et al. 2021 | beta=0.39, mu=0.2, nu=0.1; DeltaPhi=0.000540 from QPot |
| Peatland (Path C) | Constructed 1D; Clymo 1984, Frolking 2010, Hajek 2011 | NPP=0.20, d_aer=0.05, d_anaer=0.001, m=40 (free), q=8 (free) |
| Peatland (Path A) | Hilbert, Roulet & Moore 2000, Figs 4-6 | 10 published params, c1 free (calibrated ~0.5) |

Epsilon sources: lake (12 candidates tested), kelp (0.034, otter consumption), coral (0.03, Gattuso calcification energetics), forest (0.07 Brando 2014, 0.15 Eltahir 1994), peatland (0.065 Frolking, 0.10 Clymo, 0.33 Hajek).

## Replication

### Requirements
- Python 3.8+
- numpy, scipy (no other dependencies)

### Commands

All scripts are independent. Any execution order is valid. Run from the repository root:

```bash
python3 01_core_kramers_duality/phase1_lake_1d.py                  # Lake 1D Kramers baseline
python3 01_core_kramers_duality/step6_kelp_kramers.py              # Kelp duality (Path A + B)
python3 01_core_kramers_duality/step6b_kelp_immigration.py         # Kelp immigration (K artifact test)
python3 01_core_kramers_duality/step7_coral_kramers.py             # Coral duality
python3 01_core_kramers_duality/step10_tropical_forest_kramers.py  # Tropical forest duality
python3 01_core_kramers_duality/step2_savanna_log_robustness.py    # Savanna CV robustness (analytical)
python3 01_core_kramers_duality/step13_peatland_kramers.py         # Peatland Path C (constructed 1D)
python3 01_core_kramers_duality/step13_peatland_kramers_hilbert.py # Peatland Path A (Hilbert 2000)
```

Runtime: Each script runs in seconds to minutes. Only `step6_kelp_kramers.py` sets a random seed (np.random.seed(42), for Path A initial condition sampling).

## Files

| File | Contents |
|------|----------|
| `README.md` | This file -- provenance, replication, data sources |
| `RESULTS.md` | Findings, tables, interpretation, limitations |
| `phase1_lake_1d.py` | Lake 1D Kramers baseline (also in `../scripts/`) |
| `step6_kelp_kramers.py` | Kelp duality test (also in `../scripts/`) |
| `step6b_kelp_immigration.py` | Kelp immigration K artifact test (also in `../scripts/`) |
| `step7_coral_kramers.py` | Coral duality test (also in `../scripts/`) |
| `step10_tropical_forest_kramers.py` | Tropical forest duality test (also in `../scripts/`) |
| `step2_savanna_log_robustness.py` | Savanna CV robustness sweep (also in `../scripts/`) |
| `step13_peatland_kramers.py` | Peatland Path C: constructed 1D (also in `../scripts/`) |
| `step13_peatland_kramers_hilbert.py` | Peatland Path A: Hilbert 2000 (also in `../scripts/`) |

## References

- Arroyo-Esquivel, Baskett & Hastings 2024. Ecology 105(10):e4453
- Brando et al. 2014. (fire epsilon)
- Brey 2001. (echinoderm P/B compilations)
- Clymo 1984. Phil Trans R Soc B 303:605-654
- Eltahir & Bras 1994. (drought epsilon)
- Freeman et al. 2001. Nature 409:149
- Frolking et al. 2010. Biogeosciences 7:3235-3258
- Hajek et al. 2011. Soil Biol Biochem 43:325-333
- Hilbert, Roulet & Moore 2000. J Ecology 88:230-242
- Ling et al. 2015. PNAS (global regime shift dynamics)
- Loisel et al. 2014. The Holocene 24:1028-1042
- Mumby, Hastings & Edwards 2007. Nature 450:98-101
- Tinker et al. 2019. J Wildl Mgmt 83:1073
- Touboul, Staver & Levin 2018. PNAS 115:E1336-E1345
- van Nes & Scheffer 2007. (lake model, Table 1)
- Xu et al. 2021. (Staver-Levin parameters)
- Yeates et al. 2007. J Exp Biol 210:1960
