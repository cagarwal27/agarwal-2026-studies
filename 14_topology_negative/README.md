# Study 14: Topology (Negative Results)

## What this tests

Tests whether topological properties of regulatory networks (driver count, feedback vertex set, loop count) predict persistence D. All fail: topology alone is insufficient. This motivates the mu-analysis derivation (S30) which shows coupling STRENGTHS, not just graph structure, determine D.

## Provenance claims

- 9.3: F2 = (1/eps)^n fails (42 ecosystems, r = -0.22)
- 9.4: Driver count null (R^2 = 0.019 for anchors)
- 9.5: Loop count wrong sign (r = -0.575)
- 9.6: FVS fails (R^2 = 0.051)
- 7.3: Override resistance = k (weak chain, anchor R^2 = 0.019)

## Models and data sources

### 19 biological regime shift types (compute_S28, compute_fvs, compute_S29)
- **Driver counts:** Rocha et al. 2015 bipartite matrix (Figshare 1472951)
- **Causal loop diagrams:** Rocha et al. 2018, read from `data/rocha2018/RS_CLD_2018.csv`
- **D values:** 7 framework anchors (kelp D=29, savanna D=100, lake D=200, coral D=1111, trop. forest D=95, peatland D=30, toggle D=9175) + 12 estimated from tau_reg/tau_collapse ratios
- **Override resistance scores:** Framework-defined (kelp=1, savanna=2, lake=3, peatlands=3, salt marsh=3, coral=4, trop. forest=5)
- **Species richness S:** Per-system values (**NO PRIMARY SOURCE** -- not individually cited in scripts)

### 42 Cooper 2020 ecosystems (s23_trophic_coupling_test)
- **Collapse data:** Cooper et al. 2020 (42 ecosystem collapse events with area and tau_collapse)
- **Trophic transfer efficiencies:** Eddy et al. 2021 (biome-specific epsilon values)
- **Hypothesis tested:** D = (1/epsilon)^n where n = effective trophic levels

## Scripts

| Script | What it computes | Key output to verify |
|--------|-----------------|---------------------|
| `s23_trophic_coupling_test.py` | F2=(1/eps)^n on 42 Cooper ecosystems | `r = -0.22` (no predictive power) |
| `compute_S28.py` | Driver count vs D (6 regression models) | `R^2 = 0.019 (anchor only, N=7)` |
| `compute_S29.py` | Stabilizing loop count vs D (19 regime shifts) | `r = -0.575` (wrong sign) |
| `compute_fvs.py` | FVS via integer LP, edge-cut vs D | `FVS R^2 = 0.051` |

### Run order
```bash
pip install numpy scipy
python3 s23_trophic_coupling_test.py  # independent (data hardcoded)
python3 compute_S28.py               # independent (data hardcoded)
python3 compute_S29.py               # reads data/RS_CLD_2018.csv (bundled)
python3 compute_fvs.py               # reads data/RS_CLD_2018.csv (bundled)
```

### Import dependencies
- `compute_fvs.py` and `compute_S29.py` read `data/RS_CLD_2018.csv` (Rocha et al. 2018), bundled in this study folder.
- `s23_trophic_coupling_test.py` has all data hardcoded (42 Cooper ecosystems + Eddy TTE values).
- No local script imports between files.

## Replicability assessment

### YELLOW

- **s23_trophic_coupling_test.py:** Epsilon values assigned by biome from Eddy et al. 2021 TTE data (documented per-ecosystem in script). Cooper collapse data (42 ecosystems) with area and tau_collapse. All data hardcoded. Uses only math + json (no numpy/scipy).
- **compute_S28.py:** Driver counts sourced from Rocha 2015 Figshare. D values for 7 anchors from framework; 12 estimated from tau ratios (each annotated). Species richness values not individually sourced.
- **compute_S29.py:** Enumerates all simple directed cycles in Rocha 2018 CLDs. Classifies stabilizing vs reinforcing loops. Requires external CSV.
- **compute_fvs.py:** Requires external CSV data file not bundled in this folder. Override resistance scores framework-defined.
- Random seeds: none (deterministic)
- Dependencies: numpy + scipy (compute_S28/S29/fvs); math + json only (s23_trophic_coupling_test)

## Key results

- F2 = (1/eps)^n: r = -0.22 across 42 ecosystems -- trophic scaling does NOT predict D
- Driver count (N_direct) vs D: R^2 = 0.019 at N=7 anchors -- essentially zero predictive power
- Loop count vs D: r = -0.575 (wrong sign) -- more loops does NOT mean more persistence
- FVS size vs D: R^2 = 0.051 -- topological feedback structure alone does not predict D
- Combined S + drivers: R^2 = 0.652 (N=19) but driven by species richness S, not drivers
- **Conclusion:** Neither simple epsilon scaling, driver counts, loop counts, nor FVS topology predict D. Coupling strengths (epsilon values at the channel level) are what determine D. This motivates the mu-analysis derivation in S30.
