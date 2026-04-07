# Study 09: Search Equation

**Date:** 2026-04-04

## Purpose

Tests tau_search = n*S(d)/(v*P) across evolutionary transitions. S(d) = exp(gamma*d) where gamma is approximately 0.20 per parameter. Assesses whether a universal search volume S0 exists and whether independent P estimates match the formula.

## Data Provenance

### Input parameters (step12)
- **Transition timescales (tau):** Szathmary & Maynard Smith 1995 (T1-T6); Hublin et al. 2017 (T7); Zeder 2011 (T8); Betts et al. 2018, Weiss et al. 2016 (T8i industrial)
- **v (trial rates) and P (searching populations):** Author estimates -- not directly from cited sources. This is the primary reason for the YELLOW grade.

### Corrected P values (step12b)
- **T3 sub-steps:** Spang et al. 2015 (Asgard lineages), Koonin 2010, Lane & Martin 2010, Zachar & Szathmary 2017, Lopez-Garcia & Moreira 2020
- **v for T3a/T3b:** Imachi et al. 2020 (14-25 day doubling -> ~20/yr)

### T4-T6 sub-steps (step12c)
- **T4 (Plastids):** Archibald 2009, Keeling 2010
- **T5 (Multicellularity):** Grosberg & Strathmann 2007
- **T6 (Eusociality):** Hughes 2008

### Independent P estimates (step12d)
- Zaremba-Niedzwiedzka 2017, Eme 2023, Imachi 2020 (T3c); Knoll 2014 (T5a); Li & Durbin 2011 (T7); Biraben 1979, Bellwood 2005 (T8)

### Intermediate innovations (step12e)
- **C4 photosynthesis:** Sage 2011/2016, Christin 2008; Heckmann et al. 2013
- **Flight origins:** Misof 2014, Baron 2021, Brusatte 2015, Simmons 2008, Dudley 2000
- **Camera eyes:** Nilsson & Pelger 1994
- **alpha = 0.373:** Fixed from Study 10 (not fitted)

**Provenance claims:** 1.4, 11.3, Facts 35-39, 70

## Replication

### Requirements
- Python 3.8+
- numpy, scipy, matplotlib

```
pip install numpy scipy matplotlib
```

### To reproduce
Scripts should be run in order (step12 -> step12b -> step12c -> step12d -> step12e) to follow the logical progression, but each is self-contained with hardcoded data and can be run independently.

```bash
cd 09_fire_equation
python3 step12_timescale_compression.py   # 9 transitions, ~seconds, saves plots
python3 step12b_granularity_test.py       # T3 sub-steps, ~seconds, saves plots
python3 step12c_substep_T4_T5_T6.py       # T4-T6 sub-steps, ~seconds, saves plots
python3 step12d_independent_P.py          # Anti-circularity test, ~seconds, saves plots
python3 step12e_intermediate_k.py         # Intermediate k, ~seconds, saves plots
python3 v_sensitivity_analysis.py         # v-sensitivity analysis, ~seconds
```

All scripts use matplotlib with the Agg backend for plot generation. Plots are saved to `../plots/`. No import dependencies between scripts.

## Files

| File | Contents |
|------|----------|
| `README.md` | This file -- provenance, replication, data sources |
| `RESULTS.md` | Findings, tables, interpretation, limitations |
| `step12_timescale_compression.py` | Initial test across 9 transitions (T1-T8 + industrial) |
| `step12b_granularity_test.py` | P correction to lineage-level; T3 decomposition into 5 sub-steps |
| `step12c_substep_T4_T5_T6.py` | T4/T5/T6 sub-step decomposition; 23 data points total |
| `step12d_independent_P.py` | Anti-circularity test: P from biology alone |
| `step12e_intermediate_k.py` | Intermediate k=5-25 innovations (C4, flight, camera eyes) |
| `v_sensitivity_analysis.py` | v-sensitivity analysis (Fact 74) |
