# Study 09: Search Equation -- Results

**Date:** 2026-04-04
**Scripts:** `step12_timescale_compression.py`, `step12b_granularity_test.py`, `step12c_substep_T4_T5_T6.py`, `step12d_independent_P.py`, `step12e_intermediate_k.py`, `v_sensitivity_analysis.py`

## Question

Does tau_search = n*S(d)/(v*P) hold across evolutionary transitions, and is there a universal search volume S0?

## Data Summary

9 major evolutionary transitions (T1-T8 + industrial), decomposed into sub-steps for T3/T4/T5/T6 (5 sub-steps each), yielding 29 data points total. 6 intermediate-complexity innovations (C4 photosynthesis, 4 flight origins, camera eyes) tested separately.

---

## Key Results

| Metric | Value | Source script |
|--------|-------|---------------|
| S0 (grand mean) | 10^13.0 +/- 0.8 | step12c (29 data points; T3a/T3b v corrected to 20/yr per Imachi 2020) |
| S0 range (raw) | 29 OOM (T1-T8) | step12 |
| S0 range (corrected P) | 5.1 OOM | step12b |
| log10(S) slope vs log10(v*P) | -0.95 | step12 |
| Anti-circularity deviation | 0.31 OOM mean | step12d |
| Intermediate innovations | 6/6 pass | step12e |

---

## Script-by-Script Results

### step12: Raw S spans 29 OOM
Raw S spans 29 OOM because P for T3 (eukaryogenesis) was set to 10^29 (all prokaryotes on Earth). The slope of -0.95 (close to -1.0) is consistent with tau = S/(v*P).

### step12b: Corrected P reduces spread to 5.1 OOM
Correcting P to the relevant searching population (independent lineages, not total individuals) and decomposing T3 into 5 sub-steps reduces S0 spread from 29 OOM to 5.1 OOM. T3 sub-steps use ~500 Asgard archaeal lineages (citing Spang 2015, Koonin 2010, Lane & Martin 2010).

### step12c: Full sub-step decomposition yields S0 = 10^13.0
Extending to T4/T5/T6 sub-steps (5 each, citing Archibald 2009, Keeling 2010, Grosberg & Strathmann 2007, Hughes 2008) yields 23 data points with mean log10(S) = 13.0. Paulinella cross-check for T4 is consistent.

### step12d: Anti-circularity test (KEY)
P estimated from biology alone (Zaremba-Niedzwiedzka 2017, Eme 2023, Imachi 2020, Knoll 2014, Li & Durbin 2011, Biraben 1979, Bellwood 2005) -- no reference to S0 or the formula. Mean deviation from step12b/12c P values is 0.31 OOM. This is the key anti-circularity test.

### step12e: Intermediate innovations pass
Innovations with many independent origins (C4: 61-62 origins citing Sage 2011/2016; flight: 4 origins citing Misof 2014, Baron 2021, Brusatte 2015, Simmons 2008; camera eyes: 9-12+ origins citing Nilsson & Pelger 1994) give k in [5, 25], confirming intermediate regime. Heckmann et al. 2013 independently estimates C4 as ~30 mutational changes. alpha = 0.373 FIXED from Study 10. k = log10(S) / log10(1/alpha).

---

## Parameter Tables

### step12_timescale_compression.py

| Transition | tau_search (yr) | v (trials/yr) | P | log10(S) | Source |
|------------|--------------|----------------|------|----------|--------|
| T1 Protocells | 600e6 | 8760 | 1e10 | ~22.7 | Szathmary & Maynard Smith 1995 |
| T2 Genetic code | 300e6 | 365 | 1e12 | ~20.0 | Szathmary & Maynard Smith 1995 |
| T3 Eukaryotes | 1.5e9 | 365 | 1e29 | ~40.7 | Szathmary & Maynard Smith 1995 |
| T4 Plastids | 500e6 | 52 | 1e17 | ~26.4 | Szathmary & Maynard Smith 1995 |
| T5 Multicellularity | 500e6 | 52 | 1e17 | ~26.4 | Szathmary & Maynard Smith 1995 |
| T6 Eusociality | 600e6 | 1 | 1e6 | ~14.8 | Szathmary & Maynard Smith 1995 |
| T7 Language | 2.6e6 | 122 | 1e5 | ~12.5 | Hublin et al. 2017 |
| T8 Agriculture | 188e3 | 122 | 5e6 | ~12.1 | Zeder 2011 |
| T8i Industrial | 11.7e3 | 365 | 1e5 | ~11.6 | Betts et al. 2018, Weiss et al. 2016 |

**v and P are author estimates, not directly from the cited sources.** This is the primary reason for the YELLOW grade.

### step12b_granularity_test.py (T3 sub-steps)

| Sub-step | tau (yr) | v | P_corrected | Source |
|----------|----------|------|-------------|--------|
| T3a Cytoskeleton | 500e6 | 20 | 1e3 (Asgard lineages) | Spang et al. 2015; v from Imachi et al. 2020 (14-25 day doubling) |
| T3b Cell wall loss | 200e6 | 20 | 300 | Koonin 2010; v from Imachi et al. 2020 (14-25 day doubling) |
| T3c Mitochondrial endosymbiosis | 100e6 | 52 | 100 | Lane & Martin 2010 |
| T3d Gene transfer + nucleus | 400e6 | 52 | 1e3 | Zachar & Szathmary 2017 |
| T3e Meiosis / sex | 300e6 | 52 | 1e3 | Lopez-Garcia & Moreira 2020 |

### step12d_independent_P.py (anti-circularity estimates)

| Transition | P_independent | P_corrected (12b/12c) | Key source |
|------------|--------------|----------------------|------------|
| T3c Mitochondrial | 150 | 100 | Zaremba-Niedzwiedzka 2017, Eme 2023, Imachi 2020 |
| T5a Cell adhesion | 5e3 | 3e3 | Knoll 2014 |
| T7 Language | 2e5 | 1e5 | Li & Durbin 2011 |
| T8 Agriculture | 8e6 | 5e6 | Biraben 1979, Bellwood 2005 |

### step12e_intermediate_k.py

| Innovation | Origins | tau (yr) | v | P | k_implied | Source |
|-----------|---------|----------|------|------|-----------|--------|
| C4 photosynthesis | 61-62 | 30e6 | 1 | 5e4 | 8.3 | Sage 2011/2016, Christin 2008 |
| Insect flight | 1 | 100e6 | 1 | 1e6 | 12.6 | Misof 2014, Dudley 2000 |
| Pterosaur flight | 1 | 50e6 | 0.5 | 1e4 | 8.1 | Baron 2021 |
| Bird flight | 1 | 30e6 | 1 | 1e4 | 9.1 | Brusatte 2015 |
| Bat flight | 1 | 20e6 | 1 | 1e4 | 8.6 | Simmons 2008 |
| Camera eyes | 9-12+ | 50e6 | 1 | 1e5 | 10.0 | Nilsson & Pelger 1994 |

alpha = 0.373 FIXED from Study 10. k = log10(S) / log10(1/alpha).

---

## Replicability Assessment

**Overall: YELLOW**

| Script | Grade | Notes |
|--------|-------|-------|
| `step12_timescale_compression.py` | YELLOW | v and P are author estimates, not from cited sources. Uses matplotlib (Agg). |
| `step12b_granularity_test.py` | YELLOW | P_corrected values are author estimates of lineage counts. Uses matplotlib (Agg). |
| `step12c_substep_T4_T5_T6.py` | YELLOW | Sub-step decompositions cite literature but durations are author choices. Uses matplotlib (Agg). |
| `step12d_independent_P.py` | GREEN | P estimated from biology alone, independent of formula. Uses matplotlib (Agg). |
| `step12e_intermediate_k.py` | GREEN | Published innovation data. alpha = 0.373 fixed (not fitted). Uses matplotlib (Agg). |

**Partial circularity warning:** S is defined by the equation being tested (S = tau * v * P). The anti-circularity test (step12d) addresses this by estimating P independently. v remains an author estimate, but v-sensitivity analysis (v_sensitivity_analysis.py, 2026-04-05) showed v is a **nuisance parameter**: 3.9x tolerance before S exits 95% CI, v_best/v_estimated = 1.01, and 73% of 66 v-free ratio predictions pass within 1 OOM. The key testable predictions (S scaling with d, ratio tests) are v-independent.

---

## Limitations

1. **Partial circularity:** S is defined by tau_search = S/(v*P), so S = tau*v*P is circular unless v and P are independently constrained. Step12d partially addresses this for P but not for v.
2. **Author estimates:** v (trial rate per lineage per year) is an author estimate, not calibrated to specific experimental measurements. However, v-sensitivity analysis (Fact 74) showed v is a nuisance parameter with 3.9x tolerance, and the search equation's testable content (S scaling, ratio predictions) is v-independent. T3c validates with a published biological rate (Pickup et al. 2007).
3. **Sub-step decomposition:** The choice of how many sub-steps to use (5 for T3, T4, T5, T6) and how to allocate the search phase duration among them introduces degrees of freedom.
4. **S0 is model-dependent:** S0 ~ 10^13 depends on the granularity level at which P is measured. At the "total individuals" level, S spans 29 OOM. At the "independent lineages" level, S converges to ~10^13. The correct granularity is a modeling choice.
5. **Intermediate k (step12e):** tau and P for innovations like C4 photosynthesis or camera eyes are less precisely constrained than for the major transitions.
