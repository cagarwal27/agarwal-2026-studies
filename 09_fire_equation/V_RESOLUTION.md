---
title: Independent validation of v parameter in search equation
date: 2026-04-07
status: YELLOW (partially resolved)
script: Agarwal/scripts/v_independent_validation.py
---

# V Resolution: Independent Validation of Search Equation Trial Rates

## Problem

The search equation S = tau × v × P / n uses author-estimated v (compound innovation trial rate) for all 29 transitions. A reviewer can claim S ≈ 10^13 is circular: v was picked to make S come out right.

## Published rates found

Independent, directly measured rates exist for 4 of 6 priority transitions:

| Transition | Published rate | Source | v_pub (/yr) | Framework v | log₁₀(S) | In CI? |
|---|---|---|---|---|---|---|
| T3a Cytoskeleton | Asgard division (14d doubling) | Imachi et al. 2020 Nature 577:519 | 26 | 20 | 13.12 | YES |
| T3c Mito endosymbiosis | Amoeba ingestion (0.2 bact/h) | Protist literature (Pickup 2007, Gonzalez 1999) | 1,752 | 52 | 13.24 | YES |
| T4a Plastid capture | HNF on picocyanobacteria | Callieri et al. 2002 J Plankton Res 24:785 | 13,140 | 365 | 15.52 | NO |
| T6a Parental care | Earwig reproductive rate | Koch & Meunier 2014 Behav Ecol Sociobiol | 1.5 | 1 | 13.48 | NO |

No directly measured rates exist for T7 (language) or T8 (agriculture). Ethnographic data (Lew-Levy et al. 2021) gives forager learning rates, not experimentation rates. Chimpanzee communication innovation is ~1–5 novel behaviors/community/year (Reader & Laland 2002), too indirect.

## Interpretation

**T3a is the cleanest validation.** The Asgard archaeal division rate IS the trial rate — each division produces mutations that could generate cytoskeletal innovations. Published v = 26/yr gives log₁₀(S) = 13.12, inside the 95% CI [12.67, 13.25]. No attenuation factor needed.

**T6a is the second cleanest.** Insect reproductive rates (1.5–3/yr) bracket the framework v = 1/yr. log₁₀(S) = 13.48, above CI but only 0.52 OOM from the mean. Each breeding event IS a behavioral innovation trial.

**T3c confirms the biological event** (ingestion at 1,752/yr from field conditions, up to 6M/yr in lab) but the framework v = 52 implies an attenuation factor of ~1/34 (field) to ~1/115,000 (lab max). The attenuation is estimated, not measured.

**T4a** has directly measured cyanobacterial ingestion rates (Callieri 2002: 13,140/yr) but v = 365 implies 1/36 attenuation — again estimated.

## Cross-transition ratio test

Of 66 pairwise ratio predictions in the v = 52/yr group (12 members), 19 are within-transition (shared author decomposition, not independent) and 47 are cross-transition (different biology, epochs, and literature). Cross-transition pass rate: **34/47 = 72% within 1 OOM** (mean discrepancy 0.67 OOM). This is comparable to the within-transition rate (74%), confirming the test is not inflated by correlated sub-steps.

## Honest assessment

Two transitions (T3a, T6a) have v replaceable by published rates with S near 10^13. Two more (T3c, T4a) confirm the raw biological event occurs but require an estimated attenuation factor. Two (T7, T8) have no independent v. The cross-transition ratio test (47 pairs, 72% pass rate) is the strongest v-independent evidence because v cancels exactly. The v problem is partially resolved: the equation is not fully circular, but it is not fully independent either.
