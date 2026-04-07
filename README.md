A full guide on how to replicate the studies in "Why organized states persist" by Chaitanya Agarwal

## Requirements

```bash
Python 3.8+
pip install numpy scipy
```

Most scripts require only numpy and scipy. Exceptions noted in the table below.

## Studies

| # | Study | Scripts | Claims | Grade | Key result | Extra deps |
|---|-------|---------|--------|-------|------------|------------|
| 01 | [Core Kramers Duality](01_core_kramers_duality/) | 8 | 1.1, 1.3, Fact 24 | YELLOW | D_product = D_Kramers for 6 ecological systems (D = 29-1111) | -- |
| 02 | [Bridge & B Invariance](02_bridge_b_invariance/) | 7 | 1.5, 2.1-2.5 | YELLOW | B = 2*DeltaPhi/sigma*^2, CV < 5% across 5 systems | -- |
| 03 | [Bridge Algebraic Tests](03_bridge_algebraic_tests/) | 8 | Facts 12-17 | YELLOW | Bridge is transcendental; per-channel decomposition fails (356% error) | sympy (1 script), numba (1 script, optional) |
| 04 | [Toggle Switch](04_toggle_switch/) | 5 | 5.4, 9.2, Facts 8-9 | YELLOW | K_CME = 1.0 (corrected from 2.0); epsilon fails for coupled systems | -- |
| 05 | [HIV](05_hiv/) | 7 | 5.1, Fact 44 | YELLOW | Omega* = 2296 mL (SDE) = 2293 (structural, 0.1% match) | matplotlib (2 scripts), import chain |
| 06 | [Medical Systems](06_medical_systems/) | 5 | 10.1, 10.2 | YELLOW | Tumor B = 2.73 +/- 2.7%; Diabetes 1D FAILS but 2D B = 5.54 +/- 5.2% | numba (1 script) |
| 07 | [Cross-Domain Physics](07_cross_domain_physics/) | 4 | 5.3, 10.3-10.5 | RED | Thermohaline K = 0.55; Soviet 4 free params (RED); financial uncalibrated (RED) | -- |
| 08 | [Blind Tests](08_blind_tests/) | 2 | 11.1, 11.2 | GREEN | Josephson B CV = 0.3%, nanoparticle B CV = 1.5-2.7%; 0 free params | -- |
| 09 | [Search Equation](09_fire_equation/) | 6 | 1.4, 11.3, Facts 35-39, 74 | YELLOW | S0 ~ 10^13.0 +/- 0.8 (29 data points); 6/6 intermediate innovations pass; v is nuisance parameter | matplotlib (all 6) |
| 10 | [Architecture Scaling](10_architecture_scaling/) | 7 | 4.1-4.5, 11.4 | GREEN/YELLOW | f(k) = alpha^k, R^2 = 0.997; alpha model-dependent (0.37-0.91) | -- |
| 11 | [Cusp Bridge](11_cusp_bridge/) | 5 | 4.6, 11.5 | YELLOW | S(d) = exp(gamma*d), gamma_cusp = 0.197 (Hill class); d ~ 150 for S = 10^13; R²=0.995 | -- |
| 12 | [Channel Independence](12_channel_independence/) | 2 | Facts 29, 31 | GREEN | Multiplicative stacking confirmed; 12/12 systems classified correctly | -- |
| 13 | [D = 1 Threshold](13_d_threshold/) | 2 | 3.1, 3.2 | GREEN | D < 1 achievable via Kramers (D = 0.68 at sigma = 1.0) | -- |
| 14 | [Topology (Negative)](14_topology_negative/) | 4 | 9.3-9.6, 7.3 | YELLOW | F2=(1/eps)^n r=-0.22; drivers R^2=0.019; loops r=-0.575; FVS R^2=0.051 | CSV bundled in data/ |
| 15 | [Blind Search Tests](15_blind_fire_tests/) | 2 | 15.1, 15.2 | GREEN/YELLOW | 15.1 Flowers: tau=126 vs 200 Myr (-0.20 OOM, GREEN). 15.2 CAM: tau=204 vs 30 Myr (+0.83 OOM at d=72, YELLOW); multi-origin d~52 explains 62 origins | -- |
| 16 | [Hopf Bridge](16_hopf_bridge/) | 1 | 16.1, 16.2 | GREEN | gamma_Hopf = 0.172 (R^2=0.9997); exponential form confirmed for Hopf with Hill functions | -- |
| 17 | [Bautin Bridge](17_bautin_bridge/) | 1 | 17.1, 17.2 | YELLOW | B_Bautin = 4.14 (CV=2.2%), stability window YES; gamma_Bautin = 0.021 (flat, no decay) | -- |
| 18 | [Polynomial Bridge](18_poly_bridge/) | 3 | 18.1-18.3 | GREEN | gamma_poly = 0.010 (flat); poly+degradation gamma = 0.007 (flat); S(d) requires saturation+degradation (Fact 82) | -- |
| 19 | [B Bounded](19_B_bounded/) | 1 | 19.1 | GREEN | B bounded by cusp scale invariance; union B in [1.54, 5.66] across D=29-1111; stability window = geometry + selection (Fact 77) | -- |
| 20 | [Fire Tree Flight](20_fire_tree_flight/) | 1 | 20.1-20.4 | YELLOW | 14 flight clades: B in stability window for 3 integrated clades; modularity gradient 3 OOM; 5 D=1 crossings | -- |
| 21 | [Currency Peg Kramers](21_currency_peg_kramers/) | 1 | 21.1-21.3, Fact 85 | GREEN | GBP B=1.89, THB B=1.90 (0.3% match); 0 eff. free params; first organizational system in stability window | -- |
| 22 | [General B Bounded](22_general_B_bounded/) | 1 | 22.1, Fact 86 | GREEN | B boundedness universal: 4 potential families (cusp, washboard, nanomagnet, quartic); scale inv. CV=0.00%; washboard width=0.034; resolves Study 19 Limitation #1 | -- |
| 23 | [Lambda k* Count](23_lambda_k_count/) | 1 | 23.1 | GREEN | Lambda: ~30 interactions, d~27-50, k~5-8 loops; framework predicts k*=5-10, MATCH (52% overlap); k*=30 is for major transitions | -- |
| 24 | [2D B Bounded](24_2D_B_bounded/) | 1 | 24.1, Fact 87 | GREEN | 2D B boundedness: scale inv. CV=1.70% (tumor-immune SDE), Kramers-Langer prefactor bounded (toggle 0.115, tumor-immune 0.910); extends Study 22 to 2D | -- |
| 25 | [Stellar Kramers](25_stellar_kramers/) | 1 | 3.4, Fact 88 | GREEN | Stellar D ~ 1: Bonnor-Ebert virial potential, D = 2-6 (Mach 1) for star-forming cores (M/M_BE = 0.8-1.2), 0 free params; upgrades claim 3.4 WEAK CHAIN → VERIFIED | -- |
| 26 | [Sigma Existence Constraint](26_sigma_existence_constraint/) | 1 | 26.1 | YELLOW | Stability window eliminates 93.5% of prior sigma range; band width = 1.826; 4/5 ecological systems inside | -- |
| 27 | [Model Selection S(d)](27_model_selection_Sd/) | 1 | 27.1 | GREEN | Power law rejected (dAICc = +12.63); exponential preferred (50% weight); bootstrap CI for d(S=10^13) = [150, 167] | -- |
| 28 | [Xenopus Product Eq](28_xenopus_product_eq/) | 1 | 28.1 | GREEN | First blind product eq test outside ecology; LOW state B = 3.45 (in window); HIGH state fails (strong channels) | -- |
| 30 | [Data Collapse](30_data_collapse/) | 2 | 30.1 | GREEN | ln(D) - beta_0 = B for 13 systems across 7 domains, slope = 1; beta_0 = 1.06 +/- 0.37 (1D); 2D SDE confirms | matplotlib, numba |
| 31 | [Protein Stability Window](31_protein_stability_window/) | 1 | 31.1 | GREEN | 22/25 proteins B > 6.0; stability window requires barrier-noise comparable scales; scope boundary identified | -- |
| 32 | [Crossing Theorem](32_crossing_theorem/) | 1 | 32.1, 32.2 | GREEN | sigma* = sigma_process resolved: IVT crossing guaranteed; all 6 systems B_cross in [1.8, 6.0]; BW = 35.3% | -- |
| 33 | [Noise Robustness](33_noise_robustness/) | 3 | Reviewer 1.1/2.6 | GREEN | Multiplicative g(x)=sigma*sqrt(x): B_mult CV=3.07% (q=8); colored noise: B_colored CV=1.95%; Ito-Strat: both CV<5%. B invariance robust to noise type | -- |
| -- | [misc](misc/) | 4 | -- | YELLOW | Lake 2D model investigations (historical) | -- |

**Totals:** ~98 scripts, 33 studies + misc

## Grade distribution

| Grade | Studies | Description |
|-------|---------|-------------|
| GREEN | 08, 10 (core), 12, 13, 15 (flowers), 16, 18, 19, **21**, 22, 23, 24, **25**, **27**, **28**, **30**, **31**, **32**, **33** | All params sourced or synthetic by design; fully reproducible |
| YELLOW | 01-06, 09-11, 14, 15 (CAM), 17, **20**, **26** | Some modified/derived params or author estimates; reproducible with caveats |
| RED | 07 | Soviet (4 free uncalibrated params), financial generic (no real data); note: calibrated financial upgraded to GREEN in Study 21 |

## Complete script-to-study mapping

| Script | Study |
|--------|-------|
| `alpha_2d_savanna_scaling.py` | 10 - Architecture Scaling |
| `B_bounded_derivation.py` | 19 - B Bounded |
| `bautin_bridge.py` | 17 - Bautin Bridge |
| `hopf_bridge_scaling.py` | 16 - Hopf Bridge |
| `poly_bridge_scaling.py` | 18 - Polynomial Bridge |
| `poly_degradation_test.py` | 18 - Polynomial Bridge |
| `alpha_2d_savanna_targeted.py` | 10 - Architecture Scaling |
| `alpha_2d_toggle_scaling.py` | 10 - Architecture Scaling |
| `B_distribution_test.py` | 10 - Architecture Scaling |
| `blind_cam_prediction.py` | 15 - Blind Search Tests |
| `blind_flowers_prediction.py` | 15 - Blind Search Tests |
| `blind_test_josephson_junction.py` | 08 - Blind Tests |
| `blind_test_magnetic_nanoparticle.py` | 08 - Blind Tests |
| `bridge_dimensional_scaling.py` | 11 - Cusp Bridge |
| `bridge_high_d_analysis.py` | 11 - Cusp Bridge |
| `bridge_high_d_scaling.py` | 11 - Cusp Bridge |
| `bridge_q3_symbolic.py` | 03 - Bridge Algebraic Tests |
| `bridge_v2_B_invariance_proof.py` | 02 - Bridge & B Invariance |
| `compute_fvs.py` | 14 - Topology (Negative) |
| `compute_S28.py` | 14 - Topology (Negative) |
| `compute_S29.py` | 14 - Topology (Negative) |
| `s23_trophic_coupling_test.py` | 14 - Topology (Negative) |
| `conway_perelson_model.py` | 05 - HIV |
| `cusp_bridge_derivation.py` | 11 - Cusp Bridge |
| `epsilon_duality_test.py` | 05 - HIV |
| `financial_cusp_kramers.py` | 07 - Cross-Domain Physics |
| `hermite_validation.py` | 03 - Bridge Algebraic Tests |
| `noise_source_mapping_hiv.py` | 05 - HIV |
| `omega_gap_closure.py` | 05 - HIV |
| `patha_v2_what_determines_B.py` | 02 - Bridge & B Invariance |
| `pathc_dynamic_epsilon.py` | 03 - Bridge Algebraic Tests |
| `phase1_lake_1d.py` | 01 - Core Kramers Duality |
| `phase2_barrier_action.py` | misc |
| `phase2_model_c_equilibria.py` | misc |
| `phase2_model_c_v2.py` | misc |
| `phase3_channel_weakening.py` | misc |
| `power_grid_kramers.py` | 07 - Cross-Domain Physics |
| `s0_derivation_architecture_scaling.py` | 10 - Architecture Scaling |
| `soviet_kuran_kramers.py` | 07 - Cross-Domain Physics |
| `step2_savanna_log_robustness.py` | 01 - Core Kramers Duality |
| `step6_kelp_kramers.py` | 01 - Core Kramers Duality |
| `step6b_kelp_immigration.py` | 01 - Core Kramers Duality |
| `step7_coral_kramers.py` | 01 - Core Kramers Duality |
| `step8_synthetic_3channel.py` | 12 - Channel Independence |
| `step9_toggle_epsilon.py` | 04 - Toggle Switch |
| `step10_tropical_forest_kramers.py` | 01 - Core Kramers Duality |
| `step11_channel_independence.py` | 12 - Channel Independence |
| `step12_timescale_compression.py` | 09 - Search Equation |
| `step12b_granularity_test.py` | 09 - Search Equation |
| `step12c_substep_T4_T5_T6.py` | 09 - Search Equation |
| `step12d_independent_P.py` | 09 - Search Equation |
| `step12e_intermediate_k.py` | 09 - Search Equation |
| `v_sensitivity_analysis.py` | 09 - Search Equation |
| `step13_cascade_and_barrier_distribution.py` | 10 - Architecture Scaling |
| `step13_hiv_structural_omega.py` | 05 - HIV |
| `step13_peatland_kramers.py` | 01 - Core Kramers Duality |
| `step13_peatland_kramers_hilbert.py` | 01 - Core Kramers Duality |
| `step13b_3d_reduced_map.py` | 05 - HIV |
| `step13b_savanna_cascade_and_random_arch.py` | 10 - Architecture Scaling |
| `step13c_de_map.py` | 05 - HIV |
| `structural_B_coral.py` | 02 - Bridge & B Invariance |
| `structural_B_diabetes.py` | 06 - Medical Systems |
| `structural_B_diabetes_2D_SDE.py` | 06 - Medical Systems |
| `structural_B_kelp.py` | 02 - Bridge & B Invariance |
| `structural_B_savanna.py` | 02 - Bridge & B Invariance |
| `structural_B_toggle.py` | 02 - Bridge & B Invariance |
| `structural_B_tumor_immune.py` | 06 - Medical Systems |
| `structural_B_tumor_immune_2D.py` | 06 - Medical Systems |
| `structural_B_tumor_immune_SDE_scan.py` | 06 - Medical Systems |
| `structural_connection_test.py` | 02 - Bridge & B Invariance |
| `test1_barrier_epsilon.py` | 03 - Bridge Algebraic Tests |
| `test3_k_deep_barrier.py` | 03 - Bridge Algebraic Tests |
| `test3_k_universality.py` | 03 - Bridge Algebraic Tests |
| `test5_barrier_scaling.py` | 03 - Bridge Algebraic Tests |
| `test5_refinement.py` | 03 - Bridge Algebraic Tests |
| `test_D_below_one.py` | 13 - D = 1 Threshold |
| `test_D_below_one_fast.py` | 13 - D = 1 Threshold |
| `thermohaline_kramers.py` | 07 - Cross-Domain Physics |
| `toggle_kramers_test.py` | 04 - Toggle Switch |
| `toggle_prefactor_fix.py` | 04 - Toggle Switch |
| `toggle_shortcut.py` | 04 - Toggle Switch |
| `unification_test.py` | 04 - Toggle Switch |
| `study23_lambda_k_count.py` | 23 - Lambda k* Count |
| `study24_2D_B_bounded.py` | 24 - 2D B Bounded |
| `study25_stellar_kramers.py` | 25 - Stellar Kramers |
| `multiplicative_B_invariance.py` | 33 - Noise Robustness |
| `colored_noise_B_invariance.py` | 33 - Noise Robustness |
| `ito_stratonovich_correction.py` | 33 - Noise Robustness |
| `fire_tree_flight.py` | 20 - Fire Tree Flight |
| `currency_peg_kramers.py` | 21 - Currency Peg Kramers |
| `study22_coral_B_verification.py` | 22 - General B Bounded |
| `study26_sigma_existence_constraint.py` | 26 - Sigma Existence Constraint |
| `model_selection_Sd.py` | 27 - Model Selection S(d) |
| `xenopus_product_eq_test.py` | 28 - Xenopus Product Eq |
| `data_collapse.py` | 30 - Data Collapse |
| `sweep_2d_sde.py` | 30 - Data Collapse |
| `study31_protein_stability_window.py` | 31 - Protein Stability Window |
| `crossing_theorem_test.py` | 32 - Crossing Theorem |
| `route_a_eigenvalue_test.py` | 32 - Crossing Theorem |

## Known replicability flags

| Study | Issue | Severity |
|-------|-------|----------|
| 01 | Tropical forest: alpha=0.6 (published 0.2), beta=1.0 (published 0.3) -- 3x modified | YELLOW |
| 01 | Lake epsilon_SAV=0.05 model-derived (Kosten 2009/Scheffer 1998), not field | YELLOW |
| 01 | Peatland: 1-2 free params (m, q in Path C; c1 in Path A) | YELLOW |
| 02 | structural_B_kelp.py: NO literature citations for r, K, h in this specific script | RED |
| 03 | bridge_q3_symbolic.py requires sympy (beyond numpy+scipy) | YELLOW |
| 03 | pathc_dynamic_epsilon.py: optional numba, no random seed (stochastic) | YELLOW |
| 05 | Import chain: conway_perelson_model -> epsilon_duality_test -> omega_gap_closure | Structural |
| 05 | MAP optimization did not converge; Omega* is SDE-calibrated | YELLOW |
| 05 | epsilon_duality_test.py and omega_gap_closure.py use matplotlib + write plots | YELLOW |
| 06 | structural_B_diabetes_2D_SDE.py requires numba | YELLOW |
| 06 | Diabetes 2D: SIGMA_G=10, SIGMA_BETA=0.01 "biologically motivated" -- unsourced | YELLOW |
| 07 | Soviet: 4 free params, 0 calibrated, unpublished ODE conversion | RED |
| 07 | Financial cusp: generic parameters, not calibrated to real data | RED |
| 07 | Power grid: D_obs undefined (SOC/cascade, not Kramers escape) | YELLOW |
| 09 | S defined by the equation being tested (partial circularity) | YELLOW |
| 09 | v and P values are author estimates (order-of-magnitude) | YELLOW |
| 09 | ALL 5 scripts use matplotlib (Agg backend) | YELLOW |
| 11 | Extrapolation from d=62 to d=130 is 2x beyond data | YELLOW |
| 11 | Deceleration at d>50 may indicate plateau | YELLOW |
| 11 | bridge_high_d_analysis.py is new/untracked (not yet in git) | Structural |
| 14 | compute_fvs.py and compute_S29.py require Rocha 2018 CSV (now bundled in study folder) | RESOLVED |

## How to replicate a specific claim

1. Find the claim number in the **Claims** column of the studies table above
2. Go to that study folder
3. Read the study README for parameter provenance and expected outputs
4. Run the scripts in the order specified

Example:
```bash
# Replicate claim 11.1 (Josephson junction blind test)
cd 08_blind_tests/
python3 blind_test_josephson_junction.py
# Expected: B CV = 0.3-0.4%, K = 0.56
```

## Dependency summary

| Dependency | Required by | Status |
|------------|-------------|--------|
| numpy | All scripts | Required |
| scipy | All scripts | Required |
| matplotlib | 09 (all 6), 05 (2 scripts), 30 (data_collapse.py) | Required for those scripts |
| sympy | 03 (bridge_q3_symbolic.py) | Required for 1 script |
| numba | 03 (pathc_dynamic_epsilon.py), 06 (structural_B_diabetes_2D_SDE.py), 30 (sweep_2d_sde.py) | Optional (pathc) / Required (diabetes 2D, 2D SDE sweeps) |

## Relationship to scripts/

The `scripts/` directory contains scripts in a flat structure. This `studies/` directory organizes them into thematic groups with READMEs. Scripts were COPIED, not moved -- `scripts/` remains the canonical flat index referenced by all existing documentation.
