# Provenance Chain: Every Claim Traced to Its Source

**Date:** 2026-04-07
**Total claims:** 87 (added 34.1; prior: 33.1, 32.1, 32.2, 26.1, 27.1, 28.1, 30.1, 31.1, 24.1, 22.1a, 22.1, 21.1-21.3, 20.1-20.4, 17.1, 17.2, 18.1, 18.2, 18.3, 16.1, 16.2, 19.1)
**Purpose:** Trace every major claim in the framework to its source file, computation, and input data. Where the chain breaks, flag it explicitly.
**Method:** Each source file was read directly; no claim was accepted from summaries alone.

---

## Grading Key

| Grade | Meaning |
|-------|---------|
| **A** | Exact computation from published ODE, no free parameters, script reproducible |
| **B** | Computation with 1-2 free parameters or constructed model |
| **C** | Literature review / qualitative mapping / rough estimate |
| **D** | Hypothesis or dimensional analysis, not independently verified |
| **X** | Broken chain -- cannot trace to source |

---

## CATEGORY 1: Core Equations

### 1.1 Product equation verified for 6 ecological systems

**Statement:** D = prod(1/epsilon_i) verified for 6 ecological systems with D = 29 to 1111.

**Conclusion:** VERIFIED

**Script:** `step10_tropical_forest_kramers.py`, `step13_peatland_kramers.py`

**Input data:** Published ecological literature for epsilon values (see provenance grades below).

**Key numbers:**

| System | k | epsilon values | D_predicted | D_observed | Error | epsilon grade |
|--------|---|---------------|------------|-----------|-------|--------------|
| Kelp | 1 | 0.034 | 29.4 | 33 | 12% | A (Tinker 2019, Yeates 2007) |
| Savanna | 2 | 0.10, 0.10 | 100 | 100 | exact | B- (Murphy 2019, McNaughton 1985) |
| Lake | 2 | 0.05, 0.10 | 200 | 200 | exact | C (no primary source for eps_SAV=0.05) |
| Coral | 2 | 0.03, 0.03 | 1,111 | 1,000 | 11% | B- (Perry 2013, Mumby 2007) |
| Trop. forest | 2 | 0.07, 0.15 | 95.2 | 80-100 | ~1x | C (Brando 2014, Eltahir 1994; 2 free params) |
| Peatland | 2 | 0.10, 0.33 | 30.3 | 20-40 | ~1x | B (Clymo 1984, Hajek 2011; 1 free param) |

**Confidence grade:** B

**Flags:**
1. R01 initially showed 0/4 clean PASS. The "6 verified" claim reflects corrections made in Steps 10/13 after re-examining channel structure (k: 1->2 for tropical forest and peatland) and epsilon definitions.
2. Verification uses D_product = D_exact(sigma*) via Kramers MFPT, not independent D_observed measurement. The test is self-consistency of the two equations at sigma*, confirmed by independent noise measurement matching sigma*.
3. Lake epsilon_SAV = 0.05 has no identified primary field source (Grade C).
4. Tropical forest requires 2 free parameters (alpha=0.6, beta=1.0; published: 0.2, 0.3) and has channel mismatch (model: fire+competition; epsilon estimates: fire+drought).
5. The claim is numerically correct but the framing "verified" should be read as "self-consistent under the duality test" rather than "independently measured D_observed matches D_predicted."
6. Independent flux-based epsilon validation via `compute_coral_validation.py` confirms e=0.03 through coupling efficiency computation.

---

### 1.2 Product equation derived from mu-analysis (S30)

**Statement:** D = prod(1/epsilon_i) follows from structured perturbation theory (mu-analysis, Maciejowski 1989).

**Conclusion:** VERIFIED (theoretical derivation; empirical validation incomplete)

**Script:** S30 analysis scripts in `results/`

**Input data:** Maciejowski 1989 control theory; Rocha 2018 CLDs for 19 regime shifts.

**Key numbers:**
- Single-loop recovery: D = 1/epsilon (matches)
- Two-channel (eps_1 = eps_2 = 0.03): D = (1/0.03)^2 = 1100 (matches coral)
- FVS vs D: r = +0.227, R^2 = 0.051 (null correlation)
- FVS clusters at 1-2 nodes across 19 systems with D spanning 3+ OOM

**Confidence grade:** B

**Flags:**
1. Derivation assumes block-diagonal perturbation structure (channels act independently). Real ecological networks may not satisfy this.
2. The FVS empirical test (Phase 3) directly shows topology alone insufficient -- coupling strengths are essential. Framework is not yet complete: "Phase 3 Steps 3-4 -- estimate Jacobian coupling strengths" remain undone.
3. The derivation requires a priori specification of k and epsilon_i -- it does not predict them from system structure.

---

### 1.3 Kramers equation verified across systems

**Statement:** D = K * exp(2*DeltaPhi/sigma^2) / (C*tau) verified across ecological, climate, and biological systems.

**Conclusion:** VERIFIED

**Scripts:** `step6_kelp_kramers.py`, `step7_coral_kramers.py`, `step10_tropical_forest_kramers.py`, `step13_peatland_kramers.py`, `thermohaline_kramers.py`, `toggle_prefactor_fix.py`

**Key numbers:**

| System | D | K | sigma* | 2DeltaPhi/sigma*^2 | Free params | Accuracy |
|--------|---|---|--------|-------------------|-------------|----------|
| Savanna | 100 | 0.55 | 0.017 | 3.74 | 0 | 4% |
| Lake | 200 | 0.56 | 0.175 | 4.25 | 0 | 3% |
| Kelp | 29.4 | 0.34 | 10.50 | 1.80 | 1 (h) | exact |
| Coral | 1,111 | 0.56 | 0.034 | 6.03 | 0 | -- |
| Trop. forest | 95.2 | 0.34 | 0.018 | 4.00 | 2 (alpha,beta) | exact |
| Peatland | 30.3 | 0.36 | 1.42 | 3.07 | 1 (c1) | -- |
| Toggle (CME) | 9,175 | 1.0 | -- | -- | 0 | ~20% |
| Thermohaline | varies | 0.55 | varies | -- | 0 | 0.7-7% |
| HIV | -- | -- | Omega*=2296 | -- | 0 | conditional |

**Confidence grade:** A

**Flags:**
1. K is NOT universal: 0.34-0.36 (anharmonic/boundary wells), 0.55-0.57 (parabolic/interior wells), ~1.0 (CME with 2D barrier). K is a model-shape parameter, not a free parameter.
2. Toggle K=2.0 initially reported was a 1D/2D dimensional inconsistency, corrected to K~1.0.
3. Prefactor 1/(C*tau) computed from Jacobians: savanna 4.3, lake 5.0, toggle 10.5.

---

### 1.4 Search equation S0 = 10^13.0 +/- 0.8

**Statement:** tau_search = n * S0 / (v_compound * P_relevant) with S0 approx 10^13.0, verified across 29 data points spanning 7 evolutionary transitions.

**Conclusion:** VERIFIED (with corrections from SV12B/C/D)

**Scripts:** `step12b_granularity_test.py`, `step12c_substep_T4_T5_T6.py`, `step12d_independent_P.py`

**Input data:** Evolutionary transition timescales from published literature; P from phylogenetics/demographics.

**Key numbers:**
- SV12 initial: S spans 29 OOM (T1-T8i) -- P-definition inconsistency (individuals vs lineages)
- SV12B corrected: S range = 5.1 OOM; slope = -0.95 (predicted -1); variance reduced 97.8%
- SV12C: 23 data points across T3-T8i sub-steps; grand mean log10(S0) = 12.87; std = 0.79; 95% CI [12.53, 13.21]. T3a/T3b v corrected from 365 to 20/yr (Imachi et al. 2020: Asgard doubling 14-25 days).
- SV12D: P from biology alone (6 transitions) gives mean log10(S) = 13.29 +/- 0.79; P_independent/P_corrected ratio within 3x for all 6; Welch's t vs SV12C: p=0.425 (not significant)
- Combined (29 points): S0 = 10^12.96 +/- 0.79, 95% CI [12.67, 13.25]
- Anti-circularity test: 5/5 pass

**Confidence grade:** B

**Flags:**
1. S = tau * v * P is definitionally tautological (S is defined by the equation). The anti-circularity test (SV12D) partially addresses this by showing P can be estimated from biology alone. v-sensitivity analysis (Fact 74) further addresses this: v is a nuisance parameter with 3.9x tolerance, and 73% of 66 ratio predictions pass within 1 OOM without depending on v.
2. Key corrections: P must be "independent lineages with right preconditions" for genetic transitions, not total individuals. v must count "compound innovation trials," not elementary reactions. T3a/T3b v corrected from 365/yr (generic daily prokaryote) to 20/yr (Asgard archaeal generation from Imachi et al. 2020, Nature 577:519; MK-D1 doubling time 14-25 days).
3. Two outlier groups remain: T6 sub-steps systematically low (multi-origin = easier), T1 high (v granularity issue). T3a (previously highest sub-step at 14.26) now at 13.00 after v correction.
4. Genetic-derived S0 (10^14.4) and cultural-derived S0 (10^12.6) differ by 1.8 OOM -- within noise but suggestive of two regimes.
5. Slope -0.95 within 5% of predicted -1 is the strongest single result.

---

### 1.5 Bridge identity ln(D) = B + beta_0

**Statement:** The bridge identity ln(D) = B + beta_0, where B = 2*DeltaPhi/sigma*^2, is demonstrated computationally via 240 exact MFPT computations with B CV < 3% for all q.

**Conclusion:** VERIFIED

**Script:** `bridge_v2_B_invariance_proof.py`, `patha_v2_what_determines_B.py`

**Input data:** Lake 1D model, q in {3,4,5,6,8,10,16,20}, 30 loading values per q, exact MFPT integrals (not Kramers approximation).

**Key numbers:**

| q | B mean | B CV | DeltaPhi variation |
|---|--------|------|-------------------|
| 3 | 4.050 | 2.84% | 18x |
| 4 | 4.204 | 2.27% | 82x |
| 5 | 4.372 | 1.10% | 84x |
| 6 | 4.519 | 0.87% | 92x |
| 8 | 4.742 | 1.78% | 96x |
| 10 | 4.894 | 2.32% | 104x |
| 16 | 5.165 | 2.82% | 118x |
| 20 | 5.272 | 2.99% | 125x |

Maximum B CV = 2.99% at q=20. All 8 satisfy CV < 3%. Barriers vary 18-125x while B stays nearly constant. "Semi-analytical" = exact MFPT integral via numerical quadrature (not Kramers approximation).

**Confidence grade:** A

**Flags:**
1. 240 computations is numerical verification, not symbolic proof. No algebraic derivation exists (Fact 16: transcendental obstruction).
2. beta_0 is NOT universal across systems -- ranges [0.535, 1.611] driven by eigenvalue ratio |lambda_eq/lambda_sad|.
3. The bridge is an a posteriori identity showing IF product = Kramers THEN B is invariant. The converse (B invariant -> product = Kramers) is not proven.

---

### 1.6 Ceiling equation D = 1/epsilon at k=1

**Statement:** At k=1, D = 1/epsilon. Three astrophysical confirmations: molecular clouds (derived, D=50-100), stars (derived, D~1), galaxies (measured, D=67).

**Conclusion:** VERIFIED

**Script:** None (literature-based computation)

**Input data:** KM05 Eq. 30 (Krumholz & McKee 2005 ApJ 630, 250); Elbaz 2026; Fabian 2012 ARA&A 50, 455.

**Key numbers:**
- Molecular clouds: epsilon_ff = 1.4-2.2% from KM05 turbulence theory -> D = 46-71; observed 50-100; offset <= 0.34 dex
- Stars: Elbaz M^0 cancellation -> D ~ 1 (variation factor 2.4 for 0.5-5.0 M_sun); pre-informational baseline
- Galaxies: tau_persistence/tau_search = 10 Gyr / 150 Myr = 67; epsilon = 1.5% (back-derived, not independent)
- Clusters: D = 10-50 from Fabian P_cav ~ L_cool balance

**Confidence grade:** A (molecular clouds); B (galaxies: measured, epsilon back-derived); B (stars: Elbaz scaling)

**Flags:**
1. Galaxy epsilon is backward-derived from D, not independently computed from jet physics.
2. Cluster D = 10-50 is a range estimate, not precise computation (treated as "cursory" in S21).
3. The three confirmations have unequal evidential weight: clouds are derived from first principles; galaxies and clusters are measured timescale ratios.

---

## CATEGORY 2: B Invariance

### 2.1 B = 2*DeltaPhi/sigma*^2 has CV < 3% across all q

**Statement:** B is a structural invariant with CV < 3% for all Hill coefficients q in {3..20}.

**Conclusion:** VERIFIED

**Script:** `bridge_v2_B_invariance_proof.py`

**Input data:** Lake model, 240 exact MFPT computations, D_product = 200, sigma* found by bisection to 10^-8 tolerance.

**Key numbers:** See table under Claim 1.5 (same computation). Max B CV = 2.99%.

**Confidence grade:** A

**Flags:** None.

---

### 2.2 B invariance verified for 5 ecological systems

**Statement:** B invariance (CV < 5%) verified for kelp, savanna, lake, coral, and toggle.

**Conclusion:** VERIFIED

**Scripts:** `structural_B_kelp.py`, `structural_B_savanna.py`, `structural_B_coral.py`, `structural_connection_test.py`, `structural_B_toggle.py`

**Key numbers:**

| System | D_target | B mean | B CV | DeltaPhi variation | Method |
|--------|---------|--------|------|-------------------|--------|
| Kelp | 29.4 | 2.17 | 2.6% (detrended) | 27,617x | Exact MFPT |
| Savanna | 100 | 4.04 | 4.6% | 77x | Exact MFPT |
| Lake | 200 | 4.27 | 2.0% | 96x | Exact MFPT |
| Coral | 1,111 | 6.06 | 2.1% | 2,625x | Exact MFPT |
| Toggle | 1,000 | 4.83 | 3.8% | -- | CME spectral regression |

**Confidence grade:** B

**Flags:**
1. Kelp shows smooth monotonic drift (B: 2.58->1.60 as p increases). After detrending, residual CV = 2.55%.
2. Toggle uses CME spectral data regression (different method from MFPT). Toggle B CV is D-target dependent: 16.7% at D=100, 3.8% at D=1000, 6.2% at D=10000.
3. "5 systems" = 4 ecological + 1 toggle. The toggle is methodologically distinct and is not an ecological system.

---

### 2.3 Toggle B = 4.83 +/- 3.8%

**Statement:** Toggle switch B approximately constant at 4.83 with CV 3.8% across alpha in {5,6,8,10}.

**Conclusion:** VERIFIED

**Script:** `structural_B_toggle.py`

**Input data:** Gardner 2000 toggle model, Hill n=2, alpha in {5,6,8,10}, D_target=1000, CME spectral data.

**Key numbers:** B mean = 4.83, B CV = 3.8% (at D_target=1000, excluding alpha=3 near bifurcation onset).

**Confidence grade:** B

**Flags:**
1. D-target dependence: CV = 16.7% at D=100, 9.1% at D=1000, 3.8% at D=10000. The 3.8% is the favorable choice.
2. Including alpha=3: B CV = 9.1%.
3. Method is CME spectral regression, not quasi-potential MFPT.

---

### 2.4 beta_0 constant at fixed ODE (CV = 2.9%)

**Statement:** At fixed ODE shape, beta_0 = ln(D) - B has CV = 2.94% across 49 epsilon combinations spanning D = 11 to 10,000.

**Conclusion:** VERIFIED

**Script:** `patha_v2_what_determines_B.py` (Test 3)

**Input data:** Lake ODE at a=0.3266 (fixed), 7x7 grid of (eps_1, eps_2).

**Key numbers:** beta_0 mean = 1.030, beta_0 CV = 2.94%, B vs ln(D) slope = 1.016, R^2 = 0.99993.

**Confidence grade:** A

**Flags:** None. Rigorous mathematical result at a single ODE point.

---

### 2.5 beta_0 varies across systems [0.53, 1.61]

**Statement:** beta_0 is NOT universal; ranges from 0.535 (savanna) to 1.611 (kelp), driven by eigenvalue ratio |lambda_eq/lambda_sad|.

**Conclusion:** VERIFIED

**Script:** `patha_v2_what_determines_B.py` (Test 1)

**Key numbers:**

| System | D | B | beta_0 | |lambda_eq/lambda_sad| |
|--------|---|---|--------|----------------------|
| Savanna | 100 | 4.070 | 0.535 | 0.149 |
| Lake | 200 | 4.265 | 1.033 | 0.639 |
| Coral | 1,111 | 6.040 | 0.973 | 1.719 |
| Kelp | 29.4 | 1.770 | 1.611 | 6.029 |

beta_0 = ln(K_corr) + ln(pi) + 0.5*ln(|lambda_eq/lambda_sad|). Formula matches to machine precision for all 4 systems.

**Confidence grade:** A

**Flags:** None.

---

## CATEGORY 3: D = 1 Threshold

### 3.1 D < 1 achievable via Kramers at high noise

**Statement:** At sigma = 1.0, D_exact = 0.68 via Kramers MFPT.

**Conclusion:** VERIFIED

**Script:** `X2/scripts/test_D_below_one_fast.py`

**Input data:** Standard lake model, sigma sweep 0.05-5.00.

**Key numbers:** sigma=0.80: D=1.09; sigma=1.00: D=0.68; sigma=1.50: D=0.30. Crossover at sigma/sigma* ~ 5.6.

**Confidence grade:** A

**Flags:** None.

---

### 3.2 D < 1 impossible via product equation (saddle-node protects)

**Statement:** Bistability is destroyed at saddle-node bifurcation before epsilon can push D below ~4.

**Conclusion:** VERIFIED

**Input data:** Lake model, epsilon sweep with eps_2=0.10 fixed.

**Key numbers:** eps_1=0.78: D=12.8 (bistable, DeltaPhi=0.164). eps_1=0.79: bistability lost (saddle-node). Two-channel: bistability lost at D~4.0.

**Confidence grade:** A

**Flags:** None.

---

### 3.3 D = 1 NOT in existing literature as named threshold

**Statement:** Exhaustive literature search finds no prior naming of MFPT/tau_relax = 1 as a dissipative structure criterion.

**Conclusion:** VERIFIED

**Input data:** 14 major references (Kramers 1940 through 2025); 14 cross-domain Kramers applications; 6 nearest-miss papers.

**Key numbers:** Nearest analog: Deborah number (De = tau_relax / tau_observation), threshold De=1 -- structurally different (uses observation timescale, not MFPT). Bovier 2004 proved metastability iff MFPT >> tau_relax but never connected to dissipative structures.

**Confidence grade:** B

**Flags:** Literature review methodology does not specify search strings, databases, or date ranges. "Exhaustive" is a strong claim inherently subject to incompleteness.

---

### 3.4 D ~ 1 confirmed at stellar level (Study 25, previously S11)

**Statement:** D ~ 1-6 for star-forming cloud cores, computed via Kramers MFPT (Gardiner integral) for a Bonnor-Ebert virial potential with 0 free parameters.

**Conclusion:** VERIFIED

**Script:** `../scripts/study25_stellar_kramers.py`

**Key numbers:** For star-forming cores (M/M_BE = 0.80-1.22): D = 2-6 at Mach 1.0, D = 1-4 at Mach 1.5. Fiducial (1 M_sun, T=10K, P/k=2e5): D = 1.96 at Mach 1, MFPT = 1.75 t_ff. B = 0.22-1.07 across parameter sets. Consistent with observed core lifetimes of 1-3 t_ff (Enoch+ 2008).

**Confidence grade:** A

**Method:** 1D virial potential phi(x) = -3/x - 3*ln(x) + gamma*x^3 derived from energy of a uniform-density isothermal sphere (gravity + thermal pressure + external confinement). Equilibria found via Brent. Exact MFPT via Gardiner integral (300,000 grid points). Noise from turbulent Mach number (Pineda+ 2010). Physical parameters from McKee & Ostriker 2007, Andre+ 2014.

**Physical basis:** The virial theorem enforces B << 1 for marginally stable gravitational systems: the barrier is a fraction f ~ 0.03-0.15 of the binding energy, while the turbulent kinetic energy equals the binding energy (virial equilibrium). Therefore B = f and D ~ O(1) structurally.

**Previous status:** WEAK CHAIN (Grade B) based on S11's Elbaz M^0 dimensional argument. Upgraded to VERIFIED (Grade A) by direct Kramers computation (Study 25, 2026-04-06).

---

### 3.5 D = 1 confirmed at chemical level (S12)

**Statement:** D ~ 1 for chemistry; sigma_combustion / sigma_metabolism = 1.04 (state function constraint). Search/persistence distinction is purely eta-driven.

**Conclusion:** VERIFIED

**Input data:** Published thermochemistry: sigma_combustion(298K) = 9,631 J/(K*mol); sigma_metabolism(310K) = 9,258 J/(K*mol).

**Key numbers:** Ratio = 1.04 (< 5%). eta_respiration/eta_fermentation = 100%/7.9% = 12.6x (> 5x threshold). 9/10 robustness tests pass.

**Confidence grade:** A

**Flags:** Enzyme half-life is organism-dependent (E. coli passes, yeast median fails).

---

### 3.6 D = 1.16 for Rome (S23)

**Statement:** Rome: D = tau_regulated / tau_collapse = 500/432 = 1.16.

**Conclusion:** WEAK CHAIN

**Input data:** tau_regulated = 500 yr (S5 master table, Imperial period 27 BCE - 476 CE); tau_collapse = 432 yr (Cooper et al. 2020 dataset).

**Key numbers:** D = 500/432 = 1.16. Arithmetic verified.

**Confidence grade:** D

**Flags:**
1. CRITICAL METHODOLOGICAL DISCONTINUITY: This D uses historical timescale ratios, not stochastic-theoretic MFPT/tau_relax. No ODE model underlies the computation. tau_collapse is a historical decline duration, not a Kramers escape time.
2. No epsilon values are measured or inferred for Rome.
3. Only 2 civilizational D values (Rome=1.16, Maya=6.0) -- insufficient for any formal test.
4. The S23 study itself flags this as speculative.

---

## CATEGORY 4: Architecture Scaling

### 4.1 f(k) = alpha^k with R^2 = 0.997

**Statement:** Bistable fraction decays exponentially with channel count k.

**Conclusion:** VERIFIED

**Script:** `s0_derivation_architecture_scaling.py`

**Input data:** Lake model, k in {1..6}, 5000 trials/k, 3 independent seeds (42, 137, 2718).

**Key numbers:**

| k | f(k) | Bistable count |
|---|------|---------------|
| 1 | 0.3389 | 1694/5000 |
| 2 | 0.1213 | 606/5000 |
| 3 | 0.0424 | 212/5000 |
| 4 | 0.0166 | 79/5000 |
| 5 | 0.0054 | 21/5000 |
| 6 | 0.0027 | 6/5000 |

R^2 = 0.9973 (exponential). Power-law R^2 = 0.9476. Stretched exponential R^2 = 0.99997 with gamma ~ 0.99 ~ 1.0, confirming pure exponential.

**Confidence grade:** A

**Flags:** None.

---

### 4.2 alpha = 0.373 +/- 0.02

**Statement:** Per-channel bistability survival probability alpha = 0.373.

**Conclusion:** VERIFIED

**Script:** Same as 4.1.

**Key numbers:** alpha = 0.373360; A = 0.859868. 95% CI from analysis: [0.333, 0.423]. Stated +/- 0.02 encompasses [0.35, 0.39].

**Confidence grade:** B

**Flags:** 95% CI [0.333, 0.423] is wider than stated +/- 0.02. The stated uncertainty may be optimistic. alpha ~ 1/e = 0.368 is noted as "suggestive but not conclusive from 6 data points."

---

### 4.3 k* = 30.4 for S0 = 10^13

**Statement:** Extrapolation: k* = log_{1/alpha}(S0) = 30.4 at S0 = 10^13.

**Conclusion:** VERIFIED (arithmetic only)

**Key numbers:** 1/alpha = 2.6784; k* = ln(10^13)/ln(2.6784) = 30.38. Range for S0 = 10^(13+/-0.8): [28.5, 32.3].

**Confidence grade:** C

**Flags:** The arithmetic is exact given alpha and S0. But S0 = 10^13 is an external assumption not derived in this computation. k* has no independent validation.

---

### 4.4 Bistability selects low Hill exponents (n <= 2 in 65%)

**Statement:** ~65% of channels in bistable systems have Hill exponent n <= 2.

**Conclusion:** VERIFIED

**Key numbers:** Mean Hill (bistable): 2.65 +/- 0.04; mean Hill (non-bistable): 4.5-5.4. Robust across k=1-6.

**Confidence grade:** A

**Flags:** None.

---

### 4.5 DeltaPhi invariant across k (approx 0.075 for k=1-5)

**Statement:** Potential barrier DeltaPhi ~ 0.075 across channel counts k=1-5.

**Conclusion:** VERIFIED

**Key numbers:**

| k | DeltaPhi mean | DeltaPhi std | CV | n |
|---|-------------|-------------|-----|---|
| 1 | 0.0689 | 0.0197 | 29% | 1694 |
| 2 | 0.0743 | 0.0285 | 38% | 606 |
| 3 | 0.0754 | 0.0286 | 38% | 212 |
| 4 | 0.0795 | 0.0317 | 40% | 79 |
| 5 | 0.0838 | 0.0349 | 42% | 21 |
| 6 | 0.1121 | 0.0360 | 32% | 8 |

Average k=1-5: 0.0764 (claim: ~0.075). Means vary 22% (0.0689->0.0838). Within-k CV is high (29-42%).

**Confidence grade:** B

**Flags:**
1. "Invariant" is relative: means vary 22% from k=1 to k=5, and within-k scatter is large.
2. k=6 shows uptick (0.1121) but based on only 8 trials.
3. The important point is that DeltaPhi stays bounded while D_product grows super-exponentially.

---

### 4.6 Cusp bridge: S0 from bifurcation geometry

**Statement:** Cusp normal form provides analytic bridge between search equation and persistence equation. B_cusp = 2.979 (CV=3.7%), K_cusp = 0.558, P(bistable|2D) = 2/5 exactly. S0 = exp(gamma*d + c0) with d = 105-217 parameters for S0 = 10^13.

**Conclusion:** VERIFIED

**Script:** `cusp_bridge_derivation.py`

**Key numbers:** DeltaPhi = x3*(x2-x1)^3/4 verified to <10^-10. B_cusp = 2.979 (n=300). K_cusp = 0.558. gamma ~ 0.14-0.41 per parameter. For S0 = 10^13: 35-72 independent channels.

**Confidence grade:** A

**Flags:** gamma range is broad (3x). P(bistable|2D) = 2/5 is exact for cusp; real systems may differ.

---

## CATEGORY 5: Cross-Domain Evidence

### 5.1 HIV Omega* = 2296 mL

**Statement:** Conway-Perelson 5D model yields Omega* = 2296 mL; structural computation gives 2293 mL (0.1% match).

**Conclusion:** VERIFIED (conditional)

**Script:** `step13_hiv_structural_omega.py`

**Key numbers:** Omega* (SDE bisection) = 2296 mL; Omega* (structural, Kramers-regime fit) = 2293 mL; match = 0.1%. 5D MAP optimization NOT converged (s* = 2.4e-3 vs SDE-calibrated 9.7e-4, 2.5x gap). Noise anisotropy: D_TT/D_LL = 3.2x10^9.

**Confidence grade:** B

**Flags:**
1. The structural pass is conditional on SDE calibration. Analytical MAP alone gives Omega* = 566 mL (4.1x error).
2. Omega* = 2296 mL = 46% of blood volume, consistent with lymphoid tissue (1-3 L, Haase 1999).

---

### 5.2 Tumor-immune B = 2.73 +/- 2.7% (2D SDE)

**Statement:** Kuznetsov 1994 2D tumor-immune model gives B = 2.73 with CV = 2.7% (2D SDE, definitive; 1D gave 3.98, superseded).

**Conclusion:** VERIFIED

**Script:** `structural_B_tumor_immune.py`

**Key numbers:** B mean = 2.73, B CV = 2.7%, B range [2.65, 2.89] (2D SDE, 10 scan points). 1D result (B=3.98, superseded) retained for comparison. DeltaPhi varies 3.4x. Within stability window [1.8, 6.0].

**Confidence grade:** B

**Flags:**
1. DeltaPhi varies only 1.8x (vs 96-27,617x in ecological systems). B invariance test is less stringent.
2. B CV = 5.1% is the highest of verified systems.

---

### 5.3 Thermohaline K = 0.55, 0 free parameters

**Statement:** Cessi 1994 1D model achieves K = 0.55 with 0 free parameters across the bistable range.

**Conclusion:** VERIFIED

**Script:** `thermohaline_kramers.py`

**Input data:** Cessi 1994 J. Phys. Oceanogr.; mu^2=4; t_d=219 yr. p scanned (not fitted).

**Key numbers:** K = 0.55-0.57 (thermal mode), 0.55-0.61 (saline mode). Best accuracy: 0.7% at p=0.96. Worst: 7.1%. Inverse bridge: sigma* matches ERA-Interim freshwater flux to ~9% (1 free parameter: p).

**Confidence grade:** A

**Flags:** Inverse bridge test uses 1 free parameter (p, glacial forcing level). Modern vs glacial variability comparison adds uncertainty.

---

### 5.4 Toggle shortcut D(alpha, Omega) = exp(a(alpha) + S(alpha)*Omega)

**Statement:** Closed-form formula with 3-15% accuracy from CME spectral data.

**Conclusion:** VERIFIED

**Script:** `toggle_shortcut.py`, `structural_B_toggle.py`

**Key numbers:** Power-law exponent: 1.22 (vs saddle-node prediction 1.5). Typical accuracy: 3%. Worst: 15%. 76 CME data points, R^2 > 0.99.

**Confidence grade:** A

**Flags:** Exponent 1.22 differs from near-bifurcation prediction 1.5 (crossover regime).

---

### 5.5 14 domains with independent Kramers applications

**Statement:** Framework identifies 14 systems across domains where Kramers escape applies.

**Conclusion:** WEAK CHAIN

**Input data:** 8-domain literature review (~30 candidates).

**Key numbers:** 14 systems identified with bistability. 0 Grade A candidates found. Best: Frank 2021 macrophage (B+); Dovzhenok 2012 DA neuron (B-); Lac operon (B-).

**Confidence grade:** C

**Flags:**
1. This is a NEGATIVE finding: zero non-ecological systems meet the product equation criteria (separable additive channels + calibrated parameters).
2. Molecular bistable systems universally have k=1 (single feedback loop). The product equation requires k >= 2 with independent channels.

---

## CATEGORY 6: D Staircase and Cross-Scale

### 6.1 D staircase: D = 1 -> 10^2 -> 10^4

**Statement:** Empirical step function across hierarchical levels.

**Conclusion:** VERIFIED (as empirical pattern)

**Key numbers:**

| Level | D | Source |
|-------|---|--------|
| Stars (S11) | ~1 | Elbaz cancellation |
| Chemistry (S12) | ~1 | State function |
| Molecular clouds | 50-100 | KM05 derivation |
| Galaxies | 67 | tau ratio |
| Clusters | 10-50 | Fabian balance |
| Ecosystems | 29-1111 | Product equation |

Gap 0->1: ~1.8 OOM; gap 1->2: ~2.0 OOM. Symmetric within 0.18 OOM.

**Confidence grade:** C

**Flags:**
1. This is a 3-level pattern with 3 parameters -- not diagnostic.
2. The step function is empirical, not derived.
3. D = 10^(2I) is falsified at I < 2 (clouds have D=50-100 at I=0).

---

### 6.2 Molecular clouds D = 50-100 from first principles

**Statement:** KM05 turbulence theory gives epsilon_ff = 1.4-2.2%, hence D = 46-71.

**Conclusion:** VERIFIED

**Key numbers:** epsilon_ff from KM05 Eq. 30 at Mach 25-100: 1.4-2.2%. D_theory = 46-71. D_observed = 50-100. Offset <= 0.34 dex.

**Confidence grade:** A

**Flags:** None. This is the one D value derived purely from first principles.

---

### 6.3 Galaxies D = 67

**Statement:** D = tau_persistence/tau_search = 10 Gyr / 150 Myr = 67.

**Conclusion:** VERIFIED

**Key numbers:** Range: D = 10-240 (+/- 0.7 dex). Central: 67. Converges with D_cloud ~ 67 and D_cluster ~ 10-50.

**Confidence grade:** B

**Flags:** Large uncertainty range. epsilon is back-derived.

---

### 6.4 Galaxy clusters D = 10-50

**Statement:** Cooling-flow balance gives D = 10-50.

**Conclusion:** VERIFIED (cursory)

**Confidence grade:** C

**Flags:** Least detailed; acknowledged as "cursory" in source.

---

### 6.5 Rome D = 1.16, Maya D = 6.0

**Statement:** Civilizational D from historical timescale ratios.

**Conclusion:** WEAK CHAIN

**Key numbers:** Rome: 500/432 = 1.16. Maya: 600/100 = 6.0. Arithmetic correct.

**Confidence grade:** D

**Flags:** See Claim 3.6. Historical ratios, no ODE model, no epsilon measurements. Only 2 data points.

---

### 6.6 S30 civilizational k values (Rome ~ 0.1, Tokugawa ~ 2, China ~ 2.5)

**Statement:** Effective feedback channel count k derived from D = (1/epsilon)^k.

**Conclusion:** BROKEN CHAIN

**Key numbers:** Rome k~0.1, Tokugawa k~0.5-0.8, China k~2-3. All reverse-engineered from observed D and assumed epsilon.

**Confidence grade:** D

**Flags:**
1. k is computed BACKWARD: k = ln(D) / ln(1/epsilon). This is circular.
2. k values are non-integer (0.1, 0.5), violating the model assumption that k counts discrete channels.
3. epsilon is theoretical, not measured.
4. S19 explicitly states: "N cannot be independently defined."

---

## CATEGORY 7: Legacy Study Connections

### 7.1 Ganti M+T+B = product equation channels (S12, S13)

**Statement:** The M+T+B structure of Ganti chemotons maps onto product equation channels with eta gradient 0% -> 8% -> ~60% -> 100%.

**Conclusion:** VERIFIED

**Key numbers:** eta_fermentation = 7.9%, eta_respiration = 100%, ratio = 12.6x. 5 independent approaches converge. 9/10 robustness tests pass.

**Confidence grade:** A

**Flags:**
1. The eta values are fuel utilization fractions, not epsilon (regulatory coupling). The mapping is eta_regulator = 1-epsilon, which is consistent but these are different quantities.
2. M+T level (~50-70%) is estimated from biochemical reasoning, not measured.

---

### 7.2 Origin of life = first D > 1 crossing (S13)

**Statement:** The origin of life is the first search-to-persistence transition (D crosses above 1).

**Conclusion:** VERIFIED (interpretive)

**Key numbers:** eta progression: 0% (abiotic) -> 7.9% (M only) -> ~60% (M+T) -> 100% (M+T+B). D > 1 inferred when persistence advantage exceeds decay timescale.

**Confidence grade:** C

**Flags:**
1. D is NOT computed numerically for any prebiotic system. This is an interpretation, not a calculation.
2. No ODE model of early autocatalytic networks with quantified D exists.
3. "D = 1 crossing" is asserted from the eta mapping, not demonstrated from Kramers theory.

---

### 7.3 Override resistance = k (S28)

**Statement:** Override resistance (subjective 1-5 score) predicts D with R^2 = 0.95.

**Conclusion:** WEAK CHAIN

**Key numbers:**
- S + override (N=8): R^2 = 0.951
- S alone (N=19): R^2 = 0.490
- **Anchor-only (N=7): R^2 = 0.019, p = 0.77 (null)**

**Confidence grade:** B

**Flags:**
1. CRITICAL: When restricted to highest-quality D values (anchor-only, N=7), R^2 = 0.019. The R^2=0.95 at N=8 is driven by estimated D values.
2. Override resistance is subjective manual scoring (1-5).
3. S28 explicitly states: "The anchor-only result is the most important number."

---

### 7.4 eta * tau decomposition = bridge identity (S14)

**Statement:** At evolutionary transitions, ln(D) decomposes into eta (gradient) and tau (duration) components matching the bridge structure.

**Conclusion:** WEAK CHAIN

**Key numbers:** Pattern broadly confirmed: pre-informational transitions are eta-dominated; post-informational are tau-dominated. T3 is anomalous (eta-dominated despite being post-informational).

**Confidence grade:** C

**Flags:**
1. S14 does NOT compute B (barrier height) or verify ln(D) = B + beta_0 for any transition.
2. T2 and T7 estimates are qualitative only.
3. The bridge identity was not tested algebraically; the eta*tau pattern is described as consistent with it.

---

## CATEGORY 8: Constraint/Channel Hierarchy

### 8.1 "Each channel ~ 10 constraints" / 1/epsilon ~ (1/alpha)^10

**Statement:** (2.7)^10 ~ 20,000, explaining the relationship between channels and constraints.

**Conclusion:** BROKEN CHAIN / ARITHMETIC ERROR

**Source file searched:** 
**Finding:** The "10 constraints per channel" claim does NOT appear in source files. S0_DERIVATION_RESULTS.md reports alpha = 0.373 and k* = 30.4 for S0 = 10^13 but makes no claim about "10 constraints per channel." (2.7)^10 = 13,897, not 20,000 or 33. The actual architecture result is k* ~ 30 total sub-steps for S0 = 10^13; this does not map to "10 per channel."

**Confidence grade:** X

**Flags:** Possible confusion: k* = 30 total / 3 major stages (M/T/B) ~ 10/stage. But this interpretation is speculative and not present in source files.

---

### 8.2 S0 = D^(constraints/channels)

**Statement:** Power-law relationship between S0, D, constraints, and channels.

**Conclusion:** BROKEN CHAIN

**Source files searched:** S0_DERIVATION_RESULTS.md, CONSTRAINTS.md, DERIVATION.md

**Finding:** This formula does not appear in any source file.

**Confidence grade:** X

---

## CATEGORY 9: Negative Results

### 9.1 Diabetes B invariance fails (CV = 80.4%)

**Statement:** Topp diabetes model B varies from 0.8 to 189 (factor 240x), CV = 80.4%.

**Conclusion:** VERIFIED

**Key numbers:** sigma_m* varies 2x while DeltaPhi varies 890x. Basin width dominates MFPT, not barrier height. Absorbing boundary (beta=0) and multiplicative noise break the standard Kramers structure.

**Confidence grade:** A

**Flags:** None. Honest negative result with identified mechanism. 2D SDE reverses this: B=5.54, CV=5.2%. See claim 10.1.

---

### 9.2 Product equation fails for toggle (no consistent epsilon)

**Statement:** 9 epsilon definitions x 4 alpha values all fail; Omega* monotonically declining.

**Conclusion:** VERIFIED

**Key numbers:** Best definition (2a): Omega* = 3.48->1.84 (declining with alpha). D_product/D_CME ratio: 0.72-1.37 (factor-of-2 spread). Root cause: single coupled loop, not separable channels. J_12/J_21 asymmetry: 23x-98x.

**Confidence grade:** A

**Flags:** None. Definitive negative leading to the separability criterion (Fact 31).

---

### 9.3 F2 = (1/epsilon)^n fails (trophic != regulatory)

**Statement:** Trophic efficiency ratios do not equal regulatory depth ratios.

**Conclusion:** VERIFIED

**Confidence grade:** A

---

### 9.4 Driver count != channel count (R^2 = 0.019)

**Statement:** RSDB driver count does not predict D; anchor-only R^2 = 0.019.

**Conclusion:** VERIFIED

**Key numbers:** Driver alone (N=19): R^2 = 0.259, p=0.026 (weak). Anchor-only (N=7): r=-0.137, R^2=0.019, p=0.77 (null). Root cause: driver count = OR-gates (any one triggers collapse); override = AND-gates (all must fail simultaneously). Kelp: 21 drivers but D=33 (only one needed). Forest: 13 drivers but D=4600 (5 must fail).

**Confidence grade:** A

**Flags:** None.

---

### 9.5 Loop count wrong sign (r = -0.575)

**Statement:** Stabilizing feedback loop count shows negative correlation with D.

**Conclusion:** VERIFIED

**Key numbers:** S29 loop count: R^2 = 0.082, wrong sign. Marked explicitly in source as "WRONG SIGN."

**Confidence grade:** A

**Flags:** Source does not explain WHY the sign is wrong. Hypothesis: ecosystems with many loops may have weak individual couplings.

---

### 9.6 FVS topology alone insufficient (R^2 = 0.051)

**Statement:** Minimum feedback vertex set shows R^2 = 0.051 for predicting D.

**Conclusion:** VERIFIED

**Key numbers:** |FVS| alone (N=19): r=+0.227, R^2=0.051, p=0.11. Anchor-only (N=7): r=-0.109, R^2=0.012, p=1.00. FVS clusters at 1-2 across D spanning 3+ OOM. Root cause: FVS = topological codimension, not coupling strength.

**Confidence grade:** A

**Flags:** None. Well-documented negative.

---

## SUMMARY TABLE

| # | Claim | Conclusion | Grade | Source | Flags |
|---|-------|-----------|-------|--------|-------|
| 1.1 | Product eq 6 systems | VERIFIED | B | R01, SV10, SV13 | eps grades B-C; 2 systems need free params |
| 1.2 | mu-analysis derivation | VERIFIED | B | S30 Phase 1 | FVS test failed; block-diagonal assumed |
| 1.3 | Kramers across systems | VERIFIED | A | R10, SV06-13, KA-TH, XS3 | K varies 0.34-0.56 by regime |
| 1.4 | Search eq S0=10^13 | VERIFIED | B | SV12-SV12D | Corrected from 29->5.1 OOM; slope -0.95 |
| 1.5 | Bridge ln(D)=B+beta0 | VERIFIED | A | SE-BP | 240 MFPT; beta0 not universal |
| 1.6 | Ceiling D=1/eps at k=1 | VERIFIED | A/B | S19 | Clouds derived; galaxies measured |
| 2.1 | B CV<3% all q | VERIFIED | A | SE-BI | 240 computations |
| 2.2 | B invariance 5 systems | VERIFIED | B | SE-BI | Toggle uses different method |
| 2.3 | Toggle B=4.83+/-3.8% | VERIFIED | B | SE-BI, XS3 | D-target dependent |
| 2.4 | beta0 CV=2.9% fixed ODE | VERIFIED | A | SE-BP | 49 eps combinations |
| 2.5 | beta0 in [0.53,1.61] | VERIFIED | A | SE-BP | Eigenvalue ratio driven |
| 3.1 | Kramers D<1 at high sigma | VERIFIED | A | SE-DT | sigma=1.0, D=0.68 |
| 3.2 | Saddle-node protects | VERIFIED | A | SE-DT | eps=0.78, D=12.8 |
| 3.3 | D=1 novel in literature | VERIFIED | B | SE-DT | 14 works; Deborah # nearest |
| 3.4 | D~1 stellar | VERIFIED | A | Study 25 | Kramers MFPT, D=2-6 at Mach 1, 0 free params |
| 3.5 | D=1 chemical | VERIFIED | A | S12 | State function; 5 approaches |
| 3.6 | D=1.16 Rome | WEAK CHAIN | D | S23 | Historical ratio; no ODE |
| 4.1 | f(k)=alpha^k R^2=0.997 | VERIFIED | A | S0-DERIV | 5000 trials/k; 3 seeds |
| 4.2 | alpha=0.373+/-0.02 | VERIFIED | B | S0-DERIV | 95% CI wider than stated |
| 4.3 | k*=30.4 | VERIFIED (arith) | C | S0-DERIV | Depends on S0=10^13 |
| 4.4 | n<=2 in 65% bistable | VERIFIED | A | S0-DERIV | Robust k=1-6 |
| 4.5 | DeltaPhi~0.075 k=1-5 | VERIFIED | B | S0-DERIV | Means vary 22%; high within-k CV |
| 4.6 | Cusp bridge S0 derivation | VERIFIED | A | CUSP_BRIDGE | B_cusp=2.979, K_cusp=0.558 |
| 5.1 | HIV Omega*=2296 | VERIFIED | B | SV13-HIV, XS5 | MAP not converged; SDE-calibrated |
| 5.2 | Tumor B=2.73+/-2.7% (2D SDE) | VERIFIED | B | DR-TI | 1D gave 3.98, superseded |
| 5.3 | Thermohaline K=0.55 | VERIFIED | A | KA-TH | 0 free params; 0.7-7% accuracy |
| 5.4 | Toggle shortcut | VERIFIED | A | SE-S4, XS3 | 76 CME pts; R^2>0.99 |
| 5.5 | 14 domains Kramers | WEAK CHAIN | C | DR-CDS | Negative: 0 Grade A candidates |
| 6.1 | D staircase 1->10^2->10^4 | VERIFIED | C | S18,S19,S21 | 3-point pattern; empirical |
| 6.2 | Clouds D=50-100 | VERIFIED | A | S19 | KM05; 0.34 dex match |
| 6.3 | Galaxies D=67 | VERIFIED | B | S21 | +/-0.7 dex uncertainty |
| 6.4 | Clusters D=10-50 | VERIFIED | C | S21 | Cursory |
| 6.5 | Rome D=1.16, Maya D=6.0 | WEAK CHAIN | D | S23 | Historical; no ODE |
| 6.6 | Civilizational k values | BROKEN CHAIN | D | S23,S30 | Reverse-engineered; non-integer |
| 7.1 | Ganti M+T+B mapping | VERIFIED | A | S12,S13 | eta != eps; consistent mapping |
| 7.2 | Origin of life = D>1 | VERIFIED (interp) | C | S13 | No D computed |
| 7.3 | Override resistance = k | WEAK CHAIN | B | S28 | Anchor R^2=0.019 |
| 7.4 | eta*tau = bridge | WEAK CHAIN | C | S14 | Bridge not algebraically tested |
| 8.1 | 10 constraints/channel | BROKEN CHAIN | X | -- | Not in source files |
| 8.2 | S0=D^(n/k) | BROKEN CHAIN | X | -- | Formula not found |
| 9.1 | Diabetes 1D B fails CV=80% | VERIFIED | A | DR-DIABETES | Honest negative (1D); 2D SDE reverses: B=5.54 |
| 9.2 | Toggle eps fails | VERIFIED | A | SV09 | 9 defs x 4 alpha |
| 9.3 | F2=(1/eps)^n fails | VERIFIED | A | S23 | 42 ecosystems, r=-0.22 |
| 9.4 | Driver count null | VERIFIED | A | S28 | R^2=0.019 anchor |
| 9.5 | Loop count wrong sign | VERIFIED | A | S30 | r=-0.575 |
| 9.6 | FVS fails R^2=0.051 | VERIFIED | A | S30 | Topological only |
| **26.1** | **Sigma existence constraint** | **VERIFIED** | **B** | **Study 26** | **93.5% prior eliminated; partially circular** |
| **27.1** | **S(d) model selection** | **VERIFIED** | **A** | **Study 27** | **Power law rejected dAICc=+12.63** |
| **28.1** | **Xenopus product eq blind test** | **VERIFIED** | **A** | **Study 28** | **LOW passes (B=3.45); HIGH fails (strong channels)** |
| **30.1** | **Data collapse 13 systems** | **VERIFIED** | **A** | **Study 30** | **slope=1, beta_0=1.06+/-0.37 (1D)** |
| **31.1** | **Protein stability window** | **VERIFIED** | **A** | **Study 31** | **22/25 B>6.0; scope boundary = structurally coupled noise** |
| 10.1 | Diabetes B invariance | VERIFIED (both results) | A | DR-DIA | 1D B fails (CV=80.4%); 2D SDE: B=5.54+/-5.2%, inside stability window |
| 10.2 | Tumor-immune B=2.73 (2D SDE) | VERIFIED | A | DR-TI | First non-ecological B in stability window; 1D gave 3.98, superseded |
| 10.3 | Financial cusp K=0.55 | VERIFIED | B | KA-FIN | Generic params; product eq N/A |
| 10.4 | Soviet tau=3.9yr | CONSISTENCY CHECK | D | KA-SOV | 4 free params, 0 calibrated |
| 10.5 | Power grid K->1.0 | VERIFIED (structure) | A | KA-PG | D_obs undefined; kappa^(3/2) to 0.1% |
| 11.1 | JJ B invariance CV=0.3% | VERIFIED | A | blind_test_josephson_junction.py | 1D analytic potential; noise external |
| 11.2 | Nanoparticle B CV=1.5-2.7% | VERIFIED | A | blind_test_magnetic_nanoparticle.py | Strongest asymmetry test |
| 11.3 | Intermediate k: 6/6 pass | VERIFIED | A | step12e_intermediate_k.py | k=18-27 biologically plausible |
| 11.4 | Alpha model-dependent | VERIFIED | A | alpha_2d_savanna/toggle/targeted | Three experiments, alpha 0.37-0.91 |
| 11.5 | Cusp bridge gamma~0.20 | VERIFIED | B | bridge_high_d_scaling.py | d>50 deceleration resolved; ~2x extrapolation to d~150 |
| 15.1 | Blind search: flowers tau=126 Myr vs 200 Myr | VERIFIED | B | blind_flowers_prediction.py | -0.20 OOM; gamma from directed evolution |
| 15.2 | Blind search: CAM tau=204 Myr vs 30 Myr (d=72) | VERIFIED | C | blind_cam_prediction.py | +0.83 OOM; d classification uncertain; multi-origin d~52 |
| 16.1 | Hopf bridge gamma_Hopf=0.172 | VERIFIED | B | hopf_bridge_scaling.py | 4 decay-regime points, R^2=0.9997; single model class |
| 16.2 | gamma_Hopf ≠ gamma_cusp (0.87 ratio) | VERIFIED | B | hopf_bridge_scaling.py | 13% difference; "alpha pattern" |
| 17.1 | B_Bautin = 4.14 (CV 2.2%), in stability window | VERIFIED | B | bautin_bridge.py | B invariance extends to FP-LC bistability |
| 17.2 | gamma_Bautin = 0.021 (flat, no decay) | VERIFIED | B | bautin_bridge.py | FP-LC not exponentially rare; degradation helps |
| 18.1 | Polynomial gamma_poly = 0.010 (flat) | VERIFIED | B | poly_bridge_scaling.py | No decay without saturation; 1.9M samples |
| 18.2 | Poly+degradation gamma = 0.007 (flat) | VERIFIED | B | poly_degradation_test.py | Degradation alone insufficient; 7.8M samples |
| 18.3 | S(d) requires saturation + degradation | VERIFIED | B | Studies 16-18 synthesis | 2x2 matrix complete; biological scope unchanged |
| 19.1 | B bounded by cusp scale invariance | VERIFIED | A | B_bounded_derivation.py | Union B in [1.54, 5.66]; stability window = geometry + selection |

### 15.1 Blind search equation test — angiosperm flowers

**Statement:** The search equation tau = n × exp(gamma × d) / (v × P), using gamma = 0.317 from directed evolution (Keefe & Szostak 2001) and d = 51 from a published floral ODE (van Mourik et al. 2010), predicts tau = 126 Myr at central biological estimates (v = 0.05, P = 10, n = 6). Observed: 200 Myr. Discrepancy: -0.20 OOM. Observed inside predicted range [2.8, 239 Myr].

**Conclusion:** VERIFIED (blind prediction, GREEN grade)

**Script:** `15_blind_fire_tests/blind_flowers_prediction.py`

**Input data:**
- gamma = 0.317: Keefe & Szostak 2001, Nature 410:715 (fraction of random 80-aa proteins binding ATP ≈ 10^-11; gamma = ln(10^11)/80). Grade A.
- d = 51: van Mourik et al. 2010, BMC Syst Biol 4:101 (full floral organ identity ODE, 13 state variables, 51 parameters). Grade A.
- v = 0.05/yr: seed fern generation time (~20 yr). Grade C (debated: early angiosperms may have been herbaceous).
- P = 10: major seed-plant lineages in Permian–Triassic. Grade B.
- n = 6: Dilcher 2000 developmental decomposition. Grade B.
- tau_observed = 200 Myr: fossil record (365 Ma → 130 Ma). Grade B.

**Key numbers:**
- S = exp(0.317 × 51) = 1.05 × 10^7, log10(S) = 7.02
- tau_central = 126 Myr, log10(ratio) = -0.20 OOM
- At v = 0.033 (30-yr woody): d_required = 51.1 (R = 1.00 vs published model)
- Cusp bridge gamma = 0.20 fails (-2.78 OOM); validated as geometric mean of protein (0.317) and RNA (0.136): sqrt(0.317 × 0.136) = 0.208 ≈ 0.20
- Implied k = 16.9 (just below pterosaur flight at k = 18.2)

**Confidence grade:** B (v uncertainty is large; gamma choice is justified but not uniquely constrained)

**Flags:**
1. This is the first search equation test where gamma comes from laboratory measurement and d from an independently published ODE. All inputs are independent of the calibration set.
2. The v uncertainty (0.033–1.0) drives a 1.93 OOM range. The central prediction depends on early seed plants being predominantly woody.
3. The cusp bridge gamma (0.20) fails by -2.78 OOM for this system. The protein-derived gamma (0.317) is required. This suggests the cusp bridge captures an average gamma across search regimes, not the domain-specific rate.
4. No script bugs found. All computations verified by hand.

### 15.2 Blind search equation test — CAM photosynthesis

**Statement:** The search equation tau = n × exp(gamma × d) / (v × P), using gamma = 0.317 from directed evolution (Keefe & Szostak 2001) and d from Bartlett et al. 2014 (CAM ODE), tested at d = 72 (full model), 40 (CAM-specific), and 19 (core). At d = 72: tau = 204 Myr vs observed 30 Myr (+0.83 OOM). At d = 40: tau = 0.008 Myr (-3.57 OOM). Multi-origin test: d ~ 52 explains 62 independent origins. Implied k = 21.2, below C4 (24.3).

**Conclusion:** VERIFIED (blind prediction, YELLOW grade — d classification uncertain)

**Script:** `15_blind_fire_tests/blind_cam_prediction.py`

**Input data:**
- gamma = 0.317: Keefe & Szostak 2001, Nature 410:715 (same as claim 15.1). Grade A.
- d = 72: Bartlett, Vico & Porporato 2014, Plant and Soil (full CAM ODE, 4 state variables, ~72 parameters). Grade A.
- d = 40: CAM-specific subset of Bartlett model (author estimate: circadian + photosynthesis-core + stomatal/storage). Grade C.
- d = 19: Owen & Griffiths / Chomthong 2023 core biochemistry reformulation. Grade B.
- v = 0.1/yr: succulent generation time (~10 yr). Grade C.
- P = 2000: angiosperm lineages with CAM preconditions in arid habitats (inferred from 35+ families). Grade C.
- n = 5: CAM functional sub-steps (PEPc, vacuolar, stomatal, circadian, decarboxylation). Grade B.
- tau_observed = 30 Myr: geometric mean of 65 Myr (first origin, 100→35 Ma) and 15 Myr (Miocene burst). Grade B.
- N_origins = 62: Silvera et al. 2010; Edwards & Ogburn 2012. Grade A.

**Key numbers:**
- S at d=72 = exp(0.317 × 72) = 8.17 × 10^9, log10(S) = 9.91
- S at d=40 = exp(0.317 × 40) = 3.21 × 10^5, log10(S) = 5.51
- tau at d=72 = 204 Myr, log10(ratio) = +0.83 OOM
- tau at d=40 = 0.008 Myr, log10(ratio) = -3.57 OOM
- Multi-origin: d = 52 predicts 62 origins at gamma_protein
- Reverse: d_required = 65.9 at gamma_protein (near full model)
- Implied k = 21.2 (below C4 at 24.3, above bat flight at 20.1)

**Confidence grade:** C (d classification is a judgment call; d=40 fails; d=72 includes environmental parameters; multi-origin test provides independent d constraint)

**Flags:**
1. Second blind search equation test. Uses same gamma (0.317) as claim 15.1 (flowers). All inputs independent of calibration set.
2. The d classification problem is the main uncertainty. d = 72 (full model) includes soil and atmospheric parameters evolution did not invent. d = 40 (CAM-specific, the a priori best estimate) fails badly (-3.57 OOM).
3. The multi-origin test is the powerful new result: d ~ 52 independently explains 62 origins. This d falls between the two a priori estimates and nearly matches the flowers model d = 51.
4. CAM k = 21.2 < C4 k = 24.3, consistent with CAM being biologically simpler (no Kranz anatomy).
5. No script bugs found. All computations verified by hand.

---

## CATEGORY 10: Additional Computational Studies (Not in Original Prompt)

These studies have full ODE models, scripts, and computed results but were not listed in the original 42 claims.

### 10.1 Diabetes B invariance (Topp 2000)

**Statement:** B = 2*DeltaPhi/sigma*^2 scanned across d0 = [0.005, 0.073]; B invariance fails (CV = 80.4%), BUT Kramers works at each fixed d0 with DPP-consistent MFPT.

**Conclusion:** VERIFIED (both the negative B-invariance and the positive Kramers-at-fixed-point)

**Script:** `structural_B_diabetes.py`

**Input data:** Topp et al. 2000, J Theor Biol 206:605-619. BioModels BIOMD0000000341. ALL 9 parameters from published paper. Zero free parameters.

**All ODE parameters traced:**

| Parameter | Value | Source | Terminal? |
|-----------|-------|--------|-----------|
| R0 | 864 mg/dL/day | Topp 2000 Table | YES |
| EG0 | 1.44 /day | Topp 2000 Table | YES |
| SI | 0.72 mL/(muU*day) | Topp 2000 Table | YES |
| sigma_p | 43.2 muU/(day*mg) | Topp 2000 Table | YES |
| alpha | 20000 mg^2/dL^2 | Topp 2000 Table | YES |
| k | 432 /day | Topp 2000 Table | YES |
| d0 | 0.06 /day (scanned) | Topp 2000 Table | YES |
| r1 | 0.84e-3 /(mg/dL)/day | Topp 2000 Table | YES |
| r2 | 2.4e-6 /(mg/dL)^2/day | Topp 2000 Table | YES |

**D_target:** 75, from DPP placebo arm 7%/yr constant conversion -> MFPT ~ 14.3 yr, tau_relax ~ 70 days.

**Key numbers:**
- Adiabatic reduction: 29,670x timescale separation (best of any system)
- B range: [0.79, 188.7]; B CV = 80.4%
- sigma_m* varies only 2x while DeltaPhi varies 890x
- At d0=0.06: B=14.6, MFPT=14.5 yr (matches DPP 14.3 yr)
- B INVARIANCE FAILS (1D) because basin width (not barrier height) dominates MFPT
- Root cause (1D): absorbing boundary (beta=0 is death), not two-well potential

**2D SDE result:** B = 5.54 +/- 5.2% (10 scan points, R^2 = 0.98-0.999). 2D barrier is 3.1x smaller than 1D. All points inside stability window [1.8, 6.0]. Escapes are 100% glucose-driven (fast variable shortcut bypasses slow adiabatic manifold). The 1D failure (CV=80.4%) was an artifact of the invalid adiabatic reduction. **First human disease system verified in the framework.** .md, script: structural_B_diabetes_2D_SDE.py.

**Confidence grade:** A (zero free params, all from BioModels, script reproducible)

**Flags:**
1. 1D B invariance failure is structural (absorbing boundary), not computational error. 2D SDE reverses this.
2. Kramers IS accurate at any fixed d0 -- the constant 7%/yr DPP hazard rate is the Kramers signature.
3. The assessment document (DR_TUMOR_DIABETES_ASSESSMENT) notes diabetes is "computationally cleaner" than tumor-immune but has unmeasurable sigma.
4. Noise channel selection is a genuine structural prediction: only d0-noise (direct beta-cell damage) can drive escape; glucose/insulin noise propagation hits zero at G=175 mg/dL.

---

### 10.2 Tumor-immune B invariance (Kuznetsov 1994)

**Statement:** B = 2.73 +/- 2.7% (2D SDE) across the bistable range of s (CTL influx). Inside stability window [1.8, 6.0]. First verified system outside ecology.

**Conclusion:** VERIFIED

**Script:** `structural_B_tumor_immune.py`

**Input data:** Kuznetsov et al. 1994, Bull Math Biol 56(2):295-321. BioModels BIOMD0000000762. BCL1 lymphoma parameter set. Zero free parameters.

**All ODE parameters traced:**

| Parameter | Value | Units | Source | Terminal? |
|-----------|-------|-------|--------|-----------|
| s | 13,000 (scanned) | cells/day | Kuznetsov 1994 Table | YES |
| p | 0.1245 | 1/day | Kuznetsov 1994 Table | YES |
| g | 2.019e7 | cells | Kuznetsov 1994 Table | YES |
| m | 3.422e-10 | 1/(day*cell) | Kuznetsov 1994 Table | YES |
| d | 0.0412 | 1/day | Kuznetsov 1994 Table | YES |
| a | 0.18 | 1/day | Kuznetsov 1994 Table | YES |
| b | 2.0e-9 | 1/cell | Kuznetsov 1994 Table | YES |
| n | 1.101e-7 | 1/(day*cell) | Kuznetsov 1994 Table | YES |

**D_target:** 1003.8, from BCL1 dormancy MFPT ~ 730 days (Vitetta 1997 Blood), tau_relax = 0.73 days.

**Key numbers:**
- B mean = 2.73, B CV = 2.7% (2D SDE, 10 scan points). 1D result: B=3.98 +/- 5.1% (superseded -- eigenvalue factor 249x mismatch invalidates adiabatic reduction)
- DeltaPhi variation: 3.4x; 1D barrier overestimates 2D by ~5x
- Stability window: INSIDE [1.8, 6.0]
- Kramers R^2 mean = 0.960
- Noise type: environmental (CV=4.4% implied, 56x above demographic 1/sqrt(N))
- 25 scan points across bistable range s in [1151, 20060] (1D); 10 scan points (2D SDE)

**Confidence grade:** A (zero free params, all from BioModels, 25-point scan)

**Flags:**
1. DeltaPhi varies only 1.8x (vs 96x lake, 27,617x kelp). B invariance test is less stringent.
2. Dormant equilibrium is a stable spiral (complex eigenvalues), not a node. Marginal timescale separation (~4.4x).
3. B CV = 5.1% is highest of verified systems (at boundary of "strong" vs "moderate").
4. Mutation-escape confound: dominant clinical escape mechanism is immune-resistant variants (parameter change), not noise-driven barrier crossing.

---

### 10.3 Financial cusp catastrophe (Kramers verified, K=0.55)

**Statement:** Stochastic cusp catastrophe (Zeeman 1974 / Diks-Wang 2009) confirms Kramers with K=0.55, matching ecological interior-well value. Product equation not applicable (k=0 channels).

**Conclusion:** VERIFIED (Kramers structure); NOT APPLICABLE (product equation)

**Script:** `financial_cusp_kramers.py`

**Input data:** Generic cusp dx/dt = -x^3 + qx - a. 6 parameter sets (q=1-4, a=0-0.5). NOT calibrated to Diks-Wang S&P 500 specific values.

**Key numbers:**
- K_eff = 0.50-0.56 for 2*DV/sigma^2 > 2 (moderate-to-deep barrier)
- Scale-invariance confirmed: D_exact depends only on 2*DV/sigma^2, not DV or sigma separately
- At D=20: 2*DV/sigma^2 ~ 1.45-1.61 across all parameter sets
- Prefactor contributes more than barrier (opposite regime from ecology)

**Confidence grade:** B (generic parameters, not calibrated; structure verified across 6 sets)

**Flags:**
1. Generic cusp parameters, not the specific Diks-Wang S&P calibrated values.
2. D_observed targets (D=11-45) are rough estimates with tau_relax ~ 1 yr working assumption.
3. At D~20, barrier ratio ~1.5 is near lower edge of Kramers accuracy.
4. No product equation, no bridge, no sigma* prediction possible.

---

### 10.4 Soviet collapse (Kuran preference falsification, 4 free params)

**Statement:** Kramers topology confirmed for continuous-time Kuran model. tau_relax = 3.9 yr (model) vs 2-5 yr (historical). MFPT = 59 yr vs 44-69 yr (historical).

**Conclusion:** VERIFIED (consistency check only -- underdetermined)

**Script:** `soviet_kuran_kramers.py`

**Input data:** Kuran 1989 Public Choice 61(1):41-74. Discrete threshold model converted to continuous cubic ODE (UNPUBLISHED conversion).

**All parameters:**

| Parameter | Value | Source | Terminal? |
|-----------|-------|--------|-----------|
| S_lo | 0.05 | Assumption (5% opposition) | NO -- not calibrated |
| S_mid | 0.35 | Assumption (35% tipping) | NO -- not calibrated |
| S_hi | 0.90 | Assumption (90% revolution) | NO -- not calibrated |
| gamma | 1.0 /yr | Assumption (~1 yr adjustment) | NO -- not calibrated |

**Key numbers:**
- Barrier DeltaV = 0.003150; revolution well 5x deeper than regime well
- K = 0.34 at sigma=0.05 (2*DV/sigma^2 = 2.5) -- matches kelp/tropical forest boundary wells
- 4 channels identified (ideology, repression, economy, information) -- all shift S_mid, not separable in barrier

**Confidence grade:** D (4 free parameters, 0 calibrated, 2 observables -- underdetermined)

**Flags:**
1. Kuran-to-ODE conversion is unpublished. The cubic is the simplest choice, not unique.
2. 4 free params with 2 observables = underdetermined. Agreement is necessary but not sufficient.
3. std(S) ~ 7.7 pp is unobservable (Kuran's point: preference falsification hides true opposition).

---

### 10.5 Power grid (SMIB swing equation, K->1.0)

**Statement:** Kramers applies cleanly to SMIB swing equation with K_eff -> 1.0 (equal curvature symmetry). Saddle-node kappa^(3/2) scaling verified to 0.1%. But D_observed is undefined (blackouts are cascading/SOC, not Kramers escape).

**Conclusion:** VERIFIED (Kramers structure); D_observed UNDEFINED (wrong failure mode)

**Script:** `power_grid_kramers.py`

**Input data:** Standard SMIB swing equation (textbook); sigma values from Ritmeester & Meyer-Ortmanns 2022.

**Key numbers:**
- K_eff -> 1.0 in deep barrier (equal curvature at stable and saddle -- special symmetry)
- kappa^(3/2) scaling: DeltaV/kappa^1.5 = 1.8866 vs theory 1.8856 (0.05% error)
- At measured noise (sigma=0.01-0.05): 2*DV/sigma^2 ~ 40-1000 (far above ecological ~4)
- Physical interpretation: barrier and noise are DECOUPLED (infrastructure impedance vs wind variability)
- Real blackouts: cascading topology changes (lines trip, K_eff collapses barrier), not noise-driven crossing

**Confidence grade:** A (textbook ODE, kappa scaling exact, sigma from published measurements)

**Flags:**
1. D_observed undefined: blackout sizes follow power laws (SOC), no characteristic MFPT.
2. The framework gives the CORRECT answer: noise-driven desynchronization has infinite escape time at real parameters. Actual failures are parameter-driven bifurcation approach.
3. Product equation not applicable (channels regulate parameters, not drift terms).
4. Bridge provably does NOT hold (2*DV/sigma^2 >> 1), confirming that bridge requires shared physical origin of barrier and noise.

---

## CATEGORY 11: Recent Additions (Blind Tests, Architecture, Cusp Bridge)

### 11.1 Josephson junction -- B invariance CV=0.3-0.4%, K=0.56, 0 free params

**Statement:** Overdamped RCSJ model with tilted washboard potential. Blind test in new domain (superconducting electronics).

**Conclusion:** VERIFIED

**Script:** `blind_test_josephson_junction.py`

**Input data:** Standard overdamped RCSJ model V(phi) = -E_J*cos(phi) - (hbar*I_b/(2e))*phi. Published parameters, 0 free.

**Key numbers:** B CV = 0.3-0.4% across 899x barrier variation. K = 0.56 (parabolic-well 1D SDE class). Kramers accuracy at K=0.55: 4.7% mean error, 9.4% max. B at D=100: 3.26 (inside stability window).

**Confidence grade:** A

**Flags:**
1. 1D with analytic potential. Noise is external (kT), not intrinsic.
2. B invariance for JJ is partially trivial (equal curvatures at moderate tilt). K=0.56 matching ecology is nontrivial.

---

### 11.2 Magnetic nanoparticle -- B invariance CV=1.5-2.7%, K=0.57, 0 free params

**Statement:** Stoner-Wohlfarth uniaxial anisotropy model. Blind test in new domain (nanomagnetism).

**Conclusion:** VERIFIED

**Script:** `blind_test_magnetic_nanoparticle.py`

**Input data:** V(theta) = K_u*V*sin^2(theta) - mu_0*M_s*V*H*cos(theta). Published parameters, 0 free.

**Key numbers:** B CV = 1.5-2.7% across 1111x barrier variation. K = 0.57 (parabolic-well). B at D=100: 3.41 (inside stability window). Curvature ratio varies 1.0 to 0.5 (non-trivial asymmetry).

**Confidence grade:** A

**Flags:**
1. Strongest B invariance test: curvature ratio varies continuously, beta_0 shifts ~20%, yet B absorbs all variation.
2. 1D with analytic potential. Noise is external (kT).

---

### 11.3 Intermediate k innovations -- 6/6 pass at k=18-27

**Statement:** Search equation works at intermediate complexity (k=5-30). S varies with innovation complexity, confirming S(d) is NOT constant.

**Conclusion:** VERIFIED

**Script:** `step12e_intermediate_k.py`

**Input data:** Flight (pterosaurs, birds, bats, insects), C4 photosynthesis, camera-type eyes. Published paleontological/biological timescales.

**Key numbers:** k ranges from 18.2 (pterosaur flight) to 27.1 (camera eyes). S from 10^7.8 to 10^11.6. Flight internally consistent: 4 origins give k=20+/-2 (CV=8.9%). Rankings invariant to v scaling.

**Confidence grade:** A

**Flags:**
1. k vs sub-step correlation is zero (rho=0.03). C4 has 10x more origins than expected.

---

### 11.4 Architecture scaling -- alpha is model-dependent (3 experiments)

**Statement:** Exponential form f(k)=alpha^k is universal across 1D and 2D systems, but alpha depends on system architecture.

**Conclusion:** VERIFIED

**Scripts:** `s0_derivation_architecture_scaling.py`, `alpha_2d_savanna_scaling.py`, `alpha_2d_toggle_scaling.py`, `alpha_2d_savanna_targeted.py`

**Key numbers:**

| System | Dimension | Alpha | R^2 |
|--------|-----------|-------|-----|
| Lake (1D) | 1D | 0.373 | 0.997 |
| Toggle (2D, both dynamically active) | 2D | 0.503 | 0.9998 |
| Savanna (2D, one mechanism eq) | 2D | 0.844 | 0.984 |

Targeted perturbation: input variable (G) matters most, not equation containing nonlinearity.

**Confidence grade:** A

**Flags:** None. Three independent experiments with clear ordering.

---

### 11.5 Cusp bridge gamma ~ 0.20 at high d (d=38-80)

**Statement:** Extension of cusp bridge dimensional scaling to d=80. gamma = 0.197, resolving prior 0.14-0.41 ambiguity. d>50 deceleration resolved as statistical noise.

**Conclusion:** VERIFIED

**Scripts:** `bridge_high_d_scaling.py`, `bridge_high_d_extension.py`, `bridge_high_d_analysis.py`

**Key numbers:** Linear fit (d>=14, 8 points): gamma=0.197, R^2=0.9948. d for S=10^13: 163. B at d=38-50: grand mean 3.689, CV=1.7%. B at d=68-80: 3.43-3.45. All inside stability window. Quadratic and stretched-exp fits show acceleration, not deceleration.

**Confidence grade:** B

**Flags:**
1. Extrapolation from d=80 to d~150 is ~2x, not tested directly.
2. Single model class (Hill-function 1D ODEs).

---

## CATEGORY 16: Hopf Bridge

### 16.1 P(limit cycle | d) decays exponentially -- gamma_Hopf = 0.172

**Statement:** Random 2D multi-channel Hill-function ODEs show P(stable limit cycle) ~ exp(-gamma_Hopf * d) for d=24-42, with gamma_Hopf = 0.172 (R^2 = 0.9997).

**Conclusion:** VERIFIED

**Script:** `hopf_bridge_scaling.py`

**Input data:** Random 2D Hill-function ODEs with d = 6 + 6*n_channels. Sample sizes: 100K (d=12) to 5M (d=42). Seed 42.

**Key numbers:**

| d | P(limit cycle) | ln(1/P) |
|---|---------------|---------|
| 24 | 3.94e-4 | 7.839 |
| 30 | 1.43e-4 | 8.853 |
| 36 | 4.85e-5 | 9.934 |
| 42 | 1.82e-5 | 10.914 |

Linear fit (d>=24): gamma=0.172, R^2=0.9997. P peaks at d=18 before decaying (same pattern as cusp at d~14).

**Confidence grade:** B

**Flags:**
1. 4 decay-regime points (vs cusp bridge's 8). Sufficient to establish gamma but fewer.
2. Single model class (Hill-function ODEs). Same gamma could be model-class property.
3. Vectorized Newton finds ~74% of unstable spirals (calibrated, d-independent miss rate). Gamma unbiased, absolute P ~26% low.

---

### 16.2 gamma_Hopf ≠ gamma_cusp -- same form, different constant

**Statement:** gamma_Hopf = 0.172 vs gamma_cusp = 0.197 (ratio 0.87, 13% difference). The exponential form is universal but the decay constant is bifurcation-type-dependent.

**Conclusion:** VERIFIED

**Script:** `hopf_bridge_scaling.py`

**Key numbers:** Local gamma stabilizes at 0.17 for d=30-42. Does not converge to 0.197. The d=42 point (5M samples) confirms stabilization. This is the "alpha pattern" (Fact 71): exponential form universal, constant depends on system type.

**Confidence grade:** B

**Flags:**
1. 13% difference is real (confirmed by d=42), but could narrow or widen with more data points or different model classes.
2. Both bridges tested only on Hill-function ODEs. Model independence now tested (Study 18: polynomial ODEs show no decay).

---

## CATEGORY 17: Bautin Bridge

### 17.1 B_Bautin = 4.143 (CV 2.2%), inside stability window

**Statement:** B invariance extends to Bautin (subcritical Hopf) bistability between a fixed point and a limit cycle. 300 random (mu, l1) configurations in Bautin normal form give B = 4.143 (CV = 2.2%), with 100% of samples in stability window [1.8, 6.0]. K_Bautin = 0.301.

**Conclusion:** VERIFIED

**Script:** `bautin_bridge.py`

**Input data:** Bautin normal form r' = mu*r + l1*r^3 - r^5. Random (mu, l1) in bistable regime. Zero free parameters.

**Key numbers:** B = 4.143 (CV = 2.2%), K = 0.301, barrier varies 6,633x while B varies 1.1x.

**Confidence grade:** B (normal form B is analytical; multi-channel Part 3 did not converge — SDE escape fails for 2D FP-LC transitions)

**Flags:**
1. B_Bautin > B_cusp (4.14 vs 2.98) — different geometry shifts B center but invariance holds.
2. Part 3 (multi-channel B via SDE) incomplete — 1D barrier analysis inadequate for 2D FP-LC escape paths.

### 17.2 gamma_Bautin = 0.021 -- FP-LC configurations NOT exponentially rare

**Statement:** P(FP-LC bistable | d) is approximately flat across d = 12-42 for 2D Hill-function ODEs. gamma_Bautin = 0.021 (R^2 = 0.24). FP-LC configurations do NOT become exponentially rare with increasing dimension.

**Conclusion:** VERIFIED

**Script:** `bautin_bridge.py`

**Key numbers:** P ranges from 8.0e-4 to 1.7e-3 (factor ~3 variation across d=12-42). gamma_Bautin = 0.021 vs gamma_cusp = 0.197 vs gamma_Hopf = 0.172.

**Confidence grade:** B (detection methodology same as Study 16, but P is low — 50k-2M samples per d value)

**Flags:**
1. Physical interpretation: Bautin requires tr(J) < 0 at the FP, so degradation HELPS achieve the organized state — no fight between degradation and activation.
2. Consistent with the synthesis (Fact 82): exponential decay requires degradation OPPOSING the organized state.

---

## CATEGORY 18: Polynomial Bridge and Model-Class Tests

### 18.1 Polynomial bridge -- gamma_poly = 0.010, P flat at ~1.5%

**Statement:** P(stable limit cycle | d) for random polynomial 2D ODEs (no Hill functions, no saturation) is approximately constant at 1.3-2.1% across d = 12-56. gamma_poly = 0.010 (R^2 = 0.50). No exponential decay detected.

**Conclusion:** VERIFIED

**Script:** `poly_bridge_scaling.py`

**Input data:** Random polynomial coefficients scaled by degree. Cubic damping eps=0.01. Seed 42. 1.9M total samples, 27,261 limit cycles.

**Key numbers:** P: 1.86% (d=12), 2.08% (d=20), 1.32% (d=30), 1.52% (d=42), 1.31% (d=56). gamma = 0.010, R^2 = 0.50.

**Confidence grade:** B (excellent statistics — thousands of limit cycles at each d; single model class)

**Flags:**
1. Spiral fraction increases with d (23% to 42%), partially offsetting dimensional rarity — a polynomial-specific effect.
2. Polynomials are unbounded near equilibria, unlike Hill functions which saturate at [0,1].

### 18.2 Polynomial + degradation -- gamma = 0.007, degradation alone insufficient

**Statement:** Adding enforced degradation (b1,b2 ~ U[0.2,2.0], matching Study 16's range) to Study 18's polynomial model does NOT recover exponential decay. gamma_poly_degrad = 0.007 (R^2 = 0.964). Degradation suppresses P by a uniform ~40-50% at every d, but does not create d-dependent decay.

**Conclusion:** VERIFIED

**Script:** `poly_degradation_test.py`

**Input data:** Identical to Study 18 except two coefficients (x-equation's x-term, y-equation's y-term) forced to -U[0.2,2.0]. Seed 42. 7.8M total samples, 64,322 limit cycles.

**Key numbers:** P: 1.09% (d=12), 1.02% (d=20), 0.90% (d=30), 0.85% (d=42), 0.79% (d=56). Ratio to Study 18: 49-69% (uniform suppression). gamma = 0.007, R^2 = 0.964.

**Confidence grade:** B (excellent statistics at all d values; clean experimental design — one change from Study 18)

**Flags:**
1. The R^2 = 0.964 reflects a very slow drift (factor 1.4x over 44 dimensions), not true exponential decay. Compare Hill models: factor 10^4 over same range.
2. Spiral fraction drops from Study 18 levels (5% vs 23% at d=12) — degradation makes tr(J)>0 harder, explaining the uniform P suppression.

### 18.3 Search scaling requires bounded nonlinearities + degradation (synthesis)

**Statement:** The 2x2 matrix of {saturation, no saturation} x {degradation, no degradation} is complete. Exponential decay of P(organized|d) occurs ONLY when both bounded nonlinearities and degradation opposing activation are present. Neither alone is sufficient.

**Conclusion:** VERIFIED (Dead Hypotheses #18, #19; Fact 82)

**Evidence:**

| Model class | Architecture | gamma | Decays? | Study |
|-------------|-------------|-------|---------|-------|
| Hill | Degradation + saturation (cusp) | 0.197 | YES | 11 |
| Hill | Degradation + saturation (Hopf) | 0.172 | YES | 16 |
| Hill | Degradation helps (Bautin) | 0.021 | NO | 17 |
| Polynomial | No degradation, no saturation | 0.010 | NO | 18 |
| Polynomial | Degradation, no saturation | 0.007 | NO | 18 ext |

**Confidence grade:** B (each individual measurement is well-characterized; synthesis is the logical conjunction)

**Flags:**
1. The one untested cell (saturation without degradation) is partially covered by Bautin, where degradation helps rather than hinders — consistent with the synthesis.
2. Biological scope unchanged: all biological networks have both MM/Hill saturation and first-order degradation.

---

## CATEGORY 19: B Bounded Derivation

### 19.1 B bounded by cusp scale invariance -- stability window jointly explained

**Statement:** For the cusp normal form dx/dt = -x^3 + a*x + b, B = 2*DeltaPhi/sigma*^2 is a function of the shape parameter phi alone (scale 'a' cancels exactly). Since phi is bounded in (0, pi/3), B is bounded. At D=100: B in [2.609, 3.230]. Union across D=29 to 1111: B in [1.540, 5.659]. The empirical stability window [1.8, 6.0] is jointly determined by cusp geometry (69% from D variation, 20% from K variation, 11% from shape) and D selection (B < 2 too transient, B > 6 too stable).

**Conclusion:** VERIFIED (partially resolves highest-priority open question)

**Script:** `B_bounded_derivation.py`

**Input data:** Cusp normal form (Strogatz 1994). K_cusp = 0.558 from Study 11. D_target scanned at 29, 50, 100, 200, 500, 1111 (empirical ecological range). Zero external data, zero free parameters.

**Key numbers:**

| Quantity | Value |
|----------|-------|
| a-independence | CV = 0.00% across a = 0.5 to 10.0 (proved analytically + numerically) |
| B range at D=100 | [2.609, 3.230] (width 0.621, CV 6.0%) |
| Random instances (n=200) | B = 2.968 +/- 0.128, CV 4.3%, 100% in [1.8, 6.0] |
| Union across D=29-1111 | B in [1.540, 5.659] |
| Stability window decomposition | D variation 69%, K variation 20%, shape 11% |

**Proof structure:**
1. Scale invariance: DeltaPhi ~ a^2, sigma* ~ a, B = a^2/a^2 = f(phi)
2. Boundedness: phi in (0, pi/3) compact, B(phi) continuous -> finite bounds
3. Kramers: B = ln(D/K) - ln(2*pi*sqrt(R(phi))), R bounded -> B tightly constrained

**Confidence grade:** A (zero free params, analytic proof + numerical verification, cusp normal form is standard)

**Flags:**
1. Derived for cusp (saddle-node) only. Other bifurcation types may have different B bounds (cf. Bautin B=4.14 from Study 17 -- inside cusp range, consistent but not proved).
2. The D range [29, 1111] is empirical. Why these D values are observed (selection) is not derived.
3. Multi-dimensional systems may widen the B range via prefactor corrections (toggle B=4.83 at D~200 vs cusp B_max=3.9 at D=200 -- toggle exceeds 1D cusp prediction by ~0.9, attributable to 2D prefactor).

---

## CATEGORY 20: Search-Persistence Unification (Flight)

### 20.1 Search-persistence stability window test -- B in [1.8, 6.0] for integrated flight clades

**Statement:** Across 14 flight clades (4 vertebrate + 10 insect orders), D = MFPT_loss / tau_search was computed from phylogenetic data. High-integration flight clades show B_implied = ln(D) - beta_0 in the stability window: birds B=3.0, Lepidoptera B=2.7, Diptera B=2.7, bats B>4.3, Odonata B>3.1. Low-integration/modular clades fall below (Coleoptera B=0.2, Orthoptera B<0, Phasmatodea B<0). 3 precise measurements + 3 lower bounds in zone.

**Conclusion:** VERIFIED (stability window confirmed for integrated flight clades)

**Script:** `studies/20_fire_tree_flight/fire_tree_flight.py`

**Input data:** Roff 1990 Table 8 (order-level flightlessness %), Sayol et al. 2020 (bird r_loss = 11.70e-4/Myr), Condamine et al. 2016 (insect r_spec by order), Upham et al. 2019 (bat PD = 7549 lineage-Myr), Jetz et al. 2012 (bird r_spec = 0.16/Myr). Search-side: step12e (Fact 70).

**Key numbers:** Birds D=57 (B=3.04), Lepidoptera D=39 (B=2.66), Diptera D=39 (B=2.66), Bats D>210 (B>4.35), Odonata D>57 (B>3.05). All in stability window [1.8, 6.0].

**Confidence grade:** C (phylogenetic dimensional analysis; Roff f values are qualitative categories; no ODE model)

### 20.2 D = 1 threshold confirmed at biological level -- 5 flight-loss clades

**Statement:** Five clades have D < 1 for flight: Orthoptera (0.6), Hemiptera (0.7), Blattodea (0.3), Phasmatodea (0.3), Rails (0.6). Flight is lost faster than it took to evolve in these clades.

**Conclusion:** VERIFIED (5th independent confirmation level for D=1 threshold)

**Script:** `studies/20_fire_tree_flight/fire_tree_flight.py`

**Input data:** Roff 1990 Table 8, Kirchman 2012 (rails), Gaspar et al. 2020 (rails).

**Key numbers:** D_Orthoptera=0.59, D_Hemiptera=0.71, D_Blattodea=0.32, D_Phasmatodea=0.32, D_Rails=0.56.

**Confidence grade:** C (coarse phylogenetic estimates; f and r_spec approximate)

### 20.3 Modularity gradient -- D spans 3 OOM tracking flight integration

**Statement:** D = MFPT_loss / tau_search tracks flight-apparatus integration, not parametric complexity k. Gradient from D > 210 (bats: wing = hand) through D ~ 40-57 (birds, Lepidoptera, Diptera) to D ~ 0.3 (Phasmatodea: modular wings). This maps to the product equation: integrated systems have coupled channels, modular systems have decoupled channels.

**Conclusion:** VERIFIED (empirical gradient across 14 clades)

**Script:** `studies/20_fire_tree_flight/fire_tree_flight.py`

**Confidence grade:** D (qualitative integration classification; no quantitative coupling metric)

### 20.4 Cusp bridge scope boundary -- coupled vs decoupled channels

**Statement:** The cusp bridge (S and D from same geometric barrier) applies to noise-driven transitions in systems with coupled (integrated) regulatory channels. Selection-driven trait loss in modular (decoupled-channel) systems falls outside the bridge's scope. This boundary is anatomically identifiable for flight: wing = forelimb (high D) vs wing = separate appendage (low D).

**Conclusion:** PROPOSED (first identification of this scope boundary; not independently tested)

**Script:** `studies/20_fire_tree_flight/fire_tree_flight.py`

**Confidence grade:** D (interpretation of the modularity gradient; not independently confirmed)

---

## STALE FACTS AUDIT

Cross-check of all canonical facts against the provenance chain and source files. Originally conducted 2026-04-04 (48 facts); updated 2026-04-05 (80 facts).

### Previously flagged issues (ALL RESOLVED in FACTS.md as of 2026-04-05)

- **Fact 26 (HIV):** Now includes structural Omega*=2293 (0.1% match). RESOLVED.
- **Fact 36 (Cross-domain search):** Now split by equation type -- product eq (no Grade A outside ecology) AND Kramers (verified in cancer biology, climate, finance). RESOLVED.
- **Fact 41 (Soviet + power grid):** Now includes financial cusp K_eff=0.55. RESOLVED.
- **Fact 45 (B stability window):** Now includes 11 systems across 6 domains (peatland, tumor-immune, JJ, nanoparticle added). Cross-references Fact 77 (B bounded derivation). RESOLVED.

### Additional stale documents (not in FACTS.md)

**OPEN_QUESTIONS.md -- STALE-SUPERSEDED**
Central question "Why does sigma_process = sigma*?" is listed as open. NEXT_STEPS.md (2026-04-04) says this is RESOLVED by B invariance (B is structural invariant -> sigma* is determined by ODE -> if sigma_process = sigma*, it's because the environment's noise is set by the same physics as the barrier). Still legitimately open whether this is structural or selective (or both), but the framing as "the central theoretical question" should be updated.

**X2/NEXTSTEPS.md -- FULLY STALE**
From 2026-04-01. Lists Tests 1-5 as TODO. All 5 tests are COMPLETED (results in BT01, BT03-05, BT_BRIDGE_SYMBOLIC). This file should be marked HISTORICAL.

**CANONICAL_UPDATE_PLAN.md -- PARTIALLY STALE**
Missing S1-S6, domain research studies, full system grading table. The plan's structure is correct but incomplete. Should be executed with additions, then marked DONE.

---

## CATEGORY 21: Currency Peg Kramers (Study 21)

### 21.1 Currency peg Kramers -- calibrated Black Wednesday (GBP/DEM 1992)

**Statement:** The cusp catastrophe model dx/dt = -(x^3 - qx + a) + sigma*dW, calibrated to Black Wednesday observables (exchange rate gap 0.53 DEM, tau_relax = 25 days, sigma = 5% annualized implied vol), gives D = 28, B = 1.890, K = 0.597, inside the stability window. q is a gauge degree of freedom (0.0% CV across q = 1.5-4.0). 0 effective free parameters.

**Conclusion:** VERIFIED (calibrated, GREEN grade)

**Script:** `studies/21_currency_peg_kramers/currency_peg_kramers.py`

**Input data:** Bank of England historical spot rates (GBP/DEM 1990-1992), BIS Conference Paper (implied vol ~5%), ERM membership dates (Oct 1990 entry, Sep 1992 exit). All published.

**Key numbers:** D = 28, B = 1.890, K = 0.597, a/a_crit = 0.8175, sigma_predicted = 4.84% vs 5.0% observed (3% error). B invariance CV = 2.8% (20-point scan).

**Confidence grade:** A (0 effective free params, B in stability window, sigma match 3%)

**Flags:** q is nominally free but has zero physical effect (gauge). a determined by D_observed + sigma_observed.

---

### 21.2 Currency peg Kramers -- calibrated Thai Baht (THB/USD 1997)

**Statement:** Same cusp model calibrated to Thai Baht observables (exchange rate gap 15 THB, tau_relax = 180 days, sigma = 1.2% annualized realized vol) gives D = 26, B = 1.895, K = 0.609. B matches Black Wednesday (1.890) to 0.3% despite 4x different noise levels.

**Conclusion:** VERIFIED (calibrated, GREEN grade)

**Script:** `studies/21_currency_peg_kramers/currency_peg_kramers.py`

**Input data:** World Bank exchange rate data (THB/USD 1984-1997), NYU V-Lab GARCH minimum (1.52%), World Bank annual vol (0.93%). Informal peg ~1984-1997.

**Key numbers:** D = 26, B = 1.895, K = 0.609, a/a_crit = 0.9941. q-independence CV = 0.0%.

**Confidence grade:** A (0 effective free params, B in stability window, cross-crisis match)

**Flags:** Thai Baht requires a/a_crit = 0.994 (extremely close to fold) because noise is 4x lower than Black Wednesday. Sigma = 1.2% is from realized vol (no options-implied data available for THB pre-crisis).

---

### 21.3 Currency peg Kramers -- cross-crisis B consistency

**Statement:** Two independent currency pegs (GBP 1992, THB 1997) with different continents, decades, noise levels (5% vs 1.2%), devaluation sizes (18% vs 60%), and timescales (23 months vs 13 years) produce B = 1.890 and 1.895 (0.3% match) at similar D (28 vs 26). No parameters shared between computations. Both in stability window.

**Conclusion:** VERIFIED

**Script:** `studies/21_currency_peg_kramers/currency_peg_kramers.py` (Phase 6)

**Key numbers:** B_BW = 1.890, B_TB = 1.895, delta_B = 0.3%. All three computations (BW, TB full, TB initial D=339 B=4.47) in stability window [1.8, 6.0]. 3/3 in zone.

**Confidence grade:** A

**Flags:** B match is expected from B invariance (similar D → similar B). The non-trivial content is that the framework applies to organizational systems at all, and that independently measured observables produce B inside the stability window.

---

## GRADE DISTRIBUTION

| Grade | Count | Claims |
|-------|-------|--------|
| A | 37 | 1.3, 1.5, 2.1, 2.4, 2.5, 3.1, 3.2, 3.5, 4.1, 4.4, 5.3, 5.4, 6.2, 7.1, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 10.1, 10.2, 10.5, 11.1, 11.2, 11.3, 11.4, **19.1**, **21.1, 21.2, 21.3**, **22.1**, **24.1**, **27.1, 28.1, 30.1, 31.1** |
| B | 27 | 1.1, 1.2, 1.4, 1.6, 2.2, 2.3, 3.3, 3.4, 4.2, 4.5, 5.1, 5.2, 6.3, 7.3, 10.3, 11.5, 15.1, **16.1, 16.2, 17.1, 17.2, 18.1, 18.2, 18.3**, **26.1** |
| C | 8 | 4.3, 5.5, 6.1, 6.4, 7.2, 7.4, (1.6 partial), 15.2 |
| D | 3 | 3.6, 6.5, 10.4 |
| X | 3 | 6.6, 8.1, 8.2 |

---

## BROKEN CHAINS (claims requiring attention)

### Claims that cannot be traced to source:

1. **8.1: "10 constraints per channel"** -- This formula does not appear in any source file. The architecture scaling results give k*=30.4 total, not 10 per channel. Possible misinterpretation of k*/3 = 10 per M/T/B stage.

2. **8.2: S0 = D^(constraints/channels)** -- Formula not found in any source file.

### Claims where numbers don't match or methodology is weak:

4. **3.4: D~1 stellar** -- RESOLVED by Study 25. Kramers MFPT computation gives D = 2-6 for star-forming cores (M/M_BE = 0.8-1.2) at physical noise levels (Mach 1-2). Upgraded from WEAK CHAIN to VERIFIED.

5. **3.6 / 6.5: Rome D=1.16** -- Arithmetic is correct but D is computed from historical timescale ratios with no underlying ODE or Kramers model. Methodologically discontinuous with all other D values.

6. **6.6: Civilizational k values** -- Reverse-engineered from D and assumed epsilon. k=0.1 (non-integer) violates the discrete-channel model. S19 explicitly states N cannot be independently defined.

7. **7.3: Override resistance** -- R^2=0.95 at N=8, but anchor-only R^2=0.019 at N=7. The positive result is driven by estimated (not measured) D values.

8. **4.2: alpha uncertainty** -- Stated +/-0.02 is narrower than the 95% CI [0.333, 0.423] from the analysis itself.

### Claims where the chain is complete but with important caveats:

9. **1.1: Product equation** -- R01 showed 0/4 clean PASS initially. "6 verified" reflects corrections involving channel reinterpretation (k: 1->2) and free parameters. Lake epsilon_SAV = 0.05 has no identified primary source.

10. **1.4: Search equation** -- The equation tau_search = n*S0/(v*P) is tautological (S defined by the equation). Anti-circularity test (SV12D) partially addresses this for P. v-sensitivity analysis (Fact 74, 2026-04-05) showed v is a nuisance parameter: 3.9x tolerance, 73% of 66 ratio predictions within 1 OOM, v_best/v_estimated = 1.01. The 29 OOM initial failure was due to mixing individual-level and lineage-level P.

11. **5.1: HIV** -- The structural 5D MAP optimization did not converge. The 0.1% match is achieved by SDE-calibrated fitting, not first-principles barrier computation.

---

## CANONICAL FACTS NOT AUDITED

The following canonical facts (from FACTS.md) are not directly traced by any claim in this audit. They are listed here for completeness; each should be traceable via the source cited in FACTS.md.

| Fact | Summary | Source reference |
|------|---------|----------------|
| 2 | Round 1 failures were methodological | R01_REEXAMINATION |
| 3 | 1D Kramers unreliable without independent sigma | R02, R03 |
| 4 | Barrier dimension-independent, coordinate-dependent | R04 |
| 5 | Single-channel barrier ratio = 1.0 (QPot) | R05 |
| 6 | Two channels -> escape orthogonal to first coordinate | R06 |
| 7 | Endogenous formula subsumed by Kramers | R06 |
| 11 | eta = sigma_eff/sigma_obs is measurement-dependent | R11 |
| 13 | epsilon values are structural constants, not stochastic | BT_PATHC |
| 14 | Per-channel barrier decomposition fails (356% error) | BT01 |
| 15 | Growth exponent q creates the barrier | BT05 |
| 16 | Bridge NOT algebraic (transcendental obstruction) | BT_BRIDGE_SYMBOLIC |
| 17 | Hermite approximation for near-fold | BT_BRIDGE_SYMBOLIC |
| 18 | No universal structural ratio across q | BT_BRIDGE_SYMBOLIC |
| 19 | Bridge predicts CV=34.1% (lake: observed 35%) | BT04 |
| 20 | CV prediction logarithmically insensitive to epsilon uncertainty | XS2 (addendum) |
| 22 | Tokamak multiplicative stacking -- underpowered | XS4 |
| 23 | Cancer epistasis (Tian 2013 Nature) | XS1 |
| 24 | Savanna CV prediction logarithmically insensitive | SV02 |
| 29 | k=3 multiplicative confirmed (3.7x-133x discrimination) | SV08 |
| 31 | Formal criterion: 1D-reducible + separable | SV11 |
| 32 | Toggle fails even after adiabatic reduction | SV11 |
| 33 | 2x2 matrix of channel types | SV11 |
| 34 | 1/epsilon_i = exp(DeltaV_i) unification identity | SV11 |
| 36 | No Grade A cross-domain candidate (30 screened) | DR_CROSS_DOMAIN_SEARCH |
| 38 | S encodes innovation difficulty as signal | SV12C |
| 39 | Search + product equations give the full cycle | Steps 12 + Facts 1-34 |
| 41 | Soviet + power grid -- exploratory Kramers | KA_SOVIET, KA_POWER_GRID |
| 42 | sigma* matches physical noise for 5 systems | NOISE_SOURCE_MAPPING |
| 43 | Environmental forcing dominant; sigma_env derivable | NOISE_SOURCE_MAPPING |
| 46 | Kramers verified in physics (3 experiments, 0 free params) | Literature review |

### Study 22 claims (General B Boundedness Proof, 2026-04-06)

| # | Claim | Source |
|---|-------|--------|
| 22.1 | B boundedness is a universal property of Kramers escape theory. Scale invariance CV=0.00% (4 families). Washboard B width=0.034. Union B ranges match [1.8, 6.0]. | study22_general_B_bounded.py |
| 22.1a | Coral (Mumby 2007) confirms B universality: scale inv. CV=0.00%, shape B=6.07 +/- 2.15% (width 0.440), D-sweep [29-1111], cross-check 0.17% vs structural_B_coral.py. First ecological model test. | study22_coral_B_verification.py |
| 24.1 | B boundedness extends to 2D: scale invariance CV=1.70% (tumor-immune SDE, c=0.5-5.0, DeltaPhi 10x). Kramers-Langer prefactor: toggle beta_0=0.115, tumor-immune mid-range beta_0=0.910. Covers toggle (B=4.83), tumor-immune (B=2.73), diabetes (B=5.54). Resolves Study 22 Limitation #1. | study24_2D_B_bounded.py |

### Studies 26-31 claims (2026-04-06)

| # | Claim | Source |
|---|-------|--------|
| 26.1 | Sigma existence constraint — stability window constrains sigma_process/sigma* to band of width 1.826. 93.5% of prior eliminated. 4/5 ecological systems inside band. Grade: B (algebraically sound, partially circular). | study26_sigma_existence_constraint.py |
| 27.1 | S(d) model selection — power law rejected (dAICc = +12.63, 0.1% Akaike weight). Exponential preferred (50% weight). Bootstrap CI for d(S=10^13) = [150, 167]. Grade: A (formal statistical test). | model_selection_Sd.py |
| 28.1 | Xenopus product equation blind test — LOW state B = 3.45 (in window, k=1 effective), HIGH state B = 1.10 (below window, strong channels eps ~ 0.2-0.8). Product equation requires eps << 1 (weak perturbation regime), not just separability. Grade: A (0 free params, published model). | xenopus_product_eq_test.py |
| 30.1 | Data collapse — ln(D) - beta_0 = B for 13 systems across 7 domains, slope = 1, beta_0 = 1.06 +/- 0.37 (1D). 2D SDE sweeps confirm slope = 1 with different prefactors. Standard physics universality test. Grade: A. | data_collapse.py, sweep_2d_sde.py |
| 31.1 | Protein stability window — 22/25 proteins B > 6.0, B not invariant on chevron plots (Delta_B = 53-105% of window width). Scope boundary: structurally coupled noise required. Clean negative result, 0 free params. Grade: A. | study31_protein_stability_window.py |

### Study 32 claims (Crossing Theorem, 2026-04-06)

| # | Claim | Source |
|---|-------|--------|
| 32.1 | Route A negative: Q = sigma* × |lambda_eq| NOT constant (CV = 42-106% across 4 ecology systems). No algebraic identity for sigma* = sigma_process. Dead Hypothesis #23. Grade: A (exact MFPT, 4 systems, 30 pts each). | route_a_eigenvalue_test.py |
| 32.2 | Crossing Theorem: sigma*(a) → 0 and sigma_env(a) → ∞ at fold → IVT crossing guaranteed. All 4 ecology op. pts inside 1.5x bandwidth. B_cross ∈ [1.8, 6.0] for 6 systems (4 ecology + JJ + nanoparticle). Mean bandwidth = 35.3% of bistable range. sigma* = sigma_process resolved as topological + observational selection. Grade: A (exact MFPT, 6 systems, 50 pts each, independently measured SD(forcing) from Fact 43). | crossing_theorem_test.py |

### Study 33 claims (Noise Robustness, 2026-04-07)

| # | Claim | Source |
|---|-------|--------|
| 33.1 | B invariance robust to noise type. Multiplicative g(x)=sigma*sqrt(x): B_mult CV=3.07% (q=8, 35 pts, exact MFPT). Colored OU (tau_c=tau_relax): B_colored CV=1.95% (25 pts, Kramers correction). Ito-Stratonovich: both CV<5%. All B in stability window [1.8, 6.0]. Tested on lake model (van Nes & Scheffer 2005). Grade: A (exact MFPT, published params, 0 free params). Fact 96. | multiplicative_B_invariance.py, colored_noise_B_invariance.py, ito_stratonovich_correction.py |

### Study 34 claims (Kramers-Langer beta_0 Prediction, 2026-04-07)

| # | Claim | Source |
|---|-------|--------|
| 34.1 | Kramers-Langer beta_0 predicted from ODE Jacobians for all three 2D systems. Toggle: beta_0 = 2.253-2.368 (variation 0.115, matches Study 24 < 0.001). Tumor-immune: beta_0 = -1.018 to +0.885 (mid-range variation 0.910, matches Study 24 < 0.001). Diabetes (NEW): beta_0 = -1.213 to +1.234 (variation 2.447, 10 pts, all saddle points interior beta_saddle = 20-52). Toggle Figure 6 discrepancy: beta_0^KL = 2.356 vs circular 2.078, K = 0.757 (within CME-to-SDE range [0.34, 1.0]). Diabetes beta_0 at d0=0.06: 0.557 (first Kramers-Langer prediction for this system). Grade: A (0 free params, published ODEs: Gardner 2000, Kuznetsov 1994, Topp 2000). Fact 97. | studies/34_kramers_langer_beta0/study34_kramers_langer_beta0.py |

---

## METHODOLOGY NOTE

This audit was conducted on 2026-04-04 by reading source files directly. No existing file was edited. Claims were verified against actual numerical results, not against summaries in planning documents or canonical files. Where canonical summaries (FACTS.md, EQUATIONS.md) differ from source files, the source file was trusted. The main discrepancy found: Claim 1.4's initial SV12 results (29 OOM failure) are dramatically different from the corrected SV12B/C/D results (5.1 OOM, slope -0.95). The canonical summary (Fact 35) reflects the corrected results; the initial results are preserved in SV12.
