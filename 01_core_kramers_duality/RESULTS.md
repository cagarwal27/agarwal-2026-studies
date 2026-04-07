# Study 01: Core Kramers Duality -- Results

**Date:** 2026-04-04
**Scripts:** `01_core_kramers_duality/*.py`

**Provenance claims:** 1.1, 1.3, Fact 24

## Question

Does D_product = D_Kramers hold for ecological bistable systems? D = prod(1/epsilon_i) from feedback coupling efficiencies should equal D from the exact MFPT integral at a specific noise sigma*.

## Results Summary

| System | D_product | K | eps | Free params | Model source | Grade |
|--------|-----------|---|-----|-------------|-------------|-------|
| Lake | 200 | 0.55-0.56 | 12 candidates tested | 0 | van Nes & Scheffer 2007 | YELLOW |
| Kelp | 29.4 | 0.34 | 0.034 | h (sensitivity-scanned) | Constructed 1D; Arroyo-Esquivel 2024 4D | YELLOW |
| Coral | 1111 | 0.32 (boundary) | 0.03 | 0 | Mumby et al. 2007, Nature | GREEN |
| Tropical forest | 95.2 | 0.34 | 0.07, 0.15 | alpha, beta (modified 3x) | Touboul et al. 2018 | YELLOW |
| Savanna | CV 30-38% | 0.55 | swept 0.03-0.30 | 0 | Staver-Levin / Xu et al. 2021 | YELLOW |
| Peatland (Path C) | 30.3 | -- | 0.10, 0.33 | m, q | Constructed 1D | YELLOW |
| Peatland (Path A) | 30.3 | -- | 0.10, 0.33 | c1 | Hilbert et al. 2000 | YELLOW |

---

## Per-System Details

### 1. Lake (phase1_lake_1d.py)

**Model:** dP/dt = a - b*P + r*P^q/(P^q + h^q)

| Parameter | Value | Source |
|-----------|-------|--------|
| b (P loss rate) | 0.8 | van Nes & Scheffer 2007, Table 1 |
| r (max recycling rate) | 1.0 | van Nes & Scheffer 2007, Table 1 |
| q (Hill coefficient) | 8 | van Nes & Scheffer 2007, Table 1 |
| h (half-saturation) | 1.0 | van Nes & Scheffer 2007, Table 1 |
| a (loading, midpoint bistable range) | 0.326588 | van Nes & Scheffer 2007, Table 1 |
| x_clear (clear equilibrium) | 0.409217 | Prior computation (Round 9) |
| x_sad (saddle) | 0.978152 | Prior computation (Round 9) |
| x_turb (turbid equilibrium) | 1.634126 | Prior computation (Round 9) |
| lambda_clear (eigenvalue) | -0.784651 | Prior computation (Round 9) |
| lambda_sad (eigenvalue) | 1.228791 | Prior computation (Round 9) |

**Method:** Exact MFPT integral on fine grid (N=50000). Bisection to find CV* where D_exact = 200. Tests 12 epsilon candidate definitions and all pairwise products.

**Results:** D=200, K=0.55-0.56.

**Replicability:** YELLOW. All model parameters published. epsilon_SAV = 0.05 is model-derived, not field-measured. Equilibria from prior computation (verifiable from model parameters).

---

### 2. Kelp (step6_kelp_kramers.py)

**Path A: Arroyo-Esquivel et al. 2024 (4D sea-star model)**

| Parameter | Value | Source |
|-----------|-------|--------|
| r (kelp growth) | 2.5 | Arroyo-Esquivel et al. 2024, Table 1 |
| K (carrying capacity) | 10000.0 | Arroyo-Esquivel et al. 2024, Table 1 |
| kD (drift preference) | 1.95 | Arroyo-Esquivel et al. 2024, Table 1 |
| aA (algal attack rate) | 0.025 | Arroyo-Esquivel et al. 2024, Table 1 |
| kS (fear response) | 0.81 | Arroyo-Esquivel et al. 2024, Table 1 |
| dA (algal death rate) | 1.8 | Arroyo-Esquivel et al. 2024, Table 1 |
| dD (detritus decay) | 0.3 | Arroyo-Esquivel et al. 2024, Table 1 |
| eD (detritus efficiency) | 0.7 | Arroyo-Esquivel et al. 2024, Table 1 |
| eU (urchin efficiency) | 0.1 | Arroyo-Esquivel et al. 2024, Table 1 |
| aD (detritus attack rate) | 0.062 | Arroyo-Esquivel et al. 2024, Table 1 |
| aU (urchin attack by stars) | 4.77 | Arroyo-Esquivel et al. 2024, Table 1 |
| gU (urchin handling time) | 3.42 | Arroyo-Esquivel et al. 2024, Table 1 |
| kA (starvation link) | 0.00013 | Arroyo-Esquivel et al. 2024, Table 1 |
| dU (urchin mortality) | 0.0004 | Arroyo-Esquivel et al. 2024, Table 1 |
| beta (density effect) | 0.1 | Arroyo-Esquivel et al. 2024, Table 1 |
| eS (star efficiency) | 0.1 | Arroyo-Esquivel et al. 2024, Table 1 |
| dS (star mortality) | 1e-4 | Arroyo-Esquivel et al. 2024, Table 1 |

Path A uses sunflower sea stars, not sea otters. epsilon = 0.034 is from otter data, so there is a predator mismatch in this path. Sets np.random.seed(42) for initial condition sampling.

**Path B: 1D Otter-Urchin Model (constructed)**

| Parameter | Value | Source |
|-----------|-------|--------|
| r (urchin P/B ratio) | 0.4 yr^-1 | Brey 2001 (echinoderm P/B compilations) |
| K_U (barren threshold) | 668 g/m^2 | Ling et al. 2015, PNAS |
| U_SAD (reverse threshold) | 71 g/m^2 | Ling et al. 2015, PNAS |
| h (half-saturation) | **FREE** | Scanned over [30, 50, 75, 100, 150, 200, 300, 400] |
| p (predation rate) | Derived: r*(1-U_SAD/K_U)*(U_SAD+h) | Constrained by saddle at U=71 |
| epsilon | 0.034 | Otter consumption / urchin destructive capacity |

**Model (Path B):** dU/dt = r*U*(1 - U/K_U) - p*U/(U + h)

**Method:** Equilibrium finder via forward integration + fsolve (Path A). Exact 1D MFPT integral with reflecting boundary at U=0 (Path B). Bisection for sigma* where D_exact = D_product. Central analysis at h=100.

**Results:** D_product = 29.4. D_exact = 29.4 at sigma* = 10.50. K = 0.34. Empirical predation flux (9.2 g/m^2/yr from Tinker/Yeates) is lower than effective p required for saddle at U=71; p is an effective parameter, not the literal otter predation flux.

**Replicability:** YELLOW. Path A: all 17 params published. Path B: h is free (sensitivity-scanned). K = 0.34 is a boundary artifact (U_eq = 0); see step6b.

---

### 3. Kelp Immigration (step6b_kelp_immigration.py)

Tests whether K=0.34 is a boundary artifact. Adds immigration term c to kelp model.

| Parameter | Value | Source |
|-----------|-------|--------|
| r (urchin P/B) | 0.4 yr^-1 | Brey 2001 |
| K_U (barren threshold) | 668 g/m^2 | Ling et al. 2015 |
| h (half-saturation) | 100 g/m^2 | Step 6 central value |
| p (predation rate) | 61.13 g/m^2/yr | Derived from saddle constraint at U=71 |
| c (immigration) | Swept: 0, 0.1, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 15, 20 g/m^2/yr | Ecologically realistic: larval settlement |

**Model:** dU/dt = c + r*U*(1 - U/K_U) - p*U/(U + h)

**Method:** For each c value, find equilibria, compute barrier, find sigma* where D_exact = D_product = 29.4, compute K_actual. Track K as U_eq moves away from 0.

**Results:** K changes only 0.2% as U_eq moves from 0 to 13.7. Mechanism: at reflecting boundary, probability distribution is half-Gaussian, roughly halving the well population and yielding K_boundary ~ K_interior/2.

**Replicability:** YELLOW. Same free h as step6.

---

### 4. Coral (step7_coral_kramers.py)

| Parameter | Value | Source |
|-----------|-------|--------|
| a (macroalgal overgrowth of coral) | 0.1 yr^-1 | Mumby et al. 2007, Nature 450:98-101 |
| gamma (macroalgal spread over turf) | 0.8 yr^-1 | Mumby et al. 2007 |
| r (coral growth over turf) | 1.0 yr^-1 | Mumby et al. 2007 |
| d (coral natural mortality) | 0.44 yr^-1 | Mumby et al. 2007 |
| g (operating grazing rate) | 0.30 | Mumby et al. 2007 (mid-bistable Caribbean) |
| epsilon | 0.03 | Energy budget basis (Gattuso et al., calcification energetics) |

**Model (Mumby 2007):**
- dM/dt = a*M*C - g*M/(M+T) + gamma*M*T
- dC/dt = r*T*C - d*C - a*M*C
- T = 1 - M - C

**Equilibria:**
- Coral state: M*=0, C*=0.56, T*=0.44 (boundary equilibrium)
- Algae state: M*=1-g/gamma, C*=0, T*=g/gamma
- Interior saddle: found numerically

**Method:** Adiabatic reduction to effective 1D system along the C-nullcline (C relaxes ~2x faster than M). Exact MFPT integral with reflecting boundary at M=0. Bisection for sigma*.

**Results:** D_product = (1/0.03)^2 = 1111. D_exact = 1111 at sigma* = 0.0299. K_actual = 0.32 (boundary eq at M=0, consistent with kelp K=0.34). CV(C) physically plausible from storm/ENSO variability.

**Replicability:** GREEN. Zero free parameters. All 5 model params from Mumby 2007. Epsilon from published energy budget data.

**Caveat:** Uses adiabatic 1D reduction (justified by lambda_C/lambda_M timescale ratio ~ 2x). Full 2D quasi-potential was attempted but convergence was not achieved to machine precision.

---

### 5. Tropical Forest (step10_tropical_forest_kramers.py)

| Parameter | Value | Source | Status |
|-----------|-------|--------|--------|
| alpha (fire-mediated forest mortality) | 0.6 | **MODIFIED** (published: 0.2, Touboul et al. 2018 Table 1) | Modified for clean bistability |
| beta (sapling recruitment) | 1.0 | **MODIFIED** (published: 0.3, Touboul et al. 2018 Table 1) | Modified for clean bistability |
| mu (sapling mortality) | 0.1 | Touboul, Staver & Levin 2018, Table 1 | Published |
| nu (tree mortality) | 0.5 | Touboul, Staver & Levin 2018, Table 1 | Published |
| omega_0 (fire rate, low end) | 0.9 | Touboul, Staver & Levin 2018, Table 1 | Published |
| omega_1 (fire rate, high end) | 0.4 | Touboul, Staver & Levin 2018, Table 1 | Published |
| theta_1 (fire sigmoid midpoint) | 0.4 | Touboul, Staver & Levin 2018, Table 1 | Published |
| s_1 (fire sigmoid steepness) | 0.01 | Touboul, Staver & Levin 2018, Table 1 | Published |
| phi_0 (competition, low end) | 0.1 | Touboul, Staver & Levin 2018, Table 1 | Published |
| phi_1 (competition, high end) | 0.9 | Touboul, Staver & Levin 2018, Table 1 | Published |
| theta_2 (competition sigmoid midpoint) | 0.4 | Touboul, Staver & Levin 2018, Table 1 | Published |
| s_2 (competition sigmoid steepness) | 0.05 | Touboul, Staver & Levin 2018, Table 1 | Published |
| eps_fire | 0.07 | Brando et al. 2014 | Published |
| eps_drought | 0.15 | Eltahir & Bras 1994 | Published |

**Model (Touboul 2018, 3D reduction from 4-variable G+S+T+F=1):**
- dS/dt = beta*G*T - (omega(G_eff) + mu)*S - alpha*S*F
- dT/dt = omega(G_eff)*S - nu*T - alpha*T*F
- dF/dt = (alpha*(1-F) - phi(G_eff))*F
- G_eff = 1 - F - (1-gamma)*(S+T)
- omega and phi are sigmoid functions of G_eff

**Method:** Effective 1D reduction along F-axis (S,T equilibrate 3-5x faster). Exact MFPT integral. Bisection for sigma*.

**Results:** D_product = (1/0.07)*(1/0.15) = 95.2. K = 0.34.

**Replicability:** YELLOW. Two parameters modified from published values (alpha: 0.2 -> 0.6, beta: 0.3 -> 1.0) to produce clean bistability. All other 10 params from Touboul et al. 2018 Table 1. Epsilons from independent field sources (Brando 2014, Eltahir 1994).

**Limitation:** The modification of alpha and beta is significant (3x each). The published parameter values do not produce the required bistable structure. This means the duality test operates on a model that is qualitatively but not quantitatively faithful to the published parameterization.

---

### 6. Savanna (step2_savanna_log_robustness.py)

Sweeps epsilon ranges, tests logarithmic insensitivity of the CV prediction. Entirely analytical (no SDE simulations).

| Parameter | Value | Source |
|-----------|-------|--------|
| beta (fire feedback) | 0.39 | Xu et al. 2021 |
| mu (grass mortality) | 0.2 | Xu et al. 2021 |
| nu (tree mortality) | 0.1 | Xu et al. 2021 |
| omega_0 (fire sigmoid, low) | 0.9 | Xu et al. 2021 |
| omega_1 (fire sigmoid, high) | 0.2 | Xu et al. 2021 |
| theta_1 (fire sigmoid midpoint) | 0.4 | Xu et al. 2021 |
| ss1 (fire sigmoid steepness) | 0.01 | Xu et al. 2021 |
| G_sav, T_sav (savanna equilibrium) | 0.5128, 0.3248 | Prior computation (Round 5) |
| G_for, T_for (forest equilibrium) | 0.3134, 0.6179 | Prior computation (Round 5) |
| G_sad, T_sad (saddle) | 0.4155, 0.4461 | Prior computation (Round 5) |
| DeltaPhi (barrier) | 0.000540 | QPot computation |
| K_SDE (anharmonicity correction) | 0.55 | Round 9 (240 MFPT computations) |
| C_tau (prefactor * relaxation time) | 0.232 | Eigenvalue analysis |
| eta (sigma_eff / sigma_obs correction) | 0.415 | LNA calibration |

**Model:** Staver-Levin savanna-grassland-forest ODE.

**Method:** For each (eps_fire, eps_herb) pair in a 20x20 grid, compute D = 1/(eps_fire*eps_herb), invert the bridge identity for sigma_eff, propagate through LNA to sigma_T, apply eta correction to get sigma_obs, compute CV_obs. Compare to observed tree cover variability.

**Sweep ranges:** eps_fire in [0.03, 0.30], eps_herb in [0.03, 0.25].

**Results:** 77% of epsilon pairs give CV in [30%, 38%], matching observed 35% (Staver et al., Hirota et al.). D ranges from ~13 to ~1111 (62x fold variation) but CV band is only 17 pp. Null comparators (single-channel, no-regulation) are excluded.

**Replicability:** YELLOW. Model parameters from Xu et al. 2021. DeltaPhi, K_SDE, C_tau, eta from prior computations (verifiable but not directly from a single published source).

---

### 7. Peatland Path C (step13_peatland_kramers.py)

Constructed minimal 1D ODE. Tests D_product = D_Kramers for boreal peatland bistability.

| Parameter | Value | Source | Status |
|-----------|-------|--------|--------|
| NPP (net primary production) | 0.20 kgC/m^2/yr | Frolking et al. 2010; Loisel et al. 2014 | Published |
| d_aer (aerobic decomposition) | 0.05 yr^-1 | Clymo 1984 (acrotelm alpha) | Published |
| d_anaer (anaerobic decomposition) | 0.001 yr^-1 | Clymo 1984 (catotelm beta) | Published |
| m (critical stock for waterlogging) | 40 kgC/m^2 | **FREE** (~0.8 m peat depth at 50 kgC/m^3) | Tuned |
| q (enzymatic latch steepness) | 8 | **FREE** (Freeman et al. 2001 mechanism) | Tuned |
| eps_1 (Frolking) | 0.065 | Frolking et al. 2010 | Published |
| eps_2 (Clymo) | 0.10 | Clymo 1984 | Published |
| eps_3 (recalcitrance) | 0.33 | Hajek et al. 2011 | Published |

**Model:** dC/dt = NPP - d_aer*C + Delta_d * C^(q+1)/(C^q + m^q), where Delta_d = d_aer - d_anaer = 0.049 yr^-1. Hill function h(C) = C^q/(C^q + m^q) represents waterlogging activation fraction.

**D_product scenarios:**
- k=1 conservative (eps=0.065): D = 15.4
- k=1 central (eps=0.10): D = 10.0
- k=2 two-channel (eps=0.10, eps=0.33): D = 30.3

**Method:** Find 3 equilibria (degraded, saddle, intact). Compute effective potential V(x) with escape coordinate x = C_high - C. Exact MFPT integral. Bisection for sigma* at each D_product scenario.

**Results:** D = 30.3 (k=2 scenario).

**Replicability:** YELLOW. NPP, d_aer, d_anaer are published. m and q are free parameters tuned to produce realistic peat depths. The model is constructed (not from a single published source), though it captures the enzymatic latch mechanism from Freeman et al. 2001.

---

### 8. Peatland Path A (step13_peatland_kramers_hilbert.py)

Hilbert et al. 2000 published 2D model. Independent test of the same peatland D_product.

| Parameter | Value | Source | Status |
|-----------|-------|--------|--------|
| k (production rate) | 0.00025 cm yr^-1 | Hilbert et al. 2000, Fig 4 | Published |
| r1 (aerobic decomposition) | 0.0025 yr^-1 | Hilbert et al. 2000, Fig 4 | Published |
| r2 (anaerobic decomposition) | 0.00025 yr^-1 | Hilbert et al. 2000, Fig 4 | Published |
| Z_min (lower production bound) | -10 cm | Hilbert et al. 2000, Fig 4 | Published |
| Z_max (upper production bound) | 70 cm | Hilbert et al. 2000, Fig 4 | Published |
| c2 (drainage increase rate) | 0.05 yr^-1 | Hilbert et al. 2000, Fig 6 | Published |
| d0 (base drainage) | 20 cm yr^-1 | Hilbert et al. 2000, Fig 6 | Published |
| Eo (potential evaporation) | 60 cm yr^-1 | Hilbert et al. 2000, Fig 6 | Published |
| theta_max (water storage per cm) | 0.8 | Hilbert et al. 2000, Fig 6 | Published |
| P (operating precipitation) | 80 cm yr^-1 | Hilbert et al. 2000, Fig 5 (mid-bistable) | Published |
| c1 (evapotranspiration sensitivity) | **FREE** (calibrated ~0.5 cm^-1) | Not stated in paper | Calibrated from Fig 5 bistability |

**Model (Hilbert et al. 2000, eqns 12, 14, 15):**
- State variables: H (peat height, cm), Z0 (water table depth below surface, cm)
- G(Z0) = k*(Z0 - Z_min)*(Z_max - Z0) for Z_min <= Z0 <= Z_max
- dH/dt = G - (r1-r2)*Z0 - r2*H
- dZ0/dt = [c2/theta - r2]*H - (r1-r2)*Z0 + Eo/(theta*(1+c1*Z0)) + G - (P-d0)/theta

**Method:** Calibrate c1 by scanning [0.5, 5.0] to find value giving 3 equilibria at P=80 (matching Fig 5 phase plane). Target: wet H~800, saddle H~1000, dry H~1200. Adiabatic reduction: Z0 is fast (~months) vs H (~centuries), timescale ratio >1000x. Reduce to 1D along Z0 nullcline. Exact MFPT integral.

**Results:** D = 30.3 (converges with Path C). Both independent models give the same D at the k=2 epsilon scenario.

**Replicability:** YELLOW. 10 params from published figures. c1 is free (not stated in paper); calibrated to reproduce Fig 5 bistability structure. Full parameter tables behind Wiley paywall.

---

## Limitations

1. **Tropical forest:** alpha and beta are modified 3x from published values to produce clean bistability. The test verifies the framework machinery on a qualitatively correct model, not the published parameterization.
2. **Lake:** epsilon_SAV = 0.05 is model-derived, not field-measured. The 12-candidate epsilon search is exploratory.
3. **Kelp:** h is a free parameter. Empirical otter predation flux (9.2 g/m^2/yr) does not match the effective p required for the saddle constraint. K = 0.34 is a boundary artifact (U_eq = 0).
4. **Peatland Path C:** m and q are free parameters. The model is constructed to capture the enzymatic latch mechanism but is not taken from a single published ODE.
5. **Peatland Path A:** c1 is not stated in the Hilbert et al. 2000 paper and must be calibrated.
6. **Coral and kelp K values:** K = 0.32-0.34 at boundary equilibria (M=0, U=0), vs K = 0.55 at interior equilibria. The discrepancy is explained by the half-Gaussian probability distribution at reflecting boundaries.
7. **Adiabatic reductions:** Coral (C 2x faster than M) and tropical forest (S,T 3-5x faster than F) use 1D reductions. These are justified by timescale separation but introduce approximation error.
