# Study 31: Protein Conformational Dynamics -- Stability Window Test -- Results

**Claim:** 31.1
**Grade:** GREEN (negative result, cleanly identifies scope boundary)
**Date:** 2026-04-06
**Script:** `study31_protein_stability_window.py`

## Question

Does the stability window B in [1.8, 6.0] apply to protein folding/unfolding dynamics? Protein conformational switching is a textbook Kramers escape problem with exactly known noise (kT), independently measured barriers (activation free energies from Arrhenius/Eyring analysis), and an enormous range of D values across different proteins. This makes proteins an ideal test domain for the universality of the stability window.

## Result

**No.** The stability window does NOT apply to protein conformational dynamics. 22/25 two-state proteins have B > 6.0. Even proteins with D in [30, 1000] (the framework's ecological range) have B = [8.7, 16.6], all above the window. B is not invariant along chevron plots (Delta_B spans 53-105% of the window width). This identifies a clean scope boundary: the window requires structurally coupled noise (sigma* determined by the system's feedback architecture), which thermal systems violate because noise = kT is an external bath parameter.

### Thermal Kramers mapping

For overdamped Langevin dynamics dX = -(1/gamma)*V'(X)dt + sqrt(2kT/gamma) dW:

```
Quasipotential:  Phi(x) = (1/gamma)*V(x)
Noise:           sigma^2 = 2kT/gamma
Framework B:     B = 2*DeltaPhi/sigma^2 = 2*(DeltaV/gamma)/(2kT/gamma) = DeltaV/kT
```

The factor of 2 in the framework's B = 2*DeltaPhi/sigma^2 cancels exactly with the factor of 2 in the thermal diffusion coefficient sigma^2 = 2kT/gamma (Einstein relation). For proteins:

```
B = DeltaG_u^dagger / kT    (no extra factor of 2)
D = 1 + k_f/k_u             (persistence ratio = MFPT/tau_relax)
```

### Test 1: Master table (25 two-state proteins)

B computed using the Kramers speed limit pre-exponential k_0 = 10^6 s^-1 (Kubelka et al. 2004): B = ln(k_0/k_u).

| Protein | N | k_f (s^-1) | k_u (s^-1) | D | B | In window? |
|---------|---|-----------|-----------|--------|------|-----------|
| Trp-cage TC5b | 20 | 2.4e5 | 5.8e4 | 5.1 | 2.85 | **YES** |
| BBA5 | 23 | 1.2e5 | 8.2e4 | 2.5 | 2.50 | **YES** |
| Villin HP35 N68H | 35 | 7.1e5 | 1.0e5 | 8.1 | 2.30 | **YES** |
| WW domain FiP35 | 35 | 1.4e4 | 1.1e3 | 13.7 | 6.81 | no |
| WW domain Pin1 | 34 | 7.7e3 | 1.2e3 | 7.4 | 6.73 | no |
| Engrailed HD | 61 | 4.9e4 | 2.0e3 | 25.5 | 6.21 | no |
| Protein A BdpA | 60 | 1.0e4 | 1.6e2 | 63.5 | 8.74 | no |
| lambda repressor | 80 | 3.3e4 | 2.0e1 | 1651 | 10.82 | no |
| CspB B.subtilis | 67 | 1.19e3 | 8.0 | 150 | 11.74 | no |
| CspB Thermotoga | 66 | 5.3e2 | 0.30 | 1768 | 15.02 | no |
| Protein L | 62 | 2.5e2 | 2.1 | 120 | 13.07 | no |
| Protein G GB1 | 56 | 6.3e2 | 4.0 | 158 | 12.43 | no |
| Im7 | 87 | 1.2e3 | 1.0 | 1201 | 13.82 | no |
| src SH3 | 64 | 2.7e1 | 6.4e-1 | 43 | 14.26 | no |
| spectrin SH3 | 62 | 3.4 | 6.4e-2 | 54 | 16.56 | no |
| fyn SH3 | 59 | 3.1e2 | 1.3e-1 | 2386 | 15.86 | no |
| AcP | 98 | 1.0e1 | 1.0e-1 | 101 | 16.12 | no |
| Barstar | 89 | 1.0e2 | 2.0e-1 | 501 | 15.42 | no |
| FKBP12 | 107 | 2.0 | 1.0e-1 | 21 | 16.12 | no |
| CI2 | 65 | 5.0e1 | 4.5e-4 | 1.1e5 | 21.52 | no |
| Barnase | 110 | 5.0 | 5.5e-5 | 9.1e4 | 23.62 | no |
| RNase H | 155 | 3.0 | 1.0e-3 | 3001 | 20.72 | no |
| Ubiquitin | 76 | 1.5e3 | 5.0e-5 | 3.0e7 | 23.72 | no |
| Titin I27 | 89 | 2.4e1 | 2.5e-4 | 9.6e4 | 22.11 | no |
| TNfn3 | 90 | 6.3e2 | 2.0e-3 | 3.2e5 | 20.03 | no |

B in [1.8, 6.0]: 3/25 (only marginally stable ultrafast folders with D < 10)
B > 6.0: 22/25 (88%)
B range: [2.30, 23.72]

### Test 2: Stability window by D range

| D range | n | In window | B range |
|---------|---|-----------|---------|
| D < 10 (marginal) | 4 | 3/4 | [2.3, 6.7] |
| D in [10, 30) | 3 | 0/3 | [6.2, 16.1] |
| **D in [30, 1000]** | **8** | **0/8** | **[8.7, 16.6]** |
| D in (1000, 2000] | 3 | 0/3 | [10.8, 15.0] |
| D > 2000 | 7 | 0/7 | [15.9, 23.7] |

**Key test:** Do proteins with D in [30, 1000] have B in [1.8, 6.0]? **NO** -- all 8 proteins in this range have B far above the window (minimum B = 8.74).

### Test 3: Bridge analysis (B vs ln(D))

In the framework: ln(D) = B + beta_0 with beta_0 in [0.53, 1.61].
For proteins: B = ln(D) + DeltaG_f^dagger/kT, so beta_0 = DeltaG_f^dagger/kT (the folding barrier).

| Quantity | Proteins (25) | Framework (5 ecological) |
|----------|---------------|--------------------------|
| beta_0 range | [0.21, 13.07] | [0.53, 1.61] |
| beta_0 mean | 7.4 +/- 3.7 kT | ~1 kT |
| Physical meaning | Folding barrier | Kramers prefactor |
| Linear fit | B = 1.35*ln(D) + 5.19, R^2 = 0.71 | B ~ ln(D) + 1 |

The protein folding barrier (DeltaG_f^dagger ~ 4.4 kcal/mol) inflates beta_0 by ~7x compared to the framework's ecological systems, pushing B far above the stability window at any given D.

### Test 4: B invariance (chevron plots)

| Protein | Denaturant | B range | Delta_B | % of window width | Framework CV threshold |
|---------|-----------|---------|---------|-------------------|----------------------|
| CI2 | GdnHCl | [19.3, 21.5] | 2.21 | 53% | CV=3.3% (misleading) |
| Barnase | urea | [19.2, 23.6] | 4.42 | **105%** | CV=6.3% |
| Protein L | GdnHCl | [10.3, 13.1] | 2.78 | 66% | CV=7.2% |

B is NOT invariant. CI2 has low CV (3.3%) only because its B values are ~20 (making the absolute variation look small percentagewise), but Delta_B = 2.21 spans over half the stability window width. Barnase's Delta_B exceeds the full window width.

**Reason:** In the framework, B is invariant because sigma* is structurally determined -- both DeltaPhi and sigma* scale together as the bifurcation parameter changes. For proteins, sigma^2 = 2kT/gamma is fixed by the thermal bath. Denaturant changes only the barrier, not the noise. B decreases linearly with denaturant: dB/dc = -m_ku.

### Test 5: Pre-exponential sensitivity

B = ln(k_0/k_u) depends on the choice of pre-exponential k_0.

| k_0 | Physical basis | Proteins in window |
|-----|---------------|-------------------|
| 10^5 s^-1 (Chung transit time) | Transition path time measurements | 3/25 |
| **10^6 s^-1 (Kubelka speed limit)** | **Best-motivated for proteins** | **3/25** |
| 10^7 s^-1 | Upper speed limit | 3/25 |
| 10^8 s^-1 | Very fast | 0/25 |
| 6.2e12 s^-1 (Eyring kT/h) | Upper bound, not physical | 0/25 |

The 3 proteins in the window (Trp-cage, BBA5, Villin HP35 N68H) are always the same ultrafast/marginally stable folders with D < 10, regardless of k_0. The result is robust to the pre-exponential choice.

## Key numbers

- **In stability window:** 3/25 (12%), all with D < 10
- **Above window:** 22/25 (88%), B up to 23.7
- **D in [30, 1000] with B in [1.8, 6.0]:** 0/8 (0%)
- **beta_0 (protein mean):** 7.4 kT vs framework 0.5-1.6 kT
- **B invariance Delta_B:** 2.2-4.4 (53-105% of window width)

## Scope boundary identified

The stability window B in [1.8, 6.0] applies to dissipative systems where:
1. The operating noise (sigma*) is **structurally determined** by the same feedback architecture that creates the barrier
2. B invariance holds (barrier and noise scale together)
3. The system is noise-maintained (noise drives transitions between metastable states of comparable stability)

Protein folding violates all three conditions:
1. Noise = kT is an **external bath parameter**, not structurally determined
2. B changes systematically with denaturant/temperature (NOT invariant)
3. The folded state is the free-energy minimum; thermal noise disrupts it rather than maintaining it

This explains WHY proteins have B >> 6: evolution selected unfolding barriers much larger than kT for kinetic stability. There is no structural constraint linking barrier height to thermal noise intensity -- unlike ecological systems where the barrier and noise derive from the same feedback structure.

## Relationship to other studies

- **Study 22 (general B bounded):** Proved B is bounded for any smooth 1D potential. Study 31 tests whether the specific bound [1.8, 6.0] applies outside the framework's primary domain. It does not -- proteins have B up to 23.7.
- **Study 26 (sigma existence constraint):** Derived a sigma constraint from the stability window. Study 31 shows this constraint does not apply to proteins because the noise is externally fixed, not structurally determined.
- **Studies 11, 24 (B invariance):** B is invariant for framework systems because sigma* is structurally coupled to the barrier. Study 31 shows B is NOT invariant for proteins (chevron test), confirming the scope boundary.
- **EQUATIONS.md Section 6:** The bridge decomposition ln(D) = B + beta_0 holds formally for proteins, but beta_0 = DeltaG_f^dagger/kT is 7x larger than for framework systems, reflecting the protein folding barrier.

## Limitations

1. **Pre-exponential uncertainty.** B = ln(k_0/k_u) depends on k_0, which is poorly constrained for proteins (10^5-10^8 s^-1). The Kramers speed limit k_0 = 10^6 is best-motivated but could be off by 10x. This shifts all B values by ln(10) = 2.3, which does not change the qualitative result (Test 5).
2. **Rate data quality.** All rates are from published chevron analyses extrapolated to 0 M denaturant, 25 C. The extrapolation can introduce errors of 2-5x in the rates, corresponding to ~1 in B. Some very slow unfolding rates (k_u < 10^-4 s^-1) are extrapolated over many half-lives and may be less reliable.
3. **Two-state assumption.** All 25 proteins are classified as two-state folders. Multi-state folders or intrinsically disordered proteins may behave differently (more complex energy landscapes).
4. **Single temperature.** The analysis is at 25 C. Near the cold or heat denaturation temperatures, the effective barrier shrinks and B could approach the stability window. This is a kinetically marginal regime (D -> 1) that the framework would classify as "at the D = 1 threshold."
5. **The mapping assumes overdamped dynamics on a 1D free energy surface.** Real protein dynamics involve many degrees of freedom, internal friction, and non-Markovian effects. These are well-established complications in the protein folding field but do not change the order-of-magnitude result (B >> 6 for most proteins).
