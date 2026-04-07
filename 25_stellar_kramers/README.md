# Study 25: Kramers MFPT for Stellar D ~ 1

**Date:** 2026-04-06
**Script:** `../scripts/study25_stellar_kramers.py`

## Purpose

Upgrade stellar D ~ 1 from dimensional argument (WEAK CHAIN) to direct Kramers MFPT computation (VERIFIED). Compute D = MFPT/tau_relax for star-forming cloud cores near the Bonnor-Ebert critical mass using the exact Gardiner MFPT formula on a 1D potential derived from first principles.

## Data Provenance

All inputs from published astrophysics (0 free parameters):

| Parameter | Symbol | Value | Source | Where in source |
|-----------|--------|-------|--------|-----------------|
| Core mass | M | 0.66-1.0 M_sun | Andre+ 2014 | Table 1, Aquila CMF peak |
| Temperature | T | 10 K | Bergin & Tafalla 2007 | Section 2.1, "dense cores at 8-12 K" |
| Mean mol. weight | mu | 2.33 | Standard | H2 (86% by number) + He (14%) |
| External pressure | P_ext/k | 2 x 10^5 K cm^-3 | McKee & Ostriker 2007 | Table 1, midplane pressure |
| Mach number | M | 1.0-2.0 | Pineda+ 2010 | Fig. 1, "coherent cores" at trans-sonic |
| Core lifetimes (comparison) | t_core/t_ff | 1-3 | Enoch+ 2008 | Section 5.3 |
| BE critical mass formula | M_BE | 1.182 c_s^4/(G^1.5 sqrt(P)) | Bonnor 1956 | eq. 17; also Ebert 1955 |
| Gravitational self-energy | E_grav | -(3/5)GM^2/R | Kippenhahn+ 2012 | eq. 1.4 |
| Physical constants | G, k_B, m_H | CODATA 2018 | NIST | -- |

The sigma = Mach mapping (see RESULTS.md Section 5) involves an O(1) geometric factor spanned by the Mach 1-2 sweep. No parameter is fitted to the output.

## Replication

### Dependencies
```
Python 3.8+
pip install numpy scipy
```
No other dependencies.

### Running
```bash
cd ../scripts/
python3 study25_stellar_kramers.py
```

The script prints all intermediate quantities (gamma, equilibria, barrier, D(sigma) tables). To verify a specific number:
- gamma: Section 2.4 of RESULTS.md + `compute_gamma()` in script
- Equilibria: solve Eq. 2 numerically or check `find_equilibria()` in script
- DeltaPhi: evaluate Eq. 1 at x_sad and x_eq
- D: compare script output against Kramers approximation D_Kr = K exp(B)/(C tau) as a sanity check

Expected runtime: < 1 minute (numerical integration, no SDE).

## Files

| File | Description |
|------|-------------|
| `README.md` | This file (replication instructions) |
| `RESULTS.md` | Full derivation, results, interpretation, and sources |

The script is also available at `../scripts/study25_stellar_kramers.py`.

## References

| Short cite | Full reference | Used for |
|------------|---------------|----------|
| Bonnor 1956 | Bonnor W.B. (1956), MNRAS 116, 351 | Critical isothermal sphere, M_BE formula |
| Ebert 1955 | Ebert R. (1955), Z. Astrophysik 37, 217 | Same as Bonnor, independent derivation |
| McKee & Ostriker 2007 | McKee C.F. & Ostriker E.C. (2007), ARA&A 45, 565 | ISM pressure P_ext/k = 2e5 K/cm^3 |
| Bergin & Tafalla 2007 | Bergin E.A. & Tafalla M. (2007), ARA&A 45, 339 | T = 10 K for dense cores |
| Andre+ 2014 | Andre P. et al. (2014), Protostars & Planets VI, 27 | Core masses, CMF |
| Pineda+ 2010 | Pineda J.E. et al. (2010), ApJL 712, L116 | Trans-sonic turbulence in cores |
| Enoch+ 2008 | Enoch M.L. et al. (2008), ApJ 684, 1240 | Core lifetimes 1-3 t_ff |
| Kirk+ 2005 | Kirk H. et al. (2005), MNRAS 360, 1506 | Core lifetimes |
| Bertoldi & McKee 1992 | Bertoldi F. & McKee C.F. (1992), ApJ 395, 140 | Virial parameter alpha_vir |
| Konyves+ 2015 | Konyves V. et al. (2015), A&A 584, A91 | Core mass function |
| Alves+ 2007 | Alves J., Lombardi M. & Lada C.J. (2007), A&A 462, L17 | CMF-IMF similarity |
| KM05 | Krumholz M.R. & McKee C.F. (2005), ApJ 630, 250 | epsilon_ff, context |
| Kippenhahn+ 2012 | Kippenhahn R., Weigert A. & Weiss A. (2012), Stellar Structure & Evolution | E_grav = -(3/5)GM^2/R |
| Gardiner 2009 | Gardiner C.W. (2009), Handbook of Stochastic Methods, 4th ed. | MFPT formula, eq. 5.2.160 |
