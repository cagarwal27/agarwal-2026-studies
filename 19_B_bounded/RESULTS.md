# Study 19: B Bounded Derivation -- Results

**Date:** 2026-04-05
**Script:** `B_bounded_derivation.py`
**Provenance claim:** 19.1
**Grade:** GREEN

---

## Question

Why is B = 2*DeltaPhi/sigma*^2 confined to [1.8, 6.0] across 11 systems spanning 6 domains? Is this a selection effect, a structural constraint, or both?

---

## Result

**Both.** B is bounded because of cusp scale invariance (structural), and the specific range [1.8, 6.0] is set by the observed range of D values (selection). Neither alone is sufficient.

---

## The proof (3 steps)

**Step A -- Scale invariance.** In the cusp normal form dx/dt = -x^3 + a*x + b, the three roots scale as sqrt(a). Therefore DeltaPhi ~ a^2, sigma* ~ a, and B = 2*DeltaPhi/sigma*^2 ~ a^2/a^2 = function of the shape parameter phi alone. Verified numerically: CV = 0.00% across a = 0.5 to 10.0 at 5 phi values.

**Step B -- Boundedness.** phi = (1/3)*arccos(3*sqrt(3)*b/(2*a^(3/2))) is bounded in (0, pi/3). B(phi) is continuous on this compact interval, therefore it has finite maximum and minimum.

**Step C -- Why the width is small (Kramers explanation).** In the Kramers regime: B = ln(D/K) - ln(2*pi*sqrt(|lam_eq|/lam_sad)). The eigenvalue ratio R(phi) = |lam_eq|/lam_sad ranges from 1 (at folds) to 2 (at symmetric point). So the prefactor correction ln(2*pi*sqrt(R)) varies by only 0.35. The exact MFPT gives a wider range (~0.6) because Kramers fails near the folds, but B is still tightly constrained.

---

## Key numbers

| Quantity | Value |
|----------|-------|
| B range at D=100, K=0.558 | [2.609, 3.230] (width 0.621, CV 6.0%) |
| Random cusp instances (n=200) | B = 2.968 +/- 0.128, CV 4.3%, 100% in [1.8, 6.0] |
| Union across D=29 to 1111 | B in [1.540, 5.659] |
| Empirical stability window | [1.8, 6.0] |

---

## Stability window decomposition

| Source | Contribution | Share |
|--------|-------------|-------|
| D variation (29 to 1111) | 3.6 | 69% |
| K variation (0.34 to 1.0) | 1.1 | 20% |
| Shape variation (phi) | 0.6 | 11% |
| **Total** | **5.3** | covers empirical width 4.2 |

---

## B at different D targets (exact MFPT)

| D_target | B_min | B_max |
|----------|-------|-------|
| 29 | 1.540 | 1.953 |
| 50 | 2.030 | 2.487 |
| 100 | 2.695 | 3.190 |
| 200 | 3.382 | 3.901 |
| 500 | 4.299 | 4.842 |
| 1111 | 5.101 | 5.659 |

---

## Why B < 2 and B > 6 are excluded

- B < 2 requires D < ~20 (MFPT < 20*tau). Systems transition too fast to be observed as persistent. **Selection.**
- B > 6 requires D > ~1500. Systems essentially never transition on observable timescales -- they appear static, not bistable. **Selection.**
- At any fixed D, the structural constraint (B is a bounded function of phi) prevents B from being arbitrarily large or small. **Structure.**

---

## Relationship to other studies

- **Study 02 (B invariance):** Proved B is constant across each system's bistable range. Study 19 explains WHY B is bounded across systems.
- **Study 11 (Cusp bridge):** Used the same cusp normal form to derive S(d). Study 19 uses it to derive B bounds. K_cusp = 0.558 comes from Study 11.
- **Studies 08, 17 (Blind tests, Bautin):** B values from these studies fall within the derived bounds.

---

## Limitations

1. Derived for the cusp (saddle-node) normal form only. Other bifurcation types (Hopf, SNIC) may have different B bounds.
2. The proof uses the 1D cusp potential. Multi-dimensional systems may introduce prefactor corrections that widen the B range (cf. toggle B = 4.83 at D ~ 200, above the cusp prediction of ~3.9).
3. The D range [29, 1111] is empirical (observed ecological systems). The structural result is that B is bounded at any fixed D; the selection argument for why D is in [29, 1111] is separate.
