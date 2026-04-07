# Study 08: Blind Tests (New Domains)

**Date:** 2026-04-06
**Scripts:** `blind_test_josephson_junction.py`, `blind_test_magnetic_nanoparticle.py`

## Question

Do B invariance and the stability window [1.8, 6.0] hold in physics domains entirely outside the framework's training set, with zero free parameters?

## Data

Two normalized ODEs with analytic structure, tested via exact MFPT integrals on N=200,000 grids:
- Josephson junction: tilted washboard potential, gamma swept 0.10-0.99
- Magnetic nanoparticle: uniaxial anisotropy + field, h swept 0.00-0.97

## Key Results

| System | Domain | K | B CV | B range | Free params | Grade |
|--------|--------|------|------|---------|-------------|-------|
| Josephson junction | Superconducting electronics | 0.56 | 0.3-0.4% | [1.8, 6.0] (all gamma) | 0 | GREEN |
| Magnetic nanoparticle | Nanomagnetism | 0.57 | 1.5-2.7% | [1.8, 6.0] | 0 | GREEN |

## Parameters and Sources

### Josephson Junction (blind_test_josephson_junction.py)

| Parameter | Value | Source |
|-----------|-------|--------|
| Potential | V(phi) = -cos(phi) - gamma*phi | Stewart 1968, McCumber 1968 |
| Drift | dphi/dt = gamma - sin(phi) | Normalized RCSJ equation |
| gamma (bias current) | Swept: 0.10-0.99 | Bifurcation parameter (bistable for gamma in (0,1)) |
| sigma^2 | 2kT/E_J | Thermal noise (E_J = Josephson energy) |
| Free parameters | 0 | Normalized equation; all structure analytic |

Analytic structure (verified numerically):
- Equilibria: phi_min = arcsin(gamma), phi_sad = pi - arcsin(gamma)
- Barrier: DeltaV = 2*sqrt(1-gamma^2) - 2*gamma*arccos(gamma)
- Eigenvalues: lambda_eq = sqrt(1-gamma^2), lambda_sad = -sqrt(1-gamma^2)
- Curvature ratio = 1.0 for ALL gamma
- 1/(C*tau) = 2*pi for ALL gamma

Experimental references: Devoret et al. PRL 55, 1908 (1985); Martinis et al. PRB 35, 4682 (1987).

Physical context: E_J ~ 3e-21 J (Ic ~ 10 uA). At T = 1 K: B = E_J*DeltaV/kT ~ 220*DeltaV. Experiments observe thermal activation at B ~ 5-20, overlapping the stability window at high bias.

### Magnetic Nanoparticle (blind_test_magnetic_nanoparticle.py)

| Parameter | Value | Source |
|-----------|-------|--------|
| Potential | V(theta) = sin^2(theta) - 2h*cos(theta) | Stoner & Wohlfarth 1948 |
| Drift | dtheta/dt = -2*sin(theta)*(cos(theta) + h) | Neel-Brown overdamped dynamics |
| h (normalized field) | Swept: 0.00-0.97 | Bifurcation parameter (bistable for h in (0,1)) |
| sigma^2 | 2kT/(K_a*Vol) | Thermal noise |
| Free parameters | 0 | Normalized equation; all structure analytic |

Analytic structure:
- Deep well: theta = 0 (aligned). Shallow well: theta = pi (anti-aligned). Saddle: theta_s = arccos(-h).
- Barrier (shallow to saddle): DeltaV = (1-h)^2
- Eigenvalues: lambda_eq = 2(1-h), lambda_sad = -2(1-h^2)
- Curvature ratio = 1/(1+h) -- VARIES with h (key test)
- 1/(C*tau) = 2*pi/sqrt(1+h) -- varies with h

Experimental reference: Wernsdorfer et al. PRL 78, 1791 (1997).

Physical context: Co nanoparticle (5 nm): K_a = 4.5e5 J/m^3, Vol = 6.5e-26 m^3. K_a*Vol = 2.9e-20 J. At h=0: B=4 requires T ~ 530 K (superparamagnetic transition). At h=0.9: B=4 requires T ~ 5 K.

## Interpretation

- **Josephson junction:** Equal curvatures (|V''(min)| = |V''(sad)| = sqrt(1-gamma^2) for ALL gamma). 1/(C*tau) = 2*pi is constant across all gamma. K = 0.56 confirms 1D SDE parabolic-well class (same as savanna/lake/thermohaline). Equal curvatures do NOT imply K = 1.0 in 1D SDE -- K = 1.0 is specific to 2D Hamiltonian (SMIB) and discrete CME (toggle).
- **Magnetic nanoparticle:** Curvature ratio = 1/(1+h) varies continuously from 1.0 to ~0.5. This is the NON-TRIVIAL B invariance test: beta_0 varies with h, yet B CV stays below 5%. If B invariance holds here, it is a genuine structural property of fold bifurcations, not an artifact of symmetric potentials.
- Both systems land inside the stability window [1.8, 6.0] for all tested parameter values.

These are the strongest tests in the framework because:
1. Zero free parameters (normalized equations)
2. Domains entirely outside the training set (superconducting electronics, nanomagnetism)
3. Predictions made before computation (blind)
4. The nanoparticle tests B invariance under continuously varying curvature asymmetry

## Limitations

1. **1D only:** Both systems are tested in the overdamped 1D limit. Underdamped Josephson junctions (finite capacitance) are 2D and would require QPot methods.
2. **Kramers regime:** Results are valid in the thermal activation regime (barrier >> kT crossover). Below the crossover temperature, macroscopic quantum tunneling dominates and the Kramers framework does not apply.
3. **Single-well escape:** Both scripts compute escape from one well only (shallow well for nanoparticle, single well for Josephson). The reverse escape is not tested (symmetric for Josephson; DeltaV_deep = (1+h)^2 for nanoparticle).

## Conclusions

B invariance and the stability window hold in two physics domains entirely outside the framework's training set, with zero free parameters. The nanoparticle result is particularly strong: B CV stays below 5% even as the curvature ratio varies continuously from 1.0 to ~0.5, confirming B invariance is a genuine structural property of fold bifurcations and not an artifact of potential symmetry. Both tests are GREEN grade -- the strongest replicability rating in the framework.
