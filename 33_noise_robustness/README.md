# Study 33: Noise Robustness

**Date:** 2026-04-07
**Motivation:** Reviewer concern 1.1/2.6 — "Additive white noise unjustified; colored noise timescales comparable to relaxation times." Reviewer's stated #1 priority: "Demonstrate that B invariance holds under multiplicative noise g(x) = sigma*sqrt(x) for at least one ecological system."

## System

Lake eutrophication model (van Nes & Scheffer 2005):

```
f(x) = a - b*x + r*x^q / (x^q + h^q)
```

Parameters: b=0.8, r=1.0, h=1.0, q=3-16. Source: van Nes & Scheffer, Ecology 86(7):1797-1807, 2005.

## Scripts

| Script | Purpose | Method |
|--------|---------|--------|
| `multiplicative_B_invariance.py` | B invariance under g(x) = sigma*sqrt(x) | Exact MFPT integral (state-dependent diffusion) |
| `colored_noise_B_invariance.py` | B invariance under OU colored noise, tau_c = tau_relax | Kramers correction (Hanggi et al. 1990) + SDE verification |
| `ito_stratonovich_correction.py` | Ito vs Stratonovich comparison | Exact MFPT under both interpretations |

## Data Provenance

- Lake ODE parameters: van Nes & Scheffer 2005 (same as Studies 01, 02)
- Kramers colored-noise correction: Hanggi P, Talkner P, Borkovec M, Rev. Mod. Phys. 62:251-341, 1990, Eq. 4.56b
- Ito-Stratonovich conversion: Gardiner CW, Stochastic Methods, 4th ed., Springer, 2009, Ch. 4
- Multiplicative noise MFPT: standard 1D Fokker-Planck with state-dependent diffusion (Gardiner Ch. 5)

## How to Replicate

```bash
python3 multiplicative_B_invariance.py    # ~2 min
python3 ito_stratonovich_correction.py    # ~2 min
python3 colored_noise_B_invariance.py     # ~2 min (analytical) + SDE
```

Requires: numpy, scipy (no other dependencies).

## Key Equations

**Multiplicative noise MFPT** for dx = f(x)dt + sigma*sqrt(x)*dW (Ito):

Modified quasipotential: U_m(x) = -integral f(x)/x dx

MFPT = integral exp(Phi(y)) * [integral (2/(sigma^2*z)) exp(-Phi(z)) dz] dy

where Phi(x) = 2*U_m(x)/sigma^2.

B_mult = 2*DeltaU_m / sigma*^2

**Ito-Stratonovich correction** for g(x) = sigma*sqrt(x):

g(x)*g'(x) = sigma^2/2, so (1/2)*g*g' = sigma^2/4.

Stratonovich drift = f(x) - sigma^2/4 (constant shift).

Barrier correction: DeltaU_Strat - DeltaU_Ito = (sigma^2/4)*ln(x_sad/x_eq).

**Colored noise correction** (Hanggi-Talkner-Borkovec):

lambda_+ = [-1/tau_c + sqrt(1/tau_c^2 + 4*omega_b^2)] / 2

mu = lambda_+ / omega_b (rate correction factor)

MFPT_colored = MFPT_white * omega_b / lambda_+
