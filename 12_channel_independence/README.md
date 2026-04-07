# Study 12: Channel Independence

## What this tests

Tests the formal criterion for when the product equation D = prod(1/epsilon_i) applies. The hypothesis: the product equation requires (1) effective 1D escape dynamics (native 1D or valid adiabatic reduction with eigenvalue ratio > 5), and (2) separable additive channels in the effective 1D drift. This is NOT an exogenous/endogenous distinction.

## Provenance claims

- Fact 29: k=3 multiplicative stacking confirmed (3.7x-133x discrimination)
- Fact 31: Formal criterion: 1D-reducible + separable

## Models and data sources

### Synthetic 3-channel lake model (step8)
- **Paper:** Scaffold from van Nes & Scheffer 2007
- **Model:** 1D lake ODE with 3 added regulatory channels: Hill^4, Michaelis-Menten, Hill^2
- **Parameters:**
  - A_P = 0.326588 (Source: midpoint of bistable range, van Nes & Scheffer 2007)
  - B_P = 0.8 (Source: van Nes & Scheffer 2007 Table 1)
  - R_P = 1.0 (Source: van Nes & Scheffer 2007 Table 1)
  - Q_P = 8 (Source: van Nes & Scheffer 2007 Table 1)
  - H_P = 1.0 (Source: van Nes & Scheffer 2007 Table 1)
  - K1 = 0.5 (**SYNTHETIC** -- design choice, half-saturation for Hill^4 channel)
  - K2 = 2.0 (**SYNTHETIC** -- design choice, half-saturation for Michaelis-Menten channel)
  - K3 = 1.0 (**SYNTHETIC** -- design choice, half-saturation for Hill^2 channel)
- **Free parameters:** 0 (K1, K2, K3 are design choices, not free params)

### Synthetic 4-channel lake model + toggle switch CME (step11)
- **Model:** Same lake scaffold + 4th perturbative channel (Hill^3, K4=1.5). Also tests Gardner et al. 2000 toggle switch at alpha=[5,6,8,10] via Chemical Master Equation spectral gap.
- **Parameters:** Same lake base + K4 = 1.5 (synthetic). Toggle: Hill n=2, various Omega.

## Scripts

| Script | What it computes | Key output to verify |
|--------|-----------------|---------------------|
| `step8_synthetic_3channel.py` | 3-channel multiplicative stacking | `D_product/D_exact = 1.00 +/- <1%`; discrimination 3.7x-133x vs alternatives |
| `step11_channel_independence.py` | Formal criterion + toggle CME test | `12/12 systems correctly classified` |

### Run order
```bash
pip install numpy scipy
python3 step8_synthetic_3channel.py    # independent
python3 step11_channel_independence.py # independent
```

### Import dependencies
None -- all scripts are self-contained.

## Replicability assessment

### GREEN

Both scripts are synthetic tests with designed parameters. All lake model parameters trace to van Nes & Scheffer 2007. Channel half-saturations (K1-K4) are deliberate design choices for a mathematical test, not empirical parameters requiring sourcing.

- Random seeds: none (deterministic -- MFPT integrals + CME spectral method)
- Dependencies: numpy + scipy only
- Output: stdout + markdown results files

## Key results

- Multiplicative D = prod(1/eps_i) confirmed with 3.7x-133x discrimination over additive, harmonic, and geometric alternatives
- Formal 2x2 classification criterion: {1D, 2D-reducible, 2D-irreducible} x {constitutive, perturbative}
- 12/12 systems correctly classified: separable works, coupled (toggle) fails
- The product equation failure for the toggle is structural (coupled loop), not a parameter issue
