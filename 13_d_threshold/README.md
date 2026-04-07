# Study 13: D = 1 Threshold

## What this tests

Tests whether D = MFPT/tau_relax = 1 constitutes a meaningful persistence boundary: D > 1 means the organized state persists longer than it takes to form (dissipative structure), D < 1 means it doesn't. This threshold is novel in the literature -- exhaustive search confirms it does not appear as a named criterion connecting Kramers escape theory to Prigogine's dissipative structures.

## Provenance claims

- 3.1: Kramers MFPT gives D < 1 at high noise (sigma = 1.0, D = 0.68)
- 3.2: Saddle-node bifurcation protects product equation (D_product >= 12.8)

## Models and data sources

### 2-channel synthetic lake model
- **Paper:** Scaffold from van Nes & Scheffer 2007
- **Model:** 1D lake ODE with 2 regulatory channels (Hill^4, Michaelis-Menten) from Step 8
- **Parameters:**
  - A_P = 0.326588 (Source: van Nes & Scheffer 2007)
  - B_P = 0.8 (Source: van Nes & Scheffer 2007 Table 1)
  - R_P = 1.0 (Source: van Nes & Scheffer 2007 Table 1)
  - Q_P = 8 (Source: van Nes & Scheffer 2007 Table 1)
  - H_P = 1.0 (Source: van Nes & Scheffer 2007 Table 1)
  - K1 = 0.5 (**SYNTHETIC** -- half-saturation for Hill^4 channel)
  - K2 = 2.0 (**SYNTHETIC** -- half-saturation for M-M channel)
- **Free parameters:** 0 (synthetic test)

## Scripts

| Script | What it computes | Key output to verify |
|--------|-----------------|---------------------|
| `test_D_below_one_fast.py` | Fast D<1 test (N=20,000 grid) | Saddle-node at eps~0.78, minimum D_product=12.8 |
| `test_D_below_one.py` | Full D<1 test (N=80,000 grid) | `D_exact = 0.68 at sigma = 1.0` (5.6x sigma*) |

### Run order
```bash
pip install numpy scipy
python3 test_D_below_one_fast.py  # fast version (~1 min)
python3 test_D_below_one.py       # full version (~5 min)
```

### Import dependencies
None -- all scripts are self-contained.

## Replicability assessment

### GREEN

Both are synthetic mathematical tests on the lake model scaffold. All parameters trace to van Nes & Scheffer 2007 (model) or are synthetic design choices (channels). Results are deterministic.

- Random seeds: none (deterministic -- exact MFPT integrals)
- Dependencies: numpy + scipy only
- Output: stdout only

## Key results

- D < 1 IS achievable via Kramers at high noise: D = 0.68 at sigma = 1.0
- The product equation CANNOT reach D = 1: saddle-node bifurcation destroys bistability at eps_1 ~ 0.78, where D_product = 12.8
- D = 1 corresponds to B_eff << 1.8, far outside the stability window [1.8, 6.0]
- The product equation's domain is D in [~4, infinity) depending on system
