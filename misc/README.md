# Miscellaneous: Lake Model 2D Investigations

## What these are

Historical exploration scripts that investigated the 2D macrophyte-turbidity lake model ("Model C") from van Nes & Scheffer 2005/2007. These scripts explored equilibria, barrier computation methods, and channel weakening -- foundational work that informed the channel independence criterion (Study 12) and the bridge identity (Study 02). They do not directly test any of the 11 framework equations and do not produce a specific replicable claim.

## Why these are in misc

They are early-phase exploration of the 2D lake model, testing formulations and methods. The insights from these investigations (channel structure, barrier computation, epsilon candidates) were incorporated into the framework, but the scripts themselves are investigative -- they search for the right model formulation rather than testing a pre-specified hypothesis. They are included for completeness and historical provenance.

## Models and data sources

### 2D macrophyte-turbidity lake model
- **Paper:** van Nes & Scheffer 2007, Limnology and Oceanography
- **Model:** 2D ODE (submerged vegetation E, turbidity V) with Hill-function feedbacks
- **Parameters:**
  - r_E = 0.1 day^-1 (Source: van Nes & Scheffer 2007 Table 1)
  - r_V = 0.05 day^-1 (Source: van Nes & Scheffer 2007 Table 1)
  - h_V = 0.2 (Source: van Nes & Scheffer 2007 Table 1, swept in some scripts)
  - h_E = 2.0 (Source: van Nes & Scheffer 2007 Table 1, swept in some scripts)
  - p = 4 (Source: van Nes & Scheffer 2007 Table 1)
  - E_0 = 5.0 (nutrient loading, scanned for bistable range)
- **Free parameters:** 0 (all from published Table 1)

## Scripts

| Script | What it computes | Grade |
|--------|-----------------|-------|
| `phase2_model_c_equilibria.py` | Equilibria and eigenvalues; E_0 scan for bistable range; Kramers-Langer prefactor | GREEN |
| `phase2_model_c_v2.py` | Tests 4 candidate formulations for bistability (both-Hill, coupled-sigmoid, mixed, original) | YELLOW |
| `phase2_barrier_action.py` | Quasi-potential barrier via MAM/Freidlin-Wentzell action; tests adjoint and direct methods | YELLOW |
| `phase3_channel_weakening.py` | Channel weakening: barrier response to h_V and h_E changes; candidate epsilon definitions | YELLOW |

### Run order
```bash
pip install numpy scipy
python3 phase2_model_c_equilibria.py     # step 1: find equilibria
python3 phase2_model_c_v2.py             # step 2: test formulations
python3 phase2_barrier_action.py         # step 3: compute barriers
python3 phase3_channel_weakening.py      # step 4: channel weakening
```

### Import dependencies
None -- all scripts are self-contained.

## Replicability assessment

### YELLOW

- `phase2_model_c_equilibria.py` is GREEN (all params from van Nes & Scheffer 2007 Table 1)
- Other 3 scripts are YELLOW: model params are standard but not always cited in each individual file; some equilibrium coordinates are pre-computed and hardcoded
- Two scripts write to hardcoded absolute paths in `THEORY/X2/scripts/` -- these paths will not exist on external machines
- One script writes JSON to `THEORY/X2/scripts/phase3_cases.json`
- No random seeds (all deterministic)
- Dependencies: numpy + scipy only
