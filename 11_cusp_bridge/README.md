# Study 11: Cusp Bridge

**Date:** 2026-04-04

## Purpose

Derives S(d) from cusp bifurcation geometry. Uses the cusp normal form dx/dt = -x^3 + ax + b to establish analytic barrier DeltaPhi = x3*(x2-x1)^3/4, then measures how P(bistable) decays exponentially with parameter-space dimension d in a generic multi-channel ODE.

## Data Provenance

### Cusp normal form (script 1)
- **Guckenheimer & Holmes 1983:** Standard cusp normal form dx/dt = -x^3 + ax + b, potential U(x) = x^4/4 - a*x^2/2 - b*x
- **B_EMPIRICAL = 3.78:** Mean B from bridge_dimensional_scaling.py
- **Omega data (d=5-32):** Hardcoded from bridge_dimensional_scaling.py output
- **S0_TARGET = 1e13:** From Study 09

### Multi-channel ODE (scripts 2-5)
- **Model:** f(x) = a - b*x + sum_i [r_i * x^q_i / (x^q_i + h_i^q_i)], d = 2 + 3*k parameters
- **Parameter distributions:** a ~ U(0.05, 0.80), b ~ U(0.2, 2.0), r_i ~ U(0.1, 2.0), q_i ~ U(2.0, 15.0), h_i ~ U(0.3, 2.0) -- design choices
- **Hardcoded data chain:** Each later script hardcodes P values from earlier scripts (bridge_dimensional_scaling -> bridge_high_d_scaling -> bridge_high_d_analysis / bridge_high_d_extension)

**Provenance claims:** 4.6, 11.5

## Replication

### Requirements
- Python 3.8+
- numpy, scipy

```
pip install numpy scipy
```

### To reproduce
Scripts should ideally be run in order because later scripts hardcode results from earlier ones, but each is self-contained and can run independently.

```bash
cd 11_cusp_bridge

# Recommended order (following the data chain):
python3 bridge_dimensional_scaling.py     # P(bistable) at d=5-32, ~minutes (30,000 samples x 8 d values)
python3 bridge_high_d_scaling.py          # P(bistable) at d=38-62, ~hours (up to 2M samples at d=62)
python3 bridge_high_d_analysis.py         # B computation + model fitting, ~30-60 minutes
python3 bridge_high_d_extension.py        # P(bistable) at d=68-80, ~1.5 hours (up to 10M samples at d=80)
python3 cusp_bridge_derivation.py         # Cusp analytics + bridge, ~seconds (analytic + 300 MFPT)
```

Note: `bridge_high_d_extension.py` is the most expensive script. At d=80, it draws 10,000,000 random configurations. Total wall time ~1.5 hours depending on hardware. Uses two-stage scan (500pt coarse + 2000pt verify) for speed.

All scripts use numpy + scipy only. No matplotlib. No external data files. All scripts seeded (seed=42).

## Files

| File | Contents |
|------|----------|
| `README.md` | This file -- provenance, replication, data sources |
| `RESULTS.md` | Findings, tables, interpretation, limitations |
| `cusp_bridge_derivation.py` | Cusp analytics: B_cusp=2.979, K_cusp=0.558, P(bistable\|2D)=2/5 |
| `bridge_dimensional_scaling.py` | P(bistable) at d=5-32 (30,000 samples per d) |
| `bridge_high_d_scaling.py` | P(bistable) at d=38-62 (up to 2M samples) |
| `bridge_high_d_analysis.py` | B computation at d=38-50 + model fitting |
| `bridge_high_d_extension.py` | P(bistable) at d=68-80 (up to 10M samples); resolves d>50 deceleration |
