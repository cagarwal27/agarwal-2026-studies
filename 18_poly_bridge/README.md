# Study 18: Polynomial Bridge Scaling

**Date:** 2026-04-06

## Purpose

Test whether gamma ~ 0.20 (the exponential decay rate of P(organized | d parameters)) is a geometric universal or an artifact of Hill-function saturation. Uses random 2D polynomial ODEs instead of Hill-function ODEs. A follow-up degradation test (RESULTS_DEGRADATION.md) further isolates whether the decay comes from saturation or degradation architecture.

## Data Provenance

No external data. All results are from Monte Carlo sampling of random polynomial ODE coefficients.

| Parameter | Value | Source |
|-----------|-------|--------|
| gamma_cusp (reference) | 0.197, R^2 = 0.995 | Study 11, 1D Hill-function ODEs |
| gamma_Hopf (reference) | 0.175, R^2 = 0.9997 | Study 16, 2D Hill-function ODEs |
| Cubic damping eps | 0.01 (fixed) | Chosen to prevent divergence without affecting local dynamics |
| Coefficient ranges | Degree-dependent: r_k chosen so total contribution per degree is O(1) at \|x\|,\|y\| ~ 1 | See COEFF_RANGES in poly_bridge_scaling.py |
| Degradation range (b1, b2) | U[0.2, 2.0] | Matches Study 16 degradation range |
| Random seed | 42 | All scripts |

### Model (polynomial, no saturation)
```
dx/dt = sum_{i+j<=p} a_{ij} * x^i * y^j  -  0.01 * x * (x^2 + y^2)
dy/dt = sum_{i+j<=p} b_{ij} * x^i * y^j  -  0.01 * y * (x^2 + y^2)
```
Parameters: d = (p+1)(p+2) random coefficients per sample.

### Model (polynomial + degradation)
Same as above, but the x-equation's x-coefficient and y-equation's y-coefficient are forced negative: -U[0.2, 2.0].

## Replication

### Requirements
- Python 3.8+
- numpy, scipy

No other dependencies. `poly_degradation_gpu.py` optionally uses CuPy for GPU acceleration (set USE_GPU = True on Colab with GPU runtime).

### Commands
```bash
cd 18_poly_bridge/

# Main polynomial scaling test (~2.7 hours, single core)
python3 poly_bridge_scaling.py

# Degradation follow-up (~5.4 hours, single core)
python3 poly_degradation_test.py

# GPU-accelerated degradation (optional, for Colab)
python3 poly_degradation_gpu.py
```

### Runtime
- `poly_bridge_scaling.py`: ~2.7 hours (1.9M samples)
- `poly_degradation_test.py`: ~5.4 hours (7.8M samples)
- `poly_degradation_gpu.py`: faster with GPU, same computation as poly_degradation_test.py

## Files

| File | Purpose |
|------|---------|
| `README.md` | This file -- provenance, replication, data sources |
| `RESULTS.md` | Main results: P(limit cycle \| d) for polynomial ODEs, gamma_poly = 0.010 |
| `RESULTS_DEGRADATION.md` | Degradation follow-up: gamma_poly_degrad = 0.007, confirms saturation is required |
| `poly_bridge_scaling.py` | Main polynomial scaling script (Study 18 core computation) |
| `poly_degradation_test.py` | Polynomial + degradation test (CPU) |
| `poly_degradation_gpu.py` | Polynomial + degradation test (GPU-enabled, CuPy optional) |
| `run_output.txt` | Raw output from poly_bridge_scaling.py |
| `degradation_output.txt` | Raw output from poly_degradation_test.py |
