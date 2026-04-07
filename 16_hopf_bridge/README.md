# Study 16: Hopf Bridge Scaling

**Date:** 2026-04-05
**Provenance claims:** 16.1, 16.2

## Purpose

Measures P(stable limit cycle | d) for random 2D multi-channel Hill-function ODEs. Tests whether the cusp bridge result P(bistable) ~ exp(-gamma*d) extends to Hopf bifurcations. First computation testing the framework's scope beyond bistable systems.

## Data Provenance

No external data. All parameters are internal to the 2D Hill-function ODE model. Parameter ranges matched to cusp bridge (Study 11).

| Parameter | Value | Notes |
|-----------|-------|-------|
| a1, a2 | U[0.05, 0.80] | Matched to cusp bridge |
| b1, b2 | U[0.2, 2.0] | Matched to cusp bridge |
| c1, c2 | U[-1.0, 1.0] | Both signs required for oscillation |
| r1_i, r2_i | U[0.1, 2.0] | Matched to cusp bridge |
| q1_i, q2_i | U[2.0, 15.0] | Matched to cusp bridge |
| k1_i, k2_i | U[0.3, 2.0] | Matched to cusp bridge |
| seed | 42 | np.random.seed(42) |

Parameters: d = 6 + 6*n_channels (6 base: a1, a2, b1, b2, c1, c2; 6 per channel: r1_i, q1_i, k1_i, r2_i, q2_i, k2_i).

## Replication

### Requirements

```
Python 3.8+
pip install numpy scipy
```

No matplotlib. No external data files. Self-contained.

### Commands

```bash
python3 hopf_bridge_scaling.py
```

Output goes to stdout; redirect to capture:

```bash
python3 hopf_bridge_scaling.py > results_output.txt 2>&1
```

### Runtime

~75 minutes for d=12-36, ~132 minutes for d=42 (5M samples). Total ~210 minutes on a modern laptop. The script runs all d values sequentially and stops early if two consecutive d values produce zero limit cycles.

### Expected output

Best estimate: gamma_Hopf = 0.172 (from d >= 24, 4 decay-regime points, R^2 = 0.9997). Ratio to gamma_cusp: 0.87 (13% difference).

### Key numerical parameters

| Parameter | Value |
|-----------|-------|
| NEWTON_ITERS | 20 |
| NEWTON_MAX_STEP | 2.0 |
| EQ_TOL | 1e-6 |
| T_INTEGRATE | 300.0 |
| DIVERGE_THRESHOLD | 100.0 |
| PERIODIC_CV | 0.20 |
| MIN_PEAKS | 5 |
| MIN_AMPLITUDE | 0.01 |
| BATCH_SIZE | 10,000 |

Sample sizes per d: 100K (d=12), 250K (d=18), 500K (d=24), 1M (d=30), 2M (d=36), 5M (d=42). Increased ~5x over original specification because P(limit cycle) is ~100-1000x lower than P(bistable) at the same d.

## Files

| File | Description |
|------|-------------|
| `README.md` | This file (replication instructions) |
| `RESULTS.md` | Full results, interpretation, and provenance |
| `hopf_bridge_scaling.py` | Main script: P(limit cycle) at d=12-42 with 3 model fits |
| `results_output.txt` | Captured stdout from a complete run |
