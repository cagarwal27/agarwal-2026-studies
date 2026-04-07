# Study 15: Blind Search Equation Tests

**Date:** 2026-04-05
**Provenance claims:** 15.1, 15.2

## Purpose

Two blind timescale predictions of the search equation tau = n * S(d) / (v * P). Both innovations (flowers, CAM) are NOT in the calibration set (Steps 12-12e). All inputs come from independent sources. The second test (CAM) adds a multi-origin constraint not available in the first (flowers, 1 origin).

## Data Provenance

| Input | Source | Used in |
|-------|--------|---------|
| gamma = 0.317 | Keefe & Szostak 2001, Nature 410:715 (directed evolution: fraction of random 80-aa proteins that bind ATP ~ 10^-11) | Both tests |
| d = 51 (flowers) | van Mourik et al. 2010, BMC Syst Biol 4:101 (floral ODE model) | 15.1 |
| d = 72/40/19 (CAM) | Bartlett, Vico & Porporato 2014, Plant and Soil (CAM ODE model) | 15.2 |
| tau_flowers_obs = ~200 Myr | Fossil record (seed plants ~365 Ma -> angiosperms ~130 Ma) | 15.1 |
| tau_CAM_obs = ~30 Myr | Paleobotany (geom. mean of 15 and 65 Myr) | 15.2 |
| N_origins_CAM = 62 | Silvera et al. 2010; Edwards & Ogburn 2012 | 15.2 |
| v, P, n ranges | See RESULTS.md provenance tables | Both tests |

**Key innovation:** gamma comes from directed evolution experiments, NOT from the cusp bridge. This eliminates the model-reduction-ratio problem entirely.

## Replication

### Requirements

```
Python 3.8+
pip install numpy scipy
```

No external data files. No matplotlib. Self-contained.

### Commands

```bash
# Claim 15.1: Angiosperm flowers prediction
python3 blind_flowers_prediction.py

# Claim 15.2: CAM photosynthesis prediction
python3 blind_cam_prediction.py
```

### Runtime

Each script completes in < 1 second (pure arithmetic, no simulation).

### Expected output

- `blind_flowers_prediction.py`: tau_predicted = 126 Myr vs 200 Myr observed (-0.20 OOM)
- `blind_cam_prediction.py`: tau at d=72 = 204 Myr vs 30 Myr observed (+0.83 OOM); multi-origin test converges on d ~ 52

## Files

| File | Description |
|------|-------------|
| `README.md` | This file (replication instructions) |
| `RESULTS.md` | Full results, interpretation, and provenance |
| `blind_flowers_prediction.py` | Claim 15.1: angiosperm flower timescale prediction |
| `blind_cam_prediction.py` | Claim 15.2: CAM photosynthesis timescale and multi-origin prediction |
