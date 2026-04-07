# Study 30: Universal Data Collapse

**Date:** 2026-04-06

## Purpose

Construct a single figure where data from 13 physically unrelated bistable systems all collapse onto one universal Kramers curve, testing ln(D) = beta_0 + B. This is the standard of proof for universality claims in physics.

## Data Provenance

All operating-point (B, D) values come from ../SYSTEMS.md "B Stability Window" table. These were computed in prior studies:

| System | Computed in | Script |
|--------|-----------|--------|
| Lake | Study 01, 02 | phase1_lake_1d.py |
| Kelp | Study 02 | structural_B_kelp.py |
| Coral | Study 02 | structural_B_coral.py |
| Savanna | Study 02 | structural_B_savanna.py |
| Trop. forest | Study 01 | step10_tropical_forest_kramers.py |
| Peatland | Study 01 | step13_peatland_kramers.py |
| Josephson jn. | Study 08 | blind_test_josephson_junction.py |
| Magn. nanopart. | Study 08 | blind_test_magnetic_nanoparticle.py |
| GBP ERM peg | Study 21 | currency_peg_kramers.py |
| Thai Baht peg | Study 21 | currency_peg_kramers.py |
| Tumor-immune | Study 07 | structural_B_tumor_immune_2D.py |
| Toggle switch | Study 04 | structural_B_toggle.py |
| Diabetes | Study 06 | structural_B_diabetes_2D_SDE.py |

For JJ and nanoparticle, D is not fixed by a product equation (these are k=0, single-channel systems). The plotted D is estimated from the Kramers prefactor at the representative operating point: D ~ exp(B + beta_0) where beta_0 is computed from eigenvalues.

ODE sources and parameter values for all 13 systems are listed in RESULTS.md.

## Replication

### Requirements
- Python 3.8+
- numpy, scipy, matplotlib, numba (numba required for 2D SDE sweeps only)

### Commands
```bash
cd 30_data_collapse/

# Step 1: Run 2D SDE sweeps (~5-10 minutes, requires numba)
python3 sweep_2d_sde.py

# Step 2: Generate collapse figure (reads sweep_*.npz, ~30 seconds)
python3 data_collapse.py
```

### Runtime
- `sweep_2d_sde.py`: ~5-10 minutes
- `data_collapse.py`: ~30 seconds

## Method

For each 1D system (or system with a valid adiabatic reduction to 1D):

1. Fix the bifurcation parameter at a representative operating point (midpoint of bistable range).
2. Sweep noise intensity sigma continuously (50 values, log-spaced, spanning B from 0.8 to 9.0).
3. At each sigma, compute D_exact via the exact 1D MFPT integral (Gardiner formula), NOT the Kramers approximation.
4. Compute B = 2*DeltaPhi/sigma^2.
5. Plot ln(D_exact) vs B for all systems.

For the 3 irreducibly 2D systems (tumor-immune, toggle switch, diabetes), only single operating-point data is plotted (no sigma sweep), since the 2D MFPT cannot be computed from the 1D integral.

## Files

| File | Purpose |
|------|---------|
| `README.md` | This file -- provenance, replication, data sources |
| `RESULTS.md` | Findings, tables, interpretation |
| `data_collapse.py` | Main script: 1D MFPT sweeps + 2D SDE data + figure |
| `sweep_2d_sde.py` | 2D SDE sigma sweeps for tumor-immune and diabetes |
| `sweep_tumor.npz` | Cached tumor SDE results (13 sigma values) |
| `sweep_diabetes.npz` | Cached diabetes SDE results (12 eta values) |
| `plots/fig_collapse.pdf` | Vector figure |
| `plots/fig_collapse.png` | Raster preview |
