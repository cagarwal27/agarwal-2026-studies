# Study 07: Cross-Domain Physics

**Date:** 2026-04-06

## Purpose

Tests Kramers equation in 4 non-biological domains (climate, finance, political science, power systems). Quality varies widely from GREEN (thermohaline) to RED (Soviet, financial).

## Data Provenance

### Thermohaline (AMOC)
Cessi 1994 AMOC model. mu^2=4.0, t_d=219 yr. 0 free parameters. Citations are informal (no page/table references).

### Financial
Generic cusp catastrophe with 6 (q,a) parameter sets. NOT calibrated to Diks-Wang S&P data or any real market.

### Soviet
Kuran preference falsification theory converted to cubic ODE. 4 free parameters (S_lo=0.05, S_mid=0.35, S_hi=0.90, gamma=1.0), 0 calibrated. The ODE conversion is unpublished.

### Power grid
SMIB swing equation with Brazilian grid params from Ritmeester & Meyer-Ortmanns 2022. D_obs undefined (SOC dynamics, not Kramers escape).

**Provenance claims:** 5.3, 10.3, 10.4, 10.5

## Replication

### Requirements
- Python 3.8+
- numpy, scipy

```
pip install numpy scipy
```

All scripts self-contained. No import dependencies between scripts. No external data files. `power_grid_kramers.py` writes a results file to disk.

### To reproduce

Scripts can be run in any order. No dependencies between them.

```bash
cd 07_cross_domain_physics/

python3 thermohaline_kramers.py      # AMOC model (~seconds, deterministic)
python3 financial_cusp_kramers.py    # Cusp catastrophe (~seconds, deterministic)
python3 soviet_kuran_kramers.py      # Kuran model (~seconds, deterministic)
python3 power_grid_kramers.py        # SMIB swing (~seconds, writes results file)
```

### Replicability grades

**Overall: RED**

| Script | Grade | Notes |
|--------|-------|-------|
| `thermohaline_kramers.py` | YELLOW | 0 free params but informal citations (no page/table) |
| `financial_cusp_kramers.py` | RED | Generic params, not calibrated to real data |
| `soviet_kuran_kramers.py` | RED | 4 free params, 0 calibrated, unpublished ODE conversion |
| `power_grid_kramers.py` | YELLOW | Published params but D_obs undefined for SOC systems |

The Soviet and financial scripts are RED because they lack calibration to real data. The thermohaline and power grid scripts are individually YELLOW with legitimate published model sources, but the overall study grade is RED due to the two RED scripts.

## Files

| File | Contents |
|------|----------|
| `README.md` | This file -- provenance, replication, data sources |
| `RESULTS.md` | Findings, tables, interpretation, selection filter analysis |
| `thermohaline_kramers.py` | Cessi 1994 AMOC model (K=0.55, 0 free params) |
| `financial_cusp_kramers.py` | Cusp catastrophe (6 generic parameter sets) |
| `soviet_kuran_kramers.py` | Kuran preference falsification as cubic ODE |
| `power_grid_kramers.py` | SMIB swing equation (Brazilian grid params) |
