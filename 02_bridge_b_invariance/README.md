# Study 02: Bridge & B Invariance

**Date:** 2026-04-04

## Purpose

Proves B = 2*DeltaPhi/sigma*^2 is a structural invariant (CV < 5%) across the bistable range of each system, and decomposes the bridge identity ln(D) = B + beta_0. Tests B invariance for lake, savanna, kelp, coral, and toggle switch systems.

## Data Provenance

| Script | System | Model source |
|--------|--------|-------------|
| `bridge_v2_B_invariance_proof.py` | Lake (q=3..20) | van Nes & Scheffer 2007 |
| `patha_v2_what_determines_B.py` | Lake + 4 ecological systems | van Nes & Scheffer 2007; multiple |
| `structural_connection_test.py` | Lake | van Nes & Scheffer 2007; Hakanson 2000; Nature Comms 2024 (159 lakes) |
| `structural_B_savanna.py` | Savanna | Xu et al. 2021 (Staver-Levin model) |
| `structural_B_kelp.py` | Kelp | r=0.4, K=668, h=100 (provenance in Study 01: Brey 2001, Ling et al. 2015) |
| `structural_B_coral.py` | Coral | Mumby 2007, Nature 450:98-101 |
| `structural_B_toggle.py` | Toggle switch | Gardner et al. 2000, Hill n=2; CME data from STEP9 (internal) |

**Provenance claims:** 1.5, 2.1, 2.2, 2.3, 2.4, 2.5

## Replication

### Requirements
- Python 3.8+
- numpy, scipy (no other dependencies)

### Commands

All scripts are independent. No import dependencies between scripts. No random seeds. Run from the repository root:

```bash
python3 02_bridge_b_invariance/bridge_v2_B_invariance_proof.py    # ~2 min (240 MFPT integrals)
python3 02_bridge_b_invariance/patha_v2_what_determines_B.py      # ~1 min
python3 02_bridge_b_invariance/structural_connection_test.py      # seconds
python3 02_bridge_b_invariance/structural_B_savanna.py            # seconds
python3 02_bridge_b_invariance/structural_B_kelp.py              # seconds
python3 02_bridge_b_invariance/structural_B_coral.py             # seconds
python3 02_bridge_b_invariance/structural_B_toggle.py            # seconds
```

Runtime: ~3 minutes total. The core proof script (`bridge_v2_B_invariance_proof.py`) takes ~2 minutes for 240 MFPT integrals; all others run in seconds.

## Files

| File | Contents |
|------|----------|
| `README.md` | This file -- provenance, replication, data sources |
| `RESULTS.md` | Findings, tables, interpretation |
| `bridge_v2_B_invariance_proof.py` | Core B invariance proof: 240 MFPT computations (also in `../scripts/`) |
| `patha_v2_what_determines_B.py` | Beta_0 decomposition and 4 ecological systems (also in `../scripts/`) |
| `structural_connection_test.py` | Structural connection sigma*(a)*|lambda_eq(a)|/a (also in `../scripts/`) |
| `structural_B_savanna.py` | Savanna B invariance (also in `../scripts/`) |
| `structural_B_kelp.py` | Kelp B invariance (also in `../scripts/`) |
| `structural_B_coral.py` | Coral B invariance (also in `../scripts/`) |
| `structural_B_toggle.py` | Toggle B invariance (also in `../scripts/`) |
