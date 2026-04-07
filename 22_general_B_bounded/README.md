# Study 22: General B Boundedness Proof

**Date:** 2026-04-06

## Purpose

Determine whether B = 2*DeltaPhi/sigma*^2 boundedness is a property of the cusp normal form specifically or a universal property of Kramers escape theory. Resolves Study 19 Limitation #1.

## Data Provenance

All parameters are internal to the potential normal forms. Zero free parameters. Zero external data.

| Parameter | Value | Source |
|-----------|-------|--------|
| Cusp: K | 0.558 | Study 11 (cusp_bridge_derivation.py) |
| Washboard: K | 0.56 | Study 11 (blind_test_josephson_junction.py) |
| Nanomagnet: K | 0.57 | Study 11 (blind_test_magnetic_nanoparticle.py) |
| D_target | 100 (default), also 29, 500, 1111 | Ecological range |
| Scale c range | 0.1 to 10.0 | Arbitrary (result is c-independent) |
| Random seed | 42 | Reproducible |

## Replication

### Requirements
```
Python 3.8+
pip install numpy scipy
```

### Run commands
```bash
# Main study (4 normal-form families):
python3 22_general_B_bounded/study22_general_B_bounded.py

# Coral verification (Mumby 2007 ecological model):
python3 22_general_B_bounded/study22_coral_B_verification.py

# Or from scripts folder:
python3 ../scripts/study22_general_B_bounded.py
python3 ../scripts/study22_coral_B_verification.py
```

### Runtime
- Main study: ~50 seconds
- Coral verification: ~5 minutes

### Expected output (main study)
4 tests + summary. Key lines to verify:
- Test 1: "CV = 0.00%" for all 4 families
- Test 2: B widths < 1.0 for all families; washboard width = 0.034
- Test 3: All 82 B values in [1.8, 6.0]
- Final: "ALL FOUR FAMILIES: B is bounded, scale-invariant, and narrow"

### Expected output (coral verification)
4 tests + summary. Key lines to verify:
- Test 1: "CV = 0.0000%" (scale invariance)
- Test 2: B = 6.07 +/- 2.15%, width = 0.440
- Test 3: Coral B at D=29 inside cusp range; D=100/500/1111 NEAR (above by 0.35-0.51)
- Final: "ALL TESTS PASS"

## Files

| File | Contents |
|------|----------|
| `README.md` | This file -- provenance, replication, data sources |
| `RESULTS.md` | Findings, tables, interpretation, limitations |
| `study22_general_B_bounded.py` | Main study script (also in `../scripts/`) |
| `study22_coral_B_verification.py` | Coral verification script (also in `../scripts/`) |
