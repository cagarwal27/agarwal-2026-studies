# Study 02: Bridge & B Invariance -- Results

**Date:** 2026-04-04
**Scripts:** `02_bridge_b_invariance/*.py`

**Provenance claims:** 1.5, 2.1, 2.2, 2.3, 2.4, 2.5

## Question

Is B = 2*DeltaPhi/sigma*^2 a structural invariant across the bistable range of each system? How does it decompose as ln(D) = B + beta_0?

## Key Results

### B Invariance Across Systems

| System | D_target | B | CV | DeltaPhi variation | Method |
|--------|----------|------|------|-------------------|--------|
| Lake | 200 | 4.27 | 2.0% | 18-125x across q | Exact MFPT (240 computations) |
| Savanna | 100 | 3.74 | 4.6% | 77x | Exact MFPT |
| Kelp | 29.4 | 1.80 | 2.6% | 27,617x | Exact MFPT (detrended) |
| Coral | 1,111 | 6.04 | 2.1% | 2,625x | Exact MFPT |
| Toggle | 1,000 | 4.83 | 3.8% | -- | CME spectral regression |

### Beta_0 Decomposition

Beta_0 = ln(D) - B:
- CV = 2.9% across 49 epsilon combos spanning D = 11 to 10,000 (fixed ODE)
- Range [0.53, 1.61] across systems (NOT universal)
- Formula: beta_0 = ln(K_corr) + ln(pi) + 0.5*ln(|lambda_eq/lambda_sad|)
- Slope = 1.016, R^2 = 0.9999 across 49 combos

---

## Per-Script Details

### 1. bridge_v2_B_invariance_proof.py (Lake)

Core proof. 240 MFPT computations at 8 Hill coefficients (q=3..20) on the lake model. Shows B CV < 3% for all q. Decomposes ln(D) = B + beta_0.

**Parameters:** Lake model: b=0.8, r=1.0, h=1.0, D=200, q=3..20. Source: van Nes & Scheffer 2007.

**Status:** YELLOW. Self-contained, exact MFPT integrals.

### 2. patha_v2_what_determines_B.py

Four tests: beta across 4 ecological systems, 2-channel lake, epsilon grid, beta_0 formula.

Beta_0 CV=2.9% across 49 epsilon combos (slope 1.016, R^2=0.9999).

**Status:** YELLOW. Self-contained, 4 independent tests.

### 3. structural_connection_test.py (Lake)

Tests structural connection sigma*(a)*|lambda_eq(a)|/a in the lake model. References Hakanson 2000 CV(TP)=35%, Nature Comms 2024 159 lakes=35%.

**Status:** GREEN. Literature references inline.

### 4. structural_B_savanna.py

Savanna B=3.74, CV=4.6%. Staver-Levin model with Xu et al. 2021 params.

**Status:** YELLOW. Xu et al. 2021 cited but params not individually sourced.

### 5. structural_B_kelp.py

Kelp B=1.80, CV=2.6%. Params: r=0.4, K=668, h=100.

**Status:** RED. No literature citations in script. Parameter provenance exists elsewhere in the framework (Brey 2001, Ling et al. 2015) but is not stated in this script.

### 6. structural_B_coral.py

Coral B=6.04, CV=2.1%. Mumby 2007 Nature 450:98-101 params (a=0.1, gamma=0.8, r=1.0, d=0.44).

**Status:** GREEN. Mumby 2007 fully cited.

### 7. structural_B_toggle.py

Toggle B=4.83, CV=3.8%. Gardner et al. 2000 toggle switch, Hill n=2, alpha in {5,6,8,10}. CME data hardcoded from STEP9 results (internal).

**Status:** YELLOW. CME data hardcoded from internal computations.

---

## Replicability Assessment

**Overall: YELLOW**

| Script | Grade | Notes |
|--------|-------|-------|
| `bridge_v2_B_invariance_proof.py` | YELLOW | Self-contained, exact MFPT integrals |
| `patha_v2_what_determines_B.py` | YELLOW | Self-contained, 4 independent tests |
| `structural_connection_test.py` | GREEN | Literature references inline |
| `structural_B_savanna.py` | YELLOW | Xu et al. 2021 cited but params not individually sourced |
| `structural_B_kelp.py` | RED | Params unsourced in script |
| `structural_B_coral.py` | GREEN | Mumby 2007 fully cited |
| `structural_B_toggle.py` | YELLOW | CME data hardcoded from internal computations |

## Interpretation

B = 2*DeltaPhi/sigma*^2 is confirmed as a structural invariant with CV < 5% across all tested systems, despite DeltaPhi varying by orders of magnitude within each system's bistable range. This means the barrier height and noise level are structurally coupled -- when the barrier changes (e.g., by varying the bifurcation parameter), sigma* adjusts to maintain the same B.

The bridge identity ln(D) = B + beta_0 holds with beta_0 being system-specific (range [0.53, 1.61]) but extremely stable within a system (CV = 2.9% across 49 epsilon combinations). The formula beta_0 = ln(K_corr) + ln(pi) + 0.5*ln(|lambda_eq/lambda_sad|) provides the analytical decomposition.
