#!/usr/bin/env python3
"""
Phase 1: 1D Lake Analytical Verification for X² Bridge Test.
Tests whether D_product = D_Kramers for the 1D lake model.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

# ============================================================
# Model parameters (van Nes & Scheffer 2007, Table 1)
# ============================================================
b = 0.8      # P loss rate
r = 1.0      # max recycling rate
q = 8        # Hill coefficient
h = 1.0      # half-saturation
a = 0.326588 # loading (midpoint of bistable range)

# Known equilibria (from Round 9)
x_clear = 0.409217
x_sad   = 0.978152
x_turb  = 1.634126
lam_clear = -0.784651
lam_sad   =  1.228791
tau_L     = 1.0 / abs(lam_clear)  # relaxation time

def f_lake(x):
    return a - b*x + r * x**q / (x**q + h**q)

def f_lake_deriv(x):
    """Derivative f'(x) for verification."""
    return -b + r * q * x**(q-1) * h**q / (x**q + h**q)**2

# ============================================================
# Verify equilibria and eigenvalues
# ============================================================
print("=" * 70)
print("PHASE 1: 1D LAKE ANALYTICAL VERIFICATION")
print("=" * 70)

print("\n--- Equilibrium verification ---")
print(f"f(x_clear={x_clear:.6f}) = {f_lake(x_clear):.2e}")
print(f"f(x_sad={x_sad:.6f})   = {f_lake(x_sad):.2e}")
print(f"f(x_turb={x_turb:.6f}) = {f_lake(x_turb):.2e}")
print(f"f'(x_clear) = {f_lake_deriv(x_clear):.6f}  (expected: {lam_clear:.6f})")
print(f"f'(x_sad)   = {f_lake_deriv(x_sad):.6f}  (expected: {lam_sad:.6f})")

# ============================================================
# Compute barrier ΔΦ
# ============================================================
DeltaPhi, _ = quad(lambda x: -f_lake(x), x_clear, x_sad)
print(f"\nΔΦ = {DeltaPhi:.8f}")
print(f"τ = 1/|λ_clear| = {tau_L:.6f}")

# ============================================================
# STEP 1.1: Find CV where D_exact = 200
# ============================================================
print("\n" + "=" * 70)
print("STEP 1.1: Find CV where D_exact = 200")
print("=" * 70)

def compute_D_exact(cv):
    """Compute exact MFPT-based D for a given CV."""
    sigma = cv * x_clear * np.sqrt(2 * abs(lam_clear))

    N_grid = 50000
    x_grid = np.linspace(0.001, x_sad + 0.001, N_grid)
    dx_g = x_grid[1] - x_grid[0]

    neg_f = np.array([-f_lake(x) for x in x_grid])
    U_raw = np.cumsum(neg_f) * dx_g
    i_clear = np.argmin(np.abs(x_grid - x_clear))
    U_grid = U_raw - U_raw[i_clear]

    Phi = 2 * U_grid / sigma**2

    exp_neg_Phi = np.exp(-Phi)
    I_x = np.cumsum(exp_neg_Phi) * dx_g

    psi = (2 / sigma**2) * np.exp(Phi) * I_x

    i_sad = np.argmin(np.abs(x_grid - x_sad))
    MFPT_exact = np.trapz(psi[i_clear:i_sad + 1], x_grid[i_clear:i_sad + 1])
    D_exact = MFPT_exact / tau_L

    return D_exact, sigma, MFPT_exact

# Scan CV from 0.30 to 0.36
print("\nCV scan:")
cv_values = np.arange(0.300, 0.365, 0.005)
for cv in cv_values:
    D_ex, sig, mfpt = compute_D_exact(cv)
    print(f"  CV = {cv:.3f}  σ = {sig:.6f}  D_exact = {D_ex:.1f}  MFPT = {mfpt:.1f}")

# Use bisection to find exact CV where D=200
print("\nBisection for D_exact = 200:")
def D_minus_200(cv):
    D, _, _ = compute_D_exact(cv)
    return D - 200

# From the scan, D=200 is between CV=0.33 and CV=0.35
try:
    cv_star = brentq(D_minus_200, 0.30, 0.36, xtol=1e-6)
    D_star, sigma_star, mfpt_star = compute_D_exact(cv_star)
    barrier_star = 2 * DeltaPhi / sigma_star**2

    print(f"  CV* = {cv_star:.6f}")
    print(f"  σ*  = {sigma_star:.6f}")
    print(f"  D_exact = {D_star:.2f}")
    print(f"  MFPT = {mfpt_star:.2f}")
    print(f"  2ΔΦ/σ² = {barrier_star:.6f}")
except Exception as e:
    print(f"  Bisection failed: {e}")
    # Fallback: use CV=0.34 as approximate
    cv_star = 0.34
    D_star, sigma_star, mfpt_star = compute_D_exact(cv_star)
    barrier_star = 2 * DeltaPhi / sigma_star**2
    print(f"  Using CV = {cv_star:.3f}")
    print(f"  σ  = {sigma_star:.6f}")
    print(f"  D_exact = {D_star:.2f}")

# ============================================================
# STEP 1.2: Kramers prefactor
# ============================================================
print("\n" + "=" * 70)
print("STEP 1.2: Kramers prefactor")
print("=" * 70)

C_std = np.sqrt(abs(lam_clear) * abs(lam_sad)) / (2 * np.pi)
print(f"|λ_clear| = {abs(lam_clear):.6f}")
print(f"|λ_sad|   = {abs(lam_sad):.6f}")
print(f"C_std = √(|λ_clear|×|λ_sad|)/(2π) = {C_std:.6f}")
print(f"1/(C_std × τ) = {1/(C_std * tau_L):.6f}")
print(f"ln(1/(C_std × τ)) = {np.log(1/(C_std * tau_L)):.6f}")

# Also: 1D Kramers formula (from KEY_EQUATIONS.md)
# 1/(C×τ) = |f'(x_sad)| × |f'(x_eq)| / (2π) ... wait, that's not right.
# Standard 1D Kramers: MFPT ≈ (2π/√(|f'(x_eq)| × |f'(x_sad)|)) × exp(barrier)
# So D_Kramers = MFPT/τ = exp(barrier)/(C_std × τ)
D_kramers_at_star = np.exp(barrier_star) / (C_std * tau_L)
print(f"\nAt CV* = {cv_star:.6f}:")
print(f"  D_Kramers_standard = exp({barrier_star:.6f}) / ({C_std:.6f} × {tau_L:.6f})")
print(f"  D_Kramers_standard = {D_kramers_at_star:.2f}")

# ============================================================
# STEP 1.3: K factor
# ============================================================
print("\n" + "=" * 70)
print("STEP 1.3: K factor (D_exact / D_Kramers_standard)")
print("=" * 70)

K_star = D_star / D_kramers_at_star
print(f"K = D_exact / D_Kramers_standard = {D_star:.2f} / {D_kramers_at_star:.2f} = {K_star:.6f}")

# Check K across multiple CV values
print("\nK across CV range:")
for cv in [0.30, 0.32, cv_star, 0.35, 0.36]:
    D_ex, sig, _ = compute_D_exact(cv)
    barrier_cv = 2 * DeltaPhi / sig**2
    D_kr = np.exp(barrier_cv) / (C_std * tau_L)
    K_cv = D_ex / D_kr
    print(f"  CV = {cv:.4f}  D_exact = {D_ex:.1f}  D_Kramers = {D_kr:.1f}  K = {K_cv:.4f}")

# ============================================================
# STEP 1.4: Verify the numerical identity
# ============================================================
print("\n" + "=" * 70)
print("STEP 1.4: Numerical identity verification")
print("=" * 70)
print("Identity: ln(D_product) = 2ΔΦ/σ² + ln(K) + ln(1/(C×τ))")

D_product = 200
LHS = np.log(D_product)
barrier_term = 2 * DeltaPhi / sigma_star**2
K_term = np.log(K_star)
prefactor_term = np.log(1 / (C_std * tau_L))
RHS = barrier_term + K_term + prefactor_term

print(f"\nLHS = ln(200) = {LHS:.6f}")
print(f"\nRHS components:")
print(f"  2ΔΦ/σ²     = {barrier_term:.6f}")
print(f"  ln(K)       = {K_term:.6f}")
print(f"  ln(1/(C×τ)) = {prefactor_term:.6f}")
print(f"  RHS total   = {RHS:.6f}")
print(f"\nLHS - RHS = {LHS - RHS:.6f}")
print(f"Match: {'YES' if abs(LHS - RHS) < 0.01 else 'NO'} (tolerance: 0.01)")

# Also express: what σ would make it exact?
# ln(200) = 2ΔΦ/σ² + ln(K) + ln(1/(C×τ))
# => 2ΔΦ/σ² = ln(200) - ln(K) - ln(1/(C×τ))
target_barrier = LHS - K_term - prefactor_term
sigma_identity = np.sqrt(2 * DeltaPhi / target_barrier)
cv_identity = sigma_identity / (x_clear * np.sqrt(2 * abs(lam_clear)))
print(f"\nFor exact identity: σ = {sigma_identity:.6f}, CV = {cv_identity:.6f}")
print(f"Compare to CV* (D=200): {cv_star:.6f}")

# ============================================================
# STEP 1.5: ε candidate definitions
# ============================================================
print("\n" + "=" * 70)
print("STEP 1.5: ε candidate definitions")
print("=" * 70)

# Evaluate terms at equilibria
loading_eq = a
flushing_eq = b * x_clear
recyc_eq = r * x_clear**q / (x_clear**q + h**q)

loading_sad = a
flushing_sad = b * x_sad
recyc_sad = r * x_sad**q / (x_sad**q + h**q)

loading_turb = a
flushing_turb = b * x_turb
recyc_turb = r * x_turb**q / (x_turb**q + h**q)

print("\nFlux terms at equilibria:")
print(f"  At x_clear = {x_clear:.6f}:")
print(f"    loading  = a = {loading_eq:.6f}")
print(f"    flushing = bx = {flushing_eq:.6f}")
print(f"    recycling = rx^q/(x^q+h^q) = {recyc_eq:.6f}")
print(f"    total flux = loading + recycling = {loading_eq + recyc_eq:.6f}")
print(f"  At x_sad = {x_sad:.6f}:")
print(f"    loading  = a = {loading_sad:.6f}")
print(f"    flushing = bx = {flushing_sad:.6f}")
print(f"    recycling = rx^q/(x^q+h^q) = {recyc_sad:.6f}")
print(f"    total flux = loading + recycling = {loading_sad + recyc_sad:.6f}")
print(f"  At x_turb = {x_turb:.6f}:")
print(f"    loading  = a = {loading_turb:.6f}")
print(f"    flushing = bx = {flushing_turb:.6f}")
print(f"    recycling = rx^q/(x^q+h^q) = {recyc_turb:.6f}")

# Candidate 1: recycling ratio
eps_cand1 = recyc_eq / recyc_sad
print(f"\nCandidate 1: recyc_eq/recyc_sad = {recyc_eq:.6f}/{recyc_sad:.6f} = {eps_cand1:.6f}")
print(f"  1/eps = {1/eps_cand1:.1f}")

# Candidate 2: flushing ratio
eps_cand2 = flushing_eq / flushing_sad
print(f"\nCandidate 2: flushing_eq/flushing_sad = {flushing_eq:.6f}/{flushing_sad:.6f} = {eps_cand2:.6f}")
print(f"  1/eps = {1/eps_cand2:.1f}")

# Candidate 3: (total stabilizing at eq)/(total destabilizing at saddle)
eps_cand3 = flushing_eq / (loading_sad + recyc_sad)
print(f"\nCandidate 3: flushing_eq/(loading_sad+recyc_sad) = {flushing_eq:.6f}/{loading_sad + recyc_sad:.6f} = {eps_cand3:.6f}")
print(f"  1/eps = {1/eps_cand3:.1f}")

# Candidate 4: eigenvalue ratio
eps_cand4 = abs(lam_clear) / abs(lam_sad)
print(f"\nCandidate 4: |λ_clear|/|λ_sad| = {abs(lam_clear):.6f}/{abs(lam_sad):.6f} = {eps_cand4:.6f}")
print(f"  1/eps = {1/eps_cand4:.1f}")

# Candidate 5: eigenvalue product
eps_cand5 = 1.0 / (abs(lam_clear) * abs(lam_sad))
print(f"\nCandidate 5: 1/(|λ_clear|×|λ_sad|) = 1/({abs(lam_clear)*abs(lam_sad):.6f}) = {eps_cand5:.6f}")
print(f"  1/eps = {1/eps_cand5:.1f}")

# Candidate 6: barrier-based (circular but diagnostic)
eps_cand6 = np.exp(-2 * DeltaPhi / sigma_star**2)
print(f"\nCandidate 6: exp(-2ΔΦ/σ²) = {eps_cand6:.6f}")
print(f"  1/eps = {1/eps_cand6:.1f}")

# Candidate 7: recycling fraction at equilibrium
eps_cand7 = recyc_eq / (loading_eq + recyc_eq)
print(f"\nCandidate 7: recyc_eq/(loading_eq+recyc_eq) = {eps_cand7:.6f}")
print(f"  1/eps = {1/eps_cand7:.1f}")

# Candidate 8: net stabilizing / destabilizing gain
# At clear eq: net restoring = |f'(x_clear)| = 0.785
# The recycling derivative (destabilizing part) vs the loss (stabilizing part)
recyc_deriv_eq = r * q * x_clear**(q-1) * h**q / (x_clear**q + h**q)**2
recyc_deriv_sad = r * q * x_sad**(q-1) * h**q / (x_sad**q + h**q)**2
eps_cand8 = recyc_deriv_eq / b  # fraction of loss rate contributed by recycling feedback
print(f"\nCandidate 8: recyc_deriv_eq/b = {recyc_deriv_eq:.6f}/{b:.6f} = {eps_cand8:.6f}")
print(f"  1/eps = {1/eps_cand8:.1f}")

# Candidate 9: x_clear/x_sad (state ratio)
eps_cand9 = x_clear / x_sad
print(f"\nCandidate 9: x_clear/x_sad = {x_clear:.6f}/{x_sad:.6f} = {eps_cand9:.6f}")
print(f"  1/eps = {1/eps_cand9:.1f}")

# Candidate 10: x_clear/x_turb
eps_cand10 = x_clear / x_turb
print(f"\nCandidate 10: x_clear/x_turb = {x_clear:.6f}/{x_turb:.6f} = {eps_cand10:.6f}")
print(f"  1/eps = {1/eps_cand10:.1f}")

# Candidate 11: (x_clear/x_sad)^2 -- squared state ratio
eps_cand11 = (x_clear / x_sad)**2
print(f"\nCandidate 11: (x_clear/x_sad)² = {eps_cand11:.6f}")
print(f"  1/eps = {1/eps_cand11:.1f}")

# Candidate 12: ΔΦ / (0.5 * x_sad^2 * |lam_sad|) -- barrier normalized by saddle curvature
eps_cand12 = DeltaPhi / (0.5 * x_sad**2 * abs(lam_sad))
print(f"\nCandidate 12: ΔΦ/(0.5×x_sad²×|λ_sad|) = {eps_cand12:.6f}")
print(f"  1/eps = {1/eps_cand12:.1f}")

# Now check: do any PAIRS multiply to give D ≈ 200?
print("\n" + "=" * 70)
print("PAIRWISE PRODUCTS giving D = (1/ε₁)(1/ε₂) ≈ 200:")
print("=" * 70)

candidates = {
    'cand1_recyc_ratio': eps_cand1,
    'cand2_flush_ratio': eps_cand2,
    'cand3_stab/destab': eps_cand3,
    'cand4_eigenval_ratio': eps_cand4,
    'cand7_recyc_frac': eps_cand7,
    'cand8_recyc_deriv/b': eps_cand8,
    'cand9_x_ratio': eps_cand9,
    'cand10_x_cl/turb': eps_cand10,
}

for name1, e1 in candidates.items():
    for name2, e2 in candidates.items():
        if name1 < name2:
            D_pair = (1/e1) * (1/e2)
            if 100 < D_pair < 400:
                print(f"  {name1} × {name2}: ε = ({e1:.4f}, {e2:.4f}), D = {D_pair:.1f}")

# Also check single candidates where 1/ε ≈ 200
print("\nSINGLE candidates with 1/ε ≈ 200:")
for name, eps in candidates.items():
    if 100 < 1/eps < 400:
        print(f"  {name}: ε = {eps:.6f}, 1/ε = {1/eps:.1f}")

# Summary table
print("\n" + "=" * 70)
print("SUMMARY TABLE: All ε candidates")
print("=" * 70)
print(f"{'Candidate':<40} {'ε value':<12} {'1/ε':<10} {'Target?':<15}")
print("-" * 77)
all_cands = [
    ("1: recyc_eq/recyc_sad", eps_cand1, "0.05 or 0.10"),
    ("2: flushing_eq/flushing_sad", eps_cand2, "0.05 or 0.10"),
    ("3: flushing_eq/(load+recyc)_sad", eps_cand3, "0.05 or 0.10"),
    ("4: |λ_clear|/|λ_sad|", eps_cand4, "0.05 or 0.10"),
    ("5: 1/(|λ_clear|×|λ_sad|)", eps_cand5, "0.05 or 0.10"),
    ("6: exp(-2ΔΦ/σ²) [circular]", eps_cand6, "gives D"),
    ("7: recyc_eq/(load+recyc)_eq", eps_cand7, "0.05 or 0.10"),
    ("8: d(recyc)/dx|_eq / b", eps_cand8, "0.05 or 0.10"),
    ("9: x_clear/x_sad", eps_cand9, "0.05 or 0.10"),
    ("10: x_clear/x_turb", eps_cand10, "0.05 or 0.10"),
    ("11: (x_clear/x_sad)²", eps_cand11, "0.05 or 0.10"),
    ("12: ΔΦ/(½x_sad²|λ_sad|)", eps_cand12, "0.05 or 0.10"),
]

for name, val, target in all_cands:
    near = ""
    if abs(val - 0.05) < 0.02:
        near = "≈ 0.05 ✓"
    elif abs(val - 0.10) < 0.03:
        near = "≈ 0.10 ✓"
    elif abs(val - 0.005) < 0.003:
        near = "≈ 0.005 (1/D)"
    print(f"{name:<40} {val:<12.6f} {1/val:<10.1f} {near:<15}")

print("\nDone with Phase 1.")
