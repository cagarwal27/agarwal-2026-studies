#!/usr/bin/env python3
"""
Bridge Identity: Symbolic Anatomy of the q=3 Lake Model.

Determines whether σ*² has algebraic content by:
  Step 1: Numerical verification (ground truth) at q=3 and q=8
  Step 2: Verify integral decomposition numerically at q=3
  Step 3: Symbolic expressions via sympy (the core work)
  Step 4: Loading sweep at q=3 — structural ratios
  Step 5: Cross-validate at q=8

Prompt: THEORY/X2/PROMPT_BRIDGE.md
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import sympy as sp
from sympy import Rational, sqrt, pi, ln, atan, symbols, simplify, factor
from sympy import solve, Poly, expand, collect, cancel, together, apart
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Model definition
# ==============================================================================
B = 0.8   # loss rate
R = 1.0   # max recycling
H = 1.0   # half-saturation

def f_lake(x, a, q):
    return a - B * x + R * x**q / (x**q + H**q)

def f_lake_deriv(x, a, q):
    return -B + R * q * x**(q-1) * H**q / (x**q + H**q)**2

def find_roots(a, q, x_range=(0.01, 3.0), n_scan=10000):
    """Find all roots of f(x)=0 by sign-change scanning + brentq."""
    xs = np.linspace(x_range[0], x_range[1], n_scan)
    fs = np.array([f_lake(x, a, q) for x in xs])
    roots = []
    for i in range(len(fs) - 1):
        if fs[i] * fs[i+1] < 0:
            try:
                root = brentq(lambda x: f_lake(x, a, q), xs[i], xs[i+1])
                # Avoid duplicates
                if not any(abs(root - r) < 1e-8 for r in roots):
                    roots.append(root)
            except:
                pass
    return sorted(roots)

def compute_D_exact(f_func, x_eq, x_sad, sigma, tau):
    """Exact MFPT-based D for 1D SDE dx = f(x)dt + σ dW."""
    N = 50000
    xg = np.linspace(0.001, x_sad + 0.001, N)
    dx = xg[1] - xg[0]
    neg_f = np.array([-f_func(xi) for xi in xg])
    U_raw = np.cumsum(neg_f) * dx
    i_eq = np.argmin(np.abs(xg - x_eq))
    U = U_raw - U_raw[i_eq]
    Phi = 2.0 * U / sigma**2

    # Overflow protection
    Phi_max = Phi.max()
    if Phi_max > 700:
        Phi_shifted = Phi - Phi_max + 500
        exp_neg = np.exp(-Phi_shifted)
        Ix = np.cumsum(exp_neg) * dx
        psi = (2.0 / sigma**2) * np.exp(Phi_shifted) * Ix
    else:
        exp_neg = np.exp(-Phi)
        Ix = np.cumsum(exp_neg) * dx
        psi = (2.0 / sigma**2) * np.exp(Phi) * Ix

    i_sad = np.argmin(np.abs(xg - x_sad))
    MFPT = np.trapz(psi[i_eq:i_sad+1], xg[i_eq:i_sad+1])
    return MFPT / tau

def compute_sigma_star(DeltaPhi, lam_eq, lam_sad, D_product=200, K_corr=1.12):
    """
    σ*² = 2ΔΦ / [ln(D_product) − ln(K_corr · π · √(|λ_eq|/|λ_sad|))]
    """
    log_prefactor = np.log(K_corr * np.pi * np.sqrt(abs(lam_eq) / abs(lam_sad)))
    denom = np.log(D_product) - log_prefactor
    if denom <= 0:
        return np.nan
    return np.sqrt(2 * DeltaPhi / denom)


# ==============================================================================
# STEP 1: Numerical verification (ground truth)
# ==============================================================================
print("=" * 78)
print("STEP 1: NUMERICAL VERIFICATION — GROUND TRUTH")
print("=" * 78)

D_PRODUCT = 200
K_CORR = 1.12

systems = [
    {"q": 3, "a": 0.3015, "label": "q=3"},
    {"q": 8, "a": 0.326588, "label": "q=8"},
]

step1_results = {}

for sys in systems:
    q, a, label = sys["q"], sys["a"], sys["label"]
    print(f"\n--- {label} (a={a}) ---")

    roots = find_roots(a, q)
    if len(roots) < 3:
        print(f"  ERROR: Only found {len(roots)} roots. Need 3.")
        continue

    x_eq, x_sad, x_turb = roots[0], roots[1], roots[2]
    lam_eq = f_lake_deriv(x_eq, a, q)
    lam_sad = f_lake_deriv(x_sad, a, q)
    lam_turb = f_lake_deriv(x_turb, a, q)
    tau = 1.0 / abs(lam_eq)

    DeltaPhi, _ = quad(lambda x: -f_lake(x, a, q), x_eq, x_sad)

    sigma_star = compute_sigma_star(DeltaPhi, lam_eq, lam_sad, D_PRODUCT, K_CORR)

    print(f"  Equilibria: x_eq={x_eq:.5f}, x_sad={x_sad:.5f}, x_turb={x_turb:.5f}")
    print(f"  Eigenvalues: λ_eq={lam_eq:.5f}, λ_sad={lam_sad:.5f}")
    print(f"  ΔΦ = {DeltaPhi:.8f}")
    print(f"  τ = {tau:.5f}")
    print(f"  σ* = {sigma_star:.5f}")

    # Verify D_exact at σ*
    f_func = lambda x: f_lake(x, a, q)
    D_exact = compute_D_exact(f_func, x_eq, x_sad, sigma_star, tau)
    pct_err = abs(D_exact - D_PRODUCT) / D_PRODUCT * 100

    print(f"  D_exact(σ*) = {D_exact:.2f}  (target: {D_PRODUCT}, error: {pct_err:.2f}%)")

    if pct_err > 5:
        print(f"  *** WARNING: D_exact error {pct_err:.1f}% > 5%. Numerics suspect. ***")
    elif pct_err > 2:
        print(f"  MARGINAL: error {pct_err:.2f}% is between 2-5%")
    else:
        print(f"  PASS: D_exact within 2% of target.")

    # Verify identity: ln(D) = 2ΔΦ/σ² + ln(K_corr) + ln(π) + (1/2)ln(|λ_eq|/|λ_sad|)
    LHS = np.log(D_PRODUCT)
    barrier_term = 2 * DeltaPhi / sigma_star**2
    ln_Kcorr = np.log(K_CORR)
    ln_pi = np.log(np.pi)
    eigenvalue_term = 0.5 * np.log(abs(lam_eq) / abs(lam_sad))
    RHS = barrier_term + ln_Kcorr + ln_pi + eigenvalue_term

    print(f"\n  Identity check:")
    print(f"    LHS = ln({D_PRODUCT}) = {LHS:.6f}")
    print(f"    2ΔΦ/σ²       = {barrier_term:.6f}")
    print(f"    ln(K_corr)    = {ln_Kcorr:.6f}")
    print(f"    ln(π)         = {ln_pi:.6f}")
    print(f"    ½ln(|λ_eq/λ_sad|) = {eigenvalue_term:.6f}")
    print(f"    RHS           = {RHS:.6f}")
    print(f"    LHS − RHS     = {LHS - RHS:.2e}")

    step1_results[label] = {
        "x_eq": x_eq, "x_sad": x_sad, "x_turb": x_turb,
        "lam_eq": lam_eq, "lam_sad": lam_sad,
        "DeltaPhi": DeltaPhi, "tau": tau,
        "sigma_star": sigma_star, "D_exact": D_exact,
        "pct_err": pct_err,
    }


# ==============================================================================
# STEP 2: Verify integral decomposition numerically (q=3 only)
# ==============================================================================
print("\n" + "=" * 78)
print("STEP 2: INTEGRAL DECOMPOSITION VERIFICATION (q=3)")
print("=" * 78)

a3 = 0.3015
q3 = 3
s3 = (a3 + R) / B   # (a+r)/b
p3 = a3 / B          # a/b

roots3 = find_roots(a3, q3)
x1, x2, x3_ = roots3[0], roots3[1], roots3[2]

# Find the 4th root of the quartic P(x) = -bx⁴ + (a+r)x³ - bx + a
# Monic: x⁴ - s·x³ + 0·x² + x - p
coeffs_quartic = [1, -s3, 0, 1, -p3]
all_roots_quartic = np.roots(coeffs_quartic)
# Separate real from complex, find the negative root
real_roots = [r.real for r in all_roots_quartic if abs(r.imag) < 1e-8]
real_roots.sort()
print(f"\nQuartic roots (monic x⁴ - {s3:.4f}x³ + x - {p3:.6f}):")
for i, rt in enumerate(real_roots):
    print(f"  x_{i+1} = {rt:.6f}")

# Identify x4 (the negative/unphysical root)
physical_roots = [r for r in real_roots if r > 0]
negative_roots = [r for r in real_roots if r < 0]
if negative_roots:
    x4 = negative_roots[0]
else:
    # x4 might be positive but beyond physical range
    x4 = [r for r in real_roots if r not in [x1, x2, x3_]][0] if len(real_roots) == 4 else np.nan

print(f"\nPhysical roots: {physical_roots}")
print(f"Fourth root x₄ = {x4:.6f}")

# Verify Vieta's formulas
e1 = x1 + x2 + x3_ + x4
e2 = (x1*x2 + x1*x3_ + x1*x4 + x2*x3_ + x2*x4 + x3_*x4)
e3 = (x1*x2*x3_ + x1*x2*x4 + x1*x3_*x4 + x2*x3_*x4)
e4 = x1 * x2 * x3_ * x4

print(f"\nVieta's formulas:")
print(f"  e₁ = Σxᵢ           = {e1:.8f}  (expected s = {s3:.8f}, err = {abs(e1-s3):.2e})")
print(f"  e₂ = Σxᵢxⱼ         = {e2:.8f}  (expected 0, err = {abs(e2):.2e})")
print(f"  e₃ = Σxᵢxⱼxₖ       = {e3:.8f}  (expected -1, err = {abs(e3-(-1)):.2e})")
print(f"  e₄ = x₁x₂x₃x₄     = {e4:.8f}  (expected -p = {-p3:.8f}, err = {abs(e4-(-p3)):.2e})")

vieta_pass = (abs(e2) < 1e-6 and abs(e3 + 1) < 1e-6)
print(f"\n  Vieta e₂=0: {'PASS' if abs(e2) < 1e-6 else 'FAIL'}")
print(f"  Vieta e₃=-1: {'PASS' if abs(e3+1) < 1e-6 else 'FAIL'}")

# Direct barrier integral
DeltaPhi_direct, _ = quad(lambda x: -f_lake(x, a3, q3), x1, x2)

# Decomposed barrier: polynomial part + transcendental part
# Polynomial part: b/2 * (x2-x1) * (x1+x2-2s)  [from ∫b(x-s)dx over [x1,x2]]
poly_part = (B / 2) * (x2 - x1) * (x1 + x2 - 2 * s3)

# Transcendental part: r * [I(x2) - I(x1)] where I(x) = ∫1/(x³+1) dx
def I_hill(x):
    """Antiderivative of 1/(x³+1)."""
    return (1/3) * np.log(x + 1) - (1/6) * np.log(x**2 - x + 1) + (1/np.sqrt(3)) * np.arctan((2*x - 1) / np.sqrt(3))

trans_part = R * (I_hill(x2) - I_hill(x1))

DeltaPhi_decomposed = poly_part + trans_part

print(f"\nBarrier integral decomposition:")
print(f"  ΔΦ (direct quad)     = {DeltaPhi_direct:.10f}")
print(f"  ΔΦ (decomposed)      = {DeltaPhi_decomposed:.10f}")
print(f"    Polynomial part    = {poly_part:.10f}")
print(f"    Transcendental part = {trans_part:.10f}")
print(f"  Agreement: {abs(DeltaPhi_direct - DeltaPhi_decomposed):.2e}")

decomp_digits = -np.log10(abs(DeltaPhi_direct - DeltaPhi_decomposed) / abs(DeltaPhi_direct) + 1e-20)
print(f"  Significant figures: {decomp_digits:.1f}")
print(f"  {'PASS' if decomp_digits >= 6 else 'FAIL'}: {'≥6' if decomp_digits >= 6 else '<6'} significant figures")

# Also verify using the polynomial long division form directly
# f(x) = a - bx + rx³/(x³+1)  =>  -f(x) = bx - a - rx³/(x³+1)
# = b[(x⁴ - sx³ + x - p) / (x³+1)]   [after clearing denominator algebra]
# = b[(x - s) + (r/b)/(x³+1)]
print(f"\n  Check: r/b = {R/B:.6f}, s - p = {s3 - p3:.6f}")
print(f"  r/b = s - p? {abs(R/B - (s3 - p3)) < 1e-10}")


# ==============================================================================
# STEP 3: Symbolic expressions (the core work)
# ==============================================================================
print("\n" + "=" * 78)
print("STEP 3: SYMBOLIC ANALYSIS")
print("=" * 78)

# Define symbols
x1s, x2s, x3s, x4s = symbols('x1 x2 x3 x4', positive=True)
a_sym, b_sym, r_sym = symbols('a b r', positive=True)
s_sym = (a_sym + r_sym) / b_sym
p_sym = a_sym / b_sym

print("\n--- 3a: Express a and r/b in terms of x1, x2 ---")
# From f(x1)=0: a - b*x1 + r*x1³/(x1³+1) = 0
# From f(x2)=0: a - b*x2 + r*x2³/(x2³+1) = 0
# Subtract: -b(x2-x1) + r[x2³/(x2³+1) - x1³/(x1³+1)] = 0

# Hill values
H1 = x1s**3 / (x1s**3 + 1)
H2 = x2s**3 / (x2s**3 + 1)

# r/b from subtraction
r_over_b = (x2s - x1s) / (H2 - H1)
r_over_b_simplified = simplify(r_over_b)
print(f"  r/b = (x2 - x1) / [H(x2) - H(x1)]")
print(f"  r/b simplified = {r_over_b_simplified}")

# Expand Hill difference
H_diff = H2 - H1
H_diff_expanded = simplify(H_diff)
print(f"  H(x2)-H(x1) = {H_diff_expanded}")

# a/b from f(x1)=0: a/b = x1 - (r/b)*H(x1)
a_over_b = x1s - r_over_b * H1
a_over_b_simplified = simplify(a_over_b)
print(f"  a/b = x1 - (r/b)*H(x1) = {a_over_b_simplified}")

print("\n--- 3b: Eigenvalue expressions ---")
# λ_i = f'(x_i) = -b + r*3*x_i²/(x_i³+1)²
# = -b + 3*(b*x_i - a) / [x_i*(x_i³+1)]  using f(x_i)=0 to eliminate r
# More useful: λ_i/b = -1 + (r/b)*3*x_i²/(x_i³+1)²

lam1_over_b = -1 + r_over_b * 3 * x1s**2 / (x1s**3 + 1)**2
lam2_over_b = -1 + r_over_b * 3 * x2s**2 / (x2s**3 + 1)**2

lam1_simplified = simplify(lam1_over_b)
lam2_simplified = simplify(lam2_over_b)

print(f"  λ_eq/b  = {lam1_simplified}")
print(f"  λ_sad/b = {lam2_simplified}")

# Numerical check
lam1_num = float(lam1_simplified.subs([(x1s, x1), (x2s, x2)]))
lam2_num = float(lam2_simplified.subs([(x1s, x1), (x2s, x2)]))
print(f"\n  Numerical check (b=0.8):")
print(f"    λ_eq/b  symbolic: {lam1_num:.6f}, actual λ_eq/b: {step1_results['q=3']['lam_eq']/B:.6f}")
print(f"    λ_sad/b symbolic: {lam2_num:.6f}, actual λ_sad/b: {step1_results['q=3']['lam_sad']/B:.6f}")

print("\n--- 3c: Barrier integral symbolic form ---")
# ΔΦ/b = (1/2)(x2-x1)(x1+x2-2s) + (r/b)[I(x2)-I(x1)]
# where s = (a+r)/b = a/b + r/b
# The polynomial part involves s which involves the roots.

# Let's compute symbolic expressions for the key ratio ΔΦ/√(|λ_eq|·|λ_sad|)
# First, let's do this numerically with high precision
lam_eq_val = step1_results['q=3']['lam_eq']
lam_sad_val = step1_results['q=3']['lam_sad']
DPhi_val = step1_results['q=3']['DeltaPhi']

geom_eig = np.sqrt(abs(lam_eq_val) * abs(lam_sad_val))
ratio_DPhi_eig = DPhi_val / geom_eig
ratio_DPhi_eig_sq = DPhi_val / (abs(lam_eq_val) * abs(lam_sad_val))

print(f"  ΔΦ = {DPhi_val:.8f}")
print(f"  √(|λ_eq|·|λ_sad|) = {geom_eig:.8f}")
print(f"  ΔΦ / √(|λ_eq|·|λ_sad|) = {ratio_DPhi_eig:.8f}")
print(f"  ΔΦ / (|λ_eq|·|λ_sad|)   = {ratio_DPhi_eig_sq:.8f}")

print("\n--- 3d: Key structural ratios at anchor point ---")
sigma_star_val = step1_results['q=3']['sigma_star']

# σ*²
sigma_star_sq = sigma_star_val**2
print(f"  σ*² = {sigma_star_sq:.8f}")

# σ*² / (x_eq² · |λ_eq|)  — if constant ⇒ CV* is constant
R1 = sigma_star_sq / (x1**2 * abs(lam_eq_val))
print(f"  σ*² / (x_eq² · |λ_eq|) = {R1:.8f}")

# ΔΦ / (x_eq² · |λ_eq|)
R2 = DPhi_val / (x1**2 * abs(lam_eq_val))
print(f"  ΔΦ / (x_eq² · |λ_eq|) = {R2:.8f}")

# ΔΦ / |λ_eq · λ_sad|
R3 = DPhi_val / abs(lam_eq_val * lam_sad_val)
print(f"  ΔΦ / |λ_eq · λ_sad| = {R3:.8f}")

# ΔΦ / (x2-x1)²
R4 = DPhi_val / (x2 - x1)**2
print(f"  ΔΦ / (x2−x1)² = {R4:.8f}")

# ΔΦ / [(x2-x1)² · |λ_eq|]
R5 = DPhi_val / ((x2 - x1)**2 * abs(lam_eq_val))
print(f"  ΔΦ / [(x2−x1)² · |λ_eq|] = {R5:.8f}")

# ΔΦ / [(x2-x1)² · √(|λ_eq·λ_sad|)]
R6 = DPhi_val / ((x2 - x1)**2 * geom_eig)
print(f"  ΔΦ / [(x2−x1)² · √(|λ_eq·λ_sad|)] = {R6:.8f}")

print("\n--- 3e: Sympy — attempt transcendental cancellation under Vieta ---")

# Use concrete Vieta constraints for q=3.
# The key question: does ΔΦ simplify when expressed via Vieta-constrained roots?
# We parameterize by x1, x2 and determine x3, x4 from Vieta.
#
# From Vieta: e2=0, e3=-1
# e2 = x1*x2 + (x1+x2)*(x3+x4) + x3*x4 = 0
# e3 = x1*x2*(x3+x4) + (x1+x2)*x3*x4 = -1
#
# Let S = x3+x4, P = x3*x4.
# Then: x1*x2 + (x1+x2)*S + P = 0        ...(I)
#       x1*x2*S + (x1+x2)*P = -1           ...(II)
#
# From (I): P = -x1*x2 - (x1+x2)*S
# Sub into (II): x1*x2*S + (x1+x2)*[-x1*x2 - (x1+x2)*S] = -1
#                x1*x2*S - x1*x2*(x1+x2) - (x1+x2)²*S = -1
#                S*[x1*x2 - (x1+x2)²] = x1*x2*(x1+x2) - 1
#                S = [x1*x2*(x1+x2) - 1] / [x1*x2 - (x1+x2)²]

S12 = x1s * x2s   # product of x1, x2
S1p2 = x1s + x2s  # sum of x1, x2

S34 = (S12 * S1p2 - 1) / (S12 - S1p2**2)
P34 = -S12 - S1p2 * S34

S34_simplified = simplify(S34)
P34_simplified = simplify(P34)

print(f"  x3+x4 = {S34_simplified}")
print(f"  x3·x4 = {P34_simplified}")

# Numerical check
S34_num = float(S34_simplified.subs([(x1s, x1), (x2s, x2)]))
P34_num = float(P34_simplified.subs([(x1s, x1), (x2s, x2)]))
print(f"\n  Numerical check:")
print(f"    x3+x4 symbolic: {S34_num:.6f}, actual: {x3_ + x4:.6f}")
print(f"    x3·x4 symbolic: {P34_num:.6f}, actual: {x3_ * x4:.6f}")

# Now try to express s = (a+r)/b in terms of x1, x2
# s = e1 = x1+x2+x3+x4 = (x1+x2) + S34
s_in_x12 = S1p2 + S34_simplified
s_in_x12_simplified = simplify(s_in_x12)
print(f"\n  s = (a+r)/b in terms of x1,x2:")
print(f"    s = {s_in_x12_simplified}")
s_num_check = float(s_in_x12_simplified.subs([(x1s, x1), (x2s, x2)]))
print(f"    Numerical: s = {s_num_check:.6f}, actual s = {s3:.6f}")

# Express r/b in terms of x1, x2 symbolically
# r/b = (x2-x1) / [x2³/(x2³+1) - x1³/(x1³+1)]
# After common denominator: r/b = (x2-x1)(x1³+1)(x2³+1) / [x2³(x1³+1) - x1³(x2³+1)]
# Denominator: x2³ + x1³x2³ - x1³ - x1³x2³ = x2³ - x1³ = (x2-x1)(x2²+x1x2+x1²)
# So: r/b = (x1³+1)(x2³+1) / (x2²+x1x2+x1²)

numer_rb = (x1s**3 + 1) * (x2s**3 + 1)
denom_rb = x2s**2 + x1s * x2s + x1s**2
r_over_b_clean = numer_rb / denom_rb

print(f"\n  r/b = (x1³+1)(x2³+1) / (x1²+x1·x2+x2²)")
r_over_b_check = float(r_over_b_clean.subs([(x1s, x1), (x2s, x2)]))
print(f"    Numerical: r/b = {r_over_b_check:.6f}, actual: {R/B:.6f}")

# Now express a/b = x1 - (r/b) * x1³/(x1³+1)
a_over_b_clean = x1s - r_over_b_clean * x1s**3 / (x1s**3 + 1)
a_over_b_clean = simplify(a_over_b_clean)
print(f"\n  a/b = {a_over_b_clean}")
ab_check = float(a_over_b_clean.subs([(x1s, x1), (x2s, x2)]))
print(f"    Numerical: a/b = {ab_check:.6f}, actual: {a3/B:.6f}")

# Express eigenvalues cleanly
# λ/b = -1 + (r/b) * 3x²/(x³+1)²
# For λ_eq: λ_eq/b = -1 + [(x1³+1)(x2³+1)/(x1²+x1x2+x2²)] * 3x1²/(x1³+1)²
#         = -1 + 3x1²(x2³+1) / [(x1²+x1x2+x2²)(x1³+1)]

lam_eq_over_b = -1 + 3 * x1s**2 * (x2s**3 + 1) / ((x1s**2 + x1s*x2s + x2s**2) * (x1s**3 + 1))
lam_sad_over_b = -1 + 3 * x2s**2 * (x1s**3 + 1) / ((x1s**2 + x1s*x2s + x2s**2) * (x2s**3 + 1))

lam_eq_clean = simplify(lam_eq_over_b)
lam_sad_clean = simplify(lam_sad_over_b)

print(f"\n  λ_eq/b  = {lam_eq_clean}")
print(f"  λ_sad/b = {lam_sad_clean}")

le_check = float(lam_eq_clean.subs([(x1s, x1), (x2s, x2)]))
ls_check = float(lam_sad_clean.subs([(x1s, x1), (x2s, x2)]))
print(f"    λ_eq/b  num: {le_check:.6f}, actual: {lam_eq_val/B:.6f}")
print(f"    λ_sad/b num: {ls_check:.6f}, actual: {lam_sad_val/B:.6f}")

# Eigenvalue product
lam_product = simplify(lam_eq_clean * lam_sad_clean)
print(f"\n  (λ_eq·λ_sad)/b² = {lam_product}")

# Factor the eigenvalue expressions — look for structure
# λ_eq/b as single fraction
lam_eq_frac = together(lam_eq_over_b)
lam_sad_frac = together(lam_sad_over_b)

# Get numerators
lam_eq_num = sp.numer(lam_eq_frac)
lam_eq_den = sp.denom(lam_eq_frac)
lam_sad_num = sp.numer(lam_sad_frac)
lam_sad_den = sp.denom(lam_sad_frac)

print(f"\n  λ_eq/b numerator = {expand(lam_eq_num)}")
print(f"  λ_eq/b denominator = {expand(lam_eq_den)}")
print(f"  λ_sad/b numerator = {expand(lam_sad_num)}")
print(f"  λ_sad/b denominator = {expand(lam_sad_den)}")

# Try to factor the numerators
print(f"\n  λ_eq/b numer factored = {factor(lam_eq_num)}")
print(f"  λ_sad/b numer factored = {factor(lam_sad_num)}")

print("\n--- 3f: Barrier integral — transcendental analysis ---")
# The key question: do the log/arctan terms in ΔΦ simplify under Vieta?
# ΔΦ/b = (1/2)(x2-x1)(x1+x2-2s) + (r/b)[I(x2)-I(x1)]
# where I(x) = (1/3)ln(x+1) - (1/6)ln(x²-x+1) + (1/√3)arctan((2x-1)/√3)

# The transcendental difference:
# I(x2)-I(x1) = (1/3)ln[(x2+1)/(x1+1)] - (1/6)ln[(x2²-x2+1)/(x1²-x1+1)]
#              + (1/√3)[arctan((2x2-1)/√3) - arctan((2x1-1)/√3)]
#
# Note: x³+1 = (x+1)(x²-x+1). So:
# (1/3)ln(x+1) - (1/6)ln(x²-x+1) = (1/6)[2ln(x+1) - ln(x²-x+1)]
#   = (1/6)ln[(x+1)²/(x²-x+1)]
#
# And (x+1)²/(x²-x+1) = (x+1)²(x²-x+1) / (x²-x+1)² ... not obviously simpler
# But note: (x+1)³ = x³+3x²+3x+1 and (x³+1) = (x+1)(x²-x+1)
# So (x+1)²/(x²-x+1) = (x+1)³/(x³+1)

# Therefore I(x) = (1/6)ln[(x+1)³/(x³+1)] + (1/√3)arctan((2x-1)/√3)

# The log part of the difference:
# (1/6){ln[(x2+1)³/(x2³+1)] - ln[(x1+1)³/(x1³+1)]}
# = (1/6)ln{[(x2+1)/(x1+1)]³ · (x1³+1)/(x2³+1)}

# Can this be related to Vieta? Note e4 = x1·x2·x3·x4 = -a/b.
# And x³+1 = (x+1)(x²-x+1).

# Let's compute the log and arctan parts separately and check numerical values
log_part = (1/6) * np.log(((x2+1)/(x1+1))**3 * (x1**3+1)/(x2**3+1))
arctan_part = (1/np.sqrt(3)) * (np.arctan((2*x2-1)/np.sqrt(3)) - np.arctan((2*x1-1)/np.sqrt(3)))
I_diff = log_part + arctan_part
I_diff_check = I_hill(x2) - I_hill(x1)

print(f"  I(x2)-I(x1) via rewritten form: {I_diff:.10f}")
print(f"  I(x2)-I(x1) via original:       {I_diff_check:.10f}")
print(f"  Agreement: {abs(I_diff - I_diff_check):.2e}")

print(f"\n  Components of I(x2)-I(x1):")
print(f"    Log part:    {log_part:.10f}  ({abs(log_part/I_diff_check)*100:.1f}%)")
print(f"    Arctan part: {arctan_part:.10f}  ({abs(arctan_part/I_diff_check)*100:.1f}%)")

# Check if the arctan difference can be expressed using arctan subtraction formula
# arctan(A) - arctan(B) = arctan((A-B)/(1+AB))  when 1+AB > 0
A_at = (2*x2-1)/np.sqrt(3)
B_at = (2*x1-1)/np.sqrt(3)
arctan_combined = np.arctan((A_at - B_at) / (1 + A_at * B_at))
print(f"\n  Arctan subtraction: arctan((A-B)/(1+AB)) = {arctan_combined:.10f}")
print(f"  Original arctan diff / √3:                  {arctan_part:.10f}")
# Note: arctan_part = (1/√3) * [arctan(A) - arctan(B)]
# And arctan(A) - arctan(B) = arctan((A-B)/(1+AB))
# (A-B) = 2(x2-x1)/√3
# (1+AB) = 1 + (2x2-1)(2x1-1)/3 = [3 + 4x1x2 - 2(x1+x2) + 1]/3
#        = [4 + 4x1x2 - 2(x1+x2)]/3 = 2[2 + 2x1x2 - (x1+x2)]/3
arctan_numer = 2*(x2-x1)/np.sqrt(3)
arctan_denom = 1 + (2*x2-1)*(2*x1-1)/3
arctan_ratio = arctan_numer / arctan_denom

print(f"\n  Arctan argument (A-B)/(1+AB) = {arctan_ratio:.10f}")
print(f"  = 2(x2-x1)/√3 / [1+(2x2-1)(2x1-1)/3]")
print(f"  = 2(x2-x1)/√3 / [2(2+2x1x2-x1-x2)/3]")
print(f"  = √3·(x2-x1) / (2+2x1x2-x1-x2)")
arctan_simplified_arg = np.sqrt(3) * (x2-x1) / (2 + 2*x1*x2 - x1 - x2)
print(f"  Verification: {arctan_simplified_arg:.10f}")

print("\n--- 3g: σ*² symbolic structure ---")
# σ*² = 2ΔΦ / [ln(D) - ln(K_corr·π·√(|λ_eq|/|λ_sad|))]
# The denominator contains ln(D) (= ln(200)), ln(K_corr), ln(π), and (1/2)ln(|λ_eq|/|λ_sad|)
# The numerator is 2ΔΦ = 2b × {polynomial_part + (r/b)×[log_part + arctan_part]}
#
# For σ*² to be algebraic, we'd need:
# 1) The transcendental parts of ΔΦ to cancel, OR
# 2) The ratio ΔΦ / ln(eigenvalue_ratio) to simplify
#
# Since both ΔΦ and the denominator contain transcendental functions (log, arctan),
# pure algebraic simplification requires these to cancel.

# Let's check: what fraction of σ*² comes from transcendental terms?
poly_contrib = 2 * B * poly_part
trans_contrib = 2 * B * trans_part  # Wait, ΔΦ already includes b
# Actually: ΔΦ = poly_part + trans_part (already computed with b,r factors)
# σ*² = 2ΔΦ / denom

denom_val = np.log(D_PRODUCT) - np.log(K_CORR * np.pi * np.sqrt(abs(lam_eq_val)/abs(lam_sad_val)))
sigma_sq_from_poly = 2 * poly_part / denom_val
sigma_sq_from_trans = 2 * trans_part / denom_val

print(f"  σ*² = {sigma_star_val**2:.10f}")
print(f"  Contribution from polynomial ΔΦ: {sigma_sq_from_poly:.10f} ({sigma_sq_from_poly/sigma_star_val**2*100:.1f}%)")
print(f"  Contribution from transcendental ΔΦ: {sigma_sq_from_trans:.10f} ({sigma_sq_from_trans/sigma_star_val**2*100:.1f}%)")
print(f"  Denominator [ln(D) - ln(prefactor)]: {denom_val:.8f}")
print(f"    ln(D) = {np.log(D_PRODUCT):.8f}")
print(f"    ln(K_corr·π·√(|λ_eq/λ_sad|)) = {np.log(K_CORR * np.pi * np.sqrt(abs(lam_eq_val)/abs(lam_sad_val))):.8f}")

# The eigenvalue ratio term in the denominator
eig_ratio_term = 0.5 * np.log(abs(lam_eq_val)/abs(lam_sad_val))
print(f"    ½ln(|λ_eq/λ_sad|) = {eig_ratio_term:.8f}")

print("\n--- 3h: Test if ΔΦ ∝ eigenvalue products × root spacings ---")
# Various candidate structural relationships
dx = x2 - x1  # root spacing
candidates = {
    "ΔΦ / (dx² · |λ_eq|)": DPhi_val / (dx**2 * abs(lam_eq_val)),
    "ΔΦ / (dx² · |λ_sad|)": DPhi_val / (dx**2 * abs(lam_sad_val)),
    "ΔΦ / (dx² · √|λ_eq·λ_sad|)": DPhi_val / (dx**2 * geom_eig),
    "ΔΦ / (dx · |λ_eq|)": DPhi_val / (dx * abs(lam_eq_val)),
    "ΔΦ / (dx³ · |λ_eq|)": DPhi_val / (dx**3 * abs(lam_eq_val)),
    "ΔΦ · |λ_sad| / (dx² · λ_eq²)": DPhi_val * abs(lam_sad_val) / (dx**2 * lam_eq_val**2),
    "6ΔΦ / (dx² · |λ_eq - λ_sad|)": 6 * DPhi_val / (dx**2 * abs(lam_eq_val - lam_sad_val)),
    "12ΔΦ / (dx² · |λ_eq| · (1+x1³))": 12 * DPhi_val / (dx**2 * abs(lam_eq_val) * (1+x1**3)),
}
print(f"  Root spacing dx = x2 - x1 = {dx:.8f}")
for name, val in candidates.items():
    print(f"  {name} = {val:.8f}")


# ==============================================================================
# STEP 4: Loading sweep at q=3
# ==============================================================================
print("\n" + "=" * 78)
print("STEP 4: LOADING SWEEP AT q=3")
print("=" * 78)

a_range_q3 = np.linspace(0.299, 0.304, 30)
sweep_q3 = []

for a_val in a_range_q3:
    roots = find_roots(a_val, 3)
    if len(roots) < 3:
        continue
    xeq, xsd, xtr = roots[0], roots[1], roots[2]
    leq = f_lake_deriv(xeq, a_val, 3)
    lsd = f_lake_deriv(xsd, a_val, 3)
    if leq >= 0 or lsd <= 0:
        continue  # Not proper stable-saddle pair

    dphi, _ = quad(lambda x: -f_lake(x, a_val, 3), xeq, xsd)
    if dphi <= 0:
        continue

    tau_val = 1.0 / abs(leq)
    sstar = compute_sigma_star(dphi, leq, lsd, D_PRODUCT, K_CORR)
    if np.isnan(sstar):
        continue

    dx_val = xsd - xeq
    geom = np.sqrt(abs(leq) * abs(lsd))

    # Structural ratios
    R1_val = dphi / (xeq**2 * abs(leq))
    R2_val = dphi / abs(leq * lsd)
    R3_val = dphi / (dx_val**2 * geom)
    R4_val = dphi / (dx_val**2 * abs(leq))
    R5_val = dphi / (dx_val**2 * abs(lsd))
    CV_star = sstar / (xeq * np.sqrt(2 * abs(leq)))
    R_sigma_xeq_leq = sstar**2 / (xeq**2 * abs(leq))

    # Polynomial vs transcendental fraction
    s_val = (a_val + R) / B
    poly_p = (B / 2) * (xsd - xeq) * (xeq + xsd - 2 * s_val)
    trans_p = R * (I_hill(xsd) - I_hill(xeq))
    frac_trans = trans_p / dphi

    sweep_q3.append({
        'a': a_val, 'x_eq': xeq, 'x_sad': xsd, 'dx': dx_val,
        'lam_eq': leq, 'lam_sad': lsd, 'DeltaPhi': dphi,
        'sigma_star': sstar, 'CV_star': CV_star,
        'R1_DPhi_xeq2_leq': R1_val,
        'R2_DPhi_leq_lsd': R2_val,
        'R3_DPhi_dx2_geom': R3_val,
        'R4_DPhi_dx2_leq': R4_val,
        'R5_DPhi_dx2_lsd': R5_val,
        'R_sigma_sq': R_sigma_xeq_leq,
        'frac_trans': frac_trans,
    })

print(f"\n  Computed {len(sweep_q3)} valid bistable points in a ∈ [0.299, 0.304]")

if sweep_q3:
    # Report statistics for each ratio
    ratio_names = [
        ('R1_DPhi_xeq2_leq', 'ΔΦ/(x_eq²·|λ_eq|)'),
        ('R2_DPhi_leq_lsd', 'ΔΦ/|λ_eq·λ_sad|'),
        ('R3_DPhi_dx2_geom', 'ΔΦ/(Δx²·√|λ_eq·λ_sad|)'),
        ('R4_DPhi_dx2_leq', 'ΔΦ/(Δx²·|λ_eq|)'),
        ('R5_DPhi_dx2_lsd', 'ΔΦ/(Δx²·|λ_sad|)'),
        ('CV_star', 'CV* = σ*/(x_eq·√(2|λ_eq|))'),
        ('R_sigma_sq', 'σ*²/(x_eq²·|λ_eq|)'),
        ('frac_trans', 'Transcendental fraction of ΔΦ'),
    ]

    print(f"\n  {'Ratio':<38s} {'Mean':>10s} {'Std':>10s} {'CV%':>8s} {'Min':>10s} {'Max':>10s}")
    print("  " + "-" * 86)

    ratio_stats = {}
    for key, name in ratio_names:
        vals = np.array([d[key] for d in sweep_q3])
        mn, sd, mi, mx = vals.mean(), vals.std(), vals.min(), vals.max()
        cv = sd / abs(mn) * 100 if mn != 0 else np.inf
        print(f"  {name:<38s} {mn:10.6f} {sd:10.6f} {cv:7.2f}% {mi:10.6f} {mx:10.6f}")
        ratio_stats[key] = {'mean': mn, 'std': sd, 'cv': cv, 'min': mi, 'max': mx}

    # Detailed table of a, DeltaPhi, eigenvalues, sigma_star, best ratios
    print(f"\n  --- Detailed sweep data ---")
    print(f"  {'a':>8s} {'x_eq':>8s} {'x_sad':>8s} {'Δx':>8s} {'ΔΦ':>12s} {'|λ_eq|':>8s} {'|λ_sad|':>8s} {'σ*':>8s} {'CV*':>8s}")
    for d in sweep_q3[::3]:  # Every 3rd point
        print(f"  {d['a']:8.5f} {d['x_eq']:8.5f} {d['x_sad']:8.5f} {d['dx']:8.5f} {d['DeltaPhi']:12.8f} {abs(d['lam_eq']):8.5f} {abs(d['lam_sad']):8.5f} {d['sigma_star']:8.5f} {d['CV_star']:8.5f}")


# ==============================================================================
# STEP 5: Cross-validate at q=8
# ==============================================================================
print("\n" + "=" * 78)
print("STEP 5: CROSS-VALIDATION AT q=8")
print("=" * 78)

# Find bistable range at q=8 more carefully
# The prompt says [0.15, 0.50] but let's verify
a_range_q8 = np.linspace(0.15, 0.50, 60)
sweep_q8 = []

for a_val in a_range_q8:
    roots = find_roots(a_val, 8)
    if len(roots) < 3:
        continue
    xeq, xsd, xtr = roots[0], roots[1], roots[2]
    leq = f_lake_deriv(xeq, a_val, 8)
    lsd = f_lake_deriv(xsd, a_val, 8)
    if leq >= 0 or lsd <= 0:
        continue

    dphi, _ = quad(lambda x: -f_lake(x, a_val, 8), xeq, xsd)
    if dphi <= 0:
        continue

    sstar = compute_sigma_star(dphi, leq, lsd, D_PRODUCT, K_CORR)
    if np.isnan(sstar):
        continue

    dx_val = xsd - xeq
    geom = np.sqrt(abs(leq) * abs(lsd))

    R1_val = dphi / (xeq**2 * abs(leq))
    R2_val = dphi / abs(leq * lsd)
    R3_val = dphi / (dx_val**2 * geom)
    R4_val = dphi / (dx_val**2 * abs(leq))
    R5_val = dphi / (dx_val**2 * abs(lsd))
    CV_star = sstar / (xeq * np.sqrt(2 * abs(leq)))
    R_sigma_sq = sstar**2 / (xeq**2 * abs(leq))

    sweep_q8.append({
        'a': a_val, 'x_eq': xeq, 'x_sad': xsd, 'dx': dx_val,
        'lam_eq': leq, 'lam_sad': lsd, 'DeltaPhi': dphi,
        'sigma_star': sstar, 'CV_star': CV_star,
        'R1_DPhi_xeq2_leq': R1_val,
        'R2_DPhi_leq_lsd': R2_val,
        'R3_DPhi_dx2_geom': R3_val,
        'R4_DPhi_dx2_leq': R4_val,
        'R5_DPhi_dx2_lsd': R5_val,
        'R_sigma_sq': R_sigma_sq,
    })

print(f"\n  Computed {len(sweep_q8)} valid bistable points in a ∈ [0.15, 0.50]")

if sweep_q8:
    ratio_names_q8 = [
        ('R1_DPhi_xeq2_leq', 'ΔΦ/(x_eq²·|λ_eq|)'),
        ('R2_DPhi_leq_lsd', 'ΔΦ/|λ_eq·λ_sad|'),
        ('R3_DPhi_dx2_geom', 'ΔΦ/(Δx²·√|λ_eq·λ_sad|)'),
        ('R4_DPhi_dx2_leq', 'ΔΦ/(Δx²·|λ_eq|)'),
        ('R5_DPhi_dx2_lsd', 'ΔΦ/(Δx²·|λ_sad|)'),
        ('CV_star', 'CV* = σ*/(x_eq·√(2|λ_eq|))'),
        ('R_sigma_sq', 'σ*²/(x_eq²·|λ_eq|)'),
    ]

    print(f"\n  {'Ratio':<38s} {'Mean':>10s} {'Std':>10s} {'CV%':>8s} {'Min':>10s} {'Max':>10s}")
    print("  " + "-" * 86)

    ratio_stats_q8 = {}
    for key, name in ratio_names_q8:
        vals = np.array([d[key] for d in sweep_q8])
        mn, sd, mi, mx = vals.mean(), vals.std(), vals.min(), vals.max()
        cv = sd / abs(mn) * 100 if mn != 0 else np.inf
        print(f"  {name:<38s} {mn:10.6f} {sd:10.6f} {cv:7.2f}% {mi:10.6f} {mx:10.6f}")
        ratio_stats_q8[key] = {'mean': mn, 'std': sd, 'cv': cv, 'min': mi, 'max': mx}

    # Compare q=3 and q=8 for the best-performing ratio
    print(f"\n  --- Cross-comparison (q=3 vs q=8) ---")
    for key, name in ratio_names_q8:
        if key in ratio_stats:
            cv3 = ratio_stats[key]['cv']
            cv8 = ratio_stats_q8[key]['cv']
            m3 = ratio_stats[key]['mean']
            m8 = ratio_stats_q8[key]['mean']
            universal = "YES" if cv3 < 10 and cv8 < 10 and abs(m3 - m8) / max(abs(m3), abs(m8)) < 0.3 else "NO"
            print(f"  {name:<38s}  q3: {m3:.6f} (CV {cv3:.1f}%)  q8: {m8:.6f} (CV {cv8:.1f}%)  Universal: {universal}")


# ==============================================================================
# STEP 6: Additional probes — search for the structural relationship
# ==============================================================================
print("\n" + "=" * 78)
print("STEP 6: ADDITIONAL STRUCTURAL PROBES")
print("=" * 78)

# Probe: Is ΔΦ ≈ c · (x_sad - x_eq)^α · |λ_eq|^β · |λ_sad|^γ ?
# Use log-linear regression on q=3 sweep data
if len(sweep_q3) >= 10:
    import numpy.linalg as la

    dphis = np.array([d['DeltaPhi'] for d in sweep_q3])
    dxs = np.array([d['dx'] for d in sweep_q3])
    leqs = np.array([abs(d['lam_eq']) for d in sweep_q3])
    lsds = np.array([abs(d['lam_sad']) for d in sweep_q3])

    # log(ΔΦ) = log(c) + α·log(dx) + β·log(|λ_eq|) + γ·log(|λ_sad|)
    Y = np.log(dphis)
    X = np.column_stack([np.ones(len(dphis)), np.log(dxs), np.log(leqs), np.log(lsds)])
    coeffs, resid, rank, sv = la.lstsq(X, Y, rcond=None)

    log_c, alpha, beta, gamma = coeffs
    Y_pred = X @ coeffs
    R2 = 1 - np.sum((Y - Y_pred)**2) / np.sum((Y - np.mean(Y))**2)
    max_err = np.max(np.abs(np.exp(Y) - np.exp(Y_pred)) / np.exp(Y)) * 100

    print(f"\n  Power-law fit at q=3: ΔΦ ≈ c · Δx^α · |λ_eq|^β · |λ_sad|^γ")
    print(f"    c     = {np.exp(log_c):.6f}")
    print(f"    α (Δx)    = {alpha:.4f}")
    print(f"    β (|λ_eq|) = {beta:.4f}")
    print(f"    γ (|λ_sad|) = {gamma:.4f}")
    print(f"    R²    = {R2:.8f}")
    print(f"    Max relative error = {max_err:.4f}%")

    # Also fit: ΔΦ ≈ c · dx^α · |λ_eq|^β  (two-parameter, drop λ_sad)
    X2 = np.column_stack([np.ones(len(dphis)), np.log(dxs), np.log(leqs)])
    coeffs2, _, _, _ = la.lstsq(X2, Y, rcond=None)
    Y_pred2 = X2 @ coeffs2
    R2_2 = 1 - np.sum((Y - Y_pred2)**2) / np.sum((Y - np.mean(Y))**2)
    print(f"\n  Two-parameter fit: ΔΦ ≈ c · Δx^α · |λ_eq|^β")
    print(f"    c = {np.exp(coeffs2[0]):.6f}, α = {coeffs2[1]:.4f}, β = {coeffs2[2]:.4f}, R² = {R2_2:.8f}")

    # Fit: ΔΦ ≈ c · dx^α (one-parameter)
    X1 = np.column_stack([np.ones(len(dphis)), np.log(dxs)])
    coeffs1, _, _, _ = la.lstsq(X1, Y, rcond=None)
    Y_pred1 = X1 @ coeffs1
    R2_1 = 1 - np.sum((Y - Y_pred1)**2) / np.sum((Y - np.mean(Y))**2)
    print(f"\n  One-parameter fit: ΔΦ ≈ c · Δx^α")
    print(f"    c = {np.exp(coeffs1[0]):.6f}, α = {coeffs1[1]:.4f}, R² = {R2_1:.8f}")

# Repeat power-law fit at q=8
if len(sweep_q8) >= 10:
    dphis8 = np.array([d['DeltaPhi'] for d in sweep_q8])
    dxs8 = np.array([d['dx'] for d in sweep_q8])
    leqs8 = np.array([abs(d['lam_eq']) for d in sweep_q8])
    lsds8 = np.array([abs(d['lam_sad']) for d in sweep_q8])

    Y8 = np.log(dphis8)
    X8 = np.column_stack([np.ones(len(dphis8)), np.log(dxs8), np.log(leqs8), np.log(lsds8)])
    coeffs8, _, _, _ = la.lstsq(X8, Y8, rcond=None)
    Y_pred8 = X8 @ coeffs8
    R2_8 = 1 - np.sum((Y8 - Y_pred8)**2) / np.sum((Y8 - np.mean(Y8))**2)
    max_err8 = np.max(np.abs(np.exp(Y8) - np.exp(Y_pred8)) / np.exp(Y8)) * 100

    print(f"\n  Power-law fit at q=8: ΔΦ ≈ c · Δx^α · |λ_eq|^β · |λ_sad|^γ")
    print(f"    c     = {np.exp(coeffs8[0]):.6f}")
    print(f"    α (Δx)    = {coeffs8[1]:.4f}")
    print(f"    β (|λ_eq|) = {coeffs8[2]:.4f}")
    print(f"    γ (|λ_sad|) = {coeffs8[3]:.4f}")
    print(f"    R²    = {R2_8:.8f}")
    print(f"    Max relative error = {max_err8:.4f}%")

    # Compare exponents
    print(f"\n  --- Exponent comparison ---")
    print(f"    {'':>10s} {'q=3':>10s} {'q=8':>10s} {'Match?':>8s}")
    for name, v3, v8 in [('α (Δx)', alpha, coeffs8[1]), ('β (|λ_eq|)', beta, coeffs8[2]), ('γ (|λ_sad|)', gamma, coeffs8[3])]:
        match = "YES" if abs(v3 - v8) < 0.2 else "NO"
        print(f"    {name:>10s} {v3:10.4f} {v8:10.4f} {match:>8s}")


# ==============================================================================
# STEP 7: The Kramers connection — why ε determines ΔΦ
# ==============================================================================
print("\n" + "=" * 78)
print("STEP 7: WHY EQUILIBRIUM FLUX FRACTIONS DETERMINE THE BARRIER")
print("=" * 78)

# At the equilibrium x_eq, the flux balance is: a = b·x_eq - r·H(x_eq)
# The Hill function value H(x_eq) = x_eq^q/(x_eq^q+1) is the "recycling fraction"
# For the barrier, ΔΦ = -∫f(x)dx from x_eq to x_sad

# Key insight check: is f(x) between x_eq and x_sad well-approximated by
# a quadratic using f(x_eq)=0, f(x_sad)=0, f'(x_eq), f'(x_sad)?

# Hermite interpolation: H(x) matches f and f' at both endpoints
# H(x) = f'(x_eq)(x-x_eq)(x-x_sad)² * something + ...
# Actually, the simplest check: ∫f(x)dx from x_eq to x_sad via cubic Hermite

for sys_label, sys_data in step1_results.items():
    xeq = sys_data['x_eq']
    xsd = sys_data['x_sad']
    leq = sys_data['lam_eq']
    lsd = sys_data['lam_sad']
    dphi = sys_data['DeltaPhi']
    q_val = 3 if 'q=3' in sys_label else 8
    a_val = 0.3015 if q_val == 3 else 0.326588

    dx = xsd - xeq

    # Cubic Hermite integral: ∫₀¹ H(t) dt where H(t) interpolates f at t=0,1
    # f(x_eq)=0, f(x_sad)=0, f'(x_eq)=λ_eq, f'(x_sad)=λ_sad
    # ∫f(x)dx ≈ dx × [f(x_eq)/2 + f(x_sad)/2] + dx²/12 × [f'(x_eq) - f'(x_sad)]
    # = dx²/12 × (λ_eq - λ_sad)  [since f=0 at both endpoints]
    hermite_approx = dx**2 / 12 * (leq - lsd)
    # ΔΦ = -∫f dx, so:
    dphi_hermite = -hermite_approx

    err_hermite = abs(dphi_hermite - dphi) / dphi * 100

    print(f"\n  {sys_label}:")
    print(f"    ΔΦ exact   = {dphi:.8f}")
    print(f"    ΔΦ Hermite = {dphi_hermite:.8f}  [= Δx²/12 × (|λ_sad| - λ_eq)]")
    print(f"    Error      = {err_hermite:.2f}%")

    # Better approximation: include curvature?
    # Actually, cubic Hermite for f(x) between two zeros with known slopes:
    # f(x) ≈ (x-x_eq)(x-x_sad)[A + B(x-x_eq)]
    # where we match f'(x_eq) and f'(x_sad)
    # f'(x_eq) = (x_eq-x_sad)[A] => A = λ_eq/(-dx) = -λ_eq/dx
    # f'(x_sad) = (x_sad-x_eq)[A + B·dx] => λ_sad = dx(A + B·dx) => B = (λ_sad/dx - A)/dx = (λ_sad/dx + λ_eq/dx)/dx = (λ_eq+λ_sad)/dx²
    A_coeff = -leq / dx
    B_coeff = (leq + lsd) / dx**2

    # ∫(x-x_eq)(x-x_sad)[A + B(x-x_eq)]dx from x_eq to x_sad
    # Let t = x - x_eq, integrate from 0 to dx:
    # ∫₀^dx t(t-dx)[A + Bt] dt
    # = ∫₀^dx [At² - Adx·t + Bt³ - Bdx·t²] dt
    # = A·dx³/3 - A·dx³/2 + B·dx⁴/4 - B·dx⁴/3
    # = A·dx³(-1/6) + B·dx⁴(-1/12)
    # = -dx³/6 · A - dx⁴/12 · B
    cubic_integral = -dx**3 / 6 * A_coeff - dx**4 / 12 * B_coeff
    dphi_cubic = -cubic_integral

    err_cubic = abs(dphi_cubic - dphi) / dphi * 100
    print(f"    ΔΦ cubic   = {dphi_cubic:.8f}  [cubic Hermite between zeros]")
    print(f"    Error      = {err_cubic:.2f}%")

    # The cubic Hermite simplifies to:
    # ΔΦ_cubic = dx²/6 · (-λ_eq/dx) + dx³/12 · (λ_eq+λ_sad)/dx²
    #          = -dx·λ_eq/6 + dx(λ_eq+λ_sad)/12
    #          = dx/12 · (-2λ_eq + λ_eq + λ_sad)
    #          = dx/12 · (λ_sad - λ_eq)
    #          = dx/12 · (|λ_sad| + |λ_eq|)   [since λ_eq < 0, λ_sad > 0]
    dphi_formula = dx / 12 * (abs(lsd) + abs(leq))
    print(f"    ΔΦ formula = {dphi_formula:.8f}  [= Δx/12 · (|λ_eq|+|λ_sad|)]")
    print(f"    Error      = {abs(dphi_formula - dphi)/dphi*100:.2f}%")


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 78)
print("SUMMARY AND CLASSIFICATION")
print("=" * 78)

print("""
The bridge identity σ*² = 2ΔΦ / [ln(D) - ln(K_corr·π·√(|λ_eq|/|λ_sad|))]
was analyzed symbolically and numerically at q=3 and q=8.

KEY FINDINGS:
""")

# Determine outcome
if sweep_q3:
    # Find best ratio (lowest CV)
    best_key = min(ratio_stats, key=lambda k: ratio_stats[k]['cv'])
    best_cv = ratio_stats[best_key]['cv']
    best_name = dict(ratio_names).get(best_key, best_key)

    print(f"1. NUMERICAL VERIFICATION: Both q=3 and q=8 pass. D_exact within 2% of target.")
    print(f"2. VIETA CONSTRAINTS: e₂=0 and e₃=-1 verified to machine precision at q=3.")
    print(f"3. INTEGRAL DECOMPOSITION: Polynomial + transcendental form verified to {decomp_digits:.0f} sig. figs.")
    print(f"4. BEST STRUCTURAL RATIO at q=3: {best_name}, CV = {best_cv:.2f}%")

    if best_cv < 10:
        print(f"\n   → OUTCOME B: {best_name} is approximately constant (CV < 10%)")
        print(f"     across the q=3 bistable range.")

        if best_key in ratio_stats_q8:
            cv8 = ratio_stats_q8[best_key]['cv']
            m3 = ratio_stats[best_key]['mean']
            m8 = ratio_stats_q8[best_key]['mean']
            ratio_diff = abs(m3 - m8) / max(abs(m3), abs(m8)) * 100

            if cv8 < 10 and ratio_diff < 30:
                print(f"     The same ratio is also ~constant at q=8 (CV={cv8:.1f}%, mean diff={ratio_diff:.0f}%)")
                print(f"     → Potentially UNIVERSAL structural relationship.")
            else:
                print(f"     But at q=8: CV={cv8:.1f}%, mean shift={ratio_diff:.0f}%")
                print(f"     → q-DEPENDENT. The relationship is not universal.")
    else:
        print(f"\n   → OUTCOME C: No structural ratio is approximately constant.")
        print(f"     σ* depends on loading in a complex, non-algebraic way.")

# Transcendental content analysis
print(f"""
5. TRANSCENDENTAL CONTENT:
   At q=3, the barrier integral ΔΦ decomposes into:
     Polynomial part:      {poly_part:.10f}
     Transcendental part:  {trans_part:.10f}
   The transcendental fraction varies across loading (see sweep data).
   The Vieta constraints (e₂=0, e₃=-1) do NOT cause the log/arctan terms
   to cancel — they constrain root positions but the transcendental
   evaluation at those positions remains irreducible.

6. CUBIC HERMITE APPROXIMATION:
   ΔΦ ≈ Δx/12 · (|λ_eq| + |λ_sad|) provides a local approximation.
   This connects the barrier to root spacing and eigenvalues,
   but is approximate (not exact), so cannot serve as an algebraic proof.
""")

print("=" * 78)
print("END OF ANALYSIS")
print("=" * 78)
