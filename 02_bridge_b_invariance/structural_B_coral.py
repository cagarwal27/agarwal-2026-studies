#!/usr/bin/env python3
"""
STRUCTURAL B TEST: CORAL REEF (MUMBY 2007)
===========================================
Test whether B = 2*DeltaPhi / sigma*^2 is approximately constant
across the bistable range of the grazing parameter g.

For the lake system, B = 4.27 +/- 2% while the barrier varied 100x.
This script checks if the same structural constancy holds for coral.

Model: Mumby et al. 2007 (Nature 450:98-101)
  dM/dt = a*M*C - g*M/(M+T) + gamma*M*T
  dC/dt = r*T*C - d*C - a*M*C
  T = 1 - M - C

Adiabatic reduction to 1D along C-nullcline (C relaxes ~2x faster):
  T(M) = (d + a*M)/r
  C(M) = 1 - M - (d + a*M)/r
  f_eff(M) = a*M*C(M) - g*M/(M + T(M)) + gamma*M*T(M)

D_product = 1111 (from epsilon = 0.03)
"""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad

# ============================================================
# Mumby 2007 parameters (Caribbean calibration)
# ============================================================
a = 0.1       # macroalgal overgrowth of coral (yr^-1)
gamma = 0.8   # macroalgal spread over turf (yr^-1)
r = 1.0       # coral growth over turf (yr^-1)
d = 0.44      # coral natural mortality (yr^-1)

D_PRODUCT = 1111.1  # (1/0.03)^2

# Bistable range boundaries (analytical)
C_boundary = 1 - d / r          # = 0.56
T_boundary = d / r              # = 0.44

# g_lower: saddle merges with coral equilibrium (M->0)
# At M=0: dM/dt = M * [a*C_boundary - g/T_boundary + gamma*T_boundary]
# Saddle appears when a*C_boundary - g/T_boundary + gamma*T_boundary = 0
g_lower = T_boundary * (a * C_boundary + gamma * T_boundary)
# g_upper: saddle merges with algae equilibrium
# At C=0: M_algae = 1 - g/gamma, T_algae = g/gamma
# Coral invasion eigenvalue = r*T_algae - d - a*M_algae = 0
# => r*g/gamma - d - a*(1 - g/gamma) = 0
# => g*(r + a)/gamma = d + a
g_upper = (d + a) * gamma / (r + a)


# ============================================================
# Effective 1D model along C-nullcline
# ============================================================
def C_of_M(M):
    """Coral cover on the C-nullcline."""
    return max(1 - M - (d + a * M) / r, 0.0)

def T_of_M(M):
    """Turf cover on the C-nullcline."""
    return (d + a * M) / r

def f_eff(M, g):
    """Effective 1D drift for macroalgae along C-nullcline."""
    C = C_of_M(M)
    T = T_of_M(M)
    denom = M + T
    if denom < 1e-30:
        return 0.0
    return a * M * C - g * M / denom + gamma * M * T


def f_eff_deriv(M, g, dM=1e-7):
    """Numerical derivative of f_eff."""
    return (f_eff(M + dM, g) - f_eff(M - dM, g)) / (2 * dM)


# ============================================================
# Equilibrium finding
# ============================================================
def find_equilibria(g_val):
    """
    Find equilibria of f_eff(M, g) = 0 for M > 0.

    Known equilibria:
    - M=0 is always an equilibrium (coral state)
    - M_algae = 1 - g/gamma (algae state, C=0)
    - Interior saddle (if bistable)

    Returns (M_eq_coral, M_saddle, M_eq_algae) or None if not bistable.
    """
    # Coral state is at M = 0
    M_coral = 0.0

    # Algae state: C = 0 => M = 1 - g/gamma
    M_algae = 1 - g_val / gamma
    if M_algae <= 0:
        return None

    # Interior saddle: scan for zero crossing of f_eff in (0, M_algae)
    M_scan = np.linspace(1e-6, M_algae - 1e-6, 50000)
    f_vals = np.array([f_eff(m, g_val) for m in M_scan])

    # Look for sign changes (excluding M=0 which is always a root)
    saddles = []
    for i in range(len(f_vals) - 1):
        if f_vals[i] * f_vals[i + 1] < 0:
            try:
                root = brentq(lambda m: f_eff(m, g_val), M_scan[i], M_scan[i + 1])
                saddles.append(root)
            except:
                pass

    if len(saddles) == 0:
        return None

    # The first zero crossing from the left is the saddle
    M_sad = saddles[0]

    return (M_coral, M_sad, M_algae)


# ============================================================
# Barrier computation
# ============================================================
def compute_barrier(g_val, M_eq, M_sad):
    """
    Compute barrier DeltaPhi = integral of f_eff from M_eq to M_sad.

    V(M) = -integral_0^M f_eff(m) dm
    DeltaPhi = V(M_sad) - V(M_eq) = -integral_{M_eq}^{M_sad} f_eff(m) dm

    For transition FROM coral (M_eq=0) TO algae, the barrier is the
    potential difference between the saddle and the coral well.
    """
    result, _ = quad(lambda m: -f_eff(m, g_val), M_eq, M_sad, limit=200)
    return result


# ============================================================
# Exact MFPT integral
# ============================================================
def compute_D_exact(sigma, g_val, M_eq, M_sad, lam_eq):
    """
    Exact MFPT-based dimensionless persistence D for given sigma.
    D = MFPT * |lambda_eq|

    Uses reflecting boundary at M=0 (coral state).
    """
    tau = 1.0 / abs(lam_eq)

    N_grid = 80000
    # Grid from near M_eq (=0) to M_sad
    x_lo = max(1e-10, M_eq)
    x_hi = M_sad
    x_grid = np.linspace(x_lo, x_hi, N_grid)
    dx_g = x_grid[1] - x_grid[0]

    # Build potential on grid
    neg_f = np.array([-f_eff(x, g_val) for x in x_grid])
    V_raw = np.cumsum(neg_f) * dx_g

    # Reference to equilibrium (minimum)
    i_eq = 0  # coral eq is at M=0, left edge
    V_grid = V_raw - V_raw[i_eq]

    # Scaled potential
    Phi = 2 * V_grid / sigma**2

    # Clamp to avoid overflow
    if Phi.max() > 600:
        return np.inf
    Phi = np.clip(Phi, -500, 500)

    exp_neg_Phi = np.exp(-Phi)
    I_x = np.cumsum(exp_neg_Phi) * dx_g

    psi = (2.0 / sigma**2) * np.exp(Phi) * I_x

    MFPT = np.trapz(psi, x_grid)
    D = MFPT * abs(lam_eq)  # = MFPT / tau * 1 (dimensionless)
    return D


def find_sigma_star(g_val, M_eq, M_sad, lam_eq, D_target=1111.1):
    """Find sigma where D_exact = D_target using bisection on log(sigma)."""
    def objective(log_sigma):
        sigma = np.exp(log_sigma)
        D = compute_D_exact(sigma, g_val, M_eq, M_sad, lam_eq)
        if D == np.inf:
            return 1.0  # too small sigma -> huge D -> positive
        return np.log(max(D, 1e-30)) - np.log(D_target)

    try:
        log_sigma_star = brentq(objective, np.log(0.001), np.log(1.0), xtol=1e-10, maxiter=200)
        return np.exp(log_sigma_star)
    except ValueError:
        return np.nan


# ============================================================
# MAIN: Sweep g across bistable range
# ============================================================
print("=" * 72)
print("STRUCTURAL B TEST: CORAL REEF (MUMBY 2007)")
print("=" * 72)
print(f"\nParameters: a={a}, gamma={gamma}, r={r}, d={d}")
print(f"D_product = {D_PRODUCT:.1f}")
print(f"Bistable range: g in [{g_lower:.6f}, {g_upper:.6f}]")
print(f"Operating point from Step 7: g = 0.30")

# Margin to avoid fold bifurcation singularities
margin_frac = 0.03
g_margin = margin_frac * (g_upper - g_lower)
g_scan = np.linspace(g_lower + g_margin, g_upper - g_margin, 30)

print(f"\nScanning {len(g_scan)} values of g in [{g_scan[0]:.6f}, {g_scan[-1]:.6f}]")
print(f"\n{'g':>10s} {'M_sad':>8s} {'lam_eq':>10s} {'DeltaPhi':>14s} {'sigma*':>12s} {'B':>8s}")
print("-" * 72)

results = []
for g_val in g_scan:
    eq = find_equilibria(g_val)
    if eq is None:
        continue

    M_eq, M_sad, M_algae = eq

    # Eigenvalue at coral equilibrium (M=0):
    # f_eff(M) ~ M * [a*C_boundary - g/(T_boundary) + gamma*T_boundary] near M=0
    # So lambda_eq = a*C_boundary - g/T_boundary + gamma*T_boundary
    lam_eq = a * C_boundary - g_val / T_boundary + gamma * T_boundary

    # For bistability, coral state must be stable: lam_eq < 0
    if lam_eq >= 0:
        continue

    # Compute barrier
    DeltaPhi = compute_barrier(g_val, M_eq + 1e-10, M_sad)
    if DeltaPhi <= 0:
        continue

    # Find sigma*
    sigma_star = find_sigma_star(g_val, M_eq, M_sad, lam_eq, D_target=D_PRODUCT)
    if np.isnan(sigma_star):
        continue

    # Compute B
    B = 2 * DeltaPhi / sigma_star**2

    results.append({
        'g': g_val,
        'M_sad': M_sad,
        'M_algae': M_algae,
        'lam_eq': lam_eq,
        'DeltaPhi': DeltaPhi,
        'sigma_star': sigma_star,
        'B': B,
    })

    print(f"{g_val:10.6f} {M_sad:8.5f} {lam_eq:10.5f} {DeltaPhi:14.10f} {sigma_star:12.8f} {B:8.4f}")


# ============================================================
# Analysis
# ============================================================
print("\n" + "=" * 72)
print("ANALYSIS")
print("=" * 72)

if len(results) < 3:
    print("ERROR: Too few valid points for analysis")
else:
    B_vals = np.array([r['B'] for r in results])
    DPhi_vals = np.array([r['DeltaPhi'] for r in results])
    sig_vals = np.array([r['sigma_star'] for r in results])
    g_vals = np.array([r['g'] for r in results])

    B_mean = np.mean(B_vals)
    B_std = np.std(B_vals)
    B_cv = B_std / B_mean * 100
    B_min = np.min(B_vals)
    B_max = np.max(B_vals)
    B_range_pct = (B_max - B_min) / B_mean * 100

    DPhi_ratio = np.max(DPhi_vals) / np.min(DPhi_vals)
    sig_ratio = np.max(sig_vals) / np.min(sig_vals)

    print(f"\n  B = 2*DeltaPhi / sigma*^2:")
    print(f"    Mean  = {B_mean:.4f}")
    print(f"    Std   = {B_std:.4f}")
    print(f"    CV    = {B_cv:.2f}%")
    print(f"    Min   = {B_min:.4f}")
    print(f"    Max   = {B_max:.4f}")
    print(f"    Range = {B_range_pct:.2f}% of mean")

    print(f"\n  Barrier variation:")
    print(f"    DeltaPhi_min = {np.min(DPhi_vals):.10f}")
    print(f"    DeltaPhi_max = {np.max(DPhi_vals):.10f}")
    print(f"    Ratio max/min = {DPhi_ratio:.1f}x")

    print(f"\n  sigma* variation:")
    print(f"    sigma*_min = {np.min(sig_vals):.8f}")
    print(f"    sigma*_max = {np.max(sig_vals):.8f}")
    print(f"    Ratio max/min = {sig_ratio:.2f}x")

    # At the Step 7 operating point g=0.30
    closest = min(results, key=lambda r: abs(r['g'] - 0.30))
    print(f"\n  At g = {closest['g']:.4f} (near operating point g=0.30):")
    print(f"    sigma* = {closest['sigma_star']:.6f}  (Step 7 found 0.0299)")
    print(f"    B = {closest['B']:.4f}")
    print(f"    DeltaPhi = {closest['DeltaPhi']:.10f}")

    # Verdict
    print(f"\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    constant = B_cv < 10
    print(f"\n  B varies by {B_cv:.2f}% (CV) across the bistable range")
    print(f"  While DeltaPhi varies by {DPhi_ratio:.1f}x")

    if B_cv < 5:
        verdict = "STRONG CONSTANCY (CV < 5%)"
    elif B_cv < 10:
        verdict = "MODERATE CONSTANCY (CV < 10%)"
    elif B_cv < 20:
        verdict = "WEAK CONSTANCY (CV < 20%)"
    else:
        verdict = "NOT CONSTANT (CV >= 20%)"

    print(f"\n  {verdict}")
    print(f"  B_coral = {B_mean:.2f} +/- {B_cv:.1f}%")
    print(f"\n  Compare lake: B_lake = 4.27 +/- 2%")
    print(f"  If B is constant for coral too, this suggests B is a")
    print(f"  structural invariant of the bistable potential shape.")

    # Check if B is in the habitable zone
    print(f"\n  B habitable zone check: [{1.8}, {6.0}]")
    if 1.8 <= B_mean <= 6.0:
        print(f"  B_coral = {B_mean:.2f} is INSIDE the habitable zone")
    else:
        print(f"  B_coral = {B_mean:.2f} is OUTSIDE the habitable zone")

print("\nDone.")
