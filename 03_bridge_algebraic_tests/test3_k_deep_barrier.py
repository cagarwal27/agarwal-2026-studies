#!/usr/bin/env python3
"""
Extended K computation for deep barriers — checking whether K → 1.
Uses the double-well (cleanest system: K depends ONLY on barrier, not on α).
Also investigates the lake model's anomalous K decrease at high barriers.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')


def compute_K_doublewell(barrier, alpha=5.0, N_grid=100000):
    """
    Compute K for the symmetric double-well at a given barrier height.
    Uses α=5.0 (large) so that the reflecting boundary is far from the well.
    Uses N_grid=100000 for higher precision.
    """
    x_eq = -np.sqrt(alpha)
    x_sad = 0.0
    DeltaPhi = alpha**2 / 4.0

    sigma_sq = 2 * DeltaPhi / barrier
    sigma = np.sqrt(sigma_sq)

    lam_eq = 2 * alpha
    lam_sad = alpha
    tau = 1.0 / lam_eq

    # Kramers
    C_std = np.sqrt(lam_eq * lam_sad) / (2 * np.pi)
    D_kramers = np.exp(barrier) / (C_std * tau)

    # For deep barriers, we need a tighter lower boundary to avoid overflow
    # The reflecting boundary should be close enough that Φ doesn't overflow
    # but far enough to capture all probability mass.
    # Use x_lo = x_eq - n_sigma * sigma / sqrt(2*lam_eq)
    # where n_sigma captures the relevant region
    well_width = sigma / np.sqrt(2 * lam_eq)  # thermal width of the well
    x_lo = x_eq - min(20 * well_width, 3.0)  # at most 3.0 below equilibrium

    # Check that the potential at x_lo doesn't cause overflow
    def f_dw(x):
        return alpha * x - x**3

    def U_func(x):
        """Quasi-potential U(x) = -∫_{x_eq}^{x} f(z) dz"""
        val, _ = quad(lambda z: -f_dw(z), x_eq, x)
        return val

    U_lo = U_func(x_lo)
    Phi_lo = 2 * U_lo / sigma**2

    # If Phi_lo would cause overflow, tighten the boundary
    max_safe_Phi = 500  # exp(500) is within double precision
    if Phi_lo > max_safe_Phi:
        # Binary search for a safe x_lo
        x_test_lo, x_test_hi = x_eq - 0.01, x_lo
        for _ in range(50):
            x_mid = (x_test_lo + x_test_hi) / 2
            U_mid = U_func(x_mid)
            if 2 * U_mid / sigma**2 > max_safe_Phi:
                x_test_hi = x_mid
            else:
                x_test_lo = x_mid
        x_lo = x_test_lo

    x_grid = np.linspace(x_lo, x_sad + 0.001, N_grid)
    dx_g = x_grid[1] - x_grid[0]

    neg_f = np.array([-f_dw(x) for x in x_grid])
    U_raw = np.cumsum(neg_f) * dx_g
    i_eq = np.argmin(np.abs(x_grid - x_eq))
    U_grid = U_raw - U_raw[i_eq]

    Phi = 2 * U_grid / sigma**2

    # Check for overflow
    Phi_max = Phi.max()
    if Phi_max > 700:
        shift = Phi_max - 500
        Phi_s = Phi - shift
    else:
        Phi_s = Phi

    exp_neg_Phi = np.exp(-Phi_s)
    I_x = np.cumsum(exp_neg_Phi) * dx_g
    psi = (2 / sigma**2) * np.exp(Phi_s) * I_x

    i_sad = np.argmin(np.abs(x_grid - x_sad))
    MFPT_exact = np.trapz(psi[i_eq:i_sad + 1], x_grid[i_eq:i_sad + 1])
    D_exact = MFPT_exact / tau

    K = D_exact / D_kramers

    return K, D_exact, D_kramers


def compute_K_lake(a_val, cv, N_grid=100000):
    """Compute K for the lake model with higher resolution."""
    b, r, q, h = 0.8, 1.0, 8, 1.0

    def f_lake(x):
        return a_val - b * x + r * x**q / (x**q + h**q)

    def f_lake_deriv(x):
        return -b + r * q * x**(q-1) * h**q / (x**q + h**q)**2

    # Find roots
    xs = np.linspace(0.01, 2.5, 5000)
    fs = np.array([f_lake(x) for x in xs])
    roots = []
    for i in range(len(fs) - 1):
        if fs[i] * fs[i+1] < 0:
            try:
                root = brentq(f_lake, xs[i], xs[i+1])
                roots.append(root)
            except:
                pass

    if len(roots) < 3:
        return None

    x_clear, x_sad = roots[0], roots[1]
    lam_eq = abs(f_lake_deriv(x_clear))
    lam_sad = abs(f_lake_deriv(x_sad))
    tau = 1.0 / lam_eq

    sigma = cv * x_clear * np.sqrt(2 * lam_eq)

    DeltaPhi, _ = quad(lambda x: -f_lake(x), x_clear, x_sad)
    barrier = 2 * DeltaPhi / sigma**2

    C_std = np.sqrt(lam_eq * lam_sad) / (2 * np.pi)
    D_kramers = np.exp(barrier) / (C_std * tau)

    x_grid = np.linspace(0.001, x_sad + 0.001, N_grid)
    dx_g = x_grid[1] - x_grid[0]

    neg_f = np.array([-f_lake(x) for x in x_grid])
    U_raw = np.cumsum(neg_f) * dx_g
    i_eq = np.argmin(np.abs(x_grid - x_clear))
    U_grid = U_raw - U_raw[i_eq]
    Phi = 2 * U_grid / sigma**2

    Phi_max = Phi.max()
    if Phi_max > 700:
        shift = Phi_max - 500
        Phi_s = Phi - shift
    else:
        Phi_s = Phi

    exp_neg_Phi = np.exp(-Phi_s)
    I_x = np.cumsum(exp_neg_Phi) * dx_g
    psi = (2 / sigma**2) * np.exp(Phi_s) * I_x

    i_sad = np.argmin(np.abs(x_grid - x_sad))
    MFPT_exact = np.trapz(psi[i_eq:i_sad + 1], x_grid[i_eq:i_sad + 1])
    D_exact = MFPT_exact / tau

    K = D_exact / D_kramers

    return K, barrier, lam_eq, lam_sad, DeltaPhi


# ==============================================================================
# Part 1: Double-well K at extended barrier range
# ==============================================================================
print("=" * 80)
print("DOUBLE-WELL: K vs barrier (extended range)")
print("=" * 80)

barriers_extended = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                     12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 40.0, 50.0]

print(f"\n{'Barrier':<10} {'K':<12} {'D_exact':<16} {'D_Kramers':<16} {'1-K':<12}")
print("-" * 66)

dw_barrier_K = []
for b_val in barriers_extended:
    try:
        K, D_ex, D_kr = compute_K_doublewell(b_val, alpha=5.0, N_grid=100000)
        if np.isfinite(K) and K > 0:
            dw_barrier_K.append((b_val, K))
            print(f"{b_val:<10.1f} {K:<12.6f} {D_ex:<16.6e} {D_kr:<16.6e} {1-K:<12.6f}")
        else:
            print(f"{b_val:<10.1f} {'NaN':>12}")
    except Exception as e:
        print(f"{b_val:<10.1f} FAILED: {e}")

# Fit K = 1 - c/barrier  (expected from Kramers theory)
if len(dw_barrier_K) > 2:
    bs = np.array([x[0] for x in dw_barrier_K])
    ks = np.array([x[1] for x in dw_barrier_K])

    # Only fit on barrier > 5 to be in asymptotic regime
    mask = bs > 5
    if np.sum(mask) > 2:
        bs_fit = bs[mask]
        ks_fit = ks[mask]

        # Fit: K = a + b/barrier
        X = np.column_stack([np.ones_like(bs_fit), 1.0/bs_fit])
        coeffs = np.linalg.lstsq(X, ks_fit, rcond=None)[0]
        K_pred = X @ coeffs
        SS_res = np.sum((ks_fit - K_pred)**2)
        SS_tot = np.sum((ks_fit - np.mean(ks_fit))**2)
        R2 = 1 - SS_res/SS_tot if SS_tot > 0 else 0

        print(f"\nFit (barrier > 5): K = {coeffs[0]:.6f} + {coeffs[1]:.4f} / barrier")
        print(f"R² = {R2:.6f}")
        print(f"Extrapolated K(∞) = {coeffs[0]:.6f}")
        print(f"If K(∞) ≈ 1, then K = 1 + ({coeffs[1]:.4f})/barrier, correction coeff = {coeffs[1]:.4f}")

        # Also try K = 1 - c₁/barrier - c₂/barrier²
        X2 = np.column_stack([np.ones_like(bs_fit), 1.0/bs_fit, 1.0/bs_fit**2])
        coeffs2 = np.linalg.lstsq(X2, ks_fit, rcond=None)[0]
        K_pred2 = X2 @ coeffs2
        SS_res2 = np.sum((ks_fit - K_pred2)**2)
        R2_2 = 1 - SS_res2/SS_tot if SS_tot > 0 else 0
        print(f"\nFit (quadratic): K = {coeffs2[0]:.6f} + {coeffs2[1]:.4f}/B + {coeffs2[2]:.4f}/B²")
        print(f"R² = {R2_2:.6f}")
        print(f"Extrapolated K(∞) = {coeffs2[0]:.6f}")


# ==============================================================================
# Part 2: Check grid resolution effect on lake model at high barriers
# ==============================================================================
print("\n\n" + "=" * 80)
print("LAKE: Grid resolution study (a=0.15, CV=0.35)")
print("=" * 80)

for N in [50000, 100000, 200000, 500000]:
    try:
        result = compute_K_lake(0.15, 0.35, N_grid=N)
        if result is not None:
            K, barrier, le, ls, dp = result
            print(f"  N={N:>7d}: K = {K:.6f}, barrier = {barrier:.2f}")
    except Exception as e:
        print(f"  N={N:>7d}: FAILED — {e}")

# Also compare lake at moderate barrier
print("\nLake grid resolution (a=0.3266, CV=0.30):")
for N in [50000, 100000, 200000]:
    try:
        result = compute_K_lake(0.3266, 0.30, N_grid=N)
        if result is not None:
            K, barrier, le, ls, dp = result
            print(f"  N={N:>7d}: K = {K:.6f}, barrier = {barrier:.2f}")
    except Exception as e:
        print(f"  N={N:>7d}: FAILED — {e}")


# ==============================================================================
# Part 3: Lake model — separate barrier height from potential shape
# ==============================================================================
print("\n\n" + "=" * 80)
print("LAKE: K at FIXED barrier ≈ 5, varying a (= varying shape)")
print("=" * 80)
print("(This isolates shape dependence from barrier-height dependence)")

a_values = [0.25, 0.30, 0.326588, 0.35, 0.37]
for a_val in a_values:
    # Find equilibria
    b, r, q, h = 0.8, 1.0, 8, 1.0
    def f_lake(x):
        return a_val - b * x + r * x**q / (x**q + h**q)
    def f_lake_deriv(x):
        return -b + r * q * x**(q-1) * h**q / (x**q + h**q)**2

    xs = np.linspace(0.01, 2.5, 5000)
    fs = np.array([f_lake(x) for x in xs])
    roots = []
    for i in range(len(fs) - 1):
        if fs[i] * fs[i+1] < 0:
            try:
                root = brentq(f_lake, xs[i], xs[i+1])
                roots.append(root)
            except:
                pass

    if len(roots) < 3:
        continue

    x_clear, x_sad = roots[0], roots[1]
    lam_eq = abs(f_lake_deriv(x_clear))
    lam_sad = abs(f_lake_deriv(x_sad))

    DeltaPhi, _ = quad(lambda x: -f_lake(x), x_clear, x_sad)

    # Find CV that gives barrier ≈ 5
    target_barrier = 5.0
    sigma_target = np.sqrt(2 * DeltaPhi / target_barrier)
    cv_target = sigma_target / (x_clear * np.sqrt(2 * lam_eq))

    result = compute_K_lake(a_val, cv_target, N_grid=100000)
    if result is not None:
        K, barrier, le, ls, dp = result
        print(f"  a={a_val:.4f}: CV={cv_target:.4f}, barrier={barrier:.4f}, "
              f"λ_eq/λ_sad={le/ls:.4f}, K={K:.6f}")


# ==============================================================================
# Part 4: Known analytic result — pure Ornstein-Uhlenbeck MFPT
# ==============================================================================
print("\n\n" + "=" * 80)
print("ANALYTIC CHECK: Ornstein-Uhlenbeck MFPT")
print("=" * 80)
print("For f(x) = -κ(x - x₀), escape from x₀ to x₀ + L")
print("This is locally parabolic — Kramers should be EXACT in the limit.")

from scipy.special import erfi

def compute_K_OU(kappa, L, sigma):
    """
    Exact MFPT for OU process dX = -κ(X-x₀)dt + σdW
    from x₀ to x₀ + L, with reflecting boundary at x₀.

    Actually, for OU process there's no second equilibrium, so this
    isn't exactly comparable. Instead, we use the formula directly.
    """
    # V(x) = κ(x-x₀)²/2, D_noise = σ²/2
    # Barrier = κL²/σ² = V(x₀+L)/D_noise
    # f'(x₀) = -κ, so lam_eq = κ
    # f'(x₀+L) = -κ, but at the "saddle" (absorbing boundary)
    # there's no natural saddle, so Kramers doesn't directly apply.
    pass

# Actually, let me verify against the double-well where we know the
# exact answer should converge to Kramers.

# The better diagnostic: check numerical vs scipy.integrate for the double-well
print("\nDouble-well exact MFPT via scipy quad (no grid):")
print("This eliminates grid discretization error.\n")

def compute_K_dw_quad(barrier, alpha=5.0):
    """Compute K using scipy quad for both integrals (no grid)."""
    x_eq = -np.sqrt(alpha)
    x_sad = 0.0
    sigma_sq = 2 * (alpha**2/4) / barrier
    sigma = np.sqrt(sigma_sq)

    lam_eq = 2 * alpha
    lam_sad = alpha
    tau = 1.0 / lam_eq

    C_std = np.sqrt(lam_eq * lam_sad) / (2 * np.pi)
    D_kramers = np.exp(barrier) / (C_std * tau)

    def f_dw(x):
        return alpha * x - x**3

    # Quasi-potential relative to x_eq
    def U(x):
        val, _ = quad(lambda z: -f_dw(z), x_eq, x, limit=200)
        return val

    def Phi(x):
        return 2 * U(x) / sigma**2

    # Tight lower bound
    well_width = sigma / np.sqrt(2 * lam_eq)
    x_lo = x_eq - min(15 * well_width, 3.0)

    def inner_integral(x):
        """I(x) = ∫_{x_lo}^{x} exp(-Φ(y)) dy"""
        val, _ = quad(lambda y: np.exp(-Phi(y)), x_lo, x, limit=200)
        return val

    def psi(x):
        """Integrand for MFPT"""
        return (2 / sigma**2) * np.exp(Phi(x)) * inner_integral(x)

    MFPT_exact, _ = quad(psi, x_eq, x_sad, limit=200)
    D_exact = MFPT_exact / tau
    K = D_exact / D_kramers

    return K, D_exact, D_kramers

print(f"{'Barrier':<10} {'K (quad)':<14} {'K (grid)':<14} {'Difference':<12}")
print("-" * 50)
for b_val in [2.0, 4.0, 6.0, 8.0, 10.0]:
    try:
        K_quad, _, _ = compute_K_dw_quad(b_val, alpha=5.0)
        K_grid, _, _ = compute_K_doublewell(b_val, alpha=5.0, N_grid=100000)
        print(f"{b_val:<10.1f} {K_quad:<14.8f} {K_grid:<14.8f} {abs(K_quad-K_grid):<12.2e}")
    except Exception as e:
        print(f"{b_val:<10.1f} FAILED: {e}")


# ==============================================================================
# Part 5: Analytic correction term for the symmetric double-well
# ==============================================================================
print("\n\n" + "=" * 80)
print("ANALYTIC: Kramers correction for quartic double-well")
print("=" * 80)
print("""
For V(x) = -(α/2)x² + (1/4)x⁴, the well is at x=-√α, saddle at x=0.

The exact MFPT has the asymptotic expansion (Berglund 2011, J. Phys. A):
  MFPT = MFPT_Kramers × (1 + Σ c_n ε^n)
where ε = D_noise/ΔV = σ²/(2ΔV) = 1/barrier, and c_n depends on
the anharmonicity of V.

For the quartic well: V(x) ≈ V_min + (α)(x-x_eq)² - (α/4)(x-x_eq)³ + ...
The first correction involves the skewness of the well.

Key result: the corrections are NOT small for moderate barriers.
The prefactor expansion converges slowly.

Let's check: at what barrier does K reach 0.9?
""")

# Extrapolate using our data
if len(dw_barrier_K) > 5:
    bs = np.array([x[0] for x in dw_barrier_K])
    ks = np.array([x[1] for x in dw_barrier_K])

    # Log-log fit for 1-K vs barrier
    mask = (bs > 3) & np.isfinite(ks) & (ks > 0) & (ks < 1)
    if np.sum(mask) > 3:
        log_b = np.log(bs[mask])
        log_1mK = np.log(1 - ks[mask])

        # Fit log(1-K) = a + b*log(barrier)
        X = np.column_stack([np.ones_like(log_b), log_b])
        coeffs = np.linalg.lstsq(X, log_1mK, rcond=None)[0]

        print(f"Power law fit (barrier > 3): 1-K = {np.exp(coeffs[0]):.4f} × barrier^({coeffs[1]:.4f})")

        # Estimate barrier where K = 0.9
        # 1-K = 0.1 => log(0.1) = coeffs[0] + coeffs[1]*log(B)
        log_B_90 = (np.log(0.1) - coeffs[0]) / coeffs[1]
        print(f"Estimated barrier for K = 0.9: {np.exp(log_B_90):.0f}")

        # Estimate barrier where K = 0.99
        log_B_99 = (np.log(0.01) - coeffs[0]) / coeffs[1]
        print(f"Estimated barrier for K = 0.99: {np.exp(log_B_99):.0f}")

        # Estimate barrier where K = 0.95
        log_B_95 = (np.log(0.05) - coeffs[0]) / coeffs[1]
        print(f"Estimated barrier for K = 0.95: {np.exp(log_B_95):.0f}")

print("\nDone.")
