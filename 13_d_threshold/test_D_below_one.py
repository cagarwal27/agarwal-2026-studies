#!/usr/bin/env python3
"""
Test A: Can D go below 1?

Uses the 2-channel lake model (van Nes & Scheffer) with tunable regulatory
channels. Sweeps one ε past 1.0 while holding the other fixed. Tracks:
  1. Does bistability survive when ε > 1?
  2. Does D_exact go below 1?
  3. Or does bistability vanish before D can reach 1?

Also tests multi-channel scenarios where ∏εᵢ > 1 but individual εᵢ < 1.

Framework: D = ∏(1/εᵢ) = product equation
           D = MFPT/τ = Kramers definition
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# Base lake model (van Nes & Scheffer 2007)
# ================================================================
A_P = 0.326588
B_P = 0.8
R_P = 1.0
Q_P = 8
H_P = 1.0

X_CL = 0.409217   # clear-water equilibrium
X_SD = 0.978152   # saddle
X_TB = 1.634126   # turbid equilibrium
LAM_CL = -0.784651
TAU_L = 1.0 / abs(LAM_CL)

# Channel shape parameters (from Step 8)
K1 = 0.5    # Hill^4 half-saturation
K2 = 2.0    # Michaelis-Menten half-saturation

K1_4 = K1**4


def f_lake(x):
    """Original lake drift."""
    return A_P - B_P * x + R_P * x**Q_P / (x**Q_P + H_P**Q_P)


# ================================================================
# 2-channel model: drift = base + recycling - b0*x - c1*g1(x) - c2*g2(x)
# ================================================================
def g1(x):
    """Channel 1: Hill^4."""
    return x**4 / (x**4 + K1_4)

def g2(x):
    """Channel 2: Michaelis-Menten."""
    return x / (x + K2)


def make_drift_2ch(c1, c2, b0):
    """2-channel drift function."""
    def f(x):
        rec = R_P * x**Q_P / (x**Q_P + H_P**Q_P)
        return A_P + rec - b0 * x - c1 * g1(x) - c2 * g2(x)
    return f


def find_equilibria(f_func, x_lo=0.001, x_hi=4.0, N=400000):
    """Find all roots."""
    x_scan = np.linspace(x_lo, x_hi, N)
    f_scan = np.array([f_func(x) for x in x_scan])
    sign_changes = np.where(np.diff(np.sign(f_scan)))[0]
    roots = []
    for i in sign_changes:
        try:
            root = brentq(f_func, x_scan[i], x_scan[i + 1], xtol=1e-12)
            roots.append(root)
        except (ValueError, RuntimeError):
            pass
    return roots


def fderiv(f_func, x, dx=1e-7):
    return (f_func(x + dx) - f_func(x - dx)) / (2 * dx)


def compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma, N=80000):
    """Exact MFPT-based D."""
    xg = np.linspace(0.001, x_saddle + 0.001, N)
    dx = xg[1] - xg[0]
    neg_f = np.array([-f_func(x) for x in xg])
    U_raw = np.cumsum(neg_f) * dx
    i_eq = np.argmin(np.abs(xg - x_eq))
    U = U_raw - U_raw[i_eq]
    Phi = 2.0 * U / sigma**2
    # Numerical stability: shift Phi so max is manageable
    Phi_shift = Phi - np.max(Phi)
    exp_neg = np.exp(-Phi)
    Ix = np.cumsum(exp_neg) * dx
    psi = (2.0 / sigma**2) * np.exp(Phi) * Ix
    i_sad = np.argmin(np.abs(xg - x_saddle))
    if i_eq >= i_sad:
        return np.nan
    MFPT = np.trapz(psi[i_eq:i_sad + 1], xg[i_eq:i_sad + 1])
    return MFPT / tau_val


def find_sigma_star(f_func, x_eq, x_saddle, tau_val, D_target,
                    sigma_lo=0.001, sigma_hi=3.0):
    """Find σ* where D_exact = D_target."""
    try:
        D_lo = compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma_lo)
        D_hi = compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma_hi)
        if np.isnan(D_lo) or np.isnan(D_hi):
            return None
        if D_lo < D_target or D_hi > D_target:
            return None
        def obj(s):
            return compute_D_exact(f_func, x_eq, x_saddle, tau_val, s) - D_target
        return brentq(obj, sigma_lo, sigma_hi, xtol=1e-8, maxiter=200)
    except Exception:
        return None


def calibrate_2ch(eps1, eps2):
    """
    Calibrate c1, c2, b0 so that:
    - The clear-water equilibrium is at X_CL
    - ε₁ = c1*g1(x_eq) / (b0*x_eq) = eps1
    - ε₂ = c2*g2(x_eq) / (b0*x_eq) = eps2
    - Total regulation at equilibrium = B_P * X_CL (preserves equilibrium)

    b0*x_eq + c1*g1(x_eq) + c2*g2(x_eq) = B_P * X_CL + recycling_at_eq
    But we need f(x_eq) = 0, so: A_P + rec(x_eq) - b0*x_eq - c1*g1(x_eq) - c2*g2(x_eq) = 0
    Original: A_P + rec(x_eq) - B_P*x_eq = 0, so total_removal = B_P * x_eq
    We need: b0*x_eq + c1*g1(x_eq) + c2*g2(x_eq) = B_P * x_eq (same total removal)

    With ε definitions:
      ε₁ = c1*g1(x_eq) / total_removal_at_saddle
      ε₂ = c2*g2(x_eq) / total_removal_at_saddle

    Simplified (using total_removal as B_P * X_CL for calibration):
      c1*g1(x_eq) = eps1 * B_P * X_CL
      c2*g2(x_eq) = eps2 * B_P * X_CL
      b0 = (B_P * X_CL - c1*g1(x_eq) - c2*g2(x_eq)) / X_CL
    """
    total_reg = B_P * X_CL
    g1_eq = g1(X_CL)
    g2_eq = g2(X_CL)

    c1 = eps1 * total_reg / g1_eq
    c2 = eps2 * total_reg / g2_eq

    # Remaining removal goes to background loss b0*x
    b0 = (total_reg - c1 * g1_eq - c2 * g2_eq) / X_CL

    return c1, c2, b0


# ================================================================
# PART 1: Single-channel sweep — push ε₁ past 1.0
# ================================================================
print("=" * 70)
print("TEST A: CAN D GO BELOW 1?")
print("=" * 70)

print("\n" + "=" * 70)
print("PART 1: Single ε sweep (ε₁ varies, ε₂ = 0.10 fixed)")
print("=" * 70)

eps2_fixed = 0.10
eps1_values = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.87, 0.89, 0.90]

print(f"\n{'ε₁':>6s} {'ε₂':>6s} {'D_prod':>8s} {'b0':>8s} {'bistable?':>10s} {'x_eq':>8s} {'x_sad':>8s} {'ΔΦ':>12s} {'σ*':>10s} {'D_exact':>10s}")
print("-" * 100)

for eps1 in eps1_values:
    eps_product = eps1 * eps2_fixed
    D_product = 1.0 / eps_product

    c1, c2, b0 = calibrate_2ch(eps1, eps2_fixed)

    if b0 < 0:
        print(f"{eps1:6.2f} {eps2_fixed:6.2f} {D_product:8.1f} {b0:8.4f} {'b0 < 0':>10s}  — channels exceed total removal budget")
        continue

    f_func = make_drift_2ch(c1, c2, b0)

    # Check equilibria
    roots = find_equilibria(f_func)
    if len(roots) < 3:
        # Try to find at least stable + saddle
        if len(roots) == 1:
            print(f"{eps1:6.2f} {eps2_fixed:6.2f} {D_product:8.1f} {b0:8.4f} {'MONO':>10s}  — only 1 equilibrium (bistability lost)")
            continue
        elif len(roots) == 2:
            # Could be saddle-node (bistability marginal)
            stabilities = [fderiv(f_func, r) for r in roots]
            print(f"{eps1:6.2f} {eps2_fixed:6.2f} {D_product:8.1f} {b0:8.4f} {'MARGINAL':>10s}  — 2 equilibria (roots: {[f'{r:.4f}' for r in roots]})")
            continue

    # Identify stable/unstable
    stabilities = [(r, fderiv(f_func, r)) for r in roots]
    stable = [(r, fp) for r, fp in stabilities if fp < 0]
    unstable = [(r, fp) for r, fp in stabilities if fp > 0]

    if len(stable) < 1 or len(unstable) < 1:
        print(f"{eps1:6.2f} {eps2_fixed:6.2f} {D_product:8.1f} {b0:8.4f} {'NO SADDLE':>10s}")
        continue

    x_eq = stable[0][0]
    x_sad = unstable[0][0]
    lam_eq = stable[0][1]
    tau = 1.0 / abs(lam_eq)

    # Barrier
    DPhi, _ = quad(lambda x: -f_func(x), x_eq, x_sad)

    if DPhi <= 0:
        print(f"{eps1:6.2f} {eps2_fixed:6.2f} {D_product:8.1f} {b0:8.4f} {'ΔΦ ≤ 0':>10s}  — no barrier (equilibrium not a minimum)")
        continue

    # Find σ* where D_exact = D_product
    sigma_star = find_sigma_star(f_func, x_eq, x_sad, tau, D_product)

    if sigma_star is not None:
        D_check = compute_D_exact(f_func, x_eq, x_sad, tau, sigma_star)
        noise_frac = sigma_star / (x_sad - x_eq) if x_sad > x_eq else np.nan
        print(f"{eps1:6.2f} {eps2_fixed:6.2f} {D_product:8.1f} {b0:8.4f} {'YES':>10s} {x_eq:8.4f} {x_sad:8.4f} {DPhi:12.6e} {sigma_star:10.6f} {D_check:10.2f}")
    else:
        # σ* not found — check D at extreme noise levels
        D_at_high_noise = compute_D_exact(f_func, x_eq, x_sad, tau, 1.0)
        D_at_low_noise = compute_D_exact(f_func, x_eq, x_sad, tau, 0.005)
        print(f"{eps1:6.2f} {eps2_fixed:6.2f} {D_product:8.1f} {b0:8.4f} {'YES':>10s} {x_eq:8.4f} {x_sad:8.4f} {DPhi:12.6e} {'N/A':>10s}  D_range=[{D_at_high_noise:.1f},{D_at_low_noise:.1f}]")


# ================================================================
# PART 2: Both channels large — product > 1 but individual ε < 1
# ================================================================
print("\n\n" + "=" * 70)
print("PART 2: Both channels large (∏εᵢ > 1 but each εᵢ < 1)")
print("=" * 70)
print("Can we get D_product < 1 while the system remains bistable?")

# Pairs where ε₁ × ε₂ > 1 (so D_product < 1)
eps_pairs = [
    (0.60, 0.60),  # product = 0.36, D = 2.78
    (0.70, 0.70),  # product = 0.49, D = 2.04
    (0.75, 0.75),  # product = 0.5625, D = 1.78
    (0.80, 0.80),  # product = 0.64, D = 1.56
    (0.85, 0.85),  # product = 0.7225, D = 1.38
    (0.40, 0.40),  # product = 0.16, D = 6.25
    (0.50, 0.50),  # product = 0.25, D = 4.0
    # Now push past product = 1:
    (0.80, 0.90),  # product = 0.72, D = 1.39
    (0.90, 0.90),  # product = 0.81, D = 1.23
    (0.85, 0.90),  # product = 0.765, D = 1.31
    (0.90, 0.95),  # product = 0.855, D = 1.17
    (0.95, 0.95),  # product = 0.9025, D = 1.11
]

print(f"\n{'ε₁':>6s} {'ε₂':>6s} {'ε₁×ε₂':>8s} {'D_prod':>8s} {'b0':>8s} {'bistable?':>10s} {'x_eq':>8s} {'x_sad':>8s} {'ΔΦ':>12s} {'σ*':>10s}")
print("-" * 100)

for eps1, eps2 in eps_pairs:
    eps_product = eps1 * eps2
    D_product = 1.0 / eps_product

    c1, c2, b0 = calibrate_2ch(eps1, eps2)

    if b0 < 0:
        print(f"{eps1:6.2f} {eps2:6.2f} {eps_product:8.4f} {D_product:8.2f} {b0:8.4f} {'b0 < 0':>10s}  — channels exceed total removal budget")
        continue

    f_func = make_drift_2ch(c1, c2, b0)
    roots = find_equilibria(f_func)

    if len(roots) < 3:
        if len(roots) == 1:
            print(f"{eps1:6.2f} {eps2:6.2f} {eps_product:8.4f} {D_product:8.2f} {b0:8.4f} {'MONO':>10s}  — bistability lost")
        elif len(roots) == 2:
            print(f"{eps1:6.2f} {eps2:6.2f} {eps_product:8.4f} {D_product:8.2f} {b0:8.4f} {'MARGINAL':>10s}  — {[f'{r:.3f}' for r in roots]}")
        else:
            print(f"{eps1:6.2f} {eps2:6.2f} {eps_product:8.4f} {D_product:8.2f} {b0:8.4f} {'NONE':>10s}")
        continue

    stabilities = [(r, fderiv(f_func, r)) for r in roots]
    stable = [(r, fp) for r, fp in stabilities if fp < 0]
    unstable = [(r, fp) for r, fp in stabilities if fp > 0]

    if len(stable) < 1 or len(unstable) < 1:
        print(f"{eps1:6.2f} {eps2:6.2f} {eps_product:8.4f} {D_product:8.2f} {b0:8.4f} {'UNSTABLE':>10s}")
        continue

    x_eq = stable[0][0]
    x_sad = unstable[0][0]
    lam_eq = stable[0][1]
    tau = 1.0 / abs(lam_eq)

    DPhi, _ = quad(lambda x: -f_func(x), x_eq, x_sad)

    if DPhi <= 0:
        print(f"{eps1:6.2f} {eps2:6.2f} {eps_product:8.4f} {D_product:8.2f} {b0:8.4f} {'ΔΦ ≤ 0':>10s}")
        continue

    sigma_star = find_sigma_star(f_func, x_eq, x_sad, tau, D_product)

    if sigma_star is not None:
        D_check = compute_D_exact(f_func, x_eq, x_sad, tau, sigma_star)
        print(f"{eps1:6.2f} {eps2:6.2f} {eps_product:8.4f} {D_product:8.2f} {b0:8.4f} {'YES':>10s} {x_eq:8.4f} {x_sad:8.4f} {DPhi:12.6e} {sigma_star:10.6f}")
    else:
        D_high = compute_D_exact(f_func, x_eq, x_sad, tau, 1.0)
        D_low = compute_D_exact(f_func, x_eq, x_sad, tau, 0.005)
        print(f"{eps1:6.2f} {eps2:6.2f} {eps_product:8.4f} {D_product:8.2f} {b0:8.4f} {'YES':>10s} {x_eq:8.4f} {x_sad:8.4f} {DPhi:12.6e} {'no σ*':>10s}  D∈[{D_high:.1f},{D_low:.1f}]")


# ================================================================
# PART 3: The critical question — what happens to the barrier near ε→1?
# ================================================================
print("\n\n" + "=" * 70)
print("PART 3: Barrier and equilibrium structure as ε₁ → 1.0")
print("=" * 70)
print("Fine sweep: does the barrier vanish at ε = 1 (saddle-node)?")

eps2_fixed = 0.10
fine_eps1 = np.concatenate([
    np.arange(0.10, 0.80, 0.10),
    np.arange(0.80, 0.92, 0.02),
])

print(f"\n{'ε₁':>6s} {'b0':>8s} {'#roots':>7s} {'x_eq':>8s} {'x_sad':>8s} {'Δx':>8s} {'ΔΦ':>12s} {'|λ_eq|':>8s} {'|λ_sad|':>8s}")
print("-" * 90)

for eps1 in fine_eps1:
    c1, c2, b0 = calibrate_2ch(eps1, eps2_fixed)
    if b0 < 0:
        print(f"{eps1:6.2f} {b0:8.4f} {'---':>7s}  b0 negative")
        continue

    f_func = make_drift_2ch(c1, c2, b0)
    roots = find_equilibria(f_func)

    if len(roots) < 3:
        print(f"{eps1:6.2f} {b0:8.4f} {len(roots):>7d}  bistability lost (roots: {[f'{r:.4f}' for r in roots]})")
        continue

    stabilities = [(r, fderiv(f_func, r)) for r in roots]
    stable = [(r, fp) for r, fp in stabilities if fp < 0]
    unstable = [(r, fp) for r, fp in stabilities if fp > 0]

    if len(stable) < 1 or len(unstable) < 1:
        print(f"{eps1:6.2f} {b0:8.4f} {len(roots):>7d}  no clear stable+saddle")
        continue

    x_eq = stable[0][0]
    x_sad = unstable[0][0]
    lam_eq = abs(fderiv(f_func, x_eq))
    lam_sad = abs(fderiv(f_func, x_sad))

    DPhi, _ = quad(lambda x: -f_func(x), x_eq, x_sad)

    print(f"{eps1:6.2f} {b0:8.4f} {len(roots):>7d} {x_eq:8.4f} {x_sad:8.4f} {x_sad-x_eq:8.4f} {DPhi:12.6e} {lam_eq:8.4f} {lam_sad:8.4f}")


# ================================================================
# PART 4: D_exact at fixed noise — what D values are actually reachable?
# ================================================================
print("\n\n" + "=" * 70)
print("PART 4: D_exact at fixed noise levels as ε₁ increases")
print("=" * 70)
print("If we fix σ (environment doesn't change), how does D_exact respond?")

sigma_fixed_values = [0.05, 0.10, 0.20, 0.50]

for sigma_fixed in sigma_fixed_values:
    print(f"\n--- σ = {sigma_fixed} ---")
    print(f"{'ε₁':>6s} {'D_prod':>8s} {'D_exact':>10s} {'D<1?':>6s}")

    for eps1 in [0.10, 0.30, 0.50, 0.70, 0.80, 0.85, 0.88, 0.90]:
        eps_product = eps1 * eps2_fixed
        D_product = 1.0 / eps_product

        c1, c2, b0 = calibrate_2ch(eps1, eps2_fixed)
        if b0 < 0:
            continue

        f_func = make_drift_2ch(c1, c2, b0)
        roots = find_equilibria(f_func)

        if len(roots) < 3:
            print(f"{eps1:6.2f} {D_product:8.1f} {'---':>10s}  no bistability")
            continue

        stabilities = [(r, fderiv(f_func, r)) for r in roots]
        stable = [(r, fp) for r, fp in stabilities if fp < 0]
        unstable = [(r, fp) for r, fp in stabilities if fp > 0]

        if len(stable) < 1 or len(unstable) < 1:
            continue

        x_eq = stable[0][0]
        x_sad = unstable[0][0]
        lam_eq = stable[0][1]
        tau = 1.0 / abs(lam_eq)

        D_ex = compute_D_exact(f_func, x_eq, x_sad, tau, sigma_fixed)
        if np.isnan(D_ex):
            print(f"{eps1:6.2f} {D_product:8.1f} {'NaN':>10s}")
        else:
            below = "YES!" if D_ex < 1.0 else "no"
            print(f"{eps1:6.2f} {D_product:8.1f} {D_ex:10.3f} {below:>6s}")


# ================================================================
# PART 5: Summary
# ================================================================
print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Key questions answered:
1. Does bistability survive when one ε > 1?
2. Can D_exact (Kramers MFPT) go below 1?
3. What is the minimum D_exact achievable in a bistable system?
4. Is D < 1 a physical state or a mathematical impossibility?

If bistability is always lost BEFORE D reaches 1:
  → D ∈ [1, ∞) is the natural domain
  → D < 1 means "the equilibrium doesn't exist"

If bistability survives with D < 1:
  → D ∈ (0, ∞) is the natural domain
  → D < 1 means "exists but self-destructs faster than it relaxes"
""")
