#!/usr/bin/env python3
"""
Financial Cusp Catastrophe: Kramers Escape Time Analysis

Model: Stochastic cusp catastrophe for market regime persistence
  dx/dt = -(x^3 - q*x + a) + sigma*dW
  V(x) = x^4/4 - q*x^2/2 + a*x

Based on:
  - Zeeman (1974): cusp catastrophe in stock markets
  - Diks & Wang (2009): stochastic cusp fitted to S&P 500

Computes:
  1. Equilibria and saddle for various (q, a) parameter sets
  2. Barrier height DeltaPhi (analytic)
  3. Exact MFPT integral -> D_exact at various sigma
  4. Kramers prefactor and D_Kramers
  5. D(sigma) scan -> find what sigma gives D_observed ~ 15-45
  6. Implied barrier-to-noise ratio comparison with ecology

No product equation (k=0 channels). Pure Kramers test.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import time

# ================================================================
# Cusp Model
# ================================================================

def f_cusp(x, q, a):
    """Cusp catastrophe drift: dx/dt = -(x^3 - q*x + a)

    Potential: V(x) = x^4/4 - q*x^2/2 + a*x
    Drift = -dV/dx = -(x^3 - q*x + a) = -x^3 + q*x - a
    """
    return -x**3 + q * x - a


def V_cusp(x, q, a):
    """Cusp potential: V(x) = x^4/4 - q*x^2/2 + a*x"""
    return x**4 / 4.0 - q * x**2 / 2.0 + a * x


def cusp_bistability_condition(q, a):
    """Cusp has two stable states when q > 0 and |a| < a_crit.
    a_crit = (2/3) * (q/3)^(3/2) * 2 = 2*sqrt(3)/9 * q^(3/2)
    """
    if q <= 0:
        return False, 0.0
    a_crit = 2.0 * np.sqrt(3.0) / 9.0 * q**1.5
    return abs(a) < a_crit, a_crit


# ================================================================
# Utilities (same pattern as step8)
# ================================================================

def find_equilibria(f_func, x_lo=-5.0, x_hi=5.0, N=400000):
    """Find all roots of f_func in [x_lo, x_hi]."""
    x_scan = np.linspace(x_lo, x_hi, N)
    f_scan = np.array([f_func(x) for x in x_scan])
    sign_changes = np.where(np.diff(np.sign(f_scan)))[0]
    roots = []
    for i in sign_changes:
        try:
            root = brentq(f_func, x_scan[i], x_scan[i + 1], xtol=1e-12)
            roots.append(root)
        except ValueError:
            pass
    return sorted(roots)


def fderiv(f_func, x, dx=1e-7):
    """Numerical derivative."""
    return (f_func(x + dx) - f_func(x - dx)) / (2.0 * dx)


def compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma, N=100000):
    """Compute exact D = MFPT/tau via 1D MFPT integral.

    Handles both directions: x_eq < x_saddle (escape rightward)
    and x_eq > x_saddle (escape leftward, e.g. cusp upper well).

    For leftward escape, we reflect: y = -x, g(y) = -f(-y),
    so the equilibrium is now lower than the saddle.
    """
    if x_eq > x_saddle:
        # Reflect coordinate: escape from upper well
        g_func = lambda y: -f_func(-y)
        y_eq = -x_eq       # now y_eq < y_sad
        y_sad = -x_saddle
        return compute_D_exact(g_func, y_eq, y_sad, tau_val, sigma, N)

    # Standard case: x_eq < x_saddle (escape rightward)
    margin = 0.5 * abs(x_saddle - x_eq)
    x_lo = x_eq - margin
    x_hi = x_saddle + 0.01 * abs(x_saddle - x_eq)
    xg = np.linspace(x_lo, x_hi, N)
    dx_grid = xg[1] - xg[0]

    # Quasi-potential via cumulative integral of -f
    neg_f = np.array([-f_func(x) for x in xg])
    U_raw = np.cumsum(neg_f) * dx_grid

    # Shift so U(x_eq) = 0
    i_eq = np.argmin(np.abs(xg - x_eq))
    U = U_raw - U_raw[i_eq]

    # Normalized potential
    Phi = 2.0 * U / sigma**2

    # Guard against overflow
    Phi_max = Phi.max()
    if Phi_max > 600:
        Phi = Phi - (Phi_max - 500)

    exp_neg = np.exp(-Phi)
    Ix = np.cumsum(exp_neg) * dx_grid
    psi = (2.0 / sigma**2) * np.exp(Phi) * Ix

    # Integrate from x_eq to x_saddle
    i_sad = np.argmin(np.abs(xg - x_saddle))
    MFPT = np.trapz(psi[i_eq:i_sad + 1], xg[i_eq:i_sad + 1])

    return MFPT / tau_val


def find_sigma_for_D(f_func, x_eq, x_saddle, tau_val, D_target,
                     sigma_lo=0.05, sigma_hi=5.0):
    """Find sigma where D_exact(sigma) = D_target via bisection."""
    # Ensure D_lo > D_target (small sigma -> large D)
    for _ in range(5):
        D_lo = compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma_lo)
        if np.isnan(D_lo) or np.isinf(D_lo):
            sigma_lo *= 2  # increase sigma_lo to avoid overflow
            continue
        if D_lo > D_target:
            break
        sigma_lo /= 2
    else:
        return None, 0, 0

    # Ensure D_hi < D_target (large sigma -> small D)
    for _ in range(5):
        D_hi = compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma_hi)
        if np.isnan(D_hi) or np.isinf(D_hi):
            sigma_hi *= 0.8
            continue
        if D_hi < D_target:
            break
        sigma_hi *= 2
    else:
        return None, 0, 0

    if np.isnan(D_lo) or np.isnan(D_hi) or D_lo < D_target or D_hi > D_target:
        return None, D_lo, D_hi

    def obj(s):
        val = compute_D_exact(f_func, x_eq, x_saddle, tau_val, s) - D_target
        return val if np.isfinite(val) else 1e30

    sigma_star = brentq(obj, sigma_lo, sigma_hi, xtol=1e-6, maxiter=100)
    return sigma_star, D_lo, D_hi


# ================================================================
# Parameter Sets
# ================================================================

# We test multiple (q, a) combinations spanning different barrier depths.
# q controls bistability strength, a controls asymmetry.
# a=0 is symmetric double-well. Small a tilts one well deeper.

PARAM_SETS = [
    # (name, q, a, description)
    ("Symmetric_shallow",  1.0, 0.0,   "Symmetric, shallow barrier"),
    ("Symmetric_moderate", 2.0, 0.0,   "Symmetric, moderate barrier"),
    ("Symmetric_deep",     4.0, 0.0,   "Symmetric, deep barrier"),
    ("Tilted_moderate",    2.0, 0.2,   "Tilted, moderate — crisis well shallower"),
    ("Tilted_deep",        4.0, 0.5,   "Tilted, deep — crisis well shallower"),
    ("Near_fold",          1.0, 0.35,  "Near fold bifurcation — barely bistable"),
]

# D_observed targets from the financial crisis research
D_TARGETS = {
    "D_low":  11,   # Post-2008 to COVID
    "D_mid":  20,   # Central Great Moderation estimate
    "D_high": 45,   # Upper bound from Aruoba-Schorfheide
}

# Ecological comparison values
ECO_BARRIERS = {
    "Savanna": 4.22,
    "Lake":    4.25,
    "Kelp":    2.25,
}


# ================================================================
# Main Analysis
# ================================================================

def analyze_param_set(name, q, a, desc):
    """Full Kramers analysis for one (q, a) parameter set."""
    print(f"\n{'='*70}")
    print(f"  {name}: q={q}, a={a}")
    print(f"  {desc}")
    print(f"{'='*70}")

    # Check bistability condition
    is_bistable, a_crit = cusp_bistability_condition(q, a)
    print(f"  Bistability: {is_bistable} (|a|={abs(a):.3f} < a_crit={a_crit:.4f})")
    if not is_bistable:
        print("  SKIPPING — not bistable.")
        return None

    # Define drift for this parameter set
    f = lambda x: f_cusp(x, q, a)

    # Find equilibria
    roots = find_equilibria(f, x_lo=-3.0 * np.sqrt(q), x_hi=3.0 * np.sqrt(q))
    print(f"  Roots found: {len(roots)}")
    if len(roots) < 3:
        print(f"  WARNING: Expected 3 roots, found {len(roots)}. Roots: {roots}")
        if len(roots) < 2:
            print("  SKIPPING — insufficient roots.")
            return None

    # Classify stability
    stab = [(r, fderiv(f, r)) for r in roots]
    for r, fp in stab:
        s = "STABLE" if fp < 0 else "UNSTABLE"
        print(f"    x = {r:+.8f}, f'(x) = {fp:+.6f} [{s}]")

    stable = sorted([r for r, fp in stab if fp < 0])
    unstable = sorted([r for r, fp in stab if fp > 0])

    if len(stable) < 2 or len(unstable) < 1:
        print("  SKIPPING — need 2 stable + 1 unstable.")
        return None

    x_low = stable[0]       # Lower stable (e.g., "crisis" state)
    x_high = stable[-1]     # Upper stable (e.g., "normal" state)
    x_sad = unstable[0]     # Saddle between them

    print(f"\n  Lower eq:  x_low  = {x_low:+.8f}")
    print(f"  Saddle:    x_sad  = {x_sad:+.8f}")
    print(f"  Upper eq:  x_high = {x_high:+.8f}")

    # Potential values
    V_low = V_cusp(x_low, q, a)
    V_sad = V_cusp(x_sad, q, a)
    V_high = V_cusp(x_high, q, a)

    # Barriers (from each well to saddle)
    DV_low_to_sad = V_sad - V_low     # Barrier from lower well
    DV_high_to_sad = V_sad - V_high   # Barrier from upper well

    print(f"\n  V(x_low)  = {V_low:+.8f}")
    print(f"  V(x_sad)  = {V_sad:+.8f}")
    print(f"  V(x_high) = {V_high:+.8f}")
    print(f"  DeltaV (low->sad)  = {DV_low_to_sad:.8f}")
    print(f"  DeltaV (high->sad) = {DV_high_to_sad:.8f}")

    # Verify barrier via integral of -f
    DPhi_low, _ = quad(lambda x: -f(x), x_low, x_sad)
    DPhi_high, _ = quad(lambda x: -f(x), x_high, x_sad)  # Note: integrate backward
    # For escape from upper well, integrate from x_high DOWN to x_sad
    DPhi_high_correct, _ = quad(lambda x: f(x), x_sad, x_high)

    print(f"  DPhi (integral, low->sad)  = {DPhi_low:.8f} (should match DV_low)")
    print(f"  DPhi (integral, high->sad) = {DPhi_high_correct:.8f} (should match DV_high)")

    # Kramers prefactor for escape from UPPER well (normal -> crisis)
    # This is the more relevant direction: how long does "normal" persist?
    lam_high = fderiv(f, x_high)
    lam_sad = fderiv(f, x_sad)

    tau_high = 1.0 / abs(lam_high)
    C_high = np.sqrt(abs(lam_high) * abs(lam_sad)) / (2 * np.pi)
    Ctau_high = C_high * tau_high
    inv_Ctau_high = 1.0 / Ctau_high

    print(f"\n  Escape from UPPER WELL (normal -> crisis):")
    print(f"    f'(x_high) = {lam_high:.6f}")
    print(f"    f'(x_sad)  = {lam_sad:+.6f}")
    print(f"    tau        = {tau_high:.6f}")
    print(f"    C          = {C_high:.6f}")
    print(f"    1/(C*tau)  = {inv_Ctau_high:.4f}")

    # Also compute for lower well (crisis -> normal)
    lam_low = fderiv(f, x_low)
    tau_low = 1.0 / abs(lam_low)
    C_low = np.sqrt(abs(lam_low) * abs(lam_sad)) / (2 * np.pi)
    inv_Ctau_low = 1.0 / (C_low * tau_low)

    print(f"\n  Escape from LOWER WELL (crisis -> normal):")
    print(f"    f'(x_low)  = {lam_low:.6f}")
    print(f"    tau        = {tau_low:.6f}")
    print(f"    1/(C*tau)  = {inv_Ctau_low:.4f}")

    # ================================================================
    # D(sigma) scan: escape from upper well
    # ================================================================
    print(f"\n  D(sigma) scan — escape from UPPER well:")
    print(f"  {'sigma':>10s} {'D_exact':>12s} {'D_Kramers':>12s} {'K_eff':>8s} {'2DV/s^2':>10s}")
    print(f"  {'-'*54}")

    K_SDE = 0.55
    results = []

    sigma_values = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])

    for sigma in sigma_values:
        barrier_ratio = 2.0 * DV_high_to_sad / sigma**2
        if barrier_ratio > 500:
            D_ex = float('inf')
            D_kr = float('inf')
            K_eff = float('nan')
        else:
            D_ex = compute_D_exact(f, x_high, x_sad, tau_high, sigma, N=100000)
            D_kr = K_SDE * np.exp(barrier_ratio) * inv_Ctau_high
            K_eff = D_ex / (np.exp(barrier_ratio) * inv_Ctau_high) if D_ex > 0 else float('nan')

        results.append((sigma, D_ex, D_kr, K_eff, barrier_ratio))

        if D_ex == float('inf'):
            print(f"  {sigma:10.4f} {'overflow':>12s} {'overflow':>12s} {'---':>8s} {barrier_ratio:10.2f}")
        else:
            print(f"  {sigma:10.4f} {D_ex:12.2f} {D_kr:12.2f} {K_eff:8.3f} {barrier_ratio:10.4f}")

    # ================================================================
    # Find sigma that gives D_observed for each target
    # ================================================================
    print(f"\n  Sigma required for D_observed targets (escape from upper well):")
    print(f"  {'Target':>10s} {'D_target':>10s} {'sigma_req':>12s} {'2DV/s^2':>10s} {'Compare':>30s}")
    print(f"  {'-'*74}")

    for tname, D_target in D_TARGETS.items():
        sigma_req, _, _ = find_sigma_for_D(f, x_high, x_sad, tau_high, D_target,
                                           sigma_lo=0.01, sigma_hi=10.0)
        if sigma_req is not None:
            barrier_ratio_req = 2.0 * DV_high_to_sad / sigma_req**2
            D_verify = compute_D_exact(f, x_high, x_sad, tau_high, sigma_req)

            # Compare to ecology
            eco_compare = ""
            for ename, eb in ECO_BARRIERS.items():
                if abs(barrier_ratio_req - eb) / eb < 0.3:
                    eco_compare += f" ~{ename}({eb:.1f})"

            print(f"  {tname:>10s} {D_target:10d} {sigma_req:12.6f} {barrier_ratio_req:10.4f} {eco_compare}")
        else:
            print(f"  {tname:>10s} {D_target:10d} {'FAILED':>12s}")

    return {
        'name': name, 'q': q, 'a': a,
        'x_low': x_low, 'x_sad': x_sad, 'x_high': x_high,
        'DV_low': DV_low_to_sad, 'DV_high': DV_high_to_sad,
        'tau_high': tau_high, 'inv_Ctau_high': inv_Ctau_high,
        'tau_low': tau_low, 'inv_Ctau_low': inv_Ctau_low,
        'results': results,
    }


# ================================================================
# Run all parameter sets
# ================================================================
if __name__ == '__main__':
    t0 = time.time()

    print("=" * 70)
    print("FINANCIAL CUSP CATASTROPHE: KRAMERS ESCAPE TIME ANALYSIS")
    print("=" * 70)
    print()
    print("Model: dx/dt = -x^3 + q*x - a")
    print("Potential: V(x) = x^4/4 - q*x^2/2 + a*x")
    print()
    print("D_observed targets from financial crisis research:")
    for k, v in D_TARGETS.items():
        print(f"  {k}: D = {v}")
    print()
    print("Ecological barrier-to-noise ratios for comparison:")
    for k, v in ECO_BARRIERS.items():
        print(f"  {k}: 2*DeltaPhi/sigma^2 = {v}")

    all_results = {}
    for name, q, a, desc in PARAM_SETS:
        result = analyze_param_set(name, q, a, desc)
        if result is not None:
            all_results[name] = result

    # ================================================================
    # Summary
    # ================================================================
    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Scenario':<25s} {'q':>5s} {'a':>5s} {'DV_high':>10s} {'1/(Ct)':>8s} "
          f"{'sig(D=20)':>10s} {'2DV/s2':>8s} {'~Eco?':>12s}")
    print("-" * 85)

    for name, res in all_results.items():
        # Find sigma for D=20
        f = lambda x: f_cusp(x, res['q'], res['a'])
        sigma_20, _, _ = find_sigma_for_D(f, res['x_high'], res['x_sad'],
                                          res['tau_high'], 20.0,
                                          sigma_lo=0.01, sigma_hi=10.0)
        if sigma_20 is not None:
            br = 2.0 * res['DV_high'] / sigma_20**2
            eco = ""
            for ename, eb in ECO_BARRIERS.items():
                if abs(br - eb) / eb < 0.25:
                    eco = f"~{ename}"
                    break
            print(f"{name:<25s} {res['q']:5.1f} {res['a']:5.2f} {res['DV_high']:10.6f} "
                  f"{res['inv_Ctau_high']:8.3f} {sigma_20:10.6f} {br:8.4f} {eco:>12s}")
        else:
            print(f"{name:<25s} {res['q']:5.1f} {res['a']:5.2f} {res['DV_high']:10.6f} "
                  f"{res['inv_Ctau_high']:8.3f} {'FAILED':>10s}")

    print(f"\n\nKey result: the barrier-to-noise ratio 2*DV/sigma^2 required to produce")
    print(f"D_observed ~ 20 (Great Moderation persistence). Compare to ecology:")
    print(f"  Savanna: 4.22")
    print(f"  Lake:    4.25")
    print(f"  Kelp:    2.25")
    print(f"\nIf the financial cusp gives 2*DV/sigma^2 ~ 3-5, the financial system")
    print(f"operates in the same barrier regime as ecological bistable systems.")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f} seconds")
