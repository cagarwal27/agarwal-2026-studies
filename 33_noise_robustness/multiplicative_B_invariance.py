#!/usr/bin/env python3
"""
MULTIPLICATIVE NOISE B INVARIANCE TEST — LAKE MODEL
=====================================================
Addresses reviewer concern 1.1/2.6: "Demonstrate that B invariance holds
under multiplicative noise g(x) = sigma*sqrt(x) for at least one ecological
system."

Model: 1D lake ODE f(x) = a - bx + rx^q/(x^q + h^q)
       with multiplicative noise: dx = f(x)dt + sigma*sqrt(x)*dW  (Ito)

For state-dependent diffusion g(x) = sigma*sqrt(x), the MFPT formula
uses a modified quasipotential:

    U_mult(x) = -integral f(x)/x dx

and the MFPT from x_eq to x_sad is:

    T = integral_{x_eq}^{x_sad} exp(Phi(y)) * [integral_{x_ref}^{y}
        (2/(sigma^2 * z)) exp(-Phi(z)) dz] dy

where Phi(x) = 2*U_mult(x)/sigma^2.

B_mult = 2*DeltaU_mult / sigma*^2 is the natural barrier-to-noise ratio.

Citation: van Nes & Scheffer, "Implications of spatial heterogeneity for
catastrophic regime shifts in ecosystems," Ecology 86(7):1797-1807, 2005.
Lake model params: b=0.8, r=1.0, h=1.0, q=8.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Model: 1D lake  f(x) = a - bx + rx^q / (x^q + 1)
# =============================================================================
B_LOSS = 0.8
R_MAX = 1.0
H_SAT = 1.0
D_TARGET = 200

def f_lake(x, a, q):
    return a - B_LOSS * x + R_MAX * x**q / (x**q + H_SAT**q)

def f_lake_vec(x_arr, a, q):
    return a - B_LOSS * x_arr + R_MAX * x_arr**q / (x_arr**q + H_SAT**q)

def f_lake_deriv(x, a, q):
    return -B_LOSS + R_MAX * q * x**(q-1) * H_SAT**q / (x**q + H_SAT**q)**2

def find_roots(a, q, x_lo=0.001, x_hi=4.0, n_scan=5000):
    xs = np.linspace(x_lo, x_hi, n_scan)
    fs = f_lake_vec(xs, a, q)
    roots = []
    sign_changes = np.where(fs[:-1] * fs[1:] < 0)[0]
    for i in sign_changes:
        try:
            root = brentq(lambda x, a_=a, q_=q: f_lake(x, a_, q_),
                          xs[i], xs[i+1], xtol=1e-14)
            if not any(abs(root - r) < 1e-8 for r in roots):
                roots.append(root)
        except Exception:
            pass
    return sorted(roots)

def find_bistable_range(q, a_lo=0.01, a_hi=0.8, n_scan=1000):
    a_vals = np.linspace(a_lo, a_hi, n_scan)
    bistable = []
    for a in a_vals:
        xs = np.linspace(0.001, 4.0, 3000)
        fs = f_lake_vec(xs, a, q)
        if np.sum(fs[:-1] * fs[1:] < 0) == 3:
            bistable.append(a)
    if len(bistable) < 2:
        return None, None
    return bistable[0], bistable[-1]


# =============================================================================
# Multiplicative noise barrier and MFPT
# =============================================================================

def compute_barrier_mult(a, q, x_eq, x_sad):
    """Multiplicative quasipotential barrier: DeltaU_m = -integral f(x)/x dx
    from x_eq to x_sad."""
    result, _ = quad(lambda x: -f_lake(x, a, q) / x, x_eq, x_sad,
                     limit=200, epsabs=1e-14, epsrel=1e-12)
    return result

def compute_barrier_add(a, q, x_eq, x_sad):
    """Standard additive barrier for comparison: DeltaPhi = -integral f(x) dx."""
    result, _ = quad(lambda x: -f_lake(x, a, q), x_eq, x_sad,
                     limit=200, epsabs=1e-14, epsrel=1e-12)
    return result

def compute_D_exact_mult(a, q, x_eq, x_sad, tau, sigma, N=80000):
    """Exact MFPT-based D for multiplicative noise g(x) = sigma*sqrt(x).

    The MFPT for dx = f(x)dt + sigma*sqrt(x)*dW (Ito) is:

    T = integral exp(Phi(y)) * [integral (2/(sigma^2*z)) exp(-Phi(z)) dz] dy

    where Phi(x) = 2*U_mult(x)/sigma^2, U_mult(x) = -integral f(x)/x dx.
    """
    margin = 0.05 * (x_sad - x_eq)
    x_lo = max(0.01, x_eq - margin)   # stay away from 0 (1/x singularity)
    x_hi = x_sad + margin
    xg = np.linspace(x_lo, x_hi, N)
    dx = xg[1] - xg[0]

    # Build modified potential: U_mult via -f(x)/x
    neg_f_over_x = -f_lake_vec(xg, a, q) / xg
    U_raw = np.cumsum(neg_f_over_x) * dx
    i_eq = np.argmin(np.abs(xg - x_eq))
    U = U_raw - U_raw[i_eq]
    Phi = 2.0 * U / sigma**2

    Phi_max = Phi.max()
    if Phi_max > 700:
        return np.inf

    Phi = np.clip(Phi, -500, 700)

    # Inner integral: cumulative sum of (1/z)*exp(-Phi(z))
    inner = (1.0 / xg) * np.exp(-Phi)
    Ix = np.cumsum(inner) * dx

    # MFPT integrand: (2/sigma^2) * exp(Phi(y)) * Ix(y)
    psi = (2.0 / sigma**2) * np.exp(Phi) * Ix

    i_sad = np.argmin(np.abs(xg - x_sad))
    lo = min(i_eq, i_sad)
    hi = max(i_eq, i_sad)
    MFPT = np.trapz(psi[lo:hi+1], xg[lo:hi+1])
    return MFPT / tau

def compute_D_exact_add(a, q, x_eq, x_sad, tau, sigma, N=80000):
    """Standard additive noise MFPT-based D for comparison."""
    margin = 0.05 * (x_sad - x_eq)
    x_lo = max(0.001, x_eq - margin)
    x_hi = x_sad + margin
    xg = np.linspace(x_lo, x_hi, N)
    dx = xg[1] - xg[0]

    neg_f = -f_lake_vec(xg, a, q)
    U_raw = np.cumsum(neg_f) * dx
    i_eq = np.argmin(np.abs(xg - x_eq))
    U = U_raw - U_raw[i_eq]
    Phi = 2.0 * U / sigma**2

    Phi_max = Phi.max()
    if Phi_max > 700:
        return np.inf
    Phi = np.clip(Phi, -500, 700)

    exp_neg = np.exp(-Phi)
    Ix = np.cumsum(exp_neg) * dx
    psi = (2.0 / sigma**2) * np.exp(Phi) * Ix

    i_sad = np.argmin(np.abs(xg - x_sad))
    lo = min(i_eq, i_sad)
    hi = max(i_eq, i_sad)
    MFPT = np.trapz(psi[lo:hi+1], xg[lo:hi+1])
    return MFPT / tau


def find_sigma_star(a, q, x_eq, x_sad, lam_eq, D_target, noise_type='mult'):
    """Find sigma where D_exact(sigma) = D_target via bisection in log-space."""
    tau = 1.0 / abs(lam_eq)

    if noise_type == 'mult':
        DeltaU = compute_barrier_mult(a, q, x_eq, x_sad)
        compute_D = lambda sig: compute_D_exact_mult(a, q, x_eq, x_sad, tau, sig)
    else:
        DeltaU = compute_barrier_add(a, q, x_eq, x_sad)
        compute_D = lambda sig: compute_D_exact_add(a, q, x_eq, x_sad, tau, sig)

    # Initial guess from Kramers: D ~ exp(2*DeltaU/sigma^2)/(C*tau)
    lam_sad = f_lake_deriv(x_sad, a, q)
    C = abs(lam_sad) / (2 * np.pi) * np.sqrt(abs(lam_eq) / abs(lam_sad))
    arg = D_target * C * tau
    if arg > 1 and DeltaU > 0:
        sig_guess = np.sqrt(2 * DeltaU / np.log(arg))
    else:
        sig_guess = 0.1

    def objective(log_sigma):
        sigma = np.exp(log_sigma)
        D = compute_D(sigma)
        if D == np.inf or D > 1e15:
            return 50.0
        if D <= 0:
            return -50.0
        return np.log(max(D, 1e-30)) - np.log(D_target)

    log_guess = np.log(max(sig_guess, 1e-6))
    test_pts = np.linspace(log_guess - 3.0, log_guess + 3.0, 50)
    obj_vals = [objective(lp) for lp in test_pts]

    bracket_lo, bracket_hi = None, None
    for i in range(len(obj_vals) - 1):
        if obj_vals[i] > 0 and obj_vals[i+1] <= 0:
            bracket_lo = test_pts[i]
            bracket_hi = test_pts[i+1]
            break

    if bracket_lo is None:
        test_pts2 = np.linspace(np.log(1e-4), np.log(3.0), 120)
        obj_vals2 = [objective(lp) for lp in test_pts2]
        for i in range(len(obj_vals2) - 1):
            if obj_vals2[i] > 0 and obj_vals2[i+1] <= 0:
                bracket_lo = test_pts2[i]
                bracket_hi = test_pts2[i+1]
                break
        if bracket_lo is None:
            return np.nan

    try:
        log_sig = brentq(objective, bracket_lo, bracket_hi, xtol=1e-12, maxiter=300)
        return np.exp(log_sig)
    except Exception:
        return np.nan


# =============================================================================
# STEP 1: Multiplicative noise B sweep across bistable range
# =============================================================================
def run_mult_sweep(q_val, n_pts=30, label=""):
    print(f"\n{'='*78}")
    print(f"STEP 1 ({label}): Multiplicative noise B(a) sweep at q={q_val}, n={n_pts}")
    print(f"Noise model: dx = f(x)dt + sigma*sqrt(x)*dW  (Ito)")
    print(f"{'='*78}")

    a_low, a_high = find_bistable_range(q_val)
    if a_low is None:
        print("  ERROR: No bistable range found.")
        return None
    print(f"  Bistable range: [{a_low:.6f}, {a_high:.6f}]")

    margin = 0.05 * (a_high - a_low)
    a_vals = np.linspace(a_low + margin, a_high - margin, n_pts)

    results = []
    for idx, a in enumerate(a_vals):
        roots = find_roots(a, q_val)
        if len(roots) < 3:
            continue
        x_eq, x_sad, x_turb = roots[0], roots[1], roots[2]
        lam_eq = f_lake_deriv(x_eq, a, q_val)
        lam_sad = f_lake_deriv(x_sad, a, q_val)
        tau = 1.0 / abs(lam_eq)

        DeltaU_mult = compute_barrier_mult(a, q_val, x_eq, x_sad)
        DeltaPhi_add = compute_barrier_add(a, q_val, x_eq, x_sad)
        if DeltaU_mult <= 0:
            continue

        sigma_star = find_sigma_star(a, q_val, x_eq, x_sad, lam_eq, D_TARGET, 'mult')
        if np.isnan(sigma_star):
            continue

        B_mult = 2 * DeltaU_mult / sigma_star**2
        beta_mult = np.log(D_TARGET) - B_mult
        eig_ratio = abs(lam_eq) / abs(lam_sad)

        # Also get additive for comparison
        sigma_star_add = find_sigma_star(a, q_val, x_eq, x_sad, lam_eq, D_TARGET, 'add')
        B_add = 2 * DeltaPhi_add / sigma_star_add**2 if not np.isnan(sigma_star_add) else np.nan

        results.append({
            'a': a, 'DeltaU_mult': DeltaU_mult, 'DeltaPhi_add': DeltaPhi_add,
            'lam_eq': lam_eq, 'lam_sad': lam_sad, 'eig_ratio': eig_ratio,
            'sigma_star_mult': sigma_star, 'B_mult': B_mult, 'beta_mult': beta_mult,
            'sigma_star_add': sigma_star_add, 'B_add': B_add,
            'x_eq': x_eq, 'x_sad': x_sad,
        })
        if (idx + 1) % 5 == 0:
            print(f"    [{idx+1}/{n_pts}] a={a:.5f}  DeltaU_m={DeltaU_mult:.6e}  "
                  f"B_mult={B_mult:.4f}  sigma*={sigma_star:.5f}")

    if not results:
        print("  ERROR: No valid points.")
        return None

    B_mults = np.array([r['B_mult'] for r in results])
    B_adds = np.array([r['B_add'] for r in results if not np.isnan(r['B_add'])])
    betas = np.array([r['beta_mult'] for r in results])
    DUms = np.array([r['DeltaU_mult'] for r in results])
    DPas = np.array([r['DeltaPhi_add'] for r in results])

    print(f"\n  --- Multiplicative noise results (q={q_val}, {len(results)} points) ---")
    print(f"  DeltaU_mult: range [{DUms.min():.6e}, {DUms.max():.6e}], "
          f"variation = {DUms.max()/DUms.min():.1f}x")
    print(f"  B_mult:      mean={B_mults.mean():.4f}, std={B_mults.std():.4f}, "
          f"CV={100*B_mults.std()/B_mults.mean():.2f}%")
    print(f"  beta_mult:   mean={betas.mean():.4f}, std={betas.std():.4f}")
    if len(B_adds) > 0:
        print(f"  B_add (ref): mean={B_adds.mean():.4f}, CV={100*B_adds.std()/B_adds.mean():.2f}%")

    # Full table
    print(f"\n  {'a':>9} | {'DeltaU_m':>11} | {'DeltaPhi':>11} | "
          f"{'B_mult':>7} | {'B_add':>7} | {'sigma*_m':>8} | {'sigma*_a':>8} | {'beta_m':>7}")
    print(f"  {'-'*9}-+-{'-'*11}-+-{'-'*11}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")
    for r in results:
        B_a_str = f"{r['B_add']:7.4f}" if not np.isnan(r['B_add']) else "    NaN"
        s_a_str = f"{r['sigma_star_add']:8.5f}" if not np.isnan(r['sigma_star_add']) else "     NaN"
        print(f"  {r['a']:9.6f} | {r['DeltaU_mult']:11.6e} | {r['DeltaPhi_add']:11.6e} | "
              f"{r['B_mult']:7.4f} | {B_a_str} | {r['sigma_star_mult']:8.5f} | {s_a_str} | "
              f"{r['beta_mult']:7.4f}")

    return {
        'q': q_val, 'results': results,
        'B_mult_mean': B_mults.mean(), 'B_mult_std': B_mults.std(),
        'B_mult_cv': 100 * B_mults.std() / B_mults.mean(),
        'B_add_mean': B_adds.mean() if len(B_adds) > 0 else np.nan,
        'B_add_cv': 100 * B_adds.std() / B_adds.mean() if len(B_adds) > 0 else np.nan,
        'beta_mean': betas.mean(), 'beta_std': betas.std(),
        'DU_variation': DUms.max() / DUms.min(),
    }


# =============================================================================
# STEP 2: Multi-q generalization
# =============================================================================
def run_multi_q(q_values, n_pts=25):
    print(f"\n{'='*78}")
    print(f"STEP 2: Multiplicative noise B across q = {q_values}")
    print(f"{'='*78}")

    summary_rows = []
    for q in q_values:
        t0 = time.time()
        data = run_mult_sweep(q, n_pts=n_pts, label=f"q={q}")
        elapsed = time.time() - t0
        if data is not None:
            summary_rows.append({
                'q': q,
                'B_mult_mean': data['B_mult_mean'],
                'B_mult_cv': data['B_mult_cv'],
                'B_add_mean': data['B_add_mean'],
                'B_add_cv': data['B_add_cv'],
                'beta_mean': data['beta_mean'],
                'beta_std': data['beta_std'],
                'DU_variation': data['DU_variation'],
                'n_pts': len(data['results']),
                'elapsed': elapsed,
            })
            print(f"  q={q}: done in {elapsed:.1f}s")

    return summary_rows


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    t_start = time.time()

    print("=" * 78)
    print("MULTIPLICATIVE NOISE B INVARIANCE TEST")
    print("Lake model: f(x) = a - bx + rx^q/(x^q+1)")
    print("Noise: g(x) = sigma * sqrt(x)  [Ito interpretation]")
    print("Barrier: DeltaU_m = -integral f(x)/x dx")
    print("B_mult = 2*DeltaU_m / sigma*^2")
    print("=" * 78)

    # Step 1: Detailed sweep at q=8 (standard lake)
    step1 = run_mult_sweep(8, n_pts=35, label="Main")

    # Step 2: Generalize across Hill coefficients
    q_values = [3, 5, 8, 10, 16]
    summary = run_multi_q(q_values, n_pts=25)

    # =========================================================================
    # FINAL SYNTHESIS
    # =========================================================================
    print(f"\n{'='*78}")
    print(f"FINAL SYNTHESIS: MULTIPLICATIVE NOISE B INVARIANCE")
    print(f"{'='*78}")

    if summary:
        print(f"\n  {'q':>3} | {'B_mult mean':>11} | {'B_mult CV%':>10} | "
              f"{'B_add mean':>10} | {'B_add CV%':>9} | {'DU_m var':>8} | {'n':>3}")
        print(f"  {'-'*3}-+-{'-'*11}-+-{'-'*10}-+-{'-'*10}-+-{'-'*9}-+-{'-'*8}-+-{'-'*3}")
        for row in summary:
            B_add_str = f"{row['B_add_mean']:>10.3f}" if not np.isnan(row['B_add_mean']) else "       NaN"
            B_add_cv_str = f"{row['B_add_cv']:>8.2f}%" if not np.isnan(row['B_add_cv']) else "      NaN"
            print(f"  {row['q']:>3} | {row['B_mult_mean']:>11.3f} | "
                  f"{row['B_mult_cv']:>9.2f}% | {B_add_str} | {B_add_cv_str} | "
                  f"{row['DU_variation']:>7.0f}x | {row['n_pts']:>3}")

        max_cv = max(row['B_mult_cv'] for row in summary)
        all_B_means = [row['B_mult_mean'] for row in summary]

        print(f"\n  Maximum B_mult CV across all q: {max_cv:.2f}%")
        print(f"  B_mult range: [{min(all_B_means):.3f}, {max(all_B_means):.3f}]")

        if max_cv < 5.0:
            print(f"\n  RESULT: B INVARIANCE HOLDS UNDER MULTIPLICATIVE NOISE")
            print(f"  B_mult CV < 5% for all q values tested.")
            print(f"  The barrier-to-noise ratio is robust to state-dependent diffusion.")
        elif max_cv < 10.0:
            print(f"\n  RESULT: B_mult CV < 10% — moderate invariance under multiplicative noise")
        else:
            print(f"\n  RESULT: B_mult CV > 10% — B invariance weakens under multiplicative noise")

        # Stability window check
        print(f"\n  Stability window check:")
        for row in summary:
            in_window = 1.8 <= row['B_mult_mean'] <= 6.0
            print(f"    q={row['q']:>2}: B_mult = {row['B_mult_mean']:.3f}  "
                  f"{'IN [1.8, 6.0]' if in_window else 'OUTSIDE [1.8, 6.0]'}")

    elapsed = time.time() - t_start
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print(f"\n{'='*78}")
    print(f"END OF MULTIPLICATIVE NOISE TEST")
    print(f"{'='*78}")
