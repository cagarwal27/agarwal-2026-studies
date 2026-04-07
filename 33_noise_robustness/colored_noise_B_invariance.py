#!/usr/bin/env python3
"""
COLORED NOISE B INVARIANCE TEST — LAKE MODEL
==============================================
Addresses reviewer concern 2.6: "colored noise timescales comparable to
relaxation times."

Tests B invariance when the lake model is driven by Ornstein-Uhlenbeck
(exponentially correlated) noise with correlation time tau_c equal to the
system relaxation time.

System:
    dx = f(x) dt + eta(t) dt
    d(eta) = -(1/tau_c) eta dt + (sigma/tau_c) dW_eta

This gives <eta^2> = sigma^2/(2*tau_c), and in the white noise limit
(tau_c -> 0), the effective diffusion matches additive noise with intensity
sigma^2/2.

Method:
    1. Analytical: Kramers colored-noise correction (Hanggi-Talkner-Borkovec 1990)
    2. Numerical: Direct SDE simulation (Euler-Maruyama, vectorized)

Both approaches compute B = 2*DeltaPhi/sigma*^2 (using the deterministic
barrier, since colored noise does not change the potential).

Citation: Hanggi P, Talkner P, Borkovec M, "Reaction-rate theory: fifty years
after Kramers," Rev. Mod. Phys. 62:251-341, 1990.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Lake model (same as multiplicative_B_invariance.py)
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
    for i in np.where(fs[:-1] * fs[1:] < 0)[0]:
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

def compute_barrier_add(a, q, x_eq, x_sad):
    result, _ = quad(lambda x: -f_lake(x, a, q), x_eq, x_sad,
                     limit=200, epsabs=1e-14, epsrel=1e-12)
    return result

def compute_D_exact_add(a, q, x_eq, x_sad, tau, sigma, N=80000):
    """Standard additive white noise MFPT."""
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
    if Phi.max() > 700:
        return np.inf
    Phi = np.clip(Phi, -500, 700)

    exp_neg = np.exp(-Phi)
    Ix = np.cumsum(exp_neg) * dx
    psi = (2.0 / sigma**2) * np.exp(Phi) * Ix

    i_sad = np.argmin(np.abs(xg - x_sad))
    lo, hi = min(i_eq, i_sad), max(i_eq, i_sad)
    MFPT = np.trapz(psi[lo:hi+1], xg[lo:hi+1])
    return MFPT / tau


# =============================================================================
# METHOD 1: Kramers colored-noise correction (analytical)
# =============================================================================

def kramers_colored_correction(lam_sad, tau_c):
    """Compute the Kramers escape rate correction for colored noise.

    The positive eigenvalue at the saddle in the extended (x, eta) system is:
        lambda_+ = [-1/tau_c + sqrt(1/tau_c^2 + 4*omega_b^2)] / 2

    where omega_b^2 = |lam_sad| (curvature at saddle in potential units).

    The correction factor is: mu = lambda_+ / omega_b
    MFPT_colored = MFPT_white * (omega_b / lambda_+)

    Reference: Hanggi, Talkner, Borkovec (1990) Eq. 4.56b
    """
    omega_b = np.sqrt(abs(lam_sad))
    inv_tc = 1.0 / tau_c
    discriminant = inv_tc**2 + 4 * omega_b**2
    lambda_plus = (-inv_tc + np.sqrt(discriminant)) / 2.0
    mu = lambda_plus / omega_b
    return mu, lambda_plus, omega_b


def find_sigma_star_white(a, q, x_eq, x_sad, lam_eq, D_target):
    """Find sigma where D_exact_add = D_target."""
    tau = 1.0 / abs(lam_eq)
    DeltaPhi = compute_barrier_add(a, q, x_eq, x_sad)
    lam_sad = f_lake_deriv(x_sad, a, q)
    C = abs(lam_sad) / (2 * np.pi) * np.sqrt(abs(lam_eq) / abs(lam_sad))
    arg = D_target * C * tau
    if arg > 1 and DeltaPhi > 0:
        sig_guess = np.sqrt(2 * DeltaPhi / np.log(arg))
    else:
        sig_guess = 0.1

    def objective(log_sigma):
        sigma = np.exp(log_sigma)
        D = compute_D_exact_add(a, q, x_eq, x_sad, tau, sigma)
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
            bracket_lo, bracket_hi = test_pts[i], test_pts[i+1]
            break

    if bracket_lo is None:
        test_pts2 = np.linspace(np.log(1e-4), np.log(2.0), 120)
        obj_vals2 = [objective(lp) for lp in test_pts2]
        for i in range(len(obj_vals2) - 1):
            if obj_vals2[i] > 0 and obj_vals2[i+1] <= 0:
                bracket_lo, bracket_hi = test_pts2[i], test_pts2[i+1]
                break
        if bracket_lo is None:
            return np.nan

    try:
        return np.exp(brentq(objective, bracket_lo, bracket_hi, xtol=1e-12, maxiter=300))
    except Exception:
        return np.nan


def analytical_colored_sweep(q_val, n_pts=25):
    """Sweep loading, compute B_colored via Kramers correction analytically."""
    print(f"\n{'='*78}")
    print(f"METHOD 1 (Analytical): Kramers colored-noise correction at q={q_val}")
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
        x_eq, x_sad = roots[0], roots[1]
        lam_eq = f_lake_deriv(x_eq, a, q_val)
        lam_sad = f_lake_deriv(x_sad, a, q_val)
        tau = 1.0 / abs(lam_eq)
        tau_c = tau  # colored noise correlation time = relaxation time
        DeltaPhi = compute_barrier_add(a, q_val, x_eq, x_sad)
        if DeltaPhi <= 0:
            continue

        # White noise sigma*
        sigma_white = find_sigma_star_white(a, q_val, x_eq, x_sad, lam_eq, D_TARGET)
        if np.isnan(sigma_white):
            continue
        B_white = 2 * DeltaPhi / sigma_white**2

        # Colored noise correction
        mu, lambda_plus, omega_b = kramers_colored_correction(lam_sad, tau_c)

        # MFPT_colored = MFPT_white * (omega_b / lambda_plus)
        # D_colored(sigma) = D_white(sigma) * (omega_b / lambda_plus)
        # To get D_colored = D_TARGET, we need D_white = D_TARGET * (lambda_plus / omega_b)
        D_white_needed = D_TARGET * (lambda_plus / omega_b)

        # Find sigma where D_white = D_white_needed
        sigma_colored = find_sigma_star_white(a, q_val, x_eq, x_sad, lam_eq, D_white_needed)
        if np.isnan(sigma_colored):
            continue

        B_colored = 2 * DeltaPhi / sigma_colored**2
        beta_colored = np.log(D_TARGET) - B_colored

        results.append({
            'a': a, 'DeltaPhi': DeltaPhi,
            'lam_eq': lam_eq, 'lam_sad': lam_sad, 'tau_c': tau_c,
            'sigma_white': sigma_white, 'B_white': B_white,
            'sigma_colored': sigma_colored, 'B_colored': B_colored,
            'beta_colored': beta_colored,
            'mu': mu, 'lambda_plus': lambda_plus, 'omega_b': omega_b,
        })
        if (idx + 1) % 5 == 0:
            print(f"    [{idx+1}/{n_pts}] a={a:.5f}  B_white={B_white:.4f}  "
                  f"B_colored={B_colored:.4f}  mu={mu:.4f}")

    if not results:
        print("  ERROR: No valid points.")
        return None

    B_whites = np.array([r['B_white'] for r in results])
    B_coloreds = np.array([r['B_colored'] for r in results])
    mus = np.array([r['mu'] for r in results])

    print(f"\n  --- Analytical colored noise results (q={q_val}, tau_c=tau_relax) ---")
    print(f"  B_white:   mean={B_whites.mean():.4f}, CV={100*B_whites.std()/B_whites.mean():.2f}%")
    print(f"  B_colored: mean={B_coloreds.mean():.4f}, CV={100*B_coloreds.std()/B_coloreds.mean():.2f}%")
    print(f"  mu (rate correction): mean={mus.mean():.4f}, range=[{mus.min():.4f}, {mus.max():.4f}]")
    print(f"  B shift:   mean={B_coloreds.mean() - B_whites.mean():.4f}")

    # Table
    print(f"\n  {'a':>9} | {'DeltaPhi':>11} | {'B_white':>7} | {'B_colored':>9} | "
          f"{'sigma_w':>8} | {'sigma_c':>8} | {'mu':>6} | {'tau_c':>6}")
    print(f"  {'-'*9}-+-{'-'*11}-+-{'-'*7}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}")
    for r in results:
        print(f"  {r['a']:9.6f} | {r['DeltaPhi']:11.6e} | {r['B_white']:7.4f} | "
              f"{r['B_colored']:9.4f} | {r['sigma_white']:8.5f} | {r['sigma_colored']:8.5f} | "
              f"{r['mu']:6.4f} | {r['tau_c']:6.3f}")

    return {
        'q': q_val, 'results': results,
        'B_white_mean': B_whites.mean(), 'B_white_cv': 100 * B_whites.std() / B_whites.mean(),
        'B_colored_mean': B_coloreds.mean(), 'B_colored_cv': 100 * B_coloreds.std() / B_coloreds.mean(),
        'mu_mean': mus.mean(), 'mu_range': (mus.min(), mus.max()),
    }


# =============================================================================
# METHOD 2: Direct SDE simulation (verification at selected points)
# =============================================================================

def simulate_mfpt_colored(a, q, x_eq, x_sad, sigma, tau_c,
                          n_trials=200, dt=0.01, max_time=2000,
                          rng=None):
    """Simulate escape from x_eq past x_sad under colored noise.

    System:
        dx = f(x) dt + eta(t) dt
        d(eta) = -(1/tau_c) eta dt + (sigma/tau_c) dW_eta

    Returns mean MFPT and standard error.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    max_steps = int(max_time / dt)
    x = np.full(n_trials, x_eq)
    # Initialize eta from stationary distribution: eta ~ N(0, sigma^2/(2*tau_c))
    eta_std = sigma / np.sqrt(2 * tau_c)
    eta = rng.normal(0, eta_std, n_trials)

    escape_times = np.full(n_trials, np.inf)
    escaped = np.zeros(n_trials, dtype=bool)

    noise_amp = sigma / tau_c * np.sqrt(dt)
    decay = 1.0 / tau_c

    for step in range(1, max_steps + 1):
        t = step * dt

        # Update x
        f_vals = f_lake_vec(x, a, q)
        x += f_vals * dt + eta * dt

        # Reflect at x = 0.001 (phosphorus concentration >= 0)
        x = np.maximum(x, 0.001)

        # Update eta (OU process)
        dW = rng.normal(0, 1.0, n_trials)
        eta += -decay * eta * dt + noise_amp * dW

        # Check escape
        newly_escaped = (~escaped) & (x >= x_sad)
        escape_times[newly_escaped] = t
        escaped |= newly_escaped

        if escaped.all():
            break

    valid = escape_times < np.inf
    n_valid = valid.sum()
    if n_valid < 10:
        return np.inf, np.inf, n_valid

    mfpt = escape_times[valid].mean()
    se = escape_times[valid].std() / np.sqrt(n_valid)
    return mfpt, se, n_valid


def sde_verification(q_val, n_verify=3, n_trials=200):
    """Verify analytical Kramers correction with direct SDE at selected points.

    Uses points near the fold where the barrier is low (faster escape) to
    keep simulation tractable.
    """
    print(f"\n{'='*78}")
    print(f"METHOD 2 (SDE Verification): Direct simulation at {n_verify} loading values")
    print(f"n_trials={n_trials}, q={q_val}")
    print(f"{'='*78}")

    a_low, a_high = find_bistable_range(q_val)
    if a_low is None:
        print("  ERROR: No bistable range found.")
        return None

    # Use points in the upper part of bistable range (lower barrier = faster escape)
    margin_lo = 0.55 * (a_high - a_low)
    margin_hi = 0.05 * (a_high - a_low)
    a_vals = np.linspace(a_low + margin_lo, a_high - margin_hi, n_verify)

    rng = np.random.default_rng(42)
    results = []

    for idx, a in enumerate(a_vals):
        roots = find_roots(a, q_val)
        if len(roots) < 3:
            continue
        x_eq, x_sad = roots[0], roots[1]
        lam_eq = f_lake_deriv(x_eq, a, q_val)
        lam_sad = f_lake_deriv(x_sad, a, q_val)
        tau = 1.0 / abs(lam_eq)
        tau_c = tau
        DeltaPhi = compute_barrier_add(a, q_val, x_eq, x_sad)

        # White noise sigma* and B
        sigma_white = find_sigma_star_white(a, q_val, x_eq, x_sad, lam_eq, D_TARGET)
        if np.isnan(sigma_white):
            continue
        B_white = 2 * DeltaPhi / sigma_white**2

        # Use a slightly larger sigma to ensure most trials escape
        sigma_sim = sigma_white * 1.3

        # SDE: measure D at sigma_sim with colored noise
        print(f"  [{idx+1}/{n_verify}] a={a:.5f}, sigma={sigma_sim:.5f}, "
              f"simulating {n_trials} trials...")
        t0 = time.time()
        mfpt_sim, se_sim, n_valid = simulate_mfpt_colored(
            a, q_val, x_eq, x_sad, sigma_sim, tau_c,
            n_trials=n_trials, dt=0.01, max_time=2000, rng=rng)
        elapsed = time.time() - t0

        D_sim = mfpt_sim / tau if mfpt_sim < np.inf else np.inf
        D_white_at_sim_sigma = compute_D_exact_add(a, q_val, x_eq, x_sad, tau, sigma_sim)

        # Kramers correction prediction
        mu, _, omega_b = kramers_colored_correction(lam_sad, tau_c)
        D_colored_pred = D_white_at_sim_sigma / mu

        print(f"    D_white(exact)={D_white_at_sim_sigma:.1f}, D_colored(sim)={D_sim:.1f}, "
              f"D_colored(Kramers)={D_colored_pred:.1f}, "
              f"n_escaped={n_valid}/{n_trials}, time={elapsed:.1f}s")

        results.append({
            'a': a, 'DeltaPhi': DeltaPhi, 'tau_c': tau_c,
            'sigma_sim': sigma_sim, 'B_white': B_white,
            'D_white': D_white_at_sim_sigma, 'D_sim': D_sim, 'D_pred': D_colored_pred,
            'mfpt_sim': mfpt_sim, 'se_sim': se_sim, 'n_valid': n_valid,
            'mu': mu,
        })

    if results:
        print(f"\n  --- SDE verification summary ---")
        print(f"  {'a':>9} | {'D_white':>8} | {'D_sim':>8} | {'D_pred':>8} | "
              f"{'sim/pred':>8} | {'n_esc':>5}")
        print(f"  {'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*5}")
        for r in results:
            ratio = r['D_sim'] / r['D_pred'] if r['D_pred'] > 0 and r['D_sim'] < np.inf else np.nan
            print(f"  {r['a']:9.6f} | {r['D_white']:8.1f} | {r['D_sim']:8.1f} | "
                  f"{r['D_pred']:8.1f} | {ratio:8.3f} | {r['n_valid']:>5}")

    return results


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    t_start = time.time()

    print("=" * 78)
    print("COLORED NOISE B INVARIANCE TEST")
    print("Lake model: f(x) = a - bx + rx^q/(x^q+1)")
    print("Noise: Ornstein-Uhlenbeck, tau_c = tau_relax")
    print("B_colored = 2*DeltaPhi / sigma_colored*^2")
    print("=" * 78)

    # Method 1: Analytical correction sweep
    analytical = analytical_colored_sweep(8, n_pts=25)

    # Method 2: SDE verification at selected points (reduced for tractability)
    sde_results = sde_verification(8, n_verify=3, n_trials=200)

    # =========================================================================
    # FINAL SYNTHESIS
    # =========================================================================
    print(f"\n{'='*78}")
    print(f"FINAL SYNTHESIS: COLORED NOISE B INVARIANCE")
    print(f"{'='*78}")

    if analytical:
        print(f"\n  Analytical (Kramers correction, q=8):")
        print(f"    B_white:   mean={analytical['B_white_mean']:.4f}, "
              f"CV={analytical['B_white_cv']:.2f}%")
        print(f"    B_colored: mean={analytical['B_colored_mean']:.4f}, "
              f"CV={analytical['B_colored_cv']:.2f}%")
        print(f"    B shift (colored - white): "
              f"{analytical['B_colored_mean'] - analytical['B_white_mean']:.4f}")
        print(f"    mu range: [{analytical['mu_range'][0]:.4f}, "
              f"{analytical['mu_range'][1]:.4f}]")

        if analytical['B_colored_cv'] < 5.0:
            print(f"\n  RESULT: B INVARIANCE HOLDS UNDER COLORED NOISE")
            print(f"  B_colored CV = {analytical['B_colored_cv']:.2f}% < 5%")
        else:
            print(f"\n  RESULT: B_colored CV = {analytical['B_colored_cv']:.2f}%")

        # Stability window
        in_window = 1.8 <= analytical['B_colored_mean'] <= 6.0
        print(f"  B_colored = {analytical['B_colored_mean']:.3f}: "
              f"{'IN' if in_window else 'OUTSIDE'} stability window [1.8, 6.0]")

    if sde_results:
        print(f"\n  SDE verification:")
        ratios = [r['D_sim'] / r['D_pred'] for r in sde_results
                  if r['D_sim'] < np.inf and r['D_pred'] > 0]
        if ratios:
            print(f"    D_sim / D_Kramers_pred: mean={np.mean(ratios):.3f}, "
                  f"range=[{min(ratios):.3f}, {max(ratios):.3f}]")
            if 0.5 < np.mean(ratios) < 2.0:
                print(f"    Kramers correction validated by simulation (within 2x)")
            else:
                print(f"    Kramers correction has larger discrepancy")

    elapsed = time.time() - t_start
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print(f"\n{'='*78}")
    print(f"END OF COLORED NOISE TEST")
    print(f"{'='*78}")
