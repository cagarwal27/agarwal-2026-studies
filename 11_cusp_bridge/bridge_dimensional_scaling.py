#!/usr/bin/env python3
"""
BRIDGE DIMENSIONAL SCALING TEST
================================
Tests whether S₀ = exp(B × Ω) where Ω scales with effective dimensionality.

Approach: Build ODE systems with increasing numbers of parameters (d),
measure P(viable B) at each d, extract Ω(d) = ln(1/P) / B.

If Ω grows linearly with d → that's the bridge between fire and tree.

Target: Ω(d_real) ≈ 8 gives S₀ = exp(3.8 × 8) ≈ 10^13.

Dimensionality levels:
  d=5:   single channel (a, b, r, q, h)
  d=8:   two channels (a, b, r1, q1, h1, r2, q2, h2)
  d=11:  three channels
  d=14:  four channels
  d=17:  five channels
  d=20:  six channels
  d=26:  eight channels
  d=32:  ten channels

For each d, sample N random configurations, measure:
  - P(bistable)
  - P(bistable AND B ∈ [1.8, 6.0])
  - P(bistable AND B > 2)
  Then compute Ω = ln(1/P) / B_mean
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import linregress
import time
import sys
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

B_TARGET = 3.8  # from the distribution test
N_SAMPLES = 30_000  # per dimensionality level
N_MFPT = 100  # subset for B computation
D_TARGET = 100.0

# ================================================================
# Multi-channel ODE: f(x) = a - b*x + Σ r_i * x^q_i / (x^q_i + h_i^q_i)
# ================================================================

def eval_multichannel(x, a, b, channels):
    """Evaluate multi-channel drift. channels = list of (r, q, h)."""
    val = a - b * x
    for r_i, q_i, h_i in channels:
        xq = x**q_i
        hq = h_i**q_i
        val += r_i * xq / (xq + hq)
    return val


def eval_multichannel_deriv(x, a, b, channels, dx=1e-7):
    """Numerical derivative."""
    return (eval_multichannel(x+dx, a, b, channels) -
            eval_multichannel(x-dx, a, b, channels)) / (2*dx)


X_SCAN = np.linspace(0.001, 5.0, 2000)


def find_bistable(a, b, channels):
    """Check if config is bistable. Return (x_eq, x_sad, lam_eq, dphi) or None."""
    x = X_SCAN
    fv = a - b * x
    for r_i, q_i, h_i in channels:
        xq = x**q_i
        hq = h_i**q_i
        fv = fv + r_i * xq / (xq + hq)

    # Find sign changes
    sc = np.where(np.diff(np.sign(fv)))[0]
    if len(sc) < 3:
        return None

    # Refine first 3 roots
    f_func = lambda xx: eval_multichannel(xx, a, b, channels)
    roots = []
    for j in sc[:4]:
        try:
            root = brentq(f_func, x[j], x[j+1], xtol=1e-10)
            roots.append(root)
        except:
            pass
    if len(roots) < 3:
        return None

    # Check stability: stable, unstable, stable
    fps = [eval_multichannel_deriv(r, a, b, channels) for r in roots[:3]]
    if not (fps[0] < 0 and fps[1] > 0 and fps[2] < 0):
        return None

    x_eq, x_sad = roots[0], roots[1]
    lam_eq = fps[0]

    # Barrier
    try:
        dphi, _ = quad(lambda xx: -eval_multichannel(xx, a, b, channels),
                       x_eq, x_sad, limit=50)
    except:
        return None
    if dphi <= 1e-10:
        return None

    return x_eq, x_sad, lam_eq, dphi


def compute_D_exact(f_func, x_eq, x_sad, lam_eq, sigma, N_grid=25000):
    """MFPT-based D."""
    tau = 1.0 / abs(lam_eq)
    x_lo = max(0.001, x_eq - 3 * sigma / np.sqrt(2 * abs(lam_eq)))
    x_hi = x_sad + 0.001
    xg = np.linspace(x_lo, x_hi, N_grid)
    dx = xg[1] - xg[0]
    neg_f = np.array([-f_func(x) for x in xg])
    U_raw = np.cumsum(neg_f) * dx
    i_eq = np.argmin(np.abs(xg - x_eq))
    U = U_raw - U_raw[i_eq]
    Phi = 2.0 * U / sigma**2
    Phi = np.clip(Phi, -500, 500)
    exp_neg = np.exp(-Phi)
    Ix = np.cumsum(exp_neg) * dx
    psi = (2.0 / sigma**2) * np.exp(Phi) * Ix
    i_sad = np.argmin(np.abs(xg - x_sad))
    if i_eq >= i_sad:
        return 1e10
    MFPT = np.trapz(psi[i_eq:i_sad+1], xg[i_eq:i_sad+1])
    return MFPT / tau


def find_sigma_star(f_func, x_eq, x_sad, lam_eq, D_target=100.0):
    """Find sigma where D_exact = D_target."""
    def obj(log_s):
        s = np.exp(log_s)
        D = compute_D_exact(f_func, x_eq, x_sad, lam_eq, s)
        return np.log(max(D, 1e-30)) - np.log(D_target)
    try:
        log_s = brentq(obj, np.log(0.001), np.log(5.0), xtol=1e-5, maxiter=40)
        return np.exp(log_s)
    except:
        return None


# ================================================================
# Generate random k-channel configurations
# ================================================================

def sample_config(k):
    """Generate random k-channel config. Returns (a, b, channels)."""
    a = np.random.uniform(0.05, 0.80)
    b = np.random.uniform(0.2, 2.0)
    channels = []
    for _ in range(k):
        r_i = np.random.uniform(0.1, 2.0)
        q_i = np.random.uniform(2.0, 15.0)
        h_i = np.random.uniform(0.3, 2.0)
        channels.append((r_i, q_i, h_i))
    return a, b, channels


def run_dimension(k, N, N_mfpt):
    """Run test for k channels (d = 2 + 3k parameters)."""
    d = 2 + 3 * k  # a, b + (r, q, h) per channel
    n_bistable = 0
    bistable_configs = []

    for _ in range(N):
        a, b, channels = sample_config(k)
        result = find_bistable(a, b, channels)
        if result is not None:
            n_bistable += 1
            x_eq, x_sad, lam_eq, dphi = result
            bistable_configs.append({
                'a': a, 'b': b, 'channels': list(channels),
                'x_eq': x_eq, 'x_sad': x_sad,
                'lam_eq': lam_eq, 'dphi': dphi,
            })

    p_bistable = n_bistable / N
    dphi_vals = np.array([c['dphi'] for c in bistable_configs]) if bistable_configs else np.array([])

    # Compute B for a subset
    B_vals = []
    if len(bistable_configs) > 0:
        n_sub = min(N_mfpt, len(bistable_configs))
        idx = np.random.choice(len(bistable_configs), n_sub, replace=False)
        for ii, i in enumerate(idx):
            c = bistable_configs[i]
            ch = c['channels']
            a_c, b_c = c['a'], c['b']
            ff = lambda x, _a=a_c, _b=b_c, _ch=ch: eval_multichannel(x, _a, _b, _ch)
            ss = find_sigma_star(ff, c['x_eq'], c['x_sad'], c['lam_eq'], D_TARGET)
            if ss is not None and ss > 0:
                B = 2.0 * c['dphi'] / ss**2
                B_vals.append(B)
            if (ii+1) % 25 == 0:
                print(f"      MFPT {ii+1}/{n_sub}, {len(B_vals)} B found", flush=True)

    B_vals = np.array(B_vals)
    return {
        'd': d, 'k': k, 'N': N,
        'n_bistable': n_bistable, 'p_bistable': p_bistable,
        'dphi_vals': dphi_vals,
        'B_vals': B_vals,
    }


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("BRIDGE DIMENSIONAL SCALING TEST")
    print("S₀ = exp(B × Ω)  →  does Ω scale linearly with dimension d?")
    print("=" * 70)
    sys.stdout.flush()

    # k = number of channels, d = 2 + 3k total parameters
    channel_counts = [1, 2, 3, 4, 5, 6, 8, 10]
    results = []

    for k in channel_counts:
        d = 2 + 3 * k
        print(f"\n{'='*70}")
        print(f"k={k} channels, d={d} parameters, N={N_SAMPLES}")
        sys.stdout.flush()
        t0 = time.time()
        res = run_dimension(k, N_SAMPLES, N_MFPT)
        t1 = time.time()

        p_bi = res['p_bistable']
        n_bi = res['n_bistable']
        mean_dphi = np.mean(res['dphi_vals']) if len(res['dphi_vals']) > 0 else 0
        med_dphi = np.median(res['dphi_vals']) if len(res['dphi_vals']) > 0 else 0

        print(f"  Bistable: {n_bi}/{N_SAMPLES} = {p_bi*100:.3f}%")
        if len(res['dphi_vals']) > 0:
            print(f"  DeltaPhi: mean={mean_dphi:.6f} median={med_dphi:.6f}")

        if len(res['B_vals']) > 0:
            mean_B = np.mean(res['B_vals'])
            med_B = np.median(res['B_vals'])
            std_B = np.std(res['B_vals'])
            p_B_viable = np.mean((res['B_vals'] >= 1.8) & (res['B_vals'] <= 6.0))
            p_B_gt2 = np.mean(res['B_vals'] > 2)
            print(f"  B: mean={mean_B:.3f} median={med_B:.3f} std={std_B:.3f} CV={std_B/mean_B*100:.1f}%")
            print(f"  P(B in [1.8,6.0]) = {p_B_viable:.4f}")
            print(f"  P(B > 2) = {p_B_gt2:.4f}")

            # Compute Omega
            # P(viable) = P(bistable) * P(B viable | bistable)
            p_viable = p_bi * p_B_viable if p_B_viable > 0 else p_bi * 0.5
            if p_viable > 0 and p_viable < 1:
                omega = np.log(1.0 / p_viable) / mean_B
                implied_S0 = 1.0 / p_viable
                print(f"  P(viable) = P(bi)*P(B ok) = {p_viable:.6f}")
                print(f"  1/P(viable) = {implied_S0:.2f}")
                print(f"  Omega = ln(1/P)/B = {omega:.4f}")
                res['omega'] = omega
                res['p_viable'] = p_viable
                res['implied_S0'] = implied_S0
            else:
                print(f"  P(viable) = {p_viable} (degenerate)")
                res['omega'] = None
                res['p_viable'] = p_viable
                res['implied_S0'] = None
        else:
            print(f"  No B values computed (0 bistable or all MFPT failed)")
            res['omega'] = None
            res['p_viable'] = p_bi
            res['implied_S0'] = None

        print(f"  ({t1-t0:.1f}s)")
        sys.stdout.flush()
        results.append(res)

    # ================================================================
    # SCALING ANALYSIS
    # ================================================================
    print(f"\n\n{'='*70}")
    print("SCALING ANALYSIS")
    print("=" * 70)

    print(f"\n{'k':>3s} {'d':>4s} {'P(bi)%':>8s} {'N_B':>5s} {'mean_B':>8s} {'P(viable)':>12s} {'1/P':>14s} {'Omega':>8s}")
    print("-" * 70)

    d_vals = []
    omega_vals = []
    log_inv_p_vals = []

    for res in results:
        k, d = res['k'], res['d']
        p_bi = res['p_bistable']
        n_B = len(res['B_vals'])
        mean_B = np.mean(res['B_vals']) if n_B > 0 else 0
        omega = res.get('omega')
        p_v = res.get('p_viable', 0)
        s0 = res.get('implied_S0')

        omega_str = f"{omega:.4f}" if omega is not None else "N/A"
        s0_str = f"{s0:.2f}" if s0 is not None else "N/A"
        p_v_str = f"{p_v:.6f}" if p_v is not None else "N/A"

        print(f"{k:>3d} {d:>4d} {p_bi*100:>7.3f}% {n_B:>5d} {mean_B:>8.3f} {p_v_str:>12s} {s0_str:>14s} {omega_str:>8s}")

        if omega is not None:
            d_vals.append(d)
            omega_vals.append(omega)
            log_inv_p_vals.append(np.log(1.0/p_v) if p_v > 0 else 0)

    # Linear regression: Omega vs d
    if len(d_vals) >= 3:
        d_arr = np.array(d_vals)
        o_arr = np.array(omega_vals)
        lip_arr = np.array(log_inv_p_vals)

        slope_o, intercept_o, r_o, p_o, se_o = linregress(d_arr, o_arr)
        print(f"\nOmega vs d: slope={slope_o:.6f} intercept={intercept_o:.4f} R²={r_o**2:.4f} p={p_o:.6f}")

        slope_l, intercept_l, r_l, p_l, se_l = linregress(d_arr, lip_arr)
        print(f"ln(1/P) vs d: slope={slope_l:.6f} intercept={intercept_l:.4f} R²={r_l**2:.4f} p={p_l:.6f}")

        # Extrapolation to S₀ = 10^13
        target_ln_S0 = 13 * np.log(10)  # ≈ 29.93
        if slope_l > 0:
            d_needed = (target_ln_S0 - intercept_l) / slope_l
            print(f"\nExtrapolation: ln(S₀) = {target_ln_S0:.2f} requires d = {d_needed:.1f}")
            print(f"  (i.e., {(d_needed-2)/3:.0f} independent channels)")

        # Extrapolation via Omega
        if slope_o > 0:
            # S₀ = exp(B * Omega(d)) = exp(B * (slope*d + intercept))
            # 10^13 = exp(B * (slope*d + intercept))
            # 13*ln(10) = B * (slope*d + intercept)
            omega_needed = target_ln_S0 / B_TARGET
            d_needed_omega = (omega_needed - intercept_o) / slope_o
            print(f"\nVia Omega: need Omega={omega_needed:.2f}")
            print(f"  Omega(d) = {slope_o:.6f}*d + {intercept_o:.4f}")
            print(f"  d_needed = {d_needed_omega:.1f}")

    # ================================================================
    # KEY RESULTS
    # ================================================================
    print(f"\n{'='*70}")
    print("KEY RESULTS")
    print("=" * 70)

    if len(d_vals) >= 3:
        print(f"\n1. Does Omega scale linearly with d?")
        print(f"   R² = {r_o**2:.4f}")
        if r_o**2 > 0.9:
            print(f"   YES — strong linear scaling (R² > 0.9)")
        elif r_o**2 > 0.7:
            print(f"   MODERATE — approximately linear (R² > 0.7)")
        else:
            print(f"   NO — weak or nonlinear relationship")

        print(f"\n2. Scaling rate: Omega increases by {slope_o:.4f} per parameter")
        print(f"   (or {slope_o*3:.4f} per channel)")

        print(f"\n3. Does P(bistable) decrease with d?")
        p_bi_vals = [r['p_bistable'] for r in results]
        d_all = [r['d'] for r in results]
        if len(p_bi_vals) >= 3:
            # Check if P(bi) decreases
            sl_p, _, r_p, p_p, _ = linregress(d_all, np.log([max(p,1e-10) for p in p_bi_vals]))
            print(f"   ln(P_bi) vs d: slope={sl_p:.4f} R²={r_p**2:.4f}")
            if sl_p < 0:
                print(f"   YES — P(bistable) decreases exponentially with d")
                print(f"   Half-life in d: {-np.log(2)/sl_p:.1f} parameters")
            else:
                print(f"   NO — P(bistable) does not decrease with d")

        print(f"\n4. Bridge equation:")
        print(f"   S₀ = exp(B × Omega(d))")
        print(f"   Omega(d) = {slope_o:.4f} × d + {intercept_o:.4f}")
        print(f"   For S₀ = 10^13: need d ≈ {d_needed_omega:.0f} parameters")
        print(f"   That's ≈ {(d_needed_omega-2)/3:.0f} independent regulatory channels")

    print(f"\nDone.")
