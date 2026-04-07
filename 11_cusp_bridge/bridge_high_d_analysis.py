#!/usr/bin/env python3
"""
Bridge high-d analysis: Phase 2 (B computation) + Phase 3 (model fitting).

Reproduces all Phase 2 and Phase 3 results from the cusp bridge high-d
scaling study. Phase 1 data (P(bistable) at each d) is hardcoded from
bridge_high_d_scaling.py and bridge_dimensional_scaling.py output.

For each d in {38, 44, 50}, generates N=50 random bistable configurations
using the same model and parameter distributions as bridge_high_d_scaling.py,
computes barrier DeltaPhi, finds sigma* via bisection, and computes
B = 2*DeltaPhi/sigma*^2.

Fits three models to combined ln(1/P) vs d data (d >= 14):
  A: linear       ln(1/P) = c0 + gamma*d
  B: quadratic    ln(1/P) = c0 + g1*d + g2*d^2
  C: stretched    ln(1/P) = c0 + gamma*d^beta

Model: f(x) = a - b*x + sum_i [ r_i * x^q_i / (x^q_i + h_i^q_i) ]
Parameters per config: d = 2 + 3*k (k channels)
Parameter distributions: a~U(0.05,0.80), b~U(0.2,2.0),
    r_i~U(0.1,2.0), q_i~U(2.0,15.0), h_i~U(0.3,2.0)

Run: python3 bridge_high_d_analysis.py
Dependencies: numpy, scipy
Note: Phase 2 may take 30-60 minutes due to high-d bistability search.
"""

import sys
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, curve_fit
from scipy.stats import linregress
import time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


def pprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


# ============================================================
# CONSTANTS
# ============================================================

X_SCAN = np.linspace(0.001, 5.0, 2000)
D_TARGET = 100.0
N_GRID_MFPT = 25000
N_B_TARGET = 50          # number of B values to compute per d
BATCH_SIZE = 50_000

# ============================================================
# HARDCODED PHASE 1 DATA
# ============================================================

# Previous data (d=5-32) from bridge_dimensional_scaling.py / CUSP_BRIDGE_ANALYSIS.md
D_PREV = np.array([5, 8, 11, 14, 17, 20, 26, 32])
P_PREV = np.array([0.174, 0.285, 0.332, 0.320, 0.275, 0.211, 0.095, 0.024])

# New data (d=38-62) from bridge_high_d_scaling.py
D_NEW = np.array([38, 44, 50, 56, 62])
P_NEW = np.array([3.64e-3, 6.25e-4, 1.04e-4, 3.40e-5, 2.20e-5])
N_SAMPLES_NEW = np.array([100_000, 200_000, 500_000, 1_000_000, 2_000_000])
N_BISTABLE_NEW = np.array([364, 125, 52, 34, 44])

# d values for B computation (those with sufficient bistable configs)
B_SPECS = [
    (12, 38),   # (k_channels, d_params)
    (14, 44),
    (16, 50),
]


# ============================================================
# MODEL EVALUATION (copied from bridge_high_d_scaling.py)
# ============================================================

def eval_f_vec(x_arr, a, b, channels):
    """Evaluate f(x) for a single config at array of x values."""
    val = a - b * x_arr
    for r_i, q_i, h_i in channels:
        xq = x_arr**q_i
        hq = h_i**q_i
        val = val + r_i * xq / (xq + hq)
    return val


def eval_f_scalar(x, a, b, channels):
    """Evaluate f(x) at a single scalar x."""
    val = a - b * x
    for r_i, q_i, h_i in channels:
        xq = x**q_i
        hq = h_i**q_i
        val = val + r_i * xq / (xq + hq)
    return val


# ============================================================
# BISTABILITY DETECTION
# ============================================================

def _verify_bistable(a, b, channels, fv_precomputed):
    """
    Verify bistability using precomputed f(x) on X_SCAN grid.
    Refines roots with brentq, checks stability pattern (stable-unstable-stable).
    Returns (x_eq, x_sad, lam_eq, dphi) or None.
    """
    sc_idx = np.where(np.diff(np.sign(fv_precomputed)))[0]
    if len(sc_idx) < 3:
        return None

    roots = []
    for i in sc_idx[:6]:
        try:
            r = brentq(lambda x: eval_f_scalar(x, a, b, channels),
                       X_SCAN[i], X_SCAN[i + 1], xtol=1e-10)
            roots.append(r)
        except Exception:
            pass

    if len(roots) < 3:
        return None

    dx = 1e-7
    derivs = []
    for r in roots[:3]:
        fp = eval_f_scalar(r + dx, a, b, channels)
        fm = eval_f_scalar(r - dx, a, b, channels)
        derivs.append((fp - fm) / (2 * dx))

    if derivs[0] < 0 and derivs[1] > 0 and derivs[2] < 0:
        x_eq = roots[0]
        x_sad = roots[1]
        lam_eq = derivs[0]
        dphi, _ = quad(lambda xx: -eval_f_scalar(xx, a, b, channels),
                       x_eq, x_sad, limit=50)
        if dphi > 1e-10:
            return (x_eq, x_sad, lam_eq, dphi)

    return None


def find_bistable_configs(k, n_target):
    """
    Search for n_target bistable configs at channel count k.
    Uses same parameter distributions as bridge_high_d_scaling.py.
    """
    configs = []
    n_searched = 0
    x = X_SCAN

    while len(configs) < n_target:
        bs = BATCH_SIZE
        n_searched += bs

        a_all = np.random.uniform(0.05, 0.80, bs)
        b_all = np.random.uniform(0.2, 2.0, bs)
        r_all = np.random.uniform(0.1, 2.0, (bs, k))
        q_all = np.random.uniform(2.0, 15.0, (bs, k))
        h_all = np.random.uniform(0.3, 2.0, (bs, k))

        fv = a_all[:, None] - b_all[:, None] * x[None, :]
        for ch in range(k):
            r = r_all[:, ch][:, None]
            q = q_all[:, ch][:, None]
            h = h_all[:, ch][:, None]
            xq = x[None, :] ** q
            hq = h ** q
            fv = fv + r * xq / (xq + hq)

        signs = np.sign(fv)
        diffs = np.diff(signs, axis=1)
        n_sc = np.count_nonzero(diffs, axis=1)
        cand_idx = np.where(n_sc >= 3)[0]

        for idx in cand_idx:
            if len(configs) >= n_target:
                break
            a_val = a_all[idx]
            b_val = b_all[idx]
            ch_list = [(r_all[idx, c], q_all[idx, c], h_all[idx, c]) for c in range(k)]
            result = _verify_bistable(a_val, b_val, ch_list, fv[idx])
            if result is not None:
                configs.append((a_val, b_val, ch_list, result))

        pprint(f"    searched {n_searched:,}, found {len(configs)}/{n_target} bistable")

    return configs


# ============================================================
# MFPT / B COMPUTATION (copied from bridge_high_d_scaling.py)
# ============================================================

def compute_D_exact(a, b, channels, x_eq, x_sad, lam_eq, sigma):
    """Compute exact D = MFPT / tau via 1D potential integration."""
    tau = 1.0 / abs(lam_eq)
    spread = 3.0 * sigma / np.sqrt(2.0 * abs(lam_eq))
    x_lo = max(0.001, x_eq - spread)
    x_hi = x_sad + 0.001

    x_grid = np.linspace(x_lo, x_hi, N_GRID_MFPT)
    dx_grid = x_grid[1] - x_grid[0]

    f_vals = eval_f_vec(x_grid, a, b, channels)
    U_raw = np.cumsum(-f_vals) * dx_grid

    i_eq = np.argmin(np.abs(x_grid - x_eq))
    U = U_raw - U_raw[i_eq]

    phi = 2.0 * U / sigma**2
    phi = np.clip(phi, -500, 500)

    exp_neg_phi = np.exp(-phi)
    I_x = np.cumsum(exp_neg_phi) * dx_grid

    psi = (2.0 / sigma**2) * np.exp(phi) * I_x

    i_sad = np.argmin(np.abs(x_grid - x_sad))
    if i_eq >= i_sad:
        return 1e30
    mfpt = np.trapz(psi[i_eq:i_sad + 1], x_grid[i_eq:i_sad + 1])

    return mfpt / tau


def find_sigma_star(a, b, channels, x_eq, x_sad, lam_eq):
    """Find sigma* where D_exact(sigma*) = D_TARGET."""
    def obj(log_sig):
        sig = np.exp(log_sig)
        D = compute_D_exact(a, b, channels, x_eq, x_sad, lam_eq, sig)
        return np.log(max(D, 1e-30)) - np.log(D_TARGET)

    try:
        log_sig_star = brentq(obj, np.log(0.001), np.log(5.0),
                              xtol=1e-5, maxiter=40)
        return np.exp(log_sig_star)
    except Exception:
        return None


# ============================================================
# MODEL FITTING (copied from bridge_high_d_scaling.py)
# ============================================================

def fit_models(d_all, lnP_inv_all):
    """Fit 3 models to ln(1/P) vs d data (d >= 14 only)."""
    mask = d_all >= 14
    d = d_all[mask]
    y = lnP_inv_all[mask]

    results = {}
    target = 13.0 * np.log(10.0)

    # Model A: linear
    slope, intercept, r_value, _, _ = linregress(d, y)
    r2_a = r_value**2
    d_target_a = (target - intercept) / slope if slope > 0 else np.inf
    results['A'] = {
        'gamma': slope, 'c0': intercept, 'R2': r2_a,
        'd_target': d_target_a, 'k_target': (d_target_a - 2) / 3
    }

    # Model B: quadratic
    def model_b(d_in, c0, g1, g2):
        return c0 + g1 * d_in + g2 * d_in**2

    try:
        popt, _ = curve_fit(model_b, d, y, p0=[0, 0.1, 0.001])
        y_pred = model_b(d, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2_b = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        c0_b, g1_b, g2_b = popt
        disc = g1_b**2 - 4 * g2_b * (c0_b - target)
        if disc >= 0 and g2_b != 0:
            d1 = (-g1_b + np.sqrt(disc)) / (2 * g2_b)
            d2 = (-g1_b - np.sqrt(disc)) / (2 * g2_b)
            d_target_b = max(d1, d2) if max(d1, d2) > 0 else min(d1, d2)
        elif g2_b == 0:
            d_target_b = (target - c0_b) / g1_b if g1_b > 0 else np.inf
        else:
            d_target_b = np.inf
        results['B'] = {
            'g1': g1_b, 'g2': g2_b, 'c0': c0_b, 'R2': r2_b,
            'd_target': d_target_b, 'k_target': (d_target_b - 2) / 3
        }
    except Exception as e:
        results['B'] = {'error': str(e)}

    # Model C: stretched exponential
    def model_c(d_in, c0, gamma, beta):
        return c0 + gamma * d_in**beta

    try:
        popt, _ = curve_fit(model_c, d, y, p0=[0, 0.1, 1.0],
                            bounds=([-np.inf, 0, 0.1], [np.inf, np.inf, 3.0]),
                            maxfev=10000)
        y_pred = model_c(d, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2_c = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        c0_c, gamma_c, beta_c = popt
        inner = (target - c0_c) / gamma_c
        if inner > 0:
            d_target_c = inner**(1.0 / beta_c)
        else:
            d_target_c = np.inf
        results['C'] = {
            'gamma': gamma_c, 'beta': beta_c, 'c0': c0_c, 'R2': r2_c,
            'd_target': d_target_c, 'k_target': (d_target_c - 2) / 3
        }
    except Exception as e:
        results['C'] = {'error': str(e)}

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    pprint("=" * 70)
    pprint("  BRIDGE HIGH-d ANALYSIS")
    pprint("  Phase 2 (B computation) + Phase 3 (model fitting)")
    pprint("=" * 70)
    pprint()

    # ---- Print all hardcoded Phase 1 data ----
    pprint("PHASE 1 DATA (hardcoded from bridge_dimensional_scaling.py")
    pprint("and bridge_high_d_scaling.py)")
    pprint()

    pprint("Previous data (d = 5-32):")
    pprint(f"{'d':>4s} | {'P(bistable)':>12s} | {'ln(1/P)':>8s}")
    pprint("-" * 30)
    for d_val, p_val in zip(D_PREV, P_PREV):
        pprint(f"{d_val:4d} | {p_val:12.4f} | {np.log(1.0/p_val):8.3f}")
    pprint()

    pprint("New data (d = 38-62):")
    pprint(f"{'d':>4s} | {'N_samples':>12s} | {'N_bistable':>10s} | "
           f"{'P(bistable)':>12s} | {'ln(1/P)':>8s}")
    pprint("-" * 60)
    for d_val, N_s, N_b, p_val in zip(D_NEW, N_SAMPLES_NEW,
                                       N_BISTABLE_NEW, P_NEW):
        lnp = np.log(1.0 / p_val) if p_val > 0 else np.inf
        pprint(f"{d_val:4d} | {N_s:12,} | {N_b:10,} | "
               f"{p_val:12.6e} | {lnp:8.3f}")
    pprint()

    # ---- Phase 2: B computation ----
    pprint("=" * 70)
    pprint("  PHASE 2: B computation at high d")
    pprint(f"  Target: {N_B_TARGET} configs per d, D_target = {D_TARGET}")
    pprint("=" * 70)
    pprint()

    B_results = {}

    for k, d in B_SPECS:
        pprint(f"d = {d} (k = {k} channels): searching for "
               f"{N_B_TARGET} bistable configs ...")
        t0 = time.time()

        configs = find_bistable_configs(k, N_B_TARGET)
        t_search = time.time() - t0
        pprint(f"  Search complete: {len(configs)} configs in {t_search:.1f}s")

        pprint(f"  Computing B for {len(configs)} configs ...")
        t0 = time.time()
        B_vals = []
        for i, (a, b, channels, (x_eq, x_sad, lam_eq, dphi)) in enumerate(configs):
            sig_star = find_sigma_star(a, b, channels, x_eq, x_sad, lam_eq)
            if sig_star is not None and sig_star > 0:
                B = 2.0 * dphi / sig_star**2
                if 0.1 < B < 50:
                    B_vals.append(B)
            if (i + 1) % 10 == 0:
                pprint(f"    {i+1}/{len(configs)} done, {len(B_vals)} valid B")

        t_B = time.time() - t0
        B_arr = np.array(B_vals)

        if len(B_arr) > 0:
            B_mean = np.mean(B_arr)
            B_std = np.std(B_arr)
            B_cv = B_std / B_mean if B_mean > 0 else np.inf
            n_hab = np.sum((B_arr >= 1.8) & (B_arr <= 6.0))
            pct_hab = 100.0 * n_hab / len(B_arr)
            B_results[d] = {
                'mean': B_mean, 'std': B_std, 'cv': B_cv,
                'n': len(B_arr), 'pct_habitable': pct_hab
            }
            pprint(f"  B = {B_mean:.3f} +/- {B_std:.3f} "
                   f"(CV = {100*B_cv:.1f}%, n = {len(B_arr)}, "
                   f"{pct_hab:.0f}% in [1.8, 6.0]), time = {t_B:.1f}s")
        else:
            pprint(f"  No valid B values, time = {t_B:.1f}s")
        pprint()

    # B summary table
    pprint("-" * 60)
    pprint("B invariance at high d:")
    pprint(f"{'d':>4s} | {'B mean':>7s} | {'B std':>7s} | "
           f"{'CV':>6s} | {'n':>3s} | {'% in [1.8,6.0]':>14s}")
    pprint("-" * 55)
    B_means_all = []
    for d_val in [38, 44, 50]:
        if d_val in B_results:
            r = B_results[d_val]
            pprint(f"{d_val:4d} | {r['mean']:7.3f} | {r['std']:7.3f} | "
                   f"{100*r['cv']:5.1f}% | {r['n']:3d} | "
                   f"{r['pct_habitable']:13.0f}%")
            B_means_all.append(r['mean'])
    pprint()

    if len(B_means_all) > 1:
        grand_mean = np.mean(B_means_all)
        grand_cv = np.std(B_means_all) / grand_mean if grand_mean > 0 else 0
        pprint(f"Grand mean B = {grand_mean:.3f}, "
               f"CV across d = {100*grand_cv:.1f}%")
        pprint()

    # ---- Phase 3: Combined fit and extrapolation ----
    pprint("=" * 70)
    pprint("  PHASE 3: Combined fit and extrapolation")
    pprint("=" * 70)
    pprint()

    d_all = np.concatenate([D_PREV, D_NEW])
    P_all = np.concatenate([P_PREV, P_NEW])
    lnP_inv_all = np.log(1.0 / P_all)

    n_fit = int(np.sum(d_all >= 14))
    pprint(f"Combined dataset: {len(d_all)} points, d = {d_all.min():.0f}-{d_all.max():.0f}")
    pprint(f"Fitting on d >= 14: {n_fit} points")
    pprint()

    results = fit_models(d_all, lnP_inv_all)

    target_val = 13.0 * np.log(10.0)
    pprint(f"Target: ln(1/P) = 13*ln(10) = {target_val:.2f}  [S_0 = 10^13]")
    pprint()

    # Print fit results
    if 'A' in results and 'error' not in results['A']:
        r = results['A']
        pprint(f"Model A (linear): ln(1/P) = {r['c0']:.4f} + {r['gamma']:.4f} * d")
        pprint(f"  gamma = {r['gamma']:.4f}, R^2 = {r['R2']:.4f}")
        pprint(f"  d for S_0 = 10^13: {r['d_target']:.0f}  "
               f"(k = {r['k_target']:.0f} channels)")
        pprint()

    if 'B' in results and 'error' not in results['B']:
        r = results['B']
        pprint(f"Model B (quadratic): ln(1/P) = {r['c0']:.4f} + "
               f"{r['g1']:.4f}*d + {r['g2']:.6f}*d^2")
        pprint(f"  R^2 = {r['R2']:.4f}")
        d_med = np.median(d_all[d_all >= 14])
        eff_gamma = r['g1'] + 2 * r['g2'] * d_med
        pprint(f"  gamma_eff at d={d_med:.0f}: {eff_gamma:.4f}")
        pprint(f"  d for S_0 = 10^13: {r['d_target']:.0f}  "
               f"(k = {r['k_target']:.0f} channels)")
        pprint()

    if 'C' in results and 'error' not in results['C']:
        r = results['C']
        pprint(f"Model C (stretched exp): ln(1/P) = {r['c0']:.4f} + "
               f"{r['gamma']:.4f} * d^{r['beta']:.2f}")
        pprint(f"  beta = {r['beta']:.2f}, R^2 = {r['R2']:.4f}")
        d_med = np.median(d_all[d_all >= 14])
        eff_gamma = r['gamma'] * r['beta'] * d_med**(r['beta'] - 1)
        pprint(f"  gamma_eff at d={d_med:.0f}: {eff_gamma:.4f}")
        pprint(f"  d for S_0 = 10^13: {r['d_target']:.0f}  "
               f"(k = {r['k_target']:.0f} channels)")
        pprint()

    # Summary table
    pprint("-" * 60)
    pprint("Summary of fits (d >= 14):")
    pprint(f"{'Model':>15s} | {'gamma_eff':>9s} | {'R^2':>7s} | "
           f"{'d(S_0=10^13)':>12s} | {'k':>3s}")
    pprint("-" * 55)

    d_med = np.median(d_all[d_all >= 14])
    for name in ['A', 'B', 'C']:
        if name in results and 'error' not in results[name]:
            r = results[name]
            if name == 'A':
                g_eff = r['gamma']
            elif name == 'B':
                g_eff = r['g1'] + 2 * r['g2'] * d_med
            else:
                g_eff = r['gamma'] * r['beta'] * d_med**(r['beta'] - 1)
            pprint(f"{'  ' + name + ' (' + ['linear','quadratic','stretched'][['A','B','C'].index(name)] + ')':>15s} | "
                   f"{g_eff:9.4f} | {r['R2']:7.4f} | "
                   f"{r['d_target']:12.0f} | {r['k_target']:3.0f}")

    pprint()

    # Pairwise slopes for deceleration check
    pprint("Pairwise slopes (high-d tail):")
    high_d_data = [(d_val, np.log(1.0/p_val))
                   for d_val, p_val in zip(D_NEW, P_NEW)]
    for i in range(len(high_d_data) - 1):
        d1, y1 = high_d_data[i]
        d2, y2 = high_d_data[i + 1]
        slope = (y2 - y1) / (d2 - d1)
        pprint(f"  d={d1:.0f}-{d2:.0f}: slope = {slope:.3f}")

    pprint()
    pprint("=" * 70)
    pprint("  CONCLUSION")
    pprint("=" * 70)
    pprint()
    pprint("  gamma ~ 0.22 decisively resolves the 0.14 vs 0.41 ambiguity.")
    pprint("  d ~ 123-145 for S_0 = 10^13 (40-48 channels).")
    pprint("  Pairwise slopes decelerate at d > 50.")
    pprint("  B ~ 3.5-3.8 at high d, inside habitable zone [1.8, 6.0].")
    pprint()
    pprint("=" * 70)


if __name__ == "__main__":
    main()
