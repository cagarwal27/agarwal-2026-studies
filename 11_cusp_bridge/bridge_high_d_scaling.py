#!/usr/bin/env python3
"""
Bridge dimensional scaling: extend to high d.
Measures P(bistable) for 1D multi-channel ODEs at d = 38-92 parameters.
Combines with existing d = 5-32 data to fit gamma in S_0 = exp(gamma*d).

Model: f(x) = a - b*x + sum_i [ r_i * x^q_i / (x^q_i + h_i^q_i) ]
Parameters: d = 2 + 3*k  (k channels)
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
    """Print with immediate flush."""
    print(*args, **kwargs)
    sys.stdout.flush()


# ============================================================
# CONSTANTS (match bridge_dimensional_scaling.py exactly)
# ============================================================

X_SCAN = np.linspace(0.001, 5.0, 2000)
D_TARGET = 100.0
N_GRID_MFPT = 25000
N_MFPT_SUBSET = 50

# Previous data from CUSP_BRIDGE_ANALYSIS.md
D_PREV = np.array([5, 8, 11, 14, 17, 20, 26, 32])
P_PREV = np.array([0.174, 0.285, 0.332, 0.320, 0.275, 0.211, 0.095, 0.024])

# New d values: k channels -> d = 2 + 3*k
HIGH_D_SPECS = [
    # (k, d, N_samples)
    (12, 38, 100_000),
    (14, 44, 200_000),
    (16, 50, 500_000),
    (18, 56, 1_000_000),
    (20, 62, 2_000_000),
    (24, 74, 5_000_000),
    (30, 92, 10_000_000),
]

BATCH_SIZE = 50_000


# ============================================================
# VECTORIZED f(x) EVALUATION
# ============================================================

def eval_f_vec(x_arr, a, b, channels):
    """Evaluate f(x) for a single config at array of x values. Returns array."""
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
# BISTABILITY CHECK
# ============================================================

def check_bistable_batch(k, N):
    """
    Check N random k-channel configs for bistability.
    Processes in batches; vectorized within each batch.
    Returns (n_bistable, list of (a, b, channels, result) configs).
    """
    n_bistable = 0
    configs = []
    x = X_SCAN  # (2000,)

    n_done = 0
    batch_num = 0
    while n_done < N:
        bs = min(BATCH_SIZE, N - n_done)
        batch_num += 1

        # Generate all params for this batch
        a_all = np.random.uniform(0.05, 0.80, bs)
        b_all = np.random.uniform(0.2, 2.0, bs)
        r_all = np.random.uniform(0.1, 2.0, (bs, k))
        q_all = np.random.uniform(2.0, 15.0, (bs, k))
        h_all = np.random.uniform(0.3, 2.0, (bs, k))

        # Evaluate f(x) for all configs at all x-points: shape (bs, 2000)
        fv = a_all[:, None] - b_all[:, None] * x[None, :]

        for ch in range(k):
            r = r_all[:, ch][:, None]
            q = q_all[:, ch][:, None]
            h = h_all[:, ch][:, None]
            xq = x[None, :] ** q
            hq = h ** q
            fv = fv + r * xq / (xq + hq)

        # Count sign changes per config
        signs = np.sign(fv)
        diffs = np.diff(signs, axis=1)
        n_sc = np.count_nonzero(diffs, axis=1)

        # Candidates with >= 3 sign changes
        cand_idx = np.where(n_sc >= 3)[0]

        for idx in cand_idx:
            a_val = a_all[idx]
            b_val = b_all[idx]
            ch_list = [(r_all[idx, c], q_all[idx, c], h_all[idx, c]) for c in range(k)]

            result = _verify_bistable(a_val, b_val, ch_list, fv[idx])
            if result is not None:
                n_bistable += 1
                configs.append((a_val, b_val, ch_list, result))

        n_done += bs
        if batch_num % 10 == 0:
            pprint(f"    batch {batch_num}: {n_done:,}/{N:,} done, {n_bistable} bistable so far")

    return n_bistable, configs


def _verify_bistable(a, b, channels, fv_precomputed):
    """
    Verify bistability using precomputed f(x) values on X_SCAN grid.
    Refines roots with brentq, checks stability.
    Returns (x_eq, x_sad, lam_eq, dphi) or None.
    """
    sc_idx = np.where(np.diff(np.sign(fv_precomputed)))[0]
    if len(sc_idx) < 3:
        return None

    # Find roots via Brent's method
    roots = []
    for i in sc_idx[:6]:  # At most 6 sign changes
        try:
            r = brentq(lambda x: eval_f_scalar(x, a, b, channels),
                       X_SCAN[i], X_SCAN[i + 1], xtol=1e-10)
            roots.append(r)
        except:
            pass

    if len(roots) < 3:
        return None

    # Stability check via finite difference derivative
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


# ============================================================
# MFPT / B COMPUTATION
# ============================================================

def compute_D_exact(a, b, channels, x_eq, x_sad, lam_eq, sigma):
    """Compute exact D = MFPT / tau via 1D potential integration."""
    tau = 1.0 / abs(lam_eq)
    spread = 3.0 * sigma / np.sqrt(2.0 * abs(lam_eq))
    x_lo = max(0.001, x_eq - spread)
    x_hi = x_sad + 0.001

    x_grid = np.linspace(x_lo, x_hi, N_GRID_MFPT)
    dx = x_grid[1] - x_grid[0]

    # Vectorized f evaluation on entire grid
    f_vals = eval_f_vec(x_grid, a, b, channels)
    U_raw = np.cumsum(-f_vals) * dx

    i_eq = np.argmin(np.abs(x_grid - x_eq))
    U = U_raw - U_raw[i_eq]

    phi = 2.0 * U / sigma**2
    phi = np.clip(phi, -500, 500)

    exp_neg_phi = np.exp(-phi)
    I_x = np.cumsum(exp_neg_phi) * dx

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
        log_sig_star = brentq(obj, np.log(0.001), np.log(5.0), xtol=1e-5, maxiter=40)
        return np.exp(log_sig_star)
    except:
        return None


def compute_B_for_configs(configs, n_max=50):
    """Compute B = 2*dphi/sigma*^2 for up to n_max bistable configs."""
    B_vals = []
    n_use = min(len(configs), n_max)
    if len(configs) > n_max:
        indices = np.random.choice(len(configs), n_use, replace=False)
    else:
        indices = list(range(len(configs)))

    for i, idx in enumerate(indices):
        a, b, channels, (x_eq, x_sad, lam_eq, dphi) = configs[idx]
        sig_star = find_sigma_star(a, b, channels, x_eq, x_sad, lam_eq)
        if sig_star is not None and sig_star > 0:
            B = 2.0 * dphi / sig_star**2
            if 0.1 < B < 50:
                B_vals.append(B)
        if (i + 1) % 10 == 0:
            pprint(f"    B computation: {i+1}/{n_use} done, {len(B_vals)} valid")

    return np.array(B_vals)


# ============================================================
# FITTING MODELS
# ============================================================

def fit_models(d_all, lnP_inv_all):
    """Fit 3 models to ln(1/P) vs d data (d >= 14 only)."""
    mask = d_all >= 14
    d = d_all[mask]
    y = lnP_inv_all[mask]

    results = {}
    target = 13.0 * np.log(10.0)

    # Model A: ln(1/P) = c_0 + gamma * d
    slope, intercept, r_value, _, _ = linregress(d, y)
    r2_a = r_value**2
    gamma_a = slope
    c0_a = intercept
    d_target_a = (target - c0_a) / gamma_a if gamma_a > 0 else np.inf
    results['A'] = {
        'gamma': gamma_a, 'c0': c0_a, 'R2': r2_a,
        'd_target': d_target_a, 'k_target': (d_target_a - 2) / 3
    }

    # Model B: ln(1/P) = c_0 + g1*d + g2*d^2
    def model_b(d, c0, g1, g2):
        return c0 + g1 * d + g2 * d**2

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

    # Model C: ln(1/P) = c_0 + gamma * d^beta
    def model_c(d, c0, gamma, beta):
        return c0 + gamma * d**beta

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
    pprint("=" * 65)
    pprint("  BRIDGE DIMENSIONAL SCALING: HIGH-d EXTENSION")
    pprint("=" * 65)
    pprint()

    # --- Previous data ---
    pprint("Previous data (d = 5-32):")
    pprint(f"{'d':>4s} | {'P(bistable)':>11s} | {'ln(1/P)':>8s}")
    pprint("-" * 30)
    for d_val, p_val in zip(D_PREV, P_PREV):
        pprint(f"{d_val:4d} | {p_val:11.4f} | {np.log(1.0/p_val):8.3f}")
    pprint()

    # --- Phase 1: P(bistable) at high d ---
    pprint("=" * 65)
    pprint("  PHASE 1: P(bistable) at high d")
    pprint("=" * 65)
    pprint()

    new_d = []
    new_P = []
    new_n_bistable = []
    new_N_samples = []
    all_configs_by_d = {}

    consecutive_zeros = 0

    for k, d, N in HIGH_D_SPECS:
        pprint(f"d = {d} (k = {k} channels), N = {N:,} ...")
        t0 = time.time()

        n_bi, configs = check_bistable_batch(k, N)
        elapsed = time.time() - t0

        P = n_bi / N if N > 0 else 0
        lnP_inv = np.log(1.0 / P) if P > 0 else np.inf

        flag = ""
        if n_bi == 0:
            flag = " [ZERO -- upper bound P < {:.1e}]".format(1.0 / N)
            consecutive_zeros += 1
        elif n_bi < 10:
            flag = " [UNCERTAIN -- only {} found]".format(n_bi)
            consecutive_zeros = 0
        else:
            consecutive_zeros = 0

        pprint(f"  N_bistable = {n_bi:,}, P = {P:.6e}, ln(1/P) = {lnP_inv:.3f}, "
               f"time = {elapsed:.1f}s{flag}")

        new_d.append(d)
        new_P.append(P)
        new_n_bistable.append(n_bi)
        new_N_samples.append(N)
        all_configs_by_d[d] = configs

        if consecutive_zeros >= 2:
            pprint(f"\n  STOP: Two consecutive d values with zero bistable configs.")
            break

    pprint()

    # --- Phase 2: B computation ---
    pprint("=" * 65)
    pprint("  PHASE 2: B computation (for d with >= 10 bistable)")
    pprint("=" * 65)
    pprint()

    B_results = {}
    for d_val, n_bi in zip(new_d, new_n_bistable):
        if n_bi >= 10:
            configs = all_configs_by_d[d_val]
            pprint(f"d = {d_val}: computing B for {min(n_bi, N_MFPT_SUBSET)} configs ...")
            t0 = time.time()
            B_vals = compute_B_for_configs(configs, N_MFPT_SUBSET)
            elapsed = time.time() - t0
            if len(B_vals) > 0:
                B_mean = np.mean(B_vals)
                B_cv = np.std(B_vals) / B_mean if B_mean > 0 else np.inf
                B_results[d_val] = (B_mean, B_cv, len(B_vals))
                pprint(f"  B = {B_mean:.3f} +/- {np.std(B_vals):.3f} "
                       f"(CV = {B_cv:.3f}, n = {len(B_vals)}), time = {elapsed:.1f}s")
            else:
                pprint(f"  No valid B values computed, time = {elapsed:.1f}s")
        else:
            if n_bi > 0:
                pprint(f"d = {d_val}: only {n_bi} bistable -- skipping B computation")

    pprint()

    # --- Summary table ---
    pprint("=" * 65)
    pprint("  SUMMARY: New data (d >= 38)")
    pprint("=" * 65)
    pprint()
    header = f"{'d':>4s} | {'N_samples':>12s} | {'N_bistable':>10s} | {'P(bistable)':>12s} | {'ln(1/P)':>8s}"
    if B_results:
        header += f" | {'B mean':>7s} | {'B CV':>6s}"
    pprint(header)
    pprint("-" * len(header))
    for d_val, P_val, n_bi, N_s in zip(new_d, new_P, new_n_bistable, new_N_samples):
        lnP_inv = np.log(1.0 / P_val) if P_val > 0 else np.inf
        row = f"{d_val:4d} | {N_s:12,} | {n_bi:10,} | {P_val:12.6e} | {lnP_inv:8.3f}"
        if d_val in B_results:
            bm, bcv, _ = B_results[d_val]
            row += f" | {bm:7.3f} | {bcv:6.3f}"
        elif B_results:
            row += f" | {'---':>7s} | {'---':>6s}"
        pprint(row)
    pprint()

    # --- Phase 3: Fit and extrapolate ---
    pprint("=" * 65)
    pprint("  PHASE 3: Combined fit and extrapolation")
    pprint("=" * 65)
    pprint()

    valid_new = [(d_val, P_val) for d_val, P_val in zip(new_d, new_P) if P_val > 0]
    if valid_new:
        d_new_valid, P_new_valid = zip(*valid_new)
        d_all = np.concatenate([D_PREV, np.array(d_new_valid)])
        P_all = np.concatenate([P_PREV, np.array(P_new_valid)])
    else:
        d_all = D_PREV.copy()
        P_all = P_PREV.copy()
    lnP_inv_all = np.log(1.0 / P_all)

    pprint(f"Combined dataset: {len(d_all)} points, d = {d_all.min():.0f}-{d_all.max():.0f}")
    n_fit = int(np.sum(d_all >= 14))
    pprint(f"Fitting on d >= 14: {n_fit} points")
    pprint()

    results = fit_models(d_all, lnP_inv_all)

    target_val = 13.0 * np.log(10.0)
    pprint(f"Target: ln(1/P) = 13*ln(10) = {target_val:.2f}")
    pprint()

    if 'A' in results and 'error' not in results['A']:
        r = results['A']
        pprint(f"Model A (linear): ln(1/P) = {r['c0']:.4f} + {r['gamma']:.6f} * d")
        pprint(f"  gamma = {r['gamma']:.6f}, c0 = {r['c0']:.4f}, R^2 = {r['R2']:.6f}")
        pprint(f"  d for S_0 = 10^13: {r['d_target']:.1f}  (k = {r['k_target']:.0f} channels)")
        pprint()

    if 'B' in results and 'error' not in results['B']:
        r = results['B']
        pprint(f"Model B (quadratic): ln(1/P) = {r['c0']:.4f} + {r['g1']:.6f}*d + {r['g2']:.8f}*d^2")
        pprint(f"  g1 = {r['g1']:.6f}, g2 = {r['g2']:.8f}, R^2 = {r['R2']:.6f}")
        if r['g2'] > 0:
            pprint(f"  g2 > 0: decay ACCELERATES at high d")
        else:
            pprint(f"  g2 < 0: decay DECELERATES at high d")
        pprint(f"  d for S_0 = 10^13: {r['d_target']:.1f}  (k = {r['k_target']:.0f} channels)")
        pprint()
    elif 'B' in results:
        pprint(f"Model B (quadratic): FIT FAILED -- {results['B']['error']}")
        pprint()

    if 'C' in results and 'error' not in results['C']:
        r = results['C']
        pprint(f"Model C (stretched exp): ln(1/P) = {r['c0']:.4f} + {r['gamma']:.6f} * d^{r['beta']:.4f}")
        pprint(f"  gamma = {r['gamma']:.6f}, beta = {r['beta']:.4f}, R^2 = {r['R2']:.6f}")
        if r['beta'] > 1.05:
            pprint(f"  beta > 1: superlinear (decay accelerates)")
        elif r['beta'] < 0.95:
            pprint(f"  beta < 1: sublinear (decay decelerates)")
        else:
            pprint(f"  beta ~ 1: consistent with linear model")
        pprint(f"  d for S_0 = 10^13: {r['d_target']:.1f}  (k = {r['k_target']:.0f} channels)")
        pprint()
    elif 'C' in results:
        pprint(f"Model C (stretched exp): FIT FAILED -- {results['C']['error']}")
        pprint()

    # Upper bounds from zero-detection d values
    zero_d = [(d_val, N_s) for d_val, P_val, N_s in zip(new_d, new_P, new_N_samples) if P_val == 0]
    if zero_d:
        pprint("Upper bounds from zero-detection d values:")
        for d_val, N_s in zero_d:
            ub = 1.0 / N_s
            pprint(f"  d = {d_val}: P < {ub:.1e}, ln(1/P) > {np.log(N_s):.2f}")
        pprint()

    # --- Conclusion ---
    pprint("=" * 65)
    pprint("  CONCLUSION")
    pprint("=" * 65)
    pprint()

    gammas = []
    d_targets = []
    model_names = []

    if 'A' in results and 'error' not in results['A']:
        gammas.append(results['A']['gamma'])
        d_targets.append(results['A']['d_target'])
        model_names.append('A')
    if 'B' in results and 'error' not in results['B']:
        d_med = np.median(d_all[d_all >= 14])
        eff_gamma_b = results['B']['g1'] + 2 * results['B']['g2'] * d_med
        gammas.append(eff_gamma_b)
        d_targets.append(results['B']['d_target'])
        model_names.append('B')
    if 'C' in results and 'error' not in results['C']:
        d_med = np.median(d_all[d_all >= 14])
        rc = results['C']
        eff_gamma_c = rc['gamma'] * rc['beta'] * d_med**(rc['beta'] - 1)
        gammas.append(eff_gamma_c)
        d_targets.append(rc['d_target'])
        model_names.append('C')

    if gammas:
        pprint(f"  gamma (best estimates):")
        for name, g in zip(model_names, gammas):
            pprint(f"    Model {name}: gamma = {g:.4f}")
        pprint(f"    Range: [{min(gammas):.4f}, {max(gammas):.4f}]")
        pprint()
        pprint(f"  d for S_0 = 10^13:")
        for name, dt in zip(model_names, d_targets):
            pprint(f"    Model {name}: d = {dt:.0f}")
        pprint(f"    Range: [{min(d_targets):.0f}, {max(d_targets):.0f}]")
        pprint()

    # Check linearity
    valid_high = [(d_val, P_val) for d_val, P_val in zip(new_d, new_P) if P_val > 0 and d_val >= 38]
    if len(valid_high) >= 2:
        if 'A' in results and 'error' not in results['A']:
            ra = results['A']
            residuals = []
            for d_val, P_val in valid_high:
                predicted = ra['c0'] + ra['gamma'] * d_val
                actual = np.log(1.0 / P_val)
                residuals.append(actual - predicted)
            max_resid = max(abs(rv) for rv in residuals)
            pprint(f"  Linear tail (d >= 38):")
            pprint(f"    Max residual from linear fit: {max_resid:.3f}")
            if max_resid < 0.5:
                pprint(f"    --> YES: linear tail holds at high d")
            elif max_resid < 1.5:
                pprint(f"    --> INCONCLUSIVE: moderate deviations")
            else:
                pprint(f"    --> NO: significant departure from linearity")
    elif len(valid_high) == 1:
        pprint(f"  Linear tail: only 1 new point with P > 0 -- INCONCLUSIVE")
    else:
        pprint(f"  Linear tail: no new points with P > 0 -- INCONCLUSIVE")

    pprint()
    pprint("=" * 65)


if __name__ == "__main__":
    main()
