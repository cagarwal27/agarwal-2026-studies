#!/usr/bin/env python3
"""
B DISTRIBUTION TEST
===================
Tests whether the barrier ΔΦ (and B = 2ΔΦ/σ*²) is exponentially distributed
across random bistable ODE configurations.

Three ensembles:
  A — Random (a, q) with fixed structure
  B — Random multi-channel architectures
  C — Fully random parameters
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import expon, lognorm, weibull_min, kstest
import time
import sys
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

N_SAMPLES = 50_000
N_MFPT_SUBSET = 200
D_TARGET = 100.0
X_SCAN = np.linspace(0.001, 4.0, 2000)  # shared grid for root finding


# ================================================================
# Vectorized root finding for single-channel model
# ================================================================

def batch_find_bistable_A(N):
    """Method A: random (a,q), fixed b=0.8, r=1.0, h=1.0. Fully vectorized."""
    b, r, h = 0.8, 1.0, 1.0
    a_all = np.random.uniform(0.05, 0.80, N)
    q_all = np.random.uniform(2.0, 20.0, N)
    x = X_SCAN

    configs = []
    for i in range(N):
        a, q = a_all[i], q_all[i]
        # Vectorized eval
        xq = x**q
        hq = h**q
        fv = a - b * x + r * xq / (xq + hq)
        # Sign changes
        sc = np.where(np.diff(np.sign(fv)))[0]
        if len(sc) < 3:
            continue
        # Refine first 3 roots
        roots = []
        for j in sc[:4]:
            try:
                root = brentq(lambda xx: a - b*xx + r*xx**q/(xx**q+hq),
                              x[j], x[j+1], xtol=1e-10)
                roots.append(root)
            except:
                pass
        if len(roots) < 3:
            continue
        # Check stability pattern: stable, unstable, stable
        fp0 = -b + r*q*roots[0]**(q-1)*hq/(roots[0]**q+hq)**2
        fp1 = -b + r*q*roots[1]**(q-1)*hq/(roots[1]**q+hq)**2
        fp2 = -b + r*q*roots[2]**(q-1)*hq/(roots[2]**q+hq)**2
        if not (fp0 < 0 and fp1 > 0 and fp2 < 0):
            continue
        x_eq, x_sad = roots[0], roots[1]
        # Barrier via quad
        try:
            dphi, _ = quad(lambda xx: -(a - b*xx + r*xx**q/(xx**q+hq)),
                           x_eq, x_sad, limit=50)
        except:
            continue
        if dphi <= 1e-10:
            continue
        configs.append({
            'a': a, 'q': q, 'x_eq': x_eq, 'x_sad': x_sad,
            'lam_eq': fp0, 'lam_sad': fp1, 'dphi': dphi,
            'params': (a, b, r, q, h),
        })
    return configs


def batch_find_bistable_B(N):
    """Method B: random multi-channel."""
    b = 0.8
    x = X_SCAN
    configs = []
    for _ in range(N):
        k = np.random.choice([1, 2, 3])
        channels = [(np.random.uniform(0.1, 2.0),
                      np.random.uniform(2.0, 15.0),
                      np.random.uniform(0.3, 2.0)) for _ in range(k)]
        a = np.random.uniform(0.05, 0.80)
        # Eval
        fv = a - b * x
        for r_i, q_i, h_i in channels:
            xq = x**q_i
            hq = h_i**q_i
            fv = fv + r_i * xq / (xq + hq)
        sc = np.where(np.diff(np.sign(fv)))[0]
        if len(sc) < 3:
            continue

        def f_mc(xx):
            val = a - b * xx
            for r_i, q_i, h_i in channels:
                val += r_i * xx**q_i / (xx**q_i + h_i**q_i)
            return val

        roots = []
        for j in sc[:4]:
            try:
                root = brentq(f_mc, x[j], x[j+1], xtol=1e-10)
                roots.append(root)
            except:
                pass
        if len(roots) < 3:
            continue
        dx = 1e-7
        fps = [(f_mc(r+dx) - f_mc(r-dx)) / (2*dx) for r in roots[:3]]
        if not (fps[0] < 0 and fps[1] > 0 and fps[2] < 0):
            continue
        x_eq, x_sad = roots[0], roots[1]
        try:
            dphi, _ = quad(lambda xx: -f_mc(xx), x_eq, x_sad, limit=50)
        except:
            continue
        if dphi <= 1e-10:
            continue
        # Store closure for MFPT
        ch_copy = list(channels)
        a_copy, b_copy = a, b
        configs.append({
            'x_eq': x_eq, 'x_sad': x_sad,
            'lam_eq': fps[0], 'lam_sad': fps[1], 'dphi': dphi,
            'f_func': lambda xx, _a=a_copy, _b=b_copy, _ch=ch_copy: (
                _a - _b*xx + sum(r_i*xx**q_i/(xx**q_i+h_i**q_i) for r_i,q_i,h_i in _ch)),
        })
    return configs


def batch_find_bistable_C(N):
    """Method C: fully random single-channel parameters."""
    a_all = np.random.uniform(0.05, 0.80, N)
    b_all = np.random.uniform(0.2, 2.0, N)
    r_all = np.random.uniform(0.2, 2.0, N)
    q_all = np.random.uniform(2.0, 20.0, N)
    h_all = np.random.uniform(0.3, 2.0, N)
    x = X_SCAN
    configs = []
    for i in range(N):
        a, b, r, q, h = a_all[i], b_all[i], r_all[i], q_all[i], h_all[i]
        xq = x**q
        hq = h**q
        fv = a - b * x + r * xq / (xq + hq)
        sc = np.where(np.diff(np.sign(fv)))[0]
        if len(sc) < 3:
            continue
        roots = []
        for j in sc[:4]:
            try:
                root = brentq(lambda xx: a - b*xx + r*xx**q/(xx**q+hq),
                              x[j], x[j+1], xtol=1e-10)
                roots.append(root)
            except:
                pass
        if len(roots) < 3:
            continue
        fp0 = -b + r*q*roots[0]**(q-1)*hq/(roots[0]**q+hq)**2
        fp1 = -b + r*q*roots[1]**(q-1)*hq/(roots[1]**q+hq)**2
        fp2 = -b + r*q*roots[2]**(q-1)*hq/(roots[2]**q+hq)**2
        if not (fp0 < 0 and fp1 > 0 and fp2 < 0):
            continue
        x_eq, x_sad = roots[0], roots[1]
        try:
            dphi, _ = quad(lambda xx: -(a - b*xx + r*xx**q/(xx**q+hq)),
                           x_eq, x_sad, limit=50)
        except:
            continue
        if dphi <= 1e-10:
            continue
        configs.append({
            'a': a, 'b': b, 'r': r, 'q': q, 'h': h,
            'x_eq': x_eq, 'x_sad': x_sad,
            'lam_eq': fp0, 'lam_sad': fp1, 'dphi': dphi,
            'params': (a, b, r, q, h),
        })
    return configs


# ================================================================
# MFPT and sigma* (for B computation on subset)
# ================================================================

def compute_D_exact(f_func, x_eq, x_sad, lam_eq, sigma, N_grid=30000):
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
    def obj(log_s):
        s = np.exp(log_s)
        D = compute_D_exact(f_func, x_eq, x_sad, lam_eq, s)
        return np.log(max(D, 1e-30)) - np.log(D_target)
    try:
        log_s = brentq(obj, np.log(0.001), np.log(3.0), xtol=1e-5, maxiter=40)
        return np.exp(log_s)
    except:
        return None


def make_f_single(a, b, r, q, h):
    hq = h**q
    return lambda x: a - b*x + r*x**q/(x**q + hq)


# ================================================================
# Distribution fitting
# ================================================================

def fit_distributions(data, label):
    data = np.array(data)
    data = data[data > 0]
    n = len(data)
    print(f"\n  --- Distribution fits for {label} (n={n}) ---")
    print(f"  Mean={np.mean(data):.6f}  Median={np.median(data):.6f}  Std={np.std(data):.6f}")
    print(f"  Min={np.min(data):.6f}  Max={np.max(data):.6f}")
    results = {}

    loc_e, scale_e = expon.fit(data, floc=0)
    ks_e, p_e = kstest(data, 'expon', args=(loc_e, scale_e))
    results['exponential'] = {'scale': scale_e, 'ks': ks_e, 'p': p_e}
    print(f"  Exponential: alpha={scale_e:.6f}  KS={ks_e:.4f}  p={p_e:.4f}")

    shape_ln, loc_ln, scale_ln = lognorm.fit(data, floc=0)
    ks_ln, p_ln = kstest(data, 'lognorm', args=(shape_ln, loc_ln, scale_ln))
    results['lognormal'] = {'shape': shape_ln, 'scale': scale_ln, 'ks': ks_ln, 'p': p_ln}
    print(f"  Log-normal:  sigma={shape_ln:.4f} mu={np.log(scale_ln):.4f}  KS={ks_ln:.4f}  p={p_ln:.4f}")

    shape_w, loc_w, scale_w = weibull_min.fit(data, floc=0)
    ks_w, p_w = kstest(data, 'weibull_min', args=(shape_w, loc_w, scale_w))
    results['weibull'] = {'shape': shape_w, 'scale': scale_w, 'ks': ks_w, 'p': p_w}
    print(f"  Weibull:     k={shape_w:.4f} lam={scale_w:.4f}  KS={ks_w:.4f}  p={p_w:.4f}")

    best = min(results.items(), key=lambda x: x[1]['ks'])
    print(f"  >>> Best fit: {best[0]} (KS={best[1]['ks']:.4f}, p={best[1]['p']:.4f})")
    return results


def compute_exceedance(data, results, thresholds, label):
    data = np.array(data)
    data = data[data > 0]
    exp_scale = results['exponential']['scale']
    print(f"\n  --- Exceedance for {label} ---")
    print(f"  {'Thresh':>8s} {'P_emp':>10s} {'P_exp':>10s} {'1/P_emp':>12s} {'1/P_exp':>12s}")
    for thr in thresholds:
        p_emp = np.mean(data > thr)
        p_exp = np.exp(-thr / exp_scale)
        inv_emp = 1.0/p_emp if p_emp > 0 else float('inf')
        inv_exp = 1.0/p_exp
        print(f"  {thr:8.3f} {p_emp:10.6f} {p_exp:10.6f} {inv_emp:12.1f} {inv_exp:12.1f}")


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("B DISTRIBUTION TEST")
    print("=" * 70)
    sys.stdout.flush()

    all_results = {}

    # --- METHOD A ---
    print(f"\n{'='*70}")
    print(f"METHOD A: Random (a, q), fixed b=0.8, r=1.0, h=1.0")
    print(f"Generating {N_SAMPLES} configs...")
    sys.stdout.flush()
    t0 = time.time()
    cfgs_A = batch_find_bistable_A(N_SAMPLES)
    t1 = time.time()
    print(f"  Bistable: {len(cfgs_A)}/{N_SAMPLES} = {len(cfgs_A)/N_SAMPLES*100:.2f}%  ({t1-t0:.1f}s)")
    sys.stdout.flush()

    dphi_A = np.array([c['dphi'] for c in cfgs_A])
    fit_A = fit_distributions(dphi_A, "DeltaPhi (Method A)")
    compute_exceedance(dphi_A, fit_A, [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0], "DeltaPhi_A")
    sys.stdout.flush()

    # B subset for Method A
    print(f"\n  Computing B for {min(N_MFPT_SUBSET, len(cfgs_A))} configs...")
    sys.stdout.flush()
    t2 = time.time()
    idx_A = np.random.choice(len(cfgs_A), min(N_MFPT_SUBSET, len(cfgs_A)), replace=False)
    B_A = []
    for ii, i in enumerate(idx_A):
        c = cfgs_A[i]
        a, b, r, q, h = c['params']
        ff = make_f_single(a, b, r, q, h)
        ss = find_sigma_star(ff, c['x_eq'], c['x_sad'], c['lam_eq'], D_TARGET)
        if ss is not None and ss > 0:
            B_A.append(2.0 * c['dphi'] / ss**2)
        if (ii+1) % 50 == 0:
            print(f"    {ii+1}/{len(idx_A)} done, {len(B_A)} sigma* found")
            sys.stdout.flush()
    t3 = time.time()
    B_A = np.array(B_A)
    print(f"  B computed: {len(B_A)} values ({t3-t2:.1f}s)")
    fit_B_A = fit_distributions(B_A, "B (Method A)") if len(B_A) > 10 else None
    if fit_B_A:
        compute_exceedance(B_A, fit_B_A, [2, 3, 4, 5, 6, 8, 10], "B_A")
    sys.stdout.flush()

    all_results['A'] = {'n': len(cfgs_A), 'frac': len(cfgs_A)/N_SAMPLES,
                        'dphi': dphi_A, 'dphi_fit': fit_A,
                        'B': B_A, 'B_fit': fit_B_A}

    # --- METHOD B ---
    print(f"\n{'='*70}")
    print(f"METHOD B: Random multi-channel (1-3 channels)")
    print(f"Generating {N_SAMPLES} configs...")
    sys.stdout.flush()
    t0 = time.time()
    cfgs_B = batch_find_bistable_B(N_SAMPLES)
    t1 = time.time()
    print(f"  Bistable: {len(cfgs_B)}/{N_SAMPLES} = {len(cfgs_B)/N_SAMPLES*100:.2f}%  ({t1-t0:.1f}s)")
    sys.stdout.flush()

    dphi_B = np.array([c['dphi'] for c in cfgs_B])
    fit_B = fit_distributions(dphi_B, "DeltaPhi (Method B)")
    compute_exceedance(dphi_B, fit_B, [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0], "DeltaPhi_B")
    sys.stdout.flush()

    print(f"\n  Computing B for {min(N_MFPT_SUBSET, len(cfgs_B))} configs...")
    sys.stdout.flush()
    t2 = time.time()
    idx_B = np.random.choice(len(cfgs_B), min(N_MFPT_SUBSET, len(cfgs_B)), replace=False)
    B_B = []
    for ii, i in enumerate(idx_B):
        c = cfgs_B[i]
        ff = c['f_func']
        ss = find_sigma_star(ff, c['x_eq'], c['x_sad'], c['lam_eq'], D_TARGET)
        if ss is not None and ss > 0:
            B_B.append(2.0 * c['dphi'] / ss**2)
        if (ii+1) % 50 == 0:
            print(f"    {ii+1}/{len(idx_B)} done, {len(B_B)} sigma* found")
            sys.stdout.flush()
    t3 = time.time()
    B_B = np.array(B_B)
    print(f"  B computed: {len(B_B)} values ({t3-t2:.1f}s)")
    fit_B_B = fit_distributions(B_B, "B (Method B)") if len(B_B) > 10 else None
    if fit_B_B:
        compute_exceedance(B_B, fit_B_B, [2, 3, 4, 5, 6, 8, 10], "B_B")
    sys.stdout.flush()

    all_results['B'] = {'n': len(cfgs_B), 'frac': len(cfgs_B)/N_SAMPLES,
                        'dphi': dphi_B, 'dphi_fit': fit_B,
                        'B': B_B, 'B_fit': fit_B_B}

    # --- METHOD C ---
    print(f"\n{'='*70}")
    print(f"METHOD C: Fully random (a, b, r, q, h)")
    print(f"Generating {N_SAMPLES} configs...")
    sys.stdout.flush()
    t0 = time.time()
    cfgs_C = batch_find_bistable_C(N_SAMPLES)
    t1 = time.time()
    print(f"  Bistable: {len(cfgs_C)}/{N_SAMPLES} = {len(cfgs_C)/N_SAMPLES*100:.2f}%  ({t1-t0:.1f}s)")
    sys.stdout.flush()

    dphi_C = np.array([c['dphi'] for c in cfgs_C])
    fit_C = fit_distributions(dphi_C, "DeltaPhi (Method C)")
    compute_exceedance(dphi_C, fit_C, [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0], "DeltaPhi_C")
    sys.stdout.flush()

    print(f"\n  Computing B for {min(N_MFPT_SUBSET, len(cfgs_C))} configs...")
    sys.stdout.flush()
    t2 = time.time()
    idx_C = np.random.choice(len(cfgs_C), min(N_MFPT_SUBSET, len(cfgs_C)), replace=False)
    B_C = []
    for ii, i in enumerate(idx_C):
        c = cfgs_C[i]
        a, b, r, q, h = c['params']
        ff = make_f_single(a, b, r, q, h)
        ss = find_sigma_star(ff, c['x_eq'], c['x_sad'], c['lam_eq'], D_TARGET)
        if ss is not None and ss > 0:
            B_C.append(2.0 * c['dphi'] / ss**2)
        if (ii+1) % 50 == 0:
            print(f"    {ii+1}/{len(idx_C)} done, {len(B_C)} sigma* found")
            sys.stdout.flush()
    t3 = time.time()
    B_C = np.array(B_C)
    print(f"  B computed: {len(B_C)} values ({t3-t2:.1f}s)")
    fit_B_C = fit_distributions(B_C, "B (Method C)") if len(B_C) > 10 else None
    if fit_B_C:
        compute_exceedance(B_C, fit_B_C, [2, 3, 4, 5, 6, 8, 10], "B_C")
    sys.stdout.flush()

    all_results['C'] = {'n': len(cfgs_C), 'frac': len(cfgs_C)/N_SAMPLES,
                        'dphi': dphi_C, 'dphi_fit': fit_C,
                        'B': B_C, 'B_fit': fit_B_C}

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    print(f"\nDeltaPhi distribution:")
    print(f"{'Meth':>5s} {'N_bi':>7s} {'Frac%':>7s} {'mean':>9s} {'median':>9s} {'alpha':>9s} {'KS_exp':>8s} {'p_exp':>8s} {'Best':>12s}")
    print("-" * 85)
    for m in ['A', 'B', 'C']:
        R = all_results[m]
        dp = R['dphi']
        ef = R['dphi_fit']
        best = min(ef.items(), key=lambda x: x[1]['ks'])
        print(f"{m:>5s} {R['n']:>7d} {R['frac']*100:>6.2f}% {np.mean(dp):>9.5f} {np.median(dp):>9.5f} "
              f"{ef['exponential']['scale']:>9.5f} {ef['exponential']['ks']:>8.4f} {ef['exponential']['p']:>8.4f} {best[0]:>12s}")

    print(f"\nB distribution:")
    print(f"{'Meth':>5s} {'N_B':>5s} {'mean':>8s} {'median':>8s} {'alpha':>8s} {'KS':>8s} {'Best':>12s} {'P(B>2)':>8s} {'1/P':>10s}")
    print("-" * 80)
    for m in ['A', 'B', 'C']:
        R = all_results[m]
        Bv = R['B']
        if R['B_fit'] is not None:
            bf = R['B_fit']
            best = min(bf.items(), key=lambda x: x[1]['ks'])
            p2 = np.mean(Bv > 2)
            print(f"{m:>5s} {len(Bv):>5d} {np.mean(Bv):>8.3f} {np.median(Bv):>8.3f} "
                  f"{bf['exponential']['scale']:>8.4f} {bf['exponential']['ks']:>8.4f} {best[0]:>12s} "
                  f"{p2:>8.4f} {1/p2 if p2>0 else float('inf'):>10.1f}")
        else:
            print(f"{m:>5s} {len(Bv):>5d}  insufficient data")

    # KEY RESULTS
    print(f"\n{'='*70}")
    print("KEY RESULTS")
    print("=" * 70)

    print(f"\n1. Is DeltaPhi exponentially distributed?")
    for m in ['A', 'B', 'C']:
        ef = all_results[m]['dphi_fit']
        best = min(ef.items(), key=lambda x: x[1]['ks'])
        p_exp = ef['exponential']['p']
        verdict = "YES" if p_exp > 0.05 else ("MARGINAL" if p_exp > 0.01 else "NO")
        print(f"   Method {m}: Best={best[0]}, Exp p={p_exp:.4f} -> {verdict}")

    print(f"\n2. Scale parameters:")
    for m in ['A', 'B', 'C']:
        print(f"   Method {m}: alpha_Phi = {all_results[m]['dphi_fit']['exponential']['scale']:.6f}")

    print(f"\n3. Search costs from B:")
    for m in ['A', 'B', 'C']:
        Bv = all_results[m]['B']
        if len(Bv) > 0:
            for thr in [2, 4, 6]:
                p = np.mean(Bv > thr)
                inv = 1/p if p > 0 else float('inf')
                print(f"   Method {m}: 1/P(B>{thr}) = {inv:.1f}")

    print(f"\n4. Robustness:")
    bests = {m: min(all_results[m]['dphi_fit'].items(), key=lambda x: x[1]['ks'])[0] for m in ['A','B','C']}
    if len(set(bests.values())) == 1:
        print(f"   All methods agree: best fit = {list(bests.values())[0]}")
    else:
        print(f"   Methods differ: {bests}")

    print(f"\n5. Extrapolation to S0 ~ 10^13:")
    for m in ['A', 'B', 'C']:
        alpha = all_results[m]['dphi_fit']['exponential']['scale']
        dphi_13 = alpha * 13 * np.log(10)
        print(f"   Method {m}: DeltaPhi for 1/P=10^13 = {dphi_13:.4f}  (alpha={alpha:.6f})")

    print(f"\nDone.")
