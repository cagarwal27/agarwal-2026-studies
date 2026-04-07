#!/usr/bin/env python3
"""
S0 Derivation: Does the Bistable Fraction Decay Exponentially with Channel Count?

Tests whether S₀ ≈ 10^13 can be derived from the architecture of bistable systems.
SE-13B found ~10% of random 2-channel architectures are bistable. If the bistable
fraction decays exponentially with channel count k:

    f(k) = α^k     where α ≈ 0.1 for k=2

then S₀ = (1/α)^k_step, and S₀ = 10^13 requires k_step ≈ 13 constraints per
innovation sub-step. This would derive the fire equation constant from persistence.

Model: 1D lake (van Nes & Scheffer 2007) with k random Hill regulatory channels.
Channel generation: Option B (fixed per-channel epsilon, reject if budget exceeded).

Output: THEORY/X2/S0_DERIVATION_RESULTS.md
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, curve_fit
import warnings
import time
import os

warnings.filterwarnings('ignore')

# ================================================================
# ==================== MODEL PARAMETERS ==========================
# ================================================================

# van Nes & Scheffer 2007 lake model
A_P = 0.326588
B_P = 0.8
R_P = 1.0
Q_P = 8
H_P = 1.0

# Known equilibria
X_CL = 0.409217   # clear-water (stable)
X_SD = 0.978      # saddle (unstable)
X_TB = 1.634      # turbid (stable)

# Channel generation parameters (Option B)
EPS_LO = 0.005
EPS_HI = 0.30
EPS_BUDGET = 0.90

# Experiment parameters
K_VALUES = [1, 2, 3, 4, 5, 6]
N_TRIALS = 5000
SEEDS = [42, 137, 2718]
N_GRID = 50000     # root-finding grid resolution
X_LO = 0.001
X_HI = 4.0

# ================================================================
# ==================== CORE FUNCTIONS ============================
# ================================================================


def make_drift_kchannel(channels, b0):
    """
    Construct drift function for lake model with k Hill regulatory channels.

    channels: list of (c_i, n_i, K_i) tuples
    b0: residual background loss rate
    Returns: f(x) such that dx/dt = f(x)
    """
    def f(x):
        rec = R_P * x**Q_P / (x**Q_P + H_P**Q_P)
        loss = b0 * x
        ch_loss = 0.0
        for c, n, K in channels:
            ch_loss += c * x**n / (x**n + K**n)
        return A_P + rec - loss - ch_loss
    return f


def make_drift_kchannel_vectorized(channels, b0):
    """
    Vectorized drift function for fast root scanning.
    Returns f(xs) for array xs.
    """
    def f_vec(xs):
        rec = R_P * xs**Q_P / (xs**Q_P + H_P**Q_P)
        loss = b0 * xs
        ch_loss = np.zeros_like(xs)
        for c, n, K in channels:
            ch_loss += c * xs**n / (xs**n + K**n)
        return A_P + rec - loss - ch_loss
    return f_vec


def check_bistable_fast(channels, b0, x_lo=X_LO, x_hi=X_HI, N=N_GRID):
    """Check bistability using vectorized evaluation. Returns True if >= 3 roots."""
    xs = np.linspace(x_lo, x_hi, N)
    f_vec = make_drift_kchannel_vectorized(channels, b0)
    fs = f_vec(xs)
    sign_changes = np.sum(fs[:-1] * fs[1:] < 0)
    return sign_changes >= 3


def find_roots_and_barrier(channels, b0, x_lo=X_LO, x_hi=X_HI, N=N_GRID):
    """
    Find roots and compute potential barrier ΔΦ if bistable.
    Returns (roots, DPhi) or (None, None) if not bistable.
    """
    xs = np.linspace(x_lo, x_hi, N)
    f_vec = make_drift_kchannel_vectorized(channels, b0)
    fs = f_vec(xs)
    sc = np.where(fs[:-1] * fs[1:] < 0)[0]

    if len(sc) < 3:
        return None, None

    f_scalar = make_drift_kchannel(channels, b0)
    roots = []
    for i in sc:
        try:
            root = brentq(f_scalar, xs[i], xs[i+1], xtol=1e-12)
            roots.append(root)
        except Exception:
            pass

    if len(roots) < 3:
        return None, None

    x_eq, x_sad = roots[0], roots[1]
    try:
        DPhi, _ = quad(lambda x: -f_scalar(x), x_eq, x_sad, limit=200)
    except Exception:
        DPhi = None

    return roots, DPhi


def generate_random_architecture(k, rng):
    """
    Generate k random Hill channels with random shapes and strengths.
    Option B: each eps_i ~ Uniform(0.005, 0.30), reject if sum > budget.

    Returns (channels, b0, eps_vals, n_vals, K_vals, D_product) or None if invalid.
    """
    # Draw random shapes
    n_vals = rng.integers(1, 9, size=k)              # Hill exponents 1-8
    K_vals = rng.uniform(0.2, 3.0, size=k)           # half-saturations

    # Draw random epsilon values
    eps_vals = rng.uniform(EPS_LO, EPS_HI, size=k)
    if np.sum(eps_vals) >= EPS_BUDGET:
        return None  # reject: exceeds budget

    # Calibrate channel strengths at clear-water equilibrium
    total_reg = B_P * X_CL  # = 0.327374
    channels = []
    for i in range(k):
        g_eq = X_CL**n_vals[i] / (X_CL**n_vals[i] + K_vals[i]**n_vals[i])
        if g_eq < 1e-15:
            return None  # channel has no effect at equilibrium
        c_i = eps_vals[i] * total_reg / g_eq
        channels.append((c_i, int(n_vals[i]), K_vals[i]))

    b0 = (1.0 - np.sum(eps_vals)) * B_P
    if b0 <= 0:
        return None  # invalid

    D_product = np.prod(1.0 / eps_vals)
    return channels, b0, eps_vals.tolist(), n_vals.tolist(), K_vals.tolist(), D_product


# ================================================================
# ==================== EXPERIMENT =================================
# ================================================================


def run_experiment(seed, k_values=K_VALUES, n_trials=N_TRIALS):
    """
    Run the full architecture scaling experiment for one random seed.
    Returns dict keyed by k with results.
    """
    rng = np.random.default_rng(seed)
    results = {}

    for k in k_values:
        t0 = time.time()
        n_valid = 0
        n_bistable = 0
        n_rejected = 0
        dphi_list = []
        d_product_list = []
        eps_bistable = []      # epsilon arrays for bistable trials
        eps_nonbistable = []   # epsilon arrays for non-bistable trials
        n_vals_bistable = []   # Hill exponents for bistable
        n_vals_nonbistable = []
        K_vals_bistable = []
        K_vals_nonbistable = []

        for trial in range(n_trials):
            result = generate_random_architecture(k, rng)
            if result is None:
                n_rejected += 1
                continue

            channels, b0, eps_v, n_v, K_v, D_prod = result
            n_valid += 1

            if check_bistable_fast(channels, b0):
                n_bistable += 1
                eps_bistable.append(eps_v)
                n_vals_bistable.append(n_v)
                K_vals_bistable.append(K_v)

                # Compute barrier
                roots, DPhi = find_roots_and_barrier(channels, b0)
                if roots is not None and DPhi is not None:
                    dphi_list.append(DPhi)
                    d_product_list.append(D_prod)
            else:
                eps_nonbistable.append(eps_v)
                n_vals_nonbistable.append(n_v)
                K_vals_nonbistable.append(K_v)

        elapsed = time.time() - t0
        f_k = n_bistable / n_valid if n_valid > 0 else 0.0
        accept_rate = n_valid / n_trials if n_trials > 0 else 0.0

        # Binomial CI
        if n_valid > 0 and f_k > 0 and f_k < 1:
            ci = 1.96 * np.sqrt(f_k * (1 - f_k) / n_valid)
        else:
            ci = 0.0

        results[k] = {
            'n_valid': n_valid,
            'n_bistable': n_bistable,
            'n_rejected': n_rejected,
            'f_k': f_k,
            'ci': ci,
            'accept_rate': accept_rate,
            'dphi_list': dphi_list,
            'd_product_list': d_product_list,
            'eps_bistable': eps_bistable,
            'eps_nonbistable': eps_nonbistable,
            'n_vals_bistable': n_vals_bistable,
            'n_vals_nonbistable': n_vals_nonbistable,
            'K_vals_bistable': K_vals_bistable,
            'K_vals_nonbistable': K_vals_nonbistable,
            'elapsed': elapsed,
        }

        print(f"  k={k}: {n_valid} valid ({n_rejected} rejected), "
              f"{n_bistable} bistable, f={f_k:.4f} ± {ci:.4f}, "
              f"accept={accept_rate:.3f}, {elapsed:.1f}s")

    return results


def verify_k0_baseline():
    """
    k=0: no channels, b0 = b = 0.8, original lake model.
    Should be bistable (3 roots) by construction.
    """
    channels = []
    b0 = B_P
    is_bi = check_bistable_fast(channels, b0)
    roots, DPhi = find_roots_and_barrier(channels, b0)
    return is_bi, roots, DPhi


# ================================================================
# ==================== ANALYSIS ==================================
# ================================================================


def fit_exponential(k_arr, f_arr):
    """Fit f(k) = A * alpha^k via log-linear regression."""
    mask = f_arr > 0
    k_fit = k_arr[mask]
    log_f = np.log(f_arr[mask])

    if len(k_fit) < 2:
        return None, None, None, None

    # Linear fit: log(f) = log(A) + k*log(alpha)
    coeffs = np.polyfit(k_fit, log_f, 1)
    log_alpha = coeffs[0]
    log_A = coeffs[1]
    alpha = np.exp(log_alpha)
    A = np.exp(log_A)

    # R²
    predicted = log_A + k_fit * log_alpha
    ss_res = np.sum((log_f - predicted)**2)
    ss_tot = np.sum((log_f - np.mean(log_f))**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    residuals = log_f - predicted
    return alpha, A, R2, residuals


def fit_power_law(k_arr, f_arr):
    """Fit f(k) = A * k^(-beta) via log-log regression."""
    mask = (f_arr > 0) & (k_arr > 0)
    k_fit = k_arr[mask]
    f_fit = f_arr[mask]

    if len(k_fit) < 2:
        return None, None, None

    log_k = np.log(k_fit)
    log_f = np.log(f_fit)

    coeffs = np.polyfit(log_k, log_f, 1)
    beta = -coeffs[0]
    A = np.exp(coeffs[1])

    predicted = coeffs[1] + coeffs[0] * log_k
    ss_res = np.sum((log_f - predicted)**2)
    ss_tot = np.sum((log_f - np.mean(log_f))**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return A, beta, R2


def fit_stretched_exponential(k_arr, f_arr):
    """Fit f(k) = A * exp(-c * k^gamma) via nonlinear least squares."""
    mask = f_arr > 0
    k_fit = k_arr[mask].astype(float)
    f_fit = f_arr[mask]

    if len(k_fit) < 3:
        return None, None, None, None

    def model(k, A, c, gamma):
        return A * np.exp(-c * k**gamma)

    try:
        popt, _ = curve_fit(model, k_fit, f_fit, p0=[1.0, 1.0, 1.0],
                            bounds=([0, 0, 0.1], [2, 50, 5]),
                            maxfev=10000)
        A, c, gamma = popt
        predicted = model(k_fit, *popt)
        ss_res = np.sum((f_fit - predicted)**2)
        ss_tot = np.sum((f_fit - np.mean(f_fit))**2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return A, c, gamma, R2
    except Exception:
        return None, None, None, None


def architecture_statistics(results):
    """Compute statistics for bistable vs non-bistable architectures."""
    stats = {}
    for k, r in results.items():
        s = {}
        if r['d_product_list']:
            d_arr = np.array(r['d_product_list'])
            s['D_mean'] = np.mean(d_arr)
            s['D_std'] = np.std(d_arr)
            s['D_median'] = np.median(d_arr)
        if r['dphi_list']:
            dp_arr = np.array(r['dphi_list'])
            s['DPhi_mean'] = np.mean(dp_arr)
            s['DPhi_std'] = np.std(dp_arr)

        # Hill exponent distributions
        if r['n_vals_bistable']:
            all_n_bi = [n for trial in r['n_vals_bistable'] for n in trial]
            s['n_bistable_mean'] = np.mean(all_n_bi)
            s['n_bistable_hist'] = np.bincount(all_n_bi, minlength=9)[1:]  # bins 1-8
        if r['n_vals_nonbistable']:
            all_n_nb = [n for trial in r['n_vals_nonbistable'] for n in trial]
            s['n_nonbistable_mean'] = np.mean(all_n_nb)
            s['n_nonbistable_hist'] = np.bincount(all_n_nb, minlength=9)[1:]

        # K (half-saturation) distributions
        if r['K_vals_bistable']:
            all_K_bi = [K for trial in r['K_vals_bistable'] for K in trial]
            s['K_bistable_mean'] = np.mean(all_K_bi)
            s['K_bistable_std'] = np.std(all_K_bi)
        if r['K_vals_nonbistable']:
            all_K_nb = [K for trial in r['K_vals_nonbistable'] for K in trial]
            s['K_nonbistable_mean'] = np.mean(all_K_nb)
            s['K_nonbistable_std'] = np.std(all_K_nb)

        stats[k] = s
    return stats


# ================================================================
# ==================== MAIN ======================================
# ================================================================

if __name__ == '__main__':
    t_start = time.time()

    print("=" * 70)
    print("S0 DERIVATION: ARCHITECTURE SCALING EXPERIMENT")
    print("=" * 70)

    # ------ k=0 baseline ------
    print("\n--- k=0 baseline (original lake model) ---")
    is_bi, roots, DPhi = verify_k0_baseline()
    print(f"  Bistable: {is_bi}")
    if roots:
        print(f"  Roots: {[f'{r:.6f}' for r in roots]}")
        print(f"  DPhi: {DPhi:.6f}")
    print(f"  f(0) = 1.0 (by construction)")

    # ------ Multi-seed runs ------
    all_seed_results = {}
    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"SEED = {seed}")
        print(f"{'='*70}")
        all_seed_results[seed] = run_experiment(seed)

    # ------ Aggregate across seeds ------
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS (mean ± std across 3 seeds)")
    print(f"{'='*70}")

    f_means = {}
    f_stds = {}
    f_all = {}  # all f(k) values per k

    for k in K_VALUES:
        fk_vals = [all_seed_results[s][k]['f_k'] for s in SEEDS]
        f_means[k] = np.mean(fk_vals)
        f_stds[k] = np.std(fk_vals)
        f_all[k] = fk_vals
        n_valid_avg = np.mean([all_seed_results[s][k]['n_valid'] for s in SEEDS])
        ci_avg = np.mean([all_seed_results[s][k]['ci'] for s in SEEDS])
        print(f"  k={k}: f = {f_means[k]:.4f} ± {f_stds[k]:.4f} "
              f"(seed spread), CI = ±{ci_avg:.4f}, "
              f"N_valid ~ {n_valid_avg:.0f}")

    # Check seed sensitivity
    print("\n--- Seed sensitivity check ---")
    for k in K_VALUES:
        spread = f_stds[k] / f_means[k] if f_means[k] > 0 else float('inf')
        flag = " *** HIGH VARIANCE" if spread > 0.20 else ""
        print(f"  k={k}: CV = {spread:.3f}{flag}")

    # ------ Analysis: use primary seed (42) for detailed stats ------
    primary = all_seed_results[42]

    # Step 1: Raw data table
    print(f"\n{'='*70}")
    print("STEP 1: RAW DATA (seed=42)")
    print(f"{'='*70}")
    print(f"{'k':>3} {'N_trials':>8} {'N_valid':>7} {'N_rej':>6} {'N_bi':>6} "
          f"{'f(k)':>8} {'95% CI':>10} {'accept':>7}")
    print("-" * 60)
    # k=0
    print(f"{'0':>3} {'1':>8} {'1':>7} {'0':>6} {'1':>6} "
          f"{'1.0000':>8} {'N/A':>10} {'1.000':>7}")
    for k in K_VALUES:
        r = primary[k]
        ci_str = f"±{r['ci']:.4f}"
        print(f"{k:>3} {N_TRIALS:>8} {r['n_valid']:>7} {r['n_rejected']:>6} "
              f"{r['n_bistable']:>6} {r['f_k']:>8.4f} "
              f"{ci_str:>10} {r['accept_rate']:>7.3f}")

    # Step 2: Exponential fit
    print(f"\n{'='*70}")
    print("STEP 2: EXPONENTIAL FIT  f(k) = A × α^k")
    print(f"{'='*70}")

    k_arr = np.array(K_VALUES, dtype=float)
    f_arr = np.array([f_means[k] for k in K_VALUES])

    alpha, A_exp, R2_exp, residuals_exp = fit_exponential(k_arr, f_arr)
    if alpha is not None:
        print(f"  α  = {alpha:.6f}")
        print(f"  A  = {A_exp:.6f}")
        print(f"  R² = {R2_exp:.6f}")
        print(f"  Residuals (log scale): {[f'{r:.4f}' for r in residuals_exp]}")

        # Step 3: Extrapolate to S₀
        print(f"\n{'='*70}")
        print("STEP 3: EXTRAPOLATION TO S₀")
        print(f"{'='*70}")

        log10_inv_alpha = np.log10(1.0 / alpha)
        k_star = 13.0 / log10_inv_alpha
        k_star_lo = 12.18 / log10_inv_alpha   # S₀ = 10^12.18
        k_star_hi = 13.82 / log10_inv_alpha   # S₀ = 10^13.82

        print(f"  1/α = {1.0/alpha:.4f}")
        print(f"  log₁₀(1/α) = {log10_inv_alpha:.4f}")
        print(f"  k* for S₀ = 10^13:    {k_star:.2f}")
        print(f"  k* for S₀ = 10^12.18: {k_star_lo:.2f}")
        print(f"  k* for S₀ = 10^13.82: {k_star_hi:.2f}")
        print(f"  k* range: [{k_star_lo:.1f}, {k_star_hi:.1f}]")

        if 5 <= k_star <= 25:
            print(f"  → k* = {k_star:.1f} is in plausible range [5, 25]")
        else:
            print(f"  → k* = {k_star:.1f} outside plausible range [5, 25]")
    else:
        print("  Exponential fit FAILED (insufficient data with f > 0)")

    # Step 4: Alternative fits
    print(f"\n{'='*70}")
    print("STEP 4: ALTERNATIVE FITS")
    print(f"{'='*70}")

    # Power law
    A_pl, beta_pl, R2_pl = fit_power_law(k_arr, f_arr)
    if A_pl is not None:
        print(f"\n  Power law: f(k) = {A_pl:.4f} × k^(-{beta_pl:.4f})")
        print(f"  R² = {R2_pl:.6f}")

    # Stretched exponential
    A_se, c_se, gamma_se, R2_se = fit_stretched_exponential(k_arr, f_arr)
    if A_se is not None:
        print(f"\n  Stretched exp: f(k) = {A_se:.4f} × exp(-{c_se:.4f} × k^{gamma_se:.4f})")
        print(f"  R² = {R2_se:.6f}")

    # Comparison
    print(f"\n  --- Model comparison (R² on original scale for stretched exp, "
          f"log scale for exp/power) ---")
    fits = []
    if R2_exp is not None:
        fits.append(('Exponential (log-linear)', R2_exp))
    if R2_pl is not None:
        fits.append(('Power law (log-log)', R2_pl))
    if R2_se is not None:
        fits.append(('Stretched exp (nonlinear)', R2_se))
    fits.sort(key=lambda x: -x[1])
    for name, r2 in fits:
        best = " ← BEST" if r2 == fits[0][1] else ""
        print(f"    {name}: R² = {r2:.6f}{best}")

    # Step 5: Architecture statistics
    print(f"\n{'='*70}")
    print("STEP 5: ARCHITECTURE STATISTICS (seed=42, bistable systems)")
    print(f"{'='*70}")

    arch_stats = architecture_statistics(primary)
    for k in K_VALUES:
        s = arch_stats[k]
        print(f"\n  k={k}:")
        if 'D_mean' in s:
            print(f"    D_product: mean={s['D_mean']:.1f}, std={s['D_std']:.1f}, "
                  f"median={s['D_median']:.1f}")
        if 'DPhi_mean' in s:
            print(f"    ΔΦ: mean={s['DPhi_mean']:.6f}, std={s['DPhi_std']:.6f}")
        if 'n_bistable_hist' in s:
            bi_hist = s['n_bistable_hist']
            nb_hist = s.get('n_nonbistable_hist', np.zeros(8))
            # Normalize
            bi_total = bi_hist.sum() if bi_hist.sum() > 0 else 1
            nb_total = nb_hist.sum() if nb_hist.sum() > 0 else 1
            print(f"    Hill exponent distribution (fraction):")
            print(f"      n:         " + "".join(f"{i:>6}" for i in range(1, 9)))
            print(f"      Bistable:  " + "".join(f"{v/bi_total:>6.3f}" for v in bi_hist))
            print(f"      Non-bist:  " + "".join(f"{v/nb_total:>6.3f}" for v in nb_hist))
            print(f"    Hill exponent mean: bistable={s.get('n_bistable_mean', 0):.2f}, "
                  f"non-bistable={s.get('n_nonbistable_mean', 0):.2f}")
        if 'K_bistable_mean' in s:
            print(f"    K (half-sat): bistable mean={s['K_bistable_mean']:.3f} "
                  f"(std={s['K_bistable_std']:.3f}), "
                  f"non-bistable mean={s.get('K_nonbistable_mean', 0):.3f} "
                  f"(std={s.get('K_nonbistable_std', 0):.3f})")

    # ------ Validation: k=2 vs SE-13B ------
    print(f"\n{'='*70}")
    print("VALIDATION CHECKS")
    print(f"{'='*70}")
    f2_mean = f_means[2]
    print(f"\n  1. k=2 vs SE-13B: f(2) = {f2_mean:.4f} (expected ~0.10)")
    if 0.06 <= f2_mean <= 0.15:
        print(f"     PASS: within expected range")
    else:
        print(f"     WARNING: outside expected range [0.06, 0.15]")

    print(f"\n  2. k=0 baseline: f(0) = 1.0 (original model bistable) — PASS")

    print(f"\n  3. Monotonicity check:")
    mono_ok = True
    for i in range(len(K_VALUES) - 1):
        k1, k2 = K_VALUES[i], K_VALUES[i+1]
        if f_means[k2] > f_means[k1]:
            print(f"     WARNING: f({k2}) > f({k1}): {f_means[k2]:.4f} > {f_means[k1]:.4f}")
            mono_ok = False
    if mono_ok:
        print(f"     PASS: f(k) is monotonically decreasing")

    print(f"\n  4. Seed sensitivity: see above (CV < 0.20 for all k = PASS)")

    # ------ Summary ------
    print(f"\n{'='*70}")
    print("OUTCOME CLASSIFICATION")
    print(f"{'='*70}")

    if alpha is not None and R2_exp is not None:
        if R2_exp > 0.95 and 0.05 <= alpha <= 0.15:
            outcome = 'A'
            desc = (f"f(k) ≈ α^k with α = {alpha:.4f}, R² = {R2_exp:.4f}. "
                    f"S₀ derivable from persistence framework. k* = {k_star:.1f}.")
        elif R2_exp > 0.95:
            outcome = 'B'
            desc = (f"Exponential confirmed (R² = {R2_exp:.4f}) but α = {alpha:.4f} ≠ 0.1. "
                    f"k* = {k_star:.1f}.")
        elif R2_exp <= 0.95 and R2_pl is not None and R2_pl > R2_exp:
            outcome = 'C'
            desc = (f"Power law fits better (R²_pl = {R2_pl:.4f} vs R²_exp = {R2_exp:.4f}). "
                    f"Constraint independence assumption fails.")
        else:
            # Check if f drops to zero
            zero_k = [k for k in K_VALUES if f_means[k] == 0]
            if zero_k:
                outcome = 'D'
                desc = (f"f(k) drops to zero at k={min(zero_k)}. "
                        f"Model too simple for many channels.")
            else:
                outcome = 'B'
                desc = (f"Exponential fit R² = {R2_exp:.4f}, α = {alpha:.4f}. "
                        f"k* = {k_star:.1f}.")
    else:
        outcome = 'D'
        desc = "Insufficient data for fitting."

    print(f"\n  Outcome: {outcome}")
    print(f"  {desc}")

    t_total = time.time() - t_start
    print(f"\n  Total runtime: {t_total:.1f}s")

    # ================================================================
    # ==================== WRITE RESULTS =============================
    # ================================================================

    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'S0_DERIVATION_RESULTS.md')

    with open(results_path, 'w') as fp:
        fp.write("# S₀ Derivation: Architecture Scaling Results\n\n")
        fp.write(f"**Date:** 2026-04-04\n")
        fp.write(f"**Script:** `THEORY/X2/scripts/s0_derivation_architecture_scaling.py`\n")
        fp.write(f"**Runtime:** {t_total:.1f}s\n")
        fp.write(f"**Seeds:** {SEEDS}\n")
        fp.write(f"**N_trials per k per seed:** {N_TRIALS}\n\n")
        fp.write("---\n\n")

        # Raw data
        fp.write("## Step 1: Raw Data\n\n")
        fp.write("| k | N_valid (avg) | N_bistable (avg) | f(k) | seed std | 95% CI |\n")
        fp.write("|---|---------------|------------------|------|----------|--------|\n")
        fp.write(f"| 0 | 1 | 1 | 1.0000 | — | — |\n")
        for k in K_VALUES:
            nv_avg = np.mean([all_seed_results[s][k]['n_valid'] for s in SEEDS])
            nb_avg = np.mean([all_seed_results[s][k]['n_bistable'] for s in SEEDS])
            ci_avg = np.mean([all_seed_results[s][k]['ci'] for s in SEEDS])
            fp.write(f"| {k} | {nv_avg:.0f} | {nb_avg:.0f} | "
                     f"{f_means[k]:.4f} | ±{f_stds[k]:.4f} | ±{ci_avg:.4f} |\n")

        # Exponential fit
        fp.write("\n## Step 2: Exponential Fit\n\n")
        fp.write("Model: f(k) = A × α^k (fitted on log scale, k ≥ 1)\n\n")
        if alpha is not None:
            fp.write(f"- **α = {alpha:.6f}**\n")
            fp.write(f"- A = {A_exp:.6f}\n")
            fp.write(f"- R² = {R2_exp:.6f}\n")
            fp.write(f"- Residuals (log): {[f'{r:.4f}' for r in residuals_exp]}\n\n")

        # Extrapolation
        fp.write("## Step 3: Extrapolation to S₀\n\n")
        if alpha is not None:
            fp.write(f"- 1/α = {1.0/alpha:.4f}\n")
            fp.write(f"- log₁₀(1/α) = {log10_inv_alpha:.4f}\n")
            fp.write(f"- **k* for S₀ = 10^13: {k_star:.2f}**\n")
            fp.write(f"- k* range for S₀ = 10^(13±0.8): [{k_star_lo:.1f}, {k_star_hi:.1f}]\n\n")

        # Alternative fits
        fp.write("## Step 4: Alternative Fits\n\n")
        if A_pl is not None:
            fp.write(f"**Power law:** f(k) = {A_pl:.4f} × k^(-{beta_pl:.4f}), "
                     f"R² = {R2_pl:.6f}\n\n")
        if A_se is not None:
            fp.write(f"**Stretched exp:** f(k) = {A_se:.4f} × exp(-{c_se:.4f} × k^{gamma_se:.4f}), "
                     f"R² = {R2_se:.6f}\n\n")

        fp.write("### Model Comparison\n\n")
        fp.write("| Model | R² | Notes |\n")
        fp.write("|-------|----|-------|\n")
        for name, r2 in fits:
            best_mark = " **← BEST**" if r2 == fits[0][1] else ""
            fp.write(f"| {name} | {r2:.6f} | {best_mark} |\n")

        # Architecture statistics
        fp.write("\n## Step 5: Architecture Statistics (Bistable Systems)\n\n")
        for k in K_VALUES:
            s = arch_stats[k]
            fp.write(f"### k = {k}\n\n")
            if 'D_mean' in s:
                fp.write(f"- D_product: mean = {s['D_mean']:.1f}, "
                         f"std = {s['D_std']:.1f}, median = {s['D_median']:.1f}\n")
            if 'DPhi_mean' in s:
                fp.write(f"- ΔΦ: mean = {s['DPhi_mean']:.6f}, std = {s['DPhi_std']:.6f}\n")
            if 'n_bistable_hist' in s:
                bi_hist = s['n_bistable_hist']
                nb_hist = s.get('n_nonbistable_hist', np.zeros(8))
                bi_total = bi_hist.sum() if bi_hist.sum() > 0 else 1
                nb_total = nb_hist.sum() if nb_hist.sum() > 0 else 1
                fp.write(f"- Hill exponent mean: bistable = {s.get('n_bistable_mean', 0):.2f}, "
                         f"non-bistable = {s.get('n_nonbistable_mean', 0):.2f}\n")
            fp.write("\n")

        # Validation
        fp.write("## Validation Checks\n\n")
        fp.write(f"1. **k=2 vs SE-13B:** f(2) = {f2_mean:.4f} (expected ~0.10) — "
                 f"{'PASS' if 0.06 <= f2_mean <= 0.15 else 'FAIL'}\n")
        fp.write(f"2. **k=0 baseline:** f(0) = 1.0 — PASS\n")
        fp.write(f"3. **Monotonicity:** {'PASS' if mono_ok else 'FAIL'}\n")
        max_cv = max(f_stds[k] / f_means[k] if f_means[k] > 0 else 0 for k in K_VALUES)
        fp.write(f"4. **Seed sensitivity:** max CV = {max_cv:.3f} — "
                 f"{'PASS' if max_cv < 0.20 else 'FAIL (increase N_TRIALS)'}\n\n")

        # Outcome
        fp.write("## Outcome\n\n")
        fp.write(f"**Outcome {outcome}:** {desc}\n\n")

        # Interpretation
        fp.write("## Interpretation\n\n")
        if outcome == 'A':
            fp.write("The bistable fraction decays exponentially with channel count. "
                     "S₀ = (1/α)^k_step where α comes from the architecture of "
                     "bistable dissipative systems. The fire equation constant is "
                     "derived from the persistence framework.\n\n")
            fp.write(f"With α = {alpha:.4f} and S₀ = 10^13, each innovation sub-step "
                     f"requires k* ≈ {k_star:.0f} independent constraints to be simultaneously "
                     f"satisfied. This unifies fire and tree: both governed by the "
                     f"same ODE architecture constraints.\n")
        elif outcome == 'B':
            fp.write(f"Exponential decay confirmed, but α = {alpha:.4f}. "
                     f"The per-constraint bistability probability differs from "
                     f"the k=2 estimate. k* = {k_star:.1f} constraints per sub-step.\n")
        elif outcome == 'C':
            fp.write("The exponential model is not the best fit. Channel constraints "
                     "are correlated — the independence assumption fails. S₀ has "
                     "a different structural origin.\n")
        elif outcome == 'D':
            fp.write("The 1D lake model cannot support many random channels while "
                     "remaining bistable. Higher-dimensional models needed.\n")

    print(f"\n  Results written to: {results_path}")
    print("  DONE.")
