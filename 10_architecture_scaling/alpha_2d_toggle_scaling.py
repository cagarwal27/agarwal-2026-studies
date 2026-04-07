#!/usr/bin/env python3
"""
2D Architecture Scaling: Gardner Toggle Switch (2D Coupled)

Measures per-channel bistability survival probability (alpha) for the Gardner
toggle switch — a 2D system where BOTH equations are load-bearing for bistability.

Model: Gardner et al., Nature 2000
  du/dt = alpha_param / (1 + v^n) - u
  dv/dt = alpha_param / (1 + u^n) - v

Parameters: alpha_param = 8, n = 2 (well inside bistable range)

This is SYMMETRIC: both equations contain mutual repression that creates
bistability. There is no "irrelevant" equation.

Hypothesis: If alpha depends on load-bearing fraction, toggle (2/2 load-bearing)
should have alpha closer to 0.37 (1D lake) than 0.84 (savanna, 1/2 load-bearing).

Protocol mirrors alpha_2d_savanna_scaling.py.
"""

import numpy as np
from scipy.optimize import fsolve, brentq
import warnings
import time

warnings.filterwarnings('ignore')

# ================================================================
# ==================== MODEL PARAMETERS ==========================
# ================================================================

ALPHA_PARAM = 8.0
N_HILL = 2

EPS_LO = 0.005
EPS_HI = 0.30
EPS_BUDGET = 0.90

K_VALUES = [1, 2, 3, 4, 5]
N_TRIALS = 2000
SEED = 42

FP_TOL = 1e-4
RESIDUAL_TOL = 1e-9
DOMAIN_HI = 12.0

# ================================================================
# ==================== FAST NEWTON SOLVER ========================
# ================================================================


def newton_2d(rhs_func, u0, v0, maxiter=25, tol=1e-10):
    """
    Bare Newton's method for 2D system. Much faster than fsolve for
    small systems due to eliminated scipy overhead.
    Returns (u, v, converged).
    """
    u, v = float(u0), float(v0)
    h = 1e-7
    for _ in range(maxiter):
        f = rhs_func((u, v))
        f0, f1 = float(f[0]), float(f[1])
        if f0 * f0 + f1 * f1 < tol * tol:
            return u, v, True
        # Jacobian by finite difference (3 evaluations total incl. f)
        fu = rhs_func((u + h, v))
        fv = rhs_func((u, v + h))
        j00 = (float(fu[0]) - f0) / h
        j10 = (float(fu[1]) - f1) / h
        j01 = (float(fv[0]) - f0) / h
        j11 = (float(fv[1]) - f1) / h
        det = j00 * j11 - j01 * j10
        if abs(det) < 1e-15:
            return u, v, False
        du = (j11 * f0 - j01 * f1) / det
        dv = (-j10 * f0 + j00 * f1) / det
        u -= du
        v -= dv
        if u < -2 or v < -2 or u > 15 or v > 15:
            return u, v, False
    return u, v, False


def classify_fp_fast(u_fp, v_fp, rhs_func):
    """Classify fixed point by Jacobian eigenvalues. Returns type string."""
    h = 1e-7
    f = rhs_func((u_fp, v_fp))
    fu = rhs_func((u_fp + h, v_fp))
    fv = rhs_func((u_fp, v_fp + h))
    j00 = (float(fu[0]) - float(f[0])) / h
    j10 = (float(fu[1]) - float(f[1])) / h
    j01 = (float(fv[0]) - float(f[0])) / h
    j11 = (float(fv[1]) - float(f[1])) / h
    # 2D eigenvalues from trace and determinant
    tr = j00 + j11
    det = j00 * j11 - j01 * j10
    disc = tr * tr - 4 * det
    if disc >= 0:
        sq = disc ** 0.5
        e1 = (tr + sq) / 2
        e2 = (tr - sq) / 2
    else:
        # Complex eigenvalues: real part = tr/2
        e1 = tr / 2
        e2 = tr / 2
    if e1 < -1e-8 and e2 < -1e-8:
        return 'stable'
    elif (e1 > 1e-8 and e2 < -1e-8) or (e1 < -1e-8 and e2 > 1e-8):
        return 'saddle'
    elif e1 > 1e-8 and e2 > 1e-8:
        return 'unstable'
    return 'marginal'


def classify_fp_full(u_fp, v_fp, rhs_func):
    """Classify and return eigenvalues (for display)."""
    h = 1e-7
    f = rhs_func((u_fp, v_fp))
    fu = rhs_func((u_fp + h, v_fp))
    fv = rhs_func((u_fp, v_fp + h))
    J = np.array([[(float(fu[0]) - float(f[0])) / h, (float(fv[0]) - float(f[0])) / h],
                   [(float(fu[1]) - float(f[1])) / h, (float(fv[1]) - float(f[1])) / h]])
    eigs = np.linalg.eigvals(J)
    real_parts = np.real(eigs)
    if all(r < -1e-8 for r in real_parts):
        return 'stable', eigs
    elif any(r > 1e-8 for r in real_parts) and any(r < -1e-8 for r in real_parts):
        return 'saddle', eigs
    elif all(r > 1e-8 for r in real_parts):
        return 'unstable', eigs
    return 'marginal', eigs


# ================================================================
# ==================== CORE MODEL ================================
# ================================================================


def toggle_rhs_baseline(state):
    """Baseline Gardner toggle switch RHS."""
    u, v = state
    return np.array([ALPHA_PARAM / (1.0 + v * v) - u,
                     ALPHA_PARAM / (1.0 + u * u) - v])


def make_rhs_with_channels(channels_u, channels_v, b0_u, b0_v):
    """
    Create RHS for toggle with k regulatory channels.
    Uses tuples internally for speed.
    """
    # Pre-extract for speed
    cu = tuple(channels_u)
    cv = tuple(channels_v)

    def rhs(state):
        u, v = state
        dudt = ALPHA_PARAM / (1.0 + v * v) - b0_u * u
        dvdt = ALPHA_PARAM / (1.0 + u * u) - b0_v * v
        for c, n, K, var_idx in cu:
            x = u if var_idx == 0 else v
            xn = x ** n
            dudt -= c * xn / (xn + K ** n)
        for c, n, K, var_idx in cv:
            x = u if var_idx == 0 else v
            xn = x ** n
            dvdt -= c * xn / (xn + K ** n)
        return (dudt, dvdt)
    return rhs


# ================================================================
# ==================== BISTABILITY CHECK =========================
# ================================================================


def check_bistable_tiered(rhs_func, known_fps):
    """
    Tiered bistability check with fast negative detection.

    Tier 1: Newton from 3 known FPs (3 calls). If 2 stable + 1 saddle → True.
    Tier 2: If ≤1 unique FP, try 3 perturbed starts per known FP (9 more).
            If still ≤1 unique → False (fast negative).
    Tier 3: 5x5 perturbation per known FP with early exit.
    Tier 4: 10x10 global grid (only if ambiguous).
    """
    tol = FP_TOL
    found_fps = []   # list of (u, v, type)

    def try_add(u_s, v_s):
        """Try to add a new FP. Returns True if bistable confirmed."""
        if u_s < 1e-6 or v_s < 1e-6 or u_s > DOMAIN_HI or v_s > DOMAIN_HI:
            return False
        for fu, fv, _ in found_fps:
            if abs(u_s - fu) < tol and abs(v_s - fv) < tol:
                return False
        fp_type = classify_fp_fast(u_s, v_s, rhs_func)
        found_fps.append((u_s, v_s, fp_type))
        n_stable = sum(1 for _, _, t in found_fps if t == 'stable')
        n_saddle = sum(1 for _, _, t in found_fps if t == 'saddle')
        return n_stable >= 2 and n_saddle >= 1

    # Tier 1: Direct Newton from known FPs (3 calls)
    for u0, v0 in known_fps:
        u_s, v_s, ok = newton_2d(rhs_func, u0, v0)
        if ok:
            if try_add(u_s, v_s):
                return True

    n_unique = len(found_fps)
    if n_unique >= 3:
        # Found 3 FPs but not 2s+1d — check needed
        pass

    # Tier 2: Small perturbation (3 per known FP, 9 total)
    pert_small = [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (-0.5, 0.5), (0.5, -0.5)]
    for u0, v0 in known_fps:
        for du, dv in pert_small[:3]:
            us, vs = u0 + du, v0 + dv
            if us < 0.001 or vs < 0.001:
                continue
            u_s, v_s, ok = newton_2d(rhs_func, us, vs)
            if ok:
                if try_add(u_s, v_s):
                    return True

    # Fast negative: if still only 1 unique FP after 12 Newton calls, very likely mono
    if len(found_fps) <= 1:
        return False

    # Tier 3: 5x5 perturbation per known FP (75 calls)
    pert = 2.0
    for u0, v0 in known_fps:
        for du in np.linspace(-pert, pert, 5):
            for dv in np.linspace(-pert, pert, 5):
                us, vs = u0 + du, v0 + dv
                if us < 0.001 or vs < 0.001 or us > DOMAIN_HI or vs > DOMAIN_HI:
                    continue
                u_s, v_s, ok = newton_2d(rhs_func, us, vs)
                if ok:
                    if try_add(u_s, v_s):
                        return True

    # After 75 targeted calls, if still no bistability found, check if promising
    n_stable = sum(1 for _, _, t in found_fps if t == 'stable')
    n_saddle = sum(1 for _, _, t in found_fps if t == 'saddle')
    if n_stable < 2 and n_saddle < 1:
        return False  # No saddle and <2 stable — very unlikely to be bistable

    # Tier 4: 10x10 global grid (100 calls, only if ambiguous)
    grid = np.linspace(0.1, DOMAIN_HI - 0.1, 10)
    for u0 in grid:
        for v0 in grid:
            u_s, v_s, ok = newton_2d(rhs_func, u0, v0)
            if ok:
                if try_add(u_s, v_s):
                    return True

    return False


# ================================================================
# ==================== BASELINE VERIFICATION =====================
# ================================================================


def verify_baseline():
    """Step 1: Verify baseline bistability with analytic + grid methods."""
    print("=" * 70)
    print("STEP 1: VERIFY BASELINE BISTABILITY")
    print("=" * 70)
    print(f"\nModel: Gardner toggle switch")
    print(f"Parameters: alpha_param={ALPHA_PARAM}, n={N_HILL}")

    # Analytic: saddle on symmetry line
    def sym_eq(u):
        return u + u ** (N_HILL + 1) - ALPHA_PARAM
    u_saddle = brentq(sym_eq, 0.01, ALPHA_PARAM)

    def toggle_fp_eq(state):
        u, v = state
        return [ALPHA_PARAM / (1.0 + v ** N_HILL) - u,
                ALPHA_PARAM / (1.0 + u ** N_HILL) - v]

    sol_hu = fsolve(toggle_fp_eq, [ALPHA_PARAM - 0.1, 0.1], full_output=True)
    sol_hv = fsolve(toggle_fp_eq, [0.1, ALPHA_PARAM - 0.1], full_output=True)
    sol_sd = fsolve(toggle_fp_eq, [u_saddle, u_saddle], full_output=True)

    analytic_fps = []
    for sol, info, ier, msg in [sol_hu, sol_hv, sol_sd]:
        if ier == 1:
            res = np.sqrt(info['fvec'][0] ** 2 + info['fvec'][1] ** 2)
            if res < RESIDUAL_TOL:
                analytic_fps.append((sol[0], sol[1]))

    print(f"\nAnalytic/Newton fixed points found: {len(analytic_fps)}")
    for i, (u, v) in enumerate(analytic_fps):
        print(f"  FP {i + 1}: ({u:.6f}, {v:.6f})")

    # Grid verification (30x30)
    n_grid = 30
    print(f"\nGrid search: {n_grid}x{n_grid} on [0.01, {DOMAIN_HI}]")
    grid = np.linspace(0.01, DOMAIN_HI, n_grid)
    raw_fps = []
    for u0 in grid:
        for v0 in grid:
            u_s, v_s, ok = newton_2d(toggle_rhs_baseline, u0, v0)
            if ok:
                res = toggle_rhs_baseline((u_s, v_s))
                if (float(res[0]) ** 2 + float(res[1]) ** 2 < RESIDUAL_TOL ** 2
                        and u_s > 1e-6 and v_s > 1e-6
                        and u_s < DOMAIN_HI and v_s < DOMAIN_HI):
                    raw_fps.append((u_s, v_s))

    # Deduplicate
    unique_fps = []
    for u, v in raw_fps:
        is_dup = False
        for u2, v2 in unique_fps:
            if abs(u - u2) < FP_TOL and abs(v - v2) < FP_TOL:
                is_dup = True
                break
        if not is_dup:
            unique_fps.append((u, v))

    # Classify
    classified = []
    for u_fp, v_fp in unique_fps:
        fp_type, eigs = classify_fp_full(u_fp, v_fp, toggle_rhs_baseline)
        classified.append({'u': u_fp, 'v': v_fp, 'type': fp_type, 'eigs': eigs})

    n_stable = sum(1 for fp in classified if fp['type'] == 'stable')
    n_saddle = sum(1 for fp in classified if fp['type'] == 'saddle')
    n_unstable = sum(1 for fp in classified if fp['type'] == 'unstable')

    print(f"\nFixed points found: {len(classified)}")
    print(f"  Stable: {n_stable}")
    print(f"  Saddle: {n_saddle}")
    print(f"  Unstable: {n_unstable}")
    print()

    for i, fp in enumerate(classified):
        eig_str = ", ".join(f"{e.real:.6f}" for e in fp['eigs'])
        print(f"  FP {i + 1}: (u, v) = ({fp['u']:.6f}, {fp['v']:.6f})  "
              f"type={fp['type']:>8}  eigs=[{eig_str}]")

    bistable = n_stable >= 2 and n_saddle >= 1
    print(f"\nBistable: {bistable} ({n_stable} stable, {n_saddle} saddle)")

    known_fps = [(fp['u'], fp['v']) for fp in classified]

    # Cross-check tiered checker
    ok = check_bistable_tiered(toggle_rhs_baseline, known_fps)
    print(f"\nTiered checker cross-check: bistable={ok}")
    if ok != bistable:
        print("  WARNING: Tiered checker disagrees!")
    else:
        print("  OK: Tiered checker agrees.")

    stable_fps = [fp for fp in classified if fp['type'] == 'stable']
    stable_fps.sort(key=lambda fp: fp['u'], reverse=True)
    u_high = stable_fps[0]['u']
    v_high = stable_fps[0]['v']
    print(f"\nCalibration equilibrium (high-u): u={u_high:.6f}, v={v_high:.6f}")

    return classified, bistable, known_fps, u_high, v_high


# ================================================================
# ==================== CHANNEL GENERATION ========================
# ================================================================


def generate_random_channels_2d(k, rng, u_high, v_high):
    """
    Generate k random Hill regulatory channels for the toggle.
    Channels assigned 50/50 to u or v equation.
    Calibrated at the respective high-state equilibrium.
    """
    eps_vals = rng.uniform(EPS_LO, EPS_HI, size=k)
    if np.sum(eps_vals) >= EPS_BUDGET:
        return None

    n_vals = rng.integers(1, 9, size=k)
    K_vals = rng.uniform(0.5, 5.0, size=k)
    target_eqs = rng.integers(0, 2, size=k)
    dep_vars = rng.integers(0, 2, size=k)

    total_loss_u = u_high
    total_loss_v = u_high  # by symmetry

    eps_u_sum = np.sum(eps_vals[target_eqs == 0])
    eps_v_sum = np.sum(eps_vals[target_eqs == 1])
    if eps_u_sum >= 0.90 or eps_v_sum >= 0.90:
        return None

    b0_u = 1.0 - eps_u_sum
    b0_v = 1.0 - eps_v_sum
    if b0_u <= 0 or b0_v <= 0:
        return None

    channels_u = []
    channels_v = []

    for i in range(k):
        n_i = int(n_vals[i])
        K_i = float(K_vals[i])
        var_idx = int(dep_vars[i])
        eq_idx = int(target_eqs[i])

        if eq_idx == 0:
            x_eq = u_high if var_idx == 0 else v_high
            total_loss = total_loss_u
        else:
            x_eq = v_high if var_idx == 0 else u_high
            total_loss = total_loss_v

        g_eq = x_eq ** n_i / (x_eq ** n_i + K_i ** n_i)
        if g_eq < 1e-15:
            return None

        c_i = eps_vals[i] * total_loss / g_eq
        channel = (c_i, n_i, K_i, var_idx)
        if eq_idx == 0:
            channels_u.append(channel)
        else:
            channels_v.append(channel)

    return channels_u, channels_v, b0_u, b0_v


# ================================================================
# ==================== EXPERIMENT ================================
# ================================================================


def run_scaling_experiment(known_fps, u_high, v_high,
                           k_values=K_VALUES, n_trials=N_TRIALS, seed=SEED):
    """Run architecture scaling: for each k, test n_trials random architectures."""
    rng = np.random.default_rng(seed)
    results = {}

    for k in k_values:
        t0 = time.time()
        n_valid = 0
        n_bistable = 0
        n_rejected = 0

        for trial in range(n_trials):
            arch = generate_random_channels_2d(k, rng, u_high, v_high)
            if arch is None:
                n_rejected += 1
                continue

            channels_u, channels_v, b0_u, b0_v = arch
            n_valid += 1

            rhs_func = make_rhs_with_channels(channels_u, channels_v, b0_u, b0_v)
            if check_bistable_tiered(rhs_func, known_fps):
                n_bistable += 1

            if (trial + 1) % 500 == 0:
                elapsed_so_far = time.time() - t0
                rate = (trial + 1) / elapsed_so_far if elapsed_so_far > 0 else 1
                eta = (n_trials - trial - 1) / rate
                fk_so_far = n_bistable / n_valid if n_valid > 0 else 0
                print(f"    k={k}: {trial + 1}/{n_trials} done "
                      f"({n_bistable}/{n_valid} bistable, f={fk_so_far:.4f}, "
                      f"ETA {eta:.0f}s)")

        elapsed = time.time() - t0
        f_k = n_bistable / n_valid if n_valid > 0 else 0.0

        if n_valid > 0 and 0 < f_k < 1:
            ci = 1.96 * np.sqrt(f_k * (1 - f_k) / n_valid)
        else:
            ci = 0.0

        results[k] = {
            'n_valid': n_valid,
            'n_bistable': n_bistable,
            'n_rejected': n_rejected,
            'f_k': f_k,
            'ci': ci,
            'accept_rate': n_valid / n_trials if n_trials > 0 else 0.0,
            'elapsed': elapsed,
        }

        print(f"  k={k}: {n_valid} valid ({n_rejected} rej), "
              f"{n_bistable} bistable, f={f_k:.4f} +/- {ci:.4f}, "
              f"accept={results[k]['accept_rate']:.3f}, {elapsed:.1f}s")

    return results


def fit_exponential(k_arr, f_arr):
    """Fit f(k) = A * alpha^k via log-linear regression."""
    mask = f_arr > 0
    k_fit = k_arr[mask]
    log_f = np.log(f_arr[mask])

    if len(k_fit) < 2:
        return None, None, None

    coeffs = np.polyfit(k_fit, log_f, 1)
    alpha = np.exp(coeffs[0])
    A = np.exp(coeffs[1])

    predicted = coeffs[1] + k_fit * coeffs[0]
    ss_res = np.sum((log_f - predicted) ** 2)
    ss_tot = np.sum((log_f - np.mean(log_f)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return alpha, A, R2


# ================================================================
# ==================== MAIN ======================================
# ================================================================

if __name__ == '__main__':
    t_start = time.time()

    print("=" * 70)
    print("TOGGLE SWITCH ARCHITECTURE SCALING")
    print("=" * 70)
    print(f"Model: Gardner toggle (2D coupled), alpha_param={ALPHA_PARAM}, n={N_HILL}")
    print(f"Load-bearing equations: 2/2 (both u and v create bistability)")
    print(f"Trials per k: {N_TRIALS}")
    print(f"k values: {K_VALUES}")
    print(f"Seed: {SEED}")
    print()

    # ---- Step 1: Verify baseline ----
    baseline_fps, ok, known_fps, u_high, v_high = verify_baseline()

    if not ok:
        print("\nFATAL: Baseline not bistable. Check model parameters.")
        exit(1)

    # ---- Timing calibration ----
    print(f"\n--- Timing calibration ---")
    t_cal = time.time()
    rng_cal = np.random.default_rng(999)
    n_cal = 0
    for _ in range(20):
        arch = generate_random_channels_2d(2, rng_cal, u_high, v_high)
        if arch is not None:
            rhs_func = make_rhs_with_channels(*arch)
            check_bistable_tiered(rhs_func, known_fps)
            n_cal += 1
    if n_cal > 0:
        cal_time = (time.time() - t_cal) / n_cal
    else:
        cal_time = 0.1
    total_est = cal_time * N_TRIALS * len(K_VALUES)
    print(f"  Per-trial time: ~{cal_time:.4f}s")
    print(f"  Estimated total: ~{total_est:.0f}s ({total_est / 60:.1f} min)")

    # ---- Steps 2-4: Run experiment ----
    print(f"\n{'=' * 70}")
    print("STEPS 2-4: ARCHITECTURE SCALING EXPERIMENT")
    print(f"{'=' * 70}\n")

    results = run_scaling_experiment(known_fps, u_high, v_high)

    # ---- Step 5: Final output ----
    print(f"\n{'=' * 70}")
    print("=== TOGGLE SWITCH ARCHITECTURE SCALING ===")
    print(f"{'=' * 70}")
    print(f"Model: Gardner toggle (2D coupled), alpha_param={ALPHA_PARAM}, n={N_HILL}")
    print(f"Load-bearing equations: 2/2 (both u and v create bistability)")
    print()

    print("Baseline fixed points:")
    for fp in baseline_fps:
        eig_str = ", ".join(f"{e.real:.6f}" for e in fp['eigs'])
        label = "High-u" if fp['u'] > fp['v'] and fp['type'] == 'stable' else \
                "High-v" if fp['v'] > fp['u'] and fp['type'] == 'stable' else \
                "Saddle" if fp['type'] == 'saddle' else fp['type']
        print(f"  {label}: (u, v) = ({fp['u']:.4f}, {fp['v']:.4f}) "
              f"[{fp['type']}, eigenvalues: {eig_str}]")
    print()

    print("Architecture scaling:")
    print(f"{'k':>3} | {'bistable':>8} | {'valid':>6} | {'f(k)':>8} | "
          f"{'95% CI':>12} | {'time':>6}")
    print("-" * 58)
    for k in K_VALUES:
        r = results[k]
        ci_str = f"+/-{r['ci']:.4f}"
        print(f"{k:>3} | {r['n_bistable']:>8} | {r['n_valid']:>6} | "
              f"{r['f_k']:>8.4f} | {ci_str:>12} | {r['elapsed']:>5.1f}s")

    k_arr = np.array(K_VALUES, dtype=float)
    f_arr = np.array([results[k]['f_k'] for k in K_VALUES])

    print(f"\nExponential fit: f(k) = A * alpha^k")
    alpha_toggle, A_fit, R2 = fit_exponential(k_arr, f_arr)

    ALPHA_1D_LAKE = 0.373
    ALPHA_2D_SAVANNA = 0.844

    if alpha_toggle is not None:
        print(f"  alpha_toggle = {alpha_toggle:.4f}")
        print(f"  A = {A_fit:.4f}")
        print(f"  R^2 = {R2:.4f}")
        print()

        print("Comparison:")
        print(f"  alpha_1D_lake     = {ALPHA_1D_LAKE}  (1/1 equations load-bearing)")
        print(f"  alpha_2D_savanna  = {ALPHA_2D_SAVANNA}  (1/2 equations load-bearing)")
        print(f"  alpha_2D_toggle   = {alpha_toggle:.4f}  (2/2 equations load-bearing)")
        print()

        print("Load-bearing hypothesis:")
        if alpha_toggle < ALPHA_2D_SAVANNA - 0.05:
            print(f"  alpha_toggle ({alpha_toggle:.4f}) < alpha_savanna ({ALPHA_2D_SAVANNA}): "
                  f"SUPPORTED (both-coupled => harder to stay bistable)")
            if abs(alpha_toggle - ALPHA_1D_LAKE) < 0.10:
                print(f"  alpha_toggle ~ alpha_1D ({ALPHA_1D_LAKE}): "
                      f"STRONGLY SUPPORTED (alpha_intrinsic ~ {ALPHA_1D_LAKE})")
            elif alpha_toggle < ALPHA_1D_LAKE:
                print(f"  alpha_toggle ({alpha_toggle:.4f}) < alpha_1D ({ALPHA_1D_LAKE}): "
                      f"OVERSHOOTS — toggle is even more fragile than 1D")
            else:
                print(f"  alpha_toggle ({alpha_toggle:.4f}) between "
                      f"alpha_1D ({ALPHA_1D_LAKE}) and alpha_savanna ({ALPHA_2D_SAVANNA})")
        elif abs(alpha_toggle - ALPHA_2D_SAVANNA) < 0.05:
            print(f"  alpha_toggle ~ alpha_savanna ({ALPHA_2D_SAVANNA}): "
                  f"REJECTED (alpha depends on dimensionality, not coupling)")
        else:
            print(f"  alpha_toggle ({alpha_toggle:.4f}) > alpha_savanna ({ALPHA_2D_SAVANNA}): "
                  f"UNEXPECTED — toggle more robust than savanna")
    else:
        print("Exponential fit FAILED (insufficient data with f > 0)")
        print("Conclusion: INCONCLUSIVE")

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"Total runtime: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"{'=' * 70}")
