#!/usr/bin/env python3
"""
Savanna Targeted Channel Experiment: Does targeting the mechanism equation
produce alpha ~ 0.37?

Hypothesis: The savanna's high alpha (0.844) came from ~50% of channels landing
on the G equation, where they barely affect the bistability-creating omega(G)
sigmoid in the T equation. If all channels target the T equation directly,
alpha should drop because every perturbation now competes with the bistability
mechanism.

Three variants:
  A: All channels on T equation (mechanism equation with omega sigmoid)
  B: All channels on G equation (non-mechanism equation, control)
  C: 50/50 split (replication of original experiment)

Model: Staver-Levin savanna (Xu et al. 2021, verified in step13b)
"""

import numpy as np
from scipy.optimize import fsolve
import warnings
import time

warnings.filterwarnings('ignore')

# ================================================================
# ==================== MODEL PARAMETERS ==========================
# ================================================================

BETA_SV = 0.39
MU_SV = 0.2
NU_SV = 0.1
OMEGA0_SV = 0.9
OMEGA1_SV = 0.2
THETA1_SV = 0.4
SS1_SV = 0.01

G_SAV, T_SAV = 0.5128, 0.3248
G_FOR, T_FOR = 0.3134, 0.6179
G_SAD, T_SAD = 0.4155, 0.4461

KNOWN_FPS = [(G_SAV, T_SAV), (G_FOR, T_FOR), (G_SAD, T_SAD)]

EPS_LO = 0.005
EPS_HI = 0.30
EPS_BUDGET = 0.90

K_VALUES = [1, 2, 3, 4, 5]
N_TRIALS = 2000
SEED = 42

N_TARGET = 10
N_GLOBAL = 15
PERTURBATION = 0.15
N_GRID_BASELINE = 80
FP_TOL = 1e-5
RESIDUAL_TOL = 1e-9

# ================================================================
# ==================== CORE MODEL ================================
# ================================================================


def omega_func(G):
    return OMEGA0_SV + (OMEGA1_SV - OMEGA0_SV) / (1.0 + np.exp(-(G - THETA1_SV) / SS1_SV))


def savanna_rhs_baseline(state):
    G, T = state
    S = 1.0 - G - T
    w = omega_func(G)
    dGdt = MU_SV * S + NU_SV * T - BETA_SV * G * T
    dTdt = w * S - NU_SV * T
    return np.array([dGdt, dTdt])


def make_rhs_with_channels(channels_G, channels_T, b0_G, b0_T):
    def rhs(state):
        G, T = state
        S = 1.0 - G - T
        w = omega_func(G)

        dGdt = MU_SV * S + NU_SV * T - b0_G * BETA_SV * G * T
        dTdt = w * S - b0_T * NU_SV * T

        for c, n, K, var_idx in channels_G:
            x = G if var_idx == 0 else T
            dGdt -= c * x**n / (x**n + K**n)

        for c, n, K, var_idx in channels_T:
            x = G if var_idx == 0 else T
            dTdt -= c * x**n / (x**n + K**n)

        return np.array([dGdt, dTdt])
    return rhs


# ================================================================
# ==================== FIXED-POINT FINDING =======================
# ================================================================


def classify_fp(G_fp, T_fp, rhs_func):
    f0 = rhs_func([G_fp, T_fp])
    J = np.zeros((2, 2))
    dx = 1e-7
    for j in range(2):
        state_p = [G_fp, T_fp]
        state_p[j] += dx
        fp = rhs_func(state_p)
        J[:, j] = (fp - f0) / dx
    eigs = np.linalg.eigvals(J)
    real_parts = np.real(eigs)
    if all(r < -1e-8 for r in real_parts):
        return 'stable', eigs
    elif any(r > 1e-8 for r in real_parts) and any(r < -1e-8 for r in real_parts):
        return 'saddle', eigs
    elif all(r > 1e-8 for r in real_parts):
        return 'unstable', eigs
    else:
        return 'marginal', eigs


def find_fps_from_grid(rhs_func, n_grid):
    checked = set()
    fps = []
    gs = np.linspace(0.02, 0.97, n_grid)
    ts = np.linspace(0.02, 0.97, n_grid)
    for G0 in gs:
        t_max = min(0.97, 0.98 - G0)
        if t_max < 0.02:
            continue
        for T0 in ts:
            if T0 > t_max:
                break
            try:
                sol, info, ier, msg = fsolve(rhs_func, [G0, T0], full_output=True)
                if ier == 1:
                    G_s, T_s = sol
                    res = np.sqrt(info['fvec'][0]**2 + info['fvec'][1]**2)
                    if (res < RESIDUAL_TOL and
                        G_s > 1e-4 and T_s > 1e-4 and
                        G_s + T_s < 1.0 - 1e-4):
                        key = (round(G_s, 4), round(T_s, 4))
                        if key not in checked:
                            checked.add(key)
                            fps.append((G_s, T_s))
            except Exception:
                pass
    return fps


def find_fps_hybrid(rhs_func, known_fps=KNOWN_FPS,
                    n_target=N_TARGET, n_global=N_GLOBAL, pert=PERTURBATION):
    checked = set()
    fps = []
    for G0, T0 in known_fps:
        for dG in np.linspace(-pert, pert, n_target):
            for dT in np.linspace(-pert, pert, n_target):
                Gs, Ts = G0 + dG, T0 + dT
                if Gs < 0.01 or Ts < 0.01 or Gs + Ts > 0.98:
                    continue
                try:
                    sol, info, ier, msg = fsolve(rhs_func, [Gs, Ts], full_output=True)
                    if ier == 1:
                        G_s, T_s = sol
                        res = np.sqrt(info['fvec'][0]**2 + info['fvec'][1]**2)
                        if (res < RESIDUAL_TOL and
                            G_s > 1e-4 and T_s > 1e-4 and
                            G_s + T_s < 1.0 - 1e-4):
                            key = (round(G_s, 4), round(T_s, 4))
                            if key not in checked:
                                checked.add(key)
                                fps.append((G_s, T_s))
                except Exception:
                    pass
    gs = np.linspace(0.05, 0.95, n_global)
    for G0 in gs:
        for T0 in gs:
            if G0 + T0 > 0.95:
                continue
            try:
                sol, info, ier, msg = fsolve(rhs_func, [G0, T0], full_output=True)
                if ier == 1:
                    G_s, T_s = sol
                    res = np.sqrt(info['fvec'][0]**2 + info['fvec'][1]**2)
                    if (res < RESIDUAL_TOL and
                        G_s > 1e-4 and T_s > 1e-4 and
                        G_s + T_s < 1.0 - 1e-4):
                        key = (round(G_s, 4), round(T_s, 4))
                        if key not in checked:
                            checked.add(key)
                            fps.append((G_s, T_s))
            except Exception:
                pass
    return fps


def classify_all_fps(fps, rhs_func):
    classified = []
    for G_fp, T_fp in fps:
        fp_type, eigs = classify_fp(G_fp, T_fp, rhs_func)
        classified.append({
            'G': G_fp, 'T': T_fp,
            'type': fp_type,
            'eigs': eigs,
        })
    return classified


def is_bistable(classified_fps):
    n_stable = sum(1 for fp in classified_fps if fp['type'] == 'stable')
    n_saddle = sum(1 for fp in classified_fps if fp['type'] == 'saddle')
    return n_stable >= 2 and n_saddle >= 1


def check_bistable_fast(rhs_func):
    fps = find_fps_hybrid(rhs_func)
    classified = classify_all_fps(fps, rhs_func)
    return is_bistable(classified)


# ================================================================
# ==================== CHANNEL GENERATION ========================
# ================================================================


def generate_channels_targeted(k, rng, target_mode):
    """
    Generate k random Hill regulatory channels with controlled equation targeting.

    target_mode:
      'T_only' - all channels on T equation (Variant A)
      'G_only' - all channels on G equation (Variant B)
      'random' - 50/50 random split (Variant C, replication)

    Returns: (channels_G, channels_T, b0_G, b0_T) or None if invalid
    """
    eps_vals = rng.uniform(EPS_LO, EPS_HI, size=k)
    if np.sum(eps_vals) >= EPS_BUDGET:
        return None

    n_vals = rng.integers(1, 9, size=k)
    K_vals = rng.uniform(0.1, 0.9, size=k)
    dep_vars = rng.integers(0, 2, size=k)

    # Assign target equations based on mode
    if target_mode == 'T_only':
        target_eqs = np.ones(k, dtype=int)       # all -> T equation
    elif target_mode == 'G_only':
        target_eqs = np.zeros(k, dtype=int)       # all -> G equation
    elif target_mode == 'random':
        target_eqs = rng.integers(0, 2, size=k)   # 50/50 random
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    # Baseline loss fluxes at savanna equilibrium
    total_loss_G = BETA_SV * G_SAV * T_SAV   # herbivory
    total_loss_T = NU_SV * T_SAV              # mortality

    # Per-equation epsilon budget
    eps_G_sum = np.sum(eps_vals[target_eqs == 0])
    eps_T_sum = np.sum(eps_vals[target_eqs == 1])
    if eps_G_sum >= 0.90 or eps_T_sum >= 0.90:
        return None

    b0_G = 1.0 - eps_G_sum
    b0_T = 1.0 - eps_T_sum
    if b0_G <= 0 or b0_T <= 0:
        return None

    channels_G = []
    channels_T = []

    for i in range(k):
        n_i = int(n_vals[i])
        K_i = float(K_vals[i])
        var_idx = int(dep_vars[i])

        x_eq = G_SAV if var_idx == 0 else T_SAV
        g_eq = x_eq**n_i / (x_eq**n_i + K_i**n_i)
        if g_eq < 1e-15:
            return None

        total_loss = total_loss_G if target_eqs[i] == 0 else total_loss_T
        if total_loss < 1e-15:
            return None

        c_i = eps_vals[i] * total_loss / g_eq

        channel = (c_i, n_i, K_i, var_idx)
        if target_eqs[i] == 0:
            channels_G.append(channel)
        else:
            channels_T.append(channel)

    return channels_G, channels_T, b0_G, b0_T


# ================================================================
# ==================== EXPERIMENT ================================
# ================================================================


def verify_baseline():
    print("=" * 70)
    print("BASELINE VERIFICATION")
    print("=" * 70)
    print(f"\nParameters: beta={BETA_SV}, mu={MU_SV}, nu={NU_SV}")
    print(f"           omega0={OMEGA0_SV}, omega1={OMEGA1_SV}, "
          f"theta={THETA1_SV}, ss={SS1_SV}")
    print(f"Grid: {N_GRID_BASELINE}x{N_GRID_BASELINE}\n")

    fps = find_fps_from_grid(savanna_rhs_baseline, N_GRID_BASELINE)
    classified = classify_all_fps(fps, savanna_rhs_baseline)

    n_stable = sum(1 for fp in classified if fp['type'] == 'stable')
    n_saddle = sum(1 for fp in classified if fp['type'] == 'saddle')

    print(f"Fixed points: {len(classified)} ({n_stable} stable, {n_saddle} saddle)")
    for i, fp in enumerate(classified):
        eig_str = ", ".join(f"{e:.6f}" for e in fp['eigs'])
        print(f"  FP {i+1}: G={fp['G']:.6f}, T={fp['T']:.6f}  "
              f"type={fp['type']:>8}  eigs=[{eig_str}]")

    bistable = is_bistable(classified)
    print(f"Bistable: {bistable}")

    # Cross-check hybrid finder
    fps_h = find_fps_hybrid(savanna_rhs_baseline)
    classified_h = classify_all_fps(fps_h, savanna_rhs_baseline)
    n_s_h = sum(1 for fp in classified_h if fp['type'] == 'stable')
    n_d_h = sum(1 for fp in classified_h if fp['type'] == 'saddle')
    print(f"Hybrid cross-check: {len(classified_h)} FPs ({n_s_h} stable, {n_d_h} saddle)")
    if n_s_h != n_stable or n_d_h != n_saddle:
        print("  WARNING: Hybrid finder disagrees!")
    else:
        print("  OK: Hybrid agrees.")

    return bistable


def run_variant(variant_name, target_mode, k_values=K_VALUES,
                n_trials=N_TRIALS, seed=SEED):
    """Run one variant of the experiment."""
    print(f"\n{'='*70}")
    print(f"VARIANT {variant_name}: target_mode={target_mode}")
    print(f"{'='*70}")

    rng = np.random.default_rng(seed)
    results = {}

    for k in k_values:
        t0 = time.time()
        n_valid = 0
        n_bistable = 0
        n_rejected = 0

        for trial in range(n_trials):
            arch = generate_channels_targeted(k, rng, target_mode)
            if arch is None:
                n_rejected += 1
                continue

            channels_G, channels_T, b0_G, b0_T = arch
            n_valid += 1

            rhs_func = make_rhs_with_channels(channels_G, channels_T, b0_G, b0_T)
            if check_bistable_fast(rhs_func):
                n_bistable += 1

            if (trial + 1) % 500 == 0:
                elapsed_so_far = time.time() - t0
                rate = (trial + 1) / elapsed_so_far if elapsed_so_far > 0 else 1
                eta = (n_trials - trial - 1) / rate
                fk_so_far = n_bistable / n_valid if n_valid > 0 else 0
                print(f"    k={k}: {trial+1}/{n_trials} done "
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
            'elapsed': elapsed,
        }

        print(f"  k={k}: {n_valid} valid ({n_rejected} rej), "
              f"{n_bistable} bistable, f={f_k:.4f} +/- {ci:.4f}, {elapsed:.1f}s")

    return results


def fit_exponential(k_arr, f_arr):
    mask = f_arr > 0
    k_fit = k_arr[mask]
    log_f = np.log(f_arr[mask])
    if len(k_fit) < 2:
        return None, None, None
    coeffs = np.polyfit(k_fit, log_f, 1)
    log_alpha = coeffs[0]
    log_A = coeffs[1]
    alpha = np.exp(log_alpha)
    A = np.exp(log_A)
    predicted = log_A + k_fit * log_alpha
    ss_res = np.sum((log_f - predicted)**2)
    ss_tot = np.sum((log_f - np.mean(log_f))**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return alpha, A, R2


def print_variant_table(name, results, k_values):
    print(f"\n{name}")
    print(f"{'k':>3} | {'bistable':>8} | {'valid':>6} | {'f(k)':>8} | {'95% CI':>12}")
    print("-" * 50)
    for k in k_values:
        r = results[k]
        ci_str = f"+/-{r['ci']:.4f}"
        print(f"{k:>3} | {r['n_bistable']:>8} | {r['n_valid']:>6} | "
              f"{r['f_k']:>8.4f} | {ci_str:>12}")

    k_arr = np.array(k_values, dtype=float)
    f_arr = np.array([results[k]['f_k'] for k in k_values])
    alpha, A, R2 = fit_exponential(k_arr, f_arr)
    if alpha is not None:
        print(f"alpha = {alpha:.4f}  R^2 = {R2:.4f}")
    else:
        print("Exponential fit FAILED")
    return alpha, R2


# ================================================================
# ==================== MAIN ======================================
# ================================================================

if __name__ == '__main__':
    t_start = time.time()

    print("=" * 70)
    print("=== SAVANNA TARGETED CHANNEL EXPERIMENT ===")
    print("=" * 70)
    print(f"Trials per k: {N_TRIALS}")
    print(f"k values: {K_VALUES}")
    print(f"Seed: {SEED}")
    print()

    # Verify baseline
    ok = verify_baseline()
    if not ok:
        print("\nFATAL: Baseline not bistable.")
        exit(1)

    # Timing calibration
    print(f"\n--- Timing calibration ---")
    t_cal = time.time()
    rng_cal = np.random.default_rng(999)
    n_cal = 0
    for _ in range(10):
        arch = generate_channels_targeted(2, rng_cal, 'random')
        if arch is not None:
            rhs_func = make_rhs_with_channels(*arch)
            check_bistable_fast(rhs_func)
            n_cal += 1
    if n_cal > 0:
        cal_time = (time.time() - t_cal) / n_cal
    else:
        cal_time = 0.1
    total_est = cal_time * N_TRIALS * len(K_VALUES) * 3
    print(f"  Per-trial time: ~{cal_time:.3f}s")
    print(f"  Estimated total (3 variants): ~{total_est:.0f}s ({total_est/60:.1f} min)")

    # ---- Run all three variants ----
    results_A = run_variant("A: All channels on T equation (mechanism)",
                            'T_only')
    results_B = run_variant("B: All channels on G equation (non-mechanism)",
                            'G_only')
    results_C = run_variant("C: 50/50 split (replication)",
                            'random')

    # ---- Final output ----
    print(f"\n{'='*70}")
    print("=== SAVANNA TARGETED CHANNEL EXPERIMENT — RESULTS ===")
    print(f"{'='*70}")

    alpha_A, R2_A = print_variant_table(
        "VARIANT A: All channels on T equation (mechanism equation)",
        results_A, K_VALUES)

    alpha_B, R2_B = print_variant_table(
        "\nVARIANT B: All channels on G equation (non-mechanism equation)",
        results_B, K_VALUES)

    alpha_C, R2_C = print_variant_table(
        "\nVARIANT C: 50/50 split (replication)",
        results_C, K_VALUES)

    ALPHA_1D = 0.373

    print(f"\n{'='*70}")
    print("COMPARISON:")
    print(f"{'='*70}")
    if alpha_A is not None:
        print(f"  alpha_A (T-targeted) = {alpha_A:.4f}  R^2 = {R2_A:.4f}")
    else:
        print(f"  alpha_A (T-targeted) = FAILED")
    if alpha_B is not None:
        print(f"  alpha_B (G-targeted) = {alpha_B:.4f}  R^2 = {R2_B:.4f}")
    else:
        print(f"  alpha_B (G-targeted) = FAILED")
    if alpha_C is not None:
        print(f"  alpha_C (50/50)      = {alpha_C:.4f}  R^2 = {R2_C:.4f}  (should match original 0.844)")
    else:
        print(f"  alpha_C (50/50)      = FAILED")
    print(f"  alpha_1D_lake        = {ALPHA_1D}")

    print(f"\nHYPOTHESIS TEST:")
    if alpha_A is not None and alpha_C is not None:
        test1 = alpha_A < alpha_C
        print(f"  alpha_A < alpha_C?  [{('YES' if test1 else 'NO')}] — "
              f"channels on mechanism equation are more destructive")
    else:
        print(f"  alpha_A < alpha_C?  [CANNOT TEST]")

    if alpha_B is not None and alpha_C is not None:
        test2 = alpha_B > alpha_C
        print(f"  alpha_B > alpha_C?  [{('YES' if test2 else 'NO')}] — "
              f"channels on non-mechanism equation are less destructive")
    else:
        print(f"  alpha_B > alpha_C?  [CANNOT TEST]")

    if alpha_A is not None:
        close_to_1d = abs(alpha_A - ALPHA_1D) / ALPHA_1D < 0.30
        print(f"  alpha_A ~ 0.37?     [{('YES' if close_to_1d else 'NO')}] — "
              f"mechanism-targeted alpha matches 1D intrinsic value")
    else:
        print(f"  alpha_A ~ 0.37?     [CANNOT TEST]")

    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")
