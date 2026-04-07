#!/usr/bin/env python3
"""
2D Architecture Scaling: Does f(k) = alpha^k hold for a 2D model?

Tests whether the architecture scaling law discovered in the 1D lake model
(alpha_1D = 0.373, R^2 = 0.997) is universal by repeating the experiment
on the 2D Staver-Levin savanna model (grass G, trees T).

Model: Xu et al. 2021 / Staver-Levin parametrization (verified in step13b)
  dG/dt = mu * S + nu * T - beta * G * T
  dT/dt = omega(G) * S - nu * T
  S = 1 - G - T
  omega(G) = omega0 + (omega1 - omega0) / (1 + exp(-(G - theta)/ss))

Known bistable equilibria:
  Savanna:  G=0.5128, T=0.3248 (stable)
  Forest:   G=0.3134, T=0.6179 (stable)
  Saddle:   G=0.4155, T=0.4461 (saddle)

Protocol (mirrors 1D script s0_derivation_architecture_scaling.py):
  1. Verify 2D baseline is bistable (2 stable + 1 saddle)
  2. For k = 1..5, generate 2000 random k-channel architectures
  3. Each channel adds a Hill regulatory term to one of the two equations
  4. Check if modified system is still bistable
  5. Fit f(k) = A * alpha^k and compare to 1D result (alpha = 0.373)
"""

import numpy as np
from scipy.optimize import fsolve
import warnings
import time

warnings.filterwarnings('ignore')

# ================================================================
# ==================== MODEL PARAMETERS ==========================
# ================================================================

# Staver-Levin model (Xu et al. 2021, verified in step13b/Round 5)
BETA_SV = 0.39       # herbivory rate
MU_SV = 0.2          # grass colonization
NU_SV = 0.1          # tree-to-grass transition / tree mortality
OMEGA0_SV = 0.9      # omega at low grass (high tree recruitment)
OMEGA1_SV = 0.2      # omega at high grass (fire suppresses tree recruitment)
THETA1_SV = 0.4      # sigmoid midpoint
SS1_SV = 0.01        # sigmoid steepness (sharp transition)

# Known equilibria (from step13b, verified)
G_SAV, T_SAV = 0.5128, 0.3248  # savanna (stable)
G_FOR, T_FOR = 0.3134, 0.6179  # forest (stable)
G_SAD, T_SAD = 0.4155, 0.4461  # saddle

KNOWN_FPS = [(G_SAV, T_SAV), (G_FOR, T_FOR), (G_SAD, T_SAD)]

# Channel generation parameters (same as 1D)
EPS_LO = 0.005
EPS_HI = 0.30
EPS_BUDGET = 0.90

# Experiment parameters
K_VALUES = [1, 2, 3, 4, 5]
N_TRIALS = 2000
SEED = 42

# Fixed-point finding parameters
N_TARGET = 10        # perturbation grid per known FP (NxN per FP)
N_GLOBAL = 15        # coarse global grid (NxN)
PERTURBATION = 0.15  # perturbation radius around known FPs
N_GRID_BASELINE = 80 # finer grid for baseline verification
FP_TOL = 1e-5        # deduplication tolerance
RESIDUAL_TOL = 1e-9  # residual tolerance for accepting fixed point

# ================================================================
# ==================== CORE MODEL ================================
# ================================================================


def omega_func(G):
    """
    Fire-grass feedback: sigmoid transition.
    omega ~ 0.9 when grass < theta (low grass -> high tree recruitment)
    omega ~ 0.2 when grass > theta (high grass -> fire suppresses tree recruitment)
    """
    return OMEGA0_SV + (OMEGA1_SV - OMEGA0_SV) / (1.0 + np.exp(-(G - THETA1_SV) / SS1_SV))


def savanna_rhs_baseline(state):
    """Baseline savanna model RHS (no channels)."""
    G, T = state
    S = 1.0 - G - T
    w = omega_func(G)
    dGdt = MU_SV * S + NU_SV * T - BETA_SV * G * T
    dTdt = w * S - NU_SV * T
    return np.array([dGdt, dTdt])


def make_rhs_with_channels(channels_G, channels_T, b0_G, b0_T):
    """
    Create RHS function for savanna model with k regulatory channels.

    Channels modify the loss terms:
    - G equation: herbivory beta*G*T is split into b0_G*beta*G*T + channel terms
    - T equation: mortality nu*T is split into b0_T*nu*T + channel terms

    channels_G: list of (c_i, n_i, K_i, var_idx) for G equation
    channels_T: list of (c_i, n_i, K_i, var_idx) for T equation
    b0_G, b0_T: residual fractions of baseline loss
    """
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
    """Classify a fixed point by Jacobian eigenvalues."""
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
    """Find fixed points using a full grid search (for baseline verification)."""
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
    """
    Fast hybrid fixed-point search: targeted perturbation around known FPs
    plus a coarse global grid to catch any FPs that moved far.
    """
    checked = set()
    fps = []

    # Targeted: perturbation cloud around each known FP
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

    # Global: coarse grid to catch FPs that moved far
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
    """Classify a list of fixed points. Returns list of dicts."""
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
    """Check >= 2 stable + >= 1 saddle."""
    n_stable = sum(1 for fp in classified_fps if fp['type'] == 'stable')
    n_saddle = sum(1 for fp in classified_fps if fp['type'] == 'saddle')
    return n_stable >= 2 and n_saddle >= 1


def check_bistable_fast(rhs_func):
    """Fast bistability check using hybrid search."""
    fps = find_fps_hybrid(rhs_func)
    classified = classify_all_fps(fps, rhs_func)
    return is_bistable(classified)


# ================================================================
# ==================== CHANNEL GENERATION ========================
# ================================================================


def generate_random_channels_2d(k, rng):
    """
    Generate k random Hill regulatory channels for the 2D savanna model.

    Each channel:
    - Targets one equation (G or T) randomly
    - Depends on one state variable (G or T) randomly
    - Has Hill exponent n ~ {1,...,8} and half-saturation K ~ U(0.1, 0.9)
    - Has epsilon_i ~ U(0.005, 0.30), reject if sum > 0.90 per equation
    - Strength c_i calibrated so channel absorbs eps_i of the target equation's
      total loss flux at the savanna equilibrium

    Returns: (channels_G, channels_T, b0_G, b0_T) or None if invalid
    """
    # Draw random epsilon values
    eps_vals = rng.uniform(EPS_LO, EPS_HI, size=k)
    if np.sum(eps_vals) >= EPS_BUDGET:
        return None

    # Draw channel properties
    n_vals = rng.integers(1, 9, size=k)       # Hill exponents 1-8
    K_vals = rng.uniform(0.1, 0.9, size=k)    # half-saturations
    target_eqs = rng.integers(0, 2, size=k)   # 0 = G equation, 1 = T equation
    dep_vars = rng.integers(0, 2, size=k)     # 0 = depends on G, 1 = depends on T

    # Compute baseline loss fluxes at savanna equilibrium
    total_loss_G = BETA_SV * G_SAV * T_SAV   # herbivory flux at savanna eq
    total_loss_T = NU_SV * T_SAV              # mortality flux at savanna eq

    # Check per-equation epsilon budget
    eps_G_sum = np.sum(eps_vals[target_eqs == 0])
    eps_T_sum = np.sum(eps_vals[target_eqs == 1])
    if eps_G_sum >= 0.90 or eps_T_sum >= 0.90:
        return None

    # Residual fractions
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

        # Variable value at savanna equilibrium
        x_eq = G_SAV if var_idx == 0 else T_SAV

        # Hill function value at equilibrium
        g_eq = x_eq**n_i / (x_eq**n_i + K_i**n_i)
        if g_eq < 1e-15:
            return None

        # Total loss for target equation
        total_loss = total_loss_G if target_eqs[i] == 0 else total_loss_T
        if total_loss < 1e-15:
            return None

        # Calibrate: c_i * g_eq = eps_i * total_loss
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
    """Step 1: Verify the baseline model is bistable with a thorough grid search."""
    print("=" * 70)
    print("STEP 1: VERIFY 2D BASELINE BISTABILITY")
    print("=" * 70)
    print(f"\nParameters: beta={BETA_SV}, mu={MU_SV}, nu={NU_SV}")
    print(f"           omega0={OMEGA0_SV}, omega1={OMEGA1_SV}, "
          f"theta={THETA1_SV}, ss={SS1_SV}")
    print(f"Grid: {N_GRID_BASELINE}x{N_GRID_BASELINE} (feasible triangle)\n")

    fps = find_fps_from_grid(savanna_rhs_baseline, N_GRID_BASELINE)
    classified = classify_all_fps(fps, savanna_rhs_baseline)

    n_stable = sum(1 for fp in classified if fp['type'] == 'stable')
    n_saddle = sum(1 for fp in classified if fp['type'] == 'saddle')
    n_unstable = sum(1 for fp in classified if fp['type'] == 'unstable')

    print(f"Fixed points found: {len(classified)}")
    print(f"  Stable: {n_stable}")
    print(f"  Saddle: {n_saddle}")
    print(f"  Unstable: {n_unstable}")
    print()

    for i, fp in enumerate(classified):
        eig_str = ", ".join(f"{e:.6f}" for e in fp['eigs'])
        print(f"  FP {i+1}: G={fp['G']:.6f}, T={fp['T']:.6f}  "
              f"type={fp['type']:>8}  eigs=[{eig_str}]")

    bistable = is_bistable(classified)
    print(f"\nBistable: {bistable} ({n_stable} stable, {n_saddle} saddle)")

    # Also verify hybrid finder matches
    fps_hybrid = find_fps_hybrid(savanna_rhs_baseline)
    classified_hybrid = classify_all_fps(fps_hybrid, savanna_rhs_baseline)
    n_stable_h = sum(1 for fp in classified_hybrid if fp['type'] == 'stable')
    n_saddle_h = sum(1 for fp in classified_hybrid if fp['type'] == 'saddle')
    print(f"\nHybrid finder cross-check: {len(classified_hybrid)} FPs "
          f"({n_stable_h} stable, {n_saddle_h} saddle)")
    if n_stable_h != n_stable or n_saddle_h != n_saddle:
        print("  WARNING: Hybrid finder disagrees with full grid!")
    else:
        print("  OK: Hybrid finder agrees with full grid.")

    return classified, bistable


def run_scaling_experiment(k_values=K_VALUES, n_trials=N_TRIALS, seed=SEED):
    """
    Run the architecture scaling experiment.
    For each k, generate n_trials random k-channel architectures and check bistability.
    """
    rng = np.random.default_rng(seed)
    results = {}

    for k in k_values:
        t0 = time.time()
        n_valid = 0
        n_bistable = 0
        n_rejected = 0

        for trial in range(n_trials):
            arch = generate_random_channels_2d(k, rng)
            if arch is None:
                n_rejected += 1
                continue

            channels_G, channels_T, b0_G, b0_T = arch
            n_valid += 1

            rhs_func = make_rhs_with_channels(channels_G, channels_T, b0_G, b0_T)
            if check_bistable_fast(rhs_func):
                n_bistable += 1

            # Progress reporting
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

        # Binomial CI
        if n_valid > 0 and 0 < f_k < 1:
            ci = 1.96 * np.sqrt(f_k * (1 - f_k) / n_valid)
        else:
            ci = 0.0

        accept_rate = n_valid / n_trials if n_trials > 0 else 0.0

        results[k] = {
            'n_valid': n_valid,
            'n_bistable': n_bistable,
            'n_rejected': n_rejected,
            'f_k': f_k,
            'ci': ci,
            'accept_rate': accept_rate,
            'elapsed': elapsed,
        }

        print(f"  k={k}: {n_valid} valid ({n_rejected} rej), "
              f"{n_bistable} bistable, f={f_k:.4f} +/- {ci:.4f}, "
              f"accept={accept_rate:.3f}, {elapsed:.1f}s")

    return results


def fit_exponential(k_arr, f_arr):
    """Fit f(k) = A * alpha^k via log-linear regression."""
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


# ================================================================
# ==================== MAIN ======================================
# ================================================================

if __name__ == '__main__':
    t_start = time.time()

    print("=" * 70)
    print("2D ARCHITECTURE SCALING: STAVER-LEVIN SAVANNA MODEL")
    print("=" * 70)
    print(f"Trials per k: {N_TRIALS}")
    print(f"k values: {K_VALUES}")
    print(f"Seed: {SEED}")
    print(f"FP search: hybrid (target {N_TARGET}x{N_TARGET} per known FP, "
          f"global {N_GLOBAL}x{N_GLOBAL})")
    print()

    # ---- Step 1: Verify baseline ----
    baseline_fps, ok = verify_baseline()

    if not ok:
        print("\nFATAL: Baseline not bistable. Check model parameters.")
        exit(1)

    # ---- Timing calibration ----
    print(f"\n--- Timing calibration ---")
    t_cal = time.time()
    rng_cal = np.random.default_rng(999)
    n_cal = 0
    for _ in range(10):
        arch = generate_random_channels_2d(2, rng_cal)
        if arch is not None:
            rhs_func = make_rhs_with_channels(*arch)
            check_bistable_fast(rhs_func)
            n_cal += 1
    if n_cal > 0:
        cal_time = (time.time() - t_cal) / n_cal
    else:
        cal_time = 0.1
    total_est = cal_time * N_TRIALS * len(K_VALUES)
    print(f"  Per-trial time: ~{cal_time:.3f}s")
    print(f"  Estimated total: ~{total_est:.0f}s ({total_est/60:.1f} min)")

    # ---- Steps 2-4: Run experiment ----
    print(f"\n{'='*70}")
    print("STEPS 2-4: ARCHITECTURE SCALING EXPERIMENT")
    print(f"{'='*70}\n")

    results = run_scaling_experiment()

    # ---- Step 5: Analysis and output ----
    print(f"\n{'='*70}")
    print("=== 2D ARCHITECTURE SCALING RESULTS ===")
    print(f"{'='*70}")
    print(f"Model: Staver-Levin savanna (2D)")
    print(f"Parameters: beta={BETA_SV}, mu={MU_SV}, nu={NU_SV}")
    print(f"            omega0={OMEGA0_SV}, omega1={OMEGA1_SV}")
    print(f"Trials per k: {N_TRIALS}, Seed: {SEED}\n")

    # Results table
    print(f"{'k':>3} | {'bistable':>8} | {'valid':>6} | {'rejected':>8} | "
          f"{'f(k)':>8} | {'95% CI':>12} | {'time':>6}")
    print("-" * 68)
    for k in K_VALUES:
        r = results[k]
        ci_str = f"+/-{r['ci']:.4f}"
        print(f"{k:>3} | {r['n_bistable']:>8} | {r['n_valid']:>6} | "
              f"{r['n_rejected']:>8} | {r['f_k']:>8.4f} | {ci_str:>12} | "
              f"{r['elapsed']:>5.1f}s")

    # Exponential fit
    k_arr = np.array(K_VALUES, dtype=float)
    f_arr = np.array([results[k]['f_k'] for k in K_VALUES])

    print(f"\n--- Exponential Fit: f(k) = A * alpha^k ---")
    alpha_2d, A_fit, R2 = fit_exponential(k_arr, f_arr)

    ALPHA_1D = 0.373

    if alpha_2d is not None:
        print(f"alpha_2D = {alpha_2d:.4f}")
        print(f"A        = {A_fit:.4f}")
        print(f"R^2      = {R2:.4f}")
        print()
        print(f"alpha_1D = {ALPHA_1D} (reference, lake model)")
        print(f"Ratio    = alpha_2D / alpha_1D = {alpha_2d / ALPHA_1D:.3f}")
        print()

        # Assessment
        ratio = alpha_2d / ALPHA_1D
        if 0.7 <= ratio <= 1.3 and R2 > 0.95:
            verdict = "UNIVERSAL"
            explanation = (f"alpha_2D = {alpha_2d:.4f} is within 30% of alpha_1D = {ALPHA_1D}, "
                          f"and R^2 = {R2:.4f} confirms exponential scaling.")
        elif R2 > 0.90:
            if ratio < 0.7 or ratio > 1.3:
                verdict = "MODEL-DEPENDENT (exponential scaling holds, but alpha differs)"
                explanation = (f"Exponential scaling confirmed (R^2 = {R2:.4f}), "
                              f"but alpha_2D = {alpha_2d:.4f} differs from alpha_1D = {ALPHA_1D}.")
            else:
                verdict = "LIKELY UNIVERSAL (marginal R^2)"
                explanation = (f"alpha values close (ratio = {ratio:.3f}), "
                              f"but R^2 = {R2:.4f} is only marginal.")
        else:
            verdict = "INCONCLUSIVE (poor exponential fit)"
            explanation = f"R^2 = {R2:.4f} is too low to confirm exponential scaling."

        print(f"Conclusion: alpha is {verdict}")
        print(f"  {explanation}")
    else:
        print("Exponential fit FAILED (insufficient data with f > 0)")
        print("Conclusion: INCONCLUSIVE")

    # Final timing
    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")
