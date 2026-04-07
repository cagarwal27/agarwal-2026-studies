#!/usr/bin/env python3
"""
B Computation: Topp 2000 Diabetes Model — 2D SDE Method

Computes B = ln(D_target) for the 2D (G, beta) system after eliminating
fast insulin dynamics. Parallels the Kuznetsov Method 3 approach exactly.

Citation: Topp B, Promislow K, deVries G, Miura RM, Finegood DT.
"A model of beta-cell mass, insulin, and glucose kinetics: pathways to diabetes."
J Theor Biol 206:605-619, 2000.

Usage: python3 structural_B_diabetes_2D_SDE.py
"""

import numpy as np
from numba import njit, prange
from scipy.optimize import fsolve
import time
import sys

# ============================================================
# PARAMETERS (all from Topp 2000, zero free parameters)
# ============================================================
R0 = 864.0         # mg/dL/day
EG0 = 1.44         # 1/day
SI = 0.72          # mL/(muU*day)
SIGMA_P = 43.2     # muU/(mg*day)
ALPHA = 20000.0    # (mg/dL)^2
K_INS = 432.0      # 1/day
R1 = 0.84e-3       # 1/(mg/dL * day)
R2 = 2.4e-6        # 1/(mg/dL)^2/day

# Noise reference levels (biologically motivated)
SIGMA_G_BASE = 10.0    # mg/dL/sqrt(day) — ~10% CV at healthy G
SIGMA_BETA_BASE = 0.01 # 1/sqrt(day) — ~10% relative noise on beta

# SDE simulation parameters
DT = 0.01           # days (must resolve G dynamics: tau_G ~ 0.1 days)
N_TRIALS = 500      # independent realizations per (d0, eta)
T_MAX = 100_000     # days maximum simulation time
BETA_MIN = 0.1      # reflecting boundary for beta

# MFPT target: DPP clinical data
# 7%/yr conversion → MFPT ≈ 14.3 years ≈ 5220 days
MFPT_TARGET = 5220.0

SEED = 42


# ============================================================
# NUMBA-ACCELERATED SDE KERNEL
# ============================================================

@njit(parallel=True)
def sde_kernel(N_trials, max_steps, G_init, beta_init,
               d0, sigma_G, sigma_beta, G_saddle, beta_saddle,
               dt, beta_min, seed_base):
    """Run N_trials of 2D Euler-Maruyama SDE. Returns first passage times."""
    sqrt_dt = np.sqrt(dt)
    fpt = np.full(N_trials, np.nan)
    for trial in prange(N_trials):
        np.random.seed(seed_base + trial)
        G = G_init[trial]
        beta = beta_init[trial]
        for step in range(max_steps):
            I_val = SIGMA_P * beta * G * G / (K_INS * (ALPHA + G * G))
            drift_G = R0 - (EG0 + SI * I_val) * G
            drift_beta = (-d0 + R1 * G - R2 * G * G) * beta
            dW_G = np.random.randn()
            dW_beta = np.random.randn()
            G += drift_G * dt + sigma_G * sqrt_dt * dW_G
            beta += drift_beta * dt + sigma_beta * beta * sqrt_dt * dW_beta
            if beta < beta_min:
                beta = 2.0 * beta_min - beta
                if beta < beta_min:
                    beta = beta_min
            if G < 0.1:
                G = 0.1
            if beta < beta_saddle or G > G_saddle:
                fpt[trial] = (step + 1) * dt
                break
    return fpt


@njit(parallel=True)
def sde_kernel_with_trajectories(N_trials, max_steps, G_init, beta_init,
                                  d0, sigma_G, sigma_beta, G_saddle, beta_saddle,
                                  dt, beta_min, seed_base, n_record, record_interval):
    """Same as sde_kernel but records trajectories for first n_record trials."""
    sqrt_dt = np.sqrt(dt)
    fpt = np.full(N_trials, np.nan)
    max_rec = max_steps // record_interval + 1
    G_traj = np.full((n_record, max_rec), np.nan)
    beta_traj = np.full((n_record, max_rec), np.nan)
    for trial in prange(N_trials):
        np.random.seed(seed_base + trial)
        G = G_init[trial]
        beta = beta_init[trial]
        if trial < n_record:
            G_traj[trial, 0] = G
            beta_traj[trial, 0] = beta
        for step in range(max_steps):
            I_val = SIGMA_P * beta * G * G / (K_INS * (ALPHA + G * G))
            drift_G = R0 - (EG0 + SI * I_val) * G
            drift_beta = (-d0 + R1 * G - R2 * G * G) * beta
            dW_G = np.random.randn()
            dW_beta = np.random.randn()
            G += drift_G * dt + sigma_G * sqrt_dt * dW_G
            beta += drift_beta * dt + sigma_beta * beta * sqrt_dt * dW_beta
            if beta < beta_min:
                beta = 2.0 * beta_min - beta
                if beta < beta_min:
                    beta = beta_min
            if G < 0.1:
                G = 0.1
            if trial < n_record and (step + 1) % record_interval == 0:
                idx = (step + 1) // record_interval
                if idx < max_rec:
                    G_traj[trial, idx] = G
                    beta_traj[trial, idx] = beta
            if beta < beta_saddle or G > G_saddle:
                fpt[trial] = (step + 1) * dt
                break
    return fpt, G_traj, beta_traj


# ============================================================
# EQUILIBRIA
# ============================================================

def I_qss(G, beta):
    return SIGMA_P * beta * G**2 / (K_INS * (ALPHA + G**2))

def f_G(G, beta, d0):
    return R0 - (EG0 + SI * I_qss(G, beta)) * G

def f_beta(G, beta, d0):
    return (-d0 + R1 * G - R2 * G**2) * beta

def system_2d(x, d0):
    G, beta = x
    return [f_G(G, beta, d0), f_beta(G, beta, d0)]

def jacobian_2d(G, beta, d0):
    dI_dG = SIGMA_P * beta * 2 * G * ALPHA / (K_INS * (ALPHA + G**2)**2)
    dI_dbeta = SIGMA_P * G**2 / (K_INS * (ALPHA + G**2))
    df_G_dG = -(EG0 + SI * I_qss(G, beta)) - SI * dI_dG * G
    df_G_dbeta = -SI * dI_dbeta * G
    df_beta_dG = (R1 - 2 * R2 * G) * beta
    df_beta_dbeta = -d0 + R1 * G - R2 * G**2
    return np.array([[df_G_dG, df_G_dbeta],
                     [df_beta_dG, df_beta_dbeta]])


def find_equilibria(d0):
    disc = R1**2 - 4 * R2 * d0
    if disc < 0:
        return None
    G_lo = (R1 - np.sqrt(disc)) / (2 * R2)
    G_hi = (R1 + np.sqrt(disc)) / (2 * R2)
    equilibria = {}
    for label, G_eq in [('healthy', G_lo), ('saddle', G_hi)]:
        if G_eq <= 0:
            continue
        beta_eq = (R0 / G_eq - EG0) * K_INS * (ALPHA + G_eq**2) / (SI * SIGMA_P * G_eq**2)
        if beta_eq <= 0:
            continue
        sol = fsolve(system_2d, [G_eq, beta_eq], args=(d0,), full_output=True)
        x_sol, info, ier, msg = sol
        if ier == 1 and x_sol[0] > 0 and x_sol[1] > 0:
            G_eq, beta_eq = x_sol
            J = jacobian_2d(G_eq, beta_eq, d0)
            eigvals = np.linalg.eigvals(J)
            equilibria[label] = {
                'G': G_eq, 'beta': beta_eq, 'eigvals': eigvals, 'J': J
            }
    G_diab = R0 / EG0
    equilibria['diabetic'] = {
        'G': G_diab, 'beta': 0.0,
        'eigvals': np.array([-EG0, -d0 + R1*G_diab - R2*G_diab**2]),
        'J': None
    }
    return equilibria


def find_bistable_range():
    d0_crit = R1**2 / (4 * R2)
    print(f"  Critical d0 (fold): {d0_crit:.6f}")
    d0_test = np.linspace(0.001, d0_crit - 0.001, 200)
    d0_bistable = []
    for d0 in d0_test:
        eq = find_equilibria(d0)
        if eq and 'healthy' in eq and 'saddle' in eq:
            if eq['healthy']['beta'] > 0 and eq['saddle']['beta'] > 0:
                d0_bistable.append(d0)
    if len(d0_bistable) == 0:
        print("ERROR: No bistable range found!")
        sys.exit(1)
    d0_min = min(d0_bistable)
    d0_max = max(d0_bistable)
    print(f"  Bistable range: d0 in [{d0_min:.4f}, {d0_max:.4f}]")
    return d0_min, d0_max, d0_crit


# ============================================================
# ADAPTIVE ETA SELECTION
# ============================================================

def pilot_eta_scan(d0, eq, seed_offset=0):
    """
    Quick pilot scan (30 trials, 1M steps = 10k days) to find the eta range
    where escapes occur. Returns 6 eta values that will produce useful data.
    Only selects eta values where pilot escape fraction >= 5%.
    """
    G_h = eq['healthy']['G']
    beta_h = eq['healthy']['beta']
    G_s = eq['saddle']['G']
    beta_s = eq['saddle']['beta']
    N_pilot = 30
    max_steps_pilot = 1_000_000  # 10k days

    eta_candidates = np.array([3, 4, 5, 6, 7, 8, 10, 12, 14, 18, 25, 35, 50, 70, 100],
                              dtype=np.float64)
    escape_fracs = np.zeros(len(eta_candidates))

    for idx, eta in enumerate(eta_candidates):
        rng = np.random.RandomState(SEED + seed_offset + int(eta * 1000))
        G_init = G_h + 0.01 * abs(G_h) * rng.randn(N_pilot)
        beta_init = beta_h + 0.01 * abs(beta_h) * rng.randn(N_pilot)
        G_init = np.maximum(G_init, 1.0)
        beta_init = np.maximum(beta_init, BETA_MIN)

        fpt = sde_kernel(N_pilot, max_steps_pilot, G_init, beta_init,
                         d0, eta * SIGMA_G_BASE, eta * SIGMA_BETA_BASE,
                         G_s, beta_s, DT, BETA_MIN,
                         SEED + seed_offset + int(eta * 100000))
        escape_fracs[idx] = np.sum(~np.isnan(fpt)) / N_pilot

        # Early stop: once we hit 100% escape at two consecutive levels, done
        if idx >= 1 and escape_fracs[idx] >= 0.95 and escape_fracs[idx-1] >= 0.95:
            escape_fracs[idx+1:] = 1.0
            break

    # Only keep eta values with >= 5% pilot escape (will produce data at full scale)
    usable = escape_fracs >= 0.05
    usable_etas = eta_candidates[usable]
    usable_fracs = escape_fracs[usable]

    if len(usable_etas) < 4:
        # Not enough usable etas — take highest available
        usable_etas = eta_candidates[-6:]
        usable_fracs = escape_fracs[-6:]

    # Select 6 well-spaced values from usable range
    if len(usable_etas) <= 6:
        selected = usable_etas
        sel_fracs = usable_fracs
    else:
        # Pick 6 evenly spaced indices
        indices = np.round(np.linspace(0, len(usable_etas)-1, 6)).astype(int)
        selected = usable_etas[indices]
        sel_fracs = usable_fracs[indices]

    return selected, sel_fracs


# ============================================================
# KRAMERS FIT AND B EXTRACTION
# ============================================================

def kramers_fit(eta_vals, mfpt_vals):
    valid = ~np.isnan(mfpt_vals) & (mfpt_vals > 0)
    if valid.sum() < 3:
        return np.nan, np.nan, 0.0
    x = 1.0 / eta_vals[valid]**2
    y = np.log(mfpt_vals[valid])
    A = np.vstack([x, np.ones(len(x))]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = result[0]
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, R2


def extract_B(slope, intercept):
    B = np.log(MFPT_TARGET) - intercept
    if B > 0 and slope > 0:
        eta_star = np.sqrt(slope / B)
    else:
        eta_star = np.nan
    return B, eta_star


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("TOPP 2000 DIABETES MODEL — 2D SDE B COMPUTATION")
    print("=" * 80)
    print()

    # ----------------------------------------------------------
    # STEP 1: Validate 2D system at d0=0.06
    # ----------------------------------------------------------
    print("STEP 1: Validate 2D system at d0 = 0.06")
    print("-" * 60)
    d0_ref = 0.06
    eq_ref = find_equilibria(d0_ref)
    if eq_ref is None:
        print("ERROR: No equilibria found at d0=0.06!")
        sys.exit(1)

    for label in ['healthy', 'saddle', 'diabetic']:
        if label not in eq_ref:
            continue
        data = eq_ref[label]
        G, beta = data['G'], data['beta']
        res_G = f_G(G, beta, d0_ref)
        res_beta = f_beta(G, beta, d0_ref)
        stable = all(np.real(data['eigvals']) < 0)
        saddle_type = (np.real(data['eigvals'][0]) * np.real(data['eigvals'][1]) < 0)
        print(f"  {label:10s}: G={G:10.3f}, beta={beta:10.3f}, "
              f"f_G={res_G:+.2e}, f_beta={res_beta:+.2e}, "
              f"eig=[{data['eigvals'][0]:.6f}, {data['eigvals'][1]:.6f}], "
              f"{'STABLE' if stable else 'SADDLE' if saddle_type else 'UNSTABLE'}")

    eig_h = eq_ref['healthy']['eigvals']
    tau_fast = 1.0 / np.max(np.abs(np.real(eig_h)))
    tau_slow = 1.0 / np.min(np.abs(np.real(eig_h)))
    print(f"\n  Timescale separation at healthy state:")
    print(f"    tau_fast = {tau_fast:.4f} days ({tau_fast*24:.2f} hours)")
    print(f"    tau_slow = {tau_slow:.2f} days ({tau_slow/365.25:.2f} years)")
    print(f"    Separation ratio: {tau_slow/tau_fast:.0f}x")
    print(f"    dt = {DT} days — resolves fast dynamics by factor {tau_fast/DT:.1f}x")
    print()

    # ----------------------------------------------------------
    # STEP 2: Find bistable range, select 10 d0 values
    # ----------------------------------------------------------
    print("STEP 2: Find bistable range and select scan points")
    print("-" * 60)
    d0_min, d0_max, d0_crit = find_bistable_range()

    d0_range = d0_max - d0_min
    d0_lo = d0_min + 0.05 * d0_range
    d0_hi = d0_max - 0.05 * d0_range
    d0_scan = np.linspace(d0_lo, d0_hi, 10)
    print(f"  10 scan points: {np.array2string(d0_scan, precision=5)}")
    print()

    # Table 1
    print("TABLE 1: Equilibria and eigenvalues at each d0")
    print("-" * 130)
    print(f"{'d0':>8s} | {'G_healthy':>10s} | {'beta_healthy':>12s} | "
          f"{'G_saddle':>10s} | {'beta_saddle':>12s} | "
          f"{'lambda_1':>14s} | {'lambda_2':>14s} | {'tau (days)':>12s}")
    print("-" * 130)

    scan_data = {}
    for d0 in d0_scan:
        eq = find_equilibria(d0)
        if eq is None or 'healthy' not in eq or 'saddle' not in eq:
            print(f"  d0={d0:.5f}: SKIPPED (no bistability)")
            continue
        eig = eq['healthy']['eigvals']
        tau = 1.0 / np.min(np.abs(np.real(eig)))
        scan_data[d0] = {'eq': eq, 'tau': tau}
        print(f"{d0:8.5f} | {eq['healthy']['G']:10.3f} | {eq['healthy']['beta']:12.3f} | "
              f"{eq['saddle']['G']:10.3f} | {eq['saddle']['beta']:12.3f} | "
              f"{eig[0]:14.6f} | {eig[1]:14.6f} | {tau:12.2f}")
    print()

    # ----------------------------------------------------------
    # JIT warm-up
    # ----------------------------------------------------------
    print("Warming up numba JIT compilation...")
    warm_G = np.array([100.0, 100.0])
    warm_beta = np.array([300.0, 300.0])
    _ = sde_kernel(2, 10, warm_G, warm_beta, 0.06, 80.0, 0.08,
                   250.0, 37.0, 0.01, 0.1, 99999)
    _ = sde_kernel_with_trajectories(2, 10, warm_G, warm_beta, 0.06, 80.0, 0.08,
                                      250.0, 37.0, 0.01, 0.1, 99998, 2, 5)
    print("  Done.")
    print()

    # ----------------------------------------------------------
    # STEP 3: Adaptive eta + full SDE simulation
    # ----------------------------------------------------------
    print("STEP 3: 2D SDE Simulation (with adaptive eta selection)")
    print("-" * 60)

    max_steps = int(T_MAX / DT)
    all_results = {}
    eta_selections = {}
    start_time = time.time()
    config_count = 0
    total_d0 = len(scan_data)

    for d0_idx, (d0, data) in enumerate(scan_data.items()):
        eq = data['eq']
        G_h = eq['healthy']['G']
        beta_h = eq['healthy']['beta']
        G_s = eq['saddle']['G']
        beta_s = eq['saddle']['beta']

        print(f"\n  === d0={d0:.5f} ({d0_idx+1}/{total_d0}) ===")
        print(f"  Pilot scan for eta range...", end="", flush=True)
        t0 = time.time()
        eta_sel, esc_fracs = pilot_eta_scan(d0, eq, seed_offset=d0_idx * 10000)
        print(f" {time.time()-t0:.1f}s")
        print(f"  Selected eta: {eta_sel}")
        print(f"  Pilot escape fracs: {np.array2string(esc_fracs, precision=2)}")
        eta_selections[d0] = eta_sel

        results_at_d0 = []

        for eta in eta_sel:
            config_count += 1
            sigma_G = eta * SIGMA_G_BASE
            sigma_beta = eta * SIGMA_BETA_BASE

            rng = np.random.RandomState(SEED + config_count * 10000)
            G_init = G_h + 0.01 * abs(G_h) * rng.randn(N_TRIALS)
            beta_init = beta_h + 0.01 * abs(beta_h) * rng.randn(N_TRIALS)
            G_init = np.maximum(G_init, 1.0)
            beta_init = np.maximum(beta_init, BETA_MIN)
            seed_base = SEED + config_count * 100000

            # Record trajectories at d0 closest to 0.06, middle eta
            record_traj = (abs(d0 - 0.06) < 0.005 and eta == eta_sel[2])

            elapsed = time.time() - start_time
            print(f"    eta={eta:5.1f} (elapsed: {elapsed/60:.1f} min) ...", end="", flush=True)

            t0 = time.time()
            if record_traj:
                fpt, G_traj_rec, beta_traj_rec = sde_kernel_with_trajectories(
                    N_TRIALS, max_steps, G_init, beta_init,
                    d0, sigma_G, sigma_beta, G_s, beta_s,
                    DT, BETA_MIN, seed_base, 50, 100)
                trajectories = {'G': G_traj_rec, 'beta': beta_traj_rec,
                               't': np.arange(G_traj_rec.shape[1]) * DT * 100}
            else:
                fpt = sde_kernel(N_TRIALS, max_steps, G_init, beta_init,
                                 d0, sigma_G, sigma_beta, G_s, beta_s,
                                 DT, BETA_MIN, seed_base)
                trajectories = None
            dt_run = time.time() - t0

            n_escaped = int(np.sum(~np.isnan(fpt)))
            escaped_fpts = fpt[~np.isnan(fpt)]
            mfpt_mean = float(np.mean(escaped_fpts)) if len(escaped_fpts) > 0 else np.nan
            mfpt_median = float(np.median(escaped_fpts)) if len(escaped_fpts) > 0 else np.nan
            censor_rate = 1.0 - n_escaped / N_TRIALS

            mfpt_str = f"MFPT={mfpt_mean:.0f}d" if not np.isnan(mfpt_mean) else "no escapes"
            print(f" {dt_run:.0f}s | {n_escaped}/{N_TRIALS} esc ({100*(1-censor_rate):.0f}%) | {mfpt_str}")

            results_at_d0.append({
                'eta': eta,
                'mfpt_mean': mfpt_mean,
                'mfpt_median': mfpt_median,
                'n_escaped': n_escaped,
                'censor_rate': censor_rate,
                'fpt': fpt,
                'trajectories': trajectories,
            })

        all_results[d0] = results_at_d0

    total_time = time.time() - start_time
    print(f"\nTotal SDE simulation time: {total_time/60:.1f} minutes")
    print()

    # ----------------------------------------------------------
    # STEP 4: Kramers Fits and B Extraction
    # ----------------------------------------------------------
    print("STEP 4: Kramers Fits and B Extraction")
    print("-" * 60)

    # Table 2: Raw results
    print("\nTABLE 2: Raw SDE Results")
    print("-" * 100)
    print(f"{'d0':>8s} | {'eta':>6s} | {'MFPT_mean':>12s} | {'MFPT_median':>12s} | "
          f"{'N_escaped':>10s} | {'censor%':>8s}")
    print("-" * 100)
    for d0, results in all_results.items():
        for r in results:
            ms = f"{r['mfpt_mean']:12.1f}" if not np.isnan(r['mfpt_mean']) else "         nan"
            md = f"{r['mfpt_median']:12.1f}" if not np.isnan(r['mfpt_median']) else "         nan"
            print(f"{d0:8.5f} | {r['eta']:6.1f} | {ms} | {md} | "
                  f"{r['n_escaped']:>5d}/{N_TRIALS:<4d} | {r['censor_rate']*100:7.1f}%")
    print()

    # Table 3: B extraction
    print("TABLE 3: B Extraction")
    print("-" * 120)
    print(f"{'d0':>8s} | {'eta_range':>16s} | {'R^2':>8s} | {'slope':>14s} | "
          f"{'intercept':>10s} | {'eta_star':>10s} | {'B':>8s} | {'tau':>10s}")
    print("-" * 120)

    B_values = []
    B_details = []

    for d0, results in all_results.items():
        tau = scan_data[d0]['tau']
        eta_arr = np.array([r['eta'] for r in results])
        mfpt_arr = np.array([r['mfpt_mean'] for r in results])
        n_esc_arr = np.array([r['n_escaped'] for r in results])

        valid = n_esc_arr > N_TRIALS * 0.5
        if valid.sum() < 3:
            print(f"{d0:8.5f} | {'—':>16s} | INSUFFICIENT DATA ({valid.sum()} valid eta values)")
            continue

        slope, intercept, R2 = kramers_fit(eta_arr[valid], mfpt_arr[valid])
        B, eta_star = extract_B(slope, intercept)

        eta_range_str = f"[{eta_arr[valid].min():.0f}-{eta_arr[valid].max():.0f}]"
        print(f"{d0:8.5f} | {eta_range_str:>16s} | {R2:8.4f} | {slope:14.2f} | "
              f"{intercept:10.4f} | {eta_star:10.4f} | {B:8.4f} | {tau:10.2f}")

        B_details.append({
            'd0': d0, 'R2': R2, 'slope': slope, 'intercept': intercept,
            'eta_star': eta_star, 'B': B, 'tau': tau,
            'n_valid': int(valid.sum()),
            'eta_used': eta_arr[valid].copy(),
            'mfpt_used': mfpt_arr[valid].copy(),
        })

        if R2 >= 0.9:
            B_values.append(B)

    print()

    # Table 4A: Summary (all valid points)
    print("TABLE 4A: Summary Statistics (all valid points)")
    print("-" * 60)

    B_std = 0.0
    B_cv = np.nan
    if len(B_values) > 0:
        B_arr = np.array(B_values)
        B_mean_all = float(np.mean(B_arr))
        B_std_all = float(np.std(B_arr, ddof=1)) if len(B_arr) > 1 else 0.0
        B_cv_all = B_std_all / abs(B_mean_all) * 100 if B_mean_all != 0 else np.nan
        print(f"  B mean:     {B_mean_all:.4f}")
        print(f"  B std:      {B_std_all:.4f}")
        print(f"  B CV:       {B_cv_all:.1f}%")
        print(f"  N points:   {len(B_arr)} (R^2 >= 0.9)")
        print(f"  NOTE: Includes strong-noise regime — see restricted fit below")
    else:
        B_mean_all = np.nan
        print("  NO VALID B VALUES (all R^2 < 0.9)")
    print()

    # ----------------------------------------------------------
    # DEFINITIVE: Restricted Kramers fit (weak-noise regime only)
    # ----------------------------------------------------------
    MFPT_CUTOFF = 30.0  # Only include data in Kramers regime
    print("TABLE 3B: RESTRICTED Kramers Fit (MFPT >= 30 days, >50% escape)")
    print("  (Definitive analysis: excludes strong-noise regime)")
    print("-" * 120)
    print(f"{'d0':>8s} | {'eta_range':>16s} | {'N pts':>5s} | {'R^2':>8s} | "
          f"{'slope':>12s} | {'intercept':>10s} | {'B':>8s} | {'eta*':>8s}")
    print("-" * 120)

    B_values_restricted = []
    B_details_restricted = []

    for d0, results in all_results.items():
        tau = scan_data[d0]['tau']
        eta_arr = np.array([r['eta'] for r in results])
        mfpt_arr = np.array([r['mfpt_mean'] for r in results])
        n_esc_arr = np.array([r['n_escaped'] for r in results])

        # Restricted: >50% escape AND MFPT >= cutoff
        valid = (n_esc_arr > N_TRIALS * 0.5) & (~np.isnan(mfpt_arr)) & (mfpt_arr >= MFPT_CUTOFF)
        if valid.sum() < 3:
            print(f"{d0:8.5f} | {'---':>16s} | {valid.sum():5d} | INSUFFICIENT")
            continue

        slope, intercept, R2 = kramers_fit(eta_arr[valid], mfpt_arr[valid])
        B, eta_star = extract_B(slope, intercept)

        eta_range_str = f"[{eta_arr[valid].min():.0f}-{eta_arr[valid].max():.0f}]"
        print(f"{d0:8.5f} | {eta_range_str:>16s} | {valid.sum():5d} | {R2:8.4f} | "
              f"{slope:12.2f} | {intercept:10.4f} | {B:8.4f} | {eta_star:8.4f}")

        B_details_restricted.append({
            'd0': d0, 'R2': R2, 'slope': slope, 'intercept': intercept,
            'eta_star': eta_star, 'B': B, 'tau': tau,
        })
        if R2 >= 0.9:
            B_values_restricted.append(B)

    print()

    # Table 4B: Summary (restricted)
    print("TABLE 4B: Summary Statistics (DEFINITIVE — restricted fit)")
    print("-" * 60)

    if len(B_values_restricted) > 0:
        B_arr = np.array(B_values_restricted)
        B_mean = float(np.mean(B_arr))
        B_std = float(np.std(B_arr, ddof=1)) if len(B_arr) > 1 else 0.0
        B_cv = B_std / abs(B_mean) * 100 if B_mean != 0 else np.nan
        B_min_val = float(np.min(B_arr))
        B_max_val = float(np.max(B_arr))
        in_zone = 1.8 <= B_mean <= 6.0

        print(f"  B mean:     {B_mean:.4f}")
        print(f"  B std:      {B_std:.4f}")
        print(f"  B CV:       {B_cv:.1f}%")
        print(f"  B range:    [{B_min_val:.4f}, {B_max_val:.4f}]")
        print(f"  N points:   {len(B_arr)} (R^2 >= 0.9)")
        print(f"  Habitable zone [1.8, 6.0]: {'YES' if in_zone else 'NO'}")
    else:
        B_mean = np.nan
        print("  NO VALID B VALUES")

    print()

    # Comparison table
    print("COMPARISON TABLE")
    print("-" * 70)
    print(f"{'System':15s} | {'B':>8s} | {'B CV':>8s} | {'Method':20s}")
    print("-" * 70)
    for name, b, cv, method in [
        ("Kelp", 2.17, "2.6%", "1D exact MFPT"),
        ("Tumor-immune", 2.73, "2.7%", "2D SDE"),
        ("Savanna", 4.04, "4.6%", "2D QPot"),
        ("Lake", 4.27, "2.0%", "1D exact MFPT"),
        ("Toggle", 4.83, "3.8%", "CME spectral"),
        ("Coral", 6.06, "2.1%", "1D exact MFPT"),
        ("Diabetes(1D)", 17.0, "0.0%*", "1D exact MFPT"),
    ]:
        print(f"{name:15s} | {b:8.2f} | {cv:>8s} | {method:20s}")
    if not np.isnan(B_mean):
        print(f"{'Diabetes(2D)':15s} | {B_mean:8.2f} | {B_cv:7.1f}% | {'2D SDE (this)':20s}")
    print("-" * 70)
    print()

    # Diagnostic: 1D vs 2D
    print("DIAGNOSTIC: 1D vs 2D at d0 ~ 0.06")
    print("-" * 60)
    d0_closest = min(scan_data.keys(), key=lambda x: abs(x - 0.06))
    for bd in B_details_restricted:
        if bd['d0'] == d0_closest:
            print(f"  d0 used:             {d0_closest:.5f}")
            print(f"  B from 1D reduction: 17.0")
            print(f"  B from 2D SDE:       {bd['B']:.4f}")
            if bd['B'] > 0:
                print(f"  Ratio 2D/1D:         {bd['B']/17.0:.4f}")
            print(f"  Kramers R^2:         {bd['R2']:.4f}")
            print(f"  Kramers slope:       {bd['slope']:.2f}")
            print(f"  Kramers intercept:   {bd['intercept']:.4f}")
            print(f"  eta*:                {bd['eta_star']:.4f}")
            break
    print()

    # Escape path analysis
    print("ESCAPE PATH ANALYSIS")
    print("-" * 60)
    for d0, results in all_results.items():
        for r in results:
            if r['trajectories'] is not None:
                traj = r['trajectories']
                eq = scan_data[d0]['eq']
                G_h_val = eq['healthy']['G']
                G_s_val = eq['saddle']['G']
                beta_s_val = eq['saddle']['beta']

                n_G_first = 0
                n_beta_first = 0
                max_G_vals = []

                for i in range(min(50, len(r['fpt']))):
                    if np.isnan(r['fpt'][i]):
                        continue
                    Gt = traj['G'][i]
                    Bt = traj['beta'][i]
                    vm = ~np.isnan(Gt)
                    Gt = Gt[vm]
                    Bt = Bt[vm]
                    if len(Gt) == 0:
                        continue

                    G_cross = np.where(Gt > G_s_val)[0]
                    beta_cross = np.where(Bt < beta_s_val)[0]
                    if len(G_cross) > 0 and (len(beta_cross) == 0 or G_cross[0] < beta_cross[0]):
                        n_G_first += 1
                    elif len(beta_cross) > 0:
                        n_beta_first += 1
                    max_G_vals.append(np.max(Gt))

                total_esc = n_G_first + n_beta_first
                if total_esc > 0:
                    print(f"  At d0={d0:.5f}, eta={r['eta']:.1f}:")
                    print(f"    Escaped via G > G_saddle: {n_G_first}/{total_esc} "
                          f"({100*n_G_first/total_esc:.0f}%)")
                    print(f"    Escaped via beta < beta_saddle: {n_beta_first}/{total_esc} "
                          f"({100*n_beta_first/total_esc:.0f}%)")
                    if max_G_vals:
                        print(f"    Mean max G: {np.mean(max_G_vals):.1f} "
                              f"(healthy: {G_h_val:.1f}, saddle: {G_s_val:.1f})")
    print()

    # Interpretation
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    if not np.isnan(B_mean):
        if 1.8 <= B_mean <= 6.0:
            print(f"B_2D = {B_mean:.2f} +/- {B_std:.2f} (CV={B_cv:.1f}%)")
            print("RESULT: B is in the habitable zone [1.8, 6.0].")
            print("The 1D reduction (B=17) overestimated the barrier.")
            print("The 2D escape path bypasses the flat zone via glucose fluctuations.")
            print("Consistent with Kuznetsov precedent (1D: 3.98, 2D: 2.73).")
        elif B_mean > 6.0 and B_mean < 17.0:
            print(f"B_2D = {B_mean:.2f} +/- {B_std:.2f} (CV={B_cv:.1f}%)")
            print("RESULT: Partial shortcut. B between habitable zone and 1D value.")
        elif B_mean >= 17.0:
            print(f"B_2D = {B_mean:.2f} +/- {B_std:.2f} (CV={B_cv:.1f}%)")
            print("RESULT: 2D SDE confirms 1D reduction. B ~ 17 is genuine.")
        elif B_mean < 1.8:
            print(f"B_2D = {B_mean:.2f} +/- {B_std:.2f} (CV={B_cv:.1f}%)")
            print("RESULT: System flips too easily in 2D. B below habitable zone.")
    else:
        print("RESULT: Computation inconclusive — insufficient valid Kramers fits.")

    print()
    print(f"Total computation time: {(time.time() - start_time)/60:.1f} minutes")

    return {
        'scan_data': scan_data,
        'all_results': all_results,
        'B_details_full': B_details,
        'B_details_restricted': B_details_restricted,
        'B_values_restricted': B_values_restricted,
        'B_mean': B_mean,
        'B_std': B_std,
        'total_time': time.time() - start_time,
    }


if __name__ == '__main__':
    results = main()
