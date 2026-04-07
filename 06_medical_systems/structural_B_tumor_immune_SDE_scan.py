#!/usr/bin/env python3
"""
B INVARIANCE SCAN: KUZNETSOV 1994 — PURE 2D SDE
=================================================
No adiabatic reduction. No MAP optimization.
Just direct 2D SDE simulation at each point in the bistable range.

At each s value:
  1. Find equilibria
  2. Run vectorized 2D SDE at 6 sigma values (500 trials each)
  3. Fit Kramers: ln(MFPT) = slope/σ² + intercept
  4. Extract ΔΦ_SDE = slope/2
  5. Find σ*_2D where MFPT = 730 days (BCL1 dormancy)
  6. Compute B = slope/σ*² = ln(730) - intercept

This is the cleanest possible test: 0 approximations, 0 free parameters.
"""
import sys
import numpy as np
from scipy.optimize import brentq

def prt(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()

# ============================================================
# Parameters
# ============================================================
S_FIT = 13000.0
p_par = 0.1245
g_par = 2.019e7
m_par = 3.422e-10
d_par = 0.0412
a_par = 0.18
b_par = 2.0e-9
n_par = 1.101e-7
E_MAX = a_par / n_par

MFPT_TARGET = 730.0  # days (BCL1 dormancy)


# ============================================================
# Equilibria
# ============================================================
def T_qss(E):
    return max(0.0, (a_par - n_par * E) / (a_par * b_par))

def f_eff(E, s_val):
    T = T_qss(E)
    return s_val + p_par * E * T / (g_par + T) - m_par * E * T - d_par * E

def find_equilibria(s_val):
    E_scan = np.linspace(100.0, E_MAX - 100.0, 200000)
    T_scan = np.maximum(0.0, (a_par - n_par * E_scan) / (a_par * b_par))
    f_vals = (s_val + p_par * E_scan * T_scan / (g_par + T_scan)
              - m_par * E_scan * T_scan - d_par * E_scan)
    sc = np.where(f_vals[:-1] * f_vals[1:] < 0)[0]
    roots = []
    for i in sc:
        try:
            root = brentq(lambda E: f_eff(E, s_val), E_scan[i], E_scan[i + 1])
            roots.append((root, T_qss(root)))
        except Exception:
            pass
    return sorted(roots, key=lambda x: x[0])


# ============================================================
# Vectorized 2D SDE
# ============================================================
def sde_mfpt(s_val, E_d, T_d, E_s, sigma, n_trials=500, dt=0.05, max_days=50000):
    """
    Vectorized Euler-Maruyama. Escape = E drops below E_saddle.
    Returns (mean_mfpt, se_mfpt, fraction_escaped).
    """
    max_steps = int(max_days / dt)
    sqrt_dt = np.sqrt(dt)

    E = np.full(n_trials, E_d)
    T = np.full(n_trials, T_d)
    esc_time = np.full(n_trials, np.nan)
    active = np.ones(n_trials, dtype=bool)

    for step in range(max_steps):
        n_act = int(np.sum(active))
        if n_act == 0:
            break

        Ea, Ta = E[active], T[active]
        denom = g_par + Ta
        f1 = s_val + p_par * Ea * Ta / denom - m_par * Ea * Ta - d_par * Ea
        f2 = a_par * Ta * (1 - b_par * Ta) - n_par * Ea * Ta

        E[active] = Ea + f1 * dt + sigma * sqrt_dt * np.random.randn(n_act)
        T[active] = Ta + f2 * dt + sigma * sqrt_dt * np.random.randn(n_act)
        E[active] = np.maximum(E[active], 1.0)
        T[active] = np.maximum(T[active], 1.0)

        escaped = active & (E < E_s)
        esc_time[escaped] = (step + 1) * dt
        active[escaped] = False

    # Censored trials get max_days
    esc_time[np.isnan(esc_time)] = max_days
    frac_esc = np.sum(np.isfinite(esc_time) & (esc_time < max_days)) / n_trials
    return float(np.mean(esc_time)), float(np.std(esc_time) / np.sqrt(n_trials)), frac_esc


# ============================================================
# Kramers fit at one s value
# ============================================================
def kramers_fit_at_s(s_val, E_d, T_d, E_s,
                     sigma_list, n_trials=500, dt=0.05, max_days=50000):
    """
    Run SDE at multiple sigma, fit Kramers, return (slope, intercept, R2, data).
    """
    data = []
    for sigma in sigma_list:
        np.random.seed(42)  # reproducible
        mfpt, se, fesc = sde_mfpt(s_val, E_d, T_d, E_s, sigma,
                                   n_trials=n_trials, dt=dt, max_days=max_days)
        data.append({'sigma': sigma, 'mfpt': mfpt, 'se': se, 'fesc': fesc})

    inv_s2 = np.array([1.0 / d['sigma']**2 for d in data])
    ln_mfpt = np.array([np.log(max(d['mfpt'], 1.0)) for d in data])

    # Only use points where >50% escaped (avoid censoring bias)
    valid = np.array([d['fesc'] > 0.5 for d in data])
    if np.sum(valid) < 3:
        return np.nan, np.nan, 0.0, data

    coeffs = np.polyfit(inv_s2[valid], ln_mfpt[valid], 1)
    slope, intercept = coeffs

    pred = np.polyval(coeffs, inv_s2[valid])
    ss_res = np.sum((ln_mfpt[valid] - pred)**2)
    ss_tot = np.sum((ln_mfpt[valid] - np.mean(ln_mfpt[valid]))**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return slope, intercept, R2, data


# ============================================================
# MAIN
# ============================================================
prt("=" * 78)
prt("B INVARIANCE: KUZNETSOV TUMOR-IMMUNE — PURE 2D SDE SCAN")
prt("=" * 78)
prt(f"\nMFPT target: {MFPT_TARGET} days (BCL1 dormancy)")
prt(f"B = ln(MFPT_target) - intercept  (from Kramers fit at each s)")

# ============================================================
# Find bistable range
# ============================================================
s_test = np.linspace(100, 200000, 500)
E_coarse = np.linspace(100.0, E_MAX - 100.0, 20000)
n_eq_list = []
for sv in s_test:
    T_c = np.maximum(0.0, (a_par - n_par * E_coarse) / (a_par * b_par))
    fv = (sv + p_par * E_coarse * T_c / (g_par + T_c)
          - m_par * E_coarse * T_c - d_par * E_coarse)
    n_eq_list.append(int(np.sum(fv[:-1] * fv[1:] < 0)))
n_eq_arr = np.array(n_eq_list)
s_bi = s_test[n_eq_arr >= 3]
s_lo, s_hi = s_bi[0], s_bi[-1]
margin = 0.05 * (s_hi - s_lo)

# 10 points across bistable range
s_scan = np.linspace(s_lo + margin, s_hi - margin, 10)

prt(f"\nBistable range: s in [{s_lo:.0f}, {s_hi:.0f}]")
prt(f"Scanning {len(s_scan)} points")

# Sigma values: chosen to give MFPT from ~50 to ~5000 days
# Based on operating point: sigma=50k→2300d, sigma=120k→100d
sigma_list = [40000, 55000, 75000, 100000, 140000, 200000]

prt(f"Sigma values: {[int(s) for s in sigma_list]}")
prt(f"500 trials per sigma, dt=0.05")

# ============================================================
# Scan
# ============================================================
prt(f"\n{'s':>8s} {'E_d':>10s} {'E_s':>10s} {'slope':>12s} {'intcpt':>8s} "
    f"{'R2':>6s} {'DPhi':>12s} {'sigma*':>8s} {'B':>8s}")
prt("-" * 90)

results = []
for s_val in s_scan:
    eqs = find_equilibria(s_val)
    if len(eqs) < 3:
        prt(f"  {s_val:8.0f}  — monostable, skipping")
        continue

    E_d, T_d = eqs[2]  # dormant (highest E)
    E_s, T_s = eqs[1]  # saddle

    prt(f"  {s_val:8.0f} {E_d:10.0f} {E_s:10.0f} ", end="")

    slope, intercept, R2, data = kramers_fit_at_s(
        s_val, E_d, T_d, E_s, sigma_list,
        n_trials=500, dt=0.05, max_days=50000)

    if np.isnan(slope) or slope <= 0:
        prt(f"{'FAILED':>12s}")
        continue

    DPhi = slope / 2.0

    # B = ln(MFPT_target) - intercept
    ln_target = np.log(MFPT_TARGET)
    B = ln_target - intercept

    # sigma* where MFPT = target
    # ln(target) = slope/sigma*^2 + intercept
    # sigma*^2 = slope / (ln(target) - intercept) = slope / B
    if B <= 0:
        prt(f"{slope:12.4e} {intercept:8.3f} {R2:6.3f} — B<=0, skipping")
        continue

    sigma_star = np.sqrt(slope / B)

    results.append({
        's': s_val, 'E_d': E_d, 'E_s': E_s,
        'slope': slope, 'intercept': intercept, 'R2': R2,
        'DPhi': DPhi, 'sigma_star': sigma_star, 'B': B,
        'data': data,
    })

    prt(f"{slope:12.4e} {intercept:8.3f} {R2:6.3f} {DPhi:12.4e} "
        f"{sigma_star:8.0f} {B:8.4f}")

# ============================================================
# SUMMARY
# ============================================================
prt(f"\n{'=' * 78}")
prt("SUMMARY")
prt(f"{'=' * 78}")

if len(results) >= 3:
    B_vals = np.array([r['B'] for r in results])
    R2_vals = np.array([r['R2'] for r in results])
    DPhi_vals = np.array([r['DPhi'] for r in results])
    sig_vals = np.array([r['sigma_star'] for r in results])
    slope_vals = np.array([r['slope'] for r in results])
    intcpt_vals = np.array([r['intercept'] for r in results])

    B_mean = np.mean(B_vals)
    B_std = np.std(B_vals)
    B_cv = B_std / B_mean * 100

    prt(f"\n  Kramers fit quality: R² = {np.mean(R2_vals):.3f} "
        f"(range [{np.min(R2_vals):.3f}, {np.max(R2_vals):.3f}])")

    prt(f"\n  B = 2ΔΦ/σ*² (self-consistent 2D):")
    prt(f"    Mean:  {B_mean:.4f}")
    prt(f"    Std:   {B_std:.4f}")
    prt(f"    CV:    {B_cv:.2f}%")
    prt(f"    Range: [{np.min(B_vals):.4f}, {np.max(B_vals):.4f}]")

    hz = 1.8 <= B_mean <= 6.0
    prt(f"    Habitable zone [1.8, 6.0]: {'INSIDE' if hz else 'OUTSIDE'}")

    prt(f"\n  ΔΦ_SDE variation: {np.max(DPhi_vals)/np.min(DPhi_vals):.2f}x")
    prt(f"  σ*_2D  variation: {np.max(sig_vals)/np.min(sig_vals):.2f}x")
    prt(f"  Slope  variation: {np.max(slope_vals)/np.min(slope_vals):.2f}x")
    prt(f"  Intercept range:  [{np.min(intcpt_vals):.3f}, {np.max(intcpt_vals):.3f}]")

    # B invariance verdict
    if B_cv < 5:
        verdict = "STRONG CONSTANCY (CV < 5%)"
    elif B_cv < 10:
        verdict = "MODERATE CONSTANCY (CV < 10%)"
    elif B_cv < 20:
        verdict = "WEAK CONSTANCY (CV < 20%)"
    else:
        verdict = "NOT CONSTANT (CV >= 20%)"
    prt(f"\n  Verdict: {verdict}")

    prt(f"\n  {'System':>15s} {'B':>8s} {'CV':>8s} {'Domain':>18s}")
    prt(f"  {'-' * 53}")
    prt(f"  {'Kelp':>15s} {'2.17':>8s} {'2.6%':>8s} {'Ecology':>18s}")
    prt(f"  {'Savanna':>15s} {'4.04':>8s} {'4.6%':>8s} {'Ecology':>18s}")
    prt(f"  {'Lake':>15s} {'4.27':>8s} {'2.0%':>8s} {'Ecology':>18s}")
    prt(f"  {'Toggle':>15s} {'4.83':>8s} {'3.8%':>8s} {'Gene circuit':>18s}")
    prt(f"  {'Coral':>15s} {'6.06':>8s} {'2.1%':>8s} {'Ecology':>18s}")
    prt(f"  {'Tumor (2D SDE)':>15s} {B_mean:8.2f} {B_cv:7.1f}% {'Cancer biology':>18s}")

    # Detail table
    prt(f"\n  Full scan detail:")
    prt(f"  {'s':>8s} {'R2':>6s} {'DPhi':>12s} {'sigma*':>8s} {'B':>8s} "
        f"{'slope':>12s} {'intercept':>8s}")
    prt(f"  {'-' * 66}")
    for r in results:
        prt(f"  {r['s']:8.0f} {r['R2']:6.3f} {r['DPhi']:12.4e} "
            f"{r['sigma_star']:8.0f} {r['B']:8.4f} "
            f"{r['slope']:12.4e} {r['intercept']:8.3f}")

else:
    prt(f"\n  ERROR: only {len(results)} valid points")

prt(f"\nDone.")
