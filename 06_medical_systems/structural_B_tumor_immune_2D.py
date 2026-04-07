#!/usr/bin/env python3
"""
STRUCTURAL B TEST: KUZNETSOV 1994 — 2D QUASI-POTENTIAL (IMPROVED)
==================================================================
Proper 2D computation bypassing the invalid 1D adiabatic reduction.

Two independent methods:
  1. Geometric minimum action path (MAP) via global optimization
     (differential_evolution — avoids local minima that plagued Nelder-Mead)
  2. Vectorized 2D SDE simulation with Kramers fit to extract ΔΦ

Noise model: additive isotropic (σ_E = σ_T = σ).
Convention: MFPT ~ exp(2ΔΦ/σ²), B = 2ΔΦ/σ*².
"""
import sys
import numpy as np
from scipy.optimize import differential_evolution, brentq
from scipy.integrate import quad

# ============================================================
# Parameters (BCL1 lymphoma, Kuznetsov 1994)
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


def prt(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs)
    sys.stdout.flush()


# ============================================================
# Drift
# ============================================================
def drift_2d(E, T, s_val):
    """Scalar drift."""
    denom = g_par + T
    f1 = s_val + p_par * E * T / denom - m_par * E * T - d_par * E
    f2 = a_par * T * (1 - b_par * T) - n_par * E * T
    return f1, f2


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
    sign_ch = np.where(f_vals[:-1] * f_vals[1:] < 0)[0]
    roots = []
    for i in sign_ch:
        try:
            root = brentq(lambda E: f_eff(E, s_val), E_scan[i], E_scan[i + 1])
            roots.append((root, T_qss(root)))
        except Exception:
            pass
    return sorted(roots, key=lambda x: x[0])


# ============================================================
# GEOMETRIC ACTION (vectorized)
# ============================================================
def geometric_action(path_ET, s_val):
    """
    V = ∫ (|F| - F·τ̂) ds along discrete path.
    ΔΦ = V / 2.
    """
    dpath = np.diff(path_ET, axis=0)
    ds = np.sqrt(dpath[:, 0]**2 + dpath[:, 1]**2)
    mask = ds > 1e-30
    if not np.any(mask):
        return 1e30

    mids = 0.5 * (path_ET[:-1] + path_ET[1:])
    Em, Tm = mids[mask, 0], mids[mask, 1]
    dE, dT = dpath[mask, 0], dpath[mask, 1]
    ds_m = ds[mask]

    denom = g_par + Tm
    f1 = s_val + p_par * Em * Tm / denom - m_par * Em * Tm - d_par * Em
    f2 = a_par * Tm * (1 - b_par * Tm) - n_par * Em * Tm
    F_mag = np.sqrt(f1**2 + f2**2)
    F_dot_tau = (f1 * dE + f2 * dT) / ds_m

    return float(np.sum((F_mag - F_dot_tau) * ds_m))


# ============================================================
# MAP via differential_evolution (global optimizer)
# ============================================================
def find_MAP(s_val, E_d, T_d, E_s, T_s, K=6, N_path=150):
    """
    Find minimum action path using global optimization.
    Path parameterized as straight line + Fourier perturbation.
    """
    E_range = abs(E_d - E_s)
    T_range = abs(T_d - T_s)
    alpha = np.linspace(0, 1, N_path + 1)
    sin_basis = np.array([np.sin(k * np.pi * alpha) for k in range(1, K + 1)])

    def path_from_coeffs(coeffs):
        a_c = coeffs[:K]
        b_c = coeffs[K:]
        E_path = E_d + (E_s - E_d) * alpha + E_range * (sin_basis.T @ a_c)
        T_path = T_d + (T_s - T_d) * alpha + T_range * (sin_basis.T @ b_c)
        return np.column_stack([E_path, T_path])

    def objective(coeffs):
        path = path_from_coeffs(coeffs)
        if np.any(path[:, 0] < 0) or np.any(path[:, 1] < 0):
            return 1e30
        return geometric_action(path, s_val)

    # Bounds: coefficients as fraction of range
    bounds = [(-0.5, 0.5)] * K + [(-0.5, 0.5)] * K

    result = differential_evolution(
        objective, bounds, maxiter=500, tol=1e-8,
        seed=42, polish=True, init='sobol',
        mutation=(0.5, 1.5), recombination=0.9,
        popsize=20,
    )

    V_min = result.fun
    DPhi_2D = V_min / 2.0
    path_opt = path_from_coeffs(result.x)
    return DPhi_2D, V_min, path_opt, result.success


# ============================================================
# 1D barrier for comparison
# ============================================================
def barrier_1D(s_val, E_d, E_s):
    result, _ = quad(lambda E: f_eff(E, s_val), E_s, E_d, limit=500)
    return result


def f_eff_deriv(E, s_val):
    dE = max(abs(E) * 1e-7, 1.0)
    return (f_eff(E + dE, s_val) - f_eff(E - dE, s_val)) / (2 * dE)


# ============================================================
# 1D exact MFPT (for sigma* finding)
# ============================================================
def compute_D_1D(sigma, s_val, E_d, E_s, lam):
    tau = 1.0 / abs(lam)
    N = 60000
    margin = 3.0 * sigma / np.sqrt(2.0 * abs(lam))
    E_lo = max(1.0, E_s - margin)
    E_hi = E_d + margin
    E_grid = np.linspace(E_lo, E_hi, N)
    dE = E_grid[1] - E_grid[0]
    T_arr = np.maximum(0.0, (a_par - n_par * E_grid) / (a_par * b_par))
    f_arr = (s_val + p_par * E_grid * T_arr / (g_par + T_arr)
             - m_par * E_grid * T_arr - d_par * E_grid)
    V_raw = np.cumsum(-f_arr) * dE
    i_d = np.argmin(np.abs(E_grid - E_d))
    V_grid = V_raw - V_raw[i_d]
    Phi = 2.0 * V_grid / sigma**2
    if np.max(Phi) > 600:
        return np.inf
    Phi = np.clip(Phi, -500, 500)
    J = np.cumsum(np.exp(-Phi)[::-1])[::-1] * dE
    psi = (2.0 / sigma**2) * np.exp(Phi) * J
    i_s = np.argmin(np.abs(E_grid - E_s))
    lo, hi = min(i_s, i_d), max(i_s, i_d)
    return np.trapz(psi[lo:hi + 1], E_grid[lo:hi + 1]) / tau


def find_sigma_star(s_val, E_d, E_s, lam, D_target):
    def obj(log_s):
        D = compute_D_1D(np.exp(log_s), s_val, E_d, E_s, lam)
        if D == np.inf or D <= 0:
            return 1.0
        return np.log(max(D, 1e-30)) - np.log(D_target)
    try:
        return np.exp(brentq(obj, np.log(10), np.log(1e8), xtol=1e-10, maxiter=300))
    except ValueError:
        return np.nan


# ============================================================
# VECTORIZED 2D SDE SIMULATION
# ============================================================
def sde_escape_2D(s_val, E_d, T_d, E_s, T_s, sigma,
                  n_trials=500, dt=0.05, max_days=30000):
    """
    Vectorized 2D Euler-Maruyama. All trials run in parallel via numpy.
    Escape criterion: E < E_saddle.
    """
    max_steps = int(max_days / dt)
    sqrt_dt = np.sqrt(dt)

    E = np.full(n_trials, E_d)
    T = np.full(n_trials, T_d)
    escape_time = np.full(n_trials, np.nan)
    active = np.ones(n_trials, dtype=bool)

    for step in range(max_steps):
        n_act = np.sum(active)
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
        escape_time[escaped] = (step + 1) * dt
        active[escaped] = False

    # Trials that didn't escape: assign max_days
    escape_time[np.isnan(escape_time)] = max_days
    return np.mean(escape_time), np.std(escape_time) / np.sqrt(n_trials)


# ============================================================
# MAIN
# ============================================================
prt("=" * 78)
prt("2D QUASI-POTENTIAL: KUZNETSOV TUMOR-IMMUNE (IMPROVED)")
prt("=" * 78)

# ============================================================
# STEP 1: Equilibria and 2D eigenvalues
# ============================================================
eqs = find_equilibria(S_FIT)
assert len(eqs) >= 3, f"Need 3 equilibria, got {len(eqs)}"
E_unc, T_unc = eqs[0]
E_sad, T_sad = eqs[1]
E_dorm, T_dorm = eqs[2]

prt(f"\nEquilibria at s = {S_FIT:.0f}:")
prt(f"  Dormant:      E = {E_dorm:12.0f}, T = {T_dorm:12.0f}")
prt(f"  Saddle:       E = {E_sad:12.0f}, T = {T_sad:12.0f}")
prt(f"  Uncontrolled: E = {E_unc:12.0f}, T = {T_unc:12.0f}")

# 2D eigenvalue for tau
from numpy.linalg import eigvals
denom_d = g_par + T_dorm
J = np.array([
    [p_par * T_dorm / denom_d - m_par * T_dorm - d_par,
     p_par * E_dorm * g_par / denom_d**2 - m_par * E_dorm],
    [-n_par * T_dorm,
     a_par * (1 - 2 * b_par * T_dorm) - n_par * E_dorm]
])
eigs = eigvals(J)
lam_2d = np.max(np.real(eigs))
tau_2d = 1.0 / abs(lam_2d)
lam_1d = f_eff_deriv(E_dorm, S_FIT)

prt(f"\n  2D eigenvalue: {lam_2d:.6f} /day (tau = {tau_2d:.1f} days)")
prt(f"  1D eigenvalue: {lam_1d:.6f} /day (ratio: {abs(lam_1d)/abs(lam_2d):.0f}x)")

# ============================================================
# STEP 2: MAP at operating point (global optimizer)
# ============================================================
prt(f"\n{'=' * 78}")
prt("STEP 2: MINIMUM ACTION PATH (GLOBAL OPTIMIZATION)")
prt(f"{'=' * 78}")

dphi_1d_op = barrier_1D(S_FIT, E_dorm, E_sad)
prt(f"\n  1D barrier: ΔΦ_1D = {dphi_1d_op:.4e}")

prt("  Computing MAP with differential_evolution (K=6, popsize=20)...")
dphi_2d_op, V_op, path_op, converged = find_MAP(
    S_FIT, E_dorm, T_dorm, E_sad, T_sad, K=6, N_path=150)
prt(f"  Converged: {converged}")
prt(f"  2D barrier: ΔΦ_2D = {dphi_2d_op:.4e}")
prt(f"  Ratio ΔΦ_2D / ΔΦ_1D = {dphi_2d_op / dphi_1d_op:.4f}")

# Also try K=8 for convergence check
prt("  Convergence check with K=8...")
dphi_2d_k8, _, _, _ = find_MAP(
    S_FIT, E_dorm, T_dorm, E_sad, T_sad, K=8, N_path=150)
prt(f"  K=6: ΔΦ = {dphi_2d_op:.4e}")
prt(f"  K=8: ΔΦ = {dphi_2d_k8:.4e}")
prt(f"  Difference: {abs(dphi_2d_k8 - dphi_2d_op) / dphi_2d_op * 100:.1f}%")
dphi_2d_best = min(dphi_2d_op, dphi_2d_k8)
prt(f"  Best ΔΦ_2D = {dphi_2d_best:.4e}")

# ============================================================
# STEP 3: Vectorized SDE simulation
# ============================================================
prt(f"\n{'=' * 78}")
prt("STEP 3: 2D SDE SIMULATION (KRAMERS FIT)")
prt(f"{'=' * 78}")

# Choose sigma values where MFPT should be tractable
# At the operating point, ΔΦ ~ 1e10.
# MFPT ~ exp(2ΔΦ/σ²) * prefactor. For MFPT ~ 100 days: σ ~ 80k-200k
sigma_vals = [50000, 70000, 90000, 120000, 160000, 220000, 300000]
n_trials = 500

prt(f"\n  {n_trials} trials per sigma, dt=0.05, max=30000 days")
prt(f"\n  {'sigma':>10s} {'MFPT':>12s} {'SE':>10s} {'n_esc':>8s} {'ln(MFPT)':>10s}")
prt(f"  {'-' * 55}")

sde_data = []
for sv in sigma_vals:
    np.random.seed(12345)
    mfpt, se = sde_escape_2D(
        S_FIT, E_dorm, T_dorm, E_sad, T_sad,
        sv, n_trials=n_trials, dt=0.05, max_days=30000)
    # Count how many actually escaped
    n_esc = n_trials  # approximate (all get max_days if not escaped)
    sde_data.append({'sigma': sv, 'mfpt': mfpt, 'se': se})
    prt(f"  {sv:10.0f} {mfpt:12.1f} {se:10.1f} {'—':>8s} {np.log(mfpt):10.3f}")

# Kramers fit: ln(MFPT) = 2ΔΦ/σ² + const
inv_s2 = np.array([1.0 / d['sigma']**2 for d in sde_data])
ln_mfpt = np.array([np.log(d['mfpt']) for d in sde_data])

# Only use points where MFPT < max_days
valid = np.array([d['mfpt'] < 29000 for d in sde_data])
if np.sum(valid) >= 3:
    coeffs = np.polyfit(inv_s2[valid], ln_mfpt[valid], 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    dphi_sde = slope / 2.0

    ln_pred = np.polyval(coeffs, inv_s2[valid])
    ss_res = np.sum((ln_mfpt[valid] - ln_pred)**2)
    ss_tot = np.sum((ln_mfpt[valid] - np.mean(ln_mfpt[valid]))**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    prt(f"\n  Kramers fit: ln(MFPT) = {slope:.4e} / σ² + {intercept:.3f}")
    prt(f"  R² = {R2:.4f}")
    prt(f"  ΔΦ_SDE = {dphi_sde:.4e}")
else:
    dphi_sde = np.nan
    prt("\n  WARNING: insufficient valid SDE data for fit")

# ============================================================
# STEP 4: Barrier comparison
# ============================================================
prt(f"\n{'=' * 78}")
prt("STEP 4: BARRIER COMPARISON (THREE METHODS)")
prt(f"{'=' * 78}")

prt(f"\n  {'Method':>25s} {'ΔΦ':>14s} {'Ratio to 1D':>14s}")
prt(f"  {'-' * 55}")
prt(f"  {'1D adiabatic':>25s} {dphi_1d_op:14.4e} {'1.000':>14s}")
prt(f"  {'2D MAP (global opt)':>25s} {dphi_2d_best:14.4e} "
    f"{dphi_2d_best / dphi_1d_op:14.4f}")
if np.isfinite(dphi_sde):
    prt(f"  {'2D SDE (Kramers fit)':>25s} {dphi_sde:14.4e} "
        f"{dphi_sde / dphi_1d_op:14.4f}")

# ============================================================
# STEP 5: B at operating point
# ============================================================
prt(f"\n{'=' * 78}")
prt("STEP 5: B AT BCL1 OPERATING POINT")
prt(f"{'=' * 78}")

D_BCL1 = 730.0 * abs(lam_1d)
sig_star = find_sigma_star(S_FIT, E_dorm, E_sad, lam_1d, D_BCL1)
B_1d = 2 * dphi_1d_op / sig_star**2
B_2d = 2 * dphi_2d_best / sig_star**2
if np.isfinite(dphi_sde):
    B_sde = 2 * dphi_sde / sig_star**2

prt(f"\n  D_target = {D_BCL1:.1f} (BCL1 730-day dormancy)")
prt(f"  sigma* = {sig_star:.0f}")
prt(f"\n  B_1D  = {B_1d:.4f}")
prt(f"  B_2D  = {B_2d:.4f}")
if np.isfinite(dphi_sde):
    prt(f"  B_SDE = {B_sde:.4f}")

# ============================================================
# STEP 6: B INVARIANCE SCAN WITH 2D BARRIER
# ============================================================
prt(f"\n{'=' * 78}")
prt("STEP 6: B INVARIANCE SCAN (2D MAP ACROSS BISTABLE RANGE)")
prt(f"{'=' * 78}")

# Find bistable range
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
s_scan = np.linspace(s_lo + margin, s_hi - margin, 12)
D_T = round(D_BCL1, 1)

prt(f"\n  Bistable range: [{s_lo:.0f}, {s_hi:.0f}]")
prt(f"  D_target = {D_T}")
prt(f"  Scanning {len(s_scan)} points...")

prt(f"\n  {'s':>8s} {'ΔΦ_1D':>12s} {'ΔΦ_2D':>12s} {'2D/1D':>8s} "
    f"{'σ*':>10s} {'B_1D':>8s} {'B_2D':>8s}")
prt(f"  {'-' * 72}")

results = []
for s_val in s_scan:
    eqs_s = find_equilibria(s_val)
    if len(eqs_s) < 3:
        continue
    Ed, Td = eqs_s[2]
    Es, Ts = eqs_s[1]

    lam = f_eff_deriv(Ed, s_val)
    if lam >= 0:
        continue

    dphi1 = barrier_1D(s_val, Ed, Es)
    if dphi1 <= 0:
        continue

    dphi2, _, _, conv = find_MAP(s_val, Ed, Td, Es, Ts, K=6, N_path=120)
    if dphi2 <= 0 or not np.isfinite(dphi2):
        continue

    ss = find_sigma_star(s_val, Ed, Es, lam, D_T)
    if np.isnan(ss):
        continue

    b1 = 2 * dphi1 / ss**2
    b2 = 2 * dphi2 / ss**2
    r = dphi2 / dphi1

    results.append({
        's': s_val, 'dphi1': dphi1, 'dphi2': dphi2,
        'ratio': r, 'sigma': ss, 'B1': b1, 'B2': b2
    })

    prt(f"  {s_val:8.0f} {dphi1:12.4e} {dphi2:12.4e} {r:8.4f} "
        f"{ss:10.0f} {b1:8.4f} {b2:8.4f}")

# ============================================================
# SUMMARY
# ============================================================
prt(f"\n{'=' * 78}")
prt("SUMMARY")
prt(f"{'=' * 78}")

if len(results) >= 3:
    B1 = np.array([r['B1'] for r in results])
    B2 = np.array([r['B2'] for r in results])
    R = np.array([r['ratio'] for r in results])
    DP2 = np.array([r['dphi2'] for r in results])

    prt(f"\n  ΔΦ_2D / ΔΦ_1D ratio across range:")
    prt(f"    Mean: {np.mean(R):.4f}, Std: {np.std(R):.4f}")
    prt(f"    Range: [{np.min(R):.4f}, {np.max(R):.4f}]")
    prt(f"    1D overestimates barrier by ~{(1 - np.mean(R)) * 100:.0f}%")

    prt(f"\n  B (1D adiabatic): {np.mean(B1):.2f} ± {np.std(B1)/np.mean(B1)*100:.1f}%")
    prt(f"  B (2D MAP):       {np.mean(B2):.2f} ± {np.std(B2)/np.mean(B2)*100:.1f}%")
    prt(f"  B_2D range: [{np.min(B2):.2f}, {np.max(B2):.2f}]")
    prt(f"  ΔΦ_2D variation: {np.max(DP2)/np.min(DP2):.2f}x")

    hz = 1.8 <= np.mean(B2) <= 6.0
    prt(f"  Habitable zone [1.8, 6.0]: {'INSIDE' if hz else 'OUTSIDE'}")

    prt(f"\n  {'System':>15s} {'B':>8s} {'CV':>8s} {'Domain':>18s}")
    prt(f"  {'-' * 53}")
    prt(f"  {'Kelp':>15s} {'2.17':>8s} {'2.6%':>8s} {'Ecology':>18s}")
    prt(f"  {'Savanna':>15s} {'4.04':>8s} {'4.6%':>8s} {'Ecology':>18s}")
    prt(f"  {'Lake':>15s} {'4.27':>8s} {'2.0%':>8s} {'Ecology':>18s}")
    prt(f"  {'Toggle':>15s} {'4.83':>8s} {'3.8%':>8s} {'Gene circuit':>18s}")
    prt(f"  {'Coral':>15s} {'6.06':>8s} {'2.1%':>8s} {'Ecology':>18s}")
    B2m = np.mean(B2)
    B2cv = np.std(B2) / B2m * 100
    prt(f"  {'Tumor (2D)':>15s} {B2m:8.2f} {B2cv:7.1f}% {'Cancer biology':>18s}")

    if np.isfinite(dphi_sde):
        prt(f"\n  SDE cross-check at operating point:")
        prt(f"    ΔΦ_SDE = {dphi_sde:.4e} (R² = {R2:.3f})")
        prt(f"    ΔΦ_2D  = {dphi_2d_best:.4e} (MAP)")
        if dphi_sde > 0:
            prt(f"    SDE/MAP = {dphi_sde / dphi_2d_best:.2f}")
else:
    prt(f"\n  ERROR: only {len(results)} valid points")

prt(f"\nDone.")
