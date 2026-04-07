#!/usr/bin/env python3
"""
Toggle switch recomputation with CORRECTED Kramers-Langer prefactor.

Round 8 used: C = lambda_u/(2pi) * |det J_min| / |lambda_s|          [WRONG]
Correct:      C = lambda_u/(2pi) * sqrt(|det J_min| / (lambda_u * |lambda_s|))

Reuses all model functions from prefactor_test.py (toggle_jac, toggle_fps, etc.)
Only changes: the prefactor formula.

Key test: does D_obs ≈ corrected_prefactor * exp(DeltaV) for the toggle switch?
If yes, the "endogenous equation" is just Kramers with large barrier.
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigs
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Toggle switch model (copied from prefactor_test.py, unchanged)
# ============================================================

def toggle_jac(u, v, alpha, n, delta):
    return np.array([
        [-1, -alpha * n * v**(n-1) / (1 + v**n)**2],
        [-alpha * n * u**(n-1) / (1 + u**n)**2, -(1 + delta)]
    ])

def toggle_fps(alpha, n, delta):
    def system(uv):
        u, v = uv
        return [alpha / (1 + v**n) - u,
                alpha / (1 + u**n) - (1 + delta) * v]
    found = []
    for u0 in np.linspace(0.1, alpha + 1, 25):
        for v0 in np.linspace(0.1, alpha / (1 + delta) + 1, 20):
            try:
                sol = fsolve(system, [u0, v0], full_output=True)
                uv = sol[0]
                if sol[2] == 1 and uv[0] > 0 and uv[1] > 0:
                    if max(abs(system(uv)[0]), abs(system(uv)[1])) < 1e-8:
                        found.append((round(uv[0], 6), round(uv[1], 6)))
            except Exception:
                pass
    unique = []
    for c in found:
        if not any(abs(c[0]-u[0]) < 1e-4 and abs(c[1]-u[1]) < 1e-4 for u in unique):
            unique.append(c)
    results = []
    for u, v in unique:
        J = toggle_jac(u, v, alpha, n, delta)
        eigvals = np.real(np.linalg.eigvals(J))
        if np.all(eigvals < -1e-8):
            results.append((u, v, 'stable'))
        elif np.max(eigvals) > 1e-8 and np.min(eigvals) < -1e-8:
            results.append((u, v, 'saddle'))
    return results

def get_toggle_bistable(alpha, n, delta):
    fps = toggle_fps(alpha, n, delta)
    stables = sorted([(u, v) for u, v, t in fps if t == 'stable'], key=lambda p: -p[0])
    saddles = [(u, v) for u, v, t in fps if t == 'saddle']
    if len(stables) < 2 or len(saddles) < 1:
        return None
    if abs(stables[0][0] - stables[1][0]) < 0.3:
        return None
    return stables[0], stables[1], saddles[0]

def build_toggle_Q(alpha, n, Omega, n_max, delta):
    N = n_max * n_max
    Q = lil_matrix((N, N), dtype=float)
    for nu in range(n_max):
        for nv in range(n_max):
            i = nu * n_max + nv
            cu, cv = nu/Omega, nv/Omega
            a1 = Omega * alpha / (1 + cv**n) if cv > 0 else Omega * alpha
            a2 = float(nu)
            a3 = Omega * alpha / (1 + cu**n) if cu > 0 else Omega * alpha
            a4 = float(nv) + delta * float(nv)
            if nu+1 < n_max:
                j = (nu+1)*n_max+nv; Q[j,i] += a1; Q[i,i] -= a1
            if nu > 0:
                j = (nu-1)*n_max+nv; Q[j,i] += a2; Q[i,i] -= a2
            if nv+1 < n_max:
                j = nu*n_max+(nv+1); Q[j,i] += a3; Q[i,i] -= a3
            if nv > 0:
                j = nu*n_max+(nv-1); Q[j,i] += a4; Q[i,i] -= a4
    return Q.tocsc()

def toggle_rates(Q, n_max, Omega, fp_hi_u, fp_hi_v):
    eigenvalues, eigenvectors = eigs(Q, k=6, sigma=0, which='LM')
    idx = np.argsort(np.abs(np.real(eigenvalues)))
    eigenvalues = eigenvalues[idx]
    p_ss = np.real(eigenvectors[:, idx[0]])
    p_ss = p_ss / np.sum(p_ss)
    if np.min(p_ss) < 0:
        p_ss = -p_ss; p_ss /= np.sum(p_ss)
    P_ss = p_ss.reshape(n_max, n_max)
    k_sw = np.abs(np.real(eigenvalues[1]))
    u_mid = (fp_hi_u[0] + fp_hi_v[0]) / 2.0
    nu_mid = int(round(u_mid * Omega))
    nu_mid = np.clip(nu_mid, 1, n_max - 2)
    P_u = np.sum(P_ss, axis=1)
    pi_A = np.sum(P_u[nu_mid:])
    pi_B = np.sum(P_u[:nu_mid])
    k_AB = pi_B * k_sw
    return k_AB, k_sw, pi_A, pi_B, P_ss

def toggle_barrier(P_ss, n_max, Omega, fp_hi_u, fp_saddle):
    P_u = np.sum(P_ss, axis=1)
    P_u = np.maximum(P_u, 1e-300)
    V = -np.log(P_u)
    nu_min = np.clip(int(round(fp_hi_u[0] * Omega)), 0, n_max - 1)
    nu_sad = np.clip(int(round(fp_saddle[0] * Omega)), 0, n_max - 1)
    w = 2
    V_min = np.min(V[max(0,nu_min-w):min(n_max,nu_min+w+1)])
    V_sad = np.max(V[max(0,nu_sad-w):min(n_max,nu_sad+w+1)])
    return V_sad - V_min


# ============================================================
# CORRECTED prefactor (the only change from Round 8)
# ============================================================

def kramers_prefactor_corrected(alpha, n, delta, hi_u, saddle):
    """
    CORRECT Kramers-Langer prefactor for 2D overdamped systems:
    C = lambda_u/(2*pi) * sqrt(|det J_min| / (lambda_u * |lambda_s|))

    Returns C_correct, C_old (for comparison), tau, eigenvalues.
    """
    J_min = toggle_jac(*hi_u, alpha, n, delta)
    J_sad = toggle_jac(*saddle, alpha, n, delta)
    eig_min = np.real(np.linalg.eigvals(J_min))
    eig_sad = np.real(np.linalg.eigvals(J_sad))

    lambda_min_prod = np.prod(np.abs(eig_min))
    lambda_u = np.max(eig_sad)
    lambda_s = np.abs(np.min(eig_sad))

    # OLD (Round 8 bug): squares the determinant ratio
    C_old = lambda_u / (2 * np.pi) * lambda_min_prod / lambda_s

    # CORRECT: square root of determinant ratio
    C_new = lambda_u / (2 * np.pi) * np.sqrt(lambda_min_prod / (lambda_u * lambda_s))

    tau = 1.0 / np.min(np.abs(eig_min))

    return C_new, C_old, tau, eig_min, eig_sad


# ============================================================
# RUN: Recompute all 55 toggle switch configurations
# ============================================================

print("=" * 100)
print("TOGGLE SWITCH: CORRECTED KRAMERS-LANGER PREFACTOR")
print("=" * 100)

n_hill = 2
configs = [
    (5.0, [2, 3, 5], [0, 0.05, 0.1, 0.15, 0.2, 0.3]),
    (6.0, [2, 3, 5], [0, 0.05, 0.1, 0.15, 0.2, 0.3]),
    (8.0, [2, 3, 5], [0, 0.05, 0.1, 0.15, 0.2, 0.3]),
]

results = []

for alpha, Omega_list, delta_list in configs:
    for Omega in Omega_list:
        bp0 = get_toggle_bistable(alpha, n_hill, 0)
        if bp0 is None:
            continue
        hi_u_0, hi_v_0, saddle_0 = bp0

        n_max = int(2.5 * alpha * Omega) + 10
        n_max = min(n_max, 250)
        if n_max**2 > 80000:
            continue

        for delta in delta_list:
            bp = get_toggle_bistable(alpha, n_hill, delta)
            if bp is None:
                break
            hi_u, hi_v, saddle = bp

            C_new, C_old, tau_r, eig_min, eig_sad = kramers_prefactor_corrected(
                alpha, n_hill, delta, hi_u, saddle)

            Q = build_toggle_Q(alpha, n_hill, Omega, n_max, delta)
            try:
                k_AB, _, _, _, P_ss = toggle_rates(Q, n_max, Omega, hi_u, hi_v)
            except Exception:
                continue

            mfpt = 1.0 / k_AB if k_AB > 1e-15 else np.inf
            D_obs = mfpt / tau_r

            try:
                DeltaV = toggle_barrier(P_ss, n_max, Omega, hi_u, saddle)
            except Exception:
                DeltaV = np.nan

            exp_DV = np.exp(DeltaV) if not np.isnan(DeltaV) else np.nan

            inv_Ctau_old = 1.0 / (C_old * tau_r)
            inv_Ctau_new = 1.0 / (C_new * tau_r)
            D_kramers_old = inv_Ctau_old * exp_DV if not np.isnan(exp_DV) else np.nan
            D_kramers_new = inv_Ctau_new * exp_DV if not np.isnan(exp_DV) else np.nan

            # The key ratio: D_obs / exp(DeltaV) = empirical prefactor
            pf_empirical = D_obs / exp_DV if not np.isnan(exp_DV) and exp_DV > 0 else np.nan

            results.append(dict(
                alpha=alpha, Omega=Omega, delta=delta,
                D_obs=D_obs, DeltaV=DeltaV, exp_DV=exp_DV,
                inv_Ctau_old=inv_Ctau_old, inv_Ctau_new=inv_Ctau_new,
                D_kramers_old=D_kramers_old, D_kramers_new=D_kramers_new,
                pf_empirical=pf_empirical,
                bug_ratio=C_old/C_new,
            ))

# ============================================================
# OUTPUT: Table at delta=0 (baseline, no exogenous channel)
# ============================================================

print(f"\nBaseline (delta=0): Does D_obs = corrected_prefactor * exp(DeltaV)?")
print(f"\n{'a':>3} {'Om':>3} {'D_obs':>10} {'DV':>7} {'exp(DV)':>10} "
      f"{'1/Ct_old':>9} {'1/Ct_new':>9} {'pf_emp':>8} {'D_kr_old':>10} {'D_kr_new':>10} "
      f"{'obs/old':>8} {'obs/new':>8} {'bug_x':>6}")
print("-" * 120)

for r in results:
    if r['delta'] != 0:
        continue
    D = r['D_obs']
    def f(x, w=10):
        if np.isnan(x) or np.isinf(x):
            return f"{'N/A':>{w}}"
        return f"{x:>{w}.1f}" if x < 1e6 else f"{x:>{w}.1e}"

    ratio_old = D / r['D_kramers_old'] if r['D_kramers_old'] and r['D_kramers_old'] > 0 else np.nan
    ratio_new = D / r['D_kramers_new'] if r['D_kramers_new'] and r['D_kramers_new'] > 0 else np.nan

    print(f"{r['alpha']:>3.0f} {r['Omega']:>3} {f(D)} {r['DeltaV']:>7.2f} {f(r['exp_DV'])}"
          f" {f(r['inv_Ctau_old'],9)} {f(r['inv_Ctau_new'],9)} {f(r['pf_empirical'],8)}"
          f" {f(r['D_kramers_old'])} {f(r['D_kramers_new'])}"
          f" {ratio_old:>8.3f} {ratio_new:>8.3f} {r['bug_ratio']:>6.1f}x")

# ============================================================
# ANALYSIS: Prefactor constancy within families
# ============================================================

print(f"\n{'='*80}")
print("ANALYSIS: Is the empirical prefactor constant within (alpha, Omega) families?")
print(f"{'='*80}")

print(f"\n{'a':>3} {'Om':>3} {'pf_emp':>10} {'1/Ct_new':>10} {'pf/Ct':>8} {'D_obs/D_new':>12}")
print("-" * 55)

from collections import defaultdict
families = defaultdict(list)
for r in results:
    if r['delta'] == 0:
        families[(r['alpha'], r['Omega'])].append(r)

# Show delta=0 baseline for each family
for (alpha, Omega), fam in sorted(families.items()):
    r = fam[0]
    ratio = r['D_obs'] / r['D_kramers_new'] if r['D_kramers_new'] > 0 else np.nan
    pf_ratio = r['pf_empirical'] / r['inv_Ctau_new'] if r['inv_Ctau_new'] > 0 else np.nan
    print(f"{alpha:>3.0f} {Omega:>3} {r['pf_empirical']:>10.2f} {r['inv_Ctau_new']:>10.2f}"
          f" {pf_ratio:>8.2f} {ratio:>12.3f}")

# Check across all deltas within each (alpha, Omega) family
print(f"\n{'='*80}")
print("Within-family consistency (all deltas)")
print(f"{'='*80}")

print(f"\n{'a':>3} {'Om':>3} {'mean_obs/new':>13} {'CV':>8} {'n':>4}")
print("-" * 40)

for (alpha, Omega) in sorted(families.keys()):
    fam_all = [r for r in results if r['alpha'] == alpha and r['Omega'] == Omega]
    ratios = [r['D_obs']/r['D_kramers_new'] for r in fam_all
              if r['D_kramers_new'] and r['D_kramers_new'] > 0 and not np.isnan(r['D_kramers_new'])]
    if ratios:
        m = np.mean(ratios)
        cv = np.std(ratios, ddof=1) / m if m > 0 else np.nan
        print(f"{alpha:>3.0f} {Omega:>3} {m:>13.3f} {cv:>8.3f} {len(ratios):>4}")

# ============================================================
# SUMMARY
# ============================================================

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

# Overall D_obs/D_kramers_new statistics
all_ratios_new = [r['D_obs']/r['D_kramers_new'] for r in results
                  if r['D_kramers_new'] and r['D_kramers_new'] > 0 and not np.isnan(r['D_kramers_new'])]
all_ratios_old = [r['D_obs']/r['D_kramers_old'] for r in results
                  if r['D_kramers_old'] and r['D_kramers_old'] > 0 and not np.isnan(r['D_kramers_old'])]

print(f"\nD_obs / D_Kramers (all {len(all_ratios_new)} configs):")
print(f"  OLD formula: mean = {np.mean(all_ratios_old):.3f}, std = {np.std(all_ratios_old):.3f}")
print(f"  NEW formula: mean = {np.mean(all_ratios_new):.3f}, std = {np.std(all_ratios_new):.3f}")

# By alpha
for alpha in [5, 6, 8]:
    ratios = [r['D_obs']/r['D_kramers_new'] for r in results
              if r['alpha'] == alpha and r['D_kramers_new'] > 0 and not np.isnan(r['D_kramers_new'])]
    if ratios:
        print(f"  alpha={alpha}: mean = {np.mean(ratios):.3f} +/- {np.std(ratios):.3f} (n={len(ratios)})")

# The key number: at alpha=8 (largest, most in-Kramers-regime)
ratios_8 = [r['D_obs']/r['D_kramers_new'] for r in results
            if r['alpha'] == 8 and r['D_kramers_new'] > 0 and not np.isnan(r['D_kramers_new'])]
if ratios_8:
    print(f"\n  --> At alpha=8: D_obs/D_Kramers_corrected = {np.mean(ratios_8):.3f} +/- {np.std(ratios_8):.3f}")
    print(f"      This is the anharmonicity/finite-size correction for the toggle switch.")
    print(f"      Compare: savanna = 0.52 (1.9x), lake = 0.56 (1.8x)")
