#!/usr/bin/env python3
"""
Thermohaline Circulation — Kramers/MFPT Test

Cessi (1994) reduced 1D model:
    dy/dt = p - y*(1 + mu^2*(1-y)^2)

Potential:
    V(y) = -p*y + y^2/2 + mu^2*(y^2/2 - 2*y^3/3 + y^4/4)

This script follows the exact methodology of step6_kelp_kramers.py,
step8_synthetic_3channel.py, and step10_tropical_forest_kramers.py:
  1. Define the 1D drift
  2. Find fixed points (two stable + one saddle)
  3. Compute potential V(y) via integration
  4. Compute exact MFPT via Fokker-Planck integral
  5. Compute D_exact = MFPT / tau_relax
  6. Compute D_Kramers approximation (with K, C, tau)
  7. Scan sigma — find sigma* for various D_target values
  8. Compare D_Kramers to D_exact (accuracy of Kramers approximation)
  9. Report K_actual = D_exact / (exp(2*DV/sigma^2) / (C*tau))

The THC has no product equation (no separable channels after reduction),
so we test Kramers directly, and compare to D_observed from D-O events.
"""

import numpy as np
from scipy.optimize import brentq

# ============================================================
# MODEL DEFINITION: Cessi (1994) reduced 1D
# ============================================================

def f_cessi(y, p, mu2):
    """1D drift: dy/dt = p - y*(1 + mu^2*(1-y)^2)"""
    return p - y * (1.0 + mu2 * (1.0 - y)**2)

def fp_cessi(y, p, mu2):
    """df/dy = derivative of drift"""
    return -(1.0 + mu2 * (1.0 - y)**2) + 2.0 * mu2 * y * (1.0 - y)

def V_cessi(y, p, mu2):
    """Analytical potential: V(y) = -integral(f(y))dy"""
    return -p * y + y**2 / 2.0 + mu2 * (y**2 / 2.0 - 2.0 * y**3 / 3.0 + y**4 / 4.0)

# ============================================================
# FIND FIXED POINTS
# ============================================================

def find_equilibria(p, mu2, y_lo=0.001, y_hi=1.5, N=500000):
    """Find all roots of f(y) = 0 in [y_lo, y_hi] via sign-change + Brent."""
    y_scan = np.linspace(y_lo, y_hi, N)
    f_scan = np.array([f_cessi(y, p, mu2) for y in y_scan])
    sign_changes = np.where(np.diff(np.sign(f_scan)))[0]
    roots = []
    for i in sign_changes:
        try:
            root = brentq(lambda y: f_cessi(y, p, mu2), y_scan[i], y_scan[i + 1], xtol=1e-14)
            roots.append(root)
        except ValueError:
            pass
    return sorted(roots)

def classify_equilibria(roots, p, mu2):
    """Classify roots as stable (fp < 0) or unstable (fp > 0)."""
    result = []
    for r in roots:
        lam = fp_cessi(r, p, mu2)
        result.append((r, lam, 'stable' if lam < 0 else 'unstable'))
    return result

# ============================================================
# EXACT MFPT (Fokker-Planck integral)
# ============================================================

def compute_D_exact(p, mu2, y_eq, y_saddle, sigma, N=200000):
    """
    Compute exact D = MFPT / tau_relax via the 1D MFPT integral.

    MFPT = integral_{y_eq}^{y_saddle} psi(y) dy
    where psi(y) = (2/sigma^2) * exp(Phi(y)) * integral_{y_min}^{y} exp(-Phi(s)) ds
    and Phi(y) = 2*V(y)/sigma^2

    Integration domain: from a reflecting boundary below y_eq to absorbing at y_saddle.
    """
    # Integration grid from below equilibrium to saddle
    y_min = max(0.001, y_eq - 3.0 * (y_saddle - y_eq))
    y_max = y_saddle

    yg = np.linspace(y_min, y_max, N)
    dy = yg[1] - yg[0]

    # Potential on grid (analytical)
    Vg = np.array([V_cessi(y, p, mu2) for y in yg])

    # Reference potential at equilibrium
    i_eq = np.argmin(np.abs(yg - y_eq))
    Vg = Vg - Vg[i_eq]

    # Phi = 2*V/sigma^2
    Phi = 2.0 * Vg / sigma**2

    # Stabilize: shift so Phi[i_eq] = 0
    Phi = Phi - Phi[i_eq]

    # Check for overflow
    max_Phi = np.max(Phi[i_eq:])
    if max_Phi > 700:
        return np.inf  # Barrier too large for this sigma

    exp_neg_Phi = np.exp(-Phi)
    Ix = np.cumsum(exp_neg_Phi) * dy  # integral from y_min to y

    exp_pos_Phi = np.exp(Phi)
    psi = (2.0 / sigma**2) * exp_pos_Phi * Ix

    # MFPT = integral from y_eq to y_saddle of psi
    i_sad = np.argmin(np.abs(yg - y_saddle))
    MFPT = np.trapz(psi[i_eq:i_sad + 1], yg[i_eq:i_sad + 1])

    # tau_relax
    lam_eq = abs(fp_cessi(y_eq, p, mu2))
    tau = 1.0 / lam_eq

    return MFPT / tau

# ============================================================
# KRAMERS APPROXIMATION
# ============================================================

def kramers_quantities(p, mu2, y_eq, y_saddle):
    """Compute Kramers prefactor quantities."""
    lam_eq = abs(fp_cessi(y_eq, p, mu2))
    lam_sad = abs(fp_cessi(y_saddle, p, mu2))

    V_eq = V_cessi(y_eq, p, mu2)
    V_sad = V_cessi(y_saddle, p, mu2)
    DeltaV = V_sad - V_eq

    C = np.sqrt(lam_eq * lam_sad) / (2.0 * np.pi)
    tau = 1.0 / lam_eq
    inv_Ctau = 1.0 / (C * tau)

    return {
        'lam_eq': lam_eq,
        'lam_sad': lam_sad,
        'DeltaV': DeltaV,
        'C': C,
        'tau': tau,
        'inv_Ctau': inv_Ctau,
    }

def D_kramers(DeltaV, sigma, inv_Ctau, K=0.55):
    """Kramers approximation: D = K * exp(2*DV/sigma^2) / (C*tau)"""
    barrier = 2.0 * DeltaV / sigma**2
    if barrier > 700:
        return np.inf
    return K * np.exp(barrier) * inv_Ctau

# ============================================================
# SIGMA* FINDER (bisection)
# ============================================================

def find_sigma_star(p, mu2, y_eq, y_saddle, D_target, sig_lo=0.001, sig_hi=2.0):
    """Find sigma where D_exact(sigma) = D_target via bisection."""
    # Check bracket
    D_lo = compute_D_exact(p, mu2, y_eq, y_saddle, sig_lo)
    D_hi = compute_D_exact(p, mu2, y_eq, y_saddle, sig_hi)

    if D_lo == np.inf:
        D_lo = 1e30

    if D_lo < D_target:
        return None, D_lo, D_hi
    if D_hi > D_target:
        # Need wider bracket
        for s in [5.0, 10.0, 50.0]:
            D_test = compute_D_exact(p, mu2, y_eq, y_saddle, s)
            if D_test < D_target:
                sig_hi = s
                D_hi = D_test
                break
        else:
            return None, D_lo, D_hi

    def obj(s):
        d = compute_D_exact(p, mu2, y_eq, y_saddle, s)
        if d == np.inf:
            return 1e30 - D_target
        return d - D_target

    try:
        sigma_star = brentq(obj, sig_lo, sig_hi, xtol=1e-10, maxiter=300)
        D_at_star = compute_D_exact(p, mu2, y_eq, y_saddle, sigma_star)
        return sigma_star, D_at_star, None
    except ValueError:
        return None, D_lo, D_hi

# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_scenario(name, p, mu2, t_d=219.0):
    """Full Kramers analysis for a given (p, mu2) parameterization."""
    print(f"\n{'=' * 70}")
    print(f"  SCENARIO: {name}")
    print(f"  p = {p}, mu^2 = {mu2}, t_d = {t_d} years")
    print(f"{'=' * 70}")

    # 1. Find fixed points
    roots = find_equilibria(p, mu2)
    classified = classify_equilibria(roots, p, mu2)

    print(f"\n  Fixed points ({len(roots)} found):")
    for y, lam, typ in classified:
        V = V_cessi(y, p, mu2)
        tau_dim = t_d / abs(lam) if lam != 0 else np.inf
        print(f"    y = {y:.6f}  lambda = {lam:+.6f}  [{typ}]  V = {V:.8f}  tau = {tau_dim:.0f} yr")

    stable = [(y, lam) for y, lam, typ in classified if typ == 'stable']
    unstable = [(y, lam) for y, lam, typ in classified if typ == 'unstable']

    if len(stable) < 2 or len(unstable) < 1:
        print("  NOT BISTABLE at these parameters. Skipping.")
        return None

    # Identify: thermal mode (high y), saline mode (low y), saddle (middle)
    y_saline = stable[0][0]
    y_thermal = stable[-1][0]
    y_saddle = unstable[0][0]

    print(f"\n  Saline mode (low y):   y = {y_saline:.6f}")
    print(f"  Saddle:                y = {y_saddle:.6f}")
    print(f"  Thermal mode (high y): y = {y_thermal:.6f}")

    # 2. Kramers quantities for THERMAL mode (escape thermal → saline)
    kq_th = kramers_quantities(p, mu2, y_thermal, y_saddle)

    print(f"\n  --- THERMAL MODE (current AMOC) ---")
    print(f"  lambda_eq   = {kq_th['lam_eq']:.6f}")
    print(f"  lambda_sad  = {kq_th['lam_sad']:.6f}")
    print(f"  DeltaV      = {kq_th['DeltaV']:.8f}")
    print(f"  C           = {kq_th['C']:.6f}")
    print(f"  tau (nondim)= {kq_th['tau']:.6f}")
    print(f"  tau (years) = {kq_th['tau'] * t_d:.1f}")
    print(f"  1/(C*tau)   = {kq_th['inv_Ctau']:.4f}")

    # 3. Kramers quantities for SALINE mode (escape saline → thermal)
    kq_sa = kramers_quantities(p, mu2, y_saline, y_saddle)

    print(f"\n  --- SALINE MODE (collapsed AMOC) ---")
    print(f"  lambda_eq   = {kq_sa['lam_eq']:.6f}")
    print(f"  lambda_sad  = {kq_sa['lam_sad']:.6f}")
    print(f"  DeltaV      = {kq_sa['DeltaV']:.8f}")
    print(f"  C           = {kq_sa['C']:.6f}")
    print(f"  tau (nondim)= {kq_sa['tau']:.6f}")
    print(f"  tau (years) = {kq_sa['tau'] * t_d:.1f}")
    print(f"  1/(C*tau)   = {kq_sa['inv_Ctau']:.4f}")

    # 4. D_exact scan over sigma (THERMAL mode escape)
    print(f"\n  --- D_exact(sigma) SCAN (thermal mode escape) ---")
    print(f"  {'sigma':>8s}  {'2DV/s^2':>10s}  {'D_exact':>12s}  {'D_Kramers':>12s}  {'K_actual':>10s}  {'MFPT(yr)':>12s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*12}")

    results = []
    for sigma in [0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
        D_ex = compute_D_exact(p, mu2, y_thermal, y_saddle, sigma)
        barrier = 2.0 * kq_th['DeltaV'] / sigma**2
        D_kr = np.exp(barrier) * kq_th['inv_Ctau']  # Without K
        K_act = D_ex / D_kr if D_kr > 0 and D_ex < 1e20 else np.nan
        MFPT_yr = D_ex * kq_th['tau'] * t_d if D_ex < 1e20 else np.inf

        if D_ex < 1e15:
            results.append((sigma, barrier, D_ex, D_kr, K_act, MFPT_yr))
            print(f"  {sigma:8.4f}  {barrier:10.3f}  {D_ex:12.4e}  {D_kr:12.4e}  {K_act:10.4f}  {MFPT_yr:12.1f}")
        else:
            print(f"  {sigma:8.4f}  {barrier:10.3f}  {'>>1e15':>12s}  {D_kr:12.4e}  {'---':>10s}  {'>>':>12s}")

    # 5. Find sigma* for D_observed targets from D-O events
    print(f"\n  --- SIGMA* FOR D_observed TARGETS ---")

    D_targets = {
        'D=5 (short stadials)': 5.0,
        'D=8 (typical D-O)': 8.0,
        'D=11 (mean spacing)': 11.0,
        'D=14 (long stadials)': 14.0,
        'D=47 (Holocene lower bound)': 47.0,
        'D=100 (deeply stable)': 100.0,
        'D=200 (very stable)': 200.0,
    }

    sigma_results = {}
    for label, D_targ in D_targets.items():
        result = find_sigma_star(p, mu2, y_thermal, y_saddle, D_targ)
        sigma_star, D_at_star, _ = result
        if sigma_star is not None:
            barrier_star = 2.0 * kq_th['DeltaV'] / sigma_star**2
            D_kr_star = np.exp(barrier_star) * kq_th['inv_Ctau']
            K_star = D_at_star / D_kr_star if D_kr_star > 0 else np.nan
            MFPT_yr = D_at_star * kq_th['tau'] * t_d
            sigma_results[label] = {
                'D_target': D_targ,
                'sigma_star': sigma_star,
                'D_exact': D_at_star,
                'barrier': barrier_star,
                'K_actual': K_star,
                'MFPT_yr': MFPT_yr,
            }
            print(f"  {label}:")
            print(f"    sigma* = {sigma_star:.6f}")
            print(f"    D_exact(sigma*) = {D_at_star:.4f}")
            print(f"    2*DV/sigma*^2 = {barrier_star:.4f}")
            print(f"    K_actual = {K_star:.4f}")
            print(f"    MFPT = {MFPT_yr:.0f} years")
        else:
            print(f"  {label}: sigma* NOT FOUND (D_lo={D_at_star:.2e})")

    # 6. D_exact scan for SALINE mode (stadial escape → D-O warming)
    print(f"\n  --- D_exact(sigma) SCAN (saline/stadial mode escape) ---")
    print(f"  {'sigma':>8s}  {'2DV/s^2':>10s}  {'D_exact':>12s}  {'D_Kramers':>12s}  {'K_actual':>10s}  {'MFPT(yr)':>12s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*12}")

    for sigma in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30]:
        D_ex = compute_D_exact(p, mu2, y_saline, y_saddle, sigma)
        barrier = 2.0 * kq_sa['DeltaV'] / sigma**2
        D_kr = np.exp(barrier) * kq_sa['inv_Ctau']
        K_act = D_ex / D_kr if D_kr > 0 and D_ex < 1e20 else np.nan
        MFPT_yr = D_ex * kq_sa['tau'] * t_d if D_ex < 1e20 else np.inf

        if D_ex < 1e15:
            print(f"  {sigma:8.4f}  {barrier:10.3f}  {D_ex:12.4e}  {D_kr:12.4e}  {K_act:10.4f}  {MFPT_yr:12.1f}")
        else:
            print(f"  {sigma:8.4f}  {barrier:10.3f}  {'>>1e15':>12s}  {D_kr:12.4e}  {'---':>10s}  {'>>':>12s}")

    # 6b. Find sigma* for D_observed targets — SALINE mode (D-O warming events)
    print(f"\n  --- SIGMA* FOR D_observed TARGETS (saline/stadial mode escape) ---")

    saline_sigma_results = {}
    for label, D_targ in D_targets.items():
        result = find_sigma_star(p, mu2, y_saline, y_saddle, D_targ)
        sigma_star_sa, D_at_star_sa, _ = result
        if sigma_star_sa is not None:
            barrier_star_sa = 2.0 * kq_sa['DeltaV'] / sigma_star_sa**2
            D_kr_star_sa = np.exp(barrier_star_sa) * kq_sa['inv_Ctau'] if barrier_star_sa < 700 else np.inf
            K_star_sa = D_at_star_sa / D_kr_star_sa if D_kr_star_sa > 0 and D_kr_star_sa < 1e20 else np.nan
            MFPT_yr_sa = D_at_star_sa * kq_sa['tau'] * t_d
            saline_sigma_results[label] = {
                'D_target': D_targ,
                'sigma_star': sigma_star_sa,
                'D_exact': D_at_star_sa,
                'barrier': barrier_star_sa,
                'K_actual': K_star_sa,
                'MFPT_yr': MFPT_yr_sa,
            }
            print(f"  {label}:")
            print(f"    sigma* = {sigma_star_sa:.6f}")
            print(f"    D_exact(sigma*) = {D_at_star_sa:.4f}")
            print(f"    2*DV/sigma*^2 = {barrier_star_sa:.4f}")
            print(f"    K_actual = {K_star_sa:.4f}")
            print(f"    MFPT = {MFPT_yr_sa:.0f} years")
        else:
            print(f"  {label}: sigma* NOT FOUND")

    # 7. Kramers accuracy test: D_Kramers vs D_exact at fixed sigma
    print(f"\n  --- KRAMERS ACCURACY (D_Kramers vs D_exact) ---")
    print(f"  Using K = 0.55 (parabolic well estimate)")
    print(f"  {'sigma':>8s}  {'D_exact':>12s}  {'D_Kr(K=0.55)':>14s}  {'ratio':>8s}  {'error':>8s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*14}  {'-'*8}  {'-'*8}")

    for sigma in [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        D_ex = compute_D_exact(p, mu2, y_thermal, y_saddle, sigma)
        D_kr_055 = D_kramers(kq_th['DeltaV'], sigma, kq_th['inv_Ctau'], K=0.55)
        if D_ex < 1e15 and D_kr_055 < 1e15:
            ratio = D_kr_055 / D_ex
            error = abs(ratio - 1.0) * 100
            print(f"  {sigma:8.4f}  {D_ex:12.4e}  {D_kr_055:14.4e}  {ratio:8.4f}  {error:7.1f}%")

    return {
        'name': name, 'p': p, 'mu2': mu2,
        'y_saline': y_saline, 'y_saddle': y_saddle, 'y_thermal': y_thermal,
        'kq_thermal': kq_th, 'kq_saline': kq_sa,
        'sigma_results': sigma_results,
    }


# ============================================================
# RUN ALL SCENARIOS
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  THERMOHALINE CIRCULATION — KRAMERS/MFPT TEST")
    print("  Cessi (1994) reduced 1D model")
    print("  dy/dt = p - y*(1 + mu^2*(1-y)^2)")
    print("=" * 70)

    mu2 = 4.0  # Standard Cessi parameter
    t_d = 219.0  # Salinity diffusion timescale (years)

    # Saddle-node locations
    # dp/dy = 0: 12y^2 - 16y + 5 = 0 → y = 5/6 (p=0.926), y = 1/2 (p=1.0)
    print(f"\n  Bistable range for mu^2={mu2}: p in (0.9259, 1.0000)")
    print(f"  Salinity diffusion timescale: t_d = {t_d} years")
    print(f"  Temperature relaxation: t_r ~ 25 days (alpha = t_d/t_r ~ 3200)")

    results = {}

    # Scenario 1: Near upper saddle-node (GLACIAL D-O WINDOW)
    results['glacial_near_SN'] = analyze_scenario(
        "GLACIAL (near upper saddle-node) — D-O window",
        p=0.93, mu2=mu2, t_d=t_d
    )

    # Scenario 2: Slightly further from SN
    results['glacial_moderate'] = analyze_scenario(
        "GLACIAL (moderate) — D-O window",
        p=0.95, mu2=mu2, t_d=t_d
    )

    # Scenario 3: Middle of bistable range
    results['mid_bistable'] = analyze_scenario(
        "MID-BISTABLE (p=0.96)",
        p=0.96, mu2=mu2, t_d=t_d
    )

    # Scenario 4: Deep in bistable range (HOLOCENE-like)
    results['holocene'] = analyze_scenario(
        "HOLOCENE-LIKE (deep barrier, p=0.97)",
        p=0.97, mu2=mu2, t_d=t_d
    )

    # Scenario 5: p=0.98 (saline mode barrier shrinking)
    results['p098'] = analyze_scenario(
        "p=0.98 (saline barrier shrinking)",
        p=0.98, mu2=mu2, t_d=t_d
    )

    # Scenario 6: p=0.985
    results['p0985'] = analyze_scenario(
        "p=0.985",
        p=0.985, mu2=mu2, t_d=t_d
    )

    # Scenario 7: Near lower saddle-node
    results['near_lower_SN'] = analyze_scenario(
        "NEAR LOWER SADDLE-NODE (p=0.99)",
        p=0.99, mu2=mu2, t_d=t_d
    )

    # Scenario 8: Very near lower saddle-node
    results['p0995'] = analyze_scenario(
        "p=0.995 (very near lower SN)",
        p=0.995, mu2=mu2, t_d=t_d
    )

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print(f"\n\n{'=' * 70}")
    print(f"  SUMMARY TABLE")
    print(f"{'=' * 70}")

    print(f"\n  {'Scenario':<35s} {'DeltaV_th':>10s} {'DeltaV_sa':>10s} {'1/(Ct)_th':>10s} {'tau_th(yr)':>10s}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for key, res in results.items():
        if res is None:
            continue
        kq = res['kq_thermal']
        kq_s = res['kq_saline']
        print(f"  {res['name'][:35]:<35s} {kq['DeltaV']:10.6f} {kq_s['DeltaV']:10.6f} {kq['inv_Ctau']:10.4f} {kq['tau']*t_d:10.1f}")

    # D_observed comparison
    print(f"\n\n  --- D_observed from paleoclimate ---")
    print(f"  Stadial durations: exponential, mean ~1328 yr (Lohmann & Ditlevsen 2019)")
    print(f"  Event spacing: ~2800 yr (Ditlevsen 2005)")
    print(f"  tau_relax (AMOC): ~200-300 yr")
    print(f"  D_observed (stadial): ~5-14")
    print(f"  D_observed (Holocene): >47 (no events in 11,700 yr)")

    # K convergence
    print(f"\n\n  --- K_actual ACROSS SCENARIOS ---")
    print(f"  (K = D_exact / (exp(2DV/s^2) * 1/(C*tau)))")
    print(f"  Compare to: K=0.55 (parabolic), K=0.35 (anharmonic)")

    for key, res in results.items():
        if res is None or not res['sigma_results']:
            continue
        print(f"\n  {res['name'][:50]}:")
        for label, sr in res['sigma_results'].items():
            if sr['barrier'] > 1.5:  # Only report K where Kramers is valid
                print(f"    {label}: K = {sr['K_actual']:.4f}  (barrier = {sr['barrier']:.2f})")

    print(f"\n\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")
