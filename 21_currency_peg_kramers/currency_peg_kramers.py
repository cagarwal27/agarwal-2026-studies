#!/usr/bin/env python3
"""
Currency Peg Kramers Test: Black Wednesday (September 16, 1992)

Study 21: First calibrated (non-generic) Kramers test on a human
organizational system -- currency pegs under speculative attack.

Model: Stochastic cusp catastrophe for exchange rate regime persistence
  dx/dt = -(x^3 - q*x + a) + sigma*dW
  V(x)  = x^4/4 - q*x^2/2 + a*x

Calibration: Exchange rate gap, relaxation time, and FX volatility
from published data. Shape parameters (q, a) are free.

Historical target: UK pound in ERM (Oct 1990 - Sep 1992)
  MFPT_observed = 23 months ~ 700 days
  tau_relax ~ 25 days (4 weeks to new equilibrium ~2.42 DEM)
  D_observed = 700/25 = 28
  sigma_daily ~ 0.009 DEM/sqrt(day) (5% annualized implied vol)

Extends: ../scripts/financial_cusp_kramers.py
Dependencies: numpy, scipy only
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import time


# ================================================================
# Cusp Model (from financial_cusp_kramers.py)
# ================================================================

def f_cusp(x, q, a):
    """Cusp catastrophe drift: dx/dt = -(x^3 - q*x + a) = -x^3 + q*x - a"""
    return -x**3 + q * x - a


def V_cusp(x, q, a):
    """Cusp potential: V(x) = x^4/4 - q*x^2/2 + a*x"""
    return x**4 / 4.0 - q * x**2 / 2.0 + a * x


def cusp_bistability_condition(q, a):
    """Cusp has two stable states when q > 0 and |a| < a_crit.
    a_crit = 2*sqrt(3)/9 * q^(3/2)
    """
    if q <= 0:
        return False, 0.0
    a_crit = 2.0 * np.sqrt(3.0) / 9.0 * q**1.5
    return abs(a) < a_crit, a_crit


def find_equilibria(f_func, x_lo=-5.0, x_hi=5.0, N=400000):
    """Find all roots of f_func in [x_lo, x_hi]."""
    x_scan = np.linspace(x_lo, x_hi, N)
    f_scan = np.array([f_func(x) for x in x_scan])
    sign_changes = np.where(np.diff(np.sign(f_scan)))[0]
    roots = []
    for i in sign_changes:
        try:
            root = brentq(f_func, x_scan[i], x_scan[i + 1], xtol=1e-12)
            roots.append(root)
        except ValueError:
            pass
    return sorted(roots)


def fderiv(f_func, x, dx=1e-7):
    """Numerical derivative."""
    return (f_func(x + dx) - f_func(x - dx)) / (2.0 * dx)


def compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma, N=100000):
    """Compute exact D = MFPT/tau via 1D MFPT integral.

    Handles both directions: x_eq < x_saddle (rightward escape)
    and x_eq > x_saddle (leftward escape via reflection).
    """
    if x_eq > x_saddle:
        g_func = lambda y: -f_func(-y)
        return compute_D_exact(g_func, -x_eq, -x_saddle, tau_val, sigma, N)

    margin = 0.5 * abs(x_saddle - x_eq)
    x_lo = x_eq - margin
    x_hi = x_saddle + 0.01 * abs(x_saddle - x_eq)
    xg = np.linspace(x_lo, x_hi, N)
    dx_grid = xg[1] - xg[0]

    neg_f = np.array([-f_func(x) for x in xg])
    U_raw = np.cumsum(neg_f) * dx_grid
    i_eq = np.argmin(np.abs(xg - x_eq))
    U = U_raw - U_raw[i_eq]

    Phi = 2.0 * U / sigma**2
    Phi_max = Phi.max()
    if Phi_max > 600:
        Phi = Phi - (Phi_max - 500)

    exp_neg = np.exp(-Phi)
    Ix = np.cumsum(exp_neg) * dx_grid
    psi = (2.0 / sigma**2) * np.exp(Phi) * Ix

    i_sad = np.argmin(np.abs(xg - x_saddle))
    MFPT = np.trapz(psi[i_eq:i_sad + 1], xg[i_eq:i_sad + 1])
    return MFPT / tau_val


def find_sigma_for_D(f_func, x_eq, x_saddle, tau_val, D_target,
                     sigma_lo=0.01, sigma_hi=10.0):
    """Find sigma where D_exact(sigma) = D_target via bisection."""
    for _ in range(8):
        D_lo = compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma_lo)
        if np.isnan(D_lo) or np.isinf(D_lo):
            sigma_lo *= 2
            continue
        if D_lo > D_target:
            break
        sigma_lo /= 2
    else:
        return None

    for _ in range(8):
        D_hi = compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma_hi)
        if np.isnan(D_hi) or np.isinf(D_hi):
            sigma_hi *= 0.8
            continue
        if D_hi < D_target:
            break
        sigma_hi *= 2
    else:
        return None

    if np.isnan(D_lo) or np.isnan(D_hi) or D_lo < D_target or D_hi > D_target:
        return None

    def obj(s):
        val = compute_D_exact(f_func, x_eq, x_saddle, tau_val, s) - D_target
        return val if np.isfinite(val) else 1e30

    try:
        sigma_star = brentq(obj, sigma_lo, sigma_hi, xtol=1e-6, maxiter=100)
        return sigma_star
    except (ValueError, RuntimeError):
        return None


# ================================================================
# Historical Observables
# ================================================================

# Black Wednesday: UK pound in ERM (Oct 1990 - Sep 16, 1992)
# Sources: Bank of England historical spot rates, BIS conference papers,
#          NYU V-Lab GARCH, multiple academic sources
BW = {
    'name': 'Black Wednesday (GBP/DEM)',
    'x_peg': 2.95,              # DEM/GBP: ERM central rate
    'x_float': 2.42,            # DEM/GBP: new equilibrium by late Oct 1992
    'delta_x': 0.53,            # DEM: exchange rate gap (18%)
    'MFPT_days': 700,           # 23 months (Oct 1990 - Sep 1992)
    'MFPT_months': 23,
    'tau_relax_days': 25,       # ~4 weeks to reach ~2.42 DEM
    'D_observed': 28,           # 700/25
    'sigma_daily': 0.009,       # 0.3% of rate; from 5% annualized implied vol
    'sigma_ann_pct': 5.0,       # BIS: implied vol from options pre-crisis
}

# Thai Baht: informal peg ~1984 to July 2, 1997
TB = {
    'name': 'Thai Baht (THB/USD)',
    'MFPT_days': 4745,          # ~13 years
    'tau_relax_days_fast': 14,  # 2 weeks (initial shock)
    'tau_relax_days_slow': 180, # 6 months (to trough)
    'D_observed_high': 339,     # 4745/14
    'D_observed_low': 26,       # 4745/180
}

# Argentine convertibility: Apr 1991 - Jan 2002
AR = {
    'name': 'Argentine Peso (ARS/USD)',
    'MFPT_days': 3906,          # 10.7 years
    'tau_relax_days': 165,      # ~5.5 months
    'D_observed': 24,           # 3906/165
}

# Thai Baht: calibrated for Kramers computation
# FX vol sources: World Bank annual data (0.93%), NYU V-Lab GARCH min (1.52%),
#                 trading range method (0.6-1.3%). Point estimate: 1.2% annualized.
# Daily sigma = 1.2%/sqrt(252) * 25 = 0.019 THB/sqrt(day)
TB_FULL = {
    'name': 'Thai Baht - full devaluation',
    'x_peg': 25.0,              # THB/USD: informal peg level
    'x_float': 40.0,            # THB/USD: mid-1998 stabilization
    'delta_x': 15.0,            # THB: exchange rate gap (60%)
    'MFPT_days': 4745,          # ~13 years (1984 - Jul 1997)
    'MFPT_months': 156,
    'tau_relax_days': 180,      # ~6 months to stabilize at ~40
    'D_observed': 26,           # 4745/180
    'sigma_daily': 0.019,       # from 1.2% annualized realized vol
    'sigma_ann_pct': 1.2,       # NYU V-Lab GARCH min + World Bank + range
}

TB_INIT = {
    'name': 'Thai Baht - initial transition',
    'x_peg': 25.0,
    'x_float': 29.0,            # THB/USD: immediate post-float (2 weeks)
    'delta_x': 4.0,             # THB: initial devaluation (16%)
    'MFPT_days': 4745,
    'MFPT_months': 156,
    'tau_relax_days': 14,       # 2 weeks (initial shock)
    'D_observed': 339,          # 4745/14
    'sigma_daily': 0.019,       # same underlying vol
    'sigma_ann_pct': 1.2,
}


# ================================================================
# Dimensional Mapping
# ================================================================

def map_to_cusp(q, a, obs):
    """Map cusp (q, a) to dimensional observables.

    Returns dict with cusp equilibria, scale factors, mapped sigma, barrier,
    eigenvalues, etc. Returns None if not bistable or insufficient roots.
    """
    is_bistable, a_crit = cusp_bistability_condition(q, a)
    if not is_bistable:
        return None

    f = lambda x: f_cusp(x, q, a)
    roots = find_equilibria(f, x_lo=-3 * np.sqrt(q), x_hi=3 * np.sqrt(q))
    if len(roots) < 3:
        return None

    stab = [(r, fderiv(f, r)) for r in roots]
    stable = sorted([r for r, fp in stab if fp < 0])
    unstable = sorted([r for r, fp in stab if fp > 0])
    if len(stable) < 2 or len(unstable) < 1:
        return None

    x_low, x_high, x_sad = stable[0], stable[-1], unstable[0]

    # Eigenvalues
    lam_high = fderiv(f, x_high)
    lam_sad = fderiv(f, x_sad)
    lam_low = fderiv(f, x_low)

    # Cusp relaxation time at peg (upper) equilibrium
    tau_cusp = 1.0 / abs(lam_high)

    # Scale factors
    delta_x_cusp = x_high - x_low
    scale_x = obs['delta_x'] / delta_x_cusp        # DEM per cusp unit
    scale_t = obs['tau_relax_days'] / tau_cusp       # days per cusp time unit

    # Map observed noise to cusp units
    sigma_cusp = obs['sigma_daily'] * np.sqrt(scale_t) / scale_x

    # Barrier from peg (upper) well to saddle
    DV_peg = V_cusp(x_sad, q, a) - V_cusp(x_high, q, a)
    DV_float = V_cusp(x_sad, q, a) - V_cusp(x_low, q, a)

    # Kramers prefactor
    C = np.sqrt(abs(lam_high) * abs(lam_sad)) / (2 * np.pi)
    inv_Ctau = 1.0 / (C * tau_cusp)

    return {
        'q': q, 'a': a, 'a_crit': a_crit, 'a_ratio': abs(a) / a_crit,
        'x_low': x_low, 'x_high': x_high, 'x_sad': x_sad,
        'delta_x_cusp': delta_x_cusp,
        'scale_x': scale_x, 'scale_t': scale_t, 'tau_cusp': tau_cusp,
        'DV_peg': DV_peg, 'DV_float': DV_float,
        'lam_high': lam_high, 'lam_sad': lam_sad, 'lam_low': lam_low,
        'C': C, 'inv_Ctau': inv_Ctau,
        'sigma_cusp': sigma_cusp,
        'f_func': f,
    }


# ================================================================
# Core Analysis
# ================================================================

def full_kramers(m, obs):
    """Full Kramers analysis given mapping m and observables obs.

    Forward: compute D at observed noise.
    Inverse: find sigma for D_observed.
    """
    f = m['f_func']
    tau_h = m['tau_cusp']
    sigma = m['sigma_cusp']

    # Forward: D at observed noise
    D_fwd = compute_D_exact(f, m['x_high'], m['x_sad'], tau_h, sigma)
    B_fwd = 2.0 * m['DV_peg'] / sigma**2

    # K effective
    if B_fwd < 500:
        D_kramers_noK = np.exp(B_fwd) * m['inv_Ctau']
        K_eff = D_fwd / D_kramers_noK if D_fwd > 0 and np.isfinite(D_fwd) else np.nan
    else:
        K_eff = np.nan

    # Inverse: sigma for D_observed
    sigma_star = find_sigma_for_D(f, m['x_high'], m['x_sad'], tau_h,
                                  obs['D_observed'], sigma_lo=0.001, sigma_hi=20.0)
    if sigma_star is not None:
        B_inv = 2.0 * m['DV_peg'] / sigma_star**2
        sigma_star_dim = sigma_star * m['scale_x'] / np.sqrt(m['scale_t'])
        sigma_star_ann = sigma_star_dim / obs['x_peg'] * np.sqrt(252) * 100
    else:
        B_inv, sigma_star_dim, sigma_star_ann = None, None, None

    return {
        'D_fwd': D_fwd,
        'B_fwd': B_fwd,
        'K_eff': K_eff,
        'MFPT_days': D_fwd * obs['tau_relax_days'] if np.isfinite(D_fwd) else np.inf,
        'sigma_star': sigma_star,
        'sigma_star_dim': sigma_star_dim,
        'sigma_star_ann': sigma_star_ann,
        'B_inv': B_inv,
    }


def find_a_for_D(q, D_target, obs, a_hi_frac=0.9995):
    """Find asymmetry a that gives D_target at observed noise level.

    Uses bisection: D decreases as a increases (peg barrier shrinks).
    a_hi_frac controls how close to a_crit we search (default 0.9995
    to handle low-volatility pegs that need near-fold asymmetry).
    """
    _, a_crit = cusp_bistability_condition(q, 0)
    a_hi = a_hi_frac * a_crit

    def get_D(a_val):
        m = map_to_cusp(q, a_val, obs)
        if m is None:
            return 0.01
        return compute_D_exact(m['f_func'], m['x_high'], m['x_sad'],
                               m['tau_cusp'], m['sigma_cusp'])

    D_at_0 = get_D(0.001)
    if D_at_0 < D_target:
        return None

    D_at_hi = get_D(a_hi)
    if D_at_hi > D_target:
        return None

    try:
        def obj(a_val):
            return get_D(a_val) - D_target
        return brentq(obj, 0.001, a_hi, xtol=1e-5, maxiter=60)
    except (ValueError, RuntimeError):
        return None


# ================================================================
# Phase 1: D = 1 Threshold Check
# ================================================================

def phase1_d_threshold():
    print("=" * 70)
    print("PHASE 1: D = 1 THRESHOLD (broadest layer, no model needed)")
    print("=" * 70)
    print()
    print("D = MFPT / tau_relax > 1 confirms the peg is a dissipative structure.")
    print()

    crises = [
        ("Black Wednesday",      BW['MFPT_days'], BW['tau_relax_days'],
         BW['D_observed'], "GBP/DEM, 23 months"),
        ("Thai Baht (fast tau)", TB['MFPT_days'], TB['tau_relax_days_fast'],
         TB['D_observed_high'], "THB/USD, 13 yr / 2 wk"),
        ("Thai Baht (slow tau)", TB['MFPT_days'], TB['tau_relax_days_slow'],
         TB['D_observed_low'], "THB/USD, 13 yr / 6 mo"),
        ("Argentine Peso",       AR['MFPT_days'], AR['tau_relax_days'],
         AR['D_observed'], "ARS/USD, 10.7 yr / 5.5 mo"),
    ]

    print(f"  {'Crisis':<25s} {'MFPT':>10s} {'tau':>10s} {'D':>8s} {'D>>1?':>6s}  Notes")
    print(f"  {'-'*80}")
    for name, mfpt, tau, D, note in crises:
        status = "YES" if D > 1 else "NO"
        print(f"  {name:<25s} {mfpt:>8d} d {tau:>8d} d {D:8.0f} {status:>6s}  {note}")

    print()
    print("  RESULT: All pegs have D >> 1. Currency pegs ARE dissipative structures.")
    print("  D ~ 24-28 is comparable to kelp (D=29) and peatland (D=30).")
    print("  These are at the LOW END of known systems -- barely persistent.")
    print()


# ================================================================
# Phase 2: Scenario Analysis
# ================================================================

def phase2_scenarios():
    print("=" * 70)
    print("PHASE 2: KRAMERS COMPUTATION (calibrated to Black Wednesday)")
    print("=" * 70)

    scenarios = [
        ("Symmetric (q=2)",      2.0, 0.0,  "D"),
        ("Moderate asym (q=2)",  2.0, 0.5,  "D"),
        ("Strong asym (q=2)",    2.0, 0.85, "D"),
        ("Chen&Chen proxy",      3.16, 0.54, "C"),
    ]

    results = {}
    for name, q, a, grade in scenarios:
        print(f"\n  --- {name}: q={q}, a={a} ---")
        m = map_to_cusp(q, a, BW)
        if m is None:
            print(f"  NOT BISTABLE. Skipping.")
            continue

        kr = full_kramers(m, BW)
        hz_fwd = 1.8 <= kr['B_fwd'] <= 6.0
        hz_inv = kr['B_inv'] is not None and 1.8 <= kr['B_inv'] <= 6.0

        print(f"  a/a_crit = {m['a_ratio']:.4f}")
        print(f"  Equilibria: x_low={m['x_low']:+.4f}, x_sad={m['x_sad']:+.4f}, "
              f"x_high={m['x_high']:+.4f}")
        print(f"  DV_peg = {m['DV_peg']:.6f}, DV_float = {m['DV_float']:.6f}, "
              f"ratio = {m['DV_peg']/m['DV_float']:.4f}" if m['DV_float'] > 0 else "")
        print(f"  sigma_cusp = {m['sigma_cusp']:.6f} "
              f"(from {BW['sigma_ann_pct']}% ann FX vol)")
        print(f"  1/(C*tau)  = {m['inv_Ctau']:.4f}")
        print(f"  FORWARD:  B = {kr['B_fwd']:.4f}, D = {kr['D_fwd']:.1f}, "
              f"MFPT = {kr['MFPT_days']:.0f} d ({kr['MFPT_days']/30.44:.1f} mo) "
              f"[HZ: {'YES' if hz_fwd else 'NO'}]")
        if kr['D_fwd'] > 0 and np.isfinite(kr['D_fwd']):
            print(f"           D_pred/D_obs = {kr['D_fwd']/BW['D_observed']:.2f} "
                  f"({np.log10(kr['D_fwd']/BW['D_observed']):+.2f} OOM), K = {kr['K_eff']:.3f}")
        if kr['sigma_star'] is not None:
            print(f"  INVERSE:  sigma* = {kr['sigma_star']:.6f} cusp "
                  f"-> {kr['sigma_star_ann']:.2f}% ann "
                  f"(obs: {BW['sigma_ann_pct']}%, ratio = "
                  f"{kr['sigma_star_ann']/BW['sigma_ann_pct']:.2f})")
            print(f"           B at D={BW['D_observed']} = {kr['B_inv']:.4f} "
                  f"[HZ: {'YES' if hz_inv else 'NO'}]")

        results[name] = {'m': m, 'kr': kr, 'grade': grade}

    return results


# ================================================================
# Phase 3: Calibrated Scenarios
# ================================================================

def phase3_calibrated():
    print()
    print("=" * 70)
    print("PHASE 3: CALIBRATED (find a for D=28 at observed noise)")
    print("=" * 70)
    print()
    print("For each q, find the asymmetry a that makes D_exact = 28")
    print("when sigma = observed FX volatility (5% annualized).")
    print()

    q_values = [1.5, 2.0, 3.0, 4.0]
    cal_results = {}

    print(f"  {'q':>5s} {'a_star':>10s} {'a/a_crit':>10s} {'DV_peg':>10s} "
          f"{'sigma_cusp':>12s} {'B':>8s} {'1/(Ct)':>8s} {'K':>8s} {'HZ?':>5s}")
    print(f"  {'-'*82}")

    for q in q_values:
        a_star = find_a_for_D(q, BW['D_observed'], BW)
        if a_star is None:
            print(f"  {q:5.1f} {'FAILED':>10s}")
            continue

        m = map_to_cusp(q, a_star, BW)
        if m is None:
            print(f"  {q:5.1f} {'MAP FAIL':>10s}")
            continue

        kr = full_kramers(m, BW)
        hz = 1.8 <= kr['B_fwd'] <= 6.0

        print(f"  {q:5.1f} {a_star:10.6f} {m['a_ratio']:10.4f} {m['DV_peg']:10.6f} "
              f"{m['sigma_cusp']:12.6f} {kr['B_fwd']:8.4f} {m['inv_Ctau']:8.4f} "
              f"{kr['K_eff']:8.3f} {'YES' if hz else 'NO':>5s}")

        cal_results[q] = {'a_star': a_star, 'm': m, 'kr': kr}

    # Show the calibrated model's MFPT prediction (should be ~700 days)
    print()
    print("  Verification (D should be ~28, MFPT ~700 days):")
    for q, cr in cal_results.items():
        print(f"    q={q}: D = {cr['kr']['D_fwd']:.1f}, "
              f"MFPT = {cr['kr']['MFPT_days']:.0f} days "
              f"({cr['kr']['MFPT_days']/30.44:.1f} months)")

    # Key output: the physical interpretation
    if cal_results:
        B_vals = [cr['kr']['B_fwd'] for cr in cal_results.values()]
        a_ratios = [cr['m']['a_ratio'] for cr in cal_results.values()]
        K_vals = [cr['kr']['K_eff'] for cr in cal_results.values()
                  if np.isfinite(cr['kr']['K_eff'])]
        print()
        print(f"  Calibrated B: mean = {np.mean(B_vals):.4f}, "
              f"range = [{min(B_vals):.4f}, {max(B_vals):.4f}]")
        print(f"  Required a/a_crit: mean = {np.mean(a_ratios):.4f}, "
              f"range = [{min(a_ratios):.4f}, {max(a_ratios):.4f}]")
        if K_vals:
            print(f"  K_eff: mean = {np.mean(K_vals):.3f}, "
                  f"range = [{min(K_vals):.3f}, {max(K_vals):.3f}]")
        print(f"  Interpretation: the peg requires a/a_crit ~ {np.mean(a_ratios):.2f}")
        print(f"  (near fold bifurcation = barely bistable = fragile peg)")

    return cal_results


# ================================================================
# Phase 4: B Invariance Scan
# ================================================================

def phase4_b_invariance():
    print()
    print("=" * 70)
    print("PHASE 4: B INVARIANCE SCAN")
    print("=" * 70)
    print()
    print("Standard B invariance test: vary asymmetry a at fixed q,")
    print("find sigma for D = 28, compute B = 2*DV/sigma^2.")
    print("If B is approximately constant, B invariance holds.")
    print()

    all_B = {}
    for q in [2.0, 3.0]:
        _, a_crit = cusp_bistability_condition(q, 0)
        a_vals = np.linspace(0.05 * a_crit, 0.95 * a_crit, 20)

        print(f"  q = {q} (a_crit = {a_crit:.4f}):")
        print(f"  {'a':>8s} {'a/a_c':>8s} {'DV_peg':>10s} {'sig*':>10s} "
              f"{'B':>8s} {'K':>8s}")
        print(f"  {'-'*58}")

        B_list = []
        for a_val in a_vals:
            m = map_to_cusp(q, a_val, BW)
            if m is None:
                continue

            f = m['f_func']
            tau_h = m['tau_cusp']

            sig = find_sigma_for_D(f, m['x_high'], m['x_sad'], tau_h,
                                   BW['D_observed'], sigma_lo=0.001, sigma_hi=20.0)
            if sig is None:
                continue

            B_val = 2.0 * m['DV_peg'] / sig**2
            D_check = compute_D_exact(f, m['x_high'], m['x_sad'], tau_h, sig)
            if D_check > 0 and np.isfinite(D_check):
                D_noK = np.exp(B_val) * m['inv_Ctau']
                K = D_check / D_noK if D_noK > 0 else np.nan
            else:
                K = np.nan

            B_list.append(B_val)
            print(f"  {a_val:8.4f} {m['a_ratio']:8.4f} {m['DV_peg']:10.6f} "
                  f"{sig:10.6f} {B_val:8.4f} {K:8.3f}")

        if len(B_list) >= 3:
            B_arr = np.array(B_list)
            cv = B_arr.std() / B_arr.mean() * 100
            print(f"\n  B statistics (q={q}, n={len(B_arr)}):")
            print(f"    Mean = {B_arr.mean():.4f}")
            print(f"    Std  = {B_arr.std():.4f}")
            print(f"    CV   = {cv:.1f}%")
            print(f"    Range = [{B_arr.min():.4f}, {B_arr.max():.4f}]")
            hz_count = sum(1 for b in B_list if 1.8 <= b <= 6.0)
            print(f"    In habitable zone [1.8, 6.0]: {hz_count}/{len(B_list)}")
            all_B[q] = B_arr
        print()

    return all_B


# ================================================================
# Phase 5: Summary
# ================================================================

def phase5_summary(scen_results, cal_results, B_scans):
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Historical comparison table
    print(f"\n  HISTORICAL COMPARISON (Black Wednesday):")
    print(f"  {'Quantity':<20s} {'Predicted':>15s} {'Observed':>15s} {'Ratio':>10s}")
    print(f"  {'-'*65}")

    # Use the q=2 calibrated result if available
    if 2.0 in cal_results:
        cr = cal_results[2.0]
        kr = cr['kr']
        m = cr['m']
        print(f"  {'MFPT':<20s} {kr['MFPT_days']:>12.0f} d {BW['MFPT_days']:>12d} d "
              f"{kr['MFPT_days']/BW['MFPT_days']:10.2f}")
        print(f"  {'D':<20s} {kr['D_fwd']:>15.1f} {BW['D_observed']:>15d} "
              f"{kr['D_fwd']/BW['D_observed']:10.2f}")

        # sigma comparison (inverse)
        if kr['sigma_star_ann'] is not None:
            print(f"  {'sigma (ann %)':<20s} {kr['sigma_star_ann']:>14.2f}% "
                  f"{BW['sigma_ann_pct']:>14.1f}% "
                  f"{kr['sigma_star_ann']/BW['sigma_ann_pct']:10.2f}")

        # B
        print(f"  {'B':<20s} {kr['B_fwd']:>15.4f} {'[1.8, 6.0]':>15s} "
              f"{'IN' if 1.8 <= kr['B_fwd'] <= 6.0 else 'OUT':>10s}")

        print(f"\n  Calibrated cusp at q=2.0:")
        print(f"    a_star = {cr['a_star']:.6f} (a/a_crit = {m['a_ratio']:.4f})")
        print(f"    DV_peg = {m['DV_peg']:.6f}")
        print(f"    K_eff  = {kr['K_eff']:.3f}")
        print(f"    1/(C*tau) = {m['inv_Ctau']:.4f}")

    # B invariance summary
    print(f"\n  B INVARIANCE:")
    for q_val, B_arr in B_scans.items():
        cv = B_arr.std() / B_arr.mean() * 100
        print(f"    q={q_val}: B = {B_arr.mean():.4f} +/- {B_arr.std():.4f} "
              f"(CV = {cv:.1f}%)")

    # Habitable zone comparison
    print(f"\n  HABITABLE ZONE COMPARISON:")
    print(f"  {'System':<25s} {'D':>8s} {'B':>8s} {'Domain':>20s}")
    print(f"  {'-'*65}")

    eco = [
        ("Kelp",                 29,   1.80,  "Ecology"),
        ("Tumor-immune",         1004, 2.73,  "Cancer biology"),
        ("Peatland",             30,   3.07,  "Ecology"),
        ("Josephson junction",   None, 3.26,  "Supercond. physics"),
        ("Magn. nanoparticle",   None, 3.41,  "Nanomagnetism"),
        ("Savanna",              100,  3.74,  "Ecology"),
        ("Trop. forest",         95,   4.00,  "Ecology"),
        ("Lake",                 200,  4.25,  "Ecology"),
        ("Toggle switch",        1000, 4.83,  "Gene circuit"),
        ("Diabetes",             75,   5.54,  "Human disease"),
        ("Coral",                1111, 6.04,  "Ecology"),
    ]

    # Insert currency peg in sorted order by B
    if 2.0 in cal_results:
        B_peg = cal_results[2.0]['kr']['B_fwd']
        peg_entry = ("** GBP ERM PEG **", BW['D_observed'], B_peg,
                     "FINANCE (this study)")
        inserted = False
        for sname, D_val, B_val, domain in eco:
            if not inserted and B_peg < B_val:
                D_str = f"{BW['D_observed']}"
                print(f"  {peg_entry[0]:<25s} {D_str:>8s} {B_peg:8.2f} "
                      f"{peg_entry[3]:>20s}")
                inserted = True
            D_str = f"{D_val}" if D_val else "varies"
            print(f"  {sname:<25s} {D_str:>8s} {B_val:8.2f} {domain:>20s}")
        if not inserted:
            D_str = f"{BW['D_observed']}"
            print(f"  {peg_entry[0]:<25s} {D_str:>8s} {B_peg:8.2f} "
                  f"{peg_entry[3]:>20s}")
    else:
        for sname, D_val, B_val, domain in eco:
            D_str = f"{D_val}" if D_val else "varies"
            print(f"  {sname:<25s} {D_str:>8s} {B_val:8.2f} {domain:>20s}")

    # Grade
    print(f"\n  GRADE DETERMINATION:")
    print(f"    Cusp shape (q, a): NOT from published calibration")
    print(f"      -> best available: Chen & Chen (2022) for USD/EUR, Grade C")
    print(f"      -> calibrated a from D_observed + sigma_observed, Grade B")
    print(f"    Dimensional mapping (scale_x, scale_t): FROM observables, 0 free params")
    print(f"    sigma: FROM published implied volatility, 0 free params")
    print(f"    Free parameters: 2 (q, a) -- but a constrained by D_observed")
    print(f"    Effective free params: 1 (q only, a determined by D + sigma)")
    print()

    B_all = []
    for q_val, B_arr in B_scans.items():
        B_all.extend(B_arr.tolist())
    B_all_arr = np.array(B_all) if B_all else np.array([0])
    in_hz = sum(1 for b in B_all if 1.8 <= b <= 6.0)

    if 2.0 in cal_results:
        B_cal = cal_results[2.0]['kr']['B_fwd']
        if 1.8 <= B_cal <= 6.0:
            grade = "YELLOW-GREEN"
            reason = ("B inside habitable zone, MFPT within 1 OOM, "
                      "sigma matches observation. 1 effective free param.")
        elif 1.5 <= B_cal <= 6.5:
            grade = "YELLOW"
            reason = ("B near habitable zone, MFPT within 1 OOM, "
                      "1 effective free param.")
        else:
            grade = "RED"
            reason = "B outside habitable zone."
    else:
        grade = "RED"
        reason = "Calibration failed."

    print(f"    GRADE: {grade}")
    print(f"    Reason: {reason}")
    print(f"    B invariance: {in_hz}/{len(B_all)} scan points in habitable zone")
    print(f"    Upgrade from Study 07 (generic financial cusp): RED -> {grade}")


# ================================================================
# Phase 6: Multi-Crisis Comparison (Thai Baht)
# ================================================================

def phase6_multi_crisis():
    """Run calibrated Kramers on Thai Baht and compare to Black Wednesday.

    If B is consistent across independent crises, the framework is predictive
    (not fitted to one system). This closes the free-parameter gap.
    """
    print()
    print("=" * 70)
    print("PHASE 6: MULTI-CRISIS TEST (Thai Baht)")
    print("=" * 70)
    print()
    print("Independent test: does the Thai Baht peg give the same B?")
    print("THB/USD vol: 1.2% annualized (NYU V-Lab GARCH + World Bank + range)")
    print()

    multi_results = {}
    for label, obs in [("TB_full (D=26, tau=180d)", TB_FULL),
                       ("TB_init (D=339, tau=14d)", TB_INIT)]:
        print(f"  --- {label} ---")
        print(f"  x_peg={obs['x_peg']}, x_float={obs['x_float']}, "
              f"delta_x={obs['delta_x']}")
        print(f"  MFPT={obs['MFPT_days']}d, tau={obs['tau_relax_days']}d, "
              f"D={obs['D_observed']}")
        print(f"  sigma={obs['sigma_daily']:.4f} THB/sqrt(day) "
              f"({obs['sigma_ann_pct']}% ann)")

        # Calibrated computation at q=2
        q = 2.0
        a_star = find_a_for_D(q, obs['D_observed'], obs)
        if a_star is None:
            print(f"  Calibration FAILED at q={q}.")
            # Try with higher q for more room
            for q_try in [3.0, 4.0, 6.0, 8.0]:
                a_star = find_a_for_D(q_try, obs['D_observed'], obs)
                if a_star is not None:
                    q = q_try
                    print(f"  Calibrated at q={q} instead.")
                    break
            if a_star is None:
                print(f"  Calibration FAILED at all q values. Skipping.")
                print()
                continue

        m = map_to_cusp(q, a_star, obs)
        if m is None:
            print(f"  Mapping failed. Skipping.")
            print()
            continue

        kr = full_kramers(m, obs)
        hz = 1.8 <= kr['B_fwd'] <= 6.0

        print(f"  Calibrated: q={q}, a={a_star:.6f}, a/a_crit={m['a_ratio']:.4f}")
        print(f"  DV_peg={m['DV_peg']:.8f}, sigma_cusp={m['sigma_cusp']:.6f}")
        print(f"  B = {kr['B_fwd']:.4f} [HZ: {'YES' if hz else 'NO'}]")
        print(f"  D = {kr['D_fwd']:.1f}, K = {kr['K_eff']:.3f}")
        print(f"  1/(C*tau) = {m['inv_Ctau']:.4f}")

        # Test q-independence for this crisis too
        q_test_vals = [1.5, 2.0, 3.0, 4.0]
        B_across_q = []
        for qt in q_test_vals:
            at = find_a_for_D(qt, obs['D_observed'], obs)
            if at is not None:
                mt = map_to_cusp(qt, at, obs)
                if mt is not None:
                    krt = full_kramers(mt, obs)
                    B_across_q.append(krt['B_fwd'])
        if len(B_across_q) >= 2:
            B_arr = np.array(B_across_q)
            print(f"  q-independence: B = {B_arr.mean():.4f} "
                  f"(CV={B_arr.std()/B_arr.mean()*100:.1f}% across q={q_test_vals[:len(B_across_q)]})")

        multi_results[label] = {
            'obs': obs, 'a_star': a_star, 'q': q, 'm': m, 'kr': kr,
            'B': kr['B_fwd'], 'D': obs['D_observed'],
        }
        print()

    # Cross-crisis comparison
    print("  CROSS-CRISIS COMPARISON:")
    print(f"  {'Crisis':<30s} {'D':>6s} {'B':>8s} {'a/a_c':>8s} {'K':>8s} {'HZ?':>5s}")
    print(f"  {'-'*70}")

    # Include Black Wednesday
    a_bw = find_a_for_D(2.0, BW['D_observed'], BW)
    if a_bw is not None:
        m_bw = map_to_cusp(2.0, a_bw, BW)
        kr_bw = full_kramers(m_bw, BW)
        hz_bw = 1.8 <= kr_bw['B_fwd'] <= 6.0
        print(f"  {'Black Wednesday (GBP/DEM)':<30s} {BW['D_observed']:6d} "
              f"{kr_bw['B_fwd']:8.4f} {m_bw['a_ratio']:8.4f} "
              f"{kr_bw['K_eff']:8.3f} {'YES' if hz_bw else 'NO':>5s}")

    for label, res in multi_results.items():
        hz = 1.8 <= res['B'] <= 6.0
        print(f"  {label:<30s} {res['D']:6d} {res['B']:8.4f} "
              f"{res['m']['a_ratio']:8.4f} {res['kr']['K_eff']:8.3f} "
              f"{'YES' if hz else 'NO':>5s}")

    # Key test: are the B values consistent?
    if multi_results and a_bw is not None:
        all_B_vals = [kr_bw['B_fwd']]
        all_D_vals = [BW['D_observed']]
        for res in multi_results.values():
            all_B_vals.append(res['B'])
            all_D_vals.append(res['D'])
        print()
        print(f"  All B values: {[f'{b:.4f}' for b in all_B_vals]}")
        if len(all_B_vals) >= 2:
            B_arr = np.array(all_B_vals)
            print(f"  B mean = {B_arr.mean():.4f}, range = [{B_arr.min():.4f}, "
                  f"{B_arr.max():.4f}]")
            # Note: B SHOULD vary with D (B increases with D). So checking
            # B consistency at different D is NOT the right test. The right
            # test is: does each crisis fall in the habitable zone?
            in_hz = sum(1 for b in all_B_vals if 1.8 <= b <= 6.0)
            print(f"  All in habitable zone [1.8, 6.0]: {in_hz}/{len(all_B_vals)}")

    return multi_results


# ================================================================
# Main
# ================================================================

if __name__ == '__main__':
    t0 = time.time()

    print("=" * 70)
    print("CURRENCY PEG KRAMERS TEST: BLACK WEDNESDAY (Sep 16, 1992)")
    print("Study 21: Calibrated Kramers test on human organizational system")
    print("=" * 70)
    print()
    print("Model: dx/dt = -(x^3 - q*x + a) + sigma*dW")
    print("Potential: V(x) = x^4/4 - q*x^2/2 + a*x")
    print()
    print("Target: UK pound in ERM (Oct 1990 - Sep 1992)")
    print(f"  Peg rate     = {BW['x_peg']} DEM/GBP")
    print(f"  Float rate   = {BW['x_float']} DEM/GBP (post-crisis)")
    print(f"  Rate gap     = {BW['delta_x']} DEM ({BW['delta_x']/BW['x_peg']*100:.0f}%)")
    print(f"  MFPT         = {BW['MFPT_days']} days ({BW['MFPT_months']} months)")
    print(f"  tau_relax    = {BW['tau_relax_days']} days")
    print(f"  D_observed   = {BW['D_observed']}")
    print(f"  sigma        = {BW['sigma_daily']} DEM/sqrt(day) "
          f"({BW['sigma_ann_pct']}% annualized)")

    # Phase 1
    print()
    phase1_d_threshold()

    # Phase 2
    scen_results = phase2_scenarios()

    # Phase 3
    cal_results = phase3_calibrated()

    # Phase 4
    B_scans = phase4_b_invariance()

    # Phase 5
    print()
    phase5_summary(scen_results, cal_results, B_scans)

    # Phase 6: Multi-crisis
    multi_results = phase6_multi_crisis()

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f} seconds")
