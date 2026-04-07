#!/usr/bin/env python3
"""
Study 28: Product Equation Blind Test — Xenopus Cdc2-Cdc25-Wee1 Cell Cycle

BLIND TEST: Product equation D = prod(1/epsilon_i) applied to a non-ecological
bistable system for the first time.

Model: Effective 1D from Trunnell et al. 2011 (Molecular Cell 43:550-560).
  dx/dt = k25*(C-x)*f_25(x) - kw*x*f_w(x)

  x = active Cdk1 (nM), C = total cyclin (nM, bifurcation parameter)
  f_25(x) = b25 + x^n1/(EC50_1^n1 + x^n1)     [Cdc25 fractional activity]
  f_w(x)  = bw + EC50_2^n2/(EC50_2^n2 + x^n2)  [Wee1 fractional activity]

  Channel 1 (Cdc25): positive feedback via dephosphorylation
  Channel 2 (Wee1): double-negative feedback via kinase inhibition

Parameters: ALL from Trunnell et al. 2011.
  n1=11 (measured), n2=3.5 (measured), EC50_1=35nM, EC50_2=30nM,
  b25=0.2, bw=0.2, kw/k25=0.5 (fitted to hysteresis).
Free params: 1 (kw/k25 ratio).

numpy + scipy only. Self-contained. Deterministic.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
import sys

# ================================================================
# PARAMETERS — Trunnell et al. 2011
# ================================================================
N1 = 11.0           # Cdc25 Hill coefficient (measured)
N2 = 3.5            # Wee1 Hill coefficient (measured)
EC1 = 35.0          # Cdc25 EC50 (nM, measured)
EC2 = 30.0          # Wee1 EC50 (nM, measured)
B25 = 0.2           # Cdc25 basal activity (measured)
BW = 0.2            # Wee1 basal activity (measured)
K25 = 1.0           # Cdc25 rate constant (sets timescale)
KW = 0.5            # Wee1 rate constant (kw/k25 = 0.5, fitted)

EC1_N1 = EC1**N1
EC2_N2 = EC2**N2


def f25(x):
    """Cdc25 activity."""
    return B25 + x**N1 / (EC1_N1 + x**N1)


def fw(x):
    """Wee1 activity."""
    return BW + EC2_N2 / (EC2_N2 + x**N2)


def h25(x):
    """Cdc25 feedback Hill function only (0 to 1)."""
    return x**N1 / (EC1_N1 + x**N1)


def hw(x):
    """Wee1 inactivation Hill function (0 to 1)."""
    return x**N2 / (EC2_N2 + x**N2)


def drift(x, C):
    """1D drift: dx/dt."""
    if x <= 0 or x >= C:
        return 0.0
    return K25 * (C - x) * f25(x) - KW * x * fw(x)


def drift_arr(xg, C):
    """Vectorized drift on a grid."""
    xg = np.asarray(xg, dtype=float)
    xg_safe = np.clip(xg, 1e-10, C - 1e-10)
    xn1 = xg_safe**N1
    xn2 = xg_safe**N2
    f25v = B25 + xn1 / (EC1_N1 + xn1)
    fwv = BW + EC2_N2 / (EC2_N2 + xn2)
    return K25 * (C - xg_safe) * f25v - KW * xg_safe * fwv


def find_roots(C, N=20000):
    """Find all equilibria."""
    xg = np.linspace(0.1, C - 0.1, N)
    fg = drift_arr(xg, C)
    sc = np.where(np.diff(np.sign(fg)))[0]
    roots = []
    for i in sc:
        try:
            r = brentq(lambda x: drift(x, C), xg[i], xg[i+1], xtol=1e-12)
            roots.append(r)
        except (ValueError, RuntimeError):
            pass
    return sorted(roots)


def deriv(x, C, dx=1e-7):
    """Numerical derivative of drift."""
    return (drift(x + dx, C) - drift(x - dx, C)) / (2 * dx)


def classify(roots, C):
    """Return (stable_list, unstable_list)."""
    stable, unstable = [], []
    for r in roots:
        if deriv(r, C) < 0:
            stable.append(r)
        else:
            unstable.append(r)
    return stable, unstable


def compute_D_exact(C, x_eq, x_sad, tau, sigma, N=80000):
    """Exact 1D MFPT -> D."""
    margin = max(0.3 * abs(x_sad - x_eq), 2.0)
    if x_eq < x_sad:
        xlo = max(x_eq - margin, 0.1)
        xhi = min(x_sad + 0.5, C - 0.1)
    else:
        xlo = max(x_sad - 0.5, 0.1)
        xhi = min(x_eq + margin, C - 0.1)

    xg = np.linspace(xlo, xhi, N)
    dx = xg[1] - xg[0]
    fg = drift_arr(xg, C)
    U = np.cumsum(-fg) * dx
    i_eq = np.argmin(np.abs(xg - x_eq))
    U -= U[i_eq]
    Phi = np.clip(2.0 * U / sigma**2, -500, 500)

    if x_eq < x_sad:
        Ix = np.cumsum(np.exp(-Phi)) * dx
        psi = (2.0 / sigma**2) * np.exp(Phi) * Ix
        i_sad = np.argmin(np.abs(xg - x_sad))
        MFPT = np.trapz(psi[i_eq:i_sad+1], xg[i_eq:i_sad+1])
    else:
        Ix = np.cumsum(np.exp(-Phi)[::-1])[::-1] * dx
        psi = (2.0 / sigma**2) * np.exp(Phi) * Ix
        i_sad = np.argmin(np.abs(xg - x_sad))
        lo, hi = min(i_sad, i_eq), max(i_sad, i_eq) + 1
        MFPT = np.trapz(psi[lo:hi], xg[lo:hi])

    return MFPT / tau


def find_sigma_star(C, x_eq, x_sad, tau, D_target):
    """Find sigma where D_exact = D_target."""
    def obj(ls):
        s = np.exp(ls)
        try:
            D = compute_D_exact(C, x_eq, x_sad, tau, s)
            if D <= 0 or not np.isfinite(D):
                return -1e10
            return np.log(D) - np.log(D_target)
        except Exception:
            return -1e10

    log_sigmas = np.linspace(np.log(0.05), np.log(200), 80)
    vals = [obj(ls) for ls in log_sigmas]

    for i in range(len(vals)-1):
        if vals[i] * vals[i+1] < 0 and abs(vals[i]) < 50 and abs(vals[i+1]) < 50:
            try:
                return np.exp(brentq(obj, log_sigmas[i], log_sigmas[i+1], xtol=1e-10))
            except (ValueError, RuntimeError):
                continue
    return None


def p(s):
    """Print and flush."""
    print(s, flush=True)


# ================================================================
# MAIN
# ================================================================
def main():
    p("=" * 72)
    p("STUDY 28: PRODUCT EQUATION BLIND TEST")
    p("Xenopus Cdc2-Cdc25-Wee1 Cell Cycle Trigger")
    p("=" * 72)
    p("")

    # ---- PHASE 1: BISTABILITY ----
    p("=" * 72)
    p("PHASE 1: BISTABILITY VERIFICATION")
    p("=" * 72)
    p("")
    p("Parameters (Trunnell et al. 2011, Mol Cell 43:550-560):")
    p(f"  n_Cdc25={N1}, n_Wee1={N2}, EC50_Cdc25={EC1}nM, EC50_Wee1={EC2}nM")
    p(f"  bkgd_Cdc25={B25}, bkgd_Wee1={BW}, kw/k25={KW/K25}")
    p("")

    # Find bistable range
    p("Scanning Cyclin_tot...")
    bistable = {}
    for C in np.linspace(30, 180, 150):
        roots = find_roots(C)
        st, un = classify(roots, C)
        if len(st) >= 2 and len(un) >= 1:
            bistable[C] = (st[0], un[0], st[-1])

    if not bistable:
        p("*** NO BISTABILITY FOUND ***")
        return

    Cs = sorted(bistable.keys())
    p(f"  Bistable range: C = [{Cs[0]:.1f}, {Cs[-1]:.1f}] nM ({len(Cs)} points)")
    p("")

    # Representative C
    C = Cs[len(Cs)//2]
    xl, xs, xh = bistable[C]

    p(f"Representative C = {C:.2f} nM")
    p(f"  x_low (interphase)  = {xl:.6f} nM")
    p(f"  x_saddle            = {xs:.6f} nM")
    p(f"  x_high (M-phase)    = {xh:.6f} nM")

    # Eigenvalues
    lam_l = deriv(xl, C)
    lam_h = deriv(xh, C)
    lam_s = deriv(xs, C)
    tau_l = 1.0 / abs(lam_l)
    tau_h = 1.0 / abs(lam_h)

    p(f"  lam_low={lam_l:.6f} (tau={tau_l:.6f})")
    p(f"  lam_high={lam_h:.6f} (tau={tau_h:.6f})")
    p(f"  lam_saddle={lam_s:+.6f}")
    p("")

    # Barriers
    DPhi_h, _ = quad(lambda x: -drift(x, C), xh, xs)
    DPhi_l, _ = quad(lambda x: -drift(x, C), xl, xs)
    p(f"  DPhi (high->saddle) = {DPhi_h:.8f}")
    p(f"  DPhi (low->saddle)  = {DPhi_l:.8f}")
    p("")

    # ---- PHASE 2: EPSILON ----
    p("=" * 72)
    p("PHASE 2: EPSILON COMPUTATION")
    p("=" * 72)
    p("")

    # Store per-state results
    state_results = {}

    for label, xeq, x_sad_local, dphi, tau in [
        ("HIGH (M-phase)", xh, xs, DPhi_h, tau_h),
        ("LOW (interphase)", xl, xs, DPhi_l, tau_l),
    ]:
        p(f"--- {label}: x_eq = {xeq:.6f} ---")
        A_tot = K25 * (C - xeq) * f25(xeq)
        I_tot = KW * xeq * fw(xeq)
        A_bas = K25 * (C - xeq) * B25
        I_bas = KW * xeq * (BW + 1.0)
        R_c = K25 * (C - xeq) * h25(xeq)
        R_w = KW * xeq * hw(xeq)
        f_bas = A_bas - I_bas

        p(f"  A_total={A_tot:.6f}  I_total={I_tot:.6f}  A-I={A_tot-I_tot:.2e}")
        p(f"  A_basal={A_bas:.6f}  I_basal(no fb)={I_bas:.6f}")
        p(f"  f_baseline(no fb)={f_bas:.6f}")
        p(f"  R_Cdc25(fb)={R_c:.6f}  R_Wee1(fb)={R_w:.6f}")
        p(f"  R_total={R_c+R_w:.6f}  |f_baseline|={abs(f_bas):.6f}")
        p("")

        sr = {'label': label, 'xeq': xeq, 'x_sad': x_sad_local,
              'dphi': dphi, 'tau': tau, 'defs': {}}

        # Def A: flux fraction
        Rtot = R_c + R_w
        if Rtot > 0:
            eA1, eA2 = R_c / Rtot, R_w / Rtot
            DA = 1.0 / (eA1 * eA2)
            sr['defs']['A (flux frac)'] = (eA1, eA2, DA)
            p(f"  Def A (flux frac): eps_Cdc25={eA1:.6f} eps_Wee1={eA2:.6f} D={DA:.4f}")

        # Def B: saddle-normalized
        A_sad = K25 * (C - xs) * f25(xs)
        if A_sad > 0:
            eB1, eB2 = R_c / A_sad, R_w / A_sad
            DB = 1.0 / (eB1 * eB2)
            sr['defs']['B (saddle-norm)'] = (eB1, eB2, DB)
            p(f"  Def B (saddle-norm): eps_Cdc25={eB1:.6f} eps_Wee1={eB2:.6f} D={DB:.4f}")

        # Def B2: normalized by total flux at equilibrium
        if A_tot > 0:
            eB2_1, eB2_2 = R_c / A_tot, R_w / A_tot
            DB2 = 1.0 / (eB2_1 * eB2_2)
            sr['defs']['B2 (eq-norm)'] = (eB2_1, eB2_2, DB2)
            p(f"  Def B2 (eq-norm): eps_Cdc25={eB2_1:.6f} eps_Wee1={eB2_2:.6f} D={DB2:.4f}")

        state_results[label] = sr
        p("")

    # Def C: perturbation robustness
    p("Def C (perturbation robustness):")
    p("  Finding critical Cdc25 reduction...")

    # Binary search: reduce Cdc25 feedback by factor alpha
    def bistable_reduced_cdc25(alpha, C_val):
        def f(x):
            return K25*(C_val-x)*(B25+(1-alpha)*h25(x)) - KW*x*fw(x)
        xg = np.linspace(0.1, C_val-0.1, 10000)
        fg = np.array([f(xi) for xi in xg])
        sc = np.where(np.diff(np.sign(fg)))[0]
        # Count stable roots
        n_roots = 0
        for i in sc:
            try:
                r = brentq(f, xg[i], xg[i+1])
                fp = (f(r+1e-7)-f(r-1e-7))/2e-7
                if fp < 0:
                    n_roots += 1
            except:
                pass
        return n_roots >= 2

    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo+hi)/2
        if bistable_reduced_cdc25(mid, C):
            lo = mid
        else:
            hi = mid
    alpha_c25 = (lo+hi)/2
    p(f"    Cdc25 critical: alpha={alpha_c25:.6f} ({alpha_c25*100:.1f}% reduction)")

    # Same for Wee1
    p("  Finding critical Wee1 reduction...")
    def bistable_reduced_wee1(alpha, C_val):
        def f(x):
            # Reduce Wee1 feedback: Wee1 stays more active
            G = EC2_N2 / (EC2_N2 + x**N2)
            fw_red = (BW + alpha) + G*(1-alpha)
            return K25*(C_val-x)*f25(x) - KW*x*fw_red
        xg = np.linspace(0.1, C_val-0.1, 10000)
        fg = np.array([f(xi) for xi in xg])
        sc = np.where(np.diff(np.sign(fg)))[0]
        n_roots = 0
        for i in sc:
            try:
                r = brentq(f, xg[i], xg[i+1])
                fp = (f(r+1e-7)-f(r-1e-7))/2e-7
                if fp < 0:
                    n_roots += 1
            except:
                pass
        return n_roots >= 2

    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo+hi)/2
        if bistable_reduced_wee1(mid, C):
            lo = mid
        else:
            hi = mid
    alpha_w = (lo+hi)/2
    p(f"    Wee1 critical:  alpha={alpha_w:.6f} ({alpha_w*100:.1f}% reduction)")

    eps_C1 = 1.0 - alpha_c25
    eps_C2 = 1.0 - alpha_w
    DC = 1.0 / (eps_C1 * eps_C2) if eps_C1 > 0 and eps_C2 > 0 else None
    p(f"    eps_Cdc25(residual)={eps_C1:.6f} eps_Wee1(residual)={eps_C2:.6f}")
    p(f"    D_product(perturbation) = {DC:.4f}" if DC else "    D_product: UNDEFINED")
    p("")

    # ---- PHASE 3: KRAMERS MFPT ----
    p("=" * 72)
    p("PHASE 3: KRAMERS MFPT")
    p("=" * 72)
    p("")

    p("D_exact(sigma) for HIGH state:")
    p(f"  {'sigma':>8s} {'D_exact':>12s} {'B=2DPhi/s2':>12s}")
    p(f"  {'---':>8s} {'---':>12s} {'---':>12s}")

    for sig in [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 80.0]:
        try:
            D = compute_D_exact(C, xh, xs, tau_h, sig)
            B = 2*DPhi_h/sig**2
            p(f"  {sig:8.1f} {D:12.4f} {B:12.4f}")
        except Exception as e:
            p(f"  {sig:8.1f} {'ERR':>12s} {str(e)[:20]}")
    p("")

    p("D_exact(sigma) for LOW state:")
    p(f"  {'sigma':>8s} {'D_exact':>12s} {'B=2DPhi/s2':>12s}")
    p(f"  {'---':>8s} {'---':>12s} {'---':>12s}")

    for sig in [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 80.0]:
        try:
            D = compute_D_exact(C, xl, xs, tau_l, sig)
            B = 2*DPhi_l/sig**2
            p(f"  {sig:8.1f} {D:12.4f} {B:12.4f}")
        except Exception as e:
            p(f"  {sig:8.1f} {'ERR':>12s} {str(e)[:20]}")
    p("")

    # ---- PHASE 4: BRIDGE TEST ----
    p("=" * 72)
    p("PHASE 4: BRIDGE TEST (each state with its own Kramers)")
    p("=" * 72)
    p("")

    for state_label, sr in state_results.items():
        p(f"--- {state_label} ---")
        xeq = sr['xeq']
        xsd = sr['x_sad']
        dphi = sr['dphi']
        tau = sr['tau']

        for def_label, (e1, e2, Dp) in sr['defs'].items():
            if Dp is None or Dp <= 1.0:
                p(f"  Def {def_label}: D={Dp} — skipped")
                continue
            p(f"  Def {def_label}: D_product = {Dp:.4f} (eps={e1:.4f},{e2:.4f})")
            ss = find_sigma_star(C, xeq, xsd, tau, Dp)
            if ss is not None:
                Dcheck = compute_D_exact(C, xeq, xsd, tau, ss)
                Bval = 2*dphi/ss**2
                p(f"    sigma*     = {ss:.6f} nM")
                p(f"    D_exact    = {Dcheck:.4f}")
                p(f"    Ratio      = {Dcheck/Dp:.6f}")
                p(f"    B          = {Bval:.4f}")
                p(f"    CV at eq   = {ss/xeq:.4f}")
                if 1.8 <= Bval <= 6.0:
                    p(f"    B in [1.8, 6.0]: YES")
                else:
                    p(f"    B in [1.8, 6.0]: NO")
            else:
                p(f"    sigma* NOT FOUND")
            p("")

        # Also test perturbation definition for this state
        if DC is not None and DC > 1:
            p(f"  Def C (perturbation): D_product = {DC:.4f}")
            ss = find_sigma_star(C, xeq, xsd, tau, DC)
            if ss is not None:
                Dcheck = compute_D_exact(C, xeq, xsd, tau, ss)
                Bval = 2*dphi/ss**2
                p(f"    sigma*     = {ss:.6f} nM")
                p(f"    D_exact    = {Dcheck:.4f}")
                p(f"    Ratio      = {Dcheck/DC:.6f}")
                p(f"    B          = {Bval:.4f}")
                p(f"    CV at eq   = {ss/xeq:.4f}")
                if 1.8 <= Bval <= 6.0:
                    p(f"    B in [1.8, 6.0]: YES")
                else:
                    p(f"    B in [1.8, 6.0]: NO")
            p("")

    # ---- PHASE 5: B INVARIANCE ----
    p("=" * 72)
    p("PHASE 5: B INVARIANCE (flux-fraction definition)")
    p("=" * 72)
    p("")

    B_vals = []
    D_vals = []
    p(f"  {'C':>6s} {'x_high':>8s} {'DPhi':>10s} {'D_prod':>8s} {'sigma*':>10s} {'B':>8s}")
    p(f"  {'---':>6s} {'---':>8s} {'---':>10s} {'---':>8s} {'---':>10s} {'---':>8s}")

    sample_Cs = np.linspace(Cs[1], Cs[-2], min(15, len(Cs)-2))
    for Ci in sample_Cs:
        roots = find_roots(Ci)
        st, un = classify(roots, Ci)
        if len(st) < 2 or len(un) < 1:
            continue
        xi_h, xi_s = st[-1], un[0]
        li_h = deriv(xi_h, Ci)
        ti_h = 1.0 / abs(li_h)
        dp_i, _ = quad(lambda x: -drift(x, Ci), xi_h, xi_s)

        Rc = K25*(Ci-xi_h)*h25(xi_h)
        Rw = KW*xi_h*hw(xi_h)
        Rt = Rc+Rw
        if Rt <= 0:
            continue
        e1, e2 = Rc/Rt, Rw/Rt
        Di = 1.0/(e1*e2)
        if Di <= 1:
            continue

        ss_i = find_sigma_star(Ci, xi_h, xi_s, ti_h, Di)
        if ss_i is not None:
            Bi = 2*dp_i/ss_i**2
            B_vals.append(Bi)
            D_vals.append(Di)
            p(f"  {Ci:6.1f} {xi_h:8.4f} {dp_i:10.6f} {Di:8.2f} {ss_i:10.6f} {Bi:8.4f}")
    p("")

    if len(B_vals) >= 2:
        Barr = np.array(B_vals)
        p(f"  B invariance: N={len(Barr)}, mean={np.mean(Barr):.4f}, "
          f"std={np.std(Barr):.4f}, CV={np.std(Barr)/np.mean(Barr)*100:.2f}%")
        p(f"  Range: [{np.min(Barr):.4f}, {np.max(Barr):.4f}]")
        n_in = sum(1 for b in Barr if 1.8 <= b <= 6.0)
        p(f"  In [1.8, 6.0]: {n_in}/{len(Barr)}")
    p("")

    # ---- PHASE 5b: B INVARIANCE for LOW state ----
    p("  --- B INVARIANCE for LOW state (flux-fraction Def A) ---")
    p("")
    B_low_vals = []
    p(f"  {'C':>6s} {'x_low':>8s} {'DPhi':>10s} {'D_prod':>8s} {'sigma*':>10s} {'B':>8s}")
    p(f"  {'---':>6s} {'---':>8s} {'---':>10s} {'---':>8s} {'---':>10s} {'---':>8s}")

    for Ci in sample_Cs:
        roots = find_roots(Ci)
        st, un = classify(roots, Ci)
        if len(st) < 2 or len(un) < 1:
            continue
        xi_l, xi_s = st[0], un[0]
        li_l = deriv(xi_l, Ci)
        ti_l = 1.0 / abs(li_l)
        dp_i, _ = quad(lambda x: -drift(x, Ci), xi_l, xi_s)

        Rc = K25*(Ci-xi_l)*h25(xi_l)
        Rw = KW*xi_l*hw(xi_l)
        Rt = Rc+Rw
        if Rt <= 0:
            continue
        e1, e2 = Rc/Rt, Rw/Rt
        Di = 1.0/(e1*e2)
        if Di <= 1:
            continue

        ss_i = find_sigma_star(Ci, xi_l, xi_s, ti_l, Di)
        if ss_i is not None:
            Bi = 2*dp_i/ss_i**2
            B_low_vals.append(Bi)
            p(f"  {Ci:6.1f} {xi_l:8.4f} {dp_i:10.6f} {Di:8.2f} {ss_i:10.6f} {Bi:8.4f}")
    p("")

    if len(B_low_vals) >= 2:
        Barr = np.array(B_low_vals)
        p(f"  LOW state B invariance: N={len(Barr)}, mean={np.mean(Barr):.4f}, "
          f"std={np.std(Barr):.4f}, CV={np.std(Barr)/np.mean(Barr)*100:.2f}%")
        p(f"  Range: [{np.min(Barr):.4f}, {np.max(Barr):.4f}]")
        n_in = sum(1 for b in Barr if 1.8 <= b <= 6.0)
        p(f"  In [1.8, 6.0]: {n_in}/{len(Barr)}")
    p("")

    # ---- SUMMARY ----
    p("=" * 72)
    p("SUMMARY")
    p("=" * 72)
    p("")
    p(f"System:  Xenopus Cdc2-Cdc25-Wee1 (cell cycle M-phase trigger)")
    p(f"Model:   Trunnell et al. 2011, Mol Cell 43:550-560")
    p(f"Domain:  Cell biology (non-ecological)")
    p(f"Free params: 1 (kw/k25={KW/K25})")
    p(f"Bistable range: C=[{Cs[0]:.1f}, {Cs[-1]:.1f}] nM")
    p(f"Representative C={C:.1f}: x_low={xl:.2f}, x_sad={xs:.2f}, x_high={xh:.2f}")
    p(f"DPhi_high={DPhi_h:.6f}, DPhi_low={DPhi_l:.6f}")
    p("")
    p("Epsilon definitions tested:")
    for state_label, sr in state_results.items():
        p(f"  [{state_label}]")
        for def_label, (e1, e2, D) in sr['defs'].items():
            p(f"    {def_label}: eps={e1:.4f},{e2:.4f}  D={D:.2f}")
    if DC is not None:
        p(f"  [Perturbation (system-level)]")
        p(f"    C: eps={eps_C1:.4f},{eps_C2:.4f}  D={DC:.2f}")


if __name__ == '__main__':
    main()
