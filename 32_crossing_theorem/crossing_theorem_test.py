#!/usr/bin/env python3
"""
CROSSING THEOREM TEST
=====================
sigma* = sigma_process as topological necessity via IVT.
6 systems: Lake, Kelp, Savanna, Coral, Josephson Junction, Nanoparticle.
N=50 points per system. 80k grid (ecology), 200k grid (physics).
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import warnings
import time

warnings.filterwarnings('ignore')

N_PTS = 50
N_GRID = 80000
N_GRID_PHYS = 200000

# ===========================================================================
# MODEL 1: LAKE (q=8, D=200)
# ===========================================================================
LAKE_BL = 0.8; LAKE_R = 1.0; LAKE_H = 1.0; LAKE_Q = 8; LAKE_D = 200

def lake_f(x, a):
    return a - LAKE_BL*x + LAKE_R*x**LAKE_Q/(x**LAKE_Q + LAKE_H**LAKE_Q)

def lake_fv(x, a):
    return a - LAKE_BL*x + LAKE_R*x**LAKE_Q/(x**LAKE_Q + LAKE_H**LAKE_Q)

def lake_fp(x, a):
    q = LAKE_Q
    return -LAKE_BL + LAKE_R*q*x**(q-1)*LAKE_H**q/(x**q + LAKE_H**q)**2

def lake_roots(a):
    xs = np.linspace(0.001, 4.0, 5000)
    fs = lake_fv(xs, a)
    roots = []
    for i in np.where(fs[:-1]*fs[1:] < 0)[0]:
        try:
            r = brentq(lambda x: lake_f(x, a), xs[i], xs[i+1], xtol=1e-14)
            if not any(abs(r - rr) < 1e-8 for rr in roots):
                roots.append(r)
        except Exception:
            pass
    return sorted(roots)

def lake_bistable():
    bi = []
    for a in np.linspace(0.01, 0.8, 1000):
        xs = np.linspace(0.001, 4.0, 3000)
        fs = lake_fv(xs, a)
        if np.sum(fs[:-1]*fs[1:] < 0) == 3:
            bi.append(a)
    return (bi[0], bi[-1]) if len(bi) >= 2 else (None, None)

def lake_barrier(a, x_eq, x_sad):
    return quad(lambda x: -lake_f(x, a), x_eq, x_sad,
                limit=200, epsabs=1e-14, epsrel=1e-12)[0]

def lake_D_exact(a, x_eq, x_sad, tau, sigma):
    m = 0.05*(x_sad - x_eq)
    xg = np.linspace(max(0.001, x_eq - m), x_sad + m, N_GRID)
    dx = xg[1] - xg[0]
    U = np.cumsum(-lake_fv(xg, a))*dx
    ie = np.argmin(np.abs(xg - x_eq))
    U -= U[ie]
    Phi = 2*U/sigma**2
    if Phi.max() > 700:
        return np.inf
    Phi = np.clip(Phi, -500, 700)
    Ix = np.cumsum(np.exp(-Phi))*dx
    psi = (2/sigma**2)*np.exp(Phi)*Ix
    isd = np.argmin(np.abs(xg - x_sad))
    lo, hi = min(ie, isd), max(ie, isd)
    return np.trapz(psi[lo:hi+1], xg[lo:hi+1])/tau

def lake_sigma_star(a, x_eq, x_sad, lam_eq):
    tau = 1.0/abs(lam_eq)
    DP = lake_barrier(a, x_eq, x_sad)
    ls = lake_fp(x_sad, a)
    C = abs(ls)/(2*np.pi)*np.sqrt(abs(lam_eq)/abs(ls))
    arg = LAKE_D*C*tau
    sg = np.sqrt(2*DP/np.log(arg)) if arg > 1 and DP > 0 else 0.1
    lg = np.log(max(sg, 1e-6))

    def obj(lsig):
        s = np.exp(lsig)
        D = lake_D_exact(a, x_eq, x_sad, tau, s)
        if D == np.inf or D > 1e15:
            return 50.0
        if D <= 0:
            return -50.0
        return np.log(max(D, 1e-30)) - np.log(LAKE_D)

    for pts in [np.linspace(lg - 3, lg + 3, 40),
                np.linspace(np.log(1e-4), np.log(2), 100)]:
        ov = [obj(p) for p in pts]
        for i in range(len(ov) - 1):
            if ov[i] > 0 and ov[i+1] <= 0:
                try:
                    return np.exp(brentq(obj, pts[i], pts[i+1],
                                         xtol=1e-10, maxiter=300))
                except Exception:
                    pass
    return np.nan

def lake_compute(a):
    rr = lake_roots(a)
    if len(rr) < 3:
        return None
    xe, xs = rr[0], rr[1]
    le = lake_fp(xe, a)
    DP = lake_barrier(a, xe, xs)
    if DP < 1e-10:
        return None
    ss = lake_sigma_star(a, xe, xs, le)
    if np.isnan(ss):
        return None
    return {'param': a, 'sigma_star': ss, 'DeltaPhi': DP,
            'lambda_eq': abs(le), 'B': 2*DP/ss**2}


# ===========================================================================
# MODEL 2: KELP (D=29.4)
# ===========================================================================
KELP_r = 0.4; KELP_K = 668.0; KELP_h = 100.0; KELP_D = 29.4

def kelp_f(U, p):
    return KELP_r*U*(1 - U/KELP_K) - p*U/(U + KELP_h)

def kelp_fp(U, p):
    return KELP_r*(1 - 2*U/KELP_K) - p*KELP_h/(U + KELP_h)**2

def kelp_nontrivial(p):
    a_c = KELP_r
    b_c = KELP_r*KELP_h - KELP_r*KELP_K
    c_c = KELP_K*(p - KELP_r*KELP_h)
    disc = b_c**2 - 4*a_c*c_c
    if disc < 0:
        return None
    sd = np.sqrt(disc)
    U1 = (-b_c - sd)/(2*a_c)
    U2 = (-b_c + sd)/(2*a_c)
    if U1 <= 0 or U2 <= 0:
        return None
    return (U1, U2)

def kelp_barrier(p, U_sad):
    return quad(lambda U: -kelp_f(U, p), 0, U_sad)[0]

def kelp_D_exact(sigma, p, U_sad, lam_eq):
    tau = 1.0/abs(lam_eq)
    xg = np.linspace(1e-6, U_sad, N_GRID)
    dx = xg[1] - xg[0]
    fv = KELP_r*xg*(1 - xg/KELP_K) - p*xg/(xg + KELP_h)
    V = np.cumsum(-fv)*dx
    Phi = 2.0*V/sigma**2
    Phi -= Phi[0]
    if Phi.max() > 600:
        return np.inf
    Ix = np.cumsum(np.exp(-Phi))*dx
    psi = (2.0/sigma**2)*np.exp(Phi)*Ix
    return np.trapz(psi, xg)/tau

def kelp_sigma_star(p, U_sad, lam_eq):
    def obj(ls):
        s = np.exp(ls)
        D = kelp_D_exact(s, p, U_sad, lam_eq)
        if D == np.inf:
            return 50.0
        if D <= 0:
            return -50.0
        return np.log(max(D, 1e-30)) - np.log(KELP_D)
    try:
        return np.exp(brentq(obj, np.log(0.01), np.log(100.0),
                             xtol=1e-10))
    except ValueError:
        return np.nan

def kelp_compute(p):
    eq = kelp_nontrivial(p)
    if eq is None:
        return None
    U_sad, U_bar = eq
    le = kelp_fp(0, p)
    ls = kelp_fp(U_sad, p)
    if le >= 0 or ls <= 0:
        return None
    DP = kelp_barrier(p, U_sad)
    if DP < 1e-10:
        return None
    ss = kelp_sigma_star(p, U_sad, le)
    if np.isnan(ss):
        return None
    return {'param': p, 'sigma_star': ss, 'DeltaPhi': DP,
            'lambda_eq': abs(le), 'B': 2*DP/ss**2}

def kelp_bistable():
    p_min = KELP_r * KELP_h
    pts = np.linspace(p_min + 0.1, 200.0, 5000)
    bi = []
    for pv in pts:
        eq = kelp_nontrivial(pv)
        if eq is not None:
            Us, Ub = eq
            if kelp_fp(Us, pv) > 0 and kelp_fp(Ub, pv) < 0:
                bi.append(pv)
    return (bi[0], bi[-1]) if len(bi) >= 2 else (None, None)


# ===========================================================================
# MODEL 3: SAVANNA (Staver-Levin, adiabatic)
# ===========================================================================
SAV_mu = 0.2; SAV_nu = 0.1; SAV_omega0 = 0.9; SAV_omega1 = 0.2
SAV_theta1 = 0.4; SAV_ss1 = 0.01; SAV_D = 100.0

def sav_omega(G):
    return SAV_omega0 + (SAV_omega1 - SAV_omega0) / \
           (1 + np.exp(-(G - SAV_theta1)/SAV_ss1))

def sav_G_null(T, beta):
    num = SAV_mu - T*(SAV_mu - SAV_nu)
    den = SAV_mu + beta*T
    if den <= 0 or num <= 0:
        return np.nan
    G = num/den
    if G <= 0 or G >= 1 or G + T >= 1:
        return np.nan
    return G

def sav_f(T, beta):
    G = sav_G_null(T, beta)
    if np.isnan(G):
        return 0.0
    S = 1 - G - T
    if S < 0:
        return 0.0
    return sav_omega(G)*S - SAV_nu*T

def sav_fp(T, beta, dT=1e-7):
    return (sav_f(T + dT, beta) - sav_f(T - dT, beta))/(2*dT)

def sav_f_vec(T_arr, beta):
    num = SAV_mu - T_arr*(SAV_mu - SAV_nu)
    den = SAV_mu + beta*T_arr
    ok = (den > 0) & (num > 0)
    G = np.where(ok, num/den, 0.5)
    ok &= (G > 0) & (G < 1) & (G + T_arr < 1)
    S = np.where(ok, 1 - G - T_arr, 0)
    ok &= (S >= 0)
    ea = np.clip(-(G - SAV_theta1)/SAV_ss1, -500, 500)
    w = SAV_omega0 + (SAV_omega1 - SAV_omega0)/(1 + np.exp(ea))
    return np.where(ok, w*S - SAV_nu*T_arr, 0.0)

def sav_find_eq(beta):
    T_scan = np.linspace(0.01, 0.95, 50000)
    fv = sav_f_vec(T_scan, beta)
    roots = []
    for i in range(len(fv) - 1):
        if fv[i]*fv[i+1] < 0:
            try:
                roots.append(brentq(lambda T: sav_f(T, beta),
                                    T_scan[i], T_scan[i+1]))
            except ValueError:
                pass
    return sorted(roots)

def sav_barrier(beta, T_eq, T_sad):
    return quad(lambda T: -sav_f(T, beta), T_eq, T_sad,
                limit=200, epsabs=1e-14, epsrel=1e-12)[0]

def sav_D_exact(sigma, beta, T_eq, T_sad, lam_eq):
    tau = 1.0/abs(lam_eq)
    T_lo = max(0.001, T_eq - 3*sigma/np.sqrt(2*abs(lam_eq)))
    T_hi = T_sad + 0.001
    if T_lo >= T_hi:
        return 1e10
    Tg = np.linspace(T_lo, T_hi, N_GRID)
    dT = Tg[1] - Tg[0]
    neg_f = -sav_f_vec(Tg, beta)
    U = np.cumsum(neg_f)*dT
    ie = np.argmin(np.abs(Tg - T_eq))
    U -= U[ie]
    Phi = 2*U/sigma**2
    Phi = np.clip(Phi, -500, 500)
    Ix = np.cumsum(np.exp(-Phi))*dT
    psi = (2/sigma**2)*np.exp(Phi)*Ix
    isd = np.argmin(np.abs(Tg - T_sad))
    if ie >= isd:
        return 1e10
    return np.trapz(psi[ie:isd+1], Tg[ie:isd+1])/tau

def sav_sigma_star(beta, T_eq, T_sad, lam_eq):
    def obj(ls):
        return np.log(sav_D_exact(np.exp(ls), beta, T_eq, T_sad, lam_eq)) \
               - np.log(SAV_D)
    try:
        return np.exp(brentq(obj, np.log(0.0001), np.log(0.5), xtol=1e-10))
    except ValueError:
        return np.nan

def sav_bistable():
    def has_bi(beta):
        roots = sav_find_eq(beta)
        if len(roots) < 3:
            return False
        stabs = [(r, sav_fp(r, beta)) for r in roots]
        return (sum(1 for _, fp in stabs if fp < 0) >= 2 and
                sum(1 for _, fp in stabs if fp > 0) >= 1)
    btest = np.linspace(0.20, 0.60, 200)
    bi = [b for b in btest if has_bi(b)]
    return (bi[0], bi[-1]) if len(bi) >= 2 else (None, None)

def sav_compute(beta):
    roots = sav_find_eq(beta)
    if len(roots) < 3:
        return None
    stabs = [(r, sav_fp(r, beta)) for r in roots]
    stable = sorted([r for r, fp in stabs if fp < 0])
    unstable = sorted([r for r, fp in stabs if fp > 0])
    if len(stable) < 2 or len(unstable) < 1:
        return None
    T_eq, T_sad = stable[0], unstable[0]
    le = sav_fp(T_eq, beta)
    if le >= 0:
        return None
    DP = sav_barrier(beta, T_eq, T_sad)
    if DP < 1e-10:
        return None
    ss = sav_sigma_star(beta, T_eq, T_sad, le)
    if np.isnan(ss):
        return None
    return {'param': beta, 'sigma_star': ss, 'DeltaPhi': DP,
            'lambda_eq': abs(le), 'B': 2*DP/ss**2}


# ===========================================================================
# MODEL 4: CORAL (Mumby 2007, adiabatic)
# ===========================================================================
CORAL_a = 0.1; CORAL_gam = 0.8; CORAL_r = 1.0; CORAL_d = 0.44
CORAL_D = 1111.1
CORAL_Cb = 1 - CORAL_d/CORAL_r
CORAL_Tb = CORAL_d/CORAL_r
CORAL_glo = CORAL_Tb*(CORAL_a*CORAL_Cb + CORAL_gam*CORAL_Tb)
CORAL_ghi = (CORAL_d + CORAL_a)*CORAL_gam/(CORAL_r + CORAL_a)

def coral_f(M, g):
    C = max(1 - M - (CORAL_d + CORAL_a*M)/CORAL_r, 0.0)
    T = (CORAL_d + CORAL_a*M)/CORAL_r
    dn = M + T
    if dn < 1e-30:
        return 0.0
    return CORAL_a*M*C - g*M/dn + CORAL_gam*M*T

def coral_fv(M_arr, g):
    C = np.maximum(1 - M_arr - (CORAL_d + CORAL_a*M_arr)/CORAL_r, 0.0)
    T = (CORAL_d + CORAL_a*M_arr)/CORAL_r
    dn = np.maximum(M_arr + T, 1e-30)
    return CORAL_a*M_arr*C - g*M_arr/dn + CORAL_gam*M_arr*T

def coral_fp(M, g, dM=1e-7):
    return (coral_f(M + dM, g) - coral_f(M - dM, g))/(2*dM)

def coral_find_eq(g):
    M_alg = 1 - g/CORAL_gam
    if M_alg <= 0:
        return None
    Ms = np.linspace(1e-6, M_alg - 1e-6, 50000)
    fv = coral_fv(Ms, g)
    saddles = []
    for i in range(len(fv) - 1):
        if fv[i]*fv[i+1] < 0:
            try:
                saddles.append(brentq(lambda m: coral_f(m, g),
                                      Ms[i], Ms[i+1]))
            except Exception:
                pass
    if len(saddles) == 0:
        return None
    return (0.0, saddles[0])

def coral_barrier(g, M_eq, M_sad):
    return quad(lambda m: -coral_f(m, g), M_eq, M_sad, limit=200)[0]

def coral_D_exact(sigma, g, M_eq, M_sad, lam_eq):
    xg = np.linspace(max(1e-10, M_eq), M_sad, N_GRID)
    dx = xg[1] - xg[0]
    neg_f = -coral_fv(xg, g)
    V = np.cumsum(neg_f)*dx
    V -= V[0]
    Phi = 2*V/sigma**2
    if Phi.max() > 600:
        return np.inf
    Phi = np.clip(Phi, -500, 500)
    Ix = np.cumsum(np.exp(-Phi))*dx
    psi = (2.0/sigma**2)*np.exp(Phi)*Ix
    MFPT = np.trapz(psi, xg)
    return MFPT*abs(lam_eq)

def coral_sigma_star(g, M_eq, M_sad, lam_eq):
    def obj(ls):
        D = coral_D_exact(np.exp(ls), g, M_eq, M_sad, lam_eq)
        if D == np.inf:
            return 1.0
        return np.log(max(D, 1e-30)) - np.log(CORAL_D)
    try:
        return np.exp(brentq(obj, np.log(0.001), np.log(1.0),
                             xtol=1e-10, maxiter=200))
    except ValueError:
        return np.nan

def coral_compute(g):
    eq = coral_find_eq(g)
    if eq is None:
        return None
    M_eq, M_sad = eq
    le = CORAL_a*CORAL_Cb - g/CORAL_Tb + CORAL_gam*CORAL_Tb
    if le >= 0:
        return None
    DP = coral_barrier(g, M_eq + 1e-10, M_sad)
    if DP < 1e-10:
        return None
    ss = coral_sigma_star(g, M_eq, M_sad, le)
    if np.isnan(ss):
        return None
    return {'param': g, 'sigma_star': ss, 'DeltaPhi': DP,
            'lambda_eq': abs(le), 'B': 2*DP/ss**2}


# ===========================================================================
# MODEL 5: JOSEPHSON JUNCTION
# ===========================================================================

def jj_V(phi, gamma):
    return -np.cos(phi) - gamma*phi

def jj_barrier(gamma):
    return 2.0*np.sqrt(1.0 - gamma**2) - 2.0*gamma*np.arccos(gamma)

def jj_eigenvalues(gamma):
    lam = np.sqrt(1.0 - gamma**2)
    return lam, -lam

def jj_equilibria(gamma):
    return np.arcsin(gamma), np.pi - np.arcsin(gamma)

def jj_D_exact(gamma, sigma):
    phi_min, phi_sad = jj_equilibria(gamma)
    lam_eq = np.sqrt(1.0 - gamma**2)
    phi_lo = max(phi_min - np.pi, phi_min - 5.0*sigma/np.sqrt(lam_eq))
    phig = np.linspace(phi_lo, phi_sad, N_GRID_PHYS)
    dp = phig[1] - phig[0]
    Vg = jj_V(phig, gamma)
    ie = np.argmin(np.abs(phig - phi_min))
    Vg = Vg - Vg[ie]
    Phi = 2.0*Vg/sigma**2 - 2.0*Vg[ie]/sigma**2
    Phi = Phi - Phi[ie]
    if np.max(Phi[ie:]) > 700:
        return np.inf
    Ix = np.cumsum(np.exp(-Phi))*dp
    psi = (2.0/sigma**2)*np.exp(Phi)*Ix
    MFPT = np.trapz(psi[ie:], phig[ie:])
    return MFPT*lam_eq

def jj_sigma_star(gamma, D_target, slo=0.001, shi=5.0):
    Dhi = jj_D_exact(gamma, shi)
    if Dhi > D_target:
        for s in [10.0, 20.0, 50.0]:
            if jj_D_exact(gamma, s) < D_target:
                shi = s
                break
        else:
            return None
    Dlo = jj_D_exact(gamma, slo)
    if Dlo == np.inf:
        Dlo = 1e30
    if Dlo < D_target:
        return None

    def obj(s):
        d = jj_D_exact(gamma, s)
        return (1e30 if d == np.inf else d) - D_target
    try:
        return brentq(obj, slo, shi, xtol=1e-12, maxiter=300)
    except ValueError:
        return None

def jj_compute(gamma, D_target=100):
    DV = jj_barrier(gamma)
    if DV < 1e-15:
        return None
    le = np.sqrt(1.0 - gamma**2)
    ss = jj_sigma_star(gamma, D_target)
    if ss is None:
        return None
    return {'param': gamma, 'sigma_star': ss, 'DeltaPhi': DV,
            'lambda_eq': le, 'B': 2*DV/ss**2}


# ===========================================================================
# MODEL 6: MAGNETIC NANOPARTICLE
# ===========================================================================

def nano_V(theta, h):
    return np.sin(theta)**2 - 2.0*h*np.cos(theta)

def nano_barrier(h):
    return (1.0 - h)**2

def nano_eigenvalues(h):
    return 2.0*(1.0 - h), -2.0*(1.0 - h**2)

def nano_D_exact(h, sigma):
    lam_eq = 2.0*(1.0 - h)
    if lam_eq <= 0:
        return 0.0
    theta_sad = np.arccos(-h)
    theta_ref = 2.0*np.pi - theta_sad
    tg = np.linspace(theta_sad, theta_ref, N_GRID_PHYS)
    dt = tg[1] - tg[0]
    Vg = nano_V(tg, h)
    ie = np.argmin(np.abs(tg - np.pi))
    Vg = Vg - Vg[ie]
    Phi = 2.0*Vg/sigma**2
    Phi = Phi - Phi[ie]
    if np.max(Phi) > 700:
        return np.inf
    enp = np.exp(-Phi)
    J = np.cumsum(enp[::-1])[::-1]*dt
    psi = (2.0/sigma**2)*np.exp(Phi)*J
    MFPT = np.trapz(psi[:ie+1], tg[:ie+1])
    return MFPT*lam_eq

def nano_sigma_star(h, D_target, slo=0.001, shi=5.0):
    Dhi = nano_D_exact(h, shi)
    if Dhi > D_target:
        for s in [10.0, 20.0, 50.0, 100.0]:
            if nano_D_exact(h, s) < D_target:
                shi = s
                break
        else:
            return None
    Dlo = nano_D_exact(h, slo)
    if Dlo == np.inf:
        Dlo = 1e30
    if Dlo < D_target:
        return None

    def obj(s):
        d = nano_D_exact(h, s)
        return (1e30 if d == np.inf else d) - D_target
    try:
        return brentq(obj, slo, shi, xtol=1e-12, maxiter=300)
    except ValueError:
        return None

def nano_compute(h, D_target=100):
    DV = nano_barrier(h)
    if DV < 1e-15:
        return None
    le = 2.0*(1.0 - h)
    ss = nano_sigma_star(h, D_target)
    if ss is None:
        return None
    return {'param': h, 'sigma_star': ss, 'DeltaPhi': DV,
            'lambda_eq': le, 'B': 2*DV/ss**2}


# ===========================================================================
# ECOLOGY CROSSING ANALYSIS
# ===========================================================================

def run_ecology(name, param_label, D_target, SD_forcing, op_point,
                bistable_lo, bistable_hi, compute_fn, fold_at_high):
    """Run full crossing analysis for an ecology system."""
    bw_total = bistable_hi - bistable_lo
    margin = 0.01 * bw_total
    params = np.linspace(bistable_lo + margin, bistable_hi - margin, N_PTS)

    print(f"\n=== {name} (D={D_target}, SD_forcing={SD_forcing}) ===\n")
    print(f"  Bistable range: {param_label} in [{bistable_lo:.4f}, {bistable_hi:.4f}]")

    # Compute at all grid points
    data = []
    print(f"\n  {param_label:>8s}  {'sigma*':>10s}  {'sigma_env':>10s}  "
          f"{'ratio':>8s}  {'B':>8s}  {'DeltaPhi':>12s}  {'lambda_eq':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*10}")

    for p in params:
        r = compute_fn(p)
        if r is None:
            continue
        se = SD_forcing / r['lambda_eq']
        ratio = r['sigma_star'] / se
        r['sigma_env'] = se
        r['ratio'] = ratio
        data.append(r)
        print(f"  {r['param']:8.5f}  {r['sigma_star']:10.6f}  {se:10.6f}  "
              f"{ratio:8.4f}  {r['B']:8.4f}  {r['DeltaPhi']:12.6e}  "
              f"{r['lambda_eq']:10.6f}")

    if len(data) < 3:
        print(f"  INSUFFICIENT DATA ({len(data)} points)")
        return None

    pa = np.array([d['param'] for d in data])
    ra = np.array([d['ratio'] for d in data])

    # Find crossing(s)
    crossing = None
    for i in range(len(ra) - 1):
        if (ra[i] - 1)*(ra[i+1] - 1) < 0:
            frac = (1 - ra[i])/(ra[i+1] - ra[i])
            p_cross = pa[i] + frac*(pa[i+1] - pa[i])
            # Recompute at crossing
            rc = compute_fn(p_cross)
            if rc is not None:
                se_c = SD_forcing / rc['lambda_eq']
                crossing = {'param': p_cross, 'B': rc['B'],
                            'sigma_star': rc['sigma_star'],
                            'sigma_env': se_c, 'lambda_eq': rc['lambda_eq'],
                            'DeltaPhi': rc['DeltaPhi']}
                break

    if crossing is None:
        print("\n  NO CROSSING FOUND")
        return None

    c = crossing
    print(f"\n  Crossing point: {param_label}_cross = {c['param']:.4f}")
    print(f"  B at crossing: {c['B']:.2f}")
    print(f"  sigma* at crossing: {c['sigma_star']:.4f}")
    print(f"  Operating point: {param_label}_op = {op_point}")
    dist = abs(op_point - c['param'])/bw_total*100
    print(f"  Distance to crossing: |op - cross| / bistable_width = {dist:.1f}%")

    # Bandwidth
    result = {'name': name, 'param_cross': c['param'], 'B_cross': c['B']}
    for tag, lo_r, hi_r in [("1.5x", 1/1.5, 1.5), ("2x", 0.5, 2.0)]:
        ib = (ra >= lo_r) & (ra <= hi_r)
        if np.any(ib):
            idx = np.where(ib)[0]
            plo, phi_ = pa[idx[0]], pa[idx[-1]]
            bwf = (phi_ - plo)/bw_total*100
            opin = plo <= op_point <= phi_
        else:
            plo, phi_, bwf, opin = 0, 0, 0, False
        print(f"\n  Bandwidth (r within [{lo_r:.2f}, {hi_r:.2f}]):  "
              f"{param_label} in [{plo:.4f}, {phi_:.4f}], "
              f"width = {bwf:.1f}% of bistable range")
        print(f"  Operating point inside {tag} band: "
              f"{'YES' if opin else 'NO'}")
        result[f'bw_{tag}'] = bwf
        result[f'op_in_{tag}'] = opin

    # IVT check
    if fold_at_high:
        near_fold = data[-1]
        interior = data[0]
    else:
        near_fold = data[0]
        interior = data[-1]

    print(f"\n  IVT check:")
    print(f"    Near fold ({param_label}={near_fold['param']:.4f}): "
          f"sigma* = {near_fold['sigma_star']:.3e}, "
          f"|lambda_eq| = {near_fold['lambda_eq']:.3e}, "
          f"sigma_env = {near_fold['sigma_env']:.3e}")
    print(f"    Interior  ({param_label}={interior['param']:.4f}): "
          f"sigma* = {interior['sigma_star']:.4f}, "
          f"sigma_env = {interior['sigma_env']:.4f}, "
          f"ratio = {interior['ratio']:.4f}")

    return result


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':
    t0 = time.time()

    print("=" * 78)
    print("CROSSING THEOREM TEST")
    print("=" * 78)

    summary = []

    # ===================================================================
    # ECOLOGY SYSTEMS
    # ===================================================================

    # --- LAKE ---
    print("\n--- Finding Lake bistable range ---")
    la, lh = lake_bistable()
    if la is not None:
        print(f"--- Running Lake (N={N_PTS}) ---")
        r = run_ecology("LAKE (q=8)", "a", LAKE_D, 0.139, 0.35,
                        la, lh, lake_compute, fold_at_high=True)
        if r:
            summary.append(r)
    print(f"  [{time.time()-t0:.0f}s elapsed]")

    # --- KELP ---
    print("\n--- Finding Kelp bistable range ---")
    kl, kh = kelp_bistable()
    if kl is not None:
        print(f"--- Running Kelp (N={N_PTS}) ---")
        r = run_ecology("KELP", "p", KELP_D, 2.26, 64.0,
                        kl, kh, kelp_compute, fold_at_high=False)
        if r:
            summary.append(r)
    print(f"  [{time.time()-t0:.0f}s elapsed]")

    # --- SAVANNA ---
    print("\n--- Finding Savanna bistable range ---")
    sl, sh = sav_bistable()
    if sl is not None:
        print(f"--- Running Savanna (N={N_PTS}) ---")
        r = run_ecology("SAVANNA", "beta", SAV_D, 0.0021, 0.39,
                        sl, sh, sav_compute, fold_at_high=True)
        if r:
            summary.append(r)
    print(f"  [{time.time()-t0:.0f}s elapsed]")

    # --- CORAL ---
    print("\n--- Running Coral (N={N_PTS}) ---".format(N_PTS=N_PTS))
    margin_c = 0.03*(CORAL_ghi - CORAL_glo)
    r = run_ecology("CORAL", "g", CORAL_D, 0.011, 0.30,
                    CORAL_glo, CORAL_ghi, coral_compute, fold_at_high=False)
    if r:
        summary.append(r)
    print(f"  [{time.time()-t0:.0f}s elapsed]")

    # ===================================================================
    # PHYSICS SYSTEMS
    # ===================================================================

    for phys_name, phys_compute, target_B, param_label in [
        ("JOSEPHSON JUNCTION", jj_compute, 3.26, "gamma"),
        ("NANOPARTICLE", nano_compute, 3.41, "h"),
    ]:
        D_phys = 100
        print(f"\n=== {phys_name} (D={D_phys}) ===\n")
        print(f"--- Computing sigma*({param_label}) ---")

        gvals = np.linspace(0.02, 0.98, N_PTS)
        pdata = []
        print(f"\n  {param_label:>8s}  {'sigma*':>10s}  {'B_bridge':>10s}  "
              f"{'DeltaV':>12s}  {'lambda_eq':>10s}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}")

        for gv in gvals:
            r = phys_compute(gv, D_phys)
            if r is None:
                continue
            pdata.append(r)
            print(f"  {r['param']:8.4f}  {r['sigma_star']:10.6f}  "
                  f"{r['B']:10.4f}  {r['DeltaPhi']:12.6e}  "
                  f"{r['lambda_eq']:10.6f}")

        if len(pdata) < 3:
            print("  INSUFFICIENT DATA")
            print(f"  [{time.time()-t0:.0f}s elapsed]")
            continue

        ppa = np.array([d['param'] for d in pdata])
        ssa = np.array([d['sigma_star'] for d in pdata])
        Ba = np.array([d['B'] for d in pdata])
        DPa = np.array([d['DeltaPhi'] for d in pdata])
        lea = np.array([d['lambda_eq'] for d in pdata])

        B_mean = np.mean(Ba)
        print(f"\n  B mean = {B_mean:.4f},  B CV = "
              f"{np.std(Ba)/B_mean*100:.2f}%")

        # --- sigma_thermal sweep ---
        print(f"\n  sigma_thermal sweep:")
        print(f"  {'sig_th':>8s}  {'gamma_x':>8s}  {'B_cross':>8s}  "
              f"{'BW_1.5x':>8s}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

        sweep_sth = np.linspace(0.05, float(np.max(ssa)*0.95), 12)
        cal_sth = None
        cal_gx = None
        cal_B = None
        cal_bw = None

        for sth in sweep_sth:
            ratio_p = ssa / sth
            # Find crossing
            gx, Bx = None, None
            for i in range(len(ratio_p) - 1):
                if (ratio_p[i] - 1)*(ratio_p[i+1] - 1) < 0:
                    frac = (1 - ratio_p[i])/(ratio_p[i+1] - ratio_p[i])
                    gx = ppa[i] + frac*(ppa[i+1] - ppa[i])
                    Bx = Ba[i] + frac*(Ba[i+1] - Ba[i])
                    break
            if gx is None:
                continue
            # Bandwidth
            ib = (ratio_p >= 1/1.5) & (ratio_p <= 1.5)
            bwf = 0.0
            if np.any(ib):
                idx = np.where(ib)[0]
                bwf = (ppa[idx[-1]] - ppa[idx[0]])/0.96*100
            print(f"  {sth:8.4f}  {gx:8.4f}  {Bx:8.4f}  {bwf:7.1f}%")

            # Track closest to target B
            if cal_B is None or abs(Bx - target_B) < abs(cal_B - target_B):
                cal_sth = sth
                cal_gx = gx
                cal_B = Bx
                cal_bw = bwf

        # Calibration: find sigma_thermal for exact target B
        # B is approximately constant, so interpolate B(gamma) = target_B
        for i in range(len(Ba) - 1):
            if (Ba[i] - target_B)*(Ba[i+1] - target_B) < 0:
                frac = (target_B - Ba[i])/(Ba[i+1] - Ba[i])
                g_cal = ppa[i] + frac*(ppa[i+1] - ppa[i])
                s_cal = ssa[i] + frac*(ssa[i+1] - ssa[i])
                cal_sth = s_cal
                cal_gx = g_cal
                cal_B = target_B
                # Recompute bandwidth at calibrated sigma_thermal
                ratio_cal = ssa / s_cal
                ib = (ratio_cal >= 1/1.5) & (ratio_cal <= 1.5)
                if np.any(ib):
                    idx = np.where(ib)[0]
                    cal_bw = (ppa[idx[-1]] - ppa[idx[0]])/0.96*100
                break

        if cal_sth is not None:
            print(f"\n  Calibrated: sigma_thermal = {cal_sth:.4f} "
                  f"(B_cross = {cal_B:.2f}):")
            print(f"    Crossing at {param_label}_cross = {cal_gx:.4f}")
            print(f"    Bandwidth (1.5x): {cal_bw:.1f}% of bistable range")

        # IVT check
        near_fold = pdata[-1]   # high gamma/h → fold
        interior = pdata[0]     # low gamma/h → interior
        print(f"\n  IVT check:")
        print(f"    Near fold ({param_label}={near_fold['param']:.4f}): "
              f"sigma* = {near_fold['sigma_star']:.3e}, "
              f"|lambda_eq| = {near_fold['lambda_eq']:.3e}")
        print(f"    Interior  ({param_label}={interior['param']:.4f}): "
              f"sigma* = {interior['sigma_star']:.4f} (max region)")
        print(f"    Crossing exists for any sigma_thermal in "
              f"(0, {np.max(ssa):.4f})")

        summary.append({
            'name': phys_name, 'param_cross': cal_gx if cal_gx else 0,
            'B_cross': cal_B if cal_B else B_mean,
            'bw_1.5x': cal_bw if cal_bw else 0,
            'bw_2x': None, 'op_in_1.5x': None, 'op_in_2x': None,
        })
        print(f"  [{time.time()-t0:.0f}s elapsed]")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print(f"\n\n{'=' * 78}")
    print("=== SUMMARY ===")
    print("=" * 78)
    print(f"{'System':<18s} {'cross':>8s} {'B_cross':>8s} {'BW_1.5x':>8s} "
          f"{'BW_2x':>8s} {'Op in band?':>12s}")
    print("-" * 78)

    for s in summary:
        nm = s['name']
        pc = f"{s['param_cross']:.4f}" if s['param_cross'] else "---"
        bc = f"{s['B_cross']:.2f}" if s['B_cross'] else "---"
        b15 = f"{s['bw_1.5x']:.1f}%" if s.get('bw_1.5x') is not None else "---"
        b2 = f"{s['bw_2x']:.1f}%" if s.get('bw_2x') is not None else "---"
        if s.get('op_in_1.5x') is not None:
            op = "YES" if s['op_in_1.5x'] else "NO"
        else:
            op = "(calibrated)"
        print(f"{nm:<18s} {pc:>8s} {bc:>8s} {b15:>8s} {b2:>8s} {op:>12s}")

    print("=" * 78)

    # ===================================================================
    # INTERPRETATION
    # ===================================================================
    print(f"\n=== INTERPRETATION ===")

    eco = [s for s in summary if s.get('op_in_1.5x') is not None]
    all_in_15 = all(s.get('op_in_1.5x', False) for s in eco)
    all_in_2 = all(s.get('op_in_2x', False) for s in eco)
    B_crosses = [s['B_cross'] for s in summary if s['B_cross']]
    all_in_zone = all(1.8 <= b <= 6.0 for b in B_crosses)

    print(f"\n1. All ecology operating points inside 1.5x bandwidth: "
          f"{'YES' if all_in_15 else 'NO'}")
    print(f"   All ecology operating points inside 2x bandwidth: "
          f"{'YES' if all_in_2 else 'NO'}")
    print(f"\n2. B_cross in [1.8, 6.0] for all systems: "
          f"{'YES' if all_in_zone else 'NO'}")
    for s in summary:
        print(f"   {s['name']}: B_cross = {s['B_cross']:.2f}")
    bws = [s['bw_1.5x'] for s in summary if s.get('bw_1.5x')]
    if bws:
        print(f"\n3. Typical 1.5x bandwidth: {np.mean(bws):.1f}% "
              f"(range [{min(bws):.1f}%, {max(bws):.1f}%])")
        if np.mean(bws) > 20:
            print("   > 20% → 'near crossing' is generic, not fine-tuned")
        else:
            print("   < 20% → bandwidth is narrow")

    phys = [s for s in summary if s.get('op_in_1.5x') is None]
    eco_bws = [s['bw_1.5x'] for s in eco if s.get('bw_1.5x')]
    phys_bws = [s['bw_1.5x'] for s in phys if s.get('bw_1.5x')]
    if eco_bws and phys_bws:
        print(f"\n4. Physics bandwidth ({np.mean(phys_bws):.1f}%) vs "
              f"ecology ({np.mean(eco_bws):.1f}%): "
              f"{'comparable' if abs(np.mean(phys_bws) - np.mean(eco_bws)) < 15 else 'different'}")

    if all_in_15 and all_in_zone:
        print(f"""
  The sigma* = sigma_process match is a topological consequence of bistable
  dynamics plus observational selection. Systems exhibiting noise-driven
  transitions must operate near the crossing of sigma*(a) and sigma_env(a),
  which is guaranteed to exist by IVT. At the crossing, B_physical = B_bridge,
  which is bounded. The match is not an algebraic identity — it is geometric
  inevitability for observable transitioning systems.""")
    else:
        fails = [s['name'] for s in eco if not s.get('op_in_1.5x', True)]
        if fails:
            print(f"\n  Systems outside 1.5x band: {', '.join(fails)}")
            print("  Check whether SD(forcing) uncertainty could account "
                  "for this.")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"\n{'=' * 78}")
    print("END OF CROSSING THEOREM TEST")
    print("=" * 78)
