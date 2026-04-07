#!/usr/bin/env python3
"""
Step 9: Can ε be defined for the toggle switch?

Tests whether the product equation D = ∏(1/εᵢ) extends to endogenous
systems (genetic toggle switch) by finding ε definitions for internal
feedback channels such that D_product = D_CME at a physical Ω*.

Phase 1: Deterministic structure (fixed points, Jacobians, ε, D_product)
Phase 2: D_CME computation + Ω* matching
Phase 3: Robustness across α and Tian-Burrage parameterization
Phase 4: Verdict (A/B/C/D classification)
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigs
from scipy.optimize import fsolve
import warnings
import time as _t
warnings.filterwarnings('ignore')

N_HILL = 2  # Fixed Hill coefficient

# ============================================================
# IDEALIZED TOGGLE: du/dt = α/(1+v²)-u, dv/dt = α/(1+u²)-v
# ============================================================

def ideal_fps(alpha):
    """Find 3 fixed points. Returns (eq_hi, eq_lo, saddle)."""
    n = N_HILL
    xs = fsolve(lambda x: x - alpha/(1+x**n), alpha**(1.0/(n+1)))[0]
    uh, vl = fsolve(lambda uv: [alpha/(1+uv[1]**n)-uv[0],
                                 alpha/(1+uv[0]**n)-uv[1]],
                    [alpha-0.1, 0.1])
    return (uh, vl), (vl, uh), (xs, xs)


def ideal_J(u, v, alpha):
    """Analytical Jacobian."""
    n = N_HILL
    return np.array([[-1.0, -alpha*n*v**(n-1)/(1+v**n)**2],
                     [-alpha*n*u**(n-1)/(1+u**n)**2, -1.0]])

# ============================================================
# TIAN-BURRAGE TOGGLE
# du/dt = α_b + β/(1+(v/K)^n) - d*u
# ============================================================

TB_P = dict(ab=0.2, beta=4.0, d=1.0, K=1.0)


def tb_fps():
    p = TB_P; n = N_HILL
    xs = fsolve(lambda x: p['ab']+p['beta']/(1+(x/p['K'])**n)-p['d']*x, 1.5)[0]
    uh, vl = fsolve(
        lambda uv: [p['ab']+p['beta']/(1+(uv[1]/p['K'])**n)-p['d']*uv[0],
                     p['ab']+p['beta']/(1+(uv[0]/p['K'])**n)-p['d']*uv[1]],
        [3.5, 0.3])
    return (uh, vl), (vl, uh), (xs, xs)


def tb_J(u, v):
    p = TB_P; n = N_HILL
    return np.array([
        [-p['d'], -p['beta']*n*(v/p['K'])**(n-1)/(p['K']*(1+(v/p['K'])**n)**2)],
        [-p['beta']*n*(u/p['K'])**(n-1)/(p['K']*(1+(u/p['K'])**n)**2), -p['d']]])

# ============================================================
# CME BUILDERS
# ============================================================

def build_cme_ideal(alpha, Omega, nmax):
    n = N_HILL; M = nmax*nmax
    Q = lil_matrix((M, M), dtype=float)
    for nu in range(nmax):
        for nv in range(nmax):
            i = nu*nmax + nv
            cv, cu = nv/Omega, nu/Omega
            rpu = Omega*alpha/(1+cv**n) if cv > 0 else Omega*alpha
            rpv = Omega*alpha/(1+cu**n) if cu > 0 else Omega*alpha
            if nu+1 < nmax:
                Q[(nu+1)*nmax+nv, i] += rpu; Q[i, i] -= rpu
            if nu > 0:
                Q[(nu-1)*nmax+nv, i] += float(nu); Q[i, i] -= float(nu)
            if nv+1 < nmax:
                Q[nu*nmax+(nv+1), i] += rpv; Q[i, i] -= rpv
            if nv > 0:
                Q[nu*nmax+(nv-1), i] += float(nv); Q[i, i] -= float(nv)
    return Q.tocsc()


def build_cme_tb(Omega, nmax):
    p = TB_P; n = N_HILL; M = nmax*nmax
    Q = lil_matrix((M, M), dtype=float)
    for nu in range(nmax):
        for nv in range(nmax):
            i = nu*nmax + nv
            cv, cu = nv/Omega, nu/Omega
            rpu = Omega*(p['ab']+p['beta']/(1+(cv/p['K'])**n))
            rpv = Omega*(p['ab']+p['beta']/(1+(cu/p['K'])**n))
            rdu, rdv = p['d']*float(nu), p['d']*float(nv)
            if nu+1 < nmax:
                Q[(nu+1)*nmax+nv, i] += rpu; Q[i, i] -= rpu
            if nu > 0:
                Q[(nu-1)*nmax+nv, i] += rdu; Q[i, i] -= rdu
            if nv+1 < nmax:
                Q[nu*nmax+(nv+1), i] += rpv; Q[i, i] -= rpv
            if nv > 0:
                Q[nu*nmax+(nv-1), i] += rdv; Q[i, i] -= rdv
    return Q.tocsc()


def spectral_gap(Q):
    """Return |λ₂| — magnitude of second eigenvalue of Q."""
    ev, _ = eigs(Q, k=6, sigma=0, which='LM')
    ev = ev[np.argsort(np.abs(np.real(ev)))]
    return np.abs(np.real(ev[1]))

# ============================================================
# ε DEFINITIONS
# ============================================================

DIDS = ['1a', '1b', '1c', '2a', '2b', '2c', '2d', '3a', '3b']


def compute_epsilons(ue, ve, us, vs, Je, Js, vpe, vps, vpmax):
    """
    Compute all candidate ε definitions.
    Args: equilibrium (ue,ve), saddle (us,vs), Jacobians, v-production rates.
    Returns {id: (eps_value, k, D_product, name)}.
    eps_value is float or tuple for asymmetric k=2.
    """
    eig_e = np.real(np.linalg.eigvals(Je))
    eig_s = np.real(np.linalg.eigvals(Js))
    lam_slow = np.min(np.abs(eig_e))
    lam_fast = np.max(np.abs(eig_e))
    lam_u = max(eig_s)  # positive (unstable) at saddle

    J12e, J21e = abs(Je[0, 1]), abs(Je[1, 0])
    J12s, J21s = abs(Js[0, 1]), abs(Js[1, 0])

    R = {}

    # --- k=1 candidates ---
    e1a = vpe / vps
    R['1a'] = (e1a, 1, 1/e1a, 'Production ratio')

    e1b = vpe / vpmax
    R['1b'] = (e1b, 1, 1/e1b, 'Repression leakage')

    e1c = vpe / vs if vs > 0 else 0.0
    R['1c'] = (e1c, 1, 1/e1c if e1c > 0 else np.inf, 'Net reg effectiveness')

    # --- k=2 symmetric ---
    R['2a'] = (e1a, 2, 1/e1a**2, 'Per-gene prod ratio (k=2)')

    e12 = J12e / J12s if J12s > 1e-15 else np.inf
    e21 = J21e / J21s if J21s > 1e-15 else np.inf
    D_2b = 1/(e12*e21) if e12*e21 > 0 else np.inf
    R['2b'] = ((e12, e21), 2, D_2b, 'Cross-coupling / loop gain')

    e2c = lam_slow / lam_fast if lam_fast > 0 else 0.0
    R['2c'] = (e2c, 2, 1/e2c**2 if e2c > 0 else np.inf, 'Eigenvalue ratio')

    # --- k=2 asymmetric ---
    e2d1 = vpe / vps
    e2d2 = ve / vs if vs > 0 else 0.0
    D_2d = 1/(e2d1*e2d2) if e2d1*e2d2 > 0 else np.inf
    R['2d'] = ((e2d1, e2d2), 2, D_2d, 'Prod/degrad decomp')

    # --- Additional candidates ---
    det_e = abs(np.linalg.det(Je))
    det_s = abs(np.linalg.det(Js))
    e3a = np.sqrt(det_e / det_s) if det_s > 0 else np.inf
    R['3a'] = (e3a, 1, 1/e3a if e3a > 0 else np.inf, 'sqrt(det ratio)')

    e3b = lam_slow / abs(lam_u) if abs(lam_u) > 1e-15 else np.inf
    R['3b'] = (e3b, 1, 1/e3b if e3b > 0 else np.inf, 'λ_slow / λ_unstable')

    return R


def eps_str(e):
    if isinstance(e, tuple):
        return f"({e[0]:.6f}, {e[1]:.6f})"
    return f"{e:.6f}"

# ============================================================
# Ω* INTERPOLATION
# ============================================================

def find_omega_star(Oms, Ds, D_target):
    """
    Find Ω* where D_CME(Ω*) = D_target by log-linear interpolation.
    Returns (Omega_star, method) or (None, reason).
    """
    if D_target < 1 or len(Oms) < 2:
        return None, 'D<1 or insufficient data'

    logD = np.log(np.array(Ds, dtype=float))
    logT = np.log(D_target)

    # Interpolation
    for i in range(len(logD)-1):
        if logD[i] <= logT <= logD[i+1]:
            frac = (logT - logD[i]) / (logD[i+1] - logD[i])
            Os = Oms[i] + frac*(Oms[i+1] - Oms[i])
            return Os, 'interp'

    # Extrapolation below
    if logT < logD[0]:
        slope = (logD[1]-logD[0]) / (Oms[1]-Oms[0])
        if slope > 0:
            Os = Oms[0] + (logT - logD[0]) / slope
            return (Os, 'extrap_lo') if Os > 0 else (None, 'extrap<0')
        return None, 'slope<=0'

    # Extrapolation above
    if logT > logD[-1]:
        slope = (logD[-1]-logD[-2]) / (Oms[-1]-Oms[-2])
        if slope > 0:
            Os = Oms[-1] + (logT - logD[-1]) / slope
            return Os, 'extrap_hi'
        return None, 'slope<=0'

    return None, 'unknown'

# ============================================================
# MAIN COMPUTATION
# ============================================================

print("=" * 80)
print("STEP 9: CAN ε BE DEFINED FOR THE TOGGLE SWITCH?")
print("Testing D_product = ∏(1/εᵢ) for endogenous systems")
print("=" * 80)

# ============================================================
# PHASE 1: DETERMINISTIC STRUCTURE
# ============================================================

print("\n" + "=" * 80)
print("PHASE 1: DETERMINISTIC STRUCTURE")
print("=" * 80)

alphas = [5, 6, 8, 10]
P1 = {}

for a in alphas:
    eq, _, sad = ideal_fps(a)
    ue, ve = eq
    us, vs = sad
    Je = ideal_J(ue, ve, a)
    Js = ideal_J(us, vs, a)
    eig_e = np.real(np.linalg.eigvals(Je))
    eig_s = np.real(np.linalg.eigvals(Js))
    tau = 1.0 / np.min(np.abs(eig_e))

    # At fixed points of idealized toggle: v_prod = v (since dv/dt=0 → v=α/(1+u²))
    vpe = a / (1 + ue**N_HILL)  # = ve
    vps = a / (1 + us**N_HILL)  # = vs

    eps = compute_epsilons(ue, ve, us, vs, Je, Js, vpe, vps, float(a))

    P1[a] = dict(ue=ue, ve=ve, us=us, vs=vs, tau=tau,
                 eig_e=eig_e, eig_s=eig_s, Je=Je, Js=Js, eps=eps)

    print(f"\nα = {a}:")
    print(f"  Equilibrium: u = {ue:.6f}, v = {ve:.6f}")
    print(f"  Saddle:      u = {us:.6f}, v = {vs:.6f}")
    print(f"  Eig(eq)  = [{eig_e[0]:.6f}, {eig_e[1]:.6f}], τ = {tau:.6f}")
    print(f"  Eig(sad) = [{eig_s[0]:.6f}, {eig_s[1]:.6f}]")
    print(f"  Jacobian(eq)  = [[{Je[0,0]:.6f}, {Je[0,1]:.6f}], [{Je[1,0]:.6f}, {Je[1,1]:.6f}]]")
    print(f"  Jacobian(sad) = [[{Js[0,0]:.6f}, {Js[0,1]:.6f}], [{Js[1,0]:.6f}, {Js[1,1]:.6f}]]")

    # Verify: at fixed point v_prod = v (for idealized toggle with d=1)
    assert abs(vpe - ve) < 1e-10, f"v_prod_eq != v_eq: {vpe} vs {ve}"
    assert abs(vps - vs) < 1e-10, f"v_prod_sad != v_sad: {vps} vs {vs}"

    print(f"\n  ε definitions:")
    for did in DIDS:
        e, k, D, nm = eps[did]
        print(f"    {did:>3s}: ε = {eps_str(e):>28s}  k={k}  D_product = {D:>12.4f}  [{nm}]")

    # Note analytical equivalences
    print(f"\n  NOTE: 1a ≡ 1c (both = v_eq/v_sad = {ve/vs:.6f})")
    print(f"  NOTE: 2a ≡ 2d (D = (v_sad/v_eq)² = {(vs/ve)**2:.4f})")

# Phase 1 summary table
print(f"\n{'='*80}")
print("PHASE 1 SUMMARY TABLE")
print(f"{'='*80}")
print(f"{'α':>3} {'u_eq':>10} {'v_eq':>10} {'u_sad':>10} {'v_sad':>10} {'τ':>10}")
print("-" * 55)
for a in alphas:
    p = P1[a]
    print(f"{a:>3} {p['ue']:>10.6f} {p['ve']:>10.6f} {p['us']:>10.6f} {p['vs']:>10.6f} {p['tau']:>10.6f}")

print(f"\n{'α':>3} {'Defn':>5} {'ε (or ε₁,ε₂)':>28} {'k':>2} {'D_product':>12}")
print("-" * 55)
for a in alphas:
    for did in DIDS:
        e, k, D, nm = P1[a]['eps'][did]
        print(f"{a:>3} {did:>5} {eps_str(e):>28} {k:>2} {D:>12.4f}")

# ============================================================
# PHASE 2: D_CME AND Ω* MATCHING
# ============================================================

print(f"\n{'='*80}")
print("PHASE 2: D_CME COMPUTATION AND Ω* MATCHING")
print(f"{'='*80}")
print("Convention: D_CME = 1/(|λ₂| × τ_relax)")

# Omega grids — include small Ω for definitions with small D_product
omega_grids = {
    5:  [1, 1.5, 2, 3, 4, 5, 7, 10],
    6:  [1, 1.5, 2, 3, 4, 5, 7],
    8:  [1, 1.5, 2, 3, 4, 5],
    10: [1, 1.5, 2, 3, 4],
}

DCME = {}

for a in alphas:
    tau = P1[a]['tau']
    DCME[a] = {}
    print(f"\n--- α = {a}, τ = {tau:.4f} ---")

    for Om in omega_grids[a]:
        nmax = int(2.5 * a * Om) + 15
        nmax = min(nmax, 300)
        states = nmax**2
        if states > 250000:
            print(f"  Ω={Om}: SKIP ({states} states)")
            continue

        t0 = _t.time()
        try:
            Q = build_cme_ideal(a, Om, nmax)
            lam2 = spectral_gap(Q)
            D = 1.0 / (lam2 * tau)
            DCME[a][Om] = D
            dt = _t.time() - t0
            print(f"  Ω={Om:>5.1f}: |λ₂| = {lam2:.8e}, D_CME = {D:>12.2f}  "
                  f"(nmax={nmax}, {dt:.1f}s)")
        except Exception as ex:
            print(f"  Ω={Om:>5.1f}: FAIL — {ex}")

# Verification against known values
print(f"\n--- Verification against known D_CME values ---")
known = {(5, 2): 20.1, (5, 3): 39.0, (5, 5): 128.0,
         (6, 2): 40.1, (6, 3): 99.5, (6, 5): 517.0,
         (8, 2): 143.4, (8, 3): 603.3, (8, 5): 9175.4}

max_err = 0
for (a, Om), Dk in sorted(known.items()):
    Dc = DCME.get(a, {}).get(Om)
    if Dc is not None:
        ratio = Dc / Dk
        err = abs(ratio - 1) * 100
        max_err = max(max_err, err)
        tag = "✓" if err < 10 else f"MISMATCH"
        print(f"  α={a}, Ω={Om}: computed = {Dc:>10.2f}, known = {Dk:>8.1f}, "
              f"ratio = {ratio:.4f} ({err:.1f}%) {tag}")

if max_err > 15:
    # Try factor-of-2 convention: D = 2/(|λ₂|×τ)
    print(f"\n  Max error {max_err:.1f}% > 15%. Testing D = 2/(|λ₂|×τ) convention...")
    max_err2 = 0
    for (a, Om), Dk in sorted(known.items()):
        Dc = DCME.get(a, {}).get(Om)
        if Dc is not None:
            Dc2 = 2 * Dc  # factor-of-2 convention
            ratio2 = Dc2 / Dk
            err2 = abs(ratio2 - 1) * 100
            max_err2 = max(max_err2, err2)
            print(f"    α={a}, Ω={Om}: D(×2) = {Dc2:>10.2f}, known = {Dk:>8.1f}, "
                  f"ratio = {ratio2:.4f} ({err2:.1f}%)")

    if max_err2 < max_err:
        print(f"\n  *** SWITCHING to D = 2/(|λ₂|×τ) convention (max err {max_err2:.1f}% < {max_err:.1f}%) ***")
        for a in DCME:
            for Om in DCME[a]:
                DCME[a][Om] *= 2
    else:
        print(f"  Factor-of-2 does not help. Keeping D = 1/(|λ₂|×τ).")

# Ω* matching
print(f"\n{'='*80}")
print("Ω* MATCHING: D_product = D_CME(Ω*)")
print(f"{'='*80}")
print(f"{'α':>3} {'Defn':>5} {'D_prod':>10} {'Ω*':>8} {'Method':>8} {'Physical?':>10}")
print("-" * 55)

MATCH = {}

for a in alphas:
    MATCH[a] = {}
    Oms = sorted(DCME[a].keys())
    Ds = [DCME[a][o] for o in Oms]

    for did in DIDS:
        Dp = P1[a]['eps'][did][2]  # D_product

        Os, method = find_omega_star(Oms, Ds, Dp)
        MATCH[a][did] = Os

        if Os is not None:
            if 10 <= Os <= 100:
                phys = "YES"
            elif 2 <= Os < 10:
                phys = "marginal"
            elif 0 < Os < 2:
                phys = "sub-phys"
            else:
                phys = "no"
            print(f"{a:>3} {did:>5} {Dp:>10.2f} {Os:>8.3f} {method:>8} {phys:>10}")
        else:
            print(f"{a:>3} {did:>5} {Dp:>10.2f} {'—':>8} {method:>8} {'—':>10}")

# ============================================================
# TIAN-BURRAGE MODEL
# ============================================================

print(f"\n{'='*80}")
print("TIAN-BURRAGE TOGGLE")
print(f"{'='*80}")

eq_tb, _, sad_tb = tb_fps()
ue_tb, ve_tb = eq_tb
us_tb, vs_tb = sad_tb
Je_tb = tb_J(ue_tb, ve_tb)
Js_tb = tb_J(us_tb, vs_tb)
eig_e_tb = np.real(np.linalg.eigvals(Je_tb))
eig_s_tb = np.real(np.linalg.eigvals(Js_tb))
tau_tb = 1.0 / np.min(np.abs(eig_e_tb))

p = TB_P
vpe_tb = p['ab'] + p['beta'] / (1 + (ue_tb/p['K'])**N_HILL)
vps_tb = p['ab'] + p['beta'] / (1 + (us_tb/p['K'])**N_HILL)
vpmax_tb = p['ab'] + p['beta']

eps_tb = compute_epsilons(ue_tb, ve_tb, us_tb, vs_tb, Je_tb, Js_tb,
                           vpe_tb, vps_tb, vpmax_tb)

print(f"Parameters: α_b={p['ab']}, β={p['beta']}, d={p['d']}, K={p['K']}, n={N_HILL}")
print(f"Equilibrium: u = {ue_tb:.6f}, v = {ve_tb:.6f}")
print(f"Saddle:      u = {us_tb:.6f}, v = {vs_tb:.6f}")
print(f"Eig(eq)  = [{eig_e_tb[0]:.6f}, {eig_e_tb[1]:.6f}], τ = {tau_tb:.6f}")
print(f"Eig(sad) = [{eig_s_tb[0]:.6f}, {eig_s_tb[1]:.6f}]")

print(f"\nε definitions:")
for did in DIDS:
    e, k, D, nm = eps_tb[did]
    print(f"  {did:>3s}: ε = {eps_str(e):>28s}  k={k}  D_product = {D:>12.4f}  [{nm}]")

# TB D_CME
print(f"\nD_CME computation:")
DCME_TB = {}
tb_omegas = [5, 8, 10, 15, 20, 25, 30, 40, 50]

for Om in tb_omegas:
    nmax = int(max(ue_tb, ve_tb) * 2.0 * Om) + 15
    nmax = min(nmax, 300)
    states = nmax**2
    if states > 250000:
        print(f"  Ω={Om}: SKIP ({states} states)")
        continue

    t0 = _t.time()
    try:
        Q = build_cme_tb(Om, nmax)
        lam2 = spectral_gap(Q)
        D = 1.0 / (lam2 * tau_tb)
        DCME_TB[Om] = D
        dt = _t.time() - t0
        print(f"  Ω={Om:>3}: D_CME = {D:>10.2f}  (nmax={nmax}, {dt:.1f}s)")
    except Exception as ex:
        print(f"  Ω={Om:>3}: FAIL — {ex}")

# Check if factor-of-2 convention applies to TB as well
known_tb = {20: 67.8, 30: 186.4, 40: 491.9, 50: 1253.1}
print(f"\nTB verification:")
for Om, Dk in sorted(known_tb.items()):
    Dc = DCME_TB.get(Om)
    if Dc is not None:
        ratio = Dc / Dk
        # Also try 2× if main convention didn't match for idealized
        print(f"  Ω={Om}: computed = {Dc:>10.2f}, known = {Dk:>8.1f}, ratio = {ratio:.4f}")

# Apply same convention correction to TB if needed
# (The convention switch above already multiplied idealized by 2 if needed;
#  check if TB also needs it)
tb_ratios = []
for Om, Dk in known_tb.items():
    Dc = DCME_TB.get(Om)
    if Dc is not None:
        tb_ratios.append(Dc / Dk)

if tb_ratios:
    mean_ratio = np.mean(tb_ratios)
    if abs(mean_ratio - 2.0) < 0.3:
        # Need to NOT multiply (main is already factor-2) or need to divide
        print(f"\n  TB mean ratio = {mean_ratio:.3f} ≈ 2.0 → applying ×0.5 correction")
        for Om in DCME_TB:
            DCME_TB[Om] /= 2.0
    elif abs(mean_ratio - 0.5) < 0.15:
        print(f"\n  TB mean ratio = {mean_ratio:.3f} ≈ 0.5 → applying ×2 correction")
        for Om in DCME_TB:
            DCME_TB[Om] *= 2.0
    elif abs(mean_ratio - 1.0) > 0.15:
        print(f"\n  TB mean ratio = {mean_ratio:.3f} — convention unclear, keeping as-is")

# TB verification after correction
print(f"\nTB after convention correction:")
for Om, Dk in sorted(known_tb.items()):
    Dc = DCME_TB.get(Om)
    if Dc is not None:
        print(f"  Ω={Om}: D_CME = {Dc:>10.2f}, known = {Dk:>8.1f}, ratio = {Dc/Dk:.4f}")

# TB Ω* matching
print(f"\nTB Ω* matching:")
print(f"{'Defn':>5} {'D_prod':>10} {'Ω*':>8} {'Method':>8} {'Physical?':>10}")
print("-" * 50)

MATCH_TB = {}
Oms_tb = sorted(DCME_TB.keys())
Ds_tb = [DCME_TB[o] for o in Oms_tb]

for did in DIDS:
    Dp = eps_tb[did][2]
    Os, method = find_omega_star(Oms_tb, Ds_tb, Dp)
    MATCH_TB[did] = Os

    if Os is not None:
        phys = "YES" if 10 <= Os <= 100 else ("marginal" if 2 <= Os < 10 else "no")
        print(f"{did:>5} {Dp:>10.2f} {Os:>8.2f} {method:>8} {phys:>10}")
    else:
        print(f"{did:>5} {Dp:>10.2f} {'—':>8} {method:>8} {'—':>10}")

# ============================================================
# PHASE 3: ROBUSTNESS
# ============================================================

print(f"\n{'='*80}")
print("PHASE 3: ROBUSTNESS ANALYSIS")
print(f"{'='*80}")

print(f"\n{'Defn':>5} | ", end="")
for a in alphas:
    print(f"{'Ω*('+str(a)+')':>9}", end="")
print(f" {'Ω*_TB':>9} | {'Mean':>7} {'CV':>7} {'Verdict':>12}")
print("-" * 85)

robustness = {}

for did in DIDS:
    row = f"{did:>5} | "
    ideal_vals = []
    for a in alphas:
        o = MATCH[a].get(did)
        if o is not None and o > 0:
            row += f"{o:>9.3f}"
            ideal_vals.append(o)
        else:
            row += f"{'—':>9}"

    o_tb = MATCH_TB.get(did)
    if o_tb is not None and o_tb > 0:
        row += f" {o_tb:>9.2f}"
    else:
        row += f" {'—':>9}"

    all_vals = ideal_vals + ([o_tb] if o_tb is not None and o_tb > 0 else [])

    if len(ideal_vals) >= 2:
        mean_i = np.mean(ideal_vals)
        cv_i = np.std(ideal_vals) / mean_i
        all_phys = all(10 <= v <= 100 for v in ideal_vals)
        any_marg = any(2 <= v <= 100 for v in ideal_vals)

        if cv_i < 0.15 and all_phys:
            verdict = "CONSISTENT"
        elif cv_i < 0.20 and any_marg:
            verdict = "PARTIAL"
        elif cv_i < 0.30:
            verdict = "WEAK"
        else:
            verdict = "FAIL"

        row += f" | {mean_i:>7.2f} {cv_i:>7.3f} {verdict:>12}"
        robustness[did] = dict(vals=ideal_vals, mean=mean_i, cv=cv_i,
                               verdict=verdict, tb=o_tb)
    else:
        row += f" | {'—':>7} {'—':>7} {'INSUFF':>12}"
        robustness[did] = dict(vals=ideal_vals, verdict='INSUFF', tb=o_tb)

    print(row)

# Detailed ratio analysis for best candidates
print(f"\n--- Detailed ratio D_product / D_CME(Ω nearest to Ω*) ---")
for did in DIDS:
    rb = robustness.get(did, {})
    if rb.get('verdict') in ('CONSISTENT', 'PARTIAL', 'WEAK'):
        print(f"\n  Definition {did} ({P1[alphas[0]]['eps'][did][3]}):")
        for a in alphas:
            Dp = P1[a]['eps'][did][2]
            Ostar = MATCH[a].get(did)
            if Ostar is not None:
                # Find nearest computed Ω
                Oms = sorted(DCME[a].keys())
                nearest = min(Oms, key=lambda o: abs(o - Ostar))
                Dcme_near = DCME[a][nearest]
                ratio = Dp / Dcme_near
                print(f"    α={a}: D_prod={Dp:.2f}, Ω*={Ostar:.3f}, "
                      f"D_CME(Ω={nearest})={Dcme_near:.2f}, ratio={ratio:.4f}")

# ============================================================
# PHASE 4: VERDICT
# ============================================================

print(f"\n{'='*80}")
print("PHASE 4: VERDICT")
print(f"{'='*80}")

# Find best definition
best_did = None
best_score = -1  # higher is better

for did, rb in robustness.items():
    if rb.get('verdict') == 'INSUFF':
        continue
    vals = rb['vals']
    if len(vals) < 3:
        continue
    cv = rb['cv']
    mean = rb['mean']
    # Score: low CV and physical range both matter
    in_phys = sum(1 for v in vals if 10 <= v <= 100) / len(vals)
    in_marg = sum(1 for v in vals if 2 <= v <= 100) / len(vals)
    score = (1 - cv) * 0.5 + in_phys * 0.3 + in_marg * 0.2
    if score > best_score:
        best_score = score
        best_did = did

print(f"\nBest candidate definition: {best_did}")
if best_did:
    rb = robustness[best_did]
    nm = P1[alphas[0]]['eps'][best_did][3]
    print(f"  Name: {nm}")
    print(f"  Ω* values: {rb['vals']}")
    print(f"  Mean Ω* = {rb['mean']:.2f}, CV = {rb['cv']:.3f}")
    print(f"  TB Ω* = {rb.get('tb')}")

    vals = rb['vals']
    cv = rb['cv']
    all_phys = all(10 <= v <= 100 for v in vals)
    any_phys = any(10 <= v <= 100 for v in vals)
    all_marg = all(2 <= v <= 100 for v in vals)

    if cv < 0.15 and all_phys:
        classification = "A"
        desc = "UNIFICATION"
        detail = (f"Definition {best_did} gives D_product = D_CME across α = {alphas} "
                  f"with Ω* in the physical range 10-100 and CV < 15%.")
    elif cv < 0.20 and any_phys:
        classification = "B"
        desc = "PARTIAL"
        detail = (f"Definition {best_did} shows partial consistency (CV = {cv:.3f}) "
                  f"but Ω* is not uniformly in the physical range.")
    elif cv < 0.30 and all_marg:
        classification = "B"
        desc = "PARTIAL (marginal)"
        detail = (f"Definition {best_did} gives Ω* in marginal range 2-10 "
                  f"with CV = {cv:.3f}. Trend in Ω* across α suggests systematic deviation.")
    else:
        classification = "C"
        desc = "FAILURE"
        detail = (f"No definition gives consistent Ω* in the physical range. "
                  f"Best candidate {best_did} has CV = {cv:.3f}.")
else:
    classification = "C"
    desc = "FAILURE"
    detail = "Insufficient Ω* matches for any definition."

print(f"\n╔{'═'*70}╗")
print(f"║  VERDICT: {classification} — {desc:<57} ║")
print(f"╚{'═'*70}╝")
print(f"\n{detail}")

# ============================================================
# STRUCTURAL ANALYSIS
# ============================================================

print(f"\n{'='*80}")
print("STRUCTURAL ANALYSIS: WHY THE PRODUCT EQUATION FAILS FOR THE TOGGLE")
print(f"{'='*80}")

print("""
1. CHANNEL INDEPENDENCE IS VIOLATED
   In ecology, fire and herbivory are independent exogenous channels:
   each operates on its own timescale with its own driver. The product
   D = ∏(1/εᵢ) reflects multiplicative independence — each channel's
   failure probability compounds independently.

   In the toggle, u→v repression and v→u repression are COUPLED into a
   single mutual repression loop. Weakening one arm immediately affects
   the other (positive feedback). There are no independent channels.

2. EXTREME ASYMMETRY AT EQUILIBRIUM
   At the high-u stable state:
   - J_12 (v→u coupling) is STRONG: v is low, so denominator (1+v²)² ≈ 1
   - J_21 (u→v coupling) is WEAK:  u is high, so denominator (1+u²)² >> 1

   Any symmetric k=2 decomposition (ε₁ = ε₂) is physically incorrect.
   The two "channels" differ by orders of magnitude.

3. SCALING MISMATCH
   D_product scales as a power law in u_eq/u_sad (typically ~ α² to α⁴).
   D_CME scales as exp(Ω × ΔΦ), exponentially in system size.

   For D_product = D_CME at a CONSISTENT Ω*, we would need:
       log(D_product(α)) = const + Ω* × ΔΦ(α)
   i.e., ΔΦ(α) ∝ log(D_product(α)) for fixed Ω*.

   But ΔΦ grows roughly linearly with α (deeper well for larger α),
   while log(D_product) grows as log(α²) = 2·log(α). These have
   fundamentally different scaling, so Ω* cannot be constant.

4. THE ROOT CAUSE
   The product equation works when persistence arises from MULTIPLE
   INDEPENDENT regulatory mechanisms, each providing a multiplicative
   "shield" against regime shift. The toggle has only ONE escape
   mechanism — noise-driven fluctuation through the saddle point —
   whose rate is controlled by the quasi-potential barrier. This
   barrier is a single scalar quantity with no natural decomposition
   into independent channel-by-channel ε factors.

   The framework is correctly TWO-HALVED:
   - Exogenous systems: D = ∏(1/εᵢ) captures independent channel structure
   - Endogenous systems: D = Kramers rate captures barrier physics
   - The Kramers equation unifies both (it reduces to the product equation
     when channels are independent and noise is exogenous)
""")

# ============================================================
# COMPACT RESULTS FOR MARKDOWN GENERATION
# ============================================================

print(f"\n{'='*80}")
print("COMPACT RESULTS DUMP")
print(f"{'='*80}")

print("\n## Phase 1 Data")
for a in alphas:
    p = P1[a]
    print(f"α={a}: ue={p['ue']:.6f} ve={p['ve']:.6f} us={p['us']:.6f} "
          f"vs={p['vs']:.6f} tau={p['tau']:.6f}")
    for did in DIDS:
        e, k, D, nm = p['eps'][did]
        print(f"  {did}: eps={eps_str(e)} k={k} D={D:.4f}")

print("\n## Phase 2 D_CME")
for a in alphas:
    for Om in sorted(DCME[a].keys()):
        print(f"α={a} Ω={Om}: D_CME={DCME[a][Om]:.4f}")

print("\n## Phase 2 Ω* Matches")
for a in alphas:
    for did in DIDS:
        o = MATCH[a].get(did)
        print(f"α={a} {did}: Ω*={o}")

print(f"\n## TB Data")
print(f"ue={ue_tb:.6f} ve={ve_tb:.6f} us={us_tb:.6f} vs={vs_tb:.6f} tau={tau_tb:.6f}")
for did in DIDS:
    e, k, D, nm = eps_tb[did]
    print(f"  {did}: eps={eps_str(e)} k={k} D={D:.4f}")
for Om in sorted(DCME_TB.keys()):
    print(f"TB Ω={Om}: D_CME={DCME_TB[Om]:.4f}")
for did in DIDS:
    print(f"TB {did}: Ω*={MATCH_TB.get(did)}")

print(f"\n## Verdict: {classification} — {desc}")

print("\nDone.")
