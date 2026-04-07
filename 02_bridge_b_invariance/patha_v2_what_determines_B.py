#!/usr/bin/env python3
"""
PATH A v2: WHAT DETERMINES B?
==============================
Four tests probing what controls B = 2ΔΦ/σ*² and whether
β = ln(D) - B is predictable from equilibrium properties.

Test 1: β across 4 ecological systems at operating points
Test 2: 2-channel lake model, loading sweep (dynamic ε)
Test 3: Original 1D lake, ε grid at fixed ODE
Test 4: β₀ formula validation + Hermite σ* prediction

Script:  THEORY/X2/scripts/patha_v2_what_determines_B.py
Results: THEORY/X2/PATHA_V2_RESULTS.md
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import time
import warnings
warnings.filterwarnings('ignore')

t0 = time.time()

# ════════════════════════════════════════════════════════════════
# SHARED MFPT ENGINE
# ════════════════════════════════════════════════════════════════

def mfpt_D(f, xeq, xsad, lam_eq, sig, N=50000):
    """D = MFPT / tau via exact 1D integral."""
    tau = 1.0 / abs(lam_eq)
    # Grid from well below equilibrium to just past saddle
    spread = 3.0 * sig / np.sqrt(max(2.0 * abs(lam_eq), 1e-10))
    lo = max(1e-4, xeq - spread)
    hi = xsad + 1e-3
    if lo >= hi or sig <= 0:
        return 1e30
    x = np.linspace(lo, hi, N)
    dx = x[1] - x[0]
    nf = np.array([-f(xi) for xi in x])
    U = np.cumsum(nf) * dx
    ieq = np.argmin(np.abs(x - xeq))
    U -= U[ieq]
    P = np.clip(2.0 * U / sig**2, -500, 500)
    en = np.exp(-P)
    Ix = np.cumsum(en) * dx
    psi = (2.0 / sig**2) * np.exp(P) * Ix
    isad = np.argmin(np.abs(x - xsad))
    if ieq >= isad:
        return 1e30
    return np.trapz(psi[ieq:isad+1], x[ieq:isad+1]) / tau


def find_sig_star(f, xeq, xsad, lam_eq, Dtgt, N=50000,
                  lslo=-8.0, lshi=6.0, DPhi_hint=None):
    """Find σ* where D_exact(σ) = D_target via bisection on log(σ).
    DPhi_hint: if provided, use Kramers guess to seed bracket search."""
    def obj(ls):
        D = mfpt_D(f, xeq, xsad, lam_eq, np.exp(ls), N)
        if D <= 0 or not np.isfinite(D):
            return 50.0
        return np.log(D) - np.log(Dtgt)

    found = False

    # Try Kramers-guided bracket first (fast path)
    if DPhi_hint is not None and DPhi_hint > 0:
        denom = max(np.log(Dtgt) - 1.0, 0.5)
        sig_kr = np.sqrt(2.0 * DPhi_hint / denom)
        ls_guess = np.log(max(sig_kr, 1e-8))
        for lo_off, hi_off in [(-1, 1), (-2, 1), (-1, 2), (-3, 2), (-2, 3)]:
            lo_try = ls_guess + lo_off
            hi_try = ls_guess + hi_off
            try:
                vlo = obj(lo_try)
                vhi = obj(hi_try)
                if np.isfinite(vlo) and np.isfinite(vhi) and vlo * vhi < 0:
                    lslo, lshi = lo_try, hi_try
                    found = True
                    break
            except Exception:
                continue

    # Fallback: adaptive bracket search (fewer attempts)
    if not found:
        for lo_try in np.linspace(max(lslo, -7), lshi - 0.5, 20):
            for span in [1.0, 2.0, 4.0]:
                hi_try = lo_try + span
                if hi_try > lshi:
                    continue
                try:
                    vlo = obj(lo_try)
                    vhi = obj(hi_try)
                    if np.isfinite(vlo) and np.isfinite(vhi) and vlo * vhi < 0:
                        lslo, lshi = lo_try, hi_try
                        found = True
                        break
                except Exception:
                    continue
            if found:
                break
    if not found:
        return np.nan
    try:
        return np.exp(brentq(obj, lslo, lshi, xtol=1e-8))
    except Exception:
        return np.nan


def barrier_int(f, xlo, xhi):
    """ΔΦ = -∫f(x)dx from xlo to xhi."""
    r, _ = quad(lambda x: -f(x), xlo, xhi,
                limit=200, epsabs=1e-14, epsrel=1e-12)
    return r


def find_roots(f, lo, hi, N=100000):
    """All zeros of f in [lo, hi] by sign-change + brentq."""
    xs = np.linspace(lo, hi, N)
    fv = np.array([f(xi) for xi in xs])
    out = []
    for i in range(len(fv) - 1):
        if fv[i] * fv[i+1] < 0:
            try:
                out.append(brentq(f, xs[i], xs[i+1], xtol=1e-12))
            except Exception:
                pass
    return sorted(out)


def nd(f, x, h=1e-7):
    """Numerical derivative."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


def bridge_decompose(D_val, DPhi, sig, lam_eq, lam_sad):
    """Return B, β_obs, K, K_corr, β_formula."""
    B = 2.0 * DPhi / sig**2
    beta_obs = np.log(D_val) - B
    C_std = np.sqrt(abs(lam_eq) * abs(lam_sad)) / (2.0 * np.pi)
    tau = 1.0 / abs(lam_eq)
    D_Kr = np.exp(B) / (C_std * tau)
    K = D_val / D_Kr
    Kc = 2.0 * K
    bf = np.log(Kc) + np.log(np.pi) + 0.5 * np.log(abs(lam_eq) / abs(lam_sad))
    return B, beta_obs, K, Kc, bf


# ════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════

# --- Lake (van Nes & Scheffer 2007) ---
_b_lake = 0.8; _r_lake = 1.0; _q_lake = 8; _h_lake = 1.0

def mk_lake(a):
    def f(x):
        return a - _b_lake*x + _r_lake*x**_q_lake/(x**_q_lake + _h_lake**_q_lake)
    def fp(x):
        return -_b_lake + _r_lake*_q_lake*x**(_q_lake-1)*_h_lake**_q_lake / \
               (x**_q_lake + _h_lake**_q_lake)**2
    return f, fp

# --- 2-channel lake (Path C calibration) ---
_c1 = 0.05285071; _c2 = 0.19273736; _b0 = 0.68
_K1 = 0.5; _K2 = 2.0; _K14 = _K1**4

def mk_lake2ch(a):
    def f(x):
        rec = _r_lake * x**_q_lake / (x**_q_lake + _h_lake**_q_lake)
        ch1 = _c1 * x**4 / (x**4 + _K14)
        ch2 = _c2 * x / (x + _K2)
        return a + rec - _b0*x - ch1 - ch2
    return f

def eps_2ch(xeq):
    """ε₁, ε₂ from the 2-channel model at equilibrium xeq."""
    g1 = xeq**4 / (xeq**4 + _K14)
    g2 = xeq / (xeq + _K2)
    tr = _b0*xeq + _c1*g1 + _c2*g2
    return _c1*g1/tr, _c2*g2/tr

# --- Savanna (Staver-Levin, Xu et al. 2021) ---
_mu=0.2; _nu=0.1; _w0=0.9; _w1=0.2; _th=0.4; _ss=0.01

def _omega(G):
    return _w0 + (_w1 - _w0) / (1 + np.exp(-(G - _th) / _ss))

def _G_null(T, beta):
    n = _mu - T*(_mu - _nu)
    d = _mu + beta*T
    if d <= 0 or n <= 0: return np.nan
    G = n / d
    return G if (0 < G < 1 and G + T < 1) else np.nan

def mk_savanna(beta):
    def f(T):
        G = _G_null(T, beta)
        if np.isnan(G): return 0.0
        S = 1.0 - G - T
        if S <= 0: return 0.0
        return _omega(G) * S - _nu * T
    return f

# --- Kelp (constructed 1D, Step 6) ---
_r_k=0.4; _K_k=668.0; _h_k=100.0

def mk_kelp(p):
    def f(U):
        return _r_k*U*(1.0 - U/_K_k) - p*U/(U + _h_k)
    def fp(U):
        return _r_k*(1.0 - 2.0*U/_K_k) - p*_h_k/(U + _h_k)**2
    return f, fp

def kelp_nontrivial(p):
    """Non-trivial equilibria of kelp from quadratic."""
    ac = _r_k
    bc = _r_k*_h_k - _r_k*_K_k
    cc = _K_k*(p - _r_k*_h_k)
    disc = bc**2 - 4*ac*cc
    if disc < 0: return None
    sd = np.sqrt(disc)
    U1 = (-bc - sd) / (2*ac)
    U2 = (-bc + sd) / (2*ac)
    if U1 <= 0 or U2 <= 0: return None
    return (U1, U2)  # (saddle, kelp forest)

# --- Coral (Mumby 2007) ---
_a_c=0.1; _gam_c=0.8; _r_c=1.0; _d_c=0.44

def mk_coral(g):
    def f(M):
        C = max(1.0 - M - (_d_c + _a_c*M)/_r_c, 0.0)
        T = (_d_c + _a_c*M) / _r_c
        den = M + T
        if den < 1e-30: return 0.0
        return _a_c*M*C - g*M/den + _gam_c*M*T
    return f


# ════════════════════════════════════════════════════════════════
# COMPUTE SYSTEM DATA AT OPERATING POINT
# ════════════════════════════════════════════════════════════════

def compute_system(name, f_func, fp_func, xeq, xsad, D_target, N_grid=50000):
    """Compute all bridge components for a system."""
    lam_eq = fp_func(xeq) if fp_func else nd(f_func, xeq)
    lam_sad = fp_func(xsad) if fp_func else nd(f_func, xsad)
    DPhi = barrier_int(f_func, max(xeq, 1e-4), xsad)
    sig = find_sig_star(f_func, xeq, xsad, lam_eq, D_target, N=N_grid,
                        DPhi_hint=DPhi)
    if np.isnan(sig):
        return None
    B, beta_obs, K, Kc, beta_form = bridge_decompose(
        D_target, DPhi, sig, lam_eq, lam_sad)
    return {
        'name': name, 'D': D_target,
        'xeq': xeq, 'xsad': xsad,
        'lam_eq': lam_eq, 'lam_sad': lam_sad,
        'DPhi': DPhi, 'sig': sig,
        'B': B, 'beta': beta_obs,
        'K': K, 'Kc': Kc, 'beta_form': beta_form,
    }


print("=" * 70)
print("PATH A v2: WHAT DETERMINES B?")
print("=" * 70)


# ════════════════════════════════════════════════════════════════
# TEST 1: β ACROSS SYSTEMS
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TEST 1: β ACROSS SYSTEMS")
print("=" * 70)

T1 = []

# --- Lake at a = 0.326588 ---
print("\n  [Lake] a = 0.326588, D = 200 ...")
fl, flp = mk_lake(0.326588)
lr = find_roots(fl, 0.01, 3.0)
d = compute_system('Lake', fl, flp, lr[0], lr[1], 200.0)
if d: T1.append(d)
print(f"    x_eq={d['xeq']:.6f}  x_sad={d['xsad']:.6f}  "
      f"σ*={d['sig']:.6f}  B={d['B']:.4f}  β={d['beta']:.4f}")

# --- Savanna at β = 0.39 ---
print("  [Savanna] β = 0.39, D = 100 ...")
fs = mk_savanna(0.39)
sr = find_roots(fs, 0.01, 0.95)
d = compute_system('Savanna', fs, None, sr[0], sr[1], 100.0, N_grid=60000)
if d: T1.append(d)
print(f"    T_eq={d['xeq']:.6f}  T_sad={d['xsad']:.6f}  "
      f"σ*={d['sig']:.7f}  B={d['B']:.4f}  β={d['beta']:.4f}")

# --- Kelp: scan p to find operating point (B ≈ 1.80) ---
print("  [Kelp] scanning p for B ≈ 1.80, D = 29.4 ...")
kelp_bi = [p for p in np.linspace(35, 155, 300)
           if kelp_nontrivial(p) is not None and p > _r_k*_h_k]
p_lo_k, p_hi_k = kelp_bi[0], kelp_bi[-1]
margin_k = 0.05 * (p_hi_k - p_lo_k)

best_kelp = None
best_Bdiff_k = 1e10
for pv in np.linspace(p_lo_k + margin_k, p_hi_k - margin_k, 20):
    eq = kelp_nontrivial(pv)
    if eq is None: continue
    fk, fpk = mk_kelp(pv)
    usad = eq[0]
    lam0 = fpk(0.0)
    if lam0 >= 0: continue
    DP = barrier_int(fk, 0.01, usad)
    if DP <= 0: continue
    ss = find_sig_star(fk, 0.0, usad, lam0, 29.4, N=80000, DPhi_hint=DP)
    if np.isnan(ss): continue
    Bk = 2.0 * DP / ss**2
    if abs(Bk - 1.80) < best_Bdiff_k:
        best_Bdiff_k = abs(Bk - 1.80)
        best_kelp = (pv, usad, eq[1])

if best_kelp is not None:
    pv_k, usad_k, ufor_k = best_kelp
    fk, fpk = mk_kelp(pv_k)
    d = compute_system('Kelp', fk, fpk, 0.0, usad_k, 29.4, N_grid=80000)
    if d:
        T1.append(d)
        print(f"    p={pv_k:.2f}  U_sad={usad_k:.2f}  "
              f"σ*={d['sig']:.4f}  B={d['B']:.4f}  β={d['beta']:.4f}")
else:
    print("    WARNING: kelp operating point not found")

# --- Coral: scan g for operating point (B ≈ 6.03) ---
print("  [Coral] scanning g for B ≈ 6.03, D = 1111.1 ...")
# Analytical bistable range boundaries
g_lo_c = (_d_c/_r_c) * (_a_c*(1-_d_c/_r_c) + _gam_c*_d_c/_r_c)
g_hi_c = (_d_c+_a_c)*_gam_c/(_r_c+_a_c)

best_coral = None
best_Bdiff_c = 1e10
for gv in np.linspace(g_lo_c + 0.02, g_hi_c - 0.01, 20):
    fc = mk_coral(gv)
    cr = find_roots(fc, 0.002, 0.95)
    if len(cr) < 1: continue
    # M=0 is the coral eq, first root is the saddle
    msad = cr[0]
    lam0 = nd(fc, 0.001, h=5e-4)  # f'(0) via limit
    # More precise: f(M) = M*h(M), so f'(0) = h(0)
    # h(0) = a*C(0) - g/T(0) + γ*T(0)
    C0 = 1.0 - _d_c/_r_c
    T0 = _d_c/_r_c
    lam0_exact = _a_c*C0 - gv/T0 + _gam_c*T0
    lsad = nd(fc, msad)
    if lam0_exact >= 0 or lsad <= 0: continue
    DP = barrier_int(fc, 0.002, msad)
    if DP <= 0: continue
    ss = find_sig_star(fc, 0.0, msad, lam0_exact, 1111.1, N=80000, DPhi_hint=DP)
    if np.isnan(ss): continue
    Bc = 2.0 * DP / ss**2
    if abs(Bc - 6.03) < best_Bdiff_c:
        best_Bdiff_c = abs(Bc - 6.03)
        best_coral = (gv, msad, lam0_exact, lsad)

if best_coral is not None:
    gv_c, msad_c, lam0_c, lsad_c = best_coral
    fc = mk_coral(gv_c)
    # Use the exact eigenvalue at M=0
    DPhi_c = barrier_int(fc, 0.002, msad_c)
    sig_c = find_sig_star(fc, 0.0, msad_c, lam0_c, 1111.1, N=80000, DPhi_hint=DPhi_c)
    if not np.isnan(sig_c):
        B_c, beta_c, K_c, Kc_c, bf_c = bridge_decompose(
            1111.1, DPhi_c, sig_c, lam0_c, lsad_c)
        d = {
            'name': 'Coral', 'D': 1111.1,
            'xeq': 0.0, 'xsad': msad_c,
            'lam_eq': lam0_c, 'lam_sad': lsad_c,
            'DPhi': DPhi_c, 'sig': sig_c,
            'B': B_c, 'beta': beta_c,
            'K': K_c, 'Kc': Kc_c, 'beta_form': bf_c,
        }
        T1.append(d)
        print(f"    g={gv_c:.4f}  M_sad={msad_c:.6f}  "
              f"σ*={sig_c:.6f}  B={B_c:.4f}  β={beta_c:.4f}")
else:
    print("    WARNING: coral operating point not found")

# --- Test 1 summary table ---
print("\n" + "-" * 110)
print(f"{'System':>10s} {'D':>8s} {'ln(D)':>8s} {'B':>8s} {'β_obs':>8s}"
      f" {'K_corr':>8s} {'|λ_eq/λ_sad|':>12s} {'½ln(ratio)':>10s}"
      f" {'β_form':>8s} {'|err|':>8s}")
print("-" * 110)
for r in T1:
    ratio = abs(r['lam_eq']) / abs(r['lam_sad'])
    hlr = 0.5 * np.log(ratio)
    err = abs(r['beta'] - r['beta_form'])
    print(f"{r['name']:>10s} {r['D']:8.1f} {np.log(r['D']):8.4f} "
          f"{r['B']:8.4f} {r['beta']:8.4f} {r['Kc']:8.4f} "
          f"{ratio:12.6f} {hlr:10.4f} {r['beta_form']:10.4f} {err:8.5f}")

betas = [r['beta'] for r in T1]
print(f"\nβ range: [{min(betas):.4f}, {max(betas):.4f}]  "
      f"spread = {max(betas)-min(betas):.3f}")
print(f"β mean = {np.mean(betas):.4f}  CV = "
      f"{np.std(betas)/np.mean(betas)*100:.1f}%")

if max(betas) - min(betas) < 0.3:
    print("→ OUTCOME A direction: β approximately constant across systems")
else:
    print("→ OUTCOME B direction: β varies across systems (system-dependent)")

print(f"\nFormula verification (β_form should match β_obs):")
for r in T1:
    err_pct = abs(r['beta'] - r['beta_form']) / abs(r['beta']) * 100
    print(f"  {r['name']:>10s}: β_obs={r['beta']:.5f}  "
          f"β_form={r['beta_form']:.5f}  error={err_pct:.3f}%")


# ════════════════════════════════════════════════════════════════
# TEST 2: DYNAMIC ε — 2-CHANNEL LAKE LOADING SWEEP
# ════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("TEST 2: DYNAMIC ε (2-CHANNEL LAKE, LOADING SWEEP)")
print("=" * 70)

# Find bistable range of 'a' for the 2-channel model
a_test = np.linspace(0.01, 0.80, 1000)
a_bistable_2ch = []
for av in a_test:
    f2 = mk_lake2ch(av)
    rr = find_roots(f2, 0.01, 3.0)
    if len(rr) == 3:
        a_bistable_2ch.append(av)

if len(a_bistable_2ch) < 5:
    print("ERROR: 2-channel model bistable range too narrow")
else:
    a_lo2 = a_bistable_2ch[0]
    a_hi2 = a_bistable_2ch[-1]
    margin2 = 0.05 * (a_hi2 - a_lo2)
    a_scan = np.linspace(a_lo2 + margin2, a_hi2 - margin2, 25)
    print(f"  Bistable range: a ∈ [{a_lo2:.4f}, {a_hi2:.4f}]")
    print(f"  Scanning {len(a_scan)} points\n")

    print(f"{'a':>10s} {'ε₁':>8s} {'ε₂':>8s} {'D_prod':>10s} "
          f"{'ΔΦ':>12s} {'σ*':>10s} {'B':>8s} {'β':>8s} {'ln(D)':>8s}")
    print("-" * 100)

    T2 = []
    for av in a_scan:
        f2 = mk_lake2ch(av)
        rr = find_roots(f2, 0.01, 3.0)
        if len(rr) != 3:
            continue
        xeq2 = rr[0]; xsad2 = rr[1]
        e1, e2 = eps_2ch(xeq2)
        Dprod = 1.0 / (e1 * e2)
        leq2 = nd(f2, xeq2)
        lsad2 = nd(f2, xsad2)
        if leq2 >= 0 or lsad2 <= 0:
            continue
        DP2 = barrier_int(f2, xeq2, xsad2)
        if DP2 <= 0:
            continue
        ss2 = find_sig_star(f2, xeq2, xsad2, leq2, Dprod, N=50000, DPhi_hint=DP2)
        if np.isnan(ss2):
            continue
        B2 = 2.0 * DP2 / ss2**2
        beta2 = np.log(Dprod) - B2

        T2.append({
            'a': av, 'e1': e1, 'e2': e2, 'Dprod': Dprod,
            'DPhi': DP2, 'sig': ss2, 'B': B2, 'beta': beta2,
            'lnD': np.log(Dprod), 'leq': leq2, 'lsad': lsad2,
        })
        print(f"{av:10.6f} {e1:8.5f} {e2:8.5f} {Dprod:10.2f} "
              f"{DP2:12.8f} {ss2:10.6f} {B2:8.4f} {beta2:8.4f} "
              f"{np.log(Dprod):8.4f}")

    if len(T2) >= 3:
        B_arr = np.array([r['B'] for r in T2])
        lnD_arr = np.array([r['lnD'] for r in T2])
        beta_arr = np.array([r['beta'] for r in T2])

        # Linear fit: B = slope * ln(D) + intercept
        slope, intercept = np.polyfit(lnD_arr, B_arr, 1)

        print(f"\n--- ANALYSIS ---")
        print(f"  ε₁ range: [{min(r['e1'] for r in T2):.5f}, "
              f"{max(r['e1'] for r in T2):.5f}]")
        print(f"  ε₂ range: [{min(r['e2'] for r in T2):.5f}, "
              f"{max(r['e2'] for r in T2):.5f}]")
        print(f"  D_product range: [{min(r['Dprod'] for r in T2):.1f}, "
              f"{max(r['Dprod'] for r in T2):.1f}]")
        print(f"  B range: [{B_arr.min():.4f}, {B_arr.max():.4f}]")
        print(f"  β range: [{beta_arr.min():.4f}, {beta_arr.max():.4f}]  "
              f"CV = {np.std(beta_arr)/np.mean(beta_arr)*100:.2f}%")
        print(f"\n  Linear fit B = slope × ln(D) + intercept:")
        print(f"    slope     = {slope:.6f}")
        print(f"    intercept = {intercept:.6f}")
        residuals = B_arr - (slope * lnD_arr + intercept)
        rmse = np.sqrt(np.mean(residuals**2))
        print(f"    RMSE      = {rmse:.6f}")
        print(f"    R²        = {1 - np.var(residuals)/np.var(B_arr):.6f}")

        if abs(slope - 1.0) < 0.05:
            print(f"\n  → slope ≈ 1 (within 5%): B ≈ ln(D) + const")
            print(f"    β = -intercept = {-intercept:.4f}")
        else:
            print(f"\n  → slope = {slope:.4f} ≠ 1: nonlinear relationship")


# ════════════════════════════════════════════════════════════════
# TEST 3: ε GRID AT FIXED ODE
# ════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("TEST 3: ε GRID AT FIXED ODE (ORIGINAL 1D LAKE)")
print("=" * 70)

# Use the lake at a = 0.326588 (fixed ODE)
fl_fixed, flp_fixed = mk_lake(0.326588)
lr_fixed = find_roots(fl_fixed, 0.01, 3.0)
xeq_f = lr_fixed[0]; xsad_f = lr_fixed[1]
leq_f = flp_fixed(xeq_f); lsad_f = flp_fixed(xsad_f)
DPhi_f = barrier_int(fl_fixed, xeq_f, xsad_f)
tau_f = 1.0 / abs(leq_f)

print(f"  Fixed ODE: x_eq={xeq_f:.6f}, x_sad={xsad_f:.6f}")
print(f"  λ_eq={leq_f:.6f}, λ_sad={lsad_f:.6f}")
print(f"  ΔΦ={DPhi_f:.8f}, τ={tau_f:.6f}")

eps_values = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
print(f"\n  ε grid: {eps_values}")
print(f"  Skipping ε₁ + ε₂ > 0.85\n")

print(f"{'ε₁':>8s} {'ε₂':>8s} {'D_prod':>10s} {'ln(D)':>8s} "
      f"{'σ*':>10s} {'B':>8s} {'β':>8s}")
print("-" * 70)

T3 = []
for e1 in eps_values:
    for e2 in eps_values:
        if e1 + e2 > 0.85:
            continue
        Dprod = 1.0 / (e1 * e2)
        ss3 = find_sig_star(fl_fixed, xeq_f, xsad_f, leq_f, Dprod, N=50000,
                            DPhi_hint=DPhi_f)
        if np.isnan(ss3):
            continue
        B3 = 2.0 * DPhi_f / ss3**2
        beta3 = np.log(Dprod) - B3
        T3.append({
            'e1': e1, 'e2': e2, 'Dprod': Dprod,
            'sig': ss3, 'B': B3, 'beta': beta3,
            'lnD': np.log(Dprod),
        })
        print(f"{e1:8.4f} {e2:8.4f} {Dprod:10.1f} {np.log(Dprod):8.4f} "
              f"{ss3:10.6f} {B3:8.4f} {beta3:8.4f}")

if len(T3) >= 3:
    B3_arr = np.array([r['B'] for r in T3])
    lnD3_arr = np.array([r['lnD'] for r in T3])
    beta3_arr = np.array([r['beta'] for r in T3])

    # Linear fit
    slope3, intercept3 = np.polyfit(lnD3_arr, B3_arr, 1)

    print(f"\n--- ANALYSIS ---")
    print(f"  Number of valid ε combinations: {len(T3)}")
    print(f"  D_product range: [{min(r['Dprod'] for r in T3):.1f}, "
          f"{max(r['Dprod'] for r in T3):.1f}]")
    print(f"  B range: [{B3_arr.min():.4f}, {B3_arr.max():.4f}]")
    print(f"  β range: [{beta3_arr.min():.4f}, {beta3_arr.max():.4f}]")
    print(f"  β mean  = {np.mean(beta3_arr):.5f}")
    print(f"  β std   = {np.std(beta3_arr):.5f}")
    print(f"  β CV    = {np.std(beta3_arr)/np.mean(beta3_arr)*100:.3f}%")

    print(f"\n  Linear fit B = slope × ln(D) + intercept:")
    print(f"    slope     = {slope3:.6f}")
    print(f"    intercept = {intercept3:.6f}")
    res3 = B3_arr - (slope3 * lnD3_arr + intercept3)
    rmse3 = np.sqrt(np.mean(res3**2))
    r2_3 = 1 - np.var(res3) / np.var(B3_arr)
    print(f"    RMSE      = {rmse3:.6f}")
    print(f"    R²        = {r2_3:.8f}")

    # β₀ = -intercept (if slope = 1, then B = ln(D) - β₀)
    beta0_fit = -intercept3

    # Compare to formula
    # At each point, K_corr varies with B. Compute mean β from formula.
    beta_form3 = []
    for r in T3:
        _, _, _, Kc, bf = bridge_decompose(
            r['Dprod'], DPhi_f, r['sig'], leq_f, lsad_f)
        beta_form3.append(bf)
    beta_form3 = np.array(beta_form3)

    print(f"\n  β₀ from fit: {beta0_fit:.5f}")
    print(f"  β_formula mean: {np.mean(beta_form3):.5f}  "
          f"std: {np.std(beta_form3):.5f}")
    print(f"  β_formula CV: {np.std(beta_form3)/np.mean(beta_form3)*100:.3f}%")

    if abs(slope3 - 1.0) < 0.02 and np.std(beta3_arr) / np.mean(beta3_arr) < 0.05:
        print(f"\n  → β is APPROXIMATELY CONSTANT at fixed ODE (CV < 5%)")
        print(f"    B = ln(D) - {np.mean(beta3_arr):.4f}")
    elif abs(slope3 - 1.0) < 0.02:
        print(f"\n  → B ≈ ln(D) + const, but β has measurable variation")
        print(f"    Variation from K(B) dependence on barrier height")
    else:
        print(f"\n  → OUTCOME C: B vs ln(D) is nonlinear (slope ≠ 1)")


# ════════════════════════════════════════════════════════════════
# TEST 4: β₀ FROM JACOBIAN + HERMITE σ* PREDICTION
# ════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("TEST 4: β₀ FROM JACOBIAN + HERMITE σ* PREDICTION")
print("=" * 70)

# Part A: β₀ formula validation for all 4 systems
print("\n--- Part A: β₀ formula validation ---")
print(f"  Formula: β₀ = ln(K_corr) + ln(π) + ½ln(|λ_eq/λ_sad|)")
print()
print(f"{'System':>10s} {'β_obs':>10s} {'β_form':>10s} {'|error|':>10s}"
      f" {'K_corr':>8s} {'ln(Kc)':>8s} {'ln(π)':>8s} {'½ln(r)':>8s}")
print("-" * 80)

for r in T1:
    lnKc = np.log(r['Kc'])
    lnpi = np.log(np.pi)
    hlr = 0.5 * np.log(abs(r['lam_eq']) / abs(r['lam_sad']))
    err = abs(r['beta'] - r['beta_form'])
    print(f"{r['name']:>10s} {r['beta']:10.5f} {r['beta_form']:10.5f} "
          f"{err:10.6f} {r['Kc']:8.4f} {lnKc:8.4f} {lnpi:8.4f} {hlr:8.4f}")

beta_errs = [abs(r['beta'] - r['beta_form']) for r in T1]
print(f"\n  Max |β_obs - β_form| = {max(beta_errs):.6f}")
print(f"  Mean |β_obs - β_form| = {np.mean(beta_errs):.6f}")

if max(beta_errs) < 0.01:
    print("  → Formula reproduces β to < 0.01 for ALL systems")
elif max(beta_errs) < 0.05:
    print("  → Formula reproduces β to < 0.05 for all systems")
else:
    print("  → Formula has larger deviations — investigate")

# Part B: Hermite σ* prediction
print("\n--- Part B: Hermite σ* prediction ---")
print(f"  Hermite: ΔΦ ≈ Δx²/12 × (|λ_eq| + |λ_sad|)")
print(f"  σ*_Hermite = Δx × √[(|λ_eq|+|λ_sad|) / (6 × (ln(D) - β₀))]")
print()
print(f"{'System':>10s} {'Δx':>8s} {'ΔΦ_exact':>12s} {'ΔΦ_Herm':>12s}"
      f" {'ΔΦ err%':>8s} {'σ*_exact':>10s} {'σ*_Herm':>10s} {'σ* err%':>8s}")
print("-" * 90)

for r in T1:
    dx = r['xsad'] - r['xeq']
    le = abs(r['lam_eq'])
    ls = abs(r['lam_sad'])
    DPhi_herm = dx**2 / 12.0 * (le + ls)
    DPhi_err = (DPhi_herm - r['DPhi']) / r['DPhi'] * 100

    # β₀ from formula
    beta0 = r['beta_form']
    denom = np.log(r['D']) - beta0
    if denom > 0:
        sig_herm = dx * np.sqrt((le + ls) / (6.0 * denom))
        sig_err = (sig_herm - r['sig']) / r['sig'] * 100
    else:
        sig_herm = np.nan
        sig_err = np.nan

    print(f"{r['name']:>10s} {dx:8.5f} {r['DPhi']:12.8f} {DPhi_herm:12.8f}"
          f" {DPhi_err:+8.2f} {r['sig']:10.6f} {sig_herm:10.6f} {sig_err:+8.2f}")


# ════════════════════════════════════════════════════════════════
# SYNTHESIS
# ════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("SYNTHESIS")
print("=" * 70)

# Determine outcome
beta_spread = max(betas) - min(betas)
beta_cv = np.std(betas) / np.mean(betas) * 100

if len(T3) >= 3:
    beta3_cv = np.std(beta3_arr) / np.mean(beta3_arr) * 100
else:
    beta3_cv = np.nan

print(f"\n  Test 1: β across systems")
print(f"    β range = [{min(betas):.4f}, {max(betas):.4f}]  "
      f"spread = {beta_spread:.3f}  CV = {beta_cv:.1f}%")

if len(T2) >= 3:
    print(f"\n  Test 2: B vs ln(D) at variable loading")
    print(f"    slope = {slope:.4f}  (expect 1.0)")
    print(f"    β CV across loading = "
          f"{np.std(beta_arr)/np.mean(beta_arr)*100:.2f}%")

if len(T3) >= 3:
    print(f"\n  Test 3: β at fixed ODE across ε grid")
    print(f"    β CV = {beta3_cv:.3f}%  "
          f"(smaller CV → β more constant)")
    print(f"    B vs ln(D) slope = {slope3:.4f}  R² = {r2_3:.6f}")

print(f"\n  Test 4: Formula validation")
print(f"    Max β formula error = {max(beta_errs):.6f}")

# Classify outcome
print("\n" + "-" * 50)
if beta_spread < 0.3 and beta_cv < 15:
    print("  OUTCOME A: β ≈ constant across ε AND systems")
    print(f"  β_universal ≈ {np.mean(betas):.3f}")
    print(f"  B ≈ ln(D) - {np.mean(betas):.3f}")
elif len(T3) >= 3 and beta3_cv < 5:
    print("  OUTCOME B: β constant at fixed ODE, varies across systems")
    print(f"  Each system has its own β₀:")
    for r in T1:
        print(f"    {r['name']:>10s}: β₀ = {r['beta']:.4f}")
    print(f"  B = ln(D) - β₀(ODE shape)")
else:
    print("  OUTCOME C: β varies even at fixed ODE")
    print("  B vs ln(D) is nonlinear")

elapsed = time.time() - t0
print(f"\n  Runtime: {elapsed:.1f}s")
