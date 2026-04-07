#!/usr/bin/env python3
"""
Path C: Dynamic ε Test — Does D = ∏(1/ε(σ)) Hold Dynamically?

Simulates the 1D lake SDE at many noise levels σ. At each σ:
  - Measures time-averaged flux fractions ("effective ε") from simulation
  - Computes D_exact(σ) analytically via MFPT integral
  - Checks whether D_product(σ) = ∏(1/⟨εᵢ⟩) tracks D_exact(σ)

Option A: Two-channel regulatory model with ε₁=0.05, ε₂=0.10
Option B: Single-channel ratio ε_eff = f_reg/f_destab (cross-check)
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import time
import os

# ================================================================
# Numba setup
# ================================================================
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

print(f"numba: {'available' if HAS_NUMBA else 'NOT available — slower execution'}")

# ================================================================
# Physical parameters (van Nes & Scheffer 2007)
# ================================================================
A_P = 0.326588
B_P = 0.8
R_P = 1.0
Q_P = 8
H_P = 1.0

X_CL = 0.409217
X_SD = 0.978152
X_TB = 1.634126
LAM_CL = -0.784651
LAM_SD = 1.228791
TAU_L = 1.0 / abs(LAM_CL)

# Two-channel shape parameters
K1 = 0.5;  K2 = 2.0;  K1_4 = K1**4  # K1^4 = 0.0625
EPS1_T = 0.05;  EPS2_T = 0.10

# Simulation parameters
DT = 0.001
T_BURN = 1000
X_MIN = 0.001
N_TRAJ = 3
T_MAX = 200000 if HAS_NUMBA else 50000

CVS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.341, 0.40, 0.50, 0.60, 0.80]


# ================================================================
# Drift functions
# ================================================================
def f_lake(x):
    return A_P - B_P * x + R_P * x**Q_P / (x**Q_P + H_P**Q_P)

def cv_to_sigma(cv):
    return cv * X_CL * np.sqrt(2.0 * abs(LAM_CL))


# ================================================================
# Two-channel calibration
# ================================================================
print("\n" + "=" * 70)
print("TWO-CHANNEL MODEL CALIBRATION")
print("=" * 70)

total_reg_eq = B_P * X_CL
g1_eq = X_CL**4 / (X_CL**4 + K1_4)
g2_eq = X_CL / (X_CL + K2)

C1 = EPS1_T * total_reg_eq / g1_eq
C2 = EPS2_T * total_reg_eq / g2_eq
B0 = (1.0 - EPS1_T - EPS2_T) * B_P

print(f"  c1 = {C1:.8f}   c2 = {C2:.8f}   b0 = {B0:.4f}")
print(f"  g1(x_eq) = {g1_eq:.6f}   g2(x_eq) = {g2_eq:.6f}")
print(f"  ε₁ check: {C1*g1_eq/total_reg_eq:.4f} (target {EPS1_T})")
print(f"  ε₂ check: {C2*g2_eq/total_reg_eq:.4f} (target {EPS2_T})")


def f_2ch(x):
    rec = R_P * x**Q_P / (x**Q_P + H_P**Q_P)
    ch1 = C1 * x**4 / (x**4 + K1_4)
    ch2 = C2 * x / (x + K2)
    return A_P + rec - B0 * x - ch1 - ch2

def f_2ch_deriv(x, dx=1e-7):
    return (f_2ch(x + dx) - f_2ch(x - dx)) / (2.0 * dx)

print(f"  f_2ch(x_clear) = {f_2ch(X_CL):.2e}")

# Find equilibria
print("\n  Equilibria of two-channel model:")
x_scan = np.linspace(0.01, 3.0, 300000)
f_scan = np.array([f_2ch(xi) for xi in x_scan])
roots_2ch = []
for i in range(len(f_scan) - 1):
    if f_scan[i] * f_scan[i + 1] < 0:
        root = brentq(f_2ch, x_scan[i], x_scan[i + 1], xtol=1e-12)
        fp = f_2ch_deriv(root)
        roots_2ch.append(root)
        print(f"    x = {root:.8f}   f' = {fp:+.6f}   [{'stable' if fp < 0 else 'unstable'}]")

if len(roots_2ch) < 2:
    raise SystemExit("FATAL: Two-channel model not bistable")

X_CL2 = roots_2ch[0]
X_SD2 = roots_2ch[1]
X_TB2 = roots_2ch[2] if len(roots_2ch) > 2 else None
LAM_CL2 = f_2ch_deriv(X_CL2)
LAM_SD2 = f_2ch_deriv(X_SD2)
TAU_2 = 1.0 / abs(LAM_CL2)

# Equilibrium ε at actual two-channel equilibrium
g1_eq2 = X_CL2**4 / (X_CL2**4 + K1_4)
g2_eq2 = X_CL2 / (X_CL2 + K2)
treg2 = B0 * X_CL2 + C1 * g1_eq2 + C2 * g2_eq2
eps1_eq2 = C1 * g1_eq2 / treg2
eps2_eq2 = C2 * g2_eq2 / treg2

print(f"\n  Properties:")
print(f"    x_clear = {X_CL2:.8f} (orig {X_CL:.6f}, Δ={abs(X_CL2-X_CL)/X_CL*100:.3f}%)")
print(f"    x_sad   = {X_SD2:.8f} (orig {X_SD:.6f}, Δ={abs(X_SD2-X_SD)/X_SD*100:.3f}%)")
if X_TB2:
    print(f"    x_turb  = {X_TB2:.8f} (orig {X_TB:.6f}, Δ={abs(X_TB2-X_TB)/X_TB*100:.3f}%)")
print(f"    λ_clear = {LAM_CL2:+.6f} (orig {LAM_CL:+.6f})")
print(f"    λ_sad   = {LAM_SD2:+.6f} (orig {LAM_SD:+.6f})")
print(f"    τ       = {TAU_2:.6f} (orig {TAU_L:.6f})")
print(f"    ε₁(x_eq) = {eps1_eq2:.6f} (target {EPS1_T})")
print(f"    ε₂(x_eq) = {eps2_eq2:.6f} (target {EPS2_T})")

DPhi_orig, _ = quad(lambda x: -f_lake(x), X_CL, X_SD)
DPhi_2ch, _ = quad(lambda x: -f_2ch(x), X_CL2, X_SD2)
print(f"    ΔΦ_orig = {DPhi_orig:.8f}")
print(f"    ΔΦ_2ch  = {DPhi_2ch:.8f} (Δ={abs(DPhi_2ch-DPhi_orig)/DPhi_orig*100:.2f}%)")


# ================================================================
# D_exact (generalized MFPT integral)
# ================================================================
def compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma):
    N = 50000
    xg = np.linspace(0.001, x_saddle + 0.001, N)
    dx = xg[1] - xg[0]
    neg_f = np.array([-f_func(xi) for xi in xg])
    U_raw = np.cumsum(neg_f) * dx
    i_eq = np.argmin(np.abs(xg - x_eq))
    U = U_raw - U_raw[i_eq]
    Phi = 2.0 * U / sigma**2
    exp_neg = np.exp(-Phi)
    Ix = np.cumsum(exp_neg) * dx
    psi = (2.0 / sigma**2) * np.exp(Phi) * Ix
    i_sad = np.argmin(np.abs(xg - x_saddle))
    MFPT = np.trapz(psi[i_eq:i_sad + 1], xg[i_eq:i_sad + 1])
    return MFPT / tau_val


# ================================================================
# SDE simulation kernels
# ================================================================
@njit
def sim_2ch_core(sigma, dt, n_steps, n_burn, x0, x_ref,
                  c1, c2, b0, a_p, r_p, k1_4, k2, xmin):
    sqrt_dt = np.sqrt(dt)
    x = x0
    s1 = 0.0;  s2 = 0.0;  cnt = 0

    for i in range(n_steps):
        x2 = x * x;  x4 = x2 * x2;  x8 = x4 * x4
        rec = r_p * x8 / (x8 + 1.0)
        h1 = c1 * x4 / (x4 + k1_4)
        h2 = c2 * x / (x + k2)
        dr = a_p + rec - b0 * x - h1 - h2

        x = x + dr * dt + sigma * np.random.randn() * sqrt_dt

        if x < xmin:
            x = 2.0 * xmin - x
        if x > x_ref:
            x = 2.0 * x_ref - x
        if x < xmin:
            x = xmin
        if x > x_ref:
            x = x_ref

        if i >= n_burn:
            x2m = x * x;  x4m = x2m * x2m
            c1v = c1 * x4m / (x4m + k1_4)
            c2v = c2 * x / (x + k2)
            tr = b0 * x + c1v + c2v
            if tr > 1e-20:
                s1 += c1v / tr
                s2 += c2v / tr
                cnt += 1

    if cnt > 0:
        return s1 / cnt, s2 / cnt
    return 0.0, 0.0


@njit
def sim_lake_core(sigma, dt, n_steps, n_burn, x0, x_ref,
                   a_p, b_p, r_p, xmin):
    sqrt_dt = np.sqrt(dt)
    x = x0
    se = 0.0;  cnt = 0

    for i in range(n_steps):
        x2 = x * x;  x4 = x2 * x2;  x8 = x4 * x4
        rec = r_p * x8 / (x8 + 1.0)
        dr = a_p - b_p * x + rec

        x = x + dr * dt + sigma * np.random.randn() * sqrt_dt

        if x < xmin:
            x = 2.0 * xmin - x
        if x > x_ref:
            x = 2.0 * x_ref - x
        if x < xmin:
            x = xmin
        if x > x_ref:
            x = x_ref

        if i >= n_burn:
            x2m = x * x;  x4m = x2m * x2m;  x8m = x4m * x4m
            fd = a_p + r_p * x8m / (x8m + 1.0)
            fr = b_p * x
            if fd > 1e-20:
                se += fr / fd
                cnt += 1

    if cnt > 0:
        return se / cnt
    return 0.0


# ================================================================
# Warm up JIT
# ================================================================
if HAS_NUMBA:
    print("\nJIT warmup...", end=" ", flush=True)
    _ = sim_2ch_core(0.1, 0.01, 100, 10, X_CL2, X_SD2,
                      C1, C2, B0, A_P, R_P, K1_4, K2, X_MIN)
    _ = sim_lake_core(0.1, 0.01, 100, 10, X_CL, X_SD, A_P, B_P, R_P, X_MIN)
    print("done.")


# ================================================================
# Main loop
# ================================================================
print("\n" + "=" * 70)
print(f"SIMULATIONS: T_max={T_MAX}, dt={DT}, {N_TRAJ} traj/CV")
print(f"Steps/traj: {int(T_MAX/DT):,}")
print("=" * 70)

n_steps = int(T_MAX / DT)
n_burn = int(T_BURN / DT)

results_a = []
results_b = []
conv_a = []
conv_b = []

t0 = time.time()

for ci, cv in enumerate(CVS):
    sig = cv_to_sigma(cv)
    print(f"\n{'─'*55}")
    print(f"[{ci+1}/{len(CVS)}]  CV={cv:.3f}  σ={sig:.6f}")

    D_ex_2ch = compute_D_exact(f_2ch, X_CL2, X_SD2, TAU_2, sig)
    D_ex_orig = compute_D_exact(f_lake, X_CL, X_SD, TAU_L, sig)
    print(f"  D_exact: 2ch={D_ex_2ch:.6e}  orig={D_ex_orig:.6e}")

    # Option A
    e1r = [];  e2r = []
    ta = time.time()
    for tr in range(N_TRAJ):
        e1, e2 = sim_2ch_core(sig, DT, n_steps, n_burn, X_CL2, X_SD2,
                               C1, C2, B0, A_P, R_P, K1_4, K2, X_MIN)
        e1r.append(e1);  e2r.append(e2)
    ta = time.time() - ta

    e1m = np.mean(e1r);  e1s = np.std(e1r)
    e2m = np.mean(e2r);  e2s = np.std(e2r)
    Dp = 1.0 / (e1m * e2m) if e1m > 0 and e2m > 0 else np.inf
    ra = Dp / D_ex_2ch if D_ex_2ch > 0 else np.inf

    print(f"  A: ⟨ε₁⟩={e1m:.6f}±{e1s:.2e}  ⟨ε₂⟩={e2m:.6f}±{e2s:.2e}")
    print(f"     D_prod={Dp:.6e}  ratio={ra:.6f}  [{ta:.1f}s]")

    results_a.append(dict(cv=cv, sigma=sig, D_exact=D_ex_2ch,
                          e1m=e1m, e1s=e1s, e2m=e2m, e2s=e2s,
                          D_product=Dp, ratio=ra))
    conv_a.append(dict(cv=cv, e1r=list(e1r), e2r=list(e2r)))

    # Option B
    efr = []
    tb = time.time()
    for tr in range(N_TRAJ):
        ef = sim_lake_core(sig, DT, n_steps, n_burn, X_CL, X_SD,
                            A_P, B_P, R_P, X_MIN)
        efr.append(ef)
    tb = time.time() - tb

    efm = np.mean(efr);  efs = np.std(efr)
    inv_e = 1.0 / efm if efm > 0 else np.inf
    rb = inv_e / D_ex_orig if D_ex_orig > 0 else np.inf

    print(f"  B: ⟨ε_eff⟩={efm:.6f}±{efs:.2e}  1/⟨ε⟩={inv_e:.4f}  ratio={rb:.6f}  [{tb:.1f}s]")

    results_b.append(dict(cv=cv, sigma=sig, D_exact=D_ex_orig,
                          efm=efm, efs=efs, inv_e=inv_e, ratio=rb))
    conv_b.append(dict(cv=cv, efr=list(efr)))

    elapsed = time.time() - t0
    remain = elapsed / (ci + 1) * (len(CVS) - ci - 1)
    print(f"  [elapsed {elapsed/60:.1f}min  remaining ~{remain/60:.1f}min]")

total_time = time.time() - t0
print(f"\n{'='*70}")
print(f"DONE — {total_time/60:.1f} min total")
print(f"{'='*70}")


# ================================================================
# Write results
# ================================================================
out = os.path.join(os.path.dirname(__file__), 'PATHC_DYNAMIC_EPSILON_RESULTS.md')

with open(out, 'w') as f:
    f.write("# Path C: Dynamic ε Test\n\n")
    f.write(f"*Generated {time.strftime('%Y-%m-%d %H:%M')}*  \n")
    f.write(f"*T_max={T_MAX}, dt={DT}, {N_TRAJ} traj/CV, ")
    f.write(f"numba={HAS_NUMBA}, runtime={total_time/60:.1f}min*\n\n")

    # Calibration
    f.write("## Model Calibration\n\n")
    f.write("### Two-channel model\n\n")
    f.write("```\n")
    f.write("f_2ch(x) = a + r·x⁸/(x⁸+1) − b₀·x − c₁·x⁴/(x⁴+K₁⁴) − c₂·x/(x+K₂)\n\n")
    f.write(f"K₁ = {K1}   K₂ = {K2}\n")
    f.write(f"c₁ = {C1:.8f}   c₂ = {C2:.8f}   b₀ = {B0:.4f}\n")
    f.write(f"ε₁_target = {EPS1_T}   ε₂_target = {EPS2_T}\n")
    f.write("```\n\n")

    f.write("### Equilibria comparison\n\n")
    f.write("| Property | Original | Two-channel | Δ (%) |\n")
    f.write("|----------|----------|-------------|-------|\n")
    f.write(f"| x_clear | {X_CL:.6f} | {X_CL2:.8f} | {abs(X_CL2-X_CL)/X_CL*100:.3f} |\n")
    f.write(f"| x_sad | {X_SD:.6f} | {X_SD2:.8f} | {abs(X_SD2-X_SD)/X_SD*100:.3f} |\n")
    if X_TB2:
        f.write(f"| x_turb | {X_TB:.6f} | {X_TB2:.8f} | {abs(X_TB2-X_TB)/X_TB*100:.3f} |\n")
    f.write(f"| λ_clear | {LAM_CL:.6f} | {LAM_CL2:.6f} | {abs(LAM_CL2-LAM_CL)/abs(LAM_CL)*100:.2f} |\n")
    f.write(f"| λ_sad | {LAM_SD:.6f} | {LAM_SD2:.6f} | {abs(LAM_SD2-LAM_SD)/abs(LAM_SD)*100:.2f} |\n")
    f.write(f"| τ | {TAU_L:.6f} | {TAU_2:.6f} | {abs(TAU_2-TAU_L)/TAU_L*100:.2f} |\n")
    f.write(f"| ΔΦ | {DPhi_orig:.8f} | {DPhi_2ch:.8f} | {abs(DPhi_2ch-DPhi_orig)/DPhi_orig*100:.2f} |\n")
    f.write(f"| ε₁(x_eq) | — | {eps1_eq2:.6f} | target: {EPS1_T} |\n")
    f.write(f"| ε₂(x_eq) | — | {eps2_eq2:.6f} | target: {EPS2_T} |\n\n")

    # Results A
    f.write("## Results: Option A (two-channel)\n\n")
    f.write("| CV | σ | D_exact | ⟨ε₁⟩ | ⟨ε₂⟩ | D_product | D_product/D_exact |\n")
    f.write("|------|---------|-----------|---------|---------|-----------|-------------------|\n")
    for r in results_a:
        f.write(f"| {r['cv']:.3f} | {r['sigma']:.5f} | {r['D_exact']:.4e} "
                f"| {r['e1m']:.6f} | {r['e2m']:.6f} | {r['D_product']:.4e} "
                f"| {r['ratio']:.6f} |\n")
    f.write("\n")

    # Results B
    f.write("## Results: Option B (single-channel ratio)\n\n")
    f.write("| CV | σ | D_exact | ⟨ε_eff⟩ | 1/⟨ε_eff⟩ | (1/⟨ε_eff⟩)/D_exact |\n")
    f.write("|------|---------|-----------|---------|-----------|---------------------|\n")
    for r in results_b:
        f.write(f"| {r['cv']:.3f} | {r['sigma']:.5f} | {r['D_exact']:.4e} "
                f"| {r['efm']:.6f} | {r['inv_e']:.4f} | {r['ratio']:.6f} |\n")
    f.write("\n")

    # Convergence
    f.write("## Convergence\n\n")
    f.write("### Option A\n\n")
    f.write("| CV | ε₁ runs | ε₁ CV% | ε₂ runs | ε₂ CV% |\n")
    f.write("|------|---------|--------|---------|--------|\n")
    for c in conv_a:
        e1s = ", ".join(f"{v:.6f}" for v in c['e1r'])
        e2s = ", ".join(f"{v:.6f}" for v in c['e2r'])
        cv1 = np.std(c['e1r'])/np.mean(c['e1r'])*100 if np.mean(c['e1r'])>0 else 0
        cv2 = np.std(c['e2r'])/np.mean(c['e2r'])*100 if np.mean(c['e2r'])>0 else 0
        w1 = " ⚠" if cv1 > 5 else ""
        w2 = " ⚠" if cv2 > 5 else ""
        f.write(f"| {c['cv']:.3f} | {e1s} | {cv1:.2f}%{w1} | {e2s} | {cv2:.2f}%{w2} |\n")

    f.write("\n### Option B\n\n")
    f.write("| CV | ε_eff runs | CV% |\n")
    f.write("|------|------------|-----|\n")
    for c in conv_b:
        es = ", ".join(f"{v:.6f}" for v in c['efr'])
        cvp = np.std(c['efr'])/np.mean(c['efr'])*100 if np.mean(c['efr'])>0 else 0
        w = " ⚠" if cvp > 5 else ""
        f.write(f"| {c['cv']:.3f} | {es} | {cvp:.2f}%{w} |\n")
    f.write("\n")

    # Diagnostic plots
    f.write("## Key Diagnostic Plots (described)\n\n")

    rats = [r['ratio'] for r in results_a]
    max_dev = max(abs(r - 1.0) for r in rats) * 100

    f.write("### 1. D_product vs D_exact across σ (log-log)\n\n")
    Dmin = min(r['D_exact'] for r in results_a)
    Dmax = max(r['D_exact'] for r in results_a)
    f.write(f"D_exact spans [{Dmin:.2e}, {Dmax:.2e}] across CV=[{CVS[0]}, {CVS[-1]}].\n")
    if max_dev < 5:
        f.write(f"All points within {max_dev:.1f}% of y=x diagonal — tight tracking.\n")
    elif max_dev < 50:
        f.write(f"Points scatter around y=x, max deviation {max_dev:.1f}%.\n")
    else:
        f.write(f"Points deviate up to {max_dev:.1f}% from y=x.\n")
    f.write("\n")

    f.write("### 2. ⟨ε₁⟩ and ⟨ε₂⟩ vs σ\n\n")
    e1v = [r['e1m'] for r in results_a]
    e2v = [r['e2m'] for r in results_a]
    f.write(f"⟨ε₁⟩ range: [{min(e1v):.6f}, {max(e1v):.6f}]\n")
    f.write(f"⟨ε₂⟩ range: [{min(e2v):.6f}, {max(e2v):.6f}]\n")
    if e1v[-1] < e1v[0]:
        f.write("Both decrease with increasing σ (noise overwhelms regulation).\n")
    elif e1v[-1] > e1v[0]:
        f.write("Both increase with σ.\n")
    else:
        f.write("Non-monotonic behavior.\n")
    f.write("\n")

    f.write("### 3. ⟨ε⟩ at σ→0 vs calibrated values\n\n")
    f.write(f"At CV={CVS[0]}: ⟨ε₁⟩={results_a[0]['e1m']:.6f} (target {EPS1_T}), "
            f"⟨ε₂⟩={results_a[0]['e2m']:.6f} (target {EPS2_T})\n")
    r1 = abs(results_a[0]['e1m'] - EPS1_T) / EPS1_T < 0.05
    r2 = abs(results_a[0]['e2m'] - EPS2_T) / EPS2_T < 0.05
    f.write(f"Recovery: ε₁ {'YES' if r1 else 'NO'}, ε₂ {'YES' if r2 else 'NO'}\n\n")

    # Verdict
    f.write("## Verdict\n\n")

    if max_dev < 5:
        f.write(f"**D_product(σ) tracks D_exact(σ) across all noise levels "
                f"(max deviation: {max_dev:.1f}%). The identity is dynamic.**\n\n")
    elif max_dev < 20:
        best = CVS[np.argmin([abs(r-1.0) for r in rats])]
        f.write(f"D_product(σ) approximately tracks D_exact(σ) "
                f"(max deviation: {max_dev:.1f}%, best at CV={best:.3f}). "
                f"Partial dynamic character.\n\n")
    else:
        si = CVS.index(0.341) if 0.341 in CVS else np.argmin([abs(r-1.0) for r in rats])
        sd = abs(rats[si] - 1.0) * 100
        f.write(f"D_product(σ) matches D_exact only near σ* (CV≈{CVS[si]:.3f}, "
                f"deviation: {sd:.1f}%). At other noise levels, deviation reaches "
                f"{max_dev:.1f}%. **The identity is static.**\n\n")

    f.write("### Additional observations\n\n")

    f.write("**⟨ε₁⟩ and ⟨ε₂⟩ vs σ:**\n\n")
    for r in results_a:
        f.write(f"- CV={r['cv']:.3f}: ⟨ε₁⟩={r['e1m']:.6f}, ⟨ε₂⟩={r['e2m']:.6f}\n")
    f.write("\n")

    f.write("**D_product/D_exact ratio vs σ:**\n\n")
    for r in results_a:
        f.write(f"- CV={r['cv']:.3f}: ratio={r['ratio']:.6f}\n")

    mean_r = np.mean(rats);  std_r = np.std(rats)
    if std_r / abs(mean_r) < 0.05:
        trend = "approximately constant"
    elif all(rats[i+1] >= rats[i] for i in range(len(rats)-1)):
        trend = "monotonically increasing"
    elif all(rats[i+1] <= rats[i] for i in range(len(rats)-1)):
        trend = "monotonically decreasing"
    else:
        trend = "non-monotonic"
    f.write(f"\nPattern: {trend} (mean={mean_r:.4f}, std={std_r:.4f})\n\n")

    f.write("**σ→0 sanity check:**\n\n")
    d1 = abs(results_a[0]['e1m'] - EPS1_T) / EPS1_T * 100
    d2 = abs(results_a[0]['e2m'] - EPS2_T) / EPS2_T * 100
    f.write(f"- ⟨ε₁⟩ at CV={CVS[0]}: {results_a[0]['e1m']:.6f} vs {EPS1_T} "
            f"({'RECOVERED' if d1<5 else 'NOT recovered'}, Δ={d1:.1f}%)\n")
    f.write(f"- ⟨ε₂⟩ at CV={CVS[0]}: {results_a[0]['e2m']:.6f} vs {EPS2_T} "
            f"({'RECOVERED' if d2<5 else 'NOT recovered'}, Δ={d2:.1f}%)\n")

print(f"\nResults → {out}")
print("Done.")
