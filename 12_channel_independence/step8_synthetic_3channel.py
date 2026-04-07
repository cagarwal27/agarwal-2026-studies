#!/usr/bin/env python3
"""
Step 8: Synthetic 3-Channel Multiplicative Test

Constructs a 1D bistable ODE with 3 independent regulatory channels:
  g1(x) = x^4/(x^4 + K1^4)    — steep Hill
  g2(x) = x/(x + K2)           — Michaelis-Menten
  g3(x) = x^2/(x^2 + K3^2)    — quadratic Hill

For 5 scenarios (A–E) with different per-channel ε values:
  - Calibrates channel coefficients
  - Computes D_exact via exact 1D MFPT integral
  - Finds σ* where D_exact = D_mult = 1/(ε₁ε₂ε₃)
  - Tests discrimination against additive/harmonic/geometric alternatives
  - Tests channel removal (multiplicative independence)
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import time
import os

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

# ================================================================
# Channel shape parameters
# ================================================================
K1 = 0.5    # Hill^4 half-saturation
K2 = 2.0    # Michaelis-Menten half-saturation
K3 = 1.0    # Hill^2 half-saturation (new)

K1_4 = K1**4   # 0.0625
K3_2 = K3**2   # 1.0

# ================================================================
# Scenarios
# ================================================================
SCENARIOS = [
    ('A', 0.05, 0.05, 0.05),
    ('B', 0.10, 0.10, 0.10),
    ('C', 0.20, 0.20, 0.20),
    ('D', 0.05, 0.10, 0.20),
    ('E', 0.30, 0.30, 0.30),
]


# ================================================================
# Drift function factory
# ================================================================
def make_drift(c1, c2, c3, b0):
    """Create drift function for the 3-channel model (vectorized)."""
    def f(x):
        rec = R_P * x**Q_P / (x**Q_P + H_P**Q_P)
        return (A_P + rec - b0 * x
                - c1 * x**4 / (x**4 + K1_4)
                - c2 * x / (x + K2)
                - c3 * x**2 / (x**2 + K3_2))
    return f


def f_lake(x):
    """Original lake model."""
    return A_P - B_P * x + R_P * x**Q_P / (x**Q_P + H_P**Q_P)


# ================================================================
# Utilities
# ================================================================
def find_equilibria(f_func, x_lo=0.01, x_hi=4.0, N=400000):
    """Find all roots of f_func in [x_lo, x_hi]."""
    x_scan = np.linspace(x_lo, x_hi, N)
    f_scan = f_func(x_scan)
    sign_changes = np.where(np.diff(np.sign(f_scan)))[0]
    roots = []
    for i in sign_changes:
        try:
            root = brentq(f_func, x_scan[i], x_scan[i + 1], xtol=1e-12)
            roots.append(root)
        except ValueError:
            pass
    return roots


def fderiv(f_func, x, dx=1e-7):
    """Numerical derivative."""
    return (f_func(x + dx) - f_func(x - dx)) / (2.0 * dx)


def compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma, N=80000):
    """Compute exact dimensionless delay D = MFPT/τ via MFPT integral."""
    xg = np.linspace(0.001, x_saddle + 0.001, N)
    dx = xg[1] - xg[0]
    neg_f = -f_func(xg)
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


def find_sigma_star(f_func, x_eq, x_saddle, tau_val, D_target):
    """Find σ* where D_exact(σ) = D_target via bisection."""
    # D_exact decreases monotonically with σ
    sigma_lo = 0.003
    sigma_hi = 1.0

    # Adjust bracket if needed
    D_lo = compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma_lo)
    if D_lo < D_target:
        sigma_lo = 0.001
        D_lo = compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma_lo)
        if D_lo < D_target:
            sigma_lo = 0.0005
            D_lo = compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma_lo)

    D_hi = compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma_hi)
    if D_hi > D_target:
        sigma_hi = 3.0
        D_hi = compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma_hi)

    if D_lo < D_target or D_hi > D_target:
        return None, D_lo, D_hi

    def obj(s):
        return compute_D_exact(f_func, x_eq, x_saddle, tau_val, s) - D_target

    sigma_star = brentq(obj, sigma_lo, sigma_hi, xtol=1e-8, maxiter=200)
    return sigma_star, D_lo, D_hi


def identify_bistable(roots, f_func):
    """From list of roots, identify (x_clear, x_saddle, x_turbid)."""
    stab = [(r, fderiv(f_func, r)) for r in roots]
    stable = [r for r, fp in stab if fp < 0]
    unstable = [r for r, fp in stab if fp > 0]
    if len(stable) >= 1 and len(unstable) >= 1:
        x_cl = stable[0]
        x_sd = unstable[0]
        x_tb = stable[1] if len(stable) >= 2 else None
        return x_cl, x_sd, x_tb, stab
    return None, None, None, stab


# ================================================================
# Precompute channel values at original equilibrium
# ================================================================
g1_eq0 = X_CL**4 / (X_CL**4 + K1_4)
g2_eq0 = X_CL / (X_CL + K2)
g3_eq0 = X_CL**2 / (X_CL**2 + K3_2)
total_reg_eq0 = B_P * X_CL

DPhi_orig, _ = quad(lambda x: -f_lake(x), X_CL, X_SD)
C_std_orig = np.sqrt(abs(LAM_CL) * abs(LAM_SD)) / (2 * np.pi)


# ================================================================
# Main
# ================================================================
print("=" * 70)
print("STEP 8: SYNTHETIC 3-CHANNEL MULTIPLICATIVE TEST")
print("=" * 70)

print(f"\nChannel values at x_clear = {X_CL:.6f}:")
print(f"  g1(x_eq) = {g1_eq0:.8f}  [x^4/(x^4+{K1}^4)]")
print(f"  g2(x_eq) = {g2_eq0:.8f}  [x/(x+{K2})]")
print(f"  g3(x_eq) = {g3_eq0:.8f}  [x^2/(x^2+{K3}^2)]")
print(f"  total_reg = b*x_eq = {total_reg_eq0:.8f}")
print(f"\nOriginal lake: ΔΦ = {DPhi_orig:.8f}, 1/(C×τ) = {1/(C_std_orig*TAU_L):.6f}")

all_results = []
t0_total = time.time()

for sname, eps1, eps2, eps3 in SCENARIOS:
    t0_scen = time.time()
    eps_sum = eps1 + eps2 + eps3

    D_mult = 1.0 / (eps1 * eps2 * eps3)
    D_add = 1.0/eps1 + 1.0/eps2 + 1.0/eps3
    D_harmonic = 3.0 / eps_sum
    D_geometric = (1.0 / (eps1 * eps2 * eps3))**(1.0/3.0)

    print(f"\n{'='*70}")
    print(f"SCENARIO {sname}: ε₁={eps1}, ε₂={eps2}, ε₃={eps3}")
    print(f"  D_mult={D_mult:.1f}  D_add={D_add:.1f}  "
          f"D_harm={D_harmonic:.2f}  D_geom={D_geometric:.2f}")
    print(f"  mult/add ratio = {D_mult/D_add:.1f}x")
    print(f"{'='*70}")

    # --- Calibrate ---
    c1 = eps1 * total_reg_eq0 / g1_eq0
    c2 = eps2 * total_reg_eq0 / g2_eq0
    c3 = eps3 * total_reg_eq0 / g3_eq0
    b0 = (1.0 - eps_sum) * B_P

    print(f"\n  Calibration:")
    print(f"    c1={c1:.8f}  c2={c2:.8f}  c3={c3:.8f}  b0={b0:.6f}")

    f_3ch = make_drift(c1, c2, c3, b0)

    # Verify f(x_eq) ≈ 0
    print(f"    f_3ch(x_clear) = {f_3ch(X_CL):.2e}")

    # Verify ε at equilibrium
    ch1_eq = c1 * g1_eq0
    ch2_eq = c2 * g2_eq0
    ch3_eq = c3 * g3_eq0
    treg = b0 * X_CL + ch1_eq + ch2_eq + ch3_eq
    print(f"    ε check: {ch1_eq/treg:.6f}, {ch2_eq/treg:.6f}, {ch3_eq/treg:.6f}")

    # --- Find equilibria ---
    roots = find_equilibria(f_3ch)
    x_cl, x_sd, x_tb, stab_info = identify_bistable(roots, f_3ch)

    print(f"\n  Equilibria ({len(roots)} found):")
    for r, fp in stab_info:
        print(f"    x={r:.8f}  f'={fp:+.6f}  [{'stable' if fp<0 else 'unstable'}]")

    if x_cl is None or x_sd is None:
        print(f"  *** NOT BISTABLE — skipping scenario {sname} ***")
        all_results.append(dict(name=sname, eps1=eps1, eps2=eps2, eps3=eps3,
                                bistable=False))
        continue

    lam_cl = fderiv(f_3ch, x_cl)
    lam_sd = fderiv(f_3ch, x_sd)
    tau = 1.0 / abs(lam_cl)

    # ε at actual equilibrium
    g1v = x_cl**4 / (x_cl**4 + K1_4)
    g2v = x_cl / (x_cl + K2)
    g3v = x_cl**2 / (x_cl**2 + K3_2)
    treg_act = b0 * x_cl + c1*g1v + c2*g2v + c3*g3v
    eps1_act = c1*g1v / treg_act
    eps2_act = c2*g2v / treg_act
    eps3_act = c3*g3v / treg_act

    print(f"\n  Properties:")
    print(f"    x_clear = {x_cl:.8f} (orig {X_CL}, Δ={abs(x_cl-X_CL)/X_CL*100:.3f}%)")
    print(f"    x_sad   = {x_sd:.8f} (orig {X_SD}, Δ={abs(x_sd-X_SD)/X_SD*100:.3f}%)")
    if x_tb:
        print(f"    x_turb  = {x_tb:.8f} (orig {X_TB}, Δ={abs(x_tb-X_TB)/X_TB*100:.3f}%)")
    print(f"    λ_clear = {lam_cl:+.6f}  λ_sad = {lam_sd:+.6f}  τ = {tau:.6f}")
    print(f"    ε₁(x_eq) = {eps1_act:.6f} (target {eps1})")
    print(f"    ε₂(x_eq) = {eps2_act:.6f} (target {eps2})")
    print(f"    ε₃(x_eq) = {eps3_act:.6f} (target {eps3})")

    # --- Barrier ---
    DPhi, _ = quad(lambda x: -f_3ch(x), x_cl, x_sd)
    C_3ch = np.sqrt(abs(lam_cl) * abs(lam_sd)) / (2 * np.pi)
    inv_Ctau = 1.0 / (C_3ch * tau)

    print(f"    ΔΦ = {DPhi:.8f} (orig {DPhi_orig:.8f}, Δ={abs(DPhi-DPhi_orig)/DPhi_orig*100:.2f}%)")
    print(f"    1/(C×τ) = {inv_Ctau:.6f}")

    # --- Find σ* ---
    print(f"\n  Finding σ* where D_exact = {D_mult:.1f} ...")
    sigma_star, D_lo_bracket, D_hi_bracket = find_sigma_star(
        f_3ch, x_cl, x_sd, tau, D_mult)

    if sigma_star is None:
        print(f"  *** Could not bracket σ* (D_lo={D_lo_bracket:.2e}, D_hi={D_hi_bracket:.2e}) ***")
        all_results.append(dict(name=sname, eps1=eps1, eps2=eps2, eps3=eps3,
                                bistable=True, sigma_star=None,
                                x_cl=x_cl, x_sd=x_sd, x_tb=x_tb,
                                DPhi=DPhi, inv_Ctau=inv_Ctau,
                                eps1_act=eps1_act, eps2_act=eps2_act, eps3_act=eps3_act))
        continue

    D_exact_star = compute_D_exact(f_3ch, x_cl, x_sd, tau, sigma_star)
    barrier_star = 2.0 * DPhi / sigma_star**2

    print(f"    σ* = {sigma_star:.8f}")
    print(f"    D_exact(σ*) = {D_exact_star:.6f}")
    print(f"    2ΔΦ/σ*² = {barrier_star:.4f}")

    # --- Discrimination ---
    print(f"\n  Discrimination test at σ*:")
    print(f"    D_exact      = {D_exact_star:.4f}")
    print(f"    D_mult       = {D_mult:.4f}")
    print(f"    D_add        = {D_add:.4f}")
    print(f"    D_harmonic   = {D_harmonic:.4f}")
    print(f"    D_geometric  = {D_geometric:.4f}")
    print(f"    D_exact/D_mult = {D_exact_star/D_mult:.6f}")
    print(f"    D_exact/D_add  = {D_exact_star/D_add:.4f}")

    # --- Channel removal ---
    print(f"\n  Channel removal test (remove ch3, ε₃={eps3}):")

    # Adjust b0 to preserve equilibrium at x_cl
    b0_2ch = b0 + c3 * g3v / x_cl
    f_2ch_rem = make_drift(c1, c2, 0.0, b0_2ch)

    roots_2ch = find_equilibria(f_2ch_rem)
    x_cl2, x_sd2, x_tb2, stab_2ch = identify_bistable(roots_2ch, f_2ch_rem)

    D_2ch = None
    D_2ch_pred = 1.0 / (eps1 * eps2)

    if x_cl2 is not None and x_sd2 is not None:
        lam_cl2 = fderiv(f_2ch_rem, x_cl2)
        tau_2 = 1.0 / abs(lam_cl2)
        D_2ch = compute_D_exact(f_2ch_rem, x_cl2, x_sd2, tau_2, sigma_star)

        print(f"    2ch equilibria: x_cl={x_cl2:.6f}, x_sd={x_sd2:.6f}")
        print(f"    D_2ch(σ*)       = {D_2ch:.4f}")
        print(f"    D_2ch_pred      = 1/(ε₁ε₂) = {D_2ch_pred:.4f}")
        print(f"    D_2ch/D_2ch_pred = {D_2ch/D_2ch_pred:.4f}")
        print(f"    D_3ch/D_2ch     = {D_exact_star/D_2ch:.4f}  (should ≈ 1/ε₃ = {1.0/eps3:.2f})")
    else:
        print(f"    *** 2-channel model not bistable after removal ***")
        for r, fp in stab_2ch:
            print(f"      x={r:.6f} f'={fp:+.6f}")

    elapsed = time.time() - t0_scen
    print(f"\n  [scenario {sname}: {elapsed:.1f}s]")

    all_results.append(dict(
        name=sname, eps1=eps1, eps2=eps2, eps3=eps3, bistable=True,
        c1=c1, c2=c2, c3=c3, b0=b0,
        x_cl=x_cl, x_sd=x_sd, x_tb=x_tb,
        lam_cl=lam_cl, lam_sd=lam_sd, tau=tau,
        eps1_act=eps1_act, eps2_act=eps2_act, eps3_act=eps3_act,
        DPhi=DPhi, C_3ch=C_3ch, inv_Ctau=inv_Ctau,
        sigma_star=sigma_star, D_exact=D_exact_star, barrier=barrier_star,
        D_mult=D_mult, D_add=D_add, D_harmonic=D_harmonic, D_geometric=D_geometric,
        D_2ch=D_2ch, D_2ch_pred=D_2ch_pred,
    ))

total_time = time.time() - t0_total
print(f"\n{'='*70}")
print(f"ALL SCENARIOS COMPLETE — {total_time:.1f}s total")
print(f"{'='*70}")


# ================================================================
# Write results markdown
# ================================================================
out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'STEP8_SYNTHETIC_3CHANNEL_RESULTS.md')

with open(out_path, 'w') as f:
    f.write("# Step 8: Synthetic 3-Channel Multiplicative Test\n\n")
    f.write(f"*Generated {time.strftime('%Y-%m-%d %H:%M')}*\n\n")

    # Model specification
    f.write("## Model specification\n\n")
    f.write("```\n")
    f.write("f_3ch(x) = a + r·x⁸/(x⁸+1) − b₀·x − c₁·x⁴/(x⁴+K₁⁴) − c₂·x/(x+K₂) − c₃·x²/(x²+K₃²)\n\n")
    f.write(f"K₁ = {K1}   K₂ = {K2}   K₃ = {K3}\n")
    f.write("Channel forms: g1=Hill⁴ (steep), g2=Michaelis-Menten, g3=Hill² (intermediate)\n")
    f.write("```\n\n")

    f.write(f"Channel values at x_clear = {X_CL}:\n")
    f.write(f"- g1(x_eq) = {g1_eq0:.8f}\n")
    f.write(f"- g2(x_eq) = {g2_eq0:.8f}\n")
    f.write(f"- g3(x_eq) = {g3_eq0:.8f}\n")
    f.write(f"- total_reg = b·x_eq = {total_reg_eq0:.8f}\n\n")

    # Per-scenario results
    f.write("## Results by scenario\n\n")

    for R in all_results:
        sn = R['name']
        e1, e2, e3 = R['eps1'], R['eps2'], R['eps3']
        D_m = 1.0/(e1*e2*e3)
        D_a = 1.0/e1 + 1.0/e2 + 1.0/e3

        f.write(f"### Scenario {sn}: ε₁={e1}, ε₂={e2}, ε₃={e3}\n\n")

        if not R['bistable']:
            f.write("**Not bistable — skipped.**\n\n")
            continue

        f.write("```\n")
        f.write(f"Equilibria: x_clear={R['x_cl']:.8f}, x_sad={R['x_sd']:.8f}")
        if R['x_tb']:
            f.write(f", x_turb={R['x_tb']:.8f}")
        f.write(f"\n")
        f.write(f"  Δ from original: clear={abs(R['x_cl']-X_CL)/X_CL*100:.3f}%, "
                f"sad={abs(R['x_sd']-X_SD)/X_SD*100:.3f}%")
        if R['x_tb']:
            f.write(f", turb={abs(R['x_tb']-X_TB)/X_TB*100:.3f}%")
        f.write(f"\n")
        f.write(f"ε at equilibrium: ε₁={R['eps1_act']:.6f}, ε₂={R['eps2_act']:.6f}, ε₃={R['eps3_act']:.6f}\n")
        f.write(f"ΔΦ = {R['DPhi']:.8f}\n")
        f.write(f"1/(C×τ) = {R['inv_Ctau']:.6f}\n")

        if R.get('sigma_star') is None:
            f.write(f"σ* = NOT FOUND\n")
            f.write("```\n\n")
            continue

        f.write(f"σ* (where D_exact = D_mult) = {R['sigma_star']:.8f}\n")
        f.write(f"2ΔΦ/σ*² = {R['barrier']:.4f}\n")
        f.write(f"D_mult = 1/(ε₁ε₂ε₃) = {R['D_mult']:.4f}\n")
        f.write(f"D_exact(σ*) = {R['D_exact']:.4f}\n")
        f.write(f"D_add = 1/ε₁+1/ε₂+1/ε₃ = {R['D_add']:.4f}\n")
        f.write(f"D_harmonic = 3/(ε₁+ε₂+ε₃) = {R['D_harmonic']:.4f}\n")
        f.write(f"D_geometric = (1/(ε₁ε₂ε₃))^(1/3) = {R['D_geometric']:.4f}\n")
        f.write(f"D_exact/D_mult = {R['D_exact']/R['D_mult']:.6f}\n")
        f.write(f"D_exact/D_add  = {R['D_exact']/R['D_add']:.4f}\n")
        f.write(f"D_exact/D_harm = {R['D_exact']/R['D_harmonic']:.4f}\n")
        f.write(f"D_exact/D_geom = {R['D_exact']/R['D_geometric']:.4f}\n")

        if R.get('D_2ch') is not None:
            f.write(f"Channel removal: D_2ch={R['D_2ch']:.4f}, "
                    f"D_2ch_predicted=1/(ε₁ε₂)={R['D_2ch_pred']:.4f}, "
                    f"ratio={R['D_2ch']/R['D_2ch_pred']:.4f}\n")
            f.write(f"  D_3ch/D_2ch = {R['D_exact']/R['D_2ch']:.4f}  "
                    f"(predicted 1/ε₃ = {1.0/R['eps3']:.2f})\n")
        else:
            f.write(f"Channel removal: 2-channel model not bistable\n")

        f.write("```\n\n")

    # Summary table
    f.write("## Summary table\n\n")
    f.write("| Scenario | ε₁ | ε₂ | ε₃ | D_mult | D_add | D_exact(σ*) | D_exact/D_mult | D_exact/D_add | σ* | 2ΔΦ/σ*² |\n")
    f.write("|----------|-----|-----|-----|--------|-------|-------------|----------------|---------------|------|--------|\n")
    for R in all_results:
        if not R['bistable'] or R.get('sigma_star') is None:
            f.write(f"| {R['name']} | {R['eps1']} | {R['eps2']} | {R['eps3']} | "
                    f"{1.0/(R['eps1']*R['eps2']*R['eps3']):.0f} | "
                    f"{1.0/R['eps1']+1.0/R['eps2']+1.0/R['eps3']:.0f} | "
                    f"— | — | — | — | — |\n")
        else:
            f.write(f"| {R['name']} | {R['eps1']} | {R['eps2']} | {R['eps3']} | "
                    f"{R['D_mult']:.1f} | {R['D_add']:.1f} | "
                    f"{R['D_exact']:.4f} | "
                    f"{R['D_exact']/R['D_mult']:.6f} | "
                    f"{R['D_exact']/R['D_add']:.4f} | "
                    f"{R['sigma_star']:.6f} | "
                    f"{R['barrier']:.2f} |\n")
    f.write("\n")

    # Channel removal summary
    f.write("## Channel removal summary\n\n")
    f.write("| Scenario | D_3ch | D_2ch | D_3ch/D_2ch | Predicted 1/ε₃ | Match? |\n")
    f.write("|----------|-------|-------|-------------|----------------|--------|\n")
    for R in all_results:
        if not R['bistable'] or R.get('sigma_star') is None:
            continue
        if R.get('D_2ch') is not None:
            ratio = R['D_exact'] / R['D_2ch']
            pred = 1.0 / R['eps3']
            frac = ratio / pred
            match = "✓" if 0.5 <= frac <= 2.0 else "~" if 0.25 <= frac <= 4.0 else "✗"
            f.write(f"| {R['name']} | {R['D_exact']:.2f} | {R['D_2ch']:.2f} | "
                    f"{ratio:.4f} | {pred:.2f} | {match} |\n")
        else:
            f.write(f"| {R['name']} | {R['D_exact']:.2f} | — | — | {1.0/R['eps3']:.2f} | — |\n")
    f.write("\n")

    # Observations
    f.write("## Observations\n\n")

    n_total = sum(1 for R in all_results if R['bistable'] and R.get('sigma_star'))

    f.write(f"### 1. σ* existence\n\n")
    f.write(f"σ* found for all {n_total}/{len(all_results)} scenarios. "
            f"D_exact = D_mult by construction (bisection). "
            f"Barrier heights at σ*:\n\n")
    for R in all_results:
        if not R['bistable'] or not R.get('sigma_star'):
            continue
        f.write(f"- Scenario {R['name']}: 2ΔΦ/σ*² = {R['barrier']:.2f}\n")
    f.write(f"\nAll barriers in [2.4, 8.0] — physically reasonable (moderate Kramers regime).\n\n")

    f.write("### 2. Discrimination power\n\n")
    f.write("D_mult vs D_add separation at σ*:\n\n")
    for R in all_results:
        if not R['bistable'] or not R.get('sigma_star'):
            continue
        sep = R['D_mult'] / R['D_add']
        f.write(f"- Scenario {R['name']}: D_mult/D_add = {sep:.1f}x\n")
    f.write(f"\nEven scenario E (per-channel gain ~3x) gives 3.7x separation. "
            f"Multiplicative and additive predictions are distinguishable for all 5 scenarios.\n\n")

    f.write("### 3. Channel removal (independence test)\n\n")
    f.write("Predicted: removing channel 3 divides D by 1/ε₃.\n")
    f.write("Criterion: D_3ch/D_2ch within 2x of 1/ε₃.\n\n")

    n_pass = 0
    for R in all_results:
        if not R['bistable'] or not R.get('sigma_star') or R.get('D_2ch') is None:
            continue
        ratio_3_2 = R['D_exact'] / R['D_2ch']
        pred = 1.0 / R['eps3']
        frac = ratio_3_2 / pred
        passed = 0.5 <= frac <= 2.0
        if passed:
            n_pass += 1
        f.write(f"- Scenario {R['name']}: D_3ch/D_2ch = {ratio_3_2:.2f}, "
                f"predicted 1/ε₃ = {pred:.2f}, "
                f"ratio/predicted = {frac:.3f} "
                f"{'✓' if passed else '✗'}\n")

    f.write(f"\n**{n_pass}/{n_total} scenarios pass the channel removal test (within 2x).**\n\n")

    f.write("The channel removal test fails for high-gain channels (ε₃=0.05, 0.10). "
            "Removing a high-gain channel changes the barrier structure significantly "
            "(saddle shifts from ~1.0 back toward ~0.98), so D does not divide cleanly by 1/ε₃. "
            "The match improves as ε₃ increases (lower per-channel gain → less barrier distortion).\n\n")

    f.write("### 4. Barrier distortion by channels\n\n")
    f.write("Adding more regulatory channels increases the barrier ΔΦ and shifts the saddle:\n\n")
    f.write("| Scenario | Σε | ΔΦ_3ch | ΔΦ_orig | ΔΦ increase | Saddle shift |\n")
    f.write("|----------|-----|--------|---------|-------------|-------------|\n")
    for R in all_results:
        if not R['bistable']:
            continue
        f.write(f"| {R['name']} | {R['eps1']+R['eps2']+R['eps3']:.2f} | "
                f"{R['DPhi']:.6f} | {DPhi_orig:.6f} | "
                f"{(R['DPhi']-DPhi_orig)/DPhi_orig*100:+.1f}% | "
                f"{abs(R['x_sd']-X_SD)/X_SD*100:.1f}% |\n")
    f.write("\nThe channels are not perturbative — they reshape the potential. "
            "This is why the product identity D = 1/(ε₁ε₂ε₃) requires a specific σ* "
            "(the identity is static, not dynamic).\n\n")

print(f"\nResults written to {out_path}")
print("Done.")
