#!/usr/bin/env python3
"""
Test A (fast): Can D go below 1?

Uses the 2-channel lake model. Sweeps ε past 1.0.
Reduced grid sizes for speed — accuracy to ~1% is fine for this question.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# Base lake model
# ================================================================
A_P = 0.326588
B_P = 0.8
R_P = 1.0
Q_P = 8
H_P = 1.0

X_CL = 0.409217
X_SD = 0.978152
LAM_CL = -0.784651
TAU_L = 1.0 / abs(LAM_CL)

K1 = 0.5
K2 = 2.0
K1_4 = K1**4

def g1(x): return x**4 / (x**4 + K1_4)
def g2(x): return x / (x + K2)


def make_drift(c1, c2, b0):
    def f(x):
        rec = R_P * x**Q_P / (x**Q_P + H_P**Q_P)
        return A_P + rec - b0 * x - c1 * g1(x) - c2 * g2(x)
    return f


def find_eq(f_func, x_lo=0.01, x_hi=4.0, N=50000):
    xs = np.linspace(x_lo, x_hi, N)
    fs = np.array([f_func(x) for x in xs])
    sc = np.where(np.diff(np.sign(fs)))[0]
    roots = []
    for i in sc:
        try:
            roots.append(brentq(f_func, xs[i], xs[i+1], xtol=1e-10))
        except:
            pass
    return roots


def fderiv(f, x, dx=1e-7):
    return (f(x+dx) - f(x-dx)) / (2*dx)


def D_exact(f_func, x_eq, x_sad, tau, sigma, N=20000):
    xg = np.linspace(0.001, x_sad + 0.001, N)
    dx = xg[1] - xg[0]
    neg_f = np.array([-f_func(x) for x in xg])
    U = np.cumsum(neg_f) * dx
    i_eq = np.argmin(np.abs(xg - x_eq))
    U -= U[i_eq]
    Phi = 2.0 * U / sigma**2
    exp_neg = np.exp(-Phi)
    Ix = np.cumsum(exp_neg) * dx
    psi = (2.0 / sigma**2) * np.exp(Phi) * Ix
    i_sad = np.argmin(np.abs(xg - x_sad))
    if i_eq >= i_sad:
        return np.nan
    return np.trapz(psi[i_eq:i_sad+1], xg[i_eq:i_sad+1]) / tau


def find_sigma_star(f_func, x_eq, x_sad, tau, D_tgt):
    try:
        def obj(s): return D_exact(f_func, x_eq, x_sad, tau, s) - D_tgt
        # Quick bracket check
        d_lo = D_exact(f_func, x_eq, x_sad, tau, 0.003)
        d_hi = D_exact(f_func, x_eq, x_sad, tau, 2.0)
        if np.isnan(d_lo) or np.isnan(d_hi): return None
        if d_lo < D_tgt or d_hi > D_tgt: return None
        return brentq(obj, 0.003, 2.0, xtol=1e-6, maxiter=100)
    except:
        return None


def calibrate(eps1, eps2):
    total = B_P * X_CL
    c1 = eps1 * total / g1(X_CL)
    c2 = eps2 * total / g2(X_CL)
    b0 = (total - c1*g1(X_CL) - c2*g2(X_CL)) / X_CL
    return c1, c2, b0


def analyze(eps1, eps2, label=""):
    """Full analysis for one (ε₁, ε₂) pair. Returns dict."""
    D_prod = 1.0 / (eps1 * eps2)
    c1, c2, b0 = calibrate(eps1, eps2)

    result = {'eps1': eps1, 'eps2': eps2, 'D_prod': D_prod, 'b0': b0}

    if b0 < 0:
        result['status'] = 'b0 < 0'
        return result

    f = make_drift(c1, c2, b0)
    roots = find_eq(f)

    if len(roots) < 3:
        result['status'] = f'MONO ({len(roots)} roots)'
        result['roots'] = roots
        return result

    stab = [(r, fderiv(f, r)) for r in roots]
    stable = [r for r, fp in stab if fp < 0]
    unstable = [r for r, fp in stab if fp > 0]

    if not stable or not unstable:
        result['status'] = 'no stable+saddle'
        return result

    x_eq = stable[0]
    x_sad = unstable[0]
    lam_eq = fderiv(f, x_eq)
    lam_sad = fderiv(f, x_sad)
    tau = 1.0 / abs(lam_eq)

    DPhi, _ = quad(lambda x: -f(x), x_eq, x_sad)

    result.update({
        'status': 'BISTABLE',
        'x_eq': x_eq, 'x_sad': x_sad,
        'lam_eq': lam_eq, 'lam_sad': lam_sad,
        'tau': tau, 'DPhi': DPhi,
        'f_func': f,
    })

    if DPhi <= 0:
        result['status'] = 'BISTABLE (ΔΦ ≤ 0)'
        return result

    # Find σ*
    ss = find_sigma_star(f, x_eq, x_sad, tau, D_prod)
    if ss is not None:
        result['sigma_star'] = ss
        result['noise_frac'] = ss / (x_sad - x_eq)

    # D at several fixed noise levels
    for sig in [0.05, 0.10, 0.20, 0.50]:
        d = D_exact(f, x_eq, x_sad, tau, sig)
        result[f'D_at_sig{sig}'] = d

    return result


# ================================================================
# RUN
# ================================================================
print("=" * 70)
print("TEST A: CAN D GO BELOW 1?")
print("=" * 70)

# ---- PART 1: Sweep ε₁, fixed ε₂ = 0.10 ----
print("\n" + "-" * 70)
print("PART 1: Sweep ε₁ (ε₂ = 0.10 fixed)")
print("-" * 70)

eps2 = 0.10
eps1_vals = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.87, 0.89, 0.90, 0.91, 0.92, 0.95]

print(f"\n{'ε₁':>5} {'D_prod':>7} {'b0':>7} {'status':>12} {'x_eq':>7} {'x_sad':>7} {'ΔΦ':>11} {'σ*':>8} {'D@σ=0.1':>8} {'D@σ=0.5':>8}")

for e1 in eps1_vals:
    r = analyze(e1, eps2)
    if r['status'] == 'BISTABLE':
        ss = r.get('sigma_star', None)
        ss_str = f"{ss:.5f}" if ss else "N/A"
        d01 = r.get('D_at_sig0.1', float('nan'))
        d05 = r.get('D_at_sig0.5', float('nan'))
        print(f"{e1:5.2f} {r['D_prod']:7.1f} {r['b0']:7.4f} {'BISTABLE':>12} {r['x_eq']:7.4f} {r['x_sad']:7.4f} {r['DPhi']:11.3e} {ss_str:>8} {d01:8.2f} {d05:8.2f}")
    elif r['status'].startswith('BISTABLE'):
        print(f"{e1:5.2f} {r['D_prod']:7.1f} {r['b0']:7.4f} {r['status']:>12}")
    else:
        print(f"{e1:5.2f} {r['D_prod']:7.1f} {r['b0']:7.4f} {r['status']:>25}")


# ---- PART 2: Both channels large ----
print("\n\n" + "-" * 70)
print("PART 2: Both channels large — can ∏εᵢ > 1 with bistability?")
print("-" * 70)

pairs = [
    (0.40, 0.40), (0.50, 0.50), (0.60, 0.60), (0.70, 0.70),
    (0.75, 0.75), (0.80, 0.80), (0.85, 0.85), (0.90, 0.90),
    (0.80, 0.90), (0.85, 0.90), (0.90, 0.95),
]

print(f"\n{'ε₁':>5} {'ε₂':>5} {'ε₁ε₂':>7} {'D_prod':>7} {'status':>12} {'ΔΦ':>11} {'D@σ=0.1':>8} {'D@σ=0.5':>8}")

for e1, e2 in pairs:
    r = analyze(e1, e2)
    if r['status'] == 'BISTABLE':
        d01 = r.get('D_at_sig0.1', float('nan'))
        d05 = r.get('D_at_sig0.5', float('nan'))
        print(f"{e1:5.2f} {e2:5.2f} {e1*e2:7.4f} {r['D_prod']:7.2f} {'BISTABLE':>12} {r['DPhi']:11.3e} {d01:8.2f} {d05:8.2f}")
    else:
        print(f"{e1:5.2f} {e2:5.2f} {e1*e2:7.4f} {r['D_prod']:7.2f} {r['status']:>25}")


# ---- PART 3: Fine sweep near the critical ε ----
print("\n\n" + "-" * 70)
print("PART 3: Fine sweep — where exactly does bistability die?")
print("-" * 70)

print(f"\n{'ε₁':>6} {'b0':>8} {'#roots':>7} {'Δx':>8} {'ΔΦ':>12} {'|λ_eq|':>8}")

for e1 in np.arange(0.70, 0.96, 0.01):
    c1, c2, b0 = calibrate(e1, 0.10)
    if b0 < 0:
        print(f"{e1:6.2f} {b0:8.4f}    b0 < 0")
        continue
    f = make_drift(c1, c2, b0)
    roots = find_eq(f)
    if len(roots) < 3:
        print(f"{e1:6.2f} {b0:8.4f} {len(roots):>7d}    bistability lost")
        continue
    stab = [(r, fderiv(f, r)) for r in roots]
    stable = [r for r, fp in stab if fp < 0]
    unstable = [r for r, fp in stab if fp > 0]
    if not stable or not unstable:
        print(f"{e1:6.2f} {b0:8.4f} {len(roots):>7d}    no saddle")
        continue
    x_eq, x_sad = stable[0], unstable[0]
    DPhi, _ = quad(lambda x: -f(x), x_eq, x_sad)
    lam = abs(fderiv(f, x_eq))
    print(f"{e1:6.2f} {b0:8.4f} {len(roots):>7d} {x_sad-x_eq:8.4f} {DPhi:12.4e} {lam:8.4f}")


# ---- PART 4: D_exact at high noise ----
print("\n\n" + "-" * 70)
print("PART 4: Minimum achievable D_exact (high noise sweep)")
print("-" * 70)
print("What is the LOWEST D_exact you can get in a bistable system?")
print("(D_exact at very high noise, just before noise destroys the well)")

# Use the standard lake (ε₁=0.05, ε₂=0.10)
r_base = analyze(0.05, 0.10)
f_base = r_base['f_func']
x_eq_b = r_base['x_eq']
x_sad_b = r_base['x_sad']
tau_b = r_base['tau']

print(f"\nStandard lake: x_eq={x_eq_b:.4f}, x_sad={x_sad_b:.4f}, ΔΦ={r_base['DPhi']:.4e}")
print(f"\n{'σ':>8} {'D_exact':>10} {'MFPT (yr)':>10} {'D<1?':>6}")

for sig in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.0, 1.5, 2.0, 3.0, 5.0]:
    d = D_exact(f_base, x_eq_b, x_sad_b, tau_b, sig)
    if not np.isnan(d):
        mfpt = d * tau_b
        below = "YES!" if d < 1.0 else "no"
        print(f"{sig:8.2f} {d:10.4f} {mfpt:10.2f} {below:>6}")


# Also: high-ε system with barrier still present
print("\n--- Same but for ε₁=0.85, ε₂=0.10 (marginal system) ---")
r_marg = analyze(0.85, 0.10)
if r_marg['status'] == 'BISTABLE' and r_marg['DPhi'] > 0:
    f_m = r_marg['f_func']
    x_eq_m, x_sad_m, tau_m = r_marg['x_eq'], r_marg['x_sad'], r_marg['tau']
    print(f"x_eq={x_eq_m:.4f}, x_sad={x_sad_m:.4f}, ΔΦ={r_marg['DPhi']:.4e}")
    print(f"\n{'σ':>8} {'D_exact':>10} {'D<1?':>6}")
    for sig in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0]:
        d = D_exact(f_m, x_eq_m, x_sad_m, tau_m, sig)
        if not np.isnan(d):
            below = "YES!" if d < 1.0 else "no"
            print(f"{sig:8.2f} {d:10.4f} {below:>6}")
else:
    print(f"Status: {r_marg['status']}")


# ================================================================
print("\n\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("""
The key finding is whether D_exact can go below 1.0 in any configuration:

CASE A: Bistability dies before D_product reaches 1
  → The saddle-node bifurcation eliminates the state at some ε_crit < 1
  → D ∈ [1, ∞) is the natural domain
  → D < 1 has no meaning in the product equation: the state doesn't exist

CASE B: Bistability survives but D_exact never goes below 1
  → Even at extreme noise, the MFPT exceeds τ_relax
  → D ∈ [1, ∞) enforced by the Kramers structure itself

CASE C: D_exact goes below 1 at high noise
  → The system escapes before it relaxes
  → D < 1 is physically meaningful: "exists but self-destructs"
  → The well is real but the noise overwhelms it

The product equation prediction D_product < 1 (when ∏εᵢ > 1) is a
separate question from D_exact < 1. The duality D_exact = D_product
only holds at σ*. At high noise, D_exact → 1 from above (or below?).
""")
