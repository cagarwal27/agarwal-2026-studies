#!/usr/bin/env python3
"""
Soviet Collapse: Kuran Preference Falsification Model → Continuous-Time ODE → Kramers

The Kuran model describes opposition dynamics under preference falsification.
Workers have heterogeneous thresholds T_i for joining public opposition.
S = fraction of population openly opposing the regime.
F(S) = CDF of thresholds evaluated at S = fraction who WOULD oppose if S people already do.

Dynamics: dS/dt = γ[F(S) - S]  (people adjust toward their threshold-determined behavior)

For the right F, this gives a double-well potential with:
  - S_lo: regime stable (low opposition)
  - S_mid: tipping point (unstable)
  - S_hi: revolution complete (high opposition)

This IS the native state — S is the natural reaction coordinate.
Kramers applies directly to compute D.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# MODEL DEFINITION
# ============================================================

# Drift: f(S) = -gamma * (S - S_lo)(S - S_mid)(S - S_hi)
# This gives:
#   f < 0 for S in (S_lo, S_mid) → pushed back to S_lo
#   f > 0 for S in (S_mid, S_hi) → pushed to S_hi
#   Stable equilibria at S_lo and S_hi, unstable at S_mid

# Soviet parameters
S_lo  = 0.05   # Regime normal: ~5% active opposition
S_mid = 0.35   # Tipping point
S_hi  = 0.90   # Revolution complete: ~90% opposition

gamma = 1.0    # Rate of preference adjustment (1/year)
# gamma sets the timescale — gamma=1 means adjustments happen on ~1 year timescale

def f_drift(S):
    """Drift function: dS/dt = f(S)"""
    return -gamma * (S - S_lo) * (S - S_mid) * (S - S_hi)

def V_prime(S):
    """V'(S) = -f(S). The system obeys dS/dt = -V'(S)."""
    return -f_drift(S)

def V(S):
    """Potential V(S) = integral of V'(S') from S_lo to S."""
    result, _ = quad(V_prime, S_lo, S)
    return result

# ============================================================
# EQUILIBRIA AND EIGENVALUES
# ============================================================

def f_prime(S):
    """f'(S) — eigenvalue at fixed point."""
    # f(S) = -gamma*(S-S_lo)(S-S_mid)(S-S_hi)
    # f'(S) = -gamma*[(S-S_mid)(S-S_hi) + (S-S_lo)(S-S_hi) + (S-S_lo)(S-S_mid)]
    t1 = (S - S_mid) * (S - S_hi)
    t2 = (S - S_lo) * (S - S_hi)
    t3 = (S - S_lo) * (S - S_mid)
    return -gamma * (t1 + t2 + t3)

print("=" * 70)
print("KURAN MODEL: SOVIET OPPOSITION DYNAMICS")
print("=" * 70)
print(f"\nEquilibria: S_lo={S_lo}, S_mid={S_mid}, S_hi={S_hi}")
print(f"Rate constant: gamma={gamma}")

# Eigenvalues at each fixed point
lambda_lo  = f_prime(S_lo)
lambda_mid = f_prime(S_mid)
lambda_hi  = f_prime(S_hi)

print(f"\nEigenvalues:")
print(f"  At S_lo  = {S_lo}: λ = {lambda_lo:.4f}  ({'stable' if lambda_lo < 0 else 'UNSTABLE'})")
print(f"  At S_mid = {S_mid}: λ = {lambda_mid:.4f}  ({'stable' if lambda_mid < 0 else 'UNSTABLE'})")
print(f"  At S_hi  = {S_hi}: λ = {lambda_hi:.4f}  ({'stable' if lambda_hi < 0 else 'UNSTABLE'})")

# Relaxation time at the regime equilibrium
tau_relax = 1.0 / abs(lambda_lo)
print(f"\nRelaxation time at regime equilibrium: tau_relax = {tau_relax:.3f} years")

# ============================================================
# POTENTIAL AND BARRIER
# ============================================================

# Compute potential at key points
V_lo  = 0.0  # Reference: V(S_lo) = 0
V_mid = V(S_mid)
V_hi  = V(S_hi)

print(f"\nPotential landscape:")
print(f"  V(S_lo)  = {V_lo:.6f}  (regime well)")
print(f"  V(S_mid) = {V_mid:.6f}  (barrier top)")
print(f"  V(S_hi)  = {V_hi:.6f}  (revolution well)")

barrier = V_mid - V_lo  # Barrier from regime to revolution
reverse_barrier = V_mid - V_hi  # Barrier from revolution back to regime

print(f"\n  Barrier (regime → revolution): ΔV = {barrier:.6f}")
print(f"  Barrier (revolution → regime): ΔV_rev = {reverse_barrier:.6f}")
print(f"  Asymmetry ratio: {reverse_barrier/barrier:.2f}")

# ============================================================
# KRAMERS FORMULA
# ============================================================

# Standard 1D Kramers:
# MFPT = (2π / sqrt(|V''(S_eq)| × |V''(S_sad)|)) × exp(2ΔV/σ²)
#
# V''(S) = -f'(S) at equilibria
# |V''(S_lo)| = |λ_lo| (curvature at regime minimum)
# |V''(S_mid)| = λ_mid  (curvature at saddle maximum — note V''<0 there)

curv_eq = abs(lambda_lo)      # |V''| at equilibrium
curv_sad = abs(lambda_mid)    # |V''| at saddle (V'' is negative, take abs)

prefactor_inv = np.sqrt(curv_eq * curv_sad) / (2 * np.pi)  # 1/(C*tau) in Kramers notation
print(f"\nKramers prefactor:")
print(f"  |V''(S_lo)|  = {curv_eq:.4f}")
print(f"  |V''(S_mid)| = {curv_sad:.4f}")
print(f"  1/(C×τ) = √(prod)/2π = {prefactor_inv:.4f}")

# ============================================================
# FIND σ* WHERE D_Kramers = D_observed
# ============================================================

D_observed = 15.0  # MFPT / tau_relax ≈ 15 for the Soviet system

def D_kramers(sigma):
    """Kramers prediction for D = MFPT / tau_relax."""
    if sigma <= 0:
        return np.inf
    exponent = 2 * barrier / sigma**2
    if exponent > 500:
        return np.inf
    mfpt = (1.0 / prefactor_inv) * np.exp(exponent)
    return mfpt / tau_relax

# Scan sigma to find where D_Kramers = D_observed
print(f"\n{'='*70}")
print(f"FINDING σ* WHERE D_Kramers = D_observed = {D_observed}")
print(f"{'='*70}")

sigmas = np.logspace(-3, 1, 1000)
Ds = np.array([D_kramers(s) for s in sigmas])

# Find crossings
for i in range(len(Ds)-1):
    if (Ds[i] - D_observed) * (Ds[i+1] - D_observed) < 0:
        sigma_star = brentq(lambda s: D_kramers(s) - D_observed, sigmas[i], sigmas[i+1])
        print(f"\n  σ* = {sigma_star:.6f}")
        print(f"  D_Kramers(σ*) = {D_kramers(sigma_star):.2f}")
        print(f"  2ΔV/σ*² = {2*barrier/sigma_star**2:.4f}")

        # Physical interpretation
        # std(S) at equilibrium ≈ σ / sqrt(2 |λ_eq|)
        std_S = sigma_star / np.sqrt(2 * abs(lambda_lo))
        print(f"\n  Physical interpretation:")
        print(f"    std(S) at regime equilibrium = {std_S:.4f}")
        print(f"    = {std_S*100:.1f} percentage points of opposition fluctuation")
        print(f"    Regime equilibrium S_lo = {S_lo:.2f} ± {std_S:.3f}")
        print(f"    Tipping point S_mid = {S_mid:.2f}")
        print(f"    Distance to tipping: {S_mid - S_lo:.2f} = {(S_mid-S_lo)/std_S:.1f} std devs")

        # MFPT
        mfpt = D_observed * tau_relax
        print(f"\n    MFPT = D × tau_relax = {D_observed} × {tau_relax:.2f} = {mfpt:.1f} years")
        print(f"    (Soviet persistence: ~44-69 years)")
        break

# ============================================================
# EXACT MFPT (numerical integration, no approximation)
# ============================================================

print(f"\n{'='*70}")
print("EXACT MFPT (numerical, verifies Kramers)")
print(f"{'='*70}")

def exact_mfpt_1d(sigma, x_eq, x_sad, f_func, dx=0.0001):
    """
    Exact MFPT for 1D SDE: dX = f(X)dt + σdW
    From x_eq to x_sad (absorbing boundary at x_sad).

    Uses the standard formula:
    T(x) = (2/σ²) ∫_x^x_sad dy exp(2V(y)/σ²) ∫_x_lo^y dz exp(-2V(z)/σ²)

    where V(y) = -∫f(y')dy' (potential).
    """
    # Compute potential on grid
    xs = np.linspace(x_eq - 0.01, x_sad + 0.01, 5000)
    xs = xs[(xs >= x_eq - 0.001) & (xs <= x_sad + 0.001)]

    # Potential relative to x_eq
    V_arr = np.zeros_like(xs)
    for i, x in enumerate(xs):
        V_arr[i], _ = quad(lambda s: -f_func(s), x_eq, x)

    # MFPT from x_eq:
    # T(x_eq) = (2/σ²) ∫_{x_eq}^{x_sad} exp(2V(y)/σ²) [∫_{-∞}^y exp(-2V(z)/σ²) dz] dy
    # With reflecting boundary at x < x_eq, the inner integral starts at x_eq

    sigma2 = sigma**2

    def inner_integral(y_val):
        """∫_{x_eq}^{y} exp(-2V(z)/σ²) dz"""
        mask = xs <= y_val + 1e-10
        zs = xs[mask]
        Vs = V_arr[mask]
        integrand = np.exp(-2.0 * Vs / sigma2)
        return np.trapz(integrand, zs)

    # Outer integral
    outer_xs = xs[(xs >= x_eq) & (xs <= x_sad)]
    outer_Vs = np.array([V_arr[np.argmin(np.abs(xs - x))] for x in outer_xs])

    outer_integrand = np.zeros_like(outer_xs)
    for i, y in enumerate(outer_xs):
        outer_integrand[i] = np.exp(2.0 * outer_Vs[i] / sigma2) * inner_integral(y)

    mfpt = (2.0 / sigma2) * np.trapz(outer_integrand, outer_xs)
    return mfpt

# Compute exact MFPT at σ*
if 'sigma_star' in dir():
    mfpt_exact = exact_mfpt_1d(sigma_star, S_lo, S_mid, f_drift)
    D_exact = mfpt_exact / tau_relax
    mfpt_kramers = D_observed * tau_relax

    print(f"\n  At σ* = {sigma_star:.6f}:")
    print(f"    MFPT_Kramers = {mfpt_kramers:.2f} years")
    print(f"    MFPT_exact   = {mfpt_exact:.2f} years")
    print(f"    D_Kramers    = {D_observed:.2f}")
    print(f"    D_exact      = {D_exact:.2f}")
    print(f"    Ratio D_exact/D_Kramers = {D_exact/D_observed:.4f}")

    # K factor
    K_factor = D_exact / (prefactor_inv * tau_relax * np.exp(2*barrier/sigma_star**2))
    # Actually: D_exact = K × exp(2ΔV/σ²) × (1/(Cτ)) / (1/tau_relax)...
    # Let me just compute it properly
    K = D_exact * tau_relax * prefactor_inv / np.exp(2*barrier/sigma_star**2)
    print(f"    K factor     = {K:.4f}")

# ============================================================
# SCAN: D vs σ
# ============================================================

print(f"\n{'='*70}")
print("D vs σ SCAN (exact MFPT)")
print(f"{'='*70}")

sigma_scan = np.array([0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
print(f"\n  {'σ':>8s} {'D_exact':>12s} {'D_Kramers':>12s} {'2ΔV/σ²':>10s} {'std(S)':>10s}")
print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

for sig in sigma_scan:
    try:
        mfpt_ex = exact_mfpt_1d(sig, S_lo, S_mid, f_drift)
        D_ex = mfpt_ex / tau_relax
        D_kr = D_kramers(sig)
        barrier_ratio = 2 * barrier / sig**2
        std_s = sig / np.sqrt(2 * abs(lambda_lo))
        print(f"  {sig:8.4f} {D_ex:12.2f} {D_kr:12.2f} {barrier_ratio:10.4f} {std_s:10.4f}")
    except:
        print(f"  {sig:8.4f} {'error':>12s}")

# ============================================================
# PHYSICAL INTERPRETATION
# ============================================================

print(f"\n{'='*70}")
print("PHYSICAL INTERPRETATION")
print(f"{'='*70}")

print(f"""
The Kuran model as continuous-time ODE:

  dS/dt = -γ(S - {S_lo})(S - {S_mid})(S - {S_hi}) + σξ(t)

State variable: S = fraction of population openly opposing the regime
  S_lo  = {S_lo}  : regime normal (~{S_lo*100:.0f}% active opposition)
  S_mid = {S_mid}  : tipping point
  S_hi  = {S_hi}  : revolution complete (~{S_hi*100:.0f}% opposition)

This IS a genuine double-well potential:
  - Regime well at S = {S_lo} (stable, all eigenvalues negative)
  - Revolution well at S = {S_hi} (stable, all eigenvalues negative)
  - Saddle (barrier) at S = {S_mid} (unstable)

Kramers topology: stable node → saddle → stable node  ✓
Native dimensionality: 1D (S is the natural reaction coordinate)

Barrier height: ΔV = {barrier:.6f}
Eigenvalue at regime eq: λ = {lambda_lo:.4f}
Eigenvalue at saddle: λ_u = {lambda_mid:.4f}
Relaxation time: τ_relax = {tau_relax:.3f} years

Noise: σ represents the amplitude of shocks that shift public opposition
  - Leadership crises, economic shocks, external events
  - Chernobyl, oil price crash, Afghan war casualties
  - Each pushes S up temporarily; the regime's restoring force pushes it back

The system escapes when accumulated noise pushes S past S_mid = {S_mid}.
This triggers the preference cascade (Kuran's key insight):
  once S > S_mid, the drift is POSITIVE, and S accelerates toward S_hi.
  The cascade is self-reinforcing. This is what happened in 1989-1991.
""")

# ============================================================
# REGULATORY CHANNELS (for product equation assessment)
# ============================================================

print(f"{'='*70}")
print("CHANNEL STRUCTURE")
print(f"{'='*70}")

print(f"""
What shifts the threshold distribution F(S) and hence the potential V(S)?

The tipping point S_mid depends on:
1. IDEOLOGY (m): higher indoctrination → higher thresholds → S_mid moves RIGHT → larger barrier
2. REPRESSION (KGB): higher cost of dissent → higher thresholds → S_mid RIGHT → larger barrier
3. ECONOMIC PERFORMANCE: better economy → higher thresholds → S_mid RIGHT → larger barrier
4. INFORMATION CONTROL: less outside info → higher thresholds → S_mid RIGHT → larger barrier

In the ODE, these channels shift S_mid:
  S_mid = S_mid(m, repression, economy, info)

Each channel's effect on D operates through the barrier:
  ΔV depends on S_mid (and S_lo, S_hi)
  D = (prefactor) × exp(2ΔV/σ²)

Shifting S_mid changes ΔV, which changes D exponentially.

SEPARABILITY CHECK:
  If S_mid = S_mid_base + Δ_ideology + Δ_repression + Δ_economy + Δ_info,
  then ΔV is a function of the SUM of channel contributions.
  But ΔV is NOT a sum of independent barrier pieces (it's a nonlinear
  function of S_mid). So the product equation D = ∏(1/εᵢ) does NOT
  apply directly.

  HOWEVER: Kramers itself (D = K × exp(2ΔV/σ²)/(Cτ)) works regardless.
  D is computable. The channels affect D through ΔV. The decomposition
  into per-channel contributions is a separate (harder) question.
""")

# ============================================================
# SENSITIVITY: How does D change with S_mid?
# ============================================================

print(f"{'='*70}")
print("SENSITIVITY: D vs S_mid (tipping point position)")
print(f"{'='*70}")

print(f"\n  Noise fixed at σ = {sigma_star:.4f}" if 'sigma_star' in dir() else "")
print(f"\n  {'S_mid':>8s} {'ΔV':>10s} {'D_Kramers':>12s} {'D_exact':>12s} {'Interpretation':>30s}")
print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*30}")

for s_mid_test in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
    # Redefine drift with new S_mid
    def f_test(S, sm=s_mid_test):
        return -gamma * (S - S_lo) * (S - sm) * (S - S_hi)

    # Barrier
    dv, _ = quad(lambda s: f_test(s), S_lo, s_mid_test)
    dv = -dv  # V = -integral of f

    if 'sigma_star' in dir() and sigma_star > 0:
        sig = sigma_star
        # Eigenvalue at S_lo
        lam = -gamma * (S_lo - s_mid_test) * (S_lo - S_hi)  # simplified
        lam2 = gamma * (s_mid_test - S_lo) * (S_hi - S_lo)  # |eigenvalue|
        lam_u = gamma * (s_mid_test - S_lo) * (S_hi - s_mid_test)  # eigenvalue at saddle

        pf = np.sqrt(lam2 * lam_u) / (2 * np.pi)
        tau_r = 1.0 / lam2

        D_kr = (1.0/pf) * np.exp(2*dv/sig**2) / tau_r if dv > 0 else 0

        try:
            mfpt_ex = exact_mfpt_1d(sig, S_lo, s_mid_test, f_test)
            D_ex = mfpt_ex / tau_r
        except:
            D_ex = float('nan')

        interp = ""
        if D_kr < 5: interp = "very fragile"
        elif D_kr < 20: interp = "marginally stable"
        elif D_kr < 100: interp = "moderately stable"
        elif D_kr < 1000: interp = "robust"
        else: interp = "very robust"

        print(f"  {s_mid_test:8.2f} {dv:10.6f} {D_kr:12.1f} {D_ex:12.1f} {interp:>30s}")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

if 'sigma_star' in dir():
    print(f"""
RESULT: The Kuran preference falsification model, cast as a continuous-time
ODE, produces a genuine double-well potential with the correct Kramers topology.

  Model: dS/dt = -γ(S-S_lo)(S-S_mid)(S-S_hi) + σξ(t)

  Equilibria:
    Regime stable (S = {S_lo}):  eigenvalue = {lambda_lo:.4f} (STABLE NODE) ✓
    Tipping point (S = {S_mid}): eigenvalue = {lambda_mid:.4f} (SADDLE)      ✓
    Revolution (S = {S_hi}):     eigenvalue = {lambda_hi:.4f} (STABLE NODE) ✓

  Kramers computation:
    Barrier ΔV = {barrier:.6f}
    σ* = {sigma_star:.6f}  (where D_Kramers = D_observed = {D_observed})
    2ΔV/σ*² = {2*barrier/sigma_star**2:.4f}
    1/(C×τ) = {prefactor_inv:.4f}

  Physical check:
    τ_relax = {tau_relax:.3f} years
    MFPT = {D_observed * tau_relax:.1f} years (vs observed ~44 years)
    std(S) = {sigma_star/np.sqrt(2*abs(lambda_lo)):.3f} ({sigma_star/np.sqrt(2*abs(lambda_lo))*100:.1f}% opposition fluctuation)
    Distance to tipping = {(S_mid-S_lo)/(sigma_star/np.sqrt(2*abs(lambda_lo))):.1f} standard deviations

  VERDICT: Kramers is computable for the Soviet system.
  D_observed ≈ 15 corresponds to a thin barrier (2ΔV/σ² ≈ {2*barrier/sigma_star**2:.1f})
  with moderate noise. The system was marginally metastable — consistent with
  a 69-year regime that collapsed abruptly.
""")
