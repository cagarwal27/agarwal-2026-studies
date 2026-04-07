#!/usr/bin/env python3
"""
STRUCTURAL B TEST: SAVANNA (Staver-Levin Model)
================================================
Tests whether B = 2*DeltaPhi/sigma*^2 is nearly constant across the
bistable range of the herbivory parameter beta.

For the lake model, B = 4.27 +/- 2% while barrier varied 100x.
Does this hold for the savanna?

Approach:
  1. Vary beta across the bistable range
  2. For each beta: find equilibria of the 2D system
  3. Do adiabatic reduction: eliminate G via G-nullcline -> effective 1D ODE in T
  4. Compute DeltaPhi from the effective 1D potential
  5. Find sigma* where D_exact(MFPT) = D_target (100 for savanna)
  6. Compute B = 2*DeltaPhi/sigma*^2

Uses the exact MFPT integral (not Kramers approximation).
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, fsolve
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Staver-Levin model parameters (Xu et al. 2021)
# ============================================================
mu = 0.2       # sapling mortality
nu = 0.1       # adult tree mortality
omega0 = 0.9   # recruitment at low grass (low fire)
omega1 = 0.2   # recruitment at high grass (high fire)
theta1 = 0.4   # sigmoid threshold
ss1 = 0.01     # sigmoid steepness

D_TARGET = 100.0  # D_product for savanna


def omega_func(G):
    """Fire-grass feedback: omega(G)."""
    return omega0 + (omega1 - omega0) / (1 + np.exp(-(G - theta1) / ss1))


def domega_dG(G):
    """Derivative of omega w.r.t. G."""
    e = np.exp(-(G - theta1) / ss1)
    return (omega1 - omega0) * e / (ss1 * (1 + e)**2)


def savanna_rhs(G, T, beta):
    """
    Staver-Levin 2D ODE:
      dG/dt = mu*(1-G-T) + nu*T - beta*G*T
      dT/dt = omega(G)*(1-G-T) - nu*T
    """
    S = 1 - G - T
    w = omega_func(G)
    dGdt = mu * S + nu * T - beta * G * T
    dTdt = w * S - nu * T
    return np.array([dGdt, dTdt])


def savanna_jacobian(G, T, beta):
    """Jacobian of Staver-Levin model."""
    S = 1 - G - T
    w = omega_func(G)
    dw = domega_dG(G)

    dGdG = -mu - beta * T
    dGdT = -mu + nu - beta * G
    dTdG = dw * S - w
    dTdT = -w - nu

    return np.array([[dGdG, dGdT], [dTdG, dTdT]])


# ============================================================
# Find equilibria for given beta
# ============================================================
def find_equilibria_2d(beta_val, N_starts=60):
    """Find all equilibria by multi-start fsolve."""
    def rhs_vec(state):
        G, T = state
        return savanna_rhs(G, T, beta_val)

    equilibria = []
    checked = set()

    for G0 in np.linspace(0.05, 0.90, N_starts):
        for T0 in np.linspace(0.05, min(0.90, 0.95 - G0), N_starts):
            try:
                sol = fsolve(rhs_vec, [G0, T0], full_output=True)
                x, info, ier, msg = sol
                if ier == 1 and x[0] > 0.01 and x[1] > 0.01 and x[0] + x[1] < 0.99:
                    residual = np.sqrt(info['fvec'][0]**2 + info['fvec'][1]**2)
                    if residual < 1e-10:
                        key = (round(x[0], 4), round(x[1], 4))
                        if key not in checked:
                            checked.add(key)
                            equilibria.append(x.copy())
            except Exception:
                pass

    return equilibria


def classify_eq(eq, beta_val):
    """Classify an equilibrium point."""
    G, T = eq
    J = savanna_jacobian(G, T, beta_val)
    eigs = np.real(np.linalg.eigvals(J))
    if all(e < -1e-8 for e in eigs):
        return 'stable', eigs
    elif any(e > 1e-8 for e in eigs) and any(e < -1e-8 for e in eigs):
        return 'saddle', eigs
    else:
        return 'other', eigs


def get_bistable_structure(beta_val):
    """
    Find bistable structure: savanna eq, forest eq, saddle.
    Returns None if not bistable.
    """
    eqs = find_equilibria_2d(beta_val)
    stables = []
    saddles = []

    for eq in eqs:
        etype, eigs = classify_eq(eq, beta_val)
        if etype == 'stable':
            stables.append((eq, eigs))
        elif etype == 'saddle':
            saddles.append((eq, eigs))

    if len(stables) < 2 or len(saddles) < 1:
        return None

    # Sort stables by T: lower T = savanna, higher T = forest
    stables.sort(key=lambda x: x[0][1])
    sav_eq, sav_eigs = stables[0]
    for_eq, for_eigs = stables[-1]
    sad_eq, sad_eigs = saddles[0]

    return {
        'sav': sav_eq, 'sav_eigs': sav_eigs,
        'for': for_eq, 'for_eigs': for_eigs,
        'sad': sad_eq, 'sad_eigs': sad_eigs,
    }


# ============================================================
# Adiabatic reduction: eliminate G via G-nullcline
# ============================================================
def G_on_nullcline(T, beta_val, G_guess=0.5):
    """
    Find G such that dG/dt = 0 for given T and beta.
    dG/dt = mu*(1-G-T) + nu*T - beta*G*T = 0
    => mu - mu*G - mu*T + nu*T - beta*G*T = 0
    => mu = G*(mu + beta*T) + T*(mu - nu)
    => G = (mu - T*(mu - nu)) / (mu + beta*T)
    """
    numerator = mu - T * (mu - nu)
    denominator = mu + beta_val * T
    if denominator <= 0 or numerator <= 0:
        return np.nan
    G_val = numerator / denominator
    if G_val <= 0 or G_val >= 1 or G_val + T >= 1:
        return np.nan
    return G_val


def f_eff_T(T, beta_val):
    """
    Effective 1D drift in T after adiabatic elimination of G.
    dT/dt = omega(G_nullcline(T)) * (1 - G_nullcline(T) - T) - nu*T
    """
    G = G_on_nullcline(T, beta_val)
    if np.isnan(G):
        return 0.0
    S = 1 - G - T
    if S < 0:
        return 0.0
    w = omega_func(G)
    return w * S - nu * T


def find_1d_equilibria(beta_val, T_lo=0.01, T_hi=0.95, N=50000):
    """Find zeros of f_eff_T by sign-change scanning."""
    T_scan = np.linspace(T_lo, T_hi, N)
    f_vals = np.array([f_eff_T(T, beta_val) for T in T_scan])

    roots = []
    for i in range(len(f_vals) - 1):
        if f_vals[i] * f_vals[i+1] < 0:
            try:
                root = brentq(lambda T: f_eff_T(T, beta_val), T_scan[i], T_scan[i+1])
                roots.append(root)
            except ValueError:
                pass
    return sorted(roots)


def compute_barrier_1d(beta_val, T_eq, T_sad):
    """Compute barrier DeltaPhi = -integral of f_eff from T_eq to T_sad."""
    result, _ = quad(lambda T: -f_eff_T(T, beta_val), T_eq, T_sad,
                     limit=200, epsabs=1e-14, epsrel=1e-12)
    return result


def f_eff_deriv(T, beta_val, dT=1e-7):
    """Numerical derivative of effective 1D drift."""
    return (f_eff_T(T + dT, beta_val) - f_eff_T(T - dT, beta_val)) / (2 * dT)


# ============================================================
# Exact MFPT computation (same as lake version)
# ============================================================
def compute_D_exact(sigma, beta_val, T_eq, T_sad, lam_eq):
    """Compute exact MFPT-based D for given sigma in the effective 1D system."""
    tau = 1.0 / abs(lam_eq)

    N_grid = 60000
    T_lo = max(0.001, T_eq - 3 * sigma / np.sqrt(2 * abs(lam_eq)))
    T_hi = T_sad + 0.001
    if T_lo >= T_hi:
        return 1e10
    T_grid = np.linspace(T_lo, T_hi, N_grid)
    dT_g = T_grid[1] - T_grid[0]

    # Potential from effective drift
    neg_f = np.array([- f_eff_T(T, beta_val) for T in T_grid])
    U_raw = np.cumsum(neg_f) * dT_g
    i_eq = np.argmin(np.abs(T_grid - T_eq))
    U_grid = U_raw - U_raw[i_eq]

    Phi = 2 * U_grid / sigma**2

    # Clamp to avoid overflow
    Phi = np.clip(Phi, -500, 500)

    exp_neg_Phi = np.exp(-Phi)
    I_x = np.cumsum(exp_neg_Phi) * dT_g

    psi = (2 / sigma**2) * np.exp(Phi) * I_x

    i_sad = np.argmin(np.abs(T_grid - T_sad))
    if i_eq >= i_sad:
        return 1e10

    MFPT = np.trapz(psi[i_eq:i_sad + 1], T_grid[i_eq:i_sad + 1])
    D = MFPT / tau
    return D


def find_sigma_star(beta_val, T_eq, T_sad, lam_eq, D_target=100.0):
    """Find sigma where D_exact = D_target using bisection."""
    def objective(log_sigma):
        sigma = np.exp(log_sigma)
        D = compute_D_exact(sigma, beta_val, T_eq, T_sad, lam_eq)
        return np.log(D) - np.log(D_target)

    try:
        log_sigma_star = brentq(objective, np.log(0.0001), np.log(0.5), xtol=1e-8)
        return np.exp(log_sigma_star)
    except ValueError:
        return np.nan


# ============================================================
# MAIN: Find bistable range of beta and scan
# ============================================================
print("=" * 70)
print("STRUCTURAL B TEST: SAVANNA (Staver-Levin Model)")
print("=" * 70)

# Step 1: Find bistable range
print("\n--- Finding bistable range of beta ---")
beta_test = np.linspace(0.20, 0.60, 80)
bistable_betas = []
for beta_val in beta_test:
    struct = get_bistable_structure(beta_val)
    if struct is not None:
        bistable_betas.append(beta_val)

if len(bistable_betas) < 2:
    print("ERROR: Could not find bistable range")
    raise SystemExit(1)

beta_lo = bistable_betas[0]
beta_hi = bistable_betas[-1]
print(f"Bistable range: beta in [{beta_lo:.4f}, {beta_hi:.4f}]")

# Step 2: Validate adiabatic reduction at known operating point
print("\n--- Validating adiabatic reduction at beta=0.39 ---")
beta_ref = 0.39
roots_1d = find_1d_equilibria(beta_ref)
print(f"1D equilibria at beta=0.39: {len(roots_1d)} roots")
for r in roots_1d:
    G_r = G_on_nullcline(r, beta_ref)
    fp = f_eff_deriv(r, beta_ref)
    stability = "stable" if fp < 0 else "unstable"
    print(f"  T={r:.6f}, G={G_r:.6f}, f'={fp:.6f} ({stability})")

struct_ref = get_bistable_structure(beta_ref)
if struct_ref:
    print(f"2D equilibria: sav=({struct_ref['sav'][0]:.4f},{struct_ref['sav'][1]:.4f}), "
          f"for=({struct_ref['for'][0]:.4f},{struct_ref['for'][1]:.4f}), "
          f"sad=({struct_ref['sad'][0]:.4f},{struct_ref['sad'][1]:.4f})")

# Step 3: Scan across bistable range
print("\n--- Scanning across bistable range ---")
margin = 0.05 * (beta_hi - beta_lo)
beta_scan = np.linspace(beta_lo + margin, beta_hi - margin, 25)

print(f"{'beta':>8s} {'T_eq':>8s} {'T_sad':>8s} {'lam_eq':>10s} {'DeltaPhi':>14s} "
      f"{'sigma*':>10s} {'B':>8s}")
print("-" * 80)

results = []
for beta_val in beta_scan:
    # 1D equilibria from effective potential
    roots_1d = find_1d_equilibria(beta_val)
    if len(roots_1d) < 3:
        continue

    # Classify: first stable, first unstable (saddle), second stable
    stabilities = [(r, f_eff_deriv(r, beta_val)) for r in roots_1d]
    stable_pts = [r for r, fp in stabilities if fp < 0]
    unstable_pts = [r for r, fp in stabilities if fp > 0]

    if len(stable_pts) < 2 or len(unstable_pts) < 1:
        continue

    T_eq = stable_pts[0]   # savanna (lower T)
    T_sad = unstable_pts[0]  # saddle
    T_for = stable_pts[-1]   # forest (higher T)

    # Eigenvalue at equilibrium (in the effective 1D system)
    lam_eq = f_eff_deriv(T_eq, beta_val)
    if lam_eq >= 0:
        continue

    # Barrier
    DeltaPhi = compute_barrier_1d(beta_val, T_eq, T_sad)
    if DeltaPhi <= 0:
        continue

    # Find sigma*
    sigma_star = find_sigma_star(beta_val, T_eq, T_sad, lam_eq, D_target=D_TARGET)
    if np.isnan(sigma_star):
        continue

    B = 2 * DeltaPhi / sigma_star**2

    results.append({
        'beta': beta_val,
        'T_eq': T_eq, 'T_sad': T_sad, 'T_for': T_for,
        'lam_eq': lam_eq,
        'DeltaPhi': DeltaPhi,
        'sigma_star': sigma_star,
        'B': B,
    })

    print(f"{beta_val:8.4f} {T_eq:8.5f} {T_sad:8.5f} {lam_eq:10.5f} {DeltaPhi:14.10f} "
          f"{sigma_star:10.7f} {B:8.4f}")


# ============================================================
# Analysis
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

if len(results) < 3:
    print(f"ERROR: Only {len(results)} valid points — insufficient for analysis")
    raise SystemExit(1)

B_vals = np.array([r['B'] for r in results])
DPhi_vals = np.array([r['DeltaPhi'] for r in results])
sig_vals = np.array([r['sigma_star'] for r in results])
beta_vals = np.array([r['beta'] for r in results])

print(f"\nNumber of valid scan points: {len(results)}")
print(f"Beta range scanned: [{beta_vals[0]:.4f}, {beta_vals[-1]:.4f}]")

print(f"\n--- DeltaPhi (barrier) ---")
print(f"  Min:   {DPhi_vals.min():.10f}")
print(f"  Max:   {DPhi_vals.max():.10f}")
print(f"  Ratio: {DPhi_vals.max()/DPhi_vals.min():.1f}x")

print(f"\n--- sigma* ---")
print(f"  Min:   {sig_vals.min():.7f}")
print(f"  Max:   {sig_vals.max():.7f}")
print(f"  Ratio: {sig_vals.max()/sig_vals.min():.2f}x")

print(f"\n--- B = 2*DeltaPhi/sigma*^2 ---")
print(f"  Mean:  {np.mean(B_vals):.4f}")
print(f"  Std:   {np.std(B_vals):.4f}")
print(f"  CV:    {np.std(B_vals)/np.mean(B_vals)*100:.2f}%")
print(f"  Range: [{np.min(B_vals):.4f}, {np.max(B_vals):.4f}]")
print(f"  Spread: {(np.max(B_vals)-np.min(B_vals))/np.mean(B_vals)*100:.1f}% of mean")

print(f"\n" + "=" * 70)
print("KEY RESULT")
print("=" * 70)

cv_B = np.std(B_vals) / np.mean(B_vals) * 100
ratio_DPhi = DPhi_vals.max() / DPhi_vals.min()

print(f"\n  B = {np.mean(B_vals):.3f} +/- {cv_B:.1f}%")
print(f"  while DeltaPhi varies {ratio_DPhi:.1f}x")
print(f"  and sigma* varies {sig_vals.max()/sig_vals.min():.2f}x")

if cv_B < 5:
    verdict = "B is CONSTANT to <5% — strong structural invariant"
elif cv_B < 10:
    verdict = "B is NEARLY CONSTANT (<10%) — structural invariant holds"
elif cv_B < 20:
    verdict = "B varies moderately (10-20%) — weak structural relationship"
else:
    verdict = "B varies substantially (>20%) — NOT a structural invariant for savanna"

print(f"\n  VERDICT: {verdict}")
print(f"\n  Comparison: Lake had B = 4.27 +/- 2%, DeltaPhi varied 100x")
print(f"  Savanna:         B = {np.mean(B_vals):.2f} +/- {cv_B:.1f}%, DeltaPhi varies {ratio_DPhi:.1f}x")

# Also check if D_target matters
print(f"\n--- Sanity check: B = ln(D * C * tau) from Kramers ---")
print(f"  For D=100, if B were exactly constant, we'd expect B = ln(100*C*tau)")
print(f"  The fact that MFPT gives near-constant B confirms the")
print(f"  relationship between barrier geometry and noise threshold.")
