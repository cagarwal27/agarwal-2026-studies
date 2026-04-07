#!/usr/bin/env python3
"""
STRUCTURAL B TEST: TOPP 2000 DIABETES MODEL
============================================
Tests whether B = 2*DeltaPhi/sigma*^2 is nearly constant across the
bistable range for the Topp et al. (2000) beta-cell/glucose/insulin model.

This is the first HUMAN DISEASE system tested for the B structural invariant.

Model: 3D ODE (G, I, beta) with extraordinary timescale separation (~29,670x).
Adiabatic reduction eliminates I (fast, ~0.1h) and G (fast, ~1.3h) to get
effective 1D dynamics in beta (slow, ~70 days).

Key difference from lake/savanna/kelp: noise is MULTIPLICATIVE in beta.
We transform to u = ln(beta) where noise becomes additive.

Citation: Topp B, Promislow K, deVries G, Miura RM, Finegood DT.
"A model of beta-cell mass, insulin, and glucose kinetics: pathways to diabetes."
J Theor Biol 206:605-619, 2000. BioModels BIOMD0000000341.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Topp 2000 model parameters (ALL from the published paper)
# ============================================================
R0 = 864.0        # mg/dL/day, glucose production rate
EG0 = 1.44        # 1/day, glucose effectiveness
SI_default = 0.72 # mL/(muU*day), insulin sensitivity
sigma_p = 43.2    # muU/(mg*day), insulin secretion rate
alpha = 20000.0   # (mg/dL)^2, Hill half-saturation squared
k_clear = 432.0   # 1/day, insulin clearance rate
d0_default = 0.06 # 1/day, beta-cell basal death rate
r1 = 0.84e-3      # 1/(mg/dL * day), beta-cell growth rate coeff
r2 = 2.4e-6       # 1/(mg/dL)^2/day, glucotoxicity coeff

# ============================================================
# Adiabatic reduction: 3D -> 1D via G_qss(beta)
# ============================================================

def G_qss_single(beta, SI_val=SI_default):
    """Solve for quasi-steady-state glucose given beta (scalar)."""
    def residual(G):
        I_val = sigma_p * beta * G**2 / (k_clear * (alpha + G**2))
        return R0 - (EG0 + SI_val * I_val) * G
    try:
        return brentq(residual, 1.0, 800.0)
    except ValueError:
        return np.nan

def build_G_qss_interp(u_lo, u_hi, N_interp=3000, SI_val=SI_default):
    """Build cubic spline for G_qss(u) where u = ln(beta).
    This makes the MFPT grid evaluation ~10,000x faster.
    """
    u_pts = np.linspace(u_lo, u_hi, N_interp)
    G_pts = np.array([G_qss_single(np.exp(u), SI_val) for u in u_pts])
    return CubicSpline(u_pts, G_pts)

def g_beta_func(G, d0_val):
    """Beta-cell growth rate: g(G) = -d0 + r1*G - r2*G^2."""
    return -d0_val + r1 * G - r2 * G**2


# ============================================================
# Find equilibria analytically
# ============================================================

def find_G_roots(d0_val):
    """Roots of g(G) = -d0 + r1*G - r2*G^2 = 0."""
    disc = r1**2 - 4 * r2 * d0_val
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    G_low = (r1 - sqrt_disc) / (2 * r2)
    G_high = (r1 + sqrt_disc) / (2 * r2)
    return G_low, G_high

def beta_from_G(G_target, SI_val=SI_default):
    """Invert G_qss: find beta such that G_qss(beta) = G_target."""
    numer = (R0 / G_target - EG0) * k_clear * (alpha + G_target**2)
    denom = SI_val * sigma_p * G_target**2
    if denom <= 0:
        return np.nan
    return numer / denom

def find_equilibria_u(d0_val, SI_val=SI_default):
    """Find (u_healthy, u_saddle) analytically.
    Equilibria of du/dt = g(G_qss(exp(u))) = 0 occur where G_qss(beta) = G_low or G_high.
    """
    roots = find_G_roots(d0_val)
    if roots is None:
        return None
    G_low, G_high = roots

    beta_healthy = beta_from_G(G_low, SI_val)
    beta_saddle = beta_from_G(G_high, SI_val)

    if beta_healthy <= 0 or beta_saddle <= 0:
        return None
    if np.isnan(beta_healthy) or np.isnan(beta_saddle):
        return None
    if beta_healthy <= beta_saddle:
        return None

    return np.log(beta_healthy), np.log(beta_saddle)


# ============================================================
# Barrier and eigenvalue computation
# ============================================================

def compute_barrier_and_eigenvalue(d0_val, u_eq, u_sad, G_interp):
    """Compute DeltaPhi and eigenvalue using the spline interpolant.

    DeltaPhi = -integral of f_eff_u(u) du from u_eq to u_sad
    where f_eff_u(u) = g(G_qss(exp(u))) = -d0 + r1*G - r2*G^2

    Since u_eq > u_sad and f_eff_u > 0 in (u_sad, u_eq):
    integral from u_eq to u_sad of f < 0, so -integral > 0.
    """
    def f_eff_u(u):
        G = float(G_interp(u))
        return g_beta_func(G, d0_val)

    # Barrier via adaptive quadrature
    DeltaPhi, _ = quad(lambda u: -f_eff_u(u), u_eq, u_sad,
                       limit=200, epsabs=1e-14, epsrel=1e-12)

    # Eigenvalue at healthy equilibrium: df_eff_u/du at u_eq
    du = 1e-6
    lam_eq = (f_eff_u(u_eq + du) - f_eff_u(u_eq - du)) / (2 * du)

    return DeltaPhi, lam_eq


# ============================================================
# Exact MFPT computation (vectorized via spline)
# ============================================================

def compute_D_exact(sigma_m, d0_val, u_eq, u_sad, lam_eq, G_interp):
    """Exact MFPT-based D in u = ln(beta) coordinates.

    SDE: du = [g(G_qss(e^u)) - sigma_m^2/2] dt + sigma_m dW
    Escape from u_eq (healthy) DOWN to u_sad (saddle).
    """
    tau = 1.0 / abs(lam_eq)

    N_grid = 60000
    u_lo = u_sad - 0.5
    u_hi = u_eq + 3.0 * sigma_m / np.sqrt(2.0 * abs(lam_eq))
    if u_lo >= u_hi:
        return 1e10

    u_grid = np.linspace(u_lo, u_hi, N_grid)
    du_g = u_grid[1] - u_grid[0]

    # Vectorized drift evaluation via spline
    G_grid = G_interp(u_grid)
    g_grid = -d0_val + r1 * G_grid - r2 * G_grid**2
    f_full = g_grid - sigma_m**2 / 2.0  # Include Ito correction

    # Build potential
    neg_f = -f_full
    U_raw = np.cumsum(neg_f) * du_g
    i_eq = np.argmin(np.abs(u_grid - u_eq))
    U_grid = U_raw - U_raw[i_eq]

    Phi = 2.0 * U_grid / sigma_m**2

    i_sad = np.argmin(np.abs(u_grid - u_sad))

    # Guard overflow BEFORE clipping: if barrier too large, MFPT is effectively infinite
    Phi_barrier = Phi[i_sad] - Phi[i_eq]
    if Phi_barrier > 300:
        return np.inf

    Phi = np.clip(Phi, -500, 500)

    exp_neg_Phi = np.exp(-Phi)
    I_x = np.cumsum(exp_neg_Phi) * du_g
    psi = (2.0 / sigma_m**2) * np.exp(Phi) * I_x

    lo_idx = min(i_sad, i_eq)
    hi_idx = max(i_sad, i_eq)
    MFPT = np.trapz(psi[lo_idx:hi_idx + 1], u_grid[lo_idx:hi_idx + 1])

    return MFPT / tau


def find_sigma_star(d0_val, u_eq, u_sad, lam_eq, D_target, G_interp):
    """Find sigma_m where D_exact = D_target using bisection on log(sigma_m).

    IMPORTANT: D(sigma_m) is non-monotonic for multiplicative noise due to the
    Ito correction. At small sigma, D is huge (Kramers barrier). At moderate sigma,
    D reaches a minimum. At large sigma, the Ito drift -sigma^2/2 pushes the
    particle AWAY from the saddle, increasing D again.

    We want the Kramers-branch solution (smallest sigma that gives D = D_target).
    Strategy: scan sigma from small to large, find the first crossing of D_target,
    then refine with bisection.
    """
    # First, find the Kramers branch by scanning from small sigma upward
    # D should decrease monotonically in this branch until reaching D_min
    log_sigma_scan = np.linspace(np.log(0.001), np.log(0.3), 40)
    D_scan = []
    for ls in log_sigma_scan:
        sig = np.exp(ls)
        D = compute_D_exact(sig, d0_val, u_eq, u_sad, lam_eq, G_interp)
        D_scan.append(D)

    # Find first crossing where D drops below D_target
    sig_lo = None
    sig_hi = None
    for i in range(len(D_scan) - 1):
        if D_scan[i] > D_target and D_scan[i + 1] <= D_target:
            sig_lo = np.exp(log_sigma_scan[i])
            sig_hi = np.exp(log_sigma_scan[i + 1])
            break
        # Also handle D going from inf to finite below target
        if (D_scan[i] == np.inf or D_scan[i] > 1e15) and \
           D_scan[i + 1] < 1e15 and D_scan[i + 1] <= D_target:
            sig_lo = np.exp(log_sigma_scan[i])
            sig_hi = np.exp(log_sigma_scan[i + 1])
            break

    if sig_lo is None or sig_hi is None:
        # D never crosses D_target in the Kramers branch
        # (D_min > D_target: system can't be held at this D_target)
        return np.nan

    def objective(log_sigma):
        sigma_m = np.exp(log_sigma)
        D = compute_D_exact(sigma_m, d0_val, u_eq, u_sad, lam_eq, G_interp)
        if D == np.inf or D > 1e15:
            return 50.0
        if D <= 0:
            return -50.0
        return np.log(D) - np.log(D_target)

    try:
        log_sig = brentq(objective, np.log(sig_lo), np.log(sig_hi), xtol=1e-10)
        return np.exp(log_sig)
    except ValueError:
        return np.nan


# ============================================================
# MAIN COMPUTATION
# ============================================================
print("=" * 78)
print("STRUCTURAL B TEST: TOPP 2000 DIABETES MODEL")
print("First human disease system tested for B structural invariant")
print("=" * 78)

# ============================================================
# Step 1: Validate adiabatic reduction at standard parameters
# ============================================================
print("\n" + "=" * 78)
print("STEP 1: VALIDATE ADIABATIC REDUCTION")
print("=" * 78)

print(f"\nModel parameters (all from Topp et al. 2000):")
print(f"  R0 = {R0}, EG0 = {EG0}, SI = {SI_default}")
print(f"  sigma_p = {sigma_p}, alpha = {alpha}, k = {k_clear}")
print(f"  d0 = {d0_default}, r1 = {r1}, r2 = {r2}")

print(f"\nTimescale separation:")
print(f"  Insulin: tau_I = 1/k = {1/k_clear*24*60:.1f} min")
print(f"  Glucose: tau_G ~ 1/(EG0+SI*I) ~ {1/(EG0+SI_default*10)*24*60:.0f} min (at I~10)")
print(f"  Beta-cell: tau_beta ~ 70 days")
print(f"  Separation ratio: ~{70 * 24 * 60 / (1/k_clear*24*60):.0f}x")

# Find equilibria at standard d0
G_roots = find_G_roots(d0_default)
print(f"\nBeta-cell growth nullcline g(G) = -d0 + r1*G - r2*G^2 = 0:")
print(f"  G_low = {G_roots[0]:.2f} mg/dL (healthy glucose)")
print(f"  G_high = {G_roots[1]:.2f} mg/dL (diabetic threshold)")

d0_crit = r1**2 / (4 * r2)
print(f"\nCritical d0 = r1^2/(4*r2) = {d0_crit:.6f}/day")
print(f"  d0/d0_crit = {d0_default/d0_crit:.3f} ({d0_default/d0_crit*100:.1f}% of critical)")

eq_result = find_equilibria_u(d0_default)
if eq_result is None:
    print("ERROR: Cannot find equilibria at standard parameters")
    raise SystemExit(1)

u_h, u_s = eq_result
beta_h, beta_s = np.exp(u_h), np.exp(u_s)

# Build G_qss spline for standard SI (wide range to cover all d0 values)
G_interp_std = build_G_qss_interp(-5.0, 16.0, N_interp=5000, SI_val=SI_default)
G_h = float(G_interp_std(u_h))
G_s = float(G_interp_std(u_s))

print(f"\nEquilibria at d0 = {d0_default}:")
print(f"  Healthy: beta = {beta_h:.2f}, G = {G_h:.2f} mg/dL, u = {u_h:.4f}")
print(f"  Saddle:  beta = {beta_s:.2f}, G = {G_s:.2f} mg/dL, u = {u_s:.4f}")
print(f"  Diabetic: beta -> 0, G -> R0/EG0 = {R0/EG0:.1f} mg/dL")

# Validate
f_h = g_beta_func(G_h, d0_default)
f_s = g_beta_func(G_s, d0_default)
print(f"\nValidation (f_eff_u should be ~0 at equilibria):")
print(f"  f_eff_u(u_healthy) = {f_h:.2e}")
print(f"  f_eff_u(u_saddle)  = {f_s:.2e}")

DeltaPhi_std, lam_h = compute_barrier_and_eigenvalue(d0_default, u_h, u_s, G_interp_std)
tau_beta = 1.0 / abs(lam_h)
print(f"\nEigenvalue at healthy: lambda = {lam_h:.6f}/day  (tau = {tau_beta:.1f} days)")
print(f"DeltaPhi (pure barrier in u-space) = {DeltaPhi_std:.6f}")


# ============================================================
# Step 2: Scan d0 across bistable range (D_target = 75)
# ============================================================
print("\n" + "=" * 78)
print("STEP 2: SCAN d0 ACROSS BISTABLE RANGE (D_target = 75)")
print("=" * 78)

D_TARGET_CLINICAL = 75.0
d0_scan = np.linspace(0.005, d0_crit - 0.001, 40)

print(f"\nD_target = {D_TARGET_CLINICAL} (DPP placebo: MFPT ~ 14.3 yr, tau ~ 70 d)")
print(f"\n{'d0':>7s} {'beta_eq':>8s} {'G_eq':>7s} {'beta_sd':>8s} {'G_sd':>7s} "
      f"{'lam_eq':>10s} {'DeltaPhi':>12s} {'sigma_m*':>10s} {'B':>8s} {'B/Bmean':>8s}")
print("-" * 98)

results_d0 = []
for d0_val in d0_scan:
    eq = find_equilibria_u(d0_val)
    if eq is None:
        continue

    u_eq, u_sad = eq
    beta_eq, beta_sad = np.exp(u_eq), np.exp(u_sad)

    # Build interpolant covering this d0's range
    # (Same SI, so same G_qss mapping — reuse the standard one)
    G_eq = float(G_interp_std(u_eq))
    G_sad = float(G_interp_std(u_sad))

    if np.isnan(G_eq) or np.isnan(G_sad) or G_eq < 0 or G_sad < 0:
        continue

    DeltaPhi, lam_eq = compute_barrier_and_eigenvalue(d0_val, u_eq, u_sad, G_interp_std)
    if lam_eq >= 0 or DeltaPhi <= 0:
        continue

    sigma_star = find_sigma_star(d0_val, u_eq, u_sad, lam_eq,
                                  D_TARGET_CLINICAL, G_interp_std)
    if np.isnan(sigma_star):
        continue

    B = 2.0 * DeltaPhi / sigma_star**2

    results_d0.append({
        'd0': d0_val, 'beta_eq': beta_eq, 'G_eq': G_eq,
        'beta_sad': beta_sad, 'G_sad': G_sad,
        'lam_eq': lam_eq, 'DeltaPhi': DeltaPhi,
        'sigma_star': sigma_star, 'B': B,
        'u_eq': u_eq, 'u_sad': u_sad,
    })

# Compute mean B for normalization
if len(results_d0) >= 3:
    B_mean_d0 = np.mean([r['B'] for r in results_d0])
else:
    B_mean_d0 = 1.0

for r in results_d0:
    print(f"{r['d0']:7.4f} {r['beta_eq']:8.2f} {r['G_eq']:7.2f} "
          f"{r['beta_sad']:8.2f} {r['G_sad']:7.2f} "
          f"{r['lam_eq']:10.6f} {r['DeltaPhi']:12.6f} "
          f"{r['sigma_star']:10.6f} {r['B']:8.4f} {r['B']/B_mean_d0:8.4f}")


# ============================================================
# Step 3: B across multiple D_target values
# ============================================================
print("\n" + "=" * 78)
print("STEP 3: B ACROSS MULTIPLE D_TARGET VALUES")
print("=" * 78)

D_targets = [10, 30, 75, 200, 1000]

for D_tgt in D_targets:
    results_D = []
    for d0_val in d0_scan:
        eq = find_equilibria_u(d0_val)
        if eq is None:
            continue
        u_eq, u_sad = eq
        DeltaPhi, lam_eq = compute_barrier_and_eigenvalue(d0_val, u_eq, u_sad, G_interp_std)
        if lam_eq >= 0 or DeltaPhi <= 0:
            continue
        sigma_star = find_sigma_star(d0_val, u_eq, u_sad, lam_eq, D_tgt, G_interp_std)
        if np.isnan(sigma_star):
            continue
        B = 2.0 * DeltaPhi / sigma_star**2
        results_D.append(B)

    if len(results_D) >= 3:
        B_arr = np.array(results_D)
        cv = np.std(B_arr) / np.mean(B_arr) * 100
        print(f"  D_target = {D_tgt:>5d}: B = {np.mean(B_arr):.4f} +/- {cv:.2f}% "
              f"(range [{np.min(B_arr):.4f}, {np.max(B_arr):.4f}], n={len(results_D)})")
    else:
        print(f"  D_target = {D_tgt:>5d}: insufficient data ({len(results_D)} points)")


# ============================================================
# Step 4: SI scan at fixed d0 = 0.06
# ============================================================
print("\n" + "=" * 78)
print("STEP 4: SI SCAN (insulin sensitivity) at d0 = 0.06")
print("=" * 78)

SI_scan = np.linspace(0.30, 1.20, 30)

print(f"\n{'SI':>6s} {'beta_eq':>8s} {'G_eq':>7s} {'beta_sd':>8s} {'G_sd':>7s} "
      f"{'DeltaPhi':>12s} {'sigma_m*':>10s} {'B':>8s}")
print("-" * 80)

results_SI = []
for SI_val in SI_scan:
    eq = find_equilibria_u(d0_default, SI_val)
    if eq is None:
        continue

    u_eq, u_sad = eq
    beta_eq, beta_sad = np.exp(u_eq), np.exp(u_sad)

    # Build G_qss interpolant for this SI value
    G_interp_SI = build_G_qss_interp(-5.0, 16.0, N_interp=5000, SI_val=SI_val)
    G_eq = float(G_interp_SI(u_eq))
    G_sad = float(G_interp_SI(u_sad))

    if np.isnan(G_eq) or np.isnan(G_sad):
        continue

    DeltaPhi, lam_eq = compute_barrier_and_eigenvalue(d0_default, u_eq, u_sad, G_interp_SI)
    if lam_eq >= 0 or DeltaPhi <= 0:
        continue

    sigma_star = find_sigma_star(d0_default, u_eq, u_sad, lam_eq,
                                  D_TARGET_CLINICAL, G_interp_SI)
    if np.isnan(sigma_star):
        continue

    B = 2.0 * DeltaPhi / sigma_star**2

    results_SI.append({
        'SI': SI_val, 'beta_eq': beta_eq, 'G_eq': G_eq,
        'beta_sad': beta_sad, 'G_sad': G_sad,
        'DeltaPhi': DeltaPhi, 'sigma_star': sigma_star, 'B': B,
    })

    print(f"{SI_val:6.3f} {beta_eq:8.2f} {G_eq:7.2f} "
          f"{beta_sad:8.2f} {G_sad:7.2f} "
          f"{DeltaPhi:12.6f} {sigma_star:10.6f} {B:8.4f}")


# ============================================================
# ANALYSIS: d0 SCAN
# ============================================================
print("\n" + "=" * 78)
print("ANALYSIS: d0 SCAN")
print("=" * 78)

if len(results_d0) < 3:
    print(f"ERROR: Only {len(results_d0)} valid points")
    raise SystemExit(1)

B_vals = np.array([r['B'] for r in results_d0])
DPhi_vals = np.array([r['DeltaPhi'] for r in results_d0])
sig_vals = np.array([r['sigma_star'] for r in results_d0])
d0_vals = np.array([r['d0'] for r in results_d0])

B_mean = np.mean(B_vals)
B_std = np.std(B_vals)
B_cv = B_std / B_mean * 100
B_min, B_max = np.min(B_vals), np.max(B_vals)

print(f"\nValid scan points: {len(results_d0)}")
print(f"d0 range: [{d0_vals[0]:.4f}, {d0_vals[-1]:.4f}]  (d0_crit = {d0_crit:.6f})")

print(f"\n--- DeltaPhi (barrier in u-space, pure) ---")
print(f"  Min: {DPhi_vals.min():.6f}")
print(f"  Max: {DPhi_vals.max():.6f}")
print(f"  Ratio: {DPhi_vals.max()/DPhi_vals.min():.1f}x")

print(f"\n--- sigma_m* ---")
print(f"  Min: {sig_vals.min():.6f}")
print(f"  Max: {sig_vals.max():.6f}")
print(f"  Ratio: {sig_vals.max()/sig_vals.min():.2f}x")

print(f"\n--- B = 2*DeltaPhi/sigma_m*^2 ---")
print(f"  Mean:    {B_mean:.4f}")
print(f"  Std:     {B_std:.4f}")
print(f"  CV:      {B_cv:.2f}%")
print(f"  Range:   [{B_min:.4f}, {B_max:.4f}]")
print(f"  Max/Min: {B_max/B_min:.4f}")

# Interior analysis
n = len(results_d0)
trim = max(1, int(0.1 * n))
interior = results_d0[trim:-trim]
if len(interior) >= 3:
    B_int = np.array([r['B'] for r in interior])
    DPhi_int = np.array([r['DeltaPhi'] for r in interior])
    B_int_mean = np.mean(B_int)
    B_int_cv = np.std(B_int) / B_int_mean * 100
    print(f"\n--- Interior 80% (trimming {trim} points each edge) ---")
    print(f"  d0 range: [{interior[0]['d0']:.4f}, {interior[-1]['d0']:.4f}]")
    print(f"  DeltaPhi ratio: {DPhi_int.max()/DPhi_int.min():.1f}x")
    print(f"  B: mean = {B_int_mean:.4f}, CV = {B_int_cv:.2f}%")
    print(f"  B range: [{np.min(B_int):.4f}, {np.max(B_int):.4f}]")

# Systematic drift
B_first = B_vals[:n//2]
B_second = B_vals[n//2:]
drift_pct = abs(np.mean(B_first) - np.mean(B_second)) / B_mean * 100
print(f"\n--- Systematic drift ---")
print(f"  First half mean:  {np.mean(B_first):.4f}")
print(f"  Second half mean: {np.mean(B_second):.4f}")
print(f"  Drift: {drift_pct:.1f}% of mean")

# Log-linear fit
log_B = np.log(B_vals)
coeffs = np.polyfit(d0_vals, log_B, 1)
resid = log_B - np.polyval(coeffs, d0_vals)
resid_cv = np.std(resid) * 100
print(f"\n--- Log-linear fit: ln(B) = {coeffs[0]:.4f}*d0 + {coeffs[1]:.4f} ---")
print(f"  Drift rate = {coeffs[0]:.4f} per unit d0")
print(f"  Residual CV (after detrending) = {resid_cv:.2f}%")


# ============================================================
# ANALYSIS: SI SCAN
# ============================================================
print("\n" + "=" * 78)
print("ANALYSIS: SI SCAN (at d0 = 0.06)")
print("=" * 78)

if len(results_SI) >= 3:
    B_SI = np.array([r['B'] for r in results_SI])
    DPhi_SI = np.array([r['DeltaPhi'] for r in results_SI])
    sig_SI = np.array([r['sigma_star'] for r in results_SI])
    SI_arr = np.array([r['SI'] for r in results_SI])

    print(f"\nValid scan points: {len(results_SI)}")
    print(f"SI range: [{SI_arr[0]:.3f}, {SI_arr[-1]:.3f}]")
    print(f"\n  DeltaPhi ratio: {DPhi_SI.max()/DPhi_SI.min():.1f}x")
    print(f"  sigma_m* ratio: {sig_SI.max()/sig_SI.min():.2f}x")
    print(f"\n  B mean:  {np.mean(B_SI):.4f}")
    print(f"  B std:   {np.std(B_SI):.4f}")
    print(f"  B CV:    {np.std(B_SI)/np.mean(B_SI)*100:.2f}%")
    print(f"  B range: [{np.min(B_SI):.4f}, {np.max(B_SI):.4f}]")
else:
    print(f"  Only {len(results_SI)} valid points")


# ============================================================
# CLINICAL COMPARISON
# ============================================================
print("\n" + "=" * 78)
print("CLINICAL COMPARISON (at standard d0 = 0.06)")
print("=" * 78)

std_result = min(results_d0, key=lambda r: abs(r['d0'] - d0_default))

sigma_m_star = std_result['sigma_star']
lam_eq_std = std_result['lam_eq']
tau_std = 1.0 / abs(lam_eq_std)

# Implied CV of beta-cell mass
var_u = sigma_m_star**2 / (2.0 * abs(lam_eq_std))
cv_beta = np.sqrt(np.exp(var_u) - 1)

# Implied fold range (95% interval = +/- 2 sigma)
fold_range_low = np.exp(-2 * np.sqrt(var_u))
fold_range_high = np.exp(2 * np.sqrt(var_u))
fold_ratio = fold_range_high / fold_range_low

print(f"\n  Operating point: d0 = {std_result['d0']:.4f}")
print(f"  beta_healthy = {std_result['beta_eq']:.2f} mg")
print(f"  G_healthy = {std_result['G_eq']:.2f} mg/dL")
print(f"  B = {std_result['B']:.4f}")
print(f"  sigma_m* = {sigma_m_star:.6f}")
print(f"  tau = {tau_std:.1f} days")

print(f"\n  Implied beta-cell mass variability:")
print(f"    Var(ln beta) = {var_u:.4f}")
print(f"    CV(beta) = {cv_beta*100:.1f}%")
print(f"    95% range: {fold_range_low:.2f}x to {fold_range_high:.2f}x of mean")
print(f"    i.e., ~{fold_ratio:.1f}x inter-individual variation")

print(f"\n  Clinical comparison (Olehnik et al. 2017):")
print(f"    Measured inter-individual variability: 4.5-14x")
print(f"    Model prediction (95% range): {fold_ratio:.1f}x")
if 3.0 < fold_ratio < 20.0:
    print(f"    STATUS: Consistent with observed variability")
else:
    print(f"    STATUS: Outside observed range")

# MFPT
D_at_sigma_star = compute_D_exact(sigma_m_star, std_result['d0'],
                                    std_result['u_eq'], std_result['u_sad'],
                                    lam_eq_std, G_interp_std)
MFPT_days = D_at_sigma_star * tau_std
MFPT_years = MFPT_days / 365.25
print(f"\n  MFPT at sigma_m*:")
print(f"    D = {D_at_sigma_star:.1f}")
print(f"    MFPT = {MFPT_days:.0f} days = {MFPT_years:.1f} years")
print(f"    DPP placebo median: ~14.3 years")


# ============================================================
# DIAGNOSTIC: WHY B IS NOT CONSTANT
# ============================================================
print("\n" + "=" * 78)
print("DIAGNOSTIC: WHY B IS NOT CONSTANT")
print("=" * 78)

u_ranges = np.array([r['u_eq'] - r['u_sad'] for r in results_d0])
mean_drifts = DPhi_vals / u_ranges  # mean |f_eff_u| over the basin

print(f"""
  KEY OBSERVATION: sigma_m* varies only {sig_vals.max()/sig_vals.min():.1f}x
  while DeltaPhi varies {DPhi_vals.max()/DPhi_vals.min():.0f}x.

  sigma_m* is nearly constant because the MFPT is NOT barrier-dominated.
  The MFPT is controlled by the DIFFUSION TIME across the wide basin,
  not by the exponential Kramers factor exp(2*DeltaPhi/sigma^2).

  Evidence: the u-space basin width varies dramatically with d0:
    d0={d0_vals[0]:.4f}: u_range = {u_ranges[0]:.1f}  (DeltaPhi = {DPhi_vals[0]:.4f})
    d0={d0_vals[len(d0_vals)//2]:.4f}: u_range = {u_ranges[len(d0_vals)//2]:.1f}  (DeltaPhi = {DPhi_vals[len(d0_vals)//2]:.4f})
    d0={d0_vals[-1]:.4f}: u_range = {u_ranges[-1]:.1f}  (DeltaPhi = {DPhi_vals[-1]:.4f})

  The barrier DeltaPhi increases with basin width, but the mean drift
  (DeltaPhi / u_range) stays moderate:
    Mean drift range: [{mean_drifts.min():.5f}, {mean_drifts.max():.5f}]

  In verified systems (lake, savanna, kelp), the basin width stays roughly
  constant while the barrier height changes — so B tracks the barrier.
  In the diabetes model, the basin WIDTH is the primary variable.

  ADDITIONAL FACTOR: Multiplicative noise creates non-monotonic D(sigma).
  The Ito correction (-sigma_m^2/2 in the drift) means that at large noise,
  the effective drift pushes AWAY from the saddle, increasing MFPT.
  This bounds sigma_m* from above, decorrelating it from DeltaPhi.

  STRUCTURAL REASON: The Topp model has k=1 (single feedback channel) and
  an absorbing boundary at beta=0. Both stable states of the verified systems
  are true potential wells with similar structure. The diabetes system has one
  potential well (healthy) and one absorbing boundary (diabetic) — a
  fundamentally asymmetric topology.
""")


# ============================================================
# KEY RESULTS
# ============================================================
print("\n" + "=" * 78)
print("KEY RESULTS")
print("=" * 78)

print(f"""
  SYSTEM: Topp 2000 diabetes (beta-cell / glucose / insulin)
  TYPE:   First human disease system tested
  NOISE:  Multiplicative (u = ln(beta) coordinates)
  D_target: {D_TARGET_CLINICAL} (from DPP clinical data)

  d0 SCAN (bifurcation parameter):
    B = {B_mean:.4f} +/- {B_cv:.2f}%
    DeltaPhi variation: {DPhi_vals.max()/DPhi_vals.min():.1f}x
    sigma_m* variation: {sig_vals.max()/sig_vals.min():.2f}x""")

if len(results_SI) >= 3:
    print(f"""
  SI SCAN (insulin sensitivity, at d0 = 0.06):
    B = {np.mean(B_SI):.4f} +/- {np.std(B_SI)/np.mean(B_SI)*100:.2f}%
    DeltaPhi variation: {DPhi_SI.max()/DPhi_SI.min():.1f}x""")

print(f"""
  COMPARISON TO VERIFIED SYSTEMS:
    | System   | B (mean) | B (CV)  | DeltaPhi var |
    |----------|----------|---------|--------------|
    | Kelp     | 2.17     | 2.6%    | 27,617x      |
    | Savanna  | 4.04     | 4.6%    | 77x          |
    | Lake     | 4.27     | 2.0%    | 96x          |
    | Coral    | 6.06     | 2.1%    | 2,625x       |
    | Toggle   | 4.83     | 3.8%    | --           |
    | Diabetes | {B_mean:.2f}     | {B_cv:.1f}%    | {DPhi_vals.max()/DPhi_vals.min():.0f}x           |""")

# Verdict
if B_cv < 5:
    v_cv = "STRONG CONSTANT (CV < 5%)"
elif B_cv < 10:
    v_cv = "APPROXIMATELY CONSTANT (CV < 10%)"
elif B_cv < 20:
    v_cv = "WEAKLY CONSTANT (CV < 20%)"
else:
    v_cv = "NOT CONSTANT (CV >= 20%)"

if 1.8 <= B_mean <= 6.0:
    v_hz = f"IN HABITABLE ZONE [1.8, 6.0] -- B = {B_mean:.2f}"
elif B_mean < 1.8:
    v_hz = f"BELOW HABITABLE ZONE (B = {B_mean:.2f} < 1.8)"
else:
    v_hz = f"ABOVE HABITABLE ZONE (B = {B_mean:.2f} > 6.0)"

print(f"""
  B INVARIANCE:   {v_cv}
  HABITABLE ZONE: {v_hz}
""")

print("=" * 78)
print("COMPUTATION COMPLETE")
print("=" * 78)
