#!/usr/bin/env python3
"""
STRUCTURAL CONNECTION TEST
==========================
Core question: Is sigma* (from the bridge, D_product=200) structurally
related to the loading parameter 'a' and eigenvalue lambda_eq across
the full bistable range?

If sigma*(a) * |lambda_eq(a)| / a = constant, then that constant IS
the predicted CV of environmental forcing, derivable from the ODE alone.

If it matches observed CV (~0.4 for lake TP), the structural connection
between noise and feedback is proven for this system.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, fsolve

# ============================================================
# Lake model: van Nes & Scheffer 2007
# ============================================================
b = 0.8      # P loss rate
r = 1.0      # max recycling rate
q = 8        # Hill coefficient
h_param = 1.0  # half-saturation

def f_lake(x, a_val):
    """Drift function."""
    return a_val - b * x + r * x**q / (x**q + h_param**q)

def f_lake_deriv(x, a_val):
    """f'(x) for eigenvalue computation."""
    return -b + r * q * x**(q-1) * h_param**q / (x**q + h_param**q)**2

def find_equilibria(a_val, x_scan=np.linspace(0.001, 3.0, 10000)):
    """Find all equilibria by scanning for sign changes in f."""
    f_vals = np.array([f_lake(x, a_val) for x in x_scan])
    roots = []
    for i in range(len(f_vals) - 1):
        if f_vals[i] * f_vals[i+1] < 0:
            root = brentq(lambda x: f_lake(x, a_val), x_scan[i], x_scan[i+1])
            roots.append(root)
    return sorted(roots)

def compute_barrier(a_val, x_eq, x_sad):
    """Compute barrier DeltaPhi = -integral of f from x_eq to x_sad."""
    result, _ = quad(lambda x: -f_lake(x, a_val), x_eq, x_sad)
    return result

def compute_D_exact(sigma, a_val, x_eq, x_sad, lam_eq):
    """Compute exact MFPT-based D for given sigma."""
    tau = 1.0 / abs(lam_eq)

    N_grid = 50000
    x_lo = max(0.001, x_eq - 3 * sigma / np.sqrt(2 * abs(lam_eq)))
    x_hi = x_sad + 0.001
    x_grid = np.linspace(x_lo, x_hi, N_grid)
    dx_g = x_grid[1] - x_grid[0]

    # Potential
    neg_f = np.array([-f_lake(x, a_val) for x in x_grid])
    U_raw = np.cumsum(neg_f) * dx_g
    i_eq = np.argmin(np.abs(x_grid - x_eq))
    U_grid = U_raw - U_raw[i_eq]

    Phi = 2 * U_grid / sigma**2

    # Clamp to avoid overflow
    Phi = np.clip(Phi, -500, 500)

    exp_neg_Phi = np.exp(-Phi)
    I_x = np.cumsum(exp_neg_Phi) * dx_g

    psi = (2 / sigma**2) * np.exp(Phi) * I_x

    i_sad = np.argmin(np.abs(x_grid - x_sad))
    if i_eq >= i_sad:
        return 1e10  # degenerate

    MFPT = np.trapz(psi[i_eq:i_sad + 1], x_grid[i_eq:i_sad + 1])
    D = MFPT / tau
    return D

def find_sigma_star(a_val, x_eq, x_sad, lam_eq, D_target=200.0):
    """Find sigma where D_exact = D_target using bisection."""
    def objective(log_sigma):
        sigma = np.exp(log_sigma)
        D = compute_D_exact(sigma, a_val, x_eq, x_sad, lam_eq)
        return np.log(D) - np.log(D_target)

    # Bracket: small sigma -> huge D, large sigma -> small D
    try:
        log_sigma_star = brentq(objective, np.log(0.001), np.log(2.0), xtol=1e-8)
        return np.exp(log_sigma_star)
    except ValueError:
        return np.nan

# ============================================================
# Find bistable range of 'a'
# ============================================================
print("=" * 70)
print("STRUCTURAL CONNECTION TEST")
print("=" * 70)

print("\n--- Finding bistable range ---")
a_test = np.linspace(0.01, 0.80, 1000)
n_roots = []
for a_val in a_test:
    roots = find_equilibria(a_val)
    n_roots.append(len(roots))

n_roots = np.array(n_roots)
bistable_mask = n_roots == 3
a_bistable = a_test[bistable_mask]
a_min_bi = a_bistable[0]
a_max_bi = a_bistable[-1]
print(f"Bistable range: a in [{a_min_bi:.4f}, {a_max_bi:.4f}]")

# ============================================================
# Scan across bistable range
# ============================================================
print("\n--- Scanning across bistable range ---")
print(f"{'a':>8s} {'x_eq':>8s} {'x_sad':>8s} {'lambda_eq':>10s} {'DeltaPhi':>12s} {'sigma*':>10s} {'sigma*/a':>10s} {'sig*|lam|/a':>12s}")
print("-" * 95)

# Use 20 points across the bistable range, avoiding the very edges (fold bifurcations)
a_margin = 0.05 * (a_max_bi - a_min_bi)
a_scan = np.linspace(a_min_bi + a_margin, a_max_bi - a_margin, 25)

results = []
for a_val in a_scan:
    roots = find_equilibria(a_val)
    if len(roots) != 3:
        continue

    x_eq = roots[0]   # clear-water (lower stable)
    x_sad = roots[1]   # saddle
    x_turb = roots[2]  # turbid (upper stable)

    lam_eq = f_lake_deriv(x_eq, a_val)
    lam_sad = f_lake_deriv(x_sad, a_val)

    if lam_eq >= 0:  # not stable
        continue

    DeltaPhi = compute_barrier(a_val, x_eq, x_sad)
    if DeltaPhi <= 0:
        continue

    sigma_star = find_sigma_star(a_val, x_eq, x_sad, lam_eq, D_target=200.0)
    if np.isnan(sigma_star):
        continue

    ratio1 = sigma_star / a_val
    ratio2 = sigma_star * abs(lam_eq) / a_val

    # Also compute: implied CV of state = SD(x)/x_eq
    SD_x = sigma_star / np.sqrt(2 * abs(lam_eq))
    CV_state = SD_x / x_eq

    # Implied forcing SD if sigma = SD(forcing) * sqrt(2/|lambda|) [white noise]
    SD_forcing_white = sigma_star * np.sqrt(abs(lam_eq) / 2)
    CV_forcing_white = SD_forcing_white / a_val

    # Implied forcing SD if SD(x) = SD(forcing)/|lambda| [quasi-static]
    SD_forcing_qs = SD_x * abs(lam_eq)
    CV_forcing_qs = SD_forcing_qs / a_val

    results.append({
        'a': a_val, 'x_eq': x_eq, 'x_sad': x_sad,
        'lam_eq': lam_eq, 'lam_sad': lam_sad,
        'DeltaPhi': DeltaPhi, 'sigma_star': sigma_star,
        'ratio1': ratio1, 'ratio2': ratio2,
        'CV_state': CV_state,
        'CV_forcing_white': CV_forcing_white,
        'CV_forcing_qs': CV_forcing_qs,
        'SD_forcing_white': SD_forcing_white,
        'SD_forcing_qs': SD_forcing_qs,
    })

    print(f"{a_val:8.4f} {x_eq:8.4f} {x_sad:8.4f} {lam_eq:10.4f} {DeltaPhi:12.8f} {sigma_star:10.6f} {ratio1:10.4f} {ratio2:10.4f}")

# ============================================================
# Analysis: check for constancy of ratios
# ============================================================
print("\n" + "=" * 70)
print("RATIO ANALYSIS")
print("=" * 70)

if len(results) > 3:
    ratio1_vals = [r['ratio1'] for r in results]
    ratio2_vals = [r['ratio2'] for r in results]
    cv_state_vals = [r['CV_state'] for r in results]
    cv_fw_vals = [r['CV_forcing_white'] for r in results]
    cv_fq_vals = [r['CV_forcing_qs'] for r in results]
    a_vals = [r['a'] for r in results]

    print(f"\nRatio sigma*/a:")
    print(f"  Mean = {np.mean(ratio1_vals):.4f}")
    print(f"  Std  = {np.std(ratio1_vals):.4f}")
    print(f"  CV   = {np.std(ratio1_vals)/np.mean(ratio1_vals)*100:.1f}%")
    print(f"  Range: [{min(ratio1_vals):.4f}, {max(ratio1_vals):.4f}]")

    print(f"\nRatio sigma*|lambda|/a:")
    print(f"  Mean = {np.mean(ratio2_vals):.4f}")
    print(f"  Std  = {np.std(ratio2_vals):.4f}")
    print(f"  CV   = {np.std(ratio2_vals)/np.mean(ratio2_vals)*100:.1f}%")
    print(f"  Range: [{min(ratio2_vals):.4f}, {max(ratio2_vals):.4f}]")

    print(f"\nImplied CV of STATE variable:")
    print(f"  Mean = {np.mean(cv_state_vals):.4f}")
    print(f"  Std  = {np.std(cv_state_vals):.4f}")
    print(f"  CV   = {np.std(cv_state_vals)/np.mean(cv_state_vals)*100:.1f}%")
    print(f"  Range: [{min(cv_state_vals):.4f}, {max(cv_state_vals):.4f}]")

    print(f"\nImplied CV of FORCING (white noise model):")
    print(f"  Mean = {np.mean(cv_fw_vals):.4f}")
    print(f"  Std  = {np.std(cv_fw_vals):.4f}")
    print(f"  CV   = {np.std(cv_fw_vals)/np.mean(cv_fw_vals)*100:.1f}%")
    print(f"  Range: [{min(cv_fw_vals):.4f}, {max(cv_fw_vals):.4f}]")

    print(f"\nImplied CV of FORCING (quasi-static model):")
    print(f"  Mean = {np.mean(cv_fq_vals):.4f}")
    print(f"  Std  = {np.std(cv_fq_vals):.4f}")
    print(f"  CV   = {np.std(cv_fq_vals)/np.mean(cv_fq_vals)*100:.1f}%")
    print(f"  Range: [{min(cv_fq_vals):.4f}, {max(cv_fq_vals):.4f}]")

    # Check at the known operating point a=0.326588
    print(f"\n--- At known operating point a=0.326588 ---")
    closest = min(results, key=lambda r: abs(r['a'] - 0.326588))
    print(f"  a = {closest['a']:.4f}")
    print(f"  sigma* = {closest['sigma_star']:.6f}  (expected: ~0.175)")
    print(f"  lambda_eq = {closest['lam_eq']:.6f}  (expected: -0.7847)")
    print(f"  DeltaPhi = {closest['DeltaPhi']:.8f}")
    print(f"  CV_state = {closest['CV_state']:.4f}  (observed: ~0.341)")
    print(f"  CV_forcing (white) = {closest['CV_forcing_white']:.4f}")
    print(f"  CV_forcing (quasi-static) = {closest['CV_forcing_qs']:.4f}")
    print(f"  sigma*/a = {closest['ratio1']:.4f}")

    # Key test: is any ratio approximately constant?
    print(f"\n" + "=" * 70)
    print("KEY RESULT")
    print("=" * 70)
    best_cv = min(
        [('sigma*/a', ratio1_vals), ('sigma*|lam|/a', ratio2_vals),
         ('CV_state', cv_state_vals), ('CV_forcing_white', cv_fw_vals),
         ('CV_forcing_qs', cv_fq_vals)],
        key=lambda pair: np.std(pair[1]) / np.mean(pair[1])
    )
    print(f"\nMost constant ratio: {best_cv[0]}")
    print(f"  Mean = {np.mean(best_cv[1]):.4f}")
    print(f"  CV of the ratio itself = {np.std(best_cv[1])/np.mean(best_cv[1])*100:.1f}%")
    print(f"\nIf this CV < ~10%, the structural connection is strong.")
    print(f"If this CV < ~5%, it is essentially a constant of the ODE.")

    # Comparison to observed environmental CV
    print(f"\n--- Comparison to observed environmental data ---")
    print(f"Hakanson (2000): CV(TP) in well-studied lakes = 35%")
    print(f"Nature Comms 2024: CV(TP) across 159 shallow lakes = 35%")
    print(f"Lake Veluwe turbid-state monthly CV = 32.6%")
    print(f"Predicted CV_state at a=0.3266: {closest['CV_state']*100:.1f}%")

    # Dimensionless barrier B across range
    print(f"\n--- Dimensionless barrier B = 2*DeltaPhi/sigma*^2 ---")
    for res in results:
        B = 2 * res['DeltaPhi'] / res['sigma_star']**2
        print(f"  a={res['a']:.4f}  B={B:.3f}")

else:
    print("ERROR: Too few valid points for analysis")
