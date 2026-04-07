#!/usr/bin/env python3
"""
Test 3: K Universality — Is K ≈ 0.55 universal across SDE systems?

Computes K = D_exact / D_Kramers for:
  1. Lake model at varying loading (9 loading values × 3 noise levels)
  2. Symmetric double-well at varying alpha (7 alpha × 5 noise levels)
  3. Schlögl model at varying asymmetry (7 a-values × 3 noise levels)
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Core computation engine
# ==============================================================================

def compute_D_exact_and_kramers(f_func, f_deriv, x_eq, x_sad, sigma, x_lo=0.001):
    """
    Compute D_exact and D_Kramers for a 1D system.
    Returns: D_exact, D_kramers, K, DeltaPhi, barrier, tau
    """
    lam_eq = abs(f_deriv(x_eq))
    lam_sad = abs(f_deriv(x_sad))
    tau = 1.0 / lam_eq

    # Barrier
    DeltaPhi, _ = quad(lambda x: -f_func(x), x_eq, x_sad)
    barrier = 2 * DeltaPhi / sigma**2

    # Kramers
    C_std = np.sqrt(lam_eq * lam_sad) / (2 * np.pi)
    D_kramers = np.exp(barrier) / (C_std * tau)

    # Exact MFPT via double integral
    N_grid = 50000
    x_grid = np.linspace(x_lo, x_sad + 0.001, N_grid)
    dx_g = x_grid[1] - x_grid[0]

    neg_f = np.array([-f_func(x) for x in x_grid])
    U_raw = np.cumsum(neg_f) * dx_g
    i_eq = np.argmin(np.abs(x_grid - x_eq))
    U_grid = U_raw - U_raw[i_eq]

    Phi = 2 * U_grid / sigma**2

    # Overflow protection
    Phi_max = Phi.max()
    if Phi_max > 700:
        # Shift to prevent overflow: doesn't affect K since it cancels
        Phi_shifted = Phi - Phi_max + 500
        exp_neg_Phi = np.exp(-Phi_shifted)
        I_x = np.cumsum(exp_neg_Phi) * dx_g
        psi = (2 / sigma**2) * np.exp(Phi_shifted) * I_x
    else:
        exp_neg_Phi = np.exp(-Phi)
        I_x = np.cumsum(exp_neg_Phi) * dx_g
        psi = (2 / sigma**2) * np.exp(Phi) * I_x

    i_sad = np.argmin(np.abs(x_grid - x_sad))
    MFPT_exact = np.trapz(psi[i_eq:i_sad + 1], x_grid[i_eq:i_sad + 1])
    D_exact = MFPT_exact / tau

    K = D_exact / D_kramers if D_kramers > 0 else np.nan

    return D_exact, D_kramers, K, DeltaPhi, barrier, tau


# ==============================================================================
# SYSTEM 1: Lake model
# ==============================================================================
print("=" * 80)
print("SYSTEM 1: LAKE MODEL")
print("=" * 80)

b_lake = 0.8
r_lake = 1.0
q_lake = 8
h_lake = 1.0

def f_lake(x, a):
    return a - b_lake * x + r_lake * x**q_lake / (x**q_lake + h_lake**q_lake)

def f_lake_deriv(x, a):
    return -b_lake + r_lake * q_lake * x**(q_lake-1) * h_lake**q_lake / (x**q_lake + h_lake**q_lake)**2

def find_lake_roots(a, x_range=(0.01, 2.5), n_scan=5000):
    """Find roots of f_lake(x, a) = 0 by scanning for sign changes."""
    xs = np.linspace(x_range[0], x_range[1], n_scan)
    fs = np.array([f_lake(x, a) for x in xs])
    roots = []
    for i in range(len(fs) - 1):
        if fs[i] * fs[i+1] < 0:
            try:
                root = brentq(lambda x: f_lake(x, a), xs[i], xs[i+1])
                roots.append(root)
            except:
                pass
    return sorted(roots)

a_values_lake = [0.15, 0.20, 0.25, 0.30, 0.326588, 0.35, 0.37, 0.38, 0.385]
cv_values_lake = [0.30, 0.35, 0.40]

lake_results = []

for a_val in a_values_lake:
    roots = find_lake_roots(a_val)
    if len(roots) < 3:
        print(f"  a = {a_val:.4f}: Only {len(roots)} roots found — NOT BISTABLE, skipping")
        continue

    x_clear = roots[0]
    x_sad = roots[1]
    x_turb = roots[2]

    lam_eq = f_lake_deriv(x_clear, a_val)
    lam_sad = f_lake_deriv(x_sad, a_val)

    if lam_eq > 0 or lam_sad < 0:
        print(f"  a = {a_val:.4f}: Eigenvalue signs wrong — skipping")
        continue

    DeltaPhi_check, _ = quad(lambda x: -f_lake(x, a_val), x_clear, x_sad)

    print(f"\n  a = {a_val:.6f}: x_eq={x_clear:.6f}, x_sad={x_sad:.6f}, "
          f"λ_eq={lam_eq:.6f}, λ_sad={lam_sad:.6f}, ΔΦ={DeltaPhi_check:.8f}")

    for cv in cv_values_lake:
        sigma = cv * x_clear * np.sqrt(2 * abs(lam_eq))
        try:
            D_ex, D_kr, K, dPhi, barrier, tau = compute_D_exact_and_kramers(
                lambda x: f_lake(x, a_val),
                lambda x: f_lake_deriv(x, a_val),
                x_clear, x_sad, sigma, x_lo=0.001
            )
            lake_results.append({
                'system': 'Lake',
                'params': f'a={a_val:.4f}',
                'x_eq': x_clear,
                'x_sad': x_sad,
                'DeltaPhi': dPhi,
                'barrier': barrier,
                'cv': cv,
                'sigma': sigma,
                'D_exact': D_ex,
                'D_kramers': D_kr,
                'K': K
            })
            print(f"    CV={cv:.2f}: σ={sigma:.6f}, 2ΔΦ/σ²={barrier:.4f}, "
                  f"D_exact={D_ex:.4e}, D_Kramers={D_kr:.4e}, K={K:.6f}")
        except Exception as e:
            print(f"    CV={cv:.2f}: FAILED — {e}")


# ==============================================================================
# SYSTEM 2: Symmetric double-well  dx/dt = αx - x³
# ==============================================================================
print("\n" + "=" * 80)
print("SYSTEM 2: SYMMETRIC DOUBLE-WELL (αx - x³)")
print("=" * 80)

alpha_values = [0.10, 0.25, 0.50, 1.00, 2.00, 3.00, 5.00]
target_barriers = [2.0, 4.0, 6.0, 8.0, 10.0]

dw_results = []

for alpha in alpha_values:
    x_eq = np.sqrt(alpha)
    x_sad = 0.0
    DeltaPhi_analytic = alpha**2 / 4.0

    lam_eq = -(2 * alpha)   # f'(±√α) = α - 3α = -2α
    lam_sad_val = alpha      # f'(0) = α

    print(f"\n  α = {alpha:.2f}: x_eq={x_eq:.6f}, x_sad=0, "
          f"λ_eq={lam_eq:.6f}, λ_sad={lam_sad_val:.6f}, ΔΦ={DeltaPhi_analytic:.8f}")

    for target_b in target_barriers:
        # 2ΔΦ/σ² = target_b => σ² = 2ΔΦ/target_b
        sigma_sq = 2 * DeltaPhi_analytic / target_b
        sigma = np.sqrt(sigma_sq)

        # For double-well with saddle at x=0, we need x_lo < 0
        # But the escape is from x_eq = +√α to x_sad = 0
        # The reflecting boundary should be far to the right of x_eq
        # Actually: we integrate from x_lo to x_sad, with x_eq in between.
        # Since saddle is at 0 and x_eq is at √α, the well is to the RIGHT of the saddle.
        # The MFPT formula integrates from x_eq to x_sad.
        # But x_eq > x_sad here (x_eq = √α > 0 = x_sad).
        #
        # For the standard formula, we need x_eq < x_sad. So let's use the
        # negative equilibrium x_eq = -√α, saddle = 0.
        # Then x_eq = -√α < 0 = x_sad. ✓
        #
        # f(-√α) = α(-√α) - (-√α)³ = -α^{3/2} + α^{3/2} = 0 ✓
        # f'(-√α) = α - 3α = -2α ✓ (stable)
        # f'(0) = α > 0 ✓ (unstable saddle)

        x_eq_neg = -np.sqrt(alpha)
        x_lo_dw = x_eq_neg - 2.0  # reflecting boundary well below the eq

        try:
            D_ex, D_kr, K, dPhi, barrier, tau = compute_D_exact_and_kramers(
                lambda x, a=alpha: a * x - x**3,
                lambda x, a=alpha: a - 3 * x**2,
                x_eq_neg, x_sad, sigma, x_lo=x_lo_dw
            )
            dw_results.append({
                'system': 'Double-well',
                'params': f'α={alpha:.2f}',
                'x_eq': x_eq_neg,
                'x_sad': x_sad,
                'DeltaPhi': dPhi,
                'barrier': barrier,
                'cv': None,
                'sigma': sigma,
                'D_exact': D_ex,
                'D_kramers': D_kr,
                'K': K
            })
            print(f"    2ΔΦ/σ²={target_b:.1f}: σ={sigma:.6f}, "
                  f"D_exact={D_ex:.4e}, D_Kramers={D_kr:.4e}, K={K:.6f}")
        except Exception as e:
            print(f"    2ΔΦ/σ²={target_b:.1f}: FAILED — {e}")


# ==============================================================================
# SYSTEM 3: Schlögl model  dx/dt = a + bx - x³
# ==============================================================================
print("\n" + "=" * 80)
print("SYSTEM 3: SCHLÖGL MODEL (a + bx - x³)")
print("=" * 80)

b_sch = 3.0
a_values_schlogl = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
cv_values_schlogl = [0.30, 0.35, 0.40]

schlogl_results = []

for a_sch in a_values_schlogl:
    # Find roots of a + bx - x³ = 0
    # Scan for sign changes
    xs = np.linspace(-4.0, 4.0, 10000)
    fs = np.array([a_sch + b_sch * x - x**3 for x in xs])
    roots = []
    for i in range(len(fs) - 1):
        if fs[i] * fs[i+1] < 0:
            try:
                root = brentq(lambda x: a_sch + b_sch * x - x**3, xs[i], xs[i+1])
                roots.append(root)
            except:
                pass
    roots = sorted(roots)

    if len(roots) < 3:
        print(f"  a = {a_sch:.1f}: Only {len(roots)} roots — NOT BISTABLE, skipping")
        continue

    x_eq_sch = roots[0]   # smallest root (stable)
    x_sad_sch = roots[1]  # middle root (saddle)
    x_turb_sch = roots[2] # largest root (stable)

    lam_eq_sch = b_sch - 3 * x_eq_sch**2
    lam_sad_sch = b_sch - 3 * x_sad_sch**2

    if lam_eq_sch > 0 or lam_sad_sch < 0:
        print(f"  a = {a_sch:.1f}: Eigenvalue signs wrong (λ_eq={lam_eq_sch:.4f}, λ_sad={lam_sad_sch:.4f}) — skipping")
        continue

    DeltaPhi_sch, _ = quad(lambda x: -(a_sch + b_sch * x - x**3), x_eq_sch, x_sad_sch)

    print(f"\n  a = {a_sch:.1f}: x_eq={x_eq_sch:.6f}, x_sad={x_sad_sch:.6f}, "
          f"λ_eq={lam_eq_sch:.6f}, λ_sad={lam_sad_sch:.6f}, ΔΦ={DeltaPhi_sch:.8f}")

    # Handle x_eq possibly being negative -> CV convention needs |x_eq|
    x_eq_abs = abs(x_eq_sch)
    if x_eq_abs < 0.01:
        print(f"    x_eq ≈ 0, using direct σ values instead of CV convention")
        # Use σ values that give reasonable barriers
        for target_b in [2.0, 4.0, 6.0]:
            sigma_sq = 2 * DeltaPhi_sch / target_b
            if sigma_sq <= 0:
                continue
            sigma = np.sqrt(sigma_sq)
            x_lo_sch = x_eq_sch - 2.0
            try:
                D_ex, D_kr, K, dPhi, barrier, tau = compute_D_exact_and_kramers(
                    lambda x, aa=a_sch: aa + b_sch * x - x**3,
                    lambda x: b_sch - 3 * x**2,
                    x_eq_sch, x_sad_sch, sigma, x_lo=x_lo_sch
                )
                schlogl_results.append({
                    'system': 'Schlögl',
                    'params': f'a={a_sch:.1f}',
                    'x_eq': x_eq_sch,
                    'x_sad': x_sad_sch,
                    'DeltaPhi': dPhi,
                    'barrier': barrier,
                    'cv': None,
                    'sigma': sigma,
                    'D_exact': D_ex,
                    'D_kramers': D_kr,
                    'K': K
                })
                print(f"    2ΔΦ/σ²={target_b:.1f}: σ={sigma:.6f}, "
                      f"D_exact={D_ex:.4e}, D_Kramers={D_kr:.4e}, K={K:.6f}")
            except Exception as e:
                print(f"    2ΔΦ/σ²={target_b:.1f}: FAILED — {e}")
    else:
        for cv in cv_values_schlogl:
            sigma = cv * x_eq_abs * np.sqrt(2 * abs(lam_eq_sch))
            x_lo_sch = x_eq_sch - 2.0
            try:
                D_ex, D_kr, K, dPhi, barrier, tau = compute_D_exact_and_kramers(
                    lambda x, aa=a_sch: aa + b_sch * x - x**3,
                    lambda x: b_sch - 3 * x**2,
                    x_eq_sch, x_sad_sch, sigma, x_lo=x_lo_sch
                )
                schlogl_results.append({
                    'system': 'Schlögl',
                    'params': f'a={a_sch:.1f}',
                    'x_eq': x_eq_sch,
                    'x_sad': x_sad_sch,
                    'DeltaPhi': dPhi,
                    'barrier': barrier,
                    'cv': cv,
                    'sigma': sigma,
                    'D_exact': D_ex,
                    'D_kramers': D_kr,
                    'K': K
                })
                print(f"    CV={cv:.2f}: σ={sigma:.6f}, 2ΔΦ/σ²={barrier:.4f}, "
                      f"D_exact={D_ex:.4e}, D_Kramers={D_kr:.4e}, K={K:.6f}")
            except Exception as e:
                print(f"    CV={cv:.2f}: FAILED — {e}")


# ==============================================================================
# ANALYSIS
# ==============================================================================
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

all_results = lake_results + dw_results + schlogl_results

# Filter out any with invalid K
valid = [r for r in all_results if np.isfinite(r['K']) and r['K'] > 0 and r['barrier'] > 0]

print(f"\nTotal valid data points: {len(valid)}")
print(f"  Lake: {len([r for r in valid if r['system']=='Lake'])}")
print(f"  Double-well: {len([r for r in valid if r['system']=='Double-well'])}")
print(f"  Schlögl: {len([r for r in valid if r['system']=='Schlögl'])}")

# ---- Master table ----
print("\n" + "-" * 120)
print(f"{'System':<12} {'Params':<14} {'x_eq':<10} {'x_sad':<10} {'ΔΦ':<12} "
      f"{'2ΔΦ/σ²':<10} {'D_exact':<14} {'D_Kramers':<14} {'K':<10}")
print("-" * 120)
for r in valid:
    print(f"{r['system']:<12} {r['params']:<14} {r['x_eq']:<10.6f} {r['x_sad']:<10.6f} "
          f"{r['DeltaPhi']:<12.8f} {r['barrier']:<10.4f} {r['D_exact']:<14.4e} "
          f"{r['D_kramers']:<14.4e} {r['K']:<10.6f}")

# ---- K vs barrier height summary ----
print("\n\nK by barrier range:")
print(f"{'Barrier range':<20} {'Mean K':<10} {'Std K':<10} {'Min K':<10} {'Max K':<10} {'N':<5}")
print("-" * 65)
ranges = [(1, 3), (3, 5), (5, 8), (8, 12), (12, float('inf'))]
range_labels = ['1-3', '3-5', '5-8', '8-12', '12+']
for (lo, hi), label in zip(ranges, range_labels):
    subset = [r for r in valid if lo <= r['barrier'] < hi]
    if subset:
        Ks = [r['K'] for r in subset]
        print(f"{label:<20} {np.mean(Ks):<10.4f} {np.std(Ks):<10.4f} "
              f"{np.min(Ks):<10.4f} {np.max(Ks):<10.4f} {len(Ks):<5d}")
    else:
        print(f"{label:<20} {'(no data)':<10}")

# Also check barrier < 1
subset_low = [r for r in valid if r['barrier'] < 1]
if subset_low:
    Ks_low = [r['K'] for r in subset_low]
    print(f"{'< 1':<20} {np.mean(Ks_low):<10.4f} {np.std(Ks_low):<10.4f} "
          f"{np.min(Ks_low):<10.4f} {np.max(Ks_low):<10.4f} {len(Ks_low):<5d}")

# ---- Question 1: Is K constant? ----
all_K = [r['K'] for r in valid]
mean_K = np.mean(all_K)
std_K = np.std(all_K)
cv_K = std_K / mean_K
print(f"\n--- Q1: Is K constant? ---")
print(f"Mean K = {mean_K:.6f}")
print(f"Std K  = {std_K:.6f}")
print(f"CV(K)  = {cv_K:.4f} ({cv_K*100:.1f}%)")
print(f"Answer: {'YES (CV < 10%)' if cv_K < 0.10 else 'NO (CV >= 10%)'}")

# ---- Question 2: Does K depend on barrier height? ----
print(f"\n--- Q2: Barrier dependence? ---")
barriers = np.array([r['barrier'] for r in valid])
K_arr = np.array([r['K'] for r in valid])

# Fit K = constant
K_const = np.mean(K_arr)
SS_const = np.sum((K_arr - K_const)**2)
SS_total = np.sum((K_arr - K_const)**2)

# Fit K = a + b / (2ΔΦ/σ²)
# Linear regression: K = a + b * (1/barrier)
X = np.column_stack([np.ones_like(barriers), 1.0 / barriers])
coeffs = np.linalg.lstsq(X, K_arr, rcond=None)[0]
K_pred = X @ coeffs
SS_res_linear = np.sum((K_arr - K_pred)**2)
R2_linear = 1 - SS_res_linear / SS_total if SS_total > 0 else 0

# Also fit K = a + b * barrier (direct linear)
X2 = np.column_stack([np.ones_like(barriers), barriers])
coeffs2 = np.linalg.lstsq(X2, K_arr, rcond=None)[0]
K_pred2 = X2 @ coeffs2
SS_res_linear2 = np.sum((K_arr - K_pred2)**2)
R2_linear2 = 1 - SS_res_linear2 / SS_total if SS_total > 0 else 0

print(f"Fit 1: K = constant = {K_const:.6f}")
print(f"Fit 2: K = {coeffs[0]:.4f} + {coeffs[1]:.4f} / (2ΔΦ/σ²)")
print(f"  R² (1/barrier model) = {R2_linear:.4f}")
print(f"Fit 3: K = {coeffs2[0]:.4f} + {coeffs2[1]:.6f} × (2ΔΦ/σ²)")
print(f"  R² (linear barrier model) = {R2_linear2:.4f}")
print(f"Answer: K {'DEPENDS on barrier (R² > 0.1)' if R2_linear > 0.1 or R2_linear2 > 0.1 else 'does NOT strongly depend on barrier (R² ≤ 0.1)'}")

# ---- Question 3: System dependence? ----
print(f"\n--- Q3: System dependence? ---")
for sys_name in ['Lake', 'Double-well', 'Schlögl']:
    subset = [r['K'] for r in valid if r['system'] == sys_name]
    if subset:
        print(f"  {sys_name:<15}: mean K = {np.mean(subset):.6f} ± {np.std(subset):.6f} (N={len(subset)})")

sys_means = {}
for sys_name in ['Lake', 'Double-well', 'Schlögl']:
    subset = [r['K'] for r in valid if r['system'] == sys_name]
    if subset:
        sys_means[sys_name] = np.mean(subset)

if len(sys_means) >= 2:
    vals = list(sys_means.values())
    max_diff = max(vals) - min(vals)
    mean_all = np.mean(vals)
    pct_diff = max_diff / mean_all * 100
    print(f"  Max difference between systems: {max_diff:.4f} ({pct_diff:.1f}%)")
    print(f"  Answer: K {'IS system-specific (>5% diff)' if pct_diff > 5 else 'is NOT system-specific (≤5% diff)'}")

# ---- Question 4: Deep barrier limit ----
print(f"\n--- Q4: Deep-barrier limit (K → 1?) ---")
deep = [r for r in valid if r['barrier'] > 10]
shallow = [r for r in valid if 2 <= r['barrier'] <= 4]
if deep:
    K_deep = [r['K'] for r in deep]
    print(f"  At 2ΔΦ/σ² > 10: mean K = {np.mean(K_deep):.6f} (N={len(K_deep)})")
else:
    print(f"  No data at 2ΔΦ/σ² > 10")
if shallow:
    K_shallow = [r['K'] for r in shallow]
    print(f"  At 2ΔΦ/σ² ∈ [2,4]: mean K = {np.mean(K_shallow):.6f} (N={len(K_shallow)})")
if deep and shallow:
    approach = np.mean(K_deep) > np.mean(K_shallow)
    print(f"  K at deep barrier {'IS' if approach else 'is NOT'} closer to 1.0 than at shallow barrier")

# ---- Question 5: Best single K ----
print(f"\n--- Q5: Best single K value ---")
print(f"  K = {mean_K:.4f} ± {std_K/np.sqrt(len(all_K)):.4f} (mean ± SE, N={len(all_K)})")
print(f"  Median K = {np.median(all_K):.4f}")

# ---- Cross-check against known values ----
print(f"\n--- Cross-checks ---")
lake_check_30 = [r for r in valid if r['system'] == 'Lake' and 'a=0.3266' in r['params'] and r.get('cv') == 0.30]
lake_check_35 = [r for r in valid if r['system'] == 'Lake' and 'a=0.3266' in r['params'] and r.get('cv') == 0.35]
if lake_check_30:
    print(f"  Lake (a=0.3266, CV=0.30): K = {lake_check_30[0]['K']:.4f} (expected: 0.544)")
if lake_check_35:
    print(f"  Lake (a=0.3266, CV=0.35): K = {lake_check_35[0]['K']:.4f} (expected: 0.563)")


# ==============================================================================
# Store results for report generation
# ==============================================================================
print("\n\n" + "=" * 80)
print("REPORT DATA (for markdown generation)")
print("=" * 80)

# Print all results in a format easy to parse
print("\n[ALL_RESULTS]")
for r in valid:
    print(f"{r['system']}|{r['params']}|{r['x_eq']:.6f}|{r['x_sad']:.6f}|"
          f"{r['DeltaPhi']:.8f}|{r['barrier']:.4f}|{r['D_exact']:.6e}|"
          f"{r['D_kramers']:.6e}|{r['K']:.6f}")

print("\nDone.")
