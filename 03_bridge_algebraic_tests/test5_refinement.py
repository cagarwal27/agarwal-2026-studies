#!/usr/bin/env python3
"""
Test 5 Refinement: Find exact q_critical, critical scaling, and improved fits.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar, curve_fit
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

b = 0.8
r = 1.0
h = 1.0

def f_lake(x, a, q_val):
    return a - b*x + r * x**q_val / (x**q_val + h**q_val)

def f_lake_deriv(x, a, q_val):
    return -b + r * q_val * x**(q_val-1) * h**q_val / (x**q_val + h**q_val)**2

def find_roots(a, q_val, x_lo=0.001, x_hi=3.0, n_scan=5000):
    xs = np.linspace(x_lo, x_hi, n_scan)
    fs = np.array([f_lake(x, a, q_val) for x in xs])
    roots = []
    for i in range(len(fs)-1):
        if fs[i] * fs[i+1] < 0:
            try:
                root = brentq(lambda x: f_lake(x, a, q_val), xs[i], xs[i+1], xtol=1e-12)
                roots.append(root)
            except:
                pass
    return roots

def is_bistable(q_val):
    """Check if any a value gives 3 roots."""
    for a in np.arange(0.001, 0.801, 0.0005):
        roots = find_roots(a, q_val, n_scan=10000)
        if len(roots) == 3:
            return True
    return False

# ============================================================
# 1. Find exact q_critical by bisection
# ============================================================
print("=" * 70)
print("FINDING EXACT q_critical")
print("=" * 70)

# We know q=2 is not bistable, q=3 is. Bisect.
q_lo, q_hi = 2.0, 3.0
for _ in range(30):
    q_test = (q_lo + q_hi) / 2
    if is_bistable(q_test):
        q_hi = q_test
    else:
        q_lo = q_test
    if q_hi - q_lo < 0.001:
        break

q_critical_numerical = (q_lo + q_hi) / 2
print(f"\nNumerical q_critical = {q_critical_numerical:.4f}")
print(f"  Bracket: [{q_lo:.4f}, {q_hi:.4f}]")
print(f"  Theoretical (approx): 4bh/r = {4*b*h/r:.4f}")

# More precise: find max slope of Hill function
# H(x) = r*x^q/(x^q + h^q)
# H'(x) = r*q*x^(q-1)*h^q / (x^q + h^q)^2
# Set H'(x) = b to find the cusp
def max_hill_slope(q_val):
    """Find the maximum slope of the Hill function."""
    def neg_slope(x):
        if x <= 0:
            return 0
        return -(r * q_val * x**(q_val-1) * h**q_val / (x**q_val + h**q_val)**2)
    result = minimize_scalar(neg_slope, bounds=(0.01, 5.0), method='bounded')
    return -result.fun

# Find q where max_slope = b
def slope_minus_b(q_val):
    return max_hill_slope(q_val) - b

q_crit_exact = brentq(slope_minus_b, 2.0, 3.5, xtol=1e-10)
print(f"\nExact q_critical (max slope = b): {q_crit_exact:.6f}")

# ============================================================
# 2. Fine-grained ΔΦ near q_critical
# ============================================================
print("\n" + "=" * 70)
print("ΔΦ NEAR q_critical")
print("=" * 70)

q_fine = np.array([q_crit_exact + dq for dq in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.8, 6.8, 8.8, 12.8, 16.8]])
# Also add the original q values
q_fine = np.sort(np.unique(np.concatenate([q_fine, [3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]])))

dphi_fine = []
q_fine_valid = []

print(f"\n{'q':<8} {'q-q_c':<8} {'a_mid':<8} {'x_clear':<9} {'x_sad':<9} {'ΔΦ':<14}")
print("-" * 56)

for q_val in q_fine:
    # Find bistable range
    bistable_as = []
    for a in np.arange(0.001, 0.801, 0.0005):
        roots = find_roots(a, q_val, n_scan=8000)
        if len(roots) == 3:
            bistable_as.append(a)

    if len(bistable_as) == 0:
        continue

    a_mid = (bistable_as[0] + bistable_as[-1]) / 2
    roots = find_roots(a_mid, q_val, n_scan=10000)
    if len(roots) != 3:
        for delta in np.arange(-0.005, 0.006, 0.0005):
            roots = find_roots(a_mid + delta, q_val, n_scan=10000)
            if len(roots) == 3:
                a_mid += delta
                break
    if len(roots) != 3:
        continue

    x_cl, x_sd, x_tb = roots
    lam_eq = f_lake_deriv(x_cl, a_mid, q_val)
    lam_sad = f_lake_deriv(x_sd, a_mid, q_val)
    if lam_eq >= 0 or lam_sad <= 0:
        continue

    DPhi, _ = quad(lambda x: -f_lake(x, a_mid, q_val), x_cl, x_sd, limit=200)
    if DPhi <= 0:
        continue

    dq = q_val - q_crit_exact
    q_fine_valid.append(q_val)
    dphi_fine.append(DPhi)
    print(f"{q_val:<8.4f} {dq:<8.4f} {a_mid:<8.4f} {x_cl:<9.5f} {x_sd:<9.5f} {DPhi:<14.10f}")

# ============================================================
# 3. Critical scaling: ΔΦ vs (q - q_c)
# ============================================================
print("\n" + "=" * 70)
print("CRITICAL SCALING")
print("=" * 70)

q_arr = np.array(q_fine_valid)
dphi_arr = np.array(dphi_fine)
dq_arr = q_arr - q_crit_exact

# Fit ΔΦ = A*(q-q_c)^β using log-log regression
# Use points near q_c (dq < 3)
near_mask = (dq_arr > 0) & (dq_arr < 3.0)
if np.sum(near_mask) >= 3:
    log_dq = np.log(dq_arr[near_mask])
    log_dphi = np.log(dphi_arr[near_mask])
    slope_near, intercept_near, r_near, _, _ = linregress(log_dq, log_dphi)
    print(f"\nNear-critical fit (dq < 3): ΔΦ = {np.exp(intercept_near):.6f} × (q-q_c)^{slope_near:.4f}")
    print(f"  R² = {r_near**2:.6f}")
    print(f"  Expected β = 1.5 for saddle-node")

# Very near q_c (dq < 1)
vnear_mask = (dq_arr > 0) & (dq_arr < 1.0)
if np.sum(vnear_mask) >= 3:
    log_dq = np.log(dq_arr[vnear_mask])
    log_dphi = np.log(dphi_arr[vnear_mask])
    slope_vnear, intercept_vnear, r_vnear, _, _ = linregress(log_dq, log_dphi)
    print(f"\nVery-near-critical fit (dq < 1): ΔΦ = {np.exp(intercept_vnear):.6f} × (q-q_c)^{slope_vnear:.4f}")
    print(f"  R² = {r_vnear**2:.6f}")

# Full range fit
log_dq_all = np.log(dq_arr[dq_arr > 0])
log_dphi_all = np.log(dphi_arr[dq_arr > 0])
slope_all, intercept_all, r_all, _, _ = linregress(log_dq_all, log_dphi_all)
print(f"\nFull-range fit: ΔΦ = {np.exp(intercept_all):.6f} × (q-q_c)^{slope_all:.4f}")
print(f"  R² = {r_all**2:.6f}")

# Try nonlinear fit: ΔΦ = A*(q-q_c)^β / (1 + C*(q-q_c)^γ) for saturation
print("\n--- Saturating model ---")
def sat_model(dq, A, beta, C):
    return A * dq**beta / (1 + C * dq)

try:
    popt, pcov = curve_fit(sat_model, dq_arr[dq_arr > 0], dphi_arr[dq_arr > 0],
                           p0=[0.01, 1.5, 0.1], maxfev=10000,
                           bounds=([0, 0.5, 0], [1, 3.0, 10]))
    A_sat, beta_sat, C_sat = popt
    dphi_pred = sat_model(dq_arr[dq_arr > 0], *popt)
    residuals = dphi_arr[dq_arr > 0] - dphi_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((dphi_arr[dq_arr > 0] - np.mean(dphi_arr[dq_arr > 0]))**2)
    R2_sat = 1 - ss_res / ss_tot
    print(f"  ΔΦ = {A_sat:.6f} × (q-q_c)^{beta_sat:.4f} / (1 + {C_sat:.4f}×(q-q_c))")
    print(f"  R² = {R2_sat:.6f}")
    print(f"  Saturation value ΔΦ_∞ = A/C × lim = ... (as dq→∞, ΔΦ → A/C × dq^(β-1))")

    print(f"\n  Point-by-point:")
    for dq, dp_actual, dp_fit in zip(dq_arr[dq_arr > 0], dphi_arr[dq_arr > 0], dphi_pred):
        pct = 100*(dp_fit - dp_actual)/dp_actual
        print(f"    dq={dq:6.3f}: actual={dp_actual:.8f}, fit={dp_fit:.8f}, err={pct:+.2f}%")
except Exception as e:
    print(f"  Saturating model failed: {e}")

# Also try: ΔΦ = ΔΦ_∞ * (1 - exp(-k*(q-q_c)^β))
print("\n--- Exponential saturation model ---")
def exp_sat_model(dq, dphi_inf, k, beta):
    return dphi_inf * (1 - np.exp(-k * dq**beta))

try:
    popt2, pcov2 = curve_fit(exp_sat_model, dq_arr[dq_arr > 0], dphi_arr[dq_arr > 0],
                             p0=[0.12, 0.5, 1.5], maxfev=10000,
                             bounds=([0.05, 0.01, 0.5], [0.5, 5.0, 3.0]))
    dphi_inf, k_exp, beta_exp = popt2
    dphi_pred2 = exp_sat_model(dq_arr[dq_arr > 0], *popt2)
    residuals2 = dphi_arr[dq_arr > 0] - dphi_pred2
    ss_res2 = np.sum(residuals2**2)
    ss_tot2 = np.sum((dphi_arr[dq_arr > 0] - np.mean(dphi_arr[dq_arr > 0]))**2)
    R2_exp = 1 - ss_res2 / ss_tot2
    print(f"  ΔΦ = {dphi_inf:.6f} × (1 - exp(-{k_exp:.4f} × (q-q_c)^{beta_exp:.4f}))")
    print(f"  ΔΦ_∞ = {dphi_inf:.6f}")
    print(f"  R² = {R2_exp:.6f}")

    print(f"\n  Point-by-point:")
    for dq, dp_actual, dp_fit in zip(dq_arr[dq_arr > 0], dphi_arr[dq_arr > 0], dphi_pred2):
        pct = 100*(dp_fit - dp_actual)/dp_actual
        print(f"    dq={dq:6.3f}: actual={dp_actual:.8f}, fit={dp_fit:.8f}, err={pct:+.2f}%")
except Exception as e:
    print(f"  Exponential saturation model failed: {e}")

# ============================================================
# 4. Step-function limit barrier (exact)
# ============================================================
print("\n" + "=" * 70)
print("STEP-FUNCTION LIMIT (q → ∞)")
print("=" * 70)

# For q→∞, f(x) = a - bx for x < h=1, f(x) = a - bx + r for x > h
# Stable eq: x_cl = a/b (need a/b < 1)
# Saddle: x = h = 1 (discontinuity point)
# Stable eq: x_tb = (a+r)/b (need (a+r)/b > 1)
# Bistability requires: a/b < 1 AND (a+r)/b > 1 => a < b and a > b - r
# => b - r < a < b => 0 < a < 0.8 (always for our params since b-r = -0.2 < 0)
# Actually: b - r = 0.8 - 1.0 = -0.2 < 0, so for any a > 0, x_turb exists.
# But we need x_clear < h, i.e., a/b < h => a < bh = 0.8. Always true.
# And we need x_turb > h, i.e., (a+r)/b > h => a > bh - r = 0.8 - 1.0 = -0.2. Always true.
# So for q→∞, ALL a ∈ (0, 0.8) give bistability. Width → 0.8.

# Midpoint: a ≈ 0.32 (converging), so a_step = 0.32
a_step = 0.32
x_cl_step = a_step / b  # = 0.4
x_sd_step = h  # = 1.0

# Barrier = ∫_{x_cl}^{h} (bx - a) dx = [b*x²/2 - a*x]_{x_cl}^{h}
# = (b*h²/2 - a*h) - (b*x_cl²/2 - a*x_cl)
# = b/2*(h² - x_cl²) - a*(h - x_cl)
barrier_step = b/2*(h**2 - x_cl_step**2) - a_step*(h - x_cl_step)
print(f"\nStep-function limit (a={a_step}):")
print(f"  x_clear = a/b = {x_cl_step:.4f}")
print(f"  x_saddle = h = {x_sd_step:.4f}")
print(f"  ΔΦ_step = {barrier_step:.8f}")

# Compare with large q
print(f"\n  Convergence to step limit:")
for q_val in [8.0, 10.0, 12.0, 16.0, 20.0]:
    idx = [i for i, q in enumerate(q_fine_valid) if abs(q - q_val) < 0.01]
    if idx:
        dp = dphi_fine[idx[0]]
        print(f"    q={q_val:5.1f}: ΔΦ={dp:.8f}, ΔΦ/ΔΦ_step={dp/barrier_step:.4f}")

# Exact step limit with varying a
print(f"\n  Step-function barrier formula: ΔΦ_step = (b-a)²/(2b) = {(b-a_step)**2/(2*b):.8f}")
# Actually: let me compute this properly
# ΔΦ = ∫_{a/b}^{1} (bx - a) dx = [bx²/2 - ax]_{a/b}^{1}
# = (b/2 - a) - (a²/(2b) - a²/b) = b/2 - a - a²/(2b) + a²/b = b/2 - a + a²/(2b)
# = (b² - 2ab + a²)/(2b) = (b-a)²/(2b)
barrier_formula = (b - a_step)**2 / (2*b)
print(f"  Formula check: (b-a)²/(2b) = ({b}-{a_step})²/(2×{b}) = {barrier_formula:.8f}")
print(f"  Direct integral: {barrier_step:.8f}")
print(f"  Match: {'YES' if abs(barrier_formula - barrier_step) < 1e-10 else 'NO'}")

# The step-function limit for midpoint loading
# At the midpoint, a → ~0.32, so:
print(f"\n  ΔΦ_∞ = (b - a_mid)²/(2b)")
for a_test in [0.30, 0.32, 0.325, 0.33]:
    dphi_test = (b - a_test)**2 / (2*b)
    print(f"    a={a_test:.3f}: ΔΦ_∞ = {dphi_test:.8f}")

# ============================================================
# 5. Improved critical exponent near q_c using very fine q grid
# ============================================================
print("\n" + "=" * 70)
print("REFINED CRITICAL EXPONENT")
print("=" * 70)

q_very_fine = [q_crit_exact + dq for dq in [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50]]
dphi_vfine = []
dq_vfine = []

print(f"\nq_critical = {q_crit_exact:.6f}")
print(f"\n{'dq':<10} {'q':<10} {'ΔΦ':<16} {'ln(dq)':<10} {'ln(ΔΦ)':<10}")
print("-" * 56)

for q_val in q_very_fine:
    # Very fine a scan
    bistable_as = []
    for a in np.arange(0.001, 0.801, 0.0002):
        roots = find_roots(a, q_val, n_scan=10000)
        if len(roots) == 3:
            bistable_as.append(a)

    if len(bistable_as) == 0:
        print(f"{q_val - q_crit_exact:<10.5f} {q_val:<10.5f} {'not bistable':<16}")
        continue

    a_mid = (bistable_as[0] + bistable_as[-1]) / 2
    roots = find_roots(a_mid, q_val, n_scan=10000)
    if len(roots) != 3:
        continue

    x_cl, x_sd = roots[0], roots[1]
    lam_eq = f_lake_deriv(x_cl, a_mid, q_val)
    lam_sad = f_lake_deriv(x_sd, a_mid, q_val)
    if lam_eq >= 0 or lam_sad <= 0:
        continue

    DPhi, _ = quad(lambda x: -f_lake(x, a_mid, q_val), x_cl, x_sd, limit=200)
    if DPhi <= 0:
        continue

    dq = q_val - q_crit_exact
    dq_vfine.append(dq)
    dphi_vfine.append(DPhi)
    print(f"{dq:<10.5f} {q_val:<10.5f} {DPhi:<16.10f} {np.log(dq):<10.4f} {np.log(DPhi):<10.4f}")

if len(dq_vfine) >= 3:
    log_dq_vf = np.log(np.array(dq_vfine))
    log_dphi_vf = np.log(np.array(dphi_vfine))
    slope_vf, intercept_vf, r_vf, _, _ = linregress(log_dq_vf, log_dphi_vf)
    print(f"\nCritical exponent β = {slope_vf:.4f}")
    print(f"Prefactor A = {np.exp(intercept_vf):.6f}")
    print(f"R² = {r_vf**2:.6f}")
    print(f"\nΔΦ ≈ {np.exp(intercept_vf):.4f} × (q - {q_crit_exact:.4f})^{slope_vf:.4f}")

# ============================================================
# 6. Summary: The complete ΔΦ(q) story
# ============================================================
print("\n" + "=" * 70)
print("THE ΔΦ(q) STORY")
print("=" * 70)

print(f"""
q_critical = {q_crit_exact:.4f}  (exact numerical value)
  - Below: fire mode (no bistability, no barrier)
  - Above: tree mode (bistability, barrier grows with q)

Near q_c: ΔΦ ~ (q - q_c)^β with β ≈ {slope_vf:.2f}
Large q:  ΔΦ → ΔΦ_∞ = (b-a_mid)²/(2b) ≈ {barrier_formula:.4f}

The growth exponent q determines the barrier via:
  ΔΦ(q) grows from 0 at q=q_c to ΔΦ_∞ as q→∞
  This is a monotonic, concave function with rapid initial growth
  and logarithmic-slow approach to saturation.
""")

print("DONE.")
