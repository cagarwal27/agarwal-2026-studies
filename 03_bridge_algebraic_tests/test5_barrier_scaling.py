#!/usr/bin/env python3
"""
Test 5: Does the Growth Exponent Shape the Barrier?
Computes ΔΦ(q), eigenvalue scaling, duality check, and blow-up time connection.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Model parameters (van Nes & Scheffer 2007)
# ============================================================
b = 0.8    # P loss rate
r = 1.0    # max recycling rate
h = 1.0    # half-saturation

Q_VALUES = [2.0, 3.0, 3.2, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]

def f_lake(x, a, q_val):
    return a - b*x + r * x**q_val / (x**q_val + h**q_val)

def f_lake_deriv(x, a, q_val):
    return -b + r * q_val * x**(q_val-1) * h**q_val / (x**q_val + h**q_val)**2

# ============================================================
# STEP 1: Find bistable range for each q
# ============================================================
print("=" * 80)
print("STEP 1: BISTABLE RANGE FOR EACH q")
print("=" * 80)

# Theoretical q_critical
q_crit_theory = 4*b*h/r
print(f"\nTheoretical q_critical = 4bh/r = {q_crit_theory:.4f}")

def find_roots(a, q_val, x_lo=0.001, x_hi=3.0, n_scan=5000):
    """Find all positive roots of f(x,a,q)=0 in [x_lo, x_hi]."""
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

bistable_data = {}

print(f"\n{'q':<6} {'Bistable?':<10} {'a_low':<8} {'a_high':<8} {'a_mid':<8} {'Width':<10}")
print("-" * 52)

for q_val in Q_VALUES:
    a_values = np.arange(0.001, 0.801, 0.001)
    bistable_as = []
    for a in a_values:
        roots = find_roots(a, q_val)
        if len(roots) == 3:
            bistable_as.append(a)

    if len(bistable_as) == 0:
        print(f"{q_val:<6.1f} {'No':<10} {'--':<8} {'--':<8} {'--':<8} {'--':<10}")
        bistable_data[q_val] = None
    else:
        a_low = bistable_as[0]
        a_high = bistable_as[-1]
        a_mid = (a_low + a_high) / 2
        width = a_high - a_low
        print(f"{q_val:<6.1f} {'Yes':<10} {a_low:<8.4f} {a_high:<8.4f} {a_mid:<8.4f} {width:<10.4f}")
        bistable_data[q_val] = {'a_low': a_low, 'a_high': a_high, 'a_mid': a_mid, 'width': width}

# For q near q_critical, do a finer scan
print("\n--- Fine scan near q_critical ---")
for q_val in [3.1, 3.15, 3.2, 3.25, 3.3]:
    a_values = np.arange(0.001, 0.801, 0.0005)
    bistable_as = []
    for a in a_values:
        roots = find_roots(a, q_val)
        if len(roots) == 3:
            bistable_as.append(a)
    if len(bistable_as) > 0:
        a_low = bistable_as[0]
        a_high = bistable_as[-1]
        width = a_high - a_low
        print(f"  q={q_val:.2f}: a_low={a_low:.4f}, a_high={a_high:.4f}, width={width:.4f}")
    else:
        print(f"  q={q_val:.2f}: no bistability")

# ============================================================
# STEP 2: Compute ΔΦ and eigenvalues at midpoint
# ============================================================
print("\n" + "=" * 80)
print("STEP 2: BARRIER AND EIGENVALUES AT MIDPOINT")
print("=" * 80)

barrier_results = {}

print(f"\n{'q':<5} {'x_clear':<9} {'x_sad':<9} {'x_turb':<9} {'λ_eq':<10} {'λ_sad':<10} {'ΔΦ':<12} {'τ':<8} {'C_corr':<10}")
print("-" * 82)

for q_val in Q_VALUES:
    if bistable_data[q_val] is None:
        continue

    a_mid = bistable_data[q_val]['a_mid']
    roots = find_roots(a_mid, q_val)

    if len(roots) != 3:
        # Try adjusting a_mid slightly
        for delta in np.arange(-0.01, 0.011, 0.001):
            roots = find_roots(a_mid + delta, q_val)
            if len(roots) == 3:
                a_mid = a_mid + delta
                bistable_data[q_val]['a_mid'] = a_mid
                break

    if len(roots) != 3:
        print(f"{q_val:<5.1f}  Could not find 3 roots at a_mid={a_mid:.4f}")
        continue

    x_clear, x_sad, x_turb = roots[0], roots[1], roots[2]

    lam_eq = f_lake_deriv(x_clear, a_mid, q_val)
    lam_sad = f_lake_deriv(x_sad, a_mid, q_val)

    # Verify stability
    assert lam_eq < 0, f"q={q_val}: λ_eq should be negative, got {lam_eq}"
    assert lam_sad > 0, f"q={q_val}: λ_sad should be positive, got {lam_sad}"

    # Compute barrier
    DeltaPhi, _ = quad(lambda x: -f_lake(x, a_mid, q_val), x_clear, x_sad, limit=200)

    tau = 1.0 / abs(lam_eq)
    C_corr = np.sqrt(abs(lam_eq) * abs(lam_sad)) / np.pi

    barrier_results[q_val] = {
        'a_mid': a_mid,
        'x_clear': x_clear, 'x_sad': x_sad, 'x_turb': x_turb,
        'lam_eq': lam_eq, 'lam_sad': lam_sad,
        'DeltaPhi': DeltaPhi, 'tau': tau, 'C_corr': C_corr
    }

    print(f"{q_val:<5.1f} {x_clear:<9.5f} {x_sad:<9.5f} {x_turb:<9.5f} {lam_eq:<10.5f} {lam_sad:<10.5f} {DeltaPhi:<12.8f} {tau:<8.4f} {C_corr:<10.6f}")

    # Verify f=0 at roots
    for xr in roots:
        assert abs(f_lake(xr, a_mid, q_val)) < 1e-8, f"Root verification failed at x={xr}"

# ============================================================
# STEP 3: Barrier scaling law
# ============================================================
print("\n" + "=" * 80)
print("STEP 3: ΔΦ SCALING LAW")
print("=" * 80)

q_vals_bistable = sorted([q for q in barrier_results.keys()])
dphi_vals = [barrier_results[q]['DeltaPhi'] for q in q_vals_bistable]

print("\n--- ΔΦ vs q ---")
for q_val, dp in zip(q_vals_bistable, dphi_vals):
    print(f"  q = {q_val:5.1f}  ΔΦ = {dp:.8f}")

# 3a: Power law fit: ΔΦ ∝ q^α for all q > q_critical
log_q = np.log(np.array(q_vals_bistable))
log_dphi = np.log(np.array(dphi_vals))
slope_all, intercept_all, r_all, _, _ = linregress(log_q, log_dphi)
print(f"\nPower law fit (all bistable q): ΔΦ ∝ q^{slope_all:.4f}")
print(f"  Prefactor = exp({intercept_all:.4f}) = {np.exp(intercept_all):.6f}")
print(f"  R² = {r_all**2:.6f}")

# 3b: Critical scaling near q_critical: ΔΦ ∝ (q - q_c)^β
# Use q values close to q_critical
q_near_crit = [q for q in q_vals_bistable if q < 6.0]
if len(q_near_crit) >= 3:
    dphi_near = [barrier_results[q]['DeltaPhi'] for q in q_near_crit]
    dq = np.array([q - q_crit_theory for q in q_near_crit])

    # Only use points where dq > 0
    mask = dq > 0
    if np.sum(mask) >= 2:
        log_dq = np.log(dq[mask])
        log_dphi_near = np.log(np.array(dphi_near)[mask])
        slope_crit, intercept_crit, r_crit, _, _ = linregress(log_dq, log_dphi_near)
        print(f"\nCritical scaling (q near q_c): ΔΦ ∝ (q - {q_crit_theory:.1f})^{slope_crit:.4f}")
        print(f"  Expected β = 1.5 (saddle-node normal form)")
        print(f"  R² = {r_crit**2:.6f}")

        print(f"\n  Points used:")
        for q_val, dp, dq_val in zip(np.array(q_near_crit)[mask], np.array(dphi_near)[mask], dq[mask]):
            print(f"    q={q_val:.1f}, q-q_c={dq_val:.2f}, ΔΦ={dp:.8f}")

# 3c: Large-q behavior
print(f"\n--- Large-q behavior ---")
large_q = [q for q in q_vals_bistable if q >= 8.0]
for q_val in large_q:
    dp = barrier_results[q_val]['DeltaPhi']
    print(f"  q = {q_val:5.1f}  ΔΦ = {dp:.8f}")
if len(large_q) >= 2:
    dp_last = barrier_results[large_q[-1]]['DeltaPhi']
    dp_prev = barrier_results[large_q[-2]]['DeltaPhi']
    print(f"  Ratio ΔΦ(q={large_q[-1]})/ΔΦ(q={large_q[-2]}) = {dp_last/dp_prev:.6f}")
    print(f"  Appears to {'saturate' if dp_last/dp_prev < 1.1 else 'still grow'}")

# 3d: Eigenvalue scaling
print(f"\n--- Eigenvalue scaling ---")
print(f"{'q':<6} {'|λ_eq|':<10} {'|λ_sad|':<10} {'|λ_eq|/b':<10} {'|λ_sad|/b':<10}")
print("-" * 46)
for q_val in q_vals_bistable:
    le = abs(barrier_results[q_val]['lam_eq'])
    ls = abs(barrier_results[q_val]['lam_sad'])
    print(f"{q_val:<6.1f} {le:<10.5f} {ls:<10.5f} {le/b:<10.5f} {ls/b:<10.5f}")

# ============================================================
# STEP 4: Duality across q values — find σ_eff where D=200
# ============================================================
print("\n" + "=" * 80)
print("STEP 4: DUALITY ACROSS q — FIND σ_eff FOR D=200")
print("=" * 80)

def compute_D_exact(f_func, x_eq, x_sad_pt, sigma, tau, x_lo=0.001):
    """Exact MFPT-based D."""
    N_grid = 50000
    x_hi = x_sad_pt + 0.001
    if x_lo >= x_hi:
        return 0.0
    x_grid = np.linspace(x_lo, x_hi, N_grid)
    dx_g = x_grid[1] - x_grid[0]

    neg_f = np.array([-f_func(x) for x in x_grid])
    U_raw = np.cumsum(neg_f) * dx_g
    i_eq = np.argmin(np.abs(x_grid - x_eq))
    U_grid = U_raw - U_raw[i_eq]
    Phi = 2.0 * U_grid / sigma**2

    # Clip to prevent overflow
    Phi = np.clip(Phi, -500, 500)

    exp_neg_Phi = np.exp(-Phi)
    I_x = np.cumsum(exp_neg_Phi) * dx_g

    psi = (2.0 / sigma**2) * np.exp(Phi) * I_x

    i_sad = np.argmin(np.abs(x_grid - x_sad_pt))
    if i_eq >= i_sad:
        return 0.0
    MFPT = np.trapz(psi[i_eq:i_sad + 1], x_grid[i_eq:i_sad + 1])
    return MFPT / tau, MFPT

duality_results = {}

print(f"\n{'q':<5} {'ΔΦ':<12} {'σ_eff':<10} {'CV_eff':<10} {'2ΔΦ/σ²':<10} {'K_corr':<10} {'D_exact':<10}")
print("-" * 67)

for q_val in q_vals_bistable:
    res = barrier_results[q_val]
    a_mid = res['a_mid']
    x_cl = res['x_clear']
    x_sd = res['x_sad']
    lam_e = res['lam_eq']
    tau_val = res['tau']
    DPhi = res['DeltaPhi']
    C_co = res['C_corr']

    f_func = lambda x, _a=a_mid, _q=q_val: f_lake(x, _a, _q)

    def D_at_cv(cv):
        sigma = cv * x_cl * np.sqrt(2.0 * abs(lam_e))
        result = compute_D_exact(f_func, x_cl, x_sd, sigma, tau_val)
        if isinstance(result, tuple):
            return result[0] - 200.0
        return result - 200.0

    # Find CV range where D crosses 200
    cv_lo, cv_hi = 0.05, 0.90
    try:
        # First check endpoints
        D_lo = D_at_cv(cv_lo)
        D_hi = D_at_cv(cv_hi)

        if D_lo * D_hi > 0:
            # Search for sign change
            found = False
            for cv_test in np.arange(0.05, 0.91, 0.02):
                D_test = D_at_cv(cv_test)
                if D_lo * D_test < 0:
                    cv_hi = cv_test
                    found = True
                    break
                if D_test < D_lo:
                    cv_lo = cv_test
                    D_lo = D_test
            if not found:
                print(f"{q_val:<5.1f}  Could not bracket D=200 (D range: {D_lo+200:.1f} to {D_hi+200:.1f})")
                continue

        cv_star = brentq(D_at_cv, cv_lo, cv_hi, xtol=1e-6, maxiter=100)
        sigma_star = cv_star * x_cl * np.sqrt(2.0 * abs(lam_e))
        result = compute_D_exact(f_func, x_cl, x_sd, sigma_star, tau_val)
        D_star = result[0] if isinstance(result, tuple) else result
        MFPT_star = result[1] if isinstance(result, tuple) else D_star * tau_val

        barrier_ratio = 2.0 * DPhi / sigma_star**2

        # Kramers prediction
        D_kramers = np.exp(barrier_ratio) / (C_co * tau_val)
        K_corr = D_star / D_kramers if D_kramers > 0 else float('inf')

        duality_results[q_val] = {
            'sigma_eff': sigma_star, 'CV_eff': cv_star,
            'barrier_ratio': barrier_ratio, 'K_corr': K_corr,
            'D_exact': D_star, 'MFPT': MFPT_star
        }

        print(f"{q_val:<5.1f} {DPhi:<12.8f} {sigma_star:<10.6f} {cv_star:<10.6f} {barrier_ratio:<10.4f} {K_corr:<10.4f} {D_star:<10.1f}")
    except Exception as e:
        print(f"{q_val:<5.1f}  Error: {e}")

# CV_eff summary
print(f"\n--- CV_eff summary ---")
cv_vals = [duality_results[q]['CV_eff'] for q in duality_results]
if cv_vals:
    print(f"  Mean CV_eff = {np.mean(cv_vals):.4f}")
    print(f"  Std CV_eff  = {np.std(cv_vals):.4f}")
    print(f"  Range: [{min(cv_vals):.4f}, {max(cv_vals):.4f}]")

# ============================================================
# STEP 5: Blow-up time connection
# ============================================================
print("\n" + "=" * 80)
print("STEP 5: BLOW-UP TIME CONNECTION")
print("=" * 80)

print(f"\n{'q':<5} {'T_unreg':<12} {'MFPT':<12} {'D_transit':<12} {'D_exact':<12} {'Ratio':<10}")
print("-" * 63)

blowup_results = {}

for q_val in q_vals_bistable:
    if q_val not in duality_results:
        continue
    res = barrier_results[q_val]
    dual = duality_results[q_val]
    a_mid = res['a_mid']
    x_cl = res['x_clear']
    x_sd = res['x_sad']

    # Unregulated transit time: dx/dt = a + r*x^q/(x^q + h^q) (no -bx)
    def unreg_integrand(x, _a=a_mid, _q=q_val):
        rate = _a + r * x**_q / (x**_q + h**_q)
        return 1.0 / rate

    try:
        T_unreg, _ = quad(unreg_integrand, x_cl, x_sd, limit=200)
        MFPT = dual['MFPT']
        D_transit = MFPT / T_unreg
        D_exact = dual['D_exact']
        ratio = D_transit / D_exact

        blowup_results[q_val] = {
            'T_unreg': T_unreg, 'MFPT': MFPT,
            'D_transit': D_transit, 'ratio': ratio
        }

        print(f"{q_val:<5.1f} {T_unreg:<12.4f} {MFPT:<12.4f} {D_transit:<12.4f} {D_exact:<12.1f} {ratio:<10.4f}")
    except Exception as e:
        print(f"{q_val:<5.1f}  Error: {e}")

# ============================================================
# STEP 6: Additional analysis
# ============================================================
print("\n" + "=" * 80)
print("STEP 6: ADDITIONAL ANALYSIS")
print("=" * 80)

# 6a: ΔΦ fit with (q - q_c) form
print("\n--- Fitting ΔΦ = A * (q - q_c)^β ---")
# Use all points
from scipy.optimize import curve_fit

def barrier_model_crit(q_arr, A, beta, q_c):
    return A * (q_arr - q_c)**beta

q_arr = np.array(q_vals_bistable)
dphi_arr = np.array(dphi_vals)

try:
    popt, pcov = curve_fit(barrier_model_crit, q_arr, dphi_arr, p0=[0.01, 1.5, 3.2],
                           bounds=([0, 0.5, 2.0], [10, 5.0, 3.5]))
    A_fit, beta_fit, qc_fit = popt
    dphi_pred = barrier_model_crit(q_arr, *popt)
    residuals = dphi_arr - dphi_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((dphi_arr - np.mean(dphi_arr))**2)
    R2 = 1 - ss_res / ss_tot

    print(f"  A = {A_fit:.6f}")
    print(f"  β = {beta_fit:.4f}")
    print(f"  q_c = {qc_fit:.4f}")
    print(f"  R² = {R2:.6f}")
    print(f"\n  Fit quality:")
    for q_val, dp_actual, dp_fit in zip(q_arr, dphi_arr, dphi_pred):
        pct_err = 100 * (dp_fit - dp_actual) / dp_actual if dp_actual != 0 else 0
        print(f"    q={q_val:5.1f}: actual={dp_actual:.8f}, fit={dp_fit:.8f}, err={pct_err:+.1f}%")
except Exception as e:
    print(f"  Curve fit failed: {e}")

# 6b: Try power law ΔΦ = A * q^α for large q only
print("\n--- Power law for large q (≥ 6) ---")
large_q_idx = [i for i, q in enumerate(q_vals_bistable) if q >= 6.0]
if len(large_q_idx) >= 3:
    log_q_large = np.log(np.array([q_vals_bistable[i] for i in large_q_idx]))
    log_dphi_large = np.log(np.array([dphi_vals[i] for i in large_q_idx]))
    slope_large, intercept_large, r_large, _, _ = linregress(log_q_large, log_dphi_large)
    print(f"  ΔΦ ∝ q^{slope_large:.4f} for q ≥ 6")
    print(f"  Prefactor = {np.exp(intercept_large):.6f}")
    print(f"  R² = {r_large**2:.6f}")

# 6c: Saturation analysis - does ΔΦ/ΔΦ_step approach 1?
# Step function limit: when q→∞, recycling = r for x>h, 0 for x<h
# In this limit the dynamics have a known barrier
print("\n--- Step-function (q→∞) limit ---")
# For q→∞: f(x,a) = a - bx for x < h, and a - bx + r for x > h
# Equilibria: x_clear = a/b (if a/b < h), x_turb = (a+r)/b (if (a+r)/b > h)
# Saddle at x = h (the switching point)
# The "a" in the step limit should maintain the same bistability structure
# For the midpoint a of the bistable range at large q:
if 20.0 in barrier_results:
    a20 = barrier_results[20.0]['a_mid']
    # Approximate barrier in step limit
    x_cl_step = a20 / b
    x_sd_step = h  # saddle at h
    # Barrier = integral of -f from x_clear to h
    # f(x) = a - bx for x < h
    # -f(x) = bx - a
    # integral = b*x²/2 - a*x evaluated from x_cl_step to h
    barrier_step = (b * h**2 / 2 - a20 * h) - (b * x_cl_step**2 / 2 - a20 * x_cl_step)
    print(f"  Step-limit barrier estimate (q→∞): {barrier_step:.8f}")
    print(f"  ΔΦ at q=20: {barrier_results[20.0]['DeltaPhi']:.8f}")
    print(f"  Ratio: {barrier_results[20.0]['DeltaPhi'] / barrier_step:.4f}")

# 6d: Bistable width scaling
print("\n--- Bistable width (a_high - a_low) scaling ---")
for q_val in Q_VALUES:
    if bistable_data[q_val] is not None:
        w = bistable_data[q_val]['width']
        print(f"  q={q_val:5.1f}: width = {w:.4f}")

# 6e: What's special about q=8?
print("\n--- What's special about q=8? ---")
if 8.0 in barrier_results and 8.0 in duality_results:
    r8 = barrier_results[8.0]
    d8 = duality_results[8.0]
    print(f"  ΔΦ = {r8['DeltaPhi']:.8f}")
    print(f"  CV_eff = {d8['CV_eff']:.4f}")
    print(f"  x_clear/x_sad = {r8['x_clear']/r8['x_sad']:.4f}")
    print(f"  |λ_eq|/|λ_sad| = {abs(r8['lam_eq'])/abs(r8['lam_sad']):.4f}")
    print(f"  Bistable width = {bistable_data[8.0]['width']:.4f}")
    # Compare to other q
    for q_val in [4.0, 6.0, 10.0, 12.0]:
        if q_val in barrier_results:
            ratio_dphi = barrier_results[q_val]['DeltaPhi'] / r8['DeltaPhi']
            print(f"  ΔΦ(q={q_val})/ΔΦ(q=8) = {ratio_dphi:.4f}")

# 6f: Three-term decomposition check
print("\n--- Three-term decomposition ---")
print("Checking if D_gradient = x_clear^(2-q)/(q-1) appears as a factor")
for q_val in q_vals_bistable:
    if q_val in duality_results:
        x_cl = barrier_results[q_val]['x_clear']
        # Gradient factor
        if q_val != 1:
            D_gradient = x_cl**(2 - q_val) / (q_val - 1)
        else:
            D_gradient = float('inf')
        D_ex = duality_results[q_val]['D_exact']
        # If D = D_channels × D_barrier × D_gradient,
        # and D_channels = D_Kramers (from duality), then:
        # D_gradient should be ≈ 1 if absorbed, or some systematic factor if not
        K = duality_results[q_val]['K_corr']
        print(f"  q={q_val:5.1f}: D_gradient={D_gradient:.4e}, K_corr={K:.4f}, product={D_gradient*K:.4e}")

# ============================================================
# COMPREHENSIVE OUTPUT for results file
# ============================================================
print("\n" + "=" * 80)
print("COMPREHENSIVE RESULTS SUMMARY")
print("=" * 80)

print("\n## Table 1: Bistable ranges")
print(f"| {'q':>5} | {'Bistable?':>9} | {'a_low':>7} | {'a_high':>7} | {'a_mid':>7} | {'Width':>8} |")
print(f"|{'-'*7}|{'-'*11}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*10}|")
for q_val in Q_VALUES:
    if bistable_data[q_val] is None:
        print(f"| {q_val:5.1f} | {'No':>9} | {'--':>7} | {'--':>7} | {'--':>7} | {'--':>8} |")
    else:
        d = bistable_data[q_val]
        print(f"| {q_val:5.1f} | {'Yes':>9} | {d['a_low']:7.4f} | {d['a_high']:7.4f} | {d['a_mid']:7.4f} | {d['width']:8.4f} |")

print("\n## Table 2: Barrier and eigenvalues")
print(f"| {'q':>5} | {'x_clear':>8} | {'x_sad':>8} | {'x_turb':>8} | {'λ_eq':>9} | {'λ_sad':>9} | {'ΔΦ':>12} | {'τ':>6} | {'C_corr':>8} |")
print(f"|{'-'*7}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*11}|{'-'*11}|{'-'*14}|{'-'*8}|{'-'*10}|")
for q_val in q_vals_bistable:
    r_q = barrier_results[q_val]
    print(f"| {q_val:5.1f} | {r_q['x_clear']:8.5f} | {r_q['x_sad']:8.5f} | {r_q['x_turb']:8.5f} | {r_q['lam_eq']:9.5f} | {r_q['lam_sad']:9.5f} | {r_q['DeltaPhi']:12.8f} | {r_q['tau']:6.4f} | {r_q['C_corr']:8.6f} |")

print("\n## Table 3: Duality (D=200)")
print(f"| {'q':>5} | {'ΔΦ':>12} | {'σ_eff':>8} | {'CV_eff':>8} | {'2ΔΦ/σ²':>8} | {'K_corr':>8} | {'D_exact':>8} |")
print(f"|{'-'*7}|{'-'*14}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|")
for q_val in q_vals_bistable:
    if q_val in duality_results:
        d = duality_results[q_val]
        dp = barrier_results[q_val]['DeltaPhi']
        print(f"| {q_val:5.1f} | {dp:12.8f} | {d['sigma_eff']:8.6f} | {d['CV_eff']:8.6f} | {d['barrier_ratio']:8.4f} | {d['K_corr']:8.4f} | {d['D_exact']:8.1f} |")

print("\n## Table 4: Blow-up time")
print(f"| {'q':>5} | {'T_unreg':>10} | {'MFPT':>12} | {'D_transit':>10} | {'D_exact':>10} | {'Ratio':>8} |")
print(f"|{'-'*7}|{'-'*12}|{'-'*14}|{'-'*12}|{'-'*12}|{'-'*10}|")
for q_val in q_vals_bistable:
    if q_val in blowup_results:
        bl = blowup_results[q_val]
        print(f"| {q_val:5.1f} | {bl['T_unreg']:10.4f} | {bl['MFPT']:12.4f} | {bl['D_transit']:10.4f} | {200.0:10.1f} | {bl['ratio']:8.4f} |")

print("\n\nDONE.")
