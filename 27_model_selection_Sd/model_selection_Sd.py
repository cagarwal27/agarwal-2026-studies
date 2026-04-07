#!/usr/bin/env python3
"""
FORMAL MODEL SELECTION FOR S(d) = 1/P(bistable|d)
===================================================
Fits competing models to the cusp bridge P(bistable) vs d data,
computes AIC, AICc, BIC, and reports model preference.

Data: 13 points in the exponential tail regime (d >= 14), from:
  - bridge_dimensional_scaling.py (d=14-32, P directly measured)
  - bridge_high_d_scaling.py (d=38-62, P from MC)
  - bridge_high_d_extension.py (d=68-80, P from MC)

Models tested:
  M1: ln(1/P) = c0 + gamma*d                          [exponential, 2 params]
  M2: ln(1/P) = c0 + alpha*ln(d)                      [power law, 2 params]
  M3: ln(1/P) = c0 + g1*d + g2*d^2                    [quadratic, 3 params]
  M4: ln(1/P) = c0 + gamma*d^beta                     [stretched exp, 3 params]

Output: AIC, AICc, BIC, delta-AIC, Akaike weights, extrapolation to d=150.
"""

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.stats import linregress
import sys

np.random.seed(42)


# ============================================================
# DATA (all P(bistable) measurements, d >= 14)
# ============================================================

# From bridge_dimensional_scaling.py (direct MC measurements)
D_LOW = np.array([14, 17, 20, 26, 32], dtype=float)
P_LOW = np.array([0.320, 0.275, 0.211, 0.095, 0.024])

# From bridge_high_d_scaling.py (MC, 100K-2M samples each)
D_MID = np.array([38, 44, 50, 56, 62], dtype=float)
P_MID = np.array([3.64e-3, 6.25e-4, 1.04e-4, 3.40e-5, 2.20e-5])

# From bridge_high_d_extension.py (MC, 3M-10M samples each)
D_HIGH = np.array([68, 74, 80], dtype=float)
P_HIGH = np.array([8.67e-6, 4.40e-6, 1.50e-6])

# Combined
d_all = np.concatenate([D_LOW, D_MID, D_HIGH])
P_all = np.concatenate([P_LOW, P_MID, P_HIGH])
y_all = np.log(1.0 / P_all)  # ln(1/P) = ln(S)

n = len(d_all)  # 13 data points


def pprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def model_exp(d, c0, gamma):
    """M1: ln(1/P) = c0 + gamma*d"""
    return c0 + gamma * d

def model_power(d, c0, alpha):
    """M2: ln(1/P) = c0 + alpha*ln(d)"""
    return c0 + alpha * np.log(d)

def model_quad(d, c0, g1, g2):
    """M3: ln(1/P) = c0 + g1*d + g2*d^2"""
    return c0 + g1 * d + g2 * d**2

def model_stretched(d, c0, gamma, beta):
    """M4: ln(1/P) = c0 + gamma*d^beta"""
    return c0 + gamma * d**beta


# ============================================================
# FIT EACH MODEL (least squares => Gaussian likelihood)
# ============================================================

results = {}

# --- M1: Exponential (2 params) ---
slope, intercept, r_value, _, _ = linregress(d_all, y_all)
y_pred_1 = model_exp(d_all, intercept, slope)
rss_1 = np.sum((y_all - y_pred_1)**2)
results['M1_exp'] = {
    'k': 2, 'params': {'c0': intercept, 'gamma': slope},
    'RSS': rss_1, 'R2': r_value**2, 'y_pred': y_pred_1,
    'label': f'ln(1/P) = {intercept:.3f} + {slope:.4f}*d'
}

# --- M2: Power law (2 params) ---
slope2, intercept2, r_value2, _, _ = linregress(np.log(d_all), y_all)
y_pred_2 = model_power(d_all, intercept2, slope2)
rss_2 = np.sum((y_all - y_pred_2)**2)
results['M2_power'] = {
    'k': 2, 'params': {'c0': intercept2, 'alpha': slope2},
    'RSS': rss_2, 'R2': r_value2**2, 'y_pred': y_pred_2,
    'label': f'ln(1/P) = {intercept2:.3f} + {slope2:.3f}*ln(d)'
}

# --- M3: Quadratic (3 params) ---
try:
    popt3, _ = curve_fit(model_quad, d_all, y_all, p0=[0, 0.1, 0.001])
    y_pred_3 = model_quad(d_all, *popt3)
    rss_3 = np.sum((y_all - y_pred_3)**2)
    ss_tot = np.sum((y_all - np.mean(y_all))**2)
    r2_3 = 1 - rss_3 / ss_tot
    results['M3_quad'] = {
        'k': 3, 'params': {'c0': popt3[0], 'g1': popt3[1], 'g2': popt3[2]},
        'RSS': rss_3, 'R2': r2_3, 'y_pred': y_pred_3,
        'label': f'ln(1/P) = {popt3[0]:.3f} + {popt3[1]:.4f}*d + {popt3[2]:.6f}*d^2'
    }
except Exception as e:
    pprint(f"M3 fit failed: {e}")

# --- M4: Stretched exponential (3 params) ---
try:
    popt4, _ = curve_fit(model_stretched, d_all, y_all, p0=[0, 0.1, 1.0],
                         bounds=([-np.inf, 0, 0.1], [np.inf, np.inf, 3.0]),
                         maxfev=10000)
    y_pred_4 = model_stretched(d_all, *popt4)
    rss_4 = np.sum((y_all - y_pred_4)**2)
    ss_tot = np.sum((y_all - np.mean(y_all))**2)
    r2_4 = 1 - rss_4 / ss_tot
    results['M4_stretched'] = {
        'k': 3, 'params': {'c0': popt4[0], 'gamma': popt4[1], 'beta': popt4[2]},
        'RSS': rss_4, 'R2': r2_4, 'y_pred': y_pred_4,
        'label': f'ln(1/P) = {popt4[0]:.3f} + {popt4[1]:.4f}*d^{popt4[2]:.3f}'
    }
except Exception as e:
    pprint(f"M4 fit failed: {e}")


# ============================================================
# INFORMATION CRITERIA
# ============================================================
# Under Gaussian errors with unknown variance:
#   ln(L_max) = -(n/2)*ln(RSS/n) - n/2 - (n/2)*ln(2*pi)
# AIC = 2*k - 2*ln(L_max)  [k = number of model params + 1 for sigma]
# AICc = AIC + 2*k*(k+1)/(n-k-1)  [small-sample correction]
# BIC = k*ln(n) - 2*ln(L_max)

pprint("=" * 80)
pprint("  FORMAL MODEL SELECTION FOR S(d)")
pprint(f"  n = {n} data points (d = {d_all.min():.0f} to {d_all.max():.0f})")
pprint("=" * 80)

pprint(f"\n{'Data':^80s}")
pprint(f"{'d':>6s} | {'P(bistable)':>14s} | {'ln(1/P)':>10s}")
pprint("-" * 36)
for d_val, p_val, y_val in zip(d_all, P_all, y_all):
    pprint(f"{d_val:6.0f} | {p_val:14.6e} | {y_val:10.4f}")

pprint(f"\n{'=' * 80}")
pprint(f"  MODEL FITS")
pprint(f"{'=' * 80}")

for name, r in sorted(results.items()):
    pprint(f"\n{name}: {r['label']}")
    pprint(f"  k = {r['k']} params, RSS = {r['RSS']:.4f}, R^2 = {r['R2']:.6f}")
    pprint(f"  Residuals: max = {np.max(np.abs(r['y_pred'] - y_all)):.3f}, "
           f"rms = {np.sqrt(r['RSS']/n):.3f}")

# Compute information criteria
# k_eff = model params + 1 (variance is also estimated)
pprint(f"\n{'=' * 80}")
pprint(f"  INFORMATION CRITERIA")
pprint(f"{'=' * 80}")

ic_results = {}
for name, r in sorted(results.items()):
    k_model = r['k']
    k_eff = k_model + 1  # +1 for variance parameter
    rss = r['RSS']

    # Log-likelihood (Gaussian)
    ln_L = -(n / 2) * np.log(rss / n) - n / 2 - (n / 2) * np.log(2 * np.pi)

    # AIC
    aic = 2 * k_eff - 2 * ln_L

    # AICc (corrected for small samples)
    if n - k_eff - 1 > 0:
        aicc = aic + 2 * k_eff * (k_eff + 1) / (n - k_eff - 1)
    else:
        aicc = np.inf

    # BIC
    bic = k_eff * np.log(n) - 2 * ln_L

    ic_results[name] = {
        'AIC': aic, 'AICc': aicc, 'BIC': bic,
        'ln_L': ln_L, 'k_eff': k_eff
    }

# Delta-AIC and Akaike weights
aic_min = min(v['AIC'] for v in ic_results.values())
aicc_min = min(v['AICc'] for v in ic_results.values())
bic_min = min(v['BIC'] for v in ic_results.values())

pprint(f"\n{'Model':>15s} | {'k_eff':>5s} | {'AIC':>8s} | {'dAIC':>6s} | "
       f"{'AICc':>8s} | {'dAICc':>6s} | {'BIC':>8s} | {'dBIC':>6s} | "
       f"{'w(AICc)':>8s}")
pprint("-" * 90)

# Compute Akaike weights from AICc
delta_aicc = {name: ic['AICc'] - aicc_min for name, ic in ic_results.items()}
exp_terms = {name: np.exp(-0.5 * d) for name, d in delta_aicc.items()}
sum_exp = sum(exp_terms.values())
weights = {name: exp_terms[name] / sum_exp for name in exp_terms}

for name in sorted(ic_results.keys()):
    ic = ic_results[name]
    d_aic = ic['AIC'] - aic_min
    d_aicc = ic['AICc'] - aicc_min
    d_bic = ic['BIC'] - bic_min
    w = weights[name]
    pprint(f"{name:>15s} | {ic['k_eff']:5d} | {ic['AIC']:8.2f} | {d_aic:6.2f} | "
           f"{ic['AICc']:8.2f} | {d_aicc:6.2f} | {ic['BIC']:8.2f} | {d_bic:6.2f} | "
           f"{w:8.4f}")


# ============================================================
# LEAVE-ONE-OUT CROSS-VALIDATION
# ============================================================

pprint(f"\n{'=' * 80}")
pprint(f"  LEAVE-ONE-OUT CROSS-VALIDATION")
pprint(f"{'=' * 80}")

loocv = {}
for name, r in sorted(results.items()):
    sq_errors = []
    for i in range(n):
        d_train = np.delete(d_all, i)
        y_train = np.delete(y_all, i)
        d_test = d_all[i]
        y_test = y_all[i]

        try:
            if name == 'M1_exp':
                sl, it, _, _, _ = linregress(d_train, y_train)
                y_hat = it + sl * d_test
            elif name == 'M2_power':
                sl, it, _, _, _ = linregress(np.log(d_train), y_train)
                y_hat = it + sl * np.log(d_test)
            elif name == 'M3_quad':
                po, _ = curve_fit(model_quad, d_train, y_train, p0=[0, 0.1, 0.001])
                y_hat = model_quad(d_test, *po)
            elif name == 'M4_stretched':
                po, _ = curve_fit(model_stretched, d_train, y_train, p0=[0, 0.1, 1.0],
                                  bounds=([-np.inf, 0, 0.1], [np.inf, np.inf, 3.0]),
                                  maxfev=10000)
                y_hat = model_stretched(d_test, *po)
            else:
                continue
            sq_errors.append((y_hat - y_test)**2)
        except Exception:
            sq_errors.append(np.nan)

    rmse_cv = np.sqrt(np.nanmean(sq_errors))
    loocv[name] = rmse_cv

pprint(f"\n{'Model':>15s} | {'LOOCV RMSE':>12s}")
pprint("-" * 32)
for name in sorted(loocv.keys()):
    pprint(f"{name:>15s} | {loocv[name]:12.4f}")


# ============================================================
# EXTRAPOLATION TO d = 150 (S_0 = 10^13)
# ============================================================

pprint(f"\n{'=' * 80}")
pprint(f"  EXTRAPOLATION TO d = 150")
pprint(f"{'=' * 80}")

target_ln = 13.0 * np.log(10.0)  # ln(10^13) = 29.93
pprint(f"\nTarget: ln(S) = ln(10^13) = {target_ln:.2f}")
pprint(f"Current data range: d = {d_all.min():.0f} to {d_all.max():.0f}")
pprint(f"Extrapolation factor: {150/d_all.max():.2f}x beyond data")

d_extrap = 150.0
pprint(f"\n{'Model':>15s} | {'ln(S) at d=150':>15s} | {'log10(S)':>10s} | "
       f"{'d for S=10^13':>14s}")
pprint("-" * 62)

for name, r in sorted(results.items()):
    p = r['params']
    if name == 'M1_exp':
        y_150 = p['c0'] + p['gamma'] * d_extrap
        d_target = (target_ln - p['c0']) / p['gamma']
    elif name == 'M2_power':
        y_150 = p['c0'] + p['alpha'] * np.log(d_extrap)
        d_target = np.exp((target_ln - p['c0']) / p['alpha'])
    elif name == 'M3_quad':
        y_150 = p['c0'] + p['g1'] * d_extrap + p['g2'] * d_extrap**2
        # Solve quadratic for d_target
        disc = p['g1']**2 - 4 * p['g2'] * (p['c0'] - target_ln)
        if disc >= 0 and p['g2'] != 0:
            d1 = (-p['g1'] + np.sqrt(disc)) / (2 * p['g2'])
            d2 = (-p['g1'] - np.sqrt(disc)) / (2 * p['g2'])
            d_target = max(d1, d2) if max(d1, d2) > 0 else min(d1, d2)
        else:
            d_target = np.inf
    elif name == 'M4_stretched':
        y_150 = p['c0'] + p['gamma'] * d_extrap**p['beta']
        inner = (target_ln - p['c0']) / p['gamma']
        d_target = inner**(1.0 / p['beta']) if inner > 0 else np.inf

    pprint(f"{name:>15s} | {y_150:15.2f} | {y_150/np.log(10):10.2f} | "
           f"{d_target:14.0f}")


# ============================================================
# EXTRAPOLATION DIVERGENCE PROFILE
# ============================================================

pprint(f"\n{'=' * 80}")
pprint(f"  EXTRAPOLATION DIVERGENCE PROFILE")
pprint(f"  (How models differ beyond the data)")
pprint(f"{'=' * 80}")

d_profile = [80, 100, 120, 140, 150, 160, 180, 200]
pprint(f"\n{'d':>6s}", end="")
for name in sorted(results.keys()):
    pprint(f" | {name:>12s}", end="")
pprint(f" | {'spread':>8s}")
pprint("-" * 72)

for d_val in d_profile:
    vals = []
    pprint(f"{d_val:6d}", end="")
    for name in sorted(results.keys()):
        p = results[name]['params']
        if name == 'M1_exp':
            v = p['c0'] + p['gamma'] * d_val
        elif name == 'M2_power':
            v = p['c0'] + p['alpha'] * np.log(d_val)
        elif name == 'M3_quad':
            v = p['c0'] + p['g1'] * d_val + p['g2'] * d_val**2
        elif name == 'M4_stretched':
            v = p['c0'] + p['gamma'] * d_val**p['beta']
        vals.append(v)
        pprint(f" | {v:12.2f}", end="")
    spread = max(vals) - min(vals)
    pprint(f" | {spread:8.2f}")

pprint(f"\n  'spread' = max - min across models (in ln units)")
pprint(f"  At d=80 (data edge): spread shows in-sample agreement")
pprint(f"  At d=150 (target): spread shows extrapolation uncertainty")


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVAL FOR d(S=10^13)
# ============================================================

pprint(f"\n{'=' * 80}")
pprint(f"  BOOTSTRAP 95% CI FOR d(S = 10^13)")
pprint(f"  (Parametric bootstrap, 10000 resamples)")
pprint(f"{'=' * 80}")

N_BOOT = 10000
sigma_hat = np.sqrt(results['M1_exp']['RSS'] / n)

d_target_boot = {name: [] for name in results}

for _ in range(N_BOOT):
    # Parametric bootstrap: add Gaussian noise with estimated sigma
    y_boot = y_all + np.random.normal(0, sigma_hat, n)

    for name in results:
        try:
            if name == 'M1_exp':
                sl, it, _, _, _ = linregress(d_all, y_boot)
                dt = (target_ln - it) / sl if sl > 0 else np.inf
            elif name == 'M2_power':
                sl, it, _, _, _ = linregress(np.log(d_all), y_boot)
                dt = np.exp((target_ln - it) / sl) if sl > 0 else np.inf
            elif name == 'M3_quad':
                po, _ = curve_fit(model_quad, d_all, y_boot, p0=[0, 0.1, 0.001])
                disc = po[1]**2 - 4 * po[2] * (po[0] - target_ln)
                if disc >= 0 and po[2] != 0:
                    d1 = (-po[1] + np.sqrt(disc)) / (2 * po[2])
                    d2 = (-po[1] - np.sqrt(disc)) / (2 * po[2])
                    dt = max(d1, d2) if max(d1, d2) > 0 else np.inf
                else:
                    dt = np.inf
            elif name == 'M4_stretched':
                po, _ = curve_fit(model_stretched, d_all, y_boot, p0=[0, 0.1, 1.0],
                                  bounds=([-np.inf, 0, 0.1], [np.inf, np.inf, 3.0]),
                                  maxfev=10000)
                inner = (target_ln - po[0]) / po[1]
                dt = inner**(1.0 / po[2]) if inner > 0 else np.inf
            if np.isfinite(dt) and 50 < dt < 1000:
                d_target_boot[name].append(dt)
        except Exception:
            pass

pprint(f"\n{'Model':>15s} | {'d median':>10s} | {'95% CI':>22s} | {'n_valid':>8s}")
pprint("-" * 64)
for name in sorted(d_target_boot.keys()):
    arr = np.array(d_target_boot[name])
    if len(arr) > 100:
        lo, med, hi = np.percentile(arr, [2.5, 50, 97.5])
        pprint(f"{name:>15s} | {med:10.0f} | [{lo:8.0f}, {hi:8.0f}] | {len(arr):8d}")
    else:
        pprint(f"{name:>15s} | {'too few valid bootstraps':>42s} | {len(arr):8d}")


# ============================================================
# CONCLUSION
# ============================================================

pprint(f"\n{'=' * 80}")
pprint(f"  CONCLUSION")
pprint(f"{'=' * 80}")

# Find best model by each criterion
best_aic = min(ic_results, key=lambda x: ic_results[x]['AIC'])
best_aicc = min(ic_results, key=lambda x: ic_results[x]['AICc'])
best_bic = min(ic_results, key=lambda x: ic_results[x]['BIC'])
best_loocv = min(loocv, key=lambda x: loocv[x])

pprint(f"\n  Best by AIC:   {best_aic}")
pprint(f"  Best by AICc:  {best_aicc}")
pprint(f"  Best by BIC:   {best_bic}")
pprint(f"  Best by LOOCV: {best_loocv}")

# Power law rejection strength
if 'M2_power' in ic_results and 'M1_exp' in ic_results:
    d_aic_power = ic_results['M2_power']['AIC'] - ic_results['M1_exp']['AIC']
    d_bic_power = ic_results['M2_power']['BIC'] - ic_results['M1_exp']['BIC']
    pprint(f"\n  Power law vs exponential:")
    pprint(f"    dAIC = {d_aic_power:+.2f} (>10 = decisive rejection)")
    pprint(f"    dBIC = {d_bic_power:+.2f} (>10 = decisive rejection)")

# 2-param vs 3-param
if 'M3_quad' in ic_results and 'M1_exp' in ic_results:
    d_aic_quad = ic_results['M3_quad']['AICc'] - ic_results['M1_exp']['AICc']
    pprint(f"\n  Quadratic vs exponential (AICc):")
    pprint(f"    dAICc = {d_aic_quad:+.2f} (<-2 = evidence for quadratic; >2 = exponential preferred)")

if 'M4_stretched' in ic_results and 'M1_exp' in ic_results:
    d_aic_str = ic_results['M4_stretched']['AICc'] - ic_results['M1_exp']['AICc']
    pprint(f"\n  Stretched exp vs exponential (AICc):")
    pprint(f"    dAICc = {d_aic_str:+.2f} (<-2 = evidence for stretched; >2 = exponential preferred)")

pprint(f"\n  Key extrapolation numbers:")
for name in sorted(results.keys()):
    p = results[name]['params']
    if name == 'M1_exp':
        d_t = (target_ln - p['c0']) / p['gamma']
    elif name == 'M2_power':
        d_t = np.exp((target_ln - p['c0']) / p['alpha'])
    elif name == 'M3_quad':
        disc = p['g1']**2 - 4 * p['g2'] * (p['c0'] - target_ln)
        d_t = (-p['g1'] + np.sqrt(disc)) / (2 * p['g2']) if disc >= 0 else np.inf
    elif name == 'M4_stretched':
        inner = (target_ln - p['c0']) / p['gamma']
        d_t = inner**(1.0 / p['beta']) if inner > 0 else np.inf
    pprint(f"    {name}: d(S=10^13) = {d_t:.0f}")

pprint()
pprint("=" * 80)
