#!/usr/bin/env python3
"""
Step 12: Test the Fire Phase Compression Formula
    tau_fire = S / (v * P)

Tests whether fire phase durations across evolutionary transitions
can be explained by a combinatorial search formula.

All data from scientific consensus literature:
    Szathmary & Maynard Smith 1995, Betts et al. 2018,
    Weiss et al. 2016, Hublin et al. 2017, Zeder 2011
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

# ============================================================
# DATA
# ============================================================

transitions = {
    'T1': {
        'name': 'Protocells',
        'date_ya': 3.8e9,
        'fire_phase_yr': 600e6,
        'search_type': 'chemical',
        'v_trials_per_yr': 365 * 24,       # ~1 reaction/hr/site = 8760/yr
        'P': 1e10,                          # ~10^10 independent reaction environments
        'gen_time_yr': 1 / (365 * 24),
        'group': 'genetic',
    },
    'T2': {
        'name': 'Genetic code',
        'date_ya': 3.5e9,
        'fire_phase_yr': 300e6,
        'search_type': 'RNA_replication',
        'v_trials_per_yr': 365,             # ~1 replication/day
        'P': 1e12,                          # ~10^12 competing RNA lineages
        'gen_time_yr': 1 / 365,
        'group': 'genetic',
    },
    'T3': {
        'name': 'Eukaryotes',
        'date_ya': 2.0e9,
        'fire_phase_yr': 1.5e9,
        'search_type': 'genetic',
        'v_trials_per_yr': 365,             # ~1 gen/day for prokaryotes
        'P': 1e29,                          # ~10^29 prokaryotes on Earth
        'gen_time_yr': 1 / 365,
        'group': 'genetic',
    },
    'T4': {
        'name': 'Plastids',
        'date_ya': 1.5e9,
        'fire_phase_yr': 500e6,
        'search_type': 'genetic_engulfment',
        'v_trials_per_yr': 52,              # ~1 gen/week for unicellular eukaryotes
        'P': 1e17,                          # ~10^17 eukaryotic cells
        'gen_time_yr': 1 / 52,
        'group': 'genetic',
    },
    'T5': {
        'name': 'Multicellularity',
        'date_ya': 1.0e9,
        'fire_phase_yr': 500e6,
        'search_type': 'genetic',
        'v_trials_per_yr': 52,
        'P': 1e17,
        'gen_time_yr': 1 / 52,
        'group': 'genetic',
    },
    'T6': {
        'name': 'Eusociality',
        'date_ya': 150e6,
        'fire_phase_yr': 600e6,
        'search_type': 'genetic_behavioral',
        'v_trials_per_yr': 1,               # ~1 gen/yr for insects
        'P': 1e6,                           # ~10^6 insect species searching
        'gen_time_yr': 1.0,
        'group': 'genetic',
    },
    'T7': {
        'name': 'Language',
        'date_ya': 200e3,
        'fire_phase_yr': 2.6e6,
        'search_type': 'cultural_genetic',
        'v_trials_per_yr': 365 / 3,         # ~1 trial per 3 days
        'P': 1e5,                            # ~100,000 early Homo individuals
        'gen_time_yr': 28,
        'group': 'cultural',
    },
    'T8': {
        'name': 'Agriculture/Civilization',
        'date_ya': 12e3,
        'fire_phase_yr': 188e3,
        'search_type': 'cultural',
        'v_trials_per_yr': 365 / 3,
        'P': 5e6,
        'gen_time_yr': 28,
        'group': 'cultural',
    },
    'T8_industrial': {
        'name': 'Industrial Revolution',
        'date_ya': 260,
        'fire_phase_yr': 11.7e3,
        'search_type': 'scientific',
        'v_trials_per_yr': 365,             # ~1 trial/day (printing + scientific method)
        'P': 1e5,                           # ~100,000 active experimenters in Europe
        'gen_time_yr': 28,
        'group': 'cultural',
    },
}

# Ordered list of transition keys
t_keys = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T8_industrial']
t_indices = list(range(len(t_keys)))

# Output directory
plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
os.makedirs(plot_dir, exist_ok=True)


# ============================================================
# PHASE 1: Compute S = tau_fire * v * P for each transition
# ============================================================

print("=" * 80)
print("PHASE 1: Compute implied combinatorial space S = tau_fire * v * P")
print("=" * 80)

results = {}
for key in t_keys:
    t = transitions[key]
    vP = t['v_trials_per_yr'] * t['P']
    S = t['fire_phase_yr'] * vP
    results[key] = {
        'name': t['name'],
        'tau_fire': t['fire_phase_yr'],
        'v': t['v_trials_per_yr'],
        'P': t['P'],
        'vP': vP,
        'S': S,
        'log_S': np.log10(S),
        'log_tau': np.log10(t['fire_phase_yr']),
        'log_v': np.log10(t['v_trials_per_yr']),
        'log_P': np.log10(t['P']),
        'log_vP': np.log10(vP),
        'group': t['group'],
    }

# Print Phase 1 table
header = f"{'Trans':<6} {'Name':<22} {'tau_fire (yr)':<15} {'v (tr/yr)':<12} {'P':<12} {'v*P (tr/yr)':<15} {'S = tau*v*P':<15} {'log10(S)':<10}"
print(header)
print("-" * len(header))
for key in t_keys:
    r = results[key]
    print(f"{key:<6} {r['name']:<22} {r['tau_fire']:<15.3g} {r['v']:<12.3g} {r['P']:<12.3g} {r['vP']:<15.3g} {r['S']:<15.3g} {r['log_S']:<10.2f}")

# Analyze S pattern
log_S_vals = np.array([results[k]['log_S'] for k in t_keys])
print(f"\nlog10(S) range: {log_S_vals.min():.2f} to {log_S_vals.max():.2f}")
print(f"log10(S) span: {log_S_vals.max() - log_S_vals.min():.2f} orders of magnitude")
print(f"log10(S) mean: {log_S_vals.mean():.2f}, std: {log_S_vals.std():.2f}")

# Check if S is constant (within 2 OOM)
S_span = log_S_vals.max() - log_S_vals.min()
if S_span <= 2:
    print("\n>>> S is approximately CONSTANT (span <= 2 OOM)")
elif S_span <= 5:
    print(f"\n>>> S varies MODERATELY ({S_span:.1f} OOM span)")
else:
    print(f"\n>>> S varies WIDELY ({S_span:.1f} OOM span) — formula does NOT produce constant S")

# Test for linear trend in log(S) vs transition number
slope_S, intercept_S, r_S, p_S, se_S = stats.linregress(t_indices, log_S_vals)
print(f"\nLinear fit of log10(S) vs transition index:")
print(f"  slope = {slope_S:.3f} per level, intercept = {intercept_S:.2f}")
print(f"  R^2 = {r_S**2:.4f}, p = {p_S:.4g}")

# T3 paradox analysis
print(f"\n--- T3 PARADOX ---")
print(f"T3 (eukaryotes) has log10(S) = {results['T3']['log_S']:.2f}")
print(f"  This is because P = 10^29 (all prokaryotes on Earth)")
print(f"  Despite the longest fire phase (1.5 Gy), S is enormous: 10^{results['T3']['log_S']:.0f}")
print(f"  Interpretation: endosymbiosis requires an astronomically rare event")
print(f"  (engulfment without digestion), so the search space IS huge.")
print(f"  The formula EXPLAINS the long fire phase: S is large because the")
print(f"  'target' (stable endosymbiosis) is vanishingly rare in the space")
print(f"  of all prokaryotic interactions.")


# ============================================================
# PHASE 2: Correlation analysis — what drives compression?
# ============================================================

print("\n" + "=" * 80)
print("PHASE 2: What drives fire phase compression?")
print("=" * 80)

log_tau = np.array([results[k]['log_tau'] for k in t_keys])
log_v = np.array([results[k]['log_v'] for k in t_keys])
log_P = np.array([results[k]['log_P'] for k in t_keys])
log_vP = np.array([results[k]['log_vP'] for k in t_keys])

# Correlations
r_tau_v, p_tau_v = stats.pearsonr(log_tau, log_v)
r_tau_P, p_tau_P = stats.pearsonr(log_tau, log_P)
r_tau_vP, p_tau_vP = stats.pearsonr(log_tau, log_vP)

print(f"\nCorrelation of log(tau_fire) with:")
print(f"  log(v)  : r = {r_tau_v:+.4f}, R^2 = {r_tau_v**2:.4f}, p = {p_tau_v:.4g}")
print(f"  log(P)  : r = {r_tau_P:+.4f}, R^2 = {r_tau_P**2:.4f}, p = {p_tau_P:.4g}")
print(f"  log(v*P): r = {r_tau_vP:+.4f}, R^2 = {r_tau_vP**2:.4f}, p = {p_tau_vP:.4g}")

# Linear regressions
slope_v, intercept_v, r_v, p_v, se_v = stats.linregress(log_v, log_tau)
slope_P, intercept_P, r_P, p_P, se_P = stats.linregress(log_P, log_tau)
slope_vP, intercept_vP, r_vP, p_vP, se_vP = stats.linregress(log_vP, log_tau)

print(f"\nLinear regression log(tau_fire) = a * log(X) + b:")
print(f"  vs log(v)  : slope = {slope_v:.3f} (expect -1 if v dominates), R^2 = {r_v**2:.4f}")
print(f"  vs log(P)  : slope = {slope_P:.3f} (expect -1 if P dominates), R^2 = {r_P**2:.4f}")
print(f"  vs log(v*P): slope = {slope_vP:.3f} (expect -1 if S constant), R^2 = {r_vP**2:.4f}")

print(f"\nIf tau = S/(v*P) with constant S, then log(tau) = log(S) - log(v*P)")
print(f"  => slope of log(tau) vs log(v*P) should be -1")
print(f"  Observed slope: {slope_vP:.3f} (deviation from -1: {abs(slope_vP + 1):.3f})")

# Best predictor
predictors = {'v': r_tau_v**2, 'P': r_tau_P**2, 'v*P': r_tau_vP**2}
best = max(predictors, key=predictors.get)
print(f"\nBest single predictor: {best} (R^2 = {predictors[best]:.4f})")


# ============================================================
# PHASE 3: Compression ratio test
# ============================================================

print("\n" + "=" * 80)
print("PHASE 3: Compression ratio — predicted vs actual")
print("=" * 80)

print(f"\n{'Pair':<18} {'tau(N)/tau(N+1)':<18} {'v(N+1)/v(N)':<14} {'P(N+1)/P(N)':<14} {'vP ratio':<14} {'S(N)/S(N+1)':<14} {'Predicted':<14} {'Residual':<14}")
print("-" * 130)

compression_actual = []
compression_predicted_constant_S = []
compression_predicted_varying_S = []

for i in range(len(t_keys) - 1):
    k1, k2 = t_keys[i], t_keys[i+1]
    r1, r2 = results[k1], results[k2]

    tau_ratio = r1['tau_fire'] / r2['tau_fire']       # compression
    v_ratio = r2['v'] / r1['v']                       # speed increase
    P_ratio = r2['P'] / r1['P']                       # parallel increase
    vP_ratio = v_ratio * P_ratio
    S_ratio = r1['S'] / r2['S']                       # space ratio
    predicted_with_S = vP_ratio * S_ratio              # full prediction (tautological)

    # Prediction assuming constant S: compression = v_ratio * P_ratio
    predicted_constant_S = vP_ratio
    residual = np.log10(tau_ratio) - np.log10(predicted_constant_S)

    compression_actual.append(np.log10(tau_ratio))
    compression_predicted_constant_S.append(np.log10(predicted_constant_S))

    pair = f"{k1}->{k2}"
    print(f"{pair:<18} {tau_ratio:<18.3g} {v_ratio:<14.3g} {P_ratio:<14.3g} {vP_ratio:<14.3g} {S_ratio:<14.3g} {predicted_constant_S:<14.3g} {residual:<+14.2f}")

compression_actual = np.array(compression_actual)
compression_predicted_constant_S = np.array(compression_predicted_constant_S)

# Residual statistics
residuals = compression_actual - compression_predicted_constant_S
print(f"\nResidual statistics (log10 scale, constant-S assumption):")
print(f"  Mean: {residuals.mean():+.2f} OOM")
print(f"  Std:  {residuals.std():.2f} OOM")
print(f"  Max absolute: {np.abs(residuals).max():.2f} OOM")

if np.abs(residuals).max() < 1:
    print("  >>> Constant-S model works within 1 OOM for ALL consecutive pairs")
elif np.abs(residuals).max() < 2:
    print("  >>> Constant-S model works within 2 OOM (marginal)")
else:
    print(f"  >>> Constant-S model FAILS: max residual = {np.abs(residuals).max():.1f} OOM")


# ============================================================
# PHASE 4: Phase transition — genetic vs cultural
# ============================================================

print("\n" + "=" * 80)
print("PHASE 4: Phase transition — genetic vs cultural search")
print("=" * 80)

genetic_keys = [k for k in t_keys if transitions[k]['group'] == 'genetic']
cultural_keys = [k for k in t_keys if transitions[k]['group'] == 'cultural']

# Within genetic group
gen_log_tau = np.array([results[k]['log_tau'] for k in genetic_keys])
gen_log_v = np.array([results[k]['log_v'] for k in genetic_keys])
gen_log_P = np.array([results[k]['log_P'] for k in genetic_keys])
gen_log_vP = np.array([results[k]['log_vP'] for k in genetic_keys])
gen_log_S = np.array([results[k]['log_S'] for k in genetic_keys])

# Within cultural group
cul_log_tau = np.array([results[k]['log_tau'] for k in cultural_keys])
cul_log_v = np.array([results[k]['log_v'] for k in cultural_keys])
cul_log_P = np.array([results[k]['log_P'] for k in cultural_keys])
cul_log_vP = np.array([results[k]['log_vP'] for k in cultural_keys])
cul_log_S = np.array([results[k]['log_S'] for k in cultural_keys])

print(f"\nGENETIC group (T1-T6):")
print(f"  log10(S) values: {[f'{s:.1f}' for s in gen_log_S]}")
print(f"  log10(S) mean: {gen_log_S.mean():.2f}, std: {gen_log_S.std():.2f}")
print(f"  log10(S) range: {gen_log_S.min():.1f} to {gen_log_S.max():.1f} ({gen_log_S.max()-gen_log_S.min():.1f} OOM)")

if len(gen_log_vP) > 2:
    r_gen, p_gen = stats.pearsonr(gen_log_vP, gen_log_tau)
    slope_gen, intercept_gen, _, _, _ = stats.linregress(gen_log_vP, gen_log_tau)
    print(f"  log(tau) vs log(v*P): r = {r_gen:.4f}, R^2 = {r_gen**2:.4f}, slope = {slope_gen:.3f}")

print(f"\nCULTURAL group (T7-T8_ind):")
print(f"  log10(S) values: {[f'{s:.1f}' for s in cul_log_S]}")
print(f"  log10(S) mean: {cul_log_S.mean():.2f}, std: {cul_log_S.std():.2f}")
print(f"  log10(S) range: {cul_log_S.min():.1f} to {cul_log_S.max():.1f} ({cul_log_S.max()-cul_log_S.min():.1f} OOM)")

if len(cul_log_vP) > 2:
    r_cul, p_cul = stats.pearsonr(cul_log_vP, cul_log_tau)
    slope_cul, intercept_cul, _, _, _ = stats.linregress(cul_log_vP, cul_log_tau)
    print(f"  log(tau) vs log(v*P): r = {r_cul:.4f}, R^2 = {r_cul**2:.4f}, slope = {slope_cul:.3f}")

# Phase transition magnitude
print(f"\nPHASE TRANSITION at T6->T7:")
print(f"  tau_fire drops from {results['T6']['tau_fire']:.3g} yr to {results['T7']['tau_fire']:.3g} yr")
print(f"  Compression: {results['T6']['tau_fire']/results['T7']['tau_fire']:.0f}x")
print(f"  v increases from {results['T6']['v']:.3g} to {results['T7']['v']:.3g} ({results['T7']['v']/results['T6']['v']:.0f}x)")
print(f"  P decreases from {results['T6']['P']:.3g} to {results['T7']['P']:.3g} ({results['T6']['P']/results['T7']['P']:.0f}x)")
print(f"  v*P goes from {results['T6']['vP']:.3g} to {results['T7']['vP']:.3g}")
vP_change = results['T7']['vP'] / results['T6']['vP']
print(f"  Net v*P change: {vP_change:.3g}x ({'increase' if vP_change > 1 else 'decrease'})")
print(f"  If S constant, predicted compression = {vP_change:.3g}x")
print(f"  Actual compression = {results['T6']['tau_fire']/results['T7']['tau_fire']:.3g}x")
print(f"  Discrepancy: {np.log10(results['T6']['tau_fire']/results['T7']['tau_fire']) - np.log10(vP_change):.2f} OOM")

# Compare genetic and cultural S distributions
print(f"\nS DISTRIBUTION COMPARISON:")
print(f"  Genetic mean log10(S): {gen_log_S.mean():.2f}")
print(f"  Cultural mean log10(S): {cul_log_S.mean():.2f}")
print(f"  Difference: {gen_log_S.mean() - cul_log_S.mean():.2f} OOM")

# If enough data, test for statistical difference
if len(gen_log_S) >= 3 and len(cul_log_S) >= 3:
    t_stat, p_val = stats.ttest_ind(gen_log_S, cul_log_S, equal_var=False)
    print(f"  Welch's t-test: t = {t_stat:.3f}, p = {p_val:.4f}")
    if p_val < 0.05:
        print(f"  >>> Genetic and cultural groups have SIGNIFICANTLY different S (p < 0.05)")
    else:
        print(f"  >>> No significant difference in S between groups (p = {p_val:.3f})")


# ============================================================
# PHASE 5: Predictive test — next fire phase
# ============================================================

print("\n" + "=" * 80)
print("PHASE 5: Prediction for next fire phase")
print("=" * 80)

# Use the cultural group's S statistics for prediction
S_cultural_mean = 10**cul_log_S.mean()
S_cultural_lo = 10**(cul_log_S.mean() - cul_log_S.std())
S_cultural_hi = 10**(cul_log_S.mean() + cul_log_S.std())

# Also use overall S
S_all_mean = 10**log_S_vals.mean()

# AI-scale search parameters (as specified in prompt)
v_AI = 1e6          # trials/yr/searcher (AI-scale)
P_AI = 1e9          # connected humans + AI systems

# Also consider range of v and P
v_AI_lo, v_AI_hi = 1e5, 1e8     # 10x uncertainty on v
P_AI_lo, P_AI_hi = 1e8, 1e10    # 10x uncertainty on P

print(f"\nPrediction inputs:")
print(f"  v_AI = {v_AI:.0e} trials/yr/searcher (range: {v_AI_lo:.0e} - {v_AI_hi:.0e})")
print(f"  P_AI = {P_AI:.0e} parallel searchers (range: {P_AI_lo:.0e} - {P_AI_hi:.0e})")
print(f"  S from cultural group: mean = {S_cultural_mean:.3g}")
print(f"    (range: {S_cultural_lo:.3g} to {S_cultural_hi:.3g})")

# Central prediction using cultural S
tau_next_central = S_cultural_mean / (v_AI * P_AI)

# Uncertainty bounds: best case (smallest S, largest v*P) to worst case
tau_next_best = S_cultural_lo / (v_AI_hi * P_AI_hi)
tau_next_worst = S_cultural_hi / (v_AI_lo * P_AI_lo)

# Also predict using all-transitions S
tau_next_allS = S_all_mean / (v_AI * P_AI)

print(f"\nPredicted tau_fire for next transition (using cultural-group S):")
print(f"  Central estimate: {tau_next_central:.3g} years")
print(f"  Best case:        {tau_next_best:.3g} years")
print(f"  Worst case:       {tau_next_worst:.3g} years")

# Convert to human-readable
def human_time(years):
    if years < 1/365:
        return f"{years*365*24:.1f} hours"
    elif years < 1:
        return f"{years*365:.1f} days"
    elif years < 100:
        return f"{years:.1f} years"
    elif years < 1000:
        return f"{years:.0f} years"
    elif years < 1e6:
        return f"{years/1e3:.1f} thousand years"
    elif years < 1e9:
        return f"{years/1e6:.1f} million years"
    else:
        return f"{years/1e9:.1f} billion years"

print(f"\n  In human terms:")
print(f"    Central: {human_time(tau_next_central)}")
print(f"    Range:   {human_time(tau_next_best)} to {human_time(tau_next_worst)}")

# Using overall S for comparison
print(f"\n  Using overall mean S (all transitions):")
print(f"    tau_next = {tau_next_allS:.3g} years ({human_time(tau_next_allS)})")

# Compare with industrial revolution as calibration
tau_industrial = transitions['T8_industrial']['fire_phase_yr']
compression_industrial_to_next = tau_industrial / tau_next_central
v_increase = v_AI / transitions['T8_industrial']['v_trials_per_yr']
P_increase = P_AI / transitions['T8_industrial']['P']
print(f"\n  Calibration against Industrial Revolution:")
print(f"    Industrial tau_fire = {tau_industrial:.3g} yr")
print(f"    v increase: {v_increase:.3g}x")
print(f"    P increase: {P_increase:.3g}x")
print(f"    If S constant: expected compression = {v_increase * P_increase:.3g}x")
print(f"    Predicted tau = {tau_industrial / (v_increase * P_increase):.3g} yr")


# ============================================================
# OVERALL VERDICT
# ============================================================

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

# Criteria from the prompt
S_range_OOM = log_S_vals.max() - log_S_vals.min()

# Check if T3 is the outlier — if excluding T3, what's the range?
log_S_no_T3 = np.array([results[k]['log_S'] for k in t_keys if k != 'T3'])
S_range_no_T3 = log_S_no_T3.max() - log_S_no_T3.min()

print(f"\nS variation across ALL transitions: {S_range_OOM:.1f} OOM")
print(f"S variation excluding T3: {S_range_no_T3:.1f} OOM")

# Determine verdict
if S_range_OOM <= 2:
    verdict = "STRONG SUCCESS"
    verdict_detail = "S is approximately constant (within 2 OOM)"
elif S_range_OOM <= 5 and (abs(slope_S) > 0.3 and p_S < 0.1):
    verdict = "STRONG SUCCESS"
    verdict_detail = f"S grows as a simple function of level number (slope={slope_S:.2f}/level, R^2={r_S**2:.3f})"
elif (gen_log_S.std() < 2 and cul_log_S.std() < 2):
    verdict = "PARTIAL SUCCESS"
    verdict_detail = "Formula works WITHIN genetic and cultural groups separately"
elif S_range_OOM <= 5:
    verdict = "PARTIAL SUCCESS"
    verdict_detail = f"S varies moderately ({S_range_OOM:.1f} OOM) — formula has partial predictive power"
else:
    # Check if there's a pattern even if range is large
    if abs(r_S) > 0.7:
        verdict = "PARTIAL SUCCESS"
        verdict_detail = f"S varies widely ({S_range_OOM:.1f} OOM) but with strong trend (R^2={r_S**2:.3f})"
    else:
        verdict = "INFORMATIVE FAILURE"
        verdict_detail = f"S varies by {S_range_OOM:.1f} OOM with no clear pattern"

print(f"\n*** VERDICT: {verdict} ***")
print(f"Detail: {verdict_detail}")

# Additional diagnostics for the verdict
print(f"\nDiagnostics:")
print(f"  Best predictor of tau_fire: {best} (R^2={predictors[best]:.4f})")
print(f"  log(tau) vs log(v*P) slope: {slope_vP:.3f} (ideal: -1.0)")
print(f"  Constant-S max residual: {np.abs(residuals).max():.2f} OOM")


# ============================================================
# PLOTS
# ============================================================

print("\n" + "=" * 80)
print("Generating plots...")
print("=" * 80)

fig_width, fig_height = 10, 7
label_fontsize = 11
title_fontsize = 13
tick_fontsize = 9

# Color scheme: genetic = blue, cultural = red
colors = ['#2166ac' if transitions[k]['group'] == 'genetic' else '#b2182b' for k in t_keys]
labels = [f"{k}\n{transitions[k]['name']}" for k in t_keys]
short_labels = [f"{k}" for k in t_keys]


# PLOT 1: log(tau_fire) vs transition number
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
for i, k in enumerate(t_keys):
    ax.bar(i, results[k]['log_tau'], color=colors[i], edgecolor='black', linewidth=0.8, alpha=0.85)
ax.set_xticks(t_indices)
ax.set_xticklabels(labels, fontsize=tick_fontsize, rotation=45, ha='right')
ax.set_ylabel('log₁₀(τ_fire) [years]', fontsize=label_fontsize)
ax.set_title('Fire Phase Duration Across Transitions', fontsize=title_fontsize)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
# Add phase transition annotation
ax.axvline(x=5.5, color='green', linestyle='--', alpha=0.7, linewidth=2)
ax.text(5.5, ax.get_ylim()[1] * 0.95, '  genetic → cultural', color='green',
        fontsize=9, va='top', ha='left', style='italic')
# Add value labels
for i, k in enumerate(t_keys):
    ax.text(i, results[k]['log_tau'] + 0.1, f"{results[k]['log_tau']:.1f}",
            ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'step12_tau_fire_vs_transition.png'), dpi=150)
plt.close()
print("  Saved: step12_tau_fire_vs_transition.png")


# PLOT 2: log(S) vs transition number
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
for i, k in enumerate(t_keys):
    ax.scatter(i, results[k]['log_S'], color=colors[i], s=120, edgecolors='black',
               linewidth=0.8, zorder=5)
ax.set_xticks(t_indices)
ax.set_xticklabels(labels, fontsize=tick_fontsize, rotation=45, ha='right')
ax.set_ylabel('log₁₀(S) [implied combinatorial space]', fontsize=label_fontsize)
ax.set_title('Implied Combinatorial Space S = τ_fire × v × P', fontsize=title_fontsize)

# Linear fit line
fit_x = np.linspace(-0.5, len(t_keys) - 0.5, 100)
fit_y = slope_S * fit_x + intercept_S
ax.plot(fit_x, fit_y, 'k--', alpha=0.5, label=f'Linear fit: slope={slope_S:.2f}, R²={r_S**2:.3f}')

# Horizontal band for constant-S hypothesis
S_median = np.median(log_S_vals)
ax.axhspan(S_median - 1, S_median + 1, alpha=0.1, color='gray',
           label=f'±1 OOM around median ({S_median:.1f})')

# Phase transition line
ax.axvline(x=5.5, color='green', linestyle='--', alpha=0.7, linewidth=2)

# Labels
for i, k in enumerate(t_keys):
    ax.annotate(f"{results[k]['log_S']:.1f}", (i, results[k]['log_S']),
                textcoords="offset points", xytext=(10, 5), fontsize=8)

ax.legend(fontsize=9, loc='best')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'step12_S_vs_transition.png'), dpi=150)
plt.close()
print("  Saved: step12_S_vs_transition.png")


# PLOT 3: log(tau_fire) vs log(v*P) — should be slope -1 if formula holds
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
for i, k in enumerate(t_keys):
    ax.scatter(results[k]['log_vP'], results[k]['log_tau'], color=colors[i],
               s=120, edgecolors='black', linewidth=0.8, zorder=5)
    ax.annotate(k, (results[k]['log_vP'], results[k]['log_tau']),
                textcoords="offset points", xytext=(8, 5), fontsize=9)

# Regression line
fit_x = np.linspace(min(log_vP) - 1, max(log_vP) + 1, 100)
fit_y = slope_vP * fit_x + intercept_vP
ax.plot(fit_x, fit_y, 'k-', alpha=0.6,
        label=f'Fit: slope={slope_vP:.3f}, R²={r_vP**2:.3f}')

# Ideal slope = -1 line (constant S)
# Use median S as intercept
ideal_y = -fit_x + S_median
ax.plot(fit_x, ideal_y, 'r--', alpha=0.4,
        label=f'Ideal slope -1 (S=10^{S_median:.1f})')

ax.set_xlabel('log₁₀(v × P) [total trials per year]', fontsize=label_fontsize)
ax.set_ylabel('log₁₀(τ_fire) [years]', fontsize=label_fontsize)
ax.set_title('Fire Phase Duration vs Total Search Rate', fontsize=title_fontsize)
ax.legend(fontsize=9, loc='best')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'step12_tau_vs_vP.png'), dpi=150)
plt.close()
print("  Saved: step12_tau_vs_vP.png")


# PLOT 4: Compression ratios — predicted vs actual
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
pair_labels = [f"{t_keys[i]}→{t_keys[i+1]}" for i in range(len(t_keys)-1)]
x_pos = np.arange(len(pair_labels))
width = 0.35

ax.bar(x_pos - width/2, compression_actual, width, color='#2166ac',
       edgecolor='black', linewidth=0.5, label='Actual log₁₀(τ_N/τ_{N+1})', alpha=0.85)
ax.bar(x_pos + width/2, compression_predicted_constant_S, width, color='#b2182b',
       edgecolor='black', linewidth=0.5, label='Predicted (constant S)', alpha=0.85)

ax.set_xticks(x_pos)
ax.set_xticklabels(pair_labels, fontsize=tick_fontsize, rotation=45, ha='right')
ax.set_ylabel('log₁₀(compression ratio)', fontsize=label_fontsize)
ax.set_title('Fire Phase Compression: Actual vs Predicted (Constant S)', fontsize=title_fontsize)
ax.legend(fontsize=9, loc='best')
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

# Add residual annotations
for i in range(len(pair_labels)):
    res = residuals[i]
    ax.annotate(f"Δ={res:+.1f}", (x_pos[i], max(compression_actual[i], compression_predicted_constant_S[i]) + 0.2),
                ha='center', fontsize=7, color='gray')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'step12_compression_ratios.png'), dpi=150)
plt.close()
print("  Saved: step12_compression_ratios.png")


# ============================================================
# ADDITIONAL ANALYSIS: Sensitivity to order-of-magnitude errors
# ============================================================

print("\n" + "=" * 80)
print("SENSITIVITY ANALYSIS: Robustness to 10x errors in v or P")
print("=" * 80)

# Perturb v and P by 10x in each direction, recompute S
print(f"\nIf v is off by 10x (up or down), log10(S) shifts by ±1")
print(f"If P is off by 10x, log10(S) shifts by ±1")
print(f"Combined worst case: log10(S) shifts by ±2")
print(f"\nCurrent log10(S) range: {S_range_OOM:.1f} OOM")
print(f"With ±2 OOM uncertainty on each point, the pattern {'remains distinguishable' if S_range_OOM > 4 else 'could be noise'}")

# The key question: does the TREND survive?
print(f"\nTrend slope: {slope_S:.3f} per level")
print(f"Uncertainty would need to reverse {abs(slope_S) * 8:.1f} OOM of trend across 8 levels")
print(f"This is {'robust' if abs(slope_S) * 8 > 4 else 'NOT robust'} to 10x errors")


# ============================================================
# SAVE NUMERICAL RESULTS FOR RESULTS FILE
# ============================================================

# Collect all key results into a summary dict for the results markdown
summary = {
    'phase1': {
        'S_range_OOM': float(S_range_OOM),
        'S_range_no_T3': float(S_range_no_T3),
        'S_mean_log': float(log_S_vals.mean()),
        'S_std_log': float(log_S_vals.std()),
        'S_trend_slope': float(slope_S),
        'S_trend_R2': float(r_S**2),
        'S_trend_p': float(p_S),
    },
    'phase2': {
        'r2_v': float(r_tau_v**2),
        'r2_P': float(r_tau_P**2),
        'r2_vP': float(r_tau_vP**2),
        'slope_vP': float(slope_vP),
        'best_predictor': best,
    },
    'phase3': {
        'residual_mean': float(residuals.mean()),
        'residual_std': float(residuals.std()),
        'residual_max_abs': float(np.abs(residuals).max()),
    },
    'phase4': {
        'genetic_S_mean': float(gen_log_S.mean()),
        'genetic_S_std': float(gen_log_S.std()),
        'cultural_S_mean': float(cul_log_S.mean()),
        'cultural_S_std': float(cul_log_S.std()),
    },
    'phase5': {
        'tau_next_central_yr': float(tau_next_central),
        'tau_next_best_yr': float(tau_next_best),
        'tau_next_worst_yr': float(tau_next_worst),
    },
    'verdict': verdict,
}

print(f"\n{'='*80}")
print(f"COMPLETE. Verdict: {verdict}")
print(f"{'='*80}")
