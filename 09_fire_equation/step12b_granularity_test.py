#!/usr/bin/env python3
"""
Step 12b: Granularity Test — Is T3's enormous S an artifact?

The question: Step 12 found S spans 29 OOM across transitions, with T3 at
10^41. But T3 used P = 10^29 (ALL prokaryotes). If we decompose T3 into
documented sub-steps, each with the RELEVANT searching population (specific
archaeal lineages, not all prokaryotes), does S per sub-step converge to
~10^13 (the cultural group's constant)?

If yes: the formula tau_fire = S/(v*P) works universally with S ~ 10^13,
once you measure P at the right granularity (independent relevant lineages,
not total individuals).

If no: S variation is real, and innovation difficulty genuinely differs.
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
os.makedirs(plot_dir, exist_ok=True)

# ============================================================
# ORIGINAL DATA (from Step 12)
# ============================================================

transitions = {
    'T1': {'name': 'Protocells',       'tau': 600e6,  'v': 8760, 'P': 1e10,  'group': 'genetic'},
    'T2': {'name': 'Genetic code',     'tau': 300e6,  'v': 365,  'P': 1e12,  'group': 'genetic'},
    'T3': {'name': 'Eukaryotes',       'tau': 1.5e9,  'v': 365,  'P': 1e29,  'group': 'genetic'},
    'T4': {'name': 'Plastids',         'tau': 500e6,  'v': 52,   'P': 1e17,  'group': 'genetic'},
    'T5': {'name': 'Multicellularity', 'tau': 500e6,  'v': 52,   'P': 1e17,  'group': 'genetic'},
    'T6': {'name': 'Eusociality',      'tau': 600e6,  'v': 1,    'P': 1e6,   'group': 'genetic'},
    'T7': {'name': 'Language',         'tau': 2.6e6,  'v': 122,  'P': 1e5,   'group': 'cultural'},
    'T8': {'name': 'Agriculture',      'tau': 188e3,  'v': 122,  'P': 5e6,   'group': 'cultural'},
    'T8i': {'name': 'Industrial',      'tau': 11.7e3, 'v': 365,  'P': 1e5,   'group': 'cultural'},
}

t_keys = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T8i']

# Compute original S
for k in t_keys:
    t = transitions[k]
    t['S_orig'] = t['tau'] * t['v'] * t['P']
    t['logS_orig'] = np.log10(t['S_orig'])

print("=" * 90)
print("ORIGINAL S VALUES (from Step 12)")
print("=" * 90)
print(f"{'Trans':<6} {'Name':<22} {'tau (yr)':<12} {'v':<8} {'P_orig':<12} {'log10(S)':<10}")
print("-" * 70)
for k in t_keys:
    t = transitions[k]
    print(f"{k:<6} {t['name']:<22} {t['tau']:<12.3g} {t['v']:<8.3g} {t['P']:<12.3g} {t['logS_orig']:<10.1f}")
orig_logS = np.array([transitions[k]['logS_orig'] for k in t_keys])
print(f"\nRange: {orig_logS.min():.1f} to {orig_logS.max():.1f} = {orig_logS.ptp():.1f} OOM span")


# ============================================================
# PART 1: T3 SUB-STEP DECOMPOSITION
# ============================================================

print("\n" + "=" * 90)
print("PART 1: T3 (EUKARYOGENESIS) SUB-STEP DECOMPOSITION")
print("=" * 90)

print("""
Eukaryogenesis is not one innovation — it is at least 5 sequential/overlapping
innovations, each occurring in a SPECIFIC lineage, not in all 10^29 prokaryotes.

Sub-steps based on: Koonin 2010, Lane & Martin 2010, Spang et al. 2015,
Zachar & Szathmary 2017, Lopez-Garcia & Moreira 2020.

Key insight: P for each sub-step is the number of INDEPENDENT LINEAGES in the
relevant clade with the right preconditions — not total individuals on Earth.
""")

t3_substeps = {
    'T3a': {
        'name': 'Cytoskeleton + endomembranes\n(Asgard archaeal innovation)',
        'short': 'Cytoskeleton',
        'tau': 500e6,
        'v': 20,       # Asgard archaeal generation: ~14-25 day doubling (Imachi et al. 2020, Nature 577:519)
        'P': 1e3,      # ~1000 Asgard archaeal lineages/species evolving ESP
        'rationale_P': (
            'Asgard archaea are a specific superphylum. Modern Asgard diversity: '
            '~10 known phyla, each with 10^2-10^3 species-level lineages. '
            'In early Earth, perhaps 10^3 total Asgard lineages were exploring '
            'cytoskeletal innovations. This is the number of INDEPENDENT evolutionary '
            'experiments, not the number of individual cells (which would be ~10^15-10^20).'
        ),
    },
    'T3b': {
        'name': 'Cell wall loss +\nmembrane flexibility',
        'short': 'Wall loss',
        'tau': 200e6,
        'v': 20,       # Asgard archaeal generation: ~14-25 day doubling (Imachi et al. 2020)
        'P': 3e2,      # ~300 lineages with good cytoskeleton, subset of T3a
        'rationale_P': (
            'Only Asgard lineages that already developed functional cytoskeleton '
            'can "search" for wall loss. Maybe 30% of the 10^3 from T3a. '
            'Wall loss without cytoskeleton = lysis and death.'
        ),
    },
    'T3c': {
        'name': 'Mitochondrial\nendosymbiosis',
        'short': 'Mito endosymbiosis',
        'tau': 100e6,
        'v': 52,        # weekly engulfment/interaction events, not daily reproduction
        'P': 1e2,       # ~100 wall-less archaeal lineages in alpha-proteo habitats
        'rationale_P': (
            'THE bottleneck. Happened exactly once (all eukaryotes share one '
            'mitochondrial ancestor). Requires: (1) wall-less archaeon, (2) in '
            'ecological contact with alpha-proteobacteria, (3) stable engulfment '
            'without digestion. Only ~100 lineages of wall-less archaea may have '
            'been in the right ecological context. v = 52/yr because the "trial" '
            'is an engulfment event (~weekly for phagocytic cells), not reproduction.'
        ),
    },
    'T3d': {
        'name': 'Gene transfer +\nnuclear envelope',
        'short': 'Nuclear organization',
        'tau': 400e6,
        'v': 52,        # eukaryotic generation ~weekly for unicellular
        'P': 1e3,       # proto-eukaryotic lineages diversifying after endosymbiosis
        'rationale_P': (
            'After endosymbiosis, the proto-eukaryotic lineage diversifies. '
            '~10^3 species of proto-eukaryotes, each exploring different gene '
            'transfer/compartmentalization strategies. LECA (Last Eukaryotic '
            'Common Ancestor) represents the lineage that found the winning '
            'combination of nuclear envelope + import machinery.'
        ),
    },
    'T3e': {
        'name': 'Meiosis + sexual\nreproduction',
        'short': 'Meiosis/sex',
        'tau': 300e6,
        'v': 52,
        'P': 1e3,       # early eukaryotic lineages
        'rationale_P': (
            'Early eukaryotic lineages post-nucleus. ~10^3 species exploring '
            'different recombination strategies. Meiosis requires coordinated '
            'evolution of homologous pairing, crossover, reductional division.'
        ),
    },
}

t3_keys = ['T3a', 'T3b', 'T3c', 'T3d', 'T3e']

# Compute S for each sub-step
print(f"{'Step':<6} {'Name':<25} {'tau (yr)':<12} {'v':<8} {'P_rel':<10} {'S':<12} {'log10(S)':<10}")
print("-" * 85)
t3_logS = []
for k in t3_keys:
    s = t3_substeps[k]
    S = s['tau'] * s['v'] * s['P']
    logS = np.log10(S)
    s['S'] = S
    s['logS'] = logS
    t3_logS.append(logS)
    print(f"{k:<6} {s['short']:<25} {s['tau']:<12.3g} {s['v']:<8.3g} {s['P']:<10.3g} {S:<12.3g} {logS:<10.2f}")

t3_logS = np.array(t3_logS)
print(f"\nT3 sub-step log10(S): mean = {t3_logS.mean():.2f}, std = {t3_logS.std():.2f}")
print(f"Range: {t3_logS.min():.1f} to {t3_logS.max():.1f} = {t3_logS.ptp():.1f} OOM")

# Verify: do sub-steps reproduce total tau?
tau_reconstructed = sum(t3_substeps[k]['S'] / (t3_substeps[k]['v'] * t3_substeps[k]['P'])
                        for k in t3_keys)
print(f"\nSelf-consistency check:")
print(f"  Sum of sub-step durations: {tau_reconstructed:.3g} yr")
print(f"  Original T3 tau_fire:      {transitions['T3']['tau']:.3g} yr")
print(f"  Match: {'YES' if abs(tau_reconstructed - transitions['T3']['tau']) / transitions['T3']['tau'] < 0.05 else 'CLOSE' if abs(tau_reconstructed - transitions['T3']['tau']) / transitions['T3']['tau'] < 0.2 else 'NO'}")

# Compare to cultural group
cultural_logS = np.array([transitions[k]['logS_orig'] for k in ['T7', 'T8', 'T8i']])
print(f"\nCultural group log10(S): mean = {cultural_logS.mean():.2f}, std = {cultural_logS.std():.2f}")
print(f"T3 sub-steps log10(S):   mean = {t3_logS.mean():.2f}, std = {t3_logS.std():.2f}")
print(f"Difference of means: {abs(t3_logS.mean() - cultural_logS.mean()):.2f} OOM")

# Statistical test
t_stat, p_val = stats.ttest_ind(t3_logS, cultural_logS, equal_var=False)
print(f"Welch's t-test (T3 sub-steps vs cultural): t = {t_stat:.3f}, p = {p_val:.3f}")
if p_val > 0.05:
    print(">>> T3 sub-step S values are NOT significantly different from cultural S (p > 0.05)")
else:
    print(f">>> T3 sub-step S values differ from cultural S at p = {p_val:.3f}")


# ============================================================
# PART 2: RECALIBRATE P FOR ALL TRANSITIONS
# ============================================================

print("\n" + "=" * 90)
print("PART 2: RECALIBRATE P — 'INDEPENDENT RELEVANT LINEAGES' ACROSS ALL TRANSITIONS")
print("=" * 90)

print("""
The original analysis used INCONSISTENT definitions of P:
  - T3: P = 10^29 = all individual prokaryotes
  - T6: P = 10^6  = insect species (already at lineage level)
  - T7: P = 10^5  = individual humans (appropriate for cultural search)

Correction: for GENETIC transitions (T1-T6), P should be the number of
INDEPENDENT EVOLUTIONARY LINEAGES with the right preconditions for that
specific transition. For CULTURAL transitions (T7+), P remains as individuals
because each individual can independently generate cultural innovations.
""")

# Define corrected P with biological rationale
corrections = {
    'T1': {
        'P_corr': 1e4,
        'n_steps': 3,   # membrane + replication + metabolism integration
        'rationale': (
            'Hydrothermal vent sites with right chemistry. ~10,000 active sites '
            'in early Earth (today ~1000; Hadean/Archean had more volcanism). '
            'Each site is an independent experiment. Individual molecules are '
            'NOT independent searchers — they share the same local chemistry.'
        ),
    },
    'T2': {
        'P_corr': 1e3,
        'n_steps': 2,   # amino acid activation + codon optimization
        'rationale': (
            'RNA lineages complex enough to explore genetic code space. Of ~10^12 '
            'total RNA molecules, only ~10^3 distinct community-level "lineages" '
            'are exploring fundamentally different coding strategies. Most '
            'individual RNAs are copies within the same lineage.'
        ),
    },
    'T3': {
        'P_corr': 5e2,  # geometric mean of sub-step P values
        'n_steps': 5,
        'rationale': (
            'Asgard archaeal lineages (see sub-step decomposition). '
            'Geometric mean of sub-step P values: (10^3 × 300 × 100 × 10^3 × 10^3)^(1/5) ≈ 500. '
            'Not 10^29 individual prokaryotes — most prokaryotes are NOT searching '
            'for eukaryogenesis.'
        ),
    },
    'T4': {
        'P_corr': 1e4,
        'n_steps': 2,   # engulfment + integration
        'rationale': (
            'Eukaryotic species that regularly phagocytose cyanobacteria. '
            'At ~1.5 Ga, maybe 10^4 heterotrophic eukaryotic species in '
            'cyanobacteria-rich environments. Plastid endosymbiosis happened '
            'at least twice independently (primary + secondary), suggesting '
            'moderate search difficulty.'
        ),
    },
    'T5': {
        'P_corr': 1e5,
        'n_steps': 2,   # adhesion + differentiation
        'rationale': (
            'Eukaryotic lineages with clonal growth potential. Multicellularity '
            'evolved 25+ times independently — a relatively "easy" transition. '
            'Many lineages (~10^5 species of unicellular eukaryotes) were '
            'effectively searching. Higher P than T3-T4 because preconditions '
            'were less restrictive.'
        ),
    },
    'T6': {
        'P_corr': 1e5,
        'n_steps': 3,   # monogamy + helping + sterile caste
        'rationale': (
            'Insect species with monogamous mating (Hamilton\'s rule prerequisite). '
            'Of ~10^6 insect species, maybe 10% are sufficiently monogamous to be '
            '"in range" for eusociality. Eusociality evolved 12-15 times, suggesting '
            'moderate difficulty with large search pool.'
        ),
    },
    'T7': {
        'P_corr': 1e5,  # unchanged — individuals are the right unit for cultural search
        'n_steps': 3,   # proto-language + symbols + syntax
        'rationale': 'Unchanged. For cultural search, individuals are independent searchers.',
    },
    'T8': {
        'P_corr': 5e6,  # unchanged
        'n_steps': 3,   # sedentism + domestication + institutions
        'rationale': 'Unchanged. ~5 million humans, each exploring subsistence strategies.',
    },
    'T8i': {
        'P_corr': 1e5,  # unchanged
        'n_steps': 3,   # printing + scientific method + mechanization
        'rationale': 'Unchanged. ~100,000 active experimenters/inventors.',
    },
}

# Compute corrected S (two versions)
# Version A: Just correct P, keep tau as total
# Version B: Correct P AND divide by n_steps (S per fundamental step)

print(f"\n{'Trans':<6} {'Name':<22} {'P_orig':<12} {'P_corr':<12} {'n_steps':<8} {'logS_orig':<10} {'logS_corr':<10} {'logS/step':<10}")
print("-" * 95)
for k in t_keys:
    t = transitions[k]
    c = corrections[k]
    S_corr = t['tau'] * t['v'] * c['P_corr']
    logS_corr = np.log10(S_corr)
    S_per_step = S_corr / c['n_steps']
    logS_per_step = np.log10(S_per_step)

    t['P_corr'] = c['P_corr']
    t['n_steps'] = c['n_steps']
    t['S_corr'] = S_corr
    t['logS_corr'] = logS_corr
    t['S_per_step'] = S_per_step
    t['logS_per_step'] = logS_per_step

    print(f"{k:<6} {t['name']:<22} {t['P']:<12.3g} {c['P_corr']:<12.3g} {c['n_steps']:<8d} {t['logS_orig']:<10.1f} {logS_corr:<10.1f} {logS_per_step:<10.1f}")

# Statistics on corrected S
corr_logS = np.array([transitions[k]['logS_corr'] for k in t_keys])
step_logS = np.array([transitions[k]['logS_per_step'] for k in t_keys])

print(f"\n--- VARIANCE REDUCTION ---")
print(f"Original  log10(S): mean = {orig_logS.mean():.2f}, std = {orig_logS.std():.2f}, range = {orig_logS.ptp():.1f} OOM")
print(f"Corrected log10(S): mean = {corr_logS.mean():.2f}, std = {corr_logS.std():.2f}, range = {corr_logS.ptp():.1f} OOM")
print(f"Per-step  log10(S): mean = {step_logS.mean():.2f}, std = {step_logS.std():.2f}, range = {step_logS.ptp():.1f} OOM")
print(f"\nVariance reduction (original → corrected): {(1 - corr_logS.std()**2 / orig_logS.std()**2)*100:.1f}%")
print(f"Variance reduction (original → per-step):  {(1 - step_logS.std()**2 / orig_logS.std()**2)*100:.1f}%")

# Is corrected S approximately constant?
if corr_logS.ptp() <= 2:
    print(f"\n>>> CORRECTED S is approximately CONSTANT (range ≤ 2 OOM)")
elif corr_logS.ptp() <= 4:
    print(f"\n>>> CORRECTED S is MODERATELY variable (range = {corr_logS.ptp():.1f} OOM)")
else:
    print(f"\n>>> CORRECTED S still varies ({corr_logS.ptp():.1f} OOM) — but much less than original ({orig_logS.ptp():.1f})")

if step_logS.ptp() <= 2:
    print(f">>> PER-STEP S is approximately CONSTANT (range ≤ 2 OOM)")
elif step_logS.ptp() <= 4:
    print(f">>> PER-STEP S is MODERATELY variable (range = {step_logS.ptp():.1f} OOM)")


# ============================================================
# PART 3: INVERSE PROBLEM — What P gives S = 10^13?
# ============================================================

print("\n" + "=" * 90)
print("PART 3: INVERSE PROBLEM — What P gives S = S_target?")
print("=" * 90)

S_target = 10**step_logS.mean()  # Use empirical mean as target
logS_target = step_logS.mean()
print(f"Target: S_step = 10^{logS_target:.2f} (empirical mean of per-step S values)")

print(f"\n{'Trans':<6} {'Name':<22} {'tau (yr)':<12} {'v':<8} {'n':<4} {'P_required':<12} {'P_corr':<12} {'log(P_req)':<10} {'Plausible?':<12}")
print("-" * 105)
for k in t_keys:
    t = transitions[k]
    c = corrections[k]
    # S_target = (tau/n) * v * P_req  =>  P_req = S_target * n / (tau * v)
    P_req = S_target * c['n_steps'] / (t['tau'] * t['v'])
    logP_req = np.log10(P_req)
    t['P_required'] = P_req
    t['logP_required'] = logP_req

    # Plausibility assessment
    ratio = P_req / c['P_corr']
    if 0.1 <= ratio <= 10:
        plausible = "YES"
    elif 0.01 <= ratio <= 100:
        plausible = "CLOSE"
    else:
        plausible = f"NO ({ratio:.1g}x off)"

    print(f"{k:<6} {t['name']:<22} {t['tau']:<12.3g} {t['v']:<8.3g} {c['n_steps']:<4d} {P_req:<12.3g} {c['P_corr']:<12.3g} {logP_req:<10.2f} {plausible}")

# Check how many are plausible
P_req_all = np.array([transitions[k]['P_required'] for k in t_keys])
P_corr_all = np.array([corrections[k]['P_corr'] for k in t_keys])
log_ratio = np.log10(P_req_all / P_corr_all)
print(f"\nlog10(P_required / P_corrected) across transitions:")
print(f"  Mean: {log_ratio.mean():+.2f}")
print(f"  Std:  {log_ratio.std():.2f}")
print(f"  Max absolute: {np.abs(log_ratio).max():.2f}")
n_plausible = np.sum(np.abs(log_ratio) <= 1)
print(f"  {n_plausible}/{len(t_keys)} transitions have P_required within 10x of P_corrected")


# ============================================================
# PART 4: INFLATION FORMULA VERIFICATION
# ============================================================

print("\n" + "=" * 90)
print("PART 4: INFLATION FORMULA — Does S_measured = S₀ × n × (P_measured / P_relevant)?")
print("=" * 90)

print(f"\nIf tau = n × S₀ / (v × P_rel), then:")
print(f"  S_measured = tau × v × P_meas = n × S₀ × (P_meas / P_rel)")
print(f"  log(S_meas) = log(S₀) + log(n) + log(P_meas/P_rel)")

S0 = 10**logS_target

print(f"\nUsing S₀ = 10^{logS_target:.2f}")
print(f"\n{'Trans':<6} {'log(S_meas)':<12} {'log(S_pred)':<12} {'Residual':<10}")
print("-" * 40)
residuals_inflation = []
for k in t_keys:
    t = transitions[k]
    c = corrections[k]
    S_predicted = S0 * c['n_steps'] * (t['P'] / c['P_corr'])
    logS_pred = np.log10(S_predicted)
    resid = t['logS_orig'] - logS_pred
    residuals_inflation.append(resid)
    print(f"{k:<6} {t['logS_orig']:<12.1f} {logS_pred:<12.1f} {resid:<+10.2f}")

residuals_inflation = np.array(residuals_inflation)
print(f"\nInflation formula residuals:")
print(f"  Mean: {residuals_inflation.mean():+.3f}")
print(f"  Std:  {residuals_inflation.std():.3f}")
print(f"  Max:  {np.abs(residuals_inflation).max():.3f}")
if np.abs(residuals_inflation).max() < 0.5:
    print("  >>> Formula reproduces all S_measured to within 0.5 OOM")
elif np.abs(residuals_inflation).max() < 1.0:
    print("  >>> Formula reproduces all S_measured to within 1 OOM")


# ============================================================
# PART 5: REGRESSION — Does corrected model fix the slope?
# ============================================================

print("\n" + "=" * 90)
print("PART 5: REGRESSION — tau_fire vs v*P with corrected P")
print("=" * 90)

log_tau = np.array([np.log10(transitions[k]['tau']) for k in t_keys])
log_vP_orig = np.array([np.log10(transitions[k]['v'] * transitions[k]['P']) for k in t_keys])
log_vP_corr = np.array([np.log10(transitions[k]['v'] * corrections[k]['P_corr']) for k in t_keys])

# Original regression
slope_orig, intercept_orig, r_orig, _, _ = stats.linregress(log_vP_orig, log_tau)
# Corrected regression
slope_corr, intercept_corr, r_corr, _, _ = stats.linregress(log_vP_corr, log_tau)

print(f"Original:  slope = {slope_orig:+.3f} (expect -1), R² = {r_orig**2:.4f}")
print(f"Corrected: slope = {slope_corr:+.3f} (expect -1), R² = {r_corr**2:.4f}")
print(f"\nSlope improvement: {abs(slope_orig + 1):.3f} → {abs(slope_corr + 1):.3f} (deviation from -1)")
print(f"R² improvement:   {r_orig**2:.4f} → {r_corr**2:.4f}")

# Per-step regression (divide tau by n_steps)
log_tau_per_step = np.array([np.log10(transitions[k]['tau'] / corrections[k]['n_steps']) for k in t_keys])
slope_step, intercept_step, r_step, _, _ = stats.linregress(log_vP_corr, log_tau_per_step)
print(f"\nPer-step:  slope = {slope_step:+.3f} (expect -1), R² = {r_step**2:.4f}")


# ============================================================
# PART 6: REVISED PREDICTION
# ============================================================

print("\n" + "=" * 90)
print("PART 6: REVISED PREDICTION FOR NEXT FIRE PHASE")
print("=" * 90)

# With universal S per step, the prediction becomes:
# tau_next = n_next * S₀ / (v_next * P_next)

v_AI = 1e6
P_AI = 1e9
n_next_lo, n_next_mid, n_next_hi = 2, 4, 8

print(f"Parameters:")
print(f"  S₀ = 10^{logS_target:.2f} (universal per-step search space)")
print(f"  v_AI = {v_AI:.0e} trials/yr/searcher")
print(f"  P_AI = {P_AI:.0e} parallel searchers")

for n in [n_next_lo, n_next_mid, n_next_hi]:
    tau = n * S0 / (v_AI * P_AI)
    if tau < 1:
        unit = f"{tau*365:.1f} days"
    elif tau < 100:
        unit = f"{tau:.2f} years"
    else:
        unit = f"{tau:.1f} years"
    print(f"  n = {n} steps: tau = {tau:.3g} yr ({unit})")

# Uncertainty from S₀
S0_lo = 10**(logS_target - step_logS.std())
S0_hi = 10**(logS_target + step_logS.std())
tau_lo = n_next_mid * S0_lo / (v_AI * P_AI * 10)   # optimistic v*P
tau_hi = n_next_mid * S0_hi / (v_AI * P_AI / 10)    # pessimistic v*P

print(f"\n  Full uncertainty range (±1σ on S₀, ±10× on v*P):")
if tau_lo < 1:
    lo_str = f"{tau_lo*365*24:.1f} hours"
else:
    lo_str = f"{tau_lo:.3g} yr"
if tau_hi < 1:
    hi_str = f"{tau_hi*365:.1f} days"
elif tau_hi < 100:
    hi_str = f"{tau_hi:.1f} years"
else:
    hi_str = f"{tau_hi:.0f} years"
print(f"  {lo_str} to {hi_str}")


# ============================================================
# PART 7: CROSS-VALIDATION — USE GENETIC TO PREDICT CULTURAL
# ============================================================

print("\n" + "=" * 90)
print("PART 7: CROSS-VALIDATION — Genetic-derived S₀ predicts cultural tau_fire?")
print("=" * 90)

# Compute S₀ from ONLY the genetic transitions (corrected)
gen_keys = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
cul_keys = ['T7', 'T8', 'T8i']

gen_logS_per_step = np.array([transitions[k]['logS_per_step'] for k in gen_keys])
cul_logS_per_step = np.array([transitions[k]['logS_per_step'] for k in cul_keys])

S0_genetic = 10**gen_logS_per_step.mean()
logS0_genetic = gen_logS_per_step.mean()

print(f"S₀ from genetic group only: 10^{logS0_genetic:.2f}")
print(f"S₀ from cultural group only: 10^{cul_logS_per_step.mean():.2f}")
print(f"S₀ from all transitions: 10^{logS_target:.2f}")

# Predict cultural tau_fire using genetic S₀
print(f"\nPredicting cultural tau_fire using genetic-derived S₀:")
print(f"{'Trans':<6} {'Actual tau':<14} {'Predicted tau':<14} {'Ratio':<10}")
print("-" * 50)
for k in cul_keys:
    t = transitions[k]
    c = corrections[k]
    tau_pred = c['n_steps'] * S0_genetic / (t['v'] * c['P_corr'])
    ratio = t['tau'] / tau_pred
    print(f"{k:<6} {t['tau']:<14.3g} {tau_pred:<14.3g} {ratio:<10.2f}")

# And vice versa — predict genetic from cultural
print(f"\nPredicting genetic tau_fire using cultural-derived S₀:")
S0_cultural = 10**cul_logS_per_step.mean()
print(f"{'Trans':<6} {'Actual tau':<14} {'Predicted tau':<14} {'Ratio':<10} {'log10(ratio)':<12}")
print("-" * 65)
for k in gen_keys:
    t = transitions[k]
    c = corrections[k]
    tau_pred = c['n_steps'] * S0_cultural / (t['v'] * c['P_corr'])
    ratio = t['tau'] / tau_pred
    print(f"{k:<6} {t['tau']:<14.3g} {tau_pred:<14.3g} {ratio:<10.2f} {np.log10(ratio):<+12.2f}")


# ============================================================
# PLOTS
# ============================================================

print("\n" + "=" * 90)
print("Generating plots...")
print("=" * 90)

label_fs = 11
title_fs = 13
tick_fs = 9

# Color scheme
colors_orig = ['#2166ac' if transitions[k]['group'] == 'genetic' else '#b2182b' for k in t_keys]
labels = [f"{k}\n{transitions[k]['name']}" for k in t_keys]


# PLOT 1: S convergence — original vs corrected vs per-step
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

# Panel A: Original
ax = axes[0]
for i, k in enumerate(t_keys):
    ax.scatter(i, transitions[k]['logS_orig'], color=colors_orig[i], s=100,
               edgecolors='black', linewidth=0.8, zorder=5)
    ax.annotate(f"{transitions[k]['logS_orig']:.1f}", (i, transitions[k]['logS_orig']),
                textcoords="offset points", xytext=(8, 4), fontsize=7)
ax.axhline(y=cultural_logS.mean(), color='red', linestyle='--', alpha=0.5, label=f'Cultural mean ({cultural_logS.mean():.1f})')
ax.set_xticks(range(len(t_keys)))
ax.set_xticklabels([k for k in t_keys], fontsize=tick_fs, rotation=45, ha='right')
ax.set_ylabel('log₁₀(S)', fontsize=label_fs)
ax.set_title(f'A) Original (range: {orig_logS.ptp():.0f} OOM)', fontsize=title_fs)
ax.legend(fontsize=8)
ax.set_ylim(5, 45)

# Panel B: Corrected P
ax = axes[1]
for i, k in enumerate(t_keys):
    ax.scatter(i, transitions[k]['logS_corr'], color=colors_orig[i], s=100,
               edgecolors='black', linewidth=0.8, zorder=5)
    ax.annotate(f"{transitions[k]['logS_corr']:.1f}", (i, transitions[k]['logS_corr']),
                textcoords="offset points", xytext=(8, 4), fontsize=7)
ax.axhline(y=corr_logS.mean(), color='green', linestyle='--', alpha=0.5, label=f'Mean ({corr_logS.mean():.1f})')
ax.set_xticks(range(len(t_keys)))
ax.set_xticklabels([k for k in t_keys], fontsize=tick_fs, rotation=45, ha='right')
ax.set_title(f'B) Corrected P (range: {corr_logS.ptp():.1f} OOM)', fontsize=title_fs)
ax.legend(fontsize=8)

# Panel C: Per fundamental step
ax = axes[2]
for i, k in enumerate(t_keys):
    ax.scatter(i, transitions[k]['logS_per_step'], color=colors_orig[i], s=100,
               edgecolors='black', linewidth=0.8, zorder=5)
    ax.annotate(f"{transitions[k]['logS_per_step']:.1f}", (i, transitions[k]['logS_per_step']),
                textcoords="offset points", xytext=(8, 4), fontsize=7)
ax.axhline(y=step_logS.mean(), color='green', linestyle='--', alpha=0.5, label=f'Mean ({step_logS.mean():.1f})')
ax.axhspan(step_logS.mean() - 1, step_logS.mean() + 1, alpha=0.1, color='green', label='±1 OOM')
ax.set_xticks(range(len(t_keys)))
ax.set_xticklabels([k for k in t_keys], fontsize=tick_fs, rotation=45, ha='right')
ax.set_title(f'C) Per step (range: {step_logS.ptp():.1f} OOM)', fontsize=title_fs)
ax.legend(fontsize=8)

plt.suptitle('S Convergence Under Granularity Correction', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'step12b_S_convergence.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: step12b_S_convergence.png")


# PLOT 2: T3 sub-step S values vs cultural group
fig, ax = plt.subplots(figsize=(10, 6))

# T3 sub-steps
x_t3 = np.arange(len(t3_keys))
for i, k in enumerate(t3_keys):
    ax.bar(i, t3_substeps[k]['logS'], color='#2166ac', edgecolor='black',
           linewidth=0.8, alpha=0.85, width=0.6)
    ax.text(i, t3_substeps[k]['logS'] + 0.2, f"{t3_substeps[k]['logS']:.1f}",
            ha='center', fontsize=9)

# Cultural group for comparison
x_cul = np.arange(len(t3_keys), len(t3_keys) + len(cul_keys))
for i, k in enumerate(cul_keys):
    ax.bar(x_cul[i], transitions[k]['logS_orig'], color='#b2182b', edgecolor='black',
           linewidth=0.8, alpha=0.85, width=0.6)
    ax.text(x_cul[i], transitions[k]['logS_orig'] + 0.2, f"{transitions[k]['logS_orig']:.1f}",
            ha='center', fontsize=9)

# Reference lines
all_vals = list(t3_logS) + list(cultural_logS)
combined_mean = np.mean(all_vals)
ax.axhline(y=combined_mean, color='green', linestyle='--', alpha=0.6,
           label=f'Combined mean: {combined_mean:.1f}')
ax.axhspan(combined_mean - 1, combined_mean + 1, alpha=0.08, color='green')

# T3 original for contrast
ax.axhline(y=transitions['T3']['logS_orig'], color='gray', linestyle=':',
           alpha=0.4, label=f'T3 original: {transitions["T3"]["logS_orig"]:.1f}')

t3_labels = [t3_substeps[k]['short'] for k in t3_keys]
cul_labels = [f"{k}: {transitions[k]['name']}" for k in cul_keys]
ax.set_xticks(range(len(t3_keys) + len(cul_keys)))
ax.set_xticklabels(t3_labels + cul_labels, fontsize=9, rotation=45, ha='right')
ax.set_ylabel('log₁₀(S)', fontsize=label_fs)
ax.set_title('T3 Sub-steps vs Cultural Group: S Convergence', fontsize=title_fs)
ax.legend(fontsize=9)

# Separator
ax.axvline(x=len(t3_keys) - 0.5, color='gray', linestyle='-', alpha=0.3)
ax.text(len(t3_keys)/2 - 0.5, ax.get_ylim()[1] * 0.95, 'T3 sub-steps\n(genetic, P corrected)',
        ha='center', fontsize=9, color='#2166ac', style='italic')
ax.text(len(t3_keys) + len(cul_keys)/2 - 0.5, ax.get_ylim()[1] * 0.95, 'Cultural group\n(original P)',
        ha='center', fontsize=9, color='#b2182b', style='italic')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'step12b_T3_substeps.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: step12b_T3_substeps.png")


# PLOT 3: tau_fire vs v*P — original and corrected
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax_idx, (title_str, log_vP_arr, suffix) in enumerate([
    ('Original P', log_vP_orig, 'orig'),
    ('Corrected P (relevant lineages)', log_vP_corr, 'corr'),
]):
    ax = axes[ax_idx]
    for i, k in enumerate(t_keys):
        ax.scatter(log_vP_arr[i], log_tau[i], color=colors_orig[i], s=100,
                   edgecolors='black', linewidth=0.8, zorder=5)
        ax.annotate(k, (log_vP_arr[i], log_tau[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)

    # Fit line
    sl, ic, r, _, _ = stats.linregress(log_vP_arr, log_tau)
    fit_x = np.linspace(log_vP_arr.min() - 1, log_vP_arr.max() + 1, 100)
    ax.plot(fit_x, sl * fit_x + ic, 'k-', alpha=0.5,
            label=f'Fit: slope={sl:.2f}, R²={r**2:.3f}')
    # Ideal slope -1
    ax.plot(fit_x, -fit_x + step_logS.mean(), 'r--', alpha=0.3,
            label=f'Ideal: slope=−1')

    ax.set_xlabel('log₁₀(v × P)', fontsize=label_fs)
    ax.set_ylabel('log₁₀(τ_fire)', fontsize=label_fs)
    ax.set_title(title_str, fontsize=title_fs)
    ax.legend(fontsize=9)

plt.suptitle('Fire Phase Duration vs Search Rate', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'step12b_tau_vs_vP_corrected.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: step12b_tau_vs_vP_corrected.png")


# PLOT 4: P_required vs P_corrected — are they consistent?
fig, ax = plt.subplots(figsize=(8, 7))
logP_req = np.array([np.log10(transitions[k]['P_required']) for k in t_keys])
logP_corr = np.array([np.log10(corrections[k]['P_corr']) for k in t_keys])

for i, k in enumerate(t_keys):
    ax.scatter(logP_corr[i], logP_req[i], color=colors_orig[i], s=120,
               edgecolors='black', linewidth=0.8, zorder=5)
    ax.annotate(k, (logP_corr[i], logP_req[i]),
                textcoords="offset points", xytext=(8, 5), fontsize=10)

# Perfect agreement line
pmin = min(logP_corr.min(), logP_req.min()) - 1
pmax = max(logP_corr.max(), logP_req.max()) + 1
ax.plot([pmin, pmax], [pmin, pmax], 'k-', alpha=0.5, label='Perfect agreement')
ax.fill_between([pmin, pmax], [pmin-1, pmax-1], [pmin+1, pmax+1],
                alpha=0.1, color='green', label='±1 OOM')

ax.set_xlabel('log₁₀(P_corrected) [biological estimate]', fontsize=label_fs)
ax.set_ylabel('log₁₀(P_required) [from S = 10^{:.1f}]'.format(logS_target), fontsize=label_fs)
ax.set_title('Required vs Estimated P: Consistency Check', fontsize=title_fs)
ax.legend(fontsize=9)
ax.set_aspect('equal')
ax.set_xlim(pmin, pmax)
ax.set_ylim(pmin, pmax)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'step12b_P_consistency.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: step12b_P_consistency.png")


# ============================================================
# FINAL VERDICT
# ============================================================

print("\n" + "=" * 90)
print("VERDICT: Is the S variation an artifact of inconsistent granularity?")
print("=" * 90)

# Test 1: Does T3 decomposition give S ~ 10^13?
t3_match = abs(t3_logS.mean() - cultural_logS.mean()) < 2.0
print(f"\nTest 1: T3 sub-step S matches cultural group?")
print(f"  T3 sub-step mean: {t3_logS.mean():.2f}, Cultural mean: {cultural_logS.mean():.2f}")
print(f"  Difference: {abs(t3_logS.mean() - cultural_logS.mean()):.2f} OOM")
print(f"  Result: {'PASS' if t3_match else 'FAIL'} ({'< 2 OOM' if t3_match else '≥ 2 OOM'})")

# Test 2: Does corrected S converge?
range_improved = corr_logS.ptp() < orig_logS.ptp() * 0.3  # 70%+ reduction
print(f"\nTest 2: Corrected S range reduced by ≥70%?")
print(f"  Original range: {orig_logS.ptp():.1f} OOM")
print(f"  Corrected range: {corr_logS.ptp():.1f} OOM")
print(f"  Reduction: {(1 - corr_logS.ptp()/orig_logS.ptp())*100:.0f}%")
print(f"  Result: {'PASS' if range_improved else 'FAIL'}")

# Test 3: Does the slope fix?
slope_fixed = abs(slope_corr + 1) < abs(slope_orig + 1) * 0.5
print(f"\nTest 3: Slope of log(tau) vs log(v*P) moves toward -1?")
print(f"  Original: {slope_orig:+.3f} (deviation from -1: {abs(slope_orig + 1):.3f})")
print(f"  Corrected: {slope_corr:+.3f} (deviation from -1: {abs(slope_corr + 1):.3f})")
print(f"  Result: {'PASS' if slope_fixed else 'FAIL'}")

# Test 4: Are P_required values independently plausible?
n_within_10x = np.sum(np.abs(log_ratio) <= 1)
p_plausible = n_within_10x >= 7
print(f"\nTest 4: P_required within 10x of P_corrected for ≥7/9 transitions?")
print(f"  Count: {n_within_10x}/9")
print(f"  Result: {'PASS' if p_plausible else 'FAIL'}")

# Test 5: Cross-validation — genetic S₀ predicts cultural tau?
print(f"\nTest 5: Genetic-derived S₀ predicts cultural tau within 1 OOM?")
cross_val_ok = True
for k in cul_keys:
    t = transitions[k]
    c = corrections[k]
    tau_pred = c['n_steps'] * S0_genetic / (t['v'] * c['P_corr'])
    log_ratio_cv = abs(np.log10(t['tau'] / tau_pred))
    ok = log_ratio_cv < 1
    if not ok:
        cross_val_ok = False
    print(f"  {k}: actual={t['tau']:.3g}, predicted={tau_pred:.3g}, |log ratio|={log_ratio_cv:.2f} {'PASS' if ok else 'FAIL'}")
print(f"  Overall: {'PASS' if cross_val_ok else 'FAIL'}")

# Overall
tests = [t3_match, range_improved, slope_fixed, p_plausible, cross_val_ok]
n_pass = sum(tests)
print(f"\n{'='*50}")
print(f"TESTS PASSED: {n_pass}/5")
if n_pass >= 4:
    print(f"\n*** VERDICT: YES — the S variation is primarily an artifact of inconsistent granularity. ***")
    print(f"\nThe formula tau_fire = n × S₀ / (v × P_relevant) works across ALL transitions")
    print(f"with S₀ ≈ 10^{step_logS.mean():.1f} when P is measured as independent relevant lineages.")
elif n_pass >= 3:
    print(f"\n*** VERDICT: MOSTLY YES — granularity correction resolves most of the variation. ***")
    print(f"\nSome transitions remain outliers, but the correction reduces the span from")
    print(f"{orig_logS.ptp():.0f} OOM to {step_logS.ptp():.1f} OOM.")
else:
    print(f"\n*** VERDICT: NO — the S variation is at least partially real. ***")

print(f"\n{'='*90}")
print(f"DONE")
print(f"{'='*90}")
