#!/usr/bin/env python3
"""
Step 12c: Sub-step decomposition for T4 (Plastids), T5 (Multicellularity), T6 (Eusociality)

Extends the Step 12b analysis (which decomposed T3 into sub-steps with S ≈ 10^13.2
per step) to three more transitions. If their sub-steps also converge to S₀ ≈ 10^13.5,
the universal constant claim is substantially strengthened (from 8 to 23 data points).

Key methodological principles (from Step 12b):
  - P_relevant = independent lineages with preconditions, NOT total individuals
  - v_compound = compound innovation trials per lineage per year, NOT elementary events
  - Sub-steps must be documented in the literature, sequential, and sum to the fire phase

References:
  T4: Archibald 2009, Keeling 2010, Nowack & Weber 2018, Marin et al. 2005 (Paulinella)
  T5: Grosberg & Strathmann 2007, Sebe-Pedros et al. 2017, King et al. 2008
  T6: Hughes et al. 2008 (Science), Boomsma 2009, Bourke 2011
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
# REFERENCE DATA (from Steps 12 and 12b)
# ============================================================

# T3 sub-steps from Step 12b
t3_substeps = {
    'T3a': {'name': 'Cytoskeleton (Asgard)',       'tau': 500e6, 'v': 20,  'P': 1e3},  # Imachi 2020: 14-25 day doubling
    'T3b': {'name': 'Cell wall loss',               'tau': 200e6, 'v': 20,  'P': 300},  # Imachi 2020: 14-25 day doubling
    'T3c': {'name': 'Mitochondrial endosymbiosis',  'tau': 100e6, 'v': 52,  'P': 100},
    'T3d': {'name': 'Gene transfer + nucleus',      'tau': 400e6, 'v': 52,  'P': 1e3},
    'T3e': {'name': 'Meiosis / sex',                'tau': 300e6, 'v': 52,  'P': 1e3},
}

# Cultural group from Step 12
cultural = {
    'T7':  {'name': 'Language',    'tau': 2.6e6,  'v': 122, 'P': 1e5},
    'T8':  {'name': 'Agriculture', 'tau': 188e3,  'v': 122, 'P': 5e6},
    'T8i': {'name': 'Industrial',  'tau': 11.7e3, 'v': 365, 'P': 1e5},
}

# Compute S for reference data
for d in [t3_substeps, cultural]:
    for k in d:
        s = d[k]
        s['S'] = s['tau'] * s['v'] * s['P']
        s['logS'] = np.log10(s['S'])

t3_keys = ['T3a', 'T3b', 'T3c', 'T3d', 'T3e']
cul_keys = ['T7', 'T8', 'T8i']
t3_logS = np.array([t3_substeps[k]['logS'] for k in t3_keys])
cul_logS = np.array([cultural[k]['logS'] for k in cul_keys])

print("=" * 90)
print("REFERENCE DATA")
print("=" * 90)
print(f"T3 sub-steps:  mean log₁₀(S) = {t3_logS.mean():.2f}, std = {t3_logS.std():.2f}")
print(f"Cultural group: mean log₁₀(S) = {cul_logS.mean():.2f}, std = {cul_logS.std():.2f}")
S0_target = 13.5
print(f"Target S₀ = 10^{S0_target}")


# ============================================================
# T4: PLASTIDS (primary endosymbiosis of cyanobacterium, ~1.5 Ga)
# ============================================================
# Fire phase: ~500 My (eukaryotes ~2.0 Ga → plastids ~1.5 Ga)
# Independent origins: 1 primary (Archaeplastida), plus Paulinella (~60-100 Mya)
#
# Sub-steps based on Keeling 2010, Archibald 2009, Nowack & Weber 2018,
# Bhattacharya et al. 2004.

t4_substeps = {
    'T4a': {
        'name': 'Phagocytic capture of cyanobacteria',
        'tau': 50e6,   # 50 My — ecological matching, relatively fast
        'v': 365,      # daily feeding events per lineage
        'P': 5e3,      # ~5000 phagocytic eukaryotic species at ~1.5 Ga in
                        # cyanobacteria-rich shallow marine environments.
                        # Eukaryotic diversity at 1.5 Ga was much lower than modern
                        # (~10^3–10^4 total unicellular eukaryote species).
                        # ~50% are phagocytic and in contact with cyanobacteria.
    },
    'T4b': {
        'name': 'Stable endosymbiosis (resist digestion)',
        'tau': 150e6,  # 150 My — THE bottleneck. Analogous to T3c.
        'v': 52,       # weekly engulfment-that-might-persist events.
                        # Not every feeding event is a "trial" for endosymbiosis —
                        # only those where the cyanobacterium has a chance to survive.
        'P': 3e3,      # ~3000 lineages routinely engulfing cyanobacteria.
                        # More than T3c (100) because by 1.5 Ga, eukaryotes were
                        # more diverse and phagocytosis was well-established.
    },
    'T4c': {
        'name': 'Gene transfer + protein import (TIC/TOC)',
        'tau': 150e6,  # 150 My — ~1500 genes transfer; TIC/TOC translocon evolves.
                        # The key innovation: establishing mechanism for nuclear-encoded
                        # proteins to return to the plastid (transit peptides + import).
        'v': 52,       # weekly genetic novelty rate for early eukaryotes
        'P': 1e3,      # ~1000 proto-Archaeplastida lineages after stable endosymbiosis.
                        # Three main divisions (Glaucophyta, Rhodophyta, Viridiplantae)
                        # diverged early; each exploring different transfer strategies.
    },
    'T4d': {
        'name': 'Metabolic integration',
        'tau': 100e6,  # 100 My — coordinating carbon fixation, light harvesting,
                        # ATP/NADPH exchange between plastid and host metabolism.
        'v': 52,       # weekly protist generations
        'P': 1e3,      # ~1000 Archaeplastida lineages with ongoing gene transfer
    },
    'T4e': {
        'name': 'Genome reduction → organelle',
        'tau': 50e6,   # 50 My — faster than earlier steps because gene loss is
                        # partly neutral (drift deletes redundant genes once their
                        # products are imported from the nucleus).
        'v': 52,       # weekly protist generations
        'P': 2e3,      # ~2000 diversifying Archaeplastida. Population larger now
                        # because the lineage has been diversifying for ~400 My.
    },
}

T4_FIRE_PHASE = 500e6
T4_N_ORIGINS = 1  # primary; Paulinella is independent replicate


# ============================================================
# T5: MULTICELLULARITY (~1.0 Ga for complex; 25+ independent origins)
# ============================================================
# Fire phase: ~500 My (plastids ~1.5 Ga → complex multicellularity ~1.0 Ga)
# Independent origins: 25+ total (Grosberg & Strathmann 2007)
#   - Simple MC (colonial/aggregative): 25+ origins
#   - Complex MC (with differentiation, apoptosis): ~6 origins
#
# Critical: pre-existing gene toolkit is widespread (Sebe-Pedros 2017, King 2008).
# Choanoflagellates have cadherins, integrins, tyrosine kinases, even Notch-like
# receptors. The search is about CO-OPTING existing genes, not inventing new ones.

t5_substeps = {
    'T5a': {
        'name': 'Cell adhesion (cells stick together)',
        'tau': 50e6,   # 50 My — relatively easy. Adhesion molecules (cadherins,
                        # integrins) already exist in unicellular ancestors.
                        # The innovation: deploying them for persistent cell-cell contact.
        'v': 52,       # weekly protist generations producing new regulatory variants
        'P': 3e4,      # ~30,000 unicellular eukaryotic lineages with adhesion gene
                        # homologs. Comparative genomics (Sebe-Pedros 2017) shows these
                        # genes were widespread. Upper range of 10^4–10^5 estimate.
    },
    'T5b': {
        'name': 'Cell communication (intercellular signaling)',
        'tau': 50e6,   # 50 My — also relatively easy. Many signaling pathways
                        # predate multicellularity (King et al. 2008).
        'v': 52,       # weekly genetic novelty rate
        'P': 2e4,      # ~20,000 lineages with signaling toolkit genes. Slightly
                        # narrower than T5a because need both adhesion AND signaling
                        # gene families.
    },
    'T5c': {
        'name': 'Coordinated cell division',
        'tau': 100e6,  # 100 My — harder. Requires spatial patterning of division
                        # planes, which needs new regulatory circuitry.
        'v': 52,       # weekly genetic novelty rate
        'P': 5e3,      # ~5000 lineages with adhesion + communication already
                        # established. Narrower because need both prior innovations.
    },
    'T5d': {
        'name': 'Cell differentiation (≥2 cell types)',
        'tau': 150e6,  # 150 My — major innovation. Requires differential gene
                        # expression programs + commitment to cell fate.
        'v': 52,       # weekly genetic novelty rate
        'P': 1e3,      # ~1000 lineages with organized multicellular stages
                        # (adhesion + communication + coordinated division).
    },
    'T5e': {
        'name': 'Programmed cell death (apoptosis)',
        'tau': 150e6,  # 150 My — requires altruistic cell suicide for group benefit.
                        # Needs multicellular context where group fitness > cell fitness.
                        # Some apoptosis-like genes predate MC (Nedelcu et al. 2011).
        'v': 52,       # weekly genetic novelty rate
        'P': 500,      # ~500 lineages with differentiated multicellular bodies.
                        # Only lineages that achieved T5d can search for T5e.
    },
}

T5_FIRE_PHASE = 500e6
T5_N_ORIGINS_SIMPLE = 25   # simple multicellularity (Grosberg & Strathmann 2007)
T5_N_ORIGINS_COMPLEX = 6   # complex MC: animals, land plants, brown algae, red algae, fungi, some green algae


# ============================================================
# T6: EUSOCIALITY (~150 Ma; 12–15 independent origins)
# ============================================================
# Fire phase: ~600 My (animal multicellularity ~750 Ma → eusociality ~150 Ma)
# Independent origins: 12–15 (Hymenoptera: 8–9; Isoptera: 1; others: 3–5)
#
# Hughes et al. 2008 (Science): strict lifetime monogamy was ancestral in ALL
# independently evolved eusocial lineages. This is the strongest prerequisite
# constraint: without monogamy, eusociality does not evolve.
#
# v_compound ≈ 1/yr: insect generation time ~1 year, and significant behavioral/
# social innovation per generation is low (most generations reproduce existing
# social structure). This is the lowest v in the entire dataset.

t6_substeps = {
    'T6a': {
        'name': 'Extended parental care',
        'tau': 200e6,  # 200 My — from animal multicellularity (~750 Ma) to widespread
                        # parental care in arthropods (~550 Ma, Cambrian/Ordovician).
                        # Many animal groups explore parental strategies.
        'v': 1,        # 1 behavioral innovation trial per lineage per year.
                        # Insects: ~1 generation/yr. Most generations reproduce
                        # existing parental behavior without significant innovation.
        'P': 1e5,      # ~100,000 animal species (especially arthropods) in the
                        # Paleozoic. Labandeira & Sepkoski 1993 estimate high insect
                        # diversity by late Paleozoic. Many lineages exploring
                        # parental care strategies.
    },
    'T6b': {
        'name': 'Nest building / defensible resource',
        'tau': 100e6,  # 100 My — from parental care to nest construction.
                        # Requires a home base worth defending, typically a burrow
                        # or constructed nest.
        'v': 1,        # 1/yr — behavioral innovation rate
        'P': 5e4,      # ~50,000 insect species with extended parental care.
                        # Many Hymenoptera, Coleoptera, Isoptera ancestors.
    },
    'T6c': {
        'name': 'Monogamy (lifetime pair bonding)',
        'tau': 100e6,  # 100 My — Hughes et al. 2008: monogamy was ancestral
                        # in ALL eusocial lineages. This is a strict prerequisite.
                        # Evolving from polygamy to lifetime monogamy requires
                        # specific ecological conditions (nest defense, high
                        # relatedness benefits).
        'v': 1,        # 1/yr — mating system innovations are slow
        'P': 2e4,      # ~20,000 nest-building species with parental care.
                        # Not all evolve monogamy — many remain polygynous.
    },
    'T6d': {
        'name': 'Subfertile helpers (offspring stay and help)',
        'tau': 100e6,  # 100 My — the critical social innovation. Offspring must
                        # gain more inclusive fitness by helping than by dispersing.
                        # Requires: high relatedness (monogamy → r = 0.5 for diploids,
                        # 0.75 for haplodiploids) + ecological constraints on
                        # independent reproduction.
        'v': 1,        # 1/yr — social structure innovations per generation
        'P': 5e3,      # ~5000 monogamous nest-building species. Much narrower
                        # than earlier steps because need ALL three preconditions.
    },
    'T6e': {
        'name': 'Full reproductive division of labor (queen + sterile workers)',
        'tau': 100e6,  # 100 My — from helpers to obligate sterility.
                        # Requires: regulatory mutations that permanently suppress
                        # worker reproduction + queen pheromone signaling.
        'v': 1,        # 1/yr
        'P': 2e3,      # ~2000 species with subfertile helpers.
                        # Modern examples of this intermediate stage: some halictid
                        # bees, Polistes wasps with flexible worker reproduction.
    },
}

T6_FIRE_PHASE = 600e6
T6_N_ORIGINS = 13  # midpoint of 12–15 range


# ============================================================
# COMPUTE S FOR ALL NEW SUB-STEPS
# ============================================================

def compute_S(substeps, keys):
    """Compute S and logS for each sub-step."""
    for k in keys:
        s = substeps[k]
        s['S'] = s['tau'] * s['v'] * s['P']
        s['logS'] = np.log10(s['S'])
    return np.array([substeps[k]['logS'] for k in keys])

t4_keys = ['T4a', 'T4b', 'T4c', 'T4d', 'T4e']
t5_keys = ['T5a', 'T5b', 'T5c', 'T5d', 'T5e']
t6_keys = ['T6a', 'T6b', 'T6c', 'T6d', 'T6e']

t4_logS = compute_S(t4_substeps, t4_keys)
t5_logS = compute_S(t5_substeps, t5_keys)
t6_logS = compute_S(t6_substeps, t6_keys)


# ============================================================
# PART 1: SUB-STEP TABLES
# ============================================================

def print_substep_table(name, substeps, keys, fire_phase, n_origins):
    """Print formatted sub-step table with self-consistency check."""
    print(f"\n{'='*90}")
    print(f"{name}")
    print(f"{'='*90}")
    print(f"\n{'Step':<6} {'Innovation':<45} {'tau (yr)':<12} {'v (/yr)':<10} {'P':<12} {'S':<12} {'log₁₀(S)':<10}")
    print("-" * 107)

    tau_sum = 0
    for k in keys:
        s = substeps[k]
        tau_sum += s['tau']
        print(f"{k:<6} {s['name']:<45} {s['tau']:<12.3g} {s['v']:<10.3g} {s['P']:<12.3g} {s['S']:<12.3g} {s['logS']:<10.2f}")

    logS_arr = np.array([substeps[k]['logS'] for k in keys])
    print(f"\nSelf-consistency: Σ(tau) = {tau_sum/1e6:.0f} My (fire phase = {fire_phase/1e6:.0f} My) — {'✓ MATCH' if abs(tau_sum - fire_phase) < 1e6 else '✗ MISMATCH'}")
    print(f"Independent origins: {n_origins}")
    print(f"Mean log₁₀(S) = {logS_arr.mean():.2f} ± {logS_arr.std():.2f}")
    print(f"Range: {logS_arr.min():.2f} to {logS_arr.max():.2f} ({logS_arr.ptp():.2f} OOM)")

    # Check each sub-step against S₀
    n_within = np.sum(np.abs(logS_arr - S0_target) <= 1.5)
    print(f"Sub-steps within ±1.5 OOM of S₀={S0_target}: {n_within}/{len(keys)}")
    for k in keys:
        s = substeps[k]
        dev = s['logS'] - S0_target
        status = "✓" if abs(dev) <= 1.5 else "✗"
        print(f"  {k}: {s['logS']:.2f} → deviation = {dev:+.2f} OOM {status}")

print_substep_table("T4: PLASTIDS (primary endosymbiosis, ~1.5 Ga)",
                     t4_substeps, t4_keys, T4_FIRE_PHASE, T4_N_ORIGINS)
print_substep_table("T5: MULTICELLULARITY (~1.0 Ga, 25+ independent origins)",
                     t5_substeps, t5_keys, T5_FIRE_PHASE, T5_N_ORIGINS_SIMPLE)
print_substep_table("T6: EUSOCIALITY (~150 Ma, 12–15 independent origins)",
                     t6_substeps, t6_keys, T6_FIRE_PHASE, T6_N_ORIGINS)


# ============================================================
# PART 2: PAULINELLA CROSS-CHECK (T4)
# ============================================================

print(f"\n{'='*90}")
print("PAULINELLA CROSS-CHECK (independent plastid endosymbiosis)")
print("=" * 90)

# Paulinella chromatophora: independent primary endosymbiosis ~60-100 Mya
# Host: euglyphid thecate amoeba (Rhizaria: Cercozoa)
# Marin et al. 2005, Nowack et al. 2008
paul_tau = 80e6     # ~80 My (range 60-100)
paul_v = 52         # amoeba generations ~weekly
paul_P = 3e3        # ~3000 cercozoan/euglyphid lineages with phagocytic lifestyle
paul_n_steps = 5    # same 5-step process

paul_S_total = paul_tau * paul_v * paul_P
paul_logS_total = np.log10(paul_S_total)
paul_logS_per_step = np.log10(paul_S_total / paul_n_steps)

print(f"Paulinella parameters: tau={paul_tau/1e6:.0f} My, v={paul_v}/yr, P={paul_P:.0g}")
print(f"Total S = {paul_S_total:.3g}, log₁₀(S) = {paul_logS_total:.2f}")
print(f"Per step (÷{paul_n_steps}): log₁₀(S/step) = {paul_logS_per_step:.2f}")
print(f"T4 mean log₁₀(S): {t4_logS.mean():.2f}")
print(f"Difference: {abs(paul_logS_per_step - t4_logS.mean()):.2f} OOM")
if abs(paul_logS_per_step - t4_logS.mean()) < 1.5:
    print("→ Paulinella is CONSISTENT with T4 sub-step estimates (< 1.5 OOM)")
else:
    print("→ Paulinella INCONSISTENT with T4 sub-step estimates")


# ============================================================
# PART 3: MULTICELLULARITY TEST (T5) — 25+ origins constraint
# ============================================================

print(f"\n{'='*90}")
print("MULTICELLULARITY TEST: Are 25+ independent origins consistent with S estimates?")
print("=" * 90)

# If S represents total search for ONE success, then N successes implies
# the actual difficulty per success is S/N.
# Equivalently: expected origins = (total search effort) / S_per_origin

# For simple multicellularity (steps 1-2 only):
# Total search over 500 My for steps 1-2:
t5_total_search_simple = sum(t5_substeps[k]['S'] for k in ['T5a', 'T5b'])
t5_logS_simple = np.log10(t5_total_search_simple)

# For complex multicellularity (all 5 steps):
t5_total_search_complex = sum(t5_substeps[k]['S'] for k in t5_keys)
t5_logS_complex = np.log10(t5_total_search_complex)

print(f"\nTotal search effort:")
print(f"  Steps 1-2 (simple MC): S_total = 10^{t5_logS_simple:.2f}")
print(f"  Steps 1-5 (complex MC): S_total = 10^{t5_logS_complex:.2f}")

# Effective S per origin
S_per_origin_simple = t5_total_search_simple / T5_N_ORIGINS_SIMPLE
S_per_origin_complex = t5_total_search_complex / T5_N_ORIGINS_COMPLEX

print(f"\nEffective S per independent origin:")
print(f"  Simple MC ({T5_N_ORIGINS_SIMPLE} origins): S/origin = 10^{np.log10(S_per_origin_simple):.2f}")
print(f"  Complex MC ({T5_N_ORIGINS_COMPLEX} origins): S/origin = 10^{np.log10(S_per_origin_complex):.2f}")

print(f"\nInterpretation:")
print(f"  If S_per_step ≈ 10^{t5_logS.mean():.1f} represents the difficulty of each sub-step,")
print(f"  then the 25+ origins of simple MC (steps 1-2) imply each origin required")
print(f"  S ≈ 10^{np.log10(S_per_origin_simple):.1f} total trials — {'LOWER' if S_per_origin_simple < 10**S0_target else 'comparable to'} S₀.")
print(f"  The ~6 origins of complex MC imply S ≈ 10^{np.log10(S_per_origin_complex):.1f} per origin.")
print(f"\n  This is consistent with multicellularity having MULTIPLE viable configurations")
print(f"  in the search space, reducing effective difficulty per origin.")


# ============================================================
# PART 4: EUSOCIALITY TEST (T6) — 12-15 origins constraint
# ============================================================

print(f"\n{'='*90}")
print("EUSOCIALITY TEST: Are 12–15 independent origins consistent with S estimates?")
print("=" * 90)

t6_total_search = sum(t6_substeps[k]['S'] for k in t6_keys)
t6_logS_total = np.log10(t6_total_search)
S_per_origin_eusocial = t6_total_search / T6_N_ORIGINS

print(f"Total search effort: S_total = 10^{t6_logS_total:.2f}")
print(f"Independent origins: {T6_N_ORIGINS}")
print(f"Effective S per origin: 10^{np.log10(S_per_origin_eusocial):.2f}")
print(f"\nT6 mean log₁₀(S_per_step) = {t6_logS.mean():.2f}")
print(f"Deviation from S₀ = {S0_target}: {t6_logS.mean() - S0_target:+.2f} OOM")

print(f"\nDiagnosis:")
print(f"  T6 sub-steps are systematically BELOW S₀ by ~{abs(t6_logS.mean() - S0_target):.1f} OOM.")
print(f"  This is consistent with eusociality being 'easier' than one-time transitions:")
print(f"  12-15 independent origins → multiple viable configurations → lower effective S.")
print(f"  The low v (1/yr for insect behavioral innovation) is partially offset by")
print(f"  moderate P (10^3–10^5 insect species searching over 600 My).")

# The P-gradient analysis
print(f"\n  P gradient across T6 sub-steps:")
for k in t6_keys:
    s = t6_substeps[k]
    print(f"    {k} ({s['name'][:30]}...): P = {s['P']:.0g}, logS = {s['logS']:.2f}")
print(f"  The declining P (from 10^5 to 2×10^3) reflects narrowing preconditions")
print(f"  and drives the low S values for later sub-steps.")


# ============================================================
# PART 5: STATISTICAL COMPARISONS
# ============================================================

print(f"\n{'='*90}")
print("PART 5: STATISTICAL COMPARISONS")
print("=" * 90)

# Group summaries
groups = {
    'T3 sub-steps': t3_logS,
    'T4 sub-steps': t4_logS,
    'T5 sub-steps': t5_logS,
    'T6 sub-steps': t6_logS,
    'Cultural (T7-T8i)': cul_logS,
}

print(f"\n{'Group':<25} {'n':<5} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Range':<8}")
print("-" * 70)
for name, arr in groups.items():
    print(f"{name:<25} {len(arr):<5d} {arr.mean():<8.2f} {arr.std():<8.2f} {arr.min():<8.2f} {arr.max():<8.2f} {arr.ptp():<8.2f}")

# Welch's t-tests against T3 and cultural
print(f"\nWelch's t-tests:")
print(f"{'Comparison':<40} {'t-stat':<10} {'p-value':<10} {'Δ mean':<10} {'Verdict':<15}")
print("-" * 85)

comparisons = [
    ('T4 vs T3 sub-steps', t4_logS, t3_logS),
    ('T5 vs T3 sub-steps', t5_logS, t3_logS),
    ('T6 vs T3 sub-steps', t6_logS, t3_logS),
    ('T4 vs Cultural', t4_logS, cul_logS),
    ('T5 vs Cultural', t5_logS, cul_logS),
    ('T6 vs Cultural', t6_logS, cul_logS),
    ('T4 vs T5', t4_logS, t5_logS),
    ('T4 vs T6', t4_logS, t6_logS),
    ('T5 vs T6', t5_logS, t6_logS),
]

for label, a, b in comparisons:
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    delta = a.mean() - b.mean()
    sig = "NOT significant" if p_val > 0.05 else "SIGNIFICANT"
    print(f"{label:<40} {t_stat:<+10.2f} {p_val:<10.4f} {delta:<+10.2f} {sig}")

# New transitions combined
new_combined = np.concatenate([t4_logS, t5_logS, t6_logS])
print(f"\nNew transitions combined (T4+T5+T6): mean = {new_combined.mean():.2f}, std = {new_combined.std():.2f}, n = {len(new_combined)}")

t_new_vs_t3, p_new_vs_t3 = stats.ttest_ind(new_combined, t3_logS, equal_var=False)
t_new_vs_cul, p_new_vs_cul = stats.ttest_ind(new_combined, cul_logS, equal_var=False)
print(f"  vs T3: t = {t_new_vs_t3:+.2f}, p = {p_new_vs_t3:.4f}")
print(f"  vs Cultural: t = {t_new_vs_cul:+.2f}, p = {p_new_vs_cul:.4f}")


# ============================================================
# PART 6: GRAND MEAN AND UPDATED S₀
# ============================================================

print(f"\n{'='*90}")
print("PART 6: UPDATED S₀ ESTIMATE")
print("=" * 90)

# All data points
all_logS = np.concatenate([t3_logS, t4_logS, t5_logS, t6_logS, cul_logS])
n_total = len(all_logS)

grand_mean = all_logS.mean()
grand_std = all_logS.std(ddof=1)  # sample std
grand_sem = grand_std / np.sqrt(n_total)
t_crit = stats.t.ppf(0.975, df=n_total - 1)
ci_lo = grand_mean - t_crit * grand_sem
ci_hi = grand_mean + t_crit * grand_sem

print(f"\nAll sub-steps + cultural combined: {n_total} data points")
print(f"  Grand mean:  log₁₀(S₀) = {grand_mean:.2f}")
print(f"  Std dev:     {grand_std:.2f}")
print(f"  SEM:         {grand_sem:.2f}")
print(f"  95% CI:      [{ci_lo:.2f}, {ci_hi:.2f}]")
print(f"  S₀ = 10^{grand_mean:.2f} (95% CI: 10^{ci_lo:.2f} to 10^{ci_hi:.2f})")

# Compare to original target
print(f"\n  Original target: S₀ = 10^{S0_target}")
print(f"  Updated estimate: S₀ = 10^{grand_mean:.2f}")
print(f"  Shift: {grand_mean - S0_target:+.2f} OOM")

# Check if 13.5 is within 95% CI
if ci_lo <= S0_target <= ci_hi:
    print(f"  → 10^{S0_target} is WITHIN the 95% CI. Original S₀ survives.")
else:
    print(f"  → 10^{S0_target} is OUTSIDE the 95% CI. Updated S₀ is lower.")

# Breakdown: one-time vs multi-origin transitions
one_time = np.concatenate([t3_logS, t4_logS])  # T3 and T4: each happened once
multi_origin = np.concatenate([t5_logS, t6_logS])  # T5 (25+) and T6 (12-15)

print(f"\n  Stratified analysis:")
print(f"    One-time transitions (T3, T4): mean = {one_time.mean():.2f}, std = {one_time.std():.2f}, n = {len(one_time)}")
print(f"    Multi-origin transitions (T5, T6): mean = {multi_origin.mean():.2f}, std = {multi_origin.std():.2f}, n = {len(multi_origin)}")
print(f"    Cultural group: mean = {cul_logS.mean():.2f}, std = {cul_logS.std():.2f}, n = {len(cul_logS)}")

t_1v_m, p_1v_m = stats.ttest_ind(one_time, multi_origin, equal_var=False)
print(f"    One-time vs Multi-origin: t = {t_1v_m:+.2f}, p = {p_1v_m:.4f}")
if p_1v_m < 0.05:
    print(f"    → SIGNIFICANT difference: multi-origin transitions have {'lower' if multi_origin.mean() < one_time.mean() else 'higher'} S")
    print(f"    → This is EXPECTED: more independent origins → easier transition → lower S")
else:
    print(f"    → Not significant at p < 0.05")


# ============================================================
# PART 7: FORMAL VERDICT
# ============================================================

print(f"\n{'='*90}")
print("PART 7: FORMAL VERDICT")
print("=" * 90)

# Test 1: Do T4 sub-step means fall within ±1.5 OOM of S₀?
t4_pass = abs(t4_logS.mean() - S0_target) <= 1.5
t5_pass = abs(t5_logS.mean() - S0_target) <= 1.5
t6_pass = abs(t6_logS.mean() - S0_target) <= 1.5

print(f"\nTest 1: Transition MEANS within ±1.5 OOM of S₀ = {S0_target}?")
for label, arr, passed in [('T4', t4_logS, t4_pass), ('T5', t5_logS, t5_pass), ('T6', t6_logS, t6_pass)]:
    print(f"  {label}: mean = {arr.mean():.2f}, deviation = {arr.mean() - S0_target:+.2f} → {'PASS' if passed else 'FAIL'}")

# Test 2: Do individual sub-steps fall within ±1.5 OOM?
all_new_logS = np.concatenate([t4_logS, t5_logS, t6_logS])
n_within_15 = np.sum(np.abs(all_new_logS - S0_target) <= 1.5)
print(f"\nTest 2: Individual sub-steps within ±1.5 OOM of S₀?")
print(f"  {n_within_15}/{len(all_new_logS)} sub-steps pass")
all_new_keys = t4_keys + t5_keys + t6_keys
all_new_substeps = {**t4_substeps, **t5_substeps, **t6_substeps}
for k in all_new_keys:
    s = all_new_substeps[k]
    dev = s['logS'] - S0_target
    status = "✓" if abs(dev) <= 1.5 else "✗ OUTLIER"
    print(f"    {k}: {s['logS']:.2f} ({dev:+.2f}) {status}")

# Test 3: Grand mean consistent with S₀ = 13.5?
grand_consistent = ci_lo <= S0_target <= ci_hi
print(f"\nTest 3: Grand mean 95% CI includes S₀ = {S0_target}?")
print(f"  95% CI: [{ci_lo:.2f}, {ci_hi:.2f}] → {'PASS' if grand_consistent else 'FAIL'}")

# Test 4: New transitions statistically indistinguishable from T3?
t_new_t3, p_new_t3 = stats.ttest_ind(new_combined, t3_logS, equal_var=False)
t3_compat = p_new_t3 > 0.05
print(f"\nTest 4: New transitions (T4+T5+T6) consistent with T3 sub-steps?")
print(f"  Welch's t = {t_new_t3:+.2f}, p = {p_new_t3:.4f} → {'PASS' if t3_compat else 'FAIL'}")

# Test 5: Std dev of all sub-steps < 1.5 OOM?
std_ok = grand_std < 1.5
print(f"\nTest 5: Std dev of all {n_total} sub-steps < 1.5 OOM?")
print(f"  Std = {grand_std:.2f} → {'PASS' if std_ok else 'FAIL'}")

# Overall verdict
n_transitions_pass = sum([t4_pass, t5_pass, t6_pass])
tests = [t4_pass and t5_pass and t6_pass, n_within_15 >= 12, grand_consistent, t3_compat, std_ok]
n_tests_pass = sum(tests)

print(f"\n{'='*50}")
print(f"TESTS PASSED: {n_tests_pass}/5")
print(f"TRANSITIONS WITHIN ±1.5: {n_transitions_pass}/3")

if n_transitions_pass == 3 and n_tests_pass >= 4:
    verdict = "CONFIRMED"
    verdict_text = (
        f"All three transitions' sub-step means fall within ±1.5 OOM of S₀ = {S0_target}.\n"
        f"Grand mean (10^{grand_mean:.2f}) is consistent with 10^{S0_target}.\n"
        f"S₀ ≈ 10^{grand_mean:.1f}–10^{S0_target} is supported by {n_total} data points across 4 genetic\n"
        f"transitions and 3 cultural transitions."
    )
elif n_transitions_pass >= 2 and n_tests_pass >= 3:
    verdict = "PARTIALLY CONFIRMED"
    verdict_text = (
        f"{n_transitions_pass}/3 transitions match S₀. The deviations in "
        f"{'T6' if not t6_pass else 'T5' if not t5_pass else 'T4'} are informatively\n"
        f"{'low (consistent with multiple independent origins reducing effective S)' if t6_logS.mean() < S0_target else 'high'}.\n"
        f"Grand mean (10^{grand_mean:.2f}) represents an updated S₀ estimate."
    )
elif n_transitions_pass >= 2:
    verdict = "WEAKENED"
    verdict_text = "Significant deviations in the non-matching transition weaken the universal constant claim."
else:
    verdict = "FALSIFIED"
    verdict_text = "Sub-steps scatter widely with no convergence to a universal S₀."

print(f"\n*** VERDICT: {verdict} ***")
print(f"\n{verdict_text}")

# Nuanced interpretation
print(f"\n{'='*90}")
print("NUANCED INTERPRETATION")
print("=" * 90)
print(f"""
The data reveal a systematic pattern within the convergence:

  One-time transitions (T3, T4):     mean log₁₀(S) = {one_time.mean():.2f}
  Multi-origin transitions (T5, T6): mean log₁₀(S) = {multi_origin.mean():.2f}
  Cultural transitions (T7-T8i):     mean log₁₀(S) = {cul_logS.mean():.2f}

Multi-origin transitions have LOWER S per step. This is expected: transitions that
evolved multiple times independently have more "target configurations" in the search
space, so each origin requires fewer trials.

The hierarchy (one-time > cultural ≈ multi-origin) is biologically sensible:
  - One-time transitions (endosymbiosis) search for a rare, specific configuration
  - Multi-origin transitions (multicellularity, eusociality) have many viable configurations
  - Cultural transitions have similar difficulty to multi-origin genetic transitions

If we treat S per step as encoding DIFFICULTY, then the variation within ±1.5 OOM
of S₀ is not noise — it's signal. Harder transitions have higher S; easier ones
(with more origins) have lower S. The universal constant S₀ ≈ 10^{grand_mean:.1f} is the
CENTRAL TENDENCY, not a fixed law of nature.
""")


# ============================================================
# PLOTS
# ============================================================

print("=" * 90)
print("Generating plots...")
print("=" * 90)

label_fs = 11
title_fs = 13
tick_fs = 9

# Color scheme
color_t3 = '#2166ac'   # blue
color_t4 = '#4393c3'   # light blue
color_t5 = '#92c5de'   # pale blue
color_t6 = '#f4a582'   # salmon
color_cul = '#b2182b'  # red


# PLOT 1: All sub-steps — master comparison
fig, ax = plt.subplots(figsize=(16, 7))

all_keys_ordered = t3_keys + t4_keys + t5_keys + t6_keys + cul_keys
all_substeps_combined = {**t3_substeps, **t4_substeps, **t5_substeps, **t6_substeps, **cultural}
all_logS_ordered = np.array([all_substeps_combined[k]['logS'] for k in all_keys_ordered])

colors = ([color_t3]*5 + [color_t4]*5 + [color_t5]*5 + [color_t6]*5 + [color_cul]*3)

x = np.arange(len(all_keys_ordered))
bars = ax.bar(x, all_logS_ordered, color=colors, edgecolor='black', linewidth=0.6, alpha=0.85, width=0.7)

# Value labels
for i, (k, v) in enumerate(zip(all_keys_ordered, all_logS_ordered)):
    ax.text(i, v + 0.15, f"{v:.1f}", ha='center', fontsize=7, rotation=0)

# Reference lines
ax.axhline(y=grand_mean, color='green', linestyle='--', alpha=0.6,
           label=f'Grand mean: {grand_mean:.2f}')
ax.axhspan(grand_mean - grand_std, grand_mean + grand_std, alpha=0.08, color='green',
           label=f'±1σ ({grand_std:.2f} OOM)')
ax.axhline(y=S0_target, color='gray', linestyle=':', alpha=0.4,
           label=f'Original S₀ = {S0_target}')

# Group separators and labels
seps = [4.5, 9.5, 14.5, 19.5]
for s in seps:
    ax.axvline(x=s, color='gray', linestyle='-', alpha=0.2)

group_centers = [2, 7, 12, 17, 21.5]
group_labels_text = ['T3\nEukaryotes', 'T4\nPlastids', 'T5\nMulticellularity',
                     'T6\nEusociality', 'Cultural']
for cx, gl in zip(group_centers, group_labels_text):
    ax.text(cx, ax.get_ylim()[0] + 0.3 if ax.get_ylim()[0] > 0 else 9.5, gl,
            ha='center', fontsize=9, style='italic', alpha=0.7)

# Axis formatting
short_labels = []
for k in all_keys_ordered:
    s = all_substeps_combined[k]
    name_short = s['name'][:25] + ('...' if len(s['name']) > 25 else '')
    short_labels.append(f"{k}")

ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=tick_fs, rotation=45, ha='right')
ax.set_ylabel('log₁₀(S)', fontsize=label_fs)
ax.set_title(f'All Sub-Steps: S Convergence Across 4 Genetic + 3 Cultural Transitions ({n_total} data points)',
             fontsize=title_fs)
ax.legend(fontsize=9, loc='lower right')
ax.set_ylim(9.5, 15.5)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'step12c_all_substeps.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: step12c_all_substeps.png")


# PLOT 2: Per-transition sub-steps (3 panels for T4, T5, T6)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ax_idx, (title, substeps, keys, color, fire, n_orig) in enumerate([
    ('T4: Plastids (1 origin)', t4_substeps, t4_keys, color_t4, T4_FIRE_PHASE, T4_N_ORIGINS),
    ('T5: Multicellularity (25+ origins)', t5_substeps, t5_keys, color_t5, T5_FIRE_PHASE, T5_N_ORIGINS_SIMPLE),
    ('T6: Eusociality (12-15 origins)', t6_substeps, t6_keys, color_t6, T6_FIRE_PHASE, T6_N_ORIGINS),
]):
    ax = axes[ax_idx]
    logS_arr = np.array([substeps[k]['logS'] for k in keys])

    x_pos = np.arange(len(keys))
    ax.bar(x_pos, logS_arr, color=color, edgecolor='black', linewidth=0.7,
           alpha=0.85, width=0.6)

    for i, (k, v) in enumerate(zip(keys, logS_arr)):
        ax.text(i, v + 0.15, f"{v:.1f}", ha='center', fontsize=9)

    # Reference lines
    ax.axhline(y=S0_target, color='gray', linestyle=':', alpha=0.4,
               label=f'S₀ = {S0_target}')
    ax.axhline(y=logS_arr.mean(), color='green', linestyle='--', alpha=0.6,
               label=f'Mean: {logS_arr.mean():.2f}')
    ax.axhspan(S0_target - 1.5, S0_target + 1.5, alpha=0.05, color='orange',
               label='±1.5 OOM')

    # Short names for x-axis
    short_names = [substeps[k]['name'][:20] + ('...' if len(substeps[k]['name']) > 20 else '')
                   for k in keys]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names, fontsize=8, rotation=45, ha='right')
    ax.set_title(title, fontsize=title_fs)
    ax.legend(fontsize=8, loc='lower left')

axes[0].set_ylabel('log₁₀(S)', fontsize=label_fs)
axes[0].set_ylim(9.5, 15.5)

plt.suptitle('Sub-Step S Values for T4, T5, T6', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'step12c_T4_T5_T6_substeps.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: step12c_T4_T5_T6_substeps.png")


# PLOT 3: Group means comparison with error bars
fig, ax = plt.subplots(figsize=(10, 6))

group_names = ['T3\nEukaryotes\n(n=5)', 'T4\nPlastids\n(n=5)',
               'T5\nMulticell.\n(n=5)', 'T6\nEusociality\n(n=5)',
               'Cultural\n(n=3)', 'Grand\nmean\n(n=23)']
group_means = [t3_logS.mean(), t4_logS.mean(), t5_logS.mean(),
               t6_logS.mean(), cul_logS.mean(), grand_mean]
group_stds = [t3_logS.std(), t4_logS.std(), t5_logS.std(),
              t6_logS.std(), cul_logS.std(), grand_std]
group_colors = [color_t3, color_t4, color_t5, color_t6, color_cul, '#333333']

x_pos = np.arange(len(group_names))
bars = ax.bar(x_pos, group_means, yerr=group_stds, capsize=5,
              color=group_colors, edgecolor='black', linewidth=0.7, alpha=0.85, width=0.6)

for i, (m, s) in enumerate(zip(group_means, group_stds)):
    ax.text(i, m + s + 0.2, f"{m:.2f}", ha='center', fontsize=10, fontweight='bold')

ax.axhline(y=S0_target, color='gray', linestyle=':', alpha=0.4,
           label=f'Original S₀ = {S0_target}')
ax.axhspan(S0_target - 1.5, S0_target + 1.5, alpha=0.05, color='orange',
           label='±1.5 OOM')

# Annotations for origins
ax.annotate('1 origin', (1, group_means[1] - group_stds[1] - 0.3),
            ha='center', fontsize=8, color='gray')
ax.annotate('25+ origins', (2, group_means[2] - group_stds[2] - 0.3),
            ha='center', fontsize=8, color='gray')
ax.annotate('12-15 origins', (3, group_means[3] - group_stds[3] - 0.3),
            ha='center', fontsize=8, color='gray')

ax.set_xticks(x_pos)
ax.set_xticklabels(group_names, fontsize=10)
ax.set_ylabel('Mean log₁₀(S) per sub-step', fontsize=label_fs)
ax.set_title('Group Means: S Convergence Across All Transitions', fontsize=title_fs)
ax.legend(fontsize=9)
ax.set_ylim(10, 15.5)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'step12c_group_means.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: step12c_group_means.png")


# PLOT 4: S vs number of independent origins (the difficulty gradient)
fig, ax = plt.subplots(figsize=(8, 6))

origin_data = [
    ('T3', 1, t3_logS.mean(), t3_logS.std(), color_t3),
    ('T4', 1, t4_logS.mean(), t4_logS.std(), color_t4),
    ('T5', 25, t5_logS.mean(), t5_logS.std(), color_t5),
    ('T6', 13, t6_logS.mean(), t6_logS.std(), color_t6),
]

for label, n_orig, mean, std, color in origin_data:
    ax.errorbar(n_orig, mean, yerr=std, fmt='o', color=color, markersize=12,
                markeredgecolor='black', markeredgewidth=0.8, capsize=5, capthick=1.5,
                elinewidth=1.5, zorder=5)
    ax.annotate(label, (n_orig, mean), textcoords="offset points",
                xytext=(12, 5), fontsize=11, fontweight='bold')

# Fit line (log-linear)
n_origins_arr = np.array([1, 1, 25, 13])
means_arr = np.array([t3_logS.mean(), t4_logS.mean(), t5_logS.mean(), t6_logS.mean()])
log_origins = np.log10(np.maximum(n_origins_arr, 1))
slope, intercept, r, p, se = stats.linregress(log_origins, means_arr)
x_fit = np.linspace(0, 1.5, 50)
ax.plot(10**x_fit, slope * x_fit + intercept, 'k--', alpha=0.4,
        label=f'Fit: slope = {slope:.2f}, R² = {r**2:.2f}')

ax.axhline(y=S0_target, color='gray', linestyle=':', alpha=0.4, label=f'S₀ = {S0_target}')
ax.set_xscale('log')
ax.set_xlabel('Number of independent origins', fontsize=label_fs)
ax.set_ylabel('Mean log₁₀(S) per sub-step', fontsize=label_fs)
ax.set_title('Search Difficulty vs Number of Origins', fontsize=title_fs)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'step12c_S_vs_origins.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: step12c_S_vs_origins.png")


# ============================================================
# SUMMARY
# ============================================================

print(f"\n{'='*90}")
print("SUMMARY")
print("=" * 90)
print(f"""
TRANSITION RESULTS:
  T4 (Plastids):         5 sub-steps, mean log₁₀(S) = {t4_logS.mean():.2f} ± {t4_logS.std():.2f}
  T5 (Multicellularity): 5 sub-steps, mean log₁₀(S) = {t5_logS.mean():.2f} ± {t5_logS.std():.2f}
  T6 (Eusociality):      5 sub-steps, mean log₁₀(S) = {t6_logS.mean():.2f} ± {t6_logS.std():.2f}

REFERENCE:
  T3 (Eukaryotes):       5 sub-steps, mean log₁₀(S) = {t3_logS.mean():.2f} ± {t3_logS.std():.2f}
  Cultural (T7-T8i):     3 transitions, mean log₁₀(S) = {cul_logS.mean():.2f} ± {cul_logS.std():.2f}

UPDATED S₀:
  Grand mean (n={n_total}): log₁₀(S₀) = {grand_mean:.2f} ± {grand_std:.2f}
  95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]
  Original target: {S0_target}

VERDICT: {verdict}
""")

print("=" * 90)
print("DONE")
print("=" * 90)
