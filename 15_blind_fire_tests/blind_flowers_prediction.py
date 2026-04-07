#!/usr/bin/env python3
"""
Study 15: Blind Fire Equation Test — Angiosperm Flowers

BLIND PREDICTION: Predict the fire-phase duration for the origin of
angiosperm flowers, using inputs entirely independent of the fire
equation's calibration set.

The key innovation: gamma comes from directed evolution experiments
(Keefe & Szostak 2001), not from the cusp bridge. This eliminates the
model-reduction-ratio problem.

tau_predicted = n * exp(gamma * d) / (v * P)

Inputs:
  gamma:  0.32 per parameter (Keefe & Szostak 2001, random 80-aa -> ATP binding)
          Also tested at 0.22 (cusp bridge) and 0.42 (protein mutagenesis)
  d:      51 parameters (van Mourik et al. 2010, full floral ODE model)
          Also tested at 37 (reduced model)
  v:      0.03-1.0/yr (seed plant generation time, range from paleobotany)
  P:      8-15 major seed-plant lineages (paleobotanical diversity)
  n:      4-6 sequential sub-steps (Dilcher developmental decomposition)

Observed: tau ~ 200 Myr (seed plants ~365 Ma -> angiosperms ~130 Ma)
          Narrower window: 170 Myr (MADS toolkit ~300 Ma -> 130 Ma)

References:
  Keefe & Szostak 2001, Nature 410:715 (gamma from directed evolution)
  Bartel & Szostak 1993, Science 261:1411 (RNA gamma comparison)
  Guo et al. 2004, PNAS 101:9205 (protein mutagenesis gamma)
  van Mourik et al. 2010, BMC Syst Biol 4:101 (floral ODE model)
  Dilcher 2000, PNAS / ResearchGate (angiosperm innovation steps)
  Friedman 2009, Am J Bot (early angiosperm generation time)
"""

import numpy as np

# ============================================================
# OBSERVED VALUE (the target)
# ============================================================

TAU_OBSERVED_BROAD = 235e6   # 365 Ma (seed plants) -> 130 Ma (angiosperms)
TAU_OBSERVED_NARROW = 170e6  # 300 Ma (MADS toolkit) -> 130 Ma (angiosperms)
TAU_OBSERVED_CENTRAL = 200e6 # Central estimate

print("=" * 90)
print("STUDY 15: BLIND FIRE EQUATION TEST — ANGIOSPERM FLOWERS")
print("=" * 90)
print(f"\nObserved fire phase (broad):   {TAU_OBSERVED_BROAD/1e6:.0f} Myr")
print(f"Observed fire phase (narrow):  {TAU_OBSERVED_NARROW/1e6:.0f} Myr")
print(f"Observed fire phase (central): {TAU_OBSERVED_CENTRAL/1e6:.0f} Myr")


# ============================================================
# INDEPENDENT INPUTS
# ============================================================

# gamma: search difficulty per parameter dimension
# From directed evolution experiments (NOT from the cusp bridge)
GAMMA_PROTEIN = 0.317       # Keefe & Szostak 2001: ln(10^11)/80
GAMMA_CUSP = 0.22           # Cusp bridge (framework, for comparison)
GAMMA_MUTAGENESIS = 0.42    # Guo et al. 2004: protein random mutagenesis
GAMMA_RNA = 0.136           # Bartel & Szostak 1993: ln(10^13)/220

gammas = {
    'Protein function (Keefe & Szostak)': GAMMA_PROTEIN,
    'Cusp bridge (theoretical)': GAMMA_CUSP,
    'Protein mutagenesis (Guo)': GAMMA_MUTAGENESIS,
    'RNA ribozyme (Bartel & Szostak)': GAMMA_RNA,
}

# d: parametric complexity from published ODE model
D_FULL = 51    # van Mourik et al. 2010, full model (13 vars, 51 params)
D_REDUCED = 37 # van Mourik et al. 2010, reduced model (6 vars, 37 params)

# v: compound trial rate (1 / generation time)
V_HERB_FAST = 1.0    # Annual herb, 1 yr generation
V_HERB_SLOW = 0.2    # Perennial herb, 5 yr generation
V_WOODY_MED = 0.05   # Moderate woody, 20 yr generation
V_WOODY_SLOW = 0.033 # Slow woody, 30 yr generation

vs = {
    'Annual herb (1 yr)': V_HERB_FAST,
    'Perennial herb (5 yr)': V_HERB_SLOW,
    'Moderate woody (20 yr)': V_WOODY_MED,
    'Slow woody (30 yr)': V_WOODY_SLOW,
}

# P: number of independent searching lineages
P_LOW = 8      # Conservative: major seed plant orders only
P_CENTRAL = 10 # Central estimate
P_HIGH = 15    # Including smaller lineages

# n: number of sequential innovation sub-steps
N_LOW = 4
N_CENTRAL = 6
N_HIGH = 6

print("\n" + "=" * 90)
print("INDEPENDENT INPUTS")
print("=" * 90)

print("\ngamma values (search difficulty per parameter):")
for name, g in gammas.items():
    print(f"  {name}: {g:.3f}")

print(f"\nd (ODE model parameters):")
print(f"  Full model:    {D_FULL}")
print(f"  Reduced model: {D_REDUCED}")

print(f"\nv (trial rate per year):")
for name, v in vs.items():
    print(f"  {name}: {v:.3f}")

print(f"\nP (searching lineages): {P_LOW}-{P_HIGH} (central: {P_CENTRAL})")
print(f"n (sub-steps): {N_LOW}-{N_HIGH} (central: {N_CENTRAL})")


# ============================================================
# PART 1: SEARCH DIFFICULTY S AT EACH GAMMA
# ============================================================

print("\n" + "=" * 90)
print("PART 1: SEARCH DIFFICULTY S = exp(gamma * d)")
print("=" * 90)

print(f"\n{'gamma source':<42} {'gamma':>6} {'d':>4} {'S':>14} {'log10(S)':>10}")
print("-" * 80)

for name, g in gammas.items():
    for d_label, d in [('full', D_FULL), ('reduced', D_REDUCED)]:
        S = np.exp(g * d)
        logS = np.log10(S)
        print(f"  {name:<38} {g:>6.3f} {d:>4} {S:>14.3g} {logS:>10.2f}")


# ============================================================
# PART 2: PRIMARY BLIND PREDICTION (gamma_protein, d_full)
# ============================================================

print("\n" + "=" * 90)
print("PART 2: PRIMARY BLIND PREDICTION")
print(f"  gamma = {GAMMA_PROTEIN:.3f} (Keefe & Szostak 2001)")
print(f"  d = {D_FULL} (van Mourik full model)")
print("=" * 90)

S_primary = np.exp(GAMMA_PROTEIN * D_FULL)
print(f"\nS = exp({GAMMA_PROTEIN:.3f} * {D_FULL}) = {S_primary:.4g}")
print(f"log10(S) = {np.log10(S_primary):.2f}")

print(f"\n{'v scenario':<28} {'v':>6} {'P':>4} {'n':>3} {'tau_pred (Myr)':>16} {'tau_obs':>10} {'log10(ratio)':>14} {'Within 1 OOM?':>14}")
print("-" * 100)

results = []
for v_name, v in vs.items():
    for P in [P_LOW, P_CENTRAL, P_HIGH]:
        for n in [N_LOW, N_CENTRAL]:
            tau_pred = n * S_primary / (v * P)
            tau_pred_myr = tau_pred / 1e6
            ratio = tau_pred / TAU_OBSERVED_CENTRAL
            log_ratio = np.log10(ratio)
            within_1 = "YES" if abs(log_ratio) < 1.0 else "no"
            results.append((v_name, v, P, n, tau_pred_myr, log_ratio, within_1))

            # Only print central P for brevity
            if P == P_CENTRAL and n == N_CENTRAL:
                print(f"  {v_name:<28} {v:>6.3f} {P:>4} {n:>3} {tau_pred_myr:>16.1f} {'200':>10} {log_ratio:>+14.2f} {within_1:>14}")

# Best match
best = min(results, key=lambda x: abs(x[5]))
print(f"\n  ** Best match: v={best[1]}, P={best[2]}, n={best[3]}")
print(f"     tau_predicted = {best[4]:.1f} Myr vs tau_observed = 200 Myr")
print(f"     log10(ratio) = {best[5]:+.2f} OOM")


# ============================================================
# PART 3: SENSITIVITY TO GAMMA
# ============================================================

print("\n" + "=" * 90)
print("PART 3: SENSITIVITY TO GAMMA (at central v/P/n)")
print(f"  v = {V_WOODY_MED}, P = {P_CENTRAL}, n = {N_CENTRAL}, d = {D_FULL}")
print("=" * 90)

v_test = V_WOODY_MED
P_test = P_CENTRAL
n_test = N_CENTRAL

print(f"\n{'gamma source':<42} {'gamma':>6} {'S':>14} {'tau (Myr)':>12} {'log10(ratio)':>14}")
print("-" * 92)

for name, g in gammas.items():
    S = np.exp(g * D_FULL)
    tau = n_test * S / (v_test * P_test)
    tau_myr = tau / 1e6
    log_ratio = np.log10(tau / TAU_OBSERVED_CENTRAL)
    marker = " <-- PRIMARY" if g == GAMMA_PROTEIN else ""
    print(f"  {name:<38} {g:>6.3f} {S:>14.3g} {tau_myr:>12.2g} {log_ratio:>+14.2f}{marker}")

print(f"\n  Observed: 200 Myr")
print(f"  The cusp bridge gamma = 0.22 sits between RNA (0.14) and protein (0.32)")


# ============================================================
# PART 4: FULL MODEL vs REDUCED MODEL
# ============================================================

print("\n" + "=" * 90)
print("PART 4: FULL vs REDUCED MODEL")
print(f"  gamma = {GAMMA_PROTEIN}, v = {V_WOODY_MED}, P = {P_CENTRAL}, n = {N_CENTRAL}")
print("=" * 90)

for d_label, d in [('Full (51)', D_FULL), ('Reduced (37)', D_REDUCED)]:
    S = np.exp(GAMMA_PROTEIN * d)
    tau = n_test * S / (v_test * P_test)
    tau_myr = tau / 1e6
    log_ratio = np.log10(tau / TAU_OBSERVED_CENTRAL)
    print(f"  {d_label}: S = {S:.3g}, tau = {tau_myr:.2g} Myr, log10(ratio) = {log_ratio:+.2f}")


# ============================================================
# PART 5: REVERSE CALCULATION — WHAT d WOULD MATCH tau_obs?
# ============================================================

print("\n" + "=" * 90)
print("PART 5: REVERSE CALCULATION — REQUIRED d")
print("=" * 90)

print(f"\nIf tau_observed = {TAU_OBSERVED_CENTRAL/1e6:.0f} Myr, what d is required?")
print(f"  S_required = tau * v * P / n")

for v_name, v in vs.items():
    S_req = TAU_OBSERVED_CENTRAL * v * P_CENTRAL / N_CENTRAL
    for g_name, g in [('gamma_protein', GAMMA_PROTEIN), ('gamma_cusp', GAMMA_CUSP)]:
        d_req = np.log(S_req) / g
        R_full = d_req / D_FULL
        R_reduced = d_req / D_REDUCED
        print(f"  {v_name:<24} {g_name:<16}: S = {S_req:.2g}, d = {d_req:.1f} "
              f"(R_full = {R_full:.2f}, R_reduced = {R_reduced:.2f})")


# ============================================================
# PART 6: CUSP BRIDGE VALIDATION FROM DIRECTED EVOLUTION
# ============================================================

print("\n" + "=" * 90)
print("PART 6: CUSP BRIDGE INDEPENDENT VALIDATION")
print("=" * 90)

print("""
The cusp bridge predicts gamma = 0.22 from pure bifurcation geometry.
Directed evolution experiments measure gamma INDEPENDENTLY:

  System                          d     S          gamma = ln(S)/d
  ─────────────────────────────── ───── ────────── ──────────────────
  Random 80-aa → ATP binding      80    10^11      0.317
  Random 220-nt → ribozyme       220    10^13      0.136
  Protein mutagenesis (avg)       per   (1-0.34)^d 0.418
  Cusp bridge (theoretical)       per   exp(0.22d) 0.220

The cusp bridge gamma = 0.22 falls inside the experimental range [0.14, 0.42].
Geometric mean of protein (0.32) and RNA (0.14): sqrt(0.32 * 0.14) = 0.21.
This closely matches the cusp bridge value (0.22).

INTERPRETATION: The cusp bridge gamma is independently validated by
laboratory search-difficulty measurements, to within a factor of ~1.5.
""")


# ============================================================
# PART 7: COMPARISON TO CALIBRATION SET
# ============================================================

print("=" * 90)
print("PART 7: WHERE FLOWERS SIT ON THE COMPLEXITY SCALE")
print("=" * 90)

# From step12e_intermediate_k.py (FIXED alpha = 0.373)
ALPHA = 0.373
LOG10_INV_ALPHA = np.log10(1.0 / ALPHA)

# Implied S for flowers at central parameters
S_flowers = TAU_OBSERVED_CENTRAL * V_WOODY_MED * P_CENTRAL / N_CENTRAL
k_flowers = np.log10(S_flowers) / LOG10_INV_ALPHA

calibration = [
    ('Flight (pterosaurs)', 7.8, 18.2),
    ('Flight (birds)', 8.5, 19.8),
    ('Flight (bats)', 8.6, 20.1),
    ('Flight (insects)', 9.9, 23.2),
    ('C4 photosynthesis', 10.4, 24.3),
    ('Camera-type eyes', 11.6, 27.1),
    ('Major transitions', 13.0, 30.4),
]

print(f"\n  {'Innovation':<28} {'log10(S)':>10} {'Implied k':>10}")
print("  " + "-" * 50)
for name, logS, k in calibration:
    marker = ""
    # Insert flowers in order
    if logS > np.log10(S_flowers) and name == calibration[0][0]:
        print(f"  {'>>> FLOWERS (blind) <<<':<28} {np.log10(S_flowers):>10.2f} {k_flowers:>10.1f}  ** THIS PREDICTION **")
    print(f"  {name:<28} {logS:>10.1f} {k:>10.1f}{marker}")
    if logS <= np.log10(S_flowers) and (calibration.index((name, logS, k)) + 1 >= len(calibration) or calibration[calibration.index((name, logS, k)) + 1][1] > np.log10(S_flowers)):
        print(f"  {'>>> FLOWERS (blind) <<<':<28} {np.log10(S_flowers):>10.2f} {k_flowers:>10.1f}  ** THIS PREDICTION **")

print(f"\n  Flowers: log10(S) = {np.log10(S_flowers):.2f}, k = {k_flowers:.1f}")
print(f"  (at v = {V_WOODY_MED}, P = {P_CENTRAL}, n = {N_CENTRAL})")


# ============================================================
# PART 8: VERDICT
# ============================================================

print("\n" + "=" * 90)
print("VERDICT")
print("=" * 90)

# Primary prediction
S_pred = np.exp(GAMMA_PROTEIN * D_FULL)
tau_central = N_CENTRAL * S_pred / (V_WOODY_MED * P_CENTRAL)
log_ratio_central = np.log10(tau_central / TAU_OBSERVED_CENTRAL)

# Range
tau_min = N_LOW * S_pred / (V_HERB_FAST * P_HIGH)
tau_max = N_HIGH * S_pred / (V_WOODY_SLOW * P_LOW)
log_ratio_min = np.log10(tau_min / TAU_OBSERVED_CENTRAL)
log_ratio_max = np.log10(tau_max / TAU_OBSERVED_CENTRAL)

observed_inside = tau_min <= TAU_OBSERVED_CENTRAL <= tau_max

print(f"""
PRIMARY PREDICTION (gamma_protein = {GAMMA_PROTEIN}, d = {D_FULL}):
  Central: tau = {tau_central/1e6:.1f} Myr (v=0.05, P=10, n=6)
  Range:   tau = {tau_min/1e6:.1f} to {tau_max/1e6:.1f} Myr
  Observed: {TAU_OBSERVED_CENTRAL/1e6:.0f} Myr

  Central prediction off by: {log_ratio_central:+.2f} OOM
  Observed inside predicted range: {'YES' if observed_inside else 'NO'}
  Range width: {np.log10(tau_max/tau_min):.2f} OOM

INTERPRETATION:
  The fire equation, using gamma from directed evolution (0.32) and d from
  a published floral ODE (51), predicts tau within {abs(log_ratio_central):.1f} OOM of observed
  at central biological estimates. The observed {TAU_OBSERVED_CENTRAL/1e6:.0f} Myr {'falls inside' if observed_inside else 'falls outside'}
  the full predicted range.

  This is the FIRST fire equation test where:
  (a) gamma is from laboratory measurement, not the cusp bridge
  (b) d is from an independently published ODE, not reverse-engineered from S
  (c) the innovation is NOT in the calibration set
  (d) all inputs come from independent sources

  The cusp bridge gamma = 0.22 predicts tau = {N_CENTRAL * np.exp(GAMMA_CUSP * D_FULL) / (V_WOODY_MED * P_CENTRAL) / 1e6:.1g} Myr at the same
  v/P/n, which is {np.log10(N_CENTRAL * np.exp(GAMMA_CUSP * D_FULL) / (V_WOODY_MED * P_CENTRAL) / TAU_OBSERVED_CENTRAL):+.1f} OOM from observed. The protein-derived gamma gives
  a better prediction, suggesting evolutionary search difficulty scales
  closer to the protein-function rate (0.32) than the generic cusp rate (0.22)
  for GRN innovations.
""")
