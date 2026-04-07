#!/usr/bin/env python3
"""
S23 Phase 2 Analysis — Trophic Coupling Test
Tests F2: D = (1/ε)^n using Cooper 2020 collapse data + Eddy 2021 TTE

Core method: Cooper's regression (log τ_collapse ~ log area) as null model,
then test whether residuals correlate with log[(1/ε)^n].
"""

import os
import math
import json

# ============================================================
# STEP 1: Master Data Table — 42 Cooper Ecosystems
# ============================================================
# Each ecosystem: (name, area_km2, tau_collapse_yr, epsilon, n, biome, ecosystem_type)
#
# Epsilon (ε) assignment rationale:
#   - Freshwater eutrophication: ε=0.12 (Eddy temperate 9.6%, boosted by high ε₁~50% from Cebrian phytoplankton)
#   - Freshwater fishery collapse: ε=0.10 (slightly lower; fishery shifts involve higher TL)
#   - Tropical freshwater: ε=0.10 (Eddy tropical ~8.6%, ε₁~50% → ~10%)
#   - Coral reef: ε=0.11 (Eddy tropical 8.6% + sponge loop enhancement per Eddy p.5-6)
#   - Temperate shelf/coastal: ε=0.10 (Eddy temperate mean 9.6%, phytoplankton ε₁)
#   - Temperate shelf fishery: ε=0.10, n=4.0 (higher TL targets)
#   - Polar marine: ε=0.12 (Eddy polar 12.0%)
#   - Subtropical: ε=0.09 (between temperate and tropical)
#   - Upwelling: ε=0.08 (Eddy upwelling 8.0%)
#   - Mangrove: ε=0.06 (Eddy tropical 8.6% pulled down by ε₁~3% from Cebrian)
#   - Kelp forest: ε=0.12 (Eddy temperate + macroalgal ε₁~30%)
#   - Terrestrial vegetation→desert: ε=0.05 (Cebrian forest/shrub ε₁~2%, ε₂~12%)
#   - Terrestrial savanna: ε=0.08 (Cebrian grassland ε₁~15%, ε₂~12%)
#
# n (effective trophic levels) rationale:
#   - Marine shelf/pelagic fishery: 4.0 (targets TL 3.5-4.5; Christensen Chesapeake max 4.26)
#   - Marine coastal: 3.5 (moderate food chains)
#   - Marine benthic/meiofauna: 3.0 (shorter chains)
#   - Coral reef: 3.5 (complex but moderate chain length)
#   - Freshwater lake: 3.5 (typical lake food chain)
#   - Freshwater diatom shift: 3.0 (lower trophic level shift)
#   - Upwelling: 3.0 (short chains, Eddy Table II mean TL 2.5)
#   - Polar: 4.0 (long chains through marine mammals)
#   - Mangrove: 3.0 (detritus-based, shorter)
#   - Kelp: 3.5 (urchin→otter chains)
#   - Terrestrial vegetation→desert: 2.5 (plant→herbivore→predator, short)
#   - Terrestrial savanna: 3.0 (grass→ungulate→predator)

ecosystems = [
    # Freshwater (13)
    # (id, name, area_km2, tau_yr, epsilon, n, biome_class)
    (1,  "Lake Erhai, China",                    250,     2,      0.12, 3.5, "FW_eutroph"),
    (2,  "Paul & Peter Lakes, USA",              0.020,   0.077,  0.10, 3.5, "FW_foodweb"),
    (3,  "Lake Veluwe, Netherlands",             30,      2,      0.12, 3.5, "FW_eutroph"),
    (4,  "Mwanza Gulf, Lake Victoria",           500,     8,      0.10, 3.5, "FW_foodweb"),
    (5,  "Lake Krankesjön, Sweden",              2.90,    3,      0.12, 3.5, "FW_eutroph"),
    (6,  "Lake Stechlin, Germany",               4.23,    0.030,  0.12, 3.5, "FW_eutroph"),
    (7,  "Old Danube Lake, Austria",             1.60,    5,      0.12, 3.5, "FW_eutroph"),
    (8,  "Lake Kariba, Zimbabwe/Zambia",         5400,    29,     0.10, 3.5, "FW_fishery"),
    (9,  "Lake of the Woods, Canada/USA",        4350,    35,     0.12, 3.0, "FW_diatom"),
    (10, "Foy Lake, USA",                        1.95,    5,      0.12, 3.0, "FW_diatom"),
    (11, "Lake Chilika, India",                  1000,    8,      0.10, 3.5, "FW_fishery"),
    (12, "Lac de Tunis, Tunisia",                48.6,    1,      0.10, 3.5, "FW_fishery"),
    (13, "Victoria Park Lake, Australia",        0.15,    0.29,   0.12, 3.5, "FW_eutroph"),
    # Marine (25)
    (14, "Jamaican coral reef",                  1000,    15,     0.11, 3.5, "M_coral"),
    (15, "Black Sea",                            436402,  40,     0.10, 4.0, "M_tempshelf"),
    (16, "N. Gulf of Alaska",                    1250,    17,     0.12, 4.0, "M_polar"),
    (17, "N. Gulf of Mexico",                    18000,   15,     0.09, 3.5, "M_subtropical"),
    (18, "Florida Bay, USA",                     40,      5,      0.09, 3.0, "M_subtropical"),
    (19, "Ringkøbing Fjord, Denmark",            300,     3,      0.10, 3.5, "M_tempcoast"),
    (20, "Frisian Front, Germany",               2880,    5,      0.10, 3.0, "M_tempbenthic"),
    (21, "Eastern Scotian Shelf",                108000,  5,      0.10, 4.0, "M_tempshelf"),
    (22, "Gulf of Trieste",                      600,     3,      0.10, 3.5, "M_tempcoast"),
    (23, "Central Baltic Sea",                   100000,  5,      0.10, 3.5, "M_tempshelf"),
    (24, "East Florida Mangroves",               20,      4,      0.06, 3.0, "M_mangrove"),
    (25, "Newfoundland, Canada",                 400000,  28,     0.10, 4.0, "M_tempshelf"),
    (26, "Northern Benguela",                    24000,   15,     0.08, 3.0, "M_upwelling"),
    (27, "Chesapeake Bay, USA",                  11600,   23,     0.10, 3.5, "M_tempshelf"),
    (28, "Aldabra atoll",                        115,     1,      0.11, 3.5, "M_coral"),
    (29, "Alamitos Bay, USA",                    820,     0.96,   0.10, 3.5, "M_tempcoast"),
    (30, "Kongsfjorden, Svalbard",               230,     2,      0.12, 4.0, "M_polar"),
    (31, "Izmit Bay, Turkey",                    310,     1,      0.10, 3.0, "M_tempbenthic"),
    (32, "Mariager Fjord, Denmark",              20,      0.04,   0.10, 3.0, "M_tempbenthic"),
    (33, "Gulf of Finland",                      5700,    2,      0.10, 3.5, "M_tempcoast"),
    (34, "Georges Bank, USA/Canada",             43000,   19,     0.10, 4.0, "M_tempshelf"),
    (35, "Nova Scotia kelp beds",                140,     6,      0.12, 3.5, "M_kelp"),
    (36, "Peruvian anchovy fishery",             14000,   14,     0.08, 3.0, "M_upwelling"),
    (37, "Limfjorden, Denmark",                  1500,    13,     0.10, 3.5, "M_tempcoast"),
    (38, "Osaka Bay ostracods, Japan",           1400,    50,     0.10, 3.5, "M_tempcoast"),
    # Terrestrial (4)
    (39, "The Sahel",                            9400000, 400,    0.05, 2.5, "T_desert"),
    (40, "Maradi agri-system, Niger",            35100,   20,     0.05, 2.5, "T_desert"),
    (41, "Zion National Park, Utah",             4500,    8,      0.05, 2.5, "T_desert"),
    (42, "Chobe National Park, Botswana",        11700,   15,     0.08, 3.0, "T_savanna"),
]

# ============================================================
# STEP 2: Compute derived quantities
# ============================================================
print("=" * 80)
print("S23 PHASE 2 — MASTER DATA TABLE")
print("=" * 80)

header = f"{'#':>2} {'Name':<35} {'Area':>10} {'τ_c':>7} {'ε':>5} {'n':>4} {'(1/ε)^n':>10} {'log₁₀A':>7} {'log₁₀τ':>7} {'log₁₀D_pred':>11}"
print(header)
print("-" * len(header))

data = []
for eco in ecosystems:
    id_, name, area, tau, eps, n, biome = eco
    D_pred = (1.0 / eps) ** n  # F2 prediction
    log_area = math.log10(area)
    log_tau = math.log10(tau)
    log_D_pred = math.log10(D_pred)
    data.append({
        'id': id_, 'name': name, 'area': area, 'tau': tau,
        'eps': eps, 'n': n, 'biome': biome,
        'D_pred': D_pred, 'log_area': log_area, 'log_tau': log_tau,
        'log_D_pred': log_D_pred
    })
    print(f"{id_:>2} {name:<35} {area:>10.1f} {tau:>7.3f} {eps:>5.2f} {n:>4.1f} {D_pred:>10.1f} {log_area:>7.3f} {log_tau:>7.3f} {log_D_pred:>11.3f}")

# ============================================================
# STEP 3: Cooper's Null Model Regression
# log₁₀(τ_collapse) = a + b × log₁₀(area)
# ============================================================
print("\n" + "=" * 80)
print("STEP 3: COOPER'S NULL MODEL — log₁₀(τ) = a + b × log₁₀(area)")
print("=" * 80)

N = len(data)
sum_x = sum(d['log_area'] for d in data)
sum_y = sum(d['log_tau'] for d in data)
sum_xy = sum(d['log_area'] * d['log_tau'] for d in data)
sum_x2 = sum(d['log_area'] ** 2 for d in data)
sum_y2 = sum(d['log_tau'] ** 2 for d in data)

mean_x = sum_x / N
mean_y = sum_y / N

# OLS regression
Sxx = sum_x2 - N * mean_x ** 2
Syy = sum_y2 - N * mean_y ** 2
Sxy = sum_xy - N * mean_x * mean_y

b_slope = Sxy / Sxx
a_intercept = mean_y - b_slope * mean_x
R2_null = (Sxy ** 2) / (Sxx * Syy)

print(f"\nN = {N}")
print(f"mean log₁₀(area) = {mean_x:.4f}")
print(f"mean log₁₀(τ)    = {mean_y:.4f}")
print(f"Sxx = {Sxx:.4f}")
print(f"Syy = {Syy:.4f}")
print(f"Sxy = {Sxy:.4f}")
print(f"\nSlope b     = {b_slope:.4f}")
print(f"Intercept a = {a_intercept:.4f}")
print(f"R²          = {R2_null:.4f}")
print(f"\nModel: log₁₀(τ) = {a_intercept:.4f} + {b_slope:.4f} × log₁₀(area)")
print(f"Cooper's reported: log₁₀(τ) = intercept + 0.221 × log₁₀(area), R² = 0.491")

# Compute residuals
print(f"\n{'#':>2} {'Name':<35} {'log₁₀τ_obs':>10} {'log₁₀τ_pred':>11} {'Residual':>9} {'log₁₀D_F2':>10}")
print("-" * 82)

for d in data:
    d['log_tau_pred'] = a_intercept + b_slope * d['log_area']
    d['residual'] = d['log_tau'] - d['log_tau_pred']
    print(f"{d['id']:>2} {d['name']:<35} {d['log_tau']:>10.4f} {d['log_tau_pred']:>11.4f} {d['residual']:>9.4f} {d['log_D_pred']:>10.4f}")

# ============================================================
# STEP 4: Core S23 Test — Residuals vs log[(1/ε)^n]
# ============================================================
print("\n" + "=" * 80)
print("STEP 4: CORE S23 TEST — Residuals vs log₁₀[(1/ε)^n]")
print("=" * 80)
print("\nHypothesis: Ecosystems with higher (1/ε)^n should have POSITIVE residuals")
print("(collapse slower than area alone predicts), because trophic structure buffers them.")
print()

# Correlation: residuals vs log_D_pred
sum_r = sum(d['residual'] for d in data)
sum_p = sum(d['log_D_pred'] for d in data)
sum_rp = sum(d['residual'] * d['log_D_pred'] for d in data)
sum_r2 = sum(d['residual'] ** 2 for d in data)
sum_p2 = sum(d['log_D_pred'] ** 2 for d in data)

mean_r = sum_r / N
mean_p = sum_p / N

Srr = sum_r2 - N * mean_r ** 2
Spp = sum_p2 - N * mean_p ** 2
Srp = sum_rp - N * mean_r * mean_p

r_pearson = Srp / math.sqrt(Srr * Spp) if Srr > 0 and Spp > 0 else 0
R2_test = r_pearson ** 2

# Regression of residual on log_D_pred
b_test = Srp / Spp if Spp > 0 else 0
a_test = mean_r - b_test * mean_p

# t-statistic for correlation
if abs(r_pearson) < 1.0:
    t_stat = r_pearson * math.sqrt((N - 2) / (1 - r_pearson ** 2))
else:
    t_stat = float('inf')

# Approximate p-value using t-distribution (large N approximation)
# For df=40, t=2.021 → p=0.05; t=2.704 → p=0.01
print(f"Pearson r (residual vs log₁₀[(1/ε)^n]) = {r_pearson:.4f}")
print(f"R² = {R2_test:.4f}")
print(f"Slope = {b_test:.4f}")
print(f"Intercept = {a_test:.4f}")
print(f"t-statistic = {t_stat:.4f} (df = {N-2})")
print(f"\nCritical values (df=40, two-tailed):")
print(f"  p < 0.05: |t| > 2.021")
print(f"  p < 0.01: |t| > 2.704")
print(f"  p < 0.001: |t| > 3.551")
print(f"\nResult: |t| = {abs(t_stat):.4f}", end="")
if abs(t_stat) > 3.551:
    print(" → p < 0.001 ***")
elif abs(t_stat) > 2.704:
    print(" → p < 0.01 **")
elif abs(t_stat) > 2.021:
    print(" → p < 0.05 *")
else:
    print(" → NOT significant at p=0.05")

# ============================================================
# STEP 4b: Spearman rank correlation (more robust)
# ============================================================
print("\n--- Spearman Rank Correlation ---")

def rank_data(values):
    """Compute ranks (handling ties by averaging)."""
    indexed = [(v, i) for i, v in enumerate(values)]
    indexed.sort(key=lambda x: x[0])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][0] == indexed[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based average rank
        for k in range(i, j):
            ranks[indexed[k][1]] = avg_rank
        i = j
    return ranks

residuals_list = [d['residual'] for d in data]
log_D_list = [d['log_D_pred'] for d in data]

ranks_r = rank_data(residuals_list)
ranks_p = rank_data(log_D_list)

# Pearson on ranks = Spearman
sum_rk_r = sum(ranks_r)
sum_rk_p = sum(ranks_p)
sum_rk_rp = sum(a * b for a, b in zip(ranks_r, ranks_p))
sum_rk_r2 = sum(a ** 2 for a in ranks_r)
sum_rk_p2 = sum(b ** 2 for b in ranks_p)

mean_rk_r = sum_rk_r / N
mean_rk_p = sum_rk_p / N

S_rr_rk = sum_rk_r2 - N * mean_rk_r ** 2
S_pp_rk = sum_rk_p2 - N * mean_rk_p ** 2
S_rp_rk = sum_rk_rp - N * mean_rk_r * mean_rk_p

rho_spearman = S_rp_rk / math.sqrt(S_rr_rk * S_pp_rk) if S_rr_rk > 0 and S_pp_rk > 0 else 0
t_spearman = rho_spearman * math.sqrt((N - 2) / (1 - rho_spearman ** 2)) if abs(rho_spearman) < 1 else float('inf')

print(f"Spearman ρ = {rho_spearman:.4f}")
print(f"t-statistic = {t_spearman:.4f} (df = {N-2})")
print(f"Result: |t| = {abs(t_spearman):.4f}", end="")
if abs(t_spearman) > 3.551:
    print(" → p < 0.001 ***")
elif abs(t_spearman) > 2.704:
    print(" → p < 0.01 **")
elif abs(t_spearman) > 2.021:
    print(" → p < 0.05 *")
else:
    print(" → NOT significant at p=0.05")

# ============================================================
# STEP 4c: Augmented model — add log[(1/ε)^n] as predictor
# ============================================================
print("\n--- Augmented Model: log₁₀(τ) = a + b₁×log₁₀(area) + b₂×log₁₀[(1/ε)^n] ---")

# Multiple regression using normal equations
# Y = a + b1*X1 + b2*X2
# where X1 = log_area, X2 = log_D_pred, Y = log_tau
X1 = [d['log_area'] for d in data]
X2 = [d['log_D_pred'] for d in data]
Y = [d['log_tau'] for d in data]

# Normal equations: [X'X]b = X'Y
s11 = sum(x1**2 for x1 in X1) - N * (sum(X1)/N)**2
s22 = sum(x2**2 for x2 in X2) - N * (sum(X2)/N)**2
s12 = sum(x1*x2 for x1, x2 in zip(X1, X2)) - N * (sum(X1)/N) * (sum(X2)/N)
s1y = sum(x1*y for x1, y in zip(X1, Y)) - N * (sum(X1)/N) * (sum(Y)/N)
s2y = sum(x2*y for x2, y in zip(X2, Y)) - N * (sum(X2)/N) * (sum(Y)/N)

det = s11*s22 - s12**2
b1_aug = (s22*s1y - s12*s2y) / det
b2_aug = (s11*s2y - s12*s1y) / det
a_aug = sum(Y)/N - b1_aug * sum(X1)/N - b2_aug * sum(X2)/N

# R² for augmented model
SS_tot = sum((y - sum(Y)/N)**2 for y in Y)
SS_res_aug = sum((Y[i] - a_aug - b1_aug*X1[i] - b2_aug*X2[i])**2 for i in range(N))
R2_aug = 1 - SS_res_aug / SS_tot

# R² for null model (area only)
SS_res_null = sum(d['residual']**2 for d in data)
R2_null_check = 1 - SS_res_null / SS_tot

# F-test for adding log_D_pred
# F = [(SS_res_null - SS_res_aug) / 1] / [SS_res_aug / (N - 3)]
F_stat = ((SS_res_null - SS_res_aug) / 1) / (SS_res_aug / (N - 3))

print(f"\nNull model:      log₁₀(τ) = {a_intercept:.4f} + {b_slope:.4f} × log₁₀(A)")
print(f"                 R² = {R2_null_check:.4f}, SS_res = {SS_res_null:.4f}")
print(f"\nAugmented model: log₁₀(τ) = {a_aug:.4f} + {b1_aug:.4f} × log₁₀(A) + {b2_aug:.4f} × log₁₀[(1/ε)^n]")
print(f"                 R² = {R2_aug:.4f}, SS_res = {SS_res_aug:.4f}")
print(f"\nΔR² = {R2_aug - R2_null_check:.4f}")
print(f"F-statistic for adding trophic term = {F_stat:.4f} (df₁=1, df₂={N-3})")
print(f"Critical F (1, {N-3}) at p=0.05 ≈ 4.08; p=0.01 ≈ 7.31; p=0.001 ≈ 12.6")
if F_stat > 12.6:
    print(f"Result: F = {F_stat:.2f} → p < 0.001 ***")
elif F_stat > 7.31:
    print(f"Result: F = {F_stat:.2f} → p < 0.01 **")
elif F_stat > 4.08:
    print(f"Result: F = {F_stat:.2f} → p < 0.05 *")
else:
    print(f"Result: F = {F_stat:.2f} → NOT significant at p=0.05")

# Sign of b2
print(f"\nSign of b₂ (trophic coefficient): {'+' if b2_aug > 0 else '-'}{abs(b2_aug):.4f}")
if b2_aug > 0:
    print("→ POSITIVE: Higher (1/ε)^n → longer collapse time (as predicted by F2)")
else:
    print("→ NEGATIVE: Higher (1/ε)^n → shorter collapse time (OPPOSITE of F2 prediction)")

# ============================================================
# STEP 5: Direct D_obs Test (subset with τ_regulated estimates)
# ============================================================
print("\n" + "=" * 80)
print("STEP 5: DIRECT F2 TEST — D_obs vs D_pred (subset with τ_regulated)")
print("=" * 80)
print("\nτ_regulated estimates from paleoecological/historical data:")
print("  Coral reefs: reef persistence ~5,000 yr (conservative; some >10,000)")
print("  Sahel vegetation: Holocene stable ~5,500 yr (post-African Humid Period)")
print("  Savanna: stable since mid-Holocene ~3,000 yr")
print("  Temperate lakes: post-glacial ~8,000 yr")
print("  Lake Kariba fishery: pre-Nile perch ~50 yr (short fishery)")
print()

direct_test = [
    # (id, name, tau_collapse, tau_regulated_est, eps, n, source_note)
    (14, "Jamaican coral reef",    15,   5000,  0.11, 3.5, "Caribbean reef age ~5000 yr"),
    (28, "Aldabra atoll",          1,    5000,  0.11, 3.5, "Atoll reef ~5000 yr"),
    (39, "The Sahel",              400,  5500,  0.05, 2.5, "Post-African Humid Period ~5500 yr"),
    (42, "Chobe NP savanna",       15,   3000,  0.08, 3.0, "Mid-Holocene savanna ~3000 yr"),
    (1,  "Lake Erhai",             2,    8000,  0.12, 3.5, "Post-glacial ~8000 yr"),
    (8,  "Lake Kariba (fishery)",  29,   50,    0.10, 3.5, "Pre-collapse fishery ~50 yr"),
    (9,  "Lake of the Woods",      35,   8000,  0.12, 3.0, "Post-glacial ~8000 yr"),
    (27, "Chesapeake Bay",         23,   4000,  0.10, 3.5, "Pre-colonial estuary ~4000 yr"),
]

print(f"{'#':>2} {'Name':<25} {'τ_c':>6} {'τ_r':>7} {'D_obs':>8} {'ε':>5} {'n':>4} {'D_F2':>10} {'log D_obs':>9} {'log D_F2':>9} {'Ratio':>7}")
print("-" * 100)

direct_log_obs = []
direct_log_pred = []

for dt in direct_test:
    id_, name, tau_c, tau_r, eps, n, note = dt
    D_obs = tau_r / tau_c
    D_pred = (1.0 / eps) ** n
    log_D_obs = math.log10(D_obs)
    log_D_pred = math.log10(D_pred)
    ratio = D_obs / D_pred
    direct_log_obs.append(log_D_obs)
    direct_log_pred.append(log_D_pred)
    print(f"{id_:>2} {name:<25} {tau_c:>6.1f} {tau_r:>7.0f} {D_obs:>8.1f} {eps:>5.2f} {n:>4.1f} {D_pred:>10.1f} {log_D_obs:>9.3f} {log_D_pred:>9.3f} {ratio:>7.3f}")

# Correlation on direct test
Nd = len(direct_test)
m_o = sum(direct_log_obs) / Nd
m_p = sum(direct_log_pred) / Nd
S_oo = sum((x - m_o)**2 for x in direct_log_obs)
S_pp_d = sum((x - m_p)**2 for x in direct_log_pred)
S_op = sum((a - m_o)*(b - m_p) for a, b in zip(direct_log_obs, direct_log_pred))
r_direct = S_op / math.sqrt(S_oo * S_pp_d) if S_oo > 0 and S_pp_d > 0 else 0
slope_direct = S_op / S_pp_d if S_pp_d > 0 else 0

print(f"\nDirect test (N={Nd}):")
print(f"  Pearson r (log D_obs vs log D_F2) = {r_direct:.4f}")
print(f"  R² = {r_direct**2:.4f}")
print(f"  Slope (log-log) = {slope_direct:.4f} (F2 predicts slope ≈ 1.0)")

# ============================================================
# STEP 6: Sensitivity Analysis — Eddy Low/Mean/High
# ============================================================
print("\n" + "=" * 80)
print("STEP 6: SENSITIVITY ANALYSIS")
print("=" * 80)

# Biome TTE ranges from Eddy 2021 Table I
eddy_ranges = {
    # biome: (low, mean, high)
    'polar':      (0.035, 0.120, 0.255),
    'temperate':  (0.019, 0.096, 0.344),
    'tropical':   (0.008, 0.086, 0.520),
    'upwelling':  (0.003, 0.080, 0.271),
}

# Map each ecosystem to its primary Eddy biome
biome_map = {
    'FW_eutroph':   'temperate',
    'FW_foodweb':   'temperate',
    'FW_fishery':   'temperate',
    'FW_diatom':    'temperate',
    'M_coral':      'tropical',
    'M_tempshelf':  'temperate',
    'M_polar':      'polar',
    'M_subtropical': 'tropical',
    'M_tempcoast':  'temperate',
    'M_tempbenthic':'temperate',
    'M_mangrove':   'tropical',
    'M_upwelling':  'upwelling',
    'M_kelp':       'temperate',
    'T_desert':     None,  # terrestrial — no Eddy biome, use Cebrian only
    'T_savanna':    None,
}

# For sensitivity: scale ε proportionally to Eddy range
# ε_sensitivity = ε_base × (eddy_variant / eddy_mean)
# For terrestrial (no Eddy biome): use ±50% as sensitivity range

def run_residual_test(data_modified, label):
    """Run the residual correlation test on modified data."""
    N = len(data_modified)
    X = [d['log_area'] for d in data_modified]
    Y = [d['log_tau'] for d in data_modified]
    P = [d['log_D_pred_sens'] for d in data_modified]

    mx = sum(X)/N
    my = sum(Y)/N
    sxx = sum(x**2 for x in X) - N*mx**2
    syy = sum(y**2 for y in Y) - N*my**2
    sxy = sum(x*y for x, y in zip(X, Y)) - N*mx*my

    b = sxy / sxx
    a = my - b * mx

    resids = [Y[i] - a - b*X[i] for i in range(N)]
    mr = sum(resids)/N
    mp = sum(P)/N
    srr = sum((r - mr)**2 for r in resids)
    spp = sum((p - mp)**2 for p in P)
    srp = sum((r - mr)*(p - mp) for r, p in zip(resids, P))

    r_val = srp / math.sqrt(srr * spp) if srr > 0 and spp > 0 else 0
    t_val = r_val * math.sqrt((N-2)/(1-r_val**2)) if abs(r_val) < 1 else float('inf')

    sig = "***" if abs(t_val) > 3.551 else "**" if abs(t_val) > 2.704 else "*" if abs(t_val) > 2.021 else "ns"
    return r_val, t_val, sig

print("\n--- 6a: Eddy Low/Mean/High TTE ---")
print(f"{'Scenario':<25} {'r':>8} {'t':>8} {'Sig':>5}")
print("-" * 50)

for scenario, eddy_key in [("Eddy Low TTE", "low"), ("Eddy Mean TTE", "mean"), ("Eddy High TTE", "high")]:
    for d in data:
        biome_eddy = biome_map.get(d['biome'])
        if biome_eddy and biome_eddy in eddy_ranges:
            low, mean, high = eddy_ranges[biome_eddy]
            if eddy_key == 'low':
                scale = low / mean
            elif eddy_key == 'high':
                scale = high / mean
            else:
                scale = 1.0
        else:
            # Terrestrial: ±50% for sensitivity
            if eddy_key == 'low':
                scale = 0.5
            elif eddy_key == 'high':
                scale = 1.5
            else:
                scale = 1.0
        eps_sens = d['eps'] * scale
        eps_sens = max(eps_sens, 0.005)  # floor at 0.5%
        eps_sens = min(eps_sens, 0.90)   # cap at 90%
        D_sens = (1.0 / eps_sens) ** d['n']
        d['log_D_pred_sens'] = math.log10(D_sens)

    r_val, t_val, sig = run_residual_test(data, scenario)
    print(f"{scenario:<25} {r_val:>8.4f} {t_val:>8.4f} {sig:>5}")

print("\n--- 6b: n ± 0.5 sensitivity ---")
print(f"{'Scenario':<25} {'r':>8} {'t':>8} {'Sig':>5}")
print("-" * 50)

for delta_n, label in [(-0.5, "n - 0.5"), (0, "n (base)"), (0.5, "n + 0.5")]:
    for d in data:
        n_sens = max(d['n'] + delta_n, 1.5)  # floor at 1.5
        D_sens = (1.0 / d['eps']) ** n_sens
        d['log_D_pred_sens'] = math.log10(D_sens)
    r_val, t_val, sig = run_residual_test(data, label)
    print(f"{label:<25} {r_val:>8.4f} {t_val:>8.4f} {sig:>5}")

# ============================================================
# STEP 6c: Subgroup analysis
# ============================================================
print("\n--- 6c: Subgroup Analysis ---")

subgroups = {
    'Marine only (N=25)': [d for d in data if d['biome'].startswith('M_')],
    'Freshwater only (N=13)': [d for d in data if d['biome'].startswith('FW_')],
    'Marine + Freshwater (N=38)': [d for d in data if not d['biome'].startswith('T_')],
}

print(f"{'Subgroup':<30} {'N':>3} {'r':>8} {'t':>8} {'Sig':>5}")
print("-" * 60)

for label, subset in subgroups.items():
    if len(subset) < 5:
        print(f"{label:<30} {len(subset):>3}   (too few for test)")
        continue
    for d in subset:
        d['log_D_pred_sens'] = d['log_D_pred']  # use base values
    r_val, t_val, sig = run_residual_test(subset, label)
    print(f"{label:<30} {len(subset):>3} {r_val:>8.4f} {t_val:>8.4f} {sig:>5}")

# Terrestrial has only 4 points — report descriptively
print(f"\nTerrestrial (N=4): Too few for regression. Descriptive:")
terr = [d for d in data if d['biome'].startswith('T_')]
for d in terr:
    print(f"  {d['name']}: residual = {d['residual']:+.4f}, log₁₀[(1/ε)^n] = {d['log_D_pred']:.3f}")

# ============================================================
# STEP 7: Summary Statistics
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
NULL MODEL (Cooper's regression):
  log₁₀(τ_collapse) = {a_intercept:.4f} + {b_slope:.4f} × log₁₀(area)
  R² = {R2_null:.4f}  (Cooper reported R² = 0.491, slope = 0.221)

CORE S23 TEST (residuals vs trophic structure):
  Pearson r  = {r_pearson:.4f}  (R² = {R2_test:.4f})
  Spearman ρ = {rho_spearman:.4f}
  Sign of b₂ = {'POSITIVE (as F2 predicts)' if b2_aug > 0 else 'NEGATIVE (opposite of F2)'}

AUGMENTED MODEL:
  log₁₀(τ) = {a_aug:.4f} + {b1_aug:.4f}×log₁₀(A) + {b2_aug:.4f}×log₁₀[(1/ε)^n]
  R² = {R2_aug:.4f}  (ΔR² = {R2_aug - R2_null:.4f})
  F-test = {F_stat:.2f}

DIRECT D TEST (N=8 with τ_regulated):
  r (log D_obs vs log D_F2) = {r_direct:.4f}
  log-log slope = {slope_direct:.4f} (F2 predicts 1.0)
""")

# Verdict
print("INTERPRETATION:")
if r_pearson > 0 and abs(t_stat) > 2.021:
    print("  ✓ Positive, significant correlation between residuals and trophic structure.")
    print("  ✓ Ecosystems with higher (1/ε)^n collapse SLOWER than area alone predicts.")
    print("  → SUPPORTS F2 hypothesis: trophic structure contributes to ecosystem persistence.")
elif r_pearson > 0:
    print("  ~ Positive but non-significant correlation.")
    print("  → INCONCLUSIVE: Direction consistent with F2 but insufficient statistical power.")
elif r_pearson < 0 and abs(t_stat) > 2.021:
    print("  ✗ Negative, significant correlation — OPPOSITE of F2 prediction.")
    print("  → REFUTES F2: Higher trophic complexity associated with FASTER collapse.")
else:
    print("  ✗ No significant correlation between residuals and trophic structure.")
    print("  → F2 NOT SUPPORTED by this test (though ε/n assignments may be too coarse).")

# ============================================================
# Output data as JSON for plotting
# ============================================================
output = {
    'null_model': {'intercept': a_intercept, 'slope': b_slope, 'R2': R2_null},
    'core_test': {'pearson_r': r_pearson, 'R2': R2_test, 'spearman_rho': rho_spearman, 't_stat': t_stat},
    'augmented': {'intercept': a_aug, 'b1_area': b1_aug, 'b2_trophic': b2_aug, 'R2': R2_aug, 'F_stat': F_stat},
    'direct_test': {'r': r_direct, 'slope': slope_direct, 'N': Nd},
    'data': [{k: v for k, v in d.items()} for d in data]
}

with open(os.path.join(os.path.dirname(__file__), 'phase2_data.json'), 'w') as f:
    json.dump(output, f, indent=2)

print("\nData saved to phase2_data.json")
