"""
S28: Regulatory Depth Test — Statistical Analysis
Tests whether driver count from RSDB predicts Duration Amplification (D)
"""

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA: 19 biological regime shift types
# Driver counts from Rocha et al. 2015 bipartite matrix (Figshare 1472951)
# D values: 7 from framework anchors, 12 estimated from tau_reg/tau_collapse
# ============================================================================

data = {
    # name: (N_total, N_direct, D, D_lo, D_hi, S, tier, D_source)
    "Mangroves":             (22, 7,  750,  200, 2000, 1500,  3, "tau_reg=3000yr(peat)/tau_c=4yr(Cooper)"),
    "Kelp transitions":      (21, 4,   33,   10,  100, 2000, "A", "Anchor: kelp forest"),
    "Seagrass":              (20, 7,  300,  100, 1000, 1000,  3, "tau_reg=3000yr(Posidonia)/tau_c=10yr"),
    "Bivalves":              (18, 6,  150,   50,  500,  500,  3, "tau_reg=3000yr(oyster reefs)/tau_c=20yr"),
    "FW Eutrophication":    (17, 3,  200,  100, 1000, 1000, "A", "Anchor: temperate lake"),
    "Salt marshes":          (16, 5,  133,   50,  500, 1500, "A", "Anchor: salt marsh"),
    "Marine Eutrophication": (16, 2,  667,  200, 1667, 2000,  3, "tau_reg=2000yr(coastal)/tau_c=3yr(Cooper)"),
    "Coral transitions":     (15, 7, 1000, 1000,10000,15000, "A", "Anchor: coral reef"),
    "Fisheries":             (15, 2,  176,   60,  300, 3000,  3, "tau_reg=3000yr(shelf)/tau_c=17yr(Cooper)"),
    "Forest to Savannas":    (13, 9, 4600,  100, 5000,50000, "A", "Anchor: tropical forest (RATE RATIO)"),
    "Hypoxia":               (13, 4,  400,  100, 1000, 1000,  3, "tau_reg=2000yr(coastal)/tau_c=5yr"),
    "Floating plants":       (13, 3,  200,   50,  500, 1000,  3, "tau_reg=1000yr(FW)/tau_c=5yr"),
    "Bush encroachment":     (12, 3,  100,  100, 1000, 5000, "A", "Anchor: savanna"),
    "Drylands":              (10, 4,   50,   25,  200, 1000,  3, "tau_reg=5000yr/tau_c=100yr"),
    "Marine food webs":      (10, 2,  176,   75,  600, 3000,  3, "tau_reg=3000yr(shelf)/tau_c=17yr(Cooper)"),
    "Soil salinization":      (8, 4,   60,   20,  150,  500,  3, "tau_reg=3000yr/tau_c=50yr"),
    "Peatlands":              (6, 5,   80,   50,  200, 1000, "A", "Anchor: peatland"),
    "Tundra to forest":       (4, 3,   25,   10,  100,  500,  3, "tau_reg=5000yr(tundra)/tau_c=200yr"),
    "Steppe to tundra":       (3, 2,   10,    3,   30,  500,  3, "tau_reg=10000yr(steppe)/tau_c=1000yr"),
}

names = list(data.keys())
N_total = np.array([data[k][0] for k in names], dtype=float)
N_direct = np.array([data[k][1] for k in names], dtype=float)
D = np.array([data[k][2] for k in names], dtype=float)
D_lo = np.array([data[k][3] for k in names], dtype=float)
D_hi = np.array([data[k][4] for k in names], dtype=float)
S = np.array([data[k][5] for k in names], dtype=float)
tiers = [data[k][6] for k in names]

logD = np.log10(D)
logN = np.log10(N_total)
logS = np.log10(S)

N = len(names)

print("=" * 80)
print("S28: REGULATORY DEPTH TEST — RESULTS")
print("=" * 80)
print(f"\nN = {N} biological regime shift types")
print(f"  Anchor (framework D): {sum(1 for t in tiers if t == 'A')}")
print(f"  Tier 3 (estimated D): {sum(1 for t in tiers if t == 3)}")

# ============================================================================
# Print the dataset
# ============================================================================
print("\n" + "-" * 80)
print("FULL DATASET")
print("-" * 80)
print(f"{'Regime Shift':<25} {'N_tot':>5} {'N_dir':>5} {'D':>6} {'logD':>6} {'S':>6} {'Tier':>5}")
print("-" * 80)
for i, name in enumerate(names):
    tier_str = "Anch" if tiers[i] == "A" else "T3"
    print(f"{name:<25} {N_total[i]:>5.0f} {N_direct[i]:>5.0f} {D[i]:>6.0f} {logD[i]:>6.3f} {S[i]:>6.0f} {tier_str:>5}")

# ============================================================================
# MODEL A: log(D) = a + b * N_total (linear in driver count)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL A: log(D) = a + b × N_total")
print("=" * 80)

slope_a, intercept_a, r_a, p_a, se_a = stats.linregress(N_total, logD)
r2_a = r_a**2
predicted_a = intercept_a + slope_a * N_total
residuals_a = logD - predicted_a
rms_a = np.sqrt(np.mean(residuals_a**2))

rho_a, p_rho_a = stats.spearmanr(N_total, logD)

print(f"  Intercept: {intercept_a:.4f}")
print(f"  Slope:     {slope_a:.4f}")
print(f"  Pearson r: {r_a:.4f}")
print(f"  R²:        {r2_a:.4f}")
print(f"  Pearson p: {p_a:.4f}")
print(f"  Spearman ρ: {rho_a:.4f}")
print(f"  Spearman p: {p_rho_a:.4f}")
print(f"  RMS residual: {rms_a:.4f} dex")
print(f"  SE of slope: {se_a:.4f}")

# Also test with N_direct
slope_ad, intercept_ad, r_ad, p_ad, se_ad = stats.linregress(N_direct, logD)
rho_ad, p_rho_ad = stats.spearmanr(N_direct, logD)
print(f"\n  --- Using N_direct instead ---")
print(f"  Pearson r: {r_ad:.4f}, R²: {r_ad**2:.4f}, p: {p_ad:.4f}")
print(f"  Spearman ρ: {rho_ad:.4f}, p: {p_rho_ad:.4f}")

# Also test log-log: log(D) = a + b * log(N)
slope_all, intercept_all, r_all, p_all, se_all = stats.linregress(logN, logD)
rho_all, p_rho_all = stats.spearmanr(logN, logD)
print(f"\n  --- Log-log: log(D) = a + b × log(N_total) ---")
print(f"  Intercept: {intercept_all:.4f}")
print(f"  Slope:     {slope_all:.4f}")
print(f"  Pearson r: {r_all:.4f}, R²: {r_all**2:.4f}, p: {p_all:.4f}")
print(f"  Spearman ρ: {rho_all:.4f}, p: {p_rho_all:.4f}")

# ============================================================================
# MODEL B: log(D) = a + b * log(S) + c * N_total
# ============================================================================
print("\n" + "=" * 80)
print("MODEL B: log(D) = a + b × log(S) + c × N_total")
print("=" * 80)

# Multiple regression using numpy
X_b = np.column_stack([np.ones(N), logS, N_total])
beta_b, res_b, rank_b, sv_b = np.linalg.lstsq(X_b, logD, rcond=None)
predicted_b = X_b @ beta_b
ss_res_b = np.sum((logD - predicted_b)**2)
ss_tot = np.sum((logD - np.mean(logD))**2)
r2_b = 1 - ss_res_b / ss_tot
rms_b = np.sqrt(np.mean((logD - predicted_b)**2))

# Pearson r between predicted and observed
r_b = np.corrcoef(logD, predicted_b)[0, 1]

# Adjusted R²
r2_adj_b = 1 - (1 - r2_b) * (N - 1) / (N - 3)

# F-test for the model
f_stat_b = (r2_b / 2) / ((1 - r2_b) / (N - 3))
p_f_b = 1 - stats.f.cdf(f_stat_b, 2, N - 3)

print(f"  Intercept:  {beta_b[0]:.4f}")
print(f"  b (log S):  {beta_b[1]:.4f}")
print(f"  c (N_total): {beta_b[2]:.4f}")
print(f"  R²:          {r2_b:.4f}")
print(f"  Adjusted R²: {r2_adj_b:.4f}")
print(f"  r (obs vs pred): {r_b:.4f}")
print(f"  F-statistic: {f_stat_b:.3f}")
print(f"  F p-value:   {p_f_b:.4f}")
print(f"  RMS residual: {rms_b:.4f} dex")

# Test partial correlations
# Partial correlation of log(D) with N_total, controlling for log(S)
def partial_corr(x, y, z):
    """Partial correlation of x and y, controlling for z"""
    # Regress x on z, get residuals
    s1, i1, _, _, _ = stats.linregress(z, x)
    rx = x - (i1 + s1 * z)
    # Regress y on z, get residuals
    s2, i2, _, _, _ = stats.linregress(z, y)
    ry = y - (i2 + s2 * z)
    return stats.pearsonr(rx, ry)

r_partial_N, p_partial_N = partial_corr(logD, N_total, logS)
r_partial_S, p_partial_S = partial_corr(logD, logS, N_total)
print(f"\n  Partial r (D|N, controlling S): {r_partial_N:.4f}, p: {p_partial_N:.4f}")
print(f"  Partial r (D|S, controlling N): {r_partial_S:.4f}, p: {p_partial_S:.4f}")

# ============================================================================
# MODEL A': log(D) = a + b * log(S) (species richness alone, for comparison)
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON: log(D) = a + b × log(S) [species richness alone]")
print("=" * 80)

slope_s, intercept_s, r_s, p_s, se_s = stats.linregress(logS, logD)
rho_s, p_rho_s = stats.spearmanr(logS, logD)
predicted_s = intercept_s + slope_s * logS
rms_s = np.sqrt(np.mean((logD - predicted_s)**2))

print(f"  Intercept: {intercept_s:.4f}")
print(f"  Slope:     {slope_s:.4f}")
print(f"  Pearson r: {r_s:.4f}, R²: {r_s**2:.4f}, p: {p_s:.6f}")
print(f"  Spearman ρ: {rho_s:.4f}, p: {p_rho_s:.6f}")
print(f"  RMS residual: {rms_s:.4f} dex")

# ============================================================================
# MODEL C: D = Da * G^N (exponential amplification)
# We don't have Da estimates, so skip or use a simplified version
# log(D) = a + b * N_total (which is Model A, already done)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL C: log(D) = a + b × N_total [same as Model A — Da not available]")
print("=" * 80)
print("  Da estimates not available for most types.")
print("  Model C reduces to Model A without Da data.")
print("  See Model A results above.")

# ============================================================================
# MODEL D: log(D) = a + b * log(S) + c * log(N)  [log-log combined]
# ============================================================================
print("\n" + "=" * 80)
print("MODEL D: log(D) = a + b × log(S) + c × log(N_total)")
print("=" * 80)

X_d = np.column_stack([np.ones(N), logS, logN])
beta_d, res_d, rank_d, sv_d = np.linalg.lstsq(X_d, logD, rcond=None)
predicted_d = X_d @ beta_d
ss_res_d = np.sum((logD - predicted_d)**2)
r2_d = 1 - ss_res_d / ss_tot
rms_d = np.sqrt(np.mean((logD - predicted_d)**2))
r_d = np.corrcoef(logD, predicted_d)[0, 1]
r2_adj_d = 1 - (1 - r2_d) * (N - 1) / (N - 3)
f_stat_d = (r2_d / 2) / ((1 - r2_d) / (N - 3))
p_f_d = 1 - stats.f.cdf(f_stat_d, 2, N - 3)

print(f"  Intercept:  {beta_d[0]:.4f}")
print(f"  b (log S):  {beta_d[1]:.4f}")
print(f"  c (log N):  {beta_d[2]:.4f}")
print(f"  R²:          {r2_d:.4f}")
print(f"  Adjusted R²: {r2_adj_d:.4f}")
print(f"  r (obs vs pred): {r_d:.4f}")
print(f"  F-statistic: {f_stat_d:.3f}")
print(f"  F p-value:   {p_f_d:.4f}")
print(f"  RMS residual: {rms_d:.4f} dex")

# ============================================================================
# RESIDUAL ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("RESIDUAL ANALYSIS (Model A: log(D) = a + b × N_total)")
print("=" * 80)

sorted_idx = np.argsort(np.abs(residuals_a))[::-1]
print(f"\n  {'Regime Shift':<25} {'N':>4} {'logD_obs':>8} {'logD_pred':>9} {'Residual':>9}")
print("  " + "-" * 60)
for i in sorted_idx:
    print(f"  {names[i]:<25} {N_total[i]:>4.0f} {logD[i]:>8.3f} {predicted_a[i]:>9.3f} {residuals_a[i]:>+9.3f}")

# ============================================================================
# SENSITIVITY: Anchor-only subset (N=7)
# ============================================================================
print("\n" + "=" * 80)
print("SENSITIVITY: ANCHOR-ONLY (N=7)")
print("=" * 80)

anchor_mask = np.array([t == "A" for t in tiers])
N_anch = N_total[anchor_mask]
D_anch = logD[anchor_mask]
names_anch = [n for n, t in zip(names, tiers) if t == "A"]

if sum(anchor_mask) >= 5:
    slope_anch, intercept_anch, r_anch, p_anch, se_anch = stats.linregress(N_anch, D_anch)
    rho_anch, p_rho_anch = stats.spearmanr(N_anch, D_anch)
    print(f"  N = {sum(anchor_mask)}")
    print(f"  Pearson r: {r_anch:.4f}, R²: {r_anch**2:.4f}, p: {p_anch:.4f}")
    print(f"  Spearman ρ: {rho_anch:.4f}, p: {p_rho_anch:.4f}")
    for i, name in enumerate(names_anch):
        print(f"    {name}: N={N_anch[i]:.0f}, logD={D_anch[i]:.3f}")

# ============================================================================
# SENSITIVITY: Exclude tropical forest (rate ratio issue)
# ============================================================================
print("\n" + "=" * 80)
print("SENSITIVITY: EXCLUDE TROPICAL FOREST (rate ratio)")
print("=" * 80)

excl_mask = np.array([n != "Forest to Savannas" for n in names])
N_excl = N_total[excl_mask]
D_excl = logD[excl_mask]

slope_excl, intercept_excl, r_excl, p_excl, se_excl = stats.linregress(N_excl, D_excl)
rho_excl, p_rho_excl = stats.spearmanr(N_excl, D_excl)
print(f"  N = {sum(excl_mask)}")
print(f"  Pearson r: {r_excl:.4f}, R²: {r_excl**2:.4f}, p: {p_excl:.4f}")
print(f"  Spearman ρ: {rho_excl:.4f}, p: {p_rho_excl:.4f}")

# ============================================================================
# SENSITIVITY: Replace tropical forest D=4600 with D=200 (time-based estimate)
# ============================================================================
print("\n" + "=" * 80)
print("SENSITIVITY: TROPICAL FOREST D=200 (time-based instead of rate-based)")
print("=" * 80)

D_alt = D.copy()
logD_alt = logD.copy()
for i, n in enumerate(names):
    if n == "Forest to Savannas":
        D_alt[i] = 200
        logD_alt[i] = np.log10(200)

slope_alt, intercept_alt, r_alt, p_alt, se_alt = stats.linregress(N_total, logD_alt)
rho_alt, p_rho_alt = stats.spearmanr(N_total, logD_alt)
print(f"  N = {N}")
print(f"  Pearson r: {r_alt:.4f}, R²: {r_alt**2:.4f}, p: {p_alt:.4f}")
print(f"  Spearman ρ: {rho_alt:.4f}, p: {p_rho_alt:.4f}")

# S + N combined with alt D
X_alt = np.column_stack([np.ones(N), logS, N_total])
beta_alt, _, _, _ = np.linalg.lstsq(X_alt, logD_alt, rcond=None)
predicted_alt = X_alt @ beta_alt
ss_res_alt = np.sum((logD_alt - predicted_alt)**2)
ss_tot_alt = np.sum((logD_alt - np.mean(logD_alt))**2)
r2_alt = 1 - ss_res_alt / ss_tot_alt
print(f"  Model B (S+N) with alt D: R² = {r2_alt:.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF ALL MODELS")
print("=" * 80)
print(f"{'Model':<45} {'r':>6} {'R²':>6} {'ρ':>6} {'p(ρ)':>8} {'RMS':>6}")
print("-" * 80)
print(f"{'A: logD = a + b×N_total':<45} {r_a:>6.3f} {r2_a:>6.3f} {rho_a:>6.3f} {p_rho_a:>8.4f} {rms_a:>6.3f}")
print(f"{'A(log): logD = a + b×logN':<45} {r_all:>6.3f} {r_all**2:>6.3f} {rho_all:>6.3f} {p_rho_all:>8.4f} {'':>6}")
print(f"{'A(dir): logD = a + b×N_direct':<45} {r_ad:>6.3f} {r_ad**2:>6.3f} {rho_ad:>6.3f} {p_rho_ad:>8.4f} {'':>6}")
print(f"{'B: logD = a + b×logS + c×N':<45} {r_b:>6.3f} {r2_b:>6.3f} {'':>6} {p_f_b:>8.4f} {rms_b:>6.3f}")
print(f"{'D: logD = a + b×logS + c×logN':<45} {r_d:>6.3f} {r2_d:>6.3f} {'':>6} {p_f_d:>8.4f} {rms_d:>6.3f}")
print(f"{'S only: logD = a + b×logS':<45} {r_s:>6.3f} {r_s**2:>6.3f} {rho_s:>6.3f} {p_rho_s:>8.4f} {rms_s:>6.3f}")
print(f"{'Anchor-only (N=7): logD = a + b×N':<45} {r_anch:>6.3f} {r_anch**2:>6.3f} {rho_anch:>6.3f} {p_rho_anch:>8.4f} {'':>6}")

print("\n" + "=" * 80)
print("DECISION (pre-registered criteria)")
print("=" * 80)
if r2_a > 0.80 and N >= 20 and p_rho_a < 0.001:
    verdict_a = "STRONG SUPPORT"
elif r2_a > 0.60 and N >= 15 and p_rho_a < 0.01:
    verdict_a = "MODERATE SUPPORT"
else:
    verdict_a = "WEAK/NULL"
print(f"  Model A: {verdict_a} (R²={r2_a:.3f}, N={N}, p_ρ={p_rho_a:.4f})")

best_r2 = max(r2_a, r2_b, r2_d)
if best_r2 > 0.80 and N >= 20:
    verdict_best = "STRONG SUPPORT (best model)"
elif best_r2 > 0.60 and N >= 15:
    verdict_best = "MODERATE SUPPORT (best model)"
else:
    verdict_best = "WEAK/NULL (best model)"
print(f"  Best model: {verdict_best} (R²={best_r2:.3f})")

print(f"\n  Comparison: S + override (N=8) had R² = 0.951")
print(f"  This study: best R² = {best_r2:.3f} at N = {N}")

