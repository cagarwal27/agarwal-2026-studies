"""
S29: Stabilizing Feedback Loop Count Test
Does the number of stabilizing feedback loops in causal loop diagrams predict D?

Data: Rocha et al. 2018 CLD edge list (RS_CLD_2018.csv)
Method: Enumerate all simple directed cycles per regime shift type,
        classify as stabilizing (sign product negative) vs reinforcing (positive),
        test stabilizing loop count against D.
"""

import os
import numpy as np
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Parse CLD edge list
# ============================================================================

edges_by_rs = defaultdict(list)  # rs_name -> [(tail, head, polarity)]

_data_path = os.path.join(os.path.dirname(__file__), 'data', 'RS_CLD_2018.csv')
with open(_data_path, 'r') as f:
    lines = f.read().replace('\r\n', '\n').replace('\r', '\n').split('\n')

header = lines[0]
for line in lines[1:]:
    if not line.strip():
        continue
    parts = line.split(';')
    if len(parts) < 7:
        continue
    tail = parts[1].strip()
    head = parts[2].strip()
    try:
        polarity = float(parts[3].strip())
    except ValueError:
        continue
    rs = parts[6].strip()
    edges_by_rs[rs].append((tail, head, polarity))

print(f"Parsed {sum(len(e) for e in edges_by_rs.values())} edges across {len(edges_by_rs)} RS types")

# ============================================================================
# STEP 2: Build signed digraphs and enumerate cycles
# ============================================================================

def find_all_simple_cycles(adj, nodes):
    """Johnson's algorithm simplified: find all simple directed cycles.
    adj: dict of node -> list of (neighbor, polarity)
    Returns list of (cycle_nodes, sign_product)
    """
    cycles = []
    node_list = sorted(nodes)
    
    def _dfs(start, current, visited, path, sign_prod):
        for neighbor, pol in adj.get(current, []):
            new_sign = sign_prod * (1 if pol > 0 else -1)
            if neighbor == start and len(path) > 1:
                cycles.append((list(path), new_sign))
            elif neighbor not in visited and node_list.index(neighbor) >= node_list.index(start):
                visited.add(neighbor)
                path.append(neighbor)
                _dfs(start, neighbor, visited, path, new_sign)
                path.pop()
                visited.remove(neighbor)
    
    for start in node_list:
        visited = {start}
        _dfs(start, start, visited, [start], 1.0)
    
    return cycles

results = {}  # rs_name -> {nodes, edges, total_cycles, stabilizing, reinforcing, stab_lengths, reinf_lengths}

for rs, edge_list in sorted(edges_by_rs.items()):
    # Build adjacency (binarize polarity: positive > 0 -> +1, negative -> -1)
    # Following Rocha: filter to |polarity| == 1 first, then also include binarized intermediates
    adj = defaultdict(list)
    nodes = set()
    seen_edges = set()
    
    for tail, head, pol in edge_list:
        # Binarize: any positive -> +1, any negative -> -1
        binary_pol = 1 if pol > 0 else -1
        edge_key = (tail, head)
        if edge_key not in seen_edges:  # deduplicate
            seen_edges.add(edge_key)
            adj[tail].append((head, binary_pol))
            nodes.add(tail)
            nodes.add(head)
    
    cycles = find_all_simple_cycles(adj, nodes)
    
    stabilizing = [(c, s) for c, s in cycles if s < 0]  # odd number of negative edges
    reinforcing = [(c, s) for c, s in cycles if s > 0]  # even number of negative edges
    
    stab_lengths = [len(c) for c, s in stabilizing]
    reinf_lengths = [len(c) for c, s in reinforcing]
    
    results[rs] = {
        'nodes': len(nodes),
        'edges': len(seen_edges),
        'total_cycles': len(cycles),
        'n_stab': len(stabilizing),
        'n_reinf': len(reinforcing),
        'stab_lengths': stab_lengths,
        'reinf_lengths': reinf_lengths,
        'stab_cycles': stabilizing,
    }

# ============================================================================
# STEP 3: Compute independent (non-overlapping) stabilizing loops
# ============================================================================

def count_independent_loops(stab_cycles, all_edges_set):
    """Greedy approximation of maximum independent (edge-disjoint) stabilizing cycles.
    Sort by length (shortest first), greedily pick cycles that don't share edges.
    """
    # Sort by cycle length (prefer shorter, more fundamental loops)
    sorted_cycles = sorted(stab_cycles, key=lambda x: len(x[0]))
    used_edges = set()
    independent = []
    
    for cycle_nodes, sign in sorted_cycles:
        # Get edges in this cycle
        cycle_edges = set()
        for i in range(len(cycle_nodes)):
            cycle_edges.add((cycle_nodes[i], cycle_nodes[(i+1) % len(cycle_nodes)]))
        
        # Check if any edge already used
        if not cycle_edges & used_edges:
            independent.append((cycle_nodes, sign))
            used_edges |= cycle_edges
    
    return len(independent)

for rs in results:
    r = results[rs]
    r['n_indep_stab'] = count_independent_loops(r['stab_cycles'], None)

# ============================================================================
# STEP 4: Print CLD summary for all 30 types
# ============================================================================

print("\n" + "=" * 90)
print("CLD CYCLE ANALYSIS — ALL 30 REGIME SHIFT TYPES")
print("=" * 90)
print(f"{'Regime Shift':<38} {'Nodes':>5} {'Edges':>5} {'Cycles':>6} {'Stab':>5} {'Reinf':>5} {'Indep':>5} {'Stab%':>5}")
print("-" * 90)
for rs in sorted(results.keys()):
    r = results[rs]
    pct = 100 * r['n_stab'] / r['total_cycles'] if r['total_cycles'] > 0 else 0
    print(f"{rs:<38} {r['nodes']:>5} {r['edges']:>5} {r['total_cycles']:>6} {r['n_stab']:>5} {r['n_reinf']:>5} {r['n_indep_stab']:>5} {pct:>5.0f}%")

# ============================================================================
# STEP 5: Map to D values (same dataset as S28)
# ============================================================================

# Name mapping: CLD name -> (D, D_source, S, tier)
cld_to_d = {
    "Mangroves transitions":    ("Mangroves",          750,  1500, 3, "tau_reg=3000/tau_c=4"),
    "Kelps transitions":        ("Kelp transitions",     33,  2000, "A", "Anchor: kelp forest"),
    "Seagrass transitions":     ("Seagrass",            300,  1000, 3, "tau_reg=3000/tau_c=10"),
    "Bivalves collapse":        ("Bivalves",            150,   500, 3, "tau_reg=3000/tau_c=20"),
    "Freshwater eutrophication":("FW Eutrophication",   200,  1000, "A", "Anchor: temperate lake"),
    "Salt marshes to tidal flats":("Salt marshes",      133,  1500, "A", "Anchor: salt marsh"),
    "Marine eutrophication":    ("Marine Eutr.",         667,  2000, 3, "tau_reg=2000/tau_c=3"),
    "Coral transitions":        ("Coral transitions",  1000, 15000, "A", "Anchor: coral reef"),
    "Fisheries collapse":       ("Fisheries",           176,  3000, 3, "tau_reg=3000/tau_c=17"),
    "Forest to savanna":        ("Forest→Savanna",     4600, 50000, "A", "Anchor: tropical forest (rate ratio)"),
    "Hypoxia":                  ("Hypoxia",             400,  1000, 3, "tau_reg=2000/tau_c=5"),
    "Floating plants":          ("Floating plants",     200,  1000, 3, "tau_reg=1000/tau_c=5"),
    "Bush encroachment":        ("Bush encroachment",   100,  5000, "A", "Anchor: savanna"),
    "Desertification":          ("Drylands",             50,  1000, 3, "tau_reg=5000/tau_c=100"),
    "Marine foodwebs":          ("Marine food webs",    176,  3000, 3, "tau_reg=3000/tau_c=17"),
    "Soil Salinization":        ("Soil salinization",    60,   500, 3, "tau_reg=3000/tau_c=50"),
    "Peatland transitions":     ("Peatlands",            80,  1000, "A", "Anchor: peatland"),
    "Tundra to forest":         ("Tundra→forest",        25,   500, 3, "tau_reg=5000/tau_c=200"),
    "Steppe to Tundra":         ("Steppe→tundra",        10,   500, 3, "tau_reg=10000/tau_c=1000"),
}

# Additional types from 2018 CLDs (not in S28, need D estimates)
cld_extras = {
    "Coniferous to deciduous forest": ("Boreal forest shift", 40, 1000, 3, "tau_reg=5000/tau_c=125(fire cycle)"),
    "Thermokarst lakes":              ("Thermokarst",         25,  500, 3, "tau_reg=5000/tau_c=200"),
    "Arctic Benthos Borealisation":   ("Arctic benthos",      50,  500, 3, "tau_reg=5000/tau_c=100"),
    "Primary production Arctic Ocean":("Arctic ocean prod.",  100,  500, 3, "tau_reg=5000/tau_c=50"),
}

# Non-biological types (excluded)
non_bio = {"Arctic Sea-Ice Loss", "Greenland Ice Sheet collapse", "Thermohaline circulation",
           "WAIS", "Moonson", "River channel change", "Sprawling vs compact city"}

# ============================================================================
# STEP 6: Build analysis dataset
# ============================================================================

print("\n" + "=" * 90)
print("S29 ANALYSIS DATASET — BIOLOGICAL TYPES")
print("=" * 90)

# Primary dataset: 19 S28 types
names = []
D_vals = []
logD = []
S_vals = []
logS = []
n_stab_all = []
n_stab_indep = []
n_reinf_all = []
n_total_cycles = []
n_drivers = []  # from S28
tiers = []

# S28 driver counts for reference
s28_drivers = {
    "Mangroves transitions": 22, "Kelps transitions": 21, "Seagrass transitions": 20,
    "Bivalves collapse": 18, "Freshwater eutrophication": 17, "Salt marshes to tidal flats": 16,
    "Marine eutrophication": 16, "Coral transitions": 15, "Fisheries collapse": 15,
    "Forest to savanna": 13, "Hypoxia": 13, "Floating plants": 13,
    "Bush encroachment": 12, "Desertification": 10, "Marine foodwebs": 10,
    "Soil Salinization": 8, "Peatland transitions": 6, "Tundra to forest": 4,
    "Steppe to Tundra": 3,
}

print(f"\n{'Name':<25} {'D':>6} {'logD':>6} {'S':>6} {'N_drv':>5} {'Stab':>5} {'Indep':>5} {'Reinf':>5} {'Tier':>5}")
print("-" * 90)

for cld_name in sorted(cld_to_d.keys()):
    if cld_name not in results:
        print(f"WARNING: {cld_name} not in CLD data!")
        continue
    
    short_name, d, s, tier, source = cld_to_d[cld_name]
    r = results[cld_name]
    n_drv = s28_drivers.get(cld_name, 0)
    
    names.append(short_name)
    D_vals.append(d)
    logD.append(np.log10(d))
    S_vals.append(s)
    logS.append(np.log10(s))
    n_stab_all.append(r['n_stab'])
    n_stab_indep.append(r['n_indep_stab'])
    n_reinf_all.append(r['n_reinf'])
    n_total_cycles.append(r['total_cycles'])
    n_drivers.append(n_drv)
    tiers.append(tier)
    
    tier_str = "Anch" if tier == "A" else "T3"
    print(f"{short_name:<25} {d:>6} {np.log10(d):>6.3f} {s:>6} {n_drv:>5} {r['n_stab']:>5} {r['n_indep_stab']:>5} {r['n_reinf']:>5} {tier_str:>5}")

N = len(names)
D_arr = np.array(D_vals, dtype=float)
logD_arr = np.array(logD)
logS_arr = np.array(logS)
stab_arr = np.array(n_stab_all, dtype=float)
indep_arr = np.array(n_stab_indep, dtype=float)
reinf_arr = np.array(n_reinf_all, dtype=float)
total_arr = np.array(n_total_cycles, dtype=float)
drv_arr = np.array(n_drivers, dtype=float)

# Derived variables
stab_frac = stab_arr / np.where(total_arr > 0, total_arr, 1)  # fraction stabilizing
log_stab = np.log10(np.where(stab_arr > 0, stab_arr, 0.5))  # log, with 0→0.5 for log safety

print(f"\nN = {N} biological regime shift types")

# ============================================================================
# STEP 7: STATISTICAL TESTS
# ============================================================================

print("\n" + "=" * 90)
print("MODEL RESULTS")
print("=" * 90)

def report_model(label, x, y, x_label="X"):
    slope, intercept, r, p, se = stats.linregress(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    predicted = intercept + slope * x
    rms = np.sqrt(np.mean((y - predicted)**2))
    print(f"\n  {label}")
    print(f"  log(D) = {intercept:.4f} + {slope:.4f} × {x_label}")
    print(f"  Pearson r = {r:.4f}, R² = {r**2:.4f}, p = {p:.4f}")
    print(f"  Spearman ρ = {rho:.4f}, p = {p_rho:.4f}")
    print(f"  RMS = {rms:.4f} dex")
    return r, r**2, rho, p_rho, rms, predicted

# --- S1: Stabilizing loop count (all) ---
print("\n--- STABILIZING LOOPS (ALL) ---")
r1, r2_1, rho1, prho1, rms1, pred1 = report_model(
    "S1: log(D) = a + b × N_stab", stab_arr, logD_arr, "N_stab")

# --- S2: Stabilizing loop count (independent/non-overlapping) ---
print("\n--- STABILIZING LOOPS (INDEPENDENT) ---")
r2, r2_2, rho2, prho2, rms2, pred2 = report_model(
    "S2: log(D) = a + b × N_indep_stab", indep_arr, logD_arr, "N_indep_stab")

# --- S3: Log stabilizing ---
print("\n--- LOG STABILIZING LOOPS ---")
# Only for stab > 0
mask_pos = stab_arr > 0
if sum(mask_pos) >= 10:
    r3, r2_3, rho3, prho3, rms3, pred3 = report_model(
        f"S3: log(D) = a + b × log(N_stab) [N={sum(mask_pos)}, excl zeros]",
        log_stab[mask_pos], logD_arr[mask_pos], "log(N_stab)")
else:
    print("  Too few non-zero values")

# --- S4: Stabilizing fraction ---
print("\n--- STABILIZING FRACTION ---")
r4, r2_4, rho4, prho4, rms4, pred4 = report_model(
    "S4: log(D) = a + b × (N_stab/N_total_cycles)", stab_frac, logD_arr, "stab_fraction")

# --- S5: Reinforcing loop count ---
print("\n--- REINFORCING LOOPS (control — should NOT predict D) ---")
r5, r2_5, rho5, prho5, rms5, pred5 = report_model(
    "S5: log(D) = a + b × N_reinf", reinf_arr, logD_arr, "N_reinf")

# --- S6: N_stab combined with log(S) ---
print("\n--- COMBINED MODELS ---")

def multiple_regression(X_cols, y, label):
    X = np.column_stack([np.ones(len(y))] + X_cols)
    beta, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    predicted = X @ beta
    ss_res = np.sum((y - predicted)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(X_cols) - 1)
    rms = np.sqrt(np.mean((y - predicted)**2))
    k = len(X_cols)
    f_stat = (r2 / k) / ((1 - r2) / (len(y) - k - 1))
    p_f = 1 - stats.f.cdf(f_stat, k, len(y) - k - 1)
    r_pred = np.corrcoef(y, predicted)[0, 1]
    print(f"\n  {label}")
    print(f"  Coefficients: {[f'{b:.4f}' for b in beta]}")
    print(f"  R² = {r2:.4f}, Adj R² = {r2_adj:.4f}")
    print(f"  F = {f_stat:.3f}, p = {p_f:.6f}")
    print(f"  r(obs,pred) = {r_pred:.4f}, RMS = {rms:.4f} dex")
    return r2, r2_adj, rms, beta

# S + stab
r2_ss, r2adj_ss, rms_ss, beta_ss = multiple_regression(
    [logS_arr, stab_arr], logD_arr, "M6: log(D) = a + b×log(S) + c×N_stab")

# S + indep_stab
r2_si, r2adj_si, rms_si, beta_si = multiple_regression(
    [logS_arr, indep_arr], logD_arr, "M7: log(D) = a + b×log(S) + c×N_indep_stab")

# S + stab + drivers (3-predictor)
r2_ssd, r2adj_ssd, rms_ssd, beta_ssd = multiple_regression(
    [logS_arr, stab_arr, drv_arr], logD_arr, "M8: log(D) = a + b×log(S) + c×N_stab + d×N_drivers")

# S28 comparison: S + drivers
r2_sd, r2adj_sd, rms_sd, beta_sd = multiple_regression(
    [logS_arr, drv_arr], logD_arr, "M_S28: log(D) = a + b×log(S) + c×N_drivers [S28 comparison]")

# S alone
slope_s, int_s, r_s, p_s, se_s = stats.linregress(logS_arr, logD_arr)
print(f"\n  S alone: log(D) = {int_s:.4f} + {slope_s:.4f} × log(S)")
print(f"  R² = {r_s**2:.4f}, p = {p_s:.6f}")

# ============================================================================
# STEP 8: Partial correlations
# ============================================================================

print("\n" + "=" * 90)
print("PARTIAL CORRELATIONS")
print("=" * 90)

def partial_corr(x, y, z):
    s1, i1, _, _, _ = stats.linregress(z, x)
    rx = x - (i1 + s1 * z)
    s2, i2, _, _, _ = stats.linregress(z, y)
    ry = y - (i2 + s2 * z)
    return stats.pearsonr(rx, ry)

r_stab_ctrl_s, p_stab_ctrl_s = partial_corr(logD_arr, stab_arr, logS_arr)
r_s_ctrl_stab, p_s_ctrl_stab = partial_corr(logD_arr, logS_arr, stab_arr)
r_drv_ctrl_s, p_drv_ctrl_s = partial_corr(logD_arr, drv_arr, logS_arr)
r_stab_ctrl_drv, p_stab_ctrl_drv = partial_corr(logD_arr, stab_arr, drv_arr)

print(f"  Partial r (D | N_stab, ctrl S):   {r_stab_ctrl_s:.4f}, p = {p_stab_ctrl_s:.4f}")
print(f"  Partial r (D | S, ctrl N_stab):    {r_s_ctrl_stab:.4f}, p = {p_s_ctrl_stab:.4f}")
print(f"  Partial r (D | N_drv, ctrl S):     {r_drv_ctrl_s:.4f}, p = {p_drv_ctrl_s:.4f}")
print(f"  Partial r (D | N_stab, ctrl N_drv):{r_stab_ctrl_drv:.4f}, p = {p_stab_ctrl_drv:.4f}")

# ============================================================================
# STEP 9: Anchor-only test (critical sensitivity)
# ============================================================================

print("\n" + "=" * 90)
print("ANCHOR-ONLY SENSITIVITY (N = 7)")
print("=" * 90)

anchor_mask = np.array([t == "A" for t in tiers])
n_anch = sum(anchor_mask)
if n_anch >= 5:
    stab_anch = stab_arr[anchor_mask]
    logD_anch = logD_arr[anchor_mask]
    names_anch = [n for n, t in zip(names, tiers) if t == "A"]
    
    slope_a, int_a, r_a, p_a, se_a = stats.linregress(stab_anch, logD_anch)
    rho_a, prho_a = stats.spearmanr(stab_anch, logD_anch)
    print(f"  N = {n_anch}")
    print(f"  Pearson r = {r_a:.4f}, R² = {r_a**2:.4f}, p = {p_a:.4f}")
    print(f"  Spearman ρ = {rho_a:.4f}, p = {prho_a:.4f}")
    for i, name in enumerate(names_anch):
        print(f"    {name}: N_stab={stab_anch[i]:.0f}, logD={logD_anch[i]:.3f}")

# ============================================================================
# STEP 10: Residual analysis (best model)
# ============================================================================

print("\n" + "=" * 90)
print("RESIDUAL ANALYSIS (S1: N_stab)")
print("=" * 90)

residuals = logD_arr - pred1
sorted_idx = np.argsort(np.abs(residuals))[::-1]
print(f"\n{'Name':<25} {'N_stab':>6} {'logD':>6} {'pred':>6} {'resid':>7}")
print("-" * 60)
for i in sorted_idx:
    print(f"{names[i]:<25} {stab_arr[i]:>6.0f} {logD_arr[i]:>6.3f} {pred1[i]:>6.3f} {residuals[i]:>+7.3f}")

# ============================================================================
# STEP 11: Summary comparison S28 vs S29
# ============================================================================

print("\n" + "=" * 90)
print("SUMMARY: S28 vs S29 COMPARISON")
print("=" * 90)

# S28 Model A: drivers alone
slope_d28, int_d28, r_d28, p_d28, _ = stats.linregress(drv_arr, logD_arr)
rho_d28, prho_d28 = stats.spearmanr(drv_arr, logD_arr)

print(f"\n{'Model':<50} {'r':>6} {'R²':>6} {'ρ':>6} {'p(ρ)':>8} {'RMS':>6}")
print("-" * 85)
print(f"{'S28: N_drivers alone':<50} {r_d28:>6.3f} {r_d28**2:>6.3f} {rho_d28:>6.3f} {prho_d28:>8.4f} {'':>6}")
print(f"{'S29: N_stab alone':<50} {r1:>6.3f} {r2_1:>6.3f} {rho1:>6.3f} {prho1:>8.4f} {rms1:>6.3f}")
print(f"{'S29: N_indep_stab alone':<50} {r2:>6.3f} {r2_2:>6.3f} {rho2:>6.3f} {prho2:>8.4f} {rms2:>6.3f}")
print(f"{'S29: stab_fraction alone':<50} {r4:>6.3f} {r2_4:>6.3f} {rho4:>6.3f} {prho4:>8.4f} {rms4:>6.3f}")
print(f"{'S29: N_reinf alone (CONTROL)':<50} {r5:>6.3f} {r2_5:>6.3f} {rho5:>6.3f} {prho5:>8.4f} {rms5:>6.3f}")
print(f"{'S alone: log(S)':<50} {r_s:>6.3f} {r_s**2:>6.3f} {'':>6} {'':>8} {'':>6}")
print(f"{'S28: S + N_drivers':<50} {'':>6} {r2_sd:>6.3f} {'':>6} {'':>8} {rms_sd:>6.3f}")
print(f"{'S29: S + N_stab':<50} {'':>6} {r2_ss:>6.3f} {'':>6} {'':>8} {rms_ss:>6.3f}")
print(f"{'S29: S + N_indep_stab':<50} {'':>6} {r2_si:>6.3f} {'':>6} {'':>8} {rms_si:>6.3f}")
print(f"{'S29: S + N_stab + N_drivers':<50} {'':>6} {r2_ssd:>6.3f} {'':>6} {'':>8} {rms_ssd:>6.3f}")
print(f"{'Prior: S + override (N=8)':<50} {'':>6} {'0.951':>6} {'':>6} {'':>8} {'':>6}")

# Anchor-only comparison
slope_d28a, int_d28a, r_d28a, p_d28a, _ = stats.linregress(drv_arr[anchor_mask], logD_arr[anchor_mask])
print(f"\n{'ANCHOR-ONLY (N=7):':<50}")
print(f"{'  S28: N_drivers':<50} {r_d28a:>6.3f} {r_d28a**2:>6.3f} {'':>6} {p_d28a:>8.4f}")
print(f"{'  S29: N_stab':<50} {r_a:>6.3f} {r_a**2:>6.3f} {'':>6} {p_a:>8.4f}")

# ============================================================================
# STEP 12: Decision
# ============================================================================

print("\n" + "=" * 90)
print("DECISION")
print("=" * 90)

if r2_1 > 0.80 and N >= 20 and prho1 < 0.001:
    verdict = "STRONG SUPPORT"
elif r2_1 > 0.60 and N >= 15 and prho1 < 0.01:
    verdict = "MODERATE SUPPORT"
elif r2_1 > 0.30 and prho1 < 0.05:
    verdict = "WEAK SUPPORT (significant but below threshold)"
else:
    verdict = "WEAK/NULL"

print(f"  N_stab alone: {verdict} (R²={r2_1:.3f}, N={N}, p_ρ={prho1:.4f})")

# Compare to S28
if r2_1 > r_d28**2:
    print(f"  N_stab (R²={r2_1:.3f}) OUTPERFORMS N_drivers (R²={r_d28**2:.3f})")
else:
    print(f"  N_stab (R²={r2_1:.3f}) does NOT outperform N_drivers (R²={r_d28**2:.3f})")

if r2_ss > r2_sd:
    print(f"  S + N_stab (R²={r2_ss:.3f}) OUTPERFORMS S + N_drivers (R²={r2_sd:.3f})")
else:
    print(f"  S + N_stab (R²={r2_ss:.3f}) does NOT outperform S + N_drivers (R²={r2_sd:.3f})")

