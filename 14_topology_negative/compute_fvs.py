import os
"""
S30 Phase 3.1: Minimum Feedback Vertex Set Computation

Extends S29's cycle enumeration to compute FVS for all 19 biological regime
shift types using integer linear programming.

FVS = smallest set of nodes whose removal breaks ALL stabilizing cycles.
This is the graph-theoretic proxy for "codimension" / override resistance:
it measures how many independent regulatory nodes must be simultaneously
disabled to eliminate all stabilizing feedback.

Also computes: node connectivity, edge connectivity of stabilizing subnetwork,
and the "stressor channel count" (number of independent perturbation pathways).
"""

import numpy as np
from scipy import stats
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Parse CLD edge list (from S29)
# ============================================================================

edges_by_rs = defaultdict(list)

_data_path = os.path.join(os.path.dirname(__file__), 'data', 'RS_CLD_2018.csv')
with open(_data_path, 'r') as f:
    lines = f.read().replace('\r\n', '\n').replace('\r', '\n').split('\n')

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
# STEP 2: Build digraphs and enumerate stabilizing cycles (from S29)
# ============================================================================

def find_all_simple_cycles(adj, nodes):
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

results = {}

for rs, edge_list in sorted(edges_by_rs.items()):
    adj = defaultdict(list)
    nodes = set()
    seen_edges = set()

    for tail, head, pol in edge_list:
        binary_pol = 1 if pol > 0 else -1
        edge_key = (tail, head)
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            adj[tail].append((head, binary_pol))
            nodes.add(tail)
            nodes.add(head)

    cycles = find_all_simple_cycles(adj, nodes)
    stabilizing = [(c, s) for c, s in cycles if s < 0]
    reinforcing = [(c, s) for c, s in cycles if s > 0]

    results[rs] = {
        'nodes': nodes,
        'n_nodes': len(nodes),
        'edges': seen_edges,
        'n_edges': len(seen_edges),
        'adj': dict(adj),
        'total_cycles': len(cycles),
        'n_stab': len(stabilizing),
        'n_reinf': len(reinforcing),
        'stab_cycles': stabilizing,
        'reinf_cycles': reinforcing,
    }

# ============================================================================
# STEP 3: Compute MINIMUM FEEDBACK VERTEX SET (FVS) for stabilizing cycles
# ============================================================================

def compute_fvs(stab_cycles, all_nodes):
    """
    Compute minimum feedback vertex set via integer linear programming.

    Minimize: sum(x_i)
    Subject to: for each stabilizing cycle C, sum(x_i for i in C) >= 1
    x_i in {0, 1}

    The solution gives the smallest set of nodes whose removal breaks
    ALL stabilizing cycles.
    """
    if not stab_cycles:
        return set(), 0

    node_list = sorted(all_nodes)
    n = len(node_list)
    node_idx = {node: i for i, node in enumerate(node_list)}

    # Objective: minimize sum of x_i
    c = np.ones(n)

    # Constraints: for each cycle, at least one node must be in FVS
    A_rows = []
    b_rows = []
    for cycle_nodes, sign in stab_cycles:
        row = np.zeros(n)
        for node in cycle_nodes:
            row[node_idx[node]] = 1
        A_rows.append(row)
        b_rows.append(1)

    if not A_rows:
        return set(), 0

    A_ub = -np.array(A_rows)  # negate for <= constraint (we want >= 1, so -A x <= -1)
    b_ub = -np.array(b_rows)

    # Bounds: 0 <= x_i <= 1 (LP relaxation first, then round)
    bounds = [(0, 1) for _ in range(n)]

    # Try exact ILP first using milp
    try:
        constraints = LinearConstraint(
            np.array(A_rows),
            lb=np.ones(len(A_rows)),
            ub=np.full(len(A_rows), np.inf)
        )
        integrality = np.ones(n)  # all integer
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(lb=np.zeros(n), ub=np.ones(n))
        )
        if result.success:
            fvs_nodes = {node_list[i] for i in range(n) if result.x[i] > 0.5}
            return fvs_nodes, len(fvs_nodes)
    except Exception:
        pass

    # Fallback: LP relaxation + rounding
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            # Greedy rounding
            x = result.x.copy()
            fvs_nodes = set()
            uncovered = list(range(len(A_rows)))

            while uncovered:
                # Pick node with highest LP value among uncovered cycles
                scores = np.zeros(n)
                for ci in uncovered:
                    for node in stab_cycles[ci][0]:
                        scores[node_idx[node]] += x[node_idx[node]]

                best = np.argmax(scores)
                fvs_nodes.add(node_list[best])

                # Remove covered cycles
                uncovered = [ci for ci in uncovered
                            if node_list[best] not in stab_cycles[ci][0]]

            return fvs_nodes, len(fvs_nodes)
    except Exception:
        pass

    # Last resort: greedy by frequency
    fvs_nodes = set()
    uncovered = list(range(len(stab_cycles)))
    while uncovered:
        freq = defaultdict(int)
        for ci in uncovered:
            for node in stab_cycles[ci][0]:
                freq[node] += 1
        best = max(freq, key=freq.get)
        fvs_nodes.add(best)
        uncovered = [ci for ci in uncovered if best not in stab_cycles[ci][0]]

    return fvs_nodes, len(fvs_nodes)


def compute_stab_edge_connectivity(stab_cycles, adj, nodes):
    """
    Compute the minimum number of EDGES whose removal breaks all stabilizing cycles.
    This is an edge-based analog of FVS.
    """
    if not stab_cycles:
        return 0

    # Collect all edges that appear in stabilizing cycles
    stab_edges = set()
    for cycle_nodes, sign in stab_cycles:
        for i in range(len(cycle_nodes)):
            stab_edges.add((cycle_nodes[i], cycle_nodes[(i+1) % len(cycle_nodes)]))

    edge_list = sorted(stab_edges)
    n_edges = len(edge_list)
    edge_idx = {e: i for i, e in enumerate(edge_list)}

    if n_edges == 0:
        return 0

    # ILP: minimize sum of y_e such that for each cycle, sum(y_e for e in cycle) >= 1
    c = np.ones(n_edges)
    A_rows = []
    for cycle_nodes, sign in stab_cycles:
        row = np.zeros(n_edges)
        for i in range(len(cycle_nodes)):
            edge = (cycle_nodes[i], cycle_nodes[(i+1) % len(cycle_nodes)])
            if edge in edge_idx:
                row[edge_idx[edge]] = 1
        A_rows.append(row)

    try:
        constraints = LinearConstraint(
            np.array(A_rows),
            lb=np.ones(len(A_rows)),
            ub=np.full(len(A_rows), np.inf)
        )
        result = milp(
            c=c,
            constraints=constraints,
            integrality=np.ones(n_edges),
            bounds=Bounds(lb=np.zeros(n_edges), ub=np.ones(n_edges))
        )
        if result.success:
            return int(round(result.fun))
    except Exception:
        pass

    return -1  # computation failed


# ============================================================================
# STEP 4: Compute FVS and connectivity for all RS types
# ============================================================================

print("\n" + "=" * 100)
print("MINIMUM FEEDBACK VERTEX SET — ALL REGIME SHIFT TYPES")
print("=" * 100)
print(f"{'RS Type':<40} {'Nodes':>5} {'Stab':>5} {'|FVS|':>5} {'FVS nodes':<40} {'EdgeCut':>7}")
print("-" * 100)

for rs in sorted(results.keys()):
    r = results[rs]
    fvs_nodes, fvs_size = compute_fvs(r['stab_cycles'], r['nodes'])
    edge_cut = compute_stab_edge_connectivity(r['stab_cycles'], r['adj'], r['nodes'])

    r['fvs'] = fvs_nodes
    r['fvs_size'] = fvs_size
    r['edge_cut'] = edge_cut

    fvs_str = ', '.join(sorted(fvs_nodes)[:4])
    if len(fvs_nodes) > 4:
        fvs_str += f'... (+{len(fvs_nodes)-4})'

    print(f"{rs:<40} {r['n_nodes']:>5} {r['n_stab']:>5} {fvs_size:>5} {fvs_str:<40} {edge_cut:>7}")

# ============================================================================
# STEP 5: Map to D values (from S29)
# ============================================================================

cld_to_d = {
    "Mangroves transitions":    ("Mangroves",          750,  1500, 3),
    "Kelps transitions":        ("Kelp",                33,  2000, "A"),
    "Seagrass transitions":     ("Seagrass",           300,  1000, 3),
    "Bivalves collapse":        ("Bivalves",           150,   500, 3),
    "Freshwater eutrophication":("Lake",               200,  1000, "A"),
    "Salt marshes to tidal flats":("Salt marsh",       133,  1500, "A"),
    "Marine eutrophication":    ("Marine eutr.",        667,  2000, 3),
    "Coral transitions":        ("Coral",             1000, 15000, "A"),
    "Fisheries collapse":       ("Fisheries",          176,  3000, 3),
    "Forest to savanna":        ("Trop. forest",      4600, 50000, "A"),
    "Hypoxia":                  ("Hypoxia",            400,  1000, 3),
    "Floating plants":          ("Float. plants",      200,  1000, 3),
    "Bush encroachment":        ("Savanna",            100,  5000, "A"),
    "Desertification":          ("Drylands",            50,  1000, 3),
    "Marine foodwebs":          ("Marine FW",          176,  3000, 3),
    "Soil Salinization":        ("Soil salin.",         60,   500, 3),
    "Peatland transitions":     ("Peatlands",           80,  1000, "A"),
    "Tundra to forest":         ("Tundra→forest",       25,   500, 3),
    "Steppe to Tundra":         ("Steppe→tundra",       10,   500, 3),
}

# Override resistance scores for anchor ecosystems
override_scores = {
    "Kelp": 1, "Peatlands": 3, "Savanna": 2, "Salt marsh": 3,
    "Lake": 3, "Coral": 4, "Trop. forest": 5,
}

# ============================================================================
# STEP 6: Build analysis dataset and run statistical tests
# ============================================================================

print("\n" + "=" * 100)
print("S30 ANALYSIS DATASET")
print("=" * 100)

names = []
D_vals = []
logD = []
S_vals = []
logS = []
fvs_sizes = []
edge_cuts = []
n_stab_all = []
tiers = []
override_vals = []

print(f"\n{'Name':<15} {'D':>6} {'logD':>6} {'S':>6} {'Stab':>5} {'|FVS|':>5} {'ECut':>5} {'Over':>5} {'Tier':>5}")
print("-" * 80)

for cld_name in sorted(cld_to_d.keys()):
    if cld_name not in results:
        continue

    short_name, d, s, tier = cld_to_d[cld_name]
    r = results[cld_name]

    names.append(short_name)
    D_vals.append(d)
    logD.append(np.log10(d))
    S_vals.append(s)
    logS.append(np.log10(s))
    fvs_sizes.append(r['fvs_size'])
    edge_cuts.append(r['edge_cut'])
    n_stab_all.append(r['n_stab'])
    tiers.append(tier)
    override_vals.append(override_scores.get(short_name, np.nan))

    tier_str = "Anch" if tier == "A" else "T3"
    ov_str = str(override_scores.get(short_name, '-'))
    print(f"{short_name:<15} {d:>6} {np.log10(d):>6.3f} {s:>6} {r['n_stab']:>5} {r['fvs_size']:>5} {r['edge_cut']:>5} {ov_str:>5} {tier_str:>5}")

N = len(names)
logD_arr = np.array(logD)
logS_arr = np.array(logS)
fvs_arr = np.array(fvs_sizes, dtype=float)
ecut_arr = np.array(edge_cuts, dtype=float)
stab_arr = np.array(n_stab_all, dtype=float)
override_arr = np.array(override_vals)

print(f"\nN = {N} biological regime shift types")

# ============================================================================
# STEP 7: PRE-REGISTERED TESTS (S30 Phase 4)
# ============================================================================

print("\n" + "=" * 100)
print("S30 PHASE 4: PRE-REGISTERED STATISTICAL TESTS")
print("=" * 100)

def report(label, x, y, x_label):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        print(f"\n  {label}: insufficient data (N={n})")
        return None
    slope, intercept, r, p, se = stats.linregress(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    pred = intercept + slope * x
    rms = np.sqrt(np.mean((y - pred)**2))
    print(f"\n  {label} (N={n})")
    print(f"  log(D) = {intercept:.4f} + {slope:.4f} * {x_label}")
    print(f"  Pearson r = {r:.4f}, R^2 = {r**2:.4f}, p = {p:.6f}")
    print(f"  Spearman rho = {rho:.4f}, p = {p_rho:.6f}")
    print(f"  RMS = {rms:.4f} dex")
    return {'r': r, 'r2': r**2, 'rho': rho, 'p_rho': p_rho, 'rms': rms, 'n': n}

# Test 4.1: D vs |FVS| (all 19)
print("\n--- TEST 4.1: D vs |FVS| (all N=19) ---")
t41 = report("T4.1", fvs_arr, logD_arr, "|FVS|")

# Test 4.1b: D vs edge_cut (all 19)
print("\n--- TEST 4.1b: D vs edge_cut ---")
t41b = report("T4.1b", ecut_arr, logD_arr, "edge_cut")

# Comparison: D vs N_stab (S29 result)
print("\n--- COMPARISON: D vs N_stab (S29) ---")
t_s29 = report("S29 comparison", stab_arr, logD_arr, "N_stab")

# Test 4.2: D vs log(S) + |FVS| (all 19)
print("\n--- TEST 4.2: D vs log(S) + |FVS| ---")
def multi_reg(X_cols, y, label, names_list=None):
    X = np.column_stack([np.ones(len(y))] + X_cols)
    beta, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot
    k = len(X_cols)
    r2_adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) - k - 1)
    rms = np.sqrt(np.mean((y - pred)**2))
    f_stat = (r2 / k) / ((1 - r2) / (len(y) - k - 1)) if r2 < 1 else float('inf')
    p_f = 1 - stats.f.cdf(f_stat, k, len(y) - k - 1)
    print(f"\n  {label} (N={len(y)})")
    print(f"  Coefficients: {[f'{b:.4f}' for b in beta]}")
    print(f"  R^2 = {r2:.4f}, Adj R^2 = {r2_adj:.4f}")
    print(f"  F = {f_stat:.3f}, p = {p_f:.6f}")
    print(f"  RMS = {rms:.4f} dex")

    if names_list:
        residuals = y - pred
        sorted_idx = np.argsort(np.abs(residuals))[::-1]
        print(f"\n  {'System':<15} {'logD':>6} {'pred':>6} {'resid':>7}")
        for i in sorted_idx[:5]:
            print(f"  {names_list[i]:<15} {y[i]:>6.3f} {pred[i]:>6.3f} {residuals[i]:>+7.3f}")

    return {'r2': r2, 'r2_adj': r2_adj, 'rms': rms}

t42 = multi_reg([logS_arr, fvs_arr], logD_arr, "T4.2: log(D) = a + b*log(S) + c*|FVS|", names)

# S + edge_cut
t42b = multi_reg([logS_arr, ecut_arr], logD_arr, "T4.2b: log(D) = a + b*log(S) + c*edge_cut", names)

# S alone (baseline)
s_alone = report("Baseline: S alone", logS_arr, logD_arr, "log(S)")

# ============================================================================
# STEP 8: Anchor-only tests (Test 4.5)
# ============================================================================

print("\n" + "=" * 100)
print("TEST 4.5: ANCHOR-ONLY (N=7)")
print("=" * 100)

anchor_mask = np.array([t == "A" for t in tiers])
n_anch = sum(anchor_mask)

if n_anch >= 5:
    fvs_anch = fvs_arr[anchor_mask]
    logD_anch = logD_arr[anchor_mask]
    logS_anch = logS_arr[anchor_mask]
    names_anch = [n for n, t in zip(names, tiers) if t == "A"]

    t45_fvs = report("T4.5a: D vs |FVS| (anchor-only)", fvs_anch, logD_anch, "|FVS|")

    print(f"\n  Per-system detail:")
    for i, name in enumerate(names_anch):
        print(f"    {name:<15} |FVS|={fvs_anch[i]:.0f}, logD={logD_anch[i]:.3f}")

# ============================================================================
# STEP 9: FVS vs Override resistance (anchor-only, Test 4.4 partial)
# ============================================================================

print("\n" + "=" * 100)
print("TEST 4.4: FVS vs OVERRIDE RESISTANCE")
print("=" * 100)

ov_mask = np.isfinite(override_arr)
if sum(ov_mask) >= 5:
    fvs_ov = fvs_arr[ov_mask]
    ov_vals = override_arr[ov_mask]
    logD_ov = logD_arr[ov_mask]
    names_ov = [n for n, m in zip(names, ov_mask) if m]

    rho_fvs_ov, p_fvs_ov = stats.spearmanr(fvs_ov, ov_vals)
    rho_fvs_d, p_fvs_d = stats.spearmanr(fvs_ov, logD_ov)

    print(f"\n  Spearman(|FVS|, override): rho = {rho_fvs_ov:.4f}, p = {p_fvs_ov:.4f}")
    print(f"  Spearman(|FVS|, logD):     rho = {rho_fvs_d:.4f}, p = {p_fvs_d:.4f}")

    print(f"\n  {'System':<15} {'|FVS|':>5} {'Override':>8} {'logD':>6}")
    for i, name in enumerate(names_ov):
        print(f"  {name:<15} {fvs_ov[i]:>5.0f} {ov_vals[i]:>8.0f} {logD_ov[i]:>6.3f}")

# ============================================================================
# STEP 10: Summary comparison
# ============================================================================

print("\n" + "=" * 100)
print("SUMMARY COMPARISON: S28 vs S29 vs S30")
print("=" * 100)

print(f"\n{'Metric':<40} {'r':>6} {'R^2':>6} {'rho':>6} {'Sign':>6}")
print("-" * 70)
if t_s29:
    print(f"{'S29: N_stab (loop count)':<40} {t_s29['r']:>6.3f} {t_s29['r2']:>6.3f} {t_s29['rho']:>6.3f} {'NEG':>6}")
if t41:
    sign = "POS" if t41['r'] > 0 else "NEG"
    print(f"{'S30: |FVS| (min vertex set)':<40} {t41['r']:>6.3f} {t41['r2']:>6.3f} {t41['rho']:>6.3f} {sign:>6}")
if t41b:
    sign = "POS" if t41b['r'] > 0 else "NEG"
    print(f"{'S30: edge_cut (min edge cut)':<40} {t41b['r']:>6.3f} {t41b['r2']:>6.3f} {t41b['rho']:>6.3f} {sign:>6}")
if s_alone:
    print(f"{'Baseline: log(S) alone':<40} {s_alone['r']:>6.3f} {s_alone['r2']:>6.3f} {s_alone['rho']:>6.3f} {'POS':>6}")

print(f"\n{'Combined model':<40} {'R^2':>6} {'Adj R^2':>8} {'RMS':>6}")
print("-" * 60)
if t42:
    print(f"{'S30: S + |FVS|':<40} {t42['r2']:>6.3f} {t42['r2_adj']:>8.3f} {t42['rms']:>6.3f}")
if t42b:
    print(f"{'S30: S + edge_cut':<40} {t42b['r2']:>6.3f} {t42b['r2_adj']:>8.3f} {t42b['rms']:>6.3f}")
print(f"{'S28: S + drivers (N=19)':<40} {'0.652':>6} {'':>8} {'':>6}")
print(f"{'Prior: S + override (N=8)':<40} {'0.951':>6} {'':>8} {'':>6}")

# ============================================================================
# STEP 11: FVS NODE ANALYSIS (what nodes are regulatory bottlenecks?)
# ============================================================================

print("\n" + "=" * 100)
print("FVS NODE ANALYSIS — REGULATORY BOTTLENECKS")
print("=" * 100)

for cld_name in sorted(cld_to_d.keys()):
    if cld_name not in results:
        continue
    r = results[cld_name]
    short_name = cld_to_d[cld_name][0]
    if r['fvs_size'] > 0:
        print(f"\n  {short_name} (|FVS|={r['fvs_size']}): {sorted(r['fvs'])}")

print("\n\nDone.")
