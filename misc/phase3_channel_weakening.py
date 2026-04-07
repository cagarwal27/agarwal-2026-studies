#!/usr/bin/env python3
"""
Phase 3: Channel Weakening Tests for Model C (Coupled Sigmoid)
Tests whether barrier responds to feedback weakening in a way that maps to ε.

Model C (coupled sigmoid):
dE/dt = r_E * (E_0 * h_V^p / (h_V^p + V^p) - E)
dV/dt = r_V * (h_E^p / (h_E^p + E^p) - V)
"""
import os
import numpy as np
from scipy.optimize import fsolve
import json

r_E = 0.1
r_V = 0.05

def system(x, E_0, h_V, h_E, p):
    E, V = x
    if E <= 0 or V <= 0:
        return [0.0, 0.0]
    dE = r_E * (E_0 * h_V**p / (h_V**p + V**p) - E)
    dV = r_V * (h_E**p / (h_E**p + E**p) - V)
    return [dE, dV]

def jacobian_num(E, V, E_0, h_V, h_E, p):
    eps = 1e-8
    J = np.zeros((2, 2))
    f0 = system([E, V], E_0, h_V, h_E, p)
    f1 = system([E+eps, V], E_0, h_V, h_E, p)
    f2 = system([E, V+eps], E_0, h_V, h_E, p)
    J[0, 0] = (f1[0] - f0[0]) / eps
    J[1, 0] = (f1[1] - f0[1]) / eps
    J[0, 1] = (f2[0] - f0[0]) / eps
    J[1, 1] = (f2[1] - f0[1]) / eps
    return J

def classify(eigs):
    reals = [e.real for e in eigs]
    if all(r < -1e-10 for r in reals):
        return "stable"
    elif any(r > 1e-10 for r in reals) and any(r < -1e-10 for r in reals):
        return "saddle"
    else:
        return "other"

def find_equilibria(E_0, h_V, h_E, p):
    equilibria = []
    seen = set()
    for E_start in np.linspace(0.001, max(E_0*2, 10), 40):
        for V_start in np.linspace(0.001, 5.0, 40):
            try:
                sol = fsolve(system, [E_start, V_start],
                           args=(E_0, h_V, h_E, p), full_output=True)
                if sol[2] == 1:
                    E_sol, V_sol = sol[0]
                    if E_sol > 1e-6 and V_sol > 1e-6:
                        key = (round(E_sol, 5), round(V_sol, 5))
                        if key not in seen:
                            seen.add(key)
                            resid = system([E_sol, V_sol], E_0, h_V, h_E, p)
                            if abs(resid[0]) < 1e-8 and abs(resid[1]) < 1e-8:
                                J = jacobian_num(E_sol, V_sol, E_0, h_V, h_E, p)
                                eigs = np.linalg.eigvals(J)
                                equilibria.append((E_sol, V_sol, eigs, J))
            except:
                pass
    return equilibria

def analyze_equilibria(E_0, h_V, h_E, p):
    """Return (clear, saddle, turbid) or None if not bistable."""
    eqs = find_equilibria(E_0, h_V, h_E, p)
    stables = [(E, V, eigs, J) for E, V, eigs, J in eqs if classify(eigs) == "stable"]
    saddles = [(E, V, eigs, J) for E, V, eigs, J in eqs if classify(eigs) == "saddle"]
    if len(stables) >= 2 and len(saddles) >= 1:
        stables.sort(key=lambda x: x[0])
        return stables[0], saddles[0], stables[-1]
    return None

# ============================================================
# Baseline
# ============================================================
E_0 = 5.0
h_V_base = 0.5
h_E_base = 2.0
p = 4

print("=" * 70)
print("PHASE 3: CHANNEL WEAKENING TESTS")
print("=" * 70)

print(f"\nBaseline: E_0={E_0}, h_V={h_V_base}, h_E={h_E_base}, p={p}")
result = analyze_equilibria(E_0, h_V_base, h_E_base, p)
if result:
    clear, saddle, turbid = result
    print(f"  Clear:  E={clear[0]:.6f}, V={clear[1]:.6f}")
    print(f"  Saddle: E={saddle[0]:.6f}, V={saddle[1]:.6f}")
    print(f"  Turbid: E={turbid[0]:.6f}, V={turbid[1]:.6f}")

# ============================================================
# Step 3.1: Weaken macrophyte→turbidity feedback (vary h_V)
# Larger h_V = weaker feedback (macrophytes less effective at clearing water)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3.1: Vary h_V (macrophyte→turbidity feedback strength)")
print("Larger h_V = WEAKER macrophyte effect on turbidity")
print("=" * 70)

h_V_values = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.00, 1.50, 2.00, 3.00, 5.00]

print(f"\n{'h_V':<8} {'Bistable':<10} {'E_cl':<10} {'V_cl':<10} {'E_sad':<10} {'V_sad':<10} "
      f"{'E_tu':<10} {'V_tu':<10} {'λ_u':<10} {'λ_s':<10} {'det_J_cl':<12} {'det_J_sa':<12}")
print("-" * 130)

h_V_results = []
for h_V in h_V_values:
    result = analyze_equilibria(E_0, h_V, h_E_base, p)
    if result:
        clear, saddle, turbid = result
        eigs_sa = saddle[2]
        J_cl = clear[3]
        J_sa = saddle[3]
        lam_u = max(e.real for e in eigs_sa)
        lam_s = min(e.real for e in eigs_sa)
        det_J_cl = np.linalg.det(J_cl)
        det_J_sa = np.linalg.det(J_sa)
        slowest_cl = min(abs(e.real) for e in clear[2])
        tau_cl = 1.0 / slowest_cl

        # Kramers-Langer prefactor
        prefactor_KL = (lam_u / (2 * np.pi)) * np.sqrt(abs(det_J_cl) / abs(det_J_sa))

        h_V_results.append({
            'h_V': h_V,
            'E_cl': clear[0], 'V_cl': clear[1],
            'E_sa': saddle[0], 'V_sa': saddle[1],
            'E_tu': turbid[0], 'V_tu': turbid[1],
            'lam_u': lam_u, 'lam_s': lam_s,
            'det_J_cl': det_J_cl, 'det_J_sa': det_J_sa,
            'tau_cl': tau_cl, 'prefactor_KL': prefactor_KL,
        })

        print(f"{h_V:<8.2f} {'YES':<10} {clear[0]:<10.4f} {clear[1]:<10.4f} "
              f"{saddle[0]:<10.4f} {saddle[1]:<10.4f} {turbid[0]:<10.4f} {turbid[1]:<10.4f} "
              f"{lam_u:<10.6f} {lam_s:<10.6f} {det_J_cl:<12.8f} {det_J_sa:<12.8f}")
    else:
        print(f"{h_V:<8.2f} {'NO':<10}")
        h_V_results.append({'h_V': h_V, 'bistable': False})

# ============================================================
# Step 3.2: Weaken turbidity→macrophyte feedback (vary h_E)
# Smaller h_E = weaker feedback (macrophytes less sensitive to turbidity)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3.2: Vary h_E (turbidity→macrophyte feedback strength)")
print("Smaller h_E = WEAKER turbidity effect on macrophytes")
print("=" * 70)

h_E_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 10.0]

print(f"\n{'h_E':<8} {'Bistable':<10} {'E_cl':<10} {'V_cl':<10} {'E_sad':<10} {'V_sad':<10} "
      f"{'E_tu':<10} {'V_tu':<10} {'λ_u':<10} {'λ_s':<10}")
print("-" * 100)

h_E_results = []
for h_E in h_E_values:
    result = analyze_equilibria(E_0, h_V_base, h_E, p)
    if result:
        clear, saddle, turbid = result
        eigs_sa = saddle[2]
        J_cl = clear[3]
        J_sa = saddle[3]
        lam_u = max(e.real for e in eigs_sa)
        lam_s = min(e.real for e in eigs_sa)
        det_J_cl = np.linalg.det(J_cl)
        det_J_sa = np.linalg.det(J_sa)
        slowest_cl = min(abs(e.real) for e in clear[2])
        tau_cl = 1.0 / slowest_cl

        prefactor_KL = (lam_u / (2 * np.pi)) * np.sqrt(abs(det_J_cl) / abs(det_J_sa))

        h_E_results.append({
            'h_E': h_E,
            'E_cl': clear[0], 'V_cl': clear[1],
            'E_sa': saddle[0], 'V_sa': saddle[1],
            'E_tu': turbid[0], 'V_tu': turbid[1],
            'lam_u': lam_u, 'lam_s': lam_s,
            'det_J_cl': det_J_cl, 'det_J_sa': det_J_sa,
            'tau_cl': tau_cl, 'prefactor_KL': prefactor_KL,
        })

        print(f"{h_E:<8.1f} {'YES':<10} {clear[0]:<10.4f} {clear[1]:<10.4f} "
              f"{saddle[0]:<10.4f} {saddle[1]:<10.4f} {turbid[0]:<10.4f} {turbid[1]:<10.4f} "
              f"{lam_u:<10.6f} {lam_s:<10.6f}")
    else:
        print(f"{h_E:<8.1f} {'NO':<10}")
        h_E_results.append({'h_E': h_E, 'bistable': False})

# ============================================================
# Write QPot parameter files for each bistable case
# ============================================================
print("\n" + "=" * 70)
print("QPOT PARAMETER FILES FOR BISTABLE CASES")
print("=" * 70)

# Write a JSON file with all bistable cases for the R script
all_cases = []

for r in h_V_results:
    if 'E_cl' in r:
        all_cases.append({
            'type': 'h_V_sweep',
            'h_V': r['h_V'],
            'h_E': h_E_base,
            'E_cl': r['E_cl'], 'V_cl': r['V_cl'],
            'E_sa': r['E_sa'], 'V_sa': r['V_sa'],
            'E_tu': r['E_tu'], 'V_tu': r['V_tu'],
            'lam_u': r['lam_u'], 'lam_s': r['lam_s'],
            'det_J_cl': r['det_J_cl'], 'det_J_sa': r['det_J_sa'],
            'tau_cl': r['tau_cl'], 'prefactor_KL': r['prefactor_KL'],
        })

for r in h_E_results:
    if 'E_cl' in r:
        all_cases.append({
            'type': 'h_E_sweep',
            'h_V': h_V_base,
            'h_E': r['h_E'],
            'E_cl': r['E_cl'], 'V_cl': r['V_cl'],
            'E_sa': r['E_sa'], 'V_sa': r['V_sa'],
            'E_tu': r['E_tu'], 'V_tu': r['V_tu'],
            'lam_u': r['lam_u'], 'lam_s': r['lam_s'],
            'det_J_cl': r['det_J_cl'], 'det_J_sa': r['det_J_sa'],
            'tau_cl': r['tau_cl'], 'prefactor_KL': r['prefactor_KL'],
        })

with open(os.path.join(os.path.dirname(__file__), 'phase3_cases.json'), 'w') as f:
    json.dump({'E_0': E_0, 'r_E': r_E, 'r_V': r_V, 'p': p, 'cases': all_cases}, f, indent=2)

print(f"Wrote {len(all_cases)} bistable cases to phase3_cases.json")

# ============================================================
# Step 3.3: Interpretation
# ============================================================
print("\n" + "=" * 70)
print("STEP 3.3: INTERPRETATION")
print("=" * 70)

# For h_V sweep: define candidate ε
print("\n--- h_V sweep: candidate ε definitions ---")
print(f"{'h_V':<8} {'ε_1=h_V/h_V_max':<18} {'ε_2=(1+h_V)^-p':<18} {'ε_3=h_V^p/(1+h_V^p)':<22} {'λ_u':<12} {'1/ε_1':<10}")
print("-" * 90)

for r in h_V_results:
    if 'E_cl' in r:
        h_V = r['h_V']
        h_V_max = 5.0  # largest h_V tested
        eps1 = h_V / h_V_max
        eps2 = 1.0 / (1 + h_V)**p
        eps3 = h_V**p / (1 + h_V**p)
        print(f"{h_V:<8.2f} {eps1:<18.6f} {eps2:<18.6f} {eps3:<22.8f} {r['lam_u']:<12.6f} {1/eps1:<10.1f}")

# For h_E sweep: define candidate ε
print("\n--- h_E sweep: candidate ε definitions ---")
print(f"{'h_E':<8} {'ε_1=h_E_min/h_E':<18} {'ε_2=1/(1+h_E^p)':<18} {'λ_u':<12} {'1/ε_1':<10}")
print("-" * 70)

for r in h_E_results:
    if 'E_cl' in r:
        h_E = r['h_E']
        h_E_min = 0.5  # smallest h_E tested
        eps1 = h_E_min / h_E
        eps2 = 1.0 / (1 + h_E**p)
        print(f"{h_E:<8.1f} {eps1:<18.6f} {eps2:<18.8f} {r['lam_u']:<12.6f} {1/eps1:<10.1f}")

# Analyze whether λ_u correlates with feedback parameters
print("\n--- Correlation: λ_u vs feedback parameter ---")
print("\nh_V sweep (macrophyte→turbidity weakening):")
h_V_bist = [r for r in h_V_results if 'lam_u' in r]
if len(h_V_bist) >= 2:
    h_vs = [r['h_V'] for r in h_V_bist]
    lam_us = [r['lam_u'] for r in h_V_bist]
    ln_h_vs = [np.log(h) for h in h_vs]
    ln_lam_us = [np.log(l) for l in lam_us]

    # Linear fit in log-log
    if len(h_vs) >= 3:
        coeffs = np.polyfit(ln_h_vs, ln_lam_us, 1)
        print(f"  log(λ_u) vs log(h_V): slope = {coeffs[0]:.4f}, intercept = {coeffs[1]:.4f}")
        print(f"  => λ_u ∝ h_V^{coeffs[0]:.2f}")

print("\nh_E sweep (turbidity→macrophyte weakening):")
h_E_bist = [r for r in h_E_results if 'lam_u' in r]
if len(h_E_bist) >= 2:
    h_Es = [r['h_E'] for r in h_E_bist]
    lam_us = [r['lam_u'] for r in h_E_bist]
    ln_h_Es = [np.log(h) for h in h_Es]
    ln_lam_us = [np.log(l) for l in lam_us]

    if len(h_Es) >= 3:
        coeffs = np.polyfit(ln_h_Es, ln_lam_us, 1)
        print(f"  log(λ_u) vs log(h_E): slope = {coeffs[0]:.4f}, intercept = {coeffs[1]:.4f}")
        print(f"  => λ_u ∝ h_E^{coeffs[0]:.2f}")

print("\nDone with Phase 3 equilibria analysis.")
