#!/usr/bin/env python3
"""
Phase 2 v2: 2D Macrophyte-Turbidity Model (Model C)
Try multiple formulations to find the bistable version.

The equations in the prompt use (h_V + V)/V which is a simple hyperbola.
Two monotonically decreasing nullclines can only intersect once => no bistability.
The original Scheffer model must use Hill functions on BOTH interactions.
"""
import os
import numpy as np
from scipy.optimize import fsolve
from itertools import product as iterproduct

# Base parameters
r_E = 0.1
r_V = 0.05
h_V_base = 0.2
h_E_base = 2.0
p = 4

# ============================================================
# Formulation A: As written in prompt (linear V→E coupling)
# K_E = E_0*(1 + h_V/V), K_V = 1 + (h_E/E)^p
# KNOWN: not bistable (both nullclines monotonically decreasing)
# ============================================================

# ============================================================
# Formulation B: Hill function on BOTH interactions
# K_E = E_0*(1 + (h_V/V)^p), K_V = 1 + (h_E/E)^p
# ============================================================

def system_B(x, E_0, h_V, h_E, p):
    E, V = x
    if E <= 0 or V <= 0:
        return [0.0, 0.0]
    K_E = E_0 * (1 + (h_V/V)**p)
    K_V = 1 + (h_E/E)**p
    dE = r_E * E * (1 - E/K_E)
    dV = r_V * V * (1 - V/K_V)
    return [dE, dV]

# ============================================================
# Formulation C: Coupled Hill function sigmoid model
# dE/dt = r_E*(E_0 * h_V^p/(h_V^p + V^p) - E)
# dV/dt = r_V*(h_E^p/(h_E^p + E^p) - V)
# ============================================================

def system_C(x, E_0, h_V, h_E, p):
    E, V = x
    if E <= 0 or V <= 0:
        return [0.0, 0.0]
    dE = r_E * (E_0 * h_V**p / (h_V**p + V**p) - E)
    dV = r_V * (h_E**p / (h_E**p + E**p) - V)
    return [dE, dV]

# ============================================================
# Formulation D: Mixed logistic-sigmoid
# dE/dt = r_E*E*(1 - E/(E_0*h_V^p/(h_V^p + V^p)))
# dV/dt = r_V*V*(1 - V*E^p/(h_E^p + E^p))
# ============================================================

def system_D(x, E_0, h_V, h_E, p):
    E, V = x
    if E <= 0 or V <= 0:
        return [0.0, 0.0]
    K_E = E_0 * h_V**p / (h_V**p + V**p)
    if K_E <= 0:
        return [0.0, 0.0]
    dE = r_E * E * (1 - E/K_E)
    dV = r_V * V * (1 - V * E**p / (h_E**p + E**p))
    return [dE, dV]

# ============================================================
# Formulation E: Original prompt but with V carrying capacity scaled
# dE/dt = r_E*E*(1 - E*V/(E_0*(h_V + V)))
# dV/dt = r_V*V*(1 - V*E^p/(h_E^p + E^p))
# But vary E_0 over much wider range and try different h_V, h_E
# ============================================================

def system_E(x, E_0, h_V, h_E, p):
    E, V = x
    if E <= 0 or V <= 0:
        return [0.0, 0.0]
    dE = r_E * E * (1 - E*V / (E_0 * (h_V + V)))
    dV = r_V * V * (1 - V * E**p / (h_E**p + E**p))
    return [dE, dV]

def jacobian_num(system_func, E, V, E_0, h_V, h_E, p):
    eps = 1e-8
    J = np.zeros((2, 2))
    f0 = system_func([E, V], E_0, h_V, h_E, p)
    f1 = system_func([E+eps, V], E_0, h_V, h_E, p)
    f2 = system_func([E, V+eps], E_0, h_V, h_E, p)
    J[0, 0] = (f1[0] - f0[0]) / eps
    J[1, 0] = (f1[1] - f0[1]) / eps
    J[0, 1] = (f2[0] - f0[0]) / eps
    J[1, 1] = (f2[1] - f0[1]) / eps
    return J

def find_equilibria(system_func, E_0, h_V, h_E, p, E_range=(0.01, 30), V_range=(0.01, 20), n_starts=40):
    """Find all equilibria via multi-start fsolve."""
    equilibria = []
    seen = set()

    for E_start in np.linspace(E_range[0], E_range[1], n_starts):
        for V_start in np.linspace(V_range[0], V_range[1], n_starts):
            try:
                sol = fsolve(system_func, [E_start, V_start],
                           args=(E_0, h_V, h_E, p), full_output=True)
                if sol[2] == 1:
                    E_sol, V_sol = sol[0]
                    if E_sol > 1e-4 and V_sol > 1e-4:
                        key = (round(E_sol, 4), round(V_sol, 4))
                        if key not in seen:
                            seen.add(key)
                            # Verify it's actually an equilibrium
                            resid = system_func([E_sol, V_sol], E_0, h_V, h_E, p)
                            if abs(resid[0]) < 1e-8 and abs(resid[1]) < 1e-8:
                                J = jacobian_num(system_func, E_sol, V_sol, E_0, h_V, h_E, p)
                                eigs = np.linalg.eigvals(J)
                                equilibria.append((E_sol, V_sol, eigs, J))
            except:
                pass

    return equilibria

def classify(eigs):
    reals = [e.real for e in eigs]
    if all(r < -1e-10 for r in reals):
        return "stable"
    elif any(r > 1e-10 for r in reals) and any(r < -1e-10 for r in reals):
        return "saddle"
    elif all(r > 1e-10 for r in reals):
        return "unstable"
    else:
        return "marginal"

# ============================================================
# Test all formulations
# ============================================================
print("=" * 70)
print("TESTING MODEL C FORMULATIONS FOR BISTABILITY")
print("=" * 70)

formulations = {
    'B: Both Hill': system_B,
    'C: Coupled sigmoid': system_C,
    'D: Mixed logistic-Hill': system_D,
    'E: Original prompt': system_E,
}

# Test with various parameters
param_combos = [
    # (E_0, h_V, h_E, p, label)
    (5.0, 0.2, 2.0, 4, "baseline"),
    (5.0, 0.5, 2.0, 4, "h_V=0.5"),
    (5.0, 1.0, 2.0, 4, "h_V=1.0"),
    (5.0, 2.0, 2.0, 4, "h_V=2.0"),
    (5.0, 0.2, 3.0, 4, "h_E=3.0"),
    (5.0, 0.2, 5.0, 4, "h_E=5.0"),
    (5.0, 0.2, 2.0, 8, "p=8"),
    (3.0, 0.2, 2.0, 4, "E_0=3"),
    (10.0, 0.2, 2.0, 4, "E_0=10"),
    (5.0, 1.0, 3.0, 8, "combo1"),
    (5.0, 0.5, 2.0, 8, "combo2"),
    (3.0, 1.0, 2.0, 8, "combo3"),
    (4.0, 0.5, 3.0, 4, "combo4"),
]

bistable_found = []

for form_name, system_func in formulations.items():
    print(f"\n--- {form_name} ---")
    for E_0, h_V, h_E, p_val, label in param_combos:
        eqs = find_equilibria(system_func, E_0, h_V, h_E, p_val, n_starts=25)
        n_stable = sum(1 for _, _, eigs, _ in eqs if classify(eigs) == "stable")
        n_saddle = sum(1 for _, _, eigs, _ in eqs if classify(eigs) == "saddle")
        if n_stable >= 2 and n_saddle >= 1:
            print(f"  *** BISTABLE *** {label} (E_0={E_0}, h_V={h_V}, h_E={h_E}, p={p_val}): {n_stable} stable, {n_saddle} saddle")
            bistable_found.append((form_name, system_func, E_0, h_V, h_E, p_val, eqs))
            for E_sol, V_sol, eigs, J in eqs:
                eig_str = ", ".join([f"{e.real:.6f}" for e in sorted(eigs, key=lambda x: x.real)])
                print(f"      E={E_sol:.4f}, V={V_sol:.4f}  eigs=[{eig_str}]  ({classify(eigs)})")

# ============================================================
# Detailed analysis of first bistable case
# ============================================================
if bistable_found:
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS OF BISTABLE MODEL")
    print("=" * 70)

    form_name, system_func, E_0, h_V, h_E, p_val, eqs = bistable_found[0]
    print(f"Using: {form_name} with E_0={E_0}, h_V={h_V}, h_E={h_E}, p={p_val}")

    # Refine with more starting points
    eqs = find_equilibria(system_func, E_0, h_V, h_E, p_val, n_starts=60)

    stables = [(E, V, eigs, J) for E, V, eigs, J in eqs if classify(eigs) == "stable"]
    saddles = [(E, V, eigs, J) for E, V, eigs, J in eqs if classify(eigs) == "saddle"]

    stables.sort(key=lambda x: x[0])
    clear_eq = stables[0]   # low E = clear water
    turbid_eq = stables[-1] # high E = turbid

    # Pick the saddle closest to being between the two stables
    saddle_eq = saddles[0]

    print(f"\n  CLEAR equilibrium: E = {clear_eq[0]:.8f}, V = {clear_eq[1]:.8f}")
    eigs_cl = clear_eq[2]
    J_cl = clear_eq[3]
    print(f"    Eigenvalues: {[f'{e.real:.8f}' for e in sorted(eigs_cl, key=lambda x: x.real)]}")
    print(f"    Jacobian:\n      {J_cl[0]}\n      {J_cl[1]}")
    print(f"    det(J) = {np.linalg.det(J_cl):.10f}")
    print(f"    tr(J) = {np.trace(J_cl):.10f}")

    print(f"\n  TURBID equilibrium: E = {turbid_eq[0]:.8f}, V = {turbid_eq[1]:.8f}")
    eigs_tu = turbid_eq[2]
    J_tu = turbid_eq[3]
    print(f"    Eigenvalues: {[f'{e.real:.8f}' for e in sorted(eigs_tu, key=lambda x: x.real)]}")
    print(f"    Jacobian:\n      {J_tu[0]}\n      {J_tu[1]}")
    print(f"    det(J) = {np.linalg.det(J_tu):.10f}")

    print(f"\n  SADDLE: E = {saddle_eq[0]:.8f}, V = {saddle_eq[1]:.8f}")
    eigs_sa = saddle_eq[2]
    J_sa = saddle_eq[3]
    print(f"    Eigenvalues: {[f'{e.real:.8f}' for e in sorted(eigs_sa, key=lambda x: x.real)]}")
    print(f"    Jacobian:\n      {J_sa[0]}\n      {J_sa[1]}")
    print(f"    det(J) = {np.linalg.det(J_sa):.10f}")

    # 2D Kramers-Langer prefactor
    lam_u = max(e.real for e in eigs_sa)
    lam_s = min(e.real for e in eigs_sa)
    det_J_cl = np.linalg.det(J_cl)
    det_J_sa = np.linalg.det(J_sa)
    slowest_cl = min(abs(e.real) for e in eigs_cl)
    tau_cl = 1.0 / slowest_cl

    prefactor_KL = (lam_u / (2 * np.pi)) * np.sqrt(abs(det_J_cl) / abs(det_J_sa))
    inv_C_tau = prefactor_KL * tau_cl

    print(f"\n  --- 2D Kramers-Langer Prefactor ---")
    print(f"  λ_u (positive at saddle) = {lam_u:.8f}")
    print(f"  λ_s (negative at saddle) = {lam_s:.8f}")
    print(f"  |det J_clear| = {abs(det_J_cl):.10f}")
    print(f"  |det J_saddle| = {abs(det_J_sa):.10f}")
    print(f"  Slowest eigenvalue at clear = {slowest_cl:.8f}")
    print(f"  τ_clear = {tau_cl:.4f}")
    print(f"  KL prefactor = {prefactor_KL:.8f}")
    print(f"  1/(C×τ) = {inv_C_tau:.8f}")

    # Write equation string info for QPot
    print(f"\n  --- Model info for QPot ---")
    print(f"  Formulation: {form_name}")
    print(f"  E_0={E_0}, h_V={h_V}, h_E={h_E}, p={p_val}")
    print(f"  E_clear={clear_eq[0]:.8f}, V_clear={clear_eq[1]:.8f}")
    print(f"  E_turbid={turbid_eq[0]:.8f}, V_turbid={turbid_eq[1]:.8f}")
    print(f"  E_saddle={saddle_eq[0]:.8f}, V_saddle={saddle_eq[1]:.8f}")

    # Save for QPot
    with open(os.path.join(os.path.dirname(__file__), 'model_c_equilibria.txt'), 'w') as fout:
        fout.write(f"formulation={form_name}\n")
        fout.write(f"E_0={E_0}\n")
        fout.write(f"h_V={h_V}\n")
        fout.write(f"h_E={h_E}\n")
        fout.write(f"p={p_val}\n")
        fout.write(f"r_E={r_E}\n")
        fout.write(f"r_V={r_V}\n")
        fout.write(f"E_clear={clear_eq[0]:.8f}\n")
        fout.write(f"V_clear={clear_eq[1]:.8f}\n")
        fout.write(f"E_turbid={turbid_eq[0]:.8f}\n")
        fout.write(f"V_turbid={turbid_eq[1]:.8f}\n")
        fout.write(f"E_saddle={saddle_eq[0]:.8f}\n")
        fout.write(f"V_saddle={saddle_eq[1]:.8f}\n")
        fout.write(f"lam_u={lam_u:.8f}\n")
        fout.write(f"lam_s={lam_s:.8f}\n")
        fout.write(f"det_J_clear={det_J_cl:.10f}\n")
        fout.write(f"det_J_saddle={det_J_sa:.10f}\n")
        fout.write(f"tau_clear={tau_cl:.8f}\n")
        fout.write(f"prefactor_KL={prefactor_KL:.8f}\n")
        fout.write(f"inv_C_tau={inv_C_tau:.8f}\n")

    # ============================================================
    # Now scan E_0 for bistable range using this formulation
    # ============================================================
    print(f"\n  --- Bistable range for E_0 ---")
    for E0_scan in np.arange(1.0, 15.1, 0.5):
        eqs_scan = find_equilibria(system_func, E0_scan, h_V, h_E, p_val, n_starts=20)
        ns = sum(1 for _, _, eigs, _ in eqs_scan if classify(eigs) == "stable")
        nd = sum(1 for _, _, eigs, _ in eqs_scan if classify(eigs) == "saddle")
        if ns >= 2:
            print(f"    E_0 = {E0_scan:.1f}: BISTABLE ({ns} stable, {nd} saddle)")

else:
    print("\nNo bistable formulation found with tested parameters.")
    print("Trying wider parameter sweep for formulation C (coupled sigmoid)...")

    # Extended search for formulation C
    for h_V_try in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        for h_E_try in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            for E0_try in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
                for p_try in [2, 4, 6, 8]:
                    eqs = find_equilibria(system_C, E0_try, h_V_try, h_E_try, p_try, n_starts=15)
                    ns = sum(1 for _, _, eigs, _ in eqs if classify(eigs) == "stable")
                    nd = sum(1 for _, _, eigs, _ in eqs if classify(eigs) == "saddle")
                    if ns >= 2 and nd >= 1:
                        print(f"  *** BISTABLE *** Form C: E_0={E0_try}, h_V={h_V_try}, h_E={h_E_try}, p={p_try}")
                        for E_sol, V_sol, eigs, J in eqs:
                            eig_str = ", ".join([f"{e.real:.6f}" for e in sorted(eigs, key=lambda x: x.real)])
                            print(f"      E={E_sol:.4f}, V={V_sol:.4f}  eigs=[{eig_str}]  ({classify(eigs)})")
                        bistable_found.append(('C: Coupled sigmoid', system_C, E0_try, h_V_try, h_E_try, p_try, eqs))

    if not bistable_found:
        # Try formulation D extended
        print("\nTrying wider sweep for formulation D (mixed logistic-Hill)...")
        for h_V_try in [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]:
            for h_E_try in [1.0, 2.0, 3.0, 5.0, 8.0]:
                for E0_try in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0]:
                    for p_try in [2, 4, 6, 8]:
                        eqs = find_equilibria(system_D, E0_try, h_V_try, h_E_try, p_try, n_starts=15)
                        ns = sum(1 for _, _, eigs, _ in eqs if classify(eigs) == "stable")
                        nd = sum(1 for _, _, eigs, _ in eqs if classify(eigs) == "saddle")
                        if ns >= 2 and nd >= 1:
                            print(f"  *** BISTABLE *** Form D: E_0={E0_try}, h_V={h_V_try}, h_E={h_E_try}, p={p_try}")
                            for E_sol, V_sol, eigs, J in eqs:
                                eig_str = ", ".join([f"{e.real:.6f}" for e in sorted(eigs, key=lambda x: x.real)])
                                print(f"      E={E_sol:.4f}, V={V_sol:.4f}  eigs=[{eig_str}]  ({classify(eigs)})")
                            bistable_found.append(('D: Mixed logistic-Hill', system_D, E0_try, h_V_try, h_E_try, p_try, eqs))
                            break
                    if bistable_found:
                        break
                if bistable_found:
                    break
            if bistable_found:
                break

print("\n\nDone with Phase 2 formulation search.")
