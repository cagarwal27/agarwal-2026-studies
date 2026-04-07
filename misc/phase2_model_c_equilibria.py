#!/usr/bin/env python3
"""
Phase 2: 2D Macrophyte-Turbidity Model (Model C) — Equilibria and Eigenvalues.
van Nes & Scheffer 2005/2007 model.
"""
import os
import numpy as np
from scipy.optimize import fsolve
from itertools import product as iterproduct

# ============================================================
# Model C parameters (van Nes & Scheffer 2007, Table 1)
# ============================================================
r_E = 0.1    # turbidity growth rate (day^-1)
r_V = 0.05   # macrophyte growth rate (day^-1)
h_V = 0.2    # critical macrophyte cover for turbidity effect
h_E = 2.0    # critical turbidity for macrophyte survival
p   = 4      # Hill coefficient for macrophyte response

def dEdt(E, V, E_0):
    """Turbidity equation."""
    if V <= 0 or E <= 0:
        return 0.0
    return r_E * E * (1 - E * V / (E_0 * (h_V + V)))

def dVdt(E, V, E_0):
    """Macrophyte equation."""
    if V <= 0 or E <= 0:
        return 0.0
    K_V = (h_E**p + E**p) / E**p  # = 1 + (h_E/E)^p, carrying capacity for V
    # Actually: dV/dt = r_V * V * (1 - V * E^p / (h_E^p + E^p))
    # = r_V * V * (1 - V / K_V)  where K_V = (h_E^p + E^p) / E^p
    return r_V * V * (1 - V * E**p / (h_E**p + E**p))

def system(x, E_0):
    E, V = x
    return [dEdt(E, V, E_0), dVdt(E, V, E_0)]

def jacobian(E, V, E_0):
    """Analytical Jacobian of the 2D system."""
    eps = 1e-8
    J = np.zeros((2, 2))
    f0 = [dEdt(E, V, E_0), dVdt(E, V, E_0)]
    # dF/dE
    f1 = [dEdt(E+eps, V, E_0), dVdt(E+eps, V, E_0)]
    J[0, 0] = (f1[0] - f0[0]) / eps
    J[1, 0] = (f1[1] - f0[1]) / eps
    # dF/dV
    f2 = [dEdt(E, V+eps, E_0), dVdt(E, V+eps, E_0)]
    J[0, 1] = (f2[0] - f0[0]) / eps
    J[1, 1] = (f2[1] - f0[1]) / eps
    return J

# ============================================================
# Find equilibria
# ============================================================
# At equilibrium:
# dE/dt = 0: E=0 or E*V/(E_0*(h_V+V)) = 1, i.e. E = E_0*(h_V+V)/V = E_0*(h_V/V + 1)
# dV/dt = 0: V=0 or V*E^p/(h_E^p+E^p) = 1, i.e. V = (h_E^p+E^p)/E^p = 1 + (h_E/E)^p
#
# So the non-trivial equilibria satisfy:
#   E = E_0 * (h_V/V + 1)
#   V = 1 + (h_E/E)^p
#
# Substituting: E = E_0 * (h_V / (1 + (h_E/E)^p) + 1)
# This is a single equation in E.

def nullcline_eq(E, E_0):
    """E-nullcline gives V as function of E: V = E_0*(h_V+V)/E => solve for V."""
    # E = E_0*(h_V+V)/V => EV = E_0*h_V + E_0*V => V(E-E_0) = E_0*h_V
    # V = E_0*h_V/(E-E_0)  [requires E > E_0]
    if E <= E_0:
        return np.nan
    return E_0 * h_V / (E - E_0)

def nullcline_V(E, E_0):
    """V-nullcline: V = 1 + (h_E/E)^p"""
    return 1.0 + (h_E / E)**p

def equilibrium_equation(E, E_0):
    """The equation to solve: V from E-nullcline = V from V-nullcline."""
    V_E = nullcline_eq(E, E_0)
    V_V = nullcline_V(E, E_0)
    if np.isnan(V_E):
        return 1e10
    return V_E - V_V

print("=" * 70)
print("PHASE 2: 2D MACROPHYTE-TURBIDITY MODEL (MODEL C)")
print("=" * 70)

# Scan E_0 to find bistable range
print("\n--- Scanning E_0 for bistability ---")
for E_0_test in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
    # Try many starting points
    equilibria = []
    seen = set()

    for E_start, V_start in iterproduct(
        np.linspace(E_0_test + 0.01, 20.0, 30),
        np.linspace(0.01, 20.0, 30)
    ):
        try:
            sol = fsolve(system, [E_start, V_start], args=(E_0_test,), full_output=True)
            x_sol = sol[0]
            info = sol[1]
            if sol[2] == 1:  # converged
                E_sol, V_sol = x_sol
                if E_sol > 0.01 and V_sol > 0.01:
                    # Round to avoid duplicates
                    key = (round(E_sol, 3), round(V_sol, 3))
                    if key not in seen:
                        seen.add(key)
                        J = jacobian(E_sol, V_sol, E_0_test)
                        eigs = np.linalg.eigvals(J)
                        equilibria.append((E_sol, V_sol, eigs))
        except:
            pass

    n_stable = sum(1 for _, _, eigs in equilibria if all(e.real < 0 for e in eigs))
    n_saddle = sum(1 for _, _, eigs in equilibria if any(e.real > 0 for e in eigs) and any(e.real < 0 for e in eigs))
    print(f"  E_0 = {E_0_test:.1f}: {len(equilibria)} equilibria ({n_stable} stable, {n_saddle} saddle)")
    for E_sol, V_sol, eigs in equilibria:
        eig_str = ", ".join([f"{e.real:.4f}" for e in sorted(eigs, key=lambda x: x.real)])
        typ = "stable" if all(e.real < 0 for e in eigs) else ("saddle" if any(e.real > 0 for e in eigs) else "unstable")
        print(f"    E={E_sol:.4f}, V={V_sol:.4f}  eigenvalues=[{eig_str}]  ({typ})")

# ============================================================
# Detailed analysis at best E_0
# ============================================================
print("\n" + "=" * 70)
print("DETAILED ANALYSIS")
print("=" * 70)

# Try E_0 = 5.0 first, then adjust
best_E0 = None
for E_0_try in [5.0, 4.0, 6.0, 3.0, 7.0, 8.0]:
    equilibria = []
    seen = set()

    for E_start, V_start in iterproduct(
        np.linspace(E_0_try + 0.01, 25.0, 50),
        np.linspace(0.01, 25.0, 50)
    ):
        try:
            sol = fsolve(system, [E_start, V_start], args=(E_0_try,), full_output=True)
            if sol[2] == 1:
                E_sol, V_sol = sol[0]
                if E_sol > 0.01 and V_sol > 0.01:
                    key = (round(E_sol, 3), round(V_sol, 3))
                    if key not in seen:
                        seen.add(key)
                        J = jacobian(E_sol, V_sol, E_0_try)
                        eigs = np.linalg.eigvals(J)
                        equilibria.append((E_sol, V_sol, eigs, J))
        except:
            pass

    n_stable = sum(1 for _, _, eigs, _ in equilibria if all(e.real < 0 for e in eigs))
    n_saddle = sum(1 for _, _, eigs, _ in equilibria if any(e.real > 0 for e in eigs) and any(e.real < 0 for e in eigs))

    if n_stable >= 2 and n_saddle >= 1:
        best_E0 = E_0_try
        print(f"\nUsing E_0 = {E_0_try} (bistable: {n_stable} stable, {n_saddle} saddle)")

        # Sort: low E = clear, high E = turbid
        stables = [(E, V, eigs, J) for E, V, eigs, J in equilibria if all(e.real < 0 for e in eigs)]
        saddles = [(E, V, eigs, J) for E, V, eigs, J in equilibria if any(e.real > 0 for e in eigs) and any(e.real < 0 for e in eigs)]

        stables.sort(key=lambda x: x[0])  # sort by E
        clear_eq = stables[0]   # low E = clear water
        turbid_eq = stables[-1] # high E = turbid
        saddle_eq = saddles[0]

        print(f"\n  CLEAR equilibrium: E = {clear_eq[0]:.6f}, V = {clear_eq[1]:.6f}")
        J_cl = clear_eq[3]
        eigs_cl = clear_eq[2]
        print(f"    Jacobian: {J_cl}")
        print(f"    Eigenvalues: {[f'{e.real:.6f}' for e in sorted(eigs_cl, key=lambda x: x.real)]}")
        print(f"    det(J) = {np.linalg.det(J_cl):.8f}")

        print(f"\n  TURBID equilibrium: E = {turbid_eq[0]:.6f}, V = {turbid_eq[1]:.6f}")
        J_tu = turbid_eq[3]
        eigs_tu = turbid_eq[2]
        print(f"    Jacobian: {J_tu}")
        print(f"    Eigenvalues: {[f'{e.real:.6f}' for e in sorted(eigs_tu, key=lambda x: x.real)]}")
        print(f"    det(J) = {np.linalg.det(J_tu):.8f}")

        print(f"\n  SADDLE: E = {saddle_eq[0]:.6f}, V = {saddle_eq[1]:.6f}")
        J_sa = saddle_eq[3]
        eigs_sa = saddle_eq[2]
        print(f"    Jacobian: {J_sa}")
        print(f"    Eigenvalues: {[f'{e.real:.6f}' for e in sorted(eigs_sa, key=lambda x: x.real)]}")
        print(f"    det(J) = {np.linalg.det(J_sa):.8f}")

        # 2D Kramers-Langer prefactor
        lam_u = max(e.real for e in eigs_sa)  # positive eigenvalue at saddle
        lam_s = min(e.real for e in eigs_sa)  # negative eigenvalue at saddle
        det_J_clear = np.linalg.det(J_cl)

        # Kramers-Langer: rate = (lam_u/(2π)) * sqrt(|det J_min| / |det J_sad|) * exp(-barrier)
        # So: 1/(C×τ) depends on definition. Using:
        # MFPT ≈ (2π/lam_u) * sqrt(|det J_sad| / |det J_min|) * exp(barrier)
        # D = MFPT/τ where τ = 1/|slowest eigenvalue at clear eq|
        slowest_clear = min(abs(e.real) for e in eigs_cl)
        tau_clear = 1.0 / slowest_clear

        # Standard Kramers-Langer 2D:
        det_J_sad = np.linalg.det(J_sa)
        prefactor_KL = (lam_u / (2 * np.pi)) * np.sqrt(abs(det_J_clear) / abs(det_J_sad))
        inv_C_tau = prefactor_KL * tau_clear  # this gives MFPT/exp(barrier), then D = MFPT/tau

        print(f"\n  --- 2D Kramers-Langer Prefactor ---")
        print(f"  λ_u (positive at saddle) = {lam_u:.6f}")
        print(f"  λ_s (negative at saddle) = {lam_s:.6f}")
        print(f"  |det J_clear| = {abs(det_J_clear):.8f}")
        print(f"  |det J_saddle| = {abs(det_J_sad):.8f}")
        print(f"  Slowest eigenvalue at clear = {slowest_clear:.6f}")
        print(f"  τ_clear = {tau_clear:.4f}")
        print(f"  KL prefactor = λ_u/(2π) × √(|det J_cl|/|det J_sad|) = {prefactor_KL:.6f}")
        print(f"  1/(C×τ) [= prefactor × τ_clear] = {inv_C_tau:.6f}")

        # Save equilibria for QPot R script
        print(f"\n  --- For QPot R script ---")
        print(f"  E_clear = {clear_eq[0]:.6f}")
        print(f"  V_clear = {clear_eq[1]:.6f}")
        print(f"  E_turbid = {turbid_eq[0]:.6f}")
        print(f"  V_turbid = {turbid_eq[1]:.6f}")
        print(f"  E_saddle = {saddle_eq[0]:.6f}")
        print(f"  V_saddle = {saddle_eq[1]:.6f}")

        # Write equilibria to a file for other scripts
        with open(os.path.join(os.path.dirname(__file__), 'model_c_equilibria.txt'), 'w') as fout:
            fout.write(f"E_0={E_0_try}\n")
            fout.write(f"E_clear={clear_eq[0]:.8f}\n")
            fout.write(f"V_clear={clear_eq[1]:.8f}\n")
            fout.write(f"E_turbid={turbid_eq[0]:.8f}\n")
            fout.write(f"V_turbid={turbid_eq[1]:.8f}\n")
            fout.write(f"E_saddle={saddle_eq[0]:.8f}\n")
            fout.write(f"V_saddle={saddle_eq[1]:.8f}\n")
            fout.write(f"lam_u={lam_u:.8f}\n")
            fout.write(f"lam_s={lam_s:.8f}\n")
            fout.write(f"det_J_clear={det_J_clear:.8f}\n")
            fout.write(f"det_J_saddle={det_J_sad:.8f}\n")
            fout.write(f"tau_clear={tau_clear:.8f}\n")
            fout.write(f"prefactor_KL={prefactor_KL:.8f}\n")

        break
    else:
        print(f"  E_0 = {E_0_try}: not bistable ({n_stable} stable, {n_saddle} saddle)")

if best_E0 is None:
    print("\nWARNING: No bistable E_0 found! Trying wider scan...")
    for E_0_try in np.arange(1.0, 15.1, 0.5):
        equilibria = []
        seen = set()
        for E_start, V_start in iterproduct(
            np.linspace(E_0_try * 0.5, E_0_try * 5, 20),
            np.linspace(0.01, 20.0, 20)
        ):
            try:
                sol = fsolve(system, [E_start, V_start], args=(E_0_try,), full_output=True)
                if sol[2] == 1:
                    E_sol, V_sol = sol[0]
                    if E_sol > 0.01 and V_sol > 0.01:
                        key = (round(E_sol, 2), round(V_sol, 2))
                        if key not in seen:
                            seen.add(key)
                            J = jacobian(E_sol, V_sol, E_0_try)
                            eigs = np.linalg.eigvals(J)
                            equilibria.append((E_sol, V_sol, eigs, J))
            except:
                pass
        n_stable = sum(1 for _, _, eigs, _ in equilibria if all(e.real < 0 for e in eigs))
        n_saddle = sum(1 for _, _, eigs, _ in equilibria if any(e.real > 0 for e in eigs) and any(e.real < 0 for e in eigs))
        if n_stable >= 2:
            print(f"  E_0 = {E_0_try}: {n_stable} stable, {n_saddle} saddle")

print("\nDone with Phase 2 equilibria.")
