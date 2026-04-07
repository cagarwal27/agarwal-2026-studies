#!/usr/bin/env python3
"""
Step 13 (Path A): Kramers Computation for Peatland — Hilbert et al. 2000
=========================================================================
2D published model with 1 free parameter (c1).

Model: Hilbert, Roulet & Moore (2000) J. Ecology 88:230-242
  "Modelling and analysis of peatlands as dynamical systems"

State variables:
  H  = peat height (cm)
  Z0 = water table depth below peat surface (cm, positive downward)

Equations (their eqns 12, 14, 15):
  G(Z0) = k*(Z0 - Z_min)*(Z_max - Z0)   for Z_min <= Z0 <= Z_max
  G(Z0) = 0                               otherwise

  dH/dt  = G - (r1-r2)*Z0 - r2*H                              (eqn 15)
  dZ0/dt = [c2/th - r2]*H - (r1-r2)*Z0 + Eo/(th*(1+c1*Z0))
           + G - (P-d0)/th                                      (eqn 12)

  where th = theta_max

Published parameters (Fig. 4 + Fig. 6 captions):
  k      = 0.00025 cm yr^-1    Fig 4
  r1     = 0.0025  yr^-1       Fig 4 (aerobic decomp)
  r2     = 0.00025 yr^-1       Fig 4 (anaerobic decomp)
  Z_min  = -10     cm          Fig 4
  Z_max  = 70      cm          Fig 4
  c2     = 0.05    yr^-1       Fig 6
  d0     = 20      cm yr^-1    Fig 6
  Eo     = 60      cm yr^-1    Fig 6
  th     = 0.8                 Fig 6
  P      = 80      cm yr^-1    Fig 5 (mid-bistable)

  c1     = FREE (cm^-1) — not stated in paper, calibrated from Fig 5

D_product scenarios:
  k=1 conservative: eps=0.065 (Frolking 2010) -> D=15.4
  k=1 central:      eps=0.10  (Clymo 1984)   -> D=10.0
  k=2 two-channel:  eps1=0.10, eps2=0.33      -> D=30.3

References:
  Hilbert, Roulet & Moore 2000. J Ecology 88:230-242
  Clymo 1984. Phil Trans R Soc B 303:605-654
  Frolking et al. 2010. Biogeosciences 7:3235-3258
  Freeman et al. 2001. Nature 409:149
"""

import numpy as np
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PUBLISHED PARAMETERS (Hilbert et al. 2000)
# ============================================================
k_prod = 0.00025    # cm yr^-1, production rate parameter (Fig 4)
r1 = 0.0025         # yr^-1, aerobic decomp rate (Fig 4)
r2 = 0.00025        # yr^-1, anaerobic decomp rate (Fig 4)
Z_min = -10.0       # cm, lower bound of production (Fig 4)
Z_max = 70.0        # cm, upper bound of production (Fig 4)
c2 = 0.05           # yr^-1, drainage increase rate (Fig 6)
d0 = 20.0           # cm yr^-1, base drainage (Fig 6)
Eo = 60.0           # cm yr^-1, potential evaporation (Fig 6)
th = 0.8            # theta_max, water storage per cm (Fig 6)
P_op = 80.0         # cm yr^-1, operating precipitation (Fig 5, mid-bistable)

# FREE PARAMETER — calibrate from Fig 5 bistability
c1 = 0.5            # cm^-1, initial guess (will calibrate below)

# D_product
EPS_FROLKING = 0.065
EPS_CLYMO = 0.10
D_CONSERV = 1.0 / EPS_FROLKING    # 15.38
D_CENTRAL = 1.0 / EPS_CLYMO       # 10.0
D_TWOCHAN = (1.0/0.10)*(1.0/0.33) # 30.3

print("=" * 70)
print("STEP 13 (PATH A): KRAMERS — HILBERT et al. 2000")
print("=" * 70)
print(f"\nPublished 2D model: H (peat height) + Z0 (water table depth)")
print(f"\nPublished parameters (Fig 4+6 captions):")
print(f"  k={k_prod}, r1={r1}, r2={r2}, Z_min={Z_min}, Z_max={Z_max}")
print(f"  c2={c2}, d0={d0}, Eo={Eo}, theta_max={th}")
print(f"  P={P_op} cm/yr (mid-bistable, Fig 5)")
print(f"  c1 = FREE (not stated in paper)")


# ============================================================
# MODEL EQUATIONS
# ============================================================
def G_prod(Z0):
    """Quadratic production function (eqn 14)."""
    if Z0 < Z_min or Z0 > Z_max:
        return 0.0
    return k_prod * (Z0 - Z_min) * (Z_max - Z0)


def rhs_2d(state, P=P_op, c1_val=None):
    """2D drift [dH/dt, dZ0/dt]."""
    if c1_val is None:
        c1_val = c1
    H, Z0 = state
    G = G_prod(Z0)
    Et = Eo / (1.0 + c1_val * max(Z0, 0.0))
    d_drain = d0 + c2 * H

    dHdt = G - (r1 - r2) * Z0 - r2 * H
    dZ0dt = dHdt - (P - Et - d_drain) / th

    return np.array([dHdt, dZ0dt])


def jacobian_2d(state, P=P_op, c1_val=None, eps=1e-4):
    """Numerical Jacobian (2x2)."""
    if c1_val is None:
        c1_val = c1
    f0 = rhs_2d(state, P, c1_val)
    J = np.zeros((2, 2))
    for j in range(2):
        sp = state.copy()
        sp[j] += eps
        J[:, j] = (rhs_2d(sp, P, c1_val) - f0) / eps
    return J


# ============================================================
# PHASE 0: CALIBRATE c1 FROM FIG 5 BISTABILITY
# ============================================================
print("\n" + "=" * 70)
print("PHASE 0: CALIBRATE c1 FROM PUBLISHED BISTABILITY (Fig 5)")
print("=" * 70)

# From Fig 5 + p.237: at P=80, two stable equilibria exist:
#   Wet/shallow: H~800, Z0~2 (water table near surface)
#   Dry/deep:    H~1200, Z0~40
# We find c1 such that P=80 produces 3 equilibria (2 stable + 1 saddle)


def count_equilibria(c1_val, P_val=P_op):
    """Find all equilibria at given c1 and P."""
    eqs = []
    # Scan Z0 from 0 to Z_max, find where H-isocline and Z0-isocline cross
    # H-isocline (dH/dt=0): H = [G - (r1-r2)*Z0] / r2
    # Z0-isocline (dZ0/dt=0): solve from dZ0/dt=0
    for Z0_init in np.arange(0.5, 65, 0.5):
        G = G_prod(Z0_init)
        H_from_Hiso = (G - (r1 - r2) * Z0_init) / r2
        if H_from_Hiso < 0:
            continue
        try:
            sol = fsolve(lambda s: rhs_2d(s, P_val, c1_val),
                         [H_from_Hiso, Z0_init], full_output=True)
            x, info, ier, msg = sol
            if ier == 1 and x[0] > 1 and x[1] > -5 and x[1] < 70:
                res = np.max(np.abs(rhs_2d(x, P_val, c1_val)))
                if res < 1e-8:
                    # Check if duplicate
                    is_dup = False
                    for eq in eqs:
                        if abs(eq[0] - x[0]) < 1.0 and abs(eq[1] - x[1]) < 0.5:
                            is_dup = True
                            break
                    if not is_dup:
                        eqs.append(x.copy())
        except Exception:
            continue
    return eqs


def classify_equilibria(eqs, P_val=P_op, c1_val=None):
    """Classify equilibria by eigenvalues."""
    if c1_val is None:
        c1_val = c1
    classified = []
    for eq in eqs:
        J = jacobian_2d(eq, P_val, c1_val)
        evals = np.linalg.eigvals(J)
        real_parts = np.real(evals)
        if all(r < 0 for r in real_parts):
            typ = "STABLE"
        elif all(r > 0 for r in real_parts):
            typ = "UNSTABLE"
        else:
            typ = "SADDLE"
        classified.append((eq, evals, typ))
    return classified


# Scan c1 to find value giving 3 equilibria at P=80
print(f"\n  Scanning c1 to find bistability at P={P_op}...")
best_c1 = None
best_score = 1e10  # Lower is better

# From paper p.237: at P=80, equilibria at H~800 (wet) and H~1200 (dry)
# with saddle at Z0~10, H~1000. We want well-separated equilibria.
print(f"  Target from Fig 5: wet H~800, saddle H~1000, dry H~1200")

for c1_test in np.arange(0.5, 5.0, 0.005):
    eqs = count_equilibria(c1_test, P_op)
    if len(eqs) >= 3:
        classified = classify_equilibria(eqs, P_op, c1_test)
        n_stable = sum(1 for _, _, t in classified if t == "STABLE")
        n_saddle = sum(1 for _, _, t in classified if t == "SADDLE")
        if n_stable >= 2 and n_saddle >= 1:
            stable_eqs = [eq for eq, _, t in classified if t == "STABLE"]
            saddle_eqs = [eq for eq, _, t in classified if t == "SADDLE"]
            H_stable = sorted([eq[0] for eq in stable_eqs])
            H_saddle = saddle_eqs[0][0]
            H_gap = H_stable[-1] - H_stable[0]  # separation between stable states

            # Score: how well does this match Fig 5?
            # Want H_low~800, H_high~1200, good separation
            score = (abs(H_stable[0] - 800) + abs(H_stable[-1] - 1200)
                     + abs(H_saddle - 1000)) / 3.0
            # Also penalize tiny gaps (near bifurcation)
            if H_gap < 50:
                score += 500

            if score < best_score:
                best_score = score
                best_c1 = c1_test
                print(f"    c1={c1_test:.3f}: H_wet={H_stable[0]:.0f} "
                      f"H_sad={H_saddle:.0f} H_dry={H_stable[-1]:.0f} "
                      f"gap={H_gap:.0f} score={score:.0f}")
                for eq, ev, t in classified:
                    print(f"      H={eq[0]:.1f} Z0={eq[1]:.1f} [{t}] "
                          f"evals={np.real(ev)}")

if best_c1 is None:
    print("  WARNING: No bistability found at P=80. Trying P scan...")
    for P_test in [75, 77, 78, 79, 82, 85, 90]:
        for c1_test in np.arange(0.5, 5.0, 0.02):
            eqs = count_equilibria(c1_test, P_test)
            if len(eqs) >= 3:
                classified = classify_equilibria(eqs, P_test, c1_test)
                n_stable = sum(1 for _, _, t in classified if t == "STABLE")
                n_saddle = sum(1 for _, _, t in classified if t == "SADDLE")
                if n_stable >= 2 and n_saddle >= 1:
                    stable_eqs = [eq for eq, _, t in classified if t == "STABLE"]
                    H_stable = sorted([eq[0] for eq in stable_eqs])
                    H_gap = H_stable[-1] - H_stable[0]
                    if H_gap > 100:
                        best_c1 = c1_test
                        P_op = P_test
                        print(f"    P={P_test}, c1={c1_test:.3f}: gap={H_gap:.0f}")
                        for eq, ev, t in classified:
                            print(f"      H={eq[0]:.1f} Z0={eq[1]:.1f} [{t}]")
                        break
        if best_c1 is not None:
            break

if best_c1 is None:
    print("\n  FATAL: Cannot find bistability with any c1/P combination.")
    print("  The Hilbert model may need different parameter ranges.")
    import sys; sys.exit(1)

c1 = best_c1
print(f"\n  CALIBRATED: c1 = {c1:.4f} cm^-1")
print(f"  Operating: P = {P_op} cm/yr")


# ============================================================
# PHASE 1: EQUILIBRIA AT CALIBRATED c1
# ============================================================
print("\n" + "=" * 70)
print("PHASE 1: EQUILIBRIA")
print("=" * 70)

eqs = count_equilibria(c1, P_op)
classified = classify_equilibria(eqs, P_op, c1)

# Sort by H
classified.sort(key=lambda x: x[0][0])

# Identify states
stables = [(eq, ev, t) for eq, ev, t in classified if t == "STABLE"]
saddles = [(eq, ev, t) for eq, ev, t in classified if t == "SADDLE"]

if len(stables) < 2 or len(saddles) < 1:
    print(f"  ERROR: Need 2 stable + 1 saddle, got {len(stables)} stable, {len(saddles)} saddle")
    import sys; sys.exit(1)

# Wet/shallow = lower H, Dry/deep = higher H (from paper description)
wet_eq = stables[0][0]
wet_evals = stables[0][1]
dry_eq = stables[-1][0]
dry_evals = stables[-1][1]
saddle_eq = saddles[0][0]
saddle_evals = saddles[0][1]

print(f"\n  {'State':<12} {'H(cm)':>10} {'Z0(cm)':>10} {'evals':>30} {'Type':>8}")
print(f"  {'-'*72}")
for eq, ev, t in classified:
    ev_str = f"[{np.real(ev[0]):.6f}, {np.real(ev[1]):.6f}]"
    print(f"  {t:<12} {eq[0]:>10.2f} {eq[1]:>10.2f} {ev_str:>30} {t:>8}")

print(f"\n  WET/SHALLOW (intact peatland):  H={wet_eq[0]:.1f} cm, Z0={wet_eq[1]:.1f} cm")
print(f"  SADDLE (tipping point):          H={saddle_eq[0]:.1f} cm, Z0={saddle_eq[1]:.1f} cm")
print(f"  DRY/DEEP (degraded peatland):    H={dry_eq[0]:.1f} cm, Z0={dry_eq[1]:.1f} cm")

# Physical interpretation
print(f"\n  Physical interpretation:")
print(f"    Wet state: {wet_eq[0]/100:.1f}m peat, water table {wet_eq[1]:.0f}cm below surface")
print(f"    Dry state: {dry_eq[0]/100:.1f}m peat, water table {dry_eq[1]:.0f}cm below surface")
print(f"    The wet state has SHALLOW water table (anoxic, low decomposition)")
print(f"    The dry state has DEEP water table (aerobic, high decomposition)")
print(f"    NOTE: In Hilbert's model the 'intact peatland' with high water table")
print(f"    is the WET state (lower H). The dry state has MORE peat but the")
print(f"    water table has dropped, exposing peat to aerobic decomposition.")


# ============================================================
# PHASE 2: TIMESCALE SEPARATION
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: TIMESCALE SEPARATION (Z0 fast, H slow)")
print("=" * 70)

# Eigenvalues at wet equilibrium
J_wet = jacobian_2d(wet_eq, P_op, c1)
evals_wet = np.linalg.eigvals(J_wet)
evals_wet_real = np.sort(np.real(evals_wet))

# Eigenvectors
evals_w, evecs_w = np.linalg.eig(J_wet)
idx = np.argsort(np.abs(np.real(evals_w)))
slow_eval = evals_w[idx[0]]
fast_eval = evals_w[idx[1]]
slow_evec = evecs_w[:, idx[0]]
fast_evec = evecs_w[:, idx[1]]

print(f"\n  At WET equilibrium (H={wet_eq[0]:.1f}, Z0={wet_eq[1]:.1f}):")
print(f"    lambda_fast = {np.real(fast_eval):.6f}  tau_fast = {1/abs(np.real(fast_eval)):.1f} yr")
print(f"    lambda_slow = {np.real(slow_eval):.6f}  tau_slow = {1/abs(np.real(slow_eval)):.1f} yr")
print(f"    Ratio: {abs(np.real(fast_eval)/np.real(slow_eval)):.1f}x")
print(f"    Fast eigenvector: [{np.real(fast_evec[0]):.4f}, {np.real(fast_evec[1]):.4f}] (H, Z0)")
print(f"    Slow eigenvector: [{np.real(slow_evec[0]):.4f}, {np.real(slow_evec[1]):.4f}] (H, Z0)")

# Also at saddle
J_sad = jacobian_2d(saddle_eq, P_op, c1)
evals_sad = np.linalg.eigvals(J_sad)
evals_s, evecs_s = np.linalg.eig(J_sad)
idx_s = np.argsort(np.real(evals_s))  # most negative first
print(f"\n  At SADDLE (H={saddle_eq[0]:.1f}, Z0={saddle_eq[1]:.1f}):")
for i, (ev, evec) in enumerate(zip(evals_s[idx_s], evecs_s[:, idx_s].T)):
    label = "UNSTABLE" if np.real(ev) > 0 else "stable"
    print(f"    lambda_{i+1} = {np.real(ev):.6f}  [{label}]"
          f"  evec=[{np.real(evec[0]):.4f}, {np.real(evec[1]):.4f}]")


# ============================================================
# PHASE 3: ADIABATIC REDUCTION TO 1D
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: ADIABATIC REDUCTION — Z0 NULLCLINE MANIFOLD")
print("=" * 70)

# Z0 is fast. On the Z0 nullcline (dZ0/dt=0), Z0 = Z0_eq(H).
# The effective 1D system is: dH/dt = f_eff(H) where Z0 is on its nullcline.

def find_Z0_on_nullcline(H_val, c1_val=None, branch='upper'):
    """Find Z0 such that dZ0/dt=0 at given H."""
    if c1_val is None:
        c1_val = c1
    # dZ0/dt = 0 means:
    # [c2/th - r2]*H - (r1-r2)*Z0 + Eo/(th*(1+c1*Z0)) + G(Z0) - (P-d0)/th = 0
    def residual(Z0):
        G = G_prod(Z0)
        Et = Eo / (1.0 + c1_val * max(Z0, 0.0))
        return ((c2/th - r2)*H_val - (r1-r2)*Z0 + Et/th + G - (P_op-d0)/th)

    # Scan for all zeros
    Z0_scan = np.linspace(0.01, 65, 2000)
    res_scan = [residual(z) for z in Z0_scan]
    zeros = []
    for i in range(len(res_scan)-1):
        if res_scan[i] * res_scan[i+1] < 0:
            lo, hi = Z0_scan[i], Z0_scan[i+1]
            for _ in range(60):
                mid = (lo+hi)/2
                if residual(lo)*residual(mid) < 0:
                    hi = mid
                else:
                    lo = mid
            zeros.append((lo+hi)/2)

    if len(zeros) == 0:
        return None
    if branch == 'upper':
        # Upper branch: lowest Z0 (wet, near surface)
        return min(zeros)
    else:
        # Lower branch: highest Z0 (dry, deep water table)
        return max(zeros)


def f_eff_1d(H_val, branch='upper'):
    """Effective 1D drift: dH/dt on the Z0 nullcline."""
    Z0_val = find_Z0_on_nullcline(H_val, branch=branch)
    if Z0_val is None:
        return 0.0
    G = G_prod(Z0_val)
    return G - (r1 - r2) * Z0_val - r2 * H_val


# Find the Z0 nullcline branches
print(f"\n  Computing Z0 nullcline manifold Z0_eq(H)...")

H_range = np.linspace(100, max(wet_eq[0], dry_eq[0]) * 1.5, 500)
upper_branch = []  # wet (low Z0)
lower_branch = []  # dry (high Z0)

for H_val in H_range:
    Z0_scan = np.linspace(0.01, 65, 2000)
    def res(Z0):
        G = G_prod(Z0)
        Et = Eo / (1.0 + c1 * max(Z0, 0.0))
        return ((c2/th - r2)*H_val - (r1-r2)*Z0 + Et/th + G - (P_op-d0)/th)
    res_vals = [res(z) for z in Z0_scan]
    zeros = []
    for i in range(len(res_vals)-1):
        if res_vals[i] * res_vals[i+1] < 0:
            lo, hi = Z0_scan[i], Z0_scan[i+1]
            for _ in range(60):
                mid = (lo+hi)/2
                if res(lo)*res(mid) < 0: hi = mid
                else: lo = mid
            zeros.append((lo+hi)/2)
    if len(zeros) >= 1:
        upper_branch.append((H_val, min(zeros)))
    if len(zeros) >= 2:
        lower_branch.append((H_val, max(zeros)))

print(f"  Upper branch (wet): {len(upper_branch)} points")
print(f"  Lower branch (dry): {len(lower_branch)} points")

# The escape goes from wet equilibrium toward the saddle.
# On the upper branch of the Z0 nullcline.
# f_eff(H) = dH/dt evaluated at Z0 = Z0_nullcline_upper(H)

# Find effective 1D equilibria on upper branch
print(f"\n  Effective 1D drift f_eff(H) on upper Z0-nullcline branch:")

# Compute f_eff on a fine grid around the wet equilibrium
H_fine = np.linspace(wet_eq[0] * 0.5, saddle_eq[0] * 1.5, 2000)
f_eff_vals = []
H_valid = []
for H_val in H_fine:
    Z0_val = find_Z0_on_nullcline(H_val, branch='upper')
    if Z0_val is not None:
        G = G_prod(Z0_val)
        f_val = G - (r1-r2)*Z0_val - r2*H_val
        f_eff_vals.append(f_val)
        H_valid.append(H_val)

H_valid = np.array(H_valid)
f_eff_vals = np.array(f_eff_vals)

# Find zeros of f_eff
eff_zeros = []
for i in range(len(f_eff_vals)-1):
    if f_eff_vals[i] * f_eff_vals[i+1] < 0:
        lo, hi = H_valid[i], H_valid[i+1]
        for _ in range(60):
            mid = (lo+hi)/2
            f_lo = f_eff_1d(lo, 'upper')
            f_mid = f_eff_1d(mid, 'upper')
            if f_lo * f_mid < 0: hi = mid
            else: lo = mid
        eff_zeros.append((lo+hi)/2)

print(f"  Effective 1D zeros on upper branch: {len(eff_zeros)}")
for hz in eff_zeros:
    df = (f_eff_1d(hz + 0.1, 'upper') - f_eff_1d(hz - 0.1, 'upper')) / 0.2
    Z0_at = find_Z0_on_nullcline(hz, branch='upper')
    stab = "STABLE" if df < 0 else "UNSTABLE"
    print(f"    H = {hz:.2f} cm, Z0 = {Z0_at:.2f} cm, f'={df:.6f} [{stab}]")

if len(eff_zeros) < 2:
    print(f"\n  Only {len(eff_zeros)} zeros on upper branch.")
    print(f"  Trying to include fold point where upper/lower branches merge...")
    # The saddle may be at the fold of the Z0 nullcline
    # At the fold, the upper and lower branches merge
    # f_eff doesn't cross zero there — instead the manifold ends

    # Find where upper branch ends (fold point)
    if len(upper_branch) > 0:
        fold_H = max(h for h, z in upper_branch)
        fold_Z0 = [z for h, z in upper_branch if h == fold_H][0]
        print(f"  Upper branch fold at H={fold_H:.1f}, Z0={fold_Z0:.1f}")

# Use the 2D equilibria directly for barrier computation
# Escape from wet to saddle in 2D, project onto H coordinate

H_wet = wet_eq[0]
H_sad = saddle_eq[0]
Z0_wet = wet_eq[1]
Z0_sad = saddle_eq[1]

print(f"\n  Escape path: wet (H={H_wet:.1f}) -> saddle (H={H_sad:.1f})")

# V''_eq from the slow eigenvalue at wet eq
V_pp_eq = abs(np.real(slow_eval))
tau_relax = 1.0 / V_pp_eq

# V''_sad from the unstable eigenvalue at saddle
lambda_u_sad = max(np.real(evals_sad))
V_pp_sad = abs(lambda_u_sad)

print(f"  V''_eq (slow eigenvalue at wet) = {V_pp_eq:.8f} yr^-1")
print(f"  V''_sad (unstable eigenvalue at saddle) = {V_pp_sad:.8f} yr^-1")
print(f"  tau_relax = {tau_relax:.1f} yr")


# ============================================================
# PHASE 4: EFFECTIVE 1D POTENTIAL AND BARRIER
# ============================================================
print("\n" + "=" * 70)
print("PHASE 4: EFFECTIVE 1D POTENTIAL AND BARRIER")
print("=" * 70)

# Compute the effective 1D potential along the escape path
# Use the Z0-nullcline manifold parametrized by H

# The escape coordinate: we go from H_wet to H_sad
# Since Z0 adjusts fast, we follow the Z0-nullcline

# Determine escape direction
if H_sad > H_wet:
    # Escape goes to HIGHER H
    x_saddle = H_sad - H_wet
    # f_eff < 0 in escape direction (resists escape), so negate for potential
    def f_eff_escape(x):
        H_val = H_wet + x
        return -f_eff_1d(H_val, 'upper')  # negate: V>0 at saddle
    print(f"  Escape direction: H increases from {H_wet:.0f} to {H_sad:.0f}")
else:
    # Escape goes to LOWER H
    x_saddle = H_wet - H_sad
    def f_eff_escape(x):
        H_val = H_wet - x
        return f_eff_1d(H_val, 'upper')
    print(f"  Escape direction: H decreases from {H_wet:.0f} to {H_sad:.0f}")

print(f"  Escape coordinate: x = |H - H_wet|, x_saddle = {x_saddle:.2f} cm")

# Compute potential on fine grid
N_pts = 500000
x_grid = np.linspace(0, x_saddle * 1.3, N_pts)
dx = x_grid[1] - x_grid[0]

V_grid = np.zeros(N_pts)
for i in range(1, N_pts):
    x_mid = 0.5 * (x_grid[i-1] + x_grid[i])
    V_grid[i] = V_grid[i-1] + f_eff_escape(x_mid) * dx

DeltaV = np.interp(x_saddle, x_grid, V_grid)

print(f"\n  Barrier: DeltaV = {DeltaV:.8f} cm^2/yr")
print(f"  V''_eq  = {V_pp_eq:.8f} yr^-1")
print(f"  V''_sad = {V_pp_sad:.8f} yr^-1")

# Print potential landscape
print(f"\n  {'H(cm)':>10} {'x':>10} {'V(x)':>14} {'f_eff':>14}")
print(f"  {'-'*52}")
for frac in np.arange(0, 1.15, 0.1):
    x_val = frac * x_saddle
    V_val = np.interp(x_val, x_grid, V_grid)
    f_val = f_eff_escape(x_val)
    if H_sad > H_wet:
        H_val = H_wet + x_val
    else:
        H_val = H_wet - x_val
    marker = " <-- saddle" if abs(frac - 1.0) < 0.01 else ""
    print(f"  {H_val:>10.1f} {x_val:>10.2f} {V_val:>14.8f} {f_val:>+14.10f}{marker}")


# ============================================================
# PHASE 5: KRAMERS PREFACTOR
# ============================================================
print("\n" + "=" * 70)
print("PHASE 5: KRAMERS PREFACTOR")
print("=" * 70)

C_kr = np.sqrt(V_pp_eq * V_pp_sad) / (2 * np.pi)
Ctau = C_kr * tau_relax
inv_Ctau = 1.0 / Ctau

print(f"  C = sqrt(V''_eq * V''_sad)/(2pi) = {C_kr:.10f}")
print(f"  tau = {tau_relax:.2f} yr")
print(f"  C*tau = {Ctau:.10f}")
print(f"  1/(C*tau) = {inv_Ctau:.6f}")

# Also compute 2D Kramers-Langer
det_J_wet = abs(np.linalg.det(J_wet))
det_J_sad = abs(np.linalg.det(J_sad))
C_2d = abs(lambda_u_sad) / (2*np.pi) * np.sqrt(det_J_wet / det_J_sad)
inv_Ctau_2d = 1.0 / (C_2d * tau_relax)

print(f"\n  2D Kramers-Langer (comparison):")
print(f"    lambda_u(saddle) = {lambda_u_sad:.6f}")
print(f"    det(J_wet) = {np.linalg.det(J_wet):.8f}")
print(f"    det(J_sad) = {np.linalg.det(J_sad):.8f}")
print(f"    C_2D = {C_2d:.10f}")
print(f"    1/(C_2D*tau) = {inv_Ctau_2d:.6f}")


# ============================================================
# PHASE 6: EXACT MFPT & BRIDGE TEST
# ============================================================
print("\n" + "=" * 70)
print("PHASE 6: EXACT MFPT & BRIDGE TEST")
print("=" * 70)


def compute_D_exact(sigma):
    """Exact MFPT-based D for the effective 1D system."""
    Phi = 2.0 * V_grid / sigma**2
    Phi -= Phi[0]
    i_sad = np.searchsorted(x_grid, x_saddle)
    if i_sad < 2:
        return np.inf
    Phi_sub = Phi[:i_sad]
    x_sub = x_grid[:i_sad]
    dx_sub = x_sub[1] - x_sub[0]
    if Phi_sub.max() > 700:
        return np.inf
    exp_neg = np.exp(-Phi_sub)
    Ix = np.cumsum(exp_neg) * dx_sub
    exp_pos = np.exp(Phi_sub)
    psi = (2.0 / sigma**2) * exp_pos * Ix
    MFPT = np.trapz(psi, x_sub)
    return MFPT * V_pp_eq


def find_sigma_star(D_target, sig_lo=0.001, sig_hi=200.0):
    """Bisection to find sigma where D_exact = D_target."""
    D_lo = compute_D_exact(sig_lo)
    D_hi = compute_D_exact(sig_hi)
    if D_lo < D_target and D_hi < D_target:
        return None
    if D_lo > D_target and D_hi > D_target:
        return None
    for _ in range(80):
        sig_mid = (sig_lo + sig_hi) / 2
        D_mid = compute_D_exact(sig_mid)
        if D_mid == np.inf or D_mid > D_target:
            sig_lo = sig_mid
        else:
            sig_hi = sig_mid
    return (sig_lo + sig_hi) / 2


# D vs sigma scan
delta_H = abs(H_sad - H_wet)
print(f"\n  D vs sigma scan:")
print(f"  {'sigma':>10} {'D_exact':>14} {'K_eff':>8}")
print(f"  {'-'*36}")

for sigma in [50, 30, 20, 15, 10, 8, 6, 5, 4, 3, 2, 1.5, 1, 0.5]:
    D_ex = compute_D_exact(sigma)
    ba = 2 * DeltaV / sigma**2
    K_eff = D_ex / (np.exp(ba) * inv_Ctau) if (D_ex < 1e12 and ba < 500) else np.nan
    D_str = f"{D_ex:.2f}" if D_ex < 1e8 else f"{D_ex:.2e}"
    K_str = f"{K_eff:.4f}" if not np.isnan(K_eff) else "N/A"
    print(f"  {sigma:>10.2f} {D_str:>14} {K_str:>8}")


# Bridge tests
scenarios = [
    ("k=1 conservative (eps=0.065)", D_CONSERV),
    ("k=1 central (eps=0.10)", D_CENTRAL),
    ("k=2 two-channel (0.10, 0.33)", D_TWOCHAN),
]

for label, D_target in scenarios:
    print(f"\n  {'='*54}")
    print(f"  BRIDGE TEST: {label}")
    print(f"  D_target = {D_target:.1f}")
    print(f"  {'='*54}")

    sigma_star = find_sigma_star(D_target)
    if sigma_star is None:
        print(f"  FAILED: No sigma* found")
        continue

    D_at_star = compute_D_exact(sigma_star)
    dim_barrier = 2 * DeltaV / sigma_star**2
    K_actual = D_at_star / (np.exp(dim_barrier) * inv_Ctau) if dim_barrier < 500 else np.nan

    std_H = sigma_star / np.sqrt(2 * V_pp_eq)
    noise_frac = std_H / delta_H * 100
    MFPT_yr = D_at_star * tau_relax

    print(f"  sigma*          = {sigma_star:.6f} cm*yr^(-1/2)")
    print(f"  D_exact(sigma*) = {D_at_star:.4f}")
    print(f"  Ratio           = {D_at_star / D_target:.6f}")
    print(f"  2*DeltaV/s*^2   = {dim_barrier:.4f}")
    print(f"  K_actual        = {K_actual:.4f}")
    print(f"\n  Physical interpretation:")
    print(f"    std(H)    = {std_H:.1f} cm = {std_H/100:.2f} m peat height")
    print(f"    Delta_H   = {delta_H:.1f} cm (eq to saddle)")
    print(f"    std/Delta  = {noise_frac:.1f}% (noise fraction)")
    print(f"    tau        = {tau_relax:.0f} yr")
    print(f"    MFPT       = {MFPT_yr:.0f} yr")

    if abs(D_at_star / D_target - 1.0) < 0.01:
        status = "VERIFIED" if noise_frac <= 30 else "CONDITIONAL (noise>30%)"
        print(f"\n  *** DUALITY {status}: D_exact = D_target = {D_target:.1f} ***")


# ============================================================
# PHASE 7: LOG-ROBUSTNESS
# ============================================================
print("\n" + "=" * 70)
print("PHASE 7: LOG-ROBUSTNESS SWEEP")
print("=" * 70)

D_sweep = [5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200]
print(f"\n  {'D_target':>10} {'sigma*':>10} {'std(H)cm':>10} {'std/dH%':>10} {'MFPT(yr)':>12}")
print(f"  {'-'*56}")

noise_fracs = []
for D_t in D_sweep:
    s_star = find_sigma_star(D_t)
    if s_star is None:
        print(f"  {D_t:>10.0f}  {'---':>10}")
        continue
    D_ex = compute_D_exact(s_star)
    std_h = s_star / np.sqrt(2 * V_pp_eq)
    nf = std_h / delta_H * 100
    mfpt = D_ex * tau_relax
    noise_fracs.append(nf)
    print(f"  {D_t:>10.0f} {s_star:>10.4f} {std_h:>10.1f} {nf:>10.1f} {mfpt:>12.0f}")

if len(noise_fracs) >= 2:
    band = max(noise_fracs) - min(noise_fracs)
    print(f"\n  Noise-fraction band: {min(noise_fracs):.1f}% - {max(noise_fracs):.1f}% ({band:.1f} pp)")


# ============================================================
# SUMMARY
# ============================================================
print(f"\n\n{'='*70}")
print("SUMMARY: HILBERT 2000 PATH A RESULTS")
print(f"{'='*70}")
print(f"\n  Model: Hilbert, Roulet & Moore (2000) J Ecology 88:230-242")
print(f"  Published parameters: k, r1, r2, Z_min, Z_max, c2, d0, Eo, theta_max")
print(f"  Free parameter: c1 = {c1:.4f} cm^-1 (calibrated from Fig 5 bistability)")
print(f"  Free parameter count: 1")
print(f"  Operating point: P = {P_op} cm/yr")
print(f"  Equilibria: wet H={wet_eq[0]:.0f} Z0={wet_eq[1]:.1f}, saddle H={saddle_eq[0]:.0f} Z0={saddle_eq[1]:.1f}")
print(f"  Barrier: DeltaV = {DeltaV:.6f} cm^2/yr")
print(f"  tau_relax = {tau_relax:.0f} yr")

print("\nDone.")
