#!/usr/bin/env python3
"""
Phase 2: Compute quasi-potential barrier via Minimum Action Method (MAM).
For 2D non-gradient system, use Freidlin-Wentzell action integral.

Approach: String method - discretize path from clear→saddle, minimize action.
The quasi-potential barrier ΔΦ = min_path ∫ ½|φ̇ - f(φ)|² dt

For a path ending at the saddle point (which is on the separatrix),
ΔΦ = ∫ along the instanton (optimal fluctuation path).

For systems with additive noise: ΔΦ = ∫₀^∞ |φ̇|² dt along the "reverse"
trajectory following the time-reversed drift.

Simpler approach for barrier estimation:
- Use heteroclinic orbit of the reversed dynamics: dx/dt = -f(x)
- Starting from the saddle, integrate backwards to find the path to the equilibrium
- The action along this path gives ΔΦ = -∫ f·dx along the heteroclinic
"""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize

r_E = 0.1
r_V = 0.05
E_0 = 5.0
p = 4

def drift(state, h_V, h_E):
    E, V = state
    dE = r_E * (E_0 * h_V**p / (h_V**p + V**p) - E)
    dV = r_V * (h_E**p / (h_E**p + E**p) - V)
    return np.array([dE, dV])

# ============================================================
# Method 1: Geometric minimum action method
# For additive isotropic noise, ΔΦ = ∫_path (f · dn) where
# the integral is along the optimal path and dn is the
# "normal" component.
#
# More precisely, for the quasi-potential:
# Φ(x) = min over paths from x_eq to x of: ∫ ½|φ̇ - f|² dt
#
# At the saddle: Φ(x_sad) = ΔΦ = the barrier height.
#
# For the Geometric Action (Heymann & Vanden-Eijnden 2008):
# S_geom = ∫ |f⊥(φ)| |dφ| + ∫ (f_s - |f⊥|²/|f_s|) ds ... complex.
#
# Simpler: use the "quasipotential via line integral" for
# gradient-like systems: Φ(x) ≈ -∫ f · dx along steepest path.
# For non-gradient systems, this gives an approximation.
# ============================================================

# Method 2: Numerical action minimization via string method
def compute_action_on_path(path_points, h_V, h_E, T=100):
    """
    Compute the Freidlin-Wentzell action along a parameterized path.
    path_points: Nx2 array of (E,V) coordinates
    Returns the geometric action.
    """
    N = len(path_points)
    total_action = 0.0

    for i in range(N-1):
        p0 = path_points[i]
        p1 = path_points[i+1]
        dp = p1 - p0
        ds = np.linalg.norm(dp)
        if ds < 1e-15:
            continue

        # Midpoint
        mid = 0.5 * (p0 + p1)
        f = drift(mid, h_V, h_E)

        # Tangent direction
        t_hat = dp / ds

        # Drift component perpendicular to path
        f_perp = f - np.dot(f, t_hat) * t_hat
        f_perp_norm = np.linalg.norm(f_perp)

        # Geometric action contribution (Heymann & Vanden-Eijnden)
        # For additive noise with D = σ²/2 * I:
        # S_geom = ∫ |f⊥| ds  (leading order)
        total_action += f_perp_norm * ds

        # Alternative: Onsager-Machlup/FW action component
        # ½|φ̇ - f|² dt where φ̇ ≈ dp/dt, approximate as:
        # Need velocity... use geometric action instead

    return total_action

# Method 3: Use the Onsager-Machlup relation
# For the quasi-potential with isotropic additive noise σ:
# Φ(x) = min_{path from x_eq to x} ∫ ½|φ̇ - b(φ)|² dt
# where b is the drift.
#
# At the saddle, the optimal path arrives along the unstable manifold
# of the adjoint dynamics. We can compute this numerically.
def compute_quasipotential_via_adjoint(h_V, h_E, E_eq, V_eq, E_sad, V_sad):
    """
    Compute ΔΦ by integrating the adjoint dynamics from the saddle.

    The "adjoint" or "reversed" dynamics is: dx/dt = +f(x) for the quasi-potential
    (i.e., follow the FORWARD drift from the saddle towards the equilibrium,
    but this goes AWAY from the saddle in the stable direction).

    Actually for the quasi-potential:
    Φ(x) = ∫ along the heteroclinic orbit of the "doubled" system

    For practical computation:
    - Linearize at saddle to find unstable eigenvector of the REVERSED dynamics
    - The reversed dynamics near saddle: the unstable direction of x' = -f(x)
      is the STABLE direction of x' = f(x)
    - Integrate x' = -f(x) starting from x_sad + ε*v_stable
    - This gives the optimal path from saddle back to equilibrium
    - ΔΦ = ∫ along this path of ½|f|² dt ... no wait.

    Actually the correct relation for additive noise:
    Φ(x) = -∫_γ f · dx  where γ is a "gradient path"
    This only works for gradient systems.

    For non-gradient: use the Maier-Stein decomposition
    f = -∇Φ + Q  where Q·∇Φ = 0  (orthogonal decomposition)
    Then Φ(x_sad) = -∫_{eq}^{sad} (-∇Φ) · dx = ∫|∇Φ|² ds/|f| along the instanton

    The practical method: iterate between path optimization and action computation.
    """
    # Compute Jacobian at saddle
    eps = 1e-7
    f0 = drift([E_sad, V_sad], h_V, h_E)
    J = np.zeros((2,2))
    for i, (dE, dV) in enumerate([(eps, 0), (0, eps)]):
        f1 = drift([E_sad + dE, V_sad + dV], h_V, h_E)
        J[:, i] = (f1 - f0) / eps

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eig(J)
    # The stable eigenvector of f' = f gives the unstable direction of f' = -f
    idx_stable = np.argmin(eigvals.real)  # most negative = stable of forward
    v_stable = eigvecs[:, idx_stable].real

    # Integrate REVERSED dynamics from near saddle in stable direction
    # x' = -f(x), starting from x_sad + small perturbation along stable eigenvector
    pert = 1e-4 * v_stable / np.linalg.norm(v_stable)

    # Two directions along the stable manifold
    results = []
    for sign in [+1, -1]:
        x0 = np.array([E_sad, V_sad]) + sign * pert

        def reversed_drift(t, state):
            return -drift(state, h_V, h_E)

        # Integrate until we approach the equilibrium
        def near_eq(t, state):
            dist = np.sqrt((state[0] - E_eq)**2 + (state[1] - V_eq)**2)
            return dist - 0.001  # stop when within 0.001 of eq
        near_eq.terminal = True

        try:
            sol = solve_ivp(reversed_drift, [0, 10000], x0,
                          events=near_eq, max_step=0.1, rtol=1e-10, atol=1e-12)

            if sol.t_events[0].size > 0:
                # Compute action along this path
                # ΔΦ = ∫ ½|ẋ + f(x)|² dt = ∫ ½|(-f) + f|² dt = 0 ... NO
                # Wait, that's the reversed dynamics, so ẋ = -f, and
                # action = ∫ ½|ẋ - f|² dt = ∫ ½|-f - f|² dt = ∫ 2|f|² dt
                # That's not right either.

                # The correct approach: the quasi-potential is given by
                # Φ(x_sad) = ∫₀^T ½|ẋ_inst - f(x_inst)|² dt
                # where x_inst is the instanton (optimal path from eq to saddle).
                #
                # The instanton satisfies: ẋ = f(x) + ∇Φ ... from the Euler-Lagrange eq.
                # In the case of gradient systems: ẋ = -f(x), so action = ∫ 2|f|² dt.
                # For non-gradient: need to use the actual MAM.

                # For now: compute -∫ f · dx along the reversed path (gradient estimate)
                path = sol.y.T  # shape (n_steps, 2)
                action = 0.0
                for i in range(len(path) - 1):
                    dp = path[i+1] - path[i]
                    mid = 0.5 * (path[i] + path[i+1])
                    f = drift(mid, h_V, h_E)
                    action += -np.dot(f, dp)  # -f · dx

                results.append({
                    'sign': sign,
                    'final_pos': path[-1],
                    'dist_to_eq': np.linalg.norm(path[-1] - np.array([E_eq, V_eq])),
                    'action_line_integral': action,
                    'path_length': len(path),
                    'path': path,
                })
        except Exception as e:
            results.append({'sign': sign, 'error': str(e)})

    return results

# ============================================================
# Method 4: Direct Freidlin-Wentzell action minimization
# Parameterize path as N points, optimize their positions
# ============================================================
def minimize_action(h_V, h_E, E_eq, V_eq, E_sad, V_sad, N_points=100, T=200):
    """Minimize the Freidlin-Wentzell action over paths from eq to saddle."""
    # Initialize path as straight line
    path = np.zeros((N_points, 2))
    for i in range(N_points):
        t = i / (N_points - 1)
        path[i] = [(1-t)*E_eq + t*E_sad, (1-t)*V_eq + t*V_sad]

    def action_func(flat_path):
        # Endpoints are fixed
        p = np.zeros((N_points, 2))
        p[0] = [E_eq, V_eq]
        p[-1] = [E_sad, V_sad]
        p[1:-1] = flat_path.reshape(-1, 2)

        dt = T / (N_points - 1)
        total = 0.0
        for i in range(N_points - 1):
            vel = (p[i+1] - p[i]) / dt
            mid = 0.5 * (p[i] + p[i+1])
            f = drift(mid, h_V, h_E)
            diff = vel - f
            total += 0.5 * np.dot(diff, diff) * dt
        return total

    # Optimize interior points
    x0 = path[1:-1].flatten()

    # Try multiple T values to find the one that minimizes action
    best_action = np.inf
    best_T = None

    for T_try in [50, 100, 200, 500, 1000, 2000]:
        def action_T(flat_path, T=T_try):
            p = np.zeros((N_points, 2))
            p[0] = [E_eq, V_eq]
            p[-1] = [E_sad, V_sad]
            p[1:-1] = flat_path.reshape(-1, 2)

            dt = T / (N_points - 1)
            total = 0.0
            for i in range(N_points - 1):
                vel = (p[i+1] - p[i]) / dt
                mid = 0.5 * (p[i] + p[i+1])
                f = drift(mid, h_V, h_E)
                diff = vel - f
                total += 0.5 * np.dot(diff, diff) * dt
            return total

        res = minimize(action_T, x0, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-12})
        if res.fun < best_action:
            best_action = res.fun
            best_T = T_try
            best_path = np.zeros((N_points, 2))
            best_path[0] = [E_eq, V_eq]
            best_path[-1] = [E_sad, V_sad]
            best_path[1:-1] = res.x.reshape(-1, 2)

    return best_action, best_T, best_path

# ============================================================
# Run computations for baseline and all parameter sweeps
# ============================================================
print("=" * 70)
print("PHASE 2: QUASI-POTENTIAL BARRIER VIA MINIMUM ACTION METHOD")
print("=" * 70)

# Baseline parameters
cases = [
    # (h_V, h_E, E_cl, V_cl, E_sad, V_sad, label)
    (0.10, 2.0, 0.0005, 1.0000, 3.7293, 0.0764, "h_V=0.10"),
    (0.15, 2.0, 0.0025, 1.0000, 3.2219, 0.1293, "h_V=0.15"),
    (0.20, 2.0, 0.0080, 1.0000, 2.8999, 0.1845, "h_V=0.20"),
    (0.30, 2.0, 0.0402, 1.0000, 2.4658, 0.3021, "h_V=0.30"),
    (0.50, 2.0, 0.2946, 0.9995, 1.8653, 0.5693, "h_V=0.50 (baseline)"),
    # h_E sweep
    (0.50, 1.0, 0.3036, 0.9916, 0.7320, 0.7770, "h_E=1.0"),
    (0.50, 1.5, 0.2958, 0.9985, 1.2812, 0.6526, "h_E=1.5"),
    (0.50, 2.0, 0.2946, 0.9995, 1.8653, 0.5693, "h_E=2.0 (baseline)"),
    (0.50, 2.5, 0.2943, 0.9998, 2.5000, 0.5000, "h_E=2.5"),
    (0.50, 3.0, 0.2942, 0.9999, 3.2132, 0.4318, "h_E=3.0"),
]

print("\n--- Method: Adjoint/reversed dynamics line integral ---")
print(f"{'Label':<25} {'Action(+)':<15} {'Action(-)':<15} {'Dist(+)':<10} {'Dist(-)':<10}")
print("-" * 75)

all_barriers = {}

for h_V, h_E, E_cl, V_cl, E_sad, V_sad, label in cases:
    results = compute_quasipotential_via_adjoint(h_V, h_E, E_cl, V_cl, E_sad, V_sad)

    acts = []
    for r in results:
        if 'action_line_integral' in r:
            acts.append(r)

    if len(acts) == 2:
        print(f"{label:<25} {acts[0]['action_line_integral']:<15.8f} {acts[1]['action_line_integral']:<15.8f} "
              f"{acts[0]['dist_to_eq']:<10.6f} {acts[1]['dist_to_eq']:<10.6f}")
        # Use the one that reaches closer to the equilibrium
        best = min(acts, key=lambda x: x['dist_to_eq'])
        all_barriers[label] = best['action_line_integral']
    elif len(acts) == 1:
        print(f"{label:<25} {acts[0]['action_line_integral']:<15.8f} {'N/A':<15} "
              f"{acts[0]['dist_to_eq']:<10.6f} {'N/A':<10}")
        all_barriers[label] = acts[0]['action_line_integral']
    else:
        print(f"{label:<25} FAILED")

print("\n--- Method: Direct action minimization (MAM) ---")
print(f"{'Label':<25} {'ΔΦ (action)':<15} {'Best T':<10}")
print("-" * 50)

mam_barriers = {}
for h_V, h_E, E_cl, V_cl, E_sad, V_sad, label in cases:
    try:
        action, T_opt, path_opt = minimize_action(h_V, h_E, E_cl, V_cl, E_sad, V_sad)
        print(f"{label:<25} {action:<15.8f} {T_opt:<10}")
        mam_barriers[label] = action
    except Exception as e:
        print(f"{label:<25} FAILED: {e}")

# ============================================================
# Summary: barriers vs feedback parameters
# ============================================================
print("\n" + "=" * 70)
print("BARRIER SUMMARY")
print("=" * 70)

print("\n--- h_V sweep (macrophyte→turbidity) ---")
print(f"{'h_V':<8} {'ΔΦ_adjoint':<15} {'ΔΦ_MAM':<15} {'ln(ΔΦ_MAM)':<15}")
print("-" * 55)
h_V_barriers = []
for h_V, h_E, E_cl, V_cl, E_sad, V_sad, label in cases:
    if h_E == 2.0:  # h_V sweep
        adj = all_barriers.get(label, float('nan'))
        mam = mam_barriers.get(label, float('nan'))
        if not np.isnan(mam):
            print(f"{h_V:<8.2f} {adj:<15.8f} {mam:<15.8f} {np.log(mam):<15.6f}")
            h_V_barriers.append((h_V, mam))
        else:
            print(f"{h_V:<8.2f} {adj:<15.8f} {'N/A':<15}")

print("\n--- h_E sweep (turbidity→macrophyte) ---")
print(f"{'h_E':<8} {'ΔΦ_adjoint':<15} {'ΔΦ_MAM':<15} {'ln(ΔΦ_MAM)':<15}")
print("-" * 55)
h_E_barriers = []
for h_V, h_E, E_cl, V_cl, E_sad, V_sad, label in cases:
    if h_V == 0.50:  # h_E sweep
        adj = all_barriers.get(label, float('nan'))
        mam = mam_barriers.get(label, float('nan'))
        if not np.isnan(mam):
            print(f"{h_E:<8.1f} {adj:<15.8f} {mam:<15.8f} {np.log(mam):<15.6f}")
            h_E_barriers.append((h_E, mam))
        else:
            print(f"{h_E:<8.1f} {adj:<15.8f} {'N/A':<15}")

# Power-law fits
if len(h_V_barriers) >= 3:
    h_Vs = np.array([x[0] for x in h_V_barriers])
    deltas = np.array([x[1] for x in h_V_barriers])
    coeffs = np.polyfit(np.log(h_Vs), np.log(deltas), 1)
    print(f"\nPower law fit: ΔΦ ∝ h_V^{coeffs[0]:.3f}")
    print(f"  (R² would need scipy, but slope = {coeffs[0]:.3f})")

if len(h_E_barriers) >= 3:
    h_Es = np.array([x[0] for x in h_E_barriers])
    deltas = np.array([x[1] for x in h_E_barriers])
    coeffs = np.polyfit(np.log(h_Es), np.log(deltas), 1)
    print(f"\nPower law fit: ΔΦ ∝ h_E^{coeffs[0]:.3f}")

# ============================================================
# Test: does -ln(ε_candidate) ∝ ΔΦ?
# ============================================================
print("\n" + "=" * 70)
print("TEST: Does -ln(ε) ∝ ΔΦ?")
print("=" * 70)

if h_V_barriers:
    print("\nh_V sweep with ε = h_V^p / (1 + h_V^p):")
    print(f"{'h_V':<8} {'ε':<12} {'-ln(ε)':<12} {'ΔΦ':<15} {'ΔΦ/(-ln ε)':<15}")
    print("-" * 62)
    for h_V, dphi in h_V_barriers:
        eps = h_V**p / (1 + h_V**p)
        neg_ln_eps = -np.log(eps)
        ratio = dphi / neg_ln_eps if neg_ln_eps > 0 else float('nan')
        print(f"{h_V:<8.2f} {eps:<12.8f} {neg_ln_eps:<12.4f} {dphi:<15.8f} {ratio:<15.8f}")

if h_E_barriers:
    print("\nh_E sweep with ε = 1 / (1 + h_E^p):")
    print(f"{'h_E':<8} {'ε':<12} {'-ln(ε)':<12} {'ΔΦ':<15} {'ΔΦ/(-ln ε)':<15}")
    print("-" * 62)
    for h_E, dphi in h_E_barriers:
        eps = 1.0 / (1 + h_E**p)
        neg_ln_eps = -np.log(eps)
        ratio = dphi / neg_ln_eps if neg_ln_eps > 0 else float('nan')
        print(f"{h_E:<8.1f} {eps:<12.8f} {neg_ln_eps:<12.4f} {dphi:<15.8f} {ratio:<15.8f}")

print("\nDone.")
