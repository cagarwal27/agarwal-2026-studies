"""
Conway-Perelson (2015) HIV Post-Treatment Control Model
========================================================
5D ODE: (T, L, I, V, E) = (target cells, latent reservoir, infected cells, virus, CTL effectors)
All parameters from PNAS 2015;112(17):5467-72, SI Table S1.

Post-treatment: eps = 0 (no drugs).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# ============================================================
# Parameters (SI Table S1)
# ============================================================
PARAMS = dict(
    # Target cells
    lam     = 10000.0,    # cells/mL/day  (lambda)
    dT      = 0.01,       # day^-1
    beta    = 1.5e-8,     # mL/day
    # Infected cells
    delta   = 1.0,        # day^-1
    N_burst = 2000,       # burst size
    # Virus
    c       = 23.0,       # day^-1
    # Latent reservoir
    a       = 0.001,      # day^-1  (reactivation rate)
    dL      = 0.004,      # day^-1
    rho     = 0.0045,     # day^-1  (proliferation)
    alpha_L = 1e-6,       # fraction entering latency
    # CTL effectors
    lam_E   = 1.0,        # cell/mL/day
    bE      = 1.0,        # day^-1
    KB      = 0.1,        # cells/mL  (half-sat for stimulation)
    dE      = 2.0,        # day^-1
    KD      = 5.0,        # cells/mL  (half-sat for exhaustion)
    mu      = 2.0,        # day^-1  (CTL turnover)
    m       = 0.42,       # mL/cells/day  (killing rate, bifurcation param)
    # Treatment
    eps     = 0.0,        # post-treatment: no drugs
)
# Derived
PARAMS['p'] = PARAMS['N_burst'] * PARAMS['delta']  # = 2000 day^-1


def rhs(t, y, params=None):
    """Right-hand side of the Conway-Perelson 5D ODE system."""
    if params is None:
        params = PARAMS
    T, L, I, V, E = y

    lam = params['lam']
    dT = params['dT']
    beta = params['beta']
    eps = params['eps']
    alpha_L = params['alpha_L']
    rho = params['rho']
    a = params['a']
    dL = params['dL']
    delta = params['delta']
    p = params['p']
    c = params['c']
    m = params['m']
    lam_E = params['lam_E']
    bE = params['bE']
    KB = params['KB']
    dE = params['dE']
    KD = params['KD']
    mu = params['mu']

    infection = (1 - eps) * beta * V * T

    dTdt = lam - dT * T - infection
    dLdt = alpha_L * infection + (rho - a - dL) * L
    dIdt = (1 - alpha_L) * infection - delta * I + a * L - m * E * I
    dVdt = p * I - c * V
    dEdt = lam_E + bE * I / (KB + I) * E - dE * I / (KD + I) * E - mu * E

    return [dTdt, dLdt, dIdt, dVdt, dEdt]


def jacobian_matrix(y, params=None):
    """Compute the 5x5 Jacobian matrix at state y."""
    if params is None:
        params = PARAMS
    T, L, I, V, E = y

    lam = params['lam']
    dT = params['dT']
    beta = params['beta']
    eps = params['eps']
    alpha_L = params['alpha_L']
    rho = params['rho']
    a = params['a']
    dL = params['dL']
    delta = params['delta']
    p = params['p']
    c = params['c']
    m = params['m']
    lam_E = params['lam_E']
    bE = params['bE']
    KB = params['KB']
    dE = params['dE']
    KD = params['KD']
    mu = params['mu']

    bVT = (1 - eps) * beta

    J = np.zeros((5, 5))

    # dT/dt = lam - dT*T - bVT*V*T
    J[0, 0] = -dT - bVT * V          # d(dT/dt)/dT
    J[0, 3] = -bVT * T               # d(dT/dt)/dV

    # dL/dt = alpha_L * bVT*V*T + (rho - a - dL)*L
    J[1, 0] = alpha_L * bVT * V      # d(dL/dt)/dT
    J[1, 1] = rho - a - dL           # d(dL/dt)/dL
    J[1, 3] = alpha_L * bVT * T      # d(dL/dt)/dV

    # dI/dt = (1 - alpha_L)*bVT*V*T - delta*I + a*L - m*E*I
    J[2, 0] = (1 - alpha_L) * bVT * V    # d(dI/dt)/dT
    J[2, 1] = a                            # d(dI/dt)/dL
    J[2, 2] = -delta - m * E              # d(dI/dt)/dI
    J[2, 3] = (1 - alpha_L) * bVT * T    # d(dI/dt)/dV
    J[2, 4] = -m * I                      # d(dI/dt)/dE

    # dV/dt = p*I - c*V
    J[3, 2] = p                        # d(dV/dt)/dI
    J[3, 3] = -c                       # d(dV/dt)/dV

    # dE/dt = lam_E + bE*I/(KB+I)*E - dE*I/(KD+I)*E - mu*E
    # d/dI [bE*I/(KB+I)*E] = bE*KB/(KB+I)^2 * E
    # d/dI [dE*I/(KD+I)*E] = dE*KD/(KD+I)^2 * E
    # d/dE [bE*I/(KB+I)*E] = bE*I/(KB+I)
    # d/dE [dE*I/(KD+I)*E] = dE*I/(KD+I)
    J[4, 2] = bE * KB / (KB + I)**2 * E - dE * KD / (KD + I)**2 * E   # d(dE/dt)/dI
    J[4, 4] = bE * I / (KB + I) - dE * I / (KD + I) - mu              # d(dE/dt)/dE

    return J


def find_fixed_points(params=None, verbose=False):
    """
    Find all fixed points of the system by solving RHS = 0.

    Strategy: At steady state, V = p*I/c. Substitute into the remaining equations
    and solve the 4D algebraic system with multiple initial guesses.

    Returns list of (y, eigenvalues, stability_type) tuples.
    """
    if params is None:
        params = PARAMS

    lam = params['lam']
    dT = params['dT']
    beta = params['beta']
    eps = params['eps']
    alpha_L = params['alpha_L']
    rho = params['rho']
    a_rate = params['a']
    dL = params['dL']
    delta = params['delta']
    p = params['p']
    c = params['c']
    m_val = params['m']
    lam_E = params['lam_E']
    bE = params['bE']
    KB = params['KB']
    dE = params['dE']
    KD = params['KD']
    mu = params['mu']

    def equations(x):
        T, L, I, E = x
        V = p * I / c
        infection = (1 - eps) * beta * V * T

        eq1 = lam - dT * T - infection
        eq2 = alpha_L * infection + (rho - a_rate - dL) * L
        eq3 = (1 - alpha_L) * infection - delta * I + a_rate * L - m_val * E * I
        eq4 = lam_E + bE * I / (KB + I) * E - dE * I / (KD + I) * E - mu * E

        return [eq1, eq2, eq3, eq4]

    # Disease-free equilibrium guess
    T0 = lam / dT  # = 10^6
    E0 = lam_E / mu  # = 0.5

    # Multiple initial guesses spanning many orders of magnitude
    # Densely cover the PTC-saddle-VR landscape for all m values
    guesses = [
        # Disease-free / PTC (very low I) — covers m=0.42 to m=0.80
        [T0, 0.01, 1e-6, E0],
        [T0, 0.1, 1e-5, E0],
        [T0, 1.0, 1e-4, 1.0],
        [T0, 0.001, 1e-7, 0.5],
        [T0, 0.001, 1e-3, 0.7],
        [T0, 0.001, 0.01, 0.65],
        [T0, 0.001, 0.05, 0.63],
        [T0, 0.001, 0.1, 0.60],
        [T0, 0.001, 0.3, 0.72],
        [T0 * 0.99, 0.01, 1e-3, 2.0],
        [T0 * 0.95, 0.1, 1e-2, 5.0],
        [T0 * 0.999, 0.0005, 0.02, 0.65],
        [T0 * 0.9999, 0.0001, 0.005, 0.70],
        # Intermediate (saddle) — wider range for different m
        [T0 * 0.5, 10.0, 1.0, 5.0],
        [T0 * 0.8, 1.0, 0.1, 3.0],
        [T0 * 0.3, 5.0, 0.5, 10.0],
        [T0 * 0.7, 2.0, 0.5, 2.0],
        [T0 * 0.6, 50.0, 5.0, 8.0],
        [T0 * 0.999, 0.002, 0.8, 0.72],
        [T0 * 0.998, 0.005, 2.0, 0.65],
        [T0 * 0.997, 0.008, 3.0, 0.60],
        [T0 * 0.995, 0.01, 5.0, 0.55],
        [T0 * 0.99, 0.02, 10.0, 0.50],
        [T0 * 0.98, 0.05, 20.0, 0.45],
        [T0 * 0.95, 0.1, 50.0, 0.40],
        [T0 * 0.9, 0.5, 100.0, 0.38],
        # High viremia (VR)
        [T0 * 0.1, 100.0, 100.0, 20.0],
        [T0 * 0.05, 200.0, 500.0, 50.0],
        [T0 * 0.01, 50.0, 1000.0, 100.0],
        [T0 * 0.02, 500.0, 200.0, 30.0],
        [T0 * 0.005, 100.0, 2000.0, 50.0],
        [T0 * 0.15, 300.0, 50.0, 15.0],
        [T0 * 0.08, 150.0, 300.0, 40.0],
        [T0 * 0.2, 1.0, 400.0, 0.35],
        [T0 * 0.5, 5.0, 200.0, 0.34],
        [T0 * 0.3, 2.0, 600.0, 0.33],
    ]

    found = []
    tol = 1e-8

    for guess in guesses:
        try:
            sol = fsolve(equations, guess, full_output=True, maxfev=10000)
            x_sol, info, ier, msg = sol
            T_s, L_s, I_s, E_s = x_sol
            V_s = p * I_s / c

            # Check solution validity
            if ier != 1:
                continue
            residual = np.max(np.abs(info['fvec']))
            if residual > tol:
                continue
            if T_s < 0 or L_s < 0 or I_s < 0 or E_s < 0:
                continue

            y_sol = np.array([T_s, L_s, I_s, V_s, E_s])

            # Check if this is a new fixed point
            is_new = True
            for y_prev, _, _ in found:
                if np.allclose(y_sol, y_prev, rtol=1e-4):
                    is_new = False
                    break
            if not is_new:
                continue

            # Compute eigenvalues
            J = jacobian_matrix(y_sol, params)
            eigs = np.linalg.eigvals(J)

            # Classify stability
            real_parts = np.real(eigs)
            n_positive = np.sum(real_parts > 1e-10)
            n_negative = np.sum(real_parts < -1e-10)

            if n_positive == 0:
                stability = 'stable'
            elif n_positive == 1:
                stability = 'saddle'
            else:
                stability = f'unstable ({n_positive} positive)'

            found.append((y_sol, eigs, stability))

            if verbose:
                print(f"  Found {stability}: T={T_s:.2f}, L={L_s:.6f}, "
                      f"I={I_s:.6f}, V={V_s:.6f}, E={E_s:.4f}")
                print(f"    Eigenvalues: {np.sort(np.real(eigs))}")

        except Exception:
            continue

    # Sort by viral load (V = y[3])
    found.sort(key=lambda x: x[0][3])
    return found


def simulate(y0, t_span, params=None, t_eval=None, max_step=0.1):
    """Simulate the ODE forward in time."""
    if params is None:
        params = PARAMS

    sol = solve_ivp(
        lambda t, y: rhs(t, y, params),
        t_span, y0,
        method='RK45',
        t_eval=t_eval,
        max_step=max_step,
        rtol=1e-10, atol=1e-12
    )
    return sol


def get_initial_conditions(L0_per_million, params=None):
    """
    Get initial conditions for post-treatment simulation.

    L0_per_million: latent cells per 10^6 CD4+ T cells

    At treatment interruption:
    - T near disease-free: T ~ lambda/dT = 10^6 cells/mL
    - I, V suppressed to very low levels
    - E at baseline: E ~ lambda_E/mu
    - L = L0_per_million * T / 10^6
    """
    if params is None:
        params = PARAMS

    T0 = params['lam'] / params['dT']  # 10^6
    E0 = params['lam_E'] / params['mu']  # 0.5
    L0 = L0_per_million * T0 / 1e6  # convert per-million to actual
    I0 = 1e-6  # trace infected cells
    V0 = params['p'] * I0 / params['c']  # quasi-steady state

    return np.array([T0, L0, I0, V0, E0])


if __name__ == "__main__":
    print("Conway-Perelson (2015) HIV Post-Treatment Control Model")
    print("=" * 60)
    print(f"\nParameters:")
    for k, v in PARAMS.items():
        print(f"  {k:>10} = {v}")

    print(f"\nFinding fixed points at m = {PARAMS['m']}...")
    fps = find_fixed_points(verbose=True)
    print(f"\nFound {len(fps)} fixed points.")
