"""
Noise-Source Mapping: HIV CLE Test (Phase 4)
=============================================
Tests whether the Chemical Langevin Equation noise at the physical blood
volume (Omega = 5000 mL) predicts D_Kramers = D_product = 1/epsilon_CTL.

All Conway-Perelson (2015) parameters inline. No external imports beyond
numpy/scipy.

Output: sigma_CLE vs sigma*, Omega* vs Omega_physical, dominant noise
source decomposition, Kramers difficulty comparison.
"""

import numpy as np
from scipy.optimize import fsolve

# ============================================================
# Conway-Perelson (2015) parameters (SI Table S1)
# ============================================================
PARAMS = dict(
    lam     = 10000.0,    # cells/mL/day
    dT      = 0.01,       # day^-1
    beta    = 1.5e-8,     # mL/day
    delta   = 1.0,        # day^-1
    N_burst = 2000,
    p       = 2000.0,     # = N_burst * delta
    c       = 23.0,       # day^-1
    a       = 0.001,      # day^-1  (latent reactivation)
    dL      = 0.004,      # day^-1
    rho     = 0.0045,     # day^-1  (latent proliferation)
    alpha_L = 1e-6,       # fraction entering latency
    lam_E   = 1.0,        # cell/mL/day
    bE      = 1.0,        # day^-1
    KB      = 0.1,        # cells/mL
    dE      = 2.0,        # day^-1
    KD      = 5.0,        # cells/mL
    mu      = 2.0,        # day^-1
    m       = 0.42,       # mL/cells/day  (CTL killing rate)
    eps     = 0.0,        # post-treatment: no drugs
)


# ============================================================
# ODE system
# ============================================================

def rhs(y, params=PARAMS):
    """5D RHS: (T, L, I, V, E)."""
    T, L, I, V, E = y
    p = params
    infection = (1 - p['eps']) * p['beta'] * V * T
    return np.array([
        p['lam'] - p['dT'] * T - infection,
        p['alpha_L'] * infection + (p['rho'] - p['a'] - p['dL']) * L,
        (1 - p['alpha_L']) * infection - p['delta'] * I + p['a'] * L - p['m'] * E * I,
        p['p'] * I - p['c'] * V,
        p['lam_E'] + p['bE'] * I / (p['KB'] + I) * E
        - p['dE'] * I / (p['KD'] + I) * E - p['mu'] * E,
    ])


def jacobian(y, params=PARAMS):
    """5x5 Jacobian at state y."""
    T, L, I, V, E = y
    p = params
    bVT = (1 - p['eps']) * p['beta']
    J = np.zeros((5, 5))
    J[0, 0] = -p['dT'] - bVT * V
    J[0, 3] = -bVT * T
    J[1, 0] = p['alpha_L'] * bVT * V
    J[1, 1] = p['rho'] - p['a'] - p['dL']
    J[1, 3] = p['alpha_L'] * bVT * T
    J[2, 0] = (1 - p['alpha_L']) * bVT * V
    J[2, 1] = p['a']
    J[2, 2] = -p['delta'] - p['m'] * E
    J[2, 3] = (1 - p['alpha_L']) * bVT * T
    J[2, 4] = -p['m'] * I
    J[3, 2] = p['p']
    J[3, 3] = -p['c']
    J[4, 2] = (p['bE'] * p['KB'] / (p['KB'] + I)**2 * E
               - p['dE'] * p['KD'] / (p['KD'] + I)**2 * E)
    J[4, 4] = (p['bE'] * I / (p['KB'] + I)
               - p['dE'] * I / (p['KD'] + I) - p['mu'])
    return J


# ============================================================
# Fixed-point finder
# ============================================================

def find_fixed_points(params=PARAMS):
    """Find and classify all fixed points at given parameters."""
    T0 = params['lam'] / params['dT']
    E0 = params['lam_E'] / params['mu']

    guesses = [
        [T0, 0.01, 1e-6, E0],
        [T0, 0.1, 1e-5, E0],
        [T0, 1.0, 1e-4, 1.0],
        [T0, 0.001, 1e-3, 0.7],
        [T0, 0.001, 0.05, 0.63],
        [T0, 0.001, 0.3, 0.72],
        [T0 * 0.99, 0.01, 1e-3, 2.0],
        [T0 * 0.95, 0.1, 1e-2, 5.0],
        [T0 * 0.999, 0.0005, 0.02, 0.65],
        [T0 * 0.9999, 0.0001, 0.005, 0.70],
        [T0 * 0.5, 10.0, 1.0, 5.0],
        [T0 * 0.8, 1.0, 0.1, 3.0],
        [T0 * 0.3, 5.0, 0.5, 10.0],
        [T0 * 0.7, 2.0, 0.5, 2.0],
        [T0 * 0.999, 0.002, 0.8, 0.72],
        [T0 * 0.998, 0.005, 2.0, 0.65],
        [T0 * 0.995, 0.01, 5.0, 0.55],
        [T0 * 0.99, 0.02, 10.0, 0.50],
        [T0 * 0.98, 0.05, 20.0, 0.45],
        [T0 * 0.95, 0.1, 50.0, 0.40],
        [T0 * 0.9, 0.5, 100.0, 0.38],
        [T0 * 0.1, 100.0, 100.0, 20.0],
        [T0 * 0.05, 200.0, 500.0, 50.0],
        [T0 * 0.01, 50.0, 1000.0, 100.0],
        [T0 * 0.2, 1.0, 400.0, 0.35],
        [T0 * 0.5, 5.0, 200.0, 0.34],
    ]

    def equations(x):
        Tv, Lv, Iv, Ev = x
        Vv = params['p'] * Iv / params['c']
        f = rhs([Tv, Lv, Iv, Vv, Ev], params)
        return [f[0], f[1], f[2], f[4]]

    found = []
    for guess in guesses:
        try:
            sol = fsolve(equations, guess, full_output=True, maxfev=10000)
            x_sol, info, ier, _ = sol
            if ier != 1 or np.max(np.abs(info['fvec'])) > 1e-8:
                continue
            Ts, Ls, Is, Es = x_sol
            if Ts < 0 or Ls < 0 or Is < 0 or Es < 0:
                continue
            Vs = params['p'] * Is / params['c']
            y_sol = np.array([Ts, Ls, Is, Vs, Es])
            if any(np.allclose(y_sol, yp, rtol=1e-4) for yp, _, _ in found):
                continue
            J = jacobian(y_sol, params)
            eigs = np.linalg.eigvals(J)
            n_pos = np.sum(np.real(eigs) > 1e-10)
            stab = 'stable' if n_pos == 0 else ('saddle' if n_pos == 1 else f'unstable({n_pos})')
            found.append((y_sol, eigs, stab))
        except Exception:
            continue
    found.sort(key=lambda x: x[0][3])
    return found


# ============================================================
# CLE diffusion matrix
# ============================================================

REACTION_NAMES = [
    'T production (lam)',
    'T death (dT*T)',
    'Infection (beta*V*T)',
    'L proliferation (rho*L)',
    'L death (dL*L)',
    'L activation (a*L)',
    'I death (delta*I)',
    'CTL killing (m*E*I)',
    'V production (p*I)',
    'V clearance (c*V)',
    'E production (lam_E)',
    'E stimulation (bE*I/(KB+I)*E)',
    'E exhaustion (dE*I/(KD+I)*E)',
    'E death (mu*E)',
]


def cle_diffusion(y, params=PARAMS, Omega=5000.0):
    """
    CLE diffusion matrix: D = (1/Omega) * sum_j rate_j * s_j * s_j^T.
    Returns D (scaled), D_raw (unscaled), per-reaction contributions.
    """
    T, L, I, V, E = y
    p = params
    aL = p['alpha_L']

    rates_stoich = [
        (p['lam'],                              [1, 0, 0, 0, 0]),
        (p['dT'] * T,                           [-1, 0, 0, 0, 0]),
        ((1 - p['eps']) * p['beta'] * V * T,    [-1, aL, 1 - aL, 0, 0]),
        (p['rho'] * L,                          [0, 1, 0, 0, 0]),
        (p['dL'] * L,                           [0, -1, 0, 0, 0]),
        (p['a'] * L,                            [0, -1, 1, 0, 0]),
        (p['delta'] * I,                        [0, 0, -1, 0, 0]),
        (p['m'] * E * I,                        [0, 0, -1, 0, 0]),
        (p['p'] * I,                            [0, 0, 0, 1, 0]),
        (p['c'] * V,                            [0, 0, 0, -1, 0]),
        (p['lam_E'],                            [0, 0, 0, 0, 1]),
        (p['bE'] * I / (p['KB'] + I) * E,      [0, 0, 0, 0, 1]),
        (p['dE'] * I / (p['KD'] + I) * E,      [0, 0, 0, 0, -1]),
        (p['mu'] * E,                           [0, 0, 0, 0, -1]),
    ]

    D_raw = np.zeros((5, 5))
    contribs = []
    for i, (rate, stoich) in enumerate(rates_stoich):
        s = np.array(stoich, dtype=float)
        C = rate * np.outer(s, s)
        D_raw += C
        contribs.append((REACTION_NAMES[i], rate, C))

    return D_raw / Omega, D_raw, contribs


# ============================================================
# Noise-weighted barrier
# ============================================================

def barrier_noise_weighted(y_ptc, y_sad, D_cle, params=PARAMS, n_pts=20000):
    """
    Quasi-potential barrier in noise-weighted coordinates.
    x_tilde_i = x_i / sqrt(D_ii).  In these coords noise is isotropic,
    so the integral -int f_tilde . d_hat ds IS the Kramers exponent.
    """
    D_diag = np.maximum(np.diag(D_cle), 1e-30)
    scale = np.sqrt(D_diag)

    ptc_nw = y_ptc / scale
    sad_nw = y_sad / scale
    delta_nw = sad_nw - ptc_nw
    dist = np.linalg.norm(delta_nw)
    d_hat = delta_nw / dist

    s_arr = np.linspace(0, dist, n_pts)
    ds = s_arr[1] - s_arr[0]
    f_proj = np.empty(n_pts)

    for i, s in enumerate(s_arr):
        x_nw = ptc_nw + s * d_hat
        x_orig = x_nw * scale
        f_orig = rhs(x_orig, params)
        f_nw = f_orig / scale
        f_proj[i] = np.dot(f_nw, d_hat)

    Phi = np.cumsum(-f_proj) * ds
    Phi -= Phi[0]
    return Phi[-1], dist, d_hat, scale


# ============================================================
# Main computation
# ============================================================

def main():
    print("=" * 72)
    print("NOISE-SOURCE MAPPING: HIV CLE TEST (Phase 4)")
    print("=" * 72)

    # --- 1. Fixed points ---
    print("\n1. FIXED POINTS (m = 0.42)")
    print("-" * 50)
    fps = find_fixed_points()

    ptc = saddle = vr = None
    for y, eigs, stab in fps:
        if stab == 'stable' and y[3] < 100:
            ptc = (y, eigs)
        elif stab == 'saddle' and y[3] > 1e-6:
            if saddle is None or y[3] < saddle[0][3]:
                saddle = (y, eigs)
        elif stab == 'stable' and y[3] >= 100:
            vr = (y, eigs)

    if ptc is None or saddle is None:
        print("ERROR: PTC or saddle not found. Aborting.")
        return
    y_ptc, eigs_ptc = ptc
    y_sad, eigs_sad = saddle

    var_names = ['T', 'L', 'I', 'V', 'E']
    for label, y in [('PTC', y_ptc), ('Saddle', y_sad)]:
        vals = ", ".join(f"{n}={v:.8e}" for n, v in zip(var_names, y))
        print(f"  {label:7s}: {vals}")
    if vr:
        vals = ", ".join(f"{n}={v:.6f}" for n, v in zip(var_names, vr[0]))
        print(f"  {'VR':7s}: {vals}")

    # --- 2. epsilon_CTL and D_product ---
    print("\n2. EPSILON_CTL AND D_PRODUCT")
    print("-" * 50)
    T, L, I, V, E = y_ptc
    infection = (1 - PARAMS['alpha_L']) * PARAMS['beta'] * V * T
    activation = PARAMS['a'] * L
    R_prod = infection + activation
    R_CTL = PARAMS['m'] * E * I
    R_death = PARAMS['delta'] * I

    eps_CTL = R_CTL / R_prod if R_prod > 0 else 0.0
    D_product = 1.0 / eps_CTL if eps_CTL > 0 else np.inf

    print(f"  R_prod      = {R_prod:.8e}  (infection={infection:.4e}, activation={activation:.4e})")
    print(f"  R_CTL       = {R_CTL:.8e}")
    print(f"  R_death     = {R_death:.8e}")
    print(f"  Balance err = {abs(R_prod - R_CTL - R_death):.4e}")
    print(f"  epsilon_CTL = {eps_CTL:.8f}")
    print(f"  D_product   = {D_product:.6f}")

    # --- 3. Eigenvalues ---
    print("\n3. EIGENVALUES AND TIMESCALES")
    print("-" * 50)
    re_ptc = np.sort(np.real(eigs_ptc))
    re_sad = np.sort(np.real(eigs_sad))
    print(f"  PTC eigenvalues:    {re_ptc}")
    print(f"  Saddle eigenvalues: {re_sad}")

    neg_ptc = re_ptc[re_ptc < -1e-12]
    tau_relax = 1.0 / np.min(np.abs(neg_ptc))
    lambda_u = np.max(re_sad)
    print(f"  tau_relax = {tau_relax:.4f} days")
    print(f"  lambda_u  = {lambda_u:.8f} day^-1")

    # --- 4. Kramers-Langer prefactor ---
    print("\n4. KRAMERS-LANGER PREFACTOR")
    print("-" * 50)
    prod_ptc_eig = np.prod(np.abs(eigs_ptc))
    idx_u = np.argmax(np.real(eigs_sad))
    mask = np.ones(5, dtype=bool)
    mask[idx_u] = False
    prod_sad_rest = np.prod(np.abs(eigs_sad[mask]))
    prefactor = abs(lambda_u) / (2 * np.pi) * np.sqrt(prod_ptc_eig / prod_sad_rest)
    print(f"  prod|eig_PTC|   = {prod_ptc_eig:.6e}")
    print(f"  prod|eig_sad\\u| = {prod_sad_rest:.6e}")
    print(f"  1/(C*tau)       = {prefactor:.6e} day^-1")

    # --- 5. CLE noise at PTC ---
    print("\n5. CLE NOISE AT PTC")
    print("-" * 50)
    Omega_phys = 5000.0
    D_cle, D_raw, contribs = cle_diffusion(y_ptc, PARAMS, Omega_phys)

    print(f"  Omega_physical = {Omega_phys:.0f} mL")
    print(f"  D_cle diagonal (D_ii = D_raw_ii / Omega):")
    for i in range(5):
        print(f"    D_{var_names[i]}{var_names[i]} = {D_cle[i, i]:.6e}  "
              f"(raw = {D_raw[i, i]:.6e})")

    print(f"\n  Per-reaction contributions to D_raw diagonal:")
    for name, rate, C in contribs:
        diag = np.diag(C)
        parts = [(var_names[j], diag[j]) for j in range(5) if abs(diag[j]) > 1e-25]
        if parts:
            s = ", ".join(f"{n}:{v:.4e}" for n, v in parts)
            print(f"    {name:35s} rate={rate:.4e}  [{s}]")

    # --- 6. Noise along escape direction ---
    print("\n6. NOISE ALONG ESCAPE DIRECTION")
    print("-" * 50)
    J_sad = jacobian(y_sad)
    eigvals_s, eigvecs_s = np.linalg.eig(J_sad)
    idx_unstable = np.argmax(np.real(eigvals_s))
    n_u = np.real(eigvecs_s[:, idx_unstable])
    n_u /= np.linalg.norm(n_u)

    print(f"  Unstable eigenvector at saddle:")
    for i in range(5):
        print(f"    n_{var_names[i]} = {n_u[i]:+.8f}")

    sigma2_raw = n_u @ D_raw @ n_u
    sigma2_cle = n_u @ D_cle @ n_u
    print(f"\n  sigma^2_raw (Omega-independent) = {sigma2_raw:.6e}")
    print(f"  sigma^2_CLE (at Omega={Omega_phys:.0f})  = {sigma2_cle:.6e}")
    print(f"  sigma_CLE                       = {np.sqrt(sigma2_cle):.6e}")

    # Per-reaction contribution to sigma^2 along escape direction
    print(f"\n  Per-reaction contribution to sigma^2_raw along escape:")
    for name, rate, C in contribs:
        proj = n_u @ C @ n_u
        frac = proj / sigma2_raw * 100 if sigma2_raw > 0 else 0
        if abs(frac) > 0.01:
            print(f"    {name:35s} {frac:+8.3f}%  (sigma^2 = {proj:.4e})")

    # --- 7. Noise-weighted barrier ---
    print("\n7. QUASI-POTENTIAL BARRIER")
    print("-" * 50)
    barrier_phys, dist_phys, d_hat, scale_phys = barrier_noise_weighted(
        y_ptc, y_sad, D_cle, n_pts=20000)
    print(f"  Barrier (Omega={Omega_phys:.0f}) = {barrier_phys:.6f}")
    print(f"  NW distance PTC->saddle     = {dist_phys:.4f}")

    # Verify linear scaling: compute at Omega=1
    D_cle_1, _, _ = cle_diffusion(y_ptc, PARAMS, 1.0)
    barrier_1, _, _, _ = barrier_noise_weighted(y_ptc, y_sad, D_cle_1, n_pts=20000)
    print(f"\n  Barrier (Omega=1)            = {barrier_1:.6e}")
    print(f"  Omega * barrier(1)           = {Omega_phys * barrier_1:.6f}")
    print(f"  Scaling check: ratio         = {barrier_phys / (Omega_phys * barrier_1):.6f}"
          f"  (should be 1.0)")
    barrier_per_Omega = barrier_1  # = barrier_phys / Omega_phys

    # --- 8. Kramers difficulty at physical Omega ---
    print("\n8. KRAMERS D AT PHYSICAL NOISE")
    print("-" * 50)
    exponent = barrier_phys
    if exponent < 500:
        D_Kramers = np.exp(exponent) / (prefactor * tau_relax)
    else:
        D_Kramers = np.inf
    MFPT = D_Kramers * tau_relax

    print(f"  Exponent  = {exponent:.6f}")
    print(f"  D_Kramers = {D_Kramers:.6e}")
    print(f"  D_product = {D_product:.6f}")
    if D_Kramers < np.inf:
        print(f"  Ratio D_Kramers/D_product = {D_Kramers / D_product:.4f}")
        print(f"  MFPT = {MFPT:.4e} days = {MFPT / 365.25:.2f} years")

    # --- 9. Find Omega* where D_Kramers = D_product ---
    print("\n9. DUALITY CROSSING: Omega*")
    print("-" * 50)
    log_target = np.log(D_product * prefactor * tau_relax)
    if barrier_per_Omega > 0:
        Omega_star = log_target / barrier_per_Omega
        sigma_star = np.sqrt(sigma2_raw / Omega_star) if Omega_star > 0 else np.inf
        sigma_phys = np.sqrt(sigma2_raw / Omega_phys)
        ratio_Omega = Omega_star / Omega_phys

        print(f"  ln(D_product * prefactor * tau_relax) = {log_target:.6f}")
        print(f"  barrier_per_Omega                     = {barrier_per_Omega:.6e}")
        print(f"  Omega* = ln(..) / barrier_per_Omega   = {Omega_star:.1f}")
        print(f"  Omega_physical                        = {Omega_phys:.0f}")
        print(f"  Omega* / Omega_phys                   = {ratio_Omega:.4f}")
        print(f"\n  sigma* (at Omega*)                    = {sigma_star:.6e}")
        print(f"  sigma_CLE (at Omega_phys)             = {sigma_phys:.6e}")
        print(f"  sigma_CLE / sigma*                    = {sigma_phys / sigma_star:.4f}")
        print(f"  = sqrt(Omega* / Omega_phys)           = {np.sqrt(ratio_Omega):.4f}")
    else:
        Omega_star = None
        print("  Cannot compute: barrier_per_Omega <= 0")

    # --- 10. Dominant noise source ---
    print("\n10. DOMINANT NOISE SOURCE")
    print("-" * 50)
    # Identify which state variable carries most escape noise
    for i in range(5):
        frac_diag = n_u[i]**2 * D_raw[i, i] / sigma2_raw * 100
        print(f"  {var_names[i]}: n_u^2 * D_raw_{var_names[i]}{var_names[i]} / sigma^2_raw"
              f" = {frac_diag:.2f}%  "
              f"(n_u={n_u[i]:+.4f}, D_raw={D_raw[i, i]:.4e})")

    # Analytic noise estimate: dominant source = latent reactivation
    a_L0 = PARAMS['a'] * y_ptc[1]  # a * L at PTC
    total_latent_body = y_ptc[1] * Omega_phys
    print(f"\n  Analytic dominant noise estimate:")
    print(f"    Latent reactivation rate a*L = {a_L0:.6e} events/mL/day")
    print(f"    Total latent cells in body   = {total_latent_body:.4f}")
    print(f"    sigma^2_analytic ~ a*L/Omega = {a_L0 / Omega_phys:.6e}")

    # --- 11. Summary ---
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  D_product (1/epsilon_CTL)       = {D_product:.4f}")
    print(f"  D_Kramers (Omega={Omega_phys:.0f})       = {D_Kramers:.4e}")
    if Omega_star and Omega_star > 0:
        print(f"  Omega*                          = {Omega_star:.1f} mL")
        print(f"  Omega_physical                  = {Omega_phys:.0f} mL")
        print(f"  Ratio Omega*/Omega_phys         = {ratio_Omega:.4f}")
        if ratio_Omega < 1:
            assessment = "STRONGER"
            print(f"\n  Physical CLE noise is {assessment} than needed for duality.")
            print(f"  PTC is LESS stable than D_product predicts at Omega_phys.")
            print(f"  The bridge OVER-PREDICTS stability at physical noise.")
        else:
            assessment = "WEAKER"
            print(f"\n  Physical CLE noise is {assessment} than needed for duality.")
            print(f"  PTC is MORE stable than D_product predicts at Omega_phys.")
            print(f"  The bridge UNDER-PREDICTS stability at physical noise.")
        print(f"\n  Factor of discrepancy: {max(ratio_Omega, 1/ratio_Omega):.2f}x in Omega")
        print(f"  Factor in sigma: {max(sigma_phys/sigma_star, sigma_star/sigma_phys):.2f}x")

        # Clinical interpretation
        print(f"\n  CLINICAL CONTEXT:")
        if D_Kramers < np.inf and MFPT > 0:
            print(f"    MFPT at Omega_phys = {MFPT:.1f} days = {MFPT/365.25:.1f} years")
            if MFPT > 365 * 100:
                print(f"    PTC is effectively permanent against CLE noise alone.")
                print(f"    Clinical rebound is NOT driven by equilibrium fluctuations.")
            elif MFPT > 365:
                print(f"    PTC is metastable: years-scale rebound from noise alone.")
            else:
                print(f"    PTC is unstable: rebound within a year from noise alone.")
        else:
            print(f"    D_Kramers = inf: PTC permanently stable against CLE noise.")


if __name__ == "__main__":
    main()
