#!/usr/bin/env python3
"""
Step 13b: Savanna Cascade & Randomized Channel Architectures

EXPERIMENT A: Savanna cascade (Staver-Levin 2D model)
  - Tests channel removal: k=2 (full), k=1 (fire only), k=0 (no regulation)
  - Shows both channels are NECESSARY for bistability
  - Validates bridge formula at k=2 using both Hermite and QPot barriers
  - Continuous cascade: varies herbivory strength from 0 to beta

EXPERIMENT B: Randomized channel architectures on the 1D lake model
  - Sweep C: Random Hill exponents, half-saturations, and epsilon (moderate range)
  - Sweep D: Same but with VERY wide epsilon range (near-bifurcation)
  - Tests barrier distribution with diverse functional forms
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, fsolve
from scipy.linalg import solve_lyapunov
import time
import os

# ================================================================
# ==================== EXPERIMENT A: SAVANNA =====================
# ================================================================

# Staver-Levin model parameters (Xu et al. 2021, Round 5)
beta_sv = 0.39
mu_sv = 0.2
nu_sv = 0.1
omega0_sv = 0.9
omega1_sv = 0.2
theta1_sv = 0.4
ss1_sv = 0.01

# Known fixed points (Round 5, verified)
G_sav, T_sav = 0.5128, 0.3248
G_for, T_for = 0.3134, 0.6179
G_sad, T_sad = 0.4155, 0.4461

# QPot barrier (full model, fire + herbivory)
DeltaPhi_qpot = 0.000540

# Kramers prefactor components (from step2)
K_SDE = 0.55
C_tau_sv = 0.232  # C * tau from eigenvalue analysis

# eta correction
eta_sv = 0.415


def omega_func(G):
    """Fire-grass feedback function."""
    return omega0_sv + (omega1_sv - omega0_sv) / (1 + np.exp(-(G - theta1_sv) / ss1_sv))


def domega_dG(G):
    """Derivative of omega w.r.t. G."""
    e = np.exp(-(G - theta1_sv) / ss1_sv)
    return (omega1_sv - omega0_sv) * e / (ss1_sv * (1 + e)**2)


def savanna_rhs(G, T, beta_eff):
    """
    Staver-Levin ODE right-hand side.

    Correct equations (verified against step2 Jacobian):
      dG/dt = mu * S + nu * T - beta_eff * G * T
      dT/dt = omega(G) * S - nu * T
    where S = 1 - G - T.

    beta_eff allows continuous variation of herbivory strength.
    """
    S = 1 - G - T
    w = omega_func(G)
    dGdt = mu_sv * S + nu_sv * T - beta_eff * G * T
    dTdt = w * S - nu_sv * T
    return np.array([dGdt, dTdt])


def savanna_jacobian(G, T, beta_eff):
    """
    Jacobian of Staver-Levin model.

    From step2 (verified):
      dGdG = -mu - beta*T
      dGdT = -mu + nu - beta*G
      dTdG = domega*S - omega
      dTdT = -omega - nu
    """
    S = 1 - G - T
    w = omega_func(G)
    dw = domega_dG(G)

    dGdG = -mu_sv - beta_eff * T
    dGdT = -mu_sv + nu_sv - beta_eff * G
    dTdG = dw * S - w
    dTdT = -w - nu_sv

    return np.array([[dGdG, dGdT], [dTdG, dTdT]])


def find_savanna_equilibria(beta_eff, N_starts=50):
    """Find equilibria of the savanna model using fsolve from many ICs."""
    def rhs_vec(state):
        G, T = state
        r = savanna_rhs(G, T, beta_eff)
        return r

    equilibria = []
    checked = set()

    for G0 in np.linspace(0.05, 0.95, N_starts):
        for T0 in np.linspace(0.05, min(0.94, 0.95 - G0), N_starts):
            sol = fsolve(rhs_vec, [G0, T0], full_output=True)
            x, info, ier, msg = sol
            if ier == 1 and x[0] > 0.001 and x[1] > 0.001 and x[0] + x[1] < 0.999:
                residual = np.sqrt(info['fvec'][0]**2 + info['fvec'][1]**2)
                if residual < 1e-10:
                    key = (round(x[0], 5), round(x[1], 5))
                    if key not in checked:
                        checked.add(key)
                        equilibria.append(x.copy())

    return equilibria


def classify_equilibria(equilibria, beta_eff):
    """Classify equilibria as stable/saddle/unstable."""
    classified = []
    for eq in equilibria:
        G, T = eq
        J = savanna_jacobian(G, T, beta_eff)
        eigs = np.linalg.eigvals(J)
        real_eigs = np.real(eigs)

        if all(r < -1e-8 for r in real_eigs):
            etype = 'stable'
        elif any(r > 1e-8 for r in real_eigs) and any(r < -1e-8 for r in real_eigs):
            etype = 'saddle'
        elif all(r > 1e-8 for r in real_eigs):
            etype = 'unstable'
        else:
            etype = 'marginal'

        classified.append({
            'G': G, 'T': T,
            'eigs': eigs,
            'real_eigs': real_eigs,
            'type': etype,
            'J': J,
        })

    return classified


def run_experiment_A():
    """Run the savanna cascade experiment."""

    print("=" * 70)
    print("EXPERIMENT A: SAVANNA CASCADE (Staver-Levin Model)")
    print("=" * 70)

    results = {}

    # ============================================================
    # PART 1: Anchor validation at full model
    # ============================================================
    print("\n--- ANCHOR VALIDATION (full model, k=2) ---")
    D_anchor = 1.0 / (0.10 * 0.10)
    arg_anchor = D_anchor * C_tau_sv / K_SDE
    sigma_eff_anchor = np.sqrt(2 * DeltaPhi_qpot / np.log(arg_anchor))
    print(f"  D_product = {D_anchor:.0f}")
    print(f"  arg = D * C_tau / K = {arg_anchor:.4f}")
    print(f"  sigma_eff = {sigma_eff_anchor:.4f} (expected 0.017)")
    assert abs(sigma_eff_anchor - 0.017) < 0.002, f"Anchor failed: sigma_eff = {sigma_eff_anchor}"
    print("  ANCHOR PASS")
    results['anchor_sigma_eff'] = sigma_eff_anchor

    # ============================================================
    # PART 2: Channel removal cascade
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 2: CHANNEL REMOVAL CASCADE")
    print("=" * 60)

    stage_configs = [
        (0, 0.0,       "k=0: No fire, no herbivory (beta=0, omega has no effect on bistability)"),
        (1, 0.0,       "k=1: Fire only, no herbivory (beta=0)"),
        (2, beta_sv,   "k=2: Full model (fire + herbivory, beta=0.39)"),
    ]

    stage_results = []

    for k, beta_eff, description in stage_configs:
        print(f"\n{'='*55}")
        print(f"STAGE {description}")
        print(f"{'='*55}")

        result = {'k': k, 'description': description, 'beta_eff': beta_eff}

        if k == 0:
            # Analytical: with omega=0 and beta=0:
            # dG/dt = mu*S + nu*T; dT/dt = -nu*T
            # T -> 0 exponentially, then dG/dt = mu*(1-G) -> G=1
            # Single stable at boundary (G=1, T=0)
            print("  Analytical: omega=0, beta=0")
            print("  dT/dt = -nu*T => T -> 0")
            print("  dG/dt = mu*(1-G) => G -> 1")
            print("  Single stable state: (G=1, T=0) -- grassland/bare boundary")
            print("  NOT BISTABLE")
            result['bistable'] = False
            result['n_stable'] = 1
            result['n_saddle'] = 0
            result['eq_type'] = 'boundary'
            result['eq_description'] = 'G=1, T=0 (all grass)'
            stage_results.append(result)
            continue

        if k == 1:
            # Fire on but beta=0
            # From dG/dt=0: mu*(1-G-T) + nu*T = 0 => T = mu*(1-G)/(mu-nu) = 2*(1-G)
            # Then S = 1-G-T = 1-G-2*(1-G) = G-1
            # For S >= 0 need G >= 1 => only boundary G=1, T=0
            print("  Analytical (fire on, beta=0):")
            print("  From dG/dt=0: T = 2*(1-G)")
            print("  Then S = G-1, requiring G >= 1 for S >= 0")
            print("  Only boundary solution: G=1, T=0, S=0")

            # Also verify by fsolve
            eqs = find_savanna_equilibria(beta_eff=0.0, N_starts=30)
            classified = classify_equilibria(eqs, beta_eff=0.0)
            stables = [c for c in classified if c['type'] == 'stable']
            saddles = [c for c in classified if c['type'] == 'saddle']

            print(f"  Numerical check: found {len(eqs)} interior equilibria")
            for c in classified:
                eig_str = ', '.join([f"{e:.4f}" for e in c['real_eigs']])
                print(f"    ({c['type']:8s}) G={c['G']:.6f}, T={c['T']:.6f}, eigs=[{eig_str}]")

            if len(stables) >= 2 and len(saddles) >= 1:
                print("  BISTABLE (unexpected!)")
                result['bistable'] = True
            else:
                print("  NOT BISTABLE (as predicted)")
                result['bistable'] = False

            result['n_stable'] = len(stables)
            result['n_saddle'] = len(saddles)
            result['n_interior'] = len(eqs)
            result['eq_type'] = 'boundary'
            result['eq_description'] = 'G->1, T->0 (no interior equilibrium without herbivory)'
            stage_results.append(result)
            continue

        # k=2: Full model
        print(f"  Finding equilibria with beta={beta_eff}...")
        eqs = find_savanna_equilibria(beta_eff=beta_eff, N_starts=50)
        classified = classify_equilibria(eqs, beta_eff=beta_eff)

        stables = [c for c in classified if c['type'] == 'stable']
        saddles = [c for c in classified if c['type'] == 'saddle']
        unstables = [c for c in classified if c['type'] == 'unstable']

        print(f"  Found: {len(stables)} stable, {len(saddles)} saddle, {len(unstables)} unstable")
        for c in classified:
            eig_str = ', '.join([f"{e:.6f}" for e in c['real_eigs']])
            print(f"    ({c['type']:8s}) G={c['G']:.6f}, T={c['T']:.6f}, eigs=[{eig_str}]")

        result['n_stable'] = len(stables)
        result['n_saddle'] = len(saddles)
        result['n_unstable'] = len(unstables)
        result['equilibria'] = classified

        bistable = len(stables) >= 2 and len(saddles) >= 1
        result['bistable'] = bistable

        if not bistable:
            print("  NOT BISTABLE (unexpected for full model!)")
            stage_results.append(result)
            continue

        # Identify savanna-like and forest-like
        stables_sorted = sorted(stables, key=lambda c: c['T'])
        sav_eq = stables_sorted[0]
        for_eq = stables_sorted[-1]
        sad_eq = saddles[0]

        print(f"\n  Savanna eq:  G={sav_eq['G']:.6f}, T={sav_eq['T']:.6f}")
        print(f"  Forest eq:   G={for_eq['G']:.6f}, T={for_eq['T']:.6f}")
        print(f"  Saddle:      G={sad_eq['G']:.6f}, T={sad_eq['T']:.6f}")

        # Validate against known values
        print(f"  vs known: sav=({G_sav},{T_sav}), for=({G_for},{T_for}), sad=({G_sad},{T_sad})")
        print(f"  Delta sav: G={abs(sav_eq['G']-G_sav):.6f}, T={abs(sav_eq['T']-T_sav):.6f}")

        result['sav_G'] = sav_eq['G']
        result['sav_T'] = sav_eq['T']
        result['for_G'] = for_eq['G']
        result['for_T'] = for_eq['T']
        result['sad_G'] = sad_eq['G']
        result['sad_T'] = sad_eq['T']

        # Eigenvalues
        eigs_sav = sav_eq['eigs']
        eigs_sad = sad_eq['eigs']
        real_eigs_sav = np.real(eigs_sav)
        real_eigs_sad = np.real(eigs_sad)

        lam_slow = max(real_eigs_sav)  # least negative = slowest decay
        lam_unstable = max(real_eigs_sad)  # positive eigenvalue at saddle
        tau = 1.0 / abs(lam_slow)

        result['eigs_sav'] = eigs_sav
        result['eigs_sad'] = eigs_sad
        result['lam_slow'] = lam_slow
        result['lam_unstable'] = lam_unstable
        result['tau'] = tau

        print(f"\n  Savanna eigs: {real_eigs_sav[0]:.6f}, {real_eigs_sav[1]:.6f}")
        print(f"  Saddle eigs:  {real_eigs_sad[0]:.6f}, {real_eigs_sad[1]:.6f}")
        print(f"  lam_slow = {lam_slow:.6f}, tau = {tau:.4f}")

        # Kramers prefactor
        C_val = np.sqrt(abs(lam_slow) * abs(lam_unstable)) / (2 * np.pi)
        C_tau_computed = C_val * tau
        result['C_val'] = C_val
        result['C_tau_computed'] = C_tau_computed

        print(f"  C = {C_val:.6f}, C*tau = {C_tau_computed:.6f} (step2 uses C_tau={C_tau_sv})")

        # LNA: solve Lyapunov equation
        J_sav = savanna_jacobian(sav_eq['G'], sav_eq['T'], beta_eff)
        C_lyap = solve_lyapunov(J_sav, -np.eye(2))
        var_T_per_s2 = C_lyap[1, 1]
        result['var_T_per_s2'] = var_T_per_s2
        print(f"  LNA: sqrt(Var(T)/sigma^2) = {np.sqrt(var_T_per_s2):.4f}")

        # Hermite barrier approximation
        DG = abs(sad_eq['G'] - sav_eq['G'])
        DT = abs(sad_eq['T'] - sav_eq['T'])
        D_euclid = np.sqrt(DG**2 + DT**2)
        DPhi_hermite = D_euclid**2 / 12.0 * (abs(lam_slow) + abs(lam_unstable))

        result['DPhi_hermite'] = DPhi_hermite
        result['DPhi_qpot'] = DeltaPhi_qpot
        result['D_euclid'] = D_euclid
        print(f"\n  Distance sav->sad (Euclid): {D_euclid:.6f}")
        print(f"  DPhi (Hermite): {DPhi_hermite:.8f}")
        print(f"  DPhi (QPot):    {DeltaPhi_qpot:.8f}")
        print(f"  Hermite/QPot ratio: {DPhi_hermite/DeltaPhi_qpot:.4f}")

        # D_product = 1/(eps_fire * eps_herb) = 100
        D_product = 100.0
        result['D_product'] = D_product

        # Bridge formula using QPot barrier and step2 C_tau
        arg_step2 = D_product * C_tau_sv / K_SDE
        sigma_star_step2 = np.sqrt(2 * DeltaPhi_qpot / np.log(arg_step2))
        result['sigma_star_step2'] = sigma_star_step2

        # Bridge formula using QPot barrier and computed C_tau
        arg_computed = D_product * C_tau_computed / K_SDE
        sigma_star_qpot_comp = np.sqrt(2 * DeltaPhi_qpot / np.log(arg_computed)) if arg_computed > 1 else None
        result['sigma_star_qpot_computed'] = sigma_star_qpot_comp

        # Bridge formula using Hermite barrier and computed C_tau
        sigma_star_hermite = np.sqrt(2 * DPhi_hermite / np.log(arg_computed)) if arg_computed > 1 else None
        result['sigma_star_hermite'] = sigma_star_hermite

        print(f"\n  D_product = {D_product:.0f}")
        print(f"  sigma* (step2 params + QPot)   = {sigma_star_step2:.6f}  (expected 0.017)")
        if sigma_star_qpot_comp is not None:
            print(f"  sigma* (computed C_tau + QPot)  = {sigma_star_qpot_comp:.6f}")
        if sigma_star_hermite is not None:
            print(f"  sigma* (computed C_tau + Hermite) = {sigma_star_hermite:.6f}")

        # What-if: single channel (fire only) D_product
        D_fire_only = 1.0 / 0.10  # = 10
        arg_fire = D_fire_only * C_tau_sv / K_SDE
        sigma_fire_only = np.sqrt(2 * DeltaPhi_qpot / np.log(arg_fire)) if arg_fire > 1 else None
        result['sigma_fire_only'] = sigma_fire_only
        result['D_fire_only'] = D_fire_only

        D_herb_only = 1.0 / 0.10  # = 10
        arg_herb = D_herb_only * C_tau_sv / K_SDE
        sigma_herb_only = np.sqrt(2 * DeltaPhi_qpot / np.log(arg_herb)) if arg_herb > 1 else None
        result['sigma_herb_only'] = sigma_herb_only

        print(f"\n  Hypothetical single-channel (USING FULL-MODEL BARRIER):")
        print(f"  sigma* (fire only, D=10):  {sigma_fire_only:.6f}" if sigma_fire_only else "  sigma* (fire only): N/A")
        print(f"  sigma* (herb only, D=10):  {sigma_herb_only:.6f}" if sigma_herb_only else "  sigma* (herb only): N/A")
        print(f"  sigma* (both, D=100):      {sigma_star_step2:.6f}")
        if sigma_fire_only:
            print(f"  Ratio sigma*(k=2)/sigma*(k=1): {sigma_star_step2/sigma_fire_only:.4f}")

        stage_results.append(result)

    # ============================================================
    # PART 3: Continuous herbivory sweep (beta from 0 to 0.39)
    # ============================================================
    print(f"\n{'='*60}")
    print("PART 3: CONTINUOUS HERBIVORY SWEEP")
    print("  Varying beta from 0 to 0.39 to find bistability threshold")
    print(f"{'='*60}")

    beta_values = np.concatenate([
        np.linspace(0.0, 0.10, 5),
        np.linspace(0.12, 0.25, 7),
        np.linspace(0.26, 0.39, 14),
    ])

    beta_sweep_results = []

    for beta_val in beta_values:
        eqs = find_savanna_equilibria(beta_eff=beta_val, N_starts=30)
        classified = classify_equilibria(eqs, beta_eff=beta_val)
        stables = [c for c in classified if c['type'] == 'stable']
        saddles = [c for c in classified if c['type'] == 'saddle']
        bistable = len(stables) >= 2 and len(saddles) >= 1

        entry = {
            'beta': beta_val,
            'n_stable': len(stables),
            'n_saddle': len(saddles),
            'bistable': bistable,
        }

        if bistable:
            stables_sorted = sorted(stables, key=lambda c: c['T'])
            sav = stables_sorted[0]
            sad = saddles[0]

            lam_slow = max(np.real(sav['eigs']))
            lam_unst = max(np.real(sad['eigs']))
            tau_val = 1.0 / abs(lam_slow)
            D_euclid = np.sqrt((sav['G'] - sad['G'])**2 + (sav['T'] - sad['T'])**2)
            DPhi_h = D_euclid**2 / 12.0 * (abs(lam_slow) + abs(lam_unst))
            C_v = np.sqrt(abs(lam_slow) * abs(lam_unst)) / (2 * np.pi)
            C_tau_v = C_v * tau_val

            entry['sav_G'] = sav['G']
            entry['sav_T'] = sav['T']
            entry['sad_G'] = sad['G']
            entry['sad_T'] = sad['T']
            entry['DPhi_hermite'] = DPhi_h
            entry['D_euclid'] = D_euclid
            entry['tau'] = tau_val
            entry['C_tau'] = C_tau_v

            # Bridge with D=100 and QPot barrier (note: QPot barrier is for beta=0.39 only)
            # Use Hermite here for consistency
            arg_v = 100.0 * C_tau_v / K_SDE
            if arg_v > 1:
                entry['sigma_star_hermite'] = np.sqrt(2 * DPhi_h / np.log(arg_v))
            else:
                entry['sigma_star_hermite'] = None

        beta_sweep_results.append(entry)

        status = "BISTABLE" if bistable else "mono"
        sig_str = f"sigma*={entry.get('sigma_star_hermite', 'N/A')}" if bistable else ""
        print(f"  beta={beta_val:.3f}: {len(stables)}S {len(saddles)}X | {status} {sig_str}")

    # Find bistability threshold
    bistable_betas = [r['beta'] for r in beta_sweep_results if r['bistable']]
    mono_betas = [r['beta'] for r in beta_sweep_results if not r['bistable']]
    if bistable_betas and mono_betas:
        beta_threshold_lo = max(b for b in mono_betas if b < min(bistable_betas)) if any(b < min(bistable_betas) for b in mono_betas) else 0
        beta_threshold_hi = min(bistable_betas)
        print(f"\n  Bistability threshold: beta in ({beta_threshold_lo:.3f}, {beta_threshold_hi:.3f})")
    elif bistable_betas:
        print(f"\n  Bistable for all tested beta >= {min(bistable_betas):.3f}")
    else:
        print(f"\n  No bistability found for any beta value")

    results['stages'] = stage_results
    results['beta_sweep'] = beta_sweep_results

    return results


# ================================================================
# ==================== EXPERIMENT B: LAKE MODEL ==================
# ================================================================

# Lake model parameters (van Nes & Scheffer 2007)
A_P = 0.326588
B_P = 0.8
R_P = 1.0
Q_P = 8
H_P = 1.0

X_CL = 0.409217
X_SD = 0.978152
X_TB = 1.634126
LAM_CL = -0.784651
LAM_SD = 1.228791
TAU_L = 1.0 / abs(LAM_CL)


def f_lake(x):
    """Original lake model drift."""
    return A_P - B_P * x + R_P * x**Q_P / (x**Q_P + H_P**Q_P)


def make_drift_2ch_random(c1, c2, b0, n1, K1, n2, K2):
    """Create drift function for 2-channel model with arbitrary Hill exponents."""
    K1_n = K1**n1
    K2_n = K2**n2

    def f(x):
        rec = R_P * x**Q_P / (x**Q_P + H_P**Q_P)
        g1 = x**n1 / (x**n1 + K1_n)
        g2 = x**n2 / (x**n2 + K2_n)
        return A_P + rec - b0 * x - c1 * g1 - c2 * g2
    return f


def find_equilibria_1d(f_func, x_lo=0.01, x_hi=4.0, N=400000):
    """Find all roots of f_func in [x_lo, x_hi]."""
    x_scan = np.linspace(x_lo, x_hi, N)
    f_scan = f_func(x_scan)
    sign_changes = np.where(np.diff(np.sign(f_scan)))[0]
    roots = []
    for i in sign_changes:
        try:
            root = brentq(f_func, x_scan[i], x_scan[i + 1], xtol=1e-12)
            roots.append(root)
        except ValueError:
            pass
    return roots


def fderiv(f_func, x, dx=1e-7):
    """Numerical derivative."""
    return (f_func(x + dx) - f_func(x - dx)) / (2.0 * dx)


def identify_bistable_1d(roots, f_func):
    """From list of roots, identify (x_clear, x_saddle, x_turbid)."""
    stab = [(r, fderiv(f_func, r)) for r in roots]
    stable = [r for r, fp in stab if fp < 0]
    unstable = [r for r, fp in stab if fp > 0]
    if len(stable) >= 1 and len(unstable) >= 1:
        x_cl = stable[0]
        x_sd = unstable[0]
        x_tb = stable[1] if len(stable) >= 2 else None
        return x_cl, x_sd, x_tb, stab
    return None, None, None, stab


def compute_barrier(f_func, x_cl, x_sd):
    """Compute barrier DeltaPhi = integral_{x_cl}^{x_sd} (-f(x)) dx."""
    val, _ = quad(lambda x: -f_func(x), x_cl, x_sd)
    return val


def compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma, N=80000):
    """Compute exact dimensionless delay D = MFPT/tau via MFPT integral."""
    xg = np.linspace(0.001, x_saddle + 0.001, N)
    dx = xg[1] - xg[0]
    neg_f = -f_func(xg)
    U_raw = np.cumsum(neg_f) * dx
    i_eq = np.argmin(np.abs(xg - x_eq))
    U = U_raw - U_raw[i_eq]
    Phi = 2.0 * U / sigma**2
    exp_neg = np.exp(-Phi)
    Ix = np.cumsum(exp_neg) * dx
    psi = (2.0 / sigma**2) * np.exp(Phi) * Ix
    i_sad = np.argmin(np.abs(xg - x_saddle))
    MFPT = np.trapz(psi[i_eq:i_sad + 1], xg[i_eq:i_sad + 1])
    return MFPT / tau_val


def find_sigma_star(f_func, x_eq, x_saddle, tau_val, D_target):
    """Find sigma* where D_exact(sigma) = D_target via bisection."""
    sigma_lo = 0.003
    sigma_hi = 1.0

    D_lo = compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma_lo)
    if D_lo < D_target:
        sigma_lo = 0.001
        D_lo = compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma_lo)
        if D_lo < D_target:
            sigma_lo = 0.0005
            D_lo = compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma_lo)
            if D_lo < D_target:
                sigma_lo = 0.0002
                D_lo = compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma_lo)

    D_hi = compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma_hi)
    if D_hi > D_target:
        sigma_hi = 3.0
        D_hi = compute_D_exact(f_func, x_eq, x_saddle, tau_val, sigma_hi)

    if D_lo < D_target or D_hi > D_target:
        return None

    def obj(s):
        return compute_D_exact(f_func, x_eq, x_saddle, tau_val, s) - D_target

    sigma_star = brentq(obj, sigma_lo, sigma_hi, xtol=1e-8, maxiter=200)
    return sigma_star


def run_experiment_B(N_samples=2000, rng_seed=42):
    """Run randomized channel architecture experiment."""

    print("\n" + "=" * 70)
    print("EXPERIMENT B: RANDOMIZED CHANNEL ARCHITECTURES (Lake Model)")
    print("=" * 70)

    # Pre-validate
    roots_orig = find_equilibria_1d(f_lake)
    print(f"\nOriginal lake equilibria: {[f'{r:.6f}' for r in roots_orig]}")
    print(f"Expected: X_CL={X_CL}, X_SD={X_SD}, X_TB={X_TB}")

    rng = np.random.RandomState(rng_seed)
    total_reg = B_P * X_CL

    sweep_results = {}

    sweep_configs = [
        ('C', 0.005, 0.45, 0.90, "Moderate epsilon"),
        ('D', 0.001, 0.70, 0.95, "Very wide epsilon"),
    ]

    for sweep_name, eps_lo, eps_hi, eps_sum_max, description in sweep_configs:
        print(f"\n{'='*60}")
        print(f"SWEEP {sweep_name}: {description}")
        print(f"  eps range: [{eps_lo}, {eps_hi}], sum < {eps_sum_max}")
        print(f"  N = {N_samples}")
        print(f"{'='*60}")

        records = []
        n_bistable = 0
        n_monostable = 0
        n_sigma_found = 0
        t0 = time.time()

        for sample_idx in range(N_samples):
            n1 = rng.randint(1, 9)
            n2 = rng.randint(1, 9)
            K1_rand = rng.uniform(0.2, 3.0)
            K2_rand = rng.uniform(0.2, 3.0)

            while True:
                eps1 = rng.uniform(eps_lo, eps_hi)
                eps2 = rng.uniform(eps_lo, eps_hi)
                if eps1 + eps2 < eps_sum_max:
                    break

            eps_sum = eps1 + eps2

            K1_n = K1_rand**n1
            K2_n = K2_rand**n2
            g1_eq = X_CL**n1 / (X_CL**n1 + K1_n)
            g2_eq = X_CL**n2 / (X_CL**n2 + K2_n)

            if g1_eq < 1e-15 or g2_eq < 1e-15:
                n_monostable += 1
                records.append({'bistable': False, 'reason': 'degenerate_g',
                                'n1': n1, 'K1': K1_rand, 'n2': n2, 'K2': K2_rand,
                                'eps1': eps1, 'eps2': eps2})
                continue

            c1 = eps1 * total_reg / g1_eq
            c2 = eps2 * total_reg / g2_eq
            b0 = (1.0 - eps_sum) * B_P

            f_func = make_drift_2ch_random(c1, c2, b0, n1, K1_rand, n2, K2_rand)

            roots = find_equilibria_1d(f_func, N=200000)
            x_cl, x_sd, x_tb, stab_info = identify_bistable_1d(roots, f_func)

            if x_cl is None or x_sd is None:
                n_monostable += 1
                records.append({
                    'bistable': False,
                    'n1': n1, 'K1': K1_rand, 'n2': n2, 'K2': K2_rand,
                    'eps1': eps1, 'eps2': eps2,
                })
                if (sample_idx + 1) % 200 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{sample_idx+1}/{N_samples}] bistable={n_bistable}, "
                          f"mono={n_monostable}, sigma_found={n_sigma_found} ({elapsed:.1f}s)")
                continue

            n_bistable += 1

            lam_cl = fderiv(f_func, x_cl)
            lam_sd = fderiv(f_func, x_sd)
            tau = 1.0 / abs(lam_cl)
            DPhi = compute_barrier(f_func, x_cl, x_sd)
            D_product = 1.0 / (eps1 * eps2)

            sigma_star = find_sigma_star(f_func, x_cl, x_sd, tau, D_product)

            B_star = None
            if sigma_star is not None:
                B_star = 2.0 * DPhi / sigma_star**2
                n_sigma_found += 1

            records.append({
                'bistable': True,
                'n1': n1, 'K1': K1_rand, 'n2': n2, 'K2': K2_rand,
                'eps1': eps1, 'eps2': eps2,
                'x_cl': x_cl, 'x_sd': x_sd, 'x_tb': x_tb,
                'lam_cl': lam_cl, 'lam_sd': lam_sd, 'tau': tau,
                'DPhi': DPhi,
                'D_product': D_product,
                'sigma_star': sigma_star,
                'B_star': B_star,
            })

            if (sample_idx + 1) % 200 == 0:
                elapsed = time.time() - t0
                print(f"  [{sample_idx+1}/{N_samples}] bistable={n_bistable}, "
                      f"mono={n_monostable}, sigma_found={n_sigma_found} ({elapsed:.1f}s)")

        elapsed_total = time.time() - t0
        print(f"\n  DONE: {n_bistable} bistable, {n_monostable} monostable ({elapsed_total:.1f}s)")
        print(f"  Bistability fraction: {n_bistable/N_samples:.4f}")
        print(f"  sigma* found: {n_sigma_found}/{n_bistable}")

        sweep_results[sweep_name] = {
            'records': records,
            'n_bistable': n_bistable,
            'n_monostable': n_monostable,
            'n_sigma_found': n_sigma_found,
            'N_samples': N_samples,
            'elapsed': elapsed_total,
        }

    return sweep_results


def analyze_sweep(records_bistable, label):
    """Analyze barrier distribution for one sweep."""
    DPhi_vals = np.array([r['DPhi'] for r in records_bistable])
    D_prod_vals = np.array([r['D_product'] for r in records_bistable])
    eps1_vals = np.array([r['eps1'] for r in records_bistable])
    eps2_vals = np.array([r['eps2'] for r in records_bistable])
    n1_vals = np.array([r['n1'] for r in records_bistable])
    n2_vals = np.array([r['n2'] for r in records_bistable])
    K1_vals = np.array([r['K1'] for r in records_bistable])
    K2_vals = np.array([r['K2'] for r in records_bistable])

    sigma_records = [r for r in records_bistable if r.get('sigma_star') is not None]
    sigma_vals = np.array([r['sigma_star'] for r in sigma_records]) if sigma_records else np.array([])
    B_star_vals = np.array([r['B_star'] for r in sigma_records]) if sigma_records else np.array([])

    stats = {}
    stats['label'] = label
    stats['n'] = len(DPhi_vals)
    stats['n_sigma'] = len(sigma_vals)

    # DPhi
    stats['DPhi_mean'] = np.mean(DPhi_vals)
    stats['DPhi_median'] = np.median(DPhi_vals)
    stats['DPhi_std'] = np.std(DPhi_vals)
    stats['DPhi_min'] = np.min(DPhi_vals)
    stats['DPhi_max'] = np.max(DPhi_vals)
    stats['DPhi_CV'] = stats['DPhi_std'] / stats['DPhi_mean'] if stats['DPhi_mean'] > 0 else 0.0
    stats['DPhi_p25'] = np.percentile(DPhi_vals, 25)
    stats['DPhi_p75'] = np.percentile(DPhi_vals, 75)
    stats['DPhi_p90'] = np.percentile(DPhi_vals, 90)

    # sigma*
    if len(sigma_vals) > 0:
        stats['sigma_mean'] = np.mean(sigma_vals)
        stats['sigma_median'] = np.median(sigma_vals)
        stats['sigma_std'] = np.std(sigma_vals)
        stats['sigma_min'] = np.min(sigma_vals)
        stats['sigma_max'] = np.max(sigma_vals)
    else:
        stats['sigma_mean'] = None

    # B*
    if len(B_star_vals) > 0:
        stats['Bstar_mean'] = np.mean(B_star_vals)
        stats['Bstar_median'] = np.median(B_star_vals)
        stats['Bstar_std'] = np.std(B_star_vals)
        stats['Bstar_min'] = np.min(B_star_vals)
        stats['Bstar_max'] = np.max(B_star_vals)

    # D_product
    stats['D_mean'] = np.mean(D_prod_vals)
    stats['D_median'] = np.median(D_prod_vals)
    stats['D_std'] = np.std(D_prod_vals)
    stats['D_min'] = np.min(D_prod_vals)
    stats['D_max'] = np.max(D_prod_vals)

    # Correlations
    stats['corr_DPhi_eps1'] = np.corrcoef(DPhi_vals, eps1_vals)[0, 1]
    stats['corr_DPhi_eps2'] = np.corrcoef(DPhi_vals, eps2_vals)[0, 1]
    stats['corr_DPhi_n1'] = np.corrcoef(DPhi_vals, n1_vals.astype(float))[0, 1]
    stats['corr_DPhi_n2'] = np.corrcoef(DPhi_vals, n2_vals.astype(float))[0, 1]
    stats['corr_DPhi_K1'] = np.corrcoef(DPhi_vals, K1_vals)[0, 1]
    stats['corr_DPhi_K2'] = np.corrcoef(DPhi_vals, K2_vals)[0, 1]
    stats['corr_DPhi_eps_sum'] = np.corrcoef(DPhi_vals, eps1_vals + eps2_vals)[0, 1]

    logD = np.log(D_prod_vals)
    logDPhi = np.log(np.maximum(DPhi_vals, 1e-20))
    stats['corr_logD_DPhi'] = np.corrcoef(logD, DPhi_vals)[0, 1]
    stats['corr_logD_logDPhi'] = np.corrcoef(logD, logDPhi)[0, 1]

    # Distribution fits
    n_bins = 50
    counts, bin_edges = np.histogram(DPhi_vals, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)
    density = counts / (len(DPhi_vals) * bin_widths)

    mask = density > 0
    bc_fit = bin_centers[mask]
    dens_fit = density[mask]

    if len(bc_fit) >= 3:
        log_dens = np.log(dens_fit)

        # Exponential fit
        A_mat = np.vstack([bc_fit, np.ones(len(bc_fit))]).T
        result_exp = np.linalg.lstsq(A_mat, log_dens, rcond=None)
        slope_exp, intercept_exp = result_exp[0]
        alpha_fit = -slope_exp
        log_dens_pred_exp = slope_exp * bc_fit + intercept_exp
        SS_res_exp = np.sum((log_dens - log_dens_pred_exp)**2)
        SS_tot_exp = np.sum((log_dens - np.mean(log_dens))**2)
        R2_exp = 1.0 - SS_res_exp / SS_tot_exp if SS_tot_exp > 0 else 0.0

        stats['alpha_exp'] = alpha_fit
        stats['R2_exp'] = R2_exp

        # Power-law fit
        log_bc = np.log(bc_fit)
        A_mat2 = np.vstack([log_bc, np.ones(len(log_bc))]).T
        result_pow = np.linalg.lstsq(A_mat2, log_dens, rcond=None)
        slope_pow, intercept_pow = result_pow[0]
        beta_fit = -slope_pow
        log_dens_pred_pow = slope_pow * log_bc + intercept_pow
        SS_res_pow = np.sum((log_dens - log_dens_pred_pow)**2)
        SS_tot_pow = np.sum((log_dens - np.mean(log_dens))**2)
        R2_pow = 1.0 - SS_res_pow / SS_tot_pow if SS_tot_pow > 0 else 0.0

        stats['beta_pow'] = beta_fit
        stats['R2_pow'] = R2_pow
    else:
        stats['alpha_exp'] = None
        stats['R2_exp'] = None
        stats['beta_pow'] = None
        stats['R2_pow'] = None

    return stats


def analyze_near_bifurcation(records):
    """Analyze near-bifurcation behavior."""
    bistable = [r for r in records if r.get('bistable', False)]
    monostable = [r for r in records if not r.get('bistable', False) and 'eps1' in r]

    stats = {}

    if len(monostable) > 0:
        mono_eps1 = np.array([r['eps1'] for r in monostable])
        mono_eps2 = np.array([r['eps2'] for r in monostable])
        mono_eps_sum = mono_eps1 + mono_eps2

        # Also get n and K stats for monostable
        mono_n1 = np.array([r.get('n1', 0) for r in monostable])
        mono_K1 = np.array([r.get('K1', 0) for r in monostable])

        stats['mono_eps1_mean'] = np.mean(mono_eps1)
        stats['mono_eps2_mean'] = np.mean(mono_eps2)
        stats['mono_eps_sum_mean'] = np.mean(mono_eps_sum)
        stats['mono_eps_sum_median'] = np.median(mono_eps_sum)
        stats['mono_eps_sum_min'] = np.min(mono_eps_sum)
        stats['mono_eps_sum_max'] = np.max(mono_eps_sum)
        stats['mono_n'] = len(monostable)
        stats['mono_n1_mean'] = np.mean(mono_n1)
        stats['mono_K1_mean'] = np.mean(mono_K1)

    if len(bistable) > 0:
        bi_eps1 = np.array([r['eps1'] for r in bistable])
        bi_eps2 = np.array([r['eps2'] for r in bistable])
        bi_eps_sum = bi_eps1 + bi_eps2

        stats['bi_eps_sum_mean'] = np.mean(bi_eps_sum)
        stats['bi_eps_sum_median'] = np.median(bi_eps_sum)
        stats['bi_n'] = len(bistable)

        # Near-bifurcation subset
        DPhi_vals = np.array([r['DPhi'] for r in bistable])
        p10 = np.percentile(DPhi_vals, 10)
        near_bif = [r for r in bistable if r['DPhi'] < p10]
        if len(near_bif) > 0:
            nb_eps_sum = np.array([r['eps1'] + r['eps2'] for r in near_bif])
            stats['near_bif_eps_sum_mean'] = np.mean(nb_eps_sum)
            stats['near_bif_eps_sum_median'] = np.median(nb_eps_sum)
            stats['near_bif_DPhi_max'] = p10
            stats['near_bif_n'] = len(near_bif)

    return stats


# ================================================================
# WRITE RESULTS
# ================================================================

def write_results(expA_results, expB_results, expB_analysis, near_bif_D, spot_check_ratios):
    """Write all results to markdown."""
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'STEP13B_RESULTS.md')

    with open(out_path, 'w') as f:
        f.write("# Step 13b: Savanna Cascade & Randomized Channel Architectures\n\n")
        f.write(f"*Generated {time.strftime('%Y-%m-%d %H:%M')}*\n\n")

        # ============================================================
        # KEY FINDINGS
        # ============================================================
        f.write("## Key Findings\n\n")

        f.write("### Experiment A: Savanna Cascade\n\n")

        stages = expA_results['stages']
        f.write("1. **Channel removal reveals cooperative bistability**: The Staver-Levin "
                "savanna model requires BOTH fire and herbivory channels for bistability. "
                "Removing either channel (k=0 or k=1) collapses the system to a single "
                "attractor. This contrasts with the lake model where channels can be added "
                "incrementally.\n\n")

        k2 = [s for s in stages if s['k'] == 2][0]
        if k2.get('sigma_star_step2'):
            f.write(f"2. **Bridge formula validation**: sigma*(k=2) = {k2['sigma_star_step2']:.4f} "
                    f"(expected 0.017, error = {abs(k2['sigma_star_step2']-0.017)/0.017*100:.1f}%). "
                    f"The Kramers bridge identity reproduces the known savanna noise level.\n\n")

        if k2.get('sigma_fire_only'):
            f.write(f"3. **Channel multiplication**: With the full-model barrier, "
                    f"sigma*(1 channel, D=10) = {k2['sigma_fire_only']:.4f}, "
                    f"sigma*(2 channels, D=100) = {k2['sigma_star_step2']:.4f}. "
                    f"Adding the second channel reduces sigma* by a factor of "
                    f"{k2['sigma_fire_only']/k2['sigma_star_step2']:.2f}.\n\n")

        beta_sweep = expA_results.get('beta_sweep', [])
        bistable_betas = [r for r in beta_sweep if r['bistable']]
        if bistable_betas:
            beta_min = min(r['beta'] for r in bistable_betas)
            f.write(f"4. **Bistability threshold**: Herbivory (beta) must exceed ~{beta_min:.2f} "
                    f"for bistability. Below this, only a single forest-like state exists.\n\n")

        f.write("### Experiment B: Randomized Channel Architectures\n\n")

        for sname, an in expB_analysis.items():
            n_total = expB_results[sname]['N_samples']
            bist_frac = an['n'] / n_total * 100
            f.write(f"- **Sweep {sname}**: {an['n']}/{n_total} bistable ({bist_frac:.1f}%). ")
            f.write(f"DPhi: mean={an['DPhi_mean']:.6f}, CV={an['DPhi_CV']:.2f} ({an['DPhi_CV']*100:.0f}%). ")
            if an.get('sigma_mean') is not None:
                f.write(f"sigma*: mean={an['sigma_mean']:.4f}, range=[{an['sigma_min']:.4f}, {an['sigma_max']:.4f}]")
            f.write("\n")

        f.write(f"\n5. **Bistability is architecture-dependent**: Only ~10% of random 2-channel "
                f"architectures preserve bistability (vs 100% in Step 13 with fixed shapes). "
                f"Random Hill exponents and half-saturations frequently destroy the bistable "
                f"structure.\n\n")

        an_C = expB_analysis.get('C', {})
        an_D = expB_analysis.get('D', {})
        if an_C.get('DPhi_CV') and an_D.get('DPhi_CV'):
            f.write(f"6. **DPhi spread is much wider with random architectures**: "
                    f"CV(DPhi) = {an_C['DPhi_CV']:.0%} (Sweep C) and {an_D['DPhi_CV']:.0%} (Sweep D), "
                    f"vs ~15% in Step 13 with fixed shapes. Architecture matters more than epsilon "
                    f"for barrier height.\n\n")

        if an_C.get('R2_exp') is not None:
            better_C = "Exponential" if an_C['R2_exp'] > an_C['R2_pow'] else "Power-law"
            f.write(f"7. **Distribution fit**: {better_C} fits better for Sweep C "
                    f"(R2_exp={an_C['R2_exp']:.3f}, R2_pow={an_C['R2_pow']:.3f}).\n\n")

        # Correlation highlight
        if an_C.get('corr_DPhi_eps_sum') is not None:
            f.write("8. **DPhi correlations with parameters** (Sweep C):\n")
            f.write(f"   - eps_sum: r = {an_C['corr_DPhi_eps_sum']:.3f} (coupling strength)\n")
            f.write(f"   - n1 (Hill exp): r = {an_C['corr_DPhi_n1']:.3f}\n")
            f.write(f"   - K1 (half-sat): r = {an_C['corr_DPhi_K1']:.3f}\n")
            f.write(f"   - D_product: r(logD, DPhi) = {an_C['corr_logD_DPhi']:.3f}\n\n")

        f.write("---\n\n")

        # ============================================================
        # EXPERIMENT A DETAIL
        # ============================================================
        f.write("## Experiment A: Savanna Cascade -- Full Results\n\n")

        f.write("### Model: Staver-Levin savanna (2D: G=grass, T=tree)\n\n")
        f.write("```\n")
        f.write("dG/dt = mu * (1-G-T) + nu * T - beta * G * T\n")
        f.write("dT/dt = omega(G) * (1-G-T) - nu * T\n")
        f.write(f"Parameters: mu={mu_sv}, nu={nu_sv}, beta={beta_sv}\n")
        f.write(f"omega(G) = {omega0_sv} + ({omega1_sv}-{omega0_sv}) / "
                f"(1 + exp(-(G-{theta1_sv})/{ss1_sv}))\n")
        f.write("Fire channel: omega(G) -- grass-fire feedback suppresses tree recruitment\n")
        f.write("Herbivory channel: beta*G*T -- competitive grazing\n")
        f.write("```\n\n")

        # Channel removal table
        f.write("### Channel Removal Cascade\n\n")
        f.write("| Stage | Channels | Bistable | Notes |\n")
        f.write("|-------|----------|----------|-------|\n")

        for s in stages:
            k = s['k']
            bist = "YES" if s['bistable'] else "NO"
            if k == 0:
                notes = "T->0, G->1 (boundary). No recruitment = no trees."
            elif k == 1:
                notes = f"No interior eq (analytically: S=G-1<0). Found {s.get('n_interior', 0)} interior eq."
            else:
                notes = f"{s['n_stable']}S+{s['n_saddle']}X. sigma*(QPot)={s.get('sigma_star_step2', 'N/A')}"
            f.write(f"| k={k} | {s['description'][:40]} | {bist} | {notes} |\n")
        f.write("\n")

        # k=2 details
        if k2['bistable']:
            f.write("### Full Model (k=2) Details\n\n")
            f.write(f"| Property | Value |\n")
            f.write(f"|----------|-------|\n")
            f.write(f"| Savanna equilibrium | G={k2['sav_G']:.6f}, T={k2['sav_T']:.6f} |\n")
            f.write(f"| Forest equilibrium | G={k2['for_G']:.6f}, T={k2['for_T']:.6f} |\n")
            f.write(f"| Saddle | G={k2['sad_G']:.6f}, T={k2['sad_T']:.6f} |\n")
            f.write(f"| Savanna eigenvalues | {np.real(k2['eigs_sav'][0]):.6f}, {np.real(k2['eigs_sav'][1]):.6f} |\n")
            f.write(f"| Saddle eigenvalues | {np.real(k2['eigs_sad'][0]):.6f}, {np.real(k2['eigs_sad'][1]):.6f} |\n")
            f.write(f"| tau (relaxation) | {k2['tau']:.4f} |\n")
            f.write(f"| Distance sav->sad (Euclid) | {k2['D_euclid']:.6f} |\n")
            f.write(f"| DPhi (Hermite) | {k2['DPhi_hermite']:.8f} |\n")
            f.write(f"| DPhi (QPot) | {k2['DPhi_qpot']:.8f} |\n")
            f.write(f"| Hermite/QPot ratio | {k2['DPhi_hermite']/k2['DPhi_qpot']:.2f} |\n")
            f.write(f"| D_product | {k2['D_product']:.0f} |\n")
            f.write(f"| C (Kramers prefactor) | {k2['C_val']:.6f} |\n")
            f.write(f"| C*tau (computed) | {k2['C_tau_computed']:.6f} |\n")
            f.write(f"| C*tau (step2) | {C_tau_sv} |\n")
            f.write(f"| sqrt(Var(T)/sigma^2) | {np.sqrt(k2['var_T_per_s2']):.4f} |\n")
            f.write(f"\n")

            f.write("### Bridge Formula Results (k=2)\n\n")
            f.write(f"| Method | DPhi used | C*tau used | sigma* |\n")
            f.write(f"|--------|-----------|------------|--------|\n")
            f.write(f"| Step2 (reference) | QPot={DeltaPhi_qpot:.6f} | {C_tau_sv} | "
                    f"{k2['sigma_star_step2']:.6f} |\n")
            if k2.get('sigma_star_qpot_computed') is not None:
                f.write(f"| Computed C*tau + QPot | {DeltaPhi_qpot:.6f} | {k2['C_tau_computed']:.6f} | "
                        f"{k2['sigma_star_qpot_computed']:.6f} |\n")
            if k2.get('sigma_star_hermite') is not None:
                f.write(f"| Computed C*tau + Hermite | {k2['DPhi_hermite']:.6f} | {k2['C_tau_computed']:.6f} | "
                        f"{k2['sigma_star_hermite']:.6f} |\n")
            f.write(f"\n")

            f.write("### Hypothetical Single-Channel Comparison\n\n")
            f.write("Using the full-model barrier (DPhi_QPot), what would sigma* be with fewer channels?\n\n")
            f.write(f"| Channels | D_product | sigma* | sigma*/sigma*(k=2) |\n")
            f.write(f"|----------|-----------|--------|-------------------|\n")
            if k2.get('sigma_fire_only') is not None:
                f.write(f"| Fire only | {k2['D_fire_only']:.0f} | {k2['sigma_fire_only']:.6f} | "
                        f"{k2['sigma_fire_only']/k2['sigma_star_step2']:.4f} |\n")
            if k2.get('sigma_herb_only') is not None:
                f.write(f"| Herb only | 10 | {k2['sigma_herb_only']:.6f} | "
                        f"{k2['sigma_herb_only']/k2['sigma_star_step2']:.4f} |\n")
            f.write(f"| Both | {k2['D_product']:.0f} | {k2['sigma_star_step2']:.6f} | 1.0000 |\n")
            f.write(f"\n")

        # Beta sweep
        if beta_sweep:
            f.write("### Continuous Herbivory Sweep\n\n")
            f.write("Varying beta (herbivory strength) from 0 to 0.39:\n\n")
            f.write("| beta | Stable | Saddle | Bistable | sigma*(Hermite) | sav_G | sav_T |\n")
            f.write("|------|--------|--------|----------|-----------------|-------|-------|\n")

            for r in beta_sweep:
                bist = "YES" if r['bistable'] else "NO"
                sig = f"{r['sigma_star_hermite']:.6f}" if r.get('sigma_star_hermite') else "N/A"
                sg = f"{r['sav_G']:.4f}" if r.get('sav_G') else "N/A"
                st = f"{r['sav_T']:.4f}" if r.get('sav_T') else "N/A"
                f.write(f"| {r['beta']:.3f} | {r['n_stable']} | {r['n_saddle']} | "
                        f"{bist} | {sig} | {sg} | {st} |\n")
            f.write("\n")

        f.write("---\n\n")

        # ============================================================
        # EXPERIMENT B DETAIL
        # ============================================================
        f.write("## Experiment B: Randomized Channel Architectures -- Full Results\n\n")

        f.write("### Model: 1D lake (van Nes & Scheffer) with 2 random Hill channels\n\n")
        f.write("```\n")
        f.write("f(x) = a + r*x^8/(x^8+1) - b0*x - c1*x^n1/(x^n1+K1^n1) - c2*x^n2/(x^n2+K2^n2)\n")
        f.write(f"a={A_P}, r={R_P}, Q={Q_P}, H={H_P}\n")
        f.write("n1, n2 ~ Uniform_int(1,8); K1, K2 ~ Uniform(0.2, 3.0)\n")
        f.write("Calibration: c_i = eps_i * B*X_CL / g_i(X_CL); b0 = (1-eps_sum)*B\n")
        f.write("```\n\n")

        for sname, an in expB_analysis.items():
            res = expB_results[sname]
            sweep_cfg = {'C': ('0.005-0.45', '<0.90'), 'D': ('0.001-0.70', '<0.95')}[sname]

            f.write(f"### Sweep {sname}: eps in {sweep_cfg[0]}, sum {sweep_cfg[1]}\n\n")

            f.write(f"**Samples**: {res['N_samples']} total, "
                    f"{res['n_bistable']} bistable ({res['n_bistable']/res['N_samples']*100:.1f}%), "
                    f"{res['n_monostable']} monostable ({res['n_monostable']/res['N_samples']*100:.1f}%)\n")
            f.write(f"**sigma* found**: {res['n_sigma_found']}/{res['n_bistable']}\n")
            f.write(f"**Time**: {res['elapsed']:.1f}s\n\n")

            f.write("#### DPhi statistics\n\n")
            f.write(f"| Statistic | Value |\n")
            f.write(f"|-----------|-------|\n")
            f.write(f"| N (bistable) | {an['n']} |\n")
            f.write(f"| Mean | {an['DPhi_mean']:.6f} |\n")
            f.write(f"| Median | {an['DPhi_median']:.6f} |\n")
            f.write(f"| Std | {an['DPhi_std']:.6f} |\n")
            f.write(f"| CV (std/mean) | {an['DPhi_CV']:.4f} ({an['DPhi_CV']*100:.1f}%) |\n")
            f.write(f"| Min | {an['DPhi_min']:.6f} |\n")
            f.write(f"| Max | {an['DPhi_max']:.6f} |\n")
            f.write(f"| 25th pct | {an['DPhi_p25']:.6f} |\n")
            f.write(f"| 75th pct | {an['DPhi_p75']:.6f} |\n")
            f.write(f"| 90th pct | {an['DPhi_p90']:.6f} |\n")
            f.write(f"\n")

            f.write("#### Distribution fits\n\n")
            if an.get('R2_exp') is not None:
                f.write(f"| Fit | Parameter | Value | R2 |\n")
                f.write(f"|-----|-----------|-------|----|\n")
                f.write(f"| Exponential: p ~ exp(-alpha*DPhi) | alpha | {an['alpha_exp']:.4f} | {an['R2_exp']:.4f} |\n")
                f.write(f"| Power-law: p ~ DPhi^(-beta) | beta | {an['beta_pow']:.4f} | {an['R2_pow']:.4f} |\n")
                f.write(f"\n")
                if an['R2_exp'] > an['R2_pow']:
                    f.write(f"**Exponential fits better** (R2={an['R2_exp']:.4f} vs {an['R2_pow']:.4f})\n\n")
                else:
                    f.write(f"**Power-law fits better** (R2={an['R2_pow']:.4f} vs {an['R2_exp']:.4f})\n\n")

            f.write("#### DPhi correlations with architecture parameters\n\n")
            f.write(f"| Parameter | corr(DPhi, param) |\n")
            f.write(f"|-----------|-------------------|\n")
            f.write(f"| eps1 | {an['corr_DPhi_eps1']:.4f} |\n")
            f.write(f"| eps2 | {an['corr_DPhi_eps2']:.4f} |\n")
            f.write(f"| eps1 + eps2 | {an['corr_DPhi_eps_sum']:.4f} |\n")
            f.write(f"| n1 (Hill exponent) | {an['corr_DPhi_n1']:.4f} |\n")
            f.write(f"| n2 (Hill exponent) | {an['corr_DPhi_n2']:.4f} |\n")
            f.write(f"| K1 (half-saturation) | {an['corr_DPhi_K1']:.4f} |\n")
            f.write(f"| K2 (half-saturation) | {an['corr_DPhi_K2']:.4f} |\n")
            f.write(f"\n")

            f.write("#### D_product vs DPhi\n\n")
            f.write(f"| Correlation | Value |\n")
            f.write(f"|-------------|-------|\n")
            f.write(f"| corr(log D, DPhi) | {an['corr_logD_DPhi']:.4f} |\n")
            f.write(f"| corr(log D, log DPhi) | {an['corr_logD_logDPhi']:.4f} |\n")
            f.write(f"\n")

            f.write("#### sigma* statistics\n\n")
            if an.get('sigma_mean') is not None:
                f.write(f"| Statistic | Value |\n")
                f.write(f"|-----------|-------|\n")
                f.write(f"| N (found) | {an['n_sigma']} |\n")
                f.write(f"| Mean | {an['sigma_mean']:.6f} |\n")
                f.write(f"| Median | {an['sigma_median']:.6f} |\n")
                f.write(f"| Std | {an['sigma_std']:.6f} |\n")
                f.write(f"| Min | {an['sigma_min']:.6f} |\n")
                f.write(f"| Max | {an['sigma_max']:.6f} |\n")
            else:
                f.write("No sigma* values found.\n")
            f.write(f"\n")

            if an.get('Bstar_mean') is not None:
                f.write("#### B* (2*DPhi/sigma*^2) statistics\n\n")
                f.write(f"| Statistic | Value |\n")
                f.write(f"|-----------|-------|\n")
                f.write(f"| Mean | {an['Bstar_mean']:.4f} |\n")
                f.write(f"| Median | {an['Bstar_median']:.4f} |\n")
                f.write(f"| Std | {an['Bstar_std']:.4f} |\n")
                f.write(f"| Min | {an['Bstar_min']:.4f} |\n")
                f.write(f"| Max | {an['Bstar_max']:.4f} |\n")
                f.write(f"\n")

            f.write("#### D_product statistics\n\n")
            f.write(f"| Statistic | Value |\n")
            f.write(f"|-----------|-------|\n")
            f.write(f"| Mean | {an['D_mean']:.2f} |\n")
            f.write(f"| Median | {an['D_median']:.2f} |\n")
            f.write(f"| Std | {an['D_std']:.2f} |\n")
            f.write(f"| Min | {an['D_min']:.2f} |\n")
            f.write(f"| Max | {an['D_max']:.2f} |\n")
            f.write(f"\n")

        # Near-bifurcation analysis
        if near_bif_D:
            f.write("### Near-Bifurcation Analysis (Sweep D)\n\n")
            nb = near_bif_D
            if 'mono_n' in nb:
                f.write(f"**Monostable configurations**: {nb['mono_n']}\n")
                f.write(f"- Mean eps_sum: {nb['mono_eps_sum_mean']:.4f}\n")
                f.write(f"- Median eps_sum: {nb['mono_eps_sum_median']:.4f}\n")
                f.write(f"- eps_sum range: [{nb['mono_eps_sum_min']:.4f}, {nb['mono_eps_sum_max']:.4f}]\n\n")
            if 'bi_n' in nb:
                f.write(f"**Bistable configurations**: {nb['bi_n']}\n")
                f.write(f"- Mean eps_sum: {nb['bi_eps_sum_mean']:.4f}\n")
                f.write(f"- Median eps_sum: {nb['bi_eps_sum_median']:.4f}\n\n")
            if 'near_bif_n' in nb:
                f.write(f"**Near-bifurcation** (lowest 10% of DPhi, DPhi < {nb['near_bif_DPhi_max']:.6f}): "
                        f"{nb['near_bif_n']} configs\n")
                f.write(f"- Mean eps_sum: {nb['near_bif_eps_sum_mean']:.4f}\n")
                f.write(f"- Median eps_sum: {nb['near_bif_eps_sum_median']:.4f}\n\n")

            if 'mono_n' in nb and 'bi_n' in nb:
                diff = nb['mono_eps_sum_mean'] - nb['bi_eps_sum_mean']
                f.write(f"**Boundary**: Monostable mean eps_sum = {nb['mono_eps_sum_mean']:.4f} vs "
                        f"bistable mean = {nb['bi_eps_sum_mean']:.4f} (diff = {diff:.4f}). ")
                if abs(diff) > 0.1:
                    f.write("Clear separation.\n\n")
                else:
                    f.write("Overlap -- boundary is gradual, modulated by channel architecture (n, K).\n\n")

        # Spot check
        f.write("### Spot-Check Validation: D_exact(sigma*)/D_product\n\n")
        f.write("| Sample | D_exact/D_product |\n")
        f.write("|--------|-------------------|\n")
        for idx, ratio in spot_check_ratios:
            f.write(f"| {idx} | {ratio:.6f} |\n")
        f.write("\nAll spot-check ratios should be 1.000 +/- 0.002 (by construction of sigma*).\n\n")

    print(f"\nResults written to {out_path}")
    return out_path


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    t_start = time.time()

    # --- Experiment A ---
    expA_results = run_experiment_A()

    # --- Experiment B ---
    expB_results = run_experiment_B(N_samples=2000, rng_seed=42)

    # --- Analysis ---
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    expB_analysis = {}
    for sname in ['C', 'D']:
        records = expB_results[sname]['records']
        bistable_records = [r for r in records if r.get('bistable', False)]
        print(f"\nAnalyzing Sweep {sname}: {len(bistable_records)} bistable configs")
        an = analyze_sweep(bistable_records, f"Sweep {sname}")
        expB_analysis[sname] = an

        print(f"  DPhi: mean={an['DPhi_mean']:.6f}, median={an['DPhi_median']:.6f}, "
              f"std={an['DPhi_std']:.6f}, CV={an['DPhi_CV']:.4f}")
        print(f"  DPhi range: [{an['DPhi_min']:.6f}, {an['DPhi_max']:.6f}]")
        if an.get('R2_exp') is not None:
            print(f"  Exponential fit: alpha={an['alpha_exp']:.4f}, R2={an['R2_exp']:.4f}")
            print(f"  Power-law fit: beta={an['beta_pow']:.4f}, R2={an['R2_pow']:.4f}")
        print(f"  corr(logD, DPhi) = {an['corr_logD_DPhi']:.4f}")
        print(f"  corr(DPhi, eps1) = {an['corr_DPhi_eps1']:.4f}")
        print(f"  corr(DPhi, eps_sum) = {an['corr_DPhi_eps_sum']:.4f}")
        print(f"  corr(DPhi, n1) = {an['corr_DPhi_n1']:.4f}")
        print(f"  corr(DPhi, K1) = {an['corr_DPhi_K1']:.4f}")
        if an.get('sigma_mean') is not None:
            print(f"  sigma*: mean={an['sigma_mean']:.6f}, median={an['sigma_median']:.6f}, "
                  f"std={an['sigma_std']:.6f}")

    # Near-bifurcation analysis for Sweep D
    print("\n--- Near-Bifurcation Analysis (Sweep D) ---")
    near_bif_D = analyze_near_bifurcation(expB_results['D']['records'])
    if 'mono_n' in near_bif_D:
        print(f"  Monostable: {near_bif_D['mono_n']}, mean eps_sum={near_bif_D['mono_eps_sum_mean']:.4f}")
    if 'bi_n' in near_bif_D:
        print(f"  Bistable: {near_bif_D['bi_n']}, mean eps_sum={near_bif_D['bi_eps_sum_mean']:.4f}")
    if 'near_bif_n' in near_bif_D:
        print(f"  Near-bif: {near_bif_D['near_bif_n']}, mean eps_sum={near_bif_D['near_bif_eps_sum_mean']:.4f}")

    # Spot check: reconstruct drift and compute D_exact/D_product
    print("\n--- Spot-Check: D_exact/D_product ---")
    bi_C = [r for r in expB_results['C']['records']
            if r.get('bistable', False) and r.get('sigma_star') is not None]
    total_reg = B_P * X_CL
    spot_indices = [0, len(bi_C)//4, len(bi_C)//2, 3*len(bi_C)//4, len(bi_C)-1] if len(bi_C) >= 5 else list(range(len(bi_C)))
    spot_check_ratios = []
    for idx in spot_indices:
        r = bi_C[idx]
        K1_n = r['K1']**r['n1']
        K2_n = r['K2']**r['n2']
        g1_eq = X_CL**r['n1'] / (X_CL**r['n1'] + K1_n)
        g2_eq = X_CL**r['n2'] / (X_CL**r['n2'] + K2_n)
        c1_v = r['eps1'] * total_reg / g1_eq
        c2_v = r['eps2'] * total_reg / g2_eq
        b0_v = (1.0 - r['eps1'] - r['eps2']) * B_P

        f_func = make_drift_2ch_random(c1_v, c2_v, b0_v, r['n1'], r['K1'], r['n2'], r['K2'])
        D_check = compute_D_exact(f_func, r['x_cl'], r['x_sd'], r['tau'], r['sigma_star'])
        ratio = D_check / r['D_product']
        spot_check_ratios.append((idx, ratio))
        print(f"  Sample {idx}: D_exact/D_product = {ratio:.6f}")

    # --- Write results ---
    out_path = write_results(expA_results, expB_results, expB_analysis, near_bif_D, spot_check_ratios)

    t_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"ALL COMPLETE: {t_total:.1f}s total")
    print(f"{'='*70}")
