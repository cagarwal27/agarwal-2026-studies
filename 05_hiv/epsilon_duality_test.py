"""
Phase 3: Epsilon-Duality Test for HIV Post-Treatment Control
=============================================================
Step A: epsilon_CTL and D_product at PTC (m=0.42)
Step B: D(sigma) scan via SDE simulation (Euler-Maruyama with CLE noise)
Step C: Duality test — compare D_exact to D_product

Uses the same methodology as savanna Round 9 D(sigma) scan.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
from conway_perelson_model import PARAMS, find_fixed_points


# ============================================================
# STEP A: Epsilon_CTL and D_product
# ============================================================

def compute_epsilon_CTL(m_val=0.42):
    """Compute epsilon_CTL at PTC equilibrium."""
    params = dict(PARAMS)
    params['m'] = m_val
    fps = find_fixed_points(params)

    ptc = None
    for y, eigs, stab in fps:
        if stab == 'stable' and y[3] < 100:
            ptc = (y, eigs)
            break

    if ptc is None:
        raise ValueError(f"No PTC found at m={m_val}")

    y_ptc, eigs_ptc = ptc
    T, L, I, V, E = y_ptc

    beta = params['beta']
    alpha_L = params['alpha_L']
    a = params['a']
    delta = params['delta']
    m = params['m']

    infection = (1 - alpha_L) * beta * V * T
    activation = a * L
    R_prod = infection + activation

    R_CTL = m * E * I
    R_death = delta * I
    balance_err = abs(R_prod - (R_CTL + R_death))

    epsilon_CTL = R_CTL / R_prod
    D_product = 1.0 / epsilon_CTL

    real_eigs = np.real(eigs_ptc)
    tau_relax = 1.0 / np.min(np.abs(real_eigs[real_eigs < -1e-12]))

    return {
        'y_ptc': y_ptc,
        'eigs_ptc': eigs_ptc,
        'R_prod': R_prod,
        'R_CTL': R_CTL,
        'R_death': R_death,
        'infection': infection,
        'activation': activation,
        'balance_err': balance_err,
        'epsilon_CTL': epsilon_CTL,
        'D_product': D_product,
        'tau_relax': tau_relax,
    }


# ============================================================
# STEP B: Fast vectorized SDE simulation
# ============================================================

def rhs_batch(Y, params):
    """
    Vectorized RHS for N_traj trajectories simultaneously.
    Y: shape (N_traj, 5) — each row is [T, L, I, V, E]
    Returns: shape (N_traj, 5)
    """
    T = Y[:, 0]; L = Y[:, 1]; I = Y[:, 2]; V = Y[:, 3]; E = Y[:, 4]

    lam = params['lam']; dT = params['dT']; beta = params['beta']
    eps = params['eps']; alpha_L = params['alpha_L']
    rho = params['rho']; a = params['a']; dL = params['dL']
    delta = params['delta']; p = params['p']; c = params['c']
    m = params['m']; lam_E = params['lam_E']; bE = params['bE']
    KB = params['KB']; dE = params['dE']; KD = params['KD']; mu = params['mu']

    infection = (1 - eps) * beta * V * T

    dTdt = lam - dT * T - infection
    dLdt = alpha_L * infection + (rho - a - dL) * L
    dIdt = (1 - alpha_L) * infection - delta * I + a * L - m * E * I
    dVdt = p * I - c * V
    dEdt = lam_E + bE * I / (KB + I) * E - dE * I / (KD + I) * E - mu * E

    return np.column_stack([dTdt, dLdt, dIdt, dVdt, dEdt])


def noise_batch(Y, params, Omega_eff, dt, rng):
    """
    Vectorized CLE noise for N_traj trajectories.
    Returns noise increment: shape (N_traj, 5).

    Each reaction j contributes sqrt(rate_j / Omega_eff) * s_j * dW_j * sqrt(dt).
    """
    N = Y.shape[0]
    T = Y[:, 0]; L = Y[:, 1]; I = Y[:, 2]; V = Y[:, 3]; E = Y[:, 4]

    a_rate = params['a']; aL = params['alpha_L']; beta = params['beta']
    eps = params['eps']; delta = params['delta']; m = params['m']
    p_val = params['p']; c_val = params['c']; lam = params['lam']
    dT = params['dT']; rho = params['rho']; dL = params['dL']
    lam_E = params['lam_E']; bE = params['bE']; KB = params['KB']
    dE = params['dE']; KD = params['KD']; mu = params['mu']

    # Compute rates for each reaction (shape: (N,))
    rates = np.column_stack([
        np.full(N, lam),                          # 0: T production
        dT * T,                                    # 1: T death
        (1-eps) * beta * V * T,                    # 2: infection
        rho * L,                                   # 3: L proliferation
        dL * L,                                    # 4: L death
        a_rate * L,                                # 5: L activation
        delta * I,                                 # 6: I death
        m * E * I,                                 # 7: CTL killing
        p_val * I,                                 # 8: V production
        c_val * V,                                 # 9: V clearance
        np.full(N, lam_E),                         # 10: E production
        bE * I / (KB + I) * E,                     # 11: E stimulation
        dE * I / (KD + I) * E,                     # 12: E exhaustion
        mu * E,                                    # 13: E death
    ])  # shape (N, 14)

    rates = np.maximum(rates, 0.0)

    # Stoichiometric matrix: (14, 5)
    S = np.array([
        [1, 0, 0, 0, 0],       # 0
        [-1, 0, 0, 0, 0],      # 1
        [-1, aL, 1-aL, 0, 0],  # 2
        [0, 1, 0, 0, 0],       # 3
        [0, -1, 0, 0, 0],      # 4
        [0, -1, 1, 0, 0],      # 5
        [0, 0, -1, 0, 0],      # 6
        [0, 0, -1, 0, 0],      # 7
        [0, 0, 0, 1, 0],       # 8
        [0, 0, 0, -1, 0],      # 9
        [0, 0, 0, 0, 1],       # 10
        [0, 0, 0, 0, 1],       # 11
        [0, 0, 0, 0, -1],      # 12
        [0, 0, 0, 0, -1],      # 13
    ], dtype=float)

    # Draw independent Gaussian increments: (N, 14)
    dW = rng.standard_normal((N, 14))

    # noise_ij = sum_k sqrt(rate_k) * S_kj * dW_k * sqrt(dt) / sqrt(Omega_eff)
    # = sum_k (sqrt(rate_k) * dW_k) * S_kj * sqrt(dt/Omega_eff)
    weighted = np.sqrt(rates) * dW  # (N, 14)
    noise = weighted @ S  # (N, 5)
    noise *= np.sqrt(dt / Omega_eff)

    return noise


def scan_omega_eff(Omega_values, N_traj=200, dt=0.01, T_max=50000.0,
                   V_threshold=500.0, m_val=0.42, seed_base=42,
                   report_interval=1000.0):
    """
    Scan Omega_eff values, running batch SDE simulations.
    All N_traj trajectories evolve simultaneously for each Omega.
    """
    params = dict(PARAMS)
    params['m'] = m_val

    fps = find_fixed_points(params)
    y_ptc = None
    eigs_ptc = None
    for y, eigs, stab in fps:
        if stab == 'stable' and y[3] < 100:
            y_ptc = y.copy()
            eigs_ptc = eigs
            break

    real_eigs = np.real(eigs_ptc)
    tau_relax = 1.0 / np.min(np.abs(real_eigs[real_eigs < -1e-12]))

    print(f"\nPTC: V* = {y_ptc[3]:.4f}, tau_relax = {tau_relax:.2f} days")
    print(f"V_threshold = {V_threshold}, T_max = {T_max}, dt = {dt}, N_traj = {N_traj}")

    results = []

    for Omega_eff in Omega_values:
        print(f"\n--- Omega_eff = {Omega_eff} ---", flush=True)
        t0 = time.time()

        rng = np.random.default_rng(seed_base + Omega_eff)

        # Initialize all trajectories at PTC
        Y = np.tile(y_ptc, (N_traj, 1))  # (N_traj, 5)

        # dt=0.05 is stable for fastest eigenvalue -24 (|1+λdt|=0.2<1)
        if Omega_eff < 20:
            dt_use = 0.02
        else:
            dt_use = 0.05

        # BURN-IN: 500 days to settle into noise-broadened PTC distribution
        # Trajectories that escape during burn-in are transients, not equilibrium
        burn_days = 500.0
        burn_steps = int(burn_days / dt_use)
        transient_escaped = np.zeros(N_traj, dtype=bool)

        for step in range(burn_steps):
            active = ~transient_escaped
            if not np.any(active):
                break
            Y_a = Y[active]
            drift = rhs_batch(Y_a, params)
            noise_inc = noise_batch(Y_a, params, Omega_eff, dt_use, rng)
            Y_a = np.maximum(Y_a + drift * dt_use + noise_inc, 0.0)
            Y[active] = Y_a
            newly_out = active & (Y[:, 3] > V_threshold)
            transient_escaped |= newly_out

        n_transient = int(np.sum(transient_escaped))
        N_surviving = N_traj - n_transient
        print(f"  Burn-in ({burn_days:.0f}d): {n_transient} transient escapes, "
              f"{N_surviving} in PTC basin", flush=True)

        if N_surviving == 0:
            results.append({
                'Omega_eff': Omega_eff, 'N_traj': N_traj,
                'n_transient': n_transient, 'n_escaped': 0,
                'N_surviving': 0,
                'frac_escaped': 0.0, 'fpts': [],
                'mfpt': 0.0, 'mfpt_std': 0.0, 'mfpt_median': 0.0,
                'D_exact': 0.0, 'tau_relax': tau_relax,
                'elapsed_s': time.time()-t0, 'dt_used': dt_use,
            })
            continue

        # Now measure equilibrium escape from the settled distribution
        fpts = np.full(N_traj, np.inf)
        escaped = transient_escaped.copy()  # already-escaped stay marked

        steps = int(T_max / dt_use)
        t = 0.0
        next_report = report_interval

        for step in range(steps):
            active = ~escaped
            if not np.any(active):
                break

            Y_active = Y[active]
            drift = rhs_batch(Y_active, params)
            noise_inc = noise_batch(Y_active, params, Omega_eff, dt_use, rng)
            Y_active = np.maximum(Y_active + drift * dt_use + noise_inc, 0.0)
            Y[active] = Y_active
            t += dt_use

            newly_escaped = active & (Y[:, 3] > V_threshold)
            if np.any(newly_escaped):
                fpts[newly_escaped] = t
                escaped[newly_escaped] = True

            if t >= next_report:
                n_esc = int(np.sum(escaped)) - n_transient
                elapsed = time.time() - t0
                print(f"  t={t:.0f}d, {n_esc}/{N_surviving} eq-escaped, "
                      f"{elapsed:.1f}s elapsed", flush=True)
                next_report += report_interval

        elapsed = time.time() - t0

        # Count only equilibrium escapes (post burn-in)
        eq_escaped_mask = escaped & (~transient_escaped)
        n_eq_escaped = int(np.sum(eq_escaped_mask))
        frac_eq_escaped = n_eq_escaped / N_surviving if N_surviving > 0 else 0

        if n_eq_escaped > 0:
            valid_fpts = fpts[eq_escaped_mask]
            mfpt = np.mean(valid_fpts)
            mfpt_std = np.std(valid_fpts) / np.sqrt(n_eq_escaped)
            mfpt_median = np.median(valid_fpts)
            D_exact = mfpt / tau_relax
        else:
            # Lower bound: MFPT > T_max (no transitions from N_surviving trajs)
            mfpt = np.inf
            mfpt_std = np.inf
            mfpt_median = np.inf
            D_exact = np.inf

        print(f"  RESULT: {n_eq_escaped}/{N_surviving} equilibrium escapes "
              f"({frac_eq_escaped*100:.1f}%), {n_transient} transients")
        if n_eq_escaped > 0:
            print(f"  MFPT = {mfpt:.1f} +/- {mfpt_std:.1f} days "
                  f"(median {mfpt_median:.1f})")
            print(f"  D_exact = MFPT/tau = {D_exact:.4f}")
        else:
            print(f"  No eq transitions in {T_max:.0f} days → D > {T_max/tau_relax:.1f}")
        print(f"  Time: {elapsed:.1f}s (dt={dt_use})")

        results.append({
            'Omega_eff': Omega_eff,
            'N_traj': N_traj,
            'N_surviving': N_surviving,
            'n_transient': n_transient,
            'n_escaped': n_eq_escaped,
            'frac_escaped': frac_eq_escaped,
            'fpts': list(fpts[eq_escaped_mask]) if n_eq_escaped > 0 else [],
            'mfpt': mfpt,
            'mfpt_std': mfpt_std,
            'mfpt_median': mfpt_median if n_eq_escaped > 0 else np.inf,
            'D_exact': D_exact,
            'tau_relax': tau_relax,
            'elapsed_s': elapsed,
            'dt_used': dt_use,
        })

    return results, y_ptc, tau_relax


# ============================================================
# STEP C: Duality test plots
# ============================================================

def duality_test(scan_results, epsilon_data, tau_relax):
    """Plot D_exact vs Omega_eff and find sigma* where D_exact = D_product."""
    D_product = epsilon_data['D_product']

    Omegas = [r['Omega_eff'] for r in scan_results]
    D_exacts = [r['D_exact'] for r in scan_results]

    finite_mask = [D < np.inf for D in D_exacts]
    Om_finite = [O for O, m in zip(Omegas, finite_mask) if m]
    D_finite = [D for D, m in zip(D_exacts, finite_mask) if m]

    # ---- Plot 1: D vs Omega_eff ----
    fig, ax = plt.subplots(figsize=(10, 7))

    if Om_finite:
        ax.semilogy(Om_finite, D_finite, 'bo-', markersize=8, linewidth=2,
                    label='$D_{exact}$ (SDE simulation)')

    # Lower bounds for non-escaping
    for r in scan_results:
        if r['D_exact'] == np.inf:
            D_lb = 50000.0 / tau_relax
            ax.semilogy(r['Omega_eff'], D_lb, 'b^', markersize=10, alpha=0.5)
            ax.annotate(f'$>{D_lb:.0f}$', (r['Omega_eff'], D_lb),
                       textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.axhline(D_product, color='red', linestyle='--', linewidth=2,
              label=f'$D_{{product}} = 1/\\epsilon_{{CTL}}$ = {D_product:.2f}')

    # Find sigma* by interpolation
    sigma_star = None
    Omega_star = None
    if len(Om_finite) >= 2:
        pairs = sorted(zip(Om_finite, D_finite))
        Om_s, D_s = zip(*pairs)

        for i in range(len(D_s) - 1):
            if (D_s[i] - D_product) * (D_s[i+1] - D_product) < 0:
                log_D = [np.log(D_s[i]), np.log(D_s[i+1])]
                log_Om = [np.log(Om_s[i]), np.log(Om_s[i+1])]
                f_interp = interp1d(log_D, log_Om)
                Omega_star = np.exp(float(f_interp(np.log(D_product))))
                sigma_star = 1.0 / np.sqrt(Omega_star)
                ax.axvline(Omega_star, color='green', linestyle=':',
                          linewidth=2, alpha=0.7,
                          label=f'$\\Omega^* = {Omega_star:.0f}$')
                break

    ax.axvline(5000, color='purple', linestyle='-.', linewidth=1.5, alpha=0.7,
              label='$\\Omega_{CLE}$ = 5000 (physical)')
    ax.set_xlabel('$\\Omega_{eff}$ (effective system size)', fontsize=13)
    ax.set_ylabel('$D = MFPT / \\tau_{relax}$', fontsize=13)
    ax.set_title('HIV PTC: D(noise) Scan — Duality Test', fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/phase3_duality_D_vs_Omega.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/phase3_duality_D_vs_Omega.png")

    # ---- Plot 2: Escape fraction ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Omegas, [r['frac_escaped']*100 for r in scan_results],
            'rs-', markersize=8, linewidth=2)
    ax.set_xlabel('$\\Omega_{eff}$', fontsize=12)
    ax.set_ylabel('% trajectories escaping (in 50000 d)', fontsize=12)
    ax.set_title('Escape Fraction vs System Size', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/phase3_escape_fraction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/phase3_escape_fraction.png")

    # ---- Plot 3: D vs 1/Omega (noise intensity) ----
    if len(Om_finite) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        inv_Om = [1.0/O for O in Om_finite]
        ax.semilogy(inv_Om, D_finite, 'bo-', markersize=8, linewidth=2,
                    label='$D_{exact}$')
        ax.axhline(D_product, color='red', linestyle='--', linewidth=2,
                  label=f'$D_{{product}}$ = {D_product:.2f}')
        if Omega_star:
            ax.axvline(1.0/Omega_star, color='green', linestyle=':',
                      linewidth=2, alpha=0.7, label=f'$1/\\Omega^*$')
        ax.axvline(1.0/5000, color='purple', linestyle='-.', linewidth=1.5,
                  alpha=0.7, label='Physical noise')
        ax.set_xlabel('$1/\\Omega_{eff}$ (noise intensity)', fontsize=12)
        ax.set_ylabel('D', fontsize=12)
        ax.set_title('D vs Noise Intensity', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/phase3_D_vs_noise_intensity.png', dpi=150,
                    bbox_inches='tight')
        plt.close()
        print("Saved: plots/phase3_D_vs_noise_intensity.png")

    return sigma_star, Omega_star


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 72)
    print("PHASE 3: EPSILON-DUALITY TEST")
    print("=" * 72)

    # ---- Step A ----
    print("\n" + "=" * 72)
    print("STEP A: epsilon_CTL and D_product")
    print("=" * 72)

    eps_data = compute_epsilon_CTL(m_val=0.42)

    T, L, I, V, E = eps_data['y_ptc']
    print(f"\nPTC fixed point (m=0.42):")
    print(f"  T* = {T:.4f}")
    print(f"  L* = {L:.10f}")
    print(f"  I* = {I:.10f}")
    print(f"  V* = {V:.10f}")
    print(f"  E* = {E:.10f}")

    print(f"\nFlux rates:")
    print(f"  R_prod  = {eps_data['R_prod']:.8e}")
    print(f"    infection:  {eps_data['infection']:.8e} ({eps_data['infection']/eps_data['R_prod']*100:.4f}%)")
    print(f"    activation: {eps_data['activation']:.8e} ({eps_data['activation']/eps_data['R_prod']*100:.4f}%)")
    print(f"  R_CTL   = {eps_data['R_CTL']:.8e} ({eps_data['R_CTL']/eps_data['R_prod']*100:.4f}%)")
    print(f"  R_death = {eps_data['R_death']:.8e} ({eps_data['R_death']/eps_data['R_prod']*100:.4f}%)")
    print(f"  Balance error: {eps_data['balance_err']:.4e}")

    print(f"\n  >>> epsilon_CTL = {eps_data['epsilon_CTL']:.8f}")
    print(f"  >>> D_product   = {eps_data['D_product']:.6f}")
    print(f"  >>> tau_relax   = {eps_data['tau_relax']:.2f} days")

    # ---- Step B ----
    print("\n" + "=" * 72)
    print("STEP B: D(sigma) scan via SDE")
    print("=" * 72)

    # Phase 1: High noise (fast transitions, verify methodology)
    # Phase 2: Lower noise (slower, toward physical regime)
    Omega_values = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

    scan_results, y_ptc, tau_relax = scan_omega_eff(
        Omega_values, N_traj=200, dt=0.05, T_max=50000.0,
        V_threshold=500.0, m_val=0.42, seed_base=42)

    # Summary table
    print(f"\n{'='*72}")
    print("STEP B SUMMARY")
    print(f"{'='*72}")
    print(f"{'Omega':>8} {'trans':>6} {'surv':>6} {'eq_esc':>8} {'frac':>8} "
          f"{'MFPT(d)':>12} {'D_exact':>10} {'time(s)':>8}")
    print("-" * 82)
    for r in scan_results:
        n_surv = r.get('N_surviving', r['N_traj'])
        n_trans = r.get('n_transient', 0)
        esc_str = f"{r['n_escaped']}/{n_surv}"
        frac_str = f"{r['frac_escaped']*100:.1f}%"
        if r['D_exact'] < np.inf:
            print(f"{r['Omega_eff']:>8} {n_trans:>6} {n_surv:>6} {esc_str:>8} {frac_str:>8} "
                  f"{r['mfpt']:>12.1f} {r['D_exact']:>10.4f} {r['elapsed_s']:>8.1f}")
        else:
            print(f"{r['Omega_eff']:>8} {n_trans:>6} {n_surv:>6} {esc_str:>8} {frac_str:>8} "
                  f"{'  >50000':>12} {'>25':>10} {r['elapsed_s']:>8.1f}")

    # ---- Step C ----
    print(f"\n{'='*72}")
    print("STEP C: Duality Test")
    print(f"{'='*72}")

    sigma_star, Omega_star = duality_test(scan_results, eps_data, tau_relax)

    D_product = eps_data['D_product']
    print(f"\n  D_product = 1/epsilon_CTL = {D_product:.4f}")

    if Omega_star:
        sigma_CLE = 1.0 / np.sqrt(5000.0)
        print(f"  Omega* (D_exact = D_product) = {Omega_star:.1f}")
        print(f"  sigma* = 1/sqrt(Omega*) = {1.0/np.sqrt(Omega_star):.6f}")
        print(f"  sigma_CLE (physical, Omega=5000) = {sigma_CLE:.6f}")
        ratio = Omega_star / 5000.0
        print(f"  Omega*/Omega_CLE = {ratio:.4f}")
        if 0.5 < ratio < 2.0:
            print(f"\n  DUALITY HOLDS within factor {max(ratio,1/ratio):.2f}")
        else:
            log_discrepancy = abs(np.log(ratio))
            print(f"\n  Discrepancy: |ln(Omega*/Omega_CLE)| = {log_discrepancy:.2f}")
    else:
        D_finite = [r['D_exact'] for r in scan_results if r['D_exact'] < np.inf]
        if D_finite and D_product > max(D_finite):
            print(f"  D_product = {D_product:.2f} > max(D_exact) = {max(D_finite):.4f}")
            Om_f = sorted([(r['Omega_eff'], r['D_exact']) for r in scan_results
                          if r['D_exact'] < np.inf])
            if len(Om_f) >= 2:
                O1, D1 = Om_f[-2]
                O2, D2 = Om_f[-1]
                slope = (np.log(D2) - np.log(D1)) / (np.log(O2) - np.log(O1))
                logO_star = np.log(O2) + (np.log(D_product) - np.log(D2)) / slope
                Omega_star = np.exp(logO_star)
                print(f"  Extrapolated Omega* ~ {Omega_star:.0f}")
                print(f"  Omega*/Omega_CLE ~ {Omega_star/5000:.2f}")
        elif not D_finite:
            print("  No transitions at any noise level — system may not be stochastically bistable")
        else:
            print(f"  D_product < min(D_exact) — duality crossing below tested range")

    return eps_data, scan_results, sigma_star, Omega_star


if __name__ == "__main__":
    eps_data, scan_results, sigma_star, Omega_star = main()
