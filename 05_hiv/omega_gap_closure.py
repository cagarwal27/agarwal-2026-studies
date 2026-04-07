"""
Omega Gap Closure: Dense CLE scan at Omega = 2500–4500
========================================================
Fill the gap between Omega=2000 (D=8.22) and Omega=5000 (D>25)
to find precise Omega* where D_exact = D_product = 13.0.

Uses existing scan_omega_eff machinery with:
  - N_traj = 500
  - T_max = 100,000 days
  - burn_days = 2000
  - dt = 0.05
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
import json
from conway_perelson_model import PARAMS, find_fixed_points
from epsilon_duality_test import rhs_batch, noise_batch


D_PRODUCT = 13.0  # = 1/eps_eco

# Previous data points
PREV_DATA = {
    500:  {'D_exact': 0.35,  'n_escaped': 36,  'N_surviving': 200},
    1000: {'D_exact': 1.24,  'n_escaped': 128, 'N_surviving': 200},
    2000: {'D_exact': 8.22,  'n_escaped': 180, 'N_surviving': 200},
}


def run_omega_scan(Omega_values, N_traj=500, dt=0.05, T_max=100000.0,
                   V_threshold=500.0, m_val=0.42, seed_base=42,
                   burn_days=2000.0, report_interval=5000.0):
    """Run CLE SDE scan at specified Omega values."""
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

    print(f"PTC: V* = {y_ptc[3]:.4f}, tau_relax = {tau_relax:.2f} days")
    print(f"V_threshold = {V_threshold}, T_max = {T_max}, dt = {dt}")
    print(f"N_traj = {N_traj}, burn_days = {burn_days}")

    results = []

    for Omega_eff in Omega_values:
        print(f"\n{'='*60}")
        print(f"Omega_eff = {Omega_eff}")
        print(f"{'='*60}", flush=True)
        t0 = time.time()

        rng = np.random.default_rng(seed_base + Omega_eff)
        Y = np.tile(y_ptc, (N_traj, 1))
        dt_use = dt

        # BURN-IN: 2000 days to settle into noise-broadened PTC distribution
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

            if (step + 1) % (burn_steps // 4) == 0:
                elapsed = time.time() - t0
                print(f"  Burn-in {(step+1)*dt_use:.0f}/{burn_days:.0f}d, "
                      f"{int(np.sum(transient_escaped))} transient, {elapsed:.0f}s",
                      flush=True)

        n_transient = int(np.sum(transient_escaped))
        N_surviving = N_traj - n_transient
        print(f"  Burn-in done: {n_transient} transient, {N_surviving} surviving",
              flush=True)

        if N_surviving < 5:
            results.append({
                'Omega_eff': Omega_eff, 'N_traj': N_traj,
                'n_transient': n_transient, 'N_surviving': N_surviving,
                'n_escaped': 0, 'frac_escaped': 0.0,
                'fpts': [], 'mfpt': np.inf, 'D_exact': np.inf,
                'tau_relax': tau_relax, 'elapsed_s': time.time()-t0,
            })
            continue

        # Measurement phase
        fpts = np.full(N_traj, np.inf)
        escaped = transient_escaped.copy()
        steps = int(T_max / dt_use)
        t = 0.0
        next_report = report_interval

        for step in range(steps):
            active = ~escaped
            if not np.any(active):
                break
            Y_a = Y[active]
            drift = rhs_batch(Y_a, params)
            noise_inc = noise_batch(Y_a, params, Omega_eff, dt_use, rng)
            Y_a = np.maximum(Y_a + drift * dt_use + noise_inc, 0.0)
            Y[active] = Y_a
            t += dt_use

            newly_escaped = active & (Y[:, 3] > V_threshold)
            if np.any(newly_escaped):
                fpts[newly_escaped] = t
                escaped[newly_escaped] = True

            if t >= next_report:
                n_esc = int(np.sum(escaped)) - n_transient
                elapsed = time.time() - t0
                print(f"  t={t:.0f}d, {n_esc}/{N_surviving} eq-escaped, "
                      f"{elapsed:.0f}s", flush=True)
                next_report += report_interval

        elapsed = time.time() - t0

        eq_escaped_mask = escaped & (~transient_escaped)
        n_eq_escaped = int(np.sum(eq_escaped_mask))
        frac_eq = n_eq_escaped / N_surviving if N_surviving > 0 else 0

        if n_eq_escaped > 0:
            valid_fpts = fpts[eq_escaped_mask]
            mfpt = np.mean(valid_fpts)
            mfpt_std = np.std(valid_fpts) / np.sqrt(n_eq_escaped)
            mfpt_median = np.median(valid_fpts)
            D_exact = mfpt / tau_relax
        else:
            mfpt = np.inf
            mfpt_std = np.inf
            mfpt_median = np.inf
            D_exact = np.inf

        print(f"\n  RESULT: {n_eq_escaped}/{N_surviving} eq escapes "
              f"({frac_eq*100:.1f}%)")
        if n_eq_escaped > 0:
            print(f"  MFPT = {mfpt:.1f} +/- {mfpt_std:.1f} days "
                  f"(median {mfpt_median:.1f})")
            print(f"  D_exact = {D_exact:.4f}")
        else:
            print(f"  No escapes in {T_max:.0f} days → D > {T_max/tau_relax:.1f}")
        print(f"  Time: {elapsed:.0f}s")

        results.append({
            'Omega_eff': Omega_eff, 'N_traj': N_traj,
            'n_transient': n_transient, 'N_surviving': N_surviving,
            'n_escaped': n_eq_escaped, 'frac_escaped': frac_eq,
            'fpts': list(fpts[eq_escaped_mask]) if n_eq_escaped > 0 else [],
            'mfpt': float(mfpt) if mfpt < np.inf else None,
            'mfpt_std': float(mfpt_std) if mfpt_std < np.inf else None,
            'mfpt_median': float(mfpt_median) if mfpt_median < np.inf else None,
            'D_exact': float(D_exact) if D_exact < np.inf else None,
            'tau_relax': tau_relax,
            'elapsed_s': elapsed,
        })

    return results, y_ptc, tau_relax


def bootstrap_omega_star(all_data, D_target, n_boot=1000, rng_seed=123):
    """
    Bootstrap confidence interval on Omega* by resampling escape times.
    all_data: list of dicts with 'Omega_eff', 'fpts', 'tau_relax'
    """
    rng = np.random.default_rng(rng_seed)
    tau = all_data[0]['tau_relax']

    omega_stars = []
    for _ in range(n_boot):
        # Resample MFPT at each Omega
        Om_D = []
        for d in all_data:
            fpts = d['fpts']
            if len(fpts) < 2:
                continue
            boot_fpts = rng.choice(fpts, size=len(fpts), replace=True)
            D_boot = np.mean(boot_fpts) / tau
            Om_D.append((d['Omega_eff'], D_boot))

        Om_D.sort()
        if len(Om_D) < 2:
            continue

        # Find crossing
        for i in range(len(Om_D) - 1):
            O1, D1 = Om_D[i]
            O2, D2 = Om_D[i+1]
            if D1 > 0 and D2 > 0 and (D1 - D_target) * (D2 - D_target) < 0:
                try:
                    f = interp1d([np.log(D1), np.log(D2)],
                                 [np.log(O1), np.log(O2)])
                    Os = np.exp(float(f(np.log(D_target))))
                    omega_stars.append(Os)
                except:
                    pass
                break

    if len(omega_stars) > 10:
        return {
            'mean': np.mean(omega_stars),
            'median': np.median(omega_stars),
            'ci_lo': np.percentile(omega_stars, 2.5),
            'ci_hi': np.percentile(omega_stars, 97.5),
            'n_valid': len(omega_stars),
        }
    return None


def make_plot(all_points, tau_relax, omega_star_info):
    """D_exact vs Omega with all data, D_product line, Omega*."""
    fig, ax = plt.subplots(figsize=(10, 7))

    Oms = [p[0] for p in all_points]
    Ds = [p[1] for p in all_points]
    finite = [(O, D) for O, D in all_points if D is not None and D < 1e6]
    inf_pts = [(O, D) for O, D in all_points if D is None]

    if finite:
        Of, Df = zip(*finite)
        ax.semilogy(Of, Df, 'bo-', markersize=10, linewidth=2, zorder=5,
                    label='$D_{exact}$ (CLE simulation)')

        # Label each point
        for o, d in finite:
            ax.annotate(f'{d:.2f}', (o, d), textcoords="offset points",
                       xytext=(8, 5), fontsize=9, color='blue')

    # Lower bounds for non-escaping
    for O, D in inf_pts:
        D_lb = 100000.0 / tau_relax
        ax.semilogy(O, D_lb, 'b^', markersize=12, alpha=0.5)
        ax.annotate(f'$>{D_lb:.0f}$', (O, D_lb),
                   textcoords="offset points", xytext=(5, 5), fontsize=9)

    # D_product line
    ax.axhline(D_PRODUCT, color='red', linestyle='--', linewidth=2.5,
              label=f'$D_{{product}} = 1/\\epsilon_{{eco}}$ = {D_PRODUCT:.1f}')

    # Omega*
    if omega_star_info:
        Os = omega_star_info['median']
        ax.axvline(Os, color='green', linestyle=':', linewidth=2.5, alpha=0.8,
                  label=f'$\\Omega^* = {Os:.0f}$')
        if 'ci_lo' in omega_star_info:
            ax.axvspan(omega_star_info['ci_lo'], omega_star_info['ci_hi'],
                      alpha=0.15, color='green', label='95% CI')

    # Physical Omega
    ax.axvline(5000, color='purple', linestyle='-.', linewidth=2, alpha=0.7,
              label='$\\Omega_{phys}$ = 5000 mL (blood)')

    ax.set_xlabel('$\\Omega_{eff}$ (effective system size)', fontsize=14)
    ax.set_ylabel('$D = MFPT / \\tau_{relax}$', fontsize=14)
    ax.set_title('HIV PTC: Omega Gap Closure — $D_{exact}$ vs $D_{product}$',
                fontsize=15)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5500)

    plt.tight_layout()
    plt.savefig('plots/omega_gap_closure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/omega_gap_closure.png")


def main():
    print("=" * 72)
    print("OMEGA GAP CLOSURE: Dense CLE scan Omega=2500–4500")
    print("=" * 72)

    # New Omega values to fill the gap
    Omega_new = [2500, 3000, 3500, 4000, 4500]

    results, y_ptc, tau_relax = run_omega_scan(
        Omega_new, N_traj=500, dt=0.05, T_max=100000.0,
        V_threshold=500.0, m_val=0.42, seed_base=42,
        burn_days=2000.0, report_interval=5000.0)

    # Combine with previous data
    all_results = []
    for Om, prev in PREV_DATA.items():
        all_results.append({
            'Omega_eff': Om, 'D_exact': prev['D_exact'],
            'n_escaped': prev['n_escaped'],
            'N_surviving': prev['N_surviving'],
            'fpts': [],  # no raw FPTs for previous data
            'tau_relax': tau_relax,
            'source': 'previous',
        })
    for r in results:
        r['source'] = 'new'
        all_results.append(r)
    all_results.sort(key=lambda x: x['Omega_eff'])

    # Summary table
    print(f"\n{'='*72}")
    print("COMPLETE D(Omega) TABLE")
    print(f"{'='*72}")
    print(f"{'Omega':>8} {'n_esc':>8} {'N_surv':>8} {'MFPT(d)':>12} "
          f"{'D_exact':>10} {'source':>10}")
    print("-" * 62)

    all_points = []
    for r in all_results:
        D = r.get('D_exact') or r.get('D_exact')
        if D is not None and D < 1e6:
            mfpt_str = f"{D * tau_relax:.1f}" if 'mfpt' not in r or r.get('mfpt') is None else f"{r['mfpt']:.1f}"
            print(f"{r['Omega_eff']:>8} {r.get('n_escaped','-'):>8} "
                  f"{r.get('N_surviving','-'):>8} {mfpt_str:>12} "
                  f"{D:>10.3f} {r.get('source',''):>10}")
            all_points.append((r['Omega_eff'], D))
        else:
            print(f"{r['Omega_eff']:>8} {r.get('n_escaped',0):>8} "
                  f"{r.get('N_surviving','-'):>8} {'> 100000':>12} "
                  f"{'> 50':>10} {r.get('source',''):>10}")
            all_points.append((r['Omega_eff'], None))

    # Interpolate Omega*
    finite_pts = [(O, D) for O, D in all_points if D is not None]
    finite_pts.sort()
    Omega_star = None
    if len(finite_pts) >= 2:
        for i in range(len(finite_pts) - 1):
            O1, D1 = finite_pts[i]
            O2, D2 = finite_pts[i+1]
            if (D1 - D_PRODUCT) * (D2 - D_PRODUCT) < 0:
                f = interp1d([np.log(D1), np.log(D2)],
                             [np.log(O1), np.log(O2)])
                Omega_star = np.exp(float(f(np.log(D_PRODUCT))))
                break

    print(f"\nD_product = {D_PRODUCT:.1f}")
    if Omega_star:
        print(f"Omega* (interpolated) = {Omega_star:.0f}")
        print(f"Omega* / 5000 = {Omega_star/5000:.3f}")

    # Bootstrap CI using new data with raw FPTs
    boot_data = [r for r in results if len(r.get('fpts', [])) >= 2]
    boot_info = None
    if boot_data:
        boot_info = bootstrap_omega_star(boot_data, D_PRODUCT)
        if boot_info:
            print(f"\nBootstrap Omega* = {boot_info['median']:.0f} "
                  f"(95% CI: [{boot_info['ci_lo']:.0f}, {boot_info['ci_hi']:.0f}])")
            print(f"  {boot_info['n_valid']}/{1000} valid bootstrap samples")

    # If no bootstrap crossing found, use point estimate
    omega_star_info = boot_info or ({'median': Omega_star} if Omega_star else None)

    # Plot
    make_plot(all_points, tau_relax, omega_star_info)

    # Verdict
    print(f"\n{'='*72}")
    print("VERDICT")
    print(f"{'='*72}")

    # Check D_exact at Omega=4000
    D_4000 = None
    for r in results:
        if r['Omega_eff'] == 4000 and r.get('D_exact') is not None:
            D_4000 = r['D_exact']

    if D_4000 is not None:
        if 10 <= D_4000 <= 16:
            print(f"D_exact(4000) = {D_4000:.2f} — WITHIN [10,16]")
            print("Duality holds within compartmental/lymphoid volume uncertainty.")
        elif D_4000 > 20:
            print(f"D_exact(4000) = {D_4000:.2f} — ABOVE 20")
            print("The ~2x gap is real and remains unexplained.")
        else:
            print(f"D_exact(4000) = {D_4000:.2f}")
    else:
        print("D_exact(4000) not finite — too few escapes at this noise level.")

    if Omega_star:
        ratio = Omega_star / 5000.0
        if 0.5 < ratio < 1.5:
            print(f"\nOmega*/Omega_phys = {ratio:.3f} — duality holds "
                  f"within {abs(1-ratio)*100:.0f}% of physical volume.")
        else:
            print(f"\nOmega*/Omega_phys = {ratio:.3f} — significant gap remains.")

    # Save raw results as JSON for reproducibility
    save_results = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k != 'fpts'}
        sr['n_fpts'] = len(r.get('fpts', []))
        save_results.append(sr)
    with open('results/omega_gap_raw.json', 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print("\nSaved: results/omega_gap_raw.json")

    return all_results, Omega_star, omega_star_info


if __name__ == "__main__":
    all_results, Omega_star, omega_star_info = main()
