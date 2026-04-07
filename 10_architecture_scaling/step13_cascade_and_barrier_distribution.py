#!/usr/bin/env python3
"""
Step 13: Channel Cascade Test & Barrier Distribution

EXPERIMENT 1: Sequential channel addition (k=0 -> 1 -> 2 -> 3).
  Measures how barrier, sigma*, and D evolve at each stage.

EXPERIMENT 2: Statistical distribution of barrier heights across
  random channel configurations (k=2 and k=3).
  Tests whether p(DeltaPhi) decays exponentially or as power-law.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import time
import os

# ================================================================
# Physical parameters (van Nes & Scheffer 2007)
# ================================================================
A_P = 0.326588
B_P = 0.8
R_P = 1.0
Q_P = 8
H_P = 1.0

X_CL = 0.409217      # clear-water equilibrium
X_SD = 0.978152      # saddle point
X_TB = 1.634126      # turbid equilibrium
LAM_CL = -0.784651
LAM_SD = 1.228791

# ================================================================
# Channel shape parameters
# ================================================================
K1 = 0.5;    K1_4 = K1**4    # = 0.0625
K2 = 2.0
K3 = 1.0;    K3_2 = K3**2    # = 1.0

# Channel values at original equilibrium
g1_eq0 = X_CL**4 / (X_CL**4 + K1_4)
g2_eq0 = X_CL / (X_CL + K2)
g3_eq0 = X_CL**2 / (X_CL**2 + K3_2)
total_reg = B_P * X_CL   # = 0.32737360

# ================================================================
# Core functions
# ================================================================

def f_lake(x):
    """Original lake model drift."""
    return A_P - B_P * x + R_P * x**Q_P / (x**Q_P + H_P**Q_P)


def make_drift(c1, c2, c3, b0):
    """Create drift function for the 3-channel model."""
    def f(x):
        rec = R_P * x**Q_P / (x**Q_P + H_P**Q_P)
        return (A_P + rec - b0 * x
                - c1 * x**4 / (x**4 + K1_4)
                - c2 * x / (x + K2)
                - c3 * x**2 / (x**2 + K3_2))
    return f


def find_equilibria(f_func, x_lo=0.01, x_hi=4.0, N=400000):
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


def identify_bistable(roots, f_func):
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

    # Adjust bracket if needed
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


# ================================================================
# EXPERIMENT 1: Channel Cascade Test
# ================================================================

def run_experiment1():
    """Run the channel cascade experiment for 3 epsilon profiles."""

    print("=" * 70)
    print("EXPERIMENT 1: CHANNEL CASCADE TEST")
    print("=" * 70)

    # Three profiles
    profiles = [
        ("Uniform-0.10", [0.10, 0.10, 0.10]),
        ("Uniform-0.05", [0.05, 0.05, 0.05]),
        ("Mixed",        [0.05, 0.10, 0.20]),
    ]

    g_funcs_eq0 = [g1_eq0, g2_eq0, g3_eq0]  # channel values at X_CL

    all_profile_results = []

    for pname, epsilons in profiles:
        print(f"\n{'='*60}")
        print(f"PROFILE: {pname}  epsilons = {epsilons}")
        print(f"{'='*60}")

        stages = []

        for k in range(4):  # k=0,1,2,3 channels
            t0 = time.time()
            print(f"\n--- Stage k={k} ---")

            # Determine active epsilons and coefficients
            active_eps = epsilons[:k]
            eps_sum = sum(active_eps)

            # Calibrate coefficients
            c_vals = [0.0, 0.0, 0.0]
            for i in range(k):
                c_vals[i] = epsilons[i] * total_reg / g_funcs_eq0[i]

            b0 = (1.0 - eps_sum) * B_P

            print(f"  Active channels: {k}")
            print(f"  c1={c_vals[0]:.8f}, c2={c_vals[1]:.8f}, c3={c_vals[2]:.8f}, b0={b0:.6f}")

            # Build drift
            f_func = make_drift(c_vals[0], c_vals[1], c_vals[2], b0)

            # Verify equilibrium preservation
            f_at_xcl = f_func(X_CL)
            print(f"  f(X_CL) = {f_at_xcl:.2e}")

            # Find equilibria
            roots = find_equilibria(f_func)
            x_cl, x_sd, x_tb, stab_info = identify_bistable(roots, f_func)

            print(f"  Equilibria ({len(roots)} found):")
            for r, fp in stab_info:
                tag = "stable" if fp < 0 else "unstable"
                print(f"    x={r:.8f}  f'={fp:+.6f}  [{tag}]")

            stage_data = {
                'k': k,
                'epsilons': active_eps[:],
                'c_vals': c_vals[:],
                'b0': b0,
                'bistable': x_cl is not None and x_sd is not None,
                'roots': roots[:],
                'stab_info': stab_info[:],
            }

            if not stage_data['bistable']:
                print(f"  *** NOT BISTABLE at stage k={k} ***")
                stages.append(stage_data)
                continue

            # Eigenvalues and relaxation time
            lam_cl = fderiv(f_func, x_cl)
            lam_sd = fderiv(f_func, x_sd)
            tau = 1.0 / abs(lam_cl)

            # Barrier
            DPhi = compute_barrier(f_func, x_cl, x_sd)

            # Kramers prefactor
            C_val = np.sqrt(abs(lam_cl) * abs(lam_sd)) / (2 * np.pi)
            inv_Ctau = 1.0 / (C_val * tau)

            # D_product
            if k == 0:
                D_product = 1.0
            else:
                D_product = 1.0
                for ei in active_eps:
                    D_product *= (1.0 / ei)

            stage_data.update({
                'x_cl': x_cl, 'x_sd': x_sd, 'x_tb': x_tb,
                'lam_cl': lam_cl, 'lam_sd': lam_sd, 'tau': tau,
                'DPhi': DPhi, 'C_val': C_val, 'inv_Ctau': inv_Ctau,
                'D_product': D_product,
            })

            print(f"  x_cl={x_cl:.8f}, x_sd={x_sd:.8f}", end="")
            if x_tb is not None:
                print(f", x_tb={x_tb:.8f}")
            else:
                print()
            print(f"  lam_cl={lam_cl:+.6f}, lam_sd={lam_sd:+.6f}, tau={tau:.6f}")
            print(f"  DPhi={DPhi:.8f}")
            print(f"  C={C_val:.6f}, 1/(C*tau)={inv_Ctau:.6f}")
            print(f"  D_product={D_product:.4f}")

            # Find sigma* (where D_exact = D_product)
            if k == 0:
                # For k=0, D_product=1; find sigma* at a few reference values
                # Also compute D_exact at a characteristic sigma for reference
                sigma_ref = 0.30
                D_at_ref = compute_D_exact(f_func, x_cl, x_sd, tau, sigma_ref)
                print(f"  D_exact(sigma=0.30) = {D_at_ref:.4f}")
                stage_data['sigma_ref'] = sigma_ref
                stage_data['D_at_ref'] = D_at_ref

                # sigma* for D=1 is very large (almost no barrier needed)
                # Try to find it
                sigma_star = find_sigma_star(f_func, x_cl, x_sd, tau, 1.0)
                if sigma_star is not None:
                    D_check = compute_D_exact(f_func, x_cl, x_sd, tau, sigma_star)
                    B_star = 2.0 * DPhi / sigma_star**2
                    std_frac = sigma_star / x_cl
                    print(f"  sigma*(D=1) = {sigma_star:.8f}")
                    print(f"  D_exact(sigma*) = {D_check:.6f}")
                    print(f"  B* = 2*DPhi/sigma*^2 = {B_star:.4f}")
                    print(f"  sigma*/x_cl = {std_frac:.4f}")
                    stage_data['sigma_star'] = sigma_star
                    stage_data['D_check'] = D_check
                    stage_data['B_star'] = B_star
                    stage_data['std_fraction'] = std_frac
                else:
                    print(f"  sigma*(D=1) = NOT FOUND")
                    stage_data['sigma_star'] = None
            else:
                print(f"  Finding sigma* where D_exact = {D_product:.1f} ...")
                sigma_star = find_sigma_star(f_func, x_cl, x_sd, tau, D_product)
                if sigma_star is not None:
                    D_check = compute_D_exact(f_func, x_cl, x_sd, tau, sigma_star)
                    B_star = 2.0 * DPhi / sigma_star**2
                    std_frac = sigma_star / x_cl
                    duality_ratio = D_check / D_product

                    print(f"  sigma* = {sigma_star:.8f}")
                    print(f"  D_exact(sigma*) = {D_check:.6f}")
                    print(f"  D_exact/D_product = {duality_ratio:.6f}")
                    print(f"  B* = 2*DPhi/sigma*^2 = {B_star:.4f}")
                    print(f"  sigma*/x_cl = {std_frac:.4f}")

                    # VALIDATION: duality check
                    if abs(duality_ratio - 1.0) > 0.001:
                        print(f"  *** VALIDATION FAILURE: duality ratio = {duality_ratio:.6f} ***")

                    stage_data['sigma_star'] = sigma_star
                    stage_data['D_check'] = D_check
                    stage_data['B_star'] = B_star
                    stage_data['std_fraction'] = std_frac
                    stage_data['duality_ratio'] = duality_ratio
                else:
                    print(f"  *** sigma* NOT FOUND ***")
                    stage_data['sigma_star'] = None

            elapsed = time.time() - t0
            print(f"  [{elapsed:.1f}s]")
            stages.append(stage_data)

        # Compute derived quantities between consecutive stages
        transitions = []
        for idx in range(1, len(stages)):
            prev = stages[idx - 1]
            curr = stages[idx]
            if not prev['bistable'] or not curr['bistable']:
                transitions.append(None)
                continue
            t = {}
            t['DPhi_ratio'] = curr['DPhi'] / prev['DPhi']
            if prev.get('sigma_star') is not None and curr.get('sigma_star') is not None:
                t['sigma_ratio'] = curr['sigma_star'] / prev['sigma_star']
            else:
                t['sigma_ratio'] = None
            t['saddle_shift'] = (curr['x_sd'] - prev['x_sd']) / prev['x_sd']
            transitions.append(t)

        all_profile_results.append({
            'name': pname,
            'epsilons': epsilons,
            'stages': stages,
            'transitions': transitions,
        })

    # VALIDATION: check Stage 0 barrier
    s0 = all_profile_results[0]['stages'][0]
    if s0['bistable']:
        DPhi0 = s0['DPhi']
        print(f"\n--- VALIDATION ---")
        print(f"Stage 0 DPhi = {DPhi0:.8f} (expected ~0.065)")
        if abs(DPhi0 - 0.065) > 0.01:
            print(f"  WARNING: DPhi0 deviates from expected 0.065 by {abs(DPhi0-0.065):.4f}")

    # VALIDATION: Stage 3 of Uniform-0.10 should give D~1000, sigma*~0.149
    s3_u10 = all_profile_results[0]['stages'][3]
    if s3_u10['bistable'] and s3_u10.get('sigma_star') is not None:
        print(f"Uniform-0.10 Stage 3: D_product={s3_u10['D_product']:.1f} (expected 1000)")
        print(f"  sigma*={s3_u10['sigma_star']:.6f} (expected ~0.149)")

    # VALIDATION: Stage 3 of Uniform-0.05 should give D~8000
    s3_u05 = all_profile_results[1]['stages'][3]
    if s3_u05['bistable'] and s3_u05.get('sigma_star') is not None:
        print(f"Uniform-0.05 Stage 3: D_product={s3_u05['D_product']:.1f} (expected 8000)")
        print(f"  sigma*={s3_u05['sigma_star']:.6f}")

    return all_profile_results


# ================================================================
# EXPERIMENT 2: Barrier Distribution
# ================================================================

def run_experiment2(N_samples=2000, rng_seed=42):
    """Sample random channel configurations and measure barrier statistics."""

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: BARRIER DISTRIBUTION")
    print("=" * 70)

    rng = np.random.RandomState(rng_seed)

    results = {}

    for n_ch in [2, 3]:
        sweep_name = f"Sweep {'A' if n_ch == 2 else 'B'} (k={n_ch})"
        print(f"\n{'='*60}")
        print(f"{sweep_name}: sampling {N_samples} configurations")
        print(f"{'='*60}")

        records = []
        n_bistable = 0
        n_monostable = 0
        t0 = time.time()

        for sample_idx in range(N_samples):
            # Sample epsilons
            while True:
                eps_list = rng.uniform(0.01, 0.30, size=n_ch)
                if sum(eps_list) < 0.80:
                    break

            eps1 = eps_list[0]
            eps2 = eps_list[1]
            eps3 = eps_list[2] if n_ch == 3 else 0.0

            # Calibrate
            c1 = eps1 * total_reg / g1_eq0
            c2 = eps2 * total_reg / g2_eq0
            c3 = (eps3 * total_reg / g3_eq0) if n_ch == 3 else 0.0
            eps_sum = sum(eps_list)
            b0 = (1.0 - eps_sum) * B_P

            # Build drift
            f_func = make_drift(c1, c2, c3, b0)

            # Find equilibria
            roots = find_equilibria(f_func)
            x_cl, x_sd, x_tb, stab_info = identify_bistable(roots, f_func)

            if x_cl is None or x_sd is None:
                n_monostable += 1
                records.append({
                    'eps': eps_list.tolist(),
                    'bistable': False,
                })
                if (sample_idx + 1) % 200 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{sample_idx+1}/{N_samples}] bistable={n_bistable}, mono={n_monostable} ({elapsed:.1f}s)")
                continue

            n_bistable += 1

            # Eigenvalues and tau
            lam_cl = fderiv(f_func, x_cl)
            lam_sd = fderiv(f_func, x_sd)
            tau = 1.0 / abs(lam_cl)

            # Barrier
            DPhi = compute_barrier(f_func, x_cl, x_sd)

            # D_product
            D_product = 1.0
            for ei in eps_list:
                D_product *= (1.0 / ei)

            # Find sigma*
            sigma_star = find_sigma_star(f_func, x_cl, x_sd, tau, D_product)

            if sigma_star is not None:
                B_star = 2.0 * DPhi / sigma_star**2
                std_frac = sigma_star / x_cl
            else:
                B_star = None
                std_frac = None

            records.append({
                'eps': eps_list.tolist(),
                'bistable': True,
                'x_cl': x_cl, 'x_sd': x_sd, 'x_tb': x_tb,
                'lam_cl': lam_cl, 'lam_sd': lam_sd, 'tau': tau,
                'DPhi': DPhi,
                'D_product': D_product,
                'sigma_star': sigma_star,
                'B_star': B_star,
                'std_fraction': std_frac,
            })

            if (sample_idx + 1) % 200 == 0:
                elapsed = time.time() - t0
                print(f"  [{sample_idx+1}/{N_samples}] bistable={n_bistable}, mono={n_monostable} ({elapsed:.1f}s)")

        elapsed_total = time.time() - t0
        print(f"\n  DONE: {n_bistable} bistable, {n_monostable} monostable out of {N_samples} ({elapsed_total:.1f}s)")
        print(f"  Bistability fraction: {n_bistable/N_samples:.4f}")

        results[n_ch] = {
            'records': records,
            'n_bistable': n_bistable,
            'n_monostable': n_monostable,
            'N_samples': N_samples,
        }

    return results


# ================================================================
# Analysis functions
# ================================================================

def analyze_distribution(records_bistable, label):
    """Analyze the distribution of barrier heights and related quantities."""
    DPhi_vals = np.array([r['DPhi'] for r in records_bistable])
    D_prod_vals = np.array([r['D_product'] for r in records_bistable])

    # sigma* values (only where found)
    sigma_records = [r for r in records_bistable if r.get('sigma_star') is not None]
    sigma_vals = np.array([r['sigma_star'] for r in sigma_records])
    B_star_vals = np.array([r['B_star'] for r in sigma_records])

    stats = {}
    stats['label'] = label
    stats['n'] = len(DPhi_vals)
    stats['n_sigma'] = len(sigma_vals)

    # DPhi statistics
    stats['DPhi_mean'] = np.mean(DPhi_vals)
    stats['DPhi_median'] = np.median(DPhi_vals)
    stats['DPhi_std'] = np.std(DPhi_vals)
    stats['DPhi_min'] = np.min(DPhi_vals)
    stats['DPhi_max'] = np.max(DPhi_vals)
    stats['DPhi_p25'] = np.percentile(DPhi_vals, 25)
    stats['DPhi_p75'] = np.percentile(DPhi_vals, 75)
    stats['DPhi_p90'] = np.percentile(DPhi_vals, 90)

    # Fraction above thresholds
    stats['frac_ge_median'] = np.mean(DPhi_vals >= stats['DPhi_median'])
    stats['frac_ge_p75'] = np.mean(DPhi_vals >= stats['DPhi_p75'])
    stats['frac_ge_p90'] = np.mean(DPhi_vals >= stats['DPhi_p90'])

    # sigma* statistics
    if len(sigma_vals) > 0:
        stats['sigma_mean'] = np.mean(sigma_vals)
        stats['sigma_median'] = np.median(sigma_vals)
        stats['sigma_std'] = np.std(sigma_vals)
        stats['sigma_min'] = np.min(sigma_vals)
        stats['sigma_max'] = np.max(sigma_vals)
    else:
        stats['sigma_mean'] = None

    # B* statistics
    if len(B_star_vals) > 0:
        stats['Bstar_mean'] = np.mean(B_star_vals)
        stats['Bstar_median'] = np.median(B_star_vals)
        stats['Bstar_std'] = np.std(B_star_vals)
        stats['Bstar_min'] = np.min(B_star_vals)
        stats['Bstar_max'] = np.max(B_star_vals)

    # D_product statistics
    stats['D_mean'] = np.mean(D_prod_vals)
    stats['D_median'] = np.median(D_prod_vals)
    stats['D_std'] = np.std(D_prod_vals)
    stats['D_min'] = np.min(D_prod_vals)
    stats['D_max'] = np.max(D_prod_vals)

    # Correlations
    logD = np.log(D_prod_vals)
    logDPhi = np.log(DPhi_vals)
    stats['corr_logD_DPhi'] = np.corrcoef(logD, DPhi_vals)[0, 1]
    stats['corr_logD_logDPhi'] = np.corrcoef(logD, logDPhi)[0, 1]

    # Fit the tail: exponential and power-law
    # Use histogram bin centers for fitting
    n_bins = 50
    counts, bin_edges = np.histogram(DPhi_vals, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)
    # Normalize to density
    density = counts / (len(DPhi_vals) * bin_widths)

    # Only fit bins with nonzero counts
    mask = density > 0
    bc_fit = bin_centers[mask]
    dens_fit = density[mask]

    # Exponential fit: log(p) = -alpha * DPhi + const
    # Use least squares on log-density
    log_dens = np.log(dens_fit)
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

    # Power-law fit: log(p) = -beta * log(DPhi) + const
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

    # Histogram data for reporting (text-based)
    stats['histogram'] = {'bin_centers': bc_fit.tolist(), 'density': dens_fit.tolist()}

    return stats


# ================================================================
# Write results markdown
# ================================================================

def write_results(exp1_results, exp2_results, exp2_analysis):
    """Write all results to markdown file."""
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'STEP13_CASCADE_BARRIER_RESULTS.md')

    with open(out_path, 'w') as f:
        f.write("# Step 13: Channel Cascade & Barrier Distribution Results\n\n")
        f.write(f"*Generated {time.strftime('%Y-%m-%d %H:%M')}*\n\n")

        # ============================================================
        # KEY FINDINGS
        # ============================================================
        f.write("## Key Findings\n\n")

        # Summarize Experiment 1
        f.write("### Experiment 1: Channel Cascade\n\n")

        # Get stage 0 barrier from first profile
        s0 = exp1_results[0]['stages'][0]
        DPhi0 = s0['DPhi'] if s0['bistable'] else None

        f.write(f"1. **Stage 0 (original lake model)**: DPhi = {DPhi0:.6f}\n")

        for pr in exp1_results:
            stages = pr['stages']
            f.write(f"\n2. **{pr['name']}** cascade:\n")
            for s in stages:
                k = s['k']
                if not s['bistable']:
                    f.write(f"   - k={k}: NOT BISTABLE\n")
                    continue
                line = f"   - k={k}: DPhi={s['DPhi']:.6f}"
                if s.get('sigma_star') is not None:
                    line += f", sigma*={s['sigma_star']:.6f}, D_product={s['D_product']:.1f}"
                    if s.get('B_star') is not None:
                        line += f", B*={s['B_star']:.4f}"
                f.write(line + "\n")

        # Summarize Experiment 2
        f.write("\n### Experiment 2: Barrier Distribution\n\n")
        for n_ch, an in exp2_analysis.items():
            f.write(f"- **k={n_ch}**: {an['n']}/{exp2_results[n_ch]['N_samples']} bistable "
                    f"({an['n']/exp2_results[n_ch]['N_samples']*100:.1f}%). "
                    f"DPhi: mean={an['DPhi_mean']:.6f}, median={an['DPhi_median']:.6f}, "
                    f"std={an['DPhi_std']:.6f}\n")
            f.write(f"  - Exponential fit: alpha={an['alpha_exp']:.4f}, R2={an['R2_exp']:.4f}\n")
            f.write(f"  - Power-law fit: beta={an['beta_pow']:.4f}, R2={an['R2_pow']:.4f}\n")
            if an['R2_exp'] > an['R2_pow']:
                f.write(f"  - **Exponential fits better** (R2_exp={an['R2_exp']:.4f} > R2_pow={an['R2_pow']:.4f})\n")
            else:
                f.write(f"  - **Power-law fits better** (R2_pow={an['R2_pow']:.4f} > R2_exp={an['R2_exp']:.4f})\n")

        # ============================================================
        # EXPERIMENT 1 DETAIL
        # ============================================================
        f.write("\n---\n\n")
        f.write("## Experiment 1: Channel Cascade — Full Results\n\n")

        for pr in exp1_results:
            f.write(f"### Profile: {pr['name']}  (epsilons = {pr['epsilons']})\n\n")

            # Summary table for all stages
            f.write("| Stage k | Channels | DPhi | sigma* | D_product | B* | x_cl | x_sd | lam_cl | lam_sd | tau |\n")
            f.write("|---------|----------|------|--------|-----------|-----|------|------|--------|--------|-----|\n")

            for s in pr['stages']:
                k = s['k']
                if not s['bistable']:
                    f.write(f"| {k} | {k} | NOT BISTABLE | | | | | | | | |\n")
                    continue

                sig = f"{s['sigma_star']:.6f}" if s.get('sigma_star') is not None else "N/F"
                Bs = f"{s['B_star']:.4f}" if s.get('B_star') is not None else "N/A"
                f.write(f"| {k} | {k} | {s['DPhi']:.6f} | {sig} | {s['D_product']:.1f} | "
                        f"{Bs} | {s['x_cl']:.6f} | {s['x_sd']:.6f} | "
                        f"{s['lam_cl']:+.6f} | {s['lam_sd']:+.6f} | {s['tau']:.6f} |\n")

            f.write("\n")

            # Duality check
            f.write("**Duality check (D_exact(sigma*)/D_product):**\n\n")
            for s in pr['stages']:
                k = s['k']
                if s.get('duality_ratio') is not None:
                    f.write(f"- k={k}: {s['duality_ratio']:.6f}\n")
                elif k == 0 and s.get('sigma_star') is not None and s.get('D_check') is not None:
                    f.write(f"- k=0: D_exact(sigma*)={s['D_check']:.6f} (D_product=1.0)\n")
            f.write("\n")

            # Transition metrics
            f.write("**Transitions between stages:**\n\n")
            f.write("| Transition | DPhi ratio | sigma* ratio | Saddle shift |\n")
            f.write("|------------|------------|--------------|-------------|\n")
            for idx, t in enumerate(pr['transitions']):
                prev_k = idx
                curr_k = idx + 1
                if t is None:
                    f.write(f"| {prev_k}->{curr_k} | N/A | N/A | N/A |\n")
                else:
                    sig_r = f"{t['sigma_ratio']:.6f}" if t['sigma_ratio'] is not None else "N/A"
                    f.write(f"| {prev_k}->{curr_k} | {t['DPhi_ratio']:.6f} | {sig_r} | "
                            f"{t['saddle_shift']:+.6f} ({t['saddle_shift']*100:+.2f}%) |\n")
            f.write("\n")

            # Noise fraction
            f.write("**Noise as fraction of state (sigma*/x_cl):**\n\n")
            for s in pr['stages']:
                k = s['k']
                if s.get('std_fraction') is not None:
                    f.write(f"- k={k}: {s['std_fraction']:.4f} ({s['std_fraction']*100:.2f}%)\n")
            f.write("\n")

        # ============================================================
        # EXPERIMENT 2 DETAIL
        # ============================================================
        f.write("---\n\n")
        f.write("## Experiment 2: Barrier Distribution — Full Results\n\n")

        for n_ch in [2, 3]:
            an = exp2_analysis[n_ch]
            res = exp2_results[n_ch]
            f.write(f"### Sweep {'A' if n_ch == 2 else 'B'}: k={n_ch} channels\n\n")

            f.write(f"**Samples**: {res['N_samples']} total, "
                    f"{res['n_bistable']} bistable ({res['n_bistable']/res['N_samples']*100:.1f}%), "
                    f"{res['n_monostable']} monostable\n\n")

            f.write("#### DPhi statistics (bistable configurations only)\n\n")
            f.write(f"| Statistic | Value |\n")
            f.write(f"|-----------|-------|\n")
            f.write(f"| N (bistable) | {an['n']} |\n")
            f.write(f"| Mean | {an['DPhi_mean']:.6f} |\n")
            f.write(f"| Median | {an['DPhi_median']:.6f} |\n")
            f.write(f"| Std | {an['DPhi_std']:.6f} |\n")
            f.write(f"| Min | {an['DPhi_min']:.6f} |\n")
            f.write(f"| Max | {an['DPhi_max']:.6f} |\n")
            f.write(f"| 25th pct | {an['DPhi_p25']:.6f} |\n")
            f.write(f"| 75th pct | {an['DPhi_p75']:.6f} |\n")
            f.write(f"| 90th pct | {an['DPhi_p90']:.6f} |\n")
            f.write(f"\n")

            f.write("#### Fraction above thresholds\n\n")
            f.write(f"| Threshold | Fraction |\n")
            f.write(f"|-----------|----------|\n")
            f.write(f"| >= median | {an['frac_ge_median']:.4f} |\n")
            f.write(f"| >= 75th pct | {an['frac_ge_p75']:.4f} |\n")
            f.write(f"| >= 90th pct | {an['frac_ge_p90']:.4f} |\n")
            f.write(f"\n")

            f.write("#### Distribution fits\n\n")
            f.write(f"| Fit | Parameter | Value | R2 |\n")
            f.write(f"|-----|-----------|-------|----|\n")
            f.write(f"| Exponential: p(DPhi) ~ exp(-alpha*DPhi) | alpha | {an['alpha_exp']:.4f} | {an['R2_exp']:.4f} |\n")
            f.write(f"| Power-law: p(DPhi) ~ DPhi^(-beta) | beta | {an['beta_pow']:.4f} | {an['R2_pow']:.4f} |\n")
            f.write(f"\n")

            if an['R2_exp'] > an['R2_pow']:
                f.write(f"**Exponential fit is better** (R2_exp={an['R2_exp']:.4f} > R2_pow={an['R2_pow']:.4f}).\n\n")
            else:
                f.write(f"**Power-law fit is better** (R2_pow={an['R2_pow']:.4f} > R2_exp={an['R2_exp']:.4f}).\n\n")

            f.write("#### D_product vs DPhi correlations\n\n")
            f.write(f"| Correlation | Value |\n")
            f.write(f"|-------------|-------|\n")
            f.write(f"| corr(log D_product, DPhi) | {an['corr_logD_DPhi']:.4f} |\n")
            f.write(f"| corr(log D_product, log DPhi) | {an['corr_logD_logDPhi']:.4f} |\n")
            f.write(f"\n")

            f.write("#### sigma* statistics\n\n")
            if an.get('sigma_mean') is not None:
                f.write(f"| Statistic | Value |\n")
                f.write(f"|-----------|-------|\n")
                f.write(f"| N (sigma* found) | {an['n_sigma']} |\n")
                f.write(f"| Mean | {an['sigma_mean']:.6f} |\n")
                f.write(f"| Median | {an['sigma_median']:.6f} |\n")
                f.write(f"| Std | {an['sigma_std']:.6f} |\n")
                f.write(f"| Min | {an['sigma_min']:.6f} |\n")
                f.write(f"| Max | {an['sigma_max']:.6f} |\n")
            else:
                f.write("No sigma* values found.\n")
            f.write(f"\n")

            f.write("#### B* (dimensionless barrier) statistics\n\n")
            if an.get('Bstar_mean') is not None:
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

    print(f"\nResults written to {out_path}")
    return out_path


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    t_start = time.time()

    # --- Experiment 1 ---
    exp1_results = run_experiment1()

    # --- Experiment 2 ---
    exp2_results = run_experiment2(N_samples=2000, rng_seed=42)

    # --- Analysis ---
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    exp2_analysis = {}
    for n_ch in [2, 3]:
        records = exp2_results[n_ch]['records']
        bistable_records = [r for r in records if r['bistable']]
        print(f"\nAnalyzing k={n_ch}: {len(bistable_records)} bistable configs")
        an = analyze_distribution(bistable_records, f"k={n_ch}")
        exp2_analysis[n_ch] = an

        print(f"  DPhi: mean={an['DPhi_mean']:.6f}, median={an['DPhi_median']:.6f}, "
              f"std={an['DPhi_std']:.6f}")
        print(f"  DPhi range: [{an['DPhi_min']:.6f}, {an['DPhi_max']:.6f}]")
        print(f"  Exponential fit: alpha={an['alpha_exp']:.4f}, R2={an['R2_exp']:.4f}")
        print(f"  Power-law fit: beta={an['beta_pow']:.4f}, R2={an['R2_pow']:.4f}")
        print(f"  corr(logD, DPhi) = {an['corr_logD_DPhi']:.4f}")
        print(f"  corr(logD, logDPhi) = {an['corr_logD_logDPhi']:.4f}")
        if an.get('sigma_mean') is not None:
            print(f"  sigma*: mean={an['sigma_mean']:.6f}, median={an['sigma_median']:.6f}, "
                  f"std={an['sigma_std']:.6f}")

    # --- Write results ---
    out_path = write_results(exp1_results, exp2_results, exp2_analysis)

    t_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"ALL COMPLETE: {t_total:.1f}s total")
    print(f"{'='*70}")
