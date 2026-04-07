#!/usr/bin/env python3
"""
Study 30 — 2D SDE sigma sweep for tumor-immune and diabetes systems.

For each system, sweeps noise intensity sigma across ~12 values, runs
2D Euler-Maruyama SDE to measure MFPT, computes D = MFPT/tau and
B = 2*DeltaPhi/sigma^2 (where DeltaPhi is extracted from the Kramers fit).

Outputs: sweep_tumor.npz, sweep_diabetes.npz
         Each contains arrays: sigma, mfpt, lnD, B, frac_escaped

Dependencies: numpy, scipy, numba
Runtime: ~10-20 minutes total (tumor ~5 min, diabetes ~10 min)
"""
import numpy as np
from numba import njit, prange
from scipy.optimize import brentq, fsolve
import os
import time

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ====================================================================
# TUMOR-IMMUNE (Kuznetsov 1994)
# ====================================================================
# 2D ODE: dE/dt, dT/dt with additive isotropic noise sigma on both.

# Parameters (Kuznetsov 1994, BCL1 lymphoma in chimeric mice)
S_FIT    = 13000.0     # cells/day (effector cell source)
p_par    = 0.1245      # 1/day
g_par    = 2.019e7     # cells
m_par    = 3.422e-10   # 1/(day*cell)
d_par    = 0.0412      # 1/day
a_par    = 0.18        # 1/day
b_par    = 2.0e-9      # 1/cell
n_par    = 1.101e-7    # 1/(day*cell)


@njit
def drift_tumor(E, T, s_val):
    """2D drift for tumor-immune system."""
    denom = g_par + T
    f1 = s_val + p_par * E * T / denom - m_par * E * T - d_par * E
    f2 = a_par * T * (1.0 - b_par * T) - n_par * E * T
    return f1, f2


@njit(parallel=True)
def sde_tumor_kernel(N_trials, max_steps, E_init, T_init,
                     E_sad, sigma, dt, seed_base):
    """Run N_trials of 2D Euler-Maruyama for tumor-immune. Returns FPTs."""
    sqrt_dt = np.sqrt(dt)
    fpt = np.full(N_trials, np.nan)
    for trial in prange(N_trials):
        np.random.seed(seed_base + trial)
        E = E_init
        T = T_init
        for step in range(max_steps):
            f1, f2 = drift_tumor(E, T, S_FIT)
            E += f1 * dt + sigma * sqrt_dt * np.random.randn()
            T += f2 * dt + sigma * sqrt_dt * np.random.randn()
            if E < 1.0:
                E = 2.0 - E
                if E < 1.0:
                    E = 1.0
            if T < 1.0:
                T = 2.0 - T
                if T < 1.0:
                    T = 1.0
            if E < E_sad:
                fpt[trial] = (step + 1) * dt
                break
    return fpt


def find_tumor_equilibria(s_val):
    """Find dormant (high-E) and saddle equilibria for tumor-immune."""
    # T_qss(E) = max(0, (a - n*E) / (a*b))
    # Effective 1D: f_eff(E) = s + p*E*T/(g+T) - m*E*T - d*E with T=T_qss(E)
    def T_qss(E):
        return max(0.0, (a_par - n_par * E) / (a_par * b_par))

    def f_eff(E):
        T = T_qss(E)
        return s_val + p_par * E * T / (g_par + T) - m_par * E * T - d_par * E

    # Scan for roots
    E_max = a_par / n_par  # max E where T_qss > 0
    E_scan = np.linspace(100, E_max * 0.999, 500000)
    f_scan = np.array([f_eff(e) for e in E_scan])

    roots = []
    for i in range(len(f_scan) - 1):
        if f_scan[i] * f_scan[i + 1] < 0:
            try:
                roots.append(brentq(f_eff, E_scan[i], E_scan[i + 1]))
            except Exception:
                pass

    if len(roots) < 2:
        raise ValueError(f"Need >= 2 roots, found {len(roots)}")

    # Classify: highest E is dormant (stable), middle is saddle
    roots = sorted(roots)
    E_dorm = roots[-1]
    E_sad = roots[-2] if len(roots) >= 3 else roots[0]

    T_dorm = T_qss(E_dorm)
    T_sad = T_qss(E_sad)

    # 1D eigenvalue at dormant equilibrium
    dE = 1.0
    lam_1d = (f_eff(E_dorm + dE) - f_eff(E_dorm - dE)) / (2 * dE)

    return E_dorm, T_dorm, E_sad, T_sad, lam_1d


def run_tumor_sweep():
    """Run sigma sweep for tumor-immune 2D SDE."""
    print("=" * 65)
    print("TUMOR-IMMUNE 2D SDE SIGMA SWEEP")
    print("=" * 65)

    E_dorm, T_dorm, E_sad, T_sad, lam_1d = find_tumor_equilibria(S_FIT)
    tau = 1.0 / abs(lam_1d)

    print(f"  E_dorm = {E_dorm:.1f},  T_dorm = {T_dorm:.1f}")
    print(f"  E_sad  = {E_sad:.1f},   T_sad  = {T_sad:.1f}")
    print(f"  lam_1d = {lam_1d:.6f},  tau = {tau:.4f} days")
    print()

    # Sigma values: log-spaced from 40k to 350k
    sigma_values = np.logspace(np.log10(40000), np.log10(350000), 13)

    dt = 0.05
    N_trials = 300
    max_days = 20000
    max_steps = int(max_days / dt)

    print(f"  N_sigma = {len(sigma_values)},  N_trials = {N_trials}")
    print(f"  dt = {dt},  max_days = {max_days}")
    print()
    print(f"  {'sigma':>12s}  {'MFPT':>10s}  {'f_esc':>6s}  {'D':>10s}  {'ln(D)':>7s}")
    print(f"  {'-'*52}")

    # Warmup JIT
    _ = sde_tumor_kernel(2, 10, E_dorm, T_dorm, E_sad, 100000.0, dt, 0)

    results = {'sigma': [], 'mfpt': [], 'frac_escaped': [],
               'D': [], 'lnD': []}

    for sigma in sigma_values:
        t0 = time.time()
        fpt = sde_tumor_kernel(N_trials, max_steps, E_dorm, T_dorm,
                               E_sad, sigma, dt, int(sigma) % 100000)
        elapsed = time.time() - t0

        escaped = ~np.isnan(fpt)
        f_esc = np.sum(escaped) / N_trials

        if f_esc > 0.05:
            # Use only escaped trials for MFPT
            mfpt = np.mean(fpt[escaped])
        else:
            mfpt = max_days  # censored estimate

        D = mfpt / tau
        lnD = np.log(max(D, 1.0))

        results['sigma'].append(sigma)
        results['mfpt'].append(mfpt)
        results['frac_escaped'].append(f_esc)
        results['D'].append(D)
        results['lnD'].append(lnD)

        print(f"  {sigma:12.0f}  {mfpt:10.1f}  {f_esc:6.2f}  "
              f"{D:10.1f}  {lnD:7.3f}  ({elapsed:.1f}s)")

    # Convert to arrays
    for k in results:
        results[k] = np.array(results[k])

    # Kramers fit: ln(MFPT) = slope/sigma^2 + intercept
    valid = results['frac_escaped'] > 0.3
    if np.sum(valid) >= 3:
        inv_s2 = 1.0 / results['sigma'][valid]**2
        ln_mfpt = np.log(results['mfpt'][valid])
        coeffs = np.polyfit(inv_s2, ln_mfpt, 1)
        slope, intercept = coeffs
        DeltaPhi = slope / 2.0
        R2 = 1 - np.sum((ln_mfpt - np.polyval(coeffs, inv_s2))**2) / \
             np.sum((ln_mfpt - np.mean(ln_mfpt))**2)

        results['B'] = 2 * DeltaPhi / results['sigma']**2
        results['DeltaPhi'] = DeltaPhi
        results['intercept'] = intercept
        results['tau'] = tau

        beta_0 = np.mean(results['lnD'][valid] - results['B'][valid])

        print(f"\n  Kramers fit (n={np.sum(valid)}):")
        print(f"    DeltaPhi = {DeltaPhi:.4e}")
        print(f"    intercept = {intercept:.4f}")
        print(f"    R^2 = {R2:.6f}")
        print(f"    beta_0 = {beta_0:.3f}")
    else:
        results['B'] = np.full_like(results['sigma'], np.nan)
        results['DeltaPhi'] = np.nan
        print("\n  WARNING: too few valid points for Kramers fit")

    outpath = os.path.join(OUTDIR, 'sweep_tumor.npz')
    np.savez(outpath, **results)
    print(f"\n  Saved {outpath}")
    return results


# ====================================================================
# DIABETES (Topp 2000)
# ====================================================================
# 3D -> 2D adiabatic reduction: (G, beta) after eliminating fast insulin I.
# Noise: additive on G, multiplicative on beta.
# eta scales both: sigma_G = eta * SIGMA_G_BASE, sigma_beta = eta * SIGMA_BETA_BASE

R0       = 864.0       # mg/dL/day
EG0      = 1.44        # 1/day
SI       = 0.72        # mL/(muU*day)
SIGMA_P  = 43.2        # muU/(mg*day)
ALPHA    = 20000.0     # (mg/dL)^2
K_INS    = 432.0       # 1/day
R1       = 0.84e-3     # 1/(mg/dL*day)
R2       = 2.4e-6      # 1/(mg/dL)^2/day

SIGMA_G_BASE    = 10.0   # mg/dL/sqrt(day)
SIGMA_BETA_BASE = 0.01   # 1/sqrt(day)
BETA_MIN        = 0.1    # reflecting boundary


@njit
def drift_diabetes(G, beta, d0):
    """2D drift for Topp diabetes model after eliminating insulin."""
    I_val = SIGMA_P * beta * G * G / (K_INS * (ALPHA + G * G))
    f_G = R0 - (EG0 + SI * I_val) * G
    f_beta = (-d0 + R1 * G - R2 * G * G) * beta
    return f_G, f_beta


@njit(parallel=True)
def sde_diabetes_kernel(N_trials, max_steps, G_init, beta_init,
                        G_sad, beta_sad, d0, sigma_G, sigma_beta,
                        dt, seed_base):
    """Run N_trials of 2D Euler-Maruyama for diabetes. Returns FPTs."""
    sqrt_dt = np.sqrt(dt)
    fpt = np.full(N_trials, np.nan)
    for trial in prange(N_trials):
        np.random.seed(seed_base + trial)
        G = G_init
        beta = beta_init
        for step in range(max_steps):
            f_G, f_beta = drift_diabetes(G, beta, d0)
            G += f_G * dt + sigma_G * sqrt_dt * np.random.randn()
            beta += f_beta * dt + sigma_beta * beta * sqrt_dt * np.random.randn()
            # Reflecting boundaries
            if beta < BETA_MIN:
                beta = 2.0 * BETA_MIN - beta
                if beta < BETA_MIN:
                    beta = BETA_MIN
            if G < 0.1:
                G = 0.1
            # Escape: beta < beta_saddle OR G > G_saddle
            if beta < beta_sad or G > G_sad:
                fpt[trial] = (step + 1) * dt
                break
    return fpt


def find_diabetes_equilibria(d0):
    """Find healthy and saddle equilibria for Topp diabetes model."""
    # Beta-nullcline: d0 = R1*G - R2*G^2, two roots
    disc = R1**2 - 4 * R2 * d0
    if disc < 0:
        raise ValueError(f"No bistability at d0={d0}")
    G_healthy = (R1 - np.sqrt(disc)) / (2 * R2)
    G_saddle = (R1 + np.sqrt(disc)) / (2 * R2)

    # G-nullcline: G_qss(beta) from R0 - (EG0 + SI*I(G,beta))*G = 0
    def G_qss(beta):
        def residual(G):
            I_val = SIGMA_P * beta * G**2 / (K_INS * (ALPHA + G**2))
            return R0 - (EG0 + SI * I_val) * G
        return brentq(residual, 1.0, 800.0)

    # Get beta at each G from beta-nullcline: beta = arbitrary, but at
    # equilibrium both nullclines intersect.
    # Actually: at equilibrium, G is on beta-nullcline AND G-nullcline.
    # From beta-nullcline: d0 = R1*G - R2*G^2 -> G = G_healthy or G_saddle.
    # From G-nullcline: solve for beta given G.
    def beta_from_G(G):
        # R0 - (EG0 + SI * I(G,beta)) * G = 0
        # I = SIGMA_P * beta * G^2 / (K_INS * (ALPHA + G^2))
        # R0 = (EG0 + SI * SIGMA_P * beta * G^2 / (K_INS*(ALPHA+G^2))) * G
        # R0/G - EG0 = SI * SIGMA_P * beta * G^2 / (K_INS*(ALPHA+G^2))
        # beta = (R0/G - EG0) * K_INS * (ALPHA + G^2) / (SI * SIGMA_P * G^2)
        return (R0 / G - EG0) * K_INS * (ALPHA + G**2) / (SI * SIGMA_P * G**2)

    beta_healthy = beta_from_G(G_healthy)
    beta_saddle = beta_from_G(G_saddle)

    # 2D Jacobian at healthy equilibrium
    G, beta = G_healthy, beta_healthy
    I_val = SIGMA_P * beta * G**2 / (K_INS * (ALPHA + G**2))
    dI_dG = SIGMA_P * beta * 2 * G * ALPHA / (K_INS * (ALPHA + G**2)**2)
    dI_dbeta = SIGMA_P * G**2 / (K_INS * (ALPHA + G**2))

    J = np.array([
        [-(EG0 + SI * I_val) - SI * dI_dG * G,   -SI * dI_dbeta * G],
        [(R1 - 2 * R2 * G) * beta,                -d0 + R1 * G - R2 * G**2]
    ])
    eigs = np.linalg.eigvals(J)
    lam_slow = np.max(np.real(eigs))  # least negative
    tau = 1.0 / abs(lam_slow)

    return G_healthy, beta_healthy, G_saddle, beta_saddle, tau, eigs


def run_diabetes_sweep():
    """Run eta sweep for diabetes 2D SDE."""
    print()
    print("=" * 65)
    print("DIABETES 2D SDE ETA SWEEP")
    print("=" * 65)

    d0 = 0.06  # representative operating point
    G_h, beta_h, G_s, beta_s, tau, eigs = find_diabetes_equilibria(d0)

    print(f"  d0 = {d0}")
    print(f"  Healthy: G = {G_h:.2f} mg/dL,  beta = {beta_h:.4f}")
    print(f"  Saddle:  G = {G_s:.2f} mg/dL,  beta = {beta_s:.4f}")
    print(f"  Eigenvalues: {np.real(eigs)}")
    print(f"  tau = {tau:.2f} days")
    print()

    # Eta values: concentrate in Kramers regime (D > 1 at eta ~ 5-15)
    eta_values = np.array([5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10, 11, 13, 16, 20])

    dt = 0.01
    N_trials = 400
    max_days = 80000
    max_steps = int(max_days / dt)

    print(f"  N_eta = {len(eta_values)},  N_trials = {N_trials}")
    print(f"  dt = {dt},  max_days = {max_days}")
    print()
    print(f"  {'eta':>6s}  {'sig_G':>8s}  {'sig_b':>8s}  "
          f"{'MFPT':>10s}  {'f_esc':>6s}  {'D':>10s}  {'ln(D)':>7s}")
    print(f"  {'-'*64}")

    # Warmup JIT
    _ = sde_diabetes_kernel(2, 10, G_h, beta_h, G_s, beta_s, d0,
                            10.0, 0.01, dt, 0)

    results = {'eta': [], 'sigma_G': [], 'sigma_beta': [],
               'mfpt': [], 'frac_escaped': [], 'D': [], 'lnD': []}

    for eta in eta_values:
        sigma_G = eta * SIGMA_G_BASE
        sigma_beta = eta * SIGMA_BETA_BASE

        t0 = time.time()
        fpt = sde_diabetes_kernel(N_trials, max_steps, G_h, beta_h,
                                  G_s, beta_s, d0,
                                  sigma_G, sigma_beta, dt,
                                  int(eta * 1000) % 100000)
        elapsed = time.time() - t0

        escaped = ~np.isnan(fpt)
        f_esc = np.sum(escaped) / N_trials

        if f_esc > 0.05:
            mfpt = np.mean(fpt[escaped])
        else:
            mfpt = max_days

        D = mfpt / tau
        lnD = np.log(max(D, 1.0))

        results['eta'].append(eta)
        results['sigma_G'].append(sigma_G)
        results['sigma_beta'].append(sigma_beta)
        results['mfpt'].append(mfpt)
        results['frac_escaped'].append(f_esc)
        results['D'].append(D)
        results['lnD'].append(lnD)

        print(f"  {eta:6.0f}  {sigma_G:8.1f}  {sigma_beta:8.4f}  "
              f"{mfpt:10.1f}  {f_esc:6.2f}  {D:10.1f}  {lnD:7.3f}  "
              f"({elapsed:.1f}s)")

    # Convert to arrays
    for k in results:
        results[k] = np.array(results[k])

    # Kramers fit: ln(MFPT) = slope/eta^2 + intercept
    valid = results['frac_escaped'] > 0.3
    if np.sum(valid) >= 3:
        inv_e2 = 1.0 / results['eta'][valid]**2
        ln_mfpt = np.log(results['mfpt'][valid])
        coeffs = np.polyfit(inv_e2, ln_mfpt, 1)
        slope, intercept = coeffs
        DeltaPhi_eff = slope / 2.0
        R2_fit = 1 - np.sum((ln_mfpt - np.polyval(coeffs, inv_e2))**2) / \
                 np.sum((ln_mfpt - np.mean(ln_mfpt))**2)

        results['B'] = 2 * DeltaPhi_eff / results['eta']**2
        results['DeltaPhi_eff'] = DeltaPhi_eff
        results['intercept'] = intercept
        results['tau'] = tau

        beta_0 = np.mean(results['lnD'][valid] - results['B'][valid])

        print(f"\n  Kramers fit (n={np.sum(valid)}):")
        print(f"    DeltaPhi_eff = {DeltaPhi_eff:.4f}")
        print(f"    intercept = {intercept:.4f}")
        print(f"    R^2 = {R2_fit:.6f}")
        print(f"    beta_0 = {beta_0:.3f}")
    else:
        results['B'] = np.full_like(results['eta'], np.nan)
        results['DeltaPhi_eff'] = np.nan
        print("\n  WARNING: too few valid points for Kramers fit")

    outpath = os.path.join(OUTDIR, 'sweep_diabetes.npz')
    np.savez(outpath, **results)
    print(f"\n  Saved {outpath}")
    return results


# ====================================================================
# Main
# ====================================================================

if __name__ == '__main__':
    t_start = time.time()
    res_tumor = run_tumor_sweep()
    res_diabetes = run_diabetes_sweep()
    print(f"\nTotal elapsed: {time.time() - t_start:.0f}s")
