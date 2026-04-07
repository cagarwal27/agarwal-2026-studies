#!/usr/bin/env python3
"""
Toggle Switch Shortcut Formula
===============================
Derives an explicit closed-form D(alpha, Omega) for the Gardner-Collins
toggle switch (Hill n=2) by:

1. Dense CME scan: alpha from 3.0 to 12.0 (step 0.5), Omega = 2,3,4
2. Fit ln(D) = a(alpha) + S(alpha)*Omega at each alpha
3. Fit S(alpha) to candidate closed-form functions
4. Compute Kramers-Langer prefactor A(alpha)
5. Validate D_formula vs D_CME

Final formula: D_toggle(alpha, Omega) = A(alpha) * exp(S_fit(alpha) * Omega)
"""

import numpy as np
from scipy.optimize import fsolve, curve_fit
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigs
import warnings
import time as _t

warnings.filterwarnings('ignore')

N_HILL = 2

# ============================================================
# Deterministic analysis
# ============================================================

def find_fixed_points(alpha):
    """Find (eq_hi, eq_lo, saddle) for symmetric toggle with Hill n=2."""
    n = N_HILL
    xs = fsolve(lambda x: x - alpha/(1+x**n), alpha**(1.0/(n+1)))[0]
    uh, vl = fsolve(lambda uv: [alpha/(1+uv[1]**n) - uv[0],
                                 alpha/(1+uv[0]**n) - uv[1]],
                    [alpha - 0.1, 0.1])
    return (uh, vl), (vl, uh), (xs, xs)


def jacobian(u, v, alpha):
    n = N_HILL
    return np.array([
        [-1.0, -alpha*n*v**(n-1) / (1+v**n)**2],
        [-alpha*n*u**(n-1) / (1+u**n)**2, -1.0]
    ])


def tau_relax(alpha):
    """Relaxation time = 1/|slowest eigenvalue at equilibrium|."""
    eq, _, _ = find_fixed_points(alpha)
    J = jacobian(eq[0], eq[1], alpha)
    evals = np.real(np.linalg.eigvals(J))
    return 1.0 / np.min(np.abs(evals))


def kramers_prefactor(alpha):
    """
    Kramers-Langer prefactor for 2D non-gradient system.

    At minimum: eigenvalues lambda_1, lambda_2 (both negative)
    At saddle: lambda_u (positive, unstable), lambda_s (negative, stable)

    Prefactor A = |lambda_u| / (2*pi) * prod(|lambda_i^min|) / |lambda_s^saddle|

    Returns (A, eig_min, eig_saddle).
    """
    eq, _, sad = find_fixed_points(alpha)
    J_min = jacobian(eq[0], eq[1], alpha)
    J_sad = jacobian(sad[0], sad[1], alpha)

    eig_min = np.real(np.linalg.eigvals(J_min))
    eig_sad = np.real(np.linalg.eigvals(J_sad))

    lambda_min_prod = np.prod(np.abs(eig_min))
    lambda_u = np.max(eig_sad)      # positive (unstable)
    lambda_s_abs = np.min(np.abs(eig_sad))  # |stable eigenvalue at saddle|

    A = lambda_u / (2 * np.pi) * lambda_min_prod / lambda_s_abs
    return A, eig_min, eig_sad


# ============================================================
# CME builder and solver
# ============================================================

def build_cme(alpha, Omega, nmax):
    """Build transition rate matrix Q for toggle CME."""
    n = N_HILL
    M = nmax * nmax
    Q = lil_matrix((M, M), dtype=float)
    for nu in range(nmax):
        for nv in range(nmax):
            i = nu * nmax + nv
            cu = nu / Omega
            cv = nv / Omega
            # Birth of U
            rpu = Omega * alpha / (1 + cv**n) if cv > 0 else Omega * alpha
            # Death of U
            rdu = float(nu)
            # Birth of V
            rpv = Omega * alpha / (1 + cu**n) if cu > 0 else Omega * alpha
            # Death of V
            rdv = float(nv)

            if nu + 1 < nmax:
                j = (nu + 1) * nmax + nv
                Q[j, i] += rpu
                Q[i, i] -= rpu
            if nu > 0:
                j = (nu - 1) * nmax + nv
                Q[j, i] += rdu
                Q[i, i] -= rdu
            if nv + 1 < nmax:
                j = nu * nmax + (nv + 1)
                Q[j, i] += rpv
                Q[i, i] -= rpv
            if nv > 0:
                j = nu * nmax + (nv - 1)
                Q[j, i] += rdv
                Q[i, i] -= rdv
    return Q.tocsc()


def get_D_cme(alpha, Omega, nmax=None):
    """
    Compute D from CME spectral gap.
    D = MFPT / tau_relax = 2 / (|lambda_2| * tau_relax)

    For symmetric bistable system, lambda_2 gives sum of forward+backward
    rates, and MFPT per state = 2/|lambda_2| by symmetry.
    """
    if nmax is None:
        nmax = int(2.5 * alpha * Omega) + 15
        nmax = min(nmax, 300)

    N_states = nmax**2
    if N_states > 300000:
        return None, nmax

    Q = build_cme(alpha, Omega, nmax)
    ev, _ = eigs(Q, k=6, sigma=0, which='LM')
    ev = ev[np.argsort(np.abs(np.real(ev)))]
    lam2 = np.abs(np.real(ev[1]))

    tau = tau_relax(alpha)
    mfpt = 2.0 / lam2
    D = mfpt / tau
    return D, nmax


# ============================================================
# Candidate S(alpha) functions
# ============================================================

def power_law(a, c1, ac, p):
    return c1 * (a - ac)**p

def rational(a, c1, ac, c2):
    return c1 * (a - ac) / (a + c2)

def logarithmic(a, c1, ac, p):
    return c1 * np.log(a / ac)**p

def poly2(a, c1, c2, ac):
    return c1 * (a - ac) + c2 * (a - ac)**2

def poly3(a, c1, c2, c3, ac):
    return c1 * (a - ac) + c2 * (a - ac)**2 + c3 * (a - ac)**3

def sqrt_form(a, c1, ac, c2):
    """S = c1 * sqrt(a - ac) * (1 + c2*(a-ac))"""
    return c1 * np.sqrt(a - ac) * (1 + c2 * (a - ac))


# ============================================================
# Main computation
# ============================================================

def main():
    print("=" * 76)
    print("TOGGLE SWITCH SHORTCUT FORMULA DERIVATION")
    print("=" * 76)
    print("Model: du/dt = alpha/(1+v^2) - u,  dv/dt = alpha/(1+u^2) - v")
    print("Goal: D(alpha, Omega) = A(alpha) * exp(S(alpha) * Omega)")
    print()

    # ----------------------------------------------------------
    # Step 1: Dense scan
    # ----------------------------------------------------------
    alpha_values = np.arange(3.0, 12.5, 0.5)
    # Use 4 Omega values for more robust linear fits
    Omega_values = [2, 3, 4, 5]

    print(f"Scanning {len(alpha_values)} alpha values: {alpha_values[0]:.1f} to {alpha_values[-1]:.1f}")
    print(f"Omega values: {Omega_values}")
    print()

    # Check bistability threshold
    print("Checking bistability thresholds...")
    for alpha in [2.0, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]:
        try:
            eq, _, sad = find_fixed_points(alpha)
            sep = abs(eq[0] - sad[0])
            print(f"  alpha={alpha:.1f}: u_eq={eq[0]:.4f}, u_sad={sad[0]:.4f}, separation={sep:.4f}")
        except:
            print(f"  alpha={alpha:.1f}: no fixed points found")
    print()

    # Compute D at all (alpha, Omega) combinations
    results = {}  # results[alpha] = {Omega: D}

    print(f"{'alpha':>6} {'Omega':>5} {'n_max':>6} {'N_states':>8} {'D_CME':>14} {'ln(D)':>10} {'time':>6}")
    print("-" * 62)

    for alpha in alpha_values:
        results[alpha] = {}

        # Check bistability
        try:
            eq, _, sad = find_fixed_points(alpha)
            if abs(eq[0] - sad[0]) < 0.3:
                print(f"{alpha:>6.1f}  Not sufficiently bistable, skipping")
                continue
        except:
            print(f"{alpha:>6.1f}  Fixed point solver failed, skipping")
            continue

        for Omega in Omega_values:
            t0 = _t.time()
            try:
                D, nmax = get_D_cme(alpha, Omega)
                dt = _t.time() - t0
                if D is not None and D > 0:
                    results[alpha][Omega] = D
                    print(f"{alpha:>6.1f} {Omega:>5} {nmax:>6} {nmax**2:>8} {D:>14.4f} {np.log(D):>10.4f} {dt:>6.1f}s")
                else:
                    print(f"{alpha:>6.1f} {Omega:>5} {'':>6} {'':>8} {'SKIP (too large)':>14}")
            except Exception as e:
                dt = _t.time() - t0
                print(f"{alpha:>6.1f} {Omega:>5} {'':>6} {'':>8} {'FAIL':>14} {str(e)[:30]:>30} {dt:>6.1f}s")

    # ----------------------------------------------------------
    # Step 2: Fit ln(D) = a + S*Omega at each alpha
    #         with outlier detection
    # ----------------------------------------------------------
    print(f"\n{'='*76}")
    print("STEP 2: LINEAR FIT ln(D) = a(alpha) + S(alpha) * Omega")
    print(f"{'='*76}\n")

    S_data = []  # (alpha, S, intercept, R^2)

    print(f"{'alpha':>6} {'S':>10} {'intercept':>10} {'R^2':>8} {'RMSE':>8} {'points':>6}")
    print("-" * 54)

    # Track outliers for removal from global fits
    outlier_points = set()  # (alpha, Omega) pairs to exclude

    for alpha in alpha_values:
        if alpha not in results or len(results[alpha]) < 2:
            continue

        omegas = sorted(results[alpha].keys())
        x = np.array(omegas, dtype=float)
        y = np.array([np.log(results[alpha][om]) for om in omegas])

        # Linear fit
        coeffs = np.polyfit(x, y, 1)
        S = coeffs[0]
        intercept = coeffs[1]
        y_fit = np.polyval(coeffs, x)
        residuals = np.abs(y - y_fit)
        ss_res = np.sum((y - y_fit)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        rmse = np.sqrt(np.mean((y - y_fit)**2))

        # Detect outliers: if any residual > 3*rmse of other points, flag it
        if R2 < 0.99 and len(omegas) >= 3:
            # Try removing each point and see if fit improves dramatically
            best_R2_leave1 = R2
            worst_idx = -1
            for k in range(len(omegas)):
                x_k = np.delete(x, k)
                y_k = np.delete(y, k)
                c_k = np.polyfit(x_k, y_k, 1)
                y_fit_k = np.polyval(c_k, x_k)
                ss_r_k = np.sum((y_k - y_fit_k)**2)
                ss_t_k = np.sum((y_k - np.mean(y_k))**2)
                R2_k = 1 - ss_r_k / ss_t_k if ss_t_k > 0 else 1.0
                if R2_k > best_R2_leave1 + 0.05:  # substantial improvement
                    best_R2_leave1 = R2_k
                    worst_idx = k

            if worst_idx >= 0 and best_R2_leave1 > 0.99:
                om_bad = omegas[worst_idx]
                outlier_points.add((alpha, om_bad))
                print(f"{alpha:>6.1f} {S:>10.4f} {intercept:>10.4f} {R2:>8.6f} {rmse:>8.4f} {len(omegas):>6}  ** OUTLIER at Om={om_bad}")

                # Refit without outlier
                x_clean = np.delete(x, worst_idx)
                y_clean = np.delete(y, worst_idx)
                coeffs = np.polyfit(x_clean, y_clean, 1)
                S = coeffs[0]
                intercept = coeffs[1]
                y_fit = np.polyval(coeffs, x_clean)
                ss_res = np.sum((y_clean - y_fit)**2)
                ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
                R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
                rmse = np.sqrt(np.mean((y_clean - y_fit)**2))
                print(f"{'':>6} {S:>10.4f} {intercept:>10.4f} {R2:>8.6f} {rmse:>8.4f} {len(x_clean):>6}  (refit without outlier)")
            else:
                print(f"{alpha:>6.1f} {S:>10.4f} {intercept:>10.4f} {R2:>8.6f} {rmse:>8.4f} {len(omegas):>6}")
        else:
            print(f"{alpha:>6.1f} {S:>10.4f} {intercept:>10.4f} {R2:>8.6f} {rmse:>8.4f} {len(omegas):>6}")

        S_data.append((alpha, S, intercept, R2, rmse))

    if outlier_points:
        print(f"\n  Outlier points excluded: {outlier_points}")

    if len(S_data) < 3:
        print("\nInsufficient data points. Exiting.")
        return

    alphas_fit = np.array([d[0] for d in S_data])
    S_values = np.array([d[1] for d in S_data])
    intercepts = np.array([d[2] for d in S_data])

    # ----------------------------------------------------------
    # Step 3: Fit S(alpha) to candidate functions
    # ----------------------------------------------------------
    print(f"\n{'='*76}")
    print("STEP 3: FIT S(alpha) TO CLOSED-FORM FUNCTIONS")
    print(f"{'='*76}\n")

    # Known: alpha_c (pitchfork bifurcation) is around 2 for n=2
    # Exact: for x = alpha/(1+x^2), bistability when alpha > 2
    # Actually for n=2 symmetric toggle: bifurcation at alpha = 2
    # Let's verify
    print("Bifurcation point analysis:")
    print("  For n=2 toggle: bifurcation at alpha_c where the cubic")
    print("  x^3 + x - alpha = 0 transitions from 1 to 3 real roots.")
    print("  alpha_c = 2.0 for the symmetric case (saddle-node at x=1).")
    print()

    candidates = {}

    # --- Candidate 1: Power law S = c1 * (alpha - alpha_c)^p ---
    try:
        popt, pcov = curve_fit(power_law, alphas_fit, S_values,
                               p0=[0.1, 2.0, 1.5], bounds=([0, 1.0, 0.1], [10, 2.5, 5.0]))
        S_pred = power_law(alphas_fit, *popt)
        ss_res = np.sum((S_values - S_pred)**2)
        ss_tot = np.sum((S_values - np.mean(S_values))**2)
        R2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((S_values - S_pred)**2))
        candidates['power_law'] = {
            'params': {'c1': popt[0], 'alpha_c': popt[1], 'p': popt[2]},
            'formula': f'S = {popt[0]:.6f} * (alpha - {popt[1]:.4f})^{popt[2]:.4f}',
            'R2': R2, 'rmse': rmse, 'pred': S_pred,
            'func': lambda a, p=popt: power_law(a, *p)
        }
        print(f"Power law:   S = {popt[0]:.6f} * (a - {popt[1]:.4f})^{popt[2]:.4f}")
        print(f"             R^2 = {R2:.8f},  RMSE = {rmse:.6f}")
    except Exception as e:
        print(f"Power law:   FAILED — {e}")

    # --- Candidate 2: Rational S = c1 * (alpha - alpha_c) / (alpha + c2) ---
    try:
        popt, pcov = curve_fit(rational, alphas_fit, S_values,
                               p0=[1.0, 2.0, 1.0], bounds=([0, 1.0, -10], [10, 2.5, 50]))
        S_pred = rational(alphas_fit, *popt)
        ss_res = np.sum((S_values - S_pred)**2)
        ss_tot = np.sum((S_values - np.mean(S_values))**2)
        R2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((S_values - S_pred)**2))
        candidates['rational'] = {
            'params': {'c1': popt[0], 'alpha_c': popt[1], 'c2': popt[2]},
            'formula': f'S = {popt[0]:.6f} * (a - {popt[1]:.4f}) / (a + {popt[2]:.4f})',
            'R2': R2, 'rmse': rmse, 'pred': S_pred,
            'func': lambda a, p=popt: rational(a, *p)
        }
        print(f"Rational:    S = {popt[0]:.6f} * (a - {popt[1]:.4f}) / (a + {popt[2]:.4f})")
        print(f"             R^2 = {R2:.8f},  RMSE = {rmse:.6f}")
    except Exception as e:
        print(f"Rational:    FAILED — {e}")

    # --- Candidate 3: Logarithmic S = c1 * ln(alpha/alpha_c)^p ---
    try:
        popt, pcov = curve_fit(logarithmic, alphas_fit, S_values,
                               p0=[1.0, 2.0, 1.0], bounds=([0, 1.0, 0.1], [10, 2.5, 5.0]))
        S_pred = logarithmic(alphas_fit, *popt)
        ss_res = np.sum((S_values - S_pred)**2)
        ss_tot = np.sum((S_values - np.mean(S_values))**2)
        R2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((S_values - S_pred)**2))
        candidates['logarithmic'] = {
            'params': {'c1': popt[0], 'alpha_c': popt[1], 'p': popt[2]},
            'formula': f'S = {popt[0]:.6f} * ln(a / {popt[1]:.4f})^{popt[2]:.4f}',
            'R2': R2, 'rmse': rmse, 'pred': S_pred,
            'func': lambda a, p=popt: logarithmic(a, *p)
        }
        print(f"Logarithmic: S = {popt[0]:.6f} * ln(a / {popt[1]:.4f})^{popt[2]:.4f}")
        print(f"             R^2 = {R2:.8f},  RMSE = {rmse:.6f}")
    except Exception as e:
        print(f"Logarithmic: FAILED — {e}")

    # --- Candidate 4: Quadratic S = c1*(a-ac) + c2*(a-ac)^2 ---
    try:
        popt, pcov = curve_fit(poly2, alphas_fit, S_values,
                               p0=[0.1, 0.01, 2.0], bounds=([-1, -0.1, 1.0], [1, 0.1, 2.5]))
        S_pred = poly2(alphas_fit, *popt)
        ss_res = np.sum((S_values - S_pred)**2)
        ss_tot = np.sum((S_values - np.mean(S_values))**2)
        R2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((S_values - S_pred)**2))
        candidates['poly2'] = {
            'params': {'c1': popt[0], 'c2': popt[1], 'alpha_c': popt[2]},
            'formula': f'S = {popt[0]:.6f}*(a - {popt[2]:.4f}) + {popt[1]:.6f}*(a - {popt[2]:.4f})^2',
            'R2': R2, 'rmse': rmse, 'pred': S_pred,
            'func': lambda a, p=popt: poly2(a, *p)
        }
        print(f"Poly2:       S = {popt[0]:.6f}*(a - {popt[2]:.4f}) + {popt[1]:.6f}*(a - {popt[2]:.4f})^2")
        print(f"             R^2 = {R2:.8f},  RMSE = {rmse:.6f}")
    except Exception as e:
        print(f"Poly2:       FAILED — {e}")

    # --- Candidate 5: Cubic polynomial ---
    try:
        popt, pcov = curve_fit(poly3, alphas_fit, S_values,
                               p0=[0.1, 0.01, 0.001, 2.0],
                               bounds=([-1, -0.1, -0.01, 1.0], [1, 0.1, 0.01, 2.5]))
        S_pred = poly3(alphas_fit, *popt)
        ss_res = np.sum((S_values - S_pred)**2)
        ss_tot = np.sum((S_values - np.mean(S_values))**2)
        R2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((S_values - S_pred)**2))
        candidates['poly3'] = {
            'params': {'c1': popt[0], 'c2': popt[1], 'c3': popt[2], 'alpha_c': popt[3]},
            'formula': f'S = {popt[0]:.6f}*(a-{popt[3]:.4f}) + {popt[1]:.6f}*(a-{popt[3]:.4f})^2 + {popt[2]:.6f}*(a-{popt[3]:.4f})^3',
            'R2': R2, 'rmse': rmse, 'pred': S_pred,
            'func': lambda a, p=popt: poly3(a, *p)
        }
        print(f"Poly3:       S = {popt[0]:.6f}*(a-{popt[3]:.4f}) + {popt[1]:.6f}*(a-{popt[3]:.4f})^2 + {popt[2]:.6f}*(a-{popt[3]:.4f})^3")
        print(f"             R^2 = {R2:.8f},  RMSE = {rmse:.6f}")
    except Exception as e:
        print(f"Poly3:       FAILED — {e}")

    # --- Candidate 6: Sqrt form S = c1 * sqrt(a-ac) * (1 + c2*(a-ac)) ---
    try:
        popt, pcov = curve_fit(sqrt_form, alphas_fit, S_values,
                               p0=[0.2, 2.0, 0.01], bounds=([0, 1.0, -1], [5, 2.5, 1]))
        S_pred = sqrt_form(alphas_fit, *popt)
        ss_res = np.sum((S_values - S_pred)**2)
        ss_tot = np.sum((S_values - np.mean(S_values))**2)
        R2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((S_values - S_pred)**2))
        candidates['sqrt_form'] = {
            'params': {'c1': popt[0], 'alpha_c': popt[1], 'c2': popt[2]},
            'formula': f'S = {popt[0]:.6f} * sqrt(a - {popt[1]:.4f}) * (1 + {popt[2]:.6f}*(a - {popt[1]:.4f}))',
            'R2': R2, 'rmse': rmse, 'pred': S_pred,
            'func': lambda a, p=popt: sqrt_form(a, *p)
        }
        print(f"Sqrt form:   S = {popt[0]:.6f} * sqrt(a - {popt[1]:.4f}) * (1 + {popt[2]:.6f}*(a-{popt[1]:.4f}))")
        print(f"             R^2 = {R2:.8f},  RMSE = {rmse:.6f}")
    except Exception as e:
        print(f"Sqrt form:   FAILED — {e}")

    # --- Candidate 7: Simple linear (as baseline) ---
    try:
        def simple_linear(a, c1, ac):
            return c1 * (a - ac)
        popt, pcov = curve_fit(simple_linear, alphas_fit, S_values,
                               p0=[0.2, 2.0], bounds=([0, 1.0], [1, 2.5]))
        S_pred = simple_linear(alphas_fit, *popt)
        ss_res = np.sum((S_values - S_pred)**2)
        ss_tot = np.sum((S_values - np.mean(S_values))**2)
        R2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((S_values - S_pred)**2))
        candidates['linear'] = {
            'params': {'c1': popt[0], 'alpha_c': popt[1]},
            'formula': f'S = {popt[0]:.6f} * (a - {popt[1]:.4f})',
            'R2': R2, 'rmse': rmse, 'pred': S_pred,
            'func': lambda a, p=popt: simple_linear(a, *p)
        }
        print(f"Linear:      S = {popt[0]:.6f} * (a - {popt[1]:.4f})")
        print(f"             R^2 = {R2:.8f},  RMSE = {rmse:.6f}")
    except Exception as e:
        print(f"Linear:      FAILED — {e}")

    # ----------------------------------------------------------
    # Rank candidates
    # ----------------------------------------------------------
    print(f"\n{'='*76}")
    print("CANDIDATE RANKING (by RMSE)")
    print(f"{'='*76}\n")

    ranked = sorted(candidates.items(), key=lambda x: x[1]['rmse'])

    print(f"{'Rank':>4} {'Name':>15} {'R^2':>12} {'RMSE':>12} {'Formula'}")
    print("-" * 90)
    for i, (name, c) in enumerate(ranked):
        print(f"{i+1:>4} {name:>15} {c['R2']:>12.8f} {c['rmse']:>12.6f}   {c['formula']}")

    best_name, best = ranked[0]

    print(f"\nBest fit: {best_name}")
    print(f"  {best['formula']}")
    print(f"  R^2 = {best['R2']:.10f}")
    print(f"  RMSE = {best['rmse']:.8f}")

    # Show residuals for best fit
    print(f"\n  Residuals (S_data - S_fit):")
    S_best_pred = best['pred']
    for i, alpha in enumerate(alphas_fit):
        print(f"    alpha={alpha:5.1f}: S_data={S_values[i]:.4f}, S_fit={S_best_pred[i]:.4f}, "
              f"residual={S_values[i]-S_best_pred[i]:+.4f}")

    # ----------------------------------------------------------
    # Step 4: Kramers-Langer prefactor A(alpha)
    # ----------------------------------------------------------
    print(f"\n{'='*76}")
    print("STEP 4: KRAMERS-LANGER PREFACTOR A(alpha)")
    print(f"{'='*76}\n")

    print(f"{'alpha':>6} {'A_Kramers':>12} {'tau_relax':>10} {'A/tau':>12} {'ln(A/tau)':>10} {'intercept':>10} {'ratio':>8}")
    print("-" * 76)

    A_data = []
    for i, (alpha, S, intercept, R2, rmse) in enumerate(S_data):
        A, eig_min, eig_sad = kramers_prefactor(alpha)
        tau = tau_relax(alpha)
        A_over_tau = A / tau
        ln_A_tau = np.log(A_over_tau) if A_over_tau > 0 else float('nan')

        # The intercept from ln(D) = a + S*Omega should be close to ln(A/tau)
        # Actually D = MFPT/tau and MFPT = (2*pi/|lambda_u|) * (stuff) * exp(S*Omega)
        # so intercept ~ ln(prefactor expression)
        ratio = np.exp(intercept) / A_over_tau if A_over_tau > 0 else float('nan')

        A_data.append((alpha, A, tau, A_over_tau, ln_A_tau, intercept))
        print(f"{alpha:>6.1f} {A:>12.6f} {tau:>10.4f} {A_over_tau:>12.6f} {ln_A_tau:>10.4f} {intercept:>10.4f} {ratio:>8.2f}")

    # Fit A(alpha) or the effective prefactor from intercept
    print(f"\n  Note: The 'ratio' column shows exp(intercept) / (A_Kramers/tau).")
    print(f"  If the Kramers prefactor formula were exact, ratio ~ 1.")
    print(f"  Deviations indicate the 2D non-gradient correction factor.")

    # Use the actual fitted intercepts as the effective prefactor
    # D_eff(alpha, Omega) = exp(intercept(alpha)) * exp(S(alpha) * Omega)
    # = exp(a(alpha) + S(alpha) * Omega)

    # ----------------------------------------------------------
    # Step 5: Build and validate the shortcut formula
    # ----------------------------------------------------------
    print(f"\n{'='*76}")
    print("STEP 5: SHORTCUT FORMULA CONSTRUCTION AND VALIDATION")
    print(f"{'='*76}\n")

    # The full formula is:
    # ln(D) = a(alpha) + S(alpha) * Omega
    # where S(alpha) = best fit function
    # and a(alpha) needs a fit too

    # Fit a(alpha) - the intercept function
    a_alphas = np.array([d[0] for d in S_data])
    a_values = np.array([d[2] for d in S_data])  # intercepts

    print("Fitting intercept a(alpha)...")

    # Try several forms for a(alpha)
    a_candidates = {}

    # Linear in alpha
    try:
        def a_linear(a, c0, c1):
            return c0 + c1 * a
        popt, _ = curve_fit(a_linear, a_alphas, a_values, p0=[0, 0])
        pred = a_linear(a_alphas, *popt)
        ss_res = np.sum((a_values - pred)**2)
        ss_tot = np.sum((a_values - np.mean(a_values))**2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        rmse = np.sqrt(np.mean((a_values - pred)**2))
        a_candidates['a_linear'] = {'popt': popt, 'R2': R2, 'rmse': rmse,
                                     'formula': f'a = {popt[0]:.4f} + {popt[1]:.4f}*alpha',
                                     'func': lambda a, p=popt: a_linear(a, *p)}
        print(f"  Linear:    a = {popt[0]:.4f} + {popt[1]:.4f}*alpha   R^2={R2:.6f} RMSE={rmse:.4f}")
    except Exception as e:
        print(f"  Linear: FAILED - {e}")

    # Quadratic
    try:
        def a_quad(a, c0, c1, c2):
            return c0 + c1 * a + c2 * a**2
        popt, _ = curve_fit(a_quad, a_alphas, a_values, p0=[0, 0, 0])
        pred = a_quad(a_alphas, *popt)
        ss_res = np.sum((a_values - pred)**2)
        ss_tot = np.sum((a_values - np.mean(a_values))**2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        rmse = np.sqrt(np.mean((a_values - pred)**2))
        a_candidates['a_quad'] = {'popt': popt, 'R2': R2, 'rmse': rmse,
                                   'formula': f'a = {popt[0]:.4f} + {popt[1]:.4f}*alpha + {popt[2]:.6f}*alpha^2',
                                   'func': lambda a, p=popt: a_quad(a, *p)}
        print(f"  Quadratic: a = {popt[0]:.4f} + {popt[1]:.4f}*alpha + {popt[2]:.6f}*alpha^2   R^2={R2:.6f} RMSE={rmse:.4f}")
    except Exception as e:
        print(f"  Quadratic: FAILED - {e}")

    # Log form
    try:
        def a_log(a, c0, c1):
            return c0 + c1 * np.log(a)
        popt, _ = curve_fit(a_log, a_alphas, a_values, p0=[0, 1])
        pred = a_log(a_alphas, *popt)
        ss_res = np.sum((a_values - pred)**2)
        ss_tot = np.sum((a_values - np.mean(a_values))**2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        rmse = np.sqrt(np.mean((a_values - pred)**2))
        a_candidates['a_log'] = {'popt': popt, 'R2': R2, 'rmse': rmse,
                                  'formula': f'a = {popt[0]:.4f} + {popt[1]:.4f}*ln(alpha)',
                                  'func': lambda a, p=popt: a_log(a, *p)}
        print(f"  Log:       a = {popt[0]:.4f} + {popt[1]:.4f}*ln(alpha)   R^2={R2:.6f} RMSE={rmse:.4f}")
    except Exception as e:
        print(f"  Log: FAILED - {e}")

    # Constant (the simplest: just the mean)
    a_mean = np.mean(a_values)
    ss_res = np.sum((a_values - a_mean)**2)
    ss_tot = np.sum((a_values - np.mean(a_values))**2)
    rmse_const = np.sqrt(np.mean((a_values - a_mean)**2))
    a_candidates['a_const'] = {'popt': [a_mean], 'R2': 0.0, 'rmse': rmse_const,
                                'formula': f'a = {a_mean:.4f}',
                                'func': lambda a, m=a_mean: m}
    print(f"  Constant:  a = {a_mean:.4f}   RMSE={rmse_const:.4f}")

    # Pick best a(alpha)
    best_a_name = min(a_candidates, key=lambda k: a_candidates[k]['rmse'])
    best_a = a_candidates[best_a_name]
    print(f"\n  Best intercept fit: {best_a_name} -> {best_a['formula']}")

    # ----------------------------------------------------------
    # Full validation
    # ----------------------------------------------------------
    print(f"\n{'='*76}")
    print("VALIDATION: D_formula vs D_CME")
    print(f"{'='*76}\n")

    S_func = best['func']
    a_func = best_a['func']

    print(f"Formula: ln(D) = a(alpha) + S(alpha) * Omega")
    print(f"  S(alpha) = {best['formula']}")
    print(f"  a(alpha) = {best_a['formula']}")
    print()

    print(f"{'alpha':>6} {'Omega':>5} {'D_CME':>14} {'D_formula':>14} {'ln(D_CME)':>10} {'ln(D_form)':>10} {'err_lnD':>8} {'ratio':>8}")
    print("-" * 82)

    max_err_ln = 0
    max_err_ratio = 0
    errors = []

    for alpha in alpha_values:
        if alpha not in results:
            continue
        for Omega in sorted(results[alpha].keys()):
            is_outlier = (alpha, Omega) in outlier_points
            D_cme = results[alpha][Omega]
            ln_D_cme = np.log(D_cme)

            S_pred = S_func(alpha)
            a_pred = a_func(alpha)
            ln_D_form = a_pred + S_pred * Omega
            D_form = np.exp(ln_D_form)

            err_ln = abs(ln_D_form - ln_D_cme)
            ratio = D_form / D_cme

            flag = " **OUTLIER**" if is_outlier else ""
            if not is_outlier:
                max_err_ln = max(max_err_ln, err_ln)
                max_err_ratio = max(max_err_ratio, abs(ratio - 1))
                errors.append(err_ln)

            print(f"{alpha:>6.1f} {Omega:>5} {D_cme:>14.4f} {D_form:>14.4f} {ln_D_cme:>10.4f} {ln_D_form:>10.4f} {err_ln:>8.4f} {ratio:>8.4f}{flag}")

    print(f"\nMax |ln(D_form) - ln(D_CME)| = {max_err_ln:.4f}  (excluding outliers)")
    print(f"Max |D_form/D_CME - 1|       = {max_err_ratio:.4f}")
    print(f"Mean |ln(D) error|            = {np.mean(errors):.4f}")
    print(f"Median |ln(D) error|          = {np.median(errors):.4f}")

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------
    print(f"\n{'='*76}")
    print("FINAL SHORTCUT FORMULA FOR THE TOGGLE SWITCH")
    print(f"{'='*76}\n")

    print("Model: du/dt = alpha/(1+v^2) - u,  dv/dt = alpha/(1+u^2) - v")
    print(f"Valid range: alpha in [{alphas_fit[0]:.1f}, {alphas_fit[-1]:.1f}], Omega >= 1")
    print()
    print("  D_toggle(alpha, Omega) = exp( a(alpha) + S(alpha) * Omega )")
    print()
    print(f"  where S(alpha) = {best['formula']}")
    print(f"        a(alpha) = {best_a['formula']}")
    print()
    print(f"  S(alpha) fit:  R^2 = {best['R2']:.8f},  RMSE = {best['rmse']:.6f}")
    print(f"  a(alpha) fit:  R^2 = {best_a['R2']:.6f},  RMSE = {best_a['rmse']:.4f}")
    print(f"  Overall max ln(D) error: {max_err_ln:.4f}")
    print(f"  Overall mean ln(D) error: {np.mean(errors):.4f}")
    print()

    # Print a quick-reference table
    print("Quick reference table: S(alpha)")
    print(f"{'alpha':>6} {'S_CME':>10} {'S_fit':>10} {'residual':>10}")
    print("-" * 38)
    for i, alpha in enumerate(alphas_fit):
        S_f = S_func(alpha)
        print(f"{alpha:>6.1f} {S_values[i]:>10.4f} {S_f:>10.4f} {S_values[i]-S_f:>+10.4f}")

    print()
    print("Quick reference table: a(alpha)")
    print(f"{'alpha':>6} {'a_CME':>10} {'a_fit':>10} {'residual':>10}")
    print("-" * 38)
    for i, (alpha, S, intercept, R2, rmse) in enumerate(S_data):
        a_f = a_func(alpha)
        print(f"{alpha:>6.1f} {intercept:>10.4f} {a_f:>10.4f} {intercept-a_f:>+10.4f}")

    # Kramers-Langer comparison
    print(f"\n{'='*76}")
    print("KRAMERS-LANGER PREFACTOR vs FITTED INTERCEPT")
    print(f"{'='*76}\n")

    print(f"{'alpha':>6} {'A_KL':>12} {'tau':>10} {'ln(2*A_KL*tau)':>16} {'a_fitted':>10} {'correction':>10}")
    print("-" * 66)
    for alpha, A, tau, A_tau, ln_A_tau, intercept in A_data:
        # The D = MFPT/tau = 2/(|lam2|*tau)
        # Kramers: |lam2| = 2 * A * exp(-S*Omega) [factor 2 for symmetric]
        # So D = 2/(2*A*exp(-S*Omega)*tau) = 1/(A*tau) * exp(S*Omega)
        # Thus intercept should be ~ -ln(A*tau)
        # Actually: D = MFPT/tau = (1/(A*exp(-S*Omega)))/tau = exp(S*Omega)/(A*tau)
        # So ln(D) = -ln(A*tau) + S*Omega
        neg_ln_A_tau = -np.log(A * tau) if A * tau > 0 else float('nan')
        correction = intercept - neg_ln_A_tau
        print(f"{alpha:>6.1f} {A:>12.6f} {tau:>10.4f} {neg_ln_A_tau:>16.4f} {intercept:>10.4f} {correction:>10.4f}")

    print(f"\n  If the Kramers-Langer prefactor were exact for this non-gradient system,")
    print(f"  the 'correction' column would be 0. Nonzero values indicate the")
    print(f"  non-gradient correction factor that differs from the standard formula.")

    # ----------------------------------------------------------
    # Step 6: Global 2D fit — bypass per-alpha fitting
    # ----------------------------------------------------------
    print(f"\n{'='*76}")
    print("STEP 6: GLOBAL 2D FIT — ln(D) = f(alpha, Omega) directly")
    print(f"{'='*76}\n")
    print("Instead of fitting S(alpha) and a(alpha) separately, fit all data")
    print("simultaneously to parametric forms.\n")

    # Gather all (alpha, Omega, ln_D) triples, excluding outliers
    all_points = []
    for alpha in alpha_values:
        if alpha not in results:
            continue
        for Omega in sorted(results[alpha].keys()):
            if (alpha, Omega) in outlier_points:
                continue
            all_points.append((alpha, Omega, np.log(results[alpha][Omega])))

    all_points = np.array(all_points)
    a_arr = all_points[:, 0]
    om_arr = all_points[:, 1]
    lnD_arr = all_points[:, 2]

    print(f"Total data points: {len(all_points)}")

    # Model: ln(D) = [p0 + p1*ln(alpha)] + [p2*(alpha-p3)^p4] * Omega
    # This is: a(alpha) = p0 + p1*ln(alpha),  S(alpha) = p2*(alpha-p3)^p4
    def global_model_A(X, p0, p1, p2, p3, p4):
        a, om = X
        S = p2 * (a - p3)**p4
        intercept = p0 + p1 * np.log(a)
        return intercept + S * om

    # Model B: ln(D) = [p0 + p1*ln(alpha)] + [p2*(a-1) + p3*(a-1)^2] * Omega
    def global_model_B(X, p0, p1, p2, p3):
        a, om = X
        S = p2 * (a - 1) + p3 * (a - 1)**2
        intercept = p0 + p1 * np.log(a)
        return intercept + S * om

    # Model C (simplest practical): ln(D) = [p0 + p1*ln(alpha)] + p2*(alpha - p3)^1.25 * Omega
    # Fix the exponent from power_law fit
    def global_model_C(X, p0, p1, p2, p3):
        a, om = X
        S = p2 * (a - p3)**1.25
        intercept = p0 + p1 * np.log(a)
        return intercept + S * om

    # Model D: pure power law with log prefactor
    # ln(D) = [p0 + p1*ln(alpha)] + p2 * (alpha - p3)^p4 * Omega
    # 5 params, most flexible

    global_fits = {}

    # Fit Model A (5 params)
    try:
        popt, _ = curve_fit(global_model_A, (a_arr, om_arr), lnD_arr,
                            p0=[-1.0, 1.3, 0.12, 1.1, 1.25],
                            bounds=([-10, 0, 0, 0.5, 0.5], [10, 5, 5, 2.5, 3.0]),
                            maxfev=10000)
        pred = global_model_A((a_arr, om_arr), *popt)
        ss_res = np.sum((lnD_arr - pred)**2)
        ss_tot = np.sum((lnD_arr - np.mean(lnD_arr))**2)
        R2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((lnD_arr - pred)**2))
        max_err = np.max(np.abs(lnD_arr - pred))
        global_fits['A_powerlaw_log'] = {
            'popt': popt, 'R2': R2, 'rmse': rmse, 'max_err': max_err,
            'formula': f'ln(D) = [{popt[0]:.4f} + {popt[1]:.4f}*ln(a)] + {popt[2]:.6f}*(a - {popt[3]:.4f})^{popt[4]:.4f} * Om',
            'func': lambda a, om, p=popt: global_model_A((a, om), *p)
        }
        print(f"Model A (5p): ln(D) = [{popt[0]:.4f} + {popt[1]:.4f}*ln(a)] + {popt[2]:.6f}*(a-{popt[3]:.4f})^{popt[4]:.4f}*Om")
        print(f"              R^2={R2:.8f}  RMSE={rmse:.4f}  max_err={max_err:.4f}")
    except Exception as e:
        print(f"Model A: FAILED - {e}")

    # Fit Model B (4 params, quadratic S)
    try:
        popt, _ = curve_fit(global_model_B, (a_arr, om_arr), lnD_arr,
                            p0=[-1.0, 1.3, 0.10, 0.02],
                            bounds=([-10, 0, 0, -0.1], [10, 5, 1, 0.1]),
                            maxfev=10000)
        pred = global_model_B((a_arr, om_arr), *popt)
        ss_res = np.sum((lnD_arr - pred)**2)
        ss_tot = np.sum((lnD_arr - np.mean(lnD_arr))**2)
        R2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((lnD_arr - pred)**2))
        max_err = np.max(np.abs(lnD_arr - pred))
        global_fits['B_quad_log'] = {
            'popt': popt, 'R2': R2, 'rmse': rmse, 'max_err': max_err,
            'formula': f'ln(D) = [{popt[0]:.4f} + {popt[1]:.4f}*ln(a)] + [{popt[2]:.6f}*(a-1) + {popt[3]:.6f}*(a-1)^2]*Om',
            'func': lambda a, om, p=popt: global_model_B((a, om), *p)
        }
        print(f"Model B (4p): ln(D) = [{popt[0]:.4f} + {popt[1]:.4f}*ln(a)] + [{popt[2]:.6f}*(a-1) + {popt[3]:.6f}*(a-1)^2]*Om")
        print(f"              R^2={R2:.8f}  RMSE={rmse:.4f}  max_err={max_err:.4f}")
    except Exception as e:
        print(f"Model B: FAILED - {e}")

    # Fit Model C (4 params, fixed exponent 1.25)
    try:
        popt, _ = curve_fit(global_model_C, (a_arr, om_arr), lnD_arr,
                            p0=[-1.0, 1.3, 0.12, 1.1],
                            bounds=([-10, 0, 0, 0.5], [10, 5, 5, 2.5]),
                            maxfev=10000)
        pred = global_model_C((a_arr, om_arr), *popt)
        ss_res = np.sum((lnD_arr - pred)**2)
        ss_tot = np.sum((lnD_arr - np.mean(lnD_arr))**2)
        R2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((lnD_arr - pred)**2))
        max_err = np.max(np.abs(lnD_arr - pred))
        global_fits['C_power125_log'] = {
            'popt': popt, 'R2': R2, 'rmse': rmse, 'max_err': max_err,
            'formula': f'ln(D) = [{popt[0]:.4f} + {popt[1]:.4f}*ln(a)] + {popt[2]:.6f}*(a - {popt[3]:.4f})^1.25 * Om',
            'func': lambda a, om, p=popt: global_model_C((a, om), *p)
        }
        print(f"Model C (4p): ln(D) = [{popt[0]:.4f} + {popt[1]:.4f}*ln(a)] + {popt[2]:.6f}*(a-{popt[3]:.4f})^1.25*Om")
        print(f"              R^2={R2:.8f}  RMSE={rmse:.4f}  max_err={max_err:.4f}")
    except Exception as e:
        print(f"Model C: FAILED - {e}")

    # Model D: constant intercept + power law S (3 params, simplest)
    def global_model_D(X, p0, p1, p2):
        a, om = X
        S = p1 * (a - 2.0)**1.25
        return p0 + S * om

    try:
        popt, _ = curve_fit(global_model_D, (a_arr, om_arr), lnD_arr,
                            p0=[1.5, 0.12, 0],
                            bounds=([-5, 0, -10], [10, 5, 10]),
                            maxfev=10000)
        pred = global_model_D((a_arr, om_arr), *popt)
        ss_res = np.sum((lnD_arr - pred)**2)
        ss_tot = np.sum((lnD_arr - np.mean(lnD_arr))**2)
        R2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((lnD_arr - pred)**2))
        max_err = np.max(np.abs(lnD_arr - pred))
        global_fits['D_simple'] = {
            'popt': popt, 'R2': R2, 'rmse': rmse, 'max_err': max_err,
            'formula': f'ln(D) = {popt[0]:.4f} + {popt[1]:.6f}*(a - 2)^1.25 * Om',
            'func': lambda a, om, p=popt: global_model_D((a, om), *p)
        }
        print(f"Model D (2p): ln(D) = {popt[0]:.4f} + {popt[1]:.6f}*(a-2)^1.25*Om")
        print(f"              R^2={R2:.8f}  RMSE={rmse:.4f}  max_err={max_err:.4f}")
    except Exception as e:
        print(f"Model D: FAILED - {e}")

    # Model E: log intercept + power law, free exponent, alpha_c free (most general)
    def global_model_E(X, p0, p1, p2, p3, p4, p5):
        a, om = X
        S = p2 * (a - p3)**p4
        intercept = p0 + p1 * np.log(a) + p5 * np.log(a)**2
        return intercept + S * om

    try:
        popt, _ = curve_fit(global_model_E, (a_arr, om_arr), lnD_arr,
                            p0=[-1.0, 1.3, 0.12, 1.1, 1.25, 0.0],
                            bounds=([-10, -5, 0, 0.5, 0.5, -2], [10, 5, 5, 2.5, 3.0, 2]),
                            maxfev=20000)
        pred = global_model_E((a_arr, om_arr), *popt)
        ss_res = np.sum((lnD_arr - pred)**2)
        ss_tot = np.sum((lnD_arr - np.mean(lnD_arr))**2)
        R2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((lnD_arr - pred)**2))
        max_err = np.max(np.abs(lnD_arr - pred))
        global_fits['E_full'] = {
            'popt': popt, 'R2': R2, 'rmse': rmse, 'max_err': max_err,
            'formula': f'ln(D) = [{popt[0]:.4f} + {popt[1]:.4f}*ln(a) + {popt[5]:.4f}*ln(a)^2] + {popt[2]:.6f}*(a-{popt[3]:.4f})^{popt[4]:.4f}*Om',
            'func': lambda a, om, p=popt: global_model_E((a, om), *p)
        }
        print(f"Model E (6p): ln(D) = [{popt[0]:.4f}+{popt[1]:.4f}*ln(a)+{popt[5]:.4f}*ln(a)^2] + {popt[2]:.6f}*(a-{popt[3]:.4f})^{popt[4]:.4f}*Om")
        print(f"              R^2={R2:.8f}  RMSE={rmse:.4f}  max_err={max_err:.4f}")
    except Exception as e:
        print(f"Model E: FAILED - {e}")

    # Rank global fits
    print(f"\n--- Global fit ranking ---")
    ranked_g = sorted(global_fits.items(), key=lambda x: x[1]['rmse'])
    print(f"{'Rank':>4} {'Name':>18} {'R^2':>12} {'RMSE':>10} {'max_err':>10}")
    print("-" * 60)
    for i, (name, f) in enumerate(ranked_g):
        print(f"{i+1:>4} {name:>18} {f['R2']:>12.8f} {f['rmse']:>10.4f} {f['max_err']:>10.4f}")

    best_g_name, best_g = ranked_g[0]
    print(f"\nBest global fit: {best_g_name}")
    print(f"  {best_g['formula']}")

    # Detailed validation of best global fit
    print(f"\n--- Detailed validation of best global fit ---")
    print(f"{'alpha':>6} {'Omega':>5} {'D_CME':>14} {'D_formula':>14} {'ln(D_CME)':>10} {'ln(D_form)':>10} {'err_lnD':>8} {'D_form/D_CME':>12}")
    print("-" * 86)

    max_err_g = 0
    errors_g = []
    best_g_func = best_g['func']

    for alpha in alpha_values:
        if alpha not in results:
            continue
        for Omega in sorted(results[alpha].keys()):
            is_outlier = (alpha, Omega) in outlier_points
            D_cme = results[alpha][Omega]
            ln_D_cme = np.log(D_cme)
            ln_D_form = best_g_func(alpha, Omega)
            D_form = np.exp(ln_D_form)
            err_ln = abs(ln_D_form - ln_D_cme)
            ratio = D_form / D_cme
            flag = " **OUTLIER**" if is_outlier else ""
            if not is_outlier:
                max_err_g = max(max_err_g, err_ln)
                errors_g.append(err_ln)
            print(f"{alpha:>6.1f} {Omega:>5} {D_cme:>14.4f} {D_form:>14.4f} {ln_D_cme:>10.4f} {ln_D_form:>10.4f} {err_ln:>8.4f} {ratio:>12.4f}{flag}")

    print(f"\nBest global fit validation (excluding outliers):")
    print(f"  Max |ln(D) error|  = {max_err_g:.4f}")
    print(f"  Mean |ln(D) error| = {np.mean(errors_g):.4f}")
    print(f"  Max D ratio error  = {np.max([abs(np.exp(e)-1) for e in errors_g]):.4f}")

    # ----------------------------------------------------------
    # FINAL CONSOLIDATED FORMULA
    # ----------------------------------------------------------
    print(f"\n{'='*76}")
    print("CONSOLIDATED SHORTCUT FORMULA")
    print(f"{'='*76}\n")

    print("TOGGLE SWITCH: du/dt = alpha/(1+v^2) - u,  dv/dt = alpha/(1+u^2) - v")
    print(f"Valid range: alpha in [3, 12], Omega in [2, 5+]")
    print()

    # Present the best global fit
    print(f"BEST GLOBAL FIT ({best_g_name}):")
    print(f"  {best_g['formula']}")
    print(f"  R^2 = {best_g['R2']:.8f}")
    print(f"  RMSE(ln D) = {best_g['rmse']:.4f}")
    print(f"  max |ln D error| = {best_g['max_err']:.4f}")
    print()

    # Also present the separate S(alpha) fit for conceptual clarity
    print(f"CONCEPTUAL DECOMPOSITION:")
    print(f"  D = exp(a(alpha)) * exp(S(alpha) * Omega)")
    print(f"  S(alpha) = {best['formula']}")
    print(f"  S(alpha) R^2 = {best['R2']:.8f}")
    print()

    # Present a clean summary of S values
    print("S(alpha) reference values (Kramers action per unit volume):")
    print(f"{'alpha':>6} {'S':>8}")
    print("-" * 16)
    for i, alpha in enumerate(alphas_fit):
        print(f"{alpha:>6.1f} {S_values[i]:>8.4f}")


if __name__ == "__main__":
    main()
