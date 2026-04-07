#!/usr/bin/env python3
"""
Polynomial Bridge Scaling: P(stable limit cycle | d) for 2D polynomial ODEs.
Tests whether gamma ~ 0.20 (exponential decay of P(organized | d)) holds
for a structurally different model class (pure polynomials vs Hill functions).

Model:
  dx/dt = sum_{i+j<=p} a_{ij} * x^i * y^j  -  eps * x * (x^2 + y^2)
  dy/dt = sum_{i+j<=p} b_{ij} * x^i * y^j  -  eps * y * (x^2 + y^2)

Parameters: d = (p+1)(p+2) total random coefficients (a_{ij} and b_{ij}).
The cubic damping term (-eps * z * |z|^2, eps=0.01) is fixed, not random.

Detection: Vectorized Newton -> Jacobian eigenvalues -> ODE integration
Result: P(limit cycle | d), fit gamma_poly from ln(1/P) ~ gamma*d

Study 18: First computation testing model-independence of the bridge.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from scipy.stats import linregress
import time
import sys
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


def pprint(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs)
    sys.stdout.flush()


# ============================================================
# CONSTANTS
# ============================================================

# Cubic damping coefficient (fixed, not a random parameter)
EPS_DAMP = 0.01

# Newton's method: 4x4 grid of initial conditions in [-3, 3]^2
# Polynomials can have equilibria anywhere (not just positive quadrant)
INIT_POINTS = [(x, y) for x in [-2.0, -0.5, 0.5, 2.0]
                       for y in [-2.0, -0.5, 0.5, 2.0]]
NEWTON_ITERS = 20
NEWTON_MAX_STEP = 2.0
EQ_TOL = 1e-6
EQ_MAX_RADIUS = 10.0  # Reject equilibria beyond this radius

# ODE integration
T_INTEGRATE = 300.0
T_DENSITY = 5.0
DIVERGE_THRESHOLD = 100.0

# Periodicity detection (last 50% of trajectory)
PERIODIC_CV = 0.20
MIN_PEAKS = 5
MIN_AMPLITUDE = 0.01

# Batch processing
BATCH_SIZE = 10_000

# Coefficient ranges by monomial degree k = i+j
# Scaled so total contribution from each degree is O(1) at |x|,|y| ~ 1
COEFF_RANGES = {
    0: 1.0,
    1: 1.5,
    2: 0.5,
    3: 0.3,
    4: 0.2,
    5: 0.15,
    6: 0.1,
}

# d specifications: (degree_p, d, N_samples)
# P(lc) ~ 1-2% for polynomials (much higher than Hill model), so smaller N suffices.
# 100k-500k gives 1000-5000+ limit cycles per d value = excellent statistics.
D_SPECS = [
    (2, 12,    100_000),
    (3, 20,    100_000),
    (4, 30,    200_000),
    (5, 42,    500_000),
    (6, 56,  1_000_000),
]

GAMMA_CUSP = 0.197
GAMMA_HOPF = 0.175


# ============================================================
# TERM INDEX BUILDING
# ============================================================

def build_terms(p):
    """
    Build ordered list of (i, j) for all monomials x^i * y^j with i+j <= p.
    Returns: terms list, M = len(terms) = (p+1)(p+2)/2
    """
    terms = []
    for total in range(p + 1):
        for i in range(total + 1):
            j = total - i
            terms.append((i, j))
    M = len(terms)
    assert M == (p + 1) * (p + 2) // 2, f"Expected {(p+1)*(p+2)//2} terms, got {M}"
    return terms, M


# ============================================================
# PARAMETER GENERATION
# ============================================================

def generate_poly_coefficients(p, N):
    """
    Generate N random polynomial coefficient sets for degree p.

    Returns:
        a_coeffs: (N, M) array for x-equation
        b_coeffs: (N, M) array for y-equation
        terms: list of (i, j) tuples
    """
    terms, M = build_terms(p)
    a_coeffs = np.empty((N, M), dtype=np.float64)
    b_coeffs = np.empty((N, M), dtype=np.float64)

    for idx, (i, j) in enumerate(terms):
        k = i + j  # degree of this monomial
        r = COEFF_RANGES[k]
        a_coeffs[:, idx] = np.random.uniform(-r, r, N)
        b_coeffs[:, idx] = np.random.uniform(-r, r, N)

    return a_coeffs, b_coeffs, terms


# ============================================================
# VECTORIZED POLYNOMIAL EVALUATION
# ============================================================

def eval_poly_batch(a_coeffs, terms, x, y, p):
    """
    Vectorized evaluation of polynomial sum_{(i,j)} a_{ij} * x^i * y^j
    for all N samples simultaneously.

    Uses precomputed power arrays (repeated multiplication, no pow()).

    Args:
        a_coeffs: (N, M) coefficients
        terms: list of (i, j) tuples
        x, y: (N,) arrays
        p: max degree

    Returns: (N,) array of polynomial values
    """
    N = len(x)

    # Precompute x^0, x^1, ..., x^p via repeated multiplication
    x_pow = [np.ones(N, dtype=np.float64)]
    y_pow = [np.ones(N, dtype=np.float64)]
    for k in range(1, p + 1):
        x_pow.append(x_pow[-1] * x)
        y_pow.append(y_pow[-1] * y)

    # Evaluate: F = sum_m a[m] * x^i_m * y^j_m
    result = np.zeros(N, dtype=np.float64)
    for m, (i, j) in enumerate(terms):
        result += a_coeffs[:, m] * x_pow[i] * y_pow[j]

    return result


def eval_poly_jacobian_batch(a_coeffs, b_coeffs, terms, x, y, p):
    """
    Vectorized Jacobian of the polynomial system (WITHOUT damping).

    dF/dx = sum a_{ij} * i * x^(i-1) * y^j   (for i >= 1)
    dF/dy = sum a_{ij} * j * x^i * y^(j-1)    (for j >= 1)
    dG/dx = sum b_{ij} * i * x^(i-1) * y^j    (for i >= 1)
    dG/dy = sum b_{ij} * j * x^i * y^(j-1)    (for j >= 1)

    Returns: dFdx, dFdy, dGdx, dGdy (all shape (N,))
    """
    N = len(x)

    x_pow = [np.ones(N, dtype=np.float64)]
    y_pow = [np.ones(N, dtype=np.float64)]
    for k in range(1, p + 1):
        x_pow.append(x_pow[-1] * x)
        y_pow.append(y_pow[-1] * y)

    dFdx = np.zeros(N, dtype=np.float64)
    dFdy = np.zeros(N, dtype=np.float64)
    dGdx = np.zeros(N, dtype=np.float64)
    dGdy = np.zeros(N, dtype=np.float64)

    for m, (i, j) in enumerate(terms):
        if i >= 1:
            # d/dx of x^i * y^j = i * x^(i-1) * y^j
            term = i * x_pow[i - 1] * y_pow[j]
            dFdx += a_coeffs[:, m] * term
            dGdx += b_coeffs[:, m] * term
        if j >= 1:
            # d/dy of x^i * y^j = j * x^i * y^(j-1)
            term = j * x_pow[i] * y_pow[j - 1]
            dFdy += a_coeffs[:, m] * term
            dGdy += b_coeffs[:, m] * term

    return dFdx, dFdy, dGdx, dGdy


def add_damping_to_jacobian(dFdx, dFdy, dGdx, dGdy, x, y):
    """
    Add cubic damping contribution to Jacobian.

    Damping term for x-equation: -eps * x * (x^2 + y^2)
    d/dx[-eps * x * (x^2 + y^2)] = -eps * (3*x^2 + y^2)
    d/dy[-eps * x * (x^2 + y^2)] = -eps * 2*x*y

    Damping term for y-equation: -eps * y * (x^2 + y^2)
    d/dx[-eps * y * (x^2 + y^2)] = -eps * 2*x*y
    d/dy[-eps * y * (x^2 + y^2)] = -eps * (x^2 + 3*y^2)
    """
    x2 = x * x
    y2 = y * y
    xy = x * y

    dFdx -= EPS_DAMP * (3.0 * x2 + y2)
    dFdy -= EPS_DAMP * 2.0 * xy
    dGdx -= EPS_DAMP * 2.0 * xy
    dGdy -= EPS_DAMP * (x2 + 3.0 * y2)

    return dFdx, dFdy, dGdx, dGdy


# ============================================================
# VECTORIZED NEWTON'S METHOD FOR EQUILIBRIUM FINDING
# ============================================================

def vectorized_newton(a_coeffs, b_coeffs, terms, p, x0_val, y0_val):
    """
    Vectorized Newton's method: find equilibria for all N samples
    starting from (x0_val, y0_val).

    The full system is:
      F = poly_a(x, y) - eps * x * (x^2 + y^2)
      G = poly_b(x, y) - eps * y * (x^2 + y^2)

    Returns x_eq, y_eq, converged (all shape (N,)).
    """
    N = a_coeffs.shape[0]
    x = np.full(N, x0_val, dtype=np.float64)
    y = np.full(N, y0_val, dtype=np.float64)

    for _ in range(NEWTON_ITERS):
        # Evaluate F, G (polynomial + damping)
        F = eval_poly_batch(a_coeffs, terms, x, y, p)
        G = eval_poly_batch(b_coeffs, terms, x, y, p)
        r2 = x * x + y * y
        F -= EPS_DAMP * x * r2
        G -= EPS_DAMP * y * r2

        # Evaluate Jacobian (polynomial + damping)
        dFdx, dFdy, dGdx, dGdy = eval_poly_jacobian_batch(
            a_coeffs, b_coeffs, terms, x, y, p)
        dFdx, dFdy, dGdx, dGdy = add_damping_to_jacobian(
            dFdx, dFdy, dGdx, dGdy, x, y)

        # Solve 2x2: J * [dx, dy]^T = -[F, G]^T
        det = dFdx * dGdy - dFdy * dGdx
        safe_det = np.where(np.abs(det) > 1e-15, det, 1e-15)
        dx = (-F * dGdy + G * dFdy) / safe_det
        dy = (F * dGdx - G * dFdx) / safe_det

        # Damped step
        dx = np.clip(dx, -NEWTON_MAX_STEP, NEWTON_MAX_STEP)
        dy = np.clip(dy, -NEWTON_MAX_STEP, NEWTON_MAX_STEP)

        x = x + dx
        y = y + dy

    # Final residual check
    F = eval_poly_batch(a_coeffs, terms, x, y, p)
    G = eval_poly_batch(b_coeffs, terms, x, y, p)
    r2 = x * x + y * y
    F -= EPS_DAMP * x * r2
    G -= EPS_DAMP * y * r2

    residual = np.abs(F) + np.abs(G)
    radius = np.sqrt(x * x + y * y)
    converged = (residual < EQ_TOL) & (radius < EQ_MAX_RADIUS)

    return x, y, converged


# ============================================================
# VECTORIZED UNSTABLE SPIRAL CHECK
# ============================================================

def vectorized_check_spiral(a_coeffs, b_coeffs, terms, p,
                            x_eq, y_eq, converged):
    """
    Check for unstable spiral at converged equilibria.
    Criterion: tr(J) > 0 AND discriminant < 0 (complex eigenvalues).

    Returns is_spiral mask (shape (N,)).
    """
    dFdx, dFdy, dGdx, dGdy = eval_poly_jacobian_batch(
        a_coeffs, b_coeffs, terms, x_eq, y_eq, p)
    dFdx, dFdy, dGdx, dGdy = add_damping_to_jacobian(
        dFdx, dFdy, dGdx, dGdy, x_eq, y_eq)

    tr = dFdx + dGdy
    det = dFdx * dGdy - dFdy * dGdx
    disc = tr * tr - 4.0 * det

    # Unstable spiral: trace > 0 AND discriminant < 0
    return converged & (tr > 0) & (disc < 0)


# ============================================================
# VECTORIZED BATCH RK4 INTEGRATION + LIMIT CYCLE DETECTION
# ============================================================

def batch_rhs(a_batch, b_batch, terms, p, x, y):
    """
    Evaluate RHS for K systems simultaneously.
    a_batch, b_batch: (K, M) coefficient arrays
    x, y: (K,) state arrays
    Returns F, G: (K,) arrays
    """
    K = len(x)
    x_pow = [np.ones(K, dtype=np.float64)]
    y_pow = [np.ones(K, dtype=np.float64)]
    for k in range(1, p + 1):
        x_pow.append(x_pow[-1] * x)
        y_pow.append(y_pow[-1] * y)

    F = np.zeros(K, dtype=np.float64)
    G = np.zeros(K, dtype=np.float64)
    for m, (i, j) in enumerate(terms):
        mono = x_pow[i] * y_pow[j]
        F += a_batch[:, m] * mono
        G += b_batch[:, m] * mono

    # Cubic damping
    r2 = x * x + y * y
    F -= EPS_DAMP * x * r2
    G -= EPS_DAMP * y * r2

    return F, G


def vectorized_rk4_integrate(a_batch, b_batch, terms, p, x0, y0,
                              dt=0.1, n_steps=3000, save_every=2):
    """
    Vectorized fixed-step RK4 integration of K polynomial systems
    simultaneously. Saves trajectory every save_every steps.

    Returns:
        x_traj: (K, n_saved) trajectory of x-coordinate
        alive: (K,) boolean mask of non-divergent trajectories
    """
    K = len(x0)
    x = x0.copy()
    y = y0.copy()
    alive = np.ones(K, dtype=bool)

    n_saved = n_steps // save_every
    x_traj = np.empty((K, n_saved), dtype=np.float64)
    save_idx = 0

    for step in range(n_steps):
        # RK4 for alive trajectories only (but compute on all for simplicity,
        # mask divergent ones)
        k1x, k1y = batch_rhs(a_batch, terms, p, x, y) if False else (None, None)

        # Full RK4 step on all K systems
        k1x, k1y = batch_rhs(a_batch, b_batch, terms, p, x, y)
        k2x, k2y = batch_rhs(a_batch, b_batch, terms, p,
                              x + 0.5 * dt * k1x, y + 0.5 * dt * k1y)
        k3x, k3y = batch_rhs(a_batch, b_batch, terms, p,
                              x + 0.5 * dt * k2x, y + 0.5 * dt * k2y)
        k4x, k4y = batch_rhs(a_batch, b_batch, terms, p,
                              x + dt * k3x, y + dt * k3y)

        x_new = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        y_new = y + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)

        # Check divergence
        diverged = (np.abs(x_new) > DIVERGE_THRESHOLD) | \
                   (np.abs(y_new) > DIVERGE_THRESHOLD) | \
                   np.isnan(x_new) | np.isnan(y_new)
        alive &= ~diverged

        # Clamp diverged trajectories to prevent NaN propagation
        x = np.where(alive, x_new, x)
        y = np.where(alive, y_new, y)

        # Save trajectory
        if step % save_every == 0 and save_idx < n_saved:
            x_traj[:, save_idx] = x
            save_idx += 1

    # Trim if we saved fewer than expected
    x_traj = x_traj[:, :save_idx]

    return x_traj, alive


def check_periodicity_batch(x_traj, alive, dt_save):
    """
    Check periodicity in the last 50% of each trajectory.
    Returns boolean array of which trajectories are periodic limit cycles.

    x_traj: (K, n_saved) array
    alive: (K,) boolean mask
    dt_save: time between saved points
    """
    K, n_saved = x_traj.shape
    is_lc = np.zeros(K, dtype=bool)

    # Only check alive trajectories
    alive_idx = np.where(alive)[0]

    for k in alive_idx:
        x_tr = x_traj[k, n_saved // 2:]
        n = len(x_tr)
        if n < 30:
            continue

        # Peak detection
        peaks = np.where(
            (x_tr[1:-1] > x_tr[:-2]) & (x_tr[1:-1] > x_tr[2:])
        )[0] + 1

        if len(peaks) < MIN_PEAKS:
            continue

        heights = x_tr[peaks]
        times = peaks * dt_save  # relative time

        # Amplitude consistency
        mean_h = np.mean(heights)
        if mean_h == 0:
            continue
        if np.std(heights) / abs(mean_h) > PERIODIC_CV:
            continue

        # Period consistency
        periods = np.diff(times)
        if len(periods) < 2:
            continue
        mean_per = np.mean(periods)
        if mean_per == 0:
            continue
        if np.std(periods) / abs(mean_per) > PERIODIC_CV:
            continue

        # Non-trivial amplitude
        if np.max(x_tr) - np.min(x_tr) < MIN_AMPLITUDE:
            continue

        is_lc[k] = True

    return is_lc


# ============================================================
# MAIN BATCH PROCESSING
# ============================================================

def check_batch(p, N):
    """
    Screen N random polynomial systems of degree p for stable limit cycles.

    Pipeline per batch:
    1. Generate random polynomial coefficients
    2. Vectorized Newton from 16 initial conditions -> find equilibria
    3. Vectorized Jacobian check -> identify unstable spirals
    4. VECTORIZED batch RK4 integration -> verify limit cycles in parallel

    Returns (n_lc, n_spiral).
    """
    terms, M = build_terms(p)
    d = (p + 1) * (p + 2)

    # RK4 parameters: dt=0.1, T=300, save every 2 steps
    RK4_DT = 0.1
    RK4_STEPS = int(T_INTEGRATE / RK4_DT)
    RK4_SAVE_EVERY = 2
    DT_SAVE = RK4_DT * RK4_SAVE_EVERY

    n_lc = 0
    n_spiral = 0
    n_done = 0
    batch_num = 0

    while n_done < N:
        bs = min(BATCH_SIZE, N - n_done)
        batch_num += 1

        a_coeffs, b_coeffs, _ = generate_poly_coefficients(p, bs)

        # Collect spirals across all initial conditions
        spiral_eqs = {}  # local_idx -> (x_eq, y_eq)

        for x0, y0 in INIT_POINTS:
            x_eq, y_eq, conv = vectorized_newton(
                a_coeffs, b_coeffs, terms, p, x0, y0)
            is_spiral = vectorized_check_spiral(
                a_coeffs, b_coeffs, terms, p, x_eq, y_eq, conv)

            for idx in np.where(is_spiral)[0]:
                idx_int = int(idx)
                if idx_int not in spiral_eqs:
                    spiral_eqs[idx_int] = (x_eq[idx], y_eq[idx])

        n_spiral += len(spiral_eqs)

        # Vectorized ODE integration of all spirals simultaneously
        if len(spiral_eqs) > 0:
            sp_indices = list(spiral_eqs.keys())
            sp_x0 = np.array([spiral_eqs[i][0] + 0.01 for i in sp_indices])
            sp_y0 = np.array([spiral_eqs[i][1] + 0.01 for i in sp_indices])
            sp_a = a_coeffs[sp_indices]
            sp_b = b_coeffs[sp_indices]

            x_traj, alive = vectorized_rk4_integrate(
                sp_a, sp_b, terms, p, sp_x0, sp_y0,
                dt=RK4_DT, n_steps=RK4_STEPS, save_every=RK4_SAVE_EVERY)

            is_lc = check_periodicity_batch(x_traj, alive, DT_SAVE)
            n_lc += int(np.sum(is_lc))

        n_done += bs
        report_every = max(1, N // (BATCH_SIZE * 5))
        if batch_num % report_every == 0 or n_done >= N:
            pprint(f"    {n_done:>10,}/{N:,} | "
                   f"spiral {n_spiral:,} | lc {n_lc}")

    return n_lc, n_spiral


# ============================================================
# FITTING MODELS (matches cusp/Hopf bridge structure)
# ============================================================

def fit_models(d_arr, lnP_inv):
    """Fit ln(1/P) vs d: linear, quadratic, stretched-exp."""
    results = {}
    if len(d_arr) < 2:
        return results

    # Model A: linear
    slope, intercept, r_val, _, _ = linregress(d_arr, lnP_inv)
    results['A'] = {'gamma': slope, 'c0': intercept, 'R2': r_val**2}

    # Model B: quadratic
    if len(d_arr) >= 3:
        try:
            def mb(d, c0, g1, g2):
                return c0 + g1 * d + g2 * d**2
            po, _ = curve_fit(mb, d_arr, lnP_inv, p0=[0, 0.1, 0.001])
            yp = mb(d_arr, *po)
            ss_r = np.sum((lnP_inv - yp)**2)
            ss_t = np.sum((lnP_inv - np.mean(lnP_inv))**2)
            results['B'] = {
                'g1': po[1], 'g2': po[2], 'c0': po[0],
                'R2': 1 - ss_r / ss_t if ss_t > 0 else 0
            }
        except Exception:
            pass

    # Model C: stretched exponential
    if len(d_arr) >= 3:
        try:
            def mc(d, c0, gamma, beta):
                return c0 + gamma * d**beta
            po, _ = curve_fit(mc, d_arr, lnP_inv, p0=[0, 0.1, 1.0],
                              bounds=([-np.inf, 0, 0.1], [np.inf, np.inf, 3.0]),
                              maxfev=10000)
            yp = mc(d_arr, *po)
            ss_r = np.sum((lnP_inv - yp)**2)
            ss_t = np.sum((lnP_inv - np.mean(lnP_inv))**2)
            results['C'] = {
                'gamma': po[1], 'beta': po[2], 'c0': po[0],
                'R2': 1 - ss_r / ss_t if ss_t > 0 else 0
            }
        except Exception:
            pass

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    pprint("=" * 70)
    pprint("  POLYNOMIAL BRIDGE SCALING: P(stable limit cycle | d)")
    pprint("  2D polynomial ODE with cubic damping")
    pprint("=" * 70)
    pprint()
    pprint("Model:")
    pprint("  dx/dt = sum_{i+j<=p} a_{ij} * x^i * y^j - eps*x*(x^2+y^2)")
    pprint("  dy/dt = sum_{i+j<=p} b_{ij} * x^i * y^j - eps*y*(x^2+y^2)")
    pprint(f"  eps = {EPS_DAMP} (fixed cubic damping)")
    pprint(f"  d = (p+1)(p+2) total random parameters")
    pprint()
    pprint(f"Equilibria: vectorized Newton, {len(INIT_POINTS)} initial conditions, "
           f"{NEWTON_ITERS} iterations")
    pprint(f"ODE: t_max={T_INTEGRATE:.0f}, RK45, diverge at |z|>{DIVERGE_THRESHOLD}")
    pprint(f"Periodic: CV<{PERIODIC_CV}, min {MIN_PEAKS} peaks, amp>{MIN_AMPLITUDE}")
    pprint()

    # Verify d formula
    pprint("Degree-dimension table:")
    for p_deg, d_exp, N in D_SPECS:
        terms, M = build_terms(p_deg)
        d_actual = 2 * M
        assert d_actual == d_exp, f"p={p_deg}: expected d={d_exp}, got {d_actual}"
        pprint(f"  p={p_deg}: {M} terms/eq, d={d_actual}, N={N:,}")
    pprint()

    # ================================================================
    #  PHASE 1: P(limit cycle | d) measurement
    # ================================================================
    pprint("=" * 70)
    pprint("  PHASE 1: P(limit cycle | d)")
    pprint("=" * 70)
    pprint()

    d_vals, P_vals, N_vals = [], [], []
    n_lc_vals, n_sp_vals = [], []
    consecutive_zeros = 0

    for p_deg, d, N in D_SPECS:
        pprint(f"--- d = {d} (degree {p_deg} polynomial), N = {N:,} ---")
        t0 = time.time()

        n_lc, n_sp = check_batch(p_deg, N)
        elapsed = time.time() - t0

        P = n_lc / N if N > 0 else 0
        lnP = np.log(1.0 / P) if P > 0 else np.inf

        flag = ""
        if n_lc == 0:
            flag = f" [ZERO -- P < {1.0/N:.1e}]"
            consecutive_zeros += 1
        elif n_lc < 10:
            flag = f" [UNCERTAIN -- only {n_lc}]"
            consecutive_zeros = 0
        else:
            consecutive_zeros = 0

        pprint(f"  N_lc={n_lc:,}  N_spiral={n_sp:,}")
        pprint(f"  P={P:.6e}  ln(1/P)={lnP:.3f}  time={elapsed:.1f}s{flag}")
        pprint()

        d_vals.append(d)
        P_vals.append(P)
        N_vals.append(N)
        n_lc_vals.append(n_lc)
        n_sp_vals.append(n_sp)

        if consecutive_zeros >= 2:
            pprint("STOP: Two consecutive d values with zero limit cycles.\n")
            break

    # ================================================================
    #  SUMMARY TABLE
    # ================================================================
    pprint("=" * 70)
    pprint("  SUMMARY TABLE")
    pprint("=" * 70)
    pprint()
    hdr = (f"{'d':>4} | {'p':>2} | {'N':>10} | {'N_spiral':>8} | "
           f"{'N_lc':>6} | {'P(lc)':>12} | {'ln(1/P)':>8}")
    pprint(hdr)
    pprint("-" * len(hdr))
    for i in range(len(d_vals)):
        lnP = np.log(1.0 / P_vals[i]) if P_vals[i] > 0 else np.inf
        p_deg = D_SPECS[i][0] if i < len(D_SPECS) else '?'
        pprint(f"{d_vals[i]:4d} | {p_deg:2d} | {N_vals[i]:10,} | {n_sp_vals[i]:8,} | "
               f"{n_lc_vals[i]:6,} | "
               f"{P_vals[i]:12.6e} | {lnP:8.3f}")
    pprint()

    # ================================================================
    #  PHASE 2: Fit gamma_poly, compare to gamma_cusp and gamma_Hopf
    # ================================================================
    pprint("=" * 70)
    pprint("  PHASE 2: Fit gamma_poly")
    pprint("=" * 70)
    pprint()

    valid = [(d, P) for d, P in zip(d_vals, P_vals) if P > 0]
    if len(valid) < 2:
        pprint(f"INSUFFICIENT DATA: {len(valid)} point(s) with P > 0.")
        pprint("Cannot fit. Adjust parameter ranges or increase N.")
        pprint()
        pprint("=" * 70)
        return

    d_fit = np.array([v[0] for v in valid])
    P_fit = np.array([v[1] for v in valid])
    lnP_fit = np.log(1.0 / P_fit)

    # Determine fitting range: if non-monotonic, fit only decay regime
    # (past the peak of P, where ln(1/P) is increasing)
    if len(lnP_fit) >= 3:
        # Find where ln(1/P) starts consistently increasing
        min_idx = np.argmin(lnP_fit)
        if min_idx > 0 and min_idx < len(lnP_fit) - 1:
            # Peak of P is at min of ln(1/P)
            pprint(f"P(lc) peaks near d={d_fit[min_idx]:.0f}. "
                   f"Fitting decay regime: d >= {d_fit[min_idx]:.0f}")
            d_decay = d_fit[min_idx:]
            lnP_decay = lnP_fit[min_idx:]
        else:
            d_decay = d_fit
            lnP_decay = lnP_fit
    else:
        d_decay = d_fit
        lnP_decay = lnP_fit

    pprint(f"Fitting {len(d_decay)} points, d = {d_decay.min():.0f}-{d_decay.max():.0f}")
    pprint(f"Also fitting ALL {len(d_fit)} points for comparison.")
    pprint()

    # Fit on decay regime
    res_decay = fit_models(d_decay, lnP_decay)
    # Fit on all valid points
    res_all = fit_models(d_fit, lnP_fit)

    for label, res, d_used, lnP_used in [
        ("DECAY REGIME", res_decay, d_decay, lnP_decay),
        ("ALL VALID POINTS", res_all, d_fit, lnP_fit),
    ]:
        pprint(f"--- {label} (n={len(d_used)}) ---")
        pprint()

        # Model A: linear
        if 'A' in res:
            r = res['A']
            pprint(f"Model A (linear): ln(1/P) = {r['c0']:.4f} + {r['gamma']:.6f} * d")
            pprint(f"  gamma_poly = {r['gamma']:.6f}")
            pprint(f"  R^2        = {r['R2']:.6f}")
            ratio_cusp = r['gamma'] / GAMMA_CUSP
            ratio_hopf = r['gamma'] / GAMMA_HOPF
            diff_cusp = abs(ratio_cusp - 1) * 100
            diff_hopf = abs(ratio_hopf - 1) * 100
            pprint(f"  gamma_cusp = {GAMMA_CUSP:.6f}  ratio = {ratio_cusp:.3f} "
                   f"({diff_cusp:.1f}% diff)")
            pprint(f"  gamma_Hopf = {GAMMA_HOPF:.6f}  ratio = {ratio_hopf:.3f} "
                   f"({diff_hopf:.1f}% diff)")
            if diff_cusp < 20:
                pprint(f"  --> CONSISTENT with gamma_cusp (within 20%)")
            elif diff_cusp < 30:
                pprint(f"  --> MARGINAL vs gamma_cusp (within 30%)")
            elif diff_cusp < 50:
                pprint(f"  --> DIFFERENT from gamma_cusp (>30% but <50%)")
            else:
                pprint(f"  --> STRONGLY DIFFERENT from gamma_cusp (>50%)")
            pprint()

        # Model B: quadratic
        if 'B' in res:
            r = res['B']
            pprint(f"Model B (quadratic): ln(1/P) = {r['c0']:.4f} + "
                   f"{r['g1']:.6f}*d + {r['g2']:.8f}*d^2")
            pprint(f"  R^2 = {r['R2']:.6f}")
            pprint(f"  g2 {'>' if r['g2'] > 0 else '<'} 0: decay "
                   f"{'ACCELERATES' if r['g2'] > 0 else 'DECELERATES'}")
            pprint()

        # Model C: stretched exponential
        if 'C' in res:
            r = res['C']
            pprint(f"Model C (stretched exp): ln(1/P) = {r['c0']:.4f} + "
                   f"{r['gamma']:.6f} * d^{r['beta']:.4f}")
            pprint(f"  R^2 = {r['R2']:.6f}")
            if r['beta'] > 1.05:
                pprint(f"  beta > 1: superlinear (decay accelerates)")
            elif r['beta'] < 0.95:
                pprint(f"  beta < 1: sublinear (decay decelerates)")
            else:
                pprint(f"  beta ~ 1: consistent with linear")
            pprint()

        # Linearity check
        if 'A' in res and len(d_used) >= 3:
            r = res['A']
            pred = r['c0'] + r['gamma'] * d_used
            resids = np.abs(lnP_used - pred)
            pprint(f"Linearity check:")
            pprint(f"  Max |residual| from linear fit: {resids.max():.4f}")
            if resids.max() < 0.5:
                pprint(f"  --> Exponential decay CONFIRMED")
            elif resids.max() < 1.5:
                pprint(f"  --> INCONCLUSIVE")
            else:
                pprint(f"  --> Exponential decay REJECTED")
            pprint()

    # Upper bounds from zero-count d values
    zeros = [(d, N) for d, P, N in zip(d_vals, P_vals, N_vals) if P == 0]
    if zeros:
        pprint("Upper bounds (zero-detection d values):")
        for d_z, N_z in zeros:
            pprint(f"  d={d_z}: P < {1.0/N_z:.1e}, ln(1/P) > {np.log(N_z):.2f}")
        pprint()

    # Local gamma between consecutive decay points
    if len(d_decay) >= 2:
        pprint("Local gamma between consecutive points:")
        for i in range(len(d_decay) - 1):
            d1, d2 = d_decay[i], d_decay[i + 1]
            lp1, lp2 = lnP_decay[i], lnP_decay[i + 1]  # ln(1/P) values
            # Corrected: use lnP_decay not lnP_fit
            g_local = (lp2 - lp1) / (d2 - d1)
            pprint(f"  d={d1:.0f}->{d2:.0f}: gamma_local = {g_local:.4f}")
        pprint()

    # ================================================================
    #  COMPARISON TABLE
    # ================================================================
    pprint("=" * 70)
    pprint("  COMPARISON TABLE")
    pprint("=" * 70)
    pprint()

    gamma_poly = res_decay['A']['gamma'] if 'A' in res_decay else None
    r2_poly = res_decay['A']['R2'] if 'A' in res_decay else None
    beta_poly = res_decay['C']['beta'] if 'C' in res_decay else None

    pprint(f"{'Source':<15} | {'gamma':>8} | {'R^2':>8} | {'Model class':<25}")
    pprint("-" * 65)
    pprint(f"{'gamma_cusp':<15} | {GAMMA_CUSP:8.4f} | {'0.995':>8} | "
           f"{'1D Hill function':<25}")
    pprint(f"{'gamma_Hopf':<15} | {GAMMA_HOPF:8.4f} | {'0.9997':>8} | "
           f"{'2D Hill function':<25}")
    if gamma_poly is not None:
        pprint(f"{'gamma_poly':<15} | {gamma_poly:8.4f} | {r2_poly:8.4f} | "
               f"{'2D polynomial':<25}")
    pprint()

    if beta_poly is not None:
        pprint(f"Stretched exponent beta_poly = {beta_poly:.4f}")
        pprint(f"  (beta=1 is linear; Study 16 Hopf had accelerating decay)")
        pprint()

    # ================================================================
    #  CONCLUSION
    # ================================================================
    pprint("=" * 70)
    pprint("  CONCLUSION")
    pprint("=" * 70)
    pprint()

    if gamma_poly is not None:
        ratio = gamma_poly / GAMMA_CUSP

        pprint(f"  gamma_poly = {gamma_poly:.4f}  (R^2 = {r2_poly:.4f})")
        pprint(f"  gamma_cusp = {GAMMA_CUSP:.4f}")
        pprint(f"  gamma_Hopf = {GAMMA_HOPF:.4f}")
        pprint(f"  Ratio (poly/cusp) = {ratio:.3f}")
        pprint()

        if abs(ratio - 1) < 0.30:
            pprint("  INTERPRETATION: GEOMETRIC UNIVERSALITY")
            pprint("  gamma_poly ~ gamma_cusp ~ gamma_Hopf (within 30%)")
            pprint("  --> The exponential decay rate is a property of")
            pprint("      codimension-1 manifold measure in d-parameter space,")
            pprint("      NOT of Hill-function saturation.")
            pprint("  --> gamma ~ 0.20 is a geometric constant independent")
            pprint("      of the function class used to parameterize the ODE.")
            pprint("  --> The bridge is UNIVERSAL across model classes.")
        elif abs(ratio - 1) < 0.50:
            pprint("  INTERPRETATION: MODEL-CLASS-DEPENDENT")
            pprint("  Exponential form P ~ exp(-gamma*d) is universal,")
            pprint("  but gamma depends on the function class.")
            pprint("  --> Hill saturation creates a specific gamma;")
            pprint("      polynomials create a different one.")
            pprint("  --> Framework universal content: exponential FORM,")
            pprint("      not the constant.")
        else:
            pprint("  INTERPRETATION: STRONGLY MODEL-DEPENDENT or NO DECAY")
            pprint("  gamma_poly differs by >50% from gamma_cusp.")
            if gamma_poly < 0:
                pprint("  --> P INCREASES with d for polynomials.")
                pprint("      Hill-specific structure drives the cusp result.")
            else:
                pprint("  --> Different function classes produce qualitatively")
                pprint("      different parameter-space geometry.")
    else:
        pprint("  INSUFFICIENT DATA for conclusion.")
        pprint("  Could not fit gamma_poly.")

    pprint()
    pprint("=" * 70)


if __name__ == "__main__":
    main()
