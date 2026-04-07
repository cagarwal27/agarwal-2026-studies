#!/usr/bin/env python3
"""
Polynomial + Degradation Test (GPU-enabled): Does enforced degradation recover gamma ~ 0.20?

Study 18 showed P(limit cycle | d) ~ 1.5% (flat) for random polynomial ODEs.
Study 16 showed gamma = 0.175 for Hill ODEs with degradation structure.
Hypothesis: the exponential decay comes from degradation opposing activation,
not from Hill-function saturation.

Model (identical to Study 18 except two coefficients are sign-constrained):
  dx/dt = -b1*x + [remaining poly terms] - eps*x*(x^2+y^2)
  dy/dt = -b2*y + [remaining poly terms] - eps*y*(x^2+y^2)

  b1 ~ U[0.2, 2.0], b2 ~ U[0.2, 2.0]  (forced positive = degradation)
  All other coefficients identical to Study 18.

Parameters: d = (p+1)(p+2) total random parameters (same count as Study 18).
Detection: Vectorized Newton -> Jacobian eigenvalues -> batch RK4
Result: P(limit cycle | d), fit gamma and compare to Studies 11, 16, 18.

GPU: Set USE_GPU = True when running on Colab with GPU runtime.
     Requires: !pip install cupy-cuda12x
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import time
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# GPU BACKEND (CuPy if available, else NumPy)
# ============================================================

USE_GPU = False  # Set True on Colab with GPU

try:
    if USE_GPU:
        import cupy as cp
        xp = cp
        print("GPU backend: CuPy", flush=True)
    else:
        raise ImportError
except ImportError:
    xp = np
    if USE_GPU:
        print("CuPy not available, falling back to NumPy", flush=True)
    else:
        print("CPU backend: NumPy", flush=True)

np.random.seed(42)


def pprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def to_xp(arr):
    """Move numpy array to compute device."""
    return arr if xp is np else xp.asarray(arr)


def to_np(arr):
    """Move array back to CPU numpy."""
    return arr if xp is np else arr.get()


# ============================================================
# CONSTANTS
# ============================================================

EPS_DAMP = 0.01

INIT_POINTS = [(x, y) for x in [-2.0, -0.5, 0.5, 2.0]
                       for y in [-2.0, -0.5, 0.5, 2.0]]
NEWTON_ITERS = 20
NEWTON_MAX_STEP = 2.0
EQ_TOL = 1e-6
EQ_MAX_RADIUS = 10.0

T_INTEGRATE = 300.0
DIVERGE_THRESHOLD = 100.0

PERIODIC_CV = 0.20
MIN_PEAKS = 5
MIN_AMPLITUDE = 0.01

BATCH_SIZE = 10_000

COEFF_RANGES = {
    0: 1.0, 1: 1.5, 2: 0.5, 3: 0.3, 4: 0.2, 5: 0.15, 6: 0.1,
}

DEGRAD_MIN = 0.2
DEGRAD_MAX = 2.0

D_SPECS = [
    (2, 12,    100_000),
    (3, 20,    200_000),
    (4, 30,    500_000),
    (5, 42,  2_000_000),
    (6, 56,  5_000_000),
]

GAMMA_CUSP = 0.197
GAMMA_HOPF = 0.175
GAMMA_POLY = 0.010


# ============================================================
# TERM INDEX BUILDING
# ============================================================

def build_terms(p):
    terms = []
    for total in range(p + 1):
        for i in range(total + 1):
            j = total - i
            terms.append((i, j))
    M = len(terms)
    assert M == (p + 1) * (p + 2) // 2
    return terms, M


def find_degradation_indices(terms):
    idx_x = idx_y = None
    for m, (i, j) in enumerate(terms):
        if i == 1 and j == 0: idx_x = m
        if i == 0 and j == 1: idx_y = m
    assert idx_x is not None and idx_y is not None
    return idx_x, idx_y


# ============================================================
# PARAMETER GENERATION (with degradation override)
# ============================================================

def generate_poly_coefficients(p, N):
    """Generate coefficients with degradation: linear self-terms forced negative."""
    terms, M = build_terms(p)
    a_coeffs = np.empty((N, M), dtype=np.float64)
    b_coeffs = np.empty((N, M), dtype=np.float64)

    for idx, (i, j) in enumerate(terms):
        k = i + j
        r = COEFF_RANGES[k]
        a_coeffs[:, idx] = np.random.uniform(-r, r, N)
        b_coeffs[:, idx] = np.random.uniform(-r, r, N)

    # DEGRADATION OVERRIDE
    idx_x, idx_y = find_degradation_indices(terms)
    a_coeffs[:, idx_x] = -np.random.uniform(DEGRAD_MIN, DEGRAD_MAX, N)
    b_coeffs[:, idx_y] = -np.random.uniform(DEGRAD_MIN, DEGRAD_MAX, N)

    return a_coeffs, b_coeffs, terms


# ============================================================
# VECTORIZED POLYNOMIAL EVALUATION (runs on xp)
# ============================================================

def eval_poly_batch(a_coeffs, terms, x, y, p):
    N = len(x)
    x_pow = [xp.ones(N, dtype=xp.float64)]
    y_pow = [xp.ones(N, dtype=xp.float64)]
    for k in range(1, p + 1):
        x_pow.append(x_pow[-1] * x)
        y_pow.append(y_pow[-1] * y)
    result = xp.zeros(N, dtype=xp.float64)
    for m, (i, j) in enumerate(terms):
        result += a_coeffs[:, m] * x_pow[i] * y_pow[j]
    return result


def eval_poly_jacobian_batch(a_coeffs, b_coeffs, terms, x, y, p):
    N = len(x)
    x_pow = [xp.ones(N, dtype=xp.float64)]
    y_pow = [xp.ones(N, dtype=xp.float64)]
    for k in range(1, p + 1):
        x_pow.append(x_pow[-1] * x)
        y_pow.append(y_pow[-1] * y)
    dFdx = xp.zeros(N, dtype=xp.float64)
    dFdy = xp.zeros(N, dtype=xp.float64)
    dGdx = xp.zeros(N, dtype=xp.float64)
    dGdy = xp.zeros(N, dtype=xp.float64)
    for m, (i, j) in enumerate(terms):
        if i >= 1:
            term = i * x_pow[i - 1] * y_pow[j]
            dFdx += a_coeffs[:, m] * term
            dGdx += b_coeffs[:, m] * term
        if j >= 1:
            term = j * x_pow[i] * y_pow[j - 1]
            dFdy += a_coeffs[:, m] * term
            dGdy += b_coeffs[:, m] * term
    return dFdx, dFdy, dGdx, dGdy


def add_damping_to_jacobian(dFdx, dFdy, dGdx, dGdy, x, y):
    x2, y2, xy = x * x, y * y, x * y
    dFdx -= EPS_DAMP * (3.0 * x2 + y2)
    dFdy -= EPS_DAMP * 2.0 * xy
    dGdx -= EPS_DAMP * 2.0 * xy
    dGdy -= EPS_DAMP * (x2 + 3.0 * y2)
    return dFdx, dFdy, dGdx, dGdy


# ============================================================
# VECTORIZED NEWTON
# ============================================================

def vectorized_newton(a_coeffs, b_coeffs, terms, p, x0_val, y0_val):
    N = a_coeffs.shape[0]
    x = xp.full(N, x0_val, dtype=xp.float64)
    y = xp.full(N, y0_val, dtype=xp.float64)

    for _ in range(NEWTON_ITERS):
        F = eval_poly_batch(a_coeffs, terms, x, y, p)
        G = eval_poly_batch(b_coeffs, terms, x, y, p)
        r2 = x * x + y * y
        F -= EPS_DAMP * x * r2
        G -= EPS_DAMP * y * r2

        dFdx, dFdy, dGdx, dGdy = eval_poly_jacobian_batch(
            a_coeffs, b_coeffs, terms, x, y, p)
        dFdx, dFdy, dGdx, dGdy = add_damping_to_jacobian(
            dFdx, dFdy, dGdx, dGdy, x, y)

        det = dFdx * dGdy - dFdy * dGdx
        safe_det = xp.where(xp.abs(det) > 1e-15, det, 1e-15)
        dx = (-F * dGdy + G * dFdy) / safe_det
        dy = (F * dGdx - G * dFdx) / safe_det
        dx = xp.clip(dx, -NEWTON_MAX_STEP, NEWTON_MAX_STEP)
        dy = xp.clip(dy, -NEWTON_MAX_STEP, NEWTON_MAX_STEP)
        x = x + dx
        y = y + dy

    F = eval_poly_batch(a_coeffs, terms, x, y, p)
    G = eval_poly_batch(b_coeffs, terms, x, y, p)
    r2 = x * x + y * y
    F -= EPS_DAMP * x * r2
    G -= EPS_DAMP * y * r2
    residual = xp.abs(F) + xp.abs(G)
    radius = xp.sqrt(x * x + y * y)
    converged = (residual < EQ_TOL) & (radius < EQ_MAX_RADIUS)
    return x, y, converged


# ============================================================
# SPIRAL CHECK
# ============================================================

def vectorized_check_spiral(a_coeffs, b_coeffs, terms, p,
                            x_eq, y_eq, converged):
    dFdx, dFdy, dGdx, dGdy = eval_poly_jacobian_batch(
        a_coeffs, b_coeffs, terms, x_eq, y_eq, p)
    dFdx, dFdy, dGdx, dGdy = add_damping_to_jacobian(
        dFdx, dFdy, dGdx, dGdy, x_eq, y_eq)
    tr = dFdx + dGdy
    det = dFdx * dGdy - dFdy * dGdx
    disc = tr * tr - 4.0 * det
    return converged & (tr > 0) & (disc < 0)


# ============================================================
# BATCH RK4 + PERIODICITY
# ============================================================

def batch_rhs(a_batch, b_batch, terms, p, x, y):
    K = len(x)
    x_pow = [xp.ones(K, dtype=xp.float64)]
    y_pow = [xp.ones(K, dtype=xp.float64)]
    for k in range(1, p + 1):
        x_pow.append(x_pow[-1] * x)
        y_pow.append(y_pow[-1] * y)
    F = xp.zeros(K, dtype=xp.float64)
    G = xp.zeros(K, dtype=xp.float64)
    for m, (i, j) in enumerate(terms):
        mono = x_pow[i] * y_pow[j]
        F += a_batch[:, m] * mono
        G += b_batch[:, m] * mono
    r2 = x * x + y * y
    F -= EPS_DAMP * x * r2
    G -= EPS_DAMP * y * r2
    return F, G


def vectorized_rk4_integrate(a_batch, b_batch, terms, p, x0, y0,
                              dt=0.1, n_steps=3000, save_every=2):
    K = len(x0)
    x, y = x0.copy(), y0.copy()
    alive = xp.ones(K, dtype=bool)
    n_saved = n_steps // save_every
    x_traj = xp.empty((K, n_saved), dtype=xp.float64)
    save_idx = 0

    for step in range(n_steps):
        k1x, k1y = batch_rhs(a_batch, b_batch, terms, p, x, y)
        k2x, k2y = batch_rhs(a_batch, b_batch, terms, p,
                              x + 0.5 * dt * k1x, y + 0.5 * dt * k1y)
        k3x, k3y = batch_rhs(a_batch, b_batch, terms, p,
                              x + 0.5 * dt * k2x, y + 0.5 * dt * k2y)
        k4x, k4y = batch_rhs(a_batch, b_batch, terms, p,
                              x + dt * k3x, y + dt * k3y)

        x_new = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        y_new = y + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)

        diverged = (xp.abs(x_new) > DIVERGE_THRESHOLD) | \
                   (xp.abs(y_new) > DIVERGE_THRESHOLD) | \
                   xp.isnan(x_new) | xp.isnan(y_new)
        alive &= ~diverged
        x = xp.where(alive, x_new, x)
        y = xp.where(alive, y_new, y)

        if step % save_every == 0 and save_idx < n_saved:
            x_traj[:, save_idx] = x
            save_idx += 1

    return x_traj[:, :save_idx], alive


def check_periodicity_batch(x_traj, alive, dt_save):
    """Runs on CPU (numpy) — sequential per trajectory."""
    K, n_saved = x_traj.shape
    is_lc = np.zeros(K, dtype=bool)

    for k in np.where(alive)[0]:
        x_tr = x_traj[k, n_saved // 2:]
        if len(x_tr) < 30:
            continue
        peaks = np.where(
            (x_tr[1:-1] > x_tr[:-2]) & (x_tr[1:-1] > x_tr[2:])
        )[0] + 1
        if len(peaks) < MIN_PEAKS:
            continue
        heights = x_tr[peaks]
        times = peaks * dt_save
        mean_h = np.mean(heights)
        if mean_h == 0 or np.std(heights) / abs(mean_h) > PERIODIC_CV:
            continue
        periods = np.diff(times)
        if len(periods) < 2:
            continue
        mean_per = np.mean(periods)
        if mean_per == 0 or np.std(periods) / abs(mean_per) > PERIODIC_CV:
            continue
        if np.max(x_tr) - np.min(x_tr) < MIN_AMPLITUDE:
            continue
        is_lc[k] = True

    return is_lc


# ============================================================
# BATCH PROCESSING
# ============================================================

def check_batch(p, N):
    terms, M = build_terms(p)
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

        a_np, b_np, _ = generate_poly_coefficients(p, bs)
        a_coeffs = to_xp(a_np)
        b_coeffs = to_xp(b_np)

        spiral_eqs = {}
        for x0, y0 in INIT_POINTS:
            x_eq, y_eq, conv = vectorized_newton(
                a_coeffs, b_coeffs, terms, p, x0, y0)
            is_spiral = vectorized_check_spiral(
                a_coeffs, b_coeffs, terms, p, x_eq, y_eq, conv)

            is_sp_np = to_np(is_spiral)
            x_eq_np = to_np(x_eq)
            y_eq_np = to_np(y_eq)
            for idx in np.where(is_sp_np)[0]:
                idx_int = int(idx)
                if idx_int not in spiral_eqs:
                    spiral_eqs[idx_int] = (x_eq_np[idx], y_eq_np[idx])

        n_spiral += len(spiral_eqs)

        if len(spiral_eqs) > 0:
            sp_indices = list(spiral_eqs.keys())
            sp_x0 = np.array([spiral_eqs[i][0] + 0.01 for i in sp_indices])
            sp_y0 = np.array([spiral_eqs[i][1] + 0.01 for i in sp_indices])

            x_traj, alive = vectorized_rk4_integrate(
                to_xp(a_np[sp_indices]), to_xp(b_np[sp_indices]),
                terms, p, to_xp(sp_x0), to_xp(sp_y0),
                dt=RK4_DT, n_steps=RK4_STEPS, save_every=RK4_SAVE_EVERY)

            is_lc = check_periodicity_batch(to_np(x_traj), to_np(alive), DT_SAVE)
            n_lc += int(np.sum(is_lc))

        n_done += bs
        report_every = max(1, N // (BATCH_SIZE * 5))
        if batch_num % report_every == 0 or n_done >= N:
            pprint(f"    {n_done:>10,}/{N:,} | "
                   f"spiral {n_spiral:,} | lc {n_lc}")

    return n_lc, n_spiral


# ============================================================
# FITTING
# ============================================================

def fit_models(d_arr, lnP_inv):
    results = {}
    if len(d_arr) < 2:
        return results

    slope, intercept, r_val, _, _ = linregress(d_arr, lnP_inv)
    results['A'] = {'gamma': slope, 'c0': intercept, 'R2': r_val**2}

    if len(d_arr) >= 3:
        try:
            def mb(d, c0, g1, g2): return c0 + g1 * d + g2 * d**2
            po, _ = curve_fit(mb, d_arr, lnP_inv, p0=[0, 0.1, 0.001])
            yp = mb(d_arr, *po)
            ss_r = np.sum((lnP_inv - yp)**2)
            ss_t = np.sum((lnP_inv - np.mean(lnP_inv))**2)
            results['B'] = {'g1': po[1], 'g2': po[2], 'c0': po[0],
                            'R2': 1 - ss_r / ss_t if ss_t > 0 else 0}
        except Exception:
            pass

    if len(d_arr) >= 3:
        try:
            def mc(d, c0, gamma, beta): return c0 + gamma * d**beta
            po, _ = curve_fit(mc, d_arr, lnP_inv, p0=[0, 0.1, 1.0],
                              bounds=([-np.inf, 0, 0.1], [np.inf, np.inf, 3.0]),
                              maxfev=10000)
            yp = mc(d_arr, *po)
            ss_r = np.sum((lnP_inv - yp)**2)
            ss_t = np.sum((lnP_inv - np.mean(lnP_inv))**2)
            results['C'] = {'gamma': po[1], 'beta': po[2], 'c0': po[0],
                            'R2': 1 - ss_r / ss_t if ss_t > 0 else 0}
        except Exception:
            pass

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    pprint("=" * 70)
    pprint("  POLYNOMIAL + DEGRADATION: P(stable limit cycle | d)")
    pprint("  Does enforced degradation recover gamma ~ 0.20?")
    pprint("=" * 70)
    pprint()
    pprint("Model:")
    pprint("  dx/dt = -b1*x + [poly terms] - eps*x*(x^2+y^2)")
    pprint("  dy/dt = -b2*y + [poly terms] - eps*y*(x^2+y^2)")
    pprint(f"  b1, b2 ~ U[{DEGRAD_MIN}, {DEGRAD_MAX}] (degradation)")
    pprint(f"  eps = {EPS_DAMP} (fixed cubic damping)")
    pprint(f"  d = (p+1)(p+2) total random parameters")
    pprint(f"  Backend: {'GPU (CuPy)' if xp is not np else 'CPU (NumPy)'}")
    pprint()
    pprint(f"Equilibria: vectorized Newton, {len(INIT_POINTS)} init pts, "
           f"{NEWTON_ITERS} iters")
    pprint(f"ODE: t_max={T_INTEGRATE:.0f}, RK4, diverge at |z|>{DIVERGE_THRESHOLD}")
    pprint(f"Periodic: CV<{PERIODIC_CV}, min {MIN_PEAKS} peaks, amp>{MIN_AMPLITUDE}")
    pprint()

    pprint("Degree-dimension table:")
    for p_deg, d_exp, N in D_SPECS:
        terms, M = build_terms(p_deg)
        d_actual = 2 * M
        assert d_actual == d_exp
        idx_x, idx_y = find_degradation_indices(terms)
        pprint(f"  p={p_deg}: {M} terms/eq, d={d_actual}, N={N:,}, "
               f"degrad idx=({idx_x},{idx_y})")
    pprint()

    # ================================================================
    #  PHASE 1: P(limit cycle | d)
    # ================================================================
    pprint("=" * 70)
    pprint("  PHASE 1: P(limit cycle | d)")
    pprint("=" * 70)
    pprint()

    d_vals, P_vals, N_vals = [], [], []
    n_lc_vals, n_sp_vals = [], []
    consecutive_zeros = 0

    for p_deg, d, N in D_SPECS:
        pprint(f"--- d = {d} (degree {p_deg} poly+degradation), N = {N:,} ---")
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
            pprint("STOP: Two consecutive d with zero limit cycles.\n")
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
    #  DIAGNOSTIC: d=12 comparison to Study 18
    # ================================================================
    pprint("=" * 70)
    pprint("  DIAGNOSTIC: Effect of degradation at d=12")
    pprint("=" * 70)
    pprint()
    P_S18_d12 = 0.01857
    if len(P_vals) > 0 and P_vals[0] > 0:
        ratio = P_vals[0] / P_S18_d12
        pprint(f"  Study 18 (no degradation):   P(d=12) = {P_S18_d12:.4e}")
        pprint(f"  This test (with degradation): P(d=12) = {P_vals[0]:.4e}")
        pprint(f"  Ratio: {ratio:.3f}x")
        if ratio < 0.1:
            pprint("  --> STRONG effect: degradation reduces P by >10x at d=12")
        elif ratio < 0.5:
            pprint("  --> MODERATE effect: reduces P by 2-10x")
        elif ratio < 0.9:
            pprint("  --> WEAK effect: reduces P by <2x")
        else:
            pprint("  --> NO effect at d=12")
    pprint()

    # ================================================================
    #  PHASE 2: Fit gamma
    # ================================================================
    pprint("=" * 70)
    pprint("  PHASE 2: Fit gamma_poly_degrad")
    pprint("=" * 70)
    pprint()

    valid = [(d, P) for d, P in zip(d_vals, P_vals) if P > 0]
    if len(valid) < 2:
        pprint(f"INSUFFICIENT DATA: {len(valid)} point(s) with P > 0.")
        pprint("=" * 70)
        return

    d_fit = np.array([v[0] for v in valid])
    P_fit = np.array([v[1] for v in valid])
    lnP_fit = np.log(1.0 / P_fit)

    # Fitting range: decay regime (past peak of P)
    if len(lnP_fit) >= 3:
        min_idx = np.argmin(lnP_fit)
        if 0 < min_idx < len(lnP_fit) - 1:
            pprint(f"P(lc) peaks near d={d_fit[min_idx]:.0f}. "
                   f"Fitting decay: d >= {d_fit[min_idx]:.0f}")
            d_decay = d_fit[min_idx:]
            lnP_decay = lnP_fit[min_idx:]
        else:
            d_decay = d_fit
            lnP_decay = lnP_fit
    else:
        d_decay = d_fit
        lnP_decay = lnP_fit

    pprint(f"Fitting {len(d_decay)} points, d = {d_decay.min():.0f}-{d_decay.max():.0f}")
    pprint()

    res = fit_models(d_decay, lnP_decay)

    if 'A' in res:
        r = res['A']
        pprint(f"Model A (linear): ln(1/P) = {r['c0']:.4f} + {r['gamma']:.6f} * d")
        pprint(f"  gamma_poly_degrad = {r['gamma']:.6f}")
        pprint(f"  R^2               = {r['R2']:.6f}")
        pprint()

    if 'B' in res:
        r = res['B']
        pprint(f"Model B (quadratic): ln(1/P) = {r['c0']:.4f} + "
               f"{r['g1']:.6f}*d + {r['g2']:.8f}*d^2")
        pprint(f"  R^2 = {r['R2']:.6f}")
        pprint(f"  g2 {'>' if r['g2'] > 0 else '<'} 0: decay "
               f"{'ACCELERATES' if r['g2'] > 0 else 'DECELERATES'}")
        pprint()

    if 'C' in res:
        r = res['C']
        pprint(f"Model C (stretched exp): ln(1/P) = {r['c0']:.4f} + "
               f"{r['gamma']:.6f} * d^{r['beta']:.4f}")
        pprint(f"  R^2  = {r['R2']:.6f}")
        pprint(f"  beta = {r['beta']:.4f} "
               f"({'superlinear' if r['beta'] > 1.05 else 'sublinear' if r['beta'] < 0.95 else '~linear'})")
        pprint()

    if 'A' in res and len(d_decay) >= 3:
        r = res['A']
        pred = r['c0'] + r['gamma'] * d_decay
        resids = np.abs(lnP_decay - pred)
        pprint(f"Linearity check: max |residual| = {resids.max():.4f}")
        if resids.max() < 0.5:
            pprint("  --> Exponential decay CONFIRMED")
        elif resids.max() < 1.5:
            pprint("  --> INCONCLUSIVE")
        else:
            pprint("  --> Exponential decay REJECTED")
        pprint()

    zeros = [(d, N) for d, P, N in zip(d_vals, P_vals, N_vals) if P == 0]
    if zeros:
        pprint("Upper bounds (zero-detection d values):")
        for d_z, N_z in zeros:
            pprint(f"  d={d_z}: P < {1.0/N_z:.1e}, ln(1/P) > {np.log(N_z):.2f}")
        pprint()

    if len(d_decay) >= 2:
        pprint("Local gamma between consecutive points:")
        for i in range(len(d_decay) - 1):
            d1, d2 = d_decay[i], d_decay[i + 1]
            g_local = (lnP_decay[i + 1] - lnP_decay[i]) / (d2 - d1)
            pprint(f"  d={d1:.0f}->{d2:.0f}: gamma_local = {g_local:.4f}")
        pprint()

    # ================================================================
    #  COMPARISON TABLE
    # ================================================================
    pprint("=" * 70)
    pprint("  COMPARISON TABLE")
    pprint("=" * 70)
    pprint()

    gamma_deg = res['A']['gamma'] if 'A' in res else None
    r2_deg = res['A']['R2'] if 'A' in res else None
    beta_deg = res.get('C', {}).get('beta')

    pprint(f"{'Source':<20} | {'gamma':>8} | {'R^2':>8} | {'Model class':<30}")
    pprint("-" * 75)
    pprint(f"{'gamma_cusp':<20} | {GAMMA_CUSP:8.4f} | {'0.995':>8} | "
           f"{'1D Hill (degrad+sat.)':<30}")
    pprint(f"{'gamma_Hopf':<20} | {GAMMA_HOPF:8.4f} | {'0.9997':>8} | "
           f"{'2D Hill (degrad+sat.)':<30}")
    pprint(f"{'gamma_poly':<20} | {GAMMA_POLY:8.4f} | {'0.498':>8} | "
           f"{'2D poly (no degrad)':<30}")
    if gamma_deg is not None:
        pprint(f"{'gamma_poly_degrad':<20} | {gamma_deg:8.4f} | {r2_deg:8.4f} | "
               f"{'2D poly + degradation':<30}")
    pprint()

    # ================================================================
    #  CONCLUSION
    # ================================================================
    pprint("=" * 70)
    pprint("  CONCLUSION")
    pprint("=" * 70)
    pprint()

    if gamma_deg is not None:
        ratio_cusp = gamma_deg / GAMMA_CUSP

        pprint(f"  gamma_poly_degrad = {gamma_deg:.4f}  (R^2 = {r2_deg:.4f})")
        pprint(f"  gamma_cusp        = {GAMMA_CUSP:.4f}")
        pprint(f"  gamma_Hopf        = {GAMMA_HOPF:.4f}")
        pprint(f"  gamma_poly (S18)  = {GAMMA_POLY:.4f}")
        pprint()
        pprint(f"  Ratio (degrad / cusp) = {ratio_cusp:.3f}")
        pprint()

        if abs(ratio_cusp - 1) < 0.30:
            pprint("  RESULT: DEGRADATION DRIVES DECAY")
            pprint("  gamma recovers to ~0.15-0.22 with degradation structure.")
            pprint("  The architecture of biological regulation (degradation")
            pprint("  opposing activation) makes organized states exponentially")
            pprint("  rare. Hill-function saturation is not required.")
        elif gamma_deg > 0.05:
            pprint("  RESULT: PARTIAL RECOVERY")
            pprint("  Degradation creates some exponential decay but not the")
            pprint("  full gamma ~ 0.20. Both degradation AND saturation may")
            pprint("  contribute.")
        elif gamma_deg > 0.03:
            pprint("  RESULT: WEAK EFFECT")
            pprint("  Degradation has a small effect on the decay rate.")
            pprint("  Additional structure (saturation, boundedness) likely needed.")
        else:
            pprint("  RESULT: DEGRADATION ALONE DOES NOT DRIVE DECAY")
            pprint("  gamma stays flat even with degradation. The exponential")
            pprint("  decay requires Hill-function saturation specifically.")
    else:
        pprint("  INSUFFICIENT DATA for conclusion.")

    pprint()
    pprint("=" * 70)


if __name__ == "__main__":
    main()
