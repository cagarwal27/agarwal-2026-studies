#!/usr/bin/env python3
"""
Hopf Bridge Scaling: P(stable limit cycle | d) for 2D multi-channel ODEs.
Tests whether cusp bridge result (gamma_cusp ~ 0.20) generalizes to Hopf bifurcations.

Model:
  dx/dt = a1 - b1*x + c1*y + sum_i [r1_i * x^q1_i / (x^q1_i + k1_i^q1_i)]
  dy/dt = a2 - b2*y + c2*x + sum_i [r2_i * y^q2_i / (y^q2_i + k2_i^q2_i)]

Parameters: d = 6 + 6*n_channels  (6 base + 6 per channel)
Detection: Vectorized Newton equilibria -> Jacobian eigenvalues -> ODE integration
Result: P(limit cycle | d), fit gamma_Hopf from ln(1/P) ~ gamma*d

Study 16: First computation testing framework scope beyond bistable systems.
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

# Newton's method: 4x4 grid of initial conditions in [0.1, 5.0]^2
INIT_POINTS = [(x, y) for x in [0.15, 0.6, 1.5, 3.5]
                       for y in [0.15, 0.6, 1.5, 3.5]]
NEWTON_ITERS = 20
NEWTON_MAX_STEP = 2.0
EQ_TOL = 1e-6

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

# d specifications: (n_channels, d, N_samples)
# Sample sizes increased ~5x over prompt spec because P(limit cycle) << P(bistable)
# at the same d. P(lc, d=12) ~ 4e-4 vs P(bistable, d=12) ~ 0.33.
D_SPECS = [
    (1, 12,   100_000),
    (2, 18,   250_000),
    (3, 24,   500_000),
    (4, 30, 1_000_000),
    (5, 36, 2_000_000),
    (6, 42, 5_000_000),
]

GAMMA_CUSP = 0.197


# ============================================================
# PARAMETER GENERATION
# ============================================================

def generate_params(n_ch, N):
    """Generate N random parameter sets for n_ch-channel 2D Hill ODE."""
    return {
        'a1': np.random.uniform(0.05, 0.80, N),
        'a2': np.random.uniform(0.05, 0.80, N),
        'b1': np.random.uniform(0.2, 2.0, N),
        'b2': np.random.uniform(0.2, 2.0, N),
        'c1': np.random.uniform(-1.0, 1.0, N),
        'c2': np.random.uniform(-1.0, 1.0, N),
        'r1': np.random.uniform(0.1, 2.0, (N, n_ch)),
        'q1': np.random.uniform(2.0, 15.0, (N, n_ch)),
        'k1': np.random.uniform(0.3, 2.0, (N, n_ch)),
        'r2': np.random.uniform(0.1, 2.0, (N, n_ch)),
        'q2': np.random.uniform(2.0, 15.0, (N, n_ch)),
        'k2': np.random.uniform(0.3, 2.0, (N, n_ch)),
    }


def extract_single(params, idx):
    """Extract single-sample parameters (scalars + 1D arrays)."""
    return {k: v[idx] for k, v in params.items()}


def precompute_kq(params, n_ch):
    """Pre-compute k^q for all channels (constant during Newton iterations)."""
    kq1 = [params['k1'][:, ch] ** params['q1'][:, ch] for ch in range(n_ch)]
    kq2 = [params['k2'][:, ch] ** params['q2'][:, ch] for ch in range(n_ch)]
    return kq1, kq2


# ============================================================
# VECTORIZED NEWTON'S METHOD FOR EQUILIBRIUM FINDING
# ============================================================

def vectorized_newton(params, n_ch, kq1, kq2, x0_val, y0_val):
    """
    Vectorized Newton's method: find equilibria for all B samples
    starting from (x0_val, y0_val).

    Uses pre-computed k^q arrays. Computes F, G and Jacobian in
    a single pass to minimize redundant power operations.

    Returns x_eq, y_eq, converged (all shape (B,)).
    """
    B = len(params['a1'])
    x = np.full(B, x0_val, dtype=np.float64)
    y = np.full(B, y0_val, dtype=np.float64)

    a1, a2 = params['a1'], params['a2']
    b1, b2 = params['b1'], params['b2']
    c1, c2 = params['c1'], params['c2']

    for _ in range(NEWTON_ITERS):
        xc = np.maximum(x, 1e-12)
        yc = np.maximum(y, 1e-12)

        # Evaluate F, G and Jacobian simultaneously (shared x^q computation)
        F = a1 - b1 * xc + c1 * yc
        G = a2 - b2 * yc + c2 * xc
        dFdx = -b1.copy()
        dGdy = -b2.copy()

        for ch in range(n_ch):
            q1c = params['q1'][:, ch]
            xq = xc ** q1c
            xqm1 = xq / xc                 # x^(q-1) = x^q / x
            denom1 = xq + kq1[ch]
            hill1 = xq / denom1
            F += params['r1'][:, ch] * hill1
            dFdx += params['r1'][:, ch] * q1c * xqm1 * kq1[ch] / (denom1 * denom1)

            q2c = params['q2'][:, ch]
            yq = yc ** q2c
            yqm1 = yq / yc
            denom2 = yq + kq2[ch]
            hill2 = yq / denom2
            G += params['r2'][:, ch] * hill2
            dGdy += params['r2'][:, ch] * q2c * yqm1 * kq2[ch] / (denom2 * denom2)

        # Solve 2x2 linear system: J * [dx, dy]^T = -[F, G]^T
        # J^{-1} = (1/det) * [[dGdy, -c1], [-c2, dFdx]]
        det = dFdx * dGdy - c1 * c2
        safe_det = np.where(np.abs(det) > 1e-15, det, 1e-15)
        dx = (-F * dGdy + G * c1) / safe_det
        dy = (F * c2 - G * dFdx) / safe_det

        # Damped step: clip to prevent wild jumps
        dx = np.clip(dx, -NEWTON_MAX_STEP, NEWTON_MAX_STEP)
        dy = np.clip(dy, -NEWTON_MAX_STEP, NEWTON_MAX_STEP)

        x = np.maximum(x + dx, 1e-12)
        y = np.maximum(y + dy, 1e-12)

    # Final residual check
    xc = np.maximum(x, 1e-12)
    yc = np.maximum(y, 1e-12)
    F = a1 - b1 * xc + c1 * yc
    G = a2 - b2 * yc + c2 * xc
    for ch in range(n_ch):
        xq = xc ** params['q1'][:, ch]
        F += params['r1'][:, ch] * xq / (xq + kq1[ch])
        yq = yc ** params['q2'][:, ch]
        G += params['r2'][:, ch] * yq / (yq + kq2[ch])

    residual = np.abs(F) + np.abs(G)
    converged = (residual < EQ_TOL) & (x > 0) & (y > 0) & (x < 10) & (y < 10)

    return x, y, converged


def vectorized_check_spiral(params, n_ch, kq1, kq2, x_eq, y_eq, converged):
    """
    Vectorized unstable spiral check at converged equilibria.
    Returns is_spiral mask (shape (B,)).
    """
    xc = np.maximum(x_eq, 1e-12)
    yc = np.maximum(y_eq, 1e-12)
    dFdx = -params['b1'].copy()
    dGdy = -params['b2'].copy()

    for ch in range(n_ch):
        q1c = params['q1'][:, ch]
        xq = xc ** q1c
        xqm1 = xq / xc
        denom1 = xq + kq1[ch]
        dFdx += params['r1'][:, ch] * q1c * xqm1 * kq1[ch] / (denom1 * denom1)

        q2c = params['q2'][:, ch]
        yq = yc ** q2c
        yqm1 = yq / yc
        denom2 = yq + kq2[ch]
        dGdy += params['r2'][:, ch] * q2c * yqm1 * kq2[ch] / (denom2 * denom2)

    tr = dFdx + dGdy
    det = dFdx * dGdy - params['c1'] * params['c2']
    disc = tr * tr - 4 * det

    # Unstable spiral: trace > 0 AND discriminant < 0 (complex eigenvalues)
    return converged & (tr > 0) & (disc < 0)


# ============================================================
# SCALAR EVALUATION (for ODE integration)
# ============================================================

def eval_FG(x, y, p, n_ch):
    """Evaluate (F, G) at scalar point (x, y) for single parameter set."""
    xc = max(x, 1e-12)
    yc = max(y, 1e-12)
    F = p['a1'] - p['b1'] * xc + p['c1'] * yc
    G = p['a2'] - p['b2'] * yc + p['c2'] * xc
    for i in range(n_ch):
        xq = xc ** p['q1'][i]
        kq1 = p['k1'][i] ** p['q1'][i]
        F += p['r1'][i] * xq / (xq + kq1)
        yq = yc ** p['q2'][i]
        kq2 = p['k2'][i] ** p['q2'][i]
        G += p['r2'][i] * yq / (yq + kq2)
    return F, G


# ============================================================
# LIMIT CYCLE VERIFICATION VIA ODE INTEGRATION
# ============================================================

def verify_limit_cycle(p, n_ch, x_eq, y_eq):
    """
    Integrate ODE from near unstable equilibrium.
    Check: bounded + periodic in last 50% = stable limit cycle.
    """
    x0 = x_eq + 0.01
    y0 = y_eq + 0.01

    def rhs(t, z):
        F, G = eval_FG(z[0], z[1], p, n_ch)
        return [F, G]

    def diverge(t, z):
        return DIVERGE_THRESHOLD - max(abs(z[0]), abs(z[1]))
    diverge.terminal = True

    n_pts = int(T_INTEGRATE * T_DENSITY)
    t_eval = np.linspace(0, T_INTEGRATE, n_pts)

    try:
        sol = solve_ivp(rhs, (0, T_INTEGRATE), [x0, y0],
                        method='RK45', t_eval=t_eval, events=diverge,
                        rtol=1e-6, atol=1e-8, max_step=2.0)

        if sol.status == 1:
            return False
        if sol.t[-1] < T_INTEGRATE * 0.8:
            return False

        # Analyze last 50%
        n_half = len(sol.t) // 2
        x_tr = sol.y[0, n_half:]
        t_tr = sol.t[n_half:]

        if len(x_tr) < 30:
            return False

        # Peak detection
        peaks = np.where(
            (x_tr[1:-1] > x_tr[:-2]) & (x_tr[1:-1] > x_tr[2:])
        )[0] + 1

        if len(peaks) < MIN_PEAKS:
            return False

        heights = x_tr[peaks]
        times = t_tr[peaks]

        # Amplitude consistency
        mean_h = np.mean(heights)
        if mean_h == 0:
            return False
        if np.std(heights) / abs(mean_h) > PERIODIC_CV:
            return False

        # Period consistency
        periods = np.diff(times)
        if len(periods) < 2:
            return False
        mean_per = np.mean(periods)
        if mean_per == 0:
            return False
        if np.std(periods) / abs(mean_per) > PERIODIC_CV:
            return False

        # Non-trivial amplitude
        if np.max(x_tr) - np.min(x_tr) < MIN_AMPLITUDE:
            return False

        return True

    except Exception:
        return False


# ============================================================
# MAIN BATCH PROCESSING
# ============================================================

def check_batch(n_ch, N):
    """
    Screen N random parameter sets for stable limit cycles.

    Pipeline per batch:
    1. Pre-compute k^q (constant across Newton iterations)
    2. Vectorized Newton from 16 initial conditions -> find equilibria
    3. Vectorized Jacobian check -> identify unstable spirals
    4. Per-spiral ODE integration -> verify limit cycles

    Returns (n_lc, n_spiral).
    """
    n_lc = 0
    n_spiral = 0
    n_done = 0
    batch_num = 0

    while n_done < N:
        bs = min(BATCH_SIZE, N - n_done)
        batch_num += 1

        params = generate_params(n_ch, bs)
        kq1, kq2 = precompute_kq(params, n_ch)

        # Collect spirals across all initial conditions
        spiral_eqs = {}   # local_idx -> (x_eq, y_eq)

        for x0, y0 in INIT_POINTS:
            x_eq, y_eq, conv = vectorized_newton(params, n_ch, kq1, kq2, x0, y0)
            is_spiral = vectorized_check_spiral(params, n_ch, kq1, kq2,
                                                 x_eq, y_eq, conv)

            for idx in np.where(is_spiral)[0]:
                idx_int = int(idx)
                if idx_int not in spiral_eqs:
                    spiral_eqs[idx_int] = (x_eq[idx], y_eq[idx])

        n_spiral += len(spiral_eqs)

        # ODE integration for each spiral sample
        for idx, (x_e, y_e) in spiral_eqs.items():
            p = extract_single(params, idx)
            if verify_limit_cycle(p, n_ch, x_e, y_e):
                n_lc += 1

        n_done += bs
        report_every = max(1, N // (BATCH_SIZE * 5))
        if batch_num % report_every == 0 or n_done >= N:
            pprint(f"    {n_done:>10,}/{N:,} | "
                   f"spiral {n_spiral:,} | lc {n_lc}")

    return n_lc, n_spiral


# ============================================================
# FITTING MODELS (matches cusp bridge structure)
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
    pprint("  HOPF BRIDGE SCALING: P(stable limit cycle | d)")
    pprint("  2D multi-channel Hill-function ODE")
    pprint("=" * 70)
    pprint()
    pprint("Model:")
    pprint("  dx/dt = a1 - b1*x + c1*y + sum_i [r1_i * h(x, q1_i, k1_i)]")
    pprint("  dy/dt = a2 - b2*y + c2*x + sum_i [r2_i * h(y, q2_i, k2_i)]")
    pprint(f"  d = 6 + 6*n_channels")
    pprint()
    pprint(f"Equilibria: vectorized Newton, {len(INIT_POINTS)} initial conditions, "
           f"{NEWTON_ITERS} iterations")
    pprint(f"ODE: t_max={T_INTEGRATE:.0f}, RK45, diverge at |z|>{DIVERGE_THRESHOLD}")
    pprint(f"Periodic: CV<{PERIODIC_CV}, min {MIN_PEAKS} peaks, amp>{MIN_AMPLITUDE}")
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

    for n_ch, d, N in D_SPECS:
        pprint(f"--- d = {d} ({n_ch} channel{'s' if n_ch > 1 else ''}), "
               f"N = {N:,} ---")
        t0 = time.time()

        n_lc, n_sp = check_batch(n_ch, N)
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
    hdr = (f"{'d':>4} | {'N':>10} | {'N_spiral':>8} | "
           f"{'N_lc':>6} | {'P(lc)':>12} | {'ln(1/P)':>8}")
    pprint(hdr)
    pprint("-" * len(hdr))
    for i in range(len(d_vals)):
        lnP = np.log(1.0 / P_vals[i]) if P_vals[i] > 0 else np.inf
        pprint(f"{d_vals[i]:4d} | {N_vals[i]:10,} | {n_sp_vals[i]:8,} | "
               f"{n_lc_vals[i]:6,} | "
               f"{P_vals[i]:12.6e} | {lnP:8.3f}")
    pprint()

    # ================================================================
    #  PHASE 2: B analog (SKIPPED -- optional for first pass)
    # ================================================================
    pprint("Phase 2 (B analog via noise-induced amplitude death): SKIPPED")
    pprint("  Requires stochastic simulation -- deferred to follow-up.\n")

    # ================================================================
    #  PHASE 3: Fit gamma_Hopf, compare to gamma_cusp
    # ================================================================
    pprint("=" * 70)
    pprint("  PHASE 3: Fit gamma_Hopf")
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

    pprint(f"Fitting {len(d_fit)} points, d = {d_fit.min():.0f}-{d_fit.max():.0f}")
    pprint()

    res = fit_models(d_fit, lnP_fit)

    # Model A: linear
    if 'A' in res:
        r = res['A']
        pprint(f"Model A (linear): ln(1/P) = {r['c0']:.4f} + {r['gamma']:.6f} * d")
        pprint(f"  gamma_Hopf = {r['gamma']:.6f}")
        pprint(f"  R^2        = {r['R2']:.6f}")
        ratio = r['gamma'] / GAMMA_CUSP
        diff_pct = abs(ratio - 1) * 100
        pprint(f"  gamma_cusp = {GAMMA_CUSP:.6f}")
        pprint(f"  ratio      = {ratio:.3f}  ({diff_pct:.1f}% difference)")
        if diff_pct < 20:
            pprint(f"  --> CONSISTENT (within 20%)")
        elif diff_pct < 30:
            pprint(f"  --> MARGINAL (within 30%)")
        else:
            pprint(f"  --> DIFFERENT")
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
    if 'A' in res and len(d_fit) >= 3:
        r = res['A']
        pred = r['c0'] + r['gamma'] * d_fit
        resids = np.abs(lnP_fit - pred)
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

    # ================================================================
    #  CONCLUSION
    # ================================================================
    pprint("=" * 70)
    pprint("  CONCLUSION")
    pprint("=" * 70)
    pprint()

    if 'A' in res:
        gamma = res['A']['gamma']
        r2 = res['A']['R2']
        ratio = gamma / GAMMA_CUSP

        pprint(f"  gamma_Hopf = {gamma:.4f}  (R^2 = {r2:.4f})")
        pprint(f"  gamma_cusp = {GAMMA_CUSP:.4f}")
        pprint(f"  Ratio      = {ratio:.3f}")
        pprint()

        if abs(ratio - 1) < 0.20:
            pprint("  INTERPRETATION: gamma_Hopf ~ gamma_cusp (within 20%)")
            pprint("  --> UNIVERSAL concentration of measure. The persistence-search")
            pprint("      bridge generalizes beyond bistability. S(d) ~ exp(gamma*d)")
            pprint("      with gamma ~ 0.20 for codimension-1 bifurcations.")
            pprint("      The fire equation applies to ANY organized state.")
        elif abs(ratio - 1) < 0.30:
            pprint("  INTERPRETATION: MARGINAL (20-30% difference)")
            pprint("  --> Needs more data to distinguish universal vs type-dependent.")
        elif gamma > 0:
            pprint("  INTERPRETATION: gamma_Hopf != gamma_cusp")
            pprint("  --> BIFURCATION-TYPE-DEPENDENT. Exponential form P ~ exp(-gamma*d)")
            pprint("      is universal, but gamma depends on bifurcation geometry.")
            pprint("      This is the 'alpha pattern': f(k) = alpha^k universal,")
            pprint("      alpha depends on system type.")
        else:
            pprint("  INTERPRETATION: ANOMALOUS -- no exponential decay for Hopf.")
            pprint("  --> Cusp result may be specific to bistable geometry.")

    pprint()
    pprint("=" * 70)


if __name__ == "__main__":
    main()
