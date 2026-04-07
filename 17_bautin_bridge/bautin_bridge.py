#!/usr/bin/env python3
"""
BAUTIN BRIDGE COMPUTATION (Study 17)
=====================================
Tests B invariance and habitable zone [1.8, 6.0] for fixed-point / limit-cycle
bistability (Bautin / subcritical Hopf), not just two fixed points (cusp).

Part 1: B from Bautin normal form (fast, analytical) -- ~5 min
Part 2: P(FP-LC bistable | d) dimensional scaling -- hours per d
Part 3: B computation for multi-channel Bautin configs -- hours

Dependencies: numpy, scipy
Run: python3 bautin_bridge.py

Reference: EQUATIONS.md Sections 6, 9; cusp_bridge_derivation.py
"""

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq, fsolve
from scipy.stats import linregress
import sys
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# ================================================================
# CONFIGURATION
# ================================================================

SAVE_DIR = './results'
os.makedirs(SAVE_DIR, exist_ok=True)

D_TARGET = 100.0
B_HAB_LO, B_HAB_HI = 1.8, 6.0
N_GRID_MFPT = 20000

# Part 1: Bautin normal form
N_NORMAL_FORM = 300

# Part 2: Dimensional scaling
SPECS_P2 = [
    # (n_channels, d, N_samples)
    (1, 12, 50_000),
    (2, 18, 100_000),
    (3, 24, 200_000),
    (4, 30, 500_000),
    (5, 36, 1_000_000),
    (6, 42, 2_000_000),
]
BATCH_SIZE_P2 = 20_000
GRID_N_2D = 20          # 2D grid resolution for equilibrium search
EQ_THRESHOLD = 0.5      # |F|+|G| threshold for candidate equilibrium
HOPF_RE_THRESHOLD = -1.0  # max Re(lambda) for "near Hopf" filter

# Part 3: SDE-based B estimation
N_B_PER_D = 50           # B values to compute per d
SDE_N_TRIALS = 100       # SDE trials per sigma
SDE_N_SIGMA = 12         # sigma sweep points
SDE_T_ESCAPE = 500.0     # max escape time
SDE_DT = 0.01            # Euler-Maruyama time step
SDE_ESCAPE_R = 2.0       # escape distance threshold


# ================================================================
# UTILITY
# ================================================================

def pprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def save_checkpoint(data, filename):
    """Save results as JSON to SAVE_DIR."""
    path = os.path.join(SAVE_DIR, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    pprint(f"  [checkpoint saved: {path}]")


def load_checkpoint(filename):
    """Load results from SAVE_DIR if exists."""
    path = os.path.join(SAVE_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ================================================================
# SECTION 1: BAUTIN NORMAL FORM — ANALYTICAL FUNCTIONS
# ================================================================
#
# Amplitude equation: dr/dt = mu*r + l1*r^3 - r^5
#
# Effective potential: U(r) = -mu*r^2/2 - l1*r^4/4 + r^6/6
#   (from dr/dt = -dU/dr, U(0) = 0)
#
# Bistable regime: l1 > 0, mu in (-l1^2/4, 0)
#   r = 0: stable FP (eigenvalue mu < 0)
#   r = r_u: unstable limit cycle (separatrix)
#   r = r_s: stable limit cycle
#
# Equilibria (r > 0): r^4 - l1*r^2 - mu = 0 → quadratic in u = r^2
#   u_- = (l1 - sqrt(l1^2 + 4*mu)) / 2   →  r_u = sqrt(u_-)
#   u_+ = (l1 + sqrt(l1^2 + 4*mu)) / 2   →  r_s = sqrt(u_+)
#
# Barrier: DeltaPhi = U(r_u) - U(0) = U(r_u)

def bautin_roots(mu, l1):
    """Find r_unstable, r_stable for Bautin normal form.

    Returns (r_u, r_s) or None if not bistable.
    """
    if l1 <= 0 or mu >= 0:
        return None
    disc = l1**2 + 4 * mu
    if disc <= 0:
        return None
    sq = np.sqrt(disc)
    u_minus = (l1 - sq) / 2
    u_plus = (l1 + sq) / 2
    if u_minus <= 1e-15 or u_plus <= 1e-15:
        return None
    return np.sqrt(u_minus), np.sqrt(u_plus)


def bautin_barrier(mu, l1):
    """Analytical barrier DeltaPhi = U(r_u) - U(0) = U(r_u).

    U(r) = -mu*r^2/2 - l1*r^4/4 + r^6/6
    """
    roots = bautin_roots(mu, l1)
    if roots is None:
        return None
    r_u = roots[0]
    u = r_u**2
    dphi = -mu * u / 2 - l1 * u**2 / 4 + u**3 / 6
    if dphi <= 0:
        return None
    return dphi


def bautin_barrier_numerical(mu, l1):
    """Numerical barrier via direct integration of -f(r) from 0 to r_u.

    DeltaPhi = integral from 0 to r_u of [-f(r)] dr
    where f(r) = mu*r + l1*r^3 - r^5.
    """
    roots = bautin_roots(mu, l1)
    if roots is None:
        return None
    r_u = roots[0]
    result, _ = quad(lambda r: -(mu * r + l1 * r**3 - r**5), 0, r_u)
    return result


def bautin_eigenvalues(mu, l1):
    """Eigenvalues f'(r) at all three equilibria (r=0, r_u, r_s).

    f(r) = mu*r + l1*r^3 - r^5
    f'(r) = mu + 3*l1*r^2 - 5*r^4
    """
    roots = bautin_roots(mu, l1)
    if roots is None:
        return None
    r_u, r_s = roots
    lam_0 = mu                                    # at r=0: stable (< 0)
    lam_u = mu + 3 * l1 * r_u**2 - 5 * r_u**4    # at r_u: unstable (> 0)
    lam_s = mu + 3 * l1 * r_s**2 - 5 * r_s**4    # at r_s: stable (< 0)
    return lam_0, lam_u, lam_s


def bautin_kramers_prefactor(mu, l1):
    """Kramers prefactor for 1D escape: D_Kramers = prefactor * exp(B).

    For 1D overdamped: prefactor = 2*pi*sqrt(|lam_eq| / lam_sad)
    Matches cusp_bridge_derivation.py convention.
    """
    eigs = bautin_eigenvalues(mu, l1)
    if eigs is None:
        return None
    lam_0, lam_u, _ = eigs
    if lam_0 >= 0 or lam_u <= 0:
        return None
    return 2 * np.pi * np.sqrt(abs(lam_0) / lam_u)


# ================================================================
# SECTION 2: EXACT MFPT AND SIGMA* FOR BAUTIN
# ================================================================

def compute_D_mfpt_bautin(mu, l1, sigma, N_grid=N_GRID_MFPT):
    """Exact D = MFPT/tau via Gardiner's integral formula.

    Escape from r=0 (stable FP) to r=r_u (separating unstable cycle).
    Reflecting boundary at r=0.

    Same numerical method as cusp_bridge_derivation.py:compute_D_mfpt_cusp.
    """
    roots = bautin_roots(mu, l1)
    if roots is None:
        return None
    r_u = roots[0]

    lam_eq = mu  # eigenvalue at r=0
    if lam_eq >= 0:
        return None
    tau = 1.0 / abs(lam_eq)

    # Grid: [r_lo, r_hi] with reflecting boundary at r_lo ~ 0
    # Extend left to capture thermal distribution near equilibrium
    spread = max(4 * sigma / np.sqrt(2 * abs(lam_eq)), 0.2 * r_u)
    r_lo = max(1e-10, -spread)  # can't go below 0
    r_lo = 1e-10
    r_hi = r_u + 0.005 * r_u

    rg = np.linspace(r_lo, r_hi, N_grid)
    dr = rg[1] - rg[0]

    # Drift: f(r) = mu*r + l1*r^3 - r^5
    # Potential gradient: U'(r) = -f(r)
    neg_f = -(mu * rg + l1 * rg**3 - rg**5)
    U_raw = np.cumsum(neg_f) * dr

    # Set U(r=0+) = 0 as reference
    i_eq = 0  # r ~ 0 is the equilibrium
    U = U_raw - U_raw[i_eq]

    # Gardiner MFPT integral
    Phi = 2.0 * U / sigma**2
    Phi = np.clip(Phi, -500, 500)

    exp_neg_phi = np.exp(-Phi)
    Ix = np.cumsum(exp_neg_phi) * dr  # integral from r_lo to r
    psi = (2.0 / sigma**2) * np.exp(Phi) * Ix

    i_sad = np.argmin(np.abs(rg - r_u))
    if i_sad <= 1:
        return 1e30

    MFPT = np.trapz(psi[:i_sad + 1], rg[:i_sad + 1])
    return MFPT / tau


def find_sigma_star_bautin(mu, l1, D_target=D_TARGET):
    """Find sigma* where D_mfpt(sigma*) = D_target via bisection."""
    dphi = bautin_barrier(mu, l1)
    if dphi is None or dphi < 1e-12:
        return None

    # Bracket sigma*: at low sigma D is huge, at high sigma D is small
    sig_lo = np.sqrt(2 * dphi / 30)
    sig_hi = np.sqrt(2 * dphi / 0.05)

    def obj(log_s):
        s = np.exp(log_s)
        D = compute_D_mfpt_bautin(mu, l1, s)
        if D is None:
            return 10
        return np.log(max(D, 1e-30)) - np.log(D_target)

    try:
        log_s = brentq(obj, np.log(sig_lo), np.log(sig_hi),
                        xtol=1e-6, maxiter=50)
        return np.exp(log_s)
    except Exception:
        return None


# ================================================================
# SECTION 3: 2D MULTI-CHANNEL MODEL FUNCTIONS
# ================================================================
#
# dx/dt = F(x,y) = a1 - b1*x + c1*y + sum_i [r1_i * x^q1_i / (x^q1_i + k1_i^q1_i)]
# dy/dt = G(x,y) = a2 - b2*y + c2*x + sum_i [r2_i * y^q2_i / (y^q2_i + k2_i^q2_i)]
#
# Parameters: d = 6 + 6*n_channels
# Params tuple: (a1, b1, c1, a2, b2, c2, channels)
#   channels = [(r1_i, q1_i, k1_i, r2_i, q2_i, k2_i), ...]

def eval_2d(x, y, params):
    """Evaluate 2D ODE RHS at scalar (x, y)."""
    a1, b1, c1, a2, b2, c2, channels = params
    dx = a1 - b1 * x + c1 * y
    dy = a2 - b2 * y + c2 * x
    for r1_i, q1_i, k1_i, r2_i, q2_i, k2_i in channels:
        if x > 0:
            xq = x**q1_i
            dx += r1_i * xq / (xq + k1_i**q1_i + 1e-30)
        if y > 0:
            yq = y**q2_i
            dy += r2_i * yq / (yq + k2_i**q2_i + 1e-30)
    return dx, dy


def eval_2d_vec(x_arr, y_arr, params):
    """Vectorized 2D ODE RHS. x_arr, y_arr are 1D arrays of shape (N,)."""
    a1, b1, c1, a2, b2, c2, channels = params
    dx = a1 - b1 * x_arr + c1 * y_arr
    dy = a2 - b2 * y_arr + c2 * x_arr
    for r1_i, q1_i, k1_i, r2_i, q2_i, k2_i in channels:
        xq = np.maximum(x_arr, 1e-30)**q1_i
        dx += r1_i * xq / (xq + k1_i**q1_i + 1e-30)
        yq = np.maximum(y_arr, 1e-30)**q2_i
        dy += r2_i * yq / (yq + k2_i**q2_i + 1e-30)
    return dx, dy


def jacobian_2d(x, y, params):
    """Numerical Jacobian at (x, y)."""
    h = 1e-7
    fx0, fy0 = eval_2d(x, y, params)
    fx1, fy1 = eval_2d(x + h, y, params)
    fx2, fy2 = eval_2d(x, y + h, params)
    return np.array([
        [(fx1 - fx0) / h, (fx2 - fx0) / h],
        [(fy1 - fy0) / h, (fy2 - fy0) / h]
    ])


# ================================================================
# SECTION 4: FP-LC BISTABILITY DETECTION
# ================================================================

def detect_limit_cycle(params, fp, omega):
    """Detect stable limit cycle by ODE integration from distant IC.

    Returns (has_lc, amplitude, period) or (False, None, None).
    """
    period_est = 2 * np.pi / max(abs(omega), 0.1)
    T_sim = min(max(50 * period_est, 100), 500)

    def ode_rhs(t, z):
        x = max(z[0], 1e-6)
        y = max(z[1], 1e-6)
        return list(eval_2d(x, y, params))

    # 1-2 distant ICs (first catches ~90% of LCs)
    ics = [
        [fp[0] + 3.0, fp[1] + 3.0],
        [fp[0] + 5.0, fp[1]],
    ]

    for ic in ics:
        try:
            sol = solve_ivp(ode_rhs, [0, T_sim], ic,
                            method='RK45', rtol=1e-6, atol=1e-6)
            if not sol.success or len(sol.t) < 50:
                continue

            x_sol, y_sol, t_sol = sol.y[0], sol.y[1], sol.t

            # Divergence check
            if np.any(np.abs(x_sol) > 100) or np.any(np.abs(y_sol) > 100):
                continue

            # Take last 30% of trajectory
            n_pts = len(t_sol)
            i0 = int(0.7 * n_pts)
            x_last = x_sol[i0:]
            y_last = y_sol[i0:]
            t_last = t_sol[i0:]

            if len(x_last) < 30:
                continue

            # Check if converged to FP
            dist = np.sqrt((x_last - fp[0])**2 + (y_last - fp[1])**2)
            if np.mean(dist) < 0.05:
                continue  # converged to FP, no coexisting LC

            # Check for oscillation: x should cross its mean multiple times
            x_mean = np.mean(x_last)
            crossings = np.sum(np.abs(np.diff(np.sign(x_last - x_mean))) > 0)
            if crossings < 6:
                continue  # not oscillating enough

            # Amplitude consistency: limit cycle has roughly constant envelope
            amp_cv = np.std(dist) / np.mean(dist) if np.mean(dist) > 0 else 1
            if amp_cv > 0.3:
                continue  # not converged to stable LC

            # Period from zero-crossings of x - x_mean
            cross_times = []
            for j in range(1, len(x_last)):
                if (x_last[j] - x_mean) * (x_last[j - 1] - x_mean) < 0:
                    frac = (x_mean - x_last[j - 1]) / (x_last[j] - x_last[j - 1] + 1e-30)
                    cross_times.append(t_last[j - 1] + frac * (t_last[j] - t_last[j - 1]))

            if len(cross_times) < 4:
                continue

            # Full periods: every other crossing
            periods = np.diff(cross_times)[::2]
            if len(periods) < 2:
                continue

            period_cv = np.std(periods) / np.mean(periods) if np.mean(periods) > 0 else 1
            if period_cv < 0.05:
                return True, float(np.mean(dist)), float(np.mean(periods))

        except Exception:
            continue

    return False, None, None


def batch_detect_fp_lc(n_ch, N_samples, batch_size=BATCH_SIZE_P2):
    """Detect FP-LC bistability in N_samples random 2D systems.

    Pipeline per batch:
    1. Vectorized grid evaluation of |F|+|G| for all systems
    2. Threshold filter for candidate equilibria
    3. Newton refinement (scalar, per candidate)
    4. Jacobian eigenvalue classification
    5. ODE integration for limit cycle detection (stable spirals only)

    Returns: (n_bistable, configs_list)
    """
    # 2D grid for equilibrium search
    xs = np.linspace(0.1, 5.0, GRID_N_2D)
    ys = np.linspace(0.1, 5.0, GRID_N_2D)
    xg, yg = np.meshgrid(xs, ys)
    xf = xg.flatten()   # (M,) where M = GRID_N_2D^2
    yf = yg.flatten()
    M = len(xf)

    n_bistable = 0
    configs = []
    n_processed = 0
    t_start = time.time()

    while n_processed < N_samples:
        bs = min(batch_size, N_samples - n_processed)

        # --- Sample parameters ---
        a1 = np.random.uniform(0.05, 2.0, bs)
        b1 = np.random.uniform(0.2, 3.0, bs)
        c1 = np.random.uniform(-1.5, 1.5, bs)
        a2 = np.random.uniform(0.05, 2.0, bs)
        b2 = np.random.uniform(0.2, 3.0, bs)
        c2 = np.random.uniform(-1.5, 1.5, bs)

        ch_r1 = np.random.uniform(0.1, 2.0, (bs, n_ch))
        ch_q1 = np.random.uniform(2.0, 10.0, (bs, n_ch))
        ch_k1 = np.random.uniform(0.3, 2.0, (bs, n_ch))
        ch_r2 = np.random.uniform(0.1, 2.0, (bs, n_ch))
        ch_q2 = np.random.uniform(2.0, 10.0, (bs, n_ch))
        ch_k2 = np.random.uniform(0.3, 2.0, (bs, n_ch))

        # --- Vectorized grid evaluation: F, G at all grid points ---
        # F shape: (bs, M), G shape: (bs, M)
        F = (a1[:, None] - b1[:, None] * xf[None, :]
             + c1[:, None] * yf[None, :])
        G = (a2[:, None] - b2[:, None] * yf[None, :]
             + c2[:, None] * xf[None, :])

        for ch in range(n_ch):
            xq = xf[None, :] ** ch_q1[:, ch:ch + 1]   # (bs, M)
            kq_x = ch_k1[:, ch:ch + 1] ** ch_q1[:, ch:ch + 1]
            F = F + ch_r1[:, ch:ch + 1] * xq / (xq + kq_x + 1e-30)

            yq = yf[None, :] ** ch_q2[:, ch:ch + 1]
            kq_y = ch_k2[:, ch:ch + 1] ** ch_q2[:, ch:ch + 1]
            G = G + ch_r2[:, ch:ch + 1] * yq / (yq + kq_y + 1e-30)

        # --- Threshold filter: |F| + |G| < threshold ---
        FG_sum = np.abs(F) + np.abs(G)
        min_vals = np.min(FG_sum, axis=1)  # (bs,)
        cand_mask = min_vals < EQ_THRESHOLD
        cand_indices = np.where(cand_mask)[0]

        # Free F, G but KEEP FG_sum for candidate cell lookup
        del F, G

        # --- Per-candidate: Newton refinement + classification ---
        for idx in cand_indices:
            # Build params tuple
            channels = []
            for ch in range(n_ch):
                channels.append((
                    ch_r1[idx, ch], ch_q1[idx, ch], ch_k1[idx, ch],
                    ch_r2[idx, ch], ch_q2[idx, ch], ch_k2[idx, ch]
                ))
            params = (a1[idx], b1[idx], c1[idx],
                      a2[idx], b2[idx], c2[idx], channels)

            # Use ALREADY-COMPUTED FG_sum row (no recomputation!)
            sorted_cells = np.argsort(FG_sum[idx])

            refined_eqs = []
            for cell_i in sorted_cells[:5]:
                x0, y0 = xf[cell_i], yf[cell_i]
                # Skip if too close to already-refined point
                too_close = False
                for xr, yr in refined_eqs:
                    if abs(x0 - xr) < 0.4 and abs(y0 - yr) < 0.4:
                        too_close = True
                        break
                if too_close:
                    continue

                # Newton refinement
                try:
                    def rhs(z):
                        dx, dy = eval_2d(z[0], z[1], params)
                        return [dx, dy]
                    sol = fsolve(rhs, [x0, y0], full_output=True)
                    eq = sol[0]
                    if eq[0] < 0.001 or eq[1] < 0.001 or eq[0] > 10 or eq[1] > 10:
                        continue
                    dx, dy = eval_2d(eq[0], eq[1], params)
                    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                        continue
                except Exception:
                    continue

                refined_eqs.append((eq[0], eq[1]))

                # Jacobian and eigenvalue classification
                J = jacobian_2d(eq[0], eq[1], params)
                eigs = np.linalg.eigvals(J)
                re_parts = np.real(eigs)
                im_parts = np.imag(eigs)

                # Filter: stable spiral NEAR Hopf (tight filter)
                if not (np.all(re_parts < 0) and np.any(np.abs(im_parts) > 0.1)):
                    continue
                if np.max(re_parts) < HOPF_RE_THRESHOLD:
                    continue  # too far from Hopf boundary

                omega = np.max(np.abs(im_parts))

                # ODE integration for limit cycle detection
                has_lc, lc_amp, lc_period = detect_limit_cycle(
                    params, eq, omega)

                if has_lc:
                    n_bistable += 1
                    if len(configs) < N_B_PER_D * 2:
                        configs.append({
                            'fp': [float(eq[0]), float(eq[1])],
                            'eigs_re': [float(r) for r in re_parts],
                            'eigs_im': [float(i) for i in im_parts],
                            'omega': float(omega),
                            'lc_amplitude': float(lc_amp),
                            'lc_period': float(lc_period),
                            'params': {
                                'a1': float(a1[idx]), 'b1': float(b1[idx]),
                                'c1': float(c1[idx]), 'a2': float(a2[idx]),
                                'b2': float(b2[idx]), 'c2': float(c2[idx]),
                                'channels': [list(map(float, ch))
                                             for ch in channels],
                            },
                        })
                    break  # found bistability for this system, move on

        del FG_sum

        n_processed += bs
        elapsed = time.time() - t_start
        rate = n_processed / elapsed if elapsed > 0 else 1
        eta = (N_samples - n_processed) / rate if rate > 0 else 0
        pprint(f"    {n_processed:,}/{N_samples:,} | "
               f"{n_bistable} FP-LC found | "
               f"P={n_bistable / n_processed:.2e} | "
               f"ETA: {eta:.0f}s")

    return n_bistable, configs


# ================================================================
# SECTION 5: SDE-BASED B ESTIMATION
# ================================================================

def sde_escape_batch(params, fp, sigma, n_trials=SDE_N_TRIALS,
                     T_escape=SDE_T_ESCAPE, dt=SDE_DT):
    """Vectorized SDE escape simulation: all trials in parallel.

    Returns (escape_times, escaped_mask).
    escape_times[i] = escape time for trial i (inf if not escaped).
    """
    n_steps = int(T_escape / dt)
    sqrt_dt = np.sqrt(dt)

    x = np.full(n_trials, fp[0])
    y = np.full(n_trials, fp[1])
    escaped = np.zeros(n_trials, dtype=bool)
    escape_times = np.full(n_trials, np.inf)

    for step in range(n_steps):
        active = ~escaped
        n_active = np.sum(active)
        if n_active == 0:
            break

        xa, ya = x[active], y[active]
        dxa, dya = eval_2d_vec(xa, ya, params)

        x[active] = np.maximum(xa + dxa * dt + sigma * sqrt_dt * np.random.randn(n_active), 1e-6)
        y[active] = np.maximum(ya + dya * dt + sigma * sqrt_dt * np.random.randn(n_active), 1e-6)

        dist = np.sqrt((x[active] - fp[0])**2 + (y[active] - fp[1])**2)
        newly_escaped = dist > SDE_ESCAPE_R

        esc_idx = np.where(active)[0][newly_escaped]
        escaped[esc_idx] = True
        escape_times[esc_idx] = (step + 1) * dt

    return escape_times, escaped


def estimate_B_2d(params, fp):
    """Estimate B for a 2D FP-LC bistable system via SDE escape.

    1. Sweep sigma, measure MFPT from vectorized SDE escapes
    2. Find sigma* where D = D_TARGET
    3. Fit ln(D) vs 1/sigma^2 for DeltaPhi (Kramers slope)
    4. B = 2*DeltaPhi/sigma*^2

    Returns dict with B, sigma*, DeltaPhi, R^2 or None on failure.
    """
    # Eigenvalue at FP for tau
    J = jacobian_2d(fp[0], fp[1], params)
    eigs = np.linalg.eigvals(J)
    lam_max_re = np.max(np.real(eigs))
    if lam_max_re >= 0:
        return None
    tau = 1.0 / abs(lam_max_re)

    sigmas = np.geomspace(0.01, 2.0, SDE_N_SIGMA)
    D_vals = np.zeros(SDE_N_SIGMA)

    for i_sig, sigma in enumerate(sigmas):
        escape_times, escaped = sde_escape_batch(params, fp, sigma)
        n_escaped = np.sum(escaped)

        if n_escaped > 0:
            # MLE for exponential MFPT: total observation time / n events
            total_obs = np.sum(np.minimum(escape_times, SDE_T_ESCAPE))
            mfpt = total_obs / n_escaped
        else:
            mfpt = SDE_T_ESCAPE * 10  # lower bound

        D_vals[i_sig] = mfpt / tau

    # Find sigma* by interpolation: D decreases with sigma
    log_D = np.log(np.clip(D_vals, 1e-10, 1e30))
    target = np.log(D_TARGET)

    sigma_star = None
    for i in range(len(sigmas) - 1):
        if (log_D[i] - target) * (log_D[i + 1] - target) < 0:
            frac = (target - log_D[i]) / (log_D[i + 1] - log_D[i])
            sigma_star = sigmas[i] * (sigmas[i + 1] / sigmas[i])**frac
            break

    if sigma_star is None:
        return None

    # Fit ln(D) vs 1/sigma^2 for DeltaPhi (Kramers: ln(D) ~ 2*DeltaPhi/sigma^2)
    # Use data where D is in a reasonable barrier regime
    valid = (D_vals > 5) & (D_vals < 1e8) & np.isfinite(D_vals)
    if np.sum(valid) < 3:
        # Fallback: use B ≈ ln(D_TARGET) as rough estimate
        return {
            'sigma_star': float(sigma_star),
            'B_approx': float(np.log(D_TARGET)),
            'method': 'fallback',
        }

    inv_sig2 = 1.0 / sigmas[valid]**2
    log_D_v = np.log(D_vals[valid])
    slope, intercept, r_val, _, _ = linregress(inv_sig2, log_D_v)

    if slope <= 0:
        return {
            'sigma_star': float(sigma_star),
            'B_approx': float(np.log(D_TARGET)),
            'method': 'fallback_neg_slope',
        }

    DeltaPhi = slope / 2
    B = 2 * DeltaPhi / sigma_star**2

    return {
        'sigma_star': float(sigma_star),
        'DeltaPhi': float(DeltaPhi),
        'B': float(B),
        'R2_fit': float(r_val**2),
        'method': 'kramers_fit',
    }


# ================================================================
# MAIN EXECUTION
# ================================================================

if __name__ == '__main__':

    # ==============================================================
    # PART 1: B FROM BAUTIN NORMAL FORM
    # ==============================================================
    pprint("=" * 72)
    pprint("PART 1: B INVARIANCE IN THE BAUTIN NORMAL FORM")
    pprint("=" * 72)
    pprint()
    pprint("Amplitude equation: dr/dt = mu*r + l1*r^3 - r^5")
    pprint("Potential: U(r) = -mu*r^2/2 - l1*r^4/4 + r^6/6")
    pprint(f"Sampling {N_NORMAL_FORM} random (mu, l1) in bistable regime")
    pprint(f"D_target = {D_TARGET}, habitable zone B in [{B_HAB_LO}, {B_HAB_HI}]")

    t0 = time.time()

    # --- Step 1A: Verify analytical barrier ---
    pprint(f"\nSTEP 1A: Verify analytical barrier")
    pprint(f"{'l1':>6s} {'mu':>10s} {'DPhi(anal)':>12s} {'DPhi(num)':>12s} {'RelErr':>10s}")
    pprint("-" * 54)

    test_cases = [(0.5, -0.04), (1.0, -0.1), (1.0, -0.2),
                  (1.5, -0.3), (2.0, -0.5), (0.3, -0.01)]
    for l1_t, mu_t in test_cases:
        dphi_a = bautin_barrier(mu_t, l1_t)
        dphi_n = bautin_barrier_numerical(mu_t, l1_t)
        if dphi_a is not None and dphi_n is not None and dphi_a > 0:
            err = abs(dphi_a - dphi_n) / dphi_a
            pprint(f"{l1_t:6.2f} {mu_t:10.4f} {dphi_a:12.8f} {dphi_n:12.8f} {err:10.2e}")
        else:
            pprint(f"{l1_t:6.2f} {mu_t:10.4f} {'outside':>12s}")

    # --- Step 1B: Verify eigenvalue structure ---
    pprint(f"\nSTEP 1B: Eigenvalue structure (l1=1.0, mu=-0.1)")
    roots = bautin_roots(-0.1, 1.0)
    eigs = bautin_eigenvalues(-0.1, 1.0)
    pprint(f"  r_u = {roots[0]:.6f}, r_s = {roots[1]:.6f}")
    pprint(f"  lam(r=0)  = {eigs[0]:.4f} (stable FP)")
    pprint(f"  lam(r_u)  = {eigs[1]:.4f} (unstable cycle)")
    pprint(f"  lam(r_s)  = {eigs[2]:.4f} (stable cycle)")
    pprint(f"  Barrier   = {bautin_barrier(-0.1, 1.0):.6f}")

    # --- Step 1C: B distribution ---
    pprint(f"\nSTEP 1C: B distribution (n={N_NORMAL_FORM})")

    ckpt1 = load_checkpoint('part1_results.json')
    if ckpt1 and len(ckpt1.get('B_vals', [])) >= N_NORMAL_FORM:
        pprint(f"  Loaded from checkpoint ({len(ckpt1['B_vals'])} samples)")
        B_p1 = np.array(ckpt1['B_vals'])
        dphi_p1 = np.array(ckpt1['dphi_vals'])
        sigma_p1 = np.array(ckpt1['sigma_vals'])
        K_p1 = np.array(ckpt1['K_vals'])
        mu_p1 = np.array(ckpt1['mu_vals'])
        l1_p1 = np.array(ckpt1['l1_vals'])
    else:
        B_p1, dphi_p1, sigma_p1 = [], [], []
        K_p1, mu_p1, l1_p1 = [], [], []
        n_done, n_attempts = 0, 0

        while n_done < N_NORMAL_FORM:
            n_attempts += 1
            l1 = np.random.uniform(0.1, 2.0)
            mu_sn = -l1**2 / 4
            mu = np.random.uniform(mu_sn + 0.01 * abs(mu_sn), -0.01)

            roots = bautin_roots(mu, l1)
            if roots is None:
                continue
            dphi = bautin_barrier(mu, l1)
            if dphi is None or dphi < 1e-8:
                continue
            ss = find_sigma_star_bautin(mu, l1)
            if ss is None or ss <= 0:
                continue

            B = 2 * dphi / ss**2
            pf = bautin_kramers_prefactor(mu, l1)
            K = None
            if pf is not None and pf > 0:
                D_kr = pf * np.exp(B)
                if D_kr > 0:
                    K = D_TARGET / D_kr

            B_p1.append(B)
            dphi_p1.append(dphi)
            sigma_p1.append(ss)
            K_p1.append(K if K else 0)
            mu_p1.append(mu)
            l1_p1.append(l1)
            n_done += 1

            if n_done % 50 == 0:
                pprint(f"  {n_done}/{N_NORMAL_FORM} done "
                       f"(B_mean={np.mean(B_p1):.3f}, "
                       f"attempts={n_attempts})")

        B_p1 = np.array(B_p1)
        dphi_p1 = np.array(dphi_p1)
        sigma_p1 = np.array(sigma_p1)
        K_p1 = np.array(K_p1)
        mu_p1 = np.array(mu_p1)
        l1_p1 = np.array(l1_p1)

        save_checkpoint({
            'B_vals': B_p1.tolist(), 'dphi_vals': dphi_p1.tolist(),
            'sigma_vals': sigma_p1.tolist(), 'K_vals': K_p1.tolist(),
            'mu_vals': mu_p1.tolist(), 'l1_vals': l1_p1.tolist(),
        }, 'part1_results.json')

    # --- Part 1 results ---
    B_mean_p1 = np.mean(B_p1)
    B_std_p1 = np.std(B_p1)
    B_cv_p1 = B_std_p1 / B_mean_p1 * 100
    p_hab_p1 = np.mean((B_p1 >= B_HAB_LO) & (B_p1 <= B_HAB_HI))
    K_valid = K_p1[K_p1 > 0]

    pprint(f"\n{'='*72}")
    pprint(f"PART 1 RESULTS: B IN BAUTIN NORMAL FORM (n={len(B_p1)})")
    pprint(f"{'='*72}")
    pprint(f"\n  B DISTRIBUTION:")
    pprint(f"    Mean:   {B_mean_p1:.4f}")
    pprint(f"    Median: {np.median(B_p1):.4f}")
    pprint(f"    Std:    {B_std_p1:.4f}")
    pprint(f"    CV:     {B_cv_p1:.1f}%")
    pprint(f"    Range:  [{np.min(B_p1):.3f}, {np.max(B_p1):.3f}]")
    pprint(f"    IQR:    [{np.percentile(B_p1, 25):.3f}, "
           f"{np.percentile(B_p1, 75):.3f}]")
    pprint(f"    P(B in [1.8, 6.0]): {p_hab_p1:.4f}")

    pprint(f"\n  VARIATION ANALYSIS:")
    pprint(f"    DeltaPhi range: [{np.min(dphi_p1):.6f}, "
           f"{np.max(dphi_p1):.4f}]  "
           f"({np.max(dphi_p1)/np.min(dphi_p1):.0f}x)")
    pprint(f"    sigma* range:   [{np.min(sigma_p1):.6f}, "
           f"{np.max(sigma_p1):.4f}]  "
           f"({np.max(sigma_p1)/np.min(sigma_p1):.0f}x)")
    pprint(f"    B range:        [{np.min(B_p1):.3f}, "
           f"{np.max(B_p1):.3f}]  "
           f"({np.max(B_p1)/np.min(B_p1):.1f}x)")

    if len(K_valid) > 10:
        pprint(f"\n  KRAMERS K VALUES (D_exact / D_Kramers):")
        pprint(f"    Mean:  {np.mean(K_valid):.4f}")
        pprint(f"    Range: [{np.min(K_valid):.3f}, {np.max(K_valid):.3f}]")

    pprint(f"\n  COMPARISON TO CUSP:")
    pprint(f"    B_cusp   = 2.979 (CV = 3.7%, n=300)")
    pprint(f"    B_Bautin = {B_mean_p1:.3f} (CV = {B_cv_p1:.1f}%, n={len(B_p1)})")

    in_hab_p1 = B_HAB_LO <= B_mean_p1 <= B_HAB_HI
    if in_hab_p1:
        pprint(f"\n  *** B_Bautin IS IN THE HABITABLE ZONE [1.8, 6.0] ***")
    else:
        pprint(f"\n  *** B_Bautin = {B_mean_p1:.3f} IS OUTSIDE THE HABITABLE ZONE ***")

    t1 = time.time()
    pprint(f"\n  Part 1 time: {t1-t0:.1f}s")

    # ==============================================================
    # PART 2: P(FP-LC BISTABLE | d) DIMENSIONAL SCALING
    # ==============================================================
    pprint(f"\n{'='*72}")
    pprint(f"PART 2: P(FP-LC BISTABLE | d) DIMENSIONAL SCALING")
    pprint(f"{'='*72}")
    pprint(f"\nModel: 2D multi-channel Hill-function ODE")
    pprint(f"  dx/dt = a1 - b1*x + c1*y + sum [r1_i * Hill(x)]")
    pprint(f"  dy/dt = a2 - b2*y + c2*x + sum [r2_i * Hill(y)]")
    pprint(f"  d = 6 + 6*n_channels")

    # Load checkpoint
    ckpt2 = load_checkpoint('part2_results.json')
    p2_done = {}
    if ckpt2:
        for r in ckpt2.get('results', []):
            p2_done[r['d']] = r
        pprint(f"  Checkpoint: d values done = {sorted(p2_done.keys())}")

    p2_results = list(p2_done.values())

    for n_ch, d, N_samples in SPECS_P2:
        if d in p2_done:
            pprint(f"\n  d={d} already done (from checkpoint)")
            continue

        pprint(f"\n  --- d={d} (n_channels={n_ch}, N={N_samples:,}) ---")
        t_d = time.time()

        n_bist, configs_found = batch_detect_fp_lc(n_ch, N_samples)

        P_est = n_bist / N_samples if N_samples > 0 else 0
        result_entry = {
            'd': d, 'n_channels': n_ch, 'N_samples': N_samples,
            'n_bistable': n_bist, 'P': P_est,
            'time_s': time.time() - t_d,
        }
        p2_results.append(result_entry)
        p2_done[d] = result_entry

        pprint(f"\n    RESULT: P(FP-LC | d={d}) = {P_est:.4e} "
               f"({n_bist}/{N_samples:,}) "
               f"[{time.time()-t_d:.0f}s]")

        # Save checkpoint
        save_checkpoint({'results': p2_results}, 'part2_results.json')

        # Save configs for Part 3
        if configs_found:
            save_checkpoint({'configs': configs_found},
                            f'part2_configs_d{d}.json')

    # --- Part 2 summary ---
    pprint(f"\n{'='*72}")
    pprint(f"PART 2 RESULTS SUMMARY")
    pprint(f"{'='*72}")
    pprint(f"\n  {'d':>4s} {'n_ch':>5s} {'N':>12s} "
           f"{'n_bist':>8s} {'P':>12s} {'ln(1/P)':>10s}")
    pprint(f"  " + "-" * 60)

    d_vals_p2, lnP_p2 = [], []
    for r in sorted(p2_results, key=lambda x: x['d']):
        P = r['P']
        lnP_inv = -np.log(P) if P > 0 else float('inf')
        pprint(f"  {r['d']:4d} {r['n_channels']:5d} "
               f"{r['N_samples']:12,d} {r['n_bistable']:8d} "
               f"{P:12.4e} {lnP_inv:10.2f}")
        if P > 0:
            d_vals_p2.append(r['d'])
            lnP_p2.append(lnP_inv)

    gamma_bautin = None
    R2_gamma = None
    if len(d_vals_p2) >= 3:
        d_arr = np.array(d_vals_p2)
        y_arr = np.array(lnP_p2)
        slope, intercept, r_val, _, _ = linregress(d_arr, y_arr)
        R2_gamma = r_val**2
        gamma_bautin = slope

        pprint(f"\n  LINEAR FIT: ln(1/P) = {intercept:.3f} + {slope:.4f}*d")
        pprint(f"    gamma_Bautin = {slope:.4f} per parameter")
        pprint(f"    R^2 = {R2_gamma:.4f}")
        pprint(f"\n  COMPARISON:")
        pprint(f"    gamma_cusp   = 0.197 per parameter")
        pprint(f"    gamma_Bautin = {slope:.4f} per parameter")
        pprint(f"    Ratio: {slope/0.197:.2f}")

        # Extrapolation
        target_lnS = 13.0 * np.log(10)  # ln(10^13)
        if slope > 0:
            d_pred = (target_lnS - intercept) / slope
            pprint(f"\n  PREDICTION: d for S=10^13: {d_pred:.0f} parameters")

    # ==============================================================
    # PART 3: B FOR MULTI-CHANNEL FP-LC CONFIGS
    # ==============================================================
    pprint(f"\n{'='*72}")
    pprint(f"PART 3: B COMPUTATION FOR MULTI-CHANNEL FP-LC CONFIGS")
    pprint(f"{'='*72}")
    pprint(f"\nMethod: Vectorized SDE escape, sigma sweep, Kramers slope fit")
    pprint(f"n_trials={SDE_N_TRIALS}, n_sigma={SDE_N_SIGMA}, "
           f"T_escape={SDE_T_ESCAPE}")

    # Load existing Part 3 results
    ckpt3 = load_checkpoint('part3_results.json')
    B_results_p3 = ckpt3.get('B_results', []) if ckpt3 else []
    d_done_p3 = {r['d'] for r in B_results_p3}

    for r in sorted(p2_results, key=lambda x: x['d']):
        d = r['d']
        if d in d_done_p3:
            pprint(f"\n  d={d}: already done (from checkpoint)")
            continue

        cfg_data = load_checkpoint(f'part2_configs_d{d}.json')
        if cfg_data is None or len(cfg_data.get('configs', [])) == 0:
            pprint(f"\n  d={d}: no configs available, skipping")
            continue

        configs = cfg_data['configs']
        n_to_process = min(len(configs), N_B_PER_D)
        pprint(f"\n  d={d}: computing B for {n_to_process} configs")
        t_d3 = time.time()

        B_vals_d = []
        for i, cfg in enumerate(configs[:n_to_process]):
            p = cfg['params']
            channels = [tuple(ch) for ch in p['channels']]
            params = (p['a1'], p['b1'], p['c1'],
                      p['a2'], p['b2'], p['c2'], channels)
            fp = np.array(cfg['fp'])

            result = estimate_B_2d(params, fp)

            if result is not None and 'B' in result:
                B_vals_d.append(result['B'])
                pprint(f"    config {i+1}/{n_to_process}: "
                       f"B={result['B']:.3f}, "
                       f"sigma*={result['sigma_star']:.4f}, "
                       f"R2={result['R2_fit']:.3f}")
            elif result is not None and 'B_approx' in result:
                pprint(f"    config {i+1}/{n_to_process}: "
                       f"B_approx={result['B_approx']:.3f} "
                       f"({result['method']})")
            else:
                pprint(f"    config {i+1}/{n_to_process}: FAILED")

        if B_vals_d:
            B_d = np.array(B_vals_d)
            entry = {
                'd': d,
                'B_vals': B_d.tolist(),
                'B_mean': float(np.mean(B_d)),
                'B_std': float(np.std(B_d)),
                'B_cv': float(np.std(B_d) / np.mean(B_d) * 100),
                'n': len(B_d),
                'time_s': time.time() - t_d3,
            }
            B_results_p3.append(entry)

            in_hab = B_HAB_LO <= entry['B_mean'] <= B_HAB_HI
            pprint(f"    d={d}: B_mean={entry['B_mean']:.3f}, "
                   f"B_CV={entry['B_cv']:.1f}%, n={entry['n']}, "
                   f"hab={'YES' if in_hab else 'NO'}")

            save_checkpoint({'B_results': B_results_p3},
                            'part3_results.json')

    # ==============================================================
    # FULL SUMMARY
    # ==============================================================
    pprint(f"\n{'='*72}")
    pprint(f"FULL SUMMARY: BAUTIN BRIDGE (Study 17)")
    pprint(f"{'='*72}")

    pprint(f"""
1. B IN BAUTIN NORMAL FORM (Part 1)
   Amplitude equation: dr/dt = mu*r + l1*r^3 - r^5
   B_Bautin  = {B_mean_p1:.3f} (CV = {B_cv_p1:.1f}%, n={len(B_p1)})
   B_cusp    = 2.979 (CV = 3.7%, n=300)
   In habitable zone [1.8, 6.0]: {'YES' if in_hab_p1 else 'NO'}
   Barrier varies {np.max(dphi_p1)/np.min(dphi_p1):.0f}x while B varies {np.max(B_p1)/np.min(B_p1):.1f}x""")

    if len(K_valid) > 10:
        pprint(f"   Kramers K: {np.mean(K_valid):.3f} "
               f"(range [{np.min(K_valid):.3f}, {np.max(K_valid):.3f}])")

    if gamma_bautin is not None:
        pprint(f"""
2. DIMENSIONAL SCALING (Part 2)
   P(FP-LC bistable | d) at d = {', '.join(str(d) for d in d_vals_p2)}
   gamma_Bautin = {gamma_bautin:.4f} per parameter
   gamma_cusp   = 0.197 per parameter
   Ratio: {gamma_bautin/0.197:.2f}
   R^2 = {R2_gamma:.4f}""")
    else:
        pprint(f"\n2. DIMENSIONAL SCALING (Part 2): insufficient data for fit")

    if B_results_p3:
        pprint(f"\n3. B FOR MULTI-CHANNEL CONFIGS (Part 3)")
        for br in B_results_p3:
            in_hab = B_HAB_LO <= br['B_mean'] <= B_HAB_HI
            pprint(f"   d={br['d']:2d}: B = {br['B_mean']:.3f} "
                   f"(CV = {br['B_cv']:.1f}%, n={br['n']}) "
                   f"hab={'YES' if in_hab else 'NO'}")
    else:
        pprint(f"\n3. B FOR MULTI-CHANNEL CONFIGS (Part 3): no data")

    pprint(f"\n4. INTERPRETATION")
    if in_hab_p1:
        pprint(f"   B invariance EXTENDS to FP-LC (Bautin) bistability.")
        pprint(f"   The habitable zone [1.8, 6.0] is NOT cusp-specific.")
    else:
        pprint(f"   B_Bautin = {B_mean_p1:.3f} is outside [1.8, 6.0].")
        pprint(f"   The habitable zone may be cusp-specific.")

    if gamma_bautin is not None and gamma_bautin > 0:
        ratio = gamma_bautin / 0.197
        if 0.5 < ratio < 2.0:
            pprint(f"   gamma_Bautin / gamma_cusp = {ratio:.2f} "
                   f"-- consistent with universal search scaling.")
        else:
            pprint(f"   gamma_Bautin / gamma_cusp = {ratio:.2f} "
                   f"-- search scaling differs by bifurcation type.")

    if in_hab_p1 and gamma_bautin is not None and 0.5 < gamma_bautin / 0.197 < 2.0:
        pprint(f"\n   CONCLUSION: Both persistence (B) and search (gamma)")
        pprint(f"   extend beyond cusp bistability. The framework is")
        pprint(f"   genuinely universal across bistability types.")
    elif not in_hab_p1:
        pprint(f"\n   CONCLUSION: B outside habitable zone for Bautin.")
        pprint(f"   The persistence side of the bridge may be")
        pprint(f"   bifurcation-type-dependent.")

    pprint(f"\n{'='*72}")
    pprint(f"END OF STUDY 17: BAUTIN BRIDGE")
    pprint(f"{'='*72}")
