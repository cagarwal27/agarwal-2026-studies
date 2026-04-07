"""
Gardner-Collins Toggle Switch: Kramers Escape Test
===================================================
Tests whether D = exp(ΔV/σ²) holds for an endogenous bistable system
where all parameters are known precisely.

Model: du/dt = α/(1+v^n) - u,  dv/dt = α/(1+u^n) - v
Stochastic version: Gillespie SSA with system size Ω

Computes:
  1. D_observed  = MFPT / τ_relax  (directly from simulation)
  2. D_z         = exp(z²/2)        (Round 2 Gaussian formula)
  3. D_barrier   = exp(ΔV_actual)   (actual quasi-potential barrier from simulation)

The prediction: D_barrier ≈ D_observed. D_z will overshoot massively.
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import solve_lyapunov
import time

# ============================================================
# PART 1: Deterministic analysis
# ============================================================

def toggle_rhs(u, v, alpha, n):
    """Right-hand side of the toggle switch ODE."""
    du = alpha / (1 + v**n) - u
    dv = alpha / (1 + u**n) - v
    return du, dv

def find_fixed_points(alpha, n):
    """Find all fixed points of the symmetric toggle switch."""
    # Saddle point: on diagonal u = v = x where x = alpha/(1+x^n)
    def saddle_eq(x):
        return x - alpha / (1 + x**n)

    x_saddle = fsolve(saddle_eq, alpha**(1/(n+1)))[0]

    # Stable points: high-u/low-v and low-u/high-v
    def fp_system(uv):
        u, v = uv
        return [alpha/(1+v**n) - u, alpha/(1+u**n) - v]

    # High-u state
    u_hi, v_lo = fsolve(fp_system, [alpha - 0.1, 0.1])
    # Low-u state
    u_lo, v_hi = fsolve(fp_system, [0.1, alpha - 0.1])

    return (u_hi, v_lo), (u_lo, v_hi), (x_saddle, x_saddle)

def jacobian(u, v, alpha, n):
    """Jacobian of the toggle switch at (u, v)."""
    J = np.array([
        [-1, -alpha * n * v**(n-1) / (1 + v**n)**2],
        [-alpha * n * u**(n-1) / (1 + u**n)**2, -1]
    ])
    return J

def lna_covariance(u, v, alpha, n, Omega):
    """
    Linear Noise Approximation covariance matrix.
    In the system-size expansion, fluctuations satisfy:
    J @ C + C @ J^T + D/Omega = 0
    where D is the diffusion matrix from reaction rates.
    """
    J = jacobian(u, v, alpha, n)

    # Diffusion matrix: for each species, sum of production + degradation rates
    # Production of u: alpha/(1+v^n), degradation of u: u
    # These are independent reactions, so D is diagonal
    prod_u = alpha / (1 + v**n)
    prod_v = alpha / (1 + u**n)
    D = np.diag([prod_u + u, prod_v + v]) / Omega

    # Solve Lyapunov equation: J C + C J^T = -D
    C = solve_lyapunov(J, -D)
    return C

# ============================================================
# PART 2: Gillespie SSA
# ============================================================

def gillespie_toggle(alpha, n, Omega, T_max, seed=42):
    """
    Gillespie SSA for the toggle switch.
    State: (n_u, n_v) = molecule counts = Omega * (u, v)

    Reactions:
    1. ∅ → U  rate = Omega * alpha / (1 + (n_v/Omega)^n)
    2. U → ∅  rate = n_u
    3. ∅ → V  rate = Omega * alpha / (1 + (n_u/Omega)^n)
    4. V → ∅  rate = n_v
    """
    rng = np.random.default_rng(seed)

    # Start at the high-u stable state
    fp_hi, _, _ = find_fixed_points(alpha, n)
    n_u = int(round(Omega * fp_hi[0]))
    n_v = int(round(Omega * fp_hi[1]))

    # Storage for trajectory (sample at intervals)
    sample_interval = 1.0  # sample every 1 time unit
    next_sample = sample_interval
    times = [0.0]
    traj_u = [n_u]
    traj_v = [n_v]

    t = 0.0
    n_reactions = 0

    while t < T_max:
        # Propensities
        cu = n_u / Omega  # concentration u
        cv = n_v / Omega  # concentration v

        a1 = Omega * alpha / (1 + cv**n) if cv >= 0 else Omega * alpha  # production U
        a2 = float(n_u)                                                   # degradation U
        a3 = Omega * alpha / (1 + cu**n) if cu >= 0 else Omega * alpha  # production V
        a4 = float(n_v)                                                   # degradation V

        a_total = a1 + a2 + a3 + a4
        if a_total <= 0:
            break

        # Time to next reaction
        dt = rng.exponential(1.0 / a_total)
        t += dt

        # Which reaction
        r = rng.random() * a_total
        if r < a1:
            n_u += 1
        elif r < a1 + a2:
            n_u = max(0, n_u - 1)
        elif r < a1 + a2 + a3:
            n_v += 1
        else:
            n_v = max(0, n_v - 1)

        n_reactions += 1

        # Sample
        while t >= next_sample and next_sample <= T_max:
            times.append(next_sample)
            traj_u.append(n_u)
            traj_v.append(n_v)
            next_sample += sample_interval

    return np.array(times), np.array(traj_u), np.array(traj_v), n_reactions

def analyze_switching(times, traj_u, traj_v, Omega, alpha, n):
    """
    Analyze switching events from trajectory.
    Classify state as 'high-u' or 'high-v' based on which protein dominates.
    """
    fp_hi, fp_lo, fp_saddle = find_fixed_points(alpha, n)

    # Threshold: saddle point
    threshold_u = Omega * fp_saddle[0]

    # State classification: high-u if n_u > threshold, high-v if n_u < threshold
    states = np.where(traj_u > threshold_u, 1, 0)

    # Find switching events
    switches = np.diff(states)
    switch_times = times[1:][switches != 0]
    switch_dirs = switches[switches != 0]

    # Compute residence times in each state
    if len(switch_times) < 2:
        return None, None, None, None, len(switch_times)

    residence_times_hi = []  # time in high-u state
    residence_times_lo = []  # time in high-v state

    for i in range(len(switch_times) - 1):
        dt_res = switch_times[i+1] - switch_times[i]
        if switch_dirs[i] == -1:  # just entered low-u state
            residence_times_lo.append(dt_res)
        else:  # just entered high-u state
            residence_times_hi.append(dt_res)

    # Statistics at each state
    in_hi = traj_u > threshold_u
    in_lo = ~in_hi

    stats_hi = {
        'mean_u': np.mean(traj_u[in_hi]) if np.any(in_hi) else None,
        'std_u': np.std(traj_u[in_hi]) if np.sum(in_hi) > 10 else None,
        'mean_v': np.mean(traj_v[in_hi]) if np.any(in_hi) else None,
        'std_v': np.std(traj_v[in_hi]) if np.sum(in_hi) > 10 else None,
    }
    stats_lo = {
        'mean_u': np.mean(traj_u[in_lo]) if np.any(in_lo) else None,
        'std_u': np.std(traj_u[in_lo]) if np.sum(in_lo) > 10 else None,
        'mean_v': np.mean(traj_v[in_lo]) if np.any(in_lo) else None,
        'std_v': np.std(traj_v[in_lo]) if np.sum(in_lo) > 10 else None,
    }

    mfpt_hi = np.mean(residence_times_hi) if len(residence_times_hi) > 0 else None
    mfpt_lo = np.mean(residence_times_lo) if len(residence_times_lo) > 0 else None

    return mfpt_hi, mfpt_lo, stats_hi, stats_lo, len(switch_times)

def compute_quasipotential_1d(traj_u, Omega, n_bins=100):
    """
    Compute the 1D quasi-potential from the marginal distribution of u.
    V(u) = -ln(P(u)) / Omega
    """
    counts, bin_edges = np.histogram(traj_u, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Avoid log(0)
    mask = counts > 0
    V = np.full_like(counts, dtype=float, fill_value=np.nan)
    V[mask] = -np.log(counts[mask])

    # Normalize: set minimum to 0
    V_min = np.nanmin(V)
    V = V - V_min

    return bin_centers, V

# ============================================================
# PART 3: Main computation
# ============================================================

def main():
    alpha = 10.0  # standard benchmark
    n = 2          # Hill coefficient (cooperativity)

    print("=" * 72)
    print("GARDNER-COLLINS TOGGLE SWITCH: KRAMERS ESCAPE TEST")
    print("=" * 72)
    print(f"\nModel: du/dt = α/(1+v^n) - u,  dv/dt = α/(1+u^n) - v")
    print(f"Parameters: α = {alpha}, n = {n}")

    # ---- Fixed points ----
    fp_hi, fp_lo, fp_saddle = find_fixed_points(alpha, n)
    print(f"\n--- DETERMINISTIC FIXED POINTS ---")
    print(f"Stable state A (high-u): u = {fp_hi[0]:.4f}, v = {fp_hi[1]:.4f}")
    print(f"Stable state B (high-v): u = {fp_lo[0]:.4f}, v = {fp_lo[1]:.4f}")
    print(f"Saddle point:            u = {fp_saddle[0]:.4f}, v = {fp_saddle[1]:.4f}")

    # ---- Jacobian and relaxation time ----
    J_hi = jacobian(*fp_hi, alpha, n)
    eigs = np.linalg.eigvals(J_hi)
    tau_relax = 1.0 / np.min(np.abs(np.real(eigs)))  # slowest decay
    print(f"\nJacobian eigenvalues at stable A: {eigs[0]:.4f}, {eigs[1]:.4f}")
    print(f"Relaxation time (1/|λ_slow|): τ_relax = {tau_relax:.4f}")

    # ---- Run for multiple system sizes ----
    Omega_values = [5, 8, 10, 12, 15, 20]

    print(f"\n{'='*72}")
    print(f"STOCHASTIC SIMULATION (Gillespie SSA)")
    print(f"{'='*72}")

    results = []

    for Omega in Omega_values:
        print(f"\n--- Ω = {Omega} ---")

        # LNA prediction for variance
        C = lna_covariance(*fp_hi, alpha, n, Omega)
        sigma_u_lna = np.sqrt(C[0, 0]) * Omega  # in molecule units
        sigma_v_lna = np.sqrt(C[1, 1]) * Omega

        # z-score prediction (Round 2 formula)
        x_eq = Omega * fp_hi[0]  # molecule count at stable state
        x_saddle = Omega * fp_saddle[0]  # molecule count at saddle
        z = abs(x_eq - x_saddle) / sigma_u_lna
        D_z = np.exp(z**2 / 2)

        print(f"  LNA: σ_u = {sigma_u_lna:.2f} molecules (at stable A)")
        print(f"  x_eq = {x_eq:.1f}, x_saddle = {x_saddle:.1f}")
        print(f"  z-score = |{x_eq:.1f} - {x_saddle:.1f}| / {sigma_u_lna:.2f} = {z:.2f}")
        print(f"  D_z = exp(z²/2) = exp({z**2/2:.1f}) = {D_z:.2e}")

        # Run Gillespie — adapt simulation time based on expected switching rate
        # For small Omega, switching is frequent; for large, it's rare
        if Omega <= 10:
            T_max = 200_000
        elif Omega <= 15:
            T_max = 500_000
        else:
            T_max = 2_000_000

        print(f"  Running Gillespie (T_max = {T_max:.0e})...", end=" ", flush=True)
        t0 = time.time()
        times, traj_u, traj_v, n_rxn = gillespie_toggle(alpha, n, Omega, T_max, seed=42)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s, {n_rxn:.2e} reactions)")

        # Analyze switching
        mfpt_hi, mfpt_lo, stats_hi, stats_lo, n_switches = analyze_switching(
            times, traj_u, traj_v, Omega, alpha, n
        )

        print(f"  Switching events: {n_switches}")

        if n_switches < 4:
            print(f"  *** Too few switches to analyze reliably ***")
            print(f"  (System is too stable at this Ω for feasible simulation)")
            results.append({
                'Omega': Omega, 'z': z, 'D_z': D_z,
                'D_obs': None, 'D_barrier': None, 'n_switches': n_switches,
                'sigma_lna': sigma_u_lna
            })
            continue

        # MFPT (average of both directions for symmetric case)
        mfpts = [x for x in [mfpt_hi, mfpt_lo] if x is not None]
        mfpt_avg = np.mean(mfpts)
        D_obs = mfpt_avg / tau_relax

        print(f"  MFPT (high-u state): {mfpt_hi:.1f}" if mfpt_hi else "")
        print(f"  MFPT (high-v state): {mfpt_lo:.1f}" if mfpt_lo else "")
        print(f"  MFPT (average): {mfpt_avg:.1f}")
        print(f"  D_observed = MFPT/τ_relax = {mfpt_avg:.1f}/{tau_relax:.4f} = {D_obs:.1f}")

        # Measured variance
        if stats_hi['std_u'] is not None:
            sigma_u_obs = stats_hi['std_u']
            z_obs = abs(stats_hi['mean_u'] - Omega * fp_saddle[0]) / sigma_u_obs
            D_z_obs = np.exp(z_obs**2 / 2)
            print(f"  Measured σ_u = {sigma_u_obs:.2f} (LNA predicted {sigma_u_lna:.2f})")
            print(f"  Measured z = {z_obs:.2f}")
            print(f"  D_z (measured) = exp({z_obs**2/2:.1f}) = {D_z_obs:.2e}")

        # Quasi-potential from histogram
        bin_centers, V_1d = compute_quasipotential_1d(traj_u, Omega, n_bins=80)

        # Find barrier: V at saddle minus V at minimum
        saddle_mol = Omega * fp_saddle[0]
        stable_mol = Omega * fp_hi[0]

        # Find closest bins
        idx_saddle = np.argmin(np.abs(bin_centers - saddle_mol))
        idx_stable = np.argmin(np.abs(bin_centers - stable_mol))

        if not np.isnan(V_1d[idx_saddle]) and not np.isnan(V_1d[idx_stable]):
            delta_V = V_1d[idx_saddle] - V_1d[idx_stable]
            D_barrier = np.exp(delta_V)
            print(f"  Quasi-potential barrier ΔV = {delta_V:.3f}")
            print(f"  D_barrier = exp(ΔV) = exp({delta_V:.3f}) = {D_barrier:.1f}")
        else:
            delta_V = None
            D_barrier = None
            print(f"  Could not compute quasi-potential (insufficient sampling near saddle)")

        results.append({
            'Omega': Omega, 'z': z, 'D_z': D_z, 'D_obs': D_obs,
            'D_barrier': D_barrier, 'delta_V': delta_V,
            'n_switches': n_switches, 'sigma_lna': sigma_u_lna,
            'mfpt': mfpt_avg
        })

    # ---- Summary table ----
    print(f"\n{'='*72}")
    print(f"SUMMARY: THREE D VALUES COMPARED")
    print(f"{'='*72}")
    print(f"{'Ω':>4} {'#sw':>5} {'MFPT':>10} {'D_obs':>10} {'ΔV':>8} {'D_barrier':>12} {'z':>6} {'D_z':>12} {'D_z/D_obs':>10}")
    print("-" * 85)

    for r in results:
        Omega = r['Omega']
        n_sw = r['n_switches']

        if r['D_obs'] is None:
            print(f"{Omega:>4} {n_sw:>5}   {'(too few switches)':^60}")
            continue

        mfpt_str = f"{r['mfpt']:.1f}" if r.get('mfpt') else "—"
        D_obs_str = f"{r['D_obs']:.1f}"

        if r['D_barrier'] is not None:
            dV_str = f"{r['delta_V']:.2f}"
            D_bar_str = f"{r['D_barrier']:.1f}"
        else:
            dV_str = "—"
            D_bar_str = "—"

        z_str = f"{r['z']:.2f}"
        D_z_str = f"{r['D_z']:.1e}"

        if r['D_obs'] is not None and r['D_obs'] > 0:
            ratio = r['D_z'] / r['D_obs']
            ratio_str = f"{ratio:.1e}"
        else:
            ratio_str = "—"

        print(f"{Omega:>4} {n_sw:>5} {mfpt_str:>10} {D_obs_str:>10} {dV_str:>8} {D_bar_str:>12} {z_str:>6} {D_z_str:>12} {ratio_str:>10}")

    print(f"\n{'='*72}")
    print(f"INTERPRETATION")
    print(f"{'='*72}")
    print("""
D_observed = MFPT / τ_relax           (ground truth from simulation)
D_z        = exp(z²/2)                (Round 2 formula — Gaussian extrapolation)
D_barrier  = exp(ΔV_actual)           (actual quasi-potential from -ln(P_ss))

If D_barrier ≈ D_observed:  Kramers theory WORKS for this endogenous system.
If D_z >> D_observed:       The z-score formula OVERESTIMATES because it assumes
                            the potential is quadratic all the way to the saddle.
                            The actual potential flattens. This is why Round 2 failed.

The correct formula for endogenous systems is D = exp(ΔV_actual),
where ΔV must be computed from the actual quasi-potential landscape,
NOT extrapolated from local curvature via z-scores.
""")

if __name__ == "__main__":
    main()
