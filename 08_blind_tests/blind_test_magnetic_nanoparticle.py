#!/usr/bin/env python3
"""
BLIND PREDICTION TEST #2: STONER-WOHLFARTH MAGNETIC NANOPARTICLE
=================================================================
Tests B invariance with ASYMMETRIC potential (unequal curvatures).

Model: Uniaxial single-domain nanoparticle in applied field
  V(θ) = sin²(θ) - 2h·cos(θ)     [units of K_a·Vol]
  drift = -dV/dθ = -2sin(θ)(cos(θ) + h)
  noise: σ² = 2kT/(K_a·Vol)

  h = H/H_k (normalized field), bifurcation parameter
  H_k = 2K_a/(μ₀M_s) = anisotropy field
  Bistable range: h ∈ (0, 1)  [shallow well disappears at h = 1]

Domain: Nanomagnetism (NEW for framework)
Source: Stoner & Wohlfarth (1948); Néel (1949); Brown (1963)
        Data: Wernsdorfer et al. PRL 78, 1791 (1997)

ANALYTIC RESULTS:
  Escape from shallow well (θ = π) over saddle (θ_s = arccos(-h)):
  ΔV(h) = (1-h)²
  V''(π) = 2(1-h)              → λ_eq = 2(1-h)
  |V''(θ_s)| = 2(1-h)(1+h)    → |λ_sad| = 2(1-h²)
  Curvature ratio = 1/(1+h)    → VARIES with h (unequal for h > 0!)

  At h = 0: equal curvatures (symmetric double-well)
  At h > 0: curvature ratio < 1 (asymmetric, K deviates from 1.0)

FRAMEWORK PREDICTIONS BEING TESTED:
  1. B invariance: B = 2ΔV/σ*² should have CV < 5% across h ∈ (0, ~0.95)
     even though curvatures are UNEQUAL and β₀ varies with h.
     THIS IS THE NON-TRIVIAL TEST (unlike Josephson, where B inv. is trivial).
  2. K should match the eigenvalue-ratio prediction:
     K → 1.0 at h = 0 (equal curvature)
     K varies systematically with curvature ratio at h > 0
  3. β₀ varies with h: β₀ = ln(K) + ln(2π) - (1/2)·ln(1+h)

Physical context:
  Néel-Brown formula: τ = τ₀ · exp(K_a·V·(1-h)²/kT)
  τ₀ ~ 10⁻⁹ s (attempt time). Verified for individual nanoparticles.
"""
import numpy as np
from scipy.optimize import brentq

# ============================================================
# MODEL DEFINITION: Stoner-Wohlfarth nanoparticle
# ============================================================

def V_sw(theta, h):
    """Potential: V(θ) = sin²(θ) - 2h·cos(θ) [units of K_a·Vol]"""
    return np.sin(theta)**2 - 2.0 * h * np.cos(theta)


def f_sw(theta, h):
    """Drift: -dV/dθ = -2sin(θ)(cos(θ) + h)"""
    return -2.0 * np.sin(theta) * (np.cos(theta) + h)


def Vpp_sw(theta, h):
    """Second derivative: d²V/dθ² = 2cos(2θ) + 2h·cos(θ)"""
    return 2.0 * np.cos(2.0 * theta) + 2.0 * h * np.cos(theta)


# ============================================================
# ANALYTIC EQUILIBRIA AND BARRIER
# ============================================================

def equilibria_sw(h):
    """
    Analytic equilibria for h ∈ (0, 1):
      θ = 0:   deep well (aligned with field)
      θ = π:   shallow well (anti-aligned)
      θ_s = arccos(-h): saddle
    Returns (θ_deep, θ_shallow, θ_saddle).
    """
    theta_deep = 0.0
    theta_shallow = np.pi
    theta_saddle = np.arccos(-h)
    return theta_deep, theta_shallow, theta_saddle


def barrier_shallow(h):
    """Barrier from shallow well (θ=π) to saddle: ΔV = (1-h)²"""
    return (1.0 - h)**2


def barrier_deep(h):
    """Barrier from deep well (θ=0) to saddle: ΔV = (1+h)²"""
    return (1.0 + h)**2


def eigenvalues_sw(h):
    """
    Eigenvalues at shallow well and saddle.
    λ_eq  = V''(π) = 2(1-h)               [positive, stable]
    λ_sad = V''(θ_s) = -2(1-h²) = -2(1-h)(1+h)  [negative, unstable]
    """
    lam_eq = 2.0 * (1.0 - h)
    lam_sad = -2.0 * (1.0 - h**2)
    return lam_eq, lam_sad


# ============================================================
# KRAMERS QUANTITIES
# ============================================================

def kramers_quantities_sw(h):
    """Compute all Kramers prefactor quantities analytically."""
    theta_deep, theta_shallow, theta_saddle = equilibria_sw(h)
    lam_eq, lam_sad = eigenvalues_sw(h)
    DeltaV = barrier_shallow(h)

    abs_lam_eq = abs(lam_eq)
    abs_lam_sad = abs(lam_sad)

    C = np.sqrt(abs_lam_eq * abs_lam_sad) / (2.0 * np.pi)
    tau = 1.0 / abs_lam_eq
    inv_Ctau = 1.0 / (C * tau)
    # inv_Ctau = 2π · √(λ_eq / |λ_sad|) = 2π / √(1+h)

    curvature_ratio = abs_lam_eq / abs_lam_sad  # = 1/(1+h)

    return {
        'theta_shallow': theta_shallow,
        'theta_saddle': theta_saddle,
        'theta_deep': theta_deep,
        'lam_eq': lam_eq,
        'lam_sad': lam_sad,
        'abs_lam_eq': abs_lam_eq,
        'abs_lam_sad': abs_lam_sad,
        'DeltaV': DeltaV,
        'C': C,
        'tau': tau,
        'inv_Ctau': inv_Ctau,
        'curvature_ratio': curvature_ratio,
    }


# ============================================================
# EXACT MFPT (Fokker-Planck integral)
# ============================================================

def compute_D_exact(h, sigma, N=200000):
    """
    Exact D = MFPT/τ for escape from shallow well (θ=π) to saddle (θ_s).

    Escape direction: θ DECREASING from π to θ_s (towards saddle).
    Reflecting boundary: 2π - θ_s (other side of shallow well).
    Absorbing boundary: θ_s.

    MFPT = (2/σ²) ∫_{θ_s}^{π} exp(Φ(y)) [∫_y^{θ_ref} exp(-Φ(z)) dz] dy
    where Φ(y) = 2V(y)/σ², escape to the LEFT.
    """
    _, theta_shallow, theta_saddle = equilibria_sw(h)
    lam_eq = 2.0 * (1.0 - h)
    if lam_eq <= 0:
        return 0.0  # no well

    # Reflecting boundary on other side of shallow well
    theta_ref = 2.0 * np.pi - theta_saddle

    # Integration grid from saddle to reflecting boundary
    theta_grid = np.linspace(theta_saddle, theta_ref, N)
    dtheta = theta_grid[1] - theta_grid[0]

    # Potential on grid
    Vg = V_sw(theta_grid, h)

    # Reference at shallow well (θ = π)
    i_eq = np.argmin(np.abs(theta_grid - np.pi))
    Vg = Vg - Vg[i_eq]

    # Φ = 2V/σ²
    Phi = 2.0 * Vg / sigma**2
    Phi = Phi - Phi[i_eq]

    # Overflow check
    max_Phi = np.max(Phi)
    if max_Phi > 700:
        return np.inf

    # Reverse cumulative integral: J(y) = ∫_y^{θ_ref} exp(-Φ(z)) dz
    exp_neg_Phi = np.exp(-Phi)
    J = np.cumsum(exp_neg_Phi[::-1])[::-1] * dtheta

    # ψ(y) = (2/σ²) · exp(Φ(y)) · J(y)
    exp_pos_Phi = np.exp(Phi)
    psi = (2.0 / sigma**2) * exp_pos_Phi * J

    # MFPT = ∫_{θ_s}^{π} ψ(y) dy
    # θ_s is at index 0, π is at index i_eq
    i_sad = 0
    MFPT = np.trapz(psi[i_sad:i_eq + 1], theta_grid[i_sad:i_eq + 1])

    tau = 1.0 / lam_eq
    return MFPT / tau


# ============================================================
# SIGMA* FINDER (bisection)
# ============================================================

def find_sigma_star(h, D_target, sig_lo=0.001, sig_hi=5.0):
    """Find σ where D_exact(σ) = D_target via bisection."""
    D_hi = compute_D_exact(h, sig_hi)
    if D_hi > D_target:
        for s in [10.0, 20.0, 50.0, 100.0]:
            D_test = compute_D_exact(h, s)
            if D_test < D_target:
                sig_hi = s
                break
        else:
            return None

    D_lo = compute_D_exact(h, sig_lo)
    if D_lo == np.inf:
        D_lo = 1e30
    if D_lo < D_target:
        return None

    def obj(s):
        d = compute_D_exact(h, s)
        if d == np.inf:
            return 1e30 - D_target
        return d - D_target

    try:
        sigma_star = brentq(obj, sig_lo, sig_hi, xtol=1e-12, maxiter=300)
        return sigma_star
    except ValueError:
        return None


# ============================================================
# MAIN ANALYSIS
# ============================================================

if __name__ == '__main__':
    print("=" * 78)
    print("  BLIND PREDICTION TEST #2: MAGNETIC NANOPARTICLE")
    print("  Stoner-Wohlfarth — Uniaxial Double-Well with Applied Field")
    print("  V(θ) = sin²(θ) - 2h·cos(θ)")
    print("=" * 78)

    # ========================================================
    # STEP 1: VERIFY ANALYTIC STRUCTURE
    # ========================================================
    print(f"\n{'=' * 78}")
    print("STEP 1: ANALYTIC STRUCTURE — CURVATURE ASYMMETRY")
    print(f"{'=' * 78}")

    print(f"\n  Escape from SHALLOW well (θ=π) over saddle to deep well (θ=0)")
    print(f"  Barrier: ΔV = (1-h)²")
    print(f"  Key: curvature ratio = 1/(1+h) → VARIES with h")
    print(f"  At h=0: symmetric (ratio=1.0). At h=0.5: ratio=0.667. At h=0.9: ratio=0.526.")

    print(f"\n  {'h':>6s}  {'θ_sad':>8s}  {'ΔV':>10s}  {'λ_eq':>8s}  "
          f"{'|λ_sad|':>8s}  {'ratio':>6s}  {'1/(Cτ)':>8s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*8}")

    for hval in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        kq = kramers_quantities_sw(hval)
        print(f"  {hval:6.2f}  {kq['theta_saddle']:8.4f}  {kq['DeltaV']:10.6f}  "
              f"{kq['abs_lam_eq']:8.5f}  {kq['abs_lam_sad']:8.5f}  "
              f"{kq['curvature_ratio']:6.4f}  {kq['inv_Ctau']:8.4f}")

    print(f"\n  Curvature ratio ranges from 1.000 (h=0) to 0.513 (h=0.95)")
    print(f"  1/(Cτ) ranges from {2*np.pi:.3f} (h=0) to {2*np.pi/np.sqrt(1.95):.3f} (h=0.95)")
    print(f"  → β₀ is NOT constant. B invariance is a NON-TRIVIAL prediction.")

    # ========================================================
    # STEP 2: K DETERMINATION
    # ========================================================
    print(f"\n{'=' * 78}")
    print("STEP 2: K DETERMINATION ACROSS FIELD VALUES")
    print("  K should start at ~1.0 (h=0, equal curvature)")
    print("  and evolve systematically with curvature ratio")
    print(f"{'=' * 78}")

    print(f"\n  {'h':>6s}  {'σ':>8s}  {'barrier':>8s}  {'D_exact':>12s}  "
          f"{'D_Kr(K=1)':>12s}  {'K_actual':>10s}  {'ratio':>6s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*6}")

    K_by_h = {}
    for hval in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]:
        kq = kramers_quantities_sw(hval)
        K_list = []
        for sigma in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
            barrier = 2.0 * kq['DeltaV'] / sigma**2
            if barrier < 2.0 or barrier > 300:
                continue
            D_ex = compute_D_exact(hval, sigma)
            if D_ex == np.inf or D_ex < 2:
                continue
            D_kr_raw = np.exp(barrier) * kq['inv_Ctau']
            K_act = D_ex / D_kr_raw
            if 3.0 < barrier < 50.0:
                K_list.append(K_act)
            print(f"  {hval:6.2f}  {sigma:8.3f}  {barrier:8.2f}  {D_ex:12.4e}  "
                  f"{D_kr_raw:12.4e}  {K_act:10.4f}  {kq['curvature_ratio']:6.4f}")
        if K_list:
            K_by_h[hval] = np.mean(K_list)

    if K_by_h:
        print(f"\n  K vs curvature ratio summary:")
        print(f"  {'h':>6s}  {'curv_ratio':>10s}  {'K_mean':>10s}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}")
        for hval in sorted(K_by_h.keys()):
            cr = 1.0 / (1.0 + hval)
            print(f"  {hval:6.2f}  {cr:10.4f}  {K_by_h[hval]:10.4f}")

    # ========================================================
    # STEP 3: B INVARIANCE TEST (THE KEY TEST)
    # ========================================================
    print(f"\n{'=' * 78}")
    print("STEP 3: B INVARIANCE TEST (NON-TRIVIAL)")
    print("  Sweep h across bistable range, compute B = 2ΔV/σ*² at fixed D_target")
    print("  Curvature ratio changes continuously → β₀ varies → B could fail!")
    print(f"{'=' * 78}")

    D_targets = [50, 100, 500]

    for D_target in D_targets:
        print(f"\n  --- D_target = {D_target} ---")
        print(f"  {'h':>6s}  {'ΔV=(1-h)²':>10s}  {'σ*':>10s}  {'B':>10s}  "
              f"{'K_act':>10s}  {'β₀':>8s}  {'ratio':>6s}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  "
              f"{'-'*10}  {'-'*8}  {'-'*6}")

        B_values = []
        K_star_values = []
        beta0_values = []

        h_sweep = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                   0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                   0.80, 0.85, 0.90, 0.93, 0.95, 0.97]

        for hval in h_sweep:
            kq = kramers_quantities_sw(hval)
            sigma_star = find_sigma_star(hval, D_target)
            if sigma_star is None:
                print(f"  {hval:6.2f}  {kq['DeltaV']:10.6f}  {'---':>10s}  "
                      f"{'---':>10s}  {'---':>10s}  {'---':>8s}  "
                      f"{kq['curvature_ratio']:6.4f}")
                continue

            B = 2.0 * kq['DeltaV'] / sigma_star**2
            D_ex = compute_D_exact(hval, sigma_star)
            D_kr_raw = np.exp(B) * kq['inv_Ctau']
            K_act = D_ex / D_kr_raw if D_kr_raw > 0 else np.nan
            b0 = np.log(max(D_ex, 1e-30)) - B

            B_values.append(B)
            K_star_values.append(K_act)
            beta0_values.append(b0)

            print(f"  {hval:6.2f}  {kq['DeltaV']:10.6f}  {sigma_star:10.6f}  "
                  f"{B:10.4f}  {K_act:10.4f}  {b0:8.4f}  "
                  f"{kq['curvature_ratio']:6.4f}")

        if len(B_values) >= 3:
            B_arr = np.array(B_values)
            b0_arr = np.array(beta0_values)
            K_arr = np.array(K_star_values)

            # Barrier variation
            DV_max = barrier_shallow(h_sweep[0])
            DV_min = barrier_shallow(h_sweep[-1])
            barrier_ratio = DV_max / DV_min if DV_min > 0 else np.inf

            print(f"\n  *** RESULTS for D_target = {D_target} ***")
            print(f"    B mean      = {np.mean(B_arr):.4f}")
            print(f"    B std       = {np.std(B_arr):.4f}")
            print(f"    B CV        = {np.std(B_arr)/np.mean(B_arr)*100:.2f}%")
            print(f"    B range     = [{np.min(B_arr):.4f}, {np.max(B_arr):.4f}]")
            print(f"    β₀ mean     = {np.mean(b0_arr):.4f}")
            print(f"    β₀ range    = [{np.min(b0_arr):.4f}, {np.max(b0_arr):.4f}]")
            print(f"    β₀ CV       = {np.std(b0_arr)/np.mean(b0_arr)*100:.2f}%")
            print(f"    K mean      = {np.mean(K_arr):.4f}")
            print(f"    K range     = [{np.min(K_arr):.4f}, {np.max(K_arr):.4f}]")
            print(f"    Barrier var = {barrier_ratio:.0f}x  (ΔV from {DV_max:.4f} to {DV_min:.6f})")
            print(f"    Curv. ratio = 1.000 to {1.0/(1.0+h_sweep[-1]):.3f}")
            cv = np.std(B_arr) / np.mean(B_arr) * 100
            print(f"    B INVARIANCE: {'PASS (CV < 5%)' if cv < 5.0 else 'MARGINAL' if cv < 10 else 'FAIL'}")

    # ========================================================
    # STEP 4: HABITABLE ZONE
    # ========================================================
    print(f"\n{'=' * 78}")
    print("STEP 4: HABITABLE ZONE CONTEXT")
    print(f"{'=' * 78}")

    print(f"\n  Framework habitable zone: B ∈ [1.8, 6.0] (9 systems, 4 domains)")
    if B_values:
        B_mean = np.mean(B_values)
        in_zone = 1.8 <= B_mean <= 6.0
        print(f"  Nanoparticle B (mean at D=500): {B_mean:.2f}")
        print(f"  In habitable zone: {'YES' if in_zone else 'NO'}")
    print(f"\n  Physical interpretation:")
    print(f"  B = K_a·V·(1-h)² / kT")
    print(f"  For a 5 nm Co particle (K_a=4.5×10⁵ J/m³, V=6.5×10⁻²⁶ m³):")
    print(f"  K_a·V = 2.9×10⁻²⁰ J → K_a·V/k = 2100 K")
    print(f"  At h=0: B=4 requires T ≈ 530 K (superparamagnetic transition)")
    print(f"  At h=0.9: B=4 requires T ≈ 5 K (low-field blocking)")

    # ========================================================
    # STEP 5: COMPARISON WITH EXISTING FRAMEWORK SYSTEMS
    # ========================================================
    print(f"\n{'=' * 78}")
    print("STEP 5: COMPARISON WITH VERIFIED SYSTEMS")
    print(f"{'=' * 78}")

    print(f"\n  {'System':<22s}  {'Domain':<18s}  {'B':>6s}  {'B CV':>6s}  {'K':>6s}")
    print(f"  {'-'*22}  {'-'*18}  {'-'*6}  {'-'*6}  {'-'*6}")

    existing = [
        ("Kelp",              "Ecology",         1.80,  2.6,  0.34),
        ("Savanna",           "Ecology",         3.74,  4.6,  0.55),
        ("Lake",              "Ecology",         4.25,  2.0,  0.56),
        ("Coral",             "Ecology",         6.04,  2.1,  0.56),
        ("Toggle switch",     "Gene circuit",    4.83,  3.8,  1.00),
        ("Tumor-immune",      "Cancer biology",  2.73,  2.7,  None),
        ("Diabetes",          "Human disease",   5.54,  5.2,  None),
        ("Power grid (SMIB)", "Engineering",     None,  None, 1.00),
    ]

    for name, domain, B, Bcv, K in existing:
        B_str = f"{B:6.2f}" if B else "  ---"
        cv_str = f"{Bcv:5.1f}%" if Bcv else "  ---"
        K_str = f"{K:6.2f}" if K else "  ---"
        print(f"  {name:<22s}  {domain:<18s}  {B_str}  {cv_str}  {K_str}")

    # Add our result
    if B_values:
        B_mean_100 = None
        # Recompute for D=100 specifically
        B_100 = []
        for hval in h_sweep:
            ss = find_sigma_star(hval, 100)
            if ss is not None:
                kq = kramers_quantities_sw(hval)
                B_100.append(2.0 * kq['DeltaV'] / ss**2)
        if B_100:
            B_mean_100 = np.mean(B_100)
            B_cv_100 = np.std(B_100) / np.mean(B_100) * 100
            K_mean = np.mean(list(K_by_h.values())) if K_by_h else 0
            print(f"  {'Nanoparticle':<22s}  {'Nanomagnetism':<18s}  "
                  f"{B_mean_100:6.2f}  {B_cv_100:5.1f}%  {K_mean:6.2f}  ← THIS TEST")

    # ========================================================
    # SUMMARY
    # ========================================================
    print(f"\n\n{'=' * 78}")
    print("SUMMARY: MAGNETIC NANOPARTICLE BLIND PREDICTION TEST")
    print(f"{'=' * 78}")

    print(f"""
  System:          Stoner-Wohlfarth single-domain nanoparticle
  Domain:          Nanomagnetism (NEW)
  ODE:             dθ/dt = -2sin(θ)(cos(θ) + h)
  Potential:       V(θ) = sin²(θ) - 2h·cos(θ)
  Dimensions:      1D
  Free parameters: 0
  Bistability:     TRUE double-well (two stable orientations)

  Structural properties:
    Curvature ratio = 1/(1+h) → varies from 1.0 to 0.5
    1/(Cτ) = 2π/√(1+h) → varies with h
    β₀ varies with h → B invariance is NON-TRIVIAL

  Framework predictions tested:
    1. B invariance (CV < 5%)  ← THE KEY TEST (unequal curvatures)
    2. K variation with curvature ratio
    3. β₀ variation matches eigenvalue ratio prediction
    4. Kramers accuracy across asymmetry range

  Why this test matters:
    All prior B invariance tests (lake, coral, kelp, toggle, savanna)
    had approximately equal curvatures OR were tested at a single
    curvature ratio. This system CONTINUOUSLY varies the curvature
    ratio from 1.0 to ~0.5. If B invariance holds here, it is a
    genuine structural property of fold bifurcations, not an artifact
    of symmetric potentials.
""")
    print("=" * 78)
    print("  DONE")
    print("=" * 78)
