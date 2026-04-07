#!/usr/bin/env python3
"""
STUDY 31: PROTEIN CONFORMATIONAL DYNAMICS -- STABILITY WINDOW TEST
===================================================================
Tests whether protein folding/unfolding dynamics satisfy the
framework's stability window B in [1.8, 6.0].

Mapping for overdamped Langevin dynamics in a thermal potential:
  SDE: dX = -(1/gamma)*V'(X) dt + sqrt(2kT/gamma) dW
  Quasipotential: Phi(x) = (1/gamma)*V(x)
  sigma^2 = 2kT/gamma
  B = 2*DeltaPhi/sigma^2 = 2*(DeltaV/gamma)/(2kT/gamma) = DeltaV/kT

For proteins: DeltaV = DeltaG_u^dagger (activation free energy for unfolding)
  => B = DeltaG_u^dagger / kT

The factor of 2 in the framework's B = 2*DeltaPhi/sigma^2 cancels exactly
with the factor of 2 in the thermal diffusion coefficient sigma^2 = 2kT/gamma.

D = MFPT/tau_relax = (1/k_u) * (k_f + k_u) = 1 + k_f/k_u ~ K_eq for stable proteins.

B = ln(k_0/k_u) where k_0 is the Kramers pre-exponential (attempt frequency).
The bridge relation gives: B = ln(D) + DeltaG_f^dagger/kT (= ln(D) + beta_0).

Five tests:
  1. Master table: D and B for 25 two-state proteins spanning 6 OOM in D
  2. Stability window test: do proteins in D=[30,1000] have B in [1.8,6.0]?
  3. B vs ln(D) relationship: slope, offset, comparison to framework
  4. B invariance test using chevron plot data (3 proteins)
  5. Sensitivity to pre-exponential k_0

Dependencies: numpy only.
"""

import numpy as np
import sys

np.random.seed(42)


def flush():
    sys.stdout.flush()


# ================================================================
# PHYSICAL CONSTANTS
# ================================================================
kB = 1.380649e-23    # Boltzmann constant, J/K
h_planck = 6.62607e-34  # Planck constant, J*s
T_REF = 298.15       # Reference temperature, K (25 C)
kT = kB * T_REF      # Thermal energy at 25C, J
RT_kcal = 0.5922     # RT at 25C in kcal/mol (1.987 cal/(mol*K) * 298.15 K / 1000)

# Pre-exponential factors
k0_eyring = kB * T_REF / h_planck   # Eyring TST: kT/h = 6.25e12 s^-1
k0_kramers = 1.0e6   # Kramers speed limit for proteins (Kubelka et al. 2004)
                      # Based on tau_f,min ~ N/100 us, giving k_0 ~ 10^6 for N~100

# Stability window from SYSTEMS.md (13 verified systems across 7 domains)
B_LOWER = 1.8
B_UPPER = 6.0


# ================================================================
# PROTEIN DATASET
# All rates at 25C (298 K), pH ~7, extrapolated to 0 M denaturant
# from chevron plots unless noted. Two-state folders only.
# ================================================================
# Format: (name, N_residues, k_f [s^-1], k_u [s^-1], reference)

PROTEINS = [
    # --- Marginally stable / ultrafast folders ---
    ("Trp-cage TC5b",    20,  2.4e5,  5.8e4,
     "Qiu & Hagen 2004 JACS 126:3398; Streicher & Makhatadze 2007"),
    ("BBA5",             23,  1.2e5,  8.2e4,
     "Kubelka et al. 2003 JACS 125:13964"),
    ("Villin HP35 N68H", 35,  7.1e5,  1.0e5,
     "Kubelka et al. 2003 JACS 125:13964"),
    ("WW domain FiP35",  35,  1.4e4,  1.1e3,
     "Liu et al. 2008 PNAS 105:2369"),
    ("WW domain Pin1",   34,  7.7e3,  1.2e3,
     "Jager et al. 2006 PNAS 103:10648"),

    # --- Fast folders ---
    ("Engrailed HD",     61,  4.9e4,  2.0e3,
     "Mayor et al. 2003 PNAS 100:3457"),
    ("Protein A BdpA",   60,  1.0e4,  1.6e2,
     "Myers & Oas 2001 JMB 312:263"),
    ("lambda repressor",  80,  3.3e4,  2.0e1,
     "Burton et al. 1997 Nat Struct Biol 4:305"),

    # --- Standard two-state folders ---
    ("CspB B.subtilis",  67,  1.19e3, 8.0,
     "Schindler et al. 1995 Biochemistry 34:16602"),
    ("CspB Thermotoga",  66,  5.3e2,  0.30,
     "Perl et al. 1998 Nat Struct Biol 5:229"),
    ("Protein L",        62,  2.5e2,  2.1,
     "Scalley et al. 1997 Biochemistry 36:3373"),
    ("Protein G GB1",    56,  6.3e2,  4.0,
     "Park et al. 1999 JMB 285:1735"),
    ("Im7",              87,  1.2e3,  1.0,
     "Ferguson et al. 1999 JMB 286:1597"),
    ("src SH3",          64,  2.7e1,  6.4e-1,
     "Grantcharova & Baker 1997 Biochemistry 36:15685"),
    ("spectrin SH3",     62,  3.4,    6.4e-2,
     "Viguera et al. 1994 Biochemistry 33:2142"),
    ("fyn SH3",          59,  3.1e2,  1.3e-1,
     "Plaxco et al. 1998 JMB 277:985"),
    ("AcP",              98,  1.0e1,  1.0e-1,
     "Chiti et al. 1999 PNAS 96:3590"),
    ("Barstar",          89,  1.0e2,  2.0e-1,
     "Nolting et al. 1997 Biochemistry 36:9899"),
    ("FKBP12",          107,  2.0,    1.0e-1,
     "Main et al. 1999 Biochemistry 38:8553"),

    # --- Stable proteins ---
    ("CI2",              65,  5.0e1,  4.5e-4,
     "Jackson & Fersht 1991 Biochemistry 30:10428"),
    ("Barnase",         110,  5.0,    5.5e-5,
     "Matouschek et al. 1990 Nature 346:440"),
    ("RNase H",         155,  3.0,    1.0e-3,
     "Raschke & Marqusee 1997 Nat Struct Biol 4:298"),
    ("Ubiquitin",        76,  1.5e3,  5.0e-5,
     "Khorasanizadeh et al. 1996 PNAS 93:5254"),
    ("Titin I27",        89,  2.4e1,  2.5e-4,
     "Fowler & Clarke 2001 Structure 9:355"),
    ("TNfn3",            90,  6.3e2,  2.0e-3,
     "Clarke et al. 1997 Structure 5:907"),
]


# ================================================================
# CHEVRON PLOT DATA (for B invariance test)
# Parameters: ln(k_f(c)) = ln(k_f0) - m_kf * c
#             ln(k_u(c)) = ln(k_u0) + m_ku * c
# where c = [denaturant] in M, m in M^-1 (ln-rate units)
# ================================================================

CHEVRON_DATA = [
    # (name, k_f0, k_u0, m_kf [M^-1], m_ku [M^-1], denaturant, c_max [M], reference)
    ("CI2", 50.0, 4.5e-4, 1.68, 0.46, "GdnHCl",  6.0,
     "Jackson & Fersht 1991 Biochemistry 30:10428"),
    ("Barnase", 5.0, 5.5e-5, 2.05, 0.69, "urea", 8.0,
     "Matouschek et al. 1990 Nature 346:440; Fersht 1999"),
    ("Protein L", 250.0, 2.1, 1.20, 0.58, "GdnHCl", 6.0,
     "Scalley et al. 1997 Biochemistry 36:3373"),
]


# ================================================================
# MAIN ANALYSIS
# ================================================================
if __name__ == '__main__':
    print("=" * 72)
    print("STUDY 31: PROTEIN CONFORMATIONAL DYNAMICS -- STABILITY WINDOW TEST")
    print("=" * 72)
    print()

    # ------------------------------------------------------------------
    # TEST 0: Verify the thermal Kramers mapping
    # ------------------------------------------------------------------
    print("-" * 72)
    print("TEST 0: THERMAL KRAMERS MAPPING VERIFICATION")
    print("-" * 72)
    print()
    print("For overdamped Langevin: dX = -(1/gamma)*V'(X) dt + sqrt(2kT/gamma) dW")
    print()
    print("  Quasipotential:   Phi(x) = (1/gamma)*V(x)")
    print("  Noise intensity:  sigma^2 = 2kT/gamma")
    print("  Framework B:      B = 2*DeltaPhi/sigma^2")
    print("                      = 2*(DeltaV/gamma)/(2kT/gamma)")
    print("                      = DeltaV/kT")
    print()
    print("  => B = DeltaG_u^dagger / kT   (no extra factor of 2)")
    print()
    print("  The factor of 2 in B = 2*DeltaPhi/sigma^2 cancels with the")
    print("  factor of 2 in sigma^2 = 2*D_diff = 2kT/gamma (Einstein relation).")
    print()
    print(f"  kT at {T_REF:.1f} K = {kT*6.022e23/4184:.4f} kcal/mol = {RT_kcal:.4f} kcal/mol")
    print(f"  Eyring pre-exponential kT/h = {k0_eyring:.3e} s^-1")
    print(f"  Kramers speed limit k_0 = {k0_kramers:.1e} s^-1 (Kubelka et al. 2004)")
    print()
    flush()

    # ------------------------------------------------------------------
    # TEST 1: Master table
    # ------------------------------------------------------------------
    print("-" * 72)
    print("TEST 1: MASTER TABLE -- 25 TWO-STATE PROTEINS")
    print("-" * 72)
    print()
    print("D = 1 + k_f/k_u (persistence ratio = MFPT/tau_relax)")
    print("B = ln(k_0/k_u) with k_0 = 1e6 s^-1 (Kramers speed limit)")
    print("DeltaG_stab = kT * ln(D) in kcal/mol")
    print()

    # Header
    hdr = f"{'Protein':<22s} {'N':>4s} {'k_f':>10s} {'k_u':>10s} {'D':>12s} " \
          f"{'ln(D)':>6s} {'B_Kr':>6s} {'DG_stab':>7s} {'Window?':>8s}"
    print(hdr)
    print("-" * len(hdr))

    names = []
    D_vals = []
    B_kramers = []
    B_eyring_vals = []
    ln_D_vals = []
    DG_stab_vals = []  # in kcal/mol
    N_vals = []

    for (name, N, kf, ku, ref) in PROTEINS:
        D = 1.0 + kf / ku
        ln_D = np.log(D)
        B_kr = np.log(k0_kramers / ku)
        B_ey = np.log(k0_eyring / ku)
        DG_stab = RT_kcal * ln_D  # kcal/mol

        names.append(name)
        D_vals.append(D)
        B_kramers.append(B_kr)
        B_eyring_vals.append(B_ey)
        ln_D_vals.append(ln_D)
        DG_stab_vals.append(DG_stab)
        N_vals.append(N)

        in_window = "YES" if B_LOWER <= B_kr <= B_UPPER else "no"
        print(f"{name:<22s} {N:4d} {kf:10.2e} {ku:10.2e} {D:12.1f} "
              f"{ln_D:6.2f} {B_kr:6.2f} {DG_stab:7.2f} {in_window:>8s}")

    D_vals = np.array(D_vals)
    B_kramers = np.array(B_kramers)
    B_eyring_vals = np.array(B_eyring_vals)
    ln_D_vals = np.array(ln_D_vals)
    DG_stab_vals = np.array(DG_stab_vals)
    N_vals = np.array(N_vals)

    n_in_window = np.sum((B_kramers >= B_LOWER) & (B_kramers <= B_UPPER))
    n_below = np.sum(B_kramers < B_LOWER)
    n_above = np.sum(B_kramers > B_UPPER)
    print()
    print(f"  B in [{B_LOWER}, {B_UPPER}]: {n_in_window}/{len(PROTEINS)}")
    print(f"  B < {B_LOWER}: {n_below}/{len(PROTEINS)}")
    print(f"  B > {B_UPPER}: {n_above}/{len(PROTEINS)}")
    print(f"  B range: [{B_kramers.min():.2f}, {B_kramers.max():.2f}]")
    print(f"  D range: [{D_vals.min():.1f}, {D_vals.max():.1e}]")
    print()
    flush()

    # ------------------------------------------------------------------
    # TEST 2: Stability window test by D range
    # ------------------------------------------------------------------
    print("-" * 72)
    print("TEST 2: STABILITY WINDOW TEST BY D RANGE")
    print("-" * 72)
    print()

    # Define D ranges matching the user's request
    ranges = [
        ("D < 10 (marginal)",       D_vals < 10),
        ("D in [10, 30)",           (D_vals >= 10) & (D_vals < 30)),
        ("D in [30, 1000]",         (D_vals >= 30) & (D_vals <= 1000)),
        ("D in (1000, 2000]",       (D_vals > 1000) & (D_vals <= 2000)),
        ("D > 2000",                D_vals > 2000),
    ]

    for label, mask in ranges:
        n = np.sum(mask)
        if n == 0:
            print(f"  {label:30s}: no proteins")
            continue
        B_sub = B_kramers[mask]
        n_win = np.sum((B_sub >= B_LOWER) & (B_sub <= B_UPPER))
        names_sub = [names[i] for i in range(len(names)) if mask[i]]
        print(f"  {label:30s}: {n} proteins, {n_win}/{n} in window "
              f"[B range: {B_sub.min():.2f} - {B_sub.max():.2f}]")
        for i, idx in enumerate(np.where(mask)[0]):
            status = "IN" if B_LOWER <= B_kramers[idx] <= B_UPPER else "OUT"
            print(f"    {names[idx]:22s}  D={D_vals[idx]:>10.1f}  "
                  f"B={B_kramers[idx]:.2f}  {status}")
    print()

    # Key question from the user's prompt
    mask_30_1000 = (D_vals >= 30) & (D_vals <= 1000)
    n_30_1000 = np.sum(mask_30_1000)
    n_30_1000_window = np.sum((B_kramers >= B_LOWER) & (B_kramers <= B_UPPER)
                              & mask_30_1000)
    print(f"  KEY QUESTION: Do proteins with D in [30, 1000] have B in [1.8, 6.0]?")
    print(f"  Answer: {n_30_1000_window}/{n_30_1000} = "
          f"{'YES' if n_30_1000_window == n_30_1000 else 'NO'}")
    if n_30_1000 > 0:
        B_sub = B_kramers[mask_30_1000]
        print(f"  B range for this group: [{B_sub.min():.2f}, {B_sub.max():.2f}]")
        if B_sub.min() > B_UPPER:
            print(f"  All B values ABOVE the window (minimum B = {B_sub.min():.2f} > {B_UPPER})")
    print()
    flush()

    # ------------------------------------------------------------------
    # TEST 3: B vs ln(D) relationship
    # ------------------------------------------------------------------
    print("-" * 72)
    print("TEST 3: B vs ln(D) RELATIONSHIP (BRIDGE ANALYSIS)")
    print("-" * 72)
    print()
    print("Framework prediction: ln(D) = B + beta_0 with beta_0 ~ 0.5-1.6")
    print("Protein expectation:  B = ln(D) + DeltaG_f^dagger/kT")
    print("  => beta_0 = B - ln(D) = DeltaG_f^dagger/kT (the folding barrier)")
    print()

    # Compute beta_0 = B - ln(D) for each protein
    beta_0 = B_kramers - ln_D_vals

    print(f"{'Protein':<22s} {'ln(D)':>6s} {'B':>6s} {'beta_0':>7s} {'DG_f^dag':>9s}")
    print(f"{'':22s} {'':>6s} {'':>6s} {'=B-lnD':>7s} {'(kcal/mol)':>9s}")
    print("-" * 58)
    for i in range(len(PROTEINS)):
        DG_f_kcal = beta_0[i] * RT_kcal
        print(f"{names[i]:<22s} {ln_D_vals[i]:6.2f} {B_kramers[i]:6.2f} "
              f"{beta_0[i]:7.2f} {DG_f_kcal:9.2f}")

    print()
    print(f"  beta_0 range: [{beta_0.min():.2f}, {beta_0.max():.2f}]")
    print(f"  beta_0 mean:  {beta_0.mean():.2f} +/- {beta_0.std():.2f}")
    print(f"  Folding barrier range: [{beta_0.min()*RT_kcal:.1f}, "
          f"{beta_0.max()*RT_kcal:.1f}] kcal/mol")
    print()
    print(f"  Framework beta_0 range: [0.53, 1.61]  (5 ecological systems)")
    print(f"  Protein beta_0 range:   [{beta_0.min():.2f}, {beta_0.max():.2f}]  "
          f"(25 proteins)")
    print()

    # Linear regression: B = a * ln(D) + b
    # Using numpy least squares
    A = np.column_stack([ln_D_vals, np.ones(len(ln_D_vals))])
    result = np.linalg.lstsq(A, B_kramers, rcond=None)
    slope, intercept = result[0]
    B_pred = slope * ln_D_vals + intercept
    SS_res = np.sum((B_kramers - B_pred)**2)
    SS_tot = np.sum((B_kramers - B_kramers.mean())**2)
    R2 = 1.0 - SS_res / SS_tot

    print(f"  Linear fit: B = {slope:.3f} * ln(D) + {intercept:.3f}")
    print(f"  R^2 = {R2:.4f}")
    print(f"  Framework prediction: B = 1.0 * ln(D) + ~1.0")
    print(f"  If slope = 1: intercept = mean(B - ln(D)) = {beta_0.mean():.2f}")
    print()

    # Structural interpretation
    print("  INTERPRETATION:")
    print("  beta_0 = B - ln(D) = DeltaG_f^dagger / kT  (the folding barrier)")
    print(f"  Protein folding barriers: {beta_0.mean():.1f} +/- {beta_0.std():.1f} kT "
          f"({beta_0.mean()*RT_kcal:.1f} +/- {beta_0.std()*RT_kcal:.1f} kcal/mol)")
    print(f"  Framework systems:        ~1 kT (~0.6 kcal/mol)")
    print(f"  => Proteins have beta_0 ~ {beta_0.mean()/1.0:.0f}x larger than framework systems")
    print(f"  => The 'return barrier' (folding barrier) dominates the Kramers prefactor")
    print()
    flush()

    # ------------------------------------------------------------------
    # TEST 4: B invariance (chevron plots)
    # ------------------------------------------------------------------
    print("-" * 72)
    print("TEST 4: B INVARIANCE TEST (CHEVRON PLOTS)")
    print("-" * 72)
    print()
    print("Framework: B is constant across the bistable range (CV < 5%)")
    print("           because barrier and noise are structurally coupled.")
    print("Proteins:  noise = kT (fixed by temperature), so B changes when")
    print("           the barrier changes (via denaturant or mutation).")
    print()

    for (name, kf0, ku0, m_kf, m_ku, denaturant, c_max, ref) in CHEVRON_DATA:
        print(f"  {name} ({denaturant}, {ref.split(';')[0]})")
        print(f"  {'':4s}k_f,water = {kf0:.1e} s^-1, k_u,water = {ku0:.1e} s^-1")
        print(f"  {'':4s}m_kf = {m_kf:.2f} M^-1, m_ku = {m_ku:.2f} M^-1")
        print()

        # Compute B at a range of denaturant concentrations
        c_vals = np.linspace(0, c_max * 0.8, 20)  # up to 80% of max
        B_vals_chev = []
        D_vals_chev = []

        hdr2 = f"  {'[den] M':>8s} {'k_u':>10s} {'k_f':>10s} {'D':>10s} {'B':>7s}"
        print(hdr2)
        print(f"  {'-'*50}")

        for c in c_vals:
            kf_c = kf0 * np.exp(-m_kf * c)
            ku_c = ku0 * np.exp(m_ku * c)
            D_c = 1.0 + kf_c / ku_c
            B_c = np.log(k0_kramers / ku_c)
            B_vals_chev.append(B_c)
            D_vals_chev.append(D_c)

        # Print a subset (every 4th point)
        for j in range(0, len(c_vals), 4):
            c = c_vals[j]
            kf_c = kf0 * np.exp(-m_kf * c)
            ku_c = ku0 * np.exp(m_ku * c)
            D_c = D_vals_chev[j]
            B_c = B_vals_chev[j]
            print(f"  {c:8.2f} {ku_c:10.2e} {kf_c:10.2e} {D_c:10.1f} {B_c:7.2f}")

        B_arr = np.array(B_vals_chev)
        D_arr = np.array(D_vals_chev)
        B_cv = B_arr.std() / B_arr.mean() * 100
        B_range = B_arr.max() - B_arr.min()

        print()
        window_width = B_UPPER - B_LOWER  # 4.2
        frac_window = B_range / window_width * 100
        # Use both CV and absolute Delta_B criteria:
        # Framework B invariance: CV < 5% AND Delta_B small relative to window
        invariant = B_cv < 5 and B_range < 0.5
        print(f"  {'':4s}B range: [{B_arr.min():.2f}, {B_arr.max():.2f}], "
              f"Delta_B = {B_range:.2f} ({frac_window:.0f}% of window width)")
        print(f"  {'':4s}B CV = {B_cv:.1f}%  (framework systems: CV < 5%)")
        print(f"  {'':4s}D range: [{D_arr.min():.1f}, {D_arr.max():.1f}]")
        print(f"  {'':4s}B {'appears' if B_cv < 5 else 'does NOT appear'} "
              f"%-invariant (CV={B_cv:.1f}%), but Delta_B = {B_range:.2f} "
              f"spans {frac_window:.0f}% of the stability window width")
        print(f"  {'':4s}=> B is NOT invariant in any physically meaningful sense")
        print()

    print("  EXPLANATION:")
    print("  In the framework, B is invariant because sigma* is structurally")
    print("  determined: both DeltaPhi and sigma* scale together as the")
    print("  bifurcation parameter changes, keeping B = 2*DeltaPhi/sigma*^2 fixed.")
    print("  For proteins, sigma^2 = 2kT/gamma is fixed by the thermal bath.")
    print("  Denaturant changes only the barrier (DeltaG_u^dagger), not the noise.")
    print("  => B changes linearly with [denaturant]: dB/dc = -m_ku")
    print()
    flush()

    # ------------------------------------------------------------------
    # TEST 5: Sensitivity to pre-exponential k_0
    # ------------------------------------------------------------------
    print("-" * 72)
    print("TEST 5: SENSITIVITY TO PRE-EXPONENTIAL k_0")
    print("-" * 72)
    print()
    print("B = ln(k_0/k_u) depends on the choice of pre-exponential k_0.")
    print("The pre-exponential is poorly constrained for proteins (~10^5 to 10^8).")
    print()

    k0_values = [1e5, 1e6, 1e7, 1e8, k0_eyring]
    k0_labels = ["1e5 (Chung transit)", "1e6 (Kubelka SL)", "1e7 (fast SL)",
                 "1e8 (very fast)", f"{k0_eyring:.1e} (Eyring)"]

    print(f"{'Protein':<22s}", end="")
    for lbl in k0_labels:
        print(f"  {'B('+lbl.split('(')[0].strip()+')':>12s}", end="")
    print()
    print("-" * (22 + 14 * len(k0_values)))

    for i, (name, N, kf, ku, ref) in enumerate(PROTEINS):
        print(f"{name:<22s}", end="")
        for k0 in k0_values:
            B_val = np.log(k0 / ku)
            print(f"  {B_val:12.2f}", end="")
        print()

    print()
    print("  Number of proteins with B in [1.8, 6.0] by k_0 choice:")
    for k0, lbl in zip(k0_values, k0_labels):
        B_k0 = np.log(k0 / np.array([p[3] for p in PROTEINS]))
        n_win = np.sum((B_k0 >= B_LOWER) & (B_k0 <= B_UPPER))
        print(f"    k_0 = {lbl:30s}: {n_win}/{len(PROTEINS)} in window")

    print()
    print("  NOTE: Only the Kramers speed limit (k_0 ~ 10^6) is physically")
    print("  motivated for proteins (Kubelka et al. 2004; Chung et al. 2012).")
    print("  The Eyring value (kT/h) overestimates the barrier by ~15 kT.")
    print()
    flush()

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print()
    print(f"25 two-state proteins tested against the stability window B in [{B_LOWER}, {B_UPPER}].")
    print(f"Primary convention: k_0 = {k0_kramers:.0e} s^-1 (Kramers speed limit).")
    print()

    # Key numbers
    mask_win = (B_kramers >= B_LOWER) & (B_kramers <= B_UPPER)
    names_in = [names[i] for i in range(len(names)) if mask_win[i]]

    print(f"  Proteins with B in [{B_LOWER}, {B_UPPER}]: {n_in_window}/{len(PROTEINS)}")
    if names_in:
        print(f"    {', '.join(names_in)}")
    print(f"  Proteins with B > {B_UPPER}: {n_above}/{len(PROTEINS)} "
          f"(B up to {B_kramers.max():.1f})")
    print()

    # D vs B cross-tabulation
    print("  D range       | n  | B range         | In window?")
    print("  --------------|----|-----------------|-----------")
    for label, mask in ranges:
        n = np.sum(mask)
        if n == 0:
            continue
        B_sub = B_kramers[mask]
        n_w = np.sum((B_sub >= B_LOWER) & (B_sub <= B_UPPER))
        verdict = f"{n_w}/{n}"
        print(f"  {label:14s} | {n:2d} | [{B_sub.min():.1f}, {B_sub.max():.1f}]{' '*(8-len(f'[{B_sub.min():.1f}, {B_sub.max():.1f}]'))} | {verdict}")

    print()
    print("  FINDING 1: Most proteins fall ABOVE the stability window.")
    print(f"    {n_above}/{len(PROTEINS)} proteins ({100*n_above/len(PROTEINS):.0f}%) "
          f"have B > {B_UPPER}.")
    print(f"    Even proteins with D in [30, 1000] have B = "
          f"[{B_kramers[mask_30_1000].min():.1f}, {B_kramers[mask_30_1000].max():.1f}] "
          f"(all above {B_UPPER}).")
    print()
    print("  FINDING 2: beta_0 (= B - ln(D)) is much larger for proteins.")
    print(f"    Protein mean beta_0 = {beta_0.mean():.1f} kT "
          f"({beta_0.mean()*RT_kcal:.1f} kcal/mol)")
    print(f"    Framework beta_0 = 0.5-1.6 kT (~0.3-0.9 kcal/mol)")
    print(f"    The protein folding barrier DeltaG_f^dagger >> kT inflates beta_0.")
    print()
    print("  FINDING 3: B is NOT invariant for proteins (chevron test).")
    for (name, kf0, ku0, m_kf, m_ku, den, c_max, ref) in CHEVRON_DATA:
        c_test = np.linspace(0, c_max * 0.8, 20)
        B_test = np.log(k0_kramers / (ku0 * np.exp(m_ku * c_test)))
        delta_B = B_test.max() - B_test.min()
        frac_w = delta_B / (B_UPPER - B_LOWER) * 100
        print(f"    {name:12s}: Delta_B = {delta_B:.2f} "
              f"({frac_w:.0f}% of window width)")
    print(f"    Reason: noise = kT is externally fixed, not structurally coupled.")
    print()
    print("  FINDING 4: The stability window is NOT universal.")
    print("    It applies to systems where barrier and noise are structurally")
    print("    coupled (B invariance holds). Protein folding violates this")
    print("    because thermal noise (kT) is set by the environment, not by")
    print("    the protein's feedback architecture. The framework's B invariance")
    print("    requires sigma* to be structurally determined (Sec. 6 of EQUATIONS.md),")
    print("    which holds for ecological/climate/circuit systems but not for")
    print("    thermal barrier-crossing in proteins.")
    print()
    print("  SCOPE BOUNDARY IDENTIFIED:")
    print("    The stability window B in [1.8, 6.0] applies to dissipative systems")
    print("    where the operating noise is structurally determined by the same")
    print("    feedback architecture that creates the barrier. It does NOT apply to")
    print("    thermal systems where noise = kT is an external bath parameter.")
    print("    Proteins sit at B >> 6 because evolution selected barriers >> kT")
    print("    for kinetic stability, with no structural constraint linking")
    print("    barrier height to thermal noise intensity.")
    print()

    # Final pass/fail
    all_pass = True
    print("TEST RESULTS:")
    # Test 0: mapping verification (always passes -- it's analytical)
    print("  Test 0 (mapping):        VERIFIED (B = DeltaG_u^dagger/kT, no extra factor)")

    # Test 1: master table computed
    print(f"  Test 1 (master table):   {len(PROTEINS)} proteins, "
          f"D range [{D_vals.min():.1f}, {D_vals.max():.1e}], "
          f"B range [{B_kramers.min():.2f}, {B_kramers.max():.2f}]")

    # Test 2: stability window
    print(f"  Test 2 (window test):    {n_in_window}/{len(PROTEINS)} in window "
          f"({n_above} above, {n_below} below)")

    # Test 3: bridge analysis
    print(f"  Test 3 (bridge):         B = {slope:.2f}*ln(D) + {intercept:.2f}, R^2 = {R2:.4f}")
    print(f"                           beta_0 = {beta_0.mean():.1f} +/- {beta_0.std():.1f} "
          f"(framework: 0.5-1.6)")

    # Test 4: B invariance
    max_delta_B = 0
    for (name, kf0, ku0, m_kf, m_ku, den, c_max, ref) in CHEVRON_DATA:
        c_test = np.linspace(0, c_max * 0.8, 20)
        B_test = np.log(k0_kramers / (ku0 * np.exp(m_ku * c_test)))
        delta_B = B_test.max() - B_test.min()
        max_delta_B = max(max_delta_B, delta_B)
    print(f"  Test 4 (B invariance):   FAILS (Delta_B up to {max_delta_B:.1f}, "
          f"= {max_delta_B/(B_UPPER-B_LOWER)*100:.0f}% of window width)")

    # Test 5: sensitivity
    B_primary = np.log(k0_kramers / np.array([p[3] for p in PROTEINS]))
    n_primary = np.sum((B_primary >= B_LOWER) & (B_primary <= B_UPPER))
    print(f"  Test 5 (k_0 sensitivity): {n_primary}/{len(PROTEINS)} in window at k_0=1e6 "
          f"(best-motivated value)")

    print()
    print("CONCLUSION: Protein conformational dynamics do NOT satisfy the")
    print("stability window. This identifies a clear scope boundary: the window")
    print("applies to noise-maintained metastable systems with structurally")
    print("coupled noise, not to thermal barrier-crossing systems.")
    print()
    print("=" * 72)
    flush()
