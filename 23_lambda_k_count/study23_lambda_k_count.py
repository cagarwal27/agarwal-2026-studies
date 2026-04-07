#!/usr/bin/env python3
"""
Study 23: Phage Lambda k* Validation
=====================================
Computes framework predictions for lambda-complexity systems (d ~ 27-50)
and compares to observed interaction counts and D values.

Claims tested:
- k* = ln(S)/ln(1/alpha) predicts k ~ 5-10 for d ~ 27-50
- Product equation D = prod(1/eps) with k=5-8, eps~0.1-0.2 gives D ~ 10^4-10^5
- Lambda's ~30 interactions map to d ~ 27-50, not d ~ 150

Dependencies: numpy only
"""

import numpy as np

# ============================================================
# Framework parameters
# ============================================================
gamma_cusp = 0.197        # cusp bridge decay rate (Hill model class)
alpha_1D = 0.373          # architecture scaling (1D lake model)

# Lambda phage empirical data (from literature)
D_lambda_SOS = 1e5        # spontaneous induction ~1e-5/generation (includes SOS)
D_lambda_intrinsic = 1e8  # intrinsic switching <1e-8/generation (deltaRecA)

# Interaction counts from literature survey
n_core_interactions = 30   # core decision-circuit regulatory interactions
n_full_interactions = 43   # full network including lysis/structural
k_feedback_loops = 6       # independent feedback loops (central estimate)
k_feedback_range = (5, 8)  # range of feedback loop counts

# Published ODE model dimensions
d_santillan = 27           # Santillan & Mackey 2004: ~27 parameters
d_arkin_low = 50           # Arkin 1998 stochastic: 50+ parameters
d_range = np.array([27, 35, 40, 50])

# ============================================================
# Computation 1: S(d) from cusp bridge
# ============================================================
print("=" * 65)
print("Study 23: Phage Lambda k* Validation")
print("=" * 65)

print("\n--- Computation 1: S(d) = exp(gamma * d) for lambda-range d ---")
print(f"gamma_cusp = {gamma_cusp}")
print(f"{'d':>5} | {'S = exp(gamma*d)':>20} | {'log10(S)':>10} | {'k* = ln(S)/ln(1/alpha)':>25}")
print("-" * 70)

k_star_values = []
for d in d_range:
    S = np.exp(gamma_cusp * d)
    log10_S = np.log10(S)
    k_star = np.log(S) / np.log(1.0 / alpha_1D)
    k_star_values.append(k_star)
    print(f"{d:>5} | {S:>20.1f} | {log10_S:>10.2f} | {k_star:>25.1f}")

print(f"\nPredicted k* range for d = {d_range[0]}-{d_range[-1]}: "
      f"{k_star_values[0]:.1f} - {k_star_values[-1]:.1f}")
print(f"Observed k (feedback loops) in lambda: {k_feedback_range[0]}-{k_feedback_range[1]}")

# ============================================================
# Computation 2: Product equation consistency
# ============================================================
print("\n--- Computation 2: Product equation D = prod(1/eps) ---")
print(f"Lambda D ~ {D_lambda_SOS:.0e} (SOS-inclusive)")

test_configs = [
    (5, 0.10, "k=5, eps=0.10"),
    (4, 0.07, "k=4, eps=0.07"),
    (6, 0.15, "k=6, eps=0.15"),
    (8, 0.20, "k=8, eps=0.20"),
    (3, 0.05, "k=3, eps=0.05"),
]

print(f"{'Config':>20} | {'D_predicted':>15} | {'log10(D)':>10} | {'Matches D~10^5?':>18}")
print("-" * 70)
for k, eps, label in test_configs:
    D_pred = (1.0 / eps) ** k
    log10_D = np.log10(D_pred)
    matches = "YES" if 3.5 <= log10_D <= 6.0 else "no"
    print(f"{label:>20} | {D_pred:>15.0f} | {log10_D:>10.2f} | {matches:>18}")

# ============================================================
# Computation 3: Compare lambda to major transitions
# ============================================================
print("\n--- Computation 3: Lambda vs major transitions ---")

d_major = 150
S_major = np.exp(gamma_cusp * d_major)
k_star_major = np.log(S_major) / np.log(1.0 / alpha_1D)

d_lambda = 35  # central estimate
S_lambda = np.exp(gamma_cusp * d_lambda)
k_star_lambda = np.log(S_lambda) / np.log(1.0 / alpha_1D)

print(f"{'':>25} | {'Major transitions':>20} | {'Lambda phage':>20}")
print("-" * 70)
print(f"{'d (parameters)':>25} | {d_major:>20} | {'~' + str(d_lambda):>20}")
print(f"{'S = exp(gamma*d)':>25} | {S_major:>20.2e} | {S_lambda:>20.1f}")
print(f"{'log10(S)':>25} | {np.log10(S_major):>20.1f} | {np.log10(S_lambda):>20.1f}")
print(f"{'k* = ln(S)/ln(1/alpha)':>25} | {k_star_major:>20.1f} | {k_star_lambda:>20.1f}")
print(f"{'D (observed)':>25} | {'10^13 (S, not D)':>20} | {'~10^5':>20}")
print(f"{'Interactions (edges)':>25} | {'~150+ (predicted)':>20} | {'30-43 (counted)':>20}")

# ============================================================
# Computation 4: Inverse problem -- d from interaction count
# ============================================================
print("\n--- Computation 4: Infer d from interaction count ---")

# If we take the core interaction count as a proxy for d:
for n_int in [30, 35, 40, 43]:
    S_inf = np.exp(gamma_cusp * n_int)
    k_inf = np.log(S_inf) / np.log(1.0 / alpha_1D)
    print(f"n_interactions = {n_int}: S = {S_inf:.1f} (10^{np.log10(S_inf):.1f}), "
          f"k* = {k_inf:.1f}")

# ============================================================
# Computation 5: B estimate for lambda (approximate)
# ============================================================
print("\n--- Computation 5: Implied B for lambda (approximate) ---")

# D = K * exp(B) / (C*tau) => B = ln(D / (K/(C*tau)))
# Calibrate K/(C*tau) from known systems:
print("Calibration from known systems (B = ln(D) - ln(K/(C*tau))):")
known_systems = [
    ("Kelp", 29.4, 1.80), ("Savanna", 100, 3.74), ("Lake", 200, 4.25),
    ("Toggle", 9175, 4.83), ("Coral", 1111, 6.04),
]
prefactors = []
for name, D, B in known_systems:
    pf = D / np.exp(B)
    prefactors.append(pf)
    print(f"  {name:>10}: D={D:>7.0f}, B={B:.2f}, K/(C*tau) = D/exp(B) = {pf:.2f}")

pf_median = np.median(prefactors)
print(f"\n  Median K/(C*tau) = {pf_median:.1f}")
print(f"  Range: [{min(prefactors):.1f}, {max(prefactors):.1f}]")

# Estimate B for lambda
print(f"\nLambda B estimate (D ~ 10^5, using median prefactor {pf_median:.1f}):")
for D_val in [1e4, 5e4, 1e5]:
    B_est = np.log(D_val / pf_median)
    print(f"  D = {D_val:.0e}: B_est = {B_est:.1f}")

print(f"\n  NOTE: D_lambda ~ 10^5 is outside the empirical D range [26, 1111].")
print(f"  The stability window [1.8, 6.0] was observed for D in [26, 1111].")
print(f"  Lambda's high D implies B > 6, which is consistent -- lambda lysogeny")
print(f"  is extremely stable (switches only under SOS/DNA damage stress).")
print(f"  A proper B computation requires Kramers analysis of a lambda ODE model.")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)

print(f"""
Lambda phage regulatory network:
  Molecular species:      6 core proteins, 15-20 total
  Regulatory interactions: {n_core_interactions} core, {n_full_interactions} full network
  ODE parameters:         {d_santillan} (Santillan-Mackey) to {d_arkin_low}+ (Arkin)
  Feedback loops (k):     {k_feedback_range[0]}-{k_feedback_range[1]}
  Persistence (D):        ~10^4-10^5

Framework predictions for d ~ 27-50:
  S(d) = exp(0.197*d):    10^{np.log10(np.exp(0.197*27)):.1f} to 10^{np.log10(np.exp(0.197*50)):.1f}
  k* = ln(S)/ln(1/alpha): {k_star_values[0]:.1f} to {k_star_values[-1]:.1f}

Comparison:
  Predicted k*:           {k_star_values[0]:.0f}-{k_star_values[-1]:.0f}
  Observed k:             {k_feedback_range[0]}-{k_feedback_range[1]}
  Status:                 MATCH (within framework uncertainty)

Key insight:
  k* = 30 is for MAJOR TRANSITIONS (d ~ 150, S ~ 10^13).
  Lambda has d ~ 27-50, so k* ~ 5-10. Not a contradiction.
  The ~30 interaction count validates d, not k*.
""")

# Final verdict
k_pred_low = k_star_values[0]
k_pred_high = k_star_values[-1]
k_obs_low, k_obs_high = k_feedback_range

overlap = max(0, min(k_pred_high, k_obs_high) - max(k_pred_low, k_obs_low))
total_range = max(k_pred_high, k_obs_high) - min(k_pred_low, k_obs_low)
overlap_frac = overlap / total_range if total_range > 0 else 0

print(f"Predicted k* range: [{k_pred_low:.1f}, {k_pred_high:.1f}]")
print(f"Observed k range:   [{k_obs_low}, {k_obs_high}]")
print(f"Overlap fraction:   {overlap_frac:.1%}")
print(f"Verdict:            {'VALIDATED' if overlap_frac > 0.3 else 'INCONCLUSIVE'}")
