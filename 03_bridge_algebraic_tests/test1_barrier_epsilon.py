#!/usr/bin/env python3
"""
Test 1: Does the per-channel barrier map to ε?
Tests whether ΔΦ_ch1/ΔΦ_ch2 = ln(1/ε₁)/ln(1/ε₂) across a grid of calibrations.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import itertools
import sys

# ============================================================
# Original lake model parameters
# ============================================================
b = 0.8
r = 1.0
q = 8
h = 1.0
a = 0.326588

x_clear = 0.409217
x_sad   = 0.978152
x_turb  = 1.634126

def f_lake(x):
    return a - b*x + r * x**q / (x**q + h**q)

# Total flux at equilibrium (= total loss at equilibrium)
total_flux_eq = a + r * x_clear**q / (x_clear**q + h**q)
print(f"Total flux at equilibrium: {total_flux_eq:.6f}")
print(f"b*x_clear = {b*x_clear:.6f}")

# Barrier of original model
DeltaPhi_orig, _ = quad(lambda x: -f_lake(x), x_clear, x_sad)
print(f"Original barrier ΔΦ = {DeltaPhi_orig:.8f}")

# ============================================================
# Channel functional forms
# ============================================================
def g1(x, K1):
    """Hill-type (saturating, steep)"""
    return x**4 / (x**4 + K1**4)

def g2(x, K2):
    """Michaelis-Menten (saturating, gradual)"""
    return x / (x + K2)

# ============================================================
# Step 1: Calibration grid
# ============================================================
eps1_vals = [0.02, 0.05, 0.10, 0.15, 0.20]
eps2_vals = [0.02, 0.05, 0.10, 0.15, 0.20]
K1_vals   = [0.3, 0.5, 0.8, 1.0, 1.5]
K2_vals   = [0.2, 0.5, 1.0, 2.0]

results = []
total_attempted = 0
skipped_b0 = 0
skipped_bistable = 0

for eps1, eps2, K1, K2 in itertools.product(eps1_vals, eps2_vals, K1_vals, K2_vals):
    total_attempted += 1

    # Compute channel coefficients
    g1_eq = g1(x_clear, K1)
    g2_eq = g2(x_clear, K2)

    c1 = eps1 * total_flux_eq / g1_eq
    c2 = eps2 * total_flux_eq / g2_eq

    # Background flushing
    b0 = (b * x_clear - c1 * g1_eq - c2 * g2_eq) / x_clear

    if b0 <= 0:
        skipped_b0 += 1
        continue

    # Modified model
    def f_mod(x, _b0=b0, _c1=c1, _c2=c2, _K1=K1, _K2=K2):
        return a + r * x**q / (x**q + h**q) - _b0*x - _c1*g1(x, _K1) - _c2*g2(x, _K2)

    # Find equilibria - scan for sign changes
    x_scan = np.linspace(0.01, 3.0, 5000)
    f_scan = np.array([f_mod(x) for x in x_scan])

    sign_changes = []
    for i in range(len(f_scan)-1):
        if f_scan[i] * f_scan[i+1] < 0:
            try:
                root = brentq(f_mod, x_scan[i], x_scan[i+1])
                sign_changes.append(root)
            except:
                pass

    # Need at least 3 roots for bistability (stable-unstable-stable)
    if len(sign_changes) < 3:
        skipped_bistable += 1
        continue

    # Check stability: f'(x) < 0 at stable, f'(x) > 0 at saddle
    x_eq_new = sign_changes[0]
    x_sad_new = sign_changes[1]
    x_turb_new = sign_changes[2]

    # Numerical derivative check
    dx = 1e-6
    fprime_eq = (f_mod(x_eq_new + dx) - f_mod(x_eq_new - dx)) / (2*dx)
    fprime_sad = (f_mod(x_sad_new + dx) - f_mod(x_sad_new - dx)) / (2*dx)

    if fprime_eq >= 0 or fprime_sad <= 0:
        skipped_bistable += 1
        continue

    # Step 2: Compute per-channel barrier contributions
    def f_base(x, _b0=b0):
        return a + r * x**q / (x**q + h**q) - _b0*x

    DeltaPhi_total, _ = quad(lambda x: -f_mod(x), x_eq_new, x_sad_new)
    DeltaPhi_base, _  = quad(lambda x: -f_base(x), x_eq_new, x_sad_new)
    DeltaPhi_ch1, _   = quad(lambda x: c1*g1(x, K1), x_eq_new, x_sad_new)
    DeltaPhi_ch2, _   = quad(lambda x: c2*g2(x, K2), x_eq_new, x_sad_new)

    # Sanity check: decomposition
    decomp_check = abs(DeltaPhi_total - (DeltaPhi_base + DeltaPhi_ch1 + DeltaPhi_ch2))

    if DeltaPhi_total <= 0 or DeltaPhi_ch1 <= 0 or DeltaPhi_ch2 <= 0:
        skipped_bistable += 1
        continue

    # Step 3: The critical test
    barrier_ratio = DeltaPhi_ch1 / DeltaPhi_ch2
    epsilon_ratio = np.log(1/eps1) / np.log(1/eps2)

    ratio_ch1 = DeltaPhi_ch1 / np.log(1/eps1)
    ratio_ch2 = DeltaPhi_ch2 / np.log(1/eps2)

    rel_error = abs(barrier_ratio - epsilon_ratio) / epsilon_ratio
    match = rel_error < 0.05

    # Shape factors
    shape1 = quad(lambda x: g1(x, K1), x_eq_new, x_sad_new)[0] / (g1(x_eq_new, K1) * (x_sad_new - x_eq_new))
    shape2 = quad(lambda x: g2(x, K2), x_eq_new, x_sad_new)[0] / (g2(x_eq_new, K2) * (x_sad_new - x_eq_new))

    results.append({
        'eps1': eps1, 'eps2': eps2, 'K1': K1, 'K2': K2,
        'b0': b0, 'c1': c1, 'c2': c2,
        'x_eq': x_eq_new, 'x_sad': x_sad_new, 'x_turb': x_turb_new,
        'DeltaPhi_total': DeltaPhi_total,
        'DeltaPhi_base': DeltaPhi_base,
        'DeltaPhi_ch1': DeltaPhi_ch1, 'DeltaPhi_ch2': DeltaPhi_ch2,
        'decomp_check': decomp_check,
        'barrier_ratio': barrier_ratio, 'epsilon_ratio': epsilon_ratio,
        'ratio_ch1': ratio_ch1, 'ratio_ch2': ratio_ch2,
        'rel_error': rel_error, 'match': match,
        'shape1': shape1, 'shape2': shape2,
    })

print(f"\n{'='*70}")
print(f"CALIBRATION GRID SUMMARY")
print(f"{'='*70}")
print(f"Total attempted:     {total_attempted}")
print(f"Skipped (b0 <= 0):   {skipped_b0}")
print(f"Skipped (not bistable): {skipped_bistable}")
print(f"Valid calibrations:  {len(results)}")

# ============================================================
# Step 3: Report the critical test
# ============================================================
print(f"\n{'='*70}")
print(f"CRITICAL TEST: barrier_ratio vs epsilon_ratio")
print(f"{'='*70}")

# Sort by rel_error for readability
results.sort(key=lambda r: r['rel_error'])

print(f"\n{'ε₁':>5} {'ε₂':>5} {'K₁':>5} {'K₂':>5} {'ΔΦ_ch1':>10} {'ΔΦ_ch2':>10} {'barr_ratio':>11} {'ε_ratio':>9} {'rel_err':>8} {'match':>6}")
print("-"*90)

matches = 0
for r in results:
    tag = "YES" if r['match'] else "no"
    if r['match']:
        matches += 1
    print(f"{r['eps1']:>5.2f} {r['eps2']:>5.2f} {r['K1']:>5.1f} {r['K2']:>5.1f} "
          f"{r['DeltaPhi_ch1']:>10.6f} {r['DeltaPhi_ch2']:>10.6f} "
          f"{r['barrier_ratio']:>11.6f} {r['epsilon_ratio']:>9.6f} "
          f"{r['rel_error']:>8.4f} {tag:>6}")

print(f"\nMatches (within 5%): {matches}/{len(results)}")

# Summary statistics
rel_errors = [r['rel_error'] for r in results]
print(f"\nRelative error statistics:")
print(f"  Mean:   {np.mean(rel_errors):.4f}")
print(f"  Median: {np.median(rel_errors):.4f}")
print(f"  Min:    {np.min(rel_errors):.4f}")
print(f"  Max:    {np.max(rel_errors):.4f}")
print(f"  Std:    {np.std(rel_errors):.4f}")

# ============================================================
# Step 4: Shape analysis
# ============================================================
print(f"\n{'='*70}")
print(f"SHAPE ANALYSIS")
print(f"{'='*70}")

# Analyze shape factor relationship to mismatch
print(f"\n{'ε₁':>5} {'ε₂':>5} {'K₁':>5} {'K₂':>5} {'shape1':>8} {'shape2':>8} {'sh_ratio':>9} {'rel_err':>8}")
print("-"*70)
for r in results[:20]:  # show top 20 by rel_error (best matches first)
    sh_ratio = r['shape1'] / r['shape2']
    print(f"{r['eps1']:>5.2f} {r['eps2']:>5.2f} {r['K1']:>5.1f} {r['K2']:>5.1f} "
          f"{r['shape1']:>8.4f} {r['shape2']:>8.4f} {sh_ratio:>9.4f} {r['rel_error']:>8.4f}")

print("\n... (worst matches):")
for r in results[-10:]:
    sh_ratio = r['shape1'] / r['shape2']
    print(f"{r['eps1']:>5.2f} {r['eps2']:>5.2f} {r['K1']:>5.1f} {r['K2']:>5.1f} "
          f"{r['shape1']:>8.4f} {r['shape2']:>8.4f} {sh_ratio:>9.4f} {r['rel_error']:>8.4f}")

# Check ratio_ch1 vs ratio_ch2 consistency
print(f"\nσ²/2 consistency check (ratio_ch_i = ΔΦ_ch_i / ln(1/ε_i)):")
print(f"{'ε₁':>5} {'ε₂':>5} {'K₁':>5} {'K₂':>5} {'ratio_ch1':>10} {'ratio_ch2':>10} {'diff':>8}")
print("-"*65)
for r in results[:10]:
    diff = abs(r['ratio_ch1'] - r['ratio_ch2']) / max(r['ratio_ch1'], r['ratio_ch2'])
    print(f"{r['eps1']:>5.2f} {r['eps2']:>5.2f} {r['K1']:>5.1f} {r['K2']:>5.1f} "
          f"{r['ratio_ch1']:>10.6f} {r['ratio_ch2']:>10.6f} {diff:>8.4f}")

# Correlation between shape factor ratio and rel_error
shape_ratios = [r['shape1']/r['shape2'] for r in results]
corr = np.corrcoef([abs(np.log(sr)) for sr in shape_ratios], rel_errors)[0,1]
print(f"\nCorrelation between |ln(shape1/shape2)| and rel_error: {corr:.4f}")

# ============================================================
# Step 5: Linear channel control
# ============================================================
print(f"\n{'='*70}")
print(f"STEP 5: LINEAR CHANNEL CONTROL (g₁=g₂=x)")
print(f"{'='*70}")

results_linear = []
total_lin = 0
valid_lin = 0

for eps1, eps2 in itertools.product(eps1_vals, eps2_vals):
    total_lin += 1

    # With g(x) = x: c_i = eps_i * total_flux / x_eq
    c1_lin = eps1 * total_flux_eq / x_clear
    c2_lin = eps2 * total_flux_eq / x_clear
    b0_lin = b - c1_lin - c2_lin

    if b0_lin <= 0:
        continue

    def f_lin(x, _b0=b0_lin, _c1=c1_lin, _c2=c2_lin):
        return a + r * x**q / (x**q + h**q) - _b0*x - _c1*x - _c2*x

    # Note: f_lin = a + r*x^q/(x^q+h^q) - (b0+c1+c2)*x = a + r*x^q/(x^q+h^q) - b*x
    # So with linear channels, the model is IDENTICAL to the original!
    # The barrier decomposition is trivial: ΔΦ_ch_i = c_i * ∫x dx = c_i*(x_sad²-x_eq²)/2

    x_eq_lin = x_clear  # same equilibria as original
    x_sad_lin = x_sad

    DeltaPhi_ch1_lin = c1_lin * (x_sad_lin**2 - x_eq_lin**2) / 2
    DeltaPhi_ch2_lin = c2_lin * (x_sad_lin**2 - x_eq_lin**2) / 2

    barrier_ratio_lin = DeltaPhi_ch1_lin / DeltaPhi_ch2_lin  # = c1/c2 = eps1/eps2
    epsilon_ratio_lin = np.log(1/eps1) / np.log(1/eps2)

    rel_error_lin = abs(barrier_ratio_lin - epsilon_ratio_lin) / epsilon_ratio_lin

    # Analytical: barrier_ratio = c1/c2 = eps1/eps2
    # epsilon_ratio = ln(1/eps1)/ln(1/eps2)
    # These are equal only if eps1/eps2 = ln(1/eps1)/ln(1/eps2)
    # i.e., eps1*ln(1/eps2) = eps2*ln(1/eps1)

    valid_lin += 1
    results_linear.append({
        'eps1': eps1, 'eps2': eps2,
        'c1': c1_lin, 'c2': c2_lin, 'b0': b0_lin,
        'barrier_ratio': barrier_ratio_lin,  # = eps1/eps2
        'epsilon_ratio': epsilon_ratio_lin,
        'rel_error': rel_error_lin,
        'match': rel_error_lin < 0.05,
    })

results_linear.sort(key=lambda r: r['rel_error'])

print(f"Valid linear calibrations: {valid_lin}/{total_lin}")
print(f"\n{'ε₁':>5} {'ε₂':>5} {'c₁/c₂':>8} {'barr_ratio':>11} {'ε_ratio':>9} {'rel_err':>8} {'match':>6}")
print("-"*60)
lin_matches = 0
for r in results_linear:
    tag = "YES" if r['match'] else "no"
    if r['match']:
        lin_matches += 1
    print(f"{r['eps1']:>5.2f} {r['eps2']:>5.2f} {r['c1']/r['c2']:>8.4f} "
          f"{r['barrier_ratio']:>11.6f} {r['epsilon_ratio']:>9.6f} "
          f"{r['rel_error']:>8.4f} {tag:>6}")

print(f"\nLinear matches (within 5%): {lin_matches}/{valid_lin}")
print(f"\nNote: With linear channels, barrier_ratio = c₁/c₂ = ε₁/ε₂")
print(f"      epsilon_ratio = ln(1/ε₁)/ln(1/ε₂)")
print(f"      These are equal only when ε₁ = ε₂ (trivially).")
print(f"      For ε₁ ≠ ε₂: ε₁/ε₂ ≠ ln(1/ε₁)/ln(1/ε₂) in general.")

# Analytical verification for linear case
print(f"\nAnalytical check for linear case:")
print(f"  ε₁/ε₂ vs ln(1/ε₁)/ln(1/ε₂):")
for e1 in [0.02, 0.05, 0.10, 0.20]:
    for e2 in [0.02, 0.05, 0.10, 0.20]:
        if e1 != e2:
            ratio_direct = e1/e2
            ratio_log = np.log(1/e1)/np.log(1/e2)
            print(f"  ε₁={e1:.2f}, ε₂={e2:.2f}: ε₁/ε₂ = {ratio_direct:.4f}, ln(1/ε₁)/ln(1/ε₂) = {ratio_log:.4f}")

# ============================================================
# VERDICT
# ============================================================
print(f"\n{'='*70}")
print(f"VERDICT")
print(f"{'='*70}")

pct_match = matches / len(results) * 100
mean_err = np.mean(rel_errors)

print(f"\nNonlinear channels: {matches}/{len(results)} = {pct_match:.1f}% match within 5%")
print(f"Mean relative error: {mean_err:.4f} ({mean_err*100:.1f}%)")
print(f"Linear channel control: {lin_matches}/{valid_lin} match within 5%")

if pct_match > 80:
    print(f"\n>> VERDICT: The per-channel barrier contribution maps to ln(1/εᵢ).")
    print(f">> The bridge is algebraic.")
elif pct_match < 20:
    print(f"\n>> VERDICT: The per-channel barrier contribution does NOT map to ln(1/εᵢ).")
    print(f">> The bridge is a duality, not a per-channel decomposition.")
else:
    print(f"\n>> VERDICT: The mapping works for specific conditions but fails for others.")
    # Identify what conditions matter
    matched = [r for r in results if r['match']]
    unmatched = [r for r in results if not r['match']]
    if matched:
        avg_sh_ratio_match = np.mean([r['shape1']/r['shape2'] for r in matched])
        avg_sh_ratio_no = np.mean([r['shape1']/r['shape2'] for r in unmatched]) if unmatched else float('nan')
        print(f"   Mean shape_ratio for matches: {avg_sh_ratio_match:.4f}")
        print(f"   Mean shape_ratio for non-matches: {avg_sh_ratio_no:.4f}")
