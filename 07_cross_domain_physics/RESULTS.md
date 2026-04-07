# Study 07: Cross-Domain Physics

**Date:** 2026-04-06
**Scripts:** `thermohaline_kramers.py`, `financial_cusp_kramers.py`, `soviet_kuran_kramers.py`, `power_grid_kramers.py`

## Question

Does the Kramers equation apply in non-biological domains (climate, finance, political science, power systems)?

## Data

Four systems tested, spanning four domains. Quality varies from published models with 0 free parameters (thermohaline) to uncalibrated illustrative models (Soviet, financial).

## Key Results

| Domain | System | K | Key finding | Grade |
|--------|--------|------|-------------|-------|
| Climate | Thermohaline (AMOC) | 0.55 | 0 free params; D-O event spacing comparison | YELLOW |
| Finance | Cusp catastrophe | 0.55 | K_eff=0.55 when 2*DeltaV/s^2 > 2 | RED |
| Political science | Soviet collapse | -- | tau=3.9 yr; 4 free params, 0 calibrated | RED |
| Power systems | SMIB swing equation | ~1.0 | Equal curvature regime; kappa^(3/2) to 0.1% | YELLOW |

## Parameters and Sources

### Thermohaline (thermohaline_kramers.py)

| Parameter | Value | Source |
|-----------|-------|--------|
| mu^2 | 4.0 | Cessi 1994 |
| t_d | 219 yr | Cessi 1994 |
| K | 0.55 | Computed, 0 free params |

Source: Cessi 1994 AMOC model. Citations are informal (no page/table references).

### Financial (financial_cusp_kramers.py)

| Parameter | Value | Source |
|-----------|-------|--------|
| (q, a) | 6 generic sets | **No source -- generic cusp params** |
| K_eff | 0.55 | Computed for 2*DeltaV/s^2 > 2 |

**RED: Generic parameters, NOT calibrated to Diks-Wang S&P data or any empirical financial data.**

### Soviet (soviet_kuran_kramers.py)

| Parameter | Value | Source |
|-----------|-------|--------|
| S_lo | 0.05 | **Free parameter, uncalibrated** |
| S_mid | 0.35 | **Free parameter, uncalibrated** |
| S_hi | 0.90 | **Free parameter, uncalibrated** |
| gamma | 1.0 | **Free parameter, uncalibrated** |

**RED: 4 free parameters, 0 calibrated. The ODE conversion from Kuran's preference falsification theory is unpublished.**

### Power grid (power_grid_kramers.py)

| Parameter | Value | Source |
|-----------|-------|--------|
| Brazilian grid params | See script | Ritmeester & Meyer-Ortmanns 2022 |
| K | ~1.0 | Computed (equal curvature limit) |

Source: Ritmeester & Meyer-Ortmanns 2022. D_obs is undefined because power grid dynamics are SOC, not Kramers escape.

## Interpretation

- **Thermohaline:** Kramers works with K=0.55 and 0 free parameters using the Cessi 1994 model. Provides D-O event spacing as an observational comparison.
- **Financial:** K_eff=0.55 emerges for sufficiently deep wells (2*DeltaV/s^2 > 2), but the 6 (q,a) parameter sets are generic -- NOT calibrated to Diks-Wang S&P data or any real market.
- **Soviet:** Kuran's preference falsification theory is converted to a cubic ODE, but this conversion is unpublished. All 4 parameters are free and uncalibrated. The tau=3.9 yr result is illustrative only.
- **Power grid:** K approaches 1.0 in the equal-curvature regime with kappa^(3/2) scaling to 0.1%. However, D_obs is undefined because power grid failures follow SOC (self-organized criticality), not Kramers escape.

## Selection Filter Interpretation: Power Grid as Correct Negative Prediction

The power grid result is more informative than "D_obs undefined." It is evidence for the selection filter hypothesis.

The framework predicts that noise-driven bistable persistence requires sigma_process ~ sigma* (equivalently, B inside the stability window [1.8, 6.0]). The power grid violates this condition: at measured noise levels (sigma = 0.01-0.05), the barrier-to-noise ratio 2*DeltaV/sigma^2 ~ 40-1000, far above the stability window. The barrier (set by generator impedance and grid topology) and the noise (set by wind variability and demand fluctuations) have **completely independent physical origins**. They are decoupled.

The framework's prediction: no noise-driven Kramers escape should be observable. This is exactly what happens -- real blackouts are cascading topology failures (SOC), not noise-driven barrier crossing. The Kramers math works perfectly on the SMIB equation (K->1.0, kappa^(3/2) scaling to 0.1%), but the physics doesn't couple barrier to noise, so the phenomenon doesn't occur.

Contrast with the 11 verified systems (ecology, gene circuits, cancer, climate, superconducting, nanomagnetism): in every case where sigma_process ~ sigma* (B in [1.8, 6.0]), noise-driven persistence IS observed. The power grid is the one tested system where sigma != sigma* -- and the prediction "no Kramers escape" is confirmed.

This is the selection filter in action: we observe noise-driven bistable persistence only in systems where the feedback architecture couples barrier to noise (B in stability window). Systems where barrier and noise are decoupled (power grid) don't exhibit the phenomenon, exactly as predicted.

## Conclusions

Kramers equation applies cleanly in climate (thermohaline, 0 free parameters). The power grid provides a valuable correct negative prediction for the selection filter hypothesis. The financial and Soviet scripts are illustrative only (RED grade) due to uncalibrated parameters and unpublished model conversions.
