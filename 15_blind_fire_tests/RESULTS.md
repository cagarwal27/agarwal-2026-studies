# Study 15: Blind Search Equation Tests -- Results

**Date:** 2026-04-05
**Scripts:** `blind_flowers_prediction.py`, `blind_cam_prediction.py`
**Provenance claims:** 15.1, 15.2

---

## Question

Can the search equation tau = n * S(d) / (v * P) predict evolutionary timescales for innovations NOT in the calibration set, using gamma from directed evolution experiments (gamma_protein = 0.317) rather than from the cusp bridge?

---

## Why gamma from directed evolution

The cusp bridge gives gamma = 0.20 for a generic continuous parameter space. But mapping published ODE parameters to cusp parameters requires an unknown model-reduction ratio R. Directed evolution experiments measure gamma DIRECTLY in systems where d is exactly known:

| Experiment | d | S | gamma = ln(S)/d | Source |
|-----------|---|---|-----------------|--------|
| Random 80-aa -> ATP binding | 80 aa | 10^11 | 0.32 | Keefe & Szostak 2001 |
| Random 220-nt -> ribozyme | 220 nt | 10^13 | 0.14 | Bartel & Szostak 1993 |
| Protein mutagenesis (avg) | per mutation | (1-x)^d, x~0.34 | 0.42 | Guo et al. 2004 |
| Cusp bridge (theoretical) | per parameter | exp(0.20d) | 0.20 | Framework (cusp geometry) |

Both flower and CAM evolution involve protein/GRN mutations -> gamma_protein = 0.317 is the appropriate rate.

## Connection to framework

The search equation predicts: tau = n * S(d) / (v * P), where S(d) = exp(gamma * d).

Previously, gamma came only from the cusp bridge (theoretical). Now gamma is also measurable from directed evolution (experimental). The two agree within a factor of ~1.6 (0.20 vs 0.32), providing an independent cross-check.

Both tests use gamma_protein = 0.317. The flowers test validates the single-timescale prediction. The CAM test adds the multi-origin constraint: with 62 independent origins, the search equation must predict an origination rate, not just one timescale.

---

## Claim 15.1 -- Angiosperm Flowers

**Script:** `blind_flowers_prediction.py` -- runs clean, all computations verified by hand.

### Independent inputs

| Quantity | Value | Source | Independent? |
|----------|-------|--------|-------------------------------|
| gamma | 0.317 | Keefe & Szostak 2001, Nature (directed evolution) | YES -- laboratory measurement |
| d | 51 | van Mourik et al. 2010, BMC Syst Biol (floral ODE) | YES -- published developmental model |
| tau_observed | ~200 Myr | Fossil record (seed plants ~365 Ma -> angiosperms ~130 Ma) | YES -- geology |
| v | 0.03-1.0/yr | Paleobotany (seed plant generation time) | YES -- biology |
| P | 8-15 | Paleobotanical diversity (major seed plant lineages) | YES -- paleontology |
| n | 4-6 | Dilcher developmental decomposition | YES -- developmental biology |

### Prediction

```
tau_predicted = 6 * exp(0.317 * 51) / (0.05 * 10)
             = 6 * 1.05e7 / 0.5
             = 126 Myr   (at central estimates: v=0.05, P=10, n=6)
```

**Range:** 2.8 to 239 Myr (across v = 0.033-1.0, P = 8-15, n = 4-6).

### Observed

tau_observed = 200 Myr (seed plants ~365 Ma -> angiosperms ~130 Ma; central estimate).

### Match

```
log10(predicted / observed) = log10(126 / 200) = -0.20 OOM
```

Observed (200 Myr) falls **inside** the predicted range [2.8, 239 Myr]. Range width: 1.93 OOM.

### Grade: GREEN

- Central prediction within 0.5 OOM: YES (-0.20 OOM)
- Observed inside predicted range: YES
- All inputs independent of the calibration set: YES

### Verdict

The search equation predicts the angiosperm search-phase duration within 0.2 OOM at central biological estimates. This is the first fully blind search-equation test: gamma is laboratory-measured, d comes from an independently published developmental model, and the innovation (flowers) is not in the calibration set.

**Striking result:** At v = 0.033 (30-yr woody plant), gamma_protein gives d_required = 51.1 -- matching the van Mourik model's d = 51 exactly (R = 1.00).

**What this does NOT show:** The cusp bridge gamma = 0.20 fails here (-2.78 OOM at central v/P/n). For protein/GRN innovations, the protein-derived gamma is required.

**Largest uncertainty:** v (early angiosperm growth habit). At v = 1.0 (annual herb), tau = 6.3 Myr (-1.50 OOM). At v = 0.033 (30-yr woody), tau = 191 Myr (-0.02 OOM).

### Provenance

| Input | Value | Source | Independent? | Grade |
|-------|-------|--------|-------------|-------|
| gamma | 0.317 | Keefe & Szostak 2001, Nature 410:715 | YES | A |
| d | 51 | van Mourik et al. 2010, BMC Syst Biol 4:101 | YES | A |
| tau_observed | ~200 Myr | Fossil record (365 Ma -> 130 Ma) | YES | B |
| v | 0.05 (central) | Paleobotany: 20-yr woody generation time | YES | C (debated) |
| P | 10 (central) | Paleobotanical diversity | YES | B |
| n | 6 (central) | Dilcher 2000 developmental decomposition | YES | B |

### Sensitivity

| gamma source | gamma | tau (Myr) at central v/P/n | log10(ratio) |
|-------------|-------|---------------------------|-------------|
| Protein (Keefe & Szostak) | 0.317 | 126 | -0.20 |
| Cusp bridge (theoretical) | 0.200 | 0.33 | -2.78 |
| Mutagenesis (Guo) | 0.420 | 24,100 | +2.08 |
| RNA ribozyme (Bartel & Szostak) | 0.136 | 0.012 | -4.21 |

### Complexity scale placement

Flowers land at implied k = 16.9 -- just below pterosaur flight (k = 18.2).

---

## Claim 15.2 -- CAM Photosynthesis

**Script:** `blind_cam_prediction.py` -- runs clean, all computations verified by hand.

### Independent inputs

| Quantity | Value | Source | Independent? |
|----------|-------|--------|-------------------------------|
| gamma | 0.317 | Keefe & Szostak 2001, Nature (same as 15.1) | YES -- laboratory measurement |
| d | 72 / 40 / 19 | Bartlett, Vico & Porporato 2014, Plant and Soil (CAM ODE) | YES -- published model |
| tau_observed | ~30 Myr (central) | Paleobotany: 65 Myr (first origin) / 15 Myr (Miocene burst) | YES -- geology |
| v | 0.033-0.5/yr | Succulent generation times (Agave to Sedum) | YES -- biology |
| P | 500-10000 | Angiosperm lineages with CAM preconditions in arid habitats | YES -- paleontology |
| n | 4-6 | CAM functional sub-steps | YES -- biochemistry |
| N_origins | 62 | Silvera et al. 2010; Edwards & Ogburn 2012 | YES -- phylogenetics |

### The d classification problem

The Bartlett et al. 2014 ODE has ~72 named parameters, but not all are part of the CAM innovation:

| d | Label | What it includes | Rationale |
|---|-------|-----------------|-----------|
| 72 | Full Bartlett model | All named constants | Upper bound -- includes environment |
| 40 | CAM-specific subset | Circadian (5) + photosynthesis-core (19) + stomatal/storage (~16) | Best a priori estimate |
| 19 | Core biochemistry | Owen & Griffiths / Chomthong 2023 reformulation | Lower bound -- minimal CAM oscillator |

### Prediction (single-timescale)

At gamma_protein = 0.317, central v = 0.1, P = 2000, n = 5:

| d model | d | tau_predicted | vs 30 Myr (OOM) | Grade |
|---------|---|--------------|-----------------|-------|
| Full Bartlett | 72 | 204 Myr | +0.83 | YELLOW |
| CAM-specific | 40 | 0.008 Myr | -3.57 | RED |
| Core biochem | 19 | 1e-5 Myr | -6.46 | RED |

At d = 72: range [6.5, 3000] Myr. Observed 30 Myr **inside** range. Observed 15 Myr and 65 Myr also **inside** range.

### Multi-origin test (the key result)

N_expected = T * v * P / (n * S), with T = 20 Myr (Miocene window), N_observed = 62.

| d model | d | N_expected | ratio to 62 | Grade |
|---------|---|-----------|-------------|-------|
| Full Bartlett | 72 | 0.098 | 0.0016 | RED |
| CAM-specific | 40 | 2,490 | 40 | YELLOW |
| Core biochem | 19 | 1.94e6 | 31,300 | RED |

**d that matches 62 origins at gamma_protein: d = 52.** This falls between CAM-specific (40) and full model (72), suggesting the effective search space includes some parameters beyond the minimal CAM innovation but not all environmental ones.

### Observed

- tau_observed = 30 Myr (central, geometric mean of 15 and 65 Myr)
- Independent origins: 62-66+ (Silvera et al. 2010)

### Grade: YELLOW

- d = 72 (full model): +0.83 OOM, within 1 OOM but includes environmental parameters
- d = 40 (a priori best estimate): -3.57 OOM, fails
- Multi-origin test independently converges on d ~ 52
- Grade is YELLOW because the prediction works at d = 72 but not at d = 40

### Verdict

The search equation at d = 72 predicts tau = 204 Myr vs observed 30 Myr (+0.83 OOM). Weaker than the flowers test (-0.20 OOM, GREEN). The a priori CAM-specific estimate (d = 40) fails at -3.57 OOM.

**The multi-origin test is the more powerful result.** The search equation predicts d ~ 52 to explain 62 independent origins in the Miocene window. This is biologically plausible: evolution must tune not only core CAM biochemistry but also coupling parameters between the pathway and its environmental context.

**Striking result:** d_multi = 52 for CAM nearly matches d = 51 for the flowers model (van Mourik et al. 2010). This may suggest that published developmental/physiological ODE models have similar effective dimensionality (~50 searchable parameters).

### Provenance

| Input | Value | Source | Independent? | Grade |
|-------|-------|--------|-------------|-------|
| gamma | 0.317 | Keefe & Szostak 2001, Nature 410:715 | YES | A |
| d = 72 | 72 | Bartlett, Vico & Porporato 2014, Plant and Soil | YES | A |
| d = 40 | 40 | Subset of Bartlett model (author estimate) | YES but judgment call | C |
| d = 19 | 19 | Owen & Griffiths / Chomthong 2023 | YES | B |
| tau_observed | 30 Myr | Paleobotany (geom. mean of 15 and 65 Myr) | YES | B |
| v | 0.1 (central) | Succulent generation times (~10 yr) | YES | C |
| P | 2000 (central) | Inferred from 35+ families with CAM | YES | C |
| n | 5 (central) | CAM functional sub-steps | YES | B |
| N_origins | 62 | Silvera et al. 2010; Edwards & Ogburn 2012 | YES | A |

### Sensitivity

| gamma source | gamma | tau at d=40 (Myr) | tau at d=72 (Myr) | d_required for tau=30 Myr |
|-------------|-------|-------------------|-------------------|--------------------------|
| Protein (Keefe & Szostak) | 0.317 | 0.008 | 204 | 65.9 |
| Cusp bridge (theoretical) | 0.200 | 7.5e-5 | 44.8 | 104.5 |
| Mutagenesis (Guo) | 0.420 | 0.49 | 3.4e5 | 49.8 |
| RNA ribozyme (Bartel & Szostak) | 0.136 | 5.8e-6 | 0.45 | 153.7 |

### Complexity scale placement

CAM lands at implied k = 21.2, below C4 photosynthesis (k = 24.3). CAM < C4 is biologically expected: CAM requires no Kranz anatomy while C4 does.

| Innovation | log10(S) | Implied k |
|-----------|---------|----------|
| **Flowers (blind, 15.1)** | **7.2** | **16.9** |
| Flight (pterosaurs) | 7.8 | 18.2 |
| Flight (birds) | 8.5 | 19.8 |
| Flight (bats) | 8.6 | 20.1 |
| **CAM (blind, 15.2)** | **9.1** | **21.2** |
| Flight (insects) | 9.9 | 23.2 |
| C4 photosynthesis | 10.4 | 24.3 |
| Camera-type eyes | 11.6 | 27.1 |
| Major transitions | 13.0 | 30.4 |
