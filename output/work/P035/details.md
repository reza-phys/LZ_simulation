# P035 — Recasting XENONnT and PandaX-4T to inelastic dark matter: the exclusion they could publish tomorrow

*Research record. Simulated arXiv date 2026-09-09. Category XEXP (hep-ph). Author profile: recasting phenomenologists. Competes with P005 (which gave expected counts; here we give the formal limits, the discovery case and the existing XENON1T constraint). All numbers are produced by `output/code/P035_recast_xenon.py` (run from the simulation root with `.venv/bin/python`); tables in `output/work/P035/`, figures in `output/work/P035/figures/`, console record in `run_log.txt`.*

## 1. Motivation and framework

LZ (arXiv:2609.02823) fits its 248 keV event with 1.0 (+1.4, −0.7) signal events in 2.84 t·yr and publishes two-sided 90% intervals for inelastic O₁ˢ (Fig. 6 / Fig. S7) up to δ = 350 keV. P005 showed that XENONnT (3.1 t·yr) and PandaX-4T (1.54 t·yr) expect essentially nothing inside their published WIMP-search ROIs (≤100 keV) but ≈1.5 events with LZ-like 270 keV ROIs. P021 found that the O₁ˢ likelihood keeps rising beyond LZ's grid to δ ≈ 380 keV at 1 TeV. The obvious next question, asked here from the point of view of the two other collaborations, is: *what limit (or what discovery) would a high-energy extension of the existing XENONnT and PandaX-4T data sets produce, and does anything already published constrain LZ's interval?*

**Target universality.** All three detectors use natural xenon and the same halo convention, so dR/dE_R per tonne-year is detector-independent; expected counts are

  N_X(δ, κ) = κ · X · ∫ dE_R (dR/dE_R)_unit(E_R; δ) ε_X(E_R),   (1)

with X the exposure (t·yr), ε_X the NR efficiency, and κ ≡ (c₁ˢ m_v²)² the LZ/Anand isoscalar coupling squared (WimPyDD c⁰ = 2/m_v², P003 convention via `lz.wd_c_from_anand`). The cross-section follows LZ's supplement ("Interpretation of O₁ as a cross-section"):

  σ_SI = κ μ_N² / (π m_v⁴) · (ħc)²,  σ_SI(κ = 1, 1 TeV) = 2.966 × 10⁻³⁸ cm²   (2)

(μ_N = 0.9374 GeV; consistent with P021: κ̂ = 2.19 ↔ 6.5 × 10⁻³⁸ cm²). Multiplying by 3.2 gives the vector-coupling cross-section (not used).

## 2. Inputs

### 2.1 From the LZ paper
| Quantity | Value | Source |
|---|---|---|
| Exposure | 2.84 t·yr | abstract; `lz.LZ["exposure_tyr"]` |
| Best-fit signal (L10ˢ, 1 TeV) | 1.0 (+1.4, −0.7) events | Table I |
| NR efficiency | 50% at 5.4 and 269.9 keV, 0.96 plateau | Fig. S2 + caption; l.117 |
| High-S1c background | 0.0106 ± 0.0008 (whole S1c > 500 phd panel) | Fig. 5 caption |
| O₁ˢ → σ_SI recast | (c₁ˢ m_v²)² = σ π m_v⁴/μ² | supplement l.800–808 |
| Fig. S7 | two-sided 90% intervals; XENON1T 2018 (green) and PandaX-4T 2021 (blue) inelastic limits, derived by PICO | l.810–817 |
| Introduction citations | XENONnT PRL 135, 221003 (2025); PandaX-4T PRL 134, 011805 (2025) | l.28, l.347 |

### 2.2 From the corpus
| Item | Source |
|---|---|
| LZ Fig. 6/S7 interval edges (δ = 0–350 keV, 1 TeV) | `output/work/P007/lz_intervals_digitised.csv` |
| LZ best-fit κ̂(δ), 68/90% bands, Z(δ) (1 TeV O₁ˢ), 'baseline' (12-day annual) and 'sun' configurations | `output/work/P021/P021_scan_baseline.csv`, `P021_scan_all.csv` |
| Digitised Fig. S7 curves (XENON1T green to 230.8 keV; PandaX-4T blue to 303.3 keV) | `output/work/P015/figS7_curves.json` |
| Efficiency model of Fig. S2 (erf widths 2.5 / 12 keV) | P005 script |
| High-energy NR-band background b_H = 5.7 × 10⁻⁴ per 2.84 t·yr in 200–270 keV; resolution σ_E = 11√(E/248) keV | P016 / P021 / P009 |
| Expected counts in other xenon TPCs (0.9 plateau): 1.02/0.51 at 270 keV | P005 |

### 2.3 Recalled inputs (flagged)
| Item | Value | Reliability |
|---|---|---|
| XENONnT WIMP exposure | 3.1 t·yr | certain (in the cited title) |
| PandaX-4T WIMP exposure | 1.54 t·yr | certain (in the cited title) |
| XENON1T 2018 exposure and NR ROI | 1.0 t·yr; cS1 3–70 PE ≈ 4.9–40.9 keV_NR | likely / likely |
| (ħc)² | 0.3894 × 10⁻²⁷ GeV² cm² | certain |
| Poisson 90% UL constants (2.303 for n = 0; 3.890 for n = 1), FC single-count interval [0.11, 4.36] | certain / likely (reproduced numerically) |
| Wilks/Cowan asymptotics, χ² quantiles 2.30 / 4.61 (2 dof) | certain |

### 2.4 Assumptions (stated)
1. **Efficiencies.** ε_LZlike(E) = 0.96 Φ((E−5.4)/2.5)[1 − Φ((E−269.9)/12)] for both experiments (an extension with LZ-like S1c/S2c ROI and acceptance); a variant ε_hi(E) = 0.85 Φ((E−5)/2)[1 − Φ((E−300)/12)] (lower plateau, higher edge); hard-edge variants 0.96 Φ((E−5.4)/2.5) Θ(E_max − E) for E_max = 150–300 keV. The other experiments' true high-energy efficiencies are unpublished; counts scale linearly with the plateau.
2. **Background in the high-energy NR band**, per 2.84 t·yr, scaled by exposure/2.84: 2 × 10⁻⁴ (within ±2σ of the NR median, S1c > 500 phd), 5.7 × 10⁻⁴ (200–270 keV band, P016), 10⁻³ (round), 0.0106 (whole panel). No ER leakage or MSSI specific to the other detectors is modelled.
3. **Halo.** Baxter-2021 SHM. Baseline: `lz.wd_halo()` with no day, which is WimPyDD's default single-day halo (day = modulation phase − T/4, the mean-Earth-speed or "Sun-frame" halo, v_max = 794.6 km/s). P021 labelled this configuration 'sun' and found it to be the one that reproduces LZ's Fig. 6 upper edge at 350 keV. Bracket: 12-monthly-day annual average (P021's 'baseline'), which contains the June tail to 809 km/s; computed at 1 TeV for δ = 300–385 keV (nine points, because it is ~10× slower per point).
4. **Spectra.** WimPyDD 2.0.4 shell-model O₁ˢ spectra, `WD.diff_rate` via `lz.wd_rate`, E_true grid 1–450 keV in 1.5 keV steps; δ grid 100–290 keV (10 keV) and 300 keV to the June ceiling μv²_max/2 (5 keV); m = 400, 1000, 4000 GeV (ceilings 341.3 / 397.1 / 432.5 keV).

## 3. Method

### 3.1 Counts and limits
For each (m, δ, ε): N_unit = X ∫ rate · ε dE (events per unit κ). Zero-event 90% CL: κ < 2.303/N_unit (Poisson; independent of b for n = 0). One event: κ < 3.890/N_unit (Poisson-with-background, b → 0). Feldman–Cousins unified intervals are computed numerically (s grid 0–20 step 0.01, n ≤ 60) for each exposure-scaled background.

### 3.2 Comparison with LZ, normalisation-consistently
The ratio r = κ_lim/κ_1event = 2.303/(N_unit,X/N_unit,LZ) depends only on exposure and efficiency ratios (for ε_LZlike: r = 2.303/(3.1/2.84) = 2.11 for XENONnT, 2.303/(4.64/2.84) = 1.41 for the combination), so it is free of the spectrum normalisation. The fraction of LZ's 90% interval excluded (measured in ln κ) is f = [ln κ₉₀ʜɪ − ln(κ̂ r)]/[ln κ₉₀ʜɪ − ln κ₉₀ʟᴏ] with κ̂ and the edges taken from the same P021 configuration ('sun' or annual), so that the normalisation cancels.

### 3.3 Significance if the others see nothing
P021's single-event asymptotic statistic, q₀ = 2[ln(f/b) − 1 + b/f], depends on the accepted-signal density f at 248 keV per unit expected signal and the background density b. Adding zero-count exposures multiplies the total expectation per unit κ by R = N_tot/N_LZ and leaves f/b → f/(Rb); we solve f/b from P021's Z(δ) and recompute Z with R.

### 3.4 Discovery case
(a) Counting: XENONnT sees one event in 200–270 keV; P(≥1 | b_X) for XENONnT alone and P(≥2 | b_LZ + b_X) for the combined two-event sample, converted to one-sided Z. (b) Joint extended likelihood in observed energy (1 TeV, ε_LZlike in both): ln L(κ, δ) = Σ_i [ln(κ a_i(E_i) + b_i)] − κ Σ_i (S_i + Z_i) − Σ B_i, with a_i the smeared accepted spectrum (Gaussian σ_E = 11√(E/248) keV) at the event energy (248 keV in LZ; E₂ = 200, 230, 260 keV in XENONnT), S_i the accepted signal in 200–270 keV, Z_i the accepted signal in the empty 125–200 keV bin (zero counts imposed in both detectors), b_i = B_i/70 keV the flat backgrounds (B_LZ = 5.7 × 10⁻⁴, B_X = B_LZ × 3.1/2.84). κ on a log grid 10⁻⁹–10³ (1201 points). Regions from Δ(−2 ln L) ≤ 2.30 / 4.61 (2 dof); profile Z(δ) = √(2[ln L_max(δ) − ln L₀]). Energies below 125 keV are not modelled (P016's low-energy bins matter only for δ ≲ 200 keV, cf. P021); the scan is shown for δ ≥ 200 keV.

### 3.5 XENON1T 2018 and PandaX-4T 2021
The digitised green curve (P015) is compared with the P007-digitised LZ interval edges (log-linear interpolation in δ; at 231 keV between the 200 and 250 keV points). We also (i) recast XENON1T naively — 1.0 t·yr, 0.9 plateau, hard edge 40.9 keV, zero background, 2.30 events — to test whether the *shape* and *endpoint* of the PICO-derived curve follow from kinematics, and (ii) compute the lower recoil edge E₋(δ_end) at 1 TeV for several v_max choices to infer the ROI edges encoded by the curve endpoints.

## 4. Results

### 4.1 Validation (`validation_vs_P021.csv`)
Against P021's 'sun' configuration (same halo), N_unit,LZ agrees to 1.000–1.029 for δ ≤ 350 keV and 1.12–1.51 at 370–385 keV (our finer 1.5 keV energy grid resolves the narrow window better; P021 used 3 keV). Against P021's annual configuration, our 12-day annual bracket agrees to 1.006–1.023 (≤350) and 1.09–1.97 (370–385). Annual/Sun-frame ratio of N_unit: 1.08 (300), 1.15 (320), 1.37 (340), 1.71 (350), 2.73 (360), 6.6 (370), 9.7 (375), 10.4 (380), 12.2 (385 keV) — the halo-tail systematic of P018. Consistency with P005: our 0.96-plateau counts 1.09/0.54 equal P005's 1.02/0.51 × 0.96/0.90.

### 4.2 Expected counts at LZ's one-event coupling (1 TeV; `comparison_with_LZ_1000GeV.csv`, `roi_edge_scan.csv`)
With ε_LZlike the ratio is pure exposure: XENONnT 1.09, PandaX-4T 0.54, combined **1.63 events, P(0) = 0.195**, for every δ and mass. With ε_hi (0.85, 50% at 300 keV): combined 1.47 / 1.47 / 1.65 / 2.89 / 15.2 at δ = 250 / 300 / 350 / 366 / 380 keV (P(0) = 0.23 / 0.23 / 0.19 / 0.055 / 2.5 × 10⁻⁷): a 30 keV higher edge is decisive only for δ ≳ 365 keV, where the recoil window sits above LZ's roll-off.

Hard-edge scan (combined X+P, plateau 0.96), δ = 250 / 300 / 350 / 380 keV:

| E_max | 150 | 200 | 250 | 270 | 300 keV |
|---|---|---|---|---|---|
| δ = 250 | 0.65 | 1.45 | 1.63 | 1.64 | 1.65 |
| δ = 300 | 0.32 | 1.32 | 1.63 | 1.64 | 1.66 |
| δ = 350 | 0.00 | 0.56 | 1.59 | 1.63 | 1.84 |
| δ = 380 | 0.00 | 0.00 | 0.08 | 0.64 | 14.9 |

(400 and 4000 GeV rows in the CSV: identical to ≤5% at δ ≤ 300 keV; at 4000 GeV, δ = 380 keV the 300 keV edge gives 2.7.) This confirms and sharpens P005: a 200 keV edge already captures 80–90% of the δ ≤ 300 keV signal but nothing at δ ≥ 350 keV; the P021 peak (δ ≈ 380 keV) is invisible to any ROI ending below ≈250 keV and *hugely* visible to one ending at 300 keV.

### 4.3 Zero-event limits (`projected_limits.csv`, `count_limits_FC.csv`)
Event-count limits: Poisson 90% UL = 2.302 (n = 0) and 3.889 (n = 1) for all b ≤ 1.7 × 10⁻² (whole-panel scaling gives 2.285 / 3.872); FC 90%: [0, 2.43] (n = 0) and [0.11, 4.35] (n = 1) — the background is too small to matter, so Poisson and FC agree to 5%.

σ_SI-equivalent 90% upper limits, 1 TeV, ε_LZlike, zero events (Sun-frame halo):

| δ (keV) | XENONnT | X + P | XENONnT, 1 event | LZ Fig. 6 upper (P007) | (X+P)/LZ upper |
|---|---|---|---|---|---|
| 100 | 6.0 × 10⁻⁴⁶ | 4.0 × 10⁻⁴⁶ | 1.0 × 10⁻⁴⁵ | 2.4 × 10⁻⁴⁵ | 0.18 |
| 200 | 6.2 × 10⁻⁴⁴ | 4.2 × 10⁻⁴⁴ | 1.05 × 10⁻⁴³ | 1.9 × 10⁻⁴³ | 0.23 |
| 250 | 6.3 × 10⁻⁴³ | 4.2 × 10⁻⁴³ | 1.06 × 10⁻⁴² | 1.18 × 10⁻⁴² | 0.36 |
| 300 | 5.0 × 10⁻⁴² | 3.3 × 10⁻⁴² | 8.4 × 10⁻⁴² | 7.9 × 10⁻⁴² | 0.43 |
| 350 | 4.0 × 10⁻⁴⁰ | 2.7 × 10⁻⁴⁰ | 6.8 × 10⁻⁴⁰ | 6.9 × 10⁻⁴⁰ | 0.41 |
| 380 | 9.2 × 10⁻³⁷ | 6.2 × 10⁻³⁷ | — | (grid ends) | — |

400 GeV: X / X+P at δ = 300 keV 2.0 × 10⁻⁴¹ / 1.35 × 10⁻⁴¹; 4000 GeV: 9.8 × 10⁻⁴² / 6.5 × 10⁻⁴² (300), 2.0 × 10⁻⁴⁰ / 1.3 × 10⁻⁴⁰ (350). ε_hi variant (X+P): 3.7 × 10⁻⁴² (300), 2.7 × 10⁻⁴⁰ (350), 9.4 × 10⁻³⁹ (370), 6.6 × 10⁻³⁸ (380). Halo bracket (12-day annual, X+P, ε_LZlike): 3.1 × 10⁻⁴² (300), 1.6 × 10⁻⁴⁰ (350), 4.4 × 10⁻³⁹ (370), 6.0 × 10⁻³⁸ (380 keV) — ×1.1 / ×1.7 / ×6.6 / ×10 stronger than Sun-frame. The *absolute* limits above δ ≈ 360 keV therefore carry a factor-10 halo-tail systematic (as does LZ's own interval); the *relative* statements below do not.

**Relative to LZ's interval.** A zero-event XENONnT sets κ < 2.11 κ̂_LZ; the combination sets κ < 1.41 κ̂_LZ. LZ's 90% upper edge is 3.65–3.73 κ̂ (P021, both halos), its lower edge ≈ 0.1 κ̂. Hence zero events exclude the upper **15% (XENONnT) / 27% (combined)** of LZ's 90% interval in ln κ, at every δ from 200 to 385 keV (ε_LZlike; P021 'sun' or annual configuration, identical to 1%). With ε_hi: 23–27% at δ ≤ 350, 59% at 370, 89% at 380 keV. LZ's best fit itself survives at N = 1.63 expected, P(0) = 20%. In cross-section, the combined zero-event limit lies 2.3–6× below LZ's published upper edge at δ = 100–350 keV (world-leading, but not interval-closing).

**Significance if null** (`Z_reduction_if_null.csv`): LZ's local Z for O₁ˢ (P021) would fall from 2.67 → 2.29 (δ = 300), 3.12 → 2.80 (350), 3.42 → 3.13 (370), 3.59 → 3.30 (380 keV) with ε_LZlike (R = 2.63); with ε_hi, 3.59 → 2.55 at 380 keV (R = 24.5). The fitted κ̂ drops by R.

### 4.4 Discovery case (`joint_fit_results.json`, `joint_surface_*.npz`, figure `joint_discovery_case.png`)
Counting. One XENONnT event in 200–270 keV alone: Z = 3.5 / 3.2 / 3.1 / 2.3σ for b_X = 2.2 × 10⁻⁴ / 6.2 × 10⁻⁴ / 1.1 × 10⁻³ / 1.16 × 10⁻² (the four background models scaled to 3.1 t·yr). Combined two events in LZ + XENONnT: P(≥2 | b_LZ + b_X) → **Z = 5.2 / 4.8 / 4.6 / 3.5σ** — i.e. 5σ-class only if the event sits in the NR-band neighbourhood (b ≲ 6 × 10⁻⁴ per 2.84 t·yr), 3.5σ if the whole-panel background is charged.

Joint (δ, κ) fit, 1 TeV, ε_LZlike:

| Data | δ_best | Z_best | 68% δ (2 dof) | 90% δ | Z at 300 / 350 / 380 |
|---|---|---|---|---|---|
| LZ only (this model) | 375 | 3.58 | 355–385 | 325–385 | 2.68 / 3.21 / 3.57 |
| + XENONnT E₂ = 200 keV | 360 | 4.92 | 340–370 | 315–375 | 4.36 / 4.84 / 3.61 |
| + XENONnT E₂ = 230 keV | 375 | 5.11 | 360–375 | 350–380 | 4.08 / 4.74 / 4.79 |
| + XENONnT E₂ = 260 keV | 380 | 5.25 | 375–385 | 370–385 | 3.47 / 4.29 / 5.25 |

The LZ-only row reproduces P021's 'sun' configuration (peak 370 keV, Z 3.49) to 5 keV / 0.1σ. A second event's *energy* is the first spectral information: 200 keV pulls the preferred δ down to 360 keV and disfavours 380 keV (Z drops from 4.9 to 3.6 there), whereas 260 keV pins δ = 375–385 keV at 68%. Best-fit κ: 0.043 (E₂ = 200), 5.8 (230), 42 (260) — spanning three orders of magnitude, i.e. the cross-section is undetermined until δ is.

### 4.5 Existing XENON1T 2018 limit (`xenon1t_vs_LZ.csv`, `xenon1t_recast_check.csv`, `implied_roi_edges.csv`)
| δ (keV) | XENON1T (Fig. S7 green) | LZ upper | LZ lower | X1T/LZ upper | X1T/LZ lower | LZ events at X1T limit |
|---|---|---|---|---|---|---|
| 100 | 2.6 × 10⁻⁴⁴ | 2.4 × 10⁻⁴⁵ | 4.5 × 10⁻⁴⁷ | 10.8 | 566 | 90 |
| 150 | 6.0 × 10⁻⁴³ | 2.3 × 10⁻⁴⁴ | 1.5 × 10⁻⁴⁵ | 26 | 402 | 239 |
| 200 | 7.2 × 10⁻⁴¹ | 1.9 × 10⁻⁴³ | 1.4 × 10⁻⁴⁴ | 385 | 5.2 × 10³ | 2.4 × 10³ |
| 220 | 4.3 × 10⁻³⁹ | 3.9 × 10⁻⁴³ | 2.9 × 10⁻⁴⁴ | 1.1 × 10⁴ | 1.5 × 10⁵ | 5.4 × 10⁴ |
| 230.8 | 6.9 × 10⁻³⁸ | 5.8 × 10⁻⁴³ | 4.3 × 10⁻⁴⁴ | 1.2 × 10⁵ | 1.6 × 10⁶ | 5.3 × 10⁵ |

**No part of LZ's 90% interval at δ ≤ 231 keV is excluded by XENON1T 2018**: the XENON1T curve lies 11× above LZ's *upper* edge already at 100 keV and 10⁵× above it at the curve's end; at the XENON1T limit LZ would have recorded 90 to 5 × 10⁵ events. (PandaX-4T 2021 is similarly 10²–10⁶ above; P015 gives the CRESST/PICO factors.)

*Why the curve ends at 231 keV.* The naive recast (1 t·yr, 0.9, 4.9–40.9 keV, 2.30 events) is 5.8× *stronger* than the published elastic 1 TeV limit at δ = 0 (1.35 × 10⁻⁴⁶ vs 7.9 × 10⁻⁴⁶ cm²), as expected for a zero-background counting proxy of a profile-likelihood result with observed events; normalised at δ = 0, its δ-dependence follows the green curve to within a factor 0.43–0.85 for δ = 50–200 keV and 0.98 at 210 keV, and its N_unit collapses between 220 keV (2.2 per unit κ) and 230 keV (0): the PICO-derived curve ends where the lower recoil edge E₋(δ) leaves the ROI. Inverting, the endpoints encode the ROI upper edges used in the recasts: E₋(230.8 keV, 1 TeV) = 44–50 keV (v_max = 809–776 km/s) for XENON1T (recalled ROI edge 40.9 keV, consistent to ~10% given the unknown halo choice), and E₋(303.3 keV) = 93–113 keV for the PandaX-4T 2021 commissioning search. This is also the physical reason the published curves are irrelevant at δ ≥ 300 keV: no recoil below ≈ 91 keV exists there (P005).

### 4.6 Data request
None filed. The published XENONnT/PandaX-4T efficiency tables (HEPData) end at their ROI edges (≲100 keV) and cannot supply the 200–300 keV acceptance that dominates this result; only a new high-energy analysis by those collaborations can. DR-002 is therefore left free.

## 5. Figures
- `figures/delta_sigma_plane_1TeV.png` — (δ, σ_SI-equivalent) plane at 1 TeV: LZ two-sided 90% band (P007 digitisation), P021 κ̂(δ) and 68% band, projected zero-event limits for XENONnT, PandaX-4T and their combination (ε_LZlike, ε_hi, annual-halo bracket), XENONnT one-event limit, XENON1T 2018 and PandaX-4T 2021 curves from Fig. S7.
- `figures/roi_edge_scan.png` — combined expected events at LZ's one-event coupling vs hard ROI upper edge, δ = 250–380 keV.
- `figures/joint_discovery_case.png` — joint 68% (δ, σ) regions for LZ + one XENONnT event at 200/230/260 keV and the profile Z(δ).
- `figures/efficiency_models.png` — efficiency models and the δ = 300/380 keV spectra.

## 6. Robustness and caveats
- Exposure–efficiency ratios drive every relative statement; the 0.96 plateau assumed for the extensions is LZ's, and real high-energy acceptances of the other TPCs are unknown (counts scale linearly).
- Halo tail: absolute rates for δ ≥ 360 keV vary ×3–10 between the Sun-frame and 12-day annual halos (and more across Gaia-range parameters, P018); all limits above 360 keV should be read with that bracket. The Sun-frame baseline is the one P021 identified as LZ's.
- Backgrounds of XENONnT/PandaX-4T at high S1 are unmodelled; the whole-panel scaling (0.0106 per 2.84 t·yr) is a pessimistic proxy and lowers the two-event significance to 3.5σ.
- The joint fit is a 1D observed-energy proxy (as P021) with asymptotic Z in the one-/two-event regime; below 125 keV nothing is modelled.
- LZ interval edges are digitised (±5%); the XENON1T curve is itself a PICO recast whose halo, mass and efficiency assumptions are not stated in the LZ paper.

## 7. Failed or abandoned
- A first full-grid 12-day annual computation (38 δ points) exceeded the foreground time limit and was killed before writing its cache; the bracket was reduced to nine δ points (300–385 keV) with an incremental cache.
- An initial version computed the "fraction of LZ's interval excluded" by mixing our N_unit with P021's κ̂; near the ceiling the two normalisations differ (different halos), so the fraction was redefined normalisation-consistently (Sec. 3.2).

## 8. References
LZ Collaboration, arXiv:2609.02823 (2026). E. Aprile et al. (XENON), PRL 121, 111302 (2018). Y. Meng et al. (PandaX-4T), PRL 127, 261802 (2021). E. Aprile et al. (XENON), PRL 135, 221003 (2025). Z. Bo et al. (PandaX), PRL 134, 011805 (2025). E. Adams et al. (PICO), PRD 108, 062003 (2023). G. J. Feldman, R. D. Cousins, PRD 57, 3873 (1998). I. Jeong et al. (WimPyDD), CPC 276, 108342 (2022). Corpus: dossier 00, P005, P007, P015, P016, P018, P020, P021.

## 9. Tools and provenance
Mirrors `output/provenance/P035.json`. Python 3.12.13 (`.venv/bin/python`); WimPyDD 2.0.4 (`eft_hamiltonian`, `streamed_halo_function` with explicit v_min grid via `lz.wd_halo`, `diff_rate` via `lz.wd_rate`); numpy 2.5.3; scipy 1.18.1 (special.erf, stats.poisson/norm, optimize.brentq); pandas 3.0.5; matplotlib 3.11.2 (Agg); `common/lzcommon.py` (LZ constants, `wd_c_from_anand`, `wd_halo`, `wd_hamiltonian`, `wd_rate`, `E_R_range_keV`, `vmax_kms`, `v_earth_kms`, `mu_red`, `m_nucleus_gev`). Script: `output/code/P035_recast_xenon.py` (three runs; the second was killed by the time limit). Local inputs: PAPER_GUIDE, dossier, ledger, P005/P007/P015/P020/P021 papers, P007/P015/P021 work files, P005/P021 scripts, lzcommon, LZ tex (l.28, 117, 347–412, 453–459, 798–819), WimPyDD/package.py docstring (read-only). Agent tools: Read ×18, Bash ×23, Write ×4, Edit ×31 (12 on the script, 13 on the paper for word budgets, 4 on the provenance JSON, 2 here), Skill ×1 (dataviz), ToolSearch ×1, Monitor ×1. Recalled values: 8 (see Sec. 2.3). Datasets: none. Data requests: none. WimPyDD-generated files: none.
