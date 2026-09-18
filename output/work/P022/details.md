# P022 — Accidental coincidences at high S1: could an isolated 540 phd S1 and a 270-electron S2 fake the LZ event?

Research record (complete). Simulated date 2026-09-08. Author profile: xenon-TPC analysts specialising in pulse-pairing backgrounds. Category BKG; physics.ins-det (cross-list hep-ex).
Script: `output/code/P022_accidentals.py` (run from the root with `.venv/bin/python`; log in `run_log.txt`). Machine-readable results: `P022_results.json`; tables `P022_fig5_accidentals_digitised.csv`, `P022_figS3_projections.csv`, `P022_figS3b_column_500_600.csv`; figures `figures/P022_fig5_accidentals_digitised.png`, `figures/P022_figS3_rates_and_k.png`.

## 1. Motivation and framework

The LZ paper (arXiv:2609.02823) names accidental coincidences as one of the three dominant backgrounds at the location of the 248 keV candidate (Discussion: "the dominant backgrounds are accidentals, atmospheric neutrinos, and MSSI events"), and the dossier (§5, explanation E, P = 0.05) flags that the accidentals model is validated with unphysical-drift-time (UDT) events "at low energies" while "the tail to large isolated S2s is poorly sampled". P004 (wall MSSI), P010 (ER leakage), P013 (neutrons) and P019 (atmospheric neutrinos) have quantified the other conventional origins; accidentals were untouched.

An accidental is an isolated S1 (paper: "predominantly from interactions in charge-insensitive regions or pile-up of photoelectron backgrounds") randomly paired with an isolated S2 ("typically from spurious electron emission from the electrodes"). For a stationary pair of Poisson processes the expected number of pairs with apparent drift time in [0, T_max] is

  N_acc = R_S1 · R_S2 · T_max · T_live · f_cuts,                                            (1)

and the {S1c, log10 S2c} density of pairs factorises into the isolated-S1 and isolated-S2 spectra before cuts (drift-time-dependent corrections and the pulse-based cuts introduce mild correlations afterwards). We ask:

1. What does LZ's own accidentals model predict at the event's coordinates (S1c = 540 phd, log10 S2c = 3.967, drift ≈ 860 μs) and within ±2σ_NR of the NR median at S1c > 500 phd?
2. Which isolated-S1 and isolated-S2 rates does the model imply at those pulse sizes, and can identified physical sources supply them?
3. By what factor k would the model have to be wrong for an accidental to be a likely (10 %) origin, and what do LZ's UDT validation samples (Fig. S3a before cuts, Fig. S3b after all cuts) actually constrain at S1c > 500 phd?
4. What does the event's drift time (and the S2-width/drift consistency the paper reports) do to the hypothesis?

## 2. Inputs (all from the LZ paper unless flagged)

| Input | Value | Source |
|---|---|---|
| Accidentals, science sample, WS ROI | 2.7 ± 0.6 expected; 2.6 ± 0.6 fit | Table I |
| Accidentals, prompt / delayed samples | (2.7 ± 0.6)×10⁻⁴ / (8.0 ± 1.9)×10⁻² | Tables S1, S2 |
| UDT validation after all cuts | model 2.4 ± 0.2, observed 3; cut-by-cut agreement within 20 % | supplement "Accidental backgrounds" |
| Fig. 5 (three S1c panels, 0.5σ_NR bins, yellow "Accidentals", blue "Total"); bottom-panel total 0.0106 ± 0.0008 | PNG 2444×3082 | Fig. 5 + caption |
| Fig. S3a/b (accidentals PDF vs UDT data; S1c and log10 S2c projections; viridis 2D map) | PNGs 1749×1295 / 1765×1304 | Fig. S3 + caption |
| Fig. 3 right axis (drift time vs z) | PNG | Fig. 3 |
| Event: S1c 540.1 phd, S2c 9268 phd (log 3.967), 1.5σ below NR median, z = 26.4 cm above cathode, S2 shape consistent with single site at that depth, S1 hit pattern consistent with (x,y,z) at 1–2σ and inconsistent with an RFR-MSSI template | — | Results; supplement "waveform analysis" |
| ROI: S1c 3–600 phd, S2 > 645 phd, log10 S2c 2.75–4.15; FV 12.8 cm below gate, 9.0 cm above cathode; RFR depth 13.75 cm; g1 = 0.110, g2 = 34.5 | — | Data Analysis; MSSI paragraph |
| Live time 220 d; FV mass 4.71 t; 1710 science events | — | Data Analysis, Table I |
| NR band width at S1c ≈ 540: σ_NR = 0.022 dex (drawn band), 0.033 (MC) | corpus | P009 / P004 |
| Flat-ER acceptance of the ROI: 1655 continuum events / 19.4 keV = 85 events per keVee in 2.84 t·yr | corpus | P010 details §5 |
| NEST-LZ β and NR mean yields | `lz.nest_er_yields`, `lz.nest_nr_yields` (nestpy 2.1.1, Table S3/S5 parameters) | lzcommon |
| Drift length gate–cathode 145.6 cm; active radius 72.8 cm; active mass 7.0 t | recalled (likely / likely / certain) | LZ design papers |
| LXe density 2.86 g cm⁻³ | recalled (likely) | — |
| Longitudinal diffusion D_L ≈ 25 cm² s⁻¹ at ~100 V cm⁻¹ | recalled (likely, ±40 %) | LUX/LZ/EXO measurements |
| Detector-γ enhancement of the ER rate density next to the bottom PMT array relative to the inner FV: ×10–1000 | recalled (uncertain) | generic self-shielding in ~1 m LXe TPCs |
| Isolated-S2 rate above 645 phd: 1 mHz – 1 Hz bracket | recalled (uncertain; not given in the paper) | — |

## 3. Method and derivations

### 3.1 Fig. 5 digitisation (script §1)

Frames of the three panels are the long dark lines at rows 496 | 1299 | 2102 | 2905 and columns 218–2408. x: major ticks at 492.5 (−6σ) … 2135.5 (+6σ) → 136.92 px/σ, −8σ at 218.7. y: major ticks at 594.5/735.5/876.5/1017.5/1158.5 (top panel, 10² … 10⁻²; 141.0 px/decade; frame bottom = 10⁻³), 1383/1503/1622.5/1742.5/1862.5/1982.5 (middle, 10¹ … 10⁻⁴; 119.9 px/decade; bottom = 10⁻⁵), 2131.5/2324.5/2518.5/2711.5 (bottom, 10², 10⁰, 10⁻², 10⁻⁴; 96.67 px/decade; bottom = 10⁻⁶) — the bottom-panel calibration is P004's (2131, 96.75). For each 0.5σ bin the top-most pixel of the line colour (yellow (255,220,61) accidentals; blue (0,0,255) total; RGB distance < 60) in the middle 26 % of the bin is the histogram level. Bins where the yellow line is hidden (middle panel +7.25, +7.75; bottom panel +5.25 … +7.75, where the accidentals fall below 10⁻⁵ behind other lines) are filled by log-interpolation/extrapolation of the neighbours and flagged; they contribute 5×10⁻⁵ of 6.3×10⁻⁴ in the bottom panel.

Validation: the digitised bottom-panel total integrates to 0.01065 vs the caption's 0.0106 (ratio 1.005); the three-panel accidental total is 2.47 vs the post-fit 2.6 (0.95) and pre-fit 2.7 (0.92).

### 3.2 Fig. S3 digitisation (script §2)

Main map frame rows 248–1159, columns 152.4–1164 (S3a) / 169.4–1180 (S3b); S1c 0–600 phd across the frame (major ticks every 100 phd = 168.6 px). Shared log10 S2c axis: major ticks 4.0 @ 331.5, 3.8 @ 471.5 … 3.0 @ 1032.5 → 700.8 px/dex; plotted range 2.82–4.12. Top projection: ticks 10² @ 62.5 and 10⁰ @ 199.5 (S3a; 68.5 px/decade), 10⁰ @ 56 and 10⁻³ @ 235.5 (S3b; 59.8 px/decade). Right projection: linear, 0 at the frame (1164.5) and 100 @ 1415 (S3a); logarithmic, 10¹ @ 1405 and 10⁻³ at the frame (S3b; 56.2 px/decade). The UDT data points (black blobs of ≈ 80 px) sit at 15 phd spacing → S1c bins of 15 phd (40 bins) and log10 S2c bins of 0.15 dex (edges 2.779 … 4.129, 9 bins). Model lines: top-most blue pixel per S1c bin; right-most blue pixel per log10 S2c bin.

Colour bar: viridis (nearest-colour index correlates with position at 0.99998); S3b ticks at rows 431/656/882/1107 = 10⁻² … 10⁻⁵ (225.3 px/decade) → bar spans 10⁻⁵·²³ – 10⁻¹·¹⁹; S3a ticks at 507.5/792.5/1078.5 = 10⁰ … 10⁻² → 10⁻²·²⁸ – 10⁰·⁹¹. Each pixel of the S1c 500–600 phd column is converted to "predicted counts per 2D bin" (pixels within 20 RGB units of a viridis colour only, which removes the red/cyan contours), giving the isolated-S2 spectrum at high S1c. The 2D bin height is not printed; we infer it from normalisation, Σ_rows (density × dex per row / dex per bin) × 7 S1c bins = top-projection sum in 495–600 phd → 0.0135 dex per bin (≈ 100 bins over the plotted range).

Validation: S3a top-projection model sum 562 vs 537 data points; right-projection model 537.5 vs 536 data; S3b top and right model sums both 2.62 (quoted 2.4 ± 0.2; +9 %, dominated by reading the first bin, 2.08, at 60 px/decade); the S1c 500–600 column shape from the 2D map matches the all-S1c projection shape (Fig. 2, middle), i.e. the model factorises.

### 3.3 Fig. 3 drift-time axis (script §3)

Frame rows 19–1028, columns 152–1203. Left (z) long ticks include the FV solid lines; keeping the regularly spaced ones (130.0 px = 20 cm; rows 88.5 … 995) gives 6.475 px/cm with z = 0 at row 995; right (drift) major ticks 51.5 … 950.5 = 0 … 1000 μs (0.899 μs/px). The cathode row maps to T_max = 1049 μs, the gate to 0.5 μs, z = 26.4 cm to t_ev = 859 μs, v_d = 1.388 mm/μs. (The dossier's "≈ 870 μs" and the 951 μs maximum drift assumed in our assignment refer to the 193 V/cm run; at 97 V/cm the drift is slower.) FV in drift time: 93–985 μs (85 % of the drift range).

### 3.4 Factorisation (script §4)

From (1) with N = 2.7, T_max = 1.049 ms, T_live = 1.90×10⁷ s: R_S1 R_S2 f_cuts = 1.35×10⁻⁴ Hz² (science sample, ROI pulse sizes). In the event box (S1c 500–600 phd × ±2σ_NR) the product is 1.35×10⁻⁴ × f_S1 × f_S2 = 1.9×10⁻⁸ Hz².

Pre-cut products need the pre-cut PDT count, which the paper does not give; under **assumption A** (the UDT window is as long as the PDT window, T_max) the 537 pre-cut UDT events are also the pre-cut PDT count, so R_S1,tot R_S2,tot = 537/(T_max T_live) = 0.027 Hz², the overall cut+FV acceptance is 2.7/537 = 0.50 %, and R_S1(500–600) R_S2,tot = 0.027 × 0.026 = 7.0×10⁻⁴ Hz² (model; 1.1×10⁻³ with the 22 observed UDT events). Table 3 gives R_S1(500–600) for the isolated-S2 bracket.

### 3.5 Physical plausibility (script §5)

RFR mass: π(72.8 cm)² × 13.75 cm × 2.86 g cm⁻³ = 0.655 t = 9.4 % of the 7.0 t active mass (13.9 % of the FV). ER rate density in the FV: 85 per keVee in 2.84 t·yr = 0.082 /t/d/keV (8.2×10⁻⁵ /kg/d/keV). ER energy giving S1c 500–600 phd with NEST-LZ β yields (55 ph/keV at 70 keV, 53 at 100 keV) and g1 = 0.110: 84–104 keV (92 keV for 540 phd) at the mean light collection; 63–77 keV if the local collection is 1.3× the mean (bottom of the TPC), 110–139 keV at 0.8×. Rate of internal-β deposits in the RFR with S1 in 500–600 phd: 0.082 × 0.655 × 20.4 keV = 1.1 per day = 1.3×10⁻⁵ Hz; detector γ-rays next to the bottom PMT array multiply this by ×10–1000 (uncertain).

S1 that must accompany any genuine 269-electron ionisation cluster in the bulk: β ER of 10.7 keV → S1c ≈ 58 phd; NR of 177 keV → S1c ≈ 336 phd (NEST-LZ means), both ≫ the 3 phd threshold, so a real cluster never yields an S1-less S2; an "isolated" 269-electron S2 must be electrode emission (gate/anode: zero drift; cathode: full drift) or a cluster whose S1 merged with the big S1 (that is an MSSI-type topology, treated by LZ and P004, not an accidental).

S2 width: σ_t = √(2 D_L t)/v_d = 1.49 μs at 859 μs, 1.65 μs at the cathode (1049 μs); ratio 1.105; the statistical width uncertainty from 269 electrons is 1/√(2·269) = 4.3 %, so a cathode-emission S2 is 2.4σ_stat wider than a bulk S2 at the event's drift (diffusion-only; any constant width term dilutes the difference). A cathode S2 paired with a random S1 has its width within 10 % (20 %) of the diffusion expectation whenever the apparent drift exceeds 0.81 (0.64) T_max = 850 (672) μs, i.e. 19 % (36 %) of the drift range; the event (0.82 T_max) sits in that region.

### 3.6 Mismodelling (script §6)

With N_nb the model's accidental expectation in the neighbourhood, P(≥1) = 1 − e^(−k N_nb); k(10 %) = 0.105/N_nb, k(50 %) = 0.693/N_nb. The post-cut UDT sample has 0 events at S1c > 495 phd with expectation m_hi = 2.4 × 0.0068 = 0.0163, so a normalisation factor k at high S1 predicts k m_hi UDT events; k < 141 (90 %) / 183 (95 %), and P(0 | k m_hi) evaluates the k required. Shape-only mismodelling of the isolated-S2 spectrum at high S1c is bounded by "flat in log10 S2c" (f_S2(±2σ) = 4σ/1.4 dex = 0.063, ×3.0 over the map) and by the unphysical "all in the window" (×48).

## 4. Results

### 4.1 Model prediction (Fig. 5)

| quantity | value |
|---|---|
| accidentals, S1c < 250 panel (±8σ) | 2.47 (±2σ: 1.09; below median 1.70) |
| 250 < S1c < 500 | 2.94×10⁻³ (±2σ: 6.7×10⁻⁴) |
| S1c > 500 (±8σ; log10 S2c 3.82–4.15) | 6.3×10⁻⁴ (raw 5.8×10⁻⁴ + 0.5×10⁻⁴ filled) |
| S1c > 500, ±2σ_NR (log10 S2c 3.956–4.044) | **N_nb = 1.49×10⁻⁴** (−3..+1σ: 1.66×10⁻⁴) |
| bin −2..−1.5σ / −1.5..−1σ | 1.96×10⁻⁵ / 2.54×10⁻⁵ |
| accidentals share of the total model in ±2σ (1.62×10⁻³, incl. the ER tail at +1..+2σ) | 9.2 %; ≈ 50 % of the −2..0σ bins |
| fraction of the accidental spectrum above 500 phd (in-panel) | 2.5×10⁻⁴; 250–500: 1.2×10⁻³; < 250: 99.86 % |
| three-panel total / Table I fit | 2.47 / 2.6 = 0.95 |

P(≥1 accidental in the neighbourhood) = 1.5×10⁻⁴; anywhere in the S1c > 500 panel 6.3×10⁻⁴.

### 4.2 Isolated-S1 and isolated-S2 spectra (Fig. S3)

| quantity | S3a (normalisation stage) | S3b (after all cuts) |
|---|---|---|
| S1c projection sum: model / data | 562 / 537 | 2.62 / 3 (quoted 2.4) |
| log10 S2c projection sum: model / data | 537.5 / 536 | 2.62 / 3 |
| S1c ≥ 495 phd: model / data | 14.6 / 22 (ratio 1.51 ± 0.32) | 0.0178 / 0 |
| f_S1(≥ 495 phd) | 2.6 % (data 4.1 %) | 0.68 % |
| cut survival (b/a) | — | 0.47 % overall, 0.12 % at ≥ 495 phd |
| log10 S2c projection per 0.15 dex, 2.93–3.08 → 3.98–4.13 | rises 38 → 78 (of 537) | falls 0.64 → 0.068 (of 2.62) |
| S1c 500–600 column (2D map): f(±2σ_NR) / f(3.9–4.0) / f(3.82–4.15) | 9.1 % / 10 % / 30 % | **2.07 %** (3.0 % for σ = 0.033) / 2.6 % / 7.5 % |
| column density ratio log10 S2c 4.0 / 3.0 | 3.1 | 0.21 |
| map value at the event pixel (counts per 15 phd × 0.0135 dex bin) | 3.3×10⁻² | 8.5×10⁻⁶ |

Consistency between the two figures: S3b predicts, for the science sample, 2.6 × 0.0068 = 0.0177 accidentals at S1c > 500 phd over all S2c, 1.3×10⁻³ in the ±8σ panel window and 3.7×10⁻⁴ in ±2σ (map box integral 4.0×10⁻⁴). Fig. 5 gives 6.3×10⁻⁴ and 1.49×10⁻⁴, i.e. 0.48 and 0.41 of the UDT-normalised expectation — the fiducial-volume and PDT-specific acceptance (FV mass fraction 4.71/7.0 = 0.67, plus the radial cut on the isolated-S2 position and the drift-time window). The two LZ figures are therefore mutually consistent at high S1c to within a factor that has an obvious origin.

The pre-cut UDT data at S1c ≥ 495 phd exceed the model by 1.51 ± 0.32 (22 vs 14.6; 1.6σ). Taken at face value this raises N_nb to 2.2×10⁻⁴.

### 4.3 Rates implied

| assumed R_S2,tot (isolated S2 > 645 phd) | R_S1,tot (Hz) | R_S1(500–600 phd) (Hz) | per day | needed ×(RFR internal β) |
|---|---|---|---|---|
| 1 mHz | 27 | 0.70 | 60 000 | 55 000 |
| 10 mHz | 2.7 | 0.070 | 6 000 | 5 500 |
| 100 mHz | 0.27 | 0.0070 | 600 | 550 |
| 1 Hz | 0.027 | 7.0×10⁻⁴ | 60 | 55 |

(Assumption A; multiply by 1.5 for the observed pre-cut excess.) The post-cut science-sample product in the event box is R_S1(500–600) R_S2(±2σ) f_cuts = 1.9×10⁻⁸ Hz². The internal-β deposition rate in the RFR with S1 in 500–600 phd is 1.1 per day, so the model's isolated-S1 population at 500–600 phd requires a ×55–5500 larger source: detector γ-rays interacting in the LXe adjacent to the bottom PMT array (RFR and the layer below the bottom grid) are the natural candidate and were validated as a population by the 22 pre-cut UDT events. Such S1s are bottom-heavy in the PMT arrays; the paper's hit-pattern analysis finds the event's S1 consistent with its reconstructed z at 2σ and (x,y) at 1σ and "inconsistent with an RFR MSSI template" (87 % of whose light is RFR light), which applies a fortiori to a 100 % RFR-origin S1. Gas or gate-region S1s are top-heavy and even less consistent. This post-hoc check is not part of the Fig. 5 PDF; it can only reduce N_nb.

### 4.4 The isolated S2

A 269-electron cluster in the bulk carries an S1 of ≈ 58 phd (β) or ≈ 336 phd (NR): no S1-less S2 from a real interaction is possible above threshold, so an isolated 269 e S2 is electrode emission. LZ's post-cut isolated-S2 spectrum (S3b) falls by ×9 from log10 S2c 3.0 to 4.1 (pre-cut it rises, so the S2-width/drift cut removes the large isolated S2s preferentially — large "isolated" S2s are mostly wide electron trains or gas events). Gate/anode emission has zero drift and fails the width cut at 859 μs; cathode emission drifts 1049 μs and is only 2.4σ_stat wider than a bulk S2 at 859 μs (diffusion only), so cathode-origin S2s paired with a random S1 pass a ±10 % (±20 %) width tolerance for apparent drift > 850 (672) μs — 19 % (36 %) of the drift range, containing the event (0.82 T_max). This is the one accidental topology that survives every published check and it is testable by LZ: compare the event's S2 width with the cathode-drift template (1.65 vs 1.49 μs σ) and its (x,y) with the map of cathode emission hot spots.

### 4.5 Mismodelling factor and what the UDT data constrain

| quantity | value |
|---|---|
| k for P(≥1) = 10 % in ±2σ neighbourhood | **708** (445–1183 for N_nb × 1.3^±1 and ± 22 %) |
| k for 50 % | 4655 |
| k for 10 % anywhere at S1c > 500 (±8σ) | 167 |
| k allowed by post-cut UDT (0 obs at S1c > 495, expectation 0.0163) | < 141 (90 %), < 183 (95 %) |
| k allowed by pre-cut UDT for the isolated-S1 spectrum at 500–600 phd | < 2.16 (95 %); best 1.51 |
| P(≥1 nb) at k = 183 | 2.7 % |
| post-cut UDT events expected at k = 708 / P(0) / Z | 11.6 / 9.6×10⁻⁶ / 4.3σ (range 3.2σ–5.8σ) |
| same for k = 167 (anywhere in panel) | 2.7 / 6.5 % / 1.5σ |
| flat-in-log10 S2c shape at high S1 (gain ×3.0) → k_norm needed / P(0) | 233 / 2.2 % (2.0σ) |
| all high-S1 isolated S2 inside ±2σ (gain ×48, unphysical) → k_norm / P(0) | 14.7 / 79 % |

The paper's "agreement within 20 % cut by cut" is a statement about the whole UDT sample, 99 % of which has S1c < 100 phd; at S1c > 500 phd the post-cut validation is a Poisson zero against 0.016 expected. It nevertheless excludes the ×708 needed for the neighbourhood at 4.3σ, provided (i) the UDT and PDT populations share their cut acceptances at high S1 (the paper's working assumption) and (ii) the isolated-S2 spectrum at high S1c is not pathologically peaked at the NR band. A flat isolated-S2 spectrum combined with a ×233 normalisation error is excluded only at 2σ.

### 4.6 Drift time

t_ev = 859 μs; T_max = 1049 μs. For an accidental the apparent drift time is uniform: P(t ≥ t_ev) = 18 % over 0–T_max, 14 % within the FV window (93–985 μs); a genuine interaction is also uniform in z, so the drift time carries no discriminating power on its own (the assignment's 9 % assumed T_max = 951 μs and t = 870 μs). Its only leverage is through the S2-width/drift consistency (§4.4), where the acceptance for cathode-origin S2s is 19–36 % at long drift rather than the few per cent at short drift.

## 5. Figures

- `figures/P022_fig5_accidentals_digitised.png` — digitised accidental (yellow) and total (blue) histograms of the three Fig. 5 panels; grey band ±2σ_NR; dotted line the event; red crosses the hidden, interpolated bins.
- `figures/P022_figS3_rates_and_k.png` — left: isolated-S1 spectra (S3a/S3b models and UDT data, 15 phd bins; shaded 500–600 phd); middle: isolated-S2 shapes (projections, and the 2D-map column at S1c 500–600 phd; shaded ±2σ_NR at S1c = 540); right: P(≥1) vs k with the post-cut (green) and pre-cut (blue) UDT limits.

## 6. Robustness, failed approaches

- Normalisation of "events / bin width": P004 showed the values are per 0.5σ bin (total 0.01065 vs 0.0106); same here.
- σ_NR = 0.033 dex instead of 0.022 changes f_S2(±2σ) from 2.07 % to 3.0 % and the S3b-predicted ±2σ count from 3.7×10⁻⁴ to 5.3×10⁻⁴; N_nb from Fig. 5 (P004's window) is unchanged.
- Reading uncertainty: 1 px = 0.010 dex (Fig. 5 bottom), 0.017 dex (S3b top), 0.018 dex (S3b right); the S3b sums are 9 % above the quoted 2.4; we assign ×1.3 to N_nb and ×1.5 to fractions read from S3.
- Assumption A (UDT window = T_max) affects only the absolute pre-cut products in §4.3, not N_nb, k or the UDT exclusion.
- Abandoned: separating R_S1 and R_S2 absolutely (the paper gives neither), and a Compton/geometry model of the S1 hit pattern for RFR-origin light (LZ's own template test supersedes it).

## 7. Extended discussion

Three facts frame the answer. First, the model puts 1.5×10⁻⁴ accidentals in the event's neighbourhood and 6×10⁻⁴ anywhere at S1c > 500 phd — accidentals are half of the modelled background below the NR median there, but tiny. Second, the model's own structure (99.9 % of accidentals below 250 phd; the isolated-S2 spectrum falling by ×9 across the ROI after cuts) is the reason: the isolated-S1 spectrum above 500 phd is real (22 pre-cut UDT events, ×1.5 above the model) but the pulse-based cuts kill 99.9 % of it. Third, the factor ×708 that a 10 % accidental origin needs is excluded at 4.3σ by the zero post-cut UDT events at high S1, whereas the ×167 needed for "an accidental somewhere at S1c > 500 phd" is not (P = 6.5 %). Compared with wall MSSI (P004: required/allowed ≈ 380), the accidental hypothesis has a thinner margin (708/183 ≈ 4) because its high-S1 validation rests on a Poisson zero. The surviving loophole — a ≥ 270-electron cathode-emission burst paired with a bottom-origin γ S1 whose hit pattern happens to mimic (46 cm, 26 cm) — is specific enough for LZ to close with the event waveform.

## 8. References

1. LZ Collaboration, arXiv:2609.02823 (2026) — Data Analysis, Table I, Fig. 3, Fig. 5, Discussion, supplement "Accidental backgrounds", Fig. S3, waveform analysis.
2. LZ Collaboration, Phys. Rev. Lett. 131, 041002 (2023) — SR1 result; accidentals framework and isolated-pulse populations.
3. LZ Collaboration, arXiv:2410.17036 (2024) — WS2024 result; accidentals model and UDT normalisation referenced by the paper.
4. LUX Collaboration, Phys. Rev. D 102, 112002 (2020) — isolated S2 / electron-emission backgrounds in a xenon TPC.
5. P. Sorensen and K. Kamdin, JINST 13, P02032 (2018) — electron emission backgrounds ("two distinct components of the delayed single electron noise").
6. D. S. Akerib et al. (LUX), Phys. Rev. D 97, 102008 (2018) — longitudinal diffusion in LXe.
7. Corpus: P004 (Fig. 5 method), P009 (σ_NR), P010 (ROI acceptance 19.4 keV), P013, P019; dossier §5.

## 9. Tools and provenance

Mirrors `output/provenance/P022.json`.
- Agent tools: Read (PAPER_GUIDE.md; dossier; ledger; P004/P010/P013/P016 papers; P004 script and CSV; fulltext.tex lines 84–203, 220–334, 436–449, 515–584, 690–712; P010 details 112–127; P004 details head; Fig5, FigS3a, FigS3b, Fig3 PNGs; the two P022 figures), Bash (grep of the tex for accidentals passages; ls of inputs/outputs; lzcommon/version probes; four exploratory PIL scripts in the scratchpad for frames, ticks, colours, colour bar and data dots; three runs of the script; wc -w), Write (script, details.md, P022.json, P022.md), Edit (script fixes: Fig. 3 tick filtering, factorisation restructure, k exclusion block, ordering).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (stats.poisson/gamma/norm, ndimage.label/center_of_mass); matplotlib 3.11.2 (Agg; colormaps['viridis']); Pillow 12.3.0 (PIL.Image); nestpy 2.1.1 via common/lzcommon.py (nest_er_yields, nest_nr_yields, LZ constants).
- Recalled knowledge (6 items): drift length 145.6 cm (likely); active radius 72.8 cm (likely); active mass 7.0 t (certain); LXe density 2.86 g cm⁻³ (likely); D_L ≈ 25 cm² s⁻¹ (likely, ±40 %); detector-γ enhancement ×10–1000 near the PMTs and isolated-S2 rate bracket 1 mHz–1 Hz (uncertain).
- Datasets: none. Data requests: none. WimPyDD files: none.
