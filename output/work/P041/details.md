# P041 — Detector artefacts: could partial charge loss turn a 71 keV electron recoil into the LZ event?

Research record (simulated date 2026-09-10; author profile: dual-phase TPC instrumentation experts). Script: `output/code/P041_charge_loss.py`; run log `output/work/P041/run_log.txt`; results `P041_results.json`.

## 1. Motivation and framework

The dossier (`output/00_evidence_dossier.md`, §5 F) assigns a prior of 0.18 to "detector or reconstruction artefact", the second-largest single entry, on the grounds that the event "needs ~80 % charge loss if it is an ER" and that unknown unknowns dominate single-event anomalies. P010 quantified the ER hypothesis: a 69.6 keV β-like ER would have to keep only 269 of its mean 1226 electrons (r = 0.939 vs 0.720). P010 treated this as a *microphysical* recombination fluctuation and disfavoured it at ≲0.3 %. Here we ask the complementary instrumental question: which detector mechanisms could remove 78 % of the drifting or extracted electrons of an ordinary ER *after* the recombination stage while leaving S1, S2 pulse shape, position reconstruction and all published checks intact — and what each would have done to the rest of the data set.

The LZ paper's own statements that bear on this (fulltext.tex):
- l.87–88: S1c/S2c are position-corrected; g1 = 0.110 ± 0.002, g2 = 34.5 ± 1.1 from monoenergetic peaks (83mKr is the standard source; l.169, 201, Table I).
- l.166: "the S2 pulse shape is consistent with a point-like interaction at the reconstructed depth"; S1 top/bottom partition consistent with z.
- l.695 (Waveform Analysis): S2 shape consistent with a single-site template from *alpha decays* at the same location; hit pattern consistent at 2σ in z and 1σ in (x,y).
- l.303: "no abnormalities in the scintillation and ionization responses of the detector in the temporal and spatial vicinity".
- l.700: accidentals cut "ensures that the S2 pulse width matches the inferred drift time".
- Fig. 3 caption (l.125): gas-phase interactions reconstruct at drift ≈ 56 μs; Fig. 4 (l.153–155): grey points are the ER population outside the ROI, with 125I/133Xe features "slightly below the beta-decay ER band".

Mechanisms considered: (i) bulk electron-lifetime excursion, (ii) localised absorber (bubble, particulate, impurity plume) or localised field defect / charge trapping, (iii) extraction or gas-gap deficit, (iv) S2 saturation/clipping or a truncated event window, (v) S1–S2 mis-pairing (P022), (vi) MSSI hybrids (P004; not an artefact).

## 2. Inputs

| Quantity | Value | Source |
|---|---|---|
| S1c, S2c, g1, g2 | 540.1 phd, 9268 phd, 0.110, 34.5 ± 1.1 | paper l.88, l.165 via `lz.LZ` |
| N_e(observed) | 268.6 | S2c/g2 |
| N_e(ER 69.6 keV), σ(N_e) | 1226, 150 | P010 (r = 0.720, N_i = 4379); nestpy LZ-ER variant 1340 (`lz.nest_er_yields(69.6)`) |
| NR-band median / σ at S1c = 540 | 4.015 / 0.0313 dex | P009, P024 |
| Drift geometry | T_max = 1049 μs, t_ev = 859 μs, v_d = 1.388 mm/μs, FV 93–985 μs | P022 (Fig. 3 axis) |
| FV boundaries | 12.8 cm below gate, 9.0 cm above cathode | paper l.139 |
| Science sample | 1710 events / 220 live days; ROI ER prediction 1704 (Table I sum) | paper Table I via `lz.LZ` |
| Gate–cathode 145.6 cm, radius 72.8 cm | recalled (likely) | LZ NIM A 2020 |
| D_T = 55 (50–60) cm²/s, D_L = 25 cm²/s | recalled (likely) | EXO-200 PRC 95 (2017); P022 |
| LZ electron lifetime ≳ 5–10 ms | recalled (likely) | LZ SR1/WS2024 |
| Extraction efficiency ≈ 0.80 | recalled (uncertain) | P024 assumption; Xu et al. 2019 |
| Typical S2c (x,y) map amplitude ≲ 20 % | recalled (likely) | LZ detector papers |
| 136Xe abundance 8.86 %, T½ 2.165×10²¹ yr | recalled (certain/likely) | EXO-200/KamLAND-Zen |
| 222Rn in WS2024 ≈ 1 (0.5–2) μBq/kg | recalled (uncertain) | LZ 2024 |
| 83mKr decays per injection 10⁵–10⁶; S2 resolution 10 % | recalled (uncertain) | LZ calibrations |
| α recombination: N_e(5.5 MeV α) ≈ 2–4×10⁴ | recalled (uncertain) | LUX/LZ alpha S2s |
| O2 attachment: τ_e·[O2] ≈ 150–300 μs·ppb | recalled (uncertain) | Bakale et al. 1976 |

## 3. Required loss and landing regions (script §1)

Survival fraction needed f = 268.6/1226 = **0.219** (nestpy variant 0.201); loss L = **78.1 %** (79.9 %). The ER-band mean at S1c = 540 is log₁₀S2c = log₁₀(1226 × 34.5) = **4.626**; the event is at 3.967, i.e. −0.66 dex.

**Fig. 4 digitisation.** Frame columns 143–1185 = 0–800 phd; ROI edges at rows 361 (4.15) and 898 (2.75) → −383.6 px/dex (P009's calibration reproduced). Colour census of the block S1c 500–580, log S2c 4.15–4.9: (128,128,128) grey data points 13 082 px, white 8 550, ER-band navy 823, contour slate (87,103,103) 249 — the energy contours are not neutral grey and are excluded by the filter |R−G|,|G−B| < 6. Lowest grey pixel per 10-phd column at S1c 500–580: median **4.374** (columns 4.25–4.42; 0.25 dex = 3σ_ER below the mean, as expected for a few-thousand-point population). Envelope vs S1c (330/400/500/600/700/800): 4.17/4.32/4.25/4.37/4.37/4.43 (`P041_fig4_grey_envelope.csv`). **Zero grey pixels at S1c 520–800 in 4.15 < log S2c < 4.30**; 134 black pixels inside the ROI at S1c 330–600 (the event marker alone). So at high S1c the whole band from the ER cloud's lower edge down to the S2 threshold is empty except for the event.

Landing regions at S1c = 540 and the survival window each corresponds to:

| Region | log₁₀S2c | f | flat-in-loss probability |
|---|---|---|---|
| ER cloud | > 4.374 | 0.560–1 | 0.440 |
| strip (outside ROI, empty) | 4.15–4.374 | 0.334–0.560 | 0.226 |
| gap G (ROI, above NR+2σ, empty) | 4.078–4.15 | 0.283–0.334 | 0.051 |
| V (NR ± 2σ; 1 event) | 3.952–4.078 | 0.212–0.283 | 0.071 |
| below NR−2σ to S2 = 645 phd (empty) | 2.81–3.952 | 0.015–0.212 | 0.197 |

For a loss distribution flat in f, p_V/(p_V + p_empty) = 0.13 and, given one event in V, the maximum-likelihood scaling predicts 6.7 in the empty regions, P(0) = 0.001. (The same geometry holds to ±0.05 dex across S1c 250–600, where the ER–NR separation is 0.55–0.65 dex.)

**Parent population.** Table I ER sum in the ROI is 1704; the ER-band median leaves the ROI (S2c = 10^4.15, N_e = 409) at E = 18.9 keV (nestpy LZ-ER yields), so dR/dE ≈ 1704/17.4 ≈ 98 events/keV/2.84 t·yr (flat approximation; 214Pb-dominated, bracket ×0.5–2). Parents: 60–80 keV: 2521 (incl. 564 124Xe KK decays, P010); S1c 250–600 (30–78 keVee): **5261** — any of these converted into the NR band would have been "the event", so N_par = 5261 is used below.

**Monitor rates.** 136Xe 2νββ in 4.71 t: 1.91×10²⁷ atoms → 19.4 mBq = 70/h; Rn chain at 1 μBq/kg: 17/h (8–34) decays → 34/h β (214Pb+214Bi) and 51/h α. Total β+α with S2s usable as purity monitors: **155/h (112–239)**; at drift > 800 μs: 32/h; at drift > 500 μs: 84/h. All-energy ERs in the run: 5.5×10⁵.

## 4. Electron lifetime (script §2)

Measured S2 = S2_true·e^{−t/τ_true}; corrected S2c = S2·e^{t/τ_cal}. For S2c/S2c_true = 0.219 at t = 859 μs: 1/τ_true = 1/τ_cal + ln(1/0.219)/859 μs → **τ_true = 566 / 536 / 508 μs** for τ_cal = ∞ / 10 / 5 ms. LZ lifetimes are ≳ 5–10 ms (recalled), so this is a ×10–20 drop in purity. If instead the calibration lifetime were merely overestimated, the uncorrected loss at 859 μs is 8.2 / 15.8 / 29.1 % for τ = 10 / 5 / 2.5 ms — a factor 3–10 short of the requirement (Fig. 1).

Under τ_e = 566 μs a 70 keV ER lands in: ER cloud for t < 328 μs (26 % of the FV drift range), strip 328–621 μs (33 %), G 621–715 μs (11 %), V 715–878 μs (18 %), below V > 878 μs (12 %) (Fig. 2). The ER band as a whole would drop by 0.66 dex at 859 μs and 0.76 dex at the FV bottom — the ER band would be smeared over 0.8 dex in drift time, far outside the ~0.1 dex band width.

Sample statistics: science-sample rate 1710/(220 × 24) = 0.324/h, of which 0.067/h at drift > 800 μs — the *ROI* sample cannot test a 1-hour excursion. But the ROI sample is the wrong monitor: the full-FV β+α rate is 155/h, and 84/h have drift > 500 μs (loss > 59 % under τ = 566 μs). P(no such event during an excursion of Δt) = e^{−84 Δt/3600}: **0.79 (10 s), 0.50 (30 s), 0.25 (60 s), 9×10⁻⁴ (300 s), 8×10⁻⁷ (600 s), 3×10⁻³⁷ (1 h)**. A bulk purity change is set by circulation and outgassing time-scales (minutes to hours); an excursion shorter than a few minutes would have to be a localised plume (§5). In addition the excursion size must coincide with the event: for the event to fall in V, τ ∈ [554, 680] μs, a log-uniform-over-two-decades probability of 0.045.

## 5. Local absorber (script §3)

Cloud size at 859 μs: σ_T = √(2 D_T t) = **3.07 mm** (2.93–3.21 for D_T = 50–60 cm²/s); σ_L = 2.07 mm → S2 σ_t = 1.49 μs, exactly P022's value (consistency check of the drift inputs). Footprint (2σ disk) 1.19 cm² vs TPC 16 650 cm² → 7.1×10⁻⁵.

**Absorbing-disk model.** A perfectly absorbing disk of radius R_d at offset b from the cloud axis removes L(b) = 1 − Q₁(b/σ, R_d/σ) (Marcum Q; numerically via `scipy.special.i0e`). Impact parameters are uniform in area within R_d + 4σ (the "hit zone"). A centred cloud loses exactly 78 % for R_d = 1.74σ = 5.4 mm.

| R_d | hit zone | p_V | p_G | p_strip | p_below | p_empty | p_V/(p_V+p_empty) |
|---|---|---|---|---|---|---|---|
| 1.74σ (5.4 mm) | 9.8 cm² | 0.012 | 0.010 | 0.053 | 0.000 | 0.063 | 0.16 |
| 2σ (6.1 mm) | 10.7 | 0.015 | 0.011 | 0.056 | 0.016 | 0.083 | 0.15 |
| 3σ (9.2 mm) | 14.6 | 0.020 | 0.014 | 0.065 | 0.078 | 0.157 | 0.11 |
| 5σ (15 mm) | 24.0 | 0.023 | 0.016 | 0.069 | 0.117 | 0.202 | 0.10 |
| 10σ (31 mm) | 58.2 | 0.022 | 0.014 | 0.059 | 0.119 | 0.191 | 0.10 |
| 20σ (62 mm) | 171 | 0.016 | 0.008 | 0.041 | 0.087 | 0.137 | 0.11 |

Most hits are grazing (f > 0.56, back in the ER cloud); the "just-right" window V takes 1–2 % of hits and 10–16 % of the hits that leave the cloud (Fig. 3).

**Budget.** With N_par = 5261 and area fraction ε of the TPC covered by hit zones: expected N_V = N_par ε p_V, N_empty = N_par ε p_empty, all-energy affected = 5.5×10⁵ ε (1 − p_none).
- N_V = 0.1 (10 % chance of one event-like ER) needs ε = 0.8–1.6×10⁻³, i.e. 0.1–2.7 *persistent* hit zones (one 1–3 cm object present all run); it predicts 0.5–0.9 ERs in the empty regions and **65–101 all-energy ERs** with > 44 % charge loss.
- N_V = 10⁻³ needs ε ≈ 10⁻⁵ (13–170 cm² for a fraction 10⁻⁴–10⁻⁵ of the run).
- Maximum-likelihood scenario given (1 in V, 0 in empty): ε* = 1/[N_par(p_V + p_empty)] = 0.8–2.6×10⁻³; P(1 in V, 0 in empty | ε*) = p_V/(p_V+p_empty)·e⁻¹ = **0.038–0.059**; ~104 all-energy affected ERs.

**Persistent absorbers are excluded by the 83mKr map.** With 10⁵–10⁶ 83mKr decays per injection there are 6–60 per cm²; a 78 % S2 deficit in a 1 cm² cell is a 19–61σ hole in the S2c(x,y) map, which LZ uses to build the S2c correction (paper l.87–88) and which shows "no abnormalities" spatially. Charge trapping on PTFE or a floating conductor (field distortion) is likewise static and would also appear as a z-dependent position/band anomaly. Only a *transient* absorber survives — a bubble, a particulate, or an impurity plume — with an effective ε = ε_geom × duty fraction. Bubbles and PTFE fragments (2.2 g/cm³ < 2.9 g/cm³ LXe) float to the surface under the gate and become persistent; a dense particle sinks to the cathode/RFR; a neutrally buoyant absorber is the residual case.

**Impurity plume.** To remove 78 % over a path ℓ the local lifetime must be τ_loc = ℓ/(v_d ln(1/f)): 1.4 / 4.7 / 14 μs for ℓ = 0.3 / 1 / 3 cm — an O2-equivalent enrichment of ×3500 / ×1050 / ×350 over a 5 ms bulk (≈ 30–200 ppb O2 for τ·[O2] ≈ 150–300 μs·ppb). Such a plume is advected at mm/s (recalled, uncertain) and lives until it is diluted or circulated out (hours), during which it intercepts ~155/h × (A_p/A_TPC) × f_below events — for A_p = 1 cm² only 0.01/h, so it *is* statistically invisible; its improbability is in the required enrichment with no source (no calibration gas injection at that time; the 57Co source was in a sealed CSD tube, l.300).

## 6. Extraction / gas region and pulse pathologies (script §4)

- Extraction dip: local efficiency needed 0.80 × 0.219 = **0.175**; the S2c gain deficit ×4.56 (356 %) is **18× the typical ≲20 % map amplitude**. A static gate/anode sag or liquid-level tilt is in the 83mKr map (above). A transient surface wave or bubble in the gas gap changes the electroluminescence gap and hence the S2 width (gas transit ≈ 0.7–1 μs, recalled) — the S2 width matched the single-site template and the width-vs-drift cut.
- Reference scales (Fig. 1): g2 uncertainty 3.2 %; NR-band σ 7.5 %; S2 width tolerance 10–20 % (P022); lifetime-correction error 8–29 %; map amplitude 5–20 %: every instrumental scale is 3–25× below the required 78 %.
- Saturation/clipping: the alpha template LZ used for the S2-shape comparison consists of pulses of 2–4×10⁴ electrons, **75–149× the event's S2**, processed without shape distortion; the hypothetical true S2 (42 300 phd) is of the 83mKr class (27 100 phd), the calibration standard for g2 and the S2c map. Clipping a 42 300 phd pulse to 9 268 phd while 83mKr pulses calibrate the detector is self-contradictory.
- Truncated event window: keeping 22 % of a Gaussian S2 leaves a remnant with σ = **0.48σ_true and skew −1.25** — a 52 % width deficit against a 10–20 % width tolerance and a template match. Excluded.
- Mis-pairing: a 540 phd S1 paired with an unrelated 269-electron S2 is P022's accidental (1.5×10⁻⁴ modelled; ×708 needed; excluded at 4.3σ; cathode-emission loophole testable). The ER's own 42 300 phd S2 would have to be absent, i.e. the ER in a charge-dead region — the S1-only parent of the accidental, not a charge-loss artefact.
- MSSI hybrid (real 12 keV S2 + S1-only second deposit): P004 (wall, k ≥ 620 needed vs < 1.6 allowed); RFR MSSI disfavoured by LZ's rise-time and hit-pattern analyses (l.693–695). Not an artefact.

## 7. Channel summary (script §5; judgement factors, all printed)

The dossier's 0.18 is split by judgement and multiplied by a survival factor S = P(published checks pass and loss lands in V | channel):

| Channel | prior | factor S | residual | basis |
|---|---|---|---|---|
| bulk lifetime excursion | 0.03 | 4×10⁻⁵ | 1×10⁻⁶ | P(silent 300 s) 9×10⁻⁴ × τ-window 0.045 |
| local transient absorber | 0.05 | 1.2×10⁻² | 6×10⁻⁴ | p_V/(p_V+p_E) 0.11 × e⁻¹ × 0.3 transient-only |
| field distortion / charge trapping | 0.03 | 0.02 | 6×10⁻⁴ | static → Kr map; transient → as above |
| extraction / gas-gap dip | 0.02 | 0.02 | 4×10⁻⁴ | 18× map amplitude; transient → S2 shape |
| saturation / clipping / truncation | 0.02 | 10⁻³ | 2×10⁻⁵ | alpha template 75–149× larger; remnant σ 0.48 |
| mis-pairing (accidental) | 0.02 | 0.05 | 1×10⁻³ | P022 4.3σ, loophole |
| unknown pathology | 0.01 | 0.3 | 3×10⁻³ | untestable publicly |
| **total** | 0.18 | | **0.006** | shrink ×32 |

The residual is dominated by the deliberately generous "unknown pathology" entry and by the P022 loophole; every *named* charge-loss mechanism is ≤ 6×10⁻⁴. The normalised posterior depends on how the other hypotheses shrink (P004, P010, P013, P019, P022 all shrink theirs by ≥ 10²); we therefore report raw mass only.

**LZ-internal tests that would close the residual channels:** (1) electron lifetime from α and 83mKr S2/drift trends in the hour around 16 June 2023 21:22 UTC, with per-event granularity (84 long-drift monitor events per hour; a 60 s excursion is a 4-event test); (2) the S2c(x,y) 83mKr correction-map cell at the event's (x,y) for the injections before and after 16 June, and the map residual in the event's cell; (3) the S2 width of the event against the diffusion expectation 1.49 μs (P022's cathode template 1.65 μs); (4) a search for events with S2c 20–80 % below the ER band at S1c 200–800 phd in the *out-of-ROI* data (the Fig. 4 grey population), which any absorber with ε ≳ 10⁻³ populates with 65–100 events over the run; (5) the top-array hit pattern of the S2 (cloud σ 3 mm should give the template's PMT-pattern width; a partially absorbed cloud from a disk edge is asymmetric).

## 8. Validation and robustness

- σ_t = 1.49 μs reproduces P022 independently (D_L, v_d, t_ev). E_ee = 69.6 keV reproduces P010 with W = 13.44 eV.
- nestpy vs P010 N_e: 1340 vs 1226 (9 %); f_req 0.201 vs 0.219; all region windows shift by 0.04 dex, well inside the ±0.05 dex tolerance quoted.
- Fig. 4 digitisation: the P009 axis calibration is reproduced from the frame and ROI edges; the neutral-grey filter is validated by the colour census (contours are slate (87,103,103)).
- Parent population ×0.5–2: ε and N_all scale inversely, P(1 in V, 0 in empty | ε*) is independent of N_par.
- D_T 50–60 cm²/s: σ_T 2.9–3.2 mm; footprint ±10 %; disk results in σ units unchanged.
- Monitor rate 112–239/h: P(silent 300 s) 1.7×10⁻⁴–3.5×10⁻³.
- The absorbing disk is the *most favourable* absorber geometry for the "just-right" window (a soft-edged or partially transparent absorber broadens the loss distribution and lowers p_V/(p_V+p_E)).

## 9. Failed or abandoned approaches

- A first strip test with a fixed level (4.15–4.34 over S1c 330–800) returned 1366 "grey" pixels; inspection showed they were anti-aliased edges of ER-cloud points at S1c 330–520, where the cloud's envelope dips to 4.17–4.30. Replaced by a column-wise envelope and a S1c 520–800 test (0 pixels).
- A first bulk-lifetime factor used a 60 s silent window (0.25); rejected as unphysical for a bulk purity transient and replaced by 300 s plus the τ-window coincidence.
- Counting the grey points of Fig. 4 to measure the parent population directly was abandoned (saturated core); the Table I extrapolation is used with a ×0.5–2 bracket.

## 10. Figures

- `figures/P041_required_loss_vs_mechanism.png` — required S2 deficit (78 %) vs plausible size of each instrumental effect (lifetime correction error 8–29 %, map 5–20 %, g2 3.2 %, NR σ 7.5 %, width tolerance 10–20 %).
- `figures/P041_drift_time_and_lifetime.png` — left: FV drift-time range, event at 859 μs, cathode 1049 μs, gas events 56 μs, loss curves for τ = 566 μs and 5 ms; right: log₁₀S2c of a 70 keV ER vs drift time under τ = 566 μs with the landing regions.
- `figures/P041_disk_loss_distribution.png` — surviving-fraction distribution per hit for absorbing disks of 1.7σ, 3σ, 10σ with the landing regions shaded.

## 11. References

LZ Collaboration, arXiv:2609.02823 (2026); D. S. Akerib et al. (LZ), NIM A 953, 163047 (2020); J. B. Albert et al. (EXO-200), PRC 95, 025502 (2017); G. Bakale, U. Sowada, W. F. Schmidt, J. Phys. Chem. 80, 2556 (1976); J. Xu et al., PRD 99, 103024 (2019); J. Aalbers et al. (LZ), PRD 108, 012010 (2023); corpus P004, P009, P010, P022, P024; dossier 00.

## 12. Tools and provenance

Mirrors `output/provenance/P041.json`. Python 3.12.13 (`.venv/bin/python`); numpy 2.5.3; scipy 1.18.1 (`integrate.quad`, `special.i0e`, `stats.norm`); matplotlib 3.11.2 (Agg); Pillow 12.3.0 (Fig. 4 pixel digitisation); nestpy 2.1.1 via `common/lzcommon.py` (`nest_er_yields`, `combined_energy_keV`, `LZ`). Local inputs: PAPER_GUIDE, dossier, ledger, P004/P009/P010/P022/P024 papers, P022 details (drift geometry), P009 digitiser (axis calibration), fulltext.tex l.76–130, 139, 148–167, 294–305, 687–700, Fig4 PNG, ENVIRONMENT_versions. Recalled: 12 items (table in §2). No datasets, no data requests, no WimPyDD files.
