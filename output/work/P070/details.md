# P070 — Position as a discriminant: the (r, z) likelihood ratio between a uniform signal and wall/RFR MSSI at the LZ event, and what LZ left on the table by not using position

Simulated date 2026-09-12. Category BKG/STAT. Author profile: likelihood-modelling experimentalists. Script: `output/code/P070_position_likelihood.py` (run from the simulation root with `.venv/bin/python`; log `run_log.txt`; results `P070_results.json`, `LR_table.csv`, `scenarios.csv`, `fig3_points.csv`, `figS5a_points.csv`, `fv_contours.csv`, `profiles_d.csv`, `profiles_z.csv`).

## 1. Motivation and framework

The LZ paper (arXiv:2609.02823) states (supplement, l. 735): "While we use event positions to determine whether or not an event is in the FV, we do not use the position explicitly in the statistical inference." The same paragraph argues qualitatively that a ~12 keV MSSI scatter more than 20 cm from the TPC boundaries is unlikely. P033 quantified the geometry with a photon-transport Monte Carlo (the 12 keV wall-MSSI class falls as e^{−d/4.3 cm} from a simplified FV edge; f_pos ≈ 1 % at d ≥ 25 cm, z ≥ 20 cm), and P004 the mismodelling factor that wall MSSI would need. Neither asked the *statistical* question: what is the position likelihood ratio between a uniform-in-FV signal and each background at the event's coordinates, what Bayes-factor boost does it carry against the *full* background mixture, and how much would a three-dimensional (S1c, log₁₀S2c, position) likelihood have changed the single-event significance?

Framework. For one observed event at x with extended likelihood L(s) = e^{−(s+b)} [s f_s(x) + Σ_i b_i f_i(x)] (f are position PDFs normalised over the FV; the (S1c, S2c) part is common to both hypotheses and is absorbed into the neighbourhood expectations b_i), the profile-likelihood test statistic for s = 0 is

  q₀ = 2[ln(1 + ŝ/b_eff) − ŝ],  ŝ = 1 − b_eff,  b_eff(x) = Σ_i b_i f_i(x)/f_s(x) = Σ_i b_i / LR_i(x),   LR_i(x) ≡ f_s(x)/f_i(x),

so that adding position is exactly equivalent to replacing the neighbourhood background b = Σ b_i by b_eff = b·R_pos with R_pos = Σ w_i/LR_i (w_i = b_i/b), and Z = √q₀ = √(2[ln(1/b_eff) − 1 + b_eff]). The composite position likelihood ratio 1/R_pos is also the Bayes-factor boost of "uniform signal" against the background mixture from the position datum alone (the likelihood of the position under the signal hypothesis divided by that under the background mixture). For the signal PDF we take a uniform distribution in the fiducial mass; a WIMP interaction rate is uniform in LXe to far better than any effect considered here.

## 2. Inputs

| Input | Source |
|---|---|
| Fig. 3 and Fig. S5a as vector PDFs | `inputs/arXiv_2609.02823_source/Fig3_roi_events_in_fv.pdf`, `FigS5a_LE_MSSI_rz.pdf` (PyMuPDF `get_drawings`) |
| FV text numbers: stand-off 8.0 (6.0) cm true (reco); max 18.2 (10.4) cm near the bottom; mean 10.7 (7.2) cm; 12.8 cm below the gate; 9.0 cm above the cathode; 4.71 ± 0.08 t | tex l. 132–139 |
| Event: (45.9² cm², 26.4 cm); 26.9 (23.4) cm from the true (reco) wall | tex l. 118–121 (Fig. 3 caption), l. 166; `lz.LZ` |
| "we do not use the position explicitly" and the 20 cm / 60 cm argument | tex l. 735 |
| MSSI counts: wall 0.0048, RFR 0.0001 (4.7 t science); 5.4 t annulus 0.03 + 0.003 | tex l. 773–780 (Table of MSSI counts) |
| Fig. 3 / Fig. S5 captions (samples; ≥ 5σ from the ER band; min/max radial extent) | tex l. 118–128, 746–755 |
| P033 r–z maps of the wall12/rfr12/wall_roi classes per source and line (`h2_*` in `output/work/P033/runs/*_forced12.npz`, `*_general.npz`), line weights MIX_W and ²¹⁴Pb fractions from `P033_mssi_geometry_mc.py` | corpus P033 |
| f_nb = 0.035 (MSSI fraction at S1c > 500 phd within ±2σ_NR) | P004 |
| Accidentals in the neighbourhood 1.49 × 10⁻⁴; T_max 1049 μs; v_d 1.388 mm/μs; cathode-emission width-cut acceptance for drift > 850 (672) μs | P022 |
| Atmospheric-ν CEνNS at S1c > 500 phd: 3.4 × 10⁻⁵ | P019 |
| Neutron single scatters in 200–270 keV: 7 × 10⁻⁶ (≤ 5 × 10⁻⁴ adverse); λ_tot(10 MeV) = 14.2 cm | P013 |
| P033 energy-class fractions f_E = 0.053 (wall) / 0.067 (RFR) | P033 details §4.2 |
| N_eff = 12.2 (P008), 13.9 (LZ-implied, dossier) | P008, dossier |
| Recalled: TPC radius 72.8 cm (likely; the Fig. 3 axis ends at 73.0²), gate–cathode 145.6 cm (confirmed: the dashed active-volume lines sit at z = 0.000 and 145.60 cm), LXe density 2.86 g/cm³ (likely, 2.86–2.89), ²¹⁴Pb γ attenuation lengths 2.7/2.2/1.5 cm at 352/295/242 keV (likely; as used by P053), single-event PLR formula and Wilks asymptotics (certain), E₂ exponential integral for a plane sink (certain) | this paper |

## 3. Stage 1: exact digitisation of Fig. 3 and Fig. S5a

The figure PDFs are fully vectorised: 1710 filled black 8-segment paths (science sample: exactly the paper's 1710 events), 66 orange two-segment crosses (prompt veto: 66), 55 stroked blue circles (delayed veto: 55), one star, two solid black 1.4-pt polylines (min/max radial extent of the 4.7 t FV), two black 1.4-pt horizontal segments (upper/lower FV boundary), one dashed polyline (reconstructed wall = active volume) and, in Fig. S5a, the blue 5.4 t contours plus three orange crosses and one black dot. Axis calibration uses the seven major r²-ticks (0², 20², …, 70²) and eight z-ticks (0–140 cm): r² = 14.1040 cm²/pt × (x − x₀), max residual 4 × 10⁻⁴ cm²; z: 2.3318 pt/cm, residual 2 × 10⁻⁵ cm; the drift axis gives T_max = 1049.5 μs at z = 0 (P022: 1049 μs). The frame ends at r² = 73.0². The star's bounding-box centre is at r = 45.92 cm, z = 26.79 cm (the star glyph's anchor sits ≈ 0.4 cm below its bbox centre, so this is consistent with the caption's (45.9², 26.4)); the paper values are used for the event.

Contours (all in the plot's reconstructed coordinates):
- Upper boundary drawn at z = 135.75 cm, i.e. 9.85 cm below the 145.6 cm line (text: 12.8 cm → 132.8 cm); lower boundary at z = 9.64 cm (text 9.0 cm). Drawn height 126.1 cm vs 123.8 cm stated (+1.9 %). The 5.4 t volume is drawn from z = 2.15 to 135.75 cm.
- Radius of the 4.7 t contour: at the top 64.72 (min) / 65.38 (max) cm; at the bottom 54.64 / 59.21 cm; at the event's depth (26.4 cm) 58.73 / 61.22 cm, mean 59.97 cm. The dashed reconstructed wall: 71.07 cm at the top, 69.14 at mid-height, 67.94 at the bottom, 68.82 at the event's depth (→ 22.9 cm from the event, paper 23.4).
- Stand-off statistics (linear mean over z): distance to the true wall (72.8 − r): min-extent contour 8.08–18.16 cm (mean 11.81), max-extent contour 7.42–13.59 (mean 9.78); average of the two 10.8 cm — the paper's 10.7 cm and 18.2 cm are reproduced. Distance to the dashed reconstructed wall: 6.35–13.31 (mean 8.27) and 5.20–8.74 (mean 6.25); average 7.26 cm (paper 7.2 cm). The paper's minimum 6.0 and bottom maximum 10.4 cm (reco) fall between our two contours.
- **Event to the FV boundary at its depth: 14.1 cm (12.8–15.3 cm between the min and max contours)** — not the 18.9 cm implied by a uniform 8 cm stand-off, because the contour retreats from the wall below z ≈ 45 cm.

FV mass with ρ = 2.86 g/cm³, V = π∫r(z)²dz: min contour 4.22 t, mean 4.36 t, max 4.50 t (drawn z-range); 4.13 / 4.27 / 4.41 t with the text's z-range. The 5.4 t contours give 4.75 / 4.92 / 5.09 t. The active volume gives 6.93 t (LZ: 7.0 t). **The digitised contours fall 7–10 % short of both quoted masses**, while reproducing the paper's own stand-off statistics; the same shortfall for the 4.7 t and the 5.4 t volumes points to a common cause — most plausibly that the drawn contours are in reconstructed coordinates and the corresponding true-coordinate boundary lies ≈ 2.4 cm further out (since (1 + 2.4/62)² = 1.08), or a density/dimension convention. With the paper's own numbers (mean stand-off 10.7 cm from a 72.8 cm wall, height 123.8 cm) one obtains 4.29 t; 4.71 t would need ρ = 3.09 g/cm³ (mean contour) or 2.99 (max contour). We record this as an unresolved 8 % tension and note it does not affect any ratio below (all PDFs are normalised over the same digitised volume).

Uniform signal, distance distributions (mean contour, drawn z-range, V = 1.525 × 10⁶ cm³ = 4.36 t):

| distance | ≤ 2 cm | ≤ 5 | ≤ 10 | ≤ 15 | ≤ 20 | ≤ 25 | beyond the event |
|---|---|---|---|---|---|---|---|
| to the true wall | 0 | 0 | 0.5 % | 13.2 % | 27.5 % | 40.6 % | 54.5 % (event 26.9 cm) |
| to the reconstructed wall | 0 | 0 | 8.8 % | 23.4 % | 36.9 % | 49.0 % | 54.7 % (23.4 cm) |
| to the FV contour | 6.3 % | 15.5 % | 29.6 % | 42.5 % | 54.1 % | 64.3 % | 59.8 % (14.1 cm) |

The event sits at the median of every distance distribution of a uniform signal (45 % of the FV is farther from the wall). P033's position class (d ≥ 25 cm, z ≥ 20 cm) holds 54.5 % of the digitised FV.

## 4. Stage 2: position PDFs and likelihood ratios

Grid 0.2 × 0.2 cm in (r, z), volume element 2πr dr dz. Normalisation domain for the likelihood ratios: the digitised FV (mean contour) intersected with the support of the P033 maps (9 ≤ z ≤ 132.8 cm, d ≥ 8 cm); this removes 2.6 % of the drawn volume (the slice 132.8–135.75 cm). PDFs:

- **Uniform** (signal; accidentals — uniform in drift time (P022) and taken uniform in (x, y); atmospheric-ν CEνNS).
- **Wall MSSI, P033 maps.** `h2_wall12` (one 8–16 keV FV Compton deposit + 60–95 keV in the ≤ 3 mm wall shell, nothing else active) per source (wall, top, bottom, cathode planes), lines mixed with P033's MIX_W per emitted photon, "equal mix" = equal photon emission from the four surfaces; bulk ²¹⁴Pb (β + γ) map with P033's 352/295 keV fractions 0.515/0.485; and the general-mode `wall_roi` class (all wall MSSI in the ROI). The maps are (d, z) histograms (1 cm bins in d up to 30 cm, 2.5 cm to 65 cm; 5.02 cm in z); the per-volume density is h₂/(Δd·Δz_overlap·2πr), with the z-bin overlap with the simplified FV used at the two edge bins. Validation: re-integrating the maps over P033's simplified FV reproduces its f_pos: wall12 equal 0.00985 (P033 0.0098), wall 0.0137 (0.0137), top 0.0172 (0.0171), bottom 0.00184 (0.0018), cathode 0.00159 (0.0016), rfr12 0.00216 (0.0021), wall_roi 0.0108 (0.011).
- **Wall MSSI, analytic**: exp(−(d − 8)/λ) in the distance to the true wall, uniform in z, λ = 3, 4.3, 6 cm; variants with the distance to the dashed reconstructed wall and to the digitised FV contour as the reference.
- **RFR MSSI**: P033 `h2_rfr12` maps (detector-γ route: 12 keV Compton in the FV + 150–260 keV in the RFR), equal mix and bottom source; the ²¹⁴Pb route (β of ≈ 12 keV in the FV + γ into the RFR below the cathode): P(γ reaches z < 0 unscattered) = ½E₂(z/λ_γ) for an infinite plane sink, λ_γ = 2.7 / 2.2 / 1.5 cm (352 / 295 / 242 keV); and generic exp(−(z − 9)/λ_z), λ_z = 4, 6, 10 cm.
- **Neutrons**: exp(−d_surf/λ_n) with d_surf the distance to the nearest of wall, cathode plane and gate plane, λ_n = 10, 14.2, 15 cm (P013); wall-only variant.
- **Accidentals, adverse variant** (P022's open loophole): cathode-emission S2s pass the S2-width cut only for apparent drift > 850 (672) μs, i.e. z < 27.6 (52.3) cm; PDF uniform in that slab.

Real-FV re-normalisation changes the MSSI shapes materially: inside the digitised FV only 19.8 % of the wall12 class lies within 2 cm of an 8 cm stand-off (P033: 43 %, simplified FV) because the contour retreats to 13–18 cm from the wall below z ≈ 45 cm; the fitted depth scale (10–40 cm) becomes λ_eff = 4.82 cm (P033 4.38); f_pos over the real FV is 0.0210 (wall12) and 0.0127 (rfr12), versus 0.0098 and 0.0021 in the simplified FV — the position penalties of P033 were partly the bottom-corner population that LZ's real FV already excludes. The rfr12 class keeps 83 % of its weight at z < 20 cm with an exponential scale λ_z = 6.5 cm (12–60 cm).

**Likelihood ratios LR = f_signal/f_bkg at the event (r = 45.9, z = 26.4 cm)** (`LR_table.csv`):

| background | PDF | LR |
|---|---|---|
| wall MSSI | P033 map, equal photon mix | **15.1** (±1 cm/±5 cm box average 12.3) |
| | P033 map, wall source only | 49.9 (box 39.8) |
| | P033 map, top source | ∞ (MC zero: a 12 keV scatter 120 cm below the top array is unpopulated) |
| | P033 map, bottom source / cathode source | 2.27 / 2.42 (box 1.39 / 2.08) |
| | equal-weight mix of the four *normalised* source PDFs | 4.57 |
| | bulk ²¹⁴Pb map | 10.3 (box 12.1) |
| | all wall MSSI in the ROI (general mode) | 8.7 (box 17.6) |
| | exp from true wall, λ = 3 / 4.3 / 6 cm (z-uniform) | 21.7 / 5.74 / 2.63 |
| | exp from reconstructed wall, λ = 4.3 | 5.09 |
| | exp from the digitised FV contour, λ = 3 / 4.3 / 6 | 9.97 / 3.39 / 1.82 |
| | coarse position class (d ≥ 25, z ≥ 20), equal-mix map | 25.9 |
| RFR MSSI | P033 map, equal / bottom source | 1.45 / 1.35 |
| | ²¹⁴Pb E₂, λ_γ = 2.7 / 2.2 / 1.5 cm | 17.2 / 59.8 / 1514 |
| | exp(−(z−9)/λ_z), λ_z = 4 / 6 / 10 cm | 1.84 / 0.69 / 0.39 |
| | coarse position class, equal-mix map | 43.0 |
| neutrons | nearest surface, λ = 10 / 14.2 / 15 cm | 1.66 / 1.32 / 1.29 |
| | wall only, λ = 14.2 | 1.11 |
| accidentals | uniform | 1.00 |
| | cathode-emission slab (drift > 850 / > 672 μs) | 0.131 / 0.327 |
| atmospheric ν | uniform | 1.00 |

Reading. (i) Against *wall-sourced* wall MSSI the event's position is worth a factor 50; against P033's equal photon mix 15. But the bottom-array and cathode sources place their 12 keV scatters preferentially at low z (the class z-density falls as ≈ e^{−z/6.5 cm} from the FV bottom), and the event, 17 cm above the FV floor, is in that region: for those sources the position is worth only a factor 2.3–2.4. The wall-MSSI LR is therefore source-mix dependent, 2.3–50, with 4.6–15 for equal mixtures. (ii) The coarse position-class LRs (26 wall, 43 RFR), which is what an f_pos-type argument yields, overstate the *local* discrimination for the RFR class by a factor 30: after the real FV removes the bottom corner where the detector-γ RFR class peaks, what remains is spread across all radii at low z, and at the event its density is within a factor 1.4–1.7 of uniform. The ²¹⁴Pb RFR route, by contrast, is steep (LR 17–1500 by γ line). (iii) A z-uniform exponential from the FV contour, the parametrisation suggested in our assignment, gives only 3.4 at λ = 4.3 cm — the event is 14 cm, not 19 cm, inside the contour. (iv) If the event's true distance to the wall were the reconstructed 23.4 cm, the analytic LR at λ = 4.3 drops by e^{−3.5/4.3} = 0.45 (5.7 → 2.6). (v) Accidentals, CEνNS and (nearly) neutrons carry no position information; P022's cathode-emission accidental topology would be *favoured* by the event's long drift (LR 0.13).

## 5. Stage 3: population checks (`fig3_points.csv`, `figS5a_points.csv`)

Uniform-in-volume is uniform in the (r², z) plane, so a uniformity test is direct. Points are assigned u = r²/r_max(z)² and v = (z − z_bot)/(z_top − z_bot).

- **Science sample (1710)**: KS in u: D = 0.031, p = 0.080; in v: D = 0.043, p = 0.004, but a 10-bin χ² in z gives 10.8/9 d.o.f. (p = 0.29; pulls between −1.1 and +1.5; mean z 74.9 cm vs 72.7 for uniform) — a mild tilt, not a localized feature. Fraction within 10 cm of the mean FV contour 32.7 % (559) vs 29.6 % (507) for uniform (two-sided binomial p = 0.006); top 20 cm: 16.5 % vs 16.6 %; bottom 20 cm: 14.3 % vs 14.3 %. 68 events lie between the min and max contours (azimuth-dependent FV). The ER-dominated science sample is uniform to a few per cent with a 2.7σ excess near the radial boundary, as expected from residual detector-γ and wall-related ER events.
- **Delayed veto (55)**: KS p = 0.078 (u), 0.43 (v); 32.7 % within 10 cm of the contour — consistent with the caption's "random coincidences" (uniform).
- **Prompt veto (66)**: 57/66 = 86.4 % within 10 cm of the FV contour (uniform: 19.6 expected), 59/66 = 89.4 % in the top 20 cm (11.0 expected), 55/66 in both; KS p = 1 × 10⁻²³ (u) and 3 × 10⁻⁴³ (v); median distance to the contour 2.1 cm; 60/66 within 15 cm of the nearest surface (uniform 20.4 %, binomial p = 10⁻³⁴). Distance to the nearest surface (wall for 49, top for 17) follows an exponential with MLE scale λ = 3.3 cm above its minimum 7.9 cm. This is the empirical counterpart of the MSSI position PDF — the same "scatter in the FV, γ escapes to a dead/veto region" topology — and it confirms the e^{−d/(3–5 cm)} shape and the top-corner concentration from the top PMT array. Twelve crosses lie between the min and max contours.
- **Fig. S5a (≥ 5σ from the ER band in the WS ROI)**: the single black science dot lies at (45.92, 26.38) — it is the event itself, drawn under the star; there is no other science-sample event ≥ 5σ from the ER band. The three prompt-veto crosses are at (r, z) = (54.3, 3.5), (47.1, 2.9) and (61.5, 51.6) cm: 3.5 and 2.9 cm above the cathode (RFR-like, in the 5.4 t volume only) and 11.3 cm from the true wall / 0.4 cm inside the mean 4.7 t contour (wall-like). Under a uniform distribution in the 5.4 t volume the probability of lying within 4 cm of the cathode or 12 cm of the wall is q = 0.113, so P(all three) = q³ = 1.4 × 10⁻³; a KS test of the nearest-dead-region distance against uniform gives p = 0.005 for the three crosses and p = 0.058 when the event (nearest dead region 26.4 cm) is added: the vetoed NR-band-ish events are wall/RFR-concentrated exactly as the MSSI model expects, and the event does not belong to that population (a uniform point is ≥ 25 cm from both dead regions with probability 0.46). Per-point LRs over the 5.4 t volume (analytic PDFs): crosses 1.3 / 7.0 / 0.25 against wall-exp(4.3 cm) and 0.024 / 0.018 / 9 × 10⁶ against ²¹⁴Pb-RFR — the two cathode crosses are 40–60× more RFR-like than uniform, the wall cross 4× more wall-like; the event is 9.3× and 500× more uniform-like than wall- or ²¹⁴Pb-RFR-like.

## 6. Stage 4: counterfactual 3D likelihood

Neighbourhood expectations (S1c > 500 phd, within ±2σ_NR of the NR median), 2.84 t·yr:

| component | b_i | w_i | position PDF (central) | LR_i |
|---|---|---|---|---|
| wall MSSI | 0.0048 × 0.035 = 1.68 × 10⁻⁴ | 0.464 | P033 equal-mix map | 15.1 |
| RFR MSSI | 1 × 10⁻⁴ × 0.035 × 0.067/0.053 = 4.4 × 10⁻⁶ | 0.012 | P033 rfr12 map | 1.45 |
| accidentals | 1.49 × 10⁻⁴ (P022) | 0.411 | uniform | 1 |
| atmospheric ν | 3.4 × 10⁻⁵ (P019, all taken inside) | 0.094 | uniform | 1 |
| neutrons | 7 × 10⁻⁶ (P013) | 0.019 | nearest-surface exp, 14.2 cm | 1.32 |
| total | 3.62 × 10⁻⁴ | | | |

Poisson-in-a-box checks: Z_box(3.62 × 10⁻⁴) = 3.72, Z(P(≥ 1)) = 3.38; LZ's 3.4σ corresponds to b_anchor = 1.14 × 10⁻³ in the box formula (P001: 3.4 × 10⁻⁴–1.1 × 10⁻³; P016's b_H = 5.7 × 10⁻⁴ gives 3.60 without companions). We anchor at b_anchor and scale by R_pos; the result is insensitive to the anchor (ΔZ changes by 0.015 between the two).

**Central scenario**: R_pos = 0.464/15.1 + 0.012/1.45 + 0.411 + 0.094 + 0.019/1.32 = **0.559**; composite position likelihood ratio (Bayes-factor boost) **1.79**; b_eff = 6.36 × 10⁻⁴; **Z_local 3.40 → 3.57 (+0.17σ)**; global (N_eff = 12.2) 2.64 → 2.85σ, (13.9) 2.60 → 2.81σ. The hard cap, if MSSI were removed entirely by position, is R_pos = 0.520 (ΔZ = +0.19σ, boost 1.92): **position cannot buy more than 0.19σ because 52 % of the neighbourhood background (accidentals, CEνNS, neutrons) is uniform or nearly so.**

Scenarios (`scenarios.csv`): wall = 50 % ²¹⁴Pb bulk + 50 % detector-γ: +0.17; steepest (λ = 3 cm, ²¹⁴Pb RFR): +0.18; wall-source map + ²¹⁴Pb RFR (optimistic): +0.18; top-source map: +0.18; reconstructed-wall reference (λ = 4.3): +0.14; FV-edge reference: +0.12; flattest analytic (λ = 6 cm, RFR λ_z = 10 cm): +0.09; bottom-source or cathode-source wall maps (pessimistic): +0.09 / +0.10; **adverse**: accidentals as the cathode-emission slab (drift > 850 μs): R_pos = 3.28, boost 0.31, b_eff = 3.7 × 10⁻³, Z 3.40 → 3.03 (−0.37σ), global 2.64 → 2.18σ. Range excluding the adverse case: **+0.09 to +0.19σ locally, +0.11 to +0.23σ globally**.

Expected gain for a true uniform signal (distribution of ΔZ(x) over the FV, central scenario): median +0.18σ, mean +0.12σ, 16–84 % range +0.11 to +0.19; position *lowers* Z for a signal event in 10 % of the FV (the outer shell where the MSSI PDF exceeds uniform) — 31 % in the flattest scenario. Fig. 2.

Exposure left on the table by the hard cut instead of a position weight: the 5.4 t analysis volume is 1.146 × the 4.71 t FV (0.42 t·yr of the 220 live days discarded); its annulus holds 0.033 wall + RFR MSSI in the science WS ROI, i.e. 1.2 × 10⁻³ in the neighbourhood before position weighting — a likelihood with position PDFs would have retained ~15 % more signal exposure while assigning the annular MSSI its steep radial weight.

## 7. Figures

- `figures/P070_fig1_rz_and_profiles.png` — Left: vector-digitised Fig. 3 (science dots, delayed-veto circles, prompt-veto crosses, star) with the 4.7 t min/max contours (black), the 5.4 t contours (dotted), the dashed reconstructed wall, the P033 position-class boundaries (dash-dot) and, as a single-hue background, log₁₀ of the position likelihood ratio uniform/wall-MSSI (P033 equal-mix map, normalised over the digitised FV). Right: normalised density versus distance to the true wall for the wall-MSSI (P033 equal mix; bulk ²¹⁴Pb), RFR-MSSI, neutron and uniform PDFs, with λ = 3 and 6 cm exponentials; the event at 26.9 cm.
- `figures/P070_fig2_counterfactual_Z.png` — Left: single-event Z versus effective background with LZ's 3.4σ anchor and the position-augmented point (central scenario). Right: distribution over the FV of ΔZ for a uniform signal event (central scenario) with the observed event's +0.17σ.

## 8. Failed or abandoned

- Pixel digitisation of the PNGs (planned) was superseded by the exact vector paths of the PDFs.
- A per-source *renormalised* equal mix (4.6) and P033's equal-photon mix (15.1) differ by ×3; neither is "the" LZ source mix — reported as a range.
- The top-source map has zero MC content at the event (LR = ∞); it enters only through the equal mix.
- Using single map bins for the reconstructed-distance variant (r = 49.4 cm) gave LR values above the 26.9 cm ones (22 vs 15) because of MC noise in 1 × 5 cm bins; the analytic variant is used for that sensitivity instead.

## 9. Discussion

Position is a strong discriminant against *wall-sourced* wall MSSI (factor 50) and against ²¹⁴Pb-driven RFR MSSI (17–1500), moderate against the equal-mix wall class (15), weak against bottom-array-sourced MSSI (2.3) and the detector-γ RFR class (1.45), and neutral or adverse against accidentals and CEνNS. Because the latter make up half of the modelled background at the event's (S1c, S2c), a 3D likelihood would have moved the local significance from 3.4σ to 3.5–3.6σ (global 2.6 → 2.75–2.85σ). Our assignment anticipated ×10–100 against wall MSSI (found 2–50, source-dependent), ×5 against RFR (found 1.4–1500, route-dependent), ×1 against accidentals (found 1, or 0.13 in the cathode-emission topology) and +0.3–0.5σ (found +0.09–0.19σ, capped by the uniform backgrounds). The larger lesson is the one LZ's sentence implies: the FV cut already removed the region where position would have mattered, so the remaining discriminating power against MSSI is a factor ~2 on the total, and the event's position is exactly typical of a uniform signal (median distance to every boundary). Conversely, the empirical prompt-veto sample confirms the exponential surface concentration (λ ≈ 3.3 cm) that P033 derived, and the three vetoed NR-band-ish events of Fig. S5a sit where MSSI must (p = 0.005 against uniform), while the event does not.

## 10. References

LZ Collaboration, arXiv:2609.02823 (2026). G. Cowan, K. Cranmer, E. Gross, O. Vitells, Eur. Phys. J. C 71, 1554 (2011). R. E. Kass and A. E. Raftery, J. Am. Stat. Assoc. 90, 773 (1995). A. N. Kolmogorov, G. Ist. Ital. Attuari 4, 83 (1933); N. V. Smirnov, Ann. Math. Stat. 19, 279 (1948). E. Gross and O. Vitells, Eur. Phys. J. C 70, 525 (2010). Corpus: P004, P008, P013, P016, P019, P022, P033, P053, dossier 00.

## 11. Tools and provenance

- Agent tools: Read (PAPER_GUIDE.md; dossier; ledger rows P001–P012 and P013/P016/P022/P024/P033/P045–P050 via grep; P033.md, P004.md, P022.md, P029.md, P053.md, P016.md, P013.md; tex lines 116–141, 164–168, 733–736, 746–783; Fig3 and FigS5a PNGs; P033 summary/post JSON, one run JSON/npz, script lines 196–250 and 605–632; lzcommon.py lines 21–60 and API grep; P016 results JSON; P004 Fig. 5 CSV; P022 script/details grep; ENVIRONMENT_versions.txt grep), Bash (ls of papers/work dirs; greps of the tex, ledger, P033/P022 scripts; three PyMuPDF exploration scripts; two runs of the P070 script; wc -w), Write (script, details.md, P070.json, P070.md), Edit (script: ²¹⁴Pb weights, S5a duplicate handling, extra scenarios, histograms, label position, printing), Skill (dataviz; palette values read, JS validator not run per the guide).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (stats.kstest, kstwo, binom, binomtest, chi2, norm; special.expn; optimize.brentq); pandas 3.0.5; matplotlib 3.11.2 (Agg, pcolormesh, LogNorm unused); pymupdf 1.28.2 (Document, Page.get_drawings, get_text); common/lzcommon.py (LZ constants).
- Script and command: `output/code/P070_position_likelihood.py`, run as `.venv/bin/python output/code/P070_position_likelihood.py` from `/Users/reza/LZ_simulation` (≈ 1 min).
- Recalled knowledge: 7 items (TPC radius 72.8 cm — likely; gate–cathode 145.6 cm — confirmed by the figure; LXe density 2.86 g/cm³ — likely; ²¹⁴Pb γ attenuation lengths — likely; single-event PLR/Wilks formula — certain; E₂ plane-sink escape probability — certain; matplotlib star-marker anchor offset — likely).
- Datasets: none. Data requests: none. WimPyDD-generated files: none.
