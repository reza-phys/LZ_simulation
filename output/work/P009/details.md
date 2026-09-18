# P009 — research record: 248 or 265 keV? Reconstructing the recoil energy of the LZ event with the tuned NEST model

Simulated arXiv date 2026-09-04. Author profile: liquid-xenon detector-response modellers. Category RESP; physics.ins-det (cross-list hep-ex). Stance: neutral (tool/response).

All numbers below are produced by `output/code/P009_digitise_bands.py` (figure digitisation → `output/work/P009/digitised_fig.json`) and `output/code/P009_energy_reconstruction.py` (physics → `results.json`, `yield_curves.csv`, `estimators_1D.csv`, `likelihood_scan.csv`, figures). Run from the simulation root with `.venv/bin/python`. Total run time ≈ 40 s.

## 1. Motivation and framework

The LZ paper (arXiv:2609.02823) quotes the event at S1c = 540.1 phd, S2c = 9268 phd as an elastic nuclear recoil of 248 ± 23 (stat) ± 23 (sys) keV, 1.5σ below the NR-band median, in a search whose efficiency crosses 50 % at 269.9 keV. The evidence dossier (`output/00_evidence_dossier.md`, §2 check (i)) noted that the tuned NEST parameters of Supplemental Table S5, evaluated with nestpy, put the NR-band centre for 248 keV at S1c ≈ 498 phd, so that S1c = 540 phd corresponds to ≈ 265 keV. Whether the event is at 248 or 265 keV matters: at 265 keV it sits at the efficiency edge (ε ≈ 0.6) and 35 phd from the S1c < 600 phd ROI boundary; at 248 keV it is on the plateau. It also fixes the total-quanta ("keVee") energy used to compare with the 124Xe/125I double-vacancy lines, and it shifts the kinematic reach in inelastic splitting δ.

We (a) reproduce the tuned mean yields in nestpy 2.1.1, (b) build four energy estimators including a two-dimensional (S1c, log10 S2c) Monte-Carlo likelihood using NEST's own quanta fluctuations, (c) digitise the paper's Figs. 2 and 4 to obtain the collaboration's *own* S1c–energy relation along the NR band, and (d) use the Fig. S2 efficiency inset as an independent check of the S1 scale and S1 resolution.

## 2. Inputs

| Input | Value | Source |
|---|---|---|
| g1, g2 | 0.110 ± 0.002 phd/photon, 34.5 ± 1.1 phd/electron | paper, detector paragraph (fulltext.tex l. 88); `lz.LZ` |
| Event | S1c = 540.1 phd, S2c = 9268 phd → log10 S2c = 3.9670 | paper, Results (l. 165); Fig. 4 caption says "10^3.98", which is a rounding inconsistency (3.967 rounds to 3.97) |
| ROI | S1c 3–600 phd, S2c 10^2.75–10^4.15 phd | paper, Data Analysis (l. 114) |
| Quoted energy | 248 ± 23 ± 23 keV; 1.5σ below NR median; 6.7σ below ER | l. 15, 165 |
| Efficiency | 50 % at 269.9 keV; plateau 0.955 (96 % average 14–250 keV) | Fig. S2 caption (l. 456), l. 117; inset read by eye (Table 5 below) |
| NEST NR means | Table S5: α = 11.2, β = 1.1, γ = 0.052, δ = −0.0533, ε = 10.8, ζ = 0.53, η = 1.4, θ = 0.31, ι = 2.5, p = 0.50, f1 = 1.39, f2 = 1.74, a = 0.0230, b = 0.0289, E0 = 74.7 keV; defaults 11.0, 1.1, 0.048, −0.0533, 12.6, 0.30, 2.0, 0.30, 2.0, 0.5, 1, 1 | supplement l. 654–685; `lz.NEST_NR_LZ`, `lz.NEST_NR_DEFAULT` |
| p(E) break | p(E) = 0.5 + a ln[1 + b (E − E0)] above E0 | supplement Eq. (l. 602); implemented in `lz.nest_nr_params_vector` |
| ER means | Table S3 (m1…m10) | `lz.NEST_ER_LZ` (used only for the ER-band sanity check) |
| Field, density | 96.5 V/cm, 2.9 g/cm³ | nestpy `LZ_WS2024` central field; `lz.DRIFT_FIELD_VCM` |
| nestpy detector `LZ_WS2024` | g1 = 0.1122 (object value, **not used**: we apply the paper's g1 = 0.110 and g2 = 34.5 externally), NR width parameters [0.404, 0.393, 0.0383, 0.497, 0.1906, 2.22, 0.3, 0.04311, 0.15505, 0.46894, −0.26564, 0, 0], NR mean parameters [10.19, **1.11**, 0.0498, −0.0533, 12.46, 0.2942, 1.899, 0.3197, 2.066, 0.509, 0.996, 0.999], extraction efficiency 0.726 and g2 = 34.02 from `CalculateG2`, s2Fano = 4.0, sPEres = 0.338, P_dphe = 0.214 | nestpy 2.1.1 package (software input) |
| Work function | nestpy `WorkFunction(2.9).Wq_eV` = 13.44 eV; 13.7 eV (older NEST v2 / LUX Doke analyses) and 13.5 eV (literature) — recalled, likely | |
| Fig. 2 / Fig. 4 PNGs | band lines, event marker, grey constant-energy contours with labels (E_nr, E_ee) = (50, 11.8), (100, 25.5), (150, 40.0), (200, 55.1), (250, 70.6), (300, 86.5) | `inputs/figures_png/`, digitised (§4) |

## 3. Equations

Mean yields (NEST v2 NR model, recalled/likely as to the functional form; evaluated numerically by nestpy so the form is not load-bearing):
Nq(E) = α E^β; Qy(E) = [γ F^δ]⁻¹ [ε + E^p]⁻¹ (1 − [1 + (E/ζ)^η]⁻¹) with the LZ break p → p(E); Ne = Qy E; Nph = Nq − Ne (with the f1, f2 turn-on factors). Then S1c = g1 Nph, S2c = g2 Ne.

Estimators: E_S1 solves g1 Nph(E) = S1c; E_S2 solves log10(g2 Ne(E)) = log10 S2c; E_comb solves Nph(E) + Ne(E) = S1c/g1 + S2c/g2 = Nq_ev; the "paper-style" inversion is E = (Nq_ev/α)^{1/β}. Event quanta: Nph = 4910.0, Ne = 268.64, Nq = 5178.6.

Detector Monte Carlo at fixed E (N = 20 000 per energy, E = 180–320 keV in 2 keV steps): (Nph, Ne) ~ `NESTcalc.GetQuanta(yields, 2.9, width_LZ)`; n1 ~ Binomial(Nph, g1); S1c = n1 [1 + N(0, 0.338/√n1)] [1 + N(0, σ_pos)]; n_ext ~ Binomial(Ne, 0.726); S2c = n_ext (g2/0.726) [1 + N(0, √(4/(n_ext g2/0.726)))] [1 + N(0, σ_pos)], with σ_pos = 0.02 an assumed residual of the position corrections (varied 0 and 0.04). g1 is treated as phd/photon so the DPE is absorbed. The likelihood at each E is a bivariate normal fitted to the 20 000 (S1c, log10 S2c) samples, evaluated at the event; cross-checked with a Gaussian KDE every 10 keV (agreement within MC noise, Fig. 2). E_ML and the quadratic σ come from a parabola fitted to ln L within ±25 keV of the maximum; the 68 % interval is Δ ln L = 0.5 on the interpolated profile.

Constant-energy contours in Figs. 2/4: if they are lines of constant total quanta, then along each contour E_ee = W (S1c/g1 + S2c/g2), so W_implied = E_ee /(S1c*/g1 + S2c*/g2) at the digitised crossing (S1c*, S2c*) with the NR median; and the label pair (E_nr, E_ee) gives the collaboration's Nq(E_nr) = 10³ E_ee/W.

Efficiency roll-off model: ε(E) = 0.955 Φ((269.9 − E)/σ_E), σ_E fitted to the inset readings. MC efficiency: ε_MC(E) = 0.955 [1 − P(S1c > 600 | E)].

Kinematics: δ_max(E_R, m) = v √(2 m_N E_R) − m_N E_R/μ (`lz.delta_max_kev`) with v = v_E(16 June) + v_esc = 809.1 km/s.

## 4. Digitisation of Figs. 2 and 4 (`P009_digitise_bands.py`)

Both PNGs (1239 × 1039) share the axes frame: columns 143–1185 ↔ S1c 0–800 phd; the dashed ROI box gives the y-calibration (rows 363 and 896 ↔ log10 S2c 4.15 and 2.75). Checks: the ROI right edge is recovered at S1c = 601.2 phd (true 600; 0.2 %); the frame top maps to 5.014 (true 5.0; 1 pixel = 0.0026 dex). The event marker in Fig. 4 is recovered at (539.8 phd, 3.970), i.e. within one pixel of (540.1, 3.967).

NR band (Fig. 2, identical in Fig. 4) at S1c = 540.1: p10 = 3.9934, median = 4.0207, p90 = 4.0487 → 10–90 % half-width 0.0276 dex (Gaussian-equivalent σ = 0.0276/1.2816 = 0.0216 dex). Other S1c: 400 → (3.955, 3.983, 4.012); 500 → (3.984, 4.012, 4.039); 600 → (4.007, 4.034, 4.061); 700 → (4.027, 4.053, 4.080).

Contour crossings with the NR median (Fig. 4, uncontaminated by calibration points; the 100 keV crossing at 183.3 phd falls outside the assignment window and the 50 keV one is below the S1c > 120 phd search range):

| E_nr [keV] | E_ee [keVee] | S1c at NR median [phd] | log10 S2c there | W_implied [eV] |
|---|---|---|---|---|
| 150 | 40.0 | 298.8 | 3.9424 | 13.465 |
| 200 | 55.1 | 418.8 | 3.9931 | 13.463 |
| 250 | 70.6 | 543.8 | 4.0167 | 13.462 |
| 300 | 86.5 | 671.1 | 4.0495 | 13.461 |

W_implied is constant to 0.03 %: the grey contours are lines of constant S1c/g1 + S2c/g2 with W = 13.46 eV (the nestpy work function is 13.44 eV; they are not W = 13.7 eV lines). Fitting Nq = 10³ E_ee/W_paper to the six label pairs gives Nq(E) = 11.32 E^1.112 (α = 11.325, β = 1.1117). By linear interpolation between the 200 and 250 keV crossings the event's S1c = 540.1 corresponds to **248.5 keV**; interpolating its E_ee = 5178.6 × 13.46 eV = 69.72 keVee between the labels gives **247.2 keV**. The paper's 248 keV is therefore exactly what its own figure implies; the 250 keV_nr contour passes 3.7 phd to the right of the event.

In Fig. 2 the AmBe points create spurious crossings at 454, 483 and 513 phd; the script keeps only crossings matching the constant-Nq spacing (250 and 300 keV survive, at the same 543.8 and 671.1 phd).

## 5. Results

### 5.1 Mean-yield curves (`yield_curves.csv`, Fig. 1)

Five variants: LZ-tab (Table S5 as printed, with break), LZ-nobreak, NEST-default, LZ-β1.11 (Table S5 with β = 1.11, the value stored in nestpy's LZ_WS2024 detector), paper-contour (Nq = 11.32 E^1.112 from the figure labels with the LZ-tuned Ne, so Nph = Nq − Ne).

| variant | S1c(248) | log10 S2c(248) | S1c(250) | S1c(270) | Nq(250) |
|---|---|---|---|---|---|
| LZ-tab | 497.8 | 4.017 | 502.4 | 548.8 | 4869 |
| LZ-nobreak | 489.2 | 4.116 | 493.8 | 539.4 | 4869 |
| NEST-default | 476.4 | 4.149 | 480.9 | 525.5 | 4782 |
| LZ-β1.11 | 527.8 | 4.017 | 532.8 | 582.4 | 5145 |
| paper-contour | 538.7 | 4.017 | 543.7 | 594.4 | 5245 |

The p(E) break (p = 0.541 at 250 keV) lowers Ne from 380 to 302 at 250 keV (−21 %, −0.10 dex in S2c) and raises Nph by 1.7 %. NEST defaults put the band 0.13 dex above the LZ-tuned band at 248 keV; the tuned charge yield is what places the event 0.05 dex below the median rather than 0.18 dex.

The paper's S1c scale (contour crossings) exceeds LZ-tab by 299/277 = 1.077 (150 keV), 419/388 = 1.079 (200), 544/502 = 1.083 (250), 671/619 = 1.084 (300). In total quanta the excess is 5245/4869 = +7.7 % at 250 keV; β = 1.11 recovers 5145 (+5.7 %), leaving 2 %.

### 5.2 One-dimensional estimators (`estimators_1D.csv`)

| variant | E_S1 [keV] (g1 ∓ 2 %) | E_S2 [keV] (g2 ∓ 3 %) | E_comb [keV] (g1,g2 ± ) | (Nq/α)^{1/β} |
|---|---|---|---|---|
| LZ-tab | 266.2 (270.5 / 262.1) | 176.7 (194.3 / 161.3) | 264.4 (260.0 / 269.0) | 264.7 (β = 1.1) |
| LZ-nobreak | 270.3 (274.7 / 266.1) | 129.6 (137.7 / 122.3) | 264.4 | — |
| NEST-default | 276.5 (281.0 / 272.2) | 113.3 (120.2 / 107.0) | 268.8 (264.3 / 273.5) | — |
| LZ-β1.11 | 253.0 (257.0 / 249.1) | 176.7 | 251.5 (247.3 / 255.8) | 251.7 (β = 1.11) |
| paper-contour | 248.6 (252.5 / 244.7) | 176.7 | 247.2 (243.0 / 251.4) | 247.2 |

The S2-only estimator is useless here: d log10 S2c/dE = 0.0006 dex/keV, so the 0.03 dex band width corresponds to ±50 keV and a 3 % g2 change moves it by 17 keV; the event's low S2 drives E_S2 far below the S1 value in every variant. S1 and combined-quanta estimators agree within 2 keV because Ne is only 5 % of Nq.

### 5.3 Two-dimensional Monte-Carlo likelihood (`likelihood_scan.csv`, Fig. 2)

MC widths at E ≈ 250 keV: σ(S1c) = 25–26 phd (5.0 %), σ(log10 S2c) = 0.032 dex (quanta alone 0.0254 dex; binomial extraction + SE Fano add 0.018 dex; σ_pos adds 0.009 dex), correlation −0.03 after detection (−0.24 at the quanta level).

| variant | E_ML [keV] | σ (quadratic) | 68 % interval | ε(E_ML) | ε at 68 % edges | P(S1c > 600 at E_ML) | E50 from MC |
|---|---|---|---|---|---|---|---|
| LZ-tab | 262.3 | 10.9 | 251.1–272.6 | 0.71 | 0.90 / 0.39 | 0.5 % | 291.3 |
| LZ-nobreak | 251.7 | 10.0 | 241.4–261.4 | 0.90 | 0.95 / 0.73 | 0.0 % | 295.8 |
| NEST-default | 251.6 | 9.6 | 241.6–262.6 | 0.90 | 0.95 / 0.70 | 0.0 % | 302.4 |
| LZ-β1.11 | 249.9 | 10.0 | 239.2–259.1 | 0.91 | 0.95 / 0.78 | 0.5 % | 276.6 |
| paper-contour | 245.8 | 9.8 | 235.6–254.9 | 0.93 | 0.95 / 0.86 | 0.7 % | 271.7 |

E_ML is 4 keV below E_S1 in the LZ-tab and paper-contour cases because the low S2 pulls weakly toward lower E (the median log10 S2c rises with E). For LZ-nobreak and NEST-default the S2 lies 0.15–0.18 dex (5–6σ) below the model band at every E; their E_ML is meaningless as an energy but shows that untuned charge yields cannot accommodate the event at all.

Systematics (8000 MC per energy): LZ-tab: g1 ∓ 2 % → E_ML 266.3 / 258.4 (±3.9 keV); g2 ∓ 3 % → 263.3 / 261.3 (±1.0); σ_pos = 0 / 0.04 → 262.6 / 261.8 with σ = 9.6 / 12.8 keV. paper-contour: g1 → 249.7 / 242.2 (±3.7); g2 → 246.8 / 244.9; σ_pos = 0 / 0.04 → 245.9 / 245.5 with σ = 8.9 / 12.1 keV. The yield model dominates: LZ-tab vs paper-contour is 16.5 keV, vs LZ-β1.11 12.4 keV.

Statistical error: our 68 % half-width is 10–11 keV, less than half of the paper's ±23 keV (stat). The S1 resolution implied by the paper's own roll-off (σ_E = 11.8 keV, §5.5) supports a ~10–12 keV S1-driven energy resolution at 250–270 keV; we cannot reproduce ±23 keV as a purely statistical S1/S2 error with NEST fluctuations (it would need an S1 spread of ~9 %). Possibly the paper's "stat" includes the recombination-fluctuation degeneracy between Nph and Ne in a combined-energy estimator, or a flat-spectrum posterior with wider band widths; this is left as a question for the collaboration.

### 5.4 NR band at fixed S1c and the "1.5σ" (`results.json: band_fixed_S1c`, `digitised.band_at_540`)

Flat spectrum 150–400 keV, |S1c − 540.1| < 5 phd (6400–6900 events):

| | p10 | median | p90 | half-width 10–90 | σ | event offset [dex] | σ below median |
|---|---|---|---|---|---|---|---|
| MC, LZ-tab | 3.9837 | 4.0250 | 4.0664 | 0.0414 | 0.0325 | 0.058 | 1.78 |
| MC, paper-contour | 3.9750 | 4.0151 | 4.0597 | 0.0423 | 0.0329 | 0.048 | 1.46 |
| digitised Figs. 2/4 | 3.9934 | 4.0207 | 4.0487 | 0.0276 | 0.0216 | 0.0537 | 2.49 |

The paper's 1.5σ is reproduced by the MC with the paper-contour energy scale (1.46σ) and NEST-2.1.1 fluctuation widths (σ = 0.033 dex), but the dashed 10–90 % lines drawn in Figs. 2 and 4 are 1.5× narrower (σ_equiv = 0.022 dex) and would make the event a 2.5σ outlier. The 1.5σ statement requires σ ≈ 0.036 dex at S1c = 540 (offset 0.0537/1.5). Either the drawn percentiles understate the band width used in the inference, or "σ" in the text is not the Gaussian-equivalent of the 10–90 % range. The mean energy of events in the S1c = 540 ± 5 slice is 266.7 keV (LZ-tab) and 249.2 keV (paper-contour), again illustrating the two scales.

### 5.5 Efficiency (Fig. 3)

Inset readings (E, ε): (250, 0.93), (255, 0.88), (260, 0.79), (265, 0.66), (270, 0.50), (275, 0.35), (280, 0.22), (285, 0.12), (290, 0.05), (295, 0.02), (300, 0.005), uncertainty ±0.03 each. Fit with E50 = 269.9 and plateau 0.955 fixed: σ_E = 11.8 keV, maximum residual 0.032. ε(248) = 0.924, ε(265) = 0.631, ε(271) = 0.442.

Independent S1-scale check: the MC roll-off 0.955 [1 − P(S1c > 600 | E)] reproduces the inset almost point by point for the paper-contour scale (E50,MC = 271.7 keV, median S1c(269.9) = 594 phd) and gives E50 = 276.6 keV for β = 1.11, but E50 = 291.3 keV for Table S5 as printed (median S1c(269.9) = 549 phd) and 296–302 keV for the no-break/default variants. The paper's 50 % point at 269.9 keV is therefore itself a measurement of the S1 scale: it requires S1c(270 keV) ≈ 600 phd, 9 % above the nestpy evaluation of Table S5, and confirms the contour-derived scale. The roll-off width also validates the S1 resolution model (σ_S1c/S1c ≈ 5 % with σ_pos = 0.02).

Consequences: on the paper's scale the event (245.8 keV) is on the plateau (ε = 0.93; even the upper 68 % edge has ε = 0.86) and P(S1c > 600 | E_ML) = 0.7 %; on the Table-S5 scale it is at ε = 0.71 with ε = 0.39 at the upper edge, and a recoil at 272 keV fails the S1c cut 4 % of the time, at 280 keV 16 %.

### 5.6 Kinematics and ER-equivalent energy

δ_max at v = 809.1 km/s (16 June): m = 1000 GeV: 386.6 (248 keV), 387.0 (250), 389.9 (265 keV); m = 400 GeV: 341.1 / 341.2 / 341.3; m = 4000 GeV: 409.4 / 410.0 / 414.3; time-average (v = 794.6 km/s), 1000 GeV: 374.6 / 375.0 / 377.6. Minimum elastic mass: 72.8 (248) → 76.8 GeV (265). The 248-vs-265 ambiguity shifts δ_max by only 3 keV at 1000 GeV: the inelastic reach quoted in the dossier is robust.

ER-equivalent energy of the event: E_ee = W Nq = 70.9 keVee (W = 13.7 eV), 69.9 (13.5), 69.6 (nestpy 13.44), 69.7 (paper contours, 13.46). With g1 ± 2 %, g2 ± 3 %: ±1.3 keVee. This is 2.4 keV (1.8σ) above the 125I double-vacancy line (67.3 keV) and 5.4 keV (4σ) above 124Xe (64.3 keV). The dossier's "~64 keVee" is a paraphrase of the paper's statement that these lines are "similar to the reconstructed energy of the event of interest if interpreted as a pure ER"; the paper's Fig. 4 shows the 70.6 keVee contour essentially through the event, so the paper's own ER-equivalent energy is ≈ 70 keVee and does not coincide with either line. A 69.7 keV β-like ER with the LZ ER model would sit at S1c = 435 phd, log10 S2c = 4.63 with electron fraction 0.237; the event's electron fraction is 0.052, i.e. an ER explanation needs recombination of ≈ 0.95 (as in the dossier).

## 6. Validation and robustness

- Digitisation: ROI edge 601.2/600; frame top 5.014/5.0; event 539.8/540.1 and 3.970/3.967; W_implied constant to 0.03 % across four contours (a non-trivial check that the contour interpretation is right).
- nestpy means agree with the dossier numbers (S1c(248) = 497.75 vs 497.75; log10 S2c = 4.0165 vs 4.0165).
- KDE vs bivariate-normal likelihood: agree within MC noise (Fig. 2 crosses).
- MC roll-off reproduces the Fig. S2 inset for the paper-contour scale (E50 271.7 vs 269.9; shape within the ±0.03 reading error).
- σ_pos 0 → 0.04 changes E_ML by < 1 keV and σ from 9 to 12 keV.
- g1 and g2 uncertainties: ±3.8 and ±1.0 keV.

## 7. Failed or abandoned approaches

- Using nestpy `GetS1`/`GetS2` directly: they need 17/16 positional arguments (positions, drift time, calculation modes, PMT vectors) and would use the detector object's g1 = 0.1122 and g2 = 34.02; replaced by the explicit binomial/Fano detector model with the paper's g1, g2.
- Digitising the contour crossings from Fig. 2: AmBe calibration points create spurious crossings; Fig. 4 used instead (identical band and contours).
- First contour-colour mask (dark slate grey of the labels) found no contour pixels; the rendered contour is (150,166,166).
- Reading the paper's band width from the MC alone: NEST 2.1.1 fluctuations give a band 1.5× wider than the drawn 10–90 % lines; both are reported rather than one being forced onto the other.

## 8. Discussion

1. The paper's 248 keV is self-consistent: its constant-energy contours, its 269.9 keV efficiency edge and its 1.5σ statement all correspond to an NR light/quanta scale with S1c ≈ 2.2 phd/keV at 250 keV (Nq ≈ 11.3 E^1.11). The public Table S5 evaluated in nestpy 2.1.1 gives 2.0 phd/keV and places the same event at 262–266 keV. The difference is not in g1 or the p(E) break (which changes Nph by < 2 %) but in the total quanta Nq(E): +7.7 % at 250 keV. Most of it (5.7 %) is recovered if β = 1.11 rather than the printed 1.1 — the value nestpy itself stores for LZ — but we cannot exclude that the remaining 2 % (or all of it) comes from a NEST v2.4.5 feature not exposed in nestpy 2.1.1 (e.g. the "f1, f2" turn-on handling or a Lindhard-type factor). We flag this as a request to the collaboration: the Data Release NEST parameters should reproduce the figure contours.
2. Anyone in this corpus who computes NR-band positions, efficiencies or keVee equivalents from Table S5 with public nestpy inherits an 8 % S1 (7.7 % Nq) scale offset: the ROI upper edge would correspond to ≈ 292 keV instead of 270 keV, and expected signal counts near the edge would be overestimated for spectra rising with energy.
3. The event is not at the efficiency edge on the paper's scale (ε = 0.93, P(fail S1c cut) < 1 %); on the public-parameter scale it would be (ε = 0.71, 0.39 at the upper 68 % edge). The paper's ±23 keV statistical error is twice what the NEST fluctuation model and the roll-off width imply.
4. The NR-band width matters for the background discussion: with the drawn percentiles the event is a 2.5σ outlier of the NR band, with the NEST-2.1.1 widths 1.5–1.8σ. The MSSI-vs-NR overlap arguments should use whichever width LZ actually used in the inference.

## 9. Figures

- `figures/P009_fig_digitisation_check.png` — Overlay of the digitised band samples (cyan), contour crossings (magenta) and event (green) on the PNGs; validation of §4.
- `figures/P009_band_and_event.png` (Fig. 1) — Mean NR curves for the five variants in (S1c, log10 S2c), MC clouds at E_ML for LZ-tab (red) and paper-contour (black), the digitised contour crossings (squares, labelled in keV) and band percentiles at 540 phd, the event (star) and the S1c = 600 phd boundary.
- `figures/P009_likelihood_vs_E.png` (Fig. 2) — ln L(E) − max for the five variants with 68 % intervals, KDE cross-check, the paper's 248 ± 23 band, the 269.9 keV line and the efficiency model.
- `figures/P009_efficiency.png` (Fig. 3) — Fig. S2 inset readings, Gaussian roll-off fit (σ_E = 11.8 keV) and MC 0.955 P(S1c < 600 | E) for LZ-tab, LZ-β1.11 and paper-contour, with E_ML marked.

## 10. References

1. LZ Collaboration, arXiv:2609.02823 (2026) — the paper, main text and Supplemental Material (Tables S3–S5, Figs. 2, 4, S2).
2. M. Szydagis et al., "A review of NEST models for liquid xenon and exhaustive comparison to other approaches", Front. Detect. Technol. (2024), arXiv:2211.10726 — NEST v2 yield and fluctuation model.
3. B. Lenardo et al., "A global analysis of light and charge yields in liquid xenon", IEEE Trans. Nucl. Sci. 62, 3387 (2015) — NR yield power-law exponent preference at high energy (cited by LZ).
4. D. S. Akerib et al. (LUX), "Signal yields, energy resolution, and recombination fluctuations in liquid xenon", Phys. Rev. D 95, 012008 (2017) — combined-energy estimator and W = 13.7 eV.
5. J. Aalbers et al. (LZ), "Dark matter search results from 4.2 tonne-years of exposure of the LUX-ZEPLIN experiment", Phys. Rev. Lett. 135, 011802 (2025) — the WS2024 analysis whose selections and detector model this search inherits.
6. J. Aalbers et al. (LZ), "First dark matter search results from the LUX-ZEPLIN experiment", Phys. Rev. Lett. 131, 041002 (2023).
7. Corpus: `output/00_evidence_dossier.md` (checks (i) and (ii) in §2).

## 11. Tools and provenance (mirrors `output/provenance/P009.json`)

- Agent tools: Read (PAPER_GUIDE.md; 00_evidence_dossier.md; lzcommon.py; fulltext.tex lines 84–123 and 590–690; Fig2, Fig4, FigS2 PNGs; own figures ×4), Bash (grep of fulltext.tex; nestpy API probes ×4; digitisation runs ×4; main script run; results extraction ×3), Write (2 scripts, details.md, P009.json, P009.md), Edit (digitiser ×6).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (stats.multivariate_normal, stats.gaussian_kde, stats.norm, optimize.curve_fit); pandas 3.0.5; matplotlib 3.11.2 (Agg; image.imread for digitisation); nestpy 2.1.1 (detectors.LZ_WS2024, NESTcalc.GetYields, GetQuanta, WorkFunction, CalculateG2); common/lzcommon.py (LZ, NEST_NR_LZ, NEST_NR_DEFAULT, NEST_ER_LZ, nest_nr_params_vector, nest_nr_yields, nest_er_yields, combined_energy_keV, delta_max_kev, m_chi_min_gev, v_earth_kms, vmax_kms).
- Scripts: `output/code/P009_digitise_bands.py`, `output/code/P009_energy_reconstruction.py`.
- Local inputs: fulltext.tex; Fig2_Calibrations.png; Fig4_science_sample_wbands.png; FigS2_efficiency_werror.png; 00_evidence_dossier.md; work/dossier/dossier_numbers.json; code/common/lzcommon.py; environment/ENVIRONMENT_versions.txt.
- Recalled knowledge (3): W = 13.7 eV (NEST v2 older default / LUX) and 13.5 eV (literature) — likely; NEST v2 NR yield functional form — likely (not load-bearing, nestpy evaluates it); 10–90 % ↔ ±1.2816σ for a Gaussian — certain.
- Datasets: none. Data requests: none (a public LZ Data Release with the NEST parameters would settle the β/scale question but is not needed for the headline).
- WimPyDD files: none.
