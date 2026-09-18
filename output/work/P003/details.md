# P003 — research record

**Title.** Which NREFT operators can produce a lone high-energy recoil? Spectral shape ratios in xenon and a reproduction of the LZ signal spectra
**Simulated arXiv date.** 2026-09-03 · hep-ph · Category EFT · author profile: effective-field-theory phenomenologists with direct-detection expertise.
**Scripts.** `output/code/P003_fig1_digitise.py` (Fig. 1 digitisation and normalisation test), `output/code/P003_nreft_shapes.py` (operator scan, classification, L10 reconstruction, figures). Both run from the simulation root with `.venv/bin/python`. Log of the scan: `output/work/P003/P003_nreft_shapes.log`.

---

## 1. Motivation and framework

LZ (arXiv:2609.02823) reports one nuclear-recoil-like event at E_R = 248 ± 23 ± 23 keV in 2.84 t·yr, with no accompanying excess at low energy: the same 220 live days were searched in 2024 with an ROI of S1c = 3–80 phd (5.4–55 keV NR at 50 % efficiency, LZ paper "Data Analysis" paragraph) and gave no signal. Table S6/S7 of the LZ supplement show a striking pattern: the pure spin-independent models (L1^s, L5^s, O1^s with δ = 0) have exactly 0.0σ local significance, while momentum- or velocity-suppressed Lagrangians (L2, L4, L6^v, L9–L12, L16, L18–L20) reach 3.0–3.4σ. The reason is spectral: a lone recoil at ~250 keV is only acceptable if the same interaction does not predict a large population below 55 keV.

We quantify this with the Fitzpatrick et al. (2013) / Anand et al. (2014) NREFT for a spin-1/2 WIMP. For each operator O_i (i = 1, 3, …, 15; O_2 is dropped as usual because it is not generated at leading order by relativistic Lagrangians) we compute with WimPyDD 2.0.4 (the code LZ used) the xenon recoil spectrum, and define

- R_lo = ∫_{5.4}^{55} ε(E) dR/dE dE   (2024 low-energy ROI),
- R_hi = ∫_{200}^{270} ε(E) dR/dE dE  (the window containing the event),
- N_lo ≡ R_lo / R_hi = number of low-energy events that must accompany one event in 200–270 keV.

Because both integrals scale with the same coupling², N_lo is coupling-independent (only the isoscalar/isovector ratio and the mass matter). A lone event is compatible with the 2024 null result if N_lo ≲ N_max, where N_max is the number of signal events the 2024 search would have tolerated in its ROI at high mass. We do not have the 2024 likelihood; from memory the 2024 90 % CL limit at m ≳ 200 GeV corresponds to roughly 3–5 signal events in the ROI (recalled, **uncertain**), so we present results for N_max = 3, 5 and 10 and show how the classification moves.

A second, corpus-wide purpose is to fix the coupling normalisation of WimPyDD relative to the LZ paper. LZ's Fig. 1 states that its curves use "unit coupling" c_i^s = 1/m_v² in the convention of Anand et al. Eq. 21 (m_v = 246.2 GeV), where c^0 = (c_p + c_n)/2. WimPyDD's `eft_hamiltonian` docstring instead defines its isospin coefficients as [c0 = c_p + c_n, c1 = c_p − c_n]. If both statements are right, WimPyDD needs c^0 = 2/m_v² to reproduce Fig. 1, i.e. a factor 4 in rate. We test this by digitising the figure.

## 2. Inputs

| Input | Source |
|---|---|
| Fig. 1 vector paths (`Fig1_combined_recoils.pdf`) | `inputs/arXiv_2609.02823_source/`, read with PyMuPDF 1.28.2 (`page.get_drawings()`, `page.get_text("dict")`) |
| Fig. 1 caption: unit coupling c_i^s = 1/m_v², d_i^s = 1/m_v², m_v = 246.2 GeV; panels: O1^s δ = 0/200/300 keV at 1000 GeV; L10^s at 50/200/1000 GeV; grey < 5.4 and > 269.9 keV | `inputs/LZ_arXiv_2609.02823_fulltext.tex` lines 62–74 |
| 2024 ROI 5.4–55 keV; 96 % average NR efficiency 14–250 keV; 50 % at 5.4 and 269.9 keV | same file, lines 112–117 and Fig. S2 caption (line 455–457) |
| Halo: SHM with Baxter-2021 parameters (v0 = 238, v_esc = 544, v_sun,pec = (11.1, 12.2, 7.3) km/s, ρ0 = 0.3 GeV/cm³) | `lzcommon.wd_halo()` → `WimPyDD.streamed_halo_function`; LZ says it uses these (paper line 59). The default WimPyDD call omits the Earth's orbital velocity (v_max = 794.6 km/s), which we call the "average halo" |
| Degenerate Lagrangian pairs L1≡L5, L2≡L8, L3≡L17, L4≡L20, L11≡L14 (scalar multiples); s/v-indistinguishable spectra for L2, L4, L7, L8, L10, L11, L14, L15, L19, L20, O4 | same file, lines 494–497 (supplement, LEE section) |
| Table S6 / S7 local significances | `lzcommon.LSIG`, `lzcommon.OSIG` (transcribed from lines 825–909) |
| Nuclear response functions | WimPyDD 2.0.4 bundled Xe target (shell-model one-body density matrices, 9 isotopes) |
| Nucleon mass 0.938272 GeV, m_v = 246.2 GeV | `lzcommon.M_NUCLEON_GEV`, `lzcommon.M_V_GEV` |

Recalled knowledge (flagged): (R1) NREFT operator definitions and their leading nuclear responses (Fitzpatrick et al. 2013, Table 1; Anand et al. 2014) — **certain** for the response assignments used in Sec. 6; (R2) WimPyDD isospin convention c0 = c_p + c_n — read directly from the installed docstring, so not recalled; (R3) Anand et al. convention c^0 = (c_p + c_n)/2 — **likely**; (R4) the 2024 LZ WS result excluded of order 3–5 signal events in its ROI at high mass — **uncertain**; (R5) Lagrangian → operator leading-order reductions L1, L5 → O1; L2 → O10; L3 → O11; L4 → O6; L15 → O4 (and L19 → O4 at leading order) — **likely**; (R6) the tensor–tensor magnetic Lagrangian L10 = χ̄ iσ^{μν} q_ν χ N̄ iσ_{μα} q^α N /m_M² reduces at leading order to 4 (m_N²/m_M²) [(q²/m_N²) O4 − O6] — derived by hand below, **likely**; (R7) the Higgs vev 246.2 GeV — certain (also in the paper).

## 3. Fig. 1 digitisation and the normalisation convention

### 3.1 Method
`P003_fig1_digitise.py` opens the PDF, reads all text spans (tick labels) and all drawing paths. Axis calibration:
- x: the tick labels 0, 50, …, 350 (font size 19, y > 640 pt, excluding the y-axis mantissa "10") give E = (x − 67.907 pt)/1.50171 pt keV⁻¹. Check: the grey `axvspan` rectangles have edges at 0.000, 5.500, 270.00, 350.00 keV (the plotting script evidently used 5.5 and 270 rather than 5.4 and 269.9).
- y: the "10" mantissa spans plus their exponent spans give the decade spacing (31.547 pt/decade top, 116.845 pt/decade bottom). The label-bbox centres sit 1.68 pt below the ticks (font ascender/descender asymmetry) — a first fit using the label centres biased all top-panel rates upward by 0.053 dex (13 %). We therefore anchor the scale on the panel edges: the lowest label (10⁻¹ top, 10⁻³ bottom) sits exactly at the bottom edge of its axes (matplotlib `ylim`), which the label positions confirm to 0.1 pt. Resulting ranges: top 10⁻¹–10⁹ (exactly), bottom 10⁻³–0.500 (the upper limit comes out as 10^−0.30103, i.e. exactly 0.5, an independent confirmation of the calibration).
- Curves: stroked width-2 paths with matplotlib tab10 colours (blue/orange/green top; red/purple/brown bottom); the short legend handles of the same colours are rejected (< 20 points). Points outside the axes box are dropped. 408–1323 points per curve; saved on a 1 keV grid in `fig1_digitised_top.csv` and `fig1_digitised_bottom.csv`.

### 3.2 Result
WimPyDD O1, isoscalar, 1000 GeV, average halo, divided by the digitised O1^s δ = 0 curve (`fig1_normalisation.json`, `fig1_O1_ratio_curve.csv`):

| E (keV) | 10 | 20 | 50 | 80 | 150 | 200 | 250 |
|---|---|---|---|---|---|---|---|
| c⁰ = 1/m_v² | 0.253 | 0.253 | 0.257 | 0.278 | 0.226 | 0.227 | 0.221 |
| c⁰ = 2/m_v² | **1.012** | **1.011** | **1.028** | 1.11 | 0.903 | 0.909 | 0.884 |

Digitised Fig. 1 values: 2.76 × 10⁷, 1.55 × 10⁷, 2.11 × 10⁶, 1.15 × 10⁵, 3.73 × 10⁴, 1.15 × 10⁴, 426 /t/yr/keV at 10, 20, 50, 80, 150, 200, 250 keV.

**Conclusion: Fig. 1's "c_i^s = 1/m_v²" is reproduced by WimPyDD with c⁰ = 2/m_v² to 1–3 % at 5–60 keV.** The alternative (c⁰ = 1/m_v²) is low by a factor 3.9–4.5 and is excluded. Hence: WimPyDD c⁰ = c_p + c_n = 2 × (Anand c⁰); unit coupling in the LZ/Anand sense means c_p = c_n = 1/m_v², i.e. WimPyDD (c⁰, c¹) = (2/m_v², 0) for isoscalar and (0, 2/m_v²) for isovector. As a cross-check, c_p = 1/m_v² corresponds to σ_p = μ_p²/(π m_v⁴) = 2.96 × 10⁻³⁸ cm² at 1000 GeV, and the Helm-form-factor SI formula (`lzcommon.dRdE_SI`) gives 2.84 × 10⁷ /t/yr/keV at 10 keV, 3 % above the figure.

The inelastic curves confirm the same factor: with c⁰ = 1/m_v² WimPyDD/Fig. 1 = 0.228 ± 0.004 for δ = 200 keV (200–260 keV) and 0.225–0.234 for δ = 300 keV, i.e. c⁰ = 2/m_v² gives 0.91 there too.

Residual shape difference: above ~110 keV WimPyDD (c⁰ = 2/m_v²) is 9–12 % below the figure (0.903 at 150, 0.884 at 250 keV), while the M-response diffraction minima agree in position (Fig. 1: 100.4 keV and ≈ 265 keV; WimPyDD: 102 keV and ≈ 265 keV; ratio curve spikes to 2.06 at 100 keV and 1.16 at 270 keV because the minima are 1–2 keV apart and the figure's first minimum is deeper, 0.047 vs 0.082 relative to the rate 10 keV below). Adding the Earth's orbital velocity (2 June) raises the WimPyDD rate at 150–250 keV by only 1–4 %, so the residual is most plausibly the "modifications" LZ made to the DMFormFactor density matrices (paper line 59) or a slightly different high-q form factor; it does not affect any ratio in this paper at more than the 10 % level. Note that the second minimum of the M response at ≈ 265 keV sits inside the 200–270 keV window: for M-type operators (O1, O11, and the M parts of O5, O8) the rate near the event energy is suppressed by an additional order of magnitude relative to a smooth spectrum.

Figure: `figures/P003_fig1_overlay.png` (digitised curves as thick translucent lines; WimPyDD c⁰ = 1/m_v² and 2/m_v² as thin lines).

## 4. Efficiency model and window integrals

ε(E) = 0.96 · ½[1 + erf((E − 5.4)/(√2·3.4 keV))] · ½[1 − erf((E − 269.9)/(√2·8 keV))].
The plateau (0.96) and the 50 % points are from the paper; the widths are assumptions: σ_lo = 3.4 keV makes the efficiency reach its plateau by ~14 keV (where the paper says the 96 % average starts), σ_hi = 8 keV keeps ε > 0.95 at 250 keV. At the 50 % points the model gives 0.48 rather than 0.50; irrelevant here. Robustness variants computed for every entry: hard cuts (0.96 between 5.4 and 269.9 keV, else 0), σ_hi = 4 keV, σ_hi = 15 keV. The R_lo integral uses the same ε below 55 keV and a hard cut at 55 keV (the 2024 ROI edge); the 2024 efficiency curve differed in detail (different S2 threshold behaviour), which we cannot reproduce — this is a 10–20 % effect on R_lo for the flattest spectra and less for the others.

Spectra are computed on a 3 keV grid from 1 to 300 keV plus the reference energies (5.4, 10, 50, 55, 150, 200, 248, 269.9, 300 keV); WimPyDD costs 20–50 ms per energy per spectrum, and 84 spectra × 109 points took 315 s. Window integrals interpolate log(dR/dE) on a 2001-point grid; a 1 keV grid was used for the Fig. 1 comparisons (Sec. 3) and gives the same O1 N_lo to 0.1 %.

Pipeline cross-check with the digitised Fig. 1 curve itself: running the digitised O1^s (δ = 0, 1000 GeV) curve through the same window integrals gives N_lo = 2454, versus 2752 from WimPyDD (the 11 % difference is the high-energy shape residual of Sec. 3). The digitised L10^s 1000 GeV curve gives N_lo = 0.155 and R_hi = 1.91 /t/yr at unit coupling (5.4 events in 2.84 t·yr before the coupling is fitted), so the best-fit "1.0 event" of LZ corresponds to d_10 ≈ 0.43/m_v² — consistent with the O(0.1–0.5) values read from Fig. 6 in the dossier.

## 5. Operator scan: results

Coupling: c^τ = 1/m_v² in WimPyDD's convention (i.e. half the LZ unit coupling; irrelevant for ratios, and all absolute rates below can be multiplied by 4 to convert to LZ's unit coupling). Full table: `P003_operator_table.csv` (84 rows: dR/dE at 10, 50, 150, 248 keV; R_lo, R_hi, R_5–300, R_hi/R_lo, N_lo, frac_hi, and the three robustness variants of N_lo). Classification: `P003_classification.csv`. Spectra: `P003_spectra.npz`.

### 5.1 N_lo = R_lo/R_hi (low-energy events per event in 200–270 keV)

| O_i | response | q, v scaling | s 200 | s 1000 | s 4000 | v 200 | v 1000 | v 4000 | verdict (1000 GeV, N_max = 5) |
|---|---|---|---|---|---|---|---|---|---|
| O1 | M | q⁰ | 7895 | 2752 | 2384 | 1609 | 477 | 404 | excluded |
| O3 | Φ'' + v²Σ' | q⁴Φ'', q²v²Σ' | 37.3 | 12.6 | 10.9 | 14.2 | 5.04 | 4.38 | s excluded; v marginal |
| O4 | Σ' + Σ'' | q⁰ | 86.3 | 28.1 | 24.1 | 90.2 | 29.4 | 25.2 | excluded |
| O5 | Δ + v²M | q²v²M, q⁴Δ | 43.0 | 14.2 | 12.2 | 3.15 | 1.05 | 0.91 | s excluded; **v compatible** |
| O6 | Σ'' | q⁴ | 1.27 | 0.43 | 0.38 | 1.31 | 0.45 | 0.39 | **compatible** |
| O7 | Σ' | v² | 112 | 32.3 | 27.2 | 124 | 35.6 | 30.0 | excluded |
| O8 | M + Δ | v²M, q²Δ | 483 | 156 | 134 | 28.1 | 9.19 | 7.89 | s excluded; v marginal |
| O9 | Σ' | q² | 6.35 | 2.07 | 1.77 | 6.78 | 2.19 | 1.88 | **compatible** |
| O10 | Σ'' | q² | 9.14 | 3.08 | 2.66 | 9.35 | 3.15 | 2.72 | **compatible** |
| O11 | M | q² | 723 | 258 | 225 | 113 | 34.0 | 28.8 | excluded |
| O12 | Φ'' + v²Σ' | q²Φ'', v²Σ' | 300 | 100 | 86.3 | 118 | 41.1 | 35.7 | excluded |
| O13 | Σ'' | q²v² | 14.5 | 4.90 | 4.23 | 29.5 | 9.52 | 8.16 | s compatible (borderline); v marginal |
| O14 | Σ' | q²v² | 8.01 | 2.34 | 1.98 | 8.58 | 2.49 | 2.10 | **compatible** |
| O15 | Φ'' + v²Σ' | q⁶Φ'', q⁴v²Σ' | 5.56 | 1.91 | 1.65 | 2.09 | 0.75 | 0.66 | **compatible** |

R_hi/R_lo is the reciprocal (e.g. O1^s 1000 GeV: 3.6 × 10⁻⁴; O6^s: 2.30; O15^v: 1.33). Fraction of the whole 5.4–300 keV rate that lies in 200–270 keV at 1000 GeV: O1^s 0.035 %, O4 2.5 %, O9 17 %, O10 10 %, O6 23 %, O5^v 20 %, O15 13–17 %.

Robustness (1000 GeV): hard cuts change N_lo by ≤ 8 % (O1^s 2752 → 2972, O7 +9 %, O6 −3 %); σ_hi = 15 keV: +0.4 to +5 %; σ_hi = 4 keV: −0.1 to −3 %. The mass dependence is a factor ≈ 2.9 between 200 and 1000 GeV (kinematic hardening) and ≈ 0.87 between 1000 and 4000 GeV (spectrum saturates once m_χ ≫ m_Xe, as LZ notes for its LEE treatment). The classification at 1000 GeV therefore also holds at 4000 GeV; at 200 GeV only O5^v (3.2), O6 (1.3) and O15^v (2.1) stay below 5, with O9, O14 and O15^s at 5.6–8.6 and O10 at 9.1–9.4.

### 5.2 Dependence on N_max (1000 GeV; the recalled 2024 tolerance is 3–5 events, uncertain)
- N_max = 3: O5^v, O6^{s,v}, O9^{s,v}, O14^{s,v}, O15^{s,v} survive (O10 at 3.08–3.15 just fails).
- N_max = 5: add O10^{s,v}, O13^s (4.90), and O3^v (5.04) is on the line.
- N_max = 10: add O3^v, O8^v (9.2), O13^v (9.5).
- No plausible N_max (< 20) rescues O1, O4, O7, O11, O12, O3^s, O5^s, O8^s: they need 13–2750 low-energy companions.

### 5.3 Inelastic reference (O1^s, 1000 GeV, `P003_inelastic_reference.csv`)
N_lo = 2752 (δ = 0), 501 (δ = 100 keV), 7.6 (δ = 200 keV), 0 (δ = 300 keV; kinematic threshold above 55 keV). This tracks Table S7 (0.0σ, 0.8σ, 2.7σ, 3.0σ) and reproduces the dossier's statement that SI-like spectra give ~10³ low-energy events per high-energy one.

## 6. Physics of the ordering

The operators divide by (i) the power of q in the amplitude and (ii) the nuclear response (Fitzpatrick et al. 2013):
- **M (charge-like, coherent, ∝ A²):** O1 (q⁰) is the steepest spectrum (dR/dE ∝ η(v_min) F²), N_lo ~ 10³. O11 = i S_χ·q/m_N gives q² M — two extra powers of q reduce N_lo by a factor 10 but the coherent M form factor still falls by 10⁴ over the ROI, so 258 companions remain. The isovector O1^v/O11^v have smaller N_lo (477, 34) because c¹ couples to (Z − N)/A ≈ −0.18 and the neutron-skin differences make the isovector form factor relatively harder, but they are still excluded. In all M-type spectra the diffraction minimum at ≈ 265 keV sits in the event window.
- **Σ'/Σ'' (spin, ¹²⁹Xe and ¹³¹Xe only, no A² coherence):** O4 (q⁰) has a flatter spectrum than O1 (spin form factors fall more slowly) but still N_lo ≈ 28. Every extra q² buys a factor ~10: O9, O10 (q²) have N_lo ≈ 2–3; O6 (q⁴ Σ'') has 0.43 and is the most "top-heavy" operator — its spectrum rises to ~50 keV and stays flat to 250 keV (Fig. `P003_operator_spectra.png`). O13, O14 (q² v⊥²) are similar to O9/O10 in shape (v⊥² adds a mild low-energy suppression) but with absurdly small absolute rates (10⁻⁷–10⁻³ /t/yr at unit coupling).
- **v⊥² alone (O7, O8):** velocity suppression without momentum suppression does not remove the low-energy population: v⊥² = v² − q²/(4μ²) is largest for small q, so O7 (v² Σ') has N_lo ≈ 33, and O8^s (v² M) 156. O8^v is helped by its q² Δ (orbital angular momentum) part, N_lo = 9.
- **Φ'' (spin-orbit, coherent-ish in Xe because of the large-l orbitals):** O3 (q⁴ Φ'' + q² v² Σ') and O15 (q⁶ Φ'' + q⁴ v² Σ') are strongly momentum-suppressed and come with the Φ'' form-factor node near 100–140 keV, giving genuinely double-peaked spectra; O15 survives (N_lo 0.7–1.9), O3^v marginally (5.0), O3^s (12.6) and O12 (only q² Φ'', 41–100) do not.
- **Δ (O5, O8):** the anapole-like O5^v (q⁴ Δ + q² v² M) survives (N_lo = 1.05) while O5^s does not (14.2): the isoscalar M interference/admixture brings back a coherent low-energy component.

Rule of thumb from the table: **at m_χ ≥ 1000 GeV each factor q² in the amplitude divides N_lo by ≈ 10; a lone 250 keV event requires at least q⁴ in the rate (q² in the amplitude) on a spin response, or q⁴–q⁶ on a coherent one.**

## 7. Comparison with Table S6/S7 and the Lagrangian mapping

- L1^s and L5^s (both pure O1 at leading order; LZ confirms they differ only by a scalar) have 0.0σ at every mass — exactly what N_lo = 2750 implies: the fit cannot place 1 event at 248 keV without 2750 companions, so it sets the signal to zero.
- L1^v, L5^v (isovector O1) reach 1.2–1.3σ at 1000–4000 GeV, following the reduced N_lo (477 vs 2752) — a first hint that the LZ PLR responds to N_lo continuously.
- L2 → O10, L3 → O11, L4 → O6 (recalled, likely). Table S6: L2 3.0–3.1σ (our N_lo 3.1: compatible), L4 3.1σ (N_lo 0.43: compatible), L3^s 1.7–1.8σ and L3^v 2.6–2.7σ (N_lo 258 / 34: we would call these excluded; the PLR gives them intermediate significance). L15 → O4 (N_lo 28): 2.7σ; O4^s at δ = 0 in Table S7: 2.7σ.
- **Tension.** Our proxy criterion is stricter than the LZ PLR: operators that we say need 28–34 low-energy companions (O4, L15, L3^v, O11^v) still receive 2.6–2.7σ in the LZ fit, and only the N_lo ≳ 250 models are driven to 0σ. Either the LZ likelihood, fitted on the 1710-event science sample in {S1c, log S2c}, tolerates tens of low-energy NR-band signal events (i.e. the effective N_max of the combined fit is ≫ 5, because the low-energy NR band in the extended-ROI fit has a larger background budget than the dedicated 2024 analysis, or because these events sit where the NR and ER bands overlap at low S1c), or the significance is dominated by the single event with the low-energy population absorbed by background nuisance parameters. This is a well-posed question for the STAT papers: reproduce Table S7's 2.7σ for elastic O4 and read off the effective N_max. If the 2024 low-energy search really tolerated only ≲ 5 events, O4/L15/L3 should have been at ≲ 1σ.
- L10 (magnetic–magnetic tensor) and L16 (3.4σ) are the top of the table; Sec. 8 shows L10's spectrum is a pure transverse-spin Σ' response with q⁴ weighting, N_lo = 0.15–0.20.
- Not attempted (we are not confident of the full reductions including subleading terms): L6, L7, L8, L9, L11–L14, L16–L20. We note that L19 = χ̄σ^{μν}χ N̄σ_{μν}N reduces to 4 O4 at leading order but Table S6 shows L19 (2.2–3.1σ) ≠ L15 (1.1–2.7σ), so LZ's implementation keeps O(q²/m²) corrections that matter; mapping by leading order alone is therefore unreliable for the tensor Lagrangians.

## 8. Reconstruction of the double-peaked L10^s spectrum

Derivation (R6). For non-relativistic spinors, ū σ^{ij} q_j u → 2 ε^{ijk} q_j S^k = 2 (q × S)^i, while the σ^{0i} q_i pieces are O(q²/m, q·v) and subleading. Hence
L10 ∝ (q × S_χ)·(q × S_N) · 4/m_M² = 4 [q² S_χ·S_N − (q·S_χ)(q·S_N)]/m_M² = 4 (m_N²/m_M²) [(q²/m_N²) O4 − O6].
The squared amplitude of c[(q²/m_N²) O4 − O6] is c² (q⁴/m_N⁴) [Σ' + Σ'' − 2Σ'' + Σ''] = c² (q⁴/m_N⁴) Σ': the longitudinal spin response cancels exactly and a magnetic interaction couples only to the transverse spin — the q⁴ weighting kills the low-energy rate and the Σ' form-factor node of ¹²⁹Xe/¹³¹Xe (near 50–60 keV) produces the dip. WimPyDD implementation: `eft_hamiltonian({(4,'q2'): lambda q, A: [A q²/m_N², 0], 6: lambda A: [−A, 0]})` with A = 1/m_v² (q-dependent Wilson coefficient via WimPyDD's reserved `q` argument; no response-function files were written).

Results (`P003_L10_comparison.csv`, `figures/P003_L10_overlay.png`):

| m_χ | scale Fig. 1 / WimPyDD(A = 1/m_v²) | rms log₁₀ residual (10–260 keV) | max | dip Fig. 1 / WD (keV) | high peak Fig. 1 / WD (keV) | N_lo Fig. 1 / WD |
|---|---|---|---|---|---|---|
| 50 | 229 | 0.109 | 0.19 | – | – | (all rate below 110 keV) |
| 200 | 248 | 0.086 | 0.21 | 62–64 / 52 | 172 / 178 | 0.47 / 0.60 |
| 1000 | 247 | 0.086 | 0.21 | 61 / 52 | 204–205 / 196 | 0.155 / 0.198 |

The combination reproduces the double-peaked shape qualitatively (low peak ≈ 20–27 keV, dip, broad high peak) with a 20 % rms residual; WimPyDD's dip is ~10 keV lower and its high peak ~8 keV lower than the figure, i.e. the figure has slightly more weight at intermediate energies, as expected if LZ's L10 includes the O(q²) terms from σ^{0i} (which bring in O1, O5-like pieces). The alternative sign, (q²/m_N²) O4 + O6, and (q²/m_N²) O4 or O6 alone, are all single-humped (right panel of the overlay figure), so the dip is specifically the transverse-spin (Σ') signature. The absolute scale factor is 247 ≈ 2⁸ = 256 (within 4 %, i.e. within the shape residual): it decomposes as 4 (WimPyDD c = c_p + c_n convention, Sec. 3) × 16 (the prefactor 4 in the reduction, squared, if m_M = m_N) × ≈ 4 (unexplained; plausibly a factor 2 in the normalisation of the tensor current or m_M ≠ m_N in Anand's Eq. 68 — **flagged uncertain**). Shapes and N_lo are unaffected by this.

## 9. Failed or abandoned approaches
- First y-axis calibration from tick-label centres: biased by 0.053 dex; replaced by edge-anchored calibration (Sec. 3.1).
- First x-axis calibration accidentally included the y-axis "10" mantissa in the x-label set (gave 1.559 pt/keV); fixed by requiring x > 60 pt.
- Legend handles of the same colour overwrote the curves in a first pass; fixed by a minimum path length.
- A first scan on a 1 keV grid (300 points × 84 spectra) did not finish within the tool's 10-minute limit (WimPyDD is 20–50 ms per point in this environment, i.e. ~17 min); it was stopped and re-run on a 3 keV grid (315 s). The 1 keV grid was kept for the Fig. 1 comparisons.
- A Python string-quoting error in the operator-info table caused one aborted run.

## 10. Extended discussion
1. The classification is coupling-independent and nearly mass-independent above 1 TeV, so it is a clean test: any elastic NREFT explanation of the LZ event must be built from O6, O9, O10, O14, O15, the isovector O5, or (marginally) O13^s, O3^v, O8^v, with the q⁴-weighted spin operators (O6, and the L10-like pure-Σ' combination) preferred. The 3.4σ Lagrangians L10 and L16 are of exactly this kind.
2. The absolute rates matter for model building: at unit coupling (LZ convention, × 4 relative to our table) R_hi for O6 is 0.043 /t/yr, so 1 event in 2.84 t·yr requires c_6 ≈ 2.9/m_v² (≈ 5 × 10⁻⁵ GeV⁻²), whereas O14 would need c_14 ≈ 10⁴/m_v² — a non-perturbative coupling; O7, O13, O14 are formally compatible but physically irrelevant.
3. Our N_lo proxy appears stricter than the LZ PLR (Sec. 7). Reading the LZ PLR's effective tolerance for low-energy signal events off Table S7 (O4 elastic, 2.7σ vs. N_lo = 28) is the single most useful follow-up.
4. Normalisation for the corpus: to reproduce LZ "unit coupling" curves, call `lz.wd_hamiltonian(name, {i: (2/m_v², 0)})` (isoscalar) or `{i: (0, 2/m_v²)}` (isovector); equivalently, WimPyDD c_p = (c⁰ + c¹)/2. Rates scale as c², so the factor is 4. A residual −10 % at E > 110 keV relative to LZ's curves should be expected from the nuclear inputs.

## 11. Figures
- `figures/P003_fig1_overlay.png` — Fig. 1 digitised (thick) vs WimPyDD (thin): top O1^s 1000 GeV (δ = 0/200/300), c⁰ = 1/m_v² and 2/m_v²; bottom L10^s digitised.
- `figures/P003_operator_spectra.png` — 14 panels, dR/dE for s (solid) / v (dashed) at 200/1000/4000 GeV; bands mark 5.4–55 and 200–270 keV; N_lo at 1000 GeV printed per panel.
- `figures/P003_Nlo_bar.png` — N_lo per operator, isospin and mass, with the N_max = 3, 5, 10 lines.
- `figures/P003_L10_overlay.png` — left: (q²/m_N²)O4 − O6 scaled to the digitised L10^s curves at 50/200/1000 GeV; right: peak-normalised shapes of the four O4/O6 combinations vs Fig. 1 at 1000 GeV.
- `figures/P003_efficiency_model.png` — the efficiency model and its variants.

## 12. References
1. LZ Collaboration, arXiv:2609.02823 (2026) — the paper under study.
2. A. L. Fitzpatrick, W. Haxton, E. Katz, N. Lubbers, Y. Xu, JCAP 02 (2013) 004, arXiv:1203.3542.
3. N. Anand, A. L. Fitzpatrick, W. C. Haxton, Phys. Rev. C 89 (2014) 065501, arXiv:1308.6288.
4. S. Kang, S. Scopel, G. Tomar, J.-H. Yoon, "WimPyDD", Comput. Phys. Commun. (2022), arXiv:2106.06207 (as cited by LZ, ref. wimpydd:Jeong_2022).
5. D. Baxter et al., Eur. Phys. J. C 81 (2021) 907, arXiv:2105.00599.
6. LZ Collaboration, Phys. Rev. D 109 (2024) 092003, arXiv:2312.02030 (first extended-energy NREFT search).
7. LZ Collaboration, Phys. Rev. Lett. 133 (2024) 221801, arXiv:2404.17666 (covariant Lagrangian constraints).
8. LZ Collaboration, 2024 WIMP-search result (ref. LZ:SR3_WS2024 in the paper; arXiv:2410.17036 — number recalled, likely).
Corpus: `output/00_evidence_dossier.md` (P000).

## 13. Tools and provenance
Mirrors `output/provenance/P003.json`.
- Agent tools: Read (PAPER_GUIDE.md; 00_evidence_dossier.md; lzcommon.py; fulltext.tex lines 33–117, 486–505, 818–911; 5 PNG figures produced here), Bash (listing; grep of the tex; WimPyDD introspection; PyMuPDF inspection; timing; script runs; CSV inspection), Write (2 scripts, details.md, P003.json, P003.md), Edit (6 edits to the scripts), TaskStop (2, stopping the over-long 1 keV run).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (special.erf, integrate.trapezoid, signal.argrelextrema); pandas 3.0.5; matplotlib 3.11.2 (Agg); pymupdf 1.28.2 (open, get_text("dict"), get_drawings); WimPyDD 2.0.4 (eft_hamiltonian incl. q-dependent Wilson coefficients, streamed_halo_function, diff_rate via lzcommon.wd_rate); common/lzcommon.py (LZ, M_V_GEV, M_NUCLEON_GEV, GEV_TO_CM2, mu_red, dRdE_SI, wd_halo, wd_hamiltonian, wd_rate, wd).
- WimPyDD-generated files: none (checked with `find WimPyDD -newer …` after the q-dependent Hamiltonian test; `diff_rate` does not write response-function files).
- Datasets: none. Data requests: none.
