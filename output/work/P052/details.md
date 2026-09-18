# P052 — Reproducing LZ's 3.4σ local significance from the published summaries: a three-sample two-dimensional toy likelihood

Research record (simulated date 2026-09-11). Script: `output/code/P052_reproduce_significance.py` (stages `fit`, `sa`, `toys`, `final`). Results: `output/work/P052/P052_results.json` (merged), `P052_results_fit.json`, `P052_results_sa.json`, `P052_data_fits_all_configs.csv`, `P052_variants_Z_{base,lzlike}.csv`, `P052_fig5_L10_digitised.csv`, `P052_single_event_cells_*.csv`, `toys/*.npz`, logs `run_log_*.txt`. Figures in `figures/`.

## 1. Motivation and question

LZ (arXiv:2609.02823) reports local significances for 616 signal models (Tables S6/S7), obtained from an unbinned extended likelihood in {S1c, log10 S2c} over three veto-defined samples with ≥30,000 background-only toys per model (Supplement, "Local significance"). The corpus has so far reproduced the tables only with one-dimensional proxies anchored to LZ's own 3.4σ: P016 (few-bin likelihood, b_H = 5.7×10⁻⁴ chosen so that L10 gives 3.4σ; rms 0.21σ), P021 (energy likelihood, same anchor; rms 0.40σ), P008 (1D toys; 0.3–0.5σ low). We ask whether the tables can be reproduced **without anchoring**, using only published summaries: Table I / S1 / S2 counts and constraints, the three Fig. 5 panels (background and best-fit-L10 histograms in the band-distance variable d), the tagging efficiencies, the event's coordinates, and the toy-MC definition of p0. We then switch ingredients on and off to find what drives the residuals: 2D versus 1D PDFs, the signal PDF shape, the nuisance constraints, the veto samples, or the toy calibration in the one-event regime.

## 2. Framework

### 2.1 Observable space and cells
d = (log10 S2c − μ_NR(S1c))/σ_NR(S1c), as in Fig. 5. Cells: S1c bins {[3,250)} ∪ 25-phd bins over [250,600] (1 + 14 bins; the Fig. 5 top panel is published only panel-wide, so S1c < 250 phd is a single bin), d bins of 0.5 from −8 to +8 (32 bins). 15 × 32 = 480 science-sample cells. A binned Poisson likelihood on these cells approximates LZ's unbinned likelihood: the event term is ln μ_cell where μ_cell is the cell integral of the smooth densities; with 25 phd × 0.5σ cells the density is nearly constant across a cell.

### 2.2 Likelihood (mirrors LZ Eq. S1)
−ln L(s, θ) = Σ_cells [μ_c − n_c ln μ_c] + Σ_{i∈{prompt, delayed}} [μ_i − N_i ln μ_i] + ½ Σ_k ((θ_k − θ̃_k)/σ_k)²,

μ_c = s S_c + t_ER (1 − f_det)[ER_bulk,c + t_tail ER_tail,c] + t_acc ACC_c + t_MSSI R_MSSI (1 − λ_MSSI) MSSI_c/4.9×10⁻³ + t_ν NU_c + N_n NEUT_c + (1 − λ_PG)(R_det t_det + R_Xe t_Xe) PLAT_c,

μ_prompt = 0.166 t_ER + λ_PG (R_det t_det + R_Xe t_Xe) + λ_MSSI R_MSSI t_MSSI + λ_PN N_n /(1 − λ_PN − λ_DN) + 10⁻⁴ s,
μ_delayed = 50.3 t_D + λ_DN N_n /(1 − λ_PN − λ_DN) + 3.1×10⁻² s.

Parameters (14): s (signal events in the science-sample ROI), t_ER (ER bulk scale), t_tail (ER leakage into d < 2.5, all panels), t_acc, t_MSSI (≥ 0), t_ν (atm-ν + ⁸B), N_n (detector neutrons in the science sample, unconstrained ≥ 0 — LZ: "constrained solely by the veto samples"), t_det, t_Xe (detector ERs, ¹²⁷Xe+¹²⁵Xe; rates R_det = 8.5/0.12 = 70.8, R_Xe = 1.5/0.12 = 12.5 before veto partition, so science = (1−λ_PG)R, prompt = λ_PG R, reproducing Table S1's 62.2 and 11.2), t_D (delayed-sample background scale), λ_PG, λ_MSSI, λ_PN, λ_DN. R_MSSI = 4.9×10⁻³/0.06 = 0.0817 (Table S1 prompt MSSI 7.7×10⁻² ✓).

Constraints (Gaussian, σ): t_ER 0.096 (Table I ER components in quadrature: √(160²+8.4²+16.5²+3.1²+6.3²+5.1²+2.7²)/1694), t_tail 0.30 (choice; Fig. 5 systematic band; tested 0.15 and 1.0), t_acc 0.6/2.7 = 0.22, t_MSSI 1.0, t_ν 0.18, t_det 0.40, t_Xe 0.20, t_D 0.095 (Table S2 in quadrature), λ_PG 0.88 ± 0.02, λ_MSSI 0.94 ± 0.02, λ_PN 0.05 ± 0.01, λ_DN 0.87 ± 0.02 (Tables S1/S2). Signal split science : prompt : delayed = 1 : 10⁻⁴ : 3.1×10⁻² (Tables I/S1/S2 fit results for L10).

Test statistic: q0 = 2[ln L(ŝ, θ̂) − ln L(0, θ̂₀)] with ŝ ≥ 0 (q0 = 0 when the score at s = 0 is ≤ 0). Asymptotic Z = √q0. Toy-calibrated p0 = P(q0 ≥ q0_obs | H0) from background-only toys, Z = Φ⁻¹(1 − p0) one-sided, as LZ ("Local significance", Baxter et al. 2021 Eq. 6). Toys: cell counts Poisson(μ_c(θ_nominal)), prompt/delayed totals Poisson, auxiliary centres θ̃ ~ N(θ_nominal, σ); tagging efficiencies fixed in toy fits (data fits show they matter at < 0.01σ). Minimiser: iminuit 2.32.0 (Migrad, strategy 0); the s-free fit is started from the H0 fit and, if it fails to improve, retried from s = 0.3.

### 2.3 Background model (nominal, events per cell)
Per-panel d-histograms from the corpus digitisations of Fig. 5 (post-fit model):
- **Top panel (S1c < 250)**: P016's digitisation (total 1686; accidentals 2.47; "Internal γ/IC/EC"; "Continuous ERs" with gaps filled by total − other components). ER (continuous + internal) = 1608.9, of which 195.7 at d < 2.5.
- **Middle panel (250–500)**: P022's total and accidentals (accidentals 2.94×10⁻³); MSSI and NR modelled (below); ER = total − accidentals − MSSI − NR = 25.1 events (all at d ≥ 1.5).
- **Bottom panel (> 500)**: P004's total (0.01053; caption 0.0106 ± 0.0008), MSSI (5.1×10⁻⁴) and accidentals (6.3×10⁻⁴); ER = total − others = 9.4×10⁻³ (d ≥ 1).
- **MSSI**: d-shape = P004 bottom-panel histogram (the only panel where it is resolved), applied in all panels; total 4.9×10⁻³ split 45 % / 45 % / 10 % (top/middle/bottom; the bottom share is P004's 5.1×10⁻⁴); flat in S1c within a panel (Fig. S1a: the MSSI contour runs along the whole band).
- **NR (atm-ν 0.11 + ⁸B 0.057)**: d ~ N(0,1); S1c ∝ exp(−S1c/58 phd), giving 2.1×10⁻⁵ above 500 phd (P019 read 3.4×10⁻⁵ from Fig. 5) and 1.2×10⁻³ in 250–500 (Fig. 5 green ≈ 10⁻³); ⁸B in the top bin only.
- **Detector neutrons**: d ~ N(0,1); S1c ∝ exp(−S1c/35 phd) (Fig. S1b green contour within 150 phd).
- **Accidentals**: within 250–600 phd, exponential in S1c with λ_acc = 280 phd fixed by the middle/bottom panel integrals (per-phd densities 1.18×10⁻⁵ vs 6.3×10⁻⁶).
- **ER in the middle panel**: S1c ∝ exp(−(S1c − 250)/30 phd) (the ER band leaves the ROI at 250–320 phd, Fig. S1a); flat within the bottom panel.
- **Detector ERs and ¹²⁷Xe** in the science sample follow the ER-bulk (d ≥ 2.5) shape; f_det = 10/1704 of the ER bulk is re-assigned to them so that the veto coupling is right.
Totals: visible model 1636.6 + detector ER/Xe share; the ER normalisation nuisance absorbs the difference to the 1710 observed (pull +0.37σ).

### 2.4 Data representation ("Asimov-like", stated approximation)
We do not have the event list. Top panel: P016's digitised data counts per d-bin (1575; 217 at d < 2.5, including 2 events at [−0.5, 0), 9, 10, 21, 79 in the next bins); the plateau bins (d ≥ 2.5, pure ER bulk, zero signal) are rescaled by 1.082 so that the science total is 1710. Middle panel: 23 events read from Fig. 5 (1, 1, 3, 4, 10, 4 in the bins from 4.5 to 7.5σ), placed at 250–275 phd. Bottom panel: the single event at S1c = 540.1 phd, d = −1.54 (paper: 1.5σ below the median; P024: 1.54; our own band model gives −1.51), i.e. cell [525, 550) × [−2, −1.5) — the bin in which Fig. 5 plots the point. Prompt 66, delayed 55 (Tables S1/S2).

### 2.5 Signal model
Spectra dR/dE (events/(t yr keV), any normalisation): L10 100–4000 GeV from P012 (WimPyDD, d10 = 1); elastic O1, O4, O6, O10, O11 (s, v) at 200/1000/4000 GeV from P003; inelastic O1 (s, v) at 400/1000/4000 GeV, δ = 100–350 keV, from P021 ('annual' halo). Operator ↔ Lagrangian mapping for the LZ comparison: O1 ↔ L1, O4 ↔ L15, O6 ↔ L4, O10 ↔ L2, O11 ↔ L3 (P003/P016; recalled, likely). Diagnostics (not in the rms): LZ's own Fig. 1 spectra (P003's vector digitisation: L10 200/1000 GeV; O1s δ = 0/200/300 keV) and Helm-form-factor inelastic O1s (lz.dRdE_SI) at δ = 200–350 keV.

Response: E → Nph, Ne with the Table S5 NEST-LZ yields including the p(E) break (lz.nest_nr_yields, nestpy 2.1.1) on a 1 keV grid; Nph rescaled by r(E) = (11.32 E^1.112 − Ne)/Nph_NEST clipped to [1, 1.12] (LZ's own contour energy scale, P009; r = 1.084 at 250 keV, 1.066 at 20 keV), so that S1c(248 keV) = 539.5 phd and E(540 phd) = 248.2 keV (paper: 248 keV); S1c(270 keV) = 595 phd, so the S1c < 600 phd cut produces the 269.9 keV efficiency edge (P038). μ_NR(S1c) = log10(g2 Ne) interpolated along the mean curve (4.017 at 540 phd; paper-implied 4.016); σ_NR = 0.033 √(540/S1c) dex (only used for the ROI cut d_max = (4.15 − μ_NR)/σ_NR, which is 4σ at 540 phd and irrelevant elsewhere). Low-energy efficiency 0.96 × ½[1 + erf((E − 5.4)/(√2·3.4))].

Two switchable ingredients:
- **S1c resolution** `res`: 'nest' σ_S1c = 1.13 √S1c (P038 NEST MC: sd/√S1c = 1.11–1.14 at 420–580 phd; 26 phd = 4.9 % at 540 phd; 10–11 keV, P009) or 'paper' σ_S1c = 0.093 S1c (the quoted 248 ± 23 keV stat; 50 phd at 540).
- **Signal d-PDF** `dshape`: 'gauss' N(0,1) (the band model by definition) or 'fig5': the digitised best-fit L10 curve of Fig. 5 in each panel (Section 3.1), normalised, applied to every signal model (the d-distribution at fixed S1c is a detector-response property).
Configurations: base = (gauss, nest); fig5only; resonly; lzlike = (fig5, paper).

## 3. Inputs digitised here

### 3.1 The L10 curve of Fig. 5 (colour (165,42,42), P022 tick calibration)
Panel sums 0.231 / 0.657 / 0.235 = 1.123 events (LZ best fit 1.0 (+1.4 −0.7); the 12 % excess is the thick-line reading bias, common to all bins). Split 0.206 / 0.585 / 0.209. Shape: mean d = +0.11 / −0.28 / −0.13, sd 1.05 / 1.17 / 1.10 for top/middle/bottom; fraction in the event's bin [−2, −1.5): 0.041 / 0.074 / 0.070 versus 0.044 for N(0,1). The middle and bottom panels show a heavier low-d tail (×1.7 at −1.75, ×2.6–3.7 at −2.75 in the middle panel), consistent with the S1–S2 anticorrelation at fixed S1c in NEST (a high Nph fluctuation selects a low Ne). File `P052_fig5_L10_digitised.csv`; Figure 4.

### 3.2 Everything else
Table I (expected/fit counts), Tables S1/S2 (veto samples, λ's), Table S6/S7 (lz.LSIG/OSIG), Fig. 5 caption (0.0106 ± 0.0008), Fig. S1 (MSSI/NR contour extents), digitisations from P004, P016, P022, spectra from P003/P012/P021, energy scale P009, band width P024, MC efficiency P038.

## 4. Results

### 4.1 Reproduction of Tables S6/S7 (59 models, asymptotic Z; `P052_data_fits_all_configs.csv`)
| configuration | rms (σ) | mean (σ) | L10 1000 | O1s δ350 | O1s δ300 | O4s | O1v | O10s | O11s | isoscalar inel. mean | isovector inel. mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base (gauss, nest) | 0.68 | −0.48 | 2.96 | 2.78 | 2.27 | 2.13 | 1.03 | 2.64 | 0.23 | −0.95 | −0.10 |
| fig5only | 0.52 | −0.28 | 3.12 | 2.94 | 2.47 | 2.36 | 1.39 | 2.83 | 0.71 | −0.73 | +0.07 |
| resonly | 0.59 | −0.44 | 2.95 | 2.88 | 2.43 | 2.12 | 0.95 | 2.64 | 0.56 | −0.78 | −0.13 |
| lzlike (fig5, paper) | 0.42 | −0.22 | 3.11 | 3.04 | 2.61 | 2.35 | 1.32 | 2.82 | 0.98 | −0.53 | +0.05 |
| LZ | | | 3.4 | 3.3 | 3.0 | 2.7 | 1.3 | 3.0 | 1.7 | | |

By class (lzlike): L10 mass scan rms 0.30 (mean −0.28); elastic operators 0.43 (−0.18); inelastic O1 0.43 (−0.24). Excluding LZ's "0.0" entries: rms 0.43, mean −0.24. The pattern of the tables is reproduced: ≈ 0 for isoscalar O1 (all δ ≤ 100) and low masses, 1.3 for isovector O1 (1.32), the O4/O10/O6 plateau at 2.4–3.0, the rise of inelastic O1 with δ, and the isovector > isoscalar ordering (O1v δ = 300–350: 3.37–3.49 vs LZ 3.4).

Diagnostics with LZ's own spectra (lzlike): L10 1000 GeV from Fig. 1: Z = 3.17, panel split 0.20/0.59/0.21 — identical to the Fig. 5 split (0.21/0.59/0.21), validating the E → S1c mapping; the WimPyDD L10 (P012) gives 0.25/0.59/0.16 (30 % more low-energy weight, cf. P003's N_lo 0.20 vs 0.155). O1s δ = 300 from Fig. 1: 2.62 (WimPyDD 2.61) — the isoscalar-inelastic deficit is not spectral. Helm form factor: O1s δ = 300: 2.74, δ = 350: 3.18 (node moved from 267 to 279 keV; +0.13σ).

### 4.2 L10 (1000 GeV) best fit (lzlike; base in brackets)
q0 = 9.66 (8.76); ŝ = 1.18 (1.16) events, Minos 68 %: +1.60 −0.83 → [0.35, 2.78] (paper 1.0 +1.4 −0.7); asymptotic 90 % interval [0.12, 4.30]. Pulls: t_ER +0.37, t_acc −0.36, t_det −0.27, t_tail +0.23, t_D +0.18, all others < 0.03σ; maximum |pull| 0.37σ (paper: all < 2σ). Fitted totals: science 1707.7 (paper 1713 ± 39; observed 1710), prompt 66.8 (66), delayed 53.0 (paper fit 50.4; observed 55); detector neutrons N̂_n = 0.18 science events (paper interval [0, 0.118]); MSSI 4.9×10⁻³, accidentals 2.28, atm-ν+⁸B 0.167. Background at the event's cell under H0: 1.16×10⁻⁵ (per 25 phd × 0.5σ; Fig. 5 bin 3.95×10⁻⁵ per 0.5σ over 100 phd); signal there 3.07×10⁻³ per unit s (base 1.98×10⁻³; Fig. 1 spectrum 3.8×10⁻³) → likelihood ratio per unit signal 265 (171; 328).

### 4.3 Toy calibration (`toys/*.npz`; 93,559 toys in total)
| model | config | q0_obs | Z_asym | toys | exceed | p0 | Z_toy [68 %] | P(q0 > 0) | LZ |
|---|---|---|---|---|---|---|---|---|---|
| L10 1000 | lzlike | 9.69 | 3.11 | 18,559 | 22 | 1.21×10⁻³ | 3.03 [2.98, 3.10] | 0.033 | 3.4 |
| L10 1000 (Fig. 1 spectrum) | lzlike | 10.06 | 3.17 | 10,000 | 6 | 6.5×10⁻⁴ | 3.22 [3.12, 3.35] | 0.034 | 3.4 |
| L10 1000 | base | 8.79 | 2.96 | 10,000 | 8 | 8.5×10⁻⁴ | 3.14 [3.05, 3.26] | 0.032 | 3.4 |
| O1s δ = 350 | lzlike | 9.26 | 3.04 | 10,000 | 14 | 1.45×10⁻³ | 2.98 [2.91, 3.07] | 0.018 | 3.3 |
| O1s δ = 350 | base | 7.74 | 2.78 | 10,000 | 13 | 1.35×10⁻³ | 3.00 [2.93, 3.09] | 0.015 | 3.3 |
| O4s elastic | lzlike | 5.55 | 2.36 | 10,000 | 54 | 5.5×10⁻³ | 2.55 [2.50, 2.60] | 0.29 | 2.7 |
| O4s elastic | base | 4.56 | 2.14 | 10,000 | 77 | 7.8×10⁻³ | 2.42 [2.38, 2.46] | 0.30 | 2.7 |
| O1v elastic | lzlike | 1.77 | 1.33 | 10,000 | 744 | 0.074 | 1.44 [1.43, 1.46] | 0.45 | 1.3 |
| O1s elastic | lzlike | 0 | 0 | 5,000 | 5,000 | 1 | 0 | 0.45 | 0.0 |

Toy minus asymptotic: −0.08 to +0.22σ. The toy p0 is nearly configuration-independent (L10: 1.2 vs 0.85×10⁻³; O1s δ350: 1.45 vs 1.35×10⁻³) because it is essentially the background expectation in the region where the likelihood ratio exceeds the observed one, which the d-shape and resolution move less than they move q0_obs itself. The one-event regime: P(q0 > 0) = 0.03 (L10) instead of Chernoff's 0.5; the toy survival function lies far below ½χ²₁ for q0 < 8 but is heavier than the mixture P(q0>0)·χ²₁ and crosses ½χ²₁ near q0 ≈ 9–10 (Figure 2), which is why Z_toy ≈ Z_asym to 0.1–0.2σ for the high-energy models here (P008 found 0.3–0.5σ shifts in a 1D setting; P016's few-bin toys −0.09/+0.12σ). For spectra with low-energy shoulders (O4s) P(q0 > 0) = 0.29 and toys add +0.2–0.3σ.

### 4.4 Where does p0 come from? (semi-analytic single-event calibration, `P052_single_event_cells_*.csv`)
p0_SA = Σ_c μ_c·1[q0(one event in cell c) ≥ q0_obs], the observed dataset minus the event plus one event in cell c. lzlike L10: p0_SA = 6.2×10⁻⁴ (toys 1.2×10⁻³: the SA ignores multi-event and auxiliary fluctuations); by panel: bottom (S1c > 500, d ∈ [−2, +1)) 2.44×10⁻⁴, middle (S1c 350–500, |d| ≲ 1.5) 3.76×10⁻⁴, top 0. Base L10: 8.85×10⁻⁴ = 2.44×10⁻⁴ + 6.41×10⁻⁴. O1s δ350 (lzlike): 1.04×10⁻³ = 2.0×10⁻⁴ + 8.4×10⁻⁴. **The bottom panel alone — the one directly published — gives Z = 3.49σ; the deficit to LZ's 3.4 comes entirely from the background we had to distribute inside the 250 phd-wide middle panel** (flat MSSI, exponential accidentals and NRs, ER at 250–320 phd). LZ's p0 = 3.4×10⁻⁴ requires the background at 350–500 phd within |d| < 1.5 to be ≈ 1×10⁻⁴ instead of our 3.8–6.4×10⁻⁴ (i.e. concentrated at the low-S1c end of the panel), or a correspondingly larger signal density at the event. The SA is not meaningful for O1v/O4s (P(q0 > 0) ≈ 0.3–0.45 from low-energy fluctuations).

### 4.5 Ingredient switches (data fits, asymptotic; `P052_variants_Z_*.csv`)
| variant | L10 1000 (base → lzlike) | O1s δ350 | O4s | O1v | comment |
|---|---|---|---|---|---|
| baseline | 2.96 → 3.11 | 2.78 → 3.04 | 2.13 → 2.35 | 1.03 → 1.32 | |
| 1D in S1c (cells with |d| < 3 merged) | 2.52 / 2.51 | 2.26 / 2.38 | 2.31 / 2.31 | 2.51 / 2.50 | −0.6σ for L10 (the ER leakage at d = 1–3 is added to the event's background); low-energy models *gain* (their signal is compared to the band-integrated background) |
| no veto samples | +0.00 / +0.00 | +0.00 | +0.01 | +0.01 | negligible: neutrons cannot populate S1c > 300 phd and the top-panel band is already under-populated |
| all nuisances fixed | +0.03 / +0.02 | +0.02 | +0.08 | +0.15 | profiling costs ≤ 0.03σ for high-energy models |
| ER-tail constraint 15 % / 100 % | 0.00 | 0.00 | 0.00 | 0.00 | |
| MSSI × 2 | −0.14 / −0.13 | −0.15 / −0.14 | −0.19 | −0.32 | |
| accidentals + MSSI × 2 (× 0.5) | −0.21 (+0.19) / −0.20 (+0.18) | −0.23 (+0.20) | −0.30 (+0.26) | −0.48 (+0.39) | ×2 on the background at the event costs 0.2σ |
| signal d-PDF: Fig. 5 shape | +0.16 | +0.16 | +0.23 | +0.36 | the largest single signal-model effect |
| S1c resolution 9.3 % | −0.01 | +0.10 | −0.01 | −0.08 | +0.16 for O1s δ = 300; helps node-suppressed isoscalar spectra only |
| band 0.0212 dex (event at −2.33, σ_d,sig = 0.68) | 1.55 (base) | 1.22 | 0.00 | 0.00 | LZ's drawn 10–90 % lines would destroy the significance (P024: the drawn lines are 1.5× too narrow; the 1.5σ statement is the model width) |
| event one d-bin lower / higher | −0.31 / +0.19 (base); −0.39 / +0.07 (lzlike) | | | | the bin resolution of the digitised histograms |
| signal d-mean −0.25 (gauss) | +0.12 | +0.14 | +0.11 | +0.15 | |

Ranking for L10 (1000 GeV): 2D vs 1D 0.6σ > toy calibration ±0.1–0.2σ ≈ background at the event (×2: 0.2σ) ≈ signal d-tail 0.16σ > nuisance profiling 0.03σ > veto samples 0.00σ; the energy resolution matters only for spectra suppressed at 248 keV (isoscalar O1: 0.1–0.2σ).

### 4.6 Comparison with P016 and P021
P016's few-bin likelihood needed b_H = 5.7×10⁻⁴ (200–270 keV) to give 3.4σ; our unanchored 2D background yields the equivalent of a smaller effective background at the event (LR 265 per unit signal) but a larger background in the "more-signal-like" region than LZ's toys imply. P021's 1D isoscalar deficit (0.2–0.9σ low at δ ≤ 250) reappears here (−0.95σ in base) and is halved (−0.53σ) by the heavier signal d-tail plus the 23 keV resolution; the Fig. 1 test shows it is not a WimPyDD-versus-LZ spectrum issue. The isovector inelastic entries are reproduced to +0.05σ.

## 5. Caveats and failed approaches
- The event list is replaced by histograms (Section 2.4); LZ's GoF tests and exact per-event terms cannot be reproduced. Top-panel data at d ≥ 2.5 were rescaled to the published total (signal-free bins).
- Within-panel S1c shapes of the middle-panel backgrounds are modelled, not published; Section 4.4 shows this is where the remaining −0.3σ sits → DR-002.
- The Fig. 5 d-shape is the best-fit L10 curve; its tails at |d| > 3 have reading noise (dashed line, few pixels). Applying it to all models assumes the d-PDF is spectrum-independent.
- The 'paper' resolution (9.3 %) is inconsistent with the Fig. S2 roll-off width (P009: 11.8 keV); we treat it as a bracketing variant, not a claim.
- Toy p0 values have ±25–40 % statistical error (6–22 exceedances); ≤ 2×10⁴ toys per configuration as instructed.
- Operator ↔ Lagrangian mapping (recalled). WimPyDD isovector shape systematic (P008/P021) may hide in the O1v agreement.
- Abandoned: an unbinned single-event term with a continuous S1c background density (needs the unpublished S1c dependence); a full 2D KDE of the Fig. S1 contours for MSSI (contours are 68/95 % of a PDF dominated by low S1c and cannot be inverted uniquely).

## 6. Figures
- `figures/P052_fig1_Z_scatter.png` — Z (this work) vs LZ Tables S6/S7 for 59 models, base (left) and lzlike (right); diamonds: toy-calibrated Z with 68 % binomial bands; grey band ±0.3σ.
- `figures/P052_fig2_q0_distribution.png` — survival function of q0 under H0 (18,559 toys, L10 1000 GeV, lzlike) vs ½χ²₁ and P(q0>0)·χ²₁; the observed q0 = 9.7.
- `figures/P052_fig3_model_map.png` — nominal background and best-fit L10 signal per cell at S1c > 250 phd, event marked.
- `figures/P052_fig4_signal_dshape.png` — digitised Fig. 5 L10 d-shape (middle and bottom panels) vs N(0,1).

## 7. References
LZ Collaboration, arXiv:2609.02823 (2026). D. Baxter et al., Eur. Phys. J. C 81, 907 (2021). G. Cowan, K. Cranmer, E. Gross, O. Vitells, Eur. Phys. J. C 71, 1554 (2011). H. Chernoff, Ann. Math. Stat. 25, 573 (1954). H. Dembinski et al., iminuit, Zenodo (2020). M. Szydagis et al., NEST, arXiv:2211.10726. I. Jeong, S. Kang, S. Scopel, G. Tomar, Comput. Phys. Commun. 276, 108342 (2022). Corpus: P001, P003, P004, P008, P009, P012, P016, P019, P021, P022, P024, P038.

## 8. Tools and provenance (mirrors provenance/P052.json)
Agent tools: Read ×28 (PAPER_GUIDE; dossier; ledger (partial); P016, P008, P021, P024, P004, P022, P009, P001 papers; Fig. S1a/b/c and Fig. 5 PNGs; fulltext.tex l.140–264, 436–595, 815–911; lzcommon.py l.1–112, 255–386; P016_results.json; P004/P022 digitised CSVs; DR-001.md; P022_accidentals.py l.60–110; three output figures), Bash ×27 (ledger summary; file listings; spectra/yield/efficiency inspection; palette lookup; smoke run; pixel probe and L10 digitisation; stage runs: fit ×2, toys ×10 in parallel batches, sa, final; SA breakdown; version checks; word counts), Write ×6 (script ×2, details, provenance, paper, DR-002), Edit ×3, Skill ×1 (dataviz). Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (stats.norm/poisson/chi2/beta, special.erf, optimize.brentq); pandas 3.0.5; matplotlib 3.11.2; iminuit 2.32.0 (Minuit, migrad, minos); Pillow 12.3.0 (PNG pixel digitisation); nestpy 2.1.1 via common/lzcommon.py (nest_nr_yields, dRdE_SI, LZ, LSIG, OSIG, OSIG_DELTAS); WimPyDD 2.0.4 only through cached spectra of P003/P012/P021 (not re-run). Recalled: operator ↔ Lagrangian mapping (likely); Chernoff ½χ² asymptotics and Wilks (certain); Baxter et al. toy p0 prescription (certain); NEST S1–S2 anticorrelation at fixed S1c as the origin of the heavy low-d tail (likely). Datasets: none. Data request: DR-002 (filed). WimPyDD-generated files: none.
