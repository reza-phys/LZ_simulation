# P059 — Archival xenon exposures reinterpreted for the extended window: research record

Simulated date 2026-09-11 · category XEXP · author profile "former LUX/XENON analysts" · script `output/code/P059_archival_xenon.py`

## 1. Motivation and framework

LZ (arXiv:2609.02823) fits its single 248 keV event with 1.0 (+1.4, −0.7) signal events in 2.84 t·yr inside the S1c 3–600 phd window
(NR efficiency 0.96 between 14 and 250 keV, 50 % at 5.4 and 269.9 keV). Before this result four xenon exposures had already been
analysed with high-energy or EFT-type signal models and had set limits: LZ's own first science run (SR1, arXiv:2312.02030, "LZ 2024a" in
Fig. 6 top and "LZ 2024" in Fig. S7; the covariant-Lagrangian version arXiv:2404.17666 is "LZ 2024b" in Fig. 6 bottom), LUX's 311.2-day EFT
search (arXiv:2102.06998, red in Fig. 6 top), PandaX-II's SD-EFT search (arXiv:1807.01936, blue in Fig. 6 bottom) and XENON1T's 1 t·yr
SI search (arXiv:1805.12562, recast to inelastic DM by PICO and shown in Fig. S7, green). For a xenon target and a common halo model the
differential rate per tonne-year is detector independent (P005, P035), so LZ's best fit predicts a definite number of events in each of
these exposures once their live exposure and their acceptance in true recoil energy are specified. We ask:

(a) how many events each exposure should have contained, and P(0);
(b) whether the digitised previous limits in Fig. 6 are arithmetically consistent with those counts (a 90 % CL limit from zero events must
    correspond to ≈ 2–3 expected events in that exposure, and the new best fit must sit below the old limit by roughly the exposure ratio);
(c) what a joint Poisson likelihood LZ(1 event) + archival(0 events) does to the best-fit rate, its 90 % interval and the local significance;
(d) what LZ's SR1 (same detector, S1c 3–600 phd, 0.90 t·yr) should have contained and what a joint SR1 + SR3 fit would give.

## 2. Inputs

### 2.1 From the LZ paper
- Exposure 2.84 t·yr (220 live days × 4.71 t); best-fit L10ˢ 1000 GeV signal 1.0 (+1.4, −0.7) events (Table I); efficiency plateau 0.96, 50 %
  points 5.4 and 269.9 keV (Data Analysis, Fig. S2). Library values `lz.LZ`.
- "The S1c range is the same as used in the previous NREFT search with LZ [LZ:SR1_NREFT_2023]" (fulltext l.116) → SR1 ROI S1c 3–600 phd (certain).
- Fig. 6 caption (l.291): previous limits violet LZ (SR1 NREFT 2023 + covariant Lagrangians 2024), red LUX 2021, blue PandaX-II 2019.
  Fig. S7 caption (l.815): CRESST-II, PICO, XENON1T (PICO recast, starred), PandaX-4T, LZ 2024.
- Bibliography entries l.362, 371, 388, 389 (titles: "311.2 days of LUX data"; "extended energy region from LUX-ZEPLIN"; PandaX-II SD EFT).
- Fig. 6 vector PDF `inputs/arXiv_2609.02823_source/Fig6_O1_L10_limit_stacked.pdf` (digitised here: drawings 235 = LZ 2024a violet, 236 = LUX red,
  347 = LZ 2024b violet, 348 = PandaX-II blue; calibrations from P007 (top) and P012 (bottom)).

### 2.2 From the corpus
- P005 `output/work/P005/spectra_normalised_LZ1.csv`: WimPyDD spectra (1000 GeV, Sun-frame Baxter SHM) for L10, O1ˢ inelastic δ = 250/300/350 keV,
  normalised so that 2.84 t·yr × ∫ dR/dE ε_LZ dE = 1.0 event (re-verified here: 0.9996–1.0003).
- P012 `P012_L10_spectra_d10_1.npz` (L10 spectra at d10 = 1 for 100–4000 GeV, 1–339 keV) and `P012_coupling_table.csv`
  (N(d10 = 1) per 2.84 t·yr: 17.61/28.66/24.47/12.87/3.63 in the LZ Fig.-1 normalisation, 4.57/7.43/6.34/3.34/0.94 in the WimPyDD/Anand one),
  `fig6_bottom_digitised.csv` (new L10 interval edges and the x-axis calibration).
- P007 `lz_intervals_digitised.json` (new Fig. 6 top upper/lower/median edges). P035 `Nunit_table.csv` (unit-coupling counts N_unit(δ) at
  1000 GeV, Sun frame, LZ-like efficiency; used for δ = 300, 350 and as a validation at 100–250 keV). P038 acceptance A_600(δ) (context).
- P021/P035: Sun-frame halo reproduces LZ's Fig. 6 edges to 6 %; annual/Sun differences ×1.08/1.7 at δ = 300/350 keV.

### 2.3 Recalled (flagged)
| item | value used | reliability |
|---|---|---|
| LZ SR1 exposure: 60 live days × 5.5 t FV = 0.9035 t·yr | 0.9035 t·yr | likely |
| SR1 g1 = 0.114 phd/photon (SR3: 0.110) → S1c = 600 phd ≈ 269.9 × 0.110/0.114 = 260.4 keV | E50,hi = 260.4 keV (variants 250, 269.9) | likely |
| SR1 NREFT search saw no NR-band-like events at high S1c (it set limits) | n_SR1 = 0 baseline; n = 1 scenario | uncertain |
| LUX 2021 EFT: 311.2 live days (certain, bib title); 3.35 × 10⁴ kg·d = 0.0917 t·yr | 0.0917 t·yr | likely |
| LUX ROI edge in recoil energy | 150 keV baseline; 100, 250 keV variants | uncertain |
| PandaX-II 2019 (Runs 9+10): 54 t·d = 0.148 t·yr | 0.148 t·yr | likely |
| PandaX-II ROI edge | 100 keV baseline; 50, 150 variants | uncertain |
| XENON1T 2018: 1.0 t·yr, ROI 4.9–40.9 keV | as stated | certain / likely |
| in-ROI plateau efficiencies | 0.85 (XENON1T), 0.8 (LUX, PandaX-II) | likely / uncertain |
| Poisson/PLR statistics: 90 % zero-event UL = 2.303 events; two-sided 90 % Δ(−2lnL) = 2.706; asymptotic one-sided 1.642 | — | certain |

## 3. Method

**Efficiencies.** LZ-like: ε(E) = p · ½[1 + erf((E − 5.4)/(√2·2.5))] · {1 − ½[1 + erf((E − E50)/(√2·11.5))]} with p = 0.96, E50 = 269.9 keV
(P005/P021 model of Fig. S2). SR1: same with E50 = 260.4 keV. Hard-edge experiments: p · erf turn-on at E_lo (σ 1.5 keV) · Θ(E_hi − E).

**Spectra.** True-recoil spectra s(E) normalised to 1.0 LZ event: N_X = E_X ∫ s(E) ε_X(E) dE. For δ = 366 keV (P007's Higgsino solution) the
O1ˢ spectrum was computed here with WimPyDD (`lz.wd_rate`, c_p = c_n = 1/m_v², Sun-frame halo, 1–800 keV step 2). Unit-coupling counts
N_unit(δ) = 2.84 ∫ (dR/dE)_{κ=1} ε_LZ dE for δ = 0–250 keV were computed on a 0.5 keV (E < 60) / 2 keV grid; δ = 300/350 keV taken from P035.
A first attempt with 0.5 keV steps for eight δ values plus five L10 masses exceeded the 600 s foreground limit (~0.3 s per WimPyDD call,
~1 s for the q-dependent L10 Hamiltonian); the run was stopped and restarted with the reduced grids above and per-item caching
(`output/work/P059/cache/*.npy`); the L10 spectra were taken from P012 instead of being recomputed (one own 200 GeV L10 spectrum, computed
before the stop, agrees with P012's in shape to 4 significant figures: constant ratio 5.7408 × 10⁷ over 101–299 keV, a pure coupling-normalisation
factor).

**Combined likelihood.** s = signal rate in LZ-window events per t·yr. L(s) = Pois(1; 2.84 s + b_LZ) × Π_j Pois(0; E_eff,j s + b_j) with
E_eff,j = 2.84 N_j (LZ-equivalent exposure of archival set j) and b_j = b_LZ E_eff,j/2.84 (negligible). b_LZ = 2 × 10⁻⁴ (neighbourhood, dossier),
5.7 × 10⁻⁴ (P016 anchor), 0.0106 (whole S1c > 500 phd panel). ŝ from `minimize_scalar`; 90 % two-sided interval from Δ(−2lnL) = 2.706;
one-sided UL from Δ = 1.642; q0 = 2[lnL(ŝ) − lnL(0)], Z = √q0 (a Poisson-in-a-box proxy for LZ's 3.4σ; P001 obtained 3.54σ for b = 2 × 10⁻⁴
with the exact Poisson tail and 3.88σ asymptotically — our LZ-alone Z is the asymptotic value).

**Fig. 6 consistency.** For the SR1 limit κ_lim(δ): N_SR1(κ_lim) = N_unit(δ) κ_lim (0.9035/2.84) A_SR1/A_LZ. For LUX and PandaX-II, whose acceptance is
unknown, we invert: the LZ-equivalent effective exposure implied by a zero-event limit is E_eff = 2.303 × 2.84/(N_unit κ_lim). For L10:
N_SR1(m) = N(d10 = 1) d10,lim² (0.9035/2.84) A_SR1/A_LZ in both normalisations.

## 4. Results

### 4.1 Unit-coupling counts and validation (`P059_Nunit_O1_1000GeV.csv`)
| δ [keV] | 0 | 50 | 100 | 150 | 200 | 250 | 300* | 350* | 366 |
|---|---|---|---|---|---|---|---|---|---|
| N_unit (2.84 t·yr) | 1.646e9 | 6.248e8 | 1.038e8 | 1.182e7 | 1.008e6 | 9.969e4 | 1.256e4 | 154.6 | 4.85 |
| κ̂ = 1/N_unit | 6.08e-10 | 1.60e-9 | 9.63e-9 | 8.46e-8 | 9.92e-7 | 1.003e-5 | 7.96e-5 | 6.47e-3 | 0.206 |
| ratio to P035 | – | – | 1.0001 | 1.0003 | 1.0005 | 0.9998 | 1 | 1 | 1.009† |
(*from P035; †P035 log-interpolated between 365 and 370 keV.) Own L10 unit count at 1000 GeV 3.331 vs P012's 3.336 (ratio 0.999).

### 4.2 Expected counts at the LZ best fit (`P059_archival_counts.csv`, `P059_archival_totals.csv`; Fig. 1)
| exposure (t·yr) | acceptance | L10 1 TeV | δ = 250 | δ = 300 | δ = 350 | δ = 366 |
|---|---|---|---|---|---|---|
| LZ SR1 0.9035 | LZ-like | 0.318 | 0.318 | 0.318 | 0.318 | 0.318 |
| LZ SR1 | 260 keV edge (actual) | 0.308 | 0.318 | 0.317 | 0.313 | 0.286 |
| XENON1T 1.0 | 4.9–40.9 keV | 0.018 | 0 | 0 | 0 | 0 |
| LUX 0.0917 | ≤ 150 keV (100 / 250) | 0.009 (0.004 / 0.025) | 0.011 (0.003 / 0.027) | 0.005 (2e-7 / 0.027) | 0 (0 / 0.026) | 0 (0 / 0.023) |
| PandaX-II 0.148 | ≤ 100 keV (50 / 150) | 0.0065 (0.003 / 0.015) | 0.005 (0 / 0.017) | 3e-7 (0 / 0.008) | 0 | 0 |
| **total, actual ROIs** | | **0.342** | **0.333** | **0.322** | **0.313** | **0.286** |
| P(0), actual | | 0.71 | 0.72 | 0.72 | 0.73 | 0.75 |
| total without SR1 | | 0.034 | 0.016 | 0.005 | 0 | 0 |
| total, LZ-like acceptance (2.143 t·yr) | | 0.755 (P(0) = 0.47) | same | same | same | same |
SR1 variants: E50 = 250 keV → 0.294 (L10), 0.259 (366); plateau 0.85 → 0.273/0.253; E50 = 269.9 → 0.318/0.318. Scaling with LZ's 68 % band
(0.3–2.4 events) multiplies every entry by 0.3–2.4: the SR1 expectation is 0.09–0.74 events, P(0) = 0.48–0.91.

### 4.3 Combined likelihood (`P059_combined_likelihood.csv`; Fig. 2)
| case (b_LZ = 2e-4) | E_eff,arch [t·yr] | ŝ [/t·yr] | ŝ/ŝ_LZ | 90 % interval | one-sided UL | Z | ΔZ |
|---|---|---|---|---|---|---|---|
| LZ alone | – | 0.352 | 1 | 0.037–1.284 | 1.013 | 3.877 | – |
| + archival, actual ROIs, L10 | 0.970 | 0.262 | 0.745 | 0.028–0.957 | 0.755 | 3.801 | −0.077 |
| + archival, actual ROIs, δ = 250/300/350 | 0.947/0.916/0.888 | 0.264/0.266/0.268 | 0.750/0.756/0.762 | 0.028–0.96/0.97/0.98 | 0.760/0.766/0.772 | 3.803/3.805/3.807 | −0.075/−0.073/−0.071 |
| + archival, actual ROIs, δ = 366 | 0.812 | 0.274 | 0.778 | 0.029–0.999 | 0.788 | 3.812 | −0.065 |
| + all 2.143 t·yr with LZ-like acceptance | 2.143 | 0.201 | 0.570 | 0.021–0.732 | 0.578 | 3.730 | −0.148 |
Other backgrounds: b = 5.7e-4: Z 3.597 → 3.515 (actual) / 3.438 (LZ-like); b = 0.0106: 2.667 → 2.556 / 2.451. ŝ and intervals change by < 2 %.
The interval-shrinkage mechanism is purely the extra exposure: ŝ = [E_LZ/(E_LZ + E_arch) − b]/E_LZ, so the significance (set by ln(1/b)) barely moves.

### 4.4 Fig. 6 top: LZ 2024a (SR1) and LUX 2021 versus the new fit (`P059_fig6_top_consistency.csv`, `P059_fig6_top_LUX.csv`; Fig. 3)
| δ [keV] | 0 | 50 | 100 | 150 | 200 | 250 |
|---|---|---|---|---|---|---|
| κ_lim SR1 (digitised) | 9.35e-9 | 3.46e-8 | 2.84e-7 | 2.37e-6 | 1.16e-5 | 6.18e-5 |
| κ_lim/κ̂_new | 15.4 | 21.6 | 29.5 | 28.0 | 11.7 | 6.2 |
| κ_lim/new upper edge | 5.7 | 4.6 | 3.7 | 3.1 | 1.90 | 1.59 |
| κ_lim/new median sensitivity (exposure ratio 3.14) | 3.2 | 4.2 | 5.4 | 6.3 | 4.8 | 3.5 |
| implied SR1 events at κ_lim, LZ-like acceptance | 4.9 | 6.9 | 9.4 | 8.9 | 3.7 | 1.96 |
| same, 260 keV edge | 4.9 | 6.9 | 9.4 | 8.9 | 3.7 | 1.96 |
| SR1 events at the new best fit | 0.318 | 0.318 | 0.318 | 0.318 | 0.318 | 0.318 |
| LUX κ_lim | 1.41e-7 | 3.92e-7 | 3.19e-6 | 5.27e-5 | 3.79e-4 | – |
| LUX κ_lim/κ̂_new | 232 | 245 | 331 | 622 | 382 | – |
| LUX implied E_eff (2.3 events) [t·yr] / fraction of 0.0917 | 0.028 / 0.31 | 0.027 / 0.29 | 0.020 / 0.22 | 0.0105 / 0.11 | 0.017 / 0.19 | – |
Reading: at δ = 250 keV the SR1 limit corresponds to 1.96 expected events in 0.90 t·yr, i.e. a zero-event PLR-type limit (asymptotic
one-sided 90 % for n = 0 is 1.35 events; a power-constrained or two-sided construction 2–2.4), consistent with SR1 having seen nothing signal-like
at high S1c; at δ ≤ 150 keV the implied 5–9 events show the SR1 limits were background-limited (the spectra reach into the low-energy ER-leakage
region). The new best fit sits 6.2× (250 keV) to 30× (100 keV) below the SR1 limit, i.e. 0.32 events in SR1 for every δ. The new *upper* edge
is only 1.6–1.9× below the SR1 limit at δ = 200–250 keV despite 3.1× the exposure, because the event lifts it; the median sensitivity improved by
3.2–6.3× (more than the exposure ratio at 50–200 keV: smaller FV, better MSSI handling). LUX's limits imply LZ-equivalent effective exposures of
0.010–0.028 t·yr, 11–31 % of its 0.092 t·yr, consistent with an ROI ending near 100–150 keV plus efficiency < 1 (our baseline acceptance ratios
0.16–0.33 at δ = 250–300 keV).

### 4.5 Fig. 6 bottom: LZ 2024b (SR1, L10) and PandaX-II 2019 (`P059_fig6_bottom_consistency.csv`)
| m [GeV] | 100 | 200 | 400 | 1000 | 4000 |
|---|---|---|---|---|---|
| d10 SR1 limit (digitised) | 0.943 | 0.599 | 0.619 | 0.821 | 1.537 |
| new best fit d10 (P012) | 0.238 | 0.187 | 0.202 | 0.279 | 0.525 |
| (d10,lim/d10,bf)² | 15.7 | 10.3 | 9.4 | 8.7 | 8.6 |
| d10,lim / new median | 2.07 | 2.01 | 2.01 | 1.97 | 1.95 (√3.14 = 1.77) |
| implied SR1 events at limit, LZ (Fig. 1) normalisation | 4.97 | 3.22 | 2.90 | 2.67 | 2.63 |
| same, WimPyDD/Anand normalisation | 1.29 | 0.83 | 0.75 | 0.69 | 0.68 |
| PandaX-II d10 limit | 6.9 | 9.1 | 12.5 | 19.6 | 36.7 |
| PandaX-II implied E_eff, LZ norm [t·yr] | 0.0077 | 0.0027 | 0.0017 | 0.0013 | 0.0013 |
| same, WD norm | 0.030 | 0.011 | 0.0066 | 0.0051 | 0.0052 |
| 0.148 t·yr × f(E < 50 keV) / × f(E < 100 keV) | 0.054/0.084 | 0.019/0.037 | 0.012/0.025 | 0.009/0.020 | 0.008/0.018 |
Reading: in the LZ (Fig. 1) normalisation the SR1 L10 limits correspond to 2.6–3.2 events for m ≥ 200 GeV (a proper zero-event limit); in the
WimPyDD/Anand normalisation they would correspond to 0.68–0.83 events, below any 90 % CL construction (minimum 1.35 for n = 0). The SR1 limit
therefore independently supports the normalisation adopted by P012 (relevant to DR-001; the residual factor 3.86 is between the two conventions,
and the SR1 test discriminates them at the level of the ≈ 2.3-event floor). PandaX-II's limits correspond to effective exposures 7–15 % (LZ norm) or
25–65 % (WD norm) of its 0.148 t·yr × low-energy L10 fraction, so they were background- or acceptance-limited; no normalisation test is possible there.

### 4.6 SR1 look-back and joint SR1 + SR3 fit (`P059_SR1_lookback.csv`, `P059_joint_SR1_SR3.csv`)
Baseline (E50 = 260.4 keV, plateau 0.96): N_SR1 = 0.308 (L10), 0.318 (δ = 250), 0.317 (300), 0.313 (350), 0.286 (366 keV); P(0) = 0.73–0.75,
P(≥ 1) = 0.25–0.27; at LZ's 68 % upper edge (2.4 events) 0.69–0.76 events, P(0) = 0.47–0.50; at the lower edge 0.09 events.
Joint fit (b = 2e-4), n_SR1 = 0: ŝ = 0.267–0.274 /t·yr (×0.76–0.78), 90 % upper edge 1.28 → 0.97–1.00, Z −0.07.
n_SR1 = 1 (one NR-band event at S1c 400–600 phd in SR1): ŝ = 0.53–0.55 /t·yr (×1.52–1.56), 90 % interval 0.13–1.42/1.45, Z 3.88 → 5.63–5.64
(b = 0.0106: 2.67 → 3.99). A joint SR1+SR3 analysis is therefore the cheapest decisive step available to LZ: a null costs a quarter of the rate,
one event would take the Poisson-proxy significance past 5σ locally.

## 5. Figures
- `figures/P059_fig1_archival_counts.png` — expected events per archival exposure (bars: actual ROI; ticks: LZ-like acceptance), five models.
- `figures/P059_fig2_combined_profile.png` — Δ(−2 ln L) versus signal rate: LZ alone, + archival with actual ROIs (L10, δ = 366), + all 2.14 t·yr LZ-like.
- `figures/P059_fig3_fig6top_consistency.png` — digitised Fig. 6 top (new edges, SR1 and LUX limits, new best fit κ̂) and implied SR1 counts.

## 6. Robustness and caveats
- SR1 edge 250 vs 270 keV changes the SR1 expectation by −8 %/+3 % (L10) and −9 %/+11 % (δ = 366); plateau 0.85 by −11 %.
- LUX edge 100–250 keV: 0.003–0.027 events; PandaX-II 50–150 keV: 0–0.017 events. Both are below 0.03 events under every variant; the archival
  information is entirely SR1's.
- Halo: Sun-frame spectra; annual-mean shapes shift the δ = 350/366 acceptance ratios by a few per cent (P035: rates ×1.08/1.7 but the normalisation
  to 1 LZ event cancels the rate change; only the shape near the edge matters).
- Z values are Poisson-in-a-box asymptotic proxies (LZ alone 3.88 for b = 2e-4 versus LZ's 3.4 from the 2D fit); only ΔZ is meaningful.
- The digitised previous limits carry ±5 % reading uncertainty; the SR1 g1 and the "zero high-energy SR1 events" are recalled/uncertain, so the
  1.96-event implied count at δ = 250 keV is uncertain at the ±15 % level — it cannot distinguish a 1.35-event asymptotic limit from a 2.3-event one,
  but it does exclude SR1 having contained several signal-like high-energy events.
- A HEPData/Data-Release table of the SR1 NREFT limits and observed high-energy event count would sharpen 4.4 and 4.6; not filed as a data request
  because the headline (0.3 expected SR1 events, P(0) = 0.73; ŝ × 0.75) does not depend on it.

## 7. Failed or abandoned
- Full-grid WimPyDD run (0.5 keV steps, δ = 0–350 in 8 values, five L10 masses) exceeded the 600 s foreground limit twice; replaced by reduced grids,
  per-item caching and P012's L10 spectra (validated: identical shape). No result depends on the abandoned grid.
- An own δ = 350 keV spectrum cross-check against P005 was dropped for time; P005's normalisation was instead re-verified by integration (§2.2).

## 8. References
LZ Collaboration, arXiv:2609.02823 (2026); J. Aalbers et al. (LZ), PRD 109, 092003 (2024), arXiv:2312.02030; J. Aalbers et al. (LZ), PRL 133, 221801 (2024),
arXiv:2404.17666; D. S. Akerib et al. (LUX), PRD 104, 062005 (2021), arXiv:2102.06998; J. Xia et al. (PandaX-II), PLB 792, 193 (2019), arXiv:1807.01936;
E. Aprile et al. (XENON), PRL 121, 111302 (2018); E. Adams et al. (PICO), PRD 108, 062003 (2023); G. Cowan et al., EPJC 71, 1554 (2011);
I. Jeong et al. (WimPyDD), CPC 276, 108342 (2022). Corpus: dossier 00, P001, P005, P007, P012, P015, P016, P021, P035, P038.

## 9. Tools and provenance (mirrors `output/provenance/P059.json`)
- Agent tools: Read (PAPER_GUIDE, dossier, ledger via pandas, P005/P035/P015/P012/P021/P038 papers, figS7_curves.json, Fig. 6 PNG, three own
  figures), Bash (greps of the tex and corpus, exploratory pandas/pymupdf snippets, script runs), Write (script, details, provenance, paper),
  Edit (script fixes), ToolSearch/Monitor/TaskStop (harness only).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (special.erf, optimize.minimize_scalar/brentq, stats); pandas 3.0.5; matplotlib 3.11.2;
  pymupdf 1.28.2 (page.get_drawings); WimPyDD 2.0.4 (eft_hamiltonian, streamed_halo_function, diff_rate via lzcommon); common/lzcommon.py
  (LZ, wd_halo, wd_hamiltonian, wd_rate, M_V_GEV).
- Script: `output/code/P059_archival_xenon.py` (`.venv/bin/python output/code/P059_archival_xenon.py`; `--recompute` to refill the cache).
- WimPyDD-generated files: none outside `output/work/P059/cache/` (diff_rate does not write response files).
- Recalled items: 11 (see §2.3). Data requests: none. Datasets: none.
