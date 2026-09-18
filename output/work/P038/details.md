# P038 — The efficiency edge: how the S1c < 600 phd boundary shapes inference at 250–270 keV and what an extended ROI would have shown

Simulated date 2026-09-09 · category STAT/RESP · physics.data-an (cross-list hep-ex) · analysis-methods experimentalists.
Script: `output/code/P038_efficiency_edge.py` (run from the simulation root with `.venv/bin/python`; 2 s after the WimPyDD
kernels are cached, ~70 s for the first run). Results: `output/work/P038/P038_results.json` and the CSVs listed in §9.

## 1. Motivation and framework

LZ's WIMP-search ROI is S1c = 3–600 phd, S2 > 645 phd, S2c = 10^2.75–10^4.15 phd (tex l. 114). The S1c range "is the same as
used in the previous NREFT search" (l. 116), i.e. it was fixed in 2023, before the AmBe calibration that tuned the NR model above
~100 keV and before the salt (generated "before the AmBe calibration", l. 160, only "out to 250 ± 25 keV") failed to cover the
high-energy region. The S2c maximum "is chosen to keep essentially all of the signal distribution while eliminating the vast
majority of ER backgrounds" (l. 115). Fig. S2 puts the 50 % efficiency point at 269.9 keV (l. 456); the candidate is quoted at
248 ± 23 ± 23 keV, ~20 keV below it, and P009 showed that with the printed Table S5 yields it reconstructs at 262 keV (ε = 0.71)
whereas on LZ's own contour scale it is 246 keV (ε = 0.93).

The S1c edge therefore acts on three distinct pieces of the inference: (a) the per-event efficiency (and hence any
efficiency-corrected count or the position of the event relative to the accepted spectrum); (b) the total accepted signal per
unit coupling, S_acc(δ) = κ⁻¹ × expected events, which sets the fitted coupling κ̂ = 1/S_acc for one event and which the raster
scan over δ (l. 500: "we perform a raster scan over masses or masses and mass splittings … we do not set an additional threshold
requirement on the local significance") reports without regard to how much of the spectrum is hidden; and (c) the shape
term of the single-event profile likelihood, which, as P021 showed, depends only on the accepted-spectrum density at the event
energy per accepted event, f̃ = f(E_obs)/S_acc. Because the edge removes the part of an inelastic spectrum above ~270 keV, it
simultaneously lowers S_acc and raises f̃ for large δ: the δ ≈ 380 keV preference found by P021 is a statement about the
accepted spectrum, not about the physical one. We quantify all three effects for alternative edges (700, 800, 1000, 1200 phd),
estimate what backgrounds an extended ROI would face, and treat the E_true = 262 keV scenario.

## 2. Efficiency model (part 1)

**Method.** For each true NR energy E (200–600 keV, 4 keV steps, 6000 samples; 40 000 samples at 246, 248, 262, 265, 269.9 keV)
we draw (N_ph, N_e) with nestpy 2.1.1 `GetQuanta` (detector `LZ_WS2024`, its 13-entry `nr_er_width_parameters`, field 96.5 V/cm,
density 2.9) around the Table S5 yields with the p(E) break (`lz.nest_nr_params_vector`), then shift the photon mean so that
N_q = 11.32 E^1.112 (P009's "paper-contour" scale, from the six keV_ee/keV_nr contour labels of Fig. 4; W = 13.46 eV). Detector
response as in P009: S1c = Binomial(N_ph, g1 = 0.110) × (1 + N(0, 0.338/√n1)) × (1 + N(0, 0.02)); S2c = Binomial(N_e, 0.726) ×
(34.5/0.726) × (1 + N(0, √(4/S2))) × (1 + N(0, 0.02)). Efficiency for edge X:

  ε_X(E) = 0.955 × P(S1c < X and log10 S2c < 4.15 | E)   (0.955 = Fig. S2 green "+SS & cuts" level).

Below 200 keV the plateau with P021's low-energy erf (50 % at 5.4 keV, σ 2.5 keV) is used; it is irrelevant for δ ≥ 250 keV
(recoil windows start above 55 keV).

**Validation.** E50(600) = 271.6–271.8 keV over repeated runs (nestpy's internal generator is not seeded), against Fig. S2's
269.9 keV and P009's 271.7 keV; the MC curve reproduces P009's eleven by-eye inset readings (±0.03) with rms residual 0.028 and
maximum 0.06 (at 285 keV, where the MC tail is slightly fatter than the drawn curve). The erf width fitted to the MC roll-off is
10.9 keV (P009: 11.8 keV from the inset). The S1c median at E50 is 598.5 phd, as it must be. On the Table-S5-as-printed scale
ε_600(270) = 0.93 and ε_600(290) = 0.53 (edge near 291 keV, P009), inconsistent with Fig. S2; that scale is not used further.

**Edges.** (with the S2c < 10^4.15 cut applied; the S1c-only 50 % point differs by ≤ 2 keV up to 1200 phd)

| S1c edge [phd] | E50 [keV] | erf σ [keV] | S1c median at E50 |
|---|---|---|---|
| 600 | 271.6 | 10.9 | 598 |
| 700 | 310.5 | 11.7 | 698 |
| 800 | 348.7 | 12.7 | 798 |
| 1000 | 423.2 | 14.5 | 997 |
| 1200 | 495.1 | 18.0 | 1192 |

The S2c < 10^4.15 cut alone removes 10 % of NRs at 503 keV (S1c median 1212 phd) and 50 % at 600 keV (1484 phd): a fixed S2c
ceiling starts to cost signal once the S1c edge passes ~1200 phd. d ln S1c/d ln E = 1.154 between 248 and 300 keV, so g1 ± 2 %
(paper: 0.110 ± 0.002) moves E50(600) by ∓4.7 keV and E50(1000) by ∓7.3 keV.

**Efficiency at the event energies** (MC, 40 000 samples):

| E_true [keV] | ε_600 | P(S1c > 600) | 1/ε_600 | S1c median ± sd [phd] | ε_700…1200 |
|---|---|---|---|---|---|
| 246 | 0.950 | 0.53 % | 1.05 | 533 ± 26 | 0.955 |
| 248 | 0.945 | 1.06 % | 1.06 | 538 ± 26 | 0.955 |
| 262 | 0.79 | 17.1 % | 1.26 | 574 ± 27 | 0.955 |
| 265 | 0.71 | 25.3 % | 1.40 | 582 ± 27 | 0.955 |
| 269.9 | 0.56 | 41 % | 1.78 | 594 ± 28 | 0.955 |

P009's P(S1c > 600) = 0.7 % at 246 keV is reproduced (0.5 %). The LZ curve itself (P009 erf fit, E50 = 269.9, σ 11.8) gives
ε(262) = 0.71, our MC (E50 = 271.6) 0.79; the 2 keV offset is within the g1 systematic. We quote 1/ε(262) = 1.26–1.4.

## 3. Inelastic spectra over the full kinematic window (part 2)

WimPyDD 2.0.4 O1 isoscalar, m_χ = 1000 GeV, LZ unit coupling (c_1^s m_v²)² = 1 ⇔ WimPyDD c⁰ = 2/m_v² (P003 convention,
`lz.wd_c_from_anand`), per-stream kernels `WD.diff_rate(..., sum_over_streams=False)` on the explicit v grid 0–830 km/s (1661
points) dotted with the Baxter-2021 halo function: "annual" = mean of 12 monthly days (as P021), "june" = day 167. True-energy
grid 1.5–798 keV in 3 keV steps, evaluated inside the kinematic window of the lightest and heaviest isotopes (padded by one
step); δ = 250, 275, 300, 310, …, 350, 355, …, 390 keV; 3417 kernel calls; cached in `spectra_s_1000_full.npz`. On the overlap
100–330 keV the δ = 300 keV spectrum agrees with P021's cache to 1.0000 (same code path), so the only difference from P021 is the
extension to 800 keV (the δ = 300 keV window is 91–794 keV; δ = 380: 165–691 keV).

Observed-energy smearing: Gaussian σ_E = 11 √(E/248) keV (P009/P021). Exposure 2.84 t·yr. For each δ, halo and edge X:

  S_acc(X) = Σ_E dR/dE ε_X(E) ΔE × 2.84,   A(X) = S_acc(X)/S_full,   S_full = Σ_E dR/dE ΔE × 2.84,
  f(E_obs) = Σ_E dR/dE ε_X(E) × 2.84 × Gauss(E_obs − E; σ_E) ΔE,   f̃ = f(E_obs)/S_acc,
  q0 = 2[ln(f̃/b_d) − 1 + b_d/f̃],  b_d = b_H/W_H = 5.7×10⁻⁴/70 keV⁻¹,  Z = √q0,  κ̂ = 1/S_acc (μ̂ = 1).

The q0 formula is P021's companion-free single-event profile (extended likelihood, κ profiled). It ignores the low-energy bins,
which for δ ≥ 300 keV carry < 1 % of the accepted rate; it also assumes that no event appears in the extension 600 phd – X, which
is what LZ's Fig. S4 (4.7 t, science) shows: no points between 600 and 800 phd below log10 S2c = 4.15 and zero in the HE SB
800–1700 phd (Table 'MSSI comparison': 0.1 + 0.5 predicted, 0 observed). Backgrounds in the extension enter both hypotheses
identically and cancel in q0.

**Comparison with P021 at the 600 phd edge** (`P038_vs_P021.csv`): Z agrees within 0.11σ for δ ≤ 385 keV (2.670 vs 2.674 at 300;
3.123 vs 3.121 at 350; 3.475 vs 3.586 at 380); S_acc agrees to 0.5 % at δ = 300 keV and grows to ×1.5 at 380 and ×2.1 at 385 keV
because there the accepted rate is dominated by the roll-off tail, where our MC efficiency (E50 271.6, non-Gaussian tail) is a
little more permissive than P021's erf (269.9, σ 11.5). Both give the peak at δ = 380 keV.

### 3.1 Acceptance and Z per edge (annual halo; June in the CSV differs by ≤ 0.07 in A and ≤ 0.12σ)

| δ [keV] | A_600 | A_700 | A_800 | A_1000 | A_1200 | f(E_true > 270) | Z_600 | Z_700 | Z_800 | Z_1000 | Z_none | S_acc(1000)/S_acc(600) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 | 0.885 | 0.896 | 0.917 | 0.948 | 0.953 | 0.073 | 2.44 | 2.44 | 2.43 | 2.42 | 2.41 | 1.07 |
| 300 | 0.815 | 0.835 | 0.879 | 0.941 | 0.953 | 0.148 | 2.67 | 2.67 | 2.65 | 2.62 | 2.62 | 1.16 |
| 320 | 0.744 | 0.775 | 0.841 | 0.935 | 0.952 | 0.222 | 2.80 | 2.79 | 2.76 | 2.73 | 2.72 | 1.26 |
| 340 | 0.582 | 0.637 | 0.756 | 0.923 | 0.951 | 0.393 | 3.00 | 2.98 | 2.92 | 2.85 | 2.84 | 1.59 |
| 350 | 0.444 | 0.520 | 0.684 | 0.915 | 0.950 | 0.539 | 3.12 | 3.08 | 2.99 | 2.89 | 2.87 | 2.06 |
| 360 | 0.272 | 0.374 | 0.597 | 0.905 | 0.949 | 0.720 | 3.26 | 3.17 | 3.02 | 2.88 | 2.86 | 3.33 |
| 370 | 0.115 | 0.240 | 0.519 | 0.901 | 0.948 | 0.886 | 3.41 | 3.19 | 2.94 | 2.75 | 2.73 | 7.8 |
| 380 | 0.029 | 0.165 | 0.481 | 0.905 | 0.949 | 0.977 | 3.47 | 2.96 | 2.58 | 2.33 | 2.30 | 31 |
| 390 | 0.009 | 0.143 | 0.478 | 0.915 | 0.949 | 0.997 | 3.26 | 2.44 | 1.90 | 1.56 | 1.54 | 103 |

Cross-checks: A_600 = 81/44/19/2.9 % at δ = 300/350/365/380 keV vs P015's ROI capture 82/45/17/2.8 %; f(E_true > 270) at δ = 300
is 14.8 % (June 14.3 %) vs P002's 13 %. Median true energies of the full spectra: 178 (300), 298 (350), 343 (370), 347 keV (380);
the spectra above δ ≈ 350 keV are bimodal about the 266 keV M-response node (P002, P017), with the high-energy lobe peaking at
343 keV — exactly the region the 600 phd edge removes and a 1000 phd edge (E50 = 423 keV) keeps.

**Peaks of Z(δ).** Edge 600: peak δ = 380 keV, Z = 3.47, Δq0(380 − 300) = +4.9, likelihood ratio LR(380/300) = 11.9 (P021 with the
full likelihood: 3.59, Δq0 5.7). Edge 700: peak 365, Z 3.20, LR 2.3. Edge 800: peak 360, Z 3.02, LR 0.84. Edge 1000: peak 355,
Z 2.89, LR(380/300) = 0.48, Δq0(380 − 300) = −1.5. No edge (sharp S2c cut only): peak 355, Z 2.88. The observed percentile of the
event moves from the 60th (δ = 380, edge 600) to the 1.9th (edge 1000): with the full spectrum visible, 248 keV is in the
low-energy tail of every δ ≥ 360 keV spectrum, not at its median.

**Coupling per event.** κ̂_600 = 7.5×10⁻⁵ (300), 3.9×10⁻³ (350), 0.109 (370), 1.44 (380); κ̂_1000 = 6.5×10⁻⁵, 1.9×10⁻³, 0.0139,
0.0456 — the ratio is 1/gain = 1/1.16, 1/2.06, 1/7.8, 1/31. The best-fit couplings reported at δ ≥ 350 keV imply, per event fitted
in 3–600 phd, an expected 1.06 (350), 2.3 (360), 6.8 (370), 30 (380) events in 600–1000 phd; P(0 | signal) = e^−μ = 0.35, 0.10,
0.001, ~10⁻¹³. For δ ≤ 330 keV the extension adds ≤ 0.37 events per ROI event (P(0) ≥ 0.69), so the 600–1000 phd emptiness is
informative only for δ ≳ 350 keV. Table in `P038_signal_vs_background_extension.csv` (also scaled ×2.38 to P020's 6.8 t·yr).

## 4. Backgrounds in an extended ROI (part 3)

**Inputs** (Table 'MSSI comparison', 4.7 t FV, science sample, tex l. 773–778): WS ROI 3–600 phd: wall 0.0048, RFR 0.0001;
HE SB 800–1700 phd (S2c < 10^4.3): wall 0.1 (0 observed), RFR 0.5 (0 observed). In the disjoint 5.4 t annulus the HE SB RFR
prediction 21.5 vs 18 observed validates the RFR normalisation to ~20 %. The MSSI S1c spectrum between 600 and 800 phd is not
published; we bracket it with two monotonic densities fitted to the two integrals per component:

- exponential n(s) = a e^{s/λ}: λ_wall = 379 phd, λ_RFR = 129 phd;
- power law n(s) = a s^p: p_wall = 2.02, p_RFR = 7.18.

(The steep RFR rise is expected: RFR MSSI in the HE SB is ²¹⁴Pb with the γ fully contained in the reverse-field region, i.e.
S1-only deposits of 242–352 keV [recalled ²¹⁴Pb lines, likely], whose S1c peaks lie inside 800–1700 phd; below 800 phd only
partial-absorption γ's contribute.) Geometric mean of the two models taken as central.

| extension 600 → X phd | all MSSI (pow – exp) | central | with S2c < 10^4.15 (×10/18) | NR band ±2σ (central; bracket) |
|---|---|---|---|---|
| 700 | 0.0019–0.0031 | 0.0025 | 0.0014 | 0.0004 (0.0001–0.0005) |
| 800 | 0.0046–0.0076 | 0.0059 | 0.0033 | 0.0009 (0.0003–0.0011) |
| 1000 | 0.0135–0.024 | 0.018 | 0.010 | 0.0027 (0.0008–0.0034) |
| 1200 | 0.034–0.063 | 0.046 | 0.026 | 0.0069 (0.0022–0.0084) |

NR-band fraction f_NR: in Fig. S4 (5.4 t HE SB science, 18 events) 2–3 events lie within ~0.065 dex of the extrapolated NR median
(log10 S2c ≈ 4.07–4.13 at 800–1700 phd) — a visual reading giving f_NR ≈ 0.11–0.17; we take 0.15 with a bracket 0.035 (P004's
neighbourhood fraction in the ROI) to 0.25. Of the same 18 events ~8 lie above log10 S2c = 4.15 (visual), so an extended ROI
keeping the 10^4.15 ceiling admits ~55 % of the HE-SB-type population; the NR-band subset is unaffected. The NR-band MSSI in the
extension reaches 0.001/0.003/0.01 events (2.84 t·yr) at edges of 818/1022/1278 phd.

Other backgrounds in 600–1000 phd inside the NR band: ER leakage — the ER median in Fig. 4 is at log10 S2c ≈ 4.78 at 600 phd and
5.0 at 800 phd (visual), ≥ 0.6 dex ≈ 12 ER-band widths above the 4.15 ceiling; the ¹²⁴Xe/¹²⁵I/¹³³Xe features sit at fixed energies
below 300 phd (P010) — negligible. Accidentals: P022 finds 6.3×10⁻⁴ at S1c > 500 phd with a falling isolated-S1 spectrum;
≲ 3×10⁻⁴ in 600–1000 phd (estimate). Neutrons (P013) and atmospheric neutrinos (P019) are < 10⁻⁴. Hence an ROI extended to
1000 phd would have added ≈ 0.003 (0.001–0.004) NR-band events and ≈ 0.02 events in total, to be compared with 0.0106 in the
present S1c > 500 phd panel and 0.0049 MSSI in the whole ROI.

## 5. The E_true = 262 keV scenario (part 4)

(i) Efficiency-corrected count: 1/ε = 1.06 at 248 keV → 1.26 (our MC) to 1.40 (LZ curve read at 262–265 keV). In the extended
likelihood this changes nothing: κ̂ = 1/S_acc has μ̂ = 1 for companion-free spectra whatever E_obs is; only a naive
efficiency-corrected count moves.
(ii) δ_max(1000 GeV, 16 June): 386.6 (248) → 389.4 (262) → 389.9 keV (265); `lz.delta_max_kev`.
(iii) P(S1c > 600 | 262 keV) = 17 % (25 % at 265; 0.5 % at 246, 1.1 % at 248): a genuine 262 keV recoil had a one-in-six chance of
falling outside the ROI altogether.
(iv) Shape term with E_obs = 262 keV (edge 600): Z = 2.15 (300), 2.74 (350), 3.15 (370), 3.41 (380), 3.52 (385), 3.60 (390 keV);
the peak moves from 380 to 390 keV (P021's E262 variant: 385 keV), i.e. hard against the June ceiling (δ_max = 389.4 keV).
(v) "Coincidence" of sitting just inside the edge: for a detected event (edge 600), P(245 < E_obs < 295 keV) = 1.1 % (δ = 300),
2.2 % (330), 4.8 % (350), 16 % (370), 38 % (380), against 10 % for a flat spectrum; P(E_obs > 270) = 0.16 % (300), 1.0 % (350),
23 % (380). Under the low-δ spectra that LZ tabulates the location is a 1–5 % coincidence; under δ ≥ 370 keV it is expected —
but those are exactly the spectra that predict 7–30 events beyond the edge.

## 6. Recommendation (part 5)

A single number summarises the trade-off: at 1000 phd (E50 = 423 keV) the acceptance is ≥ 0.90 for every δ up to the kinematic
ceiling (vs 0.03–0.81 today for δ = 380–300 keV), while the NR-band MSSI added is 0.003 (0.001–0.003) events per 2.84 t·yr and
0.006 per 6.8 t·yr (P020's untouched exposure), still below the current whole-panel background of 0.0106; all backgrounds in the
extension total ≈ 0.02. Beyond ~1200 phd two things change: the RFR MSSI rises with λ ≈ 130 phd (0.046 total at 1200, 0.6 at 1700),
and the fixed S2c < 10^4.15 ceiling begins to remove NRs (10 % at 503 keV, S1c ≈ 1200 phd). We therefore recommend S1c < 1000 phd
with an S2c ceiling that follows the NR band (e.g. median + 3σ_NR, P024's σ = 0.031 dex) rather than a fixed 10^4.15, and a
salt sample extending to at least 450 keV so that the roll-off is covered blind. Fiducial-volume optimisation is left to P079.

## 7. Robustness and failed approaches

- Halo: June vs annual changes A_600 by ≤ 0.07 and Z by ≤ 0.12σ; the qualitative result (peak moves to 355–360 keV, Z(380) drops
  to 2.3–2.4σ) holds for both.
- Efficiency shape: the MC edge (E50 271.6, fat tail) vs P021's erf (269.9, 11.5) changes Z(380; 600) by 0.11σ and S_acc(380) by
  ×1.5; at δ ≤ 370 keV the two agree to ≤ 0.02σ. g1 ± 2 % shifts every E50 by ∓1.7 %.
- MSSI interpolation: exponential vs power law differ by ×1.8 at 1000 phd; the f_NR bracket adds ×4; the combined bracket
  0.0008–0.0034 never approaches the signal gains for δ ≥ 350 keV (≥ 1 event per ROI event).
- Resolution: the P021 study found ≤ 5 keV peak shifts for σ_E = 8–15 keV; the mechanism here (S_acc growth) does not depend on
  σ_E at all.
- Abandoned: seeding nestpy's generator (not exposed through the `NESTcalc` object used here; run-to-run E50 scatter is 0.2 keV);
  a 2D (S1c, log S2c) likelihood per edge (unnecessary: the S1c cut acts on the S1c marginal, and the S2c ceiling is irrelevant
  below 1200 phd).
- Not done: 400 and 4000 GeV (P021: peaks at the 340 keV ceiling and at 400 keV; the mechanism — the accepted fraction collapsing
  toward the ceiling — is identical); O1 isovector and O4 (P021 found O1v flat in δ; the edge effect on S_acc is operator-independent
  for a given recoil window, so the numbers above apply to the isovector case within the shape difference).

## 8. Figures

- `figures/P038_fig1_efficiency_cuts.png` — ε_X(E) for X = 600–1200 phd (NEST-LZ MC, paper scale) with the Fig. S2 inset readings
  and the S2c-only curve; 50 % points annotated.
- `figures/P038_fig2_acceptance_Z_vs_delta.png` — left: accepted fraction of the 1000 GeV O1 inelastic spectrum vs δ per edge
  (dotted: sharp true-E < 270 keV); right: single-event profile Z(δ) per edge with P021's full-likelihood curve.
- `figures/P038_fig3_background_vs_edge.png` — left: MSSI entering 600 phd – edge in the 4.7 t FV (power-law to exponential band)
  and its NR-band ±2σ subset; right: extra signal events per ROI event for δ = 300/350/370/380 keV vs the NR-band MSSI.

## 9. Result files

`P038_mc_efficiency.csv` (MC per energy: medians, sds, pass probabilities per cut), `P038_acceptance_vs_delta.csv` (per halo, δ,
edge: S_acc, A, κ̂, f̃, q0, Z for E_obs = 248 and 262, observed percentiles, P(245–295), P(> 270)), `P038_vs_P021.csv`,
`P038_mssi_vs_edge.csv`, `P038_signal_vs_background_extension.csv`, `P038_results.json`, `spectra_s_1000_full.npz` (WimPyDD
kernels × halos), `run_log.txt`.

## 10. References

LZ Collaboration, arXiv:2609.02823 (2026): Data Analysis paragraph (l. 110–117), salting (l. 160–161), raster scan (l. 500),
Fig. 4 and S4, Fig. S2 caption (l. 454–458), MSSI validation and Table 'MSSI comparison' (l. 721–783).
M. Szydagis et al., NEST (nestpy 2.1.1). I. Jeong, S. Kang, S. Scopel, G. Tomar, WimPyDD, CPC 276, 108342 (2022).
D. Baxter et al., EPJC 81, 907 (2021). G. Cowan, K. Cranmer, E. Gross, O. Vitells, EPJC 71, 1554 (2011).
N. Anand, A. L. Fitzpatrick, W. C. Haxton, PRC 89, 065501 (2014). Corpus: P002, P004, P009, P010, P013, P015, P016, P017, P019,
P020, P021, P022, P024.

## 11. Tools and provenance (mirrors provenance/P038.json)

Agent tools: Read ×30 (PAPER_GUIDE, dossier, ledger ×2, P009/P024/P021/P002/P016/P020 papers, tex l. 108–130/148–158/310–322/
450–460/718–787, lzcommon.py ×2, P009 script ×2, P021 script ×2, Fig. S2/Fig. 4/Fig. S4 PNGs (+1 failed path), own figures ×5),
Bash ×17 (grep/ls/version listing, P021/P009 cache inspection, WimPyDD timing test, palette grep, four script runs, digest,
word-count checks ×3), Write ×5 (paper written twice), Edit ×16 (script ×5, details ×2, paper trims ×9, provenance ×1),
Skill ×1 (dataviz). Software: python 3.12.13; nestpy 2.1.1 (LZ_WS2024, GetYields, GetQuanta, get_s2Fano, get_sPEres);
WimPyDD 2.0.4 (eft_hamiltonian, streamed_halo_function via lz.wd_halo, diff_rate sum_over_streams=False); numpy 2.5.3;
scipy 1.18.1 (stats.norm, special.erf, optimize.brentq/curve_fit); pandas 3.0.5; matplotlib 3.11.2; common/lzcommon.py
(LZ, NEST_NR_LZ, nest_nr_params_vector, wd, wd_halo, wd_hamiltonian, wd_c_from_anand, E_R_range_keV, delta_max_kev, vmax_kms,
v_earth_kms, M_V_GEV). Recalled: Cowan et al. asymptotic q0 and the single-event profile form (certain); Poisson P(0) = e^−μ
(certain); ²¹⁴Pb γ lines 242/295/352 keV (likely; interpretive only). Datasets: none. Data requests: none.
