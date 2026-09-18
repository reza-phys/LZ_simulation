# P046 — A multi-target confirmation programme for a 300–390 keV inelastic splitting: xenon, iodine and tungsten as a δ-meter

Simulated date 2026-09-10 · hep-ph (cross-list hep-ex) · experiment-design phenomenologists · XEXP/PROJ
Script: `output/code/P046_multi_target.py` (run from the root with `.venv/bin/python`; second run uses `output/work/P046/kernels.npz`, `--recompute` regenerates). Run log: `output/work/P046/run_log.txt`.

## 1. Motivation and framework

P015 established the kinematic blindness map for inelastic dark matter (only A ≳ 96–125 nuclei scatter for δ = 300–380 keV at 1 TeV) and showed that, at equal coupling, the tungsten-to-xenon rate per tonne rises from 16 to 700 and the iodine-to-xenon rate falls from 0.59 to 0.02 as δ goes from 300 to 380 keV. This paper turns that observation into a *measurement programme*: because the ratio of counts on two nuclei is independent of the coupling normalisation, a pair of exposures (xenon + CaWO₄) acts as a **δ-meter**. We ask (a) how precisely δ can be read from the ratio alone, (b) which recoil windows and backgrounds each target faces, (c) which exposures give 3σ separation between δ = 300 and 380 keV and between inelastic DM and the elastic L10 dipole (whose W/Xe ratio is δ-independent), (d) whether the tungsten (or iodine) spectral endpoint E₊(δ) can be seen directly.

Model: isoscalar O₁ inelastic scattering, m_χ = 1 TeV (400 and 4000 GeV for the mass degeneracy), Baxter-2021 SHM (v₀ = 238, v_esc = 544 km/s, Earth-frame, WimPyDD `streamed_halo_function`), coupling convention of P003 (LZ/Anand unit coupling c₁ˢ = 1/m_v² ⇔ WimPyDD c⁰ = 2/m_v²). L10 as in P005/P012: 4[(q²/m_N²)O₄ − O₆] with WimPyDD c₄ = 8q²/(m_v²m_N²), c₆ = −8/m_v² (d₁₀ = 1). Every count is normalised to LZ's 1.0 event in 2.84 t·yr through the P005/P015 efficiency model ε(E) = 0.96 × erf edges at 5.4 keV (σ 2.5) and 269.9 keV (σ 11.5).

## 2. Method: exact factorisation of the WimPyDD rate

WimPyDD's `diff_rate` (package.py l. 5351–5480) evaluates
dR/dE = Σ_streams Σ_isotopes n_T,i · v_j² · dσ_i(v_j, E) · Δη_j, with dσ ∝ Θ(v_j > v_min,i) /v_j² for velocity-independent operators (O₁, O₄, O₆; l. 5265: `eft=np.choose(v>vmin,[0,eft_n_vel])`). Hence

  dR/dE(E, δ) = Σ_i K_i(E) · η(v_min,i(E, δ)),  η(v) = Σ_{v_j > v} Δη_j,  v_min,i = c (m_i E/μ_i + δ)/√(2 m_i E),

with **δ-independent per-isotope kernels K_i(E)** obtained from a single stream at 3000 km/s with Δη = 1 (km/s)⁻¹ (`isotopes_list={0:[i]}`), and η from the cumulative sum of the WimPyDD halo arrays. Conventions matched to WimPyDD: m_i = 0.931 A GeV (`element.mass`), c = 3×10⁵ km/s (`get_vmin`, l. 5130–5153; using 299 792 km/s instead shifts v_min by 0.5 km/s and inflates rates within 30 km/s of the ceiling by 5–20 %), strict inequality. Kernels for 1 TeV are evaluated directly on the 1 keV analysis grid (1–1500 keV; 37 k calls, 7 min), for 400/4000 GeV on 300 log points (Xe, W only). Tungsten needs P015's runtime `np` injection into the 18xW response modules.

**Validation** (`P046_summary.json → validation`): factorised/direct ratio = 1.0000 at 14 of 15 test points (Xe δ = 300 keV at 100/200/400 keV; W δ = 300 at 60/200/800 keV; I δ = 350 at 200/400; Xe δ = 380 at 350; L10 Xe at 50/200; W δ = 366 at 100/500/1000 keV); 1.063 at (Xe, δ = 380, E = 250 keV), a point in the last few km/s of the June tail where one 0.7 km/s stream bin matters. **LZ normalisation**: unit-coupling events in LZ = 104 619 / 13 568 / 264.5 / 21.66 / 0.687 / 0.037 at δ = 250/300/350/366/380/390 keV (P015: 104 591 / 13 565 / 264.6 / 21.7 / 0.684), L10 (d₁₀ = 1) 3.338 (P012: 3.34). Hence κ̂(δ) ≡ (c₁ˢm_v²)² = 9.56×10⁻⁶, 7.37×10⁻⁵, 3.78×10⁻³, 0.0462, 1.456, 26.7.

Halos: 12-day annual mean (as P015), 16 June (day 167), 16 December (day 350), and annual means with v_esc = 528/560 km/s and v₀ = 220/250 km/s. v_max(June) = 810.9 km/s; the annual-mean η has support up to the June v_max.

## 3. Rates and ratios (Part 1)

Windows: Xe_full; Xe_LZ270 (ε(E) above); Xe_5_1000 (0.96 × [5, 1000] keV); I_full, I_5_1000; W_full, W_10_1300, W_110_1300 (²⁰⁶Pb-recoil-safe), W_220_1300 (radiogenic-neutron-safe); Ge_full, Ge_1_1000. Unit-coupling rates per t·yr of *element*, annual halo (`rates_vs_delta.csv`, 5 keV steps; internal grid 1 keV):

| δ (keV) | κ̂ | Xe_full | Xe_LZ270 | Xe_5_1000 | I_5_1000 | W_10_1300 | Ge | W/Xe full | W(10–1300)/Xe(5–1000) | W(10–1300)/Xe(LZ270) | I/Xe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 | 9.56e−6 | 41 400 | 36 840 | 39 740 | 29 080 | 381 800 | 0 | 9.1 | 9.6 | 10.4 | 0.70 |
| 300 | 7.37e−5 | 5 838 | 4 777 | 5 604 | 3 450 | 93 070 | 0 | 15.9 | 16.6 | 19.5 | 0.591 |
| 330 | 4.95e−4 | 1 040 | 711 | 999 | 481 | 34 450 | 0 | 33.1 | 34.5 | 48.5 | 0.462 |
| 350 | 3.78e−3 | 208.9 | 93.1 | 200.6 | 61.7 | 17 070 | 0 | 81.7 | 85.1 | 183 | 0.295 |
| 366 | 0.0462 | 44.19 | 7.63 | 42.42 | 6.28 | 9 764 | 0 | 221 | 230 | 1 280 | 0.142 |
| 380 | 1.456 | 8.76 | 0.242 | 8.41 | 0.199 | 6 072 | 0 | 693 | 722 | 25 100 | 0.0227 |
| 390 | 26.7 | 1.81 | 0.013 | 1.74 | 0 | 4 344 | 0 | 2 390 | 2 495 | 3.3e5 | 0 |

W/Xe = 15.9/81.7/221/693 and I/Xe = 0.591/0.295/0.142/0.0227 at 300/350/366/380 keV reproduce P015 (16/82/222/699; 0.59/0.30/0.14/0.023). Germanium is exactly zero for δ ≥ 250 keV (δ_max(Ge, 1 TeV) = 232 keV, P015). The logarithmic slope d ln R_WX/dδ = 0.0157, 0.0359, 0.0547, 0.0698, 0.0992 keV⁻¹ at 300/330/350/366/380 keV (`delta_meter_systematics.csv`): the meter is 6× more sensitive at 380 than at 300 keV. Using the LZ 270 keV ROI as the xenon window steepens the ratio further (19.5 → 25 100) because the ROI edge removes the >270 keV lobe (P038), but then the reading inherits the ROI-edge acceptance systematic.

Halo variants (full windows): W/Xe at 350 keV = 81.7 (SHM), 55 (June halo), 174 (v_esc 528), 45 (v_esc 560); at 366 keV 221 / 123 / 834 / 94. Mass: W/Xe(300) = 45.7 / 15.9 / 12.5 for 400 / 1000 / 4000 GeV; W/Xe(350) = 31.7 at 4000 GeV (∞ at 400 GeV: xenon is blind, δ_max(Xe, 400 GeV) = 341 keV).

**L10 elastic** (`rates_L10.csv`; unit d₁₀, per t·yr element, annual): Xe_full 2.769, Xe_LZ270 1.175, Xe_5_1000 2.628, I_full 21.04, W_full 0.1376, W_10_1300 0.1294, W_110_1300 3.3×10⁻⁴, Ge 0.677. Ratios: W/Xe = 0.0497 (WimPyDD), I/Xe = 7.60, Ge/Xe = 0.245. The WimPyDD tungsten spin responses are a single-particle Gaussian exp(−z²/4) with s_n = −0.17, s_p = 0 (183W file docstring), *not* a shell-model table; a zero-momentum-transfer estimate Σ' ∝ Σ_iso f_iso (J+1)/J (⟨S_p⟩+⟨S_n⟩)² (recalled ⟨S⟩: ¹²⁹Xe 0.34, ¹³¹Xe −0.28, ¹⁸³W −0.17, ¹²⁷I 0.38, ⁷³Ge 0.47; likely/uncertain) gives W/Xe = 0.074, I/Xe = 1.75, Ge/Xe = 0.31. The two W/Xe estimates agree within ×1.5; the I/Xe estimates differ ×4 (q⁴Σ′ at 200–400 keV vs q = 0, Xe Σ′ nodes at 292/375 keV, P017). For the discrimination below only the fact W/Xe(L10) ≈ 0.05–0.08 ≪ 16 matters. L10 is also mass-flat (the shape at fixed A hardly moves for m ≥ 400 GeV, P012), so its ratios carry no δ or m dependence: a flat line in Fig. 1.

## 4. The δ-meter (Part 2)

### 4.1 Fisher information from count fractions
With N_T ~ Poisson(κ r_T(δ) ℰ_T) and κ profiled, the likelihood in δ reduces to the multinomial of the fractions p_T = r_Tℰ_T/Σr_Tℰ_T (κ̂ = N_tot/Σr_Tℰ_T). Fisher information I(δ) = N_tot Σ_T (∂p_T/∂δ)²/p_T with N_tot = κ̂_LZ(δ) Σ r_T ℰ_T at the LZ best fit; σ_δ = I^{−1/2}. Derivatives from the 1 keV grid (`np.gradient`). Element exposures: CaWO₄ × 0.6385, NaI × 0.8466.

Xenon 20 t·yr (5–1000 keV) with CaWO₄ ℰ (t·yr) (`delta_meter_sigma.csv`; μ_Xe, μ_W = expected counts):

| δ (keV) | μ_Xe | μ_W per 0.01/0.1/1 t·yr | σ_δ (keV) at 0.01 / 0.03 / 0.1 / 0.3 / 1 t·yr |
|---|---|---|---|
| 300 | 8.26 | 0.044 / 0.44 / 4.4 | 306 / 178 / 99 / 60 / 38 |
| 330 | 9.90 | 0.11 / 1.09 / 10.9 | 85 / 49 / 28 / 18 / 12 |
| 350 | 15.2 | 0.41 / 4.1 / 41 | 29 / 17 / 10.2 / 7.0 / 5.5 |
| 366 | 39.2 | 2.9 / 29 / 288 | 8.7 / 5.4 / 3.5 / 2.8 / 2.4 |
| 380 | 245 | 56 / 564 / 5 645 | 1.5 / 1.0 / 0.77 / 0.69 / 0.66 |

Xenon 10/30/50 t·yr change these by < 15 % once μ_W ≳ μ_Xe (the xenon count saturates the information; Fig. 2 right). With the LZ 270 keV xenon window (μ_Xe = 7.0 for every δ by construction) and CaWO₄ 0.1 t·yr: σ_δ = 84 / 6.8 / 2.6 keV at 300 / 350 / 366 keV.

Iodine (`delta_meter_iodine.csv`, xenon 20 t·yr): a background-free discriminating iodine target of 10 (30) t·yr NaI-equivalent holds 2.2 (6.5) events at every δ (I/Xe falls as κ̂ rises) and gives σ_δ = 198 (136) / 53 (36) / 24 (15) / 10 (6.3) / 2.1 (1.2) keV at 300/330/350/366/380 keV; adding it to Xe 20 + CaWO₄ 0.1 improves σ_δ only at δ ≤ 330 (99 → 85 keV at 300).

### 4.2 Comparison with spectral-shape information
Per-event Fisher information on δ from the normalised observed-energy pdf (Gaussian smearing: Xe σ_E = 11√(E/248) keV, P009/P021; W and I 1 % of E, floor 2 keV — an *assumption* for a cryogenic calorimeter, flagged) (`shape_fisher.csv`): σ_δ per single event = 87.6 / 38.4 / 25.5 / 17.9 / 9.1 keV in Xe(LZ270), 79 / 35 / 23 / 21 / 29 keV in Xe(5–1000), 76 / 50 / 41 / 40 / 42 keV in W(10–1300), 72 / 27 / 15 / 11 / 9 keV in I(5–1000) at δ = 300/330/350/366/380 keV. So N xenon events yield σ_δ ≈ 25/√N keV at 350 keV from shape alone, while the ratio meter with 0.1 t·yr CaWO₄ yields 10 keV — comparable to a 6-event xenon shape fit; the ratio wins by ×3–10 for δ ≥ 350 and loses at δ = 300 (0.44 W events).

### 4.3 Asimov 3σ / 5σ separations (κ profiled; `separation_exposures.csv`)
q = 2Σ_T[μ_T⁰(κ̂₀) − μ_T¹ + μ_T¹ ln(μ_T¹/μ_T⁰)], Z = √q, data = expectation under the true hypothesis at LZ's best fit, H₀ tested with κ̂₀ = N_tot/Σr_T⁰ℰ_T. Xenon 20 t·yr, 5–1000 keV; CaWO₄ exposure for Z = 3 (5):

| true → tested | Z at 0.01 / 0.1 / 1 t·yr CaWO₄ | ℰ_CaWO₄ for 3σ (5σ) |
|---|---|---|
| δ = 300 → 380 | 1.7 / 4.1 / 6.0 | 0.038 (0.25) t·yr |
| δ = 380 → 300 | 17 / 49 / 101 | 0.29 (0.79) kg·yr |
| δ = 300 → 350 | 0.46 / 1.4 / 2.9 | 1.33 t·yr (never below 1000 t·yr) |
| δ = 350 → 300 | 0.82 / 2.5 / 5.9 | 0.15 (0.58) t·yr |
| δ = 350 → 380 | 2.0 / 4.4 / 5.9 | 0.028 (0.20) t·yr |
| δ = 366 → 330 | 2.4 / 7.0 / 13.8 | 0.015 (0.046) t·yr |
| δ = 330 → 366 | 0.89 / 2.4 / 4.0 | 0.21 t·yr (5σ never) |
| inelastic 300 → L10 | 0.65 / 2.1 / 6.3 | 0.215 (0.61) t·yr |
| inelastic 350 → L10 | 2.3 / 7.2 / 21.6 | 0.017 (0.047) t·yr |
| inelastic 366 → L10 | 6.5 / 20 / 59 | 2.1 (5.8) kg·yr |
| inelastic 380 → L10 | 31 / 94 / 266 | 0.09 (0.26) kg·yr |
| L10 → inelastic 300 | 0.40 / 1.3 / 3.6 | 0.64 (2.4) t·yr |
| L10 → inelastic 366 | 1.5 / 4.2 / 8.2 | 0.045 (0.17) t·yr |

The asymmetry is the Poisson asymmetry of "seeing few events where many are predicted" versus "seeing many where few are predicted". Xenon 10 t·yr: 300 → 380 needs 0.058 t·yr, 350 → 300 0.18 t·yr; xenon 50 t·yr: 0.030 and 0.14 t·yr. With the zero-q spin estimate for L10 W/Xe (0.074 instead of 0.050) the inelastic-vs-L10 exposures change by ≤ 10 %. With the LZ 270 keV xenon window the δ-pair separations need 10–30× less CaWO₄ (300 → 380: 1.1 kg·yr) because the ROI edge itself suppresses the high-δ xenon rate — but this leans on the acceptance edge that P038 shows manufactures the δ ≈ 380 preference.

### 4.4 Systematics as δ biases (`delta_meter_systematics.csv`, `shape_delta_meter_bias.csv`)
Reading a measured W/Xe with the SHM/1 TeV calibration when the truth differs:

| true δ | v_esc 528 | v_esc 560 | v₀ 220 | v₀ 250 | June-only data | W form factor ×2 | isovector (W ×1.04) | m = 400 GeV | m = 4 TeV |
|---|---|---|---|---|---|---|---|---|---|
| 300 | 311 | 290 | 318 | 286 | 290 | (> 395: n/a) | 302 | 338 | 280 |
| 330 | 342 | 319 | 347 | 319 | 321 | 302 (×0.5) | 331 | 385 | 311 |
| 350 | 364 | 338 | 368 | 339 | 341 | 362 (×2), 335 (×0.5) | 351 | Xe blind | 329 |
| 366 | 382 | 352 | 385 | 354 | 357 | 355 (×0.5) | 367 | Xe blind | 341 |
| 380 | (>395) | 365 | (>395) | 367 | 371 | 372 (×0.5) | 380 | Xe blind | 351 |

Escape velocity ±16 km/s biases the ratio meter by +14/−12 keV (350) and +16/−14 keV (366); v₀ ±12 km/s by ∓13–19 keV; the ×2 form-factor uncertainty P015 assigns to the Helm tungsten M response by ±12 keV; the isospin structure by ≤ 1 keV; using June-only data with an annual calibration by −9 keV. The **mass degeneracy is the largest**: at fixed δ the ratio depends on μ_W/μ_Xe, so a 4 TeV WIMP reads 20–29 keV low and a 400 GeV WIMP 38–55 keV high; for m → ∞ the ratio converges (12.5 at 4 TeV vs 15.9 at 1 TeV at δ = 300). The shape δ-meter (KL projection of the variant pdf onto SHM/1 TeV templates) has the same halo floor — v_esc ±16 → +9/−8 keV (W) and +11/−9 keV (Xe) at 366 keV; v₀ ±12 → +11/−7 keV; m = 4 TeV → −7 (W), −10 keV (Xe) — but a 3× smaller mass bias. The template check (SHM read with SHM) returns the input to 0.0 keV. Conclusion: below ≈ 15 keV the reading is astrophysics- and mass-limited; the ratio and shape meters share the halo floor but have different mass responses, so their *combination* constrains m.

## 5. The endpoint (Part 3; `endpoint.json`, `W_edges.csv`)

Kinematics (E₊ = (μ²v²/m_N)[1 − δ/(μv²) + √(1 − 2δ/(μv²))]; WimPyDD masses, v_max(June) = 810.9 km/s, δ = 366 keV): E₊ = 1081/1098/1106/1115/**1131** keV for ¹⁸⁰⁻¹⁸⁶W (the natural-W endpoint is ¹⁸⁶W's), 525 keV for ¹²⁷I, 647 keV for ¹³⁶Xe; E₋(¹⁸⁶W) = 86 keV. dE₊/dδ = −2.35 keV/keV (W) and −3.57 (Xe); dE₊/dv_max = 4.91 keV per km/s, so v_esc ±16 km/s moves E₊ by ∓78 keV, i.e. a δ bias of ±33 keV if δ were read from E₊; m_χ = 400/4000 GeV gives E₊ = 618/1547 keV (μ_W = 121/148/166 GeV) — a ±180–220 keV-equivalent δ shift: **E₊ fixes δ only given m and the halo**, as the assignment anticipated, and much more weakly than the ratio.

But the endpoint is not observable. The annual-averaged W spectrum at 366 keV (Fig. 3) has 0.39 % of its rate above 500 keV, 1.5×10⁻⁴ above 700, 1.2×10⁻⁷ above 1000 and 1.5×10⁻⁸ in the last 100 keV before E₊: the Helm form factor (nodes at ≈ 145, 300, 470, 780 keV for W) and the collapsing velocity tail empty the MeV region. The resolution-smeared 90/99 % quantiles are E₉₀ = 241/258/273/304/359 keV and E₉₉ = 410/430/442/451/458 keV at δ = 300/330/350/366/380 keV (dE₉₉/dδ = 0.5–0.9), pinned by the ≈ 470 keV node and insensitive to the halo (E₉₉ = 449.6 for v_esc 560, 450.1 June) and mass (411/466 keV at 400/4000 GeV). The maximum recoil of N events (order statistic F^N) has median 253/374/424/465/596/654/690 keV for N = 3/10/30/100/300/1000/3000, always > 400 keV below E₊; no N ≤ 20 000 brings the 84 % quantile within 50 keV. The Fisher information of the *conditional* spectrum above 300 keV is 1.7–5.4×10⁻⁷ per event (σ_δ ≈ 1400–2400 keV per event): the falling edge carries no δ information; what the tungsten shape measures is the onset E₋(δ) = 51 → 95 keV and the lobe populations (peak jumps 108 → 215 keV between δ = 330 and 350). Full-shape Fisher I₁(W, 366) = 6.35×10⁻⁴ (6.59×10⁻⁴ without resolution): σ_δ = 40 keV/√N — 3.5 events for the 21 keV that ±50 keV on E₊ would correspond to, 16 events for ±10 keV, 63 for ±5 keV. Toy MLE (300 toys each, δ grid 330–385 keV, parabolic refinement): mean 366.7/366.0/365.7, std 7.8/4.4/2.3 keV for N = 30/100/300 vs Fisher 7.3/4.0/2.3 — the asymptotic formula is adequate at N ≥ 30. Iodine: I₁ = 8.7×10⁻³ (σ_δ = 10.7 keV per event; E₊ = 525 keV, again form-factor-limited).

Design consequence: a tungsten ROI to **≈ 700 keV** loses < 2×10⁻⁴ of the signal at any δ ≥ 300 keV (1.2×10⁻³ above 500 keV at δ = 300); the "1.1–1.3 MeV" window of P015 is a kinematic ceiling that the form factor never lets the rate reach. The upper edge can therefore stay below the ¹⁸⁰W α line (§6).

## 6. Backgrounds in a CaWO₄ W-band search (Part 4; `backgrounds.json`; recalled inputs flagged)

- **Neutrons.** E_R,max(W) = 4m_nM_W/(m_n+M_W)² E_n = 21.7 keV/MeV: 108/217/434/1085 keV for E_n = 5/10/20/50 MeV; oxygen recoils reach 223 keV/MeV. Radiogenic (α,n) and fission neutrons (≲ 10 MeV; P013) cannot put a tungsten recoil above 220 keV, and W recoils in 100–220 keV need E_n > 4.6 MeV. A W-band threshold of 220 keV is radiogenic-neutron-free but keeps only 18/39/48/55 % of the signal at δ = 300/350/366/380 (`W_threshold_acceptance`); 110 keV keeps 59/90/97/99.6 %. Muon-induced neutrons (flux above 10 MeV ~10⁻¹⁰ cm⁻²s⁻¹ at ~3600 m.w.e., recalled, uncertain ×3; n–W elastic ~3 b, uncertain ×2) give ≈ 20 W scatters per t·yr of CaWO₄ before any veto, ~75 % above 100 keV for a 20 MeV neutron; reaching < 0.1 per t·yr needs a ≥ 99 % muon veto plus multi-crystal coincidence rejection, or a deeper site. Oxygen and calcium recoils are separated from the W band by light yield (recalled CRESST quenching: O ≈ 0.11, Ca ≈ 0.06, W ≈ 0.02–0.04 of e/γ; likely).
- **α and Pb recoils.** ²¹⁰Po surface decays give ²⁰⁶Pb recoils of 103 keV (certain) with W-like light yield — the CRESST "Pb-recoil" band; a 110 keV W threshold removes them at a signal cost of 41 % (δ = 300) to 0.4 % (380). ¹⁸⁰W (0.12 %, T½ = 1.8×10¹⁸ yr, Q = 2.516 MeV; likely) decays at 2.65 per kg·day (967 per kg·yr) — a 2.5 MeV line in the α band (¹⁷⁶Hf recoil 56 keV, below threshold if the α escapes); degraded surface α's fall into 0.1–1.3 MeV with α light yield (≈ 0.22 of e/γ, recalled). With a ROI ending at 700 keV the line is 1.8 MeV away; the α/W-band light-yield separation at 0.1–0.7 MeV, where photon statistics are ≥ 10× better than at CRESST's 10–40 keV, is a solved problem (qualitative; flagged).
- **e/γ.** Cosmogenic and intrinsic β/γ at 0.1–1 MeV are rejected by light yield by construction of the W band (LY ≈ 0.02 vs 1).
- **Neutrinos.** Atmospheric CEνNS on W above 100 keV is negligible (coherence lost, P019).
- **Iodine.** NaI(Tl) has no NR discrimination (P028): 30 t·yr would hold 6–7 signal events among ~10⁹ counts. A cryogenic scintillating NaI/CsI bolometer (COSINUS-like; CsI is fully active since Cs A = 133 is xenon-like) with zero background needs 10 t·yr for 2.2 events and the σ_δ of §4.1 — not a realistic decade-scale programme.

## 7. Programme table (Part 5; `programme_table.csv`; expected events at LZ's best fit, κ̂(δ))

| Target / exposure / window | δ = 300 | 350 | 366 | 380 | L10 | Decides | Timescale |
|---|---|---|---|---|---|---|---|
| Xenon world, existing 7.5 t·yr, 270 keV ROI | 2.63 | 2.63 | 2.63 | 2.63 | 2.63 | rate reality (P(0) = 7 %… P005: 22 % for XENONnT+PandaX alone) | now |
| Xenon 20 t·yr, 270 keV ROI | 7.0 | 7.0 | 7.0 | 7.0 | 7.0 | rate ×7; δ only from event energies | ~2030 |
| Xenon 20 t·yr, 5–1000 keV | 8.3 | 15.2 | 39 | 245 | 15.7 | opens the >270 keV lobe; shape δ-meter 25/√N keV | ~2030 |
| Xenon 50 t·yr, 5–1000 keV | 20.7 | 38 | 98 | 612 | 39 | coupling anchor | 2030s |
| CaWO₄ 10 kg·yr, W 10–1300 keV | 0.044 | 0.41 | 2.9 | 56 | 2.5e−4 | δ ≥ 366 seen or excluded; 3σ vs L10 for δ ≥ 350 | 2–3 yr |
| CaWO₄ 100 kg·yr | 0.44 | 4.1 | 29 | 564 | 2.5e−3 | δ ≥ 350 decided; σ_δ = 10/3.5/0.8 keV | ~5 yr |
| CaWO₄ 1 t·yr | 4.4 | 41 | 288 | 5 645 | 0.025 | δ = 300 confirmed (4 events); 300 vs 380 at 6σ; inelastic vs L10 at 6σ | ~10 yr |
| CaWO₄ 1 t·yr, 110–1300 keV | 2.6 | 37 | 279 | 5 620 | 6e−5 | same, Pb-recoil-safe | ~10 yr |
| NaI(Tl) 30 t·yr | 6.5 | 5.9 | 7.4 | 7.4 | 159 | nothing (no discrimination) | — |
| Cryogenic iodine 10 t·yr, zero bkg | 2.2 | 2.0 | 2.5 | 2.5 | 53 | weak second δ-meter | > 2040 |
| Germanium 1 t·yr | 0 | 0 | 0 | 0 | 0.20 | null control: any >100 keV NR population falsifies δ > 232 keV | exists (LEGEND-class) |

(The xenon 270 keV rows are δ-independent by construction of κ̂. L10 xenon counts use the same efficiency; the L10 iodine count is WimPyDD's shell-model ¹²⁷I Σ′ and is ×4 above the zero-q estimate.)

## 8. Figures
- `figures/P046_fig1_ratios_vs_delta.png` — W/Xe and I/Xe rate ratios vs δ (annual halo; v_esc 528–560 band; W/Xe with the LZ 270 keV xenon ROI dashed); L10 ratios as flat dotted lines (WimPyDD and zero-q spin estimate).
- `figures/P046_fig2_sigma_delta_vs_exposure.png` — σ_δ from the W/Xe count ratio (coupling profiled) vs CaWO₄ exposure at xenon 20 t·yr (left) and vs xenon exposure at CaWO₄ 0.1 t·yr (right), δ = 300–380 keV; grey band = halo + form-factor floor (~15 keV).
- `figures/P046_fig3_W_spectrum_endpoint.png` — tungsten dR/dE at δ = 366 keV (June, annual, December; 1 % resolution) and xenon, LZ-best-fit normalisation, with E₊(June, ¹⁸⁶W) = 1131 keV and the LZ 270 keV edge: the Helm nodes and the tail leave nothing above ≈ 600 keV.
Palette validated with the dataviz `validate_palette.js` (light mode) — see run output in the final report.

## 9. Failed or abandoned approaches
- First implementation interpolated 220-point log-grid kernels and used c = 299 792 km/s and a non-strict step: rates disagreed with direct WimPyDD calls by 5–23 % at kinematic edges and Helm nodes; fixed by evaluating kernels on the 1 keV grid and matching WimPyDD's c = 3×10⁵ km/s and strict inequality (§2).
- A direct order-statistic "endpoint" estimator (max observed energy) never converges to E₊ (§5); abandoned in favour of the full-shape Fisher/MLE, with the conditional above-300 keV Fisher reported to show why.
- L10 on tungsten from WimPyDD's Gaussian spin response is kept only as an order-of-magnitude number, bracketed by the zero-q estimate.

## 10. Extended discussion
The δ-meter is statistically superb for δ ≥ 350 keV (σ_δ ≈ 10 keV with 100 kg·yr of CaWO₄, 1 keV at 380 keV) but its *calibration* is astrophysical and nuclear: v_esc, v₀, the W form factor and above all the WIMP mass each move the reading by 10–30 keV, so a 5 keV statistical error is not a 5 keV measurement. The complementary xenon shape meter shares the halo floor and has a smaller mass bias; the two meters together, and the annual modulation (P034), are what would turn "confirmation" into a measurement of (δ, m). The endpoint question has a clean negative answer: neither tungsten nor iodine can see E₊ because the nuclear form factor empties the last ≈ 60 % of the kinematic window; the observable edges (E₉₉ ≈ 410–460 keV in W) are nuclear-structure features, not kinematic ones, which also means the tungsten upper edge can be set at 700 keV. For a δ = 300 keV splitting — the value LZ's own grid and P021's uniform-in-κ weighting favour — tungsten is slow (4 events per t·yr of CaWO₄), the ratio is only a factor 16 and the meter has σ_δ ≈ 40 keV after 1 t·yr; there the extended-window xenon shape and a t·yr-scale CaWO₄ detector are equally necessary and a 3σ 300-vs-350 separation costs 1.3 t·yr.

## 11. References
LZ Collaboration, arXiv:2609.02823 (2026). D. Tucker-Smith, N. Weiner, PRD 64, 043502 (2001). N. Anand, A. L. Fitzpatrick, W. C. Haxton, PRC 89, 065501 (2014). I. Jeong, S. Kang, S. Scopel, G. Tomar, CPC 276, 108342 (2022) (WimPyDD). D. Baxter et al., EPJC 81, 907 (2021). A. H. Abdelhameed et al. (CRESST), PRD 100, 102002 (2019). G. Cowan, K. Cranmer, E. Gross, O. Vitells, EPJC 71, 1554 (2011). Corpus: dossier 00, P003, P005, P009, P012, P013, P015, P017, P018, P019, P021, P028, P034, P035, P038.

## 12. Tools and provenance (mirrors `provenance/P046.json`)
Agent tools: Read ×20 (PAPER_GUIDE; dossier; ledger in three parts; P015, P028, P005, P035, P034, P018, P021, P038, P017 papers; lzcommon.py; P015 script; P005 and P012 scripts (L10 blocks); the three P046 figures), Bash ×14 (file listings and P015 tables; WimPyDD probes: element attributes, timing, kernel test, response-file and diff_rate/get_vmin inspection; three script runs; table extraction; node validator), Write ×2 (script, details), Edit ×10 (script), Skill ×1 (dataviz). Software: python 3.12.13; WimPyDD 2.0.4 (eft_hamiltonian incl. q-dependent Wilson coefficients, streamed_halo_function with explicit v_min grid, diff_rate with isotopes_list and single-stream halos); numpy 2.5.3; scipy 1.18.1 (special.erf/erfc, optimize.brentq, stats); matplotlib 3.11.2; common/lzcommon.py (wd, wd_halo, wd_hamiltonian, wd_rate, E_R_range_keV, LZ constants); node v26.4.0 for the palette validator only. WimPyDD-generated files: none (single-stream `diff_rate` writes no response files). Recalled knowledge: 14 items listed in the JSON. Datasets: none. Data requests: none.
