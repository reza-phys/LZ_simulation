# P087 — The atmospheric-neutrino floor in the extended window: CEνNS rates, spectral shape and uncertainty at 55–420 keV in xenon, and whether it limits the decisive tests

Simulated date 2026-09-16 · hep-ex (cross-list hep-ph) · neutrino-background / next-generation-detector group · category XEXP · builds on P060 (CEνNS/Xe form-factor machinery, DSNB spectra), P019 (atmospheric CEνNS tail), P050 (exposures, spectra cache, efficiencies), P069 (xenon world), P034 (modulation), P016/P038 (NR-band backgrounds, 1000 phd edge).

All numbers below are produced by `output/code/P087_atmospheric_floor.py` (15 s; results in `P087_results.json`, tables `P087_*.csv`, log `run_log.txt`) unless marked [recalled], [paper] or [corpus].

## 1. Motivation and framework

LZ's background model (Table I [paper], `fulltext.tex` l.234–235) lists two neutrino components in the 220 d × 4.71 t *science* sample: "Atmospheric ν (1.1 ± 0.2) × 10⁻¹" and "⁸B+hep ν (5.7 ± 0.6) × 10⁻²" expected events (fit results identical), with (1.1 ± 0.2) × 10⁻⁵ / (5.7 ± 0.6) × 10⁻⁶ in the *prompt-veto* sample and (3.3 ± 0.7) × 10⁻³ / (1.7 ± 0.2) × 10⁻³ in the *delayed-veto* sample (l.529–530, 574–575). The atmospheric spectrum "is calculated using the formalism of Ref. [Billard 2014], and flux normalizations and uncertainties follow the standard values recommended in Ref. [Baxter 2021]" (l.176). The Discussion (l.313–314) states that "while the flux of atmospheric neutrinos carries some uncertainty, the coherent scattering neutrino interaction is well understood, producing an approximately exponentially-decaying recoil spectrum with increasing energy. In addition, the reconstructed energy of the event is not consistent with scattering coherence."

The corpus has proposed decisive tests of the 248 keV hint that live in the 200–270 keV window and its 1000-phd extension to ≈ 420 keV: annual modulation (P034: 5–97 events, 15–274 t·yr), next-generation exposures of 10–100 t·yr (P050), and a pre-registered reanalysis of ≈ 19 t·yr of untouched xenon data (P069). These are counting experiments at 10⁻³–10⁻⁴ events per t·yr of background; the only irreducible physics background that produces single-site nuclear recoils at these energies is atmospheric-neutrino CEνNS. This paper computes its rate, spectral shape and uncertainty in the extended window and asks whether it — the "neutrino floor" of the high-energy search — limits those tests.

## 2. Inputs

| Input | Value | Source |
|---|---|---|
| Exposure, FV, live time | 2.84 t·yr, 4.71 t, 220 d | LZ paper (lzcommon `LZ`) |
| Table I atmospheric ν, ⁸B+hep in ROI | 0.11 ± 0.02, 0.057 ± 0.006 | LZ Table I [paper] |
| LZ efficiency (true E_R) | plateau 0.955; erf roll-off 50 % at 269.9 keV, σ = 11.5 keV; erf turn-on 50 % at 5.4 keV, σ fitted (§4) | Fig. S2 via P021/P050; turn-on calibrated here |
| Extended 1000 phd efficiency | 50 % at 423.2 keV, σ = 18 keV | P038 |
| Resolution | σ_E = 11 √(E/248) keV | P009/P021 |
| G_F = 1.1664 × 10⁻⁵ GeV⁻², sin²θ_W = 0.23867 | PDG [recalled, certain / likely] | as P019/P060 |
| Q_W = N − (1 − 4 sin²θ_W) Z; Q_W(¹³¹Xe) = 74.55 | [recalled, certain] | |
| Xe isotopes, m_A = A u, nuclei/t = 4.584 × 10²⁷ | lzcommon | |
| Helm F² (Lewin–Smith: s = 0.9 fm, a = 0.52, c = 1.23 A^{1/3} − 0.60) | lzcommon form, re-implemented with a shift Δc | [recalled, certain] |
| Neutron skin of Xe ≈ 0.15 fm (Helm variant Δc = +0.15 fm) | [recalled, likely] | |
| Shell-model weak form factor | WimPyDD 2.0.4 O₁ M-response with c_p = −(1 − 4 sin²θ_W), c_n = 1 (§3.2) | lzcommon `wd_hamiltonian`, `wd_rate` |
| Atmospheric flux normalisation Φ(> 13 MeV) = 10.5 cm⁻² s⁻¹, ±20–25 % | Billard et al. 2014 Table I / Baxter et al. 2021 [recalled, likely] | |
| Flux shape: E dΦ/dE flat (γ = 1.0) 13–100 MeV, γ ≈ 2.3 at 0.2–1 GeV, γ ≈ 3.0 above 2 GeV | Battistoni et al. 2005 (FLUKA, low energy), Honda et al. 2011 [recalled, likely] | |
| All-flavour dΦ/dE at 1 GeV ≈ 1.5–2.5 × 10⁻⁴ cm⁻² s⁻¹ MeV⁻¹ | Honda 2011 / Gaisser–Honda 2002 [recalled, likely ×1.5] | consistency target |
| SURF vs Kamioka/LNGS low-energy flux difference ≈ 10–15 % | [recalled, uncertain] | inside the ±25 % band |
| ⁸B flux 5.25 × 10⁶ cm⁻² s⁻¹, ⟨E⟩ 6.7 MeV, end point ≈ 16.5 MeV | SNO/B16 [recalled, certain to 5 % / likely] | Keil–Raffelt α = 2.5 proxy shape |
| hep flux 8.0 × 10³ cm⁻² s⁻¹, end point 18.77 MeV | [recalled, likely] | |
| DSNB 30 cm⁻² s⁻¹ all flavours, ⟨E⟩ 9/11/13 (central) and 12/15/25 MeV (very hard) | P060 [recalled, uncertain ×3] | |
| Signal spectra: isoscalar O₁, 1 TeV, δ = 300/350/366 keV, L10; Anand unit coupling c⁰ = 1/m_v², 12-day annual mean | P050 cache `P050_spectra_cache.npz` (WimPyDD 2.0.4), verified here | |
| NR-band backgrounds 200–270 keV: 5.7 × 10⁻⁴ per 2.84 t·yr; extension bins | P016, P050 `P050_background_bins.csv` [corpus] | |
| P034 modulation amplitudes a₁ = 0.428/1.179/1.521/1.656 (δ = 300/350/366/380) | P034 [corpus] | |
| Atmospheric seasonal modulation ≲ 2 % (1–4 %) of the sub-GeV flux; solar-cycle 5–10 % over 11 yr | [recalled, uncertain] | |

## 3. Method

### 3.1 CEνNS spectrum

dσ/dE_R = (G_F² m_A/4π) Q_W² (1 − m_A E_R/2E_ν²) F²(E_R); for each isotope A,

dR/dE_R = n_A (G_F² m_A/4π) Q_W² F_A²(E_R) [I₀(E_min) − (m_A E_R/2) I₂(E_min)],  I₀(x) = ∫_x^∞ Φ dE, I₂(x) = ∫_x^∞ Φ/E² dE,

with E_min = ½[E_R + √(E_R² + 2 m_A E_R)] (58.0/87.5/110.7/123.3/128.7/160.5 MeV for 55/125/200/248/270/420 keV). The cumulative integrals are tabulated once on a 6000-point log grid (1 MeV–100 GeV), which makes the spectrum a vectorised sum over isotopes on a 0.5–600 keV grid. For E_ν ≫ E_min the kinematic factor is ≈ 1 and dR/dE_R ∝ F²(E_R) Φ(> E_min): above 55 keV the *shape* is the form factor and the *normalisation* is the integral flux above 60–160 MeV.

### 3.2 Form factors

(i) Helm (natural Xe, Q_W²- and abundance-weighted): nodes at 94 and 279 keV; F²(248) = 1.63 × 10⁻⁴; F²(30/100/200/350/420) = 0.204/2.5 × 10⁻⁴/1.5 × 10⁻³/1.7 × 10⁻⁴/1.3 × 10⁻⁴. (ii) Helm with a 0.15 fm larger box radius (neutron-skin proxy): nodes 90/265 keV, F²(248) = 5.7 × 10⁻⁵. (iii) Shell model: WimPyDD's O₁ rate for a 10⁶ GeV WIMP with couplings c_p = −(1 − 4 sin²θ_W), c_n = 1 (Sun-frame halo), divided by Σ_A n_A m_A Q_W² F²_Helm η(v_min) with the velocity integral η evaluated explicitly (P019 left η inside the ratio; at 300 keV and m = 10⁶ GeV v_min = 330 km/s, a 30–40 % effect), normalised to 1 at 1 keV. Ratio shell/Helm: 0.98/0.93/0.83/0.38/0.19/0.037/2.03/0.56/0.28 at 5/20/50/100/200/248/300/350/400 keV. Shell nodes 97/245 keV; F²(248) = 6.1 × 10⁻⁶ (27× below Helm), F²(350) = 9.6 × 10⁻⁵ (Fig. 4). P019 quoted 2.6 × 10⁻⁶ at 248 keV; the difference is the η correction. These three models bracket the nuclear uncertainty of the weak (neutron-dominated) form factor at q ≈ 200–300 MeV, where no CEνNS data exist.

### 3.3 Flux parametrisation (recalled) and variants

dΦ/dE = K (E/E₁)^{−γ_a} [1 + (E/E₁)ⁿ]^{−(γ_b−γ_a)/n} [1 + (E/E₂)ⁿ]^{−(γ_c−γ_b)/n}, n = 2, all four species summed (CEνNS is flavour-blind), zero below 13 MeV. Central: γ_a = 1.0, E₁ = 120 MeV, γ_b = 2.3, E₂ = 2 GeV, γ_c = 3.0, normalised to 10.5 cm⁻² s⁻¹. It gives dΦ/dE = 0.129/0.0286/2.35 × 10⁻⁴ cm⁻² s⁻¹ MeV⁻¹ at 30/100/1000 MeV and Φ(> 110 MeV) = 2.71 cm⁻² s⁻¹. Eleven variants (`P087_flux_variants.csv`): γ_a = 0.8/1.2, E₁ = 90/160 MeV, γ_b = 2.0/2.6, γ_c = 2.7/3.3, and P019's broken power laws (γ = 1 to 100 MeV, then 2.5/2.0/3.0). Φ(> 110 MeV) spans 1.54–3.25 cm⁻² s⁻¹ (×2.1) and dΦ/dE(1 GeV) 3.7 × 10⁻⁵–4.8 × 10⁻⁴.

### 3.4 Efficiency turn-on calibrated on LZ's ⁸B+hep number

All ⁸B and hep recoils lie below E_R,max = 4.45 keV (16.5 MeV) and 5.76 keV (18.77 MeV), so LZ's 0.057 ⁸B+hep events measure the acceptance *below* the 5.4 keV 50 % point. With the raw ⁸B rate 356 per t·yr (1011 per 2.84 t·yr), an erf turn-on of width σ_lo = 2.5 keV (P050's choice) would give 45 events, σ_lo = 1.0 keV 0.39 events, and Table I's 0.057 is reproduced for σ_lo = 0.74 keV (0.72–0.75 for ±0.006). We adopt σ_lo = 0.74 keV. It affects only the sub-threshold part of the atmospheric normalisation (efficiency-weighted atmospheric count below 5.4 keV: 9.4 × 10⁻⁴ per t·yr, 2.4 % of the ROI count); nothing above 55 keV depends on it. Caveat: the ⁸B tail above 12 MeV, which makes the 3–4.4 keV recoils, is approximated by a Keil–Raffelt shape, so σ_lo is uncertain by ~0.1 keV.

### 3.5 Normalisation and anchoring

The recalled flux predicts N_ROI = 0.081 events in 2.84 t·yr with the LZ efficiency (0.067–0.089 over the shape variants; 0.079 shell, 0.079 skin), versus Table I's 0.11 ± 0.02: a pull of −1.0σ including the 25 % flux systematic. We therefore anchor every atmospheric number to Table I, scaling the central flux by 1.356 (implied Φ_total = 14.2 cm⁻² s⁻¹, Φ(> 110 MeV) = 3.68, dΦ/dE(1 GeV) = 3.2 × 10⁻⁴, at the upper edge of the recalled 1.5–2.5 × 10⁻⁴; the softer variants γ_b = 2.6 and P019-central give 2.0 and 1.7 × 10⁻⁴). Each shape variant is anchored separately, so the shape spread quoted below is the uncertainty in the ≥ 100 MeV flux *given* LZ's 20–60 MeV normalisation. This is the same procedure as P019 (anchor 1.5 for its broken shape; ours 1.48 for that shape).

### 3.6 Discovery limit and systematic floor

Billard-style: the coupling at which 90 % of experiments obtain a 3σ excess. With b = b_rate × ε expected background and a 25 % log-normal systematic marginalised (40-point Gauss–Hermite), n_crit is the smallest count with P(N ≥ n_crit | b) < 1.35 × 10⁻³, and s_DL solves P(N ≥ n_crit | s + b) = 0.9. For b → 0, n_crit = 1 and s_DL = 2.30 events; when b reaches 1.35 × 10⁻³ (with the systematic, 1.32 × 10⁻³) n_crit → 2 and s_DL → 3.89 (×1.69), etc. Coupling² κ_DL = s_DL/(s_unit ε), with s_unit the efficiency-weighted count per t·yr at Anand unit coupling (WimPyDD c⁰ = 2/m_v², `lz.wd_c_from_anand`); σ_n = κ μ_n²/(π m_v⁴) = κ × 2.964 × 10⁻³⁸ cm² at 1 TeV. LZ's best fit is κ_LZ = 1/(2.84 × N_ROI,unit) = 7.38 × 10⁻⁵/3.75 × 10⁻³/4.57 × 10⁻²/0.301 (δ = 300/350/366 keV; L10), identical to P050's normalisation table; σ_n = 2.19 × 10⁻⁴²/1.11 × 10⁻⁴⁰/1.36 × 10⁻³⁹ cm². Systematic floor: the exposure-independent coupling below which Z = s/σ_b can never reach 3, κ_floor = 3 f_sys b/s_unit with f_sys = 0.25 (normalisation) or the full shape+form-factor bracket (b_max − b_min)/b.

### 3.7 Shape separation and modulation

Signal and background spectra are efficiency-weighted, smeared with σ_E = 11√(E/248) keV, restricted to the window and normalised. With ℓ = ln(p_s/p_b): KL(s‖b) = E_s[ℓ], KL(b‖s) = −E_b[ℓ], and the Gaussian requirement N_3σ = 9 Var_b[ℓ]/(E_s[ℓ] − E_b[ℓ])² (P006/P034 convention; also quoted with Var_s). Modulation: cosine rule Z = a√(N/2) → N_3σ = 18/a².

## 4. Results

### 4.1 Validation

- P050 cache vs fresh WimPyDD (δ = 350 keV, 201/249/300/351 keV): ratios 1.005/1.005/0.995/1.000.
- Shell/Helm ratio → 0.98 at 5 keV (both → 1); κ_LZ reproduces P050 to < 0.1 %.
- P019's broken central shape in our machinery: N(200–270) = 2.25 × 10⁻⁵ per t·yr = 6.4 × 10⁻⁵ per 2.84 t·yr vs P019's 6.5 × 10⁻⁵ (P019 used a σ = 11.8 keV Gaussian roll-off). Our central shape gives 8.9 × 10⁻⁵ (×1.4, more flux at 150 MeV–1 GeV).
- Above 230 keV (≈ S1c > 500 phd) we get 1.65 × 10⁻⁵ per 2.84 t·yr vs LZ's digitised Fig. 5 green curve 3.4 × 10⁻⁵ (P019; that curve includes a fitted neutron component).
- DSNB in ROI: 4.2 × 10⁻³ (central) to 0.040 (very hard) per 2.84 t·yr vs P060's 3.6 × 10⁻³–0.037.
- Galactic ⁸B raw rate 356 per t·yr (all E_R) — the familiar ~ 10²–10³ per t·yr scale of the solar-neutrino floor [recalled, likely].

### 4.2 Atmospheric CEνNS in the bins (anchored to Table I; `P087_atm_bin_table.csv`)

| bin [keV] | efficiency | Helm, per t·yr | per 2.84 t·yr | shape range per t·yr | Helm + skin | shell model | raw (no eff.) | 1 event at [t·yr] Helm / shell |
|---|---|---|---|---|---|---|---|---|
| 5.4–55 | LZ 600 phd | 3.69 × 10⁻² | 0.105 | 3.68–3.70 × 10⁻² | 3.69 × 10⁻² | 3.71 × 10⁻² | 3.95 × 10⁻² | 27 / 27 |
| 55–125 | LZ | 6.76 × 10⁻⁴ | 1.92 × 10⁻³ | 5.6–7.3 × 10⁻⁴ | 5.6 × 10⁻⁴ | 5.8 × 10⁻⁴ | 7.1 × 10⁻⁴ | 1.5 × 10³ / 1.7 × 10³ |
| 125–200 | LZ | 2.25 × 10⁻⁴ | 6.39 × 10⁻⁴ | 1.3–2.6 × 10⁻⁴ | 2.3 × 10⁻⁴ | 8.0 × 10⁻⁵ | 2.4 × 10⁻⁴ | 4.4 × 10³ / 1.3 × 10⁴ |
| **200–270** | LZ | **3.14 × 10⁻⁵** | **8.93 × 10⁻⁵** | 1.6–3.7 × 10⁻⁵ | 2.1 × 10⁻⁵ | **4.3 × 10⁻⁶** | 3.3 × 10⁻⁵ | **3.2 × 10⁴ (2.7 × 10⁴ hardest) / 2.3 × 10⁵** |
| 270–420 | 1000 phd | 1.15 × 10⁻⁵ | 3.28 × 10⁻⁵ | 4.8–14 × 10⁻⁶ | 1.5 × 10⁻⁵ | 8.3 × 10⁻⁶ | 1.3 × 10⁻⁵ | 8.7 × 10⁴ / 1.2 × 10⁵ |

Below 5.4 keV (efficiency-weighted): 9.4 × 10⁻⁴ per t·yr; total raw rate 0.045 per t·yr; dR/dE_R(248 keV) = 1.42 × 10⁻⁷ per t·yr per keV (P019: 1.0 × 10⁻⁷). The 200–270 keV count is 16 % of P016's total NR-band background of 5.7 × 10⁻⁴ per 2.84 t·yr (accidentals 1.5 × 10⁻⁴, MSSI 1.7 × 10⁻⁴, ER/other 2.2 × 10⁻⁴; P016 carried 3.4 × 10⁻⁵ for ν). Uncertainty budget at 200–270 keV: normalisation ±20–25 % (Table I ±18 %); flux shape ×0.5–1.2 (the ≥ 110 MeV flux, 1.5–3.3 cm⁻² s⁻¹); form factor ×0.14–0.7 (shell / skin vs Helm) — the last dominates because 200–270 keV sits in the second diffraction minimum (nodes 245–279 keV depending on the model). In 270–420 keV the three form factors agree within ×0.7–1.3 (the secondary maximum near 350 keV is model-robust), so the extension bin is *better* predicted than the LZ window.

### 4.3 DSNB and solar neutrinos (`P087_other_neutrinos.csv`)

E_R,max = 4.45/5.76/14.7/58.8/163 keV for 16.5/18.77/30/60/100 MeV. ⁸B and hep contribute exactly zero above 5.8 keV (hep: 5.9 × 10⁻⁵ per t·yr in 5.4–5.76 keV). DSNB central: 1.3 × 10⁻³ per t·yr in 5.4–55 keV, 1.0 × 10⁻⁸ in 55–125, 1.3 × 10⁻¹², 1.5 × 10⁻¹⁵ and 4 × 10⁻¹⁹ in the three higher bins; very hard (⟨E_x⟩ = 25 MeV): 1.3 × 10⁻², 1.0 × 10⁻⁵, 8 × 10⁻⁸, 1.3 × 10⁻⁹, 1.2 × 10⁻¹¹. Above 55 keV the DSNB is ≤ 10⁻⁵ per t·yr even in the upper bracket (< 2 % of the atmospheric 55–125 keV count) and ≤ 10⁻⁹ at 200–270 keV: negligible everywhere in the extended window, confirming P060.

### 4.4 NR-band location

A 248 keV recoil with the LZ-tuned NEST yields (nestpy 2.1.1, Table S5 with the p(E) break): N_ph = 4525, N_e = 301 → S1c = 498 phd (539 phd on P009's paper-contour scale, factor 1.082), log₁₀S2c = 4.017; the event is at 540.1 phd, 3.967 (1.5σ below the NR median [paper]). A CEνNS recoil of 248 keV populates *exactly* the same {S1c, log₁₀S2c} distribution as a DM nuclear recoil of 248 keV — same NEST NR yields, same band, same 1.5σ probability — so there is no S1/S2 discrimination; the only handles are the spectrum (below), the low-energy companions (N_lo ≈ 1600 per 200–270 keV event, P019; here 0.037/3.1 × 10⁻⁵ = 1170 in 5.4–55 keV per 200–270 keV event) and time.

### 4.5 Neutrino-floor implications (`P087_discovery_limits.csv`, `P087_discovery_summary.csv`, Fig. 2)

Signal counts per t·yr at unit coupling: 894/49.1/6.10 (δ = 300/350/366) and 0.408 (L10) in 200–270 keV (LZ efficiency); 1623/147/38.2 and 0.78 in 200–420 keV (1000 phd). Backgrounds per t·yr: 200–270 keV atmospheric 3.14 × 10⁻⁵ (range 2.2 × 10⁻⁶–3.7 × 10⁻⁵ over shapes × form factors), all modelled NR-band 2.0 × 10⁻⁴ (P016); 200–420 keV atmospheric 4.32 × 10⁻⁵ (5.8 × 10⁻⁶–5.2 × 10⁻⁵), all NR-band 1.22 × 10⁻³ (P050 bins); 55–420 keV atmospheric 9.4 × 10⁻⁴.

*Exposure at which the atmospheric count reaches one event:* 3.2 × 10⁴ t·yr in 200–270 keV (2.7 × 10⁴ for the hardest shape, 2.3 × 10⁵ for the shell model), 8.7 × 10⁴ in 270–420 keV, 1.5 × 10³ in 55–125 keV, 27 t·yr in 5.4–55 keV.

*Discovery limit.* Without background κ_DL = 2.30/(s_unit ε), i.e. σ_n = 2.58 × 10⁻⁵/4.69 × 10⁻⁴/3.78 × 10⁻³ × 2.96 × 10⁻³⁸ cm² at 100 t·yr for δ = 300/350/366 keV (LZ window). The atmospheric background first changes the limit when b ε = 1.32 × 10⁻³, i.e. at **43 t·yr** (200–270 keV; 36 t·yr for the hardest shape) and **31 t·yr** (200–420 keV; 26), where one observed event stops being a 3σ discovery and the required signal jumps from 2.30 to 3.89 events (×1.69). At 100 t·yr the limit with atmospheric background is 4.35 × 10⁻⁵/7.91 × 10⁻⁴/6.38 × 10⁻³ (×1.69) in the LZ window and 2.39 × 10⁻⁵/2.65 × 10⁻⁴/1.02 × 10⁻³ in 200–420 keV. All modelled NR-band backgrounds do this earlier — at 6.7 t·yr (LZ window) and 1.1 t·yr (200–420 keV) — and at 100 t·yr give 4.33 × 10⁻⁵/7.88 × 10⁻⁴/6.35 × 10⁻³ (LZ; identical staircase step to the atmospheric case) and 3.20 × 10⁻⁵/3.55 × 10⁻⁴/1.36 × 10⁻³ (200–420; ×2.3). The LZ best-fit coupling is reached (3σ, 90 % power, no background) at 35/12.5/8.3 t·yr (LZ window) and 19.3/4.2/1.3 t·yr (200–420 keV) — the same ordering as P050's 5σ Asimov exposures 12.3/8.5/6.8 and 9.6/3.6/1.08 t·yr.

*Systematic floor.* κ_floor/κ_LZ = 3 × 0.25 b/(s_unit κ_LZ) = 3.6 × 10⁻⁴/1.3 × 10⁻⁴/8.5 × 10⁻⁵ (δ = 300/350/366, LZ window) and 2.7 × 10⁻⁴/5.9 × 10⁻⁵/1.9 × 10⁻⁵ (200–420 keV); with the full shape + form-factor bracket ×4.5: 1.6 × 10⁻³/5.7 × 10⁻⁴/3.8 × 10⁻⁴ and 1.2 × 10⁻³/2.5 × 10⁻⁴/7.9 × 10⁻⁵. In cross-section: σ_floor = 7.8 × 10⁻⁴⁶/1.4 × 10⁻⁴⁴/1.1 × 10⁻⁴³ cm² (LZ window, 25 %). The floor equals one event only at 1/(0.75 b) = 4.2 × 10⁴ t·yr. The inelastic hint is therefore 10³–10⁴ above its neutrino floor: no exposure discussed in the corpus (≤ 274 t·yr, P034) is limited by the atmospheric systematic, and the other NR-band backgrounds (×6 larger at 200–270 keV, ×28 at 200–420 keV) set the practical floor.

### 4.6 Spectral separation (`P087_shape_separation.csv`, Fig. 3)

| window | background F² | δ = 300 | δ = 350 | δ = 366 | L10 |
|---|---|---|---|---|---|
| 200–270 keV (LZ) | Helm | KL 0.00; N_3σ 1960 | 0.01; 338 | 0.09; 52 | 0.24; 15 |
| 200–270 keV | shell | 0.05; 85 | 0.13; 32 | 0.27; 15 | 0.42; 8 |
| 200–420 keV (1000 phd) | Helm | 0.09; 39 | 0.40; 9.1 | 0.82; 4.8 | 0.36; 10 |
| 200–420 keV | shell | 0.13; 36 | 0.09; 55 | 0.22; 26 | 0.32; 11 |

(N_3σ with the null variance; with the signal variance 3.2–65.) Inside the LZ window the smeared Helm CEνNS spectrum and the δ = 300 keV inelastic spectrum are practically identical (both fall from 200 keV towards the 279 keV node / the 270 keV roll-off): median observed energy 214 keV for both. The inelastic spectra for δ ≥ 350 keV peak at 320–340 keV in the extended window while CEνNS has its diffraction minimum at 265–280 keV, so 5 (δ = 366) to 9 (δ = 350) events separate them at 3σ with a 1000 phd window — but the shape test is moot: at those exposures (0.1–0.3 t·yr at LZ's best-fit rate) the expected CEνNS count is 10⁻⁵. Shape only matters if the atmospheric flux above 100 MeV were mis-modelled by ≥ 10³, which LZ's own low-energy population (0.037 per t·yr, ≈ 1200 companions per 200–270 keV event) excludes.

### 4.7 Annual modulation

The sub-GeV atmospheric flux varies seasonally by ≲ 2 % (1–4 %; stratospheric temperature; phase ≈ northern summer for the muon-decay component) [recalled, uncertain] and by 5–10 % over the 11-yr solar cycle (not annual) [recalled, uncertain]. A 2 % amplitude needs N_3σ = 18/a² = 45 000 events (11 000–180 000) = 1.2 × 10⁶ t·yr of ROI CEνNS, and modulates the 200–270 keV expectation by 6 × 10⁻⁷ per t·yr. P034's inelastic amplitudes a₁ = 0.43/1.18/1.52/1.66 (δ = 300/350/366/380) are 21/59/76/83 times larger and phased to 1–2 June; the Gaussian rule gives N_3σ = 98/13/8/7 (P034's exact 96.5/11.5/6.5/5.3). The atmospheric modulation is irrelevant for P034's test at any exposure.

## 5. Figures

- `figures/P087_fig1_spectra.png` — Fig. 1: left, CEνNS spectra (true energy, per t·yr per keV) for the atmospheric flux (Helm with the flux-shape band; shell-model and neutron-skin form factors), DSNB (central and very hard) and ⁸B+hep, with the bin edges and the event window; right, atmospheric CEνNS (Helm, shell) against the LZ-normalised O₁ inelastic (δ = 300/350/366 keV) and L10 spectra on a linear axis to 600 keV, with the two efficiency curves.
- `figures/P087_fig2_discovery_limits.png` — Fig. 2: 3σ discovery limit σ_n versus exposure for δ = 300/350/366 keV in the LZ window (top) and the 200–420 keV extension (bottom): no background, atmospheric central and maximal, all modelled NR-band backgrounds; LZ best fit and P034's E_3σ marked.
- `figures/P087_fig3_shapes.png` — Fig. 3: normalised, resolution-smeared shapes in the two windows.
- `figures/P087_fig4_formfactor.png` — Fig. 4: Helm, Helm + skin and shell-model weak form factors of natural xenon to 600 keV.

## 6. Robustness, failed approaches

- Turn-on width: σ_lo = 2.5 keV (P050) would predict 45 ⁸B events and raise the atmospheric ROI count by 6 %; σ_lo = 1.0 keV gives 0.39 ⁸B events; only σ_lo ≈ 0.74 keV matches Table I. Nothing above 55 keV depends on it.
- Anchoring: without the ×1.36 anchor every atmospheric number is 26 % lower; the ±25 % recalled normalisation and Table I's ±18 % are consistent at 1σ.
- Flux shape: the 200–270 keV count spans 1.6–3.7 × 10⁻⁵ per t·yr over eleven shapes anchored to the same ROI count; the ≥ 110 MeV flux is the controlling quantity (1.5–3.3 cm⁻² s⁻¹ recalled, 3.7 anchored central).
- Form factor: Helm vs shell differ by ×7 at 200–270 keV (node 279 vs 245 keV) but by only ×1.4 at 270–420 keV; the skin variant lies between. P019's shell F² was ×2.3 lower still because η(v_min) was not divided out.
- Discovery-limit definition: the exact-Poisson staircase (n_crit steps) is used instead of the Asimov formula, which gives Z = 3 for half an event when b ~ 10⁻⁴ and would misrepresent the low-count regime; the 25 % systematic moves the first step from 1.35 × 10⁻³ to 1.32 × 10⁻³ expected background.
- Counting is done in true energy with the efficiency curves; resolution moves ≈ 10 % of counts across the 200 keV edge (P069) and is irrelevant for ratios of 10⁴.
- Abandoned: a first version used P050's σ_lo = 2.5 keV turn-on and integrated only above 5.4 keV, which set the ⁸B count to zero by construction; replaced by the calibrated turn-on and full-range integration. A first flux central (γ_b = 2.0, E₂ = 1.5 GeV) gave dΦ/dE(1 GeV) = 4.5 × 10⁻⁴, ×2 above the recalled Honda value, and was re-centred to γ_b = 2.3, E₂ = 2 GeV (the old central survives as the "hard mid" variant).

## 7. Discussion

The atmospheric neutrino floor of the high-energy xenon search is quantitatively negligible: 3 × 10⁻⁵ events per t·yr in 200–270 keV and 1.2 × 10⁻⁵ in 270–420 keV, against LZ's best-fit signal of 0.35 per t·yr — four orders of magnitude. Its uncertainty is dominated not by the 20–25 % flux normalisation (fixed by LZ's own 0.11 events) but by the weak form factor at q ≈ 250 MeV (×7 between Helm and the shell model, because the window sits in xenon's second diffraction minimum) and by the ≥ 110 MeV flux (×2). None of these matter for the tests proposed in P034, P050 and P069: the atmospheric background first affects a discovery at 31–43 t·yr, where a single event would need a second one, and by then the accidental/MSSI/ER-leakage backgrounds (P016, P038) have already done so at 1–7 t·yr. The systematic floor for the inelastic rate is 10⁻⁴–10⁻³ of LZ's best fit. What atmospheric CEνNS *does* share with the signal is the detector response — a 248 keV CEνNS recoil is indistinguishable event by event from a 248 keV DM recoil — so the discrimination is entirely statistical: the ≈ 1200 low-energy companions per high-energy CEνNS event (LZ sees 0.11 per 2.84 t·yr, so a CEνNS origin of the event is excluded at the 10⁻⁴ level, P019), the spectrum (5–50 events), and the absence of a June modulation (2 % vs 40–170 %). For a 1000 phd extension the 270–420 keV bin is the best-predicted of all (form-factor spread ×1.4), which makes it a clean place for the inelastic peak at 320–340 keV.

## 8. References

LZ Collaboration, arXiv:2609.02823 (2026). J. Billard, L. Strigari, E. Figueroa-Feliciano, Phys. Rev. D 89, 023524 (2014). M. Honda, T. Kajita, K. Kasahara, S. Midorikawa, Phys. Rev. D 83, 123001 (2011). G. Battistoni, A. Ferrari, T. Montaruli, P. R. Sala, Astropart. Phys. 23, 526 (2005). D. Baxter et al., Eur. Phys. J. C 81, 907 (2021). J. D. Lewin, P. F. Smith, Astropart. Phys. 6, 87 (1996). I. Jeong, S. Kang, S. Scopel, G. Tomar, Comput. Phys. Commun. 276, 108342 (2022). T. K. Gaisser, M. Honda, Annu. Rev. Nucl. Part. Sci. 52, 153 (2002). Corpus: P009, P016, P019, P021, P034, P038, P050, P060, P069.

## 9. Tools and provenance (mirrors `output/provenance/P087.json`)

- Agent tools: Read ×19 (PAPER_GUIDE.md; P060.md; P060/details.md; P050.md; P069.md; P034.md; P019.md; P019_atm_nu.py; P060_astro_nu.py; P050_next_generation.py l.40–120; lzcommon.py l.1–62 and 108–387; fulltext.tex l.160–250, 295–330, 500–590; four P087 figures), Bash ×13 (tex grep for neutrino lines; corpus listings + lzcommon API + versions + ledger rows; dossier/P019 JSON/P050 cache/P034 details inspection; P050 cache/WimPyDD/nestpy timing test; palette grep; script run ×3; results print; wc -w checks ×4), Skill ×1 (dataviz), Write ×4 (script, details.md, P087.json, P087.md), Edit ×29 (script 23: turn-on calibration, flux re-centring, full-range ROI integration, exact single-event exposures, F² nodes; paper 6: word-budget trims).
- Software: python 3.12.13; numpy 2.5.3 (geomspace, trapezoid, interp, cumsum, polynomial.hermite_e.hermegauss); scipy 1.18.1 (special.erf, optimize.brentq, stats.poisson, stats.norm); pandas 3.0.5; matplotlib 3.11.2 (Agg); WimPyDD 2.0.4 via lzcommon (`wd_hamiltonian`, `wd_halo` with explicit v_min grid, `wd_rate`, `wd_c_from_anand`); nestpy 2.1.1 via `lz.nest_nr_yields`; common/lzcommon.py (LZ constants, XE_ISOTOPES, m_nucleus_gev, eta0, vmin_kms, v_earth_kms, GEV_TO_CM2, HBARC_GEV_FM, M_V_GEV, M_NUCLEON_GEV).
- Local inputs: fulltext.tex Table I (l.210–246), l.174–178, l.307–314, prompt/delayed tables (l.506–589); P019_results.json (S1 scale factor); P050_spectra_cache.npz and P050_background_bins.csv; P019/P034/P050/P060/P069 papers and details; results_ledger.csv rows.
- Recalled knowledge: 20 items (§2). Datasets: none. Data requests: none. WimPyDD-generated files: none (diff_rate only; no response-function files written).
