# P028 — Iodine as the second target: what DAMA/LIBRA's modulation data and COSINE/ANAIS already say about an inelastic 300–390 keV splitting

Simulated date 2026-09-08 · category XEXP · hep-ph (cross-list astro-ph.CO) · author profile: annual-modulation and NaI(Tl) phenomenologists.
Script: `output/code/P028_iodine_dama.py` (run from the simulation root with `.venv/bin/python`; 14 s). Tables in `output/work/P028/`, figures in `output/work/P028/figures/`.

## 1. Motivation and framework

P015 showed that iodine (A = 127) is the only non-xenon nucleus in an existing large exposure that can scatter inelastically at the LZ-favoured splittings (δ_max(I, 1 TeV, 16 June) = 385 keV vs 396 keV for xenon), that at LZ's one-event best fit DAMA/LIBRA's 2.46 t·yr contains ≈ 0.5 iodine recoils in 9–67 keVee, and that these are buried in ~5 × 10⁷ undiscriminated counts. P006 showed that the inelastic rate is strongly modulated (41% at δ = 300 keV, 90% at 350, ~100% at 380 keV in LZ's ROI). NaI(Tl) experiments do not count events; they measure the *modulation amplitude* S_m per energy bin, and DAMA/LIBRA claims a positive S_m at 2–6 keVee. So the natural question is: what S_m does the LZ best fit predict in DAMA's electron-equivalent bins, is it visible in DAMA/LIBRA, COSINE-100 or ANAIS-112, and what NaI exposure would be needed?

Framework (all as in P015): O₁ isoscalar inelastic (endothermic) scattering, m_χ = 1000 GeV, δ = 300, 350, 366 (P007 Higgsino δ(N = 1)) and 380 keV; WimPyDD 2.0.4 shell-model responses (127I from Anand et al.); Baxter-2021 SHM (v₀ = 238, v_esc = 544 km/s, Sun's peculiar velocity (11.1, 12.2, 7.3) km/s) with Earth's orbital velocity through WimPyDD's `streamed_halo_function(day_of_the_year=…)`, on the explicit v_min grid of `lz.wd_halo` (0–844 km/s, 1200 points; P002 fix). LZ unit coupling c_p = c_n = 1/m_v² ↔ WimPyDD c⁰ = 2/m_v² (P003 convention, `lz.wd_c_from_anand`). Normalisation: the xenon annual-mean rate (12 mid-month halos, as P015), folded with the LZ efficiency model (0.96 plateau, 50% at 5.4 and 269.9 keV, erf edges) and 2.84 t·yr, is set to 1.0 event.

## 2. Equations

*Kinematics.* For splitting δ, DM speed v and reduced mass μ, recoils occur in [E₋, E₊] = (μ²v²/m_N)[1 − δ/(μv²) ∓ √(1 − 2δ/(μv²))], requiring δ ≤ μv²/2 (`lz.E_R_range_keV`). At v_max = v_E(16 June) + v_esc = 809.5 km/s: δ_ceiling = 385.3 keV (I), 76.4 keV (Na), 397.1 keV (Xe) at 1 TeV. **Sodium is kinematically blind for every δ ≥ 300 keV at any WIMP mass (μ → m_Na gives 78 keV)**, so `E_R_range_keV(1000, 809.5, A=23, delta=300…380)` returns NaN and the NaI signal is iodine only.

*Rate.* WimPyDD's `diff_rate(target, H, m_χ, E, v_min, δη, delta=δ)` returns dR/dE_R in events/kg/day/keV as Σ_streams K(E, v_min) δη(v_min). We compute the per-stream kernel K once per (E, δ) with `sum_over_streams=False` and `delta_eta = 1`, and contract it with the halo function of each of 73 uniformly spaced days (5.0 d spacing), the 12 mid-month days (annual mean, P015 convention) and 16 June (day 167). Check: K·δη reproduces `diff_rate` with the day's δη to machine precision.

*Electron-equivalent energy.* With a constant iodine quenching Q_I, E_ee = Q_I E_R and dR/dE_ee = (1/Q_I) dR/dE_R|_{E_R = E_ee/Q_I}. Per kg of NaI: multiply by the iodine mass fraction f_I = 126.904/(126.904 + 22.990) = 0.8466. **Validation:** WimPyDD's bundled DAMA_LIBRA_2019 target (Na and I each with a `quenching` attribute: 0.3 and 0.09, `iodine_quenching.tab`) interprets `er` as E_ee and sums Na + I per kg of NaI; at 2 energies × 4 δ our manual conversion agrees with it to 0.03–4% (the 4% at δ = 300 keV, 47 keVee, is interpolation on our 90-point recoil grid near the 290 keV form-factor node); the sodium term is identically zero. DAMA's resolution σ(E_ee) = 0.0091 E + 0.448 √E keV (WimPyDD DAMA folder, from Bernabei et al. 2010) is applied as a variant.

*Time dependence and harmonics.* Rates S(t) are binned in 1-keVee bins (1–90 keVee) for each of the 73 days. With ω = 2π/365.25 and DAMA's phase t₀ = 152.5 d:
- S₀ = ⟨S⟩, S_m = 2⟨S cos ω(t − t₀)⟩ (the amplitude DAMA fits); free-phase amplitude A₁ = |2⟨S e^{iωt}⟩| and peak day t_peak; harmonics A_k = |2⟨S e^{ikωt}⟩|, k = 2, 3, 4; fraction of variance in the fundamental ½A₁²/Var(S); fraction of the year with zero rate.
For a periodic function that is non-negative and sharply peaked, A₁ → 2S₀ (a spike), so S_m/S₀ > 1 is possible and signals a non-sinusoidal shape; the conventional (S_max − S_min)/(S_max + S_min) → 1 in that limit.

*Sensitivity to a cosine amplitude.* For counts taken uniformly over the year with a flat background B (cpd/kg/keV) in a bin of width ΔE and total exposure ℰ (kg·d), the least-squares estimator Ŝ_m = (2/ℰΔE)Σ_i N_i cos ω(t_i − t₀) has Var = (4/(ℰΔE)²)·Σ_i BΔE m Δt_i cos² = 2B/(ℰΔE), so
  σ(S_m) = √(2B/(ℰ ΔE)),  z(one bin) = S_m/σ,  z²(all bins) = (ℰ/2B) Σ_bins S_m,i² ΔE_i,
and the exposure for a 3σ detection is ℰ_3σ = 18 B/(S_m² ΔE) (one bin) or 18B/Σ S_m,i² ΔE_i. For a background-free detector the fluctuations are those of the signal itself (B → S₀): ℰ = 18 S₀,win/S_m,win², to be compared with the exposure for 3 counted events, 3/S₀,win.

## 3. Inputs

| Input | Value | Source |
|---|---|---|
| LZ exposure, efficiency edges, event date | 2.84 t·yr; 5.4 / 269.9 keV (50%), 0.96 plateau; 16 June 2023 | LZ paper via `lz.LZ` |
| Halo | Baxter 2021 SHM via `lz.wd_halo`; Earth velocity via WimPyDD | lzcommon (recalled, certain) |
| Coupling convention | c⁰_WimPyDD = 2/m_v² for LZ unit coupling | P003 (settled), `lz.wd_c_from_anand` |
| LZ 1-event normalisation N_unit(δ) | 13 564 / 264.6 / 21.71 / 0.6836 events at unit coupling (δ = 300/350/366/380) | recomputed here; identical to P015 (`P015_summary.json`) to 4 digits |
| Iodine quenching Q_I | 0.09 (range 0.06–0.12) | recalled (likely); also WimPyDD `iodine_quenching.tab` |
| Sodium quenching | 0.30 (WimPyDD); 0.25–0.3 recalled | irrelevant (blind) |
| Atomic masses I, Na | 126.904, 22.990 u | recalled (certain) |
| DAMA phase | t₀ = 152.5 d (2 June) | recalled (certain); WimPyDD default |
| DAMA single-hit rate at 10–60 keVee | ≈ 1 cpd/kg/keV | recalled (likely); P015 |
| DAMA/LIBRA exposure | 2.46 t·yr (phase1 1.33 + phase2 1.13 to 2018); later updates larger (~2.9 t·yr) | recalled (likely / uncertain) |
| DAMA/LIBRA-phase2 S_m per bin (1–16 keVee) | `WimPyDD/Experiments/DAMA_LIBRA_2019/modulation_amplitudes.tab`, header "taken from Fig. 11 of Nucl. Phys. At. Energy 19 (2018) 307"; we interpret columns 3–4 as the lower/upper edges of the ±1σ band (e.g. 2–2.5 keVee: 0.0150–0.0206 → 0.0178 ± 0.0028; 6–6.5: −0.0009–0.0027; 8–16: −0.00006–0.00086 cpd/kg/keV) | local file (bundled digitisation); interpretation ours (likely) |
| DAMA S_m above 6 keVee | consistent with zero at the ~10⁻³ cpd/kg/keV level; per-0.5-keV-bin errors ≈ 0.002–0.003 at 6–20 keVee | recalled (likely / uncertain) |
| COSINE-100 | ≈ 0.2 t·yr NaI, background ≈ 3 cpd/kg/keV at 2–6 keVee, null S_m at ~0.005 | recalled (uncertain) |
| ANAIS-112 | ≈ 0.3 t·yr (3 yr × 112.5 kg; ~0.6 t·yr by 2025), background ≈ 3 cpd/kg/keV, null S_m at ~0.005 | recalled (uncertain) |
| DAMA high-energy backgrounds | ⁴⁰K (3.2 keV X-ray/Auger; 1461 keV γ), ²¹⁰Pb (46.5 keV γ) dominate the single-hit spectrum near the iodine window | recalled (likely) |

## 4. Results

### 4.1 Kinematics and unit-coupling rates (reproduction of P015)
δ ceilings at 1 TeV on 16 June: I 385.3, Na 76.4, Xe 397.1 keV. Iodine June windows: 96.6–745.0 (δ = 300), 167.6–584.6 (350), 207.6–515.9 (366), 268.6–429.9 keV (380). Window rates at unit coupling per t·yr of iodine (annual / June): 3445 / 5070 (300), 61.8 / 157.8 (350), 6.29 / 20.97 (366), 0.198 / 1.044 (380) — P015 gave 3449, 61.8, 6.30, 0.198. I/Xe (annual) = 0.589 / 0.296 / 0.143 / 0.023; June/annual for iodine 1.47 / 2.55 / 3.33 / 5.28. LZ normalisation identical to P015 (§3); best-fit (c₁ˢm_v²)² = 7.37 × 10⁻⁵, 3.78 × 10⁻³, 4.61 × 10⁻², 1.46; σ_SI = 2.19 × 10⁻⁴², 1.12 × 10⁻⁴⁰, 1.37 × 10⁻³⁹, 4.34 × 10⁻³⁸ cm².

True-energy iodine spectra at the best fit (Fig. 3, `spectra_true_energy.csv`): they peak at 2.5 × 10⁻³ events/(t·yr NaI·keV) near 180 keV (δ = 300) and show the 127I M-response node at ≈ 290 keV (cf. xenon's 266 keV, P002/P017); for δ = 380 keV the spectrum is confined to 270–430 keV.

### 4.2 Electron-equivalent spectra and modulation (Q_I = 0.09; `bins_S0_Sm.csv`, `quenching_variants.csv`)

| δ (keV) | window (keVee) | S₀ peak [cpd/kg/keVee] (bin) | S_m peak (bin) | S_m peak, DAMA resolution | window S₀ [cpd/kg] | window S_m | S_m/S₀ | t_peak (d) | A₂/A₁ | var. in fundamental | zero-rate fraction | max/min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | 8.7–67.0 | 7.6 × 10⁻⁸ (16–17) | 3.5 × 10⁻⁸ (15–16) | 2.9 × 10⁻⁸ | 5.89 × 10⁻⁷ | 2.73 × 10⁻⁷ | 0.46 | 151.9 | 0.06 | 0.997 | 0 | 2.6 |
| 350 | 15.1–52.6 | 5.7 × 10⁻⁸ (19–20) | 8.0 × 10⁻⁸ (19–20) | 5.6 × 10⁻⁸ | 5.41 × 10⁻⁷ | 6.87 × 10⁻⁷ | 1.27 | 151.9 | 0.28 | 0.93 | 0 | 59 |
| 366 | 18.7–46.4 | 6.8 × 10⁻⁸ (32–33) | 1.05 × 10⁻⁷ (32–33) | 8.1 × 10⁻⁸ | 6.72 × 10⁻⁷ | 1.07 × 10⁻⁶ | 1.59 | 151.9 | 0.48 | 0.80 | 0.33 | ∞ |
| 380 | 24.2–38.7 | 1.13 × 10⁻⁷ (31–32) | 2.1 × 10⁻⁷ (31–32) | 1.4 × 10⁻⁷ | 6.74 × 10⁻⁷ | 1.26 × 10⁻⁶ | 1.87 | 151.9 | 0.81 | 0.48 | 0.66 | ∞ |

Notes. (i) The window-integrated unmodulated rate is 5.4–6.7 × 10⁻⁷ cpd/kg of NaI for every δ, i.e. 0.20–0.25 events per t·yr — the LZ normalisation fixes the total iodine count (P015's 0.25–0.29 per t·yr of iodine ≈ 0.21–0.25 per t·yr NaI) and the shrinking window concentrates it. (ii) The phase is 151.9 d (1–2 June) at every δ and in every bin: high-v_min signals peak with v_E (no phase reversal, cf. P006). (iii) At δ = 300 keV the modulation is 39–47% in the core of the window (12–45 keVee) and rises towards 2 at the window edges (8–10 and 60–67 keVee), which are populated only in summer; the time dependence is sinusoidal to 0.3% (A₂/A₁ = 0.06). (iv) For δ ≥ 350 keV the cosine amplitude exceeds the mean: S_m/S₀ = 1.27, 1.59, 1.87, approaching the spike limit 2. The shape is a peak of FWHM ≈ 190 d (350), 120 d (366), 70 d (380) around 2 June, with the rate identically zero for 33% (366) and 66% (380 keV) of the year (Fig. 2). Second harmonics are 28%, 48%, 81% of the first; a single-cosine fit captures only 93%, 80%, 48% of the variance. DAMA's S_m definition still applies, but the ratio S_m/S₀ is no longer a "modulation fraction"; P006's 90%/100% figures (max–min based) and our 1.3–1.9 describe the same curves. (v) Resolution smearing (σ = 2.1 keV at 20 keVee) lowers the peak S_m by 17–35% and fills the window edges; it does not change any conclusion. (vi) Q_I: the window scales as Q (5.8–44.7 keVee at 0.06, 11.6–89.4 at 0.12 for δ = 300) and the per-keVee amplitudes as 1/Q (peak S_m at δ = 300: 5.3 × 10⁻⁸ / 3.5 × 10⁻⁸ / 2.6 × 10⁻⁸ for Q = 0.06/0.09/0.12); window integrals and S_m/S₀ are Q-independent. **In no case does the window reach 2–6 keVee**: even at Q_I = 0.06 it starts at 5.8 keVee; reaching 4 keVee at δ = 300 keV would need Q_I < 0.04.

### 4.3 Comparison with DAMA/LIBRA, COSINE-100, ANAIS-112 (`dama_comparison.csv`)
Poisson sensitivity of DAMA/LIBRA (ℰ = 2.46 t·yr = 8.99 × 10⁵ kg·d, B = 1 cpd/kg/keV, ΔE = 1 keVee): σ(S_m) = 1.49 × 10⁻³ cpd/kg/keV (2.1 × 10⁻³ per 0.5-keV bin), which matches the recalled DAMA/LIBRA-phase2 per-bin uncertainties (≈ 0.002–0.003 at 6–20 keVee) and the bundled table's half-widths (0.0025–0.0034 at 6–8 keVee, 0.00046 for the 8-keV-wide 8–16 keVee bin) within a factor ~1.5 — a check that B ≈ 1 cpd/kg/keV and 2.46 t·yr are the right scale.

| δ (keV) | σ(S_m)/S_m,peak (DAMA) | z, peak bin | z, all bins combined | signal events in DAMA window | background counts in window | σ/S_m,peak COSINE / ANAIS |
|---|---|---|---|---|---|---|
| 300 | 4.2 × 10⁴ | 2.4 × 10⁻⁵ | 5.4 × 10⁻⁵ | 0.53 | 5.2 × 10⁷ | 2.6 × 10⁵ / 2.1 × 10⁵ |
| 350 | 1.9 × 10⁴ | 5.4 × 10⁻⁵ | 1.2 × 10⁻⁴ | 0.49 | 3.4 × 10⁷ | 1.1 × 10⁵ / 9.2 × 10⁴ |
| 366 | 1.4 × 10⁴ | 7.0 × 10⁻⁵ | 1.8 × 10⁻⁴ | 0.60 | 2.5 × 10⁷ | 8.6 × 10⁴ / 7.0 × 10⁴ |
| 380 | 7.1 × 10³ | 1.4 × 10⁻⁴ | 2.9 × 10⁻⁴ | 0.61 | 1.3 × 10⁷ | 4.3 × 10⁴ / 3.5 × 10⁴ |

The only bundled DAMA bin overlapping a predicted window is the 8–16 keVee control bin (0.0004 ± 0.0005 cpd/kg/keV): the δ = 300 keV prediction averaged over it is 1.5 × 10⁻⁸, 3 × 10⁴ below the error. The DAMA 2–6 keVee signal (S_m ≈ 0.01–0.02 cpd/kg/keV) receives exactly zero contribution: the LZ-type inelastic signal cannot be what DAMA sees, and DAMA's positive amplitude cannot be reinterpreted as LZ-type inelastic iodine recoils (the two are separated by ≥ 3 keVee in threshold and by 5–6 orders of magnitude in amplitude). Conversely, S₀/B ≈ 10⁻⁷: the iodine window 9–67 keVee sits where DAMA's single-hit spectrum is dominated by ⁴⁰K and ²¹⁰Pb, and the predicted 0.5 events are invisible in 5 × 10⁷ counts, as P015 said. COSINE-100 and ANAIS-112 (σ ≈ 7–9 × 10⁻³ at B ≈ 3, 0.2–0.3 t·yr) are a further factor 5–6 less sensitive.

### 4.4 Required exposures (`P028_summary.json` → `comparison.*.required_exposure`)

| δ (keV) | DAMA-like NaI (B = 1), 3σ in peak 1-keVee bin | same, all bins combined | background-free iodine detector: 3σ modulation | background-free: 3 counted events |
|---|---|---|---|---|
| 300 | 4.0 × 10¹⁰ t·yr | 7.7 × 10⁹ t·yr | 390 t·yr NaI | 14.0 t·yr NaI (11.8 t·yr I) |
| 350 | 7.6 × 10⁹ | 1.6 × 10⁹ | 57 | 15.2 (12.8) |
| 366 | 4.5 × 10⁹ | 6.6 × 10⁸ | 29 | 12.2 (10.3) |
| 380 | 1.1 × 10⁹ | 2.6 × 10⁸ | 21 | 12.2 (10.3) |

A DAMA-like undiscriminated NaI experiment would need 10⁸–10¹⁰ t·yr (10⁸–10¹⁰ times the world's NaI exposure); even a perfectly background-free NR-discriminating iodine detector needs 20–400 t·yr to see the *modulation* at 3σ, versus 12–15 t·yr to count 3 events (P015: 12–15 t·yr). For iodine the modulation is a confirmation tool only after the signal has been counted; tungsten (P015) remains the decisive non-xenon target.

## 5. Validation and robustness
1. LZ normalisation and I/Xe rate ratios reproduce P015 to 4 significant figures (same convention, halo, efficiency).
2. Manual E_ee conversion vs WimPyDD's DAMA target: ratios 1.014/1.040 (δ = 300), 0.999/1.003, 0.999/1.000, 1.000/1.001.
3. Kernel contraction vs direct `diff_rate`: identical (1.3054 × 10⁻⁴ both, δ = 300 keV, 200 keV, June).
4. Poisson σ(S_m) at DAMA's exposure matches the recalled published errors within ×1.5.
5. 73-day annual mean vs 12-mid-month mean: N_unit agrees to < 1% (`N_LZ_unit_coupling_73day_mean`).
6. Harmonic analysis on 73 uniform samples is exact for k ≤ 36; the peak day 151.9 d agrees with P006 (152–153).
7. Variants: Q_I = 0.06–0.12 (§4.2 vi); DAMA resolution (17–35% lower peak S_m); background 1 vs 3 cpd/kg/keV (√3 in σ). The small wiggles in the δ = 380 keV true-energy spectrum at 350–440 keV are the 0.7 km/s v_min-grid discretisation near the kinematic edge (≤ 10% locally, negligible after binning).

## 6. Failed or abandoned
- A first summary-printing snippet keyed the CSV by '300.0' instead of '300' (formatting), fixed; no physics impact.
- We did not run WimPyDD's `wimp_dd_rate` on the DAMA experiment object (it writes response-function files and its bins stop at 16 keVee); `diff_rate` with the DAMA target was used for validation only.
- No data request: a HEPData-style DAMA S_m table would sharpen the comparison by < ×2 while the discrepancy is 4–5 orders of magnitude.

## 7. Figures
- `figures/fig1_S0_Sm_vs_DAMA.png` — Top: unmodulated NaI rate S₀(E_ee) for δ = 300/350/366/380 keV at LZ's one-event fit (Q_I = 0.09) vs DAMA's ≈ 1 cpd/kg/keV single-hit rate. Bottom: cosine amplitude S_m(E_ee) (t₀ = 2 June) with peak values labelled; grey: DAMA/LIBRA-phase2 ±1σ bands (bundled digitisation, bands reaching zero cut at 10⁻⁴); dashed: Poisson σ(S_m) for DAMA (2.46 t·yr, B = 1, 1 keV bins); dash-dotted: COSINE/ANAIS scale.
- `figures/fig2_time_dependence.png` — window-integrated iodine rate through the year normalised to its mean (solid) and the fitted first harmonic 1 + (S_m/S₀)cos ω(t − t₀) (dashed); the δ ≥ 350 keV curves are spikes around 2 June with zero rate for a third (366) to two thirds (380 keV) of the year.
- `figures/fig3_true_energy_spectra.png` — iodine recoil spectra (annual mean) in events/(t·yr NaI·keV) at the LZ best fit; 127I M-response node at ≈ 290 keV.

## 8. Discussion
The LZ event, if inelastic DM, predicts a NaI(Tl) modulation signal that is qualitatively DAMA-like — June phase, large fractional modulation, 9–67 keVee — but quantitatively 10⁻⁷ of DAMA's background and 10⁴ below DAMA's own statistical reach, with nothing at all below 8.7 keVee (5.8 keVee at Q_I = 0.06). The DAMA 2–6 keVee amplitude and the LZ event are therefore unrelated under this hypothesis, in both directions. The result is insensitive to Q_I, resolution, and the exact DAMA exposure (2.46 vs ~2.9 t·yr) and background. The one structural novelty is that for δ ≥ 350 keV the signal is not a cosine: a DAMA-style single-harmonic fit would report S_m > S₀ and lose 7–52% of the variance; experiments that could ever see such a signal should fit a template R(t) rather than a cosine (or fit the second harmonic as well). Since iodine's window rate per t·yr is fixed by LZ's count and is only 0.2–0.25 events/(t·yr NaI), the NaI route to confirmation is closed at any conceivable exposure, and its modulation would be a confirmation tool only after ~15 t·yr of a discriminating iodine detector had counted the events; tungsten (P015) and xenon with an extended ROI (P005, P015, P020) remain the decisive tests.

## 9. References
LZ Collaboration, arXiv:2609.02823 (2026). R. Bernabei et al. (DAMA/LIBRA), Nucl. Phys. At. Energy 19, 307 (2018); Eur. Phys. J. C 56, 333 (2008). G. Adhikari et al. (COSINE-100), Phys. Rev. Lett. 123, 031302 (2019). J. Amaré et al. (ANAIS-112), Phys. Rev. D 103, 102005 (2021). D. Tucker-Smith, N. Weiner, Phys. Rev. D 64, 043502 (2001). K. Freese, M. Lisanti, C. Savage, Rev. Mod. Phys. 85, 1561 (2013). I. Jeong, S. Kang, S. Scopel, G. Tomar (WimPyDD), Comput. Phys. Commun. 276, 108342 (2022). D. Baxter et al., Eur. Phys. J. C 81, 907 (2021). Corpus: dossier 00, P002, P003, P005, P006, P007, P015, P020.

## 10. Tools and provenance (mirrors `provenance/P028.json`)
- Agent tools: Read ×22 (PAPER_GUIDE, dossier, ledger, P015/P006/P002/P007 papers, lzcommon (3 ranges), P015 script, WimPyDD test script, WimPyDD package.py (3 ranges), DR-001, figures ×6), Bash ×16 (listings, greps, WimPyDD tests, 3 script runs, table prints, word counts), Write ×5 (paper written twice), Edit ×19 (5 figure fixes, 14 paper trims), Skill ×1 (dataviz).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (special.erf/erfc); matplotlib 3.11.2; WimPyDD 2.0.4 (`eft_hamiltonian`, `streamed_halo_function` via `lz.wd_halo`, `diff_rate` with `sum_over_streams=False` on `WD.I`, `WD.Xe`, `WD.DAMA_LIBRA_2019.target`); common/lzcommon.py (`LZ`, `wd`, `wd_halo`, `wd_hamiltonian`, `E_R_range_keV`, `mu_red`, `m_nucleus_gev`, `v_earth_kms`, constants).
- Script: `output/code/P028_iodine_dama.py`; command `.venv/bin/python output/code/P028_iodine_dama.py`.
- Local inputs: PAPER_GUIDE.md; 00_evidence_dossier.md; results_ledger.csv; papers P015/P006/P002/P007; work/P015/P015_summary.json (lz_normalisation), rates_table.csv, existing_exposures_counts.csv, details.md; code/P015_target_complementarity.py; common/lzcommon.py; WimPyDD/test_WimPyDD_installation.py (l. 30–90), WimPyDD/package.py (diff_rate, element, target), WimPyDD/Experiments/DAMA_LIBRA_2019/* (quenching, resolution, bins, modulation_amplitudes.tab); environment/ENVIRONMENT_versions.txt; data_requests/DR-001.md.
- Recalled knowledge (12): Q_I = 0.09 (likely), Q_Na 0.25–0.3 (likely), atomic masses (certain), DAMA phase 2 June (certain), DAMA single-hit rate ≈ 1 cpd/kg/keV at 10–60 keVee (likely), DAMA exposure 2.46 t·yr (likely; later ~2.9 uncertain), DAMA S_m above 6 keVee consistent with zero, errors 0.002–0.003 (likely/uncertain), COSINE-100 0.2 t·yr and B ≈ 3 (uncertain), ANAIS-112 0.3 t·yr and B ≈ 3 (uncertain), ⁴⁰K/²¹⁰Pb dominance of DAMA's spectrum (likely), cosine-amplitude estimator variance (certain), bibliographic details (likely).
- Datasets: WimPyDD-bundled `modulation_amplitudes.tab` (local digitisation of DAMA 2018 Fig. 11; interpretation of columns as ±1σ band edges is ours). Data requests: none. WimPyDD-generated files: none.
