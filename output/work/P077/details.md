# P077 — S1 pulse-shape discrimination at 550 phd: what singlet/triplet ratios, photon statistics and propagation in a 1.5 m TPC allow, and why LZ called it inconclusive

Simulated date 2026-09-15. Category RESP, physics.ins-det. Author profile: LXe scintillation-timing experts.
Script: `output/code/P077_s1_psd.py` (run from the simulation root with `.venv/bin/python`; ~5 min; seed 20260915).
Outputs: `output/work/P077/{P077_results.json, optics_vs_z.csv, delta_scan.csv, position_dependence.csv, recalled_inputs.json, run_log.txt}` and four figures in `figures/`.

## 1. Motivation and framework

LZ's supplement ("Waveform Analysis", tex l. 687–695) makes four pulse-shape statements about the 248 keV candidate (S1c = 540.1 phd, z = 26.4 cm above the cathode, r = 45.9 cm):

1. S1 timing "has limited discrimination power on an event-by-event basis for events with S1c ~ 550 phd"; the event "is generally compatible with both ER and NR populations".
2. "AmBe calibration NRs with approximately 150 keV are not inconsistent with an ER template built from ¹³¹ᵐXe calibration data near the position of the event of interest".
3. Events below the FV near the cathode (where RFR MSSI would sit) have "shorter rise times than all samples at the same depth as the event"; the event's waveform is "less consistent with that of a RFR MSSI than with a NR at the reconstructed position".
4. The top–bottom asymmetry is consistent with the reconstructed z; the S1 hit pattern agrees at 2σ (z) and 1σ (x, y) with SS ERs of similar size and position; RFR-MSSI template excluded, wall-MSSI template not distinguishable.

Statement 1 is at first sight surprising: 540 detected photons is a large sample, and textbook LXe timing (singlet 2–4 ns, triplet 21–27 ns, NRs singlet-rich, ERs triplet/recombination-dominated) would suggest a mean-arrival-time resolution of ~1–2 ns against an ER–NR difference of ~5–10 ns. We build a toy waveform model to ask (a) what separation the recalled parameter brackets predict at 540 phd and at 150 keV-NR size (297 phd); (b) how photon propagation in a 1.5 m PTFE-lined TPC and the resulting position dependence degrade it; (c) what statement 2 implies about the *actual* ER–NR timing difference at these energies and 97 V/cm, and hence what likelihood ratio a single-event PSD verdict could at best have carried; (d) whether the RFR-versus-FV depth discrimination (statements 3–4) is robust; (e) what LZ could still do.

Nothing here uses the event waveform (not public). All timing constants are recalled and flagged (Sec. 2); the result type is "estimated".

## 2. Inputs

| Input | Value | Source | Flag |
|---|---|---|---|
| S1c, z, r of the event | 540.1 phd; 26.4 cm above cathode; r = 45.9 cm | paper (Results, Fig. 3); `lz.LZ` | paper |
| g₁ | 0.110 phd/photon | paper; `lz.LZ["g1"]` | paper |
| RFR depth | 13.75 cm | paper (MSSI paragraph) via P022/P042 | paper |
| TPC radius, drift length | 72.8 cm, 145.6 cm | LZ design papers | recalled, likely |
| n(LXe, 175 nm) | 1.69 (group-index variant 1.9) | Solovov 2004 / Hitachi 2005; dispersion | likely / uncertain |
| Rayleigh length | 35 cm (variant 30) | Ishida 1997, Seidel 2002 | uncertain |
| PTFE reflectance in LXe | 0.95 baseline (0.97, 0.90 variants) | Neves 2017, Kravitz 2020 (≥ 0.95) | likely |
| array-plane reflectance, per-hit detection prob., grid transparency | 0.20, 0.16, 0.90 (two grids to the top array, one to the bottom) | tuned here to reproduce g₁ = 0.110 at the TPC centre | derived (Sec. 3.1) |
| PMT TTS | σ = 1.5 ns (R11410 datasheet ~9 ns FWHM would be 3.8 ns) | assignment / datasheet | uncertain |
| digitiser | 10 ns samples | LZ DAQ 100 MS/s | likely |
| τ_singlet, τ_triplet | 2–4.3 ns; 21–27 ns | Hitachi 1983; paper quotes 2–4 / 21–28 | certain (ranges) |
| recombination time (0 V/cm, electrons) | 45 ns | Kubota 1979 | likely |
| I_s/I_t ER at zero field | 0.05 | Hitachi 1983 | likely |
| I_s/I_t ER at drift field | 0.4–0.6 | assignment bracket; LUX 2018 (180 V/cm) | uncertain |
| I_s/I_t NR | 1.0–3.0 (1.6 fission fragments; ~1.5 Xe recoils) | Hitachi 1983; Akimov 2002; Dawson 2005; Kwong 2010 | uncertain |
| N_ph for a 150 keV NR | ≈ 2700 → 297 phd | P009 contour scale Nq = 11.32 E^1.112, ~10 % electrons | likely |
| ER equivalent of the event | 69.6–70.9 keV; ¹³¹ᵐXe = 164 keV | dossier, P010 | paper/certain |

`recalled_inputs.json` lists the same items with reliabilities.

## 3. Model

### 3.1 Optical photon Monte Carlo (script §1)

Photons start isotropically at (r, 0, z) in a cylinder r < 72.8 cm between the bottom PMT plane (z = −13.75 cm, below the RFR) and the liquid surface (z = 146.1 cm). Steps: distance to the wall, to the two planes, to the cathode plane and a Rayleigh free path ~ Exp(λ_R = 35 cm) are compared; the shortest wins; time advances by s/(c/n).
- Wall: Lambertian (cosine-weighted) reflection with probability R_PTFE, else absorbed.
- Bottom plane: one shield-grid crossing (T = 0.90), then detection with ε = 0.16 (QE × coverage), else diffuse reflection with R_arr = 0.20, else absorbed.
- Liquid surface: total internal reflection (specular) for sin θ > 1/n = 0.59; otherwise the photon crosses gate + anode (T²) and reaches the top array (0.2 ns gas transit): detection ε, diffuse reflection R_arr·T² (it re-crosses the grids), else absorbed.
- Cathode grid: crossing with probability 0.90, else absorbed.
- Rayleigh scattering: isotropic redirection (the 1 + cos²θ phase function is not implemented; the effect on times is second order).
- No bulk absorption (attenuation length ≫ path).

Tuning: `P077_TUNE=1` scans (R_PTFE, R_arr, ε, T) and prints the collection efficiency, which equals the g₁ equivalent because ε includes the QE. The chosen set gives 0.112 at z = 73 cm (LZ: 0.110), 0.117 at the event depth, 0.109–0.114 elsewhere. The propagation time distribution is insensitive to the tuning: for all 32 sets (R_PTFE 0.95–0.97, R_arr 0.2–0.4, ε 0.12–0.25, T 0.85–0.9) the mean at the event depth is 25–32 ns with sd 28–35 ns. This is geometry, not tuning: a photon travels ≈ 1 m (5.6 ns) per PTFE bounce and survives ~10–20 bounces before an array or a grid absorbs it.

Result (`optics_vs_z.csv`, Fig. 3 left/middle):

| z (cm) | collection | f_top | TBA | ⟨t_prop⟩ (ns) | median | sd | ⟨t⟩ bottom | ⟨t⟩ top | direct path to bottom array (ns) |
|---|---|---|---|---|---|---|---|---|---|
| −7.0 (RFR) | 0.132 | 0.078 | −0.844 | 11.9 | 2.6 | 22.0 | 9.0 | 45.7 | 0.38 |
| −2.0 | 0.128 | 0.098 | −0.803 | 14.3 | 4.2 | 23.9 | 11.0 | 45.0 | 0.66 |
| 2.0 | 0.123 | 0.118 | −0.764 | 17.3 | 6.5 | 25.1 | 13.7 | 43.9 | 0.89 |
| 10.0 | 0.121 | 0.148 | −0.704 | 20.4 | 9.6 | 26.8 | 16.3 | 44.6 | 1.34 |
| **26.4 (event)** | 0.117 | 0.212 | −0.576 | 26.7 | 16.2 | 29.0 | 22.3 | 43.1 | 2.26 |
| 40 | 0.116 | 0.275 | −0.451 | 30.6 | 20.7 | 29.5 | 26.7 | 41.2 | 3.03 |
| 60 | 0.111 | 0.358 | −0.283 | 34.8 | 24.9 | 30.8 | 32.7 | 38.4 | 4.16 |
| 73 (centre) | 0.112 | 0.414 | −0.172 | 35.7 | 26.1 | 30.7 | 35.7 | 35.6 | 4.89 |
| 100 | 0.109 | 0.527 | +0.053 | 36.1 | 26.8 | 31.8 | 42.5 | 30.3 | 6.41 |
| 120 | 0.113 | 0.601 | +0.201 | 33.9 | 25.2 | 31.7 | 46.1 | 25.7 | 7.54 |
| 140 | 0.114 | 0.666 | +0.332 | 30.8 | 21.7 | 32.1 | 48.3 | 22.0 | 8.67 |

Variants at the event depth: group index 1.9 → ⟨t⟩ 29.7 / sd 32.1 ns; λ_R = 30 cm → 27.3 / 29.5; PTFE R = 0.90 ("fast optics") → collection 0.101, ⟨t⟩ 22.1, sd 23.9, f_top 0.19. The bottom-heavy TBA (−0.58 at the event, −0.84 in the RFR, ≈ 0 at z = 100 cm) is the well-known consequence of total internal reflection at the surface; it is what makes TBA a z proxy.

The propagation spread (sd ≈ 29 ns) is *larger* than the triplet lifetime and than the recalled "5–10 ns" of the assignment; we regard the MC value as more defensible (it follows from R_PTFE ≥ 0.95 and the 1.5 m scale), and we quote the fast-optics variant as the lower bracket.

### 3.2 Emission-time model (script §2)

Each photon's emission time is
t_emit = [Exp(τ_rec) with probability f_rec, else 0] + [Exp(τ_s) with probability f_s, else Exp(τ_t)],  f_s = R/(1+R), R = I_s/I_t,
⟨t_emit⟩ = f_s τ_s + (1 − f_s) τ_t + f_rec τ_rec.

| model | R | τ_s | τ_t | f_rec, τ_rec | ⟨t_emit⟩ (ns) |
|---|---|---|---|---|---|
| NR_base | 1.5 | 3 | 24 | – | 11.40 |
| NR_lo (slowest NR in bracket) | 1.0 | 3 | 27 | – | 15.00 |
| NR_hi (fastest) | 3.0 | 3 | 21 | – | 7.50 |
| ER_base (drift field) | 0.5 | 3 | 26 | – | 18.33 |
| ER_fast | 0.6 | 3 | 24 | – | 16.12 |
| ER_slow | 0.4 | 3 | 27 | – | 20.14 |
| ER_rec (residual recombination delay) | 0.5 | 3 | 26 | 0.3, 15 ns | 22.83 |
| ER_zeroF (Hitachi/Kubota electrons, 0 V/cm) | 0.05 | 2.2 | 27 | 0.9, 45 ns | 66.32 |

The detected time is t = t_emit + t_prop (drawn from the optical-MC pool at the event's z, top and bottom arrays mixed in their MC proportions) + N(0, 1.5 ns). N_det ~ Poisson(540) (event) or Poisson(297) (150 keV NR); 10⁴ pulses per model and size. The raw S1 at z = 26.4 cm is ≈ 5 % larger than S1c in our optics (0.117/0.112); ignored.

### 3.3 Statistics

- f_p(W): fraction of photons within W ns of the first detected photon (W = 10, 20, 30, 50; 20 ns is quoted).
- ⟨t⟩ − t_first (mean arrival time after the first photon), also from the 10 ns digitised waveform (bin centres).
- Rise time t₁₀→t₅₀ of the cumulative photon count; absolute t₁₀, t₅₀ from the interaction time (for Sec. 6).
- Ideal template log-likelihood ratio: ln LR = Σ_i ln p_NR(t_i − s_NR) − Σ_i ln p_ER(t_i − s_ER), where p_h are the true detected-time densities (2×10⁶-sample histograms, 0.5 ns bins) and each hypothesis is profiled over its own start-time shift s ∈ [−6, 6] ns (0.5 ns steps). This is the best any template method could do with perfect templates at the event's position.
- Separation d = |μ_NR − μ_ER| / √((σ²_NR + σ²_ER)/2); AUC; ER leakage past the NR median (50 % NR acceptance) and past the NR 10th percentile (90 % acceptance). Gaussian conversions: LR at the NR median = e^{d²/2}, at the ER median e^{−d²/2}, midway 1; leak50 = ½ erfc(d/√2).

## 4. Results: separation for the parameter brackets (script §3)

At the event depth (`P077_results.json: pair_results`):

| pair | Δ⟨t_emit⟩ (ns) | size | d(f_p20) | leak50 | d(⟨t⟩) | d(⟨t⟩, 10 ns dig.) | d(ideal LR) | LR at NR median | LR at ER median |
|---|---|---|---|---|---|---|---|---|---|
| NR_base vs ER_base (bracket centre) | 6.9 | 540 | 3.41 | <10⁻³ | 3.73 | 3.76 | 4.54 | 5×10⁴ | 2.5×10⁻⁵ |
| | | 297 | 2.70 | 0.003 | 2.86 | 2.87 | 3.39 | 410 | 0.003 |
| NR_lo vs ER_fast (closest pair) | 1.1 | 540 | 1.15 | 0.123 | 0.58 | 0.58 | 1.39 | 3.1 | 0.36 |
| | | 297 | 0.85 | 0.197 | 0.42 | 0.42 | 0.99 | 1.8 | 0.59 |
| NR_hi vs ER_slow (farthest) | 12.6 | 540 | 6.17 | 0 | 6.99 | 7.04 | 8.35 | 4×10¹⁵ | 2×10⁻¹⁶ |
| | | 297 | 4.89 | 0 | 5.36 | 5.39 | 6.16 | 4×10⁸ | 3×10⁻⁹ |
| NR_base vs ER_rec | 11.4 | 540 | 5.59 | 0 | 6.04 | 6.07 | 7.37 | 2×10¹² | 9×10⁻¹³ |
| NR_base vs ER_zeroF | 54.9 | 540 | 15.9 | 0 | 20.7 | 20.7 | 23.0 | ≫ | ≪ |

Observations.
1. The 10 ns digitisation costs nothing for the mean-time statistic (d 3.73 → 3.76); the simple statistics carry 75–85 % of the ideal-template separation. Digitisation is not why LZ found PSD inconclusive.
2. With the recalled brackets, the *smallest* separation at 540 phd is 1.4σ (closest pair), the centre is 4.5σ, and any recombination-delayed ER (ER_rec, ER_zeroF) would be separated at ≥ 7σ. A zero-field-like 45 ns recombination component in the ER light is therefore excluded as a description of LZ's drift-field ERs by statement 1 alone, in agreement with Kubota's observation that the slow component is field-quenched.
3. The mean separation scales as √N: d(540)/d(297) = 1.34 (√(540/297) = 1.35), so the bracket-centre model predicts 3.4σ (ideal) or 2.7σ (f_p) separation already for 150 keV AmBe NRs, which would have made LZ's ¹³¹ᵐXe-template comparison of statement 2 an obvious mismatch. Statement 2 therefore rules out the bracket centre.

Optics variants (540 phd; `optics_variants`):

| optics | centre pair: d(f_p) / d(⟨t⟩) / d(ideal) | closest pair: d(f_p) / d(⟨t⟩) / d(ideal) |
|---|---|---|
| baseline (R_PTFE 0.95, sd 29 ns) | 3.41 / 3.73 / 4.54 | 1.15 / 0.58 / 1.39 |
| fast optics (R_PTFE 0.90, sd 24 ns) | 3.75 / 4.16 / 5.03 | 1.15 / 0.58 / 1.45 |
| no propagation spread (point detector) | 7.17 / 5.96 / 8.72 | 1.91 / 0.88 / 3.28 |

Photon propagation in the 1.5 m TPC halves the ideal separation (8.7 → 4.5–5.0σ): this is the quantitative content of LZ's remark that "photons produced further from PMTs experience a broader distribution of propagation times". It does not by itself make PSD inconclusive at 540 phd.

## 5. What the ER–NR timing difference must be (script §4)

Scan of the NR singlet fraction at fixed lifetimes (ER fixed at R = 0.5, τ_s = 3, τ_t = 26 ns; NR τ identical, R_NR varied; `delta_scan.csv`, Fig. 2 right):

| R_NR | f_s(NR) | Δ⟨t_emit⟩ (ns) | d ideal, 540 | d f_p, 540 | LR at NR median (540) | d ideal, 297 | LR (297) | d ideal, 100 phd |
|---|---|---|---|---|---|---|---|---|
| 0.55 | 0.355 | 0.5 | 0.20 | 0.28 | 0.9 | 0.13 | 1.0 | 0.08 |
| 0.60 | 0.375 | 1.0 | 0.41 | 0.50 | 1.05 | 0.35 | 1.1 | 0.21 |
| 0.70 | 0.412 | 1.8 | 1.07 | 0.99 | 2.0 | 0.78 | 1.5 | 0.46 |
| 0.80 | 0.444 | 2.6 | 1.64 | 1.37 | 4.2 | 1.26 | 2.5 | 0.68 |
| 1.00 | 0.500 | 3.8 | 2.51 | 2.01 | 29 | 1.92 | 7.2 | 1.08 |
| 1.25 | 0.556 | 5.1 | 3.46 | 2.75 | 450 | 2.53 | 31 | 1.49 |
| 1.50 | 0.600 | 6.1 | 4.17 | 3.27 | 8.5×10³ | 3.12 | 160 | 1.81 |
| 2.00 | 0.667 | 7.7 | 5.27 | 4.14 | 1.6×10⁶ | 3.88 | 2.6×10³ | 2.22 |
| 3.00 | 0.750 | 9.6 | 6.52 | 5.14 | 5×10⁹ | 4.92 | 3×10⁵ | 2.90 |

Rule of thumb: d(540 phd) ≈ 0.6 σ per ns of mean-emission-time difference (0.45 per ns for 297 phd, 0.3 per ns at 100 phd).

Interpretation of statement 2. If n_AmBe NRs of ~150 keV are compared with an ER template and the sample is "not inconsistent", the sample mean cannot be displaced by more than ~2σ of the sample mean, i.e. d(297) < 2/√n_AmBe:

| n_AmBe | d(297) < | Δ⟨t_emit⟩ < (ns) | d(540) < | LR at NR median < | LR at ER median > |
|---|---|---|---|---|---|
| 5 | 0.89 | 1.98 | 1.21 | 2.1 | 0.48 |
| 10 | 0.63 | 1.51 | 0.85 | 1.4 | 0.70 |
| 30 | 0.37 | 1.00 | 0.44 | 1.1 | 0.91 |

Even the loosest reading (5 events) limits the intrinsic ER–NR difference to Δ⟨t⟩ ≲ 2 ns, i.e. a singlet-fraction difference ≲ 0.08 (R_NR ≲ 0.7 against R_ER = 0.5) at these energies and 97 V/cm, and the best possible single-event separation at 540 phd to ≲ 1.2σ. Two caveats soften the bound in the direction of *more* allowed difference: the ¹³¹ᵐXe template is at 164 keV_ee (~850 phd) and the AmBe events are not at the event's position (position mismatch, Sec. 6, adds up to ~1σ of blur at ±10 cm), so LZ's comparison was less sensitive than our idealised one. Conservatively we quote Δ⟨t⟩ ≲ 2–3 ns and d(540) ≲ 1.2–1.8σ, LR ≲ 2–5.

Physical plausibility. At 70 keV (ER) and 250 keV (NR) both tracks are recombination-light dominated (recombination fractions ≈ 0.72, P010, and ≈ 0.85, P024). The classic large singlet/triplet contrast (Hitachi 1983) is between electrons and heavy ions at *zero* field, where the ER light is a 45 ns recombination continuum; at 97 V/cm the recombination-delay component is largely quenched (Kubota 1979) and both species emit through the same excimer states, differing only through the LET dependence of the singlet share. A difference of a few per cent in singlet fraction is what LUX (2018) and XMASS (2018) report as a few-ns difference in effective decay time (recalled, uncertain). Our bound is consistent with that literature and is what LZ's statement 1 requires.

## 6. Position dependence and template matching (script §5)

For the ER_base model with 540 photons (`position_dependence.csv`, Fig. 3 right):

| z (cm) | f_p20 ER | σ(f_p20) | f_p20 NR | ⟨t⟩−t_first ER (ns) | σ | ⟨t⟩ NR | rise 10–50 % ER (ns) |
|---|---|---|---|---|---|---|---|
| 2 | 0.417 | 0.024 | 0.524 | 36.6 | 1.7 | 30.0 | 20.1 ± 1.5 |
| 10 | 0.382 | 0.025 | 0.483 | 39.1 | 1.8 | 32.4 | 22.0 ± 1.6 |
| 26.4 | 0.314 | 0.024 | 0.399 | 43.9 | 1.8 | 37.3 | 25.6 ± 1.7 |
| 40 | 0.275 | 0.023 | 0.351 | 46.7 | 1.9 | 40.1 | 27.3 ± 1.7 |
| 60 | 0.245 | 0.024 | 0.315 | 49.1 | 1.9 | 42.5 | 28.7 ± 1.8 |
| 73 | 0.234 | 0.023 | 0.300 | 49.9 | 1.9 | 43.3 | 29.2 ± 1.8 |
| 100 | 0.208 | 0.022 | 0.267 | 52.3 | 2.0 | 45.8 | 30.7 ± 1.9 |
| 120 | 0.219 | 0.021 | 0.278 | 51.7 | 2.0 | 45.2 | 31.8 ± 1.8 |
| 140 | 0.245 | 0.021 | 0.310 | 50.3 | 1.9 | 43.7 | 32.0 ± 1.9 |

Moving from z = 26.4 to 100 cm shifts ⟨t⟩ by +8.4 ns and f_p20 by −0.106, i.e. 1.28× and 1.24× the *entire* ER–NR difference of the bracket-centre model (6.6 ns, 0.085). A depth-blind template would thus mis-classify by more than the effect sought; near the event the slopes are 0.24 ns/cm and −0.0034/cm, so a template built from events within ±5 cm (±10 cm) leaves a residual bias of 0.67σ (1.33σ) in ⟨t⟩ and 0.72σ (1.44σ) in f_p at 540 phd. Position-matched templates (as LZ did with ¹³¹ᵐXe "near the position") are therefore necessary, and must be matched to a few cm to keep the blur below ~0.5σ; the residual after matching within ±3 cm is ≈ 0.4σ, which does not by itself remove a ≥ 2σ physical separation but does erase a ≤ 1σ one.

## 7. RFR origin versus z = 26.4 cm (script §6)

For an ER-like S1 of 540 detected photons (`rfr`, Fig. 4):

| origin | rise 10→50 % (ns) | t₁₀ from interaction (ns) | t₅₀ | f_top | TBA | f_p20 | direct path to bottom array (ns) |
|---|---|---|---|---|---|---|---|
| z = −7 cm (RFR middle) | 16.1 ± 1.4 | 2.9 | 19.0 | 0.078 | −0.844 | 0.485 | 0.38 |
| z = −2 cm | 17.7 ± 1.4 | 3.7 | 21.4 | 0.098 | −0.803 | 0.457 | 0.66 |
| z = +2 cm | 20.1 ± 1.5 | 4.6 | 24.7 | 0.118 | −0.764 | 0.417 | 0.89 |
| z = 10 cm | 22.0 ± 1.6 | 5.9 | 27.9 | 0.148 | −0.704 | 0.382 | 1.34 |
| z = 26.4 cm (event) | 25.6 ± 1.7 | 9.1 | 34.7 | 0.212 | −0.576 | 0.312 | 2.26 |

The direct-path offset is small (33.4 cm / 17.74 cm ns⁻¹ = 1.9 ns), but the pulse shape changes because the *fraction* of early, direct-or-few-bounce light at the bottom array is much larger for an RFR origin: the rise time shortens by 9.5 ns (6.2σ at 540 phd), f_p20 rises by 0.17 (7.1σ), and the TBA moves from −0.58 to −0.84 against σ_TBA = 2√(f(1−f)/540) = 0.035 (7.6σ). For an RFR-MSSI event 87 % of the light (471 of 540 phd) is RFR light, so the separation remains ≥ 5σ. LZ's statements 3–4 (shorter rise times for all near-cathode events; TBA and hit pattern consistent with z = 26.4 cm) are thus expected to be decisive at this pulse size, and our model reproduces the qualitative ranking (RFR < event depth in rise time, Fig. 4). Depth discrimination works because it moves ~20 % of the photons between arrays and between early/late arrival; ER/NR discrimination fails because it moves ≲ 5 % of the photons between a 3 ns and a 26 ns exponential.

## 8. Single-event likelihood-ratio table (script §7)

Gaussian: d = 0.5 / 1.0 / 1.5 / 2.0 / 3.0 → LR at the NR median 1.13 / 1.65 / 3.08 / 7.39 / 90; at the ER median 0.88 / 0.61 / 0.33 / 0.135 / 0.011; midway 1; ER leakage past the NR median 0.31 / 0.16 / 0.067 / 0.023 / 0.001.
Direct (toy) values at 540 phd: closest bracket pair LR = 3.1 (NR median) / 0.36 (ER median) / 1.05 (midway); allowed by statement 2 (Sec. 5): LR ≤ 1.1–2.1 (n_AmBe = 30–5), conservatively ≤ 2–5.

Consequence: with the odds against an ER origin already at ≳ 10²–10³ from the S2/S1 band position (P010: 6.4σ charge deficit; dossier: 6.7σ below the ER median), a PSD LR of ≤ 2–5 in either direction is negligible in the ER-vs-NR question; and PSD is silent on the NR-vs-wall-MSSI question, which is the one that matters (P004, P024).

## 9. What LZ could still do

- A full position-matched template likelihood with the actual per-channel hit times (not a summed waveform) against ¹³¹ᵐXe / ⁸³ᵐKr-free ER samples and AmBe NRs *reweighted to the event's z and r* using the optical model, reporting ln LR with its calibration-sample uncertainty. Given Sec. 5 this will return |ln LR| ≲ 1.
- Publish the S1 timing statistic (e.g. f_p or fitted τ_eff) of the event together with its ER and NR calibration distributions at the same position and size; this is a two-number release, not a waveform release, and would let the community verify the "inconclusive" statement. We do not file a data request: the waveform and per-channel times are not a public dataset, and no result here depends on them.
- Use the same machinery for the RFR-vs-FV question, where the expected separation is ≥ 5σ: publishing the event's rise time and TBA with the near-cathode and same-depth distributions would turn statement 3 into a quantitative exclusion.

## 10. Validation and robustness

- Collection efficiency 0.109–0.117 across z with the tuned optics vs LZ g₁ = 0.110 ± 0.002; TBA spans −0.84 to +0.33 (bottom-heavy through TIR), qualitatively as in LUX/LZ.
- √N scaling: d(540)/d(297) = 1.34 vs 1.35 expected.
- Digitised vs undigitised mean time: 3.76 vs 3.73 (no information loss at 10 ns sampling for a mean-time statistic).
- Optics variants: ideal separation for the bracket centre 4.5 (baseline) → 5.0 (R_PTFE 0.90) → 8.7 (no propagation); the closest pair 1.39 → 1.45 → 3.28. Conclusions in Sec. 5 change by ≤ 10 % between baseline and fast optics; the point-detector case would tighten the bound on Δ⟨t⟩ by ×0.5.
- TTS (by hand, not simulated): raising σ_TTS from 1.5 to 3.8 ns adds in quadrature to a per-photon spread of ≈ √(24² + 29²) ≈ 38 ns: 38.03 → 38.19 ns, a 0.4 % change in the separations.
- Rayleigh 30 vs 35 cm and group index 1.9 change ⟨t_prop⟩ by +0.6 and +3.0 ns and sd by ≤ 3 ns (not propagated to separations; by the quadrature argument the effect is ≤ 5 %).
- The ideal LR is profiled over each hypothesis's start time (±6 ns); the profiling makes it a fair "best case" rather than an oracle that knows the interaction time.
- Statistic choice (from `pair_results`): f_p windows 10/20/30/50 ns give d = 2.06/3.41/3.68/3.41 for the bracket centre at 540 phd (1.78/2.70/2.83/2.56 at 297 phd) and 0.86/1.15/0.99/0.55 for the closest pair; 20–30 ns is optimal; ⟨t⟩ (3.73) is marginally better than any f_p for the centre but worse (0.58) for the closest pair, whose difference is mostly in the singlet share rather than in the tail.

## 11. Failed or abandoned approaches

- First optical model (R_PTFE 0.97, R_arr 0.5, ε 0.25, no grids): collection 0.28 (2.5× g₁), ⟨t_prop⟩ 36 ns; retuned to g₁ (Sec. 3.1). The retuning changed ⟨t_prop⟩ only from 36 to 27 ns.
- First Δ scan varied τ_t at fixed R = 1.0 to set the mean; the changed *shape* (singlet share 0.5 vs 0.33) gave d ≈ 1.2–1.7 even for Δ⟨t⟩ = 0.5 ns and made the interpolation for the allowed Δ degenerate. Replaced by the pure singlet-fraction scan at fixed lifetimes, which is also the physical picture (same excimer states, different population ratio).
- We did not attempt to reproduce the content of LZ's dedicated PSD paper (arXiv:2603.26877, cited by LZ), which we have not read.

## 12. Figures

- `figures/P077_fig1_fp_distributions.png` — f_p(20 ns) distributions at 540 and 297 phd for the bracket models (ER drift-field, NR, zero-field ER, slowest NR, fastest ER); right: ideal template ln LR at 540 phd for the bracket centre and the closest pair.
- `figures/P077_fig2_roc_and_scan.png` — left: ROC (NR acceptance vs ER leakage) of the ideal LR for four pairs at 540 and 297 phd; right: separation vs the NR–ER mean-emission-time difference at 540, 297 and 100 phd, with the values allowed by LZ's AmBe statement.
- `figures/P077_fig3_depth_dependence.png` — propagation time (all, bottom, top, sd) vs z from the optical MC; TBA vs z; ER and NR mean-arrival-time statistic vs z with the 540-phd single-event spread.
- `figures/P077_fig4_rise_time_rfr.png` — 10→50 % rise-time distributions for 540-photon ER-like S1s from the RFR (z = −7 cm), the event depth and z = 100 cm.

## 13. References

LZ Collaboration, arXiv:2609.02823 (2026), supplement "Waveform Analysis". A. Hitachi et al., Phys. Rev. B 27, 5279 (1983). S. Kubota et al., Phys. Rev. B 20, 3486 (1979). D. S. Akerib et al. (LUX), Phys. Rev. D 97, 112002 (2018). K. Abe et al. (XMASS), JINST 13, P12032 (2018). J. V. Dawson et al., Nucl. Instrum. Meth. A 545, 690 (2005). V. N. Solovov et al., Nucl. Instrum. Meth. A 516, 462 (2004). Corpus: dossier 00; P004, P009, P010, P022, P024, P041, P042.

## 14. Tools and provenance

Mirrors `output/provenance/P077.json`.
- Agent tools: Read (PAPER_GUIDE; dossier; ledger page 1 + pandas scan; P024, P009, P010, P041, P022, P042; tex l. 684–697; lzcommon l. 18–77; four figures), Bash (grep tex/ledger/lzcommon/details; tuning run; two full runs; log greps), Write (script, details, JSON, paper), Edit (script ×9, paper trims).
- Software: python 3.12.13; numpy 2.5.3 (vectorised photon tracking, default_rng, bincount statistics, histograms); matplotlib 3.11.2 (Agg); pandas 3.0.5 (ledger scan only); common/lzcommon.py (LZ dict: S1c, z, g₁). scipy not used; no nestpy calls (yields via P009's contour scale).
- Script: `output/code/P077_s1_psd.py`; command `.venv/bin/python output/code/P077_s1_psd.py` (and `P077_TUNE=1 .venv/bin/python output/code/P077_s1_psd.py` for the optics tuning table).
- Local inputs: tex (Waveform Analysis l. 687–695; detector l. 77–88; event l. 166; refs l. 398–403); dossier §2 table; ledger; P024/P009/P010/P041/P022/P042 papers; P022 and P042 details (geometry lines); lzcommon.py.
- Recalled knowledge: 19 items in `recalled_inputs.json` (geometry 2, optics 8, timing 8, yields 1); 9 uncertain.
- Datasets: none. Data requests: none (waveform not public; deliberately not filed). WimPyDD files: none.
