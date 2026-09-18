# P064 — Nuclear-recoil quenching at 250 keV: Lindhard theory, the NEST power law and the energy-scale systematic of the LZ event

Research record. Script: `output/code/P064_quenching.py` (run from the root with `.venv/bin/python`; stdout in `run_stdout.txt`). Results: `results.json`, `table1_L_Nq_beta.csv`, `table3_event_energy_by_model.csv`, `table3b_anchored_beta.csv`, `table3c_lindhard_k.csv`, `table3d_kink.csv`, `table4_partition.csv`, `table4b_degeneracy.csv`, `nestpy_validation.csv`; figures in `figures/`.

## 1. Motivation and framework

LZ reconstructs the event (S1c = 540.1 phd, S2c = 9268 phd) as a 248 ± 23 (stat) ± 23 (sys) keV nuclear recoil using NEST v2.4.5 tuned to D-D (0–74 keV) and AmBe (0–330 keV) calibrations (tex l. 92, 600–606, Table S5). The total-quanta law N_q = α E^β (α = 11.2 keV^{-1/β}, β = 1.1) is the energy scale; the charge yield Q_y = 1/[ς (E + ε)^{p(E)}] with the new break p(E) = 0.5 + a ln[1 + b(E − E₀)] above E₀ = 74.7 keV (a = 0.0230, b = 0.0289) sets the light/charge partition. The corpus has found (P009) that LZ's own constant-energy contours imply N_q ≈ 11.32 E^1.112, 7.7 % above Table S5 at 250 keV, and (P043) that the quoted gain uncertainties explain only 3.8 keV of the ± 23 keV systematic, so that a ≈ 10 % light-yield uncertainty at 250 keV is required to close it. P024 found that without the p(E) break the event would sit 5σ below the NR median. We ask what nuclear-recoil quenching theory says about all three.

The total quanta encode the Lindhard quenching factor L(E) = N_q W/E, the fraction of recoil energy that ends in electronic excitation (W = energy per quantum, 13.7 eV in NEST v2, 13.5 eV also used [recalled, likely]).

## 2. Equations

**Lindhard (1963), Lewin–Smith parametrisation [recalled, certain]:**
L(E) = k g(ε)/(1 + k g(ε)),  ε = 11.5 E[keV] Z^{−7/3},  g(ε) = 3 ε^0.15 + 0.7 ε^0.6 + ε,  k = 0.133 Z^{2/3} A^{−1/2}.
For Xe (Z = 54, A = 131.29): k = 0.1658; ε(248 keV) = 0.259.
N_q,Lind(E) = L(E) E/W.  Local exponent β_eff(E) = d ln N_q/d ln E = 1 + d ln L/d ln E.

**NEST v2 NR mean yields [structure recalled from the NEST source, likely; validated below]:**
N_q = α E^β;  ς = γ F^δ (ρ/2.9)^0.3;  Q_y = [ς (E + ε)^{p(E)}]^{−1} [1 − (1 + (E/ζ)^η)^{−f₁}];  L_y = N_q/E − Q_y;  N_e = Q_y E;  N_ph = L_y E [1 − (1 + (E/θ)^ι)^{−f₂}].
With the LZ parameters at F = 96.5 V/cm: ς = 0.04076.

**Thomas–Imel box model [certain]:** N_e/N_i = ln(1 + ξ)/ξ, ξ = ς N_i/4, so N_e = (4/ς) ln(1 + ς N_i/4) with N_i = N_q/(1 + N_ex/N_i). For ξ ≫ 1 the electron count grows only logarithmically with N_i.

**Anchored slope variation:** N_q(E) = N_q,S5(E₀) (E/E₀)^β′, so that the D-D-calibrated yields below 74.7 keV are untouched; then d ln E/dβ′ = −ln(E/E₀)/β at fixed observed N_q.

## 3. Inputs

| Input | Value | Source |
|---|---|---|
| S1c, S2c, g₁, g₂ | 540.1 phd, 9268 phd, 0.110, 34.5 | LZ tex l. 88, 165; `lz.LZ` |
| N_q,event, N_ph, N_e | 5178.6, 4910, 268.6 | computed |
| Table S5 NR parameters incl. a, b, E₀ | see `lz.NEST_NR_LZ` | LZ Table S5 (tex l. 654–685) |
| NEST default parameters | `lz.NEST_NR_DEFAULT` | Table S5 "NEST Default" column |
| p(E) break form; Aprile 2006 / Lenardo 2015 justification | tex l. 600–606 | LZ supplement |
| D-D range 0–74 keV; AmBe 0–330 keV; AmBe sparse near 250 keV | tex l. 92, 691 | LZ |
| Contour scale N_q = 11.3245 E^1.11167, W = 13.46 eV | P009 `results.json` | corpus |
| N_ex/N_i = 1.399 at 248 keV (nestpy), N_i = 2012, N_e = 301 | P024 details §2 | corpus |
| Gain budget 3.8 keV; 4 % scale → 8.7 keV; σ_s = 10.4 % needed | P043 | corpus |
| AmBe points in Fig. 2: 19 at 500–580 phd, max S1c = 599.6 phd | P024 `ambe_points_fig2.csv` | corpus |
| Field 96.5 V/cm | `lz.DRIFT_FIELD_VCM` | lzcommon |
| k_Lindhard = 0.1658 | formula | certain |
| Lindhard k ≈ 0.139 (global fit, W = 13.7 eV) | Lenardo et al. 2015 | recalled, likely |
| Lindhard k ≈ 0.17 (LUX D-D total-quanta fit) | LUX 2016 D-D paper | recalled, uncertain |
| Aprile 2006 L_eff at 10.4–56.5 keV, ≈ 0.2, roughly flat | PRL 97, 081302 | recalled, likely |
| Manzur 2010 (4–66 keV), Plante 2011 (3–66 keV) L_eff rising 0.09→0.19 | PRC 81, 025808; PRC 84, 045805 | recalled, likely |
| NR N_ex/N_i ≈ 1 | Dahl thesis / Sorensen & Dahl 2011 | recalled, likely |
| AmBe neutron endpoint ≈ 11 MeV | AmBe spectrum | recalled, likely |

## 4. Validation

Analytic NEST yields vs `lz.nest_nr_yields` (nestpy 2.1.1) at 5–330 keV for both parameter sets: maximum relative deviation 0.11 % (`nestpy_validation.csv`). At 248 keV: N_ph = 4520 (nestpy 4525), N_e = 300.8 (301.1); p(248) = 0.5412.

Cross-checks against the corpus: N_q offset of the contour scale at 248 keV 1.078 (P009: 1.077); E_comb = 264.7 (Table S5) and 247.2 keV (contour) vs P009 264.4/247.2; N_e(248) = 300.8 vs P024 301; d log₁₀N_e/dE = 0.00058 dex/keV (P043 0.00058); no-break shift 0.0995 dex = 3.18σ → event 4.72σ low vs P024's MC 5.02σ (the difference is the wider MC band without the break, 0.033 dex, and slice mixing).

## 5. Results

### 5.1 Lindhard vs the power laws (Table 1)

| model | N_q(50) | N_q(74.7) | N_q(150) | N_q(248) | N_q(300) | β_eff(248) | L_W=13.7(248) |
|---|---|---|---|---|---|---|---|
| Lindhard k = 0.166, W = 13.7 | 942 | 1493 | 3349 | 6038 | 7562 | 1.179 | 0.334 |
| Lindhard k = 0.166, W = 13.5 | 956 | 1515 | 3398 | 6127 | 7674 | 1.179 | 0.338 |
| Lindhard k = 0.133 (−20 %) | 794 | 1264 | 2854 | 5176 | 6498 | 1.192 | 0.286 |
| Lindhard k = 0.199 (+20 %) | 1074 | 1699 | 3787 | 6792 | 8488 | 1.168 | 0.375 |
| Lindhard k = 0.139 (Lenardo-15 fit) | 826 | 1313 | 2960 | 5362 | 6729 | 1.189 | 0.296 |
| Lindhard k = 0.174 (LUX D-D fit) | 976 | 1546 | 3462 | 6235 | 7804 | 1.176 | 0.344 |
| NEST Table S5, 11.2 E^1.1 | 828 | 1288 | 2773 | 4821 | 5944 | 1.100 | 0.266 |
| LZ contour scale, 11.32 E^1.112 | 876 | 1369 | 2972 | 5198 | 6423 | 1.112 | 0.287 |
| NEST default, 11.0 E^1.1 | 813 | 1265 | 2723 | 4735 | 5837 | 1.100 | 0.262 |

Lindhard with the textbook k = 0.166 gives L = 0.258, 0.274, 0.306, 0.334, 0.345 at 50, 74.7, 150, 248, 300 keV: a slow rise with mean d ln L/d ln E = 0.163 over 50–300 keV, i.e. β_eff = 1.15 (75 keV) → 1.18 (248 keV) → 1.19 (300 keV). The NEST power law β = 1.1 is therefore *shallower* than Lindhard by Δβ ≈ 0.06–0.08 in this range; the LZ contour scale (β = 1.112) is intermediate. In absolute terms, Table S5 corresponds to an effective L (W = 13.7 eV) of 0.227 at 50 keV and 0.266 at 248 keV, 12 % and 20 % below Lindhard k = 0.166. Expressed as an energy-dependent k, Table S5 requires k = 0.136 at 74.7 keV falling to 0.120 at 248 keV; the contour scale k = 0.147 → 0.133. Lindhard with k = 0.166 would need W = 17.2 eV (Table S5) or 15.9 eV (contour) at 248 keV — unphysical — so the LXe NR yields at all energies lie below k = 0.166 Lindhard, as the global-fit value k ≈ 0.139 [recalled, likely] already indicated at low energy. Lindhard k = 0.139 gives 5362 quanta at 248 keV, 11 % above Table S5 and 3 % above the contour scale; k = 0.133 (−20 %) reproduces the contour scale almost exactly (5176 vs 5198).

**Recalled measurements as anchors (qualitative only).** The S1-only relative scintillation efficiency L_eff (referenced to 122 keV γ at zero field) measured by Aprile 2006 (10.4–56.5 keV, ≈ 0.2, roughly flat), Manzur 2010 (4–66 keV) and Plante 2011 (3–66 keV, rising from ≈ 0.09 to ≈ 0.19) is not the Lindhard L: it multiplies L by the NR photon fraction and divides by the γ light yield, so it cannot be compared to L without a recombination model; we do not use these numbers quantitatively. Lenardo et al. 2015 fit total quanta with Lindhard, k ≈ 0.139 (W = 13.7 eV) [likely], and LZ cites the same paper for "a preference for a larger power-law exponent at high recoil energies" — consistent with our finding that Lindhard's slope exceeds β = 1.1. LUX D-D (2016) total quanta were fit with k ≈ 0.17 [uncertain]. All these measurements stop at ≤ 74 keV (D-D) or ≤ 66 keV (L_eff); none constrains the slope between 75 and 250 keV directly.

### 5.2 Energy of the event under each model (Table 3)

Combined-quanta energy from N_q,event = 5179 (and S1-only from N_ph = 4910 with the LZ N_e(E)):

| model | E_comb [keV] | E_S1 [keV] |
|---|---|---|
| NEST Table S5 | 264.7 | 266.5 |
| LZ contour scale | 247.2 | 248.5 |
| NEST default | 269.1 | 271.0 |
| Lindhard k = 0.166, W = 13.7 / 13.5 | 217.7 / 215.0 | 218.4 / 215.6 |
| Lindhard k = 0.139 | 240.8 | 242.0 |
| Lindhard k = 0.174 | 211.7 | 212.3 |
| Lindhard k = 0.133 / 0.199 | 248.1 / 196.5 | 249.4 / 196.8 |

Half-range over all models 36 keV; over the five calibrated/plausible models (Table S5, contour, Lindhard 0.139/0.166/0.174) 26.5 keV (sd 21.8). This is the spread one would quote *without* any calibration above 74 keV; it is not LZ's situation, because D-D fixes the yields at ≤ 74 keV.

### 5.3 Anchored systematics (Tables 3b–3d, Fig. 3)

*Slope β′ above the D-D anchor* (N_q fixed at 74.7 keV to Table S5): β′ = 1.00/1.05/1.08/1.10/1.112/1.12/1.15/1.20 → E_comb = 300.4/281.1/271.0/264.7/261.2/258.8/250.5/238.2 keV. Analytic d ln E/dβ′ = −ln(248/74.7)/1.1 = −1.09, so δβ′ = 0.05 ↔ 13.5 keV, and β′ ∈ [1.05, 1.15] ↔ ± 15.3 keV. The paper's ± 23 keV (9.3 %) corresponds to δβ′ = 0.085; P043's required 10.4 % light-yield scale corresponds to δβ′ = 0.087. **The ± 23 keV systematic is equivalent to not knowing the slope of the yield curve between 75 and 250 keV to better than ± 0.085, about the difference between the power law (1.10) and Lindhard (1.16–1.18).**

*Lindhard k ± 20 %.* Unanchored (W = 13.7 eV fixed): k = 0.133–0.199 → E = 248–197 keV, a ± 26 keV lever. Anchored to Table S5 at 74.7 keV (W floating to 13.44–18.07 eV): E = 244.2–249.2 keV, i.e. **± 2.5 keV only**: the D-D anchor removes k as a systematic, because the Lindhard *shape* between 75 and 250 keV barely depends on k (β_eff(75→248) = 1.175–1.155). Crucially, any anchored Lindhard curve predicts N_q(248) = 1.068–1.094 × Table S5 (1.080 for k = 0.166), which is precisely the +7.7 % contour-scale offset found by P009, and gives E = 244–249 keV, the paper's 248 keV. The energy scale LZ actually used is therefore Lindhard-like above the D-D anchor; the printed β = 1.1 power law, extrapolated from 74.7 keV, is the outlier (264.7 keV).

*Kink at E₀.* A slope change from 1.1 (below 74.7 keV) to β_hi above: β_hi = 1.0/1.05/1.16/1.2/1.3 → 300/281/248/238/218 keV. The Aprile 2006 "abrupt change in electronic stopping power" cited by LZ cannot be checked by us at 75 keV: that measurement covered 10.4–56.5 keV [recalled, likely], entirely below E₀. We treat the "kink" as a slope systematic of the same size as the β′ scan.

*Budget.* Gains (P043: 3.8 keV) ⊕ slope (± 15.3 keV for β′ = 1.05–1.15) = 15.8 keV; adding a form uncertainty equal to the power-law-vs-Lindhard offset (6 % in N_q → 13.5 keV) gives 20.8 keV, against the paper's 23 keV. A slope uncertainty of ± 0.085 alone reproduces 23 keV.

### 5.4 Partition: the p(E) break and Thomas–Imel recombination (Table 4, Fig. 4)

| E [keV] | p(E) | N_e break | N_e no break | N_e default | N_e TI box (N_ex/N_i = 1.40) | N_e TI box (1.0) | f_e break / no break |
|---|---|---|---|---|---|---|---|
| 74.7 | 0.500 | 198.2 | 198.2 | 212.5 | 183.2 | 198.5 | 0.154 / 0.154 |
| 150 | 0.527 | 253.6 | 290.2 | 312.7 | 250.0 | 266.6 | 0.091 / 0.105 |
| 248 | 0.541 | 300.8 | 378.2 | 408.3 | 301.0 | 318.1 | 0.062 / 0.078 |
| 300 | 0.546 | 319.9 | 417.5 | 451.0 | 320.7 | 337.9 | 0.054 / 0.070 |
| 330 | 0.549 | 329.8 | 438.6 | 473.9 | 330.6 | 347.9 | 0.050 / 0.066 |

At 248 keV: Q_y = 1.21 e/keV with the break vs 1.53 without; L_y = 18.2 ph/keV; the break moves 77 electrons into 77 photons (+1.7 % in N_ph), so its effect on the S1 energy scale is negligible (E_S1 266.5 vs 270.3 keV in P009), but it lowers log₁₀N_e by 0.0995 dex = 3.18σ of the 0.0313 dex band width, which turns P024's 1.54σ into 4.72σ (P024 MC: 5.0σ). The event's electron fraction is 268.6/5179 = 5.2 %, against 6.2 % (break) and 7.8 % (no break) at 248 keV.

**Is the break physics?** The Thomas–Imel box model with NEST's own recombination parameter ς = 0.04076 and nestpy's N_ex/N_i = 1.40 gives N_e(248) = 301.0 — identical to the broken power law's 300.8 — and N_e(300) = 320.7 vs 319.9, N_e(330) = 330.6 vs 329.8. With the measured N_ex/N_i ≈ 1 it gives 318 (+6 %). Both TI variants predict N_e(248)/N_e(74.7) = 1.60–1.64, vs 1.52 with the break and 1.91 for the unbroken √(E+ε) law. The reason is the logarithmic saturation N_e ≈ (4/ς) ln(ς N_i/4) once ξ = ς N_i/4 ≫ 1 (ξ = 6.6 at 74.7 keV, 22 at 248 keV): the exponent p must drift upward from 0.5 towards 1. Fitting LZ's p(E) form to the TI curve over 75–330 keV gives (a, b) = (0.0121, 0.167) for N_ex/N_i = 1.40 (rms 1.4 % in N_e) and (0.0372, 0.0076) for 1.0 (rms 0.08 %), bracketing LZ's (0.0230, 0.0289); the fitted N_e(248) = 300.8 and 318.2. The band slope d log₁₀N_e/d log₁₀N_ph at 248 keV is 0.286 with the break, 0.283–0.297 for TI, 0.453 without the break. **The break is the Thomas–Imel recombination saturation already implicit in NEST's recombination parameter, not a new stopping-power feature**; the "abrupt change in electronic stopping power" (Aprile 2006) is neither needed nor testable at 75 keV with that dataset.

**AmBe cannot fix the light-yield scale (Table 4b).** Rescaling N_q by s with N_e(E) fixed shifts the band at fixed S1c = 540 phd by −0.0098 dex (s = 1.077, the contour scale) or −0.0125 dex (s = 1.10), 1.4–1.8 × the ± 0.007 dex median precision of the 19 AmBe points (P024). But refitting (a, b) restores N_e as a function of N_ph to rms 0.0015 dex (max 0.0053) for s = 1.077 [(a, b) → (0.0306, 0.0137)] and 0.0021 dex (max 0.0076) for s = 1.10 [(0.0347, 0.0105)], with the event energy moving 264.7 → 247.4 → 242.7 keV. The AmBe band therefore constrains only N_e(N_ph), and the break parameters absorb any light-yield rescaling; the energy assignment above 74 keV rests on the yield-slope prior. The AmBe recoil endpoint (E_R,max = 0.0301 × 11 MeV = 333 keV) would sit at S1c ≈ 696 (Table S5), 756 (contour) or 904 phd (Lindhard k = 0.166), but the AmBe points shown in Fig. 2 end at ≈ 600 phd (P024 digitisation max 599.6 phd) and the neutron spectrum tail is soft, so the endpoint provides no usable scale constraint either. A ≈ 10 % light-yield uncertainty at 250 keV (P043) is thus not only plausible but expected.

## 6. Robustness

- W = 13.5 vs 13.7 eV: Lindhard N_q changes by 1.5 %; anchored results are W-independent by construction.
- Field 90–100 V/cm changes ς by ∓ 0.3 % (δ = −0.0533), negligible for the partition (P024 found the same).
- The TI comparison uses NEST's ς; with the NEST default ς = 0.0376 the TI N_e(248) would be 322 (N_ex/N_i = 1.40); the conclusion (logarithmic saturation, slope ≈ 0.29) is unchanged.
- Choice of anchor: anchoring at 60 or 74.7 keV changes the β′ elasticity by ln(248/60)/ln(248/74.7) = 1.18.
- The 10.4 % scale ↔ δβ′ = 0.087 equivalence uses ln(248/74.7) = 1.20; a 5 % scale would be δβ′ = 0.042.

## 7. Failed / abandoned

- Converting recalled L_eff values (Aprile 2006, Manzur 2010, Plante 2011) into Lindhard L requires the zero-field NR photon fraction and the 122 keV γ light yield, both recalled only roughly; abandoned as unreliable, used qualitatively only.
- A stopping-power (SRIM-like) calculation of the electronic/nuclear partition was not attempted (no tables available offline); Lindhard's g(ε) is used as the theory reference.
- First run: `tee` opened its log before the script created the output directory; rerun with redirection.

## 8. Figures

- `figures/P064_fig1_L_beta.png` — left: L(E) = N_q W/E for Lindhard (k = 0.133/0.139/0.166/0.174/0.199) and the three NEST power laws with W = 13.7 eV; D-D range (orange) and AmBe range (purple) shaded, event at 248 keV. Right: local exponent β_eff(E).
- `figures/P064_fig2_Nq_event_energy.png` — left: N_q(model)/N_q(Table S5) vs energy; right: event energy under each model (bars: combined quanta; dots: S1-only), with the paper's 248 ± 23 keV band.
- `figures/P064_fig3_slope_systematics.png` — left: event energy vs anchored slope β′ (Table S5 and contour slopes marked); right: Lindhard k ± 20 % unanchored vs D-D-anchored.
- `figures/P064_fig4_partition.png` — N_e(E) for break/no break/default/TI box; electron fraction with the event; LZ's p(E) against p(E) forms fitted to the TI box model.

## 9. Discussion

Three corpus puzzles have one common explanation. (i) P009's +7.7 % offset between Table S5 and the paper's energy contours is what any Lindhard-shaped curve anchored to the D-D data predicts at 248 keV (+6.8 to +9.4 %); LZ's energy scale behaves like Lindhard above 75 keV while the public β = 1.1 power law does not. (ii) P043's "10 % light-yield uncertainty" is a slope uncertainty δβ′ ≈ 0.085 over the factor-3.3 lever arm from the D-D anchor — the same size as the power-law-vs-Lindhard discrepancy — and the AmBe band cannot reduce it because the break parameters absorb any light-yield rescaling. (iii) P024's "no break → 5σ" is not a fragile tuning: the break reproduces the Thomas–Imel saturation of the electron count (N_e ≈ (4/ς) ln(ς N_i/4)), which is fixed once ς is calibrated at low energy. The physical picture for a 248 keV Xe recoil is a track with ≈ 2000 ion pairs and ≈ 2800 excitons in a box smaller than the diffusion scale, in which 94 % of the electrons recombine; the 5.2 % electron fraction of the event is 1.5σ below the 6.2 % median, exactly LZ's statement.

The energy is the weak point: E = 248 keV is a statement about the yield slope, and 262–265 keV (Table S5) or 218–241 keV (Lindhard k = 0.166–0.139 unanchored) are the alternatives if the anchor or the slope is doubted. For the DM interpretation the consequences are those P043 listed: acceptance is a function of S1c and does not move; δ_max moves ± 2 keV; only the large-δ (≥ 370 keV) inelastic preference is sensitive.

## 10. References

- LZ Collaboration, arXiv:2609.02823 (2026), main text and Supplemental Material (detector response, Table S5).
- J. Lindhard, V. Nielsen, M. Scharff, P. V. Thomsen, Mat. Fys. Medd. Dan. Vid. Selsk. 33, 10 (1963).
- J. D. Lewin, P. F. Smith, Astropart. Phys. 6, 87 (1996).
- J. Thomas, D. A. Imel, Phys. Rev. A 36, 614 (1987).
- E. Aprile et al., Phys. Rev. Lett. 97, 081302 (2006).
- A. Manzur et al., Phys. Rev. C 81, 025808 (2010); G. Plante et al., Phys. Rev. C 84, 045805 (2011).
- B. Lenardo et al., IEEE Trans. Nucl. Sci. 62, 3387 (2015).
- D. S. Akerib et al. (LUX), arXiv:1608.05381 (2016).
- M. Szydagis et al., NEST review, arXiv:2211.10726 (2024).
- Corpus: P009, P024, P038, P043; dossier 00.

## 11. Tools and provenance

Mirrors `output/provenance/P064.json`. Agent tools: Read × 16 (PAPER_GUIDE; dossier; ledger p.1; papers P009, P024, P043, P038; tex l. 84–123 and 586–690; lzcommon l. 18–77, 255–294; Fig2 PNG; provenance/P043.json; own figures × 3); Bash × 11 (ledger listing/grep tex/grep lzcommon; ls work dirs + versions; nestpy validation snippet; grep of P009/P024/P043 details; P024 AmBe CSV inspection; script run × 2; word-count/JSON checks × 4); Write × 5 (script, details, provenance, paper × 2); Edit × 8 (paper trims × 6, provenance × 1, details × 1). Software: python 3.12.13, numpy 2.5.3, scipy 1.18.1 (optimize.brentq, optimize.least_squares), pandas 3.0.5, matplotlib 3.11.2, nestpy 2.1.1 (validation via lzcommon.nest_nr_yields), common/lzcommon.py. Recalled knowledge: 13 items (listed in §3). Datasets: none. Data requests: none. WimPyDD files: none.
