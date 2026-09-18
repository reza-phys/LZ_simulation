# P005 — Would XENONnT and PandaX-4T have seen it? Expected high-energy counts in their published regions of interest

*Research record. Simulated arXiv date 2026-09-03. Category XEXP (hep-ph, cross-list hep-ex). Author profile: phenomenologists who recast direct-detection results. All numbers below are produced by `output/code/P005_other_xenon_expectations.py` and stored in `output/work/P005/summary.json`, `counts_table.csv`, `fractions_table.csv`, `spectra_per_tyr.csv`, `spectra_normalised_LZ1.csv`, `fig1_L10_digitised.csv`, unless marked [paper] (quoted from arXiv:2609.02823) or [recall] (training knowledge, with reliability flag).*

## 1. Motivation and framework

LZ (arXiv:2609.02823) reports one nuclear-recoil-like event at E_R = 248 ± 23 ± 23 keV in 2.84 t·yr and fits it, for a representative high-energy model (L10^s, m = 1000 GeV), with a signal of 1.0 (+1.4, −0.7) events [paper, Table I]. The natural question for the two other tonne-scale liquid-xenon experiments — XENONnT (3.1 t·yr, PRL 135, 221003 (2025)) and PandaX-4T (1.54 t·yr, PRL 134, 011805 (2025)), both cited in the LZ introduction [paper, refs XenonnT:WIMP-SI-2025, PandaX4T:WIMP-2025] — is whether their published WIMP searches, which used standard low-energy regions of interest (ROIs), should already have contained events under LZ's best-fit models, and what they would see if they re-analysed their data up to ~270 keV.

**Why the calculation is essentially efficiency-only.** All three detectors use natural xenon (same isotopic composition, same nuclear response functions), and all three adopt the Standard Halo Model with the Baxter et al. (2021) parameters (LZ states this explicitly [paper, Theory]; XENONnT and PandaX-4T also follow those conventions [recall, likely]). For a given interaction and WIMP mass, the true-recoil-energy spectrum per tonne-year, dR/dE_R, is therefore the same function in all three detectors. The expected count in experiment X is

  N_X = ε_X-exposure × ∫ dE_R (dR/dE_R) ε_X(E_R),        (1)

so once dR/dE_R is normalised to LZ's best fit, the only experiment-specific inputs are the exposure and the energy-dependent efficiency ε_X(E_R) (threshold, in-ROI acceptance, ROI upper edge). Absolute couplings, the WimPyDD c^0 normalisation convention (see PAPER_GUIDE rule 4) and the local DM density all cancel.

## 2. Inputs

### 2.1 From the LZ paper
| Quantity | Value | Source |
|---|---|---|
| Exposure | 2.84 t·yr (220 d × 4.71 t) | abstract; `lz.LZ["exposure_tyr"]` |
| Best-fit L10^s (1000 GeV) signal | 1.0 (+1.4, −0.7) events | Table I; `lz.LZ["L10s_1000_bestfit"]` |
| NR efficiency | 50% at 5.4 keV and 269.9 keV; 0.96 average plateau 14–250 keV | Fig. S2 and its caption; abstract |
| Efficiency roll-off shape | 0.9 at 250 keV, 0.5 at 269.9 keV, ~0.1 at ~283 keV, ~0 at 300 keV | read from Fig. S2 inset (this work) |
| Fig. 1 (bottom) | L10^s 1000 GeV recoil spectrum, unit coupling | digitised in this work (Sec. 4.2) |
| Table S7 | O1^s inelastic significance 2.9/3.0/3.3 σ at δ = 250/300/350 keV (1000 GeV) | motivates the δ grid |
| Signal code | WimPyDD with DMFormfactor-v6 density matrices, SHM (Baxter 2021) | Theory paragraph |

### 2.2 Recalled inputs (flagged)
| Item | Value used | Reliability | Note |
|---|---|---|---|
| XENONnT WIMP search exposure | 3.1 t·yr | certain (the number is in the paper title cited by LZ) | |
| PandaX-4T WIMP search exposure | 1.54 t·yr | certain (same) | |
| XENONnT NR ROI | cS1 ≲ 100 PE, roughly 3–60 keV_NR | **uncertain** | hence the scan of edges |
| PandaX-4T NR ROI | upper edge ~100–120 keV_NR (?) | **uncertain** | hence the scan of edges |
| In-ROI NR acceptance of both | 0.90 flat | assumption | results scale linearly |
| Low-energy threshold of both | 50% at 5 keV (erf, σ = 2 keV) | assumption | irrelevant: the models have negligible weight below 10 keV (see Sec. 5.1) |
| Anand–Fitzpatrick–Haxton (2014) Table 1, entry 10 | L10 = χ̄ iσ^{μν}q_ν/m_M χ N̄ iσ_{μα}q^α/m_M N → 4(q²/m_M²) O4 − 4(m_N²/m_M²) O6 | likely | validated in Sec. 4.2 |
| NREFT response structure | R_Σ'' ∝ (c4 + (q²/m_N²) c6)², R_Σ' ∝ c4² (spin-1/2 DM) | likely | Fitzpatrick et al. (2013) |
| Drift fields | XENONnT ≈ 23 V/cm, PandaX-4T ≈ 90 V/cm, LZ 97 V/cm | likely | affects the cS1→keV mapping of their ROI edges; motivates quoting a range |

Because the two ROI edges are uncertain, results are given for upper edges of **55, 70, 100, 150 keV** (hard cut) and for an **LZ-like extended ROI** (LZ's smooth 269.9 keV roll-off).

## 3. Method

### 3.1 Efficiency models
LZ (smooth model of Fig. S2, black curve):

  ε_LZ(E) = 0.96 · Φ((E − 5.4)/2.5) · [1 − Φ((E − 269.9)/12)],  Φ(x) = ½[1 + erf(x/√2)].   (2)

σ_low = 2.5 keV reproduces the ~95% efficiency by 10 keV visible in Fig. S2; σ_high = 12 keV reproduces the inset (0.9 at 250, 0.5 at 269.9, 0.1 at ≈285 keV). Other experiment with ROI edge E_max:

  ε_X(E) = 0.90 · Φ((E − 5)/2) · Θ(E_max − E),  or, for the "270 keV" case, 0.90 · Φ((E − 5)/2) · [1 − Φ((E − 269.9)/12)].   (3)

Figure `figures/efficiency_models.png` shows these curves.

### 3.2 Signal spectra (WimPyDD 2.0.4 via `lzcommon.wd_rate`)
Halo: `lz.wd_halo()` → WimPyDD `streamed_halo_function` with v_0 = 238 km/s, v_esc = 544 km/s, v_sun,pec = (11.1, 12.2, 7.3) km/s, `yearly_modulation=False`, `full_year_sampling=False`, i.e. η(v_min) evaluated on the day when the Earth's speed equals its annual mean (WimPyDD's default; v_E ≈ 250.6 km/s, v_max ≈ 794 km/s). Energy grid 0.5–400 keV in 0.5 keV steps. Target: WimPyDD natural Xe (all isotopes). Rates converted to events/(t·yr·keV).

Models (m_χ = 1000 GeV, j_χ = 1/2):
1. **O1^s inelastic**, δ = 250, 300, 350 keV: `wd_hamiltonian('o1s', {1:(1,0)})`, `delta_kev=δ`.
2. **L10^s (magnetic dipole–dipole)**: q-dependent Wilson coefficients, c4(q) = q²/m_N², c6 = −1 (WimPyDD keys `(4,'q2')` and `6`). This is Anand et al. Table 1 entry 10 up to the overall factor 4 d10/m_M², which cancels in the normalisation. **Consistency check:** c4 + (q²/m_N²)c6 = 0 identically, so the longitudinal Σ'' response vanishes and only the transverse Σ' response survives — exactly the tensor structure q²(S_χ·S_N) − (S_χ·q)(S_N·q) of a dipole–dipole interaction. It also explains why LZ finds the L10 isoscalar and isovector spectra indistinguishable [paper, SM "Log-Likelihood ... LEE"]: a pure spin-dependent operator has (nearly) the same shape in both isospin channels, whereas any coherent (M-response) admixture would not.
3. **Proxies / checks**: O4 elastic (SD-like), O6 elastic (q⁴Σ''), O1 elastic (SI reference), and "L10 with the O6 sign flipped" (Σ'' not cancelled) as a sign check.
4. **Digitised Fig. 1**: the brown 1000 GeV L10 curve of LZ Fig. 1 (bottom), colour-extracted from `inputs/figures_png/Fig1_combined_recoils.png` (Sec. 4.2), used as an independent shape.

**Failed first attempt (recorded).** We first used an L10 reduction recalled as (q²/2m_χm_M)O1 + 2(m_N/m_M)O5 − 2(m_N/m_M)[(q²/m_N²)O4 − O6], i.e. DM magnetic dipole × nucleon *vector* current. In WimPyDD this is dominated by the coherent O5 (M-response) term: the 1000 GeV spectrum peaks at 20 keV and falls monotonically to 270 keV (no 200 keV peak, high/low ratio 0.03), in flat contradiction with Fig. 1 and with the s/v degeneracy. It was abandoned; that structure corresponds to Anand's entry 9 (with the nucleon vector current), not 10.

### 3.3 Normalisation
Each spectrum is scaled so that N_LZ = 2.84 t·yr × ∫ dR/dE ε_LZ dE = 1.0 event. The LZ 68% band (0.3 to 2.4 events) is propagated as a multiplicative factor. Scale factors relative to WimPyDD unit coupling (c^0 = 1 in WimPyDD's convention, GeV⁻²):

| model | factor |
|---|---|
| O1^s δ=250 | 1.0896e-14 |
| O1^s δ=300 | 8.6410e-14 |
| O1^s δ=350 | 6.9843e-12 |
| L10 (c4 = q²/m_N², c6 = −1) | 5.2270e-09 |
| O4 | 5.4065e-13 |
| O6 | 2.0604e-09 |
| O1 elastic | 6.6138e-19 |
| Fig.1-digitised L10 (per unit of the plotted curve) | 7.4530e-02 |

(The last line says that the plotted unit-coupling L10 curve of Fig. 1 corresponds to 1/0.0745 = 13.4 events in LZ, i.e. LZ's best fit is d10² ≈ 0.075 in units of (1/m_v²)² for this exact spectrum. This is a by-product; absolute couplings are P003's remit.)

### 3.4 Statistics
P(0 | N) = e^{−N}; P(exactly 1 | μ) = μ e^{−μ}; exposure for 90% probability of ≥1 event: μ = −ln 0.1 = 2.303; for ≥3 events: P(≥3 | μ) = 0.9 → μ = 5.322 (brentq on `scipy.stats.poisson.sf`).

## 4. Validation

### 4.1 O1 inelastic shapes vs Fig. 1 (top)
WimPyDD gives, for δ = 300 keV at 1000 GeV, a spectrum starting at 91.5 keV, peaking at 170 keV, with form-factor minima at ≈100 keV (masked by the threshold) and ≈265 keV, and a rise beyond 270 keV — the same features as the green curve of Fig. 1 (top). Kinematic onsets (time-averaged halo): 55.0 keV (δ = 250), 91.5 keV (δ = 300), 155.0 keV (δ = 350). The dossier's kinematics (June halo) gave 90.4 keV for δ = 300 keV, consistent.

### 4.2 L10 shape vs a colour digitisation of Fig. 1 (bottom)
Digitisation (`digitise_fig1_L10()` in the script): pixels within colour distance 60 of tab:brown (140, 86, 75) in the bottom panel, median row per column; legend rows (y > 0.047 /t/yr/keV) excluded. Calibration, established by scanning the PNG: left spine at x = 188.5 px (0 keV), right edge 1649 px (350 keV); check — the gray-band edges land at 5.39 keV and 270.0 keV versus the paper's 5.4 and 269.9 keV. Vertical: tick labels 10⁻¹, 10⁻², 10⁻³ centred at rows 1145, 1470, 1785 (325 px/decade), confirmed by the minor-tick pattern. 1445 digitised points, 3.5–349.5 keV (`fig1_L10_digitised.csv`). Estimated digitisation accuracy ≈ ±5% in rate, ±1 keV in energy.

| feature | Fig. 1 digitised | WimPyDD q²O4 − m_N²O6 | flipped-sign check |
|---|---|---|---|
| low peak | 25 keV | 21 keV | none (monotonic to 150 keV) |
| dip | 58 keV | 52 keV | none |
| high peak | 202 keV | 196 keV | 147 keV |
| high/low peak ratio | 4.2 | 3.1 | — |
| RMS of (digitised/WimPyDD) − 1, 10–265 keV, after ROI-integral matching | — | 0.19 (range 0.60–1.42) | — |
| fraction of LZ-detectable signal below 55 / 70 / 100 / 150 keV | 6.3 / 7.9 / 12.5 / 29.0 % | 7.1 / 8.7 / 15.0 / 33.6 % | 9.8 / 15.4 / 28.2 / 51.9 % |

The dipole–dipole combination reproduces the double-peaked structure; the flipped-sign combination does not. Residual differences (WimPyDD is ≈30% high at 60–120 keV and ≈30% low at 250–300 keV relative to the 200 keV peak) plausibly reflect the different Σ' nuclear inputs (LZ: DMFormfactor-v6 density matrices "with modifications" [paper]; WimPyDD 2.0.4: its own shell-model tables). Both shapes are carried through the analysis; they differ by ≤15% in every count reported.

Figure `figures/L10_validation_vs_Fig1.png` shows the overlay and ratio.

### 4.3 Other checks
* Normalising the O1 elastic 1000 GeV spectrum to 1 LZ event yields N_XENONnT = 1.02–1.05 and N_PandaX = 0.50–0.52 for every ROI edge — as expected for an SI spectrum (97% of LZ-detectable events below 55 keV), confirming the exposure/efficiency bookkeeping (3.1/2.84 × 0.9/0.96 = 1.023).
* Setting the other experiments' 50% threshold to 3 keV instead of 5 keV changes L10 counts by <1% (the L10 spectrum below 10 keV carries 0.5% of the LZ-weighted integral); inelastic spectra are unaffected.
* No WimPyDD response-function files were written (directory tree compared before/after; `diff_rate` evaluates the q-dependent coefficients in memory).

## 5. Results

### 5.1 Fraction of LZ-detectable signal below each ROI edge (`fractions_table.csv`)
| model | onset | peak | median (LZ-weighted) | <55 | <70 | <100 | <150 |
|---|---|---|---|---|---|---|---|
| O1^s δ=250 | 55.0 | 158.5 | 159 | 0 | 0.029 | 0.115 | 0.399 |
| O1^s δ=300 | 91.5 | 170.0 | 173.5 | 0 | 0 | 0 | 0.197 |
| O1^s δ=350 | 155.0 | 206.0 | 208 | 0 | 0 | 0 | 0 |
| L10 (WimPyDD) | — | 196 | 179 | 0.071 | 0.087 | 0.150 | 0.336 |
| L10 (Fig. 1 digitised) | — | 203.5 | 187 | 0.063 | 0.079 | 0.125 | 0.290 |
| O4 (proxy) | — | ~0 | 24.5 | 0.736 | 0.793 | 0.867 | 0.937 |
| O6 (proxy) | — | 136 | 142.5 | 0.101 | 0.160 | 0.295 | 0.537 |
| O1 elastic (ref.) | — | ~0 | 16.5 | 0.971 | 0.991 | 0.996 | 0.998 |

### 5.2 Expected counts (best fit = 1.0 LZ event; brackets: LZ 68% band, ×0.3 and ×2.4) (`counts_table.csv`)

**XENONnT, 3.1 t·yr, ε = 0.9 in ROI**
| ROI edge | L10 (WimPyDD) | L10 (digitised) | δ=250 | δ=300 | δ=350 |
|---|---|---|---|---|---|
| 55 keV | 0.074 [0.022, 0.18] | 0.065 | 0 | 0 | 0 |
| 70 keV | 0.090 [0.027, 0.22] | 0.081 | 0.029 [0.009, 0.070] | 0 | 0 |
| 100 keV | 0.153 [0.046, 0.37] | 0.128 | 0.118 [0.035, 0.28] | 3e-6 | 0 |
| 150 keV | 0.344 [0.10, 0.83] | 0.297 | 0.408 [0.12, 0.98] | 0.202 [0.061, 0.48] | 0 |
| 270 keV (LZ-like) | 1.024 [0.31, 2.46] | 1.024 | 1.023 | 1.023 | 1.023 |

**PandaX-4T, 1.54 t·yr, ε = 0.9 in ROI**
| ROI edge | L10 (WimPyDD) | L10 (digitised) | δ=250 | δ=300 | δ=350 |
|---|---|---|---|---|---|
| 55 keV | 0.037 [0.011, 0.088] | 0.032 | 0 | 0 | 0 |
| 70 keV | 0.045 | 0.040 | 0.015 | 0 | 0 |
| 100 keV | 0.076 [0.023, 0.18] | 0.064 | 0.058 [0.018, 0.14] | 0 | 0 |
| 150 keV | 0.171 [0.051, 0.41] | 0.148 | 0.203 [0.061, 0.49] | 0.100 [0.030, 0.24] | 0 |
| 270 keV (LZ-like) | 0.509 [0.15, 1.22] | 0.509 | 0.508 | 0.508 | 0.508 |

**Both combined (4.64 t·yr)** and Poisson probability of zero events
| ROI edge | L10 N (P0) | δ=250 N (P0) | δ=300 N (P0) | δ=350 N (P0) |
|---|---|---|---|---|
| 55 keV | 0.110 (0.90) [0.097 digitised] | 0 (1) | 0 (1) | 0 (1) |
| 70 keV | 0.134 (0.87) | 0.044 (0.96) | 0 (1) | 0 (1) |
| 100 keV | 0.230 (0.79) [0.191] | 0.176 (0.84) | 0 (1) | 0 (1) |
| 150 keV | 0.516 (0.60) [0.445] | 0.611 (0.54) | 0.302 (0.74) | 0 (1) |
| 270 keV | 1.532 (0.216; 0.63 at ×0.3, 0.025 at ×2.4) | 1.532 (0.216) | 1.532 (0.216) | 1.532 (0.216) |

For the 270 keV row all models give the same count by construction (each is normalised to the same LZ efficiency-weighted integral, and the extended ε_X differs from ε_LZ only by the plateau ratio 0.9/0.96 and the irrelevant low threshold); the tiny model dependence (1.0233 vs 1.0237) comes from the low-threshold difference.

### 5.3 Xenon-world combination (7.48 t·yr = 2.84 + 3.1 + 1.54)
If all three experiments had LZ's ROI and efficiency: μ_world = 1.0 × 7.48/2.84 = 2.63 events (best fit).
* P(exactly one event in total) = 2.63 e^{−2.63} = **0.189** (0.359 at the lower 68% edge, μ = 0.79; 0.011 at the upper edge, μ = 6.32).
* P(zero in XENONnT + PandaX-4T at 270 keV, ε = 0.9 | best fit) = e^{−1.53} = **0.216**.
* A-priori probability that a single first event lands in LZ rather than the other two: 2.84×0.96 / (2.84×0.96 + 4.64×0.90) = **0.395**.

### 5.4 What it would take (`summary.json: required_exposure_*`)
With a 270 keV ROI and ε = 0.9 the best-fit rate is 0.330 events/(t·yr) (independent of model):
| | best fit | lower 68% (×0.3) | upper 68% (×2.4) |
|---|---|---|---|
| exposure for 90% chance of ≥1 event | **7.0 t·yr** | 23.3 t·yr | 2.9 t·yr |
| exposure for 90% chance of ≥3 events | **16.1 t·yr** | 53.7 t·yr | 6.7 t·yr |

Restricted ROIs (best fit, 90% chance of ≥1): L10 — 97 t·yr (55 keV), 80 (70), 47 (100), 21 (150); O1 δ=250 — 244 (70), 61 (100), 17.5 (150); O1 δ=300 — 35 t·yr (150 keV) and effectively impossible (>10⁵ t·yr) below 100 keV; O1 δ=350 — impossible below 155 keV.

Note that 7 t·yr is close to the total already collected by the three experiments (7.48 t·yr). Since XENONnT and PandaX-4T have 4.64 t·yr on disk, a re-analysis extending their ROIs to ~270 keV would expect 1.5 events at LZ's best fit and would return zero events with probability 22% (63% if the true rate is at LZ's lower 68% edge), so a null result from such a reanalysis would weaken but not exclude the interpretation; observing ≥2 events (probability 1 − e^{−1.53}(1 + 1.53) = 0.45) would be a strong corroboration given their tiny high-energy backgrounds (LZ: 0.0106 events for S1c > 500 phd [paper, Fig. 5]).

## 6. Figures
* `figures/spectra_with_ROIs.png` — spectra normalised to 1.0 LZ event (O1^s δ = 250/300/350 keV, L10 WimPyDD and digitised, O1 elastic reference) with the 55/70/100/150 keV edges shaded in blue and LZ's <5.4 and >269.9 keV regions in gray.
* `figures/expected_counts.png` — expected counts per experiment vs ROI edge (best fit), log scale.
* `figures/L10_validation_vs_Fig1.png` — WimPyDD L10 (dipole–dipole) and flipped-sign check versus the digitised Fig. 1 curve; lower panel: ratio.
* `figures/efficiency_models.png` — the efficiency curves of Eqs. (2)–(3).

## 7. Discussion
1. **The 2025 XENONnT and PandaX-4T null results are uninformative about LZ's high-energy hint.** For every model LZ fits at ≥3σ, the best-fit normalisation puts ≤0.11 events (ROI ≤55 keV), ≤0.13 (70 keV) or ≤0.23 (100 keV) into the two experiments combined; the inelastic models with δ ≥ 300 keV put exactly zero. Even at LZ's upper 68% edge the combined expectation for ROIs ≤100 keV is ≤0.55 events. Their low-energy ROIs cannot see a spectrum that, by construction, has no low-energy population — the same property that makes the LZ 2024 low-energy search irrelevant (dossier §2).
2. **The conclusion is robust to the ROI uncertainty.** Whether XENONnT's edge is 55 or 70 keV, or PandaX-4T's 100 or 120 keV, changes the combined expectation between 0.1 and 0.3 events. Only an edge ≥150 keV would give O(0.5) events.
3. **The conclusion is independent of the L10 nuclear-structure uncertainty.** The WimPyDD and Fig.-1-digitised L10 shapes differ by up to 40% locally but give sub-55 keV fractions of 7.1% and 6.3%.
4. **A xenon-world Poisson perspective.** If the signal is real at LZ's best fit, the 7.48 t·yr collected world-wide should have contained 2.6 events at LZ-like acceptance; that exactly one has been seen (with the other two experiments blind above ~60–120 keV) is unremarkable (P = 0.19 for exactly one, 0.39 that the first lands in LZ). The other two experiments' data are the cheapest available test: 1.5 expected events, 22% chance of a null.
5. **What we did not do.** We did not model XENONnT's or PandaX-4T's backgrounds at high energy, their ER/NR discrimination there, their multiple-scatter or MSSI-type populations, or their S1 saturation; a real reanalysis would need all of these. We also did not include annual modulation (the average halo is appropriate for multi-year exposures) or the dependence of the kinematic onset on the halo (the δ = 350 keV onset moves from 155 keV (average) to lower energies in June).

## 8. Failed or abandoned approaches
* L10 reduction with an O5 (coherent) term — abandoned (Sec. 3.2); it contradicts Fig. 1 and the s/v degeneracy.
* First digitisation pass included the brown legend sample at 225–250 keV, producing a spurious 227 keV peak and a high/low-peak ratio of 9.4; fixed by excluding rows above 0.047 /t/yr/keV.

## 9. References
1. LZ Collaboration, arXiv:2609.02823 (2026) — the paper under study.
2. E. Aprile et al. (XENON), "WIMP Dark Matter Search Using a 3.1 Tonne-Year Exposure of the XENONnT Experiment", Phys. Rev. Lett. 135, 221003 (2025), arXiv:2502.18005 [from LZ bibliography].
3. Z. Bo et al. (PandaX), "Dark Matter Search Results from 1.54 Tonne·Year Exposure of PandaX-4T", Phys. Rev. Lett. 134, 011805 (2025), arXiv:2408.00664 [from LZ bibliography].
4. N. Anand, A. L. Fitzpatrick, W. C. Haxton, "Weakly interacting massive particle-nucleus elastic scattering response", Phys. Rev. C 89, 065501 (2014), arXiv:1308.6288.
5. A. L. Fitzpatrick, W. Haxton, E. Katz, N. Lubbers, Y. Xu, "The effective field theory of dark matter direct detection", JCAP 02 (2013) 004, arXiv:1203.3542.
6. S. Jeong, S. Kang, S. Scopel, G. Tomar, "WimPyDD: an object-oriented Python code for the calculation of WIMP direct detection signals", Comput. Phys. Commun. 276, 108342 (2022), arXiv:2106.06207 [cited by LZ].
7. D. Baxter et al., "Recommended conventions for reporting results from direct dark matter searches", Eur. Phys. J. C 81, 907 (2021), arXiv:2105.00599 [cited by LZ].
8. Corpus: `output/00_evidence_dossier.md` (§2, §6 item 5); P003 (absolute coupling normalisation; not yet available at the time of writing).

## 10. Tools and provenance (mirrors `output/provenance/P005.json`)
* **Agent tools:** Read (PAPER_GUIDE.md; 00_evidence_dossier.md; results_ledger.csv; lzcommon.py; fulltext.tex lines 20–100, 200–310, 450–464, 800–912; Fig1_combined_recoils.png; FigS2_efficiency_werror.png; WimPyDD/package.py lines 930–1005, 1055–1075, 5351–5485; dossier_numbers.json; own figures ×5), Bash (greps of the tex/package, WimPyDD tree snapshots and diffs, script runs, wc), Write (scratch tests ×4, script, details, provenance, paper), Edit (script ×3).
* **Software:** Python 3.12.13; numpy 2.5.3 (trapezoid, interp, array ops); scipy 1.18.1 (special.erf, stats.poisson, optimize.brentq); matplotlib 3.11.2 (Agg; image.imread for the PNG digitisation); WimPyDD 2.0.4 (eft_hamiltonian with q-dependent Wilson coefficients, streamed_halo_function, diff_rate via `lzcommon.wd_rate`); `output/code/common/lzcommon.py` (LZ constants, wd_halo, wd_hamiltonian, wd_rate, M_NUCLEON_GEV).
* **Scripts:** `output/code/P005_other_xenon_expectations.py`, run as `.venv/bin/python output/code/P005_other_xenon_expectations.py` from the simulation root (≈1 min).
* **Local inputs:** `inputs/LZ_arXiv_2609.02823_fulltext.tex` (Introduction refs; Table I; Fig. 1 caption; Fig. S2 caption; SM s/v-degeneracy sentence; Table S7; bibliography entries 346–347); `inputs/figures_png/Fig1_combined_recoils.png` (digitised); `inputs/figures_png/FigS2_efficiency_werror.png` (roll-off widths); `output/00_evidence_dossier.md`; `output/work/dossier/dossier_numbers.json` (kinematic cross-checks); `output/code/common/lzcommon.py`.
* **Recalled knowledge:** 9 items, listed in Sec. 2.2 with reliabilities.
* **Datasets:** none. **Data requests:** none. **WimPyDD-generated files:** none (verified by directory diff).
