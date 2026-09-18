# P043 — Propagating the LZ gain uncertainties: what g₁ ± 2 % and g₂ ± 3 % do to the recoil energy, the band position, δ_max and the fitted couplings

Simulated date 2026-09-10 · physics.ins-det · RESP · author profile: calibration and systematics specialists.
Script: `output/code/P043_gain_systematics.py` (run from the root with `.venv/bin/python`; 60–90 s; log in `run_log.txt`).
Results: `P043_results.json`, `systematic_budget.csv`, `corpus_numbers_with_systematics.csv`, figures in `figures/`.

## 1. Motivation and framework

LZ quotes the event energy as 248 ± 23 (stat) ± 23 (sys) keV without decomposing the systematic. The paper gives the gains g₁ = 0.110 ± 0.002 phd/photon (±1.8 %) and g₂ = 34.5 ± 1.1 phd/electron (±3.2 %) "determined using monoenergetic peaks" (detector paragraph, tex l. 88) and the tuned NEST parameters (Table S5, tex l. 654–685). The corpus has already established several individual sensitivities (P009: g₁ ± 2 % → ± 3.9 keV, g₂ ± 3 % → ± 1.0 keV; P024: g₂ ± 3 % moves the band position between 1.10 and 1.99σ; P038: g₁ ± 2 % → ∓ 4.7 keV on E50; P009: the public Table S5 and LZ's own energy contours differ by 7.7 % in total quanta at 250 keV). This paper puts every one of those handles into a single, covariance-aware propagation and asks (i) whether the paper's ± 23 keV closes, (ii) what the same nuisances do to the "1.5σ below the NR median", the efficiency at the event, δ_max(m), the P021 likelihood peak and the couplings d₁₀ (P012) and κ̂ (P021), and (iii) which "systematic-inflated" numbers the corpus should carry.

Nuisance parameters (all Gaussian, independent):

| symbol | meaning | central | 1σ | source |
|---|---|---|---|---|
| g₁ | phd per photon | 0.110 | 0.002 (1.8 %) | paper l. 88 |
| g₂ | phd per electron | 34.5 | 1.1 (3.2 %) | paper l. 88 |
| s | scale of total NR quanta N_q(E) | 1 | 0.04 | half the Table-S5-vs-contour offset (7.7 %, P009) as a 1σ proxy — an assumption, stated |
| F | drift field | 96.5 V/cm | 5 V/cm | nestpy LZ_WS2024 central field; ± 5 V/cm is an assumed bracket (P024 scanned 90/100 V/cm) |
| W | eV per quantum (ER-equivalent energy only) | 13.5 | 0.2 | recalled (likely): nestpy 13.44, NEST v2 13.7 |
| σ_b | NR-band width at 540 phd (dex) | 0.0313 | 0.004 (13 %) | P024 model 0.0313, AmBe 0.036 ± 0.005 |
| plateau | efficiency plateau | 0.955 | 0.010 | Fig. S2 reading (P009); ± 0.01 assumed |

The **yield model** is P009's "paper-contour" scale: N_q(E) = s·α_p E^β_p with α_p = 11.3245, β_p = 1.11167 (from the digitised constant-energy contours of Fig. 4, `P009/results.json`), and the electron yield N_e(E, F) from the LZ-tuned NEST parameters including the p(E) break (`lz.nest_nr_yields`, nestpy 2.1.1), so N_ph = s N_q − N_e. The scale nuisance s therefore acts on the light yield; N_e is kept fixed because the S2 of the event pins the charge, and because the Table-S5-vs-contour offset is a light-yield offset (P009). N_e was tabulated on a grid E = 120–400 keV (0.5 keV) × F = 86.5–106.5 V/cm (5 V/cm) and interpolated bilinearly (2805 nestpy calls, 0.6 s).

Note on W: the NR energy scale read from LZ's contours is fixed by the S1c at which each labelled contour crosses the NR median, not by W; W enters only the keVee conversion E_ee = W(S1c/g₁ + S2c/g₂). A W shift would be degenerate with s if one instead built N_q from the keVee labels; to avoid double counting we treat W as acting on E_ee only.

## 2. Estimators and derivations

Event: S1c = 540.1 phd, S2c = 9268 phd (log₁₀ S2c = 3.9670). Three energy estimators (P009):

1. **S1-only**: solve g₁ N_ph(E; s, F) = S1c (vector bisection, 45 iterations).
   Elasticity: dlnE = −(1/β_eff)(dln g₁ + dln s) with β_eff ≡ dln S1c/dln E = 1.160 at 248 keV (the light yield rises faster than N_q because N_e/N_q falls with E). Hence g₁ ± 1.82 % → ∓ 1.57 % → ∓ 3.90 keV; s ± 4 % → ∓ 3.45 % → ∓ 8.57 keV (analytic) — the numerical propagation gives 3.90 and 9.10 (the 9.10 includes the curvature of N_e).
2. **Combined quanta**: N_q,ev = S1c/g₁ + S2c/g₂ = 4910 + 268.6 = 5178.6; E = (N_q,ev/(s α_p))^{1/β_p}. Here dln N_q,ev = −0.948 dln g₁ − 0.052 dln g₂, so g₂ enters at 5 % weight: elasticities −0.853 (g₁), −0.047 (g₂), −0.90 (s).
3. **2D-ML proxy**: maximise −½[(S1c − g₁N_ph(E))²/σ₁(E)² + (log₁₀S2c − log₁₀(g₂N_e(E,F)))²/σ_b²] − ln σ₁ over E (0.5 keV grid, parabolic refinement), with σ₁(E) = 26.15 phd × √(⟨S1c⟩/⟨S1c⟩₂₄₈) (P038's MC S1c width at 248 keV, scaled as √mean) and σ_b = 0.0313 dex. Because dlog₁₀S2c/dE = 0.00058 dex/keV, the S2 axis alone has σ_E = 54 keV, so it carries a weight (10.4/54)² ≈ 3.7 % and pulls E_ML towards the S2-only value (177 keV) by ≈ 2.6 keV. This reproduces P009's full MC: E_ML = 245.3 keV here vs 245.8 keV (P009), and gives the g₂ lever arm: g₂ ± 3.2 % → ± 0.0136 dex → ± 23 keV on the S2 axis → × 0.037 ≈ ± 0.84 keV (P009: 1.0).

Central values: E_S1 = 248.56, E_comb = 247.16, E_ML = 245.29 keV (P009: 248.56 / 247.16 / 245.82); E_ee = 69.91 keVee at W = 13.5 eV (P009 69.9, P010 69.6 at 13.44 eV).

**Linear propagation** used `uncertainties 3.2.3`: each estimator was wrapped with `uncertainties.wrap` (numerical derivatives) and evaluated on tagged `ufloat`s; `error_components()` gives the per-source σ and the covariance-aware total. **Full propagation**: 10⁵ Gaussian draws (numpy `default_rng(43)`), one source at a time and all together.

## 3. Results

### 3.1 Energy budget (keV; linear | MC)

| source | E_S1 | E_comb | E_ML (2D proxy) | E_ee (keVee) |
|---|---|---|---|---|
| g₁ ± 1.8 % | 3.90 \| 3.89 | 3.83 \| 3.82 | 3.72 \| 3.71 | 1.21 \| 1.20 |
| g₂ ± 3.2 % | 0 \| 0 | 0.37 \| 0.37 | 0.84 \| 0.84 | 0.12 \| 0.12 |
| N_q-scale ± 4 % | 9.10 \| 9.19 | 8.89 \| 8.98 | 8.70 \| 8.77 | 0 |
| field ± 5 V/cm | 0.04 \| 0.03 | 0 | 0.04 \| 0.04 | 0 |
| W ± 0.2 eV | 0 | 0 | 0 | 1.04 \| 1.04 |
| **total** | 9.90 \| 9.99 | 9.69 \| 9.78 | **9.50 \| 9.57** | 1.59 \| 1.59 |

MC (all sources) for E_ML: mean 245.7, median 245.4, 68 % 236.2–255.3, 95 % 227.7–265.9 keV, skew +0.22 (the 1/g₁ and 1/s non-linearity). Linear and MC agree to 1 %. The drift field is irrelevant (N_e changes by < 1 % for ± 5 V/cm, and N_q is fixed by the contour scale). Elasticities (dlnE/dln x): g₁ −0.862 (S1), −0.835 (ML); g₂ −0.047 (comb), −0.107 (ML); s −0.915 (S1), −0.887 (ML).

**Closure.** Our systematic total is 9.6 keV against the paper's 23 keV (ratio 2.4). The gains alone give 3.8 keV. To reach 23 keV the light-yield scale would need σ_s = 10.4 % (solving 23² = 3.72² + 0.84² + 0.04² + (8.70/0.04)² σ_s²). The full Table-S5-vs-contour shift in E_ML is 16.5 keV (262.3 − 245.8), so LZ's ± 23 keV is consistent with a yield-model systematic of roughly the size of that whole offset (or a NEST light-yield uncertainty of ≈ 10 % at 250 keV, where the paper says AmBe calibration is sparse — tex l. 691), not with the quoted gains. Combined stat ⊕ sys: ours 13.7 keV (9.8 ⊕ 9.6) vs the paper's 32.5 keV.

### 3.2 Band position

Model: at the event's S1c the NR-band median is log₁₀[g₂ N_e(E_slice, F)] with E_slice = E_S1(g₁, s, F); the event's log₁₀S2c is data and fixed. Our mean-yield evaluation gives 1.593σ; P024's full MC (skewed distribution, slice mixing) gives 1.541σ, so we anchor: nσ = 1.541 + Δ(model). Validation of the shifts against P024's one-at-a-time variants: g₂ ∓ 3.2 % → 1.09/1.98 (P024 1.10/1.99); g₁ ∓ → 1.61/1.47 (P024 1.62/1.46); Table S5 scale (s = 1/1.077) → 1.85 (P024 1.86); drawn width 0.0212 dex → 2.30 (P024 2.33). New: s ± 4 % → 1.71/1.38; F ± 5 → 1.50/1.58; σ_b = 0.036 → 1.33.

MC (σ in units of band σ): g₂ 0.442, band width 0.218, N_q-scale 0.168, g₁ 0.071, field 0.038; **total 0.538** (mean 1.56, median 1.54, 68 % 1.04–2.09, 95 % 0.54–2.71, skew +0.28). P(> 2σ) = 0.198, P(< 1σ) = 0.142, P(> 2.5σ) = 0.047. The tail probability P(≤ event | NR) is 0.062 at the central 1.54σ and 0.084 when marginalised over the systematics (the marginal is larger because the tail is convex).

Mechanism: g₂ moves the data point relative to the model band by log₁₀(1.032) = 0.0137 dex = 0.44σ directly; g₁ and s move the slice energy (± 4 keV, ± 8.6 keV) and thus the median only through dlog₁₀S2c/dE = 0.00058 dex/keV; the band width rescales the distance. g₂ dominates, as P024 anticipated.

### 3.3 Efficiency

E50 (the energy where g₁ N_ph = 600 phd, i.e. P(S1c < 600) = ½): 272.2 keV (P038 MC 271.6, Fig. S2 269.9). g₁ ∓ 1.8 % → 276.5/268.0 (∓ 4.3 keV; P038 quotes 4.7 keV per 2 %); s ∓ 4 % → 282.5/262.6 keV (∓ 10 keV); F: no change; Table-S5 scale → 291.3 keV (P009 291). ε(E) = plateau × Φ((600 − g₁N_ph)/σ₁): ε(246/248/262) = 0.950/0.946/0.794 (P038 0.950/0.945/0.792).

Three ways of quoting the event's efficiency:
- **naive** (E fixed at 248 keV, curve moves with the nuisances): g₁ → 0.952/0.931; MC over all sources mean 0.911, median 0.942, sd 0.088, 68 % 0.87–0.96 (the N_q-scale tail drags E50 down to 262 keV where ε(248) = 0.8).
- **correlated** (evaluate at E_ML of the same draw): 0.951 ± 0.010, where the ± 0.010 is entirely the assumed plateau uncertainty; g₁ and s cancel because both the event's reconstructed energy and the edge are set by S1c (540.1 vs 600 phd): ε(E_ML) ≈ 0.955 Φ(60/26) irrespective of the energy scale. 1/ε = 1.052 ± 0.011.
- **posterior-averaged acceptance correction** ⟨1/ε⟩ with E_true ~ N(E_ML, 9.84 keV) inside each draw: median 1.057, 68 % 1.042–1.106, mean 1.091 (sd 0.226, skew 57 — the mean is set by the few-per-cent tail of E_true > 270 keV where 1/ε → 2–5); stat-only mean 1.090, sd 0.155, so the systematics add 0.165 in quadrature to the mean but nothing to the median. ⟨ε⟩ = 0.926 (median 0.946).

Recommendation: quote ε = 0.95 ± 0.01 and 1/ε = 1.06 (68 % 1.04–1.11); the acceptance is a function of the observed S1c and is not moved by the gain or yield scale.

### 3.4 Kinematics

June v_max = v_E(day 167) + v_esc = 265.1 + 544 = 809.1 km/s (`lz.vmax_kms`, `lz.v_earth_kms(167)`). δ_max(E, m) = v_max√(2m_N E) − m_N E/μ (`lz.delta_max_kev`):

| m (GeV) | δ_max(248) | dδ_max/dE | σ_stat (9.8 keV) | σ_sys (9.6 keV) | σ_tot | MC 68 % | paper ± 32.5 keV |
|---|---|---|---|---|---|---|---|
| 400 | 341.1 | 0.035 | 0.3 | 0.4 | 0.7 | 340.2–341.3 | 338.4–340.9 |
| 1000 | 386.6 | 0.218 | 2.1 | 2.1 | 3.1 | 382.8–388.9 | 378.0–392.4 |
| 4000 | 409.4 | 0.310 | 3.1 | 3.0 | 4.4 | 404.0–412.7 | 397.8–418.1 |

(The 400 GeV slope is tiny because 248 keV is near the ceiling μv²/2 = 341.3 keV, where δ_max(E) is stationary.) The MC means are 0.7 keV below the 248 keV values because E_ML = 245.3 keV.

P021's likelihood peak follows E_obs at 5 keV/16 keV = 0.31 keV/keV (peaks 380 at E_obs = 246–248, 385 at 262; 5 keV grid, so 0.2–0.5 is the honest bracket): σ_sys(δ_peak) = 3.0 keV, σ_tot = 4.3 keV. P007's Higgsino δ(N = 1) = 366 keV is set by the rate (falling × 5 per 10 keV), not by E_obs, and is untouched by the energy systematics; its own uncertainty is the halo tail (P018: −17/+11 keV).

### 3.5 Couplings

In the single-event profile (P021, P038) the fitted coupling is κ̂ = 1/N_unit with N_unit = exposure × ∫ dR/dE ε(E) dE; the energy scale enters only through ε (the δ-preference, not κ̂, depends on E_obs). We move E50 by the g₁ and s shifts above (erf edge with σ = 10.93 keV from P038; low edge 5.4 keV/2.5 keV).

- **L10 (P012 cached spectra, d₁₀ = 1)**: 1000 GeV: N_unit × 1.0127/0.9867 (g₁ ∓), × 1.0286/0.9683 (s ∓); d₁₀ ∝ N_unit^{−1/2} → ± 0.65 % (g₁), ± 1.5 % (s); combined 1.6 %: **d₁₀ = 0.279 ± 0.005 (sys)** against the statistical 0.15–0.43. 200 GeV: ± 0.33 %/0.8 %; 4000 GeV: ± 0.7 %/1.6 %. Only 17 % of the 1 TeV L10 rate lies above 250 keV, hence the insensitivity.
- **Inelastic O₁ (P038 full-window spectra, 1000 GeV, annual halo)**: acceptance A₆₀₀ = 0.853/0.464/0.120/0.030 at δ = 300/350/370/380 keV (P038 0.81/0.44/0.12/0.029; P038 includes the 0.955 plateau, ours does not — ratios are unaffected). κ̂ changes by × 1.00/1.00, 0.99/1.01, 0.96/1.03, **0.85/1.14** for g₁ ∓ and by × 1.00, 0.98/1.01, 0.89/1.06, **0.65/1.30** for s ∓; the Table-S5 scale (E50 = 291 keV) would give × 0.99, 0.95, 0.76, 0.42. So the acceptance systematic on the coupling is < 1 % for δ ≤ 350 keV and grows to ± 14 % (g₁) / ± 30 % (yield scale) at δ = 380 keV, where only 3 % of the spectrum is accepted — still far below the statistical factor 3 (68 %: 0.66–5.2) and P038's × 31 edge dependence.

### 3.6 Corpus numbers with systematics (`corpus_numbers_with_systematics.csv`)

| quantity | value ± stat ± sys | dominant systematic |
|---|---|---|
| E_R (2D-ML, LZ contour scale) | 245.3 ± 9.8 ± 9.6 keV (68 % sys 236–255) | N_q-scale 8.7, g₁ 3.7, g₂ 0.8 |
| E_S1 | 248.6 ± 10.4 ± 10.0 keV | N_q-scale 9.1, g₁ 3.9 |
| E_ee | 69.9 ± 1.2 ± 1.6 keVee | g₁ 1.2, W 1.0 |
| σ-distance below NR median | 1.54 ± 0.54 (68 % 1.04–2.09); P(> 2σ) = 0.20, P(< 1σ) = 0.14 | g₂ 0.44, width 0.22, N_q 0.17 |
| ε at the event | 0.95 ± 0.01; 1/ε = 1.06 (1.04–1.11) | plateau; g₁/s cancel |
| δ_max(1 TeV, June) | 386.6 ± 2.1 ± 2.1 keV (400 GeV 341.1 ± 0.3 ± 0.4; 4 TeV 409.4 ± 3.1 ± 3.0) | via σ_E |
| P021 peak δ (1 TeV) | 380 ± 3 ± 3 keV | via σ_E |
| d₁₀ˢ (1 TeV) | 0.28 (0.15–0.43) ± 0.005 | acceptance |
| κ̂(δ = 380) = 2.19 | (0.66–5.2) ×/÷ 1.14 (g₁) ×/÷ 1.30 (N_q-scale) | acceptance |

## 4. Validation and robustness

- Central estimators reproduce P009 to 0.0 keV (E_S1, E_comb) and 0.5 keV (E_ML Gaussian proxy vs full MC).
- g₁ and g₂ sensitivities reproduce P009 (3.9/1.0 keV) and P024 (1.10–1.99σ, 1.46–1.62σ, 1.86σ for Table S5, 2.33σ for the drawn lines); E50 shift 4.3 vs P038's 4.7 keV per 2 % (P038 used the MC S1c median, we the mean yield; β_eff differs by 1 %).
- ε(E) reproduces P038 at 246/248/262 keV to 0.002; A₆₀₀ reproduces P038 at 350–380 keV to 5 %.
- Linear (uncertainties) vs MC totals agree to 1 %; the MC skew (+0.22) is the 1/g₁ non-linearity.
- Field variation is negligible everywhere (< 0.05 keV, 0.04σ), consistent with P024's 90/100 V/cm variants (1.51/1.57σ).
- Robustness of the budget to the s prior: σ_E,sys scales linearly with σ_s above 2 % (8.7 keV per 4 %); the closure statement (10.4 % needed) is a direct inversion.

## 5. Failed or abandoned approaches

- Analytic (symbolic) derivatives through the nestpy yield were not attempted; `uncertainties.wrap` numerical derivatives were used instead and agree with the analytic elasticity formula to 1 %.
- Treating W as an independent nuisance on the NR scale was rejected as double counting (Section 1).
- A full nestpy `GetQuanta` MC per draw (P009/P024 style) for 10⁵ nuisance draws was unnecessary: the Gaussian 2D proxy reproduces the full-MC E_ML to 0.5 keV and the band shifts to 0.01σ; instead we anchor the band baseline to P024's MC.
- The posterior-averaged 1/ε mean is dominated by a heavy tail (skew 57) and is not a useful summary; the median and 68 % interval are quoted.

## 6. Discussion

The paper's ± 23 keV systematic cannot be made of the quoted gain uncertainties: g₁ and g₂ together contribute 3.8 keV to the 2D reconstruction, and even a ± 4 % light-yield scale (half the public-vs-contour offset) only brings the total to 9.6 keV. Closing the budget needs a ≈ 10 % uncertainty on the NR light yield at 250 keV, which is plausible where the AmBe calibration is sparse and the new p(E) break is untested, and is essentially the size of the Table-S5-vs-contour discrepancy identified by P009. Users of the corpus should therefore read the paper's systematic as a yield-model uncertainty. Its consequences are asymmetric: it barely touches the acceptance (which is a function of S1c, not energy), δ_max (2 keV at 1 TeV), the L10 coupling (1.6 %) or κ̂ below δ = 350 keV, but at δ ≥ 370 keV the coupling and the δ-preference inherit ± 15–30 % and ± 3 keV respectively, on top of P038's edge effect. The band position is the one place where g₂ matters: the 1.5σ statement carries ± 0.54σ once g₂, the width and the yield scale are marginalised, and the probability that the event is actually more than 2σ below the median is 20 %, that it is within 1σ 14 %.

## 7. Figures

- `figures/P043_fig1_budget.png` — four panels: per-source σ on E_ML (paper's 23 keV dashed), on the band σ-distance, on δ_max(1 TeV) (statistical σ dashed), and half-range coupling shifts for d₁₀ and κ̂(δ) from g₁ and the N_q-scale.
- `figures/P043_fig2_energy_distribution.png` — MC E_ML distribution (systematics only, σ 9.6 keV), stat ⊕ sys (13.7 keV), P009 statistical Gaussian (9.8 keV) and the paper's 248 ± 32.5 keV; Table S5 (262 keV) and the 269.9 keV 50 % point marked.
- `figures/P043_fig3_band_distance.png` — distribution of the event's σ-distance below the NR median: all sources (σ 0.54), g₂ only (0.44), band width only (0.22).

## 8. References

LZ Collaboration, arXiv:2609.02823 (2026). M. Szydagis et al., NEST review, arXiv:2211.10726 (2024). B. Lenardo et al., IEEE TNS 62, 3387 (2015). E. O. Lebigot, `uncertainties` Python package (https://pythonhosted.org/uncertainties/), v3.2.3. D. Baxter et al., EPJC 81, 907 (2021). J. Aalbers et al. (LZ), PRL 131, 041002 (2023). Corpus: P002, P007, P009, P010, P012, P018, P021, P024, P038.

## 9. Tools and provenance (mirrors `provenance/P043.json`)

- Agent tools: Read × 29 (PAPER_GUIDE; dossier; ledger 3 pages; papers P009, P024, P010, P038, P002, P021, P012, P007; tex l. 82–123 and 586–695; lzcommon.py l. 1–75, 159–218, 255–294; P009/results.json; P024/results.json (part); P038/P038_results.json; P021/P021_peaks.csv; P012/P012_summary.json; provenance/P038.json; P009_energy_reconstruction.py l. 36–235; own figures × 4), Bash × 10 (grep of tex and lzcommon; ls/versions; npz key inspection + kinematics check; three script runs incl. log capture and JSON digest; four word-count checks + JSON validation), Write × 5 (script, details.md, provenance JSON, paper × 2), Edit × 14 (figure layout × 2; paper trims × 11; provenance tally), Skill × 1 (dataviz).
- Software: python 3.12.13; numpy 2.5.3 (default_rng(43), vectorised bisection, trapezoid); scipy 1.18.1 (stats.norm, stats.skew, special.erf); uncertainties 3.2.3 (ufloat, wrap, error_components); nestpy 2.1.1 via lzcommon (LZ_WS2024, GetYields NR with Table S5 + p(E) break); pandas 3.0.5; matplotlib 3.11.2 (Agg); common/lzcommon.py (LZ, NEST_NR_LZ, DRIFT_FIELD_VCM, nest_nr_yields, vmax_kms, v_earth_kms, delta_max_kev).
- Local inputs: tex detector paragraph (l. 88), Results (l. 165), supplement NEST tables (l. 591–685), waveform paragraph (l. 691); P009 results (α_p, β_p, σ_stat, estimator values); P024 results (band width, decomposition, variants); P038 results (E50, σ_erf, ε, A₆₀₀, S1c sd); P012 L10 spectra npz; P038 inelastic spectra npz; P021 peaks CSV.
- Recalled: W = 13.5 ± 0.2 eV (likely); Okabe–Ito palette (certain, cosmetic); Gaussian error propagation and elasticity algebra (certain).
- Datasets: none. Data requests: none. WimPyDD files: none written (spectra read from P012/P038 caches).
