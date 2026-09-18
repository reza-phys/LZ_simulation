# P081 — Posterior predictive counts for LZ's next release: P(0), P(1), P(≥2) high-energy events under the corpus posterior

Simulated arXiv date 2026-09-15 · physics.data-an (cross-list hep-ex) · STAT · astrostatistics group.
Script: `output/code/P081_posterior_predictive.py` (run from the simulation root with `.venv/bin/python`; 61 s).
All tables quoted below are written to `output/work/P081/` (`P081_predictive_counts.csv`, `P081_variants.csv`,
`P081_decision_table.csv`, `P081_energy_placement.csv`, `P081_bestfit_vs_posterior_mean.csv`, `P081_Pge1_vs_exposure.csv`,
`P081_class_decomposition.csv`, `P081_model_table.csv`, `P081_results.json`, `run_log.txt`).

## 1. Motivation and framework

P020 and P069 forecast LZ's untouched exposure with *plug-in* best-fit rates (L10 or single inelastic points) and P001's
three-hypothesis posterior. P027 has since replaced the single best fit with a model-marginalised posterior over 298
distinguishable spectra with a marginalised coupling, P061 has replaced the point prior odds with a hyper-prior distribution
of P(DM | event) (median 0.015, 68 % 0.0016–0.137), P038 quantified how the S1c < 600 phd edge hides the high-energy lobe of
large-δ inelastic spectra, P068 quantified the astrophysical band on every inelastic quantity, and P034 tabulated the annual
time PDFs of inelastic signals. Here everything is folded into one *posterior predictive* for the number of events LZ's next
release will contain, in three regions and for four assumed additional exposures, plus the Bayes-factor update each outcome
would produce.

Hypotheses and their predictives for an exposure k = E/2.84 t·yr (E = 1.0, 2.8, 5.6, 10 t·yr):

* **DM.** For every spectrum m with P027 posterior weight w_m (`post_uniform`; variant `post_classes`), the rate posterior in
  s (expected 200–270 keV NR-band events per 2.84 t·yr) is

  p(s | m, data) ∝ π(s) · e^{−s} (1 + ρ_m s/b_H) · W_m(s/f_hi,m),

  exactly P027's joint likelihood: one event on b_H = 5.695 × 10⁻⁴ (P016), the in-window shape factor ρ_m, and W_m the
  marginal likelihood ratio of the empty 125–200 keV bin (b_M = 0.02) and the 20 digitised low-energy Fig. 5 bins with the
  θ nuisance (σ_θ = 0.3). Priors π(s): log-uniform on [10⁻³, 30] (P027 baseline), uniform on [0, 10], Jeffreys on [0, 10].
  Counts in region R: N_R ~ Poisson(k · s · g_R(m) · F_R(m)) with g_H = 1, g_L2M = (f_L2 + f_M)/f_hi (55–200 keV),
  g_X = ∫dR/dE [ε₁₀₀₀ − ε₆₀₀] dE / ∫₂₀₀²⁷⁰ dR/dE ε₆₀₀ dE (events accepted by a 1000 phd edge but not by the 600 phd edge:
  "270–420 keV extension"), and F the seasonal factor of the release window (Sec. 4).
* **B.** Modelled background, Poisson with mean k·b_R: b_H = 5.7 × 10⁻⁴ (P016; corpus bracket 3 × 10⁻⁴ [P070/P004 with wall
  MSSI k = 0.62] to 3.5 × 10⁻³ [P010 flat ER-tail extrapolation]); b_M = 0.02 (P016, 125–200 keV); b_L2 = 0.10 for 55–125 keV
  (**estimate**, bracket 0.03–0.3: no corpus number exists; interpolates between P016's b_M and the low-energy leakage);
  b_X = 0.0027 NR-band MSSI added by a 1000 phd edge (P038; bracket 0.0008–0.0034) + 10⁻⁴ accidentals (P022). Whole S1c > 500
  phd panel 0.0106 (LZ Fig. 5).
* **U.** Unmodelled background (P061): one-off (f_trans = 0.5) → predicts B only; steady (1 − f_trans) → rate with log-uniform
  prior updated by one event → Exponential(1) per 2.84 t·yr → N_H ~ Poisson–Gamma (negative binomial) P(n) = kⁿ/(1+k)ⁿ⁺¹,
  localised in the 200–270 keV band (an LZ-specific effect that reproduces the observed event class).
* **Weights.** P061's hyper-prior sampled (2 × 10⁴ draws, log-uniform: π_DM ∈ [10⁻³, 0.3], π_U ∈ [0.01, 0.5], L_rest ∈
  [10⁻³, 0.3], f_FP ∈ [1, 10], B_DM ∈ [6.6, 45]; L_acc = 9.3 × 10⁻³, b = 5.7 × 10⁻⁴): P(DM) median 0.0153, 68 % 0.0016–0.135,
  P(DM) > 0.5 in 2.4 % (P061: 0.015, 0.0016–0.137, 2.4 %); P(B) median 0.148, P(U) median 0.789. Every mixture quantity is
  reported as the median and 68 % range over these draws, i.e. the *full* P061 distribution is propagated.

## 2. Inputs (all local)

| input | file | use |
|---|---|---|
| P027 model table (752 rows; 175 with non-zero weight; f_lo/f_L2/f_M/f_hi, ρ, N_M, B, posterior weights) | `work/P027/P027_models.csv` | weights, fractions |
| P027 spectra cache (388 spectra, 2–400 keV) and P008 cache (501 spectra, 2–400 keV; P027's elastic source) | `work/P027/P027_spectra.npz`, `work/P008/spectra_cache.npz` | extension ratio g_X |
| P016 digitised Fig. 5 bins, b_H | `work/P016/P016_results.json` | likelihood |
| P038 edge parameters (E50 = 271.6/423.2 keV, σ = 10.9/14.5 keV for 600/1000 phd), acceptance table, full 1 TeV O₁ inelastic spectra (16 δ, to 800 keV) | `work/P038/P038_results.json`, `P038_acceptance_vs_delta.csv`, `spectra_s_1000_full.npz` | 1000 phd edge, truncation correction, astro δ-shift map h(δ) |
| P050 L10 spectrum to 1498 keV | `work/P050/P050_spectra_cache.npz` | elastic truncation factor |
| P034 time PDFs (δ = 300/330/350/366/380 keV, 1 TeV) and a₁ | `work/P034/time_pdfs.csv`, `time_pdf_summary.csv` | seasonal factor |
| P068 a₁ 68 % bands, δ_max band | `work/P068/marginals_and_sensitivities.csv`, P068.md | astro spread |
| P069 LZ-untouched counts, P020 predictive, P027 predictive | `work/P069/P069_hidden_events.csv`, P020.md, `work/P027/P027_posterior_predictive.csv` | cross-checks |
| LZ paper: run 27 March 2023 – 1 April 2024, 220 live d (l. 111); data taking continues since 1 April 2024 (l. 334); Fig. 5 caption 0.0106 ± 0.0008 | `inputs/LZ_arXiv_2609.02823_fulltext.tex` | windows, panel background |

## 3. Rate posteriors per model

Rebuilding P027's marginal likelihoods reproduces its `B_logu_rho` column to a ratio median 1.0000 (range 0.9605–1.0000; the
weights follow the ρ-including column). Companion-free reference (W = 1): posterior mean of s = 0.998 (log-uniform,
Exponential(1)), 1.995 (flat, Gamma(2,1)), 1.497 (Jeffreys, Gamma(1.5,1)) against the maximum-likelihood ŝ = 1.0 — the
"Gamma posterior" effect. Selected models (f_hi, N_M, ŝ, ⟨s⟩_logu, ⟨s⟩_flat, w):

| model | f_hi | N_M | ŝ | ⟨s⟩ logu | ⟨s⟩ flat | w_uniform |
|---|---|---|---|---|---|---|
| L10ˢ 1000 GeV | 0.356 | 1.164 | 0.417 | 0.415 | 0.832 | 0.0146 |
| O₁ᵛ 1000 GeV δ = 350 | 0.989 | 0.012 | 0.994 | 0.987 | 1.973 | 0.0344 |
| O₄ˢ 1000 GeV δ = 350 | 0.933 | 0.072 | 0.937 | 0.931 | 1.863 | 0.0325 |
| O₁ˢ 1000 GeV δ = 350 | 0.649 | 0.540 | 0.648 | 0.646 | 1.298 | 0.0227 |
| O₁ˢ 1000 GeV δ = 300 | 0.195 | 4.055 | 0.196 | 0.193 | 0.392 | 0.0069 |
| O₆ˢ 1000 GeV | 0.232 | 1.516 | 0.315 | 0.312 | 0.624 | 0.0110 |

The empty 125–200 keV bin pulls every model with companions below 1.0: ŝ ≈ 1/(1 + N_M + low-E term). The mixture
posterior of s (200–270 keV per 2.84 t·yr, baseline): mean 0.385, median 0.189, 68 % 0.036–0.694, 90 % 0.009–1.44;
flat-prior mean 0.767; Jeffreys 0.57. Class weights (uniform prior): elastic-isoscalar 0.253, elastic-isovector 0.171,
inelastic-O₁ 0.326, inelastic-O₄ 0.250. Weighted fractions: ⟨f_hi⟩ = 0.338, ⟨g_L2M⟩ = 4.08 (class prior 3.20),
⟨g_X⟩ = 0.99 (1.05).

## 4. Region ratios, edge effects, seasonal factor, astrophysical band

*Extension ratio.* g_X uses ε₆₀₀ (P003/P016 erf edge at 269.9 keV, σ = 8 keV, plateau 0.96) and ε₁₀₀₀ (P038 E50 = 423.2 keV,
σ = 14.5 keV). Caches end at 400 keV; the truncation is corrected with P038's full 1 TeV O₁ spectra (correction 1.12 at
δ = 300 → 1.09 at 380 keV) and for elastic models with the L10 factor from P050's spectrum to 1498 keV (g_X 0.933 full vs
0.728 truncated → ×1.281, applied to all elastic models: approximation). P038-grid extension per 200–270 keV event
h(δ) = 0.63/0.83/2.04/8.27/41.6/413 at δ = 250/300/350/370/380/390 keV. Cross-check with P038's own table (extra 600–1000 phd
events per ROI event 0.155/1.06/6.84/30.6 at 300/350/370/380): dividing by the 200–270 keV share of the accepted ROI
(0.178–0.186 at 300, 0.505 at 350 from P069/P038) gives 0.83–0.87 and 2.10, matching our 0.83 and 2.04.

*Seasonal factor.* First run (27 Mar 2023 – 1 Apr 2024, uniform livetime): F = 1.002/1.003/1.003/1.000/0.995 at
δ = 300/330/350/366/380 keV — a full year, so the first-run rate is the annual mean. Release windows start 1 April 2024 with
LZ's live fraction 220/371 (P020): 1.0 t·yr = 131 calendar days (to 9 Aug 2024): F = 1.35/1.64/2.03/2.43/2.60; 2.8 t·yr
(366 d) and 5.6 t·yr (732 d): 0.995–1.003; 10 t·yr (1306 d, ending in the 2027 autumn): 1.025–1.112. For δ < 300 keV,
(F − 1) is scaled by P006's ROI modulation fractions (0.10/0.24/0.42 at 100/200/300 keV); elastic models use a₁ = 0.025
(P068 L10) vs 0.428 (δ = 300). The 1 TeV time PDFs are used for all masses.

*Astrophysical band (P068).* (i) Modulation: (F − 1) is multiplied by a log-normal ξ with σ = ½ ln(a₁⁸⁴/a₁¹⁶) from P068's
a₁ bands (300: 0.376–0.877; 350: 0.78–1.74; 366: 1.00–1.79; 380: 1.20–1.81). (ii) Extension: v_esc dominates the position
of a spectrum relative to the kinematic ceiling (P068 δ_max 68 % ±31 keV), so g_X of every inelastic model with δ ≥ 250 keV
is multiplied by h(δ + Δ)/h(δ) with Δ ~ N(0, 25 keV). The 200–270 keV prediction is astrophysics-independent to first
order (same detector, same halo, rate fixed by the observed count): the "no astro spread" variant reproduces the baseline to
0.001 in every P(N) there. The extension mean is *tail-dominated* by the exponential rise of h(δ) (mean 4.5 events at
2.8 t·yr with the shift vs 0.46 without); we therefore quote medians and probabilities for the extension.

*P038 sideband variant.* If the 600–1000 phd NR band of the first run is empty (P038's reading of Fig. S4), the DM posterior
acquires the factor exp(−s g_X) per model; implemented as importance weights on the Monte Carlo (400 000 draws of
model, s, ξ, Δ).

## 5. Results

### 5.1 Predictive counts (baseline DM: uniform model prior, log-uniform s, astro on; mixture = P061 median [68 %])

200–270 keV NR band (600 phd ROI):

| E (t·yr) | ⟨N⟩_DM | DM P(0)/P(1)/P(≥2) | P038-wt P(0) | B P(1) / P(≥2) | mixture P(0) / P(1) / P(≥2) |
|---|---|---|---|---|---|
| 1.0 | 0.210 | 0.849 / 0.113 / 0.038 | 0.913 | 2.0e-4 / 2e-8 | 0.887 [0.873, 0.927] / 0.084 [0.054, 0.094] / 0.029 [0.019, 0.033] |
| 2.8 | 0.378 | 0.752 / 0.169 / 0.079 | 0.832 | 5.6e-4 / 1.6e-7 | 0.788 [0.759, 0.865] / 0.110 [0.072, 0.123] / 0.101 [0.062, 0.119] |
| 5.6 | 0.756 | 0.621 / 0.207 / 0.173 | 0.719 | 1.1e-3 / 6.3e-7 | 0.712 [0.676, 0.814] / 0.102 [0.068, 0.112] / 0.183 [0.115, 0.212] |
| 10 | 1.406 | 0.492 / 0.215 / 0.293 | 0.595 | 2.0e-3 / 2.0e-6 | 0.657 [0.617, 0.777] / 0.081 [0.056, 0.090] / 0.257 [0.164, 0.293] |

(the P(≥2) bands are quantiles of P(≥2) itself, not 1 − q(P0) − q(P1); `P081_predictive_counts.csv`, columns `Pge2_mix_lo/hi`). Whole S1c > 500 phd panel background: P(≥1) = 0.37/1.0/2.1/3.7 %.
Steady-unknown predictive alone: P(0)/P(1)/P(≥2) = 0.740/0.193/0.068 (1.0), 0.504/0.250/0.246 (2.8), 0.336/0.223/0.440
(5.6), 0.221/0.172/0.607 (10 t·yr). The mixture's P(≥2) exceeds the DM branch's because P061's posterior puts 0.79 on U,
half of it steady with an Exponential(1) rate.

55–270 keV NR band (whole extended ROI above the old 55 keV edge):

| E | ⟨N⟩_DM | DM P(0)/P(1)/P(≥2) | P038-wt P(0) | B P(0) | mixture P(0)/P(1)/P(≥2) |
|---|---|---|---|---|---|
| 1.0 | 0.483 | 0.683 / 0.211 / 0.106 | 0.748 | 0.958 | 0.844 / 0.120 / 0.036 |
| 2.8 | 1.028 | 0.499 / 0.246 / 0.255 | 0.561 | 0.888 | 0.689 / 0.181 / 0.129 |
| 5.6 | 2.057 | 0.336 / 0.218 / 0.446 | 0.395 | 0.788 | 0.553 / 0.210 / 0.237 |
| 10 | 3.759 | 0.219 / 0.167 / 0.614 | 0.268 | 0.654 | 0.425 / 0.230 / 0.344 |

B here is dominated by the estimated b_L2 = 0.10 (P(≥1 | B) = 0.11 at 2.8 t·yr; 0.04–0.28 for the 0.03–0.3 bracket).

55–420 keV with a 1000 phd edge: DM P(0) = 0.597/0.413/0.267/0.171 (P038-weighted 0.701/0.504/0.340/0.224); B P(0) =
0.958/0.886/0.784/0.648. Extension 270–420 keV alone (annual window): median ⟨N_X⟩ = 0.057/0.159/0.318/0.567, DM P(0) =
0.856/0.736/0.625/0.519, P(1) = 0.089/0.144/0.175/0.187; P038-weighted P(0) = 0.919/0.850/0.750/0.640; B P(1) = 1.0e-3/
2.8e-3/5.5e-3/9.8e-3. Class decomposition at 2.8 t·yr (weight, ⟨N_H⟩, P(0)_H, ⟨N_X⟩): elastic-isoscalar 0.25, 0.21, 0.83,
0.17; elastic-isovector 0.17, 0.17, 0.86, 0.19; inelastic-O₁ 0.33, 0.57, 0.66, 8.6; inelastic-O₄ 0.25, 0.43, 0.72, 6.3.

### 5.2 Variants (200–270 keV, 2.8 t·yr): ⟨N⟩ and P(0)

baseline 0.378/0.752; flat prior 0.754/0.584; Jeffreys 0.566/0.660; class model prior 0.454/0.714; no astro spread
0.378/0.752; P038 sideband reweighting 0.213/0.832; annual-average window 0.379/0.752. Full table for all exposures in
`P081_variants.csv` (flat prior at 10 t·yr: ⟨N⟩ = 2.80, P(0) = 0.278, P(≥2) = 0.513).

### 5.3 Best fit versus posterior mean (2.8 t·yr)

200–270 keV: N at the maximum-likelihood rates Σ w_m ŝ_m = 0.383 vs posterior means 0.380 (log-uniform), 0.760 (flat),
0.571 (Jeffreys); 55–270 keV: 1.047 vs 1.034 (log-uniform). The log-uniform prior makes mean = MLE (Exponential posterior);
flat and Jeffreys priors double / ×1.5 the expectation. Companion-free single model: 1.0 (MLE) vs 1.0/2.0/1.5.

### 5.4 Exposure for P(≥1 | DM)

| region / variant | E for P = 0.5 | 0.9 | 0.95 |
|---|---|---|---|
| 200–270 keV, baseline | 9.8 t·yr | 123 t·yr | > 170 |
| 200–270 keV, P038 reweighted | 15.5 | > 170 | > 170 |
| 200–270 keV, flat prior | 4.0 | 31.1 | 57.9 |
| 200–270 keV, plug-in best fit (Σ w_m Poisson(k ŝ_m)) | 6.2 | 33.8 | 55.1 |
| 55–270 keV, baseline | 2.8 | 26.1 | 54.8 |
| 55–420 keV (1000 phd), baseline | 1.9 | 19.3 | 40.8 |

The heavy low-rate tail of the mixture (companion-penalised models, 5 % of the posterior below s = 0.009) makes P(≥1) in the
narrow band approach 0.9 only logarithmically; the plug-in best fit reaches it at 34 t·yr.

### 5.5 Decision table (200–270 keV; BF = P(N | DM)/P(N | ·); posterior = median [68 %] over the P061 hyper-prior)

| E | N | BF_DM:B | BF_DM:U_steady | P(DM | event, N) | at P(DM)=0.5 today |
|---|---|---|---|---|---|
| 1.0 | 0 | 0.85 | 1.15 | 0.014 [0.0015, 0.126] | 0.488 |
| 1.0 | 1 | 561 | 0.58 | 0.025 [0.0022, 0.235] | 0.581 |
| 1.0 | ≥2 | 1.9e6 | 0.56 | 0.025 [0.0021, 0.229] | 0.573 |
| 2.8 | 0 | 0.75 | 1.49 | 0.014 [0.0015, 0.123] | 0.488 |
| 2.8 | 1 | 300 | 0.67 | 0.029 [0.0025, 0.261] | 0.615 |
| 2.8 | 2 | 3.2e5 | 0.40 | 0.018 [0.0015, 0.175] | 0.488 |
| 2.8 | ≥2 | 5.0e5 | 0.32 | 0.014 [0.0012, 0.145] | 0.433 |
| 2.8 | ≥3 | 1.0e9 | 0.24 | 0.011 [0.0009, 0.113] | 0.364 |
| 5.6 | 0 | 0.62 | 1.84 | 0.013 [0.0014, 0.111] | 0.463 |
| 5.6 | 1 | 184 | 0.93 | 0.039 [0.0035, 0.325] | 0.686 |
| 5.6 | ≥2 | 2.7e5 | 0.39 | 0.017 [0.0015, 0.171] | 0.482 |
| 10 | 0 | 0.49 | 2.23 | 0.011 [0.0012, 0.095] | 0.423 |
| 10 | 1 | 107 | 1.25 | 0.052 [0.0046, 0.389] | 0.744 |
| 10 | 2 | 5.5e4 | 0.82 | 0.035 [0.0031, 0.303] | 0.661 |
| 10 | ≥3 | 1.4e8 | 0.39 | 0.017 [0.0015, 0.169] | 0.478 |

Against the modelled background (and the one-off unknown, which predicts the same counts) one new band event is decisive
(BF ≥ 107); against a steady LZ-specific unknown the counts never help (BF 0.24–2.2), so P(DM | event, N) stays within
×3.5 of today's 0.015 for every count. Zero events in 10 t·yr leave P(DM) at 0.011 — the count alone does not falsify DM
either, because the rate posterior extends to small s.

### 5.6 Energy placement of two new events (2.8 t·yr; regions 55–200 / 200–270 / 270–420 keV with a 1000 phd edge)

| placement | P(· | DM) [P038-wt] | P(· | B) | P(· | U_steady) | BF_DM:B | BF_DM:U_s | P(DM | ·) median |
|---|---|---|---|---|---|---|---|
| both 200–270 | 0.0272 [0.0122] | 1.4e-7 | 0.110 | 1.9e5 | 0.25 | 0.011 |
| 200–270 + 55–200 | 0.0384 [0.0332] | 5.9e-5 | 0.0263 | 651 | 1.46 | 0.061 |
| both 55–200 | 0.0584 [0.0614] | 6.2e-3 | 3.1e-3 | 9.4 | 18.7 | 0.153 |
| 200–270 + 270–420 | 0.0184 [0.0129] | 1.4e-6 | 6.1e-4 | 1.3e4 | 30 | 0.571 |
| both 270–420 | 0.0128 [0.0074] | 3.4e-6 | 1.7e-6 | 3.8e3 | 7.5e3 | 0.986 |
| 55–200 + 270–420 | 0.0209 [0.0198] | 2.9e-4 | 1.5e-4 | 72 | 143 | 0.581 |
| one event, 200–270 only | 0.0922 [0.0696] | 5.0e-4 | 0.222 | 185 | 0.42 | 0.018 |
| nothing anywhere 55–420 | 0.413 [0.504] | 0.885 | 0.446 | 0.47 | 0.93 | 0.009 |

Energy placement, not counting, separates DM from a steady LZ-specific unknown: events in the 270–420 keV extension (which
LZ has not yet analysed and which the inelastic posterior populates) or in the 55–200 keV shoulder carry BF 19–7500.

## 6. Validation and cross-checks

* P027 predictive for 6.76 t·yr, 200–270 keV, no season/astro: ⟨N⟩ = 0.915, P(0) = 0.582, P(≥2) = 0.206 vs P027's
  0.92 / 0.58 / 0.20.
* P069 LZ-untouched (6.76 t·yr, best fit, 200–270 keV): L10 0.815 vs ours f_hi k = 0.848 (+4 %); δ = 300 keV 0.425 vs
  0.464 (+9 %); δ = 350 keV 1.203 vs 1.546 (+29 %; P069 uses P050's 12-day annual halo and hard/erf edges; P068's
  annual/Sun frame ratio ×1.7 at 350 keV brackets the difference). With a 400 keV edge: 1.485/0.759/3.487 vs
  1.657/0.857/5.288. P069's joint-likelihood rate for L10 (ŝ_H k = 0.99) exceeds its plug-in because P027's s
  parametrisation normalises the *band* count, not the ROI.
* P020 L10 whole-ROI plug-in 2.38 events (6.76 t·yr) vs ours ŝ/f_hi k = 2.78: the P016/P027 likelihood fits L10 with
  ŝ_H = 0.417 against a band-normalised 0.356 × 1.0, the difference being the one-dimensional few-bin likelihood.
* P034: a₁ = 0.428/0.764/1.179/1.521/1.656 at δ = 300/330/350/366/380 keV enter through the time PDFs; the first-run window
  factor is 1.000 ± 0.003, confirming P020's statement that the extra summer weight is a property of the *next* window.
* P038: h(δ) reproduces P038's extra-event table within 5 % (Sec. 4).
* P061: median/68 %/fraction > 0.5 reproduced (0.0153 / 0.0016–0.135 / 2.4 %).
* Marginal likelihoods: ratio to P027's `B_logu_rho` median 1.0000 (0.9605–1.0000).

## 7. Robustness

* b_H bracket (3 × 10⁻⁴ – 3.5 × 10⁻³) rescales B's P(1) linearly (2.9 × 10⁻⁴ – 3.4 × 10⁻³ at 2.8 t·yr) and BF_DM:B inversely
  (≥ 50 for N = 1 at 10 t·yr with the upper bracket); no qualitative change.
* Class model prior: ⟨N⟩_H +20 %, P(0) 0.714 vs 0.752 at 2.8 t·yr.
* Flat prior on s: doubles ⟨N⟩ and brings P(≥1 | DM) = 0.9 to 31 t·yr; this is the Poisson–Gamma statement of P020/P069.
* P038 sideband: ⟨N⟩_H −44 %, P(0) = 0.832 at 2.8 t·yr; extension medians halve.
* U_steady localisation: spreading the steady unknown uniformly over 55–420 keV would lower BF_DM:U_s for placements outside
  200–270 keV by ≈ ×3 and raise it inside; the qualitative table is unchanged.
* f_trans = 1 (one-off only): the mixture collapses onto DM + B, and P(DM | N = 1, 2.8 t·yr) becomes 0.82 at the P061
  median; f_trans = 0 gives 0.02. This is the single most important judgemental input (P061).
* The 1.0 t·yr window's seasonal boost (×2.0–2.6 at δ ≥ 350 keV) raises the inelastic classes' ⟨N_H⟩ by 55 % relative to an
  annual window (0.210 vs 0.135); a release not starting 1 April 2024 should use the annual-window row.

## 8. Failed or abandoned approaches

* Using P027's `post_uniform` rows alone left 114 elastic models without spectra (39.5 % of weight); resolved by loading
  P008's cache, which P027 itself used.
* A first version anchored the P069 comparison on every "untouched" dataset row (three experiments) instead of LZ's; fixed.
* Quoting the astro-shifted extension *mean* was abandoned (tail-dominated: 4.5 vs median 0.16 at 2.8 t·yr); medians and
  probabilities are reported.

## 9. Figures

* `figures/P081_fig1_predictive_counts.png` — P(N = 0, 1, ≥ 2) in the 200–270 keV band for 1.0/2.8/5.6/10 t·yr under DM
  (P027 posterior), the P061 corpus mixture (bars = median, whiskers = 68 % of the hyper-prior) and the modelled background.
* `figures/P081_fig2_exposure_and_decision.png` — left: P(≥ 1 | DM) vs additional exposure for the three regions, the
  plug-in best fit and the P038-reweighted posterior, with the 0.9 line; right: P(DM | first event, N new band events) vs
  exposure for N = 0, 1, 2, ≥ 3 (medians; 68 % band shown for N = 1), against today's 0.015.

## 10. References

LZ Collaboration, arXiv:2609.02823 (2026). Kass & Raftery, JASA 90, 773 (1995). Cowan, Cranmer, Gross & Vitells, EPJC 71,
1554 (2011). Gelman et al., *Bayesian Data Analysis* (3rd ed., CRC 2013) — posterior predictive checks. Baxter et al., EPJC
81, 907 (2021). Lindley, Ann. Math. Stat. 27, 986 (1956). Corpus: P001, P004, P006, P010, P016, P020, P022, P027, P034,
P038, P050, P061, P068, P069, P070.

## 11. Tools and provenance (mirrors `output/provenance/P081.json`)

* Agent tools: Read ×22 (PAPER_GUIDE; ledger in 6 pages; papers P001, P020, P027, P038, P061, P068, P069; P027 script ×2;
  one persisted tool output; 5 figure views), Bash ×20 (listings, cache inspections, greps of tex/dossier/P016/P061, four
  script runs, one P008/P069 check, table prints, four word counts), Write ×5, Edit ×25 (12 script, 2 details, 10 paper,
  1 provenance), Skill ×1 (dataviz; JS validator skipped per PAPER_GUIDE).
* Software: python 3.12.13; numpy 2.5.3 (default_rng, quantile, trapezoid); scipy 1.18.1 (stats.norm, special.erf/
  logsumexp/gammaln); pandas 3.0.5; matplotlib 3.11.2 (Agg); common/lzcommon.py (imported; no numerical call). WimPyDD
  spectra enter only through the P027/P008/P038/P050 caches (not re-run).
* Recalled knowledge (5, all certain): Poisson–Gamma = negative binomial; log-uniform prior + one event → Exponential
  posterior; 1 April 2024 = day 92 of a leap year and calendar arithmetic; Bayes-factor/posterior-odds algebra; Okabe–Ito
  palette (cosmetic).
* Datasets: none. Data requests: none. WimPyDD-generated files: none.
