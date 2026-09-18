# P034 — Annual modulation as the decisive test for inelastic dark matter near the kinematic edge: research record

Simulated date 2026-09-09. Author profile: direct-detection statisticians / phenomenologists. Category PROJ. Refines and competes with P006 (single-event date likelihood ratio, N_3σ from dates alone) and P020 (seasonal clustering in the untouched exposure); uses P018 (halo-tail spread), P021 (likelihood peak at δ ≈ 380 keV), P028 (non-sinusoidal spike, S_m/S₀), P002 (kinematics), P007 (Higgsino window, fixed coupling).

Scripts: `output/code/P034_modulation_test.py` (all physics and statistics; 97 s), `output/code/P034_postprocess.py` (derived numbers). Results: `output/work/P034/` (`P034_results.json`, `derived_numbers.json`, `time_pdf_summary.csv`, `time_pdfs.csv`, `required_exposure.csv`, `discovery_scan.csv`, `degradations.csv`, `run_log.txt`, `figures/`).

## 1. Motivation and framework

Every inelastic reading of the LZ event (P002, P007, P011, P021) puts the splitting δ within 10–90 keV of the kinematic ceiling at 1 TeV (δ_max(248 keV) = 387 keV on 16 June, P002/P006). There the rate lives in the last tens of km/s of the halo tail, so it is strongly seasonal: P006 quotes modulation fractions 41/90/100 % at δ = 300/350/380 keV, P020 finds 59–78 % of events in May–July for δ ≥ 350 keV, and P028 finds a non-sinusoidal spike (first-harmonic amplitude S_m exceeding the mean S₀ by 1.3–1.9). P006 estimated ~107 (δ = 300) or ~18 (δ = 350 keV) events for a 3σ time-dependence detection "from dates alone", but its δ ≥ 350 keV time PDFs predate the `wd_halo` v_min-grid fix (coordinator note; P020 found them too flat). None of the earlier papers asked the projection question properly: *what exposure*, for *which test statistic*, against *which alternative*, and how robustly.

Two alternatives are distinct and matter differently:

* **(a) A time-flat population of the same rate.** This is the alternative that a rate measurement cannot address: an unmodelled background (wall MSSI, ER tail, accidentals) or an elastic dark-matter interaction (SI modulation 2.4 %, P006) both produce time-flat high-energy NR-band events. Here timing is the *only* discriminant.
* **(b) The modelled background only** (0.0106 events per 2.84 t·yr in the whole S1c > 500 phd panel; ~2 × 10⁻⁴ in the event's neighbourhood; dossier). Here the rate alone is already extremely discriminating and the question is how much timing adds.

We normalise every δ to LZ's best fit, 1.0 event per 2.84 t·yr (Table I; two-sided 68 % band 0.30–2.36 events = the n = 1 Poisson interval, P020). Then the number of events N is the only quantity the physics fixes and the exposure is E = N × 2.84 t·yr / μ.

### 1.1 Time PDF

For a time-independent detector and uniform livetime the arrival-time density of signal events over the year is

p(t | δ) = R_ROI(t; δ) / ∫₀ᵀ R_ROI(t'; δ) dt',  R_ROI(t) = ∫ dE ε(E) dR/dE(E, t),  T = 365.25 d,

with ε(E) the P006/P020 parametrisation of LZ Fig. S2 (50 % at 5.4 keV via a logistic of width 1 keV, plateau 0.96, roll-off Φ((269.9 − E)/15 keV)). The flat alternative is p₀ = 1/T. With a livetime function L(t) ∈ {0, 1} both densities are multiplied by L and renormalised.

dR/dE is computed with WimPyDD (shell-model O₁ isoscalar response, natural xenon, m_χ = 1000 GeV, spin 1/2) as a per-stream kernel: `WD.diff_rate(..., vmin=VGRID, delta_eta=ONES, sum_over_streams=False)` returns K(E, v_i) such that dR/dE(E, t) = Σ_i K(E, v_i) Δη_i(t). The halo weights Δη_i(t) come from `lz.wd_halo(day_of_year=t, vmin=VGRID, ...)` (Baxter-2021 SHM: v₀ = 238, v_esc = 544 km/s, solar peculiar velocity (11.1, 12.2, 7.3) km/s, Earth orbit included by WimPyDD). VGRID = 0–900 km/s in 0.5 km/s steps (1801 streams), well beyond v_max ≈ 811 km/s (and beyond v_max for the v_esc = 560 km/s variant); the explicit grid is the P002 fix. E grid 60–400 keV in 2 keV steps (171 points). Days: 36 equally spaced (10.14 d) plus days 152 and 167.9; the annual curve is a periodic cubic spline evaluated on a 0.25 d grid (1461 points).

Validation: K @ Δη reproduces `lz.wd_rate` to 1.00000 (probe, δ = 350 keV, 200 keV, 16 June); a 73-day grid deviates from the 36-day spline by at most 9.4 × 10⁻⁵ of the peak at δ = 380 keV (the sharpest curve). The absolute unit-coupling normalisation is irrelevant for the shapes; it is used only for (i) the halo-rate rescaling discussion and (ii) a check against P018.

### 1.2 Fourier decomposition and duty cycle

Writing T p(t) = 1 + Σ_k a_k cos(kω(t − t_k)), a_k = |2 ∫ p(t) e^{−ikωt} dt|. The variance of T p about 1 equals Σ a_k²/2, so the fraction of the "power" in the fundamental is a₁²/(2 Var). a₁ is the DAMA-style S_m/S₀ (P028). Duty cycle: fraction of the year with R > 10 % (and 1 %) of the peak; zero fraction: R ≤ 10⁻⁴ of peak. KL = ∫ p ln(p/p₀) dt is the expected single-event log-likelihood ratio under the signal (P006's "information content").

### 1.3 Test statistics against the flat alternative

For N events with times t_i, all three tests are sums of a per-event statistic s(t) (or a count):

* **(i) Cosine amplitude, known phase**: s_c(t) = cos(ω(t − t₁)) with t₁ the phase of the model's fundamental (day 151.9 for every δ). Under the flat null E[s_c] = 0, Var = 1/2; under the signal E[s_c] = a₁/2. Hence the textbook estimate Z ≈ (N a₁/2)/√(N/2) = a₁√(N/2), i.e. N_Zσ ≈ 2(Z/a₁)² (recalled, certain). For a pure cosine of fractional amplitude A, a₁ = A. For a spike a₁ can exceed 1 (limit 2), so the estimator does *not* saturate — but it ignores the harmonics.
* **(ii) Unbinned likelihood ratio**: s_ℓ(t) = ln[p(t)/p₀(t)]. Times where p/p₀ < 10⁻⁴ (the zero-rate season at δ = 380 keV; 9 % of the year under the flat null, 8 × 10⁻⁷ of the signal probability) are treated as an absorbing "dead" state: a null event landing there gives s_ℓ = −∞ and the null experiment can never exceed the signal median. This is the Neyman–Pearson optimal test for simple hypotheses (recalled, certain).
* **(iii) Summer-window counting**: k = number of events in a window W; under the null k ~ Binom(N, f₀ = |W ∩ live|/|live|), under the signal Binom(N, f₁ = ∫_W p). Windows: May–August (days 121–243; f₀ = 0.337), May–July (121–212; 0.252), June (152–181; 0.082) and the optimal single window {t : p(t) > p₀} (f₀ = 0.48/0.46/0.42/0.37/0.32 for δ = 300/330/350/366/380 keV). Critical value k_crit = smallest k with P(K ≥ k | f₀) ≤ α, α = Φ(−3) = 1.35 × 10⁻³ or Φ(−5) = 2.87 × 10⁻⁷ (no randomisation, so the test is conservative); required N = smallest N with power P(K ≥ k_crit | f₁) ≥ 0.5.

**Median significance, exactly.** For (i) and (ii) we do not rely on Gaussian asymptotics. The per-event statistic is binned on 800 values; its probability mass under the signal (weights p Δt) and under the null (weights p₀ Δt, excluding dead bins) gives two single-event pmfs; the pmf of the N-fold sum is the N-th convolution power, computed exactly with an FFT (`scipy.fft.rfft`, length N(L−1)+1, real arithmetic; the null pmf sums to 1 − q_dead so its N-th power automatically carries the survival factor (1 − q_dead)^N). The median of the sum under the signal defines the "median experiment"; the null tail probability at that median gives p and Z = Φ⁻¹(1 − p). Z(N) is evaluated on N = 1…40 (step 1), 45…100 (5), 110…300 (10), 320…1000 (20), 1050…3000 (50) and the required N for 3σ/5σ is interpolated linearly between grid points.

**Toy cross-check** (10⁴ toys, N = 2, 3, 5, 10, 20, 30, each δ; inverse-CDF sampling; `P034_results.json → toy_checks`). Medians agree to ≤ 2 % (e.g. δ = 380, N = 5: LLR 4.870 toy vs 4.851 exact; cosine 4.247 vs 4.244); tail probabilities agree within toy statistics wherever p ≳ 10⁻³ (δ = 350, N = 10: 0.0025 vs 0.0026; δ = 380, N = 5: 0.0021 vs 0.0018; δ = 300, N = 30: 0.0453 vs 0.0470). Below p ~ 10⁻⁴ only the exact method resolves the tail, which is why the 5σ requirements come from the convolution and not from toys.

Gaussian comparison: P006's formula N_3σ = 9 Var_null(s_ℓ)/(E_sig − E_null)² is also evaluated (it ignores the dead-zone bonus and the non-Gaussian null at small N).

### 1.4 Discovery against background only

Simple-vs-simple extended likelihood: H₁ has s = μ k signal events (time density p) plus b = b₀ k background events (flat), H₀ has b only, with k = E/2.84 t·yr, μ = 1.0, b₀ = 0.0106 (panel) or 2 × 10⁻⁴ (neighbourhood). The log-likelihood ratio is

q = 2 ln(L₁/L₀) = 2[−s + Σ_i ln(1 + (s/b) T p(t_i))].

Median of q under H₁ from 2 × 10⁴ toys (N ~ Poisson(s + b), times from the mixture). Exact null tail: P(q ≥ q_med | H₀) = Σ_n Poisson(n | b) P_n(Σ u_i ≥ q_med/2 + s) with u = ln(1 + (s/b) T p(t)) and P_n from the n-fold convolution of the single-event pmf of u under flat times (n ≤ 29). The rate-only comparison uses the count N as statistic with p = P(N ≥ N_med | b), N_med the median of Poisson(s + b). The exposure grid is k = 0.25…20 in steps of 0.25 (0.71 t·yr); "required exposure" is the first grid point whose median experiment reaches the target Z, so exposures are quantised at 0.71 t·yr and the "gain" factor accordingly.

### 1.5 Degradations

* **Halo variants** (shape and rate): truncated Maxwellians with v_esc = 528/560 km/s and v₀ = 220/250 km/s built with `lz.wd_halo(vesc=…)`/`(v0=…)` on the same 36 days; nominal efficiency. P018's Lisanti-k and SHM++ tails are not available through the wrapper; their *rate* effect is taken from P018's tables (`rate_spread_summary.csv`, `rate_ratios_isoscalar.csv`, m = 1000 GeV, annual epoch), and their shape effect is bracketed by the v_esc/v₀ variants.
* **Efficiency roll-off**: Φ((edge − E)/width) with (edge, width) = (269.9, 15) nominal, (269.9, 5), (250, 15), (290, 15), and no roll-off (plateau 0.96 up to 400 keV, which covers the full kinematic window at every δ considered).
* **Livetime gaps**: a 21-day dead block (≈ 3 weeks/year of calibration) centred on the peak (days 141–162), in December (335–356), in March (60–81), and averaged over 12 equally spaced start days. For the cosine estimator we also record the null-mean bias E₀[s_c]/√Var₀[s_c] of a *naive* fit that does not subtract the livetime-weighted mean (the induced spurious significance is this number × √N).

### 1.6 Calendar translation

LZ: 4.71 t fiducial × live fraction 220/371 = 0.593 → 2.79 t·yr per calendar year (P020), new data since 1 April 2024; the first 2.84 t·yr count towards N. Year reached = 2024.25 + (E − 2.84)/2.79. Milestones: 2 Sep 2026 (6.76 + 2.84 = 9.60 t·yr), end-2027 (10.47 t·yr; P020), 1000 live days (12.9 t·yr at 4.71 t; the 1000-live-day design goal is recalled, likely), end-2028 (16.1 t·yr). "60 t detector": 60 t × assumed 0.80 live fraction = 48 t·yr per year, LZ-like acceptance (assumption; XLZD-scale, recalled uncertain).

## 2. Results

### 2.1 Time PDFs (`time_pdf_summary.csv`, Fig. 1)

| δ (keV) | peak day | mod. fraction | June/Dec | duty >10 % | duty >1 % | zero frac | peak/mean | a₁ | a₂ | a₃ | a₂/a₁ | power in a₁ | KL (nats) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | 152 | 0.419 | 2.36 | 1.00 | 1.00 | 0 | 1.45 | 0.428 | 0.023 | 0.000 | 0.054 | 1.00 | 0.047 |
| 330 | 152 | 0.700 | 5.29 | 1.00 | 1.00 | 0 | 1.86 | 0.764 | 0.094 | 0.002 | 0.123 | 0.99 | 0.153 |
| 350 | 152 | 0.932 | 24.6 | 0.69 | 1.00 | 0 | 2.50 | 1.179 | 0.294 | 0.025 | 0.249 | 0.94 | 0.391 |
| 366 | 152 | 0.995 | 305 | 0.50 | 0.83 | 0 | 3.39 | 1.521 | 0.684 | 0.171 | 0.450 | 0.82 | 0.717 |
| 380 | 152 | 1.000 | 4.8 × 10⁴ | 0.41 | 0.63 | 0.15 | 4.22 | 1.656 | 0.962 | 0.420 | 0.581 | 0.71 | 0.903 |

The fundamental phase is day 151.9 for every δ (1 June; P028 finds 151.9 for iodine). Comparison with the corpus: P020 (post-fix) mod. fraction 0.4175/0.930/0.9946/0.99998 and June/Dec 2.40/26.5/345/∞ at δ = 300/350/366/380 keV — ours 0.419/0.932/0.995/1.000 and 2.36/24.6/305/4.8 × 10⁴ (the last is finite only because our spline floor is 10⁻⁴ of the peak; P020 quotes ∞); P006 (pre-fix) 0.41/0.90/1.00 at 300/350/380. P028's iodine a₂/a₁ = 0.06/0.28/0.48/0.81 vs our xenon 0.05/0.25/0.45/0.58; P028's S_m/S₀ = 0.46/1.27/1.59/1.87 vs our a₁ = 0.43/1.18/1.52/1.66 (different target and window; same ordering and non-sinusoidality). KL per event: ours 0.047/0.39/0.72/0.90 vs P020 0.043/0.33/0.58/0.74 and P006 0.043/0.30/–/0.60: our shapes are 15–20 % more informative at δ ≥ 350 keV than P020's, most likely from our finer energy grid (2 keV to 400 keV vs 3 keV to 330 keV) near the roll-off, where the winter rate is decided. Unit-coupling counts per 2.84 t·yr (annual mean): 1.24 × 10¹³, 1.83 × 10¹², 2.39 × 10¹¹, 1.95 × 10¹⁰, 6.7 × 10⁸ — the ratios 350/300 = 0.019 and 380/300 = 5.4 × 10⁻⁵ compare with P002's 0.032 and 1.3 × 10⁻⁴ (16 June halo, no efficiency) and P021's κ̂ ratios.

### 2.2 Tests against a time-flat population (`P034_results.json → tests_vs_flat`, Fig. 3)

| δ (keV) | a₁ | N_3σ LLR | N_5σ LLR | N_3σ cos (exact) | N_5σ cos | N_3σ cos (a₁√(N/2)) | N_3σ LLR Gaussian (P006 formula) | N_3σ/N_5σ May–Aug | May–Jul | June | optimal window |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | 0.428 | 96.5 | 268 | 96.9 | 269 | 98.1 | 99.0 | 194/539 | 197/544 | 619/1710 | 122/335 |
| 330 | 0.764 | 29.4 | 81.8 | 29.6 | 82.4 | 30.9 | 31.8 | 60/170 | 62/166 | 193/523 | 39/106 |
| 350 | 1.179 | 11.5 | 32.0 | 11.7 | 32.4 | 12.9 | 14.0 | 26/68 | 23/63 | 75/197 | 17/43 |
| 366 | 1.521 | 6.5 | 17.6 | 6.5 | 17.7 | 7.8 | 8.7 | 14/37 | 11/32 | 34/92 | 10/22 |
| 380 | 1.656 | 5.3 | 14.1 | 5.3 | 14.1 | 6.6 | 10.6 | 12/28 | 10/23 | 25/61 | 6/19 |

Window fractions f₀ → f₁ (May–Aug): 0.337 → 0.441/0.526/0.640/0.749/0.807; optimal window f₀ → f₁: 0.483 → 0.620, 0.461 → 0.706, 0.424 → 0.809, 0.366 → 0.882, 0.318 → 0.896. (P020's 41/59/71/78 % "May–July" fractions refer to its specific 884-day window containing three summers, where the flat fraction is 0.31, not to a full year, where it is 0.25 and ours are 0.35/0.55/0.69/0.77.)

Findings.

1. **The full likelihood ratio and the known-phase cosine estimator are equivalent to ≤ 3 % in N** at every δ, even though the fundamental carries only 71 % of the variance at δ = 380 keV. Both per-event statistics are monotonic in |t − t_peak|; the exact median-experiment significance depends weakly on the exact functional form. The *rule* Z = a₁√(N/2), however, overstates N by 6–25 % at small N because the null distribution of a sum of a few bounded cosines is lighter-tailed than a Gaussian near its edge. P028's inference that "a cosine fit captures only 48–93 % of the variance" is correct about the *amplitude* but does not translate into a loss of *test power* when the phase is known.
2. P006's N_3σ = 107 (δ = 300) and 18 (350) become 96.5 and 11.5: the δ = 300 keV change is the difference between the Gaussian formula (99) and the exact small-N distribution plus a slightly larger a₁; the δ = 350 keV change is mostly the corrected (sharper) halo-grid time PDF (KL 0.39 vs 0.30). The P006 Gaussian formula applied to our PDFs gives 99/32/14/8.7/10.6 — it *fails* at δ = 380 keV (10.6 vs exact 5.3) because it ignores the zero-rate season, where a single flat-null event refutes the model.
3. Summer counting needs ×2 the events of the likelihood ratio (May–Aug), ×1.2–1.5 with the optimal single window; the June-only window is ×5–6 worse.

### 2.3 Required exposure and calendar (`required_exposure.csv`, `derived_numbers.json`, Fig. 2)

E = N × 2.84 t·yr / μ, μ = 1.0 (68 % band μ = 2.36 → ×0.42, μ = 0.30 → ×3.3).

| δ (keV) | E_3σ LLR (t·yr) [band] | E_5σ LLR [band] | E_3σ May–Aug | E_3σ optimal | LZ years (3σ/5σ) | LZ calendar year 3σ / 5σ | 60 t yr (3σ/5σ) |
|---|---|---|---|---|---|---|---|
| 300 | 274 [116–914] | 762 [323–2540] | 551 | 347 | 98 / 273 | 2121 / 2297 | 5.7 / 15.9 |
| 330 | 84 [35–278] | 232 [99–775] | 170 | 111 | 30 / 83 | 2053 / 2106 | 1.7 / 4.8 |
| 350 | 33 [14–109] | 91 [39–303] | 74 | 48 | 11.7 / 32.5 | 2035.0 / 2055.8 | 0.68 / 1.9 |
| 366 | 18.6 [7.9–62] | 50 [21–167] | 40 | 28 | 6.6 / 17.9 | 2029.9 / 2041.2 | 0.39 / 1.04 |
| 380 | 15.0 [6.4–50] | 40 [17–134] | 34 | 17 | 5.4 / 14.4 | 2028.6 / 2037.6 | 0.31 / 0.84 |

Median Z against the flat alternative at LZ milestones (LLR; best-fit rate | upper 68 % rate μ = 2.36), δ = 300/330/350/366/380 keV:

| milestone | E (t·yr) | best fit | μ = 2.36 |
|---|---|---|---|
| 2 Sep 2026 (6.76 + 2.84) | 9.6 | 0.57/1.03/1.61/2.10/2.35 | 0.87/1.57/2.50/3.33/3.73 |
| end-2027 | 10.5 | 0.59/1.07/1.68/2.20/2.47 | 0.91/1.64/2.60/3.48/3.90 |
| 1000 live days | 12.9 | 0.66/1.19/1.88/2.47/2.76 | 1.00/1.81/2.89/3.88/4.34 |
| end-2028 | 16.1 | 0.73/1.32/2.10/2.78/3.11 | 1.12/2.03/3.23/4.35/4.86 |

So LZ, at its best-fit rate, does not reach 3σ time dependence within a 1000-live-day programme for any δ; it reaches 3σ for δ ≥ 366 keV only if the true rate is near the upper end of its 68 % band, and 5σ never. A 60 t detector reaches 3σ within a year for δ ≥ 350 keV and within 2 years reaches 5σ for δ ≥ 350 keV.

### 2.4 Discovery against background only (`discovery_scan.csv`)

| background b₀ / 2.84 t·yr | δ | E_5σ counting (median N) | E_5σ with time (median N) | gain | E_3σ counting / with time |
|---|---|---|---|---|---|
| 0.0106 (panel) | 300 | 10.65 (4) | 10.65 (4) | 1.00 | 4.97 / 4.97 |
| | 330 | 10.65 (4) | 9.94 (3) | 1.07 | 4.97 / 4.97 |
| | 350 | 10.65 (4) | 9.94 (3) | 1.07 | 4.97 / 4.26 |
| | 366 | 10.65 (4) | 8.52 (3) | 1.25 | 4.97 / 4.26 |
| | 380 | 10.65 (4) | 7.81 (3) | 1.36 | 4.97 / 3.55 |
| 2 × 10⁻⁴ (neighbourhood) | all | 4.97 (2) | 4.97 (2) | 1.00 | 2.13 / 2.13 |

Mechanics (δ = 380, panel): at 7.81 t·yr the median experiment has N = 3, s = 2.75, b = 0.029; counting gives P(N ≥ 3 | b) = 4.0 × 10⁻⁶ (4.46σ), while the time-weighted likelihood ratio gives 2.7 × 10⁻⁷ (5.01σ) because three flat-time background events rarely all fall near June. Counting needs the median N = 4 (10.65 t·yr; P(N ≥ 4 | b = 0.040) = 1.0 × 10⁻⁷, 5.2σ). With the neighbourhood background two events already give 5σ (P(N ≥ 2 | 3.5 × 10⁻⁴) = 6 × 10⁻⁸) at the 4.97 t·yr where the median N first reaches 2, and timing cannot beat one event's 3.5σ ceiling. Exposures are quantised at 0.71 t·yr; the gains are therefore 1.0–1.4 and never more than "one event's worth". Time information is not decisive against the modelled background: the rate is.

### 2.5 Degradations (`degradations.csv`, `derived_numbers.json`)

**Halo tail (shape).** N_3σ (LLR) and a₁:

| variant | δ = 300 | 330 | 350 | 366 | 380 | log₁₀ rate/SHM (annual) at 300/350/380 |
|---|---|---|---|---|---|---|
| SHM (544, 238) | 96.5 (0.43) | 29.4 (0.76) | 11.5 (1.18) | 6.5 (1.52) | 5.3 (1.66) | 0/0/0 |
| v_esc = 528 | 65.2 (0.52) | 16.5 (1.00) | 7.0 (1.47) | 5.1 (1.68) | 4.5 (1.77) | −0.14/−0.65/−1.52 (P018: −0.14/−0.66/−1.65) |
| v_esc = 560 | 131.9 (0.37) | 48.3 (0.60) | 20.6 (0.91) | 10.1 (1.25) | 6.5 (1.53) | +0.10/+0.43/+1.19 (P018: +0.10/+0.43/+1.24) |
| v₀ = 220 | 58.8 (0.55) | 15.2 (1.04) | 6.7 (1.50) | 5.0 (1.69) | 4.2 (1.80) | −0.51/−1.13/−2.12 (P018: −0.51/−1.14/−2.27) |
| v₀ = 250 | 129.4 (0.37) | 43.7 (0.63) | 17.9 (0.97) | 8.8 (1.33) | 6.0 (1.57) | +0.27/+0.55/+1.16 (P018: +0.27/+0.55/+1.21) |

Our WimPyDD rate ratios reproduce P018's independent halo-function construction to ≤ 0.02 dex at δ ≤ 350 keV and to 0.05–0.15 dex at 380 keV (P018 noted WimPyDD's bin-averaged η is 2–5 % high within 30 km/s of v_max; the discrepancy sits where the rate is a pure tail extrapolation). Because the shape is controlled by δ_max − δ and δ_max shifts by 0.82 keV per km/s (P018), a lower v_esc makes a given δ look "closer to the edge": v_esc = 528 km/s (Δv_max = −16 km/s ≈ −13 keV) moves N_3σ(350) from 11.5 to 7.0 ≈ N_3σ(366 keV, SHM); v_esc = 560 moves it to 20.6 ≈ N_3σ(340). So the halo-shape uncertainty on the required *number of events* is a factor 0.6–1.8 at δ = 350 keV, 0.8–1.5 at 366 keV, and it is degenerate with a ±13–16 keV shift in δ. For a model normalised to the observed event this is the whole effect. For a *fixed-coupling* model (e.g. P007's Higgsino) the exposure additionally scales with the inverse rate: P018's Gaia-range spread (m = 1000 GeV, annual) is 0.78/1.69/3.8 dex at δ = 300/350/380 keV, i.e. exposure factors 6/49/6300, which dominate everything else.

**Efficiency roll-off.**

| variant | a₁ (350/366/380) | duty >10 % (350/366/380) | N_3σ LLR (300/330/350/366/380) | unit rate relative to nominal (300/330/350/366/380) |
|---|---|---|---|---|
| nominal (269.9, 15) | 1.18/1.52/1.66 | 0.69/0.50/0.41 | 96.5/29.4/11.5/6.5/5.3 | 1/1/1/1/1 |
| sharp (269.9, 5) | 1.18/1.54/1.72 | 0.69/0.49/0.37 | 96.5/29.4/11.5/6.4/4.8 | 1.00/1.00/0.99/0.97/0.77 |
| edge 250 | 1.19/1.55/1.75 | 0.68/0.48/0.35 | 96.2/29.0/11.2/6.3/4.6 | 0.99/0.98/0.96/0.89/0.60 |
| edge 290 | 1.16/1.44/1.54 | 0.71/0.53/0.50 | 96.9/29.8/12.1/7.3/6.4 | 1.01/1.02/1.06/1.21/2.34 |
| none (to 400 keV) | 0.97/1.19/1.47 | 0.92/0.69/0.54 | 100.6/35.0/17.8/11.3/7.1 | 1.14/1.34/1.98/4.92/28.1 |

The roll-off *enhances* the modulation. Near the ceiling the recoil window at the threshold speed collapses to E* = δμ/m_N (P021): 326 keV at δ = 366, 339 keV at 380 keV (μ = 109 GeV for 1 TeV on A = 131). In winter only the fastest particles scatter and their recoils sit near E*, above the 270 keV edge, so they are removed; in June the window opens down to ~200 keV, inside the ROI. Removing the roll-off therefore lowers a₁(366) from 1.52 to 1.19 and raises N_3σ from 6.5 to 11.3 — but it raises the *rate* by ×4.9 (×28 at 380 keV) at fixed coupling. If the one observed event fixes the coupling, an ROI extended to ~400 keV would reach 3σ time dependence in 6.6 t·yr at δ = 366 keV (×2.8 sooner than 18.6) and 0.72 t·yr at 380 keV (×21 sooner); at δ ≤ 350 keV the gain is only ×1.1–1.3. Conversely, a roll-off 20 keV lower than Fig. S2 implies (edge 250) changes N_3σ by ≤ 13 %. The efficiency thus matters through the normalisation (P020) far more than through the shape.

**Livetime gaps** (21 days; N_3σ LLR / cosine):

| gap | δ = 300 | 330 | 350 | 366 | 380 | naive-cosine null bias (σ × √N) |
|---|---|---|---|---|---|---|
| none | 96.5/96.9 | 29.4/29.6 | 11.5/11.7 | 6.5/6.5 | 5.3/5.3 | 0 |
| on the peak (141–162) | 100.2/100.5 | 29.8/30.0 | 11.3/11.4 | 6.2/6.2 | 5.0/5.0 | −0.089 |
| December (335–356) | 105.9/106.2 | 32.6/32.9 | 12.8/12.9 | 7.0/7.1 | 5.6/5.6 | +0.087 |
| March (60–81) | 91.0/91.2 | 27.7/27.8 | 10.9/11.0 | 6.5/6.5 | 5.4/5.4 | −0.014 |
| random start (mean of 12) | 97.1 (91–106) | 29.6 (28–33) | 11.6 (11–13) | 6.6 (6.0–7.0) | 5.3 (4.8–5.6) | ≤ 0.087 |

A 3-week gap anywhere changes the required N by −8 % to +10 %. Cutting the peak *helps* slightly for δ ≥ 350 keV because the remaining live time contains a larger share of the zero-rate season (flat-null events there refute the model), and because the null's cosine mean becomes negative so the signal–null separation grows once the mean is subtracted. The real danger is a cosine fit that does not subtract the livetime-weighted mean: a June gap biases it by −0.089√N σ (a 21-day December gap by +0.087√N σ), i.e. 0.5σ spurious (anti-)modulation at N = 30 — small for LZ-scale N but not for a 60 t detector.

## 3. Discussion

1. **What is decisive, and against what.** Against the modelled background the rate is decisive on its own: with LZ's panel background, four events discover at 5σ, and timing saves at most one event (≤ 36 % in exposure; nothing for the neighbourhood background, where two events suffice). Against a time-flat *unknown* population of the same rate — the only alternative that a discovery-by-counting cannot exclude — timing is the sole discriminant and the price is 5–12 events (15–33 t·yr) at δ ≥ 350 keV and ~100 events at δ = 300 keV. This reframes P006 and P020: the June date and any future summer clustering are cheap *consistency* checks, but the *decisive* modulation test is a G3 measurement except for δ within ~20 keV of the ceiling.
2. **Estimator choice hardly matters; the null model does.** The unbinned likelihood ratio and a known-phase cosine test are within 3 % in N; summer-window counting costs ×2. The non-sinusoidality found by P028 (a₁ > 1, harmonics with up to 29 % of the variance) is a feature of the *shape* that does not cost a cosine test power. What does matter is (a) knowing the phase (1 June — fixed by the halo, not by the model), (b) modelling the livetime, and (c) using the zero-rate season at δ ≳ 375 keV, where a single winter event refutes the model (P020's "a winter event excludes δ ≥ 366 keV").
3. **Halo and efficiency.** Shape-wise, the halo tail is degenerate with a ±13–16 keV shift in δ (factor 0.6–1.8 in N at δ = 350 keV). Rate-wise, for fixed-coupling models P018's 0.8–3.8 dex dominates. The efficiency roll-off enhances the modulation because winter recoils sit at E* ≈ 326–339 keV; the corollary is that the fastest route to a decisive test is not exposure but an ROI extended past ~300 keV (×2.8 at δ = 366, ×21 at 380 keV in exposure at fixed coupling), provided MSSI/ER backgrounds at S1c > 600 phd can be controlled — a question for LZ, not for us.
4. **Relation to P021.** P021's likelihood peak at δ ≈ 380 keV is where the time test is cheapest (5.3 events) and where the zero-rate season (15 % of the year) makes the test partly deterministic. P007's Higgsino at 366 keV needs 6.5 events for 3σ against a flat alternative — LZ would need to run to ~2030 at the best-fit rate.

## 4. Failed or abandoned approaches

* A first version of the exact convolution multiplied the null tail by (1 − q_dead)^N a second time; the FFT power of an un-normalised pmf already carries the survival factor. Fixed before any result was recorded.
* The livetime-gap block initially produced NaN statistics inside the gap (0/0); gap bins now carry zero weight and unit ratio.
* Toy MC alone cannot resolve 5σ tails (p = 3 × 10⁻⁷) with 10⁴ toys; toys are kept as a cross-check and the exact N-fold convolution supplies the requirements.
* Lisanti-k and SHM++ tail *shapes* were not propagated (not available through `lz.wd_halo`); their rates are taken from P018.

## 5. Figures

* `figures/fig1_time_pdfs.png` — left: T p(t | δ) for δ = 300–380 keV with the flat line and 16 June 2023; right: Fourier amplitudes a₁…a₄.
* `figures/fig2_required_exposure.png` — required exposure for a median 3σ (left) and 5σ (right) time-dependence detection against a flat alternative vs δ, for the unbinned likelihood ratio, the cosine estimator (indistinguishable) and May–August counting; band = 68 % rate interval; right axis LZ calendar years; LZ's 2.84 t·yr and end-2027 exposures marked.
* `figures/fig3_Z_vs_N.png` — median Z(N) from the exact N-fold convolution (solid: likelihood ratio; dotted: cosine).

## 6. References

LZ Collaboration, arXiv:2609.02823 (2026). K. Freese, J. Frieman, A. Gould, PRD 37, 3388 (1988). K. Freese, M. Lisanti, C. Savage, RMP 85, 1561 (2013). D. Tucker-Smith, N. Weiner, PRD 64, 043502 (2001). G. Cowan, K. Cranmer, E. Gross, O. Vitells, EPJC 71, 1554 (2011). D. Baxter et al., EPJC 81, 907 (2021). I. Jeong, S. Kang, S. Scopel, G. Tomar, CPC 276, 108342 (2022) (WimPyDD). Corpus: dossier 00, P002, P006, P007, P018, P020, P021, P028.

## 7. Tools and provenance

Mirrors `output/provenance/P034.json`.

* Agent tools: Read ×17 (PAPER_GUIDE; dossier; results_ledger.csv ×2 pages; P006, P020, P021, P028, P018, P002, P007 papers; lzcommon.py lines 100–390; P020_future_lz.py lines 130–190; three figure PNGs), Bash ×15 (directory listings + ENVIRONMENT_versions + lzcommon index; P018/P020 tables; grep of P018/P021/WimPyDD kernel usage; WimPyDD probe; palette grep; two full runs of P034_modulation_test.py; P018 halo labels; output/toy inspection; post-processing run; five word-count/JSON checks), Write ×5, Edit ×11 (5 script fixes, 4 paper trims, 2 tally updates), Skill ×1 (dataviz).
* Software: python 3.12.13; WimPyDD 2.0.4 (`diff_rate` with `sum_over_streams=False`, `streamed_halo_function` via `lz.wd_halo` with explicit v_min grid, `eft_hamiltonian`); numpy 2.5.3; scipy 1.18.1 (`fft.rfft/irfft/next_fast_len`, `interpolate.CubicSpline`, `stats.norm/binom/poisson`); pandas 3.0.5; matplotlib 3.11.2; common/lzcommon.py (`LZ`, `wd`, `wd_halo`, `wd_hamiltonian`, `wd_rate` for validation).
* Local inputs: LZ paper numbers via lzcommon (exposure 2.84 t·yr, 4.71 t, 220 live days/371 d, efficiency 5.4/269.9 keV and 0.96 plateau, Table I best fit and interval, Fig. 5 panel background); dossier (neighbourhood background 2 × 10⁻⁴); P018 `rate_ratios_isoscalar.csv`, `rate_spread_summary.csv`; P020 `inelastic_seasonal.csv`, `seasonal_visibility.csv`, `exposure_scenarios.csv`, code (efficiency parametrisation); papers P002/P006/P007/P018/P020/P021/P028.
* Recalled knowledge: cosine-estimator variance and Z = a₁√(N/2) (certain); Neyman–Pearson optimality of the likelihood ratio (certain); calendar day numbers of month boundaries (certain); LZ design goal ~1000 live days (likely); XLZD-scale 60 t with ~80 % live fraction (uncertain; assumption); bibliographic details of the six references (likely).
* Datasets: none. Data requests: none. WimPyDD-generated files: none (diff_rate does not write response-function files).
