# P020 — What LZ's data since April 2024 should show: falsifiable predictions for the untouched exposure

Simulated date 2026-09-05. Category PROJ. Author profile: phenomenologists and former experimentalists making projections.
Script: `output/code/P020_future_lz.py` (run from the simulation root with `.venv/bin/python`; ~5.5 min, dominated by WimPyDD).
All tables referenced below are in `output/work/P020/`; figures in `output/work/P020/figures/`; the full console log is `run_log.txt`; every number quoted here is in `P020_results.json` or one of the CSVs.

## 1. Motivation and framework

The LZ paper (arXiv:2609.02823) reports one high-energy NR-band event in 2.84 t·yr (220 live days, 27 March 2023 – 1 April 2024, 4.71 t) and states in its Conclusion that "LZ has continued to take data with the same electric field conditions since 1 April 2024". By the simulated date of this paper (5 September 2026) roughly 2.4 calendar years of data exist that have not been analysed in the high-energy ROI. Because the signal models that fit the event (L10 dipole-like elastic at ≥ 400 GeV; O1/O4 inelastic at δ = 300–390 keV) all predict a *rate* — and the inelastic ones a *season* — the untouched exposure is a pre-registered test with no look-elsewhere effect. We turn the published best fit, the corpus' inelastic spectra (P002, P006, P007) and the corpus' Bayesian framework (P001) into predictions for the number and arrival times of events, and into a decision table for what each observed count would mean.

Two remarks frame everything:
1. The single-event Poisson likelihood L(s) ∝ s e^{−s} has −2ΔlnL = 1 at s = 0.30 and 2.36 (script check, `L10_interval_check`); this reproduces Table I's 1.0 (+1.4, −0.7) to the quoted precision, so the LZ best fit is simply "one event with negligible background". We therefore treat the published interval as the profile likelihood of a unit-efficiency Poisson count.
2. The background near the event is tiny: 2 × 10⁻⁴ (±2σ of the NR median at S1c > 500 phd, dossier), 10⁻³ (LZ-equivalent effective value implied by the 3.4σ local, P001) or 0.0106 (the whole S1c > 500 phd panel, Fig. 5 caption). All three are carried through.

## 2. Inputs

| Input | Value | Source |
|---|---|---|
| First run window | 27 Mar 2023 – 1 Apr 2024 = 371 calendar days, 220 live days | LZ Data Analysis paragraph (tex line 111) |
| Live fraction | 220/371 = 0.593; range 0.50–0.75 explored | derived; range is our assumption |
| Fiducial mass | 4.71 ± 0.08 t (same cuts); 5.5 t scenario if MSSI can be controlled (the 2024 FV was 14.5% larger, i.e. ≈ 5.5 t) | tex line 132 |
| First exposure | 2.84 t·yr | abstract; `lz.LZ['exposure_tyr']` |
| Data since | 1 April 2024, same field conditions | Conclusion (tex line 334) |
| End dates considered | 2 Sep 2026 (arXiv posting of the LZ paper), 31 Dec 2026, 31 Dec 2027 | assignment |
| Best-fit signal | L10s (1000 GeV): 1.0 (+1.4, −0.7) events | Table I (tex line 238) |
| Background, S1c > 500 phd panel | 0.0106 ± 0.0008 | Fig. 5 caption (tex line 262) |
| Background within ±2σ of NR median | 2 × 10⁻⁴ | dossier §2; P001 |
| LZ-equivalent effective b | 10⁻³ | P001 (b_eff from the 3.4σ local) |
| NR efficiency | 0.96 plateau; 50% at 5.4 and 269.9 keV; roll-off Φ((269.9 − E)/15 keV) | Fig. S2 as parametrised by P006 |
| Halo | Baxter-2021 SHM (v0 = 238, vesc = 544 km/s, Sun peculiar (11.1, 12.2, 7.3)) via `lz.wd_halo(day_of_year)` with the explicit v_min grid | lzcommon (recalled, certain) |
| Inelastic spectra | WimPyDD 2.0.4, O1 isoscalar, m = 1000 GeV, δ = 300, 350, 366, 380 keV | `lz.wd_rate` |
| P001 framework | π_DM = 0.01, π_U = 0.10, L_U = 0.10 → μ_U = −ln(0.9) = 0.1054; flat prior s ∈ [0, 10] | P001 §2.5 |
| P001 posterior predictive (equal exposure) | P(0) = 0.84, P(1) = 0.11, P(≥2) = 0.05 | P001 details §3.6 |
| P002/P006 seasonal numbers | June/Dec = 2.4 (δ = 300), 28 (350); modulation fractions 41/90/100% at 300/350/380 keV | P002, P006 |
| P007 Higgsino | δ(N = 1) = 366 keV at 1 TeV; 4–5.2 events per 1000 live days; June/Dec = 46 | P007 |

Recalled knowledge used (all flagged certain): the negative-binomial form of the Poisson–Gamma mixture; the Jeffreys prior s^{−1/2} for a Poisson mean; Wilks' one-sided 90% threshold −2ΔlnL = 2.706; Jeffreys/Kass–Raftery scale (Bayes factor > 10 = "strong"); Gregorian calendar arithmetic (2024 leap year).

## 3. Method and equations

### 3.1 Exposure
E = (calendar days) × f_live / 365.25 × M_fid; k ≡ E/2.84. Calendar days from Python `datetime`.

### 3.2 Background
μ_b = b × k for each of the three b; P(≥1) = 1 − e^{−μ_b}, P(≥2) = 1 − e^{−μ_b}(1 + μ_b).

### 3.3 L10 predictions
- Plug-in: N ~ Poisson((s + b)k) for s = 1.0, and for the interval ends s = 0.3, 2.4.
- Poisson–Gamma predictive: with a flat prior the one-event posterior is Gamma(α = 2, rate 1) (mode 1, mean 2); with Jeffreys' prior Gamma(1.5, 1). Marginalising Poisson(ks) over Gamma(α, 1) gives the negative binomial NB(α, p = 1/(1 + k)): P(N = n) = Γ(n + α)/(n! Γ(α)) p^α (1 − p)^n, mean αk. The flat-prior predictive has mean 2k — twice the plug-in — because the posterior mean of a one-event Poisson rate is 2, not 1; we report both and regard the plug-in as "the best fit" and the predictive as "the best fit with its uncertainty".

### 3.4 Inelastic seasonal PDFs
For each δ the in-ROI rate R(d) = ∫ dR/dE (E; halo(d)) ε(E) dE on a 60–330 keV grid (91 points; inelastic onsets are ≥ 85 keV for δ ≥ 300 keV, P002), evaluated on 54 days of the year (8-day grid plus 152, 152.5, 153, 156, 160, 167.9, 335, 336) and interpolated with a periodic cubic spline (period 365.25 d, clipped at zero). Quantities:
- June/Dec ratio R(167.9)/R(335); modulation fraction (R_max − R_min)/(R_max + R_min); fraction of the year with R > 10⁻³ R_max.
- Efficiency survival at peak: ∫ R ε / ∫ R; fraction of the un-efficiencied spectrum above 270 keV.
- Calendar-time PDF over a window [t₀, t₁] with uniform livetime: p(t) = R(doy(t)) / ∫ R dt. The expected number of events in the new window, given exactly one expected event in the first window (uniform livetime, same mass), is N_new = ∫_new R dt / ∫_first R dt. This differs from k because the new window (884 d) contains three summers.
- Month fractions: f(May–Jul), f(Nov–Jan), f(May–Aug) from p(t) and from the flat PDF.
- Information per event: KL = ∫ p ln(p/p_flat) dt and its sd.
- Visibility: for N events drawn from p (40 000 toys) or from flat, the log-likelihood ratio ln LR = Σ ln[p(t_i)/p_flat(t_i)]; we tabulate P(LR > 10) and P(LR > 3) under both hypotheses (power and false-positive rate) and, as a model-independent observable, P(all N events fall in May–August).

### 3.5 Decision table (P001 framework)
Likelihoods for the first event (n₁ = 1) and the new count n₂ with exposure ratio k:
L_DM = ∫₀¹⁰ P(1 | s + b) P(n₂ | k(s + b)) ds/10; L_K = P(1 | b) P(n₂ | kb); L_U = P(1 | b + μ_U) P(n₂ | k(b + μ_U)).
Posterior P(h) ∝ π_h L_h with π = (0.01, 0.10, 0.89). Also: the Bayes factor of the second exposure alone, BF₂(DM:K) = E_post[P(n₂ | k(s + b))]/P(n₂ | kb), where the expectation is over the s-posterior from the first event.

Frequentist: local significance of the combined sample, Z = Φ⁻¹(1 − P(≥ 1 + n₂ | b(1 + k))), and of the new exposure alone, Z_new = Φ⁻¹(1 − P(≥ n₂ | bk)) — the latter is free of any look-elsewhere effect because the region and models are fixed in advance. The smallest n₂ giving Z ≥ 5 is tabulated.

N = 0: classical 90% CL upper limit from the new exposure alone, s < 2.303/k (per 2.84 t·yr, b ≈ 0); combined with the first event, L(s) ∝ (s + b) e^{−(s + b)(1 + k)}: Bayesian 90% UL with flat prior (incomplete-gamma inversion) and a one-sided profile-likelihood UL (−2ΔlnL = 2.706 from ŝ = 1/(1 + k) − b). The "fraction of the Table I interval excluded" is (2.4 − UL_new)/(2.4 − 0.3).

### 3.6 Corpus-prior mixture predictive
Posterior weights after the first event w = (P(DM), P(U), P(K)) = posterior with n₂ absent; predictive P(N) = w_DM E_post[P(N | k(s + b))] + w_U P(N | k(b + μ_U)) + w_K P(N | kb).

## 4. Results

### 4.1 Exposure (`exposure_scenarios.csv`)
1 Apr 2024 → 2 Sep 2026 = **884 calendar days**; at f = 0.593 that is 524 live days and **E = 6.76 t·yr, k = 2.38** (4.71 t). Range: 5.70 (f = 0.50) to 8.55 t·yr (0.75), k = 2.0–3.0; with 5.5 t, 7.89 t·yr (k = 2.78; up to 9.98 at f = 0.75). To 31 Dec 2026: 1004 d, 7.68 t·yr (k = 2.70); to 31 Dec 2027: 1369 d, 10.47 t·yr (k = 3.69). LZ's own 1000-live-day design goal, used by P007, corresponds to k ≈ 4.5.

### 4.2 Background (`P020_results.json → background`)
| b (per 2.84 t·yr) | μ to 2 Sep 2026 | P(≥1) | P(≥2) | μ to Dec 2026 | μ to Dec 2027 (P≥1) |
|---|---|---|---|---|---|
| 0.0106 (whole S1c > 500 panel) | 0.0252 | 2.49% | 3.1 × 10⁻⁴ | 0.0287 (2.8%) | 0.0391 (3.8%) |
| 10⁻³ (LZ-equivalent) | 2.4 × 10⁻³ | 0.24% | 2.8 × 10⁻⁶ | 2.7 × 10⁻³ | 3.7 × 10⁻³ |
| 2 × 10⁻⁴ (±2σ neighbourhood) | 4.8 × 10⁻⁴ | 0.048% | 1.1 × 10⁻⁷ | 5.4 × 10⁻⁴ | 7.4 × 10⁻⁴ |

Even the most inclusive background predicts one event *anywhere in the high-S1c panel* with only 2.5% probability; a second event in the NR band would be background at the 10⁻³ level or below.

### 4.3 L10 predictions (`predictive_L10_2026-09-02.csv`; Fig. 1)
k = 2.38 (b = 2 × 10⁻⁴ included):

| hypothesis | mean | P(0) | P(1) | P(2) | P(3) | P(≥4) |
|---|---|---|---|---|---|---|
| s = 1.0 plug-in | 2.38 | 0.092 | 0.220 | 0.262 | 0.208 | 0.217 |
| s = 0.3 (lower end) | 0.71 | 0.489 | 0.350 | 0.125 | 0.030 | 0.006 |
| s = 2.4 (upper end) | 5.71 | 0.003 | 0.019 | 0.054 | 0.103 | 0.821 |
| Poisson–Gamma, flat prior (Gamma(2,1)) | 4.76 | 0.088 | 0.123 | 0.130 | 0.122 | 0.537 |
| Poisson–Gamma, Jeffreys (Gamma(1.5,1)) | 3.57 | 0.161 | 0.170 | 0.150 | 0.123 | 0.397 |
| background, whole panel | 0.025 | 0.975 | 0.025 | 3 × 10⁻⁴ | — | — |

P(N = 0) is 9% for both the plug-in and the flat-prior predictive (the coincidence is exact only near k ≈ 2.4: e^{−k} vs (1 + k)^{−2}). To 31 Dec 2026 the plug-in mean is 2.70 (P(0) = 0.067; P(≥4) = 0.29); to 31 Dec 2027, 3.69 (P(0) = 0.025; P(≥4) = 0.50). Flat-prior predictive means 5.4 and 7.4.

### 4.4 Inelastic seasonal structure (`inelastic_seasonal.csv`; Fig. 2)
| δ (keV) | June/Dec | mod. fraction | peak doy | ε-survival at peak | frac > 270 keV | N_new per first-run event | N_new/k | f(May–Jul) | f(Nov–Jan) | f(May–Aug) | KL (nats/event) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | 2.40 | 0.418 | 152 | 0.917 | 0.044 | 2.50 | 1.050 | 0.410 | 0.127 | 0.519 | 0.043 |
| 350 | 26.5 | 0.930 | 152 | 0.775 | 0.195 | 2.72 | 1.144 | 0.592 | 0.025 | 0.686 | 0.327 |
| 366 | 345 | 0.995 | 152 | 0.513 | 0.474 | 2.85 | 1.197 | 0.708 | 0.003 | 0.770 | 0.585 |
| 380 | Dec rate = 0 (ratio numerically 10⁵) | 1.000 | 152 | 0.131 | 0.884 | 2.91 | 1.222 | 0.780 | 0.001 | 0.818 | 0.742 |
| flat | 1 | 0 | — | — | — | 2.38 | 1 | 0.312 | 0.208 | 0.416 | 0 |

Notes. (i) The δ = 300 and 350 keV June/Dec ratios (2.40, 26.5) and modulation fractions (0.42, 0.93) agree with P002 (2.4, 28) and P006 (41%, 90%). (ii) At δ = 380 keV the in-ROI rate is non-zero for 77% of the year and the December rate is zero to numerical precision; events are confined to roughly March–September with 82% in May–August. (iii) Because the 884-day window contains three summers (2024, 2025, 2026) but only 2.42 winters' worth of time, a June-peaked signal normalised to one event in the first (near-integer-year) run predicts 5–22% more events than the flat scaling k: 2.50, 2.72, 2.85, 2.91 for δ = 300, 350, 366, 380 keV. (iv) Efficiency: at δ = 366 keV only 51% of the June spectrum survives the 269.9 keV roll-off and 47% of the true-energy spectrum lies above 270 keV; at 380 keV only 13% survives. Normalising to one *observed* event absorbs this, but it means any future extension of the ROI (or a better-measured roll-off) would change the near-δ_max predictions by factors of 2–8, whereas δ = 300 keV predictions are insensitive (92% survival). (v) The Higgsino cross-check: P007's 4–5.2 events per 1000 live days scale to 2.1–2.7 events in our 524 live days; our δ = 366 keV seasonal normalisation gives 2.85 — consistent once the three-summer effect is included. (vi) In Fig. 2 the peak of p_mod/p_flat over the Apr 2024–Dec 2026 window is 1.4 (δ = 300), 2.4 (350), 3.2 (366) and 3.9 (380 keV) (`peak_over_flat_dec26`).

Comparison with P006's stored `rate_vs_doy` tables: at δ = 300 keV the shapes agree to 1.6% of the peak; at 350 keV P006's curve is flatter (June/Dec 18.8 vs our 26.5; P002 quoted 28) and at 380 keV much flatter (maximum deviation 0.59 of the peak; P006 has R(doy 81)/R_max = 0.50 versus our 0.13). We recomputed our δ = 380 keV curve on P006's own energy grid (1–320 keV, 120 points) and reproduced our result to 3 digits, so the difference is not the energy grid. The most likely explanation is that P006's tables were produced before `lz.wd_halo` received the explicit v_min grid (P002 found the default grid truncates the June halo at 795 km/s); a truncated high-velocity tail suppresses the June peak and flattens exactly the near-δ_max curves. This does not affect P006's headline numbers at δ ≤ 350 keV beyond the 20–30% level, but P006's N_3σ ≈ 7 at δ = 380 keV is probably an overestimate (our KL = 0.74 nats/event implies ~5 events for 3σ by P006's formula).

### 4.5 Seasonal visibility (`seasonal_visibility.csv`)
| δ (keV) | N | E[ln LR] | P(LR > 10 | mod) | P(LR > 10 | flat) | P(LR > 3 | mod) | P(LR > 3 | flat) | P(all in May–Aug | mod) | (| flat) |
|---|---|---|---|---|---|---|---|---|
| 300 | 2 | 0.09 | 0 | 0 | 0 | 0 | 0.27 | 0.17 |
| 300 | 3 | 0.13 | 0 | 0 | 0 | 0 | 0.14 | 0.07 |
| 300 | 5 | 0.21 | 0 | 0 | 0.08 | 0.02 | 0.04 | 0.01 |
| 350 | 2 | 0.66 | 0 | 0 | 0.40 | 0.11 | 0.47 | 0.17 |
| 350 | 3 | 0.99 | 0.001 | 10⁻⁴ | 0.58 | 0.11 | 0.33 | 0.07 |
| 350 | 5 | 1.63 | 0.39 | 0.02 | 0.70 | 0.08 | 0.15 | 0.01 |
| 366 | 2 | 1.18 | 0 | 0 | 0.67 | 0.13 | 0.59 | 0.17 |
| 366 | 3 | 1.75 | 0.39 | 0.03 | 0.79 | 0.10 | 0.46 | 0.07 |
| 366 | 5 | 2.93 | 0.74 | 0.02 | 0.89 | 0.05 | 0.27 | 0.01 |
| 380 | 2 | 1.48 | 0.16 | 0.015 | 0.75 | 0.12 | 0.67 | 0.17 |
| 380 | 3 | 2.23 | 0.58 | 0.03 | 0.85 | 0.08 | 0.55 | 0.07 |
| 380 | 5 | 3.70 | 0.83 | 0.02 | 0.93 | 0.03 | 0.37 | 0.01 |

Reading: with two events the season is never "strong" evidence (LR > 10 is unreachable for δ ≤ 366 keV because the per-event ratio is at most 3.2); with three events the pattern is strong with 40–60% probability for δ ≥ 366 keV; with five events, 74–83% for δ ≥ 366 keV and 39% at 350 keV, at a 2–3% false-positive rate. δ = 300 keV modulation is invisible with ≤ 5 events (consistent with P006's N_3σ = 107). The model-free observable "all events in May–August" has a flat-hypothesis probability of 0.17/0.07/0.012 for N = 2/3/5.

### 4.6 Decision table (`decision_table.csv`; Fig. 3), exposure to 2 Sep 2026 (k = 2.38)
| N new | P(DM) b = 2e-4 | P(DM) b = 1e-3 | P(DM) b = 0.0106 | P(U) b = 2e-4 | P(DM), π_DM = 0.05 | BF₂(DM:K) b = 2e-4 / 1e-3 / 0.0106 | Z_comb (2e-4 / 1e-3 / 0.0106) | Z_new-only (2e-4 / 1e-3 / 0.0106) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.011 | 0.010 | 0.005 | 0.965 | 0.055 | 0.088 / 0.088 / 0.090 | 3.20 / 2.71 / 1.81 | — |
| 1 | 0.062 | 0.061 | 0.049 | 0.938 | 0.249 | 259 / 52 / 5.0 | 5.04 / 4.39 / 3.23 | 3.30 / 2.82 / 1.96 |
| 2 | 0.358 | 0.354 | 0.302 | 0.642 | 0.736 | 1.2 × 10⁶ / 4.6 × 10⁴ / 420 | 6.46 / 5.69 / 4.33 | 5.18 / 4.54 / 3.42 |
| 3 | 0.862 | 0.859 | 0.816 | 0.138 | 0.969 | 6.8 × 10⁹ / 5.5 × 10⁷ / 4.7 × 10⁴ | 7.67 / 6.79 / 5.27 | 6.62 / 5.87 / 4.55 |
| 4 | 0.989 | 0.988 | 0.983 | 0.011 | 0.998 | 5 × 10¹³ / 8 × 10¹⁰ / 6.5 × 10⁶ | 8.74 / 7.78 / 6.12 | 7.85 / 6.99 / 5.52 |

Checks: the framework reproduces P001's "now" posterior (0.094 vs P001's 0.090; P(U) = 0.890, P(K) = 0.017) and P001's equal-exposure updates 0.028/0.217/0.797 exactly.

Smallest N for 5σ local (`five_sigma`): combined sample N = 1 (b = 2 × 10⁻⁴), 2 (10⁻³), 3 (0.0106); new exposure alone N = 2, 3, 4. For the Dec 2027 exposure the same thresholds hold except that b = 2 × 10⁻⁴ combined needs N = 2.

The structure of the table is set by P001's unknown-background hypothesis U with μ_U = 0.105 per 2.84 t·yr: because U predicts 0.25 events in the new exposure, one new event is "explained" by U almost as well as by DM (P(DM) rises only to 0.06), two events split the odds (0.36), and three or more events decide (≥ 0.86). Against the *modelled* background alone the second exposure is decisive already at N = 1 (BF₂ = 259 at b = 2 × 10⁻⁴; 52 at the LZ-equivalent 10⁻³). By 31 Dec 2027 (k = 3.69) the same counts are less favourable to DM (P(DM) = 0.007/0.028/0.148/0.584/0.934 for N = 0–4) because the DM hypothesis then predicts more events.

### 4.7 N = 0 (`N0_upper_limits.csv`)
| end | k | UL90 (new only) | UL90 combined, Bayes | UL90 combined, PLR | ŝ combined | fraction of [0.3, 2.4] excluded | P(N = 0 | s = 1) |
|---|---|---|---|---|---|---|---|
| 2 Sep 2026 | 2.38 | 0.97 | 1.15 | 1.08 | 0.30 | 0.68 | 0.092 |
| 31 Dec 2026 | 2.70 | 0.85 | 1.05 | 0.98 | 0.27 | 0.74 | 0.067 |
| 31 Dec 2027 | 3.69 | 0.62 | 0.83 | 0.78 | 0.21 | 0.85 | 0.025 |

Zero events by 2 Sep 2026 would exclude the published best-fit rate (s = 1.0 per 2.84 t·yr) at essentially exactly 90% CL from the new data alone (UL = 0.97), excluding 68% of the Table I 68% interval; the combined best fit would fall to 0.30 with UL ≈ 1.1. Zero events by the end of 2027 would push the UL to 0.62 (new only) and exclude 85% of the interval. In the P001 framework N = 0 gives P(DM) = 0.011 (0.055 for π_DM = 0.05) — the interpretation is effectively dead, but the event itself would then be attributed to U (P(U) = 0.97), not to the modelled background (P(K) = 0.023).

### 4.8 Corpus-prior mixture predictive (`mixture_predictive`)
Weights after the first event: DM/U/K = 0.094/0.890/0.017. Check at k = 1: P(0/1/≥2) = 0.841/0.108/0.051 (P001: 0.84/0.11/0.05). For the new exposure (k = 2.38): **P(0) = 0.717, P(1) = 0.185, P(2) = 0.034, P(3) = 0.013, P(≥4) = 0.050** (P(≥2) = 0.098). To Dec 2026: 0.692/0.201/0.038/0.013/0.055; to Dec 2027: 0.624/0.241/0.054/0.014/0.067. The P(1) mass is dominated by U (0.89 × 0.20); the P(≥4) mass entirely by DM.

### 4.9 The forecast
For the exposure LZ had in hand when it posted the paper (884 calendar days, ≈ 524 live days, 6.8 t·yr, k = 2.4):
- **L10 best fit:** N = 2.4 events (68% likelihood range of the rate: 0.7–5.7); P(N = 0) = 9%; P(N ≥ 2) = 69%; with the rate uncertainty marginalised, mean 4.8 and P(N ≥ 4) = 54%.
- **Background:** P(N ≥ 1) = 2.5% anywhere in the S1c > 500 phd panel, 0.05% within ±2σ of the NR median.
- **Inelastic, δ ≥ 350 keV:** 2.7–2.9 events, 59–78% of them in May–July and ≤ 2.5% in Nov–Jan; at δ = 380 keV none between October and March.
- **Corpus prior (P001):** P(0) = 72%, P(1) = 19%, P(≥2) = 10%.
- **Decision:** N = 0 kills the best-fit rate at 90% CL and leaves P(DM) ≈ 0.01; N = 1 is 4.4σ local combined (LZ-equivalent b) but P(DM) only 0.06; N = 2 is 5.7σ combined / 4.5σ new-only and P(DM) = 0.36; N ≥ 3 is > 5σ from the new data alone and P(DM) ≥ 0.86. Any second event in November–February excludes δ ≥ 366 keV; three or more events all in May–August would be a 0.07-probability coincidence under a flat background.

## 5. Validation and robustness
1. Table I interval reproduced by the n = 1 Poisson likelihood ([0.30, 2.36] vs [0.3, 2.4]).
2. P001's posteriors reproduced (now: 0.094 vs 0.090 — the 4% difference is quadrature/normalisation of the flat prior; equal-exposure updates identical to three digits; predictive 0.841/0.108/0.051).
3. P002/P006 seasonal numbers reproduced at δ = 300 and 350 keV (June/Dec 2.40 and 26.5; modulation fractions 0.42 and 0.93).
4. Energy-grid independence of the δ = 380 keV seasonal shape verified (91-point 60–330 keV grid vs P006's 120-point 1–320 keV grid: identical to 3 digits).
5. Live fraction 0.50–0.75 changes k by ×0.84–1.26 and all Poisson means proportionally; the 5.5 t scenario adds ×1.17.
6. The Jeffreys-prior predictive (mean 3.57) sits between the plug-in (2.38) and flat-prior (4.76) predictives; P(N = 0) ranges 0.09–0.16 across the three.
7. Toy sample size 40 000: binomial uncertainty on the visibility probabilities ≤ 0.003.

## 6. Failed or abandoned approaches
- A first version of Fig. 2 normalised the time PDF per year instead of per window, misplacing the "flat" reference; corrected (the toy LR calculations were never affected).
- The δ = 380 keV June/December ratio is numerically ~10⁵ and is reported as "December rate = 0" rather than as a number.
- We did not attempt a 2D (S1c, log S2c) likelihood or an L10-specific seasonal PDF (P006 found LR ≈ 0.98 for dipole-like elastic spectra, i.e. no usable modulation); the elastic best fit is treated as time-flat.

## 7. Discussion
The central point is that the post-April-2024 data are a pre-registered experiment. The paper's 616-model scan and its ~14 effective trials (P001) apply to the discovery of the first event; the same models now make a fixed prediction for a fixed region, so the new count carries its full local significance. That is why two events in the new data — a perfectly ordinary outcome under the best fit (P = 0.26) — would already constitute 5.7σ (combined) or 4.5σ (new data alone) against the LZ-equivalent background, and three events would exceed 5σ from the new data alone under any of the three background choices.

The Bayesian table is more sober, and for a reason worth stating: within the P001 framework the competitor to DM is not the modelled background but an unknown background U whose rate (0.1 per 2.84 t·yr) was chosen to make the first event plausible. Such a U predicts 0.25 events in the new exposure, so a single new event barely discriminates; only the *multiplicity* discriminates, and it does so quickly (P(DM) = 0.36, 0.86, 0.99 for N = 2, 3, 4). The season adds an independent handle for inelastic models: a second event in winter would exclude δ ≥ 366 keV outright, and three summer events would be a 7% coincidence under any time-flat background.

The single most informative outcome is N = 0. It costs the DM interpretation its best-fit rate at 90% CL, and — more tellingly — under the corpus prior it is the *expected* outcome (72%), because the corpus assigns only 9% to DM after the first event. A non-zero count is therefore the surprise, and the size of the surprise is set by the Poisson–Gamma tail: P(N ≥ 4) = 5% under the corpus prior versus 54% under the best fit with its uncertainty.

Caveats: the live fraction of the new data is unknown (0.50–0.75 is a ±25% band on every mean); the fiducial mass may change; the efficiency roll-off near 270 keV (which halves the δ = 366 keV prediction's sensitivity) is read from Fig. S2; the U hypothesis is judgemental; livetime is assumed uniform in calendar time (a long calibration campaign in a summer would reduce the inelastic predictions by up to the fraction of summer livetime lost); and a genuine time-varying background (e.g. calibration-linked, cf. AmBe 8 days before the event) is not modelled.

## 8. Figures
- `figures/predictive_distributions.png` — Fig. 1: P(N) for N = 0…≥4 in the 1 Apr 2024–2 Sep 2026 exposure under the whole-panel background, the L10 plug-in, the Poisson–Gamma predictive and the P001 mixture.
- `figures/seasonal_pdf.png` — Fig. 2: arrival-time density relative to flat for O1 inelastic, m = 1000 GeV, δ = 300/350/366/380 keV over Apr 2024–Dec 2026 (May–July shaded; 2 Sep 2026 marked).
- `figures/posterior_vs_N.png` — Fig. 3: P(DM | 1 + N events) vs N for the three background choices (P001 framework).

## 9. References
LZ Collaboration, arXiv:2609.02823 (2026). Corpus: dossier 00; P001 (Bayesian framework); P002 (kinematics, June/Dec ratios); P006 (seasonal PDFs, efficiency parametrisation); P007 (Higgsino forecast); P004/P010 (background hypotheses at the 10⁻³–10⁻² level). G. Cowan, K. Cranmer, E. Gross, O. Vitells, Eur. Phys. J. C 71, 1554 (2011). G. J. Feldman, R. D. Cousins, Phys. Rev. D 57, 3873 (1998). K. Freese, J. Frieman, A. Gould, Phys. Rev. D 37, 3388 (1988). D. Tucker-Smith, N. Weiner, Phys. Rev. D 64, 043502 (2001). D. Baxter et al., Eur. Phys. J. C 81, 907 (2021). R. E. Kass, A. E. Raftery, J. Am. Stat. Assoc. 90, 773 (1995). S. Jeong, S. Kang, S. Scopel, G. Tomar, Comput. Phys. Commun. 276, 108342 (2022) (WimPyDD).

## 10. Tools and provenance (mirrors `output/provenance/P020.json`)
- Agent tools: Read (PAPER_GUIDE.md; dossier; results_ledger.csv; P001/P002/P004/P006/P007/P010.md; fulltext.tex lines 108–137, 208–267, 330–335; lzcommon.py lines 1–125 and 285–373; P006_event_date.py lines 84–133; P001 details §2.5 and §3.6; three PNG figures); Bash (grep of tex/library; listing of prior work; environment versions; two full runs of the script plus one diagnostic WimPyDD comparison and two JSON/CSV print-outs); Write (script, details.md, P020.json, P020.md); Edit (two figure fixes); Skill (dataviz).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (stats.poisson/nbinom/norm, integrate.quad, special.gammainc, optimize.brentq, interpolate.CubicSpline); matplotlib 3.11.2 (Agg); WimPyDD 2.0.4 (streamed_halo_function via lz.wd_halo, eft_hamiltonian, diff_rate via lz.wd_rate); common/lzcommon.py (LZ dict, wd_halo, wd_hamiltonian, wd_rate); Python stdlib datetime/csv/json.
- Script: `output/code/P020_future_lz.py`, command `.venv/bin/python output/code/P020_future_lz.py` (run twice; second run after figure fixes; numbers identical).
- Local inputs: as in §2. Recalled knowledge: 5 items (all certain). Datasets: none. Data requests: none. WimPyDD-generated files: none (diff_rate used; no response-function files written).
