# P099 — Decisive tests and a falsifiable forecast for LZ's next release: research record

Simulated arXiv date 2026-09-16 · hep-ex (cross-list hep-ph, physics.data-an) · PROJ · cross-collaboration working group (position paper)

## 1. Motivation and framework

Ninety-odd corpus papers have dissected LZ's single 248 keV nuclear-recoil-like event (S1c = 540.1 phd, 2.84 t·yr, local 3.4σ, global 2.6σ). The projection papers (P020, P034, P050, P069, P079, P081, P088) each answer one question; nobody has yet written down, in one place and in advance of LZ's next release, (a) which tests decide, (b) what the community should *expect* to see under each hypothesis, and (c) what observation would kill each surviving interpretation. This paper is that pre-registration. It deliberately recomputes **no physics**: every rate, acceptance, background, halo factor and spectral ratio is taken from the paper that computed it and cited by P-number. We recompute only the counting and decision statistics needed to state the forecast crisply, and validate each against the number its source quotes.

Hypothesis classes (following P061/P081): **DM** (any of the corpus's compatible spectra: inelastic O₁ with δ = 335–390 keV per P082, elastic q⁴-spin L10/O6/O9×q² per P003/P016/P057, exothermic per P072); **B** (LZ's modelled background, or a one-off of the corpus's unmodelled classes — accidental, activation, artefact); **U** (an LZ-specific unknown, either a one-off transient or a steady rate). Prior weights: P061 community-prior medians P(DM) = 0.015, P(B) = 0.148, P(U) = 0.789 with f_trans = 0.5 of U one-off (as propagated by P081, `P081_results.json` → P061_reproduction).

## 2. Equations

Exposure ratio k = E/2.84 t·yr. For a Poisson mean μ: P(N) = μ^N e^{−μ}/N!.

**Gamma–Poisson predictive.** If the band rate per 2.84 t·yr is s ~ Gamma(a, θ), the count in k units is negative-binomial: P(N) = Γ(a+N)/(Γ(a) N!) · p^a (1−p)^N with p = 1/(1+kθ). P(0) = (1+kθ)^{−a}.

**Steady unknown** (P081): rate per 2.84 t·yr ~ Exponential(1), i.e. Gamma(1,1): P(N | k) = k^N/(1+k)^{N+1}; P(0) = 1/(1+k).

**Mixture posterior.** P(DM | N) = π_DM L_DM / (π_DM L_DM + π_B L_B + π_U L_U), with L_B = Poisson(N; b k), L_U = f_trans·𝟙[N = 0] + (1 − f_trans)·k^N/(1+k)^{N+1}.

**Null branch** (P088 recipe). After LZ's one event with a flat prior the best-fit-scaled rate is Gamma(2, 1); zero new events in k best-fit-expected events give L_DM = (1+k)^{−2} while every alternative predicts zero with probability ≈ 1: P(DM | 0) = π(1+k)^{−2}/[π(1+k)^{−2} + 1 − π]. Solving P(DM | 0) = 10⁻³ for π = 0.015 gives k = 2.90 expected events; a date follows from the on-disk expectation and the accrual rate.

**Exact-Poisson discovery.** n_5σ(b) is the smallest n with P(≥n | b) ≤ Φ(−5) = 2.87×10⁻⁷; P(5σ | μ) = P(≥n_5σ | μ). Counting significance for N observed on b: Z = Φ⁻¹(1 − P(≥N | b)).

## 3. Inputs (all with source)

| quantity | value | source |
|---|---|---|
| LZ exposure, FV | 2.84 t·yr, 4.71 t | LZ paper via `lz.LZ` |
| best-fit 200–270 keV rate per t·yr (600 phd), L10/δ300/350/366/380 | 0.121/0.063/0.178/0.274/0.278 | P069 normalisation (P088 §2.2; agrees with P050 ≤ 0.3 %); LZ's whole-ROI fit normalised to one event in 2.84 t·yr |
| best-fit 200–400 keV rate per t·yr (1000 phd) | 0.220/0.112/0.516/1.63/10.5 | P069/P050 |
| band-anchored L10 (one *band* event per 2.84 t·yr) | 0.352 per t·yr; 2.38 events in 524 live days | P020, P079 (P079 caveat: 2.7× LZ's whole-ROI fit) |
| corpus band-rate posterior per 2.84 t·yr | mean 0.385, median 0.19, 68 % 0.036–0.69 | P081 |
| NR-band background 200–270 keV per 2.84 t·yr | 5.7×10⁻⁴ (bracket 2×10⁻⁴–0.0106) | P016 (adopted by P069/P081/P088) |
| corpus-revised neighbourhood background | 9.4×10⁻⁴ (artefact class 64 %) | P093 |
| 270–420 keV NR-band MSSI at 1000 phd per 2.84 t·yr | 0.0028 | P038 |
| 55–200 keV background per 2.84 t·yr | 0.12 (0.10 estimate + 0.02) | P081, P016 |
| P079 untouched-exposure inputs | b = 8.9×10⁻⁴ (600 phd) → 8.0×10⁻³ (1000 phd) per 524 d; signal ×1.89/3.0/6.3 (L10/δ350/366) | P079 |
| 600–1000 phd sideband expectation in SR3 (2.84 t·yr) | 0.28/0.155/1.06/3.81/30.6 (L10/δ300/350/366/380) | P088 `P088_LZ_sideband.csv` (from P038's extra-events-per-ROI-event) |
| exposure ledger (t·yr) | LZ untouched 6.76; XENONnT 3.1 analysed (ROI ≲ 60 keV) + 3.0 untouched; PandaX-4T 1.54 + 2.5; unexamined above 200 keV 17.9 | P069, P088 (XENONnT/PandaX numbers recalled by P069, ±50 %) |
| accrual | LZ 2.79, XENONnT 2.4, PandaX-4T 1.62 t·yr/yr | P069/P088 |
| hypothesis weights | 0.015 / 0.148 / 0.789; f_trans 0.5 | P061, P081 |
| exothermic companions per band event, 125–200 keV | 5.7/5.4/4.5/2.1 at abs δ = 278/300/350/500 keV | P072 |
| L10 vs inelastic-366 separation | 2–7 events (600 phd), 2–5 (1000 phd) | P057 |
| modulation 3σ | 5.3/6.5/11.5/29/97 events = 15/19/33/84/274 t·yr (δ = 380/366/350/330/300) | P034 |
| CaWO₄ 10 kg·yr | 2.8/0.37/0.03 events, P(3σ) = 0.62/0.03/0.004 (δ366/350/300) | P046, P088 |
| ¹³⁶Xe pair | 9 events, 12.8 t·yr per detector | P047 |
| directional | < 1 event/yr per 1000 m³ heavy-gas TPC | P067 |
| topology handles | log₁₀ odds NR:ART +2.5, NR:ER +3.6; unpublished PSD ±0.45 dex, audit ±0.2 | P093 (provisional pending DR-003) |
| surviving inelastic region | m ≥ 400 GeV, δ = 335–390 keV, σ_n = 8×10⁻⁴²–3×10⁻³⁶ cm²; Higgsino δ = 365–380 keV | P082 |
| Higgsino excluded by solar capture at every δ | Γ_A/Γ_lim ≥ 15–770 | P076 |
| exothermic viability | B ≤ 2 vs background, 0.04–0.67 vs endothermic | P072 |
| data requests | DR-001 (L10 normalisation; P012/P045), DR-002 (event list/PDFs; P052), DR-003 (waveforms; P093) | `output/data_requests/index.csv` |

No new recalled knowledge enters this paper beyond standard statistics (Gamma–Poisson conjugacy, Asimov/exact Poisson significance) — see §10.

## 4. Results

### 4.1 Gamma match to P081's posterior — `P099_results.json` → posterior_gamma

Matching mean 0.385 and median 0.19 gives a = 0.551, θ = 0.699 (per 2.84 t·yr). Its 68 % interval is [0.021, 0.755] versus P081's [0.036, 0.69]: the single Gamma is slightly wider at both ends than P081's 298-spectrum mixture. Consequences: count probabilities at k ≤ 3.5 reproduce P081 to ≤ 0.013 (§5) but the exposure for P(≥1 | DM) = 0.9 is 261 t·yr against P081's 123 (the Gamma's low-rate tail is heavier). We quote P081's 123 t·yr and our 261 as a bracket; for P(≥1) = 0.5 we get 10.2 vs P081's 9.8 t·yr.

### 4.2 Forecast counts — `tables/P099_forecast_counts.csv`

200–270 keV band (600 phd edge), P(0) / P(1) / P(≥2):

| E (t·yr) | DM corpus posterior | modelled B: P(≥1) | corpus-revised B | steady unknown P(0) | plug-in L10 (ROI) P(≥1) | plug-in δ366 P(≥1) | band-anchored L10 P(≥1) |
|---|---|---|---|---|---|---|---|
| 1.0 | 0.865 / 0.101 / 0.034 | 2.0×10⁻⁴ | 3.3×10⁻⁴ | 0.740 | 0.114 | 0.240 | 0.297 |
| 2.8 | 0.749 / 0.168 / 0.082 | 5.6×10⁻⁴ | 9.3×10⁻⁴ | 0.504 | 0.288 | 0.536 | 0.627 |
| 5.6 | 0.628 / 0.192 / 0.180 | 1.1×10⁻³ | 1.9×10⁻³ | 0.336 | 0.492 | 0.784 | 0.861 |
| 6.76 | 0.583 / 0.195 / 0.222 | 1.4×10⁻³ | 2.2×10⁻³ | 0.296 | 0.559 | 0.843 | 0.907 |
| 10.0 | 0.505 / 0.198 / 0.298 | 2.0×10⁻³ | 3.3×10⁻³ | 0.221 | 0.702 | 0.935 | 0.970 |

(the 1.0 t·yr DM row is read from the CSV: P0 = 0.865; P081 gives 0.849 because its 1 t·yr window is summer-weighted, which we do not model.)

Other windows at 2.8 t·yr: 55–200 keV background P(≥1) = 0.112; 270–420 keV (1000 phd) background P(≥1) = 0.0028; DM in 55–270 keV P(≥1) = 0.50 and in 270–420 keV 0.26 (both quoted from P081, whose extension numbers are tail-dominated by δ ≥ 366 keV models and given as medians). 200–400 keV plug-in at 6.76 t·yr: L10 1.49, δ300 0.76, δ350 3.49, δ366 11.0, δ380 71 expected events (the last two conflict with LZ's empty sideband, §4.6).

**Anchoring spread.** The "best fit" is ambiguous by ×2.9: P020/P079 anchor L10 to one *band* event per 2.84 t·yr (0.352 per t·yr), P069/P088 to one event in LZ's whole ROI (0.121 per t·yr in the band), and P081's posterior mean (0.136 per t·yr) sits with the latter because the empty 125–200 keV bin pulls companion-bearing spectra down. All forecasts below give the posterior number first and the plug-in bracket second.

### 4.3 Pre-registered predictions — `tables/P099_preregistered_predictions.csv`

Assumed: LZ's next release adds ≈ 2.8 t·yr of new WS-like exposure with the present 600 phd edge (a second block equal to WS2024); F6 covers the full untouched 6.76 t·yr (884 calendar days since 1 April 2024 at live fraction 0.593, P020); F7 covers a XENONnT + PandaX-4T reanalysis of 10.1 t·yr above 200 keV (P069 ledger). Probabilities are (DM corpus posterior) / (modelled background).

1. **F1** 200–270 keV, N = 0: 0.749 / 0.9994 (P081 0.75 / 0.9994). Mixture over all hypotheses (P081): 0.79.
2. **F2** N ≥ 1: 0.251 / 5.6×10⁻⁴; 9.3×10⁻⁴ for the corpus-revised background (P093); plug-in bracket 0.29 (L10 ROI) – 0.63 (band-anchored).
3. **F3** N ≥ 2: 0.082 / 1.6×10⁻⁷.
4. **F4** 55–270 keV, N ≥ 1: 0.50 (P081) / 0.112 (background dominated by the 0.10 estimate in 55–125 keV).
5. **F5** 270–420 keV only (if the edge is raised to 1000 phd), N ≥ 1: 0.26 (P081 median) / 0.0028.
6. **F6** full untouched 6.76 t·yr, 200–270 keV, N ≥ 1: 0.417 / 1.4×10⁻³; plug-in 0.56 (L10 ROI), 0.84 (δ366), 0.91 (band-anchored L10, P020).
7. **F7** XENONnT + PandaX-4T 10.1 t·yr reanalysed, N ≥ 1: 0.498 / 2.0×10⁻³; plug-in 0.71 (L10) – 0.94 (δ366) (P069: 0.70–0.995 for its slightly larger 19.1 t·yr set).

### 4.4 Decision rule — `tables/P099_decision_rule.csv`

At 2.8 t·yr with the P061 weights: P(DM | N = 0) = 0.0149, P(DM | N = 1) = 0.0250, P(DM | N = 2) = 0.0161 (P081: 0.0143 / 0.0290 / 0.0176). Bayes factors for N = 1: DM:B = 300 (P081 300), DM:U_steady = 0.67 (P081 0.67), DM:U_mixed = 1.35. Hence a single new band event, by itself, is not confirmation: it excludes the modelled and one-off backgrounds but is exactly what a steady LZ-specific unknown predicts. Energy placement (P081 `P081_energy_placement.csv`, cited): two events both in 270–420 keV → P(DM) = 0.987 (BF 3.8×10³ vs B, 7.5×10³ vs U; our background-side check P(2 | 0.0028) = 3.9×10⁻⁶ matches P081's 3.4×10⁻⁶ joint); one band + one extension event → 0.57; band + 55–200 keV → 0.061; both 55–200 keV → 0.153; both in band → 0.011; nothing anywhere in 55–420 keV → 0.0089. **Rule:** P(DM) > 0.5 requires two events with at least one in 270–420 keV (or, marginally, one band + one extension); a second detector's event (F7) is the other route (P069: N_new = 2 → P(DM) = 0.38 for L10, 3 → 0.87).

### 4.5 Null branch — `tables/P099_null_branch.csv`

k for P(DM | 0) < 10⁻³ is 2.90 expected best-fit events. Present generation (17.9 t·yr on disk, accrual 6.81 t·yr/yr, 600 phd): on-disk expectation 2.17/1.13/3.19/4.90 (L10/δ300/350/366) → P(DM) < 10⁻³ by 2027.60 (Aug 2027) for L10, 2030.84 (Nov 2030) for δ300, already now for δ ≥ 350 keV (P088: 2027.61 / 2030.86 / 2026.38 / 2025.16 — P088 dates the on-disk exposure by when it was accrued, we floor at "now"). LZ alone at 600 phd: 2032.9 (L10), 2040.8 (δ300), 2030.1 (δ350), 2028.1 (δ366) — i.e. LZ alone cannot retire the elastic reading before 2032 at its present edge (P088: "inf" before 2035 for L10 and δ300 at LZ-600). LZ alone at 1000 phd (200–400 keV rate on the untouched data plus the already-empty SR3 sideband): 2028.55 (L10), 2033.1 (δ300), now for δ ≥ 350. P088 quotes 2027.69 for LZ-1000/L10; the 0.9 yr difference comes from P088 counting additional on-disk exposure (SR1 and dating by accrual) and its Gamma-marginalised toys; we flag it as the calendar uncertainty of this branch.

### 4.6 Sideband already on disk — `tables/P099_sideband.csv`

P(0 | δ) for the empty 600–1000 phd region: SR3 alone 0.755/0.856/0.346/0.022/5×10⁻¹⁴ (L10/δ300/350/366/380; P088 0.02 for δ366 reproduced); untouched 6.76 t·yr 0.51/0.69/0.080/1.2×10⁻⁴/2×10⁻³²; SR3 + untouched 0.39/0.59/0.028/2.6×10⁻⁶/≈0. Therefore δ ≥ 366 keV (which contains P082's joint best fit at 380 keV and its Higgsino window 365–380 keV) is already disfavoured at P = 0.02 and will be dead (P < 10⁻⁴) the moment LZ looks at 600–1000 phd in the untouched data — no new exposure needed. δ = 350 keV survives the sideband (0.08 untouched, 0.028 combined) and is the inelastic point the community should treat as live; δ = 300 keV is untouched by the sideband but pays with a 248 keV percentile above 99 % (P002/P068).

### 4.7 Reanalysis on disk — `tables/P099_reanalysis_on_disk.csv`

Unexamined 17.9 t·yr: 2.17/1.13/3.19/4.90/4.98 expected (P088 2.2/1.1/3.2/4.9), P(0) = 0.115/0.324/0.041/0.007/0.007 (P088 0.12/0.32/0.04/0.007); posterior-mean expectation 2.43 with Gamma–Poisson P(0) = 0.395 (the posterior's low tail keeps P(0) high even at 17.9 t·yr). XENONnT + PandaX-4T 10.1 t·yr: 1.23–2.82 events, P(≥1) = 0.71–0.94 at best fit, 0.50 posterior. Two new events on b = 3.6×10⁻³ give Z = 4.36σ (P088 4.3), 5.58σ with LZ's event (P088 5.5; P069 5.5). The verdict "population or one-off" is therefore on disk at the best fit; P061's value-of-information ranking (XENONnT + PandaX first) is confirmed.

### 4.8 LZ's untouched exposure at 600 versus 1000 phd — `tables/P099_LZ_untouched_5sigma.csv`

With P079's inputs: n_5σ = 3 for both b = 8.9×10⁻⁴ and 8.0×10⁻³; P(5σ) = P(≥3 | 2.38) = 0.425 at 600 phd and P(≥3 | 4.50) = 0.826 at 1000 phd for band-anchored L10 (P079 0.43/0.83), 0.973 for δ350 (μ = 7.14) and 1.000 for δ366 (μ = 15.0; P079 1.00). With the ROI-anchored L10 (0.82 expected at 600 phd) P(≥3) would be only 0.05, and 0.19 at 1000 phd — the anchoring spread again.

### 4.9 Exothermic companions — `tables/P099_exothermic_companions.csv`

P(no 125–200 keV companion | one band event) = e^{−5.7}/e^{−5.4}/e^{−4.5}/e^{−2.1} = 0.003/0.005/0.011/0.12 at abs δ = 278/300/350/500 keV; for two band events 1×10⁻⁵–0.015. The exothermic reading is already effectively dead (P072 B ≤ 2) and any confirmed band event without a companion buries it.

### 4.10 Kill conditions — `tables/P099_kill_conditions.csv`

| interpretation | kill condition | expected date |
|---|---|---|
| inelastic O₁ δ = 335–390 keV (P082) | empty 600–1000 phd in the untouched 6.76 t·yr: P(0 | δ366) = 1.2×10⁻⁴, P(0 | δ350) = 0.08; any event < 125 keV (LR ≤ 0.04, P057); Higgsino already excluded by solar capture (P076) | 2027 (next LZ release) |
| elastic q⁴-spin L10/O6/O9×q² | zero band events in 17.9 t·yr (P(0) = 0.12 best fit; UL90 → rate < 1.0× best fit, P069) — weak; second event > 300 keV at 1000 phd (LR 6–7 for δ366/380, P057); 0.105 lower edge needs 49–62 t·yr (P050) | 2027 (weak) / 2033 XLZD (strong) |
| exothermic (P072) | P(no companion) = 0.005 already; any band event without a 125–200 keV companion | now / next event |
| background one-off | any new NR-band event above 200 keV (BF ≥ 300, P081); expected with P = 0.25 per 2.8 t·yr | 2027 |
| steady LZ-specific unknown | an event in XENONnT/PandaX-4T (P(≥1) = 0.71–0.94 best fit); LZ events in 270–420 or 55–200 keV (two extension events → 0.99) | mid-2027 / 2027 |
| artefact / charge-poor ER | DR-003 waveform quantities (PSD ±0.45 dex, audit ±0.2, P093); P041's predicted 65–101 charge-deficient ERs in the empty strip and ⁸³ᵐKr map hole | at LZ's discretion |

### 4.11 What cannot be settled before 2030

- **Mass** above 400 GeV: 642 events by shape or 243 by modulation (P062); no xenon experiment gets there.
- **δ beyond the halo floor**: ±16 keV from v_esc (P050, P046); tungsten's δ-meter needs 100 kg·yr CaWO₄ and is limited by the same floor.
- **Modulation** for δ ≤ 350 keV: 33–274 t·yr (P034); LZ reaches 3σ only for δ ≥ 366 keV around 2029–2030, which the sideband has already disfavoured.
- **Operator identity** among the compatible elastic operators: 15–60 events (P057) → 40–400 t·yr at 0.14–0.35 events/t·yr.
- **Any rate ≤ 0.3× best fit**: 5σ only with XLZD, 2033 (P088; P050).
- **DM versus a free-rate steady unknown** in LZ alone: counting caps P(DM) at 0.09 (P001/P069); only a second detector or energy placement breaks it.
- **The particle** (Higgsino vs alternatives) at colliders: HL-LHC blind, FCC-hh marginal, muon collider decisive (P014/P048).

### 4.12 Test × interpretation table — `tables/P099_test_x_interpretation.{csv,md}`

Eighteen rows (test, interpretation, expected signature, P(outcome), date, source), reproduced in the markdown file; the paper's Fig. 1 summarises the counting side.

## 5. Validation and robustness

| quantity | this work | source value |
|---|---|---|
| DM P0/P1/P≥2, 2.8 t·yr band | 0.749/0.168/0.082 | P081 0.752/0.169/0.079 |
| DM P0/P1/P≥2, 10 t·yr | 0.505/0.198/0.298 | P081 0.492/0.215/0.293 |
| background P(1), 2.8 t·yr | 5.6×10⁻⁴ | P081 5.6×10⁻⁴ |
| E for P(≥1 | DM) = 0.5 / 0.9 | 10.2 / 261 t·yr | P081 9.8 / 123 (Gamma tail heavier; flagged) |
| P(DM | N = 0/1/2), 2.8 t·yr | 0.0149/0.0250/0.0161 | P081 0.0143/0.0290/0.0176 |
| BF DM:B, DM:U_steady (N = 1) | 300 / 0.67 | P081 300 / 0.67 |
| unexamined expectation, P(0) | 2.17/1.13/3.19/4.90; 0.115/0.324/0.041/0.007 | P088 2.2/1.1/3.2/4.9; 0.12/0.32/0.04/0.007 |
| Z for two new events (with LZ's) | 4.36σ (5.58σ) | P088 4.3 (5.5); P069 5.5 |
| null-branch dates, present 600 phd, L10/δ300 | 2027.60 / 2030.84 | P088 2027.61 / 2030.86 |
| null-branch LZ-1000 L10 | 2028.55 | P088 2027.69 (flagged, ±1 yr) |
| P(5σ) band-anchored L10, 600/1000 phd; δ366 | 0.425 / 0.826; 1.000 | P079 0.43 / 0.83; 1.00 |
| sideband P(0 | δ366), SR3 | 0.022 | P088 0.02 |

Robustness: (i) with the corpus-revised background (P093) all background P(≥1) rise ×1.65, no conclusion changes; (ii) with f_trans = 1 (unknown purely one-off) P(DM | N = 1) rises to 0.82 (P081) — the whole decision rule hinges on whether LZ-specific unknowns can be steady, which is why the second-detector test ranks first; (iii) the anchoring spread ×2.9 brackets every plug-in probability (§4.2, §4.8); (iv) the P061 prior median 0.015 carries a 68 % range 0.0016–0.137; at 0.137 the same N = 1 outcome gives P(DM) = 0.20 (P081 decision table), still below 0.5; (v) calendar dates inherit P088's recalled exposures (XENONnT/PandaX untouched ±50 %).

## 6. Figures

`figures/P099_fig1_forecast.png` — Left: P(≥1 event in 200–270 keV) versus additional exposure for the corpus DM posterior (Gamma-matched to P081), the two plug-in anchorings, the steady-unknown predictive, and the modelled and corpus-revised backgrounds; vertical lines mark 2.8 t·yr, LZ's untouched 6.76 t·yr and the 17.9 t·yr unexamined by the xenon generation. Right: P(DM | outcome) after 2.8 t·yr for the eight pre-registered outcomes (P081 energy-placement table; N = 0, 1 recomputed here), with the P061 prior (0.015) and the 0.5 threshold marked. Colours: dataviz default categorical slots (blue/orange/violet); no JS validator run (PAPER_GUIDE rule).

## 7. Failed approaches

- A first attempt to reproduce P081's 55–270 keV and 270–420 keV DM probabilities from band ratios failed: those windows are dominated by the δ ≥ 366 keV tail of P081's model mixture (means 1.03 and 4.48 per 2.8 t·yr against medians far lower), so a single Gamma cannot represent them; we quote P081's numbers directly for those windows.
- Reproducing P088's LZ-1000 phd null date to better than a year would require its exposure-by-accrual dating and toy marginalisation; not attempted (dates are "needed exposure on disk" in both papers).
- We considered re-deriving P081's energy-placement Bayes factors; they require the per-model window ratios of 298 spectra (P027 cache) and would be a recomputation of P081, contrary to this paper's remit. Only the background denominators were checked.

## 8. Discussion

The corpus has converged on a clear structure. Counting in LZ alone cannot settle the event: a lone new band event is what both the DM posterior (P = 0.17 for exactly one) and a steady unknown (0.25) predict, and the modelled background is already so small (5.6×10⁻⁴) that beating it is not the issue. What decides is *where* the next events fall and *in which detector*. Two of the decisive datasets already exist: LZ's own 600–1000 phd region in the untouched data (which will kill δ ≥ 366 keV or produce the 9–73 events those models require) and the 17.9 t·yr of xenon data never examined above 200 keV. The honest forecast for LZ's next 2.8 t·yr is emptiness with probability 0.75 (DM) to 0.79 (mixture); a single event should be reported as "consistent with a population, not confirmation", and two events should be published with their energies and the 1000 phd extension. The base rate (P083: 0/25 unpredicted ~3σ anomalies became real) and the prior sensitivity (P061) mean that even the best realistic 2027 outcome — two events, one above 270 keV — leaves P(DM) near 0.6, not 0.99; the 0.99 needs both events above 270 keV or a second detector.

## 9. References

LZ Collaboration, arXiv:2609.02823 (2026). G. Cowan, K. Cranmer, E. Gross, O. Vitells, EPJC 71, 1554 (2011). R. E. Kass, A. E. Raftery, JASA 90, 773 (1995). A. Gelman et al., *Bayesian Data Analysis*, 3rd ed. (CRC, 2013). G. J. Feldman, R. D. Cousins, PRD 57, 3873 (1998). D. Baxter et al., EPJC 81, 907 (2021). Corpus: P001, P002, P003, P012, P014, P016, P020, P027, P034, P038, P041, P045, P046, P047, P048, P050, P052, P057, P061, P062, P067, P068, P069, P072, P076, P079, P081, P082, P083, P088, P093.

## 10. Tools and provenance (mirrors provenance/P099.json)

- **Script:** `output/code/P099_decisive_tests.py` — `.venv/bin/python output/code/P099_decisive_tests.py > output/work/P099/run_log.txt 2>&1` (runtime ≈ 1.5 s). Outputs: `tables/P099_forecast_counts.csv`, `P099_decision_rule.csv`, `P099_null_branch.csv`, `P099_reanalysis_on_disk.csv`, `P099_LZ_untouched_5sigma.csv`, `P099_sideband.csv`, `P099_exothermic_companions.csv`, `P099_test_x_interpretation.{csv,md}`, `P099_kill_conditions.csv`, `P099_preregistered_predictions.csv`, `P099_results.json`, `figures/P099_fig1_forecast.png`.
- **Software:** python 3.12.13; numpy 2.5.3; scipy 1.18.1 (stats.gamma, stats.nbinom, stats.poisson, stats.norm, optimize.brentq); pandas 3.0.5; matplotlib 3.11.2 (Agg); common/lzcommon.py (LZ constants dict only). WimPyDD not called.
- **Local inputs:** PAPER_GUIDE.md; results_ledger.csv (all 88 rows); papers P081, P088, P079, P069, P082, P093, P034, P050, P057, P038, P061, P083, P086, P072, P076, P052 (full) and P089/P092/P096/P097 (headers); work tables P081 (predictive_counts, decision_table, energy_placement, Pge1_vs_exposure, bestfit_vs_posterior_mean, variants, results.json), P088 (null_branch, timeline_table, LZ_sideband, settled, AvsB, cawo4, isotope_pair, results.json), P069 (hidden_events, exposure_ledger); provenance/P088.json (format); data_requests/index.csv; environment/ENVIRONMENT_versions.txt; lzcommon function list.
- **Recalled knowledge:** Gamma–Poisson (negative-binomial) predictive and Exponential(1) = Gamma(1,1) (certain); exact-Poisson counting significance Z = Φ⁻¹(1 − P(≥N | b)) and the 5σ threshold 2.87×10⁻⁷ (certain). Calendar-month conversion of decimal years (certain).
- **Datasets:** none. **Data requests:** none new; DR-001/002/003 discussed (status pending).
- **WimPyDD-generated files:** none.
