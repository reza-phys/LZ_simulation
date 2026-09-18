# P045 — What a lower limit that lifts off zero means for one event: coverage, flip-flopping and the Baxter et al. conventions — research record

Simulated date 2026-09-10. Category STAT (physics.data-an, cross-list hep-ex). Author profile: frequentist statisticians in particle physics.
Scripts: `output/code/P045_lower_limit.py` (all computations; 648 s, dominated by the exact Feldman–Cousins coverage tables) and
`output/code/P045_figures.py` (figures from the saved tables; also called by the main script). Run from the simulation root with
`.venv/bin/python output/code/P045_lower_limit.py`. Numbers below are in `P045_results.json`, `P045_single_event_intervals.csv`,
`P045_spectral_beta_intervals.csv`, `P045_coverage_b{0.00057,1,3}.csv`, `P045_raster_liftoff.csv`, `P045_pcl_check.csv` and `P045_run.log`.

## 1. Motivation and what LZ says

LZ (arXiv:2609.02823) reports "two-sided 90% confidence level intervals" for every model, notes that "given the observed local significance,
the lower limit is non-zero for some masses" (Results, tex line 280) and that "the lower limit lifts off from zero because of the presence
of the event of interest" (Fig. 6 caption, line 290). The statistics paragraph (line 248) says the analysis "uses a Profile Likelihood
Ratio (PLR), using a two-sided test statistic", "following the recommendations in" Cowan et al. (2011) and Baxter et al. (2021). The LEE
supplement (line 500) states: "we perform a raster scan over masses ... We do not set an additional threshold requirement on the local
significance of any model to report a lower limit on the coupling strength in this scan." The local-significance supplement (lines 820–823)
says a minimum of 30,000 toy datasets were generated per model to obtain p₀.

Corpus inputs: P012 digitised Fig. 6 bottom from the vector PDF (1000 GeV: d₁₀ = 0.171 / 0.417 / 0.614 for lower / median sensitivity /
upper; N(d₁₀ = 1) = 12.9 events at 1000 GeV in LZ's normalisation, so 0.378 / 2.24 / 4.87 events); P017 read the same axis as 0.154 / 0.375 /
0.553 (0.306 / 1.81 / 3.94 events); the two differ by a common 11% scale in d₁₀ (log axis calibration), but agree on the *ratio* lower/upper
= 0.2785 in d₁₀, i.e. 0.0777 in events. P007/P011 digitised the top panel (O₁ inelastic, 1000 GeV): converted intervals 1.5×10⁻⁴²–3.0×10⁻⁴¹ cm²
(δ = 300 keV) and 1.1×10⁻⁴⁰–1.8×10⁻³⁹ cm² (350 keV), ratios 0.050 and 0.061 in events; P007 finds the upper edges correspond to 3.2–3.8 events.
P021 quotes the single-event FC interval as [0.105, 3.65]; in fact (Sec. 3) 0.105 is the FC *and* asymptotic lower edge but 3.65 is the
asymptotic upper edge (FC gives 4.36). P001: b_eff implied by 3.4σ is 3.4×10⁻⁴ (exact) to 1.1×10⁻³ (asymptotic). P016: b_H = 5.7×10⁻⁴ in
200–270 keV. P008: N_eff = 12.2 at 3.4σ, 4.8 at 2σ; NR-band background 2.92 events in 5.4–270 keV (accidentals 2.7 with a 15 keV
exponential, atm-ν 0.11, MSSI 0.0049 flat above 50 keV, neutrons 0.05), 0.0022 above 200 keV. We also looked at `inputs/figures_png/Fig6_O1_L10_limit_stacked.png`
ourselves: the bottom-panel lower curve at 1000 GeV sits at d₁₀ ≈ 0.17 on a log axis, a factor ≈ 3.6 below the upper curve; an FC lower
edge of 0.105 events would be at d₁₀ = √(0.105/12.9) = 0.090, visibly below the 0.1 tick. The digitisation is not the issue.

Questions: (a) what lower edge do the proper two-sided constructions give for one event on b ≈ 5.7×10⁻⁴, and which construction (if any)
reproduces LZ's 0.31–0.38 (bottom) or ≈0.2 (top)? (b) what is the coverage of the two-sided interval reported without a discovery
threshold, and what does flip-flopping do? (c) how often does a background-only raster scan produce a non-zero lower limit somewhere?
(d) how should "the lower limit is non-zero" be read?

## 2. Framework and formulae

Single-bin Poisson: n ~ Poisson(μ + b), n_obs = 1. Test statistic (Cowan et al. t̃_μ; identical to the Feldman–Cousins likelihood-ratio
ordering for a single bin): t̃_μ(n) = −2 ln[P(n|μ+b)/P(n|μ̂+b)], μ̂ = max(0, n−b). For n = 1 and μ̂ = 1−b:
  t̃_μ(1) = 2[(μ+b) − 1 − ln(μ+b)],  t̃_μ(0) = 2μ.
Constructions compared:
1. **Feldman–Cousins (FC)**: exact Neyman construction; at each μ the acceptance region is filled in order of increasing t̃_μ(n) until
   Σ P(n|μ+b) ≥ 0.90; μ is in the interval iff n_obs is accepted. Edges by bisection (grid + 40 bisections). For n = 1 the lower edge is
   analytic: n = 1 leaves the acceptance region when P(0|μ+b) ≥ 0.90, i.e. μ_lo = −ln 0.90 − b = 0.1054 − b.
2. **Toy-calibrated t̃_μ**: at each μ on a grid (Δμ = 0.002 below 1), 3×10⁴ toys n ~ Poisson(μ+b); critical value c(μ) = smallest t with
   P(t̃ ≤ t) ≥ 0.90 (inverted cdf); accept if t̃_μ(1) ≤ c(μ). Must reproduce FC up to MC noise (it does).
3. **Naive Δχ²**: {μ ≥ 0 : t̃_μ(1) ≤ 2.706}. Also the unbounded t_μ variant (μ̂ = n − b; identical for n = 1).
4. **Cowan asymptotic t̃_μ**: the boundary-aware asymptotic cdf of t̃_μ under μ′ = μ (Cowan et al. 2011, Sec. 3.7):
   F(t̃|μ) = 2Φ(√t̃) − 1 for t̃ ≤ μ²/σ², and Φ(√t̃) + Φ((t̃ + μ²/σ²)/(2μ/σ)) − 1 above; accept μ if 1 − F(t̃_obs|μ) > 0.10. σ² = μ + b
   (Fisher information at μ) or σ² = μ²/q_{μ,A} with the μ′ = 0 Asimov data set (σ² = μ/2 as b → 0). For μ/σ → 0 the 90% critical value
   is 1.64 (Φ(√c) = 0.9) instead of 2.706; for μ/σ ≫ 1 it is 2.706.
5. **Classical central** (α/2 each tail), FC and Δχ² = 1 intervals at 68.27% (to compare with Table I's 1.0 (+1.4, −0.7)).
6. **Smeared calibration**: if the toy distribution of t̃_μ is continuous (nuisance parameters, ~1700 fluctuating background events, the
   toy event's position), the n = 1 "cluster" of toys is no longer an atom and the observed statistic is accepted only if a fraction f of
   the one-event toys is at least as extreme: p(μ) = f·P(1|μ+b) + P(≥2|μ+b), edge where p = 0.10 (valid while t̃_μ(0) < t̃_obs, i.e. μ ≲ 0.6).
   f = 1 reproduces FC (ties count as "at least as extreme").

Spectral information (Part B): with the event at x and the full-PDF likelihood L(s) ∝ e^{−s−b}[s f_s(x) + b f_b(x)] ∝ e^{−s}(s + β),
β = b f_b(x)/f_s(x) = b/r. Every s-independent factor cancels; the interval is the single-bin one with b → β, and profiled nuisance
parameters only add s-independent factors at leading order. The lower edge is monotonically *decreasing* in β: 0.1054 − β (FC) or the
root of 2[(s+β) − 1 − ln(s+β)] = 2.706.

Coverage (Part C1): for each construction and b ∈ {5.7×10⁻⁴, 1, 3}, intervals I_n for n = 0…59; coverage(μ) = Σ_n 1[μ ∈ I_n] P(n|μ+b) on
751 μ values in [0, 12]. Reporting policies with a 3σ threshold (Z(n) = Φ⁻¹(1 − P(≥n|b)), one-sided): (i) FC always (LZ); (ii) Baxter-type:
FC always, but only the upper edge is quoted below 3σ (interval [0, FC_hi]); (iii) flip-flop classical: one-sided 90% UL (P(≤n|μ+b) = 0.10)
below 3σ, classical central 90% interval above; (iv) flip-flop mixed: one-sided 90% UL below 3σ, FC above.

Raster-scan toy (Part C2): 12 independent bins in observed recoil energy with edges 5.4, 9, 14, 22, 33, 48, 66, 90, 118, 150, 190, 230,
270 keV (widths growing roughly like the resolution σ_E = 23√(E/248) keV, so each bin is one "resolution-wide window" in P008's sense; 12 is
P008's N_eff at 3.4σ). Background per bin from P008's NR-band model (accidentals 2.7 × exp(−E/15 keV), atm-ν 0.11 × exp(−E/28.6 keV) [scale
chosen so that 10⁻⁴ events lie above 200 keV, P019], MSSI 0.0049 flat in 50–270 keV, neutrons 0.05 × exp(−E/50 keV) [assumed shape]):
b = 0.593, 0.622, 0.655, 0.491, 0.296, 0.129, 0.052, 0.016, 0.0060, 0.0031, 0.0017, 0.0012 (total 2.865; 190–270 keV 0.0030, cf. P008's 0.0022;
variant scaled to P016's 5.7×10⁻⁴). "Lift-off" in bin i: n_i outside the FC acceptance region of μ = 0 (equivalently, for the naive Δχ²
construction, t̃_0(n_i) > 2.706 with n_i > b_i). Family-wise probability under H0 = 1 − Π(1 − p_i), accumulated from the high-energy end
(M = 1 is the 230–270 keV bin). Monte Carlo of the full procedure (2×10⁵ experiments) under s_true = 0, 1, 2.4 signal events in the 230–270 keV bin.
Analytic reference for a continuous statistic: P(some model lifts off) = 1 − 0.9^{N_eff}, since "μ = 0 excluded at 90% two-sided" ⇔ t̃₀ > c₉₀(0)
⇔ p₀ < 0.10 (under μ = 0 the t̃₀ distribution is ½δ(0) + ½χ²₁, so c₉₀(0) = 1.64 and Z_local > 1.28σ one-sided).

Power-constraint check (Part D): from the P012 digitised table, ρ = (d₁₀,upper/d₁₀,median)² is observed/median upper limit in events. In the
Gaussian regime the two-sided 90% upper edge is μ̂ + 1.645σ and the median 1.645σ, so μ̂/σ = 1.645(ρ − 1); a power constraint at the −1σ band
(M_min = 0.16, Cowan–Cranmer–Gross–Vitells 2011 PCL, recalled/likely) replaces ρ by max(ρ, 0.645/1.645 = 0.392).

## 3. Results

### 3.1 Single-event intervals (n = 1), expected signal events

| b | FC 90% | toy t̃_μ 90% | naive Δχ² = 2.706 | Cowan asympt. (σ² = μ+b) | Cowan asympt. (Asimov σ) | central 90% | FC 68.27% | Δχ² = 1 |
|---|---|---|---|---|---|---|---|---|
| 0 | [0.1054, 4.357] | [0.108, 4.350] | [0.1057, 3.647] | [0.179, 3.647] | [0.160, 3.647] | [0.051, 4.744] | [0.368, 2.751] | [0.302, 2.358] |
| 5.7×10⁻⁴ | [0.1048, 4.357] | [0.106, 4.350] | [0.1051, 3.646] | [0.179, 3.646] | [0.160, 3.646] | [0.051, 4.743] | [0.368, 2.750] | [0.301, 2.357] |
| 10⁻³ | [0.1044, 4.356] | [0.102, 4.350] | [0.1047, 3.646] | [0.179, 3.646] | [0.160, 3.646] | [0.050, 4.743] | [0.367, 2.750] | [0.301, 2.357] |
| 10⁻² | [0.0954, 4.347] | [0.100, 4.340] | [0.0957, 3.637] | [0.172, 3.637] | [0.157, 3.637] | [0.041, 4.734] | [0.362, 2.741] | [0.292, 2.348] |

Checks: FC at b = 0 reproduces the Feldman–Cousins Table IV entry 0.11–4.36 (recalled, certain) and the 68.27% entry 0.37–2.75 (recalled, likely);
the toy-calibrated construction reproduces FC to the grid step (0.002) and MC noise. The lower edge is the same, 0.105, for FC, toy-calibrated
and naive Δχ²; only the upper edge distinguishes them (4.36 exact vs 3.65 asymptotic). Expected (n = 0) upper limits at b = 5.7×10⁻⁴: FC 2.435,
naive Δχ² 1.353, Cowan-asymptotic 1.277, one-sided classical 2.302.

Table I of the paper, 1.0 (+1.4, −0.7), is reproduced by the Δχ² = 1 interval on a pure count, [0.30, 2.36] → 1.0 (+1.36, −0.70); the FC 68% interval
would read 1.0 (+1.75, −0.63). So LZ's quoted 1σ interval is the asymptotic (MINOS-type) one on what is effectively a single count.

### 3.2 Comparison with the digitised LZ edges (provisional pending DR-001)

| quantity | FC 90% | naive asympt. | Cowan asympt. | LZ Fig. 6 bottom P012 | P017 | LZ Fig. 6 top (P011 ratios × 3.65) |
|---|---|---|---|---|---|---|
| lower edge [events] | 0.105 | 0.105 | 0.16–0.18 | 0.378 | 0.306 | 0.18 (δ = 300), 0.22 (350 keV) |
| upper edge | 4.36 | 3.65 | 3.65 | 4.87 | 3.94 | 3.65 (P007: 3.2–3.8) |
| lower/upper | 0.0242 | 0.0290 | 0.044–0.049 | 0.0777 | 0.0777 | 0.050, 0.061 |
| median sensitivity | 2.44 | 1.35 | 1.28 | 2.24 | 1.81 | — |

Findings. (i) The bottom-panel *upper* edge (3.9–4.9) and median sensitivity (1.8–2.2) are compatible with either the FC or the asymptotic
construction within the ±11% (d₁₀) digitisation spread; the top-panel upper edges (3.2–3.8, P007) favour the asymptotic 3.65. (ii) The *lower*
edges are not reproduced by any 90% construction on ŝ = 1.0: the bottom panel's 0.31–0.38 is 3–3.6× the FC/asymptotic value and 1.7–2.1× the
Cowan-asymptotic value; the top panel's ≈ 0.18–0.27 matches the Cowan boundary-aware asymptotic formula (0.16–0.18 for σ² = μ/2 … μ+b) and is
1.7–2.6× FC. (iii) Numerical coincidences: a lower edge of 0.38 events is exactly the FC lower edge at CL = e^{−0.38} = 68.4% (FC: μ_lo = −ln CL),
and 0.31 is within 3% of the Δχ² = 1 lower edge 0.301 — i.e. the digitised bottom-panel lower edge equals the 1σ lower edge of Table I (0.3). The
CL equivalents of 0.105 / 0.18 / 0.20 / 0.31 / 0.38 events are 90.0 / 83.5 / 81.9 / 73.3 / 68.4%. (iv) Because the lower and upper edges come
from the same construction in LZ's caption, either the bottom panel's lower curve is not the 90% lower edge, or the construction is one whose
discreteness handling differs from FC (Sec. 3.3). The Data Release interval tables (DR-001, pending) would settle this; we mark all edge
comparisons provisional.

### 3.3 Why the lower edge is fragile: the smeared-calibration mechanism

For n = 1 on an empty background the FC lower edge is set by a single atom of probability: n = 1 is accepted at μ as long as the n = 0 toys do
not already fill 90%, i.e. down to P(0|μ) = 0.90. Any continuous component in the toy statistic (nuisance-parameter fluctuations, the ~1700
fluctuating background events, the toy signal event's position in the (S1c, log S2c) plane) splits the n = 1 atom, and the observed data are
accepted only if a fraction f of the one-event toys is at least as extreme. The edge is then
  μ_lo(f) = 0.105 (f = 1), 0.116 (0.9), 0.139 (0.75), 0.201 (0.5), 0.284 (0.31), 0.321 (0.25), 0.394 (0.15);
a lower edge of 0.18 / 0.20 / 0.31 / 0.38 requires f = 0.57 / 0.50 / 0.27 / 0.17. So a legitimately toy-calibrated two-sided construction can
report a lower edge anywhere between 0.105 and ≈ 0.4 events depending on how the observed data rank among one-signal-event toys — f = 0.5 is
"typical", and f ≈ 0.2–0.3 means the real data set is less μ-compatible than 70–80% of toys with one signal event (e.g. through the profiled
nuisance terms or a signal event at −1.5σ in a model where β varies across the band). The upper edge does not suffer from this (the n ≥ 2
atoms are small and the statistic is monotonic there), which is why the upper edges are stable at 3.6–4.4 while the lower edges scatter
between 0.18 and 0.38 across the two panels. In the effective one-event picture the event's position actually pushes the other way: larger
β at −1.5σ gives a *smaller* t̃ than a median-position toy event (dt̃/dβ = 2[1 − 1/(μ+β)] < 0), i.e. f > 0.5 and an edge below 0.14. We
therefore cannot identify the mechanism from outside; we can only state that no construction applied to ŝ = 1.0 and Z = 3.4σ yields a 90%
lower edge above ≈ 0.2 events without such a tie-breaking effect.

### 3.4 Spectral information cannot raise the lower edge

| fit region (b) | f_s/f_b at the event | β | ŝ | Δχ² 90% | FC 90% | Z (asympt.) |
|---|---|---|---|---|---|---|
| NR band, 2.92 | 10² | 0.0292 | 0.971 | [0.0765, 3.617] | [0.0762, 4.328] | 2.26 |
| NR band, 2.92 | 10³ | 2.9×10⁻³ | 0.997 | [0.1028, 3.644] | [0.1024, 4.354] | 3.11 |
| NR band, 2.92 | 10⁴ | 2.9×10⁻⁴ | 1.000 | [0.1054, 3.646] | [0.1051, 4.357] | 3.78 |
| science sample, 1713 | 10⁴ | 0.171 | 0.829 | [0, 3.475] | [0, 4.186] | 1.37 |
| science sample, 1713 | 10⁵ | 0.0171 | 0.983 | [0.0886, 3.629] | [0.0882, 4.340] | 2.48 |
| science sample, 1713 | 10⁶ | 1.7×10⁻³ | 0.998 | [0.1040, 3.645] | [0.1036, 4.356] | 3.28 |

The paper's 3.4σ (asymptotic) corresponds to β = 1.14×10⁻³ (P001's 1.1×10⁻³), i.e. a signal-to-background density ratio at the event of
1.5×10⁶ relative to the whole science sample or 2.6×10³ relative to the NR band; the 90% lower edge there is 0.1046. Adding the 2D PDF
information moves the lower edge from 0.1054 to 0.1046 — it cannot produce 0.38, or even 0.18.

### 3.5 Coverage

Single model, b = 5.7×10⁻⁴ (Fig. 2 left; `P045_coverage_b0.00057.csv`): FC never drops below 0.900 (minimum 0.900 at μ = 8.6; mean 0.947 over
μ ≤ 5); the naive Δχ² interval under-covers for 56% of μ ∈ [0, 12], with a minimum of 0.694 at μ = 1.36 (0.711 at μ = 1.5, 0.812 at 2.0; mean 0.891
over μ ≤ 5); the Cowan boundary-aware asymptotic interval is no better (minimum 0.681 at μ = 1.28; below 0.9 for 59% of μ). The hole at μ ≈ 1.3–2.5
is the n = 0 outcome: its asymptotic upper limit 1.35 (Cowan 1.28) excludes true signals of 1.4–2.4 events that FC's 2.44 still covers. Point values:
μ = 0.3: 0.963 for all; μ = 0.5: FC 0.910, Δχ² 0.986; μ = 1.0: 0.920 / 0.981; μ = 1.5: 0.981 / 0.711; μ = 2.0: 0.983 / 0.812; μ = 3.0: 0.917 / 0.917.

Reporting policies with a 3σ threshold (Fig. 2 right; b = 1 and 3):
- FC always (LZ's procedure): minimum coverage 0.900 (b = 1: at μ = 7.6; b = 3: at μ = 5.6). Correct by construction.
- Baxter-type (FC, but only the upper edge quoted below 3σ): minimum 0.900, mean 0.950 (b = 1) / 0.955 (b = 3) over μ ≤ 5 — conservative,
  because [0, FC_hi] ⊇ [FC_lo, FC_hi]. The threshold changes what is *communicated*, not the coverage.
- Flip-flop, one-sided UL / central interval at 3σ: minimum 0.855 (b = 1, μ = 2.9; below 0.9 for 31% of μ ≤ 12) and 0.855 (b = 3, μ = 2.3; 51% of μ).
- Flip-flop, one-sided UL / FC at 3σ: minimum 0.849 (b = 1, μ = 8.3; 25% of μ) and 0.849 (b = 3, μ = 6.3; 47%).
The classic Feldman–Cousins point survives in the Poisson regime: mixing a one-sided upper limit with a two-sided interval at a data-dependent
threshold under-covers by up to 5%; a unified construction with or without a communicated threshold does not. LZ's "no threshold" choice is
therefore harmless for coverage; its cost is interpretive (Sec. 3.6–3.7). At b = 5.7×10⁻⁴ the threshold is moot (any n ≥ 1 is 3.25σ) — the flip-flop
policies only under-cover there through the one-sided UL for n = 0 (2.30 vs FC 2.44; minimum 0.870 at μ = 2.3).

### 3.6 Raster scan: how often does a lower limit lift off under background only?

Per-bin lift-off probabilities (FC), high to low energy: 0.0012, 0.0017, 0.0031, 0.0059, 0.016, 0.050, 0.0076, 0.036, 0.087, 0.029, 0.025, 0.022
(the 33–48 keV bin with b = 0.49 lifts off already at n = 2 under FC ordering, 0.087, but needs n = 3 under Δχ², 0.014; all others agree).
Family-wise probability that at least one 90% lower limit is non-zero under H0, versus the number M of windows counted from the high-energy end
(`P045_raster_liftoff.csv`): M = 1 (230–270 keV only): 0.0012; M = 2: 0.0030; M = 3: 0.0061; M = 4: 0.012; M = 6: 0.077; M = 8: 0.117; M = 12 (whole
NR band): **0.254** (FC) / 0.194 (Δχ²). Variants: high-energy background rescaled to b_H = 5.7×10⁻⁴: 0.252; ×3: 0.259 — the rate is set by the
low-energy bins with b ~ 0.3–0.7, where an upward fluctuation of 2–3 counts excludes zero. For a continuous statistic the analytic rate is
1 − 0.9^{N_eff}: 0.10, 0.19, 0.27, 0.34, 0.47, 0.72 for N_eff = 1, 2, 3, 4, 6, 12; P008's N_eff(Z) = 4.8 at 2σ extrapolates to ≈ 3–4 at the 1.28σ
lift-off threshold, i.e. 0.27–0.34. Our bracket is therefore **25–35% (up to 72% if all 12 windows were independent with continuous statistics)**,
against LZ's global p = 4.7×10⁻³: a non-zero lower limit *somewhere* in the scan is 50–150 times more probable under H0 than the reported global excess.

Monte Carlo of the full 12-bin procedure (2×10⁵ experiments; signal in the 230–270 keV bin):
- s_true = 0: P(any lift-off) = 0.253; P(lift-off in the high-energy bin) = 0.0013; coverage of that bin's interval 0.9987; mean number of
  lifted-off models 0.28.
- s_true = 1: P(lift-off in the true bin) = 0.634 (= P(n ≥ 1) = 1 − e^{−1.0012}); P(any lift-off) = 0.726; P(false lift-off elsewhere) = 0.253;
  coverage of the true model's FC interval 0.919 (Δχ²: 0.981).
- s_true = 2.4: true-bin lift-off 0.909; coverage FC 0.988, Δχ² 0.872.
Under the L₁₀ best fit, a 90% lower limit lifts off in 63% of repeat experiments; under background only it lifts off in that window in 0.13%
but somewhere in the scan in 25%.

### 3.7 The Baxter et al. conventions and what LZ did

Recalled (reliability in brackets): Baxter et al. (2021) recommend a 90% CL two-sided PLR construction with Cowan et al.'s t̃_μ (certain);
that a lower limit / two-sided interval be reported only when the (global) significance exceeds 3σ, an upper limit being reported otherwise
(likely); the Cowan–Cranmer–Gross–Vitells power-constrained limit with M_min = 0.16 to avoid excluding signals below the −1σ sensitivity
(likely for the recommendation, certain for the PCL definition). LZ states (line 500) that no threshold was applied. Consequences:
1. Coverage is unaffected (Sec. 3.5); the intervals are valid 90% intervals model by model.
2. The lower edges displayed in Fig. 6 and the Data Release are statements that p_local < 0.10 for each model — a 1.28σ-equivalent per-model
   condition, satisfied for ~10% of null experiments per independent window and for 25–35% of null experiments somewhere in the scan.
   They add no information beyond the significance tables (S6/S7) and the 2.6σ global p.
3. With the 3σ threshold applied globally (2.6σ < 3σ), Baxter's convention would display upper limits only for every model; applied locally,
   the L₁₀ (≥ 400 GeV), L₁₆, L₂/L₄/L₆ᵛ/L₉/L₁₁/L₁₂/L₁₈–L₂₀ (≥ 200–400 GeV) and inelastic O₁ᵛ/O₄ (δ ≥ 300 keV) rows with Z ≥ 3.0 would
   keep two-sided intervals, and the ≈ 100–200 GeV points (Z = 2.4–2.9) that lift off in Fig. 6 bottom would not.
4. Power constraint: the low-mass (10–50 GeV) upper limits in Fig. 6 bottom sit at 0.62–0.74 of the median sensitivity in d₁₀ (0.39–0.55 in
   events), i.e. μ̂/σ = −0.73 to −1.01 in the Gaussian regime (consistent with P016's deficit, 42 observed vs 58.6 expected within ±1.5σ of the NR
   median). A −1σ power constraint would move only the 30 GeV point, by +0.8% in events (`P045_pcl_check.csv`); PCL is immaterial here.
   For m ≥ 100 GeV the observed limits are 1.39–1.47× the median in d₁₀ (1.93–2.17 in events; μ̂/σ = 1.5–1.9).

### 3.8 Interpretation of "the lower limit is non-zero" (question d)

For a model whose signal PDF concentrates on the event, "the 90% two-sided interval excludes zero" is exactly "an event was observed where fewer
than 0.105 background events were expected" (FC) or "p_local < 0.10" (any continuous two-sided construction). It is a per-model 1.28σ-level
statement, not a detection claim, and its numerical value (0.105–0.4 events) depends on how the construction treats discreteness. What is
informative is (i) the upper edge (3.6–4.4 events, robust across constructions to ±20%), (ii) the significance tables, and (iii) the global
p-value. Readers should translate "lower limit lifts off" as "this model would have produced ≥ 1 event with probability > 10%" and nothing more.

## 4. Robustness and validation
- FC edges agree with the published FC table (0.11–4.36 at 90%, 0.37–2.75 at 68.27%; recalled) and with the analytic −ln 0.9 − b.
- Toy calibration reproduces FC (0.106 vs 0.1048; 4.350 vs 4.357) — the toy grid step is 0.002 (0.01 above μ = 1).
- Background dependence: b from 0 to 10⁻² moves every lower edge by −b (≤ 0.01) and the upper edges by ≤ 0.01.
- The Cowan-asymptotic lower edge depends on the σ choice (0.160 Asimov, 0.179 Fisher); we quote 0.16–0.18.
- Raster-scan family-wise rate: 0.252–0.259 for high-energy backgrounds ×0.2–×3; the low-energy bins dominate. Bin definition is a
  simplification (the real light-WIMP likelihood fits 1710 events with ER leakage and nuisance parameters; there the per-model rate is the
  continuous 10%, giving the 1 − 0.9^{N_eff} curve as the upper envelope).
- The Gaussian-regime PCL check uses the digitised d₁₀ and is indicative only.

## 5. Failed or abandoned
- First run crashed: the one-sided upper limit and central interval for n ≪ b (b = 3, n = 0) have no positive root; guarded to return 0.
- A `pgrep`-based wait loop failed inside the sandbox (process listing blocked); replaced by a file-existence wait.
- Figures were first drawn inside the main script (650 s per rerun); moved to `P045_figures.py`, which reads the saved tables.
- We could not construct any 90% two-sided interval on ŝ = 1.0 with lower edge 0.31–0.38 without invoking the smeared-calibration
  mechanism (Sec. 3.3); this is reported as a result, not hidden.

## 6. Figures
- `figures/P045_fig1_intervals_by_construction.png` — the 90% (and 68%) intervals for one event at b = 5.7×10⁻⁴ by construction, with LZ's
  Table I and the digitised Fig. 6 edges (bottom: P012, P017; top: P011 ratio × 3.65).
- `figures/P045_fig2_coverage.png` — left: exact coverage vs true μ at b = 5.7×10⁻⁴ for FC (= toy-calibrated), Cowan-asymptotic and naive Δχ²;
  right: b = 1, the four reporting policies with a 3σ threshold.
- `figures/P045_fig3_liftoff_vs_M.png` — family-wise probability under H0 that some 90% lower limit lifts off, versus number of windows, for the
  12-bin toy (FC and Δχ²) and the continuous-statistic reference 1 − 0.9^{N_eff}, with LZ's global p-value.

## 7. References
LZ Collaboration, arXiv:2609.02823 (2026). G. J. Feldman, R. D. Cousins, Phys. Rev. D 57, 3873 (1998). G. Cowan, K. Cranmer, E. Gross, O. Vitells,
Eur. Phys. J. C 71, 1554 (2011); erratum 73, 2501 (2013). G. Cowan, K. Cranmer, E. Gross, O. Vitells, "Power-constrained limits", arXiv:1105.3166 (2011).
D. Baxter et al., Eur. Phys. J. C 81, 907 (2021). E. Gross, O. Vitells, Eur. Phys. J. C 70, 525 (2010). Corpus: P001, P007, P008, P011, P012, P016,
P017, P021; DR-001.

## 8. Tools and provenance
Agent tools: Read (PAPER_GUIDE; dossier; ledger rows 1–14 + compact print of 14–40; P001, P007, P008, P011, P012, P016, P017, P021; fulltext.tex
lines 244–293, 483–504, 808–827; lzcommon.py lines 1–125, 288–312; DR-001.md; P016.json; P012 fig6_bottom_digitised.csv; Fig6 PNG; three P045
figures twice), Bash (grep of tex / lzcommon / P012–P017 details; ls; versions; ledger print; two script runs; background waits; figure regeneration;
CSV summaries), Write (script, figures script, details.md, provenance JSON, paper), Edit (two script fixes, one figure title), Skill (dataviz),
ToolSearch (Monitor). Software: python 3.12.13; numpy 2.5.3 (default_rng(45045).poisson, quantile inverted_cdf, cumprod); scipy 1.18.1
(stats.poisson/norm/chi2, optimize.brentq); pandas 3.0.5; matplotlib 3.11.2 (Agg); common/lzcommon.py (LZ constants, imported; no WimPyDD).
Recalled knowledge: FC Table IV/II entries (certain/likely); Cowan et al. t̃_μ asymptotic cdf (certain); χ²₁ quantiles (certain); Baxter et al.
recommendations — two-sided t̃_μ at 90% (certain), 3σ threshold for lower limits (likely), PCL (likely); PCL M_min = 0.16 (likely); the
½δ + ½χ²₁ null distribution of t̃₀ (certain). Local inputs, data requests (DR-001 pending, filed by P012) and the full command list are in
`output/provenance/P045.json`.
