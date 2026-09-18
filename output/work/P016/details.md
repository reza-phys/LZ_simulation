# P016 — Why does LZ's likelihood still give 2.7σ to spectra with tens of low-energy events? A joint low/high-energy reassessment: research record

Simulated date 2026-09-05. Author profile: statistics-minded direct-detection phenomenologists. Category STAT (physics.data-an, cross-list hep-ph).
Script: `output/code/P016_joint_constraint.py` (run: `.venv/bin/python output/code/P016_joint_constraint.py`, 18 s). All numbers below are in
`output/work/P016/P016_results.json`, `P016_operator_table.csv`, `P016_calibration_baseline.csv`, `P016_calibration_scan.csv`,
`P016_Z_vs_Nlo_curves.csv`, `P016_two_bin_illustration.csv`, `P016_sensitivity_background.csv`, `P016_toy_check.csv`, and the run log `run_log.txt`.

## 1. Motivation and question

P003 computed, for every NREFT operator, the number N_lo of 5.4–55 keV signal events that accompany one 200–270 keV event, and found a tension:
elastic O4 (LZ's L15) needs N_lo ≈ 28 companions in the region searched by LZ in 2024 (same 220 live days, S1c 3–80 phd, no excess), yet Tables S6/S7
of arXiv:2609.02823 give it 2.7σ, only 0.7σ below the companion-free L10 (3.4σ). Only the N_lo ≳ 250 models (O1, O11^s) are driven towards 0σ.
P003 asked a STAT paper to extract the "effective N_max" of the LZ fit. We ask, concretely:

1. In an unbinned extended likelihood, how much low-energy signal does the fit actually tolerate, and why does N_lo ≈ 30 cost so little significance?
2. If the 2024 low-energy null is imposed explicitly (an external constraint allowing at most N_max ∈ {3, 5, 10, 20} extra events), what happens to
   the local significances, and which operators keep ≥ 3σ?
3. What do Bayes factors per operator look like once the low-energy likelihood is folded into P001's single-event formula, and which classes carry the posterior mass?
4. How sensitive is all this to the (poorly known) background near the NR median at low energy?

## 2. Inputs

| Input | Source |
|---|---|
| Unbinned extended likelihood, Eq. (S1); nuisance constraints; two-sided PLR; local Z from ≥ 30 000 toys per model | fulltext.tex lines 205–208, 248, 462–483, 820–823 |
| 2024 search: S1c 3–80 phd = 5.4–55 keV NR, 5.5 t FV; this analysis: same data, 4.71 t FV, S1c 3–600 phd | lines 110–117, 132 |
| Table I: accidentals 2.7 ± 0.6; atm-ν 0.11; ⁸B 0.057; detector NR [0, 0.118]; total 1713 ± 39 vs 1710 observed; L10^s best fit 1.0 (+1.4/−0.7) | lines 210–246 |
| Fig. 5 (three S1c panels of (log₁₀S2c − μ_NR)/σ_NR in 0.5σ bins; bottom panel integrates to 0.0106 ± 0.0008) | lines 255–266; `inputs/figures_png/Fig5_NR_distance_3panels_no_sig.png` |
| Tables S6/S7 (local significances by Lagrangian/operator and mass) | lines 825–909; `lz.LSIG`, `lz.OSIG` |
| Spectra dR/dE for O1, O3–O15 (s, v; 200/1000/4000 GeV), WimPyDD, Baxter SHM, natural Xe | `output/work/P003/P003_spectra.npz`; N_lo from `P003_operator_table.csv` |
| P003 efficiency model: plateau 0.96, erf roll-offs σ = 3.4 keV at 5.4 keV and 8 keV at 269.9 keV | `output/work/P003/details.md` sec. 3 |
| Lagrangian → operator mapping L1, L5 → O1; L2 → O10; L3 → O11; L4 → O6; L15 → O4; L10 → (q²/m_N²)O4 − O6 with N_lo = 0.20 | P003 (recalled there, likely) |
| Single-event Bayes factor B₁₀ = 1 + E_π[s e^{−s}]/b, flat prior s ∈ [0, 10] | P001 |
| Effective trials N_eff ≈ 12–14 | P008 work directory (`output/work/P008/results.json`; not yet in the ledger), P001 |

Recalled knowledge (flagged): (R1) Wilks/Cowan et al. asymptotics for the one-sided discovery statistic q₀, Z = √q₀ — certain; (R2) a Gaussian
90 % one-sided upper limit at best fit 0 corresponds to 1.2816σ — certain; (R3) Fisher information / Cauchy–Schwarz argument for the effective background — certain;
(R4) the 2024 LZ search (arXiv:2410.17036, likely) reported no excess with a best-fit signal near zero and a limit corresponding to a few events at high mass — uncertain
(same recall as P003 R4); (R5) Kass–Raftery interpretation scale for Bayes factors — certain.

## 3. Digitisation of Fig. 5, top panel (S1c < 250 phd)

The three panels share the x-axis (frame columns 219–2409 = −8…+8σ, as found by P004 for the bottom panel). For the top panel the frame rows are 497 (top) and
1299 (bottom), found from long dark horizontal runs; major y ticks were detected at rows 594.5, 735.5, 876.5, 1017.5, 1158.5, 1299.5, i.e. 10², 10¹, …, 10⁻³ with
141.0 px per decade (frame bottom = 10⁻³). Step-histogram levels were read as the top-most pixel of each coloured line in the middle 60 % of each 0.5σ bin
(colours as P004: total (0,0,255), accidentals (255,220,61), NRs (124,174,0), continuous ERs (0,194,249), internal γ/IC/EC (255,0,255)). Data points were found as
black connected components of 14–60 px with near-unit aspect ratio and > 60 % fill, centred within 0.08σ of a bin centre; values < 200 were rounded to integers.
P004 established that the y axis is events per 0.5σ bin (digitised bottom-panel integral 0.01053 vs 0.0106 in the caption).

Result (bin low edge, model total, data): −1.5: 1.08, 0; −1.0: 1.67, 0; −0.5: 2.96, 2; 0.0: 6.38, 9; +0.5: 14.9, 10; +1.0: 31.6, 21; +1.5: 59.8, 79; +2.0: 92.9, 96;
+2.5: 119, 125; +3.0…+7.5: 131–140 per bin (data 109–147). Below −1.5σ the model falls from 0.73 (−2.0) to 0.030 (−8.0) per bin with no data. Full table in
`P016_results.json: fig5_top_digitised`; Fig. 1 (`figures/fig1_top_panel_digitised.png`).

Checks. Sum of digitised data 1575 and model 1686; the science sample has 1710 events / 1713 fitted, of which ~25 lie in the middle and bottom panels (Fig. 5), so the
model sum is consistent to 0.1 % and the data sum is 7 % low, the shortfall being spread over the high-σ ER bulk (e.g. +3.0: 109 vs 131), which carries no signal and
only enters through the ER-scale nuisance θ. Near the NR median the relevant quantities are:

- within ±1.5σ of the NR median: **b = 58.6 model, n = 42 observed**; accidentals contribute 0.83 (the yellow curve is 0.1–0.2 per bin; the 2.7 accidentals of Table I are
  mostly at |σ| > 1.5), neutrino/detector NRs 0.0 (below the 10⁻³ floor in this panel), the rest is ER leakage (continuous β and the ¹²⁵I/¹³³Xe/EC "internal" component);
- **below the median (−1.5 … 0σ): b = 5.71, n = 2** — a deficit that matters below;
- Fisher-information effective background for a unit-normal signal, b_eff = 1/Σ_i g_i²/b_i = **22.5 events** (g_i = Gaussian bin fractions). By Cauchy–Schwarz this is
  smaller than the 58.6 in ±1.5σ because half of the signal sits below the median where the background is only ~6.

We adopt these numbers with the following ranges: the panel is S1c < 250 phd (≈ 5.4–125 keV NR), while the 2024 ROI is S1c < 80 phd; the fraction f_exp of the
near-median background that the low-energy signal actually overlaps is scanned in {0.5, 1} (Asimov-like scaling of both model and data); the background scale κ is
scanned in {0.5, 1, 2, 3} (Sec. 8).

## 4. Companion counts per 200–270 keV event

With the P003 spectra and efficiency we recomputed N_lo = R(5.4–55)/R(200–270) (agreement with P003's table to 0.3 % max) and added the two regions that the LZ
likelihood also sees: N_L2 = R(55–125)/R(200–270) (rest of the top panel) and N_M = R(125–200)/R(200–270) (middle panel of Fig. 5, S1c 250–500 phd, where the
background within ±1.5σ is 3 × 10⁻⁴ … 10⁻² per bin, total ≈ 0.02, with zero observed events). At 1000 GeV:

| model | N_lo | N_55–125 | N_125–200 | | model | N_lo | N_55–125 | N_125–200 |
|---|---|---|---|---|---|---|---|---|
| L10 | 0.20 | 0.63* | 0.70* | | O10^s | 3.08 | 3.60 | 2.19 |
| O6^s | 0.43 | 1.37 | 1.52 | | O3^v | 5.03 | 5.13 | 5.22 |
| O15^v | 0.75 | 2.07 | 3.70 | | O13^s | 4.90 | 2.25 | 0.78 |
| O5^v | 1.05 | 1.60 | 1.32 | | O8^v | 9.18 | 4.41 | 1.93 |
| O15^s | 1.91 | 2.41 | 0.54 | | O4^s | 28.1 | 7.14 | 2.84 |
| O9^s | 2.07 | 1.07 | 1.61 | | O11^v | 34.0 | 2.66 | 1.07 |
| O14^s | 2.34 | 1.15 | 1.67 | | O11^s | 258 | 24.9 | 7.22 |
| | | | | | O1^v | 477 | 6.67 | 1.84 |
| | | | | | O1^s | 2745 | 82.4 | 9.81 |

*L10's 55–200 keV companions are taken from O6 (same q⁴ transverse/longitudinal spin response family), scaled to N_lo = 0.20; this is an approximation.

## 5. The few-bin likelihood

We replace LZ's unbinned {S1c, log₁₀S2c} likelihood by an extended Poisson likelihood over: the top-panel bins i from −8 to +2σ (20 bins; signal beyond +2σ is 2.3 %),
one middle-panel bin (b_M = 0.02, n_M = 0), and the high-energy bin H (200–270 keV NR band; n_H = 1, background b_H). With s the expected number of signal events in
200–270 keV, the signal expectation in top-panel bin i is s (N_lo + N_L2) g_i with g_i = Φ(e_{i+1} − μ) − Φ(e_i − μ) (NR events are Gaussian in σ_NR units by
construction; μ = 0, variant −0.3), in M it is s N_M, in H it is s:

  ln L(s, θ) = Σ_i [n_i ln(θ b_i + s N_top g_i) − θ b_i − s N_top g_i] + n_M ln(b_M + s N_M) − (b_M + s N_M) + ln(b_H + s) − (b_H + s) − (θ − 1)²/(2σ_θ²)

with θ an ER-leakage scale (σ_θ = 0.3; variants 1.0 and 0.1 change nothing at the 0.01σ level because n ≈ b). q₀ = 2[ln L(ŝ, θ̂) − ln L(0, θ̂₀)], Z = √q₀ (Wilks),
ŝ ≥ 0. Optimisation: θ profiled by bounded scalar minimisation for each s on a 90-point log grid plus refinement.

**Anchor.** b_H is the effective background under one signal event's worth of PDF around the observed event. P001 showed LZ's 3.4σ implies b_eff = 3.4 × 10⁻⁴ (exact
Poisson) to 1.1 × 10⁻³ (asymptotic). Here we anchor b_H such that LZ's maximum-significance model, L10 at 1000 GeV *including its own companions*, gives exactly 3.4σ:
**b_H = 5.70 × 10⁻⁴** (with the companion-free asymptotic anchor 1.14 × 10⁻³, L10 would only reach 3.19σ because of the −s(0.63 + 0.70) companion penalty). Everything
below is therefore a statement about *differences* between models at fixed b_H; the absolute scale is LZ's.

Consistency: the fitted total L10 signal (all energies) is ŝ(1 + N_lo + N_L2 + N_M) = 1.26 events, against LZ's post-fit 1.0 (+1.4/−0.7) in Table I.

### 5.1 The mechanism (two-bin closed form)

Take one low-energy bin with n_L = b_L and a companion count N_lo. Then ln L(s) − ln L(0) = ln(1 + s/b_H) − s + b_L[ln(1 + u) − u], u = s N_lo/b_L. In the regime
1/b_H ≫ 1 the maximum is at b_L u² = 1 + u, i.e. **u = [1 + √(1 + 4b_L)]/(2b_L)** (0.56, 0.29, 0.16 for b_L = 5, 15, 45), so ŝ = u b_L/N_lo ≈ √b_L/N_lo and the
low-energy penalty saturates at b_L[ln(1+u) − u] ≈ −½. The gain from the single event is ln(ŝ/b_H): **it falls only logarithmically as N_lo grows**, while the penalty
for keeping s fixed would grow quadratically — so the fit shrinks s instead. Hence

  Z² ≈ 2[ln(√b_eff / (N_top b_H)) − ½ − ŝ(1 + N_M)]     (large N_lo; `Z_analytic` column of the operator table),

which for O4 gives 2.98σ (full model 2.70σ; the difference is the observed deficit below the median, which steepens the penalty). Two-bin illustration
(`P016_two_bin_illustration.csv`, b_H = 1.14 × 10⁻³, n_L = b_L):

| N_lo | b_L = 5 | 15 | 45 |
|---|---|---|---|
| 0.2 | 3.40 | 3.40 | 3.40 |
| 3 | 3.29 | 3.34 | 3.38 |
| 28 | 2.76 | 2.91 | 3.05 |
| 258 | 1.88 | 2.10 | 2.32 |
| 2752 | 0.64 | 0.90 | 1.18 |

A 2.7σ result for N_lo ≈ 28 is therefore *expected* for any b_L of several events, as anticipated in the assignment.

### 5.2 Calibration against Tables S6/S7 (`P016_calibration_baseline.csv`)

| LZ model = operator | N_lo | Z_LZ | Z_model | ŝ (events in 200–270 keV) | fitted 5.4–55 keV events |
|---|---|---|---|---|---|
| L10^s | 0.20 | 3.4 | 3.40 (anchor) | 0.499 | 0.10 |
| L4^s = O6^s | 0.43 | 3.1 | 3.26 | 0.317 | 0.14 |
| L4^v = O6^v | 0.45 | 3.1 | 3.25 | 0.305 | 0.14 |
| L2^s = O10^s | 3.08 | 3.0 | 3.08 | 0.175 | 0.54 |
| L2^v = O10^v | 3.15 | 3.1 | 3.07 | 0.169 | 0.53 |
| L15^s = O4^s | 28.1 | 2.7 | **2.70** | 0.057 | 1.60 |
| L15^v = O4^v | 29.4 | 2.7 | 2.69 | 0.054 | 1.59 |
| L3^v = O11^v | 34.0 | 2.6 | 2.73 | 0.061 | 2.07 |
| L3^s = O11^s | 258 | 1.7 | 1.90 | 0.008 | 2.04 |
| L1^v = O1^v | 477 | 1.3 | 1.65 | 0.005 | 2.23 |
| L1^s = O1^s | 2752 | 0.0 | 0.48 | 0.0004 | 1.01 |

rms residual 0.21σ over 11 points; residuals ≤ 0.2σ except O1^v (+0.35) and O1^s (+0.48). The other mass columns behave the same: O4 at 200 GeV 2.33 vs 2.3,
O11^v 200 GeV 2.30 vs 2.2, O11^s 200 GeV 1.37 vs 1.1, O10^s 200 GeV 2.86 vs 2.8, O1^v 4000 GeV 1.74 vs 1.2. The calibration scan over f_exp ∈ {0.5, 1}, σ_θ ∈ {0.3, 1},
μ ∈ {0, −0.3}, with/without the 55–200 keV companions (`P016_calibration_scan.csv`) gives rms 0.13–0.24σ; the choice of f_exp and σ_θ changes Z by < 0.03σ.
The residual excess for O1 (0.5σ instead of 0) indicates that the real low-S1c NR-band data show a somewhat larger deficit relative to the model than the S1c < 250 phd
projection, or that LZ's binned toys give p₀ ≈ 0.5 whenever q₀ is a few tenths; we do not tune to it.

**The fitted low-energy population is 1.6–2.2 events for every N_lo ≥ 28 model** (last column): this is the effective tolerance of the fit. It is not that the likelihood
tolerates 28 companions; it is that it fits 6 % of an event at 248 keV.

### 5.3 The Z(N_lo) curve (`P016_Z_vs_Nlo_curves.csv`, Fig. 2)

Baseline (O4-like 55–200 keV companion ratios): Z < 3σ for N_lo > 10.9, < 2σ for N_lo > 160, < 1σ for N_lo > 937. Without 55–200 keV companions: 16.6 / 243 / 1421.
σ_θ = 1, f_exp = 0.5 and κ = 3 curves are indistinguishable from the baseline (crossings within 3 %).

### 5.4 What the best fit means

For O4 the best fit predicts ŝ = 0.057 events in 200–270 keV, i.e. P(≥ 1 event) = 5.5 %, together with 1.6 events at 5.4–55 keV, 0.4 at 55–125 keV and 0.16 at 125–200 keV
(2.2 events in total). For L10: ŝ = 0.50, P(≥ 1) = 39 %, total 1.26. For O10: ŝ = 0.175, P = 16 %. For O1^v: ŝ = 0.005, P = 0.5 %, yet Z = 1.65σ. The discovery
statistic measures ln[L(ŝ)/L(0)] — dominated by ln(ŝ/b_H) ≈ ln(100) for O4 — not whether the model predicts the observation. Local Z in Tables S6/S7 therefore says little
about the spectral compatibility of the models beyond a logarithmic dependence.

## 6. Explicit 2024 low-energy null (Sec. 4 of the script, `P016_operator_table.csv` columns Z_Nmax*)

We add −½ (s N_lo/σ_N)² with σ_N = N_max/1.2816 (a Gaussian measurement of the 5.4–55 keV signal count with best fit 0 and 90 % UL N_max), N_max ∈ {3, 5, 10, 20},
(a) on top of the Fig. 5 bins and (b) instead of them. Result (1000 GeV): the largest change in Z over all 29 models is **0.16σ for N_max = 3** and 0.08σ for N_max = 5
(O1^v: 1.65 → 1.49; O4: 2.70 → 2.64; O11^v: 2.73 → 2.64). The ≥ 3σ / 2–3σ / < 2σ classes are identical for all N_max:

- ≥ 3σ (13 models, all N_lo ≤ 4.9): L10 3.40, O15^s 3.27, O6^s 3.26, O6^v 3.25, O5^v 3.25, O9^s/^v 3.21, O14^s/^v 3.19/3.20, O13^s 3.16, O10^s 3.08, O15^v 3.08, O10^v 3.07;
- 2–3σ (13 models, N_lo 5–156): O8^v 2.97, O13^v 2.94, O5^s 2.92, O3^s 2.92, O3^v 2.89, O11^v 2.73, O7^s 2.71, O4^s 2.70, O7^v 2.69, O4^v 2.69, O12^v 2.50, O12^s 2.29, O8^s 2.16;
- < 2σ: O11^s 1.90, O1^v 1.65, O1^s 0.48.

With the 2024 constraint *alone* (no Fig. 5 bins) every model with N_lo ≤ 50 stays above 2.85σ even for N_max = 3. Reason: a soft (quadratic) constraint on s N_lo is
satisfied by shrinking s, which costs only ln s. The low-energy null cannot remove a single-event PLR significance; only a lower bound on s (a requirement that the model
actually predict the event) or a Bayesian treatment can.

## 7. Bayes factors and posterior weights (Sec. 4 of the script)

B₁₀ = 1 + (1/b_H) ∫₀¹⁰ (1/10) s e^{−s} W(s) ds, where W(s) = [∫dθ π(θ) L_L(s, θ)]/[∫dθ π(θ) L_L(0, θ)] is the marginal low-energy likelihood ratio from the Fig. 5 bins
(θ marginalised over its N(1, 0.3) prior on a 41-point grid) times e^{−s N_M}. Without any low-energy information B₁₀ = 176.5 at b_H = 5.7 × 10⁻⁴. With it (1000 GeV):

| model | N_lo | B₁₀ (Fig. 5 folded in) | B₁₀ (e^{−s N_lo}) | B₁₀ (log-uniform prior) | weight |
|---|---|---|---|---|---|
| L10 | 0.20 | 44.8 | 123 | 109 | 0.200 |
| O6^s / O6^v | 0.43 / 0.45 | 18.3 / 17.5 | 86 / 85 | 68 / 66 | 0.082 / 0.078 |
| O15^s | 1.91 | 18.2 | 21.8 | 68 | 0.081 |
| O5^v | 1.05 | 17.0 | 42.8 | 65 | 0.076 |
| O9^s / O9^v | 2.07 / 2.19 | 13.2 / 13.5 | 19.7 / 18.2 | 57 / 58 | 0.059 / 0.060 |
| O14^s / O14^v | 2.34 / 2.49 | 12.1 / 12.2 | 16.7 / 15.4 | 54 / 55 | 0.054 / 0.055 |
| O13^s | 4.90 | 9.5 | 6.0 | 48 | 0.042 |
| O10^s / O10^v | 3.08 / 3.15 | 6.3 / 6.0 | 11.6 / 11.2 | 37 / 36 | 0.028 / 0.027 |
| O15^v | 0.75 | 6.3 | 58 | 37 | 0.028 |
| O8^v | 9.2 | 3.6 | 2.7 | 26 | 0.016 |
| O3^s, O5^s | 12.6, 14.2 | 2.9, 2.9 | 1.9, 1.8 | 22 | 0.013 |
| O4^s / O4^v | 28.1 / 29.3 | 1.56 / 1.51 | 1.21 / 1.19 | 11.6 / 11.1 | 0.007 |
| O11^v, O7^s, O7^v | 34, 32, 36 | 1.64, 1.59, 1.52 | 1.1–1.2 | 11–12.5 | 0.007 |
| O12^v, O12^s, O8^s | 41, 100, 156 | 1.20, 1.07, 1.04 | 1.0–1.1 | 6.6, 3.6, 2.6 | 0.005 |
| O11^s, O1^v, O1^s | 258, 477, 2745 | 1.01, 1.005, 1.000 | 1.0 | 1.55, 1.15, 1.00 | 0.005, 0.004, 0.004 |

Equal prior weight per model (s and v listed separately). Cumulative: N_lo ≤ 1 → 39 %; N_lo ≤ 5 → 87 %; N_lo > 20 → 6.3 %; the top five (L10, O6^s, O15^s, O6^v, O5^v) → 52 %.
The log-uniform prior (P001 alternative) raises all B₁₀ by ~×2.5–7 because it puts weight at small s, exactly where the N_lo ≳ 30 models live; even so O4 has B₁₀ ≈ 11
against 109 for L10, and the ordering is unchanged. The e^{−s N_lo} variant (a background-free low-energy region) is harsher for N_lo ≳ 10 and gentler for N_lo ≲ 1
(no penalty from the 2-vs-5.7 deficit), and again leaves the ordering unchanged. Note that the Bayes factors already include the "Occam" effect that P001 applied by hand
(÷N_eff): here the operator-averaged B₁₀ over the 29 models is 9.3 with the Fig. 5 penalty, versus 176 without any low-energy information.

## 8. Sensitivity to the low-energy background (`P016_sensitivity_background.csv`)

Scaling the background model by κ = 0.5, 2, 3 with the data fixed (a mis-modelled leakage) changes Z by ≤ 0.02σ for κ ≥ 1 and by +0.07 (O4: 2.77) to +0.19 (O1^s: 0.67)
for κ = 0.5; scaling model and data together (a genuinely larger background) changes Z by ≤ 0.02σ for every model. This is the √b_eff inside the logarithm of Sec. 5.1:
a ×3 background moves ln L by ½ ln 3 = 0.55 at most, i.e. ~0.1σ at Z ≈ 2.7. Accidentals (2.7 events, mostly at |σ| > 1.5 and very low S1c) contribute 0.83 to the ±1.5σ
window; even if all 2.7 sat under the signal the conclusions would not change. The conclusion "the local significance is insensitive to N_lo and to the low-energy
background" is therefore robust; conversely, no plausible change to the low-energy background can rescue the O1/O11^s class or hurt the O6/L10 class.

## 9. Toy-MC check of the Wilks mapping (`P016_toy_check.csv`)

200 000 background-only toys of the few-bin model (θ fixed), ŝ by vectorised bisection: L10 q₀ = 11.56 → Z_Wilks 3.40 vs Z_toy 3.31 (p = 4.65 × 10⁻⁴); O10^s 3.08 vs 3.18;
O4^s 2.70 vs 2.82; O11^s 1.89 vs 1.95. Differences ≤ 0.12σ, in both directions, so the q₀ → Z mapping is adequate for model-to-model comparisons. P(q₀ > 0 | background) is
2 % for L10 (a signal template with almost no low-energy content is rarely preferred by pure background) but 30 % for O4 and 44 % for O11^s: a template with a large low-energy
population fits upward fluctuations of the ER leakage, which is another way of saying that its s is nearly unconstrained by the high-energy bin.

## 10. Failed or abandoned approaches

- Bayes factors with W(s) = ∫dθ π(θ) L(s,θ)/L(0,θ) (ratio inside the integral) overflowed (B₁₀ ~ 10⁷⁰): the correct quantity is the ratio of θ-marginals; fixed.
- The companion-free anchor b_H = 1.14 × 10⁻³ left the whole calibration curve ~0.2σ below LZ's (L10 at 3.19σ); anchoring on L10 with its companions fixed it.
- A full WimPyDD/NEST 2D likelihood was not attempted (no LZ PDFs); the Fig. 5 projection is the best public proxy.

## 11. Figures

- Fig. 1 `figures/fig1_top_panel_digitised.png`: digitised Fig. 5 top panel (model total, components, data) with the O4 companion population at s = 1 (35 events, dashed) and at ŝ = 0.057 (solid): at the best fit the companions are 0.4 events per bin under 3–15 background events per bin.
- Fig. 2 `figures/fig2_Z_vs_Nlo.png`: Z vs N_lo for the baseline and variants, with LZ's Table S6/S7 values at 1000 GeV.
- Fig. 3 `figures/fig3_operator_Z_Nmax.png`: per-operator Z (1000 GeV) with and without the explicit 2024 constraint, N_max = 3–20, with LZ's values.
- Fig. 4 `figures/fig4_posterior_weights.png`: relative posterior weights per operator (Fig. 5 likelihood vs e^{−sN_lo} penalty).

## 12. Discussion

The tension P003 identified is not a tension. LZ's profile likelihood does not "tolerate 28 low-energy events"; for O4 it fits ŝ = 0.057 events at 200–270 keV plus 1.6
events at 5.4–55 keV, and the 2.7σ is ln(1 + ŝ/b_H) ≈ ln 100 from one event sitting where the background is ~6 × 10⁻⁴. Because the gain is logarithmic in ŝ and the penalty
for companions is quadratic, the local Z falls from 3.4σ only to 2.7σ across a factor 140 in N_lo and reaches 2σ only at N_lo ≈ 160. Imposing the 2024 null explicitly, even
at N_max = 3, changes nothing (≤ 0.16σ), because the fit has already put the companions at the ~2-event level. The frequentist tables are therefore almost blind to spectral
shape; what discriminates is the Bayes factor (or equivalently the requirement that the fitted signal actually predict the event): the O4/O7/O11^v/O12 class has B₁₀ ≈ 1.0–1.6
(no evidence, Kass–Raftery), O10/O13^s/O8^v 3.6–9.5 (positive), O6/O5^v/O9/O14/O15^s 12–18 (positive to strong), L10 45 (strong). 87 % of the posterior mass, with equal model
priors, is in the N_lo ≤ 5 class. Relative to P001's model-averaged B₁₀ ≈ 7–37 (÷N_eff), the operator-averaged B₁₀ here is 9.3 — consistent, and now the average is dominated
by a small identifiable set of q²-suppressed spin operators and L10-like combinations, as P003 argued on spectral grounds. The event's evidential weight for elastic O1-, O4-,
O7-, O11-, O12-type interactions is nil, and the 2.6–2.7σ entries for them in Tables S6/S7 should not be read as support.

## 13. References

1. LZ Collaboration, arXiv:2609.02823 (2026) — Eq. (S1), Table I, Fig. 5, Tables S6/S7.
2. G. Cowan, K. Cranmer, E. Gross, O. Vitells, Eur. Phys. J. C 71, 1554 (2011).
3. D. Baxter et al., Eur. Phys. J. C 81, 907 (2021).
4. R. E. Kass, A. E. Raftery, J. Am. Stat. Assoc. 90, 773 (1995).
5. A. L. Fitzpatrick et al., JCAP 02 (2013) 004; N. Anand, A. L. Fitzpatrick, W. C. Haxton, Phys. Rev. C 89, 065501 (2014).
6. LZ Collaboration, 2024 WIMP-search result (arXiv:2410.17036; number recalled, likely).
7. Corpus: P001 (single-event Bayes factors), P003 (N_lo table and spectra), P004 (Fig. 5 digitisation geometry), P008 (LEE toys, work directory).

## 14. Tools and provenance

Mirrors `output/provenance/P016.json`. Agent tools: Read (PAPER_GUIDE, dossier, ledger, P001.md, P003.md, P003 operator table, fulltext.tex lines 108–167/205–285/460–501/820–911, lzcommon.py, P004 script excerpt, Fig. 5 PNG, four output figures), Bash (directory listings, tick/frame detection, three script runs, result inspection), Write (script, details, provenance, paper), Edit (script fixes). Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (stats.norm/poisson, optimize.brentq/minimize_scalar, special.erf, ndimage.label/find_objects, integrate.quad); pandas 3.0.5; matplotlib 3.11.2 (Agg); Pillow 12.3.0; common/lzcommon.py (LZ, LSIG, OSIG). WimPyDD was not called directly (spectra taken from P003's cache). Recalled knowledge: 5 items (Sec. 2). No datasets, no data requests, no WimPyDD-generated files.
