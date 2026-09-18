# P056 — ER/NR discrimination at 250 keV: the leakage predicted by the double-skew-Gaussian recombination model and how far the tails must be trusted

Research record (simulated date 2026-09-11). Script: `output/code/P056_er_leakage_model.py` (run from the root with
`.venv/bin/python output/code/P056_er_leakage_model.py`, ≈ 12 min; the first complete-computation pass crashed in the
figure section on a format-string bug after all numbers had been computed and logged — `run_log_first_pass.txt`; a second
pass was stopped to fix an empty-column selection in Fig. 1; the final pass (192 s) is `run_log.txt` and produced
`results.json` and the four figures; the script also writes `H3_store.npz` (195 MB of cached response arrays), which was
deleted from the corpus for size; all numbers are deterministic and seeded, and the final log's result sections are
byte-identical to the first pass). Tables:
`nest_skewness_and_widths.csv`, `digitised_bands_fig2_fig4.csv`, `pb212_points_fig2.csv`, `fig5_digitised_all_panels.csv`,
`band_calibration.csv`, `beta_leakage_by_S1c_bin.csv`, `fig5_bottom_projection.csv`, `ec_lines_leakage.csv`,
`sensitivity_summary.csv`. All LZ inputs from arXiv:2609.02823 (Tables I, S3, S4; Figs 2, 4, 5; Discussion; detector-response
supplement) via `lzcommon.LZ`. Recalled items are flagged **[recalled: reliability]**.

## 1. Motivation

LZ states that "very little leakage [is] predicted for S1c > 250 phd", that the ER-band widths needed a new fluctuation model
(NEST v2.4.0 failed; v2.4.5 parameterises σ_p as a double skew-Gaussian in x = log10 N_q, Table S4), that EC decays have
enhanced recombination, and that "there is a systematic uncertainty in projecting into the tail of the distribution that we do
not include". Fig. 5 (bottom) shows the ER components of the background model at S1c > 500 phd only between +1 and +5.5 σ_NR
(light blue "Continuous ERs", magenta "Internal γ + ICs + ECs", total 0.0106). P010 showed that an ER origin needs r = 0.939
(6.4σ deficit on a band-calibrated width, k = 0.50 on the Table S4 σ_p) and that LZ's "6.7σ" is 6.8 half-widths. Here we ask
the model question: with LZ's own tuned fluctuation model, what is the ER band's width *and shape* at S1c = 300–600 phd, what
leakage does it predict for β-like ERs and for the EC lines, how does that compare with Fig. 5, and how much does the answer
depend on the parts of the model that the calibration data cannot see (the skew of the recombination distribution and the way σ_p
is combined with N_i)?

## 2. Model

Notation: g₁ = 0.110 phd/photon, g₂ = 34.5 phd/electron; N_q = N_ph + N_e; N_i = N_q/(1+α_ex); r = 1 − N_e/N_i.

* **Mean yields**: NEST v2 β model with LZ Table S3 parameters (`lz.nest_er_yields`, nestpy 2.1.1, 96.5 V/cm, 2.9 g/cm³);
  the exciton/ion ratio α_ex(E) is taken from the same nestpy `YieldResult` (0.05 at 5 keV, 0.15 at 20 keV, 0.182 above
  40 keV). nestpy's own W is 13.44 eV/quantum; the event's N_q = 5179 corresponds to E_ee = 69.6 keV.
* **Quanta fluctuations**: N_q ~ N(μ_q, √μ_q) (Fano 1.0, measured from nestpy `GetQuanta`: 0.98–1.01);
  N_i | N_q ~ N(N_q/(1+α), √(N_q α)/(1+α)) (binomial exciton/ion partition; reproduces nestpy's sd(N_i) = 66 at 70 keV).
* **Recombination fluctuation (the object of the paper)**: Var(N_e | N_i) = r(1−r) N_i + (k σ_p(N_q) N_i)², with
  σ_p(x) = Σᵢ Aᵢ[1+erf(αᵢ(x−μᵢ)/(√2σᵢ))] exp(−(x−μᵢ)²/2σᵢ²), x = log10 N_q, Table S4 parameters
  (A₁ = 0.03174, μ₁ = 2.588, σ₁ = 0.3627, α₁ = −0.3583; A₂ = 0.06211, μ₂ = 5.2, σ₂ = 1.359, α₂ = −2.599). The form
  "σ_p × N_i" is NEST's GetQuanta convention **[recalled: likely]**; k = 1 is the literal reading, k is otherwise a calibration
  factor (§4). At N_q = 5179 (x = 3.714): term 1 = 7×10⁻⁵, term 2 = 0.0679, σ_p = 0.0683.
* **Shape of N_e | N_i**: skew-normal with mean (1−r)N_i, the sd above, and shape parameter a = s·a_NEST(E).
  a_NEST(E) is NEST's ER skewness, *measured* from nestpy `GetQuanta` (LZ_WS2024 detector, default `SkewnessER`
  formula) by inverting the sample skewness of 40 000 draws (`nest_skewness_and_widths.csv`): a = 2.2 (3 keV), 3.2 (10),
  3.7 (20–30), 2.8 (50), 1.8 (70), 1.0 (100), 0.6 (130) and 0 above ≈135 keV (NEST switches the skew off for N_q > 10⁴).
  The NEST skew is *positive*: the tail is towards MORE electrons (less recombination); the NR-ward (low-N_e) tail is
  lighter than Gaussian, falling as exp(−z²(1+a²)/2). Variants: s = 1 (NEST), 0 (Gaussian), 0.5, 1.5, −1 (mirrored skew,
  a heavy low-N_e tail of the same magnitude, used as a toy). Whether LZ's v2.4.5 model kept the v2.4.0 skewness is not stated
  in the paper; the Table S4 αᵢ are *not* skewness parameters of N_e — they shape σ_p(x) (P010 made the same point).
* **Detector** (as P024): S1c = Binomial(N_ph, g₁) with single-phe resolution 0.338 (nestpy LZ_WS2024) and a 2 % position
  residual; S2c = Binomial(N_e, ε = 0.80) × g₂/ε with single-electron Fano 4 and a 2 % position residual. ε = 0.80
  **[recalled: uncertain; LZ SR1 ≈ 80 %]**. Both are Gaussian-approximated.
* **Engine**: everything is integrated deterministically on a grid of 10-phd S1c columns (240–660) × 0.0025-dex log10 S2c rows
  (3.60–5.20): for each E (25–160 keV, 2-keV steps), 3×3 Gauss–Hermite nodes in (N_q, N_i), a 200-point N_e grid spanning
  ±14 sd, and Gaussian S1c/S2c response matrices. Tail probabilities are therefore numerically exact for the stated model
  (down to 10⁻²⁰ and below), with no MC-statistics floor. A 1.5×10⁶-event Monte Carlo with the same physics reproduces the
  engine's fixed-S1c percentiles to ≤ 0.0005 dex (540 phd: 4.6575/4.7735/4.8976 vs 4.6573/4.7736/4.8971).
* **ER continuum normalisation**: Table I continuum components in the ROI (internal β 1341 + ν-ER 140.6 + ¹³⁶Xe 110.0 +
  CH₃T/¹⁴C 55.3 + detector ERs 8.5 = 1655) divided by the model's ROI acceptance integral for a flat spectrum (1–40 keV,
  18.6 keV) → **89 events per keV_ee** in 2.84 t·yr (P010: 85), extrapolated flat to 40–160 keV_ee (±50 %: ²¹⁴Pb/²¹²Pb β
  shapes, ¹³⁶Xe rise, tritium absent above 18.6 keV). This gives 700–900 ERs per 50 phd at S1c = 300–600 (the grey cloud of
  Fig. 4; 1760 at 500–600).

## 3. Digitisations

**Fig. 4 ER/NR lines** (tick calibration as P024: 384.2 px/dex, residual 0.0005 dex; x = 0–800 phd across the frame). Blue
(ER) and red (NR) clusters every 10 phd from 250 to 800 (`digitised_bands_fig2_fig4.csv`); Fig. 2 gives the same lines to
±0.001 dex. Selected values:

| S1c | ER p10 / p50 / p90 | half-width | upper/lower | NR p10 / p50 / p90 | NR half-width |
|---|---|---|---|---|---|
| 300 | 4.376 / 4.465 / 4.553 | 0.088 | 0.993 | 3.912 / 3.943 / 3.975 | 0.031 |
| 400 | 4.492 / 4.596 / 4.699 | 0.103 | 0.983 | 3.951 / 3.979 / 4.008 | 0.029 |
| 500 | 4.617 / 4.734 / 4.849 | 0.116 | 0.988 | 3.980 / 4.008 / 4.034 | 0.027 |
| 540 | 4.667 / 4.789 / 4.910 | 0.122 | 0.990 | 3.989 / 4.016 / 4.044 | 0.027 |
| 600 | 4.746 / 4.871 / 4.995 | 0.125 | 0.991 | 4.003 / 4.029 / 4.056 | 0.027 |

Event: 0.0493 dex below the NR median = **1.80 drawn half-widths** — Fig. 5 plots it in the (−2, −1.5) bin, so Fig. 5's
"σ_NR" is the drawn 10–90 % half-width (0.027 dex), consistent with P010/P024. We use "event depth" = μ_NR − 0.0493 dex as
the operational "−1.5σ_NR" of the paper, and hw = drawn half-width as the Fig.-5 unit.

**Fig. 2 ²¹²Pb points** (green, 46/139/87), blob-labelled; overlapping markers merge, so each blob is weighted by
round(area/25 px). About the drawn ER median, S1c 450–560 (the top of the plot, 5.0, clips the upper side above ≈560 phd):
N ≈ 490 (108 blobs, 75 singles), mean residual −0.018 dex, weighted sd **0.134 dex**, weighted 10–90 % half-width **0.173 dex**
(drawn 0.117), p10/p90 = −0.20/+0.14 dex (lower-heavy), 5.9 % of the weight below −2 hw (Gaussian: 0.5 %), 0.4 % below −3 hw
(Gaussian 0.01 %), lowest point −3.6 hw (−0.44 dex, log10 S2c ≈ 4.35, still 0.33 dex above the NR band); nothing between
−3.6 hw and the 4σ grey cut. LZ's caption attributes such points to wall leakage in the out-of-FV calibration sample. Grey
(>4σ) points at 450–650 phd: 307, of which 4 lie between the bands and 8 within ±3 hw of the NR median. Blob weighting is
approximate (merged cores are under-counted, so the sd is an upper estimate, tail fractions upper estimates by ≲ ×2–3).

**Fig. 5** (all three panels; frames 496–1299, 1299–2102, 2102–2905 px; major ticks on the right frame edge: 141, 120,
96.75 px/decade; panel bottoms 10⁻³, 10⁻⁵, 10⁻⁶). Colour masks as P004. Checks: bottom total 0.0108 (caption 0.0106);
middle total 24.6 events (data points in that panel ≈ 25). Bottom (S1c > 500): continuous ERs **0.0072** in +1…+5.5 hw
(3.1e-4, 8.0e-4, 7.1e-4, 1.4e-4, 2.3e-4, 7.2e-4, 2.1e-3, 1.9e-3, 1.9e-4 per bin from +1), nothing below +1; internal
γ+IC+EC **0.0025** in +1.5…+5.5 (1.3e-4 … 5.1e-4 per bin). Middle (250–500): continuous ERs 1.76 in total, with a flat floor of
(2–8)×10⁻⁴ per bin from −3 to +1 hw (0.0025 below +1), i.e. *below the NR median*; internal 20.8.

## 4. Band width, k, and the shape test

Half-width ratio model/drawn (mean over 300–600 phd) vs k: 0.762 (k = 0.35), 0.995 (0.50), 1.247 (0.65), 1.508 (0.80),
1.859 (1.00) → **k = 0.503** (NEST skew) and 0.501 (Gaussian): P010's k = 0.50 is reproduced exactly. With k = 0.50 the model
half-widths are 0.088/0.101/0.115/0.120/0.125 at 300/400/500/540/600 vs drawn 0.088/0.103/0.116/0.122/0.125; medians are
0.010–0.015 dex below the drawn ones (Table S3 means vs LZ's drawn median; P009 found a similar few-per-cent quanta offset).
σ(N_e) at 70 keV: literal k = 1 → **303 e**; k = 0.50 → **154 e** (P010: 150); nestpy 2.1.1's own LZ_WS2024 v2.4.0-style
single skew-Gaussian in the electron fraction gives **86 e** (effective σ_p = 0.018). The literal Table S4 model therefore
predicts a band 1.85× wider than drawn and 1.3× wider than the ²¹²Pb scatter; the ²¹²Pb scatter (0.173) corresponds to
k ≈ 0.75 (model 0.172 at 540) but contains wall leakage. We keep k = 0.50 as LZ's drawn model, 0.75 as the "calibration
scatter" upper case, 1.0 as excluded-literal.

**Shape test.** The drawn lines are almost symmetric in log space: (p90−p50)/(p50−p10) = 0.983–0.993 at 300–600 phd
(digitisation ±0.005 dex per line → ±0.05 at 300, ±0.04 at 540). Model: Gaussian 0.974–0.988 (matches); NEST skew
**1.25 / 1.17 / 1.09 / 1.07 / 1.03** (300/400/500/540/600; 5σ off at 300–400, 2σ at 540); skew×0.5 1.08–0.99; mirrored
0.77–0.94. The 10–90 % lines of LZ's own model therefore favour |a| ≲ 1 at 45–70 keV (where NEST's formula gives 2.8–3.7):
either v2.4.5 dropped/reduced the v2.4.0 skewness or its lines are not drawn from the skewed distribution. Note that fixed-S1c
mixing itself produces a *lower*-heavy band (a Gaussian in N_e gives 0.974), so the near-symmetry is a non-trivial constraint.
The ²¹²Pb points are lower-heavy (p10/p90 −0.20/+0.14), but wall leakage acts in the same direction.

## 5. Leakage of the β continuum (`beta_leakage_by_S1c_bin.csv`)

Fractions of flat-spectrum ERs at a given S1c (and expected events in 2.84 t·yr at 89/keV) below the ROI top (S2c < 10^4.15),
below the NR median and below the event's depth; ROI leakage summed over 300–600 phd; gap G = S1c 480–600 between −1 hw and
the ROI top (0 events observed); event region V = S1c 500–600, log10 S2c 3.90–4.00.

| variant (all reproduce the drawn width unless k ≠ 0.5) | frac in ROI @300 | @500 | N(ROI, 300–600) | N(ROI, 500–600) | N(G) | N(V) |
|---|---|---|---|---|---|---|
| k = 0.50, NEST skew (baseline) | 1.4e-11 | 2.2e-12 | 1.8e-8 | 3.4e-9 | 4.1e-9 | 1.2e-12 |
| k = 0.50, Gaussian | 9.2e-6 | 3.9e-8 | 8.4e-3 | 4.3e-5 | 7.3e-5 | 6.5e-7 |
| k = 0.50, skew ×0.5 | 2.5e-7 | 4.5e-9 | 2.6e-4 | 5.2e-6 | 8.0e-6 | 2.7e-8 |
| k = 0.50, skew ×1.5 | 8.8e-16 | 1.5e-17 | 6.6e-13 | 3.8e-14 | 4.0e-14 | 5.9e-19 |
| k = 0.50, mirrored skew | 3.1e-4 | 5.1e-6 | 0.36 | 5.7e-3 | 8.8e-3 | 5.1e-4 |
| k = 0.75, NEST skew | 6.5e-6 | 3.0e-6 | 0.018 | 4.7e-3 | 5.6e-3 | 1.3e-4 |
| k = 0.75, Gaussian | 1.3e-3 | 1.1e-4 | 2.0 | 0.14 | 0.18 | 0.017 |
| k = 1.0, NEST skew | 9.4e-4 | 4.1e-4 | 2.6 | 0.65 | 0.72 | 0.073 |
| k = 1.0, Gaussian | 1.0e-2 | 2.1e-3 | 21 | 3.1 | 3.1 | 0.61 |
| k = 0.35, NEST skew | 7e-18 | 3e-23 | 5e-15 | 3e-20 | 4e-20 | 5e-27 |
| α₁ ×0.5 / ×1.5 | | | | 3.6e-9 / 3.3e-9 | | |
| α₂ ×0.5 / ×1.5 | | | | 4.3e-11 / 3.8e-9 | | |
| α₂ sign flipped (σ_p → 2×10⁻⁴, band half-width 0.05) | | | | 7e-109 | | |

Reading: (i) the literal k = 1 Gaussian is excluded by the data (21 ROI events at S1c > 300 phd, 3.1 in the empty gap;
P(0 | 3.1) = 4 %), the k = 1 NEST-skew case (2.6 ROI events, mostly at 300–350 phd where LZ attributes the observed
ROI-edge cluster to EC/⁸³ᵐKr; 0.72 in G) is marginal; (ii) for variants that reproduce the drawn band, the ROI leakage at
500–600 phd spans **4×10⁻¹⁴ … 6×10⁻³ events — eleven orders of magnitude — set entirely by the skew** of the recombination
distribution, which the width calibration does not fix; (iii) the Table S4 αᵢ move σ_p at x ≈ 3.7–3.9 by ≤ 8 % because the
erf factor is already on its plateau there (α₂ ×0.5: −8 % in σ_p → leakage ÷80; α₁ ±50 %: ±5 %); the αᵢ do *not* control
the tail; (iv) the drawn-line asymmetry (§4) points to the Gaussian row as the realistic central case: **4×10⁻⁵ leaked
β-ERs at S1c 500–600 (7×10⁻⁵ in the gap, 7×10⁻⁷ at the event)**; the NEST-skew row is 4 orders lower, the mirrored row
2 orders higher.

**Comparison with Fig. 5 bottom** (`fig5_bottom_projection.csv`, projected onto x = (log10 S2c − μ_NR)/hw for 500–600 phd):
LZ's continuous-ER curve holds 7.2×10⁻³ in +1…+5.5 and nothing below +1. Our Gaussian k = 0.50 gives 4.2×10⁻⁵ there (LZ is
170× heavier), NEST skew 3×10⁻⁹, mirrored 4.9×10⁻³ and k = 0.75 NEST skew 4.3×10⁻³ (both ≈ LZ). So LZ's PDF tail for
continuous ERs is far heavier than the band-calibrated Gaussian, i.e. conservative, but it is truncated at +1 hw: in the
(−2, −1.5) bin of the event, LZ has 0 (its floor there is accidentals + MSSI ≈ 4×10⁻⁵), our Gaussian 8×10⁻⁸, mirrored
6×10⁻⁵, k = 0.75 Gaussian 2×10⁻³, k = 1 Gaussian 0.08. Middle panel: LZ's light-blue floor of (2–8)×10⁻⁴ per bin down to
−3 hw at 250–500 phd (0.0025 below +1 hw) is reproduced by no recombination model (ours: < 10⁻⁸); it is a PDF-construction
floor (finite MC or smoothing) and, again, conservative.

## 6. EC lines (`ec_lines_leakage.csv`)

Lines at fixed E with mean N_e reduced by f (quanta conserved; N_q of the line from Table S3), same fluctuation model.
¹²⁴Xe KK: 564 decays (P010); ¹²⁵I K-capture: 10³ and 10⁴ decays (P010's range). Counts at S1c 480–600 in G (gap) and V
(event region) and in the ROI at any S1c:

| line, f | model | N(ROI) | N(G) | N(V) |
|---|---|---|---|---|
| KK 64.3, f = 0.1 | Gaussian k=0.5 / NEST skew / k=0.75 NEST | 4.4e-3 / 4.7e-6 / 0.085 | 2.4e-3 / 2.3e-6 / 0.046 | 3.8e-5 / 6e-10 / 1.4e-3 |
| KK 64.3, f = 0.2 | Gaussian / NEST skew / mirrored / k=0.75 NEST / k=0.75 Gauss | 0.127 / 2.4e-3 / 0.85 / 1.29 / 4.9 | 0.069 / 1.2e-3 / 0.45 / 0.70 / 2.5 | 2.1e-3 / 1.7e-6 / 0.047 / 0.040 / 0.40 |
| KK 64.3, f = 0.3 | Gaussian / NEST skew / mirrored | 1.95 / 0.29 / 5.2 | 1.06 / 0.15 / 2.7 | 0.060 / 1.0e-3 / 0.36 |
| ¹²⁵I 67.3, f = 0.2, 10⁴ | Gaussian / NEST skew / mirrored / k=0.75 NEST | 1.38 / 0.029 / 10.2 / 18.7 | 1.16 / 0.025 / 8.1 / 15.4 | 0.033 / 4.7e-5 / 0.73 / 0.92 |
| ¹²⁵I 67.3, f = 0.3, 10⁴ | Gaussian / NEST skew | 23 / 3.3 | 19 / 2.8 | 0.98 / 0.022 |

With f = 0.20 (the value that reproduces LZ's 21 ¹²⁴Xe ROI events, P010) and the Gaussian shape, the KK line leaks 0.13 events
into the ROI (all at S1c > 481 phd), 0.07 into the gap and 2×10⁻³ into V; 10⁴ ¹²⁵I decays give 1.4 / 1.2 / 0.03 (P(0 in G) =
31 %). f = 0.3 puts 1–19 events into the empty gap (P(0) ≤ 35 % for KK alone, < 10⁻⁸ for ¹²⁵I at 10⁴) and is excluded, as
P010 found with a cruder model (our V-counts agree with P010's to within ×3). LZ's magenta curve at S1c > 500 (2.5×10⁻³,
+1.5…+5.5 hw) equals the *light-tailed* NEST-skew KK prediction at f = 0.2 (2.4×10⁻³) or the Gaussian at f ≈ 0.1
(4×10⁻³): LZ's EC-line tail is therefore the light one, and the Gaussian f = 0.2 tail would be 50× heavier than what LZ
modelled — still 2×10⁻³ at the event's location.

## 7. What calibration could pin the tail (`results.json: calibration_needed`)

A sample of N ER calibration events at S1c 500–600 with none below a given depth bounds the tail fraction at 90 % CL to
p < 2.3/N. The present ²¹²Pb sample there (≈ 460 points, with wall leakage) bounds p ≳ 5×10⁻³ and shows 1 % of points
below −3 hw. Fraction of β decays landing at S1c 500–600 (allowed β shapes **[recalled: certain]** × our P(S1c | E)):
²¹²Pb 6.1 %, ¹⁴C 10.2 %, ²¹⁴Pb 3.3 %. To reach p = 10⁻⁴ (10⁻⁵) one needs 2.3×10⁴ (2.3×10⁵) events at 500–600 phd, i.e.
3.8×10⁵ (3.8×10⁶) ²¹²Pb decays or 2.3×10⁵ (2.3×10⁶) ¹⁴C decays in the analysed volume, with the ¹⁴C run scheduled away from
neutron calibrations (the ¹³³Xe cut at 500 phd removed exactly this region from the present ¹⁴C sample). That tests the
mirrored/heavy-tail and k ≥ 0.75 hypotheses (ROI fractions 3–100×10⁻⁶) but not the Gaussian k = 0.50 one (2×10⁻⁸, which would
need 10⁸ events): the event's location is beyond any achievable ER-tail calibration, so the "very little leakage" statement is
necessarily model-based. A monoenergetic line near 90 keV_ee in the bulk would be ideal but none of the internal sources
(⁸³ᵐKr 41.5, ¹²⁵I 67.3, ¹³¹ᵐXe 164, ¹²⁹ᵐXe 236 keV) falls at S1c 500–600.

## 8. Results summary

- Literal Table S4 (σ_p N_i) gives σ(N_e) = 303 e at 70 keV, a band 1.85× the drawn 10–90 % lines and 1.3× the ²¹²Pb scatter;
  k = 0.503 (NEST skew) / 0.501 (Gaussian) reproduces the drawn lines (P010's 0.50); nestpy 2.1.1's v2.4.0-style model gives 86 e.
- Drawn-line asymmetry 0.98–0.99 favours a near-Gaussian N_e distribution (|a| ≲ 1); NEST's positive skew (a = 2.8–3.7 at
  45–70 keV) predicts 1.25–1.17 at 300–400 phd, 5σ off; the mirrored skew 0.77.
- β-continuum leakage into the ROI at S1c 500–600 for band-reproducing variants: 4×10⁻¹⁴ (skew×1.5) → 3×10⁻⁹ (NEST) →
  4×10⁻⁵ (Gaussian) → 6×10⁻³ (mirrored); α₁ ±50 %: ±5 %; α₂ ±50 %: ÷80 / ×1.1; k = 0.75: ×10⁶; k = 1: ×10⁸ (excluded).
- LZ's Fig. 5 continuous-ER tail at S1c > 500 (7×10⁻³ in +1…+5.5 hw) is 170× the Gaussian k = 0.5 prediction, ≈ the k = 0.75 or
  mirrored ones, and is cut at +1 hw; its middle-panel floor of (2–8)×10⁻⁴/bin down to −3 hw is a PDF-construction artefact.
- Event region (S1c 500–600, log10 S2c 3.90–4.00): β 7×10⁻⁷ (Gaussian), 5×10⁻⁴ (mirrored), 0.017 (k = 0.75 Gaussian);
  KK f = 0.2: 2×10⁻³ (Gaussian), 0.05 (mirrored); ¹²⁵I 10⁴ decays f = 0.2: 0.03 (Gaussian), 0.7 (mirrored) — the latter with 8
  events in the empty gap.
- Gap constraint: any variant with N(V) ≳ 0.05 puts ≥ 0.5–8 events in G (0 observed); the literal k = 1 Gaussian (0.61 in V) has
  21 ROI events at S1c > 300 and 3.1 in G.

## 9. Validation, robustness, failed approaches

- Engine vs MC: percentiles agree to ≤ 0.0005 dex; energies feeding S1c = 540 ± 10 are 81–100 keV (10–90 %).
- Fig. 5 digitisation recovers the caption total (0.0108 vs 0.0106) and the middle-panel event count (24.6 vs ≈25 points).
- k-calibration insensitive to the shape (0.503 vs 0.501). Medians 0.01–0.015 dex below the drawn ones: does not affect widths or
  leakage fractions measured relative to μ_NR.
- The α₂ sign flip collapses σ_p (band half-width 0.05) and is unphysical; it is listed only to show that the αᵢ act on the width,
  not on the tail.
- First attempt used a 5-node quadrature, a 1-keV grid and a log10 S2c grid to 4.40 (which omitted the band itself): fixed to 3
  nodes, 2 keV and 3.60–5.20; the k-minimiser (bounded scalar) was replaced by a 5-point grid with interpolation for speed.
- The ²¹²Pb blob weighting is approximate; the single-marker area (25 px) is taken from the sparse region; results quoted as
  upper estimates. Grey-point counts near the NR band in the calibration sample are reported without interpretation.
- Recalled/assumed detector inputs (ε = 0.80, 2 % position residual) move the fixed-S1c width by < 3 % (P024) and are common to
  all variants.

## 10. Figures

- `figures/P056_fig1_er_band_variants.png` — ER band medians and 10–90 % lines for k = 0.5 with NEST/Gaussian/mirrored skew and
  for the literal k = 1, the digitised Fig. 4 lines, the Fig. 2 ²¹²Pb points, the NR band, ROI top, gap G, region V, the event.
- `figures/P056_fig2_leakage_vs_S1c.png` — fraction of ERs entering the ROI, below the NR median and below the event's depth vs
  S1c for the variants.
- `figures/P056_fig3_fig5_projection.png` — Fig. 5 bottom panel (digitised light-blue and magenta) against our projected
  β-continuum predictions for five variants and the KK / ¹²⁵I lines at f = 0.2 (Gaussian).
- `figures/P056_fig4_sigma_p_and_shapes.png` — σ_p(x) with the αᵢ variants, nestpy's effective σ_p and the calibrated k σ_p;
  the N_e pdf shapes at 69.6 keV with the event's N_e and the ROI edge.

## 11. References

1. LZ Collaboration, arXiv:2609.02823 (2026): Discussion, Fig. 2, 4, 5, Table I, Tables S3–S4, detector-response supplement.
2. J. Aalbers et al. (LZ), Phys. Rev. D 112, 012024 (2025), arXiv:2503.05679 (enhanced recombination after inner-shell vacancies).
3. M. Szydagis et al., "A review of NEST models for liquid xenon", Front. Detect. Sci. Technol. 2, 1480975 (2024), arXiv:2211.10726.
4. NEST Collaboration, NEST v2.4.0 (2023) / v2.4.5 (2026), as cited by LZ; nestpy 2.1.1.
5. A. Azzalini, Scand. J. Statist. 12, 171 (1985) (skew-normal distribution).
6. J. Aalbers et al. (LZ), Phys. Rev. Lett. 131, 041002 (2023) (first LZ results; ER/NR calibration description).
7. Corpus: P004, P009, P010, P024, P041; dossier 00.

## 12. Tools and provenance (mirrors `output/provenance/P056.json`)

- Agent tools: Read (PAPER_GUIDE, dossier, ledger, P010/P024/P009/P041/P004 papers, P010 details and script, P024 script, P004
  script part, lzcommon parts, tex lines 84–130/150–270/296–330/586–700, Fig2/Fig4/Fig5 PNGs, P010's Fig. 5 crop), Bash (ledger
  listing, tex grep, directory listings, nestpy probes ×3, colour probes, Fig. 5 tick probe, timing test, three script runs, log
  and CSV extraction), Write (script, details, JSON, paper), Edit (script fixes ×16), TaskStop ×1, ToolSearch ×2.
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (special.ndtr/erf, stats.skew, optimize.brentq, ndimage.label/
  center_of_mass/sum); pandas 3.0.5; matplotlib 3.11.2 (Agg; image.imread for digitisation); nestpy 2.1.1 (LZ_WS2024,
  GetYields, GetQuanta with nr_er_width_parameters, RandomGen.set_seed); common/lzcommon.py (LZ, NEST_ER_LZ, NEST_ER_FLUCT_LZ,
  nest_er_yields, DRIFT_FIELD_VCM).
- Recalled knowledge: NEST GetQuanta Var(N_e) form and σ_p × N_i combination (likely); NEST ER skewness switched off above
  N_q = 10⁴ (likely, confirmed numerically); extraction efficiency 0.80 (uncertain); Fig. 5 colour/axis conventions from the
  figure itself; allowed β-spectrum shape with Fermi function (certain); ²¹²Pb/¹⁴C/²¹⁴Pb endpoints 570/156.5/1019 keV (likely).
- Datasets: none. Data requests: none. WimPyDD files: none.
