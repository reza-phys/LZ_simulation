# P010 — ER leakage: the recombination fraction required and the ¹²⁴Xe/¹²⁵I double-vacancy hypothesis

Research record (simulated date 2026-09-04). Script: `output/code/P010_er_leakage.py`; machine-readable results:
`output/work/P010/P010_results.json`; tables: `line_tail_probabilities.csv`, `xe124_modes_in_ROI.csv`, `I125_f_N_scan.csv`,
`gap_likelihood.csv`, `er_band_fixed_S1c.csv`; console log `run_log.txt` (two consecutive runs are byte-identical).
All LZ inputs are from arXiv:2609.02823 via `lzcommon.LZ`, Tables I, S3, S4, S5, Figs 4 and 5. Recalled inputs are flagged **[recalled: reliability]**.

## 1. Motivation and framework

The LZ paper offers one explicit non-NR reading of the 248 keV event: an electron recoil of ≈64–71 keV whose charge was suppressed
by the enhanced recombination that follows inner-shell vacancies (¹²⁴Xe 2νECEC KK at 64.3 keV; ¹²⁵I K-capture at 67.3 keV),
and it states that "there is a systematic uncertainty in projecting into the tail of the distribution that we do not include",
while noting "the lack of events between the event of interest and the bottom of the ER band". We quantify (i) the recombination
fraction an ER would need, (ii) how far that is from the mean in units of LZ's own fluctuation model, (iii) how many candidate
decays existed, and (iv) whether any monotonic tail heavy enough to produce the event is compatible with the empty region between it
and the ER band.

Notation: g₁ = 0.110 phd/photon, g₂ = 34.5 phd/electron; N_ph = S1c/g₁, N_e = S2c/g₂, N_q = N_ph + N_e; ions N_i = N_q/(1+α) with
α the exciton-to-ion ratio; recombination fraction r = 1 − N_e/N_i. The WS ROI upper edge S2c < 10^4.15 phd corresponds to
N_e < 409.

## 2. Event quanta and ER-equivalent energy (script §1)

| quantity | value | source |
|---|---|---|
| N_ph | 4910 | 540.1/0.110 |
| N_e | 268.6 | 9268/34.5 |
| N_q | 5179 ± 90 | g₁ ± 0.002, g₂ ± 1.1 propagated |
| E_ee (W = 13.7 eV) | 70.95 keV | W [recalled: likely, NEST literature value] |
| E_ee (W = 13.5 eV) | 69.91 keV | |
| E_ee (W = 13.44 eV, nestpy's own W at 2.9 g/cm³) | **69.6 keV** | used as the reference ER energy |
| W that would give 64.3 keV / 67.3 keV | 12.42 / 13.00 eV | neither is a plausible LXe W |

Energy consistency with the two lines, using quanta conservation (S1 photon-counting resolution √(g₁(1−g₁)N_ph)/g₁ plus a 3 % S1c
position-correction term (assumption), N_e counting, Fano σ²(N_q) = 1.01·N_q measured from nestpy): σ(N_q) ≈ 245–253. The event's
N_q is 1.6σ above the ¹²⁴Xe KK line (8.2 % high) and 0.7σ above the ¹²⁵I K line (3.4 % high); with the g₁,g₂ uncertainty,
1.5σ and 0.6σ. So ¹²⁵I is energetically comfortable, ¹²⁴Xe KK mildly strained; the "64 keV" reading requires W = 12.4 eV.

## 3. Required recombination fraction (script §1)

NEST ER mean yields with the LZ Table S3 parameters (`lz.nest_er_yields(E, params=lz.NEST_ER_LZ)`, 96.5 V/cm):

| E (keV) | N_ph | N_e | N_q | N_i (α=0.182) | r_mean | S1c | log₁₀S2c |
|---|---|---|---|---|---|---|---|
| 64.3 | 3666 | 1118 | 4784 | 4046 | 0.724 | 403 | 4.586 |
| 67.3 | 3829 | 1179 | 5007 | 4235 | 0.722 | 421 | 4.609 |
| 69.6 | 3953 | 1226 | 5179 | 4380 | 0.720 | 435 | 4.626 |
| 70.9 | 4021 | 1254 | 5275 | 4461 | 0.719 | 442 | 4.636 |

α = 0.067366 + 0.039693·ρ = 0.182 at ρ = 2.9 g/cm³ **[recalled: likely — NEST v2 ER exciton-to-ion ratio]**.
Required: N_i = 5179/1.182 = 4379, **r_req = 1 − 268.6/4379 = 0.939** (0.945 for α = 0.06, 0.938 for α = 0.20).
β-ER mean at the same energy: r = 0.720 → **Δr = 0.219**; only 269 of a mean 1226 electrons survive (deficit 958 e, 78 %).

## 4. Fluctuation model, band width, and what LZ's "6.7σ" means (script §2)

### 4.1 Table S4 σ_p and the raw width
σ_p(x) = Σᵢ Aᵢ[1+erf(αᵢ(x−μᵢ)/(√2σᵢ))]exp(−(x−μᵢ)²/2σᵢ²), x = log₁₀N_q, Table S4 parameters. At N_q = 5179: term 1 = 7×10⁻⁵,
term 2 = 0.0681, **σ_p = 0.0682**. Taking NEST's GetQuanta form Var(N_e) = r(1−r)N_i + σ_p²N_i² **[recalled: likely]** gives the
raw σ(N_e) = √(30² + 299²) = **300 e** at 69.6 keV, so the deficit would be only 3.2σ in linear N_e — a heavy tail that the data
below contradict.

### 4.2 Digitised Fig. 4 bands
Axis calibration from the PNG frame and tick pixels (x: col = 143.5 + 1.30·S1c; y: row = 35 + 383·(5 − log₁₀S2c); verified on
the 200/400/500/700/800 phd and 4.0/4.5 ticks). Band lines found as dark-blue / red pixel clusters in a 5-px column:

| S1c | ER 90 % | ER median | ER 10 % | half-width (10–90) | NR lines |
|---|---|---|---|---|---|
| 200 | 4.393 | 4.317 | 4.240 | 0.076 | 3.922 / 3.889 / 3.858 |
| 300 | 4.554 | 4.465 | 4.373 | 0.090 | 3.974 / 3.943 / 3.910 |
| 400 | 4.698 | 4.597 | 4.493 | 0.102 | 4.007 / 3.978 / 3.950 |
| 540 | 4.910 | 4.791 | 4.668 | 0.121 | 4.044 / 4.016 / 3.987 |

Event (3.967) vs ER median at 540: 0.824 dex = **6.82 half-widths** — the paper's 6.7σ is reproduced if "σ" denotes the
10–90 % half-width (1.28 Gaussian σ). In Gaussian σ (0.094 dex) the event is 8.7σ below; likewise the NR distance is 1.7
half-widths (paper 1.5) = 2.2 Gaussian σ. Digitisation uncertainty ±0.005 dex (±0.2 in these units).

### 4.3 Fixed-S1c vs fixed-E widths and the calibration factor k
A flat-spectrum MC (N_q Fano 1.01, binomial N_ex/N_i, N_e drawn as skew-normal with the nestpy shape a = 2.10 measured from 40 000
nestpy `GetQuanta` draws at 64.3 keV (sample skewness +0.47), S1c binomial + 3 %, S2c with √(1/N_e + 0.03²)) shows that at
fixed S1c the band is 1.57× wider than at fixed energy (1 + (dN_e/dE)/(dN_ph/dE) at 88 keV) because S1c = 540 mixes energies
81–100 keV. With the raw σ_p the MC 10–90 half-widths are 0.166/0.199/0.231 dex at 300/400/540 — twice the drawn band. Scaling
σ(N_e) by k reproduces the digitised band for **k = 0.50** (MC 0.088/0.103/0.125 vs 0.090/0.102/0.121; MC median 4.773 vs 4.791).
We therefore do not know how LZ's NEST v2.4.5 combines σ_p with N_i; we adopt the calibrated width (k = 0.50) as LZ's model and carry
k = 1 as a pessimistic upper bound. Independent check: the k = 1 continuum leakage into the ROI at 250 < S1c < 500 would be 9–58
events (tail-shape dependent) against LZ's ~2.5 modelled continuum events and 23 observed in total (mostly EC/⁸³ᵐKr), so k = 1 is
excluded by the data themselves; k = 0.5 gives 0.0–0.5.

Calibrated σ(N_e) at 69.6 keV = **150 e → the 958-electron deficit is 6.4σ in linear N_e at fixed energy**, consistent with the 6.8
half-width / 8.7 Gaussian-σ log-space statement.

## 5. Line model with recombination enhancement f and four tail shapes (script §3)

For a line at E (N_q from Table S3), mean N_e reduced by f (N_ph increased correspondingly; total quanta conserved), σ(N_e) from §4
with k = 0.50, and four shapes with identical mean and σ: `nest_skew` (skew-normal a = +2.10 as in nestpy — light low-N_e tail),
`gauss`, `exp_tail` (Gaussian core, exponential lower tail matched in value and slope at −2σ), `neg_skew` (skew-normal a = −2.60,
the Table S4 α₂ used only as a heavy-lower-tail toy; note that the Table S4 αᵢ shape σ_p(x), not the N_e distribution).
Regions: **V** (event-like) S1c 500–600, log₁₀S2c 3.90–4.00 (N_e 230–290); **G** (gap, 0 events) S1c 475–600, 4.00–4.15
(N_e 290–409). P(S1c range | N_e) from N_ph = N_q − N_e with the S1 resolution above.

Quanta conservation fixes where a leaked line lands: a 64.3 keV ER with N_e = 269 has S1c = 497 phd; a 67.3 keV ER, 521 phd.
Hence **every KK/K-capture decay that enters the ROI (N_e < 409) appears at S1c > 481 (¹²⁴Xe) or > 506 (¹²⁵I)** — i.e. in
Fig. 5's bottom panel, where LZ's IC+EC model integrates to ≈0.002 events.

Calibrated tail probabilities per decay (`line_tail_probabilities.csv`):

| line | shape | f | N_e mean | σ | z_event | P(V) | P(G) | P(G)/P(V) |
|---|---|---|---|---|---|---|---|---|
| ¹²⁴Xe KK | nest_skew | 0 | 1118 | 135 | 6.3 | 2×10⁻¹⁸ | 2×10⁻¹³ | — |
| | nest_skew | 0.2 | 895 | 135 | 4.6 | 3.9×10⁻¹⁰ | 9.5×10⁻⁷ | 2400 |
| | nest_skew | 0.4 | 671 | 135 | 3.0 | 9.0×10⁻⁵ | 7.6×10⁻³ | 84 |
| | gauss | 0 | 1118 | 135 | 6.3 | 1.8×10⁻¹⁰ | 4.9×10⁻⁸ | 260 |
| | gauss | 0.2 | 895 | 135 | 4.6 | 1.5×10⁻⁶ | 1.0×10⁻⁴ | 68 |
| | gauss | 0.4 | 671 | 135 | 3.0 | 8.3×10⁻⁴ | 1.6×10⁻² | 19 |
| ¹²⁵I K | nest_skew | 0.2 | 943 | 143 | 4.7 | 3.5×10⁻¹⁰ | 5.4×10⁻⁷ | 1560 |
| | gauss | 0.2 | 943 | 143 | 4.7 | 1.8×10⁻⁶ | 8.8×10⁻⁵ | 48 |
| | gauss | 0.4 | 707 | 143 | 3.1 | 1.1×10⁻³ | 1.5×10⁻² | 14 |

With the raw width (k = 1) all probabilities rise by 4–8 orders of magnitude at f = 0 (e.g. KK gauss P(V) = 2.7×10⁻⁴), but k = 1
is excluded (§4.3).

Continuum β/Compton ERs: from Table I the continuum components in the ROI sum to 1655 events; our ROI acceptance integral for a
flat spectrum is 19.4 keV, i.e. **≈85 events per keV_ee** in 2.84 t·yr (±50 % when extrapolated to 60–80 keV). Integrating 56–84 keV:
N(V) = 2×10⁻⁷ (gauss), 3×10⁻³ (exp_tail), 7×10⁻⁴ (neg_skew); N(G) = 4×10⁻⁵ / 0.028 / 0.015.

## 6. Candidate populations (script §4)

**¹²⁴Xe.** Abundance 0.095 % (periodictable), M = 131.29 g/mol, 4.71 t → N = 2.05×10²⁵ atoms; T½ = 1.1×10²² yr
**[recalled: likely — XENONnT 2022 / LZ 2024; XENON1T 2019 gave 1.8×10²²]**; λ = 6.3×10⁻²³ yr⁻¹; 220 d → **779 decays**
(476 for 1.8×10²² yr). Branching KK 0.724, KL 0.200, KM+KN 0.056, LL+LM+MM 0.020 **[recalled: uncertain, ±0.03]** →
564 KK, 156 KL, 44 KM/KN, 16 low-energy. Mode energies 64.3, 36.7, 33.2, 9.8 keV (Te binding K 31.8, L 4.9, M 1.0 keV
**[recalled: likely]**).

Where the modes sit and how many enter the ROI (calibrated width, nest_skew):

| mode | E | f | N_e mean | S1c mean | S1c needed for ROI | P(ROI) | N in ROI |
|---|---|---|---|---|---|---|---|
| KK | 64.3 | 0 / 0.2 / 0.3 | 1118 / 895 / 783 | 403 / 428 / 440 | > 481 | 3×10⁻¹³ / 1.5×10⁻⁶ / 3×10⁻⁴ | 0 / 0.0009 / 0.17 |
| KL | 36.7 | 0 / 0.2 / 0.3 | 658 / 527 / 461 | 228 / 242 / 250 | > 255 | 1×10⁻⁷ / 0.017 / 0.21 | 0 / 2.6 / 33 |
| KM/KN | 33.2 | 0 / 0.2 / 0.3 | 606 / 485 / 424 | 205 / 218 / 225 | > 227 | 3×10⁻⁶ / 0.070 / 0.43 | 0 / 3.1 / 19 |
| LL/LM/MM | 9.8 | any | 243–170 | 54–62 | > 35 | 1.0 | 15.6 |

Sum vs LZ's 21.0 ± 6.3: f = 0 → 15.6; 0.1 → 15.7; **0.2 → 21.3**; 0.3 → 67.6. A common suppression **f = 0.20 (1σ: 0–0.22)**
reproduces LZ's ROI count, with the events at S1c ≈ 50–60 (low-energy modes) and 255–330 phd (KL/KM leakage) — exactly where
Fig. 4 shows the ROI-edge cluster — and essentially none from the KK line. The premise that "21 events in the ROI" signals a heavily
leaking KK population is therefore not supported: for the KK line alone to supply 21 events, P(N_e<409) = 0.037 would be needed,
i.e. f = 0.44 (k = 0.5) or 0.25 (k = 1), and those events would populate S1c > 481 phd inside the ROI, where Fig. 4 shows none.
The consistency also matches the qualitative statement in Fig. 4's caption that the EC features sit "slightly below" the β band
(f = 0.2 is a 0.10 dex shift, 0.8 half-widths; f = 0.4 would be 0.22 dex, outside the 10–90 % band) and the order of the
charge-yield reductions reported in LZ's inner-shell paper **[recalled: uncertain, of order 10–20 %]**.

**¹²⁵I.** Half-lives from `radioactivedecay` (ICRP-107): ¹²⁵I 59.4 d, ¹²⁵Xe 16.9 h, ¹²⁷Xe 36.4 d, ¹³¹ᵐXe 11.84 d, ¹²⁹ᵐXe 8.88 d,
¹³³Xe 5.24 d, ⁸³ᵐKr 1.83 h. With t½ᵉᶠᶠ = 3.6 d, the fraction of ¹²⁵I atoms decaying in the LXe is λ_dec/λ_eff = 3.6/59.4 = 6.1 %.
Bateman profile after a prompt ¹²⁵Xe production: peak at 2.1 d, rate at +8 d = 0.40 of peak, 27 % of all ¹²⁵I decays after day 8,
10 % in days 7–9; 97 % within 20 d, so three calibrations confine ¹²⁵I to ≈27 % of the livetime. ¹²⁵Xe itself is gone
(3.8×10⁻⁴ left after 8 d); ¹²⁷Xe is 86 % present. The absolute ¹²⁵I population cannot be derived from the paper: the 67.3 keV
K-line can enter the ROI only at S1c > 506, so LZ's 8.9 ROI events must be L/M-capture (40.4 keV, ≈20 % **[recalled: likely]**)
leakage at S1c > 280 phd; inverting gives N(¹²⁵I decays) ≈ 5×10² (f = 0.3) to 1.2×10⁴ (f = 0.2) — an order-of-magnitude range,
so we scan N = 10²–10⁴.

¹²⁵I K-line, Gaussian tail, expected event-like events N(V) and P(0 in G) (`I125_f_N_scan.csv`):

| f | N = 100 | 1000 | 10 000 |
|---|---|---|---|
| 0.20 | 1.8×10⁻⁴ (0.99) | 1.8×10⁻³ (0.92) | 0.018 (0.42) |
| 0.25 | 1.2×10⁻³ (0.96) | 0.012 (0.67) | 0.12 (0.017) |
| 0.30 | 6×10⁻³ (0.85) | 0.06 (0.20) | 0.6 (<10⁻³) |
| 0.40 | 0.11 (0.21) | 1.1 (<10⁻³) | 11 (0) |

Other candidates: ⁸³ᵐKr (41.5 keV; 17.1 in ROI) has N_q ≈ 3090 and cannot reach S1c = 540 (N_ph ≤ 3090 → S1c ≤ 340);
¹³¹ᵐXe (164 keV), ¹²⁹ᵐXe (236 keV), ¹²⁷Xe (203+33 keV) and ¹²⁵Xe (188 or 55+33 keV) deposits are far from 70 keV; ¹³³Xe
(β + 81 keV γ) gives ≥81 keV. None is an alternative for a 70 keV ER.

## 7. The gap argument (script §5)

**LZ's own tail.** Reading Fig. 5's bottom panel (S1c > 500) from the PNG crop: the ER components (continuous ERs + "internal γ +
IC + EC") form a plateau of 3×10⁻⁴–2.3×10⁻³ events per 0.5σ_NR bin from +5.5 down to +1σ_NR (sum 0.0089; IC+EC 0.0022) and then
drop off-scale (< 10⁻⁶) below +1σ_NR; the remaining floor (accidentals 2.5×10⁻⁵, MSSI 1.5×10⁻⁵, NR 5×10⁻⁶ per bin) gives
0.0102 in total versus the quoted 0.0106 ± 0.0008. The abrupt edge at +1σ_NR is the "systematic in projecting into the tail":
the PDFs are MC-statistics limited there. A flat extrapolation of the low-side plateau (7×10⁻⁴/bin) across the 2.5σ_NR
(0.07 dex) gap to ±1σ_NR around the event gives **μ_V = 2.8×10⁻³ (P(≥1) = 0.28 %)**, 18× the modelled floor of 1.6×10⁻⁴ but still
≪ 1.

**Shape test.** Given exactly one ER-tail event in the ROI at S1c > 500 (beyond the ROI edge, 6.8 Gaussian-σ below the ER
median), P(it lies at ≥ 8.7σ rather than in 6.8–8.5σ) = 2×10⁻⁷ (Gaussian in log-space), 0.02 (exponential, e-fold 0.5σ), 0.14
(e-fold 1σ), 0.34 (flat). Only a tail that is flat or nearly flat over 2σ_ER makes the location unremarkable, and such a tail is
exactly what would populate G.

**Normalisation-free likelihood.** For a leaking line with shape ratio ρ = P(G)/P(V), the probability of "1 in V, 0 in G" is
μ_V e^{−μ_V(1+ρ)}, maximal at μ_V = 1/(1+ρ). Compared with the modelled floor (μ = 1.6×10⁻⁴), the best-tuned ER tail is favoured by
at most ×149 (Gaussian, f = 0.4, μ_V = 0.065) and by ≤ ×7.5 for f ≤ 0.3 with the NEST skew — a post-hoc one-parameter fit that
still leaves the event a ≤6 % occurrence. With the physically motivated normalisations (564 KK decays; ¹²⁵I 10²–10⁴), any scenario
predicting N(V) ≳ 0.1 predicts N(G) ≳ 1–9 with P(0 in G) ≤ 0.2–5 % (table above; `gap_likelihood.csv`).

## 8. Results summary

- r_req = 0.939 (0.938–0.945) vs β mean 0.720 at 69.6 keV; Δr = 0.22; 958 of 1226 electrons missing.
- Calibrated σ(N_e) = 150 e at fixed energy → 6.4σ deficit; in log-space at fixed S1c the event is 6.8 band half-widths = 8.7
  Gaussian σ below the ER median (LZ's "σ" = 10–90 % half-width).
- Raw Table S4 σ_p with Var = σ_p²N_i² would give a band 2× wider than drawn and 9–58 continuum leakage events at 250–500 phd —
  excluded by the data; k = 0.50 reproduces the band.
- ¹²⁴Xe: 779 decays (564 KK). LZ's 21.0 ± 6.3 ROI events are reproduced with a common charge suppression f = 0.20 (0–0.22) and
  come from the low-energy and KL/KM modes; KK contributes ≤0.001 (f = 0.2) to 0.17 (f = 0.3).
- Event-like leakage: KK line 2×10⁻⁷ (f = 0.2, skew) to 0.47 (f = 0.4, Gauss); ¹²⁵I 10⁻⁴–0.02 for f ≤ 0.2 even with 10⁴ decays;
  continuum ≤3×10⁻³; LZ-plateau extrapolation 2.8×10⁻³.
- Any tail giving ≳0.1 event-like events puts ≥1–9 events in the empty gap (P₀ ≤ 5 %); the energy requires N_q 1.6σ above the
  ¹²⁴Xe KK line (0.7σ for ¹²⁵I); the event occurred when the ¹²⁵I rate was 40 % of its post-AmBe peak (27 % of the livetime is
  within 20 d of a calibration).

## 9. Validation, robustness, failed approaches

- ER mean yields checked against `lzcommon` (N_e = 1254 at 70.9 keV vs the dossier's 1235 with W = 13.7 — the difference is the
  energy grid; both give r ≈ 0.72). NR band from `nest_nr_yields` matches the digitised red lines to ±0.01 dex.
- The paper's 6.7σ/1.5σ statements are recovered to ±0.2 with the half-width convention; the Gaussian-σ convention fails (8.7/2.2).
- Digitised black-point counting is unreliable where points overlap (250–330 phd); we use it only to confirm that S1c > 350 phd
  contains ≤1 ROI event other than the event of interest (one blob at ≈(400, 4.1) may be a marker or a point), and the gap G is empty.
- First attempt used the raw σ_p width directly; it over-predicted the drawn band by ×2 and the observed ROI leakage by ×4–20, so it
  was demoted to an upper bound. The Table S4 αᵢ were briefly considered as tail-skewness parameters — they are not; the negative-skew
  toy is labelled as such.
- k and a are MC-fitted; run-to-run scatter (nestpy RNG now seeded) was k = 0.48–0.50, changing tail probabilities by ≤×3 at f = 0.4.
- Dependence on α (0.06–0.20): r_req 0.945–0.938; on T½(¹²⁴Xe) 1.1→1.8×10²² yr: all ¹²⁴Xe counts ×0.61; on branching ±0.03:
  KL/KM leakage ±15 %, the fitted f moves by ≤0.02.

## 10. What LZ could check

(1) Publish the EC-line model tails (or MC statistics) below +1σ_NR at S1c > 250; (2) the S2 pulse width of the event vs the ER
and NR calibration populations at the same drift time (an ER with r = 0.94 has no distinguishing S2 width, so a normal width is
neutral, but an anomalous one would point to charge loss); (3) the ¹²⁵I/¹²⁵Xe/¹²⁷Xe rates in the ±24 h around the event
(the ¹²⁵I rate should be ≈40 % of its post-AmBe peak); (4) the number of ROI events at S1c > 475 phd and log S2c 4.00–4.15 in the
post-2024 data — a leaking KK/K-capture line predicts ≥10× more events there than at the event's location.

## 11. Figures

- `figures/P010_fig1_bands_event_lines.png` — (S1c, log₁₀S2c) plane: NEST-LZ ER mean with the calibrated ±1σ(N_e) band, digitised
  Fig. 4 ER band (10–90 %), NR mean, ROI, the event with r_req, the constant-N_q loci of the 64.3 and 67.3 keV lines with markers
  at f = 0–0.4, and the regions V (event-like) and G (gap).
- `figures/P010_fig2_NV_NG_vs_enhancement.png` — expected event-like (solid) and gap (dashed) counts vs f for the KK line (564 decays)
  and the ¹²⁵I K line (500 decays), four tail shapes, calibrated width; the modelled non-ER floor in V is the dotted line.
- `figures/P010_fig3_I125_timing.png` — ¹²⁵I decay-rate profile (Bateman, 16.9 h feed, 3.6 d effective removal), ¹²⁵Xe and ¹²⁷Xe vs
  days after AmBe; the event at +8 d.
- `figures/fig5_bottom_crop.png`, `fig5_middle_crop.png` — crops of the paper's Fig. 5 used for the plateau read-off.

## 12. References

1. LZ Collaboration, arXiv:2609.02823 (the paper), Tables I, S3–S5, Figs 4, 5, Discussion.
2. J. Aalbers et al. (LZ), "Measurements and models of enhanced recombination following inner-shell vacancies in liquid xenon",
   Phys. Rev. D 112, 012024 (2025), arXiv:2503.05679.
3. M. Szydagis et al., NEST v2 (Noble Element Simulation Technique), J. Instrum. 2023 / nestpy 2.1.1.
4. E. Aprile et al. (XENON), "Observation of two-neutrino double electron capture in ¹²⁴Xe with XENON1T", Nature 568, 532 (2019).
5. E. Aprile et al. (XENON), "Double-weak decays of ¹²⁴Xe and ¹³⁶Xe in the XENON1T and XENONnT experiments", Phys. Rev. C 106,
   024328 (2022).
6. J. Aalbers et al. (LZ), "Dark matter search results from 4.2 tonne-years of exposure of the LUX-ZEPLIN (LZ) experiment",
   Phys. Rev. Lett. (2025), arXiv:2410.17036.
7. ICRP Publication 107 decay data via `radioactivedecay` 0.6.1.
8. `output/00_evidence_dossier.md` (§2, §5 C, §6.3).

## 13. Tools and provenance (mirrors `output/provenance/P010.json`)

- Agent tools: Read (PAPER_GUIDE.md; 00_evidence_dossier.md; LZ tex lines 96–175, 196–310, 580–660; lzcommon.py; Fig5 and Fig4
  PNGs; two Fig. 5 crops; own figures ×4), Bash (grep of the tex; ls/ledger/versions; nestpy/radioactivedecay/periodictable probe;
  Fig. 4 frame and tick detection; Fig. 5 cropping; band-width debug; script runs ×5 incl. reproducibility check; nestpy seed API check),
  Write (script; details.md; P010.json; P010.md), Edit (×7 on the script).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (stats.skewnorm, stats.norm, special.erf, optimize.brentq, integrate.quad,
  ndimage.label); pandas 3.0.5; matplotlib 3.11.2; nestpy 2.1.1 (NESTcalc.GetYields, GetQuanta, RandomGen.set_seed,
  detectors.LZ_WS2024); radioactivedecay 0.6.1 (Nuclide.half_life); periodictable 2.1.0 (Xe[124].abundance, Xe.mass);
  Pillow 12.3.0 (image loading); common/lzcommon.py (LZ, NEST_ER_LZ, NEST_ER_FLUCT_LZ, nest_er_yields, nest_nr_yields, DRIFT_FIELD_VCM).
- Script: `output/code/P010_er_leakage.py`, run as `.venv/bin/python output/code/P010_er_leakage.py` from the root (≈7 s).
- Recalled knowledge: 9 items (see JSON). Datasets: none. Data requests: none. WimPyDD files: none.
