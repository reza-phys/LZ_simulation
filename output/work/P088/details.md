# P088 — Decisive-test timeline: research record

Simulated date 2026-09-16. Experimental-strategy / projections group. Category PROJ, hep-ex.

## 1. Motivation and framework

Fourteen corpus papers have produced, piece by piece, everything needed to say *when* the LZ 248 keV event will be
judged: signal rates per tonne-year in every relevant window (P050, P069), the acceptance of LZ's 600 phd edge and of
an extended 1000 phd edge (P038), the background above 200 keV (P016, P050, P061), the events required to separate
spectra (P057, P050), to see the June clustering (P034), to read a second target (P046) or an isotopic pair (P047),
the on-disk exposure ledger (P069, P020, P059) and a prior-sensitivity analysis of P(DM) (P061). What is missing is a
single calendar. This paper builds it. We deliberately recompute **no physics**: every rate, acceptance and event
requirement is taken from the corpus tables listed in §2, and only the counting statistics (Asimov significances,
exact-Poisson first-crossing toys, Gamma–Poisson Bayes factors, posterior updates) are computed here.

Hypotheses. **A**: inelastic isoscalar O₁, m_χ = 1 TeV, δ = 300/350/366/380 keV at LZ's best-fit rate (1.0 event per
2.84 t·yr in LZ's ROI). **B**: elastic L10 (q⁴ spin operator) at the best fit. **C**: background/one-off — the event
has no successor population (a steady unknown background is discussed separately: P001/P069 show that counting cannot
separate it from DM, P(DM) ≤ 0.09). Rate band: ×0.30–×2.36 of the best fit (LZ Table I 68 %, P050/P069 convention);
Poisson–Gamma marginalisation with Gamma(2,1) (flat prior after one event) or Gamma(1.5,1) (Jeffreys).

Questions. (i) For each experiment and window, at what calendar date is the expected exposure enough for a 3σ/5σ
discrimination of A/B against C by counting (new data alone, and combined with LZ's event)? (ii) What is the
probability that the question is *settled* (Bayes factor > 100 either way) by end-2027/2028/2030? (iii) If C is true,
by when do continued nulls push P(DM) below 10⁻³ from P061's median 0.015? (iv) When can A be separated from B by
shape, timing, tungsten or isotopes?

## 2. Inputs (all with source)

### 2.1 Exposure ledger (t·yr) — P069 Table 1 / `P069_exposure_ledger.csv`, P020 for LZ
| dataset | t·yr | dates | status | source / reliability |
|---|---|---|---|---|
| LZ SR1 | 0.77 | 2021.98–2022.35 | analysed ≲70 keV; HE untouched | P069/P059 (recalled, likely) |
| LZ SR3 (WS2024) | 2.84 | 2023.23–2024.25 | analysed to 270 keV (the result) | paper |
| LZ untouched | 6.76 | 2024.25–2026.67 | untouched | P020 (884 d × 0.593 × 4.71 t) |
| XENONnT SR0+SR1 | 3.1 | 2021.5–2023.6 | analysed ≲60 keV; HE untouched | recalled, certain (exposure) |
| XENONnT SR2+ | 3.0 (2–5) | 2023.6–2026.67 | untouched | recalled, uncertain |
| PandaX-4T Run0+1 | 1.54 | 2020.9–2022.4 | analysed ≲100 keV | recalled, certain |
| PandaX-4T Run2/3 | 2.5 (1.5–3.5) | 2023.9–2026.67 | untouched | recalled, uncertain |
| LUX, PandaX-II, XENON1T | 1.45 | archival | windows end below the signal (P059) | excluded from our ledger |

Future accrual (t·yr per calendar year): LZ 4.71 × 0.593 = 2.79 (P020), XENONnT 4.0 × 0.6 = 2.4, PandaX-4T 2.7 × 0.6 =
1.62 (P050/P069 assumptions, recalled), all assumed to run until 2032.0; XLZD 60 t at 48 t·yr/yr from 2032.0 with a
400 keV ROI (P050; recalled, uncertain). On disk at 2026.71: LZ 10.48, XENONnT 6.20, PandaX-4T 4.10, total 20.78 t·yr;
never examined above 200 keV: 20.78 − 2.84 = 17.94 t·yr (P069 quoted 19.1 including the 1.45 t·yr of archival data).
By end-2027/2028/2030: 29.6/36.4/50.0 t·yr (present generation); world 104.8 t·yr by end-2032.

Every "date" below is the date at which the *needed exposure is on disk* (P069/P050 convention). Verdicts follow the
reanalysis: P069 puts the earliest pre-registered joint publication at mid-2027; we quote ≈1 yr latency for future data.

### 2.2 Signal rates per t·yr at LZ's best fit — `P069_normalisation.csv` (reproduces P050 ≤ 0.3 %)
| model | LZ ROI (E50 270) | 200–270 keV | 400 keV ROI | 200–400 keV | A_600 (P038) | A_400 (P050) |
|---|---|---|---|---|---|---|
| L10 | 0.3521 | 0.1205 | 0.4468 | 0.2196 | 0.42 | 0.54 |
| δ = 300 | 0.3521 | 0.0628 | 0.4012 | 0.1123 | 0.81 | 0.93 |
| δ = 350 | 0.3521 | 0.1780 | 0.6875 | 0.5159 | 0.44 | 0.87 |
| δ = 366 | 0.3521 | 0.2737 | 1.7029 | 1.6344 | 0.17 | 0.83 |
| δ = 380 | 0.3521 | 0.2783 | 10.497 | 10.494 | 0.028 | 0.83 |

The "600 phd" scenario counts 200–270 keV (P069's clean window, b = 5.7×10⁻⁴ per 2.84 t·yr, P016); the "1000 phd"
scenario counts 200–400 keV using P050's E50 = 400 keV proxy for P038's 1000 phd edge (E50 = 423 keV), with
b = 2.74×10⁻³ per 2.84 t·yr (LZ-like) or 1.6×10⁻³ (60 t) above 200 keV (P050). Whole-ROI rates are used for the
shape and timing requirements (P057 and P034 count every event in the window). Other TPCs are assumed LZ-like in
acceptance (P069 assumption). Background brackets: 2×10⁻⁴ (P001), whole panel 0.0106 (LZ), P061's steady unmodelled
L_acc = 9.3×10⁻³ per 2.84 t·yr (LZ-specific) — these change Z by ≤ 0.4σ at N = 3 (P069) and are not re-scanned here.

### 2.3 Event requirements from the corpus
- Shape, L10 vs inelastic-366 at 3σ (P057 toys): 2 events if L10 is true, 7 if inelastic (LZ edge); 2/5 (1000 phd). P050 (400 keV window): 4/6.
- June clustering vs a time-flat population (P034, LZ-like window): N_3σ = 96.5/11.5/6.5/5.3, N_5σ = 268/32/17.6/14.1 at δ = 300/350/366/380 keV; with a 400 keV ROI the exposure shrinks ×2.8 (366) and ×21 (380) (P034); for 300/350 we keep N unchanged with the 400 keV rate (approximation).
- LZ's 600–1000 phd sideband (P038, annual halo): 0.28 (L10, from the rate difference) / 0.155 / 1.06 / 3.81 / 30.6 events per LZ-ROI event at δ = 300/350/366/380; added NR-band MSSI 0.0027 per 2.84 t·yr.
- CaWO₄ (P046, W band 110–1300 keV, ²⁰⁶Pb-safe): 2.60/36.95/279/5622 events per t·yr(compound) at δ = 300/350/366/380, L10 6.3×10⁻⁵; 3σ vs L10 needs 215/17/2.1/0.09 kg·yr. Background: 0.01 (baseline) and 0.1 per kg·yr (recalled placeholder, uncertain). Schedules: 10 kg·yr over 2027–2030 (CRESST-scale), 100 kg·yr 2029–2034, 1 t·yr 2030–2035 (hypothetical).
- ¹³⁶Xe-enriched/natural pair (P047): 9 window events in total, 12.8 t·yr per detector at LZ's rate.
- P(DM | event) prior for the null branch: 0.015 (68 % 0.0016–0.137), P061 community prior.

## 3. Methods (statistics only)

1. **Cumulative exposure** E_i(t) piecewise-linear per dataset (§2.1); expected new signal μ_i(t) = r_i E_i^new(t) with
   E^new excluding LZ's analysed 2.84 t·yr in the 200–270 window (for the 200–400 window SR3's sideband is new).
2. **Asimov median significance** Z_A = √(2[(s+b)ln(1+s/b) − s]) for new data only and combined with LZ's event
   (s + 1, b + 5.7×10⁻⁴). The first date with Z_A ≥ 3, 5 is tabulated.
3. **Exact-Poisson first-crossing toys** (20 000 per configuration, seed 88). Events are a Poisson process with
   cumulative mean μ(t) + b(t); we draw the first 30 arrival positions as a cumulative sum of Exp(1) variates in
   cumulative-mean space and map them to dates by inverting the cumulative curve. The j-th event at date t_j crosses
   the threshold if P(≥ j | b(t_j)) < p_thr (p_3σ = 1.35×10⁻³, p_5σ = 2.87×10⁻⁷) — with b ≪ 1 this means 2 events for
   3σ and 3 (occasionally 4) for 5σ; adding LZ's event (n_prior = 1, b_prior = 5.7×10⁻⁴) reduces both by one. We report
   the 16/50/84 % quantiles of the first-crossing date and the probability of crossing by end-2027/2028/2030, for the
   best fit, for ×0.30, and with the rate marginalised over Gamma(2,1) (Poisson–Gamma, "PG").
4. **Bayes factor** DM-with-continuing-rate versus one-off: BF(n) = ∫ Gamma(s; 2, 1) Pois(n | k s + b) ds / Pois(n | b),
   k = expected new events at s = 1. "Settled" = BF > 100 or < 1/100. P(settled | truth) sums the Poisson predictive
   of n under truth A/B (best fit or ×0.30) or C (b only). For n = 0, BF = (1+k)⁻².
5. **Null branch**: P(DM | 0 new) = πL/(πL + 1 − π) with L = (1+k)^(−α), α = 2 or 1.5, π from P061.
6. **A vs B dates**: the date at which the pooled whole-ROI count (LZ's event included) reaches the P057/P050/P034
   requirement; quantiles from the Gamma distribution of the N-th arrival.

## 4. Results

### 4.1 Expected events on disk and by date (present generation, best fit)
| window | model | μ now | P(0) | μ end-2027 | μ end-2028 | b now |
|---|---|---|---|---|---|---|
| 200–270 | L10 | 2.16 | 0.115 | 3.22 | 4.04 | 0.0036 |
| 200–270 | δ300 | 1.13 | 0.324 | 1.68 | 2.11 | 0.0036 |
| 200–270 | δ350 | 3.19 | 0.041 | 4.76 | 5.97 | 0.0036 |
| 200–270 | δ366 | 4.91 | 0.007 | 7.32 | 9.18 | 0.0036 |
| 200–270 | δ380 | 4.99 | 0.007 | 7.44 | 9.34 | 0.0036 |
| 200–400 | L10 | 4.57 | 0.010 | 6.50 | 7.99 | 0.020 |
| 200–400 | δ300 | 2.34 | 0.097 | 3.32 | 4.09 | 0.020 |
| 200–400 | δ350 | 10.7 | 2×10⁻⁵ | 15.3 | 18.8 | 0.020 |
| 200–400 | δ366 | 34.0 | ~0 | 48.3 | 59.5 | 0.020 |
| 200–400 | δ380 | 218 | ~0 | 310 | 382 | 0.020 |

Per experiment now (200–270): LZ 0.92/0.48/1.36/2.09/2.13, XENONnT 0.75/0.39/1.10/1.70/1.72, PandaX-4T
0.49/0.26/0.73/1.12/1.14 (L10/δ300/350/366/380). P069's 2.30/1.20/3.40/5.23 are reproduced up to its archival 0.27/0.61.

### 4.2 Discovery dates — `P088_timeline_table.csv`, `P088_table_experiment_x_model.csv`
Median toy first-crossing date, new data only, best fit ("on disk" = before Sep 2026):

| experiment (edge) | L10 3σ / 5σ | δ300 | δ350 | δ366 | δ380 |
|---|---|---|---|---|---|
| LZ (600) | on disk / Oct 2031 | Sep 2031 / >2035 | on disk / Mar 2029 | on disk / Apr 2027 | on disk / Mar 2027 |
| LZ (1000) | on disk / Apr 2027 | Nov 2027 / >2035 | on disk / on disk | on disk / on disk | on disk / on disk |
| XENONnT (600) | on disk / >2035 | >2035 / >2035 | on disk / Mar 2030 | on disk / Dec 2027 | on disk / Nov 2027 |
| XENONnT (1000) | Nov 2026 / Mar 2029 | Nov 2029 / >2035 | on disk / on disk | on disk / on disk | on disk / on disk |
| PandaX-4T (600) | Sep 2027 / >2035 | >2035 / >2035 | on disk / >2035 | on disk / Oct 2029 | on disk / Sep 2029 |
| PandaX-4T (1000) | May 2028 / Aug 2031 | >2035 / >2035 | on disk / May 2027 | on disk / on disk | on disk / on disk |
| present generation (600) | on disk / Apr 2027 | Mar 2027 / Apr 2030 | on disk / on disk | on disk / on disk | on disk / on disk |
| present generation (1000) | on disk / on disk | on disk / May 2028 | on disk / on disk | on disk / on disk | on disk / on disk |
| XLZD 60 t (400 keV) | Feb 2032 / Mar 2032 | Apr 2032 / Aug 2032 | Jan 2032 / Feb 2032 | Jan 2032 / Jan 2032 | Jan 2032 / Jan 2032 |

Present generation, 600 phd, L10: 5σ median Apr 2027 (2027.29), 16–84 % Jan 2025–Aug 2029; Asimov Nov 2026 (2026.90);
Poisson–Gamma median Aug 2025; combined with LZ's event Oct 2025. δ300: toy 5σ Apr 2030 (2030.32), Asimov Mar 2030.
δ350/366/380: Asimov 5σ 2025.5/2024.5/2024.4 (on disk). 1000 phd: L10 Asimov Jan 2025, toy Nov 2024 (on disk);
δ300 Asimov Dec 2027, toy May 2028.

Probabilities of a 5σ new-only crossing (present generation, 600 phd), by end-2027 / end-2028 / end-2030:
L10 0.63/0.78/0.93; δ300 0.24/0.36/—; δ350 0.85/0.94/—; δ366 0.98/0.995/1.0. At ×0.30 by end-2028: 0.13/0.03/0.27/0.52;
median 5σ dates >2035 / >2035 / Jun 2031 / Oct 2028. Poisson–Gamma (flat) by end-2028: L10 0.82, δ366 0.95.
Combined with LZ's event by end-2027: 0.83/0.51/0.95/0.995. 1000 phd by end-2027: 0.90/0.45/1.0/1.0.
LZ alone (600 phd, L10): P(5σ) 0.18/0.26/0.43 by end-2027/2028/2030; with the 1000 phd edge 0.53/0.59/0.75.
XLZD (400 keV, new only): Asimov 5σ at 2032.24 (L10), 2032.6 (δ300), 2032.1 (δ350), 2032.02 (δ366); toys ×0.30: Feb 2033 (L10), Sep 2034 (δ300), May 2032 (δ350), Feb 2032 (δ366).

### 4.3 LZ's own 600–1000 phd sideband — `P088_LZ_sideband.csv`
| model | SR3 2.84 t·yr: μ, P(0) | untouched 6.76: μ, P(0) | SR3+untouched 9.6: μ, P(0), Z_Asimov |
|---|---|---|---|
| L10 | 0.28, 0.75 | 0.67, 0.51 | 0.95, 0.39, 2.7σ |
| δ300 | 0.155, 0.86 | 0.37, 0.69 | 0.52, 0.59, 1.8σ |
| δ350 | 1.06, 0.35 | 2.53, 0.08 | 3.6, 0.028, 6.0σ |
| δ366 | 3.81, 0.022 | 9.07, 1.2×10⁻⁴ | 12.9, 2.5×10⁻⁶, 12.7σ |
| δ380 | 30.6, 5×10⁻¹⁴ | 72.8, ~0 | 103, ~0, 42σ |
Background 0.0027/0.0064/0.0091. SR3's empty sideband (Fig. S4) already disfavours δ = 366 keV at the best fit at
97.8 % CL and kills δ = 380 (P038's point); opening the untouched 6.76 t·yr decides δ ≥ 350 keV outright.

### 4.4 Settled? Bayes factors — `P088_settled.csv`
BF > 100 needs 2 new events in the 200–270 window (3 in 200–400 keV from 2028, 4–5 for the very large δ ≥ 366 counts).
P(settled | A/B true at best fit), present generation, 600 phd, by end-2027 / 2028 / 2030 / 2032:
L10 0.83/0.91/0.98/0.99; δ300 0.50/0.62/0.80/0.85; δ350 0.95/0.98/1.0/1.0; δ366 0.99/1.0/1.0/1.0.
At ×0.30: L10 0.25/0.34/0.51/0.58; δ300 0.09/0.13/0.23/0.27; δ350 0.42/0.54/0.72/0.84; δ366 0.65/0.83/0.92/0.95.
P(settled | C one-off), i.e. BF(0) < 1/100: L10 never before 2033 in the 200–270 window (BF(0) = 0.056/0.039/0.022/0.018
at end-2027/2028/2030/2032; k = 9 → 75 t·yr needed); δ350 only by end-2032 (0.99); δ366/380 by end-2028 (0.99).
With the 1000 phd window: L10 by end-2030 (0.95), δ350 by end-2027 (0.97), δ366 by end-2027 (0.97).
1000 phd, A true at best fit by end-2027: 0.99/0.85/1.0/1.0 (L10/δ300/350/366).

### 4.5 Null branch — `P088_null_branch.csv`
P(DM | 0 new events) from π = 0.015 (present generation, 200–270 keV): now 0.0015 (L10, flat) / 0.0027 (Jeffreys);
below 10⁻³ by Aug 2027 (2027.61, L10 flat), May 2030 (Jeffreys), Nov 2030 (δ300 flat; never before 2035 Jeffreys),
May 2026 = on disk (δ350), on disk (δ366/380: 2025.2); below 10⁻⁴: never (L10), May 2030 (δ366). Prior 0.137 (P061
upper 68 %): < 10⁻² by Sep 2027 (L10), < 10⁻³ never before 2035 (L10), May 2030 (δ366). Prior 0.0016: < 10⁻³ on disk,
< 10⁻⁴ Oct 2027 (L10). With the 1000 phd window: L10 < 10⁻³ on disk (2025.1), < 10⁻⁴ Apr 2031.
Note the asymmetry with §4.4: P(DM) < 10⁻³ is reached years before BF < 1/100 because the prior odds already
supply a factor 66.

### 4.6 A versus B — `P088_AvsB_shape_modulation.csv`
Pooled whole-ROI events (LZ's event included) on disk now: 1 + 0.352 × 17.9 = 7.3 (600 phd) for every model; 1000 phd:
1 + 0.447 × 20.8 = 10.3 (L10), 36 (δ366). Hence the shape requirement (2 events if L10 true, 5–7 if inelastic) is on
disk: median dates 2023.2 (L10 true), 2026.3 (inelastic true, 600 phd; 84 % quantile Aug 2027), 2022.2–2022.5 (1000 phd).
Timing (P034), 600 phd: δ366 3σ (6.5 events) on disk (median 2026.0; 16–84 % 2024.7–2027.4), 5σ (17.6) Nov 2030;
δ350 3σ (11.5) Apr 2028 (2027.1–2029.8), 5σ (32) >2035; δ380 3σ 2025.3 (on disk), 5σ Apr 2029; δ300 (96.5) never
before XLZD. 1000 phd: δ366 3σ/5σ on disk (2023.5/2025.8), δ350 3σ Nov 2027. XLZD collects the P034 counts within its
first year (P050). Caveat: the whole-ROI count includes 100–200 keV background (0.042/t·yr, P050: ≈0.8 events in the
present data), which P057's signal-only counts ignore.

### 4.7 Tungsten and isotopes — `P088_cawo4.csv`, `P088_isotope_pair.csv`
| CaWO₄ scenario | model | μ | b (0.01/kg·yr) | 3σ median | 5σ median | P(3σ) / P(5σ) by end |
|---|---|---|---|---|---|---|
| 10 kg·yr, 2027–30 | δ300 | 0.026 | 0.1 | never | never | 0.004 / 0 |
| | δ350 | 0.37 | 0.1 | never | never | 0.03 / 0.001 |
| | δ366 | 2.8 | 0.1 | Feb 2029 | never | 0.62 / 0.18 (0.25 / 0.01 at b = 0.1/kg·yr) |
| | δ380 | 56 | 0.1 | Jan 2027 | Feb 2027 | 1 / 1 |
| 100 kg·yr, 2029–34 | δ300 | 0.26 | 1 | never | never | 0.01 / 0 |
| | δ350 | 3.7 | 1 | never (median) | never | 0.41 / 0.03 |
| | δ366 | 28 | 1 | May 2029 | Dec 2029 | 1 / 1 |
| 1 t·yr, 2030–35 | δ300 | 2.6 | 10 | never | never | 0.05 / 0 |
| | δ350 | 37 | 10 | Aug 2030 | Oct 2031 | 1 / 1.0 |
L10 gives ≤ 6×10⁻⁵ events per kg·yr: tungsten is a pure inelastic-δ meter (P046). A realistic 1–10 kg·yr CaWO₄
programme is therefore useless for δ ≤ 350 keV and redundant for δ ≥ 366 keV, where xenon has already decided.
¹³⁶Xe pair (P047): 12.8 t·yr per detector needs 4.3 yr of a 5 t enriched TPC at 60 % live (21 yr for 1 t); no such
detector exists or is scheduled, so this test is a mid-2030s option.

### 4.8 Exothermic, directional, mass
P058: a surviving χ₂ would already have flooded LZ's 125–200 keV bin; the exothermic reading is constrained by
existing data, not by the calendar. P067: directional detectors need decades (≥ 41 yr per 1000 m³). P062: xenon
cannot fix m_χ above ~700 GeV with < 600 events; no date is attached.

## 5. Validation and robustness
- Present-generation hidden counts reproduce P069 (2.30/1.20/3.40/5.23) once its archival 0.27/0.61 are added;
  LZ untouched 0.81 (L10) matches P069's 0.815 and P020's 2.38 ROI events.
- Asimov and exact-Poisson toy medians agree to ≤ 0.4 yr where both are finite (e.g. L10 600 phd: 2026.90 vs 2027.29).
- Two new events on b = 0.004 give P(≥2) = 7.7×10⁻⁶ → 4.3σ, and three with LZ's event 5.5σ: P069's values.
- Background bracket: P069 showed Z_comb changes by ±0.4σ at N = 3 between 2×10⁻⁴ and 0.0106; with b ≤ 0.05 the toy
  crossing count (2 for 3σ, 3 for 5σ) is unchanged, so the dates here are insensitive to the bracket.
- Exposure uncertainty (P069): hidden exposure −16 %/+26 % → μ_now scales linearly; the 16–84 % toy bands (≈ ±2 yr for
  L10) dominate over it.
- Rate band: ×0.30 moves every 5σ date by ≥ 4 yr (L10 beyond 2035 for the present generation); Poisson–Gamma medians
  are *earlier* than the best fit because the flat posterior (mean 2) weights high rates.

## 6. Figures
- `figures/P088_fig1_timeline.png` — expected new events (top) and Asimov new-only significance (bottom) vs calendar
  date for the present generation, 600 phd (left) and 1000 phd (right) edges; markers: toy median first crossing of
  3σ (circles) and 5σ (squares), bars 16–84 %; crossings already on disk drawn at Sep 2026.
- `figures/P088_fig2_decision_tree.png` — decision tree from the on-disk reanalysis (LZ sideband; joint 200–270 keV)
  through N = 0/1/≥2 branches, the A-vs-B stage (shape, timing, tungsten, isotopes) and XLZD, with verdict dates.
- `figures/P088_fig3_settled_null.png` — left: P(BF > 100 either way) vs year for each truth (solid best fit, dashed
  ×0.30, black one-off); right: P(DM | no new event) vs date from π = 0.015 (flat and Jeffreys).

## 7. Failed approaches
Full-N toy generation (arrays up to 20 000 × 2000) exceeded five minutes and was replaced by first-30-arrival
sampling (exact for first crossings). `np.quantile` with inf entries gave NaN medians (fixed). A "world" row counting
XLZD in the 200–270 window was replaced. P081/P072 do not exist; own Gamma–Poisson predictive used.

## 8. Discussion
The calendar is short. Under the best fit, the 18 t·yr already on disk contain the two events that settle
"population or one-off" with probability 0.64 (L10) to 0.96 (δ = 366 keV); by end-2028 the present generation reaches
that verdict with probability 0.9–1.0 for every model but δ = 300 keV (0.6). The other half of the question — is it
inelastic or an elastic q⁴ operator — also needs no new hardware: 5–7 events of shape or 6.5 events of June clustering
are on disk for δ ≥ 366 keV, and δ ≥ 350 keV is decided by LZ's own 600–1000 phd sideband the day it is opened. What
the present generation cannot do is (i) exclude a ×0.3 rate (only XLZD, 2033), (ii) see the δ = 300 keV modulation
(97 events), or (iii) beat P001's 0.09 cap against a steady unknown with counting alone. Tungsten at realistic
exposures and isotopic xenon arrive after the decision has been made.

## 9. References
LZ Collaboration, arXiv:2609.02823 (2026). G. Cowan, K. Cranmer, E. Gross, O. Vitells, EPJC 71, 1554 (2011).
R. E. Kass, A. E. Raftery, JASA 90, 773 (1995). J. Aalbers et al. (XLZD), J. Phys. G 50, 013001 (2023).
A. H. Abdelhameed et al. (CRESST), PRD 100, 102002 (2019). D. Baxter et al., EPJC 81, 907 (2021).
Corpus: P001, P016, P020, P034, P038, P046, P047, P050, P057, P058, P059, P061, P062, P067, P069.

## 10. Tools and provenance (mirrors provenance/P088.json)
- Software: python 3.12.13 (.venv); numpy 2.5.3 (rng exponential/gamma/poisson, searchsorted, trapezoid); scipy 1.18.1
  (stats.poisson, gamma, norm); pandas 3.0.5; matplotlib 3.11.2 (Agg); common/lzcommon.py imported (LZ constants);
  WimPyDD not called (spectra enter via P050/P069/P038/P034/P046/P057 tables). No WimPyDD files generated.
- Script: `output/code/P088_timeline.py`, `.venv/bin/python output/code/P088_timeline.py` (73 s); outputs in
  `output/work/P088/` (`P088_timeline_table.csv`, `P088_table_experiment_x_model.{csv,md}`, `P088_LZ_sideband.csv`,
  `P088_settled.csv`, `P088_null_branch.csv`, `P088_AvsB_shape_modulation.csv`, `P088_cawo4.csv`,
  `P088_isotope_pair.csv`, `P088_results.json`, `run_log.txt`, three figures).
- Local inputs: PAPER_GUIDE; ledger rows P001/P005/P016/P020/P021/P027/P035/P045/P070; papers and work tables of
  P034, P038, P046, P047, P050, P057, P058, P059, P061, P062, P067, P069; lzcommon.py; ENVIRONMENT_versions.txt.
- Recalled knowledge (10 items): XENONnT 4.0 t / 60 % live (likely/uncertain); PandaX-4T 2.7 t / 60 % (likely/uncertain);
  XLZD 60 t, 48 t·yr/yr from 2032 (uncertain); LZ running to 2032 (uncertain); CRESST-scale 1–10 kg·yr (likely);
  CaWO₄ W-band background 0.01–0.1 per kg·yr (uncertain); nEXO-class 5 t enriched ¹³⁶Xe (likely); Gamma–Poisson
  predictive (certain); Jeffreys Gamma(1/2) (certain); Asimov/exact-Poisson Z (certain).
- Agent tools: Read ×24, Bash ×13, Write ×4, Edit ×9, Skill ×1 (dataviz), ToolSearch ×1, TaskStop ×1. Data requests: none.
