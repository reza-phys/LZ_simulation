# P050 — How quickly would a 60-tonne xenon detector confirm or exclude the LZ high-energy hint? (research record)

Simulated date 2026-09-10. Author profile: next-generation xenon detector planners. Category PROJ; hep-ex (cross-list hep-ph).
Script: `output/code/P050_next_generation.py` (run from the simulation root with `.venv/bin/python`; ~5 min on the first run,
which computes and caches the WimPyDD spectra, 15 s afterwards). All tables quoted below are in `output/work/P050/`; every
number in the paper appears here or in `run_log.txt`.

## 1. Motivation and framework

LZ (arXiv:2609.02823) reports one NR-like event at 248 keV in 2.84 t·yr, fitted as 1.0 (+1.4, −0.7) events of the L10ˢ
interaction at 1 TeV (Table I), or as inelastic O₁ scattering at δ = 300–380 keV (Table S7, P021). P020 forecast LZ's own
untouched exposure (6.76 t·yr to Sept 2026), P034 the exposures needed for the annual-modulation test, P038 the effect of
LZ's 600 phd (E50 = 269.9 keV) efficiency edge, P005/P035 the counts in XENONnT and PandaX-4T. Here we ask what a detector
of the XLZD class (40–60 t fiducial, data from about 2032; recalled/uncertain) and the present generation
(LZ 4.7 t, XENONnT ≈ 4 t, PandaX-4T ≈ 2.7 t fiducial; recalled/likely) would do with an ROI extended to E50 = 300, 400 or
600 keV, for four questions: (a) 5σ discovery of the LZ best-fit rates, (b) exclusion of the LZ 90 % lower edge with zero
events, (c) 3σ separation of the L10 spectrum from inelastic δ = 366 keV, (d) determination of δ to ±10 keV from the xenon
spectrum alone.

Everything is expressed relative to LZ's normalisation: each model is scaled so that it gives exactly 1.0 event per
2.84 t·yr inside LZ's ROI (efficiency model below); the LZ 68 % band (×0.30–2.36, Table I) is carried as a rate band.
Rates per tonne-year are detector independent for a xenon target (P005), so only the efficiency window, the exposure and
the background differ between detectors.

## 2. Signal models (WimPyDD 2.0.4, 1 TeV)

- Halo: Baxter-2021 SHM (v₀ = 238, v_esc = 544, v_sun = (11.1, 12.2, 7.3) km/s) through `lz.wd_halo(day_of_year=d)` with an
  explicit v_min grid to 844 km/s, averaged over 12 days (15, 45, …, 345), as in P038 ("annual").
- Inelastic O₁ isoscalar, LZ unit coupling c₁ˢ = 1/m_v² → WimPyDD c⁰ = 2/m_v² (P003 convention, `lz.wd_c_from_anand`),
  δ = 300, 348, 350, 352, 364, 366, 368, 378, 380, 382 keV on E_true = 1.5–799.5 keV (3 keV steps).
  Full-window counts per 2.84 t·yr at unit coupling: 1.665×10⁴ (300), 600 (350), 127 (366), 25.2 (380). P038 quotes
  16464 at δ = 300 keV (1 % difference from the day sampling) and 584 at 350 keV.
- L10 = 4[(q²/m_N²)O₄ − O₆] (P012 reduction, m_M = m_N; WimPyDD c₄ = 8q²/(m_v²m_N²), c₆ = −8/m_v²) on E_true = 1.5–1498.5 keV,
  because the 1 TeV L10 spectrum extends to ≈ 1.4 MeV: 55 % of its rate lies above 270 keV (Fig. 1). Its LZ-ROI count at
  d10 = 1 (WimPyDD convention) is 3.32 events per 2.84 t·yr (P012: 3.34).

Efficiency (true energy): ε(E) = 0.955 · ½[1+erf((E−5.4)/(√2·2.5))] · ½[1−erf((E−E50)/(√2·σ))] with
(E50, σ) = (269.9, 11.5) keV for LZ (Fig. S2; P021), (300, 13), (400, 17) and (600, 22) keV for the extended windows
(P038: erf widths 10.9 keV at 600 phd → 18 keV at 1000 phd). Resolution σ_E = 11√(E/248) keV (P009/P021), applied as a
Gaussian kernel on a 1 keV observed-energy grid. The S2c ceiling is assumed to track the NR band (P038 recommendation).

Validation of acceptances (fraction of the full-window rate inside LZ's ROI): 0.814/0.444/0.172/0.028 at
δ = 300/350/366/380 keV versus P038's 0.81/0.44/(0.19 at 365, 0.115 at 370)/0.029.

**Table 1 — normalisation (`P050_normalisation.csv`).** Rates per t·yr at the LZ best fit (1 event per 2.84 t·yr in LZ's ROI).

| model | N_LZ at unit coupling (per 2.84 t·yr) | A_LZ | A_300 | A_400 | rate, full window | rate, E50 = 300 | rate, E50 = 400 | fraction of true recoils > 270 keV |
|---|---|---|---|---|---|---|---|---|
| L10 | 3.32 | 0.422 | 0.453 | 0.536 | 0.834 | 0.378 | 0.447 | 0.554 |
| δ = 300 | 13557 | 0.814 | 0.826 | 0.928 | 0.433 | 0.357 | 0.401 | 0.148 |
| δ = 350 | 266.7 | 0.444 | 0.489 | 0.867 | 0.793 | 0.387 | 0.688 | 0.536 |
| δ = 366 | 21.86 | 0.172 | 0.240 | 0.832 | 2.047 | 0.491 | 1.703 | 0.822 |
| δ = 380 | 0.700 | 0.028 | 0.106 | 0.826 | 12.71 | 1.346 | 10.50 | 0.975 |

The LZ-ROI rate is 1/2.84 = 0.352 per t·yr by construction; A_LZ for L10 is 0.42 because the elastic spectrum continues to
1.4 MeV (and 7 % lies below 55 keV). For δ ≥ 366 keV the extended window multiplies the rate by 4.8–30: P038's point that
LZ's own 600–1000 phd region should then have held 3.8 (366) to 29 (380 keV) events per ROI event, which Fig. S4 shows
empty. For δ ≥ 366 keV the best-fit "discovery" numbers below are therefore of formal interest only; the physically
relevant statements are the exclusion ones.

## 3. Background model

NR band (±2σ around the NR median), per 2.84 t·yr in LZ's 4.7 t FV, binned in observed energy
(`P050_background_bins.csv`). Each bin is split into a per-tonne-constant part (ER leakage, accidentals, atmospheric
CEνNS) and a surface part (MSSI, ∝ FV surface/volume).

| E_obs bin (keV) | constant | surface (MSSI) | total (LZ-like) | source |
|---|---|---|---|---|
| 100–125 | 0.10 | 0 | 0.10 | ER leakage into the NR band; estimate (×3 bracket) |
| 125–200 | 0.020 | 0 | 0.020 | P016 b_M (empty 125–200 keV bin) |
| 200–270 | 4.0×10⁻⁴ | 1.7×10⁻⁴ | 5.7×10⁻⁴ | P016 b_H; decomposition: accidentals 1.5×10⁻⁴ (P022), ν 3.4×10⁻⁵ (P019), MSSI 1.7×10⁻⁴ (P004), ER/other 2.2×10⁻⁴ |
| 270–300 | 3.1×10⁻⁵ | 2.6×10⁻⁴ | 2.9×10⁻⁴ | P038 MSSI vs S1c edge × NR-band share 0.15 (= 0.0027/0.018); accidentals 10⁻⁴ per 100 keV (estimate); ν 10⁻⁵ above 270 keV |
| 300–350 | 5.2×10⁻⁵ | 6.5×10⁻⁴ | 7.0×10⁻⁴ | idem |
| 350–400 | 5.2×10⁻⁵ | 1.07×10⁻³ | 1.12×10⁻³ | idem |
| 400–450 | 5.2×10⁻⁵ | 1.87×10⁻³ | 1.93×10⁻³ | idem |
| 450–500 | 5.2×10⁻⁵ | 3.39×10⁻³ | 3.44×10⁻³ | idem |
| 500–600 | 1.0×10⁻⁴ | 0.0201 | 0.0202 | idem (RFR MSSI rises with λ ≈ 130 phd) |

Energy → S1c edge mapping from P038's MC (600/700/800/1000/1200 phd ↔ 271.6/310.5/348.7/423.2/495.1 keV; linear
extrapolation above). Extension bins are weighted by the window acceptance at the bin centre, and only bins whose
acceptance exceeds 1 % are used (the same bins enter the signal sums). The 5.4–100 keV region is not used: it holds
~60 background events per 2.84 t·yr (P016) and only 15 % of the L10 rate (no inelastic rate).

Scaling with detector size (stated assumptions):
- Wall MSSI ∝ wall area / volume = 2πRH/(πR²H) = 2/R; RFR (cathode) MSSI ∝ πR²/(πR²H) = 1/H. For H = 2R both scale as 1/R.
  FV as a cylinder of LXe density 2.86 t/m³ (recalled, likely): R = 0.728 m (LZ TPC radius, recalled/likely; P004),
  1.306 m (40 t), 1.495 m (60 t) → per-tonne surface backgrounds ×0.558 (40 t), ×0.487 (60 t). Detector neutrons (≤ 10⁻⁴
  above 200 keV, P013) scale the same way and are neglected.
- Accidentals: N ∝ R_S1·R_S2·T_max·T_live; per tonne-year this is ∝ R_S1R_S2T_max/M, which is constant if the isolated-pulse
  rates are held ∝ M^{1/2} by cuts, and grows for internal-radioactivity-dominated rates. We take the per-tonne rate
  constant (baseline) and note that a ×10 pessimistic variant changes no exposure below by more than 3 % because the
  discovery regime is background-free (s ≈ 2–4 events at 5σ).
- Atmospheric CEνNS and ER leakage: ∝ mass (per tonne constant).

**Table 2 — background per t·yr in the NR band (`P050_background_rates.csv`).**

| window | detector | 100–200 keV | 200–270 keV | extension | total ≥ 200 keV | ≥ 200 keV per 2.84 t·yr |
|---|---|---|---|---|---|---|
| LZ | LZ-like | 0.0423 | 2.0×10⁻⁴ | 1×10⁻⁵ | 2.1×10⁻⁴ | 6.0×10⁻⁴ |
| LZ | 60 t | 0.0423 | 1.7×10⁻⁴ | 5×10⁻⁶ | 1.8×10⁻⁴ | 5.0×10⁻⁴ |
| E50 = 300 | LZ-like | 0.0423 | 2.0×10⁻⁴ | 9.5×10⁻⁵ | 3.0×10⁻⁴ | 8.5×10⁻⁴ |
| E50 = 400 | LZ-like | 0.0423 | 2.0×10⁻⁴ | 7.6×10⁻⁴ | 9.7×10⁻⁴ | 2.7×10⁻³ |
| E50 = 400 | 40 t | 0.0423 | 1.8×10⁻⁴ | 4.5×10⁻⁴ | 6.2×10⁻⁴ | 1.8×10⁻³ |
| E50 = 400 | 60 t | 0.0423 | 1.7×10⁻⁴ | 4.0×10⁻⁴ | 5.7×10⁻⁴ | 1.6×10⁻³ |
| E50 = 600 | LZ-like | 0.0423 | 2.0×10⁻⁴ | 9.6×10⁻³ | 9.9×10⁻³ | 0.028 |
| E50 = 600 | 60 t | 0.0423 | 1.7×10⁻⁴ | 4.8×10⁻³ | 4.9×10⁻³ | 0.014 |

A 60 t detector with a 400 keV window therefore sees 1.6×10⁻³ background per 2.84 t·yr-equivalent above 200 keV — the
same order as LZ's present 200–270 keV band — against a signal of 1.3–30 events per 2.84 t·yr.

## 4. Discovery

Statistic: Asimov significance summed over the observed-energy bins, Z² = 2Σᵢ[(sᵢ+bᵢ)ln(1+sᵢ/bᵢ) − sᵢ] (Cowan et al. 2011;
recalled, certain), with sᵢ = k·(signal per t·yr in bin i), bᵢ = k·(background per t·yr) and exposure k solved by
`brentq`. The formula is asymptotic; in this one-to-four-event regime it agrees with exact Poisson significances to
≈ 0.3σ (e.g. three events on b = 2.5×10⁻³ give Z = 6.05 Asimov vs 5.8 exact; cf. P020's N = 3 → 6.8σ at b = 10⁻³).
Cross-check: the 5σ exposures correspond to s = 2.0–4.0 expected events in every scenario (column `s_at_5sigma`).

**Table 3 — exposures (t·yr) at the LZ best fit (`P050_discovery.csv`).** Bracket = lower/upper 68 % rate (×0.30/×2.36).

| window, background | model | rate/t·yr | E_3σ | E_5σ | E_5σ (×0.30 / ×2.36) | E(N̄ = 3) | E(P(N≥3) = 90 %) | s at 5σ |
|---|---|---|---|---|---|---|---|---|
| LZ, LZ-like | L10 | 0.299 | 4.44 | 12.3 | 57.5 / 4.27 | 10.0 | 17.8 | 3.69 |
| LZ, LZ-like | δ300 | 0.352 | 4.12 | 11.5 | 57.0 / 3.87 | 8.52 | 15.1 | 4.03 |
| LZ, LZ-like | δ350 | 0.352 | 3.07 | 8.54 | 38.4 / 3.02 | 8.53 | 15.1 | 3.00 |
| LZ, LZ-like | δ366 | 0.351 | 2.45 | 6.80 | 28.8 / 2.49 | 8.54 | 15.2 | 2.39 |
| LZ, LZ-like | δ380 | 0.346 | 2.07 | 5.74 | 23.5 / 2.15 | 8.67 | 15.4 | 1.99 |
| E50 = 300, LZ-like | L10 / δ300 / δ350 / δ366 / δ380 | 0.325/0.357/0.387/0.490/1.345 | 3.94/4.05/2.71/1.61/0.41 | 10.9/11.2/7.53/4.48/1.15 | — | 9.2/8.4/7.7/6.1/2.2 | — | 3.56/4.02/2.92/2.20/1.55 |
| E50 = 400, LZ-like | L10 / δ300 / δ350 / δ366 / δ380 | 0.394/0.401/0.687/1.703/10.50 | 3.22/3.59/1.41/0.42/0.051 | 8.95/9.97/3.91/1.17/0.141 | — | 7.6/7.5/4.4/1.8/0.29 | 13.5/13.3/7.7/3.1/0.51 | 3.53/4.00/2.69/1.99/1.48 |
| E50 = 400, 40 t | L10 / δ300 / δ350 / δ366 / δ380 | idem | 3.08/3.49/1.32/0.39/0.048 | 8.55/9.70/3.68/1.09/0.133 | — | idem | idem | 3.37/3.89/2.53/1.86/1.39 |
| E50 = 400, 60 t | L10 / δ300 / δ350 / δ366 / δ380 | idem | 3.05/3.47/1.31/0.39/0.047 | 8.47/9.65/3.63/1.08/0.131 | 38.7/47.3/15.6/4.34/0.50 (×0.30); 2.97/3.28/1.32/0.41/0.051 (×2.36) | idem | idem | 3.34/3.87/2.49/1.83/1.37 |
| E50 = 600, 60 t | L10 / δ300 / δ350 / δ366 / δ380 | 0.547/0.413/0.757/1.955/12.13 | 2.31/3.40/1.21/0.35/0.042 | 6.42/9.45/3.36/0.97/0.116 | 30.0/46.5/14.5/3.92/0.45 (×0.30) | 5.5/7.3/4.0/1.5/0.25 | 9.7/12.9/7.0/2.7/0.44 | 3.51/3.90/2.54/1.89/1.41 |

Observations. (i) Detector size enters only through the surface backgrounds and changes the exposures by ≤ 6 %: the
question is exposure, not background. (ii) The ROI edge matters far more: for δ = 350/366/380 keV a 400 keV edge cuts
E_5σ by ×2.2/×5.8/×41 relative to the LZ window (P038's acceptance argument); for L10 by ×1.4 (×1.9 at 600 keV, which
picks up the 51 % of the L10 rate above 270 keV, at the price of ×5 more MSSI). (iii) For L10 and δ = 300 keV the
exposure is set by the rate alone: E_5σ ≈ 3.5–4 events / rate.

Calendar time (live rates: LZ 4.71 t × 220/371 = 2.79 t·yr/yr; present generation (4.71 + 4.0 + 2.7) t × 0.6 = 6.8 t·yr/yr;
40 t × 0.8 = 32; 60 t × 0.8 = 48 t·yr/yr):
- 60 t, E50 = 400 keV, best fit: 5σ after 8.5/9.6/3.6/1.1/0.13 t·yr = 64/73/28/8/1 live days (L10/300/350/366/380);
  3σ after 3.0/3.5/1.3/0.4/0.05 t·yr. At the lower 68 % rate: 0.81/0.99/0.33/0.09/0.01 yr.
- present generation, LZ-like ROI, best fit: E_5σ = 12.3 t·yr (L10) = 1.8 yr of the combined live rate.
- N ≥ 3 with 90 % probability (60 t, 400 keV): 13.5/13.3/7.7/3.1/0.5 t·yr = 0.28/0.28/0.16/0.065/0.011 yr.
- June clustering (P034): timing information shortens the background-only discovery exposure by at most ×1.36 for
  δ ≥ 350 keV; not applied here (the exposures above are counting-only).

Timeline (`P050_timeline_milestones.csv`, Fig. 2). LZ alone: exposure 2.84 + 2.79 (t − 2024.25) t·yr until end-2028
(16.1 t·yr, P034's milestone), then flat. Present generation: LZ plus, from Sept 2026, the existing XENONnT + PandaX-4T
4.64 t·yr (P005/P035, if reanalysed to high energy) growing at 4.0 t·yr/yr to 2030. 60 t: 48 t·yr/yr from 2032.0.

| programme | model | 5σ year, best fit (exposure) | 3σ / 5σ year at ×0.30 |
|---|---|---|---|
| LZ alone | L10 / δ300 / δ350 / δ366 / δ380 | 2027.65 (12.3) / 2027.35 (11.5) / 2026.30 (8.6) / 2025.70 (6.9) / 2025.30 (5.8) | 3σ: never / never / 2028.2 / 2026.95 / 2026.3; 5σ never |
| present generation | idem | 2026.70 (14.4, at the reanalysis step) / 2026.70 / 2026.30 / 2025.70 / 2025.30 | 3σ: 2027.65 / 2027.6 / 2026.7 / 2026.7 / 2026.3; 5σ: never / never / never / 2028.85 / 2028.05 |
| 60 t, 400 keV | idem | 2032.18 (8.5) / 2032.20 (9.7) / 2032.08 (3.7) / 2032.02 (1.1) / 2032.00 (0.24) | 5σ: 2032.81 / 2032.99 / 2032.33 / 2032.09 / 2032.01 |

The LZ-alone line reproduces P020's message (2.4 best-fit events by Sept 2026; 5σ combined with two new events) and P034's
background-only 5σ at 10.6 t·yr (ours 12.3 with the 100–200 keV bins included, 8.5 t·yr for δ = 350). The Asimov curves
are medians at a fixed rate; the actual outcome is a Poisson draw (P020).

## 5. Exclusion with zero events

If the true LZ-ROI rate is s_low per 2.84 t·yr, the rate in a window w is s_low/(2.84 A_LZ/A_w) per t·yr, and zero events
exclude it at CL 1−α when the expected count reaches −ln α (CLs; identical to the Poisson limit and independent of b):
2.303 (90 %), 2.996 (95 %). We bracket the lower edge by 0.105 (Feldman–Cousins single-event lower limit, P021) and 0.38
(P012's reading of Fig. 6's lower edge); Table I's 0.30 is also tabulated in `P050_exclusion.csv`.

| window | model | E₉₀ (0.105) | E₉₅ (0.105) | E₉₀ (0.38) | E₉₅ (0.38) |
|---|---|---|---|---|---|
| LZ | any (rate defined in this window) | 62.3 | 81.0 | 17.2 | 22.4 |
| E50 = 300 | L10 / δ300 / δ350 / δ366 / δ380 | 58.0/61.4/56.6/44.7/16.3 | 75.4/79.8/73.7/58.2/21.2 | 16.0/17.0/15.6/12.4/4.5 | 20.8/22.1/20.4/16.1/5.9 |
| E50 = 400 | L10 / δ300 / δ350 / δ366 / δ380 | 49.1/54.7/31.9/12.9/2.09 | 63.8/71.1/41.5/16.8/2.72 | 13.6/15.1/8.8/3.6/0.58 | 17.6/19.6/11.5/4.6/0.75 |
| E50 = 600 | L10 / δ300 / δ350 / δ366 / δ380 | 36.2/53.1/29.0/11.2/1.81 | 47.1/69.1/37.7/14.6/2.35 | 10.0/14.7/8.0/3.1/0.50 | 13.0/19.1/10.4/4.0/0.65 |

In calendar terms: a 60 t detector with the 400 keV window excludes the 0.105 edge at 90 % in 1.0 yr (L10), 1.1 yr (δ =
300), 0.66 yr (350), 0.27 yr (366), 16 days (380); the 0.38 edge in 0.28/0.31/0.18/0.07/0.01 yr. The present generation
accumulates ≈ 31 t·yr of zero-event-capable exposure by 2030 (LZ 13.3 new + XENONnT/PandaX 4.6 existing + 13.3), enough to
exclude 0.38 at 95 % (22.4 t·yr) but not 0.105 (62 t·yr) with the LZ-like window. For δ = 380 keV, 2.1 t·yr with a
400 keV window already excludes the lower edge: LZ's existing 2.84 t·yr HE sideband, empty in the NR band (Fig. S4, P038),
does this today.

(The column `E_90_P0` in the CSV, P(0 | s+b) ≤ 0.1, is shown for completeness but is not a valid exclusion because the
low-energy bins' background alone makes zero events unlikely; CLs is the statement we use.)

## 6. Shape discrimination: L10 versus inelastic δ = 366 keV

Observed-energy pdfs p (L10) and q (δ = 366) are the efficiency-folded, resolution-smeared spectra normalised over
E_obs ≥ 100 keV within each window. The inelastic spectrum has support only between E_thr (0.1 % quantile) and E_top
(99.9 %): 160–306 keV (LZ), 161–345 (300), 167–437 (400), 168–497 keV (600). The L10 spectrum puts a fraction f_out =
0.28/0.27/0.25/0.30 of its events outside that support.

Two tests (`P050_shape_discrimination.csv`):
1. Support counting. If the inelastic model is true, zero events outside its support rejects L10 at p = (1−f_out)^N →
   N_3σ = ln(0.00135)/ln(1−f_out) = 19.9/21.4/22.8/18.4. If L10 is true, one event outside the support is fatal to the
   inelastic hypothesis, and the median L10 dataset contains one as soon as N ≥ 2–3 (P = 1 − 0.75^N).
2. Full log-likelihood ratio t = Σ ln(p/q), with both pdfs regularised by a uniform floor of weight w (w = 10⁻² mimics the
   background-to-signal ratio in the 100–200 keV bins of a 60 t detector, b/s ≈ 0.042/1.7; w = 10⁻³ is optimistic).
   KL divergences (w = 10⁻²): KL(L10‖inel) = 1.22/1.22/1.46/1.51 nats, KL(inel‖L10) = 0.52/0.60/0.89/0.95 nats. The
   Gaussian estimate N = 9σ²/(KL_pq+KL_qp)² is unreliable at N ~ 1–10; we use toys (20 000 per N; RNG seed 50): the
   smallest N whose median statistic under the true model is a ≥ 3σ exclusion of the other.

| window | reject inelastic if L10 true, N (w = 10⁻³ / 10⁻²) | reject L10 if inelastic true, N (10⁻³ / 10⁻²) | exposure at best fit (60 t): L10 true / inelastic true (t·yr, w = 10⁻²) | counting-only exposure, inelastic true |
|---|---|---|---|---|
| LZ | 3 / 4 | 10 / 10 | 11.4 / 28.4 | 56.5 |
| E50 = 300 | 3 / 4 | 8 / 9 | 10.6 / 18.3 | 43.6 |
| E50 = 400 | 3 / 4 | 5 / 6 | 9.0 / 3.5 | 13.4 |
| E50 = 600 | 3 / 3 | 5 / 5 | 5.0 / 2.6 | 9.4 |

So 3σ separation needs 4 events if L10 is true (one event below 165 keV or above the inelastic top) and 5–6 if the
inelastic model is true, in a 400 keV window; 3–4 weeks (L10 true) to 9 weeks of 60 t live time at the best-fit rates.
The LZ-like window needs 10 events for the inelastic-true direction because both spectra are truncated at 270 keV and
differ mainly through the L10 events at 100–160 keV. P057 will treat the degeneracies among the elastic operators
(L2, L4, L9–L12); here only L10 vs inelastic is addressed.

## 7. δ from the xenon spectrum

Per-event Fisher information on δ from the normalised observed spectrum (coupling free, so only the shape counts):
I(δ) = Σ_E (∂p/∂δ)²/p, with ∂p/∂δ from spectra at δ ± 2 keV. σ_δ(N) = 1/√(N I) (Cramér–Rao; recalled, certain).

| δ (keV) | window | σ_δ per event (keV) | N for ±10 keV | fraction of information in the top 10 % of the spectrum | exposure at 60 t best fit (t·yr) | live time (60 t) |
|---|---|---|---|---|---|---|
| 350 | LZ / 300 / 400 / 600 | 26.0 / 24.1 / 22.8 / 23.4 | 6.8 / 5.8 / 5.2 / 5.5 | 0.13 / 0.22 / 0.06 / 0.04 | 19.2 / 15.0 / 7.6 / 7.2 | 0.40 / 0.31 / 0.16 / 0.15 yr |
| 366 | LZ / 300 / 400 / 600 | 18.7 / 14.9 / 19.4 / 20.7 | 3.5 / 2.2 / 3.8 / 4.3 | 0.21 / 0.15 / 0.02 / 0.01 | 9.9 / 4.5 / 2.2 / 2.2 | 0.21 / 0.09 / 0.05 / 0.05 yr |
| 380 | LZ / 300 / 400 / 600 | 9.5 / 11.0 / 26.9 / 28.7 | 0.9 / 1.2 / 7.3 / 8.2 | 0.27 / 0.03 / 0.001 / 0.02 | 2.6 / 0.9 / 0.69 / 0.68 | 0.05 / 0.02 / 0.014 / 0.014 yr |

Findings. (i) One event fixes δ to 15–27 keV, consistent with P035's ±15 keV for a single XENONnT event; ±10 keV needs
2–8 events, i.e. 0.7–7.6 t·yr at the best-fit rates in a 60 t detector with a 400 keV window (7–80 days). (ii) The
information does *not* come from the kinematic endpoint E₊: even with a 600 keV window the upper 10 % of the spectrum
carries 1–4 % of it (the endpoint is populated only by the fastest halo particles and fades smoothly); it comes from the
steep low-energy onset E₋(δ) and the position of the peak. In LZ's own window the accepted sliver is very δ-sensitive per
event (σ_δ = 9.5 keV at 380) but the rate is 30× lower, so per unit exposure the extended window still wins.
(iii) Systematics: both edges depend on v_max; P018's 0.82 keV per km/s implies ±16 keV for v_esc ± 20 km/s, larger than
the ±10 keV statistical target — the xenon δ-meter is halo-limited beyond ~4 events unless the halo tail is fixed
independently (e.g. by the modulation, P034).
(iv) Tungsten ratio method (P015 numbers; P046, which will treat this in detail, is not yet in the corpus): W/Xe per
tonne = 82/222/699 at δ = 350/366/380 keV gives d ln(W/Xe)/dδ = 0.071 per keV, so ±10 keV requires the ratio to 71 %,
i.e. ≈ 4 events in each target (σ_ratio/ratio = √(2/N)). Comparable in event count, but W has the far larger rate per
tonne (16–700×, P015) and is not limited by the xenon window; the two methods share the halo systematic through v_max.

## 8. Robustness and failed approaches

- Detector-size scaling: 40 t vs 60 t changes the discovery exposures by 1–2 %; a ×10 accidentals variant changes them by
  < 3 % (the sensitive bins hold < 0.01 background events at 5σ). The 100–125 keV bin (0.10 per 2.84 t·yr, our estimate)
  contributes to L10 and δ = 300 keV only; setting it to 0.3 moves E_5σ(L10) by < 4 %.
- Halo: the 12-day annual average is used throughout; June/December rates differ by ×2.4 (300 keV) to ×∞ (380 keV, P020),
  so the exposures for δ ≥ 350 keV are means over full years; P034 gives the timing statistics.
- Edge widths: erf σ 17 keV at 400 keV vs P038's 18 keV; ±5 keV changes acceptances by < 2 %.
- First implementation solved Z(k) = 5 over all bins including those beyond the ROI edge, where the smeared signal tail is
  non-zero but the acceptance-weighted background vanishes, giving spurious Z at tiny exposures; fixed by restricting both
  s and b to bins with acceptance > 1 %.
- The Gaussian (KL-based) N for the shape test underestimates the L10-true case (N ≈ 1) because the LLR distribution is
  extremely skewed (one out-of-support event); toys are used instead.
- The Asimov Z is asymptotic; exact Poisson significances are lower by ≲ 0.3σ at N = 3–4 (P020 Table), i.e. exposures
  for 5σ may be 10–15 % larger than quoted.

## 9. Figures

- `figures/P050_fig1_spectra.png` — observed-energy spectra at the LZ best fit (per t·yr per keV, resolution folded) for
  L10 and δ = 300/350/366/380 keV, with the four efficiency windows (scaled ×2×10⁻³). The 366 and 380 keV spectra peak at
  343 keV, outside LZ's ROI; L10 continues past 600 keV.
- `figures/P050_fig2_timeline.png` — Asimov Z vs calendar year for LZ alone, the present generation (with the Sept 2026
  reanalysis step) and a 60 t detector from 2032 with a 400 keV window; solid at the best fit, dashed at ×0.30.
- `figures/P050_fig3_Z_vs_exposure.png` — Z vs exposure for the six window/detector scenarios, one panel per model.

## 10. Result files

`P050_normalisation.csv`, `P050_background_bins.csv`, `P050_background_rates.csv`, `P050_discovery.csv`,
`P050_exclusion.csv`, `P050_shape_discrimination.csv`, `P050_delta_fisher.csv`, `P050_timeline_milestones.csv`,
`P050_results.json` (everything incl. toy Z(N) curves), `P050_spectra_cache.npz` (WimPyDD spectra), `run_log.txt`.

## 11. References

LZ Collaboration, arXiv:2609.02823 (2026). G. Cowan, K. Cranmer, E. Gross, O. Vitells, Eur. Phys. J. C 71, 1554 (2011).
G. J. Feldman, R. D. Cousins, Phys. Rev. D 57, 3873 (1998). A. L. Read, J. Phys. G 28, 2693 (2002) (CLs).
D. Baxter et al., Eur. Phys. J. C 81, 907 (2021). I. Jeong, S. Kang, S. Scopel, G. Tomar, Comput. Phys. Commun. 276, 108342 (2022) (WimPyDD).
J. Aalbers et al. (XLZD), J. Phys. G 50, 013001 (2023) (next-generation xenon observatory; recalled, likely).
N. Anand, A. L. Fitzpatrick, W. C. Haxton, Phys. Rev. C 89, 065501 (2014).
Corpus: dossier 00; P004, P005, P009, P012, P013, P015, P016, P018, P019, P020, P021, P022, P034, P035, P038.

## 12. Tools and provenance (mirrors provenance/P050.json)

- Agent tools: Read ×22 (PAPER_GUIDE; dossier; ledger in three pages; papers P020, P034, P038, P021, P012, P005, P035,
  P004, P022, P010, P015; P046 requested but absent; lzcommon.py; own figures ×4). Bash ×19 (directory listings; P038/P012/
  P034/P020 cached tables; greps of P038 details, P019, tex, plan, P012/P038 scripts, WimPyDD signature, P016 results;
  WimPyDD timing/normalisation test; four script runs; WimPyDD file check; tex/P038.json check; five word-budget checks,
  two with in-place trims of the paper). Write ×5 (script, details.md, provenance JSON, paper ×2). Edit ×20 (script ×9,
  paper trims ×11). Skill ×1 (dataviz).
- Software: python 3.12.13; WimPyDD 2.0.4 (eft_hamiltonian with q-dependent coefficients, streamed_halo_function via
  lz.wd_halo with explicit v_min grid, diff_rate via lz.wd_rate; 2670 + 500 rate evaluations); numpy 2.5.3; scipy 1.18.1
  (special.erf, stats.norm/poisson, optimize.brentq); pandas 3.0.5; matplotlib 3.11.2 (Agg); common/lzcommon.py.
- Recalled knowledge (10 items): XLZD-class 40–60 t fiducial from ~2032 (uncertain); XENONnT ≈ 4 t and PandaX-4T ≈ 2.7 t
  fiducial (likely), 60 % live fraction (assumption); LZ TPC radius 72.8 cm and drift 145.6 cm (likely); LXe density
  2.86 t/m³ (likely); Asimov significance formula (certain); Poisson/CLs zero-event limits 2.303/2.996 (certain);
  Cramér–Rao bound (certain); surface-to-volume scaling of surface backgrounds (certain, geometry); Okabe–Ito palette
  (certain, cosmetic); XENONnT 3.1 and PandaX-4T 1.54 t·yr exposures (certain, via P005).
- Datasets: none. Data requests: none. WimPyDD-generated files: none (diff_rate only; verified with `find -newer`).
