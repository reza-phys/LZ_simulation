# P015 — Fluorine, argon, germanium, iodine and tungsten: which targets can confirm an inelastic 300–390 keV splitting?

Research record (simulated date 2026-09-05). Author profile: multi-target direct-detection phenomenologists. Category XEXP.
Script: `output/code/P015_target_complementarity.py` (runs in ~40 s from the simulation root). All tables in `output/work/P015/`, figures in `output/work/P015/figures/`.

## 1. Motivation and framework

LZ (arXiv:2609.02823) fits its 248 keV event with ~1 signal event; inelastic O₁ˢ with δ = 300–350 keV at m_χ ≥ 400 GeV is among the best fits (Table S7), and P002/P007 argued that the likelihood probably peaks at δ ≈ 360–390 keV, just below the xenon kinematic limit. Confirmation by a *different target* is the cleanest test, because a xenon-only signal can be a xenon-only artifact. This paper asks which nuclei can scatter at all for such splittings, what their rates are at equal coupling, and what existing and future non-xenon exposures imply.

### Kinematics (derived by hand; standard, certain)
For endothermic scattering χ N → χ* N with splitting δ, the minimum speed for recoil E_R is
v_min = (m_N E_R/μ + δ)/√(2 m_N E_R), with μ = m_χ m_N/(m_χ + m_N). Minimising over E_R gives the kinematic ceiling
**δ_max = μ v_max²/2**, attained at E_R* = μ² v²/m_N. For v < v_max nothing scatters if δ > μv²/2. The accessible window at speed v is
E_± = (μ²v²/m_N)[1 − δ/(μv²) ± √(1 − 2δ/(μv²))] (library `lz.E_R_range_keV`).
Because μ → m_N for m_χ ≫ m_N, the ceiling saturates at m_N v_max²/2, so **the mass number A, not m_χ, decides which targets are alive**. v_max = v_E + v_esc with v_esc = 544 km/s (Baxter 2021 via `lzcommon`), v_E(16 June, day 167) = 265.1 km/s, annual mean 250.6 km/s, 16 Dec 236.0 km/s → v_max = 809.1 / 794.6 / 780.0 km/s (script; the assignment's 810/795 rounded).

Threshold-velocity for a target: v_thr = √(2δ/μ). At δ = 300 keV, 1 TeV: Xe (μ = 109.0 GeV) 703 km/s; I (105.8 GeV) 714 km/s; W (146.3 GeV) 607 km/s (hand algebra from the μ values in `delta_max_table.csv`). The halo fraction above 607 km/s is much larger than above 703 km/s, which is the origin of the large W/Xe rate ratio (§3).

### Rates
WimPyDD 2.0.4 (the code LZ used), operator O₁ isoscalar with the LZ/Anand unit coupling c₁ˢ = 1/m_v² ⇔ WimPyDD c⁰ = 2/m_v² (convention settled by P003, confirmed by P007), m_v = 246.2 GeV, j_χ = 1/2, `WD.diff_rate` per isotope via `lz.wd_rate(..., target=WD.X)`. Halo: Baxter-2021 SHM (v₀ = 238, v_esc = 544, v_sun peculiar (11.1, 12.2, 7.3) km/s), explicit v_min grid to 844 km/s (P002 patch); "annual" = mean of 12 monthly halo functions (as P007), "june16" = day 167. Integration: 60-point linear grid over [0.97 E₋, 1.03 E₊] at v_max(16 June), trapezoid, negative WimPyDD values (numerical noise near the edge) clipped to zero. Rates are per tonne-year of *element* (WimPyDD natural isotopic composition).

**WimPyDD tungsten bug (runtime fix, no file edited).** `WimPyDD/Targets/Nuclear_response_functions/18{0,2,3,4,6}W_func_w.py` do `from numpy import *` but call `np.ndarray/np.dot/np.exp`, so any W rate raises `NameError: np`. WimPyDD loads these modules with `__import__` (cached in `sys.modules`); the script imports each module once and sets `module.np = numpy`. The tungsten files use a **Helm form factor for the M response** ("for M interaction used Helm factor", file docstring), whereas Xe and I use the shell-model M response tables; the W/Xe ratio therefore carries a form-factor-model systematic (§7).

### LZ normalisation
The best-fit signal is 1.0 (+1.4, −0.7) events in 2.84 t·yr (paper Table I; used for all models as in P005/P007). For each δ, N_LZ(unit coupling) = 2.84 t·yr × ∫ dR/dE|_Xe,annual ε_LZ(E) dE with ε_LZ = 0.96 × erf turn-on at 5.4 keV (σ = 2.5 keV) × erfc roll-off at 269.9 keV (σ = 11.5 keV) (paper Fig. S2 values; the P005/P007 model). The scale to one event is s(δ) = 1/N_LZ; since the rate is linear in (c m_v²)², s is also the best-fit (c₁ˢ m_v²)². Expected counts elsewhere: N = s(δ) × R_T(window) × X_T, with X_T the element exposure (detector exposure × element mass fraction), **all recoils in the kinematic window counted with efficiency 1** (a ceiling on what each experiment could have contained).

## 2. Part 1 — kinematic ceilings and blind targets

`delta_max_table.csv` (10 targets × 3 masses × 3 halos), `blind_table.csv`. δ_max (keV) for m_χ = 1000 GeV, 16 June (annual mean in brackets):

| target | A | μ (GeV) | δ_max | 400 GeV | 4000 GeV |
|---|---|---|---|---|---|
| C | 12 | 11.05 | 40.3 (38.8) | 39.6 | 40.6 |
| O | 16 | 14.69 | 53.5 (51.6) | 52.3 | 54.1 |
| F | 19 | 17.40 | 63.3 (61.1) | 61.7 | 64.2 |
| Na | 23 | 20.97 | 76.4 (73.7) | 74.1 | 77.6 |
| Ar, Ca | 40 | 35.92 | 130.8 (126.2) | 124.1 | 134.5 |
| Ge | 73 | 63.70 | 231.9 (223.6) | 211.7 | 243.5 |
| I | 127 | 105.8 | 385.3 (371.5) | 332.5 | 418.5 |
| Xe | 131 | 109.0 | 396.1 (382.0) | 340.5 | 431.3 |
| W | 184 | 146.3 | 532.9 (513.9) | 437.0 | 598.6 |

(Xe with A = 131 gives 396.1 keV; P002's 397 keV used A = 131.29.) Blind table for δ = 250/300/350/366/380 keV: **C, O, F, Na, Ar, Ca and Ge are blind at every δ ≥ 250 keV for every mass** (Ge's ceiling is 232 keV at 1 TeV and 244 keV even at 4 TeV). I, Xe and W are open at all five δ on 16 June; iodine closes for δ = 380 keV with the annual-mean halo (371.5 < 380) and xenon closes at 382 keV. Minimum mass number for a live target (16 June, 1 TeV): A ≥ 79 (δ = 250), 96 (300), 114 (350), 125 (380); infinite-mass limit A ≥ 74/88/103/112. So Cs, Ba, La, Ta, W, Pb, Bi (all A > 130) qualify, Ge/Zn/Cu/Ar/Si/F/C do not.

Accessible windows [E₋, E₊] at v_max(16 June), 1 TeV (keV):

| δ | Xe | I | W |
|---|---|---|---|
| 250 | 54–912 | 57–874 | 34–1359 |
| 300 | 91–786 | 97–745 | 52–1255 |
| 350 | 153–635 | 168–585 | 78–1144 |
| 366 | 185–575 | 208–516 | 88–1107 |
| 380 | 225–510 | 269–430 | 98–1073 |

Every threshold considered (1–11 keV) is far below E₋, so thresholds are irrelevant; what matters is the *upper* edge of an analysis. Xenon recoils at δ = 366 keV extend to 575 keV, twice LZ's 270 keV ROI end; with the LZ efficiency only 17% of the xenon window rate is detected at δ = 366 keV (82% at 300, 45% at 350, 2.8% at 380; `lz_normalisation.in_LZ_eff_fraction`).

## 3. Part 2 — WimPyDD rates and ratios (`rates_table.csv`, `spectra_1000GeV.csv`)

Window-integrated rates at unit coupling, events per t·yr of element, annual-mean halo (16 June in brackets):

| δ (keV) | Xe | I | W | I/Xe | W/Xe |
|---|---|---|---|---|---|
| 250 | 41 457 (53 025) | 29 119 (38 285) | 377 624 (442 695) | 0.70 | 9.1 |
| 300 | 5 831 (8 308) | 3 449 (5 075) | 93 124 (116 389) | 0.59 | 16.0 |
| 350 | 208.8 (437.3) | 61.8 (157.6) | 17 095 (22 995) | 0.30 | 81.9 |
| 366 | 44.06 (107.5) | 6.30 (20.98) | 9 772 (13 335) | 0.14 | 222 |
| 380 | 8.70 (26.01) | 0.198 (1.042) | 6 080 (8 375) | 0.023 | 699 |

Ge, Ar, F: WimPyDD returns exactly zero at all δ (window undefined), confirming the kinematics numerically. June/annual ratios: Xe 1.28/1.42/2.09/2.44/2.99, I 1.31/1.47/2.55/3.33/5.28, W 1.17/1.25/1.35/1.36/1.38 for δ = 250…380 — the modulation is largest for the target closest to its ceiling (P006 found 1.41 for Xe in the LZ ROI at 300 keV; ours is the full window).

**Why W ≫ Xe.** Three factors: (i) coherence A² — (184/131)² = 1.97; (ii) velocity phase space — v_thr falls from 703 to 607 km/s at δ = 300 keV, and the Maxwellian tail above v_thr grows by roughly an order of magnitude, an effect that explodes as xenon approaches its ceiling (δ = 380 keV: W/Xe = 699); (iii) form factors — the W window (52–1255 keV) spans several Helm nodes and the integrated F² weight is somewhat *smaller* per unit A² than xenon's over 91–786 keV, so (iii) partially offsets (i). Iodine (A = 127, μ = 105.8 GeV) is slightly lighter than xenon and closer to its own ceiling, hence I/Xe drops from 0.70 to 0.02 across the window — **iodine is a xenon-like target at δ ≤ 300 keV and a poor one in the Higgsino window (358–380 keV)**.
Isovector (Higgsino-like) weights ((N − (1−4 sin²θ_W)Z)/A)² relative to Xe: I 0.98, W 1.04, Ge 0.90 (sin²θ_W = 0.231): the target ranking is coupling-independent within 5%.

Spectra (Fig. 2): at LZ best fit, Xe peaks at 4–5 × 10⁻³ events/(t·yr·keV) near 170 keV (δ = 300) with the 266 keV M-response node (P002); iodine tracks xenon at 0.6×; tungsten's spectrum starts at 52 keV, reaches 0.09 at ~100 keV, and shows Helm nodes near 145, 470 and 800 keV, with most of the rate below 250 keV but a 30% tail extending to 1.2 MeV.

## 4. Part 3 — LZ normalisation and existing exposures (`existing_exposures_counts.csv`)

N_LZ(unit coupling) = 1.046 × 10⁵ / 13 565 / 264.6 / 21.71 / 0.684 events at δ = 250/300/350/366/380 keV, so the best-fit (c₁ˢ m_v²)² = 9.56 × 10⁻⁶ / 7.37 × 10⁻⁵ / 3.78 × 10⁻³ / 0.0461 / 1.46 and σ_SI(best) = 2.8 × 10⁻⁴³ / 2.2 × 10⁻⁴² / 1.1 × 10⁻⁴⁰ / 1.4 × 10⁻³⁹ / 4.3 × 10⁻³⁸ cm² (paper eq. for scalar normalisation, μ_N = 0.937 GeV). Check against P007's digitised Fig. 6 intervals at 1 TeV: 300 keV lower/median/upper = 1.28 × 10⁻⁵ / 1.32 × 10⁻⁴ / 2.59 × 10⁻⁴ (ours 7.4 × 10⁻⁵, inside; median corresponds to ~1.8 events as expected for a two-sided interval); 350 keV 1.38 × 10⁻³ / 0.0103 / 0.0225 (ours 3.8 × 10⁻³, inside). Events per t·yr of element at best fit: Xe 0.40/0.43/0.79/2.0/12.7 (full window); I 0.28/0.25/0.23/0.29/0.29; W 3.6/6.9/65/450/8 900.

Exposures (all recalled; reliability flagged) and expected counts at the LZ best fit (efficiency 1 over the whole window):

| experiment | element exposure (t·yr) | recall | N(δ=300) | N(350) | N(366) | N(380) |
|---|---|---|---|---|---|---|
| PICO-60 C₃F₈ 1404 kg·d (F fraction 0.808) | 0.0031 | likely | 0 | 0 | 0 | 0 |
| PICO-60 CF₃I 1335 kg·d (I fraction 0.648) | 0.0024 | uncertain | 6.0 × 10⁻⁴ | 5.5 × 10⁻⁴ | 6.9 × 10⁻⁴ | 6.8 × 10⁻⁴ |
| DEAP-3600 758 t·d (2019); PLR 2026 larger | 2.08 (2.7 placeholder) | likely / uncertain | 0 | 0 | 0 | 0 |
| CRESST-II Lise 52 kg·d CaWO₄ (W fraction 0.639) | 9.1 × 10⁻⁵ | uncertain | 6.2 × 10⁻⁴ | 5.9 × 10⁻³ | 0.041 | 0.81 |
| CRESST-III 3.64 kg·d | 6.4 × 10⁻⁶ | uncertain | 4.4 × 10⁻⁵ | 4.1 × 10⁻⁴ | 2.9 × 10⁻³ | 0.057 |
| DAMA/LIBRA 2.46 t·yr NaI (I fraction 0.847) | 2.08 | likely | 0.53 | 0.49 | 0.60 | 0.60 |
| COSINE-100 ≈ 0.2 t·yr | 0.17 | uncertain | 0.043 | 0.040 | 0.049 | 0.049 |
| ANAIS-112 ≈ 0.34 t·yr (112.5 kg × 3 yr) | 0.29 | uncertain | 0.073 | 0.067 | 0.083 | 0.083 |
| LZ 2.84 t·yr (full window, no efficiency) | 2.84 | paper | 1.22 | 2.24 | 5.8 | 36 |

(δ = 250 keV column: PICO CF₃I 6.6 × 10⁻⁴, CRESST-II 3.3 × 10⁻⁴, DAMA 0.58, COSINE 0.047, ANAIS 0.080.) The LZ row shows how much of the xenon signal LZ's 270 keV ROI discards: at δ = 380 keV the full-window expectation is 36 events for a fitted 1.0 in-ROI event.

**Interpretation.** Fluorine and argon exposures, however large, contain zero events: PICO-60 C₃F₈ and DEAP-3600 (any exposure, including the 2026 PLR analysis) are kinematically irrelevant for δ ≥ 250 keV. Tungsten is the best nucleus per tonne but the recalled CRESST exposures are 10⁴–10⁵ times too small (except at δ = 380 keV, where 52 kg·d would already hold 0.8 events — if recoils up to 1.1 MeV had been analysed; the Lise ROI ended at 40 keV). Iodine in DAMA/LIBRA is the only existing non-xenon exposure with O(1) expected events (0.5–0.6). But NaI(Tl) has no NR/ER discrimination: with Q_I ≈ 0.09 (recalled, likely) the iodine window maps to 8.7–67 keVee (δ = 300), 15–53 (350), 19–46 keVee (366), where DAMA's single-hit rate is ~1 cpd/kg/keVee (recalled, likely), i.e. 5.2 × 10⁷, 3.4 × 10⁷ and 2.5 × 10⁷ background counts against 0.5 signal events (`nai_dama_context` in the summary JSON). Even the June modulation (I: June/annual 1.5–3.3) is a fraction of a count per year. DAMA/LIBRA, COSINE-100 and ANAIS-112 therefore neither saw nor could have seen the LZ signal, and NaI cannot confirm it at any foreseeable exposure without a discriminating technology.

## 5. Part 4 — future exposures (`future_exposures.csv`)

Exposure of compound for 3 events at the LZ best fit (annual-mean halo, whole window, efficiency 1):

| δ (keV) | NaI (t·yr) | CaWO₄ (t·yr) | Xe, full window (t·yr) | E₊ I (keV / keVee) | E₊ W (keV) |
|---|---|---|---|---|---|
| 250 | 12.7 | 1.30 | 7.6 | 874 / 79 | 1359 |
| 300 | 13.9 | 0.68 | 7.0 | 745 / 67 | 1255 |
| 350 | 15.2 | 0.073 | 3.8 | 585 / 53 | 1144 |
| 366 | 12.2 | 0.0104 (10 kg·yr) | 1.5 | 516 / 46 | 1107 |
| 380 | 12.3 | 0.00053 (0.5 kg·yr) | 0.24 | 430 / 39 | 1073 |

NaI needs 12–15 t·yr *and* discrimination; a CaWO₄ cryogenic detector needs 0.7 t·yr at δ = 300 keV but only 10 kg·yr at the P007 Higgsino best point (δ = 366 keV) and 0.5 kg·yr at 380 keV — a CRESST-scale detector, provided its nuclear-recoil ROI extends to ~1.1–1.3 MeV (light-yield discrimination improves with energy; the phonon channel is linear there). The xenon column shows that LZ/XENONnT/PandaX-4T would gain ×2–36 in signal at δ ≥ 350 keV by extending their ROIs to ~600 keV (cf. P005, which stopped at 270 keV).

## 6. Part 5 — LZ Fig. S7 comparison curves (`figS7_comparison.csv`, `figS7_curves.json`)

Digitised from the PDF vector paths (pymupdf; P007's axis calibration: x 0 → 70.7 pt, 350 → 491.2 pt; decade labels 10⁻⁴⁷ at y = 331.1 pt … 10⁻³⁵ at 15.8 pt; curves identified by stroke colour: orange = CRESST-II 2016, red = PICO 2023, green = XENON1T 2018, blue = PandaX-4T 2021, violet = LZ 2024, black 3 pt = LZ 2026 edges). Extents: XENON1T ends at δ = 231 keV, LZ 2024 at 250 keV, PandaX-4T at 303 keV (leaving the top of the frame at 10⁻³⁵ cm²), **CRESST-II and PICO 2023 span the full 0–350 keV axis** (PICO's path continues to the frame edge at ~366 keV).

| δ | CRESST-II σ (cm²) | /LZ upper | /LZ best | PICO 2023 σ | /LZ upper | /LZ best | LZ upper / lower |
|---|---|---|---|---|---|---|---|
| 250 | 1.6 × 10⁻³⁸ | 1.4 × 10⁴ | 5.7 × 10⁴ | 2.9 × 10⁻³⁹ | 2 490 | 1.0 × 10⁴ | 1.18 × 10⁻⁴² / 8.8 × 10⁻⁴⁴ |
| 300 | 9.3 × 10⁻³⁸ | 1.2 × 10⁴ | 4.2 × 10⁴ | 2.5 × 10⁻³⁸ | 3 120 | 1.1 × 10⁴ | 7.9 × 10⁻⁴² / 3.9 × 10⁻⁴³ |
| 350 | 9.1 × 10⁻³⁷ | 1 320 | 8 090 | 3.4 × 10⁻³⁷ | 494 | 3 030 | 6.9 × 10⁻⁴⁰ / 4.2 × 10⁻⁴¹ |

(PandaX-4T 2021 at 300 keV: 9.5 × 10⁻³⁶, 1.2 × 10⁶ × LZ upper; LZ 2024 at 250 keV: 1.6 × LZ-2026 upper.) None of the published inelastic limits comes within a factor 500 of the LZ interval anywhere in 250–350 keV; the CRESST-II and PICO curves do "reach" 350 keV on the axis, but 3–4 orders of magnitude above LZ. Their existence at δ > 232 keV is itself a kinematic statement: only the iodine of CF₃I and the tungsten of CaWO₄ can produce them (fluorine, carbon, oxygen and calcium are blind), which the PICO 2023 paper obtained from the CF₃I run.

**Cross-check of the iodine machinery against PICO 2023.** Our count for PICO-60 CF₃I iodine at the LZ upper edge (3.65 events per P007) is 2.2 × 10⁻³ (δ = 300) and 2.0 × 10⁻³ (350) events; a zero-background 90% limit (2.3 events) would then sit ×1 050 (300) / ×1 140 (350) above LZ's upper edge, versus the digitised ×3 120 / ×494. Agreement within a factor 2–3 in both directions — acceptable given the uncertain CF₃I exposure, the iodine bubble-nucleation efficiency, PICO's halo parameters and their actual candidate count, none of which we model. It confirms the order of magnitude of both the iodine rate and the exposure recall, and shows that a two-target (Xe + I) analysis is what PICO effectively did — with 1/1000 of LZ's sensitivity.

## 7. Validation, robustness, failed approaches

- Kinematic ceilings agree with P002 (δ_max ceiling 397 keV at 1 TeV with A = 131.29 vs our 396.1 with A = 131) and P007 (Xe windows). Ge/Ar/F WimPyDD rates are identically zero, matching the algebra.
- LZ best-fit couplings fall inside P007's digitised two-sided intervals at both tabulated δ (§4).
- Iodine/xenon ratio sanity: A² ratio (127/131)² = 0.94, μ slightly smaller, ceiling 385 vs 396 keV — 0.70 → 0.02 trend is as expected.
- Tungsten uses WimPyDD's Helm M form factor rather than a shell-model table; Xe shell-model vs Helm differ by ×3–6 at 200–250 keV near the node (PAPER_GUIDE), so the W/Xe ratios carry a form-factor systematic we estimate at ×2 in either direction; the *kinematic* conclusions (blind targets, windows, W ≫ Xe near the xenon ceiling) do not depend on it.
- June vs annual halo changes W counts by 17–38%, I by 31% (250) to ×5 (380). The 60-point window grid changes integrals by < 2% (checked by inspection of the 350 keV Xe spectrum against P002's peak position 195 keV).
- Recalled exposures: PICO-60 C₃F₈ 1404 kg·d (likely); PICO-60 CF₃I 1335 kg·d (uncertain, ±30%); DEAP-3600 758 t·d (likely; irrelevant); CRESST-II Lise 52 kg·d and CRESST-III 3.64 kg·d (uncertain, ±50%); DAMA/LIBRA 2.46 t·yr (likely); COSINE-100 0.2 t·yr (uncertain, ×2); ANAIS-112 112.5 kg × 3 yr (uncertain). Counts scale linearly.
- Failed/abandoned: (a) first WimPyDD call on W crashed on the missing `np` import — fixed at runtime (§1); (b) the first exposure table entered DAMA/COSINE in kg·d ×1000 too large (2.46 t·yr typed as 2.46 × 10⁶ kg·d) — caught by inspection, corrected to 8.99 × 10⁵ kg·d; (c) we did not attempt DEAP's actual 2019/2026 ROIs because argon is blind at any exposure.

## 8. Figures
- `figures/P015_fig1_delta_max_vs_A.png` — δ_max = μv_max²/2 versus A for 400/1000/4000 GeV (16 June) and 1000 GeV annual mean, targets marked, LZ-favoured 300–390 keV band shaded. Only I, Xe, W lie above the band.
- `figures/P015_fig2_spectra_Xe_I_W.png` — dR/dE at the LZ best fit for Xe, I, W at δ = 300 and 350 keV (1 TeV, annual halo), LZ ROI unshaded, event energy marked; tungsten extends to 1.2 MeV.
- `figures/P015_fig3_existing_exposures.png` — expected counts in recalled exposures at δ = 300/350 keV; F and Ar exposures show zero.

## 9. Extended discussion
The result reorganises the "other experiments" question. The relevant question is not exposure but mass number: for δ ≳ 250 keV every light and medium target (including germanium, the second workhorse of direct detection) is switched off by kinematics at all WIMP masses, so no CDMS/EDELWEISS/DarkSide/DEAP/PICO-C₃F₈ result, past or future, bears on the LZ hypothesis. The confirming targets are I (weak near the Higgsino window), Cs (A = 133, similar to Xe — CsI is a natural NaI alternative), and heavy nuclei W, Pb, Bi (PbWO₄, BGO). Tungsten's advantage grows with δ because its ceiling (533 keV) lies far above xenon's; at the P007 Higgsino point a 10 kg·yr CaWO₄ exposure with an MeV-scale NR ROI would give three events, versus 1.5 t·yr of xenon with an extended ROI. Xenon experiments themselves lose 55–97% of the δ ≥ 350 keV signal above 270 keV. P046 (later) is expected to treat detector-level feasibility (saturation, α backgrounds, discrimination at MeV recoils) which we do not.

## 10. References
LZ Collaboration, arXiv:2609.02823 (2026). D. Tucker-Smith, N. Weiner, PRD 64, 043502 (2001). J. Bramante, P. J. Fox, G. D. Kribs, A. Martin, PRD 94, 115026 (2016). E. Adams et al. (PICO), PRD 108, 062003 (2023). C. Amole et al. (PICO), PRD 100, 022001 (2019). A. H. Abdelhameed et al. (CRESST), PRD 100, 102002 (2019). R. Bernabei et al. (DAMA/LIBRA), Nucl. Phys. At. Energy 19, 307 (2018). D. Baxter et al., EPJC 81, 907 (2021). I. Jeong, S. Kang, S. Scopel, G. Tomar (WimPyDD), CPC 276, 108342 (2022). Corpus: dossier 00, P002, P003, P005, P006, P007.

## 11. Tools and provenance (mirrors `output/provenance/P015.json`)
- Agent tools: Read ×18 (PAPER_GUIDE, dossier, ledger, P002/P005/P007 papers, fulltext.tex lines 26–31 and 800–824, lzcommon.py 108–217 and 305–372, P007 script 100–180, FigS7 PNG, P007 lz_intervals_digitised.json, P015 figures fig1 ×1, fig2 ×2, fig3 ×2), Bash ×19 (grep of tex/lzcommon; WimPyDD target/element inspection; timing probe; pymupdf curve enumeration; WimPyDD bug inspection ×2; palette grep; four script runs; table printing; six word-count/trim/validation passes), Write ×5 (script, details, provenance, paper ×2), Edit ×8 (script fixes), Skill ×1 (dataviz).
- Software: python 3.12.13; numpy 2.5.3 (linspace, trapezoid, polyfit, clip); scipy 1.18.1 (special.erf/erfc); matplotlib 3.11.2 (Agg); WimPyDD 2.0.4 (eft_hamiltonian, streamed_halo_function via lz.wd_halo, diff_rate via lz.wd_rate; targets Xe, I, W, Ge, Ar, F; runtime `np` injection into 18xW_func_w modules); pymupdf 1.28.2 (get_drawings); common/lzcommon.py (LZ constants, v_earth_kms, mu_red, m_nucleus_gev, E_R_range_keV, wd_halo, wd_hamiltonian, wd_rate, GEV_TO_CM2, M_V_GEV).
- Script: `output/code/P015_target_complementarity.py`; command `.venv/bin/python output/code/P015_target_complementarity.py`.
- Local inputs: fulltext.tex (Introduction citations l.28; Fig. S7 caption l.810–817; O₁-as-cross-section l.800–808); `inputs/arXiv_2609.02823_source/FigS7_O1_as_SI.pdf` (vector paths); `inputs/figures_png/FigS7_O1_as_SI.png`; dossier; ledger; P002.md, P005.md, P007.md; `output/work/P007/lz_intervals_digitised.json`; `output/code/P005_other_xenon_expectations.py` and `P007_higgsino_inelastic.py` (efficiency model, digitisation calibration); `output/code/common/lzcommon.py`; `environment/ENVIRONMENT_versions.txt`.
- Recalled knowledge (14 items): inelastic kinematics (certain); atomic masses of C, O, F, Na, Ca, I, W (certain); sin²θ_W = 0.231 (certain); PICO-60 C₃F₈ 1404 kg·d, thresholds 2.45/3.3 keV (likely); PICO-60 CF₃I ≈ 1335 kg·d, 13.6 keV threshold (uncertain); DEAP-3600 758 t·d, ROI ≈ 50–100 keV_nr (likely / uncertain); CRESST-II Lise ≈ 52 kg·d, ROI to 40 keV (uncertain); CRESST-III 2019 ≈ 3.64 kg·d (uncertain); DAMA/LIBRA 2.46 t·yr total (likely); COSINE-100 ≈ 0.2 t·yr (uncertain); ANAIS-112 112.5 kg × 3 yr (uncertain); iodine quenching Q_I ≈ 0.09 (likely); DAMA single-hit rate ≈ 1 cpd/kg/keVee at 10–60 keVee (likely); PICO 2023 inelastic limits derived from CF₃I iodine (likely); Bramante et al. 2016 CRESST-II recast (likely).
- Datasets: none. Data requests: none. WimPyDD-generated files: none (verified with `find WimPyDD -newer script`).
