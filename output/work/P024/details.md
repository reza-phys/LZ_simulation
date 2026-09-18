# P024 — How wide is the NR band at 250 keV? Sensitivity of the "1.5σ below the median" statement to the yield break and fluctuation model

Simulated date 2026-09-08 · category RESP · physics.ins-det · author profile: NEST/LXe microphysics modellers.
Script: `output/code/P024_nr_band_width.py` (run from the root with `.venv/bin/python`; 20 s; seeds: numpy 20260908, nestpy 20260908).
Machine-readable results: `output/work/P024/results.json`; tables `band_variants.csv`, `digitised_NR_lines_Fig{2,4}*.csv`, `ambe_points_fig2.csv`; console log `run_log.txt`; figures under `figures/`.

## 1. Motivation and framework

LZ (arXiv:2609.02823) places its candidate (S1c = 540.1 phd, S2c = 9268 phd, log₁₀S2c = 3.9670) "1.5σ below the median of the modeled NR band at that S1c" (Data Analysis paragraph, tex l. 165). The same paper draws the NR band in Figs. 2 and 4 as the median and the 10–90 % percentiles of the NEST v2.4.5 model for a flat spectrum, evaluated at fixed S1c (Fig. 2 caption, l. 103). P009 found that these two statements disagree: its NEST-MC width (σ = 0.033 dex) reproduces 1.5σ, whereas the drawn 10–90 % lines (σ_eq = 0.022 dex) put the event 2.5σ low; P010 likewise read 2.2 Gaussian σ from the drawn lines. Since P(NR-like | genuine NR) changes from 7 % to 0.6 % between these readings, and the paper itself notes that "existing NR calibration data around 250 keV are limited" (l. 691) and that the NR charge-yield model needed a new power-law break above E₀ = 74.7 keV to accommodate AmBe (l. 600–606), we ask what physically sets the width, how it depends on every tunable ingredient, what LZ's own AmBe points say, and how the NR PDF overlaps the MSSI containment regions.

All numbers below are produced by the script unless marked [paper] or [recalled].

## 2. Model

### 2.1 Quanta (nestpy 2.1.1, NEST v2 API)
For an NR of energy E, `NESTcalc.GetYields(NR, E, ρ = 2.9, F = 96.5 V/cm, A = 131.293, Z = 54, params)` gives mean (N_ph, N_e) with the LZ Table S5 parameters and the p(E) break implemented in `lzcommon.nest_nr_params_vector` (p = 0.5 + a ln[1 + b(E − E₀)], a = 0.0230, b = 0.0289, E₀ = 74.7 keV [paper]). Fluctuations come from `GetQuanta(yields, ρ, W)` with W = `LZ_WS2024.nr_er_width_parameters` = [0.404, 0.393, 0.0383, 0.497, 0.1906, 2.22, 0.3, 0.04311, 0.15505, 0.46894, −0.26564, 0, 0]. From the NEST source [recalled, likely] and verified numerically here: N_i ~ Normal(N_q/(1+N_ex/N_i), √(W₀ N_i)), N_ex ~ Normal(N_i·N_ex/N_i, √(W₁ N_ex)), recombination probability r = 1 − (1 + N_ex/N_i)·N_e/N_q fixed from the means, and

  Var(N_e) = r(1−r) N_i + ω² N_i²,  ω = W₂ exp[−(f_e − W₃)²/(2W₄²)],  f_e = N_e/N_q,

with a skew-normal draw of shape W₅ = 2.22. Entries 6–10 are ER parameters; entries 11–12 are additional NR terms (zero in the LZ set). The roles were confirmed empirically by the ±30 % scan (§4.2).

Energy scale. P009 showed that Table S5 as printed gives S1c = 498 phd at 248 keV while the paper's own constant-energy contours imply N_q = 11.32 E^1.112 (S1c = 539 phd at 248 keV). Our baseline ("paper scale") shifts the photon count by the mean difference so that the MC reproduces the paper's S1c(E) relation while keeping the LZ charge yield and all fluctuations; "Table S5 as printed" is carried as a variant. The width at fixed S1c is insensitive to this choice (0.0313 vs 0.0312 dex); the median is not.

### 2.2 Detector
S1c = Binomial(N_ph, g₁ = 0.110) × (1 + N(0, 0.338/√n₁)) × (1 + N(0, σ_pos)); S2c = Binomial(N_e, ε_ext) × (g₂/ε_ext) × (1 + N(0, √(F_SE/S2))) × (1 + N(0, σ_pos)); g₂ = 34.5 phd/e [paper], F_SE = 4.0 and sPEres = 0.338 from the nestpy LZ_WS2024 object, ε_ext = 0.80 [recalled: LZ SR1 80.5 %, uncertain; nestpy's `CalculateG2` gives 0.726, carried as a variant], σ_pos = 2 % [assumption, as P009; 0/3/6 % variants]. Double-PE emission (P_dphe = 0.214 in the nestpy object) is absorbed into g₁ (phd/photon) and not simulated separately; it would add ≈ √(P(1−P)/n₁) ≈ 1.8 % to S1c only and does not touch log₁₀S2c.

### 2.3 Fixed-S1c band
Flat spectrum E ∈ [200, 320] keV (300 000 draws; 150 000 for the ±30 % scan), 1-keV bins; the slice |S1c − 540.1| < 10 phd retains ≈ 19 700 events with mean energy 249.0 ± 10.7 keV. Statistics: percentiles, sd, 16–84 half-width, 10–90 half-width, Gaussian-equivalent σ = (p90 − p10)/(2 × 1.2816), skewness, event offset Δ = p50 − 3.9670, σ-distance Δ/sd, and P(log₁₀S2c ≤ 3.9670) directly from the samples. MC statistical precision on sd: ±0.5 %.

## 3. Results — variance decomposition at 248 keV (script §2)

At E = 248 keV (paper scale): N_ph = 4897, N_e = 301.1, N_q = 5198, N_i = 2012, N_ex = 2814 (N_ex/N_i = 1.399), r = 0.8503, f_e = 0.0624, ω = 0.00285. Check of the recalled NEST form: predicted sd(N_e) = √(r(1−r)N_i + ω²N_i² + (1−r)²W₀N_i) = 17.52 e vs MC 17.56 e; sd(N_i) predicted 28.5 vs MC 28.4. The MC therefore follows the stated variance form to 0.3 %.

| term | σ(log₁₀S2c) [dex] | share of variance |
|---|---|---|
| recombination binomial r(1−r)N_i | 0.0231 | 54 % |
| extraction binomial (ε = 0.80) | 0.0125 | 16 % |
| position residual 2 % | 0.0087 | 7.7 % |
| SE-size Fano (F = 4) | 0.0085 | 7.4 % |
| recombination ω²N_i² | 0.0083 | 7.0 % |
| ion Fano W₀ → N_e | 0.0062 | 3.9 % |
| energy mixing in the S1c slice (σ_E = 10.6 keV × 0.00058 dex/keV) | 0.0061 | 3.8 % |
| **quadrature sum** | **0.0313** | |
| MC (fixed S1c, baseline) | 0.0313 | |
| MC quanta only (perfect detector) | 0.0254 (analytic 0.0253) | |

The band is dominated by the binomial statistics of the ≈ 300 escaping electrons: σ_binom = 0.434 √(r/N_e). The ω term, which is what `nr_er_width_parameters` mostly tunes, is sub-dominant because its Gaussian is centred at f_e = 0.497 while an NR at 250 keV has f_e = 0.06. There is no gain-spread term beyond SE Fano and the position residual; an electron-lifetime/g₂ drift would enter as an extra position-like smearing (variants 3 %/6 %).

**Floor.** For fixed N_e = 300 and any physically allowed r ≤ 0.9 the recombination term alone is ≥ 0.021 dex; sub-binomial recombination would require anti-correlated escape and is not observed in LXe [recalled, likely]. Pure Poisson on the model's N_e = 10^4.016/34.5 = 300 e gives 0.0250 dex (10–90 half-width 0.032 dex).

## 4. Results — fixed-S1c band variants (script §1; `band_variants.csv`, Fig. `P024_band_width_variants.png`)

### 4.1 Baseline
median 4.0152 (drawn median in Fig. 4: 4.0162), sd 0.0313 dex, 16–84 half-width 0.0308, 10–90 half-width 0.0399 (σ_eq 0.0311), skewness +0.09; event offset 0.0482 dex → **1.54σ**; P(log₁₀S2c ≤ 3.967) = **5.65 %** (Gaussian 6.2 %). Fixed energy 248 keV (no slice): sd 0.0306, P = 5.27 % (unconditional and conditional on the slice); 240/256/265 keV: 7.3/3.7/2.5 %.

### 4.2 Variants (sd in dex; σ-distance of the event; P(≤ observed))

| variant | sd | median | Δ/sd | P(≤obs) |
|---|---|---|---|---|
| baseline (paper scale, LZ widths, ε = 0.80, σ_pos 2 %) | 0.0313 | 4.0152 | 1.54 | 0.057 |
| Table S5 as printed (S1c = 540 ↔ 266 keV) | 0.0312 | 4.0250 | 1.86 | 0.027 |
| no p(E) break | 0.0332 | 4.134 | 5.0 | <1e-5 |
| NEST default yields | 0.0320 | 4.172 | 6.4 | <1e-5 |
| field 90 / 100 V/cm | 0.0311 / 0.0314 | | 1.51 / 1.57 | |
| W₀ (ion Fano) ×0.7 / ×1.3 | 0.0308 / 0.0315 | | 1.59 / 1.53 | |
| W₁ (exciton Fano) ×0.7 / ×1.3 | 0.0317 / 0.0316 | | 1.51 / 1.52 | |
| W₂ (ω amplitude) ×0.7 / ×1.3 | 0.0308 / 0.0321 | | 1.56 / 1.50 | |
| W₃ (f_e centre) ×0.7 / ×1.3 | **0.0477** / 0.0302 | | 0.98 / 1.61 | 0.152 / |
| W₄ (f_e width) ×0.7 / ×1.3 | 0.0305 / **0.0383** | | 1.59 / 1.22 | / 0.101 |
| W₅ (skew) ×0.7 / ×1.3 | 0.0317 / 0.0313 | | 1.56 / 1.52 | |
| W₆–W₁₀ (ER entries) ×0.7 / ×1.3 | 0.0309–0.0316 | | 1.53–1.56 | |
| W₁₁ = +0.3 (v2.4 extra NR term, zero in LZ) | 0.195 | 4.002 | 0.18 | 0.43 |
| W₁₂ = +0.3 | 0.0351 | | 1.42 | |
| g₁ −2 % / +2 % | 0.0311 / 0.0315 | | 1.62 / 1.46 | |
| g₂ −3 % / +3 % | 0.0315 / 0.0314 | 4.0016 / 4.0293 | 1.10 / 1.99 | 0.127 / 0.019 |
| σ_pos 0 / 3 / 6 % | 0.0301 / 0.0327 / 0.0409 | | 1.60 / 1.49 / 1.19 | 0.111 (6 %) |
| ε_ext 0.726 / 1.0 | 0.0324 / 0.0289 | | 1.49 / 1.65 | 0.061 / 0.037 |
| S2 Fano 0 | 0.0303 | | 1.57 | |
| quanta only (perfect detector) | 0.0254 | 4.0135 | 1.83 | 0.016 |
| binomial floor (ω = 0, Fano → 0, perfect detector) | 0.0233 | 4.0149 | 2.06 | 0.007 |
| ω = 0, full detector | 0.0301 | | 1.60 | |
| k_e = 0.5 (electron fluctuations halved) | 0.0227 | | 2.18 | |
| k_e = 0 (no electron fluctuations at all) | 0.0184 | | 2.70 | 0.004 |

Reading: (i) the yield *scale* (break, defaults) moves the median by 0.01–0.16 dex and is what makes the "1.5σ" possible at all — without the p(E) break the event would be 5σ low, with NEST defaults 6.4σ; (ii) the *width* is remarkably stiff: only W₃ and W₄ (the position and width of ω's Gaussian in f_e) change it by more than 3 %, and they can only widen it appreciably; the field, g₁, W₀, W₁, W₂, W₅ are irrelevant at the ±30 % level; (iii) g₂ ± 3 % moves the σ-distance between 1.1 and 2.0 by moving the median (this is the largest systematic on the "1.5σ" statement); (iv) nothing within the NEST parameter space brings the width below 0.030 dex with a realistic detector, or below 0.0233 dex with a perfect one.

## 5. Results — the drawn 10–90 % lines (script §3; `digitised_NR_lines_*.csv`)

Axis calibration: x from the frame (0–800 phd; P009 verified with the ROI edge at 600 phd); y from the major tick marks on the right frame edge (22-px runs at rows 226.5/418.5/610.5/802.5 = 4.5/4.0/3.5/3.0) plus the frame top (5.0): 384.2 px/dex, fit residual 0.0005 dex, frame bottom 2.739 (expected 2.75). Red-line centroids in ±6-px columns at S1c = 480–600 (7 columns, both figures):

| S1c | p10 | median | p90 | half-width |
|---|---|---|---|---|
| 480 | 3.975 | 4.002 | 4.030 | 0.0275 |
| 500 | 3.980 | 4.007 | 4.034 | 0.0272 |
| 520 | 3.985 | 4.0125 | 4.040 | 0.0276 |
| 540.1 | 3.989 | 4.0162 | 4.0436 | 0.0273 |
| 560 | 3.994 | 4.021 | 4.048 | 0.0269 |
| 580 | 3.998 | 4.026 | 4.052 | 0.0269 |
| 600 | 4.003 | 4.029 | 4.056 | 0.0265 |

Fig. 2 and Fig. 4 agree to < 0.0003 dex. Mean half-width **0.0271 ± 0.0004 dex** (symmetric: lower 0.0272, upper 0.0270), σ_eq = **0.0212 dex**. P009's 0.0276 (ROI-box calibration) and P010's 0.0285 (tick calibration, 540 only) are consistent. Event: offset from the drawn median 0.0492 dex = **2.33σ_eq** = 1.82 half-widths; Gaussian P = 1.0 %. The σ implied by the paper's own statement is 0.0492/1.5 = **0.0328 dex**, i.e. the dashed lines sit at ±0.83σ_paper, not ±1.28σ — they are consistent with 20–80 % percentiles of a σ ≈ 0.032 model, or with 10–90 % of a σ = 0.021 model, but not with both statements at once.

Which is right? (a) σ = 0.0212 is below the recombination-binomial floor (0.0231) and below pure Poisson on the model's own 300 electrons (0.0250). With our detector terms (0.0185 dex in quadrature) it would need the electron fluctuations scaled by k_e = 0.41; even with a perfect detector k_e = 0.84. (b) The paper's σ = 0.0328 agrees with the full MC (0.0313) to 5 %. (c) LZ's AmBe points decide (§6). We conclude that the model width is ≈ 0.031–0.033 dex and the dashed lines under-draw the 10–90 % range by a factor ≈ 1.5; the "1.5σ" is the correct reading, P009's "2.5σ" and P010's "2.2 Gaussian σ" for the NR band should be retired. We note the ER-band puzzle raised by P010 (6.7σ = 6.8 drawn half-widths) suggests LZ's "σ" is not defined identically for the two bands; we cannot resolve that here.

Independent check from Fig. S1a: the L₁₀ (1000 GeV) signal contours (brown) at S1c = 540 span 3.976–4.054 (68 %, half-extent 0.039) and 3.943–4.083 (95 %, half-extent 0.070). For a 2D Gaussian the vertical half-extent of the 68 %/95 % containment contour at the mode column is 1.51σ/2.45σ [recalled, certain]; away from the mode it is smaller, so σ ≥ 0.070/2.45 = 0.029 dex (95 %) and ≥ 0.026 (68 %) — again incompatible with 0.021 and consistent with 0.031–0.033.

## 6. Results — AmBe calibration statistics (script §4; `ambe_points_fig2.csv`, Fig. `P024_decomposition_and_ambe.png`, `P024_ambe_residuals.png`)

Purple (144, 0, 208) pixels in Fig. 2 were labelled as connected components; the median single-marker blob (24 px, sparse region) sets the per-blob multiplicity. Counts (multiplicity-corrected): S1c > 450: 43; > 500: 22; 500–580 phd (233–264 keV on the paper scale): **19**; 520–560: 10; > 580: 3; last point at 600 phd (272 keV; NEST itself warns above 300 keV: "beyond the AmBe endpoint of about 300 keV"). Below 450 phd the markers overlap (152 blobs, ≥ 565 points) so the total is not recoverable from the figure. The caption states the plotted sets include events outside the FV.

Spectrum model. AmBe ISO 8529-1 shape as a 1-MeV histogram [recalled, uncertain; mean 4.4 MeV], E_R,max = 0.0301 E_n [certain], flat recoil spectrum per E_n (isotropic CM) or black-disk |2J₁(qR)/qR|² with R = 6.35 fm (P013): fraction of single-scatter recoils in 230–270 keV = 1.21 % (isotropic) or 0.073 % (black disk). The observed 19–21 points per 50–80 phd bin above 450 phd imply ≈ 1800 events in the plotted AmBe set under the isotropic model (30 000–76 000 under black-disk, implausible; the calibration neutrons are moderated and multiply scattered, so the black-disk single-scatter pattern does not apply). For a "few thousand"-event calibration one expects 12–60 events in 230–270 keV; LZ's figure shows ≈ 19.

Precision. With N = 19: σ(median) = 0.031/√19 = 0.0072 dex (15 % of the event's 0.048 dex offset, ±0.23 on the "1.5σ"); σ(width)/width = 1/√(2N) = 16 %. Fixing the width to 5 % needs 200 events at 230–270 keV.

Measured width from the points. Residuals of single-marker AmBe points about the digitised median curve (2 700 columns): S1c 450–600: N = 31, sd = **0.0359 ± 0.0046 dex** (68 % interval 0.031–0.041), robust 1.4826·MAD = 0.032, 10–90 half-width 0.0475, mean −0.002; 500–580: N = 13, sd 0.033 ± 0.006. Likelihood ratio for σ = 0.0313 vs 0.0212: Δln L = +11.4 (N = 31), +2.7 (N = 13). Five of 31 points lie below −1.5σ_MC (expected 2.1; P(≥5) = 6 %), and four lie below −2.33σ_drawn = −0.049 dex, i.e. as low as or lower than the event relative to the median (expected 0.31 for σ = 0.0212: P(≥4) = 3 × 10⁻⁴). LZ's own high-energy AmBe points are therefore incompatible with the drawn band width and consistent with (if anything wider than) the NEST width. In 300–450 phd the residual sd is 0.048 but there the blob centroids are contaminated by overlaps and the wall-leakage population, so we do not use it.

## 7. Results — MSSI overlap (script §5; Fig. `P024_pdf_at_540.png`)

Fig. S1a (same x frame; 472.3 px/dex from the right-edge ticks, fit residual < 0.001 dex). Dark-blue (0, 80, 128) MSSI contour rows in ±12-px columns:

| S1c | 95 % lower edge | 68 % lower edge |
|---|---|---|
| 500 | 3.689 | 3.978 |
| 520 | 3.688 | 3.980 |
| 540.1 | 3.695 | **3.988** |
| 560 | 3.704 | 4.007 |
| 580 | 3.716 | 4.037 |

Upper edges lie above the ROI top (4.15) and are clipped; the 68 % region at 540 is therefore [3.988, ≥ 4.15]. The event (3.967) is 0.021 dex *below* the 68 % edge (line thickness ≈ 0.01 dex, marker radius ≈ 0.013 dex — "touching" in the figure) and 0.27 dex above the 95 % edge. Fraction of the NR PDF at S1c = 540 inside the MSSI 68 % region: **81 %** (MC), 80 % (Gaussian 0.031), 91 % (drawn width); inside the 95 % region: 100 % for every width; 0 % of the NR PDF lies above the ROI. Conversely, P004 found that only 3.5 % of the wall-MSSI PDF at S1c > 500 lies within ±2σ_NR of the median. Band position thus cannot separate a genuine NR from an MSSI at this S1c; the discrimination is entirely in the MSSI *rate*, which P004 addressed.

## 8. Validation, robustness, failed approaches
- NEST variance form reproduced to 0.3 % (sd N_e 17.52 predicted vs 17.56 MC); analytic quadrature 0.0313 = MC 0.0313.
- Fig. 2 vs Fig. 4 digitisation agree to 0.0003 dex; tick fit residual 0.0005 dex; frame bottom 2.739 vs nominal 2.75 (the frame is 0.011 dex below the ROI edge, immaterial).
- First digitisation attempt used the dashed grey ROI box for the y scale (P009 method); it failed on Figs. 4 and S1a because the grey ER cloud dominates the grey-pixel row sums (nonsense medians of −2.2 and 5.6). Replaced by the tick-based calibration; results for Fig. 2 unchanged.
- Tick detection from the left frame edge failed on Fig. 4 (black science points cover the edge); the right edge is clean in all three figures.
- Baseline uses σ_pos = 2 % and ε_ext = 0.80; the "1.5σ" moves between 1.19 (σ_pos 6 %) and 1.65 (ε = 1). g₂ ± 1σ is the largest systematic on the σ-distance (1.10–1.99).
- The AmBe residual measurement rests on 31 points read from a raster figure; marker-centroid precision ≈ 0.003 dex; possible mis-assignment of overlapping markers is excluded by using single-marker blobs only. Events outside the FV are included in LZ's plotted set and may widen the observed scatter (position corrections are worse near the wall), which is why we quote it as "consistent with, if anything wider than" the MC.
- The isotropic/black-disk spectrum bracket is not a calibration model (no transport, no multiple scattering); it is used only to show that 19 events in 230–270 keV are what a ~2000-event AmBe set gives.

## 9. Discussion
The NR-band width at 250 keV is not a free parameter of NEST: 54 % of its variance is the binomial statistics of ≈ 300 escaping electrons and a further 16 % the extraction binomial; the tunable recombination term ω contributes 7 % because it is centred at f_e = 0.5 while an NR has f_e = 0.06. Retuning `nr_er_width_parameters` by ±30 % changes the width by ≤ 3 % except through W₃/W₄, which can only widen it. The energy-scale ingredients (p(E) break, N_q normalisation) shift the median, not the width; without the break the event would be 5σ low, so the "1.5σ" depends entirely on the AmBe-tuned break, which the paper says was fixed before unsalting.

The paper's two descriptions of the band are mutually inconsistent by ×1.55. Physics (binomial floor), the NEST MC, LZ's L₁₀ signal contours and LZ's own AmBe points all favour σ ≈ 0.031–0.036 dex, i.e. the "1.5σ" reading with P(≤ observed | NR) ≈ 6 %; under the drawn lines it would be 1 %. The AmBe sample at 450–600 phd contains 4 of 31 points at or below the event's relative depth, so the event's charge deficit is unremarkable for an NR. Statistically the 19 AmBe events at 230–270 keV fix the band median to ±0.007 dex (±0.23 on the 1.5σ) and its width to ±16 %; the relevant systematic is instead g₂ (±3 % → 1.1–2.0σ) and the position corrections, for which "no such calibration events within a few cm of the event" exist.

For the NR-vs-MSSI question the band offers nothing: 81 % of the NR PDF at S1c = 540 lies inside the MSSI 68 % region and 100 % inside its 95 % region.

Recommendations to LZ: publish the band's Gaussian σ (or 16–84 %) alongside the 10–90 % lines and state which is used for "1.5σ"; publish the AmBe (S1c, log₁₀S2c) points above 450 phd with FV flags; give g₂ and the S2 position-correction residual at the event's (r, z).

## 10. Figures
- `figures/P024_band_width_variants.png` — left: σ(log₁₀S2c) at S1c = 540 ± 10 for every variant with the MC baseline, the drawn-line σ_eq, the σ implied by the paper's 1.5σ and the binomial floor; right: the event's σ-distance per variant.
- `figures/P024_pdf_at_540.png` — MC PDF at S1c = 540 vs Gaussians with the drawn and paper widths, event line, ROI top and the MSSI 68 %/95 % regions from Fig. S1a.
- `figures/P024_decomposition_and_ambe.png` — left: variance decomposition; right: AmBe points per 20 phd from Fig. 2 with the 500–580 window.
- `figures/P024_ambe_residuals.png` — AmBe residuals about the digitised median at 450–600 phd with the two Gaussian widths and the event's offset.

## 11. References
1. LZ Collaboration, arXiv:2609.02823 (2026) — Data Analysis paragraph, Fig. 2/4 captions, Supplemental "Details of the Detector Response modeling" (Tables S3–S5), "Waveform Analysis", Fig. S1 caption.
2. M. Szydagis et al., "A Review of NEST Models for Liquid Xenon and Exhaustive Comparison to Other Approaches", Front. Detect. Sci. Technol. (2024), arXiv:2211.10726.
3. M. Szydagis et al., NEST v2.x, Zenodo (nestpy 2.1.1 used here).
4. B. Lenardo et al., "A Global Analysis of Light and Charge Yields in Liquid Xenon", IEEE TNS 62, 3387 (2015).
5. J. Aalbers et al. (LZ), "First Dark Matter Search Results from the LUX-ZEPLIN (LZ) Experiment", PRL 131, 041002 (2023) — extraction efficiency (recalled).
6. ISO 8529-1:2001, reference neutron radiations (AmBe spectrum; recalled).
7. Corpus: dossier 00; P004 (MSSI neighbourhood fraction), P009 (energy scale, MC band, digitisation), P010 (band-width calibration k = 0.50), P013 (black-disk pattern).

## 12. Tools and provenance (mirrors provenance/P024.json)
- Agent tools: Read ×21 (PAPER_GUIDE; dossier; ledger; P009.md, P004.md, P010.md; P009 scripts and results.json; lzcommon.py; fulltext.tex l. 84–123 and 586–695; Fig2, Fig4, FigS1a PNGs; four P024 figures, two viewed twice), Bash ×20 (tex grep; directory listings and P004/P010 details grep; nestpy probes ×2; PNG colour/frame probe; tick-run probe; dataviz palette grep; script runs ×6 with log inspections ×3; results extraction; word-count checks ×4), Write ×6 (script, details.md, P024.json, P024.md ×3), Edit ×15 (script ×9, paper ×6), Skill ×1 (dataviz).
- Software: python 3.12.13; nestpy 2.1.1 (LZ_WS2024, NESTcalc.GetYields/GetQuanta/CalculateG2, RandomGen.set_seed); numpy 2.5.3; scipy 1.18.1 (stats.norm/skew/poisson, special.j1, ndimage.label/center_of_mass); pandas 3.0.5; matplotlib 3.11.2 (Agg, image.imread); common/lzcommon.py (LZ, NEST_NR_LZ, NEST_NR_DEFAULT, nest_nr_params_vector, DRIFT_FIELD_VCM).
- Recalled knowledge (9 items): NEST GetQuanta NR variance/ω form (likely; verified numerically); NEST v2.4 13-entry width vector with two extra NR terms (uncertain); LZ extraction efficiency ≈ 80 % (uncertain); AmBe ISO 8529-1 spectrum shape (uncertain); n–Xe elastic kinematics E_R,max = 4m_nM/(m_n+M)² E_n (certain); black-disk angular pattern R = 6.35 fm (likely, via P013); LXe recombination fluctuations ≥ binomial (likely); 10–90 % = ±1.2816σ and 2D-Gaussian 68/95 % radii 1.51/2.45 (certain); measured NR N_ex/N_i ≈ 1 (likely).
- Datasets: none. Data requests: none. WimPyDD files: none.
