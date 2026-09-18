# P090 — Propagating the 100 % MSSI uncertainty into the local and global significance: what a factor-2, factor-10 or mis-shaped wall/RFR background does to the 3.4σ

Simulated arXiv date 2026-09-16 · STAT · physics.data-an (cross-list hep-ex) · particle-statistics group.
Script: `output/code/P090_mssi_systematic.py` (single stage, 2 s wall time; run from the simulation root with `.venv/bin/python`).
Outputs: `output/work/P090/P090_results.json`, `P090_table1_priors.csv` … `P090_table7_fg_product.csv`, `P090_summary_table.csv`, `run_log.txt`, `P052_engine_log.txt`, `figures/P090_fig{1,2,3}_*.png`.

## 1. Motivation and framework

LZ (arXiv:2609.02823) lists multiple-scintillation single-ionization (MSSI) events as the background with the largest fractional uncertainty: Table I gives (4.9 ± 4.9) × 10⁻³ events in the science sample of the 4.7 t FV (fit result 4.6 +4.8 −4.6 × 10⁻³), Table S1 gives (7.7 ± 7.7) × 10⁻² in the prompt-veto sample with λ_MSSI = 0.94 ± 0.02, and Table S2 (1.4 ± 1.4) × 10⁻⁴ in the delayed sample (tex l. 236, 531, 539, 576). The main text (l. 194–196) explains the 100 % as a geometry uncertainty on the charge-dead wall layer (0.3 % of the active volume; the uncertainty "encompasses a charge dead region covering up to 0.6 %"). The MSSI supplement (l. 710–785) validates the model with a 5.4 t analysis volume, the HE sideband (800 < S1c < 1700 phd) and the prompt-veto sample; the comparison table (l. 764–782) gives, outside the blind 4.7 t WS-ROI science bin, six wall bins and six RFR bins (simulated / observed):

| region | type | volume | science sim / obs | prompt sim / obs |
|---|---|---|---|---|
| WS ROI | wall | 4.7 t | 0.0048 / — | 0.09 / — |
| WS ROI | RFR | 4.7 t | 0.0001 / — | 0.10 / — |
| WS ROI | wall | 5.4 t annulus | 0.03 / 0 | 0.1 / 1 |
| WS ROI | RFR | 5.4 t annulus | 0.003 / 0 | 1.9 / 2 |
| HE SB | wall | 4.7 t | 0.1 / 0 | 0.3 / 1 |
| HE SB | RFR | 4.7 t | 0.5 / 0 | 0.5 / 0 |
| HE SB | wall | 5.4 t annulus | 0.5 / 0 | 2.2 / 0 |
| HE SB | RFR | 5.4 t annulus | 21.5 / 18 | 5.2 / 3 |

In LZ's likelihood (Eq. S1) MSSI enters as one rate parameter with a Gaussian constraint and a fixed (S1c, log₁₀S2c) shape; the tagging efficiency λ_MSSI couples the science and prompt samples. The corpus has attacked the MSSI hypothesis from the rate side (P004: a wall-MSSI origin needs k ≈ 620, sidebands allow k < 1.64), the geometry side (P033: depth profile e^{−d/4.3 cm}, k ≳ 6 × 10⁴; P079: LZ's own table implies λ = 1.23 cm; P070: position LR 15 (2–50)), the veto side (P073: silence costs a further 0.46) and the radon side (P053). None of these has been fed back into the *significance*. P052 reproduced LZ's likelihood from published summaries (rms 0.42σ over 59 table entries; L10 at 1 TeV: 3.11σ asymptotic, 3.03 [2.98, 3.10]σ toy-calibrated vs LZ's 3.4σ), with MSSI carried as a 100 % Gaussian nuisance t_MSSI, exactly as LZ. We use that engine to answer: how much of the 3.4σ is hostage to the MSSI model, along which deformation directions (rate prior, rate central value, depth profile, d-tail, S1c distribution, RFR share), and are the deformations that matter already excluded by LZ's own sidebands?

## 2. Method

### 2.1 Engine
P052's script is imported by executing its model-building header (everything before its stage dispatch, i.e. binning, response, Fig. 5 digitisations, background cells, data cells, the `Likelihood` class) in a private namespace, with its log redirected to `work/P090/P052_engine_log.txt` and its one CSV write sent to `os.devnull`; nothing under `output/work/P052/` is touched (verified by file timestamps). Configuration `lzlike` (Fig. 5 signal d-shape, 9.3 % S1c resolution), model `L10_1000`. All P052 definitions apply: cells 25 phd × 0.5σ over 250–600 phd plus one S1c < 250 phd bin, d ∈ [−8, 8]; event cell [525, 550) × [−2, −1.5); components ER bulk/tail, accidentals, MSSI, atm-ν + ⁸B, detector neutrons, detector ERs/¹²⁷Xe; prompt (66) and delayed (55) Poisson totals; 14 parameters; constraints from Tables I/S1/S2; q0 with ŝ ≥ 0; Z = √q0 (asymptotic).

P052's nominal MSSI map: total 4.9 × 10⁻³ split 45 %/45 %/10 % over the three Fig. 5 panels (the bottom share is P004's digitised 5.06 × 10⁻⁴ at S1c > 500 phd), flat in S1c within a panel, d-shape = P004's bottom-panel MSSI histogram in all panels. Derived quantities (this work): fraction of bottom-panel MSSI within |d| < 2: **0.343** (P004's f_nb = 0.035 of the total is 0.343 × 0.103); fraction in the event's d-bin [−2, −1.5): **0.0316**; MSSI expectation at the event cell **3.995 × 10⁻⁶** = 8.15 × 10⁻⁴ of the total; background at the event cell under H0 1.159 × 10⁻⁵, so **MSSI is 34.5 % of the background at the event** (accidentals most of the rest). Neighbourhood N = {S1c > 500 phd, |d| < 2}: 32 cells holding 3.54 % of the MSSI model (c₀; P004: 3.5 %, P033: 5.3 %). If all MSSI were spread uniformly over N, the event cell would hold 1/32 of it: **g_max = (1/32)/(8.15 × 10⁻⁴) = 38.3** times the nominal cell content.

### 2.2 Extensions (class `MLike`)
1. **Log parametrisation.** t_MSSI = e^u, u ∈ [−9, 9], start u = 0. The base-class Gaussian on t is switched off (σ → 10⁹) and the prior is added explicitly:
   Gaussian 100 % (LZ, P052): ½(e^u − 1)²; log-normal: ½(u/σ_ln)² with σ_ln = 0.7, 1.5, 2.3 (68 % ranges ×2, ×4.5, ×10; medians 1); flat: 0.
2. **Fiducial rate factor f.** R_MSSI → f R_MSSI in both the science cells (t f R_MSSI (1 − λ_MSSI) MSSI_c/0.0049) and the prompt total (λ_MSSI f R_MSSI t), so f is a deformation of the *model*, on top of which the ±100 % (or other) prior still acts.
3. **Sideband terms.** Optional Poisson terms Σ_i [μ_i − n_i ln μ_i] for the six wall bins, μ_i = k m_i, with k = t f ("common": a geometry error of the dead layer scales everything, P004's model) or k = t ("FV-only": f is a shape/extrapolation error that the sidebands cannot see). The 4.7 t wall share of the FV MSSI is 0.0048/0.0049 = 98 %, so the RFR bins are not attached to t; RFR is treated separately (§2.5).
4. **Replaceable MSSI map** (panel split, d-shape mixture) and a per-event position weight: multiplying the MSSI content of the event's cell by 1/LR_pos implements a position-aware likelihood for the single high-S1c event (P070's construction), because only that event's density changes while totals are unaffected to 4 × 10⁻⁶.
5. **Minuit**: migrad strategy 0, retried with strategy 1 if invalid; q0 as in P052 (H0 fit, score test, s-free fit started from H0, fallback start s = 0.3).

### 2.3 LEE mapping
Global significance uses the effective-trials mapping rather than new toys: p_global = 1 − (1 − p_local)^{N_eff(Z_local)} with N_eff from P071's debiased table 4.87/6.12/9.01/14.2 at Z = 2.0/2.5/3.0/3.4 (log-log interpolation, power-law extrapolation outside; P008's 4.8/6.0/8.3/12.2 as alternative). Check: Z_local 3.4 → 2.59σ (P071) / 2.64σ (P008); LZ 2.6σ. N_eff at our baseline 3.11σ is 10.3. Below 2σ the mapping is an extrapolation (N_eff = 2.4 at 1σ) and only indicative. Toy calibration: P052's lzlike L10 toys gave 3.03 [2.98, 3.10] vs 3.11 asymptotic; we quote asymptotic Z and note the −0.08σ offset (column `Z_toy_est`). Because P052 sits 0.29σ below LZ (unpublished 250–500 phd S1c structure, DR-002), the summary also lists "LZ-anchored" values Z + 0.29 and their global mapping (column `_LZanch`), which reproduce 3.4 → 2.59σ at the baseline.

### 2.4 Depth-scale systematic
LZ's table gives annulus(5.4 t)/FV(4.7 t) = 0.03/0.0048 = 6.25 for wall MSSI in the WS ROI. P079 integrated e^{−(R−r)/λ} over the digitised FV contours and found this ratio requires λ = 1.23 cm, while λ = 3/4.3/4.8/6/8 cm give 1.34/0.86/0.76/0.60/0.46. If the true depth profile has scale λ but the annulus expectation is held at its calibrated value (the annulus is where the model is checked: 0 observed vs 0.03), the FV expectation becomes f_λ = 6.25/ratio(λ) times LZ's — a shape error that turns into an FV-only rate factor: f_λ = 1.0/2.3/4.7/7.3/10.4/13.6 for λ = 1.23/2/3/4.3/6/8 cm (log-log interpolation of P079's points).
Position: the FV volume per 1-cm shell of distance d from the wall is P070's `profiles_d.csv` column `uniform` (d = 8.5–45.5 cm). Wall-MSSI position PDF p_λ(d) ∝ V(d) e^{−(d−8)/λ}; the event's distance from the wall is 26.9 cm (P033/P073; r = 45.9 cm, R = 72.8 cm). LR_pos(λ) = [V(d_ev)/ΣV]/p_λ(d_ev) = 2.7 × 10⁴, 247, 23.9, 6.55, 3.07, 2.02 for λ = 1.23/2/3/4.3/6/8 cm; P070's `exp_true` values 21.7/5.73/2.63 at 3/4.3/6 cm agree within 10–17 % (P070 used the 2D (r, z) FV map). Fraction of wall MSSI more than 25 cm from the wall: 4.5 × 10⁻⁶ (1.23 cm), 0.026 (4.3 cm; P033 f_pos ≈ 0.010–0.014 with its z cut), 0.067 (6 cm).

### 2.5 Other shape deformations
d-shape: mixture (1 − a) P004 + a N(0, 1), a = 0.25/0.5/1, and "flat within |d| < 2"; S1c: bottom-panel share f_bot = 0.2/0.5/1.0 with the rest split equally between the two lower panels; total fixed at 4.9 × 10⁻³. RFR: FV expectation 10⁻⁴ (49× below wall); scaling it by k_RFR and assuming it shares the wall (S1c, d) shape is equivalent to a total factor f_eq = 1 + (k_RFR − 1)/49; its own six sideband bins (29.6 predicted, 23 observed; HE-SB 5.4 t science 21.5 / 18) give the p-values.

### 2.6 Joint deformation
Because MSSI elsewhere in the ROI totals < 0.01 events, Z_local depends on the event-cell MSSI expectation f·g·3.995 × 10⁻⁶ (g = shape concentration at the event cell relative to LZ's map). Z(f·g) is tabulated (`P090_table7_fg_product.csv`) and f·g for Z = 3/2/1 found by Brent's method.

## 3. Results

### 3.1 Validation
Baseline (u = ln t, Gaussian 100 %, no sideband terms): q0 = 9.664, Z = 3.109, ŝ = 1.178, b_event = 1.159 × 10⁻⁵ — identical to P052 (9.664, 3.109, 1.181, 1.159 × 10⁻⁵). MSSI × 2: 2.976 (P052 variant 2.975). Six-wall-bin profile alone: k_ML = 0.62, k < 1.64 (one-sided 95 % LR; two-sided 1.91) — P004's 0.62/1.64. Position LR vs P070: §2.4. No-MSSI limit (f → 0): Z = 3.294, so the entire MSSI channel is worth **0.19σ** (its 34 % share of the event-cell background).

### 3.2 Part 1: priors and the rate ladder (`P090_table1_priors.csv`, `P090_table2_rate_ladder.csv`, `P090_table3_sideband_pvalues.csv`)

| prior on t_MSSI | LZ-style (no sideband terms): Z, t̂_H0, MSSI_FV(H0) | + six wall sideband bins: Z, t̂_H0 |
|---|---|---|
| Gaussian 100 % | **3.109**, 1.36, 0.0067 | 3.169, 0.78 |
| log-normal σ_ln = 0.7 (×2) | 3.117, 1.25, 0.0061 | 3.158, 0.84 |
| log-normal 1.5 (×4.5) | 3.003, 5.85, 0.029 | 3.171, 0.76 |
| log-normal 2.3 (×10) | 2.652, 47.2, 0.23 | 3.175, 0.74 |
| flat | 1.756, 165, 0.81 | 3.179, 0.72 |

With a wide prior the H0 fit inflates MSSI to cover the event (t̂_H0 = 47 for σ_ln = 2.3), and the science-sample counts alone stop it near one event (flat prior: 0.81 events) because 45 % of the MSSI map sits at 250–500 phd below the ER band where no events are observed; along the pure-rate ladder without sideband terms Z has a floor of 1.48σ at f ≈ 200 (t̂_H0 falls to 0.89, then 0.39 at f = 500). With the six wall bins in the likelihood, every prior returns 3.16–3.18σ (the sidebands pull t̂ to 0.72–0.84): **the prior width is irrelevant once the calibration data are part of the fit.**

Rate ladder (Gaussian prior kept):

| f | LZ-style (no SB) Z | FV-only (SB at k) Z | common (SB at k f) Z | six-wall-bin p (common) | annulus p | HE-SB wall p |
|---|---|---|---|---|---|---|
| 1 | 3.109 | 3.169 | 3.169 | 0.37 | 0.97 | 0.55 |
| 2 | 2.976 | 3.070 | 3.167 | 0.044 | 0.94 | 0.30 |
| 5 | 2.731 | 2.868 | 3.173 | 1.4 × 10⁻⁵ | 0.86 | 0.050 |
| 10 | 2.503 | 2.666 | 3.176 | 5.2 × 10⁻¹² | 0.74 | 2.5 × 10⁻³ |
| 20 | 2.247 | 2.428 | 3.177 | 1.9 × 10⁻²⁵ | 0.55 | 6 × 10⁻⁶ |
| 50 | 1.881 | 2.068 | 3.178 | 10⁻⁶⁶ | 0.22 | 10⁻¹³ |
| 100 | 1.623 | 1.770 | 3.178 | 10⁻¹³⁶ | 0.050 | 10⁻²⁶ |
| 200 | 1.478 | 1.472 | 3.179 | 10⁻²⁷⁵ | 2.5 × 10⁻³ | 10⁻⁵² |
| 500 | 1.578 | 1.147 | 3.179 | 0 | 3 × 10⁻⁷ | 10⁻¹³⁰ |

Required FV-only factors: **Z = 3 at f = 2.87, Z = 2 at f = 58.7, Z = 1 at f = 977** (Brent, ±0.1 %). A common factor never reaches 3σ: the sideband terms fix k f ≈ 0.62–0.78 so the FV expectation stays at 0.0035–0.0039 whatever f. Sideband p-values at the required FV-only factors, *if* they were common: P(N₆ ≤ 2 | 3.23 f) = 5.0 × 10⁻³ (f = 2.87), 7.9 × 10⁻⁷⁹ (58.7), 0 (977); the 5.4 t annulus science bin alone (0 vs 0.03 f) gives 0.92/0.17/2 × 10⁻¹³; the HE-SB wall science bins (0 vs 0.6 f) 0.18/5 × 10⁻¹⁶/10⁻²⁵⁵; the 4.7 t prompt total (66 vs 66 − 0.072 + 0.072 f) 0.53/0.33/10⁻¹². A 5 % sideband p-value is crossed at f ≈ 2 (six bins) or f ≈ 5 (HE-SB wall science alone).

### 3.3 Part 2: shape (`P090_table4_depth_scale.csv`, `P090_table5_dshape.csv`, `P090_table6_rfr.csv`)

Depth scale (annulus-anchored, sideband terms at k, Gaussian prior):

| λ [cm] | annulus/FV | f_λ | LR_pos | f(d > 25 cm) | Z position-blind | Z position-aware | Z_global blind (LZ-anch.) |
|---|---|---|---|---|---|---|---|
| 1.23 (LZ table, P079) | 6.25 | 1.00 | 2.7 × 10⁴ | 4.5 × 10⁻⁶ | 3.169 | 3.294 | 2.39 (2.65) |
| 2.0 | 2.70 | 2.32 | 247 | 5 × 10⁻⁴ | 3.043 | 3.293 | 2.29 (2.53) |
| 3.0 | 1.34 | 4.66 | 23.9 | 0.006 | 2.886 | 3.268 | 2.14 (2.40) |
| 4.3 (P033 MC) | 0.86 | 7.27 | 6.55 | 0.026 | **2.764** | **3.158** | 2.03 (2.30) |
| 6.0 | 0.60 | 10.4 | 3.07 | 0.067 | 2.653 | 2.966 | 1.92 (2.20) |
| 8.0 | 0.46 | 13.6 | 2.02 | 0.119 | 2.564 | 2.790 | 1.84 (2.11) |

The two faces of a longer depth scale nearly cancel: a 4.3 cm profile raises the FV rate ×7.3 (−0.35σ in LZ's position-blind likelihood) but predicts the event's depth with probability 1/6.6 relative to a uniform signal (+0.39σ back). At λ = 1.23 cm the position-aware fit is the no-MSSI value (3.29σ) because wall MSSI cannot reach 26.9 cm.

(S1c, d) re-shaping inside the ROI (total fixed; sideband terms common):

| deformation | band fraction (bottom panel, |d| < 2) | event-bin fraction | g at event cell | MSSI at S1c > 500 | Z | Z_global (LZ-anch.) |
|---|---|---|---|---|---|---|
| baseline P004 shape, 10 % at S1c > 500 | 0.343 | 0.0316 | 1.00 | 5.1 × 10⁻⁴ | 3.169 | 2.39 (2.65) |
| d: 25 % N(0,1) admixture | 0.496 | 0.035 | 1.10 | 5.1 × 10⁻⁴ | 3.158 | |
| d: 50 % N(0,1) | 0.649 | 0.038 | 1.20 | 5.1 × 10⁻⁴ | 3.148 | |
| d: 100 % N(0,1) (MSSI on the NR band) | 0.954 | 0.044 | 1.40 | 5.1 × 10⁻⁴ | 3.127 | 2.36 (2.61) |
| d: flat in |d| < 2 only | 1.000 | 0.125 | 3.96 | 5.1 × 10⁻⁴ | 2.926 | 2.18 (2.43) |
| S1c: 20 % at S1c > 500 | 0.343 | 0.0316 | 1.94 | 1.0 × 10⁻³ | 3.076 | |
| S1c: 50 % | 0.343 | 0.0316 | 4.84 | 2.4 × 10⁻³ | 2.875 | 2.13 (2.39) |
| S1c: 100 % | 0.343 | 0.0316 | 9.68 | 4.9 × 10⁻³ | 2.672 | 1.94 (2.22) |
| all MSSI in N (g_max) | 1.000 | 0.125 | 38.3 | 4.9 × 10⁻³ | 2.153 | 1.41 (1.72) |

The d-tail is a weak lever: P004's digitised shape is already nearly flat in d, so even putting every bottom-panel MSSI on the NR band costs only 0.04σ; the S1c distribution is the strong one (×9.7 at the event if all MSSI sat above 500 phd, −0.50σ). The maximal shape deformation gives 2.15σ; pure shape cannot reach 2σ.

RFR: k_RFR = 10/100/500/1000/5000 → Z = 3.08/2.87/2.46/2.22/1.61 (f_eq = 1.18/3.0/11/21/103), while the HE-SB 5.4 t RFR bin (18 observed) would expect 215/2150/…: p = 7 × 10⁻⁶⁸ already at k_RFR = 10. The RFR route is closed by the best-calibrated sideband in the paper.

### 3.4 Part 4: the MSSI-only explanation (`P090_table7_fg_product.csv`, `P090_results.json:mssi_only`)
f·g required: **2.87 (Z = 3), 53.8 (Z = 2), 301 (Z = 1)**; the shape-only and rate-only routes agree on the product to 10 % up to f·g ≈ 100 (beyond, the rate route is capped by the ROI counts, §3.2).
For Z_local < 2 (f·g = 54):
- rate only (common): P(N₆ ≤ 2 | 3.23 × 54) = 6 × 10⁻⁷²; annulus alone 0.20; HE-SB wall 10⁻¹⁵ — excluded;
- shape only: impossible (g_max = 38 → 2.15σ);
- g_max with f = 1.40: sideband p = 0.17 — *not* excluded by the counts, but it requires LZ's simulation to have misplaced 90 % of its MSSI in S1c (Fig. 5 bottom panel shows 5.1 × 10⁻⁴, not 4.9 × 10⁻³ × 1.4) and to have a d-distribution 3× more band-concentrated than drawn; it predicts 0.11 prompt-veto MSSI events in N (P(0) = 0.90: untestable at this exposure) and 0.0069 science events in N;
- depth route: λ = 4.3 cm (f = 7.27) plus g = 7.4 (essentially all MSSI at S1c > 500 phd) reaches 2σ position-blind, but position-aware needs g = 48 > g_max.
For Z_local < 3 (f·g = 2.87): a common rate ×2.87 has sideband p = 5 × 10⁻³; FV-only ×2.87 or λ ≳ 2.3 cm suffice and are not testable by the counts — this is the realistic size of the MSSI systematic.

### 3.5 Part 5: summary (`P090_summary_table.csv`)

| MSSI hypothesis | Z_local | ΔZ | N_eff | Z_global | Z_global (LZ-anchored) | sideband p |
|---|---|---|---|---|---|---|
| LZ model, 0.0049 ± 100 % (P052 baseline) | 3.109 | 0 | 10.3 | 2.34 | 2.59 | 0.37 |
| MSSI removed | 3.294 | +0.19 | 12.4 | 2.50 | 2.76 | — |
| log-normal σ_ln = 2.3, no sideband terms | 2.652 | −0.46 | 6.9 | 1.92 | 2.20 | — |
| flat prior, no sideband terms | 1.756 | −1.35 | 4.3 | 1.00 | 1.30 | — |
| any prior + six wall bins | 3.16–3.18 | +0.05…+0.07 | 11.0 | 2.38–2.40 | 2.64–2.65 | — |
| rate ×2 FV-only / common | 3.070 / 3.167 | −0.04 / +0.06 | | 2.31 / 2.39 | 2.56 / 2.64 | 0.37 / 0.044 |
| rate ×5 FV-only / common | 2.868 / 3.173 | −0.24 / +0.06 | | 2.13 / 2.40 | 2.38 / 2.65 | 0.37 / 1.4 × 10⁻⁵ |
| rate ×10 FV-only / common | 2.666 / 3.176 | −0.44 / +0.07 | | 1.93 / 2.40 | 2.21 / 2.65 | 0.37 / 5 × 10⁻¹² |
| rate ×50 FV-only / common | 2.068 / 3.178 | −1.04 / +0.07 | | 1.32 / 2.40 | 1.63 / 2.65 | 0.37 / 10⁻⁶⁶ |
| λ = 4.3 cm, position-blind / aware | 2.764 / 3.158 | −0.35 / +0.05 | | 2.03 / 2.38 | 2.30 / 2.64 | 0.37 |
| λ = 1.23 cm, position-blind / aware | 3.169 / 3.294 | +0.06 / +0.19 | | 2.39 / 2.50 | 2.65 / 2.76 | 0.37 |
| MSSI on the NR band (d) | 3.127 | +0.02 | | 2.36 | 2.61 | — |
| all MSSI at S1c > 500 phd | 2.672 | −0.44 | | 1.94 | 2.22 | — |
| all MSSI in the neighbourhood (g_max) | 2.153 | −0.96 | 5.2 | 1.41 | 1.72 | — |
| RFR ×100 | 2.874 | −0.23 | | 2.13 | 2.39 | 0 (HE-SB) |

Toy-calibrated estimates are 0.08σ lower throughout (P052). The LEE cost Z_local − Z_global is 0.75–0.8σ for Z_local between 2 and 3.4 (P071's flat-cost result), so every ΔZ_local translates almost one-to-one into ΔZ_global.

## 4. Figures
- `figures/P090_fig1_rate_ladder.png` — left: Z_local (solid) and Z_global (dashed) vs the MSSI rate factor f for the three treatments (prior only; FV-only with sidebands at k; common with sidebands at k f), with the FV-only factors for Z = 3/2/1 marked; right: Poisson p-values of LZ's own sideband counts for a common factor f (six wall bins; 5.4 t annulus science bin; HE-SB wall science bins).
- `figures/P090_fig2_shape.png` — left: Z_local vs wall depth scale λ, position-blind and position-aware, labelled with f_λ and LR_pos; right: Z_local vs the MSSI expectation at the event cell (f·g), with the explicit (S1c, d) re-shapings as points and the rate-only route dashed (showing the ROI-count floor).
- `figures/P090_fig3_fg_plane.png` — the (f, g) plane with Z = 1/2/3 contours, the sideband limit k < 1.64 (common rate), the λ = 4.3 cm annulus-anchored factor and g_max.

## 5. Validation and robustness
- Exact reproduction of P052's q0, ŝ, event-cell background and MSSI × 2 variant (§3.1); the log parametrisation changes Z by < 10⁻⁵.
- Six-wall-bin profile reproduces P004's k_ML and 95 % limit to two digits; position LRs within 10–17 % of P070; N_eff mapping reproduces LZ's 2.6σ at 3.4σ.
- Prior width: with sideband terms all five priors agree to 0.02σ.
- Depth scale bracket 1.23–8 cm: position-blind 3.17–2.56σ, position-aware 3.29–2.79σ.
- The joint deformation is a function of f·g alone to 10 % for f·g ≤ 100 (rate vs shape routes, `table7`).
- Runtime 2 s; no stochastic element (no toys), so results are exactly reproducible.

## 6. Failed or abandoned approaches
- New background-only toys for each deformation: P052's toy loop costs ≈ 50 ms per toy, so a 4-minute budget yields ≈ 4000 toys — 4 expected exceedances at p ≈ 10⁻³, too few to resolve 0.1σ shifts. We use P052's measured toy offset (−0.08σ) and P071's N_eff mapping instead (stated in the paper).
- A single common 'MSSI' parameter tied to the RFR sideband bins as well: rejected because RFR is 2 % of the FV expectation and its sidebands would dominate k while saying nothing about wall MSSI; RFR treated separately (§2.5).
- Digitising Fig. S4 (prompt-veto sample) to count prompt events at S1c > 500 phd in the band, to test the shape route directly: not attempted (time); the predicted 0.11 events at g_max show it would be uninformative anyway.
- Re-deriving the annulus/FV ratio with a simplified cylinder: a thin-annulus estimate (ratio ≈ e^{Δ/λ} − 1 with Δ = 2.3 cm) reproduces P079's 6.25 at λ = 1.23 cm and gives 0.74 at 4.3 cm vs P079's 0.86 (the 5.4 t volume also extends in z); P079's contour integrals were used.

## 7. Discussion
1. **The 100 % is not the problem.** In a likelihood that includes the calibration counts, the MSSI rate is pinned at k̂ ≈ 0.7 with k < 1.64, and Z_local is 3.17 ± 0.01 for every prior and every common rate factor; LZ's Gaussian ±100 % is, if anything, conservative (it lets t̂_H0 rise to 1.36 and costs 0.06σ relative to the sideband-anchored fit). Widening it to a log-normal ×10 costs 0.46σ only if the sidebands are ignored, and a flat prior would let the fit "explain" the event with 0.8 MSSI events — a warning against profiling weakly-constrained backgrounds without their control samples in the same likelihood.
2. **The realistic systematic is shape, and it is ≈ 0.3–0.5σ.** Deformations invisible to the integrated sideband counts — the depth profile (×7 in the FV for P033's 4.3 cm), the S1c distribution inside the ROI (×10 at the event if MSSI were concentrated above 500 phd) — move Z_local by −0.35 to −0.5σ (global −0.35 to −0.4σ). These are exactly the quantities LZ's validation table does not test: it compares counts, not (S1c, d, r) distributions.
3. **Position is the antidote to the depth-profile systematic.** The same long depth scale that raises the FV rate lowers the MSSI density at the event; with position in the likelihood the net effect of λ = 1.23 → 4.3 cm is −0.14σ instead of −0.41σ. This complements P070 (position adds +0.1–0.2σ at fixed model) and P079 (a position likelihood is worth 16–20 % of mass).
4. **No MSSI-only explanation survives.** Z_local < 2 needs f·g = 54: rate alone is excluded at 10⁻⁷¹ by the six wall bins; shape alone cannot do it; the one surviving corner (g_max with f = 1.4) requires LZ's own Fig. 5 MSSI distribution to be wrong by ×10 in S1c and ×3 in d and is untestable by the prompt-veto sample (0.11 events). This is consistent with P004/P033/P073 (k_required 620 → 1350 → 6 × 10⁴) seen from the likelihood side: 0.19σ is the most the channel can give back.
5. **Recommendation for the next LZ release.** (i) Add the MSSI validation bins (annulus, HE-SB, prompt) as Poisson terms in the likelihood rather than as an external p = 0.7 check; (ii) replace the single rate nuisance by shape nuisances — the S1c fraction above 500 phd, the band fraction and the wall depth scale — each constrained by the (S1c, d) and r distributions of the HE-SB and annulus samples (18 RFR and the prompt wall events exist for this), or use position in the likelihood, which makes the depth-scale nuisance self-calibrating; (iii) publish the MSSI (S1c, d) template and the annulus/HE-SB distributions so that the shape systematic can be assessed externally; (iv) quote the significance with MSSI removed (3.29σ vs 3.11σ here; +0.19σ) as the bracketing value.

## 8. References
- LZ Collaboration, arXiv:2609.02823 (2026): Table I, Tables S1/S2, main text l. 184–196, 316–319, supplement l. 710–785 (MSSI comparison table).
- G. Cowan, K. Cranmer, E. Gross, O. Vitells, Eur. Phys. J. C 71, 1554 (2011) — asymptotic q0.
- D. Baxter et al., Eur. Phys. J. C 81, 907 (2021) — DM statistics conventions (LZ's reference).
- S. Baker, R. D. Cousins, Nucl. Instrum. Meth. 221, 437 (1984) — Poisson likelihood χ².
- H. Dembinski et al., iminuit (Zenodo, 2020).
- E. Gross, O. Vitells, Eur. Phys. J. C 70, 525 (2010) — trials factors.
- Corpus: P001, P004, P008, P016, P033, P052, P053, P070, P071, P073, P079.

## 9. Tools and provenance (mirrors `output/provenance/P090.json`)
- Agent tools: Read (PAPER_GUIDE; P052/P004/P033/P073/P079/P070/P053/P008/P071/P001/P016 papers; P052 script; three figures), Bash (tex extraction l. 184–200/222–250/300–320/470–485/520–582/764–785; listing and printing P052/P004/P070/P073/P079/P033/P008/P071 work files; ledger rows; script runs), Write (script, details, provenance, paper), Edit (script, 5 edits), Skill (dataviz; JS validator skipped per PAPER_GUIDE).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (stats.norm/poisson/chi2, optimize.brentq); pandas 3.0.5; matplotlib 3.11.2 (Agg); iminuit 2.32.0 (Minuit.migrad); Pillow 12.3.0 and nestpy 2.1.1 only through P052's header (Fig. 5 digitisation; yield cache read); WimPyDD 2.0.4 only through P003/P012/P021 caches read by P052's header; common/lzcommon.py (LZ numbers, sigma_to_p).
- Scripts: `output/code/P090_mssi_systematic.py` — `.venv/bin/python output/code/P090_mssi_systematic.py` (2 s).
- Local inputs: `inputs/LZ_arXiv_2609.02823_fulltext.tex` (Table I l. 222–250; Tables S1/S2 l. 520–582; MSSI text l. 184–196, 300–320; supplement l. 710–785); `output/code/P052_reproduce_significance.py` (header executed); `output/work/P052/P052_results.json` (baseline, toys); `output/work/P052/P052_yield_cache.npz`, `output/work/P003/P003_spectra.npz`, `output/work/P012/P012_L10_spectra_d10_1.npz`, `output/work/P021/spectra_{s,v}_{400,1000,4000}.npz`, `output/work/P003/fig1_digitised_{top,bottom}.csv`, `output/work/P016/P016_results.json`, `output/work/P004/P004_fig5_digitised.csv`, `output/work/P022/P022_fig5_accidentals_digitised.csv`, `inputs/figures_png/Fig5_NR_distance_3panels_no_sig.png` (all via P052's header); `output/work/P070/profiles_d.csv` (FV volume profile); `output/work/P004/P004_k_limits.csv`, `P004_results.json`; `output/work/P070/LR_table.csv`, `P070_results.json`; `output/work/P079/P079_results.json`, `details.md` (annulus ratios); `output/work/P033/P033_summary.json`; `output/work/P073/P073_veto_silence_LR.csv`, `P073_summary.json`; `output/work/P071/details.md` (N_eff table); `output/results_ledger.csv` (rows P004, P033, P053, P070, P071, P073, P079); papers P001, P004, P008, P016, P033, P052, P053, P070, P071, P073, P079.
- Recalled knowledge (4): asymptotic Z = √q0 for a one-sided single-parameter test (Wilks/Chernoff; certain); one-sided 95 % profile-likelihood limit at Δ(2 ln L) = 2.71 (certain); Šidák form p_global = 1 − (1 − p_local)^N_eff (certain); log-normal as the standard model of a multiplicative (factor) uncertainty (certain).
- Datasets: none. Data requests: none (DR-002, P052's request for the 250–500 phd S1c structure, would sharpen the baseline but not the differences). WimPyDD-generated files: none.
