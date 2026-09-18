# P004 — Wall MSSI as the origin of the LZ event: the mismodelling factor required and what the sidebands allow

Research record (complete). Simulated date 2026-09-03. Author profile: former xenon-TPC experimentalists. Category BKG; hep-ex (cross-list physics.ins-det).
Script: `output/code/P004_wall_mssi.py` (run from the root with `.venv/bin/python`). Machine-readable results: `output/work/P004/P004_results.json`; tables `P004_k_limits.csv`, `P004_topology.csv`, `P004_fig5_digitised.csv`; figure `figures/P004_k_likelihood_and_Pge1.png`.

## 1. Motivation and framework

The LZ paper (arXiv:2609.02823) identifies multiple-scintillation single-ionization (MSSI) events as one of the three dominant backgrounds at the location of the 248 keV candidate, and its own waveform analysis "could not distinguish between a single scatter at the reconstructed position of the event of interest and a wall MSSI template" (supplement, Waveform Analysis). The wall-MSSI rate is simulation-derived, with the charge-dead volume (0.3 % of the active volume, scalloped pattern up to 3 mm from the wall) assigned a 100 % uncertainty (main text, MSSI paragraphs). The evidence dossier (`output/00_evidence_dossier.md`, §4 point 3 and §6 question 2) flags that a large wall-MSSI mismodelling "is not obviously excluded by data" and calls for exactly the calculation done here.

We define a single scale factor k multiplying the modelled **wall**-MSSI rate everywhere (the paper states the wall rate is proportional to the dead volume "in this regime", so a geometry error acts as a common factor across analysis volumes and energy sidebands). We ask:

1. What k is required for a wall MSSI to be a *likely* origin of the event (P(≥1) = 10 % or 50 % in the event's neighbourhood)?
2. What k do the paper's own MSSI sideband counts (supplement, Table "Comparison of simulated and observed MSSI counts") allow?
3. What additional suppression does the event's specific topology (12 keV deposit 26.9 cm from the wall, 77 keV in a ≤3 mm dead layer) imply beyond what the {S1c, log10 S2c} PDF already encodes?
4. How does the 94 ± 2 % prompt-veto efficiency propagate?

## 2. Inputs (all from the LZ paper unless flagged)

| Input | Value | Source |
|---|---|---|
| Wall MSSI, science, WS ROI, 4.7 t | 0.0048 (blind search bin) | Supp. Table MSSI comparison, row 1 |
| Wall MSSI, prompt, WS ROI, 4.7 t | 0.09 | same |
| RFR MSSI, science/prompt, 4.7 t | 0.0001 / 0.10 | same |
| WS ROI wall 5.4 t: sim sci / obs / sim prompt / obs | 0.03 / 0 / 0.1 / 1 | same |
| HE SB wall 4.7 t | 0.1 / 0 / 0.3 / 1 | same |
| HE SB wall 5.4 t | 0.5 / 0 / 2.2 / 0 | same |
| HE SB RFR 5.4 t | 21.5 / 18 / 5.2 / 3 | same (used only in the mis-classification variant) |
| WS ROI RFR 5.4 t; HE SB RFR 4.7 t | 0.003/0/1.9/2; 0.5/0/0.5/0 | same (variant only) |
| Total MSSI, science sample (pre-fit) | (4.9 ± 4.9) × 10⁻³ | Table I |
| Prompt-veto tagging efficiency for MSSI | 94 ± 2 % | main text |
| Fig. 5 bottom panel (S1c > 500 phd) total background | 0.0106 ± 0.0008 | Fig. 5 caption |
| Event location | −1.5 σ_NR below NR median; S1c = 540.1 phd | main text |
| MSSI decomposition of the event | first scatter 12 ± 2 keV (S1c 69 ± 17 phd); second 471 phd → 77 ± 7 keV (wall) | supplement, MSSI section |
| Event distance to true wall / above cathode | 26.9 cm / 26.4 cm | main text |
| FV stand-off from true wall: min / mean | 8.0 / 10.7 cm | main text |
| Dead-layer thickness; RFR depth | ≤ 3 mm; 13.75 cm | main text |
| TPC active radius | 72.8 cm | recalled (likely; LZ TPC 145.6 cm diameter) |
| LXe γ attenuation lengths (ρ = 2.9 g cm⁻³) | 122 keV: 0.30 cm (paper: < 4 mm); 200: 1.0; 300: 2.0; 500: 3.7; 662: 4.7; 1000: 6.0; 1500: 7.6; 1764: 8.3; 2615: 9.8 cm | recalled NIST-XCOM-like totals (likely, ±20 %); the 122 keV anchor is from the paper |
| Electron mass 510.999 keV; Klein–Nishina formula | — | certain |

## 3. Method and derivations

### 3.1 Fraction of wall MSSI in the event's neighbourhood (f_nb) — digitisation of Fig. 5

Fig. 5 (bottom panel, S1c > 500 phd) shows the post-fit background components in 0.5 σ_NR bins of (log10 S2c − μ_NR)/σ_NR. We digitised the MSSI line (RGB 2,81,128) and the total-model line (0,0,255) from `inputs/figures_png/Fig5_NR_distance_3panels_no_sig.png` (2444 × 3082 px). Frame of the bottom panel: rows 2103–2905, columns 219–2409 (long dark lines). Axis calibration from detected tick marks: y ticks at rows 2131 (10²), 2324 (10⁰), 2518 (10⁻²), 2711 (10⁻⁴) → 96.75 px/decade, 10⁻⁶ at the frame bottom (2905); x ticks at columns 492, 766, …, 2135 for −6, −4, …, +6 σ → −8 σ at 218 and +8 σ at 2409 (frame edges). For each bin the histogram level is the median over the central 60 % of the bin of the top-most matching pixel.

**Validation.** The digitised total model sums to 0.01053 events over the panel versus the caption's 0.0106 (ratio 0.993). (With the first-pass calibration that assumed the frame edges were 10² and 10⁻⁶ the ratio was 0.78; the tick-mark calibration fixed it.) This confirms that "events / bin width" is per 0.5 σ bin and that the digitisation is accurate to a few per cent.

Digitised MSSI per bin (events): 6.4e-6 (−8), rising to ≈1.5e-5 from −6 to −0.5, 2.1–2.9e-5 from 0 to +4.5, then falling (full table in `P004_fig5_digitised.csv`). Sums:

- MSSI, whole S1c > 500 panel: **5.06 × 10⁻⁴** → f_{S1c>500} = 5.06e-4 / 4.9e-3 = **0.103**
- MSSI, |Δ| ≤ 2 σ_NR (8 bins): **1.74 × 10⁻⁴** → f_nb = **0.0354**
- MSSI, −3 ≤ Δ ≤ +1 σ (window centred on the event): 1.48e-4 → 0.030
- MSSI in the event's bin [−2, −1.5): **1.6 × 10⁻⁵**
- Total model, |Δ| ≤ 2 σ: 1.60 × 10⁻³; of which non-MSSI 1.43 × 10⁻³ (MSSI share 10.8 % at k = 1)

Since the wall component is 98 % of the total science-sample MSSI (0.0048 of 0.0049) we take the Fig. 5 MSSI curve as the wall-MSSI PDF. Reading uncertainty: we adopt a factor 1.5 range, f_nb ∈ [0.024, 0.053], which comfortably covers the alternative window choice and any line-overlap ambiguity. Independent check against Fig. S1a: the MSSI 68 % contour at S1c ≈ 540 spans log10 S2c ≈ 3.88–4.05 (about 5 σ_NR wide for σ_NR ≈ 0.033 dex) and the 95 % contour ≈ 3.65–4.08; the NR ±2 σ band (3.95–4.08) overlaps roughly the upper third of the 68 % region, i.e. perhaps 25–40 % of the MSSI density *at that S1c*. Combined with f_{S1c>500} ≈ 0.10 this gives f_nb ≈ 0.025–0.04, consistent with the digitised 0.035.

### 3.2 Sideband likelihood for k

Wall bins with observed counts (excluding the blind WS-ROI 4.7 t science bin): three science bins (predicted 0.03, 0.1, 0.5; observed 0, 0, 0) and three prompt bins (predicted 0.1, 0.3, 2.2; observed 1, 1, 0). Sums: science 0.63 predicted vs 0 observed; prompt 2.6 vs 2; total 3.23 vs 2.

Likelihood: L(k) = Π_i Pois(n_i | k·P_i + b_i), with b_i an optional non-MSSI contamination. Upper limits: (a) likelihood-ratio, one-sided, −Δln L = χ²₁(2CL−1)/2 (0.23, 0.82, 1.35 for 68/90/95 %); (b) Bayesian with a flat prior on k ≥ 0. Variants:

| Variant | k_ML | k < (LR 68/90/95 %) | k < (Bayes 68/90/95 %) |
|---|---|---|---|
| All six wall bins | 0.62 | 0.85 / **1.36** / **1.64** | 1.09 / 1.65 / 1.95 |
| Science bins only (veto-independent) | 0.00 | 0.17 / 1.30 / 2.15 | 1.81 / 3.65 / 4.74 |
| Prompt bins only | 0.77 | 1.05 / 1.69 / 2.04 | 1.35 / 2.05 / 2.42 |
| + 0.1 non-MSSI events per prompt bin | 0.10 | 0.31 / 0.81 / 1.08 | 0.69 / 1.22 / 1.51 |
| + 0.3 non-MSSI events per prompt bin | 0.00 | 0.06 / 0.40 / 0.63 | 0.49 / 0.96 / 1.23 |
| 20 % wall/RFR split nuisance (ε ~ N(0, 0.2) on predicted wall counts, profiled) | 0.62 | 0.86 / 1.44 / 1.77 | 1.18 / 1.89 / 2.30 |
| Extreme: 20 % of *observed* RFR events re-assigned to wall (adds 3.6 science, 1.0 prompt) | 2.04 | 2.44 / 3.24 / 3.64 | 2.66 / 3.49 / 3.91 |

Under k = 1: P(total ≤ 2 | 3.23) = 0.37 and P(0 science | 0.63) = 0.53 — the model is compatible, as the paper's p = 0.7 (12 bins, both MSSI types) says. The paper's own prior (100 % uncertainty, Gaussian(1,1) truncated at 0) has a 95 % quantile at k = 2.96; the sidebands tighten this to 1.6–2.0.

Comment on the contamination variant: the prompt-sample wall bins are in the sub-ER-band region where MSSI dominates, but detector-γ single scatters that are promptly vetoed (62 expected in the whole prompt WS ROI) have tails; any such contamination *lowers* the allowed k. The mis-classification extreme is not physically motivated (the wall/RFR split is by reconstructed position, and 7 of 12 radon-tag-eligible HE-SB RFR science events were radon-tagged, confirming their ²¹⁴Pb-RFR nature); it is quoted only as a bound.

### 3.3 Required k and probabilities

Predicted wall MSSI in the 4.7 t science WS ROI = 0.0048·k. In the neighbourhood (S1c > 500 phd, |Δ| ≤ 2 σ_NR): μ_nb(k) = 0.0048·f_nb·k = 1.70 × 10⁻⁴·k. P(≥1) = 1 − exp(−μ). k_required(P) = −ln(1−P)/(0.0048 f).

| Target | k for P = 10 % | k for P = 50 % |
|---|---|---|
| Anywhere in WS ROI (f = 1) | 22.0 | 144 |
| S1c > 500 phd (f = 0.103) | 212 | 1397 |
| Neighbourhood, f_nb = 0.035 | **619** (range 413–929 for f_nb 0.053–0.024) | **4074** (2716–6112) |

At k = 1: P(≥1 in ROI) = 0.48 %, P(≥1 in neighbourhood) = 1.7 × 10⁻⁴. At the 95 % sideband limit k = 1.64: 0.79 % and 2.8 × 10⁻⁴. At the pessimistic-extreme limit k = 3.64: 6.2 × 10⁻⁴.

Ratios: k_required(10 %, neighbourhood)/k_95 = 619/1.64 = **377** (range ≈ 250–570); even k_required(10 %, whole ROI)/k_95 = 13.4.

**Bayes factor** for "wall MSSI with k free" vs "as modelled (k = 1)", for the datum "one event in the neighbourhood" with likelihood μ e^{−μ}:
- k ~ flat prior × sideband likelihood (posterior mean 0.93): BF = **0.93**
- k ~ paper's truncated N(1,1) prior only: BF = 1.80
- k ~ paper's prior × sideband likelihood (posterior mean 1.38): BF = 1.38
Allowing the wall-MSSI normalisation to float within what the sidebands permit changes the plausibility of the MSSI hypothesis by less than a factor 2; it cannot rescue it. The MSSI share of the modelled background in the ±2 σ window is 11 % at k = 1 and 17 % at k = 1.64, so even conditional on "a background event happened here", wall MSSI is not the leading candidate (the modelled total there is 1.6 × 10⁻³; accidentals and atmospheric-ν NRs dominate).

### 3.4 Topology: Compton kinematics and attenuation

Compton kinematics (certain): T = E − E′, E′ = E/[1 + (E/m_e c²)(1 − cos θ)]. Klein–Nishina dσ/dΩ ∝ (E′/E)²[E′/E + E/E′ − sin²θ], integrated numerically over θ to get the fraction of scatters with T in [10, 14] keV and in [70, 84] keV.

Path lengths. Ordering "wall first" (γ from wall materials scatters in the ≤3 mm dead layer, travels inward, deposits 12 keV at the event position, escapes): after the dead layer the γ must travel ≥ 26.9 cm to the event, scatter by only θ₁₂ (2–18°), and then leave the active LXe without interacting. A straight line through a point at r = 45.9 cm in a cylinder of R = 72.8 cm has a horizontal chord ≥ 2√(R² − r²) = 113 cm; the shortest escape is downward through the cathode plane 26.4 cm below: minimising L(Δz) = √(26.9² + Δz²) + 26.4 √(1 + 26.9²/Δz²) gives **L_min = 75.4 cm** at Δz = 26.7 cm (plus 13.75 cm of RFR LXe to exit without a third S1). A naive "in and out" bound is 2 × 26.9 = 53.8 cm. Ordering "event first" (γ enters, deposits 12 keV, then reaches the wall dead layer) needs the same ≥ 113 cm chord or a steep entry; it is never shorter. A generic wall MSSI at the FV edge (mean stand-off 10.7 cm) needs ≈ 2 × 10.7 = 21.4 cm.

| E_γ (keV) | λ (cm) | θ for 12 keV | KN frac T∈[10,14] | KN frac T∈[70,84] | P(int. 3 mm) | e^{−53.8/λ} | e^{−75.4/λ} | e^{−21.4/λ} | suppression vs generic (54 / 75 cm) |
|---|---|---|---|---|---|---|---|---|---|
| 352 | 2.4 | 18.4° | 0.024 | 0.061 | 0.12 | 2e-10 | 3e-14 | 1.5e-4 | 1e-6 / 2e-10 |
| 500 | 3.7 | 12.9° | 0.014 | 0.042 | 0.078 | 5e-7 | 1.4e-9 | 3.1e-3 | 1.6e-4 / 5e-7 |
| 1000 | 6.0 | 6.4° | 0.0048 | 0.016 | 0.049 | 1.3e-4 | 3.5e-6 | 0.028 | 4.5e-3 / 1.2e-4 |
| 1461 | 7.5 | 4.4° | 0.0027 | 0.0094 | 0.039 | 7.6e-4 | 4e-5 | 0.057 | 0.013 / 7e-4 |
| 1764 | 8.3 | 3.6° | 0.0021 | 0.0072 | 0.036 | 1.5e-3 | 1.1e-4 | 0.076 | 0.020 / 1.5e-3 |
| 2615 | 9.8 | 2.4° | 0.0012 | 0.0042 | 0.030 | 4.1e-3 | 4.6e-4 | 0.11 | 0.037 / 4.1e-3 |

The energy-selection factors (12 ± 2 keV first scatter, 77 ± 7 keV dead-layer deposit) are *already* encoded in the {S1c, log10 S2c} PDF and hence in f_nb; the **position** factor is not (the paper states positions are not used in the inference). We therefore estimate a position penalty f_pos = fraction of wall-MSSI FV scatters at ≥ 25 cm from the true wall, with a toy radial pdf p(d) ∝ (R − d) exp(−n d/λ) for d ≥ 8 cm, n = 1 (one attenuated leg) or 2 (both legs):

| λ (cm) | n = 1 | n = 2 |
|---|---|---|
| 3.7 | 7.3e-3 | 7.5e-5 |
| 6.0 | 0.042 | 2.5e-3 |
| 8.3 | 0.090 | 0.012 |
| 9.8 | 0.12 | 0.022 |

f_pos ∈ [7 × 10⁻⁵, 0.12]; a defensible central range for the MeV γ's that dominate deep penetration is 0.003–0.05. Including f_pos, k_required(10 %) ≥ 619/0.12 = **5 × 10³** (up to 8 × 10⁶ for the most suppressed case). This is the quantitative version of the paper's remark that a 12 keV scatter more than 20 cm from any boundary "is unlikely for any MSSI category".

### 3.5 Veto efficiency

Science-sample wall MSSI ∝ (1 − ε), ε = 0.94 ± 0.02 → untagged fraction 0.06 ± 0.02, i.e. ±33 % on the 0.0048 (range 0.0032–0.0064 at 1 σ). The table itself implies ε = 0.09/(0.09 + 0.0048) = 0.949 for the 4.7 t WS ROI wall, 0.81 for the HE SB 5.4 t wall and 0.77 for the WS ROI 5.4 t wall. The three science sideband bins constrain the *untagged* normalisation directly, independently of ε: k_sci < 1.30 (90 %) / 2.15 (95 %) LR, 3.65 / 4.74 Bayesian. The total (tagged + untagged) wall MSSI in the 4.7 t WS ROI is 0.0948; even with zero veto efficiency the neighbourhood expectation would be 0.0948 × 0.0354 = 3.4 × 10⁻³ (P ≈ 0.3 %); an untagged fraction of 31 (i.e. > 100 %, impossible) would be needed for P = 10 % at k = 1. The veto systematic is therefore irrelevant to the conclusion.

## 4. Figure

`figures/P004_k_likelihood_and_Pge1.png` — Left: digitised Fig. 5 bottom panel (total and MSSI), event at −1.5 σ, ±2 σ window shaded; title gives the total-integral validation. Middle: −Δln L(k) for the six wall bins, science-only, prompt-only and the 20 % split-nuisance variant, with one-sided 90 %/95 % thresholds. Right: P(≥1 wall MSSI | k) anywhere in the WS ROI, in the neighbourhood (band = f_nb range), and additionally ≥ 25 cm from the wall (f_pos = 0.12); green = k allowed by the sidebands (95 %), orange = the pessimistic mis-classification extreme.

## 5. Robustness and failed approaches

- First digitisation pass assumed the frame edges were the 10² and 10⁻⁶ ticks; the total integrated to 0.0083 (78 % of the caption). Detecting the tick marks fixed this (0.01053, 99.3 %). Kept as a record of the calibration check.
- Using the −3…+1 σ window instead of ±2 σ changes f_nb from 0.035 to 0.030; the k_required numbers change by 15 %.
- If one insists on a k that scales only the inner-FV wall MSSI (i.e. rejects the common-k assumption), the sidebands give no constraint and only the paper's 100 % prior (k < 3) and the topology arguments remain; the required k ≈ 600 is then "unconstrained by data" but requires the dead volume to be ≈ 200 % of the active volume, which is absurd — the 0.3 % dead volume would need to be ×600.
- We did not attempt a full 2D PDF reconstruction of MSSI from Fig. S1a/S6 contours; the 1D Fig. 5 projection is the direct, validated route.

## 6. Extended discussion

The sideband table is dominated by the 5.4 t annulus (2.7 of the 3.23 predicted wall counts), where wall MSSI is most sensitive to the dead-layer geometry; a geometry error that increased the inner-FV wall rate by ×600 would increase these predictions by the same factor (≈ 1900 events expected, 2 seen). The only way around this is a *qualitatively different* wall-MSSI class that populates the inner FV without populating the outer annulus or the HE SB — but deep penetration is exactly what the attenuation argument forbids: the inner-FV population is the exponential tail of the outer one. Two things this paper does not address: RFR MSSI (disfavoured by the paper's S1 hit-pattern and rise-time analysis, and constrained by the 18/21.5 HE-SB agreement) and MSSI-like topologies absent from the model altogether (e.g. plate-out β in the dead layer with a penetrating ²¹⁴Bi γ; these still face the ≥ 54–75 cm traversal penalty, e^{−54/9.8} ≈ 4 × 10⁻³ at best).

## 7. Result tables

See `P004_k_limits.csv`, `P004_topology.csv`, `P004_fig5_digitised.csv`, `P004_results.json`.

## 8. References

1. LZ Collaboration, arXiv:2609.02823 (2026) — main text and supplement (MSSI sections, Table of MSSI counts, Fig. 5, Fig. S1a).
2. S. Baker and R. D. Cousins, Nucl. Instrum. Meth. 221, 437 (1984) — binned Poisson likelihood χ².
3. G. J. Feldman and R. D. Cousins, Phys. Rev. D 57, 3873 (1998) — unified intervals (our LR limits are the asymptotic one-sided analogue).
4. O. Klein and Y. Nishina, Z. Phys. 52, 853 (1929) — Compton cross-section.
5. M. J. Berger et al., NIST XCOM photon cross-section database (attenuation coefficients; values recalled).
6. LZ Collaboration, Phys. Rev. D 108, 012010 (2023) — LZ SR1 backgrounds (wall populations, detector radioactivity normalisation).
7. LZ Collaboration, arXiv:2410.17036 (2024) — WS2024 result (FV/wall-background criterion referenced by the paper).
8. Corpus: `output/00_evidence_dossier.md` (§4.3, §6.2).

## 9. Tools and provenance

Mirrors `output/provenance/P004.json`.
- Agent tools: Read (PAPER_GUIDE.md; 00_evidence_dossier.md; fulltext.tex lines 130–325, 440–585, 685–790; Fig5 PNG; FigS1a PNG; the P004 figure), Bash (grep of the tex for MSSI passages; listing of lzcommon/environment; PIL colour/frame probe; two script runs; tick-mark probe; wc -w), Write (script, details.md, P004.json, P004.md), Edit (script ×3).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (stats.poisson, stats.chi2, stats.truncnorm, optimize.minimize_scalar, integrate.quad); matplotlib 3.11.2 (Agg); Pillow 12.3.0 (PIL.Image) for pixel reading; common/lzcommon.py (imported; LZ constants).
- Recalled knowledge (3 items): LXe γ attenuation lengths 0.3–9.8 cm for 122–2615 keV (likely, ±20 %); LZ TPC active radius 72.8 cm (likely); electron mass and Klein–Nishina formula (certain).
- Datasets: none. Data requests: none. WimPyDD files: none.
