# P053 — ²¹⁴Pb in the mixed-flow state: what the unavailable radon tag costs and whether ²¹⁴Pb-driven MSSI could be elevated around 16 June 2023

Simulated date 2026-09-11 · category BKG · physics.ins-det (cross-list hep-ex) · author profile: radon-background experts.
Script: `output/code/P053_pb214_mixed_flow.py` (run from the root with `.venv/bin/python`; ~20 s). Machine-readable results: `P053_results.json`, `P053_geometry_points.csv`, full console log `P053_run.log`; figures in `figures/`.

## 1. Motivation and framework

LZ's Discussion (tex l. 303) states that the radon tag of Ref. [LZ:2025xxf] ("Flow-dependent tagging of ²¹⁴Pb decays in the LZ dark matter detector", arXiv:2508.19117) "cannot be applied to the event of interest as it occurred while the detector was in the mixed flow state". ²¹⁴Pb β+γ decays are one of the two dominant sources of the MSSI background (l. 187–189); the ²¹⁴Pb MSSI rate "is constrained in the same way as the SS ²¹⁴Pb component" (l. 193), i.e. through the single-scatter ER rate. The dossier (§2, "Radon tag" row; §5 hypothesis B) lists "²¹⁴Pb in mixed-flow" as an open background route. We ask four questions:

(a) what fraction of the 220 live days was mixed-flow, from the paper's own tag statistics (supplement l. 729–730: 12 of the 18 HE-SB RFR *science* events in tag-active periods, 7 tagged; 1 wall + 2 RFR *prompt* events in tag-active periods, one tagged);
(b) how the flow state can change the ²¹⁴Pb SS and MSSI rates (transport of daughters during their lifetime);
(c) what ²¹⁴Pb activity Table I implies, and what elevation factor a ²¹⁴Pb-MSSI origin of the candidate would need, versus what the ER and α rates allow;
(d) what a negative or positive tag would have proved.

Radon-tag principle (recalled, likely): ²²²Rn → ²¹⁸Po (α, 3.10 min) → ²¹⁴Pb (β, 26.8 min) → ²¹⁴Bi (β, 19.9 min) → ²¹⁴Po (α, 164 μs). A ²¹⁴Pb β can be identified by a preceding ²¹⁸Po α (parent) or a following ²¹⁴Bi–²¹⁴Po delayed coincidence (daughters) at a position displaced by the xenon flow and ion drift; in the "laminar" (tag-active) flow state of LZ's 2024 run the displacement is predictable and ~60% of ²¹⁴Pb decays are tagged, in the "mixed" state the correlation is scrambled and the tag is off (recalled from the title of Ref. [LZ:2025xxf] and LZ WS2024 arXiv:2410.17036; flagged likely/uncertain).

## 2. Inputs

| Input | Value | Source |
|---|---|---|
| Live days, FV mass, ROI | 220 d, 4.71 t, S1c 3–600 phd, log₁₀S2c 2.75–4.15, S2 raw > 645 phd | paper / `lz.LZ` |
| Event position | 26.4 cm above cathode; 26.9 (23.4) cm from true (reconstructed) wall | tex l. 166 |
| FV cuts | ≥ 8.0 cm from true wall (mean 10.7 cm), 12.8 cm below gate, 9.0 cm above cathode | tex l. 137–139 |
| Charge-dead regions | 13.75 cm RFR below cathode; ≤ 3 mm scalloped wall shell (0.3 % of active volume) | tex l. 186, 195–196 |
| MSSI counts | Table S (tex l. 773–780): WS ROI 4.7 t wall 0.0048, RFR 0.0001; HE-SB RFR 5.4 t 21.5 predicted / 18 observed; HE-SB RFR 4.7 t 0.5 / 0 | supplement |
| Radon-tag statistics | 12 of 18 HE-SB RFR science events in tag-active periods, 7 tagged (60 % efficiency); prompt: 1 wall + 2 RFR in active periods, 1 tagged | tex l. 729–730 |
| Internal β decays | 1341 ± 160 (dominated by ²¹⁴Pb, ²¹²Pb) | Table I |
| Science-sample counts | 1710 observed (7.77 per day) | Table I |
| Prompt-veto efficiency for MSSI | 94 ± 2 % | tex l. 194 |
| RFR light yield | 471 phd ↔ 204 keV → 2.31 phd/keV | tex l. 734 |
| P004 | k_required(10 %, ±2σ_NR neighbourhood) = 619; μ_nb(all MSSI) = 1.7 × 10⁻⁴ | P004.md |
| P029 | λ = 1.42 cm (243 keV), 2.9 cm (375 keV) | P029.md |
| NEST-LZ ER yields | `lz.nest_er_yields` (nestpy 2.1.1, Table S3 parameters, 96.5 V/cm) | lzcommon |
| Chain half-lives | radioactivedecay 0.6.1 (ICRP-107) | software |

Recalled inputs (all flagged in the script's `RECALLED` dict and in `P053_results.json`): TPC radius 72.8 cm and drift length 145.6 cm (likely); LXe density 2.9 g cm⁻³ (likely); ²¹⁴Pb Q_β = 1019 keV, levels 351.9/295.2/53.2 keV, γ intensities per decay 35.6 % (352), 18.4 % (295), 7.3 % (242), 1.1 % (53) (likely); ground-state β branch 9.2 % (alternative 11 %) and β feedings 46/41/2 % (uncertain); LXe attenuation lengths 2.7/2.2/1.5 cm at 352/295/242 keV with ±15 % brackets (likely; consistent with P029); Compton share of the total cross-section 0.55/0.45/0.35 (uncertain); LZ purification throughput 1–4 t/day (uncertain); positive-ion mobility in LXe 10⁻⁴–10⁻³ cm² V⁻¹ s⁻¹ (uncertain); LZ ²²²Rn 1–5 μBq/kg (uncertain); false-tag rate 1–10 % (uncertain); active LXe mass 7 t (likely); 5.4 t volume radius 67.3 cm and bottom cut 2–5 cm (derived/uncertain). Assumption: ²¹⁴Pb contributes 50–90 % (central 70 %) of the 1341 internal β events (the rest ²¹²Pb, ⁸⁵Kr).

## 3. Section 1 — tag-active and mixed-flow fractions

Binomial with Clopper–Pearson intervals. If the ²¹⁴Pb RFR-MSSI rate does not depend on the flow state, the fraction of HE-SB events in tag-active periods estimates the tag-active livetime fraction.

| Sample | k/n | f_active | 68 % | 95 % | mixed-flow days (68 %) |
|---|---|---|---|---|---|
| HE-SB RFR science | 12/18 | 0.667 | 0.52–0.79 | 0.41–0.87 | 73 (46–105); 95 %: 29–130 |
| HE-SB prompt | 3/3 | 1.0 | 0.54–1 | 0.29–1 | 0 (0–101) |
| combined | 15/21 | 0.714 | 0.58–0.82 | 0.48–0.89 | 63 (39–92) |

Tag efficiency check: 7/12 = 0.58 (68 % 0.40–0.75; 95 % 0.28–0.85), P(≥ 7 | ε = 0.6) = 0.67 — consistent with LZ's 60 %.

P(a randomly timed event falls in a mixed-flow period) = 1 − f_active = 0.33 (68 % 0.21–0.48; 95 % 0.13–0.59). The candidate's mixed-flow timing is therefore a 1-in-3 coincidence, not an anomaly.

Degeneracy: if the RFR-MSSI rate differs between states by a factor ε = (6/f_m)/(12/(1−f_m)), then with the true mixed-flow livetime fraction f_m: ε = 4.5 (f_m = 0.1), 2.0 (0.2), 1.0 (0.33), 0.5 (0.5), 0.25 (0.67). Consistency of the 6 mixed-flow HE-SB RFR events with 21.5·f_m expected: f_m = 0.1 → expected 2.15, P(≥ 6) = 0.023, ε_ML = 2.8, ε < 5.5 (95 %); f_m = 0.2 → 4.3, P(≥ 6) = 0.26, ε_ML = 1.4, ε < 2.8; f_m = 0.33 → 7.1, ε_ML = 0.85, ε < 1.67; f_m = 0.5 → 10.75, P(≤ 6) = 0.089, ε < 1.10. Only LZ's flow-state timeline can break this degeneracy.

## 4. Section 2 — chain and transport

Half-lives (radioactivedecay 0.6.1): ²²²Rn 3.8235 d, ²¹⁸Po 186.0 s (3.10 min), ²¹⁴Pb 1608 s (26.8 min), ²¹⁴Bi 1194 s (19.9 min), ²¹⁴Po 164.3 μs. Mean life of ²¹⁴Pb τ = 2320 s = 38.7 min (mean delay between the parent ²¹⁸Po α and the ²¹⁴Pb β); from the ²²²Rn α, 43.1 min. Bateman check: a pure ²²²Rn injection of 1 Bq gives ²¹⁴Pb activities of 0.48/0.76/0.94/0.97/0.84 Bq after 0.5/1/2/4/24 h.

Displacement during one ²¹⁴Pb mean life, d = v τ: v = 0.01/0.03/0.1/0.3/1/3/10 mm s⁻¹ → 2.3/7.0/23/70/232/696/2320 cm. Bulk throughput velocity through the 1.66 m² TPC cross-section for 1–4 t/day: 0.0024–0.0096 mm s⁻¹ → 0.6–2.2 cm (negligible). Ion drift of charged daughters at 96.5 V cm⁻¹ with μ = 10⁻⁴–10⁻³ cm² V⁻¹ s⁻¹: 0.10–0.97 mm s⁻¹ → 22–224 cm, comparable with the 145.6 cm drift length. Convective flows of order mm s⁻¹ would likewise move daughters by metres. Consequence for (b): the flow state changes the spatial distribution of ²¹⁴Pb (e.g. accumulation near the cathode/RFR by ion drift, or its suppression when the flow is mixed), and therefore the MSSI/SS ratio, but not the total activity; the SS rate in the FV measures the FV activity in either state. LZ's assumption of ²¹⁴Pb "distributed throughout the LXe" is untested state by state; the 6-versus-12 split above is the only handle in the paper.

## 5. Section 3 — ROI ER window, β acceptance, ²¹⁴Pb activity

NEST-LZ mean yields (`lz.nest_er_yields`): S1c = 3 phd at E = 1.35 keV; log₁₀S2c = 4.15 at E = 18.9 keV; S2 raw = 645 phd at 0.25 keV (not limiting). The WS ROI therefore accepts β energies 1.35–18.9 keVee (mean-yield basis; fluctuations smear the edges). For the HE-SB, log₁₀S2c < 4.30 gives a β window 1.35–29.1 keV.

β spectrum: N(T) ∝ F(Z = 83, T) p E (Q − T)² with the non-relativistic Fermi function F = 2πη/(1 − e^{−2πη}), η = αZE/p (allowed shape; the 0⁺ → 1⁻ ground-state transition is first-forbidden non-unique, treated as allowed — a shape uncertainty of order 10–20 % at these energies).

| Branch (endpoint keV) | ROI 1.35–18.9 | 10–14 keV | HE-SB 1.35–29.1 | flat ROI |
|---|---|---|---|---|
| ground state (1019) | 2.18 % | 0.50 % | 3.48 % | 1.72 % |
| → 352 level (667) | 4.35 % | 0.99 % | 6.89 % | 2.63 % |
| → 295 level (724) | 3.83 % | 0.88 % | 6.09 % | 2.43 % |
| → 53 level (966) | 2.39 % | 0.55 % | 3.81 % | 1.82 % |

Fermi enhancement over a flat spectrum: ×1.27.

Activity inversion: N_ROI(²¹⁴Pb) = N_dec [BR_gs A_gs + Σ_γ I_γ A_exc P_escape w], with P_escape the FV-averaged probability that the γ leaves the active LXe unscattered (Section 6) and w = 0–1 the fraction of such escaped-γ events that survive the Skin/OD veto (central 0.5). Central (share 0.7, BR_gs 9.2 %, w = 0.5): per-decay ROI probability 2.01 × 10⁻³, N_dec = 4.66 × 10⁵ per run = 2120 decays/day in the FV = **5.2 μBq/kg**. Grid over share 0.5/0.7/0.9, BR 9.2/11 %, w 0/0.5/1: 3.1–6.7 μBq/kg. This is at the upper end of the recalled 1–5 μBq/kg range for LZ (uncertain); the number is only as good as the ²¹⁴Pb share of the 1341 and the ground-state branch. ²¹⁴Pb ROI ER rate: 3.0/4.3/5.5 per day for share 0.5/0.7/0.9; total ROI ER rate 7.77 per day.

## 6. Section 4 — γ ray-tracing in the TPC

Isotropic rays from (r₀, z₀) in a cylinder of radius 72.8 cm and height 145.6 cm; the first active-volume boundary is the inner face of the 3 mm wall shell (r = 72.5 cm), the cathode plane (z = 0) or the gate plane (z = 145.6 cm). Survival exp(−L/λ); probability of interacting in the dead region 1 − exp(−ΔL/λ) with ΔL the path inside the 3 mm shell or inside the 13.75 cm RFR. Validation: a point on the axis 5 cm above the cathode gives P(reach RFR) = 0.02274/0.01317/0.00353 (MC) versus ½E₂(d/λ) = 0.02279/0.01320/0.00355 (analytic half-space) for 352/295/242 keV.

| Point | γ | reach wall shell | reach RFR | interact in shell | interact in RFR | escape any | straight-line e^{−d/λ} wall / RFR |
|---|---|---|---|---|---|---|---|
| event (r 45.9, z 26.4) | 352 | 2.9e-6 | 2.4e-6 | 3.2e-7 | 2.4e-6 | 5.3e-6 | 5.3e-5 / 5.7e-5 |
| | 295 | 2.6e-7 | 2.2e-7 | 3.4e-8 | 2.2e-7 | 4.8e-7 | 5.6e-6 / 6.1e-6 |
| | 242 | 6.4e-10 | 5.8e-10 | 1.2e-10 | 5.8e-10 | 1.2e-9 | 2.0e-8 / 2.3e-8 |
| reco-wall distance 23.4 cm | 352 | 1.1e-5 | 2.4e-6 | 1.3e-6 | 2.4e-6 | 1.4e-5 | 1.9e-4 / 5.7e-5 |
| FV wall edge (r 64.8, z 26.4) | 352 | 6.9e-3 | 2.0e-6 | 8.7e-4 | 2.0e-6 | 6.9e-3 | 5.8e-2 / 5.7e-5 |
| FV bottom (r 45.9, z 9.0) | 352 | see CSV | | | | | |
| 5.4 t bottom (r 45.9, z 2.0) | 352 | 1.7e-6 | 0.110 | 1.9e-7 | 0.110 | 0.110 | 5.3e-5 / 0.48 |

(full table for all points and energies: `P053_geometry_points.csv`). The 352 keV wall+RFR interaction probability at the event is 2.7 × 10⁻⁶ (λ bracket 2.3–3.1 cm: 4.4 × 10⁻⁷–1.1 × 10⁻⁵). The paper's remark that a 12 keV Compton vertex needs > 60 cm of traversal (l. 735) concerns Compton scattering of an external γ; for the ²¹⁴Pb route the γ is emitted at the vertex and must merely cross ≥ 26.4 cm, and even that costs 10⁻⁵–10⁻⁹ per emitted γ.

Volume averages (uniform ²¹⁴Pb; r-weighted grid 14 × 14, 2 × 10⁴ rays per point): FV (r < 62.1 cm, 9 < z < 132.8 cm; 4.35 t cylinder): 352 keV interact-in-shell 3.3 × 10⁻⁵, interact-in-RFR 2.1 × 10⁻⁴, escape 5.2 × 10⁻⁴; (r < 64.8 cm: 1.1 × 10⁻⁴, 2.0 × 10⁻⁴, 1.1 × 10⁻³). 5.4 t volume (r < 67.3 cm): z > 5 cm: 3.5 × 10⁻⁴ / 1.27 × 10⁻³ / 4.0 × 10⁻³; z > 2 cm: 3.4 × 10⁻⁴ / 5.96 × 10⁻³ / 8.6 × 10⁻³.

Cross-checks against LZ's MSSI table (our model: β in the relevant window, γ reaching the dead region unscattered, Compton share × Klein–Nishina fraction with T_e ≤ 100 keV × 0.8 outward escape for the wall shell; 0.8 full-energy deposition for the RFR; 6 % prompt-veto survival for wall MSSI):
- WS ROI wall MSSI, 4.7 t, science sample: **0.0037** from ²¹⁴Pb alone (0.062 before the veto) versus LZ 0.0048 for all sources → a ²¹⁴Pb share of ≈ 0.77 if taken at face value (crude; used only as a bracket).
- HE-SB RFR, 5.4 t: 352-line only 14.1, all lines 18.0 for a 5 cm bottom cut, versus LZ 21.5 predicted / 18 observed; a 2 cm bottom cut gives 68–95 (the RFR count is dominated by decays within ~2λ of the cathode, so the bottom cut of the 5.4 t volume, which we do not know, controls it).
- HE-SB RFR, 4.7 t (z > 9 cm): 1.9–2.2 versus LZ 0.5 (×4 high; the FV contour narrows near the bottom — up to 18.2 cm from the wall — and the 352 keV full-absorption S1c ≈ 812 phd sits at the 800 phd edge, both lowering LZ's number).
- WS ROI RFR MSSI: 0.19 versus LZ 0.0001 — **failed**: our Compton-in-RFR-then-escape factor (0.2) is far too generous for a 13.75 cm thick dead layer; this quantity is not used anywhere.

The wall and HE-SB comparisons show that the activity of Section 5 and the geometry are right to within a factor ~2–3 in absolute terms; ratios (position penalties) are much better determined.

## 7. Section 5 — elevation factors versus rate constraints

Per-decay probability of the hypothesised topology (β 10–14 keV in the FV, γ interacting in a dead region) at the event: p_ev = 1.0 × 10⁻⁸; FV average (10.7 cm stand-off) 1.0 × 10⁻⁶; FV average (8 cm) 1.3 × 10⁻⁶; FV wall edge 3.9 × 10⁻⁶. Position penalty p_ev/⟨p⟩_FV = **0.010** (0.0076 for the 8 cm average; λ bracket 0.0019–0.045; 0.0026 relative to the FV edge). Expected such topologies per run in the whole FV at k = 1 (before any energy/veto requirement): 0.46; at the event's per-decay probability × N_dec: 0.0047.

Required elevation of the ²¹⁴Pb activity for a 10 % chance of one event in the ±2σ_NR neighbourhood (μ_target = −ln 0.9 = 0.105; μ_nb(²¹⁴Pb) = 1.7 × 10⁻⁴ s):

| ²¹⁴Pb share s | whole run | 24 h | 6 h | 1 h |
|---|---|---|---|---|
| 0.3 | 2063 | 4.5e5 | 1.8e6 | 1.1e7 |
| 0.5 | 1238 | 2.7e5 | 1.1e6 | 6.5e6 |
| 0.7 | 884 | 1.9e5 | 7.8e5 | 4.7e6 |

(k_T = 1 + μ_target/μ_nb × 220 d/T). Multiplying by 1/0.010 for the position penalty gives 2.7 × 10⁷ (24 h) and 6.5 × 10⁸ (1 h) at s = 0.5.

Allowed by the ROI ER rate (baseline 7.77/day total, ²¹⁴Pb 4.3/day at s = 0.7; Poisson, smallest n with P(≥ n | b) < 1.35 × 10⁻³ or 2.87 × 10⁻⁷):

| window T | baseline b | 3σ: n, k_min | 5σ: n, k_min | extra events at k = 10 |
|---|---|---|---|---|
| 1 h | 0.32 | 4, 21.7 | 7, 38.6 | 1.6 |
| 3 h | 0.97 | 6, 10.4 | 10, 17.9 | 4.8 |
| 6 h | 1.94 | 8, 6.7 | 13, 11.4 | 9.6 |
| 12 h | 3.89 | 12, 4.8 | 18, 7.6 | 19.2 |
| 24 h | 7.77 | 18, 3.4 | 26, 5.3 | 38.4 |
| 72 h | 23.3 | 40, 2.3 | 52, 3.2 | 115 |
| 7 d | 54.4 | 79, 1.8 | 96, 2.4 | 269 |

Gap required/allowed at s = 0.5: 24 h — 2.7 × 10⁵ / 3.4 = 8 × 10⁴ (8 × 10⁶ with the position penalty); 1 h — 6.5 × 10⁶ / 22 = 3 × 10⁵ (3 × 10⁷). A whole-run elevation of ~10³ is excluded trivially by the measured 1341 ± 160 internal-β count (≈ 12 %).

α monitor: at 5.2 μBq/kg the 7 t active volume has 131 ²¹⁸Po α (and 131 ²¹⁴Po α) per hour; a factor k excursion is 5σ-detectable after 25/(A (k−1)²): k = 1.5 in 46 min, k = 2 in 11 min, k = 3 in 2.9 min, k = 10 in 8 s. The "no anomalous populations" statement of the paper (l. 302) is thus, in principle, a ≲ ×2 constraint over any hour-scale window — provided LZ publishes the BiPo/α time series.

## 8. Section 6 — tag counterfactual

With ε = 0.6 (68 % range 0.40–0.75 from 7/12) and false-tag rate f_false: LR(negative) = (1−ε)/(1−f_false) = 0.404/0.412/0.444 for f_false = 1/3/10 % (ε range: 0.25–0.67); LR(positive) = ε/f_false = 60/20/6 (ε range 40–75, 13–25, 4.0–7.5). Expected ln LR (f_false = 3 %): +1.44 nats if the event is a ²¹⁴Pb decay, −0.77 nats if not. Illustrative posteriors: prior P(²¹⁴Pb MSSI) = 10⁻³/10⁻²/0.1 → 4 × 10⁻⁴/4 × 10⁻³/0.044 after a negative tag; 0.020/0.17/0.69 after a positive tag. A negative tag would have been nearly uninformative (factor 2.4 in odds); a positive tag would have been decisive against the NR interpretation. The unavailable tag therefore cost the possibility of *refuting* the DM interpretation, not of confirming it.

## 9. Figures

- `figures/P053_fig1_gamma_reach.png` — probability that a 352/295/242 keV γ reaches a charge-dead region unscattered versus distance (half-space ½E₂(d/λ)), with the 8 cm FV margin and the event's 26.4/26.6 cm distances marked; dot: ray-traced event value (352 keV, wall + RFR, 5.3 × 10⁻⁶).
- `figures/P053_fig2_elevation_vs_window.png` — elevation factor required for a 10 % ²¹⁴Pb-MSSI explanation versus window duration (with and without the position penalty), against the 3σ and 5σ detectability thresholds from the ROI ER rate.

## 10. Robustness, failed approaches, discussion

- The event-weighted mixed-flow fraction equals the livetime fraction only if the RFR-MSSI rate is flow-independent (Section 3 table gives the mapping otherwise).
- Activity: ×0.6–1.3 from the ²¹⁴Pb share, ground-state branch and escaped-γ term; the HE-SB cross-check (14–18 vs 21.5 for a 5 cm bottom cut) supports the central value but is itself sensitive to the unknown 5.4 t bottom cut (×4 between 2 and 5 cm).
- Position penalty: 0.0019–0.045 across the λ bracket; 0.0076–0.010 across the FV stand-off; using the reconstructed-wall distance (23.4 cm) raises the event probability by ×1.4 (shell) but not the RFR term.
- Failed: the RFR WS-ROI estimate (0.19 vs LZ 10⁻⁴) — the Compton-escape treatment for a thick dead layer is inadequate; excluded from all conclusions. An attempt to infer the ²¹⁴Pb share of MSSI from our absolute wall estimate (0.77) is reported but only the 0.3–0.7 bracket is used.
- Not treated: ²¹⁴Bi (BiPo-tagged by LZ), plate-out on the cathode (LZ Ref. [LZ:2026hpq]), ²¹²Pb MSSI (Q = 570 keV, 239 keV γ — same geometry, lower activity), Skin/OD veto for escaped γ's beyond the 6 % factor.

Conclusion for hypothesis B (dossier): the ²¹⁴Pb-in-mixed-flow route cannot produce the candidate at its position; the mixed-flow timing has P = 0.33; the ²¹⁴Pb activity is pinned by the SS rate; the only residual freedom is the flow-state dependence of the MSSI/SS ratio, which LZ can close by publishing the flow timeline, the per-state HE-SB RFR counts and the ±24 h BiPo/α rates.

## 11. References

LZ Collaboration, arXiv:2609.02823 (2026); J. Aalbers et al. (LZ), Flow-dependent tagging of ²¹⁴Pb decays in the LZ dark matter detector, arXiv:2508.19117 (2025); J. Aalbers et al. (LZ), Dark matter search results from 4.2 tonne-years of exposure of the LUX-ZEPLIN experiment, arXiv:2410.17036 (2024); A. Malins and T. Lemoine, radioactivedecay, JOSS 7, 3318 (2022) (ICRP-107 data); C. J. Clopper and E. S. Pearson, Biometrika 26, 404 (1934); M. J. Berger et al., NIST XCOM; O. Klein and Y. Nishina, Z. Phys. 52, 853 (1929); corpus P004, P010, P029, P041; dossier 00.

## 12. Tools and provenance

Mirrors `output/provenance/P053.json`. Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (stats.beta/binom/poisson, integrate.quad, optimize.brentq, special.expn); matplotlib 3.11.2; radioactivedecay 0.6.1 (Nuclide.half_life/progeny, Inventory.decay); nestpy 2.1.1 via `common/lzcommon.py` (`nest_er_yields`, `LZ`, `DRIFT_FIELD_VCM`). Script: `output/code/P053_pb214_mixed_flow.py`. Local inputs: PAPER_GUIDE.md; 00_evidence_dossier.md; results_ledger.csv (rows P001–P013 and grep of later rows); P004.md, P029.md, P010.md, P041.md; P033 details head (unpublished, not cited); fulltext.tex l. 128–139, 165–244, 296–320, 708–785 and grep for radon/MSSI/flow; lzcommon.py l. 18–75; ENVIRONMENT_versions.txt. Recalled knowledge: 14 items (Section 2). Datasets: none. Data requests: none. Failed/abandoned: RFR WS-ROI Compton-escape estimate; inferring the ²¹⁴Pb MSSI share from the absolute wall estimate.
