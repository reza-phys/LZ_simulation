# P079 — Optimal ROI and fiducial design for the next LZ high-energy analysis: trading fiducial mass against MSSI with a 3D likelihood

Simulated date 2026-09-15. Category PROJ (hep-ex, cross-list physics.ins-det). Author profile: analysis-optimisation experimentalists. Script `output/code/P079_roi_fv_optimisation.py` (run from the simulation root with `.venv/bin/python`; log `run_log.txt`; results `P079_results.json`, `P079_scan_familyA.csv` (3465 configurations), `P079_recommendation_table.csv`; figures in `figures/`). Runtime 13 s.

## 1. Motivation and question

LZ's extended search (arXiv:2609.02823) kept the 2023 S1c < 600 phd edge and shrank the fiducial volume by 14.5 % (5.5 → 4.71 t) with an 8.0 cm minimum stand-off from the true wall (6.0 cm from the reconstructed wall; mean 10.7 cm), z from 9.0 cm above the cathode to 12.8 cm below the gate, the radial contour set by "wall backgrounds < (1.0 ± 0.5) × 10⁻⁴ events" over the whole 3–600 phd ROI (Data Analysis, lines 132–139). The two dominant high-energy backgrounds are surface-concentrated: wall MSSI falls as e^{−d/λ} from the wall (P033: λ = 4.3 cm; P070: 4.8 cm re-fitted in the real FV; prompt-veto data 3.3 cm) and RFR MSSI as e^{−z/λ_z} from the cathode (P070: ²¹⁴Pb 2.7 cm, detector-γ 6.5 cm), while the signal is uniform. P038 showed that moving the edge to 1000 phd (E50 = 423 keV) costs 0.003 NR-band MSSI events while accepting ≥ 90 % of every inelastic spectrum. We ask: (a) which (s, z_b) maximise the discovery sensitivity of the 200 keV–E50 window, and is LZ's 8 cm optimal or a low-energy legacy; (b) what a position-dependent ("soft FV") likelihood adds; (c) the joint (edge, s, z_b) optimum for L10 and inelastic δ = 350/366 keV and the 5σ exposure in LZ's untouched 524 live days (P020); (d) the systematic cost at small stand-off and a robust recommendation.

## 2. Model

### 2.1 Geometry
- True wall R = 72.8 cm (recalled, likely; P033/P070), gate–cathode 145.6 cm (recalled, likely; P070 confirmed from Fig. 3), ρ_LXe = 2.86 g/cm³ (recalled, likely). Upper boundary z_t = 132.8 cm (text). LZ cuts: s = 8.0 cm, z_b = 9.0 cm.
- Baseline contour: P070's vector digitisation of Fig. 3, `fv_contours.csv`, column `r_mean47(z)` (mean of the min/max radial-extent contours, reconstructed coordinates; 583 samples), and the dashed reconstructed wall `r_wall_reco(z)`; 5.4 t contours `r_min54/r_max54` (mean used), drawn z-range 2.15–135.75 cm.
- Three families parametrised by the minimum true-wall stand-off s: **A "shift"** r(z; s) = r_c(z) − (s − 8) (keeps LZ's bottom flare; primary); **B "cylinder"** r = R − s; **C "reco-margin"** r(z; m) = r_reco(z) − m.
- Mass M = ρ π ∫ r(z)² dz. Raw mass at LZ's cuts 4.267 t (P070: 4.36 t with the drawn z-range, 4.27 t with the text z-range — reproduced). Calibration factor κ_M = 4.71/4.267 = **1.104** applied to all masses; the same factor turns the 5.4 t contour's raw 4.914 t into **5.42 t**, and the active volume is 6.93 t (LZ 7.0 t), so the 8–10 % shortfall is a common scale (P070's unresolved tension) that cancels in every ratio.
- Stand-off statistics of the mean contour from the true wall: min 7.75, mean 10.89 (paper 10.7), max 15.87 cm (paper 18.2 for the min-extent contour). Distance to the reconstructed wall: mean 7.30 (paper 7.2), min 6.21 cm. Reconstructed-wall inward offset from the true wall: ≈ 0 at the top, 3.63 cm at mid-height, 6.22 cm at the bottom — this offset is why LZ's contour flares to 16–18 cm at the bottom, and why the cylinder family B is infeasible there (d_reco,min = 0.6 cm at s = 6).
- Calibrated masses. Family A, z_b = 9: M(s) = 5.67/5.50/5.34/5.18/5.02/**4.71**/4.41/4.12 t for s = 2/3/4/5/6/8/10/12 cm (dM/ds ≈ 0.155 t/cm, 3.3 %/cm). z_b at s = 8: 5.00/4.93/4.84/4.77/4.71/4.61 t for z_b = 0/2/5/7/9/12 cm (0.033 t/cm). Family B: 6.15/5.98/5.81/5.64/5.48/5.16/4.84/4.54 t. Family C: 5.71/5.55/5.38/5.22/4.91 t for m = 1/2/3/4/6 cm.

### 2.2 Wall-MSSI depth scale: P033 versus LZ's own table
Wall MSSI in a volume V: N_w(V) = N_w(FV_LZ) · I(V)/I(FV_LZ), with I(V) = ∫_V e^{−(R−r)/λ} dV (radial integral analytic: 2π[(λr − λ²)e^{−(R−r)/λ}]). The LZ MSSI table (supplement, science sample, WS ROI) gives 0.0048 in the 4.7 t FV and 0.03 in the *disjoint* 5.4 t annulus (0.7 t), ratio 6.25. Our integrals of the digitised contours give annulus/FV = 1.34/0.86/0.76/0.60/0.46 for λ = 3/4.3/4.8/6/8 cm; the ratio 6.25 requires **λ = 1.23 cm** (radial part alone, same z-range: 1.16 cm). The HE-SB wall rows (0.5/0.1 = 5.0) give 1.36 cm; the prompt-veto rows (2.2/0.09 = 24) would need 0.75 cm (prompt-vetoed wall MSSI live in the annulus). LZ's simulated wall-MSSI model is therefore ~7× more edge-concentrated than P033's photon-transport 12 keV class (λ = 4.3 cm) — plausible if the LXe deposit of typical wall MSSI is a photoabsorption or large-angle scatter of a ²¹⁴Pb 352/295 keV γ (λ_att = 2.7/2.2 cm) rather than a shallow-angle Compton of a MeV line. We carry both: λ = 4.3 cm (central, P033) with 3.3 (P070 prompt-veto data) and 6 cm as the bracket, and λ = 1.23 cm as the "LZ-table" scenario. The two agree by construction at s = 8 and diverge inward (§4.4).

### 2.3 RFR-MSSI z-scale
N_r(V) = 1 × 10⁻⁴ · J(V)/J(FV_LZ), J = ∫_V e^{−z/λ_z} dV. The 5.4 t annulus (z down to 2.15 cm) holds 0.003, ratio 30; our integrals give 29.8/11.8/4.7/1.9 for λ_z = 2.0/2.7/4.0/6.5 cm → **λ_z = 2.00 cm** reproduces LZ's model, consistent with the ²¹⁴Pb E₂ scale (2.7 cm at 352 keV, P070). Central 2.7, bracket 2.0–6.5 cm.

### 2.4 Wall leakage and the reconstruction floor
The two contour criteria (1 × 10⁻⁴ in the 4.7 t FV, 1 × 10⁻² in the 5.4 t volume, both for the whole 3–600 phd ROI) and the mean reconstructed stand-offs of the two contours (7.30 and 5.01 cm) imply an empirical scale λ_leak = 2.29/ln 100 = **0.50 cm**: the wall population that sets LZ's contour is a near-step function of the reconstructed radius, dominated by low-S2 events with cm-scale position resolution. At S2c ≈ 10⁴ phd the resolution is ≈ 3 mm (P041: σ_T = 3.07 mm), so for the high-energy window we impose a **reconstruction floor d_reco ≥ 3 cm everywhere** (≈ 10σ) instead of a background term, and keep the low-energy exponential only as a diagnostic (`b_leak_lowE`). Family A reaches the floor at s = 4.79 cm; family C by construction at m = 3.

### 2.5 Backgrounds in the NR band (±2σ) of 200 keV–E50(edge), per 220 live days
- Wall MSSI: 0.0048 × f_nb(0.035, P004) = 1.68 × 10⁻⁴ at 600 phd, plus P038's added MSSI for the extended edge (mid of the exp/pow models: wall 0.0145, RFR 0.0043 at 1000 phd) × NR-band share 0.15 (P038: 0.0027 of 0.018) → wall 2.34 × 10⁻³, RFR 6.45 × 10⁻⁴ at 1000 phd (total 2.98 × 10⁻³ vs P038's 0.0027 + 0.00017).
- Uniform: accidentals 1.5 × 10⁻⁴ (P022) + atm-ν 3.4 × 10⁻⁵ (P019) + residual ≈ 2 × 10⁻⁵ → 2.0 × 10⁻⁴ (bracket 4.0 × 10⁻⁴ = P016's b_H minus MSSI), ∝ M/4.71 t, plus 1.03 × 10⁻⁴ per 100 keV of window above 270 keV (P050). ER leakage above 250 keV neglected (LZ Fig. 5; P010).
- LZ cuts: 600 phd total 3.73 × 10⁻⁴ (wall 1.68 × 10⁻⁴, RFR 3.5 × 10⁻⁶, uniform 2.02 × 10⁻⁴); 1000 phd 3.34 × 10⁻³. MSSI normalisation factor k = 0.5–2 (LZ's 100 % uncertainty).

### 2.6 Signal
Spectra from P050's WimPyDD cache (1 TeV, Baxter-2021 halo, 12-day annual average): L10 (dipole–dipole reduction, P012) and inelastic O₁ at δ = 300/350/366 keV. Efficiency 0.955 × erf roll-off with P038's (E50, σ): 600 → (271.6, 10.9), 700 → (310.5, 11.7), 800 → (348.7, 12.7), 1000 → (423.2, 14.5), 1200 → (495.1, 18.0) keV; resolution σ_E = 11√(E/248) keV; window E_obs ≥ 200 keV. Acceptance (fraction of the full spectrum):

| edge | L10 | δ300 | δ350 | δ366 |
|---|---|---|---|---|
| 600 | 0.154 | 0.161 | 0.235 | 0.137 |
| 700 | 0.191 | 0.181 | 0.308 | 0.250 |
| 800 | 0.218 | 0.224 | 0.471 | 0.505 |
| 1000 | 0.292 | 0.288 | 0.705 | 0.866 |
| 1200 | 0.374 | 0.300 | 0.743 | 0.917 |

Ratios A(1000)/A(600) = **1.89 / 1.79 / 3.00 / 6.32**; A(800)/A(600) = 1.41/1.39/2.00/3.69 (P038: 0.44 → 0.91 at δ = 350, 0.12 → 0.90 at 370, consistent). Signal normalisation (per assignment and P020/P069): 1.0 event per 220 live days at LZ's cuts in the LZ window, scaled by M/4.71 t and A(edge)/A(600). Note: LZ's L10 best fit spreads 1.0 event over the whole 5.4–270 keV ROI, of which only 0.364 lies above 200 keV, so for L10 our anchor is 2.7× LZ's best-fit *rate*; all gain factors, optimum positions and the 3D/hard comparisons are ratios and change little, but absolute Z, P(5σ) and 5σ live times for L10 should be read with that in mind (the "lower 68 % rate" rows, × 0.30, bracket this).

### 2.7 Metrics
- Asimov discovery Z = √(2[(s+b) ln(1+s/b) − s]) (Cowan et al. 2011) in the untouched exposure: 884 calendar days × 220/371 = **524 live days** (P020), i.e. 6.76 t·yr at 4.71 t.
- Exact-Poisson few-event metrics: N_5σ(b) = smallest N with P(≥N | b) ≤ 2.87 × 10⁻⁷; P(5σ) = P(N ≥ N_5σ | s + b); median Z of Z_N = Φ⁻¹(1 − P(≥N | b)). N_5σ = 2 needs b ≤ 7.6 × 10⁻⁴, N_5σ = 3 needs b ≤ 0.012, N_5σ = 4 needs b ≤ 0.050 (per 524 live days).
- Live time to Z = 5 at the best-fit rate and at 0.30 × it; zero-background exclusion reach (M·A)/(M·A)_LZ; the Gaussian figure of merit M/√b (reported, inapplicable when b ≲ 0.01).
- Soft FV: the FV is binned in (distance to the true wall, z) cells (0.5 × 2 cm); s_i ∝ volume, b_i = wall (analytic exponential integral per cell) + RFR (e^{−z/λ_z}) + uniform; Z_3D² = Σ_i 2[(s_i+b_i) ln(1+s_i/b_i) − s_i] versus Z_hard of the summed cell. The equivalent hard-cut mass M_eq solves Z_hard(M_eq, b ∝ M_eq, LZ-shaped FV, 1000 phd) = Z_3D.

## 3. Results

### 3.1 Reference and the stand-off scan (family A)
Untouched exposure, LZ cuts, 600 phd: M = 4.71 t, b = 8.9 × 10⁻⁴ (wall 4.0 × 10⁻⁴, RFR 8 × 10⁻⁶, uniform 4.8 × 10⁻⁴), s = 2.38 → **Z = 5.73σ** (P020: 5.7σ for two events).

Z_L10(s) at z_b = 9: 600 phd: 5.92/5.92/5.89/5.82/**5.73**/5.62/5.48/5.16 for s = 0/2/4/6/8/10/12/16 cm — peak s = 1 cm (5.92), within 1 % for s ≤ 4.5 cm, LZ's 8 cm is 3 % below the peak. 1000 phd: 6.71/6.84/6.92/6.95/**6.95**/6.90/6.81/6.52 — peak s = 6.5 cm (6.955), within 1 % for s = 3.5–10 cm, LZ's 8 cm is 0.1 % below the peak. 800 phd: peak s = 6 (6.34), within 1 % for 2–9.5 cm, Z(8) = 6.32. Background at 1000 phd, z_b = 9: b = 0.044/0.028/0.018/0.012/0.0080/0.0041 for s = 0/2/4/6/8/12 cm. The reason the optimum moves outward with the edge: in this regime Z ≈ √(2s[ln(s/b) − 1]), so d ln Z/d ln M ≈ 0.55 and d ln Z/d ln b ≈ −0.09; a mass gain g breaks even against a background factor (1+g)⁶, and the MSSI that the edge extension adds is itself edge-concentrated.

z_b at s = 8, 1000 phd: Z = 5.90/6.31/6.72/**6.95**/6.90 for z_b = 0/2/5/9/15 cm — lowering the bottom cut *costs* sensitivity at 1000 phd (RFR MSSI added by the extension: 1.6 × 10⁻³ at z_b = 9 → 7.2 × 10⁻³ (λ_z 2.7) or 1.2 × 10⁻² (2.0) at z_b = 5, 2.2–5.4 × 10⁻² at z_b = 2, per 524 d, for + 2.8–4.7 % mass). At 600 phd the RFR term is 3.5 × 10⁻⁶ and z_b can go to 1–5 cm for free (optimum z_b = 1 at 600 phd).

M/√b (600 phd): 101/116/132/146/158/167/172/173 for s = 0–16 cm, maximum at s = 14.5, z_b = 8 (M = 3.80 t): the Gaussian figure of merit would shrink the FV, but it presumes b ≫ 1 and is meaningless at b ~ 10⁻³.

### 3.2 Counterfactual: what the 14.5 % shrink bought
Had LZ kept the 5.4 t contour (calibrated 5.30 t, d_reco,min 4.06 cm): with the profile model b = 1.36 × 10⁻³ → Z = 5.94 (vs 5.73); with LZ's table numbers used directly (wall 0.0048 + 0.03, RFR 0.0001 + 0.003, × f_nb) b = 3.7 × 10⁻³ → Z = 5.48 (ΔZ = −0.26). Exclusion reach × 1.12. For the high-energy window the shrink was neutral for discovery (|ΔZ| ≤ 0.26) and cost 12 % of exclusion reach; its actual motivation — the whole-ROI wall leakage 10⁻² → 10⁻⁴ with λ_leak = 0.5 cm — is a low-S2 (low-energy) consideration. Answer to (a): the 8 cm stand-off is a low-energy legacy, but for the extended window it happens to sit at the Asimov optimum (Fig. 1).

### 3.3 Joint optimum (edge, s, z_b), d_reco,min ≥ 3 cm
| model | optimum | Z (LZ cuts 5.73) | gain | worst-case Z (k = 2, λ = 1.23, λ_z = 2.0, uniform × 2) |
|---|---|---|---|---|
| L10 | 1200 phd, s = 9, z_b = 14 | 7.54 | × 1.32 | 7.44 (b 0.010 → 0.012) |
| δ = 350 | 1000 phd, s = 6.5–7, z_b = 10 | 9.14 | × 1.59 | 7.9–8.2 (b 0.010 → 0.03–0.04) |
| δ = 366 | 1000 phd, s = 5, z_b = 9 | 14.16 | × 2.47 | 11.25 (b 0.014 → 0.14) |
| δ = 366, P(≥1 bkg) ≤ 0.01 | 1000 phd, s = 6.5, z_b = 11 | 14.10 | × 2.46 | 12.42 |

The gain is entirely the edge; the FV terms move Z by ≤ 1 %. 1200 phd (E50 = 495 keV) helps only the L10 tail and pushes b to 0.019 (N_5σ = 4); P038 notes the S2c ceiling bites at 503 keV.

### 3.4 Recommendation table (untouched 524 live days; nominal [worst case])
| configuration | M (t) | d_reco,min | b (524 d) | Z_L10 Asimov | Z_3D | Z_366 | 5σ live d (L10/350/366; L10 × 0.30) | reach L10/366 | N_5σ | P(5σ, L10 best) | P(5σ, 366; × 0.30) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LZ 2026 (600, 8, 9) | 4.71 | 6.2 | 8.9e-4 [1.8e-3] | 5.73 [5.44] | 5.77 | 5.73 | 399/399/399; 1608 | 1.00/1.00 | 3 [3] | 0.43 [0.43] | 0.43; 0.04 |
| 800, 8, 9 | 4.71 | 6.2 | 3.3e-3 [6.3e-3] | 6.32 [5.96] | 6.60 | 11.0 | 328/218/108; 1368 | 1.41/3.69 | 3 [3] | 0.65 [0.65] | 0.99; 0.49 |
| **1000, 8, 9** | 4.71 | 6.2 | 8.0e-3 [1.6e-2] | 6.95 [6.50] | 7.47 | 14.1 | 272/158/66; 1162 | 1.89/6.32 | 3 [4] | 0.83 [0.66] | 1.00; 0.83 |
| 1200, 8, 9 | 4.71 | 6.2 | 1.9e-2 [3.7e-2] | 7.40 [6.87] | 8.28 | 13.6 | 239/173/71; 1056 | 2.42/6.70 | 4 [4] | 0.83 [0.83] | 1.00; 0.71 |
| **1000, 7, 9** | 4.86 | 5.2 | 9.6e-3 [3.0e-2] | 6.95 [6.16] | 7.54 | 14.1 | 271/157/66; 1168 | 1.95/6.53 | 3 [4] | 0.84 [0.69] | 1.00; 0.85 |
| **1000, 6, 9** | 5.02 | 4.2 | 1.2e-2 [6.4e-2] | 6.95 [5.71] | 7.61 | 14.1 | 271/157/66; 1179 | 2.02/6.74 | 3 [5] | 0.86 [0.54] | 1.00; 0.86 |
| 1000, 6, 5 | 5.16 | 4.2 | 1.7e-2 [8.4e-2] | 6.79 [5.58] | 7.65 | 13.9 | 284/163/68; 1261 | 2.07/6.92 | 4 [5] | 0.73 [0.56] | 1.00; 0.73 |
| 1000, 5, 5 | 5.32 | 3.2 | 2.0e-2 [1.6e-1] | 6.81 [5.14] | 7.71 | 14.0 | 283/162/67; 1264 | 2.14/7.14 | 4 [6] | 0.75 [0.43] | 1.00; 0.75 |
| 1000, reco-margin 3 cm, 5 | 5.55 | 3.0 | 2.5e-2 [2.9e-1] | 6.82 [4.75] | 7.80 | 14.1 | 282/161/66; 1275 | 2.23/7.45 | 4 [7] | 0.78 [0.33] | 1.00; 0.78 |
| 1000, cylinder 6, 5 | 5.66 | 0.6 (infeasible) | 2.8e-2 [4.3e-1] | 6.81 [4.45] | 7.83 | 14.1 | 282/161/66 | 2.27/7.59 | 4 [8] | 0.79 [0.23] | — |
| 600, 6, 5 | 5.16 | 4.2 | 1.2e-3 [5.4e-3] | 5.90 [5.21] | 5.99 | 5.90 | 377; 1531 | 1.09 | 3 [3] | 0.48 [0.49] | 0.48; 0.05 |
| 600, reco-margin 3 cm, 5 | 5.55 | 3.0 | 1.7e-3 [2.0e-2] | 5.99 [4.74] | 6.15 | 5.99 | 365; 1497 | 1.18 | 3 [4] | 0.53 [0.31] | 0.53; 0.05 |

Exact-Poisson optimum of P(5σ | L10) per edge (d_reco,min ≥ 3): 600 phd s = 9.5, z_b = 7 (0.67 vs 0.43 at LZ's cuts — LZ's b = 8.9 × 10⁻⁴ narrowly misses the two-event threshold 7.6 × 10⁻⁴, a threshold effect that depends on the uniform-background bracket and is not a recommendation); 800 phd s = 5, z_b = 1 (0.75 vs 0.65); 1000 phd s = 6, z_b = 9 (0.86 vs 0.83); 1200 phd s = 8.5, z_b = 13 (0.91 vs 0.83).

5σ exposure in t·yr at the best-fit rate: LZ cuts 399 d × 4.71 t = 5.15 t·yr (L10 = 350 = 366, by construction of the anchor); 1000 phd, 8, 9: 3.51 (L10), 2.04 (350), 0.85 t·yr (366); 1000, 7, 9: 3.61/2.09/0.88 t·yr in 271/157/66 live days. The untouched 524 live days exceed the 5σ live time in every configuration if the best-fit rate holds; at 0.30 × the rate, 1162–1608 live days are needed (2029–2030).

### 3.5 Soft FV (position-binned likelihood)
- 600 phd, LZ volume: Z_3D = 5.77 vs hard 5.69 (+0.08σ) — consistent with P070's +0.1–0.2σ for the actual event (uniform backgrounds are 54 % of b at 600 phd).
- 1000 phd, LZ volume: **7.47 vs 6.90** (+8 %); MSSI is 89 % of b there and is edge-concentrated, so position information is worth an equivalent hard-cut mass of **5.45 t (+16 %)**.
- 1000 phd, s = 6, z_b = 9: 7.61 (M_eq 5.65 t, +20 %); s = 6, z_b = 5: 7.65 (5.71 t, +21 %); s = 5, z_b = 5: 7.71 (5.81 t); s = 3, z_b = 5: 7.83 (5.98 t, +27 %); s = 3, z_b = 2: 7.88 (6.06 t). Beyond s ≈ 6 the 3D likelihood adds only 0.05σ per cm: the added shell carries little signal weight once its LR is small.
- Model dependence: with the LZ-table profile (λ = 1.23 cm) the hard cut at s = 6, z_b = 5 falls to 6.21 (k = 1) or 5.68 (k = 2) while Z_3D stays 7.93/7.80 (M_eq 6.14/5.93 t): the position likelihood confines a steeper wall profile to the outermost cells. Position LR (uniform/wall, λ = 4.3, volume s = 3, z_b = 5) at d = 3/5/8/12/20/27 cm: 0.06/0.10/0.21/0.52/3.4/17 — an event 3–5 cm from the wall enters with weight ≤ 0.1 whatever the true profile, which is the robustness argument for a soft FV over a hard cut at the same radius.
- Caveat: the binned Asimov Z with cells of b_i ≪ s_i ≪ 1 overstates the finite-sample gain (P050: exact exposures ≲ 15 % larger), and an analysis that assumes λ = 4.3 when the truth is 1.2 cm mis-weights edge events between LR 0.06 and 0.006 — harmless for discovery (both tiny) but relevant for the background estimate of near-edge candidates.

### 3.6 Model dependence at small stand-off (systematic cost)
Wall MSSI (NR band, 1000 phd, 524 d, z_b = 5, k = 1) for λ = 1.23 / 3.3 / 4.3 / 6 cm:

| s (cm) | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| λ = 1.23 | 0.82 | 0.36 | 0.155 | 0.068 | 0.029 | 0.013 | 0.0056 |
| λ = 3.3 | 0.038 | 0.028 | 0.020 | 0.015 | 0.011 | 0.0077 | 0.0056 |
| λ = 4.3 | 0.025 | 0.020 | 0.015 | 0.012 | 0.0093 | 0.0072 | 0.0056 |
| λ = 6 | 0.017 | 0.014 | 0.012 | 0.0098 | 0.0082 | 0.0068 | 0.0056 |
| spread | × 48 | × 25 | × 13 | × 6.9 | × 3.6 | × 1.9 | × 1.0 |

With k = 2 on top, s = 5 can hold 0.14 and s = 6 0.06 wall-MSSI events — a P(≥1) of 6–13 % that would spoil a few-event claim (N_5σ rises to 5–6). At s = 7 the worst case is 0.03 (N_5σ = 4). RFR MSSI vs z_b (s = 6, 1000 phd, 524 d; λ_z = 2.0/2.7/6.5): z_b = 2: 0.054/0.022/0.0047; 5: 0.012/0.0072/0.0030; 7: 0.0044/0.0034/0.0022; 9: 0.0016.

### 3.7 Robust recommendation
1. **S1c edge 1000 phd** (E50 = 423 keV) with the S2c ceiling tracking the NR band (P038): × 1.9 (L10), × 3.0 (δ = 350), × 6.3 (δ = 366) signal for b = 0.008 per 524 d; P(5σ | best fit) 0.43 → 0.83 (L10), 0.43 → 1.00 (δ = 366; 0.83 at 0.30 × the rate).
2. **Hard cut s = 7 cm from the true wall, z_b = 9 cm** (M = 4.86 t, +3 %): Z unchanged (6.95), b = 0.0096 (worst 0.030), N_5σ = 3 (worst 4), d_reco,min = 5.2 cm. Going to s = 6 cm (5.02 t) is justified **only** with a position-dependent likelihood (Z_3D 7.61, equivalent 5.65 t) or after LZ validates the wall-MSSI radial profile with the HE-SB wall sample (its own table implies λ ≈ 1.2–1.4 cm, P033's transport 4.3 cm); s ≤ 5 cm is not recommended (worst-case b ≥ 0.15).
3. **Do not lower z_b at 1000 phd** (each cm below 9 cm adds more RFR MSSI than mass is worth; z_b = 5 costs 2.5 % in Z and doubles b); at ≤ 800 phd, z_b = 2–5 cm is free.
4. Design criterion: keep b(524 d) ≤ 0.012 so that three events suffice for 5σ; the wall stand-off is the only knob whose systematic is unbounded from LZ's data (100 % normalisation × a factor-7 profile disagreement), so spend the mass budget on the edge, not the wall.

## 4. Validation and cross-checks
- Mass calibration: one factor (1.104) reproduces 4.71 t (4.7 t FV) and 5.42 t (5.4 t volume) and gives 6.93 t for the active volume (LZ 7.0 t).
- Stand-off statistics reproduce the paper's mean (10.9 vs 10.7 cm) and P070's reconstructed mean (7.30 vs 7.26).
- Reference Z = 5.73 vs P020's 5.7σ for two events at b = 10⁻³ (with s = 2.38 the Asimov and the two-event Poisson agree to 0.1σ).
- Acceptance ratios vs P038/P050: 1000/600 for δ = 350: 3.0 (window-only) vs P038's 0.91/0.44 = 2.1 (full ROI incl. the constant low-energy part); consistent after removing the < 200 keV part.
- Added MSSI at 1000 phd: 2.98 × 10⁻³ vs P038's 0.0027 + 0.00017 NR-band.
- 5.4 t annulus check exposes the λ discrepancy (§2.2) — reported, not hidden.

## 5. Failed or abandoned approaches
- Treating the wall leakage as an exponential background term with the empirical λ_leak = 0.5 cm (from the 10⁻⁴/10⁻² criteria): it dominated every configuration inside s = 7 cm (b = 2 × 10³ at s = 0) and is inappropriate for high-S2 events; replaced by the d_reco ≥ 3 cm floor.
- A uniform-λ single scenario: the LZ-table-implied λ = 1.23 cm differs from P033's 4.3 cm by × 7 in the inward extrapolation, so we report both and make the recommendation robust to the steeper one.
- Family B (cylinder) as a serious candidate: infeasible at the bottom (contour outside the reconstructed wall).

## 6. Figures
- `figures/P079_fig1_Z_map.png` — Left: Asimov Z (L10 best fit, 1000 phd, 524 live days) over (s, z_b), LZ's cuts (star), the robust hard cut s = 7 (circle), s = 6 with the position likelihood (square); grey: d_reco,min < 3 cm. Right: Z versus s at z_b = 9 (600, 1000 phd) and z_b = 5 (1000 phd).
- `figures/P079_fig2_bkg_mass_vs_s.png` — Left: NR-band background components versus s at 1000 phd, z_b = 5: wall MSSI with the λ = 3–6 cm band and the LZ-table λ = 1.2 cm line, RFR, uniform, total. Right: fiducial mass (family A and cylinder B) and the Gaussian M/√b relative to s = 8.

## 7. References
LZ Collaboration, arXiv:2609.02823 (2026). G. Cowan, K. Cranmer, E. Gross, O. Vitells, Eur. Phys. J. C 71, 1554 (2011). I. Jeong, S. Kang, S. Scopel, G. Tomar, Comput. Phys. Commun. 276, 108342 (2022) (WimPyDD, via P050's cache). D. Baxter et al., Eur. Phys. J. C 81, 907 (2021). LZ Collaboration, Phys. Rev. Lett. 135, 011802 (2025) [2024 WS result; z boundaries of the 5.5 t FV]. Corpus: P004, P016, P019, P020, P022, P033, P038, P041, P050, P070.

## 8. Tools and provenance (mirrors provenance/P079.json)
- Agent tools: Read ×18 (PAPER_GUIDE; dossier; ledger (partial + grep); P038/P033/P070/P004/P050/P020/P022 papers; tex l. 112–141, 184–197, 300–320, 765–784; lzcommon l. 1–125; figures ×3), Bash ×21 (ledger listing, tex grep, file listings, P070/P033/P038/P004/P050 work-file inspection, four script runs, word counts), Write ×5 (script, details, provenance, paper ×2), Edit ×19 (script 14, paper 4, provenance 1).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (special.erf, optimize.brentq, stats.poisson, stats.norm); pandas 3.0.5; matplotlib 3.11.2 (Agg); common/lzcommon.py (imported; constants only); WimPyDD 2.0.4 spectra reused from `output/work/P050/P050_spectra_cache.npz` (not re-run).
- Local inputs: fulltext.tex (Data Analysis/FV paragraph, MSSI paragraphs, Discussion sideband paragraph, MSSI table); P070 `fv_contours.csv`, `profiles_z.csv`, `profiles_d.csv`, `P070_results.json`, details §2; P033 `P033_summary.json`; P038 `P038_mssi_vs_edge.csv`, `P038_results.json` (edges); P050 `P050_spectra_cache.npz`, `P050_background_bins.csv`, `P050_normalisation.csv`; P004 f_nb; P022/P019/P016 background numbers (from papers/ledger); P041 σ_T; ENVIRONMENT_versions.txt.
- Recalled knowledge (4): TPC radius 72.8 cm (likely); gate–cathode 145.6 cm (likely; confirmed by P070's digitisation); LXe density 2.86 g/cm³ (likely); Asimov/Wilks formula and the 5σ one-sided p = 2.87 × 10⁻⁷ (certain).
- Datasets: none. Data requests: none. WimPyDD-generated files: none.
