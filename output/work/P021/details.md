# P021 — A full likelihood fit of the single LZ event in (m_χ, δ, σ): where does inelastic dark matter actually prefer to sit? — research record

Simulated date 2026-09-08. Author profile: phenomenologists doing global fits of direct-detection data. Category IDM (hep-ph).
Script: `output/code/P021_full_likelihood.py` (run from the root: `.venv/bin/python output/code/P021_full_likelihood.py`;
first run builds the WimPyDD kernels, 111 s for the isoscalar and 195 s for isovector + Higgsino spectra; with the caches the analysis takes 7–11 s).
All numbers below are in `output/work/P021/P021_results.json`, `P021_scan_baseline.csv`, `P021_scan_all.csv`, `P021_tableS7_comparison.csv`,
`P021_peaks.csv`, `P021_vs_LZ_intervals.csv`, `P021_higgsino_fixed_coupling.csv`, `P021_surface_1000GeV_O1s.npz`, and the run log `run_log.txt`.

## 1. Motivation and question

LZ (arXiv:2609.02823) tested inelastic O₁ and O₄ on a grid δ = 0–350 keV for m_χ = 400/1000/4000 GeV (Table S7). P002 showed that kinematics allow
δ up to 341/387/409 keV for a 248 keV recoil on 16 June and that the O₁ spectrum only becomes "typical" at 248 keV within ~15 keV of that edge, at 10³–10⁴ lower
rate. P007 found that a pure Higgsino (σ_n fixed by G_F) needs δ = 358–380 keV, beyond the grid. Nobody in the corpus has yet asked the likelihood where it
actually peaks in (m_χ, δ, σ) once the energy resolution, the efficiency roll-off, the high-energy NR-band background and the low-energy null are all in one fit.
We do that, compare the profile Z(δ) with Table S7, give best-fit couplings and 68/90 % regions, and test the Higgsino line.

## 2. Model

### 2.1 Signal in observed energy
For coupling strength κ ≡ (c₁ m_v²)² (LZ/Anand normalisation; WimPyDD c⁰ = 2/m_v² gives κ = 1, `lz.wd_c_from_anand`, P003 convention) the expected
observed-energy density in 2.84 t·yr is

  dN/dE_obs(κ) = κ · E · ∫ dE_t (dR/dE_t)_WimPyDD(E_t; m, δ) ε(E_t) G(E_obs − E_t; σ_E(E_t)),  σ_E(E) = σ₂₄₈ √(E/248 keV),

with E = 2.84 t·yr, ε(E) = 0.96 · ½[1 + erf((E − 5.4)/(√2·2.5))] · ½ erfc((E − 269.9)/(√2·11.5)) (plateau and 50 % points from the paper; σ_hi = 11.5 keV
reproduces the Fig. S2 inset, P007; P009 finds σ_E = 11.8 keV for the roll-off), σ₂₄₈ = 11 keV (P009; 8 and 15 keV variants). True energies on a 3 keV
grid (1.5–340 keV), observed energies on a 1 keV grid. Spectra: WimPyDD 2.0.4 shell-model O₁ isoscalar ('s'), isovector ('v'), and the P007 pure-Higgsino
Z-exchange couplings (c⁰_WD = −7.62×10⁻⁶, c¹_WD = 8.87×10⁻⁶ GeV⁻², 'hig', 1000 GeV only). Per-stream kernels `WD.diff_rate(..., sum_over_streams=False)`
on an explicit v_min grid 0–830 km/s (1661 points; the default grid truncates the June tail, P002) are contracted with three halo functions
(Baxter-2021 SHM): 'annual' = mean of 12 monthly days (baseline; LZ used a time-averaged SHM, P007), 'june' (day 167), 'sun' (Sun frame, no orbital
motion, WimPyDD default). Negative kernel values near the edge are clipped to zero. Grid: m = 400, 1000, 4000 GeV; δ = 100–290 keV in 10 keV steps and
300 keV to the June ceiling μv²_max/2 in 5 keV steps (29/40/47 points); 2589/3240/3562 kernel calls per (coupling, mass).

Kinematics used (keV; `lz.delta_max_kev`, `lz.mu_red`; v_max(June) = 809.1 km/s, Sun frame 794.6 km/s):

| m (GeV) | μv²_max/2 June | δ_max(248) June | δ_max(270) June | δ_max(248) Sun | ceiling Sun | E* = δμ/m_N at the ceiling |
|---|---|---|---|---|---|---|
| 400 | 341.3 | 341.1 | 341.2 | 329.1 | 329.1 | 261 |
| 1000 | 397.1 | 386.6 | 390.8 | 374.6 | 383.0 | 354 |
| 4000 | 432.5 | 409.4 | 415.6 | 397.4 | 417.1 | 420 |

At the ceiling the allowed window [E₋, E₊] collapses to E* = δ_ceiling · m_χ/(m_χ + m_N): for 400 GeV this is 261 keV (on the event), for 1000 GeV 354 keV
and for 4000 GeV 420 keV (far above the ROI). This single fact controls the shape of Z(δ) near the edge (Sec. 4.2).

### 2.2 Data and likelihood
Following P016 for the low-energy part:
* Fig. 5 top panel (S1c < 250 phd ≈ 5.4–125 keV NR): P016's digitised 0.5σ_NR bins from −8 to +2σ (20 bins; model b_i sum 121.1, observed n_i sum 121;
  read from `output/work/P016/P016_results.json`), signal fraction per bin g_i = Φ(hi) − Φ(lo) for a Gaussian NR band (Σg_i = 0.977), all background scaled by an
  ER-leakage nuisance θ with a Gaussian constraint σ_θ = 0.3;
* 125–200 keV bin: b_M = 0.02, n_M = 0 (P016);
* 200–290 keV: unbinned; background b_H = 5.7×10⁻⁴ events (P016's anchor that gives L₁₀ 3.4σ), flat in 200–270 keV (density b_H/70 keV = 8.14×10⁻⁶ keV⁻¹);
  one event at E_obs = 248 keV (variants 246 and 262 keV, P009).

  ln L(κ, θ) = Σ_i [n_i ln(θ b_i + κ S_L g_i) − (θ b_i + κ S_L g_i)] + [n_M ln(b_M + κ S_M) − (b_M + κ S_M)] + ln(b_H/70 + κ f₂₄₈) − (b_H + κ S_H) − ½((θ−1)/0.3)²,

where S_L, S_M, S_H are the unit-κ accepted signal counts in 5.4–125, 125–200, 200–290 keV (observed) and f₂₄₈ the unit-κ density at the event.
θ is profiled by a vectorised bisection on ∂lnL/∂θ = 0 (monotonic). κ is profiled on a grid of the total expected accepted signal μ = κ S_tot
(0 and 651 log-spaced points 10⁻⁴–10^2.5), so the fit is well conditioned even when S_tot → 0 near the ceiling.
q₀ = 2[ln L(κ̂) − ln L(0)], Z = √q₀ (asymptotic, one-sided); 68 % and 90 % intervals from Δ(−2 ln L) = 1.0 and 2.706 (two-sided, Feldman–Cousins-like
via the profile ratio; the lower end is 0 when q₀ < Δ). The 2-dof (δ, κ) regions use Δ = 2.30 and 4.61. Conversion: (c₁ˢ m_v²)² = σ_SI π m_v⁴/μ_N²
(LZ supplement "Interpretation of O₁ as a cross-section"), i.e. σ_n = κ μ_N²/(π m_v⁴) × 0.3894×10⁻²⁷ cm²; at 1000 GeV κ = 1 ↔ 2.96×10⁻³⁸ cm² (check: κ = 0.0777 ↔ 2.30×10⁻³⁹ cm², P007's σ_SI,eq).

Gaussian variant ('gauss3'): the 20 Fig. 5 bins are replaced by −½[κ S_L/(3/1.2816)]² (a 90 % upper limit of 3 signal events in 5.4–125 keV), the 125–200 keV bin is kept.
'obscut' variant: no high-energy roll-off in true energy; instead a sharp cut at E_obs = 272 keV (P009's MC E₅₀ = 271.7 keV), i.e. the S1c < 600 phd cut applied in observed space.
Exact single-event check (companion-free spectra, S_L < 10⁻³ S_tot): for one background event at energy E the profile statistic is
q₀(E) = 2[ln(f̃(E)/b(E)) − 1 + b(E)/f̃(E)] for f̃ > b (else 0), f̃ = d/S_tot; the toy-free p-value is p = Σ_E b(E)·1[q₀(E) ≥ q₀_obs] dE (+O(b²)), evaluated with
b_H = 5.7×10⁻⁴ and with the exact-Poisson anchor b_H = Φ̄(3.4) = 3.37×10⁻⁴ (P016).

## 3. Inputs

| Input | Source |
|---|---|
| Table S7 local significances (O₁ˢ, O₁ᵛ, 400/1000/4000 GeV, δ = 0–350 keV; dash at (400, 350)) | fulltext.tex lines 880–909; `lz.OSIG` |
| LEE section: δ grid 0–350 keV, masses 400/1000/4000, 616 → 293 models, raster scan in coupling at each (m, δ) | lines 485–500 |
| Fig. 6 top caption (two-sided 90 % intervals, O₁ˢ 1000 GeV vs δ), Fig. S7 caption, O₁-as-σ recast formula, vector factor 3.2 | lines 284–294, 800–818 |
| Exposure 2.84 t·yr, efficiency plateau 0.96, 50 % at 5.4 and 269.9 keV, event 248 ± 23 ± 23 keV | lines 117, 455–457; `lz.LZ` |
| Digitised Fig. 6 intervals (c₁ˢ m_v²)² lower/median/upper at δ = 0–350 keV | `output/work/P007/lz_intervals_digitised.csv` |
| Higgsino WimPyDD couplings, (c_eq m_v²)² = 0.0777, σ_n = 7.40×10⁻³⁹ cm², vector factor 3.214 | `output/work/P007/higgsino_couplings.json`; P007 |
| Fig. 5 top-panel digitisation (bins, model, data), b_H anchor 5.70×10⁻⁴, b_M = 0.02, σ_θ = 0.3 | `output/work/P016/P016_results.json`; P016 |
| Efficiency edge σ = 11.5 keV, per-stream kernel method | `output/code/P007_higgsino_inelastic.py` |
| σ_E ≈ 10–12 keV at 248 keV; E_ML = 246 keV (paper scale) / 262 keV (Table S5 as printed); observed-space cut 271.7 keV | P009 |
| 248 keV at the 99.5th percentile (true energy) at δ = 300; δ_max values | P002 |
| N_eff = 9.3 (δ scan), 12.2 (LZ-like space), 14.5 (space + scan) | P008 |
| Baxter-2021 SHM, kinematics, WimPyDD wrapper with explicit v_min grid, coupling convention | `output/code/common/lzcommon.py` |

Recalled knowledge (flagged): (R1) (ħc)² = 0.3894×10⁻²⁷ GeV² cm² — certain; (R2) Wilks/Cowan asymptotics, Z = √q₀ for the one-sided discovery test — certain;
(R3) χ² quantiles 1.0/2.706 (1 dof, 68.3/90 %) and 2.30/4.61 (2 dof) — certain; (R4) the Feldman–Cousins-like profile interval for one event with negligible
background is μ ∈ [0.105, 3.65] at 90 % — likely (reproduced numerically: μ_90 = [0.097–0.105, 3.64–3.65] for δ ≥ 300 keV); (R5) at the kinematic ceiling the
allowed recoil window collapses to E* = δ μ/m_N — certain (derived from v_min(E_R, δ)); (R6) Okabe–Ito colour set is colour-vision-deficiency safe — certain (cosmetic).

## 4. Results

### 4.1 Calibration against Table S7 (baseline: annual halo, σ_E = 11 keV, E_obs = 248 keV, b_H = 5.7×10⁻⁴)

| δ (keV) | O₁ˢ 400 LZ / ours | O₁ˢ 1000 | O₁ˢ 4000 | O₁ᵛ 400 | O₁ᵛ 1000 | O₁ᵛ 4000 |
|---|---|---|---|---|---|---|
| 100 | 0.8 / 0.29 | 0.8 / 0.61 | 1.0 / 0.73 | 2.6 / 2.41 | 2.6 / 2.56 | 2.6 / 2.62 |
| 150 | 2.2 / 1.33 | 2.2 / 1.52 | 2.3 / 1.60 | 2.8 / 2.95 | 2.8 / 3.05 | 2.9 / 3.09 |
| 200 | 2.6 / 2.07 | 2.7 / 2.17 | 2.7 / 2.20 | 2.8 / 3.24 | 2.9 / 3.30 | 2.9 / 3.32 |
| 250 | 2.9 / 2.42 | 2.9 / 2.49 | 2.9 / 2.51 | 2.8 / 3.38 | 3.0 / 3.42 | 3.1 / 3.43 |
| 300 | 2.9 / 2.68 | 3.0 / 2.67 | 3.0 / 2.68 | 3.1 / 3.58 | 3.4 / 3.55 | 3.4 / 3.55 |
| 350 | – / – | 3.3 / 3.12 | 3.3 / 3.01 | – / – | 3.4 / 3.65 | 3.4 / 3.65 |

rms over the 34 defined entries 0.40σ, mean offset −0.09σ (O₁ˢ alone 0.47σ, always low by 0.18–0.87σ; O₁ᵛ 0.32σ, high by up to 0.58σ; δ ≥ 250 keV: 0.35σ).
The O₁ˢ deficit is largest at δ = 150–250 keV, where the fit places 1.2–1.9 signal events in the Fig. 5 bins (the data there show 42 observed vs 58.6
expected within ±1.5σ of the NR median, P016), so our low-energy penalty is stronger than LZ's 2D fit. The isovector–isoscalar gap is 1.95σ at δ = 100 keV
(LZ 1.8), 0.88σ at 300 keV (LZ 0.4), 0.53σ at 350 keV (LZ 0.1), 0.09σ at 380 keV: WimPyDD's isovector M response is more high-energy-weighted than LZ's
spectra imply, a systematic also seen by P008 (O₁ᵛ elastic 1.5 vs 1.3) and P016 (1.65 vs 1.3). Because the mean offset is only −0.09σ, P016's anchor
b_H is consistent on average; doubling/halving b_H shifts every Z by −0.20/+0.19σ at Z ≈ 3.5. Our reproduction of the (400 GeV, 350 keV) dash: δ = 350 keV is
above the 400 GeV ceiling (341.3 keV), so no spectrum exists there even with resolution (the dash is kinematic, as P002 said).

### 4.2 The shape of Z(δ) and where the maximum lies

1000 GeV, O₁ˢ, baseline (full table in `P021_scan_baseline.csv`):

| δ (keV) | Z | q₀ | κ̂ = (c₁ˢm_v²)² | 68 % κ | 90 % κ | σ̂_n (cm²) | N_unit = S_tot(κ=1) | f̃(248) (keV⁻¹) | percentile of 248 keV |
|---|---|---|---|---|---|---|---|---|---|
| 200 | 2.17 | 4.70 | 1.66e-6 | 4.6e-7–3.9e-6 | 1.2e-7–6.0e-6 | 4.9e-44 | 1.04e6 | 1.2e-4 | 99.87 |
| 250 | 2.49 | 6.19 | 1.08e-5 | 3.1e-6–2.6e-5 | 9.7e-7–4.0e-5 | 3.2e-43 | 1.04e5 | 4.2e-4 | 99.56 |
| 300 | 2.67 | 7.15 | 7.42e-5 | 2.2e-5–1.8e-4 | 7.2e-6–2.8e-4 | 2.2e-42 | 1.35e4 | 7.7e-4 | 99.17 |
| 330 | 2.89 | 8.35 | 5.0e-4 | 1.5e-4–1.2e-3 | 5.0e-5–1.8e-3 | 1.5e-41 | 1993 | 1.4e-3 | 98.4 |
| 350 | 3.12 | 9.74 | 3.87e-3 | 1.16e-3–9.1e-3 | 4.0e-4–1.41e-2 | 1.15e-40 | 258.7 | 2.9e-3 | 96.5 |
| 360 | 3.26 | 10.64 | 1.64e-2 | 4.9e-3–3.9e-2 | 1.7e-3–6.0e-2 | 4.9e-40 | 61.0 | 4.5e-3 | 94.0 |
| 370 | 3.42 | 11.71 | 0.116 | 0.035–0.273 | 0.012–0.42 | 3.4e-39 | 8.63 | 7.7e-3 | 87.4 |
| 375 | 3.50 | 12.24 | 0.413 | 0.124–0.973 | 0.043–1.51 | 1.2e-38 | 2.42 | 1.0e-2 | 79.5 |
| **380** | **3.59** | **12.86** | **2.19** | **0.66–5.17** | **0.23–8.0** | **6.5e-38** | **0.456** | **1.37e-2** | **56.9** |
| 385 | 3.47 | 12.07 | 12.5 | 3.8–29.5 | 1.3–45.6 | 3.7e-37 | 0.080 | 9.2e-3 | 11.6 |
| 390 | 3.16 | 10.0 | 43.1 | 12.9–101.5 | 4.4–157 | 1.3e-36 | 0.023 | 3.3e-3 | 1.6 |
| 395 | 0 | 0 | 0 (κ̂ = 0) | – | < 1.1e6 | – | 1.6e-6 | 2.8e-6 | 0.0007 |

The expected accepted signal at the best fit is μ̂ = κ̂ N_unit = 1.00 events for every δ ≥ 300 keV (1.0–2.0 at lower δ where companions enter), with μ ∈ [0.10, 3.65] at 90 %.
Mechanism. Since κ is free, the profile q₀ depends on the spectrum only through the normalised accepted density at the event, f̃(248) = f₂₄₈/S_tot:
q₀ = 2[ln(f̃/b_d) − 1 + b_d/f̃] with b_d = 8.14×10⁻⁶ keV⁻¹ (exact when S_L = S_M = 0; check: f̃ = 7.7×10⁻⁴ gives Z = 2.66). f̃(248) rises ×18 from δ = 300 to 380 keV because the
in-ROI spectrum narrows and moves up: 248 keV goes from the 99.2th observed percentile (P002: 99.5th in true energy) to the median at δ ≈ 381 keV
(P002: 385–390 keV in true energy). Above ≈ 385 keV the kinematic window [E₋, E₊] slides above the event (E* → 354 keV at the ceiling, Sec. 2.1), the density at
248 keV survives only through the 11 keV resolution tail, and at δ = 395 keV f̃ (2.8×10⁻⁶) falls below the background density, so κ̂ = 0 and Z = 0.
The collapse of Z within 15 keV of the ceiling is therefore a spectral-position effect, not a rate effect: the vanishing rate only drives κ̂ ∝ 1/N_unit up
(×10⁵ from δ = 300 to 385 keV; N_unit = 0.46 at 380 keV means the LZ unit coupling gives half an event).
Maxima (baseline): O₁ˢ: δ = 380 keV (1000 GeV; Z 3.59; 6.6 keV below δ_max(248) and 17 keV below the ceiling), 400 keV (4000 GeV; Z 3.60; κ̂ 4.66, σ̂_n 1.4×10⁻³⁷ cm²;
9 keV below δ_max(248)), and for 400 GeV Z rises monotonically to the ceiling at 340 keV (Z 3.39; κ̂ 2.41, σ̂_n 7.1×10⁻³⁸ cm²) because E* = 261 keV sits on the event.
O₁ᵛ: 380/380/395 keV with Z 3.68/3.68/3.67, but the curve is almost flat from 300 keV (3.55) on. Δq₀ (1000 GeV O₁ˢ) between the peak and LZ's grid points:
5.7 vs δ = 300 keV (likelihood ratio 17) and 3.1 vs 350 keV (ratio 4.7); for O₁ᵛ 0.9 and 0.2 (ratios 1.6, 1.1).
2-dof regions in (δ, κ) at 1000 GeV O₁ˢ (Fig. 2; global maximum δ = 380 keV, κ = 2.19, q₀ = 12.86 vs background): 68 %: δ = 360–385 keV, κ = 0.013–35;
90 %: δ = 330–390 keV, κ = 3.6×10⁻⁴–126. Below 330 keV every coupling is disfavoured at > 90 % relative to the maximum; δ = 300 keV is excluded at Δ = 5.7 (1-dof 2.4σ).
Local significance with P008's trials factors (p_local = 1.68×10⁻⁴ at the peak): N_eff = 9.3 (δ scan alone) → 2.96σ, 12.2 → 2.87σ, 14.5 (LZ space + scan) → 2.82σ.

### 4.3 Robustness (1000 GeV O₁ˢ; Z at δ = 300 / 350 / peak; peak δ)

| variant | Z(300) | Z(350) | Z_peak | δ_peak | κ̂(380) |
|---|---|---|---|---|---|
| baseline (annual, σ_E 11, E_obs 248, roll-off in true E, Fig. 5 bins, b_H 5.7e-4) | 2.67 | 3.12 | 3.59 | 380 | 2.19 |
| σ_E = 8 keV | 2.62 | 3.09 | 3.56 | 380 | 2.16 |
| σ_E = 15 keV | 2.75 | 3.17 | 3.60 | 380 | 2.24 |
| E_obs = 246 keV | 2.73 | 3.17 | 3.61 | 380 | 2.19 |
| E_obs = 262 keV (Table S5 scale) | 2.14 | 2.72 | 3.68 | 385 | 12.5 (at 385) |
| June-16 halo | 2.66 | 3.06 | 3.61 | 380 | 0.58 |
| Sun-frame halo | 2.68 | 3.21 | 3.49 | 370 | 20.2 |
| cut in observed space at 272 keV | 2.68 | 3.13 | 3.57 | 380 | 1.83 |
| Gaussian low-energy penalty (N_max = 3) | 2.68 | 3.12 | 3.59 | 380 | 2.19 |
| b_H × 2 | 2.41 | 2.89 | 3.39 | 380 | 2.19 |
| b_H / 2 | 2.92 | 3.34 | 3.77 | 380 | 2.19 |

Exact single-event p-values (companion-free δ ≥ 335 keV, 1000 GeV O₁ˢ): Z_exact = 3.36 (335–370 keV), 3.39 (375), 3.53 (380), 3.57 (385–390) with b_H = 5.7×10⁻⁴,
and 3.50–3.70 with the exact-Poisson anchor 3.37×10⁻⁴; asymptotic values 2.94–3.59. The ordering-based exact p is flatter in δ (it only knows how much of the
200–270 keV window is more signal-like than 248 keV, ≈ 47 keV for δ ≤ 370 and ≈ 22 keV for δ ≥ 385) but peaks in the same place (380–390 keV).
Peak location by mass and variant: 400 GeV O₁ˢ always at the ceiling (340; Sun frame 335); 4000 GeV at 400 (Sun 390, E_obs 262: 410); O₁ᵛ 380–400.

### 4.4 Comparison with LZ's Fig. 6 intervals (1000 GeV O₁ˢ; LZ values digitised by P007)

| δ (keV) | LZ lower / upper | ours (annual) 90 % lower / upper | ratio upper | ratio lower | ours (Sun frame) upper, ratio | events at LZ upper edge (annual / Sun) |
|---|---|---|---|---|---|---|
| 250 | 2.89e-6 / 3.87e-5 | 9.7e-7 / 3.96e-5 | 1.02 | 0.34 | 4.12e-5, 1.06 | 4.0 / 3.8 |
| 300 | 1.28e-5 / 2.60e-4 | 7.2e-6 / 2.75e-4 | 1.06 | 0.56 | 2.97e-4, 1.14 | 3.5 / 3.2 |
| 350 | 1.38e-3 / 2.25e-2 | 4.0e-4 / 1.41e-2 | 0.63 | 0.29 | 2.42e-2, 1.08 | 5.8 / 3.4 |

Upper edges agree to 2–6 % at 250–300 keV; at 350 keV the annual halo (which includes June-like days and hence more rate at large δ) gives an edge 37 % below LZ's, while the
Sun-frame halo reproduces LZ within 8 % at all three δ (P007 found the same: LZ's edges correspond to 3.2–3.8 events with the Sun-frame halo). LZ's lower edges are
1.8–3.5× higher than ours (0.17–0.36 events vs our 0.10; a single-event two-sided interval in a 2D PDF where the event sits 1.5σ below the NR median can be narrower).
The best-fit couplings κ̂ are 0.28–0.56 of LZ's median sensitivity (median sensitivity is not a best fit).

### 4.5 Higgsino (P007)
Fixed coupling (κ = 1 with the Higgsino Hamiltonian; N(δ) = 790, 11.0, 2.5, 1.09, 0.46, 0.20, 0.089, 0.037, 0.0078 at δ = 300, 350, 360, 365, 370, 375, 380, 385, 390 keV, annual halo;
P007: 790, 11.1, 2.6, 0.49, 0.098 at 300/350/360/370/380): best δ = 370 keV (Δ(−2lnL) ≤ 1: 365–375; ≤ 2.706: 360–380), δ(N = 1) = 365.5 keV (P007 366, 90 % 358–380);
June halo 375 (370–380 / 370–385), Sun frame 365 (360–370 / 355–375); σ_E = 8 or 15 keV: unchanged. q₀ against background at δ = 370 keV is 11.7 (3.42σ).
The isoscalar-equivalent Higgsino line κ_H = 0.0777 (σ_SI,eq = 2.3×10⁻³⁹ cm²; σ_n = 7.4×10⁻³⁹ cm² for the neutron) lies inside the O₁ˢ 68 % band for δ = 365–370 keV at
1000 GeV (90 %: 365–375), crossing κ̂(δ) at 368 keV where Z = 3.39; at 400 GeV the crossing is at 332 keV (band 330–335 keV, Z 3.14) and at 4000 GeV at 380 keV
(band 375–420 keV, Z 3.37). So the Higgsino sits 10–12 keV below the O₁ˢ likelihood maximum at 1000 GeV, inside the 68 % 2-dof region (360–385 keV).

### 4.6 Mass degeneracy and isovector
For δ ≤ 300 keV the 400 vs 4000 GeV curves differ by ≤ 0.44σ (O₁ˢ; ≤ 0.22σ O₁ᵛ), the difference being largest at δ = 100–150 keV (kinematic onset); at δ = 300 keV all
three masses give Z = 2.68 ± 0.01 (s) and 3.55–3.58 (v). Above 300 keV the masses separate through their different ceilings (Fig. 1): the peak moves 340 → 380 → 400 keV.
The isovector curve is flat (3.55 → 3.68) because its low-energy weight is already small at δ = 300 keV; O₁ᵛ − O₁ˢ shrinks from 1.95σ (100 keV) to 0.09σ (380 keV).

## 5. Discussion: the prior on σ

The profile likelihood is invariant under rescaling κ, so the preference for δ ≈ 380 keV is a statement about spectral shape (where 248 keV sits in the accepted
spectrum), not about rate. Two other readings are legitimate: (i) a Bayesian marginal likelihood with a log-uniform prior on κ (scale-free) gives the same
δ-dependence as the profile (∫ dlnκ L = ∫ dlnμ L); a prior uniform in κ over [0, κ_max] instead weights each δ by N_unit/κ_max and favours the low-δ, high-rate
end by factors of 10³–10⁵ — i.e. with such a prior the maximum returns to δ ≲ 300 keV; (ii) a physical model fixes κ (Higgsino: 0.0777) and then the likelihood over δ is
genuinely peaked (Sec. 4.5) with both flanks set by the rate. The "required σ explodes" picture (κ̂ ∝ 1/N_unit) is what a physical prior sees; the likelihood itself is
flat in that direction until the spectrum leaves the event. Whether σ̂_n ~ 10⁻³⁸–10⁻³⁷ cm² at δ = 380–390 keV is acceptable depends on the model (perturbative Z-exchange gives 7×10⁻³⁹).

Implications for LZ's grid: the O₁ˢ maximum at 1000 GeV is 30 keV beyond the last grid point and the likelihood ratio peak/350 keV is 4.7; for 4000 GeV the peak is
50 keV beyond. A finer scan to δ_max would raise the maximum local significance quoted for O₁ˢ from 3.3 to ≈ 3.6σ (and add trials: P008's continuous-scan N_eff 9.3 → global 2.8–3.0σ).

## 6. Failed or corrected approaches
* First implementation of the exact single-event p summed the background density wherever d(E) ≥ f₂₄₈ without requiring q₀(E) > 0; at the ceiling (q₀_obs = 0) this gave a
  finite p instead of p = 1. Replaced by the q₀(E)-ordering formula and p = 1 when q₀_obs = 0. It also only ran where S_M ≈ 0 (δ ≥ 385); relaxed to S_L < 10⁻³ S_tot (δ ≥ 335).
* The 'gauss3' variant initially dropped the 125–200 keV Poisson term, giving Z = 3.22 instead of 2.68 at δ = 300 keV; fixed (the term is kept; only the Fig. 5 bins are replaced).
* The δ at which 248 keV is the observed median was interpolated on a non-monotonic array (returned 395 keV); fixed by sorting (380.8 keV).
* The Higgsino-line crossing used log κ̂ including κ̂ = 0 at the ceiling; restricted to the rising branch up to the peak.
* A finer 2 keV true-energy grid was considered and rejected (kernel time ×1.5 for < 1 % change in the smeared densities).

## 7. Figures
* `figures/fig1_Z_vs_delta.png` — Z(δ) for m = 400 (blue), 1000 (vermilion), 4000 GeV (green); O₁ˢ solid, O₁ᵛ dashed; LZ Table S7 (filled = s, open = v); shaded band = envelope of the
  eight non-b_H variants for 1000 GeV O₁ˢ; dotted verticals δ_max(248 keV, June); grey = beyond LZ's grid.
* `figures/fig2_delta_kappa_1000GeV.png` — (δ, (c₁ˢm_v²)²) plane at 1000 GeV: 68 %/90 % 2-dof regions, κ̂(δ) with its 90 % band, LZ Fig. 6 intervals (P007 digitisation), the Higgsino line,
  δ_max(248) and the ceiling; right axis σ_n.
* `figures/fig3_mechanism_higgsino.png` — left: accepted observed-energy densities per event for δ = 300–395 keV with the event and the background density; right: −2ΔlnL(δ)
  for the fixed-coupling Higgsino for three halos with P007's window.

## 8. References
LZ Collaboration, arXiv:2609.02823 (2026). G. Cowan, K. Cranmer, E. Gross, O. Vitells, Eur. Phys. J. C 71, 1554 (2011). G. J. Feldman, R. D. Cousins, Phys. Rev. D 57, 3873 (1998).
D. Baxter et al., Eur. Phys. J. C 81, 907 (2021). I. Jeong, S. Kang, S. Scopel, G. Tomar, Comput. Phys. Commun. 276, 108342 (2022) (WimPyDD). N. Anand, A. L. Fitzpatrick, W. C. Haxton,
Phys. Rev. C 89, 065501 (2014). D. Tucker-Smith, N. Weiner, Phys. Rev. D 64, 043502 (2001). Corpus: P002, P003, P007, P008, P009, P016.

## 9. Tools and provenance (mirrors provenance/P021.json)
Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (stats.norm, special.erf/erfc); pandas 3.0.5; matplotlib 3.11.2 (Agg); WimPyDD 2.0.4 (eft_hamiltonian, streamed_halo_function via
lzcommon.wd_halo, diff_rate with sum_over_streams=False); common/lzcommon.py (LZ, OSIG, kinematics, wd_* wrappers, wd_c_from_anand). Script: output/code/P021_full_likelihood.py
(three invocations: `--spectra s --no-analysis`, `--spectra v hig`, then plain re-runs after fixes). Local inputs: fulltext.tex (Results/Fig. 6, LEE supplement, O₁-as-σ, Table S7),
P002/P003/P007/P008/P009/P016 papers, P007 CSV/JSON/script, P016 results JSON/script, lzcommon.py, dossier, ledger, PAPER_GUIDE, ENVIRONMENT_versions. Recalled: 6 items (Sec. 3).
Datasets: none. Data requests: none. WimPyDD-generated files: none outside output/work/P021 (per-stream kernels only; no response-function files written).
Agent tools: Read ×27, Bash ×15, Write ×6, Edit ×16 (13 script, 3 paper trims), Skill ×1 (dataviz).
