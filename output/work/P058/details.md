# P058 — If the excited state survives: exothermic down-scattering signals in LZ's data and the lifetime window they exclude

Research record (simulated date 2026-09-11; hep-ph; COMP/IDM). Script: `output/code/P058_exothermic.py` (helper
`output/code/P058_kernels_worker.py` computes the same WimPyDD kernels in parallel into the same cache). All tables below are
in `output/work/P058/*.csv`, `P058_results.json` and `run_log.txt`; figures in `output/work/P058/figures/`.

## 1. Motivation and question

Every inelastic-DM reading of the LZ 248 keV event (P002, P007, P011, P021) uses the *endothermic* transition χ₁N → χ₂N, which needs
v ≥ v_min and can only produce recoils below E ≈ 270 keV for δ close to the kinematic ceiling. P026 showed that in both LZ-motivated models
(Higgsino via Z exchange; pseudo-Dirac fermion with a kinetically mixed dark photon) the χ₂ ↔ χ₁ bath transitions decouple at T_* = 0.85–27 MeV and
leave f₂ = 0.42–0.50 of the dark matter in the excited state. If χ₂ has not decayed by today, the fraction f₂(today) = f₂,fo e^{−t_U/τ}
down-scatters *exothermically*, χ₂N → χ₁N, releasing δ into the recoil with no velocity threshold. P011 used this to require ≤ 1
exothermic event in LZ's ROI (f₂ < 1.2×10⁻³ at δ = 300 keV, ε ≳ 5×10⁻⁶); P026 turned that into τ < 7×10¹⁶ s and m_A′ ≥ 2.45/0.92/0.55/0.25 GeV
at δ = 300/350/365/380 keV.

We ask: (i) what does the exothermic spectrum look like — is it a narrow line near E* = δμ/m_N (≈ 260–340 keV, i.e. *above* LZ's 270 keV edge, in
the empty 600–1000 phd region of P038) or broad? (ii) Which LZ regions constrain it, and what are the resulting limits on f₂, τ(χ₂), the
dark-photon mixing ε and mass m_A′? (iii) Could the observed event itself be an exothermic down-scatter (input to P072)? (iv) What is the time
dependence of the exothermic rate?

## 2. Framework and equations

Two-body kinematics for a DM particle of mass m on a nucleus of mass m_N, reduced mass μ, with mass splitting δ (δ > 0 endothermic, δ < 0 exothermic):

  v_min(E_R) = |m_N E_R/μ + δ| / √(2 m_N E_R),                                                                      (1)
  E_±(v) = (μ²v²/m_N) [1 − δ/(μv²) ± √(1 − 2δ/(μv²))].                                                              (2)

For δ = −|δ| the discriminant exceeds unity for every v, so there is no velocity threshold, and at v → 0

  E_± → E* ± Δ(v),  E* = |δ| μ/m_N,  Δ(v) ≃ μ v √(2μ|δ|)/m_N  (leading order in v),                                  (3)

i.e. the recoil window is centred on E* but its half-width grows linearly with v. Because the halo speeds are ~250 km/s, Δ is *not* small:
Δ(250 km/s) ≈ 200 keV at 1 TeV (Table A1). The differential rate for the O₁ operator is

  dR/dE_R = (ρ₀/m) Σ_isotopes (N_T m_T/(2μ_T²)) |c|² F²_iso(E_R) η(v_min,iso(E_R)),  η(v_min) = ∫_{v>v_min} f(v)/v d³v,                       (4)

so the spectrum is F²(E) (the shell-model M-response, with nodes at ≈ 102 and 266 keV in xenon, P002/P003) times η evaluated at the
isotope-dependent v_min of Eq. (1), which vanishes at E*_iso. For E away from E*, v_min rises linearly and η falls; the F² fall-off
pulls the maximum of the product well below E*. The total exothermic cross-section at small v is σ ∝ √(2μ|δ|)/(μ v) ∝ 1/v, so the total
rate nv σ is velocity-independent to leading order: the annual modulation is expected to be small and to arise only through the
efficiency window and the F²-weighting of the changing width.

WimPyDD implements Eq. (1) with the absolute value (`WimPyDD/package.py`, `get_vmin`, line 5153: `abs(mn*er/mu+delta)`) and its
velocity-dependent operator pieces use (v² − δ/μ), which for δ < 0 correctly becomes v² + |δ|/μ; O₁ has no such piece. We verified
(run_log, Part B) that at E = 200 keV, 1 TeV, δ = −300 keV the per-stream kernel switches on at v = 100 km/s (our formula with the nine isotope
masses: 99.5–107.3 km/s) and shows seven distinct partial-sum levels (0.068, 0.168, 0.409, 0.638, 0.677, 0.981, 1.0 of the maximum) — the
isotope thresholds — so the kernel cannot be factorised as F²(E)·θ(v − v_min) for natural xenon; we therefore computed full (E, v) kernels.

`lzcommon.E_R_range_keV(m, v, delta_kev=-|δ|)` reproduces Eq. (2) for δ < 0 (columns `lz_E_R_range_*` in `P058_exo_kinematics.csv` are
identical to our own formula, e.g. 37.29 / 1916 keV at 1 TeV, 300 keV, 810 km/s; P011 quoted 37 keV and 1.9 MeV). **Pitfall:**
`lzcommon.vmin_kms` lacks the absolute value and returns *negative* values for E_R < E* when delta_kev < 0 (e.g. −26.3 km/s at 248 keV,
1 TeV, −300 keV); the physical v_min is |·|. Do not feed `vmin_kms(..., delta_kev<0)` to a halo function without `abs()`.

## 3. Inputs

| Input | Value | Source |
|---|---|---|
| LZ exposure, ROI, efficiency | 2.84 t·yr; 5.4–269.9 keV at 50 %; 0.96 plateau; erf edges σ = 2.5 / 11.5 keV | LZ paper (Fig. S2); P011/P021 convention |
| High-energy edges (600, 1000, 1700 phd) | E50 = 271.6, 423.2 keV, σ = 10.9, 14.5 keV; d lnS1c/d lnE = 1.154; plateau 0.955 | P038 (NEST-LZ rebuild); 1700 phd extrapolated: E50 = 423.2×1.7^{1/1.154} = 670 keV, σ ∝ E50 → 23 keV |
| Empty high-energy region | 4.7 t science sample, 600 < S1c < 1700 phd, S2c < 10^{4.3}: 0 events; MSSI 0.1 (wall) + 0.5 (RFR) in the HE SB + 0.006 in 600–800 phd | LZ Fig. S4 top-left panel and MSSI table (`tab:MSSI_comp`, lines 765–783 of the tex; Fig. 4 for 600–800 phd); P038 |
| Low-energy counts | 5.4–125 keV (S1c < 250 phd, ±1.5σ of the NR median): 42 observed vs 58.6 model; 125–200 keV: 0 vs 0.02; 200–270 keV: 1 vs 5.7×10⁻⁴ | P016 (digitised Fig. 5), P021 anchor |
| NR acceptance of the ±1.5σ band | 0.866 | recalled (Gaussian ±1.5σ), certain |
| Halo | Baxter-2021 SHM (v₀ = 238, v_esc = 544 km/s, ρ₀ = 0.3 GeV/cm³), Earth motion on 24 days of the year; annual mean = 24-day average; also 16 June (day 167), 16 Dec (day 350), Sun frame | `lz.wd_halo(day_of_year, vmin=linspace(0,844,1200))` |
| Couplings | isoscalar unit c_p = c_n = 1/m_v² (WimPyDD c⁰ = 2/m_v²), proton-only c_p = 1/m_v², Higgsino c_p = (G_F/√2)(1−4s_W²) = 6.20×10⁻⁷, c_n = −G_F/√2 = −8.25×10⁻⁶ GeV⁻² | P003 convention; P007/P011 |
| Freeze-out fractions | f₂,fo = 0.42 (Higgsino), 0.50 (dark photon) | P026 |
| χ₂ → χ₁νν̄ width (dark photon) | Γ = 3G_eff²δ⁵/(120π³), G_eff = g_D ε tanθ_W g/(4cosθ_W m_Z²) | P011 eq. (3) |
| σ_p for the dark photon | σ_p = c_p² μ_p²/π = 2.964×10⁻³⁸ cm² at unit coupling; σ_p = 16παα_Dε²μ_p²/m_A′⁴ | P011 |
| Constants | G_F = 1.1664×10⁻⁵ GeV⁻², s_W² = 0.2312, α = 1/137.036, m_Z = 91.19 GeV, ħ = 6.582×10⁻²⁵ GeV s, t_U = 13.8 Gyr = 4.35×10¹⁷ s, AMU = 0.9315 GeV | recalled, certain |
| LZ best-fit signal | μ̂ = 1.0 (+1.4, −0.7) ROI events | LZ paper (L₁₀ˢ fit), P021 (μ̂ = 1.0 for O₁ˢ) |

## 4. Computation

WimPyDD (`WD.diff_rate(WD.Xe, ham, m, E, VGRID, ONES, j_chi=0.5, delta=±δ, sum_over_streams=False)`) on a 1200-point stream grid
0–844 km/s gives the kernel K(E, v_i) in events/(t·yr·keV) per unit δη; the rate on any day is K·δη_day. Grids: exothermic E = 1–700 keV
(2.5 keV) and 700–2500 keV (5 keV); endothermic 1–450 keV (2.5 keV). Configurations: (isoscalar, proton-only, Higgsino) × 1 TeV × δ =
250, 278, 300, 350, 366, 380 keV (Higgsino: 300–380), isoscalar × (400, 4000 GeV) × δ = 300, 350, 380 keV; 21 configurations, each
with both signs of δ (≈ 17 000 diff_rate calls, 39 min; cached in `work/P058/cache/`). Counts: N_X = 2.84 t·yr × ∫ dR/dE ε_X(E) dE with
ε_ROI = 0.96 Φ((E−5.4)/2.5)[1−Φ((E−269.9)/11.5)], ε_HE = 0.955 Φ((E−271.6)/10.9)[1−Φ((E−670)/23)] (or the 1000 phd edge 423.2/14.5).
Regions: lo 5.4–55, mid 55–125, gap 125–200, hi 200–270 keV (ε_ROI), he1000/he1700 (ε_HE), roi (ε_ROI over all E), all (no efficiency).
Ratios R_X(δ) ≡ N_exo,X / N_endo,ROI are coupling-strength independent: exothermic events in region X per unit f₂ per endothermic ROI event.

Statistics. (a) Per-region 90 % CLs upper limits s₉₀ on extra events: lomid (n = 42, b = 58.6) 6.61; gap (0, 0.02) 2.30; hi with the one
endothermic event already counted (n = 1, b = 1 + 5.7×10⁻⁴) 3.27; he1700 (0, 0.606) 2.30. Then f₂ < s₉₀/(acc R_X) with μ_endo = 1.
(b) Joint Poisson likelihood over {lomid, gap, hi, he1700}, μ_X = b_X + acc_X κ (n_endo,X + f₂ n_exo,X) with κ = μ_endo/N_endo,ROI fixed by the
endothermic interpretation (μ_endo = 1; variants 0.3 and 2.4 from LZ's band); f₂ UL at Δ(−2lnL) = 2.706 above the minimum over f₂ ≥ 0.
(c) The same likelihood with κ profiled at each f₂ (κ free). (d) Four-bin local Z of one-parameter hypotheses vs background only, for a
pure exothermic origin (n_endo = 0) and a pure endothermic origin (n_exo = 0). (e) Lifetimes: τ_min = t_U / ln(f₂,fo/f₂,max);
dark photon: with ε²α_D = X m_A′⁴, X = σ_p(N=1)/(16παμ_p²), the lifetime is α_D-independent, τ(m_A′) = τ₀ (m_A′/GeV)⁻⁴, and the mass floor is
m_A′,min = (τ₀/τ_min)^{1/4}; ε_min(α_D) = √(τ(ε=1, α_D)/τ_min).

## 5. Results

### 5.1 Kinematics (`P058_exo_kinematics.csv`, `P058_delta_for_Estar248.csv`; A = 131.29)

| m [GeV] | δ [keV] | E* [keV] | Δ(250 km/s) [keV] | E₋(810) | E₊(810) [keV] |
|---|---|---|---|---|---|
| 400 | 300 / 350 / 380 | 229.7 / 268.0 / 291.0 | 159 / 171 / 178 | 35.9 / 46.7 / 53.7 | 1471 / 1537 / 1576 |
| 1000 | 250 / 300 / 350 / 366 / 380 | 222.7 / 267.3 / 311.8 / 326.1 / 338.6 | 186 / 202 / 216 / 221 / 224 | 27.0 / 37.3 / 48.8 / 52.7 / 56.2 | 1837 / 1916 / 1993 / 2018 / 2039 |
| 4000 | 300 / 350 / 380 | 291.1 / 339.6 / 368.7 | 230 / 247 / 256 | 38.1 / 49.9 / 57.6 | 2226 / 2312 / 2362 |

E* = 248 keV requires |δ| = 400 / 324 / 278 / 263 / 256 / 248 keV for m = 200 / 400 / 1000 / 2000 / 4000 / ∞ GeV, with Δ(250 km/s) =
132–222 keV. Exothermic v_min(248 keV) at 1 TeV: 26 / 87 / 107 / 124 km/s for δ = 300 / 350 / 366 / 380 keV (every halo particle contributes).
**The exothermic recoil window is ~400 keV wide at typical halo speeds; it is not a line at E*.**

### 5.2 Spectra (`P058_counts_per_unit_coupling.csv`, Fig. 1)

Isoscalar, 1 TeV, annual halo — peak / 16th / 50th / 84th percentiles of dR/dE (no efficiency) and fractions below 270, in 270–670, above 670 keV:

| δ [keV] | peak | p16 | p50 | p84 | < 270 | 270–670 | > 670 |
|---|---|---|---|---|---|---|---|
| 250 | 56 | 52 | 77 | 178 | 0.960 | 0.039 | 0.0007 |
| 278 | 61 | 61 | 142 | 195 | 0.934 | 0.065 | 0.0013 |
| 300 | 158 | 73 | 156 | 206 | 0.908 | 0.090 | 0.0020 |
| 350 | 164 | 138 | 173 | 244 | 0.845 | 0.151 | 0.0040 |
| 366 | 166 | 142 | 177 | 310 | 0.823 | 0.172 | 0.0048 |
| 380 | 166 | 145 | 180 | 327 | 0.802 | 0.193 | 0.0057 |

Proton-only: p50 = 164 / 183 / 190 keV at δ = 300 / 350 / 380, fractions below 270 keV 0.93 / 0.87 / 0.84 (node at ≈ 290 keV instead of 266).
Higgsino: p50 = 152 / 168 / 176 keV, 0.86 / 0.78 / 0.72 below 270 keV. 400 GeV: p50 = 150 / 168 / 175; 4000 GeV: 159 / 176 / 183 keV.
The spectrum peaks at 155–180 keV, far below E* = 267–339 keV, because the M-response falls steeply and has its node at 266 keV — exactly where
E* sits for δ = 300 keV at 1 TeV (visible as the dip through the E* marker in Fig. 1). 80–93 % of the exothermic recoils fall *inside* the WS ROI;
7–19 % (isoscalar) fall in the empty 272–670 keV region; < 1 % above. In observed energy (σ_E = 11√(E/248) keV) the ROI-accepted spectrum has
16/50/84 percentiles 70/150/190 (δ = 300), 132/165/200 (350), 139/170/203 keV (380): its width (±60 keV) is 5–6 times the resolution.

### 5.3 Counts per unit coupling and ratios (annual halo; `P058_counts_per_unit_coupling.csv`)

| coupling, m | δ | N_endo,ROI | N_exo,ROI | R_ROI | R_lo | R_gap | R_hi | R_he1000 | R_he1700 |
|---|---|---|---|---|---|---|---|---|---|
| iso 1 TeV | 300 | 1.36×10⁴ | 1.24×10⁷ | 916 | 35.9 | 544 | 92.9 | 76.5 | 89.5 |
| | 350 | 265 | 7.35×10⁶ | 2.78×10⁴ | 21.2 | 2.05×10⁴ | 4.22×10³ | 4.15×10³ | 4.92×10³ |
| | 366 | 21.7 | 6.35×10⁶ | 2.93×10⁵ | 10.8 | 2.21×10⁵ | 4.88×10⁴ | 5.12×10⁴ | 6.09×10⁴ |
| | 380 | 0.689 | 5.59×10⁶ | 8.11×10⁶ | 1.5 | 6.18×10⁶ | 1.45×10⁶ | 1.62×10⁶ | 1.93×10⁶ |
| p 1 TeV | 300 | 3.45×10³ | 2.96×10⁶ | 859 | 30 | 445 | 151 | 45.7 | 66.6 |
| | 350 | 99.2 | 1.80×10⁶ | 1.81×10⁴ | 12.3 | 1.16×10⁴ | 4.70×10³ | 1.70×10³ | 2.53×10³ |
| | 366 | 11.0 | 1.57×10⁶ | 1.43×10⁵ | 4.7 | 9.31×10⁴ | 4.03×10⁴ | 1.56×10⁴ | 2.33×10⁴ |
| | 380 | 0.531 | 1.40×10⁶ | 2.64×10⁶ | 0.45 | 1.73×10⁶ | 7.93×10⁵ | 3.26×10⁵ | 4.91×10⁵ |
| hig 1 TeV | 300 | 798 | 7.69×10⁵ | 963 | 38.8 | 614 | 50.5 | 133 | 146 |
| | 366 | 1.01 | 3.86×10⁵ | 3.82×10⁵ | 14.2 | 3.07×10⁵ | 3.34×10⁴ | 1.10×10⁵ | 1.24×10⁵ |
| iso 400 GeV | 300 | 4.01×10³ | 3.76×10⁷ | 9.36×10³ | 511 | 5.19×10³ | 818 | 557 | 635 |
| iso 4000 GeV | 300 / 350 / 380 | 6.77×10³ / 392 / 14.1 | 2.79×10⁶ / 1.64×10⁶ / 1.22×10⁶ | 413 / 4.17×10³ / 8.65×10⁴ | 13.5 / 1.6 / 0 | 252 / 3.09×10³ / 6.58×10⁴ | 44.8 / 666 / 1.62×10⁴ | 40 / 714 / 1.98×10⁴ | 47.4 / 855 / 2.39×10⁴ |

Cross-check with P011 (proton-only, 1 TeV, annual): N_exo,ROI/unit = 6.201×10⁶ / 2.960×10⁶ / 1.800×10⁶ / 1.403×10⁶ vs P011 6.20×10⁶ / 2.96×10⁶ /
1.80×10⁶ / 1.40×10⁶ at δ = 250 / 300 / 350 / 380; N_endo/unit 2.474×10⁴ / 3445 / 99.2 / 0.531 vs 2.473×10⁴ / 3449 / 99.3 / 0.530 (≤ 0.2 %; our
366 keV vs their 365 keV differ by 15 % in the endothermic count as expected). Halo dependence of R_ROI (iso, 1 TeV): June / annual / Sun / Dec =
659 / 916 / 984 / 1460 (δ = 300), 1.18×10⁴ / 2.78×10⁴ / 4.74×10⁴ / 2.75×10⁵ (350), 2.1×10⁶ / 8.1×10⁶ / 8.7×10⁷ / 3×10¹¹ (380): the
*exothermic* counts vary by only ±4 % over the year (N_exo,ROI = 1.20–1.28×10⁷), all the halo sensitivity sits in the endothermic denominator.
Expected exothermic counts in 2.84 t·yr for f₂ = 10⁻³–0.42 at μ_endo = 1 are tabulated in `P058_expected_counts_vs_f2.csv`; e.g. isoscalar δ = 300 keV,
f₂ = 0.42: 15 (5.4–55 keV), 102 (55–125), 228 (125–200), 39 (200–270), 32 (600–1000 phd), 38 (600–1700 phd), 0.9 (> 670 keV); δ = 366 keV, f₂ = 10⁻³:
0.01 / 23 / 221 / 49 / 51 / 61 / 1.9.

### 5.4 Limits on f₂ and τ (`P058_f2_limits.csv`, Fig. 2)

Per-region 90 % ULs (μ_endo = 1) and the joint fixed-κ limit, 1 TeV:

| coupling | δ | 5.4–125 keV | 125–200 keV | 200–270 keV | 600–1000 phd | 600–1700 phd | P011 criterion | **joint** | joint (μ = 0.3 / 2.4) | ROI-only joint |
|---|---|---|---|---|---|---|---|---|---|---|
| iso | 300 | 0.027 | 0.0042 | 0.035 | 0.030 | 0.026 | 0.0011 | **2.8×10⁻³** | 1.3×10⁻² / 9.0×10⁻⁴ | 3.3×10⁻³ |
| iso | 350 | 2.5×10⁻³ | 1.1×10⁻⁴ | 7.8×10⁻⁴ | 5.6×10⁻⁴ | 4.7×10⁻⁴ | 3.6×10⁻⁵ | **5.7×10⁻⁵** | 2.8×10⁻⁴ / 2.1×10⁻⁵ | 7.1×10⁻⁵ |
| iso | 366 | 3.4×10⁻⁴ | 1.0×10⁻⁵ | 6.7×10⁻⁵ | 4.5×10⁻⁵ | 3.8×10⁻⁵ | 3.4×10⁻⁶ | **4.8×10⁻⁶** | 2.2×10⁻⁵ / 1.8×10⁻⁶ | 6.1×10⁻⁶ |
| iso | 380 | 1.6×10⁻⁵ | 3.7×10⁻⁷ | 2.3×10⁻⁶ | 1.4×10⁻⁶ | 1.2×10⁻⁶ | 1.2×10⁻⁷ | **1.7×10⁻⁷** | 7.7×10⁻⁷ / 6.3×10⁻⁸ | 2.2×10⁻⁷ |
| p | 300 / 350 / 366 / 380 | 0.029 / 4.2×10⁻³ / 8.0×10⁻⁴ / 6.3×10⁻⁵ | 5.2×10⁻³ / 2.0×10⁻⁴ / 2.5×10⁻⁵ / 1.3×10⁻⁶ | 0.022 / 7.0×10⁻⁴ / 8.1×10⁻⁵ / 4.1×10⁻⁶ | 0.050 / 1.4×10⁻³ / 1.5×10⁻⁴ / 7.1×10⁻⁶ | 0.035 / 9.1×10⁻⁴ / 9.9×10⁻⁵ / 4.7×10⁻⁶ | 1.2×10⁻³ / 5.5×10⁻⁵ / 7.0×10⁻⁶ / 3.8×10⁻⁷ | **3.1×10⁻³ / 9.7×10⁻⁵ / 1.1×10⁻⁵ / 5.7×10⁻⁷** | — | — |
| hig | 300 / 350 / 366 / 380 | 0.026 / 1.5×10⁻³ / 2.0×10⁻⁴ / 2.8×10⁻⁵ | 3.8×10⁻³ / 7.4×10⁻⁵ / 7.5×10⁻⁶ / 8.7×10⁻⁷ | 0.065 / 1.1×10⁻³ / 9.8×10⁻⁵ / 1.1×10⁻⁵ | 0.017 / 2.4×10⁻⁴ / 2.1×10⁻⁵ / 2.1×10⁻⁶ | 0.016 / 2.1×10⁻⁴ / 1.9×10⁻⁵ / 1.9×10⁻⁶ | 1.0×10⁻³ / 2.5×10⁻⁵ / 2.6×10⁻⁶ / 3.1×10⁻⁷ | **2.4×10⁻³ / 3.5×10⁻⁵ / 3.2×10⁻⁶ / 3.6×10⁻⁷** | — | — |
| iso 400 GeV | 300 | 2.3×10⁻³ | 4.4×10⁻⁴ | 4.0×10⁻³ | 4.1×10⁻³ | 3.6×10⁻³ | 1.1×10⁻⁴ | **2.8×10⁻⁴** | | |
| iso 4000 GeV | 300 / 350 / 380 | 0.066 / 0.019 / 1.7×10⁻³ | 9.1×10⁻³ / 7.4×10⁻⁴ / 3.5×10⁻⁵ | 0.073 / 4.9×10⁻³ / 2.0×10⁻⁴ | 0.058 / 3.2×10⁻³ / 1.2×10⁻⁴ | 0.049 / 2.7×10⁻³ / 9.6×10⁻⁵ | 2.4×10⁻³ / 2.4×10⁻⁴ / 1.2×10⁻⁵ | **6.0×10⁻³ / 3.9×10⁻⁴ / 1.5×10⁻⁵** | | |

Findings. (i) The **empty 125–200 keV bin** (0 events, b = 0.02) gives the strongest single-region limit at every δ and coupling: 55–70 % of the
exothermic spectrum lands there. The empty 600–1700 phd region alone is 3–6 times weaker (0.026 vs 0.0042 at δ = 300 keV; ×4.2 at 350; ×3.6 at 366; ×3.2 at
380), and adding it to the ROI-only likelihood improves the joint limit by only 15–25 %. The assignment's expectation that exothermic events "pile up
near E*" above the ROI edge is not borne out: at the halo speeds the window is ~400 keV wide and the form factor moves the spectrum down to ~160 keV.
(ii) The joint limits are 1.5–3× *weaker* than P011's "≤ 1 ROI event" criterion, because LZ's data actually allow 6.6 extra NR-like events in 5.4–125 keV
(42 observed vs 58.6 modelled) and 2.3 in 125–200 keV; P011's criterion was a shortcut. (iii) The limit scales as 1/μ_endo (×4.7 / ×0.32 for LZ's 0.3–2.4 band).
(iv) With **κ free** (profiled) f₂ is unconstrained: the fit trades the endothermic event for an exothermic one, e.g. isoscalar δ = 300 keV gives
Δ(−2lnL) = +0.65 at f₂ = 1 with κ̂ = 9.2×10⁻⁸ (vs 6.4×10⁻⁵ at f₂ = 0); at δ = 380 keV the mixed hypothesis even fits *better* (Δ = −3.7), because the pure
endothermic δ = 380 keV spectrum puts 97 % of its events in the empty 272–670 keV region (P038). The f₂ exclusion therefore holds *conditionally* on the
endothermic interpretation of the event; the alternative — the event is itself an exothermic scatter of a surviving χ₂ at a ~10³ smaller coupling — is
examined in §5.6.

Lifetimes (τ_min = t_U/ln(f₂,fo/f₂,max); τ > τ_min excluded, i.e. the excluded window is [τ_min, ∞)):

| δ [keV] | iso (f₂,fo = 0.5) | p, dark photon (0.5) | p, μ_endo = 0.3 / 2.4 | Higgsino (0.42) | P011/P026 |
|---|---|---|---|---|---|
| 250 | 1.26×10¹⁷ | 1.28×10¹⁷ | 2.3×10¹⁷ / 9.6×10¹⁶ | — | 7.9×10¹⁶ |
| 300 | 8.4×10¹⁶ | 8.6×10¹⁶ | 1.23×10¹⁷ / 7.0×10¹⁶ | 8.2×10¹⁶ | 6.5×10¹⁶ |
| 350 | 4.8×10¹⁶ | 5.1×10¹⁶ | 6.3×10¹⁶ / 4.5×10¹⁶ | 4.6×10¹⁶ | 4.4×10¹⁶ |
| 366 | 3.8×10¹⁶ | 4.1×10¹⁶ | 4.8×10¹⁶ / 3.7×10¹⁶ | 3.6×10¹⁶ | 3.7×10¹⁶ (365) |
| 380 | 2.9×10¹⁶ | 3.2×10¹⁶ | 3.6×10¹⁶ / 3.0×10¹⁶ | 3.1×10¹⁶ | 2.9×10¹⁶ |

Because τ_min depends only logarithmically on f₂,max, the 1.5–3× weaker f₂ limits move τ_min by ≤ 30 %: the χ₂ must have decayed with τ ≲ (3–9)×10¹⁶ s ≈ 0.07–0.2 t_U.
For the Higgsino this is trivially satisfied (τ_γ ~ 0.1 s, τ_νν̄ ~ 10⁶ s; P014/P026) — no constraint on the Z-coupling structure follows. For a photon
channel P026's Planck window [2×10¹², 10¹⁹ s] and ours overlap, so with a γ channel every τ > 2×10¹² s is excluded.

Dark photon (`P058_darkphoton_floors.csv`; 1 TeV, proton-only, annual halo): σ_p(N = 1) = 1.20×10⁻⁴² / 3.47×10⁻⁴² / 8.60×10⁻⁴² / 2.99×10⁻⁴⁰ / 2.69×10⁻³⁹ /
5.59×10⁻³⁸ cm² at δ = 250 / 278 / 300 / 350 / 366 / 380 keV (P011: 1.20×10⁻⁴², 8.59×10⁻⁴², 2.99×10⁻⁴⁰, 2.29×10⁻³⁹ at 365, 5.60×10⁻³⁸). τ(m_A′ = 1 GeV) =
5.0×10¹⁹ / 1.0×10¹⁹ / 2.78×10¹⁸ / 3.7×10¹⁶ / 3.3×10¹⁵ / 1.3×10¹⁴ s (P026: 2.6×10¹⁸ s at 300 keV). **m_A′ ≥ 4.44 / 3.15 / 2.39 / 0.92 / 0.53 / 0.25 GeV**
(P026: 2.45 / 0.92 / 0.55 / 0.25 at 300 / 350 / 365 / 380) — the fourth-root dependence makes the floor insensitive to the statistical treatment.
ε ≥ 6.1 / 5.2 / 4.7 / 4.2 / 4.2 / 4.3 ×10⁻⁶ for α_D = 0.1 (P011: 7.8 / — / 5.4 / 4.5 / 4.4 / 4.5 ×10⁻⁶). The ε floor scales as α_D^{−1/2}; the m_A′ floor is α_D-independent.

### 5.5 Which structure is most constrained?

At fixed δ the isoscalar, proton-only and Higgsino R_ROI agree within ±10 % at δ = 300 keV (916 / 859 / 963) but differ up to ×3 at 366–380 keV
(2.9×10⁵ / 1.4×10⁵ / 3.8×10⁵), following the endothermic denominator (the isovector-dominated Higgsino form factor puts more endothermic rate below the edge).
The proton-only spectrum, whose node sits at ≈ 290 keV, puts the least (7–15 %) into the empty high region; the Higgsino the most (13–28 %), which is why the
600–1700 phd region is competitive with 125–200 keV only for the Higgsino (0.016 vs 0.0038 at 300 keV).

### 5.6 The event as an exothermic down-scatter (`P058_event_as_exothermic.csv`; 1 TeV)

| coupling | δ | E* | percentile of 248 keV (true / observed) | fraction of accepted ROI events in 200–270 | 125–200 keV per 200–270 event | 600–1700 phd per 200–270 event | Z₄bin exo-origin (μ̂_hi) | Z₄bin endo-origin (μ̂_hi) |
|---|---|---|---|---|---|---|---|---|
| iso | 278 | 248 | 0.998 / 0.997 | 0.077 | 6.3 | 0.89 | 2.90 (0.10) | 3.00 (0.14) |
| iso | 300 | 267 | 0.997 / 0.996 | 0.102 | 5.9 | 0.96 | 2.94 (0.12) | 3.05 (0.16) |
| iso | 350 | 312 | 0.996 / 0.993 | 0.152 | 4.9 | 1.16 | 3.00 (0.14) | 3.19 (0.25) |
| iso | 380 | 339 | 0.995 / 0.992 | 0.179 | 4.3 | 1.33 | 3.03 (0.15) | 2.34 (0.02) |
| p | 300 | 267 | 0.987 / 0.984 | 0.176 | 2.9 | 0.44 | 3.13 (0.21) | 3.23 (0.28) |
| p | 380 | 339 | 0.976 / 0.971 | 0.300 | 2.2 | 0.62 | 3.20 (0.26) | 2.97 (0.13) |
| hig | 366 | 326 | 0.986 / 0.987 | 0.088 | 9.2 | 3.7 | 2.77 (0.07) | 2.71 (0.06) |

Even at the splitting that puts E* exactly at 248 keV (δ = 278 keV), 248 keV lies at the **99.7th percentile** of the ROI-accepted exothermic spectrum in
observed energy (97–99 % for all couplings and δ): the exothermic hypothesis is *spectrally unattractive*, contrary to the naive E*-line picture, because the
F²-weighted spectrum peaks at 150–180 keV. A pure exothermic origin predicts 2.2–9 events in the empty 125–200 keV bin per event in 200–270 keV and 0.4–3.9
in 600–1700 phd; the four-bin fit copes by shrinking the signal to μ̂_hi = 0.07–0.26 events in 200–270 keV (P016's mechanism), reaching Z = 2.7–3.2 — comparable to
the endothermic O₁ at δ = 300–366 keV (3.0–3.2; P021: 2.7–3.4 with resolution) but explaining the observed event only at the 7–26 % level (P016's O₄ analogue:
5.5 %). Our four-bin endothermic Z at δ = 380 keV, 2.34, reproduces P038's extended-ROI value Z₁₀₀₀(380) = 2.33. The coupling needed for one exothermic
ROI event at f₂ = 0.5 is κ = 1.6×10⁻⁷ (iso) — 460× below the endothermic-fit κ = 7.4×10⁻⁵ at δ = 300 keV — or σ_p = 2.0 / 3.3 / 4.2 ×10⁻⁴⁴ cm² (p, δ = 300 / 350 / 380);
at that coupling the χ₁ population gives 10⁻³–10⁻⁷ endothermic events, so the event would carry no June preference (P006) and no δ_max relation to 248 keV;
it requires τ(χ₂) ≳ t_U, i.e. for the dark photon m_A′ ≲ 1.4 GeV at δ = 300 keV [(2.78×10¹⁸/4.35×10¹⁷)^{1/4}], and predicts a *time-flat*, broad,
mostly sub-200 keV population (§5.7). These are the inputs for P072.

### 5.7 Annual modulation (`P058_modulation.csv`, Fig. 3)

Cosine amplitude a₁ (phase fixed at 2 June) relative to the mean, 1 TeV isoscalar: exothermic ROI rate 5.3 / 4.2 / 3.2 / 2.3 / 2.5 / 2.8 % at δ = 250 / 278 / 300 /
350 / 366 / 380 keV (maximum near day 145, late May); total exothermic rate 5.0 / 3.8 / 2.7 / 1.6 / 1.6 / 1.7 %; rate in 272–670 keV −1.2 / −1.7 / −2.1 / −2.7 / −2.8 / −2.8 %
(maximum in December: faster DM in June narrows nothing but shifts weight toward the ROI through F²). Proton-only 3.0 / 1.8 / 2.1 %, Higgsino 3.2 / 2.8 / 3.5 % (300 / 350 / 380).
Endothermic ROI a₁ = 0.43 / 1.18 / 1.53 / 1.68 at δ = 300 / 350 / 366 / 380 keV (P034: 0.43 / 1.18 / 1.52 / 1.66; a₁ > 1 means non-sinusoidal). The exothermic
signal is 15–60 times less modulated than the endothermic one, as anticipated from nvσ ∝ v⁰; the residual 2–5 % (June phase) comes from the F²-weighting and
the ROI window.

## 6. Validation and robustness

- P011 exothermic and endothermic counts reproduced to ≤ 0.2 % (§5.3); P034 modulation amplitudes to ≤ 2 %; P038 Z(δ = 380) with the extended ROI to 0.01σ.
- `lz.E_R_range_keV` with negative δ agrees with Eq. (2); `lz.vmin_kms` sign pitfall documented (§2).
- WimPyDD negative-δ handling verified against the per-isotope thresholds (§2).
- Halo: exothermic counts change by ±4 % between June and December; the f₂ limits inherit the endothermic denominator's halo dependence (×0.7 June, ×1.6 December at δ = 300; ×0.42/×10 at 350 keV); we quote the 24-day annual mean (LZ used a time-averaged SHM, P007/P021).
- Efficiency above the ROI: the 1700 phd edge (670 keV) is an extrapolation of P038's NEST-LZ edges; using the 1000 phd edge (423 keV) changes the joint limit by ≤ 3 % because the high region is sub-dominant. The S2c < 10^{4.3} ceiling of the HE SB removes < 5 % of the NR band below 670 keV (P038: the 10^{4.15} ceiling bites 10 % at 503 keV).
- The 5.4–125 keV bin uses P016's digitised counts within ±1.5σ of the NR median; scaling its background ×0.5–2 changes the joint limit by < 5 % (the bin is sub-dominant).
- Resolution smearing is applied only to the percentile statements; for the broad exothermic spectrum it changes region counts by < 3 % (edge leakage), well below the halo/efficiency systematics.

## 7. Failed or abandoned approaches

- Factorising the WimPyDD kernel as F²(E)·θ(v − v_min(E)) to avoid recomputing kernels per δ: invalid for natural xenon because each isotope has its own threshold (seven partial-sum levels); we computed full (E, v) kernels instead (39 min, 21 configurations).
- A κ-profiled joint likelihood as the headline f₂ limit: it returns no limit (the exothermic population replaces the endothermic event); reported as the κ–f₂ degeneracy instead.
- A first attempt to read the xenon isotope masses via `WD.Xe.element[0].mass` returned a scalar; `WD.Xe.mass` holds the nine isotope masses.

## 8. Discussion

The exothermic channel is the one *guaranteed* companion signal of every pseudo-Dirac interpretation of the LZ event: it needs no velocity tail, it is
time-flat to 2–5 %, and at the endothermic-fit coupling it is 10³–10⁷ times larger than the endothermic rate. LZ's own data — chiefly the empty 125–200 keV bin,
with the empty 600–1700 phd region contributing a 15–25 % improvement — force f₂(today) below 2.8×10⁻³ (δ = 300 keV) to 1.7×10⁻⁷ (380 keV), i.e. τ(χ₂) < (3–9)×10¹⁶ s.
This sharpens P011/P026 in the statistical treatment but leaves their conclusions intact: the Higgsino is safe, and a dark photon must be heavier than 2.4 GeV
(δ = 300 keV) or 0.25 GeV (380 keV), independent of α_D. The reverse reading — LZ saw the down-scatter of a *stable* χ₂ — cannot be excluded by rate (the coupling is
free) but is disfavoured by shape: 248 keV sits at the 97–99.7th percentile of the exothermic spectrum, and the hypothesis predicts 2–9 companions in 125–200 keV per
248-keV-class event, which LZ does not see. P072 should quantify this with LZ's full 2D likelihood; the prediction for LZ's next data under that hypothesis is a
time-flat population with 70–85 % of events below 200 keV.

## 9. Figures

- `figures/P058_fig1_spectra.png` — Left: dR/dE per unit isoscalar coupling, 1 TeV, annual halo; exothermic (solid, f₂ = 1) vs endothermic (dashed) for δ = 300, 350, 380 keV; dots mark E*; shaded: WS ROI and the empty 600–1700 phd region; dotted line: 248 keV. Right: cumulative fraction of the exothermic spectrum.
- `figures/P058_fig2_f2_tau_mA.png` — 90 % CL upper limits on f₂ vs δ (three couplings, 400/4000 GeV variants, P011 criterion), the excluded lifetime boundary τ_min vs δ (P011/P026's 7×10¹⁶ s shown), and the α_D-independent m_A′ floor with P026's values.
- `figures/P058_fig3_modulation.png` — ROI rate / annual mean vs day of year for the exothermic signal (δ = 300, 350, 380 keV) and the endothermic δ = 300 keV signal.

## 10. References

1. LZ Collaboration, arXiv:2609.02823 (2026) — event, ROI, Fig. 4, Fig. S2, Fig. S4, MSSI table.
2. D. Tucker-Smith, N. Weiner, Phys. Rev. D 64, 043502 (2001) — inelastic dark matter.
3. B. Batell, M. Pospelov, A. Ritz, Phys. Rev. D 79, 115019 (2009) — exothermic scattering of excited dark-sector states.
4. P. W. Graham, D. E. Kaplan, S. Rajendran, M. T. Walters, Phys. Rev. D 82, 063512 (2010) — exothermic dark matter kinematics.
5. J. Bramante, P. J. Fox, G. D. Kribs, A. Martin, Phys. Rev. D 94, 115026 (2016) — inelastic frontier.
6. I. Jeong, S. Kang, S. Scopel, G. Tomar, Comput. Phys. Commun. 276, 108342 (2022) — WimPyDD.
7. D. Baxter et al., Eur. Phys. J. C 81, 907 (2021) — halo conventions.
8. Corpus: dossier 00, P002, P003, P006, P007, P011, P014, P016, P021, P026, P034, P038.

## 11. Tools and provenance (mirrors provenance/P058.json)

Agent tools: Read (guide, dossier, P002/P003/P006/P007/P011/P014/P016/P021/P026/P034/P038 papers, P011 details §3.6–3.7, tex lines 44–75, 120–164, 440–465, 712–786, 880–909, lzcommon lines 21–62/126–215/310–387, Fig. 4 and Fig. S4 PNGs, own figures ×3 (two rounds), dataviz palette reference), Bash (file listings, greps of tex/lzcommon/P011/P038/WimPyDD source, P011/P021/P038 result files, timing test, script runs, worker runs, debug snippet), Write (script, worker, details, provenance, paper), Edit (script ×9), Skill (dataviz).
Software: python 3.12.13; WimPyDD 2.0.4 (`diff_rate` per-stream kernels with δ < 0, `streamed_halo_function` via `lz.wd_halo`, `eft_hamiltonian` via `lz.wd_hamiltonian`); numpy 2.5.3; scipy 1.18.1 (special.erf, stats.poisson/norm, optimize.brentq/minimize_scalar); pandas 3.0.5; matplotlib 3.11.2; common/lzcommon.py (LZ, constants, E_R_range_keV, vmin_kms, wd, wd_halo, wd_hamiltonian, GEV_TO_CM2).
Recalled values (11): AMU, G_F, sin²θ_W, α, m_Z, ħ, t_U = 13.8 Gyr (all certain), ±1.5σ Gaussian acceptance 0.866 (certain), ⟨σv⟩/α_D relic relation not used, LZ 2024 low-energy tolerance ~3 events (uncertain; used only as a cross-check column), exothermic total rate ∝ v⁰ (textbook, certain).
WimPyDD-generated files: none outside `output/` (kernels cached under `output/work/P058/cache/`, 42 npz files).
