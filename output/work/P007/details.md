# P007 · Higgsino-like inelastic dark matter confronting the LZ event — research record

Simulated date 2026-09-04. Author profile: supersymmetry / electroweakino phenomenology group. Category IDM (model), hep-ph.
Competes with P011 (dark-photon pseudo-Dirac fit). Scripts: `output/code/P007_higgsino_inelastic.py`, `output/code/P007_higgsino_figures.py`.
All numbers below are printed by those scripts (logs `run_log.txt`, `run_log_cached.txt`) or stored in the CSV/JSON files listed in §9.

## 1. Motivation and framework

The LZ paper (arXiv:2609.02823) says its inelastic isoscalar operator O1^s "resembles a Higgsino model" and cites Graham, Ramani & Wong, PRD 111, 055030 (2025). A (nearly) pure Higgsino is a Dirac SU(2) doublet with hypercharge 1/2 whose neutral Dirac state is split into two Majorana states χ1, χ2 by mixing with the bino and wino. Z exchange couples χ1 to χ2 (not χ1 to χ1), so a pure Higgsino scatters *only inelastically* at tree level, with a coupling fixed by gauge invariance. The single free parameter that controls the LZ rate is therefore the splitting δ (and, weakly, the mass μ). We ask: (i) what is the fixed Z-exchange cross-section, (ii) which δ gives ≈ 1 event in 2.84 t·yr, (iii) is that compatible with LZ's two-sided intervals, (iv) what gaugino masses produce such a δ, and (v) what does the model predict for modulation and for LZ's next exposure.

Earlier corpus papers: at the time of writing no ledger rows exist (`output/results_ledger.csv` has only a header; `output/papers/` is empty), so the kinematic map (P002) and the WimPyDD normalisation (P003) could not be cited as results; we settled the WimPyDD convention ourselves (§3) and used `lzcommon` kinematics, which match the dossier's δ_max values.

## 2. Z-exchange couplings of a pure Higgsino (Part A of the script)

Inputs (recalled, certain): G_F = 1.166×10⁻⁵ GeV⁻², sin²θ_W = 0.231, m_Z = 91.19 GeV, m_W = 80.37 GeV; m_v = 246.2 GeV (paper); Xe: Z = 54, ⟨A⟩ = 131.29 (`lzcommon.XE_ISOTOPES`).

**Z coupling of the neutral Higgsino.** The neutral Dirac fermion ψ built from H̃_d⁰ (T₃ = +1/2) and H̃_u⁰ (T₃ = −1/2) has the same |T₃| = 1/2, Q = 0 for both chiralities, hence a pure vector coupling
  L_Z = (g / 2c_W) ψ̄ γ^μ ψ Z_μ .
Writing ψ = (χ1 + iχ2)/√2 with Majorana χ_i, the diagonal vector currents vanish and ψ̄γ^μψ = i χ̄1 γ^μ χ2: the whole vector coupling becomes the χ1–χ2 transition. Decomposing amplitudes, A(ψ→ψ) = i A(χ1→χ2) (fermion-number conservation forces A(ψ→ψ̄)=0, hence A(1→2) = −A(2→1)), so σ(χ1 N → χ2 N)|_{δ→0} = σ(ψ N → ψ N): the Majorana inelastic cross-section equals the Dirac elastic one. All local DM is χ1 (χ2 decays in weeks, §8), so the total rate is the same as for a Dirac Higgsino.

**Effective operator.** With g²/(c_W² m_Z²) = 4√2 G_F and the nucleon vector charges g_V^p = ¼(1 − 4 sin²θ_W) = 0.019, g_V^n = −¼,
  L_eff = 2√2 G_F g_V^N (χ̄1 γ^μ χ2)(N̄ γ_μ N)  ⇒  c_p^{O1} = (G_F/√2)(1 − 4 sin²θ_W) = 6.27×10⁻⁷ GeV⁻², c_n^{O1} = −G_F/√2 = −8.24×10⁻⁶ GeV⁻².
Per-nucleon cross-sections (σ_N = c_N² μ_N²/π, the paper's and WimPyDD's normalisation):
  σ_n = G_F² μ_n²/(2π) = **7.40×10⁻³⁹ cm²** at 1 TeV (7.37, 7.39, 7.40, 7.41, 7.41 ×10⁻³⁹ for 300, 500, 1000, 2000, 4000 GeV; heavy limit 7.42×10⁻³⁹), σ_p = 4.28×10⁻⁴¹ cm² (σ_p/σ_n = (1 − 4 sin²θ_W)² = 0.0058).
Nuclear level: σ_A = G_F² μ_A²/(2π) [(1 − 4 sin²θ_W)Z − N]² = 5.30×10⁻³¹ cm² for ¹³¹Xe at q → 0, 1 TeV (`higgsino_couplings.json`).

**Isospin decomposition.** Anand et al. convention c⁰ = (c_p + c_n)/2 = −3.81×10⁻⁶ GeV⁻², c¹ = (c_p − c_n)/2 = +4.44×10⁻⁶ GeV⁻², c¹/c⁰ = −1.165: the Higgsino is *dominantly isovector*, not isoscalar. (c⁰ m_v²)² = 0.0533, (c¹ m_v²)² = 0.0723. WimPyDD convention (§3): c⁰_WD = c_p + c_n = −7.62×10⁻⁶, c¹_WD = c_p − c_n = +8.87×10⁻⁶ GeV⁻².

**Equivalent isoscalar coupling and the paper's factor 3.2.** The O1 nuclear amplitude is c_p Z + c_n N. An isoscalar coupling c^s gives c^s A. Matching at q → 0 on ⟨A⟩ = 131.29: c_eq^s = |c_p Z + c_n N|/A = (G_F/√2)·|N − (1−4s_W²)Z|/A = 4.60×10⁻⁶ GeV⁻², so
  **(c_eq^s m_v²)² = 0.0777**, σ_SI,eq = c_eq² μ_n²/π = **2.30×10⁻³⁹ cm²** (1 TeV; 2.29–2.31 for 300–4000 GeV).
The ratio σ_n^vector/σ_SI,eq = (A/((A−Z) − (1−4 sin²θ_W)Z))² = **3.214** for ⟨A⟩ (3.05 for ¹³⁶Xe … 3.54 for ¹²⁴Xe; abundance-weighted amplitude 3.215), which is exactly the "≃ 3.2 (for Xe)" the supplement quotes: a scalar-normalised σ_SI limit must be multiplied by 3.2 to become a limit on the per-neutron vector cross-section because the proton's vector charge nearly vanishes.

## 3. WimPyDD conventions and validation (Part B)

- Isoscalar δ = 0: with WimPyDD's own mapping (`lz.wd_c_SI_from_sigma_n`, c⁰ = c_p + c_n) for σ_n = 10⁻⁴⁵ cm², 1000 GeV, the WimPyDD rate / Helm rate (`lz.dRdE_SI`) at 10, 20, 30, 40, 50 keV = 1.000, 0.990, 0.986, 0.992, 1.014. Hence **WimPyDD's c⁰ = c_p + c_n, c¹ = c_p − c_n** (a factor 2 relative to Anand's c^τ; factor 4 in rate), confirming the note in PAPER_GUIDE.
- Isospin sign: neutron-only (c⁰, c¹) = (c, −c) over proton-only (c, c) rate at 0.5, 2, 5 keV = 2.052, 2.035, 2.000 versus Σ_i f_i N_i²/Z² = 2.055 at q → 0. Sign and normalisation of c¹ confirmed; the fall with E is the different neutron/proton form factors.
- Per-stream kernel: `WD.diff_rate(..., delta_eta = ones, sum_over_streams=False)` returns the contribution per v_min bin; dotting it with any halo's δη reproduces the summed call to 1e-15 relative (probe), which lets one WimPyDD call per (E, m, δ) serve all halos.
- Halo grid: `wd_halo()` without an explicit `vmin` grid truncates the June halo at v_sun + v_esc = 794.6 km/s (grid artefact). All halos here are evaluated on a common grid 0–830 km/s (0.5 km/s steps): Sun-frame vmax 795.5, 16 June 810.5 km/s.

## 4. Digitised LZ intervals (Part C)

The two-sided 90% intervals are vector paths in the source PDFs (Fig6_O1_L10_limit_stacked.pdf drawings 228/229/230 = median/upper/lower; FigS7_O1_as_SI.pdf drawings 96/97), read with PyMuPDF and mapped with the tick-label positions. Table (`lz_intervals_digitised.csv`), 1000 GeV:

| δ (keV) | lower (c₁^s m_v²)² | median | upper | σ_SI lower (cm²) | σ_SI upper (cm²) |
|---|---|---|---|---|---|
| 0 | – | 2.9e-9 | 1.64e-9 | – | 5.0e-47 |
| 100 | 1.5e-9 | 5.2e-8 | 7.7e-8 | 4.5e-47 | 2.4e-45 |
| 200 | 4.5e-7 | 2.4e-6 | 6.1e-6 | 1.4e-44 | 1.9e-43 |
| 250 | 2.9e-6 | 1.7e-5 | 3.9e-5 | 8.8e-44 | 1.2e-42 |
| 300 | 1.28e-5 | 1.32e-4 | 2.60e-4 | 3.9e-43 | 7.9e-42 |
| 350 | 1.38e-3 | 1.03e-2 | 2.25e-2 | 4.2e-41 | 6.9e-40 |

Cross-check: converting the digitised couplings with the paper's formula σ = (c m_v²)² μ²/(π m_v⁴) reproduces the digitised Fig. S7 within 3 % (6.66e-40 vs 6.86e-40 cm² at 350 keV), so both digitisations and the formula are consistent. Values quoted to two figures are "approximate" at the ±5 % level of the line width.

## 5. Expected LZ counts (Part D)

N(m, δ) = 2.84 t·yr × ∫ ε(E) dR/dE dE, dR/dE from WimPyDD (shell-model responses, natural Xe, ρ₀ = 0.3 GeV/cm³) with the fixed couplings (c⁰_WD, c¹_WD) of §2; E grid 2 keV from E₋(v = 830 km/s) to 330 keV; trapezoid.
Efficiency model (paper: plateau 0.96, 50 % at 5.4 and 269.9 keV; Fig. S2 inset shape): ε(E) = 0.96 · ½[1 + erf((E−5.4)/(√2·2.5))] · ½ erfc((E−269.9)/(√2·σ_hi)), σ_hi = 11.5 keV (reproduces 75 % at ≈262 and 25 % at ≈278 keV); σ_hi = 8 and 15 keV as variations. No additional energy smearing: the paper's efficiency is already defined in true NR energy including the ROI cut.
Halos (Baxter-2021 SHM, v₀ = 238, v_esc = 544, v_sun,pec = (11.1, 12.2, 7.3) km/s): `sun` = Sun frame, no Earth orbital motion (WimPyDD default usage); `annual` = mean of 12 day-of-year halos (linear in δη, verified equal to the mean of daily rates); `june16` = day 167; `dec16` = day 350; `annual_vesc528/560`.

**Grid (annual halo, σ_hi = 11.5), events in 2.84 t·yr** (`N_events_grid.csv`):

| δ (keV) | 300 GeV | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|
| 0 | 3.8e8 | 2.4e8 | 1.2e8 | 6.0e7 | 3.0e7 |
| 100 | 1.7e7 | 1.2e7 | 7.2e6 | 3.9e6 | 2.0e6 |
| 200 | 6.8e4 | 8.1e4 | 6.4e4 | 4.0e4 | 2.2e4 |
| 250 | 3590 | 6650 | 6580 | 4470 | 2590 |
| 300 | 23.2 | 439 | 790 | 644 | 401 |
| 350 | 0 | 0.139 | 11.1 | 21.3 | 18.1 |
| 360 | 0 | 0 | 2.59 | 7.13 | 6.90 |
| 370 | 0 | 0 | 0.49 | 1.98 | 2.24 |
| 380 | 0 | 0 | 0.098 | 0.485 | 0.628 |
| 390 | 0 | 0 | 0.0093 | 0.124 | 0.171 |
| 400 | 0 | 0 | 0 | 0.028 | 0.049 |

Without efficiency and over all energies (`N_noeff_all_E`): 939 (300 keV) and 26.3 (350 keV) at 1 TeV, i.e. the ROI edge removes 58 % of the rate at δ = 350 keV, where most recoils lie above 270 keV (Fig. 2).

**Normalisation check against the paper.** Using the pure-isoscalar coupling (c⁰_WD, c¹_WD) = (2/m_v², 0) (the paper's unit coupling c_p = c_n = 1/m_v²), N_unit(1000 GeV) = 1.04e5, 1.35e4, 259 events at δ = 250, 300, 350 keV (annual). Multiplying by the digitised interval edges gives the number of signal events LZ's interval corresponds to (`implied_N_at_LZ_interval_edges.csv`):

| δ | halo | N at lower edge | at median | at upper edge |
|---|---|---|---|---|
| 250 | sun / annual / june | 0.29 / 0.30 / 0.39 | 1.73 / 1.81 / 2.33 | 3.84 / 4.03 / 5.18 |
| 300 | sun / annual / june | 0.16 / 0.17 / 0.25 | 1.65 / 1.78 / 2.55 | 3.24 / 3.49 / 5.01 |
| 350 | sun / annual / june | 0.21 / 0.36 / 0.87 | 1.55 / 2.66 / 6.41 | 3.38 / 5.82 / 14.0 |

A profile-likelihood two-sided 90 % interval for one observed event with negligible background is [0.105, 3.65] signal events (2(μ−1−ln μ) = 2.706; Feldman–Cousins [0.11, 4.36]). The Sun-frame halo reproduces the upper edges to 5–12 % at all three δ (3.84, 3.24, 3.38 vs 3.65), the annual mean to 10–60 %; the June-only halo overshoots. We infer that LZ's spectra were computed with the time-averaged (Sun-frame) SHM, as WimPyDD does by default; the lower edges (0.16–0.29 vs 0.105) suggest a power constraint or asymptotic approximation on LZ's side. Either way our rate normalisation and efficiency reproduce the published intervals to within tens of per cent, which is sufficient for the δ determination (N falls by a factor ≈ 5 per 10 keV of δ near 370 keV, so a 50 % rate error shifts δ by ≈ 2.5 keV).

## 6. Required splitting, comparison with the paper, robustness (Part E)

δ giving N = 1 (log-interpolation on the falling branch), `delta_for_N_events.csv`, σ_hi = 11.5 keV:

| m (GeV) | δ(N=3.65) | δ(N=3) | **δ(N=1)** | δ(N=0.3) | δ(N=0.105) | Sun frame δ(N=1) | June-only δ(N=1) | v_esc 528 / 560 | δ_max(248 keV, 16 June) | δ_max(269.9 keV) |
|---|---|---|---|---|---|---|---|---|---|---|
| 300 | 306.8 | 307.5 | **310.2** | 310.9 | 311.4 | 302.6 | 310.9 | 300.3 / 311.4 | 315.8 | – |
| 500 | 336.4 | 337.4 | **342.4** | 347.1 | 351.1 | 335.8 | 347.4 | 331.4 / 350.9 | 356.3 | – |
| 1000 | 357.8 | 359.1 | **365.8** | 373.0 | 379.6 | 360.4 | 372.0 | 355.0 / 374.7 | 386.6 | 390.8 |
| 2000 | 365.4 | 366.9 | **374.9** | 383.4 | 391.2 | 370.1 | 381.6 | 364.4 / 383.9 | 401.8 | – |
| 4000 | 365.8 | 367.5 | **376.4** | 385.7 | 393.9 | 372.1 | 383.4 | 366.3 / 385.5 | 409.4 | – |

Efficiency width 8 / 15 keV changes δ(N=1) by −0.3 / +0.4 keV (1 TeV) — negligible. The dominant systematics are the halo tail (v_esc ±16 km/s: ∓10 keV; Sun-frame vs annual: 5 keV) and the choice of time-averaging.
Summary: **a pure Higgsino gives one LZ event only for δ ≈ 358–380 keV at 1 TeV (90 % one-event band; best value 366 keV), 365–391 keV at 2 TeV, 366–394 keV at 4 TeV; ≈ 342 keV at 500 GeV and ≈ 310 keV at 300 GeV.** In every case δ(N=1) lies 10–33 keV below the kinematic limit δ_max(248 keV), so the event energy is compatible, and the table-S7 grid (δ ≤ 350 keV) does not reach the allowed window for m ≥ 1 TeV.

**Comparison with the published intervals (1000 GeV)** (`comparison_with_LZ_intervals.json`): the Higgsino's (c_eq^s m_v²)² = 0.0777 exceeds the upper edge by ×2000 (δ = 250), ×299 (300) and ×3.5 (350 keV); in expected events the excess is ×1630, ×226 and ×1.9 (N_Higgsino = 6580, 790, 11.1 vs N_upper = 4.0, 3.5, 5.8). The pure Higgsino is therefore excluded at 90 % for every tabulated δ ≤ 350 keV and enters the allowed band only at δ ≳ 358 keV — beyond the paper's grid. The N-ratio is smaller than the coupling ratio (0.81, 0.76, 0.55 of it at 250, 300, 350 keV): the q → 0 equivalence c_eq^s A = c_pZ + c_nN does not hold at q ≈ 250 MeV, where the neutron and proton shell-model densities differ, so the isovector-dominated Higgsino spectrum is 20–45 % softer than the isoscalar one in the ROI. The "factor 3.2" is thus a q → 0 statement; at 350 keV the effective conversion is ≈ 3.2/0.55 ≈ 6.

**Modulation and forecast** (`modulation_and_forecast.csv`, annual halo, δ rounded to the grid):

| m (GeV) | δ (keV) | N annual | N June-16 rate | N Dec-16 rate | June/Dec | June/annual | events in next 1000 live days (4.71 t) |
|---|---|---|---|---|---|---|---|
| 300 | 310 | 1.54 | 6.35 | 0 | ∞ | 4.1 | 7.0 |
| 500 | 340 | 1.79 | 6.08 | 0.008 | 775 | 3.4 | 8.2 |
| 1000 | 365 | 1.14 | 3.24 | 0.070 | 46 | 2.8 | 5.2 |
| 2000 | 375 | 0.99 | 2.55 | 0.119 | 21 | 2.6 | 4.5 |
| 4000 | 377.5 | 0.87 | 2.14 | 0.130 | 16.5 | 2.5 | 4.0 |
| 1000 | 350 | 11.1 | 27.4 | 1.16 | 23.7 | 2.5 | 50.6 |
| 1000 | 300 | 790 | 1154 | 464 | 2.5 | 1.5 | 3593 |

At the fitted δ the signal is almost entirely a June phenomenon (June/December 16–∞); LZ's next 1000 live days should contain 4–8 events, essentially all within ±2 months of 2 June. The 16 June 2023 date of the event is the expected phase.

## 7. Gaugino masses required for δ (Part F)

Neutralino mass matrix in (B̃, W̃³, H̃_d⁰, H̃_u⁰); for |μ| ≪ M₁, M₂ integrating out the gauginos gives the 2×2 Higgsino matrix
  M_H = [[−ε c_β², −μ + ε c_β s_β], [−μ + ε c_β s_β, −ε s_β²]],  ε ≡ m_Z² (sin²θ_W/M₁ + cos²θ_W/M₂).
sympy (series in ε): |λ₋| − |λ₊| = ε + O(ε²/μ) — the tan β dependence cancels in the neutral–neutral splitting at leading order (the (1 ± sin 2β) factors familiar from the literature belong to the individual mass shifts and to the charged–neutral splitting). So
  **δ ≃ m_Z² (sin²θ_W/M₁ + cos²θ_W/M₂)**  [derived here; consistent with the standard result, recalled: likely].
Numerical diagonalisation of the full 4×4 (`gaugino_masses_for_splitting.csv`; μ = 1 TeV, tan β = 10; tan β = 2 and μ = 300, 4000 GeV also tabulated, differences < 3 %):

| hierarchy | δ = 300 keV | 350 | 370 | 385 |
|---|---|---|---|---|
| bino only (M₂ → ∞): M₁ | 6540 TeV | 5590 | 5280 | 5070 |
| M₂ = 2M₁: M₁ (M₂) | 17 100 (34 100) TeV | 14 600 (29 200) | 13 800 (27 700) | 13 300 (26 600) |
| wino only (M₁ → ∞): M₂ | 21 500 TeV | 18 400 | 17 400 | 16 700 |
| M₂ = −2M₁ (partial cancellation): M₁ | 4250 TeV | 3650 | 3450 | 3310 |

The leading-order formula reproduces these to ≤ 2 %. For orientation, M₁ = 10 TeV (bino only) gives δ = 198 MeV, M₁ = 100 TeV gives 19.3 MeV, and M₁ = M₂ = 1000 TeV gives 8.3 MeV (`splitting_summary.json`). **The LZ-compatible window δ ≈ 360–390 keV requires gaugino masses of 3–30 PeV**, i.e. an extremely split spectrum in which the Higgsino is the only light superpartner (the regime of Graham–Ramani–Wong). Electroweak loops do not split the two neutral Majorana states (they conserve the Dirac fermion number of ψ), so no radiative floor competes with this tree-level δ.
Charged–neutral splitting: tree-level 0.12–0.35 MeV at these gaugino masses (table) plus the radiative ≈ 355 MeV [Thomas & Wells 1998; recalled: likely, the assignment's "≈ 340 MeV" is within the usual quoted range], so m_χ± − m_χ1 ≈ 0.35 GeV and the chargino decays promptly (cτ ~ mm–cm).

## 8. Constraints and context (Part G and recalled bounds)

- LEP: m_χ± > 92–103.5 GeV (recalled, certain). LHC: for Δm± ≈ 0.35 GeV, soft-lepton searches do not apply; ATLAS disappearing-track searches exclude pure Higgsinos up to ≈ 200 GeV (recalled, likely); mono-jet limits are weaker. All masses considered (300–4000 GeV) are allowed.
- Elastic scattering: the tree-level Z-exchange elastic amplitude vanishes for Majorana χ1; residual elastic couplings arise at O(m_Z²/(μM)) ~ 10⁻⁴ (mixing) and at one loop, where Hill & Solon (2014) find σ_SI ≈ 10⁻⁴⁹–10⁻⁴⁸ cm² for a pure doublet with a strong cancellation (recalled, likely) — 2–3 orders below LZ's 2024 limit at 1 TeV (~5×10⁻⁴⁷ cm², recalled: likely). The elastic channel cannot test this model; the inelastic channel is the only one.
- Relic abundance: a thermal pure Higgsino gives Ω h² ≈ 0.12 at μ ≈ 1.1 TeV (recalled, certain); scaling Ω h² ∝ μ²: 0.0089, 0.025, 0.099, 0.40, 1.6 for 300, 500, 1000, 2000, 4000 GeV. So 300–500 GeV Higgsinos are 7–21 % of the DM if thermal, and their LZ rate scales down accordingly (δ(N=1) would move ≈ 5–10 keV lower); 2–4 TeV Higgsinos are over-produced thermally and need dilution (late entropy injection) or a non-standard history; non-thermal production (moduli/gravitino decay) can fill the deficit at low mass. The 1 TeV case is the natural thermal candidate (fraction 0.83).
- Excited state: Γ(χ2 → χ1 ν ν̄) ~ 3·2·G_F² δ⁵/(192π³) gives τ ≈ 2×10⁶, 9×10⁵, 6×10⁵ s for δ = 300, 350, 375 keV (order-of-magnitude estimate; the radiative χ2 → χ1 γ channel is comparable) — χ2 is not a cosmological relic; only endothermic up-scattering occurs (no exothermic signal in low-energy data).

## 9. Figures

- `figures/P007_fig1_N_vs_delta_and_required_delta.png` — (a) expected LZ events vs δ for 300–4000 GeV at the fixed Higgsino coupling (annual-mean SHM); grey band = 0.105–3.65 events (90 % one-event PLR band), dashed lines N = 3, 1, 0.3; coloured ticks at δ_max(248 keV, 16 June). (b) δ(N = 1) vs mass with the N = 3…0.3 and 90 % bands, the Sun-frame and June-only alternatives, and δ_max(248 keV) and δ_max(269.9 keV).
- `figures/P007_fig2_spectra_1000GeV.png` — 1 TeV recoil spectra (before efficiency) for δ = 300–390 keV; shading where efficiency < 50 %. The shell-model M-response minimum at ≈ 235–240 keV sits just below the event energy; for δ ≥ 350 keV most of the rate lies above the 269.9 keV edge.

## 10. Failed or abandoned approaches

- First run crashed in Part E because the 300 GeV curve fell to exactly zero between grid points (the N = 1 crossing was not bracketed); fixed by flooring N at 10⁻⁸ in the log-interpolation and caching the WimPyDD grid (`--recompute` to redo).
- The default `wd_halo(day_of_year=…)` grid truncates the June tail at 794.6 km/s; abandoned in favour of an explicit 0–830 km/s grid.
- A twin-axis efficiency overlay in Fig. 2 was replaced by shaded low-efficiency regions.
- Reading Fig. 6/S7 by eye was superseded by exact vector-path digitisation.

## 11. Discussion

The Higgsino is the most predictive inelastic model on the table: gauge invariance fixes σ_n = 7.4×10⁻³⁹ cm², six orders of magnitude above the σ_SI ≈ 10⁻⁴²–10⁻⁴⁰ cm² that LZ's scalar-normalised intervals read at δ = 300–350 keV. The huge coupling is tamed only by kinematics: N(δ) falls by ~10⁴ between δ = 300 and 370 keV at 1 TeV, so a single event pins δ to a 20 keV window just below δ_max. Three consequences follow. (1) The LZ-preferred O1^s points at δ = 300–350 keV are *not* Higgsino points — a Higgsino at those splittings would have given hundreds to tens of events. (2) The required gaugino masses (3–30 PeV) place the model in the extreme split-SUSY corner, where nothing else is accessible at colliders; the only cross-checks are indirect detection (thermal 1.1 TeV Higgsino annihilation, P025/P084 territory) and the modulation/forecast above. (3) The prediction is sharply falsifiable: 4–8 events in LZ's next 1000 live days, concentrated around June, with recoil energies ≥ 200 keV and a spectrum piling up at the ROI edge; extending the ROI above 270 keV would roughly double the sensitivity (58 % of the rate at δ = 350 keV is above the edge). If XENONnT/PandaX-4T extend their ROIs, a fixed-coupling model gives absolute predictions for them too. Caveats: the halo tail (v_esc, non-Maxwellian tails) shifts δ(N=1) by ±10 keV; the paper's grid ends at 350 keV so LZ's own likelihood in the window is unknown (the grid's "not physical" entries at 400 GeV/350 keV are consistent with our δ_max); the Sun-frame vs annual-mean choice changes N by 2.5× at fixed δ; and one event cannot distinguish a Higgsino from any other O1-like inelastic model — only the fixed normalisation makes the Higgsino case special.

## 12. References

1. LZ Collaboration, arXiv:2609.02823 (2026) — Theory paragraph, Fig. 1, Fig. 6, Table S7, Fig. S2, Fig. S7, supplement "Interpretation of O1 as a cross-section".
2. P. W. Graham, H. Ramani, S. S. Y. Wong, "Enhancing direct detection of higgsino dark matter", Phys. Rev. D 111, 055030 (2025), arXiv:2409.07768.
3. D. Tucker-Smith, N. Weiner, Phys. Rev. D 64, 043502 (2001) — inelastic dark matter.
4. J. Bramante, P. J. Fox, G. D. Kribs, A. Martin, Phys. Rev. D 94, 115026 (2016) — inelastic frontier.
5. N. Nagata, S. Shirai, "Higgsino dark matter in high-scale supersymmetry", JHEP 01 (2015) 029 — Higgsino splittings and direct detection (recalled reference).
6. R. J. Hill, M. P. Solon, Phys. Rev. Lett. 112, 211602 (2014); Phys. Rev. D 91, 043505 (2015) — loop-level WIMP–nucleon scattering for electroweak multiplets (recalled reference).
7. S. D. Thomas, J. D. Wells, Phys. Rev. Lett. 81, 34 (1998) — radiative chargino–neutralino splitting (recalled reference).
8. N. Anand, A. L. Fitzpatrick, W. C. Haxton, Phys. Rev. C 89, 065501 (2014) — NREFT conventions.
9. S. Kang, S. Scopel, G. Tomar, J.-H. Yoon, WimPyDD, Comput. Phys. Commun. (2022).
10. Corpus: `output/00_evidence_dossier.md` (δ_max values, June timing); P002, P003 planned but not yet available at the time of writing; P011 (competing dark-photon fit), P014, P025, P037, P048, P084 (follow-ups).

## 13. Tools and provenance (mirrors `output/provenance/P007.json`)

- Agent tools: Read (PAPER_GUIDE.md, 00_evidence_dossier.md, 01_landscape_and_plan.md, lzcommon.py, fulltext.tex lines 40–80, 270–300, 795–825, 880–911, Fig6/FigS7/FigS2 PNGs, two output figures), Bash (directory listing/ledger/env versions; grep of the tex; four WimPyDD/PyMuPDF probes; two script runs; figure run; table printing), Write (two scripts, details.md, P007.json, P007.md), Edit (script fixes), Skill (dataviz).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (special.erf/erfc, optimize.brentq); pandas 3.0.5; sympy 1.14.0 (series/eigenvals); matplotlib 3.11.2; pymupdf 1.28.2 (get_drawings, get_text); WimPyDD 2.0.4 (eft_hamiltonian, streamed_halo_function, diff_rate with sum_over_streams=False, Xe target); common/lzcommon.py (LZ, XE_ISOTOPES, mu_red, E_R_range_keV, delta_max_kev, vmax_kms, v_earth_kms, dRdE_SI, wd, wd_halo, wd_hamiltonian, wd_c_SI_from_sigma_n, wd_rate, M_V_GEV, GEV_TO_CM2).
- Recalled knowledge (12 items): G_F, sin²θ_W, m_Z, m_W (certain); Z couplings of fermions / nucleon vector charges (certain); neutralino and chargino mass matrices (certain); LEP chargino bound 92–103.5 GeV (certain); thermal Higgsino Ω h² = 0.12 at 1.1 TeV and Ω ∝ μ² (certain/likely); radiative Δm± ≈ 355 MeV (likely); Hill–Solon loop σ_SI 10⁻⁴⁹–10⁻⁴⁸ cm² (likely); LZ 2024 SI limit ≈ 5×10⁻⁴⁷ cm² at 1 TeV (likely); ATLAS disappearing-track Higgsino bound ≈ 200 GeV (likely); μ-decay-like three-body width formula (certain, applied as order of magnitude).
- WimPyDD-generated files: none (diff_rate does not write response-function files; no `wimp_dd_rate` calls).
- Datasets: none. Data requests: none.
