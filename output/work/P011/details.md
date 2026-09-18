# P011 — Dark-photon-mediated pseudo-Dirac dark matter: fitting the LZ event (research record)

Simulated date 2026-09-04. Author profile: dark-sector model builders. Category IDM/MODEL. Competes with P007 (Higgsino). Builds on P002 (kinematics), P003 (WimPyDD normalisation), P007 (Higgsino counts, digitised LZ intervals).

All numbers below are produced by `output/code/P011_dark_photon_idm.py` (results in `output/work/P011/*.csv|json`, log `run_log.txt`) unless marked [recall] or [paper].

## 1. Motivation and framework

P007 showed that the most predictive inelastic model, the pure Higgsino, has a gauge-fixed cross-section (σ_n = 7.4×10⁻³⁹ cm²) so large that it fits one LZ event only in a 20 keV window below the kinematic limit (δ ≈ 358–380 keV at 1 TeV). The other canonical inelastic model — Tucker-Smith–Weiner pseudo-Dirac DM with a dark-photon mediator — has a *free* normalisation. We ask what (m_χ, δ, m_A′, ε²α_D) reproduce one event, how the required cross-section compares with the Higgsino, and whether the parameter space survives dark-photon searches, the relic density and the requirement that the excited state χ₂ not survive to today.

**Model.** A Dirac fermion χ with U(1)_D charge, gauge boson A′ (mass m_A′, coupling g_D, α_D = g_D²/4π), kinetic mixing (ε/2cosθ_W) F′_{μν}B^{μν} [recall, certain: Holdom 1986]. A small Majorana mass splits χ into Majorana states χ₁, χ₂ with m₂ − m₁ = δ. Writing χ = (χ₁ + iχ₂)/√2, the vector current is purely off-diagonal,
χ̄γ^μχ = ½[χ̄₁γ^μχ₁ + χ̄₂γ^μχ₂ + iχ̄₁γ^μχ₂ − iχ̄₂γ^μχ₁] = iχ̄₁γ^μχ₂,
because Majorana vector currents vanish and χ̄₂γ^μχ₁ = −χ̄₁γ^μχ₂. Hence nuclear scattering is inelastic (χ₁N → χ₂N) with the *same* coupling strength g_D as the Dirac theory [recall, certain; Tucker-Smith & Weiner 2001]. After diagonalising the mixing, A′ couples to the electromagnetic current with εe (m_A′ ≪ m_Z), i.e. to protons only at the nucleon level (the neutron's charge radius/magnetic moment contributions are q²-suppressed and neglected).

**Heavy-mediator cross-section.** Integrating out A′ for m_A′ ≫ q gives L_eff = (e g_D ε/m_A′²)(χ̄₂γ^μχ₁)(p̄γ_μp), whose non-relativistic limit is the O₁ operator with
c_p = e g_D ε/m_A′² = 4π√(αα_D) ε/m_A′²,  c_n = 0.  (1)
With the NREFT normalisation σ_N = c_N²μ_N²/π (the one WimPyDD's test script and LZ's supplement formula (c m_v²)² = σ π m_v⁴/μ² use),
σ_p = c_p²μ_p²/π = 16π α α_D ε² μ_p²/m_A′⁴.  (2)  [recall, certain]
Sign conventions do not matter (rates ∝ c_p²). In WimPyDD's isospin convention (c⁰ = c_p + c_n, cⁱ = c_p − c_n; settled by P003) a proton-only coupling is c⁰ = c¹ = c_p; in Anand's convention c⁰_A = c¹_A = c_p/2.

**Light mediator.** For m_A′ ≲ q the coupling is q-dependent, c_p(q) = c_p(0) m_A′²/(m_A′² + q²), so the rate carries m_A′⁴/(m_A′² + q²)². q = √(2m_N E_R) = 0.246 GeV at 248 keV on ¹³¹Xe, so m_A′ = 0.3 GeV is already in the transition regime. Implemented with WimPyDD's q-dependent Wilson coefficient `{(1,'q2'): lambda q, A: [...]}` (syntax from P003); we verified that the lambda receives q in GeV (a coupling c ∝ q rescales the 248 keV rate by exactly q² = 0.0607) and that the propagator ratio at 248 keV is 0.3574 vs the analytic 0.3569 for m_A′ = 0.3 GeV.

**Unit normalisation.** We compute all counts for c_p = 1/m_v² (m_v = 246.2 GeV), which corresponds to σ_p,unit = 2.964×10⁻³⁸ cm² at 1 TeV (2.951×10⁻³⁸ at 300 GeV, 2.968×10⁻³⁸ at 3 TeV; the tiny mass dependence is μ_p²). Since N ∝ σ_p, σ_p(N = N₀) = σ_p,unit N₀/N_unit(δ).

## 2. Rate machinery

Identical to P007 so that the Higgsino comparison is apples-to-apples:
- WimPyDD 2.0.4, natural Xe, `diff_rate(..., sum_over_streams=False)` kernels on a common v_min grid (0–844 km/s, 1200 points, lzcommon default) dotted with three halo functions: **June 16** (day 167), **annual mean** (12 monthly samples), **Sun frame** (WimPyDD default, no Earth orbital motion — P007 found this reproduces LZ's interval edges best). Baxter-2021 SHM (v₀ = 238, v_esc = 544, v_⊙,pec = (11.1, 12.2, 7.3) km/s).
- Energy grid: 3 keV steps from 4 keV below the kinematic onset (¹²⁴Xe, v_max) to 330 keV; window integrals on a 1501-point interpolated grid (rate zero below the onset).
- Efficiency [paper]: plateau 0.96, 50 % at 5.4 and 269.9 keV, erf roll-offs with σ_lo = 2.5 keV and σ_hi = 11.5 keV (P007's calibration to the Fig. S2 inset), variants σ_hi = 8, 15 keV. Exposure 2.84 t·yr [paper].
- Three O₁ hamiltonians per (m, δ): proton-only unit (c⁰ = c¹ = 1/m_v²), isoscalar unit (c⁰ = 2/m_v², LZ's unit coupling), Higgsino (c_p = (G_F/√2)(1 − 4s_W²), c_n = −G_F/√2 [P007]).
- Grid: m_χ = 300, 1000, 3000 GeV; δ = 200 … 320 / 400 / 415 keV in 5 keV steps (25 + 41 + 44 = 110 points × 3 hamiltonians × ~80 energies; 463 s). Stored in `N_events_grid.csv` (columns N_{p,iso,hig}, their 5.4–55 keV and 200–270 keV sub-windows, and the no-efficiency total).
- Light mediators: m_A′ = 0.1, 0.3, 1 GeV at 1000 GeV, δ = 200–400 keV in 10 keV steps (`light_mediator_suppression.csv`).
- Exothermic (χ₂N → χ₁N): same kernels with δ → −δ, E = 1–330 keV (`exothermic.csv`).

**Validation.**
(i) Proton-only vs isoscalar unit rate at 3 and 5 keV (δ = 0): 0.1705, 0.1716 vs (Z/A)² = 0.1689 (1–2 %, isotope averaging).
(ii) Kernel·halo vs `lz.wd_rate` at 200 keV, δ = 300 keV, June halo: ratio 1.0000.
(iii) Higgsino counts vs P007 (1 TeV, annual, σ_hi = 11.5): 799/790 (δ = 300), 11.4/11.1 (350), 0.52/0.49 (370), 0.103/0.098 (380) — agreement 1–6 % (P007 used 2 keV steps and a 0.5 km/s velocity grid).
(iv) Low-energy companions per 200–270 keV event for the isoscalar unit coupling at δ = 200 keV: 7.99 vs P003's 7.6 (P003 used the Sun-frame halo and a slightly different roll-off).
(v) Light-mediator suppression at 248 keV: 0.3574 (WimPyDD) vs 0.3569 (analytic).
(vi) Efficiency-width sensitivity of N_p (1 TeV): σ_hi = 8/15 keV changes N by +0.2/−0.2 % (δ = 300), +0.9/−1.2 % (365), +1.9/−2.1 % (380 keV) — negligible compared with the halo choice.

## 3. Results

### 3.1 Proton-only spectra and the LZ conversion factor
The proton density's second M-response node lies at **284–291 keV** (proton-only) versus 257–267 keV (isoscalar) and 230–245 keV (Higgsino, isovector-dominated) — `spectra_1000GeV.json`. Consequently the proton-only spectrum keeps more of its rate inside the ROI: the fraction above 269.9 keV is 0.01/0.01/0.05/0.42 (δ = 250/300/350/380) against 0.02/0.05/0.26/0.92 for the isoscalar case. The q → 0 conversion between a proton-only and an isoscalar per-nucleon cross-section is (A/Z)² = 5.92; the **effective** conversion N_iso/N_p (unit couplings, 1 TeV, annual) is only **4.35, 4.23, 3.94, 2.67, 2.02, 1.29** at δ = 200, 250, 300, 350, 365, 380 keV. LZ's scalar-normalised Fig. S7 must therefore be multiplied by ≈ 4 (not 5.9) at δ = 250–300 keV and by ≈ 2.7 at 350 keV to read off proton-only cross-sections.

### 3.2 Cross-section for one event (`sigma_p_required.csv`, `summary.json`)
N_p,unit (annual, 1 TeV) = 2.41×10⁵, 2.47×10⁴, 3450, 2110, 99.3, 13.0, 0.53 at δ = 200, 250, 300, 310, 350, 365, 380 keV.

| m_χ | δ [keV] | σ_p(N = 1) annual | June 16 | Sun frame | 0.3–3 events (annual) |
|---|---|---|---|---|---|
| 300 | 250 | 2.41e-42 | 1.64e-42 | 2.64e-42 | 0.72–7.2e-42 |
| 300 | 300 | 2.22e-40 | 7.42e-41 | 8.10e-40 | 0.67–6.7e-40 |
| 300 | 310 | 2.29e-39 | | | |
| 1000 | 200 | 1.23e-43 | | | |
| 1000 | 250 | 1.20e-42 | 9.31e-43 | 1.26e-42 | 0.36–3.6e-42 |
| 1000 | 300 | 8.59e-42 | 6.10e-42 | 9.23e-42 | 2.6–26e-42 |
| 1000 | 350 | 2.99e-40 | 1.32e-40 | 4.60e-40 | 0.9–9e-40 |
| 1000 | 365 | 2.29e-39 | 7.77e-40 | 6.96e-39 | 0.7–6.9e-39 |
| 1000 | 380 | 5.60e-38 | 1.44e-38 | 1.25e-36 | 1.7–17e-38 |
| 3000 | 250 | 2.36e-42 | 1.87e-42 | 2.47e-42 | |
| 3000 | 300 | 1.40e-41 | 1.05e-41 | 1.48e-41 | |
| 3000 | 350 | 2.00e-40 | 1.12e-40 | 2.41e-40 | |
| 3000 | 365 | 7.36e-40 | | | |
| 3000 | 380 | 4.33e-39 | | | |

(cm²; the 90 % one-event PLR band 0.105–3.65 events is also tabulated.) The June/annual count ratio at 1 TeV is 1.29, 1.41, 2.26, 2.94, 3.89 at δ = 250, 300, 350, 365, 380 keV, so a single June event makes the required σ_p up to ×4 smaller if one conditions on the date (cf. P006).

**Comparison with LZ's published intervals** (`lz_band_proton_only.csv`; digitised edges from P007). Converting Fig. 6's (c₁ˢm_v²)² edges via N_edge = N_iso,unit × (c m_v²)²_edge and σ_p,edge = σ_p,unit N_edge/N_p,unit (i.e. through expected counts; shape differences between proton-only and isoscalar spectra are second order):

| δ | σ_SI upper [paper, digitised] | σ_p lower | σ_p median | σ_p upper | events at upper edge (Sun / annual) |
|---|---|---|---|---|---|
| 200 | 1.81e-43 | 5.8e-44 | 3.1e-43 | 7.9e-43 | 6.2 / 6.4 |
| 250 | 1.15e-42 | 3.6e-43 | 2.2e-42 | 4.9e-42 | 3.9 / 4.1 |
| 300 | 7.69e-42 | 1.5e-42 | 1.5e-41 | 3.0e-41 | 3.3 / 3.5 |
| 350 | 6.66e-40 | 1.1e-40 | 8.1e-40 | 1.8e-39 | 3.5 / 6.0 |

Our σ_p(N = 1) lies inside LZ's converted two-sided interval at every tabulated δ (e.g. 8.6×10⁻⁴² vs 1.5–30×10⁻⁴² cm² at 300 keV), as it must: the interval is LZ's own statement that one event is observed. The upper edges correspond to 3.3–6 expected events (Sun-frame halo 3.3–3.9 except at 200 keV), consistent with P007's finding.

### 3.3 Higgsino comparison (`summary.json`, Fig. 1–2)
The Higgsino's Xe amplitude, expressed as the proton-only cross-section giving the same LZ count, is 1.37×10⁻³⁸ cm² at q → 0 [(σ_n)((N − (1 − 4s_W²)Z)/Z)²] and, from the WimPyDD spectra, **7.9, 6.9, 3.4, 2.7, 5.8 ×10⁻³⁹ cm²** at δ = 250, 300, 350, 365, 380 keV (1 TeV; the shape mismatch between isovector and proton-only densities lowers it at large δ). The Higgsino line crosses the N = 1 contour at δ = **311.2 / 366.1 / 376.6 keV** for 300 / 1000 / 3000 GeV (annual; Sun frame 303.8 / 360.6 / 372.1; June 315.0 / 372.6 / 383.7), reproducing P007's 310 / 366 / ~376 keV. Away from those points the Higgsino overshoots by N_hig = 6620 (250 keV), 799 (300), 11.4 (350) at 1 TeV — the dark photon simply has σ_p smaller by the same factors: at δ = 300 keV the fit needs σ_p that is **800× below** the Higgsino-equivalent value. The dark-photon model therefore fits any δ from ≈ 204 keV (below which more than 3 low-energy 5.4–55 keV events accompany each 200–270 keV event: N_lo/N_hi = 4.1, 1.95, 0.77 at δ = 200, 210, 220 keV, 1 TeV) to δ_max(248 keV) = 316 / 387 / 407 keV, with σ_p rising from 1.2×10⁻⁴³ to 10⁻³⁷ cm² across that range.

### 3.4 Light mediators (`light_mediator_suppression.csv`)
S = N_light/N_heavy at equal c_p(0) (annual, 1 TeV): m_A′ = 0.1 GeV: 0.051, 0.034, 0.027 (δ = 250, 300, 350); 0.3 GeV: 0.495, 0.445, 0.403; 1 GeV: 0.926, 0.915, 0.903 (analytic values at q(248 keV): 0.020, 0.357, 0.889 — the spectrum peaks at lower q than 248 keV, so the effective suppression is milder). The required σ_p(q = 0) rises by 1/S, but the required ε does not fall as m_A′² indefinitely: for m_A′ ≪ q the effective coupling is e g_D ε/q², independent of m_A′, giving a **floor** ε_floor ≈ √(σ_req q⁴/(16πα α_D μ²)) — visible in Fig. 3 below m_A′ ≈ 0.2 GeV.

### 3.5 Kinetic mixing required (`epsilon_required.csv`, Fig. 3)
From (2): ε²α_D/m_A′⁴ = σ_p/(16πα μ_p²) (σ in GeV⁻²). At 1 TeV, δ = 300 keV: 1.04×10⁻¹³ GeV⁻⁴. Required ε for α_D = 0.1 (1 TeV, annual, N = 1, including S):

| δ [keV] | m_A′ = 0.3 GeV | 1 GeV | 3 GeV | 10 GeV |
|---|---|---|---|---|
| 250 | 4.0e-8 | 3.2e-7 | 2.8e-6 | 3.1e-5 |
| 300 | 1.1e-7 | 8.7e-7 | 7.4e-6 | 8.3e-5 |
| 350 | 6.9e-7 | 5.1e-6 | 4.4e-5 | 4.9e-4 |
| 365 | 2.0e-6 | 1.4e-5 | 1.2e-4 | 1.35e-3 |
| 380 | 1.0e-5 | 7.1e-5 | 6.1e-4 | 6.7e-3 |

For 300 GeV: ε(1 GeV, 10 GeV) = 4.6e-7, 4.4e-5 (δ = 250); 4.4e-6, 4.2e-4 (300 keV). For 3 TeV: 4.5e-7, 4.3e-5 (250); 1.1e-6, 1.1e-4 (300); 4.2e-6, 4.0e-4 (350 keV). Compared with the recalled BaBar bound ε ≲ 10⁻³ for 0.02 < m_A′ < 10 GeV [recall, likely: BaBar 2014 visible A′ → ℓℓ, BaBar 2017 invisible; NA64 and LHCb reach ε ~ 10⁻⁴–10⁻³ in sub-GeV/ dimuon windows, recall uncertain], every entry with δ ≤ 350 keV is allowed by 1–5 orders of magnitude; only the Higgsino-like corner δ ≥ 365 keV with m_A′ ≳ 8 GeV needs ε > 10⁻³. Since m_A′ ≤ 10 GeV ≪ 2m_χ, the A′ cannot decay to χχ̄ and decays visibly (A′ → ℓ⁺ℓ⁻, hadrons), so the visible-search bounds are the relevant ones; an A′ lighter than χ is of course allowed.

### 3.6 Relic density (`relic_lifetime.json`)
For m_A′ < m_χ the dominant annihilation is χχ̄ → A′A′ (χ₁χ₁, χ₂χ₂, χ₁χ₂ all contribute at freeze-out, T_f ≈ m/25 ≫ δ) with ⟨σv⟩ ≈ πα_D²/m_χ² [recall, likely; the (1 − m_A′²/m²)^{3/2} correction is negligible]. Setting ⟨σv⟩ = 2.2×10⁻²⁶ cm³/s [recall, certain: Steigman–Dasgupta–Beacom 2012] gives **α_D = 0.0073, 0.0245, 0.073** (g_D = 0.30, 0.55, 0.96) for 300, 1000, 3000 GeV. With thermal α_D the required ε at 1 TeV is 2.3e-7, 1.75e-6, 1.5e-5, 1.7e-4 (δ = 300 keV; m_A′ = 0.3, 1, 3, 10 GeV) and 4.0e-6, 2.9e-5, 2.5e-4, 2.7e-3 (365 keV). The s-channel χ₁χ₂ → A′* → ff̄ contributes a fraction ~ (αε²/α_D) Σ N_cQ_f² ≲ 10⁻³ (ε ≤ 10⁻³) and is irrelevant unless m_A′ > m_χ, in which case the model cannot be thermal with ε ≲ 10⁻³ — so m_A′ < m_χ is required.

### 3.7 χ₂ lifetime and the exothermic constraint (`chi2_lifetime.csv`, `exothermic.csv`, `mA_window.csv`)
For δ < 2m_e = 1.022 MeV, χ₂ → χ₁e⁺e⁻ is closed. χ₂ → χ₁γ vanishes at one loop for an on-shell photon (the A′*→γ transition through a fermion loop is transverse, Π^{μν} ∝ q²g^{μν} − q^μq^ν, and vanishes at q² = 0) [recall, likely]. The leading channel is χ₂ → χ₁νν̄ through the A′'s induced coupling to the Z current, g_ν = ε tanθ_W (m_A′²/m_Z²) g/(4cosθ_W) for m_A′ ≪ m_Z [recall, likely; hypercharge mixing]. The A′ propagator 1/m_A′² cancels the m_A′² in g_ν, giving an m_A′-independent four-fermion coupling
G_eff = g_D ε tanθ_W g/(4cosθ_W m_Z²) = 1.185×10⁻⁵ g_D ε GeV⁻².
*Derivation of the width (NR limit, δ ≪ m_χ).* M = G_eff ū₁γ^μu₂ ū_νγ_μP_Lv_ν; only μ = 0 survives for the heavy fermions (ūγ⁰u → 2m δ_ss′), Σ|ū_νγ⁰P_Lv_ν|² = 2E₁E₂(1 + cosθ), spin-averaged |M|² = 8G²m²E₁E₂(1 + cosθ), dΦ₃ ≈ (1/2m) d³p₁d³p₂/((2π)⁵4E₁E₂) δ(δ − E₁ − E₂). The angular average kills cosθ and ∫₀^δ E₁²(δ − E₁)²dE₁ = δ⁵/30, so
Γ(χ₂ → χ₁νν̄) = N_ν G_eff² δ⁵/(120π³), N_ν = 3.  (3)
(This is 1.6× the muon-decay-like G²δ⁵/(192π³) estimate P007 used for the Higgsino; the assignment's scaling ∝ ε²α_Dδ⁵/m_A′⁴ applies to the electron channel, closed here, or to a photon-only mixing where the neutrino coupling vanishes altogether.)
Lifetimes (α_D = 0.1): τ = 1.9×10¹⁸ s (60 Gyr) for ε = 10⁻⁶, δ = 300 keV; 1.9×10¹⁴ s for ε = 10⁻⁴; 4.7×10¹⁸ s (ε = 10⁻⁶, 250 keV); 7.2×10¹⁷ s (ε = 10⁻⁶, 365 keV). τ ∝ 1/(ε²α_Dδ⁵).

*Why χ₂ must decay.* χ₁ and χ₂ freeze out together (T_f ≫ δ) with equal abundances, and the χ₂f → χ₁f depletion rate at T ~ MeV, ~ n_f σv ~ T³ × 16πα α_D ε² T²/m_A′⁴ ≈ 4×10⁻²⁹ GeV for ε = 10⁻⁶, is far below H ~ T²/M_Pl ≈ 10⁻²⁵ GeV, so the primordial χ₂ fraction f₂ ≈ ½ survives unless χ₂ decays. A surviving χ₂ down-scatters *exothermically* with no velocity threshold: at δ = 300 keV the spectrum starts at 37 keV (E₋ at v_max) and extends to ~1.9 MeV; the exothermic ROI count per unit proton coupling is 2.96×10⁶ events (vs 3450 endothermic), i.e. **N_exo = 858 events** at the σ_p that gives one endothermic event, if f₂ = 1 (251, 858, 1.8×10⁴, 1.2×10⁵, 2.7×10⁶ at δ = 250, 300, 350, 365, 380 keV; the 5.4–55 keV part alone is 1.24×10⁶, 1.04×10⁵, 1270 per unit coupling at 250, 300, 350). Requiring ≤ 1 exothermic event gives f₂ ≤ 4.0×10⁻³, 1.2×10⁻³, 5.5×10⁻⁵, 8.2×10⁻⁶, 3.8×10⁻⁷ and, with f₂ = ½e^{−t_U/τ} ≈ e^{−t_U/τ} (t_U = 13.8 Gyr = 4.35×10¹⁷ s), τ ≤ 7.9, 6.5, 4.4, 3.7, 2.9 ×10¹⁶ s. Through (3) this is a **lower bound on ε** independent of m_A′:
ε ≥ 7.8, 5.4, 4.5, 4.4, 4.5 ×10⁻⁶ (α_D = 0.1); 1.6, 1.1, 0.90, 0.89, 0.90 ×10⁻⁵ (thermal α_D = 0.0245) at δ = 250, 300, 350, 365, 380 keV.
Because both this floor and the required ε scale as α_D^{−1/2}, the resulting **lower bound on m_A′ is α_D-independent**: combining ε_req(m_A′) ∝ m_A′² (with S) with the floor and with the recalled BaBar ceiling ε ≤ 10⁻³ gives the allowed window (1 TeV):

| δ [keV] | m_A′,min [GeV] | m_A′,max (α_D = 0.1) | m_A′,max (thermal α_D) |
|---|---|---|---|
| 250 | 5.0 | 57 | 40 |
| 300 | 2.6 | 34 | 24 |
| 350 | 0.94 | 14 | 10 |
| 365 | 0.52 | 8.5 | 6.0 |
| 380 | 0.09 | 3.8 | 2.7 |

(The upper bounds above 10 GeV extrapolate the 10⁻³ ceiling beyond BaBar's range; LHCb's dimuon search reaches ~70 GeV with comparable sensitivity [recall, uncertain].) A sub-GeV dark photon is thus **excluded for δ ≤ 350 keV** not by any search but by the consistency requirement that χ₂ has decayed. The window shrinks toward the Higgsino corner: at δ = 380 keV only m_A′ ≈ 0.1–4 GeV remains.

## 4. Figures
- `figures/P011_fig1_sigma_p_plane_1000GeV.png` — (δ, σ_p) plane at 1 TeV: N = 1 contour (annual; June and Sun-frame variants), 0.3–3 and 0.105–3.65 event bands, LZ's converted two-sided intervals (violet bars at 200–350 keV), the Higgsino-equivalent line (orange, crossing at 366 keV), light-A′ contours (σ_p at q = 0 for 0.3 and 0.1 GeV), δ_max(248 keV) and the δ < 204 keV region with > 3 low-energy companions.
- `figures/P011_fig2_sigma_p_three_masses.png` — σ_p(N = 1) with 0.3–3 bands and the Higgsino line for 300, 1000, 3000 GeV.
- `figures/P011_fig3_epsilon_vs_mA.png` — ε(m_A′) for N = 1 at α_D = 0.1 and δ = 250–380 keV, the recalled BaBar ceiling, the χ₂-decay floor band (4–8 ×10⁻⁶), and the propagator floor below m_A′ ≈ 0.2 GeV.
- `figures/P011_fig4_spectra_and_companions.png` — left: 1 TeV spectra, proton-only vs isoscalar (node at ≈ 290 vs ≈ 265 keV); right: 5.4–55 keV events per 200–270 keV event vs δ.

## 5. Failed or abandoned approaches
- First full run exceeded the 600 s foreground limit (3 hamiltonians × 110 δ × ~80 energies at 11 ms per WimPyDD call = 463 s plus light mediators); the harness moved it to the background; results were cached and the remaining parts re-run from cache. Grid caching (`--recompute` to redo) was added.
- δ = 366 keV (P007's Higgsino best fit) is not on the 5 keV grid; tables use 365 keV instead.
- The "low-energy companion" ratio was first normalised per ROI event (0.2 at δ = 200 keV); it was replaced by P003's per-200–270-keV-event definition (4.1 proton-only, 8.0 isoscalar at 200 keV) for comparability.
- A direct P007-grid interpolation for the Higgsino line was replaced by recomputing the Higgsino hamiltonian on our own kernels (agreement 1–6 %).

## 6. Discussion
The dark-photon pseudo-Dirac model is the mirror image of the Higgsino: where P007's coupling is fixed and δ is pinned to 358–380 keV, here σ_p is free and every δ between ~205 keV and δ_max fits with σ_p ranging over four decades — the LZ event constrains ε²α_D/m_A′⁴ at each δ but says nothing about δ itself. Three physics points nevertheless sharpen the model. (1) The proton-only nuclear response is measurably different from LZ's isoscalar template: its form-factor node sits at ≈ 290 keV rather than 265 keV, so more of the spectrum is inside the ROI and the isoscalar-to-proton conversion is ≈ 4 rather than (A/Z)² = 5.9; LZ's Data Release could include proton-only O₁ templates at negligible cost. (2) The relic density fixes α_D ≈ 0.007–0.07, so the LZ fit becomes a one-parameter relation ε ∝ m_A′² per δ; with δ ≤ 350 keV the required ε is 10⁻⁷–10⁻⁴ for m_A′ = 0.3–10 GeV, far below present dark-photon limits, so accelerator searches cannot test the model at these δ — unlike the Higgsino corner (δ ≳ 365 keV, m_A′ ≳ 8 GeV) where ε ≳ 10⁻³ is required and BaBar/LHCb already bite. (3) The excited state must have decayed: with δ < 2m_e its only relevant channel is νν̄ through Z mixing, and a surviving χ₂ would give 10²–10⁶ exothermic events in LZ at the fitted coupling. This yields ε ≳ 5×10⁻⁶ and hence m_A′ ≳ 0.5–5 GeV (α_D-independent) — the first argument we know of that excludes light (sub-GeV) dark photons for this event without invoking any laboratory bound. A caveat: the floor rests on the hypercharge form of the mixing (a photon-only mixing has no neutrino coupling and would leave χ₂ stable, which then excludes the model outright unless another decay channel exists) and on our NR width (3), whose order-of-magnitude uncertainty enters ε_min as a square root.

Predictions: the modulation is the same as any O₁ inelastic model (June/annual 1.4 at 300 keV, 2.9 at 365 keV); for δ ≥ 300 keV no low-energy companions; direct A′ searches are irrelevant for δ ≤ 350 keV; the exothermic signature is absent by construction. Unlike the Higgsino, the model makes no falsifiable rate forecast for LZ's next data beyond Poisson repetition of ~1 event per 2.8 t·yr at the fitted σ_p.

## 7. References
1. LZ Collaboration, arXiv:2609.02823 (2026) — Theory paragraph, Fig. 6 and S7 captions, supplement "Interpretation of O₁ as a cross-section", Table S7.
2. B. Holdom, Phys. Lett. B 166, 196 (1986) — kinetic mixing.
3. D. Tucker-Smith, N. Weiner, Phys. Rev. D 64, 043502 (2001) — inelastic DM.
4. B. Batell, M. Pospelov, A. Ritz, Phys. Rev. D 79, 115019 (2009) — multi-component secluded WIMPs, χ₂ decays and exothermic scattering.
5. J. Bramante, P. J. Fox, G. D. Kribs, A. Martin, Phys. Rev. D 94, 115026 (2016) — inelastic frontier.
6. J. P. Lees et al. (BaBar), Phys. Rev. Lett. 113, 201801 (2014) — dark-photon search, ε ≲ 10⁻³.
7. G. Steigman, B. Dasgupta, J. F. Beacom, Phys. Rev. D 86, 023506 (2012) — thermal ⟨σv⟩.
8. N. Anand, A. L. Fitzpatrick, W. C. Haxton, Phys. Rev. C 89, 065501 (2014); I. Jeong, S. Kang, S. Scopel, G. Tomar, Comput. Phys. Commun. 276, 108342 (2022) — NREFT and WimPyDD.
9. Corpus: P002 (kinematics), P003 (normalisation, N_lo definition), P006 (date), P007 (Higgsino, digitised intervals).

## 8. Tools and provenance (mirrors provenance/P011.json)
- Agent tools: Read (PAPER_GUIDE.md; 00_evidence_dossier.md; results_ledger.csv; papers P002/P003/P007; work/P007/N_events_grid.csv (first page) and lz_intervals_digitised.csv; lzcommon.py; P003_nreft_shapes.py; P007_higgsino_inelastic.py lines 85–100, 150–230; fulltext.tex lines 44–70, 282–295, 798–819; four output figures twice); Bash (greps of the tex, P007 grid/details/JSON, WimPyDD timing/syntax probe, palette lookup, five script runs incl. two aborted, verification snippets); Write (script, details.md, P011.json, P011.md); Edit (script fixes, 12); Skill (dataviz); ToolSearch + TaskStop (background task handling).
- Software: python 3.12.13; WimPyDD 2.0.4 (eft_hamiltonian incl. q-dependent WC, streamed_halo_function via lz.wd_halo, diff_rate per-stream kernels); numpy 2.5.3; scipy 1.18.1 (special.erf/erfc); pandas 3.0.5; matplotlib 3.11.2; common/lzcommon.py (LZ constants, kinematics, wd_halo, wd_hamiltonian, wd_rate, E_R_range_keV, delta_max_kev, vmax_kms, v_earth_kms). Algebra (Majorana current decomposition, width (3)) by hand.
- Recalled knowledge (14 items) listed in the JSON with reliabilities.
- WimPyDD-generated files: none (diff_rate used; no response-function files written).
- Data requests: none. Datasets: none.
