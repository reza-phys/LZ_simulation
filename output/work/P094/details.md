# P094 — Multi-component dark matter with a populated excited state: electron-recoil lines, low-threshold recoils and the two-state phenomenology across experiments

Research record (simulated date 2026-09-16; hep-ph; MODEL). Script `output/code/P094_multicomponent.py` (runtime 26 s with the
WimPyDD spectra cached in `output/work/P094/cache/`, ~50 s cold). Tables in `output/work/P094/*.csv`, `P094_results.json`,
`run_log.txt`; figures in `figures/`. P072 (exothermic origin of the event) and P075 (thermal history) did not exist in
`output/papers/` at writing time; P058 §5.6 is used in their place.

## 1. Motivation and question

P026 leaves f₂ = 0.42–0.50 of the dark matter in χ₂ after freeze-out; P058 showed that a surviving fraction would flood LZ's
125–200 keV bin and derived f₂(today) < 2.8×10⁻³ (δ = 300 keV) … 1.7×10⁻⁷ (380 keV), τ(χ₂) < (3–9)×10¹⁶ s and m_A′ ≥ 2.4 GeV.
We take the *model-building* view: suppose χ₂ is genuinely metastable (τ ≳ t_U, e.g. only χ₂ → χ₁νν̄ through Z mixing, or
χ₂ → χ₁3γ) or is continuously re-populated. What direct-detection signals follow **outside LZ's nuclear ROI**?
(1) exothermic DM–electron scattering χ₂e → χ₁e, a mono-energetic electron-recoil (ER) line at E ≈ δ; (2) the same line in
other detectors with high-energy ER spectra; (3) exothermic nuclear recoils on light targets (Ge, Si, CaWO₄, Ar, NaI);
(4) re-population of χ₂ today (self-scattering, up-scattering in the Earth, Sun, ISM, cosmic rays); (5) the combined (τ, f₂)
map from P058, P066, P026 and this work. We do **not** redo P058's nuclear limits; we use its R_ROI, f₂ and τ₀ tables.

## 2. Framework and equations

### 2.1 Exothermic DM–electron kinematics (Part A)
Exact relativistic two-body kinematics: χ₂ (mass m₂ = m + δ) with lab speed v strikes a free electron at rest.
s = m₂² + m_e² + 2γm₂m_e; CM electron energy E*_f = (s + m_e² − m₁²)/(2√s), p*_f = √(E*_f² − m_e²); lab energy
E_e = γ_cm(E*_f + β_cm p*_f cosθ*), β_cm = γm₂β/(γm₂ + m_e). For m → ∞ the electron kinetic energy is
  T_e = δ − p_f²/(2m) + **p**_f·**v** + O(v²),  p_f = √(δ² + 2m_eδ)  (relativistic electron: 630 keV at δ = 300 keV). (1)
The deposited energy is T_e for a free electron; for a *bound* electron the vacancy energy is re-emitted locally, so the
calorimetric deposit is δ + KE(χ₂) − KE(χ₁) = δ + **q**·**v** − q²/(2m) regardless of the binding (only the rate, not the
line position, feels the atomic momentum distribution; with q = 630 keV ≫ p_bound for all but the K shell the impulse
approximation holds to a few per cent).

### 2.2 Exothermic DM–electron cross-section (Part B)
Vector (dark-photon) exchange, heavy DM: the DM current is 2m δ^{μ0}; the electron trace gives
½Σ|ū(p_f)γ⁰u(p_i)|² = 2(E_iE_f + **p**_i·**p**_f + m_e²) → 2m_e(2m_e + T) for an electron at rest. With
dσ/dΩ = |M̄|²/(64π²s)·(p_f/p_i) and s ≃ m²:
  σ = (g_Dεe)²/m_A′⁴ · m_e(2m_e + T)/(2π) · p_f/p_i  →  σ v = σ_e c (p_f/m_e)(1 + δ/2m_e),  (2)
where σ_e = 16παα_Dε²μ_e²/m_A′⁴ is the standard non-relativistic DM–electron cross-section (the T → 0, p_f = p_i limit of (2)
reproduces it exactly; check in the script). Because σ ∝ 1/p_i ∝ 1/v the rate is velocity-independent:
  R_e = n_e (ρ₀f₂/m) σ_e c (p_f/m_e)(1 + δ/2m_e),  n_e = 54 × 4.587×10²⁷ per tonne of LXe.  (3)
The same ε, α_D, m_A′ fix σ_p (P011), so σ_e/σ_p = μ_e²/μ_p² = 2.97×10⁻⁷. For the propagator, q = p_f = 0.63–0.73 GeV; with
m_A′ ≥ 2.4 GeV the suppression is ≥ 0.87 (not applied; noted).

### 2.3 Exothermic nuclear kinematics (Part D)
E* = |δ|μ/m_N, half-width Δ(v) = μv√(2μ|δ|)/m_N (P058 Eq. 3). For heavy DM μ → m_N so E* → δ(1 − m_N/m_χ): 268 keV (Xe),
290 (Ar), 281 (Ge), 292 (Si), 295 (O), 256 keV (W) at 1 TeV, δ = 300 keV. The spectrum is the WimPyDD O₁ rate with δ < 0.

### 2.4 Re-population (Part E)
χ₁χ₁ → χ₂χ₂ (both A′ vertices off-diagonal in the Majorana basis; χ₁χ₁ → χ₁χ₂ needs a diagonal vertex and is absent at tree
level): μv_rel²/2 ≥ 2δ with μ = m/2 ⇒ v_rel ≥ √(8δ/m). Up-scattering on a nucleus χ₁N → χ₂N: v_min(E) is minimal at
E = μδ/m_N with v_min,opt = √(2δ/μ). Self-scattering cross-section in the contact limit: σ₀ = c²μ²/π with c = g_D²/m_A′²,
μ = m/2 ⇒ σ₀ = 4πα_D²m²/m_A′⁴; inelastic: σ(v) = (σ₀/2)∫dcosθ (p_f/p_i)[m_A′²/(m_A′² + q²)]², q² = p_i² + p_f² − 2p_ip_f cosθ.
Γ_up = n_χ⟨σv_rel⟩ over the relative-speed distribution (Maxwellian with v₀,rel = √2·238 km/s, cut at 2v_esc); steady state
f₂,ss = Γ_up τ (1 − e^{−t_U/τ}). Identical-particle factors are O(1) and omitted (flagged).

## 3. Inputs

| Input | Value | Source |
|---|---|---|
| LZ exposure, WS ROI | 2.84 t·yr; S1c 3–600 phd, log₁₀S2c 2.75–4.15; 1710 events; internal β 1341, ¹³⁶Xe 110 (Table I) | LZ paper; `lz.LZ` |
| ER band vs ROI | ER median leaves log₁₀S2c = 4.15 at S1c ≈ 150 phd (Fig. 4) | LZ Fig. 4 PNG; NEST check §4.1 |
| NEST ER yields | `lz.nest_er_yields(E, params=NEST_ER_LZ)`, g₁ = 0.110, g₂ = 34.5 | lzcommon |
| σ_p(N = 1), 1 TeV | 1.20e-42 / 8.59e-42 / 2.99e-40 / 2.69e-39 / 5.60e-38 cm² at δ = 250/300/350/366/380 keV | P011 (366: P058) |
| P058 nuclear | R_ROI(p) = 859 / 1.81e4 / 1.43e5 / 2.64e6 per 2.84 t·yr; f₂ < 3.1e-3 / 9.7e-5 / 1.1e-5 / 5.7e-7 (p), 2.8e-3 … 1.7e-7 (iso); τ₀(m_A′ = 1 GeV) = 2.78e18 / 3.7e16 / 1.3e14 s at 300/350/380 keV | P058 |
| f₂,fo | 0.50 (dark photon), 0.42 (Higgsino) | P026 |
| α_D (relic) | 0.0073 / 0.0245 / 0.073 at 300 / 1000 / 3000 GeV | P011 |
| Halo | Baxter SHM, v₀ = 238, v_esc = 544, v_E = 250.6 km/s (Sun frame, P035 labelling), ρ₀ = 0.3 GeV cm⁻³ | lzcommon, `wd_halo()` |
| ¹³⁶Xe 2νββ | T½ = 2.165×10²¹ yr (EXO-200), Q = 2457.8 keV, abundance 8.857 %, Primakoff–Rosen shape | recalled: likely / certain / certain / certain |
| LXe ER resolution | σ/E = 0.32/√E(keV) + 0.0015 → 6.0/6.5/6.8 keV at 300/350/380 keV | recalled (XENON1T-type), likely |
| ²¹⁴Pb continuum | allowed β shape, Q = 1019 keV, Z = 83, Fermi function; S(300)/S(20) = 1.22 | recalled, likely |
| Earth | core 0.325 M_E (Fe/Ni), R_E = 6371 km, R_core = 3480 km; bulk Pb 0.2 ppm | recalled: likely / certain / uncertain |
| Exposures (other experiments) | XENONnT 3.1 t·yr (uncertain); PandaX-4T 1.54 (likely); XENON1T 1.0 (certain); DEAP-3600 758 t·d (likely); DarkSide-50 16 660 kg·d (likely); SuperCDMS iZIP 1690 kg·d (likely); EDELWEISS-III 496 kg·d (likely); CRESST-III 5.6 kg·d (likely); COSINE-100 172 kg·yr (uncertain) | recalled |
| Constants | m_e = 0.511 MeV, m_p = 0.9383 GeV, N_A, t_U = 4.35e17 s, G_F | recalled, certain |

## 4. Results

### 4.1 Where the ER line sits in LZ (`P094_ER_in_LZ_space.csv`, `P094_electron_line_kinematics.csv`)
With LZ-tuned NEST: a 300 keV ER gives N_ph = 12 800, N_e = 9520 → **S1c = 1408 phd, log₁₀S2c = 5.52**; 350 keV: 1595 phd, 5.60;
380 keV: 1706 phd, 5.64. Both coordinates lie far outside the WS ROI (600 phd, 4.15). The ER-band median crosses the ROI ceiling
log₁₀S2c = 4.15 at **20 keV_ee** (S1c = 119 phd; Fig. 4 shows the band leaving the ROI at ~150 phd), so the 1710 WS-ROI events are
ER events below ≈ 20–40 keV_ee (≈ 40–80 events per keV_ee) and contain no information on a 300–380 keV line. The line must be
sought in LZ's high-energy ER data (¹³⁶Xe/ER analyses), not in this paper's ROI.

Kinematics (1 TeV, δ = 300 keV): T_e = 299.9998 keV at v = 0 (recoil correction −0.20 eV); at 250/550/810 km/s the electron
energy spans δ ± 0.53 / 1.16 / 1.70 keV, exactly p_f v (p_f = 629.8 keV), mean − δ = +0.26 / +2.0 / +4.6 eV. The halo (Sun frame)
has ⟨v⟩ = 354, v_rms = 379 km/s → intrinsic Doppler σ = 0.46 / 0.51 / 0.53 keV at δ = 300 / 350 / 380 keV, versus detector
σ_E = 6.0 / 6.5 / 6.8 keV: the line is resolution-limited (FWHM ≈ 14–16 keV). 400 GeV and 4 TeV change these at the 10⁻³ level.

### 4.2 Electron-line rate versus the nuclear channel (`P094_electron_rate.csv`)

| δ [keV] | σ_e = σ_p μ_e²/μ_p² [cm²] | σv/(σ_e c) = (p_f/m_e)(1+δ/2m_e) | R_e per t·yr (f₂ = 1) | R_e per f₂σ_e [t⁻¹yr⁻¹cm⁻²] | R_nuc,ROI per t·yr (P058) | R_nuc/R_e |
|---|---|---|---|---|---|---|
| 250 | 3.57e-49 | 1.37 | 3.4e-5 | 9.7e43 | — | — |
| 300 | 2.55e-48 | 1.59 | 2.9e-4 | 1.12e44 | 302 | **1.06e6** |
| 350 | 8.89e-47 | 1.82 | 1.1e-2 | 1.28e44 | 6.4e3 | 5.6e5 |
| 366 | 7.99e-46 | 1.89 | 0.106 | 1.33e44 | 5.0e4 | 4.7e5 |
| 380 | 1.66e-44 | 1.96 | 2.29 | 1.38e44 | 9.3e5 | 4.1e5 |

At the P011 coupling the ER line is 4×10⁵–10⁶ times weaker per unit f₂ than the nuclear down-scatter: the electron loses Z²(μ_N/μ_p)²/Z
≈ 2×10¹² in coherence and mass, wins ≈ 700 in final-state velocity (p_f/m_e ≈ 1.2 vs √(2δ/μ_N) = 2.4×10⁻³) and ≈ 3×10³ in the
nuclear form factor (F² ≈ 3×10⁻⁴ averaged over the Xe window). Even f₂ = 0.5 gives 1.4×10⁻⁴ (δ = 300) to 1.1 (380 keV) line events
per t·yr.

### 4.3 LZ high-energy ER background and line sensitivity (ESTIMATE; `P094_ER_line_sensitivity.csv`)
2νββ: 1.30×10⁵ decays per t·yr; dN/dE = 51 / 61 / 67 per (t·yr·keV) at 300 / 350 / 380 keV. Check: the same spectrum gives
70 (1.5–20 keV) to 288 (1.5–40 keV) events in 2.84 t·yr versus LZ's 110 ¹³⁶Xe events in the ROI, i.e. an effective ER window of
≈ 28 keV, consistent with §4.1. ²¹⁴Pb: LZ's 1341 internal-β events over W_eff = 25 (15–40) keV_ee give 18.9 (12–31) per (t·yr·keV)
at 20 keV and ×1.22 at 300 keV → 23 (14–38). Total B = 74 (66–89) / 84 (75–98) / 89 (81–103) per (t·yr·keV); detector-material
Compton continuum not modelled (would only weaken the estimate). Line search in ±1.5σ_E (86.6 % containment): B_win = 3.8–5.2×10³
events in 2.84 t·yr, S₉₀ = 1.64√B_win/0.866 = 117 / 129 / 136 events ⇒
  **f₂σ_e < 3.7×10⁻⁴³ cm²** (3.4–4.0) at δ = 300 keV; 3.5×10⁻⁴³ at 350; 3.5×10⁻⁴³ at 380 (LZ 2.84 t·yr);
  6×10⁻⁴³ for 1 t·yr (XENON1T/PandaX-4T class), 1.3×10⁻⁴³ for 20 t·yr.
In the dark-photon model (σ_e = 2.55×10⁻⁴⁸ cm² at 300 keV) this is f₂ < 1.4×10⁵ (300), 4×10³ (350), 21 (380 keV): **no constraint**,
10⁷–10⁸ times weaker than P058. The ER line is nevertheless the *only* direct-detection handle on the DM–electron coupling of TeV
DM (elastic χe recoils are ≤ 2m_ev² ≈ 2 eV), so f₂σ_e ≲ 4×10⁻⁴³ cm² is a new, if weak, model-independent statement; for a
leptophilic χ₂ (no nuclear channel) it is the only one.

Other detectors (recalled, reliability in brackets): XENON1T/XENONnT double-weak-decay analyses fit ER spectra to ≳ 2.5 MeV
[likely]; PandaX-4T 2νββ/0νββ ER spectra [likely]; XENONnT and LZ low-energy ER searches stop at 140 keV_ee and ~40 keV_ee
[certain]; EXO-200 and KamLAND-Zen thresholds ≈ 1 MeV / 0.5 MeV [likely] do not reach 300–380 keV. Any of the LXe TPC spectra
would give the "1 t·yr" row above to within a factor 2 (Rn-dominated backgrounds differ by ×2–3); LZ's own high-energy data
(10⁴ events per keV-window) would give the first row. None changes the model conclusion.

### 4.4 Exothermic nuclear recoils on light targets (`P094_target_spectra_summary.csv`, `P094_target_spectra.csv`, Fig. 1)
WimPyDD O₁, proton-only (dark photon), 1 TeV, Sun-frame halo, σ_p(N = 1), f₂ = 1, no efficiency. Validation: Xe total 339 per
t·yr at δ = 300 keV versus P058's ROI-accepted 302 (ratio 0.89 = ROI fraction × efficiency); Xe median 162 keV (P058: 164).

| target | A | E* [keV] | Δ(250 km/s) [keV] | rate per kg·yr (300) | p16/p50/p84 [keV] | frac < 100 keV | rate/Xe per kg: 300 / 350 / 380 |
|---|---|---|---|---|---|---|---|
| Xe | 131 | 268 | 189 | 0.339 | 71/162/214 | 0.26 | 1 / 1 / 1 |
| Ar | 38 | 290 | 115 | 7.08 | 180/245/334 | 2×10⁻⁴ | **20.9 / 24.0 / 24.5** |
| Ge | 73 | 281 | 153 | 1.60 | 111/147/199 | 0.065 | 4.7 / 3.6 / 3.0 |
| Si | 29 | 292 | 102 | 8.5 | 210/280/373 | 0 | 25.1 / 33.5 / 37.5 |
| O | 17 | 295 | 79 | 5.41 | 239/301/378 | 0 | 16.0 / 24.4 / 29.7 |
| Ca | 43 | 288 | 122 | 8.18 | 179/242/330 | 2×10⁻⁴ | 24.1 / 27.4 / 27.6 |
| W | 183 | 256 | 211 | 0.405 | 81/104/204 | 0.39 | 1.2 / 1.2 / 1.1 |
| Na | 23 | 294 | 92 | 7.74 | 223/292/381 | 0 | 22.9 / 32.6 / 37.9 |
| I | 127 | 268 | 188 | 0.32 | 69/161/217 | 0.34 | 0.95 / 0.88 / 0.87 |
| CaWO₄ | — | — | — | 2.60 | 182/265/355 | — | 7.7 / 10.0 / 11.1 |
| NaI | — | — | — | 1.46 | 186/272/371 | — | 4.3 / 5.7 / 6.6 |

Isoscalar coupling (δ = 300): Xe 1.45, Ar 31.9, Ge 6.99, Si 34.0 per kg·yr → Ar/Xe = 22, Si/Xe = 23, Ge/Xe = 4.8, i.e. the same
hierarchy. **No exothermic recoil falls below 20 keV on any target** (fraction < 20 keV = 0 for all), and on Si/O/Na/Ar < 0.1 % lie
below 100 keV: the down-scatter spectrum sits at 100–450 keV, far above every low-threshold ROI (CRESST-III 0.03–16 keV,
SuperCDMS HV/CDMSlite ≤ 25 keV, EDELWEISS ≤ 100 keV, DarkSide-50 ≤ 200 keV_nr). Light nuclei win per kilogram because their form
factor is still ≈ 0.3 where xenon's has collapsed (nodes at 102 and 266 keV), overcompensating Z²A^{1/2}: rate ∝ Z²A^{1/2}F²_eff.

Per experiment (`P094_experiments.csv`; whole spectrum, no efficiency, δ = 300 keV, events per unit f₂ at σ_p(N = 1)): LZ 962
(→ f₂ < 2.4×10⁻³ for zero events; P058's full treatment: 3.1×10⁻³), XENONnT 3.1 t·yr 1.05×10³, PandaX-4T 521, XENON1T 339,
**DEAP-3600 (2.08 t·yr Ar) 1.47×10⁴** → f₂ < 1.6×10⁻⁴ if a 100–500 keV_nr NR search were empty (δ = 350: 3.8×10⁵ → 6×10⁻⁶;
380: 5.9×10⁷ → 3.9×10⁻⁸, vs LZ 3.3×10⁶); DarkSide-50 323; COSINE-100 251; SuperCDMS iZIP 7.4 (f₂ < 0.31); EDELWEISS 2.2;
CRESST-III 0.04 (f₂ < 58). Only tonne-scale argon competes with xenon, and it would beat LZ's f₂ limit by ≈ 15× per δ *if* the
LAr experiments analyse 100–500 keV nuclear recoils (³⁹Ar β at ~1 Bq/kg must be removed by pulse-shape discrimination, which is
strongest at high energy) — a projection, not a limit.

### 4.5 Re-population of χ₂ today (`P094_selfscatter_thresholds.csv`, `P094_repopulation.csv`, `P094_self_upscatter.csv`)
*Thresholds.* χ₁χ₁ → χ₂χ₂: v_rel ≥ √(8δ/m) = 464 / 502 / 523 km/s at 1 TeV (δ = 300/350/380), 734–827 km/s at 400 GeV,
232–261 km/s at 4 TeV; halo v_rel reaches 2×810 km/s. The assignment's "v_rel > 2√(δ/m) = 3.5×10⁻² c" used δ/m = 3×10⁻⁴; the
correct δ/m = 3×10⁻⁷ gives 328 km/s for a single excitation (forbidden at tree level) and 464 km/s for the double one: **halo
self-up-scattering is kinematically open**; 28 / 22 / 19 % of pairs exceed the threshold (⟨v_rel⟩ = 380 km/s).
*Nuclear up-scattering.* v_min,opt = √(2δ/μ) = 7612 (H), 3811 (He), 1916 (O), 1457 (Si), 1043 (Fe), 704 (Xe), 576 km/s (Pb) at
1 TeV, δ = 300 keV (check: `lz.vmin_kms` at E* gives 703 km/s for Xe). Hence: Earth core (Fe/Ni) — kinematically forbidden
(v_max = 810 km/s; WimPyDD returns exactly 0); Earth Pb (3.5×10⁴² atoms, per-atom rate bounded by 30× xenon's 0.377 per
t·yr-equivalent) — P_conv ≤ 2×10⁻¹⁴ per crossing, induced f₂ ≤ 10⁻¹⁴; Sun — H/He forbidden even at the core (v ≈ 1400 km/s <
1900 km/s for He), Fe possible but P(cross the Sun in t_U) = 1.9×10⁻²¹ → f₂ ~ 4×10⁻²⁶; ISM heavy nuclei (n ~ 10⁻⁹ cm⁻³, uncertain)
with ⟨σv⟩_up(Xe) = 8.7×10⁻³³ cm³ s⁻¹ → Γt_U = 4×10⁻²⁴; cosmic rays Φσ_p t_U = 4×10⁻²⁴ (300) to 2×10⁻²⁰ (380 keV). All negligible.
*Halo self-up-scattering (dark photon, α_D = 0.0245 from the relic density, 1 TeV):* σ₀ = 4πα_D²m²/m_A′⁴ = 8.9×10⁻²⁶ cm² at
m_A′ = 2.4 GeV (σ/m = 5×10⁻⁵ cm² g⁻¹, harmless), Γ_up = 1.6×10⁻²² s⁻¹; with τ = 2.78×10¹⁸(m_A′/GeV)⁻⁴ s:

| m_A′ [GeV] | τ [s] | Γ_up [s⁻¹] | f₂,ss = Γ_upτ | f₂,primordial = 0.5e^{−t_U/τ} | LZ exothermic events (2.84 t·yr): ss / primordial |
|---|---|---|---|---|---|
| 2.4 | 8.4e16 | 1.6e-22 | 1.3e-5 | 2.8e-3 | 0.011 / 2.4 |
| 2.6 | 6.1e16 | 1.2e-22 | 7.3e-6 | 3.9e-4 | 0.006 / 0.33 |
| 2.9 | 3.9e16 | 8.2e-23 | 3.2e-6 | 7.7e-6 | 0.0028 / 0.0066 |
| 3.0 | 3.4e16 | 7.3e-23 | 2.5e-6 | 1.5e-6 | 0.0022 / 0.0013 |
| 3.5 | 1.9e16 | 4.2e-23 | 7.9e-7 | 3e-11 | 6.7e-4 / 0 |
| 5.0 | 4.5e15 | 1.1e-23 | 5.1e-8 | 0 | 4.3e-5 / 0 |
| 10 | 2.8e14 | 7.8e-25 | 2.2e-10 | 0 | 1.9e-7 / 0 |

f₂,ss ∝ α_D²m_A′⁻⁸; it exceeds the primordial remnant for **m_A′ > 2.97 GeV** and never decays away. At δ = 350 / 380 keV
(τ₀ smaller) f₂,ss = 1.3×10⁻⁷ / 3.8×10⁻¹⁰ at m_A′ = 2.4 GeV. For the Higgsino (Z exchange, σ ≈ 2×10⁻³³ cm², τ_νν̄ = 1.2×10⁶ s)
f₂,ss = 5×10⁻²⁴. With α_D = 0.1 all f₂,ss scale by (0.1/0.0245)² = 17.

### 4.6 Combined (τ, f₂) map (`P094_f2_tau_map.csv`, Fig. 2)
f₂(today) = 0.5e^{−t_U/τ} ⇒ τ = 8.4×10¹⁶ s (f₂ = 2.8×10⁻³), 7.0×10¹⁶ (10⁻³), 5.1×10¹⁶ (10⁻⁴), 4.0×10¹⁶ (10⁻⁵), 3.3×10¹⁶ (10⁻⁶);
dark-photon m_A′ (300 keV) = 2.40, 2.51, 2.72, 2.88, 3.03 GeV. A populated excited state with f₂ between 10⁻⁵ and 3×10⁻³ therefore
exists only for τ within a factor 2.1 of 6×10¹⁶ s (m_A′ = 2.4–2.9 GeV), P066's γ-line bound (f₂/τ_γ < 5×10⁻²³ s⁻¹) is irrelevant
for the invisible νν̄ channel and bites only at τ_γ > 10²⁰ s for f₂ ≳ 10⁻², and P026's Planck window concerns the photon channel
only. Below f₂ ≈ 10⁻⁵ the surviving population is the self-scattering steady state (§4.5), an irreducible 10⁻⁵(m_A′/2.4 GeV)⁻⁸,
which predicts 0.011(m_A′/2.4 GeV)⁻⁸ exothermic events in LZ's 2.84 t·yr and ≈ 1 event per 200 t·yr at m_A′ = 2.4 GeV.

## 5. Validation and robustness
- Electron kinematics: exact two-body formula reproduces δ at v = 0 to 0.2 eV (χ₁ recoil p_f²/2m) and the Doppler term p_f v to
  0.01 % at 810 km/s; results identical at 400 GeV and 4 TeV to 3 significant figures.
- Eq. (2) reduces to σ_e = c²μ_e²/π at T → 0 (algebraic check in §2.2); the relativistic factor (1 + δ/2m_e)(p_f/m_e) = 1.59 at
  300 keV is the only difference from the naive σ_e√(2δ/m_e).
- WimPyDD: Xe total exothermic 339 vs P058 ROI 302 per t·yr (0.89, expected ROI × efficiency); median 162 vs 164 keV; endothermic Xe
  0.377 per t·yr at σ_p(N = 1) vs 1/2.84 = 0.352 ROI events (Sun-frame vs annual halo, ROI acceptance). W and Ca required the
  P015/P046/P065 runtime patch (`np` injected into the 18xW/4xCa response modules; no WimPyDD file edited); W's spin response is
  WimPyDD's placeholder but O₁ (M response) is Helm-based and adequate.
- 2νββ spectrum: reproduces LZ's 110 ROI events with W_eff ≈ 28 keV, matching the NEST estimate of the ER window (20–40 keV).
- Background range (66–103 per t·yr·keV) moves the f₂σ_e limit by ±10 %; the unmodelled Compton continuum can only weaken it.
- Halo: Sun-frame; exothermic rates vary ±4 % over the year (P058); the self-scattering Γ_up uses the Galactic-frame relative-speed
  distribution (untruncated Maxwellian cut at 2v_esc; truncation changes the above-threshold fraction by < 10 %).
- Self-scattering normalisation: identical-particle and spin factors O(1), propagator included (q² ≈ p_i² + p_f² ≈ 0.6 GeV², a
  0.8 factor at m_A′ = 2.4 GeV); α_D from P011's relic relation (factor ~2 uncertain) enters squared.

## 6. Failed or abandoned approaches
- `lz.dRdE_SI` with δ < 0 cannot be used for light targets (P058's `vmin_kms` sign pitfall); WimPyDD with δ < 0 used throughout.
- WimPyDD Ca and W targets crashed with `NameError: np` (42Ca/180W response files); fixed at runtime as in P065.
- First threshold formula v_min,opt = √(δ/2μ) was wrong by 2 (WimPyDD Fe up-scattering returned exactly 0 and exposed it); corrected
  to √(2δ/μ) and checked against `lz.vmin_kms`.
- Using LZ's WS ROI counts for the ER line: impossible, the line lies at S1c ≈ 1400–1700 phd, log₁₀S2c ≈ 5.5–5.6.

## 7. Discussion
For a dark-photon pseudo-Dirac interpretation of the LZ event the *nuclear* down-scatter is the only relevant exothermic signal:
the ER line is a million times weaker, low-threshold experiments never see the 100–450 keV recoils, and the sole competitor is
tonne-scale argon, where the intact form factor gives 21–25× xenon's rate per kilogram. The population dynamics are simple: no
Galactic up-scattering on nuclei is possible (Fe forbidden, Pb ≤ 10⁻¹⁴), but χ₁χ₁ → χ₂χ₂ in the halo is open and maintains
f₂ ≈ 10⁻⁵(m_A′/2.4 GeV)⁻⁸ — below every present bound, above the primordial remnant for m_A′ > 3 GeV, and an eventual target for
≳ 100 t·yr xenon or argon exposures. A metastable χ₂ with f₂ ≳ 10⁻⁵ survives only in the narrow lifetime band (4–8)×10¹⁶ s.

## 8. Figures
- `figures/P094_fig1_spectra_and_line.png` — Left: exothermic dR/dE per kg·yr per unit f₂ on Xe, Ar, Ge, Si, O, W (δ = 300 keV,
  1 TeV, σ_p(N = 1)); shaded: LZ WS ROI and the < 16 keV low-threshold regime. Right: the χ₂e → χ₁e deposited-energy line at
  δ = 300/350/380 keV, intrinsic Doppler width (dotted, σ ≈ 0.5 keV) versus LXe resolution (σ_E = 6–7 keV).
- `figures/P094_fig2_f2_tau_map.png` — (τ, f₂) plane: P026 freeze-out curves, P058 nuclear limits (300/350/380 keV), P066 SPI
  line (γ channel), P026 Planck window (γ channel), the Ar 2.1 t·yr projection, and the halo self-up-scattering steady state.

## 9. References
1. LZ Collaboration, arXiv:2609.02823 (2026). 2. D. Tucker-Smith, N. Weiner, PRD 64, 043502 (2001). 3. B. Batell, M. Pospelov,
A. Ritz, PRD 79, 115019 (2009). 4. P. W. Graham, D. E. Kaplan, S. Rajendran, M. T. Walters, PRD 82, 063512 (2010). 5. R. Essig,
J. Mardon, T. Volansky, PRD 85, 076007 (2012) (DM–electron scattering formalism). 6. J. B. Albert et al. (EXO-200), PRC 89, 015502
(2014) (¹³⁶Xe 2νββ half-life). 7. I. Jeong, S. Kang, S. Scopel, G. Tomar, CPC 276, 108342 (2022) (WimPyDD). 8. Corpus: P011, P026,
P058, P065, P066, P035, dossier 00.

## 10. Tools and provenance (mirrors provenance/P094.json)
Agent tools: Read (PAPER_GUIDE; P058/P026/P011/P065/P066 papers; P058 details; lzcommon; tex lines 148–159; Fig. 4 PNG; own figures ×4),
Bash (file listings; tex/dossier/ledger greps; WimPyDD target probes ×3; P065 patch inspection ×2; script runs ×6; table prints; word counts),
Write (script, details, provenance, paper), Edit (script ×9), Skill (dataviz).
Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (stats.norm, special); pandas 3.0.5; matplotlib 3.11.2; nestpy 2.1.1 via
`lz.nest_er_yields`; WimPyDD 2.0.4 (`diff_rate` with δ < 0 on Xe/Ar/Ge/Si/O/Ca/W/Na/I/Fe via `lz.wd_rate`, `wd_halo`, `wd_hamiltonian`);
common/lzcommon.py. Recalled values: 19 (listed in the JSON). WimPyDD-generated files: none outside `output/` (spectra cached in
`output/work/P094/cache/`, 31 npy files).
