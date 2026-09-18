# P096 — research record

**Title.** Large liquid-scintillator detectors as high-energy recoil detectors: can Borexino, KamLAND, JUNO or SNO+ see the LZ models through ¹²C, ¹³C and ¹H recoils?
**Simulated date.** 2026-09-16. **Category.** COMP (hep-ex, cross-list hep-ph). **Author profile.** Large-liquid-scintillator (neutrino-detector) group.
**Script.** `output/code/P096_scintillators.py` (run from the simulation root with `.venv/bin/python`; 66 s cold, 1 s from the kernel cache `P096_cache.npz`). All numbers below are printed in `run_log.txt` or stored in the CSV/JSON files of this directory.

## 1. Motivation and framework

Kiloton organic scintillators hold 10³–10⁴ times LZ's target mass but are made of the lightest nuclei (¹H, ¹²C, with 1.1 % ¹³C; ¹⁶O in water phases). The LZ event (248 keV Xe recoil, 2.84 t·yr, arXiv:2609.02823) is read by the corpus as (A) inelastic isoscalar O₁ with δ = 300–380 keV at ~1 TeV (P021 peak δ = 380 keV; Higgsino window 358–385 keV), (B) an elastic q⁴-suppressed spin interaction (LZ's L10 best fit, P003/P012; O6), or (C) — for reference only — elastic SI, which LZ's own 2024 search excludes as the explanation. P046 and P067 showed that heavy targets (I, W) are needed for (A) and gave the kinematic ceilings; P058 showed that a surviving excited state χ₂ down-scatters (δ < 0) with *no* velocity threshold and constrained its fraction f₂. We ask what each of these predicts in Borexino (278 t PC), KamLAND (~1 kt), SNO+ (780 t LAB; also its 905 t water phase) and JUNO (20 kt LAB), after quenching, against threshold and ¹⁴C.

Detector inputs (all recalled; reliability in §9): masses 278 / 1000 / 780 / 20 000 t; PC = C₉H₁₂ (mass fractions C 0.899, H 0.101), LAB ≈ C₁₈H₃₀ (C 0.877, H 0.123), KamLAND 80 % dodecane + 20 % PC (C 0.857, H 0.143), water (H 0.112, O 0.888); thresholds 50 / 200 / 200 / 100 keV_ee (Borexino hardware; KamLAND/SNO+ physics; JUNO trigger — all *uncertain*); ¹⁴C/¹²C = 2.7 × 10⁻¹⁸ (Borexino, likely) and 10⁻¹⁷ (others, uncertain).

## 2. Kinematics (`P096_kinematics.csv`; Fig. 1 left)

Halo: Baxter-2021 SHM through `lzcommon` (v₀ = 238, v_esc = 544 km/s); v_E(16 June, day 167) = 265.1 km/s, v_max = 809.1 km/s. Formulae (`lz.E_R_range_keV`, `lz.mu_red`): inelastic ceiling δ_ceil = μv²_max/2; elastic E_R,max = 2μ²v²_max/m_N; minimum speed for any endothermic scatter v_min = √(2δ/μ).

| nucleus (1 TeV) | δ_ceil [keV] | E_R,max elastic [keV] | v_min(δ = 300) [km/s] | v_min(380) |
|---|---|---|---|---|
| ¹H | 3.4 | 13.5 | 7612 | 8567 |
| ¹²C | 40.3 | 159.3 | 2209 | 2486 |
| ¹³C | 43.6 | 172.2 | 2123 | 2389 |
| ¹⁶O | 53.5 | 210.8 | 1916 | 2157 |
| ¹⁹F | 63.3 | 248.9 | 1761 | — |
| Xe (131.3) | 397.1 | 1415 | 703 | 791 |
| ¹⁸⁴W | 532.9 | 1820 | 607 | — |

A_min(δ) (smallest integer A with δ_ceil ≥ δ, m_N = A·u): δ = 300/350/366/380 keV → **97/115/120/126** at 1 TeV, 112/136/145/152 at 400 GeV, 91/106/111/116 at 4 TeV. P067 quoted 96/114/120/125 (they used a slightly different v_E; ±1 unit). Every scintillator nucleus sits 2.4–9.4× below the speed needed for δ = 300 keV: interpretation (A) is kinematically forbidden on H, C and O (and on F, Ca, S — P067). Xe ceilings reproduce P067 (F 63, I 385, Xe 397, W 533 keV).

## 3. Elastic q⁴-spin models (B) on ¹H and ¹³C

### 3.1 Hamiltonians and normalisation to LZ's fit
WimPyDD convention c⁰ = c_p + c_n (P003): unit isoscalar O₁ ↔ c⁰ = 2/m_v². L10 (P012): 4[(q²/m_N²)O₄ − O₆] with d10 = 1, m_M = m_N ↔ c₄ = 8q²/(m_v²m_N²), c₆ = −8/m_v²; O₆ alone with c⁰₆ = 2/m_v². Rather than carry the factor-2 convention of P012 (DR-001), we normalise each model to **1.0 accepted event in LZ** (2.84 t·yr, efficiency 0.96·Φ((E−5.4)/2.5)·[1−Φ((E−269.9)/11.5)], June halo, natural Xe, WimPyDD): N_unit(L10) = 3.405 (200–270 keV: 1.207), N_unit(O6) = 0.536 → scale factors 0.2937 and 1.867. Sun-frame check: N_unit(L10) = 3.331 vs P012's 3.34.

### 3.2 Hydrogen (free proton)
WimPyDD has 1H response functions (`itar = 21`); the rate on natural hydrogen (0.9998 ¹H) with the June halo is computed directly (`lz.wd_rate(..., target=WD.H)`). Free-proton analytic check: in the Anand et al. normalisation dσ/dE_R = (2m_T/4πv²)·|M|²_avg with |M|²_avg the spin-averaged squared amplitude; for a spin-½ nucleon ⟨S_iS_j⟩ = δ_ij/4, so
O₆ = (S_χ·q)(S_N·q)/m_N²: |M|²_avg = c_p² q⁴/(16 m_N⁴); L10 = 4(c/m_N²)(q×S_χ)·(q×S_N): |M|²_avg = 16c² (q⁴/m_N⁴)·(2/16) = 2c² q⁴/m_N⁴ (two transverse directions), c = d10/m_v².
Equivalently, the free-proton Σ″ and Σ′ responses are the spin averages ⟨(S_N·q̂)²⟩ = ¼ and ⟨|q̂×S_N|²⟩ = ½ — *not* 1/(4π); the 4π is already in the prefactor. dR/dE = n_T (ρ/m_χ)(2m_T/4π)|M|²_avg η(v_min) with η from `lz.eta0` (v_E = 265.1 km/s), converted with 1 GeV⁻² = 3.894 × 10⁻²⁸ cm². Result: analytic/WimPyDD = 0.940 ± 0.072 (O6), 0.969 ± 0.075 (L10) over 0.5–13 keV; WimPyDD's L10/O6 ratio at 4 keV is 31.0 (analytic 32). The 3–6 % offset is WimPyDD's m(¹H) = 0.931 GeV vs m_N = 0.938 GeV in q⁴/m_N⁴ plus grid effects — the WimPyDD proton spin normalisation is confirmed.

**Rates at LZ's fit (per tonne of hydrogen per year):** L10 1.12 × 10⁻⁷, O6 2.30 × 10⁻⁸; median E_R = 4.6 keV, E_R,max = 13.5 keV. Fig. 2.

### 3.3 ¹³C (J = ½, 1.07 % abundance)
WimPyDD's ¹³C entry has `itar = 0`, i.e. all nuclear response functions vanish (documented in `package.py`), and the element `WD.C` therefore gives exactly zero for L10/O6 (¹²C is spin-0). We estimate ¹³C analytically with the free-nucleon formula of §3.2 times (i) the spin factor (2⟨S_n⟩)² relative to a free proton — for isoscalar couplings and J = ½ in both cases, the (J+1)/J·(a_p⟨S_p⟩+a_n⟨S_n⟩)² scaling gives (⟨S_n⟩/½)²; single-particle p₁/₂ neutron ⟨S_n⟩ = −j/[2(j+1)] = −1/6 (shell-model literature ≈ −0.17), so the factor is **1/9**; (ii) a 1p harmonic-oscillator spin form factor F(q) = (1 − q²b²/6)e^{−q²b²/4}, b = 1.64 fm (F² = 0.785 at E_R,max = 172 keV); (iii) ¹³C kinematics (m = 13u, E_R up to 172 keV, so q is 13× larger than on H at equal E_R and the q⁴ penalty is 13⁴ smaller). Mass fraction of ¹³C in natural carbon: 0.0116.
**Result:** L10 3.95 × 10⁻³ events per tonne of ¹³C per year (median E_R 57 keV; O6 7.85 × 10⁻⁴), i.e. 4.6 × 10⁻⁵ per tonne of natural carbon — 350× the hydrogen rate per tonne of LAB. JUNO: **0.80 (L10) / 0.16 (O6) ¹³C recoils per year** and 2.8 × 10⁻⁴ / 5.7 × 10⁻⁵ proton recoils; Borexino 0.011 / 0.0023 and 3 × 10⁻⁶; SNO+ 0.031; KamLAND 0.039 per year. Uncertainty: factor ≈ 2 from ⟨S_n⟩ and the form factor.

### 3.4 SI reference (C)
O₁ with σ_n = 10⁻⁴⁷ cm² (≈ LZ's 2024 limit at 1 TeV, recalled/likely), WimPyDD on C, H, O (June): 3.33 × 10⁻³, 2.0 × 10⁻⁶, 7.75 × 10⁻³ events per tonne of element per year; median E_R 17 (C), 1.6 (H), 22 keV (O). JUNO 58 carbon recoils/yr, SNO+ water 6.2 oxygen recoils/yr, Borexino 0.83/yr — at 1–6 keV_ee after quenching.

## 4. Exothermic down-scattering (D) of a surviving χ₂

### 4.1 Method
Following P058: WimPyDD O₁ isoscalar kernels K(E, v_i) = `diff_rate(..., sum_over_streams=False)` on the 1200-stream grid 0–844 km/s, rates on any day as K·δη(day) (`lz.wd_halo(day_of_year, vmin=grid)`), 24-day annual mean. Endothermic Xe (δ = +300/350/366/380 keV, 100–450 keV, 2.5 keV steps) fixes the coupling from the LZ event: κ = 1/N_endo,ROI(unit) (μ_endo = 1, LZ acceptance as §3.1). Exothermic kernels (δ < 0) on C (2.5–800 keV, 5 keV), H (150–450, 2 keV), O (2.5–850, 5 keV); Xe at δ = −300 for validation. WimPyDD carbon = 0.989 ¹²C (the ¹³C M response is zero with `itar = 0`; a +1.3 % correction, ignored); oxygen = 0.9976 ¹⁶O.

**Validation.** N_endo,ROI(unit, annual) = 1.338 × 10⁴ / 255.6 / 20.9 / 0.663 at δ = 300/350/366/380 keV (P021: 1.35 × 10⁴ / 258.7 / – / 0.456 — the 380 keV tail depends on the day sampling: June 2.71, December 1.5 × 10⁻⁵); κ = 7.47 × 10⁻⁵ / 3.91 × 10⁻³ / 0.0479 / 1.51. Xe exothermic ROI count per unit coupling 1.24 × 10⁷ → R_ROI = 926 (P058: 916, 1 %). Exothermic rates vary by < 0.1 % between June and the annual mean (P058: ±4 % in the ROI window; our totals are window-free).

### 4.2 Spectra (`P096_exothermic_spectra.csv`; Fig. 1 right)
E* = |δ|μ/m_N = 297/346/362/376 keV on ¹²C, 300/350/366/380 keV on ¹H, 296–374 keV on ¹⁶O. Because μ ≈ m_N for light nuclei the window half-width μv√(2μ|δ|)/m_N is small — the spectrum is a **quasi-line**: 16/50/84 % quantiles 249/305/371 keV (C, δ = 300), 283/301/319 keV (H), 240/303/380 keV (O); E₊(v_max) = 608 (C), 371 (H), 672 keV (O). On xenon the same process is a broad 73–206 keV distribution (P058) because there μ v√(2μδ)/m_N ≈ 200 keV and F² is falling. The peak position measures δ directly: E_peak = 292/338/358/368 keV (C).
Unit-coupling totals (annual): C 4.34/4.42/4.43/4.44 × 10⁷, H 1.25/1.35/1.38/1.40 × 10⁵, O 7.46/7.37/7.33/7.28 × 10⁷ events per tonne of element per year (δ = 300/350/366/380). Check of the C/H ratio 347: for |δ| ≫ μv², σv → σ₀F²√(2|δ|/μ), so per tonne the ratio is A·(μ_C/μ_p)^{3/2}·F² ≈ 12 × 40.5 × 0.7 ≈ 340. ✓
Times κ (f₂ = 1): C 3.24 × 10³ / 1.73 × 10⁵ / 2.12 × 10⁶ / 6.70 × 10⁷; H 9.35 / 527 / 6.59 × 10³ / 2.12 × 10⁵ per tonne of element per year.

### 4.3 Events at P058's f₂ limits (isoscalar, joint 90 % CL: 2.8 × 10⁻³ / 5.7 × 10⁻⁵ / 4.8 × 10⁻⁶ / 1.7 × 10⁻⁷)
| detector | ¹²C recoils / yr (δ = 300/350/366/380) | ¹H recoils / yr |
|---|---|---|
| Borexino (250 t C) | 2.3 / 2.5 / 2.5 / 2.9 × 10³ | 0.73 / 0.84 / 0.89 / 1.0 |
| KamLAND (857 t C) | 7.8 / 8.4 / 8.7 / 9.8 × 10³ | 3.8 / 4.3 / 4.5 / 5.2 |
| SNO+ (684 t C) | 6.2 / 6.7 / 7.0 / 7.8 × 10³ | 2.5 / 2.9 / 3.0 / 3.4 |
| JUNO (17 550 t C) | 1.59 / 1.73 / 1.78 / 2.00 × 10⁵ | 64 / 74 / 78 / 88 |
| SNO+ water (804 t O) | ¹⁶O: 1.25 / 1.32 / 1.35 / 1.50 × 10⁴ | 2.7–3.6 |
So a χ₂ population *just allowed* by LZ would give 10⁵ carbon recoils per year in JUNO — the only LZ-motivated signal to which scintillators are kinematically open.

## 5. Detector response

### 5.1 Quenching (recalled inputs; §9)
Birks: dL/dE = 1/(1 + kB·ρ·S_e), kB = 0.0098 cm/MeV (LAB), ρ = 0.86 g/cm³; proton electronic stopping from a PSTAR-like table (170, 270, 430, 640, 810, 790, 680, 580, 420, 262, 116, 45.7 MeV cm²/g at 1, 3, 10, 30, 80, 100, 200, 300, 500, 1000, 3000, 10⁴ keV), log-log interpolated; Q(E) = L(E)/E by integration. Q_p = 0.34 (4.6 keV; electronic only — nuclear stopping lowers it further), 0.26 (14 keV), 0.16 (100 keV), 0.153 (300 keV), 0.22 (1 MeV). Cross-check with the Cecil–Anderson–Madey NE-213 form L = 0.83E − 2.82[1 − e^{−0.25E^0.93}] MeVee: 0.093 (300 keV), 0.136 (500), 0.206 (1 MeV) — we adopt Birks and quote Cecil as the low side. Carbon ions: LSS electronic stopping S_e ≈ 2500 (E/300 keV)^{1/2} MeV cm²/g with an electronic fraction 0.6 → Q_C = 0.096 (57 keV), 0.048 (300 keV), 0.044 (380 keV); adopted band ×0.6–1.7 (0.029–0.082 at 300 keV), consistent with the "few per cent" literature values.
Quenched signals: L10 on ¹H 0.9–2.5 keV_ee (E_R 1–8 keV); L10 on ¹³C 3–9 keV_ee (E_R 30–90 keV); SI on C 0.7–6 keV_ee; **exothermic ¹²C line 12–18 keV_ee (δ = 300) to 14–20 keV_ee (380), band 7–33 keV_ee for the Q_C uncertainty; exothermic ¹H line 44–49 keV_ee (300) to 58–64 keV_ee (380)** (Cecil: 28–35 keV_ee). Water: a 300 keV oxygen ion or proton is far below the Cherenkov threshold (β < 1/n; protons need ≳ 480 MeV) — SNO+ water phase is blind to all of it.

### 5.2 ¹⁴C
Allowed β spectrum, Q = 156.5 keV, Fermi function (Z = 7); half-life 5730 yr → λ = 3.83 × 10⁻¹² s⁻¹; 5.01 × 10²⁸ carbon atoms per tonne. Rates: JUNO (17 550 t C, 10⁻¹⁷) 1.06 × 10¹² decays/yr (3.4 × 10⁴ Bq); Borexino (250 t, 2.7 × 10⁻¹⁸) 4.1 × 10⁹/yr (130 Bq, consistent with the ~40 Bq/100 t recalled); SNO+ 4.1 × 10¹⁰; KamLAND 5.2 × 10¹⁰. Fractions of decays in windows: 5–15 keV_ee 0.110, 10–30 keV_ee 0.230, 20–40 keV_ee 0.229, above 50 keV_ee 0.447. In the ¹²C exothermic window (16–84 % quantiles ×Q_C): JUNO 7.1 × 10¹⁰/yr, Borexino 2.7 × 10⁸, SNO+ 2.8 × 10⁹, KamLAND 3.5 × 10⁹; in the ¹H window (44–64 keV_ee): JUNO 6.0 × 10¹⁰, Borexino 2.3 × 10⁸.

### 5.3 Threshold, S/B and reach (`P096_model_detector_table.csv`, `P096_exothermic_reach.csv`)
Every signal in §3–4 lies below every recalled threshold except the exothermic proton line (52–64 keV_ee at δ ≥ 350 keV), which clears Borexino's 50 keV_ee hardware threshold at ~1 event/yr. S/B (no pulse-shape discrimination): exothermic ¹²C in JUNO 2.3–2.8 × 10⁻⁶, Borexino 0.8–1.0 × 10⁻⁵ (its lower ¹⁴C), KamLAND/SNO+ 2.3–2.8 × 10⁻⁶; ¹H line 1–4 × 10⁻⁹; L10 on ¹³C 10⁻¹¹; SI 10⁻⁹.
Statistics-only 90 % CL reach after one year, f₂,₉₀ = 1.28√B/S(f₂ = 1): JUNO 6.0 × 10⁻³ / 1.1 × 10⁻⁴ / 9.2 × 10⁻⁶ / 2.9 × 10⁻⁷ = **1.7–2.1× P058's LZ limits**; Borexino 8–9×; KamLAND 9–11×; SNO+ 10–12×. Matching LZ statistically needs an electron-recoil rejection of 0.22–0.34 in JUNO (0.01 in Borexino) — or ~4 years — but requires the ¹⁴C rate and shape in a 6 keV_ee window at 10–20 keV_ee to be known to ~10⁻⁶. With a 10⁻³ relative systematic on ¹⁴C the reach is f₂ > 1.2 (δ = 300; no constraint) to 6 × 10⁻⁵ (380), 360–560× weaker than LZ, and the signal sits below JUNO's trigger threshold in any case. For the hydrogen line the required rejection is 4 × 10⁻⁸–8 × 10⁻⁸ (JUNO) — impossible.

## 6. Conclusion table (`P096_model_detector_table.csv`, 88 rows; extract)
| model → detector | events / yr | E_R → E_ee | threshold | S/B |
|---|---|---|---|---|
| (A) inelastic O₁ δ = 300–380, all | 0 (forbidden; A_min 97–126) | — | — | — |
| (B) L10 on ¹H: JUNO / Borexino | 2.8 × 10⁻⁴ / 3 × 10⁻⁶ | 4.6 → 0.9–2.5 keV_ee | fail | 10⁻¹⁴ |
| (B) L10 on ¹³C: JUNO / SNO+ / KamLAND / Borexino | 0.80 / 0.031 / 0.039 / 0.011 | 57 → 3–9 keV_ee | fail | 10⁻¹¹ |
| (C) SI 10⁻⁴⁷ on C: JUNO / Borexino | 58 / 0.83 | 17 → 1–6 keV_ee | fail | 10⁻⁹ |
| (D) exothermic ¹²C at LZ's f₂ limit: JUNO / SNO+ / KamLAND / Borexino | 1.6–2.0 × 10⁵ / 6–8 × 10³ / 8–10 × 10³ / 2–3 × 10³ | 300–380 → 12–20 keV_ee | fail | 2–10 × 10⁻⁶ |
| (D) exothermic ¹H: JUNO / Borexino | 64–88 / 0.7–1.0 | 300–380 → 44–64 keV_ee | pass Borexino (δ ≥ 350) only | 10⁻⁹ |
| (D) exothermic ¹⁶O, SNO+ water | 1.3–1.5 × 10⁴ | 300–380 → no Cherenkov light | fail | — |

## 7. Robustness and failed approaches
- WimPyDD `WD.C13` cannot be used as a target (isotope object; `TypeError`) and has `itar = 0`; hence the analytic ¹³C route. Using the naive Cecil formula below 0.3 MeV gave Q_p = 0.014 at 100 keV (the fit fails there) — replaced by Birks integration; the Cecil value at 300 keV (0.093) is kept as the low edge.
- Halo: elastic results use 16 June (the assignment's v_max); the Sun-frame L10 normalisation differs by 2 %. Exothermic totals are halo-independent (< 0.1 %); the κ normalisation inherits the endothermic day dependence (×1.45 June / ×0.62 December at δ = 300; ×2.5 / 0.10 at 350; the δ = 380 value is tail-dominated and 45 % above P021's).
- Q_C uncertainty ×0.6–1.7 moves the ¹²C line between 7 and 33 keV_ee; ¹⁴C in the window changes by < ×1.5; no conclusion changes. A ¹⁴C/¹²C of 10⁻¹⁸ instead of 10⁻¹⁷ in JUNO improves the stat-only reach by √10 (still ≥ 0.6× LZ) and the systematic floor by 10.
- Not done: pile-up of ¹⁴C (at 3 × 10⁴ Bq in JUNO, ~0.4 % of events in a 100 ns window) and ¹⁴C–¹⁴C summing, which further fill the 10–60 keV_ee region; NR/ER pulse-shape discrimination at 10–60 keV_ee in LAB (no reliable recalled value; treated as a free rejection factor); the P058 κ-free case (f₂ unconstrained) — with κ free the JUNO rate is simply 4.3 × 10⁷ × κ f₂ × 17 550 events/yr.

## 8. Figures
- `figures/P096_fig1_ceilings_and_exothermic.png` — Left: inelastic ceiling μv²_max/2 versus A at 0.4/1/4 TeV and the elastic E_R,max (1 TeV) with δ = 300/350/366/380 keV lines and A_min; scintillator nuclei (red) lie 5–100× below the lines. Right: JUNO, 1 year: exothermic ¹²C (solid) and ¹H (dashed) recoil spectra in keV_ee at P058's f₂ limits for δ = 300 and 380 keV against the ¹⁴C β spectrum (10⁻¹⁷); the recalled JUNO trigger threshold is dotted.
- `figures/P096_fig2_L10_light_nuclei.png` — L10 and O6 at LZ's best fit on ¹H (WimPyDD, with the analytic free-proton curve) and L10 on ¹³C per tonne of natural carbon (single-particle estimate), events/(kt·yr·keV).

## 9. Recalled knowledge (flagged)
| item | source | reliability |
|---|---|---|
| Borexino 278 t PC+PPO, ended 2021; KamLAND ≈ 1 kt; JUNO 20 kt LAB (commissioning 2025–26); SNO+ 780 t LAB, 905 t water phase | experiment papers | certain / certain / certain / likely |
| PC = C₉H₁₂; LAB ≈ C₁₈H₃₀; KamLAND 80 % dodecane + 20 % PC | detector papers | certain / likely / likely |
| thresholds: Borexino ~50 keV_ee hardware; KamLAND, SNO+ ~200 keV_ee; JUNO ~100 keV_ee trigger | detector papers | uncertain |
| ¹⁴C/¹²C: Borexino 2.7 × 10⁻¹⁸; LAB detectors ~10⁻¹⁷ | Borexino; JUNO/SNO+ design | likely / uncertain |
| ¹⁴C half-life 5730 yr, Q_β = 156.5 keV, allowed spectrum | nuclear data | certain |
| Birks kB = 0.0098 cm/MeV for LAB; ρ = 0.86 g/cm³ | von Krosigk et al. 2013 | likely / certain |
| proton electronic stopping table (PSTAR-like, hydrocarbon/water) | NIST PSTAR | likely (±20 %) |
| carbon-ion S_e ≈ 2500 MeV cm²/g at 300 keV, electronic fraction 0.6; literature Q_C = 0.02–0.06 | SRIM/LSS; scintillator papers | uncertain |
| Cecil–Anderson–Madey NE-213 proton light output | NIM 161 (1979) 439 | likely |
| ¹³C: 1/2⁻ ground state, p₁/₂ neutron, single-particle ⟨S_n⟩ = −1/6 (shell model ≈ −0.17); abundance 1.07 %; HO b = 1.64 fm | nuclear structure | likely / certain / likely |
| SD scaling (J+1)/J (a_p⟨S_p⟩ + a_n⟨S_n⟩)²; spin averages ⟨S_iS_j⟩ = δ_ij/4; Anand dσ/dE prefactor 2m_T/(4πv²) | Engel–Pittel–Vogel; Anand et al. 2014 | certain |
| σ_n = 10⁻⁴⁷ cm² ≈ LZ 2024 SI limit at 1 TeV | LZ PRL 2025 | likely |
| Cherenkov threshold: protons ≳ 480 MeV in water; heavy ions give no light | textbook | certain |
| exothermic kinematics E* = |δ|μ/m_N, half-width μv√(2μ|δ|)/m_N; σv → σ₀F²√(2|δ|/μ) | Tucker-Smith–Weiner; Graham et al. 2010 | certain |
| LSS electronic stopping ∝ v at low energy | Lindhard et al. | certain |

## 10. References
LZ Collaboration, arXiv:2609.02823 (2026). D. Tucker-Smith, N. Weiner, PRD 64, 043502 (2001). P. W. Graham, D. E. Kaplan, S. Rajendran, M. T. Walters, PRD 82, 063512 (2010). N. Anand, A. L. Fitzpatrick, W. C. Haxton, PRC 89, 065501 (2014). I. Jeong, S. Kang, S. Scopel, G. Tomar, CPC 276, 108342 (2022). J. B. Birks, *The Theory and Practice of Scintillation Counting* (Pergamon, 1964). R. A. Cecil, B. D. Anderson, R. Madey, NIM 161, 439 (1979). B. von Krosigk et al., EPJC 73, 2390 (2013). Borexino Collaboration, Nature 512, 383 (2014). JUNO Collaboration, J. Phys. G 43, 030401 (2016). D. Baxter et al., EPJC 81, 907 (2021). Corpus: P003, P012, P021, P046, P057, P058, P067.

## 11. Tools and provenance (mirrors `provenance/P096.json`)
- Agent tools: Read (PAPER_GUIDE; P046, P067, P012, P003, P021, P057, P058 papers; lzcommon.py; both figures), Bash (ledger/DR/environment listings; WimPyDD target and response-function inspection; P012/P046/P058/P021 details greps; timing tests; 5 script runs), Write/Edit (script, details, provenance, paper).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (stats.norm, integrate.cumulative_trapezoid, special); pandas 3.0.5; matplotlib 3.11.2 (Agg); WimPyDD 2.0.4 (`diff_rate` per-stream kernels, `eft_hamiltonian`, `streamed_halo_function` via `lz.wd_halo`; targets H, C, O, Xe; H `itar = 21`, C13 `itar = 0`); `common/lzcommon.py` (kinematics, `eta0`, `wd_halo`, `wd_hamiltonian`, `wd_rate`, `wd_c_SI_from_sigma_n`, constants).
- Script: `output/code/P096_scintillators.py` — `.venv/bin/python output/code/P096_scintillators.py [--recompute]`.
- Local inputs: `output/provenance/PAPER_GUIDE.md`; `output/papers/P046.md, P067.md, P012.md, P003.md, P021.md, P057.md, P058.md`; `output/work/P012/details.md` (L10 reduction, N(d10 = 1)); `output/work/P058/details.md`, `P058_f2_limits.csv` (method, f₂ limits, R_ROI); `output/work/P021/details.md` (κ̂, N_unit table); `output/code/P046_multi_target.py` (per-isotope kernel pattern); `output/code/common/lzcommon.py`; `WimPyDD/package.py` (itar table), `WimPyDD/Targets/{C,H,O}.tab`; `environment/ENVIRONMENT_versions.txt`; `output/results_ledger.csv`, `output/data_requests/index.csv`, `output/00_evidence_dossier.md` (§1–5).
- Datasets: none. Data requests: none. WimPyDD-generated files: none (only `WD.diff_rate` was used; verified with `find WimPyDD -newer`).
- Failed/abandoned: `WD.C13` as a target (TypeError; `itar = 0`); Cecil formula below 0.3 MeV; a 24-day-averaged endothermic normalisation at δ = 380 keV differs from P021 by 45 % (tail sampling) — reported, not resolved.
