# P082 — A global direct-detection combination for inelastic dark matter after the LZ event: the surviving (m_χ, δ, σ_n) region — research record

Simulated date 2026-09-16 · hep-ph · category IDM · author profile: global-fit / phenomenology group.
Script `output/code/P082_global_idm.py` (run from the root: `.venv/bin/python output/code/P082_global_idm.py --spectra-only 200 300 600`,
`… --spectra-only 2000 10000` (WimPyDD kernels, 106 s + 140 s, cached in `output/work/P082/cache/`), then `.venv/bin/python output/code/P082_global_idm.py`
(analysis, 7 s)). All numbers below are in `P082_results.json`, `P082_scan.csv`, `P082_region_m_delta.csv`, `P082_region_delta_sigma.csv`,
`P082_bestfit_shift.csv`, `P082_archival_counts_1000GeV.csv`, `P082_archival_variants.csv`, `P082_higgsino_1000GeV.csv`,
`P082_future_exposures_1000GeV.csv`, `P082_surfaces.npz` and `run_log.txt`. Marked COMPETING with P059 (archival xenon exposures) and
P021 (LZ-only likelihood): we rebuild both pieces independently in one likelihood and state agreement/disagreement in §5.

## 1. Motivation and question

LZ (arXiv:2609.02823) fits one 248 keV NR-like event in 2.84 t·yr. P021 mapped the isoscalar-O₁ likelihood of that event in (m_χ, δ, σ) at
three masses (peak δ = 380 keV at 1 TeV, 3.6σ local); P038 showed the peak is manufactured by the S1c < 600 phd efficiency edge; P059 counted
the events that four zero-event archival xenon exposures owed the fit (0.29–0.34) and found that their Poisson zeros lower the rate by 22–25 %;
P035/P069 asked what XENONnT/PandaX-4T would see. Nobody has yet written the *global* likelihood — LZ's event, every archival zero, the
non-xenon limits — over a continuous mass grid, nor asked (i) what region of (m_χ, δ, σ_n) survives, (ii) what the joint best fit is, (iii) where a
fixed-cross-section Higgsino may sit, (iv) what the combination excludes if the event is background, and (v) which future exposure shrinks the region
most. We do that here.

## 2. Model

### 2.1 Signal
WimPyDD 2.0.4 shell-model O₁ isoscalar spectra for the unit LZ/Anand coupling κ ≡ (c₁ˢ m_v²)² = 1 (WimPyDD c⁰ = 2/m_v², `lz.wd_c_from_anand`;
P003 convention), per-stream kernels `WD.diff_rate(…, sum_over_streams=False)` on an explicit v_min grid 0–830 km/s (1661 points) contracted with
the Baxter-2021 SHM halo function. **Halo: annual average** (mean of 12 monthly days; the corpus finding is that LZ used the time-averaged SHM,
P007/P021); June-16 and Sun-frame halos are variants. True-energy grid 1.5–340 keV in 3 keV steps. Masses 200, 300, 400, 600, 1000, 2000, 4000,
10000 GeV: the 400/1000/4000 GeV spectra are P021's cached kernels (`output/work/P021/spectra_s_*.npz`, identical construction); the five others were
computed here (690/1150/1864/2363/2537 kernel calls). δ grid: 100 (P021 masses) or 200 keV to 290 keV in 10 keV steps, then 300 keV to the June
kinematic ceiling μv²_max/2 in 5 keV steps (ceilings 276.5/316.6/341.3/370.2/397.1/420.0/432.5/440.4 keV). σ_n = κ μ_N²/(π m_v⁴) × 0.3894×10⁻²⁷ cm²
(LZ supplement "O₁ as a cross-section"; κ = 1 ↔ 2.96×10⁻³⁸ cm² at 1 TeV).

### 2.2 LZ detector and likelihood (P021 model with the P038 edge)
dN/dE_obs(κ) = κ · 2.84 t·yr · ∫dE_t (dR/dE_t) ε(E_t) G(E_obs − E_t; σ_E), σ_E = 11√(E/248) keV (P009).
**Efficiency edge (baseline):** P038's NEST-LZ Monte-Carlo roll-off, ε(E) = 0.955 · ½[1+erf((E−5.4)/(√2·2.5))] · P(S1c < 600 phd ∧ log₁₀S2c < 4.15 | E)
read from `output/work/P038/P038_mc_efficiency.csv` (E50 = 271.6 keV, non-Gaussian tail); variant: the P021/P007 erf model (0.96, 50 % at 269.9 keV,
σ 11.5 keV). Likelihood (P016/P021): the 20 digitised Fig.-5 NR-band bins (S1c < 250 phd, −8 to +2σ_NR; ER-leakage scale θ with a 30 % Gaussian
constraint, profiled by bisection), the empty 125–200 keV bin (b_M = 0.02), and the unbinned 200–290 keV window with b_H = 5.7×10⁻⁴ flat over
200–270 keV and one event at E_obs = 248 keV:
ln L_LZ(κ,θ) = Σ_i[n_i ln(θb_i + κS_L g_i) − (θb_i + κS_L g_i)] + [−(b_M + κS_M)] + ln(b_H/70 + κ f₂₄₈) − (b_H + κS_H) − ½((θ−1)/0.3)².
κ on a grid 0 ∪ logspace(−7.5, 3.5, 551).

### 2.3 Archival zero-event exposures (recalled, flagged)
L_joint(κ; m, δ) = L_LZ(κ) · Π_j exp(−κ N_j(m,δ)), N_j = E_j ∫_{E>100 keV} (dR/dE)_{κ=1} ε_j(E) dE (true energy, as P059).
**100 keV floor:** the archival exposures are treated as zero-event data only above 100 keV true recoil energy, where every one of them was
background-free at high S1; below it they were background-limited (P059 §4.4: the SR1 limits at δ ≤ 150 keV imply 5–9 events; P035: the XENON1T
recast lies 11–10⁵× above LZ's edge), so their information there is the published limits, which do not reach the region. Without the floor the
low-energy ROIs of XENON1T/XENONnT/PandaX-4T would wrongly drive κ̂ → 0 at δ ≤ 150 keV (a first run did this; §7).

| exposure | E_j (t·yr) | acceptance ε_j(E) | reliability |
|---|---|---|---|
| LZ SR1 2022 (NREFT 2023) | 60 d × 5.5 t = 0.9035 | LZ curve with E → E·0.114/0.110 (g1 rescaling: 600 phd ↔ 262 keV), plateau 0.955 | exposure likely (P069: 0.77 with the 4.71 t FV); same S1c range certain (paper l.116); g1 likely; zero high-S1c NR-band events uncertain |
| XENON1T 2018 | 1.0 | 0.85 × [4.9, 40.9] keV | exposure certain; ROI likely |
| LUX 2021 EFT | 3.35×10⁴ kg·d = 0.0917 | 0.8 × [3, 150] keV (100/250 variants) | live days certain; kg·d likely; edge uncertain |
| PandaX-II 2019 SD-EFT | 54 t·d = 0.148 | 0.8 × [3, 100] keV (50/150) | exposure likely; edge uncertain |
| XENONnT 2025 | 3.1 | 0.9 × [3, 60] keV | exposure certain (cited by LZ); edge uncertain |
| PandaX-4T 2025 | 1.54 | 0.9 × [3, 100] keV | exposure certain; edge uncertain |

Archival backgrounds are dropped (with n_j = 0 they are additive constants). Z = √q₀ with q₀ = 2[ln L(κ̂) − ln L(0)]; ln L(0) is identical for the
LZ-only and joint likelihoods because every archival term vanishes at κ = 0.

### 2.4 Regions and limits
(m, δ) region: profile over κ, Δ = 2[ln L_max,global − max_κ ln L(m,δ,κ)] ≤ 2.30 (68 %) / 4.61 (90 %), 2 dof, on the mass grid.
(δ, σ_n) at fixed m (1 and 4 TeV): Δ relative to the fixed-mass maximum, 2 dof; boundaries per δ stored in `P082_region_delta_sigma.csv`.
Area metric: Σ (Δ ≤ 4.61) × Δδ × Δlog₁₀κ in keV·dex. **Exclusion mode:** the event is attributed to background, ln(b_H/70 + κf₂₄₈) → ln(b_H/70); the
90 % CL upper limit is the κ with −2[ln L(κ) − ln L(0)] = 4.606 (exactly the zero-event Poisson limit κS = 2.303 where the low-energy bins are
negligible; P035/P059 convention). Variant "event of unknown origin": Δ = 2(3.89 − 1 − ln 3.89) = 3.063 from the maximum (the n = 1 Poisson
90 % UL of 3.89 events). **Higgsino:** P021's 1000 GeV pure-Higgsino Hamiltonian spectra (`spectra_hig_1000.npz`; c⁰_WD = −7.62×10⁻⁶,
c¹_WD = 8.87×10⁻⁶ GeV⁻², P007) at fixed coupling, plus the isoscalar-equivalent line κ_H = σ_SI,eq π m_v⁴/μ_N² = 0.0777 (P007) on the O₁ˢ surfaces.
**Non-xenon:** the CRESST-II (tungsten) and PICO-60 CF₃I (iodine) curves of LZ Fig. S7 as digitised by P015 (`figS7_curves.json`; end at 366.8 keV),
extrapolated beyond their end with their own log-slopes above 300 keV (0.0211 and 0.0271 dex/keV) — flagged as an extrapolation.
**Future datasets (1 TeV):** Poisson terms with N_X(δ) = E_X × (unit-κ rate in the dataset's window): LZ-like 600 phd windows for LZ (+6.76 t·yr,
P069/P020 recalled ±25 %), XENONnT (3.1; +3.0 untouched recalled ±50 %) and PandaX-4T (1.54; +2.5); a 1000 phd LZ edge using P038's
`P_pass_1000` and P038's 1000 GeV spectra to 800 keV (acceptance gain interpolated in δ from 16 points; ×2.06 at 350, ×31.6 at 380 keV);
CaWO₄ with P046's unit-κ tungsten rate in 10–1300 keV (`rates_vs_delta.csv`, annual halo, per t·yr of tungsten; CaWO₄ tungsten mass fraction 0.6385).
Scenarios: Asimov data at truth A (1 TeV joint maximum, δ = 380 keV, κ = 1.66), truth B (Higgsino point δ = 370 keV, κ = 0.0777), and zero
events. Metric: the 90 % (δ, log σ) area relative to today's joint area (85.3 keV·dex).

## 3. Inputs
| Input | Source |
|---|---|
| Exposure 2.84 t·yr, plateau 0.96, 50 % points 5.4/269.9 keV, event 248 keV, O₁-as-σ formula, Fig. S7 caption, "same S1c range" as SR1 | `inputs/LZ_arXiv_2609.02823_fulltext.tex` (l.116, 117, 455–457, 798–818); `lz.LZ` |
| O₁ˢ spectra 400/1000/4000 GeV (three halos), Higgsino spectra 1000 GeV | `output/work/P021/spectra_s_*.npz`, `spectra_hig_1000.npz` (P021) |
| MC efficiency P_pass_600/P_pass_1000 vs E; 1000 GeV spectra to 800 keV | `output/work/P038/P038_mc_efficiency.csv`, `spectra_s_1000_full.npz` (P038) |
| Fig. 5 bins, b_H anchor | `output/work/P016/P016_results.json` (P016) |
| Higgsino couplings and σ_SI,eq(m) | `output/work/P007/higgsino_couplings.json` (P007) |
| Fig. S7 previous limits (digitised) | `output/work/P015/figS7_curves.json` (P015) |
| Tungsten/iodine unit-coupling rates vs δ | `output/work/P046/rates_vs_delta.csv` (P046) |
| Archival exposures, ROI edges | P059 §2.3 and P069 ledger (recalled there; re-flagged here) |
| Kinematics, halo, WimPyDD wrappers | `output/code/common/lzcommon.py` |

Recalled knowledge (flagged): R1 (ħc)² = 0.3894×10⁻²⁷ GeV²cm² (certain); R2 χ² quantiles 1.0/2.706 (1 dof), 2.30/4.61 (2 dof) and Wilks
asymptotics Z = √q₀ (certain); R3 Poisson 90 % UL 2.303 (n=0) and 3.89 (n=1) events (certain); R4 LZ SR1 60 live days × 5.5 t, g1 = 0.114
(likely); R5 XENON1T 1.0 t·yr, ROI 4.9–40.9 keV (certain/likely); R6 LUX 2021 3.35×10⁴ kg·d (likely), ROI edge 100–250 keV (uncertain); R7
PandaX-II 54 t·d (likely), edge 50–150 keV (uncertain); R8 XENONnT 3.1 t·yr and PandaX-4T 1.54 t·yr (certain), low-energy ROI edges ≈ 60/100 keV
(uncertain); R9 PICO-60 CF₃I exposure ≈ 1335 kg·d (likely), CF₃I iodine mass fraction 127/196.9 (certain); R10 PICO-60 CF₃I inelastic search
published 2023 (certain — in LZ's bibliography); R11 CDMS-II inelastic-DM search (2011) covering δ ≲ 150 keV (likely); R12 no NaI(Tl) or
germanium analysis with a > 200 keV inelastic window is known to us (statement of ignorance, not a datum); R13 CaWO₄ tungsten mass fraction 0.6385
(certain, from atomic masses); R14 Okabe–Ito palette is CVD-safe (certain, cosmetic).

## 4. Results

### 4.1 Validation against P021/P038/P059
1000 GeV, LZ-only, P038 edge: Z = 2.68/3.12/3.42/3.57/3.44/3.15 at δ = 300/350/370/380/385/390 keV (P021 erf edge: 2.67/3.12/3.42/3.59/3.47/3.16;
P038 MC edge: 2.67/3.12/3.41/3.47); κ̂(380) = 2.09 (P021 2.19; erf variant here 1.74); 2-dof LZ-only regions δ = 360–385 (68 %) and 330–390 keV (90 %) —
identical to P021. Archival counts at the LZ best fit, 1 TeV (`P082_archival_counts_1000GeV.csv`): SR1 0.316/0.323/0.309/0.307/0.247/0.109 at
δ = 250/300/350/365/380/390 keV; LUX 0.009/0.006/0/0/0/0; XENON1T, PandaX-II, XENONnT, PandaX-4T exactly 0 above 100 keV; totals 0.32/0.33/0.31/0.31/0.25/0.11,
P(0) = 0.72–0.90. P059: 0.29–0.34 and P(0) = 0.71–0.75 (they include SR1 down to 5 keV; without our floor SR1 at δ = 250 is 0.361 — the only entry that moves).

### 4.2 Joint best fit and how far the archival zeros move it (`P082_bestfit_shift.csv`)
| δ (keV), 1 TeV | Z_LZ | Z_joint | ΔZ | κ̂_LZ | κ̂_joint | ratio | μ̂_LZ events (joint) | N_arch at LZ fit | 90 % upper κ ratio |
|---|---|---|---|---|---|---|---|---|---|
| 300 | 2.677 | 2.570 | −0.107 | 7.59e-5 | 5.75e-5 | 0.759 | 0.77 | 0.328 | 0.752 |
| 350 | 3.124 | 3.035 | −0.089 | 3.80e-3 | 2.88e-3 | 0.759 | 0.74 | 0.309 | 0.760 |
| 370 | 3.422 | 3.344 | −0.078 | 0.115 | 0.0871 | 0.759 | 0.76 | 0.303 | 0.768 |
| 380 | 3.572 | 3.511 | −0.061 | 2.09 | 1.66 | 0.794 | 0.81 | 0.247 | 0.805 |
| 385 | 3.442 | 3.402 | −0.039 | 10.5 | 9.12 | 0.871 | 0.87 | 0.142 | 0.875 |
| 390 | 3.145 | 3.112 | −0.033 | 34.7 | 31.6 | 0.912 | 0.91 | 0.109 | 0.901 |
Over all masses and δ ≥ 300 keV the ratio κ̂_joint/κ̂_LZ is 0.72–1.0 and ΔZ = −0.11 to −0.01; 99.6 % of the archival expectation is SR1's. **Agreement with
P059** (×0.75–0.78, ΔZ −0.07) for δ ≤ 370 keV; **new:** above 375 keV the shift fades (×0.79 → 0.91) because SR1's edge, 9.6 keV lower than SR3's,
removes more of the narrowing spectrum. Variants (`P082_archival_variants.csv`): SR1 with 0.77 t·yr (P069's FV) → 0.26–0.28 instead of 0.31–0.32
events (ratio 0.79); erf edge → ≤ 0.5 % change; LUX edge 250 keV → 0.027 events at every δ; PandaX-II edge 150 keV → 0.014 (250) / 0.009 (300) / 0.
Global maximum: on the grid edge, m = 10 TeV, δ = 405 keV (Z_LZ 3.59, Z_joint 3.53); the profile is flat in mass above 1 TeV (Δ ≤ 0.14 between 1 and 10 TeV).

### 4.3 Regions (`P082_region_m_delta.csv`, `P082_region_delta_sigma.csv`; Figs. 1, 2)
(m, δ), profiled over σ_n, joint (LZ-only in brackets):
| m (GeV) | 200 | 300 | 400 | 600 | 1000 | 2000 | 4000 | 10000 |
|---|---|---|---|---|---|---|---|---|
| Δ_min | 9.2 (9.1) | 5.3 (5.2) | 1.5 (1.4) | 0.17 (0.23) | 0.14 (0.11) | 0.005 (0.02) | 0.03 (0.01) | 0 (0) |
| 68 % δ | – | – | 340 (340) | 350–365 (350–365) | 365–385 (360–385) | 370–400 (370–400) | 375–410 (375–405) | 380–410 (375–410) |
| 90 % δ | – | – | 320–340 (320–340) | 325–370 (325–365) | 335–390 (330–390) | 340–405 (335–405) | 340–410 (335–410) | 340–415 (340–415) |
m ≤ 300 GeV is excluded at > 90 % (2 dof) by LZ alone (the ceiling is too low to place 248 keV near the spectrum's median); m ≥ 400 GeV is allowed with no
upper bound; the allowed δ band tracks δ_max(248 keV) from below, 5–30 keV wide at 68 %.
(δ, σ_n) at fixed mass, joint (LZ-only):
| | δ 68 % | δ 90 % | σ_n 68 % (cm²) | σ_n 90 % (cm²) | best (δ, κ̂, σ̂_n) | area90 (keV·dex) |
|---|---|---|---|---|---|---|
| 1 TeV | 360–385 (360–385) | 330–390 (330–390) | 2.8e-40 – 7.4e-37 (3.1e-40 – 8.2e-37) | 8.2e-42 – 2.8e-36 (9.0e-42 – 3.0e-36) | 380 keV, 1.66, 4.9e-38 (2.09, 6.2e-38) | 85.3 (87.1) |
| 4 TeV | 375–410 (375–405) | 340–410 (335–410) | 5.2e-40 – 1.3e-36 (5.9e-40 – 1.4e-36) | 1.9e-41 – 3.7e-36 (2.2e-41 – 3.9e-36) | 400 keV, 3.47, 1.0e-37 (4.29, 1.3e-37) | 102.3 (105.3) |
The archival zeros shave 2–3 % of the region's area: the upper σ boundary drops ×0.75–0.90, the lower boundary ×0.9, the δ extent is unchanged at 5 keV
resolution. Selected joint 1 TeV rows: κ̂ = 5.75e-5 / 2.88e-3 / 0.0302 / 0.0871 / 1.66 / 9.12 / 31.6 at δ = 300/350/365/370/380/385/390 keV, with 1-dof 90 %
κ intervals 5.3e-6–2.1e-4, 3.0e-4–1.07e-2, 3.2e-3–0.112, 9.2e-3–0.321, 0.173–6.02, 0.96–33.6, 3.2–114.

### 4.4 Higgsino (`P082_higgsino_1000GeV.csv`)
Fixed physical coupling, 1 TeV: joint best δ = 370 keV (N_LZ = 0.50, N_arch = 0.107 events), 68 % 365–375, 90 % 365–380 keV (LZ-only 365–375 / 360–380;
P021 365–375 / 360–380); Z 3.43 → 3.40. The isoscalar-equivalent line κ_H = 0.0777 lies inside the joint 68 % / 90 % (δ, κ) bands at δ = 370–375 / 365–380 keV
(1 TeV), 335 / 330–340 (400 GeV), 355 / 350–360 (600), 380–385 / 375–390 (2 TeV), 380–390 / 375–395 (4 TeV), 380–385 / 375–395 (10 TeV). In exclusion mode
(§4.5) the Higgsino σ_SI,eq = 2.30×10⁻³⁹ cm² is excluded for δ ≤ 365 keV (UL 2.09×10⁻³⁹) and allowed from ≈ 366 keV (UL 6.0×10⁻³⁹ at 370).

### 4.5 Exclusion mode (event treated as background; `P082_scan.csv` columns UL_excl_*)
| δ (keV), 1 TeV | LZ-only σ_UL (cm²) | joint σ_UL | ratio | joint, n = 1 convention | P035 XENONnT+PandaX-4T 270 keV projection |
|---|---|---|---|---|---|
| 250 | 7.5e-43 | 5.7e-43 | 0.76 | 9.5e-43 | 4.2e-43 |
| 300 | 5.2e-42 | 3.9e-42 | 0.75 | 6.6e-42 | 3.3e-42 |
| 350 | 2.6e-40 | 2.0e-40 | 0.76 | 3.4e-40 | 2.7e-40 |
| 365 | 2.7e-39 | 2.1e-39 | 0.76 | 3.5e-39 | – |
| 370 | 7.8e-39 | 6.0e-39 | 0.77 | 1.0e-38 | – |
| 380 | 1.4e-37 | 1.1e-37 | 0.81 | 1.9e-37 | – |
| 390 | 2.4e-36 | 2.1e-36 | 0.90 | 3.6e-36 | – |
4 TeV joint: 7.9e-42 (300), 1.3e-40 (350), 3.8e-39 (380), 2.4e-37 (400), 2.5e-36 (410). The combination improves LZ's own exclusion by 24–25 % up to
δ = 370 keV and by 10–19 % above — about what a 270 keV re-analysis of XENONnT+PandaX-4T would add (P035).

### 4.6 Non-xenon targets
Published high-energy analyses we can recall with confidence: **iodine** — PICO-60 CF₃I inelastic search (PRD 108, 062003, 2023; LZ Fig. S7 red curve);
**tungsten** — CRESST-II as recast in LZ Fig. S7 (orange; references CRESST 2020 and Bramante et al. 2016); **germanium** — CDMS-II (2011) inelastic
search, δ ≲ 150 keV (likely). No NaI(Tl) (DAMA/LIBRA, COSINE-100, ANAIS-112) high-energy inelastic analysis is known to us (P028 found none) and we
invent none. Kinematics: germanium's ceiling is 248 keV even for infinite mass (blind to δ ≥ 250); iodine's 1 TeV ceiling is 385 keV (below xenon's 397),
so iodine drops out *before* xenon; tungsten reaches 533 keV. Ratios limit/joint-90 %-upper-edge at 1 TeV: CRESST-II 3848/614/19/4.7/2.5 and PICO
1440/317/12/3.3/1.9 at δ = 350/365/380/385/390 keV (the last three points extrapolated with the curves' own slopes; flagged). Neither bites, but at
δ ≥ 385 keV both come within a factor 2–5 of the upper edge — the only place where a non-xenon result is close. Sanity check of the PICO curve: with the
recalled 1335 kg·d CF₃I exposure (iodine 0.0024 t·yr) and P046's iodine rates it corresponds to 6.8/6.8/1.7 events at δ = 250/300/350 keV — a few-event
limit, as expected; the disagreement with a pure count scaling (56× rate fall vs 14× limit rise from 300 to 350) shows the recast's assumptions are not
reproducible from the plot, hence no count-based extrapolation is attempted.

### 4.7 Which future exposure shrinks the region most (`P082_future_exposures_1000GeV.csv`; Fig. 3)
| dataset | events at truth A (δ 380) | truth B (Higgsino 370) | area fraction A / B | if zero: area, δ 90 %, Z |
|---|---|---|---|---|
| LZ +6.76 t·yr, 600 phd | 1.93 | 1.61 | 0.55 / 0.59 | 1.03, 330–390, 3.19 |
| LZ +6.76 t·yr, 1000 phd edge | 60.9 | 12.6 | 0.043 / 0.19 | 3.6, 200–385, 2.50 |
| XENONnT 3.1 t·yr → 270 keV | 0.88 | 0.74 | 0.70 / 0.74 | 1.02, 330–390, 3.33 |
| XENONnT 6.1 t·yr | 1.74 | 1.45 | 0.58 / 0.61 | 1.03, –, 3.21 |
| PandaX-4T 1.54 t·yr → 270 keV | 0.44 | 0.37 | 0.81 / 0.84 | 1.01, –, 3.41 |
| PandaX-4T 4.0 t·yr | 1.15 | 0.96 | 0.65 / 0.69 | 1.02, –, 3.29 |
| CaWO₄ 100 kg·yr | 643 | 42 | 0.006 / 0.059 | 4.7, 150–375, 2.58 |
| CaWO₄ 10 kg·yr | 64 | 4.2 | 0.027 / 0.20 | 2.9, 200–375, 2.97 |
Reading: same-window xenon exposures only add rate information (region ×0.55–0.8, δ extent unchanged, because their N_X(δ) ∝ LZ's own). Datasets that
see the > 270 keV lobe — LZ's own data with a 1000 phd edge, or CaWO₄ — collapse the region by 5–200× if the fit is right (the tungsten count is 643 at
δ = 380 keV, 42 at the Higgsino point), and if they see nothing they remove the δ ≥ 380 keV solution altogether (the region re-centres at lower δ and *grows*,
κ upper edge 6.0 → 0.005 (LZ 1000 phd) / 0.0013 (CaWO₄)), lowering Z to 2.5–2.6. Expected counts at the fit for δ = 350/365 keV: LZ 1000 phd 3.7/8.6;
CaWO₄ 100 kg·yr 3.1/19.5 (P046: 4.1/29 with its own normalisation; P069's "2–5 hidden events" in 19 t·yr is consistent with our 0.44 + 0.88 + 1.93 = 3.3 at 380).

### 4.8 Robustness (1 TeV, joint)
| variant | δ_best | κ̂ | Z_best | δ 68 % | δ 90 % | area90 |
|---|---|---|---|---|---|---|
| baseline (P038 edge, annual halo) | 380 | 1.66 | 3.51 | 365–385 | 335–390 | 85.3 |
| P021 erf edge | 380 | 1.74 | 3.52 | 365–385 | 330–390 | 83.0 |
| June-16 halo | 380 | 0.44 | 3.53 | 365–385 | 340–390 | 73.8 |
| Sun-frame halo | 370 | 0.60 | 3.42 | 350–385 | 315–385 | 104.0 |
(Region δ ranges here from the profile over κ, 2-dof levels.) The Sun-frame halo (which reproduces LZ's Fig. 6 edges, P021/P035) moves the peak to 370 keV
and widens the region by 22 %; the June halo lowers κ̂ ×3.8 at 380 keV (rate ×3.8 higher near the edge).

## 5. Agreement and disagreement with P021 and P059
* P021: our LZ-only regions coincide (68 % 360–385, 90 % 330–390 keV at 1 TeV; peaks 340/380/400 keV at 400/1000/4000 GeV); with the P038 edge Z_peak is
  3.57 instead of 3.59 and κ̂ 2.09 instead of 2.19. New: masses 200–300 GeV are excluded at > 90 % (2 dof), 600 GeV sits at Δ = 0.17, and the profile is flat
  above 1 TeV, so m_χ is bounded only from below (≈ 350 GeV).
* P059: we confirm 0.31–0.33 archival events at the fit (SR1 dominated), κ̂ × 0.76 and ΔZ −0.09 to −0.11 for δ ≤ 370 keV. We add that the shift fades to ×0.91
  at 390 keV, that the low-energy ROIs must not be treated as zeros (their zero-event treatment would wrongly kill δ ≤ 150 keV), and that the effect on the
  2-dof region is only 2–3 % in area. P059's headline is upheld; its "rate down 22–25 %" applies to δ ≤ 370 keV.
* P069: the 2–5 hidden events are reproduced in order of magnitude (3.3 at 380 keV for LZ + XENONnT + PandaX-4T with 600-phd-like windows), and we agree
  that the decisive step is a window beyond 270 keV, not more of the same window.
* P046: the CaWO₄ 100 kg·yr count at the Higgsino point (42) and at δ = 350/365 (3.1/19.5) agrees with P046 to 30 %; the tungsten programme is the strongest
  single region-shrinker, but it depends on the tungsten form factor (×2, P015/P046) and a zero-background assumption.

## 6. Figures
* `figures/P082_fig1_m_delta_region.png` — (m_χ, δ) 68/90 % 2-dof regions profiled over σ_n, LZ-only (blue) and joint (vermilion), best-fit ridges, kinematic
  ceiling and δ_max(248 keV); star = joint maximum at the grid edge (10 TeV, 405 keV).
* `figures/P082_fig2_delta_sigma.png` — (δ, σ_n) at 1 and 4 TeV: 68 % fills and 90 % dashed contours (LZ-only, joint), joint best fit σ̂_n(δ), exclusion-mode
  90 % ULs (joint solid, LZ-only dotted), Higgsino σ_SI,eq, CRESST-II and PICO Fig.-S7 curves with dotted extrapolations (1 TeV panel).
* `figures/P082_fig3_future.png` — left: 90 % (δ, log σ) area relative to today for each future dataset under truth A, truth B and zero events (zero bars clipped
  at 1.6); right: expected counts at the joint best fit for true δ = 300/350/365/380 keV.

## 7. Failed or corrected approaches
* First full run treated every archival ROI as zero-event data down to 5 keV: the XENON1T/XENONnT/PandaX-4T low-energy windows then predicted 2.4–4.4 events
  at δ = 110–190 keV and drove κ̂ → 0 there, contradicting the published (background-limited) low-energy limits, which sit far above the region (P035, P059 §4.4).
  Replaced by the 100 keV floor (§2.3); no result at δ ≥ 300 keV changed by more than 0.1 %.
* The future-exposure block first took the *global* (m, δ) maximum (10 TeV, 405 keV) and mapped it onto the 1 TeV grid, landing at δ = 395 keV where κ̂ = 0, so
  every Asimov count was zero. Replaced by the 1 TeV conditional maximum and a second truth (Higgsino point).
* Region area as the sole shrinkage metric is ambiguous for null outcomes (the region re-centres at lower δ and grows); the κ upper edge and δ range are reported alongside.
* A count-based extrapolation of the PICO iodine curve was abandoned (its δ-slope is inconsistent with a fixed-count limit); the curves are extrapolated with their own slopes only.

## 8. References
LZ Collaboration, arXiv:2609.02823 (2026). G. Cowan, K. Cranmer, E. Gross, O. Vitells, EPJC 71, 1554 (2011). G. J. Feldman, R. D. Cousins, PRD 57, 3873 (1998).
D. Baxter et al., EPJC 81, 907 (2021). I. Jeong, S. Kang, S. Scopel, G. Tomar, CPC 276, 108342 (2022). N. Anand, A. L. Fitzpatrick, W. C. Haxton, PRC 89, 065501 (2014).
D. Tucker-Smith, N. Weiner, PRD 64, 043502 (2001). J. Bramante, P. J. Fox, G. D. Kribs, A. Martin, PRD 94, 115026 (2016). E. Adams et al. (PICO), PRD 108, 062003 (2023).
J. Aalbers et al. (LZ), PRD 109, 092003 (2024). D. S. Akerib et al. (LUX), PRD 104, 062005 (2021). E. Aprile et al. (XENON), PRL 121, 111302 (2018).
Corpus: P002, P003, P005, P007, P009, P015, P016, P021, P035, P038, P046, P059, P069.

## 9. Tools and provenance (mirrors `output/provenance/P082.json`)
Agent tools: Read ×20 (PAPER_GUIDE; papers P059/P021/P038/P069/P005/P046/P035; P021 and P059 details; P021 and P059 scripts; lzcommon ×2; own figures ×6),
Bash ×16 (directory listings, cache/format inspection, WimPyDD timing test, corpus greps, two spectra-building runs, five analysis runs, JSON print, three word-count checks),
Write ×4 (script, details, provenance, paper), Edit ×16 (12 script fixes, 4 paper trims), Skill ×1 (dataviz; its node validator not run per PAPER_GUIDE).
Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (stats.norm, special.erf, optimize); pandas 3.0.5; matplotlib 3.11.2 (Agg); WimPyDD 2.0.4
(eft_hamiltonian via lzcommon.wd_hamiltonian, streamed_halo_function via lzcommon.wd_halo, diff_rate with sum_over_streams=False); common/lzcommon.py
(LZ, kinematics, wd_halo, wd_hamiltonian, wd_c_from_anand, mu_red, m_nucleus_gev, vmax_kms, v_earth_kms, delta_max_kev, E_R_range_keV).
Script: `output/code/P082_global_idm.py` (commands in §0). Local inputs: tex passages, P021/P038/P016/P007/P015/P046/P059/P069 work files, papers
P005/P021/P035/P038/P046/P059/P069, lzcommon, ledger, PAPER_GUIDE, ENVIRONMENT_versions. Recalled: 14 items (§3). Datasets: none. Data requests: none.
WimPyDD-generated files: none outside `output/work/P082/cache/` (five spectra .npz; diff_rate writes no response-function files).
