# P051 — details: inelastic dark matter with a light mediator and the LZ 248 keV event

Simulated date 2026-09-11. Author profile: light-mediator dark-sector phenomenologists. Category IDM (hep-ph).
Script: `output/code/P051_light_mediator.py` (stages `kernels`, `exact`, `analysis`; all results under `output/work/P051/`).

## 1. Motivation and framework

P011 fitted the LZ event with a pseudo-Dirac fermion and a kinetically mixed dark photon and, in a side calculation,
found that a light A′ (m_A′ = 0.1/0.3/1 GeV) suppresses the rate by S = 0.034/0.445/0.915 at δ = 300 keV. P002 showed that
the contact O₁ spectrum puts 248 keV at the 99.5th in-ROI percentile for δ = 300 keV, and P021/P038 that the single-event
profile likelihood peaks at δ ≈ 380 keV (1 TeV) largely because the S1c < 600 phd edge removes the high-energy lobe.
The question here: since a mediator lighter than the momentum transfer q = √(2 m_N E_R) ≈ 0.246 GeV (248 keV, A = 131.3)
weights the recoil spectrum toward low q *within* the kinematic window [E⁻, E⁺], does it (a) make 248 keV typical,
(b) move the preferred δ down to 300–350 keV, (c) remain compatible with the χ₂-decay (exothermic) floor of P011/P026,
the Planck bound of P025 and dark-photon searches, and (d) what does it predict for the empty NR-band regions
(100–200 keV below the event; 600–1000 phd above it, P038)?

### 1.1 Interaction

For t-channel exchange of a vector or scalar of mass m_med coupling to the nucleon number/charge, the NREFT O₁
Wilson coefficient acquires the propagator:

    c(q) = c₀ · m_med² / (m_med² + q²)          (both isospin components; c₀ = coupling in the contact limit)

so the rate is the contact rate times

    P(E_R) = m_med⁴ / (m_med² + q²)²,   q² = 2 m_A E_R      (m_A = nuclear mass of the isotope).

For m_med ≫ q, P → 1 (contact); for m_med ≪ q, P → m_med⁴/q⁴ and the rate no longer depends on m_med once
c₀ ∝ 1/m_med² is absorbed (the ε "floor" of P011's Fig. 3).

Mediators: m_med = 0.05, 0.1, 0.2, 0.3, 0.5, 1 GeV and ∞ (contact). Two isospin structures:
* isoscalar, LZ unit coupling c₁ˢ = 1/m_v² (Anand) = WimPyDD c⁰ = 2/m_v², c¹ = 0 (P003 convention, `lz.wd_c_from_anand`);
* proton-only (dark photon, P011): c_p = 1/m_v², c_n = 0 → WimPyDD c⁰ = c¹ = 1/m_v².
m_χ = 1000 GeV throughout; δ = 250, 260, …, 370, 375, 380, 385, 390 keV.

### 1.2 Kinematics (16 June halo, v_max = 809 km/s, A = 131.29)

E± = (μ²v²/m_A)[1 ± √(1 − 2δ/(μv²))]² (`lz.E_R_range_keV`); windows: δ = 250: 54–923 keV; 300: 90–798; 350: 150–648;
380: 217–528; 390: 257–470 keV. δ_max(248 keV) = 386.6 keV (P002). E⁻ > 55 keV for every δ ≥ 260 keV, so the 2024
low-energy search (5.4–55 keV) is untouched; the affected empty region is 100–200 keV NR (≈ 150–400 phd).

### 1.3 Halos (all on one v_min grid 0–830 km/s, 831 points)

* `sun`: `lz.wd_halo()` with no day — WimPyDD's single-day **Sun-frame** halo (v_E = 250.6 km/s, no Earth orbital motion;
  P035); η > 0 up to 796 km/s.
* `june`: day 167; η > 0 up to 811 km/s.
* `annual`: mean of 12 monthly days (15 + 365.25 k/12). This is the P021/P038 convention and our default for Z(δ).
Baxter-2021 parameters (v₀ = 238, v_esc = 544 km/s, Sun peculiar velocity) via lzcommon.

### 1.4 Detector model

* Efficiency in true NR energy (P038, `P038_results.json['edges']`): plateau 0.955 × erf low edge (5.4 keV, σ 2.5 keV) ×
  ½ erfc((E − E50)/√2σ) with E50 = 271.64 keV, σ = 10.93 keV for S1c < 600 phd and E50 = 423.24 keV, σ = 14.55 keV for
  S1c < 1000 phd. "600–1000 phd" events = S_acc(1000) − S_acc(600).
* Resolution σ_E = 11 √(E/248) keV (P009/P021); observed-energy grid 0.5–720 keV in 1 keV steps; true grid 40–720 keV in
  4 keV steps restricted to the kinematic window (A = 124 lower edge, A = 136 upper edge, ±4 keV).
* Exposure 2.84 t·yr.

### 1.5 Single-event statistic (P021/P038 companion-free form)

With κ profiled, the extended likelihood of one event at E_obs = 248 keV over a flat NR-band background
b = b_H/W_H = 5.7×10⁻⁴/70 keV⁻¹ (P016 anchor) gives

    q₀ = 2 [ln(f̃/b) − 1 + b/f̃],   f̃ = f_ev / S_acc,

where f_ev is the accepted, smeared signal density at 248 keV (per unit event of the unit-coupling model) and S_acc the
accepted number of events (600 phd edge); κ̂ = 1/S_acc gives one accepted event. Z = √q₀. The statistic depends only on
the accepted spectral shape, and it *does* account for events predicted elsewhere in the ROI (through S_acc). The
68 % range in δ is the set with q₀ ≥ q₀,peak − 1 (1 d.o.f., linear interpolation of q₀ on the δ grid).

### 1.6 Dark-photon translation (P011)

σ_p(q→0) = c_p² μ_p²/π (μ_p = 0.9374 GeV at 1 TeV); unit coupling → σ_p,unit = 1.2447×10⁻⁴⁰ cm²;
for one accepted event σ_p = κ̂ σ_p,unit. Kinetic mixing: σ_p = 16π α α_D ε² μ_p²/m_A′⁴, so
ε² α_D = σ_p m_A′⁴/(16π α μ_p²). We quote ε for α_D = 0.1 (ε ∝ α_D^−1/2; thermal α_D = 0.035 per P025 raises ε by 1.69).

χ₂ → χ₁ ν ν̄ through A′–Z mixing (δ < 2 m_e; the γ channel vanishes at one loop, P011):
Γ = 3 G_eff² δ⁵/(120 π³), G_eff = g_D ε tanθ_W g/(4 cosθ_W m_Z²), g_D = √(4π α_D), g = e/sinθ_W — independent of m_A′ for
m_A′ ≪ m_Z. Survival: f₂ = exp(−t_U/τ), t_U = 13.8 Gyr. Exothermic events from surviving χ₂ (χ₂N → χ₁N, δ → −δ):
N_exo = f₂ · κ̂ · S_exo,unit(m_med), with S_exo,unit the accepted (600 phd) exothermic rate at unit c_p and the same
propagator factor P(E). Requiring N_exo ≤ 1 (P011 criterion) gives f₂,max = 1/N_exo(f₂ = 1), τ_max = t_U/ln N_exo(f₂=1),
ε_min = √(τ(ε = 1, α_D, δ)/τ_max). Both ε_req and ε_min scale as α_D^−1/2, so ε_req/ε_min and the m_A′ floor are
α_D-independent (as P011 noted).

## 2. Computation

### 2.1 Kernels
`WD.diff_rate(WD.Xe, ham, 1000, E, VGRID, ONES, j_chi=0.5, delta=δ, sum_over_streams=False)` returns the per-v_min-bin
kernel K(E, v) (×1000×365.25 → /t/yr/keV); the spectrum for any halo is K @ Δη. 5174 kernel calls for the 17 δ × 2 isospin
contact grid (317 s), 264 for the exothermic proton-only kernels (δ = −300, −350, −380 keV; E = 2–350 keV, 4 keV steps),
184 validation calls and 1170 "exact" light-mediator calls (Sec. 2.2). Cached in `kernels_1000GeV.npz`,
`kernels_exothermic.npz`, `kernels_exact_light.npz`, `halos.npz`.

### 2.2 Light mediators: factorisation and its validation
Light-mediator spectra are obtained from the contact kernel times P(E) evaluated with the *mean* xenon mass
(A = 131.29). This is exact for a single isotope; for natural xenon it neglects the fact that the isotope mix at fixed
(E_R, v) differs from the natural abundances (heavier isotopes have larger q at fixed E_R and larger kinematic windows).
Validation against WimPyDD's own q-dependent Wilson coefficients — `eft_hamiltonian(name, {(1,'q2'): f})` with f a
closure `f(q)` returning [c⁰(q), c¹(q)] (no default arguments, avoiding the P031 shared-parameter pitfall; WimPyDD
passes q in GeV and normalises the q-dependence to q = 1 GeV internally, which we checked reproduces the absolute
factor: 0.359 vs analytic 0.357 at 248 keV for 0.3 GeV):

| test | points | max |rel. dev.| |
|---|---|---|
| δ = 300 keV, m_med = 0.05/0.1/0.3/1 GeV, E = 100–500 keV, June & Sun frame, both isospins | 112 | 5.3 % (0.05 GeV, Sun frame, 100 keV); ≤ 1.9 % for m ≥ 0.3 GeV |
| δ = 380 keV, same masses, E = 248/300/400 keV | 48 | 6 % typical; **23 %** at (0.05 GeV, isoscalar, 248 keV, Sun frame) where 248 keV is at the kinematic edge (δ_max(248, Sun frame) ≈ 376 keV) and the rate is negligible |
| exothermic δ = −300 keV, proton-only, E = 50/150/248 keV | 24 | 0.9 % |
| all | 184 | rms 3.0 % |

Impact on the final observables (`P051_exact_vs_factorised.csv`; full WimPyDD light-mediator spectra for
m_med = 0.05 and 0.1 GeV at δ = 300 and 380 keV, both isospins, three halos): S_acc(fact)/S_acc(exact) = 1.004–1.064,
|ΔZ| ≤ 0.0071, |Δ percentile| ≤ 0.0043, N(100–200)/N(200–270) within +0.2…+2.4 %. The factorised grid is therefore
adequate at the precision quoted (two significant figures in ratios, 0.01 in Z).

### 2.3 Contact-limit cross-checks
* vs P038 (annual halo, 600 phd): A_600 = 0.814/0.446/0.0294 (P038 0.81/0.44/0.029) at δ = 300/350/380; Z = 2.669/3.117/3.488
  (P038 2.67/3.12/3.47; P021 2.67/3.12/3.59 with a different efficiency model); 600–1000 phd events per ROI event
  0.156/1.05/29.8 (P038 0.16/1.06/30).
* vs P011 (annual halo, proton-only): σ_p(N = 1, δ = 300) = 8.57×10⁻⁴² cm² (P011 8.59×10⁻⁴²); ε(α_D = 0.1, 1 GeV) = 8.63×10⁻⁷
  (P011 8.7×10⁻⁷); light suppression S(300 keV) = 0.034/0.454/0.915 at 0.1/0.3/1 GeV (P011 0.034/0.445/0.915);
  χ₂-decay floor 5.43/4.45/4.44×10⁻⁶ at 300/350/380 keV (P011 5.4/4.5/4.5×10⁻⁶); N_exo(f₂ = 1, contact) = 853 (P011 858);
  m_A′ floor 2.56/0.93/0.11 GeV (P011 2.6/0.94/0.09; P026 2.45/0.92/0.25 GeV).
* vs P002: percentile of 248 keV in the accepted true spectrum, δ = 300 keV, June halo: 0.996 (P002 0.995 for the in-ROI
  spectrum). Over the *full* window (to 720 keV, no efficiency) the percentile is 0.854 because 15 % of the δ = 300 keV
  spectrum lies above 270 keV (P038's high-energy lobe).

## 3. Results

### 3.1 Spectral shape (isoscalar unless stated; annual halo; June values in `P051_observables.csv`)

δ = 300 keV (window 90–798 keV):

| m_med [GeV] | A_600 | peak [keV] | median (acc.) [keV] | pct(248) true acc. | pct(248) obs. | frac ±23 keV (obs.) | N(100–200)/N(200–270) | P(0 in 100–200) | N(600–1000 phd)/ROI ev. | Z |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.916 | 160 | 162 | 0.9982 | 0.9958 | 0.028 | 6.85 | 0.0011 | 0.039 | 2.413 |
| 0.1 | 0.910 | 160 | 163 | 0.9979 | 0.9953 | 0.031 | 6.35 | 0.0017 | 0.046 | 2.450 |
| 0.2 | 0.891 | 164 | 166 | 0.9973 | 0.9942 | 0.037 | 5.41 | 0.0045 | 0.067 | 2.530 |
| 0.3 | 0.871 | 164 | 168 | 0.9968 | 0.9932 | 0.041 | 4.87 | 0.0077 | 0.089 | 2.582 |
| 0.5 | 0.845 | 168 | 170 | 0.9963 | 0.9922 | 0.046 | 4.42 | 0.012 | 0.119 | 2.629 |
| 1 | 0.824 | 168 | 171 | 0.9959 | 0.9915 | 0.049 | 4.18 | 0.015 | 0.144 | 2.658 |
| contact | 0.814 | 168 | 171 | 0.9957 | 0.9911 | 0.051 | 4.08 | 0.017 | 0.156 | 2.669 |

Proton-only, δ = 300 keV: pct(248) obs. 0.969 (contact) → 0.984 (0.05 GeV); frac ±23 keV 0.116 → 0.070;
N(100–200)/N(200–270) 2.16 → 3.57 (P(0) 0.116 → 0.028); N(600–1000 phd)/ROI event 0.089 → 0.024; Z 3.042 → 2.848.
(The proton-only M-response node sits at 284–291 keV, P011, hence the milder percentiles.)

The tilt is real but bounded: the M-response node at ≈ 266 keV (P002/P003) and the 600 phd edge, not the propagator,
control the spectrum near 248 keV. Between E⁻ = 90 keV and the node, the propagator changes the rate by a factor
P(90)/P(248) = 2.6 (0.05 GeV) but this mostly moves weight from 200–266 keV to 100–170 keV: the peak shifts from 168 to
160 keV and 248 keV becomes *more* extreme.

δ = 330 keV: A_600 0.68 → 0.86; pct(248) obs. 0.983 → 0.990; N(100–200)/N(200–270) 2.13 → 3.18 (P(0) 0.119 → 0.042);
N(600–1000 phd)/ROI 0.37 → 0.11; Z 2.888 → 2.711.
δ = 350 keV: A_600 0.446 → 0.689; pct 0.962 → 0.976; ratio 0.914 → 1.235 (P(0) 0.40 → 0.29); N(600–1000) 1.05 → 0.37; Z 3.117 → 3.005.
δ = 380 keV: A_600 0.029 → 0.059; pct 0.637 → 0.710; ratio 0.043 → 0.053; N(600–1000) 29.8 → 14.7; Z 3.488 → 3.477.
(All contact → 0.05 GeV, isoscalar, annual halo. Proton-only δ = 380: N(600–1000) 5.8 → 2.6.)

### 3.2 Profile significance Z(δ) (`P051_Z_peaks.csv`, Fig. 2)

Annual halo, isoscalar: peak at δ = 380 keV for every m_med, Z_peak = 3.477 (0.05 GeV) … 3.488 (contact); 68 % range
369–390 keV (0.05 GeV; upper end at the grid edge) vs 366–388 keV (contact): the range moves **up** by ≈ 3 keV.
Z(300) = 2.413/2.450/2.530/2.582/2.629/2.658/2.669 for 0.05/0.1/0.2/0.3/0.5/1/∞ GeV; Z(330) = 2.711 → 2.888;
Z(350) = 3.005 → 3.117; Z(380) = 3.477 → 3.488. Δq₀(peak − 300) = 6.27 (0.05 GeV) vs 5.04 (contact): likelihood ratio
peak/300 keV = 23.0 vs 12.4; Δq₀(peak − 330) = 4.74 vs 3.82; Δq₀(peak − 350) = 3.06 vs 2.45.
Proton-only: peak at the 390 keV grid edge (P021 found the collapse at 395 keV) with Z_peak = 3.750 for all m_med;
Z(300) = 2.848 → 3.042; Z(350) = 3.271 → 3.360; LR(peak/300) = 19.7 (0.05 GeV) vs 11.1.
June halo: isoscalar peak 380 keV (Z 3.48–3.50), Z(300) 2.383 → 2.648; proton-only peak 390 (edge), Z(300) 2.827 → 3.028.
Sun-frame halo: isoscalar peak **370** keV (Z 3.47–3.48; δ_max(248) is 376 keV in this frame, cf. P021's Sun-frame 370),
Z(300) 2.423 → 2.675; proton-only peak 380 keV, Z 3.76.
In no case does a light mediator move the peak down or make δ = 300–330 keV more competitive; the tilt lowers Z at
low δ (where the window is wide and the extra weight lands at 100–170 keV) and leaves Z near δ_max unchanged (the window
is narrow and the propagator nearly constant across it, P(217)/P(528) = 1.3 for 0.05 GeV... see Fig. 1 right panel).

### 3.3 Couplings and the (m_A′, ε) plane (`P051_couplings.csv`, Fig. 3; annual halo, proton-only, α_D = 0.1)

| δ [keV] | m_A′ [GeV] | S = N_light/N_contact | σ_p(q→0), N = 1 [cm²] | ε_req | N_exo(f₂ = 1) | ε_floor | ε_req/ε_floor | τ_χ₂(ε_req) [s] |
|---|---|---|---|---|---|---|---|---|
| 300 | 0.05 | 0.0030 | 2.9e-39 | 3.8e-8 | 2130 | 5.8e-6 | 0.0066 | 1.3e21 |
| 300 | 0.1 | 0.034 | 2.5e-40 | 4.5e-8 | 1670 | 5.7e-6 | 0.0078 | 9.6e20 |
| 300 | 0.2 | 0.22 | 3.8e-41 | 7.0e-8 | 1180 | 5.6e-6 | 0.013 | 3.9e20 |
| 300 | 0.3 | 0.45 | 1.9e-41 | 1.1e-7 | 1010 | 5.5e-6 | 0.021 | 1.5e20 |
| 300 | 0.5 | 0.72 | 1.2e-41 | 2.5e-7 | 914 | 5.5e-6 | 0.045 | 3.2e19 |
| 300 | 1 | 0.92 | 9.4e-42 | 8.6e-7 | 869 | 5.4e-6 | 0.16 | 2.6e18 |
| 300 | 3 | 0.99 | 8.7e-42 | 7.7e-6 | 854 | 5.4e-6 | 1.4 | 3.2e16 |
| 300 | 10 | 1.00 | 8.6e-42 | 8.3e-5 | 853 | 5.4e-6 | 15 | 2.8e14 |
| 350 | 0.05 | 0.0022 | 1.3e-37 | 2.6e-7 | 3.2e4 | 4.6e-6 | 0.056 | 1.3e19 |
| 350 | 0.3 | 0.41 | 7.1e-40 | 7.0e-7 | 2.1e4 | 4.5e-6 | 0.16 | 1.8e18 |
| 350 | 1 | 0.90 | 3.2e-40 | 5.1e-6 | 1.8e4 | 4.5e-6 | 1.14 | 3.4e16 |
| 380 | 0.05 | 0.0017 | 3.0e-35 | 3.9e-6 | 4.8e6 | 4.5e-6 | 0.86 | 3.8e16 |
| 380 | 0.1 | 0.021 | 2.4e-36 | 4.4e-6 | 4.3e6 | 4.5e-6 | 0.97 | 3.0e16 |
| 380 | 0.2 | 0.17 | 3.2e-37 | 6.3e-6 | 3.5e6 | 4.5e-6 | 1.4 | 1.5e16 |
| 380 | 1 | 0.89 | 5.8e-38 | 6.8e-5 | 2.5e6 | 4.4e-6 | 15 | 1.3e14 |

ε saturation for m_A′ ≪ q (α_D = 0.1, annual): 3.7×10⁻⁸ (300), 9.4×10⁻⁸ (330), 2.5×10⁻⁷ (350), 3.8×10⁻⁶ (380 keV).
The exothermic spectrum of surviving χ₂ extends to low recoil energies where q is small, so it is *less* suppressed by the
propagator than the 248 keV up-scatter: N_exo(f₂ = 1) rises from 853 (contact) to 2130 (0.05 GeV) at δ = 300 keV and the
floor ε_min moves up slightly (5.44 → 5.79×10⁻⁶). Result: the m_A′ floor (ε_req = ε_min) is 2.56 GeV (δ = 300), 0.93 GeV (350),
0.11 GeV (380 keV) with the annual halo and 2.75/1.13/0.27 GeV with the June halo (P011: 2.6/0.94/0.09; P026: 2.45/0.92/0.55/0.25
at 300/350/365/380). The BaBar ceiling ε ≤ 10⁻³ (recalled, likely) is reached at m_A′ = 22.6 (330), 14.4 (350), 3.9 GeV (380 keV)
and beyond 32 GeV at 300 keV (P011: 34 GeV).
Planck (P025): a thermal α_D (0.035 at 1 TeV) with Sommerfeld enhancement excludes m_A′ < 9.2 GeV. Every light mediator
considered here is therefore excluded twice for δ ≤ 350 keV (χ₂ floor and Planck) and once (Planck) for δ = 380 keV
unless α_D is not thermal. Non-thermal options: (i) α_D ≪ thermal with the relic set by another mechanism (freeze-in or a
UV asymmetry) — but a Majorana splitting δ violates the dark U(1) number, so an asymmetric pseudo-Dirac relic oscillates
into the symmetric state unless δ is generated late (Tucker-Smith–Weiner-type models with tiny δ/m ≈ 3×10⁻⁷ are natural
for this); (ii) α_D large but m_A′ tuned into a Sommerfeld valley (P025: valleys survive only at 9–32 GeV, so not for light A′).
We leave these as model-building caveats; the direct-detection statements above do not depend on them.

Recalled accelerator constraints for a *visibly decaying* A′ (m_A′ < 2 m_χ, so A′ → e⁺e⁻, μ⁺μ⁻, hadrons):
BaBar (2014, visible) ε ≲ 10⁻³ for 0.02–10 GeV [likely]; NA48/2 (2015) ε ≲ (0.5–1)×10⁻³ for 10–70 MeV [likely];
LHCb (2018/2020) prompt ε² ≲ 10⁻⁶…10⁻⁷ in 0.21–70 GeV windows and displaced ε ~ 10⁻⁵–10⁻⁴ near 0.21–0.35 GeV [uncertain];
electron/proton beam dumps E137, E141, E774, Orsay, NuCal, CHARM: roughly ε ≈ 10⁻⁸–10⁻⁵ excluded for m_A′ ≲ 0.1 GeV,
narrowing to ε ≈ (1–3)×10⁻⁶ near 0.3–0.4 GeV [uncertain, ×3 in ε]; FASER (2023–24) ε ≈ 3×10⁻⁵–10⁻⁴ for 17–70 MeV [uncertain].
Reading: for δ = 300–350 keV the required ε ≈ (0.4–7)×10⁻⁷ at m_A′ ≤ 0.3 GeV falls inside or just below the beam-dump band;
for δ = 380 keV, ε ≈ (4–10)×10⁻⁶ at m_A′ = 0.05–0.3 GeV sits in the historically open gap between beam dumps and prompt
searches (τ_A′ ∝ 1/(αε²m_A′) ~ 10⁻⁹–10⁻⁸ s — displaced, the FASER/HPS/Belle II regime). These statements are qualitative
and flagged; the χ₂-decay floor and the P025 Planck bound are the constraints we rely on.

### 3.4 Distinguishing predictions (Fig. 4)

Low side: with the spectrum normalised to one expected event in 200–270 keV, the 100–200 keV band (≈ 150–400 phd,
inside the region LZ's Fig. 4 shows empty apart from the event) contains

| δ [keV] | contact | 1 GeV | 0.3 GeV | 0.1 GeV | 0.05 GeV | P(0) contact → 0.05 GeV |
|---|---|---|---|---|---|---|
| 300 (iso) | 4.08 | 4.18 | 4.87 | 6.35 | 6.85 | 1.7 % → 0.11 % |
| 300 (p) | 2.16 | 2.21 | 2.57 | 3.32 | 3.57 | 11.6 % → 2.8 % |
| 330 (iso) | 2.13 | 2.17 | 2.45 | 3.01 | 3.18 | 11.9 % → 4.2 % |
| 350 (iso) | 0.91 | 0.93 | 1.02 | 1.19 | 1.24 | 40 % → 29 % |
| 380 (iso) | 0.043 | 0.043 | 0.047 | 0.052 | 0.053 | 96 % → 95 % |

With P016's empty bin (125–200 keV) instead: 3.9 → 6.4 (δ = 300, iso). The profile fit already pays for this population
through S_acc (Sec. 1.5), which is why Z(300) falls; the explicit Poisson probability makes the size of the penalty visible.
Note the alternative normalisation "one event in total" gives P(0 in 100–200) = exp(−f_100–200) ≥ e⁻¹ by construction and
is not a test; the conditional normalisation above is the relevant one for a model that claims to explain the event.
High side (600–1000 phd, P038): events per ROI event fall from 0.156/0.37/1.05/29.8 (contact) to 0.039/0.11/0.37/14.7
(0.05 GeV) at δ = 300/330/350/380 keV — a light mediator halves the P038 tension at δ = 380 keV (P(0) = 4×10⁻⁷ instead of
10⁻¹³) but does not remove it, and at δ = 350 keV it makes the empty 600–1000 phd region comfortable (P(0) = 0.69).

## 4. Robustness

* Halo: June vs annual changes Z by ≤ 0.03 and the N(100–200) ratios by ≤ 8 % (δ = 300); Sun frame lowers the isoscalar
  peak to 370 keV (contact and light alike). The m_A′ floors move 2.56 → 2.75, 0.93 → 1.13, 0.11 → 0.27 GeV (annual → June).
* Factorisation: |ΔZ| ≤ 0.007, rate ≤ 6 %, ratios ≤ 2.4 % (Sec. 2.2).
* δ grid: 10 keV (5 keV at 370–390); peaks quoted to the grid; 68 % ranges interpolated linearly in q₀.
* Efficiency: P038 erf edges; the P021 erf (269.9 keV, σ 11.5) gives Z(380) = 3.59 instead of 3.49 for the contact case
  (P038 documented this); mediator-induced *differences* are insensitive to this choice.
* Thermal vs α_D = 0.1: ε ∝ α_D^−1/2 for both the requirement and the floor; the ratio and the m_A′ floor are α_D-independent.
* One event, E_obs = 248 keV; P038 shows E_obs = 262 keV moves the contact peak to 390 keV — not repeated here.

## 5. Failed or abandoned approaches

* A full WimPyDD q-dependent-coefficient grid for all 7 mediators (≈ 27 000 kernel calls, ~40 min) was abandoned in favour
  of the validated factorisation plus exact spectra for the two lightest mediators at δ = 300 and 380 keV.
* An earlier idea of extracting isotope-resolved kernels to make the factorisation exact was not pursued (WimPyDD's
  `diff_rate` sums isotopes internally); the residual isotope-mix effect was quantified instead.

## 6. Figures

* `figures/P051_fig1_spectra_vs_mmed.png` — true-energy spectra at δ = 300/350/380 keV (isoscalar, 16 June halo) for the
  seven mediators at the same c₀, normalised to the contact peak; 248 keV and the 600 phd edge marked.
* `figures/P051_fig2_Z_vs_delta.png` — Z(δ) for isoscalar and proton-only O₁, annual halo, seven mediators.
* `figures/P051_fig3_mA_eps_plane.png` — (m_A′, ε) plane at α_D = 0.1: ε for one event at δ = 300/350/380 keV; χ₂-decay
  floors (dashed, shaded); BaBar (recalled, likely); beam-dump band (recalled, uncertain, hatched); Planck line (P025).
* `figures/P051_fig4_predictions.png` — N(100–200 keV) per event in 200–270 keV and N(600–1000 phd) per ROI event vs δ.

## 7. Result files

`P051_observables.csv` (714 rows: halo × isospin × δ × m_med), `P051_Z_peaks.csv`, `P051_couplings.csv`,
`P051_exact_vs_factorised.csv`, `factorisation_validation.csv/.json`, `P051_results.json`, `run_log.txt`, kernel caches.

## 8. References

LZ Collaboration, arXiv:2609.02823 (2026) · D. Tucker-Smith, N. Weiner, PRD 64, 043502 (2001) · B. Holdom, PLB 166, 196 (1986) ·
A. L. Fitzpatrick, W. Haxton, E. Katz, N. Lubbers, Y. Xu, JCAP 02 (2013) 004 · N. Anand, A. L. Fitzpatrick, W. C. Haxton, PRC 89,
065501 (2014) · I. Jeong, S. Kang, S. Scopel, G. Tomar, CPC 276, 108342 (2022) (WimPyDD) · D. Baxter et al., EPJC 81, 907 (2021) ·
J. P. Lees et al. (BaBar), PRL 113, 201801 (2014) · J. D. Bjorken, R. Essig, P. Schuster, N. Toro, PRD 80, 075018 (2009) (beam dumps) ·
corpus: P002, P003, P009, P011, P016, P021, P025, P026, P031, P035, P038.

## 9. Tools and provenance (mirrors provenance/P051.json)

Python 3.12.13 (.venv), WimPyDD 2.0.4 (`diff_rate` per-stream kernels, `eft_hamiltonian` incl. q-dependent Wilson
coefficients via closures, `streamed_halo_function` through `lz.wd_halo` with explicit v_min grid), numpy 2.5.3, scipy 1.18.1
(`special.erf/erfc`, `stats.norm`), pandas 3.0.5, matplotlib 3.11.2 (Agg), common/lzcommon.py (`wd`, `wd_halo`,
`wd_hamiltonian`, `wd_c_from_anand`, `E_R_range_keV`, `m_nucleus_gev`, constants). Inputs read: PAPER_GUIDE.md, dossier,
results_ledger.csv, papers P002/P003/P011/P021/P025/P026/P038, scripts P003_nreft_shapes.py, P011_dark_photon_idm.py,
P038_efficiency_edge.py (parts), P038_results.json (edges), P021 result files (conventions), lzcommon.py (WimPyDD wrapper),
WimPyDD/package.py (q-dependence handling, read-only). Recalled items: 11 (see JSON). WimPyDD-generated files: none.
Agent tools: Read ×20, Bash ×18, Write ×4, Edit ×9 (3 script, 6 paper trims), Skill ×1 (dataviz; palette validator not run).
