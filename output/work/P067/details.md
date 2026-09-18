# P067 — Directional and multi-target strategies to confirm a high-energy recoil population: research record

Simulated date 2026-09-12. Category PROJ, hep-ex (cross-list astro-ph.IM). Author profile: directional-detection experimentalists.
Script: `output/code/P067_directional.py` (run from the simulation root with `.venv/bin/python`; 467 s on the first run, 56 s on the
verification rerun; seeded `default_rng(67)`, and the rerun reproduced every number of `P067_angular_cases.csv` exactly). Outputs in `output/work/P067/` (CSV/JSON) and `output/work/P067/figures/` (PNG).
Full console log: `P067_run_log.txt`, `P067_stdout.txt`.

## 1. Motivation and framework

LZ's single 248 keV NR-like event (arXiv:2609.02823) carries no spectral or temporal information; the corpus has so far proposed
counting (P020, P050), annual modulation (P034), a tungsten δ-meter (P046) and isotopically modified xenon (P047) as confirmation
routes. Direction is the one observable not yet examined. A recoil's direction is the most model-independent signature of a
galactic origin (dipole away from Cygnus, Spergel 1988), and — as we show — for *inelastic* scattering near the kinematic ceiling it
becomes a near-collinear beam. But a 248 keV xenon recoil in LXe is a ~0.15 μm track, invisible directionally; directional
sensitivity exists only in low-pressure gas TPCs (mm tracks) and, marginally, in nuclear emulsions (~100 nm). We therefore ask:

(a) which directional target nuclei can scatter at all for the LZ-favoured splittings δ = 300–380 keV (P002, P007, P021),
(b) what the recoil-direction distributions look like for inelastic O₁ and elastic L10 (P012) interpretations and how many events a
    realistic directional detector (30° resolution, 70 % head–tail) needs for a 3σ anisotropy,
(c) where the June 16 recoil would have pointed in the SURF horizon frame,
(d) how directional detection fits into the multi-target decision tree with P034/P046/P047/P050.

Conventions. m_χ = 1 TeV throughout (LZ's best-fitting mass class; P002 shows the inelastic kinematics is mass-insensitive above
~1 TeV). Halo: Baxter-2021 SHM via `lzcommon` (v₀ = 238, v_esc = 544 km/s, Sun peculiar (11.1, 12.2, 7.3) km/s, LSR rotation 238 km/s),
truncated Maxwellian in the Galactic frame. Lab velocity on 16 June 2023 21:22:39 UTC from astropy's barycentric Earth velocity
(as P042): |v_lab| = 265.8 km/s (lzcommon cosine model 265.1), v_max = 809.8 km/s; "Sun frame" = v_lab = v_sun = 250.6 km/s
(the WimPyDD single-day halo; see PAPER_GUIDE halo-labelling note). "Wind direction" ŵ = −v̂_lab is the direction the DM flows
*toward* (anti-apex); recoils point along +ŵ, away from Cygnus.

## 2. Kinematics per target (Part 1; `P067_target_kinematics.csv`, Fig. 1a)

Formulae (lzcommon `delta_max_kev`, `E_R_range_keV`, `vmin_kms`; standard, certain):
- v_min(E_R, δ) = (m_N E_R/μ + δ)/√(2 m_N E_R); inelastic ceiling δ_ceil = μ v_max²/2 (any E_R); δ_max(E_R) = v_max√(2 m_N E_R) − m_N E_R/μ.
- Elastic end-point E_max = 2 μ² v_max²/m_N. Nuclear masses A·amu (error < 0.1 %).

| target | A | δ_ceil (1 TeV, June) [keV] | δ_ceil 0.4 / 4 TeV | δ_max(248 keV) | elastic E_max [keV] | window δ = 300 | window δ = 366 |
|---|---|---|---|---|---|---|---|
| He | 4 | 13.5 | 13.5 / 13.5 | – | 54 | blind | blind |
| C | 12 | 40.3 | 39.1 / 40.6 | – | 160 | blind | blind |
| O | 16 | 53.6 | 51.6 / 54.2 | – | 211 | blind | blind |
| F | 19 | 63.4 | 60.6 / 64.2 | 0.7 (i.e. 248 keV is F's end-point) | **249** | blind | blind |
| S | 32 | 105.6 | 98.2 / 107.7 | 73 | 410 | blind | blind |
| Ca | 40 | 131.1 | 119.9 / 134.3 | 110 | 505 | blind | blind |
| Br | 79.9 | 252.7 | 213.7 / 264.6 | 252 | 941 | blind | blind |
| Ag | 107.9 | **333.2** | 269.4 / 355.0 | 330 | 1211 | 142–524 | **blind** |
| I | 126.9 | 385.7 | 303.4 / 416.0 | 377 | 1380 | 96–747 | 207–518 |
| Xe | 131.3 | 397.8 | 311.0 / 430.4 | 387 | 1418 | 90–793 | 182–583 |
| W | 183.8 | 533.3 | 391.9 / 592.9 | 497 | 1822 | 52–1257 | 88–1108 |

A_min(δ) at 1 TeV, 16 June: A ≥ 79 / 96 / 114 / 120 / 125 for δ = 250 / 300 / 350 / 366 / 380 keV (P015: 96/114/125, reproduced).
Xe δ_max(248 keV) = 387.2 keV reproduces P002 (386.6, cosine v_E). The Sun-frame ceilings are ×0.96 (Xe 376 keV).

Consequences for directional targets:
- Every fluorine, helium, carbon, sulfur, calcium or oxygen directional detector (CYGNUS He:SF₆, DRIFT CS₂/CF₄, DMTPC, NEWAGE CF₄,
  CYGNO He:CF₄, MIMAC, diamond, CaWO₄'s light nuclei) is **kinematically blind** to the inelastic interpretation, at any WIMP mass.
- Silver (A = 107/109) reaches δ ≤ 333 keV at 1 TeV (355 at 4 TeV): NEWSdm emulsions (AgBr) could in principle see δ = 300 keV but not
  the P007/P021 Higgsino window 358–385 keV; bromine (ceiling 253 keV) is blind.
- Only iodine (CF₃I gas), xenon (gas TPC) and tungsten (crystal) are live across the whole preferred window; tungsten has no directional
  readout (crystal-defect direction reconstruction proposals concern light nuclei in diamond; recalled, uncertain).
- Elastic case: fluorine's end-point at 1 TeV is 249 keV on 16 June — a fluorine "analogue" of the 248 keV event is at F's kinematic
  edge, and the L10 (q⁴Σ′) spectrum on F is weighted toward 100–200 keV (§4).

### 2.1 Track lengths (`P067_track_lengths.csv`, Fig. 1b) — ALL anchors recalled, flagged uncertain (factor ~2)

Model R(E) = R₁₀₀ (E/100 keV)^0.8 (SRIM-like exponent, uncertain); gas ranges scale with 1/density (ideal gas at 293 K; densities:
CF₄@40 Torr 0.193, SF₆@20 Torr 0.160, He@740 Torr 0.162, CF₃I@40 Torr 0.429, Xe@40 Torr 0.287, Xe@10 bar 54.6 kg/m³).
Anchors at 100 keV: F in CF₄@40 Torr 2.0 mm (DMTPC ~1 mm at 75 Torr); S in 40 Torr CS₂ 1.5 mm (DRIFT); Xe in LXe 0.07 μm;
Ag in emulsion 0.10 μm (NEWSdm quotes 100 nm ≈ 30 keV C / ~100 keV Ag); W in CaWO₄ 0.03 μm; iodine in CF₃I approximated by Xe-in-Xe
scaled by density.

| medium | R(100 keV) | R(250 keV) | R(300 keV) | directional? (thresholds ~1 mm gas, 100 nm emulsion) |
|---|---|---|---|---|
| F in CF₄ @ 40 Torr | 2.0 mm | 4.2 mm | 4.8 mm | yes |
| F in He:SF₆ 740:20 Torr | 1.2 mm | 2.5 mm | 2.9 mm | yes |
| S in SF₆ @ 20 Torr | 1.5 mm | 3.1 mm | 3.6 mm | yes |
| I in CF₃I @ 40 Torr | 0.47 mm | 0.99 mm | 1.1 mm | marginal (lower pressure helps) |
| Xe in Xe gas @ 40 Torr | 0.71 mm | 1.5 mm | 1.7 mm | yes (marginal) |
| Xe in Xe gas @ 10 bar | 3.7 μm | 7.7 μm | 8.9 μm | no (columnar recombination only; recalled, uncertain) |
| Xe in LXe | 0.07 μm | 0.15 μm | 0.17 μm | no |
| Ag in AgBr emulsion | 0.10 μm | 0.21 μm | 0.24 μm | yes (NIT, 2× above threshold) |
| W in CaWO₄ | 0.03 μm | 0.06 μm | 0.07 μm | no |

So a 250 keV heavy recoil is imageable only in a heavy gas at ≲ 40 Torr (Xe or CF₃I, mm tracks) or as a ~200 nm emulsion grain chain
(Ag). A 40 Torr, 1000 m³ TPC holds 287 kg of xenon or 278 kg of iodine (CF₃I).

## 3. Recoil-direction Monte Carlo (Part 2; `P067_angular_cases.csv`, `P067_kernels.csv`, Fig. 2)

### 3.1 Method
The directional rate is the Radon transform of the lab-frame velocity distribution: dR/(dE_R dΩ_q) ∝ ∫d³v f(v) δ(v·q̂ − v_min(E_R)),
equivalently: for each DM velocity v with |v| > v_min the recoil direction lies on a cone of half-angle θ_R about v̂ with
cos θ_R = v_min/|v| (two-body kinematics; azimuth uniform), and the rate weight is 1/|v| (flux × 1/v² cross-section).
Implementation: importance sampling in lab speed — v uniform in [v_lo, v_max], isotropic direction, weight
w₀ = f(u) v² (v_max − v_lo) with u = v + v_lab (Galactic frame), f the truncated Maxwellian; N_MC = 200 000 per case (reduced from
400 000 at the coordinator's request; statistical precision on ⟨cos γ⟩ ≈ 10⁻³ for elastic cases; for the δ ≥ 366 keV tail cases
the weights are broad and the effective sample is ~10³, so those anisotropies carry ~10–15 % MC uncertainty; the Sun-frame
δ = 366 keV row is the noisiest).
- Fixed-energy cases (E_R = 248 keV): weight w₀/v; v_lo = v_min.
- Spectrum cases: E_R uniform in the allowed window [E₋(v), E₊(v)], weight w₀ (E₊ − E₋)/v × K(E_R), with K the velocity-independent
  nuclear/operator kernel obtained from WimPyDD by passing a "flat halo" (v_min = [0, 3000], δη = [0, 1], i.e. η ≡ 1); validation:
  K(E)·η₀(v_min) reproduces WimPyDD's direct Sun-frame L10 rate on ¹⁹F to 0.2–0.6 % at 10–200 keV. The joint sampler's F spectrum
  matches K·η₀ bin by bin to 1–5 % from 20 to 200 keV (deviations only in the last 30 keV before the 249 keV end-point where the
  rate is ≪ 1 % of the total).
- Kernels: L10 = 4[(q²/m_N²)O₄ − O₆] (P012 construction, isoscalar, c⁰_WD = 8/m_v² and −8/m_v²), on ¹⁹F and natural Xe;
  O₁ isoscalar with δ = 366 keV on Xe (WimPyDD shell-model responses).
- γ = angle between the recoil direction and ŵ (anti-apex). Statistics: ⟨cos γ⟩, forward–backward asymmetry A_FB = ⟨sign cos γ⟩,
  fraction within 30° and 60°, median and 90 % γ, median and maximum θ_R.
- Detector response: each direction rotated by |N(0, 30°)| about a random perpendicular axis, then the sense flipped with
  probability 0.30 (head–tail efficiency 0.70). Both numbers are assumptions typical of gas TPCs at 50–250 keV (recalled, uncertain).
- N_3σ (Gaussian, one-sided 3σ against isotropy): dipole statistic S = mean cos γ (null var 1/(3N)): N₃ = 3/⟨cos γ⟩_obs²; counting
  A_FB: N₃ = 9/A_obs²; axial statistic (no head–tail needed) T = mean cos²γ − 1/3 (null var 4/(45N)): N₃ = 0.8/T_obs².
  Toy check: for N events resampled from the smeared distribution, power P(S > S_crit) with S_crit the 99.865th percentile of
  10⁵ isotropic toys (1500 signal toys per N).
- Isotropic control: ⟨cos γ⟩ = 0.001, A_FB = 0.004 (as expected).

### 3.2 Analytic bound on the recoil angle
At fixed E_R, cos θ_R = v_min/|v| ≥ v_min/v_max. For Xe, E_R = 248 keV, 1 TeV: v_min = 339 / 704 / 765 / 784 / 802 km/s for
δ = 0 / 300 / 350 / 366 / 380 keV, so θ_R ≤ 65.3° / 29.7° / 19.2° / 14.5° / 8.4° (June, v_max = 809.8). Iodine: 28.0° / 16.4° / 10.4° at
δ = 300 / 350 / 366 (δ = 380 forbidden on I in June: v_min = 814 > 810). Near the ceiling the recoil is collinear with the DM
velocity, and the qualifying DM particles (u within ~30° of anti-v_lab for v > 784 km/s) are themselves nearly anti-parallel to the
lab motion, so the recoils form a narrow beam along ŵ.

### 3.3 Results (16 June frame unless stated)

| case | ⟨cos γ⟩ | A_FB | f(γ<30°) | γ₅₀ / γ₉₀ | θ_R med/max | ⟨cos⟩_obs | N₃ dipole (obs) | N₃ FB (obs) | N₃ axial (obs) | N₃ ideal dipole |
|---|---|---|---|---|---|---|---|---|---|---|
| Xe E=248, elastic | 0.594 | 0.884 | 0.22 | 48° / 82° | 37.6° / 65.2° | 0.209 | 69 | 91 | 124 | 8.5 |
| Xe E=248, δ=300 | 0.887 | 1.000 | 0.65 | 25° / 41° | 15.0° / 29.5° | 0.314 | 30 | 57 | 8.1 | 3.8 |
| Xe E=248, δ=350 | 0.947 | 1.000 | 0.96 | 17° / 27° | 10.1° / 19.1° | 0.346 | 25 | 53 | 5.4 | 3.3 |
| Xe E=248, δ=366 | 0.969 | 1.000 | 1.00 | 13° / 21° | 7.7° / 14.5° | 0.352 | 24 | 53 | 4.6 | 3.2 |
| Xe E=248, δ=380 | 0.989 | 1.000 | 1.00 | 8° / 12° | 4.5° / 8.3° | 0.364 | 23 | 50 | 4.3 | 3.1 |
| I E=248, δ=300 / 350 / 366 | 0.896 / 0.961 / 0.983 | 1 | 0.68 / 1 / 1 | 24/39°, 15/23°, 10/15° | – | 0.316 / 0.350 / 0.330 | 30 / 24 / 28 | 57 / 53 / 64 | 7.7 / 5.1 / 4.1 | 3.7 / 3.2 / 3.1 |
| Xe O₁ δ=366, full window (E₅₀ = 340 keV, 10–90 % 218–410) | 0.966 | 1.000 | 1.00 | 13° / 22° | 7.8° / 15.8° | 0.337 | 26 | 56 | 4.8 | 3.2 |
| Xe O₁ δ=366, 200–270 keV | 0.977 | 1.000 | 1.00 | 12° / 18° | 6.9° / 14.9° | 0.254 (noisy) | 47 (noisy) | 107 | 4.6 | 3.1 |
| F L10 elastic, E > 20 keV (E₅₀ = 81 keV, 10–90 % 38–144) | 0.700 | 0.951 | 0.34 | 39° / 71° | 27.7° / 72.9° | 0.245 | 50 | 75 | 32 | 6.1 |
| F L10, E > 50 keV | 0.730 | 0.976 | 0.36 | 37° / 67° | 25.8° / 61.9° | 0.243 | 51 | 77 | 27 | 5.6 |
| F L10, 150–250 keV | 0.863 | 1.000 | 0.59 | 27° / 45° | 16.3° / 38.0° | 0.262 | 44 | 78 | 9.9 | 4.0 |
| Xe L10 elastic, 200–270 keV | 0.576 | 0.863 | 0.22 | 50° / 84° | 39.0° / 67.1° | 0.210 | 68 | 100 | 153 | 9.0 |
| Xe L10 elastic, E > 5.4 keV | 0.600 | 0.870 | 0.24 | 47° / 83° | 35.0° / 86.1° | 0.211 | 67 | 93 | 87 | 8.3 |
| Sun frame, Xe E=248: elastic / δ=300 / 350 / 366 | 0.581 / 0.889 / 0.957 / 0.981 | | | 49/84°, 25/40°, 15/25°, 10/17° | max 64.8 / 27.6 / 15.8 / 9.3° | 0.205 / 0.312 / 0.317 / 0.286 | 72 / 31 / 30 / 37 | | 138 / 7.9 / 5.1 / 4.4 | |

Reading: (i) inelastic recoils near the ceiling are a beam — at δ = 366 keV *all* 248 keV recoils lie within 21° of the wind
axis, at δ = 380 keV within 12°; a perfect detector would need only ~3 events (ideal dipole) or, with no head–tail information at
all, ~5 events (axial statistic) for 3σ. (ii) With a 30°/70 % detector the head–tail flips dominate: ⟨cos⟩_obs ≈ (2ε−1)·e^{−σ²/2}·⟨cos⟩
≈ 0.35 and N₃ ≈ 23–30 for every δ ≥ 300 keV; the analytic estimate 24.8/⟨cos γ⟩² gives 31/28/26/25 for δ = 300/350/366/380 keV,
consistent with the MC to ~10 %. (iii) Elastic distributions are the familiar broad dipole (⟨cos γ⟩ = 0.58–0.70): N₃ = 50 (fluorine,
L10) to 69 (xenon, 248 keV) with the realistic detector, 6–9 ideal. (iv) The Sun-frame results differ from June by ≤ 5 % in ⟨cos γ⟩;
the maximum recoil angle shrinks (v_min/v_max larger) and δ = 380 keV becomes forbidden (v_min 801 > 795 km/s), as in P002.
(v) Toy check (median discovery): δ = 366 fixed-E power 0.44 at N = 20, 0.62 at 30 → N₃ ≈ 23 (analytic 24); F L10 0.54 at 50 → ≈ 47
(50); Xe elastic 0.37 at 50, 0.59 at 80 → ≈ 68 (69). Gaussian formulae are adequate.

Fig. 2: `figures/P067_fig2_angular.png` — (a) true dR/dcos γ for Xe 248 keV at δ = 0/300/366/380 and the F L10 spectrum (log scale;
the inelastic cases are confined to cos γ > 0.6/0.9/0.98); (b) the same after 30° smearing and 30 % sense flips (linear scale).

## 4. SURF geometry at the event time (Part 3; `P067_apex_altaz_16June2023.csv`, Fig. 3)

astropy's AltAz frame is unusable offline (leap-second/IERS configuration error, as in P042), so the horizon transform is analytic:
GMST = 18.697374558 + 24.06570982441908 D h (USNO, certain), LST(SURF, 103.751° W) = 8.120 h; alt/az from the hour angle with
latitude 44.352° N (SURF coordinates recalled, likely). Galactic → ICRS via astropy SkyCoord (fixed rotation, works offline).
- Apex (direction the lab moves toward = where DM comes from): RA 319.4°, Dec +43.7° (l = 87.4°, b = −3.9°), 5.1° from the
  textbook "Cygnus" RA 20h50m/Dec +45°. At 21:22:39 UTC: **altitude −0.5°, azimuth 347°** (NNW, on the horizon; P042: +0.5° for the
  anti-apex, reproduced).
- Anti-apex (where recoils go): RA 139.4°, Dec −43.7°; **altitude +0.5°, azimuth 167°** (SSE).
- Over 16 June 2023 the apex altitude ranges −1.9° to +89.4° (it culminates near the zenith at 10.6 UTC and dips below the horizon
  for 11 % of the day; |alt| < 20° for 42 % of the day). The event fell within 40 minutes of the wind being exactly horizontal.
- Expected recoil direction of the event (MC directions rotated into the horizon frame):
  inelastic δ = 366 keV: altitude +1° (10–90 %: −13° to +14°), azimuth 167° ± 11° (68 % half-width), upward fraction 0.52;
  elastic Xe 248 keV: altitude 0° (−46° to +46°), azimuth 167° ± 49°; F L10: 0° (−40° to +40°), 167° ± 38°.
  Under the inelastic interpretation the 248 keV track would have been a horizontal ~150 nm xenon recoil heading SSE — a "head–tail"
  prediction no xenon TPC can test, but which a heavy-gas directional detector in the same laboratory would have recorded as a
  horizontal mm-scale track.

Fig. 3: `figures/P067_fig3_surf_geometry.png` — (a) apex altitude vs UTC on 16 June with the event time; (b) recoil-direction density
in (az, alt) for δ = 366 keV (orange) over the elastic distribution (grey), with apex and anti-apex marked.

## 5. Rates, exposures and the decision tree (Part 4; `P067_decision_table.csv`)

### 5.1 L10 on fluorine (own WimPyDD computation)
Unit-coupling L10 (P012 construction) on natural Xe with the Sun-frame halo and a logistic LZ efficiency (50 % at 5.4 and 269.9 keV,
96 % plateau; widths 1.5/8 keV assumed) gives 3.32 events in 2.84 t·yr (P012 "ours": 3.34) → LZ's one event ⇔ scale 0.301.
At that coupling the ¹⁹F rate is 0.150 / 0.146 / 0.114 / 0.044 / 0.009 events per **tonne**·yr of fluorine above 10 / 20 / 50 / 100 /
150 keV (F/Xe per tonne = 0.36 above 20 keV; the xenon rate above 5.4 keV without efficiency is 0.41 per t·yr). The F spectrum
(q⁴Σ′ on a J = 1/2 nucleus) has median 81 keV (10–90 %: 38–144 keV) and ends at 249 keV.
Example directional volumes (recalled designs, uncertain): 1000 m³ He:SF₆ at 740:20 Torr → 125 kg F; 1000 m³ CF₄ at 40 Torr → 166 kg F.
Event rates 0.018 / 0.024 per year (E > 20 keV). With N₃ = 50: **2700 / 2050 years**, i.e. ≈ 340 t·yr of fluorine (a 100 000 m³-class
volume for a century). A directional confirmation of the L10 interpretation is therefore out of reach; the F recoil spectrum
itself (median 81 keV, no events above 249 keV) would however be a distinctive counting signature if a tonne-scale fluorine
detector with a 20 keV threshold existed (≈ 0.15 events per t·yr).

### 5.2 Inelastic on heavy gas / emulsions (rates from P015/P046/P050 fractions, kinematics ours)
Xenon full-window rate at LZ's fit = 1/(2.84 t·yr × f_ROI) with f_ROI = 0.82/0.45/0.17/0.028 (P015): 0.43 / 0.78 / 2.07 / 12.6 per t·yr
for δ = 300/350/366/380 keV (P050's 400 keV-window rates 0.40/0.69/1.70/10.5 are 83–93 % of these). I/Xe = 0.59/0.30/0.14/0.023.

| detector (1000 m³, 40 Torr) | δ = 300 | 350 | 366 | 380 |
|---|---|---|---|---|
| Xe gas, 287 kg: events/yr | 0.12 | 0.22 | 0.60 | 3.6 |
| years to N₃ = 30/25/24/23 | 246 | 111 | **41** | **6** |
| CF₃I, 278 kg I: events/yr | 0.070 | 0.065 | 0.081 | 0.080 |
| years to N₃ | 431 | 384 | 301 | 282 |

Emulsion (NEWSdm-like, 100 kg·yr; AgBr 78 % of the gel mass → 45 kg Ag): Ag/Xe per tonne at δ = 300 keV = 0.099 (Helm form factors
on both, June halo; 0.27 at 250 keV, 0.019 at 320 keV) → 1.9 × 10⁻³ events per 100 kg·yr; blind for δ > 333 keV. Not viable.
Note δ = 380 keV, the only case where a 1000 m³ xenon-gas TPC would work within a decade, is the splitting for which LZ's own empty
600–1000 phd region already implies 1–30 unseen events (P038, P050) — the fastest directional route is the least likely splitting.

### 5.3 Decision table
| step | strategy | 3σ/5σ exposure | events | date / duration | source |
|---|---|---|---|---|---|
| 1 | keep counting, present generation (LZ-like ROI) | 5σ at best fit 12.3 (L10)–5.7 (δ380) t·yr | 2–4 | 2027–28; LZ already holds 6.8 t·yr untouched | P050, P020 |
| 1′ | extend ROI to 400 keV | acceptance 0.83–0.93 (δ ≥ 350) | 2–4 | by ~2030 | P050, P038 |
| 2 | 60 t xenon, 400 keV ROI | 5σ 8.5/9.6/3.6/1.1/0.13 t·yr | 4–6 for L10-vs-inelastic shape | 2032 + 1 day–2 months | P050 |
| 2′ | annual modulation | 3σ 274/84/33/19/15 t·yr (δ 300–380) | 97–5 | LZ 2029–30 only if δ ≥ 366; 60 t 0.3–5.7 yr | P034 |
| 3a | CaWO₄ δ-meter (if inelastic) | 100 kg·yr + 20 t·yr Xe: σ_δ 99/10/3.5/0.8 keV | 3 (2.1 kg·yr for L10 vs δ366) | few years | P046, P015 |
| 3b | ¹³⁶Xe-enriched xenon | 12.8 t·yr per detector | 9 (5 at δ = 380) | 2030s | P047 |
| × | NaI modulation | 2 × 10⁸–8 × 10⁹ t·yr | – | excluded route | P028 |
| 4 | directional heavy-gas TPC (Xe/CF₃I, 1000 m³) | 30–23 events | 23–30 | 6 yr (δ = 380) … 250–430 yr | this work |
| × | directional F/He/S TPCs | kinematically blind (inelastic); L10: 50 events | 50 | 2000–2700 yr per 1000 m³ | this work |
| × | nuclear emulsion (Ag) | ceiling 333 keV; 2 × 10⁻³ events per 100 kg·yr | – | not viable | this work |

Tree: (1) xenon counts decide existence by 2027–30; (2) shape/isotopes/tungsten decide L10 vs inelastic and read δ with 4–9 events;
(3) direction adds a model-independent galactic signature only if a ≳ 1 t heavy-gas directional volume exists, and then only for
δ ≳ 366 keV; otherwise timing (P034) remains the sole astrophysical signature.

## 6. Validation and robustness
- Kinematic map reproduces P002 (δ_max 387 keV), P015 (A_min 96/114/125; ceilings 63/385/396/533 keV for F/I/Xe/W).
- Kernel trick vs direct WimPyDD: 0.2–0.6 %; MC spectrum vs K·η₀: 1–5 % (20–200 keV).
- L10 normalisation 3.32 vs P012's 3.34 events at unit coupling.
- Isotropic control: anisotropies < 0.005.
- Toy N₃ agree with Gaussian formulae to ≤ 10 %.
- Apex altitude −0.5° / anti-apex +0.5° reproduces P042; LST 8.12 h identical.
- Frame dependence June vs Sun: ≤ 5 % in ⟨cos γ⟩, ≤ 25 % in N₃ for the noisiest tail case.
- MC precision: tail cases (δ ≥ 366 keV, n_eff ~ 10³) ±10–15 %; the "200–270 keV, δ = 366" observed statistic is noisy (N₃ 47 vs 26
  for the full window; the true anisotropy differs by only 1 %), and we quote the full-window value.
- Detector assumptions: N₃(obs) ∝ 1/[(2ε−1)² e^{−σ²}]: ε = 0.6/0.8 → ×4 / ×0.44; σ = 20°/45° → ×0.84 / ×1.6.

## 7. Failed or abandoned approaches
- astropy `AltAz` transform fails offline (leap-second auto-update TypeError) → analytic GMST horizon transform.
- Plain rejection sampling of the halo starved at δ ≥ 366 keV (114 of 400 000 samples above v_min) and would have crashed the later
  look-ups; replaced by importance sampling in lab speed (first run aborted).
- First foreground run exceeded the 600 s limit because of pipeline buffering (not the script); rerun unbuffered with N_MC halved.
- Decision-table note for fluorine rows initially printed a mass ratio instead of the years factor; corrected and rerun (seeded, identical numbers).

## 8. Figures
- Fig. 1 `figures/P067_fig1_ceiling_and_tracks.png`: (a) inelastic ceiling μv_max²/2 vs A at 0.4/1/4 TeV with δ = 300/366/380 keV
  lines and the eleven targets; (b) recoil range at 250 keV per medium (recalled anchors) against the gas-TPC (1 mm) and emulsion (100 nm) thresholds.
- Fig. 2 `figures/P067_fig2_angular.png`: recoil-direction distributions (true; smeared + head–tail).
- Fig. 3 `figures/P067_fig3_surf_geometry.png`: SURF geometry on 16 June 2023 and the predicted recoil direction.

## 9. References
LZ Collaboration, arXiv:2609.02823 (2026). D. N. Spergel, PRD 37, 1353 (1988). S. E. Vahsen et al. (CYGNUS), arXiv:2008.12587 (2020).
F. Mayet et al., Phys. Rept. 627, 1 (2016). D. Tucker-Smith, N. Weiner, PRD 64, 043502 (2001). D. Baxter et al., EPJC 81, 907 (2021).
I. Jeong et al. (WimPyDD), CPC 276, 108342 (2022). N. Anand, A. L. Fitzpatrick, W. C. Haxton, PRC 89, 065501 (2014).
Corpus: dossier 00, P002, P012, P015, P020, P028, P034, P038, P042, P046, P047, P050.

## 10. Tools and provenance (mirrors `output/provenance/P067.json`)
Software: python 3.12.13 (.venv), numpy 2.5.3, scipy (via lzcommon), pandas 3.0.5, matplotlib 3.11.2 (Agg), astropy 8.0.1
(Time, get_body_barycentric_posvel, SkyCoord Galactic↔ICRS), WimPyDD 2.0.4 (eft_hamiltonian with q-dependent c₄, diff_rate via
lz.wd_rate on F and Xe, streamed_halo_function via lz.wd_halo; "flat halo" kernel extraction), common/lzcommon.py (LZ dict,
kinematics, eta0, dRdE_SI, wd_* wrappers). No WimPyDD files generated (diff_rate only). Recalled items: 21 = 19 registered by the script
(`P067_summary.json` → `recalled`) + 2 method items (Radon-transform/statistics formulae, columnar-recombination and crystal-defect
proposals) listed in the JSON; 6 certain, 6 likely, 9 uncertain (track-length anchors, detector designs, resolution/head–tail).
Local inputs: PAPER_GUIDE, dossier, ledger, P002/P015/P028/P034/P042/P046/P047/P050 papers, P042 and P012 and P015 scripts (method
snippets), lzcommon.py, ENVIRONMENT_versions.txt. Data requests: none.
