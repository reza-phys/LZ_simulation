# P085 — Earth's velocity on 16 June 2023 21:22:39 UTC and gravitational focusing: the exact kinematic inputs for the LZ event

Simulated date 2026-09-16 · Category HALO · astro-ph.GA (cross-list hep-ph) · Author profile: astrophysics / celestial-mechanics group.
Builds on P006 (date), P018 (tail), P055 (from-scratch Earth velocity, 265.96 km/s), P034 (modulation), P068 (astro band), P007 (Higgsino window, 366 keV annual), P035 (Sun-frame labelling).

Script: `output/code/P085_earth_velocity.py` (96 s wall time; foreground). Tables in `output/work/P085/`, figures in `output/work/P085/figures/`, full numeric log in `P085_log.json`, stdout in `run_stdout.txt`, final input table in `P085_kinematic_inputs.json`.

## 1. Motivation and framework

Every inelastic reading of the LZ event (E_R = 248 keV at 21:22:39 UTC on 16 June 2023; LZ paper §"Results", line 165 of the fulltext) lives within ~100 km/s of the kinematic edge v_max = v_esc + |v_E|, and δ_max(248 keV) moves by 0.82 keV per km/s of v_max (P018). The corpus so far used three approximate Earth-velocity models (P006: agreement to 1.9 km/s; P055: circular orbit, 265.96 km/s; wimprates 265.98; WimPyDD 265.92; lzcommon cosine 265.06). We compute the observer velocity at the exact event time with a real ephemeris (astropy 8.0.1, built-in ERFA `epv00`-based ephemeris, fully offline), add the rotation of the Earth at SURF, quantify the solar-motion uncertainty, and then treat two effects nobody in the corpus has included: gravitational focusing by the Sun (Liouville mapping of the halo velocity distribution through the solar potential) and by the Earth, on the density, on the high-velocity tail η(v_min), on the hard edge, and on the inelastic rate near δ_max. Finally we give the fraction of the year for which a near-edge interpretation is kinematically open.

Conventions: Galactic Cartesian (U toward the Galactic centre, V along rotation, W toward the NGP) = wimprates' and WimPyDD's axes. Baxter-2021 SHM from lzcommon: v₀ = 238, v_esc = 544 km/s, **v**_pec = (11.1, 12.2, 7.3) km/s, ρ₀ = 0.3 GeV cm⁻³. Rates: WimPyDD 2.0.4 per-stream kernels (O₁ Higgsino Z-exchange couplings of P007, c_p = (G_F/√2)(1 − 4 sin²θ_W), c_n = −G_F/√2; WimPyDD c⁰ = c_p + c_n, c¹ = c_p − c_n), m_χ = 1 TeV, LZ efficiency model of P007 (plateau 0.96, 50 % at 5.4 and 269.9 keV, erf edges σ = 2.5/11.5 keV), exposure 2.84 t·yr, E ≤ 330 keV in 3 keV steps. Every *ratio* below is coupling-independent.

## 2. Part A — the observer velocity at the event time

### 2.1 Ephemeris

`Time('2023-06-16T21:22:39', scale='utc')`: JD_UTC 2460112.39073, JD_TDB 2460112.39153, day of year 167.891 (1 Jan 00:00 = 1.0), UT1 − UTC = −0.044 s (bundled IERS-B; `auto_download = False`). `get_body_barycentric_posvel('earth')` − `('sun')` with `solar_system_ephemeris = 'builtin'` gives the heliocentric Earth velocity, ICRS (29.2097, −2.4135, −1.0474) km/s, |v| = **29.328 km/s** at r = **1.01589 AU** (17 days before aphelion, hence 0.46 km/s below the mean 29.79). Rotated to Galactic with astropy's ICRS→Galactic frame: **v**_orb = (1.0119, 14.7240, −25.3438) km/s; the recalled J2000 equatorial→Galactic matrix (Hipparcos; certain) reproduces this to 3 × 10⁻⁶ km/s. The Earth–EMB velocity difference (lunar term) is 0.0129 km/s and is included. The Sun's apparent ecliptic longitude is 85.480° (WimPyDD's day-based approximation uses 86.687°).

Ephemeris accuracy: the ERFA built-in ephemeris is quoted (recalled, likely) at ≲ 5 km in position and ≲ 1 mm/s in velocity for 1900–2100 — five orders below anything relevant here. No network was needed (no JPL kernel download attempted).

### 2.2 Rotation at SURF

`EarthLocation(lat = 44.352°, lon = −103.751°, h = +100 m)` (Lead, SD; recalled, likely; the lab is 4850 ft below a ~1600 m surface — the height is irrelevant at the 10⁻⁵ level), `get_gcrs_posvel`: **v**_rot = (0.1673, −0.0624, 0.2812) km/s (Galactic), |v_rot| = **0.333 km/s** (0.465 cos 44.35°). Local apparent sidereal time 8.120 h (15:22 local MDT). The rotation velocity makes an angle of **102.7°** with **v**_E, so its projection is **−0.074 km/s**: at 21:22 UTC the lab was moving slightly *against* the Earth's Galactic motion (the apex was on the northern horizon, see below). δ_max shift from rotation: **−0.061 keV**.

### 2.3 Solar motion and totals (`observer_velocity_variants.csv`)

**v**_E = (0, v₀, 0) + **v**_pec + **v**_orb + **v**_rot.

| solar motion | |v_⊙| | **v**_E (U, V, W) km/s | |v_E| with rotation | without | apex (l, b) | v_max(544) |
|---|---|---|---|---|---|---|
| **Baxter: v₀ = 238, pec (11.1, 12.2, 7.3)** | 250.552 | **(12.279, 264.862, −17.763)** | **265.740** | 265.814 | (87.35°, −3.83°) | **809.74** |
| Schönrich pec (11.1, 12.24, 7.25), v₀ = 238 | 250.591 | (12.279, 264.902, −17.813) | 265.784 | 265.857 | (87.35°, −3.84°) | 809.78 |
| v₀ = 220 | 232.580 | (12.279, 246.862, −17.763) | 247.804 | 247.878 | (87.15°, −4.11°) | 791.80 |
| v₀ = 230 | 242.564 | | 257.768 | 257.841 | | 801.77 |
| v₀ = 233 (SHM++) | 245.560 | | 260.757 | 260.831 | | 804.76 |
| v₀ = 250 | 262.536 | (12.279, 276.862, −17.763) | 277.702 | 277.775 | (87.46°, −3.67°) | 821.70 |

Reference value: **|v_E| = 265.74 km/s** (265.81 without rotation). The orbital velocity projects +14.01 km/s onto **v̂**_⊙. The mean DM arrival direction (the "wind") is −**v̂**_E = (l, b) = **(267.35°, +3.83°)**; the apex is at (RA, Dec) = (319.35°, +43.76°), in Cygnus, and at the event instant it stood at **altitude −0.38°, azimuth 347.3°** at SURF — exactly on the northern horizon, so the mean DM flux was horizontal, from the north. Solar apex of **v**_⊙ alone: (87.46°, +1.67°).

Uncertainty (linear propagation of recalled σ(v₀) = 1.5 km/s, σ(pec) = (1.2, 2.0, 0.6) km/s; Baxter 2021 / Schönrich 2010; uncertain): **σ(|v_E|) = 2.5 km/s** (1.5 from v₀ alone), i.e. **σ(δ_max) = 2.0 keV**; Schönrich vs Baxter peculiar velocity: +0.04 keV; v₀ = 220/250: −14.7/+9.8 keV.

### 2.4 Comparison with the corpus models (`P085_log.json`)

| model | |v_E|(event) | difference from this work |
|---|---|---|
| **this work (astropy, incl. rotation)** | **265.74** | — |
| this work, no rotation | 265.81 | +0.07 |
| wimprates 0.5.0 `v_earth(j2000 = 8567.391)` (circular, LLS e₁/e₂, 29.8 km/s) | 265.98 | +0.24 |
| WimPyDD 2.0.4 `v_earth_sun(λ = 86.69°)` (circular, 29.79 km/s) | 265.92 | +0.18 |
| P055 (from scratch, circular) | 265.96 | +0.22 |
| P006 | 266 | +0.3 |
| lzcommon cosine 250.55 + 15 cos(2π(t − 153)/365.25) | 265.06 | −0.68 |
| WimPyDD Sun frame (`wd_halo()` with no day; P035) | 250.55 | −15.19 |

Orbital-vector differences: astropy − wimprates = (0.88, −0.17, 0.47) km/s (|Δ| = 1.0 km/s, mostly the 0.46 km/s aphelion deficit plus the e₁/e₂ approximation); astropy − WimPyDD = (1.29, −0.13, 0.48). All corpus values are high by 0.2–0.3 km/s (0.15–0.25 keV in δ_max) — immaterial for every published conclusion but now pinned.

### 2.5 The annual curve (hourly astropy scan, 27 Mar 2023 – 1 Apr 2024; `vE_run_window_6h.csv`, Fig. 1)

- Maximum **266.19 km/s at 2 June 2023 10:00 UTC (±1 h grid; day 153.4)**; minimum 237.19 km/s at 4 Dec 2023 10:00 UTC. The event is **14.47 d after the maximum, 0.45 km/s below it** (δ_max 0.37 keV below its annual maximum 387.50 keV).
- Annual mean speed (365.25 d from 27 Mar 2023): **252.11 km/s**; run-window mean 252.21. This is 1.56 km/s *above* the Sun-frame 250.55 (the perpendicular orbital component adds ⟨v_⊥²⟩/2v_⊙); the WimPyDD Sun-frame halo (P035) is therefore neither the June nor the annual value.
- First harmonic: 252.11 + 14.49 cos(2π(t − 154.6)/365.25) km/s, rms residual 0.31 km/s (the fitted phase lags the true maximum by 1.2 d because the eccentric orbit makes the curve non-sinusoidal); lzcommon's (250.55, 15.0, 153) is off by −1.6 km/s in level.

## 3. Part B — kinematics at the event (`kinematics_event.csv`)

δ_max(E_R, m) = (v_max/c) √(2 m_N E_R) − m_N E_R/μ, E_R = 248 keV, m_N = 122.3 GeV (A = 131.29), dδ_max/dv_max = **0.8218 keV per km/s**.

| m_χ | v_esc | v_max | δ_max event (with/without rotation) | δ_max annual mean (252.11) | Sun frame (250.55) | at v_E max | E_max elastic | E_max (δ = 366) | E_min (δ = 366) |
|---|---|---|---|---|---|---|---|---|---|
| 0.4 TeV | 544 | 809.74 | **341.60** / 341.66 | 330.39 | 329.12 | 341.97 | 1047 keV | — | — |
| 1 TeV | 500 | 765.74 | 350.96 | 339.76 | 338.48 | 351.33 | 1268 | — | — |
| 1 TeV | 528 | 793.74 | 373.98 | 362.77 | 361.49 | 374.35 | 1362 | 495.1 | 214.8 |
| **1 TeV** | **544** | **809.74** | **387.13** / 387.19 | **375.92** | 374.64 | **387.50** | **1417.5** | **582.9** | **182.4** |
| 1 TeV | 560 | 825.74 | 400.27 | 389.07 | 387.79 | 400.65 | 1474 | 661.1 | 160.9 |
| 1 TeV | 580 | 845.74 | 416.71 | 405.51 | 404.23 | 417.08 | 1546 | 753.0 | 141.2 |
| 1 TeV | 600 | 865.74 | 433.15 | 421.94 | 420.67 | 433.52 | 1620 | 841.9 | 126.3 |
| 4 TeV | 544 | 809.74 | **409.89** / 409.95 | 398.69 | 397.41 | 410.26 | 1681 | 816.5 | 154.5 |

P006/P007 quoted 387.3 keV (June), 375.9 (mean), 363.8 (December); our exact values are 387.13 (event), 375.92 (mean), 363.7 (minimum, 4 Dec). The 0.2 keV June difference is the 0.2 km/s of §2.4.

## 4. Part C — gravitational focusing by the Sun

### 4.1 Formalism

Liouville: the phase-space density is conserved along orbits, f_E(**v**) = f_∞(**v**_∞(**v**)), where **v** is the Sun-frame velocity at the Earth's position **r** and **v**_∞ the asymptotic incoming velocity of the Kepler hyperbola through (**r**, **v**). With u = |**v**_∞| = √(v² − 2GM_⊙/r):

  **v**_∞ = [u² **v** + u (GM_⊙/r) **r̂** − u (**v**·**r̂**) **v**] / [u² + GM_⊙/r − u (**v**·**r̂**)]

(Alenazi & Gondolo 2006; Lee, Lisanti, Peter & Safdi 2014 — recalled, likely). **Verification:** six random (r = 1 AU, v = 150–800 km/s) states integrated backwards with `solve_ivp` (DOP853, rtol 10⁻¹¹) to r = 6000–16000 AU agree with the formula to ≤ 4 × 10⁻⁴ km/s in vector and 0.000° in direction (`focusing_map_validation.json`); the formula reduces to **v** for GM → 0 and gives the correct 180° swing for radial orbits. GM_⊙ = 1.327124 × 10¹¹ km³ s⁻² (certain); v_esc,⊙(1.0159 AU) = **41.79 km/s**.

Galactic-frame velocity at infinity **v**_gal = **v**_∞ + **v**_⊙; f_gal(**v**_gal) = N exp(−v_gal²/v₀²) Θ(v_esc − |**v**_gal|), N = [π^{3/2} v₀³ N_esc]⁻¹ (unit density at infinity). Lab-frame velocity **w**: **v** = **w** + **v**_orb(t) (Sun focusing acts on the heliocentric velocity; the Earth's rotation is added to the observer velocity but not to the heliocentric mapping). Then n(t)/n_∞ = ∫ f d³w, η(v_min) = ∫_{w > v_min} f/w d³w, and the speed density f_w(w) = w² ∮ f dΩ.

Numerics: speeds 0–1100 km/s in 2 km/s steps; angular quadrature 320 Gauss–Legendre nodes in cos θ × 48 uniform φ, polar axis along the wind arrival direction; 0.58 s per distribution. **Grid validation** (unfocused vs analytic `lz.eta0`, event day): η ratio 1.0000 at v_min = 0–600, 1.0004/1.0005/1.0010/1.008/0.993 at 700/750/780/790/800 km/s; density 1.000002. Hard edges are resolved separately by bisection along directions within 12° of the wind axis (`exact_edge`, 10⁻⁹ km/s), because the 2 km/s grid cannot resolve the ~1 km/s focusing shift. WimPyDD Δη arrays are built as Δη_i = η(v_{i−1}) − η(v_i) on `wd_halo`'s 1200-point grid (sum check 0.0033581 vs WimPyDD 0.0033567 vs analytic 0.0033581). WimPyDD's own June η is 0.8/1.2/2.0/4.7 % high at 600/700/750/790 km/s (its bin averaging; P018/P055 found the same), so all focused/unfocused ratios use our arrays on both sides.

### 4.2 Results at the event time and through the year (`focusing_through_year.csv`, Fig. 2)

The Earth's heliocentric position makes an angle of **96.6°** with the downstream axis (−**v̂**_⊙) at the event; over the year the angle ranges only **60.7° (≈ 5 March) to 119.4° (≈ 5 September)** — the ecliptic is inclined ~60° to the solar apex, so the Earth is never truly "behind" the Sun in the wind.

| quantity | 16 June 21:22 | ≈ 5 March (max) | ≈ 5 Sept (min) | annual mean (36 dates) |
|---|---|---|---|---|
| n/n_∞ − 1 | **+0.84 %** | +1.95 % | +0.58 % | +1.11 % |
| η ratio focused/unfocused, v_min = 600 km/s | **1.0216** | 1.0232 | 1.0227 | 1.0226 (ratio of annual means 1.0226) |
| 700 | **1.0308** | 1.0327 | 1.0363 | 1.0339 (1.0333) |
| 750 | **1.0486** | 1.0577 | 1.0602 | 1.0613 (1.0560) |
| 790 | **1.118** | 1.30 | 1.46 | (finite on 23 dates; ratio of annual means 1.155) |
| hard edge v_max, unfocused → focused | **809.74 → 810.84 km/s** | 797.13 → 798.26 | 795.93 → 797.04 | shift 1.09–1.13 km/s all year |

The edge shift is the analytic √(v_max² + v_esc,⊙²) − v_max = 1.10 km/s: **δ_max(248 keV, 1 TeV) rises by 0.90 keV, 387.13 → 388.03 keV** (0.4 TeV: +0.90; 4 TeV: +0.90 — the shift is mass-independent at fixed E_R since dδ_max/dv is). The tail enhancement is dominated by the speed boost (dlnη/dv ≈ −2(v − v_E)/v₀² ≈ −1.5 % per km/s at 700 km/s, ×1.1 km/s ≈ +1.7 %, plus the density term), *not* by the direction of the Earth relative to the wind; hence it is nearly date-independent, while the density enhancement varies ×3.4 through the year. Deflection angle for a 750 km/s particle at 1 AU ≈ v_esc,⊙²/v² ≈ 0.18° — negligible against P055's 8–12° directional cones.

### 4.3 Effect on the inelastic rate (WimPyDD kernels, 1 TeV Higgsino; `higgsino_rates_1TeV.csv`, Fig. 3)

N(δ) in 2.84 t·yr:

| δ (keV) | Sun frame (WimPyDD) | annual (WimPyDD, 24 d) | annual, ours, unfocused / focused | 16 June (WimPyDD) | 16 June, ours, unfocused / focused / +Earth | ≈ 5 March, unfocused / focused |
|---|---|---|---|---|---|---|
| 300 | 733.1 | 796.7 | 787.2 / 820.7 | 1160.8 | 1143.0 / 1186.8 / 1190.2 | 787.6 / 820.8 |
| 350 | 6.288 | 11.39 | 11.01 / 12.28 | 27.92 | 26.80 / 29.58 / 29.77 | 8.14 / 9.20 |
| 366 | 0.402 | 1.009 | 0.963 / 1.115 | 2.876 | 2.709 / 3.137 / 3.171 | 0.544 / 0.628 |
| 380 | 0.0250 | 0.1027 | 0.0985 / 0.1141 | 0.3174 | 0.3000 / 0.3464 / 0.3499 | 0.0432 / 0.0535 |

Ratios: **focused/unfocused = 1.038 / 1.104 / 1.158 / 1.156 (16 June), 1.043 / 1.115 / 1.158 / 1.170 (annual), 1.042 / 1.130 / 1.155 / 1.242 (March)** at δ = 300/350/366/380 keV. Ours vs WimPyDD (same day): 0.985/0.960/0.942/0.944 (June), 0.988/0.967/0.955/0.952 (annual) — the known 1–6 % bin-averaging offset (P018, P055). June/annual: 1.452/2.433/2.813/3.067 unfocused → 1.446/2.409/2.814/3.029 focused (P055's date LR 1.455/2.451/2.857/3.101), i.e. **focusing changes the timing evidence by ≤ 1.3 %**; the modulation phase is untouched at these v_min (the density term that shifts the phase for low v_min, Lee et al. 2014, is a 1 % effect here and is swamped by the 14.5 km/s speed modulation).

**Higgsino δ(N = 1), 1 TeV** (log-interpolation of the δ grid 352.5–392.5 keV): Sun frame 360.6; annual WimPyDD 366.05 (P007: 366; P018: 365.7); **annual, ours: 365.77 → 366.66 keV with focusing (+0.89 keV)**; 16 June WimPyDD 372.42; **16 June, ours: 372.05 → 373.00 (Sun) → 373.08 keV (Sun + Earth)**; ≈ 5 March: 362.39 → 363.24. So at the actual event date a pure Higgsino needs δ ≈ 373 keV for one expected event (vs 366 for the annual mean): the "date-specific" Higgsino window of P007 sits 7 keV higher than the annual one, and 14 keV below δ_max(event, focused) = 388.0 keV.

## 5. Part D — Earth's focusing and shadowing

Same mapping with GM_E = 3.986004 × 10⁵ km³ s⁻², r = 6371.1 km (v_esc,E = 11.19 km/s), radial direction = the Galactic direction of the SURF zenith at 21:22 UTC (−0.851, 0.064, 0.521), applied to the geocentric velocity **w** + **v**_rot before the solar mapping. Results (event time): density ×1.00054; η ratio 1.0015 / 1.0022 / 1.0036 / 1.0082 at 600/700/750/790 km/s; rate ×1.0029 / 1.0065 / 1.0108 / 1.0106 at δ = 300/350/366/380; hard edge 810.84 → 810.91 km/s (analytic √(810² + 11.19²) − 810 = 0.077 km/s), **δ_max + 0.063 keV**; δ(N=1) + 0.08 keV. Ten to twenty times smaller than the Sun's effect, and comparable to the rotation term — "negligible" at the ≤ 1 % level, as expected for a TeV particle at 800 km/s.

Shadowing by the Earth: for δ = 300 keV and m = 1 TeV the threshold speed c√(2δ/μ_A) is 1916 (O), 1457 (Si), 1043 (Fe), 1026 (Ni) km/s — all above v_max = 811 km/s — so inelastic up-scattering inside the Earth is kinematically forbidden for every abundant terrestrial nucleus (Pb, 576 km/s, is negligible in abundance; Xe itself needs 703 km/s). Elastic loop-level scattering (σ_n ~ 10⁻⁴⁸ cm², P007) through the full Earth column (4.2 × 10³³ nucleons cm⁻², mean density 5.51 g cm⁻³) has probability 7 × 10⁻¹⁰ even with A² coherence on iron. The Earth is transparent; no shadowing correction exists at any level relevant here (the apex was on the horizon at the event time, so the mean flux path through rock was in any case a grazing chord).

## 6. Part E — how special is 16 June? (`fraction_of_year_deltamax.csv`)

Hourly scan, one year (and the 371-day run window with uniform livetime), δ_max(248 keV, m; v_esc + |v_E(t)|):

| m_χ, v_esc | ≥ 370 keV | ≥ 380 | ≥ 385 | ≥ 387 | ≥ δ_max(event) | ≥ 390 |
|---|---|---|---|---|---|---|
| 1 TeV, 544 | 67.1 % (245 d) | **39.6 % (144.7 d)** | **22.2 % (80.9 d)** | **9.8 % (35.8 d)** | **8.5 % (31.1 d)** [387.13 keV] | 0 (max 387.50) |
| — run window | 67.6 % | 40.6 % | 21.8 % | 9.7 % | 8.4 % | 0 |
| 1 TeV, 528 | 29.5 % | 0 (max 374.3) | 0 | 0 | 8.5 % [373.98] | 0 |
| 1 TeV, 560 | 100 % | 77.4 % | 61.8 % | 56.4 % | 8.5 % [400.27] | 48.4 % |
| 0.4 TeV, 544 | 0 (max 342.0) | 0 | 0 | 0 | 8.5 % [341.60] | 0 |
| 4 TeV, 544 | 100 % | 100 % | 100 % | 90.6 % | 8.5 % [409.89] | 76.0 % |

P006's "144 days per year at δ = 380 keV, never at 390" is reproduced exactly (144.7 d; never). The event sits in the **top 8.5 % of the year in |v_E|** (31 days centred on 2 June), so an interpretation requiring δ within 0.4 keV of the annual maximum δ_max is open 8.5 % of the time, within 2 keV (≥ 385) 22 % of the time, within 7 keV (≥ 380) 40 % of the time. With solar focusing every threshold shifts by +0.90 keV (i.e. "≥ 387 keV with focusing" has the duty cycle of "≥ 386.1 keV without", ≈ 15 %). This is the kinematic face of P006's odds ratio 1.4–2.5: the date is favourable but not remarkable unless δ is within a few keV of the edge.

## 7. Final table of kinematic inputs (recommended for the corpus; `P085_kinematic_inputs.json`)

| input | value |
|---|---|
| event time | 2023-06-16 21:22:39 UTC, JD 2460112.39073, day 167.891 |
| **v**_⊙ (U, V, W) | (11.1, 250.2, 7.3) km/s, |v_⊙| = 250.55 |
| **v**_orb (heliocentric, Galactic) | (1.012, 14.724, −25.344) km/s, |v| = 29.328, r = 1.0159 AU |
| **v**_rot (SURF) | (0.167, −0.062, 0.281) km/s, 0.333 km/s, projection −0.074 |
| **v**_E (U, V, W) | **(12.279, 264.862, −17.763) km/s; |v_E| = 265.74 ± 2.5 (solar motion)** |
| apex / wind | (l, b) = (87.35°, −3.83°) / (267.35°, +3.83°); apex (RA, Dec) = (319.35°, +43.76°), alt −0.4° at SURF |
| v_max (v_esc = 544) | 809.74 km/s; **810.84 with solar focusing; 810.91 with Earth** |
| δ_max(248 keV) 0.4/1/4 TeV | 341.60 / **387.13** / 409.89 keV (+0.90 keV with focusing; ±2.0 keV solar motion; ±0.82 keV per km/s of v_esc) |
| E_max elastic (1 TeV) | 1417.5 keV; at δ = 366: 182.4–582.9 keV |
| annual: max / min / mean / Sun frame | 266.19 (2 Jun 10 h UTC) / 237.19 (4 Dec) / **252.11** / 250.55 km/s |
| δ_max(1 TeV) annual mean / max / min | 375.92 / 387.50 / 363.7 keV |
| solar focusing | n/n_∞ = 1.0084 (event; 1.0195 March, 1.0058 Sept, 1.011 mean); η ×1.022/1.031/1.049/1.12 at 600/700/750/790 km/s; rate ×1.04/1.10/1.16/1.16 at δ = 300/350/366/380 |
| Higgsino δ(N=1), 1 TeV | annual 365.8 → 366.7 (focused); event date 372.1 → 373.0 keV |
| duty cycle δ_max ≥ 380/385/387 keV (1 TeV, 544) | 39.6 / 22.2 / 9.8 % of the year; |v_E| ≥ event value 8.5 % |

## 8. Figures

- `figures/P085_fig1_vE_run.png` — top: |v_E|(t) through the LZ run from astropy (event marked, Sun-frame 250.6 dotted); bottom: δ_max(248 keV, 1 TeV, v_esc = 544) with the 380/385/387 keV thresholds.
- `figures/P085_fig2_focusing.png` — left: solar-focusing density enhancement through the year (max ≈ 5 March, min ≈ 5 September, event +0.84 %); right: η_focused/η_unfocused − 1 at v_min = 600–790 km/s (nearly flat at +2–6 %; the 790 km/s curve diverges where the unfocused edge drops below ~795 km/s).
- `figures/P085_fig3_higgsino_rates.png` — Higgsino events vs δ for the annual mean, 16 June with and without solar focusing, and the WimPyDD Sun frame.

## 9. Failed or abandoned approaches

- First implementation of the lab-frame distribution added **v**_⊙ twice (the observer velocity already contained it before the Galactic-frame shift): η(0) came out 0.58 × analytic and the hard edge at 1058 km/s. Fixed (heliocentric velocity = **w** + **v**_orb only); the validation ratios are now 1.0000–1.008.
- The one-year mask for the annual mean initially spanned June 2023–April 2024 (the scan ends 1 April 2024), giving a spurious 249.8 km/s; fixed to 365.25 d from 27 March 2023 (252.11 km/s, equal to the harmonic-fit constant).
- The 2 km/s speed grid's "hard edge" (808 → 810 km/s) cannot resolve the 1.1 km/s focusing shift; replaced by direction-scanned bisection.
- Attempting `Time.ut1` before overriding `iers.conf.auto_max_age` raised a configuration error (astropy's default `None` is rejected by the bundled config validator in this environment); set `auto_max_age = 1e9`, `auto_download = False`, `iers_degraded_accuracy = 'ignore'` — bundled IERS-B covers 2023.

## 10. Caveats

Solar-motion uncertainties are recalled (±2.5 km/s total). The Maxwellian is truncated sharply; focusing ratios for a non-Maxwellian tail (P018's k-forms) would differ in size but not in sign or date-independence. Focusing is treated as a pure Liouville map of an unbound, collisionless, steady halo (no bound Sun-captured population; the Earth's own potential is applied at the mean radius). Rates use O₁ shapes at 1 TeV, 3 keV energy steps and the P007 efficiency model; our absolute counts are 1–6 % below WimPyDD's day halos for the reason documented in P018/P055. Uniform livetime in the run-window fractions.

## 11. References

LZ Collaboration, arXiv:2609.02823 (2026). D. Baxter et al., EPJC 81, 907 (2021). R. Schönrich, J. Binney, W. Dehnen, MNRAS 403, 1829 (2010). S. K. Lee, M. Lisanti, A. H. G. Peter, B. R. Safdi, PRL 112, 011301 (2014). M. S. Alenazi, P. Gondolo, PRD 74, 083518 (2006). C. McCabe, JCAP 02 (2014) 027. Astropy Collaboration, ApJ 935, 167 (2022). Corpus: P006, P007, P018, P034, P035, P055, P068.

## 12. Tools and provenance (mirrors provenance/P085.json)

- Agent tools: Read (PAPER_GUIDE.md; P055.md; work/P055/details.md; P006.md; P018.md; P034.md; P068.md; P007.md; lzcommon.py lines 108–217 and 310–380; three output figures), Bash (lzcommon/ledger/environment greps; LZ fulltext grep for the event time; astropy offline tests ×2; wimprates/WimPyDD source inspection; P007 coupling/efficiency code inspection; WimPyDD timing test; three full runs of the script; table prints), Write (script, details.md, provenance JSON, paper), Edit (script fixes: heliocentric-velocity bug, annual mask, exact edge, inf handling, figure axis; paper trimmed to the word budgets; JSON tool counts), Skill (dataviz; JS validator skipped per PAPER_GUIDE). Bash word-count checks of the paper: whole file 846 words, body 541.
- Software: python 3.12.13; astropy 8.0.1 (`Time`, `get_body_barycentric_posvel`, `solar_system_ephemeris` = builtin, `EarthLocation.get_gcrs_posvel`, `SkyCoord` ICRS→Galactic/AltAz, `GeocentricTrueEcliptic`, `get_body`, `Time.sidereal_time`, `utils.iers.conf`); astropy-iers-data 0.2026.9.14 (bundled IERS-B); numpy 2.5.3 (`leggauss`, `lstsq`, `trapezoid`); scipy 1.18.1 (`special.erf/erfc`, `integrate.solve_ivp` DOP853); pandas 3.0.5; matplotlib 3.11.2; WimPyDD 2.0.4 (`diff_rate` with `sum_over_streams=False`, `streamed_halo_function` via `lz.wd_halo`, `eft_hamiltonian` via `lz.wd_hamiltonian`, `v_earth_sun`); wimprates 0.5.0 (`v_earth`, `earth_velocity`, `j2000_from_ymd`); numericalunits 1.28; common/lzcommon.py (`V0_KMS`, `VESC_KMS`, `V_SUN_PEC`, `LZ`, `eta0`, `delta_max_kev`, `E_R_range_keV`, `mu_red`, `m_nucleus_gev`, `v_earth_kms`, `wd`, `wd_halo`, `wd_hamiltonian`).
- Recalled items (12): GM_⊙, GM_E, AU, R_E (certain); J2000 equatorial→Galactic matrix (certain); SURF coordinates 44.352° N, 103.751° W (likely) and depth/elevation (uncertain, irrelevant); Schönrich (11.1, 12.24, 7.25) km/s and uncertainties (likely/uncertain); Baxter σ(v₀) = 1.5 km/s (uncertain); Liouville/hyperbolic-orbit mapping formula (likely; verified numerically); ERFA built-in ephemeris accuracy (likely); G_F, sin²θ_W (certain; from P007); Earth mean density 5.51 g cm⁻³ (certain); Lee et al. 2014 statement that solar focusing peaks around 1 March (likely; reproduced: ≈ 5 March).
- WimPyDD-generated files: none (`diff_rate` and `streamed_halo_function` only; no response-function files written).
