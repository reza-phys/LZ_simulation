# P092 — Sidereal and diurnal effects on the high-velocity tail: is there any time-of-day information in a δ ≈ 300–385 keV event?

Simulated date 2026-09-16. Author profile: dark-matter phenomenology group. Category IDM (HALO aspects). Primary hep-ph, cross-list astro-ph.GA.
Builds on P006 (event date), P018 (tail; dδ_max/dv = 0.82 keV per km/s), P034 (annual modulation N_3σ), P042 (SURF geometry, lab frame), P055 (from-scratch Earth velocity, 265.96 km/s on 16 June), P067 (apex alt −0.5°, az 347° at 21:22:39 UTC). P085 (exact Earth velocity) did not exist when this paper was written (`output/papers/P085.md` absent).

Script: `output/code/P092_diurnal.py` (runs in 23–37 s from the simulation root; the per-isotope WimPyDD response is cached in `P092_unit_stream_kernel.npz`). Outputs in `output/work/P092/`: `run_log.txt`, `P092_results.json`, `P092_hourly_inputs.csv`, `P092_modulation_summary.csv`, `P092_vlab_apex_vs_UTC.csv`, `P092_rates_rel_vs_UTC.csv`, `figures/fig1_vlab_apex_vs_UTC.png`, `figures/fig2_rate_modulation_vs_UTC.png`, `figures/fig3_N3sigma_diurnal_vs_annual.png`.

## 1. Motivation and framework

The inelastic (endothermic) reading of the LZ 248 keV event (16 June 2023 21:22:39 UTC; LZ paper, Data Analysis paragraph) lives on the high-velocity tail: for δ ≥ 300 keV only Earth-frame speeds above ≈ 700 km/s scatter (P002, P018). The kinematic edge δ_max(E_R) = v_max √(2 m_N E_R) − m_N E_R/μ moves by 0.82 keV per km/s of the lab speed (P018), so anything that changes |v_lab| at the 0.1 km/s level is, in principle, visible near the edge. Earth's orbital motion (±14.5 km/s) is the annual modulation studied by P006/P034. Earth's *rotation* is the next term: at SURF (latitude 44.352° N) the lab moves eastward at ω_⊕ R cos φ = 0.332 km/s, and the projection of that velocity on the DM-wind direction changes sign every half sidereal day. This paper asks how large the resulting sidereal modulation of δ_max and of the in-ROI inelastic rate is, whether the *hour* of the LZ event therefore carries likelihood information, how the diurnal test compares with P034's annual one, what a directional detector at SURF would have seen through the day, and whether the Earth's and Sun's gravitational focusing or Earth-shadowing (DM scattering inside the Earth) have any daily structure worth worrying about.

## 2. Frames and the lab velocity vector

### 2.1 Construction

v_lab(t) = v_⊙ + v_orb(t) + v_rot(t), all in Galactic (U, V, W) km/s.

- v_⊙ = (0, 238, 0) + (11.1, 12.2, 7.3) = (11.1, 250.2, 7.3), |v_⊙| = 250.552 km/s (Baxter et al. 2021 via `lzcommon`).
- v_orb(t): Earth's heliocentric velocity from astropy 8.0.1's built-in (ERFA) ephemeris, `get_body_barycentric_posvel('earth') − ('sun')`, in ICRS, rotated to Galactic with astropy's fixed ICRS→Galactic matrix (built from the three Galactic basis vectors; no IERS data needed; it agrees with P055's recalled Hipparcos matrix to 1e-7). No AltAz frame is used anywhere.
- v_rot(t): eastward, (−sin θ, cos θ, 0) in equatorial axes with θ the local sidereal time at SURF, magnitude V_ROT = ω_⊕ (R − d) cos φ with ω_⊕ = 7.2921150e-5 rad/s, geocentric radius R = 6367.5 km at 44.35° (WGS84), depth d = 1.478 km, φ = 44.352°: **V_ROT = 0.3319 km/s**. LST from the analytic GMST: GMST(h) = 18.697374558 + 24.06570982441908 D, D = JD − 2451545.0, longitude −103.751°. Neglected: geodetic–geocentric latitude difference (0.19°) and 23 years of precession (0.32°) in the orientation of the equatorial axes: < 0.003 km/s.

### 2.2 Ephemeris validation and a correction to the corpus

| check | astropy | expectation |
|---|---|---|
| perihelion 4 Jan 2023 | 30.275 km/s at 0.9833 AU, ∠(r,v) = 90.00° | 30.29 km/s (recalled, certain) |
| aphelion 6 Jul 2023 | 29.282 km/s at 1.0167 AU, ∠(r,v) = 90.00° | 29.29 km/s |
| event 16 Jun 2023 | 29.328 km/s at 1.0159 AU, ∠(r,v) = 89.69° | — |

v_orb(event) = (1.012, 14.724, −25.344) km/s. The geocentric lab speed (no rotation) is **|v_⊙ + v_orb| = 265.814 km/s**. wimprates 0.5.0 (`earth_velocity`, v₀ = 238) gives 265.977 km/s and P055's from-scratch circular-orbit frame 265.963 km/s (P055 `earth_velocity_validation.json`); WimPyDD's `v_earth_sun` gives 265.92 km/s; `lz.v_earth_kms(167.89)` (cosine model) 265.06. The astropy − wimprates orbital vector difference is (0.881, −0.171, 0.466) km/s, |Δ| = 1.01 km/s, and its projection on the apex is −0.162 km/s — exactly the 265.81 vs 265.98 difference. The cause is orbital eccentricity (e = 0.0167): wimprates, WimPyDD and P055 use a circular orbit with constant 29.79–29.80 km/s and mean motion, whereas on 16 June (20 days before aphelion) the true speed is 29.33 km/s and the true longitude lags the equinox-anchored mean longitude by ≈ 1.2° (equation of centre 1.84° at the equinox, 0.63° on 16 June). **On 16 June the corpus values of |v_lab| are 0.16 km/s too high**, which moves δ_max(248 keV) by −0.13 keV and the in-ROI rate by −0.4 % (δ = 300) to −3.5 % (380 keV) at the dlnR/dv values of §4. P067, which used the astropy barycentric velocity, already had 265.8 km/s.

### 2.3 Rotation term over 16 June 2023 (1-min grid; Fig. 1)

Apex (direction of the geocentric lab velocity, where the wind comes from): RA 319.44°, Dec +43.75°, (l, b) = (87.38°, −3.89°) — P067: 319.4/43.7. Because the rotation vector lies in the equatorial plane, its projection on the apex is V_ROT cos(Dec_apex) sin(−HA): **amplitude 0.2398 km/s, peak-to-peak 0.4796 km/s**, maximal when the apex is due east (HA = −6 h) and minimal when due west.

Rotation-only series (orbital velocity frozen at the event): mean over one sidereal day 265.8140 km/s (the rotation adds only +1.5e-4 km/s to the geocentric speed, as expected from ⟨u_⊥²⟩/2v ≈ V_ROT²/4v = 1e-4), min 265.574, max 266.054 km/s; maximum at 04:36 UTC, minimum at 16:34 UTC. Full series (rotation + orbital drift within the day): 265.946 at 00:00 → 265.897 at 24:00 UTC, orbital drift −0.052 km/s per day, 24-h mean 265.835 km/s.

**At 21:22:39 UTC:** LST = 8.1199 h (P067: 8.120 h); apex HA = +10.82 h, altitude −0.51°, azimuth 347.4° (P067: −0.5°, 347.4°), 1.17 h before lower culmination (22:33 UTC). The rotation projection is −0.0727 km/s, so **|v_lab| = 265.741 km/s = daily mean − 0.073 km/s** (0.30 of the amplitude). Apex over the day: altitude −1.90° (22:33 UTC) to +89.40° (10:35 UTC; Dec_apex ≈ latitude, so the apex passes within 0.6° of the zenith), below the horizon 11.5 % of the day; due east 04:36 UTC, due west 16:34 UTC.

## 3. Kinematics

dδ_max/dv_max = √(2 m_N E_R)/c = **0.8218 keV per km/s** at 248 keV, A = 131.29 (P018: 0.8218), mass-independent. With v_max = |v_lab| + 544 km/s:

| m_χ (GeV) | δ_max mean (keV) | at the event | daily range | peak-to-peak |
|---|---|---|---|---|
| 400 | 341.658 | 341.598 (−0.060) | 341.461–341.855 | 0.394 |
| 1000 | 387.186 | 387.126 (−0.060) | 386.989–387.383 | 0.394 |
| 4000 | 409.950 | 409.890 (−0.060) | 409.753–410.147 | 0.394 |

P006's 387.3 keV (circular orbit) becomes 387.13 keV with the eccentric orbit and rotation; the rotation alone contributes −0.06 keV at the event hour. The Sun's potential (§6) would add a constant +0.9–1.0 keV not included here or in any corpus halo.

## 4. The inelastic rate over the sidereal day

### 4.1 Method ("analytic-η scaling")

For the O₁ operator the differential rate is exactly dR/dE = Σ_A C_A(E) η(v_min,A(E); |v_lab|) — no explicit velocity dependence beyond the halo function. C_A(E) (events/(t·yr·keV) per unit η, WimPyDD unit coupling c⁰ = 1 GeV⁻², c¹ = 0, m_χ = 1 TeV) is obtained per xenon isotope from WimPyDD 2.0.4 `diff_rate(..., isotopes_list={0:[i]}, sum_over_streams=True)` with a *single* unit stream at 2000 km/s (Δη = 1), on a 2-keV grid from 60 to 400 keV (9 isotopes × 171 energies = 1539 calls, 4 s). Check: the response is identical for δ = 0 and δ = 366 keV at E = 150/248/300 keV (ratios 1.000000). η is `lz.eta0` (analytic truncated Maxwellian, v₀ = 238, v_esc = 544 km/s) evaluated at the exact |v_lab(t)| with per-isotope v_min,A(E) using WimPyDD's isotope masses (115.44–126.62 GeV). This avoids the 0.5 km/s quantisation of a stream grid, which would otherwise swamp a 0.24 km/s effect. The ROI integral uses P034's parametrisation of the Fig. S2 efficiency: 0.96 × logistic(E − 5.4 keV) × Φ((269.9 − E)/15 keV).

Validation against WimPyDD's own halo on the event day (`lz.wd_halo(day_of_year=167.89)`, explicit 0.5 km/s grid; WimPyDD's v_lab = 265.92 km/s), in-ROI rate ratio analytic/WimPyDD: **1.0001 / 1.0031 / 1.0074 / 1.0111 / 1.0163** at δ = 300/350/366/380/385 keV (WimPyDD's bin-averaged η is slightly low near v_max, as P018 noted); point-wise spectra agree to 0.2 % (300 keV) and within 2 % except at kinematic onsets. Unit-coupling counts per 2.84 t·yr on 16 June: 1.77e13 / 5.70e11 / 6.14e10 / 2.50e9 / 6.23e8; P034's annual means 1.24e13 / 2.39e11 / 1.95e10 / 6.7e8 at 300/350/366/380 keV give June/annual ratios 1.42 / 2.39 / 3.15 / 3.7, consistent with P006/P055's date likelihood ratios.

### 4.2 Results (rotation-only series, 5-min grid; Fig. 2; `P092_modulation_summary.csv`)

| δ (keV) | ⟨R⟩ (unit c, /t/yr) | peak-to-peak | a₁ (cosine fit) | dlnR/dv (per km/s) | KL (nats/event) | N_3σ (LLR) | N_3σ (18/a₁²) | LR(event) | LR (full series) |
|---|---|---|---|---|---|---|---|---|---|
| 300 | 6.221e12 | 1.26 % | 0.630 % | 0.0263 | 9.91e-6 | 4.54e5 | 4.54e5 | 0.9981 | 0.9976 |
| 350 | 2.008e11 | 3.63 % | 1.815 % | 0.0757 | 8.24e-5 | 5.46e4 | 5.47e4 | 0.9945 | 0.9931 |
| 366 | 2.162e10 | 6.27 % | 3.135 % | 0.1308 | 2.46e-4 | 1.83e4 | 1.83e4 | 0.9904 | 0.9879 |
| 380 | 8.815e8 | 10.47 % | 5.233 % | 0.2183 | 6.85e-4 | 6569 | 6573 | 0.9836 | 0.9796 |
| 385 | 2.193e8 | 10.50 % | 5.251 % | 0.2191 | 6.90e-4 | 6524 | 6528 | 0.9835 | 0.9794 |

Phase: maximum at LST 229.4° = 15.30 h for every δ (apex due east; 04:35 UTC on 16 June). The modulation is a pure sinusoid to the precision shown (a₁ = p2p/2). a₁ = (dlnR/dv) × 0.240 km/s to 1 %.

*Why the lever arm saturates.* dlnR/dv grows from 0.026 (δ = 300) to 0.22 per km/s at δ = 380 and then stops: at δ = 385 keV the in-ROI rate comes from E ≈ 240–300 keV, where v_min(E) is ≈ 10 km/s below v_max, and the same is true at 380 keV once the 270 keV roll-off removes E > 300 keV. Near v_max, η ∝ (v_max − v_min)², so dlnR/dv ≈ 2/(v_max − v_min,eff) ≈ 0.2 per km/s; the recoil window is bounded by the efficiency edge, not by δ_max. P034 found the same roll-off control of the δ ≥ 366 keV normalisation.

Statistics. KL = ∫ p ln(p/p₀) over the sidereal day is the expected log-likelihood ratio per event; N_3σ (LLR) = 9 Var_flat(s)/(E_sig − E_flat)² (P006's Gaussian formula, exact in the small-amplitude regime) agrees with the known-phase cosine estimator 18/a₁² to 0.1 %. LR(event) = R(t_ev)/⟨R⟩_sidereal for the rotation-only series and R(t_ev)/⟨R⟩_24h for the full series.

### 4.3 Comparison with the annual modulation (P034)

| δ (keV) | a₁ annual (P034) / diurnal | KL annual / diurnal | N_3σ diurnal / annual |
|---|---|---|---|
| 300 | 0.43 / 0.0063 = 68 | 0.047 / 9.9e-6 = 4740 | 4.54e5 / 96.5 = 4700 |
| 350 | 1.18 / 0.0181 = 65 | 0.39 / 8.2e-5 = 4730 | 5.46e4 / 11.5 = 4750 |
| 366 | 1.52 / 0.0313 = 48 | 0.72 / 2.5e-4 = 2930 | 1.83e4 / 6.5 = 2820 |
| 380 | 1.66 / 0.0523 = 32 | 0.90 / 6.9e-4 = 1310 | 6569 / 5.3 = 1240 |

The ratio of velocity amplitudes is 14.5/0.24 = 60; the rate-amplitude ratio is smaller at high δ only because the annual curve is non-sinusoidal and saturated (a₁ > 1). Even the most favourable case (δ ≥ 380 keV) needs ~6500 events, versus 5 for the annual test — at LZ's one event per 2.84 t·yr, 18 000 t·yr.

### 4.4 Season and halo dependence of the diurnal amplitude

a₁ ≈ (R₊ − R₋)/(R₊ + R₋) with R± at v ± 0.240 km/s:

| date (v_geo) | 300 | 350 | 366 | 380 | 385 |
|---|---|---|---|---|---|
| 2 June (266.4) | 0.63 % | 1.79 % | 3.07 % | 5.17 % | 5.27 % |
| 16 June (265.8) | 0.63 % | 1.81 % | 3.13 % | 5.23 % | 5.25 % |
| 1 Sep (250.8) | 0.74 % | 2.70 % | 5.21 % | 5.48 % | 8.39 % |
| 1 Dec (237.4) | 0.88 % | 4.29 % | 6.08 % | 56.8 % | rate zero |
| 16 June, v_esc = 528 | 0.74 % | 2.78 % | 5.36 % | 5.64 % | 9.57 % |
| 16 June, v_esc = 560 | 0.55 % | 1.31 % | 2.01 % | 3.24 % | 3.92 % |

The amplitude grows when the rate is smaller (winter; low v_esc), because a given δ then sits closer to the edge — the same degeneracy P018/P034 describe (Δv_esc ≈ ∓13 keV in δ). In December the δ = 380 keV amplitude is 57 % but the rate is ~10⁻⁴ of June's (P034: zero 15 % of the year), so the diurnal information is still negligible in absolute terms. The apex direction changes by a few degrees over the year (v_orb tilts the lab velocity by ≤ 3.2°), so V_ROT cos(Dec_apex) is constant to ~1 %.

## 5. Directional aspect (Fig. 1, lower panel; P067 for detectors)

Near the edge, recoils are collinear with the wind (within 14.5° at δ = 366 keV; P067). At SURF the apex describes a circle of radius 46° around the celestial pole: altitude from −1.9° (due north, 22:33 UTC on 16 June) to +89.4° (10:35 UTC), east at 04:36 UTC, west at 16:34 UTC. A directional detector at SURF would see the recoil beam rotate once per sidereal day from *horizontal, heading south* (at lower culmination) to *vertically downward* (at upper culmination) — the textbook Cygnus signature, with the unusual SURF property that the sweep covers the full 90° in altitude. At the event time the wind came from alt −0.5°, az 347°, so the recoil would have been horizontal, heading SSE (az 167° ± 11°; P067). The Earth-rotation speed projection is zero at both culminations and ±0.24 km/s at the east/west passages; the event sits at 0.30 of the amplitude.

## 6. Gravitational focusing and Earth shadowing (order of magnitude)

Earth's potential. At LZ's depth Φ ≈ −v_esc,⊕²/2 with v_esc,⊕ = 11.19 km/s; energy conservation gives a direction-independent blueshift Δv = v_esc²/(2v) = 0.089 / 0.084 / 0.077 km/s at v = 700 / 750 / 810 km/s (fractional 1.3e-4 to 9.5e-5), i.e. δ_max + 0.064–0.074 keV, constant in time (not diurnal). The focusing (Liouville compression of trajectories) distorts the *angular* distribution at relative order (v_esc/v)² = 2.6e-4 / 2.2e-4 / 1.9e-4; this is the only diurnal piece, it enters the rate linearly (speeds are isotropic at O(Φ)), so the rate modulation from Earth focusing is ≲ 3e-4, two orders below the rotation effect at δ ≥ 350 keV.

Sun's potential. v_esc,⊙(1 AU) = 42.1 km/s: constant blueshift 1.27 / 1.18 / 1.09 km/s (δ_max +1.0 keV — a systematic larger than the whole diurnal peak-to-peak, ignored by every corpus halo and by the SHM convention itself); solar focusing density modulation (v_esc/v)² = 3.6e-3 / 3.2e-3 / 2.7e-3, annual with a March maximum (Lee, Lisanti, Peter & Safdi 2014, recalled; relevant mainly at low v_min); its daily change is (2 R_⊕/1 AU) × that = 2–3e-7.

Earth shadowing. (a) Kinematics: the inelastic channel in Earth nuclei needs v ≥ √(2δ/μ_A). Ceilings μ_A v_max²/2 at 1 TeV, v_max = 809.8 km/s: O 54, Mg 80, Al 90, Si 93, S 106, Ca 131, Fe 181, Ni 187 keV; v_min(δ = 300 keV) = 1916 (O), 1457 (Si), 1043 (Fe), 1026 km/s (Ni) — all far above v_max. **For δ > 187 keV no inelastic scattering occurs anywhere in the Earth, at any cross-section**; the χ₂ produced in LZ is irrelevant to the incoming flux. (b) An elastic component σ_el would scatter: for a heavy WIMP σ_A ≈ σ_n A⁴ (coherent, F² = 1, conservative), so λ⁻¹ = σ_n (ρ/m_u) Σ f_A A³ with Earth ρ = 5.51 g/cm³ and the recalled bulk composition (Σ f_A A³ = 6.8e4): **λ_Earth = 4409 km × (1e-38 cm²/σ_el)**; crust rock (2.7 g/cm³): λ_rock = 1.48 km (the overburden) at σ_n = 1.7e-34 cm², so anything "strongly interacting" is stopped before reaching LZ. Path length from the detector (R − 1.478 km) to the surface along the apex: 1.48 km (zenith), 137 km (horizon), 205 km at the event (alt −0.5°), 463 km maximum (alt −1.9°). Diurnal shadow amplitude ≈ L_max/λ = 0.10 / 1.0e-3 / 1.0e-5 / 1.0e-8 for σ_el = 1e-38 / 1e-40 / 1e-42 / 1e-45 cm²; **< 1e-4 requires σ_el < 9.5e-42 cm²**, and LZ's own elastic SI limit at 1 TeV (~1e-45 cm², recalled) bounds it below 1e-8. Note that a scatter off Fe removes up to 20 % of a tail particle's kinetic energy (10 % of its speed), so treating a scatter as removal from the tail is appropriate. (c) The one-event *inelastic* cross-sections implied by the 16 June rate (σ_n = (c⁰/2)² μ_n²/π = 2.40e-29 cm² per unit coupling, divided by the unit-coupling counts): 1.54e-42 / 4.77e-41 / 4.43e-40 / 1.09e-38 / 4.37e-38 cm² at δ = 300/350/366/380/385 keV. If the *same* coupling mediated elastic scattering, λ would be 2.9e7 / 9.2e5 / 9.9e4 / 4054 / 1008 km — so at δ ≥ 380 keV an elastic partner of equal strength would shadow at the 10–40 % level; pure inelastic models have no such channel (elastic scattering arises only at loop level or is suppressed by (δ/m_χ)²), and LZ's elastic limit excludes it by seven orders of magnitude anyway.

## 7. Time-of-day kinematic inputs (hourly; full table `P092_hourly_inputs.csv`)

| UTC | LST (h) | apex HA (h) | apex alt | apex az | v_rot·â (km/s) | v_lab (km/s) | δ_max(1 TeV) | Earth path (km) | R/⟨R⟩ 300 | 366 | 380 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 00:00 | 10.68 | −10.61 | +0.0° | 15° | +0.085 | 265.899 | 387.256 | 134 | 1.0022 | 1.0110 | 1.0181 |
| 03:00 | 13.69 | −7.60 | +15.8° | 43° | +0.219 | 266.033 | 387.366 | 5.4 | 1.0058 | 1.0288 | 1.0482 |
| 05:00 | 15.70 | −5.60 | +32.5° | 58° | +0.239 | 266.052 | 387.382 | 2.7 | 1.0063 | 1.0314 | 1.0527 |
| 08:00 | 18.71 | −2.59 | +62.3° | 77° | +0.150 | 265.965 | 387.309 | 1.7 | 1.0040 | 1.0196 | 1.0327 |
| 10:00 | 20.71 | −0.59 | +83.7° | 92° | +0.037 | 265.851 | 387.216 | 1.5 | 1.0010 | 1.0046 | 1.0074 |
| 12:00 | 22.72 | +1.42 | +74.7° | 275° | −0.087 | 265.727 | 387.114 | 1.5 | 0.9977 | 0.9885 | 0.9805 |
| 15:00 | 1.72 | +4.43 | +43.6° | 294° | −0.220 | 265.594 | 387.005 | 2.1 | 0.9942 | 0.9714 | 0.9525 |
| 17:00 | 3.73 | +6.43 | +25.1° | 308° | −0.238 | 265.576 | 386.990 | 3.5 | 0.9938 | 0.9690 | 0.9486 |
| 20:00 | 6.74 | +9.44 | +4.5° | 333° | −0.149 | 265.665 | 387.064 | 18 | 0.9961 | 0.9805 | 0.9674 |
| **21:22:39** | **8.12** | **+10.82** | **−0.5°** | **347°** | **−0.073** | **265.741** | **387.126** | **205** | **0.9982** | **0.9907** | **0.9842** |
| 22:00 | 8.74 | +11.45 | −1.6° | 354° | −0.035 | 265.780 | 387.157 | 401 | 0.9991 | 0.9953 | 0.9919 |
| 23:00 | 9.75 | −11.55 | −1.7° | 5° | +0.028 | 265.842 | 387.209 | 421 | 1.0007 | 1.0035 | 1.0056 |

(GMST at 00:00 UTC 16 June 2023 = 17.6006 h. v_lab is the rotation-only series; the full series is 0.00–0.05 km/s higher in the first half of the day.)

## 8. Robustness, failed approaches, discussion

- *Stream-grid method rejected.* Building Δη on WimPyDD's stream grid (P034's kernel approach) quantises v_min to the 0.5 km/s bin width, comparable to the 0.24 km/s effect; the analytic-η scaling (exact for O₁) was used instead and validated against the WimPyDD day halo (§4.1). For operators with explicit v⊥² dependence (O₃, O₅, O₇, O₈, O₁₂–O₁₅) the scaling would need a second (v²-weighted) halo function; P018 found rate *ratios* hamiltonian-insensitive to 0.02 dex, so a₁ would change little.
- *Frames.* P055's analytic circular-orbit frame and wimprates agree with each other to 0.08 km/s but both differ from the ephemeris by 1.0 km/s (vector) on 16 June, because of orbital eccentricity; the projection on the apex is −0.16 km/s. The geocentric speed adopted here (265.814 km/s) supersedes P055's 265.963 and P006's 266; the peak-day speed and phase are unaffected at the km/s level, so no P006/P034 conclusion changes (rates shift by ≤ 3.5 %).
- *Mass dependence.* The δ_max daily range (0.394 keV) is mass-independent; rate amplitudes were computed at 1 TeV only (P002: mass-insensitive shapes for δ ≥ 350 keV).
- *Efficiency shape.* dlnR/dv at δ ≥ 380 keV is set by the roll-off (§4.2); a 400 keV ROI would let δ_max control it and raise a₁(385) — not computed.
- *Livetime.* A real diurnal analysis would fold live time in sidereal time; LZ's 220 live days are presumably uniform in hour of day, which is all the LR calculation assumes.
- *Extended discussion.* The diurnal channel is the weakest clock available to inelastic DM: velocity amplitude 60× below the annual one, information per event 10³–5×10³× below, and — since the phase is fixed by geometry (apex due east) — with no free parameter that could mimic it. Its one useful corollary is negative: nothing about the *hour* 21:22 UTC is peculiar under either hypothesis (LR 0.98–1.00). The kinematic by-product matters more: at the 0.1 km/s level the corpus's circular-orbit Earth velocities are wrong by 0.16 km/s on the event date, the Sun's potential adds a constant +1 km/s that no halo model in the corpus includes, and both dwarf the rotation term. Gravitational focusing and Earth shadowing have no bearing on the LZ inelastic interpretation: the former is ≤ 3e-4 diurnally, the latter is kinematically absent for δ > 187 keV and, for any elastic admixture LZ itself allows, < 1e-8.

## 9. Figures

- Fig. 1 `figures/fig1_vlab_apex_vs_UTC.png`: upper — |v_lab| minus the sidereal mean vs UTC on 16 June 2023 for the rotation-only series (blue) and rotation + orbital drift (dashed), right axis the corresponding δ_max(248 keV) shift; the event (red) sits at −0.073 km/s. Lower — apex altitude (green) and azimuth (dashed) at SURF; the event is on the northern horizon 1.2 h before lower culmination.
- Fig. 2 `figures/fig2_rate_modulation_vs_UTC.png`: per-cent deviation of the in-ROI O₁ inelastic rate from its sidereal mean vs UTC for δ = 300/350/366/380/385 keV (1 TeV); the 380 and 385 keV curves coincide (roll-off-limited lever arm). Maximum 04:35 UTC.
- Fig. 3 `figures/fig3_N3sigma_diurnal_vs_annual.png`: events for a 3σ modulation detection against a time-flat population, diurnal (this work) vs annual (P034), δ = 300–380 keV.

## 10. References

- LZ Collaboration, arXiv:2609.02823 (2026).
- J. D. Lewin, P. F. Smith, Astropart. Phys. 6, 87 (1996) — Earth-velocity conventions.
- D. Baxter et al., Eur. Phys. J. C 81, 907 (2021) — SHM parameters.
- S. K. Lee, M. Lisanti, A. H. G. Peter, B. R. Safdi, Phys. Rev. Lett. 112, 011301 (2014) — gravitational focusing.
- B. J. Kavanagh, R. Catena, C. Kouvaris, JCAP 01 (2017) 012 — Earth-scattering signatures.
- J. I. Collar, F. T. Avignone III, Phys. Lett. B 275, 181 (1992) — diurnal modulation from Earth shadowing.
- D. Tucker-Smith, N. Weiner, Phys. Rev. D 64, 043502 (2001) — inelastic DM.
- Corpus: P002, P006, P018, P034, P042, P055, P067.

## 11. Tools and provenance (mirrors `output/provenance/P092.json`)

Software: python 3.12.13 (`.venv/bin/python`); numpy 2.5.3; scipy 1.18.1 (`stats.norm.cdf`); matplotlib 3.11.2 (Agg); pandas 3.0.5 (Timestamp for wimprates); astropy 8.0.1 (`Time`, `get_body_barycentric_posvel` built-in ephemeris, `SkyCoord` ICRS↔Galactic rotation only; `iers.conf.auto_download=False`; no AltAz); wimprates 0.5.0 (`earth_velocity`, `j2000`) + numericalunits; WimPyDD 2.0.4 (`diff_rate` with `isotopes_list`, `eft_hamiltonian` via `lz.wd_hamiltonian`, `v_earth_sun`, `streamed_halo_function` via `lz.wd_halo`; Xe isotope masses/abundances); `output/code/common/lzcommon.py` (LZ dict, V0/VESC/V_SUN_PEC, `eta0`, `mu_red`, `m_nucleus_gev`, `delta_max_kev`, `wd`, `wd_halo`, `wd_hamiltonian`, `wd_rate`, `GEV_TO_CM2`, `M_NUCLEON_GEV`).
Script: `output/code/P092_diurnal.py`, command `.venv/bin/python output/code/P092_diurnal.py` (23–37 s). WimPyDD-generated files: none (diff_rate does not write response files; the cached `P092_unit_stream_kernel.npz` is written by this script).
Local inputs: PAPER_GUIDE.md; `inputs` via lzcommon (event time, efficiency points, exposure); papers P006, P018, P034, P042, P055, P067; `output/work/P055/earth_velocity_validation.json`; `output/work/P067/P067_summary.json`, `P067_run_log.txt`, `P067_apex_altaz_16June2023.csv`; scripts P034_modulation_test.py (kernel/efficiency conventions), P055_gaia_substructures.py (frame), P067_directional.py (GMST/alt-az); `output/work/P034/details.md` (a₁, KL, N_3σ); `output/results_ledger.csv` (header, recent rows).
Recalled knowledge (13 items, listed in the JSON): SURF coordinates and depth; ω_⊕; WGS84 radius; GMST formula; Earth and solar escape speeds; perihelion/aphelion dates and speeds; Earth bulk and crust compositions and densities; LZ elastic limit at 1 TeV (~1e-45 cm²); Lee et al. 2014 focusing result; small-angle neglects.
Datasets: none. Data requests: none. Failed/abandoned: stream-grid Δη approach (quantisation), see §8.
