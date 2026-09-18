# P055 — Known substructures and the June event: Sagittarius, the Gaia Sausage, S1/S2 and the Helmi streams as sources of a 248 keV recoil

Simulated date 2026-09-11 · Category HALO · astro-ph.GA (cross-list hep-ph) · Author profile: Gaia-era Galactic-dynamics group.
Competes with / extends P030 (generic streams: lab speeds and rate boosts on 16 June). Here: *specific* Gaia substructures with their (recalled) anisotropic kinematics, their **time dependence** (each has its own modulation phase and amplitude), the **date likelihood ratio** of 16 June, and the **directional** signature.

Scripts: `output/code/P055_gaia_substructures.py` (main, 723 s wall time under load; 416 s in a first identical run) and `output/code/P055_postprocess.py` (53 s; first-harmonic fits of the saved rate series and the elastic-248 keV date LR on a reduced 48×32 angular grid — its numbers coincide with the full run's `elastic_248keV_ratios.csv` columns, which were produced by the same code path at 96×64). All tables under `output/work/P055/`, figures under `output/work/P055/figures/`.

## 1. Motivation and framework

The inelastic reading of the LZ event (E_R = 248 keV, 16 June 2023 21:22:39 UTC) lives in the last ~100 km/s of the Earth-frame speed distribution: v_min(248 keV, 1 TeV, δ) = 704 / 765 / 801 km/s for δ = 300 / 350 / 380 keV against v_max = v_esc + v_E = 810 km/s on 16 June (P002, P006, P018). P030 showed that a *bound* stream cannot raise δ_max(248 keV, 1 TeV) above 387 keV, that only a retrograde stream within ~15 km/s of v_esc changes the δ = 380 keV rate appreciably, and that known substructures (recalled, isotropised) have δ_max ≤ 194 keV. Three questions remain and are the subject of this paper:

1. **Time dependence.** A substructure with Galactic-frame bulk velocity **u** has an Earth-frame speed v_lab(t) = |**u** − **v**_⊙ − **v**_orb(t)|, whose annual variation has its *own* phase and amplitude (the projection of the Earth's orbital velocity onto the stream's relative-velocity direction). Does any known substructure make 16 June a preferred date beyond the SHM's 1–2 June peak?
2. **Composite rates through the year.** With the halo (1 − f) SHM + f × substructure, what happens to the in-ROI inelastic rate at δ = 250–380 keV on 16 June, to its June/December ratio and phase, and to the date likelihood ratio LR(16 June) = R(t_ev)/⟨R⟩_run of P006?
3. **Directional signature.** Where would a directional detector see the recoils come from for each component, relative to the SHM dipole (wind from the solar apex, near Cygnus)?

Framework: Baxter-2021 SHM (v₀ = 238, v_esc = 544 km/s, **v**_pec = (11.1, 12.2, 7.3) km/s, ρ₀ = 0.3 GeV/cm³ fixed; the substructure *replaces* a fraction f of the local density). Galactic Cartesian axes (U toward the Galactic centre, V along rotation, W toward the NGP) = WimPyDD's and wimprates' axes. m_χ = 1 TeV, O₁ isoscalar-dominated Higgsino Z-coupling of P007 (σ_n = 7.4 × 10⁻³⁹ cm²) so that absolute counts compare with P007/P030; every *ratio* is coupling-independent. LZ efficiency (plateau 0.96, 50 % at 5.4 and 269.9 keV, erf edges as in P007/P018/P030), exposure 2.84 t·yr.

## 2. Earth's velocity vector (Part 0)

Implemented from scratch (not taken from wimprates) and validated against it:

- Circular orbit: Sun's apparent ecliptic longitude λ_⊙(t) = ω (t − t_eq), ω = 2π/365.25 d, t_eq = 2023-03-20 21:24 UTC (recalled, likely, ±1 h). The Earth moves at heliocentric longitude λ_⊙ + 180° with velocity direction (sin λ_⊙, −cos λ_⊙, 0) in ecliptic coordinates, |v_orb| = 29.79 km/s (recalled, certain).
- Ecliptic → equatorial: rotation about x by the J2000 obliquity ε = 23.4393° (certain).
- Equatorial → Galactic: the J2000 (Hipparcos) rotation matrix, rows (−0.0549, −0.8734, −0.4838), (0.4941, −0.4448, 0.7470), (−0.8677, −0.1981, 0.4560) (certain).
- **v**_obs(t) = (0, 238, 0) + (11.1, 12.2, 7.3) + **v**_orb(t).

Validation (`earth_velocity_validation.json`): |**v**_orb,mine − **v**_orb,wimprates| ≤ 0.08 km/s on nine test days; vs WimPyDD `v_earth_sun` ≤ 0.40 km/s. Unit vectors at the equinox and a quarter-year later: e₁ = (0.9938, 0.1110, 0.0004), e₂ = (−0.0549, 0.4941, −0.8677), against the literature (0.9931, 0.1170, −0.0103), (−0.0670, 0.4927, −0.8676) (Lee–Lisanti–Safdi 2013 / McCabe 2014, recalled, likely; the differences reflect the circular-orbit and exact-equinox approximations). Earth speed: peak day 152.5 (1 June), v_E,max = 266.44, v_E,min = 237.36 km/s, first-harmonic amplitude 14.54 km/s; **v_E(16 June 21:22) = 265.96 km/s** (wimprates 265.98, WimPyDD 265.92; P006: 266). Solar apex on 16 June at (l, b) = (87.6°, −4.0°); the SHM "wind" (mean Earth-frame DM arrival velocity, −**v̂**_obs) points from (l, b) = (267.6°, +4.0°).

## 3. Substructure catalogue (Part 1; all recalled, flagged)

| key | component | **u** = (U, V, W) km/s | σ_(U,V,W) km/s | f (range) | reliability | source |
|---|---|---|---|---|---|---|
| Sgr_m | Sagittarius stream, −W | (0, 0, −300) | (30, 30, 30) | 0.03 (0.01–0.05) | speed likely; vertical sense and f uncertain | Freese+2004; Purcell+2012 |
| Sgr_p | Sagittarius, +W | (0, 0, +300) | (30, 30, 30) | 0.03 | as above | — |
| GSE | Gaia Sausage/Enceladus, SHM++ form | (0, 0, 0) | (266.1, 84.1, 84.1) [β = 0.9] | 0.20 (0.10–0.30) | likely | Evans, O'Hare, McCabe 2019; Necib+2019 |
| S1 | S1 stream (retrograde) | (+30, −297, −73) | (83, 27, 59) | 0.10 (0.01–0.10) claimed | velocities likely; sign of U and f uncertain | Myeong+2018; O'Hare+2018 |
| S2 | S2 stream (prograde, vertical) | (6, 164, −250) | (30, 20, 40) | 0.01 (0.003–0.02) | uncertain | O'Hare+2020 |
| Helmi_p / Helmi_m | Helmi streams, two clumps | (0, 150, ±250) | (30, 30, 30) | 0.005 each | v_φ, |v_z| likely; f uncertain | Helmi+1999; Koppelman+2019 |
| Shards | retrograde shards (Rg-type) | (0, −290, 0) | (60, 30, 60) | 0.01 (0.003–0.02) | uncertain | O'Hare+2020 |
| Vesc_retro | reference: retrograde stream at v_esc, anti-parallel to **v**_⊙ | −544 **v̂**_⊙ | (20, 20, 20) | 0.01 | hypothetical (P030 limiting case) | P030 |

SHM++ Sausage construction (Evans et al. 2019): σ_r = v₀ √(3/(2(3 − 2β))) = 266.1 km/s, σ_φ = σ_z = σ_r √(1 − β) = 84.1 km/s for β = 0.9 and v₀ = 238 (v₀, v_esc deliberately kept at the Baxter values to isolate the *anisotropy* effect; P018 covered the literal SHM++ v₀ = 233, v_esc = 528). Note on the assignment's prior ("S2, Helmi v_lab ≈ 400–500 km/s"): the recalled Galactic-frame vectors (|**u**| ≈ 290–300 km/s, prograde with v_φ ≈ +150–165) give **lab speeds of only 230–300 km/s** (Table 4); a lab speed of 400–500 km/s would require retrograde or much faster debris than the literature attributes to S2/Helmi. We use the Galactic-frame vectors and flag the discrepancy.

### 3.1 Earth-frame speed distribution of a truncated anisotropic Gaussian

Every component is f_gal(**u**) ∝ exp[−½ Σ_i (u_i − μ_i)²/σ_i²] Θ(v_esc − |**u**|). With **w** = **u** − **v**_obs the Earth-frame speed density is

  f(w, t) = w² ∮ dΩ f_gal(w **n̂** + **v**_obs(t)),   η(v_min, t) = ∫_{v_min}^∞ f(w, t)/w dw,

evaluated on a 1 km/s grid (0–1100 km/s) with a Gauss–Legendre × uniform angular grid (96 × 64) whose polar axis is the mean lab-velocity direction (so the v_esc cap, which is small and centred there for the fast tail, is well sampled), restricted to |w − v_lab| ≤ 7 max σ. WimPyDD arrays: Δη_i = η(v_{i−1}) − η(v_i) on upper bin boundaries (P018/P030 convention). For isotropic, untruncated components (Sgr, S2, Helmi; ≥ 6σ inside v_esc) the exact closed form f(w) = w/(√(2π) σ v_lab) [e^{−(w−v_lab)²/2σ²} − e^{−(w+v_lab)²/2σ²}] is used (its η is P030's analytic η_s).

Validation (`speed_distribution_validation.json`), 16 June:
- SHM numerically vs `lz.eta0`: η ratio 1.0000 (v_min = 0–600), 0.9998 (700), 1.0010 (750), 1.0126 (780), 1.048 (800 km/s); numerical hard edge 809 vs 810 km/s. The last 30 km/s of the cap is resolved to a few per cent by the 96 × 64 grid (a 128 × 96 test run gave 0.992 at 800). The SHM part of every composite uses the *analytic* η, so this only touches the truncation edge of the GSE and of the v_esc reference stream.
- Isotropic stream (Sgr, σ = 30): numerical η vs P030's analytic η_s: 1.00000 (0–350), 1.00004 (400), 1.0006 (460 km/s); the closed-form speed density reproduces it identically. Windowed vs full-range evaluation for S1: η identical to 10⁻¹⁶.
- Rates: my SHM Δη vs `lz.wd_halo(day_of_year = 167.89)` on the same kernels: N ratio 0.990 / 0.986 / 0.976 / 0.962 / 0.944 / 0.944 at δ = 250 / 300 / 330 / 350 / 366 / 380 keV (`rate_validation.json`) — the same 1–6 % bin-averaging offset found by P018 and P030.
- SHM counts (Higgsino, 16 June): 8416 / 1154 / 187.6 / 27.48 / 2.813 / 0.3126 events at δ = 250 / 300 / 330 / 350 / 366 / 380 keV (P007/P030: 1155 / 27.5 / 0.31).

## 4. Time dependence of v_lab: phase and amplitude (Part 2)

Sampling: 31 days (12-day step) + 16 June 21:22 + 2 December; first-harmonic least-squares fit v_lab(t) = a₀ + A cos ω(t − t_peak). Fit residual rms ≤ 0.6 km/s (0.04–0.4 for the fast components). Table (`modulation_phase_deltamax.csv`, `vlab_timeseries.json`, Fig. 1):

| component | v_lab(16 Jun) | ⟨v_lab⟩ (fit) | amplitude A | peak day (date) | annual range | σ_∥ | δ_max mono / +2σ_∥ / hard edge (16 Jun) [keV] | δ_max at own max (2σ / hard, year) | E_max elastic (16 Jun / +2σ) |
|---|---|---|---|---|---|---|---|---|---|
| SHM (|v_obs|) | 266.0 | 252.1 | 14.54 | 152.5 (1 Jun) | 237.4–266.4 | 168.3 | — / — / **387.3** | 387.7 | 153 / 785 |
| Sgr −W | 386.8 | 397.5 | 11.15 | 4.0 (4 Jan) | 386.3–408.5 | 30 | 40 / 89 / 190 | 107 / 208 | 323 / 432 |
| Sgr +W | 414.5 | 385.8 | **29.29** | **165.3 (14 Jun)** | 356.0–414.5 | 30 | 62 / 112 / 213 | 112 / 213 | 372 / 487 |
| GSE (zero mean) | 266.0 | 252.1 | 14.54 | 152.5 | as SHM | 84.8 | — / 80 / 372 | 374 | 153 / 410 |
| S1 | 565.0 | 554.1 | 11.08 | 159.3 (8 Jun) | 543.0–565.1 | 27.6 | 186 / 231 / 348 | 231 / 354 | 690 / 832 |
| S2 | 252.7 | 272.6 | 19.84 | 358.6 (25 Dec) | 252.5–292.2 | 37.5 | < 0 / < 0 / 122 | 23 / 159 | 138 / 232 |
| Helmi +W | 292.3 | 263.6 | 29.49 | 166.3 (15 Jun) | 233.3–292.3 | 30 | < 0 / 11 / 113 | 11 / 113 | 185 / 268 |
| Helmi −W | 258.8 | 277.6 | 18.87 | 361.2 (27 Dec) | 258.4–296.2 | 30 | < 0 / < 0 / 86 | 14 / 116 | 145 / 220 |
| Shards | 555.5 | 541.1 | 14.84 | 155.7 (5 Jun) | 526.1–555.8 | 30.1 | 178 / 228 / 338 | 228 / 339 | 667 / 819 |
| v_esc retrograde (ref.) | 809.1 | 795.0 | 14.61 | 152.5 | 780.4–809.6 | 20 | 386.6 / 419.5* / 386.5 | 387.3 | 1415 / 1559 |

"hard edge" = largest lab speed with f > 10⁻⁸ f_max, i.e. the truncated 6σ tail; *the +2σ value for the v_esc stream ignores truncation and is unphysical (P030). σ_∥ = dispersion along the lab-velocity direction: for S1 it is only 27.6 km/s (the direction is nearly along −V, where σ_φ = 27), not the 60 km/s P030 used after isotropising.

Observations.
- The modulation phase of a stream is set by the direction of **u** − **v**_⊙ relative to the ecliptic: components whose relative velocity is nearly anti-parallel to the solar motion (S1, shards, v_esc reference) inherit the SHM's phase (peak days 152–159) and amplitude (11–15 km/s). Vertical streams (Sgr, Helmi) have their relative-velocity direction close to the ecliptic plane and therefore amplitudes up to 29.3–29.5 km/s (the full orbital speed) with phases set by the sign of W: **Sgr +W and Helmi +W peak on 14–15 June, two weeks after the SHM, i.e. within a day of the event; Sgr −W, S2, Helmi −W peak in late December/early January.**
- No known substructure comes within 240 km/s of the 801 km/s needed for δ = 380 keV at 248 keV, or within 140 km/s of the 704 km/s needed for δ = 300 keV, even at its truncated 6σ hard edge: the largest hard-edge δ_max is 348 keV (S1; 354 keV at its annual maximum), and the 2σ_∥ values are ≤ 231 keV. (P030's 194 keV for S1 was the monochromatic value; 186 keV here.) The kinematic ceiling 387 keV (P002/P006/P030) is untouched.

## 5. Rates through the year, boosts, June/December, date likelihood ratio (Part 3)

Kernels K[E, v_i] = dR/dE per unit Δη at stream speed v_i from `WD.diff_rate(..., sum_over_streams=False)` (2 keV steps, E ≤ 330 keV), N_ROI = 2.84 t·yr × ∫ ε(E) (K Δη) dE. Composite: N_c = (1 − f) N_SHM + f N_pure. Date LR (P006): p(t | M) ∝ R_M(t) with uniform livetime over the 371-day run 27 Mar 2023 – 1 Apr 2024, LR = R(t_ev)/⟨R⟩_run, rates periodically interpolated from the 33-day grid. First-harmonic amplitude a₁ and peak day fitted to R(t)/⟨R⟩ (the 12-day grid argmax would be 157 for every curve; the fit gives the phase to ≈ 0.1 d). Full table `rates_boosts_LR.csv`; rate series `rate_timeseries.json`; fits `rate_phase_fits.csv`.

**SHM alone** (1 TeV): LR(16 June) = 1.276 / 1.455 / 1.877 / 2.451 / 2.857 / 3.101 at δ = 250 / 300 / 330 / 350 / 366 / 380 keV; a₁ = 0.277 / 0.454 / 0.817 / 1.189 / 1.340 / 1.494; peak day 152.55–152.73 for all δ; June/December ratio 1.74 / 2.54 / 6.39 / 25.8 / 56.1 / ∞ (December rate zero at 380 keV; fraction of the year with non-zero rate 0.97). Cross-checks: P006 gave LR = 1.41 (300), P034 a₁ = 0.43 / 1.18 / 1.66 at 300 / 350 / 380 keV — agreement to 3 % at 300–350 keV; at 380 keV the 12-day sampling under-resolves the summer spike (a₁ 1.49 vs 1.66) and our LR(380) = 3.10 should be read as ±10 %.

**Composites at nominal f, 16 June:**

| δ [keV] | Sgr ±W (3 %) | GSE (20 %) | S1 (10 %) | shards (1 %) | S2 (1 %) | Helmi ±W (0.5 %) | v_esc ref. (1 %) |
|---|---|---|---|---|---|---|---|
| 250 | 0.970 | 0.811 | 0.920 | 0.991 | 0.990 | 0.995 | 2.36 |
| 300 | 0.970 | 0.8003 | 0.9001 | 0.9900 | 0.990 | 0.995 | 3.75 |
| 330 | 0.970 | 0.8000 | 0.9000 | 0.9900 | 0.990 | 0.995 | 12.4 |
| 350 | 0.970 | 0.8000 | 0.9000 | 0.9900 | 0.990 | 0.995 | 28.5 |
| 366 | 0.970 | 0.8000 | 0.9000 | 0.9900 | 0.990 | 0.995 | 46.7 |
| 380 | 0.970 | 0.8000 | 0.9000 | 0.9900 | 0.990 | 0.995 | 47.2 |

Every known component is a **pure deficit of exactly (1 − f)** for δ ≥ 300 keV: the pure-component rates are negligible — S1 gives 0.905 events at δ = 300 keV (vs 1154 for the SHM; boost per unit f = −0.9992), 1.1 × 10⁻⁵ at 350 keV, 2.4 × 10⁻¹¹ at 380 keV; the GSE gives 1.45 / 2.8 × 10⁻⁴ / 6.4 × 10⁻⁷; Sgr, S2, Helmi give identically zero. Only at δ = 250 keV (v_min = 610 km/s) do the fast components contribute: S1 pure 1659 events vs SHM 8416 (boost per unit f = −0.80), GSE 479 (−0.94), shards 8338 → composite 0.9908. With f over the quoted ranges the 16 June boosts span 0.90–0.70 (GSE), 0.99–0.90 (S1), 0.99–0.95 (Sgr). **This supersedes P030's S1 entries (1.09 / 1.05 / 1.67 at 300 / 350 / 380 keV):** with the anisotropic dispersion (σ_∥ = 27.6 km/s) and the v_esc truncation, S1 has no tail near 700–800 km/s (fraction of S1 above v_min(δ = 300) = 2.4 × 10⁻⁵, above v_min(350) = 2 × 10⁻¹⁰), so its effect is the (1 − f) deficit.

**Phase and LR.** Composite peak days 152.5–153.2 and a₁ within 0.6 % of the SHM's for every known component; LR ratios to the SHM: 1.0000 for δ ≥ 300 keV (all), 0.9979 (GSE) / 1.0058 (S1) / 1.0004 (shards) at δ = 250 keV. **No known substructure makes 16 June a preferred date beyond the SHM's 1–2 June peak** — the two components that *do* peak on 14–15 June (Sgr +W, Helmi +W) are 400–550 km/s too slow to enter the inelastic rate. The reference v_esc-stream composite (P030's best case, f = 1 %) *lowers* LR: 0.762 / 0.768 / 0.863 / 0.978 / 0.881 × the SHM value at δ = 300 / 330 / 350 / 366 / 380 keV, because a cold stream anti-parallel to **v**_⊙ has the same phase but a weaker relative modulation (a₁ = 0.033 / 0.45 / 1.02 / 1.30 / 1.39 pure; June/Dec 1.26 / 2.73 / 14.2 / 33.6 at 300–366 keV vs 2.54 / 6.39 / 25.8 / 56.1). Thus the one stream configuration that can boost the high-δ *rate* ×47 (Table above; P030: ×51 with a finer cap resolution) weakens the *timing* evidence by 12–24 %.

**Elastic channel** (`elastic_248keV_ratios.csv`, `elastic_248_dateLR.csv`; K at E = 248 keV, δ = 0, v_min = 339 km/s): the SHM dR/dE(248 keV) modulates with a₁ = 4.24 %, peak day 152.5, LR(16 June) = 1.040. The Sagittarius streams *can* produce 248 keV recoils (E_max = 323–372 keV; pure-stream dR/dE(248) = 1.95 / 1.90 × the SHM's, S1 1.38 ×, shards 1.41 ×), raising the composite level by +2.8 % (Sgr, f = 3 %), +3.8 % (S1, 10 %), +0.4 % (shards) and the 200–270 keV band by +2.3 / +2.2 / +0.2 %. Sgr +W alone modulates with a₁ = 5.9 % peaking on day 165 (14 June), Helmi +W with a₁ = 1.43 (a spike, its speed straddling v_min) peaking on day 166 — but the composite phase moves only to day 153.5–153.7 and the LR ratios are 0.998 (Sgr −W), 0.999 (Sgr +W), 0.992 (S1), 1.001 (Helmi +W), 1.007 (GSE), 0.999 (shards): **≤ 0.8 % changes in either direction.**

## 6. The Gaia Sausage (Part 4; `sausage.json`, Fig. 3)

Composite 0.8 SHM + 0.2 GSE (β = 0.9), 16 June. η_GSE/η_SHM = 1.11 (v_min = 300), 0.89 (400), 0.64 (500), 0.27 (600), 0.072 (650), 0.0055 (700), 8.6 × 10⁻⁵ (750 km/s): the radially elongated Sausage has few particles moving anti-parallel to the solar motion (that direction is its σ_φ = 84 km/s axis), so η_comp/η_SHM = 1.023 / 0.977 / 0.927 / 0.854 / 0.814 / 0.801 / 0.8000 at 300 / 400 / 500 / 600 / 650 / 700 / 750 km/s — **reproducing P018's 0.80 above 650 km/s**. Its truncated hard edge on 16 June is 791 km/s (δ_max 372 keV), 19 km/s below the SHM's.

| δ [keV] | N ratio (16 Jun) | annual ratio | June/Dec SHM → comp | a₁ SHM → comp | peak day SHM → comp | LR SHM → comp |
|---|---|---|---|---|---|---|
| 250 | 0.811 | 0.813 | 1.737 → 1.750 | 0.277 → 0.282 | 152.55 → 151.19 | 1.2758 → 1.2731 |
| 300 | 0.8003 | 0.8003 | 2.5418 → 2.5425 | 0.4542 → 0.4544 | 152.56 → 152.52 | 1.4550 → 1.4549 |
| 330–380 | 0.8000 | 0.8000 | unchanged (≤ 10⁻⁵) | unchanged | unchanged | unchanged |

So at the LZ-relevant splittings the Sausage is a pure 20 % normalisation deficit with **no effect on the June/December ratio or the phase** (the composite tail is 0.8 × SHM). Its anisotropy does change the *elastic, low-v_min* modulation, where the recalled "SHM++ modulation differs by 10–20 %" applies: at E_R = 10 / 50 keV (1 TeV) a₁ rises from 2.84 → 3.04 % (+7 %) and 1.48 → 1.65 % (+12 %) with the December phase (peak day 335.2 → 335.9) unchanged; at 248 keV (June phase) a₁ = 4.24 → 4.94 % (+17 %), peak day 152.5 → 153.7. These are the *only* modulation changes any known substructure produces.

## 7. Directional signature (Part 5; `directional.csv`, `directional_summary.json`)

Arrival direction = direction of the DM's Earth-frame velocity (recoils are forward, along it); for inelastic scattering at threshold cos θ_R = v_min/w, so the recoil lies within arccos(v_min/v_max) of the arrival direction: **≤ 29.7° / 19.3° / 8.5° for δ = 300 / 350 / 380 keV at 248 keV on 16 June** (68 % of the SHM tail: 18.6° / 12.3° / 5.4°). Density-weighted mean arrival directions and 68 % cones on 16 June:

| component | mean arrival (l, b) | angle to SHM wind [16 Jun; annual range] | 68 % cone (all particles) | v > 339 km/s (elastic 248 keV): fraction, cone, 68 % recoil angle | v > v_min(δ = 300/350/380): fraction |
|---|---|---|---|---|---|
| SHM | (267.6°, +4.0°) | 0 | 49.9° | 0.557, 42.4°, 46.7° | 7.5 × 10⁻³ / 9.9 × 10⁻⁴ / 3.0 × 10⁻⁵; cones 25.7° / 18.2° / 8.9° |
| GSE | (268.2°, +4.0°) | 0.6° | 46.7° | 0.550, 51.2°, 42.4° | 3.2 × 10⁻⁵ / 2 × 10⁻⁸ / 2 × 10⁻¹¹ |
| Sgr −W | (267.6°, −46.7°) | 50.7° [46.7–50.8] | 7.0° | 0.955, 7.0°, 33.0° | 0 |
| Sgr +W | (267.6°, +50.2°) | 46.2° [46.2–56.6] | 7.0° | 0.996, 7.0°, 38.1° | 0 |
| S1 | (271.9°, −5.5°) | 10.5° [4.7–10.9] | 10.7° | 1.000, 10.7°, 54.7° | 2.4 × 10⁻⁵ / 2 × 10⁻¹⁰ / 0 |
| shards | (268.9°, +1.9°) | 2.4° [1.8–5.3] | 8.9° | 1.000, 8.9°, 54.0° | 2 × 10⁻⁶ / 4 × 10⁻¹¹ / 0 |
| S2 | (267.1°, −66.0°) | 70.0° [67.6–70.8] | 9.1° | 0.014, 6.8°, 16.9° | 0 |
| Helmi +W / −W | (264.5°, ±66°) | 62.8° / 67.5° | 8.9° / 10.7° | 0.074 / 0.005 | 0 |
| v_esc ref. | (267.5°, +0.2°) | 3.8° [3.8–4.7] | 1.5° | 1.000, 1.5°, 65.0° | 1 / 0.98 / 0.29; cone 1.5° |

Angle between the substructure's Galactic velocity and the solar motion: Sgr 92°/88°, S1 165°, shards 177°, S2 58°, Helmi 57°/61°. Interpretation: an inelastic recoil near δ_max (any halo) must arrive within ≈ 12° of the anti-apex (analytic cap half-angle for w = 801 km/s: 12.2°) and recoil within 8.5° of that — a needle in Galactic direction (l, b) ≈ (268°, +4°), the same for the SHM, the Sausage-composite and a v_esc stream (which merely narrows the cone to 1.5°). By contrast the substructures that *can* give elastic 248 keV recoils — the Sagittarius streams — would arrive from b ≈ ±47–50°, 46–57° away from the SHM wind, in a 7° cone with a 33–38° recoil opening: a directional detector distinguishes "Sagittarius elastic" from "halo inelastic" at a glance, while S1 and the shards (10° and 2.4° from the wind) do not separate from the SHM dipole.

## 8. Figures

- `figures/P055_fig1_vlab_year.png` — (a) v_lab(t) of each component and the Earth speed; (b) v_lab − ⟨v_lab⟩: the SHM/S1/shards/v_esc family peaks 1–8 June with 11–15 km/s amplitude; Sgr +W peaks 14 June with 29 km/s; Sgr −W peaks in January.
- `figures/P055_fig2_boost_LR.png` — (a) composite/SHM in-ROI inelastic rate vs δ (16 June): all known components ≤ 1 (0.80–0.995); only the hypothetical v_esc stream boosts. (b) LR(16 June) vs δ: known components indistinguishable from the SHM; the v_esc stream lowers LR.
- `figures/P055_fig3_rate_year.png` — normalised rate vs day for δ = 300/350/380 keV: SHM, +20 % Sausage (identical) and +1 % v_esc stream (flatter).

## 9. Failed or abandoned approaches

- First full run with a 128 × 96 angular grid, 6-day sampling and the numerical grid for every component (including isotropic ones): 2.5 min per component, > 10 min total — stopped; replaced by the exact closed-form speed density for isotropic untruncated streams, a windowed 96 × 64 grid for anisotropic ones and 12-day sampling (33 days). The 128 × 96 test gave SHM η(800)/analytic = 0.992 vs 1.048 at 96 × 64; the SHM in composites is analytic, so no result depends on this.
- The 12-day-grid argmax as the rate peak day (157 for everything) was replaced by first-harmonic fits.
- A two-lobe (±250 km/s, σ = 80) Sausage variant (P030) was not rerun; P030's 0.80 at δ ≥ 300 keV coincides with the SHM++ form here.

## 10. Caveats

Recalled kinematics and density fractions (Table 1) — S2, Helmi and shard parameters are uncertain; Sagittarius' local density is contested (0–5 %); the S1 fraction of 10 % is an upper claim. Gaussian (truncated) velocity distributions; ρ₀ fixed (the substructure replaces halo density). O₁ shapes at 1 TeV only (P018: rate ratios hamiltonian-insensitive to 0.02 dex). Uniform livetime in LR (P006: gaps change LR by < 4 %). 12-day sampling under-resolves the δ = 380 keV spike (a₁ 1.49 vs P034's 1.66). Cap resolution of the numerical grid near v_esc + v_E: ≤ 5 % in η at 800 km/s, affecting only the GSE hard edge and the v_esc reference (boost 47 vs P030's 51).

## 11. References

LZ Collaboration, arXiv:2609.02823 (2026). K. Freese, P. Gondolo, H. J. Newberg, M. Lewis, PRL 92, 111301 (2004). C. W. Purcell, A. R. Zentner, M.-Y. Wang, JCAP 08 (2012) 027. G. C. Myeong, N. W. Evans, V. Belokurov, J. L. Sanders, S. E. Koposov, ApJL 863, L28 (2018). C. A. J. O'Hare, C. McCabe, N. W. Evans, G. Myeong, V. Belokurov, PRD 98, 103006 (2018). C. A. J. O'Hare, N. W. Evans, C. McCabe, G. Myeong, V. Belokurov, PRD 101, 023006 (2020). N. W. Evans, C. A. J. O'Hare, C. McCabe, PRD 99, 023012 (2019). L. Necib, M. Lisanti, V. Belokurov, ApJ 874, 3 (2019). A. Helmi, S. D. M. White, P. T. de Zeeuw, H. Zhao, Nature 402, 53 (1999). H. H. Koppelman et al., A&A 625, A5 (2019). S. K. Lee, M. Lisanti, B. R. Safdi, JCAP 11 (2013) 033. C. McCabe, JCAP 02 (2014) 027. D. Baxter et al., EPJC 81, 907 (2021). Corpus: P002, P006, P007, P018, P030, P034.

## 12. Tools and provenance (mirrors provenance/P055.json)

- Agent tools: Read (PAPER_GUIDE; dossier; ledger (two pages); P030/P018/P006/P034/P002/P007 papers; P030_streams.py; lzcommon.py lines 108–217, 310–400; task output files; two figures), Bash (ledger summary; API listing; wimprates/WimPyDD source inspection; P006 method grep; environment versions; two full runs of the main script — first stopped, second completed in the background — and the post-processing run; table prints), Write (main script, post-processing script, details.md, provenance JSON, paper), Edit (script speed-ups and phase-fit/elastic-LR additions), Skill (dataviz; JS validator skipped per PAPER_GUIDE), ToolSearch (TaskStop), TaskStop (first over-long run).
- Software: python 3.12.13; numpy 2.5.3 (leggauss, lstsq, trapezoid); scipy 1.18.1 (special.erf/erfc); pandas 3.0.5; matplotlib 3.11.2; WimPyDD 2.0.4 (`diff_rate` per-stream kernels, `eft_hamiltonian` via `lz.wd_hamiltonian`, `streamed_halo_function` via `lz.wd_halo`, `v_earth_sun` for validation); wimprates 0.5.0 (`earth_velocity`, `j2000` for validation); numericalunits 1.28; common/lzcommon.py (eta0, vmin_kms, E_R_range_keV, delta_max_kev, wd_halo, wd_hamiltonian, LZ constants).
- Recalled items (13): J2000 equatorial→Galactic matrix (certain); obliquity 23.4393° (certain); Earth orbital speed 29.79 km/s (certain); 2023 March equinox 20 Mar 21:24 UTC (likely); Lee–Lisanti–Safdi e₁, e₂ vectors (likely); SHM++ Sausage β = 0.9, η = 0.2 and σ construction (likely); Sagittarius local debris (0, 0, ±300) km/s, σ 20–40, f ≲ 1–5 % (speed likely, sign/f uncertain); S1 (v_r, v_φ, v_z) = (−30, −297, −73), σ = (83, 27, 59), f ≤ 10 % (likely / uncertain); S2 (6, 164, −250) (uncertain); Helmi (0, 150, ±250) (likely), f (uncertain); retrograde shards v_φ ≈ −290 (uncertain); G_F, sin²θ_W (certain); LZ run window 27 Mar 2023 – 1 Apr 2024 (P006, from the paper).
- WimPyDD-generated files: none (`diff_rate` only; no response-function files written).
