# P018 — The high-velocity tail: escape velocity, non-Maxwellian tails and the reach of inelastic dark matter at LZ

Research record (simulated date 2026-09-05). Author profile: Galactic-dynamics and dark-matter astrophysicists. Category HALO.
Script: `output/code/P018_halo_tail.py` (single run, 432 s; log in `run_log.txt`). All tables in `output/work/P018/`, figures in `figures/`.

## 1. Motivation and framework

P002 showed that a 248 keV xenon recoil on 16 June 2023 is kinematically possible for inelastic DM only up to δ_max = 341/387/409 keV (400/1000/4000 GeV) and that the O₁ inelastic rate falls by 10³–10⁴ between δ = 300 keV and δ_max. P006 fixed the Earth velocity (v_E = 266 km/s on 16 June). P007 found that a pure Higgsino gives one LZ event only for δ = 358–380 keV at 1 TeV and warned that this window moves by ∓11/+9 keV for v_esc = 528/560 km/s. All three used the Baxter-2021 truncated Maxwellian (v₀ = 238, v_esc = 544 km/s) with a sharp cut-off. But inelastic scattering with δ ≥ 250 keV is sourced *only* by the tail: the smallest speed that can scatter at all is v_min,min = √(2δ/μ) = 642/703/760/792 km/s (Earth frame) for δ = 250/300/350/380 keV at 1 TeV (μ = 108.9 GeV). These are speeds where the local velocity distribution is essentially unmeasured and where models disagree by orders of magnitude. This paper quantifies how the escape velocity, the circular speed, the local density and the *shape* of the tail change (a) δ_max(248 keV) and (b) the in-ROI inelastic rate, the coupling inferred from one event, and P007's Higgsino window.

### Astrophysical inputs (all recalled)
| Item | Value used | Source (recalled) | Reliability |
|---|---|---|---|
| SHM parameters | v₀ = 238, v_esc = 544, v_sun,pec = (11.1, 12.2, 7.3) km/s, ρ₀ = 0.3 GeV/cm³ | Baxter et al. 2021 (via lzcommon) | certain |
| Gaia-era escape speed | 520–580 km/s (Piffl+2014 RAVE 533⁺⁵⁴₋₄₁; Deason+2019 Gaia 528 ± 25; Koppelman & Helmi 2021 ~500) | literature | likely |
| Circular speed range | 220–250 km/s (Eilers+2019 229; McMillan 2017 233; Reid+2014 240) | literature | likely |
| Local density range | 0.3–0.5 GeV/cm³ (0.3 classic, 0.4–0.5 recent Gaia analyses) | literature | likely |
| Smoothly truncated tail | f ∝ [exp(−v²/v₀²) − exp(−v_esc²/v_esc²)]^k, k ≈ 1.5–3.5 from N-body | Lisanti, Strigari, Wacker, Wechsler 2011 | certain (form) / likely (k range) |
| SHM++ | v₀ = 233, v_esc = 528, Sausage fraction η_S = 0.2, β = 0.9; σ_r² = 3v₀²/(2(3−2β)), σ_θ² = σ_φ² = 3v₀²(1−β)/(2(3−2β)) | Evans, O'Hare, McCabe 2019 | likely |
| Sausage orientation | radially elongated (major axis toward the Galactic centre, x), perpendicular to solar motion (y) | Belokurov+2018; Evans+2019 | certain |
| Earth orbital velocity vector | WimPyDD `v_earth_sun` (29.79 km/s, ecliptic axes) | WimPyDD 2.0.4 | certain (code) |
| Higgsino couplings | c_p = (G_F/√2)(1 − 4 sin²θ_W), c_n = −G_F/√2; G_F = 1.166e−5 GeV⁻², sin²θ_W = 0.231 | P007 / PDG | certain |

LZ inputs (paper, via lzcommon): exposure 2.84 t·yr, efficiency plateau 0.96, 50 % at 5.4 and 269.9 keV (P007 erf edges with σ_lo = 2.5, σ_hi = 11.5 keV, matching the Fig. S2 inset), event energy 248 keV, event date 16 June 2023 (day 167). WimPyDD isoscalar unit coupling c⁰ = 2/m_v² (P003 convention).

## 2. Equations

**Kinematics.** v_min(E_R, δ) = (m_N E_R/μ + δ)/√(2 m_N E_R); inverting at v = v_max = v_E + v_esc:
δ_max(E_R) = v_max √(2 m_N E_R) − m_N E_R/μ. Hence ∂δ_max/∂v_max = √(2 m_N E_R)/c = **0.8218 keV per km/s** at E_R = 248 keV (A = 131.29), independent of m_χ. v_E enters through |v_rot + v_sun,pec + v_orbit(t)|: v_sun = 232.6/250.6/262.6 km/s for v₀ = 220/238/250; the orbital term adds +14.6 (June) / −14.6 (December) km/s (cosine model of lzcommon; WimPyDD's vector gives 266.0 vs 265.1 km/s on 16 June).

**Halo function.** η(v_min, t) = ∫_{v>v_min} f_E(v, t)/v d³v with f_E(v) = f_gal(v + v_obs(t)), v_obs = v_rot + v_sun,pec + v_orbit. In Earth-frame spherical coordinates with the polar axis along −v_obs: |v n̂ + v_obs|² = v² + v_obs² − 2 v v_obs cosθ, so the truncation |u| ≤ v_esc is the cap cosθ ≥ c_min(v) = (v² + v_obs² − v_esc²)/(2 v v_obs) (clipped to [−1, 1]). Then
S(v) ≡ v ∫_cap dΩ f_gal(v n̂ + v_obs),  η(v_min) = ∫_{v_min}^{v_obs+v_esc} S(v) dv.
The integrand is smooth on the cap, so Gauss–Legendre in cosθ (48 nodes, mapped to [c_min, 1]) × uniform φ (64 nodes) converges rapidly; the anisotropic Sausage is handled by the φ dependence. Normalisation ∫_{|u|<v_esc} f_gal d³u computed with the same quadrature (400 radial GL nodes). WimPyDD convention: with grid points v_i (upper interval boundaries, 0–900 km/s in 0.5 km/s steps), Δη_i = η(v_{i−1}) − η(v_i) = ∫_{v_{i−1}}^{v_i} S dv (Simpson with the midpoint), Δη_0 = 0, and η(v_i) = Σ_{j>i} Δη_j; `diff_rate(..., vmin, delta_eta)` sums n_T v_i² dσ/dE(v_i) Δη_i. With unit weights and `sum_over_streams=False` it returns the kernel K(E, v_i) so that dR/dE = K·Δη for any halo (P007 technique).

**Distributions (Galactic frame, all truncated at |u| < v_esc):**
(i) Maxwellian f ∝ exp(−u²/v₀²); (ii) same with v_esc = 500/528/560/600; (iii) v₀ = 220/250 (v_obs changes accordingly); (iv) Lisanti form f ∝ [exp(−u²/v₀²) − exp(−v_esc²/v₀²)]^k, k = 1, 2, 3 (k = 1 is the classic "smooth" truncation; the sharp cut-off is k = 0); (v) SHM++-like: 0.8 × round Maxwellian + 0.2 × anisotropic Gaussian with (σ_r, σ_θ = σ_φ) = (266.1, 84.1) km/s for v₀ = 238, β = 0.9 (total dispersion 3v₀²/2, same as the round component), radial axis along x; (vi) the assignment's suggested approximation — Sausage replaced by an *isotropic* Maxwellian with per-axis σ = σ_r (v₀' = √2 σ_r = 376 km/s) — computed to show what the approximation does; (vii) "literal SHM++" with Evans et al. values v₀ = 233, v_esc = 528, η_S = 0.2, β = 0.9; (viii) envelope low = (v₀ 220, v_esc 500, k = 3) and envelope high = (v₀ 250, v_esc 600, sharp).

**Rates.** N = 2.84 t·yr × ∫ ε(E) (K·Δη)(E) dE, E from the kinematic onset to 330 keV in 2 keV steps. Isoscalar O₁ with (c^s m_v²)² = 1 gives N_unit; the coupling inferred from one event is (c^s m_v²)² = 1/N_unit, so log₁₀ shifts of the coupling are minus those of the rate. Higgsino: WimPyDD (c⁰, c¹) = (c_p + c_n, c_p − c_n), σ_n = 7.4 × 10⁻³⁹ cm² (P007). "Annual" = mean of Δη over 12 days (15 + 30.44 k), exactly the time-averaged rate; "June" = day 167.

**Spread metric.** For each (m, δ, epoch) the log₁₀ ratio of N to the standard halo for each variant; the *spread* of a tier is max − min over the tier (−∞ when a variant gives zero rate). Tiers: **gaia** = {std, v_esc 528, 560, v₀ 220, 250, SHM++, literal SHM++} (Gaia-era parameter range, conventional tails); **shape** = {std, k = 1, 2, 3} (tail shape at fixed v₀, v_esc); **single** = all single-parameter variants incl. v_esc 500/600; **envelope** = {env_low, env_high}. ρ₀ = 0.3–0.5 adds log₁₀(5/3) = 0.22 dex to any tier.

## 3. Validation (`validation.json`)
- Numerical normalisation of the truncated Maxwellian vs analytic π^{3/2} v₀³ [erf(z) − 2z e^{−z²}/√π]: ratio 1.000000 (2 × 10⁻¹⁴).
- v_obs(16 June) = (11.27, 265.09, −18.50) km/s, |v_obs| = 265.98 km/s, v_max = 809.98 km/s (P006: 266, 810).
- η(v_min) mine vs analytic `lz.eta0` (same |v_obs|): ratio 1.00000 at v_min = 0, 200, 400, 500, 600, 700, 750, 780 km/s and 1.00003 at 800 km/s. max|Δη_mine − Δη_WimPyDD|/η(0) = 1.8 × 10⁻⁶.
- η mine vs WimPyDD `streamed_halo_function` (same grid, same v_obs): WimPyDD/analytic = 1.0000 (0), 1.0008 (200), 1.0021 (400), 1.0029 (500), 1.0039 (600), 1.0062 (700), 1.0100 (750), 1.0183 (780), 1.0525 (800 km/s). Cause: WimPyDD's Maxwellian routine uses `eta0_ave_bin` (η averaged over each bin, i.e. a half-bin shift toward lower v_min); near v_max, where η ∝ (v_max − v)³, a 0.25 km/s shift is a 3 × 0.25/(v_max − v) relative excess = 7.5 % at v_max − v = 10 km/s. The effect is a property of WimPyDD's discretisation, not of the physics; it vanishes as the grid is refined.
- Rates, 1 TeV isoscalar, mine/WimPyDD: 0.9925 (δ = 300, June), 0.9770 (δ = 350, annual), 0.9428 (δ = 380, June). The < 1 % target is met at δ = 300 keV; at 350–380 keV the 2–6 % difference is entirely the WimPyDD tail bias just described (P002/P006/P007 rates within ~20 keV of δ_max are therefore 2–6 % high, immaterial for their conclusions).
- Higgsino cross-check with P007 (annual halo, 1 TeV): N(δ = 300) = 783.7 vs P007 790; N(350) = 10.88 vs 11.1; δ(N = 1) = 365.7 vs 366 keV; v_esc 528/560 shifts −10.9/+10.7 vs P007's −11/+9 keV. Isoscalar N_unit (annual, 1 TeV) = 1.035e5, 1.335e4, 253 at δ = 250/300/350 vs P007 1.04e5, 1.35e4, 259.
- Ratios are hamiltonian-insensitive: Higgsino (isovector-dominated) and isoscalar log₁₀ ratios agree to < 0.02 dex at all common δ (compare `rate_ratios_isoscalar.csv` with `spread_vs_delta.csv`).

## 4. Results

### 4.1 δ_max(248 keV) (`deltamax_vs_vesc.csv`, `deltamax_vs_v0.csv`, `part1_deltamax.json`)
Slope 0.8218 keV/(km/s), all masses. Standard halo: June 341.1/386.6/409.4, annual-mean 329.1/374.6/397.4, December 317.2/362.7/385.5 keV (400/1000/4000 GeV) — P002/P006 reproduced.

| m_χ (GeV) | v_esc 500, June | 544, June | 600, June | 500, Dec | 500, Dec, v₀ 220 | 600, June, v₀ 250 |
|---|---|---|---|---|---|---|
| 400 | 304.9 | 341.1 | 387.1 | 281.0 | 266.2 | 397.0 |
| 1000 | 350.5 | 386.6 | 432.6 | 326.5 | 311.7 | 442.5 |
| 4000 | 373.2 | 409.4 | 455.4 | 349.3 | 334.5 | 465.3 |

v_esc ± 50 km/s ⇒ ∓41 keV; v₀ 220–250 ⇒ ∓10 keV (through v_sun); season ±12 keV. Full envelope at 1 TeV: 312–443 keV. Note that at the Gaia-favoured v_esc ≈ 528 km/s, δ_max(248 keV, 1 TeV, June) = 373 keV, so the (1000 GeV, 350 keV) Table-S7 point remains allowed for all v_esc ≥ 500 km/s but the (400 GeV, 350 keV) dash persists for v_esc ≤ 555 km/s (June) — at v_esc = 600 km/s it would be physical (δ_max = 387 keV).

### 4.2 Tails (`eta_tails.csv`, `eta_tail_ratios_june.csv`, Fig. 1)
η/η_SHM on 16 June:

| variant | v = 400 | 600 | 700 | 750 | 780 | 800 km/s | v_max |
|---|---|---|---|---|---|---|---|
| v_esc 500 | 0.987 | 0.869 | 0.546 | 0.113 | 0 | 0 | 766 |
| v_esc 528 | 0.996 | 0.960 | 0.848 | 0.636 | 0.260 | 0 | 794 |
| v_esc 560 | 1.003 | 1.033 | 1.133 | 1.359 | 1.969 | 5.63 | 826 |
| v_esc 600 | 1.008 | 1.089 | 1.378 | 2.114 | 4.454 | 23.0 | 866 |
| v₀ 220 | 0.831 | 0.551 | 0.390 | 0.256 | 0.081 | 0 | 792.5 |
| v₀ 250 | 1.098 | 1.370 | 1.650 | 2.006 | 2.784 | 6.96 | 822 |
| k = 1 | 0.973 | 0.808 | 0.557 | 0.347 | 0.188 | 0.066 | 810 |
| k = 2 | 0.607 | 0.097 | 0.0176 | 0.0043 | 9.5e−4 | 9.8e−5 | 810 |
| k = 3 | 0.393 | 0.0114 | 5.0e−4 | 4.5e−5 | 4.0e−6 | 1.2e−7 | 810 |
| SHM++ (anisotropic Sausage) | 0.977 | 0.855 | 0.801 | 0.800 | 0.800 | 0.800 | 810 |
| SHM++ (isotropic σ_r approx.) | 1.065 | 1.380 | 1.737 | 1.979 | 2.141 | 2.255 | 810 |
| literal SHM++ (233/528) | 0.926 | 0.694 | 0.531 | 0.343 | 0.073 | 0 | 789 |
| envelope low | 0.225 | 1.5e−3 | 5.1e−6 | 0 | 0 | 0 | 748.5 |
| envelope high | 1.108 | 1.496 | 2.225 | 3.801 | 8.779 | 49.6 | 878 |

The Sausage result deserves emphasis. High Earth-frame speeds require Galactic-frame velocities nearly antiparallel to v_obs, i.e. along −y (tangential); the Sausage has σ_φ = 84 km/s there, so its contribution to the tail above ~650 km/s is negligible and η → (1 − η_S) η_round = 0.80 η_SHM. The "broader radial Maxwellian" approximation (isotropic σ_r = 266 km/s) puts the radial dispersion into all directions and predicts an *enhanced* tail (×1.7–2.3) — the opposite sign. The anisotropy, not the dispersion, is what matters at the tail; we use the anisotropic calculation as the SHM++ result and quote the isotropic one only as a caution.

### 4.3 O₁ inelastic rates, isoscalar unit coupling (`rate_ratios_isoscalar.csv`, `rate_spread_summary.csv`)
N_unit for (c^s m_v²)² = 1, standard halo (June | annual): 400 GeV: 1.186e5 | 8.65e4 (δ = 250), 7738 | 3893 (300), 6.1e−4 | 1.0e−4 (350), 0 (380); 1000 GeV: 1.329e5 | 1.035e5, 19171 | 13354, 613.6 | 253.3, 2.428 | 0.6156; 4000 GeV: 5.167e4 | 4.112e4, 8977 | 6692, 701.1 | 381.4, 36.99 | 13.38. Coupling from one event (c^s m_v²)² = 1/N_unit: 1 TeV June 7.5e−6, 5.2e−5, 1.63e−3, 0.41 at δ = 250/300/350/380.

log₁₀(N/N_SHM), 1 TeV, 16 June (annual in parentheses where different by > 0.05):

| δ (keV) | v_esc 500 | 528 | 560 | 600 | v₀ 220 | v₀ 250 | k=1 | k=2 | k=3 | SHM++ | SHM++ iso | literal SHM++ | env low | env high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 | −0.18 | −0.06 | +0.05 | +0.16 | −0.36 | +0.20 | −0.19 | −1.41 | −2.63 | −0.09 | +0.21 | −0.24 | −3.86 | +0.37 |
| 300 | −0.43 (−0.55) | −0.11 | +0.08 | +0.22 | −0.47 | +0.25 | −0.33 | −1.98 | −3.65 | −0.10 | +0.27 | −0.34 | −6.06 | +0.45 |
| 350 | −2.85 (−3.11) | −0.56 | +0.35 | +0.82 (+1.02) | −1.03 (−1.14) | +0.49 | −0.71 | −2.93 | −5.19 | −0.10 | +0.33 | −1.00 (−1.13) | −∞ | +1.15 (+1.37) |
| 380 | −∞ | −1.60 | +1.16 | +2.38 (+2.68) | −2.20 | +1.15 | −1.10 | −3.69 | −6.28 | −0.10 | +0.35 | −2.43 | −∞ | +2.82 (+3.16) |

4000 GeV, June: v_esc 528/560: −0.08/+0.06 (300), −0.28/+0.20 (350), −0.77/+0.48 (380); v₀ 220/250: −0.41/+0.22, −0.69/+0.36, −1.27/+0.59; k = 1: −0.26, −0.53, −0.80; k = 3: −3.24, −4.48, −5.45; SHM++ −0.10 throughout. 400 GeV, June: v_esc 528/560: −0.34/+0.22 (300); at δ = 350 the standard rate is only 6 × 10⁻⁴ events (kinematic edge) and ratios span 10 dex.

Tier spreads (dex), 16 June:

| m (GeV) | δ | N_SHM | gaia | shape | single | envelope |
|---|---|---|---|---|---|---|
| 400 | 250 | 1.19e5 | 0.67 | 3.28 | 3.52 | 5.62 |
| 400 | 300 | 7738 | 1.15 | 4.79 | 5.30 | ∞ |
| 1000 | 250 | 1.33e5 | 0.56 | 2.63 | 2.83 | 4.22 |
| 1000 | 300 | 19171 | 0.72 | 3.65 | 3.91 | 6.51 |
| 1000 | 350 | 613.6 | 1.52 | 5.19 | 6.01 | ∞ |
| 1000 | 380 | 2.43 | 3.59 | 6.28 | ∞ | ∞ |
| 4000 | 300 | 8977 | 0.63 | 3.24 | 3.46 | 5.39 |
| 4000 | 350 | 701.1 | 1.05 | 4.48 | 4.97 | 9.57 |
| 4000 | 380 | 37.0 | 1.88 | 5.45 | 6.56 | 24 |

### 4.4 Higgsino window (`higgsino_N_vs_delta.csv`, `higgsino_window_shift.csv`, Fig. 3)
δ(N = 1) [δ(N = 3.65) – δ(N = 0.105), the 90 % one-event band] in keV:

| variant | 400 GeV annual | 1000 GeV annual | 1000 GeV June | 4000 GeV annual |
|---|---|---|---|---|
| SHM (std) | 330.3 [325.0–337.1] | **365.7 [357.6–379.5]** | 372.1 [364.2–386.4] | 376.3 [365.7–393.8] |
| v_esc 500 | 300.0 | 335.3 [328.9–346.1] | 341.1 | 347.9 |
| v_esc 528 | 319.2 | 354.8 [347.4–367.5] | 361.0 | 366.1 |
| v_esc 560 | 341.3 | 376.4 [367.7–391.3] | 383.0 | 386.2 |
| v_esc 600 | 368.2 | 402.1 [391.7–419.9] | 409.2 | 409.8 |
| v₀ 220 | 314.6 | 348.6 [340.7–361.1] | 354.8 | 358.5 |
| v₀ 250 | 340.4 | 376.9 [368.5–391.5] | 383.7 | 387.7 |
| k = 1 | 321.3 | 354.1 [345.5–367.4] | 360.6 | 363.3 |
| k = 2 | 292.3 | 317.4 [304.8–335.3] | 324.5 | 320.2 |
| k = 3 | 256.5 | 277.7 [263.3–299.6] | 284.7 | 277.5 |
| SHM++ (anisotropic) | 329.5 | 364.3 [356.1–378.1] | 370.7 | 374.5 |
| SHM++ (isotropic approx.) | 332.8 | 370.4 | 377.2 | 382.2 |
| literal SHM++ (233/528) | 314.0 | 349.0 [341.5–361.3] | 355.1 | 359.8 |
| envelope low | 222.6 | 243.4 [230.6–263.0] | 250.6 | 244.7 |
| envelope high | 379.2 | 414.5 [403.8–433.1] | 421.9 | 422.6 |

Shifts of δ(N = 1) at 1 TeV (annual): v_esc 500/528/560/600: −30.4/−10.9/+10.7/+36.4; v₀ 220/250: −17.1/+11.2; k = 1/2/3: −11.6/−48.3/−88.0; SHM++ −1.3; literal SHM++ −16.7; envelope −122/+49 keV. The union of the Gaia-tier 90 % bands at 1 TeV is 341–391 keV (vs P007's 358–380); including k ≤ 3 it is 263–391 keV. Shifts are nearly mass-independent (v_esc 528: −11.1/−10.9/−10.2 at 400/1000/4000 GeV) except k = 3 (−74/−88/−99).

### 4.5 Where astrophysics dominates (`spread_vs_delta.csv`, `astro_dominance_delta.csv`, `astro_dominance_extra.json`, Fig. 2)
Gaia-tier spread vs δ at 1 TeV (June): 0.21 (δ = 100), 0.37 (150), 0.50 (200), 0.55 (250), 0.75 (300), 1.29 (340), 1.58 (350), 1.89 (360), 2.05 (370), 2.79 (380), ∞ (≥ 390, v_esc 528 forbids 248 keV). Shape-tier spread: 0.76 (100), 1.54 (150), 2.10 (200), 2.64 (250), 3.72 (300), 5.14 (350), 5.87 (380).

First δ at which a tier's spread exceeds a threshold (June | annual), keV:

| m (GeV) | gaia ×3 | gaia ×5 (nuclear-response scale) | gaia ×10 | gaia ×35 (one-event 90 % Poisson band) | shape ×10 | shape ×100 | envelope ×100 |
|---|---|---|---|---|---|---|---|
| 400 | 167 \| 162 | 261 \| 252 | 291 \| 284 | 309 \| 304 | 105 \| 101 | 160 \| 154 | 122 \| 117 |
| 1000 | 188 \| 182 | 293 \| 285 | 324 \| 317 | 349 \| 344 | 115 \| 110 | 189 \| 179 | 135 \| 130 |
| 4000 | 206 \| 196 | 309 \| 301 | 343 \| 335 | 373 \| 368 | 121 \| 116 | 215 \| 203 | 144 \| 138 |

The "single" and "envelope" tiers exceed ×10 at every δ ≥ 100 keV scanned (spread 0.83 and 1.19 dex already at δ = 100). Interpretation: (i) with only the Gaia-era range of (v₀, v_esc) and an SHM++ Sausage, the halo uncertainty exceeds the shell-model-vs-Helm nuclear-response uncertainty (×3–6, PAPER_GUIDE/P003) for δ ≳ 260–310 keV, exceeds ×10 for δ ≳ 290–345 keV, and exceeds the statistical information in one event (a factor 35 between 0.105 and 3.65 expected events) for δ ≳ 305–375 keV; (ii) if the functional form of the tail is left free (k = 0–3) the rate is undetermined by ×10 already at δ ≈ 100–120 keV — for any inelastic interpretation of the LZ event.

## 5. Figures
- `figures/P018_fig1_eta_tails.png` — (a) η(v_min) on 16 June for all 15 variants (log scale, 300–880 km/s), with v_min(248 keV, 1 TeV) marked for δ = 300/350/380 keV (703/760/792 km/s); (b) ratio to the sharp SHM. The SHM++ curve is flat at 0.80 above ~650 km/s; the isotropic approximation is flat at ~2.2.
- `figures/P018_fig2_rate_ratio_vs_delta.png` — in-ROI rate relative to the sharp SHM vs δ (100–420 keV) for 400/1000/4000 GeV (Higgsino kernels; ratios are hamiltonian-insensitive), 16 June; dashed lines ×10 and ×0.1; shaded: Gaia-tier spread > ×10 (δ > 291/324/343 keV); dotted vertical: δ_max(248 keV, June).
- `figures/P018_fig3_higgsino_window.png` — expected Higgsino events (1 TeV, annual mean) vs δ for each halo; grey band 0.105–3.65 events; line N = 1.

## 6. Failed / abandoned approaches
- First run used a δ grid starting at 250 keV and dropped zero-rate variants from the spread; the Gaia-tier ×10 crossing was then never bracketed (spread already > 1 dex at 250 keV for the wider tiers) and N = 0 was mis-scored as "no spread". Fixed by extending the grid to 100–440 keV and scoring N = 0 as −∞.
- Applying the assignment's isotropic "broader radial Maxwellian" for the Sausage was found to give the wrong sign of the tail effect; the anisotropic Gaussian was computed directly instead (kept as a comparison variant).
- The < 1 % agreement target with WimPyDD is met at δ = 300 keV but not within 30 km/s of v_max (2–6 %), traced to WimPyDD's bin-averaged η rather than to this construction, which matches the analytic η to 3 × 10⁻⁵ everywhere.

## 7. Discussion
The LZ inelastic interpretation is a tail measurement. At δ = 300 keV the rate comes from Earth-frame speeds above 703 km/s, i.e. Galactic-frame speeds above ~440 km/s, where the SHM's Maxwellian form is an extrapolation of a fit to the inner distribution; Gaia has measured the escape speed to ±25–50 km/s and the circular speed to ±10 km/s, and N-body haloes show tails between k ≈ 1.5 and 3.5 (Lisanti et al.). Three conclusions follow. (1) Kinematic statements are robust and cheap to correct: δ_max moves 0.82 keV per km/s of v_max, so P002's map shifts rigidly by −36 keV (v_esc = 500) to +46 keV (600). (2) Rate statements at δ ≥ 300 keV carry an irreducible astrophysical factor of ×5–30 (Gaia range) even for a sharp Maxwellian tail; P007's Higgsino window widens from 358–380 to 341–391 keV at 1 TeV. Any coupling read off LZ's two-sided intervals at δ = 300–350 keV inherits the same factor, which is larger than the nuclear-response and efficiency systematics combined. (3) The functional form of the tail is decisive: a Lisanti k = 2–3 tail lowers the rate by 10²–10⁵ and would put a Higgsino at δ ≈ 280–320 keV — inside LZ's tabulated grid where the paper's own intervals apply. Conversely, the SHM++ Sausage, often cited as making the local distribution "hotter", *reduces* the tail by 20 % because its anisotropy is perpendicular to the solar motion. Halo substructure (streams, debris flows) is not modelled here and could add localized tail power; the modulation phase (P006) is the observable that would distinguish tail shapes once several events exist, since the June/December ratio scales with the slope of η at v ≈ 700–800 km/s.

## 8. References
- LZ Collaboration, arXiv:2609.02823 (2026).
- D. Baxter et al., "Recommended conventions for reporting results from direct dark matter searches", EPJC 81, 907 (2021).
- N. W. Evans, C. A. J. O'Hare, C. McCabe, "Refinement of the standard halo model for dark matter searches in light of the Gaia Sausage", PRD 99, 023012 (2019).
- M. Lisanti, L. E. Strigari, J. G. Wacker, R. H. Wechsler, "Dark matter at the end of the Galaxy", PRD 83, 023519 (2011).
- T. Piffl et al., "The RAVE survey: the Galactic escape speed and the mass of the Milky Way", A&A 562, A91 (2014).
- C. McCabe, "The astrophysical uncertainties of dark matter direct detection experiments", PRD 82, 023530 (2010).
- A. J. Deason et al., "The local high-velocity tail and the Galactic escape speed", MNRAS 485, 3514 (2019). [recalled, likely]
- S. Kang, S. Scopel, G. Tomar, J.-H. Yoon, WimPyDD, CPC 279, 108423 (2022).
- Corpus: P002, P003 (coupling convention), P006, P007.

## 9. Tools and provenance (mirrors `output/provenance/P018.json`)
- Agent tools: Read ×16 (PAPER_GUIDE, dossier, ledger, P002/P006/P007 papers, lzcommon.py, P007 script ×2, WimPyDD package.py ×4, three figures); Bash ×17 (listing/versions, greps of P007 and WimPyDD, timing/normalisation test, two script runs, table dumps, crossing computation, wc ×4); Write ×4 (script, details, provenance, paper); Edit ×19 (6 script fixes, 13 paper word-count trims, final tool counts); Skill ×1 (dataviz).
- Software: python 3.12.13; numpy 2.5.3 (polynomial.legendre.leggauss, cumsum, trapezoid, cross); scipy 1.18.1 (special.erf/erfc); pandas 3.0.5; matplotlib 3.11.2 (Agg); WimPyDD 2.0.4 (`diff_rate` with `sum_over_streams=False`, `eft_hamiltonian`, `streamed_halo_function`, `v_earth_sun`, target `Xe`); common/lzcommon.py (LZ constants, delta_max_kev, vmin_kms, E_R_range_keV, v_earth_kms, eta0, wd, wd_halo, wd_hamiltonian, XE_ISOTOPES, M_V_GEV).
- Script and command: `output/code/P018_halo_tail.py`; `.venv/bin/python output/code/P018_halo_tail.py` (432 s).
- Local inputs: `output/provenance/PAPER_GUIDE.md`; `output/00_evidence_dossier.md`; `output/results_ledger.csv`; `output/papers/P002.md`, `P006.md`, `P007.md`; `output/work/P007/details.md` (halo definitions, δ(N=1) table); `output/code/P007_higgsino_inelastic.py` (efficiency model, kernel technique, Higgsino couplings); `output/code/common/lzcommon.py`; `WimPyDD/package.py` (read-only: `diff_rate`, `streamed_halo_function[_maxwellian]`, `v_earth_sun`); `environment/ENVIRONMENT_versions.txt`.
- Recalled knowledge: 9 items (table in §1).
- Datasets: none. Data requests: none. WimPyDD-generated files: none (no `Halo_functions/` output; `diff_rate` used directly).
