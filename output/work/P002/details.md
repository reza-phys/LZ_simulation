# P002 — Kinematic map of inelastic dark matter consistent with a 248 keV xenon recoil: research record

Simulated date 2026-09-03. Author profile: dark-matter phenomenology group. Category IDM (hep-ph).
Script: `output/code/P002_inelastic_kinematics.py` (run from the simulation root with `.venv/bin/python`). All numbers below are read from
`output/work/P002/results.json`, the CSV tables in `output/work/P002/` and the run log `run_log.txt`, all produced by that script.

## 1. Motivation and framework

LZ (arXiv:2609.02823) reports one NR-like event at E_R = 248 ± 23 (stat) ± 23 (sys) keV, recorded 16 June 2023, and tests inelastic
O₁ and O₄ models on a grid m_χ ∈ {400, 1000, 4000} GeV × δ ∈ {0, 50, …, 350} keV (LEE section, fulltext lines 491–493). Table S7 shows a
dash ("not physical") at (400 GeV, 350 keV) and the significance rising monotonically with δ up to the edge of the grid; Table S6 shows zero
significance for all Lagrangians at m_χ ≤ 50 GeV. Two purely kinematic questions follow:

1. For which (m_χ, δ) can a 248 ± 32 keV xenon recoil (stat ⊕ sys = 32.5 keV) be produced at all by a halo particle on 16 June 2023?
2. For the allowed δ, where does the inelastic recoil spectrum actually sit relative to 248 keV, and how fast does the rate collapse as δ → δ_max?

Framework: endothermic inelastic scattering χ N → χ* N with mass splitting δ = m_χ* − m_χ > 0 (Tucker-Smith & Weiner 2001). Standard Halo Model
with the Baxter et al. (2021) parameters that LZ states it adopted: v₀ = 238 km/s, v_esc = 544 km/s, solar peculiar velocity (11.1, 12.2, 7.3) km/s,
ρ₀ = 0.3 GeV/cm³ (recalled, reliability: certain; encoded in `lzcommon.py`). Earth's speed in the halo frame from `lz.v_earth_kms(doy)` =
|v_sun| + 15.0 cos(2π(doy − 153)/365.25): 265.1 km/s on 16 June (day 167), 250.6 km/s annual average, 235.6 km/s on 2 December (day 336).
Hence v_max = v_E + v_esc = 809.1 / 794.6 / 779.6 km/s (June / average / December). Variations: v_esc = 500 and 600 km/s (v_max = 765.1, 865.1 km/s in June).
WimPyDD's own Earth-velocity model gives v_E(day 167) = 266.5 km/s (its June halo function vanishes above 810.5 km/s), 1.4 km/s above the cosine
model; this shifts δ_max by ≈ 1 keV and is ignored.

## 2. Equations

Minimum speed for recoil E_R on a nucleus of mass m_N with reduced mass μ = m_χ m_N/(m_χ + m_N):

  v_min(E_R, δ) = (1/√(2 m_N E_R)) (m_N E_R/μ + δ).                                              (1)

Setting v_min = v_max and solving for δ gives the largest splitting able to produce E_R:

  δ_max(E_R, m_χ, v_max) = v_max √(2 m_N E_R) − m_N E_R/μ.                                            (2)

Maximising (2) over E_R gives the absolute ceiling δ_abs = μ v_max²/2 at E* = μ² v_max²/(2 m_N).   (3)

Recoil-energy interval at speed v for splitting δ:

  E_± = (μ² v²/m_N) [1 − δ/(μ v²) ± √(1 − 2δ/(μ v²))].                                               (4)

The minimum mass for given (E_R, δ) is obtained by bisection on v_min(E_R, m_χ, δ) = v_max (`lz.m_chi_min_gev`). Unless stated, m_N is the
abundance-weighted natural-xenon mass A = 131.29 u (`lz.A_XE_MEAN`); the per-isotope spread is given in §3.4.

Differential rate (WimPyDD, `WD.diff_rate` via `lz.wd_rate`): dR/dE_R = Σ_isotopes (ρ₀/m_χ) (1/2 m_χ ... ) Σ_τ,τ' c^τ c^τ' F^{ττ'}_M(q²) η(v_min(E_R, δ)),
with the shell-model M response of Anand, Fitzpatrick & Haxton (2014) and η(v_min) = ∫_{v_min} f(v)/v d³v built as a streamed halo function
(`lz.wd_halo`). Coupling: O₁ isoscalar, c₁⁰ = 1/m_v², m_v = 246.2 GeV (Fig. 1 caption); c₁¹ = 0. The absolute normalisation convention of
WimPyDD's c⁰ relative to Anand et al. (a possible factor 2 in c, 4 in rate) is being settled by P003 — every rate here is quoted at
"WimPyDD unit coupling" and only ratios and shapes are used for conclusions.

## 3. Part 1 — kinematic map (files `kinematics_*.csv`, `figures/fig1_kinematic_map.png`)

### 3.1 δ_max(248 keV) and the absolute ceiling

| v_max scenario | v_max [km/s] | δ_max(248; 400 GeV) | δ_max(248; 1000 GeV) | δ_max(248; 4000 GeV) | δ_max(248; m→∞) |
|---|---|---|---|---|---|
| 16 June, v_esc 544 | 809.1 | 341.1 | 386.6 | 409.4 | 417 |
| annual average | 794.6 | 329.1 | 374.6 | 397.4 | 405 |
| December | 779.6 | 316.8 | 362.3 | 385.1 | 393 |
| June, v_esc 500 | 765.1 | 304.9 | 350.5 | 373.2 | — |
| June, v_esc 600 | 865.1 | 387.1 | 432.6 | 455.4 | — |

For E_R = 216 / 280 keV (June): δ_max = 338.5 / 340.9 keV (400 GeV), 378.1 / 392.3 keV (1000 GeV), 398.0 / 418.0 keV (4000 GeV).
Absolute ceiling δ_abs = μ v_max²/2 (June): 341.3 keV (400 GeV, attained at E* = 261 keV), 397.1 keV (1000 GeV, E* = 354 keV),
432.5 keV (4000 GeV, E* = 420 keV), 445 keV (m → ∞). At 400 GeV the observed 248 keV is within 13 keV of the energy that maximises the
reachable splitting, so δ_max(248) ≈ δ_abs there; at 1000 GeV the ceiling is 10 keV above δ_max(248).

### 3.2 Minimum masses

Elastic (δ = 0), June: m_min = 65.3 / 72.8 / 80.3 GeV for E_R = 216 / 248 / 280 keV (74.9 GeV for 248 keV with the annual-average halo,
77.3 GeV in December). Inelastic, E_R = 248 keV, June: m_min = 95.8 (δ = 100), 139.9 (200), 259.5 (300), 453.2 (350), 821 (380),
1789 GeV (400 keV). Within the ±1σ energy band: m_min(δ = 300) = 253–271 GeV, m_min(δ = 350) = 448–484 GeV. Annual average: m_min(300) = 289,
m_min(350) = 552 GeV; December: 328 and 711 GeV; v_esc = 500: 376 and 985 GeV; v_esc = 600: 186 and 269 GeV. (`kinematics_min_mass.csv`)

### 3.3 The LZ grid and the tables

`kinematics_grid_points.csv`: all 24 (m_χ, δ) grid points are allowed for E_R = 216, 248 and 280 keV in June, in the annual average and in
December, except (400 GeV, 350 keV), which is forbidden at every energy and in every halo scenario with v_esc = 544 km/s
(δ_abs(400 GeV) = 341.3 keV < 350 keV; m_min(350 keV) = 453 GeV). This reproduces the dash in Table S7 and shows it is robust to the ±32 keV
energy uncertainty and to the day of the year; only v_esc ≳ 555 km/s (June) would open it. The kinematic recoil window at v_max (June) for
each grid point (Eq. 4) is listed in the CSV; e.g. 1000 GeV: 90–790 keV at δ = 300, 152–640 keV at δ = 350; 400 GeV: 111–475 keV at δ = 300.

Table S6 (elastic Lagrangians): the maximum elastic recoil in June, E_+ = 2 μ² v_max²/m_N, is 10.2, 14.2, 18.8, 26.5, 38.2, 69.1, 108.2
and 150.0 keV for m_χ = 10 … 50 GeV — all below 216 keV — and 360.5 keV at 100 GeV (`kinematics_tableS6_masses.csv`). A 248 ± 32 keV recoil
therefore cannot be produced by any m_χ ≤ 50 GeV WIMP regardless of operator, which is exactly where every row of Table S6 is 0.0; the
switch-on between 50 and 100 GeV (m_min = 72.8 GeV) is kinematic, not spectral.

### 3.4 Isotope dependence (`kinematics_isotopes.csv`)

δ_max(248 keV, June) ranges from 369.4 keV (¹²⁴Xe) to 397.1 keV (¹³⁶Xe) at 1000 GeV (381.1 keV for ¹²⁹Xe, 388.0 keV for ¹³²Xe), and from
326.4 to 350.0 keV at 400 GeV. For ¹³⁶Xe (8.9 % abundance) δ_abs(400 GeV) = 350.3 keV, so a ¹³⁶Xe recoil of ≈ 265 keV at (400 GeV, 350 keV)
is allowed by ~0.3 keV. The "not physical" entry is thus exact for the mean nucleus and for the dominant isotopes, and the true rate at that
grid point is not identically zero but comes from a sliver of phase space: WimPyDD gives an in-ROI rate of 4.3 × 10⁻⁵ /t/yr at unit coupling
there, 1.0 × 10⁻⁷ of the rate at δ = 300 keV (§5), i.e. zero for any practical purpose.

### 3.5 Figure 1

`figures/fig1_kinematic_map.png`: allowed region in the (m_χ, δ) plane for E_R = 248 keV on 16 June (light blue), the band swept by
E_R = 216–280 keV (darker), the annual-average and December boundaries (dashed/dotted), v_esc = 500/600 variations (dash-dot), the absolute
ceiling μ v_max²/2 (grey) and the 24 LZ grid points (green = allowed; red cross = the Table S7 dash). Vertical dashed line: elastic floor 73 GeV.

## 4. Part 2 — WimPyDD inelastic spectra on 16 June (files `spectra_summary.csv`, `band_fraction.csv`, `spectra_cache.npz`, `figures/fig2_spectra_1000GeV.png`)

### 4.1 Set-up and a technical finding

Spectra were computed on a 1 keV grid from 1 to 420 keV for m_χ = 400, 1000, 4000 GeV and δ = 0, 25, …, 400 keV plus 310, 320, 330, 340,
360, 370, 380, 390 keV (June halo, day 167); for δ = 200–400 keV with the December halo (day 336); for 1000 GeV with the annual-average halo;
and for m_χ = 100 GeV (δ = 0, 100, 200) and 200 GeV (δ = 0, 100, 200, 300) for the degeneracy study. 4.7 s per spectrum; 126 spectra, cached.

WimPyDD's `streamed_halo_function` builds its default v_min grid from 0 to v_esc + |v_sun| = 794.6 km/s even when `day_of_the_year` is set,
so for the June halo the tail 795–810 km/s (η(795 km/s) = 1.2 × 10⁻⁷ s/km, 4 % of η(750)) is silently dropped. With the default grid the rate
at (1000 GeV, δ = 380 keV, 248 keV) came out negative (−1.1 × 10⁻⁵ /t/yr/keV) and the δ = 350 rate was 11 % low. All spectra here therefore use
an explicit grid `vmin = linspace(0, 830, 2000)` passed through `lz.wd_halo(..., vmin=...)`; the June halo function is then non-zero up to
810.5 km/s, the December one up to 781.8 km/s and the average one up to 795.1 km/s. Residual negative values (< 10⁻¹² relative) are clipped to 0.
This matters for any corpus paper evaluating inelastic rates within ~30 keV of δ_max.

### 4.2 Reproduction of Fig. 1 (top panel, 1000 GeV) — shape

| δ [keV] | onset E₋ (June / avg) [keV] | Eq. 4 E₋ (June) | first dip [keV] | second dip [keV] | peak [keV] | peak rate June / avg [/t/yr/keV] |
|---|---|---|---|---|---|---|
| 0 | 1 / 1 (grid edge) | 0 | 102 | 267 | < 5 | 1.10e7 / 1.14e7 (7.0e6 at 10 keV, avg) |
| 200 | 30 / 32 | 30.9 | 102 | 266 | 53 | 2.53e3 / 1.90e3 |
| 300 | 85 / 92 | 90.4 | — (onset above it) | 266 | 166 | 24.0 / 15.7 |

Fig. 1 (read by eye from `inputs/figures_png/Fig1_combined_recoils.png`): onset of the δ = 300 keV curve just below 100 keV, δ = 200 curve at
≈ 30 keV; dips at ≈ 100 and ≈ 265 keV for all three; the δ = 300 curve peaks near 170–180 keV at ≈ 60–80 /t/yr/keV, the δ = 200 curve at
≈ 5–8 × 10³ near 55 keV, and the δ = 0 curve is ≈ 2–3 × 10⁷ at 10 keV. Shapes agree (dip positions to ±2 keV, onsets to a few keV, peak positions
within ~10 keV — the figure presumably uses the time-averaged halo, whose onset is 92 keV). The absolute level of our curves is ×3–5 below the
figure at every δ, consistent with the coupling-convention factor flagged in the guide; P003 settles it. Only shapes and ratios are used below.

The dips are the zeros of the isoscalar M-response form factor of xenon (first node at q ≈ 1.0 fm⁻¹, E_R ≈ 100 keV, second at ≈ 266 keV).
Their position does not depend on m_χ or δ. **The 248 keV event lies 18 keV below the second node, on the steeply falling edge of the main
lobe, for every inelastic O₁ spectrum.** Note that the dossier's phrase "exactly where … inelastic DM predict their spectra to peak" is not
right for O₁: the peak of the in-ROI spectrum is at 157–205 keV for δ = 250–360 keV (1000 GeV) and reaches 248 keV only at δ ≥ 385 keV.

### 4.3 Where the spectrum sits relative to 248 keV (1000 GeV, June)

Unweighted in-ROI (5.4–270 keV) quantiles, peak, fractions (`spectra_summary.csv`; f_band and percentiles from `band_fraction.csv` use the
efficiency of §5):

| δ [keV] | onset | peak (ROI) | q16 – q50 – q84 [keV] | CDF at 248 | f(5.4–55)/ROI | f(200–270)/ROI | f(216–280), eff | f(>270)/total |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 6 | 8.9 – 17.2 – 34.0 | 0.99999 | 0.968 | 3.6e-4 | — | 1.2e-4 |
| 100 | 7 | 26 | 19.9 – 32.2 – 50.7 | 0.99997 | 0.882 | 1.7e-3 | — | 7.9e-4 |
| 200 | 30 | 53 | 48.1 – 65.3 – 157 | 0.9994 | 0.303 | 0.032 | — | 0.017 |
| 250 | 52 | 157 | 102 – 156 – 191 | 0.9977 | 2.8e-4 | 0.107 | 0.043 | 0.064 |
| 300 | 85 | 166 | 145 – 172 – 202 | 0.9957 | 0 | 0.179 | 0.075 | 0.129 |
| 325 | 108 | 176 | 156 – 181 – 210 | 0.9933 | 0 | 0.253 | 0.111 | 0.210 |
| 350 | 139 | 195 | 177 – 198 – 222 | 0.9841 | 0 | 0.471 | 0.230 | 0.446 |
| 375 | 183 | 221 | 208 – 224 – 242 | 0.9068 | 0 | 0.939 | 0.681 | 0.915 |
| 380 | 194 | 226 | 214 – 229 – 247 | 0.8461 | 0 | 0.992 | 0.767 | 0.965 |
| 390 | 221 | 270 | 236 – 264 – 269 | 0.3076 | 0 | 1.000 | 0.719 | 0.996 |
| 400 | 259 | 270 | 266 – 268 – 269 | 0.0000 | 0 | 1.000 | 0.519 | 0.9996 |

Ratio of the 200–270 keV rate to the 5.4–55 keV rate (the 2024 low-energy ROI): 3.7 × 10⁻⁴ (δ = 0), 2.0 × 10⁻³ (100), 0.10 (200), 0.82 (225),
385 (250), and formally infinite for δ ≥ 275 keV, where the spectrum onset is above 55 keV (E₋ = 67 keV at δ = 275). So δ ≳ 240 keV is what
removes the low-energy population that the 2024 search would otherwise have seen; this is the kinematic reason Table S7 jumps from
0.8σ (δ = 100) to 2.2σ (150) to ≈ 3σ (≥ 250).

**248 keV lies inside the central 68 % (q16–q84) of the in-ROI spectrum only for δ = 390 keV at 1000 GeV and δ = 400 keV at 4000 GeV, and for no
δ at 400 GeV** (`results.json: delta_range_248_in_central68_June`). At those δ the spectrum is a 20–50 keV sliver pinned between its kinematic
onset and the 266 keV node, and the rate is 3 × 10⁻³ (1000 GeV, δ = 390) and 10⁻⁴ (4000 GeV, δ = 400) of the δ = 300 keV rate (§5).
Efficiency-weighted percentile of 248 keV: 99.5 % (δ = 300), 98.1 % (350), 86 % (375), 74 % (380), 14 % (390) for 1000 GeV; 99.6 %, 98.7 % (δ = 300, 325)
for 400 GeV; 99.5 %, 98.7 %, 95 %, 67 % (δ = 300, 350, 380, 400) for 4000 GeV. The fraction of the efficiency-weighted spectrum inside the event's
±1σ band (216–280 keV) first exceeds 16 % at δ = 325 / 340 / 350 keV and 50 % at δ = 350 / 370 / 390 keV for 400 / 1000 / 4000 GeV.

The fraction of the *total* spectrum lost above the 270 keV efficiency edge is 13 % at δ = 300, 45 % at δ = 350, 97 % at δ = 380 (1000 GeV) and
7 %, 75 % at δ = 300, 350 (400 GeV); for δ ≥ 370 keV (1000 GeV) the global maximum of dR/dE is in the second lobe at ≈ 343 keV, outside the ROI.

### 4.4 400 and 4000 GeV

400 GeV: δ = 300 → onset 103, peak 169, q16–q84 = 150–204 keV; δ = 325 → 142, 195, 177–219; δ = 340 → 181, 214, 200–230; δ = 350 → sliver 240–270
keV (¹³⁶Xe only). 4000 GeV: δ = 300 → 79, 167, 144–202; δ = 350 → 124, 187, 166–217; δ = 380 → 162, 212, 195–234; δ = 400 → 197, 234, 219–252.

### 4.5 Figure 2

Left: June spectra for 1000 GeV, δ = 0…390 keV, with the ROI edges (grey), the event's ±1σ band (red) and 248 keV (red line). Right: the same
spectra normalised to unit area in 5.4–270 keV — the peak marches from 157 to 226 keV as δ goes from 250 to 380 keV and only the δ ≥ 390 curves
have their bulk at 248 keV, hard against the node.

## 5. Part 3 — total in-ROI rate versus δ (files `rate_vs_delta.csv`, `figures/fig3_rate_vs_delta.png`)

Efficiency model ε(E): piecewise-linear through (5.4 keV, 0.50), (14, 0.96), (250, 0.96), (269.9, 0.50), (289.8, 0); anchors from the LZ text
(50 % at 5.4 and 269.9 keV, 96 % average 14–250 keV; Fig. S2). R_eff(δ) = ∫ ε dR/dE. A sharp-cut variant 0.96 × ∫₅.₄²⁷⁰ dR/dE is in the CSV.
Rates at WimPyDD unit coupling (normalisation pending P003; ratios are convention-free):

| m_χ [GeV] | R(0) | R(200) | R(300) | R(325)/R(300) | R(350)/R(300) | R(380)/R(300) | R(400)/R(300) | δ at 10 % of R(300) | δ at 1 % |
|---|---|---|---|---|---|---|---|---|---|
| 400 | 3.3e8 | 1.5e5 | 687 | 0.044 | 1.0e-7 | 0 | 0 | 320 | 331 |
| 1000 | 1.4e8 | 1.1e5 | 1696 | 0.263 | 0.032 | 1.3e-4 | 2.5e-7 | 338 | 359 |
| 4000 | 3.5e7 | 3.9e4 | 794 | 0.344 | 0.078 | 4.2e-3 | 1.1e-4 | 346 | 373 |

R(300)/R(0) = 2.1 × 10⁻⁶, 1.2 × 10⁻⁵, 2.3 × 10⁻⁵ and R(300)/R(200) = 4.6 × 10⁻³, 0.015, 0.021 (400, 1000, 4000 GeV). Between δ = 300 and 350 keV
the rate falls by ×30 at 1000 GeV and ×13 at 4000 GeV; at 400 GeV it falls by ×10⁷ because δ_max is 341 keV. Reaching the δ ≥ 385 keV region
where the spectrum peaks at 248 keV costs a further ×10³–10⁴ relative to δ = 300 keV. Coupling-wise (rate ∝ c²) this is ×30–100 in c₁⁰ — the
lower ends of the Fig. 6 two-sided intervals would have to rise accordingly if LZ extended its δ grid.

Annual modulation of the in-ROI rate (June / December): 1.65, 1.76, 2.41, 27.7 at δ = 200, 250, 300, 350 keV (1000 GeV); 1.83, 2.08, 8.77, ∞
(400 GeV; December cannot reach δ = 350); 1.58, 1.66, 1.99, 5.63 (4000 GeV). June / annual average (1000 GeV): 1.30 (δ = 200), 1.55 (300),
4.2 (350), 38 (375). The dossier's 2.5× at (1000 GeV, δ = 300) is confirmed (2.41 with the efficiency-weighted ROI).

Figure 3: left, R_eff(δ) for June (solid) and December (dashed) with δ_max(248 keV) marked; right, R(δ)/R(300) on a log scale.

## 6. Part 4 — mass degeneracy (files `degeneracy.csv`, `figures/fig4_mass_degeneracy.png`)

Shape metric: Kolmogorov–Smirnov distance D = max |CDF₁ − CDF₂| between spectra normalised to unit area in 5.4–270 keV (June halo).

| δ [keV] | D(400, 1000) | D(1000, 4000) | D(400, 4000) | D(200, 1000) | D(100, 1000) | peak shift 400 vs 4000 [keV] |
|---|---|---|---|---|---|---|
| 0 | 0.007 | 0.003 | 0.011 | 0.022 | 0.060 | 0 |
| 100 | 0.009 | 0.004 | 0.013 | 0.027 | 0.074 | −1 |
| 200 | 0.027 | 0.011 | 0.038 | 0.100 | 0.545 | +2 |
| 250 | 0.059 | 0.025 | 0.084 | — | — | −2 |
| 300 | 0.059 | 0.011 | 0.069 | (200 GeV forbidden) | — | +2 |
| 325 | 0.280 | 0.048 | 0.322 | | | +20 |
| 350 | 0.976 | 0.147 | 0.978 | | | +83 |

For δ ≤ 300 keV the 400–4000 GeV spectra are degenerate at the D ≲ 0.07 level (peak positions within 3 keV), confirming the paper's statement
that masses ≥ 400 GeV are "nearly degenerate" (it made the statement for the elastic Lagrangians and kept only 1000 GeV). The degeneracy is
broken within ~40 keV of δ_max(400 GeV): D = 0.28 at δ = 325 and 0.98 at δ = 350 keV, where the 400 GeV spectrum has been squeezed to a 240–270 keV
sliver. For comparison, 200 GeV differs from 1000 GeV by D = 0.10 already at δ = 200 keV and 100 GeV by D = 0.54 (its δ = 200 spectrum is confined
to 67–90 keV). The LZ grid's use of three masses is therefore adequate everywhere except within ~40 keV of each mass's δ_max, i.e. exactly the
region where Table S7's significance is still rising.

## 7. Validation and robustness

- Eq. 4 endpoints satisfy v_min(E_±) = v (checked in P000); WimPyDD onsets agree with Eq. 4 E₋ to within the 1 keV grid plus the 1.4 km/s
  difference in v_E (30 vs 30.9 keV at δ = 200; 85 vs 90.4 at δ = 300 — the WimPyDD onset is lower because ¹³⁴Xe/¹³⁶Xe extend E₋ downward and its v_max is 810.5 km/s).
- Elastic WimPyDD O₁ spectrum equals the Helm SI rate of `lzcommon` at 10–50 keV (P000 check) and shows the expected node positions.
- The default-grid truncation (§4.1) changes R(δ) by < 0.3 % at δ = 300, 11 % at δ = 350 and gives unphysical (negative) values at δ ≥ 380; all
  results use the extended grid.
- Halo variations: δ_max(248) moves by −12 / −24 keV from June to average / December, and by −36 / +46 keV for v_esc = 500 / 600 km/s.
  The kinematic conclusions (dash at (400, 350); floor 73 GeV; 248 keV below the node for δ ≤ 385 keV) are unchanged by any of these.
- The node positions come from the shell-model M response in WimPyDD; the Helm form factor puts the second zero within a few keV of the same place.

## 8. Failed or abandoned approaches

- First background run stalled (sandbox process suspension) and a `pkill` to remove it failed ("Cannot get process list"), leaving two processes
  racing on the cache file; both were stopped with TaskStop and the script re-run in the foreground from the cache. No numerical consequences
  (each cache entry is a full spectrum written atomically by `np.savez`; the final run recomputed anything missing).
- The default WimPyDD v_min grid was abandoned in favour of an explicit grid (§4.1).

## 9. Discussion

The kinematics alone fix the stage: the event needs m_χ ≥ 73 GeV (elastic) and, for inelastic O₁-like scattering, δ ≤ 341 / 387 / 409 keV at
400 / 1000 / 4000 GeV on 16 June; every LZ grid point except (400 GeV, 350 keV) is allowed, and the (400, 350) dash is robust to the energy
uncertainty and to the halo phase. What kinematics do *not* say is that inelastic spectra "peak at" 248 keV. The O₁ spectrum has a nuclear form-factor
node at 266 keV; for δ ≤ 350 keV the in-ROI spectrum peaks at 160–200 keV and puts only 0.4–2 % of its events above 248 keV. Making 248 keV a typical
recoil requires δ within ~15 keV of δ_max (δ ≥ 385 keV at 1000 GeV), a region LZ did not scan and where the rate is 10³–10⁴ times smaller than at
δ = 300 keV (a ×30–100 larger coupling) and 4–40 times larger in June than in December. The monotonic rise of Table S7 with δ is driven by the
disappearance of the low-energy population (the 200–270 / 5.4–55 keV ratio goes from 10⁻⁴ to ∞ between δ = 0 and 275 keV), not by the peak
approaching the event. The natural next steps are: (i) an extended-likelihood scan out to δ_max in true-energy space with resolution smearing
(the ±23 keV resolution partly fills the node — a RESP/STAT paper), (ii) the same map for O₄ (different form factor, no M-node) and for the
elastic Lagrangians whose q²-suppressed spectra do peak near 200 keV (P003), (iii) the absolute-coupling normalisation (P003).

## 10. References

- LZ Collaboration, arXiv:2609.02823 (2026) — Theory paragraph, Fig. 1 and caption, Fig. S2 caption, LEE section (grid), Tables S6–S7.
- D. Tucker-Smith and N. Weiner, Phys. Rev. D 64, 043502 (2001) [hep-ph/0101138].
- J. Bramante, P. J. Fox, G. D. Kribs, A. Martin, Phys. Rev. D 94, 115026 (2016) [1608.02662].
- D. Baxter et al., Eur. Phys. J. C 81, 907 (2021) [2105.00599] — SHM parameters.
- I. Jeong, S. Kang, S. Scopel, G. Tomar, Comput. Phys. Commun. 276, 108342 (2022) — WimPyDD.
- N. Anand, A. L. Fitzpatrick, W. C. Haxton, Phys. Rev. C 89, 065501 (2014) — NREFT nuclear responses, coupling convention c = 1/m_v².
- Corpus: `output/00_evidence_dossier.md` (P000 numbers), P003 (normalisation; pending).

## 11. Tools and provenance (mirrors `output/provenance/P002.json`)

- Agent tools: Read ×10 (PAPER_GUIDE.md; 00_evidence_dossier.md; lzcommon.py; fulltext.tex lines 40–75, 452–499, 800–819, 876–912; Fig1 PNG;
  own figures 1–4), Bash ×21 (greps of the paper, environment and WimPyDD source; WimPyDD probes; script runs; table prints; wc -w), Write ×4, Edit ×1,
  ToolSearch ×2, Monitor ×1, TaskStop ×3.
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (via lzcommon); matplotlib 3.11.2; pandas 3.0.5 (table printing only); WimPyDD 2.0.4
  (`streamed_halo_function`, `eft_hamiltonian`, `diff_rate`, target `Xe`); `output/code/common/lzcommon.py` (`v_earth_kms`, `vmax_kms`, `vmin_kms`,
  `E_R_range_keV`, `delta_max_kev`, `m_chi_min_gev`, `wd_halo`, `wd_hamiltonian`, `wd_rate`, constants `LZ`, `OSIG`, `LSIG`, `XE_ISOTOPES`, `M_V_GEV`).
- Recalled knowledge: Baxter-2021 SHM parameters (certain, via lzcommon); inelastic kinematics Eqs. 1–4 (certain, derived); xenon isotopic
  abundances (certain, via lzcommon); the M-response form-factor node of Xe near 100 and 265 keV as a form-factor zero (likely; confirmed numerically here).
- Datasets: none. Data requests: none. WimPyDD-generated files: none (halo functions computed in memory; `find WimPyDD -newer` returned nothing).
