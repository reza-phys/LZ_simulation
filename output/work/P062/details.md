# P062 — Mass–splitting degeneracy: research record

Simulated date 2026-09-12. Category IDM, hep-ph. Author profile: direct-detection parameter-estimation theorists.
Script: `output/code/P062_mass_splitting_degeneracy.py` (run from the simulation root with `.venv/bin/python`; first pass 644 s
incl. 312 s of WimPyDD kernel generation, cached in `kernels_xe_o1_1000gev.npz`; reruns 66 s). Logs: `run_log_first_pass.txt`
(Parts 1–4 of the first pass, which crashed in Part 5 on a root-finder bracket, see §11) and `run_log.txt` (final run).

## 1. Motivation and framework

LZ's look-elsewhere section states: "for all of the Lagrangians, masses of 400 GeV/c² and above have nearly degenerate recoil
spectra, so only one mass (1000 GeV/c²) is kept" (fulltext.tex l. 497); Table S7 gives the same local significance for 400/1000/4000 GeV
at every physical δ. P002 quantified this with KS(400 vs 4000) = 0.07 at δ = 300 keV and 0.98 at 350 keV, P021 found the likelihood
maxima at δ = 340/380/400 keV for 400/1000/4000 GeV, P046 found that its tungsten/xenon δ-meter reads −21 keV (4 TeV) / +38 keV
(400 GeV) if the mass is wrong, P050 found σ(δ) = 19–27 keV/√N at fixed mass. We ask what combination of (m_χ, δ) a xenon spectrum
actually fixes, and what could break the residual degeneracy.

Kinematics (all from `lzcommon` conventions; m_N = 122.322 GeV is the abundance-weighted xenon nuclear mass, μ = m_χ m_N/(m_χ+m_N)):

- v_min(E) = (m_N E/μ + δ)/√(2 m_N E)  ⟹  v_min(E) = √(E/2m_N)·(1 + m_N/m_χ) + δ/√(2 m_N E).  The mass enters only through
  (1 + m_N/m_χ) = 1.12, 1.06, 1.03, 1.01 at 1, 2, 4, 10 TeV; δ enters the second term directly.
- E* = argmin v_min = μδ/m_N;  v_min* = √(2δ/μ).
- Recoil window at speed v: E± = (μ²v²/m_N)[1 − x ∓ √(1 − 2x)], x = δ/(μv²).  Hence √(E₊E₋) = μδ/m_N = E* and
  (E₊ + E₋)/2 = (μ²v² − μδ)/m_N: matching both edges of two spectra requires the same μ, so the degeneracy is only approximate.
- Ceiling δ_max = μ v_max²/2 (no recoil at any energy above it): 329/383/417/425 keV at 400/1000/4000/10000 GeV (Sun-frame,
  v_max = 795.4 km/s); 342/398/434/441 keV on 16 June (810.9 km/s).

## 2. Method

**Exact factorisation (P046 method).** For the velocity-independent operator O₁, WimPyDD's rate is
dR/dE = Σ_i K_i(E) η(v_min,i(E; m, δ)), with η(v) = Σ_{v_j > v} Δη_j. The nine per-isotope kernels K_i(E) were generated once at
1000 GeV on a 1 keV grid 1–1300 keV from a single 3000 km/s stream with Δη = 1 (km/s)⁻¹ (11,700 `diff_rate` calls, 312 s), unit
LZ/Anand coupling c₁ˢ = 1/m_v² (WimPyDD c⁰ = 2/m_v², P003). Checks: Σ_i K_i equals the full-target call to 1.000 (1.004 at 250 keV, a
form-factor node where K ≈ 0); K(m)/K(1000) × m/1000 = 1.000000 at 200/400/4000/10000 GeV and six energies (the only m_χ dependence
of the O₁ kernel is the ρ/m_χ number density), so K_i(E; m) = K_i(E; 1000)·(1000/m). Factorised spectra agree with direct
`lz.wd_rate` calls to 1.000 (1.010 at two form-factor-node energies) at 13 test points (Sun-frame, June, annual halos; 400–4000 GeV;
δ = 300–366 keV).

**Halos.** `lz.wd_halo(vmin=grid)` with the Baxter-2021 SHM, explicit 1200-point v_min grid to 844 km/s: Sun-frame (day = None; v_E =
250.6 km/s, v_max = 795.4 km/s, labelled as such per P035), 16 June (day 167; 810.9), 16 December (day 350; 782.8), a 12-day annual
mean (v_max 810.9) and 24 days at 15.2-day spacing for the modulation.

**Observed-energy pdf.** p(E_obs) ∝ [dR/dE × ε] ⊗ Gauss(σ_E), σ_E = 11√(E/248) keV (P009/P021; constant 11 keV as a variant),
1 keV bins. Windows: "extended" = ideal efficiency, 5 ≤ E_obs ≤ 1000 keV; "LZ-like" = 0.96 plateau with erf edges at 5.4/269.9 keV
(σ 2.5/11.5 keV, P021/P046), no upper cut in E_obs.

**Statistics.** KL(p‖q) = Σ p ln(p/q) per event. For two fixed hypotheses the number of events for a 3σ separation is
N₃ = [3(√V_p + √V_q)/(KL_pq + KL_qp)]² with V the variance of ln(p/q) under each hypothesis (mean log-likelihood-ratio separated by 3 sd
in both directions). Fisher information per event in θ = (ln m, δ [keV]): I_ij = Σ_bins ∂_i p ∂_j p / p (central differences, h = 0.02 in
ln m, 1 keV in δ, bins with p > 10⁻¹²); the rate term for a fixed-coupling model is I^rate_ij = ∂_i ln R ∂_j ln R (Poisson with
expectation R ∝ N). Reported: σ(δ) with m fixed = I_δδ^{-1/2}, with m free = (I⁻¹)_δδ^{1/2}, likewise for ln m, correlation, the ridge slope
−I_{lnm,δ}/I_δδ (δ shift per unit ln m that keeps the likelihood maximal), eigenvalues/vectors.

## 3. Part 1 — grid, pairwise KL, the ridge ("banana")

Grid: m = 200, 300, 400, 700, 1000, 2000, 4000, 10000 GeV; δ = 250–400 keV in 10 keV steps; Sun-frame and 16 June
(`grid_spectra_summary.csv`: E*, E±, observed median/16/84 %, fractions above 400/500 keV, unit rates). Pairwise KL among the 98 physical
Sun-frame points in `pairwise_kl_sunframe.npz`.

**Same-δ mass pairs (Sun-frame; `same_delta_mass_pairs_sunframe.csv`).** 400 vs 4000 GeV: KS on the true-energy spectrum in the LZ ROI
(5.4–270 keV, P002's statistic) = 0.083/0.058/0.032/0.030/0.060/0.129/0.244/0.39 at δ = 250/260/270/280/290/300/310/320 keV; observed
extended-window KS = 0.072 (250), 0.114 (300); N₃(extended window) = 252/240/260/292/219/130/68/36 events. (400 GeV, ≥ 330 keV) is
unphysical in the Sun frame (ceiling 329 keV), which is LZ's (400 GeV, 350 keV) dash. 1000 vs 4000 GeV: KS_ROI = 0.026/0.017/0.222 at
250/300/350 keV, N₃ = 2477/3185/192. P002's KS = 0.07 at 300 keV (16 June halo) compares with our 0.129 (Sun-frame; the June ceiling for
400 GeV is 13 keV higher, which softens the 400 GeV spectrum less); the trend (small below 300 keV, rising steeply above) is the same.

**KL maps and ridge (`kl_maps.npz`, `banana_ref{300,366}_{sunframe,june16,annual}.csv`).** For 41 masses (200–10000 GeV) and δ = 150–400
keV (2 keV) we computed KL(p_ref ‖ p(m, δ)) for the reference spectra (1000 GeV, 366 keV) [P007/P021 Higgsino point] and (1000 GeV, 300
keV) [LZ grid], and for each m the δ_b(m) minimising it (parabolic refinement). Table (δ_b in keV; KL_min in nats/event; E*-rule
δ_ref μ_ref/μ(m); onset rule solves E₋(v_max; m, δ) = E₋(ref); "ceil" = kinematic ceiling):

| ref | halo | m [GeV] | 300 | 400 | 700 | 1000 | 2000 | 4000 | 10000 |
|---|---|---|---|---|---|---|---|---|---|
| 366 | Sun-frame | δ_b | 239.5 (ceil 305, KL 3.8) | 295.0 (ceil 329, KL 2.7) | 355.8 (KL 0.091) | 366.0 | 375.6 (0.037) | 380.3 (0.063) | 383.0 (0.079) |
| 366 | Sun-frame | E*-rule | 459 | 426 | 383 | 366 | 346 | 336 | 330 |
| 366 | Sun-frame | onset rule | 305 (=ceil) | 327 | 355 | 366 | 379 | 385.5 | 389 |
| 366 | June | δ_b | 250.7 (KL 2.9) | 319.5 (KL 1.7) | 358.7 (0.033) | 366 | 374.0 (0.015) | 378.0 (0.029) | 380.3 (0.039) |
| 366 | annual | δ_b | 244.7 (KL 3.3) | 320.7 (KL 1.9) | 359.2 (0.039) | 366 | 373.2 (0.019) | 376.8 (0.035) | 379.0 (0.046) |
| 300 | Sun-frame | δ_b | 249.9 (KL 0.42) | 282.7 (KL 0.13) | 298.6 (0.004) | 300 | 299.7 (0.003) | 299.2 (0.006) | 298.7 (0.008) |
| 300 | Sun-frame | E*-rule | 376 | 349 | 314 | 300 | 284 | 275 | 271 |
| 300 | Sun-frame | onset rule | 272 | 282 | 295 | 300 | 306 | 309 | 311 |
| 300 | annual | δ_b | 256.3 (0.36) | 286.4 (0.10) | 299.3 (0.003) | 300 | 299.4 (0.003) | 299.0 (0.006) | 298.6 (0.008) |

Findings. (i) The ridge is nearly flat in δ above ~700 GeV: the spectrum measures δ almost independently of m_χ. For the 300 keV
reference δ_b stays within 1.3 keV of 300 keV from 700 GeV to 10 TeV; for the 366 keV reference it rises by +10/+14/+17 keV at 2/4/10 TeV
(Sun-frame; +7/+11/+13 annual). (ii) The naive rule δμ/m_N = E* = const (δ ∝ 1/μ) has the wrong sign and magnitude (−20/−30/−36 keV
at 2/4/10 TeV); the onset-matched rule E₋(v_max) = const is much closer (+13/+19.5/+23 keV for ref 366; +6/+9/+11 for ref 300) but
overshoots: the ridge lies between "flat δ" and "onset matched", as expected since the observed shape weights the onset, the peak and the
form-factor node at 266 keV (m-independent) together. (iii) Below ~600–700 GeV the 366 keV reference cannot be mimicked: the ceiling
of 400 GeV (329–342 keV) lies below the reference onset requirement, the best mimic sits at the ceiling with KL ≥ 1.7 nats/event, i.e.
distinguishable with ~10 events. So LZ's "degenerate above 400 GeV" is correct for δ ≤ 300 keV but not near the ceiling (P002's 0.98 KS
at 350 keV is the same statement). (iv) Halo choice moves δ_b by ≤ 3.5 keV for m ≥ 700 GeV (Sun-frame vs annual).

Comparison with P046: P046's W/Xe count ratio reads δ with a mass bias of −21 keV (4 TeV) / +38 keV (400 GeV) at 366 keV; the xenon shape
ridge gives the opposite sign and smaller size for heavy masses (+11 keV at 4 TeV, annual) because the two observables weight different
parts of the velocity integral.

## 4. Part 2 — Fisher information (`fisher_table.csv`)

Per event, annual halo, extended window (σ_E = 11√(E/248) keV):

| (m, δ) | σ_δ m fixed | σ_δ m free | σ_lnm δ fixed | σ_lnm δ free | corr | ridge dδ/dlnm | with rate: σ_lnm, σ_δ | ∂lnR/∂lnm, ∂lnR/∂δ |
|---|---|---|---|---|---|---|---|---|
| (400, 300) | 39.4 | 43.5 | 0.91 | 1.01 | −0.42 | +18.3 | 0.84, 37.9 | 4.37, −0.097 |
| (1000, 300) | 79.1 | 79.2 | 5.8 | 5.8 | −0.02 | +0.2 | 5.80, 22.3 | 0.07, −0.047 |
| (4000, 300) | 91.5 | 92.2 | 26.9 | 27.1 | +0.12 | −0.4 | 4.59, 92.2 | −0.79, −0.038 |
| (1000, 350) | 23.3 | 36.1 | 1.80 | 2.79 | −0.76 | +9.9 | 1.88, 35.6 | 1.80, −0.090 |
| (1000, 366) | 20.9 | 36.3 | 1.18 | 2.06 | −0.82 | +14.5 | 1.28, 34.5 | 2.98, −0.104 |
| (2000, 366) | 24.6 | 48.1 | 2.88 | 5.65 | −0.86 | +7.3 | 4.70, 25.5 | 0.30, −0.074 |
| (4000, 366) | 26.9 | 33.7 | 9.3 | 11.7 | −0.60 | +1.7 | 3.42, 21.2 | −0.45, −0.065 |
| (10000, 366) | 27.8 | 52.8 | 11.8 | 22.5 | −0.85 | +2.0 | 2.10, 24.2 | −0.81, −0.060 |
| (1000, 380) | 29.2 | 41.8 | 0.95 | 1.36 | −0.71 | +22.0 | 1.17, 40.8 | 4.67, −0.133 |
| (1100, 366) | 21.7 | 36.9 | 1.39 | 2.37 | −0.81 | +12.6 | 1.62, 36.0 | 2.27, −0.097 |

(units: keV/√N for σ_δ; 1/√N for σ_lnm; keV per unit ln m for the ridge; the E*-rule slope −δ m_N/(m+m_N) would be −39.9 keV per ln m at
(1000, 366).) LZ-like window, same points: σ_δ(m free) = 255/287/235 keV/√N at (1000, 366)/(1000, 350)/(1000, 300) with correlation
−0.997/−0.996/−0.93 (vs 36/36/79 in the extended window): the onset alone cannot break the (ln m, δ) correlation; the peak and high-energy
side in an extended window do. Constant 11 keV resolution changes every entry by < 5 %. Sun-frame halo: σ_δ(m fixed) = 19.6 keV at
(1000, 366), 18.0 at (1000, 350); P050's 19–27 keV/√N at fixed m is reproduced (20.9/23.3/29.2 at 366/350/380 keV, annual).

Eigen-directions: the small-eigenvalue direction is essentially pure ln m at every point (eigenvector slope dδ/dlnm ≈ 0.1–1 keV per unit ln m
when δ is measured in keV), i.e. the degenerate parameter is the mass at (nearly) fixed δ; the well-measured combination is δ. The mass
information scales as I_lnm,lnm ∝ (m_N/m)²: σ_lnm(δ fixed) = 1.18 → 2.9 → 9.3 → 11.8 from 1 → 2 → 4 → 10 TeV at 366 keV.

Because ∂p/∂ln m saturates (∝ m_N/m), the local Fisher over-states separations over finite mass intervals; we therefore quote exact
two-hypothesis numbers (`separation_along_banana.csv`, annual halo, shape only, both points on the ridge):

| ref δ | pair (GeV) | δ_b pair (keV) | KL_12 / KL_21 | KS | N₃ extended | N₃ LZ-like |
|---|---|---|---|---|---|---|
| 366 | 400 vs 4000 | 320.7 vs 376.8 | 1.72 / 2.31 | 0.71 | 11 | 47 |
| 366 | 700 vs 4000 | 359.2 vs 376.8 | 0.116 / 0.204 | 0.18 | 223 | 750 |
| 366 | 1000 vs 4000 | 366.0 vs 376.8 | 0.035 / 0.070 | 0.089 | 642 | 1544 |
| 366 | 1000 vs 2000 | 366.0 vs 373.2 | 0.019 / 0.028 | 0.066 | 1065 | 3398 |
| 366 | 2000 vs 4000 | 373.2 vs 376.8 | 0.0038 / 0.0050 | 0.023 | 4370 | 14281 |
| 366 | 1000 vs 10000 | 366.0 vs 379.0 | 0.046 / 0.107 | 0.10 | 508 | 1140 |
| 300 | 400 vs 4000 | 286.4 vs 299.0 | 0.075 / 0.130 | 0.12 | 245 | 5927 |
| 300 | 1000 vs 4000 | 299.9 vs 299.0 | 0.0055 / 0.0081 | 0.025 | 3170 | 11580 |
| 300 | 2000 vs 4000 | 299.4 vs 299.0 | 0.0006 / 0.0007 | 0.007 | 28961 | 95811 |

## 5. Part 3 — annual modulation along the ridge (`modulation_banana.csv`, `modulation_separation.csv`)

Rates on 24 days of the year in the extended window; cosine fit R = R₀[1 + a₁ cos ω(t − t₀)] (a₁ > 1 means the rate vanishes for part of
the year and the time pdf is non-cosine, cf. P034); June-16/Dec-16 ratio; May–August fraction of events. Phase t₀ = day 152 everywhere.

| ref | m [GeV] | δ_b | a₁ | June/Dec | May–Aug frac |
|---|---|---|---|---|---|
| 366 | 700 | 359.2 | 1.45 | 272 | 0.72 |
| 366 | 1000 | 366.0 | 1.20 | 27.9 | 0.64 |
| 366 | 2000 | 373.2 | 0.90 | 7.9 | 0.56 |
| 366 | 4000 | 376.8 | 0.79 | 5.6 | 0.53 |
| 366 | 10000 | 379.0 | 0.74 | 4.8 | 0.52 |
| 300 | 400 | 286.4 | 0.63 | 3.8 | 0.49 |
| 300 | 1000 | 299.9 | 0.42 | 2.3 | 0.43 |
| 300 | 4000 | 299.0 | 0.33 | 1.9 | 0.41 |
| — | (1000, 300) | | 0.42 | 2.32 | 0.43 |
| — | (1000, 350) | | 0.96 | 9.6 | 0.58 |
| — | (4000, 350) | | 0.62 | 3.65 | 0.49 |

The modulation depends on the depth v_max − v_min* with v_min* = √(2δ/μ) = 776 (1 TeV, 366 keV) vs 756 km/s (4 TeV, 376.8 keV): along the
ridge the heavier mass sits deeper in the tail and modulates less, which the energy spectrum barely registers. Time-only 3σ separations
(exact two-pdf N₃): ref 366: 400 vs 4000: 61; 700 vs 4000: 71; 1000 vs 4000: 243; 1000 vs 10000: 194; 2000 vs 4000: 4074. Ref 300:
400 vs 4000: 674; 1000 vs 4000: 7713. Near the ceiling the timing beats the spectrum (243 vs 642 events for 1 vs 4 TeV); adding the
two KL's (energy and time approximately independent) gives N₃ ≈ 180 for 1 vs 4 TeV at the Higgsino point. P006's June/Dec = 2.4 at (1000,
300) and P002's 28 at (1000, 350, LZ ROI) bracket our 2.32 and 9.6 (extended window; the LZ ROI keeps only the summer-dominated onset of the
350 keV spectrum). The a₁ = 0.42 at (1000, 300) equals P034's 0.428 at δ = 300.

## 6. Part 4 — the endpoint in xenon (`endpoint_xe.csv`)

| halo | (m, δ) | E₋ | E* | E₊ | frac > 400 keV (obs) | frac > 500 keV (obs) | last 50 keV below E₊ (true) | E₉₉ (true) |
|---|---|---|---|---|---|---|---|---|
| Sun | (1000, 300) | 97 | 267 | 733 | 2.7 % | 0.10 % | 3×10⁻⁵ | 431 |
| Sun | (1000, 350) | 170 | 312 | 572 | 9.4 % | 0.07 % | 1.5×10⁻⁴ | 451 |
| Sun | (1000, 366) | 212 | 326 | 501 | 10.9 % | 0.02 % | 0.7 % | 445 |
| Sun | (1000, 380) | 283 | 339 | 406 | 8.0 % | 0.002 % | 38 % | 432 |
| Sun | (4000, 366) | 171 | 355 | 738 | 16.9 % | 0.93 % | 4×10⁻⁴ | 490 |
| Sun | (4000, 380.3) ridge | 200 | 369 | 681 | 21.3 % | 0.81 % | 1.2×10⁻³ | 486 |
| Sun | (10000, 383) ridge | 198 | 378 | 724 | 22.8 % | 1.3 % | 9×10⁻⁴ | 515 |
| June | (1000, 366) | 182 | 326 | 585 | 12.7 % | 0.12 % | 3×10⁻⁴ | 459 |
| June | (4000, 378) ridge | 173 | 367 | 776 | 19.0 % | 1.4 % | 3×10⁻⁴ | 567 |

The kinematic endpoint moves from 501 to 681–738 keV between 1 and 4 TeV at fixed shape, but the rate near it is suppressed by the halo
tail (η → 0 at v_max) and the Helm/shell-model form factor: ≤ 1.4 % of events lie above 500 keV and ≤ 0.1 % within 50 keV of E₊ (except
at the 380 keV ceiling, where the whole window is 120 keV wide). With ≈ 6 extended-window events per 2.84 t·yr at the LZ best fit (§7),
the endpoint region receives ~0.05 events per 2.84 t·yr even at 4 TeV. This confirms P046's tungsten conclusion (0.39 % above 500 keV)
for xenon; P015's 786 keV endpoint at 300 keV is E₊(June) = 795 keV here (Sun-frame 733). What does carry the residual mass information
is the 350–450 keV high-side slope: the fraction above 400 keV rises from 10.9 % (1 TeV) to 21 % (4 TeV) along the ridge, and E₉₉ from 445
to 486 keV.

## 7. Part 5 — fixed-coupling rate normalisation and interpretation (`fixed_coupling_delta_N1.csv`, `rate_along_banana.csv`, `interpretation_higgsino.csv`)

LZ-window counts N_LZ(m, δ) = 2.84 t·yr × ∫ ε_LZ dR/dE (annual halo) with the isoscalar O₁ shape, normalised so that N_LZ(1000 GeV, 366 keV)
= 1 (P007's Higgsino δ(N=1) = 366 keV). Unit-coupling N_LZ(1000, 366) = 21.66, so the anchor is (c m_v²)² = 0.0462 versus P007's
(c_eq m_v²)² = 0.0777: a factor 1.7 that reflects P007's isovector-dominated Z-coupling normalisation ("vector factor 3.214", σ_SI,eq
convention) and halo/efficiency choices; only the anchored relative counts are used here. Validation of the relative normalisation:
δ(N=1) = 311.7/343.1/366.0/375.2/377.1 keV at 300/500/1000/2000/4000 GeV versus P007's 310/342/366/375/376 (≤ 2 keV); multiplets with
(2Y)² = 4/9/16 at 1 TeV: 372.4/375.6/377.7 keV versus P037's 374/379/383 (P037 used P007's grid; the 1–5 keV offsets are within the halo
conventions of P037's caveats). At 300 GeV the (2Y)² = 16 multiplet gives 1.3 LZ events even at the kinematic ceiling → no solution
(excluded at that mass); (2Y)² = 9 requires δ = 316.8 keV, 0.6 keV below the ceiling. δ(N=3.65)/δ(N=0.105) (90 % one-event band):
358.6/375.8 keV at 1 TeV, 366.7/390.6 at 4 TeV.

Counts along the shape ridge (annual): ref 366: N_LZ = 0.54/1.01/1.36/1.04/0.52 at 700/1000/2000/4000/10000 GeV (extended window per
2.84 t·yr: 2.6/5.8/8.3/6.4/3.2); ref 300: N_LZ = 568/580/629/517/326/149 at 400/700/1000/2000/4000/10000 GeV. Local slopes
d ln N/d ln m along the ridge: +0.44 (1–2 TeV), +0.02 (1–4 TeV), −0.40 (2–4 TeV), −0.75 (4–10 TeV) for ref 366; −0.28/−0.47/−0.66/−0.85 for
ref 300. The fixed-coupling N = 1 curve (δ = 366/368/375/377/374 keV at 1/1.1/2/4/10 TeV) therefore runs within 4 keV of the shape ridge
(366/373/377/379 keV) between 1 and 10 TeV: for the Higgsino the rate does not lift the mass degeneracy because the 1/m_χ number density
is compensated by the deeper v_min* at heavier mass. Against P018's ×5 (0.7 dex) halo-tail rate uncertainty, a slope |d ln N/d ln m| ≤ 0.45
translates into Δ ln m ≥ 3.6 — no mass information. The local Fisher rate term (σ_lnm 2.06 → 1.28 at (1000, 366)) is an over-statement
for the same nonlinearity reason as in §4.

Interpretation table (Higgsino, annual halo; E± quoted for the June v_max; 248 keV percentile in the LZ-accepted spectrum; a₁ and
May–Aug fraction in the extended window; N₃ = events for 3σ separation from the thermal 1.1 TeV case):

| m [GeV] | δ(N=1) | ceiling (June) | E* | v_min* | E₋–E₊ (June) | pct(248) | frac > 400 | ext events per LZ event | a₁ | June/Dec | May–Aug | N₃ shape / time vs 1.1 TeV |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 (non-thermal) | 311.7 | 317.4 | 221 | 803 | 169–291 | 0.99 | 0 | 1.05 | 1.75 | ∞ (no winter rate) | 0.84 | 8 / 63 |
| 1100 (thermal) | 367.9 | 402.2 | 331 | 776 | 181–604 | 0.91 | 13 % | 6.2 | 1.15 | 21.7 | 0.63 | — |
| 4000 | 377.1 | 433.6 | 366 | 756 | 172–780 | 0.92 | 20 % | 6.3 | 0.79 | 5.6 | 0.53 | 783 / 328 |

(300 GeV: Sun-frame E± undefined because 311.7 keV exceeds the Sun-frame ceiling 305 keV — this mass scatters only around the June
maximum.) Consequences: (a) direct detection alone cannot distinguish a 1.1 TeV from a 4 TeV Higgsino with fewer than several hundred
events (783 shape-only, 328 timing-only, ~230 combined in a 5–1000 keV xenon window), i.e. > 100 t·yr at the LZ rate of ~2 extended-window
events per t·yr; (b) the 300 GeV non-thermal Higgsino is separable with ~8 events because it must sit 6 keV below its June ceiling, where
the spectrum is a narrow 169–291 keV lobe with no winter rate; (c) the mass therefore has to come from elsewhere: P014/P048 find that the LZ
Higgsino (cτ = 0.71 cm) is beyond HL-LHC, marginal at FCC-hh (5σ at 1.1 TeV) and measured to ~3 TeV at a 10 TeV muon collider, while δ
is unmeasurable at colliders; the two programmes are exactly complementary — direct detection fixes δ (to ±20–36 keV/√N at 366 keV, ±80
keV/√N at 300 keV) and the collider fixes m. (d) The relic-density argument (thermal 1.1 TeV) is the only "mass measurement" available
today, and it is a theory prior, not a measurement.

## 8. Figures

- `figures/fig1_banana_kl_maps.png`: log₁₀ KL(ref ‖ p(m, δ)) maps (Sun-frame halo, extended window) for the (1000 GeV, 300 keV) and
  (1000 GeV, 366 keV) references, with contours KL = 0.003–0.3, the KL-minimising ridge δ_b(m), the E*-matched (δ ∝ 1/μ) and onset-matched
  curves, the kinematic ceiling, the Higgsino N_LZ = 1 curve (annual halo) and LZ's grid points. The ridge is flat in δ above ~700 GeV;
  the E* rule bends the wrong way.
- `figures/fig2_fisher_spectra_modulation.png`: (left) 1σ Fisher ellipses for N = 30 events at (1000 GeV, 366 keV), annual halo: shape-only
  extended window, shape + fixed-coupling rate, shape-only LZ-like window (an unbounded band), and the KL ridge; (middle) observed spectra
  along the 366 keV ridge (700–10000 GeV, Sun-frame) — nearly identical peaks with a slowly rising high-energy side; (right) cosine
  modulation amplitude a₁ along both ridges.

## 9. Robustness and validation summary

- Kernel factorisation exact to 10⁻⁴ (1 % at form-factor nodes); 1/m scaling exact to 10⁻⁶.
- Halo variants: ridge shifts ≤ 3.5 keV (m ≥ 700 GeV) between Sun-frame, June and annual; Fisher σ_δ(m fixed) 19.6 vs 20.9 keV.
- Resolution: constant 11 keV vs 11√(E/248) changes Fisher entries by < 5 %.
- Rate normalisation: δ(N=1) reproduces P007 to ≤ 2 keV and P037 to ≤ 5 keV.
- P050 (σ_δ at fixed m 19–27 keV/√N) and P034 (a₁ = 0.428 at δ = 300) reproduced.

## 10. Caveats

O₁ isoscalar shape only (P021: O₁ᵛ differs by ≤ 0.4σ in single-event Z; multiplet shapes equal P007's per P037); O₄ lacks the M-response
node and was not run. Ideal efficiency in the extended window and no background (P050: ≲ 3×10⁻³ events per 2.84 t·yr above 200 keV). The
Fisher analysis is local and asymptotic; the exact two-hypothesis N₃ values are the ones to use for finite mass intervals. Baxter-2021 SHM
family only; P018 shows the tail normalisation shifts δ by ∓11–17 keV and the rate by ×5–30, which is why we do not credit the rate with
any mass information. The Higgsino normalisation is anchored to P007 at one point; absolute couplings are not re-derived.

## 11. Failed or abandoned

- First pass crashed in Part 5: `scipy.optimize.brentq` had no sign change for the (2Y)² = 16 multiplet at 300 GeV because the model gives
  > 1 event even at the kinematic ceiling. Fixed by returning NaN ("excluded at this mass") when N(ceiling) > target; Parts 1–4 were reloaded
  from the saved CSV/npz tables on the rerun (identical grids checked).
- The first pass exceeded the 600 s foreground limit (644 s, 312 s of which were kernel generation); the kernel cache makes reruns 66 s.
- The E* = δμ/m_N = const "banana" proposed in the assignment was tested and rejected as a description of the ridge (wrong sign).

## 12. References

LZ Collaboration, arXiv:2609.02823 (2026). D. Tucker-Smith, N. Weiner, PRD 64, 043502 (2001). N. Anand, A. L. Fitzpatrick, W. C. Haxton,
PRC 89, 065501 (2014). I. Jeong, S. Kang, S. Scopel, G. Tomar, CPC 276, 108342 (2022) (WimPyDD). D. Baxter et al., EPJC 81, 907 (2021).
G. Cowan, K. Cranmer, E. Gross, O. Vitells, EPJC 71, 1554 (2011). Corpus: P002, P006, P007, P009, P014, P015, P018, P021, P034, P035,
P037, P038, P046, P048, P050.

## 13. Tools and provenance

Mirrors `output/provenance/P062.json`. Agent tools: Read ×20 (PAPER_GUIDE, dossier, ledger p.1, P002/P021/P038/P037/P046/P050/P018,
tex l.484–499 and 850–915, lzcommon l.126–220 and 305–400, P046 script l.1–170, run log, result tables, two figures), Bash ×15 (ledger
row extraction, tex/lzcommon/P046 greps, WimPyDD timing/scaling probe, two full runs and one figure rerun, log/table inspection, debugging
probe of the root-finder, word counts), Write ×4, Edit ×5, Skill ×1 (dataviz; JS validator skipped per guide), ToolSearch ×1.
Software: python 3.12.13; WimPyDD 2.0.4 (`eft_hamiltonian`, `streamed_halo_function` via `lz.wd_halo`, `diff_rate` via `lz.wd_rate`
with `isotopes_list`); numpy 2.5.3; scipy 1.18.1 (`special.erf`, `optimize.brentq`, `linalg`); matplotlib 3.11.2; pandas 3.0.5
(table printing only); common/lzcommon.py (LZ constants, `wd_halo`, `wd_hamiltonian`, `wd_rate`, `M_V_GEV`, `VESC_KMS`). Recalled
knowledge: inelastic kinematics E±, E*, v_min* (certain); Fisher/KL asymptotics and the two-hypothesis N₃ formula (certain); thermal
Higgsino mass 1.1 TeV (certain, also in P007); Poisson rate Fisher term (certain). Datasets: none. Data requests: none.
WimPyDD-generated files: none outside `output/work/P062/` (`diff_rate` used; no response-function files written).
