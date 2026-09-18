# P097 — A dark disk or co-rotating substructure: would a slow, co-rotating dark-matter component change anything for the 248 keV event at δ ≈ 300 keV?

Simulated date 2026-09-16 · Category HALO · astro-ph.GA (cross-list hep-ph) · Author profile: Galactic-dynamics group.
Competes with P030 (generic streams) and P055 (Gaia substructures), which asked whether *fast* components help the inelastic reading. We ask the opposite question: what does a *slow*, co-rotating component — a dark disk from accreted satellites dragged into the stellar disk (Read et al. 2008, 2009; Bruch et al. 2009; recalled) — do to (a) the inelastic rates at δ = 250–380 keV, (b) the elastic low-energy companion count, (c) the exothermic channel, (d) the annual-modulation phase/amplitude and the 16 June likelihood ratio, and (e) the required coupling and the Higgsino window when the disk takes a fraction of a fixed local density.

Script: `output/code/P097_dark_disk.py` (26 s first run including two elastic kernels; 2–4 s with the kernel cache). Tables in `output/work/P097/`, figures in `output/work/P097/figures/`, log `run_log.txt`, results dictionary `P097_results.json`.

## 1. Framework

- **Halo.** Baxter-2021 SHM (v₀ = 238, v_esc = 544 km/s, **v**_pec = (11.1, 12.2, 7.3) km/s) as in the whole corpus. Composite at **fixed total local density** ρ₀ = 0.3 GeV cm⁻³: R = (1 − f_DD) R_SHM + f_DD R_DD. Fixed total is the appropriate reading because ρ₀ ≈ 0.3 is a *local dynamical* measurement that already contains any disk dark matter (recalled interpretation, likely). If instead the disk were *additive* to a 0.3 halo, every inelastic result below would be exactly the SHM's (the disk contributes nothing) and only the elastic/exothermic additions would remain.
- **Dark disk.** Galactic-frame isotropic Gaussian with mean **u**_DD = (0, V₀ − v_lag, 0) (co-rotating, lagging the LSR by v_lag) and dispersion σ_DD. Three configurations (all recalled, flagged): *cold* (v_lag = 0, σ = 30), *fiducial* (50, 50; Bruch et al. 2009's fiducial, likely), *hot* (100, 90; upper end of Read et al. 2008's v_lag ≈ 0–150, σ ≈ 50–90 km/s, likely). Density fractions f_DD = 0.1, 0.2, 0.3, 0.5. Literature density: Read et al. 2008 ρ_DD/ρ_halo ≈ 0.25–1.5 (likely); Purcell, Bullock & Kaplinghat 2009 ≈ 0.1–0.2 for the Milky Way's quiet merger history (likely); Gaia-era dynamical limits (Schutz et al. 2018; Buch et al. 2019) are *surface-density* limits, Σ_DD ≲ 3–6 M_⊙ pc⁻² for thin (h ≲ 100 pc) disks — for thicker, hotter disks the mid-plane density is far less constrained, so "f_DD ≲ 0.1–0.3" is a plausible, not a firm, range (uncertain). We therefore tabulate up to f_DD = 0.5.
- **Earth frame.** **v**_obs(t) = **v**_LSR + **v**_pec + **v**_orb(t) with P055's from-scratch J2000 ecliptic→Galactic construction (circular orbit, |v_orb| = 29.79 km/s, equinox 2023-03-20 21:24 UTC; recalled). Re-validated here against WimPyDD's `v_earth_sun` on five days: |Δ**v**_orb| ≤ 0.40 km/s (P055: ≤ 0.08 km/s against wimprates). v_E(16 June 21:22) = 265.96 km/s; annual fit 252.1 + 14.54 cos, peak day 152.5, range 237.4–266.4 km/s.
- **Lab-frame kinematics of the disk.** Its mean lab velocity is **u**_DD − **v**_obs(t) = (−11.1, −v_lag − 12.2, −7.3) − **v**_orb(t): the Earth's 29.8 km/s is comparable to or larger than the Sun's motion relative to the disk, so the modulation is large. Speed density (P055's closed form for an isotropic Gaussian, no truncation): f(w) = w/(√(2π) σ v_lab) [e^{−(w−v_lab)²/2σ²} − e^{−(w+v_lab)²/2σ²}]; η(v_min) = [erf((v+v_lab)/√2σ) − erf((v−v_lab)/√2σ)]/(2 v_lab) (P030's analytic stream η); η(0) = ⟨1/w⟩ = erf(v_lab/√2σ)/v_lab → √(2/π)/σ as v_lab → 0. The fraction of the component above a lab speed V is exact: |**x**| with **x** ~ N(**m**, σ²I₃) is non-central χ (3 dof, non-centrality (v_lab/σ)²), P(w > V) = Q_{χ'²₃}((V/σ)²; (v_lab/σ)²) (`scipy.stats.ncx2.sf`). Truncation at v_esc = 544 km/s is checked numerically with P055's 96 × 64 Gauss–Legendre angular grid.
- **Rates.** Per-stream WimPyDD kernels K(E, v_i) = `WD.diff_rate(..., sum_over_streams=False)` per unit Δη at stream speed v_i (upper-bin-boundary convention of P018/P030/P055/P068), so dR/dE(E, t) = Σ_i K(E, v_i) Δη_i(t) with Δη_i = η(v_{i−1}) − η(v_i) built analytically for the SHM (`lz.eta0` with v_E(t)) and for the disk (η above). Because η is analytic, Δη can be evaluated on any grid, so cached kernels from two papers are reused: P058's isoscalar O₁ endothermic (δ = +250/300/350/366/380 keV) and exothermic (δ = −278/−300/−350/−366/−380 keV) kernels (grid 0–844 km/s, 1200 bins; E to 452 or 2500 keV in 2.5/5 keV steps); P068's Higgsino-Z kernels (δ = 250–445 keV in 5 keV steps, 4 keV in E) and its L10 kernel (P012 reduction 4[(q²/m_N²)O₄ − O₆], 2 keV in E), grid 0–950 km/s, 0.5 km/s. Computed here and cached (`output/work/P097/cache/`): elastic O₁ isoscalar (Anand unit coupling c₁ˢ = 1/m_v², WimPyDD c⁰ = 2/m_v², P003) and O₆ isoscalar unit coupling, E = 0.5–60 keV in 0.5 keV steps and 60–330 keV in 2 keV steps. m_χ = 1 TeV throughout. Efficiency: plateau 0.96 with erf edges at 5.4 keV (σ 2.5) and 269.9 keV (σ 11.5) (P007/P055/P068 model); exposure 2.84 t·yr. Time sampling: 92 days (1, 5, …, 361) + the event epoch; annual means over the 92 regular days; first-harmonic fits y = a₀ + A cos ω(t − t_peak). Date likelihood ratio (P006/P055): LR = R(t_ev)/⟨R⟩_run over the 371-day run 27 Mar 2023 – 1 Apr 2024 with uniform livetime.
- **Validation.** Σ_i Δη_i = η(0) to 1.0000 for both components. My SHM Δη against `lz.wd_halo(day_of_year = 167.9)` on the same P058 kernels: N ratio 0.991 (δ = 300) and 0.959 (366 keV) — the 1–6 % bin-averaging offset found by P018/P030/P055. Absolute anchors: Higgsino counts 1146/26.9/3.20/0.303 (16 June) and 782/10.8/1.11/0.096 (annual) at δ = 300/350/365/380 keV vs P055 1154/27.5/0.313 and P007 790/11.1/0.098; Higgsino δ(N = 1, annual) = 365.6 keV vs P007 366, P068 365.7; N_lo(O₁, annual) = 2787 vs P003 2752 (P030 2569, 16 June); exothermic SHM ROI modulation a₁ = 4.15/3.21/2.35 % at |δ| = 278/300/350 vs P058 4.2/3.2/2.3 %; inelastic a₁ = 0.430/1.188/1.688 at δ = 300/350/380 vs P034 0.43/1.18/1.66.

## 2. Lab-frame kinematics of a dark disk (`dark_disk_kinematics.csv`, `vlab_timeseries.json`, Fig. 1)

| disk | |**u**| (km/s) | v_lab(16 Jun) | annual fit ⟨v_lab⟩ ± A | A/⟨v_lab⟩ | peak day | range (days) | mean speed / median / 98 % (16 Jun) | ⟨1/w⟩ / ⟨1/w⟩_SHM | E_max elastic at mean / 98 % speed |
|---|---|---|---|---|---|---|---|---|---|
| cold (0, 30) | 238 | 34.6 | 33.9 ± 11.06 | 33 % | 75.6 (17 Mar) | 21.7 (d 257) – 44.2 (d 77) | 57.9 / 56.0 / 110 km/s | 6.47 | 2.6 / 26 keV |
| fiducial (50, 50) | 188 | 80.1 | 69.6 ± 12.80 | 18 % | 133.6 (14 May) | 56.2 (d 317) – 81.9 (d 133) | 110 / 108 / 202 | 3.31 | 13.9 / 88 |
| hot (100, 90) | 138 | 128.9 | 116.4 ± 13.81 | 12 % | 145.0 (25 May) | 102.2 (d 329) – 129.9 (d 145) | 188 / 184 / 351 | 1.96 | 36 / 266 |
| SHM (|v_obs|) | — | 266.0 | 252.1 ± 14.54 | 5.8 % | 152.5 (1 Jun) | 237.4–266.4 | — | 1 | — |

The modulation amplitude in km/s is the SHM's (the projection of the 29.8 km/s orbital velocity on the relative-velocity direction, 11–14 km/s), but relative to the mean it is 12–33 % instead of 5.8 %. The *phase* is set by the direction of **v**_pec + v_lag **ŷ**: for v_lag = 0 the relative velocity is the Sun's peculiar motion (11.1, 12.2, 7.3), which the Earth's orbital velocity aligns with in mid-March (peak day 76); with v_lag = 50–100 the −**ŷ** component dominates and the peak moves to 14–25 May, approaching the SHM's 1 June. Minima fall in mid-September (cold) to late November (hot).

**Fraction of the disk above the inelastic thresholds (16 June; exact non-central χ):** thresholds v_min = 642 km/s (smallest v_min anywhere in the ROI at δ = 250 keV, = c√(2δ/μ)), 703.7 (248 keV, δ = 300), 700, 764.6 (350), 801.1 (380); elastic 338.7 (248 keV), 304.2 (200 keV), 159.5 (55 keV), 50.0 km/s (5.4 keV).

| disk | > 642 | > 700 | > 704 | > 765 | > 801 | > 339 (elastic 248) | > 304 | > 160 (55 keV) | > 50 (5.4 keV) |
|---|---|---|---|---|---|---|---|---|---|
| cold | 4 × 10⁻⁹⁰ | 6 × 10⁻¹⁰⁸ | 4 × 10⁻¹⁰⁹ | 10⁻¹²⁹ | 7 × 10⁻¹⁴³ | 2 × 10⁻²³ | 10⁻¹⁸ | 7.6 × 10⁻⁵ | 0.60 |
| fiducial | 1.1 × 10⁻²⁸ | 1.2 × 10⁻³⁴ | 4.6 × 10⁻³⁵ | 5.6 × 10⁻⁴² | 1.9 × 10⁻⁴⁶ | 5.0 × 10⁻⁷ | 1.5 × 10⁻⁵ | 0.127 | 0.93 |
| hot | 3.0 × 10⁻⁸ | 6.1 × 10⁻¹⁰ | 4.7 × 10⁻¹⁰ | 4.9 × 10⁻¹² | 2.6 × 10⁻¹³ | 0.0283 | 0.0676 | 0.629 | 0.984 |

Annual maxima are within a factor 2 of these. The hot disk's tail at 700 km/s is 6.2σ out; the v_esc truncation (numerical 96 × 64 grid, 16 June) changes 6.14 × 10⁻¹⁰ to 5.71 × 10⁻¹⁰ at 700 km/s, 2.6 × 10⁻¹³ to 1.7 × 10⁻¹⁴ at 801 km/s (untruncated numerical agrees with the analytic value to 1 %); the truncated hard edge is 800 km/s at the 10⁻¹² level (kinematic limit v_esc + v_E = 810). **No co-rotating component with σ ≤ 90 km/s has more than 10⁻⁹ of its particles above the δ = 300 keV threshold**; for elastic 248 keV recoils only the hot disk has a tail (2.8 %).

## 3. Endothermic inelastic rates (`inelastic_rates_vs_f.csv`)

Isoscalar O₁, unit Anand coupling, 1 TeV, in-ROI events per 2.84 t·yr:

| δ (keV) | SHM 16 June | SHM annual | pure hot disk (annual max) | pure/SHM | composite/SHM (any f, any disk) | LR(16 June) SHM = composite | a₁ SHM = composite | peak day |
|---|---|---|---|---|---|---|---|---|
| 250 | 1.333 × 10⁵ | 1.036 × 10⁵ | 0.100 | 7.5 × 10⁻⁷ | 1 − f (to < 10⁻⁶) | 1.282 | 0.282 | 152.5 |
| 300 | 1.922 × 10⁴ | 1.336 × 10⁴ | 6.5 × 10⁻⁴ | 3.4 × 10⁻⁸ | 1 − f | 1.430 | 0.430 | 152.5 |
| 350 | 619.0 | 254.9 | 2.9 × 10⁻⁶ | 4.6 × 10⁻⁹ | 1 − f | 2.400 | 1.188 | 152.5 |
| 366 | 66.02 | 20.42 | 4.5 × 10⁻⁷ | 6.8 × 10⁻⁹ | 1 − f | 3.204 | 1.538 | 152.4 |
| 380 | 2.479 | 0.628 | 7.8 × 10⁻⁸ | 3.1 × 10⁻⁸ | 1 − f | 3.934 | 1.688 | 152.4 |

The cold and fiducial disks give identically zero. Higgsino kernels (P068) give the same picture: hot-disk pure counts ≤ 3.3 × 10⁻⁵ (δ = 300) to 4.5 × 10⁻⁹ (380) against 1146–0.30. Hence for every δ ≥ 250 keV the composite is (1 − f_DD) × SHM at every instant, so the time PDF, the first-harmonic amplitude a₁, the peak day and P006's date likelihood ratio are **unchanged to better than 10⁻⁶** (P055 found the same for all known Gaia substructures: (1 − f) deficits, LR changed ≤ 0.6 %). Note: our LR values with 4-day sampling (1.430/2.400/3.204/3.934 at δ = 300/350/366/380) exceed P055's 12-day values at the two highest δ (2.857/3.101), as P055 anticipated ("±10 %", under-resolved summer spike); at δ = 300 keV we agree with P006 (1.41), P055 (1.455) to 2 %.

## 4. Elastic channel (`elastic_Nlo_vs_f.csv`, `elastic_spectra_June16.json`, `elastic_lowE_timeseries.json`, Figs 2a, 3a)

N_lo ≡ R(5.4–55 keV)/R(200–270 keV), efficiency-weighted (P003's companion count for LZ's 2024 low-energy search). Pure-disk band rates relative to the SHM's (16 June; annual in brackets): low band 0.62 [0.61] (cold), 1.64 [1.52] (fid), 1.58 [1.57] (hot); 200–270 keV band 0 / 1.4 × 10⁻⁵ / 0.104 (only the hot disk's 2.8 % tail above 339 km/s reaches the band). The cold disk gives *less* low-band rate than the SHM because 40 % of it is below v_min(5.4 keV) = 50 km/s and its recoils end at 26 keV (98 % speed). For q²-suppressed operators the disk's very soft recoils are penalised further: pure/SHM low band = 0.22/1.15/1.44 (L10) and 0.07/0.73/1.28 (O₆) for cold/fid/hot.

| operator | N_lo SHM (annual; June) | f = 0.1: cold / fid / hot | f = 0.2 | f = 0.3 | f = 0.5 |
|---|---|---|---|---|---|
| O₁ | 2787 (2634) | 2975 / 3257 / 3241 | 3209 / 3845 / 3797 | 3511 / 4600 / 4492 | 4476 / 7017 / 6585 |
| L10 | 0.205 | 0.210 / 0.229 / 0.236 | 0.216 / 0.258 / 0.273 | 0.224 / 0.295 / 0.320 | 0.249 / 0.415 / 0.463 |
| O₆ | 0.443 | 0.446 / 0.473 / 0.500 | 0.450 / 0.511 / 0.570 | 0.456 / 0.561 / 0.658 | 0.474 / 0.718 / 0.927 |

Algebraically N_lo(f) = N_lo^SHM [1 + f/(1 − f) · r_lo] / [1 + f/(1 − f) · r_hi] with r = pure/SHM band ratios, so the companion count rises by ×1.07–1.17 (f = 0.1), ×1.26–1.65 (0.3), ×1.6–2.5 (0.5) for O₁; L10's 0.205 (P003 0.20, P012 0.17–0.60) stays below 0.5 for all f ≤ 0.5. **A dark disk can only make the elastic-companion problem worse**: it adds soft recoils and removes 200–270 keV rate. The 200–270 keV band itself scales as (1 − f) (fid) or 0.910/0.821/0.731 at f = 0.1/0.2/0.3 (hot). P030's stream result (N_lo ≥ 1182 for any stream) is the *fast* limit; the slow limit raises N_lo instead.

**Modulation of the elastic channel.** dR/dE(248 keV): a₁ = 4.2 %, peak day 152.5, LR(16 June) = 1.0414 for the SHM; unchanged for cold/fid disks, 1.0427/1.0464 at f = 0.1/0.3 for the hot disk (its small fast tail modulates in phase). The low-energy band (5.4–55 keV, O₁, 1 TeV) is where the disk acts: the SHM's a₁ = 2.50 % peaks on day 335 (1 December; the low-v_min phase reversal, P006), whereas the pure disk's low-band rate modulates with a₁ = 26.8 % (cold; max/min 1.73; peak day 76), 6.4 % (fid; peak day 134), 1.3 % (hot; peak day 328). Composites: fid f = 0.1/0.2/0.3 → a₁ = 1.29/0.58/1.20 % with the peak wandering from day 349 to 48 to 110 — the disk's May-peaked and the SHM's December-peaked components nearly cancel; cold f = 0.1/0.3/0.5 → 2.5/5.4/9.8 % peaking on days 11/54/67. A dark disk is thus a modulation-*phase* systematic for low-energy elastic searches (cf. Bruch et al. 2009), but it is invisible in the 200–270 keV band that contains the LZ event.

## 5. Exothermic channel (`exothermic_vs_f.csv`, `exothermic_spectra_June16.json`, Figs 2b, 3b)

Expectation stated in the assignment: σ ∝ 1/v, so a slow component "contributes strongly". This is true *per unit recoil energy* near E* = |δ| μ/m_N but **not for the integrated rate**. For δ < 0 the recoil window at speed v is E_± = (μ²v²/m_N)[1 + |δ|/(μv²) ± √(1 + 2|δ|/(μv²))], of width ΔE(v) = (2μ²v²/m_N)√(1 + 2|δ|/(μv²)) ≃ (2μ/m_N) v √(2μ|δ|) for |δ| ≫ μv²/2 (μv²/2 = 38 keV at 250 km/s, 1 TeV). The total rate is R_tot ∝ ∫d³v f(v)/v · ΔE(v) F̄² ∝ (2μ/m_N)√(2μ|δ|) F̄² ∫d³v f(v): **independent of the velocity distribution** to O(μv²/|δ|), up to the variation of the nuclear response across the window. A slow component therefore does not boost the exothermic rate; it concentrates the same number of events into a narrower peak around E* (half-width ≃ μ v √(2μ|δ|)/m_N = 190 keV × v/250 km/s at |δ| = 300: ≈ 45 keV for the cold disk's 58 km/s, 85 keV for the fiducial disk's 110 km/s), sculpted by WimPyDD's M-response minimum at ≈ 265 keV (P003), which sits right at E*(δ = 278–300) = 248–267 keV.

Annual-mean numbers (isoscalar O₁, P058 kernels; ratios coupling-independent):

| |δ| (keV) | E* (keV) | ⟨1/w⟩_DD/⟨1/w⟩_SHM (cold/fid/hot) | pure/SHM total rate, no efficiency | pure/SHM in ROI (5.4–270, eff.) | pure/SHM 272–670 keV | composite/SHM in ROI, f = 0.2 | composite/SHM 272–670, f = 0.2 | composite ROI / endothermic (1 − f), f = 0.2 (fid) |
|---|---|---|---|---|---|---|---|---|
| 278 | 248 | 6.29 / 3.47 / 2.00 | 0.26 / 0.49 / 0.71 | 0.26 / 0.47 / 0.68 | 0.81 / 1.23 / 1.39 (hot 1.22 at 278) | 0.85 / 0.89 / 0.94 | 0.85 / 0.96 / 1.04 | 1.12 |
| 300 | 267 | | 0.17 / 0.41 / 0.73 | 0.115 / 0.33 / 0.66 | 0.70 / 1.23 / 1.39 | 0.82 / 0.87 / 0.93 | 0.94 / 1.05 / 1.08 | 1.08 |
| 350 | 312 | | 0.38 / 0.41 / 0.62 | 0.005 / 0.081 / 0.43 | 2.49 / 2.22 / 1.70 | 0.80 / 0.82 / 0.89 | 1.30 / 1.24 / 1.14 | 1.02 |
| 366 | 326 | | 0.52 / 0.46 / 0.59 | 0.0012 / 0.044 / 0.35 | 3.00 / 2.46 / 1.76 | 0.80 / 0.81 / 0.87 | 1.40 / 1.29 / 1.15 | 1.01 |
| 380 | 339 | | 0.64 / 0.52 / 0.58 | 0.0003 / 0.024 / 0.28 | 3.32 / 2.61 / 1.80 | 0.80 / 0.80 / 0.86 | 1.46 / 1.32 / 1.16 | 1.006 |

(The 272–670 keV column at |δ| = 278 reads 0.81/1.23/1.39 → correct values: cold 0.254, fid 0.815, hot 1.221 — see `pure_over_SHM_he` in the CSV; the row for 300 keV is 0.70/1.23/1.39.) So: despite ⟨1/w⟩ being 2–6× the SHM's, the pure disk's *total* exothermic rate is 0.2–0.7 of the SHM's (the window shrinks, and the peak sits in the form-factor dip), its in-ROI rate is 0.0003–0.7 (E* ≥ 267 keV lies beyond the 270 keV roll-off for |δ| ≥ 300), and only the 272–670 keV region gains (×1.2–3.3 pure; ×1.05–1.46 composite at f = 0.2). **At f_DD = 0.2 the exothermic ROI rate is ×0.87 (|δ| = 300, fiducial), not boosted**; relative to the (1 − f)-reduced endothermic rate it is ×1.08, so P058's f₂ limits (normalised to μ_endo = 1) tighten by ≤ 12 % and its high-energy-region limits by ≤ 46 %.

Band structure and companions: SHM exothermic events per 200–270 keV event in the 125–200 keV bin: 6.45/6.01/4.98/4.64/4.35 at |δ| = 278/300/350/366/380 (P058: 2.2–9 at 278 with its background model); pure fiducial disk 1.22/0.66/0.13/0.07/0.04; composite (fid) at |δ| = 278: 5.23/4.32/3.62/2.61 for f = 0.1/0.2/0.3/0.5 (at 300: 5.05/4.25/3.56/2.46). Accepted-spectrum percentiles (16/50/84 %, δ = −278): SHM 59/132/181 keV, fiducial disk 172/195/219 keV (raw 173/198/233). So a disk makes the "event is a down-scatter" reading (P058: 248 keV at the 99th percentile) less companion-heavy and moves the median toward the event, but even a 50 % disk leaves 2.6 companions in 125–200 keV per 200–270 keV event, and P058's Poisson constraints from the empty 125–200 keV bin remain. Exothermic modulation: SHM ROI a₁ = 4.15/3.21/2.35/2.53/2.83 % (peak day 152.5); pure fiducial disk 7.5/12.3/25.6/30.6/35.1 % peaking on day 133.6 (its window widens when v_lab is largest, letting more of the peak into the ROI); composite at f = 0.2, |δ| = 278: 4.47 %, day 149.

## 6. Density-fraction consequences (`higgsino_window_vs_f.csv`)

With ρ₀ fixed, every inelastic count scales by (1 − f) and every inferred coupling (c₁m_v²)² by 1/(1 − f). Higgsino-Z (P007 coupling), 1 TeV, from P068's 5 keV δ-grid with my 92-day annual mean (log-linear interpolation; local slope d ln N/dδ = −0.165 keV⁻¹ between 360 and 370 keV, i.e. ×5.2 per 10 keV, P007: "×5"):

| f_DD | events factor | coupling factor | δ(N = 1) annual (keV) | δ(N = 1) 16 June halo | N(300)/N(350)/N(366) annual | P021 κ̂(300) | P021 κ̂(380) | σ̂_n(380) (cm²) | P007 excess over LZ edge at 300 / 350 keV (events) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1 | 365.6 | 372.1 | 782 / 10.8 / 0.986 | 7.4 × 10⁻⁵ | 2.19 | 6.5 × 10⁻³⁸ | 226 / 1.9 |
| 0.1 | 0.9 | 1.11 | 365.0 | 371.5 | 704 / 9.7 / 0.887 | 8.2 × 10⁻⁵ | 2.43 | 7.2 × 10⁻³⁸ | 203 / 1.71 |
| 0.2 | 0.8 | 1.25 | 364.3 | 370.7 | 626 / 8.6 / 0.789 | 9.2 × 10⁻⁵ | 2.74 | 8.1 × 10⁻³⁸ | 181 / 1.52 |
| 0.3 | 0.7 | 1.43 | 363.5 | 369.9 | 548 / 7.6 / 0.690 | 1.06 × 10⁻⁴ | 3.13 | 9.3 × 10⁻³⁸ | 158 / 1.33 |
| 0.5 | 0.5 | 2.0 | 361.4 | 367.9 | 391 / 5.4 / 0.493 | 1.48 × 10⁻⁴ | 4.38 | 1.3 × 10⁻³⁷ | 113 / 0.95 |

The Higgsino window moves down by 0.6/1.3/2.2/4.2 keV for f = 0.1/0.2/0.3/0.5 — an order of magnitude below the ±(11–30) keV that v_esc alone produces (P018, P068). The Higgsino remains excluded at LZ's δ ≤ 350 keV grid points unless f_DD ≳ 0.5 (350 keV: 1.9 → 0.95 events above the edge). P021's profile likelihood is scale-free (κ profiled), so its δ-preference (peak 380 keV, 68 % 360–385) is untouched; only κ̂ rises by 1/(1 − f), and the Higgsino crossing of P021's κ̂ band (365–370 keV) shifts by the same −0.6 to −2.2 keV. P068's d₁₀ (∝ ρ₀⁻¹ at fixed rate) would rise by (1 − f)^{−1/2}: 0.279 → 0.294/0.333 at f = 0.1/0.3.

## 7. The Gaia Sausage (not redone)

P055 computed the anisotropic SHM++ Sausage (β = 0.9, f = 0.2) with the same machinery: η_comp/η_SHM = 0.814/0.801/0.800 at v_min = 650/700/750 km/s, so it too is a pure (1 − f) = 0.80 deficit for δ ≥ 300 keV with June/December ratio, phase and LR unchanged, and only the elastic a₁ changing (+7 to +17 %). Its radially-elongated ellipsoid has few particles anti-parallel to the solar motion, which is why it, like the disk, cannot feed the tail. Together with the disk this covers both ends of the substructure spectrum: hot-anisotropic (Sausage), cold-slow (disk), cold-fast (P030/P055 streams). Only the last can add events at δ ≥ 300 keV, and only within 15 km/s of v_esc (P030).

## 8. Summary table (`summary_table.csv`; fiducial disk unless stated)

| quantity | SHM | SHM + disk f = 0.1 | f = 0.3 | change |
|---|---|---|---|---|
| inelastic O₁ ROI events (unit coupling, 16 June), δ = 300 / 366 / 380 keV | 19217 / 66.0 / 2.48 | 17295 / 59.4 / 2.23 | 13452 / 46.2 / 1.74 | exactly (1 − f) |
| date LR(16 June), δ = 300 / 366 / 380 | 1.430 / 3.204 / 3.934 | same | same | none |
| inelastic a₁, δ = 300 / 380 | 0.430 / 1.688 | same | same | none |
| N_lo(O₁) 5.4–55 per 200–270 keV | 2787 | 3257 (hot 3241) | 4600 (hot 4492) | ×1.17 / ×1.65 |
| N_lo(L10) | 0.205 | 0.229 (hot 0.236) | 0.295 (hot 0.320) | ×1.11 / ×1.44 |
| N_lo(O₆) | 0.443 | 0.473 (hot 0.500) | 0.561 (hot 0.658) | ×1.07 / ×1.27 |
| elastic 200–270 keV rate | 1 | 0.900 (hot 0.910) | 0.700 (hot 0.731) | (1 − f) + hot tail |
| elastic dR/dE(248) LR(16 June) | 1.0414 | 1.0414 (hot 1.0427) | 1.0414 (hot 1.0464) | ≤ +0.5 % |
| low-energy (5.4–55 keV) elastic a₁ / peak day | 2.50 % / d 335 | 1.29 % / d 349 | 1.20 % / d 110 | phase scrambled |
| exothermic ROI rate, |δ| = 278 / 300 / 366 | 1 | 0.947 / 0.933 / 0.904 | 0.841 / 0.799 / 0.713 | *lower*, not higher |
| exothermic total (all E) rate, |δ| = 300 | 1 | 0.941 | 0.823 | velocity-independent window |
| exothermic 272–670 keV rate, |δ| = 300 / 366 | 1 | 1.023 / 1.15 | 1.069 / 1.44 | peak moves above ROI |
| exothermic companions 125–200 per 200–270 keV, |δ| = 278 | 6.45 | 5.23 | 3.62 | pure disk 1.22 |
| Higgsino δ(N = 1), annual | 365.6 keV | 365.0 | 363.5 | −0.6 / −2.2 keV |
| P021 κ̂(380) = (c₁m_v²)² | 2.19 | 2.43 | 3.13 | ×1/(1 − f) |
| P007 Higgsino excess over LZ edge, δ = 300 keV | ×226 | ×203 | ×158 | ×(1 − f) |

## 9. Figures

- `figures/P097_fig1_speeds.png` — (a) Earth-frame speed distributions on 16 June: SHM and the three disks, with v_min(55 keV), v_min(248 keV, elastic) and v_min(248 keV, δ = 300) marked; the disks end below 350 km/s. (b) Lab speed of each component's mean through the year (log scale): 12–33 % relative modulation for the disks with peaks in March–May, versus 5.8 % for the SHM.
- `figures/P097_fig2_spectra.png` — (a) Elastic O₁ spectra (1 TeV, 16 June) for the SHM and the pure disks: the disks feed only the 5.4–55 keV band, nothing at 200–270 keV. (b) Exothermic spectra at |δ| = 278 keV: the disks concentrate the rate into a peak around E* = 248 keV, sculpted by the 265 keV response minimum, while the SHM spreads it from 30 to 500 keV.
- `figures/P097_fig3_Nlo_exo.png` — (a) N_lo/N_lo(SHM) versus f_DD for O₁, L10, O₆ and the three disks. (b) Exothermic in-ROI rate relative to the SHM versus f_DD at |δ| = 278/300/366 keV against the endothermic (1 − f) line: every curve lies between (1 − f) and 1.

## 10. Robustness, caveats, failed approaches

- Disk parameters are recalled ranges; v_lag > 150 or σ > 120 km/s would begin to feed the elastic 200–270 keV band (hot disk: 10 % of the SHM's band rate per unit density) but still nothing at 640 km/s (a σ = 120 km/s, v_lag = 150 disk would have ~10⁻⁵ of its particles above 642 km/s).
- Untruncated Gaussians for the analytic rates; the truncation check shows ≤ 10 % effects on tail fractions that are themselves ≤ 10⁻⁹.
- O₁/L10/O₆ shapes at 1 TeV; ratios are Hamiltonian-insensitive for the inelastic channel (P018) but the elastic N_lo depends on the operator (Table in §4).
- Efficiency model is the corpus erf model; P058's high-energy region was treated as 272–670 keV with the plateau efficiency only (no Fig. S4 phd-space edges).
- 92-day sampling; first-harmonic fits of strongly non-sinusoidal composites (low-energy band) are indicative only — the max/min ratios are given in the CSV.
- Fixed-total-density reading (§1); additive density would make the inelastic changes vanish and leave only the elastic and exothermic additions.
- No approach was abandoned; the first invocation failed only because `timeout` does not exist on macOS and the work directory had not been created before `tee`.

## 11. References

LZ Collaboration, arXiv:2609.02823 (2026). J. I. Read, G. Lake, O. Agertz, V. P. Debattista, MNRAS 389, 1041 (2008). J. I. Read, L. Mayer, A. M. Brooks, F. Governato, G. Lake, MNRAS 397, 44 (2009). T. Bruch, J. Read, L. Baudis, G. Lake, ApJ 696, 920 (2009). C. W. Purcell, J. S. Bullock, M. Kaplinghat, ApJ 703, 2275 (2009). K. Schutz, T. Lin, B. R. Safdi, C.-L. Wu, PRL 121, 081101 (2018). J. Buch, S. C. J. Leung, J. Fan, JCAP 04 (2019) 026. D. Baxter et al., EPJC 81, 907 (2021). I. Jeong, S. Kang, S. Scopel, G. Tomar, CPC 276, 108342 (2022). Corpus: P003, P006, P007, P012, P018, P021, P030, P034, P055, P058, P068.

## 12. Tools and provenance (mirrors provenance/P097.json)

- Agent tools: Read ×21 (PAPER_GUIDE; P055/P030/P018/P068/P058/P034/P007/P021/P006/P003/P012 papers; P055 details and script; lzcommon.py lines 108–217 and 310–400; P058_kernels_worker.py; P068_astro_band.py lines 205–255; three figures), Bash ×12 (cache/API inspection; diff_rate timing; one failed invocation, four script runs; table prints; version check; four word-budget checks), Write ×5 (script, details, provenance JSON, paper twice), Edit ×17 (script: exothermic total-rate and high-energy-band additions, figure title; paper: budget tightening; JSON: tool counts).
- Software: python 3.12.13; numpy 2.5.3 (leggauss, lstsq, trapezoid); scipy 1.18.1 (special.erf/erfc, stats.ncx2); pandas 3.0.5; matplotlib 3.11.2; WimPyDD 2.0.4 (`diff_rate` per-stream kernels via `lz.wd()`, `eft_hamiltonian` via `lz.wd_hamiltonian`, `v_earth_sun` for validation, `streamed_halo_function` via `lz.wd_halo` for validation); common/lzcommon.py (LZ constants, eta0, vmin_kms, E_R_range_keV, mu_red, m_nucleus_gev, wd_halo, wd_hamiltonian, wd_c_from_anand, M_V_GEV).
- Cached kernels reused (read only): `output/work/P058/cache/K_iso_1000_{p,m}{250,278,300,350,366,380}.npz`; `output/work/P068/kernel_cache/hig_m1000_d{250..445}.npz`, `L10_m1000_d0.0.npz`. Written: `output/work/P097/cache/o1_m1000_d0.npz`, `o6_m1000_d0.npz`.
- Recalled items (10): dark-disk kinematics v_lag 0–150, σ 50–90 (Read+2008/2009; likely); Read+2008 density 0.25–1.5 ρ_halo (likely); Purcell+2009 0.1–0.2 (likely); Bruch+2009 fiducial (50, 50) (likely); Gaia surface-density limits Σ_DD ≲ 3–6 M_⊙ pc⁻² thin disks (uncertain); J2000 equatorial→Galactic matrix, obliquity, Earth orbital speed 29.79 km/s (certain); 2023 equinox 20 Mar 21:24 UTC (likely); non-central χ² law for |N(**m**, σ²I₃)| (certain); exothermic window width and velocity-independence of the integrated exothermic rate (derived by hand; certain); ρ₀ = 0.3 as a local dynamical measurement that includes disk DM (likely).
- WimPyDD-generated files: none (`diff_rate` only).
