# P030 — Streams and substructure: can a fast dark-matter stream produce the 248 keV event without a low-energy population? (research record)

Simulated date 2026-09-08. Category HALO (astro-ph.GA, cross-list hep-ph). Author profile: Galactic-archaeology-minded dark-matter theorists. Competes/overlaps with P055 (known stream candidates) and P097 (dark disk).

All numbers below are produced by `output/code/P030_streams.py` (run from the simulation root with `.venv/bin/python`; 147 s; stdout in `output/work/P030/run_log.txt`). Result tables are in `output/work/P030/*.csv|json`, figures in `output/work/P030/figures/`. Unless stated otherwise: m_χ = 1000 GeV, natural xenon, 16 June 2023 (day 167) Earth velocity, LZ exposure 2.84 t·yr, LZ efficiency model (0.96 plateau, 50 % at 5.4 and 269.9 keV, erf edges σ_lo = 2.5 keV, σ_hi = 11.5 keV; from P007), ρ₀ = 0.3 GeV/cm³.

## 1. Motivation and framework

P003 showed that elastic spin-independent (O1) scattering of a TeV WIMP in the Baxter-2021 halo predicts ≈ 2750 events in 5.4–55 keV per event in 200–270 keV, so a lone 248 keV recoil excludes O1 by three orders of magnitude. P002/P006/P007/P018 showed that inelastic dark matter reaches 248 keV only for δ ≤ δ_max = 387 keV (1 TeV, 16 June) and that the rate within ~40 keV of δ_max is set by the halo's high-velocity tail. A cold stream — a tidal debris component with bulk Galactic-frame velocity **u**_s, small dispersion σ and local density fraction f_s — adds a nearly monochromatic component to the Earth-frame speed distribution at v_lab = |**u**_s − **v**_obs|. We ask:

(a) Elastic: does any stream change R(200–270 keV)/R(5.4–55 keV) enough to allow a lone high-energy event?
(b) Inelastic: how does a stream extend δ_max and boost the rate at δ = 350–450 keV; which (v_lab, f_s) give one pure-Higgsino event (σ_n = 7.4 × 10⁻³⁹ cm², P007) at δ = 380/400/450 keV; what is the annual modulation of a stream signal?
(c) What do known substructures imply?
(d) What is the structural ceiling imposed by the Galactic escape speed?

## 2. Halo-function construction (Part 0)

WimPyDD's `diff_rate` accepts arrays (v_min,i, Δη_i). Following P018's convention check, v_min,i are upper bin boundaries and Δη_i = η(v_{i−1}) − η(v_i) ≥ 0 in (km/s)⁻¹, where η(v_min) = ∫_{v>v_min} f_E(**v**)/v d³v is the mean inverse speed. `diff_rate(..., sum_over_streams=False)` with unit weights returns the per-stream kernel K(E, v_i) (events/(t·yr·keV) per unit Δη), so dR/dE = K·Δη and everything is linear in Δη. Grid: v = 0–1100 km/s in 1 km/s steps (1101 boundaries).

**Stream.** A Galactic-frame Gaussian f(**u**) ∝ exp(−|**u** − **u**_s|²/2σ²) seen from Earth is a Maxwellian with v₀,s = √2 σ boosted by v_lab, hence (untruncated)

η_s(v_min) = [erf((v_min + v_lab)/v₀,s) − erf((v_min − v_lab)/v₀,s)] / (2 v_lab),

which is the McCabe/Lewin–Smith form with v_esc → ∞ (recalled, certain). η_s(0) = erf(v_lab/v₀,s)/v_lab → 1/v_lab (cold limit). Δη is obtained by differencing η_s on the grid. σ = 20 km/s per axis (assignment); the stream's speed distribution then spans v_lab ± 23 km/s at half maximum (validation.json: 777–824 km/s for v_lab = 800).

**SHM.** The same differencing applied to the analytic truncated Maxwellian `lz.eta0(v, v_e, 238, 544)` with v_e = |**v**_obs(16 June)| = 265.98 km/s (WimPyDD Earth-velocity vector **v**_obs = (11.27, 265.09, −18.50) km/s; **v**_orb = (0.17, 14.89, −25.80) km/s, |**v**_orb| = 29.79 km/s; **v**_sun = (11.1, 250.2, 7.3), |**v**_sun| = 250.55 km/s).

**Composite.** Δη = (1 − f_s) Δη_SHM + f_s Δη_s at fixed ρ₀; f_s is the stream's share of the local density.

**Bound stream truncated at v_esc (Part 4b).** For a stream anti-parallel to **v**_obs the problem is axisymmetric. With polar axis ê₃ = −**v**_obs/v_o: **u** = v**n** − v_o ê₃, **u**_s = +v_gal ê₃, |**u** − **u**_s|² = v² + V² − 2vVc (V = v_o + v_gal, c = cos θ), and |**u**| ≤ v_esc ⇔ c ≥ c_min = (v² + v_o² − v_esc²)/(2 v v_o). The angular integral is analytic:

S(v) = v ∫dΩ f = 2π v (σ²/vV) exp(−(v² + V²)/2σ²) [exp(vV/σ²) − exp(vV c_min/σ²)],  η(v_min) = ∫_{v_min}^∞ S(v) dv / N,

with N = ∫_{|u|≤v_esc} f d³u (1-D quadrature, 2 × 10⁵ points) and the v-integral on a 0.05 km/s grid. The truncated stream is renormalised to the same local density (f_s keeps its meaning). A first attempt with P018's 3-D shell quadrature (48 × 64 angular nodes) was abandoned because a σ = 20 km/s blob at 800 km/s subtends ~1.4°, below the 5.6° azimuthal spacing (η₀ off by 4–7 %); see "failed approaches".

### Validation (validation.json, bound_truncated_streams.csv)
- η₀ = ΣΔη: mine / WimPyDD = 1.000000 (η₀ = 3.3563 × 10⁻³ (km/s)⁻¹, both).
- η(v_min) mine / WimPyDD: 0.998 (200 km/s), 0.996 (400), 0.992 (600), 0.988 (700), 0.980 (750), 0.964 (780), 0.903 (800). WimPyDD's bin-averaged Δη puts more weight in the tail and its last non-zero bin is 811 km/s versus the analytic v_max = 809.98 km/s (P018 found the same).
- In-ROI Higgsino counts, my Δη_SHM vs WimPyDD's Δη: 1154.6 vs 1172.8 (δ = 300 keV, ratio 0.984), 27.51 vs 28.71 (350; 0.958), 0.3131 vs 0.3333 (380; 0.940). Assigning each bin's mass to the lower instead of the upper boundary gives 1118.7 / 25.24 / 0.2762 (ratios 0.954 / 0.880 / 0.829 to WimPyDD), so my upper-boundary construction carries at most a half-bin (0.5 km/s) bias, i.e. ≤ +1.5 % / +4 % / +6 % at δ = 300/350/380 keV, and sits between WimPyDD and the lower-boundary bracket. P007 (annual halo) gives N(380 keV) = 0.098; ours is the 16 June value.
- Stream normalisation: ΣΔη_s × v_lab = 1.0000 for v_lab = 600, 800, 1000 km/s.
- Axisymmetric exact integral without truncation reproduces the analytic stream: η₀ ratio 1.000000, max |ΔΔη|/max Δη = 5.2 × 10⁻⁷.

## 3. Part 1 — Elastic scattering (elastic_Nlo_Nhi.csv, part1_elastic.json)

Kinematics: elastic recoils from speed v span 0 ≤ E_R ≤ E_max = 2μ²v²/m_N. For 1 TeV on ⟨A⟩ = 131.3: E_max = 195, 540, 666, 778, 1059, 1384, 1751, 2162 keV for v_lab = 300, 500, 555, 600, 700, 800, 900, 1000 km/s. v_min(248 keV) = 338.7 km/s, v_min(200) = 304.2, v_min(270) = 353.4 km/s. A 300 km/s stream cannot produce 248 keV; any stream above ~340 km/s can.

Because η_s is flat (= 1/v_lab) for v_min < v_lab − 2σ, the recoil spectrum from a stream is the kernel column K(E, v_lab): flat in E_R times the nuclear response (M for O1; q²Σ″-type for O4; q⁴ for O6; q² for O10, with WimPyDD shell-model responses). Hence N_lo ≡ R(5.4–55 keV)/R(200–270 keV) (efficiency-weighted, as in P003) is **independent of v_lab** as long as E_max > 270 keV:

| operator (isoscalar) | SHM 16 June | monochromatic stream, any v_lab ≥ 500 km/s | SHM + 20 % stream at 800 km/s | SHM + 1 % stream at 800 km/s |
|---|---|---|---|---|
| O1 | 2569 | 1182 | 2322 | 2557 |
| O4 | 26.8 | 11.7 | 24.0 | — |
| O6 | 0.421 | 0.191 | 0.377 | — |
| O10 | 2.96 | 1.32 | 2.65 | — |

(P003's SHM values are 2752 / 28 / 0.43 / 3.1 with an annual halo; the 16 June halo is 7 % lower for O1.) The stream/SHM ratio is 2.17–2.30 for every operator: it is simply η_SHM(v_min ≈ 300–350 km/s)/η_SHM(v_min ≲ 100 km/s) ≈ 0.45, i.e. how much of the SHM is above the 200–270 keV threshold. Helm form factor: ∫F² ε over 5.4–55 keV = 12.09 keV, over 200–270 keV = 0.0342 keV, ratio 353 (mean F² 0.254 vs 6.1 × 10⁻⁴, width ratio 49.6/70 = 0.71); the shell-model value 1182 is 3.3× larger because the M-response node sits at 267 keV (inside the window; P017) instead of Helm's 279 keV. Composite halos with f_s = 1–20 % at 500–1000 km/s change O1's N_lo by < 10 % (2212–2582). **Conclusion (a):** no stream configuration brings elastic O1 below N_lo ≈ 1180; the low-energy population is fixed by the nuclear response, not by the velocity distribution. For the q⁴ spin operator O6 the SHM already gives N_lo < 1 (P003) and a stream halves it; for inelastic scattering with δ ≥ 300 keV N_lo = 0 for every halo (E₋ > 55 keV except at v_lab ≳ 1000 km/s, where E₋(δ = 300) = 45 keV).

## 4. Part 2 — Inelastic scattering with streams (inelastic_stream_grid.csv, deltamax_vs_vlab.csv, fs_for_one_event.csv, vlab_needed_for_fs.csv, fs_dispersion_sensitivity.csv)

Hamiltonian: O1 with WimPyDD c⁰ = c_p + c_n, c¹ = c_p − c_n, c_p = (G_F/√2)(1 − 4 sin²θ_W), c_n = −G_F/√2 (G_F = 1.166 × 10⁻⁵ GeV⁻², sin²θ_W = 0.231; recalled, certain; P007): σ_n = 7.4 × 10⁻³⁹ cm². δ grid 300–500 keV in 5 keV steps (41 kernels, each 2 keV in E from below E₋(1100 km/s) to 330 keV); v_lab grid 600–1000 km/s in 25 km/s steps.

### 4.1 δ_max and the recoil window
dδ_max/dv = √(2 m_N E_R)/c = 0.8218 keV per km/s (P018: 0.822). For a 248 keV recoil on ⟨A⟩: v_min = 703.7 / 764.6 / 801.1 / 825.4 / 886.2 km/s at δ = 300 / 350 / 380 / 400 / 450 keV. δ_max(248 keV) for a monochromatic stream: 296.9 (700 km/s), 338.0 (750), 379.1 (800), 399.7 (825), 420.2 (850), 461.3 (900), 543.5 keV (1000); with the +3σ dispersion tail 346 / 387 / 428 / 449 / 470 / 511 / 593 keV. The δ at which the pure-stream in-ROI rate falls to 10⁻³ of its δ = 300 value: 344 (700), 372 (750), 409 (800), 428 (825), 447 (850), 481 keV (900) — between the monochromatic and 3σ values, as expected. Recoil window at δ = 380 keV: 252–454 keV (800 km/s), 190–605 (825), 158–727 (850), 120–954 (900), 82–1403 keV (1000): fast streams push most of the inelastic spectrum above the ROI, so the in-ROI count saturates (2.4–2.8 × 10⁵ pure-stream events at δ = 380 for 900–1000 km/s).

### 4.2 Expected counts (pure stream f_s = 1, ρ₀ in the stream; 16 June)
SHM: N = 1155 (δ = 300), 27.5 (350), 3.32 (365), 0.313 (380), 1.4 × 10⁻³ (400), 0 (450).

| v_lab (km/s) | N(300) | N(380) | N(400) | N(450) |
|---|---|---|---|---|
| 600 | 3.6e−3 | 0 | 0 | 0 |
| 700 | 3.2e4 | 9.6e−3 | 2.8e−5 | 0 |
| 725 | 1.24e5 | 1.43 | 1.4e−2 | 1e−9 |
| 750 | 2.39e5 | 62.8 | 1.83 | 3.4e−6 |
| 775 | 3.03e5 | 1113 | 66.8 | 2.4e−3 |
| 800 | 3.20e5 | 1.00e4 | 973 | 0.436 |
| 825 | 3.39e5 | 4.59e4 | 7945 | 20.3 |
| 850 | 4.06e5 | 1.15e5 | 3.65e4 | 296 |
| 900 | 7.79e5 | 2.42e5 | 1.65e5 | 1.10e4 |
| 1000 | 2.46e6 | 2.82e5 | 2.56e5 | 1.88e5 |

Composite counts N = (1 − f_s)N_SHM + f_s N_stream and boosts N/N_SHM for f_s = 0.01/0.05/0.2 are tabulated for all (δ, v_lab) in inelastic_stream_grid.csv and plotted in Fig. 2a. Examples (f_s = 1 %): v_lab = 800 km/s gives ×2.9 at δ = 300, ×40 at 350, ×321 at 380 keV; v_lab = 700 gives ×1.3 / ×1.0 / ×1.0; v_lab = 1000 gives ×22 / ×1400 / ×9000.

### 4.3 f_s for one Higgsino event
f_s(N = 1) = (1 − N_SHM)/(N_stream − N_SHM):

| v_lab (km/s) | δ = 380 | δ = 400 | δ = 450 |
|---|---|---|---|
| 725 | 0.61 | 77 (impossible) | — |
| 750 | 1.1e−2 | 0.55 | — |
| 775 | 6.2e−4 | 1.5e−2 | 408 (impossible) |
| 800 | 6.9e−5 | 1.0e−3 | 2.3 (impossible) |
| 825 | 1.5e−5 | 1.3e−4 | 4.9e−2 |
| 850 | 6.0e−6 | 2.7e−5 | 3.4e−3 |
| 900 | 2.8e−6 | 6.1e−6 | 9.1e−5 |
| 1000 | 2.4e−6 | 3.9e−6 | 5.3e−6 |

v_lab needed for f_s = 1 / 0.2 / 0.05 / 0.01 / 0.001: δ = 380 keV: 725 / 732 / 741 / 751 / 771 km/s; δ = 400: 747 / 757 / 767 / 779 / 800; δ = 450: 805 / 816 / 825 / 840 / 866 km/s. At δ ≤ 365 keV the SHM alone gives ≥ 1 event for this coupling, so no stream is needed (P007: the Higgsino is excluded there by the LZ intervals).

**Dispersion dependence.** Values with v_lab below the monochromatic requirement (825 km/s at δ = 400, 886 at 450) come from the stream's Gaussian tail. At (δ = 400 keV, v_lab = 800 km/s) f_s(N=1) = 2.7 % / 0.55 % / 0.10 % / 0.010 % for σ = 5 / 10 / 20 / 40 km/s; at (450 keV, 850 km/s) 0.57 / 0.024 / 0.0034 / 2.9 × 10⁻⁴; at (380 keV, 800 km/s) 3.3 × 10⁻⁴ / 2.1 × 10⁻⁴ / 6.9 × 10⁻⁵ / 2.1 × 10⁻⁵. Any statement about δ above the mean-speed δ_max is a statement about the stream's velocity tail.

### 4.4 Annual modulation of a stream signal (stream_modulation.csv, modulation_timeseries.json, Fig. 3)
v_lab(t) = |**u**_s − **v**_sun − **v**_orb(t)| sampled every 3 days with WimPyDD's `v_earth_sun`. Semi-amplitude of v_lab and day of maximum for |**u**_s| = 544 km/s: anti-aligned with **v**_sun (retrograde) 14.6 km/s, day 151 (same phase as the SHM; the orbital projection on the solar-motion direction is 14.6 km/s); aligned (prograde) 14.6 km/s, day 334, but v_lab is only 281 km/s; vertical (+z, toward the NGP) 29.7 km/s, day 166; radial (+x, toward the GC) 26.4 km/s, day 244; perpendicular direction in the ecliptic plane 27.7 km/s, day 73; the maximal orbital projection is 29.8 km/s. Curiosity: because **v**_orb(16 June) = (0.2, 14.9, −25.8) km/s points mostly toward −z, a stream moving toward the north Galactic pole has its maximal lab speed on 15 June — but no such stream is fast enough (Sec. 5).

Rate modulation of a pure retrograde stream (v_gal = 544, v_lab = 780–809 km/s over the year) versus the SHM: peak/mean (max/min) = 1.55 (3.1) at δ = 350 keV, 2.22 (10) at 380, 2.56 (18) at 400 keV, against 2.57 (27) and 3.30 (10⁶) for the SHM at 350 and 380 keV; an unbound v_gal = 600 stream gives 1.08 (1.2) at 350 and 1.71 (4.1) at 400 keV. 16 June/mean: stream 1.53 (350), 2.15 (380); SHM 2.47 (350), 3.12 (380). A stream signal is therefore *less* seasonally modulated than the SHM tail at the same δ (its particles all sit above threshold and only the 1/v factor and the recoil window move), unless δ is within ~10 keV of the stream's own δ_max. The modulation fraction (max − min)/(max + min) at δ = 350 keV for pure anti-aligned streams: 0.999 (v_gal 400), 0.87 (500), 0.51 (544), 0.093 (600). Perpendicular streams at 544–600 km/s have non-zero δ = 380 keV rates for only 15–67 % of the year, peaking on days 73, 163–166 or 247–253 depending on direction — a distinct phase signature relative to the 2 June SHM peak (relevant to P055).

## 5. Part 3 — Known substructures (known_substructures.csv; all inputs recalled and flagged)

WimPyDD axes: x toward the GC, y along rotation, z toward the NGP. Speeds on 16 June; δ_max for a 248 keV recoil at 1 TeV; elastic E_max = 2μ²v_lab²/m_N; boosts are composite/SHM at the quoted maximal f_s with the quoted dispersion treated as isotropic Gaussian (untruncated).

| substructure (recalled) | **u**_s (km/s) | σ | f_s,max | v_lab 16 June | δ_max mono / +2σ (keV) | E_max elastic (keV) | boost δ = 300 / 350 / 380 |
|---|---|---|---|---|---|---|---|
| Sagittarius stream, −z (Freese+2004; Purcell+2012) | (0, 0, −300) | 20 | 0.03 | 387 | 40 / 72 | 324 | 0.97 / 0.97 / 0.97 |
| Sagittarius, +z variant | (0, 0, +300) | 20 | 0.03 | 415 | 62 / 95 | 372 | 0.97 / 0.97 / 0.97 |
| S1 stream (Myeong+2018; O'Hare+2018) | (−34, −306, −64) | 60 | 0.10 | 575 | 194 / 293 | 714 | 1.09 / 1.05 / 1.67* |
| S2 stream (O'Hare+2020) | (6, 164, −250) | 40 | 0.01 | 253 | < 0 / < 0 | 138 | 0.99 / 0.99 / 0.99 |
| Gaia-Enceladus/Sausage, +x | (250, 0, 0) | 80 | 0.20 | 357 | 15 / 147 | 276 | 0.80 / 0.80 / 0.80 |
| Gaia-Enceladus/Sausage, −x | (−250, 0, 0) | 80 | 0.20 | 373 | 28 / 159 | 300 | 0.80 / 0.80 / 0.80 |
| retrograde shards (Rg-type, O'Hare+2020) | (0, −280, 0) | 60 | 0.01 | 546 | 170 / 269 | 643 | 0.99 / 0.99 / 1.00 |
| Helmi streams | (0, 150, 250) | 30 | 0.01 | 292 | < 0 / 11 | 185 | 0.99 / 0.99 / 0.99 |
| hypothetical bound retrograde stream at v_esc | −544 v̂_sun | 20 | 0.01 | 809 | 387 / 419 | 1415 | 3.8 / 52 / 603 (untruncated; truncated 30 / 51, Sec. 6) |

*The S1 value at δ = 380 keV comes from the 4σ Gaussian tail (575 + 4 × 60 = 815 km/s), which is unphysical for a bound stream (its Galactic speed would exceed v_esc); with truncation it is ≈ 1. Sagittarius and S2 remove a few per cent of the SHM tail (boost 0.97–0.99) because their particles are all below threshold; the Sausage reproduces P018's 0.80. No known substructure has v_lab within 230 km/s of the 801 km/s needed at δ = 380 keV; S1, the fastest, reaches δ_max = 194 keV (293 keV at 2σ) and boosts the δ = 300–350 keV rate by ≤ 9 %. For elastic scattering the Sagittarius and S1 streams *can* produce 248 keV recoils (E_max 324–714 keV) but with N_lo = 1182 for O1 (pure-stream column in the CSV). Reliability flags: speeds of Sgr and S1 "likely"; directions (signs) and density fractions "uncertain"; S2, shards and Helmi parameters "uncertain".

## 6. Part 4 — Escape-velocity ceiling (part4_ceiling.json, tail_enhancement.csv, bound_stream_boost.csv, bound_truncated_streams.csv, truncated_stream_eta.json)

A stream bound to the Galaxy has |**u**_s| ≤ v_esc = 544 km/s (Baxter 2021; recalled likely: RAVE/Gaia estimates 500–580 km/s, P018), so v_lab ≤ v_esc + v_E = 809.98 km/s on 16 June (794.6 km/s annual mean), exactly the SHM's v_max. Therefore δ_max(248 keV) = 387.3 keV (⟨A⟩) is **identical** for the SHM and for any bound stream; heavier isotopes have higher ceilings (δ_max(248 keV, 810 km/s) = 393.3 keV on ¹³⁴Xe, 397.9 on ¹³⁶Xe; for a 270 keV recoil 391.3 / 397.8 / 402.5 keV; absolute ceilings μv²/2 = 397.7 / 405.0 / 410.4 keV). Beyond ≈ 400–410 keV no bound particle can produce an in-ROI recoil; δ = 400 / 420 / 450 / 500 keV at 248 keV need v_lab = 825 / 850 / 886 / 947 km/s, i.e. Galactic-frame speeds ≥ 559 / 584 / 620 / 681 km/s even when perfectly anti-aligned with the solar motion on 16 June — an unbound (extragalactic or hypervelocity) component. Its local density fraction is not known; estimates for unbound/extragalactic dark matter near the Sun are ≪ 1 % (recalled, uncertain). From Table 4.3 such a component would need f_s ≈ 1.3 × 10⁻⁴ (825 km/s) to 6 × 10⁻⁶ (900 km/s) at δ = 400 keV and 3 × 10⁻³ (850) to 9 × 10⁻⁵ (900 km/s) at δ = 450 keV, so even a 10⁻⁴ unbound fraction at ≳ 850 km/s would suffice at δ = 400 keV — but nothing in the data selects such a component, and the 16 June date carries no information about it (an unbound stream at 600 km/s modulates by only ±8 % at δ = 350 keV).

What a bound stream *can* do is concentrate density just below the ceiling. Tail enhancement η_composite/η_SHM (Gaussian stream, σ = 20 km/s):

| v_lab | f_s | v_min = 600 | 700 | 750 | 780 | 800 | 805 |
|---|---|---|---|---|---|---|---|
| 750 | 0.01 | 1.16 | 2.17 | 3.74 | 2.76 | 2.67 | 4.3 |
| 780 | 0.01 | 1.16 | 2.13 | 5.9 | 13.8 | 42 | 115 |
| 800 | 0.001 | 1.015 | 1.11 | 1.51 | 3.09 | 13.7 | 43 |
| 800 | 0.01 | 1.15 | 2.10 | 6.1 | 21.9 | 128 | 421 |
| 800 | 0.05 | 1.76 | 6.5 | 26.6 | 106 | 634 | 2103 |
| 810 | 0.01 | 1.15 | 2.09 | 6.1 | 23.9 | 174 | 621 |

Rate boosts (composite/SHM, Gaussian untruncated) at δ = 350 / 365 / 370 / 380 / 385 keV: v_lab = 750, f_s = 1 %: 2.9 / 2.9 / 3.1 / 3.0 / 2.9; 780: 17.5 / 34 / 44 / 59 / 68; 800: 40 / 122 / 181 / 321 / 411; 810: 53 / 196 / 313 / 639 / 873. f_s(N = 1) at δ = 380 keV: 1.1 × 10⁻² (750), 3.8 × 10⁻⁴ (780), 6.9 × 10⁻⁵ (800), 3.4 × 10⁻⁵ (810).

**Truncation at v_esc (exact axisymmetric integral, Sec. 2).** For retrograde streams anti-parallel to **v**_obs with v_gal = 500 / 530 / 544 km/s (v_lab = 766 / 796 / 810 km/s): 1.5 / 25 / 51 % of the Gaussian lies beyond v_esc and is removed; the truncated distribution ends exactly at 810 km/s. Pure-stream counts, untruncated → truncated: δ = 350: 19059 → 16635, 93166 → 59569, 143921 → 78882; δ = 380: 429 → 211, 7359 → 1078, 19936 → 1564; δ = 390: 98.9 → 42.2, 2303 → 297, 7389 → 475; δ = 400: 20.7 → 2.17, 661 → 20.8, 2396 → 37.6. Composite boosts with f_s = 1 %: δ = 350: 7.0 / 22.6 / 29.7; 365: 7.7 / 34.2 / 50.0; 380: 7.7 / 35.4 / 51.0; 385: 8.1 / 42.9 / 64.3; 390: 8.9 / 56.9 / 90.5; 400: 16.5 / 150 / 270. f_s(N = 1) at δ = 380 keV: 3.3 × 10⁻³ / 6.4 × 10⁻⁴ / 4.4 × 10⁻⁴; at δ = 400 keV: 0.46 / 0.048 / 0.027. The residual δ = 400 keV rate of bound streams is real but comes entirely from ¹³⁴Xe/¹³⁶Xe recoils at 257–300 keV (E₋(¹³⁶Xe, 810 km/s, 400 keV) = 257 keV) on the efficiency roll-off — it requires the event to be a ≳ 260 keV recoil (P009 finds 262 ± 11 keV with the printed NEST parameters) and a stream within 15 km/s of the escape speed carrying ≥ 3–5 % of the local density.

**Structural conclusion.** Bound streams cannot move δ_max; they can raise the rate just below it by ×30–50 (f_s = 1 %, v_gal ≥ 530 km/s) to ×300 (f_s = 5 %), turning P007's one-event window (δ = 358–380 keV at 1 TeV, annual halo) into one shifted up by ≈ 10–15 keV toward δ_max (from Table 4.3: at v_lab = 800, f_s = 10⁻³–10⁻² the Higgsino gives one event at δ ≈ 385–395 keV instead of 366 keV), at the price of postulating a component that no Galactic-archaeology survey has reported: a cold, retrograde stream with v_gal ≳ 500 km/s aligned within ~30° of the solar antapex. Extending δ beyond ≈ 400 keV requires unbound dark matter.

## 7. Figures
- `figures/P030_fig1_composite_eta.png` — (a) η(v_min) on 16 June for the SHM and SHM + 1 % streams at v_lab = 700–1000 km/s; dashed verticals v_min(248 keV, 1 TeV, δ = 300–450 keV); red: bound ceiling 810 km/s. (b) η_composite/η_SHM for f_s = 1 % (solid) and 20 % (dashed): enhancement 10²–10⁵ within 10 km/s of the ceiling, 0.8–1 for streams slower than the threshold.
- `figures/P030_fig2_rate_boost.png` — (a) expected pure-Higgsino LZ events vs δ (16 June) for the SHM and composites with v_lab = 600–1000 km/s, f_s = 1 % (dotted) and 20 % (solid); grey band 0.105–3.65 events (one-event PLR band, P007); red: δ_max = 387 keV. (b) f_s giving one event vs v_lab at δ = 380/400/450 keV (Gaussian σ = 20 km/s, untruncated); red: bound/unbound boundary.
- `figures/P030_fig3_modulation.png` — (a) v_lab(t) for retrograde streams with v_gal = 544 and 600 km/s; (b) rate/annual mean for the SHM (δ = 350) and pure streams (544 km/s at δ = 350/380; 600 km/s at 400).

## 8. Failed or abandoned approaches
1. First run: a keyword `delta_N_1e-3_of_300` was an invalid identifier (renamed).
2. The 3-D shell quadrature copied from P018 (48 Gauss–Legendre nodes in cos θ × 64 azimuths) cannot resolve a σ = 20 km/s stream at 800 km/s (angular size 1.4°): the untruncated numerical η₀ came out 4.4–6.7 % high and the truncated counts were unreliable. Replaced by the exact axisymmetric formula (valid for streams anti-parallel to **v**_obs), which reproduces the analytic stream to 5 × 10⁻⁷.
3. The Gaussian (untruncated) treatment of bound streams overstates the rate above δ ≈ 380 keV by up to ×60 at δ = 400 keV (Sec. 6); the untruncated numbers are kept for unbound streams and as an upper envelope, and flagged wherever they matter.

## 9. Discussion
- The elastic answer is structural: for a flat-in-E_R spectrum the low/high ratio is a property of the nuclear response, and WimPyDD's shell-model M response makes it 3.3× worse than Helm because the node lies inside the 200–270 keV window (P017). Streams can only reduce N_lo by the factor ≈ 2.2 that the SHM's own velocity weighting supplied. P003's operator ranking is therefore halo-independent to within that factor; O6 (q⁴) and L10-like couplings remain the only elastic survivors, with or without streams.
- For inelastic DM the stream question splits cleanly at the escape speed. Below δ_max, streams are a rate amplifier of order 10–10³ concentrated in the last 40 keV, which broadens P018's "astrophysics dominates" conclusion: even a 0.1 % retrograde stream near v_esc changes the δ = 380 keV rate by ×7–60. Above ≈ 400 keV, only unbound matter contributes; the fractional density of unbound dark matter near the Sun is unmeasured and presumably ≪ 1 %, but only 10⁻⁴–10⁻⁵ would be needed at 850–900 km/s for the Higgsino cross-section at δ = 400 keV.
- The seasonal signature distinguishes the scenarios: SHM-tail signals at δ ≥ 350 keV are strongly June-peaked (P006/P020), retrograde-stream signals less so (peak/mean 1.5–2.2), and perpendicular streams peak at other dates. With the 2–5 events P020 forecasts for the post-2024 exposure this is not yet testable.
- Overlap with P055/P097: P055 will address specific stream candidates in more detail; our Table 5 gives the kinematic ceiling each must obey (δ_max ≤ 194 keV for S1 with mean speed). A co-rotating dark disk (P097) is prograde and slow (v_lab ≲ 100 km/s) and is irrelevant for δ ≥ 300 keV by the same kinematics (E_max(100 km/s) = 22 keV elastic).

## 10. References
LZ Collaboration, arXiv:2609.02823 (2026). K. Freese, P. Gondolo, H. J. Newberg, M. Lewis, PRL 92, 111301 (2004). C. Savage, K. Freese, P. Gondolo, PRD 74, 043531 (2006). G. C. Myeong, N. W. Evans, V. Belokurov, J. L. Sanders, S. E. Koposov, ApJL 863, L28 (2018). C. A. J. O'Hare, C. McCabe, N. W. Evans, G. Myeong, V. Belokurov, PRD 98, 103006 (2018). N. W. Evans, C. A. J. O'Hare, C. McCabe, PRD 99, 023012 (2019). D. Baxter et al., EPJC 81, 907 (2021). C. McCabe, PRD 82, 023530 (2010). Corpus: P002, P003, P006, P007, P009, P017, P018, P020.

## 11. Tools and provenance (mirrors provenance/P030.json)
- Agent tools: Read ×15 (PAPER_GUIDE, dossier, ledger, P002/P003/P006/P007/P018 papers, lzcommon.py, P018_halo_tail.py, P007_higgsino_inelastic.py, P018.json, P030 figures ×4 views); Bash ×16 (directory/version listing; P018 validation + WimPyDD signatures; kernel timing test; script runs ×5 incl. one syntax failure; bin-convention check; elastic-table extraction; number collection; word counts ×5); Write ×5 (script, details.md, P030.json, P030.md ×2); Edit ×19 (script fixes/additions ×6, paper trims ×11, tool-count updates ×2); Skill ×1 (dataviz).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (special.erf/erfc); pandas 3.0.5; matplotlib 3.11.2; WimPyDD 2.0.4 (diff_rate per-stream kernels with sum_over_streams=False, eft_hamiltonian via lzcommon, streamed_halo_function via lz.wd_halo for validation, v_earth_sun); common/lzcommon.py (eta0, vmin_kms, E_R_range_keV, delta_max_kev, helm_F2, XE_ISOTOPES, wd, wd_halo, wd_hamiltonian, LZ constants); analytic derivations (stream η, axisymmetric truncated integral).
- Script: output/code/P030_streams.py — `.venv/bin/python output/code/P030_streams.py` (cwd = root, 147 s).
- Local inputs: PAPER_GUIDE.md; 00_evidence_dossier.md; results_ledger.csv; papers P002/P003/P006/P007/P018; code/common/lzcommon.py; code/P018_halo_tail.py; code/P007_higgsino_inelastic.py; provenance/P018.json; work/P018/validation.json; environment/ENVIRONMENT_versions.txt.
- Recalled knowledge: 12 items (Sec. 5 table and Sec. 6; listed with reliabilities in P030.json).
- Datasets: none. Data requests: none. WimPyDD-generated files: none (diff_rate does not write files).
