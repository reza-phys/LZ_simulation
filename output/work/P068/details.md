# P068 — details: a joint astrophysical uncertainty band on everything inferred from the LZ event

Simulated date 2026-09-12 · Category HALO · astro-ph.GA (cross-list hep-ph) · author profile: astroparticle systematics working group.
Script: `output/code/P068_astro_band.py` (run from the simulation root with `.venv/bin/python`; ~570 s on first run, 2 s from cache).
All tables in `output/work/P068/`, figures in `output/work/P068/figures/`.

## 1. Motivation and framework

The corpus has quantified the halo dependence of the LZ 248 keV event one parameter at a time: P018 varied v_esc, v₀, ρ₀ and the
tail shape separately (δ_max slope 0.82 keV per km/s; Gaia-range rate spread 0.7/1.5/3.6 dex at δ = 300/350/380 keV; Higgsino window
shifts of −17…+11 keV), P035 found the frame convention (WimPyDD's single-day Sun frame vs the true annual mean) changes the δ = 300/350/380 keV
rate by ×1.08/1.7/10, P030 treated streams, and P043 propagated the *detector* nuisances (g₁, g₂, light-yield scale) through the same
inferred quantities (E ± 9.6 keV; band ± 0.54σ; d₁₀ ± 1.6 %; κ̂(380) ×/÷ 1.14 and 1.30; δ_max ± 3 keV). What is missing is a *joint*
propagation: a prior over all halo parameters at once, with their correlations, giving marginal 68/95 % bands and telling us which input
dominates which quantity, so that the corpus can quote "astro-inflated" numbers and compare them with the detector budget.

Quantities propagated (all at m_χ = 1000 GeV, 2.84 t·yr, LZ efficiency as in P007/P018: plateau 0.96, erf edges to 50 % at 5.4 and
269.9 keV with σ = 2.5 and 11.5 keV):

* (a) the elastic L10 coupling d₁₀ giving one event (P012: 0.28 in LZ's normalisation, provisional DR-001);
* (b) the inelastic isoscalar O₁ coupling κ̂(δ) = (c₁ˢm_v²)² giving one event at δ = 300/350/366/380 keV (P021's κ), and the pure-Higgsino
  splitting δ_H(N = 1) at which P007's gauge-fixed cross-section gives one event;
* (c) δ_max(248 keV) on 16 June;
* (d) the annual-modulation fundamental amplitude a₁(δ) and phase (P034's definition) for δ = 300/350/366/380 keV and for L10;
* (e) the efficiency-weighted percentile of 248 keV in the δ = 300 and 350 keV spectra (P002: 99.5 / 98.1 on 16 June).

### 1.1 Joint prior

| input | prior | source (recalled unless stated) |
|---|---|---|
| ρ₀ | log-uniform 0.3–0.5 GeV cm⁻³ | Gaia-era local density range (likely) |
| v₀ | uniform 220–250 km/s | (likely) |
| v_esc | uniform 500–600 km/s (set C: 528–580 "Gaia-central") | RAVE/Gaia escape-speed determinations 500–600, central values 528–580 (likely) |
| ΔV_⊙ | Gaussian σ = 5 km/s on the V (rotation) component of (11.1, 12.2, 7.3) km/s, truncated at ±2.5σ | Schönrich-type systematic on V_⊙ (likely). U and W perturbations change |v_obs| only at second order (d|v|/dU = 11/250) and are not varied |
| k | uniform 0–2 | Lisanti–Strigari–Wacker–Wechsler tail index (their preferred 1.5–3.5 from simulations; likely); k → 0 is the sharp Maxwellian |
| frame | discrete equiprobable {Sun frame, 16 June, 12-day annual mean} | the three conventions used in the corpus (P035) |
| correlation | Gaussian copula ρ(v₀, v_esc) = +0.5 (sets A, C); 0 (set B) | assumption: in Galactic mass models a larger circular speed implies a deeper potential and a larger v_esc (qualitative, likely). Realised Spearman 0.47/0.03/0.52 |

Three prior sets: **A** (main; N = 400), **B** (independent v₀–v_esc; N = 200), **C** (Gaia-central v_esc; N = 200). Latin-hypercube
sampling (`scipy.stats.qmc.LatinHypercube`, seeds 68/168/268); the copula is applied to the two LHS columns via normal scores
(Iman–Conover-like), which preserves the uniform marginals and approximately the stratification. Frame draws: 134/133/133 (A).

The frame is not a physical parameter — the annual mean is the right description of a rate integrated over LZ's exposure, June of the event-date
likelihood, the Sun frame of nothing (P035) — so results are reported both **joint** (frame drawn) and **conditional on each frame**; the
frame's share of the joint variance is a measure of how much of the corpus spread is convention.

## 2. Method

### 2.1 Earth-frame halo functions (η)

P018's construction: η(v_min) = ∫_{v_min}^∞ S(v) dv with S(v) = v ∫dΩ f_gal(v n + v_obs), integrated over the cap |v n + v_obs| ≤ v_esc
(polar axis along −v_obs, c_min = (v² + v_obs² − v_esc²)/(2 v v_obs)). Every model here is **isotropic** in the Galactic frame, so f depends on
|u|² = v² + v_obs² − 2 v v_obs c only, the φ integral is 2π, and S(v) = 2π v ∫_{c_min}^{1} f(v² + v_obs² − 2 v v_obs c) dc — a one-dimensional
Gauss–Legendre integral (64 nodes) instead of P018's 48 × 64 angular grid. Δη_i on the WimPyDD grid (upper boundaries 0–950 km/s, 0.5 km/s) by
Simpson with mid-points; normalisation 4π ∫₀^{v_esc} u² f du (600 nodes). Observer velocity from WimPyDD's `v_earth_sun` with the P018 phase
convention (day 167 = 16 June; 12 days at 15 + 365.25 j/12).

Tail form (Lisanti et al. 2011): f(u) ∝ [exp((v_esc² − u²)/(k v₀²)) − 1]^k Θ(v_esc − u). Written as
exp(−u²/v₀²) [1 − exp(−(v_esc² − u²)/(k v₀²))]^k (the constant exp(v_esc²/v₀²) cancels in the normalisation) it is overflow-safe for small k and
shows that the bulk stays a Maxwellian of dispersion v₀ for every k; k → 0 gives the sharp Maxwellian, k = 1 the King model. **Note on P018:** P018's
"k" form [exp(−u²/v₀²) − exp(−v_esc²/v₀²)]^k coincides with Lisanti's at k = 1 but for k ≥ 2 also shrinks the bulk dispersion to v₀/√k, which is why its
k = 2, 3 tails were so extreme (η/η_SHM(700 km/s) = 0.018, 5 × 10⁻⁴). With the Lisanti form: η/η_SHM(700 km/s, 16 June) = 0.841/0.557/0.336/0.197
for k = 0.5/1/1.5/2 (k = 1 identical to P018's 0.56).

### 2.2 Kernels

WimPyDD `diff_rate(..., sum_over_streams=False)` with unit weights gives K(E, v_i) such that dR/dE = K @ Δη (P018 convention), computed once per
(Hamiltonian, δ) and cached under `kernel_cache/` (45 files): O₁ isoscalar LZ unit coupling (WimPyDD c⁰ = 2/m_v², P003) at δ = 300/350/366/380 keV
(E step 2 keV, 1–330 keV); P007 Higgsino (c_p = (G_F/√2)(1 − 4 sin²θ_W), c_n = −G_F/√2; WimPyDD (c_p + c_n, c_p − c_n)) at δ = 250–445 keV in
5 keV steps (E step 4 keV); L10 = 4[(q²/m_N²) O₄ − O₆] with WimPyDD c₄ = 8 q²/(m_v² m_N²), c₆ = −8/m_v² (P012; coefficient functions built as
closures per the P031 pitfall), elastic. In-ROI count N = 2.84 t·yr × ∫ ε(E) (K @ Δη) dE. ρ₀ enters as N ∝ ρ₀/0.3.

Outputs per sample and frame: κ̂(δ) = 1/N_iso(δ) (κ̂ = ∞ when N = 0, i.e. v_max below threshold: no coupling gives one event);
d₁₀ = [F_LZ N_L10]^{−1/2} with F_LZ = 12.842/3.3295 = 3.857 (P012's LZ-normalisation factor, σ_hi = 11.5 row; provisional DR-001);
δ_H(N = 1) by log-linear interpolation of N_hig(δ) (the 5 keV grid is adequate because N falls ≈ ×5 per 10 keV, so log N is nearly linear);
δ_max(248 keV, 1 TeV) = `lz.delta_max_kev` at v_max = |v_obs(16 June)| + v_esc (independent of k, ρ₀ and frame);
a₁ and phase from the twelve daily in-ROI counts R_j: T p(t) = 1 + Σ a_k cos(kω(t − t_k)), a₁ = |(2/12) Σ_j (R_j/⟨R⟩) e^{−iωt_j}|, t₁ = −arg(·)/ω
(a₁ is undefined if all twelve rates vanish); pct(248 | δ) = 100 × ∫₀^{248} ε dR/dE dE / ∫ ε dR/dE dE.

### 2.3 Statistics

Median, 16/84 and 2.5/97.5 percentiles over finite samples, plus the undefined fraction; for κ̂ also the quantiles with undefined samples counted as
+∞ (`inverted_cdf`). Spearman correlations with each input. First-order variance indices S_i = Var(E[Y | X_i ∈ bin])/Var(Y) with 8 quantile bins
(3 groups for the frame): a Sobol-like main effect; its small-sample floor is ≈ (n_bins − 1)/N ≈ 0.02; under the v₀–v_esc correlation the indices
are not orthogonal and can sum to more than one (set B, independent, is the clean reference). Detector comparison: P043's 1σ half-widths in the same
transformed units (log₁₀ κ̂, log₁₀ d₁₀, keV, percentile points).

## 3. Validation (all at the Baxter point ρ₀ = 0.3, v₀ = 238, v_esc = 544 km/s, k = 0, ΔV = 0; `P068_results.json → validation`)

| check | result | reference |
|---|---|---|
| η(k = 0) vs analytic `lz.eta0` (v ≤ 780 km/s) | max rel. dev. 2.6 × 10⁻⁵ (Sun), 3.0 × 10⁻⁶ (June) | P018: 3 × 10⁻⁵ |
| η vs WimPyDD `wd_halo(167)` at 0/400/600/700/750/780/800 km/s | 1.000/0.998/0.996/0.994/0.990/0.982/0.950 | WimPyDD's bin-averaged η is 2–5 % high near v_max (P018) |
| v_E: Sun 250.55, 16 June 265.98 km/s (vector) | cosine model 265.1 (lzcommon) | P006/P018 |
| N_iso(1 TeV) June 300/350/380: 1.917e4 / 613.7 / 2.435; annual 1.335e4 / 253.4 / 0.618 | P018: 1.917e4 / 613.6 / 2.428; 1.335e4 / 253.3 / 0.6155 | ≤ 0.4 % |
| N_iso(366): June 65.05, annual 20.20, Sun 4.372 | — | new |
| Higgsino N(1 TeV) annual 300/350/370/380: 784 / 10.9 / 0.485 / 0.097 | P007: 790 / 11.1 / 0.49 / 0.098 | 1–2 % (P007's table is the annual mean; the pure Sun frame gives 721 / 5.94 / 0.197 / 0.023) |
| δ_H(N = 1): annual 365.7, June 372.1, Sun 360.3 keV | P018 annual 365.7; P021 365.5 | exact |
| L10 N(d₁₀ = 1) WimPyDD, Sun frame | 3.3238 (annual 3.3305, June 3.3975) | P012: 3.3295 (σ_hi 11.5) → d₁₀ = 0.2793 |
| κ̂ annual 300/350/380: 7.49e-5 / 3.95e-3 / 1.62 | P021: 7.4e-5 / 3.9e-3 / 2.19 | P021 smears with 11 keV resolution, which matters only at 380 keV (×0.74) |
| pct(248) June 300/350: 99.52 / 98.09 | P002: 99.5 / 98.1 | |
| a₁ 300/350/366/380: 0.430 / 1.185 / 1.533 / 1.683; L10 0.0213; phase 151.9 d | P034: 0.428 / 1.179 / 1.521 / 1.656 (36-day spline), phase 151.9 | ≤ 1.6 % with 12 days |
| δ_max(248, 1 TeV) June 387.3 keV (Sun 374.6) | P002/P018/P043 386.6 (cosine-model v_E) | 0.7 keV from the v_E vector |
| Lisanti k = 1 η ratio at 700 km/s | 0.557 | P018 King k = 1: 0.56 |

## 4. Results

### 4.1 Marginal bands (set A, N = 400; `marginals_and_sensitivities.csv`, `P068_results.json → astro_inflated`)

Reference values (Baxter halo) in brackets: annual mean for κ̂, δ_H, pct; June for δ_max; d₁₀ 0.279.

**Frame drawn from the prior (A_joint)**

| quantity | median | 68 % | 95 % | undefined |
|---|---|---|---|---|
| d₁₀ [0.279] | 0.246 | 0.226–0.269 | 0.218–0.281 | 0 |
| κ̂(300) [7.5e-5] | 1.46e-4 | 4.4e-5 – 8.5e-4 | 2.1e-5 – 5.4e-3 | 0 |
| κ̂(350) [4.0e-3] | 0.0145 (0.017 incl. ∞) | 8.4e-4 – 4.6 (12 incl. ∞) | 2.0e-4 – 7.5e3 (5.8e8 incl. ∞) | 2.5 % |
| κ̂(366) [0.050] | 0.11 (0.43 incl. ∞) | 3.2e-3 – 153 (1.9e5 incl. ∞) | 6.1e-4 – 3.0e5 (∞ incl. ∞) | 13.3 % |
| κ̂(380) [1.6] | 0.64 (21 incl. ∞) | 0.014 – 1.2e3 (∞ incl. ∞) | 1.8e-3 – 3.9e7 (∞) | 26.3 % |
| δ_H(N = 1) [365.7 keV] | 355.4 | 328.6–384.5 | 312.6–405.4 | 0 |
| δ_max(June) [387.3 keV] | 390.4 | 358.1–421.1 | 340.8–438.8 | 0 |
| a₁(300) [0.430] | 0.553 | 0.376–0.877 | 0.291–1.21 | 0 |
| a₁(350) [1.185] | 1.27 | 0.78–1.74 | 0.55–1.90 | 0.7 % |
| a₁(366) [1.533] | 1.49 | 1.00–1.79 | 0.71–1.93 | 8.5 % |
| a₁(380) [1.683] | 1.60 | 1.20–1.81 | 0.87–1.93 | 21.8 % |
| a₁(L10) [0.0213] | 0.0248 | 0.0202–0.0296 | 0.0175–0.0344 | 0 |
| phase t₁ (all δ, L10) | 151.8 d | 151.6–152.1 | | |
| pct(248 | 300) [99.49] | 99.42 | 99.16–99.53 | 98.77–99.58 | 0 |
| pct(248 | 350) [97.55] | 96.7 | 71.6–98.7 | 0–99.1 | 2.5 % |

"Undefined" = the halo's v_max lies below the threshold speed for a 248 keV recoil at that δ over the whole ROI, so the rate is zero (κ̂ = ∞, a₁
undefined). Zero-rate fractions per frame: δ = 350: 7.8 % (Sun), 0.75 % (June, annual); 366: 22 / 8.5 / 8.5 %; 380: 34 / 22 / 22 %. pct(248 | 350) = 0
occurs when v_max barely exceeds threshold: the kinematic window collapses to E* = δμ/m_N ≈ 313 keV and the entire (tiny) spectrum lies above 248 keV.

**Annual-mean frame only (A_annual)**: d₁₀ 0.247 (0.226–0.270; 0.219–0.282); κ̂(300) 1.48e-4 (4.9e-5–8.8e-4; 2.2e-5–5.2e-3); κ̂(350) 0.017 (1.0e-3–7.7;
2.2e-4–8.2e3), 0.75 % undefined; κ̂(366) 0.115 (4.5e-3–330; 6.4e-4–2.8e5, ∞ incl.), 8.5 % undefined; κ̂(380) 0.72 (0.017–1.3e3), 22 % undefined;
δ_H 355.9 (330.5–381.8; 314.2–404.7); pct(248 | 300) 99.41 (99.17–99.53); pct(248 | 350) 96.9 (76.8–98.6).
**16 June (A_june)**: κ̂(300) 9.5e-5 (3.5e-5–4.6e-4); κ̂(366) 0.039 (2.0e-3–69); δ_H 362.3 (336.5–388.7; 320.2–412.0); pct(248 | 300) 99.46 (99.26–99.55).
**Sun frame (A_sun)**: κ̂(300) 1.7e-4 (5.2e-5–1.1e-3); κ̂(366) 0.13 (4.5e-3–197), 22 % undefined; δ_H 350.8 (325.0–376.9; 308.8–399.9).

**Gaia-central v_esc = 528–580 km/s, annual (C_annual_gaia, N = 200)**: d₁₀ 0.246 (0.226–0.267); κ̂(300) 1.36e-4 (5.1e-5–4.4e-4; 3.2e-5–1.6e-3);
κ̂(350) 9.3e-3 (1.4e-3–0.29; 4.3e-4–9.0); κ̂(366) 0.090 (7.2e-3–23; 1.8e-3–2.6e3), none undefined; κ̂(380) 1.4 (0.054–2.6e3), 3.5 % undefined;
δ_H 361.6 (340.5–379.0; 326.3–391.9); δ_max 395.0 (373.3–412.4; 360.9–421.7); a₁(300) 0.514 (0.397–0.704); pct(248 | 300) 99.44 (99.30–99.51);
pct(248 | 350) 97.5 (92.6–98.5).

**Set B (independent v₀, v_esc; N = 200)**: bands within sampling noise of set A (e.g. δ_H 356 (333–380), κ̂(366) 68 % width 3.8 dex vs 4.7 in A,
δ_max 390 (360–419)); the correlation widens the tails slightly because v₀ and v_esc then push v_max in the same direction.

### 4.2 Sensitivities (first-order indices S_i; `marginals_and_sensitivities.csv`; Fig. 2)

Set A joint (frame drawn) / set B independent:

| output | ρ₀ | v₀ | v_esc | ΔV_⊙ | k | frame | dominant |
|---|---|---|---|---|---|---|---|
| log κ̂(300) | 0.04 / 0.08 | 0.40 / 0.27 | 0.57 / 0.40 | 0.04 / 0.08 | 0.21 / 0.19 | 0.03 / 0.04 | v_esc |
| log κ̂(350) | 0.02 / 0.02 | 0.33 / 0.15 | 0.69 / 0.61 | 0.04 / 0.05 | 0.07 / 0.08 | 0.01 / 0.01 | v_esc |
| log κ̂(366) | 0.03 / 0.03 | 0.27 / 0.08 | 0.70 / 0.61 | 0.02 / 0.06 | 0.09 / 0.07 | 0.01 / 0.02 | v_esc |
| log κ̂(380) | 0.02 / 0.10 | 0.19 / 0.07 | 0.67 / 0.52 | 0.03 / 0.04 | 0.11 / 0.10 | 0.03 / 0.00 | v_esc |
| δ_H(N = 1) | 0.03 / 0.05 | 0.41 / 0.19 | 0.79 / 0.68 | 0.04 / 0.08 | 0.10 / 0.08 | 0.04 / 0.03 | v_esc |
| δ_max(June) | 0.03 / 0.04 | 0.42 / 0.14 | 0.91 / 0.89 | 0.04 / 0.06 | 0.02 / 0.02 | — | v_esc |
| log d₁₀ | 0.95 / 0.95 | 0.05 / 0.06 | 0.02 / 0.05 | 0.05 / 0.03 | 0.01 / 0.04 | 0.01 / 0.00 | ρ₀ |
| a₁(300) | 0.03 / 0.05 | 0.35 / 0.14 | 0.79 / 0.73 | 0.04 / 0.08 | 0.12 / 0.09 | — | v_esc |
| a₁(350/366/380) | ≤ 0.03 | 0.38/0.32/0.27 | 0.89/0.88/0.86 | ≤ 0.04 | ≤ 0.04 | — | v_esc |
| a₁(L10) | 0.02 / 0.09 | 0.69 / 0.68 | 0.34 / 0.09 | 0.04 / 0.05 | 0.20 / 0.22 | — | v₀ |
| pct(248 | 300) | 0.03 / 0.06 | 0.32 / 0.14 | 0.69 / 0.58 | 0.04 / 0.08 | 0.11 / 0.07 | 0.02 / 0.05 | v_esc |
| pct(248 | 350) | 0.01 / 0.02 | 0.21 / 0.08 | 0.58 / 0.57 | 0.03 / 0.05 | 0.01 / 0.04 | 0.01 / 0.01 | v_esc |

Spearman correlations with the inputs (set A joint): κ̂(δ) with v_esc −0.76/−0.88/−0.88/−0.87 (δ = 300…380), with v₀ −0.66…−0.49, with k +0.45…+0.30;
δ_H with v_esc +0.90; δ_max with v_esc +0.97, v₀ +0.64; d₁₀ with ρ₀ −0.99; a₁(350) with v_esc −0.95; a₁(L10) with v₀ −0.85; pct(248 | 300) with v_esc +0.88.
Frame: |ρ| ≤ 0.10 everywhere. Output–output (set A, `spearman_matrix_setA.csv`): κ̂(366)–δ_H −0.993, κ̂(366)–pct(248 | 300) −0.982, δ_max–a₁(350) −0.987,
δ_H–pct +0.991, d₁₀ with everything |ρ| ≤ 0.17. In words: all inelastic quantities are one number — the high-velocity reach v_max — in different units; d₁₀
is a different number (ρ₀).

d₁₀ at fixed ρ₀ (log₁₀ d₁₀ + ½ log₁₀(ρ₀/0.3)): standard deviation 0.0057 dex, 68 % 0.276–0.284, i.e. ±1.3 % from all velocity-moment inputs together;
the joint band is the ρ₀ prior: d₁₀ = 0.279 (ρ₀/0.3)^{−1/2}.

Frame convention at the Baxter point (N_iso ratios): June/annual = 1.44/2.42/3.22/3.94 and Sun/annual = 0.926/0.577/0.216/0.092 at δ = 300/350/366/380
(annual/Sun = 1.08/1.73/4.6/10.9; P035: 1.08/1.7/10 at 300/350/380). Yet the frame's share of the joint variance is 0.6–3.6 % for every quantity, because
the v_esc range moves the same rates by orders of magnitude.

Derived probabilities (set A joint / set C): P(δ_H ≤ 350 keV) = 0.43 / 0.33 — P007's "excluded at every tabulated δ ≤ 350 keV" holds for 57 % (67 %) of the
prior; P(δ_H ≥ 380) = 0.21 / 0.13; P(358 ≤ δ_H ≤ 380, annual) = 0.30 / 0.43. P(δ_max(June) < 366) = 0.26 / 0.075 (a Higgsino at P007's best δ = 366 keV
is kinematically forbidden for a quarter of the prior); P(δ_max < 380) = 0.40 / 0.26; P(δ_max > 400) = 0.39 / 0.41. P(pct(248 | 300) > 99) = 0.93 / 0.995.
P(a₁(300) > 1) = 0.085 / 0.005; the 68 % ratio of a₁(300) is 2.33, so P034's N_3σ ∝ a₁⁻² varies by ×5.4 (×3.2 in set C) over the band (274 t·yr → ~120–640).
Sharp-tail subset k < 0.3 (n = 60): δ_H annual 369 (342–399), κ̂(366) annual 68 % 9.5e-4–22: the tail index narrows nothing by itself; v_esc does.

### 4.3 Astrophysics versus detector (P043; `astro_vs_detector.csv`; Fig. 3)

| quantity | astro 68 % half-width (A joint) | astro 95 % | detector 1σ (P043) | ratio 68 %/det |
|---|---|---|---|---|
| log₁₀ κ̂(300) | 0.64 dex | 1.20 | 0.004 (κ ×/÷ 1.01) | 160 |
| log₁₀ κ̂(350) | 1.87 | 3.78 | 0.006 | 296 |
| log₁₀ κ̂(366) | 2.34 | 4.34 | 0.029 (interpolated ×1.07) | 80 |
| log₁₀ κ̂(380) | 2.47 (∞ incl. undefined) | 5.2 | 0.127 (×1.14 ⊕ ×1.30) | 19 |
| δ_H(N = 1) | 28.0 keV | 46.4 | 0 (rate-set) | ∞ |
| δ_max(June) | 31.5 keV | 49.0 | 3.0 keV (2.1 ⊕ 2.1) | 10.6 |
| log₁₀ d₁₀ | 0.038 dex (8.7 %) | 0.055 | 0.0069 (1.6 %) | 5.5 (≈ 0.8 at fixed ρ₀) |
| pct(248 | 300) | 0.18 points | 0.41 | 0.56 (E_obs ± 9.6 keV → 98.69–99.81) | 0.33 |
| pct(248 | 350) | 13.6 points | 49.6 | 1.93 (95.27–99.12) | 7.0 |

The detector percentile shifts use the Baxter June spectrum with E_obs = 248 ± 9.6 keV (P043 systematic; with stat ⊕ sys = 13.7 keV: 98.08–99.86 and
93.25–99.28). Reading: astrophysics dominates every inelastic coupling, the kinematic edge and the Higgsino window by one to two orders of magnitude; for
d₁₀ the ρ₀ prior alone is ×5.5 the detector budget while the velocity-moment part equals it; the position of 248 keV in the δ = 300 keV spectrum is the
single quantity where the energy scale matters more than the halo (and it is robust either way: > 98.8th percentile for all halos and all E_obs within ±1σ).
The band position (P043 ± 0.54σ) has no astrophysical counterpart.

### 4.4 The recommended "astro-inflated" table (1 TeV)

| corpus number | source | with the joint prior (68 %; 95 %) | Gaia-central v_esc, annual |
|---|---|---|---|
| d₁₀ = 0.28 | P012 (ρ₀ = 0.3) | 0.246 (+0.023/−0.020; +0.035/−0.028) ≡ 0.279 (ρ₀/0.3)^{−1/2} ± 1.3 % | 0.246 (+0.021/−0.020) |
| κ̂(300) = 7.4 × 10⁻⁵ | P021 | 1.5 × 10⁻⁴ (4.4 × 10⁻⁵ – 8.5 × 10⁻⁴; 2.1 × 10⁻⁵ – 5.4 × 10⁻³) | 1.4 × 10⁻⁴ (5.1 × 10⁻⁵ – 4.4 × 10⁻⁴) |
| κ̂(366) = 0.050 | this work, annual | 0.11 (3 × 10⁻³ – 150; 6 × 10⁻⁴ – ∞), 13 % of halos give no event | 0.090 (7 × 10⁻³ – 23; 2 × 10⁻³ – 2.6 × 10³) |
| κ̂(380) = 2.2 | P021 | 0.64 (0.014 – 1.2 × 10³; upper edges ∞), 26 % give no event | 1.4 (0.054 – 2.6 × 10³) |
| δ_max = 387 keV | P002/P018 | 390 (+31/−32; +49/−49) | 395 (+17/−22; +27/−34) |
| Higgsino δ(N = 1) = 366 keV | P007/P018/P021 | 355 (+30/−26; +50/−42) | 362 (+17/−22; +30/−36) |
| a₁(300/350/366/380) = 0.43/1.18/1.52/1.66 | P034 | 0.55/1.27/1.49/1.60 (0.38–0.88 / 0.78–1.74 / 1.00–1.79 / 1.20–1.81) | 0.51/1.16/1.48/1.64 |
| pct(248 | 300) = 99.5 | P002 | 99.4 (99.2–99.5; 98.8–99.6) | 99.4 (99.3–99.5) |
| pct(248 | 350) = 98.1 | P002 | 96.7 (72–98.7; 0–99.1) | 97.5 (92.6–98.5) |

## 5. Figures

* `figures/P068_fig1_corner.png` — corner plot (set A) of log₁₀ κ̂(366), δ_H(N = 1), δ_max(June), log₁₀ d₁₀, a₁(350), pct(248 | 300), coloured by the
  drawn frame (blue Sun, orange June, aqua annual). The five inelastic quantities lie on one curve (|ρ| ≥ 0.92); d₁₀ is uncorrelated with them; the three
  frame colours overlap almost completely.
* `figures/P068_fig2_variance_decomposition.png` — stacked first-order indices per output, (a) frame drawn, (b) annual only. v_esc (aqua) dominates all
  inelastic rows, ρ₀ (blue) the d₁₀ row, v₀ (orange) the elastic modulation a₁(L10). Sums above one reflect the v₀–v_esc correlation.
* `figures/P068_fig3_astro_vs_detector.png` — astro 68 %/95 % half-widths in units of P043's 1σ (log axis); only pct(248 | 300) falls below one.

## 6. Failed approaches and fixes

1. The first full run used P018's two-dimensional angular cap integral for every η (0.09 s per evaluation on a loaded machine) and exceeded the 600 s
   foreground budget before any kernel was computed; it was stopped. The isotropic one-dimensional reduction (§2.1), per-kernel and per-set caching made
   the whole run 567 s (kernels 510 s, halo library 55 s) and re-runs 2 s. The 60 MB halo cache was deleted after the run (regenerated in ~60 s).
2. `timeout` is not available on macOS; the tool's own limit was used instead.
3. The modulation phase came out at day 213 in the first pass: the sign convention c = a₁ e^{−iωt₁} requires t₁ = −arg(c)/ω; corrected to 151.8 d
   (P034: 151.9). a₁ was unaffected.
4. Quantiles were first computed over finite samples only, hiding that κ̂ = ∞ for a sizeable fraction of halos at δ ≥ 350 keV; both versions are now reported.
5. The assignment cites P055 for streams; no such paper exists yet (papers run to P053), so streams are represented by P030 only.

## 7. Extended discussion

* **One number.** Within an isotropic-halo prior the inelastic quantities are all monotonic functions of v_max = |v_obs| + v_esc (and, at fixed v_max, weakly
  of the tail index): κ̂(366)–δ_H correlate at −0.993. A future measurement of any one of them (e.g. a second event fixing δ via its energy, P035) would fix
  the others only if v_esc were known; conversely a δ-window quoted without a v_esc is a statement about the halo.
* **What the frame debate was worth.** P035's factors ×1.08/1.7/10 are real but contribute ≤ 4 % of the joint variance: the corpus's "Sun-frame vs annual"
  discrepancies are a small part of the astrophysical budget for δ ≥ 350 keV. The annual mean remains the right convention for rate-based inferences.
* **P007's window.** The Higgsino δ(N = 1) = 366 keV moves to 355 (329–385) keV; 43 % of the prior puts it at or below LZ's 350 keV grid point, where LZ's own
  (Baxter-halo) two-sided intervals excluded it. A quarter of the prior makes δ = 366 keV kinematically unreachable on 16 June. Within Gaia-central v_esc the
  window is 362 (340–379) keV and P007's exclusion survives in two thirds of the prior.
* **P034's exposures.** N_3σ ∝ a₁⁻²: at δ = 300 keV the 68 % band of a₁ (0.38–0.88) spans ×5.4 in exposure; at δ ≥ 350 keV a₁ saturates near 1.5–1.9 and the
  modulation test is more robust to the halo than the rate is.
* **d₁₀.** The only astrophysical uncertainty that matters is ρ₀; quoting d₁₀ at ρ₀ = 0.3 with the scaling (ρ₀/0.3)^{−1/2} is exact to 1.3 %.
* **Limits of the prior.** Envelope-type inputs (uniform 500–600 km/s) are not likelihoods; a Gaussian v_esc = 540 ± 25 km/s would give bands close to set C.
  Anisotropic components (P018's Sausage, −20 % in the tail) and streams (P030) are outside this isotropic family; P018's k ≥ 2 tails are a different (bulk-
  shrinking) form and are not reproduced here. Energy resolution is omitted (P021's κ̂(380) is ×1.35 ours); O₁ shapes only (P018: ratios are Hamiltonian-
  insensitive to 0.02 dex); a single mass.

## 8. References

LZ Collaboration, arXiv:2609.02823 (2026). D. Baxter et al., EPJC 81, 907 (2021). M. Lisanti, L. E. Strigari, J. G. Wacker, R. H. Wechsler, PRD 83, 023519
(2011). N. W. Evans, C. A. J. O'Hare, C. McCabe, PRD 99, 023012 (2019). T. Piffl et al., A&A 562, A91 (2014). M. D. McKay, R. J. Beckman, W. J. Conover,
Technometrics 21, 239 (1979). I. Jeong, S. Kang, S. Scopel, G. Tomar, CPC 276, 108342 (2022) (WimPyDD). Corpus: dossier 00, P002, P003, P007, P012, P018,
P021, P030, P034, P035, P043.

## 9. Tools and provenance (mirrors `output/provenance/P068.json`)

* Agent tools: Read (PAPER_GUIDE; dossier; ledger first page; P018/P021/P043/P035/P034/P012/P003/P007/P030 papers; P055 (absent); P018_halo_tail.py;
  lzcommon.py §§2–4, 8; three figures), Bash (ledger/grep lookups, P018/P021/P043/P012 work tables, timing tests, script runs, result prints, cache removal),
  Write (script, details, provenance, paper), Edit (script fixes), Skill dataviz, ToolSearch, TaskStop.
* Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (stats.qmc.LatinHypercube, stats.norm, stats.spearmanr, stats.rankdata, special.erf/erfc); pandas 3.0.5;
  matplotlib 3.11.2; WimPyDD 2.0.4 (eft_hamiltonian incl. q-dependent coefficient, diff_rate with sum_over_streams=False, streamed_halo_function via
  lz.wd_halo, v_earth_sun); common/lzcommon.py (LZ, V0_KMS, VESC_KMS, V_SUN_PEC, eta0, delta_max_kev, E_R_range_keV, XE_ISOTOPES, M_V_GEV, M_NUCLEON_GEV,
  wd, wd_halo, wd_hamiltonian, wd_c_from_anand).
* Local inputs: LZ paper numbers via lzcommon; P018 (η construction, tables), P012 (L10 Hamiltonian, F_LZ = 12.842/3.3295, d₁₀ = 0.279), P007 (Higgsino
  couplings, N table), P021 (κ̂ values), P002 (percentiles), P034 (a₁ definition and values), P043 (detector 1σ budget), P035 (frame factors), P030 (streams
  context).
* Recalled knowledge: 12 items (prior ranges for ρ₀, v₀, v_esc, V_⊙; Lisanti tail form; v₀–v_esc correlation sign; G_F and sin²θ_W; Anand L10 reduction;
  LHS; Gaussian copula; binned first-order index; Fourier amplitude definition) — see JSON.
* Datasets: none. Data requests: none (d₁₀ inherits P012's DR-001). WimPyDD-generated files: none (own kernel cache under work/P068/kernel_cache, 45 npz).
