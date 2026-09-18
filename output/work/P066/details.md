# P066 · A 300–380 keV line from the sky: INTEGRAL/SPI, COMPTEL and future MeV telescopes versus a long-lived excited dark-matter state — research record

Simulated date 2026-09-12. Author profile: MeV γ-ray astronomers. Category COMP, astro-ph.HE (cross-list hep-ph).
Script: `output/code/P066_gamma_line.py` (run from the simulation root with `.venv/bin/python`; ≈ 140 s, dominated by the three
HEALPix D-maps). Every number below is printed in `output/work/P066/run_log.txt` or stored in the CSV/JSON files of §9.
Recalled inputs are flagged [recall: reliability] and collected in `P066_results.json["recalled_knowledge"]` (16 items).

## 1. Motivation and framework

The inelastic readings of the LZ 248 keV event (P002, P007 Higgsino, P011 dark photon, P021 likelihood peak at δ ≈ 360–385 keV)
introduce an excited state χ₂ split from the ground state by δ ≈ 300–380 keV. P026 showed that χ₂ ↔ χ₁ bath transitions freeze
out with f₂ ≈ 0.42–0.50 of the dark matter left excited, and that the Higgsino χ₂ decays in ~10⁶ s (to νν̄; ~0.06 s if a photon
channel exists, P014) whereas the dark-photon χ₂ decays only to χ₁νν̄ with τ = 2.6×10¹⁸ s (m_A′/GeV)⁻⁴. P011 and P026 also noted that
LZ itself limits the *surviving* excited fraction: exothermic χ₂ N → χ₁ N down-scattering has no velocity threshold and, at the
LZ-fit cross-section, would give 858 events per unit f₂ at δ = 300 keV, so f₂(today) ≤ 1.2×10⁻³ (δ = 300), 5.5×10⁻⁵ (350),
3.8×10⁻⁷ (380 keV) [corpus P011 §3.7, P026 §5.5]. P042 and P023 treat the *prompt* de-excitation photon in the detector for
τ ≲ 10² s.

This paper asks what the MeV γ-ray sky says about the complementary regime τ_γ ≫ t_U: a population of χ₂ in the Galactic halo
decaying radiatively, χ₂ → χ₁γ, produces a narrow line at E_γ = δ − δ²/(2m_χ) ≈ δ. The observables are

* the Galactic line flux from a region Ω: **Φ_Ω = f₂ D_Ω / (4π m_χ τ_γ)**, with D_Ω = ∫_Ω dΩ ∫ ρ(r(s, ψ)) ds (GeV cm⁻² sr),
  m_χ in GeV, τ_γ in s → Φ in ph cm⁻² s⁻¹;
* the isotropic redshifted continuum from all decays since z ~ ∞ (dominated by z ≲ 1): for a rest-frame line at E₀ and decay rate
  Γ = 1/τ_γ per χ₂, with comoving number density n₀ = f₂ Ω_DM ρ_c/m_χ,
  **dI/dE = c Γ n₀ / [4π E H(z)]**, 1 + z = E₀/E, E ≤ E₀ [recall: certain; derived in §3.3];
* the line shape (Doppler width from the halo velocity dispersion; dipole from the Sun's motion) and morphology (∝ D(l, b)).

Because f₂ and τ_γ enter only as the ratio f₂/τ_γ (for τ_γ ≫ t_U), all γ-ray results are bounds on f₂/τ_γ; the (τ_γ, f₂) plane
(Fig. 1) combines them with the P026 Planck windows and the LZ exothermic bound, and shows where the corpus models sit.

## 2. Inputs

| Input | Value | Source |
|---|---|---|
| ρ_⊙ | 0.3 GeV cm⁻³ | `lz.RHO0_GEV_CM3` (Baxter et al. 2021 SHM) |
| v₀, Sun-frame speed | 238 km/s; \|v_⊙\| = 250.6 km/s | `lz.V0_KMS`, `lz.V_SUN_PEC` |
| Earth orbital speed | 29.8 km/s | `lz.V_EARTH_ORBIT` |
| R_⊙, r_s (NFW, Einasto) | 8.2 kpc, 20 kpc | [recall: likely] |
| Einasto α | 0.17 | [recall: likely] |
| Burkert core r_c | 9 kpc | [recall: uncertain] |
| halo truncation r_max | 250 kpc | assumption (D₁₀° changes −0.1 %/+0.03 % for 150/400 kpc) |
| m_χ (reference) | 1000 GeV | corpus (P007/P011 fits); all fluxes ∝ 1/m_χ |
| Planck 2018 | H₀ = 67.4, Ω_m = 0.315, Ω_c h² = 0.120, ρ_c = 1.054×10⁻⁵ h² GeV cm⁻³, t_U = 13.8 Gyr = 4.35×10¹⁷ s | [recall: certain] |
| CXB > 60 keV | Gruber et al. 1999: E dN/dE = 0.0259(E/60)⁻⁵·⁵ + 0.504(E/60)⁻¹·⁵⁸ + 0.0288(E/60)⁻¹·⁰⁵ keV cm⁻² s⁻¹ sr⁻¹ keV⁻¹ | [recall: uncertain ×2; COMPTEL lies ×2–3 below the SMM extrapolation at 1–10 MeV] |
| SPI narrow-line 3σ sensitivity, 1 Ms, point-like, 300 keV | ~3×10⁻⁵ ph cm⁻² s⁻¹ (bracket 2–5×10⁻⁵) | [recall: uncertain ×2; Roques et al. 2003] |
| SPI full-mission inner-Galaxy diffuse narrow line | 3×10⁻⁶–3×10⁻⁵ (10° region); 10⁻⁵–10⁻⁴ (30° and 47.5° boxes) | [recall: uncertain ×3; anchored on ⁶⁰Fe 1173/1332 keV ≈ 3×10⁻⁵ each at ~5σ, ²⁶Al 1809 keV 3×10⁻⁴, 511 keV bulge 10⁻³ at > 50σ; Teegarden & Watanabe 2006 first-year diffuse-line limits ~10⁻⁴; Calore et al. 2023 16-yr DM-line search in \|l\|,\|b\| < 47.5°] |
| SPI energy resolution at 300 keV | ≈ 2 keV FWHM | [recall: likely] |
| COMPTEL | 0.75–30 MeV | [recall: certain] — no sensitivity at 300–380 keV |
| COSI (launch ~2027) | 0.2–5 MeV; 2-yr 3σ narrow-line sensitivity ~10⁻⁵ near 511 keV (bracket 3×10⁻⁶–3×10⁻⁵ at 300 keV); Ge resolution ~1 % FWHM; ~4° angular resolution | [recall: uncertain ×3; Tomsick et al. 2023] |
| e-ASTROGAM / AMEGO(-X) concepts | narrow-line ~(1–5)×10⁻⁶ at 0.3–1 MeV | [recall: uncertain; concept projections] |
| Exothermic f₂ bound & τ_tot | f₂ ≤ 1.2×10⁻³ / 5.5×10⁻⁵ / 3.8×10⁻⁷ and τ_tot ≤ 7.2 / 4.8 / 3.1×10¹⁶ s at δ = 300 / 350 / 380 keV | corpus P011 §3.7, P026 §5.5 |
| Lifetimes | Higgsino τ_νν̄ = 1.2×10⁶ s, τ_γ ~ 0.06 s; dark photon τ_νν̄ = 2.6×10¹⁸ s (m_A′/GeV)⁻⁴, BR_γ = 0 at tree level; photon-M1 composite τ = 66 μs | corpus P014, P026, P011, P023 |
| P026 D-factor | D(10° cone, NFW) = 7.41×10²¹ GeV cm⁻² sr; SPI bound f₂/τ_γ < 5.1×10⁻²³ s⁻¹; EGB τ > 5.7×10¹⁹ s (f₂ = 0.5) | corpus P026 §5.4 |

## 3. Method and derivations

### 3.1 Density profiles
All three profiles are normalised to ρ(R_⊙) = 0.3 GeV cm⁻³:
NFW ρ = ρ_s/[(r/r_s)(1 + r/r_s)²] with ρ_s = ρ_⊙ (R_⊙/r_s)(1 + R_⊙/r_s)² = 0.2445 GeV cm⁻³;
Einasto ρ ∝ exp{−(2/α)[(r/r_s)^α − 1]}, α = 0.17; Burkert ρ ∝ 1/[(1 + r/r_c)(1 + r²/r_c²)], r_c = 9 kpc.
Galactocentric radius along a line of sight at angle ψ from the GC: r² = R_⊙² + s² − 2 R_⊙ s cos ψ, 0 ≤ s ≤ s_max(ψ) =
R_⊙ cos ψ + √(r_max² − R_⊙² sin²ψ).

### 3.2 Line-of-sight and region integrals
Two independent integrators: (i) `scipy.integrate.quad` with a break point at the closest approach s = R_⊙ cos ψ, followed by a 1-D
azimuthal quad for cones (2π ∫ sin θ dθ); (ii) a vectorised trapezoid in ln u on both sides of the closest approach (600 log-spaced
nodes from u = 10⁻⁷ L to L per side), evaluated at the centres of all 196 608 HEALPix pixels (nside = 128, 0.458° pixels). Regions
are pixel masks: cones θ < 5°, 10°, 16°, 30° about the GC; boxes |l|, |b| < 10°, 30°, 47.5°; all sky; anti-centre cones (ψ > 150°,
170°); |b| > 30°. D_Ω = Σ_pix D(ψ_pix) Ω_pix. The NFW cusp is log-singular exactly at ψ = 0; the pixel-centre sampling of it is
checked against the quad cone integrals (§5.1). The exact-centre intensity quoted in §4.5 is the 1° cone average.

### 3.3 Redshifted continuum
Decays between t and t + dt inject Γ n₀ dt photons per comoving volume; today they form an isotropic population of the same number
density, so dI = (c/4π) Γ n₀ dt per sr. With E = E₀/(1 + z), dt = dz/[(1 + z)H(z)] and dE = −E₀ dz/(1 + z)²,
dI/dE = (c/4π) Γ n₀ (1 + z)/[E₀ H(z)] = c Γ n₀/[4π E H(z)]. H(z) = H₀√(Ω_m(1 + z)³ + Ω_Λ). The spectrum has a sharp edge at E₀ and
falls toward low E as ~E^{1/2} in the matter era; 59 % of the photons arrive in E₀/2 < E < E₀ (z < 1). Because the CXB steepens as
E^{−1.6} toward low E, the decay/CXB ratio is maximal at E = E₀ (z = 0): the bound is set at the edge. Criteria: decay ≤ 100 % of the
Gruber CXB (conservative) and ≤ 10 % (a stand-in for the residual after AGN-population modelling); repeated with a CXB halved.

### 3.4 Line shape
Isotropic SHM: σ_los = v₀/√2 = 168 km/s → σ_E = E₀ σ_los/c, FWHM = 2.355 σ_E. Inner-halo dispersion 250 km/s as the upper case.
Solar motion: ΔE(l, b) = E₀ (v_⊙/c) cos(angle to the solar apex), amplitude ±0.25 keV at 300 keV (blue toward the apex, red toward
the anti-apex); Earth's orbit adds ±0.03 keV annually. Recoil shift δ²/(2m_χ) = 0.045 eV. Gravitational redshift from the GC potential
(~10⁻⁶) is 0.3 eV. All negligible against SPI's ≈ 2 keV and COSI's ≈ 3 keV FWHM: the line is *unresolved*; only its narrowness
(≤ 1 keV) is testable, and the dipole would need a ≲ 0.1 keV-resolution instrument.

### 3.5 Corpus mapping
For each model: f₂(today) = f₂^fo e^{−t_U/τ_tot} with f₂^fo = 0.42, τ_γ = τ_tot/BR_γ, Φ₁₀° = f₂ D₁₀°/(4π m τ_γ), compared with the
SPI bracket. For the dark photon the maximum LZ-allowed population sits at the exothermic boundary (f₂ = 1.2×10⁻³, τ_tot =
7.2×10¹⁶ s at δ = 300 keV); a UV transition dipole with branching BR_γ then gives Φ₁₀° = f₂ BR_γ D₁₀°/(4π m τ_tot). Along the physical
photon-only curve f₂(τ) e^{−t_U/τ}/τ peaks at τ = t_U.

## 4. Results

### 4.1 D-factors (GeV cm⁻² sr; `D_factors.csv`)

| Region | Ω (sr) | NFW | Einasto | Burkert |
|---|---|---|---|---|
| cone 5° | 0.0239 | 2.354×10²¹ | 3.062×10²¹ | 9.16×10²⁰ |
| cone 10° | 0.0963 | **7.480×10²¹** (quad 7.430×10²¹) | 9.337×10²¹ | 3.611×10²¹ |
| cone 16° | 0.243 | 1.559×10²² | 1.871×10²² | 8.80×10²¹ |
| cone 30° | 0.842 | 3.929×10²² | 4.439×10²² | 2.732×10²² |
| box \|l\|,\|b\| < 10° | 0.122 | 9.020×10²¹ | 1.115×10²² | 4.544×10²¹ |
| box < 30° | 1.044 | 4.546×10²² | 5.080×10²² | 3.269×10²² |
| box < 47.5° | 2.453 | 8.040×10²² | 8.629×10²² | 6.451×10²² |
| all sky | 12.57 | 2.027×10²³ | 2.055×10²³ | 1.781×10²³ |
| anti-centre cone 30° | 0.842 | 6.809×10²¹ | 6.568×10²¹ | 6.112×10²¹ |
| anti-centre cone 10° | 0.0963 | 7.62×10²⁰ | 7.35×10²⁰ | 6.82×10²⁰ |
| \|b\| > 30° | 6.316 | 8.832×10²² | 8.685×10²² | 8.193×10²² |

Per-sr intensities (`D_per_sr_directions.csv`): NFW 1.47×10²³ (1° average at the GC), 6.30×10²² (10° off), 3.31×10²² (30°),
1.24×10²² (poles / l = 90°), 7.89×10²¹ (anti-centre); Burkert 3.87×10²² at the GC. Contrast GC(1°)/anti-centre = 18.6 (NFW), 25.2
(Einasto), 5.5 (Burkert). Fraction of the all-sky flux inside 10°: 3.7 / 4.5 / 2.0 %; inside the 30° box: 22 / 25 / 18 %.

### 4.2 Line flux (`line_flux_table.csv`)
Φ_Ω = 5.95×10⁻⁵ (D_Ω/D₁₀°)(f₂/τ_γ / 10⁻²² s⁻¹)(TeV/m_χ) ph cm⁻² s⁻¹ for NFW. At f₂/τ_γ = 10⁻²² s⁻¹, m_χ = 1 TeV:
cone 5° 1.87×10⁻⁵; cone 10° 5.95×10⁻⁵ (Einasto 7.43×10⁻⁵, Burkert 2.87×10⁻⁵); box 10° 7.18×10⁻⁵; cone 30° 3.13×10⁻⁴;
box 30° 3.62×10⁻⁴; box 47.5° 6.40×10⁻⁴; all sky 1.61×10⁻³; anti-centre 30° 5.42×10⁻⁵; |b| > 30° 7.03×10⁻⁴ (Fig. 2).

### 4.3 Bounds on f₂/τ_γ from line sensitivities (`f2_tau_bounds.csv`; NFW unless stated)

| Instrument / region | S (ph cm⁻² s⁻¹) | f₂/τ_γ < (s⁻¹) | τ_γ > (f₂ = 0.42) | τ_γ > (f₂ = 1.2×10⁻³) |
|---|---|---|---|---|
| SPI 1 Ms point-like, 5° | 2–5×10⁻⁵ | 1.1–2.7×10⁻²² | 1.6–3.9×10²¹ s | 4.5×10¹⁸–1.1×10¹⁹ s |
| SPI full mission, 10° cone | 3×10⁻⁶–3×10⁻⁵ | **5.0×10⁻²⁴–5.0×10⁻²³** | 8.3×10²¹–8.3×10²² s | 2.4×10¹⁹–2.4×10²⁰ s |
| SPI, 30° box | 10⁻⁵–10⁻⁴ | 2.8×10⁻²⁴–2.8×10⁻²³ | 1.5×10²²–1.5×10²³ s | 4.3×10¹⁹–4.3×10²⁰ s |
| SPI 16-yr DM-line ROI, 47.5° box | 10⁻⁵–10⁻⁴ | 1.6×10⁻²⁴–1.6×10⁻²³ | 2.7×10²²–2.7×10²³ s | 7.7×10¹⁹–7.7×10²⁰ s |
| COSI 2 yr, 10° | 3×10⁻⁶–3×10⁻⁵ | 5.0×10⁻²⁴–5.0×10⁻²³ | 8.3×10²¹–8.3×10²² s | — |
| e-ASTROGAM/AMEGO, 10° | 1–5×10⁻⁶ | 1.7×10⁻²⁴–8.4×10⁻²⁴ | 5×10²²–2.5×10²³ s | 1.4–7.1×10²⁰ s |

Profile dependence at 10°: Einasto ×0.80, Burkert ×2.1 on the f₂/τ_γ bound; in the 47.5° box the spread shrinks to ×0.93–1.25.
P026's single-number bound (5.1×10⁻²³ s⁻¹ with S = 3×10⁻⁵) is the conservative edge of our bracket.

### 4.4 CXB continuum (`cxb_bound.csv`, Fig. 3)
At E₀ = 300 keV the decay continuum for f₂/τ_γ = 10⁻²² s⁻¹ is dI/dE = 4.6×10⁻⁷ ph cm⁻² s⁻¹ sr⁻¹ keV⁻¹ at the edge versus the CXB
1.50×10⁻⁴ (E² dN/dE = 13.5 keV cm⁻² s⁻¹ sr⁻¹). Bounds: f₂/τ_γ < 3.3×10⁻²⁰ (100 % Gruber), 3.3×10⁻²¹ (10 %), 1.6×10⁻²⁰ / 1.6×10⁻²¹
(CXB halved); at 350 keV 2.6×10⁻²⁰ / 2.6×10⁻²¹, at 380 keV 2.3×10⁻²⁰ / 2.3×10⁻²¹. For f₂ = 0.42, τ_γ > 1.3×10¹⁹ (100 %) to
2.6×10²⁰ s (10 %, halved CXB); P026's 5.7×10¹⁹ s lies inside this spread. The continuum bound is ~650× weaker than the 10° line bound
but independent of the halo profile and of the coded-mask diffuse-emission systematics. The observable signature would be a step
("300 keV bump" edge) in the CXB at E = δ with a ~E^{1/2} low-energy tail.

### 4.5 Corpus models (`corpus_models.csv`)
* **Higgsino** (P007/P014/P026): τ_νν̄ = 1.2×10⁶ s → f₂(today) = 0.42 e^{−3.6×10¹¹} = 0; τ_γ ~ 0.06 s. No line.
* **Photon-M1 composite / MiDM** (P023, P042): τ = 66 μs; f₂ = 0. No Galactic line (the photon is emitted in the detector, P042).
* **Dark photon, pure kinetic mixing** (P011): BR_γ = 0 — the photon attaches to the dark current only through the A′–γ mixing
  insertion ∝ ε q², which vanishes for an on-shell photon (q² = 0), to all orders in the SM vacuum polarisation (Π_{A′γ}(0) = 0 by
  gauge invariance). Only χ₂ → χ₁νν̄ (Z mixing). No line at any f₂.
* **Dark photon + UV transition dipole** at the exothermic boundary (δ = 300 keV: f₂ = 1.2×10⁻³, τ_tot = 7.2×10¹⁶ s):
  Φ₁₀° = 9.9×10⁻⁶ (BR_γ/10⁻³) ph cm⁻² s⁻¹ ≡ 1.0×10⁻² BR_γ. SPI: **BR_γ < 3.0×10⁻⁴ (S = 3×10⁻⁶) – 3.0×10⁻³ (3×10⁻⁵)**; e-ASTROGAM
  1.0×10⁻⁴. At δ = 350 keV (f₂ = 5.5×10⁻⁵, τ_tot = 4.8×10¹⁶ s): Φ₁₀° = 6.8×10⁻⁷ (BR_γ/10⁻³) → BR_γ < 4.4×10⁻³–4.4×10⁻². At 380 keV
  (3.8×10⁻⁷, 3.1×10¹⁶ s): 7.3×10⁻⁹ (BR_γ/10⁻³) → unconstrained (BR_γ < 0.4–4). If the exothermic bound is ignored (f₂ = 0.42 relic,
  m_A′ = 1 GeV, τ_νν̄ = 2.6×10¹⁸ s): Φ₁₀° = 9.7×10⁻⁵ BR_γ/10⁻³ — SPI-visible for any BR_γ ≥ 10⁻³, but that population is already
  excluded 350-fold by LZ (P011).
* **Generic photon-only long-lived χ₂**, f₂ = 0.42 e^{−t_U/τ_γ}: Φ₁₀° = 2.4×10⁻² (τ_γ = 10¹⁹ s), 2.5×10⁻³ (10²⁰), 2.5×10⁻⁴ (10²¹),
  2.5×10⁻⁵ (10²²), 2.5×10⁻⁶ (10²³), 2.5×10⁻⁹ (10²⁶). The τ_γ = 10²⁰ s case is 2.5× the 511 keV bulge flux (10⁻³, recalled). SPI/COSI
  exclude τ_γ < 8.3×10²¹ (S = 3×10⁻⁵) – 8.3×10²² s (3×10⁻⁶); e-ASTROGAM would reach 2.5×10²³ s. The physical curve peaks at τ_γ = t_U
  with f₂ = 0.155, f₂/τ = 3.5×10⁻¹⁹ s⁻¹, Φ₁₀° = 0.21 ph cm⁻² s⁻¹ (200× the 511 keV line) — grossly excluded. Note that *if this χ₂ is
  the partner of LZ's scatterer*, any f₂ > 1.2×10⁻³ is excluded by exothermic scattering regardless of γ-rays; the generic scenario
  survives only for a χ₂ decoupled from the LZ rate (e.g. a different, weakly coupled state) or for f₂ capped at 1.2×10⁻³, where SPI
  still excludes τ_γ < 2.4×10¹⁹–2.4×10²⁰ s (Φ₁₀° = 7.1×10⁻⁵ at 10¹⁹ s, 7.1×10⁻⁶ at 10²⁰ s).

### 4.6 Distinctive signatures (`doppler.json`)
At E₀ = 300 keV: σ_E = 0.168 keV, FWHM = 0.40 keV (0.59 keV for 250 km/s inner-halo dispersion); dipole amplitude ±0.25 keV
(0.50 keV peak-to-peak across the sky); annual modulation ±0.03 keV; recoil 0.045 eV. Scaling ∝ E₀: FWHM 0.46 / 0.50 keV at 350 /
380 keV. Morphology: smooth, centrally peaked, symmetric in l and b, GC/anti-centre contrast 19 (NFW) / 25 (Einasto) / 5.5 (Burkert),
no disk or bulge component, ~22 % of the flux inside |l|,|b| < 30° and 44 % at |b| > 30° — unlike the 511 keV emission (bulge +
disk) or ²⁶Al (disk). The 300–380 keV window contains no known astrophysical narrow line (nearest ⁷Be 478 keV, e⁺e⁻ 511 keV; ⁴⁴Ti
68/78 keV, ⁵⁷Co 122/136 keV below) [recall: likely]; it sits on the smooth positronium three-photon continuum of the bulge, which
does not mimic a narrow line; no strong SPI instrumental line is recalled in this window [recall: uncertain].

## 5. Validation and robustness

### 5.1 Integrators and P026 (`validation.json`)
NFW cone D, quad vs HEALPix: 5° 2.3547 vs 2.3537×10²¹ (ratio 0.9996); 10° 7.4302 vs 7.4801×10²¹ (1.0067); 16° 1.5608 vs
1.5588×10²² (0.9987); 30° 3.9278 vs 3.9285×10²² (1.0002). The +0.7 % at 10° is the pixelised cusp (pixel-centre sampling of the
log-singular NFW centre); it is the level of the D-factor numerical uncertainty and far below the sensitivity brackets. Single
directions ψ = 0.5°, 2°, 10°, 45°, 90°, 180°: vectorised and quad agree to ≤ 1×10⁻⁴. P026's D₁₀° = 7.41×10²¹ is reproduced (our quad
value is 0.27 % higher). r_max = 150 / 400 kpc changes D₁₀° by −0.09 % / +0.03 %.

### 5.2 Robustness
* Profile: 10°-cone fluxes span ×0.48 (Burkert) to ×1.25 (Einasto) of NFW; the 47.5° box only ×0.80–1.07. Bounds on f₂/τ_γ scale
  inversely. A ρ_⊙ of 0.4 GeV cm⁻³ (Gaia-era values, P018) would tighten everything ×1.33.
* Sensitivity bracket (×10 spanned) dominates every γ-ray number; the paper quotes brackets, not central values.
* CXB: level ×2 and criterion ×10 → bound spread 3×10⁻²⁰–1.6×10⁻²¹ s⁻¹ (§4.4).
* Mass: all fluxes ∝ 1/m_χ; the corpus fits allow 0.26–4 TeV (P002, P021), i.e. ×0.25–3.9.
* Line width: with a k = 2–3 Lisanti tail or the Sausage anisotropy (P018) σ_los changes by ≲ 20 %; the line stays unresolved.

## 6. Figures
* `figures/P066_fig1_f2_tau_plane.png` — Fig. 1: (τ_γ, f₂(today)) plane at m_χ = 1 TeV, δ = 300 keV, NFW: SPI 10° bracket (orange),
  CXB continuum 100 %/10 % (violet), Planck windows from P026 (red polygon), LZ exothermic f₂ > 1.2×10⁻³ (blue, hatched), the physical
  photon-only curve f₂ = 0.42 e^{−t_U/τ}, dark photon + dipole points at the exothermic boundary (BR_γ = 10⁻³, 10⁻², 10⁻¹), generic
  long-lived χ₂ triangles; Higgsino and MiDM annotated off-scale.
* `figures/P066_fig2_flux_vs_roi.png` — Fig. 2: line flux per region for the three profiles at f₂/τ_γ = 10⁻²² s⁻¹ with recalled
  sensitivity brackets (SPI/COSI 10°, SPI 30°, e-ASTROGAM/AMEGO).
* `figures/P066_fig3_cxb.png` — Fig. 3: E² dN/dE of the redshifted decay continuum for E₀ = 300/350/380 keV at the CXB-saturating
  f₂/τ_γ and at the SPI bound, against the Gruber (1999) CXB (grey band: ×0.5).
* `figures/P066_fig4_skymap.png` — Fig. 4: HEALPix Mollweide map of log₁₀ D(l, b) for NFW (line intensity morphology).

## 7. Failed or abandoned approaches
* A first attempt to draw the HEALPix map into an existing matplotlib figure produced a healpy warning (figure reuse); replaced by a
  dedicated figure number.
* Dictionary keys containing "1e-22" were invalid Python identifiers (SyntaxError); renamed.
* No attempt was made to model SPI's coded-mask response to extended emission or its instrumental background; the sensitivity is a
  recalled bracket. No data request is raised: SPI/COSI line-sensitivity curves are instrument-team products, not tabulated public data
  we could recast here.
* A P058 exothermic paper was not available (only partial work files); the exothermic bound is taken from P011/P026.

## 8. Discussion
The MeV sky and LZ constrain orthogonal combinations: LZ bounds f₂(today) directly (through the LZ-fit cross-section), the γ-ray line
bounds f₂/τ_γ. Their product statement is the useful one: for *any* model that fits LZ at δ ≈ 300 keV, the halo χ₂ population is
≤ 1.2×10⁻³ and decays in ≤ 7×10¹⁶ s, so the brightest possible 300 keV line is Φ₁₀° ≈ 10⁻² BR_γ ph cm⁻² s⁻¹; SPI's archival
non-detection of such a line (recalled sensitivity 3×10⁻⁶–3×10⁻⁵) then caps the radiative branching at BR_γ ≲ 3×10⁻⁴–3×10⁻³. Neither
LZ model predicts a line: the Higgsino χ₂ is gone, and the kinetic-mixing dark photon has no photon channel. A detection of a
0.4 keV-wide, cusp-shaped 300–380 keV line would therefore point to a *third* ingredient (a transition dipole or a decoupled excited
population), not to the models on the table. COSI will survey the whole sky at ~4° resolution from 2027; its narrow-line sensitivity
near 10⁻⁵ will re-derive the SPI bound with independent systematics and a Compton (non-coded-mask) response to diffuse emission.

## 9. Output files
`D_factors.csv`, `D_per_sr_directions.csv`, `validation.json`, `line_flux_table.csv`, `f2_tau_bounds.csv`, `sensitivities.csv`,
`cxb_bound.csv`, `corpus_models.csv`, `doppler.json`, `P066_results.json`, `run_log.txt`, `figures/P066_fig1–4*.png`.

## 10. References
LZ Collaboration, arXiv:2609.02823 (2026) · J. P. Roques et al., A&A 411, L91 (2003) · B. J. Teegarden, K. Watanabe, ApJ 646, 965
(2006) · D. E. Gruber et al., ApJ 520, 124 (1999) · R. Essig, E. Kuflik, S. D. McDermott, T. Volansky, K. Zurek, JCAP 11, 193 (2013) ·
J. A. Tomsick et al., arXiv:2308.12362 (2023) · T. Siegert et al., A&A 586, A84 (2016) · F. Calore, A. Dekker, P. D. Serpico, T. Siegert,
MNRAS 520, 4167 (2023) [recalled, likely] · J. F. Navarro, C. S. Frenk, S. D. M. White, ApJ 490, 493 (1997) · A. Burkert, ApJ 447, L25
(1995) · corpus: P002, P007, P011, P014, P018, P021, P023, P026, P042, dossier 00.

## 11. Tools and provenance (mirrors `provenance/P066.json`)
* Agent tools: Read ×18 (PAPER_GUIDE, dossier, ledger, P026/P011/P014/P042/P023 papers, P026 details §1 and §5.4–5.5, P026.json,
  figures ×7); Bash ×15 (ls/grep/cat inspections ×7 incl. ledger via pandas, lzcommon constants, ENVIRONMENT_versions, P058 work files,
  dataviz palette; script runs ×4 incl. one SyntaxError; JSON checks; word-count checks ×3); Write ×4; Edit ×16 (script 9, paper 7);
  Skill ×1 (dataviz).
* Software: python 3.12.13; numpy 2.5.3 (trapezoid, linalg.norm, logspace); scipy 1.18.1 (integrate.quad, optimize.brentq); healpy
  1.20.0 (nside2npix, pix2ang, nside2pixarea, mollview, graticule); pandas 3.0.5; matplotlib 3.11.2 (Agg); common/lzcommon.py
  (RHO0_GEV_CM3, V0_KMS, V_SUN_PEC, V_EARTH_ORBIT, C_KMS).
* 16 recalled items (§2); datasets none; data requests none; WimPyDD files none.
