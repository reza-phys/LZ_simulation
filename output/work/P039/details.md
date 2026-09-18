# P039 — Neutron-star kinetic heating by TeV inelastic dark matter: does the LZ-fit cross-section light up old neutron stars?

Research record (simulated date 2026-09-09; author profile: compact-object astrophysicists; category COMP; arXiv astro-ph.HE, cross-list hep-ph).
Script: `output/code/P039_ns_heating.py` (run from the simulation root with `.venv/bin/python`; full stdout in `output/work/P039/run_log.txt`). All numbers quoted below are printed by the script or stored in `output/work/P039/results.json` and the CSV tables listed in §9.

## 1. Motivation and framework

Every inelastic-DM reading of the LZ 248 keV event (P007 Higgsino, σ_n = 7.4×10⁻³⁹ cm², δ ≈ 366 keV; P011 dark photon, σ_p = 1.2×10⁻⁴²–2.3×10⁻³⁹ cm² for δ = 250–365 keV; P021 isoscalar best fit σ_n(δ) = 2.2×10⁻⁴², 1.2×10⁻⁴⁰, 6.5×10⁻³⁸ cm² at δ = 300, 350, 380 keV) relies on a splitting that is only barely accessible in the Galactic halo: on 16 June the largest xenon-accessible splitting for 248 keV is 387 keV (P002). In a compact star the situation reverses: dark matter falling onto a neutron star arrives at ≈ 0.6c, and the kinetic energy available per nucleon (hundreds of MeV) makes a sub-MeV δ irrelevant. Neutron-star kinetic heating (Baryakhtar et al. 2017; Raj, Tanedo & Yu 2018; inelastic case: Bell, Busoni & Robles 2018) therefore probes exactly the cross-sections the LZ fits require, independently of the halo tail. We ask: (i) is capture saturated for the corpus cross-sections; (ii) what surface temperature results and is it observable with JWST/ELT; (iii) what would a ~2000 K old neutron star imply; (iv) what do white dwarfs and the Sun add for δ = 300–380 keV.

Halo: Baxter-2021 SHM as implemented in `lzcommon.wd_halo()` (v₀ = 238 km/s, v_esc = 544 km/s, Sun peculiar velocity, annual average). From the WimPyDD halo function (vmin, δη): ⟨1/v⟩⁻¹ = 287.9 km/s, ⟨v⟩ = 355.5 km/s, √⟨v²⟩ = 379.9 km/s. For a neutron star with a 400 km/s space velocity (recalled/likely typical kick), `lz.eta0(0, v_e=400)` gives ⟨1/v⟩⁻¹ = 405.2 km/s. Local density ρ_χ = 0.3–0.4 GeV cm⁻³ (recalled/likely); 1.0 GeV cm⁻³ is shown as a variation.

## 2. Neutron-star model and capture geometry

Benchmark (recalled/likely): M = 1.5 M_⊙, R = 12 km. Schwarzschild factor B ≡ 1 − 2GM/(Rc²) = 0.6308.

* Locally measured escape speed v_esc = c√(1−B) = 0.6077c (numerically identical to the Newtonian √(2GM/R); the assignment's "≈ 0.63c" corresponds to a slightly more compact star).
* Lorentz factor of DM falling from rest at infinity, as measured at the surface: γ = B^{−1/2} = 1.259; local kinetic energy per particle (γ−1)m_χc² = 0.259 m_χc² (Newtonian v²/2 = 0.185).
* Energy released at infinity per captured particle: m_χc²[1 − √B] = 0.2058 m_χc² (the DM at rest on the surface retains energy-at-infinity √B m_χc²). This is the kinetic-heating efficiency.
* Apparent radius R_∞ = R/√B = 15.11 km. Angular-momentum conservation for a particle with asymptotic speed v ≪ c gives the GR capture impact parameter b_max² = R_∞²(B + v_esc²/v²); with the Maxwellian averages the mass capture rate is

  Ṅ m_χ = Ṁ = ρ_χ π R_∞² [B⟨v⟩ + v_esc²⟨1/v⟩].

  For ρ_χ = 0.4 GeV cm⁻³: Ṁ = 58.96 g s⁻¹ (Newtonian, R instead of R_∞: 37.2 g s⁻¹); ρ = 0.3: 44.2 g s⁻¹; v_star = 400 km/s: 41.9 g s⁻¹ (ρ = 0.4).

Optical depth. N_n = M/m_n = 1.781×10⁵⁷ neutrons, n_n = 2.46×10³⁸ cm⁻³ (uniform-density proxy). The mean chord through a uniform sphere is 4R/3, so the mean optical depth is τ̄ = σ N_n/(πR²) ≡ σ/σ_crit with

  σ_crit,n = πR²/N_n = 2.54×10⁻⁴⁵ cm²  (assignment estimate m_nR²/M ≈ 2×10⁻⁴⁵).

For a proton-only coupling (dark photon; P011 has c_n = 0) the target count is Y_p N_n with Y_p ≈ 0.07 (recalled/likely core proton fraction 0.05–0.1): σ_crit,p = 3.63×10⁻⁴⁴ cm². The chord-averaged single-scatter capture probability for a uniform sphere is derived in closed form (y = 2nσR = 1.5 τ̄):

  f(τ̄) = 1 − (2/y²)[1 − e^{−y}(1+y)] → τ̄ for τ̄ ≪ 1, → 1 for τ̄ ≫ 1,

and is compared with the commonly used 1 − e^{−τ̄} (both tabulated in `T_vs_sigma.csv`; they differ by < 25 % for τ̄ ~ 1 and are indistinguishable in both limits).

Single versus multiple scattering. A heavy DM particle at βγ = √((1−B)/B) = 0.765 hitting a neutron at rest can transfer at most ΔE_max = 2m_nβ²γ² = 1.10 GeV; capture needs a loss of only KE_∞ = m_χ v²/2 = 0.5 MeV × (m_χ/TeV)(v/300 km/s)², so one scatter captures up to m_χ = 1.37×10⁶ GeV (single-scatter regime for all corpus masses 0.3–4 TeV). The typical momentum transfer q ≈ m_nβγ = 0.72 GeV exceeds the neutron Fermi momentum (~0.4 GeV, recalled/likely), so Pauli blocking is absent; the nucleon form factor and relativistic corrections at q ~ 0.7–1 GeV can reduce the cross-section by ~0.03–0.3 (recalled/likely, Bell et al.), which we include as a ×0.03 stress test.

### 2.1 Saturation for the corpus cross-sections (Table 1, `saturation_table.csv`)

| model | σ [cm²] | σ_crit used | σ/σ_crit | f | mean free path of χ₂ | R/λ | f with ×0.03 form-factor |
|---|---|---|---|---|---|---|---|
| Higgsino (P007) σ_n | 7.4e-39 | 2.54e-45 | 2.9e6 | 1.0000 | 0.55 cm | 2.2e6 | 1.000 |
| dark photon σ_p, δ=250 | 1.2e-42 | 3.63e-44 | 33 | 0.9992 | 484 m | 25 | 0.604 |
| dark photon σ_p, δ=300 | 8.6e-42 | 3.63e-44 | 237 | 1.0000 | 68 m | 178 | 0.982 |
| dark photon σ_p, δ=350 | 3.0e-40 | 3.63e-44 | 8.3e3 | 1 | 1.9 m | 6.2e3 | 1 |
| dark photon σ_p, δ=365 | 2.3e-39 | 3.63e-44 | 6.3e4 | 1 | 25 cm | 4.8e4 | 1 |
| P021 σ_n, δ=300 | 2.2e-42 | 2.54e-45 | 865 | 1.0000 | 18.5 m | 648 | 0.999 |
| P021 σ_n, δ=350 | 1.2e-40 | 2.54e-45 | 4.6e4 | 1 | 35 cm | 3.4e4 | 1 |
| P021 σ_n, δ=380 | 6.5e-38 | 2.54e-45 | 2.6e7 | 1 | 0.06 cm | 1.9e7 | 1 |

Every LZ-fit cross-section lies 30 to 3×10⁷ above the geometric threshold; capture is saturated. Only the weakest dark-photon point (δ = 250 keV, proton-only) drops to f ≈ 0.6 if a ×0.03 form-factor suppression is imposed.

## 3. The inelastic threshold inside the star and the fate of χ₂

Available centre-of-mass kinetic energy for χ₁ + n at the surface: non-relativistic μv_esc²/2 = 173 MeV; relativistic √s − m_χ − m_n with s = m_χ² + m_n² + 2γm_χm_n: 243 MeV. A splitting of 380 keV is 1.6×10⁻³ of this: the inelastic threshold that governs LZ (δ ≤ 387 keV) is irrelevant in a neutron star, and the up-scattered χ₂ is produced with essentially the incoming velocity.

Fate of χ₂ (`chi2_fate.csv`). Light-crossing time R/c = 40 μs. Decay lengths cτ: 6.0×10³ km (τ = 0.02 s, assignment lower bracket), 1.8×10⁴ km (Higgsino τ_γ ≈ 0.06 s, P014/P026), 1.4×10¹¹ km (τ_νν̄ = 4.6×10⁵ s), 2×10²² km (dark photon, P011/P026). R/(cτ) ≤ 2×10⁻³: decays inside the star are negligible. But χ₂ never exits: with the same cross-section for exothermic down-scattering (P011, P026) its mean free path is 0.06 cm–480 m (Table 1), i.e. R/λ = 25–2×10⁷ scatterings across the star and P(exit unscattered) = e^{−R/λ} ≤ 10⁻¹¹. Whether χ₂ decays or down-scatters is therefore immaterial for capture: the energy is deposited locally either way and χ₁ thermalises by further scatters (the neutron-star kinetic-heating literature treats inelastic DM with δ up to ~100 MeV in this way; Bell et al. 2018).

## 4. Heating, temperature and near-infrared detectability

L_∞ = Ṁc²(1−√B); locally L_loc = L_∞/B = 4πR²σ_SB T_loc⁴; T_∞ = √B T_loc. Full capture (f = 1), from `ns_heating_table.csv`:

| ρ_χ [GeV cm⁻³] | star velocity | Ṁ [g s⁻¹] | L_∞ [erg s⁻¹] | T_loc [K] | T_∞ kinetic [K] | T_∞ kinetic + annihilation [K] |
|---|---|---|---|---|---|---|
| 0.3 | Sun-like | 44.2 | 8.18e21 | 1885 | 1497 | 2223 |
| 0.3 | 400 km/s | 31.4 | 5.81e21 | 1731 | 1375 | 2041 |
| 0.4 | Sun-like | 59.0 | 1.09e22 | 2026 | **1609** | **2389** |
| 0.4 | 400 km/s | 41.9 | 7.75e21 | 1860 | 1477 | 2193 |
| 1.0 | Sun-like | 147 | 2.73e22 | 2548 | 2023 | 3004 |
| 1.0 | 400 km/s | 105 | 1.94e22 | 2339 | 1858 | 2758 |

The full-capture kinetic temperature T_∞ = 1500–1600 K reproduces the recalled literature value ≈ 1700 K (Baryakhtar et al. 2017; likely) within the ρ/velocity/radius conventions; with annihilation of the captured DM (rest mass deposited too, L_∞ = Ṁc²) T_∞ = 2200–2400 K, versus the recalled ≈ 2500 K. Since f = 1 for all corpus cross-sections, **every LZ-fit model gives the same T_∞ ≈ 1600 K (ρ = 0.4)**; the σ-dependence appears only below σ_crit (T ∝ σ^{1/4}; Fig. 1). A 2000 K measurement requires ρ_χ = 0.95 GeV cm⁻³ for kinetic-only heating or 0.20 GeV cm⁻³ with annihilation; it cannot discriminate Higgsino from dark photon, and it is confounded by any other heat source (ISM Bondi–Hoyle accretion at n = 1 cm⁻³, v = 30 km/s would give Ṁ = 3.1×10¹⁰ g s⁻¹, L = 5.7×10³⁰ erg s⁻¹ = 5×10⁸ × the DM heating, if not magnetically inhibited; recalled/likely). Conversely an old (≳ 10⁷–10⁸ yr, recalled/likely to have cooled below ~10³ K) isolated neutron star measured at T_∞ < 1400 K in a ρ_χ ≈ 0.3–0.4 GeV cm⁻³ environment would exclude the whole σ ≳ 3×10⁻⁴⁴ cm² range, i.e. every corpus fit, under the standard capture assumptions.

Near-infrared flux (`nir_flux_table.csv`): F_ν = πB_ν(T_∞)(R_∞/d)². At 2 μm (NIRCam F200W) for ρ = 0.4:

| d [pc] | kinetic (1609 K) | kinetic + annihilation (2389 K) | hypothetical 2000 K |
|---|---|---|---|
| 10 | 0.43 nJy, AB 32.3; t(5σ) ≈ 1.1×10⁶ s | 1.94 nJy, AB 30.7; 5.4×10⁴ s | 1.06 nJy, AB 31.3; 1.8×10⁵ s |
| 30 | 0.048 nJy, AB 34.7; 8.7×10⁷ s | 0.22 nJy, AB 33.1; 4.4×10⁶ s | 0.12 nJy, AB 33.7; 1.5×10⁷ s |
| 100 | 0.0043 nJy, AB 37.3 | 0.019 nJy, AB 35.7 | 0.011 nJy, AB 36.3 |

Exposure times scale a recalled/uncertain NIRCam F200W point-source sensitivity of ≈ 9 nJy (AB 28.9) at 10σ in 10⁴ s as t ∝ (F_lim/F)² (background-limited). A kinetically heated neutron star is detectable only within ~10 pc and with ~10⁶ s; with annihilation heating a 10 pc star needs ~5×10⁴ s. The nearest known isolated neutron stars are ≳ 100 pc away and young/hot (recalled/uncertain); with a local number density of ~(1–4)×10⁻⁴ pc⁻³ (recalled/uncertain) one expects ~0.4–1.7 neutron stars within 10 pc, none yet identified. The ELT (K band, ~40 m aperture) would improve on these times by roughly an order of magnitude for a point source (qualitative).

## 5. White dwarfs

Benchmark (assignment): M = 1 M_⊙, R = 8000 km (a real 1 M_⊙ WD is more compact, ~5500–6000 km; recalled/likely). Surface v_esc = 5761 km/s; the WimPyC White_Dwarf profile (0.49 M_⊙, R = 0.01348 R_⊙ = 9390 km, carbon only; v_esc 3722 km/s surface, 5866 km/s centre) gives a centre/surface ratio 1.58, hence v_esc,c ≈ 9078 km/s for the benchmark.

Accessible splitting δ_max = μw²/2, w² = v_esc² + u² with u = ⟨1/v⟩⁻¹ = 288 km/s (`delta_max_by_body.csv`):
¹²C 2046 keV (surface) / 5073 keV (centre); ¹⁶O 2718 / 6739 keV; WimPyC WD ¹²C 857 / 2121 keV. **δ = 300–380 keV DM scatters freely in any white dwarf** (contrary to the assignment's suspicion), because μ_C ≈ 11 GeV at 1 TeV and w ≈ 0.02c.

Capture and heating. N_C = 9.98×10⁵⁵; geometric cross-section per carbon nucleus πR²/N_C = 2.01×10⁻³⁸ cm². Coherence factors σ_A/σ_nucleon = (μ_A/μ_n)²×{A² = 144 isoscalar; Z² = 36 proton-only; [(Zc_p+Nc_n)/c_n]² = 30.7 Higgsino} = 19 970, 4993, 4270. Capture requires an energy loss E_R + δ ≥ KE_∞ = 0.5 MeV (1 TeV, u = 288 km/s); with the recoil window at the surface speed, E_R ∈ [E₋, E₊] = [11.8, 7488] keV (δ = 300), [17.8, 7352] keV (366), and the Helm form factor of carbon (vectorised copy of `lz.helm_F2`, checked to 10⁻¹²), the F²-weighted capture probability per scatter is P_cap = 0.047 (δ = 0), 0.078 (300), 0.085 (350), 0.0875 (366), 0.090 (380) (`wd_capture_thresholds.csv`; δ helps because it removes kinetic energy). The saturation thresholds per nucleon are σ_crit = πR²/(N_C × coherence × P_cap): isoscalar 1.15×10⁻⁴¹, proton-only 4.6×10⁻⁴¹, Higgsino 5.4×10⁻⁴¹ cm² (δ = 366; 1.1–1.3×10⁻⁴¹ isoscalar over δ = 300–380). Thus the Higgsino (σ/σ_crit = 1.4×10⁵) and all fits with δ ≥ 350 keV are saturated (f ≥ 0.98), whereas the δ = 300 keV fits are partially captured (f = 0.15) and the δ = 250 keV dark photon f = 0.026 (`wd_heating_table.csv`).

Heating at f = 1: Ṁ = ρπR²[⟨v⟩ + v_esc²⟨1/v⟩] = 1.66×10⁴ g s⁻¹ (ρ = 0.4), L_kin = Ṁv_esc²/2 = 2.8×10²¹ erg s⁻¹ → T_kin = 50 K; with annihilation L = Ṁc² = 1.5×10²⁵ erg s⁻¹ → T_ann = 425 K. Both are far below the coolest known field WDs (T_eff ≈ 3000–4000 K, recalled/likely): white dwarfs in the solar neighbourhood carry no constraint. In the M4 globular cluster with an assumed ρ_χ = 10³ GeV cm⁻³ (recalled/uncertain; globular clusters may hold little DM): Ṅm = 4.1×10⁷ g s⁻¹, L_ann = 3.7×10²⁸ erg s⁻¹ (T = 3006 K), which is 0.3 × the faintest observed M4 WD luminosity 10⁻⁴·⁵ L_⊙ = 1.2×10²⁹ erg s⁻¹ (recalled/uncertain; Bertone & Fairbairn 2008). The M4 argument would constrain the corpus models only if ρ_χ(M4 core) ≳ 3×10³ GeV cm⁻³ and only via annihilation, not via kinetic heating (350 K).

## 6. The Sun

Profile: WimPyC `WD.Sun` (135 radial points; densities stored normalised to M/R³ — converted with M_⊙/R_⊙³ = 5.89 g cm⁻³; check: mass integral = 1.000 M_⊙). v_esc = 1373 km/s (centre), 618 km/s (surface).

δ_max = μw²/2 (surface / centre, u = 288 km/s): ¹H 2.4 / 10.3 keV; ⁴He 9.6 / 40.7; ¹⁶O 37.9 / 161; ⁵⁶Fe 128 / 542; ⁵⁸Ni 132 / 561 keV. Only iron-group nuclei in the core can up-scatter δ ≈ 300–380 keV DM: on ⁵⁶Fe the required w = √(2δ/μ) is 1044, 1127, 1153, 1174 km/s for δ = 300, 350, 366, 380 keV, met inside r ≤ 0.34, 0.27, 0.25, 0.23 R_⊙ (enclosing 70, 53, 49, 42 % of the solar mass, `sun_threshold_radius.csv`).

Capture rate. Own Gould-type integral, C = n_χ Σ_r N_i(r) Σ_u δη(u) σ_A c² [m_A/(2μ²)] ∫_{E_lo}^{E₊} F²(E) dE with E_lo = max(E₋, KE_∞(u) − δ) and Helm F², validated against WimPyC `wimp_capture` at σ = 10⁻⁴² cm², m = 1 TeV, ρ = 0.4: ⁵⁶Fe elastic 2.47×10²⁰ vs 2.42×10²⁰ s⁻¹ (2 %); ⁵⁶Fe δ = 300 keV 2.16×10²⁰ vs 2.14×10²⁰ (1 %); ¹H elastic 2.53×10¹⁸ vs 2.53×10¹⁸ (0.1 %). The inelastic rate is only 12 % below the elastic one because δ itself removes kinetic energy. WimPyC results for the LZ-fit couplings on ⁵⁶Fe + ⁵⁸Ni (`sun_capture_wimpyc.csv`; Higgsino c⁰ = c_p + c_n, c¹ = c_p − c_n with c_p = (G_F/√2)(1 − 4s_W²), c_n = −G_F/√2; dark photon c⁰ = c¹ = c_p; isoscalar via `lz.wd_c_SI_from_sigma_n`):

| model | δ [keV] | C_Fe [s⁻¹] | C_Ni [s⁻¹] | C/C_geom | Γ_A (thermal equilibrium) [s⁻¹] | t_eq [yr] | r_apo [R_⊙] | Γ_A (orbit-confined bracket) |
|---|---|---|---|---|---|---|---|---|
| Higgsino | 0 | 4.22e23 | 2.12e22 | 4.3e-4 | 2.2e23 | 3.9e6 | – | – |
| Higgsino | 300 | 3.68e23 | 2.00e22 | 3.8e-4 | 1.9e23 | 4.1e6 | 0.46 | 1.5e23 |
| Higgsino | 350 | 1.69e23 | 9.5e21 | 1.7e-4 | 8.9e22 | 6.1e6 | 0.61 | 2.5e22 |
| Higgsino | 366 | 1.24e23 | 7.1e21 | 1.3e-4 | 6.5e22 | 7.1e6 | 0.68 | 1.1e22 |
| Higgsino | 380 | 9.1e22 | 5.3e21 | 9.4e-5 | 4.8e22 | 8.3e6 | 0.75 | 4.7e21 |
| dark photon σ_p | 250 | 9.7e19 | 5.7e18 | 1.0e-7 | 5.2e19 | 2.5e8 | 0.36 | 5e16 |
| dark photon σ_p | 300 | 4.2e20 | 2.5e19 | 4.3e-7 | 2.2e20 | 1.2e8 | 0.46 | 4e17 |
| dark photon σ_p | 350 | 6.9e21 | 4.2e20 | 7.2e-6 | 3.7e21 | 3.0e7 | 0.61 | 5e19 |
| dark photon σ_p | 365 | 4.0e22 | 2.5e21 | 4.1e-5 | 2.1e22 | 1.3e7 | 0.67 | 1.3e21 |
| P021 isoscalar | 300 | 4.7e20 | 2.7e19 | 4.8e-7 | 2.5e20 | 1.2e8 | 0.46 | 6e17 |
| P021 isoscalar | 350 | 1.15e22 | 6.7e20 | 1.2e-5 | 6.1e21 | 2.3e7 | 0.61 | 1.4e20 |
| P021 isoscalar | 380 | 3.5e24 | 2.1e23 | 3.6e-3 | 1.9e24 | 1.3e6 | 0.75 | 1.7e24 |

C_geom = 1.02×10²⁷ s⁻¹ (WimPyC). Γ_A = ½C tanh²(t_⊙/t_eq) with WimPyC's Griest–Seckel thermal volume (V_eff = 2.0×10²⁶ cm³ at 1 TeV) and ⟨σv⟩ = 3×10⁻²⁶ cm³ s⁻¹: all models equilibrate (t_eq ≤ 2.5×10⁸ yr). Because inelastic-only DM stops scattering once its speed at the centre falls below √(2δ/μ) (loop-level elastic Higgsino scattering, σ ~ 10⁻⁴⁹ cm², is too weak to complete thermalisation; recalled/likely), we bracket the annihilation volume by the orbits it is left on, apoapsis r_a with v_esc(r_a)² = v_esc(0)² − 2δ/μ: r_a = 0.46–0.75 R_⊙, V = 4πr_a³/3, giving t_eq = 3.5×10⁹–1.4×10¹⁰ yr and the last column. Reference scale: the recalled/uncertain IceCube 1 TeV W⁺W⁻ solar limit σ_SD ~ 10⁻⁴⁰ cm² corresponds (WimPyC, elastic on H) to C = 2.5×10²⁰ s⁻¹, Γ_A ≈ 1.3×10²⁰ s⁻¹. The Higgsino at δ = 366 keV therefore annihilates in the Sun at (1–7)×10²² s⁻¹, ~10²–10³ above that scale, into W⁺W⁻/ZZ (hard neutrinos); the P021 isoscalar fit at δ = 380 keV even more (1.9×10²⁴, but its annihilation channel is unspecified); the dark-photon fits are near or below the scale and annihilate to A′ pairs that decay mostly to charged leptons/hadrons (few neutrinos). P076 is to treat the IceCube analysis in detail.

## 7. Robustness and validation checks

1. Kinetic efficiency: GR (1 − √B = 0.206) vs Newtonian (v²/2c² = 0.185): +11 % in L, +2.7 % in T. Capture area R_∞² vs R²: ×1.585 in Ṁ (T ×1.12).
2. Uniform-chord f versus 1 − e^{−τ̄}: irrelevant here (τ̄ ≥ 33).
3. Halo: v_star = 400 km/s lowers Ṁ by 29 % (T by 8 %); ρ = 0.3 vs 0.4: T ×0.93.
4. Sun: own Gould integral vs WimPyC 0.1–2 %; density-unit conversion verified by the mass integral (1.000 M_⊙); the first run without the conversion gave 0.17 M_⊙ and capture rates ×5.9 too low — corrected.
5. WD: P_cap uses the surface speed; at the centre speed the window is wider and P_cap smaller, but f is saturated for the Higgsino either way. A 1 M_⊙ WD with the realistic R ≈ 5800 km would raise v_esc by 17 % and σ_crit by ~×2, without changing any conclusion.
6. The dark-photon coupling in the neutron star is applied to protons only; the neutron's charge form factor at q ~ 0.7 GeV would add a comparable neutron coupling (not included; conservative).

## 8. Failed or abandoned approaches

* First Sun run used WimPyC densities as if in g cm⁻³ (they are normalised to M/R³): mass integral 0.17 M_⊙, capture ×5.9 low; fixed.
* `lz.helm_F2` is scalar-only; a vectorised copy was written (checked to 10⁻¹²).
* A first "orbit-confined volume" used the threshold radius r_thr instead of the apoapsis r_a; replaced by the correct energy argument.
* The literature JWST exposure numbers could not be reproduced beyond the order of magnitude because NIRCam sensitivities are recalled/uncertain.

## 9. Files

`results.json` (all numbers, recalled list), `ns_heating_table.csv`, `T_vs_sigma.csv`, `saturation_table.csv`, `chi2_fate.csv`, `nir_flux_table.csv`, `delta_max_by_body.csv`, `wd_capture_thresholds.csv`, `wd_heating_table.csv`, `sun_threshold_radius.csv`, `sun_capture_wimpyc.csv`, `run_log.txt`; figures `figures/P039_T_vs_sigma.png` (Fig. 1: T_∞ vs σ for neutron and proton-only couplings and with annihilation; σ_crit lines; corpus cross-sections marked) and `figures/P039_delta_max_by_body.png` (Fig. 2: δ_max = μw²/2 at surface and centre escape speeds for the NS, two WD models, five solar nuclei and the LZ xenon value 387 keV, with the 300–387 keV band).

## 10. References

LZ Collaboration, arXiv:2609.02823 (2026) · M. Baryakhtar, J. Bramante, S. W. Li, T. Linden, N. Raj, PRL 119, 131801 (2017) · N. Raj, P. Tanedo, H.-B. Yu, PRD 97, 043006 (2018) · N. F. Bell, G. Busoni, S. Robles, JCAP 09 (2018) 018 · G. Bertone, M. Fairbairn, PRD 77, 043515 (2008) · M. McCullough, M. Fairbairn, PRD 81, 083520 (2010) · S. Nussinov, L.-T. Wang, I. Yavin, JCAP 08 (2009) 037 · A. Menon, R. Morris, A. Pierce, N. Weiner, PRD 82, 015011 (2010) · A. Gould, ApJ 321, 571 (1987) · IceCube Collaboration, EPJC 77, 146 (2017) · WimPyDD/WimPyC: I. Jeong, S. Kang, S. Scopel, G. Tomar, CPC 276, 108342 (2022); WimPyC arXiv:2510.21185 · corpus: P002, P007, P011, P014, P021, P026.

## 11. Tools and provenance

Mirrors `output/provenance/P039.json`. Agent tools: Read (PAPER_GUIDE, dossier, ledger, P007/P011/P021/P012/P014/P026, lzcommon.py sections, WC_package.py capture routine, figures), Bash (directory listings, WimPyC probe, script runs), Write/Edit (script, details, provenance, paper), Skill (dataviz). Software: python 3.12.13, numpy 2.5.3, pandas 3.0.5, matplotlib 3.11.2, WimPyDD 2.0.4 incl. WimPyC (`wimp_capture`, `wimp_capture_geom`, `wimp_capture_annihilation`, `streamed_halo_function` via `lz.wd_halo`, celestial bodies Sun/White_Dwarf), common/lzcommon.py (`wd_halo`, `eta0`, `mu_red`, `E_R_range_keV`, `delta_max_kev`, `vmax_kms`, `v_earth_kms`, `helm_F2`, `wd_hamiltonian`, `wd_c_SI_from_sigma_n`, constants). WimPyDD-generated files: `WimPyDD/WimPyC/Response_functions/spin_1_2/{__init__.py, 1H_c_1_c_1.npy, 56Fe_c_1_c_1.npy, 58Ni_c_1_c_1.npy}`. Recalled knowledge: 17 items listed in results.json / provenance (constants certain; NS/WD benchmarks, Fermi momentum, proton fraction, form-factor suppression, kick velocities, 1700/2500 K literature temperatures, cooling ages, Bondi accretion, coolest WDs, IceCube scale and inelastic thermalisation stall: likely; JWST sensitivity, NS distances/density, M4 density and faintest WD luminosity: uncertain). Datasets: none. Data requests: none.
