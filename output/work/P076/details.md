# P076 — Solar capture of inelastic dark matter with δ ≈ 300 keV and the neutrino-telescope constraints on the LZ-fitted couplings

Research record (simulated date 2026-09-15; author profile: astroparticle / neutrino-telescope phenomenology group; category COMP; arXiv astro-ph.HE, cross-list hep-ph).
Script: `output/code/P076_solar_capture.py` (run from the simulation root with `.venv/bin/python`, 72 s; filtered stdout in `output/work/P076/run_log.txt`). Every number quoted below is printed by the script or stored in the CSV/JSON tables listed in §10.

## 1. Motivation and framework

The inelastic-DM readings of the LZ 248 keV event need splittings δ ≈ 250–380 keV that are barely accessible in the Galactic halo (δ_max = 387 keV on xenon in June, P002/P007). P039 pointed out that the Sun reverses the situation for heavy nuclei: a DM particle falling into the Sun has local speed w² = u² + v_esc(r)², with v_esc rising from 618 km/s at the surface to 1373 km/s at the centre, so iron-group nuclei in the core can up-scatter χ₁ → χ₂ with δ up to ~550 keV. P039 (with Fe/Ni only) found that the LZ-fitting Higgsino would annihilate in the Sun at (1–7)×10²² s⁻¹, "10²–10³ above the IceCube scale", and left the neutrino-telescope confrontation to us. We (i) recompute the capture with the full AGSS09ph composition (20 isotopes) and the v_esc(r) profile, (ii) add the two-step process (failed up-scatter followed by an exothermic χ₂ down-scatter before exit) and the fate of χ₂ (decay vs down-scatter), (iii) evaluate the annihilation rate both in Griest–Seckel equilibrium and for a population whose thermalisation stalls (inelastic-only DM stops scattering once its core speed drops below √(2δ/μ_Fe)), and (iv) translate recalled IceCube/ANTARES/Super-K elastic SI/SD limits into annihilation-rate limits using our own elastic capture rates, with channel factors for WW/ZZ (Higgsino), A′A′ → 4f (dark photon) and Z′Z′ → 4f (Z′).

Models and LZ-required couplings (all at m_χ = 1 TeV):
* **Higgsino** (P007): Z exchange, c_n = −G_F/√2 = −8.248×10⁻⁶ GeV⁻², c_p = (G_F/√2)(1 − 4 sin²θ_W) = 6.196×10⁻⁷ GeV⁻², σ_n = 7.4×10⁻³⁹ cm² fixed; nucleus factor f_A = ((Z c_p + N c_n)/c_n)² = 786.6 (⁵⁶Fe), 778.2 (⁵⁸Ni), 0.0056 (H).
* **Dark photon** (P011): proton-only, f_A = Z², σ_p(N = 1) interpolated in log from `output/work/P011/sigma_p_required.csv` (1 TeV, 'annual' label = Sun-frame halo per P035, σ_eff = 8 keV): 1.20×10⁻⁴² (250), 8.58×10⁻⁴² (300), 2.97×10⁻⁴⁰ (350), 2.27×10⁻³⁹ (365), 5.49×10⁻³⁸ cm² (380 keV).
* **Z′** (P054): isoscalar c_p = c_n = G, f_A = A², σ_n(N = 1) from `output/work/P054/lz_requirement.csv` (1 TeV): 2.83×10⁻⁴³ (250), 2.18×10⁻⁴² (300), 1.12×10⁻⁴⁰ (350), 1.39×10⁻³⁹ (366), 4.33×10⁻³⁸ cm² (380 keV). Below 250 keV the grid is extrapolated flat (the Z′ curve at δ < 250 keV is therefore only indicative).

## 2. Inputs

* **Solar profile**: WimPyC `Sun` celestial body (WimPyDD 2.0.4), which is the AGSS09ph standard solar model (Serenelli et al. 2009; the file header in `WC_package.py` line 988 names it): 135 radial points, 20 isotopes (¹H, ³He, ⁴He, ¹²C, ¹³C, ¹⁴N, ¹⁵N, ¹⁶O, ¹⁷O, ¹⁸O, ²⁰Ne, ²³Na, ²⁴Mg, ²⁷Al, ²⁸Si, ³²S, ⁴⁰Ar, ⁴⁰Ca, ⁵⁶Fe, ⁵⁸Ni), v_esc(r) from 1373 km/s (centre) to 618 km/s (surface). Densities are stored normalised to M_⊙/R_⊙³ = 5.89 g cm⁻³ (P039's finding) and converted back: ρ_c = 143.6 g cm⁻³, mass integral 1.0000 M_⊙, T_c = 1.4×10⁷ K. Integrated mass fractions: H 0.682, He 0.302, O 6.4×10⁻³, C 2.0×10⁻³, N 1.5×10⁻³, Fe 1.45×10⁻³, Ne 1.4×10⁻³, Mg 7.9×10⁻⁴, Si 7.5×10⁻⁴, S 3.5×10⁻⁴, Ni 8.0×10⁻⁵, Ar 7.5×10⁻⁵, Ca 7.2×10⁻⁵ (`results.json`). Reliability: the SSM itself is a published model (recalled/likely that WimPyC reproduces it faithfully; the mass integral and the P039 threshold radii check out).
* **Halo**: `lz.wd_halo()` Sun-frame Baxter-2021 SHM (v₀ = 238, v_esc = 544 km/s, solar peculiar motion), 1199 streams (u, δη): ⟨1/v⟩⁻¹ = 287.9 km/s, ⟨v⟩ = 355.5 km/s, halo edge u_max = 795 km/s. The Sun frame is the correct frame for solar capture (no Earth motion). ρ_χ = 0.4 GeV cm⁻³ (as P039; 0.3 shown as a variation; recalled/likely).
* **Constants**: G_F = 1.1664×10⁻⁵ GeV⁻², sin²θ_W = 0.2312 (recalled/certain), lzcommon `AMU_GEV`, `M_NUCLEON_GEV`, `HBARC_GEV_FM`, `GEV_TO_CM2`, `C_KMS`.
* **Corpus**: P039 `details.md`, `sun_capture_wimpyc.csv`, `sun_threshold_radius.csv`, `delta_max_by_body.csv`; P011 `sigma_p_required.csv`, `epsilon_required.csv`, `chi2_lifetime.csv`, `couplings.json`; P054 `lz_requirement.csv`, `chi2_lifetime.csv`; papers P007, P011, P021, P025, P026, P039, P054.

## 3. Kinematics: which nuclei can up-scatter δ ≈ 300 keV DM

For a particle of asymptotic speed u at radius r, w² = u² + v_esc(r)², and the recoil window for χ₁ A → χ₂ A is

  E_± = (μ² w²/m_A) [1 − δ/(μ w²) ± √(1 − 2δ/(μ w²))],  δ_max(A, r) = μ w²/2.

`delta_max_by_isotope.csv` (u = 288 km/s unless stated), centre / surface:

| isotope | δ_max centre [keV] | δ_max surface [keV] | centre, u_max = 795 km/s |
|---|---|---|---|
| ¹H | 10.3 | 2.4 | 13.1 |
| ⁴He | 40.7 | 9.6 | 52 |
| ¹²C | 121 | 28.6 | 155 |
| ¹⁴N | 141 | 33.3 | 180 |
| ¹⁶O | 161 | 37.9 | 206 |
| ²⁰Ne | 200 | 47.2 | 256 |
| ²⁴Mg | 239 | 56.5 | 306 |
| ²⁸Si | 278 | 65.6 | 356 |
| ³²S | 317 | 74.7 | 405 |
| ⁴⁰Ar, ⁴⁰Ca | 393 | 92.7 | 503 |
| ⁵⁶Fe | 542 | 128 | 694 |
| ⁵⁸Ni | 561 | 132 | 717 |

P039's H/He/O/Fe/Ni values (10/41/161/542/561 keV) are reproduced exactly. New: for δ = 250 keV, ⁴⁰Ar/⁴⁰Ca (inside r ≤ 0.28 R_⊙), ³²S (≤ 0.18) and ²⁸Si (≤ 0.11 R_⊙) also contribute; for δ ≥ 330 keV only Ar/Ca (r ≤ 0.15 R_⊙) join Fe/Ni; with the fastest halo particles Mg and Si reach 306–356 keV at the very centre. Threshold radii for ⁵⁶Fe (`threshold_radius.csv`): the required w = √(2δ/μ) = 953, 1044, 1127, 1153, 1174 km/s at δ = 250, 300, 350, 366, 380 keV is met inside r ≤ 0.441, 0.344, 0.270, 0.255, 0.233 R_⊙, enclosing 84, 70, 53, 49, 42 % of the solar mass (P039's 0.344/0.270/0.255/0.233 confirmed). Hydrogen, helium and the CNO group never reach the LZ splittings anywhere in the Sun.

## 4. Capture rate

Gould-type capture summed over streams and shells (dσ/dE_R = σ_A m_A F²(E_R)/(2μ² w²); the factor w² cancels against the flux):

  C = n_χ Σ_A Σ_r N_A(r) Σ_u δη(u) σ_A c² [m_A/(2μ_A²)] ∫_{E_lo}^{E_+} F²(E) dE,  E_lo = max(E_−, m_χu²/2 − δ),

with σ_A = σ_ref f_A (μ_A/μ_N)² and the Helm form factor (vectorised copy of `lz.helm_F2`; F = 1 for hydrogen), 48-point energy grids, n_χ = ρ_χ/m_χ. The capture condition E_R + δ ≥ m_χu²/2 uses the fact that the splitting itself removes kinetic energy.

**Validation** (`validation.json`): at σ = 10⁻⁴² cm², isoscalar, 1 TeV, ρ = 0.4 our integral gives H elastic 2.579×10¹⁸, ⁵⁶Fe elastic 2.484×10²⁰, ⁵⁶Fe δ = 300 keV 2.172×10²⁰ s⁻¹, versus live WimPyC `wimp_capture` 2.535×10¹⁸, 2.421×10²⁰, 2.136×10²⁰ (+1.8, +2.6, +1.7 %; WimPyC uses shell-model response functions, we use Helm). Our geometric rate n_χπR_⊙²[⟨v⟩ + v_esc,s²⟨1/v⟩] = 1.0239×10²⁷ s⁻¹ equals WimPyC's `wimp_capture_geom` to 5 digits.

**Results** (`capture_vs_delta.csv`, 1 TeV, ρ = 0.4; C_el = elastic capture with the same coupling):

| model | δ [keV] | σ_ref [cm²] | C [s⁻¹] | Fe / Ni / Ca+Ar / S / Si share | C/C_el |
|---|---|---|---|---|---|
| Higgsino | 0 | 7.4e-39 | 4.10e24 | O 32 %, He 12 %, Ne 11 %, Fe 11 % | 1 |
| Higgsino | 250 | 7.4e-39 | 8.08e23 | 83 / 4.3 / 5.5 / 4.7 / 2.0 % | 0.197 |
| Higgsino | 300 | 7.4e-39 | 4.43e23 | 91 / 4.8 / 4.0 / 0.4 / 0 % | 0.108 |
| Higgsino | 350 | 7.4e-39 | 2.02e23 | 93 / 5.1 / 1.4 / 0 / 0 % | 0.049 |
| Higgsino | 366 | 7.4e-39 | 1.48e23 | 94 / 5.2 / 0.7 % | 0.036 |
| Higgsino | 380 | 7.4e-39 | 1.09e23 | 94 / 5.3 / 0.2 % | 0.027 |
| dark photon | 250 | 1.20e-42 | 1.17e20 | 80 / 4.9 / 5.8 / 6.2 / 2.7 % | 0.156 |
| dark photon | 300 | 8.58e-42 | 4.49e20 | 90 / 5.5 / 4.3 / 0.5 % | 0.084 |
| dark photon | 350 | 2.97e-40 | 7.06e21 | 93 / 5.9 / 1.6 % | 0.038 |
| dark photon | 365 | 2.27e-39 | 4.02e22 | 93 / 6 / 0.8 % | 0.028 |
| dark photon | 380 | 5.49e-38 | 7.06e23 | 94 / 6 / 0.2 % | 0.021 |
| Z′ | 250 | 2.83e-43 | 1.26e20 | 82 / 4.6 / 5.6 / 5.5 / 2.3 % | 0.175 |
| Z′ | 300 | 2.18e-42 | 5.25e20 | 90 / 5.2 / 4.2 / 0.4 % | 0.095 |
| Z′ | 350 | 1.12e-40 | 1.22e22 | 93 / 5.5 / 1.5 % | 0.043 |
| Z′ | 366 | 1.39e-39 | 1.11e23 | 94 / 5.6 / 0.7 % | 0.032 |
| Z′ | 380 | 4.33e-38 | 2.57e24 | 94 / 5.8 / 0.2 % | 0.023 |

Comparison with P039 (Fe+Ni only, WimPyC): Higgsino 3.88×10²³ (300), 1.79×10²³ (350), 1.31×10²³ (366), 9.6×10²² (380) versus our Fe+Ni 4.23, 1.99, 1.47, 1.09×10²³ — we are 9–13 % higher, the Helm-vs-shell-model difference for the isovector-dominated Higgsino coupling (for the isoscalar test point the difference is 1.7 %); the additional isotopes add a further 4 % at 300 keV and 12 % at 250 keV. Dark photon: P039 1.03×10²⁰ (250), 4.45×10²⁰ (300), 7.3×10²¹ (350), 4.25×10²² (365) versus ours 1.17, 4.49, 70.6, 402×10²⁰ (1–14 %). **P039's numbers are confirmed; the full element list changes them by ≤ 15 %.**

The inelastic suppression C(δ)/C_el(same coupling) is 0.20 → 0.03 from 250 to 380 keV (Fig. 3, right) — not the 10⁻⁵ suppression LZ suffers, because the solar core supplies w ≈ 1000–1400 km/s. Relative to a *geometric* Sun, C/C_geom = 4.3×10⁻⁴ (Higgsino, 300 keV) down to 10⁻⁷ (dark photon, 250 keV): the Sun is optically thin for all models, so single-scatter capture is the right description.

Variations (`variations.csv`): ρ_χ = 0.3 gives ×0.75; Higgsino at its P007/P011 δ(N = 1) points: m_χ = 300 GeV, δ = 310 keV: C = 2.00×10²⁴; 1 TeV, 366 keV: 1.48×10²³; 3 TeV, 377 keV: 1.67×10²² s⁻¹ (∝ 1/m_χ times the shrinking kinematic window).

## 5. Two-step capture and the fate of χ₂

*Failed up-scatters.* Up-scatters with E_R + δ < m_χu²/2 leave a χ₂ that is still unbound. Their rate is 1.1–1.5 × C (`two_step_correction.csv`): most up-scatters by fast halo particles (u ≳ 500 km/s, KE_∞ ≳ 1.4 MeV) do not capture. Such a χ₂ can down-scatter exothermically (window with −δ, no threshold) on any nucleus along its outward path; it is captured if E_R′ − δ ≥ ΔE, the residual excess energy. We compute P₂ = 1 − exp[−Σ_A N_A^col(r) σ_A^cap(w, ΔE)] with the outward radial column densities N_A^col(r) = ∫_r^R n_A dr′ and the F²-weighted distribution of ΔE over the failed window, on a halo coarse-grained by 8 (first-step rate agrees with the fine one to 5 %). Result: the two-step channel adds 0.55–0.59 % to the Higgsino capture (δ = 300–366 keV) and ≤ 0.04 % for the dark photon and Z′ (their cross-sections are 10³–10⁴ smaller, so the second-scatter optical depth is negligible). Hydrogen never helps: E_+′(H) ≈ 2μ_H²w²/m_H ≈ 25 keV ≪ δ, so a down-scatter on H *returns* energy to the DM. The assignment's expectation that down-scattering "helps capture" is therefore correct in sign but numerically irrelevant (< 1 %).

*Fate of a captured χ₂* (`chi2_fate.csv`). Transit time 2R_⊙/w ≈ 1400 s. Higgsino: τ_γ ≈ 0.06 s (P014/P026) → decay length 60 km = 8.6×10⁻⁵ R_⊙: χ₂ decays to χ₁γ in flight before it can down-scatter (the exothermic down-scatter time at the centre would be 345 s); τ_νν̄ = 1.2×10⁶ s would instead let it down-scatter first. Dark photon (τ ≥ 7×10¹⁶ s) and Z′ (B−L, 1.7×10¹⁰ s): the bound χ₂ down-scatters on core nuclei within 2.5×10⁵ s, i.e. within one solar orbit. Either way the released δ (≤ 380 keV) cannot unbind the particle: the binding energy m_χv_esc²/2 is 10.5 MeV at the centre and 4.2 MeV at 0.5 R_⊙. Decay-in-flight does not change the capture energetics (the up-scatter already removed E_R + δ; the decay photon/neutrinos carry away δ).

*Thermalisation.* After capture, χ₁ can lose energy only by further up-scatters, which require a core speed above √(2δ/μ_Fe) = 1044 (300 keV) – 1174 km/s (380 keV). Once the orbit's core speed drops below this, the particle stops scattering and stays on an orbit with apoapsis r_a given by v_esc(r_a)² = v_esc(0)² − 2δ/μ_Fe: r_a = 0.36, 0.46, 0.61, 0.68, 0.75 R_⊙ at δ = 250, 300, 350, 366, 380 keV (as P039). Loop-level elastic scattering (σ ~ 10⁻⁴⁹ cm² for the Higgsino, P007) cannot complete thermalisation over the solar age (recalled/likely, Nussinov–Wang–Yavin 2009; Menon et al. 2010). We therefore bracket the annihilation volume between the Griest–Seckel thermal volume and 4πr_a³/3.

## 6. Annihilation rate and equilibrium

Γ_A = ½ C tanh²(t_⊙/t_eq), t_eq = (C C_A)⁻¹ᐟ², C_A = ⟨σv⟩/V_eff. Thermal volume (WimPyC's Griest–Seckel formula with T_c = 1.4×10⁷ K, ρ_c = 143.6 g cm⁻³): V_eff = 1.98×10²⁶ cm³ (r_th = 0.0052 R_⊙). Orbit-confined volume: 4πr_a³/3 = 6.6×10³¹–5.9×10³² cm³. ⟨σv⟩ brackets: Higgsino 9.6×10⁻²⁷ × (1–3) (P025 WW+ZZ with Sommerfeld); dark photon 4.4×10⁻²⁶ × S, S = 1–23 (P025's Planck-allowed saturation for m_A′ ≥ 9 GeV); Z′ 2.2×10⁻²⁶ × (1–2.2) (P054 χχ → Z′Z′ with S₀). Evaporation: 3kT_c = 3.6 keV against a 10.5 MeV well — irrelevant at 1 TeV (evaporation mass ~3–4 GeV, recalled/likely).

`annihilation.csv`:

| model | δ | C | t_eq thermal [yr] | Γ_A thermal | r_a | t_eq orbit [yr] | Γ_A orbit |
|---|---|---|---|---|---|---|---|
| Higgsino | 300 | 4.43e23 | (3.9–6.8)e6 | 2.21e23 | 0.46 | (3.3–5.8)e9 | (0.97–1.7)e23 |
| Higgsino | 350 | 2.02e23 | (5.8–10)e6 | 1.01e23 | 0.61 | (0.75–1.3)e10 | (1.2–3.0)e22 |
| Higgsino | 366 | 1.48e23 | (6.8–12)e6 | 7.4e22 | 0.68 | (1.0–1.8)e10 | (0.48–1.3)e22 |
| Higgsino | 380 | 1.09e23 | (7.9–14)e6 | 5.5e22 | 0.75 | (1.4–2.4)e10 | (0.20–0.58)e22 |
| dark photon | 250 | 1.17e20 | (0.4–2)e8 | 5.9e19 | 0.36 | (2.4–11)e10 | 9.7e16–2.2e18 |
| dark photon | 300 | 4.49e20 | (0.2–1)e8 | 2.25e20 | 0.46 | (1.8–8.5)e10 | 6.6e17–1.5e19 |
| dark photon | 350 | 7.06e21 | (0.5–2.5)e7 | 3.5e21 | 0.61 | (0.7–3.2)e10 | 7.0e19–1.2e21 |
| dark photon | 366 | 4.70e22 | (0.2–1)e7 | 2.35e22 | 0.68 | (0.3–1.5)e10 | 2.2e21–1.9e22 |
| Z′ | 300 | 5.25e20 | (0.9–1.3)e8 | 2.6e20 | 0.46 | (0.7–1.1)e11 | (4.6–10)e17 |
| Z′ | 350 | 1.22e22 | (1.8–2.7)e7 | 6.1e21 | 0.61 | (2.3–3.5)e10 | (1.1–2.4)e20 |
| Z′ | 366 | 1.11e23 | (0.6–0.9)e7 | 5.6e22 | 0.68 | (0.9–1.4)e10 | (0.6–1.2)e22 |

All models reach equilibrium if thermalised (t_eq ≤ 3×10⁸ yr ≪ 4.6 Gyr); the orbit-confined population does not (t_eq = 3×10⁹–10¹¹ yr), which lowers Γ_A by ×1.3–2.3 (Higgsino, 300 keV) to ×10³ (dark photon/Z′, 250–300 keV). The Higgsino at δ = 366 keV gives Γ_A = (0.5–7.4)×10²² s⁻¹, bracketing P039's (1–7)×10²².

## 7. Neutrino-telescope limits translated to Γ_A

Neutrino telescopes quote σ_SD (capture on hydrogen) or σ_SI (coherent on all nuclei) assuming elastic scattering and equilibrium (Γ_A = C/2). We convert with our own elastic capture rates at the same mass (`elastic_conversion.csv`): at 1 TeV, C_SD(H, 10⁻⁴⁰ cm²) = 2.540×10²⁰ s⁻¹ (P039/WimPyC: 2.5×10²⁰) and C_SI(all nuclei, 10⁻⁴³ cm²) = 2.540×10²⁰ s⁻¹ — the two recalled limit scales (σ_SD ~ 10⁻⁴⁰, σ_SI ~ 10⁻⁴³) map onto the *same* Γ_A, a useful self-consistency check of the recall. At 500 GeV: 1.017×10²¹ and 8.89×10²⁰; at 300 GeV: 2.81×10²¹ and 2.08×10²¹.

Recalled limits (all **uncertain**, bracketed by a factor ~3 either way; `neutrino_limits.json`): IceCube 2016–2022 solar searches (IC79/86, 3–7 yr), 1 TeV W⁺W⁻: σ_SD = 3×10⁻⁴¹ [10⁻⁴¹–10⁻⁴⁰], σ_SI = 10⁻⁴³ [3×10⁻⁴⁴–3×10⁻⁴³]; τ⁺τ⁻ similar; bb̄ ~30× weaker (10⁻³⁹); ANTARES 11 yr W⁺W⁻ ~10⁻⁴⁰ [5×10⁻⁴¹–3×10⁻⁴⁰]; Super-K reaches only m ≤ 200 GeV and is irrelevant at 1 TeV. Resulting Γ_A limits (central [bracket]):

* WW (and ZZ, τ⁺τ⁻) at 1 TeV: 3.8×10¹⁹ [1.3×10¹⁹–1.3×10²⁰] s⁻¹ (ANTARES: 1.3×10²⁰ [0.6–3.8×10²⁰]).
* bb̄-like at 1 TeV: 1.3×10²¹ [0.4–3.8×10²¹].
* **Higgsino**: χ₁χ₁ → W⁺W⁻ (54 %) + ZZ (46 %) (P025) → WW limit directly.
* **Dark photon**: χ₁χ₁ → A′A′, each A′ carrying E = m_χ. For the Planck-allowed m_A′ ≥ 9 GeV (P025) the open channels give BR(A′ → τ⁺τ⁻) ≈ 1/6.5 = 0.15 (Σ N_c Q_f²: e, μ, τ = 3; u, c = 2.67; d, s = 0.67; b = 0.33 above 10.5 GeV). Muons (range ≈ 20 m in the 150 g cm⁻³ core versus a 3×10⁶ m decay length), pions and kaons stop before decaying and give only ~30 MeV neutrinos; light-quark jets likewise; only τ (and, softer, c/b semileptonic decays, ignored) give hard neutrinos. Two A′ with 0.15 τ-pairs each ≈ 0.3 τ-pairs per annihilation, each pair sharing E = m_χ, i.e. the spectrum of a 500 GeV τ⁺τ⁻ annihilation: Γ_lim = Γ_lim(ττ, 500 GeV)/(2 × 0.15) = 5.0×10²⁰ [1.7×10²⁰–1.7×10²¹] s⁻¹. **The A′ decays inside the Sun for every P011 point** (`dark_photon_decay_length.csv`): with ε fixed by σ_p(N = 1) = 8.6×10⁻⁴² cm² at δ = 300 keV (ε = 8.3×10⁻⁷ at m_A′ = 1 GeV, α_D = 0.1 — P011: 8.7×10⁻⁷; 6.7×10⁻⁵ at 9 GeV; 1.5×10⁻³ at 30 GeV with α_D = 0.0245) the boosted decay length γcτ is 30 m (1 GeV) down to 6×10⁻⁶ cm (30 GeV), i.e. ≤ 4×10⁻⁸ R_⊙; escape (L = R_⊙) would need ε ≤ 1.7×10⁻¹⁰, four orders below P011's χ₂-decay floor ε ≥ 5.4×10⁻⁶. Secluded "escaping mediator" scenarios are thus not available to the LZ-fitted dark photon.
* **Z′**: in the Sun only χ₁ is present (χ₂ has down-scattered), so the off-diagonal s-channel χ₁χ₂ → Z′* → ff̄ is absent and the annihilation is χ₁χ₁ → Z′Z′ (P054's light-Z′ regime, g_χ ≈ 0.7). B−L Z′ decays: BR(νν̄) = 1.5/6.5 = 23 %, BR(τ⁺τ⁻) = 15 %, quarks 31 %: hard neutrinos; we weight the νν̄ line as 3× a τ pair (recalled/uncertain) → Γ_lim = Γ_lim(ττ, 500 GeV)/(2 × (3 × 0.23 + 0.15)) = 9.0×10¹⁹ [3×10¹⁹–3×10²⁰]. U(1)_B Z′ decays only to quarks (u, d, s, c, b equally): bb̄-like at 500 GeV, Γ_lim = 2.5×10²¹ [0.8–7.6×10²¹].

The stalled-orbit population annihilates at r ~ 0.4–0.7 R_⊙ rather than at the centre; neutrino absorption in the Sun is then weaker than in the telescopes' central-source assumption (their limits become slightly *stronger*), and the angular extent (≤ 0.2°) remains inside the Sun's disc. We ignore both effects.

## 8. Confrontation (`confrontation.csv`, Fig. 1)

Classification: **excluded** if even the orbit-confined Γ_A with the low ⟨σv⟩ exceeds the weakest bracket of the limit; **excluded if thermalised** if only the equilibrium Γ_A = C/2 does; **marginal** if C/2 lies inside the limit bracket; **unconstrained** if C/2 is below the strongest bracket.

| model | δ [keV] | Γ_A range [s⁻¹] | Γ_lim (central) | ratio | verdict |
|---|---|---|---|---|---|
| Higgsino | 250 | (3.4–4.0)e23 | 3.8e19 | (0.9–1.1)e4 | excluded |
| Higgsino | 300 | (0.97–2.2)e23 | 3.8e19 | (2.6–5.8)e3 | excluded |
| Higgsino | 350 | (1.2–10)e22 | 3.8e19 | (3.1–26)e2 | excluded |
| Higgsino | 366 | (0.48–7.4)e22 | 3.8e19 | (1.2–19)e2 | excluded |
| Higgsino | 380 | (0.20–5.5)e22 | 3.8e19 | 53–1400 | excluded |
| dark photon | 250 | 9.7e16–5.9e19 | 5.0e20 | 2e-4–0.12 | unconstrained |
| dark photon | 300 | 6.6e17–2.3e20 | 5.0e20 | 1.3e-3–0.45 | marginal |
| dark photon | 340 | 1.7e19–1.6e21 | 5.0e20 | 0.03–3.1 | marginal |
| dark photon | 350 | 7.0e19–3.5e21 | 5.0e20 | 0.14–7.1 | excluded if thermalised |
| dark photon | 366 | 2.2e21–2.4e22 | 5.0e20 | 4.4–47 | excluded |
| dark photon | 380 | (2.2–3.5)e23 | 5.0e20 | 440–710 | excluded |
| Z′ (B−L) | 250 | 5.6e16–6.3e19 | 9.0e19 | 6e-4–0.70 | marginal |
| Z′ (B−L) | 300 | 4.6e17–2.6e20 | 9.0e19 | 5e-3–2.9 | marginal |
| Z′ (B−L) | 350 | 1.1e20–6.1e21 | 9.0e19 | 1.2–68 | excluded if thermalised |
| Z′ (B−L) | 366 | (0.6–5.6)e22 | 9.0e19 | 66–620 | excluded |
| Z′ (U(1)_B) | ≤ 320 | ≤ 6.3e20 | 2.5e21 | ≤ 0.25 | unconstrained |
| Z′ (U(1)_B) | 330–350 | ≤ 6.1e21 | 2.5e21 | 0.5–2.4 | marginal |
| Z′ (U(1)_B) | 360–366 | (0.1–5.6)e22 | 2.5e21 | 8–22 | excluded if thermalised |
| Z′ (U(1)_B) | ≥ 370 | ≥ 2.3e22 | 2.5e21 | ≥ 46 | excluded |

Boundaries: the Higgsino is excluded at every δ from 200 to 380 keV, by ≥ 15× (380 keV) to ≥ 770× (300 keV) even against the *weakest* bracket with the *stalled* population, and by 120–1900× at the P007 best fit δ = 366 keV against the central limit; at its 300 GeV (δ = 310 keV) and 3 TeV (δ = 377 keV) fits C = 2.0×10²⁴ and 1.7×10²² s⁻¹, still ≥ 100× above the corresponding limits (the 300 GeV limit is ~10× stronger in Γ_A). The dark photon is unconstrained for δ ≤ 280 keV, marginal for 290–340 keV, excluded if thermalised for 350–365 keV and excluded outright for δ ≥ 366 keV. The B−L Z′ is marginal up to 300 keV, excluded-if-thermalised for 310–350 keV, excluded for ≥ 360 keV; the leptophobic U(1)_B Z′ is unconstrained up to 320 keV, marginal 330–350, excluded-if-thermalised 360–366 and excluded ≥ 370 keV. P054's viable Z′ window (δ ≲ 340 keV for U(1)_B, ≲ 300 keV for B−L) therefore sits in the marginal/unconstrained region, and P021's isoscalar preference δ = 360–385 keV is excluded for every model in which χ₁χ₁ annihilates with ⟨σv⟩ ≳ 10⁻²⁶ cm³ s⁻¹ into anything with a τ, W, Z or ν content.

## 9. Robustness, caveats and failed approaches

1. Helm versus shell-model form factors: +1.7–2.6 % (isoscalar), +9–13 % (Higgsino isovector) relative to WimPyC — far below the 10²–10³ margins.
2. Halo: Sun-frame SHM is the correct frame; the annual-average or June halos used by LZ-fitting papers only redefine the *coupling* grids (P011/P054 'annual' = Sun frame per P035), which we take as given.
3. ρ_χ = 0.3: ×0.75 in C and Γ_A.
4. The two-step channel is < 0.6 %; multiple scattering is irrelevant (C/C_geom ≤ 4×10⁻⁴).
5. The thermalisation question is the dominant *physics* uncertainty (×1.3 to ×10³, model- and δ-dependent) and is bracketed explicitly; the Higgsino verdict does not depend on it.
6. The recalled neutrino-telescope limits carry a factor ~3 uncertainty each way; the channel factors (τ yield of A′ and Z′ decays, νν̄ weight 3, bb̄ ×30) another factor ~2–3. These are bracketed and only move the marginal boundaries by ±10–20 keV in δ.
7. Sommerfeld factors are bracketed as in P025/P054; the stalled population moves at v ~ 3×10⁻³c where the enhancement is smaller than at thermal speeds — the low end of our bracket.
8. Coarse-grained halo in the two-step routine: first-step rate agrees with the fine calculation to 5 %.
9. Failed approaches: (a) the first script computed the elastic reference by re-running the full integral for every (model, δ) — replaced by a single unit-cross-section run per model (linear scaling); (b) the first two-step implementation built (r × u × E × E′) arrays of ~10⁸ doubles and the process was killed (SIGTERM) — replaced by an 8-fold stream coarse-graining and 16-point grids; (c) a dict-key syntax error (`C_SD_H_at_1e-40=`) fixed; (d) a broadcasting error in the exothermic window fixed.

## 10. Files and figures

`capture_vs_delta.csv` (C per isotope and C/C_el for 3 models × 22 δ), `delta_max_by_isotope.csv`, `threshold_radius.csv`, `validation.json`, `variations.csv`, `two_step_correction.csv`, `chi2_fate.csv`, `annihilation.csv`, `elastic_conversion.csv`, `neutrino_limits.json`, `dark_photon_decay_length.csv`, `confrontation.csv`, `results.json` (all of the above plus the recalled list), `run_log.txt`.
Figures: `figures/P076_GammaA_vs_delta.png` (Fig. 1: Γ_A(δ) band from orbit-confined to thermal equilibrium, C/2 line, recalled limit bracket, per model); `figures/P076_delta_max_by_isotope.png` (Fig. 2: δ_max surface → centre per isotope, with the halo-edge value and the 250–380 keV band); `figures/P076_composition_and_ratio.png` (Fig. 3: isotope shares of the Higgsino capture versus δ; C_inel/C_el for the three models).

## 11. References

LZ Collaboration, arXiv:2609.02823 (2026) · A. Gould, ApJ 321, 571 (1987) · K. Griest, D. Seckel, Nucl. Phys. B 283, 681 (1987) · S. Nussinov, L.-T. Wang, I. Yavin, JCAP 08 (2009) 037 · A. Menon, R. Morris, A. Pierce, N. Weiner, PRD 82, 015011 (2010) · B. Batell, M. Pospelov, A. Ritz, Y. Shang, PRD 81, 075004 (2010) · IceCube Collaboration, EPJC 77, 146 (2017) · ANTARES Collaboration, PLB 759, 69 (2016) · Super-Kamiokande Collaboration, PRL 114, 141301 (2015) · A. M. Serenelli, S. Basu, J. W. Ferguson, M. Asplund, ApJL 705, L123 (2009) · G. Jungman, M. Kamionkowski, K. Griest, Phys. Rep. 267, 195 (1996) · WimPyDD/WimPyC: I. Jeong, S. Kang, S. Scopel, G. Tomar, CPC 276, 108342 (2022); arXiv:2510.21185 · corpus: P007, P011, P021, P025, P026, P039, P054 (also P002, P014, P035 via those).

## 12. Tools and provenance

Mirrors `output/provenance/P076.json`. Agent tools: Read (PAPER_GUIDE; P039 details.md and paper; P007/P011/P021/P025/P026/P054 papers; P039 script lines 285–410; three figures, twice), Bash (directory listings, ledger extraction, CSV dumps, WimPyC probes, WC_package docstring, 8 script runs, table printing), Write (script, details, provenance, paper), Edit (7 script fixes), Skill dataviz (palette/mark rules; JS validator skipped per PAPER_GUIDE). Software: python 3.12.13, numpy 2.5.3, pandas 3.0.5, matplotlib 3.11.2, WimPyDD 2.0.4 incl. WimPyC (`Sun` celestial body = AGSS09ph profile, `wimp_capture`, `wimp_capture_geom` for validation), common/lzcommon.py (`wd`, `wd_halo`, `wd_hamiltonian`, `wd_c_SI_from_sigma_n`, `mu_red`, `m_nucleus_gev`, `helm_F2` (vectorised copy), constants). WimPyDD-generated files: none new (the response-function files for 1H/56Fe with c₁ were already created by P039; the validation call re-used them). Recalled knowledge: 14 items (12 registered by the script plus the thermalisation-stall argument and the corpus χ₂ lifetimes; constants certain; SSM fidelity, ρ_χ, Sommerfeld brackets, evaporation mass, muon/pion stopping, Super-K reach, stall, lifetimes: likely; IceCube/ANTARES limits, channel factors, νν̄ weight: uncertain). Agent-tool counts: Read 15, Bash 21, Write 4, Edit 12, Skill 1. Datasets: none. Data requests: none.
