# P042 — The de-excitation photon: what the excited state χ₂ must do after scattering in LZ, and what a single-site event already implies

Simulated date 2026-09-10 · IDM · hep-ph (cross-list physics.ins-det) · author profile: phenomenologists thinking about detector signatures.
Script: `output/code/P042_deexcitation_photon.py` (runs in 114–125 s from the simulation root; seeded MC, fully reproducible). Log: `P042_run.log`.

## 1. Motivation and framework

Every inelastic-DM reading of the LZ event (P002, P007, P011, P021, P023, P037) has the scatter χ₁ Xe → χ₂ Xe leave an excited state χ₂ at the vertex. With δ ≈ 250–390 keV, χ₂ must eventually return to χ₁ by emitting a photon (χ₂ → χ₁γ, E_γ = δ up to a negligible recoil), a neutrino pair (χ₂ → χ₁νν̄), or — only if δ > 2m_e = 1022 keV, which is not the case here (P014) — an e⁺e⁻ pair. The corpus quotes lifetimes ranging from 66 μs (photon-M1 MiDM, P023) through 0.06 s (Higgsino loop dipole, P014) to 10¹⁸ s (dark-photon νν̄ channel, P011). Nothing in the LZ paper constrains the lifetime directly, but the paper does describe the event topology in detail: a single scatter with an S2 pulse shape consistent with a point-like interaction at the reconstructed depth, nothing else in the event, and no veto signal (science sample). This paper turns that topology into a quantitative statement about τ(χ₂ → χ₁γ), maps it onto transition magnetic moments and the corpus models, and predicts what a population of inelastic events would look like in the three LZ samples.

The physical picture. For m_χ ≫ m_N the χ₂ carries essentially the incoming momentum: it leaves the vertex at 630–750 km/s (Sec. 2) and crosses the active TPC (0.3–1.5 m of chord) in 0.4–1.7 μs, the Skin in ≈ 0.1 μs, the 61 cm of GdLS in ≈ 1 μs, and is out of the water tank after ≈ 3–5 μs. A δ-photon emitted while χ₂ is still inside the active LXe converts within λ ≈ 2.3 cm and produces a second ER site (second S1 and second S2) — the event would not have passed the single-scatter selection (or, if the S1s merge, would carry S1 and S2 sums utterly unlike the observed ones). A photon emitted in the RFR gives an S1-only deposit (MSSI-like extra S1). A photon emitted in the Skin or OD deposits energy in a veto detector: within ±0.25/0.3 μs of the S1 that is a *prompt veto*; between 0.3 and 600 μs with E_Skin > 300 keV or E_OD > 200 keV it is a *delayed veto* (paper, samples paragraph). A χ₂ that survives ≈ 5 μs has left the experiment: the length of the 600 μs delayed window is irrelevant, only the transit time matters.

## 2. χ₂ kinematics (Part 1 of the script)

Energy and momentum conservation in the lab (non-relativistic): with incoming speed u, momentum transfer q = √(2m_N E_R) = 0.2463 GeV (m_N = 122.3 GeV, E_R = 248 keV),

  v₂ = u − q/m_χ,   |v₂|² = u² − 2(E_R + δ)/m_χ,   cos θ_q = (E_R + δ + q²/2m_χ)/(q u).

For m_χ = 1 TeV, u ≥ v_min(248 keV, δ) = 643/704/765/801 km/s at δ = 250/300/350/380 keV (`lz.vmin_kms`). The speed loss is 9–10 % (2(E_R+δ)/m_χ u² ≈ 0.15 in u²), the deflection ≲ 3°. Halo: Baxter-2021 SHM (v₀ = 238, v_esc = 544 km/s, lzcommon), boosted by the lab velocity on 16 June 2023 21:22:39 UTC. The Earth's barycentric velocity was taken from astropy's built-in ephemeris (ICRS (29.21, −2.43, −1.05) km/s) and rotated to Galactic axes; |v_lab| = 265.8 km/s (lzcommon's cosine model: 265.1). Rate weighting at fixed E_R is 1/u.

Direction in the detector. Fast DM moves toward the Solar anti-apex (RA 139.4°, Dec −43.7° for the full lab velocity). At SURF (44.352° N, 103.751° W; LST = 8.12 h from the USNO GMST formula, UT1 ≈ UTC) that direction was at altitude +0.5°: the DM wind was *horizontal* at the event time, so the χ₂ direction distribution is symmetric up/down (⟨cos zenith⟩ = +0.01; 10–90 % range −0.35 to +0.37 at δ = 300 keV, narrowing to ±0.12 at 380 keV because only the fastest particles, most aligned with the wind, qualify). The vertex azimuth is not published, so the wind azimuth relative to the vertex is averaged uniformly. An isotropic direction distribution is run as a bracket and changes the results by < 10 %.

Table (`P042_kinematics.csv`):

| δ [keV] | v_min [km/s] | ⟨u⟩ | ⟨v₂⟩ | v₂ range | speed loss | ⟨cos zenith⟩ | upward fraction |
|---|---|---|---|---|---|---|---|
| 250 | 643 | 686 | 617 | 569–752 | 10.1 % | +0.01 | 0.52 |
| 300 | 704 | 734 | 663 | 630–746 | 9.6 % | +0.01 | 0.52 |
| 350 | 765 | 779 | 706 | 691–740 | 9.3 % | +0.01 | 0.52 |
| 380 | 801 | 804 | 730 | 727–737 | 9.2 % | +0.01 | 0.54 |

δ = 390 keV is kinematically forbidden on 16 June (v_min = 813 > v_max = 810 km/s; cf. P002's δ_max = 387 keV) and is skipped.

## 3. Detector model and photon transport (Part 2)

Geometry (cm; z = 0 at the cathode, axis vertical; paper values in bold, others recalled and flagged in `recalled_inputs.json`): active TPC r < 72.8, 0 < z < 145.6 (likely); RFR **13.75 cm** below the cathode (paper); PTFE wall/field cage 3 cm; side Skin 6 cm of LXe (75.8–81.8 cm) up to the liquid level, bottom Skin 25 cm below the bottom PMT array (arrays 15 cm thick, treated as dead steel-like structure of 1.5 g/cm³); ICV and OCV titanium 0.8 cm each with a 7 cm vacuum gap; 4 cm water; GdLS 61 cm (side, 94.4–155.4 cm; top/bottom tanks 60 cm); water to r = 380 cm. Vertex **(r = 45.9, z = 26.4) cm** (paper, Fig. 3 caption).

Photon interactions: mass attenuation coefficients for Xe anchored on the corpus values (P004: 0.3 cm at 122 keV; P029: 0.62/0.99/1.42/2.9 cm at 164/203/243/375 keV) with recalled XCOM-like values elsewhere; light materials Compton-dominated. Resulting LXe attenuation lengths: 1.52/2.30/2.69/3.02 cm at 250/300/350/380 keV; GdLS 10.1 cm, water 8.5 cm, Ti 2.13 cm at 300 keV. The P029 "long bracket" (λ ≈ 4 cm at 375 keV) is covered by the ×0.7 attenuation variant. Photoelectric fraction in Xe: 0.62/0.50/0.40 at 243/300/375 keV. Transport uses Woodcock (delta) tracking with the Xe coefficient as majorant, photoabsorption (local deposit; K-fluorescence escape ignored) and Klein–Nishina Compton scattering (rejection sampling), continuing until E < 5 keV or the photon leaves the world. Energy is scored separately in the active TPC, the RFR, the Skin and the GdLS; the first TPC interaction position is stored.

Classification of an emission at path length s (time t = s/v₂), precedence from the top: (0/1) any TPC deposit > 1 keV → *second site in the TPC* (fails SS; sub-class *merged S1* if t < 0.15 μs); (2) RFR deposit → *S1-only* (MSSI-like extra S1, also incompatible with the recorded event); (3) Skin ≥ 5 keV within 0.25 μs or OD ≥ 30 keV within 0.3 μs → *prompt veto* (thresholds 2.5/4.5 phd translated with recalled light yields; flagged uncertain); (4) Skin reconstructed energy ≥ 300 keV (10 % resolution) or OD ≥ 200 keV, 0.3 < t ≤ 600 μs → *delayed veto*; (6) otherwise *clean* (escape, or sub-threshold deposits). P(class | τ) = Σ_traj w Σ_s [e^{−s_lo/vτ} − e^{−s_hi/vτ}] 1[class], with the decays beyond the relevant region counted as clean. 2000 trajectories × 3 cm steps × 2 photons (≈ 1.7×10⁵ histories) for the baseline; 1500 × 1 for variants; 500 × 1 per δ.

Path-length bookkeeping (δ = 300 keV, wind): chord in the active TPC mean 65 cm (10–90 %: 30–117; range 27–147 cm), i.e. 0.98 μs (0.44–1.74); Skin 7.1 cm / 0.11 μs; GdLS 66 cm / 0.99 μs; only 7 % of trajectories cross the RFR (mean 1.6 cm); χ₂ leaves everything after 3.0 μs on average (max 4.8). A photon emitted *at the vertex* never leaves the TPC (0 of 20 000).

## 4. Topology probabilities and the single-event limit (Part 3)

Long-lifetime regime (τ ≫ 2 μs): P_notclean(τ) = 1.68 μs/τ, i.e. the "sensitive time" of LZ for a 300 keV de-excitation photon is 1.68 μs, split as **TPC second site 60.7 %** (52.2 % with resolved S1s, 8.5 % merged), **RFR S1-only 1.7 %**, **prompt veto 0.0 %**, **delayed veto 37.7 %**. The prompt window cannot be populated: χ₂ needs ≥ 0.36 μs to reach the Skin from a vertex 26.9 cm from the wall, and photons from the first 0.25 μs are absorbed in the TPC (leakage below the MC sensitivity of ~10⁻⁴). The delayed share comes almost entirely from the OD: a 300 keV photon entering 61 cm of GdLS is absorbed with high probability and deposits > 200 keV; the Skin (threshold 300 keV = E_γ) contributes only through full-absorption events at the resolution edge.

`P042_topology_vs_tau_d300_wind.csv` (τ = 10⁻⁹–10² s, 221 points):

| τ | P(TPC 2nd site) | merged | P(RFR) | P(prompt) | P(delayed) | P(clean) |
|---|---|---|---|---|---|---|
| 0.1 μs | 0.997 | 0.75 | 1.6e-5 | 0 | 9.2e-4 | 0.0020 |
| 0.14 μs (MiDM 366) | 0.988 | 0.63 | 1.1e-4 | 0 | 3.6e-3 | 0.0080 |
| 0.5 μs | 0.802 | 0.25 | 4.8e-3 | 0 | 0.069 | 0.124 |
| 1 μs | 0.591 | 0.13 | 7.9e-3 | 0 | 0.124 | 0.275 |
| 1.4 μs (MiDM 350) | 0.483 | 0.096 | 8.0e-3 | 0 | 0.136 | 0.371 |
| 10 μs | 0.095 | 0.014 | 2.4e-3 | 0 | 0.052 | 0.849 |
| 66 μs (MiDM 300) | 0.0151 | 0.0021 | 4.1e-4 | 0 | 0.0092 | 0.9749 |
| 1 ms | 1.0e-3 | 1.4e-4 | 2.8e-5 | 0 | 6.2e-4 | 0.9983 |
| 0.06 s (Higgsino) | 1.7e-5 | 2.3e-6 | 4.6e-7 | 0 | 1.0e-5 | 1.0000 |

Single-event inference. The recorded class is "clean". The likelihood of τ given this single observation is L(τ) ∝ P_clean(τ), relative to L(τ → ∞) = 1. Values of τ for which the observed class had probability < 10 % are excluded at 90 % CL: **τ(χ₂ → χ₁γ) > 0.43 μs** (P_clean = 0.5 at 2.1 μs; P_clean = 0.9 at 15.7 μs; 0.99 at 167 μs). The "expected-topology" line P_clean ≥ 0.9 at 15.7 μs is the analogue of P023's τ ≥ 12.7 μs (P023 used a fixed 1 m chord and λ ≈ 2 cm) and agrees with it to 25 %; the two criteria answer different questions (90 % probability of the observed topology vs. 90 % CL exclusion) and both are reported. With a log-uniform prior over 10⁻⁹–10² s the posterior 10th percentile is 12 μs, but this is prior-dominated and not used. Branching ratio: P_clean = 1 − BR_γ[1 − P_clean,γ(τ)] ≥ 0.10 gives BR_γ ≤ 0.9/P_notclean(τ), i.e. BR_γ ≤ 0.90 for prompt decays (τ ≲ 0.05 μs) and no constraint for τ > 0.43 μs (column `BR_gamma_max_90`).

Robustness (`variants` in `P042_summary.json`): isotropic χ₂ directions τ₉₀ = 0.456 μs, P_clean = 0.9 at 17.1 μs; Xe attenuation ×0.7 / ×1.4: 0.414 / 0.414 μs and 15.5 / 14.5 μs; OD prompt threshold 100 keV: 0.428 μs (unchanged, prompt is empty anyway); Skin resolution 0: 0.486 μs. P_notclean(66 μs) = 0.0250 (baseline), 0.0273 (isotropic), 0.0248, 0.0233 (attenuation variants). The limits are therefore robust to ±15 %; the dominant systematic is the transit time, i.e. the TPC size and χ₂ speed, both well known.

δ dependence (`P042_photonM1_delta_scan.csv`): τ₉₀ = 0.35, 0.36, 0.39, 0.43, 0.46, 0.50, 0.53, 0.54, 0.55, 0.58, 0.59, 0.58, 0.56 μs at δ = 250, 270, 290, 300, 310, 320, 330, 340, 350, 360, 366, 370, 380 keV; P_clean = 0.9 at 10.8–18.0 μs. The slow rise with δ reflects the longer attenuation length (more OD tags) and the faster χ₂ at larger δ, which shortens the chord time slightly.

## 5. Mapping to transition moments and corpus models (Part 4)

Width Γ(χ₂ → χ₁γ) = μ_tr² δ³/π (MiDM; recalled/likely, the same formula as P014 and P023). Check: μ_tr = 2.107×10⁻⁴ μ_N at 300 keV → τ = 6.625×10⁻⁵ s (P023: 6.626×10⁻⁵ s). The ceiling μ_max(δ) = √(ħπ/(τ₉₀ δ³)) follows from τ₉₀(δ): 2.62×10⁻³ μ_N (300 keV), 2.04×10⁻³ (330), 1.84×10⁻³ (350), 1.72×10⁻³ (360), 1.66×10⁻³ (366), 1.61×10⁻³ μ_N (380).

Photon-M1 (MiDM/composite, P023). P023's LZ-rate requirement μ_tr(N_ROI = 1; 0.105; 3.65) from `P023_single_site_ceiling.csv` is compared with the ceiling (Fig. 2, `P042_photonM1_delta_scan.csv`):

| δ [keV] | μ_tr(N=1) [μ_N] | τ | P_clean | headroom μ_max/μ_tr |
|---|---|---|---|---|
| 300 | 2.11e-4 | 66 μs | 0.975 | 12.4 |
| 330 | 4.85e-4 | 9.4 μs | 0.820 | 4.2 |
| 350 | 1.14e-3 | 1.4 μs | 0.316 | 1.6 |
| 360 | 2.11e-3 | 0.38 μs | 0.054 | 0.8 |
| 366 | 3.40e-3 | 0.14 μs | 0.006 | 0.5 |
| 380 | 2.16e-2 | 3.1 ns | 0.000 | 0.1 |

Crossings: for N = 1, P_clean = 0.9 at **δ = 321 keV** (P023: 326 keV, reproduced), 0.5 at 344 keV, 0.1 (90 % CL exclusion) at **δ = 358 keV**. For the LZ interval edges: N = 3.65 → 302/330/348 keV; N = 0.105 → 346/362/370 keV. The P007/P021 Higgsino-like window δ = 358–385 keV is therefore excluded at ≥ 90 % CL for any model whose LZ rate is carried by a photon-M1 transition; P023's stronger statement (δ ≤ 326 keV) is where the single-site topology becomes *expected* rather than merely allowed. Between 321 and 358 keV, L(δ) = P_clean(δ) is a likelihood factor (0.9 → 0.1) that should multiply P021's spectral likelihood for photon-M1 models.

Higgsino (P007/P014/P026): τ_γ ≈ 0.06 s from the one-loop transition dipole μ₁₂ ≈ (α₂/2π)e/2μ (P014, uncertain to an O(1) loop factor), τ_νν̄ = 4.6×10⁵ s. P_notclean = 2.8×10⁻⁵ at 0.06 s; even a 100× shorter τ_γ gives 3×10⁻³. Safe in the escape regime; the χ₂ decays 10⁴ km away.

Dark photon (P011/P026): with m_A′ ≥ 2.6 GeV there is no on-shell A′ channel, the one-loop χ₂ → χ₁γ vanishes (P011), and τ_νν̄ = 1.9×10¹⁸ s (ε = 10⁻⁶). P_notclean ~ 10⁻²⁴; the final state is invisible anyway. A sub-MeV A′ (m_A′ < δ) would open χ₂ → χ₁A′ but A′ → e⁺e⁻ is closed below 1.022 MeV; also invisible.

L10-type contact (P012). The elastic L10 has no χ₂. If the same dimension-6 tensor contact (C = d10/m_v² = 4.6×10⁻⁶ GeV⁻² for d10 = 0.28) is made inelastic, the transition dipole is loop-induced: μ_tr ~ e C m_f ln/(16π²) (dimensional estimate, flagged uncertain ×10). With the nucleon mass in the loop (m_f = m_N, ln ≈ 2): μ_tr ≈ 1.0×10⁻⁷ μ_N → τ ≈ 280 s (decay length 2×10⁵ km); with quark-level tensor couplings (m_f ≈ 5 MeV): 5×10⁻¹⁰ μ_N → 10⁷ s. Both are in the escape regime by ≥ 8 orders of magnitude.

## 6. Population predictions (Part 5)

`P042_population.csv`. For N inelastic events at the LZ vertex geometry:

| model (τ) | clean | two-site (TPC) | merged-S1 | RFR S1-only | prompt | delayed | events per two-site | per veto tag |
|---|---|---|---|---|---|---|---|---|
| MiDM δ=300 (66 μs) | 0.975 | 1.5 % | 0.21 % | 0.04 % | 0 | 0.92 % | 66 | 109 |
| MiDM δ=330 (9.4 μs) | 0.82 | 9.8 % | 1.5 % | 0.24 % | 0 | 7.8 % | 10 | 13 |
| MiDM δ=350 (1.4 μs) | 0.32 | 46 % | 10 % | 0.3 % | 0 | 22 % | 2.2 | 4.5 |
| MiDM δ=366 (0.14 μs) | 0.006 | 98 % | 70 % | — | 0 | 1.0 % | 1.0 | 98 |
| τ = 1 μs | 0.275 | 59 % | 13 % | 0.8 % | 0 | 12 % | 1.7 | 8 |
| τ = 10 μs | 0.85 | 9.5 % | 1.4 % | 0.24 % | 0 | 5.2 % | 11 | 19 |
| Higgsino (0.06 s) | 1.000 | 1.7e-5 | 2e-6 | 5e-7 | 0 | 1.0e-5 | 6×10⁴ | 10⁵ |

The two-site companions (τ ≫ transit): a ≈ 300 keV ER (98 % of conversions deposit the full δ within the TPC; mean 297 keV) at a mean distance of 42 cm from the NR vertex (10–90 %: 7–88 cm), with the second S1 arriving Δt = 0.63 μs later on average (90 % within 1.33 μs). The pair obeys separation/Δt = 70 cm/μs = 700 km/s, the χ₂ speed — a kinematic fingerprint no background reproduces (a Compton-scattered γ gives Δt = separation/c ≈ ns). Such events fail LZ's SS selection and would sit in the multiple-scatter sample as "NR-band S2 + ER-band S2 with two resolved S1s".

Delayed-veto sample. Given the one clean science event, the expected number of inelastic events in the *delayed-veto* sample is 0.0094 for MiDM at δ = 300 keV (66 μs), 0.70 at δ = 350 keV (1.4 μs); the expected number of rejected two-site events is 0.016 and 1.45 respectively. LZ's 55 delayed-veto events (50.4 ± 2.0 fitted, dominated by random coincidences with ER backgrounds; Fig. S1b) would need a high-energy NR-band member with an OD pulse > 200 keV and no prompt signal; for τ ≳ 10 μs the expectation is below 0.1 events and the check is not yet informative, but it becomes so for any population of ≳ 10 events at τ ≲ 10 μs. The prompt-veto sample is never populated by de-excitation photons.

## 7. Validation and consistency checks

* Lifetime formula reproduces P023's 66 μs at 300 keV to 0.02 %.
* P_clean = 0.9 at 15.7 μs vs P023's 12.7 μs (1 m chord, λ ≈ 2 cm): consistent given P023's fixed geometry; the P023 ceiling 326 keV becomes 321 keV with the full geometry.
* Long-τ scaling P_notclean = t_sens/τ verified (1.68 μs from the τ = 1 s point matches the 66 μs and 1 ms rows to 1 %).
* Direction: |v_lab| from the astropy ephemeris agrees with lzcommon's cosine model to 0.3 %; the anti-apex altitude of +0.5° is consistent with the known geometry (Cygnus low on the horizon for a 44° N site at LST ≈ 8 h).
* Isotropic vs wind directions differ by < 10 % in all limits; attenuation ×0.7–1.4 by < 8 %.
* The MC classes are exclusive by construction; Σ P = 1 checked implicitly (the clean class is 1 − others plus the beyond-world remainder).

## 8. Failed or abandoned approaches

* First run used per-δ astropy conversions inside the rejection loop and included δ = 390 keV, which is kinematically forbidden on 16 June; the loop never terminated and exceeded the 600 s foreground budget twice. Fixed by batching the halo-tail rejection in numpy and guarding v_min ≥ v_max.
* astropy's AltAz transformation could not be used offline (IERS configuration error); the horizon altitude is computed analytically from GMST.
* Xe K-fluorescence escape and the exact Skin/OD light yields are not modelled; brackets are used instead.

## 9. Discussion

The de-excitation photon is the one piece of inelastic-DM phenomenology that LZ has already measured, by not seeing it. The measurement is weak in absolute terms (τ > 0.43 μs) because the χ₂ leaves the detector in microseconds, but it is decisive for models in which the same coupling fixes both the scattering rate and the decay width: photon-M1 hyperfine models must have δ < 358 keV (90 % CL), removing the large-δ region where P002/P021 find the spectral preference (360–385 keV). Models with Z- or A′-mediated transitions (Higgsino, dark photon) are unaffected: their photon channels are loop-suppressed or absent and their χ₂ lifetimes exceed the transit time by 10⁴–10²⁴. For a future population the signature is robust and specific: 1.7 μs of sensitive time per event, shared 60:38:2 between rejected two-site TPC events, OD-tagged delayed-veto events and RFR S1-only events, never prompt vetoes, with two-site pairs obeying separation = 700 km/s × Δt_S1.

## 10. Figures

* `figures/P042_fig1_topology_vs_tau.png` — P(class | τ) for δ = 300 keV at the LZ vertex with DM-wind directions: clean, TPC second site, RFR S1-only, prompt veto (identically zero, off scale), delayed veto; vertical lines at the MiDM lifetimes (300/350/366 keV), the Higgsino loop lifetime and the 90 % CL limit 0.43 μs.
* `figures/P042_fig2_mu_tr_ceiling.png` — transition moment vs δ: P023's μ_tr for 0.105–3.65 LZ events (band) and N = 1 (line) against this work's ceilings (P_clean < 10 % excluded; P_clean = 90 % line); crossings at 321 and 358 keV; P012's excluded elastic photon dipole for reference.

## 11. References

LZ Collaboration, arXiv:2609.02823 (2026) · D. S. Akerib et al. (LZ), NIM A 953, 163047 (2020) (detector description) · S. Chang, N. Weiner, I. Yavin, PRD 82, 125011 (2010) (MiDM width) · D. Tucker-Smith, N. Weiner, PRD 64, 043502 (2001) · D. P. Finkbeiner, T. R. Slatyer, N. Weiner, I. Yavin, JCAP 09 (2009) 037 (metastable excited WIMPs) · M. J. Berger, J. H. Hubbell, XCOM, NBSIR 87-3597 (1987) · corpus: P002, P004, P007, P011, P012, P014, P021, P023, P026, P029, P037.

## 12. Tools and provenance (mirrors `provenance/P042.json`)

* Software: python 3.12.13; numpy 2.5.3 (seeded default_rng(42), einsum, vectorised MC); scipy not used; pandas 3.0.5 (tables); matplotlib 3.11.2 (Agg); astropy 8.0.1 (Time, get_body_barycentric_posvel built-in ephemeris, SkyCoord Galactic↔ICRS rotation; AltAz not usable offline); common/lzcommon.py (LZ dict, vmin_kms, m_nucleus_gev, constants V0/VESC/V_SUN_PEC, v_earth_kms, M_V_GEV, M_NUCLEON_GEV, C_KMS).
* Script: `output/code/P042_deexcitation_photon.py`, command `.venv/bin/python output/code/P042_deexcitation_photon.py` (114–125 s).
* Local inputs: LZ tex (detector paragraph l.76–84, Fig. 3 caption l.119–130, samples paragraph l.139–147, event description l.162–166, MSSI/RFR l.186–194, supplement samples l.425–435, veto-sample tables l.500–585, waveform analysis l.693–695); dossier; ledger; papers P011, P012, P014, P023, P026; `work/P023/P023_single_site_ceiling.csv` and `P023_chi2_lifetime.csv`; `work/P014/details.md` §8; `work/P011/relic_lifetime.json`; P029.md (attenuation lengths); `code/P023_composite_idm.py` (structure); lzcommon.py; ENVIRONMENT_versions.txt.
* Recalled knowledge: 21 items listed in `recalled_inputs.json` (5 certain, 8 likely, 8 uncertain), dominated by detector dimensions beyond the TPC and veto light yields.
* WimPyDD: not run (P023's WimPyDD-based μ_tr(δ) reused). No datasets, no data requests.
