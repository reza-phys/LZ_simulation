# P026 · Cosmological constraints on the excited state: BBN, CMB and the χ₂ → χ₁ transition for δ ≈ 300 keV — research record

Simulated date 2026-09-08. Author profile: early-Universe cosmologists. Category COMP, astro-ph.CO (cross-list hep-ph).
Script: `output/code/P026_excited_state_cosmology.py` (run from the simulation root with `.venv/bin/python`; 3 s). Every number below
is printed in `output/work/P026/run_log.txt` or stored in the CSV/JSON files listed in §9. Recalled inputs are flagged [recall: reliability]
and collected in `recalled_knowledge.json` (17 items).

## 1. Motivation and framework

The inelastic-DM readings of the LZ 248 keV event (P002, P007 Higgsino with δ ≈ 366 keV, P011 dark-photon pseudo-Dirac with δ = 204–387 keV)
share a two-level dark sector: a ground state χ₁ and an excited state χ₂ split by δ ≈ 200–400 keV, with m_χ ≈ 0.3–4 TeV. At chemical
freeze-out (T_f ≈ m/25 ≈ 40 GeV ≫ δ) the two states are equally populated. Three questions follow for cosmology:

1. **Does the excited state survive freeze-out?** After chemical decoupling χ₂ can still be depleted by inelastic *down-scattering* on the
   SM bath, χ₂ f → χ₁ f (exothermic, no threshold), through the same Z or A′ coupling that produces the LZ signal, while *up-scattering*
   χ₁ f → χ₂ f re-populates it as long as T ≳ δ. The surviving fraction f₂ ≡ n₂/(n₁+n₂) is set by the temperature T_* at which these
   transitions decouple (kinetic decoupling of the two-level system).
2. **How, and when, does χ₂ decay?** For δ < 2m_e = 1022 keV the e⁺e⁻ channel is closed (P014); the channels are χ₂ → χ₁νν̄ (Z or
   Z-mixing) and, if a transition dipole exists, χ₂ → χ₁γ. The photon channel is *visible* to BBN/CMB/γ-ray probes; the neutrino channel is
   *invisible* except through ΔN_eff.
3. **Is the injected energy f₂ δ per DM particle constrained** by BBN, CMB spectral distortions, CMB anisotropies, the 300 keV γ-ray sky,
   and by the requirement (P011) that no χ₂ population survives today (an exothermic χ₂N → χ₁N population would flood LZ)?

The organising number is the energy budget: the excited state stores f₂ δ/m ≈ 1.3 × 10⁻⁷ of the DM mass energy (f₂ = 0.42, δ = 300 keV,
m = 1 TeV). Relative to the photon bath this is negligible at any time before recombination, but per baryon it is
f₂ δ (n_DM/n_b) = 633 eV — 47 times the hydrogen ionisation energy — so the *epoch* and *channel* of its release decide everything.

## 2. Thermodynamics of the bath (script part A)

- Effective degrees of freedom g_*(T) with exact Fermi–Dirac e± integrals, muons, three neutrinos at T_ν(T), and a crude quark–gluon
  step above 150 MeV; checks: g_*(10 MeV) = 10.76, g_*(1 MeV) = 10.56, g_*(50 keV) = 3.385 (3.36 standard), T_ν/T_γ → 0.7138 = (4/11)^{1/3}
  from (s_γ + s_e) a³ = const after an instantaneous ν decoupling at 2 MeV [recall: certain].
- H(T) = 1.66 √g_* T²/M_Pl in the radiation era; late-time Friedmann H(z) with Ω_m = 0.3134, Ω_r = 9.1 × 10⁻⁵, Ω_Λ = 0.6865
  (Planck 2018 [recall: certain]); age 4.353 × 10¹⁷ s (13.8 Gyr), t(z = 1100) = 1.16 × 10¹³ s, t(z = 2 × 10⁶) = 5.98 × 10⁶ s,
  t(z = 5 × 10⁴) = 9.4 × 10⁹ s. For t < 10⁴ s the photon temperature is obtained by inverting H_early(T) = 1/(2t) (the e± heating of the
  photons is thereby included; the z(t) map at t < 10 s ignores it, a 10 % effect used only for labelling).
- ρ_DM,0 = 1.264 × 10⁻⁶ GeV cm⁻³, ρ_γ,0 = 2.603 × 10⁻¹⁰ GeV cm⁻³ (ρ_DM/ρ_γ = 4858 today, ∝ 1/(1+z)), n_b,0 = 2.516 × 10⁻⁷ cm⁻³,
  n_DM/n_b = (Ω_c/Ω_b)(m_p/m) = 5.03 × 10⁻³ at 1 TeV.

## 3. Transition rates and the excited fraction (part A)

### 3.1 Couplings
Higgsino (P007): the neutral Dirac state is vector-like with T₃ = +½ for both chiralities, so its Z current is ψ̄γ^μψ with g_V = 1, g_A = 0
(twice a left-handed neutrino's). Integrating out the Z,
  L_eff = −√2 G_F [ψ̄γ^μψ][f̄γ_μ(g_V^f − g_A^f γ₅)f],  g_V^f = T₃ − 2Q s_W², g_A^f = T₃  [recall: certain].
In the Majorana basis ψ = (χ₁ + iχ₂)/√2 the current is purely off-diagonal, ψ̄γ^μψ = iχ̄₁γ^μχ₂, so every Z-mediated process is a
χ₁ ↔ χ₂ transition. Check: for a neutron (g_V = −½) the non-relativistic cross-section c²μ²/π with c = G_F/√2 gives
σ_n = 7.408 × 10⁻³⁹ cm², P007's 7.40 × 10⁻³⁹ (run log).

Dark photon (P011): L = c_D [χ̄₁γ^μχ₂][Σ_f Q_f f̄γ_μf], c_D = e ε g_D/m_A′², so c_D² = π σ_p/μ_p². The LZ fit fixes σ_p(N = 1) (P011,
1 TeV, annual halo): 1.20 × 10⁻⁴², 8.59 × 10⁻⁴², 2.99 × 10⁻⁴⁰, 2.29 × 10⁻³⁹, 5.60 × 10⁻³⁸ cm² at δ = 250, 300, 350, 365, 380 keV, i.e.
c_D²/G_F² = 8.1 × 10⁻⁵, 5.8 × 10⁻⁴, 2.0 × 10⁻², 0.154, 3.78. Only charged fermions (e, μ, quarks above 150 MeV) participate; neutrinos
do not couple to the A′.

### 3.2 Cross-section on a relativistic bath fermion (derived)
For a heavy χ at rest and a massless fermion of energy E, with L = [χ̄₁γ^μχ₂][f̄γ_μ(c_V − c_Aγ₅)f], only the time component of the
heavy current survives (ūγ⁰u → 2m δ_ss′), the lepton trace is Tr[p̸′γ⁰p̸γ⁰] = 4EE′(1 + cosθ) (the γ₅ cross-term vanishes), the final
heavy-particle phase space contributes 1/(2m), and the outgoing energy is E′ = E ± δ:

  dσ/dΩ = (c_V² + c_A²) E′² (1 + cosθ)/(8π²),  σ_down(E) = (c_V² + c_A²)(E + δ)²/(2π),  σ_up(E) = (c_V² + c_A²)(E − δ)² θ(E − δ)/(2π).

Checks: with c = √2 G_F g_V it reproduces the standard total CEνNS cross-section G_F² Q_W² E²/(4π) for a neutrino on a nucleus of weak
charge Q_W [recall: certain]; per neutrino state σ(χ₂ν → χ₁ν) = G_F²(E + δ)²/π = 1.43 × 10⁻⁴⁴ cm² at E = 1 MeV, δ = 300 keV. Neutrinos are
counted as 2 states per flavour (ν_L, ν̄_R) with the Dirac spin average undone (equivalently g = 4, g_V² + g_A² = ½). Thermal weights
g_f(g_V² + g_A²): ν 2.0 per flavour, e 1.006, μ 1.006, u 3.44, d 4.44 (×3 colours included).

### 3.3 Rates, detailed balance and Boltzmann equation
With Maxwell–Boltzmann statistics for the bath (the E⁴ moment differs from Fermi–Dirac by 3 %),
  Γ_down(T) = Σ_f g_f/(2π²) · (c_V² + c_A²)_f/(2π) · K_f,  K = ∫dE pE (E + δ)² e^{−E/T}  (= 24T⁵ + 12δT⁴ + 2δ²T³ for m_f = 0),
with T → T_ν for neutrinos and the electron/muon mass kept in the density (the cross-section formula is used with the total energy; this is
crude only where e± are already Boltzmann-suppressed). Detailed balance for a heavy two-level system in a thermal bath gives
Γ_up = Γ_down e^{−δ/T} exactly for MB statistics [recall: certain]. The excited fraction obeys
  df₂/dt = −(Γ_down + Γ_up) f₂ + Γ_up − f₂/τ,  f₂^eq = e^{−δ/T}/(1 + e^{−δ/T}),
integrated with `scipy.integrate.solve_ivp` (LSODA) in ln T from 200 MeV (Higgsino; 1 GeV for the dark photon), f₂ = f₂^eq, down to 0.2 keV
with τ = ∞ (decays are treated separately in §4–5). The decoupling temperature T_* solves Γ_down(1 + e^{−δ/T}) = H.

### 3.4 Results (`freeze_out.csv`, Fig. 2)

| model | δ (keV) | T_* (MeV) | f₂^eq(T_*) | **f₂ (freeze-out)** |
|---|---|---|---|---|
| Higgsino | 200 | 0.85 | 0.442 | **0.446** |
| Higgsino | 250 | 0.85 | 0.427 | **0.433** |
| Higgsino | 300 | 0.85 | 0.413 | **0.420** |
| Higgsino | 350 | 0.85 | 0.398 | **0.407** |
| Higgsino | 366 | 0.85 | 0.394 | **0.403** |
| Higgsino | 400 | 0.85 | 0.384 | **0.394** |
| dark photon (LZ fit) | 250 | 26.7 | 0.498 | **0.498** |
| dark photon (LZ fit) | 300 | 15.0 | 0.495 | **0.495** |
| dark photon (LZ fit) | 350 | 4.7 | 0.482 | **0.483** |
| dark photon (LZ fit) | 365 | 2.4 | 0.462 | **0.464** |
| dark photon (LZ fit) | 380 | 0.8 | 0.386 | **0.396** |

The Higgsino's χ₂ ↔ χ₁ transitions have exactly the neutrino-like G_F²E² cross-section, so they decouple together with the neutrinos, at
T_* = 0.85 MeV ≈ 2.8 δ, when the Boltzmann factor e^{−δ/T} is still 0.70: **f₂ = 0.40–0.45 survives** for every δ in the LZ window.
Robustness: a neutrino-only bath gives T_* = 0.89 MeV, f₂ = 0.424; scaling all rates by ×0.5 (×2) gives T_* = 1.07 (0.68) MeV and
f₂ = 0.436 (0.400); the freeze-out value is insensitive because f₂^eq varies only logarithmically with T_* while Γ/H ∝ T³. For the dark
photon the LZ-fit coupling is 10⁻⁴–10⁻² of G_F² at δ ≤ 350 keV, so decoupling occurs at 5–27 MeV ≫ δ and **f₂ = 0.48–0.50**; only at
δ = 380 keV, where the fitted σ_p is 6500× larger, does T_* reach 0.8 MeV (f₂ = 0.40). P011's statement "f₂ ≈ ½ unless χ₂ decays" is
thereby confirmed and quantified. The DM–DM channel χ₂χ₂ → χ₁χ₁ (σ ≈ c²m²/π ≈ 8 × 10⁻³³ cm² at 1 TeV) has Γ/H = 4 × 10⁻⁵ at T_*
(n_χ = 2 × 10¹⁹ cm⁻³) and is irrelevant after chemical freeze-out.

## 4. Decays (part B; `higgsino_decays.csv`, `dark_photon_tau.csv`)

Higgsino: Γ(χ₂ → χ₁νν̄) = G_F²δ⁵/(20π³) (P014, three flavours) and Γ(χ₂ → χ₁γ) = μ₁₂²δ³/π with μ₁₂ = κ (α₂/2π) e/(2μ)
[P014 estimate; recall: uncertain; κ = 0.1–3 band]:

| μ (GeV) | δ (keV) | τ_νν̄ (s) | T at τ_νν̄ | τ_γ (s), κ = 1 [κ = 3, 0.1] | T at τ_γ |
|---|---|---|---|---|---|
| 300 | 310 | 1.05 × 10⁶ | 1.1 keV | 9.4 × 10⁻³ [1.0 × 10⁻³, 0.94] | 8.9 MeV |
| 500 | 342 | 6.4 × 10⁵ | 1.4 keV | 1.9 × 10⁻² [2.2 × 10⁻³, 1.9] | 6.2 MeV |
| 1000 | 366 | 4.57 × 10⁵ | 1.7 keV | 6.4 × 10⁻² [7.1 × 10⁻³, 6.4] | 3.4 MeV |
| 2000 | 375 | 4.05 × 10⁵ | 1.8 keV | 0.24 [0.026, 24] | 1.8 MeV |
| 4000 | 376 | 3.99 × 10⁵ | 1.8 keV | 0.94 [0.10, 94] | 0.9 MeV |
| 1000 | 300 | 1.23 × 10⁶ | 1.0 keV | 0.115 [0.013, 12] | 2.5 MeV |
| 1000 | 250 | 3.07 × 10⁶ | 0.66 keV | 0.20 [0.022, 20] | 1.9 MeV |
| 1000 | 200 | 9.4 × 10⁶ | 0.38 keV | 0.39 [0.043, 39] | 1.4 MeV |

Whichever channel dominates, the Higgsino χ₂ is gone by t ≲ 10⁷ s (T ≳ 0.4 keV), i.e. 10⁶ times before recombination and 10¹¹ times
before today: **f₂(today) = 0** (formally 0.42 e^{−3.5×10¹¹}).

Dark photon (P011 width, hypercharge mixing): Γ = 3G_eff²δ⁵/(120π³), G_eff = g_D ε tanθ_W g/(4cosθ_W m_Z²); τ ∝ 1/(ε²α_D δ⁵). Because the
LZ fit fixes ε²α_D = σ_p m_A′⁴/(16πα μ_p²), the lifetime becomes a function of m_A′ alone (α_D cancels):
  δ = 300 keV: τ_νν̄ = 2.6 × 10¹⁸ s (m_A′/GeV)⁻⁴ → 3.2 × 10²⁰, 4.2 × 10¹⁹, 2.6 × 10¹⁸, 1.6 × 10¹⁷, 5.7 × 10¹⁶, 4.2 × 10¹⁵, 2.6 × 10¹⁴ s at
  m_A′ = 0.3, 0.5, 1, 2, 2.6, 5, 10 GeV; δ = 365 keV: 3.7 × 10¹⁵ s (m_A′/GeV)⁻⁴; δ = 250: 4.6 × 10¹⁹; 350: 3.5 × 10¹⁶; 380: 1.2 × 10¹⁴ s (m_A′/GeV)⁻⁴.
χ₂ → χ₁γ is absent for kinetic mixing (the photon couples to the dark current ∝ q² and vanishes on shell; P011 [recall: likely]), so the
dark-photon χ₂ is *invisible*: it decays to neutrinos or not at all.

## 5. Energy injection and constraints (part C; `injection_scan.csv`, `cmb_exclusion_curve.csv`)

Energy budget (m = 1 TeV, δ = 300 keV, f₂ = 0.420): f₂δ/m = 1.26 × 10⁻⁷ of ρ_DM; Δρ/ρ_γ(z) = f₂ (δ/m)(ρ_DM/ρ_γ)₀/(1+z) = 6.1 × 10⁻⁴/(1+z);
633 eV per baryon.

### 5.1 BBN
Photodissociation thresholds [recall: certain]: D 2.224 MeV, ⁷Be 1.587, ⁷Li 2.467, ³He 5.49, ⁴He 19.81 MeV. A 300 keV photon (or the
electrons it Compton-heats) is below every threshold; there is no hadronic channel; e⁺e⁻ is closed. Photons injected at τ_γ ~ 0.06 s
(T ≈ 3.4 MeV) carry Δρ/ρ_γ = 3 × 10⁻¹⁴ and thermalise instantly. Neutrinos injected at τ_νν̄ = 4.6 × 10⁵–1.2 × 10⁶ s (T ≈ 1–1.7 keV)
add Δρ/ρ_γ = 0.8–1.4 × 10⁻¹⁰, i.e. **ΔN_eff = 4.6–6.1 × 10⁻¹⁰**. The bath heating by the down-scattering itself before T_* is of the same
10⁻¹³ order. **BBN is blind to the excited state.**

### 5.2 CMB spectral distortions (FIRAS |μ| < 9 × 10⁻⁵, |y| < 1.5 × 10⁻⁵ [recall: certain])
Injection scan for f₂ = 0.42, m = 1 TeV, δ = 300 keV (photon channel): τ = 10⁸ s (z = 4.9 × 10⁵, μ era) → Δρ/ρ_γ = 1.3 × 10⁻⁹, μ ≈ 1.8 × 10⁻⁹;
τ = 10¹⁰ s (z = 4.8 × 10⁴, y era) → 1.3 × 10⁻⁸, y = 3 × 10⁻⁹; τ = 10¹² s (z = 4400) → 1.4 × 10⁻⁷, y = 3.5 × 10⁻⁸; τ = 10¹³ s (z ≈ 1200) →
5 × 10⁻⁷, y = 1.3 × 10⁻⁷. Maximising over τ with f₂ = 0.5: **μ_max = 2.0 × 10⁻⁸, y_max = 1.5 × 10⁻⁷** (5 × 10⁻⁷ at m = 300 GeV) — 100–4500
times below FIRAS and near the 10⁻⁸ reach of a PIXIE-class mission [recall: uncertain] only in the y window. **No present spectral-distortion
constraint** for any (τ, f₂); the μ era is unreachable even in principle (μ ≲ 2 × 10⁻⁸).

### 5.3 CMB anisotropies (photon channel only)
Criterion adopted: the electromagnetic energy *deposited* per baryon between recombination (t_rec = 1.16 × 10¹³ s) and today must not exceed
E_crit = 0.2 eV. This is our re-expression of the Planck decaying-DM bound f_eff Γ ≲ 10⁻²⁵ s⁻¹ for τ ≫ t_U [Slatyer & Wu 2017; recall:
uncertain, factor ~3]: 5.1 GeV of DM mass per baryon × 10⁻²⁵ s⁻¹ × t_U = 0.22 eV per baryon over the age of the Universe, and independently
≈ 1.5 % of the hydrogen ionisation energy, the level at which extra ionisation shifts τ_reio by ~1σ (0.007) for injection at z ~ 30
(computed in the log: full ionisation from z = 30 gives τ = 0.47). Deposition efficiency for 300 keV photons: f_eff(z) = 0.5[1 − e^{−τ_C(z)}]
with the Compton optical depth per Hubble time τ_C = n_b(1+z)³ σ_KN c/H(z), σ_KN(300 keV) = 3.53 × 10⁻²⁵ cm² (Klein–Nishina, computed);
τ_C = 61, 11, 2.2, 0.37 at z = 1000, 300, 100, 30 — the universe becomes transparent to 300 keV photons below z ≈ 50, which weakens the
bound for τ ≳ 10¹⁶ s. Deposited energy per baryon for f₂ = 0.42: 0.003 eV (τ = 10¹² s), 97 eV (10¹³), 281 eV (10¹⁴), 235 eV (10¹⁵),
81 eV (10¹⁶), 16 eV (10¹⁷), 2.1 eV (10¹⁸), 0.022 eV (10²⁰).

**Result:** with f₂ = 0.42 the photon channel is excluded for **τ_γ ∈ [2 × 10¹², 1 × 10¹⁹] s**; the exclusion holds for f₂ down to
2.9 × 10⁻⁴ (at τ ≈ 2 × 10¹⁴ s; 8.6 × 10⁻⁴ if E_crit is relaxed ×3). Windows: f₂ = 0.5: [1.6 × 10¹², 1.3 × 10¹⁹] s; 0.1: [2 × 10¹², 2.5 × 10¹⁸];
0.01: [4 × 10¹², 2 × 10¹⁷]; 10⁻³: [10¹³, 7.9 × 10¹⁵] s. Neither LZ model enters this window: the Higgsino's τ_γ ≲ 100 s and the
dark photon has no photon channel.

### 5.4 Today: the 300 keV line and the diffuse background (part D)
NFW decay D-factor (r_s = 20 kpc, R_⊙ = 8.2 kpc, ρ_⊙ = 0.3 GeV cm⁻³ [recall: likely]): 2.35 × 10²¹ (5°), **7.41 × 10²¹ (10°)**,
1.56 × 10²² GeV cm⁻² sr (16°). Line flux Φ = f₂ D/(4π τ m): for m = 1 TeV, 10° cone, f₂ = 10⁻³: 5.9 × 10⁻⁴, 5.9 × 10⁻⁶, 5.9 × 10⁻⁸ ph cm⁻² s⁻¹
at τ_γ = 10¹⁸, 10²⁰, 10²² s; f₂ = 0.5: 0.30, 3.0 × 10⁻³, 3.0 × 10⁻⁵, 3.0 × 10⁻⁷ at 10¹⁸–10²⁴ s. Against an INTEGRAL/SPI narrow-line
sensitivity of 3 × 10⁻⁵ ph cm⁻² s⁻¹ [recall: uncertain, ×3] this gives **f₂(today)/τ_γ ≤ 5.1 × 10⁻²³ s⁻¹**, i.e. τ_γ ≥ 9.8 × 10²¹ s
(f₂ = 0.5) or 2.0 × 10¹⁹ s (10⁻³). The isotropic extragalactic decay intensity (z < 1 shell) at 300 keV stays below the measured
~2 keV cm⁻² s⁻¹ sr⁻¹ [recall: uncertain] for τ_γ ≥ 5.7 × 10¹⁹ s (f₂ = 0.5). 21-cm probes are not used: the same 0.6 keV/baryon budget
would heat the z ~ 20 gas only if τ ~ 10²³–10²⁵ s (f₂ = 0.5), and no robust 21-cm limit exists.

### 5.5 The exothermic requirement (P011) and the dark-photon window
P011 computed the exothermic ROI count at the fitted σ_p if f₂ = 1: N_exo = 251, 858, 1.8 × 10⁴, 1.2 × 10⁵, 2.6 × 10⁶ at δ = 250–380 keV, so
f₂(today) ≤ 4.0 × 10⁻³, 1.17 × 10⁻³, 5.5 × 10⁻⁵, 8.2 × 10⁻⁶, 3.8 × 10⁻⁷. With our f₂^fo (0.498–0.396) and f₂(today) = f₂^fo e^{−t_U/τ} this
needs τ ≤ 9.0, 7.2, 4.8, 4.0, 3.1 × 10¹⁶ s and, through τ(m_A′), **m_A′ ≥ 4.8, 2.45, 0.92, 0.55, 0.25 GeV** (P011: 5.0, 2.6, 0.94, 0.52,
0.09 GeV; the 380 keV value differs because P011 assumed f₂ = ½ where we find 0.40 and it also used the June halo in one step). The
lifetime equals the age of the Universe at m_A′ = 3.2, 1.56, 0.53, 0.30, 0.13 GeV. For the Higgsino the equivalent exothermic count is
~10⁵–10⁶ per unit f₂ (P011's Higgsino-equivalent σ_p), but f₂(today) = 0 identically.

## 6. Figures
- `figures/P026_fig1_tau_f2_map.png` — Fig. 1: excluded regions in (τ, f₂^fo) for m = 1 TeV, δ = 300 keV. Left, visible channel:
  CMB anisotropies (red; E_dep > 0.2 eV/baryon), SPI Galactic-centre 300 keV line (orange), LZ exothermic requirement (blue hatched;
  f₂ e^{−t_U/τ} > 1.2 × 10⁻³). Right, invisible channel: only the exothermic requirement applies; dark-photon LZ-fit points at
  m_A′ = 0.5–10 GeV. The Higgsino (τ ≤ 10⁷ s) lies off scale to the left of both panels.
- `figures/P026_fig2_f2_evolution.png` — Fig. 2: f₂(T) from the Boltzmann integration for the Higgsino (δ = 300, 366 keV) and the
  dark photon (δ = 300 keV) versus the equilibrium curve.

## 7. Failed or abandoned approaches
- The first t(z) table (quad to z = ∞ in linear z) lost accuracy above z ≈ 5 × 10⁶ and gave a 12.6 Gyr age; replaced by a log-variable
  integral to z = 10⁶ plus the analytic radiation-era t = 1/(2H), and T(t) at t < 10⁴ s by inverting H_early(T) (asserted monotonic).
- Fermi–Dirac statistics with Pauli blocking for the bath were not implemented; MB statistics keep detailed balance exact (3 % rate effect).
- Hadronic (pion) channels between 1 and 150 MeV and the A′ propagator at E ≳ m_A′ were neglected (irrelevant for T_* ≤ 27 MeV and
  m_A′ ≥ 0.3 GeV except marginally at 0.3 GeV).
- A full Slatyer–Wu-style principal-component evaluation of the CMB bound was not attempted (no CLASS/HyRec offline); the single
  energy-per-baryon criterion is used and flagged.

## 8. Discussion
The excited state of TeV pseudo-Dirac DM is cosmologically *inert* but *not depleted*: half of the DM leaves chemical freeze-out excited and
stays so until T_* ≈ 1 MeV (Higgsino) or 5–30 MeV (dark photon), with f₂ = 0.40–0.50 thereafter, because the same weak-scale coupling that
gives the LZ event decouples from the bath, like neutrinos, while T ≫ δ. What happens next is decided by the decay. The Higgsino χ₂
decays within seconds (dipole) or days (νν̄), injecting 10⁻¹⁴–10⁻¹⁰ of the photon energy density into a fully thermalising bath — invisible
to BBN, FIRAS and Planck; the P007 interpretation needs no cosmological fine print. The dark-photon χ₂ decays only to neutrinos, so the
entire (τ, f₂) plane is cosmologically open, and the *only* constraint is LZ's own: with f₂ ≈ 0.5 surviving, any m_A′ below 2.45 GeV
(δ = 300 keV) leaves ≥ 10⁻³ of the DM excited today and would produce ≥ 1 exothermic event in the same 2.84 t·yr — P011's floor,
confirmed with the computed f₂. The *generic* lesson is a template for later exothermic (P058/P072) and line (P066) papers: a model with a
photon channel and f₂ ~ 0.5 is excluded by Planck for 2 × 10¹² s < τ_γ < 10¹⁹ s and by SPI for τ_γ < 10²² s, so a 300 keV Galactic line
from the LZ dark sector requires either f₂(today) ≲ 10⁻³ (already forced by LZ) and τ_γ ~ 10¹⁹–10²⁰ s, or no photon channel at all.
Spectral distortions cannot help: the stored energy is 10⁻⁷ of the DM mass and at most 1.5 × 10⁻⁷ of the photon energy at z ~ 1000.

## 9. Result files
`P026_results.json` (all numbers), `freeze_out.csv`, `higgsino_decays.csv`, `dark_photon_tau.csv`, `injection_scan.csv`,
`cmb_exclusion_curve.csv`, `recalled_knowledge.json`, `run_log.txt`, `figures/P026_fig1_tau_f2_map.png`, `figures/P026_fig2_f2_evolution.png`.

## 10. References
1. LZ Collaboration, arXiv:2609.02823 (2026).
2. D. Tucker-Smith, N. Weiner, Phys. Rev. D 64, 043502 (2001) — inelastic DM.
3. B. Batell, M. Pospelov, A. Ritz, Phys. Rev. D 79, 115019 (2009) — excited-state decays and exothermic scattering.
4. D. P. Finkbeiner, N. Weiner, Phys. Rev. D 76, 083519 (2007) — excited DM and the 511 keV / de-excitation phenomenology.
5. T. R. Slatyer, C.-L. Wu, Phys. Rev. D 95, 023010 (2017) — CMB constraints on decaying DM (recalled bound).
6. D. J. Fixsen et al., Astrophys. J. 473, 576 (1996) — FIRAS μ, y limits.
7. J. Chluba, R. A. Sunyaev, MNRAS 419, 1294 (2012) — spectral-distortion windows.
8. M. Kawasaki, K. Kohri, T. Moroi, Phys. Rev. D 71, 083502 (2005) — BBN photodissociation thresholds.
9. Corpus: P002, P007, P011 (σ_p(N=1), exothermic counts, width), P014 (χ₂ widths), dossier 00.

## 11. Tools and provenance (mirrors `output/provenance/P026.json`)
- Agent tools: Read (PAPER_GUIDE.md; 00_evidence_dossier.md; results_ledger.csv; papers/P007, P014, P002, P011; work/P014/details.md;
  work/P011/details.md §3.6–3.7; provenance/P014.json; lzcommon.py constants; Fig. 1 ×3, Fig. 2 ×1) — 14 Reads; Bash ×10 (directory
  listings and ENVIRONMENT_versions; cat of P011/P014 result files; grep of lzcommon API; six script runs; three word-count/JSON checks);
  Write ×4 (script, details.md, P026.json, P026.md); Edit ×17 (script 9: z(t)/t(z) mapping ×4, warning suppression ×2, figure labels ×3;
  paper trims ×5; provenance/details tool-count corrections ×3).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (integrate.quad, integrate.solve_ivp LSODA, optimize.brentq); matplotlib 3.11.2 (Agg);
  common/lzcommon.py (RHO0_GEV_CM3, GEV_TO_CM2). No WimPyDD calls; no WimPyDD-generated files. camb not used.
- Recalled knowledge: 17 items in `recalled_knowledge.json`.
- Datasets: none. Data requests: none.
