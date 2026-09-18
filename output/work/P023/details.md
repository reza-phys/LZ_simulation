# P023 — Composite dark-baryon dark matter with a ~300 keV hyperfine splitting: a UV origin for the LZ event?

Research record. Script: `output/code/P023_composite_idm.py` (run from the simulation root with `.venv/bin/python`; the first run takes ~6 min for the WimPyDD grids, later runs use `P023_spectra_cache.npz`). All tables are in `output/work/P023/`, figures in `output/work/P023/figures/`, the run log in `P023_run.log`, the machine-readable results in `P023_summary.json`, and every recalled input with its reliability flag in `recalled_inputs.json` (29 items).

## 1. Motivation and framework

LZ (arXiv:2609.02823) fits its single 248 keV nuclear-recoil-like event with inelastic dark matter at δ = 300–350 keV and m_χ ≥ 400 GeV (Table S7; O₁ and O₄ inelastic both reach 3.3–3.4σ) or with momentum-suppressed spin couplings (L10, O₄/O₆-type). The corpus has so far treated δ as a free parameter: P002 mapped the kinematically allowed (m, δ); P007 showed that a Higgsino fits only at δ = 358–380 keV (1 TeV) and needs PeV gauginos to make δ that small; P011 showed that a dark-photon pseudo-Dirac fermion fits anywhere in δ = 204–387 keV with ε ≈ 10⁻⁷–10⁻³, but that the excited state must have decayed (ε ≳ 5×10⁻⁶); P012 translated the L10 fit into a dimension-8 dipole–dipole contact strength (20.9 GeV)⁻⁴ and excluded elastic photon-mediated dipole DM (N_lo = 376 low-energy companions per high-energy event).

This paper asks whether a *composite* dark-matter state supplies δ ≈ 300 keV without tuning: in QCD every hadron has a spin-flip (hyperfine) partner, and the splitting is set by the constituent masses and the confinement scale, not by an independent parameter. We (i) write the hyperfine scaling laws for three composite regimes, calibrate and validate them on recalled QCD and atomic data, and solve them for δ = 250–400 keV at M = 0.3–4 TeV; (ii) note that a spin-flip transition can only be driven by a spin-dependent operator, so the natural mediators are a magnetic-dipole (M1) transition through the photon or a kinetically mixed dark photon, or an axial Z coupling, and compute the required transition strength with WimPyDD (LZ's code) at δ = 300 and 366 keV; (iii) derive the excited-state lifetime from the same transition moment and confront it with the requirement that the LZ event be a clean single-site nuclear recoil; (iv) evaluate self-interactions and the relic abundance; (v) tabulate three benchmarks.

Conventions: natural units with Heaviside–Lorentz e = √(4πα) = 0.3028; μ_N = e/(2m_p) = 1.0515×10⁻¹⁴ e·cm; WimPyDD isospin couplings c⁰ = c_p + c_n, c¹ = c_p − c_n (P003 convention); LZ exposure 2.84 t·yr; P003 efficiency model (0.96 plateau, erf edges 50 % at 5.4 and 269.9 keV with σ = 3.4 and 8 keV); N_ROI = counts in the whole 5.4–270 keV region, N_hi = counts in 200–270 keV, N_lo = counts in 5.4–55 keV. "Annual" halo = mean of 12 Baxter-2021 SHM halo functions spaced through the year (P007 recipe, explicit v_min grid to 844 km/s); "June 16" = day 167.

## 2. Hyperfine splittings (Part 1)

### 2.1 Formulae

**Coulombic (heavy–heavy) bound state.** For two point-like spin-½ constituents of masses m₁, m₂ and g-factors g₁, g₂ bound by a Coulomb potential with effective coupling α_eff, the Fermi contact interaction splits the 1S level by

  ΔE_hf = (8/3) α_eff⁴ μ_red³/(m₁m₂) · (g₁g₂/4),  μ_red = m₁m₂/(m₁+m₂)   [recalled, certain]

(|ψ_1S(0)|² = (μ_red α_eff)³/π and V_ss = (8π α_eff/3m₁m₂)(g₁g₂/4) S₁·S₂ δ³(r)). For equal masses m_Q and g = 2 this is ΔE_hf = α_eff⁴ m_Q/3 (spin–spin only); a particle–antiparticle pair coupled to the same U(1) adds the annihilation term, giving positronium's 7/12 α⁴ m_e. For a colour-singlet QQ̄ of a dark SU(3) one has α_eff = C_F α_D = (4/3)α_D and the familiar quarkonium form ΔE_hf = 32πα_D|ψ(0)|²/(9m_Q²) → (256/243) α_D⁴ m_Q; for SU(2), C_F = 3/4 (and the QQ "baryon" of SU(2) is the same singlet channel, since 2 ≅ 2̄). Binding energy E_B = α_eff² m_Q/4, Bohr radius a₀ = 2/(α_eff m_Q), Bohr momentum p_B = α_eff m_Q/2, first radial excitation 2S−1S = 3E_B/4.

**Heavy–light hadron (HQET).** For a hadron with one heavy constituent Q and light degrees of freedom, the heavy-quark spin decouples as m_Q → ∞ and the spin-flip splitting is a chromomagnetic 1/m_Q correction: m_{H*}² − m_H² = 4λ₂ ≈ const, i.e. ΔE_hf ≃ K/m_Q with K ∝ Λ² [recalled, certain]. We calibrate K on QCD data: K_B = (m_{B*} − m_B) m_b = 0.189 GeV², K_D = (m_{D*} − m_D) m_c = 0.180 GeV² (mean K_meson = 0.184 GeV² = 1.69 Λ_QCD² with Λ_QCD^(3) = 0.33 GeV); for the Σ_Q–Σ_Q* baryons K_{Σ_b} = 0.084, K_{Σ_c} = 0.082 GeV² (mean K_baryon = 0.083 GeV² = 0.76 Λ²). The consistency of the B/D and Σ_b/Σ_c pairs at the 5 % level is itself the validation of the 1/m_Q law: (B*−B)/(D*−D) = 0.320 and (Σ_b*−Σ_b)/(Σ_c*−Σ_c) = 0.310 versus m_c/m_b = 0.304. For a dark sector we scale K by (Λ_D/Λ_QCD)² and take m_Q ≈ M − 3.3Λ_D (m_B − m_b = 1.1 GeV = 3.3Λ_QCD). Note that with a single light dark flavour the lowest heavy baryon is Σ_Q-like (the two identical light quarks must be in the spin-1 diquark), so both the meson-like and the Σ_Q-like calibrations are relevant; a Λ_Q-like ground state (spin-0 diquark) has no S-wave hyperfine partner. The assignment's alternative Λ_D³/m_Q² estimate was not used: HQET fixes the power to 1/m_Q and the data confirm it.

**Light-constituent baryon.** When all constituents are light (M ~ few Λ_D, a technibaryon), the hyperfine splitting is of order Λ_D: (Δ−N)/N = 0.31 and (Σ*−Σ)/Σ = 0.16 in QCD. A δ/M = 3×10⁻⁷ therefore needs a 10⁻⁶ suppression of the chromomagnetic interaction. The only QCD mechanism for a parametrically small splitting between otherwise degenerate baryons is explicit flavour breaking: Σ⁰−Λ = 77 MeV for m_s − m_ud ≈ 93 MeV (ratio 0.83), so δ = 300 keV would require a dark-quark mass difference Δm_q ≈ 0.36 MeV in a sector with Λ_D ~ 300 GeV, i.e. a 10⁻⁶ hierarchy of a different kind. Neither is "natural".

### 2.2 Validation

| check | formula | measured/expected |
|---|---|---|
| hydrogen 21 cm (m_e, m_p, g_p = 5.5857) | 1418.8 MHz | 1420.4 MHz (leading order 1418.8) |
| positronium 1S (7/12 α⁴ m_e) | 204.4 GHz | 203.4 GHz |
| (Υ−η_b)/(J/ψ−η_c) vs (α_s(m_b)/α_s(m_c))⁴ m_b/m_c | 0.37 | 0.55 |
| Coulombic SU(3) absolute: J/ψ−η_c, Υ−η_b | 28, 10 MeV | 113, 62 MeV |
| (B*−B)/(D*−D), (Σ_b*−Σ_b)/(Σ_c*−Σ_c) vs m_c/m_b | 0.320, 0.310 | 0.304 |

The QCD quarkonia are 4–6× more compact than Coulombic (α_s ~ 0.3 is not perturbative at the Bohr scale), so the α⁴m law is only qualitatively confirmed there; the dark case below has α_eff ≈ 0.03 and a₀Λ_D ≪ 1, i.e. it is far more Coulombic than any QCD system. The heavy–light law is confirmed to 5 %.

### 2.3 Solutions (table `P023_hyperfine_solutions.csv`, Fig. 1)

At M = 1 TeV, δ = 300 keV:

* **Coulombic** (m_Q = M/2 = 500 GeV, spin–spin only): α_eff = (3δ/m_Q)^{1/4} = **0.0366**; E_B = 168 MeV; a₀ = 0.0215 fm; p_B = 9.2 GeV; 2S−1S = 126 MeV. Across the grid α_eff = 0.047–0.053 (300 GeV), 0.035–0.039 (1 TeV), 0.025–0.028 (4 TeV) for δ = 250–400 keV; the one-quarter power makes δ ∈ [250, 400] keV correspond to a ±6 % range of α_eff. If the binding force is a *confining* dark SU(3) with α_D = α_eff/C_F = 0.0275 at p_B, one-loop running (b₀ = 11, n_f = 0 below m_Q) gives Λ_D = p_B exp(−2π/(11α_D)) = **8.5 eV** (SU(2): 220 eV): the "dark baryon" would be a Coulombic state in a theory whose glueballs are eV-scale dark radiation. A Coulombic 300 keV splitting therefore points to a dark U(1) (atomic dark matter, a QQ̄ or Q Q′ "dark atom" with a massless or light dark photon) or to a very weakly coupled non-abelian sector, not to a QCD-like dark baryon.
* **Heavy–light**: Λ_D = Λ_QCD √(δ m_Q/K): **0.42 GeV** (meson-like, K_meson) or **0.63 GeV** (Σ_Q-like, K_baryon). Equivalently, at Λ_D = Λ_QCD the prediction is δ = K/m_Q = **184 keV** (meson-like) or 83 keV (Σ_Q-like) at 1 TeV, and δ = 300 keV is reached at M = 614 GeV (meson-like) or 276 GeV (Σ_Q-like). For δ = 366 keV (P007's Higgsino value) Λ_D = 0.46/0.69 GeV. This is the "B*–B splitting scaled by m_b/m_Q": δ/M ≈ 1.7 (Λ_D/M)², a natural small number with no parameter beyond Λ_D ≈ Λ_QCD.
* **Light baryon**: suppression 9.6×10⁻⁷ relative to Δ−N, or Δm_q = 0.36 MeV explicit breaking.

Fig. 1 (`figures/P023_fig1_delta_vs_mass.png`): δ(M) for the Coulombic law at α_eff = 0.030/0.037/0.045 (blue), the heavy–light law at Λ_D = 0.33/0.42/0.45 GeV (orange) and the Σ_Q-like calibration at 0.42 GeV (green), against the LZ-compatible band (δ = 250–400 keV, M ≥ 259 GeV from P002). The two laws have opposite mass dependence (δ ∝ M at fixed α_eff versus δ ∝ 1/M at fixed Λ_D), so a future measurement of both M and δ would discriminate them.

## 3. Transition cross-sections (Part 2)

### 3.1 Which mediators can drive a hyperfine transition

The two hyperfine partners differ only in the total spin of the constituents (S = 0 ↔ 1 for two spin-½ constituents; J = ½ ↔ 3/2 for a Σ_Q-like baryon). Any interaction that is spin-independent at leading order in the non-relativistic expansion — a scalar (Higgs-portal) exchange, or the charge (time) component of a vector current — has vanishing matrix elements between them, since the spin wavefunctions are orthogonal. Consequently:

* the **Higgs portal** gives only *elastic* scattering (O₁), which P003 excludes for this event (N_lo = 2752 low-energy companions per 200–270 keV event);
* the **vector (charge) current** of a dark photon, which drives P011's pseudo-Dirac O₁ inelastic transition, does *not* connect hyperfine partners; the dark photon couples to them only through the constituents' **magnetic (spatial) current**, i.e. an M1 transition moment μ_D ~ g_D/(2m_const);
* the **Z** couples through its **axial** current (spin-dependent): an O₄-type inelastic transition with G_F-strength coupling if the constituents carry weak isospin;
* a **SM-photon M1 transition** (MiDM, Chang–Weiner–Yavin 2010) arises if the constituents have electric charge or millicharge.

So a hyperfine iDM interpretation of the LZ event is necessarily of the spin-flip, O₄/O₆-type (L10-like) class rather than the O₁ class — consistent with LZ's Table S7, where O₄ inelastic and O₁ inelastic fit equally well at δ = 300–350 keV. A spin-0 ground state (para-state, pseudoscalar dark meson, spin-0 diquark) has no diagonal magnetic moment or axial charge, so it evades the *elastic* dipole and SD constraints (P012; LZ 2024) automatically; a spin-½ ground state would need its diagonal moment ≤ 2×10⁻²⁰ e·cm (P012) — a strong argument for a scalar ground state.

### 3.2 Photon-mediated M1 transition

We use P012's NREFT reduction of L = (μ/2) χ̄₂σ^{μν}χ₁F_{μν} (Anand normalisation, per nucleon; recalled/likely, validated by P012 against LZ's L10 shape): c₁ = eμQ_N/(2m_χ), c₅ = 2eμm_NQ_N/q², c₄ = eμg_N/m_N, c₆ = −eμg_Nm_N/q², with (c_p+c_n, c_p−c_n) fed to WimPyDD, evaluated at μ = 1 μ_N, WimPyDD `diff_rate` with `delta` = δ, m_χ = 1 TeV, E = 1–329 keV (2 keV), annual halo. Rates scale as μ². The spin-½ operator basis is used for what is really a 0 → 1 (or ½ → 3/2) transition; the spin-sum factor differs by O(1) (4/3 for 0→1 versus 3/4-normalised spin-½), which we absorb into the O(1) uncertainty of the composite matrix element.

Validation: at δ = 0 we get N(1 μ_N) = 1.188×10¹² ROI events and N_lo/N_hi = 356, versus P012's 1.193×10¹² and 376 (Sun-frame halo): agreement to 0.4 % in normalisation.

Results (`P023_transition_rates.csv`, Fig. 2 left, Fig. 4):

| δ [keV] | N_ROI(1 μ_N) | μ_tr(N_ROI = 1) [μ_N] | [e·cm] | [e/(2m_χ)] | N_lo per hi event |
|---|---|---|---|---|---|
| 200 | 6.9×10⁸ | 3.8×10⁻⁵ | 4.0×10⁻¹⁹ | 0.041 | 0.26 |
| 250 | 1.5×10⁸ | 8.2×10⁻⁵ | 8.7×10⁻¹⁹ | 0.088 | 6×10⁻⁶ |
| **300** | **2.25×10⁷** | **2.11×10⁻⁴** | **2.2×10⁻¹⁸** | **0.225** | 0 |
| 350 | 7.6×10⁵ | 1.1×10⁻³ | 1.2×10⁻¹⁷ | 1.22 | 0 |
| **366** | **8.7×10⁴** | **3.40×10⁻³** | **3.6×10⁻¹⁷** | **3.62** | 0 |
| 380 | 2.1×10³ | 2.2×10⁻² | 2.3×10⁻¹⁶ | 23 | 0 |

The 90 % single-count band (0.105–3.65 events, P012 convention) multiplies μ_tr by 0.32–1.9. For N_hi = 1 instead: 3.0×10⁻⁴ (300), 3.5×10⁻³ μ_N (366). Mass dependence at δ = 300 keV: μ_tr = 4.0×10⁻⁴ μ_N (400 GeV, 0.17 e/2m_χ), 2.1×10⁻⁴ (1 TeV), 3.0×10⁻⁴ (4 TeV, 1.26 e/2m_χ). June-16 halo: rate ×1.42 (δ = 300) and ×3.16 (366) relative to annual (P006 found 1.41 for O₁ at 300 keV). Compared with P012: the inelastic moment is 14× the *excluded elastic* moment (2.1×10⁻⁵ μ_N for one high-energy event) — allowed here because the low-energy shoulder vanishes for δ ≥ 250 keV — and 7.5× P012's L10-equivalent moment 2.8×10⁻⁵ μ_N.

In units of the DM magneton e/(2m_χ), the moment at δ = 300 keV is 0.22, a plausible size for a composite whose transition moment is carried by a constituent lighter than M; at δ = 366 keV it is 3.6 e/(2m_χ) = 3.4×10⁻³ μ_N, i.e. the moment of a constituent of mass ~ 280 GeV carrying unit electric charge — hard to reconcile with an electrically neutral, collider-safe composite.

### 3.3 Kinetically mixed massive dark photon

For a dark M1 moment μ_D coupled to F′ with kinetic mixing ε, the nucleus couples through ε e and the A′ propagator, so every coefficient above is multiplied by q²/(q²+m_{A′}²) (→ 1 for m_{A′} ≪ q ≈ 0.25 GeV; the on-shell photon coupling then vanishes, cf. P011). WimPyDD with the q-dependent factor gives rate suppressions S = N/N_photon = 0.122 (m_{A′} = 0.3 GeV), 2.19×10⁻³ (1 GeV), 2.96×10⁻⁵ (3 GeV) at δ = 300 keV, so the required εμ_D grows by S^{−1/2} = 2.9, 21, 184.

### 3.4 Translation to composite parameters (`P023_epsilon_translation.csv`)

The SM-visible transition moment is μ_tr = ε √(α_D/α) g_tr e/(2m_const), with m_const the constituent whose spin flips (both heavy constituents in the Coulombic case, m_const = m_Q = 500 GeV; the light constituent in the heavy–light case, m_const ≈ Λ_D, as in D* → Dγ) and g_tr = O(1). Hence ε = (μ_tr/μ_N)(m_const/m_p)√(α/α_D)/g_tr:

| scenario | δ = 300 keV | δ = 366 keV |
|---|---|---|
| A Coulombic dark atom, massless A′ (α_D = α_eff = 0.037, m_const = 500 GeV) | ε = 0.050 | ε = 0.79 |
| B heavy–light, massless A′, α_D = 0.1, m_const = Λ_D | ε = 2.6×10⁻⁵ | 4.6×10⁻⁴ |
| B heavy–light, m_{A′} = 1 GeV | ε = 5.5×10⁻⁴ | 9.7×10⁻³ |
| P011 vector pseudo-Dirac, m_{A′} = 1 GeV, α_D = 0.1 | 8.7×10⁻⁷ | 1.4×10⁻⁵ |

Scenario A with a massless dark photon means millicharged constituents with ε ≈ 0.05 (0.8 at 366 keV) — heavy millicharged particles at this level are constrained by their ionised fraction and by collider/cosmology bounds (recalled, uncertain; not evaluated here); a massive A′ makes it worse by S^{−1/2}. Scenario B needs ε ≈ 10⁻⁵–10⁻³ at δ = 300 keV, 30–600× P011's vector-portal value because the M1 amplitude lacks the A² coherence of the charge coupling, but within the recalled BaBar ceiling ε ≲ 10⁻³ for m_{A′} ≲ 1 GeV; at δ = 366 keV a GeV A′ needs ε ≈ 10⁻², excluded.

**EW-charged constituents (scenario C).** A light-constituent baryon with electrically charged constituents has an M1 moment ~ e/(2Λ_D); with Λ_D = M/3 = 333 GeV this is 2.8×10⁻³ μ_N, which would give 178 events at δ = 300 keV (0.69 at 366 keV): such a state must have EW-neutral constituents, or δ must be near the kinematic edge.

### 3.5 Z-axial spin flip

If the constituents carry weak isospin, the Z's axial current flips their spin with G_F strength. We take c₄^N = 4√2 G_F g_A^χ a_N with g_A^χ = T₃ = ½ (composite matrix element g_tr = 1) and nucleon axial charges a_p = ½(Δu−Δd−Δs) = 0.677, a_n = −0.592 (Δu = 0.842, Δd = −0.427, Δs = −0.085; recalled/likely): c₄^p = 2.23×10⁻⁵, c₄^n = −1.95×10⁻⁵ GeV⁻² (dominantly isovector, WimPyDD c⁰ = 2.8×10⁻⁶, c¹ = 4.2×10⁻⁵). The normalisation of c₄ relative to the Lorentz-invariant operator is a recalled reduction (γ^μγ₅ ⊗ γ_μγ₅ → −4 S_χ·S_N) and carries a factor-2 convention risk; results scale as (g_A^χ a_N)².

WimPyDD O₄ inelastic, 1 TeV, annual halo: **N = 0.64 events at δ = 300 keV**, 0.29 in 200–270 keV; N = 1 at δ = 290 keV; N = 22 at 200 keV, 0.022 at 350, 0.0027 at 366 keV (Fig. 2 right); June-16 halo: 0.92 at 300 keV. For N = 1 at 300 keV one needs g_A^χ = 0.62; at 366 keV, 9.6 (unphysical). So, in sharp contrast with the Z-*vector* (Higgsino) case, where G_F strength gives 790 events at δ = 300 keV (P007) and forces δ to within 20 keV of the kinematic edge, a G_F-strength *axial spin flip* lands within the LZ interval exactly at the tabulated δ = 300 keV. This is because the spin response of xenon (odd isotopes ¹²⁹Xe, ¹³¹Xe, 47 % of natural Xe) lacks the A² coherence of the O₁ response: at 300 keV the O₄ rate per unit c⁴ is ~10³ below O₁ (P007: (c m_v²)² = 0.078 gives 790 events; here c₄ ~ 2×10⁻⁵ ≈ 1.2/m_v² gives 0.64).

Spectra (Fig. 4): the photon-M1 and Z-axial δ = 300 keV spectra peak at ~200 keV with 248 keV on the falling side, as for O₁ (P002); the δ = 366 keV spectrum rises to the edge (the visible jitter is WimPyDD's velocity-grid discreteness close to δ_max, which affects the integrated counts at the few-% level).

## 4. Excited-state lifetime and the single-site requirement (Part 3)

For an M1 transition moment μ_tr between spin-½ states, Γ(χ₂ → χ₁γ) = μ_tr² δ³/π (Chang–Weiner–Yavin 2010; recalled/likely). Check: applied to hydrogen with μ = μ_B and δ = 5.87 μeV it gives A = 8.6×10⁻¹⁵ s⁻¹ versus the measured 2.87×10⁻¹⁵ s⁻¹ — a factor 3.0 from the atomic spin matrix element, so the formula is right to within the O(1) spin-structure factor we already carry. τ = πħ/(μ_tr²δ³).

With the LZ-fixed moments (`P023_chi2_lifetime.csv`):

| δ | μ_tr(N=1) | τ | decay length at 750 km/s | P(decay inside 1 m) | τ/t_U |
|---|---|---|---|---|---|
| 300 keV | 2.1×10⁻⁴ μ_N | 66 μs | 50 m | 2 % | 1.5×10⁻²² |
| 366 keV | 3.4×10⁻³ μ_N | 0.14 μs | 10.5 cm | 100 % | 3×10⁻²⁵ |

Detector logic: the χ₂ leaves the scattering point at essentially the incoming speed (v ≳ v_min(248 keV, δ = 300) ≈ 700 km/s; the nucleus takes < 1 % of the momentum) and crosses the 1.46 m TPC in t_cross = 1.95 μs; the mean remaining chord is ~1 m. If it decays inside, the 300 keV photon converts within ~2 cm (recalled attenuation length, ±50 %) and deposits a ~300 keV electron recoil — a second S1/S2 pair or, if within the S1 window, an ER-like event with S2 ≫ 9268 phd. The recorded event is a clean single-site NR, so P(decay inside) must be small: P = 1 − exp(−L/vτ) ≤ 10 % requires **τ ≥ 12.7 μs** (L = 1 m). Because the LZ rate fixes μ_tr(δ) and τ ∝ 1/(μ_tr²δ³), this becomes a **ceiling on δ** for any photon-M1 hyperfine model (`P023_single_site_ceiling.csv`): δ ≤ **326 keV** for N = 1 (306 keV at the 3.65-event edge, 350 keV at the 0.105-event edge); requiring only τ ≥ t_cross gives 347/333/364 keV. The P007 Higgsino-like window δ = 358–380 keV is therefore not available to a photon-M1 composite: at 366 keV the χ₂ decays after 10 cm.

At δ = 300 keV the moment that would make the decay "prompt" (τ = 2 μs) is 1.2×10⁻³ μ_N, 6× the LZ value (33× in rate). Conversely τ/t_U ~ 10⁻²², so cosmologically every χ₂ has decayed and no exothermic (down-scattering) population exists — P011's ε ≳ 5×10⁻⁶ floor does not arise in the photon-M1 case. For the *massive* dark-photon variant the on-shell photon decouples, χ₂ → χ₁γ is absent, and the lifetime is P011's χ₂ → χ₁νν̄ value (~10¹⁸ s at ε = 10⁻⁶): then P011's exothermic floor applies and the single-site requirement is trivially met. The Z-axial case decays through Z* → νν̄ with P007's τ ~ 10⁶ s at δ = 300 keV: also compatible with both requirements.

Fig. 3 (`figures/P023_fig3_lifetime.png`): τ(μ_tr) for δ = 300 and 366 keV, the LZ 90 % band at 300 keV, the 12.7 μs detector line and the age of the Universe.

## 5. Cosmology and self-interactions (Part 4; `P023_cosmology.csv`)

**Asymmetric relic.** n_DM/s = Ω_DM ρ_c/(M s₀) = 4.4×10⁻¹³ at 1 TeV (1.5×10⁻¹² at 300 GeV, 1.1×10⁻¹³ at 4 TeV), i.e. η_D = 5.0×10⁻³ η_B: the dark asymmetry must be 200× smaller than the baryon asymmetry, not equal to it (equal asymmetries give the classic 5 GeV ADM mass). A TeV composite is asymmetric DM with a suppressed asymmetry; the symmetric component annihilates away efficiently in every scenario (below).

**Scenario A (Coulombic).** Size ~ a₀ = 0.0215 fm: σ/m = 4πa₀²/M = 3.3×10⁻⁸ cm²/g. Constituent annihilation ⟨σv⟩ = πα_eff²/m_Q² = 2.0×10⁻²⁵ cm³/s (recalled/likely formula, no Sommerfeld or bound-state enhancement): a symmetric thermal relic would give Ω_sym h² = 0.013, 11 % of the DM, so the bulk must be asymmetric. Conversely, the coupling that would make the constituents a *thermal* relic, α = 0.0122, gives δ_hf = α⁴m_Q/3 = 3.7 keV — a thermal Coulombic dark atom cannot have a 300 keV splitting. Dark recombination must be efficient (Kaplan et al. 2010) and the millicharge implied by §3.4 is a further constraint.

**Scenario B (heavy–light).** σ/m = 4π/(Λ_D²M) = 1.6×10⁻⁵ cm²/g with Λ_D = 0.42 GeV — negligible for small-scale structure. Strong annihilation ⟨σv⟩ ~ (4π/Λ_D²)v_FO = 2.5×10⁻¹⁶ cm³/s leaves a symmetric remnant of 9×10⁻¹¹, so no indirect-detection signal. Dark glueballs at ~5.5Λ_D ≈ 2.3 GeV and the light-flavour excitations (~Λ_D) must decay before BBN through the portal (Forestell–Morrissey–Sigurdson 2017) — the main cosmological constraint on this scenario, not evaluated quantitatively here.

**Scenario C (light baryon).** σ/m = 2.5×10⁻¹¹ cm²/g; symmetric remnant 6×10⁻⁵ of the DM, whose annihilation with ⟨σv⟩ ~ 10⁻¹⁹ cm³/s (v ~ 10⁻³) gives a signal ~ (6×10⁻⁵)² × 10⁻¹⁹/2.2×10⁻²⁶ ~ 10⁻² of a thermal WIMP's — marginal but not excluded by recalled limits.

## 6. Benchmarks (Part 5; `P023_benchmarks.csv`)

| | A: Coulombic dark atom / SU(2)_D diquark | B: heavy–light dark hadron (SU(3)_D × U(1)_D) | C: light-constituent dark baryon |
|---|---|---|---|
| parameters | m_Q = 500 GeV, α_eff = 0.037, E_B = 168 MeV | m_Q ≈ 1 TeV, one light dark quark, Λ_D = 0.42 GeV (Σ_Q-like: 0.63) | Λ_D ~ M/3 |
| δ | 300 keV; ∝ M α⁴ | 300 keV = (B*−B)·m_b/m_Q·(Λ_D/Λ_QCD)²; ∝ 1/M | needs 10⁻⁶ suppression or Δm_q = 0.36 MeV |
| μ_tr needed (1 TeV, 300 keV) | 2.1×10⁻⁴ μ_N | 2.1×10⁻⁴ μ_N | 2.1×10⁻⁴ μ_N |
| portal | ε = 0.05 millicharge (massless A′) | ε = 2.6×10⁻⁵ (massless A′) or 5.5×10⁻⁴ (1 GeV), α_D = 0.1 | EW-charged constituents give 178 events → must be neutral; Z-axial (T₃ = ½) gives 0.64 events |
| τ(χ₂ → χ₁γ) | 66 μs (P_in = 2 %) | 66 μs (massless A′); ~10¹⁸ s for massive A′ (P011 regime) | ~10⁶ s (Z, P007) |
| relic | asymmetric, η_D = 4.4×10⁻¹³; thermal α would give δ = 3.7 keV | asymmetric; symmetric remnant 10⁻¹⁰ | asymmetric; remnant 6×10⁻⁵ |
| σ/m | 3×10⁻⁸ cm²/g | 1.6×10⁻⁵ cm²/g | 2.5×10⁻¹¹ cm²/g |
| distinguishing | single hyperfine partner (no tower at 2δ, 3δ); 2S at 126 MeV; δ ∝ M; SU(3)_D version confines at 8.5 eV (dark radiation) | glueballs ~2.3 GeV, light-quark excitations ~0.5 GeV; δ ∝ 1/M (184 keV at 1 TeV if Λ_D = Λ_QCD); ε ~ 10⁻⁵–10⁻³ testable | no natural 300 keV scale; tower at ~Λ_D |

In all hyperfine scenarios there is exactly one spin-flip partner (S = 0 ↔ 1 or J = ½ ↔ 3/2), so a "tower of excitations at multiples of δ" is *not* predicted; higher excitations are at 2S−1S ≈ 126 MeV (A) or ~Λ_D (B), far beyond the direct-detection kinematic ceiling (δ_max ≈ 387 keV at 1 TeV, P002). A rotational tower with 3δ, 6δ spacings would instead indicate a dark nucleus or a large-N state.

## 7. Robustness and failed approaches

* Halo: June-16 versus annual changes N by 1.42 (δ = 300) and 3.16 (366), i.e. μ_tr by 0.84 and 0.56; the δ ceiling of §4 moves by ≲ 10 keV.
* Efficiency: P003's σ_hi = 8 keV; P007/P011 found ≤ 3 % differences for σ_hi = 11.5 keV at δ ≤ 350 keV, more near the edge.
* The O(1) spin-structure factor between the spin-½ MiDM operator basis and a genuine 0 → 1 transition, and the O(1) composite matrix elements g_tr, g_A^χ, are the dominant uncertainties: rates ∝ g²; the lifetime and rate depend on the *same* μ_tr, so the τ(δ) ceiling is insensitive to them (τ ∝ 1/N at fixed δ: a factor 2 in the spin factor moves the ceiling by ~5 keV).
* The c₄ normalisation for the Z-axial case carries a factor-2 convention risk (N = 0.16–2.6 at δ = 300 keV for c₄ × 0.5–2).
* WimPyDD velocity-grid jitter is visible in the δ = 366 keV spectrum; integrated counts change by a few per cent between adjacent grid choices (P011 found similar).
* Not done: the Λ_D³/m_Q² alternative scaling of the assignment (superseded by HQET); millicharge bounds for scenario A; glueball decay cosmology for scenario B; a genuine 0 → 1 NREFT with spin-1 response (WimPyDD supports j_χ but not mixed-spin transitions).
* No abandoned computations; a first draft used an undefined variable (`sub`) in the single-site block before the figure section defined it, corrected before running; the Fig. 3 label was hard-coded and replaced by the computed 12.7 μs.

## 8. Discussion

1. δ ≈ 300 keV at 1 TeV is *natural* in exactly one composite regime: a heavy dark quark dressed by light dark degrees of freedom of a QCD-like sector, where δ = (B*−B)(m_b/m_Q)(Λ_D/Λ_QCD)² requires Λ_D = 0.42 GeV (0.63 GeV for a Σ_Q-like baryon). A Coulombic dark atom also works with one small coupling (α ≈ 0.037), but a confining SU(N) at that coupling has an eV-scale Λ_D, and a thermal symmetric history is incompatible with the splitting; a light-constituent dark baryon needs a 10⁻⁶ tuning.
2. Because hyperfine partners are connected only by spin-dependent operators, the LZ interpretation of such a model is of the O₄/O₆ (M1 or axial) class: the Higgs portal is out, and the dark-photon route is an M1 (not charge) transition, requiring 30–600× larger ε than P011's pseudo-Dirac fermion.
3. The M1 moment that gives one LZ event fixes the excited-state lifetime; demanding that the event be a single-site NR caps δ ≤ 326 keV (306–350). Together with P002's spectral preference for δ ≥ 385 keV this is a tension: photon-M1 hyperfine models can have the LZ event only in δ ≈ 250–330 keV, where 248 keV sits at the ≥ 98th percentile of the spectrum.
4. A G_F-strength axial Z coupling — natural for constituents with weak isospin — gives 0.64 (June: 0.92) events at exactly δ = 300 keV: a "weak-scale, weak-strength" inelastic explanation that, unlike the Higgsino, needs no proximity to the kinematic edge. Its low-energy companions vanish for δ ≥ 260 keV.
5. Predictions: exactly one partner; δ ∝ 1/M (B) or ∝ M (A); ε ~ 10⁻⁵–10⁻³ GeV-scale dark photons (B); a de-excitation photon *after* the χ₂ has left the detector (66 μs, 50 m) — invisible to LZ but, for a photon-M1 model, a monochromatic ~δ line from up-scattering in the Galactic Centre is in principle a target; June-phased events (×1.4 at 300 keV).

## 9. References

1. LZ Collaboration, arXiv:2609.02823 (2026).
2. D. S. M. Alves, S. R. Behbahani, P. Schuster, J. G. Wacker, "Composite inelastic dark matter", Phys. Lett. B 692, 323 (2010) [arXiv:0903.3945].
3. S. Chang, N. Weiner, I. Yavin, "Magnetic inelastic dark matter", Phys. Rev. D 82, 125011 (2010).
4. D. E. Kaplan, G. Z. Krnjaic, K. R. Rehermann, C. M. Wells, "Atomic dark matter", JCAP 05 (2010) 021.
5. G. D. Kribs, T. S. Roy, J. Terning, K. M. Zurek, "Quirky composite dark matter", Phys. Rev. D 81, 095001 (2010).
6. O. Antipin, M. Redi, A. Strumia, E. Vigiani, "Accidental composite dark matter", JHEP 07 (2015) 039.
7. J. Bagnasco, M. Dine, S. Thomas, "Detecting technibaryon dark matter", Phys. Lett. B 320, 99 (1994).
8. L. Forestell, D. E. Morrissey, K. Sigurdson, "Non-Abelian dark forces and the relic densities of dark glueballs", Phys. Rev. D 95, 015032 (2017).
9. M. Neubert, "Heavy-quark symmetry", Phys. Rept. 245, 259 (1994).
10. N. Anand, A. L. Fitzpatrick, W. C. Haxton, Phys. Rev. C 89, 065501 (2014); A. L. Fitzpatrick et al., JCAP 02 (2013) 004.
11. Corpus: P002, P003, P006, P007, P011, P012; dossier 00.

(Bibliographic details of refs. 2–9 are recalled, reliability likely.)

## 10. Tools and provenance

Mirrors `output/provenance/P023.json`.

* Agent tools: Read ×14 (PAPER_GUIDE.md; 00_evidence_dossier.md; results_ledger.csv; papers P002, P003, P007, P011, P012, P014; provenance/P011.json; lzcommon.py lines 1–125 and 300–373; P012 script lines 36–59 and 220–259; figures ×4); Bash ×17 (ls/versions; library API grep; P011/P012/P007 result extraction ×3; P012 summary + plan grep; P012/P007/P003 script grep; WimPyDD timing test; full run 354 s; cached re-run + table prints; dataviz palette grep; word-count checks ×6); Write ×5 (script, details.md, P023.json, P023.md twice); Edit ×20 (script: single-site block, Fig. 3 label, summary key, `scan_ph`; paper: 14 word-budget trims, one a no-op; tallies in details.md and P023.json ×2); Skill ×1 (dataviz).
* Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (special.erf, optimize.brentq); pandas 3.0.5; matplotlib 3.11.2 (Agg); WimPyDD 2.0.4 (eft_hamiltonian with q- and m_χ-dependent coefficients, streamed_halo_function via lz.wd_halo with explicit v_min grid, diff_rate with delta > 0); common/lzcommon.py (LZ constants, M_NUCLEON_GEV, HBARC, GEV_TO_CM2, VESC_KMS, wd, wd_halo, wd_hamiltonian, wd_rate); hand derivations (hyperfine formulae, HQET calibration, ε translation, lifetime/geometry, relic algebra).
* Script: `output/code/P023_composite_idm.py` — `.venv/bin/python output/code/P023_composite_idm.py` (first run 354 s; `--recompute` to redo the WimPyDD grids).
* Local inputs: LZ paper numbers via lzcommon (exposure, efficiency edges, event energy); dossier (context); ledger; P002 (δ_max, m_min, spectral percentiles); P003 (convention, efficiency, N_lo criterion); P007 (Higgsino N(δ), χ₂ lifetime, halo recipe); P011 (σ_p(N=1), ε values, χ₂ lifetime and exothermic floor, A′ propagator convention); P012 (NREFT photon-dipole coefficients, elastic μ = 2.1×10⁻⁵ μ_N, L10-equivalent 2.8×10⁻⁵ μ_N, N(1 μ_N) = 1.193×10¹²); P014 (format model; Higgsino χ₂ lifetimes); ENVIRONMENT_versions.txt.
* Recalled knowledge: 29 items in `recalled_inputs.json` (constants and hadron masses certain; Λ_QCD, α_s values, axial charges, MiDM width, Coulombic annihilation, glueball mass, TPC dimensions likely; LXe attenuation length and NREFT dipole coefficients likely/uncertain; bibliographic details likely).
* Datasets: none. Data requests: none. WimPyDD-generated files: none (diff_rate kernels only).
