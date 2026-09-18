# P084 — Sommerfeld-enhanced annihilation and the CMB, dwarf-spheroidal and antiproton constraints on the electroweak multiplets that fit the LZ event — research record

*Simulated date 2026-09-16. Author profile: indirect-detection phenomenology group. Category COMP (MODEL content); hep-ph, cross-list astro-ph.HE. Result type: computed (cross-sections, Sommerfeld factors), estimated (limit comparison: every experimental limit is recalled). Script `output/code/P084_sommerfeld_indirect.py` (runs in 26 s from the root); every number below is printed in `output/work/P084/run_log.txt` or stored in the CSV/JSON tables listed in Sec. 9. Competes with P025 (indirect constraints, Higgsino only, Sommerfeld bracketed S = 1–3); builds on P037 (multiplet fits), P007 (Higgsino), P054 (Z′, non-thermal option). P075 (thermal history) had not appeared at the time of writing, so thermal-history remarks cite P054/P025 only.*

## 1. Motivation and question

P037 showed that every SU(2)_L fermion multiplet with hypercharge Y ≠ 0 scatters in LZ like a Higgsino with σ_n = 7.4×10⁻³⁹(2Y)² cm², and that one event at 1 TeV requires δ(N=1) = 365.8, 374.1, 379.2, 382.7 keV for the (n, Y) = (2, ½), (3, 1), (4, 3/2), (5, 2) multiplets (windows [358–387], [366–387], [371–387], [375–387] keV). P025 confronted only the Higgsino with indirect probes and bracketed its Sommerfeld factor as S = 1–3. Here we (i) compute the tree-level s-wave annihilation of the neutral state of each multiplet into W⁺W⁻, ZZ, Zγ, γγ; (ii) solve the coupled-channel Schrödinger equation for the Sommerfeld enhancement with W, Z and photon exchange and the one-loop charged–neutral splittings, as a function of velocity (Milky Way v_rel ≈ 10⁻³, dwarfs 10⁻⁵–10⁻⁴, CMB v → 0) and mass (0.5–12 TeV, resonance structure); (iii) compare with recalled Planck, Fermi-LAT dwarf, H.E.S.S. continuum and line, and AMS-02 antiproton limits; (iv) fold in the thermal relic fraction f = Ω/Ω_DM of an under-abundant multiplet (indirect ∝ f², LZ ∝ f); (v) check P025's Higgsino numbers and ask whether any LZ-fitting mass sits on a Sommerfeld resonance.

## 2. Framework and derivations

### 2.1 Multiplets and two-body channels

We take the four P037 multiplets with Y = T = (n−1)/2, whose neutral component is the bottom state T₃ = −Y (charges Q = T₃ + Y = 0, 1, …, n−1). The neutral Dirac fermion ψ⁰ is split into Majorana χ₁, χ₂ by δ ≈ 0.37 MeV, which is negligible for annihilation (Sec. 2.5); we therefore work in the Dirac basis. The Q_tot = 0, ¹S₀ two-body sector that mixes with ψ⁰ψ̄⁰ under gauge-boson exchange is {ψ^(Q) ψ̄^(Q)}, Q = 0 … n−1 (n channels). Because C_−² ≡ |⟨T₃−1|T₋|T₃ = −T⟩|² = 0, the ψ⁰ψ⁰ and ψ̄⁰ψ̄⁰ states neither annihilate into gauge bosons nor couple to charged pairs; they carry half of the χ₁χ₁ norm and are inert, exactly as in P025's Higgsino counting.

Couplings (g = 0.6516, α₂ = 1/29.6, s_W² = 0.2312, c_W = 0.8768 [recalled, certain]):
- W: (g/√2) C ψ̄^(Q+1)γ^μψ^(Q) W⁺_μ with C² ≡ C₊²(Q) = (T − T₃)(T + T₃ + 1) = (n − 1 − Q)(Q + 1).
- Z: g_Z(Q) = (g/c_W)(T₃ − Q s_W²) = (g/c_W)(Q c_W² − Y). The neutral state has g_Z(0) = −gY/c_W (P037).
- photon: e Q, e = g s_W (α(m_Z) = α₂ s_W² = 1/128 for annihilation vertices; α(0) = 1/137.04 for the long-range Coulomb potential).

| multiplet | Q | g_Z/(g/c_W) | C₊²(Q→Q+1) | Δm(Q) [GeV] (Sec. 2.3) |
|---|---|---|---|---|
| (2, ½) | 0, 1 | −0.5, 0.269 | 1 | 0, 0.358 |
| (3, 1) | 0, 1, 2 | −1, −0.231, 0.538 | 2, 2 | 0, 0.549, 1.432 |
| (4, 3/2) | 0…3 | −1.5, −0.731, 0.038, 0.806 | 3, 4, 3 | 0, 0.740, 1.814, 3.222 |
| (5, 2) | 0…4 | −2, −1.231, −0.462, 0.306, 1.075 | 4, 6, 6, 4 | 0, 0.930, 2.195, 3.794, 5.728 |

### 2.2 Tree-level annihilation

Threshold s-wave amplitudes with t- and u-channel fermion exchange have the structure of e⁺e⁻ → γγ. Para-positronium fixes the normalisation: σv(¹S₀ e⁺e⁻ → γγ) = Γ/|ψ(0)|² = (α⁵m/2)/(m³α³/8π) = 4πα²/m² [certain], and the spin-averaged value is πα²/m². With a_j ≡ product of the two couplings of diagram j (a = e² for each of the two positronium diagrams),

  σv_spin-avg(ψψ̄ → V₁V₂) = (Σ_j a_j)² / (32π m²) × (½ if V₁ = V₂).

For the Majorana χ₁χ₁ initial state this same expression (with the Dirac-basis diagrams of ψ⁰ψ̄⁰) reproduces P025's Higgsino results, πα₂²/8m² (WW) and πα₂²/16c_W⁴m² (ZZ), and the wino 2πα₂²/m² [recalled, likely], so it is our normalisation. Channel amplitude vectors (the same Σa for each channel Q, used again for the Sommerfeld matrix):

  a^WW(Q) = (g²/2)[C₊²(Q−1) + C₊²(Q)],  a^ZZ(Q) = 2 g_Z(Q)²,  a^γγ(Q) = 2e²Q²,  a^Zγ(Q) = 2 e Q g_Z(Q).

s-channel γ/Z exchange does not contribute to ¹S₀ (J^PC = 0⁻⁺ cannot couple to one vector boson). For the neutral channel a^WW(0) = (g²/2)(n−1), a^ZZ(0) = 2g²Y²/c_W², a^γγ(0) = a^Zγ(0) = 0, hence

  ⟨σv⟩_WW = πα₂²(n−1)²/(8m²),  ⟨σv⟩_ZZ = πα₂² Y⁴/(c_W⁴ m²),  ⟨σv⟩_γγ = ⟨σv⟩_Zγ = 0 at tree level,

times phase-space factors (1 − m_V²/m²)^{1/2} (≤ 0.4 %). Numbers at 1 TeV (`P084_tree_xsec.csv`; 1 GeV⁻² = 1.1673×10⁻¹⁷ cm³ s⁻¹ [certain]):

| multiplet | WW | ZZ | total | ZZ/WW |
|---|---|---|---|---|
| (2, ½) | 5.22e-27 | 4.41e-27 | 9.62e-27 | 0.85 |
| (3, 1) | 2.09e-26 | 7.05e-26 | 9.14e-26 | 3.4 |
| (4, 3/2) | 4.69e-26 | 3.57e-25 | 4.04e-25 | 7.6 |
| (5, 2) | 8.34e-26 | 1.13e-24 | 1.21e-24 | 13.5 |

At the recalled thermal masses: 7.96e-27 (1.1 TeV doublet), 2.29e-26 (2 TeV triplet), 7.04e-26 (2.4 TeV quadruplet), 6.01e-26 cm³/s (4.5 TeV quintuplet). The doublet numbers agree with P025 (9.6e-27, 7.9e-27) to 0.3 %. The ZZ channel grows as Y⁴ and dominates for Y ≥ 1: the Z coupling of a Y = 2 neutral state is 2g/c_W = 2.3 g.

Recalled formula from Cirelli–Fornengo–Strumia (2006) used as a cross-check: the wino/real-multiplet result σv_WW = g⁴(n²−1)²/(512πM²) [recalled, likely] is reproduced by our C₊² algebra for (3,0) (2πα₂²/M²) and (5,0) (18πα₂²/M²). The CFS full freeze-out formula (all coannihilations) is not used here beyond P025's validation of the doublet thermal mass.

### 2.3 Charged–neutral splittings (channel thresholds)

The one-loop gauge splitting in the heavy-mass limit follows from δM_i = −(1/8π) Σ_V g_{V,i}² m_V: with the W weight g²[T(T+1) − T₃²], Z weight (g²/c_W³)(T₃ − Q s_W²)² m_W and zero photon weight,

  M_Q − M_0 = α₂ m_W [ Q² sin²(θ_W/2) + Q Y (1 − c_W)/c_W ]  (Cirelli–Fornengo–Strumia 2006 form; recalled, likely; re-derived here).

Checks: wino Q = 1: 166 MeV (known 165–170 MeV) [certain]; Higgsino Q = 1: 358 MeV (P007: 355 MeV; P037: 342 MeV with finite-mass corrections); (3,1), (4,3/2), (5,2) Q = 1: 549, 740, 930 MeV versus P037's 511, 680, 850 MeV (P037's finite-M/m_W form is 7 % lower; the effect on S is negligible because the charged channels are closed by ≫ the kinetic energy for all velocities of interest). Higher charges: Table 2.1.

### 2.4 Coupled-channel Sommerfeld problem

For reduced mass m/2 and relative velocity v_rel (kinetic energy E = m v_rel²/4, momentum k = m v_rel/2), the radial s-wave functions u_Q(r) obey

  u″ = m [ V(r) + diag(2Δm_Q) − E ] u,
  V_QQ = −[ α(0) Q² + g_Z(Q)²/4π · e^{−m_Z r} ] / r,   V_{Q,Q+1} = −(α₂/2) C₊²(Q) e^{−m_W r}/r.

Off-diagonal entries are single W exchange between the pair (one diagram for distinguishable ψ^(Q)ψ̄^(Q)); diagonal entries are attractive photon and Z exchange between particle and antiparticle. For real multiplets (Y = 0, used only as solver checks) the identical-particle state |χ⁰χ⁰⟩ carries the usual √2: V_{0,1} → √2 (α₂/2) C₊²(0) = √2 α₂ for the wino, V_{11} = −α/r − α₂ c_W² e^{−m_Z r}/r + 2Δm, exactly Hisano–Matsumoto–Nojiri–Saito (2005) [recalled, likely], and the annihilation vector becomes (a₀/√2, a₁) ∝ (√2, 1) for WW, i.e. their Γ ∝ [[2, √2],[√2, 1]]. Our code prints both checks (run log Sec. 1).

Enhanced cross-sections: with s_Q ≡ u_Q′(0) for the solution with unit incoming flux in channel 0 (free: s = (1, 0, …)),

  σv_X = |Σ_Q s_Q a^X(Q)|² /(32π m²) × (½ if identical),  S_X = σv_X/σv_X^tree,  S_tot = (σv_WW + σv_ZZ)/(σv_WW + σv_ZZ)_tree.

The γγ and Zγ lines come entirely from the charged components s_{Q≥1}: this is the Hisano mechanism that produces the wino line. We quote the line-equivalent ⟨σv⟩_γγ + ½⟨σv⟩_Zγ (2 photons per γγ, 1 per Zγ) [certain convention].

Numerical method (not Hulthén): inward integration from r_max = 12/m_W = 0.149 GeV⁻¹ (where the Yukawa potentials are e^{−12} ≈ 6×10⁻⁶ of their scale; the residual Coulomb tail acts only in closed channels) of n+1 basis solutions — pure decaying e^{−κ_Q r} in each closed channel (κ_Q = [m(2Δm_Q − E)]^{1/2}), pure outgoing in any open charged channel, e^{±ikr} in channel 0 — with scipy `solve_ivp` (DOP853, rtol 10⁻¹⁰, complex arithmetic), down to r₀ = 0.02/m; the (n+1)×(n+1) linear system {u(r₀) = 0, incoming amplitude in channel 0 = that of sin(kr)/k} fixes the physical solution and s = u′(r₀). Inward integration is stable because the outward-growing closed-channel modes decay inward. One evaluation costs 0.02–0.04 s.

### 2.5 The inelastic splitting δ

In the Majorana basis the Z exchange is off-diagonal, χ₁χ₁ ↔ χ₂χ₂, with the χ₂χ₂ channel closed by 2δ = 0.73 MeV at v_rel ≲ 1.7×10⁻³; W exchange couples both to χ⁺χ⁻ with strength (α₂/2)C₊²/√2. Rotating the Dirac-basis Hamiltonian gives V_{11,22} = −α_Z e^{−m_Z r}/r, V_{11,11} = V_{22,22} = 0, a_{11} = a_{22} = a₀/√2 (details: ⟨χ₁χ₁| = (⟨ψψ̄|_sym + ⟨inert|)/√2, |χ₂χ₂⟩ = (|ψψ̄⟩ − |inert⟩)/√2). Slatyer (2010) [recalled, likely] showed the elastic result holds when δ ≪ α_Z m_Z and δ ≪ α²m/4; here 2δ/(α_Z m_Z) = 7×10⁻⁴ (doublet, α_Z = 0.011) to 5×10⁻⁵ (quintuplet, α_Z = 0.176) and 2δ/(α₂²m) ≈ 6×10⁻⁴. We verified this numerically (Sec. 4.1): the 3-channel Majorana calculation with δ = 366 keV differs from the Dirac-basis result by 0.2 % (1 TeV) and 1.6 % (3 TeV, v = 10⁻⁶).

## 3. Inputs

- Electroweak constants, conversion factor, positronium rate: recalled, certain.
- Wino tree cross-section and Hisano matrices: recalled, likely. Wino first resonance 2.3–2.5 TeV: recalled, likely. Wino 3 TeV line ~(1–3)×10⁻²⁶ cm³/s: recalled, uncertain.
- Hulthén closed form (Cassel 2010; Slatyer 2010) as in P025 §3.2: recalled, likely.
- Thermal masses: doublet 1.1 TeV (certain; P025 1.17 TeV without Sommerfeld), (3,1) ≈ 2 TeV, (4,3/2) ≈ 2.4 TeV, (5,2) ≈ 4.5 TeV (P037's recalled MDM-literature values, uncertain ±30 %; Sommerfeld effects at freeze-out push the true values up, e.g. (5,0) 4.4 → 9.4 TeV [likely]).
- δ(N=1): P037 (1 TeV: 365.8/374.1/379.2/382.7 keV; 2 TeV: 384.7 (3,1), 390.8 (4,3/2); 4 TeV: 398.1 (5,2)); P007 (doublet 2 TeV: 375 keV). Interpolated labels (not computed here): doublet 1.1 TeV ≈ 367 keV, (4,3/2) 2.4 TeV ≈ 392 keV, (5,2) 2 TeV ≈ 393 keV, 4.5 TeV ≈ 398 keV. The "×5 per 10 keV" fall of N(δ) near δ_max is P007/P037's (local input).
- Recalled limits (WW-like continuum unless stated), anchor at 1 TeV, mass scaling, band:
  - Fermi-LAT dSphs: 1×10⁻²⁵ cm³/s, ∝ m, band (0.5–2)×10⁻²⁵ — likely (×2).
  - H.E.S.S. Galactic centre, Einasto: 2×10⁻²⁶, ∝ m^0.5, band (0.5–4)×10⁻²⁶ (assignment "few 10⁻²⁷–10⁻²⁶"; P025 4×10⁻²⁶) — uncertain; cored profiles ×10 weaker.
  - AMS-02 antiprotons: 3×10⁻²⁶, ∝ m, band (1–10)×10⁻²⁶ — uncertain (propagation).
  - Planck: p_ann = f_eff⟨σv⟩/m < 3.5×10⁻²⁸ cm³ s⁻¹ GeV⁻¹ with f_eff(WW) = 0.35 → ⟨σv⟩ < 1.0×10⁻²⁴ (m/TeV); band to 1.6×10⁻²⁴ (p_ann 3.2×10⁻²⁸, f_eff 0.2) — likely.
  - H.E.S.S. line (γγ, Einasto): 2×10⁻²⁸, ∝ m, band (1–4)×10⁻²⁸ — uncertain.
  Velocities: Milky Way v_rel = 10⁻³ (range 3×10⁻⁴–2×10⁻³), dwarfs 3×10⁻⁵ (and 10⁻⁴), CMB 10⁻⁶ (saturated) — certain scales.

## 4. Results

### 4.1 Solver validation (run log Sec. 1; `P084_results.json["validation"]`)

| test | numerical | reference | ratio |
|---|---|---|---|
| Coulomb α = 0.01, v_rel = 0.1 (r_max = 480/m_W) | 1.3433 | 2πη/(1−e^{−2πη}) = 1.3468 | 0.997 |
| Coulomb α = 0.03, v = 0.05 / 0.02 | 3.832 / 9.476 | 3.859 / 9.426 | 0.993 / 1.005 |
| Yukawa (m_W) α = 0.03, m = 2 TeV, v = 10⁻³ / 10⁻² | 5.970 / 5.591 | Hulthén 6.126 / 5.811 | 0.975 / 0.962 |
| Yukawa α = 0.05, m = 2 TeV, v = 10⁻³ | 41.72 | Hulthén 46.94 | 0.889 |
| (5,2) 1 TeV: r₀ × 0.25 / × 4; r_max 18/m_W, 8/m_W; rtol 10⁻⁸ | S_tot ratio 1.016 / 0.988; 1.00000 / 0.9997; 1.00000 | | |
| wino matrices | V₀₁ = 1.4142 α₂; V₁₁^Z = 0.7688 α₂; a ∝ (1.414, 1) | Hisano √2, c_W², (√2, 1) | exact |
| wino first resonance | 2.37 TeV (grid, ±3 %) | 2.3–2.5 TeV (recalled) | ✓ |
| wino 3 TeV | S_MW = 55; ⟨σv⟩ = 5.2×10⁻²⁵; line 3.2×10⁻²⁶ | line ~(1–3)×10⁻²⁶ (recalled, uncertain) | ✓ (×1–3) |
| wino 1 TeV | S = 3.4 | | |
| doublet Dirac vs Majorana δ = 0 (1 TeV, 3 TeV) | 1.66265 vs 1.66265; 5.82147 vs 5.82147 | identical | 1.0000 |
| doublet Majorana δ = 366 keV, v = 10⁻³ (10⁻⁶) | 1.6594 (1.6586); 5.8236 (5.7326) | Dirac 1.6627; 5.8215 (5.8231) | 0.998; 1.000 (0.984) |

The Hulthén form is known to be a 10 % approximation to the Yukawa problem, so the 3–11 % differences are the expected size (Hulthén overestimates S, as noted in the literature for α m/m_φ near a resonance). The Coulomb test at 0.5 % validates the matching; the r₀ variation (±1.5 %) is the numerical accuracy we quote.

### 4.2 Sommerfeld factors at the LZ-fitting benchmarks (`P084_sommerfeld_benchmarks.csv`)

| multiplet | m [TeV] | δ(N=1) [keV] | ⟨σv⟩_tree | S_MW (v = 10⁻³) | S_WW / S_ZZ | S_dwarf (3×10⁻⁵) | S_CMB (10⁻⁶) | ⟨σv⟩_MW | line (γγ+½Zγ) |
|---|---|---|---|---|---|---|---|---|---|
| (2, ½) | 1.0 | 366 | 9.62e-27 | 1.663 | 1.83 / 1.47 | 1.663 | 1.663 | 1.60e-26 | 1.19e-28 |
| (2, ½) | 1.1 | 367 | 7.96e-27 | 1.755 | 1.94 / 1.54 | 1.755 | 1.755 | 1.40e-26 | 1.22e-28 |
| (2, ½) | 2.0 | 375 | 2.41e-27 | 2.963 | 3.46 / 2.38 | 2.963 | 2.963 | 7.15e-27 | 1.58e-28 |
| (3, 1) | 1.0 | 374 | 9.14e-26 | 5.10 | 9.57 / 3.78 | 5.10 | 5.10 | 4.66e-25 | 2.44e-27 |
| (3, 1) | 2.0 | 385 | 2.29e-26 | 101.5 | 250 / 57.5 | 101.8 | 101.8 | 2.33e-24 | 6.40e-26 |
| (4, 3/2) | 1.0 | 379 | 4.04e-25 | 88.0 | 295 / 60.8 | 88.0 | 88.0 | 3.55e-23 | 3.09e-25 |
| (4, 3/2) | 2.0 | 391 | 1.01e-25 | 130.5 | 889 / 30.8 | 130.5 | 130.5 | 1.32e-23 | 2.64e-24 |
| (4, 3/2) | 2.4 | 392 | 7.04e-26 | 569 | 4902 / 0.40 | 575 | 575 | 4.01e-23 | 4.07e-23 |
| (5, 2) | 1.0 | 383 | 1.21e-24 | 136.9 | 1075 / 67.6 | 136.9 | 136.9 | 1.66e-22 | 6.27e-24 |
| (5, 2) | 2.0 | 393 | 3.03e-25 | 204.7 | 2683 / 21.6 | 205.5 | 205.5 | 6.22e-23 | 1.38e-22 |
| (5, 2) | 4.5 | 398 | 6.01e-26 | 531 | 5538 / 161 | 532 | 532 | 3.19e-23 | 2.57e-24 |

(⟨σv⟩ in cm³ s⁻¹.) Velocity dependence (Fig. 3, `P084_S_vs_v.csv`): S is flat from v_rel = 10⁻⁶ to ≈ 10⁻² and only decreases above the charged-pair threshold v_thr = (8Δm₁/m)^{1/2} = 0.054/0.066/0.077/0.086 (1 TeV). Reason: for the neutral initial state there is no Coulomb 1/v growth (no charge), the W/Z Yukawa range 1/m_W is short compared with the de Broglie wavelength already at v ~ 0.1, and the charged channels are closed for all astrophysical velocities; so S_MW = S_dwarf = S_CMB to < 1 % off-resonance and to 4 % on the (4,3/2) 2.4 TeV point (S 552–575 across v = 2×10⁻³–10⁻⁶). The "saturation at the splitting scale" is thus at v ≈ v_thr ≈ 0.05–0.09, far above halo velocities. Within the Milky-Way range v = 3×10⁻⁴–2×10⁻³ S varies by < 1 % (< 4 % at 2.4 TeV), so no Maxwellian averaging is needed.

Higgsino check of P025: S = 1.66 (1 TeV) and 1.76 (1.1 TeV), inside P025's 1–3 bracket and in the lower half; the line, 1.19×10⁻²⁸ (Sommerfeld-induced χ⁺χ⁻ component), coincides with P025's one-loop estimate 1.2×10⁻²⁸ (the two are the potential- and hard-region pieces of the same amplitude; adding them coherently would give up to ×4 in rate, so we quote the line as (1.2–4.8)×10⁻²⁸ and use 1.2×10⁻²⁸ as the anchor in the ratios).

### 4.3 Resonance structure versus mass (Fig. 1, `P084_mass_scan.csv`, 48 log-spaced masses 0.5–12 TeV, grid spacing 6.6 %)

Local maxima of S_sat(m) (saturated, v = 10⁻⁶), with S at the grid peak:

| system | resonance masses [TeV] (S_sat at grid peak) | S_MW at 1 / 2 / 3 TeV |
|---|---|---|
| (2, ½) doublet | 6.99 (8.1×10³) | 1.66 / 2.97 / 5.84 |
| (3, 1) triplet | 2.53 (1.1×10⁴), 5.70 (4.2×10³), 10.5 (8.4×10⁴) | 5.1 / 110 / 216 |
| (4, 3/2) quadruplet | 1.29 (9.3×10³), 2.37 (2.0×10³), 4.66 (1.5×10⁵), 7.48 (6.2×10⁴), 9.16 (1.3×10⁴) | 94 / 139 / 15 |
| (5, 2) quintuplet | 0.75 (1.7×10⁵), 1.29 (2.5×10³), 2.07 (9.5×10⁴), 3.10 (5.7×10³), 4.98 (5×10⁶), 6.53, 8.56, 10.5 | 140 / 1.4×10⁴ / 3.9×10³ |
| (3, 0) wino (check) | 2.37 (1.2×10⁴), 9.16 (2.2×10⁴) | 3.4 / 70 / 55 |
| (5, 0) MDM quintuplet (check) | 0.80 (4×10⁵), 1.81 (88), 3.10 (2.7×10⁴), 6.10 (7.7×10³), 8.00 (51) | 70 / 32 / 1.3×10⁴ |

Resonance positions are given to the grid spacing (±3 %), and S at a peak depends on the grid point's distance from the zero-energy bound state (true peak values are larger). Comments: (i) the doublet has no resonance below 7 TeV, so S = 1.7–3 over the whole LZ-relevant range 1–2 TeV and S ≤ 6 up to 3 TeV; (ii) the (3,1) triplet's first resonance at 2.5 TeV sits at the upper edge of its recalled thermal-mass band (2 ± 0.6 TeV): S_MW = 12.7, 30, 104, 1100, 3400 at 1.4, 1.7, 2.0, 2.3, 2.6 TeV; (iii) the (4,3/2) and (5,2) multiplets, with α_Z = 0.10 and 0.18 on the neutral channel and W couplings 3–6 × α₂/2, have resonances every 0.6–1.5 TeV from 0.75 TeV on, so S ≳ 50 at every mass above 0.8 TeV; (iv) the assignment's recollection of a "quintuplet resonance at 9–10 TeV" is not what we find for either quintuplet — the real (5,0) has peaks at 0.8, 3.1, 6.1 and (weakly) 8.0 TeV and S ≈ 60 at its 9.4 TeV thermal mass; we flag this as a discrepancy with an uncertain recollection, whereas the wino's 2.37 TeV is reproduced. No LZ-fitting mass sits exactly on a resonance, but 2.4 TeV (4,3/2) and 2.0 TeV (3,1) are within ~1 grid step of one (S_MW = 570 and 100).

### 4.4 Limit comparison and verdicts (`P084_verdict_table.csv`)

Ratios ⟨σv⟩/limit at the anchor values, [range over the recalled band], for full local density (ρ_local = 0.3 GeV cm⁻³, as the LZ normalisation assumes):

| multiplet, m | Fermi dSph | H.E.S.S. GC | AMS p̄ | Planck | line | verdict (full density) |
|---|---|---|---|---|---|---|
| (2,½) 1.0 TeV | 0.16 [0.08–0.32] | 0.80 [0.4–3.2] | 0.53 [0.16–1.6] | 0.016 | 0.60 [0.3–1.2] | allowed, < ×2 headroom (H.E.S.S., line) |
| (2,½) 1.1 TeV | 0.13 | 0.67 [0.33–2.7] | 0.42 | 0.013 | 0.55 [0.28–1.1] | allowed, marginal |
| (2,½) 2.0 TeV | 0.04 | 0.25 [0.13–1.0] | 0.12 | 0.004 | 0.40 [0.2–0.8] | allowed |
| (3,1) 1.0 TeV | 4.7 [2.3–9.3] | 23 [12–93] | 16 [4.7–47] | 0.47 | 12 [6–24] | excluded beyond every band |
| (3,1) 2.0 TeV | 12 [6–23] | 82 [41–330] | 39 | 1.17 [0.73–1.2] | 160 [80–320] | excluded |
| (4,3/2) 1.0 TeV | 356 | 1.8×10³ | 1.2×10³ | 36 | 1.5×10³ | excluded |
| (4,3/2) 2.0 / 2.4 TeV | 66 / 169 | 470 / 1300 | 220 / 560 | 6.6 / 17 | 6.6×10³ / 8.5×10⁴ | excluded |
| (5,2) 1.0 TeV | 1.7×10³ | 8.3×10³ | 5.5×10³ | 166 | 3.1×10⁴ | excluded |
| (5,2) 2.0 / 4.5 TeV | 312 / 71 | 2.2×10³ / 750 | 1.0×10³ / 240 | 31 / 7.1 | 3.5×10⁵ / 2.9×10³ | excluded |

The Planck bound, which does not depend on halo profiles or J-factors, excludes on its own the (4,3/2) and (5,2) multiplets at every LZ-fitting mass (×7–170) and the (3,1) at its 2 TeV thermal mass (×1.2, marginal). The Y = 1 triplet at 1 TeV is excluded by Fermi dwarfs (×4.7, robust to the ×2 band), H.E.S.S. and the line, but not by Planck (0.47).

Mass ranges excluded at the anchor limits along the scan (full density, MW velocity): doublet — only 0.5–0.86 TeV (H.E.S.S.) and the 6.1–7.5 TeV resonance (H.E.S.S.; Fermi at 7 TeV); triplet (3,1) — Fermi: 0.5–3.8 and 5.0–6.1 TeV, H.E.S.S.: all masses ≤ 6.5 TeV and ≥ 9.2 TeV; (4,3/2), (5,2) — every grid mass by H.E.S.S., 46–48/48 by Fermi, 32–44/48 by Planck. The doublet at 6.1–7.5 TeV is not an LZ-fitting mass anyway (δ_max reasoning, P002/P007).

### 4.5 Sub-component scaling and the interplay with LZ

If the multiplet is a thermal relic with Ω/Ω_DM = f < 1 (f ≈ (M/M_th)² for M < M_th, from Ω ∝ 1/⟨σv⟩_eff ∝ M² [likely]; doublet at 1 TeV f = 0.8 from P007/P025's 0.83/0.73) and the local density scales the same way, all indirect ratios scale as f² while the LZ rate scales as f. LZ then needs N_full(δ) = 1/f events at full density; with N(δ) falling ×5 per 10 keV near δ_max (P007/P037) the required splitting shifts by Δδ = −10 keV × ln(1/f)/ln 5, staying inside P037's windows:

| multiplet, m | f_thermal | worst ratio × f² | δ(N=1) → | f needed to evade the worst anchor (band edge) | δ(N=1) at that f |
|---|---|---|---|---|---|
| (2,½) 1.0 TeV | 0.80 | 0.51 (H.E.S.S.) | 364 keV | — (allowed) | 366 |
| (3,1) 1.0 TeV | 0.25 | 1.5 (H.E.S.S.), 0.29 (Fermi) | 365 keV | ≤ 0.21 (≤ 0.29) | 364 keV |
| (3,1) 2.0 TeV | 1 (thermal mass) | 160 | 385 | ≤ 0.08 (≤ 0.11) | 369 keV |
| (4,3/2) 1.0 TeV | 0.17 | 54 | 368 | ≤ 0.024 (0.034) | 356 keV |
| (4,3/2) 2.4 TeV | 1 | 8.5×10⁴ | 392 | ≤ 0.003 (0.005) | 357 keV |
| (5,2) 1.0 TeV | 0.05 | 76 | 364 | ≤ 0.006 (0.008) | 351 keV |
| (5,2) 4.5 TeV | 1 | 2.9×10³ | 398 | ≤ 0.019 (0.026) | 373 keV |

So: the thermal (3,1) triplet at 1 TeV (f = 0.25) is marginal — ×1.5 over the H.E.S.S. anchor but inside its band and ×0.3 of Fermi — and viable only as a ≤ 20–30 % sub-component with δ ≈ 364 keV; at its thermal mass it is excluded ×160. The (4,3/2) and (5,2) multiplets would need f ≲ 0.3–3 %, i.e. a non-thermal dilution far below their thermal fractions (17 %, 5 %), which pushes δ(N=1) to 351–357 keV, just above LZ's tabulated 350 keV point where P037 found the full-density coupling ×3.5(2Y)² above the Fig. 6 edge (at f ≲ 0.01 that edge is relaxed by ≥ 100, so LZ itself does not forbid it). A non-thermal, ≲ 1 % Y ≥ 3/2 sub-component that produces exactly one event is therefore not excluded by indirect probes, but it is no longer the dark matter; P054's heavy-mediator non-thermal discussion applies. Conversely, the doublet's f = 0.8 lowers its indirect ratios to ≤ 0.51 and moves δ(N=1) by only −1.4 keV.

### 4.6 Planck specifics

p_ann = f_eff ⟨σv⟩_CMB/m with S_CMB = S_MW (Sec. 4.2): doublet p_ann = 0.35 × 1.6×10⁻²⁶/1000 = 5.6×10⁻³⁰ (1.6 % of the bound); (3,1) 1 TeV 1.6×10⁻²⁸ (47 %), 2 TeV 4.1×10⁻²⁸ (117 %); (4,3/2) 1 TeV 1.2×10⁻²⁶ (×36); (5,2) 1 TeV 5.8×10⁻²⁵ (×166). Because there is no 1/v growth, the CMB does not gain over the halo probes for these models (unlike P025's dark-photon case); its strength is being profile-independent.

## 5. Figures

- `figures/P084_fig1_S_vs_mass.png` — Sommerfeld factor S(WW+ZZ) versus mass, 0.5–12 TeV, for the four Y ≠ 0 multiplets at Milky-Way velocity (left) and saturated (right); wino dashed with the recalled 2.3–2.5 TeV resonance band; circles 1 TeV LZ fits, squares recalled thermal masses. The doublet is resonance-free below 7 TeV; the Y ≥ 1 multiplets have S ≥ 5 (triplet) and S ≥ 50 (Y ≥ 3/2) at all LZ-relevant masses.
- `figures/P084_fig2_sigv_vs_limits.png` — MW-enhanced ⟨σv⟩(WW+ZZ) versus mass against the recalled Fermi dSph, H.E.S.S. Einasto and Planck bands (solid: full density; dotted: × f_thermal²). Only the doublet stays below the bands.
- `figures/P084_fig3_S_vs_v.png` — S versus v_rel at 1 TeV: flat below the charged-pair thresholds (dotted), so S_MW = S_dwarf = S_CMB.

## 6. Robustness, caveats and failed approaches

- Numerical: r₀ ±1.5 %, r_max < 0.03 %, tolerance < 10⁻⁵. Threshold formula (heavy-mass limit) vs P037's finite-mass splittings (7 %): irrelevant for closed channels. α(0) vs α(m_Z) in the Coulomb terms: charged channels only, negligible.
- Physics approximations: leading order in m_W/m (transverse WW; no Goldstone/Higgs channels — for pure gauge multiplets without Higgs couplings this is exact at tree level); no electroweak Sudakov resummation (would reduce the exclusive WW/line rates by O(10–30 %) at TeV [likely]); the Sommerfeld-induced line and the hard one-loop line are quoted separately, not coherently; p-wave and finite-width effects ignored; resonance peaks quoted at grid resolution (true peak S larger, peak positions ±3 %). Near a resonance S ∝ 1/v² sets in only within a narrow mass window (not at our benchmarks) and would make S_CMB > S_MW there.
- Recalled limits carry ×2 (Fermi, Planck) to ×10 (H.E.S.S. profile, AMS propagation, line) uncertainties; the exclusion of the Y ≥ 1 multiplets exceeds every band by ≥ ×2.3 (triplet, Fermi) to ≥ ×10³ (quintuplet) and Planck alone excludes Y ≥ 3/2 without profile assumptions, so the verdicts do not hinge on any single recalled number. The doublet's "marginal" status (0.4–3.2 of H.E.S.S., 0.3–1.2 of the line) does hinge on them, as P025 found.
- Thermal masses other than the doublet's are uncertain (±30 %); the triplet's S varies ×270 across its band (Sec. 4.3), so its thermal-mass verdict ranges from ×12 (1.4 TeV) to ×10⁴ (2.6 TeV) over the Fermi anchor — excluded everywhere.
- Restriction to maximal-hypercharge multiplets (Y = T): for (4,½) and (5,1) (P037's alternatives) the neutral state is not extremal, C_− ≠ 0, and the ψ⁰ψ⁰ sector couples to ψ⁺ψ⁻-like states; not treated.
- Failed/abandoned: a first draft used outward integration with a matrix of regular solutions; with several closed channels the closed-channel rows become degenerate (all columns collapse onto the fastest-growing mode), so we switched to inward integration of decaying basis solutions. Two column names with hyphens caused a syntax error on the first run (fixed). The dataviz palette validator (JavaScript) was not run (no node allowed).

## 7. Discussion and conclusion

Once the Sommerfeld factor is computed, the LZ-fitting electroweak multiplets separate cleanly. The Higgsino-like doublet has S = 1.7 (1 TeV) to 3.0 (2 TeV) with no resonance below 7 TeV, so P025's verdict stands with its bracket resolved: ⟨σv⟩ = 1.6×10⁻²⁶ cm³/s at 1 TeV, 0.16 of Fermi, 0.4–3.2 of H.E.S.S. (Einasto), 0.5 of AMS, 0.016 of Planck, 0.3–1.2 of the line — allowed, CTA-testable. Every heavier-hypercharge multiplet that fits LZ at 1 TeV is excluded as the dominant dark matter: the Y = 1 triplet by ×4.7 (Fermi) to ×23 (H.E.S.S.), the Y = 3/2 quadruplet by ×36 (Planck alone) to ×1800, the Y = 2 quintuplet by ×170 (Planck) to ×8000, with S = 5, 88, 137 on top of ZZ-dominated tree rates that grow as Y⁴. Thermal under-abundance (f² = 0.06, 0.03, 0.003) rescues none of them except, marginally, the triplet at 1 TeV as a ≤ 20–30 % component with δ ≈ 364 keV. LZ therefore does not single out the Higgsino kinematically (P037), but indirect detection does: among Y ≠ 0 fermion multiplets, only the doublet (and possibly a diluted Y = 1 triplet) can be the LZ scatterer.

## 8. References

1. LZ Collaboration, arXiv:2609.02823 (2026).
2. J. Hisano, S. Matsumoto, M. M. Nojiri, O. Saito, Phys. Rev. D 71, 063528 (2005) — non-perturbative (Sommerfeld) effect for electroweak WIMPs; wino matrices.
3. M. Cirelli, N. Fornengo, A. Strumia, Nucl. Phys. B 753, 178 (2006); M. Cirelli, A. Strumia, M. Tamburini, Nucl. Phys. B 787, 152 (2007) — minimal dark matter: splittings, annihilation, Sommerfeld.
4. T. R. Slatyer, JCAP 02 (2010) 028; S. Cassel, J. Phys. G 37, 105009 (2010) — Hulthén Sommerfeld factor, inelastic case.
5. N. Arkani-Hamed, A. Delgado, G. F. Giudice, Nucl. Phys. B 741, 108 (2006) — Higgsino relic density.
6. T. Cohen, M. Lisanti, A. Pierce, T. R. Slatyer, JCAP 10 (2013) 061; J. Fan, M. Reece, JHEP 10 (2013) 124 — wino line constraints.
7. M. Ackermann et al. (Fermi-LAT), Phys. Rev. Lett. 115, 231301 (2015) — dSph limits.
8. H. Abdallah et al. (H.E.S.S.), Phys. Rev. Lett. 117, 111301 (2016); Phys. Rev. Lett. 120, 201101 (2018); Phys. Rev. Lett. 129, 111101 (2022) — GC continuum and line.
9. Planck Collaboration, A&A 641, A6 (2020); T. R. Slatyer, Phys. Rev. D 93, 023527 (2016) — p_ann, f_eff.
10. A. Cuoco, M. Krämer, M. Korsmeier, Phys. Rev. Lett. 118, 191102 (2017) — AMS-02 antiprotons.
11. Corpus: P007, P025 (competing), P037, P054; PAPER_GUIDE; results ledger.

## 9. Tools and provenance (mirrors `output/provenance/P084.json`)

- Agent tools: Read ×11 (PAPER_GUIDE.md; papers P025, P037, P007, P054; work/P025/details.md; three figures, Fig. 2 twice), Bash ×11 (directory/ledger/version listings, grep of P037 details and lzcommon, timing test of the solver, 4 script runs, CSV extraction of exclusion ranges, word-budget counts ×4 and JSON check), Write ×6 (script, details.md, provenance JSON, paper ×3), Edit ×12 (script ×3: column names, evasion fractions and mass-band print-out, Fig. 2 layout; paper ×7 word-budget trims; JSON and details tool counts), Skill ×1 (dataviz; palette read via grep; JS validator skipped).
- Software: python 3.12.13 (.venv); numpy 2.5.3 (linalg.solve, array algebra); scipy 1.18.1 (integrate.solve_ivp DOP853, complex); pandas 3.0.5 (tables); matplotlib 3.11.2 (Agg, three PNGs). common/lzcommon.py: not needed (no rate calculation; constants re-declared in the script). Algebra by hand: SU(2) ladder factors, Z couplings, threshold amplitude vectors, one-loop splitting formula, Majorana ↔ Dirac basis rotation, incoming-flux normalisation.
- Scripts: `output/code/P084_sommerfeld_indirect.py` — `.venv/bin/python output/code/P084_sommerfeld_indirect.py` (26 s).
- Outputs: `P084_tree_xsec.csv`, `P084_sommerfeld_benchmarks.csv`, `P084_S_vs_v.csv`, `P084_mass_scan.csv`, `P084_verdict_table.csv`, `P084_results.json`, `run_log.txt`, `figures/P084_fig{1,2,3}_*.png`.
- Local inputs: PAPER_GUIDE.md; papers/P025.md, P037.md, P007.md, P054.md; work/P025/details.md; work/P037/details.md (thermal masses, splittings, δ table via grep); results_ledger.csv (header and rows P007/P025/P037/P054); environment/ENVIRONMENT_versions.txt; dataviz palette.md.
- Recalled knowledge: 23 items with reliability flags (JSON). Datasets: none. Data requests: none. WimPyDD-generated files: none.
