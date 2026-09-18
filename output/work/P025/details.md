# P025 — Indirect-detection constraints on TeV inelastic dark matter after LZ: γ-rays, antiprotons and the CMB

*Research record. Simulated date 2026-09-08. Author profile: indirect-detection phenomenologists. Category COMP (astro-ph.HE, cross-list hep-ph). Result type: estimated. Script `output/code/P025_indirect.py`; all numbers below are printed in `output/work/P025/run_log.txt` and stored in `P025_results.json`, `higgsino_sigv_vs_mass.csv`, `darkphoton_sommerfeld_grid.csv`, `summary_table.csv`.*

## 1. Motivation and framework

The corpus has produced three dark-matter (DM) readings of the LZ 248 keV event (arXiv:2609.02823) that fix, or nearly fix, the annihilation sector:

- **(a) Pure Higgsino doublet** (P007, P014): Z-mediated inelastic scattering with a gauge-fixed cross-section; one event requires δ ≈ 358–380 keV at 1 TeV (best 366 keV) and PeV gauginos, so the neutralino is maximally pure. The thermal mass 1.1 TeV lies in the window. Annihilation today is fixed by gauge couplings.
- **(b) Pseudo-Dirac fermion with a kinetically mixed dark photon** (P011): σ_p is free, but the relic density fixes α_D (P011: 0.0073/0.0245/0.073 for 300/1000/3000 GeV) and the χ₂-decay requirement gives m_A′ ≳ 0.5–5 GeV, with a ceiling m_A′ ≲ 8.5–57 GeV from ε ≲ 10⁻³. The mediator is light: annihilation is χχ → A′A′ (secluded), and the light mediator generates a Sommerfeld enhancement.
- **(c) Contact magnetic-dipole (L10) WIMP** (P012): d10^s = 0.28 at 1 TeV, i.e. a tensor–tensor contact strength d10/m_v² = (465 GeV)⁻² or, in the dimension-8 reading with the q²/m_N² factor, (20.9 GeV)⁻⁴.

We ask what each predicts for ⟨σv⟩ today and at recombination, into which final states, and whether recalled limits from Fermi-LAT dwarf spheroidals (dSphs), H.E.S.S./MAGIC Galactic-centre (GC) searches, AMS-02 antiprotons and Planck exclude them. **All experimental limits are recalled** (no data access); each carries a reliability flag (Sec. 6) and we never quote a recalled number to better than a factor ~2. Sommerfeld enhancement of the Higgsino is only bracketed (S = 1–3); P084 will treat it in detail.

Unit conversion: 1 GeV⁻² = (ħc)²c = 1.1673×10⁻¹⁷ cm³ s⁻¹ [certain]. Electroweak inputs: sin²θ_W = 0.2312, α₂ = 1/29.6 (g = 0.6516), α = 1/137.04, m_W = 80.377, m_Z = 91.1876 GeV [certain]. Canonical thermal ⟨σv⟩ = 2.2×10⁻²⁶ cm³ s⁻¹ (Majorana; Steigman–Dasgupta–Beacom 2012) [certain]; Ω_DM h² = 0.120 [certain].

## 2. Higgsino

### 2.1 Freeze-out (effective, coannihilating) cross-section

Arkani-Hamed, Delgado & Giudice (2006) [recalled/likely]:

  ⟨σv⟩_eff = g⁴ (21 + 3 tan²θ_W + 11 tan⁴θ_W) / (512 π μ²) = 2.57×10⁻³ GeV²/μ² (tan²θ_W = 0.3007).

Cross-check with our recollection of the Cirelli–Fornengo–Strumia (2006) fermion n-plet formula (n = 2, Y = 1/2): g⁴[81 + 12 tan²θ_W + 10.75 tan⁴θ_W]/(4·512π μ²), 6.5 % lower. Results (`higgsino_sigv_vs_mass.csv`):

| μ [GeV] | ⟨σv⟩_eff (ADG) | (CFS) | Ωh² = 0.12 × 2.2e-26/⟨σv⟩_eff |
|---|---|---|---|
| 300 | 3.33e-25 | 3.11e-25 | 0.008 |
| 500 | 1.20e-25 | 1.12e-25 | 0.022 |
| 1000 | 2.99e-26 | 2.80e-26 | 0.088 |
| 1100 | 2.48e-26 | 2.31e-26 | 0.107 |
| 2000 | 7.49e-27 | 7.00e-27 | 0.353 |
| 4000 | 1.87e-27 | 1.75e-27 | 1.41 |

Ωh² = 0.12 is reached at **μ = 1167 GeV** (no Sommerfeld), against the literature 1.1 TeV [certain]; P007 quoted Ωh² = 0.099 at 1 TeV from Ωh² ≈ 0.10 (μ/TeV)², we get 0.088 — the recalled formula is validated at the 10–20 % level, which is the sanity check requested.

### 2.2 Annihilation today: derivation

Today only χ₁ survives (χ₂ lifetime ~10⁶ s, P007; chargino cτ = 0.7 cm, P014). Pure-Higgsino χ₁ has no diagonal Z or Higgs coupling (the Z current is off-diagonal, χ̄₁γ^μχ₂; Higgs couplings vanish as m_W/M_gaugino ~ 10⁻⁵ for PeV gauginos), so χ₁χ₁ → ff̄, Zh are absent at tree level and the only s-wave channels are **W⁺W⁻ (t/u-channel chargino) and ZZ (t/u-channel χ₂)**.

Threshold formula. A Majorana pair annihilating through t- and u-channel exchange of a degenerate fermion partner into two vector bosons, each vertex of strength g_eff, has the same amplitude structure as e⁺e⁻ → γγ (two diagrams; there because photons are identical, here because χ is Majorana). Para-positronium gives the spin-averaged σv(e⁺e⁻ → γγ) = πα²/m² = e⁴/(16πm²) [certain], which includes the identical-photon phase-space factor ½ and the ¼ spin-singlet fraction. Hence

  ⟨σv⟩(χχ → V V′) = g_eff⁴/(8π m²) × (½ if V = V′).

Calibration on the wino (g_eff = g): 2πα₂²/m² — the standard heavy-wino result [recalled/likely]; at 3 TeV it gives 9.3×10⁻²⁷ cm³/s, consistent with the tree-level wino literature (~10⁻²⁶) [likely].

Higgsino vertices. Writing the Dirac neutral Higgsino ψ⁰ = (χ₁ + iχ₂)/√2 with the doublet W coupling (g/√2)ψ̄⁰γ^μψ⁻W⁺ gives χ₁–χ^±–W of strength **g/2** (vector-like; the apparent axial form for one of the two states is a Majorana phase convention — in the exact Dirac limit the U(1) symmetry rotating χ₁ ↔ χ₂ forces equal χ₁χ₁ and χ₂χ₂ rates). The Z coupling (g/2c_W)ψ̄⁰γ^μψ⁰ = (g/2c_W)·iχ̄₁γ^μχ₂ gives χ₁–χ₂–Z of strength **g/(2c_W)** (as found by P011 for the dark photon). Therefore

  ⟨σv⟩_WW = (g/2)⁴/(8πm²) = πα₂²/(8m²),  ⟨σv⟩_ZZ = ½ (g/2c_W)⁴/(8πm²) = πα₂²/(16 c_W⁴ m²),  ZZ/WW = 1/(2c_W⁴) = 0.846,

times (1 − m_V²/m²)^{3/2} (≤ 1 % at 1 TeV). Numbers (`higgsino_sigv_vs_mass.csv`):

| m [GeV] | WW | ZZ | total (S=1) | ×3 |
|---|---|---|---|---|
| 300 | 5.20e-26 | 4.25e-26 | 9.45e-26 | 2.84e-25 |
| 500 | 2.01e-26 | 1.68e-26 | 3.70e-26 | 1.11e-25 |
| 1000 | 5.18e-27 | 4.37e-27 | 9.55e-27 | 2.87e-26 |
| 1100 | 4.29e-27 | 3.62e-27 | 7.91e-27 | 2.37e-26 |
| 2000 | 1.30e-27 | 1.10e-27 | 2.41e-27 | 7.22e-27 |
| 4000 | 3.27e-28 | 2.76e-28 | 6.03e-28 | 1.81e-27 |

Today/freeze-out-effective ratio = 0.32 at every mass: three-quarters of the freeze-out annihilation was chargino and χ₂ coannihilation. The Sommerfeld factor of a doublet below ~3 TeV is modest (no resonance until several TeV) [recalled/likely]; we carry S = 1–3 as a band and defer the calculation to P084.

### 2.3 γ-ray line

One-loop heavy-wino line: ⟨σv⟩_γγ → 4πα²α₂²/m_W² (Bergström–Ullio 1997; mass-independent) [recalled/likely] = 1.4×10⁻²⁷ cm³/s. For the Higgsino each of the four gauge vertices carries g/2 instead of g, a factor 1/16 (our estimate; the W-loop topology differs and the neglected terms are O(1)): **⟨σv⟩_γγ ≈ 8.6×10⁻²⁹ cm³/s** [uncertain, ×3]. γZ from the chargino loop: Z coupling g(½ − s_W²)/c_W versus e = g s_W and two non-identical bosons → γZ/γγ = 0.81, so ⟨σv⟩_γγ + ½⟨σv⟩_γZ = 1.2×10⁻²⁸ cm³/s (S = 1), 3.6×10⁻²⁸ (S = 3), with α(0); α(m_Z) raises this by 15 %.

### 2.4 Comparison with recalled limits (Table in `higgsino_sigv_vs_mass.csv`)

Anchors at 1 TeV and our mass scaling (only the anchors are recalled; the scaling ∝ m^1 for continuum γ-ray limits is our approximation, ∝ m^0.5 for H.E.S.S. whose sensitivity peaks at TeV):

| Probe | ⟨σv⟩ limit at 1 TeV | scaling | reliability |
|---|---|---|---|
| Fermi-LAT 15 dSphs 6 yr (Ackermann+2015; Albert+2017 with DES dSphs), WW ≈ bb̄ | 1×10⁻²⁵ | ∝ m | likely (×2) |
| H.E.S.S. GC 254 h (2016) / 546 h (2022), Einasto, WW | 4×10⁻²⁶ | ∝ m^0.5 | likely for Einasto; ×10 weaker for cored profiles |
| AMS-02 p̄ (Cuoco+2017, Cui+2017), WW ≈ bb̄ | 1×10⁻²⁵ | ∝ m | uncertain (propagation ×3–10) |
| Planck p_ann = f_eff⟨σv⟩/m < 3.5×10⁻²⁸ cm³ s⁻¹ GeV⁻¹ (3.2×10⁻²⁸ in Planck 2018) | 1.7×10⁻²⁴ (f_eff,WW = 0.2) | ∝ m | likely |
| H.E.S.S. line search (Abdallah+2018), γγ, Einasto | 4×10⁻²⁸ | ∝ m | likely (order) |
| MAGIC Segue 1 (158 h) | ~10⁻²⁴ at 1 TeV | – | likely; not competitive |

Ratios ⟨σv⟩/limit at 1000 (1100) GeV: Fermi 0.10–0.29 (0.07–0.22) for S = 1–3; H.E.S.S. Einasto 0.24–0.72 (0.19–0.57); AMS 0.10 (0.07); Planck 0.016 at S = 3 — irrelevant for an s-wave TeV WIMP because the bound scales ∝ m; line 0.30–0.91 (0.28–0.83). **At 300 GeV** the pure Higgsino would exceed the Fermi dSph limit ×3.1 (S = 1) if it made up all of the local DM, which the LZ rate normalisation assumes (ρ₀ = 0.3 GeV/cm³); at 500 GeV the ratio is 0.74. So indirect detection independently disfavours P007's low-mass fit points (300 GeV; marginal at 500 GeV), which P007 already disfavoured on relic grounds (thermal fraction 7 %/21 %).

**Verdict:** the 1–1.1 TeV Higgsino that fits LZ is allowed by every recalled bound; the tightest are the H.E.S.S. Einasto continuum (×1.4–4 headroom, vanishing for S ≈ 3 with a cuspy profile) and the line (×1.1–3.3). It is a CTA target.

## 3. Dark-photon pseudo-Dirac model

### 3.1 Relic density and today's cross-section

At freeze-out (T_f ≈ m/25 ≫ δ) χ₁ and χ₂ are equally populated. Tree-level channels into A′A′: χ₁χ₁ via virtual χ₂ and χ₂χ₂ via virtual χ₁, each with the Majorana formula at g_eff = g_D (off-diagonal current iχ̄₁γ^μχ₂, full strength) and identical bosons: ⟨σv⟩₁₁ = ⟨σv⟩₂₂ = g_D⁴/(16πm²) = **πα_D²/m²** (equal to the Dirac spin-averaged e⁺e⁻ → γγ analogue, as it must be). χ₁χ₂ → A′A′ vanishes at tree level (the intermediate χ₂χ₂A′ or χ₁χ₁A′ vertex does not exist) and χ₁χ₂ → A′* → ff̄ is ε²-suppressed (P011). The coannihilation-weighted cross-section with equal weights is σ_eff = (σ₁₁ + σ₂₂ + 2σ₁₂)/4 = **πα_D²/(2m²)**. Setting σ_eff = 2.2×10⁻²⁶ cm³/s gives α_D = 0.0104, 0.0346, 0.104 at 300, 1000, 3000 GeV — √2 above P011, which set πα_D²/m² = 2.2×10⁻²⁶ (α_D = 0.0073/0.0245/0.073). This is the familiar factor 2 between Dirac and Majorana thermal cross-sections. We carry both ("eff" and "P011") conventions.

Today all DM is χ₁, so **⟨σv⟩₀ = πα_D²/m² = 2σ_eff = 4.4×10⁻²⁶ cm³/s** (eff) or 2.2×10⁻²⁶ (P011), mass-independent by construction. Final states: A′ → e⁺e⁻ only for m_A′ < 211 MeV; e⁺e⁻/μ⁺μ⁻/hadrons in ratio ≈ 1 : 1 : R(s) ≈ 2–3 for m_A′ = 1–3 GeV; plus τ⁺τ⁻ above 3.55 GeV and heavy quarks above. The photon spectrum from these four-body cascades is soft (energies ≲ m/2), so continuum γ-ray limits are weaker than for direct two-body channels.

Sommerfeld at freeze-out (exact Coulomb, v = 0.3, an upper bound): S_fo = 1.06/1.19/1.64 at 300/1000/3000 GeV — a ≤ 20 % (1 TeV) correction to α_D, ignored below (it goes the safe way: less α_D needed).

### 3.2 Sommerfeld enhancement with a Yukawa potential

For A′ exchange with δ negligible (see 3.4) the potential is V = −α_D e^{−m_A′ r}/r. We use the Hulthén-potential closed form (Cassel 2010; Slatyer 2010; Feng, Kaplinghat & Yu 2010) [recalled/likely], with ε_v = v/α_D (v = single-particle CM velocity = v_rel/2), ε_φ = m_A′/(α_D m_χ), ε* = π²ε_φ/6:

  S = (π/ε_v) sinh(2πε_v/ε*) / [cosh(2πε_v/ε*) − cos(2π√(1/ε* − ε_v²/ε*²))].

Checks (run log): ε_φ → 0 reproduces the Coulomb π/ε_v (31.42 at ε_v = 0.1); the large-ε_v branch reproduces the exact Coulomb S = (π/ε_v)/(1 − e^{−π/ε_v}) (1.346 vs 1.347 at ε_v = 5); S_sat(ε_φ = 10) = 1.23 → 1. Saturation (v → 0): S_sat = 2π²/[ε*(1 − cos(2π/√ε*))], with resonances at ε_φ = 6/(π²n²) (bound states at threshold) and valley minima S_sat,min ≈ π²/ε* = 6/ε_φ = 6α_D m_χ/m_A′. The Coulomb approximation asked for in the assignment, S ≈ πα_D/v, gives S = 1.1×10⁷ at v = 10⁻⁸ (1 TeV) — the saturation at m_A′ is therefore essential; the recombination-era velocity (10⁻⁸–10⁻¹⁰ for TeV DM decoupled at MeV–GeV temperatures) is deep in the saturated regime for any m_A′ ≳ 10⁻⁵ GeV.

Resonance positions (eff convention): m_A′ = 21.1/5.3/2.3/1.3/0.84 GeV (1 TeV, n = 1–5); 1.90/0.47/0.21 GeV (300 GeV); 190/47/21/12/7.6 GeV (3 TeV).

S_sat at m_A′ = 0.1 / 1 / 10 GeV: 233 / 22 / 1.9 (300 GeV); 2081 / 225 / 21 (1 TeV); 18 900 / 4170 / 233 (3 TeV). Galactic (v = 5×10⁻⁴, GC/halo): 218 / 166 / 21 at 1 TeV — i.e. the halo is already near saturation for m_A′ ≲ 10 GeV. Dwarfs (v = 3×10⁻⁵) are saturated.

### 3.3 CMB constraint

Planck: f_eff⟨σv⟩_rec/m < p_ann = 3.5×10⁻²⁸ cm³ s⁻¹ GeV⁻¹ [likely; 3.2×10⁻²⁸ in Planck 2018]. f_eff (Slatyer 2016) [likely]: 0.45 for e⁺e⁻, 0.20 for μ⁺μ⁻ and hadrons at TeV; we use 0.35 for the mixed e/μ/hadron cascade above 211 MeV and 0.45 below. Allowed enhancement S_allowed = p_ann m/(f_eff ⟨σv⟩₀):

| m_χ | α_D (eff) | S_allowed (CMB) | m_A′ with any CMB-allowed point | m_A′ above which all heavier values pass | sub-GeV fraction allowed |
|---|---|---|---|---|---|
| 300 GeV | 0.0104 | 6.8 | 4.0 GeV | 4.0 GeV | 0 |
| 1 TeV | 0.0346 | 22.7 | 9.2 GeV (valleys 9–32 GeV) | 31.6 GeV | 0 |
| 3 TeV | 0.104 | 68 | 29 GeV | 61 GeV | 0 |
| 1 TeV (P011 α_D = 0.0245) | 0.0245 | 45 | 5.2 GeV | 19.9 GeV | 0 |
| 3 TeV (P011) | 0.0735 | 136 | 10.5 GeV | 40 GeV | 0 |

At m_A′ = 1 GeV and 1 TeV the CMB is exceeded ×10 (eff) or ×18 (P011); at 0.1 GeV by ×90. **No sub-GeV dark photon survives, resonance valleys included** (S_sat,min = 6α_D m/m_A′ = 208 at 1 GeV, 1 TeV). This is independent of ε and of δ (Sec. 3.4).

Confrontation with P011's windows (from the χ₂ lifetime floor and the ε ≤ 10⁻³ BaBar ceiling): 5.0–57 GeV (δ = 250 keV), 2.6–34 (300), 0.94–14 (350), 0.52–8.5 (365), 0.09–3.8 GeV (380 keV) at 1 TeV. Intersected with the CMB (eff): δ = 250 keV → 9–57 GeV survives (valleys below 32 GeV); δ = 300 → 9–34 GeV; δ = 350 → only the valleys at 9–14 GeV; **δ = 365 and 380 keV → nothing survives**. With P011's α_D the corresponding floors are 5.2/19.9 GeV, and δ = 365 keV keeps only 5.2–8.5 GeV in valleys. So the Higgsino-like corner of the dark-photon model (δ ≳ 360 keV) is excluded by the CMB for a thermal α_D, unless ε > 10⁻³ is allowed (it is not, for m_A′ = 1–10 GeV, by BaBar/LHCb [likely]) or the relic density is non-thermal (P075 will discuss).

### 3.4 Does the inelastic splitting spoil the enhancement?

The A′ exchange converts χ₁χ₁ ↔ χ₂χ₂ (off-diagonal potential), the excited pair lying 2δ = 0.73 MeV higher; at v_rel ≲ 1.7×10⁻³ (513 km/s) the excited channel is kinematically closed. Slatyer (2010) showed that the enhancement is essentially the elastic one as long as δ is small compared with the potential-energy scales [recalled/likely]: α_D m_A′ (the potential at the Yukawa range) = 35 MeV × (m_A′/GeV) and α_D² m_χ/4 = 0.30 GeV at 1 TeV, i.e. 2δ/(α_D m_A′) = 0.021 (m_A′ = 1 GeV) and 2δ/(α_D² m/4) = 2.4×10⁻³. Both ratios are ≪ 1 for all m_A′ ≳ 0.05 GeV, so the elastic Hulthén treatment applies; for m_A′ ≲ 20 MeV the splitting would start to suppress S (not relevant here).

### 3.5 Dwarfs, Galactic centre and positrons

Fermi dSph cascade limits (Elor, Rodd, Slatyer & Xue 2016-type recasts) at 1 TeV [recalled/uncertain]: ~10⁻²³ for 4e (final-state radiation only), ~10⁻²⁴ for 4μ, ~3×10⁻²⁵ for mixed hadronic/leptonic cascades. With saturated S in dwarfs, S_allowed = 6.8 (1 TeV, eff) and 2.0 (300 GeV), so the dwarfs would require m_A′ ≳ 45 GeV (1 TeV) or ≳ 9.5 GeV (300 GeV) — stronger than the CMB but resting on an uncertain limit; not reached below 100 GeV at 3 TeV. H.E.S.S. GC: with S_GC = 21–170 for m_A′ = 10–1 GeV, the GC ⟨σv⟩ is (1–7)×10⁻²⁴ into soft cascades, ×10–100 above the Einasto τ⁺τ⁻ limit (10⁻²⁶ ×~10 for the softer spectrum) but profile-dependent; the ordering GC > dSph > CMB in constraining power holds only for cuspy profiles. AMS-02 positrons: a local ⟨σv⟩ of (1–7)×10⁻²⁴ into e⁺ and μ⁺ (m_A′ ≲ 10 GeV) would overshoot the measured positron flux at 100–800 GeV by orders of magnitude (published leptophilic limits at 1 TeV are ~10⁻²⁴, uncertain [John & Linden 2021]); for m_A′ ≳ 30 GeV (S_local ≲ 3) the model is at ~10⁻²⁵, probably allowed. We do not compute positron propagation.

## 4. Contact-dipole (L10) WIMP

Inputs (P012): d10 = 0.28 (LZ normalisation, factor-2 convention ambiguity, DR-001); G_rel = d10/m_v² = 4.62×10⁻⁶ GeV⁻² = (465 GeV)⁻²; dimension-8 reading G₈ = G_rel/m_N² = 5.25×10⁻⁶ GeV⁻⁴ = (20.9 GeV)⁻⁴. The isoscalar nucleon coupling arises from quark tensor operators with tensor charges δu = 0.8, δd = −0.2 [likely]; c_p = c_n requires equal quark couplings C_q = G_rel/(δu + δd) = 7.7×10⁻⁶ GeV⁻².

Annihilation χχ̄ → qq̄ through the same dimension-6 tensor operator, s-wave: σv ≈ κ N_c n_f C_q² m_χ²/π with N_c = 3, n_f = 5 open flavours (t adds 20 %) and an operator-dependent κ = 1–6 [uncertain]: **⟨σv⟩ = 3.3×10⁻²¹–2×10⁻²⁰ cm³/s at 1 TeV**. The literal dimension-8 reading, m²/(πΛ⁴) with Λ = 20.9 GeV, gives 2×10⁻¹⁷ cm³/s, which violates the s-wave unitarity bound σv ≤ 4π/(m² v) = 4.9×10⁻²² (v = 0.3) and 1.5×10⁻¹⁹ cm³/s (v = 10⁻³); even the dimension-6 estimate sits at the unitarity bound at freeze-out. Consequences:

1. **Symmetric thermal relic:** Ω/Ω_DM ≈ 2.2×10⁻²⁶/⟨σv⟩ ≈ 7×10⁻⁶. Rescaling d10 by ×388 to compensate for the lower density in LZ raises ⟨σv⟩ ×1.5×10⁵ — a runaway; the L10 fit cannot be a symmetric thermal relic that is all of the DM.
2. **Indirect bounds:** ⟨σv⟩(qq̄) exceeds the Fermi/H.E.S.S. qq̄ limit (~10⁻²⁵ at 1 TeV [likely]) by 3×10⁴ (dim-6) to 2×10⁸ (dim-8) if χ = χ̄ populates the halo at full density.
3. **EFT validity:** for annihilation at √s = 2 TeV the contact operator is valid only if the mediator mass M > 2 TeV, and then the coupling product g_χ g_N = G_rel M² = 18.5 (M = 2 TeV) or 4.6 (M = 1 TeV): non-perturbative. A light mediator (M < m_χ), the natural reading of Λ ≈ 21 GeV, makes the annihilation dominated by χχ̄ → VV with the large dipole coupling μ_χ^V ~ 200 μ_N (P012), even larger, and the EFT estimate above becomes meaningless; only an explicit mediator model can be confronted with indirect data.
4. **The escape:** χ̄σ^{μν}χ vanishes for a Majorana fermion, so L10 requires Dirac DM. An **asymmetric** Dirac relic has no χχ̄ pairs today, produces no indirect signal, and its enormous ⟨σv⟩ is exactly what removes the symmetric component. Indirect detection therefore either excludes the L10 WIMP by ≥ 4 orders of magnitude (symmetric) or is blind to it (asymmetric); it cannot test the LZ L10 fit itself.

## 5. Summary table (`summary_table.csv`)

| Model | ⟨σv⟩ today [cm³/s] | Channel | Strongest bound | ratio | Allowed? |
|---|---|---|---|---|---|
| Higgsino 1 TeV, δ = 366 keV | 9.6e-27 (S=1) – 2.9e-26 (S=3) | WW 54 %, ZZ 46 % | H.E.S.S. GC Einasto ~4e-26 (likely, ×10 profile) | 0.24–0.72 | yes; marginal for S ≈ 3 + cusp; CTA-testable |
| Higgsino line | 1.2e-28 (S=1) | γγ + γZ/2 | H.E.S.S. line ~4e-28 | 0.30–0.91 | yes (×1.1–3.3) |
| Dark photon 1 TeV, α_D = 0.035, m_A′ < 9 GeV | 4.4e-26 × S_sat (225 at 1 GeV) | A′A′ → 4ℓ, 2ℓ+hadrons | Planck (S_allowed = 23) | 10 at 1 GeV; 90 at 0.1 GeV | **no**, at every sub-GeV and few-GeV mass |
| Dark photon 1 TeV, m_A′ = 9–32 GeV (valleys) or ≥ 32 GeV | 4.4e-26 × S < 23 | soft cascades | Planck; dSph cascades (uncertain) need ≥ 45 GeV | < 1 | yes (CMB); dwarfs marginal below ~50 GeV |
| L10 contact dipole, d10 = 0.28, symmetric Dirac | 3e-21–2e-20 (dim-6); 2e-17 (dim-8, unitarity-violating) | qq̄ | Fermi/H.E.S.S. qq̄ ~1e-25; unitarity | 3e4–2e8 | no as symmetric thermal WIMP; viable only as asymmetric DM (then untestable) |

## 6. Figures

- `figures/P025_fig1_higgsino_sigv_vs_mass.png` — Pure-Higgsino ⟨σv⟩(WW+ZZ) today (blue band: tree level × S = 1–3) and the freeze-out effective cross-section (dotted) versus mass, against recalled limit bands: Fermi-LAT dSphs (orange), H.E.S.S. GC Einasto (green), AMS-02 antiprotons (yellow; same anchor as Fermi, wider band). Vertical line: thermal 1.1 TeV. The thermal Higgsino sits a factor 1.4–10 below every band; only masses ≲ 500 GeV are excluded.
- `figures/P025_fig2_darkphoton_sommerfeld_cmb.png` — Saturated Hulthén Sommerfeld factor at recombination versus m_A′ for m_χ = 300 GeV, 1 TeV, 3 TeV with relic-density α_D (eff convention); dashed lines: Planck-allowed S for the mixed cascade (f_eff = 0.35). Grey: A′ → e⁺e⁻ only. The 1 TeV curve first touches the allowed level at 9 GeV (valley) and stays below it above 32 GeV.

## 7. Robustness and failed approaches

- Higgsino formula choice (ADG vs CFS): 6.5 %. α(0) vs α(m_Z) in the line: 15 %. Sommerfeld band S = 1–3 is the dominant uncertainty and is deferred to P084.
- Recalled limit anchors are good to ×2 (Fermi, H.E.S.S. Einasto, Planck) or ×3–10 (AMS p̄, dSph cascades, GC profile). No Higgsino conclusion changes within these ranges except the H.E.S.S. Einasto margin at S = 3, which is already marginal.
- Dark-photon: the CMB exclusion of sub-GeV A′ is robust to α_D convention (×√2), f_eff (0.2–0.45), p_ann (3.2–3.5×10⁻²⁸) and to the Hulthén approximation (valley minima are analytic, 6/ε_φ); the exact m_A′ floor (9–32 GeV at 1 TeV) shifts by ×1.5 between conventions and depends on the resonance structure, which the Hulthén potential reproduces only approximately (resonance positions ±10 %).
- First script version: NaN at freeze-out velocities from sinh/cosh overflow in the Hulthén formula; fixed with a large-argument branch, validated against the exact Coulomb factor. The dSph threshold for 3 TeV is not reached within the m_A′ ≤ 100 GeV grid (reported as such).
- Not attempted: positron propagation (no propagation code offline), a proper J-factor treatment, the full two-state (inelastic) Sommerfeld calculation (argued negligible in 3.4), and the electroweak Sudakov/Sommerfeld resummation for the Higgsino line.

## 8. References

1. LZ Collaboration, arXiv:2609.02823 (2026).
2. N. Arkani-Hamed, A. Delgado, G. F. Giudice, Nucl. Phys. B 741, 108 (2006) — Higgsino effective annihilation cross-section.
3. M. Cirelli, N. Fornengo, A. Strumia, Nucl. Phys. B 753, 178 (2006) — minimal dark matter.
4. L. Bergström, P. Ullio, Nucl. Phys. B 504, 27 (1997) — heavy-neutralino γγ line.
5. J. Hisano, S. Matsumoto, M. M. Nojiri, O. Saito, Phys. Rev. D 71, 063528 (2005) — Sommerfeld effect for electroweak WIMPs.
6. M. Ackermann et al. (Fermi-LAT), Phys. Rev. Lett. 115, 231301 (2015); A. Albert et al. (Fermi-LAT, DES), ApJ 834, 110 (2017) — dSph limits.
7. H. Abdallah et al. (H.E.S.S.), Phys. Rev. Lett. 117, 111301 (2016); Phys. Rev. Lett. 120, 201101 (2018) — GC continuum and line searches; H.E.S.S. 2022 update, Phys. Rev. Lett. 129, 111101.
8. A. Cuoco, M. Krämer, M. Korsmeier, Phys. Rev. Lett. 118, 191102 (2017); M.-Y. Cui et al., Phys. Rev. Lett. 118, 191101 (2017) — AMS-02 antiprotons.
9. Planck Collaboration, A&A 641, A6 (2020) — p_ann.
10. T. R. Slatyer, Phys. Rev. D 93, 023527 (2016) — f_eff.
11. S. Cassel, J. Phys. G 37, 105009 (2010); T. R. Slatyer, JCAP 02 (2010) 028; J. L. Feng, M. Kaplinghat, H.-B. Yu, Phys. Rev. D 82, 083525 (2010) — Hulthén Sommerfeld factor and inelastic case.
12. G. Elor, N. L. Rodd, T. R. Slatyer, W. Xue, JCAP 06 (2016) 024 — cascade-annihilation limits.
13. G. Steigman, B. Dasgupta, J. F. Beacom, Phys. Rev. D 86, 023506 (2012).
14. Corpus: P007, P011, P012, P014; dossier 00; P075, P084 (forthcoming).

## 9. Tools and provenance (mirrors `provenance/P025.json`)

- Agent tools: Read (PAPER_GUIDE, dossier, ledger, P007/P011/P012/P014 papers, P014.json, palette.md, two figures ×3), Bash (directory listings and version file, grep of corpus/lzcommon, JSON dumps of P007/P011/P012 work files, 6 script runs, word counts), Write (script, details, provenance, paper), Edit (script fixes), Skill (dataviz).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (optimize.brentq); pandas 3.0.5; matplotlib 3.11.2 (Agg); common/lzcommon.py (M_V_GEV, M_NUCLEON_GEV); algebra by hand (Majorana threshold formula, Higgsino vertices, coannihilation counting).
- Recalled knowledge: 24 items listed in the JSON with reliability flags.
- Datasets: none. Data requests: none. WimPyDD-generated files: none.
