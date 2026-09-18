# P054 — Pseudo-Dirac dark matter with a heavy Z′: coupling relations, relic density and the LZ event without a dark photon

Simulated date 2026-09-11 · hep-ph · MODEL · authors: Z′ and gauge-extension model builders · competes with P011 (dark photon); uses P007, P021, P025, P026, P037, P014, P015.
Script: `output/code/P054_zprime_idm.py` (run from the root with `.venv/bin/python`, 111–142 s). All tables below are written by the script into `output/work/P054/`; the run log is `run_log.txt`.

## 1. Motivation and framework

P011 fitted the LZ event with Tucker-Smith–Weiner pseudo-Dirac DM coupled to a *light, kinetically mixed* dark photon, and found (with P025, P026) that χ₂ decay and the CMB squeeze m_A′ into a 9–34 GeV window at δ = 300 keV. We ask whether the same pseudo-Dirac fermion can instead talk to quarks through a *heavy* U(1)′ gauge boson Z′ (m_Z′ = 0.1–10 TeV) with direct quark charges, i.e. without kinetic mixing, and what LZ, the relic density, colliders, the origin of δ and the χ₂ lifetime then imply.

Lagrangian: L = Z′_μ [ g_χ χ̄γ^μχ + Σ_q g_q q̄γ^μq (+ Σ_ℓ g_ℓ ℓ̄γ^μℓ for B−L) ] with a Dirac χ of mass m_χ split into Majorana states χ₁, χ₂ (χ = (χ₁+iχ₂)/√2, m₂−m₁ = δ) by a Majorana mass. Two charge assignments: **B−L** (quarks 1/3, charged leptons and ν_L −1; χ charge q_χ, S charge 2q_χ) and **leptophobic U(1)_B** (quarks 1/3 only, extra anomaly-cancelling fermions assumed heavy). In both, the *nucleon* vector charge is 1 for the proton and the neutron, so g_N ≡ 3g_q and the interaction is exactly isoscalar O₁ — the same template LZ used, so LZ's Fig. 6 intervals and P021's κ̂ apply without an isospin conversion (unlike P011's proton-only A′).

Majorana algebra (certain): χ̄γ^μχ = iχ̄₁γ^μχ₂ (the diagonal Majorana vector currents vanish), hence
* scattering is purely inelastic, χ₁N → χ₂N, with c_p = c_n = G ≡ g_χ g_N/m_Z′² (LZ/Anand normalisation, WimPyDD c⁰ = 2G, c¹ = 0; `lz.wd_c_from_anand`); σ_n = G²μ_n²/π. No tree-level elastic scattering.
* annihilation at freeze-out (T_f ≈ m/25 ≫ δ) is coannihilation-like χ₁χ₂ → Z′* → ff̄, with |M|² equal to that of Dirac χχ̄ → ff̄; plus χ₁χ₁, χ₂χ₂ → Z′Z′ (t/u-channel) if m_Z′ < m_χ (χ₁χ₂ → Z′Z′ vanishes, P025). For n_tot = n₁+n₂ the Boltzmann equation has σ_eff = σ_{χχ̄}/2 (same as Dirac DM).

## 2. Equations

**S1. LZ rate.** N = κ N_unit(δ, m_χ) with κ = (G m_v²)², m_v = 246.2 GeV, N_unit the count for the isoscalar unit coupling (2.84 t·yr, annual-mean Baxter halo, LZ efficiency 0.96 plateau with erf edges at 5.4 and 269.9 keV, σ = 2.5/11.5 keV). N_unit is taken from P011's grid `output/work/P011/sigma_p_required.csv` (column `N_iso_unit`, halo `annual`, eff_sigma 11.5, 5-keV steps, log-interpolated). G(N) = √(N/N_unit)/m_v².
Validation: live WimPyDD (`lz.wd_hamiltonian('P054_iso_unit', {1: (2/m_v², 0)})`, 12-monthly-day averaged `lz.wd_halo`, 3 keV grid, whole ROI 1–330 keV × efficiency × 2.84) reproduces the grid at δ = 300 and 365 keV to 0.1 % (`wimpydd_validation.json`: 13574.4/13574.4, 26.2/26.2). A first attempt integrating only 200–330 keV gave 0.19 of the grid at 300 keV: the Xe inelastic window at δ = 300 keV, 1 TeV starts near 91 keV (P015), so the whole ROI must be counted — recorded as a failed approach.
Cross-checks: κ(N=1) = 7.37×10⁻⁵ (300), 3.77×10⁻³ (350), 3.8×10⁻² (365), 1.46 (380 keV) versus P021's shape-aware κ̂ = 7.4×10⁻⁵, 3.9×10⁻³, 4.0×10⁻², 2.19 (ratios 1.00, 0.98, 0.95, 0.67). P007's digitised LZ Fig. 6 upper edges κ = 2.6×10⁻⁴ (300) and 2.25×10⁻² (350 keV) correspond to G = 2.66×10⁻⁷ and 2.48×10⁻⁶ GeV⁻² and to 3.5 and 6.0 events on our annual halo (P007: 3.2–3.8 on the Sun-frame halo). The Higgsino-equivalent isoscalar coupling is G_H = √0.0777/m_v² = 4.60×10⁻⁶ GeV⁻² (P007).

**S2. Annihilation cross-sections** (derived by hand, checked numerically):
σ(χχ̄ → Z′* → ff̄)(s) = Σ_f N_c g_χ² g_f² β_f (s+2m_χ²)(s+2m_f²) / [12π s β_χ ((s−m_Z′²)² + m_Z′²Γ²)], with a factor ½ for the chiral ν_L current. Threshold limit: σv_rel = Σ_f N_c g_χ² g_f² √(1−m_f²/m_χ²)(2m_χ²+m_f²) / [2π((4m_χ²−m_Z′²)² + m_Z′²Γ²)] → Σ N_c g_χ²g_f² m_χ²/[π(4m_χ²−m_Z′²)²] for massless f (s-wave, ³S₁ → vector; no p-wave suppression). Derivation: spin-averaged |M|² = 8g⁴/D × [(p·k)(p′·k′)+(p·k′)(p′·k)+m_f²p·p′+m_χ²k·k′+2m_χ²m_f²] (Peskin–Schroeder e⁺e⁻→μ⁺μ⁻ with massive initial state), threshold bracket 2m²(2m²+m_f²), two-body phase space β_f/(8π), flux 1/(4m²) per unit v_rel.
Width: Γ(Z′→ff̄) = N_c g_f² m_Z′ β_f(1+2m_f²/m_Z′²)/(12π) (ν_L: g_ν² m_Z′/(24π)); Γ(Z′→χχ̄) added when m_Z′ > 2m_χ.
χχ̄ → Z′Z′ (s-wave, recalled/likely): σv = g_χ⁴/(16πm_χ²) (1−r²)^{3/2}/(1−r²/2)², r = m_Z′/m_χ; massless limit πα_χ²/m² as in P011/P025.
Thermal average (Gondolo–Gelmini): ⟨σv⟩(x) = x/(4K₂²(x)) ∫₀^∞ du σ(s) u(4+u)(2+u)² K₁(x(2+u)), s = m²(2+u)², evaluated on a log grid in u plus a dense patch of ±300 Γ/(2m) around the pole; scaled Bessel functions (`scipy.special.kve`). Check: for m_Z′ = 20 TeV (contact) ⟨σv⟩(x=1000)/threshold = 0.9985, (x=20) = 0.984.
Relic: dY/dx = −√(π/45) M_Pl m √g* ⟨σv⟩_eff (Y²−Y_eq²)/x², g = 4 dof, g* = 86.25, Y_eq = 45g/(4π⁴g*) x²K₂(x), Ωh² = 2.755×10⁸ Y_∞ m/GeV (Radau ODE, x = 3→1000). Fast map evaluator: Kolb–Turner x_f iteration (x_f = ln[0.038 g M_Pl m ⟨σv⟩(x_f)/√g*] − ½ln x_f) and Ωh² = 1.07×10⁹ GeV⁻¹/(√g* M_Pl ∫_{x_f}^∞⟨σv⟩x⁻²dx), calibrated on the ODE (factors 0.996/0.997/0.998 at 0.3/1/3 TeV).
Validation (`relic_method_validation.csv`): constant ⟨σv⟩_eff = 2.2×10⁻²⁶ cm³/s gives Ωh² = 0.119/0.125/0.131 (ODE) at 0.3/1/3 TeV; required 2.18/2.30/2.40×10⁻²⁶ cm³/s. Z′Z′-only thermal coupling at 1 TeV: g_χ = 0.668, α_χ = 0.0355 (P025: 0.035). Semi-analytic/ODE for resonant cases m_Z′ = 1.9/2.0/2.1 TeV: 0.987/0.974/1.020; off resonance 0.999–1.000.

**S3. Contact limit (m_Z′ ≫ 2m_χ).** Both observables scale with G²: N_LZ = N_unit (Gm_v²)², ⟨σv⟩_{χχ̄} = Σ_f N_c(g_f/g_N)² G² m_χ²/π with Σ = 6.5 (B−L: 6 quarks×3×1/9 + 3 leptons + 3×½ ν) or 2 (U(1)_B). Thermal G_relic = √(2π⟨σv⟩_req/(Σm²)), and for a relic rescaled by Ω_χ/Ω_DM = (G_relic/G)² the LZ count is the coupling-independent fixed point N_th(δ) = N_unit(δ)(G_relic m_v²)².

**S4. Maximal thermal LZ count map.** For each (m_χ, model, m_Z′): set g_N to the recalled collider ceiling g_N^max(m_Z′), solve Ωh² = 0.12 for g_χ ≤ √(4π) (both channels), and record G_max = g_χg_N/m_Z′²; then N_max(δ) = N_unit(δ)(G_max m_v²)². Why this is the maximum: for m_Z′ < m_χ the relic pins g_χ (Z′Z′) so N ∝ g_N²; for m_Z′ > m_χ (ff̄ only) N depends only on the product, which the relic fixes, and larger g_N only helps feasibility. Points where even g_χ = √(4π) leaves Ωh² > 0.12 are "infeasible".
Recalled ceilings (ALL UNCERTAIN): U(1)_B dijet g_q ≤ 0.05 (m_Z′ < 0.5 TeV), 0.10 (0.5–1.5), 0.20 (1.5–3), 0.30 (3–5), perturbativity (g_N ≤ 3) above 5 TeV; B−L: LHC dilepton m_Z′/g′ ≥ 20 TeV (LEP-II contact m_Z′/g′ ≥ 7 TeV would be weaker).

**S5. Splitting.** A scalar S with U(1)′ charge 2q_χ and Yukawa y_S S χχ gives δ = y_S v_S/√2 (S = (v_S+s)/√2); with m_Z′ = g′q_S v_S = 2g_χ v_S, v_S = m_Z′/(2g_χ) and y_S = 2√2 δ g_χ/m_Z′. If instead only a charge-q_χ scalar exists, the dimension-5 operator χχ(S₁†)²/Λ gives Λ = v₁²/(2δ) (P037-type scale).

**S6. χ₂ lifetime.** For δ < 2m_e only νν̄ (and loop γ) are open. With M = G_ν ū₁γ^μu₂ ū_νγ_μP_Lv_ν, P011's derivation gives Γ = N_ν G_ν²δ⁵/(120π³), N_ν = 3. B−L: |g_ν| = g′ = g_N so G_ν = G exactly — the lifetime is fixed by the LZ fit. U(1)_B: no tree-level coupling; with Z–Z′ mass mixing θ, G_ν = g_χ sinθ (g/2c_W)/m_Z². Floor: τ < 7.2×10¹⁶ s (P026/P011).

**S7. Sommerfeld/CMB.** Hulthén saturation value S₀ = 2π²/[ε′(1−cos(2π/√ε′))], ε′ = (π²/6) m_Z′/(α_χ m_χ) (P025's approach); Planck p_ann < 3.5×10⁻²⁸ cm³s⁻¹GeV⁻¹, f_eff = 0.35 → ⟨σv⟩ < 1.0×10⁻²⁴ cm³/s at 1 TeV.

## 3. Results

### 3.1 LZ requirement (`lz_requirement.csv`, m_χ = 1 TeV, annual halo)

| δ [keV] | N_unit | G(N=1) [GeV⁻²] | σ_n [cm²] | g_χg_N at m_Z′ = 1 / 3 TeV | m_Z′ max (g_χg_N = 1) |
|---|---|---|---|---|---|
| 250 | 1.05e5 | 5.10e-8 | 2.8e-43 | 0.051 / 0.46 | 4.4 TeV |
| 280 | 3.28e4 | 9.11e-8 | 9.0e-43 | 0.091 / 0.82 | 3.3 |
| 300 | 1.357e4 | 1.416e-7 | 2.18e-42 | 0.142 / 1.27 | 2.66 |
| 320 | 4257 | 2.53e-7 | 7.0e-42 | 0.25 / 2.3 | 1.99 |
| 340 | 815 | 5.78e-7 | 3.6e-41 | 0.58 / 5.2 | 1.32 |
| 350 | 265 | 1.013e-6 | 1.12e-40 | 1.01 / 9.1 | 0.99 |
| 360 | 63.0 | 2.08e-6 | 4.7e-40 | 2.1 / 18.7 | 0.69 |
| 366 | 21.4 | 3.57e-6 | 1.39e-39 | 3.6 / 32 | 0.53 |
| 370 | 9.43 | 5.37e-6 | 3.1e-39 | 5.4 / 48 | 0.43 |
| 380 | 0.684 | 1.99e-5 | 4.3e-38 | 20 / 180 | 0.22 |

90 % one-event band (0.105–3.65 events): G × [0.52, 3.1]. At 300 GeV: G(N=1) = 4.6e-8 (250), 4.6e-7 (300 keV); at 3 TeV: 6.4e-8 (250), 1.80e-7 (300), 7.8e-7 (350), 1.6e-6 (366), 4.5e-6 GeV⁻² (380 keV). The Higgsino-equivalent 4.6e-6 GeV⁻² is crossed at δ ≈ 368 keV (1 TeV), as in P007/P021. Since G_F/√2 = 8.2e-6 GeV⁻², a TeV Z′ with perturbative couplings (G ≲ 10⁻⁶) lives naturally at δ ≲ 340–350 keV: P021's preferred 365–385 keV needs g_χg_N ≥ 3.6–20 at m_Z′ = 1 TeV, i.e. a sub-TeV Z′ (m_Z′ ≤ 530/220 GeV for g_χg_N = 1 at 366/380 keV).

### 3.2 Contact-limit tension and thermal fixed point (`contact_limit_tension.csv`)

m_χ = 1 TeV, δ = 300 keV: G_LZ/G_relic = 3.25 (B−L, G_relic = 4.36e-8) and 1.80 (U(1)_B, 7.86e-8), i.e. ×10.5 and ×3.2 in rate; a thermal relic with the LZ coupling would have Ωh² = 0.011 (B−L) / 0.037 (U(1)_B). Fixed point N_th(300 keV) = 0.095 / 0.31 events; N_th = 1 at δ = 242 keV (B−L) / 272.5 keV (U(1)_B). At 300 GeV: 272 / 285 keV; at 3 TeV: none / 204 keV (N_th(300) = 0.007/0.022). At δ = 350 keV the mismatch is ×540 (B−L) / ×166 (U(1)_B) in rate.

### 3.3 Thermal maximum map (`thermal_max_map.csv`, `P054_results.json`; Fig. 1 right)

m_χ = 1 TeV, U(1)_B (g_N at the dijet ceiling): N_max(300 keV) = 4770 (m_Z′ = 0.1 TeV), 24 (0.5), 3.0 (1.0), 1.2 (1.18), 0.40 (1.39), 0.003 (1.93, resonance), 0.11 (3.2), 0.36 (10 TeV). δ(N_max = 1) = 375 (0.1 TeV), 343 (0.5), 319 (1.0), 303 (1.2), 279 (1.4), 230 (2.7), 246 (3.2), 272 (5.2), 277 keV (10 TeV); undefined (N_max < 1 everywhere) for m_Z′ = 1.9–2.3 TeV. B−L (g′ = m_Z′/20 TeV): N_max(300) = 5.6 (0.1 TeV), 0.30 (0.44), 0.12 (0.85), 0.93 (1.0 TeV, g_χ = 2.73), 0.12 (1.39), 9e-4 (1.93), 0.033 (3.2), 0.059 (4.4 TeV); infeasible (overabundant even at g_χ = √4π) for m_Z′ ≥ 5.2 TeV; δ(N_max=1) = 328 (0.1 TeV), 298 (1.0), 231 keV (heavy, 4.4 TeV). Summary of δ_max for a thermal, collider-allowed relic: U(1)_B 313/375/394 keV at 0.3/1/3 TeV (all at m_Z′ = 0.1 TeV), heavy-only (m_Z′ ≥ 2 TeV) 287/277/282 keV; B−L 291/328/343 keV, heavy-only none/231/222 keV. The g_q step function makes the N = 1 contour jagged at 0.5, 1.5, 3 and 5 TeV; this is the recalled ceiling, not physics.

### 3.4 Benchmarks (`benchmarks.csv`; Ωh² by full ODE; N_resc = N × min(1, Ωh²/0.12))

| tag | model | m_Z′ [GeV] | δ | g_χ | g_N (g_q) | Ωh² | N_resc | v_S [GeV] | y_S | τ(χ₂) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | U(1)_B | 500 | 300 | 0.69 | 0.051 (0.017) | 0.120 | 1.00 | 361 | 1.2e-6 | needs Z–Z′ mixing | viable |
| B2 | U(1)_B | 300 | 340 | 0.67 | 0.078 (0.026) | 0.120 | 1.00 | 224 | 2.1e-6 | idem | viable |
| B3 | U(1)_B | 3000 | 300 | 2.12 | 0.60 (0.20) | 0.013 | 0.11 | 706 | 6.0e-7 | idem | under-abundant |
| B4 | U(1)_B | 3000 | 270 | 1.12 | 0.60 (0.20) | 0.041 | 0.34 | 1335 | 2.9e-7 | idem | under-abundant |
| B5 | B−L | 1000 | 300 | 2.83 | 0.050 | 0.112 | 0.93 | 177 | 2.4e-6 | 1.7e10 s | viable, α_χ = 0.64 |
| B6 | B−L | 3000 | 300 | 8.5 | 0.15 | 0.040 | 0.33 | 177 | 2.4e-6 | 1.7e10 s | non-perturbative |
| B7 | U(1)_B | 500 | 366 | 5.9 | 0.15 | ~0 | 0 | 42 | 1.2e-5 | — | non-perturbative, no relic |
| B8 | U(1)_B | 200 | 380 | 5.3 | 0.15 | ~0 | 0 | 19 | 2.9e-5 | — | idem |
| B9 | U(1)_B | 2000 (=2m_χ) | 300 | 0.94 | 0.60 | ~0 | 0 | 1059 | 4.0e-7 | — | resonance: Ω → 0 |
| B10 | U(1)_B (3 TeV χ) | 1500 | 300 | 1.16 | 0.35 (0.12) | 0.120 | 1.00 | 649 | 6.5e-7 | idem | viable |
| B11 | U(1)_B (300 GeV χ) | 150 | 280 | 0.38 | 0.012 (0.004) | 0.120 | 1.00 | 200 | 2.0e-6 | idem | viable |
| B13 | B−L | 300 | 300 | 0.67 | 0.019 | 0.120 | 1.00 | 222 | 1.9e-6 | 1.7e10 s | dilepton-excluded (m/g′ = 16 TeV) |

B12 (B−L, m_Z′ = 1 TeV along the LZ constraint) found the second root g_χ = 0.063, g′ = 2.2 (collider-excluded); B5 is the physical branch. Monojet: for m_Z′ < 2m_χ the Z′ cannot decay to χχ̄, so mono-X searches are irrelevant to every viable benchmark; only B3/B4/B6 (m_Z′ = 3 TeV > 2m_χ) have an invisible width, Γ(χχ̄)/Γ_SM ~ g_χ²/(2g_N²) ≈ 6 (B3), which is itself a collider handle (dijet BR reduced) not evaluated here.

### 3.5 Splitting (`splitting_yukawa.csv`)
y_S(δ = 300 keV) = 4.2×10⁻⁷ (v_S = 1 TeV), 4.2×10⁻⁸ (10 TeV), 4.2×10⁻⁹ (100 TeV); y_e = 2.9×10⁻⁶. The viable benchmarks have v_S = 0.18–0.72 TeV, y_S = (0.6–2.4)×10⁻⁶ — the size of the electron Yukawa, and 10⁶ below the O(1) Yukawa the same S must give right-handed neutrinos in B−L. No symmetry forbids the Yukawa once S exists; a radiative origin is impossible (the operator must carry charge 2q_χ). With a charge-q_χ scalar only: Λ = v₁²/(2δ) = 1.7×10⁶ TeV for v₁ = 1 TeV (cf. P037's doublet, 1.7×10⁵ TeV).

### 3.6 χ₂ lifetime (`chi2_lifetime.csv`)
B−L (G_ν = G_LZ): τ = 3.2e11 (250), 1.68e10 (300), 1.5e8 (350), 9.8e6 (366), 2.6e5 s (380 keV) — 10⁻⁶–10⁻¹¹ of the floor, so χ₂ has decayed (f₂(today) = 0) and P026's exothermic bound is satisfied by 6–12 decades. τ ∝ G⁻²δ⁻⁵. U(1)_B: required G_ν ≥ 6.8×10⁻¹¹ GeV⁻² at 300 keV, i.e. g_χ sinθ_ZZ′ ≥ 1.5×10⁻⁶ (2.4×10⁻⁶ at 250, 0.85×10⁻⁶ at 380 keV); any Z–Z′ mixing above 10⁻⁶ (EWPT allows ~10⁻³; a quark-loop kinetic mixing ε ~ e g_q/(6π²)×ln ~ 10⁻³ (uncertain) also suffices) makes χ₂ decay. A strictly unmixed leptophobic Z′ would leave χ₂ stable and the model excluded by 858 exothermic events (P011).

### 3.7 Sommerfeld and CMB (`sommerfeld_cmb.csv`)
At the Z′Z′-thermal coupling (1 TeV): α_χ = 0.035–0.073, ε_φ = m_Z′/(α_χm) = 2.8 (0.1 TeV), 8.3 (0.3), 13 (0.5), 12 (0.9 TeV); S₀ = 2.16, 1.28, 1.17, 1.18; ⟨σv⟩_CMB = S₀ × 2.3×10⁻²⁶ = (2.7–5.0)×10⁻²⁶ cm³/s, 0.03–0.05 of the Planck limit 1.0×10⁻²⁴. For m_Z′ > m_χ the surviving χ₁ has no tree-level annihilation at all (vector current off-diagonal, χ₂ decayed): zero indirect signal. P025's m_A′ ≥ 9 GeV problem does not arise.

## 4. Figures
`figures/P054_fig1_G_vs_delta_and_Nmax_map.png` — Left: G = g_χg_N/m_Z′² giving one LZ event vs δ for m_χ = 0.3/1/3 TeV (1 TeV band 0.105–3.65 events); horizontal lines: Higgsino-equivalent (P007), U(1)_B thermal maxima at m_Z′ = 0.5 TeV (Z′Z′ × dijet ceiling), 3 and 10 TeV (exact ff̄ thermal lines), B−L at 1 TeV. Right: log₁₀ N_max(m_Z′, δ) for U(1)_B, 1 TeV (diverging palette centred on N = 1; contours 0.105/1/3.65; dotted lines at m_Z′ = m_χ and 2m_χ).

## 5. Failed or abandoned
* Validation window 200–330 keV (gave 0.19 of the grid at 300 keV) — replaced by the whole ROI.
* Quad-based thermal average + Radau ODE inside every root-finding step: 200–250 s per (m_χ, model) block; replaced by a vectorised trapezoid average and a calibrated semi-analytic freeze-out (ODE kept for validation and benchmarks). A linear u-grid gave 0.66 of the threshold value at x = 1000 (weight at u ~ 1/x unresolved) — replaced by a log grid.
* Benchmark auto-mode that rescaled g_N after fixing g_χ (Ωh² = 0.13–0.18) — replaced by a root-find along the LZ constraint.
* B12 landed on a collider-excluded second root; kept in the CSV, B5 reported.

## 6. Discussion
The heavy-Z′ version of pseudo-Dirac DM behaves very differently from P011's dark photon. (i) LZ fixes G = g_χg_N/m_Z′² = 1.4×10⁻⁷ (300 keV) to 2×10⁻⁵ GeV⁻² (380 keV); with perturbative couplings a TeV Z′ therefore lives at δ ≲ 340 keV, and the P021 maximum at 365–385 keV requires m_Z′ ≲ 0.2–0.5 TeV *and* g_χ > √(4π) if g_q respects dijet limits — the Higgsino corner is not available to a Z′. (ii) Because scattering and s-channel annihilation share G², a heavy thermal Z′ has a coupling-independent LZ prediction: 0.3 (U(1)_B) or 0.1 (B−L) events at 300 keV, reaching one event only at δ ≈ 272 / 242 keV (1 TeV) — the classic Z′-portal tension in reverse (LZ wants more coupling than the relic allows). Near m_Z′ ≈ 2m_χ the resonance makes it hopeless (N_max = 0.003). (iii) A Z′ lighter than χ escapes: the relic is set by χχ → Z′Z′ (g_χ ≈ 0.66–0.7 at 1 TeV) and LZ by g_N alone; leptophobic benchmarks (m_Z′ = 0.3–0.5 TeV, g_q = 0.02–0.03) satisfy LZ, Ωh² = 0.12, dijets and Planck (S₀ ≤ 1.3, 3–5 % of the bound) — but B−L is dilepton-squeezed and survives only as a strongly coupled point (m_Z′ ≈ 1 TeV, g′ = 0.05, α_χ = 0.6). (iv) δ must come from a charge-2q_χ scalar with y_S ~ 10⁻⁶ — electron-Yukawa-sized, unexplained but not absurd. (v) χ₂ decays in 500 yr (B−L) or needs any Z–Z′ mixing > 10⁻⁶ (U(1)_B), so P026's floor is irrelevant, and the CMB bound that removed P011's light dark photons never bites. The price is that the Z′ model *prefers δ ≈ 250–340 keV*, where P021 puts 2.5–3.0σ rather than 3.6σ and where P002 finds 248 keV at the 98th percentile of the recoil spectrum.

## 7. References
1. LZ Collaboration, arXiv:2609.02823 (2026). 2. D. Tucker-Smith, N. Weiner, PRD 64, 043502 (2001). 3. P. Gondolo, G. Gelmini, NPB 360, 145 (1991). 4. E. W. Kolb, M. S. Turner, The Early Universe (1990). 5. G. Steigman, B. Dasgupta, J. F. Beacom, PRD 86, 023506 (2012). 6. A. Alves, S. Profumo, F. S. Queiroz, JHEP 04 (2014) 063 (Z′ portal). 7. P. Langacker, Rev. Mod. Phys. 81, 1199 (2009). 8. S. Cassel, J. Phys. G 37, 105009 (2010). 9. Corpus: P002, P007, P011, P014, P015, P021, P025, P026, P037.

## 8. Tools and provenance (mirrors provenance/P054.json)
Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (integrate.solve_ivp Radau, optimize.brentq, special.kve/erf/erfc); pandas 3.0.5; matplotlib 3.11.2; WimPyDD 2.0.4 (eft_hamiltonian, diff_rate, streamed_halo_function via lzcommon); common/lzcommon.py (LZ, M_V_GEV, M_NUCLEON_GEV, wd_halo, wd_hamiltonian, wd_rate, E_R_range_keV). Hand derivations: Majorana current decomposition, s-channel cross-section and threshold limit, Gondolo–Gelmini change of variables, splitting relation, lifetime from P011's width.
Local inputs: P011 sigma_p_required.csv (N_iso_unit grid) and P011 script lines 100–142 (halo/efficiency recipe); P021 P021_scan_baseline.csv (κ̂); papers P007, P011, P014, P021, P025, P026, P031, P037; dossier; ledger; PAPER_GUIDE; ENVIRONMENT_versions.txt.
Recalled knowledge (17 items): PDG constants, Planck Ωh² = 0.12 (certain); g* = 86.25 (likely); Ωh² = 2.755e8 Y m (certain); K&T x_f/Ωh² formulae (certain); Gondolo–Gelmini average (certain); Dirac vector s-channel cross-section (derived; textbook, certain); Z′→ff̄ width (certain); χχ̄→Z′Z′ formula (likely); Planck p_ann 3.5e-28, f_eff 0.35 (likely); Hulthén S₀ (likely); dijet g_q ceilings (uncertain); B−L dilepton m/g′ ≥ 20 TeV (uncertain); LEP-II m/g′ ≥ 7 TeV (likely); EWPT Z–Z′ mixing ≲ 10⁻³ (likely); quark-loop kinetic mixing size (uncertain); B−L charges and anomaly-free vector-like χ (certain); electron Yukawa (certain).
WimPyDD-generated files: none (diff_rate only).
