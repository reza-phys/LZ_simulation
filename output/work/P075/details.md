# P075 — Thermal history of ~1 TeV pseudo-Dirac dark matter with the LZ-fitted coupling: freeze-out, coannihilation and non-thermal production

Simulated arXiv date 2026-09-15 · hep-ph · particle-cosmology phenomenology group · category MODEL.
Scripts: `output/code/P075_wimpydd_grid.py` (cached WimPyDD grids, 5 min) and `output/code/P075_thermal_history.py` (3 min). All tables in `output/work/P075/`, figures in `output/work/P075/figures/`.

## 1. Motivation and framework

Three model families have been proposed for the LZ 248 keV event as inelastic (pseudo-Dirac) dark matter with δ ≈ 250–390 keV: the pure Higgsino (P007; gauge-fixed σ_n = 7.4×10⁻³⁹ cm², one event only at δ = 358–380 keV for 1 TeV), a kinetically mixed dark photon (P011; σ_p free, any δ from 204 keV to δ_max), and a heavy U(1)′ boson (P054; G = g_χg_N/m_Z′² fixed by LZ, coupling-independent 0.1–0.3 events in the contact limit). P025 and P026 treated indirect detection and the excited state; P054 noted that in the contact limit the LZ rate and the s-wave relic annihilation both scale as G². We ask, for each family and every point of the (m_χ, δ) plane with m = 0.3–10 TeV and δ = 250–390 keV, whether the coupling that gives one LZ event is compatible with (a) standard freeze-out of the coannihilating χ₁/χ₂ pair, (b) freeze-out into mediator pairs (A′A′, Z′Z′), (c) non-thermal production (moduli/gravitino decay, freeze-in, asymmetric DM) and (d) the gauge-fixed Higgsino relic, and we quantify the non-thermal fraction, entropy dilution D or reheating temperature T_RH needed where thermal production fails.

Conventions. LZ/Anand unit coupling c = 1/m_v², m_v = 246.2 GeV; WimPyDD c⁰ = c_p + c_n (P003). Isoscalar unit count N_iso(m, δ) [c_p = c_n = 1/m_v²] and proton-only unit count N_p(m, δ) [c_p = 1/m_v², c_n = 0] are the expected LZ events (2.84 t yr, annual-mean Baxter halo, efficiency 0.96 plateau with erf edges at 5.4/269.9 keV, σ = 2.5/11.5 keV — the P011/P054 recipe). A coupling G (GeV⁻²) gives N = N_unit (G m_v²)². The LZ 90 % two-sided band for one observed event is 0.105–3.65 expected events (P007/P045). Ω_DM h² = 0.120.

## 2. Inputs

| input | value / source |
|---|---|
| N_iso, N_p at 300/1000/3000 GeV, δ = 240–400 keV (5 keV) | `output/work/P011/sigma_p_required.csv` (halo = annual, eff σ = 11.5 keV) |
| N_iso, N_p at 500/2000/5000/10000 GeV, δ = 250–390 keV (10 keV) | live WimPyDD (this work, `N_unit_grid_live.csv`); reproduces the P011 grid at 1 TeV to 1×10⁻⁴ (300 and 365 keV) |
| Higgsino N(m, δ) at 300/500/1000/2000/4000 GeV | `output/work/P007/N_events_grid.csv` (annual, 11.5 keV) |
| Higgsino N at 5000/10000 GeV | live WimPyDD with P007's couplings c⁰_wd = −7.618×10⁻⁶, c¹_wd = 8.871×10⁻⁶ GeV⁻² (`output/work/P007/higgsino_couplings.json`); check at 1 TeV: ratio to P007 = 1.009/1.025/1.044/1.067 at 300/350/365/375 keV (P007 used a slightly different v-grid) |
| P054 contact-limit numbers | `output/work/P054/contact_limit_tension.csv` (N_th = 0.095/0.31 at 1 TeV, 300 keV) |
| P025 Higgsino freeze-out formula and A′A′ conventions | `output/work/P025/details.md` §2.1, §3.1 |
| P011 χ₂-decay floor on m_A′ | 5.0/2.45/0.92/0.55/0.25 GeV at δ = 250/300/350/365/380 keV (P011 table; P026) |
| P025 Planck-allowed Sommerfeld factor | S_allowed = 23 at 1 TeV for the thermal A′A′ cross-section (f_eff = 0.35); ∝ m at fixed ⟨σv⟩ |
| lzcommon | M_V_GEV, M_NUCLEON_GEV, GEV_TO_CM2, VESC_KMS = 544, V_EARTH_ORBIT = 29.8, A_XE_MEAN, AMU_GEV, wd_halo/wd_hamiltonian/wd_rate, E_R_range_keV, LZ efficiency numbers |

Recalled (flagged): M_Pl = 1.2209×10¹⁹ GeV, s₀/(ρ_c/h²) = 2.755×10⁸ GeV⁻¹ (certain); g_*(T) SM table (likely, ±5 %); ⟨σv⟩_canonical = 2.2×10⁻²⁶ cm³/s for Ωh² = 0.12 (Steigman–Dasgupta–Beacom 2012; certain); Higgsino ⟨σv⟩_eff = g⁴(21 + 3tan²θ_W + 11tan⁴θ_W)/(512πμ²) (Arkani-Hamed–Delgado–Giudice 2006; likely), g = 0.6517, sin²θ_W = 0.2312; thermal Higgsino mass 1.1 TeV (certain); Griest–Seckel coannihilation formula (certain); non-thermal annihilation-limited yield Y = H/(s⟨σv⟩) at T_RH (Moroi–Randall 2000, Giudice–Kolb–Riotto 2001; likely); BBN floor T_RH ≳ 4–5 MeV (Hannestad 2004, de Salas et al. 2015; likely); dark-photon ε ≤ 10⁻³ for m_A′ = 1–10 GeV (BaBar/LHCb; likely); Y_eq and freeze-in collision-term formulae (Gondolo–Gelmini 1991; certain); ∫u^μK_ν du = 2^{μ−1}Γ((1+μ+ν)/2)Γ((1+μ−ν)/2) (certain); Σ_f N_c(g_f/g_N)² = 6.5 (B−L), 2 (U(1)_B) (P054), 8 for the dark photon (Σ N_c Q_f² with top open; certain); asymmetric-DM oscillation argument (Tulin–Yu–Zurek 2012; likely).

## 3. Method and equations

### 3.1 Mass interpolation of the LZ grids
Each tabulated mass has its own δ grid; the kinematic ceiling δ_c(m) = ½ μ(m, m_Xe) v_max², v_max = v_esc + v_⊙ + v_orb = 544 + 250.6 + 29.8 = 824.4 km/s (δ_c(1 TeV) = 412 keV), moves with m, so we interpolate ln N linearly in ln m at fixed ξ = δ/δ_c(m). Leave-one-out test (predict 1 TeV from 0.5 and 2 TeV, i.e. a factor-4 leap, twice the actual spacing): ratios 1.04/1.12/1.18 at δ = 300/350/366 keV but ×32 at 380 keV; predicting 3 TeV from 2 and 5 TeV: 1.01–1.05 at 300–380 keV (`mass_interpolation_test.csv`). Contours at δ ≳ 375 keV between tabulated masses are therefore only indicative; every number quoted below is at a tabulated mass unless stated.

### 3.2 Relic density
Boltzmann equation for the total yield Y = (n₁ + n₂)/s, g = 4 internal degrees of freedom (two Majorana states):
dY/dx = −√(π/45) M_Pl m √g_*(T) ⟨σv⟩_eff(x) (Y² − Y_eq²)/x², Y_eq = 45g/(4π⁴g_*) x² K₂(x), x = m/T, integrated with `solve_ivp` (Radau, rtol 10⁻⁸) from x = 3 to 2000; Ωh² = 2.755×10⁸ Y_∞ (m/GeV). g_*(T) from the recalled SM table (g_*s = g_* assumed; the derivative term is neglected — at T_fo = 10–400 GeV it is ≤ 2 %). Freeze-out x_f defined by Y = 2.5 Y_eq.
Validation (`relic_validation.json`): constant ⟨σv⟩ = 2.2×10⁻²⁶ cm³/s gives Ωh² = 0.119/0.123/0.124 at m = 100/1000/10000 GeV (x_f = 24.4/26.6/28.8), i.e. the canonical value to 3 %; the required ⟨σv⟩_eff(Ωh² = 0.12) is 2.19/2.26/2.30/2.28×10⁻²⁶ cm³/s at 0.3/1/3/10 TeV. A constant g_* = 86.25 (P054's choice) changes Ω at 1 TeV by 0.3 %. Higgsino: with the ADG effective cross-section Ωh² = 0.120 at μ = 1150 GeV (literature 1.1 TeV; P025's semi-analytic estimate 1167 GeV) and Ωh²(1 TeV) = 0.0914 (P007 0.099, P025 0.088).

### 3.3 Coannihilation with the δ-split partner (Griest–Seckel)
σ_eff = Σ_ij σ_ij (g_ig_j/g_eff²)(1+Δ_i)^{3/2}(1+Δ_j)^{3/2} e^{−x(Δ_i+Δ_j)}, g_eff = Σ g_i(1+Δ_i)^{3/2}e^{−xΔ_i}, Δ₂ = δ/m. For a pseudo-Dirac pair only two structures occur: off-diagonal (σ₁₂ ≠ 0 only: vector s-channel χ₁χ₂ → Z′*/A′* → ff̄) and diagonal (σ₁₁ = σ₂₂, σ₁₂ = 0: χχ → A′A′, Z′Z′ via the off-diagonal vertex, P025). With w = (1+Δ)^{3/2}e^{−xΔ} = 1 − ε: σ_eff(off)/σ_eff(0) = 4w/(1+w)² ≈ 1 − ε²/4 and σ_eff(diag)/σ_eff(0) = 2(1+w²)/(1+w)² ≈ 1 + ε²/4 — the first-order effect cancels because χ₂ enters both the numerator and g_eff. Results at the solved x_f (`coannihilation_delta_effect.csv`): δ/T_f = 2.5×10⁻⁵ (300 GeV) … 8.7×10⁻⁷ (10 TeV) for δ = 300 keV; |1 − σ_eff ratio| = 1.4×10⁻¹⁰ (300 GeV) to 1.7×10⁻¹³ (10 TeV), largest 2.3×10⁻¹⁰ at (300 GeV, 380 keV). The first-order change of Y_eq (g_eff = 4 − 2ε) is ≤ 1.6×10⁻⁵. For comparison the Higgsino's charginos (Δm± = 342 MeV, P014) carry Boltzmann weights 0.974/0.992/0.997 at 0.3/1/3 TeV (a 0.3–2.6 % effect, already inside the ADG formula). Conclusion: δ ≈ 300 keV is invisible to freeze-out; χ₁ and χ₂ are produced with equal abundance (as assumed by P026) and all Dirac-limit annihilation formulae hold.

### 3.4 Non-thermal helpers
Required yield Y_req = 0.12/(2.755×10⁸ m). Annihilation-limited non-thermal production (moduli/gravitino/inflaton decay with T_RH < T_fo): Y_NT = H(T_RH)/(s(T_RH)⟨σv⟩) with H = 1.66√g_* T²/M_Pl, s = 2π²g_*T³/45, so Y_NT ∝ 1/T_RH. Setting Y_NT = Y_req defines T_RH*; a branching-limited yield (small BR into DM) can only sit *below* Y_NT, so T_RH ≤ T_RH* is a necessary condition for any non-thermal scenario with the given ⟨σv⟩ (it coincides with "T_RH = T_fo × f_th" to within g_* factors). Entropy dilution after freeze-out: D = Ω_th/Ω_DM (the χ₂ fraction f₂ of P026 is unaffected). Freeze-in (§3.8) uses the collision term R(T) = Σ_f (2N_c)² T/(32π⁴) ∫ds σ_f(s) s^{3/2} K₁(√s/T) with the colour-averaged σ_f(ff̄ → χχ̄) = (g_f/g_N)² G² β_χ (s+2m²)/(12πN_c) (from P054's σ(χχ̄ → ff̄) by detailed balance, g_χ²/g_f² and (β_χ/β_f)²), i.e. Σ_f(2N_c)²σ_f = 4 S G² β_χ(s+2m²)/(12π); Y_FI = ∫₀^{T_RH} 2R/(sHT) dT. Massless closed form (UV freeze-in): R = 8SG²T⁸/π⁵, Y_FI = (16SG²/π⁵)(45/2π²g_*)(M_Pl/1.66√g_*) T_RH³/3; the numerical integrator reproduces it to 0.1 % at T_RH = 20m.

### 3.5 Higgsino (Part D)
f_th(m) = Ω_th/0.12 from the ODE with the ADG cross-section; N_LZ(m, δ) = min(f_th, 1) × N_H(m, δ) (P007 kernels), i.e. the local density is rescaled by the thermal fraction when under-abundant (ρ₀ = 0.3 GeV cm⁻³ is the total DM density; P025 used the same rescaling). Above 1150 GeV the thermal Higgsino is over-abundant and must be diluted to Ω_DM (D = f_th), after which N_LZ = N_H.

### 3.6 Heavy Z′, contact limit (Part E)
For m_Z′ ≫ 2m_χ: N_LZ = N_iso (G m_v²)², ⟨σv⟩(χχ̄ → ff̄) = S G² m²/π (s-wave, massless f, P054 S3), σ_eff = ½⟨σv⟩_{χχ̄} (only σ₁₂ ≠ 0; P025 counting). Thermal G_relic = √(2π⟨σv⟩_req/(S m²)); with a relic rescaled by (G_relic/G_LZ)² the count is the coupling-independent fixed point N_th(m, δ) = N_iso(G_relic m_v²)² = Ω_th(G_LZ)/Ω_DM. Non-thermal fraction 1 − N_th (N_th < 1), dilution D = N_th (N_th > 1), T_RH* from §3.4 with ⟨σv⟩ at G_LZ.

### 3.7 Dark photon (Part F)
Relic from χ₁χ₁, χ₂χ₂ → A′A′ (m_A′ < m_χ): σ_eff = πα_D²/(2m²) (P025 "eff"; P011 used πα_D²/m², √2 lower α_D) → α_D(m) = √(2m²⟨σv⟩_req/π) = 0.0104/0.0175/0.0351/0.0707/0.106/0.176/0.352 at 0.3/0.5/1/2/3/5/10 TeV (P011 convention: 0.0073/0.0248/0.075/0.249 at 0.3/1/3/10 TeV; P025: 0.0104/0.0346/0.104). LZ fixes σ_p(N=1) = c_p²μ_p²/π with c_p = 1/(m_v²√N_p) and hence ε²α_D/m_A′⁴ = σ_p/(16παμ_p²) (P011). With thermal α_D and ε ≤ 10⁻³: ceiling m_A′^max = (16πα α_D ε_max² μ_p²/σ_p)^{1/4}. Floors: Planck via P025's saturated Sommerfeld criterion S_sat = 6α_D m/m_A′ ≤ S_allowed, S_allowed = 23 (m/TeV)(α_D^th/α_D)² [p_ann ∝ S⟨σv⟩₀/m, ⟨σv⟩₀ ∝ α_D²], and the χ₂-decay floor (P011/P026). A thermal window exists iff ceiling > floor. If closed, lowering α_D by k (Ω_th grows by k⁻², so a dilution D = k⁻² is required) moves floor ∝ k³ and ceiling ∝ k^{1/4}: the window reopens for k < (ceiling/floor)^{4/11}, D_open = (ceiling/floor)^{−8/11}. Contact-limit fixed point for a *heavy* dark photon (m_A′ ≫ 2m_χ, no A′A′ channel): as §3.6 with N_p and S = Σ N_cQ_f² = 8 (t open for √s = 2m ≥ 600 GeV). For m_A′ < 2m_χ the s-channel piece at the LZ coupling relative to ⟨σv⟩_req is (G_LZ,p/G_relic,8)² × m_A′⁴/(4m² − m_A′²)² = N_th,A′⁻¹ × m_A′⁴/(4m² − m_A′²)².

### 3.8 Freeze-in / low reheating (Part G) and asymmetric DM (Part H)
Freeze-in needs the χ population never to reach equilibrium: Y_FI(T_RH) ≪ Y_eq. We compute Y_FI for T_RH = 10m (the UV-dominated regime) and the T_RH at which Y_FI = Y_req (Boltzmann-suppressed production below m, annihilation neglected, so a lower limit on the true T_RH). Asymmetric DM: a Majorana splitting δ makes χ ↔ χ̄ oscillate with period 2π/δ; the asymmetry is erased on t ~ 1/δ unless δ is generated below T_fo.

## 4. Results

### 4.1 Higgsino (`higgsino_thermal_history.csv`, `higgsino_delta_windows.csv`)
| m [GeV] | Ωh²_th | f_th | x_f | T_fo [GeV] | non-thermal fraction | D | T_RH* [GeV] | δ(N=1) full ρ | 90 % band | δ(N=1) thermal ρ | band |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | 0.0088 | 0.073 | 28.1 | 10.7 | 0.93 | – | 0.86 | 310 | 307–310 | 302 | 295–310 |
| 500 | 0.0237 | 0.198 | 27.6 | 18.1 | 0.80 | – | 3.8 | 342 | 336–351 | 335 | 328–345 |
| 700 | – | 0.381 | – | – | 0.62 | – | – | 357 | 350–368 | 352 | 344–364 |
| 1000 | 0.0914 | 0.762 | 27.0 | 37.1 | 0.24 | – | 29.5 | 366 | 358–380 | 364 | 356–378 |
| 1100 | 0.110 | 0.917 | 26.8 | 41.0 | 0.08 | – | 39 | 368 | 360–383 | 367 | 359–383 |
| 1150 | 0.120 | 1.000 | 26.8 | 42.9 | 0 | 1 | (= T_fo) | 369 | 360.5–385 | 369 | 360.5–385 |
| 1300 | – | 1.27 | – | – | 0 | 1.27 | – | 371 | 362–389 | – | – |
| 2000 | 0.350 | 2.92 | 26.3 | 76 | 0 | 2.9 | – | 375 | 365–391 | – | – |
| 3000 | 0.764 | 6.36 | 25.8 | 116 | 0 | 6.4 | – | 377 | 367–394 | – | – |
| 5000 | 2.02 | 16.8 | 25.3 | 198 | 0 | 17 | – | 376 | 365–≥390 | – | – |
| 10000 | 7.64 | 63.6 | 24.6 | 406 | 0 | 64 | – | 373 | 361–≥390 | – | – |
The Higgsino is thermal (f_th = 0.8–1.25) only for m = 1025–1290 GeV, where LZ's one event requires δ = 360–385 keV (N = 1 at 368–371 keV). Below, the thermal fraction f_th ≈ (m/1.15 TeV)² lowers the LZ count proportionally: at 1 TeV N(366 keV) = 0.73 instead of 0.96 and δ(N=1) moves from 366 to 364 keV; at 500 GeV from 342 to 335 keV, at 300 GeV from 310 to 302 keV — shifts of 2–8 keV, smaller than P018's halo-tail uncertainty (±10–17 keV). A non-thermal top-up of 24 % (1 TeV) to 93 % (300 GeV) requires T_RH ≤ 29.5 GeV (1 TeV; T_fo = 37 GeV), 3.8 GeV (500 GeV), 0.86 GeV (300 GeV) — all above the BBN floor, so a light Higgsino with a moduli-type history is viable. Above 1.3 TeV dilution D = 2.9 (2 TeV) to 64 (10 TeV) is mandatory; the LZ window is then P007's full-density window and hardly moves with mass (373–377 keV). The classic "thermal Higgsino at 1.1 TeV" therefore coincides with the LZ-preferred corner δ = 360–385 keV of P021 — but *only* there, and only in a 25 % mass band.

### 4.2 Heavy Z′ contact limit (`zprime_contact_map.csv`, `zprime_contact_windows.csv`)
G_relic (GeV⁻²): B−L 1.42×10⁻⁷/4.32×10⁻⁸/1.45×10⁻⁸/4.34×10⁻⁹ and U(1)_B 2.56×10⁻⁷/7.80×10⁻⁸/2.62×10⁻⁸/7.82×10⁻⁹ at 0.3/1/3/10 TeV. Fixed point N_th = Ω_th(G_LZ)/Ω_DM:
| m [GeV] | model | N_th(250) | N_th(300) | N_th(350) | N_th(366) | δ(N_th = 1) | δ(N_th = 0.105) | T_RH* (300 keV) | T_RH* < 5 MeV beyond |
|---|---|---|---|---|---|---|---|---|---|
| 300 | B−L | 4.04 | 0.035 | 0 | 0 | 272 | 293 | 0.47 GeV | 317 keV |
| 300 | U(1)_B | 13.1 | 0.113 | 0 | 0 | 285 | 300 | 1.5 GeV | 320 keV |
| 1000 | B−L | 0.72 | 0.093 | 0.0018 | 1.5×10⁻⁴ | < 250 (P054: 242) | 297.5 | 3.7 GeV | 372 keV |
| 1000 | U(1)_B | 2.34 | 0.303 | 0.0059 | 4.8×10⁻⁴ | 272 | 318 | 12 GeV | 377 keV |
| 3000 | B−L | 0.041 | 0.0065 | 3.5×10⁻⁴ | 7×10⁻⁵ | – | < 250 | 0.82 GeV | 380 keV |
| 3000 | U(1)_B | 0.133 | 0.021 | 0.0011 | 2.3×10⁻⁴ | – | 256 | 2.6 GeV | 385 keV |
| 10000 | B−L | 1.3×10⁻³ | 2.1×10⁻⁴ | 1.4×10⁻⁵ | 3×10⁻⁶ | – | – | 0.15 GeV | – |
| 10000 | U(1)_B | 4.1×10⁻³ | 6.9×10⁻⁴ | 4.6×10⁻⁵ | 1.2×10⁻⁵ | – | – | 0.31 GeV | – |
P054's 1 TeV values (0.095/0.31) are reproduced (0.093/0.303; the 2 % difference is g_*(T) and the ODE ⟨σv⟩_req). Standard freeze-out with the LZ coupling (0.105 ≤ N_th ≤ 3.65) is possible only in the low-δ, low-mass corner: δ ≤ 297.5 keV (B−L) / 318 keV (U(1)_B) at 1 TeV, ≤ 293/300 keV at 300 GeV, ≤ 256 keV (U(1)_B) at 3 TeV and nowhere above 5 TeV. At 300 GeV and δ ≤ 270 keV the LZ coupling *under*-annihilates (N_th = 4–13) and a dilution D = N_th is needed. Everywhere else the LZ coupling over-annihilates: the non-thermal fraction is 1 − N_th = 70–91 % at (1 TeV, 300 keV), > 99 % for δ ≥ 350 keV, and the annihilation-limited yield forces T_RH ≤ 3.7–12 GeV (300 keV), 0.14–0.27 GeV (350 keV), 20–60 MeV (365 keV) and < 5 MeV beyond δ = 372 (B−L) / 377 keV (U(1)_B) at 1 TeV — below the BBN floor, so *no* production mechanism (thermal, non-thermal or freeze-in) delivers Ω_DM for a heavy contact Z′ in the P021 corner: the s-wave annihilation at the LZ coupling depletes any population present after T ≈ 5 MeV. Light Z′ (m_Z′ < m_χ, relic from Z′Z′ with g_χ ≈ 0.7) is P054's escape and is not re-analysed here (P054: δ ≲ 340 keV with recalled collider ceilings).

### 4.3 Dark photon (`darkphoton_map.csv`, `darkphoton_windows.csv`, `darkphoton_schannel_fraction.csv`)
Thermal A′A′ freeze-out fixes α_D(m) independently of ε and m_A′, so a thermal relic is *always* available; the question is whether the LZ ε fits between the Planck floor and the ε ≤ 10⁻³ ceiling. At 1 TeV: σ_p(N=1) = 1.20×10⁻⁴²/8.59×10⁻⁴²/2.99×10⁻⁴⁰/2.29×10⁻³⁹/5.60×10⁻³⁸ cm² (P011), ε(m_A′ = 10 GeV, α_D = 0.0351) = 5.2×10⁻⁵/1.4×10⁻⁴/8.2×10⁻⁴/2.3×10⁻³/1.1×10⁻² at δ = 250/300/350/365/380 keV; m_A′ ceiling 43.8/26.8/11.0/6.6/3.0 GeV against the Planck floor 9.15 GeV (χ₂ floor 5.0/2.45/0.92/0.55/0.25 GeV, always weaker). The window closes at δ = 316/345/356/338/312/259 keV for m = 0.3/0.5/1/2/3/5 TeV and is closed at all δ for 10 TeV (ceiling/floor = 0.55 at 250 keV; Planck floor 92 GeV because α_D = 0.35). Reopening by dilution needs only modest D: at 1 TeV D = 1.27 (365 keV), 2.26 (380 keV) [α_D = 0.031, 0.023]; at 3 TeV 1.48/1.88/2.60 at 350/365/380 keV; at 10 TeV 2.1–5.4 for 300–380 keV. Because floor ∝ α_D³ and ceiling ∝ α_D^{1/4}, a factor-1.5 reduction of α_D (D ≈ 2) opens the Higgsino corner for the dark photon — the CMB exclusion of P025 is a statement about a strictly thermal α_D.
Contact-limit fixed point (heavy A′, m_A′ ≫ 2m_χ, S = 8, proton-only): N_th,A′ = 0.0080/0.0193/0.0013/4×10⁻⁵ at (300 keV; 0.3/1/3/10 TeV), ≈ 0.2 × the B−L value because N_p/N_iso ≈ 1/4 and S = 8 vs 6.5; N_th = 1 only at δ ≈ 245 keV (300 GeV) and below 250 keV elsewhere; N_th = 0.105 at 279/285/257 keV (0.3/0.5/1 TeV). So the coupling-independent prediction "0.1–0.3 events at 300 keV" becomes 0.02 events for a heavy dark photon — but this regime needs ε g_D = G_LZ m_A′² ≳ 0.14 (m_A′/TeV)² × (N_iso/N_p)^{1/2} ≈ 0.28 (m_A′/TeV)², excluded by dilepton searches/perturbativity for m_A′ ≥ 2m_χ. For the realistic m_A′ < m_χ the s-channel χ₁χ₂ → A′* → ff̄ at the LZ coupling is 3.2×10⁻¹²/3.2×10⁻⁸/3.3×10⁻⁴ of ⟨σv⟩_req at m_A′ = 1/10/100 GeV (1 TeV, 300 keV) and exceeds it only for m_A′ ≳ 1 TeV: LZ rate (∝ ε²α_D/m_A′⁴) and relic (∝ α_D²) decouple, and P054's fixed point does not apply to the dark photon of P011/P025.

### 4.4 Freeze-in and low reheating (`freeze_in_low_reheating.csv`)
With the LZ coupling G_LZ = 1.4×10⁻⁷ GeV⁻² (1 TeV, 300 keV) and T_RH = 10m the UV freeze-in yield exceeds Y_req by 2.4×10²⁵ (U(1)_B), 7.9×10²⁵ (B−L), 1.6×10²⁸ (366 keV) and 8.4×10²⁸ (Higgsino-equivalent G = 4.6×10⁻⁶); range 5.7×10²⁴–8.4×10³² over m = 0.3–10 TeV. Since Y_FI ≫ Y_eq the sector thermalises: freeze-in is not an available mechanism for any LZ-fitting coupling (which is within a factor 2–30 of the thermal coupling). The T_RH at which Boltzmann-suppressed production alone gives Y_req is m/26–m/36 (11 GeV at 300 GeV, 33–37 GeV at 1 TeV, 280–310 GeV at 10 TeV), i.e. T_fo itself — the "low-reheating" scenario merges with freeze-out and does not constitute a distinct option.

### 4.5 Asymmetric DM (`asymmetric_oscillation.json`)
1/δ = 2.2×10⁻²¹ s (300 keV; 2.6 and 1.7×10⁻²¹ s at 250 and 380 keV), corresponding to a radiation temperature of 10⁷ GeV. Any χ–χ̄ asymmetry oscillates away long before freeze-out (T_fo ≈ 10–400 GeV), after which the symmetric population annihilates with the full ⟨σv⟩ — asymmetric DM is excluded for a pseudo-Dirac state with δ ≈ 300 keV unless the splitting is generated after freeze-out (a phase transition below ~40 GeV), a scenario we do not pursue.

### 4.6 Map (Fig. 1)
Panel 1 (Higgsino): log₁₀ N_LZ with the thermal density; N = 0.105/1/3.65 contours; vertical line 1150 GeV. Panels 2–3 (U(1)_B, B−L contact Z′): log₁₀ N_th; the N_th = 1 contour is the thermal fixed point, the band 0.105–3.65 the "thermal-consistent" strip, blue = non-thermal fraction 1 − N_th, red = dilution D = N_th. Panel 4 (dark photon): log₁₀(ceiling/floor); the window closes along the black contour; D_open = (ceiling/floor)^{−8/11} outside. Fig. 2: Ω_th(m) for the Higgsino, N_th(m) at 300/350 keV for the three contact models, and T_RH*(m).

## 5. Robustness and caveats
- Halo: all N_unit use the annual-mean Baxter halo; the Sun-frame or June halos change N_unit by ×1.4–4 at δ ≥ 350 keV (P011/P035), moving the δ-boundaries by ≤ 5–10 keV but not N_th at 300 keV (×1.08).
- Relic: g_*(T) table ±5 % → ≤ 3 % in Ω; Sommerfeld enhancement at freeze-out ignored (≤ 20 % for the dark photon at 1 TeV, P025; a few % for a doublet below 3 TeV) — it lowers the required α_D, g_χ or G_relic and pushes N_th down by the same factor; NLO/electroweak corrections to the ADG formula ±10 %. The Higgsino "thermal band" 1025–1290 GeV is a ±20–25 % statement.
- Mass interpolation near δ_c: factor ≥ 2 uncertainty for δ ≥ 375 keV at non-tabulated masses.
- The Planck floor is P025's saturated-Hulthén one-number criterion (factor ~3, resonance valleys ignored); the ε ≤ 10⁻³ ceiling is recalled and mass-dependent in reality.
- Contact-limit annihilation uses massless final states and the s-wave threshold value (P054 checked the thermal average to 1.6 %); the Z′ width and resonance region are excluded by assumption (m_Z′ ≫ 2m_χ).
- T_RH* assumes instantaneous reheating and a radiation-dominated Universe after T_RH; the BBN floor 4–5 MeV is recalled.
- Freeze-in uses massless SM fermions and Maxwell–Boltzmann statistics (factor ~2), irrelevant against 10²⁵.

## 6. Failed or abandoned approaches
- First version interpolated ln N linearly in ln m at fixed δ: at masses whose ceiling lies below δ the neighbour is zero and the result collapsed to 0 (bogus Higgsino windows 320/360 keV); replaced by the ξ = δ/δ_c(m) scaling.
- The Higgsino was first taken as 0.0777 × N_iso (P007's isoscalar equivalence); against P007's own isovector-dominated grid this overestimates by ×1.3–1.85 (the isovector M-node at 182 keV, P032), so P007's kernels were used directly (live check 1.01–1.07).
- The freeze-in s-integration initially stopped at √s = 2m + 6T and reproduced the massless closed form only to 0.68; extended to 40 T (1.000).
- A single script with live WimPyDD ran 6.8 min; the grids were moved to a cached helper (P075_wimpydd_grid.py).

## 7. Reference list
LZ Collaboration, arXiv:2609.02823 (2026) · D. Tucker-Smith, N. Weiner, PRD 64, 043502 (2001) · K. Griest, D. Seckel, PRD 43, 3191 (1991) · P. Gondolo, G. Gelmini, NPB 360, 145 (1991) · N. Arkani-Hamed, A. Delgado, G. F. Giudice, NPB 741, 108 (2006) · G. Steigman, B. Dasgupta, J. F. Beacom, PRD 86, 023506 (2012) · T. Moroi, L. Randall, NPB 570, 455 (2000) · G. F. Giudice, E. W. Kolb, A. Riotto, PRD 64, 023508 (2001) · L. J. Hall, K. Jedamzik, J. March-Russell, S. M. West, JHEP 03 (2010) 080 · S. Tulin, H.-B. Yu, K. M. Zurek, JCAP 05 (2012) 013 · S. Hannestad, PRD 70, 043506 (2004) · P. F. de Salas et al., PRD 92, 123534 (2015) · corpus: P002, P007, P011, P014, P018, P021, P025, P026, P032, P035, P045, P048, P054.

## 8. Tools and provenance (mirrors provenance/P075.json)
- Agent tools: Read (PAPER_GUIDE; papers P054, P011, P007, P025, P026, P021, P014, P048; P054 script lines 20–320; P025 details §2; two figures), Bash (ledger and corpus table prints, grep of P054/P011/P025 details and lzcommon, WimPyDD timing test, helper run, four main-script runs, output checks), Write (2 scripts, details, JSON, paper), Edit (script fixes, paper trims).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (integrate.solve_ivp Radau, optimize.brentq, special.kve/erf/erfc); pandas 3.0.5; matplotlib 3.11.2 (Agg); WimPyDD 2.0.4 via common/lzcommon.py (wd_halo, wd_hamiltonian, wd_rate, E_R_range_keV, LZ constants).
- Scripts: output/code/P075_wimpydd_grid.py (`.venv/bin/python output/code/P075_wimpydd_grid.py`, 5 min, cached), output/code/P075_thermal_history.py (`.venv/bin/python output/code/P075_thermal_history.py`, 3 min).
- Local inputs: as in §2. Recalled knowledge: 17 items (§2). Datasets: none. Data requests: none. WimPyDD-generated files: none beyond the two cached CSVs in output/work/P075/ (wd_rate/diff_rate writes no response-function files).
