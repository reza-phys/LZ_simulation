# P086 — BBN and CMB constraints on the mediator required for the LZ inelastic cross-section: dark photons, dark Higgs and light Z′ in the early Universe

Simulated arXiv date 2026-09-16 · astro-ph.CO (cross-list hep-ph) · early-Universe cosmology group · category COMP.
Script: `output/code/P086_mediator_cosmology.py` (run from the simulation root with `.venv/bin/python`; runtime ≈ 40–70 s).
Every number quoted below is printed in `output/work/P086/run_log.txt` or stored in the CSV/JSON files listed in §10.
Recalled inputs are flagged [recall: certain | likely | uncertain]; the full list is in `output/provenance/P086.json`.

## 1. Motivation and framework

P011 fits the LZ 248 keV event with pseudo-Dirac dark matter (m_χ ≈ 1 TeV, splitting δ = 250–380 keV) scattering
inelastically through a kinetically mixed dark photon A′: at α_D = 0.1 the required mixing is ε ≈ 1.1×10⁻⁷ (m_A′ = 0.3 GeV) to
8.3×10⁻⁵ (10 GeV) at δ = 300 keV, ε ∝ m_A′² above ≈ 1 GeV. The χ₂-decay requirement (P011, refined by P026 and P058) demands
ε ≳ 5×10⁻⁶, i.e. m_A′ ≳ 2.4–2.6 GeV at δ = 300 keV; P025 finds Planck excludes m_A′ < 9.2 GeV for the relic-normalised
model via Sommerfeld enhancement; P051 and P074 show mediators lighter than ≈ 0.5 GeV reshape the recoil spectrum.
P026 asked what the *excited state* does in the early Universe. Here we ask what the *mediator sector* does:

1. the A′ itself — lifetime, thermalisation with the SM plasma, freeze-in abundance if not thermalised, BBN / CMB-spectral-distortion
   limits on late decays, and N_eff;
2. the dark Higgs h_D that must exist to give the A′ its mass (and, through a charge-2 vev, the Majorana splitting δ) — its mass
   relative to m_A′, its relic abundance, its lifetime, and whether it is a late-decaying-relic problem;
3. whether α_D = 0.01–0.5 makes χ₁ self-interacting at an astrophysically relevant level;
4. the heavy Z′ of P054 (nothing early-Universe applies);
5. a combined (m_A′, ε) map at 1 TeV, α_D = 0.1.

P075 (thermal history) does not exist in the corpus at the time of writing; where a thermal-history reference is needed we cite P054.

Model conventions (P011): Dirac fermion χ with dark charge 1 split into Majorana χ₁, χ₂ by a charge-2 scalar vev; A′ couples to the
off-diagonal current with g_D = √(4πα_D) = 1.121 (α_D = 0.1) and to the SM through ε e Q_f. Dark Higgs: V = λ(|Φ|² − v_D²/2)²,
so m_h = √(2λ) v_D and m_A′ = q_h g_D v_D with q_h the Higgs charge (2 if the same scalar generates δ, 1 for a separate Stueckelberg-free
Higgs). Ratio r ≡ m_{h_D}/m_A′ = √(2λ)/(q_h g_D).

## 2. Thermodynamics: g*(T), g*s(T), H(T), t(T), ΔN_eff

We tabulate g* and g*s on 561 log-spaced temperatures 10⁻¹⁰–10⁴ GeV from exact Bose/Fermi integrals
ρ_i/T⁴ = (g_i/2π²)∫x²√(x²+y²)/(e^{√(x²+y²)}±1)dx, p_i = (g_i/6π²)∫x⁴/√(x²+y²)/(e^{…}±1)dx (y = m_i/T) for γ, e, μ, τ, ν (6), u d s c b t (12 each),
gluons (16), W (6), Z (3), h (1), and the light hadrons π (3), K (4), η (1). Quarks and gluons are weighted by w(T) = ½[1+tanh((T−170 MeV)/25 MeV)]
and hadrons by 1−w [T_QCD recall: likely]. Neutrinos decouple instantaneously at 2 MeV [recall: likely]; below it T_ν/T_γ = (g*s,em(T)/g*s,em(2 MeV))^{1/3}.
H = 1.66 √g* T²/M_Pl (M_Pl = 1.22×10¹⁹ GeV [certain]); t(T) = ∫ dlnT (1 + ⅓ dln g*s/dlnT)/H_full with H_full including Ω_m = 0.315, Ω_Λ = 0.685,
H₀ = 67.4 km/s/Mpc [recall: likely] for the late epochs.

Validation (`run_log.txt` [1]): g*/g*s = 3.371/3.918 at 10 keV (textbook 3.363/3.909), 10.589 at 1 MeV, 17.2 at 100 MeV, 76.35 at 1 GeV,
86.22 at 10 GeV, 106.72 at 1 TeV; t(1 MeV) = 0.745 s, t(0.1 MeV) = 119 s, t(z = 1100) = 1.16×10¹³ s (textbook 1.2×10¹³ s).
Table: `gstar_table.csv`; Fig. 1 left.

ΔN_eff of a relic that decoupled from the SM at T_dec and is still relativistic at the CMB:
ΔN_eff = (g_b + 7/8 g_f)/(7/4) × (g*s(2 MeV)/g*s(T_dec))^{4/3}. Results (`delta_neff_table.csv`, Fig. 1 right): a 1-dof scalar gives 0.57
(T_dec ≤ 10 MeV), 0.31 (100 MeV), 0.060 (200 MeV), 0.042 (1 GeV), 0.027 (1 TeV); a 3-dof vector 1.71/0.93/0.18/0.13/0.080; the full dark
sector (A′ + h_D + χ₁,₂, g_eff = 7.5) 4.3/2.3/0.45/0.32/0.20. Planck's ΔN_eff < 0.3 [recall: likely; N_eff = 2.99 ± 0.17] is satisfied by a
scalar for T_dec > 109 MeV and by a vector for T_dec > 169 MeV. **For the LZ-fit masses (m_A′ ≥ 0.3 GeV, m_{h_D} ≳ 0.5 m_A′) all dark-sector species
are non-relativistic long before neutrino decoupling** (Boltzmann factor e^{−m/T} = 7×10⁻⁶⁶ for 0.3 GeV at 2 MeV): ΔN_eff = 0 unless
something decays *after* ν decoupling (τ ≳ 1 s), which is the same region as the BBN bound below.

## 3. The dark photon

### 3.1 Widths and lifetime
Γ(A′→ℓ⁺ℓ⁻) = (αε²m_A′/3)(1 + 2m_ℓ²/m_A′²)√(1 − 4m_ℓ²/m_A′²) [certain]; Γ_had = (αε²m_A′/3) R(m_A′) with a recalled piecewise
R(s) [recall: uncertain, ×1.5–2 below 2 GeV]: 2π via the ρ Breit–Wigner form factor R_ππ = ¼(1−4m_π²/s)^{3/2} m_ρ⁴/((s−m_ρ²)²+m_ρ²Γ_ρ²)
(peak 5.5; data ≈ 9 — ρ–ω interference and the running width ignored), uds continuum 2 switched on over 1.0–1.5 GeV, charm +4/3 over
3.7–4.2 GeV, bottom +1/3 over 10.5–11 GeV, all ×1.05 for QCD corrections; the narrow ω, φ, J/ψ, Υ poles are ignored (irrelevant for
the lifetime scale). R = 0.02, 0.38, 5.5, 0.46, 2.1, 3.5, 3.85 at m_A′ = 0.3, 0.5, 0.775, 1, 3, 10, 30 GeV.
At ε = 10⁻⁶: τ_A′ = 4.7×10⁻¹⁰, 2.3×10⁻¹⁰, 4.7×10⁻¹¹, 1.1×10⁻¹⁰, 2.2×10⁻¹¹, 4.2×10⁻¹², 1.3×10⁻¹² s for the same masses (τ ∝ ε⁻²);
BR(ee, μμ, ττ, had) = 0.53/0.47/0/0.01 (0.3 GeV), 0.41/0.41/0/0.19 (1 GeV), 0.24/0.24/0/0.51 (3 GeV), 0.15/0.15/0.15/0.54 (10 GeV).
Grid `tau_Aprime_grid_s.csv` (241×241, Fig. 2).

**τ_A′ = 1 s** requires ε = 2.2×10⁻¹¹ (0.3 GeV), 1.05×10⁻¹¹ (1), 4.7×10⁻¹² (3), 2.1×10⁻¹² (10), 1.1×10⁻¹² (30 GeV) (`eps_thermalisation.csv`).
**Along the P011 band (δ = 300 keV):** τ_A′ = 3.9×10⁻⁸ s (0.3 GeV, ε = 1.1×10⁻⁷), 1.45×10⁻¹⁰ (1 GeV), 4.0×10⁻¹³ (3 GeV), 6.1×10⁻¹⁶ (10 GeV),
2.3×10⁻¹⁸ s (30 GeV); the corresponding cosmic time is reached at T = 2.6, 42, 760, 2×10⁴, 3×10⁵ GeV, i.e. the A′ decays and is re-formed in
equilibrium while T ≫ m_A′ and its population simply Boltzmann-suppresses away at T ≲ m_A′/3. The band sits 10⁶–10⁹ above the τ = 1 s line.

### 3.2 Thermalisation with the SM plasma
Inverse decays f f̄ → A′ are the dominant production channel near T ~ m_A′. Criterion A (adopted): the freeze-in yield from inverse decays,
dY/dlnT = γ_ID/(sH) with γ_ID = 3 m_A′² T K₁(m_A′/T) Γ_A′/(2π²), integrated over T = m/60–200m, equals the equilibrium abundance
Y_eq(T = m_A′). Criterion B (cross-check): Γ_A′ K₁/K₂ = H at T = m_A′/3. Results (`eps_thermalisation.csv`):
ε_therm(A) = 2.3×10⁻⁹, 2.0×10⁻⁹, 6.4×10⁻⁹, 1.0×10⁻⁸, 1.6×10⁻⁸, 2.7×10⁻⁸ at m_A′ = 0.1, 0.3, 1, 3, 10, 30 GeV; criterion B gives 1.9, 2.5, 5.7, 7.8, 12, 20 ×10⁻⁹.
This confirms the "ε ≳ 10⁻⁹–10⁻⁸" expectation. The LZ band lies 570× (2.56 GeV), 4.5×10³ (9.2 GeV), 3.4×10⁴ (34.8 GeV) above ε_therm:
**the dark photon of every LZ fit point was in full thermal equilibrium with the SM.** At ε_therm the lifetime is already 5×10⁻⁴–2×10⁻⁹ s,
so a thermalised A′ never decays late. A rough χ–SM kinetic-coupling estimate (Γ(χf→χf) = n_f σ_t, σ_t ≈ 4πα α_D ε²/max(T, m_A′)² [×3])
gives ε ≥ 2.9×10⁻⁷ at T = 1 TeV, 5.6×10⁻⁸ at 40 GeV (χ freeze-out) and 8×10⁻⁹ at 1 GeV: the whole dark sector shares the SM temperature
at χ freeze-out for ε ≳ 10⁻⁷, i.e. everywhere in the band (this underlies P011/P025's relic-density normalisation and our h_D treatment).

### 3.3 Freeze-in population and late decays (ε < ε_therm)
Y_FI ∝ ε² (checked: ratio 4.00 for 2× ε); Y_FI(ε = 10⁻¹¹) = 2.2×10⁻⁷, 1.8×10⁻⁸, 2.5×10⁻⁹ at 0.3, 1, 10 GeV (constant-g* analytic estimate
1.1×10⁻⁷, 3.2×10⁻⁸, 7.2×10⁻⁹). For τ > 1 s the energy m Y released at T(τ) is tested against a recalled envelope of Kawasaki–Kohri–Moroi(–Takaesu)
bounds for a GeV-scale particle with mixed EM/hadronic decays, ξ_max = E_vis Y_X: 10⁻⁵ (1 s), 3×10⁻⁷ (10 s), 10⁻⁸ (10² s), 10⁻⁹ (10³ s),
10⁻¹⁰ (10⁴ s), 3×10⁻¹² (10⁵ s), 3×10⁻¹³ (10⁶ s), 10⁻¹³ (10⁷ s), 2×10⁻¹³ (10⁸ s), 5×10⁻¹³ (10⁹ s), 10⁻¹² (10¹⁰–10¹² s) GeV [recall: uncertain, ×10];
FIRAS μ = 1.4 Δρ/ρ_γ (10⁶ < τ < 10⁹ s) < 9×10⁻⁵ and y = ¼Δρ/ρ_γ (10⁹–10¹³ s) < 1.5×10⁻⁵ [recall: certain], Δρ/ρ_γ = m Y g*s(T_τ)/(1.5 T_τ);
for τ > 10¹³ s the P026 re-expression of the Planck decaying-DM bound (> 0.2 eV per baryon deposited with f_eff = 0.3 excluded) and, for τ > t_U,
f_eff (ρ_X/ρ_DM) Γ < 10⁻²⁵ s⁻¹ [recall: likely, ×3] or Ω_X > Ω_DM.
Scan (`Aprime_late_decay_scan.csv`): 446 points with τ > 1 s below ε_therm, 88 excluded, all at m_A′ ≤ 0.23 GeV and ε ≤ 2.5×10⁻¹¹
(m_A′ = 0.05 GeV: ε ∈ [1.6×10⁻¹⁴, 2.5×10⁻¹¹]; 0.107 GeV: [1.6×10⁻¹⁴, 6.3×10⁻¹²]; 0.23 GeV: [2.5×10⁻¹⁴, 10⁻¹³]); the largest violation is mY/ξ_max = 13.
This is the low-ε, low-mass corner of the "very dark photon" exclusions of Fradette et al. (2014) [recall: uncertain; their band extends to a few GeV
because their D/H and N_eff treatment is stronger than our envelope]. **Nothing here comes within four decades of the LZ band.**

## 4. The dark Higgs

### 4.1 Mass ratio
With α_D = 0.1: q_h = 2 gives r = 0.20, 0.35, 0.63, 0.89, 1.26 for λ = 0.1, 0.3, 1, 2, 4 (r ≥ 1 needs λ ≥ 2.51; r ≥ 2, i.e. prompt h_D → A′A′, needs λ ≥ 10.1,
non-perturbative); q_h = 1 gives r = 0.40, 0.69, 1.26, 1.78, 2.52 (r ≥ 1 at λ ≥ 0.63; r ≥ 2 at λ ≥ 2.5). So for perturbative quartics the dark Higgs is
generically **lighter than the dark photon** when the same scalar generates δ, and h_D → A′A′ is closed.

### 4.2 Relic abundance (Boltzmann solve)
dY/dx = −√(π/45) M_Pl m_h (g*s/√g*) ⟨σv⟩ x⁻² (Y² − Y_eq²), x = m_h/T, Y_eq = (45/4π⁴)(1/g*s) x² K₂(x), integrated in W = lnY with Radau + analytic
Jacobian from x = 2 to 2000 (LSODA failed silently on this stiff problem — see §9). For r > 1: ⟨σv⟩(h h → A′A′) = πα_D²/m_h² × √(1 − 1/r²) [κ = 1, O(1)
normalisation flagged]. For r < 1 the channel is forbidden at zero temperature and proceeds on the Boltzmann tail; detailed balance gives
⟨σv⟩_{hh→A′A′} = ⟨σv⟩_{A′A′→hh} (g_A′/g_h)² (m_A′/m_h)³ e^{−2Δx}, Δ = (m_A′ − m_h)/m_h, with ⟨σv⟩_{A′A′→hh} = πα_D²/m_A′² √(1 − r²).
Other depletion channels are negligible: ε²-suppressed h h → A′ f f̄ (∝ α_D α ε² ~ 10⁻¹³ relative), h_D → χχ closed (m_χ = 1 TeV), 3 → 2 cannibal
rate at x = 20 Γ₃₂ ~ n²λ³/(4π)³m⁵ = 1.1×10⁻²⁷ GeV ≪ H = 1.3×10⁻²¹ GeV (m_h = 1 GeV). Kinetic equilibrium with the SM bath holds through
freeze-out for band values of ε (§3.2), so the h_D temperature is T.

Results (`dark_higgs_relic.csv`, Fig. 3 left), m_h Y_h in GeV (ρ_DM/s = 4.4×10⁻¹⁰ GeV):

| r | m_A′ = 1 GeV | 3 GeV | 10 GeV |
|---|---|---|---|
| 0.5 | 7.8×10⁻⁷ (1.8×10³ × DM) | 2.5×10⁻⁶ (5.6×10³) | 3.8×10⁻⁶ (8.7×10³) |
| 0.6 | 3.9×10⁻⁸ (88) | 1.4×10⁻⁷ (315) | 2.5×10⁻⁷ (572) |
| 0.7 | 1.1×10⁻⁹ (2.4) | 4.4×10⁻⁹ (10) | 9.7×10⁻⁹ (22) |
| 0.8 | 1.5×10⁻¹¹ (0.034) | 7.1×10⁻¹¹ (0.16) | 2.0×10⁻¹⁰ (0.44) |
| 0.9 | 8.3×10⁻¹⁴ (1.9×10⁻⁴) | 4.8×10⁻¹³ (1.1×10⁻³) | 1.7×10⁻¹² (4×10⁻³) |
| 0.95 | 4.1×10⁻¹⁵ | 2.6×10⁻¹⁴ | 1.1×10⁻¹³ |
| 1.05–2.5 | (2.6–6.3)×10⁻¹⁶ | (2.0–3.7)×10⁻¹⁵ | (1.2–2.8)×10⁻¹⁴ |

Check: for r = 1.5, m_A′ = 1 GeV, ⟨σv⟩ = 0.0104 GeV⁻² and Ωh² = 1.07×10⁹ x_f/(√g* M_Pl⟨σv⟩) = 4.2×10⁻⁸ vs ODE 8.1×10⁻⁸ (×2, as expected for the
crude analytic form). A **stable** dark Higgs with r ≤ 0.75 overcloses the Universe; with r = 0.8–0.9 it is 0.02–40 % of the DM.

### 4.3 Lifetime
Channels (all A′-mediated; no direct SM coupling without a Higgs portal):
- **A′ loop, h_D → f f̄ (ε⁴):** derived here by integrating out the two A′ propagators, y_eff ≈ g_D q_h m_A′ (εeQ_f)² m_f/(8π² m_A′²) × F, giving
  Γ_loop = α_D α² ε⁴ Σ_f N_c Q_f⁴ m_f² m_h β_f³/(8π² m_A′²) with F = 1 [derived; ×3 uncertainty on the loop function]. (Batell–Pospelov–Ritz 2009
  give the same parametric form [recall: likely].)
- **Tree h_D → A′*A′* → 4f (ε⁴):** dimensional estimate |M|² ≈ 4g_D² e⁴ ε⁴ m_h⁴/m_A′⁶ × (2+R)²/(1−r²)², Φ₄ = m_h⁴/(2(4π)⁵·3!·2!), Γ = |M|²Φ₄/(2m_h)
  [estimate, ×10]. It dominates over the loop near r → 1 (at m_A′ = 1 GeV, r = 0.9, ε = 10⁻⁶: 1.9×10⁻³¹ vs 6.2×10⁻³⁴ GeV).
- **h_D → A′ f f̄ (ε², r > 1):** Γ ≈ (α_D m_h³/8m_A′²) × αε²(2+R)/(3π) × 0.1(1 − 1/r)³ [crude, ×10; only its smallness matters].
- **h_D → A′A′ (r > 2):** Γ = g_hAA² m_h³(1 − 4x + 12x²)√(1−4x)/(128π m_A′⁴), g_hAA = 2q_h g_D m_A′, x = m_A′²/m_h² [certain].
- **Higgs portal:** Γ = θ² Γ_SM-like(m_h) with Γ_SM-like = Σ_f N_c m_f² m_h β³/(8πv²) (thresholds 2m_K, 2m_D, 2m_B for s, c, b) [certain formula].

Lifetimes without a portal (`run_log.txt` [3b], Fig. 3 right): at m_A′ = 1 GeV, r = 0.9: τ_h = 3.5×10¹⁰, 3.5×10⁶, 3.5×10², 3.5×10⁻², 3.5×10⁻⁶ s for
ε = 10⁻⁷…10⁻³; r = 0.5: 1.5×10¹³ … 1.5×10⁻³ s; r = 1.2: 3×10⁻³ … 3×10⁻¹¹ s; r = 2.5: 2×10⁻²⁴ s. At m_A′ = 10 GeV, r = 0.9: 5.8×10⁸ … 5.8×10⁻⁸ s.
**For r < 1 the dark Higgs lives longer than 1 s over essentially the whole LZ band** (τ_h < 1 s needs ε ≥ 6.2, 4.2, 2.7, 1.6, 1.2 ×10⁻⁵ at
m_A′ = 0.3, 1, 3, 10, 28 GeV for r = 0.9; 3.5, 2.0, 1.2, 0.87, 0.65 ×10⁻⁴ for r = 0.5), whereas for r > 1 it decays in ≲ 10⁻³ s everywhere.

### 4.4 Exclusion in the (m_A′, ε) plane
For each r ∈ {0.5, 0.7, 0.9, 1.2}, on a 61 × 97 grid (m_A′ = 0.05–100 GeV, ε = 10⁻¹⁰–10⁻²), we apply §3.3's criteria to (m_h Y_h, τ_h)
(`dark_higgs_excluded_grid_r*.csv`, `dark_higgs_yield_line_r*.csv`, samples with reasons in `dark_higgs_exclusion_samples.csv`):
- r = 0.5: excluded for all ε ≤ 1.8×10⁻⁴ (0.3 GeV) … 5.6×10⁻⁵ (28 GeV) — BBN, FIRAS μ/y, Planck and overclosure in turn.
- r = 0.7: excluded for ε ≤ 1.8×10⁻⁵ (≤ 3 GeV), 1.2×10⁻⁵ (≥ 10 GeV).
- r = 0.9: only a window, ε ∈ [7×10⁻¹⁰, 4×10⁻⁸] (0.3 GeV), [3×10⁻¹⁰, 7×10⁻⁸] (1 GeV), [10⁻¹⁰, 7–8×10⁻⁷] (3–28 GeV): Planck (τ = 10¹³–10²¹ s) plus BBN/FIRAS
  at the heavier masses where m Y_h reaches 10⁻¹²; the yield is too small for a bound when τ_h < 10⁶ s.
- r = 1.2: nothing excluded.
Intersection with the P011 band (`run_log.txt` [6]; `P086_summary.json` → combined_map): at δ = 300 keV the r = 0.5 dark Higgs removes
m_A′ = 2.56–9.6 GeV (50 % of the band above the χ₂ floor, only 9.2–9.6 GeV of the Planck-allowed segment), r = 0.7 removes 2.56–4.4 GeV (21 %),
r ≥ 0.9 removes nothing. At δ = 250 keV: r = 0.5 removes 5.0–15.6 GeV (29 % of the surviving 9.2–56.9 GeV segment), r = 0.7 removes 5.0–6.6 GeV.
At δ = 350 keV: 0.92–4.7 (r = 0.5) and 0.92–1.9 GeV (r = 0.7), both below the Planck floor. At δ = 380 keV: 0.13–1.4 and 0.13–0.46 GeV.

### 4.5 Higgs-portal escape
The mixing needed for τ_h ≤ 1 s (0.1 s) is θ ≥ 1.0×10⁻⁸ (3.3×10⁻⁸) at m_A′ = 1 GeV, r = 0.9 [1.6×10⁻⁸ for r = 0.5]; 3.4×10⁻⁹ (3 GeV), 1.3×10⁻¹⁰ (10 GeV),
2.8×10⁻¹¹ (30 GeV); at m_A′ = 0.3 GeV: 3.7×10⁻⁸ (r = 0.9) but 5.1×10⁻⁶ (r = 0.5, m_h = 150 MeV < 2m_μ). The implied portal quartic
λ_hφ ≈ θ m_H²/(v v_D) is 1.5×10⁻⁶ (1 GeV) … 1.3×10⁻¹⁰ (30 GeV) — far below any collider or naturalness constraint (θ ≲ 10⁻⁴–10⁻³ for GeV
scalars from B decays and CHARM [recall: uncertain]). **The dark-Higgs problem is therefore a model-building requirement, not an exclusion:**
either m_{h_D} ≳ 0.9 m_A′ (λ ≳ 2 for a charge-2 Higgs at α_D = 0.1; λ ≳ 0.5 for charge 1), or a Higgs-portal quartic λ_hφ ≳ 10⁻⁶–10⁻¹⁰.
Table `dark_higgs_portal_escape.csv`.

## 5. Self-interactions of χ₁
In the Majorana basis the A′ vertex is off-diagonal, so χ₁χ₁ scattering proceeds through the nearly degenerate (χ₁χ₁, χ₂χ₂) system; at cluster
speeds the kinetic energy (5.6 MeV at 1000 km/s) far exceeds 2δ = 0.6 MeV (at 30 km/s, 5 keV, the χ₂χ₂ channel is closed), so we bracket the
transfer cross-section by the average of the attractive and repulsive Yukawa results (Schutz–Slatyer-type treatment [recall: likely]). Classical
formulae (Tulin–Yu–Zurek 2013 [recall: likely]) with β = 2α_D m_A′/(m_χ v²); validity m_χ v/m_A′ > 1, otherwise we report min(classical, 4× s-wave
unitarity) as an upper bound. Results (`sidm_sigma_over_m.csv`): σ_T/m = 5.5×10⁻⁴, 6.4×10⁻⁵, 9.6×10⁻⁷ cm²/g at v = 30 km/s for m_A′ = 0.3, 1, 10 GeV
(quantum regime, upper bounds); 7.6×10⁻⁵, 1.0×10⁻⁵, 2.1×10⁻⁷ at 1000 km/s; 5.3×10⁻⁶, 2.4×10⁻⁶, 8.1×10⁻⁸ at 4000 km/s. Everything is ≥ 1.8×10³
below the Bullet-cluster bound σ/m ≲ 1 cm²/g [recall: likely; Randall et al. 2008 quote 1.25] and ≥ 200 below the 0.1 cm²/g needed for dwarf
cores: **a TeV pseudo-Dirac particle with α_D ≤ 0.5 and a ≥ 0.3 GeV mediator is collisionless for all astrophysical purposes** (σ/m ∝ α_D²·log² at
most, so even α_D = 0.5 stays below 10⁻² cm²/g).

## 6. Heavy Z′ (P054)
For m_Z′ = 0.5 (2) TeV and g_q = 0.017 (0.02), Γ(Z′→qq̄) = 3·6 g_q² m_Z′/(12π) = 0.069 (0.38) GeV, τ = 9.5×10⁻²⁴ (1.7×10⁻²⁴) s: the Z′ decays
at T ~ TeV and its U(1)′ Higgs (v_S = 0.2–0.7 TeV, m_S ~ v_S) decays to Z′Z′ or through Higgs mixing at the weak scale. No BBN, CMB, N_eff or
self-interaction constraint applies; the only cosmology is P054's standard freeze-out.

## 7. Combined map (Fig. 4; `P086_summary.json` → combined_map)
Layers at m_χ = 1 TeV, α_D = 0.1: the P011 band (reconstructed from `output/work/P011/epsilon_required.csv` via ε = √(K m_A′⁴/(α_D S(m_A′))) with the
tabulated rate suppression S at 0.3/1/3/10 GeV and S(0.1 GeV) = 0.034 from P011's text; reproduces P011's 8.65×10⁻⁷ (1 GeV), 8.28×10⁻⁵ (10 GeV),
1.12×10⁻⁷ (0.3 GeV) at δ = 300 keV); the χ₂-decay floor ε ≥ 5.4×10⁻⁶ (P011; P026/P058 give m_A′ ≥ 2.45/2.39 GeV); the P025 Planck floor m_A′ ≥ 9.2 GeV
(relic-normalised α_D = 0.035; at α_D = 0.1 the χ relic is 12 % of Ω_DM and the same criterion gives m_A′ ≳ 3 GeV, while a full-density χ at
α_D = 0.1 would exceed p_ann at S = 1 for every m_A′ ≤ 10 GeV — we keep P025's line as the reference); BaBar ε < 10⁻³ [recall: likely]; LHCb prompt
~3×10⁻⁴ over 10–70 GeV [recall: uncertain]; this work's ε_therm, τ_A′ = 1 s, freeze-in BBN points and the r = 0.7 / 0.9 dark-Higgs regions.

| δ [keV] | χ₂ floor m_A′ ≥ | Planck (P025) | BaBar m_A′ ≤ | surviving m_A′ [GeV] | surviving ε | h_D (r = 0.5 / 0.7 / ≥ 0.9) removes |
|---|---|---|---|---|---|---|
| 250 | 5.0 | 9.2 | 56.9 | 9.2–56.9 | 2.6×10⁻⁵–10⁻³ | 9.2–15.6 / none / none |
| 300 | 2.56 | 9.2 | 34.8 | 9.2–34.8 | 7.0×10⁻⁵–10⁻³ | 9.2–9.6 / none / none |
| 350 | 0.92 | 9.2 | 14.3 | 9.2–14.3 | 4.1×10⁻⁴–10⁻³ | none / none / none |
| 380 | 0.13 | 9.2 | 3.9 | none | — | — |

**Early-Universe physics of the mediator does not remove any of the P011 fit region above the P025 floor** (except 9.2–15.6 GeV at δ = 250 keV for an
r = 0.5 dark Higgs without a portal); the mediator constraints that matter are the already known χ₂-decay floor and the Planck Sommerfeld bound.

## 8. Robustness
- R(s): ×2 in Γ_had changes τ_A′ by ≤ ×1.5 and ε_therm by ≤ ×1.2; irrelevant given the 10⁶–10⁹ margins.
- BBN envelope ×10: the A′ exclusion moves by ×3 in ε and stays ≤ 10⁻¹⁰; the r = 0.9 dark-Higgs window shifts but never reaches the band (ε ≥ 5×10⁻⁶).
- Loop/4f width ×10: ε(τ_h = 1 s) moves by ×1.8; the r = 0.5/0.7 exclusions remain, the surviving Planck-allowed band is unaffected at δ ≥ 300 keV.
- ⟨σv⟩ normalisation κ = 0.3–3: the forbidden yield changes by ≤ ×3 (logarithmic), so the r boundary between "problem" and "no problem" shifts by ≲ 0.03 in r.
- Thermalisation criteria A and B agree within ×1.4.
- α_D = 0.01–0.5: ε_therm and τ_A′ do not depend on α_D; the h_D yields scale as 1/α_D² (log) and r ∝ 1/g_D, so smaller α_D makes r = 1 easier
  (λ ≥ 2g_D²q_h²/… → λ ≥ 0.25 at α_D = 0.01, q_h = 2).

## 9. Failed or abandoned approaches
- Thermalisation defined by max_T[Γ K₁/K₂ / H]: the ratio grows without bound towards T ≪ m (no fermions left to make an A′), giving ε_therm ≈ 10⁻¹⁰
  and a freeze-in yield only 0.3 % of equilibrium at that ε; replaced by criterion A (Y_FI = Y_eq), cross-checked by B.
- LSODA for the dark-Higgs Boltzmann equation returned the initial condition unchanged (Ωh² ≈ 10⁵) without raising an error; replaced by Radau with an
  analytic Jacobian (agrees with the analytic Ωh² estimate to ×2).
- Two implementation errors caught by the validation block: an inverted T_ν/T_γ ratio (g* = 17.9 instead of 3.36 at 0.1 MeV) and a missing GeV⁻¹ → s
  conversion plus a decreasing-abscissa `np.interp` in t(T) (t(1 MeV) = 10²⁴); both fixed, t(1 MeV) = 0.745 s.
- The s-wave unitarity "ceiling" was first applied also in the classical regime, where many partial waves contribute; now used only when m_χv/m_A′ < 1.
- A first r = 1.0 point gave zero phase space in both annihilation formulae (no evolution); dropped in favour of 0.99 and 1.05.

## 10. Files
`P086_summary.json` (all headline numbers), `run_log.txt`, `gstar_table.csv`, `delta_neff_table.csv`, `tau_Aprime_grid_s.csv`, `eps_thermalisation.csv`,
`Aprime_late_decay_scan.csv`, `dark_higgs_relic.csv`, `dark_higgs_yield_line_r{0.5,0.7,0.9,1.2}.csv`, `dark_higgs_excluded_grid_r{…}.csv`,
`dark_higgs_exclusion_samples.csv`, `dark_higgs_portal_escape.csv`, `sidm_sigma_over_m.csv`;
figures: `figures/P086_fig1_gstar_neff.png` (g*, g*s vs T; ΔN_eff vs T_dec for 1, 3, 7.5 dof with the Planck line),
`figures/P086_fig2_tau_map.png` (log τ_A′ in the (m_A′, ε) plane with the τ = 1 s line, ε_therm, the freeze-in BBN points and the P011 bands),
`figures/P086_fig3_dark_higgs.png` (left: m_h Y_h/(ρ_DM/s) vs r; right: τ_{h_D}(ε) for r = 0.7/0.9/1.2 at m_A′ = 1, 3, 10 GeV with the BBN/FIRAS and Planck
windows and the δ = 300 keV LZ band), `figures/P086_fig4_combined_map.png` (all constraints, §7).

## 11. References
1. LZ Collaboration, arXiv:2609.02823 (2026). 2. B. Holdom, PLB 166, 196 (1986). 3. D. Tucker-Smith, N. Weiner, PRD 64, 043502 (2001).
4. B. Batell, M. Pospelov, A. Ritz, PRD 79, 115008 (2009) — secluded U(1) at B-factories, dark-Higgs decays. 5. M. Kawasaki, K. Kohri, T. Moroi, PRD 71, 083502 (2005);
M. Kawasaki, K. Kohri, T. Moroi, Y. Takaesu, PRD 97, 023502 (2018) — BBN bounds on late decays. 6. D. J. Fixsen et al., ApJ 473, 576 (1996) — FIRAS.
7. T. R. Slatyer, C.-L. Wu, PRD 95, 023010 (2017) — CMB bounds on decaying DM. 8. A. Fradette, M. Pospelov, J. Pradler, A. Ritz, PRD 90, 035022 (2014) — very dark photons.
9. S. Tulin, H.-B. Yu, K. M. Zurek, PRD 87, 115007 (2013) — Yukawa self-scattering. 10. R. T. D'Agnolo, J. T. Ruderman, PRL 115, 061301 (2015) — forbidden DM.
11. L. J. Hall, K. Jedamzik, J. March-Russell, S. M. West, JHEP 03 (2010) 080 — freeze-in. 12. Planck Collaboration, A&A 641, A6 (2020).
13. K. Schutz, T. R. Slatyer, JCAP 01 (2015) 021 — self-scattering with an excited state. Corpus: P011, P025, P026, P051, P054, P058, P074.

## 12. Tools and provenance (mirrors `output/provenance/P086.json`)
- Agent tools: Read (PAPER_GUIDE; P011/P025/P026/P051/P054/P058/P074 papers; figures), Bash (ls/grep/head of P011, P025, P051, P026 work files, ENVIRONMENT
  versions, ledger; running the script), Write/Edit (script, details, JSON, paper).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (integrate.quad, integrate.solve_ivp Radau, special.kve, optimize.brentq); pandas 3.0.5; matplotlib 3.11.2;
  common/lzcommon.py (GEV_TO_CM2). Hand derivations: loop-induced h_D width, 4-body estimate, detailed-balance forbidden cross-section, ΔN_eff formula.
- Local inputs: `output/work/P011/epsilon_required.csv` (band, S factors), P011/P025/P026/P051/P054/P058/P074 papers, P011 details §3.5–3.6, P025 details (Planck criterion),
  P051 details §3.3 (accelerator recalls), P026 details (FIRAS/Planck criteria).
- Recalled knowledge: 30 items (constants, particle masses, R(s) shape, T_QCD, ν decoupling, Planck cosmology, ΔN_eff bound, KKM envelope, FIRAS, Slatyer–Wu,
  BaBar/LHCb/beam-dump limits, Bullet cluster, TYZ formulae, BPR width form, Fradette et al. region, portal-scalar limits).
- Datasets: none. Data requests: none. WimPyDD-generated files: none.
