# P080 — Millicharged particles, strongly interacting massive relics and other exotic heavy particles as the source of a single 248 keV xenon recoil

Simulated arXiv date 2026-09-15 · category EXO · hep-ph · exotic-relics phenomenology group.
Script: `output/code/P080_exotic_relics.py` (runs in ≈5 s; log `run_log.txt`; all tables in this directory; figures in `figures/`).

## 1. Motivation and framework

LZ (arXiv:2609.02823) reports one single-scatter (SS) nuclear-recoil-like event at 248 ± 23 ± 23 keV (S1c = 540.1 phd, S2c = 9268 phd) in 2.84 t·yr with no Skin/OD activity and no other NR-like event in the WIMP-search ROI (S1c 3–600 phd). The corpus has established (P003, P030, P040) that for any velocity-independent contact interaction the nuclear response alone forces ≥ 350 (Helm) – 1184 (shell) events in 5.4–55 keV per 200–270 keV event, and P016 finds LZ's likelihood tolerates only 1.6–2.2 fitted low-energy events. Here we ask whether a heavy relic *other than* a WIMP-like particle can evade this: (a) millicharged particles (MCP), (b) strongly interacting massive particles and composite relics, (c) heavy charged relics, monopoles, Planck-scale DM, (d) inelastic composites (dark nuclei).

Inputs from the LZ paper: exposure 220 live days × 4.71 t; FV z-boundaries 9.0 cm above the cathode and 12.8 cm below the gate; WS ROI S1c 3–600 phd; NR efficiency 96 % between 14 and 250 keV with 50 % points at 5.4 and 269.9 keV; the high-energy sideband HE SB (800 < S1c < 1700 phd, log10 S2c < 4.3) with **0 observed science events in the 4.7 t FV** (Supplemental Table, wall 0.1 simulated / 0 observed, RFR 0.5 / 0); veto definitions; the statement that MS events fail the SS classification; 7 t active LXe. Recalled: TPC 1.456 m diameter × 1.456 m drift (LZ instrument paper, likely).

Shared code: `lzcommon` (`helm_F2`, `eta0`, `vmin_kms`, `m_chi_min_gev`, `delta_max_kev`, `dRdE_SI`, `nest_nr_yields`, constants); P017's WimPyDD shell-model M response table (`output/work/P017/P017_response_curves.npz`, natural Xe) for the shell F²; P003's efficiency model (plateau 0.96, erf roll-offs σ = 3.4 / 8.0 keV at 5.4 / 269.9 keV).

### Geometry (computed; `P080_results.json: geometry`)
- LXe density 2.9 g/cm³ (lzcommon NEST density) → n_Xe = 1.329×10²² cm⁻³; N_T(FV) = 2.159×10²⁸ nuclei.
- FV: h = 145.6 − 9.0 − 12.8 = 123.8 cm, V = 4.71×10⁶/2.9 = 1.624×10⁶ cm³ → r_FV = 64.6 cm; surface S_FV = 7.65×10⁴ cm²; mean isotropic chord L̄ = 4V/S = 84.9 cm.
- Active TPC: r = 72.8, h = 145.6 cm → 7.03 t, S_TPC = 9.99×10⁴ cm², L̄_TPC = 97.1 cm (chord MC: mean 97.0, median 103, 10–90 % 24–155 cm).
- Live time 1.9008×10⁷ s.

### Validation checks
- Re-implementation of the contact-SI rate reproduces `lz.dRdE_SI` to 1.0000 at 10/50/100/248 keV (unit check of the MCP machinery).
- F²(30)/F²(248) = 1249.3 (Helm, A²-weighted natural Xe) and 6809.4 (shell) — the P040 values (1249/6809) are confirmed.
- N_lo for contact O1 (Helm, 1 TeV, SHM) with our grids = 820.0 (P040: 820).
- Chord MC mean 96.98 cm vs analytic 97.07 cm.

## 2. (a) Millicharged particles

### 2.1 Halo MCPs (m = 60 GeV – 10 TeV, charge εe)
Rutherford scattering of a heavy slow projectile on a nucleus of charge Z (screening irrelevant for E_R > keV, q ≫ ħ/a_TF):

  dσ/dE_R = 2π Z² ε² α² (ħc)² F²(E_R) / (m_N v² E_R²),

so that dR/dE_R = N_T (ρ/m) [2πZ²ε²α²(ħc)²c²/(m_N E_R²)] F²(E_R) η(v_min(E_R)), with η = ⟨1/v⟩ from `lz.eta0` (Baxter SHM, v_E = 250.6 km/s, v₀ = 238, v_esc = 544), summed over isotopes (Z = 54 for all). The spectrum is the contact O1 spectrum times 1/E_R²: it *favours* low energies by an extra factor (248/E)².

Elastic kinematics: `lz.m_chi_min_gev(248)` = 74.9 GeV (63.3 GeV for 200 keV): MCPs below ≈ 75 GeV cannot make the event at all (m = 60 GeV row: R_win = 0).

Table 1 (`P080_mcp_halo.csv`): N_lo ≡ R(5.4–55)/R(200–270) with the LZ efficiency, and ε₁ giving one event in 200–270 keV in 2.84 t·yr.

| m [GeV] | N_lo Helm | N_lo shell | ε₁ Helm | ε₁ shell | low-E events at ε₁ (Helm) |
|---|---|---|---|---|---|
| 100 | 6.58×10⁶ | 2.14×10⁷ | 3.07×10⁻⁸ | 5.58×10⁻⁸ | 6.6×10⁶ |
| 200 | 6.86×10⁵ | 2.27×10⁶ | 1.35×10⁻⁸ | 2.47×10⁻⁸ | 6.9×10⁵ |
| 400 | 3.25×10⁵ | 1.08×10⁶ | 1.29×10⁻⁸ | 2.38×10⁻⁸ | 3.3×10⁵ |
| 1000 | 2.29×10⁵ | 7.67×10⁵ | 1.70×10⁻⁸ | 3.14×10⁻⁸ | 2.3×10⁵ |
| 4000 | 1.97×10⁵ | 6.62×10⁵ | 3.14×10⁻⁸ | 5.80×10⁻⁸ | 2.0×10⁵ |
| 10000 | 1.92×10⁵ | 6.43×10⁵ | 4.90×10⁻⁸ | 9.04×10⁻⁸ | 1.9×10⁵ |

The MCP N_lo is ≈ 280× the contact-O1 value (820), as expected from ⟨(248/E)²⟩ over the low window. At ε₁(1 TeV) = 1.7×10⁻⁸ the MCP's 5.4–55 keV rate equals that of a contact-SI WIMP with σ_SI = 4.5×10⁻⁴² cm² (`mcp_equivalent_sigma_SI_1TeV_cm2`), i.e. ≈ 10⁵ above LZ's own 2024 TeV limit (recalled ≈ 10⁻⁴⁷ cm², likely). Verdict: excluded by the empty low-energy NR band by five orders of magnitude, independent of any external MCP bound.

### 2.2 Stopping, shielding and magnetic deflection at ε₁ (`P080_mcp_stopping.csv`)
Kinetic energy at 700 km/s: ½mv² = 2.73 keV × (m/GeV) → 0.27 MeV (100 GeV), 2.7 MeV (1 TeV), 27 MeV (10 TeV).
- Nuclear (Rutherford) energy loss in the overburden, dE/dx = n Σ_i 2πZ_i²ε²α²(ħc)²/(m_i v²) ln(E_max,i/E_min,i), E_max = 2μ²v²/m_i, E_min = (ħ/a_TF)²/2m_i with a_TF = 0.885 a₀ Z^{-1/3} (recalled, likely); column X_rock = 1478 m × 2.7 g/cm³ = 3.99×10⁵ g/cm² (SURF depth and standard rock recalled, likely; crustal O/Si/Al/Fe/Ca/K/Na/Mg mass fractions recalled, likely) plus 1033 g/cm² of air.
- Electronic loss estimated with Lindhard–Scharff scaled by ε² for a point charge ε ≪ 1 (recalled formula; the ε² scaling of a fractional charge is an estimate, uncertain).
- Results at ε₁: fractional energy loss 2.7×10⁻⁷ (100 GeV), 8.4×10⁻⁹ (1 TeV), 7.0×10⁻⁹ (10 TeV): the overburden is transparent. It would stop the particles (ΔE = KE) only at ε = 5.9×10⁻⁵ / 1.9×10⁻⁴ / 5.9×10⁻⁴.
- Larmor radii at ε₁: Earth field 3×10⁻⁵ T → 8.5×10⁸ km (100 GeV) and larger; heliospheric 5 nT → 3.4×10⁴ AU (100 GeV), 6.1×10⁵ AU (1 TeV). Heliospheric evacuation (r_L = 1 AU; cf. recalled Chuzhoy–Kolb 2009, Dunsky–Hall–Harigaya 2019) requires ε ≥ 1.0×10⁻³ (100 GeV), 1.0×10⁻² (1 TeV), 0.10 (10 TeV). None of these mechanisms is active at ε ~ 10⁻⁸: the halo-MCP route fails purely on the spectrum.

### 2.3 Relativistic MCPs (cosmic-ray-produced, supernova-accelerated)
For β → 1 and a heavy projectile the Rutherford spectrum is dσ/dE_R = 2πZ²ε²α²(ħc)²F²/(m_N β² E_R²): the same F²/E_R² shape without the velocity integral. Results (`mcp_relativistic`):
- N_lo = 9.64×10⁴ (Helm) / 3.22×10⁵ (shell) — the beam version of P040's floor, ×275 larger because of 1/E_R².
- σ(200–270 keV, with efficiency) = 2.23×10⁻³⁰ ε² cm²; σ(5.4–55) = 2.15×10⁻²⁵ ε² cm². One event needs Φ ε² = 1.09×10⁻⁶ cm⁻² s⁻¹ (4π-integrated flux through the FV).
- MCP–electron scattering at equal deposited energy: dσ_N/dσ_e = Z m_e/m_N = 2.25×10⁻⁴ → 4.4×10³ electron recoils per nuclear recoil at the same energy (and ∝ 1/E² more at low energy): the population would appear in the ER band, not silently.
- Continuous ionisation: ε² × 1.25 MeV/(g/cm²) (recalled MIP value, likely) × 2.9 g/cm³ × 97 cm = 352 MeV × ε² along the chord → a single-site topology needs ε < 1.7×10⁻³ (< 1 keV along the track); overburden loss ≈ 1000 GeV × ε², irrelevant for ε ≲ 10⁻².
Verdict: excluded by N_lo ≈ 10⁵ regardless of flux normalisation; recalled cosmic-ray MCP bounds (Plestid et al. 2020, ArgoNeuT, SENSEI; uncertain) are not needed.

## 3. (b) SIMPs and composite relics

### 3.1 Mean free path and single-scatter probability
λ = 1/(n_Xe σ_χXe). λ = L̄_TPC = 97.1 cm ↔ σ = 7.75×10⁻²⁵ cm²; λ = 1.456 m ↔ 5.17×10⁻²⁵ cm². In the heavy limit the coherent-contact relation σ_χXe = σ_n A⁴ gives σ_n ≈ 2.6×10⁻³³ cm² — the classic SIMP window region.

Chord Monte Carlo (2×10⁵ isotropic chords through the TPC cylinder, `P080_simp_P1.csv`, Fig. 3): P(exactly one scatter) = ⟨μe^{-μ}⟩ with μ = nσL peaks at 0.311 for σ = 6.8×10⁻²⁵ cm²; at σ_λ, P1 = 0.310, P(≥2) = 0.270, P(0) = 0.42. Any traversal with ≥ 2 scatters above ~1 keV produces ≥ 2 S2 pulses (the 700 km/s transit takes 2 μs, so all S1 light merges into one prompt S1, but S2s separated by Δz > ~3 mm are resolved in drift time; scatters with the same z but Δxy > few cm are resolved by the S2 hit pattern) and fails LZ's SS classification. We take a visible-scatter threshold of 1 keV (≈ 8 electrons at LZ NR yields); the flat-spectrum fraction below 1 keV is only 0.18 %, so the threshold choice (0.5–3 keV) is immaterial.

### 3.2 Recoil spectrum of a heavy relic at halo velocities
For m ≫ m_N and isotropic CM scattering, E_R is flat up to E_max = 2m_N v² (1335 keV at 700 km/s, 1719 keV at v_max = 794.6 km/s); averaged over the SHM, dR/dE ∝ η(v_min) with v_min = √(E/2m_N) (302 km/s at 248 keV). Two limiting couplings: (i) pointlike coupling to nucleons → × F²_Xe (Helm or shell); (ii) opaque composite (hard sphere with R_χ ≳ nuclear size scattering the nucleus as a whole; classical regime since 1/q = 0.8 fm ≪ R_χ) → no form factor. The HE SB acceptance is mapped to NR energy with the LZ-tuned NEST yields (`lz.nest_nr_yields`, S1c = g₁N_ph; nestpy warns that it is beyond the AmBe calibration endpoint above 300 keV): S1c = 600/800/1700 phd ↔ 292/375/732 keV; S1c(248 keV) = 498 phd.

Table 2 (`P080_simp_spectral_fractions.csv`): fractions of visible scatters and companions per 200–270 keV event.

| spectrum | f_win | f_lo | f(55–200) | f_HESB | N_lo | N(55–200)/win | N_HESB/win | N(>270)/win | P(0 in 55–200 ∪ HE SB) |
|---|---|---|---|---|---|---|---|---|---|
| flat (opaque composite) | 0.107 | 0.153 | 0.338 | 0.176 | 1.43 | 3.17 | 1.65 | 3.30 | 0.0081 |
| contact × Helm | 9.8×10⁻⁴ | 0.664 | 0.026 | 1.7×10⁻⁴ | 676 | 26.8 | 0.17 | 0.39 | 1.9×10⁻¹² |
| contact × shell | 2.9×10⁻⁴ | 0.664 | 0.023 | 1.4×10⁻⁴ | 2270 | 79.9 | 0.48 | 1.00 | 1.2×10⁻³⁵ |

The pointlike SIMP inherits the contact-O1 problem (N_lo = 676–2270 in the heavy limit, plus 27–80 events in 55–200 keV). The opaque composite is the one candidate for which 200–270 keV is *not* disfavoured relative to 5.4–55 keV (N_lo = 1.43, within P016's fitted tolerance and P003's 3–5). But its spectrum is broad: per window event it predicts 3.2 single-scatter NRs in 55–200 keV (an NR-band region LZ finds empty; P016's digitised 125–200 keV bin has b = 0.02, n = 0) and 1.65 in the HE SB (0 observed in the 4.7 t FV). Joint Poisson probability of no companion: P(0) = e^{−4.81} = 0.0081 (2.4σ one-sided). This is a spectral statement, independent of σ and m.

### 3.3 Multiple-scatter companions (`P080_simp_multiscatter.csv`)
Multi-scatter traversals per SS window event = P(≥2)/(P1 f_win): 8.2 (flat) / 888 (contact×Helm) at σ_λ; 0.79 / 86 at 10⁻²⁵; 0.08 / 8.3 at 10⁻²⁶ cm². Fewer than one MS companion (flat) requires σ < 1.23×10⁻²⁵ cm² (λ > 6.1 m), where P1 = μ ≈ 0.16. LZ analysed its multiple-scatter NR sample to cross-check the SS neutron estimate (0.02 ± 0.02), so a handful of ~MeV multi-vertex NR events would have been conspicuous; we flag this as a check rather than a measured constraint.

### 3.4 Flux and mass for one event (`P080_simp_mass_for_one_event.csv`)
N = (ρ/m) ⟨v⟩ (S_TPC/4) T P1(σ) f_win (V_FV/V_TPC), ⟨v⟩ = 354 km/s (Baxter SHM, lab frame; computed). Setting N = 1:
- opaque composite: m = 1.12×10¹⁷ GeV at σ_λ (45 TPC traversals in 220 d), 4.6×10¹⁵ GeV at 10⁻²⁶ cm², 1.1×10¹⁶ at 10⁻²³;
- pointlike (Helm): 1.03×10¹⁵ GeV at σ_λ (4.9×10³ traversals); shell 3.1×10¹⁴.
The N = 1 contour in the (m, σ) plane (Fig. 2) is a closed loop: above m = 1.1×10¹⁷ GeV (flat) or 1.0×10¹⁵ GeV (pointlike) not even the optimal σ yields one event; for m → 3.9×10¹⁸ GeV fewer than one particle traverses the FV per run (Sec. 4).

### 3.5 Overburden (`P080_simp_overburden.csv`)
At σ_λ, for coherent A⁴ scaling to rock nuclei (σ_A = σ_Xe (A/A_Xe)² (μ_A/μ_Xe)²): 14.6 scatters in 1478 m of rock, each removing on average 2μ²/(m m_A) of the kinetic energy → fractional loss 1.15 (1 TeV), 0.108 (10 TeV), 1.1×10⁻² (100 TeV): transparent (< 30 % loss) for m > 3.7×10³ GeV. For geometric (σ-constant) scaling: 8.8×10³ scatters, transparent for m > 1.16×10⁶ GeV. Both are far below the 10¹⁵–10¹⁷ GeV masses required, so the SURF overburden is not what excludes these relics. Note the ~10⁴ rock scatters for the opaque case leave the direction essentially unchanged (each deflection ~ m_A/m).

### 3.6 Continuous-track objects (`continuous_tracks` in results.json)
- Nuclearites / strange-quark nuggets (De Rújula–Glashow: σ = π(10⁻⁸ cm)² for M < 1.5 ng, recalled likely): dE/dx = σρv² = 2.8×10⁹ keV/cm = 2.8 TeV/cm, λ = 2.4×10⁻⁷ cm — a continuous track of 270 TeV across the TPC. Q-balls (electrically neutral, σ ~ 10⁻¹⁶–10⁻¹² cm² via nucleon absorption; recalled) are similar or larger.
- Opaque composite of R = 4 fm (σ = 5.0×10⁻²⁵): 4.5 keV/cm mean → 433 keV per chord, i.e. one ~400 keV scatter per traversal on average — the case of Sec. 3.2.
- Unit-charge CHAMP at 700 km/s: Lindhard–Scharff electronic stopping (recalled formula, likely) 74 MeV/cm in LXe → 7.1 GeV along a 97 cm chord; overburden loss 52 TeV → survives with < 30 % loss only for m > 6.3×10¹⁰ GeV. Bound CHAMP–nucleus atoms and the same Rutherford N_lo argument with ε = 1 add to this.
- Dirac monopole at β = 2.3×10⁻³: Z_eff = gβ/e = β/(2α) = 0.16 → 1.9 MeV/cm (Lindhard scaling) to 12 MeV/cm (Ahlen–Kinoshita-type estimate with L = 3; recalled, uncertain) → 0.2–1.2 GeV along the chord.
All deposit ≫ 248 keV continuously along a track: incompatible with the point-like S2 (consistent with LZ's alpha-derived single-site template) and with the ~500 phd S1.

## 4. (c) Flux ceiling from N = 1 and the ρ/m bound (`flux_ceiling`, `P080_flux_mass.csv`)
For an isotropic flux F (cm⁻² s⁻¹ sr⁻¹) the entry rate through a convex surface S is πFS. One expected event in 220 d through the FV surface (7.65×10⁴ cm²) gives F = 2.19×10⁻¹³ cm⁻² s⁻¹ sr⁻¹ (1.68×10⁻¹³ for the TPC surface; 90 % Feldman–Cousins upper limit for n = 1, b = 0 is 4.36× higher, 9.5×10⁻¹³). This is 219× the Parker bound (10⁻¹⁵, recalled certain) and 1564× the MACRO monopole limit (1.4×10⁻¹⁶, recalled likely): a monopole origin is excluded by flux alone, before its dE/dx. For DM with ρ = 0.3 GeV/cm³ and ⟨v⟩ = 354 km/s, Φ_4π = 4πF = 2.75×10⁻¹² cm⁻² s⁻¹ = (ρ/m)⟨v⟩ p gives

  m ≤ 3.86×10¹⁸ GeV × p,

p being the interaction probability per traversal. A Planck-mass relic (1.22×10¹⁹ GeV) traverses the FV 0.32 times per run. Required (p, σ_χXe) per mass: 10¹⁰ GeV → p = 2.6×10⁻⁹ (σ = 2.3×10⁻³³ cm²), 10¹⁴ → 2.6×10⁻⁵ (2.3×10⁻²⁹), 10¹⁶ → 2.6×10⁻³ (2.3×10⁻²⁷), 10¹⁸ → 0.26 (2.7×10⁻²⁵). Whatever the mass, the spectral requirement of Sec. 3.2 remains.

## 5. (d) Inelastic composites (dark nuclei) — bridge to the IDM papers (`P080_inelastic_composites.csv`)
δ_max(248 keV) at v_max = 794.6 km/s (`lz.delta_max_kev`): 101 keV (100 GeV), 375 (1 TeV; P030 quotes 387 with its 16 June halo), 402 (10 TeV), 405.0 keV for m → ∞ (423 keV for a 270 keV recoil on ¹³⁶Xe). A heavy composite therefore gains only 30 keV of δ_max over a 1 TeV WIMP. The transition form factor is unsuppressed only if qR_D ≲ 1 with q = 246 MeV, i.e. R_D ≲ 0.80 fm — a dark nucleus must be more compact than a nucleon, or pay (qR_D)^{-2}-type suppression. The de-excitation constraint τ(χ₂→χ₁γ) > 0.43 μs (P042) and the flux ceiling m ≤ 3.9×10¹⁸ p GeV apply unchanged; the low-energy population is removed by δ ≳ 300 keV exactly as in P007/P021, so a dark nucleus is viable only as a UV completion of that window (cf. P023's hyperfine composite).

## 6. Summary table (`P080_summary_table.csv`)

| candidate | required | existing bound (recalled, flagged) | verdict |
|---|---|---|---|
| halo MCP, 75 GeV–10 TeV | ε = (1.3–4.9)×10⁻⁸ | none needed; equivalent σ_SI 4.5×10⁻⁴² ≫ LZ 2024 (~10⁻⁴⁷, likely) | N_lo = 2×10⁵–2×10⁷: excluded |
| relativistic MCP (CR/SN) | Φε² = 1.1×10⁻⁶ cm⁻² s⁻¹ | Plestid et al. 2020, ArgoNeuT, SENSEI (uncertain) | N_lo ≈ 10⁵, 4.4×10³ ER per NR: excluded |
| pointlike SIMP, one scatter/traversal | σ_χXe ≈ 8×10⁻²⁵ cm² (σ_n ≈ 3×10⁻³³), m ≈ 10¹⁵ GeV | MJD, CRESST/EDELWEISS surface, XENON1T/DEAP-3600 multiscatter (uncertain) | N_lo = 676–2270, 27–80 events in 55–200 keV, 888 MS traversals: excluded |
| opaque composite (flat spectrum) | σ ≈ 8×10⁻²⁵ cm² (R ≈ 4 fm), m ≈ 10¹⁷ GeV (10¹⁶ at 10⁻²⁶ cm²) | DEAP-3600/MJD multiscatter, MICA, rocky planets (uncertain) | N_lo = 1.4 but 3.2 + 1.65 companions in 55–200 keV and HE SB, 0 observed: p = 0.008; 8 MS traversals unless σ < 1.2×10⁻²⁵ |
| nuclearites, Q-balls (σ ≥ 10⁻¹⁶ cm²) | — | MACRO, SLIM, MICA (uncertain) | 2.8 TeV/cm continuous track: not a point-like NR |
| CHAMP (charge e), monopole | — | MACRO/Parker; CHAMP stopping (uncertain) | 74 MeV/cm and 2–12 MeV/cm tracks; CHAMP needs m > 6×10¹⁰ GeV to arrive; monopole flux 200–1600× above Parker/MACRO |
| superheavy / Planck-scale DM | F = 2.2×10⁻¹³ cm⁻² s⁻¹ sr⁻¹; m ≤ 3.9×10¹⁸ p GeV | — | flux-limited; Planck mass: 0.32 traversals per run |
| inelastic composite (dark nucleus) | δ ≤ 405 keV, R_D ≲ 0.8 fm, m ≤ 3.9×10¹⁸ p GeV | P042 τ > 0.43 μs; P007/P021 window | viable only as the IDM window with a composite UV completion |

## 7. Figures
- Fig. 1 `figures/P080_fig1_spectra.png`: recoil spectra normalised to one 200–270 keV event: halo MCP (∝ F²η/E²), contact SI (Helm), opaque heavy composite (flat × η), pointlike heavy SIMP (flat × F² × η); shaded 5.4–55 keV ROI, 200–270 keV window and the HE SB (375–732 keV).
- Fig. 2 `figures/P080_fig2_simp_plane.png`: (m, σ_χXe) plane: N = 1 loops for the opaque and pointlike spectra, λ = 97 cm line, overburden opacity thresholds (3.7×10³ GeV coherent, 1.2×10⁶ GeV geometric), the < 1 traversal/run line at 3.9×10¹⁸ GeV, and a schematic grey band for recalled exclusions (uncertain).
- Fig. 3 `figures/P080_fig3_P1.png`: P(0), P(1), P(≥2) scatters per isotropic TPC traversal vs σ_χXe.

## 8. Robustness and failed approaches
- Efficiency variants (σ_hi 4–15 keV, P003) move N_lo by ≤ 9 % (P040) — irrelevant at N_lo ≥ 10⁵ and at the 1.4 of the flat case (which is dominated by the window's own acceptance).
- The flat-spectrum companion counts depend on the SHM through η(v_min): at fixed v = 700 km/s the spectrum is exactly flat to 1335 keV and N(55–200)/win = 145/70 = 2.1, N_HESB/win = 357/70·(0.96/0.96) = 5.1 — larger, not smaller; the SHM average is the conservative choice.
- Monopole stopping: two estimates differ by ×6; the conclusion (≫ 248 keV over the chord) is insensitive.
- Abandoned: a "SIMP window" in which the overburden filters the velocity distribution to favour high-energy recoils — for m ≳ 10⁴ GeV the rock is transparent at σ_λ and for m ≲ 10³ GeV the particles are thermalised, not accelerated; no regime enhances 248 keV relative to 5.4–55 keV.
- Abandoned: using the assignment's "5.5 t, 1.3 m × 1.3 m" fiducial; the paper's 4.71 t with the stated z-cuts gives 1.29 m × 1.24 m (used).

## 9. References
LZ Collaboration, arXiv:2609.02823 (2026); D. S. Akerib et al. (LZ), NIM A 953, 163047 (2020); A. De Rújula, S. L. Glashow, Nature 312, 734 (1984); S. P. Ahlen, K. Kinoshita, PRD 26, 2347 (1982); G. D. Starkman, A. Gould, R. Esmailzadeh, S. Dimopoulos, PRD 41, 3594 (1990); J. Bramante, B. Broerman, R. F. Lang, N. Raj, PRD 98, 083516 (2018); L. Chuzhoy, E. W. Kolb, JCAP 07 (2009) 014; R. Plestid et al., PRD 102, 115032 (2020); D. Dunsky, L. J. Hall, K. Harigaya, JCAP 07 (2019) 015; J. Lindhard, M. Scharff, Phys. Rev. 124, 128 (1961); corpus P003, P007, P016, P017, P021, P023, P030, P036, P040, P041, P042.

## 10. Tools and provenance (mirrors `output/provenance/P080.json`)
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (special.erf); pandas 3.0.5; matplotlib 3.11.2 (Agg); nestpy 2.1.1 via `lz.nest_nr_yields` (HE SB energy map); common/lzcommon.py (LZ dict, XE_ISOTOPES, helm_F2, eta0, vmin_kms, m_chi_min_gev, delta_max_kev, vmax_kms, v_earth_kms, mu_red, dRdE_SI, nest_nr_yields, constants); WimPyDD 2.0.4 not re-run — P017's cached shell-model M table reused.
- Script and command: `output/code/P080_exotic_relics.py`; `.venv/bin/python output/code/P080_exotic_relics.py > output/work/P080/run_log.txt 2>&1`.
- Local inputs: fulltext.tex (detector §, Data Analysis l.113–147, FV l.136–139, event l.166, HE SB l.316–317/721–731 and Supplemental MSSI table l.767–778, waveform l.695, MS neutrons l.791); PAPER_GUIDE; results_ledger.csv; P040/P030/P036/P041/P042/P016 papers; P040 tables and provenance; P017 `P017_response_curves.npz`, `P017_M_ratios.csv`; P003 script (efficiency constants); dossier (hypothesis H); ENVIRONMENT_versions.txt.
- Recalled knowledge (16 items, see JSON): Rutherford/Coulomb cross-section (certain); fine-structure constant (certain); TPC 1.456 m × 1.456 m (likely); SURF 1478 m, standard rock 2.7 g/cm³, crustal composition (likely); atmosphere 1033 g/cm² (certain); Thomas–Fermi screening length (likely); Lindhard–Scharff stopping (likely; ε² scaling uncertain); MIP dE/dx 1.25 MeV cm²/g in Xe (likely); De Rújula–Glashow nuclearite σ (likely); monopole Z_eff = β/2α and Ahlen–Kinoshita-type estimate (uncertain); Parker bound 10⁻¹⁵ (certain) and MACRO 1.4×10⁻¹⁶ cm⁻² s⁻¹ sr⁻¹ (likely); Feldman–Cousins 4.36 (certain); heliospheric field 5 nT, Earth field 3×10⁻⁵ T (likely); LZ 2024 SI limit ~10⁻⁴⁷ cm² at TeV (likely); SIMP/heavy-DM exclusion regions (XENON1T/LZ single-scatter, CDMS-I/XQC, CRESST/EDELWEISS surface, MJD, DEAP-3600, MICA, rocky planets; uncertain, schematic only); cosmic-ray MCP bounds (uncertain, not used quantitatively).
- Datasets: none. Data requests: none. WimPyDD-generated files: none.
