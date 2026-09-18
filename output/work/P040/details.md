# P040 — Cosmic-ray-boosted and otherwise fast light dark matter at the extended LZ window: research record

Simulated date 2026-09-09. Author profile: boosted-dark-matter phenomenologists. Category EXO (hep-ph, cross-list astro-ph.HE).
Script: `output/code/P040_boosted_dm.py` (run from the simulation root with `.venv/bin/python`, 7 s). Log: `run_log.txt`. All numbers below are read from the script output or the CSV/JSON files in this directory.

## 1. Motivation and framework

The LZ event (arXiv:2609.02823) is a lone 248 ± 23 ± 23 keV nuclear recoil in 2.84 t·yr with no companion population in 5.4–55 keV (the 2024 low-energy search used the same 220 live days; LZ Data Analysis paragraph, Table I). P003 turned this into the *lone-event criterion*: any explanation must give N_lo = R(5.4–55 keV)/R(200–270 keV) ≲ 3–5 (recalled tolerance, uncertain; P016 showed the profile fit actually tolerates 1.6–2.2 fitted low-energy events). Halo WIMPs with spin-independent (SI) couplings fail this by three orders of magnitude (O1: 2752, P003), because the halo velocity distribution piles recoils at low energy *and* the coherent form factor collapses at q ≈ 246 MeV (P017: shell-model F²(248) = 2.9×10⁻⁵, Helm 1.6×10⁻⁴; the event is 19 keV below the second M-response node).

The dossier's hypothesis K asks whether a *fast* light particle escapes this: a sub-GeV DM particle up-scattered by Galactic cosmic rays (CRDM; Bringmann & Pospelov 2019), blazar- or decay-boosted DM, arriving with β ~ 0.1–1, can deposit hundreds of keV on xenon even at m_χ ~ MeV, and its recoil spectrum is flat in E_R (isotropic CM scattering) rather than exponentially falling. The intuition to test is that a hard, flat recoil spectrum has N_lo ≈ (55 − 5.4)/70 ≈ 0.7 times a form-factor ratio. This paper computes that ratio properly, asks which incident spectrum minimises N_lo, normalises the CRDM flux, and checks Earth attenuation. Also: whether a fast (β ≈ 0.01) sub-population of TeV DM changes the SI conclusion.

## 2. Kinematics (exact two-body)

For projectile mass m, kinetic energy T, target mass m_N at rest: s = m² + m_N² + 2 m_N (T + m) = (m + m_N)² + 2 m_N T; lab momentum p² = T² + 2 m T; CM momentum p* = p m_N/√s; maximum momentum transfer q_max = 2p*, so

  E_R,max = q_max²/(2 m_N) = 2 m_N p²/s = 2 m_N (T² + 2 m T) / [(m + m_N)² + 2 m_N T]   (certain; derived here).

Inverting for T at fixed E_R gives a quadratic: 2 m_N T² + (4 m m_N − 2 m_N E_R) T − E_R (m + m_N)² = 0. Non-relativistic check: m ≫ m_N gives E_R,max → 2 μ² v²/m_N (Table 5 agrees to 0.04 %). Neutron check: T_min(248 keV, m = 939.6 MeV) = 8.17 MeV vs P013's 8.20 MeV (per-isotope treatment there).

Table 1 (`P040_kinematics.csv`; natural Xe, m_N = 122.4 GeV):

| m_χ | T_min(200) | T_min(248) | T_min(270) | β_min(248) | p_min |
|---|---|---|---|---|---|
| 1 MeV | 109.7 MeV | 122.3 MeV | 127.7 MeV | 0.99997 | 123.3 MeV |
| 10 MeV | 101.2 | 113.7 | 119.1 | 0.9967 | 123.3 |
| 100 MeV | 49.2 | 58.8 | 63.0 | 0.777 | 123.4 |
| 1 GeV | 6.20 | 7.68 | 8.36 | 0.123 | 124.2 |
| 10 GeV | 0.716 | 0.888 | 0.967 | 0.0133 | 133.3 |
| 1 TeV | 0.515 | 0.638 | 0.695 | 0.00113 | 1130 |

For m ≪ m_N the requirement is simply p ≥ √(m_N E_R/2) = 123 MeV/c; a 1 MeV particle must be ultra-relativistic (T ≈ 122 MeV, γ ≈ 123).

## 3. Recoil spectra and the lone-event statistic

### 3.1 Ingredients
- Efficiency ε(E): P003's model, plateau 0.96 with erf roll-offs to 50 % at 5.4 keV (σ = 3.4 keV) and 269.9 keV (σ = 8.0 keV) (LZ Fig. S2 / Data Analysis).
- Coherent SI response: Helm F² per isotope from `lz.helm_F2` (Lewin–Smith), combined for natural Xe with weights f_i A_i² (rate weights); shell-model M response F²_shell(E) for natural Xe and per isotope from P017's `P017_response_curves.npz` (WimPyDD/DMFormFactor tables). Check at 248 keV: Helm 1.634×10⁻⁴ (P017's own Helm 1.644×10⁻⁴), shell 2.911×10⁻⁵; F²(30)/F²(248) = 1249 (Helm), 6809 (shell). So the form factor suppresses 248 keV relative to the 2024 ROI by ×10³–10⁴, not ×5–10.
- Spin response for the q⁴ strawman: Σ″ isoscalar W₀₀(q) for ¹²⁹Xe and ¹³¹Xe (P017 npz), abundance-weighted, normalised at q = 0.
- Windows: low 5.4–55 keV, high 200–270 keV; N_lo = ∫_lo S ε dE / ∫_hi S ε dE.

### 3.2 Beam spectra
For a monochromatic beam, dσ/dE_R = (σ_χN/E_R,max) A(E_R, T) F²(E_R) Θ(E_R < E_R,max), where A is the squared amplitude relative to t → 0:
- constant (the Bringmann–Pospelov assumption): A = 1;
- vector mediator, Dirac χ and point nucleus: |M|² ∝ 2(s − Σ)² + 2 s t + t² with Σ = m² + m_N² (derived from the e-μ-type trace: 8[(p₁·p₂)(p₃·p₄) + (p₁·p₄)(p₂·p₃) − m²(p₂·p₄) − m_N²(p₁·p₃) + 2m²m_N²]; recalled/likely). In the lab, s − Σ = 2 m_N E_χ, t = −2 m_N E_R, giving A = 1 − E_R(Σ + 2 m_N E_χ)/(2 m_N E_χ²) + E_R²/(2E_χ²): the familiar CEνNS-like endpoint suppression;
- scalar mediator: |M|² ∝ (4m² − t)(4m_N² − t), so A = (1 + m_N E_R/(2m²))(1 + E_R/(2m_N)) (recalled/likely). For m ≪ √(m_N E_R) ≈ 60 MeV the spectrum is ∝ q² F²: the relativistic analogue of P003's q²-suppressed operators;
- pseudoscalar–pseudoscalar: |M|² ∝ t² and the nucleon vertex is spin-dependent, so the spectrum is ∝ q⁴ Σ″(q) — the relativistic version of O6.

### 3.3 Result: a floor on N_lo (Table 2; `P040_Nlo_beam_summary.csv`, `P040_Nlo_vs_Emax.csv`)
Flat spectrum with E_R,max ≫ 270 keV, constant amplitude:

| response | N_lo |
|---|---|
| none (efficiency only) | 0.72 |
| Helm, natural Xe | **349.9** |
| shell-model M | **1184** |
| q² × Helm / q² × shell | 33.2 / 113.5 |
| q⁴ × Helm / q⁴ × shell | 4.12 / 14.3 |
| q⁴ × Σ″ (spin, pseudoscalar) | 0.18 |

Cross-check: P030 obtained 353 (Helm) and 1182 (shell) for the same flat-spectrum construction with WimPyDD; P003's SHM O1 value is 2752 (shell) and `lz.dRdE_SI` at 1 TeV gives 820 (Helm) here — the SHM velocity distribution adds a further ×2.3 on top of the form factor.

Theorem-like statement: for any velocity-independent contact amplitude, every incident energy contributes a flat spectrum truncated at its own E_R,max(T). Truncation at E_R,max < 270 keV removes rate from the high window only, and E_R,max < 200 keV contributes to the numerator only, so N_lo ≥ N_lo,flat for *every* incident spectrum. The scan over E_R,max (Fig. 3) confirms it: N_lo = 362 (Helm) at E_R,max = 250 keV, 350 at ≥ 270 keV, minimum 349.9; shell 1204 → 1184. The vector-mediator endpoint suppression makes a beam tuned just above threshold worse (m = 1 MeV, T = 1.2 T_min: N_lo = 757 Helm / 2528 shell) and returns to the floor at T ≫ T_min (353/1193). The scalar mediator lowers N_lo to 33–36 (Helm) / 114–122 (shell) for m ≤ 10 MeV, 169/575 at 100 MeV, and gives no gain at m ≥ 1 GeV. Only the q⁴ spin structure passes (0.18), and q⁴ with a coherent form factor is borderline-to-failing (4.1 Helm, 14 shell).

Power-law incident spectra dΦ/dT ∝ T^−γ (γ = 1.5, 2, 2.7; `P040_Nlo_powerlaw.csv`): with T_lo = T_min(270 keV) every particle reaches the window and N_lo equals the floor exactly (349.9/1184, all γ, all masses); with T_lo = 10 MeV, N_lo = 5.6×10³–1.2×10⁵ for m ≤ 100 MeV because most particles cannot reach 200 keV. Self-consistent CRDM spectra (Sec. 4): N_lo = 1.7×10⁴ (1 MeV), 1.5×10⁴ (10 MeV), 2.2×10³ (100 MeV), 762 (1 GeV), 675 (10 GeV) with Helm; ×3.3 with the shell response.

## 4. CRDM normalisation, the cross-section for one LZ event, and existing constraints

### 4.1 Flux
dΦ_χ/dT_χ = D_eff (ρ_χ/m_χ) ∫ dT_p (dΦ_p/dT_p) (σ_χp/T_χ,max(T_p)) G_p²(q²) Θ(T_χ < T_χ,max), the Bringmann–Pospelov form with a constant amplitude, D_eff = 1 kpc (their conservative effective column; 10 kpc also shown), ρ_χ = 0.3 GeV cm⁻³ (Baxter 2021, as LZ), proton dipole form factor G = (1 + q²/0.71 GeV²)⁻² with q² = 2 m_χ T_χ (recalled/likely). Cosmic-ray protons: PDG all-nucleon intensity I_N(E) ≈ 1.8×10⁴ (E/GeV)^−2.7 m⁻² s⁻¹ sr⁻¹ GeV⁻¹, E = total energy per nucleon (recalled/likely; valid above a few GeV, extended to T_p = 0.1 GeV as an order-of-magnitude LIS; helium neglected). T_χ,max(T_p) follows from Sec. 2 with the roles swapped (m_p projectile, m_χ target).

Table 3 (`P040_crdm_flux.csv`; σ_χp = 10⁻³⁰ cm², D_eff = 1 kpc; cm⁻² s⁻¹):

| m_χ | Φ_total | Φ(T > 10 MeV) | Φ(T > T_min(248)) |
|---|---|---|---|
| 1 MeV | 1.1×10⁻⁵ | 1.7×10⁻⁶ | 2.1×10⁻⁷ |
| 10 MeV | 1.1×10⁻⁶ | 6.6×10⁻⁷ | 1.3×10⁻⁷ |
| 100 MeV | 9.1×10⁻⁸ | 8.4×10⁻⁸ | 5.2×10⁻⁸ |
| 1 GeV | 2.6×10⁻⁹ | 2.3×10⁻⁹ | 2.4×10⁻⁹ |
| 10 GeV | 9.0×10⁻¹¹ | 4.0×10⁻¹¹ | 8.4×10⁻¹¹ |

The assignment's recalled scaling "Φ(T > 10 MeV) ≈ 10⁻⁷ (σ/10⁻³⁰)(GeV/m_χ)" (uncertain) is reproduced within ×1.2 at 100 MeV, is ×17 too low at 1 MeV (most of the light-DM flux sits at T > 10 MeV) and ×40 too high at 1 GeV (proton form factor and 1/T_max weighting). Fig. 4 shows T dΦ/dT.

### 4.2 Rate in LZ
R_hi = Σ_i N_i t_live ∫dT (dΦ/dT) σ_χN,i ∫_{200}^{270} dE ε F_i² A/E_R,max,i, with σ_χN,i = A_i² (μ_χN/μ_χp)² σ_χp (coherent; μ ratio → 1 for light χ), N_i the isotope nuclei in 4.71 t, t_live = 220 d. R ∝ σ_χp², so σ_χp(1 event) = K^−1/2.

Table 4 (`P040_sigma_required.csv`):

| m_χ | σ_χp(1 event), Helm, 1 kpc | shell, 1 kpc | Helm, 10 kpc | N_lo (CRDM spectrum) Helm / shell |
|---|---|---|---|---|
| 1 MeV | 3.6×10⁻³⁰ cm² | 6.7×10⁻³⁰ | 1.2×10⁻³⁰ | 1.7×10⁴ / 5.7×10⁴ |
| 10 MeV | 4.4×10⁻³⁰ | 8.1×10⁻³⁰ | 1.4×10⁻³⁰ | 1.5×10⁴ / 5.0×10⁴ |
| 100 MeV | 6.3×10⁻³⁰ | 1.2×10⁻²⁹ | 2.0×10⁻³⁰ | 2.2×10³ / 7.3×10³ |
| 1 GeV | 2.4×10⁻²⁹ | 4.4×10⁻²⁹ | 7.4×10⁻³⁰ | 762 / 2.6×10³ |
| 10 GeV | 2.5×10⁻²⁹ | 4.6×10⁻²⁹ | 7.8×10⁻³⁰ | 675 / 2.3×10³ |

### 4.3 Comparison with existing constraints (recalled, uncertain)
Bringmann & Pospelov's XENON1T recast excludes roughly σ_χp ∈ [10⁻³¹, 10⁻²⁸] cm² for m_χ ≲ 0.1–1 GeV; PandaX-4T's dedicated CRDM search (2022) pushes the lower edge to a few ×10⁻³² (both recalled, uncertain by ×3). The required 3.6–6.3×10⁻³⁰ cm² (m ≤ 100 MeV) sits 40–60× inside that band, and 200× for PandaX. Independently of any recall, the same normalisation predicts 2×10³–6×10⁴ events in LZ's own 5.4–55 keV ROI in the same 220 live days (Table 4), against a 2024 null that tolerated a few (P003) — an excess by 10³–10⁴, and even larger than the *total* 1710 events of the whole science sample.

### 4.4 Earth attenuation (`P040_attenuation.csv`)
Standard rock, depth 1478 m (4850 ft), ρ = 2.7 g cm⁻³, X = 3.99×10⁵ g cm⁻²; composition by mass O 0.47, Si 0.34, Al 0.08, Ca 0.04, Fe 0.05, Mg 0.02 (recalled/likely). For a particle arriving with T = 1.5 T_min(248): mean number of scatters N_sc = Σ n_A A²(μ_A/μ_p)² ⟨F_A²⟩ σ_χp and mean fractional energy loss Σ n_A σ_N ⟨E_R⟩/T with Helm form factors over the flat spectrum (small-loss approximation).

| m_χ | N_sc at 10⁻³⁰ | fractional loss at 10⁻³⁰ | σ_χp for 30 % loss |
|---|---|---|---|
| 1 MeV | 0.51 | 0.08 % | 3.8×10⁻²⁸ cm² |
| 10 MeV | 0.54 | 0.09 % | 3.3×10⁻²⁸ |
| 100 MeV | 0.82 | 0.26 % | 1.2×10⁻²⁸ |
| 1 GeV | 3.1 | 7.6 % | 4.0×10⁻³⁰ |
| 10 GeV | 84 | (stopped) | 1.9×10⁻³² |

For m_χ ≤ 100 MeV the overburden is transparent at the required cross-sections (each scatter on O/Si removes ≲ 0.3 % of T); the 30 % ceiling of (1–4)×10⁻²⁸ cm² reproduces the recalled upper edge of the CRDM exclusion bands. For m_χ ≥ 1 GeV the required σ_χp (2.4×10⁻²⁹) exceeds the ceiling (4×10⁻³⁰): such particles are degraded below T_min before reaching the FV, so the GeV-scale constant-amplitude CRDM scenario cannot produce the event at all.

## 5. Heavy fast component (`P040_heavy_fast.csv`)
TeV DM with β = 0.005/0.01/0.02: E_R,max = 4.9/19.4/77.8 MeV at 1 TeV (5.8/23/92 MeV at 4 TeV), identical to 2μ²v²/m_N to 0.04 %. The SI spectrum is flat over the whole ROI, so N_lo = 349.9 (Helm) / 1184 (shell) — the same floor as any relativistic beam and ×2.3 *better* than the SHM (820 Helm / 2752 shell) but still ×70–240 above the tolerance. A fast SI sub-population does not change the elastic-SI conclusion; the low-energy population is set by the nuclear response, not by the velocity distribution.

## 6. Figures
- Fig. 1 `figures/P040_fig1_Nlo_vs_T.png`: N_lo vs incident kinetic energy for m = 1 MeV–10 GeV, constant and scalar-mediator amplitudes, Helm (left) and shell-model (right) responses; grey band 3–5.
- Fig. 2 `figures/P040_fig2_sigma_required.png`: σ_χp for one window event vs m_χ (Helm/shell, D_eff 1/10 kpc), recalled exclusion band (grey), 30 % attenuation ceiling (dash-dot).
- Fig. 3 `figures/P040_fig3_Nlo_vs_Emax.png`: N_lo of a flat spectrum vs its end point (minimum at E_R,max ≥ 270 keV).
- Fig. 4 `figures/P040_fig4_crdm_flux.png`: T dΦ/dT for σ_χp = 10⁻³⁰ cm², D_eff = 1 kpc, with T_min(248) marked.

## 7. Validation and robustness
- Flat-floor N_lo reproduces P030 (353/1182) to 1 %; efficiency-only value 0.72 vs (55−5.4)/70 = 0.709.
- Helm natural-Xe F² agrees with P017's Helm curve to 0.6 % at 248 keV.
- Neutron T_min 8.17 vs P013 8.20 MeV; TeV E_R,max vs NR formula 0.04 %.
- N_lo floor is independent of γ, T_lo (as long as T_lo ≥ T_min(270)), mass and beam energy; the vector amplitude only raises it; efficiency variants move it by ≤ 9 % (P003).
- σ_χp(1 event) scales as (D_eff Φ_p)^−1/2: a ×3 flux uncertainty (LIS shape, helium, D_eff) is ×1.7 in σ; the shell response costs ×1.85. None of this approaches the ×40–200 gap to the recalled exclusions or the ×10³–10⁴ low-energy excess.
- Attenuation uses a mean-loss approximation; for m ≥ 1 GeV the loss per scatter is 8–26 % so the "stopped" statement is qualitative but the direction is robust.

## 8. Failed or abandoned
- The working hypothesis that F² costs only ×5–10 between 10 and 248 keV (and hence N_lo ≈ 3–7, "borderline") was abandoned after evaluating the form factors: the ratio is 10³–10⁴ and N_lo ≥ 350.
- The recalled CRDM flux scaling 10⁻⁷ (σ/10⁻³⁰)(GeV/m_χ) was replaced by the numerical integral (Sec. 4.1), which differs from it by ×0.06–17 depending on mass.
- Blazar- and decay-boosted fluxes were not normalised separately: their N_lo obeys the same floor (Sec. 3.3), so only the flat-floor argument is needed.

## 9. Discussion
Being fast does not help. The lone-event criterion is a statement about the *response* at q ≈ 246 MeV relative to q ≈ 50–100 MeV: for any velocity-independent contact interaction the ratio of low- to high-window rates is bounded below by ∫F²ε(lo)/∫F²ε(hi) = 350 (Helm) or 1184 (shell), because each incident energy produces a flat spectrum that can only be cut off from above. Boosting removes the SHM velocity penalty (×2.3) but not the form-factor one (×10³). The only way down is explicit q-dependence in the amplitude — q² (scalar mediator on a light fermion: N_lo 33–114), q⁴ (4–14) or, decisively, spin-dependent q⁴ Σ″ (0.18), i.e. the same operators P003 identified in the non-relativistic case. CRDM with the standard constant amplitude needs σ_χp ≈ (3.6–6.3)×10⁻³⁰ cm² for m_χ ≤ 100 MeV, inside the recalled XENON1T/PandaX-4T exclusion bands and, more robustly, predicting 10³–10⁴ events in LZ's own low-energy ROI; at m_χ ≥ 1 GeV the required cross-section is above the Earth-attenuation ceiling. The dossier's 0.04 probability for hypothesis K is generous for the SI/CRDM realisation; what survives is a relativistic beam with q⁴-spin couplings, whose flux would then need its own normalisation.

## 10. References
1. LZ Collaboration, arXiv:2609.02823 (2026).
2. T. Bringmann, M. Pospelov, Phys. Rev. Lett. 122, 171801 (2019) — CRDM mechanism, XENON1T recast.
3. Y. Ema, F. Sala, R. Sato, Phys. Rev. Lett. 122, 181802 (2019) — light DM at neutrino experiments.
4. C. V. Cappiello, J. F. Beacom, Phys. Rev. D 100, 103011 (2019) — CRDM limits from neutrino experiments.
5. J. B. Dent, B. Dutta, J. L. Newstead, I. M. Shoemaker, Phys. Rev. D 101, 116007 (2020) — CRDM in simplified models (q-dependent amplitudes).
6. PandaX-4T Collaboration, Phys. Rev. Lett. 128, 171801 (2022) — CRDM search.
7. J. D. Lewin, P. F. Smith, Astropart. Phys. 6, 87 (1996) — Helm form factor.
8. Particle Data Group, Review of Particle Physics, cosmic-ray section (nucleon spectrum).
9. Corpus: P003, P013, P016, P017, P019, P030.

## 11. Tools and provenance (mirrors `output/provenance/P040.json`)
- Agent tools: Read (PAPER_GUIDE, dossier, ledger, P003/P013/P016/P017/P019 papers, lzcommon.py, fulltext.tex Table I/Data Analysis/Fig. 5, P017 CSV/npz, figures), Bash (grep of tex and P003 script; npz inspection; mkdir; three script runs), Write (script, details, provenance, paper), Edit (script fixes).
- Software: python 3.12.13; numpy 2.5.3 (trapezoid, interp, logspace); scipy 1.18.1 (special.erf); pandas 3.0.5 (tables); matplotlib 3.11.2 (Agg); common/lzcommon.py (LZ constants, XE_ISOTOPES, helm_F2, mu_red, dRdE_SI); P017 response tables (WimPyDD 2.0.4 output, not re-run). Kinematic and amplitude formulae derived by hand.
- Local inputs: fulltext.tex (Data Analysis l.110–117, Table I l.210–246, Fig. 5 caption l.255–266); dossier §5 (hypothesis K); ledger; P003.md, P013.md, P016.md, P017.md, P019.md, P030 ledger row; `output/work/P017/P017_response_curves.npz`, `P017_M_ratios.csv`; `output/code/P003_nreft_shapes.py` (efficiency constants); lzcommon.py; ENVIRONMENT_versions.txt.
- Recalled knowledge (11 items): two-body kinematics (certain); vector-/scalar-/pseudoscalar-mediator squared amplitudes (likely); PDG nucleon spectrum 1.8×10⁴ E^−2.7 (likely); proton dipole form factor Λ² = 0.71 GeV² (likely); BP19 CRDM formula with D_eff = 1 kpc (likely); XENON1T-recast band 10⁻³¹–10⁻²⁸ cm² (uncertain); PandaX-4T lower edge few ×10⁻³² (uncertain); SURF depth 1478 m and standard-rock density/composition (likely); Helm parametrisation (certain, via lzcommon); 2024 tolerance 3–5 events (uncertain, corpus); recalled flux scaling 10⁻⁷(σ/10⁻³⁰)(GeV/m) (uncertain, superseded).
- Datasets: none. Data requests: none. WimPyDD-generated files: none (P017's cached tables reused).
