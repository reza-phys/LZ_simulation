# P019 — Atmospheric-neutrino processes at 248 keV: the CEνNS tail, loss of coherence, and exotic neutrino interactions

Research record (simulated date 2026-09-05). Script: `output/code/P019_atm_nu.py` (run from the simulation root with `.venv/bin/python`; ≈2 min). All numbers below are printed to `output/work/P019/run_log.txt` and stored in `P019_results.json`; tables in the CSV files listed in §9.

## 1. Motivation and framework

LZ (arXiv:2609.02823) lists atmospheric-neutrino CEνNS as one of the three backgrounds that dominate the (S1c, log₁₀S2c) neighbourhood of the 248 keV candidate (Discussion, l. 307), expects 0.11 ± 0.02 such events in the whole 5.4–270 keV WS ROI (Table I) and states that "the reconstructed energy of the event is not consistent with scattering coherence" (l. 314). The dossier (§5, hypothesis A and H) assigns 10% to "a fluctuation of a correctly modelled background" and 2% to "non-DM new physics (exotic neutrino interactions …)". We quantify, as neutrino phenomenologists, four things:

1. the expected CEνNS rate at E_R ≥ 200 keV and in the event window 225–271 keV, normalised to LZ's 0.11 events so that the (poorly recalled) absolute atmospheric flux drops out and only the flux *shape* above ~100 MeV matters;
2. what "loss of coherence" means quantitatively at q = 246 MeV: the weak form factor, the node structure across the nine xenon isotopes, and the point where incoherent (quasi-elastic) scattering overtakes coherent scattering;
3. whether any incoherent process (nucleon knock-out, nuclear excitation) can leave a *lone* 248 keV nuclear recoil;
4. whether an exotic neutrino interaction (magnetic moment, light Z′, scalar mediator, dipole-portal up-scattering into a heavy neutral lepton) can produce a 248 keV recoil without a large low-energy population.

Solar neutrinos are dismissed kinematically (§7).

## 2. Inputs

| Input | Value | Source |
|---|---|---|
| Atmospheric ν in WS ROI, 2.84 t·yr | 0.11 ± 0.02 events | LZ Table I (l. 234) |
| ⁸B + hep ν in WS ROI | 0.057 ± 0.006 | LZ Table I (l. 235) |
| Recoil-spectrum formalism used by LZ | Billard, Strigari, Figueroa-Feliciano 2014; flux normalisation Baxter et al. 2021 | LZ l. 175–176 |
| Detector NRs (neutrons) post-fit | [0, 0.118] | LZ Table I (l. 237) |
| Veto tagging of (α,n) neutrons scattering in the TPC | 92 ± 4% | LZ l. 178 |
| Fig. 5: "Neutrino + Detector NRs" green histogram in three S1c panels; bottom-panel total 0.0106 ± 0.0008 | digitised here | LZ Fig. 5 + caption (l. 255–266), PNG `inputs/figures_png/Fig5_NR_distance_3panels_no_sig.png` |
| Efficiency: 50% at 5.4 and 269.9 keV; Gaussian roll-off σ_E = 11.8 keV; plateau 0.955 | ε(E) = 0.955 Φ((269.9−E)/11.8) Φ((E−5.4)/1.0) | LZ l. 117, Fig. S2; P009 §5.5 |
| S1c(E) on the paper's scale: 543.8 phd at 250 keV | nestpy LZ-tuned yields (Table S5 incl. p(E) break) × 1.082 | P009 (Fig. 4 contour crossings); `lz.nest_nr_yields` |
| S1 resolution at 250 keV | 5% | P009 |
| Helm form factor, Xe isotopic abundances, G_F-scale constants | `lzcommon.helm_F2`, `lz.XE_ISOTOPES` | `output/code/common/lzcommon.py` |
| WimPyDD shell-model M response (DMFormfactor densities) | `lz.wd_hamiltonian`, `lz.wd_rate` | WimPyDD 2.0.4 |
| Event window | 248 ± 23 keV → 225–271 keV | LZ l. 165 |

Recalled knowledge (flagged; complete list in §10): atmospheric flux normalisation 10.5 cm⁻²s⁻¹ above ~10 MeV, all flavours (Billard et al. 2014; likely) and its shape (uncertain); sin²θ_W = 0.23867 at low q (likely); Llewellyn-Smith NC elastic formula with dipole form factors M_A = 1.03 GeV, M_V = 0.84 GeV, g_A = 1.267, μ_p = 2.793, μ_n = −1.913 (likely); Fermi-gas Pauli factor and k_F = 260 MeV/c for a heavy nucleus (likely/uncertain); n–Xe total cross-section ≈ 2.7 b at 30 MeV (uncertain); ¹²⁹Xe 39.58 keV (3/2⁺) and ¹³¹Xe 80.19 keV (1/2⁺) first excited states (likely); Vogel–Engel magnetic-moment formula (certain); scalar-mediator spectral form (Cerdeño et al. 2016; likely); solar hep end point 18.8 MeV, ⁸B ≈ 16.3 MeV (certain).

## 3. Atmospheric flux model and CEνNS spectrum

### 3.1 Flux shape family (recalled, bracketed)

dΦ/dE = C (E/E_b)^(−γ₁) for 10 MeV < E < E_b, C (E/E_b)^(−γ₂) for E > E_b [cm⁻² s⁻¹ MeV⁻¹]. Eight shapes:

| label | γ₁ | E_b [MeV] | γ₂ | comment |
|---|---|---|---|---|
| central | 1 | 100 | 2.5 | E·dΦ/dE flat 10–100 MeV (assignment), E^−2.5 above |
| soft high-E | 1 | 100 | 3 | |
| hard high-E | 1 | 100 | 2 | |
| Honda-like | 1.5 | 1000 | 2.7 | E²dΦ/dE rising to ~1 GeV, then steepening (recalled Honda/Battistoni behaviour) |
| steep low-E | 2 | 100 | 2.5 | |
| low break | 1 | 50 | 2.5 | |
| high break | 1 | 150 | 2.5 | |
| single power law | 2 | — | 2 | E^−2 from 10 MeV |

Neutral-current scattering is flavour-blind and ν/ν̄-symmetric at leading order, so only the total flux enters.

### 3.2 Cross-section and rate

dσ/dE_R = (G_F² m_N / 4π) Q_W² [1 − m_N E_R/(2E_ν²)] F²(q),  Q_W = N − (1 − 4 sin²θ_W) Z,  q² = 2 m_N E_R.

Q_W(¹³¹Xe) = 74.55; q(248 keV, A = 131) = 246.0 MeV. Exact kinematic threshold E_ν,min = [E_R + √(E_R² + 2 m_N E_R)]/2: 110.6 MeV for 200 keV, 123.1 MeV for 248 keV.

dR/dE_R = Σ_isotopes n_T,i ∫_{E_min} dE_ν Φ(E_ν) dσ_i/dE_R, with n_T,i nuclei per tonne of natural xenon, in events/(t·yr·keV), evaluated on a 0.5–330 keV grid (`scipy.integrate.quad`).

Two form-factor models:
- **Helm** (Lewin–Smith parametrisation, `lz.helm_F2`) per isotope — this is what the Billard et al. formalism uses, hence what LZ's model uses.
- **WimPyDD shell-model weak form factor**: the WimPyDD O₁ rate with couplings c_p = −(1 − 4 sin²θ_W), c_n = 1 (WimPyDD c⁰ = c_p + c_n, c¹ = c_p − c_n) for a 10⁶ GeV WIMP (velocity integral independent of E_R), divided by the Helm Q_W²F² sum and normalised to 1 at 1 keV, defines F²_W(E) for natural xenon from the one-body density matrices used by LZ for its signal models. Because the weak charge is 97% neutron, this probes the neutron density; the ratio to Helm is already 0.89 at 20 keV and 0.7 at 50 keV.

### 3.3 Normalisation

For each shape and form-factor model, C is fixed so that ∫ dR/dE_R ε(E) dE_R × 2.84 t·yr = 0.11. Sanity check of the machinery with the *recalled* absolute normalisation (10.5 cm⁻²s⁻¹): the central shape predicts 0.073 events, the Honda-like 0.067, the hard 0.082 (Table `cevns_tail_summary.csv`, column `N_pred_with_recalled_flux`), i.e. our first-principles rate is within a factor 1.5 of LZ's 0.11 — the implied total flux is 14–17 cm⁻²s⁻¹ for the γ₁ = 1 shapes (37 for the steep low-E shape, which is thereby disfavoured as a description of the flux LZ used). The fit-normalised approach removes this uncertainty from everything that follows.

### 3.4 Results: the high-energy tail

Efficiency-weighted events in 2.84 t·yr (from `cevns_tail_summary.csv`):

| shape | FF | N(E_R > 200 keV) | N(225–271 keV) | frac > 100 (raw) | frac > 150 | frac > 200 | frac > 248 | N_lo = N(5.4–200)/N(200–270) |
|---|---|---|---|---|---|---|---|---|
| central | Helm | 6.5 × 10⁻⁵ | 1.66 × 10⁻⁵ | 4.0 × 10⁻³ | 2.1 × 10⁻³ | 4.2 × 10⁻⁴ | 4.1 × 10⁻⁵ | 1631 |
| central | WimPyDD | 4.8 × 10⁻⁶ | 6.2 × 10⁻⁷ | 8.9 × 10⁻⁴ | 3.6 × 10⁻⁴ | 4.3 × 10⁻⁵ | 1.8 × 10⁻⁵ | 23 000 |
| soft high-E | Helm | 4.6 × 10⁻⁵ | 1.13 × 10⁻⁵ | 3.0 × 10⁻³ | 1.5 × 10⁻³ | 2.8 × 10⁻⁴ | 2.6 × 10⁻⁵ | 2329 |
| hard high-E | Helm | 9.7 × 10⁻⁵ | 2.51 × 10⁻⁵ | 5.6 × 10⁻³ | 3.1 × 10⁻³ | 6.4 × 10⁻⁴ | 6.8 × 10⁻⁵ | 1099 |
| Honda-like | Helm | 1.21 × 10⁻⁴ | 3.18 × 10⁻⁵ | 6.5 × 10⁻³ | 3.7 × 10⁻³ | 7.8 × 10⁻⁴ | 8.6 × 10⁻⁵ | 879 |
| steep low-E | Helm | 4.7 × 10⁻⁵ | 1.19 × 10⁻⁵ | 2.3 × 10⁻³ | 1.3 × 10⁻³ | 2.4 × 10⁻⁴ | 2.4 × 10⁻⁵ | 2241 |
| low break | Helm | 4.3 × 10⁻⁵ | 1.09 × 10⁻⁵ | 2.4 × 10⁻³ | 1.3 × 10⁻³ | 2.5 × 10⁻⁴ | 2.5 × 10⁻⁵ | 2460 |
| high break | Helm | 9.1 × 10⁻⁵ | 2.32 × 10⁻⁵ | 5.5 × 10⁻³ | 3.0 × 10⁻³ | 6.0 × 10⁻⁴ | 6.0 × 10⁻⁵ | 1174 |
| single E^−2 | Helm | 7.4 × 10⁻⁵ | 1.91 × 10⁻⁵ | 3.6 × 10⁻³ | 2.0 × 10⁻³ | 4.0 × 10⁻⁴ | 4.3 × 10⁻⁵ | 1429 |

Ranges over all 16 variants: N(>200 keV) = 3.1 × 10⁻⁶ – 1.2 × 10⁻⁴; N(225–271) = 4.0 × 10⁻⁷ – 3.2 × 10⁻⁵; raw tail fraction above 200 keV = 2.5 × 10⁻⁵ – 7.8 × 10⁻⁴; N_lo = 880 – 35 000. Helm only: N(225–271) = 1.1–3.2 × 10⁻⁵. The differential rate at 248 keV (central, Helm) is 1.0 × 10⁻⁷ events/(t·yr·keV), versus 5.8 × 10⁻³ at 20 keV (the "approximately exponentially-decaying" spectrum of LZ l. 313 falls by 5.5 × 10⁴ between 20 and 248 keV; the Helm decline is faster than exponential because of the two nodes).

Poisson: P(≥1 event above 200 keV) ≤ 1.2 × 10⁻⁴; in the ±23 keV window ≤ 3.2 × 10⁻⁵.

### 3.5 Comparison with LZ's own model (Fig. 5 green curve)

We digitised the green "Neutrino + Detector NRs" histogram in all three panels of Fig. 5 by pixel colour (swatch sampled from the legend, RGB ≈ (124,174,0)), using P004's frame/tick calibration extended to the upper panels (frame rows 496/1299/2102/2905; 141, 120 and 96.5 px per decade; bottom-panel labels every two decades). Validation on the blue "Background Model Total": top-panel sum 1672 (Table I total 1713 minus the ≈25 events of the other panels), middle 24.6 (data points sum to ≈ 24), bottom 0.01082 vs caption 0.0106 ± 0.0008 (P004 obtained 0.01053).

| panel | S1c range | ≈ E_R range (paper scale) | green sum (digitised) | within ±2σ_NR | CEνNS prediction, central Helm | Honda-like Helm | central WimPyDD |
|---|---|---|---|---|---|---|---|
| top | < 250 phd | < 128 keV | 0.176 | 0.108 | 0.109 | 0.109 | 0.110 |
| middle | 250–500 | 128–232 keV | 8.2 × 10⁻⁴ | 7.5 × 10⁻⁴ | 5.2 × 10⁻⁴ | 8.7 × 10⁻⁴ | 1.1 × 10⁻⁴ |
| bottom | > 500 | > 232 keV | 3.4 × 10⁻⁵ | 3.1 × 10⁻⁵ | 1.1 × 10⁻⁵ | 2.2 × 10⁻⁵ | 4.9 × 10⁻⁷ |

The top-panel green also contains the ⁸B+hep 0.057 and whatever detector-NR normalisation the fit returned: 0.176 − 0.11 − 0.057 = 0.009, consistent with a small fitted neutron component (interval [0, 0.118]). The middle and bottom sums are upper limits on the atmospheric contribution (the neutron component's high-energy part is included). The Helm-based predictions with the harder flux shapes (Honda-like: ratios observed/predicted 0.94 and 1.6; hard high-E: 1.13 and 2.0) reproduce LZ's curve; the central shape is low by 1.6 and 3.0; the WimPyDD shell-model form factor is low by 5–100 in the middle panel and 35–100 in the bottom panel. We conclude that (i) LZ's model uses a Helm-type form factor and a flux with a substantial 100–300 MeV component (as Honda/FLUKA fluxes have), and (ii) its own prediction for S1c > 500 phd is ≈ 3 × 10⁻⁵ events, of which ≈ 6 × 10⁻⁶ lies in the event's 0.5σ bin (digitised value at −2 < σ < −1.5: 5.9 × 10⁻⁶). Within ±23 keV our best estimate is N = (1–3) × 10⁻⁵ (Helm; LZ-like flux), a factor ~15 smaller with the shell-model density.

## 4. Coherence loss at q = 246 MeV

Helm F² per isotope at 248 keV and position of the second node (first node ≈ 94–100 keV):

| A | abundance | F²(248 keV) | node₂ [keV] |
|---|---|---|---|
| 124 | 0.00095 | 5.7 × 10⁻⁴ | 307 |
| 126 | 0.00089 | 4.3 × 10⁻⁴ | 298 |
| 128 | 0.0191 | 3.1 × 10⁻⁴ | 291 |
| 129 | 0.2644 | 2.6 × 10⁻⁴ | 287 |
| 130 | 0.0408 | 2.1 × 10⁻⁴ | 283 |
| 131 | 0.2118 | 1.7 × 10⁻⁴ | 280 |
| 132 | 0.2689 | 1.3 × 10⁻⁴ | 276 |
| 134 | 0.1044 | 7.4 × 10⁻⁵ | 269 |
| 136 | 0.0887 | 3.4 × 10⁻⁵ | 262 |

Natural-xenon Q_W²-weighted F²(248 keV) = 1.6 × 10⁻⁴ (Helm) and 2.6 × 10⁻⁶ (WimPyDD shell model); F²(20 keV) = 0.72 (Helm) / 0.68 (WimPyDD). The natural-mixture minimum lies at 278.5 keV (Helm) and 246 keV (WimPyDD) — with shell-model densities the event sits at the coherence zero of natural xenon. The coherent weight at the event, Q_W² F² = 74.55² × 1.7 × 10⁻⁴ = 0.9 for ¹³¹Xe, is smaller than the weight of a *single* free nucleon (≈ 0.25–0.5 from vector+axial couplings): the N² coherent enhancement (5560 at q → 0) is completely gone at 248 keV. This is the quantitative content of LZ's statement that the energy "is not consistent with scattering coherence". The struck-nucleon kinetic energy corresponding to this q on a free nucleon is Q²/2m_N = 32.3 MeV (Q² = 0.0605 GeV²).

## 5. Incoherent processes

### 5.1 Quasi-elastic NC knock-out ν N → ν N′

Free-nucleon NC elastic cross-sections from the Llewellyn-Smith form dσ/dQ² = (G_F² M²/8πE²)[A ∓ B (s−u)/M² + C (s−u)²/M⁴] with NC form factors F₁,₂^NC = ±½(F₁,₂^p − F₁,₂^n) − 2 sin²θ_W F₁,₂^{p,n}, F_A^NC = ±½ g_A G_D(Q²; M_A), dipole G_D, ν/ν̄ averaged, times a Fermi-gas Pauli factor P(q) = 1.5x − 0.5x³ (x = q/2k_F < 1; k_F = 260 MeV/c) (recalled, uncertain at the factor-2 level):

| E_ν [MeV] | σ_n (Pauli) [cm²] | σ_n free | Σ nucleons (54σ_p + 77σ_n) | σ_coh(¹³¹Xe, Helm) | incoh/coh |
|---|---|---|---|---|---|
| 30 | 2.3 × 10⁻⁴² | 2.0 × 10⁻⁴¹ | 2.8 × 10⁻⁴⁰ | 1.7 × 10⁻³⁸ | 0.017 |
| 50 | 9.5 × 10⁻⁴² | 5.2 × 10⁻⁴¹ | 1.2 × 10⁻³⁹ | 3.3 × 10⁻³⁸ | 0.036 |
| 100 | 5.9 × 10⁻⁴¹ | 1.8 × 10⁻⁴⁰ | 7.3 × 10⁻³⁹ | 4.8 × 10⁻³⁸ | 0.15 |
| 150 | 1.6 × 10⁻⁴⁰ | 3.3 × 10⁻⁴⁰ | 1.9 × 10⁻³⁸ | 5.1 × 10⁻³⁸ | 0.37 |
| 200 | 2.9 × 10⁻⁴⁰ | 5.0 × 10⁻⁴⁰ | 3.4 × 10⁻³⁸ | 5.2 × 10⁻³⁸ | 0.66 |
| 300 | 5.7 × 10⁻⁴⁰ | 8.0 × 10⁻⁴⁰ | 6.8 × 10⁻³⁸ | 5.3 × 10⁻³⁸ | 1.3 |
| 500 | 9.9 × 10⁻⁴⁰ | 1.2 × 10⁻³⁹ | 1.2 × 10⁻³⁷ | 5.3 × 10⁻³⁸ | 2.2 |
| 1000 | 1.4 × 10⁻³⁹ | 1.6 × 10⁻³⁹ | 1.6 × 10⁻³⁷ | 5.6 × 10⁻³⁸ | 2.8 |

Incoherent scattering overtakes coherent at E_ν ≈ 250 MeV — exactly the neutrino energies that could produce a 248 keV coherent recoil (threshold 123 MeV). The naive "1/N ≈ 1%" recalled estimate holds only at E_ν ≲ 30 MeV where Pauli blocking is strong and F² ≈ 1. Flux-weighted with the normalised central flux: neutron knock-out 1.45 × 10⁻² /(t·yr) → 0.041 events in the exposure; proton knock-out 8.0 × 10⁻³ /(t·yr) → 0.023 events. These are not negligible compared with the 0.11 coherent events, but their signatures are different:

- **Proton knock-out**: the proton carries T_p = Q²/2m_N (tens of MeV for the Q² that dominate; even Pauli-allowed low-Q² events give T_p ≳ 1 MeV), deposits it within millimetres → S1 of 10⁴–10⁵ phd, far outside the S1c < 600 phd ROI. Not a 248 keV NR.
- **Neutron knock-out**: the ejected neutron (T_n ~ 10–100 MeV, λ ≈ 27 cm in LXe for σ_tot ≈ 2.7 b — recalled, uncertain) either scatters again in the TPC (multi-site, rejected), or escapes into the skin/OD where LZ tags neutrons with 92 ± 4% efficiency, or leaves undetected. The **residual (A−1) nucleus recoils with the struck neutron's Fermi momentum** p: E_res = p²/2M_{A−1}, maximum k_F²/2M = 279 keV for k_F = 260 MeV/c. In a sharp Fermi sphere the fraction of removals with 225 < E_res < 271 keV is [(p_hi³ − p_lo³)/k_F³] = 0.23; above 200 keV, 0.34. The daughter must be in its ground state (excited daughters emit ≥ 0.5 MeV γ-rays absorbed within centimetres → ER-dominated or multi-site event). Estimate: N_lone = 0.041 × P_escape (0.3; range 0.1–0.35) × P_untagged (0.2; 0.08–0.3) × P_gs (0.3; 0.1–0.5) × 0.23 = **1.7 × 10⁻⁴ events (range 8 × 10⁻⁶ – 5 × 10⁻⁴)**. This is 1–30 times the *coherent* rate in the same window (1.7 × 10⁻⁵, central Helm) — i.e. at 248 keV the lone-NR neutrino background is dominated by the incoherent Fermi-recoil channel, not by the CEνNS tail that LZ models — but the absolute expectation remains ≤ 5 × 10⁻⁴, P(≥1) ≤ 0.05%. The Fermi-sphere treatment overestimates the high-momentum ground-state fraction (real mean-field momentum distributions fall smoothly before k_F, and high-momentum components come from short-range-correlated pairs whose removal leaves the daughter highly excited), so the upper end of the range is conservative.

### 5.2 NC excitation of low-lying levels

Allowed-approximation σ ≈ (G_F² E_ν²/π)(g_A²/4) B(σ) with B(σ) = 0.1 (recalled, uncertain by ×3) for ¹²⁹Xe (39.58 keV, 3/2⁺) and ¹³¹Xe (80.19 keV, 1/2⁺), flux-weighted with the normalised central flux: 3.8 × 10⁻⁵ and 3.0 × 10⁻⁵ events in 2.84 t·yr (3 × 10⁻⁴ of the CEνNS rate). Their signature is a fully absorbed γ/conversion-electron cascade of 39.6 or 80.2 keV (ER) plus a CEνNS-like recoil of a few to tens of keV. With nestpy LZ-tuned yields (P009 S1 scale) a 39.6 keV ER + 20 keV NR lands at S1c ≈ 286 phd, log₁₀S2c ≈ 4.46, and 80.2 keV ER + 20 keV NR at S1c ≈ 545 phd, log₁₀S2c ≈ 4.77 — inside the ER band and *above* the WS ROI upper boundary log₁₀S2c = 4.15. The NR-band centre at 248 keV is (539 phd, 4.02). Giant-resonance excitations (E_x ≳ 10 MeV, σ ~ 10⁻⁴⁰ cm² at 50–100 MeV, recalled/uncertain) decay by neutron and γ emission with MeV visible energy — again not a lone 248 keV NR.

## 6. Exotic neutrino interactions: can anything harden the spectrum?

We compute, for the central flux shape and Helm form factor, N_lo ≡ N(5.4–200 keV)/N(200–270 keV) (efficiency-weighted), the number of low-energy events that must accompany one event in the 200–270 keV band; P003 argued that LZ's 2024 low-energy null result tolerates N_lo ≲ 3–5.

| interaction | dσ/dE_R shape (× F²) | N_lo | frac(200–270) |
|---|---|---|---|
| SM CEνNS / any heavy mediator (rescaling) | Q_W² [1 − m_N E_R/2E_ν²] | 1631 | 5.9 × 10⁻⁴ |
| ν magnetic moment on the nucleus (Vogel–Engel) | Z² (1/E_R)(1 − E_R/2E_ν)² | 20 100 | 4.5 × 10⁻⁵ |
| light vector Z′, m = 10 / 30 / 100 / 246 / 1000 MeV | SM × m⁴/(q² + m²)² | 5.2 × 10⁵ / 3.0 × 10⁵ / 3.3 × 10⁴ / 4950 / 1790 | 1.6 × 10⁻⁶ … 5.4 × 10⁻⁴ |
| scalar mediator, m = 30 / 246 / 1000 / 10⁵ MeV | A² m_N² E_R/(E_ν² (q² + m²)²) | 1.24 × 10⁵ / 2350 / 862 / 786 | 7 × 10⁻⁶ … 1.2 × 10⁻³ |
| dipole-portal HNL, m₄ = 0 / 0.3 / 10 / 50 / 100 / 200 / 230 / 240 / 245 MeV | derived below | 20 100 / 20 100 / 19 600 / 12 600 / 4970 / 1620 / 1370 / 1310 / 1280 | 4.5 × 10⁻⁵ … 7.6 × 10⁻⁴ |

Observations. (a) A neutrino magnetic moment mostly enhances ν–e scattering (an ER); its coherent nuclear piece is ∝ 1/E_R and is the *softest* spectrum of all. (b) A light vector mediator has a propagator falling with q², so it can only soften the spectrum; an "enhancement at 248 keV" requires m_Z′ ≫ q, which is a pure rescaling of the SM shape with N_lo = 1600. (c) A scalar mediator gives dσ/dE_R ∝ E_R — the hardest spectrum available — yet the flux fall-off above 120 MeV and the form factor still leave N_lo = 790–860. (d) Dipole-portal up-scattering ν N → N₄ N: unlike inelastic DM (v ≲ 800 km/s), a neutrino beam of E_ν ~ 100–300 MeV can pay for a heavy final state without kinematic restriction; the E_R range at E_ν = 200 MeV is 10⁻⁴–653 keV for m₄ = 10 MeV and 3–569 keV for m₄ = 100 MeV, so a sub-MeV "sterile splitting" (m₄ = 0.3 MeV) is indistinguishable from m₄ = 0. Only near threshold, m₄ → E_ν, is the recoil pinned to E_R ≈ E_ν²/2M: a 248 keV recoil at threshold needs m₄ = E_ν = 246 MeV, but the flux above 246 MeV and the 1/E_R photon pole still give N_lo = 1280 for m₄ = 245 MeV. No exotic ν–nucleus interaction with an atmospheric-flux source can produce one 248 keV recoil without ≳ 800 recoils below 200 keV.

### 6.1 Dipole-portal cross-section (derived here)

L = d ν̄_L σ^{μν} N₄ F_{μν}; nucleus treated as a spin-0 charge Ze F(q²) (coherent). With p₁ (ν, massless), p₂ (N₄, mass m₄), P, P′ (nucleus, mass M), q = p₁ − p₂, t = q² = −2ME_R, K = P + P′ (K·q = 0, K² = 4M² − t): the vertex σ^{μν}K_μ q_ν = i K̸q̸, and the spin sum gives

L·H = K² m₄² (t − m₄²) − 4t (p₁·K)²,  p₁·K = 2ME_ν + (t − m₄²)/2,

dσ/dE_R = Z² α d² F²(q²) · L·H / (32 M³ E_R² E_ν²),
L·H = 8ME_R [2ME_ν − ME_R − m₄²/2]² − (4M² + 2ME_R) m₄² (2ME_R + m₄²).

Kinematic limits from the CM frame: E_R^± = −t^∓/2M with t^∓ = m₄² − 2E₁*(E₂* ∓ p₂*), E₁* = (s − M²)/2√s, E₂* = (s + m₄² − M²)/2√s, s = M² + 2ME_ν; threshold E_ν ≥ m₄ + m₄²/2M. Check: for m₄ → 0, dσ/dE_R → Z² α d² F² (1 − E_R/2E_ν)²/E_R, the Vogel–Engel coherent magnetic-moment formula (with d ↔ μ_ν); the script confirms that the m₄ = 0 spectrum coincides with the magnetic-moment kernel to 10⁻¹² (relative spread). The bracket L·H is positive throughout the allowed range (clipped at 0 as a guard; never triggered on the grid).

## 7. Solar neutrinos

E_R,max = 2E_ν²/(m_N + 2E_ν): hep (18.8 MeV) → 5.79 keV; ⁸B (16.3 MeV) → 4.35 keV (A = 131). A 248 keV recoil needs E_ν ≥ 123 MeV, 200 keV needs 110.6 MeV. Solar neutrinos are irrelevant at the event energy (they appear in the ROI only through the 5.4 keV threshold's 50% efficiency point; LZ's 0.057 events are at the very bottom of the ROI).

## 8. Discussion and conclusions

1. The CEνNS tail is tiny: normalised to LZ's own 0.11 ROI events, the Helm-based expectation is 4–12 × 10⁻⁵ events above 200 keV and 1–3 × 10⁻⁵ in 225–271 keV; LZ's own Fig. 5 green curve has 3.4 × 10⁻⁵ at S1c > 500 phd (including any fitted neutrons), consistent with the harder (Honda-like) flux shapes. A shell-model neutron density (WimPyDD M response with weak-charge couplings) moves the natural-Xe coherence node from 278 to 246 keV and cuts the tail by another factor ~15; whichever is right, P(≥1 CEνNS event in the window) ≤ 3 × 10⁻⁵.
2. Coherence is genuinely lost: Q_W²F² = 0.9 at the event (vs 5560 at q → 0), less than one free nucleon's weight, and the incoherent NC cross-section per nucleus exceeds the coherent one for E_ν ≳ 250 MeV.
3. The incoherent channels do not mimic the event except for one: neutron knock-out with an escaping, untagged neutron and a ground-state daughter, whose Fermi recoil (≤ 279 keV) would be a lone NR of the right energy. Its expectation, 1.7 × 10⁻⁴ (8 × 10⁻⁶ – 5 × 10⁻⁴) events, exceeds the modelled coherent tail in the window but is still ≤ 0.05% probable; it is not in LZ's background model and should be added for future high-energy searches (its spectrum, ∝ p² up to k_F, is *rising* toward 250 keV, unlike anything else in the model).
4. No exotic neutrino–nucleus interaction fed by the atmospheric flux (magnetic moment, light or heavy vector, scalar, dipole-portal HNL of any mass up to threshold) yields N_lo < 790: each would have produced hundreds of low-energy NR-band events in the same exposure. The "kinematically special" loophole that exists for inelastic dark matter does not exist for a relativistic neutrino beam.
5. Stance: favours neither DM nor a neutrino origin — the event is not an atmospheric neutrino of any kind (SM or exotic) at the > 99.9% level; it constrains hypothesis A (as far as neutrinos are concerned) and H in the dossier to ≲ 10⁻³ each.

Failed/abandoned: (i) reusing P004's green colour (124,174,0) with tolerance 60 returned zero pixels in all panels — replaced by sampling the legend swatch; (ii) the first tick calibration of the bottom panel (labels every two decades) gave 193 px/decade and underestimated the bottom-panel sum by ×2 — fixed by the per-panel label step and validated against the caption total; (iii) the dipole-portal spectrum with a step-function kinematic kernel triggered `quad` round-off warnings — replaced by a bisection for E_ν,min(E_R, m₄) and integration only over the allowed range.

## 9. Files

- `output/code/P019_atm_nu.py` — everything.
- `output/work/P019/P019_results.json` — all numbers; `run_log.txt` — printed log.
- `cevns_tail_summary.csv` — 16 variants (shape × FF): normalisation, implied fluxes, tail fractions, window counts, N_lo.
- `cevns_spectra.csv` — dR/dE_R for all variants on the 0.5–330 keV grid.
- `weak_form_factor.csv` — Helm vs WimPyDD natural-Xe weak F²(E).
- `fig5_green_digitised.csv`, `fig5_panel_comparison.csv` — digitised green histograms; predicted vs observed panel sums.
- `incoherent_vs_coherent.csv` — Llewellyn-Smith NC elastic vs coherent cross-sections.
- `exotic_Nlo.csv` — N_lo for each exotic interaction.
- Figures: `figures/P019_cevns_spectrum_and_formfactor.png` (left: CEνNS spectra for all flux shapes, WimPyDD variant, efficiency-weighted central, event window; right: weak F² Helm vs shell model with the 248 keV line); `figures/P019_fig5_green_digitised.png` (digitised green histograms in the three panels with the predicted per-bin average); `figures/P019_exotic_shapes.png` (normalised recoil spectra for SM, magnetic moment, heavy scalar and dipole HNL m₄ = 50–240 MeV); `figures/P019_coherent_vs_incoherent.png` (σ per nucleus vs E_ν).

## 10. Tools and provenance (mirrors `output/provenance/P019.json`)

- Agent tools: Read ×12 (PAPER_GUIDE, dossier, ledger, P004.md, P010.md, fulltext.tex l.170–250 and 253–325, Fig5 PNG, lzcommon.py ×3 ranges, P004_wall_mssi.py l.70–125, P004_fig5_digitised.csv, four P019 figures), Bash ×8 (listings/greps; WimPyDD test ×2; script run ×2; table print), Write ×4 (script, details, provenance, paper), Edit ×9 (script fixes).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (integrate.quad, stats.norm, optimize); pandas 3.0.5; matplotlib 3.11.2; nestpy 2.1.1 (via lzcommon nest_nr_yields/nest_er_yields); WimPyDD 2.0.4 (eft_hamiltonian, streamed_halo_function, diff_rate via lzcommon wd_*); Pillow 12.3.0 (figure digitisation); common/lzcommon.py (LZ constants, helm_F2, XE_ISOTOPES, m_nucleus_gev, nest yields, WimPyDD wrappers).
- Recalled knowledge (18 items, see §2 and JSON).
- Datasets: none. Data requests: none. WimPyDD-generated files: none (diff_rate only).
