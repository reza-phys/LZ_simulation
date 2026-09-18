# P017 — Xenon nuclear response at q ≈ 250 MeV: how form-factor nodes and isotope mixtures shape the inferred couplings

*Research record. Simulated date 2026-09-05. Category NUC (nucl-th, cross-list hep-ph). Author profile: nuclear-structure theorists working on dark-matter response functions. All numbers below are printed by `output/code/P017_nuclear_response.py` (log: `output/work/P017/run_log.txt`) unless marked [paper], [corpus] or [recalled].*

## 1. Motivation and framework

The LZ event (arXiv:2609.02823) is interpreted as a 248 ± 23 (stat) ± 23 (sys) keV nuclear recoil. Its momentum transfer, q = √(2 m_N E_R) = 246 MeV for the mean xenon mass, is unusually large for direct detection and sits where the spin-independent (M) nuclear response of xenon has its second diffraction minimum. P002 and P003 [corpus] already noted that WimPyDD's isotope-summed M response has a node at ≈265–266 keV and that WimPyDD's SI rate falls ×3–6 below the Helm approximation at 200–250 keV (process log step 036). LZ computes all signal spectra with WimPyDD using the one-body density matrices distributed with DMFormFactor-v6 "with modifications as described in" the 2023 LZ NREFT paper [paper, Theory paragraph]. Because a single event at fixed energy fixes the product coupling² × response(q), any uncertainty in the response at q ≈ 246 MeV maps one-to-one onto the inferred couplings. We quantify: (i) the M response per isotope in three descriptions (shell model as in WimPyDD, Helm, two-parameter Fermi), the node positions and their isotope spread; (ii) the spin responses Σ′, Σ″ (and Δ, Φ″) of ¹²⁹Xe and ¹³¹Xe, including the q⁴-weighted L10 kernel; (iii) the rate in the ±23 keV window and its dependence on the response description for O1 (elastic and δ = 300 keV inelastic), O4, O6 and L10; (iv) the resulting nuclear-structure band on LZ's two-sided intervals.

**Framework (Fitzpatrick et al. 2013; Anand, Fitzpatrick, Haxton 2014).** The NREFT differential rate factorises as dR/dE ∝ Σ_{ττ′} Σ_X R_X^{ττ′}(v⊥², q²) W_X^{ττ′}(q), with X ∈ {M, Σ′, Σ″, Δ, Φ″, Φ̃′} and interference terms (Φ″–M, Δ–Σ′). WimPyDD exposes, per isotope, the tabulated W_X^{ττ′}(q) as `func_w(q_GeV)[X, τ, τ′]` (8 entries: M, Σ″, Σ′, Φ″, Φ̃′, Δ, Φ″–M, Δ–Σ′). We verified the normalisation: W_M^{00}(0) = A²(2J+1)/(16π) to four digits for all seven active isotopes (e.g. ¹³²Xe 346.62 vs 346.64; ¹³¹Xe 1365.52 vs 1365.63), so WimPyDD's isoscalar index 0 is (c_p + c_n) as settled by P003 and the tables carry the (2J+1) factor. **WimPyDD has no density matrices for ¹²⁴Xe and ¹²⁶Xe** (all responses identically zero; 0.18 % of natural xenon); every isotope sum below is over the seven isotopes 128–136 (99.8 %).

Operator ↔ response (isoscalar couplings, spin-1/2 WIMP): O1 → M; O4 → Σ′ + Σ″ (equal weights); O6 → q⁴ Σ″; the L10 magnetic-tensor Lagrangian reduces at leading order to ∝ (q²/m_N²) O4 − O6, a pure q⁴ Σ′ response [P003, corpus; recalled likely]. O4, O6 and this L10 combination are velocity-independent, so at fixed E_R the rate under an alternative response is exactly the WimPyDD rate rescaled by the ratio of the normalised responses (Sec. 4).

## 2. Inputs

| Input | Value | Source |
|---|---|---|
| Event energy, resolution | 248 ± 23 (stat) ± 23 (sys) keV | LZ paper [paper] |
| Window | 225–271 keV (±1σ stat); Gaussian smearing σ = 23 keV as alternative | this work |
| Isotope masses, abundances | WimPyDD `Xe.mass`, `Xe.abundance` (¹²⁹Xe 26.40 %, ¹³¹Xe 21.23 %, ¹³²Xe 26.91 %, …) | WimPyDD 2.0.4 |
| Shell-model responses | `WD.Xe.func_w` (DMFormFactor-v6 one-body density matrices as shipped with WimPyDD) | WimPyDD |
| Helm form factor | Lewin–Smith: c = 1.23A^{1/3} − 0.60 fm, a = 0.52 fm, s = 0.9 fm (`lzcommon.helm_F2`) | corpus library |
| 2pF density | ρ ∝ [1 + exp((r − c)/a)]⁻¹, c = 5.42 fm, a = 0.57 fm for ¹³²Xe, c ∝ A^{1/3} | **recalled, likely** (charge-density compilation) |
| Charge radius ¹³²Xe | ≈ 4.79 fm | **recalled, likely** (Angeli–Marinova compilation) |
| ⟨S_n⟩, ⟨S_p⟩ | ¹²⁹Xe 0.329, 0.010; ¹³¹Xe −0.272, −0.009 | **recalled, likely** (Menéndez, Gazit, Schwenk 2012) |
| Halo | Baxter-2021 SHM, time-averaged (Sun frame) via `lz.wd_halo()`; P007 found this reproduces LZ's intervals best | corpus |
| Unit coupling | c_i^s = 1/m_v² (Anand) = WimPyDD c⁰ = 2/m_v², m_v = 246.2 GeV | P003, P007 [corpus] |
| Efficiency model (full-ROI integrals) | 0.96 plateau, erf roll-offs to 50 % at 5.4 keV (σ 2.5) and 269.9 keV (σ 11.5) | P003/P007 [corpus] |
| LZ two-sided intervals, O1^s inelastic 1000 GeV | `output/work/P007/lz_intervals_digitised.csv` | P007 [corpus] |
| LZ two-sided interval, L10^s vs mass | own PyMuPDF digitisation of Fig. 6 bottom (drawings 349 upper, 350 lower, 346 median; x ticks 10¹/10²/10³ GeV at 73.1/231.9/390.7 pt, y decades 10⁻¹…10² at 629.1/536.6/444.0/351.4 pt) | LZ source PDF |
| P003 L10 normalisation | LZ's L10^s curve = 247.5 × [(q²/m_N²)O4 − O6] at A = 1/m_v², i.e. 61.9 × at our A = 2/m_v² | P003 [corpus] |

## 3. Part 1 — the M response

### 3.1 Momentum transfer
q(248 keV) = 246.4 MeV for ⟨A⟩ = 131.29; per isotope 243.1 (¹²⁸Xe) … 250.6 MeV (¹³⁶Xe). The ±23 keV window spans q = 234.7–257.6 MeV.

### 3.2 Form factors
For each isotope i we form the normalised responses F²_i(E_R) = W_M^{00}(q_i)/W_M^{00}(0) with q_i = √(2 m_i E_R), the Helm |F|² at the same q_i, and the 2pF form factor F(q) = ∫ρ(r) j₀(qr) r² dr / ∫ρ r² dr computed on a 4001-point radial grid to 20 fm. The natural-xenon response is the abundance × A² weighted mean (the weight of each isotope in the isoscalar SI rate at fixed E_R). Curves are stored in `P017_response_curves.npz` and plotted in Fig. 1.

### 3.3 Nodes (`P017_M_nodes.csv`, Fig. 4)

| description | node 1 (nat. Xe) | node 2 (nat. Xe) | node 2 in q | node-2 spread ¹²⁸Xe→¹³⁶Xe | node 2 for ¹²⁹/¹³¹/¹³²Xe |
|---|---|---|---|---|---|
| shell model (WimPyDD) | 102.0 keV | **266.9 keV** | 255.6 MeV | 277.6 → 251.1 keV | 276.0 / 268.4 / 263.3 keV |
| Helm | 94.3 keV | 278.9 keV | 261.3 MeV | 290.8 → 262.5 keV | 287.0 / 279.7 / 276.1 keV |
| 2pF (c = 5.42, a = 0.57) | 103.3 keV | 308.8 keV | 274.9 MeV | 321.7 → 291.2 keV | 317.6 / 309.7 / 305.8 keV |

The shell-model node at 266.9 keV agrees with P002 (266 keV) and P003 (≈265 keV). The event's central energy is 19 keV (0.8σ_stat) below it. Isotope mixing spreads the node over 26 keV (∝ A^{−5/3}: heavier isotopes have larger radii and smaller q at fixed E_R) and fills it: the minimum of the natural-Xe shell response is 3.5 × 10⁻⁶ versus 1.5 × 10⁻⁷ for ¹³²Xe alone (×23 shallower), but it remains a factor ≈ 400 below the secondary maximum at ≈150 keV. At 248 keV the natural-Xe shell F² is 2.9 × 10⁻⁵ (¹³²Xe alone 1.65 × 10⁻⁵).

### 3.4 Ratios at fixed E_R (`P017_M_ratios.csv`)

| E_R (keV) | q (MeV) | F²_shell | F²_Helm | F²_2pF | shell/Helm | 2pF/Helm |
|---|---|---|---|---|---|---|
| 50 | 110.6 | 5.47e-2 | 5.48e-2 | 6.76e-2 | 0.998 | 1.23 |
| 100 | 156.5 | 6.0e-5 | 2.48e-4 | 1.0e-4 | 0.243 | 0.40 |
| 150 | 191.6 | 1.29e-3 | 3.41e-3 | 2.54e-3 | 0.378 | 0.74 |
| 200 | 221.3 | 5.05e-4 | 1.51e-3 | 1.79e-3 | 0.335 | 1.19 |
| 225 | 234.7 | 1.67e-4 | 6.07e-4 | 9.88e-4 | 0.275 | 1.63 |
| **248** | **246.4** | **2.9e-5** | **1.64e-4** | **4.50e-4** | **0.177** | **2.73** |
| 265 | 254.7 | 4e-6 | 3.2e-5 | 2.03e-4 | 0.114 | 6.25 |
| 270 | 257.1 | 4e-6 | 1.6e-5 | 1.53e-4 | 0.244 | 9.3 |
| 300 | 271.0 | 3.8e-5 | 4.3e-5 | 1.0e-5 | 0.90 | 0.23 |

Shell/Helm = 1.00 at 50 keV (so all three descriptions share the same normalisation and rms-radius regime) but 0.34 at 200, 0.27 at 225, 0.18 at 248 and 0.11 at 265 keV — this is the ×3–6 of process-log step 036, and it is entirely a node-position effect. If the true recoil energy were 262–266 keV (P009's Table-S5 reading), F²_shell would be another factor 7 lower than at 248 keV.

### 3.5 Robustness: radius versus shape (`P017_robustness_132Xe.csv`, ¹³²Xe)

| variant | rms (fm) | node 2 (keV) | F²(248) |
|---|---|---|---|
| shell model | – | 263.3 | 1.7e-5 |
| Helm s = 0.8 / 0.9 / 1.0 fm | 4.79 | 269.4 / 276.1 / 284.0 | 1.0e-4 / 1.3e-4 / 1.6e-4 |
| 2pF c = 5.42, a = 0.57 | 4.70 | 305.8 | 4.1e-4 |
| 2pF c = 5.551 (rms matched to Helm) | 4.79 | 292.1 | 2.5e-4 |
| 2pF a = 0.50 / 0.65 | 4.59 / 4.84 | 308.6 / 302.6 | 6.2e-4 / 2.5e-4 |

The second node is not fixed by the rms radius: at equal rms (4.79 fm) the 2pF node lies 16 keV above Helm's, and the shell-model node is 13 keV below Helm's. Beyond the first node the response is controlled by the surface shape (and, for the shell model, by the point-nucleon density and orbital occupations), not by ⟨r²⟩. The recalled 2pF parameters give an rms 2 % below the recalled charge radius; we flag the 2pF curve as a shape illustration, not as a competing precision prediction.

## 4. Part 2 — spin responses of ¹²⁹Xe and ¹³¹Xe (`P017_spin_responses.csv`, Fig. 2)

Proton/neutron combinations from WimPyDD's isospin indices (c⁰ = c_p + c_n, c¹ = c_p − c_n): W_pp = W⁰⁰ + W⁰¹ + W¹⁰ + W¹¹, W_nn = W⁰⁰ − W⁰¹ − W¹⁰ + W¹¹.

**q → 0 checks.** Σ′/Σ″ = 1.975 (¹²⁹Xe), 1.898 (¹³¹Xe) (expected 2 for the transverse/longitudinal split). W_nn/W_pp = 1162 and 1505, i.e. |⟨S_n⟩/⟨S_p⟩| = 34 and 39 (recalled Menéndez-2012 values give 33 and 30), with ⟨S_p⟩⟨S_n⟩ > 0 in both. The neutron Σ″(0) ratio ¹²⁹Xe/¹³¹Xe with (2J+1) removed is 2.73, versus (J+1)/J ⟨S_n⟩² from the recalled spins 2.63 — the shipped density matrices are consistent with the Menéndez–Gazit–Schwenk spin expectation values at the 4 % level.

**q dependence.**

| isotope | response | minima below 420 keV | W(248)/W(0) | W(150)/W(0) |
|---|---|---|---|---|
| ¹²⁹Xe | Σ′ (isoscalar 00) | 90, 375 keV | 6.9e-3 | 4.2e-3 |
| ¹²⁹Xe | Σ′ (nn) | 75, 110, 401 keV | 6.5e-3 | 2.4e-3 |
| ¹²⁹Xe | Σ″ (00 and nn) | none | 1.3e-2 | 1.7e-2 |
| ¹³¹Xe | Σ′ (00) | 36, 292 keV | 9.8e-3 | 4.8e-2 |
| ¹³¹Xe | Σ″ (00) | none | 6.4e-2 | 2.3e-1 |

Neither Σ″ has a node below 420 keV; ¹²⁹Xe's Σ′ node lies at 375 keV and ¹³¹Xe's at 292 keV — 21 keV above the ±1σ window and within 2σ (stat) of the event. The q⁴-weighted L10 kernel q⁴W_Σ′^{00} of ¹²⁹Xe peaks at 233 keV (¹²⁹Xe nn: 241 keV), i.e. essentially at the event energy (Fig. 2, right); ¹³¹Xe's kernel has a broad maximum near 180 keV and a dip at 292 keV. All isoscalar responses at 248 keV relative to q → 0 (abundance-weighted natural Xe): M 2.9 × 10⁻⁵, Σ′ 7.2 × 10⁻³, Σ″ 2.4 × 10⁻², Δ 4.5 × 10⁻³, Φ″ 4.2 × 10⁻⁴ (`P017_all_responses_at_248.csv`): the M response is suppressed 250–800 × more strongly than the spin responses at the event.

**Isotope shares at 248 keV** (`P017_isotope_shares_248.csv`, from `WD.diff_rate(..., isotopes_list={0:[i]})`): L10: ¹²⁹Xe 70 %, ¹³¹Xe 30 % (68 % at 50 keV, 23 % at 150 keV — the share swings with the ¹²⁹Xe Σ′ node at 90 keV); O4: 52/48; O6: 40/60; O1 elastic: ¹²⁹Xe 49 %, ¹³¹Xe 24 %, ¹³²Xe 16 %, ¹³⁶Xe −0.1 % (WimPyDD's interpolated ¹³⁶Xe and ¹²⁸Xe M responses cross slightly below zero at their nodes, F² ≈ −1 × 10⁻⁶; a negligible interpolation artefact, noted for the record).

## 5. Part 3 — rates in the ±23 keV window (`P017_window_rates.csv`, Fig. 3, Fig. 5)

Method: for each Hamiltonian (O1^s, O4^s, O6^s at c⁰ = 2/m_v²; L10-like (q²/m_N²)O4 − O6 at A = 2/m_v²) and each active isotope, dR_i/dE from `WD.diff_rate` at 1000 GeV (2 keV steps, 2–420 keV); the alternative-response spectrum is dR_i/dE × F²_alt(q_i)/F²_shell,i(q_i) with F²_shell,i the normalised isoscalar response of the operator's own channel (M; Σ′ + Σ″; Σ″; Σ′). For velocity-independent operators this is exact. **Validation:** the Helm-reweighted WimPyDD O1 spectrum reproduces `lzcommon.dRdE_SI` (independent Helm implementation with σ_n = c_n² μ_n²/π = 2.96 × 10⁻³⁸ cm² for c_n = 1/m_v², 1000 GeV) to 0.999–1.002 at 10–225 keV and 0.978/0.995 at 248/270 keV (`P017_validation_helm.csv`); the raw shell/Helm rate ratio is 0.984 (10 keV), 0.998 (50), 0.237 (100), 0.379 (150), 0.336 (200), 0.275 (225), 0.176 (248), 0.258 (270).

Window integral R (225–271 keV, events/(t·yr), unit coupling, 1000 GeV) and ratios alt/shell:

| case | R_shell | Helm-shaped/shell | 2pF or thin-shell/shell | smeared (σ = 23 keV): Helm, alt | dR/dE max/min over 230–270 keV (shell) |
|---|---|---|---|---|---|
| O1 elastic | 3.53 × 10⁴ | 4.47 | 10.2 (2pF) | 3.40, 5.55 | **40.7** (×4.7 at 230, ×0.13 at 270 keV) |
| O1 inelastic δ = 300 keV | 187 | 4.56 | 10.8 (2pF) | 3.48, 6.14 | **29.5** |
| L10 | 1.52 × 10⁻² | 0.038 | 1.64 (thin shell j₀²) | 0.054, 1.58 | 1.81 |
| O4 | 9.05 | 0.023 | 0.88 (thin shell) | 0.035, 0.75 | 1.95 |
| O6 | 2.69 × 10⁻² | 0.011 | 0.48 (thin shell) | 0.015, 0.45 | 1.24 |

Reading: for O1 the inferred coupling² would be ×4.5 (Helm) to ×10 (2pF) smaller had LZ used a phenomenological form factor, and even the smeared rate changes ×3.4–6; the shell-model dR/dE itself varies by a factor 30–40 across the ±1σ window, so a ±23 keV shift of the assumed energy moves the O1 coupling² by an order of magnitude. For the spin operators the shell-model spectrum varies by only ×1.2–1.9 across the window. The "Helm-shaped spin form factor" rows are a deliberate strawman (older analyses sometimes assumed the spin form factor follows the density): imposing the M-node at 279 keV on a spin response reduces the window rate ×25–90 — it shows what is *not* happening in the shell model, whose Σ′/Σ″ responses have no node near the event. The Lewin–Smith thin-shell j₀²(q r_n) proxy differs from the shell model by ×0.5–1.6.

## 6. Part 4 — propagation to LZ's intervals

### 6.1 O1^s inelastic (P007 digitised intervals; `P017_O1_interval_propagation.csv`)

Coupling² factor f = R_shell/R_alt: with the window, 0.22 (Helm) / 0.09 (2pF) at δ = 250–350 keV; smeared, 0.29 / 0.16; efficiency-weighted full ROI, 0.38/0.40 (δ = 250), 0.35/0.33 (300), 0.30/0.18 (350 keV). The full-ROI factor is the closest proxy to what a PLR that fits the whole spectrum sees, because with 23 keV resolution the node is partly filled and the low-energy shoulder of the inelastic spectrum contributes. Applying the full-ROI Helm factor to P007's digitised intervals at 1000 GeV:

| δ (keV) | LZ (c₁^s m_v²)² lower–upper (shell, as published) | if the true M response were Helm-like |
|---|---|---|
| 250 | 2.9e-6 – 3.9e-5 | 1.1e-6 – 1.5e-5 |
| 300 | 1.28e-5 – 2.60e-4 | 4.5e-6 – 9.0e-5 |
| 350 | 1.38e-3 – 2.25e-2 | 4.1e-4 – 6.7e-3 |

Unit-coupling shell-model counts in the ROI (2.84 t·yr): 9.97 × 10⁴, 1.26 × 10⁴, 154 at δ = 250/300/350 keV (P007: 1.04 × 10⁵, 1.35 × 10⁴, 259 with its own efficiency model — consistent at 5–40 %, the largest difference at 350 keV where the spectrum is confined to the roll-off region).

We therefore quote a nuclear-structure band on the O1 inelastic coupling² of a factor ≈ 3 (full-spectrum) to ≈ 5 (±1σ window), one-sided towards smaller couplings if the shell-model node is too deep or too low in energy, and two-sided by a factor ≈ 7 if the recoil energy is 265 rather than 248 keV.

### 6.2 L10^s (own digitisation of Fig. 6 bottom; `P017_L10_interval_digitised.json`, `P017_L10_interval_propagation.csv`)

The bottom y-axis label tokens are "d^s_10 (dimensionless)" (top panel: "(c^s_1 m_v²)²"), so the bottom axis is the *linear* coupling d₁₀^s m_v². Digitised values: 1000 GeV upper 0.553, median sensitivity 0.375, lower 0.154; 200 GeV 0.39/0.27/0.10; 400 GeV 0.41/0.28/0.11; 4000 GeV 1.03/0.71/0.29; the lower limit lifts off zero from 101 GeV upwards. Consistency: with P003's normalisation (LZ's L10^s = 61.9 × our combination at A = 2/m_v²) the unit-coupling ROI count at 1000 GeV is 12.9 events, so the interval edges correspond to 0.31–3.9 events (median sensitivity 1.8), matching a one-event PLR interval (≈ 0.1–3.7) — under the squared-axis reading they would be 2.0–7.1 events, which is excluded. The dossier's "d₁₀ ≈ 0.1–0.5" [corpus] is thus (d₁₀ m_v²) ∈ [0.15, 0.55].

Nuclear-structure factors on d₁₀² (shell/alt): thin-shell proxy 0.87 (full ROI), 0.61 (window), 0.63 (smeared); Helm-shaped strawman 2.85 (full ROI), 26 (window). On the linear coupling the thin-shell proxy moves the interval to 0.14–0.52 (from 0.15–0.55), i.e. ±7 %; even the strawman only moves it to 0.26–0.93.

### 6.3 O4 and O6
From the window ratios (Sec. 5) the thin-shell proxy changes the inferred coupling² by ×1.14 (O4) and ×2.1 (O6); the strawman by ×44 and ×90 — again a statement about what a node *would* do, not about the shell-model uncertainty.

## 7. Discussion

1. **The event sits 19 keV below the shell-model M node (267 keV), inside its isotope-spread (251–278 keV).** Any SI-like (M-response) interpretation therefore extracts a coupling from a response that varies by ×40 across the ±1σ energy window and by ×3–10 between response descriptions at fixed energy. This is the nuclear-physics counterpart of P002's kinematic observation that the O1 inelastic spectrum has its minimum where the event is.
2. **The spin responses are smooth there.** Σ″ has no node below 420 keV; ¹²⁹Xe Σ′ (70 % of the L10 rate at 248 keV) has its node at 375 keV; the q⁴Σ′ kernel of ¹²⁹Xe peaks at 233 keV. Couplings extracted for L10, O4, O6 (and by extension O9, O10, O14 which share Σ′/Σ″) are robust at the ±10–15 % (shape) level against the phenomenological alternatives we could test. This complements P003's shape argument (operator selection by the absence of low-energy events) with a robustness argument.
3. **Which node is right?** Helm and shell model agree to 0.2 % at 50 keV but place the second node 12 keV apart (279 vs 267 keV); the 2pF shape puts it above 290 keV even at matched rms. Elastic electron scattering on xenon isotopes (the natural arbiter, since the charge form factor's diffraction minima are measured to ≲1 % in q) is, to our recollection, sparse for xenon (recalled, uncertain); the isoscalar point-nucleon density needed here also involves the neutron distribution, for which the shell model with a phenomenological interaction is the only available guide. The ¹³²Xe shell-model node (263 keV) versus Helm (276 keV) corresponds to a 5 % difference in the effective diffraction radius.
4. **Two-body currents and chiral corrections (recalled).** For the SI/M response, chiral two-body scalar currents and the nucleon radius correction modify the coherent response at the few-per-cent level at q → 0 with a q² dependence that grows towards 200–250 MeV (Hoferichter, Klos, Menéndez, Schwenk 2016–2018) — a 10–30 % effect at the event's q is plausible (**recalled, uncertain**), and it shifts the *depth and position* of the node only through the q-dependent isovector/isoscalar mixing. For the spin responses, two-body axial currents reduce the neutron-dominated structure factors by ≈ 10–30 % and generate a proton-like contribution (Klos et al. 2013; **recalled, likely**); these are smooth in q and change couplings by ≲ 15 %, comparable to our shape band. Neither effect is in the one-body density matrices WimPyDD ships.
5. **LZ's modifications.** LZ states that the DMFormFactor-v6 density matrices were "modified as described in" its 2023 NREFT paper. We cannot reproduce that modification; if it affects the ¹²⁸–¹³⁶Xe M responses near the node (e.g. a re-fitted oscillator parameter or corrected isotopes), LZ's own node may differ from WimPyDD's shipped 267 keV by several keV — a direct test is to compare Fig. 1's O1 inelastic dip (P002/P003 read 266 keV from the figure, consistent with the shipped tables to ≤ 1 keV).
6. **Recommendation.** Report SI/O1-type couplings from a 248 keV event with an explicit nuclear-response systematic (factor ≈ 3 on coupling²), or better, report the fitted *event count* per model (as LZ does for L10: 1.0 (+1.4, −0.7)) alongside the coupling; and prefer operators whose response is smooth at q ≈ 250 MeV when quoting couplings from this event.

## 8. Failed or abandoned approaches
- Initial `isotopes_list=[...]` calls to `WD.diff_rate` failed (it expects a dict `{element_index: [isotope indices]}`); fixed.
- The response-ratio re-weighting produces a numerical spike at the ¹²⁹Xe Σ′ node (375 keV) for the thin-shell proxy (ratio of two near-zero interpolations); outside the ROI, so Fig. 3 is restricted to 350 keV; window results are unaffected.
- A first reading of the Fig. 6 bottom axis as (d₁₀ m_v²)² was rejected by the event-count consistency check and by the label tokens.

## 9. Figures
- `figures/P017_fig1_M_response.png` — normalised isotope-weighted M response vs E_R: shell model, Helm, 2pF, plus ¹²⁹Xe and ¹³⁶Xe shell curves; lower panel shell/Helm and 2pF/Helm ratios; event window shaded.
- `figures/P017_fig2_spin_responses.png` — neutron and proton Σ′, Σ″ of ¹²⁹Xe/¹³¹Xe normalised at q → 0, with Helm and thin-shell proxies; right: q⁴W_Σ′ L10 kernel.
- `figures/P017_fig3_spectra.png` — dR/dE (unit coupling, 1000 GeV) for O1 elastic, O1 δ = 300 keV, L10-like, under shell model / Helm-shaped / 2pF or thin-shell responses, and the 23 keV-smeared shell spectrum.
- `figures/P017_fig4_nodes.png` — first and second M-node energies per isotope and description.
- `figures/P017_fig5_window_ratios.png` — 225–271 keV rate ratios alt/shell for the five cases.

## 10. Result tables
`P017_M_nodes.csv`, `P017_M_ratios.csv`, `P017_robustness_132Xe.csv`, `P017_spin_responses.csv`, `P017_all_responses_at_248.csv`, `P017_isotope_shares_248.csv`, `P017_window_rates.csv`, `P017_validation_helm.csv`, `P017_O1_interval_propagation.csv`, `P017_L10_interval_digitised.json`, `P017_L10_interval_propagation.csv`, `P017_response_curves.npz`, `P017_summary.json`, `run_log.txt`.

## 11. References
1. LZ Collaboration, arXiv:2609.02823 (2026) — Theory paragraph, Fig. 1, Fig. 6 (source PDF `Fig6_O1_L10_limit_stacked.pdf`).
2. A. L. Fitzpatrick, W. Haxton, E. Katz, N. Lubbers, Y. Xu, JCAP 02 (2013) 004.
3. N. Anand, A. L. Fitzpatrick, W. C. Haxton, Phys. Rev. C 89, 065501 (2014) (DMFormFactor).
4. I. Jeong, S. Kang, S. Scopel, G. Tomar, Comput. Phys. Commun. 276, 108342 (2022) (WimPyDD).
5. J. D. Lewin, P. F. Smith, Astropart. Phys. 6, 87 (1996) (Helm parametrisation, thin-shell SD form factor).
6. J. Menéndez, D. Gazit, A. Schwenk, Phys. Rev. D 86, 103511 (2012); P. Klos, J. Menéndez, D. Gazit, A. Schwenk, Phys. Rev. D 88, 083516 (2013) (xenon spin structure factors, two-body currents).
7. M. Hoferichter, P. Klos, J. Menéndez, A. Schwenk, Phys. Rev. D 94, 063505 (2016) (chiral SI responses).
8. Corpus: P000 dossier; P002; P003; P007; P009.

## 12. Tools and provenance (mirrors `output/provenance/P017.json`)
- Agent tools: Read (PAPER_GUIDE.md; 00_evidence_dossier.md; results_ledger.csv; P002.md; P003.md; fulltext.tex lines 57–73 and 280–295; lzcommon.py sections 2, 5, 8; P007 lz_intervals_digitised.csv; P007_higgsino_inelastic.py lines 110–140; five/seven PNG figure checks), Bash (grep of process_log step 036 and WimPyDD test pattern; directory listings; WimPyDD API probes ×3; PyMuPDF inventory of Fig. 6; grep of WimPyDD isotopes_list; two full script runs), Write (script, details.md, P017.json, P017.md), Edit (script ×5), Skill (dataviz).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (integrate.trapezoid, optimize.brentq, special.erf); pandas 3.0.5; matplotlib 3.11.2; pymupdf 1.28.2 (get_drawings, get_text); WimPyDD 2.0.4 (Xe.func_w, nuclear_current, eft_hamiltonian with q-dependent coefficients, diff_rate with isotopes_list, streamed_halo_function via lzcommon); common/lzcommon.py (wd, wd_halo, wd_hamiltonian, wd_c_from_anand, helm_F2, dRdE_SI, mu_red, constants).
- Script: `output/code/P017_nuclear_response.py` — `.venv/bin/python output/code/P017_nuclear_response.py` (≈ 50 s).
- Recalled knowledge (6 items): 2pF parameters for ¹³²Xe (likely); ¹³²Xe charge radius ≈ 4.79 fm (likely); Menéndez-2012 ⟨S_n⟩, ⟨S_p⟩ (likely); L10 → (q²/m_N²)O4 − O6 (likely, from P003); two-body-current sizes for SI and SD responses (uncertain / likely); sparse electron-scattering data on Xe (uncertain).
- Datasets: none. Data requests: none. WimPyDD-generated files: none (diff_rate used; no response-function files written).
