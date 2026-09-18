# P057 — Operator interference and the degeneracy among L2, L4, L9–L12 and L16: what a second event (and a tenth) would distinguish (research record)

Simulated date 2026-09-11. Author profile: direct-detection model-discrimination phenomenologists. Category EFT (hep-ph).
Script: `output/code/P057_operator_degeneracy.py` (run from `/Users/reza/LZ_simulation` with `.venv/bin/python`). Outputs in `output/work/P057/`.

## 1. Motivation and framework

LZ (arXiv:2609.02823) fits its single 248 keV recoil equally well (3.0–3.4σ local) with L2, L4, L6^v, L9–L12, L16, L18–L20, O4 and inelastic O1/O4 at δ ≥ 300 keV (dossier §4.7; Table S6/S7). The LEE section (tex lines 494–498) states that L1/L5, L2/L8, L3/L17, L4/L20 and L11/L14 "differ only by a scalar constant", that the isoscalar and isovector spectra of L2, L4, L7, L8, L10, L11, L14, L15, L19, L20 and inelastic O4 are indistinguishable, and that all masses ≥ 400 GeV are nearly degenerate, reducing 616 models to 293 spectra. P003 showed that only q²-suppressed spin operators (O6, O10, O9, O14, O15, O5^v, L10) are compatible with the absence of a low-energy population (N_lo ≤ 5), P027 that the posterior is spread over 65–99 effective models, P044 that L16 must be a companion-free q⁴/q⁶-spin structure, P050 that L10 and inelastic δ = 366 keV separate with 4–6 events, P047 that isotopes cannot tell spin operators apart.

We ask, at the level of the nonrelativistic operators themselves: (i) which pairs are exactly or effectively degenerate in xenon (a power of q, a velocity factor, an isospin flip); (ii) how many events separate every pair at 3σ with LZ's resolution and either LZ's 600 phd edge or a 1000 phd edge; (iii) whether two-operator interference can hide an SI-like (O1) or SD-like (O4) admixture in a lone high-energy event; (iv) what a second event's energy would tell.

### 1.1 Nuclear-response structure (recalled, certain: Fitzpatrick et al. 2013 eq. 40 / Anand et al. 2014 eq. 38–40)

For spin-1/2 DM, the squared amplitude is a sum over nuclear responses with coefficient functions

- R_M = c1² + (j(j+1)/3)[(q²/m_N²) v⊥² c5² + v⊥² c8² + (q²/m_N²) c11²]
- R_Σ″ = (q²/4m_N²) c10² + (j(j+1)/12)[c4² + 2(q²/m_N²) c4c6 + (q⁴/m_N⁴) c6² + v⊥² c12² + (q²/m_N²) v⊥² c13²]
- R_Σ′ = (1/8)[(q²/m_N²) v⊥² c3² + v⊥² c7²] + (j(j+1)/12)[c4² + (q²/m_N²) c9² + (v⊥²/2)(c12 − (q²/m_N²) c15)² + (q²/2m_N²) v⊥² c14²]
- R_Δ = (j(j+1)/3)[(q²/m_N²) c5² + c8²]; R_ΔΣ′ = (j(j+1)/3)[c5c4 − c8c9]; R_Φ″, R_Φ″M, R_Φ̃′ as in Anand et al.

Consequences that we test numerically: O6 ∝ q⁴Σ″ and O10 ∝ q²Σ″ share the nuclear response and are velocity-independent, so dR/dE(O6)/dR/dE(O10) ∝ q² ∝ E_R exactly (not a constant). O9 ∝ q²Σ′ and O14 ∝ q² v⊥² Σ′ differ by the mean v⊥² of the scattering WIMPs, a slowly varying function of E_R. L10 → 4[(q²/m_N²)O4 − O6] (P003/P012): in R_Σ″ the square (c4 + (q²/m_N²)c6)² vanishes identically, leaving c4² Σ′ ∝ q⁴Σ′; O9 × q² is q⁶Σ′, so L10/(O9×q²) ∝ 1/E_R. Interference: O1 (M) and O6 (Σ″) never interfere; O4 and O6 interfere through the 2(q²/m_N²)c4c6 Σ″ cross term; O6/O10 and O9/O14 do not (no cross terms). WimPyDD's `coeff_squared_list` confirms the bookkeeping: {O1,O6} → (1,1),(6,6); {O4,O6} → (4,4),(4,6),(6,4),(6,6); {O6,O10} → (6,6),(10,10); {O9,O14} → (9,9),(14,14); {O4,O5} → (4,4),(5,4),(5,5); {O8,O9} → (8,8),(8,9),(9,9).

## 2. Inputs

| input | value / source |
|---|---|
| WimPyDD 2.0.4 spectra, natural xenon, shell-model responses | `lz.wd_rate` (`WD.diff_rate`), `lz.wd_halo()` Sun-frame Baxter-2021 SHM (v0 = 238, v_esc = 544, v_E = 250.6 km/s; explicit v_min grid, P002 fix). Labelled "Sun frame" per P035. |
| coupling convention | WimPyDD c⁰ = c_p + c_n = 2/m_v² for LZ's unit coupling (P003); isovector c¹ = 2/m_v². Shapes are coupling-independent. |
| true-energy grid | 1.25–598.75 keV in 2.5 keV steps (240 points) |
| efficiency, LZ edge | 0.96 × ½[1+erf((E−5.4)/(√2·3.4))] × ½[1−erf((E−269.9)/(√2·8.0))] (P003 model; paper: 50 % at 5.4 and 269.9 keV, 96 % plateau) |
| efficiency, extended edge | same with E50 = 423.2 keV (P038: 1000 phd), roll-off σ = 12.5 keV (assumption: 8 keV scaled by 423/270) |
| resolution | σ_E = 11 √(E/248) keV (P009/P021), Gaussian, true → observed on a 1 keV observed grid to 700 keV |
| annual-average inelastic spectra | `output/work/P050/P050_spectra_cache.npz` (O1^s, 1 TeV, 12-day average Baxter halo, 3 keV grid to 800 keV), re-gridded |
| window definitions | N_lo = R(5.4–55)/R(200–270 keV) (P003, LZ-edge efficiency, true energy); R(100–200)/R(200–270); R(270–423)/R(200–270) with extended-edge efficiency |
| KL floor | q_f = (1−w) q + w·uniform(E_obs grid), w = 10⁻² (P050) and 10⁻³ |
| 2024 low-energy tolerance | N_max = 3, 5 (recalled, uncertain; P003/P012) and 1.6–2.2 (P016's fitted tolerance) |
| second-event date information | P006/P034 (not recomputed): date LR 1.41/2.08/2.55 for δ = 300/350/380 keV, 0.98 for elastic |

Model set at 1 TeV: O1^s, O4^s (excluded references), O6^{s,v}, O9^{s,v}, O10^{s,v}, O14^{s,v}, O15^{s,v}, O5^v, O13^s, L10, O9×q²/m_v² ("O9q2"), inelastic O1^s δ = 300/350/366/380 keV (Sun frame and P050 annual), inelastic O4^s δ = 300/350 keV (Sun frame). At 400 GeV: O1^s, O4^s, O6^s, O9^s, O10^s, O14^s, O15^s, O5^v, L10, O9q2, inelastic O1^s δ = 300 keV (δ = 340 keV is empty in the Sun frame: v_max = 794.6 km/s gives δ_max(248 keV, 400 GeV) < 340 keV).

## 3. Method

1. **Observed pdf.** p(E_obs) ∝ Σ_i dR/dE(E_i) ε(E_i) G(E_obs − E_i; σ(E_i)) ΔE, normalised over the observed grid. Both edges.
2. **Shape metrics.** Hellinger H = √(1 − Σ√(pq)); KL(p‖q) = Σ p_f ln(p_f/q_f) with the floor; peak position (true energy, > 20 keV); observed 16/50/84 percentiles (extended edge); N_lo; R(100–200)/R(200–270); R(270–423)/R(200–270); fraction of accepted events above 270 keV with the extended edge.
3. **Events to separate.** Per-event log-likelihood ratio λ = ln(p_f/q_f). Gaussian estimate, direction "p true, reject q": N = 9 Var_q(λ)/(D_pq + D_qp)²; reverse with Var_p; "simple" symmetric N = 9/(D_pq + D_qp) = 3²/(2 D̄). Toy MC (20 000 toys per N, seed 57, N ∈ {1,…,200}): smallest N for which the median of Σλ under the true model exceeds the 99.865 % quantile under the alternative; run for all pairs with Gaussian N ≤ 150. Convention note: P044's N = 9/Z₁² with Z₁ = (D_pq + D_qp)/(σ_p + σ_q) equals our two-direction sum and is ≈ 4× our single-direction number.
4. **Clustering.** Average-linkage hierarchical clustering on H; clusters cut at H = 0.15 (≈ N_3σ ≳ 50–100).
5. **Interference.** Hamiltonians {1: c1, 6: c6} and {4: c4, 6: c6} with c6 = 2/m_v², c1 or c4 = ±ρ c6, closures without default arguments. O1+O6: compare WimPyDD with R6 + ρ²R1. O4+O6: even part (R₊ + R₋)/2 vs R6 + ρ²R4; cross term X46 = (R₊ − R₋)/(4ρ), its ρ-independence and its shape relative to O10 (q²Σ″) and O6 (q⁴Σ″); node of the Σ″ amplitude at c4 + c6 q²/m_N² = 0, i.e. E_node = ρ m_N²/(2 m_T) = 3.60 MeV × ρ (m_T = 122.3 GeV). Continuous ρ scan (10⁻⁶·⁵–10¹·⁵, 2000 points) of N_lo for both signs using the pure spectra and X46; maximal ρ and maximal O1/O4 share of the 200–270 keV rate with N_lo ≤ 1.6, 2.2, 3, 5. Hidden SI cross-section: O6 normalised to 1.0 accepted event in 2.84 t·yr, c1 = ρ_max c6, σ_n = (c1/2)² μ² ħ²c²/π (WimPyDD's own SI mapping, `lz.wd_c_SI_from_sigma_n` inverted; m_n = 0.931 GeV, ħ²c² = 0.389 × 10⁻²⁷ cm² GeV²).
6. **Second event.** p(E₂ | model) per keV at E₂ = 100, 150, 200, 248, 300, 350, 400 keV for both edges; likelihood ratios relative to L10; band probabilities (< 55, 55–200, 200–270, > 270 keV); posterior over the 12 compatible spectra (uniform prior) after E₂.

## 4. Results

### 4.1 Response-level degeneracies (1 TeV, 20–400 keV; `P057_results.json: spectral_ratios`)

| ratio a/b | log-log slope | rms about power law [dex] | ratio at 50 / 248 keV | verdict |
|---|---|---|---|---|
| O6^s/O10^s | 0.9995 | 0.0006 | 0.0035 / 0.0173 | exactly ∝ E_R (q²): same response, NOT the same shape |
| O6^v/O10^v | 0.9999 | 0.0006 | | idem |
| O6^s/O6^v | 0.0002 | 0.0115 | 1.04 / 1.09 | isoscalar ≡ isovector (LZ's L4 statement confirmed) |
| O10^s/O10^v | 0.0006 | 0.0115 | 1.04 / 1.09 | idem (L2) |
| O9^s/O14^s | 0.082 | 0.0073 | | ⟨v⊥²⟩ nearly flat: effectively degenerate |
| O9^s/O9^v | −0.30 | 0.089 | 1.79 / 1.27 | isovector Σ′ is harder; distinguishable in principle |
| O15^s/O15^v | −1.01 | 0.75 | 36 / 23 | Φ″ vs Σ′ dominance flips: not degenerate |
| L10/O9q2 | −1.002 | 0.0020 | | q⁴Σ′ vs q⁶Σ′: exactly one power of E_R apart |
| L10/O6^s | 0.38 | 0.19 | 1.54 / 9.1 | Σ′ vs Σ″ form factors differ |
| O9^s/O4^s | 1.30 | 0.145 | | q² and Σ′ vs Σ′+Σ″ |
| O9q2/O9^s | 2.003 | 0.004 | | q⁴ as constructed (check of the closure) |
| O5^v/O6^s | −0.42 | 0.077 | | Δ + v²M vs Σ″; accidentally similar |
| L10 (Sun) / L10 (P050 annual) | −0.005 | 0.0008 | 0.996 / 0.995 | halo labelling irrelevant for elastic |
| inel366 Sun/annual | 2.86 | 0.23 | — / 0.34 | the halo matters near the kinematic edge |

The LZ "scalar constant" pairs are single-operator identities (L2, L8 → O10; L4, L20 → O6; L1, L5 → O1; L3, L17 → O11; L11, L14 → O9; recalled Anand reductions, likely). At the operator level we find one further exact power-law relation (O6 = E_R × O10, L10 = O9q2/E_R) and one effective degeneracy not used by LZ (O9 ≡ O14 to 0.007 dex).

### 4.2 Shape metrics (1 TeV; `P057_shape_metrics.csv`)

| model | N_lo | R(100–200)/R(200–270) | R(270–423)/R(200–270) | fraction above 270 keV (ext. edge) | true peak [keV] | observed 16/50/84 % (ext. edge) [keV] |
|---|---|---|---|---|---|---|
| L10 | 0.201 | 1.40 | 0.89 | 0.234 | 196 | 121/204/326 |
| O9q2 | 0.024 | 0.96 | 1.35 | 0.387 | 211 | 165/244/380 |
| O6^s | 0.429 | 2.06 | 1.42 | 0.241 | 131 | 83/178/316 |
| O10^s | 3.09 | 3.29 | 0.99 | 0.087 | 36 | 36/97/215 |
| O9^s | 2.09 | 2.08 | 0.59 | 0.089 | 21 | 19/126/233 |
| O14^s | 2.37 | 2.17 | 0.56 | 0.079 | 21 | 18/117/227 |
| O15^s | 1.88 | 0.76 | 0.31 | 0.049 | 54 | 41/74/222 |
| O15^v | 0.759 | 4.83 | 0.77 | 0.089 | 151 | 81/150/218 |
| O5^v | 1.05 | 1.85 | 1.12 | 0.179 | 51 | 52/147/282 |
| O13^s | 4.92 | 0.83 | 0.55 | 0.056 | 36 | 27/54/203 |
| inel300 (annual) | 0 | 4.52 | 0.87 | 0.133 | 169 | 148/180/234 |
| inel350 | 0 | 0.89 | 2.09 | 0.513 | 201 | 191/285/369 |
| inel366 | 0 | 0.21 | 5.44 | 0.811 | 341 | 240/336/383 |
| inel380 | 0 | 0.006 | 40 | 0.975 | 346 | 308/347/387 |
| inelO4 300 (Sun) | 0 | 1.10 | 0.89 | 0.290 | 196 | 165/225/323 |
| inelO4 350 (Sun) | 0 | 0.065 | 1.83 | 0.618 | 251 | 233/293/370 |
| O4^s | 27.9 | 4.56 | 0.58 | 0.013 | 21 | 9/25/93 |
| O1^s | 2833 | 11.9 | 0.47 | 0.000 | 21 | 8/17/34 |

N_lo agrees with P003 (O6 0.43, O10 3.1, O9 2.1, O14 2.3, O5^v 1.05, O15^v 0.75, O13^s 4.9, L10 0.20, O1 2752 → 2833 here on a different grid) and P044 (O9q2 0.024). L10 accepted rate at d10 = 1 (WimPyDD convention) 1.175 /t·yr → 3.34 events in 2.84 t·yr, P012's 3.34. At 400 GeV N_lo is 1.4× larger (P003: 200 GeV 2.9×), the elastic peaks move down by 5–25 keV.

### 4.3 Events to separate pairs at 3σ (1 TeV; `P057_pairwise.csv`, `P057_key_pairs.csv`; Fig. `P057_N3sigma_matrix_{LZ,EXT}.png`)

Toy numbers (LZ edge → extended edge), "a true, reject b / b true, reject a":

| pair | H (LZ/EXT) | Gaussian N (worse direction) LZ/EXT | toys LZ | toys EXT |
|---|---|---|---|---|
| O6 vs O10 | 0.24/0.27 | 25/20 | 20/20 | 20/15 |
| O9 vs O14 | 0.018/0.022 | 3650/2520 | — | — |
| O6^s vs O6^v | 0.008/0.009 | 17 100/12 900 | — | — |
| O10^s vs O10^v | 0.008/0.008 | 18 600/17 000 | — | — |
| O9^s vs O9^v | 0.032/0.066 | 1180/305 | — | — |
| L10 vs O9q2 | 0.155/0.177 | 78/53 | 40/60 | 40/50 |
| L10 vs O6 | 0.18/0.17 | 41/49 | 40/30 | 50/40 |
| O6 vs O9 | 0.30/0.31 | 26/23 | 20/8 | 20/8 |
| O9 vs O10 | 0.23/0.22 | 21/24 | 25/20 | 25/25 |
| O10 vs O14 | 0.23/0.22 | 21/22 | 25/25 | 25/25 |
| O6 vs O15^v | 0.15/0.22 | 72/36 | 50/60 | 25/30 |
| O6 vs O5^v | 0.13/0.14 | 95/82 | 80/60 | 80/60 |
| O10 vs O5^v | 0.13/0.155 | 78/52 | 80/80 | 50/50 |
| L10 vs inel366 | 0.51/0.54 | 10.0/6.9 | 2/7 | 2/5 |
| L10 vs inel350 | 0.45/0.42 | 14/15 | 3/10 | 3/10 |
| L10 vs inel380 | 0.60/0.66 | 6.1/3.5 | 1/6 | 2/4 |
| O6 vs inel366 | 0.63/0.58 | 4.9/6.4 | 1/5 | 2/5 |
| inel350 vs inel366 | 0.25/0.26 | 29/28 | 20/25 | 12/20 |
| inel366 vs inel380 | 0.30/0.21 | 16/57 | 10/15 | 20/40 |
| inel300 vs inelO4 300 | 0.28/0.30 | 26/19 | 20/12 | 15/12 |
| inel350 vs inelO4 350 | 0.46/0.34 | 5.3/13 | 5/5 | 10/12 |
| L10 vs O4 | 0.61/0.64 | 2.3/2.2 | 3/3 | 3/3 |
| O6 vs O1 | 0.81/0.83 | 0.8/0.6 | 2/2 | 1/2 |

Among the 36 pairs of the nine compatible elastic shapes (L10, O9q2, O6, O10, O9, O14, O15^{s,v}, O5^v), the median worse-direction Gaussian N_3σ is 21 (LZ) / 20 (EXT); 81–83 % of pairs need ≤ 30 events and only O9/O14 (3 %) needs > 100. The extended edge changes most elastic N_3σ by ≤ 30 % in either direction (the q⁴/q⁶ spectra gain information above 270 keV, the low-energy-peaked ones lose relative weight); it halves N_3σ for inel366 vs inel380 in the L10-like direction but the inelastic pair becomes harder in the other direction because the annual-average δ = 380 keV spectrum (peak 346 keV) then overlaps δ = 366 keV. The comparison with P050 (L10 vs inel366: 4 if L10 true, 5–6 if inelastic true): our inelastic-true numbers agree (5–7); our L10-true number is 2 because we use the whole observed window, and 5–7 % of L10 events fall below 100 keV where the inelastic pdf is empty (P050 restricted to E_obs ≥ 100 keV). P044's L10 vs O9q2 190–200 is our 25–78 in the two-direction convention (× ≈ 4, §3.3) and 40–60 in toys — degenerate for the ~2 events expected in the untouched LZ exposure (P020), separable in a 60 t programme (P050: 2–4 events per t·yr at best fit).

Halo labelling (`P057_halo_labelling.csv`): the same inelastic model in the Sun frame vs the 12-day annual average needs N_3σ = 2 × 10⁴ (δ = 300), 128–171 (350), 47–98 (366) and 5.5 (380 keV, LZ edge) events to be told apart — for δ ≥ 366 keV the halo convention is a shape systematic comparable to the L10/inelastic difference and must be stated. Elastic L10: 10⁶ events (irrelevant).

Mass (`P057_mass_dependence.csv`): 400 GeV vs 1 TeV for the same elastic operator needs 250–19 000 events (H = 0.008–0.07); inelastic δ = 300 keV: 78–139. LZ's "masses ≥ 400 GeV nearly degenerate" holds for any conceivable exposure.

Clusters at H < 0.15 (extended edge): {O9^s, O9^v, O14^s, O14^v}, {O10^s, O10^v}, {O6^s, O6^v, O5^v}; everything else singletons. With the LZ edge O15^v joins the O6 cluster and O5^v the O10 cluster; inel380 and inelO4 350 cluster.

### 4.4 Interference (`P057_results.json: interference`, `P057_admixture_scan.csv`; Fig. `P057_interference.png`)

- **O1 + O6.** WimPyDD reproduces R6 + ρ²R1 to 4 × 10⁻¹⁶ for ρ = 10⁻³, 10⁻² (both signs) and 1: M and Σ″ do not interfere; the sign is irrelevant. ρ → ∞ (ρ = 1) gives H = 0.000 to pure O1; ρ → 0 gives N_lo → 0.4298 (pure O6 0.4286; the scan's smallest ρ = 3 × 10⁻⁷).
- **O4 + O6.** Even part equals R6 + ρ²R4 to 6 × 10⁻¹⁶; the cross term X46 = (R₊ − R₋)/4ρ is ρ-independent to 4 × 10⁻¹⁶ (2 × 10⁻¹⁴ at ρ = 10) and has the log-log slope 0.0000 relative to O10 (−0.9995 relative to O6): it is the q²Σ″ term of the response as expected, positive for c4c6 > 0. ρ → ∞ (ρ = 10) gives H = 7 × 10⁻⁴ to pure O4. The destructive combination has a Σ″ node at 3.60 MeV × ρ (248 keV for ρ = 0.069); because c4² Σ′ survives, the total R₋/R₊ minimum is 0.063 at 314 keV, displaced from 248 keV by the Σ′ contamination (analytically the minimum sits at a²/b² = 1 + s, s = Σ′/Σ″ share). Only the momentum-dependent c4 = −(q²/m_N²)c6 of L10 cancels Σ″ at every energy.
- **How much O1 can hide** (1 TeV; pure N_lo: O1 2833, O4 27.9, O6 0.429): the O1 share of the 200–270 keV rate is f_hi(O1) = (N_max − N_lo(O6))/(N_lo(O1) − N_lo(O6)) = 4.1 × 10⁻⁴ (N_max = 1.6), 6.3 × 10⁻⁴ (2.2), 9.1 × 10⁻⁴ (3), 1.6 × 10⁻³ (5), i.e. ρ = c1/c6 ≤ 0.98, 1.20, 1.46, 1.94 × 10⁻⁵. With c6 fixed to one accepted LZ event (c6 = 4.54 × 10⁻⁵ GeV⁻² in WimPyDD units, Anand c6 m_v² = 1.38), the hidden O1 coupling is c1 ≤ 4.5–8.8 × 10⁻¹⁰ GeV⁻², σ_n ≤ 5.3 × 10⁻⁴⁸ (1.6), 8.0 × 10⁻⁴⁸ (2.2), 1.2 × 10⁻⁴⁷ (3), 2.1 × 10⁻⁴⁷ cm² (5); its total ROI contribution would be 0.30–1.18 events, all below 55 keV. (Recalled, likely: LZ's 2024 SI limit at 1 TeV is a few × 10⁻⁴⁷ cm², so the tolerance-based bound is the same constraint restated.) A second event cannot see such an admixture in the high-energy window: the N_max = 3 (5) mixture differs from pure O6 by H = 0.35 (0.43), N_3σ = 21–26 (12–15) events, and all of that information is in the low-energy shoulder.
- **How much O4 can hide.** N_lo ≤ 3 allows |c4/c6| ≤ 0.016 for either sign (0.0085/0.013 for N_max = 1.6 with +/−; 0.0118/0.0145 for 2.2; 0.0256/0.0192 for 5). The incoherent O4 share of the 200–270 keV rate reaches 6 % (constructive) or 17 % (destructive) at N_max = 3 and 13 %/27 % at N_max = 5; the destructive sign hides more O4 at fixed N_lo because the cross term removes Σ″ rate at low energy first (N_lo dips to 0.25 at ρ ≈ −0.004 before the Σ′ part takes over). The high-energy rate itself changes by +82 %/−20 % at ρ = ±0.069.

### 4.5 What a second event would tell (`P057_second_event_lookup.csv`; `P057_results.json: second_event_posterior_uniform_prior`)

Likelihood ratios p(E₂ | model)/p(E₂ | L10), extended edge (LZ edge in brackets where different):

| model | E₂ = 100 | 150 | 200 | 248 | 300 | 350 | 400 |
|---|---|---|---|---|---|---|---|
| O9q2 | 0.47 | 0.71 | 0.93 | 1.15 | 1.39 | 1.65 | 1.87 |
| O6 | 1.74 | 0.97 | 0.62 | 0.70 | 1.31 | 1.20 | 0.69 |
| O10 | 2.12 [1.76] | 0.79 | 0.38 | 0.34 | 0.53 | 0.42 | 0.21 |
| O9 | 1.31 | 0.88 | 0.67 | 0.54 | 0.45 | 0.38 | 0.33 |
| O14 | 1.33 | 0.86 | 0.64 | 0.51 | 0.41 | 0.33 | 0.29 |
| O15^s | 1.66 | 0.135 | 0.51 | 0.64 | 0.52 | 0.060 | 0.024 |
| O5^v | 1.76 | 0.79 | 0.54 | 0.68 | 1.22 | 0.84 | 0.32 |
| inel300 | 0.041 | 2.5 | 1.51 | 0.21 | 0.51 | 0.96 | 0.40 |
| inel350 | 6 × 10⁻⁹ | 0.14 | 1.58 [2.46] | 0.43 [0.69] | 1.94 | 3.7 | 1.5 |
| inel366 | 2 × 10⁻¹⁵ | 9 × 10⁻⁴ | 0.54 [2.2] | 0.36 [1.46] | 3.0 [2.2] | 6.0 | 2.3 |
| inel380 | 2 × 10⁻²⁴ | 2 × 10⁻⁸ | 0.027 [0.72] | 0.10 [2.6] | 3.6 [16.7] | 7.4 | 2.6 |
| O4 | 0.94 | 0.27 | 0.11 | 0.080 | 0.085 | 0.058 | 0.030 |
| O1 | 0.007 | 0.014 | 0.004 | 3 × 10⁻⁴ | 6 × 10⁻⁴ | 10⁻³ | 4 × 10⁻⁴ |

Band probabilities with the LZ edge: L10 puts 7/57/34/2 % of accepted events below 55 / 55–200 / 200–270 / above 270 keV; O6 10/66/22/2; O10 31/58/10/1; O9 37/46/17/1; inel300 0/80/20/0; inel366 0/21/77/2; inel380 0/4/85/12 %. A second event *below 55 keV* excludes every inelastic model (LR ≤ 10⁻⁸ for δ ≥ 350 keV) and O9q2 (P = 1.2 %) but leaves O6/O10/O9/O14/O5^v/L10 within a factor 3 of each other. A second event *at 248 ± 20 keV* is the least informative outcome: all compatible elastic models lie within LR 0.34–1.45 of L10, and with the LZ edge the inelastic models are at 0.7–3.4. A second event *above 300 keV* (only possible with an extended edge, or at LZ's edge with the 1.2 % L10 tail) favours inelastic δ ≥ 350 keV by 3–7 over L10 and by 10–100 over O10/O9/O14/O15^s. Posterior over the 12 compatible spectra (uniform prior) after E₂: E₂ = 100 keV (LZ edge) → O10 0.17, O6 0.17, O5^v 0.16, O15^s 0.13, O9 0.11, O14 0.11, L10 0.10, O9q2 0.06, inelastic < 0.004; E₂ = 248 keV (ext.) → O9q2 0.17, L10 0.15, O6 0.10, O5^v 0.10, O15^s 0.10, O9 0.08, O14 0.08, inel350 0.07, inel366 0.05, O10 0.05, inel300 0.03, inel380 0.015; E₂ = 350 keV (ext.) → inel380 0.31, inel366 0.25, inel350 0.16, O9q2 0.07, O6 0.05, L10 0.04, inel300 0.04, O5^v 0.035, O10/O9/O14 0.014–0.017, O15^s 0.003.

Date and position: for a second event the date carries LR 1.4/2.1/2.6 (δ = 300/350/380 keV, P006; P034's corrected halo grid lowers the N_3σ from dates alone to 97/12 events for δ = 300/350) and 0.98 for every elastic model; the position is uniform in the fiducial volume for every model (a genuine signal; only backgrounds are non-uniform), so the position discriminates signal from wall/RFR MSSI (P004) but not model from model.

## 5. Validation and robustness

- N_lo for all single operators reproduces P003 to ≤ 3 % (grid differences); L10 event count reproduces P012 (3.34); O9q2 N_lo reproduces P044 (0.024).
- L10 in the Sun frame vs P050's annual-average L10: ratio 0.996 with 0.0008 dex scatter — the two halos give the same elastic shapes.
- WimPyDD interference bookkeeping verified twice: by `coeff_squared_list` and by numerical sums to machine precision.
- Gaussian vs toy N_3σ: agree within a factor 1.5 for N ≳ 10; for N ≲ 5 the LLR distribution is strongly skewed (one out-of-support event decides) and only the toys are reliable. The floor w = 10⁻³ (columns `*_w1e3` in `P057_pairwise.csv`) changes N_3σ for populated pairs by ≤ 20 % and lowers it for pairs with disjoint supports.
- Efficiency-edge width: not varied here; P003 found ≤ 9 % effects on N_lo.

## 6. Failed or abandoned

- A first analysis run divided by zero for the empty inelastic δ = 340 keV spectrum at 400 GeV (Sun frame); guarded, the model is reported as kinematically forbidden.
- The admixture scan first started at ρ = 10⁻⁴, above the O1 limit (ρ_max ≈ 10⁻⁵); extended to 10⁻⁶·⁵.
- The ρ → 0 check by direct WimPyDD evaluation at ρ = 10⁻² is not a clean limit (at ρ = 10⁻² O4 already contributes H = 0.21); the limit was taken from the analytic scan instead.
- The initial O4+O6 scan with eight ρ values was trimmed to four (0.01, 0.069, 0.3, 10; both signs) to stay within the foreground time budget; the cross term is ρ-independent to 10⁻¹⁶ so nothing is lost.
- The first compute run exceeded the 600 s foreground limit and was moved to the background by the harness; the script was given a 530 s budget with per-spectrum caching and re-run in the foreground until complete.

## 7. Figures

- `figures/P057_spectra_overlay.png` — normalised observed-energy pdfs, 1 TeV, LZ edge (left) and 1000 phd edge (right); solid: elastic compatible set; dashed: inelastic δ = 300/366/380 keV (annual average). Dotted line: 248 keV.
- `figures/P057_N3sigma_matrix_LZ.png`, `_EXT.png` — Gaussian worse-direction N_3σ for 16 shapes.
- `figures/P057_dendrogram_LZ.png`, `_EXT.png` — Hellinger average-linkage dendrograms; coloured clusters below H = 0.15.
- `figures/P057_interference.png` — left: O4+O6 spectra for c4/c6 = ±0.01, ±0.069, ±0.3 and L10; right: N_lo vs coupling ratio for O1+O6 and O4+O6 (both signs), dotted lines at N_lo = 3 and 5.

## 8. Discussion

The LZ degeneracies are exact by construction (same operator). At the operator level the only degeneracies that survive for realistic exposures are O9 ≡ O14 (∼3000 events), the isoscalar/isovector pairs of Σ″ operators (∼10⁴), and 400 GeV vs 1 TeV (≥ 250). Everything else — including O6 vs O10, which share a nuclear response and differ by one power of E_R — separates with 15–25 events, and L10 from its neighbours O6 and O9q2 with 30–60. Against the inelastic class the elastic spectra separate with 2–7 events, where the halo convention (Sun frame vs annual average) becomes a comparable systematic for δ ≥ 366 keV. A second event is decisive only if it lands below 55 keV (kills inelastic) or above 300 keV (favours δ ≥ 350 keV by 3–7 and needs an extended ROI); at 248 keV it teaches almost nothing about the operator. Interference cannot hide an SI admixture: the O1 share of the high-energy rate is bounded at 10⁻³ by the low-energy null, σ_n ≲ 10⁻⁴⁷ cm² at 1 TeV, and the only interference that matters (O4–O6 through Σ″) is the mechanism by which L10 becomes companion-free.

## 9. References

LZ Collaboration, arXiv:2609.02823 (2026). A. L. Fitzpatrick, W. Haxton, E. Katz, N. Lubbers, Y. Xu, JCAP 02 (2013) 004. N. Anand, A. L. Fitzpatrick, W. C. Haxton, PRC 89, 065501 (2014). I. Jeong, S. Kang, S. Scopel, G. Tomar, CPC 276, 108342 (2022). D. Baxter et al., EPJC 81, 907 (2021). S. Kullback, R. A. Leibler, Ann. Math. Stat. 22, 79 (1951). Corpus: P002, P003, P006, P009, P012, P016, P020, P021, P027, P031, P034, P035, P038, P044, P047, P050.

## 10. Tools and provenance (mirrors `output/provenance/P057.json`)

- Agent tools: Read (PAPER_GUIDE, dossier, P003/P027/P031/P044/P012/P021/P050/P047 papers, tex LEE section and Tables S6/S7, lzcommon wd_* section, WimPyDD package.py interference/eft_hamiltonian sections, P044/P050 scripts, own outputs and figures), Bash (ledger extraction, grep of tex/scripts, WimPyDD timing test, compute runs, analysis runs, summary extraction), Write/Edit (script, details, provenance, paper), Skill (dataviz), ToolSearch/Monitor (waiting for the compute stage).
- Software: python 3.12.13; WimPyDD 2.0.4 (`eft_hamiltonian` with closure coefficient functions, `diff_rate` via `lz.wd_rate`, `streamed_halo_function` via `lz.wd_halo`); numpy 2.5.3; scipy 1.18.1 (`special.erf`, `stats.norm`, `cluster.hierarchy`); pandas 3.0.5; matplotlib 3.11.2 (Agg); `output/code/common/lzcommon.py`.
- Local inputs: `inputs/LZ_arXiv_2609.02823_fulltext.tex` (lines 485–498 LEE; 825–909 Tables S6/S7); `output/00_evidence_dossier.md`; `output/results_ledger.csv`; `output/papers/P003, P012, P021, P027, P031, P044, P047, P050.md`; `output/work/P003/P003_spectra.npz` (inspected, not used numerically), `P003_summary.json`; `output/work/P050/P050_spectra_cache.npz` (annual inelastic O1^s and L10 spectra); `output/code/P044_anapole_edm.py`, `P050_next_generation.py`, `P003_nreft_shapes.py`, `P012_magnetic_dipole.py` (conventions); `WimPyDD/package.py` (read-only).
- Recalled knowledge: NREFT response coefficient functions (certain); Anand Lagrangian → operator identities L1/L5 → O1, L2/L8 → O10, L4/L20 → O6, L3/L17 → O11, L11/L14 → O9 (likely); LZ 2024 SI limit at 1 TeV of a few × 10⁻⁴⁷ cm² (likely); 2024 low-energy tolerance 3–5 events (uncertain); ħ²c² = 0.389 × 10⁻²⁷ cm² GeV² (certain); Hellinger/KL definitions and Gaussian LLR asymptotics (certain); Okabe–Ito palette (certain).
- Datasets: none. Data requests: none.
- WimPyDD-generated files: none (`diff_rate` does not write response-function files; the `WimPyDD/WimPyC/Response_functions/spin_1_2/*.npy` files predate this run).
