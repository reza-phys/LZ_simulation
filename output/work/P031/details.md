# P031 — Relativistic operator matching for the LZ-preferred nonrelativistic operators with directdm

Simulated date 2026-09-09. Author profile: DM effective-field-theory practitioners. Category EFT (hep-ph).
Script: `output/code/P031_directdm_matching.py`. Tables: `output/work/P031/*.csv`, `P031_summary.json`. Figures: `output/work/P031/figures/`.
Logs: `run_log.txt` (full grid, first complete run), `run_log_1000_final.log`, `run_log_200_4000_final.log`, `run_log_final.log` (aggregation used for all quoted numbers).

## 1. Motivation and framework

P003 showed that a lone 248 keV recoil without a 5.4–55 keV population selects the q²-suppressed spin operators of the Anand–Fitzpatrick–Haxton NR basis (O6, O10, O9, O14, O15, O5^v and the L10-like transverse-spin combination (q²/m_N²)O4 − O6), while O1, O4, O7, O8^s, O11, O12 predict tens to thousands of low-energy companions (N_lo). P016 and P027 quantified that the posterior mass sits in exactly those operators. The open question is *which relativistic (quark/gluon/photon-level) operators actually produce those NR operators*, with what Wilson coefficient at the hadronic scale μ = 2 GeV, and which relativistic structures are excluded because their NR reduction is dominated by the coherent or unsuppressed operators.

We use `directdm` 2.2.2 (Bishara, Brod, Grinstein, Zupan), which implements the tree-level matching of the dimension-5 to -7 DM–SM operators of arXiv:1707.06998 and 1801.04240 onto the NR coefficients c_i^{p,n} of arXiv:1308.6288 (including the pion/η-pole contributions to O6 and O10 and the photon-pole 1/q² contributions of the dipoles), plus QCD×QED RG running from M_Z to 2 GeV. The NR rates are computed with WimPyDD 2.0.4 (LZ's own code) through `lzcommon.wd_rate`, with WimPyDD's convention c⁰ = c_p + c_n, c¹ = c_p − c_n (P003, P007).

### Relativistic operator basis (directdm names; normalisation as in 1707.06998, recalled — certain for the structure, likely for the prefactors)

| directdm | operator | dim |
|---|---|---|
| C51 | (e/8π²) χ̄σ^{μν}χ F_{μν} (magnetic dipole) | 5 |
| C52 | (e/8π²) χ̄σ^{μν}iγ₅χ F_{μν} (electric dipole) | 5 |
| C61q, C62q, C63q, C64q | χ̄γ_μχ q̄γ^μq (V–V); χ̄γ_μγ₅χ q̄γ^μq (A–V); χ̄γ_μχ q̄γ^μγ₅q (V–A); χ̄γ_μγ₅χ q̄γ^μγ₅q (A–A) | 6 |
| C71–C74 | (α_s/12π) χ̄χ GG; (α_s/12π) χ̄iγ₅χ GG; (α_s/8π) χ̄χ GG̃; (α_s/8π) χ̄iγ₅χ GG̃ | 7 |
| C75q–C78q | m_q χ̄χ q̄q; m_q χ̄iγ₅χ q̄q; m_q χ̄χ q̄iγ₅q; m_q χ̄iγ₅χ q̄iγ₅q | 7 |
| C79q, C710q | m_q χ̄σ^{μν}χ q̄σ_{μν}q (T–T); m_q χ̄iσ^{μν}γ₅χ q̄σ_{μν}q | 7 |
| C715q–C718q | derivative "dipole-type" operators; from directdm's NR output they are χ̄σ^{μν}χ ∂_ν(q̄γ_μq) [→ O5 ∝ F₁, O6 and q²O4 ∝ F₁+F₂], χ̄σ^{μν}iγ₅χ ∂_ν(q̄γ_μq) [→ O11], χ̄σ^{μν}χ ∂_ν(q̄γ_μγ₅q) [→ O9 ∝ F_A], χ̄σ^{μν}iγ₅χ ∂_ν(q̄γ_μγ₅q) [→ O14 ∝ F_A] (structure inferred from the code output; **likely**) | 7 |
| C723q, C725 | twist-2 quark / gluon operators | 7 |
| C711–C714 | four flavour-universal dim-7 coefficients; directdm's 3-flavour matching returns **no** NR coefficient for them at this order (checked) | 7 |

Twenty-nine "structures" (operator × flavour assignment) were run: flavour-universal (u = d = s), isovector (u = −d), Z-like axial charges (T₃ = +½, −½, −½), quark-charge weighted (anapole: C62q = e Q_q; Q15 with Q_q); see `STRUCTURES` in the script. Majorana DM admits only C62, C64, C71–C78 (directdm docstring): no vector current, no tensors, no dipoles, no Q15–Q18.

### directdm normalisation (read from the source, `wilson_coefficients.py`)
`WC_3f(coeffs, 'D').cNR(m_χ, q)` returns c_i^{p,n} in GeV⁻² such that L_NR = Σ c_i^N O_i^N with the 1308.6288 operators (q/m_N, v⊥, S_χ, S_N), q the spatial momentum transfer in GeV, and
c_i(q) = c_i^{const} + q² c_i^{q²} + c_i^{1/q²}/q² + c_i^{π}/(m_π²+q²) + c_i^{η}/(m_η²+q²) + q² c_i^{q²π}/(m_π²+q²) + q² c_i^{q²η}/(m_η²+q²)
(`_my_cNR` gives the pieces, `cNR` assembles them). Example checks against the standard dipole reduction (Del Nobile 2021, recalled — likely): c₁^p = −α C51/(2π m_χ) = e μ_χ/(2m_χ) sign aside, c₅^p = 2α m_N C51/(π q²) = 2e μ_χ m_N/q², c₄^N = −(2α/π)(μ_N/m_N)C51 = −e μ_χ g_N/m_N, c₆^N = +e μ_χ g_N m_N/q², all consistent with **μ_χ = e C51/(4π²)** for the dipole normalisation (e/8π²)χ̄σFχ, i.e. L = (μ_χ/2)χ̄σ^{μν}χF_{μν}.

## 2. Method

1. **Stage A** — for every structure, m_χ = 200/1000/4000 GeV and q = 50 MeV and q_event = √(2 m_T E_R) = 0.2463 GeV (E_R = 248 keV, m_T = 122.3 GeV for A = 131.3): c_i^{p,n}(q) → `P031_nr_coefficients.csv`.
2. **Stage B** — each distinct q-structure of each NR operator is a separate WimPyDD operator (tuple key `(i, label)`; verified that WimPyDD reproduces exactly a single mixed-q coefficient split into pieces, and that per-isospin-component q-dependence is handled component-wise — `package.py` line ~201). Spectra dR/dE on 1–330 keV (4 keV steps + key points), Sun-frame Baxter SHM (`lz.wd_halo()`), natural Xe, j_χ = ½. Efficiency = P003/P012 form (0.96 plateau, erf roll-offs to 50 % at 5.4 and 269.9 keV, σ = 3.4/8 keV). N_lo = R(5.4–55)/R(200–270 keV); R_ROI = efficiency-weighted 1–330 keV. Per-piece *diagonal* rates at 20 and 248 keV identify the leading NR operator (interference is dropped in this diagnostic but kept in the full spectrum; the ratio full/diagonal-sum is logged for every structure).
3. **Stage C** — with N_ROI(C) = C² N₁ (single relativistic operator ⇒ quadratic), the coefficient giving LZ's best fit of 1.0 event (Table I, 1.0 +1.4 −0.7) is C₁ = N₁^{−1/2}; Λ ≡ C^{−1/(d−4)} (dim-6: Λ⁻², dim-7: Λ⁻³ with the m_q or α_s factor part of the operator, dim-5: Λ⁻¹). The 0.3–2.4 event range gives Λ_lo68 (2.4 events) and Λ_hi68 (0.3 events).
4. **Stage D** — RG running M_Z → m_b → m_c → 2 GeV with `WC_5f(...).cNR(m, q, RGE=True/False)` and the chained `match()` dictionaries.
5. **Stage E** — P016's baseline Z(N_lo) curve (`output/work/P016/P016_Z_vs_Nlo_curves.csv`, log-interpolated) turns each N_lo into a predicted LZ local significance, compared with Table S6 (`lz.LSIG`) at 1000 GeV.

Classification thresholds (P003/P016): N_lo ≤ 5 compatible (≥ 3σ class), 5–20 marginal, > 20 excluded.

### Two technical problems (both solved; recorded in `failed_or_abandoned`)
- **directdm + numpy 2.5.3:** `WC_5flavor.__init__`/`WC_4flavor.__init__` call `np.delete` with slices running past the array end (e.g. `np.s_[27:144]` on length 131) → `IndexError`. We wrap `np.delete` in the script to drop out-of-range indices (the pre-1.19 numpy behaviour). The installed package is untouched. `WC_3f` (matching only) needs no workaround.
- **WimPyDD coefficient functions:** WimPyDD treats every non-reserved argument of a coefficient function, *including default arguments*, as a global Hamiltonian parameter shared by all coefficients. A first version passed the piece values as `lambda c0=…, c1=…` defaults; the pieces silently overwrote each other (e.g. the V–A rate collapsed to its 10³-times-smaller O9 admixture; the photon dipole gave N_lo = 24.5 instead of ≈ 400). Caught by the per-piece diagonal check; fixed with closures (`_const_fn`, `_q_fn`). All quoted numbers are from the corrected code (full/diagonal ratios = 1.000 for single-piece structures).

### Validation
- L10 reference (P012 Hamiltonian, d₁₀ = 1, WimPyDD normalisation): N = 3.334 events (P012: 3.336), N_lo = 0.198 (P012 0.199); d₁₀(1 event) = 0.279 in LZ normalisation (P012 0.28) — pipeline reproduces P012.
- Photon magnetic dipole: N_lo = 422 at 1 TeV (P012: 376 with a recalled reduction), μ_χ(1 ROI event) = 9.29 × 10⁻⁷ μ_N (P012: 9.16 × 10⁻⁷ μ_N) — 1.4 % agreement of an independent (directdm) implementation with P012.
- Single-operator N_lo reproduced from P003 where the reduction is a single NR operator: A–A → O4 28.2 (P003 O4: 28), V–V → O1 2756 (2752), S–GG → O1 2755, pure O9 (Q17) 2.06 (P003 2.1), pure O14 (Q18) 2.34 (2.3), P–S → O11 260 (258).

## 3. Results

### 3.1 NR reduction at q = 245 MeV, m_χ = 1000 GeV (`P031_nr_coefficients.csv`; leading |c_i| listed; GeV⁻² per unit Wilson coefficient)
| structure | c_i^p, c_i^n | leading NR operator(s), rate fraction at 248 keV |
|---|---|---|
| MDM (C51) | c₆ 0.201/−0.137, c₅^p 0.072, c₄ −0.014/0.009, c₁^p −1.2e-6 | O4 59 %, O6 41 % (20 keV: O5 68 %) |
| EDM (C52) | c₁₁^p 0.072 | O11 |
| V–V univ | c₁ 3/3, c₃ −0.036, c₆ −3e-5 | O1 |
| A–V univ | c₈ 6/6, c₉ 5.06/5.06 | O8 78 %, O9 22 % (20 keV: O8 99.5 %) |
| anapole (e Q_q A–V) | c₈^p 0.606, c₈^n 0, c₉ 1.69/−1.16 | O9 99 % at 248 keV but O8 90 % at 20 keV |
| V–A univ | c₇ −0.81/−0.81, c₉ 7.6e-4 (m_N/m_χ) | O7 97 % |
| A–A univ | c₄ −1.62/−1.62 (poles cancel for u = d = s) | O4 |
| A–A Z-like | c₄ −2.62/2.48, **c₆ 29.3/−27.7** (110/−108 at 50 MeV) | O4 70 %, O6 30 % (20 keV: O6 2.8 %) |
| S–GG (C71) | c₁ −0.062 | O1 |
| P–GG (C72) | c₁₁ 5.8e-5 | O11 |
| S–GG̃ (C73) | c₁₀ −0.240/−0.140 (−0.393/−0.012 at 50 MeV) | O10 |
| P–GG̃ (C74) | c₆ −2.25e-4/−1.31e-4 | O6 |
| S–S univ | c₁ 0.102/0.104 | O1 |
| P–S univ | c₁₁ −9.6e-5/−9.8e-5 | O11 |
| S–P univ | c₁₀ −0.312/−0.112 (−0.635/+0.128 at 50 MeV: π–η cancellation) | O10 |
| S–P isov | c₁₀ 0.273/−0.279 (1.05/−1.06 at 50 MeV: pion pole) | O10 |
| P–P univ / isov | c₆ −2.9e-4/−1.1e-4 ; 2.6e-4/−2.6e-4 | O6 |
| T–T univ / isov | c₄ −0.0142/0.0057 ; 0.0212/−0.0328 | O4 |
| PT–T univ | c₁₁ 0.012/0.031, c₁₂ 0.014/−0.006, c₁₀ 3e-6 | O11 99 % |
| Q15 univ | c₅ −5.63, c₆ −4.75, c₄ = q²·(…) 0.327 | O5 46 %, O4 33 %, O6 21 % (20 keV: O5 98 %) |
| Q15 charge-weighted | c₆ −5.24/3.59, c₅^p −1.88, c₄ 0.361/−0.247 | O4 59 %, O6 41 % (20 keV: O5 69 %) — the L10 structure c₄/c₆ = −q²/m_N² |
| Q16 univ | c₁₁ −5.63 | O11 |
| Q17 univ | c₉ −1.52/−1.52 | O9 |
| Q18 univ | c₁₄ 1.52/1.52 | O14 |
| twist-2 q / g | c₁ 0.403 ; 0.295 | O1 |

Ratios of subleading to leading coefficients: A–V c₉/c₈ = 0.84 (univ) and 2.8 (anapole, p); V–A c₉/c₇ = 9.4 × 10⁻⁴ = 2 m_N/m_χ × (F_A weights); MDM c₄/c₅ = 0.19, c₆/c₅ = 2.8 (= μ_p); Q15 c₄(q_event)/c₆ = −0.069 = −q²/m_N²; A–A Z-like c₆/c₄ = −11 at 245 MeV and −42 at 50 MeV.

### 3.2 Spectral classification (`P031_structure_table.csv`; m_χ = 1000 GeV; 200 / 4000 GeV in brackets)
| class | structure → NR | N_lo (200 / 4000) | Z_pred (P016) |
|---|---|---|---|
| compatible | P–P univ → O6 | 0.069 (0.22 / 0.059) | 3.59 |
| compatible | P–GG̃ → O6 | 0.082 (0.24 / 0.071) | 3.59 |
| compatible | S–GG̃ → O10 | 0.47 (1.37 / 0.41) | 3.53 |
| compatible | S–P univ → O10 | 0.85 (2.69 / 0.72) | 3.49 |
| compatible | Q17 (dipole × axial current) → O9 | 2.06 (6.35 / 1.77) | 3.38 |
| compatible | Q18 → O14 | 2.34 (8.0 / 1.97) | 3.36 |
| compatible | Q15 charge-weighted → q²O4 − O6 (+O5) | 2.52 (7.7 / 2.17) | 3.34 |
| compatible | P–P isov → O6/(m_π²+q²) | 3.33 (9.7 / 2.87) | 3.29 |
| marginal | Q15 univ → O5 + spin | 15.3 (47 / 13) | 2.90 |
| excluded | S–P isov → O10/(m_π²+q²) | 25.8 (77 / 22) | 2.73 |
| excluded | anapole → O8^p + O9 | 26.9 (82 / 23) | 2.72 |
| excluded | A–A univ → O4 | 28.2 (86 / 24) | 2.70 |
| excluded | T–T isov / univ → O4 | 29.2 / 30.2 (90 / 25) | 2.68 |
| excluded | V–A → O7 | 31.5 (66 / 27) | 2.66 |
| excluded | A–A Z-like → O4 + O6 pole | 55.9 (173 / 48) | 2.45 |
| excluded | A–V univ → O8 + O9 | 167 (517 / 143) | 1.98 |
| excluded | P–S, P–GG, Q16 → O11 | 258–260 (723 / 224) | 1.77 |
| excluded | PT–T → O11 (+O12) | 357 (993 / 311) | 1.59 |
| excluded | magnetic dipole (photon) | 422 (1332 / 361) | 1.50 |
| excluded | V–V isov → O1 | 477 (1611 / 404) | 1.43 |
| excluded | S–S, S–GG, V–V univ, twist-2 → O1 | 2755–2776 (7908 / 2387) | 0.16 |
| excluded | electric dipole → O11/q² | 18 200 (54 500 / 15 700) | 0 |

The **pion pole** is decisive for the pseudoscalar structures. Anand's L2 (iχ̄χ N̄γ₅N) and L4 (χ̄γ₅χ N̄γ₅N) are nucleon-level *contact* operators (O10, O6 with constant coefficients; P003 N_lo = 3.1 and 0.43). At the quark level the nucleon pseudoscalar current is pole dominated: for an isovector coupling (u = −d) c₁₀, c₆ ∝ 1/(m_π²+q²), which enhances low q by (m_π²+q_event²)/(m_π²+q_20keV²) = 3.4 in amplitude and pushes N_lo to 25.8 (S–P) and 3.33 (P–P); for a flavour-universal coupling the π and η poles cancel at low q (full/diagonal rate ratio 0.036 at 20 keV, 0.25 at 248 keV), and N_lo drops to 0.85 and 0.069. Isovector A–A (Z-like charges) acquires an induced-pseudoscalar O6 term that carries 30 % of the 248 keV rate but, being pole enhanced, raises N_lo from 28 to 56.

### 3.3 Wilson coefficients and scales for LZ's best fit (1.0 event; brackets: 2.4 / 0.3 events), m_χ = 1000 GeV
| structure | N₁ (events at C = 1) | C₁ | Λ [GeV] | Λ/m_χ |
|---|---|---|---|---|
| P–P univ (m_q χ̄iγ₅χ q̄iγ₅q) | 11.7 | 0.292 GeV⁻³ | 1.51 [1.30, 1.84] | 0.0015 |
| P–GG̃ | 23.6 | 0.206 GeV⁻³ | 1.69 [1.46, 2.07] | 0.0017 |
| P–P isov | 384 | 0.051 GeV⁻³ | 2.70 [2.33, 3.30] | 0.0027 |
| Q18 → O14 | 3.1 × 10⁴ | 5.7 × 10⁻³ GeV⁻³ | 5.6 [4.9, 6.9] | 0.0056 |
| S–P univ → O10 | 1.35 × 10⁹ | 2.7 × 10⁻⁵ GeV⁻³ | 33.2 [28.7, 40.6] | 0.033 |
| S–GG̃ → O10 | 2.74 × 10⁹ | 1.9 × 10⁻⁵ GeV⁻³ | 37.4 [32.3, 45.7] | 0.037 |
| Q15 charge-weighted (L10-like) | 1.30 × 10¹⁰ | 8.8 × 10⁻⁶ GeV⁻³ | 48.5 [41.9, 59.3] | 0.049 |
| Q17 → O9 | 5.9 × 10¹⁰ | 4.1 × 10⁻⁶ GeV⁻³ | 62.4 [53.9, 76.3] | 0.062 |
| anapole a χ̄γ^μγ₅χ ∂^νF_{μν} (excluded, N_lo 27) | 1.34 × 10¹¹ | a = 2.73 × 10⁻⁶ GeV⁻² | 605 [486, 818] | 0.61 |
| T–T univ / isov (excluded) | 2.0 × 10⁸ / 7.3 × 10⁹ | 7.1e-5 / 1.2e-5 GeV⁻³ | 24.2 / 44.1 | 0.02–0.04 |
| A–A univ (excluded) | 1.9 × 10¹³ | 2.3 × 10⁻⁷ GeV⁻² | 2099 [1690, 2840] | 2.1 |
| V–V univ / isov (excluded) | 5.4 × 10¹⁹ / 1.3 × 10¹⁷ | 1.4e-10 / 8.6e-9 GeV⁻² | 8.6 × 10⁴ / 1.9 × 10⁴ | 86 / 19 |
| magnetic dipole (excluded) | 2.6 × 10⁹ | C51 = 1.96 × 10⁻⁵ GeV⁻¹ | μ_χ = 9.29 × 10⁻⁷ μ_N (200 GeV 4.25e-7, 4000 GeV 1.85e-6) | — |

Mass dependence for the compatible set: Λ(200 / 4000 GeV) = P–P 2.89 / 0.77, P–GG̃ 3.31 / 0.86, S–P 39.2 / 26.8, S–GG̃ 43.9 / 30.2, Q17 75.5 / 50.2, Q18 6.76 / 4.52, Q15-charge 59.2 / 39.0, P–P isov 5.60 / 1.36 GeV.

**Naturalness/EFT validity.** The direct-detection EFT only needs Λ ≫ q ≈ 0.25 GeV, but (i) the O6-type structures need Λ = 1.5–2.7 GeV (0.8–1.4 GeV at 4 TeV) — at or below the 2 GeV matching scale, so the quark-level contact description is not valid at all; any O6 explanation must come from a light pseudoscalar mediator (m_a ≲ q) where the contact form itself fails, or from the nucleon-level L4 with a non-perturbative origin; (ii) the O10, O9, O14 and L10-like structures need Λ = 5–62 GeV, above 2 GeV but far below M_Z, so directdm's running from the electroweak scale is not applicable (the mediator must be integrated out below M_Z) and Λ/m_χ ≈ 0.005–0.06: for a mediator of mass M and couplings g_χ g_q/M² Λ' ~ 1/Λ³ the mediator is lighter than the DM by one to two orders of magnitude unless couplings are non-perturbative; (iii) only the O1/O4-type dim-6 currents (Λ = 2–90 TeV) are "natural" heavy-mediator contact interactions, and precisely these are excluded by the low-energy null. The magnetic moment 9.3 × 10⁻⁷ μ_N and the anapole a = 2.7 × 10⁻⁶ GeV⁻² = (605 GeV)⁻² would be natural sizes, but both predict N_lo ≥ 27. Compared with P012's dimension-8 L10 strength d₁₀/(m_v² m_N²) = (21 GeV)⁻⁴, the dimension-7 contact analogue Q15 (charge weights) needs (48 GeV)⁻³ — the same "tens of GeV" scale.

### 3.4 RG running M_Z → 2 GeV (`P031_rg_running.csv`, 1000 GeV, q = q_event)
- Tensor operators C79, C710: coefficient 1 → 0.5734 (QCD anomalous dimension of m_q q̄σq); the O4 (and O11/O12) NR coefficients scale by 0.573, so a tensor Λ quoted at 2 GeV corresponds to Λ(M_Z) = 0.573^{1/3} Λ = 0.83 Λ. **No tensor → dipole (C51) mixing is generated by directdm 2.2.2** (checked to 10⁻¹⁴); the code's ADM does not contain it at this order.
- Gluon GG̃ operators C73/C74: self-renormalisation +1.0 % and mixing into the quark pseudoscalars C77q/C78q at 6.0 × 10⁻³ per unit C73/C74 (NR c₁₀/c₆ change by +1.8 %).
- Vector, axial, scalar, pseudoscalar quark currents and the dipoles: no running at this order for flavour-universal inputs; for a single-flavour vector coupling (C61u) QED penguins generate C61d,s = 3.4 × 10⁻³, C61c = −6.8 × 10⁻³, C61ℓ = 1.0 × 10⁻² and NR c₃, c₄, c₆ at ≤ 10⁻⁴ of c₁ (probe run, not tabulated).
- Consequence: running does not move any structure between classes; it only rescales the tensor Λ by 17 %.

### 3.5 Anand Lagrangians and Table S6 (`P031_tableS6_comparison.csv`, Fig. 3)
Identifications (P003 R5 for L1–L5, L15; LZ supplement degeneracies L1≡L5, L2≡L8, L3≡L17, L4≡L20, L11≡L14; tex lines 494–497) and predicted vs LZ local Z at 1000 GeV:

| Anand L | relativistic structure | confidence | Z_LZ | N_lo | Z_pred |
|---|---|---|---|---|---|
| L1^s, L5^s (S–S, V–V) | S–S / S–GG / V–V univ | certain | 0.0 | 2760 | 0.16 |
| L5^v | V–V isov | certain | 1.3 | 477 | 1.43 |
| L3^s (P–S) | P–S univ / P–GG | certain | 1.7 | 260 | 1.76 |
| L2^s (S–P) | S–P univ (π–η cancellation) | certain | 3.0 | 0.85 | 3.49 |
| **L2^v** | S–P isov (pion pole) | certain | **3.1** | **25.8** | **2.73** |
| L4^s (P–P) | P–P univ | certain | 3.1 | 0.069 | 3.59 |
| L4^v | P–P isov | certain | 3.1 | 3.33 | 3.29 |
| L15^s (A–A) | A–A univ | likely | 2.7 | 28.2 | 2.70 |
| L15^v | A–A Z-like | likely | 2.7 | 55.9 | 2.45 |
| L7 (V–A) | V–A univ → O7 | likely | 2.7 | 31.5 | 2.66 |
| L6^s (dipole × N̄γ_μN) | Q15 univ | likely | 2.6 | 15.3 | 2.90 |
| L16 = anapole? | e Q_q A–V | uncertain | 3.4 | 26.9 | 2.72 → **rejected** |
| L8 = A–V? | A–V univ | uncertain | 3.0 | 167 | 1.98 → rejected (LZ: L8 ≡ L2 ∝ O10) |
| L13 (A–V) | A–V univ | uncertain | 2.5 | 167 | 1.98 |
| L9 / L11≡L14 (dipole × axial) | Q17 → O9 | uncertain | 3.0 / 3.2 | 2.06 | 3.38 |

The certain and likely identifications reproduce Table S6 with rms 0.25σ except L2^v (and L4^s, saturated at the top of the P016 curve): LZ's L2 is the nucleon-level contact O10, s/v-indistinguishable (LZ supplement), whereas the quark-level isovector pseudoscalar current is pion-pole dominated and drops to 2.7σ. The 3.3–3.4σ Lagrangians L10 and L16 do correspond to the compatible set (L10 = pure transverse spin, P003/P012), but L16 is *not* the anapole: the anapole's proton-charge O8 term carries 90 % of the 20 keV rate. The best relativistic matches to the 3.2–3.4σ rows are the pure-O9 dipole × axial-current operator (Q17; L9, L11≡L14) and the charge-weighted Q15 (L10-like, N_lo 2.5).

### 3.6 Dirac vs Majorana
Majorana DM has no vector or tensor currents and no dipoles, so of the compatible set only P–P (O6), S–P (O10), S–GG̃ (O10) and P–GG̃ (O6) survive — all requiring Λ ≲ 37 GeV and, for O6, Λ ≈ 1.5–1.7 GeV. Every compatible structure with a natural-looking scale (Q17 O9 at 62 GeV, Q15-charge at 48 GeV) needs Dirac DM. Wino-/Higgsino-like Majorana electroweak multiplets couple through A–A (Z) and S–S (Higgs), both excluded here, in line with P007's conclusion that the elastic Higgsino cannot fit and only inelastic scattering can.

## 4. Figures
- `figures/P031_nr_coefficient_map.png` — Fig. 1: fraction of the 248 keV rate (diagonal terms) from each NR operator for every structure at 1000 GeV; rows sorted by N_lo and coloured by class.
- `figures/P031_lambda_vs_mass.png` — Fig. 2: Λ for 1.0 (bars 0.3–2.4) LZ events vs m_χ for compatible/marginal structures; lines Λ = m_χ, M_Z, 2 GeV.
- `figures/P031_Z_vs_tableS6.png` — Fig. 3: predicted Z (N_lo through the P016 curve) vs LZ Table S6 at 1000 GeV.

## 5. Caveats
- Tree-level matching (directdm LO; the NLO coherent tensor terms `NLO=True` not used); nucleon form factors at q = 0 except the explicit poles and the strangeness slope; no two-body currents.
- Efficiency model and N_lo thresholds as in P003/P016; Z_pred uses P016's baseline curve, which saturates at 3.59 and has 0.2σ scatter; b_H anchored to LZ's 3.4σ.
- Leading-operator fractions ignore interference (the full spectra include it; the ratio is logged: 0.036–1.02).
- Anand L identifications beyond L1–L5, L15 rest on the LZ degeneracy statement and on the significance pattern (flagged).
- Q15–Q18 structures inferred from directdm's output, not from the papers.
- Λ assumes a single operator and C real; interference between several relativistic operators could change N_lo.

## 6. References
LZ Collaboration, arXiv:2609.02823 (2026). F. Bishara, J. Brod, B. Grinstein, J. Zupan, JCAP 02 (2017) 009 [arXiv:1611.00368]; JHEP 11 (2017) 059 [arXiv:1707.06998]; directdm code (arXiv:1708.02678). J. Brod, A. Gootjes-Dreesbeimdiek, M. Tammaro, J. Zupan, JHEP 10 (2018) 065 [arXiv:1801.04240]. N. Anand, A. L. Fitzpatrick, W. C. Haxton, PRC 89 (2014) 065501. A. L. Fitzpatrick et al., JCAP 02 (2013) 004. S. Kang, S. Scopel, G. Tomar, J.-H. Yoon, WimPyDD, CPC 276 (2022) 108342. E. Del Nobile, "The Theory of Direct Dark Matter Detection", Lect. Notes Phys. 996 (2022) [arXiv:2104.12785]. Corpus: P003, P007, P012, P016, P027.

## 7. Tools and provenance (mirrors `output/provenance/P031.json`)
- Software: python 3.12.13; directdm 2.2.2 (`WC_3f`, `WC_5f`, `WC_4f`, `.cNR`, `._my_cNR`, `.match`, `ip` inputs); WimPyDD 2.0.4 (`eft_hamiltonian` with tuple keys and q-dependent coefficients, `diff_rate` via `lz.wd_rate`); numpy 2.5.3 (with an in-script `np.delete` wrapper); scipy 1.18.1 (`special.erf`, `integrate.trapezoid`); pandas 3.0.5; matplotlib 3.11.2; `common/lzcommon.py` (`LZ`, `LSIG`, `LSIG_MASSES`, `M_V_GEV`, `M_NUCLEON_GEV`, `A_XE_MEAN`, `m_nucleus_gev`, `wd`, `wd_halo`, `wd_rate`, `wd_c_from_anand`).
- Script and commands: `output/code/P031_directdm_matching.py` run as `.venv/bin/python output/code/P031_directdm_matching.py --masses 1000,200,4000 --final` (full grid, cache), `--masses 1000 --final`, `--masses 200,4000 --only PP_isov,Q15_charge,Q17_univ,Q18_univ --final`, `--skip-spectra --final` (aggregation used).
- Local inputs: `inputs/LZ_arXiv_2609.02823_fulltext.tex` (Theory paragraph lines 35–74; LEE degeneracies 494–497; Table S6 lines 825–878; Table S7 880–909); `output/00_evidence_dossier.md`; `output/results_ledger.csv`; `output/papers/P003.md`, `P012.md`, `P016.md`, `P027.md`; `output/code/P012_magnetic_dipole.py` (efficiency, L10 Hamiltonian); `output/work/P012/P012_coupling_table.csv`; `output/work/P003/P003_operator_table.csv`, `details.md`; `output/work/P016/P016_Z_vs_Nlo_curves.csv`, `details.md`; directdm and WimPyDD source files (read for conventions).
- Recalled knowledge: operator definitions and normalisations of the directdm basis (likely); μ_χ = e C51/(4π²) (derived from the code's dipole coefficients, likely); α = 1/137.036 (certain); Z (T₃) axial charges ±½ (certain); quark charges (certain); standard dipole NR reduction c₁, c₄, c₅, c₆ (likely); Anand L1–L5, L15 structures (likely, from P003); L6/L7 = dipole×vector, vector×axial (likely); nothing else.
- WimPyDD-generated files: none (`diff_rate` does not write response files); cache under `output/work/P031/cache/` is our own.
