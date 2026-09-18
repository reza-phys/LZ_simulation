# P089 — Identical spectra, different couplings: the isoscalar/isovector and Lagrangian degeneracies of LZ's Table S6 and the coupling ratios that relate them (research record)

Simulated date 2026-09-16. Author profile: EFT phenomenology group. Category EFT (hep-ph).
Script: `output/code/P089_table_s6_degeneracies.py` (run from `/Users/reza/LZ_simulation` with `.venv/bin/python`; first call computes and caches 60 WimPyDD spectra in 239 s, later calls analyse from cache in ~1 s). Outputs in `output/work/P089/`.

## 1. Motivation and scope

LZ (arXiv:2609.02823) reports local significances for 40 Lagrangian entries L1–L20 (isoscalar s, isovector v; Table S6, tex lines 825–878) and for O1/O4 elastic and inelastic (Table S7, lines 880–909), and states (LEE section, lines 494–498) that L1/L5, L2/L8, L3/L17, L4/L20 and L11/L14 "differ only by a scalar constant", that s and v are indistinguishable for L2, L4, L7, L8, L10, L11, L14, L15, L19, L20 and inelastic O4, and that masses ≥ 400 GeV are nearly degenerate. The Lagrangians follow Anand, Fitzpatrick and Haxton (PRC 89, 065501, 2014; "Anand"), with unit couplings d_i = 1/m_v² (tex line 71).

P057 worked at the operator level (O6 = E_R × O10 exactly; O9 ≡ O14; isoscalar ≡ isovector for Σ″ operators). P031 and P044 identified a few Lagrangians by pattern matching. This paper does the Lagrangian-level job: (1) derive the nonrelativistic (NR) reduction of every L_i, flagging reliability; (2) compute the xenon spectra of all 40 L_i^{s,v} and the 28 O_i^{s,v} at 1 TeV with WimPyDD; (3) cluster them by shape in LZ's observed-energy space; (4) give the coupling ratios that map one entry's best-fit coupling onto another's; (5) explain the Table S6 pattern; (6) count the distinct spectral classes for the look-elsewhere discussion.

## 2. The Anand Lagrangians and their NR reductions

### 2.1 Recalled structure (reliability: likely)

Anand's Table 1 consists of the four scalar/pseudoscalar products L1 = χ̄χ N̄N, L2 = iχ̄χ N̄γ⁵N, L3 = iχ̄γ⁵χ N̄N, L4 = χ̄γ⁵χ N̄γ⁵N, followed by the 4 × 4 grid of DM bilinears {V = χ̄γ^μχ, T = χ̄iσ^{μν}q_ν/m_M χ, A = χ̄γ^μγ⁵χ, T̃ = iχ̄iσ^{μν}q_ν/m_M γ⁵χ} against nucleon bilinears {V, T, A, T̃} in the order L5 = V·V, L6 = V·T, L7 = V·A, L8 = V·T̃, L9 = T·V, L10 = T·T, L11 = T·A, L12 = T·T̃, L13 = A·V, L14 = A·T, L15 = A·A, L16 = A·T̃, L17 = T̃·V, L18 = T̃·T, L19 = T̃·A, L20 = T̃·T̃ (factors of i where needed for hermiticity; m_M is the tensor mass scale, taken equal to m_N as in P012 — recalled/likely). This ordering is the only one we found that reproduces all five of LZ's "scalar constant" pairs (§2.3), which we take as its confirmation.

### 2.2 Derivation (standard Dirac-bilinear reductions; certain)

With Anand's normalisation (each bilinear divided by 2m), q = p′ − p on the nucleon, q⁰ set to zero, P = p + p′, v_N = P_N/2m_N, v_χ = P_χ/2m_χ and v⊥ = v_χ − v_N (up to an overall sign convention), the leading NR forms of the bilinears are

| bilinear | time component | space component |
|---|---|---|
| S = N̄N | 1 | — |
| P = N̄iγ⁵N | ∓ i q·S_N/m_N = ∓O10 | — |
| V^μ | 1 | v_N − i(q × S_N)/m_N |
| T^μ = N̄iσ^{μν}q_νN/m_M | [q²/2m_N − 2i S_N·(q × v_N)]/m_M | 2i(q × S_N)/m_M |
| A^μ | 2 S_N·v_N | 2 S_N |
| T̃^μ = N̄iσ^{μν}q_νγ⁵N/m_M | −2 q·S_N/m_M | −2 v_N (q·S_N)/m_M |

and the same for the DM with N → χ (the 1/m_χ-suppressed pieces of V and the q²/2m_χ Darwin term of T are kept only where they are the leading term of a Lagrangian). Contracting A·B = A⁰B⁰ − A·B and using q·P_N = 0 (elastic) gives, in terms of Anand's O_i (O3 = iS_N·(q/m_N × v⊥), O5 = iS_χ·(q/m_N × v⊥), O6 = (S_χ·q/m_N)(S_N·q/m_N), O7 = S_N·v⊥, O8 = S_χ·v⊥, O9 = iS_χ·(S_N × q/m_N), O10 = iS_N·q/m_N, O11 = iS_χ·q/m_N, O12 = S_χ·(S_N × v⊥), O13 = i(S_χ·v⊥)(S_N·q/m_N), O14 = i(S_χ·q/m_N)(S_N·v⊥), O15 = −(S_χ·q/m_N)((S_N × v⊥)·q/m_N)), the reductions of Table 1. Two derivation details matter:

- Every velocity-dependent term appears as v_χ − v_N, i.e. Galilean invariance is manifest (L6: −2iS_N·(q × (v_N − v_χ)); L9: −2iS_χ·(q × (v_χ − v_N)); L12, L16, L18, L19 likewise). This is a non-trivial check of the algebra.
- L12 produces the operator (S_N·q)(S_χ·(q × v⊥))/m_N². Decomposing S_N in the orthogonal basis {q, v⊥, q × v⊥} (q ⊥ v⊥) gives the identity (S_N·q)(S_χ·(q × v⊥))/m_N² = (q²/m_N²) O12 + O15 (derived; the same operator Fitzpatrick et al. call O16). Because the nucleon spin enters longitudinally, the Σ′ and Φ″ responses — which couple to the combination c12 − (q²/m_N²)c15 in Anand's response functions — must cancel exactly, leaving q⁴v²Σ″ and q⁴Φ̃′. WimPyDD confirms this: flipping the relative sign of c12 and c15 raises the rate by ×5400 (s) and ×218 (v) (`P089_closure_checks.csv`).

**Table 1 — NR reductions at unit coupling d_i (Anand normalisation, m_M = m_N, m_χ = 1 TeV, R ≡ m_N/m_χ = 9.38 × 10⁻⁴).** Overall signs are irrelevant for spectra; relative signs inside a Lagrangian matter only where marked.

| L | reduction | responses (q-power, velocity) | reliability |
|---|---|---|---|
| L1 | O1 | M | certain |
| L2 | O10 | q²Σ″ | certain |
| L3 | −R O11 | q²M | certain |
| L4 | −R O6 | q⁴Σ″ | certain |
| L5 | O1 (+O(v², q²/m²)) | M | certain |
| L6 | (q²/2m_N²) O1 + 2 O3 | q⁴M + q²Φ″ + q⁴Φ″M + q²v²Σ′ | likely; O1–O3 relative sign uncertain |
| L7 | −2 O7 + 2R O9 | v²Σ′ + R²q²Σ′ | likely |
| L8 | −2 O10 | q²Σ″ | likely (LZ: ∝ L2) |
| L9 | (q²/2m_χm_N) O1 − 2 O5 − 2[(q²/m_N²)O4 − O6] | R²q⁴M + q²v²(M+Δ) + q⁴Σ′ (Σ″ cancels) | likely; O5 sign uncertain |
| L10 | 4[(q²/m_N²)O4 − O6] (+ (q⁴/4m_χm_N³) O1) | q⁴Σ′ | certain (P003/P012 vs LZ Fig. 1) |
| L11 | −4 O9 | q²Σ′ | likely |
| L12 | −4[(q²/m_N²)O12 + O15] − R(q²/m_N²) O10 | q⁴v²Σ″ + q⁴Φ̃′ + R²q⁴Σ″ | likely (derived) |
| L13 | 2 O8 − 2 O9 | v²(M+Δ) + q²Σ′ − ΔΣ′ | likely; O8–O9 sign uncertain |
| L14 | 4 O9 | q²Σ′ | likely (LZ: ∝ L11) |
| L15 | −4 O4 | Σ′ + Σ″ | certain |
| L16 | −4 O13 | q²v²Σ″ + q²Φ̃′ | likely (q⁰ dropped) |
| L17 | −2 O11 | q²M | likely (LZ: ∝ L3) |
| L18 | −(q²/m_N²) O11 + 4 O15 | q⁶M + q⁴Φ″ + q⁶Φ″M + q⁴v²Σ′ | likely; O11–O15 sign uncertain |
| L19 | 4 O14 | q²v²Σ′ | likely |
| L20 | 4 O6 | q⁴Σ″ | likely (LZ: ∝ L4) |

### 2.3 Consistency with LZ's own statements

- The five "scalar constant" pairs are reproduced exactly: L1/L5 (both O1), L2/L8 (O10, factor 2), L3/L17 (O11, factor 2m_χ/m_N), L4/L20 (O6, factor 4m_χ/m_N), L11/L14 (O9, factor −1). No sixth exact pair exists among the 40 entries (Table 2).
- The ten Lagrangians LZ declares isoscalar/isovector-indistinguishable are exactly the ten whose reduction contains no M, Δ or Φ″/Φ̃′ response (pure Σ′/Σ″): our Hellinger distance between s and v is 0.006–0.032 for all ten and 0.135–0.52 for the other ten (`P089_isoscalar_isovector.csv`). The reductions therefore reproduce LZ's list without any tuning.

## 3. Inputs

| input | value / source |
|---|---|
| Table S6/S7 significances | `lz.LSIG`, `lz.OSIG` (tex lines 834–873, 889–904) |
| WimPyDD 2.0.4 | `lz.wd()`, `eft_hamiltonian` with closures (P031 pitfall), `WD.diff_rate` via `lz.wd_rate`; natural xenon shell-model responses |
| halo | `lz.wd_halo()` Sun-frame Baxter-2021 SHM, explicit v_min grid (P002/P035); elastic shapes insensitive (P057: 0.0008 dex) |
| coupling convention | LZ unit d_i = c_i = 1/m_v² (Anand) = WimPyDD c^τ = 2/m_v² (P003; τ = 0 isoscalar, τ = 1 isovector) |
| nucleon mass, m_M | m_N = 0.938 GeV (`lz.M_NUCLEON_GEV`); m_M = m_N |
| true-energy grid | 1.25–596.25 keV, 5 keV steps (120 points); P057's 2.5 keV cache sub-sampled for O1s, O4s, O5v, O6s/v, O9s/v, O10s/v, O13s, O14s/v, O15s/v, L10s |
| efficiency | 0.96 × ½[1+erf((E−5.4)/(√2·3.4))] × ½[1−erf((E−269.9)/(√2·8.0))] (P003 model of LZ's 50 % points) |
| resolution | σ_E = 11√(E/248) keV (P009/P021), Gaussian, observed grid 0.5–699.5 keV in 1 keV bins |
| shape metrics | Hellinger H = √(1 − Σ√pq); KL with 1 % flat floor (P050/P057); Gaussian N_3σ = 9 max(Var)/(D_pq+D_qp)² (P057 convention) |
| N_lo | accepted rate 5.4–55 keV per accepted event 200–270 keV (P003) |
| Z(N_lo) proxy | `output/work/P016/P016_Z_vs_Nlo_curves.csv`, baseline column |
| other corpus numbers | P012 d10 = 0.28 (LZ normalisation); P057 O6/O10, O9/O14, O6s/O6v numbers; P032 isovector M nodes 69/182 keV vs isoscalar 102/267 keV; P001 N_eff = 13.9; P008 12.2; P027 B_max/B_DM = 10.3 |

Spectra computed: 40 L_i^{s,v} (single-operator entries taken as factor² × the operator spectrum; L3s, L8s, L20s also computed explicitly as a bookkeeping check), 28 O_i^{s,v}, 10 sign variants (L6, L9, L12, L13, L18 × s, v), L10s with the dropped q⁴O1 term. 60 WimPyDD evaluations of 120 points, 2–10 s each.

## 4. Results

### 4.1 Bookkeeping checks (`P089_closure_checks.csv`)

L8s/O10s = 4.000, L20s/O6s = 16.000, L3s/O11s = 8.80 × 10⁻⁷ = (m_N/m_χ)² to 3 × 10⁻¹⁶ relative scatter. L10s with the (q⁴/4m_χm_Nm_M²)O1 term differs from the leading form by ≤ 4 × 10⁻⁶ (10–400 keV). L10s events at d10 = 1: 3.34 in 2.84 t·yr (P012: 3.34). O6s/O6v H = 0.008, O9/O14 same class (H 0.018) as in P057. A first run had omitted the 1/m_v² in the coefficient closures; shapes were unaffected but the rate checks caught it (ratios 3.67 × 10⁹ = m_v⁴ too large); the cache was cleared and recomputed.

### 4.2 Spectral classes (`P089_hellinger_matrix.csv`, `P089_symKL_matrix.csv`, `P089_results.json: class_counts, clusters`; Fig. `P089_dendrogram.png`)

Average-linkage clustering of the observed-energy pdfs. Number of classes vs Hellinger threshold (Gaussian N_3σ ≈ 1500 at H = 0.02, ≈ 250 at 0.05, ≈ 60 at 0.10, ≈ 25 at 0.15):

| set | H < 0.005 | 0.01 | 0.02 | 0.03 | 0.05 | 0.10 | 0.15 | 0.20 |
|---|---|---|---|---|---|---|---|---|
| 40 Table S6 entries | 30 | 27 | 24 | 24 | 22 | 21 | 19 | 16 |
| + elastic O1^{s,v}, O4^{s,v} (44) | 30 | 27 | 24 | 24 | 22 | 21 | 19 | 16 |
| 25 entries with Z ≥ 3.0 at 1 TeV | 19 | 17 | 14 | 14 | 13 | 13 | 12 | 10 |
| all 68 (adding the 28 O_i^{s,v}) | 40 | 37 | 34 | 34 | 31 | 29 | 19 | 17 |

Robustness: with true-energy (unsmeared) pdfs the counts are 30/27/25/24/22/21/20/16 for Table S6; with 1.5× resolution 30/27/24/23/22/21/19/15. The count at H < 0.005 (30) is exactly LZ's 40 minus the five exact pairs; the 24 at H < 0.02 is what remains after the effective degeneracies below.

**Table 2 — the 24 classes of Table S6 at 1 TeV (H < 0.02), ordered by LZ significance.** N_lo and the true-energy peak of the accepted spectrum are for the representative.

| class (representative) | members | responses | LZ Z (1 TeV) | N_lo | peak [keV] |
|---|---|---|---|---|---|
| L10s | L10s, L10v | q⁴Σ′ | 3.4 | 0.20 | 196 |
| L16s | — (= O13s) | q²v²Σ″ + q²Φ̃′ | 3.4 | 4.7 | 36 |
| L6v | — | q⁴M + q²Φ″ + q⁴Φ″M + q²v²Σ′ | 3.3 | 1.41 | 26 |
| L11s | L11s, L14s, L19s (= O9s, O14s) | q²Σ′ | 3.0–3.2 | 2.07 | 16 |
| L11v | L11v, L14v, L19v (= O9v, O14v) | q²Σ′ | 3.0–3.2 | 2.21 | 16 |
| L12s | — | q⁴v²Σ″ + q⁴Φ̃′ | 3.2 | 1.03 | 51 |
| L12v | — | idem, isovector | 3.2 | 1.28 | 66 |
| L16v | — (= O13v) | q²v²Σ″ + q²Φ̃′ | 3.2 | 9.0 | 46 |
| L20s | L4s, L4v, L20s, L20v (= O6s, O6v) | q⁴Σ″ | 3.1–3.2 | 0.40 | 131 |
| L9v | — | q²v²(M+Δ) + q⁴Σ′ | 3.2 | 0.69 | 31 |
| L18s | — | q⁶M + q⁴Φ″ + q⁶Φ″M + q⁴v²Σ′ | 3.1 | 4.2 | 146 |
| L2v | L2s, L2v, L8s, L8v (= O10s, O10v) | q²Σ″ | 3.0–3.1 | 2.95 | 36 |
| L13v | — | v²(M+Δ) + q²Σ′ | 3.0 | 6.4 | 16 |
| L9s | — (≈ O5s, H 0.075) | q²v²(M+Δ) + q⁴Σ′ | 3.0 | 7.8 | — |
| L18v | — | q⁶M + q⁴Φ″ + q⁴v²Σ′ | 2.9 | 0.18 | 241 |
| L7s | — (= O7s) | v²Σ′ | 2.7 | 31 | — |
| L7v | — (= O7v); joins L7s at H < 0.05 | v²Σ′ | 2.7 | 34 | — |
| O4s | L15s, L15v (= O4s, O4v) | Σ′ + Σ″ | 2.7 | 28 | — |
| L3v | L3v, L17v (= O11v) | q²M | 2.6 | 35 | — |
| L6s | — | q⁴M + q²Φ″ + … | 2.6 | 32 | — |
| L13s | — (≈ O8s, H 0.032) | v²(M+Δ) + q²Σ′ | 2.5 | 84 | — |
| L17s | L3s, L17s (= O11s) | q²M | 1.7–1.8 | 261 | — |
| L1v | L1v, L5v (= O1v) | M | 1.3 | 473 | — |
| L1s | L1s, L5s (= O1s) | M | 0.0 | 2770 | — |

Selected distances: H(L12s, L20s) = 0.30, H(L12v, L20v) = 0.32, H(L12s, L12v) = 0.24, H(L12s, L16s) = 0.23; H(L16s, L2s) = 0.29, H(L16s, L20s) = 0.45, H(L16s, L16v) = 0.25; H(L9v, L10s) = 0.18, H(L9v, O5v) = 0.13, H(L9s, O5s) = 0.075; H(L6v, O3v) = 0.38, H(L6v, L10s) = 0.40; H(L18v, L10s) = 0.30, H(L18v, O15v) = 0.47; H(L13v, O8v) = 0.105, H(L13v, O9v) = 0.22; H(L10s, L20s) = 0.18, H(L10s, L2s) = 0.37, H(L20s, L2s) = 0.24, H(L11s, L2s) = 0.23. Effective degeneracies: L19 vs L11/L14 (O14 vs O9) H = 0.018, Gaussian N_3σ = 1140–1180; L10v vs L10s H = 0.019, N_3σ = 3240; L7 vs O7 H = 0.002; L13s vs O8s H = 0.032; L7s vs L7v 0.030 (N_3σ 1400); L11s vs L11v 0.032 (1180).

Three findings correct expectations in the corpus: (i) L12 is *not* O6-like although it shares the q⁴Σ″ structure — the Φ̃′ response and the v² weighting make it a separate class (peak 51 keV vs 131 keV; log-log slope of L12s/L20s −0.63 with 0.46 dex scatter). (ii) L16 = O13 (q²v²Σ″ + q²Φ̃′) is not a q⁴-spin structure degenerate with L10 as P044 inferred: H(L16s, L10s) > 0.4, and O13's N_lo = 4.7 is compatible with LZ's 3.4σ once the P016 calibration curve is used (§4.4). (iii) L9 is a mixture of O5 (q²v²(M+Δ)) and the L10 transverse-spin structure; the isoscalar version is O5s-dominated (H 0.075 to O5s) and the isovector one sits between O5v and L10 (H 0.13 / 0.18).

### 4.3 Coupling ratios (`P089_exact_pairs.csv`, `P089_mapping_table.csv`, `P089_results.json: operator_coupling_maps`)

Exact (constant-factor) maps, Anand normalisation, at 1 TeV: c1 = d1 = d5; c10 = d2 = −2 d8, so d2 = 2 d8; c11 = −(m_N/m_χ) d3 = −2 d17, so d3 = 2(m_χ/m_N) d17 = 2132 d17; c6 = −(m_N/m_χ) d4 = 4 d20, so d4 = 4(m_χ/m_N) d20 = 4263 d20; c9 = −4 d11 = 4 d14, so |d11| = |d14|; c4 = −4 d15 (LZ's Table S7 O4 coupling (c4 m_v²)² equals 16 (d15 m_v²)²); c13 = −4 d16; c14 = 4 d19; c7 = −2 d7 with c9 = 2(m_N/m_χ) d7; c8 = 2 d13, c9 = −2 d13. Rates scale as the square of these factors (verified to 10⁻¹⁶ by the explicit L3s/L8s/L20s spectra). Converting LZ's L10 fit: d10 = 0.28 (P012, LZ normalisation, provisional pending DR-001) means c4 = 1.12 q²/m_N² and c6 = −1.12 in units of 1/m_v²; P057's O6 normalisation of one accepted event (Anand c6 m_v² = 1.38) corresponds to d20 = 0.345 or d4 = 1470 (= 4m_χ/m_N × d20).

Rate-matched maps for effective degeneracies (equal accepted 200–270 keV rate at 1 TeV): d19 = 1425 d11 (the ⟨v⊥²⟩ suppression of O14 relative to O9: rate ratio 4.9 × 10⁻⁷); d10^v = 1.134 d10^s; d11^v = 1.135 d11^s; d2^v = 1.044 d2^s; d20^v = 1.044 d20^s (isovector Σ responses are 8–22 % weaker at fixed coupling); d16 = 0.25 c13.

LZ's tables give significances, not couplings, so the maps are checked against the significances of the exact pairs: the 50 differences (five pairs × s, v × five masses ≥ 100 GeV) have rms 0.12σ and maximum 0.4σ (L1v/L5v at 400 GeV: 0.3 vs 0.7; L2v/L8v at 200 GeV: 2.5 vs 2.9). This is the reproducibility floor of Tables S6/S7 (30 000 toys per entry): differences below 0.4σ between entries carry no spectral information. Because the s/v Σ-pairs are also degenerate at H ≤ 0.03, the six-entry "O6 block" (L4s, L4v, L20s, L20v; LZ Z 3.0–3.3 across masses) and the "O10 block" (L2, L8; 2.3–3.1) are single measurements each.

### 4.4 Why Table S6 looks the way it does (`P089_mapping_table.csv`; Fig. `P089_Z_vs_Nlo.png`)

The local significance at 1 TeV follows the low-energy companion count N_lo: the P016 curve Z(N_lo) reproduces the 40 entries with rms 0.25σ (0.23σ without L18v; maximum 0.67σ at L18v), and a two-parameter fit Z = 5.94 + 0.213 log₁₀N_lo + 1.05 log₁₀p(248 keV) gives 0.26σ (log N_lo alone 0.38σ). The pattern by response type:

- **M-dominated (O1: L1, L5; q²M: L3, L17; L6s).** Coherent, falling spectra: N_lo = 2770 (O1s), 473 (O1v), 261 (O11s), 35 (O11v), 32 (L6s) → Z = 0, 1.3, 1.7–1.8, 2.6, 2.6. The isovector entries score higher because the isovector amplitude ZF_p − NF_n has its nodes at 69 and 182 keV instead of 102 and 267 keV, so 248 keV sits at the isovector post-node maximum and 19 keV below the isoscalar node (P032): O1v/O1s rate at equal c_p is 0.022 at 5.4–55 keV but 0.99 at 248 keV, hence Z = 1.3 for O1v against 0.0 for O1s (P032 explains the residual 0.5σ by nuclear-structure modifications LZ mentions). Each q² divides N_lo by ≈ 10 (P003): O1s → O11s 2770 → 261; the extra q² of L18 over L3 (q⁶M) plus its Φ″ part brings L18s to N_lo 4.2 and 3.1σ.
- **Σ″ (L2, L8 = O10, q²; L4, L20 = O6, q⁴).** N_lo = 2.95 → 3.0–3.1σ and 0.40 → 3.1–3.2σ. Isoscalar and isovector Σ″ spectra are identical to H = 0.008 (P057), so the s/v entries duplicate each other; O6 = E_R × O10 exactly (P057), and both sit on the saturated part of the Z(N_lo) curve where the peak position (131 keV for O6, 36 keV for O10) rather than N_lo sets the remaining 0.2–0.4σ differences — hence O6 (3.1–3.2) below L10 (3.4, peak 196 keV) despite the smaller N_lo.
- **Σ′ (L11, L14 = O9, q²; L19 = O14, q²v²; L10 = q⁴).** N_lo 2.07–2.5 → 3.0–3.2σ; L19's 3.0 against L11's 3.2 is within the 0.4σ floor for a pair with H = 0.018. L10's q⁴Σ′ (N_lo 0.20) is the hardest pure-spin spectrum and gives the table's maximum 3.4σ; the O4 + O6 interference that cancels Σ″ (P057) is what makes L10 companion-free. L7 (v²Σ′) keeps N_lo = 31–34 → 2.7σ: velocity suppression alone does not harden the spectrum (P003).
- **Σ′ + Σ″ without q (L15 = O4).** N_lo 28 → 2.7σ, identical for s and v (H 0.006), matching Table S7's O4 row at δ = 0 (2.7).
- **Σ″v² + Φ̃′ (L16 = O13, q²; L12, q⁴).** N_lo 4.7/9.0 (L16 s/v) and 1.03/1.28 (L12) → 3.4/3.2 and 3.2/3.2σ. Isoscalar and isovector differ (H 0.25, 0.24) because Φ̃′ has a different isospin structure from Σ″, which is why LZ does not list L12 and L16 as s/v-degenerate.
- **Mixtures with M or Δ (L6, L9, L13; L18).** These carry an isoscalar-coherent component that is isovector-suppressed by (N−Z)²/A² = 0.03, so the isovector entry is always harder: L6 N_lo 32 → 1.4 (2.6 → 3.3σ), L9 7.8 → 0.69 (3.0 → 3.2σ), L13 84 → 6.4 (2.5 → 3.0σ), L18 4.2 → 0.18 (3.1 → 2.9σ). The last is the one entry the reduction does not explain: L18v as derived is the hardest spectrum of all (peak 241 keV, highest pdf at 248 keV, 0.0089 keV⁻¹) and the proxy predicts 3.4–3.6σ (both signs of the O11–O15 cross term), yet LZ reports 2.9σ at every mass. Either Anand's L18 differs from our T̃·T reduction (e.g. in the relative weight of the q²O11 and O15 pieces), or LZ's two-dimensional likelihood penalises a spectrum concentrated at the efficiency edge in a way the one-dimensional proxy cannot capture. The released L18 spectrum (Data Release; cf. DR-001/DR-002 for L10/O1) would settle it.

Sign variants (`P089_sign_variants.csv`): flipping the uncertain relative sign changes N_lo by 0.5–9× (L6v 1.4 → 12.5, L18v 0.18 → 1.42, L13s 84 → 158, L9s 7.8 → 14.5) and the predicted Z by ≤ 0.5σ; the sign used in Table 1 matches LZ better for L6v (3.43 vs 2.96 predicted; LZ 3.3) and L13s (2.29 vs 2.01; LZ 2.5), is indifferent for L9, L12, L18s, and neither sign rescues L18v.

### 4.5 The count for the look-elsewhere discussion

At 1 TeV LZ's 40 Table S6 entries contain 24 spectral classes distinguishable with ≲ 1500 events (H > 0.02), 22 with ≲ 250 (H > 0.05), 21 with ≲ 60 and 19 with ≲ 25 events. Restricted to the 25 entries with Z ≥ 3.0 — the ones that can set the maximum local significance — the count is 14 (H < 0.02), 13 (0.05–0.10) and 12 (0.15). For a single event the relevant notion of "different trial" is a spectrum that gives a different p-value, which for shapes this close is coarser still; the frequentist and Bayesian estimates in the corpus (P001 N_eff = 13.9 from p_global/p_local; P008 12.2 ± 0.5 from toys over 456 models; P027 B_max/B_DM = 10.3) all fall at the 12–14 classes we count among the compatible spectra, not at LZ's 293 or at the 616 raw models. Adding the 28 elastic O_i^{s,v} raises the total to 34 classes (H < 0.02); the inelastic entries of Table S7 add further classes in δ (P062), outside our scope.

## 5. Validation and robustness

- LZ's five constant pairs and its ten-member s/v list are reproduced without tuning (§2.3); N_lo values reproduce P003/P057 (O6 0.40 vs 0.43 on the coarser grid, O10 2.95 vs 3.1, O9 2.07 vs 2.1, O14 2.35 vs 2.3, L10 0.197 vs 0.20, O13s 4.7 vs 4.9, O1v 473 vs 477); L10 event count reproduces P012.
- Class counts change by at most one unit between smeared, unsmeared and 1.5×-resolution pdfs (H ≤ 0.10).
- The 5 keV true-energy grid sub-samples P057's grid exactly; the same halo and efficiency code are used, so cached and fresh spectra are consistent (checked by the L8s/L20s ratios).
- Not varied: efficiency-edge width (P003: ≤ 9 % on N_lo), halo (elastic shapes: 0.0008 dex, P057), mass (LZ's 400–4000 GeV degeneracy, P057: ≥ 250 events to separate).

## 6. Failed or abandoned

- First compute run omitted the 1/m_v² factor in the coefficient closures (shapes right, absolute rates 3.67 × 10⁹ too large relative to P057's cache); caught by the L8s/O10s and L20s/O6s checks, cache cleared, recomputed (239 s).
- A ZeroDivisionError for pairs of identical (aliased) pdfs in the Gaussian N_3σ; guarded.
- A first version of the L12 identification expected an O6-like shape (v² weighting only); WimPyDD's Φ̃′ response makes L12 a separate class, so the expectation was dropped, not the calculation.
- We could not reproduce LZ's "293 distinguishable spectra" from its stated reductions (40 → 23 entries × 11 masses = 253 leaves 40 for the inelastic set, which has 60–69 physical entries after merging O4^{s,v}); the discrepancy is noted, not resolved.

## 7. Figures

- `figures/P089_dendrogram.png` — average-linkage dendrogram (Hellinger) of the 40 Table S6 entries plus O1^{s,v}, O4^{s,v} at 1 TeV, LZ local Z at 1 TeV in brackets; dashed lines at H = 0.02, 0.05, 0.15.
- `figures/P089_class_spectra.png` — observed-energy pdf of one representative per class (H < 0.05), linear and log scale, 248 keV dotted.
- `figures/P089_Z_vs_Nlo.png` — LZ local Z (1 TeV) versus N_lo for the 40 entries, coloured by dominant response, with the P016 calibration curve; L18v is the outlier.

## 8. Discussion

Every degeneracy LZ used is a single-operator identity, and the reductions derived here reproduce both LZ lists exactly, which we regard as confirmation of the recalled Table 1 structure for L6–L20. The Lagrangian table adds only three effective degeneracies to the operator picture of P057 (L19 ≡ L11/L14 through O14 ≡ O9; L10 s ≡ v; L7 ≡ O7) and, contrary to expectation, several *new* classes (L12 s and v, L16 s and v, L6 s and v, L9 s and v, L13 v, L18 s and v). Table S6 is therefore not 40 measurements but 24 shapes, and its 3.0–3.4σ band is populated by 14 of them, matching the corpus' N_eff ≈ 12–14. The significance pattern is almost entirely the low-energy companion count: coherent M spectra are killed by the 2024 null (0–1.8σ), q²-spin spectra cluster at 3.0–3.2σ, q⁴-spin at 3.1–3.4σ, and isovector couplings score higher wherever an M component is suppressed. The one unexplained entry is L18v.

## 9. References

LZ Collaboration, arXiv:2609.02823 (2026). N. Anand, A. L. Fitzpatrick, W. C. Haxton, Phys. Rev. C 89, 065501 (2014). A. L. Fitzpatrick, W. Haxton, E. Katz, N. Lubbers, Y. Xu, JCAP 02 (2013) 004. I. Jeong, S. Kang, S. Scopel, G. Tomar, Comput. Phys. Commun. 276, 108342 (2022). D. Baxter et al., Eur. Phys. J. C 81, 907 (2021). LZ Collaboration, Phys. Rev. Lett. 133, 221801 (2024). Corpus: P001, P002, P003, P008, P009, P012, P016, P021, P027, P031, P032, P035, P044, P050, P057, P062.

## 10. Tools and provenance (mirrors `output/provenance/P089.json`)

- Agent tools: Read (PAPER_GUIDE; P057.md and P057 details; tex lines 30–80, 225–300, 484–501, 800–911; P003, P012, P027, P031, P032, P044 papers; P057 script; WimPyDD package.py docstring; three own figures), Bash (greps of tex, ledger, lzcommon, P031 details, P044 script, WimPyDD; cache inspection; WimPyDD timing test; three script runs; two result-extraction snippets; data-request listing; version check), Write (script, details, provenance, paper), Edit (four script fixes).
- Software: python 3.12.13; WimPyDD 2.0.4 (`eft_hamiltonian` with closure coefficient functions incl. q-dependence, `diff_rate` via `lz.wd_rate`, `streamed_halo_function` via `lz.wd_halo`); numpy 2.5.3; scipy 1.18.1 (`special.erf`, `stats.norm`, `cluster.hierarchy.linkage/fcluster/dendrogram`); pandas 3.0.5; matplotlib 3.11.2 (Agg); `output/code/common/lzcommon.py` (LZ, LSIG, OSIG, M_V_GEV, M_NUCLEON_GEV, wd, wd_halo, wd_rate).
- Local inputs: `inputs/LZ_arXiv_2609.02823_fulltext.tex` (Theory 35–74; Table I 225–246; Results 253–295; LEE 485–501; Tables S6/S7 820–909); `output/provenance/PAPER_GUIDE.md`; `output/papers/P003, P012, P027, P031, P032, P044, P057.md`; `output/work/P057/details.md`, `output/work/P057/cache/m1000_*.npy` (16 spectra sub-sampled); `output/work/P016/P016_Z_vs_Nlo_curves.csv`; `output/work/P003/P003_spectra.npz` (inspected only); `output/code/P057_operator_degeneracy.py`, `P044_anapole_edm.py` (conventions); `WimPyDD/package.py` (read-only); `output/results_ledger.csv` (P001/P008/P027 rows); `output/data_requests/DR-001..003.md`.
- Recalled knowledge: structure and ordering of Anand's Table 1 (likely); NR reductions of Dirac bilinears (certain); Anand response coefficient functions (certain); m_M = m_N (likely); L10 = 4[(q²/m_N²)O4 − O6] (likely, corpus-confirmed); the O16 = (q²/m_N²)O12 + O15 identity (derived here, certain); Hellinger/KL definitions and Gaussian LLR asymptotics (certain); (N−Z)²/A² = 0.03 for xenon (certain).
- Datasets: none. Data requests: none filed (L18 spectrum would be covered by an extension of DR-001/DR-002).
- WimPyDD-generated files: none (`diff_rate` writes no response-function files; checked by modification times under `WimPyDD/WimPyC/Response_functions/`).
