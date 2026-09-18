# P012 — research record

**Title.** Magnetic-dipole dark matter at 1 TeV: the coupling from the LZ fit and the low-energy shoulder
**Simulated arXiv date.** 2026-09-04 · hep-ph · Category EFT · author profile: dark-matter EFT phenomenologists. Competes with P044 (later).
**Script.** `output/code/P012_magnetic_dipole.py` (run from the simulation root with `.venv/bin/python`; 85–120 s). Log: `output/work/P012/P012_run.log`. All numbers below are from `P012_summary.json` and the CSV tables in `output/work/P012/` unless marked [paper] (quoted from arXiv:2609.02823), [P0XX] (corpus) or [recall] (training knowledge with reliability flag).

---

## 1. Motivation and framework

LZ [paper, Theory paragraph] describes L10^s as "a magnetic moment interaction" and fits the 248 keV event with 1.0 (+1.4, −0.7) L10^s events at 1000 GeV [paper, Table I]; Table S6 gives L10^s local significances 2.9/3.1/3.4/3.4/3.4σ at 100/200/400/1000/4000 GeV (0.0σ for m ≤ 50 GeV) [paper]. Fig. 6 (bottom) shows the two-sided 90% interval on the dimensionless coupling d10^s versus mass; the paper says that for L1–L20 "the quantity constrained is the interaction coupling parameter d_j" [paper, Results]. Fig. 1's caption fixes the unit coupling as d_i^s = 1/m_v² "as in Eq. 68 of Anand et al.", m_v = 246.2 GeV [paper].

Questions: (1) what d10^s gives 1.0 event in 2.84 t·yr for m_χ = 100, 200, 400, 1000, 4000 GeV, and does our normalisation reproduce LZ's Fig. 6 interval; (2) what magnetic moment does the coupling correspond to; (3) does the low-energy shoulder of the double-peaked L10 spectrum survive the 2024 3–80 phd search; (4) how does the contact L10 differ from photon-mediated (long-range) magnetic-dipole DM.

Lagrangian (Anand et al. 2014 Table 1, entry 10, as recalled — **likely**; P003 R6, P005):
L10 = (d10/m_v²) (χ̄ iσ^{μν} q_ν/m_M χ)(N̄ iσ_{μρ} q^ρ/m_M N), with m_M a normalisation mass that we set to the nucleon mass m_N = 0.938272 GeV (stated assumption; Anand's m_M is the nucleon mass to our recollection — likely).

## 2. Inputs

| Input | Source |
|---|---|
| Best fit L10^s 1000 GeV: 1.0 +1.4 −0.7 events | [paper] Table I, line 238 of the tex; `lz.LZ['L10s_1000_bestfit']` |
| Exposure 2.84 t·yr; efficiency 0.96 plateau, 50% at 5.4 and 269.9 keV | [paper]; `lz.LZ` |
| Fig. 6 bottom (vector PDF) | `inputs/arXiv_2609.02823_source/Fig6_O1_L10_limit_stacked.pdf`, PyMuPDF `get_drawings`, `get_text` |
| Fig. 1 L10^s digitised curves (200, 1000 GeV) | `output/work/P003/fig1_digitised_bottom.csv` [P003] |
| WimPyDD convention c^0 = c_p + c_n = 2 × Anand c^0 | `lz.wd_c_from_anand` (settled by P003, confirmed by P007) |
| Halo | `lz.wd_halo()`: WimPyDD Sun-frame SHM, v0 = 238, v_esc = 544 km/s, v_sun,pec = (11.1, 12.2, 7.3) km/s (v_max = 794.6 km/s); the configuration that reproduces Fig. 1 to 1–3% [P003] |
| Efficiency model | P003 form: ε = 0.96 · ½[1+erf((E−5.4)/(√2·3.4))] · ½[1−erf((E−269.9)/(√2·8))]; variants σ_hi = 11.5 keV (P007's Fig. S2 fit) and hard cuts |
| 2024 low-energy search: ROI 5.4–55 keV; exposure 4.2 t·yr (5.5 t × 280 live days, SR1 + WS2024) | ROI [paper, Data Analysis]; exposure [recall, certain]; tolerated signal ≲ 3–5 events [recall, uncertain; P003 R4] |
| Nucleon g-factors g_p = 5.5857, g_n = −3.8261; α = 1/137.036; e = √(4πα) = 0.3028 | [recall, certain] |
| Nuclear magneton μ_N = e/(2m_p) = 0.16137 GeV⁻¹ (natural units) = ħc/(2 m_p c²) · e = 0.10515 fm · e = 1.0515 × 10⁻¹⁴ e·cm | [recall, certain]; computed in the script from `lz.HBARC_GEV_FM` and `lz.M_NUCLEON_GEV` |
| Single-count PLR reference: q(μ; n) = 2[μ − n + n ln(n/μ)], two-sided 90% threshold 2.706 (asymptotic) or exact Poisson p-value | [recall, certain] (Cowan et al. 2011); computed in Part G |
| P009: the event is on the efficiency plateau (ε = 0.93 at E_ML = 246 keV) | [P009] — justifies treating the ROI count as insensitive to the roll-off width |

## 3. Derivations

### 3.1 NR reduction of L10 (contact dipole–dipole)
For Dirac spinors normalised to ū u = 2m, ū′σ^{ij}u → 4m ε^{ijk}S^k (S = σ/2), while the σ^{0i} components are O(q/m) smaller. Dividing by 4 m_χ m_N (Anand's normalisation of NR operators):
(χ̄ iσ^{μν}q_ν χ)(N̄ iσ_{μρ}q^ρ N) → −4 (q × S_χ)·(q × S_N) = −4[q² S_χ·S_N − (q·S_χ)(q·S_N)] = −4 m_N² [(q²/m_N²) O4 − O6],
so L10 → (d10/m_v²)·4(m_N²/m_M²)[(q²/m_N²)O4 − O6] up to an overall sign (irrelevant for the rate). With m_M = m_N and A = d10/m_v²: c4(q) = 4A q²/m_N², c6 = −4A (Anand isoscalar c^0). The squared amplitude is ∝ A² (q⁴/m_N⁴)[Σ′ + Σ″ − 2Σ″ + Σ″] = A²(q⁴/m_N⁴)Σ′: the longitudinal spin response cancels identically; L10 is a pure transverse-spin q⁴Σ′ response [P003 §8; P005 §3]. WimPyDD implementation: `eft_hamiltonian({(4,'q2'): lambda q: [8 q²/(m_v² m_N²), 0], 6: lambda: [−8/m_v², 0]})`, i.e. Anand coefficients × 2 (`lz.wd_c_from_anand`). Rates scale as d10².

**Residual normalisation.** P003 found the digitised Fig. 1 L10^s curve to be 247.5× the WimPyDD rate for c4 = q²/m_N², c6 = −1 at A = 1/m_v² (c^0_WD = 1/m_v²). Our normalisation is 8² = 64× that rate, so we expect Fig. 1/ours = 247.5/64 = 3.87. We recompute it directly (Sec. 4.2): 3.859 (200 GeV) and 3.857 (1000 GeV). The factor is mass-independent to 0.1%, so it is a pure convention (a factor 1.964 ≈ 2 in the amplitude), not a nuclear-physics or halo effect. Candidates: LZ's d_j multiplies c_p and c_n each with an extra factor 2 relative to Anand's isoscalar definition; m_M ≠ m_N in Anand's Eq. 68 (m_M = m_N/√2 would do it); or an extra factor 2 in the tensor-current normalisation of DMFormFactor. We cannot decide this from the paper (DR-001). Throughout we call "LZ normalisation" the Fig. 1 one (N_unit,LZ = 3.857 × N_unit,ours) and "Anand×WimPyDD normalisation" ours; d10 values differ by √3.857 = 1.964.

### 3.2 Photon-mediated magnetic-dipole DM (long range)
L = (μ_χ/2) χ̄σ^{μν}χ F_μν. The dipole vertex is μ_χ ū′σ^{μν}q_ν u; photon propagator 1/q²; nucleon current e[F₁γ^μ + F₂ iσ^{μν}q_ν/(2m_N)]. In the NR limit (same spinor algebra as above; ū′σ^{0i}q_i u → −i q² − 4 S·(q × P) with P = p + q/2, and the nucleon convection + magnetisation currents ū′γ^i u → 2P^i + i g_N (q × S_N)^i), combining the time-component (q²) piece with the charge, the spin–velocity pieces into the Galilean-invariant v⊥, and the spatial dipole current with the nucleon magnetisation current, one obtains per nucleon (Anand normalisation):
c1^N = e μ_χ Q_N/(2m_χ),  c5^N = 2 e μ_χ m_N Q_N/q²,  c4^N = e μ_χ g_N/m_N,  c6^N = −e μ_χ g_N m_N/q²,
with Q_p = 1, Q_n = 0. This agrees with the standard result (Fitzpatrick et al. 2012, Eq. 45; Gresham & Zurek 2014; Banks–Fox–Weiner 2010; Barger–Keung–Marfatia 2011) as recalled — **likely**. Two remarks: (i) c4 + (q²/m_N²)c6 = 0, so the dipole–dipole term is again purely transverse — it is L10's structure with 1/m_M² replaced by the photon propagator 1/q²: e g_N μ_χ (m_N/q²)[(q²/m_N²)O4 − O6] versus 4 d10 m_N²/(m_v² m_M²) [(q²/m_N²)O4 − O6]; (ii) the c5 term gives the (q²/m_N²) v⊥² M response ∝ v⊥²/q² ∝ (1/E_R)[1 − E_R/(2 m_A v²) …], the familiar Z²/E_R charge–dipole enhancement (in the heavy-DM limit; the 1/m_χ corrections are included through c1 and v⊥). Isospin for WimPyDD: charge terms (c_p + c_n, c_p − c_n) = (c_p, c_p); spin terms g_p + g_n = 1.7596, g_p − g_n = 9.4118. WimPyDD's reserved arguments `q` (GeV) and `mchi` (GeV) supply the momentum and mass dependence; the O4–O5 (Δ–Σ′) interference is included by WimPyDD. Computed at μ_χ = 1 μ_N; rates scale as μ_χ².

### 3.3 Matching contact and long-range dipole–dipole amplitudes
Equating the two transverse-spin coefficients at a momentum transfer q: μ_χ,eff(q) = 4 d10 m_N q²/(e g_N m_v² m_M²) → with m_M = m_N and the isoscalar g^0 = (g_p + g_n)/2 = 0.880: μ_eff = 4 d10 q²/(e g^0 m_N m_v²). This is an illustrative single-q matching of the spin–spin piece only (the photon model additionally has the charge–dipole piece).
Heavy-mediator reading: a vector V of mass M with dipole couplings (μ_χ^V/2)χ̄σ^{μν}χV_μν and (μ_N^V/2)N̄σ^{μν}N V_μν gives, after integrating out V, μ_χ^V μ_N^V (χ̄σ^{μν}q_μχ)(N̄σ_{ρν}q^ρN)/M², i.e. d10/(m_v² m_M²) = μ_χ^V μ_N^V/M² (the ½'s cancel against the two terms of F_μν; sign aside — **likely**).

### 3.4 Single-count two-sided PLR reference
For n observed events and negligible background, q(μ; n) = 2[μ − n + n ln(n/μ)]. Asymptotic 90% (q ≤ 2.706): n = 1 → [0.1057, 3.647]; expected (n = 0) upper limit 1.353. Exact Poisson p-values (Part G): n = 1 → [0.1054, 4.357]; expected n = 0 upper limit 2.436 (the Feldman–Cousins n = 1, b = 0 interval is [0.11, 4.36] — recall, certain). These are what an interval on the *number of events* would look like if the fit were a pure count.

## 4. Results

### 4.1 Fig. 6 bottom digitised (`fig6_bottom_digitised.csv`)
Curves: drawing 349 (black, width 3, 13 vertices) = upper edge; 350 (black, width 3, from 100 GeV with a vertical drop to the frame) = lower edge; 346 (black, width 1.5, dashed) = median sensitivity; 347 = LZ 2024b; 348 = PandaX-II 2019. x calibration: the 13 vertices of the upper curve are the 13 Table S6 masses (10 … 4000 GeV); a straight-line fit of log10 m vs x has rms 2 × 10⁻⁵ dex (158.80 pt/decade); the frame spans 9.0–4000 GeV and the lower curve starts at 99.9 GeV. y calibration: decade tick marks on the left spine at y = 355.46, 448.04, 540.61, 633.18 pt (10², 10¹, 10⁰, 10⁻¹; 92.573 pt/decade); the frame spans d10 = 0.089–111. (Mantissa-label centres sit 3.2 pt above the ticks; we use the ticks.)

| m (GeV) | lower | median sensitivity | upper |
|---|---|---|---|
| 100 | 0.154 | 0.455 | 0.632 |
| 200 | 0.112 | 0.299 | 0.429 |
| 400 | 0.125 | 0.309 | 0.450 |
| 1000 | 0.171 | 0.417 | 0.614 |
| 4000 | 0.322 | 0.789 | 1.144 |

Digitisation precision ≈ 0.5 pt → 1.2% in d10, 2.5% in N.

### 4.2 L10 shape and normalisation versus Fig. 1 (`P012_L10_vs_fig1.csv`)
Geometric-mean ratio Fig. 1/ours over 10–260 keV: 3.859 (200 GeV), 3.857 (1000 GeV); rms log10 residual 0.084 dex, max 0.20 dex (both masses). Extrema (Fig. 1 / ours): 1000 GeV dip 59 / 53 keV, high peak 203 / 197 keV; 200 GeV dip 63 / 53 keV, high peak 175 / 173 keV; our low peak is at 21 keV (200 GeV) (the "peak1 = 119" entries in the CSV are an artefact of the < 120 keV search window when the curve at 119 keV exceeds the low peak). N_lo per 200–270 keV event: 0.155 (Fig. 1) vs 0.199 (ours) at 1000 GeV; 0.468 vs 0.603 at 200 GeV — identical to P003 (0.155/0.198). Events in 2.84 t·yr at unit coupling from the digitised curve itself: 13.50 (1000 GeV), 28.7 (200 GeV) [P005: 13.4]; ours × 3.857 gives 12.87 and 28.66 — consistent to 5%. Fraction of the ROI rate below 55 keV: 0.062 (Fig. 1) vs 0.070 (ours).

### 4.3 Couplings (`P012_coupling_table.csv`, P003 efficiency; three variants change every entry by < 0.3% for m ≥ 200 GeV)

| m (GeV) | N(d10 = 1), ours | N(d10 = 1), LZ norm | d10(N = 1) LZ | d10(N = 0.3–2.4) LZ | d10(N = 0.105–3.65) LZ | d10(N = 1) ours |
|---|---|---|---|---|---|---|
| 100 | 4.567 | 17.61 | 0.238 | 0.131–0.369 | 0.077–0.455 | 0.468 |
| 200 | 7.430 | 28.66 | 0.187 | 0.102–0.289 | 0.061–0.357 | 0.367 |
| 400 | 6.343 | 24.46 | 0.202 | 0.111–0.313 | 0.066–0.386 | 0.397 |
| 1000 | 3.336 | 12.87 | 0.279 | 0.153–0.432 | 0.090–0.533 | 0.548 |
| 4000 | 0.942 | 3.633 | 0.525 | 0.287–0.813 | 0.170–1.002 | 1.030 |

Comparison with Fig. 6: expected events at the digitised edges.

| m (GeV) | N at lower edge, LZ norm | N at upper edge, LZ norm | N at lower, ours | N at upper, ours |
|---|---|---|---|---|
| 100 | 0.420 | 7.03 | 0.109 | 1.82 |
| 200 | 0.361 | 5.27 | 0.094 | 1.37 |
| 400 | 0.380 | 4.95 | 0.099 | 1.28 |
| 1000 | 0.377 | 4.85 | 0.098 | 1.26 |
| 4000 | 0.376 | 4.76 | 0.097 | 1.23 |

Median sensitivity at 1000 GeV: 2.23 events (LZ norm), 0.58 (ours). For m ≥ 200 GeV the implied counts at the edges are mass-independent to ±3% (lower) and ±5% (upper): the Fig. 6 axis is linear in d10 (N ∝ d10²) and our spectra have the right mass dependence. Which normalisation is LZ's? A two-sided 90% interval from one background-free count must have an expected (n = 0) upper limit of 1.35 (asymptotic) to 2.44 (exact) events and an observed n = 1 upper edge of 3.65–4.36 (Sec. 3.4). In the LZ (Fig. 1) normalisation the median sensitivity (2.23) and upper edge (4.85) match these to 9–11%; in ours they are 0.58 and 1.26, impossible for any interval construction (and the 68% interval 0.29–0.83 of Table I would then protrude above the 90% upper edge 0.61). **Fig. 6 therefore confirms the Fig. 1 normalisation; the factor 3.86 lies between the recalled Anand reduction and LZ's definition of d10.** The lower edge in the LZ normalisation corresponds to 0.38 events instead of 0.105: LZ's likelihood is more constraining at small μ than a pure count (the extended 2D likelihood with the L10 low- and intermediate-energy components and nuisance parameters; possibly a toy-calibrated statistic); we cannot reproduce this without the Data Release (DR-001). The Table I 68% interval maps to d10 = 0.153–0.432, whose lower end sits 10% below the 90% lower edge (0.171) — a 25% inconsistency in N, within what different interval constructions give.

**Headline (LZ normalisation, 1000 GeV): d10^s = 0.279 (0.153–0.432 for 0.3–2.4 events); 0.187 (200 GeV), 0.202 (400 GeV), 0.238 (100 GeV), 0.525 (4000 GeV). In the Anand×WimPyDD normalisation multiply by 1.964.** Provisional pending DR-001 (the factor 2).

### 4.4 Scale of the coupling and magnetic-moment conversions (`P012_summary.json` → conversions)
- d10/m_v² = 4.60 × 10⁻⁶ GeV⁻² = (466 GeV)⁻² (dimension-6 reading with Anand's m_M = m_N normalisation).
- Dipole–dipole contact strength G_dd ≡ d10/(m_v² m_M²) = 5.22 × 10⁻⁶ GeV⁻⁴ = (20.9 GeV)⁻⁴ (dimension-8 reading). The momentum transfer of the event is q = √(2 m_A E_R) = 0.246 GeV (A = 131.3, E_R = 248 keV), so q ≪ 21 GeV and the contact treatment is self-consistent at the detector, but any UV completion has μ_χ^V μ_N^V = M²/(20.9 GeV)⁴: for M = 1 TeV, √(μ_χ^V μ_N^V) = 2.29 GeV⁻¹ = 14 nuclear magnetons, i.e. μ_χ^V = 200 μ_N if the nucleon coupling to V is one magneton (Sec. 3.3). Such large dipole couplings are typical of a composite or strongly coupled dark sector, not of a loop-induced dipole.
- Photon-equivalent moment at q_event (spin–spin piece only): μ_eff = 4.47 × 10⁻⁶ GeV⁻¹ = 2.77 × 10⁻⁵ μ_N = 2.91 × 10⁻¹⁹ e·cm. Cross-check: the photon-mediated model needs μ_χ = 2.10 × 10⁻⁵ μ_N = 2.21 × 10⁻¹⁹ e·cm for one event in 200–270 keV (Sec. 4.6), where 89% of its high-energy rate is the spin–spin piece — the two agree to 30%, the difference being the 1/q² weighting across 200–270 keV.
- Conversion used (recall, certain): 1 μ_N = e/(2m_p) = 0.16137 GeV⁻¹ in Heaviside–Lorentz natural units (e = 0.30282) = 1.0515 × 10⁻¹⁴ e·cm. All moment numbers depend on Anand's normalisation choices for L10 (Sec. 3.1) and inherit the factor-2 ambiguity — **flagged likely**.

### 4.5 Low-energy shoulder (`P012_low_energy_shoulder.csv`)
Fraction f_lo of the efficiency-weighted ROI rate in 5.4–55 keV and derived counts (P003 efficiency; 2024 exposure 4.2 t·yr, ratio 1.479 to 2.84 t·yr; hard cut at 55 keV):

| m (GeV) | f_lo | N_lo per 200–270 keV event | N_mid (55–200) per hi event | shoulder events in 2024 at N = 1.0 | at N = 2.4 | at N = 3.65 | at Fig. 6 upper edge | LZ ROI events allowed for tolerance 3 / 5 / 10 | d10 for tolerance 5 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 0.371 | 6.08 | 9.27 | 0.548 | 1.32 | 2.00 | 3.86 | 5.5 / 9.1 / 18.2 | 0.72 |
| 200 | 0.140 | 0.603 | 2.67 | 0.207 | 0.498 | 0.757 | 1.09 | 14.5 / 24.1 / 48.2 | 0.92 |
| 400 | 0.0888 | 0.283 | 1.88 | 0.131 | 0.315 | 0.479 | 0.65 | 22.8 / 38.1 / 76.1 | 1.25 |
| 1000 | 0.0702 | 0.199 | 1.61 | 0.104 | 0.249 | 0.379 | 0.50 | 28.9 / 48.2 / 96.3 | 1.93 |
| 4000 | 0.0633 | 0.171 | 1.51 | 0.0935 | 0.225 | 0.341 | 0.44 | 32.1 / 53.4 / 107 | 3.84 |

With the digitised Fig. 1 shape at 1000 GeV: f_lo = 0.062, N_lo = 0.155, N_mid = 1.30, shoulder 0.092 events at best fit (12% below ours). Efficiency variants (σ_hi = 11.5 keV, hard cuts) change the 1000 GeV shoulder by ≤ 1.5%. Caveat: the 2024 efficiency curve below 55 keV differed in detail (S2 threshold), a 10–20% effect on these small numbers [P003 §4].

Conclusion: for m ≥ 200 GeV the shoulder is 0.09–0.21 events at best fit, ≤ 0.5 events at the 68% upper edge and ≤ 1.1 events at the 90% upper edge of Fig. 6 — far below any plausible tolerance (3, 5 or 10 events); the 2024 null result would only bite at ≥ 29 (48, 96) LZ ROI events at 1 TeV. At 100 GeV the shoulder is 37% of the rate: 0.55 events at best fit but 3.9 events at the Fig. 6 upper edge, i.e. at 100 GeV the 2024 search is comparable to LZ's own upper edge (d10 < 0.72 for a tolerance of 5 versus 0.63 from Fig. 6). This matches Table S6, where L10's significance drops from 3.4σ (≥ 400 GeV) to 2.9σ at 100 GeV. The 55–200 keV range carries 1.5–1.9 events per 200–270 keV event at m ≥ 400 GeV; with 1.0 fitted ROI events the total expected in 55–270 keV is ≈ 0.93, so nothing else is predicted in the extended ROI at the best fit.

### 4.6 Photon-mediated dipole (`P012_photon_dipole.csv`, `photon_decomposition_1000GeV`)

| m (GeV) | N_lo per hi event | N_mid per hi | f_lo | f_hi | μ_χ for 1 ROI event (μ_N / e·cm) | μ_χ for 1 event in 200–270 keV (μ_N / e·cm) | 2024 shoulder events if 1 hi event |
|---|---|---|---|---|---|---|---|
| 100 | 12 900 | 125 | 0.698 | 5.4 × 10⁻⁵ | 3.1 × 10⁻⁷ / 3.2 × 10⁻²¹ | 4.2 × 10⁻⁵ / 4.4 × 10⁻¹⁹ | 19 100 |
| 200 | 1 180 | 20.4 | 0.707 | 6.0 × 10⁻⁴ | 4.2 × 10⁻⁷ / 4.4 × 10⁻²¹ | 1.7 × 10⁻⁵ / 1.8 × 10⁻¹⁹ | 1 750 |
| 400 | 541 | 11.9 | 0.710 | 1.3 × 10⁻³ | 5.8 × 10⁻⁷ / 6.1 × 10⁻²¹ | 1.6 × 10⁻⁵ / 1.7 × 10⁻¹⁹ | 801 |
| 1000 | 376 | 9.34 | 0.712 | 1.9 × 10⁻³ | 9.2 × 10⁻⁷ / 9.6 × 10⁻²¹ | 2.1 × 10⁻⁵ / 2.2 × 10⁻¹⁹ | 556 |
| 4000 | 322 | 8.45 | 0.712 | 2.2 × 10⁻³ | 1.8 × 10⁻⁶ / 1.9 × 10⁻²⁰ | 3.9 × 10⁻⁵ / 4.1 × 10⁻¹⁹ | 475 |

Decomposition at 1000 GeV: the charge–dipole part (O1 + O5, Z²/E_R) has N_lo = 18 000 and carries 89% of the 5.4–55 keV rate; the dipole–dipole part ((O4 − O6)/q², a Σ′ response with no q⁴ weighting) has N_lo = 31.9 (close to O4's 28 [P003]) and carries 89% of the 200–270 keV rate (90% of dR/dE at 248 keV); the Δ–Σ′ interference adds 9% to R_hi. A photon-coupled dipole producing one 200–270 keV event therefore predicts 556 events in the 2024 ROI; a tolerance of 5 events limits the high-energy count to 0.0090 events, i.e. μ_χ ≤ 2.0 × 10⁻⁶ μ_N = 2.1 × 10⁻²⁰ e·cm at 1 TeV. Photon-mediated magnetic-dipole DM is excluded as the explanation by two orders of magnitude in rate at every mass considered. (Recalled photon-dipole limits from XENON1T/PandaX are of order 10⁻²⁰ e·cm at these masses — uncertain; not used.)

Physically, LZ's L10 is a *contact* dipole–dipole interaction: the same transverse-spin structure as the photon-mediated dipole–dipole term with the propagator 1/q² replaced by 1/m_M², which removes the 1/E_R enhancement and, because no charge–dipole term accompanies it, the Z² coherent low-energy population. The double peak is the Σ′ form-factor node of ¹²⁹Xe/¹³¹Xe weighted by q⁴ [P003]. A UV completion needs a heavy mediator with tensor (dipole-like) couplings to both the DM and the nucleon (quark tensor charges δu ≈ 0.8, δd ≈ −0.2 — recall, likely), with the product of couplings over M² fixed to (20.9 GeV)⁻⁴ (Sec. 4.4).

### 4.7 Sanity checks
- Shape vs Fig. 1: rms 0.084 dex, dip 53 vs 59 keV, high peak 197 vs 203 keV, N_lo 0.199 vs 0.155 — identical to P003 (0.086 dex, 52/61 keV, 196/205 keV, 0.198/0.155) with the finer 2 keV grid.
- Absolute scale: 3.857 vs P003's 247.5/64 = 3.87 (0.3%); mass-independent (200 vs 1000 GeV: 0.05%).
- Implied counts at the Fig. 6 edges mass-independent to ±3–5% for m ≥ 200 GeV (Sec. 4.3).
- Photon dipole–dipole part reproduces the O4-like N_lo (32 vs 28); charge–dipole part steeper than O1 (18 000 vs 2 752) as expected from the extra 1/E_R.
- Photon-equivalent moment from the amplitude matching (2.8 × 10⁻⁵ μ_N) agrees with the count-based μ for one high-energy event (2.1 × 10⁻⁵ μ_N) to 30%.
- No WimPyDD response-function files were written (`find WimPyDD -newer script` empty).

## 5. Figures
- `figures/P012_L10_spectra_bestfit.png` — L10 spectra at 100–4000 GeV normalised to 1.0 LZ ROI event (LZ normalisation), with the digitised Fig. 1 1000 GeV curve on the same scale; bands: 2024 ROI (red), 200–270 keV (green), < 5.4 and > 269.9 keV (grey); dotted line at 248 keV.
- `figures/P012_d10_vs_mass.png` — d10 for N = 1.0 in both normalisations, the N = 0.3–2.4 band and the N = 0.105/3.65 lines (LZ normalisation), versus the digitised Fig. 6 interval and median sensitivity. The Anand×WimPyDD N = 1 curve lies on LZ's upper edge; the LZ-normalised best fit lies inside the interval at every mass.
- `figures/P012_contact_vs_photon_dipole.png` — 1000 GeV: L10 versus the photon-mediated dipole (full, charge–dipole and dipole–dipole parts), each normalised to one event in 200–270 keV.

## 6. Failed or abandoned approaches
- Piping the first run through `tee` before the script had created `output/work/P012/` lost the log; re-run with redirection (no change in numbers).
- The peak finder's "< 120 keV" window reports 119 keV as the low "peak" when the curve at 119 keV exceeds the true low peak (Fig. 1 curves, and ours at 1000 GeV); the dip and high-peak positions are unaffected and we report those.
- Using label-centre positions for the Fig. 6 y calibration would bias d10 by 3.2 pt = 0.035 dex (8%); the decade tick marks were used instead.

## 7. Extended discussion
1. The coupling that LZ's best fit requires is d10^s ≈ 0.19–0.28 for 200–1000 GeV, i.e. d10/m_v² ≈ (470–570 GeV)⁻² in Anand's normalisation — "weak-scale" as a dimension-6 coefficient, but the underlying dipole–dipole operator is dimension 8 with Λ = (m_v² m_M²/d10)^{1/4} ≈ 21 GeV. Any concrete model must supply large tensor couplings (Sec. 4.4); this is the model-building cost of a spectrum with no low-energy population.
2. The normalisation convention matters at the factor-2 level for d10 and hence for any recast (e.g. collider or relic-density comparisons); we recommend that follow-up work quote d10 with an explicit statement of the reduction and of the WimPyDD isospin convention, and that the Data Release normalisation be checked (DR-001). The low-energy-shoulder and photon-dipole conclusions are normalisation-independent.
3. The Fig. 6 lower edge corresponds to 0.38 expected events (LZ normalisation), 3.6× the 0.105 of a pure count. If the extended likelihood tightens the lower edge, the Table I 68% interval (0.3–2.4) is consistent; the effective information beyond a single count would be worth understanding (a STAT paper could compare with P001's single-count treatment).
4. The 100 GeV case is instructive: the 2024 search and the extended search have comparable reach in d10 (0.72 vs 0.63), and the L10 spectrum there falls steeply through 200–270 keV, so the 248 keV event is on the tail (the Fig. 6 upper edge corresponds to 7.0 events at 100 GeV versus 4.8 at ≥ 400 GeV).
5. Relation to P005: our N(d10 = 1) = 12.87 (1000 GeV, LZ normalisation) agrees with P005's 13.4 from the digitised Fig. 1 to 4%; P005's XENONnT/PandaX-4T expectations are therefore unchanged. Relation to P007: our Fig. 6 bottom digitisation uses the tick marks, whereas P007 used label centres for the top panel; if the same 3 pt offset applies there, P007's c1 values would be ≈ 0.09 dex low — worth checking (not done here).

## 8. References
1. LZ Collaboration, arXiv:2609.02823 (2026).
2. N. Anand, A. L. Fitzpatrick, W. C. Haxton, Phys. Rev. C 89, 065501 (2014), arXiv:1308.6288.
3. A. L. Fitzpatrick, W. Haxton, E. Katz, N. Lubbers, Y. Xu, JCAP 02 (2013) 004, arXiv:1203.3542.
4. T. Banks, J.-F. Fortin, S. Thomas, arXiv:1007.5515 (2010); T. Banks, J. F. Fox, N. Weiner, Phys. Rev. D 82, 075004 (2010), arXiv:1005.5651 (dipole DM direct detection).
5. V. Barger, W.-Y. Keung, D. Marfatia, Phys. Lett. B 696, 74 (2011), arXiv:1007.4345.
6. M. I. Gresham, K. M. Zurek, Phys. Rev. D 89, 123521 (2014), arXiv:1401.3739.
7. S. Kang, S. Scopel, G. Tomar, J.-H. Yoon (WimPyDD), Comput. Phys. Commun. 276, 108342 (2022), arXiv:2106.06207.
8. G. Cowan, K. Cranmer, E. Gross, O. Vitells, Eur. Phys. J. C 71, 1554 (2011).
9. LZ Collaboration, Phys. Rev. Lett. 135, 011802 (2025), arXiv:2410.17036 (WS2024 result, 4.2 t·yr — exposure recalled).
Corpus: P003 (L10 reduction, Fig. 1 digitisation, N_lo), P005 (L10 normalised to 1 event), P007 (Fig. 6 top digitisation method), P009 (event on the efficiency plateau), dossier 00.

## 9. Tools and provenance
Mirrors `output/provenance/P012.json`.
- Agent tools: Read ×19 (PAPER_GUIDE.md; 00_evidence_dossier.md; results_ledger.csv; P003.md; P005.md; P009.md; work/P003/details.md; code/P003_nreft_shapes.py; code/P007_higgsino_inelastic.py; common/lzcommon.py; Fig6 PNG; fulltext.tex lines 50–80, 208–298, 825–855; PROMPT.md §4; three P012 figures), Bash ×19 (tex grep; directory/version listing; PyMuPDF text spans, drawings, vertices of Fig. 6; P003 CSV inspection; WimPyDD docstring; PROMPT/process_log greps; three script runs; CSV inspection; four word counts + JSON validation), Write ×5 (script, DR-001.md, details.md, P012.json, P012.md), Edit ×9 (script Part G ×2; paper trims ×5; JSON/details tool counts ×2).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (special.erf, integrate.trapezoid, optimize.brentq, stats.poisson); pandas 3.0.5; matplotlib 3.11.2 (Agg); pymupdf 1.28.2 (open, get_drawings, get_text); WimPyDD 2.0.4 (eft_hamiltonian with q- and mchi-dependent Wilson coefficients, diff_rate via lzcommon.wd_rate, streamed_halo_function via lzcommon.wd_halo); common/lzcommon.py (LZ, M_V_GEV, M_NUCLEON_GEV, HBARC_GEV_FM, A_XE_MEAN, m_nucleus_gev, wd, wd_halo, wd_rate, wd_c_from_anand).
- WimPyDD-generated files: none.
- Datasets: none. Data requests: DR-001 (pending) — affects the factor-2 normalisation of d10 only.
- Recalled knowledge: 12 items (see JSON).
