# P013 — Neutron origins of a 248 keV single scatter: spectra, multiplicity and what the 2.84 t·yr already exclude

Research record (simulated date 2026-09-05). Script: `output/code/P013_neutron_origins.py` (runs in ≈20 s from the simulation
root with `.venv/bin/python`, seeded `numpy.random.default_rng(20260905)`; two consecutive runs give identical numbers).
Machine-readable results: `output/work/P013/P013_results.json`; tables `kinematics_fractions.csv`, `alpha_n_endpoints.csv`,
`first_scatter_fractions.csv`, `ss_spectrum_fractions.csv`, `fission_silent_factor.csv`, `expected_counts.csv`; console log
`run_log.txt`; figures in `figures/`. LZ inputs are from arXiv:2609.02823 (via `lzcommon.LZ` or quoted by line). Every number
that is not in the LZ paper and not derived here is marked **[recalled: reliability]**.

## 1. Motivation and framework

LZ's Discussion (tex l. 321–327) argues that neutrons are the "most likely cause of NR interactions" but that a neutron "must
have at least 8 MeV in order to impart 250 keV", that elastic scattering "becomes highly forward peaked as the neutron energy
increases", and that "it is very unlikely to observe a single recoil at this energy without also observing several more events
at lower energies". The supplement (l. 787–798) adds: spontaneous fission is "up to two orders of magnitude lower than (α,n)"
with high multiplicity; a multiple-scatter cross-check predicts 0.02 ± 0.02 SS neutrons in the WS ROI; 2.1 × 10⁹ simulated
muons (1200 yr) gave no signal-like deposit; rock neutrons of 50 MeV–1 GeV are tagged with 80.0 ± 4.0 %; the 90 % UL on
muon-induced SS of any energy is 4.6 × 10⁻⁴. Table I gives the fitted detector-NR count in the science sample as [0, 0.118]
(constrained only by the veto samples through λ_DN = 0.87 ± 0.02 and λ_PN = 0.05 ± 0.01, Tables S1–S2; total (α,n) tagging
92 ± 4 %, l. 178).

We quantify each of these statements: (i) which sources can reach the kinematic threshold; (ii) what recoil spectrum a
neutron of given energy produces on xenon; (iii) how often a neutron gives *exactly one* deposit in the active volume; (iv)
how many low-energy single scatters accompany one 200–270 keV single scatter for each source; (v) the expected 200–270 keV
count implied by LZ's own normalisations; (vi) whether inelastic (n,n′γ) hybrids could mimic the event's position.

## 2. Elastic kinematics (script §1)

For a neutron of kinetic energy E_n on a nucleus of mass M, E_R = q²/(2M) with q the momentum transfer, q ≤ 2p_cm, so

E_R,max = 4 m_n M/(m_n + M)² · E_n ≡ κ E_n.

Masses from `periodictable` (natural mixture 131.293 u; isotopic masses for the table). Results:

| isotope | abundance | κ | E_n,min(248 keV) [MeV] |
|---|---|---|---|
| ¹²⁴Xe | 0.0009 | 0.03204 | 7.74 |
| ¹²⁸Xe | 0.0191 | 0.03105 | 7.99 |
| ¹²⁹Xe | 0.2644 | 0.03082 | 8.05 |
| ¹³⁰Xe | 0.0408 | 0.03058 | 8.11 |
| ¹³¹Xe | 0.2118 | 0.03035 | 8.17 |
| ¹³²Xe | 0.2689 | 0.03013 | 8.23 |
| ¹³⁴Xe | 0.1044 | 0.02968 | 8.36 |
| ¹³⁶Xe | 0.0887 | 0.02925 | 8.48 |
| natural | — | 0.03026 | **8.20** (200 keV: 6.61; 270 keV: 8.92; 248 ± 46 keV: 7.43–9.71) |

This reproduces the paper's "at least 8 MeV". The momentum transfer for 248 keV is q = 1.248 fm⁻¹, independent of E_n.

**Angular distributions.** Three models for dσ/dE_R on 0 ≤ E_R ≤ E_R,max:
- isotropic in the CM (flat in E_R) — an upper bound on large-angle scattering;
- black disk (sharp-cutoff Fraunhofer diffraction) dσ/dE_R ∝ |2J₁(qR)/(qR)|² with R = 1.25 A^{1/3} = 6.35 fm
  **[recalled: likely; R varied 5.6–7.1 fm]**; because E_R = q²/2M this pattern is *independent of E_n* apart from the cut-off,
  with zeros at E_R = 58, 194, 408, 700 keV. The 200–270 keV window therefore sits just above the second zero, on the third
  diffraction maximum whose height is ≈ 4 × 10⁻³ of the forward peak — consistent with measured n–Pb backward/forward ratios
  of ~10⁻³ at 14 MeV **[recalled: likely]**;
- the exponential form requested, dσ/dΩ ∝ exp(−a(1−cos θ)) with a = (kR)²/2 (Gaussian approximation of the first
  diffraction lobe), which gives dσ/dE_R ∝ exp(−q²R²/4). This form has no secondary maxima and gives f(> 200 keV) ≈ 3 × 10⁻⁶
  at all energies — an underestimate of the large-angle tail by three orders of magnitude; it is reported but not used.

Fractions of elastic recoils (`kinematics_fractions.csv`):

| E_n [MeV] | E_R,max [keV] | kR | f(>200) iso | f(>200) black disk (R 5.6 / 7.1) | f(200–270) iso | f(200–270) black disk (R 5.6 / 7.1) |
|---|---|---|---|---|---|---|
| 8.5 | 257 | 4.05 | 0.223 | 7.2e-3 (4.0e-3 / 1.8e-2) | 0.223 | 7.2e-3 (4.0e-3 / 1.8e-2) |
| 10 | 303 | 4.39 | 0.339 | 1.95e-2 (6.6e-3 / 2.4e-2) | 0.231 | 1.05e-2 (4.2e-3 / 2.1e-2) |
| 15 | 454 | 5.39 | 0.559 | 3.0e-2 (3.2e-2 / 3.5e-2) | 0.154 | 1.04e-2 (4.1e-3 / 2.1e-2) |
| 50 | 1513 | 9.92 | 0.868 | 6.0e-2 (5.8e-2 / 5.8e-2) | 0.046 | 1.01e-2 (4.0e-3 / 2.0e-2) |
| 200 | 6053 | 20.6 | 0.967 | 7.5e-2 (7.7e-2 / 7.2e-2) | 0.012 | 9.9e-3 (3.9e-3 / 2.0e-2) |

So in the diffraction picture about 1 % (0.4–2 % over the R range) of elastic scatters of any neutron above 10 MeV fall in
200–270 keV, versus 1–23 % for isotropic scattering. Figure 1 (`figures/P013_fig1_recoil_spectra.png`) shows the spectra
and the window fractions versus E_n.

## 3. Source spectra (script §2)

### 3.1 (α,n) endpoints
Two-body kinematics for α + A → n + B with E_n,max = ½ m_n (v_n,cm + V_cm)² (ground state, forward emission), Q values
**[recalled: likely, ±0.05 MeV]**, α energies of the chains **[recalled: certain]** (`alpha_n_endpoints.csv`):

| target | Q [MeV] | ²¹⁴Po 7.687 (U) | ²¹²Po 8.785 (Th) | ²¹⁶Po 6.778 (Th) |
|---|---|---|---|---|
| ¹⁹F | −1.951 | 5.26 | **6.31** | 4.39 |
| ¹³C | +2.216 | 9.53 | 10.56 | 8.68 |
| ¹⁷O | +0.587 | 7.90 | 8.94 | 7.04 |
| ¹⁸O | −0.697 | 6.57 | 7.61 | 5.71 |
| ¹⁰B | +1.059 | 8.20 | 9.21 | 7.36 |
| ¹¹B | +0.158 | 7.27 | 8.28 | 6.43 |
| ²⁷Al | −2.644 | 4.66 | 5.73 | 3.78 |
| ⁹Be | +5.702 | 13.08 | 14.10 | 12.24 |
| ²⁵Mg | +2.654 | 10.14 | 11.20 | 9.27 |
| ²⁹Si | −1.527 | 5.86 | 6.92 | 4.98 |

**¹⁹F(α,n)²²Na — the dominant LZ (α,n) source (PTFE) — cannot produce a 248 keV xenon recoil by elastic scattering at all**
(endpoint 6.31 MeV even for the 8.785 MeV ²¹²Po α; E_R,max = 191 keV). Ti, Cu and Fe have Q ≲ −3 to −7 MeV and Coulomb
barriers > 10 MeV; their (α,n) yields are negligible **[recalled: certain]**. The only routes above 8.2 MeV are trace light
nuclei: ¹³C (1.1 % of the carbon in PTFE), ¹⁷O, ¹⁰B/¹¹B (borosilicate glass, if any), ⁹Be and ²⁵Mg (not significant LZ
materials). For ¹³C(α,n) to reach 8.20 MeV the α must exceed 6.26 MeV, i.e. only ²¹⁴Po (U chain) and ²¹⁶Po/²¹²Po (Th chain).

A semi-quantitative estimate of the fraction of PTFE (α,n) neutrons above 8.2 MeV uses thick-target yields per 10⁶ α at
5.3 MeV of ≈ 12 (F) and ≈ 0.11 (natural C), scaling as E_α^{3.5}, a 60 % ground-state branch and the isotropic-CM
angular fraction above threshold **[recalled: uncertain, ×2–3]**: U chain 4.1 × 10⁻⁴, Th chain 6.2 × 10⁻⁴. We model the
(α,n) spectrum as (1 − f_C)·[E e^{−E/1.1 MeV}, truncated at 6.3 MeV] **[shape recalled: uncertain; irrelevant above 6.3 MeV]**
plus f_C·[flat, 6.5–10.5 MeV] with **f_C = 2.5 × 10⁻³ central** (fraction above 8.20 MeV 1.44 × 10⁻³, above 6.61 MeV
2.43 × 10⁻³), and vary f_C = 5 × 10⁻⁴ … 10⁻² (the upper value would correspond to a boron- or beryllium-bearing component
with a sizeable share of the total yield). The 8 % untagged fraction (l. 178) applies to all of these.

### 3.2 ²³⁸U spontaneous fission
Watt spectrum dN/dE ∝ e^{−E/a} sinh√(bE) with a = 0.988 MeV, b = 2.249 MeV⁻¹ **[recalled: likely; SOURCES-4C convention]**:
mean 2.03 MeV (check: 3a/2 + a²b/4 = 2.03 ✓); fraction above 6.61 MeV 1.74 × 10⁻², above 8.20 MeV 5.2 × 10⁻³, above 10 MeV
1.3 × 10⁻³. ν̄ = 2.0 and ≈ 7 prompt γ per fission **[recalled: likely]**; P(ν) = 0.05, 0.25, 0.37, 0.25, 0.07, 0.01
**[recalled: uncertain]** (model ν̄ = 2.07).

### 3.3 Muon-induced neutrons
dN/dE ∝ E⁻¹ from 10 MeV to 1 GeV (central) and E⁻² (variant) **[recalled: likely]**; all above threshold by construction.
LZ's own rock-neutron simulation range is 50 MeV–1 GeV. Figure 2 (`figures/P013_fig2_source_spectra.png`).

### 3.4 Cross sections
n–Xe elastic/nonelastic cross sections **[recalled: likely, ±30 %]** (barns): 0.5 MeV 6.0/0.3; 1: 5.5/0.9; 2: 4.8/1.4; 3: 4.2/1.7;
5: 3.7/1.9; 8: 3.4/1.9; 10: 3.5/1.8; 14: 2.9/1.8; 20: 2.6/1.7; 50: 2.3/1.6; 100: 2.0/1.6; 300: 1.3/1.5; 1000: 1.0/1.6
(log-log interpolation). With n_Xe = 1.33 × 10²² cm⁻³ (ρ = 2.9 g cm⁻³): λ_el = 21.5 cm and λ_tot = 14.2 cm at 10 MeV;
λ_el = 37.6 cm, λ_tot = 20.9 cm at 100 MeV; λ_tot = 12.1 cm at 2 MeV, 28.9 cm at 1 GeV. The assignment's σ_el ≈ 4 b
(10 MeV) and ≈ 2 b (100 MeV) lie within the adopted range.

### 3.5 First-scatter fractions (thin-target, weight S(E)·σ_el(E))
`first_scatter_fractions.csv`:

| source | f(E_R > 200) iso | black disk | f(200–270) iso | black disk | f(E_R < 55) black disk |
|---|---|---|---|---|---|
| (α,n) model | 3.7e-4 | 1.6e-5 | 3.1e-4 | 1.1e-5 | 0.990 |
| ²³⁸U SF | 1.8e-3 | 6.1e-5 | 1.55e-3 | 4.3e-5 | 0.990 |
| muon E⁻¹ | 0.75 | 5.7e-2 | 0.067 | 1.0e-2 | 0.868 |
| muon E⁻² | 0.63 | 4.0e-2 | 0.129 | 1.03e-2 | 0.883 |

## 4. Monte Carlo transport and single-scatter classification (script §3)

Geometry: homogeneous LXe cylinder of radius 72.8 cm and height 145.6 cm (7 t active) **[recalled: likely]**. Two fiducial
volumes: FV-A r < 63 cm, 4 < z < 135 cm (4.74 t, matching LZ's 4.71 t) and FV-B r < 60 cm, 15 < z < 130 cm (3.77 t, the
assignment's suggestion). Sources: (α,n) and fission from the lateral wall (PTFE) with isotropic inward-hemisphere directions;
a variant from the end faces (PMT arrays); muon-induced neutrons area-weighted over the whole surface. Transport: distance to
next interaction from λ_tot(E); nonelastic interactions deposit ER+NR energy and terminate the history (they are never a
clean NR single scatter); elastic scatters sample x = qR from the black-disk CDF truncated at x_max = 2kR (inverse-CDF on a
260 001-point grid), E_R = (qħc)²/2M, cos θ = 1 − q²/2k², direction rotated, energy reduced; tracking stops below 0.1 MeV
(E_R,max = 3 keV). A "single scatter" is exactly one deposit above threshold E_thr (3 keV default; 1 and 5 keV variants; a
sub-threshold second deposit would not create a separate S2 above the 645 phd raw-S2 cut) anywhere in the active volume,
elastic, with the deposit inside the FV. Sample sizes: 1.5 × 10⁶ histories for each high-energy component (¹³C-like flat
6.5–10.5 MeV; Watt above 6.61 MeV; muon E⁻¹ and E⁻²), 6 × 10⁵ for the bulk components (F below 6.3 MeV; Watt below
6.61 MeV), 6 × 10⁵ for the end-face variant, 1.5 × 10⁶ for the isotropic-angle variant, 3 × 10⁵ per monoenergetic run.
The bulk components have E_R,max < 200 keV and contribute exactly zero to the window — by kinematics, not statistics.

Monoenergetic results (wall source, FV-A, 3 keV):

| E_n [MeV] | P(any deposit) | P(SS in FV \| deposit) | SS(200–270)/SS(FV) | ⟨n_el⟩ | nonelastic fraction |
|---|---|---|---|---|---|
| 8.5 | 0.893 | 0.006 | 4.8e-2 | 1.32 | 0.72 |
| 10 | 0.893 | 0.006 | 5.1e-2 | 1.38 | 0.71 |
| 15 | 0.878 | 0.007 | 3.4e-2 | 1.14 | 0.72 |
| 50 | 0.860 | 0.010 | 1.3e-2 | 1.02 | 0.70 |
| 100 | 0.850 | 0.011 | 1.1e-2 | 0.89 | 0.71 |
| 200 | 0.827 | 0.014 | 1.3e-2 | 0.70 | 0.70 |

Only ≈ 0.6–1.4 % of neutrons that deposit anything produce a clean single NR deposit inside the FV: with λ_tot ≈ 14 cm a
neutron typically interacts more than once before escaping and 70 % of histories end in a nonelastic (ER-producing)
interaction. Among clean single scatters from 8.5–10 MeV neutrons, ≈ 5 % lie in 200–270 keV (the forward-peaked recoils
cluster below 60 keV and around the 100 keV diffraction maximum).

Combined per-source SS spectra (`ss_spectrum_fractions.csv`; FV-A, 3 keV unless stated):

| source | P(SS in ROI) per neutron entering LXe | F ≡ N_SS(200–270)/N_SS(5.4–270) | companions N_SS(5.4–55)/N_SS(200–270) | N_SS(5.4–55)/N_SS(>200) |
|---|---|---|---|---|
| (α,n), f_C = 2.5e-3 | 7.9e-3 | 5.1e-5 (MC stat. ±6.5 %) | 1.9 × 10⁴ | 1.2 × 10⁴ |
| (α,n), f_C = 1e-2 | 7.9e-3 | 2.0e-4 | 4.7 × 10³ | 3.0 × 10³ |
| (α,n), f_C = 5e-4 | 7.9e-3 | 1.0e-5 | 9.5 × 10⁴ | 6.1 × 10⁴ |
| (α,n), isotropic angles, f_C = 2.5e-3 | 7.9e-3 | 1.05e-3 | 917 | 754 |
| ²³⁸U SF (Watt) | 8.4e-3 | 1.9e-4 (±8.5 %) | 5.1 × 10³ | 3.4 × 10³ |
| muon E⁻¹ | 8.9e-3 | 2.2e-2 | 39.5 | 5.3 |
| muon E⁻² | 7.0e-3 | 3.5e-2 | 23.5 | 5.0 |

Robustness: over E_thr = 1–5 keV and both FVs, F for the central (α,n) model spans 3.9–7.7 × 10⁻⁵ and the companion ratio
1.2–2.5 × 10⁴; for the muon spectra F spans 2.0–2.2 × 10⁻² (E⁻¹) and 3.1–4.0 × 10⁻² (E⁻²). The end-face source gives
F = 180/5015 = 3.6 % for the high-energy component versus 240/6765 = 3.5 % from the wall. The black-disk radius changes the
single-energy window fraction by ×0.4 (R = 5.6 fm) to ×2 (7.1 fm) (Table in §2), so F carries a further factor ≈ 2.5 either
way; the isotropic variant (×20) is the extreme upper bound.

For the muon spectra the ratio of single deposits of any kind (including single nonelastic deposits) to elastic ROI single
scatters is ≈ 17, so relative to *all* single scatters the 200–270 keV elastic fraction is 1.3 × 10⁻³ (E⁻¹) and
2.0 × 10⁻³ (E⁻²).

## 5. Expected counts (script §4)

Veto partition (Tables S1–S2): science 0.08, delayed 0.87, prompt 0.05. One neutron single scatter in the science sample
implies 10.9 in the delayed-veto sample and 0.63 in the prompt sample; the delayed fit interval [0, 1.3] maps to a science
count of 0.120, reproducing Table I's 0.118. The paper's numbers are therefore already the *untagged* (science-sample)
counts; changing the untagged fraction from 8 % to 4–12 % rescales them by 0.5–1.5.

Fission silent-companion factor: a fission that yields the observed single scatter must have its other ν − 1 neutrons and
≈ 7 γ leave no signal in the TPC, Skin or OD. With per-particle "silent" probabilities p_n = 0.07 (0.05–0.15) and
p_γ = 0.15 (0.10–0.30) **[recalled/estimated: uncertain]**, f_silent = Σ_ν P(ν) ν p_n^{ν−1}/ν̄ × e^{−7(1−p_γ)} =
3.8 × 10⁻⁴ (2.6 × 10⁻⁴ – 1.4 × 10⁻³) (`fission_silent_factor.csv`). This is the quantitative content of LZ's "high probability
of multiple interactions".

Expected 200–270 keV single scatters in 2.84 t·yr (`expected_counts.csv`):

| scenario | normalisation | N(200–270) | P(≥ 1) |
|---|---|---|---|
| (α,n) central (f_C = 2.5e-3, black disk) | 0.118 × F | 6.0e-6 | 6.0e-6 |
| (α,n) central | 0.02 (MS cross-check) × F | 1.0e-6 | 1.0e-6 |
| (α,n) central | ≤ 3 NR events in 5.4–55 keV ÷ companions | 1.6e-4 | 1.6e-4 |
| (α,n) f_C = 1e-2 | 0.118 × F | 2.4e-5 | 2.4e-5 |
| (α,n) f_C = 5e-4 | 0.118 × F | 1.2e-6 | 1.2e-6 |
| (α,n) isotropic angles | 0.118 × F | 1.2e-4 | 1.2e-4 |
| (α,n) isotropic angles | ≤ 3 low-E NR events | 3.3e-3 | 3.3e-3 |
| ²³⁸U SF, 1 % of (α,n), central silent factor | 0.118 × 0.01 × 1.06 × 3.8e-4 × F_SF | 9.3e-11 | 9.3e-11 |
| ²³⁸U SF, no multiplicity penalty | 0.118 × 0.01 × 1.06 × F_SF | 2.4e-7 | 2.4e-7 |
| muon E⁻¹ | 4.6e-4 × F(all SS) | 5.9e-7 | 5.9e-7 |
| muon E⁻¹, elastic-only denominator | 4.6e-4 × F | 9.9e-6 | 9.9e-6 |
| muon E⁻² | 4.6e-4 × F(all SS) / elastic-only | 9.4e-7 / 1.6e-5 | — |
| muon, ≤ 3 low-E NR events | 3 ÷ companions | 0.076 (E⁻¹) / 0.128 (E⁻²) | 0.07 / 0.12 |
| **all sources, central** | | **6.6e-6** | **6.6e-6** |
| **all sources, every conservative choice stacked** (isotropic angles, f_C = 10⁻², maximal silent factor, elastic-only muon denominator) | | **5.1e-4** | **5.1e-4** |

The stacked-conservative (α,n) entry combines isotropic angles with f_C = 10⁻²: F scales linearly with f_C in this regime
(the bulk is kinematically zero), so F = 4 × 1.05 × 10⁻³ = 4.2 × 10⁻³, companions = 917/4 ≈ 230, and N = 0.118 × F =
5.0 × 10⁻⁴; this is the value quoted as "≤ 4 × 10⁻³ / ≥ 230 / ≤ 5 × 10⁻⁴" in the paper's table.

The "≤ 3 low-energy NR events" bound uses the recalled tolerance of the 2024 LZ search **[recalled: uncertain; P003 used 3–5]**
and is the weakest of the three normalisations for (α,n); it is the only one that is informative *without* trusting the veto
constraint. For muon-induced neutrons it is weak (their SS spectrum is hard: only 24–40 companions below 55 keV per
200–270 keV event, 5 per event above 200 keV) — for this source the operative constraint is LZ's simulated UL, with its
stated factor-2 uncertainty: ≤ 2 × 10⁻⁵ even when doubled.

What one (α,n) event in 200–270 keV would require: 1/F = 2.0 × 10⁴ NR single scatters in the science ROI, i.e. 2.1 × 10⁵
NR-band events in the delayed-veto sample (fit interval [0, 1.3]; 55 events observed in total) and 1.9 × 10⁴ science-sample NR
events below 55 keV.

Muon UL consistency: zero signal-like events in 1200 simulated years give a 90 % UL of 2.30 × 0.602/1200 = 1.16 × 10⁻³ for
220 live days before tagging, 2.3 × 10⁻⁴ after 80 % tagging; the paper's 4.6 × 10⁻⁴ lies between the two, consistent with
partial tagging of muon-accompanied events.

## 6. Inelastic (n,n′γ) hybrids (script §5)

¹²⁹Xe 39.58 keV (t½ 0.97 ns, mostly converted) and ¹³¹Xe 80.19 keV (0.48 ns) **[recalled: certain]** are prompt, so an
inelastic scatter deposits NR + ER at one site. With `lz.nest_nr_yields` (Table S5 incl. the p(E) break) and
`lz.nest_er_yields(…, NEST_ER_LZ)` (Table S3), g₁ = 0.110, g₂ = 34.5:

| configuration | S1c [phd] | log₁₀S2c | comment |
|---|---|---|---|
| pure NR median at S1c = 540.1 | 540.1 | 4.027 | E_NR = 266 keV; event at 3.967 is −0.060 dex = −1.8σ for σ = 0.033 dex (P009), paper: −1.5σ |
| 248 keV NR + 39.6 keV ER | 745 | 4.539 | outside the ROI (S1c > 600, log S2c > 4.15) |
| 248 keV NR + 80.2 keV ER | 992 | 4.786 | outside the ROI |
| NR + 39.6 keV with S1c = 540 | 540 (E_NR = 157 keV) | 4.520 | +0.49 dex = +15σ above the NR median |
| NR + 80.2 keV with S1c = 540 | 540 (E_NR = 34 keV) | 4.740 | +0.71 dex = +22σ above the NR median |

Any ER admixture (inelastic γ, capture γ, fission γ) moves an event *up* in log S2c toward the ER band; the event is *below*
the NR median. Hybrids are excluded as an explanation of this particular event (Figure 4, `figures/P013_fig4_hybrid_loci.png`).

## 7. AmBe-related neutron emitters (script §6: none computed)

The 8 June 2023 AmBe calibration (l. 301) activates xenon to ¹²⁵Xe, ¹²⁷Xe, ¹²⁹ᵐXe, ¹³¹ᵐXe, ¹³³Xe, ¹³⁵Xe, ¹³⁷Xe — all
EC/IT/β⁻ emitters producing ERs **[recalled: certain]**; β-delayed neutron precursors reachable by neutron capture on LZ
materials (¹⁷N from ¹⁷O(n,p), ⁸⁷Br, ¹³⁷I) have half-lives of seconds to minutes and are gone within an hour, eight days before
the event. Photoneutrons from the 4.44 MeV AmBe γ exist only while the source is deployed. The source was removed; the
¹²⁵Xe → ¹²⁵I chain it left behind is an ER background treated by P010. No neutron-emitting activation product survives.

## 8. Validation, robustness and failed approaches

- Kinematic threshold reproduces the paper's "≥ 8 MeV" (8.20 MeV natural Xe; 7.7–8.5 across isotopes).
- Watt mean 2.03 MeV matches 3a/2 + a²b/4; the veto-partition arithmetic reproduces Table I's 0.118 from the delayed [0, 1.3].
- Black-disk backward/forward ratio (≈ 4 × 10⁻³ at the third maximum) is consistent with recalled 14 MeV n–Pb data (~10⁻³);
  optical-model fits generally give *lower* secondary maxima than a sharp disk, so our window fractions are, if anything, high.
- Thresholds 1/3/5 keV and two FVs change F by < ×1.5; the source face (wall vs PMT arrays) changes it by 3 %.
- The seeded MC is byte-reproducible; MC statistical errors on F are 6.5 % ((α,n) high component, 240 events) and 8.5 % (SF).
- Failed/abandoned: (i) the requested exponential angular form exp(−a(1 − cos θ)) with a = (kR)²/2 was implemented but gives
  f(> 200 keV) ≈ 3 × 10⁻⁶, three orders of magnitude below the diffraction pattern — it lacks the secondary maxima that
  dominate large-angle scattering, so it is reported only as a failed parametrisation; (ii) the first script run failed on a
  missing keyword argument (fixed); (iii) the MC error propagation initially used max(count, 1) for zero-count bulk
  components, inflating the quoted error — the bulk is kinematically zero above 200 keV, so this was removed; (iv) an
  attempt to derive the absolute ¹³C share from first principles was replaced by the recalled thick-target yields with a
  ×3 uncertainty, bracketed by f_C = 5 × 10⁻⁴ – 10⁻².

## 9. Discussion

Three independent facts each exclude a neutron origin. (1) *Spectrum*: the dominant ¹⁹F(α,n) source ends at 6.3 MeV and can
give at most 191 keV; the residual high-energy tail (¹³C, ¹⁷O, B) is ~10⁻³ of the yield, and the diffraction-limited angular
distribution puts only ~1 % of its scatters into 200–270 keV. (2) *Multiplicity*: only ~1 % of neutrons that interact give a
single clean NR deposit inside the FV, and for fission the silent-companion penalty is a further ~4 × 10⁻⁴. (3) *Population*:
one (α,n) event at 200–270 keV would be accompanied by ~10⁴ single scatters below 55 keV and ~2 × 10⁵ NR-band events in the
delayed-veto sample, against LZ's ≤ 0.118 and 55 total. Muon-induced neutrons are the only source whose SS spectrum is hard
enough (5 companions above 200 keV per window event) to evade the population argument, but LZ's 1200-year simulation caps them
at 4.6 × 10⁻⁴ of any energy, of which ≈ 0.1–2 % lie in the window. Summing, we expect 7 × 10⁻⁶ neutron single scatters in
200–270 keV, or ≤ 5 × 10⁻⁴ when every uncertain input is pushed against the conclusion. This confirms and quantifies LZ's
qualitative statement; together with P004 (wall MSSI) and P010 (ER leakage) the conventional single-site backgrounds are now
each disfavoured below the 10⁻³ level, leaving accidentals, RFR MSSI and detector artifacts as the remaining non-DM options.

## 10. References

1. LZ Collaboration, arXiv:2609.02823 (2026) — Data Analysis (l. 141–178), Table I, Discussion (l. 296–327), Tables S1–S2,
   supplement "Neutrons" (l. 787–798).
2. B. E. Watt, Phys. Rev. 87, 1037 (1952) — fission neutron spectrum.
3. W. B. Wilson et al., SOURCES-4C, LA-UR-02-1839 (2002) — (α,n) yields and spectra (Watt parameters convention).
4. V. A. Kudryavtsev, Comput. Phys. Commun. 180, 339 (2009) — MUSUN.
5. D.-M. Mei and A. Hime, Phys. Rev. D 73, 053004 (2006) — muon-induced neutron yields and spectra.
6. LZ Collaboration, Astropart. Phys. 125, 102480 (2021) — LZ simulations (fission multiplicity).
7. LZ Collaboration, Phys. Rev. D 108, 012010 (2023) — LZ backgrounds.
8. Corpus: P003 (low-energy tolerance), P004 (wall MSSI), P009 (band width 0.033 dex), P010 (ER leakage), dossier P000 §6.7.

## 11. Tools and provenance (mirrors `output/provenance/P013.json`)

- Agent tools: Read ×16 (PAPER_GUIDE; dossier; results_ledger.csv; P004.md; P010.md; fulltext.tex l. 140–250, 296–330, 466–540,
  540–590, 785–800; lzcommon.py l. 1–75 and 255–300; provenance/P010.json; three own figures), Bash ×12 (grep of tex;
  lzcommon API/versions; nestpy/periodictable probe; script runs ×4 incl. one failed; table inspection; word-count/JSON
  checks ×5), Write ×4 (script, details.md, P013.json, P013.md), Edit ×15 (script ×5: frac_between signature; R-sensitivity ×2;
  figure y-range; error propagation; paper ×7: word-budget trims; details ×2; provenance ×1).
- Software: python 3.12.13; numpy 2.5.3 (random.default_rng, interp, trapezoid, cross, cumsum); scipy 1.18.1 (special.j1,
  special.jn_zeros, integrate.quad, optimize.brentq, stats); pandas 3.0.5; matplotlib 3.11.2 (Agg); periodictable 2.1.0
  (isotopic masses); nestpy 2.1.1 via lzcommon (nest_nr_yields, nest_er_yields with LZ Tables S3/S5); common/lzcommon.py (LZ
  dict, XE_ISOTOPES, NEST parameter sets). Hand derivations: two-body kinematics, black-disk pattern, silent-companion factor.
- Recalled knowledge: 12 items, listed in the JSON with reliabilities.
- Datasets: none. Data requests: none. WimPyDD files: none.
