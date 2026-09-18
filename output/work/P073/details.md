# P073 — What the vetoes know: research record

Simulated date 2026-09-15 · Category BKG · physics.ins-det · author profile: veto-system experimentalists.
Script: `output/code/P073_veto_information.py` (run from the simulation root with `.venv/bin/python`; 385 s).
All numbers below are printed by the script or stored in `output/work/P073/P073_summary.json` and the CSV tables listed in §9.

## 1. Motivation and framework

LZ partitions its data into *science*, *prompt-veto* and *delayed-veto* samples (tex l. 141–147): prompt = Skin > 2.5 phd within
±0.25 μs or OD > 4.5 phd within ±0.3 μs of the S1; delayed = Skin > 300 keV or OD > 200 keV within 600 μs; random coincidences
veto 0.01 % (prompt) and 2.86 % (delayed) of unrelated TPC events. The 248 keV candidate is in the *science* sample: no Skin or OD
pulse in either window. For a wall MSSI the second interaction sits ≤ 3 mm from the PTFE, and the outgoing γ must leave through
the Skin; the paper's simulation gives a prompt tagging efficiency λ_MSSI = 94 ± 2 % for all WS-ROI MSSI (l. 194) and states that
λ_MSSI is much lower in the HE SB because ²¹⁴Pb RFR MSSI have the γ fully contained (l. 723). We ask (a) what tagging efficiencies
the paper's own numbers imply for wall vs RFR MSSI, (b) what the efficiency is for the *specific* decomposition LZ gives for the
event (12 ± 2 keV at the vertex + 77 ± 7 keV in the wall shell, or 204 keV in the RFR; l. 733–734), (c) what likelihood ratio the
event's silence in the vetoes carries, (d) whether the delayed-sample count (55 vs 50.4 ± 2.0) is informative, and (e) what LZ could
still extract from the veto waveforms.

Key definitions. For a background class H, P(science | H) = (1 − ε_prompt,H)(1 − f_rp)(1 − ε_delayed,H)(1 − f_rd) with
f_rp = 10⁻⁴, f_rd = 0.0286. For DM, accidentals and atmospheric-ν CEνNS (uncorrelated with the vetoes) P(science) = 0.9713. The
veto-silence likelihood ratio is LR_H = P(science | H)/P(science | DM). Because LZ's science-sample MSSI expectation (0.0048 wall)
already contains the factor (1 − 0.949) from the simulated prompt efficiency, the *extra* information for the event topology is the
ratio (1 − ε_topology)/(1 − 0.949), which multiplies P004's and P033's science-sample expectations (and divides their k_required).

## 2. Part 1 — tagging efficiencies from the paper's numbers

Supplement MSSI table (l. 765–783), simulated columns; ε = N_prompt/(N_prompt + N_science):

| region | type | vol (t) | sim sci | sim prompt | ε_sim | untagged | obs sci | obs prompt | ε_data (68 %) | 90 % LL |
|---|---|---|---|---|---|---|---|---|---|---|
| WS ROI | wall | 4.7 | 0.0048 | 0.09 | **0.949** | 5.1 % | – | – | – | – |
| WS ROI | RFR | 4.7 | 0.0001 | 0.10 | 0.999 | 0.1 % | – | – | – | – |
| WS ROI | wall | 5.4 | 0.03 | 0.1 | 0.769 | 23 % | 0 | 1 | 1 (0.16–1) | 0.10 |
| WS ROI | RFR | 5.4 | 0.003 | 1.9 | 0.998 | 0.2 % | 0 | 2 | 1 (0.40–1) | 0.32 |
| HE SB | wall | 4.7 | 0.1 | 0.3 | 0.75 | 25 % | 0 | 1 | 1 (0.16–1) | 0.10 |
| HE SB | RFR | 4.7 | 0.5 | 0.5 | 0.50 | 50 % | 0 | 0 | – | – |
| HE SB | wall | 5.4 | 0.5 | 2.2 | 0.815 | 19 % | 0 | 0 | – | – |
| HE SB | RFR | 5.4 | 21.5 | 5.2 | **0.195** | 81 % | 18 | 3 | 0.143 (0.066–0.262) | 0.054 |

Aggregates (`P073_tagging_efficiency_aggregated.csv`): all six wall sideband bins 0.63 sci / 2.60 prompt → ε_sim = 0.805; observed
0 / 2 → ε_data = 1 (68 %: 0.40–1), 90 % lower limit 0.32. RFR excluding HE-SB 5.4 t: sim 0.83, observed 0 / 2 → same limits.
The only category with a real data test is the ²¹⁴Pb-dominated HE-SB 5.4 t RFR bin: 3 of 21 tagged, ε_data = 0.143 (0.066–0.262)
vs simulated 0.195, binomial p = 0.78 (agreement). **The data test the wall tagging efficiency only to ε_wall > 0.32 (90 % CL).**

Tables I/S1: MSSI 7.7 × 10⁻² prompt vs 4.9 × 10⁻³ science → ε = 0.940 = λ_MSSI; detector ERs 62.2 vs 8.5 → 0.880 = λ_PG. Both
prompt expectations are therefore *derived* from the λ's (0.0049 × 0.94/0.06 = 0.0768). The direct simulation counts in the MSSI
table give 0.09 + 0.10 = 0.19 prompt MSSI in the same 4.7 t WS ROI, 2.47× more than Table S1, i.e. a combined ε = 0.975 (wall 0.949,
RFR 0.999), untagged 2.5 % rather than 6 %. Either the table's "prompt" column uses a wider definition or λ_MSSI = 0.94 is a
conservative rounding; the science-sample expectation (0.0048) is unaffected, the prompt-sample MSSI expectation is uncertain by ×2.5
(irrelevant among 62 detector ERs). The wall/RFR split of the efficiency is strongly energy- and region-dependent: 0.95 (WS ROI 4.7 t
wall) to 0.75–0.82 (wall sidebands, more energy left in the dead region, less γ energy to escape) and 0.999 (WS ROI RFR, escaping
γ's) to 0.195 (HE SB RFR, contained γ's).

## 3. Part 2 — topology-specific tagging efficiency (photon-transport MC)

### 3.1 Geometry and materials (cm; z = 0 at the cathode; recalled unless bold)
Active TPC r < 72.8, 0 < z < 145.6; wall dead shell 72.5–72.8 (**≤ 3 mm**, l. 195); **RFR 13.75 cm** below the cathode plus a 2.25 cm
gap to the bottom PMT array (15 cm thick, effective density ρ_PMT = 0.8 g cm⁻³, Compton only, silent; bracket 0.4–1.2); bottom Skin
dome 25 cm LXe below the array; side PTFE 2 cm (1–3); side Skin LXe 6 cm (4–8) up to the liquid level; Ti ICV 0.8 cm, vacuum 7 cm,
Ti OCV 0.8 cm, water 4 cm, GdLS 61 cm side annulus (**17 t GdLS**), 45 cm top/bottom tanks (30–60); water beyond to r = 260 cm
(silent). Gas above the liquid (4.4 cm) and the top PMT array (15 cm) are silent. Skin, RFR and active LXe are "visible" to the
TPC PMTs; deposits there are scored separately (active → extra S2; shell/RFR → extra S1-only light).

Cross-sections: Klein–Nishina exact (σ_KN(1 MeV) = 0.2112 b, check passed); electron densities from ρ and Z/A (LXe 2.86 g cm⁻³,
Z/A = 0.411; PTFE 2.2, 0.48; Ti 4.5, 0.459; GdLS 0.86, 0.571; water 1.0, 0.555). LXe photoabsorption = 1/λ_total − μ_KN with the
corpus attenuation anchors λ = 0.30/0.62/0.99/1.42/2.30/2.75/2.9/6.3/9.1 cm at 122/164/203/243/300/352/375/1000/2615 keV (P004,
P029, P042, P033; recalled XCOM-like, ±20 %), log-log interpolated; the 2615 keV anchor folds pair production into "absorption".
Ti photoabsorption = Xe value × (22/54)^4.5 × atom-density ratio (0.076 ×; matches XCOM at 300 keV to ~10 %). Light materials
Compton-only. Resulting λ_LXe = 2.30/4.52/6.3/7.35/9.1 cm at 300/609/1000/1461/2615 keV.

Transport: 0.25 cm ray-march with P_int = 1 − e^{−μ ds}, photo vs Compton by μ_a/μ, Klein–Nishina angle by rejection sampling,
tracking to E < 5 keV (deposited locally) or escape. Skin threshold 2.5 phd ≙ 5 keV (bracket 2–20), OD 4.5 phd ≙ 50 keV (20–200);
both are recalled/uncertain, and §3.4 shows the Skin threshold is irrelevant.

### 3.2 Histories
Initial γ lines E₀ = 609, 1120, 1764 (²¹⁴Bi), 1250 (⁶⁰Co proxy), 1461 (⁴⁰K), 2615 keV (²⁰⁸Tl), equal weights. Two orderings:
* **A** (vertex first): lines through the vertex (r = 45.9, z = 26.4) isotropic, kept if the forward ray hits the side wall shell
  (z in 0–145.6); weight = exp(−d_in/λ(E₀) − d_out/λ(E₀−12)) × P_shell(ψ) with P_shell = 1 − exp(−0.3 cm/(λ cos ψ)) (glancing
  incidence favoured), × exp(−13.75/λ) for entries through the cathode (silent RFR crossing). The 77 keV Compton fixes the polar
  angle (θ = 29.2°/15.1°/13.4°/11.4°/9.4°/6.3° for the six E₀), azimuth random; the γ (E₀ − 89 keV) is transported from a point
  uniform within the shell path.
* **B** (wall first): shell points uniform over the side wall, ray to the vertex, weight exp(−d/λ(E₀−77)); 12 keV Compton at the
  vertex (θ ≈ 4–7°); the active-TPC crossing is weighted analytically by exp(−t_exit/λ(E₀−89)) and the photon started at the exit
  point (shell edge, cathode or gate plane).

Classification (conditional on reproducing the observed event): any deposit > 1 keV in the active LXe (fails single-scatter), in
the shell (extra S1-only light beyond the 471 phd budget) or in the RFR (idem) → *excluded from the topology*; otherwise Skin deposit
≥ threshold → Skin veto, GdLS ≥ threshold → OD veto, else *silent* (the event would be in the science sample). ε_topology =
W(veto)/W(kept); MC error from the effective sample size n_eff = (Σw)²/Σw².

### 3.3 Central results (`P073_topology_tagging_central.csv`, N = 30 000 per run)

| E₀ (keV) | E_out | A: f_tpc | f_shell | ε_A | Skin / OD share | 1−ε_A | B: f_RFR | f_keep | ε_B |
|---|---|---|---|---|---|---|---|---|---|
| 609 | 520 | 0.159 | 0.159 | 0.988 ± 0.003 | 0.93 / 0.06 | 1.2 % | 0.956 | 0.002 | 0.996 ± 0.031 |
| 1120 | 1031 | 0.123 | 0.075 | 0.977 ± 0.003 | 0.84 / 0.13 | 2.3 % | 0.900 | 0.033 | 0.992 ± 0.013 |
| 1250 | 1161 | 0.124 | 0.083 | 0.975 ± 0.003 | 0.82 / 0.16 | 2.5 % | 0.888 | 0.047 | 0.993 ± 0.012 |
| 1461 | 1372 | 0.103 | 0.076 | 0.977 ± 0.003 | 0.80 / 0.17 | 2.3 % | 0.873 | 0.061 | 0.991 ± 0.010 |
| 1764 | 1675 | 0.100 | 0.062 | 0.975 ± 0.003 | 0.77 / 0.20 | 2.5 % | 0.851 | 0.077 | 0.992 ± 0.008 |
| 2615 | 2526 | 0.086 | 0.056 | 0.970 ± 0.003 | 0.73 / 0.24 | 3.0 % | 0.818 | 0.124 | 0.978 ± 0.008 |

Ordering A mix: **ε = 0.9767 ± 0.0013, untagged 2.33 %**, of which side route 2.19 %, top corner 0.10 %, bottom 0.03 %.
Ordering B mix: ε = 0.990 ± 0.006 (untagged 0.99 %); every B survivor exits through the cathode (f_exit_top_up = 0), 82–96 % are
then excluded by RFR light, so B is a rare sub-class (0.2–12 % kept) and A is taken as central; equal A/B mix 1.66 %.
Among silent A histories 19.4 % have a sub-threshold deposit (OD 18.6 %, Skin 1.1 %).
If extra shell/RFR light were *allowed* (i.e. the class of all wall/RFR MSSI regardless of S1), ε drops to 0.53: this is the regime
of the HE SB, and it is why the strict S1 budget matters.

RFR decomposition (12 keV + 204 keV in the RFR, γ heading down from the vertex; `P073_rfr_topology.csv`): 66–83 % of histories
deposit further energy in the RFR (excluded); of the rest ε = 0.992/0.994/0.989 ± 0.002–0.003 at 1120/1461/2615 keV — consistent
with the paper's 0.999 for WS-ROI RFR MSSI. Allowing the extra RFR light gives ε = 0.27/0.37/0.49, the contained-γ regime the paper
invokes for the HE SB (0.195 for 352 keV ²¹⁴Pb γ's).

### 3.4 Robustness (`P073_threshold_scan.csv`, `P073_topology_variants.csv`; variants N = 10 000, MC error 0.13–0.27 %)
Thresholds (A): Skin 2/5/10/20 keV changes 1−ε by ≤ 0.1 %; OD 20/50/100/200 keV → 2.1/2.3/2.8/3.8 %.
Geometry/attenuation (A): Skin 4 cm 3.5 %, 8 cm 1.4 %; PTFE 1/3 cm 2.3/2.2 %; OD top/bottom 30/60 cm 2.2/2.2 %; Xe μ × 0.8 /
× 1.25: 2.7/1.55 %; PMT-array ρ 0.4/1.2: 2.2/2.3 %; thin arrays + no water gap 1.4 %; weakest corner (ρ_PMT 1.2, OD top 30, Skin 4)
3.6 %; strongest (ρ_PMT 0.4, OD top 60, Skin 8) 1.5 %. Ordering B ranges 0.5–2.0 % (MC errors 0.5–1.5 %).
**Adopted: untagged = 2.3 % (central), range 0.5–4.0 %**; the side Skin thickness and the OD threshold dominate; the top corner is
< 0.2 % because the γ leaves the shell sideways and the Skin extends to the liquid level.
Seed check (skin 8 cm, three seeds, N = 9000): 0.86/1.23/0.96 % total; the first run's 2.3 % for this variant was a heavy-weight
fluctuation (n_eff/N ≈ 0.1 from the 1/cos ψ shell weights), which motivated the larger variant statistics and the MC-error column.

Slab validation (radial 1372 keV photons): GdLS λ = 18.8 cm, analytic P(no interaction in 61 cm) = 0.039, MC without the outer
water 0.0415, with it 0.0198 (back-scatter from the water tank into the OD, a real effect); Skin λ = 7.17 cm, analytic 0.433, MC
0.321 (back-scatter from Ti/OD into the Skin); P(Skin silent and OD < 50 keV) = 0.031 for radial rays, vs 2.2 % side-route silence
for the weighted angular distribution.

## 4. Part 3 — likelihood ratios from the silent vetoes (`P073_veto_silence_LR.csv`)

| hypothesis | P(science) | LR vs DM |
|---|---|---|
| DM, atmospheric ν, accidental | 0.9713 | 1 |
| wall MSSI, λ_MSSI = 0.94 | 0.0583 | 0.060 |
| wall MSSI, MSSI-table WS-ROI wall (0.949) | 0.0492 | 0.0506 |
| wall MSSI, event topology (A, central) | 0.0226 | **0.0233** (0.0052–0.0397) |
| wall MSSI, event topology, equal A/B mix | 0.0161 | 0.0166 |
| RFR MSSI, table WS-ROI (0.999) | 9.7 × 10⁻⁴ | 0.0010 |
| RFR MSSI, event topology 12+204 keV | 0.0082 | 0.0085 |
| ²¹⁴Pb RFR MSSI, γ contained (0.195) | 0.782 | 0.805 |
| detector-ER single scatter (λ_PG = 0.88) | 0.117 | 0.120 |
| neutron single scatter (λ_PN = 0.05, λ_DN = 0.87) | 0.120 | 0.124 |

Extra factor for the event topology relative to what LZ's science-sample expectation already contains: 0.0233/0.0506 = **0.46
(0.10–0.79)**; relative to λ = 0.94: 0.39. Combination: P004's k_required(10 %, neighbourhood) 619 (413–929) → 1350 (530–9000);
P033's 6.4 × 10⁴ → 1.4 × 10⁵. The delayed window carries no information for γ-MSSI (no delayed signal expected under either
hypothesis; both have the 2.86 % random factor).

## 5. Part 4 — the delayed-sample excess

55 observed vs 50.4 ± 2.0 fitted. The "+2.3σ" uses only the fitted-mean error; the Poisson probability P(N ≥ 55 | μ = 50.4) = 0.277,
marginalised over the ±2.0 Gaussian 0.283 → 0.57σ (Gaussian-combined (55 − 50.4)/√(50.4 + 4) = 0.62σ). The random-coincidence
prediction alone reproduces the fit: 1710 × 0.0286/(1 − 0.0287) = 50.35; the Table S2 components sum to 50.16. An inelastic χ₂-photon
population gives 0.0094 (δ = 300 keV, τ = 66 μs) to 0.70 (δ = 350 keV, 1.4 μs) delayed events per clean science event (P042); the
4.6-event excess would need 490 or 6.6 clean events versus the one observed. As neutrons (λ_DN = 0.87) the whole excess would imply
5.3 neutrons, 0.65 untagged in the science sample over the entire ROI (the fit's intervals are [0, 1.3] delayed and [0, 0.118]
science). Conclusion: a 0.6σ fluctuation of random coincidences, uninformative. Implied veto-pulse rates: prompt (Skin > 2.5 phd
plus OD > 4.5 phd) 10⁻⁴/1.1 μs ≈ 90 Hz (170 Hz if only one 0.6 μs window is meant); delayed (Skin > 300 keV or OD > 200 keV)
0.0286/600 μs = 48 Hz.

## 6. Part 5 — signal side
P(a DM event is randomly vetoed) = 0.01 % + 2.86 % = 2.87 %, already in the likelihood: Table S1 puts 1.0 × 10⁻⁴ and Table S2
3.1 × 10⁻² of the 1.0-event best fit in the veto samples (1.0 × 0.0286 = 0.029, consistent within rounding). Under DM the event
had a 97.1 % chance of being in the science sample; under wall MSSI of the event topology 2.3 %.

## 7. Recommendation (e) — low-threshold Skin/OD sideband around the event
Among silent wall-MSSI histories of the event topology, 19 % leave a sub-threshold deposit (OD 18.6 %, mostly < 50 keV; Skin 1.1 %).
A search of the Skin and OD waveforms within ±1 μs of the S1 with thresholds lowered until the pulse rate rises ×5 (×20) has a random
coincidence probability of 9 × 10⁻⁴ (3.6 × 10⁻³) (from the 90 Hz above-threshold rate). A genuine sub-threshold pulse would carry
LR(wall MSSI : DM) ≈ 210 (50); a null result LR ≈ 0.81 (mild). LZ should also publish the Skin/OD pulse rates versus threshold and the
sub-threshold pulse record for the 66 prompt-veto events as a calibration of the method.

## 8. Caveats and failed approaches
* Geometry beyond the TPC (PTFE, Skin thickness, vessel walls, OD tank thicknesses, PMT arrays) is recalled; the Skin thickness
  (4–8 cm) is the largest single lever (1.4–3.5 %). Veto light yields are recalled; the Skin threshold is irrelevant, the OD one
  moves the result by ×1.7.
* Weights: the 1/cos ψ shell factor and exponential survival make n_eff/N ≈ 0.1; MC errors are quoted per run; a first pass with
  N = 9000 variants showed a 1 % heavy-weight fluctuation (skin 8 cm), corrected by larger statistics and a seed check.
* First classification allowed extra shell/RFR light: it gave ε = 0.53 (wall) and 0.27–0.49 (RFR), which describes the HE-SB regime
  rather than the observed event (fixed S1 budget); replaced by the strict definition. The first route bookkeeping (first material
  left) misattributed side-route escapes to the top corner; replaced by the final-position route.
* Ordering B ignores the shell-interaction probability of the incoming γ (source direction unknown) and is a rare sub-class; it enters
  only the range. Source distributions (PMT arrays vs wall vs cryostat) are not modelled beyond the survival weights; P033 covers the
  position dependence.
* The ×2.5 discrepancy between the MSSI table's prompt counts and Table S1 cannot be resolved from the paper.
* K-fluorescence, Cherenkov light in the water and Gd captures are ignored (all would only increase tagging).

## 9. Files
`P073_tagging_efficiency_table.csv`, `P073_tagging_efficiency_aggregated.csv`, `P073_topology_tagging_central.csv`,
`P073_threshold_scan.csv`, `P073_topology_variants.csv`, `P073_rfr_topology.csv`, `P073_veto_silence_LR.csv`, `P073_summary.json`;
figures `figures/P073_fig1_tagging_efficiency.png` (efficiency by category: paper table, data check, event topology),
`figures/P073_fig2_untagged_and_LR.png` (untagged fraction vs E₀ for both orderings with the variant band; LR bar chart).

## 10. References
LZ Collaboration, arXiv:2609.02823 (2026); D. S. Akerib et al. (LZ), NIM A 953, 163047 (2020) [detector, Skin, OD];
O. Klein & Y. Nishina, Z. Phys. 52, 853 (1929); M. J. Berger et al., NIST XCOM; C. J. Clopper & E. S. Pearson, Biometrika 26, 404
(1934); corpus P004, P029, P033, P042, P053, P063, P070; dossier 00.

## 11. Tools and provenance (mirrors provenance/P073.json)
Agent tools: Read (PAPER_GUIDE, dossier, ledger head, P004/P033/P070/P042/P053/P029 papers, tex l. 80–149/184–248/425–484/515–589/
706–785, lzcommon l. 21–80, P042 details l. 35–37, P063 details l. 115–135, two figures ×2), Bash (ledger/tex greps; 4 script runs;
3 scratch diagnostics; seed check), Write (script, details, provenance, paper), Edit (script fixes ×~20).
Software: python 3.12.13, numpy 2.5.3, scipy 1.18.1 (stats.beta, stats.binomtest, stats.poisson, stats.norm), pandas 3.0.5,
matplotlib 3.11.2 (Agg), common/lzcommon.py (imported; LZ constants cross-checked). Recalled knowledge: 14 items (§3.1 and JSON).
No datasets, no data requests, no WimPyDD files.
