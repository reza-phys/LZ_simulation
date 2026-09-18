# P063 — Spontaneous fission of ²³⁸U in LZ's detector components: multiplicity, veto probability and the odds of a lone 248 keV recoil

Simulated date 2026-09-12 · category BKG · physics.ins-det (cross-list nucl-ex) · radioassay and neutron-background specialists.
Script: `output/code/P063_fission.py` (run from the root with `.venv/bin/python`, ~95 s). Tables in `output/work/P063/`:
`P063_inventory.csv`, `P063_f_gamma.csv`, `P063_expected_by_component.csv`, `P063_sensitivity.csv`, `P063_burst_signatures.csv`,
`P063_results.json`, console log `P063_console.txt`; figures in `figures/`.

## 1. Motivation and framework

LZ's supplement (`fulltext.tex` l. 787–791) says that spontaneous-fission (SF) neutrons are included in the inference, that their
rate is "up to two orders of magnitude lower than (α,n)", and that "the high neutron and γ-ray multiplicity of spontaneous fission
leads to a high probability of multiple interactions". P013 quantified this with a single-number silent-companion factor
(3.8 × 10⁻⁴, per-particle silent probabilities p_n = 0.07, p_γ = 0.15 applied to a Poisson(7) γ multiplicity) and a fission rate
fixed at 1 % of the (α,n) neutron rate, obtaining 9 × 10⁻¹¹ expected 200–270 keV single scatters (SS). Two things in that estimate
deserve a closer look. (i) The SF rate follows the ²³⁸U *parent* activity, whereas the (α,n) rate follows the *late* chain (mostly the
7.7 MeV ²¹⁴Po α) in fluorine-bearing materials; titanium and PMT glass/ceramics have negligible (α,n) yields but can carry most of the
²³⁸U mass, so a 1 % scaling from the (α,n) budget can under-count fissions. (ii) The γ-silence probability depends strongly on
where the fission happens: a γ born in the PTFE wall must cross the TPC or the Skin, while a γ born in the cryostat wall or the PMT
bodies sees thinner LXe and can be photo-absorbed silently in Ti, Kovar or glass. We rebuild the chain component by component.

A "lone 248 keV recoil" from SF requires, within one fission (all prompt emission within ~10⁻¹⁴ s, so every deposit falls in the
±0.25/±0.3 μs prompt-veto windows and inside the TPC S1):
(a) exactly one neutron enters the active LXe, scatters once at 200–270 keV inside the FV (E_n > 8.19 MeV: P013) and leaves untagged;
(b) the other ν−1 neutrons leave no TPC deposit > 3 keV, no Skin signal > 2.5 phd, no OD prompt signal > 4.5 phd and no Gd/H
capture γ above 200 keV_OD / 300 keV_Skin within 600 μs (`fulltext.tex` l. 143–144);
(c) all ~6.5 prompt γ's are silent by the same criteria;
(d) the two fission fragments (~170 MeV, range ≲ 10 μm in solids) stop inside the component (they do unless the fission occurs within
~10 μm of a LXe-facing surface: ≲ 2 × 10⁻³ of a 1 cm PTFE panel; ignored, and such an event would be a huge Skin/TPC flash anyway).

## 2. Inputs

### 2.1 From the LZ paper (`inputs/LZ_arXiv_2609.02823_fulltext.tex`)
- Veto samples and thresholds, l. 141–147: prompt veto Skin > 2.5 phd within ±0.25 μs or OD > 4.5 phd within ±0.3 μs; delayed veto Skin
  > 300 keV / OD > 200 keV within 600 μs; random-coincidence veto rates 0.01 % / 2.86 %.
- Neutron tagging efficiency for (α,n) neutrons scattering in the TPC 92 ± 4 % (l. 178) → untagged 0.08.
- Table I: detector-NR fit interval [0, 0.118] events (science sample, 220 d × 4.71 t); delayed sample 55 observed vs 50.4 ± 2.0.
- Supplement "Neutrons" l. 787–798: SF included, "up to two orders of magnitude lower than (α,n)"; MS cross-check 0.02 ± 0.02 SS;
  rock-neutron tagging 80 ± 4 %.
- Detector description l. 82–83: Skin 2 t LXe; OD 17 t GdLS (0.1 % Gd) + 229 t water.

### 2.2 From the corpus
- P013 (`output/work/P013/P013_results.json`): E_n,min(248 keV) = 8.195 MeV, (200 keV) 6.609 MeV; Watt a = 0.988 MeV, b = 2.249 MeV⁻¹;
  transport-MC probabilities per neutron *entering* the active LXe (threshold 3 keV, FV r < 63 cm, 4 < z < 135 cm, 4.74 t):
  p_SS,ROI(α,n) = 7.89 × 10⁻³, p_SS,ROI(SF) = 8.38 × 10⁻³, **p_win(SF) ≡ P(exactly one deposit, in the FV, 200–270 keV) = 1.61 × 10⁻⁶**
  (6.3 × 10⁻⁷ – 2.05 × 10⁻⁶ over thresholds 1/3/5 keV and the two FVs); P(any deposit | enter, Watt) = 0.871
  (`MC.samples.watt_bulk.thr3_fv63x131`); P013's P(ν) recall (0.05, 0.25, 0.37, 0.25, 0.07, 0.01) and silent factor 3.8 × 10⁻⁴.
- P049: Gd capture time ~30 μs, water ~200 μs (well inside 600 μs); OD untagged-fraction reasoning.
- `lzcommon.LZ`: live days, neutron_veto_eff, detector_NR_fit_interval.

### 2.3 Library
- radioactivedecay 0.6.1 (ICRP-107): ²³⁸U T½ = 4.468 × 10⁹ y, decay modes α and SF, **SF branching 5.45 × 10⁻⁷**; ²³²Th has no SF branch
  tabulated; ²⁴⁴Cm SF branching 1.37 × 10⁻⁶ (T½ 18.1 y); ²⁵²Cf 3.09 % (T½ 2.645 y).

### 2.4 Recalled (flagged)
| item | value used | bracket | reliability |
|---|---|---|---|
| ²³⁸U SF prompt-neutron ν̄ | 2.01 | — | likely (Holden–Zucker compilation ~1.99–2.01) |
| P(ν): Terrell Gaussian width | σ = 1.08 | — | likely (Terrell 1957 universal width) |
| Watt parameters ²³⁸U SF | a = 0.988 MeV, b = 2.249 MeV⁻¹ | — | likely (SOURCES-4C convention; same as P013) |
| prompt γ multiplicity and mean energy | 6.5 γ, ⟨E⟩ = 0.9 MeV, exponential spectrum 0.1–7 MeV | multiplicity models Poisson / Gaussian σ = 2.5 with k ≥ 1 or k ≥ 2 | uncertain (²⁵²Cf-like values; ²³⁸U SF data sparse) |
| ²³²Th SF branching | ~10⁻¹¹ | — | likely (negligible) |
| PTFE mass and ²³⁸U(early) | 1000 kg × 10 μBq/kg | 3–100 μBq/kg; mass 0.5–1.5 t | uncertain |
| Ti cryostat mass and ²³⁸U(early) | 2200 kg × 0.5 mBq/kg | 0.1–1.6 mBq/kg (LZ Ti assay UL 1.6 mBq/kg, ²³⁸U_late 0.09) | uncertain |
| R11410 PMTs, ²³⁸U(early) | 625 × 2 mBq | 1–3 mBq/PMT | uncertain |
| cables, feedthroughs, bases, misc. | 0.3 Bq | 0.1–0.6 Bq | uncertain |
| (α,n) yield of the ²³⁸U late chain in PTFE | 1 × 10⁻⁴ n per chain decay | 0.5–2 × 10⁻⁴ | uncertain |
| late/early chain activity ratio in PTFE | 1 | 0.3–3 | assumption |
| geometric fraction of emitted neutrons entering the active LXe | wall 0.5, cryostat 0.35 (0.5 × e^{−5 cm/14 cm}), PMT arrays 0.4, misc. 0.2 | — | estimated |
| per-companion-neutron silent probability p_n | 0.10 | 0.05–0.20 | estimated (LZ: 8 % untagged for TPC-scattering (α,n) n, 20 % for rock n) |
| self-untagged probability of the window neutron u | 0.10 | 0.08–0.20 | estimated (same basis) |
| P(≥ 2 deposits > 3 keV in LXe | neutron enters) | 0.5 | 0.3–0.7 | estimated (λ_el ≈ 16 cm at 2 MeV, ⟨E_R⟩ ≈ 30 keV) |
| P(neutron ends in a tagged capture within 600 μs) | 0.6 | 0.5–0.8 | estimated |
| mass attenuation coefficients (Xe, LAB, Ti, SiO₂, water) 0.1–7 MeV | XCOM-like table in the script | ±20 % | likely |
| photoelectric fraction in Ti / SiO₂ vs energy | Ti 0.55/0.15/0.06/0.02/0.005 at 0.1/0.2/0.3/0.5/1 MeV; SiO₂ ×0.1 | — | uncertain |
| LZ geometry: TPC 72.8 × 145.6 cm, Skin outer radius 79 cm, RFR 20 cm, PMT arrays 15 cm, ICV r = 80 cm, OCV r = 92 cm, Ti 0.8 cm, GdLS side 61 cm, top/bottom 30 cm | — | ±20 % | likely/uncertain |
| ≤ 3 unexplained NR multiple scatters in the MS neutron sample | 3 (1–5) | — | uncertain (tolerance, not a published count) |

## 3. Nuclear data (script §1)

P(ν) from a Gaussian of mean 2.01 and σ = 1.08 discretised on integers (negative tail folded into ν = 0):
P(0…5) = 0.081, 0.237, 0.357, 0.241, 0.073, 0.010 (mean 2.02). P013's recall gives 0.05, 0.25, 0.37, 0.25, 0.07, 0.01 (mean 2.07).
Watt: mean 2.03 MeV; f(E_n > 6.61 MeV) = 1.74 × 10⁻²; **f(E_n > 8.19 MeV) = 5.24 × 10⁻³**; f(> 10 MeV) = 1.29 × 10⁻³ (all reproduce P013).
γ multiplicity models: Poisson(6.5): P(0) = 1.5 × 10⁻³, P(1) = 9.8 × 10⁻³, P(2) = 3.2 × 10⁻²; narrow Gaussian(6.5, 2.5) k ≥ 1: P(1) = 1.5 × 10⁻²,
P(2) = 3.2 × 10⁻² (mean 6.56); k ≥ 2 variant: P(2) = 3.3 × 10⁻². The Poisson model's zero- and one-γ tails are unphysical (both fragments
are left excited after neutron emission and de-excite by γ's), which is why the narrow model is central.

## 4. ²³⁸U inventory and fission rate (script §2)

N_SF = A(²³⁸U) × 5.45 × 10⁻⁷ × 1.9008 × 10⁷ s.

| component | ²³⁸U(early) activity, Bq (bracket) | SF in 220 d (bracket) | p_enter |
|---|---|---|---|
| PTFE (1000 kg × 10 μBq/kg) | 0.010 (0.003–0.10) | 0.10 (0.03–1.0) | 0.50 |
| Ti cryostat (2200 kg × 0.5 mBq/kg) | 1.10 (0.22–3.52) | 11.4 (2.3–36.5) | 0.35 |
| R11410 PMTs (625 × 2 mBq) | 1.25 (0.63–1.88) | 12.9 (6.5–19.4) | 0.40 |
| cables, feedthroughs, bases | 0.30 (0.10–0.60) | 3.1 (1.0–6.2) | 0.20 |
| **total** | **2.66 (0.95–6.09)** | **27.6 (9.8–63)** | |

SF neutrons emitted in 220 d: 55 (20–127); **above 8.19 MeV: 0.29 (0.10–0.67)**. Before any veto or geometry, fewer than one fission
neutron capable of a 248 keV recoil is emitted in the whole exposure.

Route B — (α,n) budget back-calculation (P013 factors, wall geometry): untagged SS ≤ 0.118 → pre-veto SS ≤ 1.48 → ≤ 187 (α,n) neutrons
entering the LXe → ≤ 374 emitted. SF/(α,n) neutron ratio for PTFE = 5.45 × 10⁻⁷ × 2.01 / (Y_αn × R_late/early) = 1.1 × 10⁻² (1.8 × 10⁻³ –
7.3 × 10⁻²), reproducing LZ's "up to two orders of magnitude" and P013's 1 %. This gives ≤ 4.1 SF neutrons, i.e. **≤ 2.0 fissions (0.3–14)
in fluorine-bearing components**; with the MS cross-check normalisation (0.02 SS) 0.35 fissions (0.06–2.3). Route B does not constrain Ti or
PMT fissions, whose (α,n) yields are negligible. Route A's total SF-neutron rate is 0.15 of the route-B (α,n) neutron rate — i.e. "up to two
orders below (α,n)" holds for the *PTFE* fission share but not for the *total* fission rate if the Ti and PMT ²³⁸U activities are as recalled.

Route C — TPC multiple-scatter cap: per fission P(≥ 1 NR multiple scatter in the active LXe) = 1 − Σ_ν P(ν)(1 − p_enter P_MS)^ν = 0.34
(p_enter = 0.4, P_MS = 0.5). Allowing ≤ 3 (1–5) unexplained MS NR events in LZ's MS neutron sample caps **N_SF ≤ 8.7 (2.9–14.5)**
(6.6–13.6 for P_MS 0.7–0.3). Route A's central 27.6 fissions would produce 8.6 TPC multiple scatters, 13.4 fissions with ≥ 1 TPC deposit and 9.8
with ≥ 2 OD/Skin captures in 220 d (§7) — a population LZ can look for; we therefore quote results for route A (central), route C and route B.

## 5. Prompt-γ transport (script §3)

Straight-line sampled Monte Carlo (60 000 γ per location, energies exponential with mean 0.9 MeV truncated to 0.1–7 MeV, isotropic).
Regions (cm): TPC LXe r < 72.8, 0 < z < 145.6; Skin/RFR LXe r < 79, −20 < z < 150 outside the TPC, plus a 15 cm LXe dome below the bottom
PMT array; PMT arrays 15 cm thick (SiO₂-like, ρ = 1 g cm⁻³ effective) above/below the LXe; Ti ICV shell r = 80–80.8 cm (−60 < z < 170) and
OCV r = 92–92.8 with end caps; GdLS side annulus r = 95–156 cm, top/bottom slabs 30 cm; water beyond. Interaction sampling per 0.5 cm
step with log-log interpolated μ(E). Visible: any interaction in LXe (TPC or Skin — a Compton electron of even a few keV exceeds 3 keV /
2.5 phd); an OD interaction with probability v_OD = 0.9 (deposit above 4.5 phd ≈ a few tens of keV; 0.7–1.0 varied). Invisible materials (Ti, PMT
arrays): photoelectric absorption (recalled fraction) is silent, otherwise the γ Compton-scatters and continues along its line with an energy
sampled uniformly between the backscatter energy and E (direction change neglected); water is silent. A γ is "silent" if it never produces a
visible interaction. Sources: PTFE wall at r = 73.4 cm; ICV wall (r = 80.4); PMT bodies uniform over the array discs and thickness, top or
bottom; OCV wall (proxy for cables/conduits in the vacuum space).

| source | ⟨p_γ silent⟩ (± MC) | E_γ > 1 MeV | E_γ < 0.5 MeV | v_OD = 0.7 / 1.0 | PMT arrays half visible | all Ti/PMT interactions silent (over-conservative) |
|---|---|---|---|---|---|---|
| PTFE wall | 0.0036 ± 0.0002 | 0.0069 | 0.0012 | 0.0063 / 0.0030 | 0.0034 | 0.030 |
| Ti inner cryostat | 0.134 ± 0.001 | 0.069 | 0.235 | 0.144 / 0.134 | 0.127 | 0.459 |
| PMT arrays | 0.148 ± 0.001 | 0.087 | 0.231 | 0.155 / 0.147 | 0.033 | 0.690 |
| outer cryostat / cables | 0.142 ± 0.001 | 0.076 | 0.244 | — | — | — |

A wall γ is almost never silent: it must cross ≥ 6 cm of Skin LXe (λ ≈ 6 cm at 1 MeV) or the TPC itself. For cryostat and PMT sources the
silent fraction is dominated by sub-0.5 MeV γ's photo-absorbed in Ti/Kovar/glass and by γ's leaving through the 30 cm top/bottom GdLS
(τ ≈ 1.8 at 1 MeV). P013's uniform p_γ = 0.15 is therefore right for the cryostat and PMTs and ~40× too generous for the PTFE wall.

f_γ = Σ_k P(k) p_γ^k (independent γ's, mean p; angular correlations ignored) — `P063_f_gamma.csv`:

| source | Poisson(6.5) | narrow k ≥ 1 (central) | narrow k ≥ 2 |
|---|---|---|---|
| PTFE wall | 1.5 × 10⁻³ | 5.3 × 10⁻⁵ | 4.3 × 10⁻⁷ |
| Ti cryostat | 3.6 × 10⁻³ | 2.7 × 10⁻³ | 7.8 × 10⁻⁴ |
| PMT arrays | 3.9 × 10⁻³ | 3.1 × 10⁻³ | 9.7 × 10⁻⁴ |
| outer cryostat / cables | 3.8 × 10⁻³ | 3.0 × 10⁻³ | 8.9 × 10⁻⁴ |
| P013 (p = 0.15, Poisson 7) | 2.6 × 10⁻³ | | |

For the wall the Poisson result is set entirely by the unphysical P(k ≤ 1) tail (1.1 %); with k ≥ 1 enforced it drops 30×.

## 6. Lone-recoil probability and expected counts (script §4)

Per emitted neutron, the other ν − 1 must be silent: f_n = Σ_ν P(ν) ν p_n^{ν−1} / ν̄ = **0.156** (p_n = 0.10; 0.136 at 0.05, 0.204 at 0.20; P013's
inputs reproduce its 0.148). Note P(ν = 1) = 0.237: 12 % of emitted neutrons have no sibling, so f_n cannot fall below ≈ 0.12 whatever p_n is.
Per fission:

P_lone = ν̄ · p_enter · p_win · u · f_n · f_γ,   p_win = 1.61 × 10⁻⁶, u = 0.10.

| component | N_SF | p_γ | f_γ (narrow) | P_lone per SF | N_lone (narrow) | N_lone (Poisson) | no-companion bound |
|---|---|---|---|---|---|---|---|
| PTFE | 0.10 | 0.0036 | 5.3 × 10⁻⁵ | 1.4 × 10⁻¹² | 1.4 × 10⁻¹³ | 4.0 × 10⁻¹² | 1.7 × 10⁻⁸ |
| Ti cryostat | 11.4 | 0.134 | 2.7 × 10⁻³ | 4.8 × 10⁻¹¹ | 5.5 × 10⁻¹⁰ | 7.3 × 10⁻¹⁰ | 1.3 × 10⁻⁶ |
| PMTs | 12.9 | 0.148 | 3.1 × 10⁻³ | 6.3 × 10⁻¹¹ | 8.2 × 10⁻¹⁰ | 1.0 × 10⁻⁹ | 1.7 × 10⁻⁶ |
| cables/misc. | 3.1 | 0.148 | 3.1 × 10⁻³ | 3.2 × 10⁻¹¹ | 9.8 × 10⁻¹¹ | 1.2 × 10⁻¹⁰ | 2.0 × 10⁻⁷ |
| **total** | **27.6** | | | | **1.47 × 10⁻⁹** | 1.88 × 10⁻⁹ | 3.2 × 10⁻⁶ |

Effective silent-companion factor (total / no-companion bound) = 4.6 × 10⁻⁴, coincidentally close to P013's 3.8 × 10⁻⁴; the 16× larger central
count relative to P013's 9.3 × 10⁻¹¹ comes from the fission rate (27.6 vs ≈ 2 fissions), not from the companion physics.

Sensitivity (`P063_sensitivity.csv`; ratio to central 1.47 × 10⁻⁹): Poisson γ ×1.28; narrow k ≥ 2 ×0.30; p_n 0.20/0.05 ×1.30/0.87; u = 0.20 ×2.0;
v_OD 0.7/1.0 ×1.08/1.00; all Ti/PMT-array interactions silent ×30 (over-conservative bracket, 4.4 × 10⁻⁸); PMT arrays half visible ×0.45;
per-γ silent floor 0.05/0.15 ×1.00/1.08; p_win max/min ×1.27/0.39; Ti and PMTs at assay ULs ×2.2; inventory lower/upper ×0.36/2.3;
route C cap (8.7 SF) ×0.32 → 4.7 × 10⁻¹⁰; route B PTFE-only (2.0 SF) ×0.074 → 1.1 × 10⁻¹⁰ (≈ P013).
**Stacked high** (upper inventory 63 SF, Poisson γ's, p_γ ≥ 0.15 everywhere, p_n = 0.2, u = 0.2, p_win max): **1.5 × 10⁻⁸**.
**Stacked low**: 4.3 × 10⁻¹¹. The range 4 × 10⁻¹¹ – 1.5 × 10⁻⁸ (4 × 10⁻⁸ with the over-conservative γ bookkeeping) is ≥ 7.5 orders of magnitude
below one event.

Not included, and irrelevant at this level: ²³²Th SF (branching ~10⁻¹¹, ≤ 10⁻⁵ of the ²³⁸U rate); ²⁴⁴Cm/²⁵²Cf contamination (no such sources
were used by LZ; a hypothetical 1 mBq ²⁵²Cf speck would give 5.9 × 10² SF in 220 d and hundreds of tagged bursts — trivially excluded by the veto
samples); ²³⁵U (no SF branch; ≤ 4.6 % of ²³⁸U activity).

## 7. What SF bursts look like, and the check LZ can do (script §5)

Per fission (route A weights): P(≥ 1 visible prompt γ) ≥ 0.997 for every location; P(≥ 1 TPC deposit) = 0.31–0.62; P(≥ 1 tagged capture) = 0.75;
P(≥ 2 captures) = 0.35. In 220 d the recalled inventory predicts 13.4 fissions with a TPC NR deposit (nearly all prompt-tagged by γ's and
followed by ≥ 1 capture), 8.6 TPC multiple scatters, 9.8 fissions with ≥ 2 OD/Skin captures within 600 μs. For comparison, (α,n) at the fit UL
gives ≈ 94 pre-veto MS events (187 entering neutrons × 0.5), and ≈ 16 at the MS cross-check normalisation. The SF signature is therefore
distinct: prompt γ light in OD/Skin + TPC NR + ≥ 2 delayed captures; LZ's MS neutron sample and the OD prompt-multiplicity / delayed-capture-
multiplicity distributions can bound N_SF directly (route C). A published count of MS NR events with two or more delayed OD captures would
turn our recalled inventory into a measurement.

## 8. Discussion

1. The only way SF can give a lone high-energy recoil is a ν = 1 fission (12 % of emitted neutrons) or a ν = 2 fission whose sibling escapes
   (p_n ≈ 0.1), with all γ's silent (≤ 3 × 10⁻³ from the cryostat/PMTs, 5 × 10⁻⁵ from the wall) and the neutron itself above 8.19 MeV
   (5 × 10⁻³ of the Watt spectrum), single-scattering in the 200–270 keV window (black-disk pattern; P013) and escaping untagged: the product
   is 10⁻¹² – 10⁻¹⁰ per fission against 10–60 fissions per run.
2. LZ's "two orders of magnitude" is a statement about neutron *rates in fluorine-bearing materials*; if the Ti and PMT ²³⁸U activities are
   as recalled, the *total* fission rate is ~15 % of the (α,n) neutron rate, but it enters the science sample only through the multiplicity
   penalty, so the conclusion is unchanged.
3. Compared with P013, the silent factor is now location-resolved (the wall value is 50× smaller, the cryostat/PMT values 1.2× larger) and the
   fission rate 14× larger; the two changes partly cancel, leaving a central 1.5 × 10⁻⁹.
4. The dominant uncertainties are the recalled ²³⁸U(early) activities (Ti UL vs value) and the γ-multiplicity tail; neither can lift the
   expectation above 10⁻⁷.

## 9. Failed or abandoned approaches
- First γ-transport bookkeeping treated every interaction in Ti / PMT arrays as a silent absorption; it gave p_γ = 0.46 (cryostat) and 0.63–0.69
  (PMTs) and a total of 3.2 × 10⁻⁸. Replaced by the sampled Compton-continuation model; kept as the "over-conservative" bracket.
- A deterministic weight-propagation scheme (survival weights per step) could not carry the energy degradation after Compton scatters; replaced
  by sampled interactions.
- No attempt to model the fission-fragment surface-escape channel beyond the ≲ 2 × 10⁻³ geometric fraction.

## 10. Figures
- `figures/P063_fig1_silent_gamma.png` — (a) per-γ silent probability by source location vs P013's uniform 0.15; (b) f_γ for the three
  multiplicity models vs P013's 2.6 × 10⁻³.
- `figures/P063_fig2_expected.png` — (a) expected lone 200–270 keV recoils in 220 d by component (Poisson vs narrow γ multiplicity);
  (b) scenario bars vs P013's 9 × 10⁻¹¹ and the one-event line.

## 11. References
1. LZ Collaboration, arXiv:2609.02823 (2026) — Table I, l. 141–147, 178, 787–798.
2. B. E. Watt, Phys. Rev. 87, 1037 (1952) — fission neutron spectrum.
3. J. Terrell, Phys. Rev. 108, 783 (1957) — Gaussian neutron-multiplicity distribution.
4. W. B. Wilson et al., SOURCES-4C, LA-UR-02-1839 (2002) — (α,n) and SF source terms.
5. D. S. Akerib et al. (LZ), Eur. Phys. J. C 80, 1044 (2020) — LZ radioassay programme (material activities recalled from memory).
6. D. S. Akerib et al. (LZ), Astropart. Phys. 96, 1 (2018) — titanium radioassay (²³⁸U_e < 1.6 mBq/kg recalled).
7. A. Malins and T. Lemoine, radioactivedecay, JOSS 7, 3318 (2022) — ICRP-107 decay data.
8. Corpus: P013 (transport factors, silent factor), P049 (capture timing, untagged fraction), P004, P029.

## 12. Tools and provenance (mirrors `output/provenance/P063.json`)
- Agent tools: Read (PAPER_GUIDE, dossier, P013/P049/P029/P004 papers, P013_results.json, fission_silent_factor.csv, tex l. 136–245 and
  780–805, figures), Bash (ledger/tex greps, radioactivedecay query, script runs), Write/Edit (script, details, provenance, paper),
  Skill dataviz (palette only; JS validator skipped).
- Software: python 3.12.13; numpy 2.5.3 (default_rng seed 63, vectorised region lookup, interp); scipy 1.18.1 (integrate.quad, stats.norm,
  stats.poisson); pandas 3.0.5; matplotlib 3.11.2 (Agg); radioactivedecay 0.6.1 (Nuclide.branching_fractions/decay_modes/half_life);
  common/lzcommon.py (LZ dict). Hand derivations: f_n, f_γ, P_lone, route B/C algebra.
- Recalled items: 20 (table §2.4). Datasets: none. Data requests: none.
