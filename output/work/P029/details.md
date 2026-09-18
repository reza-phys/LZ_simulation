# P029 — Activation products eight days after AmBe: activities on 16 June 2023 and which decays could produce an MSSI-like topology

Research record for paper P029 (simulated arXiv date 2026-09-08; physics.ins-det, cross-list nucl-ex; BKG).
Script: `output/code/P029_activation.py` (run from the simulation root with `.venv/bin/python`; console log in
`output/work/P029/run_log.txt`). All tables referenced below are in `output/work/P029/`; the master results file is
`P029_results.json`.

## 1. Motivation and framework

The LZ candidate (S1c = 540.1 phd, S2c = 9268 phd, 16 June 2023 21:22:39 UTC, 26.4 cm above the cathode, 26.9 cm from
the true wall) was recorded 8.4 days after an AmBe neutron calibration (8 June 2023; "one of three" NR calibrations in
the run) and 25 min after a ⁵⁷Co deployment (LZ Discussion). Neutron captures on xenon during calibrations populate the
LXe with ¹²⁵Xe, ¹²⁷Xe, ¹²⁹ᵐXe, ¹³¹ᵐXe, ¹³³Xe, ¹³³ᵐXe, ¹³⁵Xe, ¹³⁷Xe and the daughter ¹²⁵I. LZ's own model contains
¹²⁷Xe + ¹²⁵Xe (1.5 ± 0.3 ROI events) and ¹²⁵I (8.9 ± 2.7), cuts its tritium/¹⁴C calibration data above S1c = 500 phd
"to avoid ¹³³Xe contamination from activation", and states in the supplement that ¹²⁴Xe, ¹²⁷Xe, ¹³³Xe and ¹²⁵I were
simulated for MSSI and found subdominant, adding that "low energy (≲ 1.5 MeV) γ-rays in general do not penetrate into
the analysis volumes without depositing more than 100 keV".

Question. Which activation products were present on 16 June, at what activities (normalised through LZ's Table I), which
decay schemes contain the two ingredients of a wall/RFR MSSI at the event position — a **12 ± 2 keV** deposit at the
vertex (LZ's MSSI decomposition: 12 ± 2 keV with S1c = 69 ± 17 phd, plus a 471 phd S1-only pulse, i.e. 77 ± 7 keV in the
wall dead layer or 204 (+65 −38) keV in the RFR) and a separate γ-ray able to travel ≳ 20 cm — and what is the expected
number of such topologies compared with LZ's MSSI model (4.9 × 10⁻³ in the ROI; 1.7 × 10⁻⁴ in the event's neighbourhood
per P004) and with the one observed event?

Three MSSI-like routes are distinguished:
- **A (correlated, forward):** activation decay at the vertex deposits 12 ± 2 keV locally; one of its γ-rays travels
  ≥ 20 cm to a charge-dead region and deposits 77 ± 7 keV (wall) or 166–269 keV (RFR) there.
- **B (correlated, reverse):** activation decay in a dead region (S1-only); its γ travels ≥ 20 cm into the FV, Compton
  scatters depositing 10–14 keV at the vertex, and the scattered γ escapes the active volume (≥ 26.4 cm) without a
  second S2.
- **C (uncorrelated):** an activation decay in a dead region produces an S1-only pulse of ≈ 471 phd; an unrelated
  10–14 keV ER in the FV occurs within the S1 merging time Δt; the two S1s merge.

## 2. Inputs

| Input | Value | Source |
|---|---|---|
| Event time/position | 16 June 2023 21:22:39 UTC; z = 26.4 cm above cathode; 26.9 cm from true wall | LZ paper (Results), `lz.LZ` |
| AmBe date; three NR calibrations; ⁵⁷Co 25 min before | 8 June 2023 | LZ Discussion (tex l. 300–302) |
| Table I: ¹²⁵I 8.9 ± 2.7; ¹²⁷Xe + ¹²⁵Xe 1.5 ± 0.3; total 1710 observed | 220 d × 4.71 t | LZ Table I (l. 224–242) |
| ¹²⁵I effective half-life 3.6 ± 0.2 d; ⁸³ᵐKr injections, resume after ≈ 13 half-lives | | LZ Data Analysis (l. 168–171) |
| Fig. 4 caption: features above 25 keVee from ¹²⁵I and ¹³³Xe; Fig. 2 caption: tritium/¹⁴C cut above S1c = 500 phd (¹³³Xe) | | tex l. 104, 155 |
| MSSI decomposition 12 ± 2 keV / 69 ± 17 phd / 471 phd / 77 ± 7 keV (wall) / 204 (+65 −38) keV (RFR) | | supplement l. 733–735 |
| Supplement: ¹²⁴Xe, ¹²⁷Xe, ¹³³Xe, ¹²⁵I simulated and subdominant for MSSI; ≲ 1.5 MeV γ do not penetrate without > 100 keV deposit | | l. 719 |
| Charge-dead volume 0.3 % of active; RFR 13.75 cm deep; MSSI 4.9 ± 4.9 × 10⁻³ | | l. 186, 196, 236 |
| WS ROI: S1c 3–600 phd, S2c 10^2.75–10^4.15 phd; g₁ = 0.110, g₂ = 34.5 | | l. 114; `lz.LZ` |
| Half-lives, decay modes, progeny for all nuclides | ICRP-107 via `radioactivedecay` 0.6.1 | `decay_data_icrp107.csv` |
| Xe isotopic abundances | `periodictable` 2.1.0 | `production_relative.csv` |
| NEST-LZ ER yields (Table S3) | `lz.nest_er_yields` (nestpy 2.1.1) | |
| P010: EC charge suppression f = 0.20 reproducing LZ's 21 ¹²⁴Xe ROI events; width k·σ_p = 0.5 × 0.068; α = 0.182 | corpus | P010.md, work/P010/details.md |
| P004: neighbourhood MSSI 1.7 × 10⁻⁴; attenuation lengths 0.30 (122 keV) … 9.8 cm (2615 keV) | corpus | P004.md |

Recalled knowledge (flagged; also in `provenance/P029.json`):
1. Emission schemes (likely): ¹²⁵Xe EC, γ 188.4 (54 %), 243.4 (30 %), 55.0 (6.8 %), I K-capture ≈ 80 % (K binding 33.2 keV),
   L ≈ 17 % (4.9 keV); ¹²⁵I EC to the 35.5 keV level of ¹²⁵Te (γ 6.7 %, IC 93 %), K/L/M capture 80/17/3 % → 67.3/40.4/36.5 keV
   total; ¹²⁷Xe EC, γ 202.9 (68.7 %), 172.1 (25.7 %), 375.0 (17.3 %), 145.3 (4.3 %), 57.6 (1.2 %), K/L 83/14 %; ¹²⁹ᵐXe IT
   236.1 keV = 196.6 + 39.6 cascade, γ 4.6 %/7.5 %, rest IC; ¹³¹ᵐXe IT 163.9 keV, γ 1.95 %; ¹³³Xe β⁻ (endpoint 346 keV) to the
   81.0 keV level (γ 38 %, IC 62 %); ¹³³ᵐXe IT 233.2 keV, γ 10 %; ¹³⁵Xe β⁻ (endpoint 915 keV) + γ 249.8 (90 %), 608.2 (2.9 %);
   ¹³⁵ᵐXe IT 526.6 keV (γ 81 %); ¹³⁷Xe β⁻ Q = 4.17 MeV, γ 455.5 (31 %); ¹³⁷Cs β⁻ 514 keV, γ 661.7 (85 %); ⁸³ᵐKr 32.1 + 9.4 keV IC.
2. Thermal (n,γ) cross-sections (uncertain, ± 30 %; isomer partials ×2): ¹²⁴Xe 165 b, ¹²⁶Xe 3.5 b, ¹²⁸Xe 8 b (→¹²⁹ᵐXe 0.48 b),
   ¹²⁹Xe 21 b, ¹³⁰Xe 26 b (→¹³¹ᵐXe 0.45 b), ¹³¹Xe 85 b, ¹³²Xe 0.45 b (→¹³³Xe 0.40, →¹³³ᵐXe 0.05), ¹³⁴Xe 0.265 b (→¹³⁵Xe 0.26,
   →¹³⁵ᵐXe 0.003), ¹³⁶Xe 0.26 b. Check: natural-Xe thermal σ = 25.1 b (recalled ≈ 24 b).
3. Fast (~1 MeV) capture set (uncertain ×2–3): 50, 40, 30, 25, 15, 8, 1.5 mb for ¹²⁴…¹³⁶Xe, isomer shares 40/50/10/10 %.
4. Xenon mass-attenuation coefficients (likely ± 30 %): iodine-proxy XCOM-like table scaled ×1.04, K-edge 34.56 keV
   (8.0 cm²/g at 30 keV; 20 at 40; 3.4 at 80; 1.15 at 122 [LZ: < 4 mm ✓]; 0.36 at 200; 0.165 at 300; 0.108 at 400;
   0.069 at 662 keV); ρ = 2.9 g cm⁻³. These are 2–4× shorter than the lengths handed to us in the assignment
   (0.07 cm @ 40, 0.25 @ 80, 1.3 @ 164, 2.3 @ 203, 3.5 @ 250, 5.5 cm @ 375 keV), which are carried as a longer bracket.
5. NEST exciton-to-ion ratio α = 0.067366 + 0.039693 ρ (likely); Klein–Nishina (certain); β allowed shape (certain).
6. LZ geometry: total circulated xenon 10 t, active 7 t, TPC radius 72.8 cm (likely); WS data resume ≈ 1 d after a
   calibration, deployment ≈ 1 d (assumptions); ¹³⁷Cs removed by the getter with the same 3.6 d effective half-life as ¹²⁵I
   (assumption, bracketed by "retained").

## 3. Method and derivations

### 3.1 Production and Bateman decay
Captures on isotope i during a constant 1-day irradiation: C_i ∝ a_i σ_i (a_i abundance). Atoms at the end of irradiation
N_i(0) = C_i (1 − e^{−λ_i T})/(λ_i T). Decay to the event (t_ev = 8.39 d: 8 June 12:00 → 16 June 21:22) with
`radioactivedecay.Inventory.decay` (Bateman with progeny; ¹³³ᵐXe → ¹³³Xe, ¹³⁵ᵐXe → ¹³⁵Xe, ¹³⁷Xe → ¹³⁷Cs).
¹²⁵I is treated separately with purification removal: dN_Xe/dt = P − λ_Xe N_Xe; dN_I/dt = λ_Xe N_Xe − λ_eff N_I with
λ_eff = ln2/3.6 d, decay rate in LXe λ_dec N_I (λ_dec = ln2/59.4 d). Integrated numerically (exact one-step updates,
dt = 0.005 d). Fraction of ¹²⁵I atoms that decay in the LXe: λ_dec/λ_eff = 6.06 % (numerical 6.061 %, check).
Profile: peak 1.61 d after the end of irradiation; rate at +8.39 d = 0.337 of peak (P010 with prompt production: 0.40 ✓);
fraction of ¹²⁵I decays occurring after WS resumption at +0.5/1/2 d: 93.3/87.5/74.8 %.

### 3.2 Absolute normalisation from Table I
ROI acceptance of a ¹²⁵I decay, ε: each mode (67.3/40.4/36.5 keV, 80/17/3 %) is placed with NEST-LZ β yields
(`lz.nest_er_yields`), mean electron count reduced by the EC charge suppression f (P010: f = 0.20 reproduces LZ's 21 ¹²⁴Xe
ROI events), width σ(N_e) = 0.5 × 0.068 × N_i (P010, calibrated on the digitised Fig. 4 band); ROI requires
N_e < 10^4.15/34.5 = 409 electrons (S1c is always < 600 phd for these lines). `I125_ROI_acceptance.csv`:

| f | P(67.3 in ROI) | P(40.4) | P(36.5) | ε (all modes) |
|---|---|---|---|---|
| 0.0 | 4.6e-8 | 2.1e-4 | 8.2e-4 | 6.1e-5 |
| 0.1 | 3.1e-6 | 3.5e-3 | 1.0e-2 | 9.1e-4 |
| **0.2** | 1.1e-4 | 3.1e-2 | 7.1e-2 | **7.4e-3** |
| 0.3 | 2.0e-3 | 0.148 | 0.264 | 3.5e-2 |

Consistency: the ER-band median (NEST-LZ) crosses the ROI edge log₁₀S2c = 4.15 at E_ee = 20.2 keV, so the 1710-event
science sample is a ≲ 20 keVee β sample plus tails; a 12 keV ER gives S1c = 67.4 phd, log₁₀S2c = 3.985 (LZ: 69 ± 17 phd ✓);
an 81 keV ER gives S1c = 499 phd (explains LZ's ¹³³Xe cut at 500 phd ✓).

¹²⁵I decays in the FV inside WS data over the run: N = 8.9/ε = **1200** (f = 0.2; 260 for f = 0.3; 9800 for f = 0.1).
Per calibration (N_cal = 3), including decays before WS resumed (÷ 0.875) and the 6.06 % branch, scaled from the FV
(4.71 t) to the 10 t inventory: **C₁₂₄ = 1.6 × 10⁴ ¹²⁴Xe captures per calibration** (central); 4.8 × 10⁴ if all 8.9 events
came from the 8 June AmBe alone ("upper"); 3.4 × 10³ for f = 0.3. (`normalisation_from_TableI.csv`)

### 3.3 Capture-spectrum bracket and the ¹²⁷Xe cross-check
Relative captures C_i/C₁₂₄ (`production_relative.csv`):

| product | thermal | fast | **intermediate** (Table-I anchored, w = 0.665) |
|---|---|---|---|
| ¹²⁷Xe | 0.020 | 0.75 | 0.22 |
| ¹²⁹ᵐXe | 0.058 | 4.6 | 1.10 |
| ¹³¹ᵐXe | 0.117 | 9.6 | 2.36 |
| ¹³³Xe | 0.69 | 84 | 15.8 |
| ¹³³ᵐXe | 0.086 | 9.2 | 1.82 |
| ¹³⁵Xe | 0.173 | 17.8 | 3.5 |
| ¹³⁷Xe | 0.147 | 2.7 | 1.04 |

The AmBe spectrum is fast; captures in the TPC occur over an unknown epithermal/fast mix, so the two sets bracket the
truth. Anchor: a ¹²⁷Xe decay uniform in the FV is a single-scatter ROI event only if its γ leaves the active LXe without
interacting (≥ 8 cm stand-off: hemisphere-averaged slab transmission 1.3 × 10⁻³ summed over the 145–375 keV lines) and the
local deposit is in the ROI (L capture 4.9 keV, 14 %: always; K 33.2 keV, 83 %: P = 0.145 with f = 0.2) →
ε₁₂₇ ≈ 3.3 × 10⁻⁴ (×3 uncertain). LZ's 1.5 ¹²⁷Xe(+¹²⁵Xe) events then imply ≈ 4500 ¹²⁷Xe decays in the FV over the run,
versus 400 (thermal) and 15 100 (fast) from the ¹²⁵I normalisation: the implied ¹²⁷Xe/¹²⁵Xe production ratio is 0.22.
The "intermediate" set is the geometric interpolation thermal^(1−w) fast^w with w = 0.665 fixed by this ratio; it is used
as the central case, the two pure sets as brackets.

### 3.4 Activities and decays on 16 June (`activities_16June.csv`)
Activity in the 10 t inventory at 21:22 UTC on 16 June and decays in the FV (4.71 t) and RFR (0.66 t) during ±1 h,
central normalisation, three capture sets (thermal / intermediate / fast):

| nuclide | T½ | remaining | A (Bq, 10 t) | decays in FV ±1 h | decays in RFR ±1 h |
|---|---|---|---|---|---|
| ¹²⁵Xe | 16.9 h | 2.6e-4 | 3.0e-5 | 0.10 | 0.014 |
| ¹²⁵I | 59.4 d (3.6 d eff.) | — | 4.9e-4 | 1.65 | 0.23 |
| ¹²⁷Xe | 36.4 d | 0.85 | 5.9e-5 / 6.6e-4 / 2.2e-3 | 0.20 / 2.2 / 7.6 | 0.03 / 0.32 / 1.1 |
| ¹²⁹ᵐXe | 8.88 d | 0.52 | 4.2e-4 / 8.0e-3 / 3.5e-2 | 1.4 / 27 / 118 | 0.20 / 3.8 / 17 |
| ¹³¹ᵐXe | 11.84 d | 0.61 | 7.5e-4 / 1.5e-2 / 6.9e-2 | 2.6 / 52 / 234 | 0.36 / 7.3 / 33 |
| ¹³³Xe | 5.24 d | 0.38 | 6.0e-3 / 0.136 / 0.66 | 20 / 462 / 2230 | 2.9 / 65 / 315 |
| ¹³³ᵐXe | 2.19 d | 0.070 | 3.0e-4 / 6.4e-3 / 3.0e-2 | 1.0 / 22 / 102 | 0.15 / 3.1 / 14 |
| ¹³⁵Xe | 9.14 h | 2.3e-7 | 6e-9 / 1.3e-7 / 5.8e-7 | 2e-5 / 4e-4 / 2e-3 | — |
| ¹³⁷Xe, ¹³⁵ᵐXe | 3.8 min, 15 min | 0 | 0 | 0 | 0 |
| ¹³⁷Cs (retained) | 30 yr | 1 | 6.6e-9 / 4.6e-8 / 1.2e-7 | 2e-5 / 1.6e-4 / 4e-4 | — |
| **total** | | | | **27 / 570 / 2700** | **3.9 / 80 / 380** |

¹²⁷Xe via the Table-I route (1.5 events, ε₁₂₇), including two earlier calibrations at −30 and −60 d: 1.25 × 10⁻³ Bq,
4.2 decays in the FV in ±1 h. "Upper" normalisation (all 8.9 ¹²⁵I from the 8 June AmBe) multiplies everything except
¹²⁵I by 3. Dead-region (RFR + wall layer) S1-only rate from activation at the event time: 5.5 × 10⁻⁴ / 1.1 × 10⁻² /
5.4 × 10⁻² Hz.

### 3.5 γ transport in LXe (`gamma_transport.csv`, Fig. 2)
λ = 1/(μ/ρ · ρ) from the recalled table; P(d) = exp(−d/λ) for d = 20 cm (assignment threshold), 26.9 cm (event–wall) and
26.4 cm (event–cathode). Klein–Nishina fraction of Compton scatters with T ∈ [10, 14] keV, f_KN, and the Compton share of
the total attenuation give the reverse-topology factor per γ, p_B = e^{−20/λ(E)} · (μ_C/μ) · f_KN · e^{−26.4/λ(E−12)}.

| E (keV) | source | λ (cm) | P(20 cm) | P(26.9 cm) | λ_bracket (cm) | P(20 cm)_bracket | f_KN(10–14 keV) | p_B |
|---|---|---|---|---|---|---|---|---|
| 35.5 | ¹²⁵I | 0.012 | 0 | 0 | — | — | 0 | 0 |
| 39.6 | ¹²⁹ᵐXe | 0.017 | 0 | 0 | — | — | 0 | 0 |
| 81.0 | ¹³³Xe | 0.105 | 9e-84 | 2e-112 | 0.26 | 2e-34 | 0.16 | 0 |
| 163.9 | ¹³¹ᵐXe | 0.62 | 8e-15 | 1e-19 | 1.30 | 2.0e-7 | 0.069 | 1e-38 |
| 172.1 | ¹²⁷Xe | 0.69 | 2e-13 | 1e-17 | 1.48 | 1.3e-6 | 0.065 | 8e-35 |
| 188.4 | ¹²⁵Xe | 0.84 | 4e-11 | 1e-14 | 1.88 | 2.5e-5 | 0.058 | 1e-28 |
| 196.6 | ¹²⁹ᵐXe | 0.92 | 4e-10 | 2e-13 | 2.11 | 7.7e-5 | 0.055 | 3e-26 |
| 202.9 | ¹²⁷Xe | 0.99 | 1.5e-9 | 1.4e-12 | 2.30 | 1.7e-4 | 0.052 | 1e-24 |
| 233.2 | ¹³³ᵐXe | 1.30 | 2.2e-7 | 1.1e-9 | 3.04 | 1.4e-3 | 0.043 | 6e-19 |
| 243.4 | ¹²⁵Xe | 1.42 | 7.7e-7 | 6.0e-9 | 3.32 | 2.4e-3 | 0.041 | 1e-17 |
| 249.8 | ¹³⁵Xe | 1.50 | 1.6e-6 | 1.6e-8 | 3.49 | 3.3e-3 | 0.039 | 9e-17 |
| 375.0 | ¹²⁷Xe | 2.90 | 1.0e-3 | 9.5e-5 | 5.50 | 2.6e-2 | 0.021 | 1.1e-9 |
| 455.5 | ¹³⁷Xe | 3.67 | 4.3e-3 | 6.6e-4 | — | — | 0.016 | 3e-8 |
| 526.6 | ¹³⁵ᵐXe | 4.22 | 8.7e-3 | 1.7e-3 | — | — | 0.013 | 1.6e-7 |
| 661.7 | ¹³⁷Cs | 5.0 | 1.8e-2 | 4.6e-3 | — | — | 0.009 | 7e-7 |

P(20 cm) reaches 10⁻³ at 375 keV (recalled attenuation) or at 228 keV (longer bracket). With the recalled table every
activation line ≤ 375 keV has P(20 cm) ≤ 1.0 × 10⁻³ and P(26.9 cm) ≤ 9.5 × 10⁻⁵; with the bracket the 375 keV line has
2.6 × 10⁻² (7.5 × 10⁻³ at 26.9 cm) and the 233–250 keV lines (1.4–3.3) × 10⁻³. The paper's "≲ 1.5 MeV γ-rays do not
penetrate into the analysis volumes without depositing > 100 keV" is quantified: penetration ≥ 20 cm is < 2 % even at
662 keV.

### 3.6 Route A: the vertex must deposit 12 ± 2 keV (`topologyA_correlated.csv`)
Local deposits are quantised by atomic physics: EC daughters (¹²⁵Xe → I, ¹²⁷Xe → I) deposit the K-shell binding 33.2 keV
(K capture, 80–83 %) or 4.9 keV (L, 14–17 %), the daughter K X-rays (28–33 keV, λ ≈ 0.4 mm) being absorbed at the site;
¹²⁵I deposits 67.3/40.4/36.5 keV; the isomers deposit their full 236.1/163.9/233.2 keV within a millimetre (IC electrons
+ X-rays). None is in 10–14 keV. NEST-LZ S2c for the candidate local deposits: 4.9 keV → 5320 phd (4260 with f = 0.2);
12 keV → 9660; 33.2 keV → 20 900 (16 700 with f = 0.2); observed 9268 phd. The only 10–14 keV local deposits are β
continua: ¹³³Xe (fraction 1.04 % of the spectrum in 10–14 keV, but its 81 keV γ has λ = 1 mm), ¹³⁵Xe (0.18 %, γ 608 keV
2.9 %), ¹³⁷Xe (6 × 10⁻⁵), ¹³⁷Cs (0.53 %, γ 662 keV 85 %). Per-decay route-A factor f_vertex · p_γ · P(20 cm):
¹³⁵Xe 7.4 × 10⁻⁷, ¹³⁷Cs 8.2 × 10⁻⁵, ¹³⁷Xe 7.6 × 10⁻⁸, ¹³³Xe 4 × 10⁻⁸⁶, all EC/IT nuclides exactly 0.

Expected route-A events over the run: Σ_i N_i(FV decays in WS data, three calibrations) × factor_i × F_DEAD, with
F_DEAD ≈ 0.03 the probability that the far γ deposits an energy compatible with 471 phd in a dead region (RFR: solid angle
≈ 0.35 × interaction ≈ 0.94 × KN fraction 166–269 keV ≈ 0.2 × escape of the scattered γ ≈ 0.5 ≈ 0.033; wall 3 mm layer
≈ 1.5 × 10⁻³). ¹³⁷Cs: central "removed" (3.6 d effective, like ¹²⁵I) → 3.3 × 10⁻⁴ of atoms decay in the run; "retained"
→ 1.4 × 10⁻². Results (`P029_results.json: topologyA_expected_run`):

| capture set / normalisation | Cs removed | Cs retained |
|---|---|---|
| thermal / central | 9.1e-6 | 1.2e-4 |
| **intermediate / central** | **1.5e-4** | 9.3e-4 |
| fast / central | 6.4e-4 | 2.7e-3 |
| fast / upper (single calibration) | 1.9e-3 | 8.2e-3 |

Leading channel: ¹³⁵Xe β (10–14 keV) + 608 keV γ, i.e. the first day after each calibration (¹³⁵Xe T½ = 9.1 h; only
16 % of its decays fall after WS resumption). Excluding ¹³⁷Cs altogether the maximum is 1.8 × 10⁻³. **At the event time**
(±1 h) the route-A expectation is 5.5 × 10⁻¹¹ / 4.0 × 10⁻¹⁰ / 1.1 × 10⁻⁹ (thermal/intermediate/fast), ≤ 3.2 × 10⁻⁹ with
the upper normalisation, because ¹³⁵Xe has decayed to 2.3 × 10⁻⁷ of its initial number by +8.4 d.

### 3.7 Route B (reverse)
p_B per γ from §3.5; the best line is ¹²⁷Xe 375 keV with p_B = 1.1 × 10⁻⁹ (2 × 10⁻⁵ with the longer bracket, taking
λ = 5.5 cm for both legs), times the S1-only requirement in the dead region (K capture 33.2 keV alone gives S1c ≈ 200 phd,
not 471; 33.2 + 172.1 = 205 keV in the RFR matches the RFR window only for the 375-then-203 cascade, which is not the
375 keV γ). Multiplying by the ¹²⁷Xe decays in the dead regions over the run (thermal 26, intermediate 300, fast 1000;
Table-I route ≈ 600) gives ≤ 10⁻⁶ (bracket ≤ 2 × 10⁻²·10⁻⁵ ≈ 10⁻⁵) — never above 10⁻⁵.

### 3.8 Route C: random coincidence (`P029_results.json: random_coincidence`)
ER rate in 10–14 keV in the FV: the 1710 science events populate ≲ 20.2 keVee (flat approximation, 84.7 per keV) →
339 events in 220 d → R_ER = 1.79 × 10⁻⁵ s⁻¹. Activation decays in the dead regions (RFR + wall layer = 6.85 % of the
xenon) over the run, all nuclides and calibrations: 5.5 × 10³ (thermal), 7.3 × 10⁴ (intermediate), 3.4 × 10⁵ (fast),
up to 1.0 × 10⁶ (fast, upper). Expected merged coincidences N_C = N_dead × R_ER × Δt:

| Δt | thermal | intermediate | fast | fast, upper |
|---|---|---|---|---|
| 0.2 μs | 2.0e-8 | 2.6e-7 | 1.2e-6 | 3.6e-6 |
| 1 μs | 9.9e-8 | 1.3e-6 | 6.0e-6 | 1.8e-5 |

This assumes every dead-region decay gives an S1 of ≈ 471 phd; the energy requirement lowers it further. LZ's accidental
model (2.7 ± 0.6 in the ROI) is built from measured isolated-S1 rates, which already contain the (time-averaged)
activation contribution; the dead-region activation S1-only rate on 16 June is 5.5 × 10⁻⁴ – 5.4 × 10⁻² Hz.

### 3.9 What activation does produce (`activation_lines_S1S2.csv`)
| deposit | S1c (phd) | log₁₀S2c | where |
|---|---|---|---|
| ¹²⁵I 67.3 keV | 421 | 4.61 | ER band, above ROI (tail leaks 1.1 × 10⁻⁴ at f = 0.2) |
| ¹²⁵I 40.4 keV | 252 | 4.39 | ER band, above ROI (3 % leak) |
| ¹²⁷Xe/¹²⁵Xe K 33.2 keV (γ escaped) | 205 | 4.32 | ER band edge (14 % leak) |
| ¹³³Xe ≥ 81 keV (β + 81) | ≥ 499 | ≥ 4.71 | ER band at S1c > 500 phd — LZ's tritium cut |
| ¹²⁹ᵐXe 236.1 keV | 1164 | 5.38 | HE-SB S1 range, ER band |
| ¹³¹ᵐXe 163.9 keV | 877 | 5.16 | HE-SB S1 range, ER band |
| ¹³³ᵐXe 233.2 keV | 1152 | 5.38 | HE-SB S1 range |
| ⁸³ᵐKr 41.5 keV | 259 | 4.40 | ER band |

¹³³Xe decays in the FV over the run: 1.5 × 10⁴ / 3.3 × 10⁵ / 1.6 × 10⁶ (thermal/intermediate/fast) — all at S1c ≥ 499 phd
in the ER band, i.e. outside the ROI (log₁₀S2c > 4.15) but inside the S1c range of Fig. 5's bottom panel and of the
HE SB; only their S2 tails could enter the MSSI validation regions.

Activation RFR-MSSI (LZ's own category): a ¹²⁷Xe decay within a few cm above the cathode whose γ is absorbed in the RFR.
Effective source layer (λ/2)∫₀¹ μ e^{−h₀/(λμ)} dμ with h₀ = 2 cm FV stand-off, summed over the 145–375 keV lines, divided by
the 130 cm FV height: 4.1 × 10⁻⁴ per ¹²⁷Xe FV decay → 0.16–1.8 events over the run (400–4500 decays); only the L-capture
share (14 %) lands in the ROI: **0.02–0.26 events at S1c ≈ 492 phd, log₁₀S2c ≈ 3.73, z < 5 cm above the cathode** — far
below the NR band and 21 cm below the event. This is presumably the "subdominant" ¹²⁷Xe MSSI LZ simulated; nothing is seen
there in Fig. 4.

## 4. Results summary
1. On 16 June 2023 the activation inventory (10 t) was: ¹³³Xe 6 × 10⁻³ – 0.66 Bq (central 0.14), ¹³¹ᵐXe 7.5 × 10⁻⁴ –
   6.9 × 10⁻² (0.015), ¹²⁹ᵐXe 4.2 × 10⁻⁴ – 3.5 × 10⁻² (8 × 10⁻³), ¹³³ᵐXe 3 × 10⁻⁴ – 3 × 10⁻² (6 × 10⁻³), ¹²⁷Xe
   6 × 10⁻⁵ – 2.2 × 10⁻³ (1.25 × 10⁻³ from Table I), ¹²⁵I 4.9 × 10⁻⁴ (34 % of its post-AmBe peak), ¹²⁵Xe 3 × 10⁻⁵
   (2.6 × 10⁻⁴ remaining), ¹³⁵Xe ≤ 6 × 10⁻⁷, ¹³⁷Xe/¹³⁵ᵐXe 0. Decays in the FV in ±1 h: 27–2700 (central 570), of which
   ¹²⁵I 1.65, ¹²⁵Xe 0.10, ¹²⁷Xe 0.2–7.6; in the RFR 3.9–380.
2. No activation product has a 12 ± 2 keV local deposit except β continua; the EC/IT nuclides (¹²⁵Xe, ¹²⁵I, ¹²⁷Xe,
   ¹²⁹ᵐXe, ¹³¹ᵐXe, ¹³³ᵐXe) give exactly zero correlated MSSI at the event's S2c. Their γ-rays (36–375 keV) have
   λ = 0.012–2.9 cm and P(20 cm) ≤ 1.0 × 10⁻³ (≤ 2.6 × 10⁻² with the longer attenuation bracket).
3. Correlated route-A expectation over the whole 220 d: 1.5 × 10⁻⁴ (central; 9 × 10⁻⁶ – 8 × 10⁻³ over all brackets), from
   ¹³⁵Xe/¹³⁷Cs β + hard γ in the first day after each calibration; at the event time ±1 h: 4 × 10⁻¹⁰ (≤ 3 × 10⁻⁹).
   Route B ≤ 10⁻⁵; route C 2 × 10⁻⁸ – 2 × 10⁻⁵.
4. Compared with LZ's MSSI model (4.9 × 10⁻³ ROI; 1.7 × 10⁻⁴ neighbourhood, P004) activation adds ≤ 3 % (central) and
   at most ×1.7 (extreme corner), all of it in the days after calibrations, none on 16 June.

## 5. Validation and robustness
- `radioactivedecay` half-lives agree with the dossier's recalled values (¹²⁵Xe 16.9 h, ¹²⁷Xe 36.4 d, ¹²⁹ᵐXe 8.88 d,
  ¹³¹ᵐXe 11.84 d, ¹³³Xe 5.24 d); ¹²⁵I removal fraction numerical 6.061 % vs analytic 6.061 %.
- ¹²⁵I timing reproduces P010 (peak ≈ 2 d, 34–40 % of peak at +8 d).
- NEST-LZ ER yields reproduce LZ's MSSI decomposition (12 keV ↔ 67 phd vs 69 ± 17) and the 500 phd ¹³³Xe cut (81 keV ↔ 499 phd).
- Natural-Xe thermal cross-section from the recalled isotopic set: 25.1 b (literature ≈ 24 b).
- Attenuation table reproduces LZ's "< 4 mm at 122 keV" (0.30 cm) and P004's 200/300/662/1000 keV values within 10 %.
- Variations: f = 0.1/0.2/0.3 (ε ×0.12/1/4.7, i.e. N(¹²⁵I) 9800/1200/260); N_cal 3 vs 1 (×3); t_resume 0.5–2 d
  (×1.07–0.86); capture set thermal/intermediate/fast (¹³³Xe ×0.044/1/4.8); Δt 0.2–1 μs (×5); Cs removed/retained (×6);
  attenuation recalled vs longer bracket (P(20 cm) at 375 keV ×26). The conclusion (no activation route ≥ 10⁻² over the
  run, ≤ 10⁻⁸ on 16 June) survives every corner.

## 6. Failed or abandoned approaches
- A first attempt to normalise on ¹²⁷Xe + ¹²⁵Xe = 1.5 directly was abandoned as primary because ε₁₂₇ needs the γ-escape
  geometry (×3 uncertain); it is used as a cross-check that fixes the intermediate capture set.
- Including ¹³⁷Cs as fully retained for 220 d made the correlated sum ×6 larger and was demoted to an upper bracket:
  Cs, like I, is a chemically active daughter removed by the getter (LZ's measured 3.6 d for ¹²⁵I).
- The first script run failed only because the output directory did not exist for the shell redirect (created).

## 7. Discussion
The activation inventory on 16 June was dominated by ¹³³Xe, ¹³¹ᵐXe and ¹²⁹ᵐXe — sources of ER-band events at
S1c > 500 phd and in the HE-SB S1 range, but with log₁₀S2c ≥ 4.7, far above both the ROI and the MSSI sideband regions —
and by ¹²⁵I at 34 % of its post-AmBe peak. The MSSI-like topologies require two things activation cannot supply
together: a 12 keV point deposit at the vertex (EC/IT deposits are quantised at 4.9, 33.2, 36.5–67.3, 164, 233, 236 keV)
and a γ-ray surviving ≥ 20 cm of LXe (P ≤ 10⁻³ for every line ≤ 375 keV). LZ's supplement statement that ¹²⁴Xe, ¹²⁷Xe,
¹³³Xe and ¹²⁵I are subdominant for MSSI is thus confirmed and made quantitative: ≤ 1.5 × 10⁻⁴ correlated events over the
run (≤ 8 × 10⁻³ in the most adverse corner), ≤ 10⁻⁹ within ±1 h of the event. The one activation-MSSI population that
does exist — ¹²⁷Xe within ~2 cm of the cathode with the γ absorbed in the RFR — predicts 0.02–0.26 ROI events at
S1c ≈ 490 phd, log₁₀S2c ≈ 3.7, z < 5 cm, a location and S2c distinct from the candidate. The calibration proximity
therefore does not open an activation route to the event; its remaining relevance is indirect (mixed-flow state, radon
tag unavailable, elevated isolated-S1 rate from ¹³³Xe in the RFR: 10⁻³–5 × 10⁻² Hz).

LZ-internal checks we propose: (i) the time series of the ¹³¹ᵐXe 164 keV and ¹²⁹ᵐXe 236 keV line rates after each of the
three calibrations, which measures the capture yield and the capture-spectrum mix directly (our thermal/fast bracket
spans ×80 in ¹³³Xe); (ii) the isolated-S1 rate in the 400–550 phd range versus time since calibration, entering the
accidental PDF for the June period; (iii) the S1c > 500 phd ER-band and HE-SB event rates versus time since
calibration, to confirm that the ER tail of Fig. 5's bottom panel is calibration-correlated; (iv) a search for the
predicted ¹²⁷Xe RFR-MSSI population at z < 5 cm, S1c ≈ 490 phd, log₁₀S2c ≈ 3.7.

## 8. Figures
- `figures/P029_fig1_activities_vs_time.png` — Activities (Bq, 10 t inventory) of ¹²⁵Xe, ¹²⁵I (with 3.6 d removal), ¹²⁷Xe,
  ¹²⁹ᵐXe, ¹³¹ᵐXe, ¹³³Xe, ¹³³ᵐXe, ¹³⁵Xe versus days after the end of a 1-day AmBe deployment, thermal-capture ratios
  normalised to LZ Table I (¹²⁵I = 8.9); dashed line: the event (+8.39 d). Intermediate/fast sets scale ¹³³Xe by 23/110.
- `figures/P029_fig2_gamma_transmission.png` — exp(−d/λ) for d = 20 cm and 26.9 cm versus γ energy (recalled XCOM-like
  attenuation; red: the longer bracket), with the activation lines marked and the 10⁻³ level.

## 9. References
1. LZ Collaboration, arXiv:2609.02823 (2026) — Table I, Fig. 2/4 captions, Discussion, supplement MSSI section.
2. LZ Collaboration, "Xenon activation and ¹²⁵I removal" (LZ:2024wvs, cited by the paper for the 3.6 d effective half-life).
3. ICRP Publication 107, Nuclear decay data (2008), via `radioactivedecay` 0.6.1 (A. Malins & T. Lemoine, JOSS 2022).
4. M. J. Berger et al., NIST XCOM photon cross-sections (values recalled).
5. S. F. Mughabghab, Atlas of Neutron Resonances (thermal cross-sections recalled).
6. O. Klein & Y. Nishina, Z. Phys. 52, 853 (1929).
7. Corpus: P004 (wall MSSI, neighbourhood fraction), P010 (¹²⁵I timing, f = 0.2 charge suppression, band width), P013.

## 10. Tools and provenance (mirrors `provenance/P029.json`)
- Agent tools: Read ×13 (PAPER_GUIDE, dossier, ledger, P004/P010/P013 papers, LZ tex l. 148–159, 208–337, 710–735,
  P013.json, lzcommon l. 1–125, two own figures); Bash ×15 (tex greps; lzcommon/versions/ls; radioactivedecay and
  periodictable probe; nestpy ER-yield probe; six script runs incl. one failed redirect; four word-count checks); Write ×4
  (script, details, JSON, paper); Edit ×25 (script ×13: intermediate set, run-integrated topology A, Cs removal,
  coincidence/print fixes, figure cosmetics; paper ×8: word-budget trims; JSON ×2 and details ×2: tool counts).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (integrate.quad, stats.norm, optimize.brentq); pandas 3.0.5;
  matplotlib 3.11.2; radioactivedecay 0.6.1 (Nuclide.half_life/decay_modes/branching_fractions/progeny,
  Inventory.decay/activities/numbers); periodictable 2.1.0 (Xe isotopic abundances); nestpy 2.1.1 via
  `lzcommon.nest_er_yields` (NEST-LZ Table S3 β yields); `common/lzcommon.py` (LZ constants).
- Script: `output/code/P029_activation.py` — `.venv/bin/python output/code/P029_activation.py > output/work/P029/run_log.txt 2>&1`.
- Recalled knowledge: 6 groups (emission schemes; thermal σ; fast σ; attenuation table; NEST α, KN, β shape; LZ geometry
  and removal assumptions) — see §2.
- Datasets: none. Data requests: none. WimPyDD files: none.
