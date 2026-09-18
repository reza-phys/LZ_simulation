# P091 — The corpus posterior after two weeks: updating the explanation probabilities for the LZ 248 keV event with P001–P099

*Simulated date 2026-09-16. Author profile: meta-analysis / evidence-synthesis group. Category STAT. Script: `output/code/P091_corpus_posterior.py` (run log `run_log.txt`; all tables in this directory).*

## 1. Motivation and framework

The Phase-1 dossier (`output/00_evidence_dossier.md`, Section 5, 3 September) assigned probabilities to eleven explanation classes A–K of the single 248 keV nuclear-recoil-like event: A statistical fluctuation of a correctly modelled background 0.10; B wall/RFR MSSI 0.19; C ER leakage 0.13; D neutron 0.08; E accidental 0.05; F instrumental artefact 0.18; G calibration-related 0.05; H non-DM new physics 0.02; I inelastic DM 0.10; J momentum/spin-suppressed elastic EFT DM 0.06; K other DM 0.04 (DM total 0.20). Two weeks and 98 papers later (P001–P099, with P090, P094 and P099 appearing while this paper was written and included), most of those classes have quantitative corpus results. We ask: what are the updated class probabilities, how robust are they to the two judgemental inputs, which papers carried the information, and which classes moved most?

**Formula.** For class *i* with prior π_i and corpus event-likelihood L_i,

    P(i | corpus) = π_i L_i / Σ_j π_j L_j .                                              (1)

L_i is the expected number of *event-like occurrences* of class *i* in the event's neighbourhood — S1c > 500 phd, |d| ≤ 2σ_NR below the NR median (P024/P093 convention), single site, veto-silent, at the observed position and on the observed date — built as a product of factors, each attributed to the paper that established it:

    L_i = R_i × Π_f F_{i,f}  (+ additive sub-channels),                                  (2)
    L_DM = (B_DM b_H / f_FP),   L_I = L_DM · share_I · S_I · T_I,  L_J = L_DM · share_J · S_J · T_J,  L_K = L_DM · share_K · S_K .   (3)

Here B_DM is P027's model-marginalised Bayes factor against the modelled background b_H = 5.7×10⁻⁴ (P016; so B_DM b_H is the marginal DM event-likelihood in the same currency as the background classes), f_FP is P061's forking-paths divisor for the non-blind fiducial-volume change and salting failure, share_{I,J,K} are P027's posterior class masses, S are survival factors from UV-completion/cosmology constraints established after P027, and T the 16 June date odds (P006). Every factor with a quoted corpus range is drawn log-uniformly within that range (Monte Carlo, 200 000 draws, seed 91); the two data-constrained mismodelling factors have their own posteriors (Section 2). Medians and 16–84 % intervals of Eq. (1) are the deliverable. The corpus likelihood ratio quoted in the class table is Λ_i = L_i / L_A (relative to a fluctuation of the modelled background).

**What "likelihood" means here and the double-counting caveat.** The dossier's Section-5 weights were written with the event in hand (energy, band position, "inside FV"), so multiplying them by event likelihoods that include the *rate* at the event's location partly double-counts, as P061 noted. The corpus results are largely *new analyses of the same published information* (sidebands, position, veto silence, pulse shape, low-energy null, the model space), so most of L_i is genuinely new relative to the dossier; the shared part (the coarse rate) is the same for all classes only if the reference is the same, which is why the sensitivity table also includes a flat prior (corpus likelihoods alone).

## 2. The likelihood table (inputs, with sources)

All ranges are log-uniform unless stated. Medians of the MC draws are in `P091_factor_table.csv`.

| class | factor | range | median | source (primary; confirming) |
|---|---|---|---|---|
| A | b_nb, modelled background in the neighbourhood | 3.5×10⁻⁴ – 6.4×10⁻⁴ | 4.7×10⁻⁴ | P016 b_H = 5.7×10⁻⁴; P093 3.5×10⁻⁴; P070 6.4×10⁻⁴; P001 2×10⁻⁴–10⁻³ |
| A | T_A, position LR of the modelled mixture vs uniform (inverse) | 0.56 – 1 | 0.75 | P070 (uniform favoured ×1.8 because 41 % accidentals and 9 % ν are uniform) |
| A | ν_incoh, incoherent atmospheric-ν lone NR (additive, known physics not in LZ's model) | 8×10⁻⁶ – 5×10⁻⁴ | 6.3×10⁻⁵ | P019 1.7×10⁻⁴; P060 1.77×10⁻⁴; P087 |
| B | R_B, wall MSSI as modelled, ±2σ_NR at S1c > 500 | 1.15×10⁻⁴ – 2.5×10⁻⁴ | 1.7×10⁻⁴ | P004: 0.0048 × f_nb 0.035 (0.024–0.053) |
| B | k_B, sideband-allowed mismodelling | Gamma(2, 1/3.23) | 0.52 (mean 0.62; 95 % < 1.47) | P004 (2 obs vs 3.23 pred; k_ML 0.62, k < 1.64 at 95 %); P090 (k̂ 0.62, k < 1.64); P053 |
| B | P_pos, inverse position LR wall MSSI vs uniform | 0.02 – 0.43 | 0.093 (≈1/11) | P033 (f_pos 0.98 %, exp(−d/4.3 cm)); P070 LR 15 (2.3–50 by γ source); P079 (LZ's own λ = 1.2 cm would be 7× steeper) |
| B | V_B, veto-silence penalty | 0.10 – 0.79 | 0.28 | P073 (0.46 central; k_req 619 → 1350) |
| B | W_B, hit-pattern/S1 consistency for wall MSSI | 0.3 – 1 | 0.55 | P093 |
| B | RFR + ²¹⁴Pb MSSI at the position (additive) | 10⁻⁹ – 10⁻⁷ | 10⁻⁸ | P033 (RFR 1.4×10⁻⁸), P053 (²¹⁴Pb: k_req ≥ 2.7×10⁵ vs 3.4 allowed), P070 (RFR LR 17–1500), P093 (10^6.9) |
| C | R_C, ER-tail extrapolation floor → plateau | 1.6×10⁻⁴ – 2.8×10⁻³ | 6.7×10⁻⁴ | P010 (flat extrapolation of Fig. 5 tail 2.8×10⁻³, floor 1.6×10⁻⁴, ≤ 10⁻² cap; any tail giving ≥ 0.1 events predicts 1–9 gap events, 0 seen); P024 (event is a typical NR-band member, 1.5σ, P = 6 %); P095 (Migdal/brems shift the event *up*, wrong sign) |
| C | F_C, recombination-tail shape | 0.014 – 1 | 0.12 | P056 (Gaussian 4×10⁻⁵ at 500–600 phd, 7×10⁻⁷ at the event; band-preserving skews 4×10⁻¹⁴–6×10⁻³); P093 NR:ER 10^3.6 [2.3, 4.7] |
| D | L_D, neutron single scatters in 200–270 keV | 5×10⁻⁶ – 2×10⁻⁴ | 3.2×10⁻⁵ | P013 (6.6×10⁻⁶ central; ≤ 5×10⁻⁴ stacked-conservative); P049 (muon-induced 1.3×10⁻⁵, 3.3×10⁻⁶–5.4×10⁻⁵); P063 (fission 1.5×10⁻⁹); P093 10^4.6 |
| E | R_E, accidentals as modelled in the neighbourhood | 1.0×10⁻⁴ – 2.0×10⁻⁴ | 1.4×10⁻⁴ | P022 (1.49×10⁻⁴ from digitised Figs. 5/S3) |
| E | k_E, allowed high-S1 mismodelling | log-uniform[1, 10³] × e^{−0.016k} | 6.6 (mean 17; 84 % 31) | P022 (UDT: 0.016 expected, 0 observed above 495 phd → k < 183 at 95 %; a 10 % origin needs k = 708) |
| E | T_E, S2-width/drift and cathode topology | 0.01 – 1 | 0.10 | P093 W-range; P022 (cathode-emission S2 width 1.65 vs 1.49 μs, 2.4σ) |
| F | L_F, artefact residual incl. unknown unknowns (judgemental; scanned) | 6×10⁻⁴ – 3×10⁻² | 4.3×10⁻³ | P041 (residual 0.006: bulk lifetime 10⁻⁶, transient absorber 1.6×10⁻³, clipping 2×10⁻⁵, mis-pairing/unknowns 4×10⁻³); P093 (0.006 × π_U → 6×10⁻⁴); P061 (0.006 + L_rest 10⁻³–0.3, central 0.016); P077 (PSD inconclusive; single-event LR ≲ 2–5) |
| G | R_G, activation MSSI-like events per run | 9×10⁻⁶ – 8×10⁻³ | 2.7×10⁻⁴ | P029 (1.5×10⁻⁴ central; ≤ 10⁻⁹ within ±1 h); P053 (mixed-flow timing unremarkable, P = 0.33) |
| G | P_pos,G, position penalty for an MSSI-like topology | 0.01 – 0.43 | 0.065 | P033, P070 |
| G | (energy-scale route) | — | no event-creating mechanism | P009/P064/P098: E = 246–248 keV on LZ's scale, σ_E 11.4 keV, the "265 keV" reading was a Table-S5 artefact; acceptance 0.93–0.95 |
| H | L_H, kinematically special non-DM new physics | 10⁻⁵ – 5.7×10⁻⁴ | 7.5×10⁻⁵ | upper: P061 convention (fits no better than a background of equal rate); lower: every enumerated candidate excluded — P019 (exotic ν need ≥ 790 low-E companions), P036 (nuclear excitation +15.8σ off band), P060 (astrophysical ν ≥ 10⁵ short), P080 (MCP/SIMP/relics), P087 (atm-ν floor 3×10⁻⁵/t·yr) |
| DM | R_DM = B10 × b_H (P001, superseded) → B_DM × b_H | 16.4–28.7 × 5.7×10⁻⁴ = 9.3×10⁻³ – 1.64×10⁻² | 1.24×10⁻² | P027 (uniform over 298 spectra 16.4; class priors 28.7; Bayesian trials factor 10 ≈ N_eff 12–14 of P008/P071); P001 (7–37 after Occam; SBB cap 14.7); P016 (single-model L10 44.8) |
| DM | 1/f_FP | f_FP 1 – 10 | 0.32 | P061 (non-blind FV shrink, salting failure); P071 (corpus spectra 2.59 → 2.50σ global) |
| I | share_I | 0.58 – 0.85 | 0.70 | P027 (58–85 % inelastic); P021, P082 |
| I | S_I, UV-completion survival | 0.3 – 1 | 0.55 | P076 (solar capture excludes the Higgsino at every δ; closes δ ≥ 366/360/370 keV for dark photon / B–L Z′ / U(1)_B); P084 (Y ≥ 1 multiplets excluded ×4.7–8300); P025 (m_A′ < 9 GeV excluded); P086 (m_A′ 9.2–34.8 GeV survives); P075 (thermal windows δ ≤ 356 dark photon, ≤ 298/318 Z′); P054; P038 (the δ = 380 keV likelihood peak is an efficiency-edge effect); P082 (joint region δ 335–390 keV, m ≥ 400 GeV) |
| I | T_I, date odds | 1.4 – 2.5 | 1.87 | P006 (1.41/2.08/2.55 at δ = 300/350/380 keV); P034; P055 (substructure changes LR by ≤ 0.6 %); P085 (focusing ≤ 1.3 %); P092 (diurnal ≤ 1.6 %) |
| J | share_J | 0.11 – 0.30 | 0.18 | P027; P016 (87 % of mass in N_lo ≤ 5 operators); P003/P012 (L10/O6 N_lo 0.2–0.4) |
| J | S_J, EFT/UV consistency | 0.2 – 1 | 0.45 | P031 (Λ = 1.5–62 GeV at 1 TeV, Λ/m_χ ≤ 0.06: needs a light mediator); P074 (mediators ≥ 0.11–0.41 GeV keep N_lo ≤ 5; contact-like above 0.45 GeV); P025 (L10 dipole WIMP annihilates ≥ 3×10⁴ above bounds → asymmetric DM only); P044/P057/P089 (degeneracies: neutral) |
| J | T_J, date odds elastic | 0.98 (fixed) | 0.98 | P006 |
| K | share_K | 0.02 – 0.08 | 0.04 | P027 remainder |
| K | S_K, survival | 0.05 – 0.5 | 0.16 | P040 (CRDM/fast SI: N_lo ≥ 350); P080 (MCP N_lo 2×10⁵, SIMP 676–2270); P072/P058 (exothermic B ≤ 2, B(exo/endo) 0.04–0.67); P030/P055/P097 (no stream or disk helps); P065 (MiDM alive for δ < 355 keV but Ωh² = 6.2); P023 (composite alive for δ ≤ 326 keV); P094 (multi-component signals elsewhere, neutral) |

Two factors deserve comment. (i) *k_B* is data-dominated: with a log-uniform prior on k the six wall sideband bins give k ∝ k e^{−3.23k}, i.e. Gamma(2, 1/3.23), whose mean 0.62 and 95 % point 1.47 reproduce P004/P090. (ii) *k_E* is prior-dominated: the UDT zero (0.016 k expected) only bites above k ≈ 60, so with a log-uniform prior on [1, 10³] the marginal has median 6.6 and mean 17 — the accidental class is the one background whose mismodelling is not pinned by data (P022's "one testable loophole"); its 84 % point (k = 31) is what puts E's upper tail at 0.03.

## 3. Results

### 3.1 The A–K table under the dossier prior (`P091_class_table.csv`, Fig. 1)

| class | prior | L median | Λ = L/L_A median | posterior median | 68 % | log₁₀(post/prior) |
|---|---|---|---|---|---|---|
| A fluctuation of modelled/known bkg | 0.10 | 4.5×10⁻⁴ | 1 | 0.033 | 0.013 – 0.076 | −0.48 |
| B wall/RFR MSSI | 0.19 | 1.2×10⁻⁶ | 2.6×10⁻³ | 1.6×10⁻⁴ | 3.3×10⁻⁵ – 7.5×10⁻⁴ | −3.08 |
| C ER leakage | 0.13 | 7.9×10⁻⁵ | 0.17 | 0.0073 | 0.0013 – 0.041 | −1.25 |
| D neutron | 0.08 | 3.2×10⁻⁵ | 0.069 | 0.0018 | 4.5×10⁻⁴ – 0.0072 | −1.65 |
| E accidental | 0.05 | 1.0×10⁻⁴ | 0.22 | 0.0036 | 4.7×10⁻⁴ – 0.028 | −1.14 |
| F artefact + unknowns | 0.18 | 4.3×10⁻³ | 9.3 | 0.64 | 0.31 – 0.87 | +0.55 |
| G calibration-related | 0.05 | 1.8×10⁻⁵ | 0.038 | 6.2×10⁻⁴ | 5.0×10⁻⁵ – 0.0077 | −1.90 |
| H non-DM new physics | 0.02 | 7.5×10⁻⁵ | 0.17 | 0.0011 | 2.4×10⁻⁴ – 0.0047 | −1.28 |
| I inelastic DM | 0.10 | 2.8×10⁻³ | 6.1 | 0.23 | 0.074 – 0.50 | +0.36 |
| J q⁴-spin elastic EFT | 0.06 | 3.1×10⁻⁴ | 0.68 | 0.014 | 0.0045 – 0.037 | −0.63 |
| K other DM | 0.04 | 2.5×10⁻⁵ | 0.053 | 7.2×10⁻⁴ | 2.1×10⁻⁴ – 0.0023 | −1.74 |
| **DM total (I+J+K)** | **0.20** | | | **0.25** | **0.081 – 0.54** (95 %: 0.029 – 0.76) | +0.09 |
| mismodelled backgrounds B–G | 0.68 | | | 0.70 | 0.39 – 0.90 | |

Central (all factors at range medians): F 0.68, I 0.25, A 0.037, J 0.017, C 0.009, E 0.004, D 0.002, H 0.0013, K 0.0009, G 0.0008, B 0.0002. P(DM) exceeds 0.5 in 19.8 % of the MC volume and is below 0.1 in 20.8 %.

### 3.2 Which classes moved, and why

- **B (−3.1 dex, 0.19 → 1.6×10⁻⁴)** is the largest mover: four independent handles each cost a decade or more — sideband-pinned rate (P004, P090), position (P033/P070/P079: exp(−d/4.3 cm) from the wall, LR 15), veto silence (P073, ×0.46) and hit pattern (P093) — and the RFR/²¹⁴Pb sub-channels are 10⁻⁸ (P033, P053). It is dead several times over; no single B-paper is now indispensable (leave-one-out KL ≤ 10⁻³ nats).
- **G (−1.9 dex)**: activation decays cannot deposit 12 keV locally and their γ's cannot reach 27 cm (P029); ²¹⁴Pb in mixed flow is closed (P053); the energy-scale question was resolved in LZ's favour (P009/P064/P098).
- **K (−1.7), D (−1.65), H (−1.3), C (−1.25), E (−1.1)**: boosted/exotic DM and every non-DM new-physics candidate fail the low-energy null (P040, P080, P019, P036, P060); neutrons need E_n ≥ 8.2 MeV and ~2×10⁴ companions (P013, P049, P063); the ER tail is bounded by the empty gap and the Gaussian recombination model (P010, P056); accidentals are bounded by the UDT zero (P022) and topology (P093).
- **J (−0.63)**: consistent with the event (N_lo 0.2–0.4) but only 11–30 % of the DM model mass (P027) and UV-fragile (P031, P025).
- **A (−0.48)**: the modelled background at the location is 3.5–6.4×10⁻⁴ (P016/P093/P070), 6–9× below the DM marginal likelihood and 10× below the artefact residual; position favours uniform sources ×1.8 (P070).
- **F (+0.55, 0.18 → 0.64)** and **I (+0.36, 0.10 → 0.23)** absorb the mass. F wins not because it fits well (L_F ≈ 4×10⁻³, "3–25× any published detector scale", P041) but because its likelihood is the largest of the non-DM classes and every alternative was eliminated by 1–3 decades. I wins within DM because P027 places 58–85 % of the model mass there and the June date adds ×1.4–2.5 (P006), partly offset by the UV-completion survival factor 0.3–1 (P076 kills the Higgsino; the dark photon at m_A′ = 9–35 GeV and δ ≲ 355 keV, and Z′ at δ ≲ 340 keV, survive).

### 3.3 Sensitivity to the two judgemental inputs (`P091_sensitivity.csv`, Fig. 2)

Variance decomposition of logit P(DM) over the MC factors (squared correlation, normalised): L_F 0.60, f_FP 0.29, S_I 0.07, B_DM 0.02, T_I 0.01, everything else < 0.01. The artefact likelihood and the DM prior/penalty are the whole story.

P(DM | corpus), central factors (MC median and 68 % in the CSV):

| prior (π_DM; non-DM split) | L_F = 6×10⁻⁴ (P093) | L_F = 4.2×10⁻³ (range median) | L_F = 3×10⁻² (P061 upper) | L_F log-uniform |
|---|---|---|---|---|
| sceptic π_DM = 0.01 (P083 implied 0.009); dossier split | 0.067 | 0.015 | 0.0022 | 0.013 (0.0035–0.046) |
| P083 base rate π_DM = 0.062; dossier split | 0.32 | 0.088 | 0.014 | 0.080 (0.023–0.24) |
| dossier π_DM = 0.20 | 0.64 | 0.27 | 0.052 | 0.25 (0.081–0.54) |
| "theorist" π_DM = 0.20; uniform over A–H | 0.70 | 0.38 | 0.089 | 0.33 (0.13–0.62) |
| flat 1/11 each (π_DM = 0.27) | 0.71 | 0.39 | 0.093 | 0.34 (0.13–0.63) |

P(F) at the corresponding cells runs from 0.14 (flat, L_F = 6×10⁻⁴) to 0.99 (sceptic, L_F = 0.03). Note that the dossier's DM total is already 0.20, so a "theorist" prior with π_DM = 0.2 differs from the dossier only through the background split (uniform vs dossier), which moves P(DM) by 0.27 → 0.38; a theorist who also drops the forking-paths penalty (f_FP = 1) and takes B_DM = 28.7 reaches 0.64 at the central L_F. With P083's base rate (π_DM = 0.062) and the central artefact likelihood, P(DM) = 0.09 (0.02–0.24).

**Comparison with P061/P083.** P061's binary formula at its central values gives P(DM) = 0.021 (our re-evaluation; P061's community-prior median 0.015, P083 0.012). Our base-rate row gives 0.09 for the same π_DM ≈ 0.06 because the two frameworks distribute the non-DM prior differently: P061 gives the *modelled* background as modelled 0.88 of the prior (π_B) and the unknowns π_U = 0.1, whereas the dossier gives the modelled-background class A only 0.10 and the mismodelled/unknown classes 0.68. Since the corpus killed B, D, E, G and mostly C, the dossier framing leaves DM competing essentially against F alone, with L_I/L_F ≈ 0.66 and π_I/π_F = 0.56. Under the P061 framing, π_A L_A = 0.88 × 5.7×10⁻⁴ = 5×10⁻⁴ is an additional competitor comparable to π_F L_F. We regard the dossier's A = 0.10 as a post-event judgement (the dossier wrote "the model itself gives ~10⁻⁴ for this location"), i.e. already partly updated, which is the double-counting caveat of Section 1; the sceptic row (π_DM = 0.01) is the closest analogue of P061/P083 and gives 0.013 (0.0035–0.046), matching them.

### 3.4 Information contribution by paper (`P091_paper_contributions.csv`, Fig. 3)

We process the primary papers in P-number order (≈ chronological) and record KL(post_after ‖ post_before) at central factor values. Rate-setting factors need a null reference; we use the corpus's own viability yardstick L_ref = 0.1 (the "10 % chance of one event" convention of P004/P033/P041), and repeat with L_ref = b_H = 5.7×10⁻⁴ ("as modelled"). We also give the total decades moved (Σ|log₁₀(factor/null)|) and a leave-one-out KL under the b_H reference.

| rank | paper | classes | KL_seq (L_ref 0.1) | KL_seq (L_ref b_H) | dex moved | leave-one-out KL (b_H) |
|---|---|---|---|---|---|---|
| 1 | P041 | F | 1.10 | 0.29 | 1.37 | 0.48 |
| 2 | P004 | B | 0.26 | 0.027 | 3.05 | 0.0008 |
| 3 | P016 | A | 0.26 | 0.0003 | 2.32 | 0.0005 |
| 4 | P029 | G | 0.22 | 0.002 | 2.57 | 0.0003 |
| 5 | P010 | C | 0.22 | 0.0003 | 2.17 | 0.0001 |
| 6 | P013 | D | 0.17 | 0.012 | 3.50 | 0.031 |
| 7 | P022 | E | 0.17 | 0.001 | 3.63 | 0.0003 |
| 8 | P061 | DM (f_FP) | 0.16 | 0.16 | 0.50 | 0.15 |
| 9 | P001 | DM (B10) | 0.15 | 0.83 | 1.04 | 0 (superseded by P027) |
| 10 | P019 | A, H | 0.063 | 0.002 | 3.12 | 0.007 |
| 11–20 | P076 0.037, P093 0.025, P056 0.021, P027 0.020, P006 0.004, P031 0.001, P033 0.001, P070 0.001, P040 0.0005, P073 0.0004 | | | (b_H ref: P027 0.18, P006 0.05, P076 0.037, P093 0.025, P056 0.021) | | |

Under the viability reference the eliminations dominate (each closes a class that started "viable"); under the as-modelled reference the DM-evidence papers dominate (P001 0.83, P027 0.18, P061 0.16) because the backgrounds start at their modelled level and only the DM likelihood departs from it. P041 is first under both (1.10 / 0.29) and has by far the largest leave-one-out KL (0.48 nats): the artefact class is the one whose likelihood is both large and singly sourced. Confirming papers (P033, P070, P073, P090 for B; P049, P063 for D; P056/P093 for C; P034/P055/P085/P092 for the date; P084/P025/P086/P075/P054/P082/P038 for S_I) score low in the sequential KL because the class they touch was already settled by an earlier paper; their value is robustness, not movement — e.g. removing all four B-handles at once would raise B to ≈ 0.03, but removing any one leaves it < 10⁻³.

### 3.5 Predictive check

Combining the class posterior with P081/P099's predictives (P(≥ 1 band event in LZ's next 2.8 t·yr | DM) = 0.25; a steady LZ-specific unknown 0.5 with a one-off share 0.5 → 0.25; modelled background 5.6×10⁻⁴), the mixture predictive for ≥ 1 new 200–270 keV event is 0.23 (0.21–0.24), nearly independent of P(DM) because the two live classes (F, I) predict alike. This is P081's point restated: a new band event will not separate F from I; only off-band placement, a second detector (P069, P088) or waveform-level artefact tests (DR-003, P093) will.

## 4. Validation and robustness

1. **P004/P090 reproduction.** Gamma(2, 1/3.23) gives mean 0.62, 90 % 1.20, 95 % 1.47 versus P004's k_ML 0.62, k < 1.36 (90 %), < 1.64 (95 % LR); the LR interval is slightly wider than the Bayesian one, as P004 also found (1.95 Bayes).
2. **P093 consistency.** Our class medians in P093's currency (0.19 × 10^−odds): WALL 7.6×10⁻⁶ vs ours 1.2×10⁻⁶ (we include P073's veto factor and the P033 lower position bracket; P093's MC range [1.2×10⁻⁶, 1.2×10⁻⁵] overlaps); ER 4.8×10⁻⁵ vs 7.9×10⁻⁵; NEU 4.8×10⁻⁶ vs 3.2×10⁻⁵ (we include the muon channel and P013's conservative bracket); ACC 1.5×10⁻⁵ vs 1.0×10⁻⁴ (we allow k_E); ART 6×10⁻⁴ vs 4.3×10⁻³ (we do not multiply by π_U, which is a prior). All within the quoted 68 % ranges.
3. **P061 reproduction.** P061's Eq. at central values → 0.021 (their table 0.02–0.03 central; median 0.015). Our sceptic row 0.013 (0.0035–0.046).
4. **Reference dependence of the ranking** (Section 3.4): the top paper is the same under both references; the next nine reorder between "eliminations" and "DM evidence" — reported as such.
5. **k_E prior.** Restricting k_E to [1, 30] would lower E's median to 0.002 and its 84 % point to 0.012; using k_E = 1 (accidentals exactly as modelled) gives E = 6×10⁻⁴. None changes any other class by more than 1 %.
6. **P001 vs P027.** Using P001's B10 range (7–37) instead of P027's 16.4–28.7 changes P(DM) medians by < 0.02.

## 5. Failed or abandoned approaches

- A single "unmodelled" hypothesis U with a free rate (P001/P061 framing) cannot be mapped one-to-one onto the dossier's classes B–G; we kept the dossier's eleven classes and put unknown unknowns into F, as the dossier itself does.
- Attributing the position factor to P070 rather than P033 for the sequential ranking: P070's LR 15.1 was computed from P033's maps, so P033 keeps the credit and P070 the confirmation.
- A leave-one-out KL with rate factors reset to the 10 % yardstick (first run) was abandoned: it rewards any paper that happened to *define* a class's rate regardless of whether the class matters, and inflated P016/P013/P019 to 1–2 nats. The as-modelled reference is used for leave-one-out instead.
- First run of the variance decomposition returned NaN because a fixed factor (T_J) passed a `std == 0` guard by 10⁻¹⁷; fixed with a tolerance.

## 6. Figures

- **Fig. 1** `figures/P091_fig1_prior_posterior.png`: dossier prior (blue) vs corpus posterior median (orange) for A–K on a log scale; bars are 16–84 % over the corpus ranges. Caption numbers = column "posterior median" of Section 3.1.
- **Fig. 2** `figures/P091_fig2_sensitivity.png`: P(DM) for five prior choices × three artefact-class likelihoods (central factors elsewhere).
- **Fig. 3** `figures/P091_fig3_papers.png`: the ten most consequential papers by sequential KL under the viability reference (blue) and the as-modelled reference (aqua).

## 7. Discussion

The corpus has done what a two-week community response can do with a published single event: it has quantitatively closed every *specific* background (wall/RFR MSSI, ²¹⁴Pb, activation, neutrons of every origin, accidentals within their validation, the ER recombination tail, nuclear excitation, exotic neutrinos and exotic relics) by one to three decades each, and it has quantified the DM evidence as a model-marginalised Bayes factor of 16–29 against the modelled background, i.e. a marginal DM event-likelihood of ~10⁻² divided by a forking-paths penalty of 1–10. What is left is a two-horse race between an instrumental/unknown effect that no one can size better than "≈ 4×10⁻³, from 6×10⁻⁴ to 3×10⁻²" and inelastic dark matter whose most motivated UV completion (the Higgsino) is excluded by solar capture while its phenomenological spectrum survives. Which horse leads is set by the prior on DM (P083's base rate 0.06 vs the dossier's 0.20) and by the artefact likelihood; the data on disk (XENONnT/PandaX-4T reanalyses, LZ's extended sideband and waveform-level tests, P069/P088/P093/P099) decide it, not further reanalysis of the published event.

## 8. References

LZ Collaboration, arXiv:2609.02823 (2026). R. E. Kass, A. E. Raftery, J. Am. Stat. Assoc. 90, 773 (1995). T. Sellke, M. J. Bayarri, J. O. Berger, Am. Stat. 55, 62 (2001). S. Kullback, R. A. Leibler, Ann. Math. Stat. 22, 79 (1951). D. V. Lindley, Ann. Math. Stat. 27, 986 (1956). A. Gelman, E. Loken, Am. Sci. 102, 460 (2014). Corpus: dossier 00; P001, P003, P004, P006, P008, P009, P010, P012, P013, P016, P019, P021, P022, P023, P024, P025, P027, P029, P030, P031, P033, P034, P036, P038, P040, P041, P044, P049, P052, P053, P054, P055, P056, P057, P058, P060, P061, P063, P064, P065, P069, P070, P071, P072, P073, P074, P075, P076, P077, P079, P080, P081, P082, P083, P084, P085, P086, P087, P088, P089, P090, P092, P093, P094, P095, P097, P098, P099.

## 9. Tools and provenance (mirrors `provenance/P091.json`)

- **Agent tools.** Read ×18: PAPER_GUIDE; dossier; ledger (95 rows, all columns, via three pandas dumps read as four pages); P061.md, P093.md, P094.md, P083.md, P041.md, P090.md, P099.md; P093.json (format); own figures (fig1, fig2, fig3 ×2). Bash ×17: ls/wc of corpus + versions; three ledger dumps; grep of P061 details/P027 + ls work dirs; palette grep (dataviz skill; JS validator skipped); script runs ×4 (one failed on a missing directory, one variance-NaN); NaN probe; factor-table dump; ls for P090/P099; word counts ×5. Write ×6 (script, details, JSON, paper ×3). Edit ×16 (script ×8: variance guard, leave-one-out reference, column renames, legend, P090 citation; paper ×8: word trimming). Skill ×1 (dataviz).
- **Software.** python 3.12.13; numpy 2.5.3 (default_rng(91).uniform/gamma, quantile, median, corrcoef, interp, logspace, cumsum); pandas 3.0.5 (DataFrame, pivot, to_csv, read_csv); matplotlib 3.11.2 (Agg; bar, barh, errorbar, imshow, LinearSegmentedColormap); common/lzcommon.py (LZ dict, import cross-check only). Bayesian algebra (Eqs. 1–3, Gamma posterior for k_B, k_E marginal) derived by hand.
- **Script.** `output/code/P091_corpus_posterior.py` — `.venv/bin/python output/code/P091_corpus_posterior.py > output/work/P091/run_log.txt 2>&1` (1.9 s).
- **Local inputs.** `output/00_evidence_dossier.md` (Sections 2, 4, 5); `output/results_ledger.csv` (all 95 rows: headline, key numbers, stance, confidence); `output/papers/P041.md, P061.md, P083.md, P090.md, P093.md, P094.md, P099.md`; `output/work/P061/details.md` §1.3 and §4.4; `output/provenance/P093.json` (format); `environment/ENVIRONMENT_versions.txt`; `output/code/common/lzcommon.py`.
- **Recalled knowledge.** (1) Poisson colouring theorem: for superposed independent Poisson processes P(H | point) = μ_H p_H / Σ μ p — certain. (2) A Poisson count n with expectation a k and a 1/k prior gives a Gamma(n, 1/a) posterior on k — certain. (3) Kullback–Leibler divergence as expected log-likelihood-ratio information — certain. Everything else is a corpus number cited by P-number.
- **Datasets.** none. **Data requests.** none (the artefact class is provisional pending DR-003 via P093, which would tighten L_F). **WimPyDD files.** none.
