# P093 — Event-topology odds for the LZ 248 keV candidate: complete research record

Simulated arXiv date 2026-09-16 · category BKG · physics.ins-det (cross-list hep-ex) · detector-physics and statistics group.

## 1. Motivation and framework

The corpus has, paper by paper, quantified every *topological* handle LZ published for its single 248 keV
candidate (S1c = 540.1 phd, S2c = 9268 phd, 26.4 cm above the cathode, 26.9 cm from the wall, science sample,
16 June 2023): position (P033, P070), band distance (P009, P024, P056), pulse shape (P077, LZ waveform supplement),
veto (P042, P063), timing (P006, P053, P049) and the energy dependence of each background (P004, P013, P019, P022,
P056). Nobody has yet put them into one table. We do so and compute a *classification posterior* over origins.

For a marked Poisson process with independent components H, the probability that one observed point with
handle values D belongs to component H is

    P(H | D) = mu_H p_H(D) / sum_H' mu_H' p_H'(D),                              (1)

where mu_H is the expected number of events of type H in the WS-ROI science sample and p_H(D) the density of the
observed handles under H. This conditions on the observed event and is independent of how many other events
there are. We factorise p_H(D) into the handles

    rate  mu_ROI(H)                    expected type-H events in the WS ROI (pre-veto where V is quoted)
    E     P(S1c > 500 phd | H, ROI)    energy: the bottom Fig. 5 panel
    B     P(|d| <= 2 sigma_NR | H, S1c > 500)   band distance (LZ's Fig. 5 variable; event at d = -1.5)
    P     p_H(r,z) / p_uniform(r,z)    position-density ratio at the event (P070 convention: L_P = 1/LR)
    W     pulse/waveform handles exactly as LZ states them
    V     veto quietness: (1 - tagging efficiency) for hypotheses that are vetoed
    T     timing relative to a time-flat background

so that mu_H p_H(D) = rate x E x B x P x W x V x T. The neighbourhood mu_nb(H) = rate x E x B x V is the corpus's
standard "S1c > 500 phd, within +-2 sigma_NR" region (P004, P022, P070). The single-NR hypothesis carries the
free normalisation s (expected DM-like events in the ROI); LZ's L10 fit gives s = 1.0 (+1.4 -0.7), the 90 %
lower edge is 0.105 (P045). Because LZ fitted s to the event, we present (a) odds per unit s, (b) the posterior at
s = 1, and (c) the break-even s_1/2 at which P(NR) = 1/2, plus P(NR) for s = 10^-3 ... 2.4.

Hypotheses: NR (single nuclear recoil, DM-like), WALL (wall MSSI), RFR (reverse-field-region MSSI), ER
(charge-poor / leakage electron recoil incl. EC lines), ACC (accidental S1-S2 pairing), NEU (radiogenic +
cosmogenic neutron), ART (instrumental artefact), NU (atmospheric-neutrino NR; topologically identical to DM).

Correlations. For WALL and RFR the position PDFs of P070 are *conditional on the 12 keV MSSI class* (P033 maps),
so E x B (class fraction) times 1/LR_pos (class-conditional position density) is the correct joint probability; we do
not multiply the class fraction by P033's region-integrated f_pos as well. For ACC the cathode-emission sub-topology
(P022) correlates position (long drift) with the S2-width handle; it enters through the range on P. For ART the
P041 residual is already a joint E x B x W x T statement and is entered once.

Three cases: (i-a) LZ's own Table I / Fig. 5 counts and only the (S1c, d) information LZ's likelihood uses;
(i-b) LZ counts plus the corpus handles P, W, V, T; (ii) corpus-revised counts, including the unmodelled classes
ER-leakage and ART, plus all handles. Ranges are propagated by Monte Carlo (40 000 draws, log-uniform inside each
quoted bracket; fixed where no bracket exists). Central values and MC medians are both reported.

## 2. Inputs (every number with its source)

LZ paper (fulltext.tex): l.113-147 (samples, veto definitions: Skin > 2.5 phd within +-0.25 us, OD > 4.5 phd within
+-0.3 us; delayed 600 us; random-veto 0.01 % / 2.86 %), l.165-166 (event, 1.5 sigma below NR median, position, S1
partitioning consistent with z, S2 shape point-like), l.178 (neutron tagging 92 +- 4 %), l.184-196 (MSSI, 94 +- 2 %
prompt-veto tagging, 100 % rate uncertainty), l.210-246 (Table I), l.296-305 (Discussion: 57Co 25 min before,
AmBe 8 June, muons 41/127 min, mixed flow), l.687-695 (waveform: PSD inconclusive; rise time less consistent with
RFR; S2 shape matches alpha SS template; TBA consistent with z; hit pattern 2 sigma in z, 1 sigma in (x,y); pattern
inconsistent with RFR template, cannot distinguish SS from wall-MSSI template), l.710-724 (MSSI topologies,
validation), l.787-798 (neutrons, muon UL 4.6e-4). Dossier §2 (handle table), §5 (initial probabilities: B 0.19,
C 0.13, D 0.08, E 0.05, F 0.18, DM 0.20). `lzcommon.LZ` dict for the veto efficiencies and event coordinates.

| H | handle | central | range | source |
|---|---|---|---|---|
| NR | rate | s | scan 0.001-2.4 | LZ L10 fit 1.0 (+1.4 -0.7); P045 lower edge 0.105 |
| NR | E | 0.21 | 0.15-0.45 | P052 L10 panel split 0.21/0.59/0.21; P038 (delta >= 350 keV puts more above 500 phd) |
| NR | B | 0.90 | 0.85-0.95 | Gaussian 0.954; LZ Fig. 5 L10 d-curve heavier-tailed (P052: 0.070 vs 0.044 in the event bin) |
| NR | P, W, V, T | 1 | — | uniform; all pulse checks passed; P042 P(not clean) <= 3e-5 for Z/A'/contact models; T = 1.4-2.6 for inelastic (P006) as a variant |
| WALL | rate (pre-veto) | 0.080 x 0.62 | k 0.2-1.64 | 0.0048/0.06; P004 k_ML = 0.62, k < 1.64 (95 %) |
| WALL | E | 0.103 | 0.08-0.13 | P004 Fig. 5 digitisation (5.06e-4 of 4.9e-3 at S1c > 500; flat in S1c) |
| WALL | B | 0.343 | 0.30-0.40 | P004: 1.74e-4 of 5.06e-4 within +-2 sigma_NR (flat in d) |
| WALL | P | 1/15.1 | 1/50-1/2.3 | P070 LR(uniform/wall) 15.1 equal-mix; 49.9 wall source; 2.3 bottom/cathode source |
| WALL | W | 1 | 0.3-1 | LZ: hit pattern cannot distinguish SS from wall-MSSI template; TBA constraint on the wall deposit's z unquantified |
| WALL | V | 0.06 | 0.04-0.08 (not sampled; inside k) | LZ 94 +- 2 % |
| RFR | rate (pre-veto) | 1e-4/0.06 | x0.7-1.4 | LZ RFR ROI expectation (P033/P070); HE-SB validation 21.5 vs 18 |
| RFR | E, B | 0.103, 0.343 | 0.05-0.20, 0.30-0.40 | as wall; 204 +- 65 keV RFR deposit broadens S1c |
| RFR | P | 1/1.45 | 1/60-1/1.35 | P070: detector-gamma RFR map 1.45; 214Pb E2 profiles 17-60 |
| RFR | W | 0.01 | 1e-3-0.1 | LZ: pattern inconsistent with RFR template, rise time less consistent; P077 6-7.6 sigma (toy) |
| ER | rate | 1700 | fixed | Table I total |
| ER | E | 5.5e-6 | 4e-6-7e-6 | ER at S1c > 500 inside ROI = 0.0106 - MSSI - acc - nu = 9.4e-3 |
| ER | B | 4.7e-3 | 7.4e-5-0.30 | leakage to d ~ -1.5: P056 Gaussian 7e-7 .. P010 flat extrapolation 2.8e-3 (P061 adopted 2.8e-3); LZ model 0 |
| ER | T | 1 | 1-3 | 125I sub-case: activity at +8.4 d is 0.2-0.4 of peak, 2-4x the run-average time density (P010, P029) |
| ACC | rate | 2.47 | x0.3-3 | P022 (Table I 2.6); UDT zero above 495 phd: k_ML = 0, k < 183 |
| ACC | E | 2.55e-4 | 2e-4-3e-4 | P022: 6.3e-4 at S1c > 500 |
| ACC | B | 0.236 | 0.20-0.30 | P022: 1.49e-4 within +-2 sigma |
| ACC | P | 1 | 1-7.6 | uniform; cathode-emission slab favours the event's drift by 1/0.13 (P070 adverse) |
| ACC | W | 0.1 | 0.01-1 | isolated S1 must come from z ~ 26 cm (TBA, hit pattern); isolated S1s are RFR/bottom-array dominated (P022); unknown whether LZ's cuts already enforce this |
| NEU | rate (pre-veto) | 0.118/0.08 | fixed | LZ detector-NR fit interval upper edge |
| NEU | E | 6.2e-5 | 2.7e-5-4.4e-3 | P013 F(200-270|ROI SS) 5.1e-5 + muon 1.3e-5 (P049); stacked-conservative 5e-4 total |
| NEU | B | 0.95 | fixed | genuine NR |
| NEU | P | 1/1.4 | 1/1.7-1/1.1 | P070 exp(-d/14 cm) from surfaces |
| NEU | V | 0.08 | 0.04-0.12 (not sampled) | LZ 92 +- 4 % |
| ART | rate | 0.1 | 0.01-0.5 | pi_U, P061 hyper-prior for an LZ-specific unmodelled class |
| ART | E (joint) | 0.006 | 0.002-0.01 | P041 residual artefact likelihood (prior 0.18 -> 0.006) |
| ART | P | 1 | 1-3 | lifetime-type channels favour long drift (P041) |
| NU | rate | 0.167 | fixed | Table I atm-nu 0.11 + 8B/hep 0.057 |
| NU | E | 2.0e-4 | 6e-5-3e-3 | P019: coherent tail 3.4e-5 at S1c > 500; incoherent Fermi-recoil channel up to 5e-4 |
| NU | B | 0.95 | fixed | genuine NR |

LZ-model case (i) differences: k = 1 for WALL/RFR/ACC; ER B = 0 (LZ's ER PDF is truncated at +1 sigma_NR); NEU
E = 0 (LZ's Fig. 5 NR curve digitises to zero at S1c > 500; 19F(alpha,n) endpoint 191 keV, P013); ART rate 0.

## 3. Results

### 3.1 Neighbourhood expectations and the (S1c, d) likelihood ratio
LZ model, S1c > 500 phd and |d| <= 2 sigma_NR: NR 0.189 s; WALL 1.70e-4; ACC 1.49e-4; NU 3.2e-5; RFR 3.5e-6;
ER, NEU, ART 0. Sum 3.54e-4 (P070 used 3.6e-4). Corpus central sum 9.4e-4 (ART 6.0e-4, WALL 1.06e-4, ACC 1.49e-4,
ER 4.4e-5, NU 3.2e-5, NEU 7e-6, RFR 3.5e-6). The (S1c, d)-only likelihood ratio per signal event is 533 for the
+-2 sigma neighbourhood and 372 for the event bin (0.21 x 0.070 / 3.95e-5), bracketing P052's 265 from a 25 phd cell.

### 3.2 Posteriors (Table `P093_posteriors.csv`; Fig. 1)
s = 1, central (MC median [68 %]):

| H | i-a LZ (S1c,d) | i-b LZ + handles | ii corpus + handles |
|---|---|---|---|
| NR | 0.998 | 1.000 | 0.996 (0.994 [0.984, 0.998]) |
| WALL | 9.0e-4 | 5.9e-5 | 3.7e-5 (2.0e-5 [5.7e-6, 6.9e-5]) |
| RFR | 1.9e-5 | 1.3e-7 | 1.3e-7 (1.5e-8 [2.3e-9, 1.0e-7]) |
| ER | 0 | 0 | 2.3e-4 (3.1e-4 [1.9e-5, 5.1e-3]) |
| ACC | 7.9e-4 | 7.9e-5 | 7.9e-5 (1.7e-4 [3.0e-5, 9.4e-4]) |
| NEU | 0 | 0 | 2.6e-5 (1.0e-4 [1.8e-5, 5.9e-4]) |
| ART | 0 | 0 | 3.2e-3 (2.3e-3 [5.8e-4, 9.5e-3]) |
| NU | 1.7e-4 | 1.7e-4 | 1.7e-4 (2.9e-4 [7.6e-5, 1.1e-3]) |

Background-only posterior (conditional on "not NR", independent of s; `P093_background_only_posterior.csv`):
i-a: WALL 0.48, ACC 0.42, NU 0.09, RFR 0.01. i-b: NU 0.55, ACC 0.26, WALL 0.19, RFR 4e-4. ii: ART 0.85, ER 0.06,
NU 0.05, ACC 0.02, WALL 0.010, NEU 0.007, RFR 3.5e-5.

Break-even s_1/2 (P(NR) = 1/2): i-a 1.9e-3 (MC 1.5e-3 [1.0e-3, 2.2e-3]); i-b 3.1e-4 (4.0e-4 [2.0e-4, 1.1e-3]);
ii 3.7e-3 (6.0e-3 [2.2e-3, 1.6e-2]). P(NR) in case ii at s = 0.001 / 0.01 / 0.105 / 1 / 2.4: 0.21 / 0.73 / 0.966 /
0.996 / 0.998 central; MC medians 0.14 [0.06, 0.32] / 0.63 [0.38, 0.82] / 0.95 [0.87, 0.98] / 0.994 / 0.998 (Fig. 4).

### 3.3 Odds matrix (Fig. 2; `P093_odds_matrix_central.csv`, MC bands in `P093_odds_pairs_mc.csv`)
log10 odds NR : H at s = 1, central (MC median [68 %]): WALL +4.43 (+4.69 [4.15, 5.24]); RFR +6.89 (+7.81 [6.99, 8.63]);
ER +3.64 (+3.51 [2.28, 4.73]); ACC +4.10 (+3.77 [3.03, 4.53]); NEU +4.58 (+3.99 [3.22, 4.74]); ART +2.50 (+2.63
[2.02, 3.24]); NU +3.77 (+3.55 [2.96, 4.12]). Among backgrounds: ART : WALL +1.9, ART : ER +1.1, ART : ACC +1.6,
ART : NU +1.3, ER : ACC +0.5, WALL : ACC -0.3, WALL : RFR +2.5, NEU : WALL -0.15. LZ case (i-b) matrix in
`P093_odds_matrix_LZ_ib.csv` (NR : WALL +4.2, NR : ACC +4.1, NR : NU +3.8, NR : RFR +6.9).

### 3.4 Most discriminating handle per pair (`P093_pair_discriminants.csv`)
NR-WALL: rate +1.30, veto +1.22, position +1.18, band +0.42, energy +0.31 — no single dominant handle; the product
of three ~x15-20 factors makes the 4.4 dex. NR-RFR: rate +2.78, pulse shape +2.00, veto +1.22 (position only +0.16).
NR-ER: energy x band jointly +6.9 dex (E +4.58, B +2.31) against rate -3.23. NR-ACC: energy +2.92 (rate -0.39,
band +0.58, pulse shape +1.0). NR-NEU: energy +3.53, veto +1.10. NR-ART: the P041 residual (+1.54) and rate (+1.0).
NR-NU: energy +3.01, rate +0.78. WALL-RFR: pulse shape +2.0. WALL-ACC: energy +2.61 vs rate -2.71 cancel; band
+0.16, position -1.18, pulse shape +1.0. ER-ACC / ER-NU: rate (+2.8 / +4.0) against band (-1.7 / -2.3).
ACC-NEU and NEU-NU: veto (+-1.1). Timing never exceeds +-0.4 dex for any pair (variant: inelastic 1.4-2.6).

### 3.5 Which unpublished handle would change the odds most (`P093_unpublished_handles.csv`)
Binary-outcome model: P(bkg-like | H in target) = 0.8, else 0.8/LR. Shift of log10 odds(NR : all) if the outcome is
NR-like / bkg-like (s-independent): S1 pulse shape (targets WALL, RFR, ER, ACC, ART; LR 2-5, P077) +0.46 / -0.44;
artefact audit (hour-resolved lifetime, 83mKr S2c-map cell, per-PMT S2 footprint; LR 1.3-5, P041) +0.19 / -0.21;
S2 width vs cathode-drift template (ACC slab; LR 3-20) +0.01 / -0.08; TBA + per-PMT hit pattern vs wall/isolated-S1
templates (LR 1-3) +-0.01; (x,y) vs gamma-source map (LR 1-3) +-0.001; everything together +0.49 / -0.63. Expected
information gain at the break-even s = 3.7e-3 (posterior entropy 1.09 nats): PSD 0.123 nats, artefact audit 0.068,
S2 width 0.020, TBA/pattern 0.004, (x,y) 0.000, all 0.172; at s = 0.01: 0.115 / 0.049 / 0.012 / 0.002 / 0.000 / 0.163;
at s = 1 all <= 0.005 nats because P(NR) is already 0.994. A bkg-like waveform verdict at s = 0.01 would lower P(NR)
from 0.63 to 0.28, an NR-like one raise it to 0.82.

### 3.6 Variants (P(NR) at s = 1; central / MC median [68 %])
baseline 0.996 / 0.994 [0.984, 0.998]; inelastic timing 1.4-2.6: 0.998 / 0.997; ER at P010/P061 2.8e-3: 0.982 / 0.975
[0.960, 0.985] (ER becomes the leading background at 0.014); ER at P056 Gaussian 7e-7: 0.997; no artefact class:
0.999 / 0.998; ACC without slab loophole or TBA penalty: 0.996; WALL adverse (k = 1.64, bottom-source LR 2.3): 0.996;
NEU stacked-conservative 5e-4: 0.994 / 0.993. The two leading backgrounds (ART, ER) span 3 decades in the corpus;
their ranges, not the modelled backgrounds, set the 68 % band on P(NR).

## 4. Validation and robustness
* Sum of LZ neighbourhood expectations 3.54e-4 reproduces P070's 3.6e-4 (shares WALL 48 %, ACC 42 %, NU 9 %, RFR 1 %).
* (S1c, d)-only LR 372-533 per signal event brackets P052's 265 (different cell definitions).
* Case (i-b) with P070's central position LRs reproduces P070's qualitative finding: position removes wall/RFR MSSI
  and leaves accidentals and atmospheric neutrinos as the residual modelled background (NU 55 %, ACC 26 %).
* P(NR) in case ii at s = 1 (0.996) is consistent with the P001 Bayes-factor picture (B10 ~ 100-500 at b ~ 1e-3
  to 2e-4 per unit signal); the break-even s ~ 4e-3 to 6e-3 is the corpus-wide effective background 9e-4 divided
  by the NR neighbourhood fraction 0.19 times MC broadening.
* MC medians differ from centrals where a bracket is asymmetric in log space (RFR position, ACC slab, NEU energy);
  both are reported.

## 5. Failed or abandoned approaches
* Treating P033's region-integrated f_pos (0.98 %) as an additional factor on top of P070's density ratio double-counts
  the position information; abandoned in favour of the class-conditional density ratio alone.
* A per-25-phd energy slice for the NR signal (to match P052's cell exactly) needs the within-panel S1c shape of every
  background (DR-002); we kept the +-2 sigma, S1c > 500 neighbourhood.
* An explicit veto factor for bulk-TPC hypotheses (0.97 random-veto survival) cancels identically and was dropped.
* A first plotting pass failed on negative error-bar lengths from clipped zeros (LZ case has exact zeros); fixed.

## 6. Extended discussion
The topological handles do three different jobs. Energy (the S1c > 500 phd panel) and the veto remove the modelled
backgrounds: every one of them (wall MSSI, accidentals, neutrinos, neutrons) is a *population* that lives at low S1c
or gets tagged, and the corpus's revisions (P004 k < 1.64; P022 k < 183; P013/P049/P063 neutrons <= 5e-4; P019
neutrinos <= 5e-4) do not change that picture. Position (P070) and pulse shape (LZ, P077) then re-order the modelled
backgrounds: wall MSSI falls by 15, RFR MSSI by 100-1000, so the modelled residual becomes accidentals and atmospheric
neutrinos, which are uniform and single-site exactly like a DM recoil. Neither the band distance (-1.5 sigma; a 6 %
NR quantile, P024) nor the date (T <= 2.6 for inelastic, 0.98 elastic, P006) moves any pair by more than half a decade.

What the modelled handles cannot touch are the two unmodelled classes: a charge-poor ER (recombination or EC tail:
7e-7 to 2.8e-3 events, P010/P056) and an instrumental artefact (P041's 0.006 residual times an unknown prior weight).
In the corpus case they carry 91 % of the background posterior, and the NR : ART odds (+2.5 dex) is the smallest
margin in the matrix. Both are ER-like in their S1, so the S1 pulse shape is the one published-but-unquantified
handle that addresses them together; P077 bounds its power to LR 2-5. That is why the waveform-level quantities
are worth requesting (DR-003) even though no single one can flip the classification: together they move log-odds
by 0.5-0.6 dex, the largest lever left, and they would move ER and ART, not the already-negligible MSSI.

The posterior itself is a statement about s: at LZ's fitted s = 1 the event is a nuclear recoil with probability
0.994 [0.984, 0.998]; at a prior-weighted s ~ 0.01 (P061's community prior gives P(DM) median 0.015 by a different
route) it is 0.63 [0.38, 0.82]; the break-even is s ~ 4e-3 to 6e-3. Topology therefore establishes "single nuclear
recoil" far more securely than it establishes "dark matter": the irreducible NR-like background (atmospheric
neutrinos, 3e-5 to 5e-4) and the prior on s, not any topological handle, decide the latter.

## 7. Figures
* Fig. 1 `figures/P093_fig1_posteriors.png` — ranked posterior, three cases, s = 1; MC median, 68 % band, central tick.
* Fig. 2 `figures/P093_fig2_odds_matrix.png` — log10 odds matrix, corpus case, central.
* Fig. 3 `figures/P093_fig3_handle_contributions.png` — per-handle log10 L(handle|H)/L(handle|NR) and totals.
* Fig. 4 `figures/P093_fig4_PNR_vs_s.png` — P(NR) versus s for the three cases with 68 % bands.

## 8. Result tables
`P093_handle_table.csv`, `P093_posteriors.csv`, `P093_background_only_posterior.csv`, `P093_PNR_vs_s.csv`,
`P093_odds_matrix_central.csv`, `P093_odds_matrix_LZ_ib.csv`, `P093_odds_pairs_mc.csv`, `P093_pair_discriminants.csv`,
`P093_unpublished_handles.csv`, `P093_results.json`, `run_log.txt`.

## 9. References
LZ Collaboration, arXiv:2609.02823 (2026). G. Cowan, K. Cranmer, E. Gross, O. Vitells, Eur. Phys. J. C 71, 1554
(2011). R. E. Kass and A. E. Raftery, J. Am. Stat. Assoc. 90, 773 (1995). D. V. Lindley, Ann. Math. Stat. 27, 986
(1956). T. Sellke, M. J. Bayarri, J. O. Berger, Am. Stat. 55, 62 (2001). J. F. C. Kingman, Poisson Processes (Oxford,
1993). Corpus: dossier 00; P004, P006, P009, P010, P013, P019, P022, P024, P027, P029, P033, P036, P041, P042, P045,
P049, P052, P053, P056, P061, P063, P070, P077. (P073 had not appeared when this record was written.)

## 10. Tools and provenance (mirrors provenance/P093.json)
Agent tools: Read x39 (PAPER_GUIDE; ledger; DR-002; two persisted ledger/P070 dumps; fulltext.tex l.160-210,
210-250, 286-306, 686-698, 708-722, 786-806; dossier l.9-69; papers P004, P033, P053, P070, P009, P024, P056, P010,
P022, P041, P013, P063, P019, P042, P036, P029, P061, P027, P052, P077, P006, P049; four figures), Bash x18
(listings, ledger compaction, PROMPT §4, tex/dossier greps, lzcommon dump, P070/P052/P004/P022 result files, palette
grep, two script runs, summaries, five word-count checks), Write x6 (paper written twice), Edit x21 (script 9, paper 11,
JSON 1), Skill x1 (dataviz; JS validator skipped per rule 2).
Software: python 3.12.13; numpy 2.5.3 (default_rng(93), quantile, prod, log10); pandas 3.0.5; matplotlib 3.11.2 (Agg,
TwoSlopeNorm, LinearSegmentedColormap); common/lzcommon.py (LZ dict). Scripts: output/code/P093_topology_odds.py
(`.venv/bin/python output/code/P093_topology_odds.py`, 6 s). Recalled knowledge: 1 item (classification posterior
for a marked Poisson process, certain). Datasets: none. Data requests: DR-003 (filed, pending). WimPyDD files: none.
