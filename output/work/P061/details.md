# P061 — Prior sensitivity of the dark-matter posterior: how robust is any probability statement about the LZ event?

Simulated date 2026-09-12. Author profile: Bayesian epistemology-minded physicists. Category STAT (physics.data-an, cross-list hep-ph).
Script: `output/code/P061_prior_sensitivity.py` (run from the simulation root with `.venv/bin/python`; 35 s). Results: `output/work/P061/P061_results.json`,
`P061_table1_channels.csv`, `P061_table3_dossier_update.csv`, `P061_map.npz`, `run_log.txt`; figures in `output/work/P061/figures/`.

## 1. Motivation and framework

P001 showed that for one event the posterior probability of dark matter is set not by the modelled background but by the prior odds of DM against
an *unmodelled* background: P(DM) > 1/2 needs pi_DM > 10 pi_U L_U. Since then the corpus has (i) quantified the residual likelihood of every specific
background channel (P004, P010, P013, P019, P022, P029, P033, P036, P040, P041, P049), (ii) marginalised the DM Bayes factor over LZ's model space
(P027: B_DM = 16.4–28.7, range 6.6–45.7 over coupling priors and the continuous-delta class), (iii) confirmed the trials factor (P008: N_eff = 12.2) and
(iv) shown that non-zero lower limits carry no extra evidence (P045). We ask how P(DM | event) depends on the judgemental inputs that remain:

- (a) pi_DM: prior probability that DM produces any event in this window (1e-3–0.3);
- (b) pi_U and L_U: prior mass and likelihood of an unmodelled background/artefact (pi_U 0.01–0.5; L_U = L_acc + L_rest with L_rest 1e-3–0.3);
- (c) the model prior within DM: P027's B_DM variants (6.6–45);
- (d) a "garden of forking paths" penalty f_FP (1–10) for the non-blind choices (FV shrink 14.5%, 5.4 t MSSI sideband defined with the event in hand,
  salting failure in the signal region; dossier Section 4.2), applied as a divisor of the DM Bayes factor;
- (e) the historical base rate r of ~3 sigma single-event anomalies that turned out to be signal (P083 will estimate it; here parametrised 0.01–0.3).

### 1.1 The three-hypothesis posterior

Following P001, hypotheses H_DM, H_B (modelled background), H_U (unmodelled background or artefact) with priors pi_DM, pi_B = 1 - pi_DM - pi_U, pi_U.
Likelihoods of the observed datum (one event at the observed location, none elsewhere in the high-energy band) up to a common factor:

    L_B  = b_eff                          (expected modelled-background count in the event's neighbourhood)
    L_DM = (B_DM / f_FP) * b_eff          (P027's model-marginalised Bayes factor is defined relative to L_B; f_FP divides it)
    L_U  = L_acc + L_rest                 (expected event-like count per 2.84 t yr under the unmodelled hypothesis)

    P(DM | event) = pi_DM L_DM / ( pi_DM L_DM + pi_B L_B + pi_U L_U ).                                           (1)

Posterior odds O = P(DM)/(1 - P(DM)) = pi_DM L_DM / (pi_B L_B + pi_U L_U). The prior pi_DM at which P(DM) = P_t is analytic:

    pi_DM(P_t) = P_t [ (1 - pi_U) b + pi_U L_U ] / [ L_DM (1 - P_t) + P_t b ].                                    (2)

Note that B_DM scales as 1/b_H in P027, so L_DM = B_DM b_H is the anchored quantity (the DM marginal likelihood of the event, 9.35e-3 at the central
values); varying b_eff therefore only moves L_B (Section 5.3).

### 1.2 Reference-class (base-rate) parametrisation

A typical ~3 sigma anomaly of the historical class carries a Bayes factor of order the Sellke–Bayarri–Berger cap at its global p,
B_typ = -1/(e p ln p) = 14.60 at p = 4.7e-3 (LZ's 2.6 sigma global; P001 quotes 14.7 with a slightly different p). If a fraction r of such anomalies
proved real, the class-calibrated posterior for the LZ event is

    P_ref = r B' / ( r B' + 1 - r ),   B' = (B_DM / f_FP) / B_typ.                                                    (3)

Setting P_ref equal to (1) and solving (2) gives the pi_DM implied by a given base rate — a translation between the two ways of stating a prior.

### 1.3 Dossier Section-5 table update

Rule: posterior_i ∝ P_dossier,i × L_i, where L_i is the corpus-computed likelihood of the event under explanation i (Table 3), the DM rows I/J/K share
L_DM with internal split 0.70/0.25/0.05 from P027's class masses (inelastic 0.58–0.85, q^2-suppressed spin 0.11–0.30, everything else, incl. elastic O1
< 1e-4, ~0.05). H (non-DM new physics) is given L = b_eff, i.e. it fits the event no better than a background of the same rate (P019: every exotic
neutrino interaction needs >= 790 low-energy recoils per window event; P040: boosted/fast light DM needs >= 350). Caveat: the dossier weights were
assigned with the event in hand, so multiplying them by event likelihoods partly double-counts the event's location information; we therefore present
the result as one group's judgement and give the optimistic/pessimistic and hyper-prior ranges.

### 1.4 Value of information

For a future measurement M with outcomes m and likelihoods P(m | h), h in {DM, B, U}, the posterior at hyper-parameter point theta becomes
P(h | theta, m) ∝ P(h | theta) P(m | h). The expected information gain on the binary question DM-vs-not is the mutual information

    IG(theta) = H2[P(DM | theta)] - sum_m P(m | theta) H2[P(DM | theta, m)],   H2(p) = -p ln p - (1-p) ln(1-p),      (4)

averaged over the community hyper-prior (Lindley's expected information). We also report the expected 68 % width of P(DM) across the community after the
measurement, sum_m P̄(m) W68[P(DM | ., m)] with P̄(m) = E_theta P(m | theta) and W68 weighted by P(m | theta): a direct measure of how much the
prior-sensitivity spread would shrink. Predictives (Table 4):

- LZ next exposure, k = 2.38 (6.76 t yr; P020): H_DM: P027's model-averaged predictive P(0/1/>=2) = 0.58/0.22/0.20 (variant: P020 L10 plug-in
  0.092/0.220/0.688); H_B: Poisson with mu_B = k b_eff = 1.36e-3; H_U: a share f_trans is a one-off effect (predicts N = 0) and 1 - f_trans a steady
  rate with a log-uniform rate prior, whose posterior after one event is Exponential(1) and whose predictive is negative-binomial(r = 1):
  P(0) = 1/(1+k) = 0.296, P(1) = k/(1+k)^2 = 0.208, P(>=2) = 0.496. Central f_trans = 0.5; 0 and 1 as brackets.
- XENONnT + PandaX-4T 270 keV reanalysis (k_X = (3.1 + 1.54)/2.84 = 1.63): H_DM: 1.63 events at LZ's best-fit coupling (P035) scaled by P027's
  model-average/L10 ratio 0.92/2.4 -> mu_X = 0.62 (P(>=1) = 0.46); variant at the best fit (1.63, P(>=1) = 0.80); H_B: 1e-3; H_U: a share f_spec = 0.7 of
  steady unknowns is LZ-specific (artefact, MSSI geometry, accidentals, activation), the rest generic (ER-leakage physics), so P(>=1 | U) =
  (1 - f_trans)(1 - f_spec)(1 - 1/(1 + k_X)) = 0.093.
- Wall-MSSI calibration, outcome k > 100 vs k <= 2: P(k > 100 | U) = share of the MSSI channel in L_U = 3.3e-6 / L_U; 1e-4 under DM and B.
- LZ internal artefact audit (83mKr S2c map, cathode-emission S2 test; P041, P022), outcome "artefact found": P = 0.006 / L_U under U; 0.01 false
  positive under DM and B (judgement).

## 2. Inputs and sources

| quantity | value | source |
|---|---|---|
| b_eff (modelled background in the event's neighbourhood; gives 3.4 sigma for L10) | 5.7e-4 (range 2e-4–1.1e-3) | P016 anchor 5.70e-4; P001 bracket |
| B_DM central / class priors / range | 16.4 / 28.7 / 6.6–45 | P027 (uniform-s 6.6; continuous-delta class B 45.7) |
| N_eff, p_global | 12.2, 4.7e-3 | P008; LZ paper |
| LZ exposure; next exposure ratio; XENONnT+PandaX exposure ratio | 2.84 t yr; 2.38; 1.63 | LZ paper; P020; dossier/P005 |
| wall/RFR MSSI residual at the position class | 2e-6 × k, k < 1.64 (95 %) -> 2e-6 / 3.3e-6 | P033 (k_req 6.4e4, k_req/k_allowed 3.9e4); P004 |
| accidentals in ±2 sigma_NR at S1c > 500 phd | 1.5e-4 (as modelled); UDT allows k < 183 -> 2.7e-2 | P022 (10 % origin needs k = 708, excluded at 4.3 sigma) |
| ER leakage incl. 124Xe/125I | 2.8e-3 (flat extrapolation of Fig. 5 tail); <= 1e-2 | P010 ("disfavoured at <~ 0.3 %") |
| neutrons of any origin | 6.6e-6 central; <= 5e-4 stacked-conservative | P013; P049 (muon-induced 1.3e-5) |
| detector artefact / partial charge loss | 0.006 residual; × 3 upper (our judgement) | P041 |
| activation, MSSI-like | 1.5e-4 per run (9e-6–8e-3) | P029 |
| atmospheric-nu incoherent lone NR | 1.7e-4 (8e-6–5e-4) | P019 |
| nuclear-excitation hybrid; boosted/fast light DM | 0 | P036; P040 |
| P027 class masses (for the I/J/K split) | inelastic 0.58/0.85; q^2-spin 0.30/0.11; O1 elastic < 1e-4 | P027 |
| P020 P(DM | N = 0..4) for comparison | 0.011/0.062/0.358/0.862/0.989 | P020 |
| dossier Section-5 table | A 0.10, B 0.19, C 0.13, D 0.08, E 0.05, F 0.18, G 0.05, H 0.02, I 0.10, J 0.06, K 0.04 | 00_evidence_dossier.md |

**Table 1 (accounted channels).** L_acc = sum of central values = 9.279e-3 (dominated by artefacts 6.0e-3 and ER 2.8e-3); upper bracket
L_acc,up = 0.066 (dominated by accidentals at the 95 %-allowed k = 183: 0.027; artefacts 0.02; ER 0.01; activation 0.008). Summing channel residuals
treats H_U as the union of channels each carrying the whole pi_U; since a proper mixture would give L_U = sum_c w_c L_c <= max_c L_c, the sum overstates
L_U by at most a factor L_acc/max_c L_c = 1.55 (central) and is conservative against DM.

Community hyper-prior: log-uniform on pi_DM in [1e-3, 0.3], pi_U in [0.01, 0.5], L_rest in [1e-3, 0.3], f_FP in [1, 10], B_DM in [6.6, 45];
b_eff fixed at 5.7e-4 and L_acc at its central value (both varied in Section 5.3). N = 400 000 samples, seed 20260912.
Central point for single-number statements: pi_U = 0.1, f_FP = 3, B_DM = 16.4 (and pi_DM = 0.03, L_rest = 0.01 where needed).

## 3. Validation

Setting B_DM -> 1 + E/b with E = 0.100 (P001's uniform s in [0, 10]), b = 2e-4, f_FP = 1, L_acc = 0 reproduces P001's three-hypothesis posteriors:
(pi_U, L_U) = (0.01, 0.1): 0.456 vs 0.46; (0.1, 0.1): 0.0896 vs 0.09; and P001's two-hypothesis 0.835 vs 0.84 (pi_DM = 0.01). The first-order
sensitivity indices sum to 0.997, confirming that logit P is nearly additive in the log hyper-parameters (as expected from Eq. 1 when one term dominates
the denominator), and the analytic elasticities (Section 4.1) match the ±1 exponents of B_DM and f_FP exactly.

## 4. Results

### 4.1 The (pi_DM, L_rest) map (Fig. 1)

Central values (pi_U = 0.1, f_FP = 3, B_DM = 16.4, L_acc = 9.3e-3): P(DM) = 0.1 requires pi_DM >= 0.085 (L_rest = 0.01) or 0.40 (L_rest = 0.1);
P(DM) = 0.5 requires pi_DM >= 0.66 (L_rest = 0.01) — outside the scanned box and above any defensible prior; P(DM) = 0.9 would need pi_DM > 1.
Over the log-area of the box: 0 % has P > 0.5, 15.3 % has 0.1–0.5, 84.7 % has P < 0.1.
Optimistic corner (pi_U = 0.05, f_FP = 1, B_DM = 28.7): P = 0.1 at pi_DM = 0.010 (L_rest = 0.01) / 0.041 (0.1); P = 0.5 at 0.089 / 0.35; P = 0.9 at 0.63 / 2.5.
Box fractions 14.8 % / 35.6 % / 49.6 %. Pessimistic corner (pi_U = 0.3, f_FP = 10, B_DM = 6.6, L_acc = 0.066): P < 0.1 everywhere (P = 0.1 needs pi_DM = 5.9).
Full boundary table for L_rest = 1e-3 … 0.3 in `P061_results.json["pi_DM_boundaries"]`.

Elasticities d ln O / d ln theta at (pi_DM, pi_U, L_rest, f_FP, B_DM) = (0.03, 0.1, 0.01, 3, 16.4), where P(DM) = 0.0371:
pi_DM +1.007, pi_U -0.772, L_rest -0.413, f_FP -1.000, B_DM +1.000. So the posterior odds scale as pi_DM^1.0 pi_U^-0.77 L_rest^-0.41 (B_DM/f_FP)^1.

### 4.2 Community-prior marginal (Fig. 2)

| hyper-prior | median | 68 % (16–84 %) | 95 % | frac P > 0.5 | frac 0.1–0.5 | frac P < 0.1 |
|---|---|---|---|---|---|---|
| community prior (all five log-uniform) | 0.0153 | 0.0016–0.137 | 0.00019–0.495 | 0.024 | 0.178 | 0.797 |
| f_FP = 1 | 0.047 | 0.005–0.317 | | 0.081 | | 0.636 |
| f_FP = 10 | 0.005 | 0.001–0.044 | | 0.000 | | 0.928 |
| B_DM = 28.7 | 0.025 | 0.003–0.202 | | 0.040 | | 0.732 |
| B_DM = 6.6 | 0.006 | 0.001–0.054 | | 0.002 | | 0.909 |
| pi_DM = 0.2 (dossier DM total) | 0.173 | 0.038–0.459 | | 0.131 | | 0.347 |
| pi_DM = 0.01 | 0.010 | 0.002–0.037 | | 0.000 | | 0.973 |
| L_rest = 1e-3 | 0.036 | 0.005–0.243 | | 0.049 | | 0.684 |
| L_rest = 0.1 | 0.006 | 0.001–0.055 | | 0.003 | | 0.905 |
| pi_U = 0.01 | 0.051 | 0.007–0.309 | | 0.072 | | 0.626 |
| pi_U = 0.5 | 0.003 | 0.000–0.028 | | 0.000 | | 0.959 |
| L_acc upper bracket 0.066 | 0.007 | 0.001–0.060 | | 0.005 | | 0.897 |
| L_acc = 0 | 0.022 | 0.002–0.197 | | 0.044 | | 0.743 |
| b_eff = 2e-4 (L_DM fixed) | 0.019 | 0.002–0.172 | | 0.039 | | 0.766 |
| b_eff = 1.1e-3 (L_DM fixed) | 0.012 | 0.001–0.110 | | 0.013 | | 0.827 |
| linear-uniform hyper-prior | 0.012 | 0.003–0.055 | | 0.003 | | 0.916 |

Mean P(DM) = 0.071; mean binary entropy 0.171 nats. First-order variance shares of logit P(DM): pi_DM 0.559, pi_U 0.144, L_rest 0.140, f_FP 0.090,
B_DM 0.064 (linear P: pi_DM 0.36, pi_U 0.17, L_rest 0.16, f_FP 0.13, B_DM 0.06). The modelled background itself is nearly irrelevant: b_eff over its
full P001 bracket moves the median only between 0.019 and 0.012 (P001's asymptotic statement that B drops out as b -> 0).

### 4.3 Reference class

B' = (16.4/3)/14.60 = 0.374 at central values (range 0.045 for f_FP = 10, B_DM = 6.6 to 1.97 for f_FP = 1, B_DM = 28.7). P_ref and the implied pi_DM
(Eq. 2 at pi_U = 0.1, f_FP = 3, B_DM = 16.4):

| r | P_ref | pi_DM implied (L_rest = 0.01) | pi_DM implied (L_rest = 0.1) |
|---|---|---|---|
| 0.02 | 0.0076 | 0.0060 | 0.028 |
| 0.05 | 0.0193 | 0.0154 | 0.072 |
| 0.10 | 0.0399 | 0.0323 | 0.152 |
| 0.20 | 0.0856 | 0.0721 | 0.338 |

So a historical base rate of 5–20 % corresponds, in the structural model, to pi_DM = 0.015–0.07 (L_rest = 0.01), i.e. to the central region of our
hyper-prior; the dossier's DM total of 0.20 corresponds to r ~ 0.4 (P_ref = 0.2 needs r = 0.40 at B' = 0.374).

### 4.4 Dossier Section-5 update (Table 3; `P061_table3_dossier_update.csv`)

| explanation | dossier | L(event) central | posterior central | 68 % over hyper-prior | optimistic | pessimistic |
|---|---|---|---|---|---|---|
| A fluctuation of modelled background | 0.10 | 5.7e-4 | 0.014 | 0.002–0.021 | 0.011 | 0.002 |
| B wall/RFR MSSI mismodelled | 0.19 | 2.0e-6 | 1.0e-4 | 1.6e-5–1.4e-4 | 7.6e-5 | 2.5e-5 |
| C ER leakage / EC tail | 0.13 | 2.8e-3 | 0.092 | 0.015–0.134 | 0.073 | 0.052 |
| D neutron | 0.08 | 6.6e-6 | 1.3e-4 | 2.2e-5–2.0e-4 | 1.1e-4 | 1.6e-3 |
| E accidental coincidence | 0.05 | 1.5e-4 | 0.0019 | 3e-4–2.8e-3 | 0.0015 | 0.054 |
| F detector artefact + unknown unknowns | 0.18 | 0.006 + L_rest = 0.016 | 0.729 | 0.567–0.952 | 0.253 | 0.870 |
| G calibration-related | 0.05 | 1.5e-4 | 0.0019 | 3e-4–2.8e-3 | 0.0015 | 0.016 |
| H non-DM new physics | 0.02 | 5.7e-4 | 0.0029 | 5e-4–4.2e-3 | 0.0023 | 4.6e-4 |
| I inelastic DM | 0.10 | 3.12e-3 (× 0.70 of DM) | 0.110 | 0.016–0.205 | 0.460 | 0.0021 |
| J heavy WIMP, q-suppressed EFT | 0.06 | (× 0.25) | 0.039 | 0.006–0.073 | 0.164 | 7.6e-4 |
| K other DM | 0.04 | (× 0.05) | 0.008 | 0.001–0.015 | 0.033 | 1.5e-4 |
| **DM total** | **0.20** | **3.12e-3** | **0.158** | **0.023–0.293** | **0.657** | **0.0030** |

Central: L_rest = 0.01, f_FP = 3, B_DM = 16.4, channel likelihoods central. Optimistic: L_rest = 1e-3, f_FP = 1, B_DM = 28.7. Pessimistic:
L_rest = 0.1, f_FP = 10, B_DM = 6.6, channel likelihoods at their upper brackets. Hyper-prior range: L_rest, f_FP, B_DM drawn from the community prior
(60 000 samples), channel likelihoods central. For comparison, the coordinator-style judgement mapping quoted in our assignment (B 0.01, C 0.02,
D 0.005, E 0.01, F 0.05, G 0.005, H 0.01) leaves 0.89 for A + DM; our likelihood rule instead moves most of the eliminated mass into F because the
unaccounted L_rest = 0.01 exceeds every quantified channel except the artefact residual itself — the two mappings differ in where "unknown unknowns"
live, not in the eliminations (B, D, E, G are < 0.01 under both).

### 4.5 Value of information (Fig. 3; Table 4)

Hyper-prior mean binary entropy before: 0.1712 nats; 68 % width of P(DM) across the community: 0.137.

| measurement | outcome probabilities (community) | expected IG (nats, % of 0.171) | expected 68 % width after | central-point P(DM) 0.037 -> per outcome | P(U) per outcome |
|---|---|---|---|---|---|
| LZ 6.76 t yr, N = 0/1/>=2 (f_trans = 0.5) | 0.72/0.09/0.19 | 0.0078 (5 %) | 0.141 | 0.030/0.093/0.038 | 0.70/0.90/0.96 |
| same, H_U one-off (f_trans = 1) | 0.97/0.016/0.014 | 0.0521 (30 %) | 0.071 | 0.022/0.968/1.000 | 0.78/0/0 |
| same, H_U steady (f_trans = 0) | | 0.0053 | 0.137 | 0.048/0.049/0.019 | |
| same, L10 plug-in predictive (P020), f_trans = 0.5 | 0.69/0.09/0.23 | 0.0544 (32 %) | 0.177 | 0.005/0.093/0.119 | 0.71/0.90/0.88 |
| XENONnT+PandaX 270 keV, 0/>=1 (mu_X = 0.62) | 0.90/0.10 | 0.0273 (16 %) | 0.145 | 0.022/0.194 | 0.76/0.80 |
| same at LZ best-fit coupling (mu_X = 1.63) | 0.88/0.12 | 0.0738 (43 %) | 0.131 | 0.008/0.295 | 0.77/0.70 |
| wall-MSSI calibration, k <= 2 / k > 100 | 1.000/2e-4 | 0.0000 (0 %) | 0.137 | 0.037/0.024 | 0.77/0.85 |
| LZ internal artefact audit, nothing/found | 0.84/0.16 | 0.0106 (6 %) | 0.144 | 0.048/0.002 | 0.70/0.99 |

Reading: (i) LZ's own next exposure is informative about DM only to the extent that the unknown is a one-off effect. If it is a steady unmodelled rate,
one more event in 6.76 t yr is exactly what that rate predicts too (NB predictive P(>=1) = 0.70 vs 0.42 for P027's DM average), so N >= 2 at the
central point raises P(U) to 0.96 and P(DM) only to 0.04. The community spread can even *grow* (0.137 -> 0.141, 0.177) because a positive outcome
polarises: those with small pi_U move to P(DM) ~ 1, those with large pi_U stay near 0. (ii) A second detector is the only measurement that separates
DM from an LZ-specific unknown; its information gain is 0.027–0.074 nats (16–43 % of the entropy), and one XENONnT/PandaX high-energy event would
lift the central-point P(DM) to 0.19–0.30 (P035: 4.6–5.2 sigma combined as a frequentist statement). (iii) A wall-MSSI calibration carries no
information because the corpus has already reduced that channel to 3e-6 of L_U ~ 0.02. (iv) An internal artefact audit is worth 0.011 nats: finding an
artefact would settle the case (P(U) -> 0.99), but the community expects that outcome with only 16 % probability.

## 5. Robustness and variations

5.1 Hyper-prior shape: linear-uniform instead of log-uniform gives median 0.012, 68 % 0.003–0.055 (the linear prior puts more weight at large pi_U and
L_rest). 5.2 L_acc: zero (no accounted channels) 0.022 (0.002–0.197); upper bracket 0.066: 0.007 (0.001–0.060). 5.3 b_eff: 2e-4 / 1.1e-3 with L_DM
fixed: 0.019 / 0.012. 5.4 Union-vs-mixture treatment of H_U (Section 2): at most × 1.55 on L_acc, i.e. < 0.2 dex on the odds. 5.5 The I/J/K split
(0.70/0.25/0.05) affects only the DM sub-rows, not the total. 5.6 f_trans, f_spec in the VOI: Table 4 shows the full f_trans bracket; f_spec = 0.5
instead of 0.7 raises P(>=1 | U) for XENONnT from 0.093 to 0.155 and lowers its IG by roughly a quarter (not tabulated; direction follows from Eq. 4).

## 6. Failed or abandoned approaches

- A per-channel mixture model for H_U with dossier-relative weights w_c was drafted but dropped: it requires a prior weight for the "unaccounted"
  component that is not separable from L_rest, and the union treatment is conservative by a known bounded factor (Section 2).
- We considered folding the base rate r into the hyper-prior as a sixth parameter; since r and pi_DM are two parametrisations of the same prior
  (Section 4.3), this would double-count, so r is reported only as a translation.

## 7. Discussion

Every number that has been offered for "the probability that the LZ event is dark matter" — P001's 0.09–0.46, P020's 0.011 after a null next
exposure, the dossier's 0.20 — lies inside the 95 % range of the community prior (0.0002–0.50). The ordering of the drivers is instructive: the prior
pi_DM (56 % of the variance) and the two unknown-background parameters (28 %) dominate; the data-side quantities the corpus has argued about most —
B_DM (6 %) and even the forking-paths penalty (9 %) — matter least, and the modelled background not at all. The corpus's eliminations of wall MSSI,
neutrons, accidentals, activation and atmospheric neutrinos have been decisive for those channels (each now < 0.2 % of the posterior) but have moved
P(DM) little, because the posterior is now a contest between DM and residual ER/artefact/unknown channels whose likelihoods (0.003–0.02) are of the
same order as L_DM = 3e-3–9e-3. This is the Bayesian form of the frequentist statement that 2.6 sigma global is weak evidence: at 2.6 sigma the
prior does the work. The practical consequence is that only a second detector can move the posterior for everyone; LZ's own next exposure moves it
only for those who already believe the unknown must be a one-off.

## 8. Figures

- Fig. 1 `figures/P061_fig1_map.png`: P(DM | event) over (pi_DM, L_rest) for pessimistic / central / optimistic values of the other hyper-parameters;
  contours P = 0.1 (orange), 0.5 (black), 0.9 (aqua); dotted line at the dossier DM total 0.20.
- Fig. 2 `figures/P061_fig2_distribution.png`: left, distribution of P(DM) over the community hyper-prior (filled), with f_FP = 1 and pi_DM = 0.2
  variants (outlines); right, first-order sensitivity indices of logit P(DM).
- Fig. 3 `figures/P061_fig3_voi.png`: expected information gain (nats, and % of the 0.171 nats prior entropy) and expected community 68 % width after each
  measurement (dashed: before, 0.137).

## 9. References

LZ Collaboration, arXiv:2609.02823 (2026). R. E. Kass, A. E. Raftery, J. Am. Stat. Assoc. 90, 773 (1995). T. Sellke, M. J. Bayarri, J. O. Berger,
Am. Stat. 55, 62 (2001). A. Gelman, E. Loken, Am. Sci. 102, 460 (2014) ("garden of forking paths"). D. V. Lindley, Ann. Math. Stat. 27, 986 (1956)
(expected information of an experiment). R. Trotta, Contemp. Phys. 49, 71 (2008). Corpus: 00_evidence_dossier.md; P001, P004, P008, P010, P013, P016,
P019, P020, P022, P027, P029, P033, P035, P036, P040, P041, P045, P049.

## 10. Tools and provenance (mirrors provenance/P061.json)

Agent tools: Read (PAPER_GUIDE.md; dossier Sections 3–6; ledger dump (50 rows, two pages); papers P001, P008, P016, P027, P045; provenance/P027.json as
format template; figures P061_fig1/2/3 for visual checks) — 15 calls; Bash (grep dossier sections; pandas dump of the ledger; grep lzcommon function
index + ENVIRONMENT_versions + ls work dirs; grep dataviz palette.md; six runs of the P061 script incl. one SyntaxError and one KeyError fix) — 10 calls;
Write (script, details.md, provenance JSON, paper) — 4; Edit (script: voi signature, table column, IMPLIED dict keys, figure labels ×3, f_trans = 1 row,
b_eff variant, print keys) — 9; Skill dataviz — 1.
Software: python 3.12.13 (.venv); numpy 2.5.3 (default_rng, geomspace, meshgrid, quantile, bincount, searchsorted); scipy 1.18.1 (stats imported;
posterior algebra is closed-form); pandas 3.0.5 (DataFrame, to_csv); matplotlib 3.11.2 (Agg; pcolormesh, contour, hist, barh); common/lzcommon.py (LZ
dict for the exposure). Posterior, elasticities, boundaries, negative-binomial predictive and mutual information derived by hand (Eqs. 1–4).
Recalled knowledge (6): Kass–Raftery scale (certain); SBB bound formula (certain); Lindley expected information gain / mutual information (certain);
Gelman–Loken "garden of forking paths" (certain); Poisson–Gamma -> negative-binomial predictive (certain); historical reference class of ~3 sigma
single-event anomalies (CDMS-II Si, CRESST-II, XENON1T ER excess, 750 GeV diphoton) used only qualitatively (likely). Datasets: none. Data requests: none.
WimPyDD-generated files: none.
