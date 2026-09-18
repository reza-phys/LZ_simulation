# P001 — How much evidence is one event? A Bayesian and frequentist reassessment of the LZ 248 keV recoil

*Complete research record. Simulated arXiv date 2026-09-03. Author profile: astrostatistics / particle-statistics group. Category STAT; primary arXiv category physics.data-an (cross-list hep-ex). All numbers below are produced by `output/code/P001_bayes_single_event.py` (results in `output/work/P001/P001_results.json` and the CSV tables listed in Sec. 9) unless marked [paper], [dossier] or [recalled].*

## 1. Motivation and framework

LZ (arXiv:2609.02823) reports one nuclear-recoil-like event (S1c = 540.1 phd, S2c = 9268 phd, E_R ≈ 248 keV) in 2.84 t·yr and quotes a maximum local significance of 3.4σ over 616 tested signal models (293 distinguishable spectra) and a toy-MC global significance of 2.6σ [paper, abstract and supplement "Log-Likelihood Function and the Look Elsewhere Effect"]. The only quantitative statement of evidence is therefore a pair of p-values. We ask what these mean as *evidence for dark matter*, i.e. as a Bayes factor and a posterior probability, and how strongly the answer depends on (i) the modelled background in the region where the event fell, (ii) the prior on the signal strength, (iii) the prior probability that dark matter shows up in this window, and (iv) an explicit "unknown background" alternative.

The framework is deliberately minimal: a **Poisson-in-a-box** counting model. One event (n = 1) is observed in a region ("box") of the {S1c, log₁₀S2c} plane in which the modelled background expectation is b and a putative signal contributes s expected events with efficiency 1 inside the box. This caricatures the LZ unbinned extended likelihood (Eq. S-likelihood, supplement); Sec. 3.2 shows exactly how the two are related and why the box gives a *higher* significance than the paper's spectra-based likelihood.

Three background values are used throughout:

| b | Meaning | Source |
|---|---|---|
| 2 × 10⁻⁴ | modelled background within ±2σ of the NR median at S1c > 500 phd (order of magnitude from the Fig. 5 bottom panel, ≈ 5 × 10⁻⁵ per 0.5σ bin near −1.5σ) | [dossier Sec. 2, 3] |
| 1 × 10⁻³ | intermediate value; also ≈ the effective single-event background implied by the paper's 3.4σ (Sec. 3.2) | this work |
| 0.0106 ± 0.0008 | total integrated background of the whole S1c > 500 phd panel of Fig. 5 | [paper, Fig. 5 caption]; `lz.LZ["bkg_highS1_panel"]` |

## 2. Equations and derivations

### 2.1 Likelihood for one event
P(n = 1 | μ) = μ e^(−μ). With μ = s + b under H₁ and μ = b under H₀,

  LR(s) ≡ P(1 | s + b) / P(1 | b) = ((s + b)/b) · e^(−s).                        (1)

d ln LR/ds = 1/(s + b) − 1 = 0 ⇒ ŝ = 1 − b, and

  LR_max = e^(b − 1) / b,   q₀ = 2 ln LR_max,   Z_PLR = √q₀ (asymptotic, one-sided, Wilks/Cowan et al. 2011 [recalled, certain]).   (2)

The exact frequentist tail probability of the box is P(≥ 1 | b) = 1 − e^(−b) with one-sided Z = Φ⁻¹(1 − P).

### 2.2 Bayes factor
  B₁₀ = ∫ P(1 | s + b) π(s) ds / P(1 | b) = ∫ (1 + s/b) e^(−s) π(s) ds = **1 + E_π[s e^(−s)] / b**.   (3)

Since s e^(−s) ≤ e⁻¹ (at s = 1), **B₁₀ ≤ 1 + e⁻¹/b** for every prior, which equals LR_max to O(b). Closed forms for E ≡ E_π[s e^(−s)]:

- uniform on [0, s_max]: E = [1 − (1 + s_max) e^(−s_max)] / s_max;
- log-uniform on [a, c]: E = (e^(−a) − e^(−c)) / ln(c/a);
- Gamma(shape α, rate β): E = α β^α / (β + 1)^(α+1);
- δ(s − 1): E = e⁻¹ = 0.3679.

Each closed form was verified against `scipy.integrate.quad` to < 10⁻⁶ relative (assertion in the script).

### 2.3 Posteriors
Two hypotheses (DM vs. modelled background): P(DM | 1) = π B₁₀ / (π B₁₀ + 1 − π).   (4)

Three hypotheses — DM (prior π_DM), unknown/unmodelled background U (prior π_U, likelihood P(data | U) ≡ L_U taken as a free number), known background K (prior 1 − π_DM − π_U, likelihood P(1 | b) = b e^(−b)):

  P(DM | 1) = π_DM B₁₀ b e^(−b) / [π_DM B₁₀ b e^(−b) + π_U L_U + (1 − π_DM − π_U) b e^(−b)].   (5)

Using (3), π_DM B₁₀ b e^(−b) → π_DM E as b → 0, so

  **P(DM | 1) → π_DM E / (π_DM E + π_U L_U)** (b → 0), and P(DM | 1) > ½ ⇔ π_DM > π_U L_U / E.   (6)

For the central prior (uniform s ∈ [0, 10], E = 0.1000) this reads π_DM > 10 π_U L_U. The known-background term drops out: once b ≪ E, making the modelled background even smaller buys nothing, because the contest is between DM and whatever is *not* in the model.

### 2.4 Look-elsewhere effect ↔ Bayesian Occam penalty
Frequentist: (1 − p_local)^N_eff = 1 − p_global ⇒ N_eff = ln(1 − p_global)/ln(1 − p_local). Bayesian: if H₁ is an equal-weight mixture over N spectra, B_mix = (1/N) Σ_k B_k. Three accounting scenarios: (A) one spectrum has B₁₀, the other N − 1 are uninformative (B_k = 1): B_mix = (B₁₀ + N − 1)/N; (B) the others predict unseen low-energy populations and are excluded (B_k → 0): B_mix = B₁₀/N; (C) weight by the paper's own significance tables: B_mix ≈ f(≥ 3σ) · B₁₀, where f is the fraction of table cells at ≥ 3.0σ.

Independent check — the p-value calibration bound of Sellke, Bayarri & Berger (2001) [recalled, certain]: for a point null and any prior on the alternative, B₁₀ ≤ −1/(e p ln p) for p < e⁻¹.

### 2.5 Future events
For a second exposure k × 2.84 t·yr yielding n₂ events: L_DM = ∫ P(1 | s + b) P(n₂ | k(s + b)) π(s) ds; L_K = P(1 | b) P(n₂ | kb); L_U = P(1 | b + μ_U) P(n₂ | k(b + μ_U)) with μ_U = −ln(1 − L_U) chosen so that P(≥ 1 | μ_U) = L_U in the first exposure (for L_U = 0.1, μ_U = 0.1054 and P(1 | b + μ_U) = 0.095 ≈ L_U, consistent with Sec. 2.3).

## 3. Results

### 3.1 Poisson in a box (Table 1, `P001_table1_poisson_box.csv`)

| b | P(≥1 | b) | Z (exact, 1-sided) | ŝ | LR(s = 1) = LR_max | q₀ | Z_PLR = √q₀ |
|---|---|---|---|---|---|---|
| 2 × 10⁻⁴ | 2.000 × 10⁻⁴ | 3.54σ | 0.9998 | 1840 | 15.03 | 3.88σ |
| 1 × 10⁻³ | 9.995 × 10⁻⁴ | 3.09σ | 0.9990 | 368 | 11.82 | 3.44σ |
| 0.0106 | 1.054 × 10⁻² | 2.31σ | 0.9894 | 35.1 | 7.12 | 2.67σ |

(LR at s = 1 equals LR_max to four digits because ŝ = 1 − b ≈ 1; the paper's best-fit L₁₀ˢ signal is 1.0 (+1.4/−0.7) events [paper, Table I].) Note the asymptotic Z_PLR exceeds the exact Poisson Z by 0.3–0.35σ at these tiny b: Wilks' theorem is not accurate for a 1-event, b ≪ 1 problem, which is why LZ used ≥ 30 000 toys per model.

### 3.2 Why the box gives more than the paper's 3.4σ
In the LZ unbinned likelihood the only event contributing to the signal term is the event of interest (all other 1709 events sit where f_sig/f_bkg ≈ 0), so ln L(s) − ln L(0) = −s + ln(1 + s ρ) with ρ = f_sig(x_e)/(B f_bkg(x_e)) the signal-to-background *density* ratio at the event for unit signal. This is Eq. (1) with **b_eff = 1/ρ = b_box / ε_box**, where ε_box is the fraction of the signal PDF inside the box. Inverting the paper's 3.4σ (p_local = 3.37 × 10⁻⁴):

- exact Poisson inversion (tail-mass reading): b_eff = −ln(1 − p_local) = 3.37 × 10⁻⁴ ⇒ ε_box = 0.59;
- asymptotic profile-LR inversion (density reading): e^(b−1)/b = e^(Z²/2) ⇒ b_eff = 1.14 × 10⁻³ ⇒ ε_box = 0.18.

Both are plausible: the L₁₀ˢ, L₁₆ˢ and inelastic O₁/O₄ spectra spread signal over S1c ≈ 100–600 phd and the full NR band (Fig. 1 of the paper), so only ~20–60% of a unit signal lands in a ±2σ, S1c > 500 phd box. The box drawn around the event (a look-elsewhere effect *within the plane*) therefore overstates the evidence; **b ≈ 10⁻³ is the LZ-likelihood-equivalent value**, and b = 2 × 10⁻⁴ is an optimistic bound.

### 3.3 Bayes factors (Table 2, `P001_table2_bayes_factors.csv`)

| Prior π(s) | E[s e⁻ˢ] | B₁₀ (b = 2 × 10⁻⁴) | B₁₀ (b = 10⁻³) | B₁₀ (b = 0.0106) |
|---|---|---|---|---|
| uniform [0, 3] | 0.2670 | 1336 | 268 | 26.2 |
| **uniform [0, 10] (central)** | 0.1000 | **501** | **101** | **10.4** |
| uniform [0, 30] | 0.0333 | 168 | 34.3 | 4.1 |
| log-uniform [0.01, 30] | 0.1237 | 619 | 125 | 12.7 |
| Gamma(k = 2, θ = 0.5), mean 1 ("expected ≈ 1 near the limit") | 0.2963 | 1483 | 297 | 29.0 |
| Gamma(k = 2, θ = 1), mean 2 | 0.2500 | 1251 | 251 | 24.6 |
| δ(s = 1) = maximum-likelihood bound | 0.3679 | 1840 | 369 | 35.7 |

Kass–Raftery scale [recalled, certain]: B 1–3 "not worth more than a bare mention", 3–20 "positive", 20–150 "strong", > 150 "very strong". The prior dependence is a factor ~8 (uniform[0,30] vs Gamma mean 1) at fixed b; the b dependence is a factor 50 across the three b values. Against the modelled background alone the evidence is "very strong" for b ≤ 10⁻³ and only "positive" for the whole-panel b.

Sellke–Bayarri–Berger caps (independent of b and π): from p_local = 3.37 × 10⁻⁴, **B₁₀ ≤ 137**; from p_global = 4.66 × 10⁻³, **B₁₀ ≤ 14.7**; from the whole-panel P(≥ 1) = 0.0105, B₁₀ ≤ 7.7; from the box P(≥ 1) = 2 × 10⁻⁴, B₁₀ ≤ 216. The box values in the table exceed the local-p cap because the box is not the paper's test (Sec. 3.2); the b = 10⁻³ values (34–300) straddle it as expected.

### 3.4 Posterior probability of dark matter
Two hypotheses (Eq. 4), central prior uniform [0, 10] (`P001_table3a_posterior_2hyp.csv`, full prior grid in the file):

| b | B₁₀ | π_DM = 0.001 | 0.01 | 0.05 | 0.2 |
|---|---|---|---|---|---|
| 2 × 10⁻⁴ | 501 | 0.334 | 0.835 | 0.963 | 0.992 |
| 10⁻³ | 101 | 0.092 | 0.505 | 0.842 | 0.962 |
| 0.0106 | 10.4 | 0.010 | 0.095 | 0.354 | 0.723 |

Three hypotheses (Eq. 5), b = 2 × 10⁻⁴, B₁₀ = 501 (`P001_table3b_posterior_3hyp.csv`; Fig. 1 right panel):

| π_DM \ (π_U, L_U) | (0.01, 0.1) | (0.01, 0.5) | (0.1, 0.1) | (0.1, 0.5) | (0.3, 0.1) | (0.3, 0.5) |
|---|---|---|---|---|---|---|
| 0.001 | 0.077 | 0.019 | 0.010 | 0.002 | 0.003 | 0.001 |
| 0.01 | 0.456 | 0.162 | 0.090 | 0.020 | 0.032 | 0.007 |
| 0.05 | 0.808 | 0.491 | 0.330 | 0.091 | 0.142 | 0.032 |
| 0.2 | 0.945 | 0.795 | 0.664 | 0.285 | 0.400 | 0.118 |

The corresponding posterior on U is 0.45–0.998 in every cell with π_DM ≤ 0.05 except (π_DM = 0.05, π_U = 0.01, L_U = 0.1), where P(U) = 0.16; the known-background posterior is ≤ 0.15 everywhere. For b = 10⁻³ and 0.0106 the three-hypothesis numbers are almost identical for π_U ≥ 0.1 (the small-b limit, Eq. 6, holds already at b = 10⁻³; see the CSV). The condition for P(DM | 1) > ½ is π_DM > 10 π_U L_U for the central prior (π_DM > 3.4 π_U L_U for the most favourable Gamma prior, E = 0.296).

### 3.5 Look-elsewhere effect versus Occam penalty (`P001_table4_occam.csv`)
p_local = 3.369 × 10⁻⁴, p_global = 4.661 × 10⁻³ ⇒ **N_eff = 13.86** (ratio p_global/p_local = 13.83), ln N_eff = 2.63. From the paper's Tables S6–S7 (612 numerical cells = 616 models − 4 unphysical "–" entries; `lz.LSIG`, `lz.OSIG`): 338 cells (55%) are at 0.0σ; f(≥ 2.0σ) = 0.391, f(≥ 2.5σ) = 0.353, **f(≥ 3.0σ) = 0.209**, f(≥ 3.3σ) = 0.056, f(≥ 3.4σ) = 0.029.

| b | B₁₀ (central) | (A) others uninformative, N = 13.86 | (B) others excluded, B/N | (C) f(≥3σ)·B |
|---|---|---|---|---|
| 2 × 10⁻⁴ | 501 | 37.0 | 36.1 | 105 |
| 10⁻³ | 101 | 8.2 | 7.3 | 21.1 |
| 0.0106 | 10.4 | 1.7 | 0.8 | 2.2 |

Scenario (C) is generous: it treats the ~21% of model cells that fit at ≥ 3σ as all delivering the full B₁₀, and the 55% at 0.0σ as delivering B_k = 0, whereas in fact a uniform-in-s prior on an SI-like model (≈ 900 unseen low-energy events per high-energy event [dossier Sec. 2]) gives B_k ≈ 10⁻⁴, and the models at 2–3σ give intermediate values; and the paper's N_eff = 14 already accounts for the strong correlations among the ~128 high-significance cells. Scenarios (A)/(B) with N_eff ≈ 14 are the Bayesian analogue of the paper's LEE and reduce B₁₀ by ln 14 ≈ 2.6 in log-evidence, exactly as the frequentist correction multiplies p by 13.8. **The Occam-penalised Bayes factor is 7–37 for the LZ-equivalent range b = 10⁻³–2 × 10⁻⁴, bracketing the SBB cap of 14.7 derived from the paper's own global p-value.** On the Kass–Raftery scale that is "positive" to "strong", not "very strong".

### 3.6 Further LZ events (`P001_table5_future_events.csv`)
Equal further exposure (k = 1, 2.84 t·yr), central signal prior, L_U = 0.1 (μ_U = 0.105):

| b | π_DM | π_U | now | n₂ = 0 | n₂ = 1 | n₂ = 2 | n₂ = 3 |
|---|---|---|---|---|---|---|---|
| 2 × 10⁻⁴ | 0.01 | 0 (DM vs K only) | 0.835 | 0.558 | 1.000 | 1.000 | 1.000 |
| 2 × 10⁻⁴ | 0.05 | 0 | 0.963 | 0.868 | 1.000 | 1.000 | 1.000 |
| 2 × 10⁻⁴ | 0.01 | 0.01 | 0.456 | 0.192 | 0.735 | 0.975 | 0.999 |
| 2 × 10⁻⁴ | 0.01 | 0.1 | 0.090 | 0.028 | 0.217 | 0.797 | 0.987 |
| 2 × 10⁻⁴ | 0.05 | 0.1 | 0.330 | 0.125 | 0.581 | 0.952 | 0.997 |
| 10⁻³ | 0.01 | 0 | 0.505 | 0.202 | 0.996 | 1.000 | 1.000 |
| 10⁻³ | 0.01 | 0.1 | 0.088 | 0.026 | 0.215 | 0.794 | 0.986 |
| 10⁻³ | 0.05 | 0.1 | 0.325 | 0.117 | 0.577 | 0.951 | 0.997 |

Reading: **zero** further events in an equal exposure do not push P(DM) below 0.05 in the DM-vs-known-background comparison (a flat prior on s allows s ~ 0.3 and the second exposure only says s ≲ 2); they do (0.03) once an unknown background with π_U = 0.1 is on the table. **One** further event raises P(DM) above 0.5 only if the unknown-background prior is small (π_U L_U ≲ 10⁻³) or π_DM ≥ 0.05; **two** further events do so for every central choice (P(DM) ≥ 0.79). k = 2 rows are in the CSV (qualitatively the same; zero events at k = 2 give P(DM) = 0.36 for the DM-vs-K, π_DM = 0.01, b = 2 × 10⁻⁴ case, and 0.75 for π_DM = 0.05). Caveat: the discrimination between DM and U from *counting* comes entirely from the assumed U rate (μ_U = 0.105 vs the DM posterior mean s ≈ 1.5); an unknown background with a rate comparable to the signal is not distinguishable by counts alone.

Posterior predictive for the next equal exposure (central three-hypothesis case b = 2 × 10⁻⁴, π_DM = 0.01, π_U = 0.1, L_U = 0.1; posterior weights DM/U/K = 0.094/0.890/0.017): P(0) = 0.84, P(1) = 0.11, P(2) = 0.022, P(≥ 3) = 0.030. For the DM-vs-K-only case with π_DM = 0.05 (weights 0.963/0/0.037): P(0) = 0.28, P(1) = 0.24, P(2) = 0.18, P(≥ 3) = 0.30.

## 4. Validation checks and robustness
1. Closed forms for E_π[s e⁻ˢ] agree with numerical quadrature to < 10⁻⁶ (script assertion).
2. LR_max and B₁₀ for the δ(s = 1) prior agree (1840 vs 1840 at b = 2 × 10⁻⁴), as required by Eq. (3).
3. The dossier's `LR_s1_vs_b_in_box` = (1 + 0.0106)/0.0106 = 95.3 omits the e^(−s) factor; the correct single-event LR at s = 1 is 35.1 (Table 1). We flag this as a correction to the dossier number (it was labelled a Bayes factor there; it is neither).
4. N_eff = 13.86 reproduces the dossier's "≈ 14" from `lz.sigma_to_p`.
5. The two b_eff inversions of 3.4σ (3.4 × 10⁻⁴ and 1.1 × 10⁻³) bracket the intermediate b = 10⁻³ used as the LZ-equivalent case; the b = 10⁻³ asymptotic Z_PLR = 3.44σ reproduces the paper's 3.4σ.
6. Prior sensitivity of B₁₀ at fixed b: factor 8 between uniform[0,30] and Gamma(mean 1), factor 3.7 between uniform[0,30] and uniform[0,3]; posterior sensitivity is dominated instead by π_DM and π_U L_U (Eq. 6).
7. The three-hypothesis posterior at b = 10⁻³ differs from b = 2 × 10⁻⁴ by < 0.01 for π_U ≥ 0.1 (small-b limit), confirming Eq. (6).

## 5. Figures
- `figures/P001_fig1_posterior_vs_prior.png` — Left: P(DM | one event) versus π_DM for b = 2 × 10⁻⁴, 10⁻³, 0.0106 (DM vs modelled background; uniform s ∈ [0, 10]), and for b = 2 × 10⁻⁴ with an unknown-background hypothesis (π_U = 0.1 dashed, 0.01 dotted; L_U = 0.1). Right: annotated grid of P(DM | event) over π_DM × (π_U, L_U) for b = 2 × 10⁻⁴, B₁₀ = 501.
- `figures/P001_fig2_bayes_factor_vs_b.png` — B₁₀ = 1 + E[s e⁻ˢ]/b versus b for four priors, the maximum-likelihood bound, and the Sellke–Bayarri–Berger caps from the paper's local (137) and global (14.7) p-values.

## 6. Failed or abandoned approaches
- A Poisson model for the unknown-background likelihood with P(exactly 1 | μ_U) = L_U cannot represent L_U = 0.5 (max of μe^(−μ) is 0.368); we therefore treat L_U as a given likelihood in Sec. 3.4 and use P(≥ 1 | μ_U) = L_U only for the forward projection (Sec. 3.6).
- We did not attempt to reconstruct the 2D signal and background PDFs to compute the true density ratio ρ at the event; that is left to P052 (reproducing the 3.4σ) in the corpus plan.

## 7. Extended discussion
1. Against the *modelled* background the single event is very strong evidence (B₁₀ ≈ 100–500 for reasonable priors, b = 10⁻³–2 × 10⁻⁴), consistent with the paper's 3.4σ. But this Bayes factor is the wrong quantity for the question "is it dark matter?", for two reasons. First, H₁ is not one spectrum but ~14 effectively distinct ones (293 nominally), so the honest Bayes factor carries an Occam factor ~1/14 and lands at 7–37, in line with the SBB cap of 15 from the paper's 2.6σ global p-value. Second, and more importantly, the relevant alternative to DM is not the modelled background but an *unmodelled* one; Eq. (6) shows that once b ≪ 0.1 the modelled background is irrelevant and P(DM) is set by π_DM E versus π_U L_U. With E ≤ 0.37 for any prior, P(DM) > ½ requires π_DM > 2.7–10 × π_U L_U. Given LZ's own list of partially validated backgrounds (wall MSSI with 100% uncertainty and 0–1 sideband events; the excluded ER-tail systematic; non-blind selections) [paper, supplement; dossier Sec. 4], π_U L_U ≳ 0.01 is hard to argue against, so P(DM) ≲ 0.1–0.5 for π_DM ≤ 0.05.
2. The direction of the reassessment is downward relative to a naive reading of "2 × 10⁻⁴ background", for three compounding reasons: the box is post hoc (ε_box < 1, Sec. 3.2), the model space is wide (Sec. 3.5), and the alternative is unknown background (Sec. 3.4). It is not downward relative to the paper's own global 2.6σ, which our Occam-penalised B₁₀ ≈ 7–37 reproduces in Bayesian terms.
3. What would change the picture: a second event in the same region within an equal exposure (P(DM) → 0.2–0.7 depending on π_U), two events (→ ≥ 0.8), or — more decisively than counting — a demonstrated *spectral or positional* prediction (annual-modulation phase, energy distribution) that separates DM from unknown backgrounds, since counting alone cannot separate a DM signal from an unknown background of similar rate.

## 8. References
- LZ Collaboration, arXiv:2609.02823 (2026) — the paper under study (abstract; Results; Table I; Fig. 5 caption; supplement "Log-Likelihood Function and the Look Elsewhere Effect"; Tables S6–S7).
- R. E. Kass and A. E. Raftery, "Bayes Factors", J. Am. Stat. Assoc. 90, 773 (1995).
- T. Sellke, M. J. Bayarri and J. O. Berger, "Calibration of p Values for Testing Precise Null Hypotheses", Am. Stat. 55, 62 (2001).
- J. O. Berger and T. Sellke, "Testing a Point Null Hypothesis: The Irreconcilability of P Values and Evidence", J. Am. Stat. Assoc. 82, 112 (1987).
- G. Cowan, K. Cranmer, E. Gross and O. Vitells, "Asymptotic formulae for likelihood-based tests of new physics", Eur. Phys. J. C 71, 1554 (2011).
- E. Gross and O. Vitells, "Trial factors for the look elsewhere effect in high energy physics", Eur. Phys. J. C 70, 525 (2010).
- D. Baxter et al., "Recommended conventions for reporting results from direct dark matter searches", Eur. Phys. J. C 81, 907 (2021) (cited by LZ for the LEE procedure).
- Corpus: `output/00_evidence_dossier.md` (P000 dossier).

## 9. Tools and provenance (mirrors `output/provenance/P001.json`)
- **Agent tools:** Read (PAPER_GUIDE.md; 00_evidence_dossier.md; fulltext.tex lines 250–294, 460–504, 820–911; two PNG figures); Bash (ls/cat/grep/sed of the ledger, environment versions, lzcommon, dossier script and JSON; mkdir; two runs of the script; wc -w on the paper); Write (script, details.md, P001.json, P001.md); Skill (dataviz, for the figure palette).
- **Software:** python 3.12.13; numpy 2.5.3 (logspace, array ops); scipy 1.18.1 (stats.poisson, stats.norm, stats.gamma, integrate.quad, optimize.brentq); matplotlib 3.11.2 (Agg, pyplot, LinearSegmentedColormap); `output/code/common/lzcommon.py` (LZ dict, LSIG, OSIG, sigma_to_p, p_to_sigma, poisson_p_at_least); json/csv/math (stdlib).
- **Script:** `output/code/P001_bayes_single_event.py`, run as `.venv/bin/python output/code/P001_bayes_single_event.py` from `/Users/reza/LZ_simulation`.
- **Local inputs:** `inputs/LZ_arXiv_2609.02823_fulltext.tex` (abstract, Results paragraph, Fig. 5 caption, Table I via lzcommon, LEE supplement, Tables S6–S7); `output/00_evidence_dossier.md` (b ≈ 2 × 10⁻⁴, N_eff ≈ 14, ≈ 900 low-E events per high-E event); `output/work/dossier/dossier_numbers.json` (p_local, p_global, N_eff cross-check); `environment/ENVIRONMENT_versions.txt`.
- **Recalled knowledge (5 items):** Sellke–Bayarri–Berger bound (certain); Kass–Raftery scale (certain); Wilks/Cowan et al. asymptotic q₀ = Z² (certain); Gross–Vitells trials-factor framework (certain, context only); Berger–Sellke 1987 p-value/evidence discrepancy (certain, context only).
- **Datasets:** none. **Data requests:** none. **WimPyDD-generated files:** none.
- **Tables written:** `P001_table1_poisson_box.csv`, `P001_table2_bayes_factors.csv`, `P001_table3a_posterior_2hyp.csv`, `P001_table3b_posterior_3hyp.csv`, `P001_table4_occam.csv`, `P001_table5_future_events.csv`, `P001_results.json`.
