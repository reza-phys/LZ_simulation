# P008 — An independent look-elsewhere estimate for 616 models tested on one event: research record

Simulated date 2026-09-04. Author profile: statistical methods group in astroparticle physics. Category STAT (physics.data-an, cross-list hep-ex).
Script: `output/code/P008_lee_toys.py` (run: `N_TOYS=300000 N_TOYS_VAR=100000 .venv/bin/python output/code/P008_lee_toys.py`, 300 s after the
WimPyDD spectrum cache is built; the cache itself takes ~17 min, 501 spectra × 60 energy points at 12–27 ms per WimPyDD call).
All numbers below are in `output/work/P008/results.json`, `subspace_neff.csv`, `robustness.csv`, `data_local_Z.csv`,
`gross_vitells_deltascan.csv`, `asymptotic_check.csv`, `background_components.csv`, `models.csv`.

## 1. Motivation and question

LZ (arXiv:2609.02823) tests 616 signal models (20 Lagrangians × isoscalar/isovector × 13 masses = 520; inelastic O1, O4 × s/v × 3 masses × 8 splittings = 96),
reduced by spectral degeneracies to 293 distinguishable spectra, against a data set whose only signal-like feature is a single 248 keV event.
The maximum local significance is 3.4σ (p = 3.37 × 10⁻⁴); the toy-MC global significance is 2.6σ (p = 4.7 × 10⁻³). Under the definition
p_global = 1 − (1 − p_local)^N_eff this implies N_eff = ln(1 − p_global)/ln(1 − p_local) = 13.9 effective independent trials
(`results.json: N_eff_implied_by_LZ_3.4_to_2.6`; the dossier quotes ≈ 14). We ask: (i) is a trials factor of ~14 for ~300 spectra what one
expects when the data consist of one high-energy event; (ii) does it follow from an independent toy calculation; (iii) how does it depend on
the choice of model space (only inelastic, only heavy elastic, everything, a continuous δ scan)?

## 2. Framework

### 2.1 Simplified one-dimensional likelihood
The LZ likelihood is unbinned in {S1c, log₁₀S2c} with three samples and nuisance parameters (Supplement, "Log-Likelihood Function").
We replace it by an extended unbinned likelihood in a single observable, the *observed* NR-equivalent recoil energy E ∈ [5.4, 270] keV,
for events *inside the NR band* of the science sample:

  ln L(s) = −(B + s) + Σ_i ln[ b(E_i) + s f_m(E_i) ]          (1)

with B the total background expectation (fixed), b(E) the background density (∫b = B), f_m the normalised signal shape of model m and
s ≥ 0 the fitted number of signal events. The test statistic is q₀ = 2[ln L(ŝ) − ln L(0)], ŝ = argmax (ŝ = 0 if Σ_i f_m(E_i)/b(E_i) ≤ 1;
otherwise the unique root of Σ_i f_i/(b_i + s f_i) = 1, found by bisection on [0, n+1], vectorised over toys × models).
Background normalisations are not profiled (the 22% accidentals constraint and the veto-sample constraints are ignored); see §7.

### 2.2 Detector response
Efficiency ε(E) = 0.96 · logistic((E − 5.4)/1.5 keV) · [1 − logistic((E − 269.9)/6 keV)]: 50% at 5.4 and 269.9 keV, 0.96 plateau
(paper: 50% points 5.4 and 269.9 keV; 96% average 14–250 keV). Resolution: Gaussian, σ(E) = 23 keV × √(E/248 keV)
(paper: E = 248 ± 23 (stat) keV; √E scaling assumed). True spectra on a 0.5 keV grid (0.5–420 keV) are convolved with the Gaussian and
multiplied by ε(E) on the observed grid (5.4–270 keV, 0.5 keV steps, 530 points).

### 2.3 Background model (Table I of the paper, projected onto E inside the NR band)
| component | expected (Table I) | shape assumed | fraction > 100 keV | fraction > 200 keV | density at 248 keV [/keV] |
|---|---|---|---|---|---|
| accidentals | 2.7 | exp(−E/15 keV) in observed E (isolated-S1 spectrum falls steeply; scale is an assumption, varied 10–20 keV) | 1.8 × 10⁻³ | 2.2 × 10⁻⁶ | 1.7 × 10⁻⁸ |
| atmospheric-ν CEνNS | 0.11 | computed: dR/dE ∝ F²_Helm(E) ∫_{E_min} Φ(E_ν)[1 − M E/(2E_ν²)] dE_ν, E_min = √(M E/2); Φ ∝ E_ν below 100 MeV, flat to 300 MeV, ∝ E_ν⁻³ above (recalled, uncertain); smeared, × ε | 1.6 × 10⁻² | 2.2 × 10⁻³ | 1.5 × 10⁻⁶ |
| ⁸B + hep | 0.057 | exp(−(E − 5.4)/1 keV): entirely below ~8 keV | 0 | 0 | 0 |
| MSSI | 0.0049 | flat in NR-equivalent E from 50 keV (logistic turn-on, width 10 keV) to 270 keV (12 keV first scatter + 471 phd S1-only second scatter, Supplement) | 0.77 | 0.32 | 2.2 × 10⁻⁵ |
| detector neutrons | 0.05 | exp(−E/40 keV) true (α,n) recoil spectrum (assumption; varied 25–60 keV and 0.02–0.118 events), smeared, × ε | 0.10 | 7.1 × 10⁻³ | 3.3 × 10⁻⁶ |
| **total** | **2.92** | | 5.3 × 10⁻³ | 7.4 × 10⁻⁴ (= 0.0022 events) | **2.7 × 10⁻⁵** |

ER leakage is *not* included: Fig. 5 of the paper shows that in the S1c > 500 phd panel the 0.0106 background events sit at +1 to +5σ from the NR
median (ER tail); inside the NR band above ~100 keV it is negligible. Below ~50 keV our accidentals exponential stands in for all low-energy
NR-band backgrounds (accidentals, ER leakage into the band, ⁸B). The Table I entry for detector NRs is a fit interval [0, 0.118]; the multiple-scatter
cross-check predicts 0.02 ± 0.02 single scatters, so 0.05 is a middle value. Our total NR-band background above 200 keV, 0.0022 events, and the
density at 248 keV, 2.7 × 10⁻⁵/keV (≈ 1.2 × 10⁻³ events in 248 ± 23 keV), are about 6× larger than the paper's ≈ 2 × 10⁻⁴ within ±2σ of the NR
median at S1c > 500 phd (dossier §2), because the one-dimensional projection puts all of the atm-ν, MSSI and neutron expectation on the NR median.
This makes our local significances slightly conservative (§5.5) but is irrelevant for the trials factor, which is a ratio of probabilities.

### 2.4 Signal model space
Spectra from WimPyDD 2.0.4 (shell-model nuclear response functions; the code LZ used), through `lzcommon.wd_hamiltonian/wd_rate`, with the
time-averaged Baxter-2021 halo (`wd_halo()`: v₀ = 238, v_esc = 544, v_⊙ = (11.1, 250.2, 7.3) km/s), j_χ = 1/2, natural xenon, on a 60-point
logarithmic energy grid 2–400 keV (interpolated linearly to 0.5 keV):
- **Elastic**: NR operators O₁, O₃–O₁₅ (14 operators; the basis onto which all 20 relativistic Lagrangians L₁–L₂₀ of Anand–Fitzpatrick–Haxton reduce),
  each isoscalar (c⁰ ≠ 0) and isovector (c¹ ≠ 0), at the 13 LZ masses 10–4000 GeV → 364 spectra. We do not map Lagrangians to operators one by one
  (the exact mapping is recalled only with low confidence); the operator basis spans at least the same set of spectral *shapes*, which is all the LEE
  depends on.
- **Inelastic**: O₁ and O₄, isoscalar and isovector, m = 400, 1000, 4000 GeV, δ = 0, 50, …, 350 keV → 96 (exactly LZ's inelastic grid).
  4 spectra are identically zero (400 GeV, δ = 350 keV: kinematically forbidden) — matching the four "–" entries of Table S7.
- **Continuous scan** (for question (d)): O₁ᵗ, 1000 GeV, δ = 0, 10, …, 400 keV (41; δ = 400 keV is forbidden → 40).
Total 460 LZ-like models, 456 physical, plus 40 scan models. Shapes are normalised on the observed grid; only shape + fitted count enter.

**Degeneracy check** (max |ΔCDF| between smeared observed shapes): O₄ s vs v 0.5%, O₆ 0.9%, O₁₀ 0.5% (LZ: s/v indistinguishable for L₂, L₄, L₇, L₈,
L₁₀, L₁₁, L₁₄, L₁₅, L₁₉, L₂₀ and O₄ ✓); O₁ and O₁₁ s vs v 10% (distinct; LZ keeps L₁ˢ/L₁ᵛ and L₃ˢ/L₃ᵛ separate ✓); 400 vs 1000 GeV 0.7–4.9%,
1000 vs 4000 GeV 0.3–2.1% (LZ: "nearly degenerate" above 400 GeV ✓); 100 vs 200 GeV 4–24% (distinct). Counting distinct spectra at a 1% (0.5%, 2%)
threshold gives 287 (349, 205) of 456 — bracketing LZ's 293. Duplicates cannot change a minimum p-value, so the toy N_eff is independent of this counting;
it matters only for Bonferroni.

### 2.5 Toy Monte Carlo and calibration
Background-only toys: n ~ Poisson(B = 2.92), energies from b(E)/B. 150,000 *calibration* toys give, per model, the empirical distribution of q₀
under H₀ and hence the calibrated local p-value p_m(q) = P(q₀,m ≥ q | H₀) (the paper: ≥ 30,000 toys per model, one-sided Z from p₀). An independent
150,000 *evaluation* toys are converted to p_m and, for any model subspace S, p_min = min_{m∈S} p_m. The global p-value at a local threshold p_loc is
P(p_min ≤ p_loc), and N_eff = ln(1 − p_global)/ln(1 − p_loc). Statistical errors are binomial (√k/N). Mean toy multiplicity 2.92, maximum 12;
the fraction of toys containing an event above 200 keV is 1 − e^{−0.0022} = 0.22%.

## 3. Validation

**Asymptotics fail in the bulk but hold in the tail** (`asymptotic_check.csv`). Wilks/Chernoff predicts P(q₀ > 0) = 0.5 and P(q₀ > 11.56) = 3.4 × 10⁻⁴.
Toys: P(q₀ > 0) = 0.45 (light elastic), 0.28 (heavy elastic), 0.18 (inelastic); P(q₀ > 11.56) = 3.5 × 10⁻⁴, 2.3 × 10⁻⁴, 1.6 × 10⁻⁴. So √q₀ is a
poor significance measure for high-energy models at low q₀ (the fit prefers s = 0 whenever no event lies in the model's window) and mildly conservative
at 3.4σ (a factor ≤ 2 in p). This is why toy calibration, as LZ did, is necessary, and why we use it throughout.

**Single-model sanity check**: N_eff for a single model should be 1. We obtain 0.73 ± 0.12 (O₁ˢ 1 TeV) and 1.42 ± 0.17 (O₆ˢ 1 TeV) at 3.4σ, 0.82–1.03 at 3.0σ.
The scatter reflects the discreteness of the calibrated p-values (with ~3 events, q₀ takes a nearly discrete set of values) and the finite calibration
sample; we therefore assign a ±30–40% method systematic to any N_eff below ~3 and ±10% (twice the statistical error) at N_eff ~ 10.

**Single-event picture** (analytic). For a model whose window contains one event and no others, ŝ = 1 − b/f ≈ 1 and q₀ ≈ 2[ln(f/b) − 1].
For O₆ˢ at 1 TeV, f/b at 248 keV = 115 → √q₀ = 2.74; the background expected where the likelihood ratio exceeds its value at 248 keV is 3.4 × 10⁻⁴
events, i.e. a Poisson "local p" of 3.4 × 10⁻⁴ = 3.4σ. The toy-calibrated local p for the same model is 1.4 × 10⁻³ (3.0σ) because low-energy
multi-event fluctuations can also reach the observed q₀. The local significance of *any* high-energy model is thus essentially the probability that the
background produces one event as deep in the signal-dominated region as the observed one — the same event drives all of them.

## 4. Main result: trials factor of the LZ-like model space

| threshold Z_local | p_local | p_global (toys) | Z_global | N_eff |
|---|---|---|---|---|
| 2.0 | 2.28 × 10⁻² | 0.105 | 1.25 | 4.8 |
| 2.5 | 6.21 × 10⁻³ | 0.0366 | 1.79 | 6.0 |
| 3.0 | 1.35 × 10⁻³ | 0.0112 | 2.29 | 8.3 |
| **3.4** | **3.37 × 10⁻⁴** | **(4.1 ± 0.2) × 10⁻³** | **2.64** | **12.2 ± 0.5 (stat) ± 1.2 (method)** |

(456 physical models, 287 distinct at 1%; `subspace_neff.csv` row "(c)".) LZ: p_global = 4.7 × 10⁻³, 2.6σ, N_eff = 13.9. Our independent estimate
agrees within its uncertainty: **the 3.4σ → 2.6σ reduction follows**. N_eff grows with the threshold (4.8 → 12.2 from 2σ to 3.4σ), as expected when
the model set contains quasi-continuous families (Gross–Vitells behaviour); a single trials factor is therefore threshold-specific.
Bonferroni with 293 (616) models would give p_global = 0.099 (0.21), Z = 1.29σ (0.81σ): far too conservative by a factor 20–50 in N.
With 14 trials Bonferroni gives 2.60σ, i.e. LZ's number *is* the Bonferroni count for ~14 independent tests.

## 5. Model-space dependence (`subspace_neff.csv`; all at local 3.4σ)

| model space | N models (raw / distinct 1%) | p_global | Z_global | N_eff | Bonferroni Z (distinct) |
|---|---|---|---|---|---|
| single model O₁ˢ 1 TeV | 1 / 1 | 2.5 × 10⁻⁴ | 3.48 | 0.73 ± 0.12 | 3.40 |
| single model O₆ˢ 1 TeV | 1 / 1 | 4.8 × 10⁻⁴ | 3.30 | 1.42 ± 0.17 | 3.40 |
| O₁ˢ elastic, 13 masses | 13 / 12 | 6.9 × 10⁻⁴ | 3.20 | 2.06 ± 0.20 | 2.65 |
| O₁ˢ+O₁ᵛ elastic | 26 / 20 | 7.4 × 10⁻⁴ | 3.18 | 2.20 ± 0.21 | 2.47 |
| inelastic O₁ˢ 1 TeV, LZ δ grid | 8 / 8 | 1.4 × 10⁻³ | 2.99 | 4.20 ± 0.29 | 2.78 |
| (a′) inelastic O₁/O₄ isoscalar | 46 / 40 | 3.0 × 10⁻³ | 2.75 | 8.9 ± 0.4 | 2.21 |
| (a) inelastic O₁/O₄ s+v (LZ grid) | 92 / 65 | 3.3 × 10⁻³ | 2.71 | 9.9 ± 0.4 | 2.02 |
| (b′) elastic m ≥ 100 GeV, isoscalar | 70 / 65 | 1.4 × 10⁻³ | 2.99 | 4.10 ± 0.29 | 2.02 |
| (b) elastic m ≥ 100 GeV, s+v | 140 / 116 | 1.5 × 10⁻³ | 2.96 | 4.51 ± 0.30 | 1.76 |
| elastic m ≤ 50 GeV | 224 / 115 | 1.3 × 10⁻³ | 3.01 | 3.92 ± 0.28 | 1.77 |
| elastic, all masses | 364 / 229 | 2.2 × 10⁻³ | 2.85 | 6.6 ± 0.4 | 1.42 |
| **(c) full LZ-like space** | **456 / 287** | **4.1 × 10⁻³** | **2.64** | **12.2 ± 0.5** | 1.30 |
| (d) δ scan O₁ˢ 1 TeV, 10 keV steps, δ ≤ 100 | 11 / 10 | 5.5 × 10⁻⁴ | 3.27 | 1.6 | 2.71 |
| (d) … δ ≤ 200 | 21 / 20 | 8.1 × 10⁻⁴ | 3.15 | 2.4 | 2.47 |
| (d) … δ ≤ 300 | 31 / 30 | 1.6 × 10⁻³ | 2.95 | 4.7 | 2.32 |
| (d) … δ ≤ 350 | 36 / 35 | 2.5 × 10⁻³ | 2.81 | 7.4 ± 0.4 | 2.26 |
| (d) … δ ≤ 400 (kinematic limit) | 40 / 39 | 3.1 × 10⁻³ | 2.73 | 9.3 ± 0.4 | 2.22 |
| (c) + (d) | 496 / 317 | 4.9 × 10⁻³ | 2.58 | 14.5 ± 0.5 | 1.24 |

Reading: (i) the 140 heavy elastic models are worth only ~4.5 trials — they all "see" the same event; (ii) the 224 light models are worth ~4 trials via
low-energy count fluctuations (Poisson(2.7) ≥ 9–10 has p ~ 10⁻³–10⁻⁴), a mode absent from the data but present in toys; (iii) the inelastic grid is the
largest single contributor (~10), because each δ selects a different ~2σ-wide energy window in which a lone background event would register;
(iv) a continuous δ scan is worth as many trials as the whole LZ inelastic grid: N_eff rises from 1.6 to 9.3 as δ_max goes from 100 to 400 keV,
and from 7.4 to 9.3 (+25%) when the scan is extended from LZ's 350 keV to the kinematic limit; the 8-point LZ grid (4.2) captures only about half of
the 10-keV-step value at δ ≤ 350 (7.4). Adding the fine scan to the LZ space raises N_eff from 12.2 to 14.5 and lowers the global significance from 2.64σ to 2.58σ.
Global significances of the physically motivated subspaces span 2.6–3.0σ.

### 5.1 Gross–Vitells comparison (`gross_vitells_deltascan.csv`)
For the continuous δ scan we counted, per toy, upcrossings of the asymptotic Z(δ) = √q₀(δ) curve above z₀ and applied
P(max Z > z) ≈ P(Z > z) + ⟨N_up(z₀)⟩ exp[−(z² − z₀²)/2]: ⟨N_up⟩ = 0.41, 0.24, 0.10 at z₀ = 0.5, 1.0, 1.5, giving N_eff^GV(3.4σ) = 5.2, 4.7, 3.9 against
3.7 from the toys evaluated in the same asymptotic-Z metric (agreement at z₀ ≥ 1; GV over-predicts at z ≤ 2.5 where the asymptotic tail is invalid).
The *calibrated* N_eff for the same scan is 9.3: the calibrated p of high-δ models is smaller than the Wilks value at the same q₀ (§3), so their calibrated
Z exceeds √q₀ and they win the minimum more often. Gross–Vitells in the asymptotic-Z metric therefore under-estimates the calibrated trials factor by ~2.5
in this few-event regime, though it correctly captures the *growth* of N_eff with the scan range.

### 5.2 Correlation-based estimate (uninformative)
The Nyholt/Cheverud M_eff = 1 + (M − 1)(1 − Var(λ)/M) from the eigenvalues of the correlation matrix of √q₀ across toys gives 302 of 456 — it measures linear
correlation of variables that are zero in most toys and says nothing about the joint tail; abandoned as a method.

### 5.3 Robustness (`robustness.csv`, 50k + 50k toys each)
| variant | B > 200 keV | N_eff(3.4σ) | Z_global(3.4σ) | data max local Z |
|---|---|---|---|---|
| nominal | 0.0022 | 12.5 ± 0.9 | 2.63 | 3.67 |
| accidentals τ = 10 keV | 0.0022 | 13.2 ± 0.9 | 2.62 | 3.67 |
| accidentals τ = 20 keV | 0.0023 | 11.1 ± 0.8 | 2.67 | 3.54 |
| high-E background ×3 (neutrons 0.118, τ = 60 keV; MSSI ×2) | 0.0067 | 8.7 ± 0.7 | 2.75 | 3.06 |
| high-E background ÷2 (neutrons 0.02; MSSI ×0.5) | 0.0012 | 11.4 ± 0.8 | 2.67 | 3.78 |
| atm-ν flux flat below 100 MeV | 0.0022 | 12.5 ± 0.9 | 2.63 | 3.67 |
N_eff stays within 8.7–13.2 (Z_global 2.62–2.75σ) for a factor 3 up or 2 down in the high-energy background and a factor 2 in the accidentals scale.
More high-energy background *reduces* the trials factor (the high-energy windows become less exclusive) while reducing the local significance.

## 6. The data in the simplified likelihood (`data_local_Z.csv`)
Data set: the 248 keV event plus n_low "typical" low-energy events at the (k − ½)/n quantiles of the accidentals spectrum (n_low = 3: 8.1, 15.8, 32.3 keV;
we do not know the actual low-energy NR-band events). Calibrated local Z (n_low = 3):

| operator (elastic) | 50 GeV | 100 | 200 | 400 | 1000 | 4000 | LZ analogue (Table S6, 100→4000 GeV) |
|---|---|---|---|---|---|---|---|
| O₁ˢ | 0.2 | 0.2 | 0.2 | 0.3 | 0.4 | 0.4 | L₁ˢ/L₅ˢ: 0.0 everywhere |
| O₁ᵛ | 0.2 | 0.3 | 0.7 | 1.1 | 1.4 | 1.5 | L₁ᵛ: 0.0, 0.0, 0.3, 1.3, 1.2 |
| O₃ˢ / O₁₁ˢ | 0.0 / 0.2 | 0.8 / 0.2 | 2.0 / 0.6 | 2.3 / 1.0 | 2.4 / 1.1 | 2.5 / 1.2 | L₃ˢ/L₁₇ˢ: 0.0, 1.1, 1.5, 1.7, 1.8 |
| O₄ˢ | 0.2 | 0.6 | 1.8 | 2.1 | 2.3 | 2.3 | O₄ˢ δ = 0: 2.6, 2.7, 2.8 (400–4000) |
| O₆ˢ | 0.0 | 2.0 | 2.6 | 2.9 | 3.0 | 3.0 | L₄ˢ/L₂₀ˢ: 2.6, 3.1, 3.1, 3.1, 3.0 |
| O₁₀ˢ | 0.0 | 1.5 | 2.3 | 2.6 | 2.7 | 2.7 | L₂ˢ/L₈ˢ: 2.4, 2.8, 2.9, 3.0, 3.1 |
| O₅ᵛ, O₉, O₁₃ˢ, O₁₄, O₁₅ | 0–0.2 | 1.4–1.9 | 2.3–2.7 | 2.4–2.9 | 2.4–3.0 | 2.5–3.0 | L₆ᵛ, L₉–L₁₂, L₁₆, L₁₈–L₂₀: 2.2–3.0, 2.8–3.3, 2.9–3.4 |
| all m ≤ 50 GeV | ≤ 0.4 | | | | | | 0.0–0.1 |

| inelastic (1000 GeV) | δ=0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 |
|---|---|---|---|---|---|---|---|---|
| O₁ˢ ours / LZ | 0.4 / 0.0 | 0.4 / 0.0 | 0.7 / 0.8 | 1.3 / 2.2 | 2.0 / 2.7 | 2.4 / 2.9 | 2.5 / 3.0 | 2.8 / 3.3 |
| O₁ᵛ ours / LZ | 1.4 / 1.3 | 1.7 / 1.6 | 2.2 / 2.6 | 2.7 / 2.8 | 3.1 / 2.9 | 3.3 / 3.0 | 3.4 / 3.4 | 3.4 / 3.4 |
| O₄ˢ ours / LZ | 2.3 / 2.7 | 2.3 / 2.8 | 2.5 / 2.8 | 2.5 / 3.0 | 2.6 / 3.2 | 2.8 / 3.2 | 3.1 / 3.4 | 3.5 / 3.4 |

The qualitative pattern of Tables S6/S7 is reproduced: ≈ 0 for all m ≤ 50 GeV (kinematically unable to give 248 keV), ≈ 0 for isoscalar O₁ at every mass
(the SI spectrum is so steep that its density at 248 keV is comparable to the background even for 4 TeV: ŝ = 0 for n_low ≤ 2, 0.4 for n_low = 3), rising
isovector O₁ (1.5 at 4 TeV vs LZ 1.2), 2.5–3.0 for the momentum-suppressed operators at m ≥ 400 GeV (LZ 3.0–3.4), and inelastic significances rising with
δ to 3.4–3.5 (LZ 3.3–3.4). Our heavy-model values are typically 0.3–0.5σ below LZ's, consistent with our ~6× larger 1D background density at 248 keV
(§2.3; median over m ≥ 400 GeV elastic models: 2.47 vs LZ 3.00). 22 of 456 models reach ≥ 3σ (LZ: 128 of 612); 242 are ≤ 0.5σ (LZ: 346).
The maximum, 3.76σ (O₄ˢ 4 TeV δ = 350 keV, p = 8.7 × 10⁻⁵), is independent of n_low ∈ {0, 2, 3, 5} — the low-energy events are irrelevant for high-energy models.
Evaluated against our own min-p distribution the data's global p is 1.2 × 10⁻³ (3.04σ), N_eff = 13.8: the same trials factor as at the 3.4σ threshold.

## 7. Caveats and what would change the numbers
1. One observable instead of {S1c, log S2c}: models that differ only in their S2c/S1c band position (none among the tested spectra) would add trials; the
   real analysis also profiles ~15 nuisance parameters, which broadens the q₀ distributions slightly (more trials-like freedom) but is shared by all models.
2. The low-energy NR-band background (2.7 accidentals with an assumed 15 keV scale) is a stand-in for accidentals + ER leakage + ⁸B; it controls the
   ~4 trials contributed by light models. The variants τ = 10–20 keV move N_eff by ±1.
3. No nuisance profiling: a floating accidentals normalisation would absorb low-energy fluctuations and *reduce* the light-model trials, pushing N_eff toward ~9–10.
4. Method floor: single-model N_eff = 0.7–1.4 → ±1.2 systematic on 12.2.
5. Spectra: WimPyDD's coupling normalisation is irrelevant here (shapes only); the halo is time-averaged (LZ's 220 live days span > 1 yr).
6. Our operator basis is not the LZ Lagrangian list; it contains more distinct heavy-mass shapes (28 vs LZ's 23 types) and the trials factor of that block is
   only ~4.5, so this cannot change the total by more than ~1.

## 8. Failed or abandoned approaches
- Nyholt/Cheverud eigenvalue M_eff (302 of 456): meaningless for zero-inflated q₀ — abandoned.
- Gross–Vitells with asymptotic Z on the δ scan: reproduces the asymptotic-metric toys but underestimates the calibrated trials factor by ~2.5 — kept as a comparison only.
- A first test run with 2000 calibration toys floored all local p-values at 2.5 × 10⁻⁴ (Z ≤ 3.48) and returned NaN trials factors for the variants; superseded by the 300k run.
- A one-to-one mapping of L₁–L₂₀ to NR operators was not attempted (low-confidence recall).

## 9. Figures
- `figures/fig1_maxZ_distribution.png` — P(max local Z ≥ Z | H₀) for the full space (456), the inelastic block (96) and a single model, with the Gaussian tail;
  the full-space curve crosses LZ's p_global = 4.7 × 10⁻³ at Z ≈ 3.35, i.e. at LZ's local 3.4σ.
- `figures/fig2_neff_vs_modelspace.png` — N_eff at 3.4σ and 3.0σ versus number of models in the subspace, with the Bonferroni line and LZ's implied 13.9.
- `figures/fig3_spectra_background.png` — background components and representative smeared signal shapes (O₁ˢ 30 GeV and 1 TeV, O₆ˢ, O₄ˢ, O₁ˢ δ = 300 keV);
  the dip at 100 keV is the xenon form-factor minimum.
- `figures/fig4_neff_vs_deltamax.png` — N_eff of a continuous δ scan versus δ_max, with Bonferroni.

## 10. References
LZ Collaboration, arXiv:2609.02823 (2026). E. Gross, O. Vitells, Eur. Phys. J. C 70, 525 (2010). G. Cowan, K. Cranmer, E. Gross, O. Vitells, Eur. Phys. J. C 71, 1554 (2011).
D. Baxter et al., Eur. Phys. J. C 81, 907 (2021). N. Anand, A. L. Fitzpatrick, W. C. Haxton, Phys. Rev. C 89, 065501 (2014). I. Jeong, S. Kang, S. Kim, S. Scopel,
Comput. Phys. Commun. 276, 108342 (2022) (WimPyDD). J. Billard, L. Strigari, E. Figueroa-Feliciano, Phys. Rev. D 89, 023524 (2014) (atmospheric-ν CEνNS floor).
Corpus: P001 (same-day Bayesian/frequentist reassessment; cited as companion, its numbers were not available at writing).

## 11. Tools and provenance (mirrors `output/provenance/P008.json`)
Software: python 3.12.13; numpy 2.5.3 (random.default_rng Poisson/uniform, searchsorted, linalg.eigvalsh, trapezoid); scipy 1.18.1 (stats.norm pdf/sf/isf, special.expit);
matplotlib 3.11.2 (Agg); pandas 3.0.5 (CSV tables); WimPyDD 2.0.4 (eft_hamiltonian, diff_rate, streamed_halo_function via lzcommon.wd_*); common/lzcommon.py
(LZ, LSIG, OSIG, LSIG_MASSES, OSIG_DELTAS, helm_F2, A_XE_MEAN, AMU_GEV, p_to_sigma, sigma_to_p, wd_halo, wd_hamiltonian, wd_rate).
Local inputs: LZ fulltext.tex (Results paragraph l.253–283; Table I l.210–246; Fig. 5 caption l.258–264; Supplement l.462–500 LEE; l.697–735 accidentals/MSSI;
l.787–798 neutrons; l.820–909 local-significance tables), `output/00_evidence_dossier.md`, `output/01_landscape_and_plan.md` (P008 row), `output/code/common/lzcommon.py`,
`output/code/P000_dossier_checks.py` (style), `environment/ENVIRONMENT_versions.txt`.
Recalled knowledge (flagged): atmospheric-ν flux shape (uncertain); (α,n) recoil-spectrum scale (uncertain); CEνNS cross-section formula (certain); sin²θ_W = 0.238 (likely, unused for shape);
Wilks/Chernoff half-χ² (certain); Gross–Vitells upcrossing formula (certain); Nyholt/Cheverud M_eff (likely); Bonferroni (certain); Anand et al. Lagrangian→operator reduction pairs (uncertain, not used quantitatively).
WimPyDD-generated files: none outside `output/` (only `output/work/P008/spectra_cache.npz`; `WD.diff_rate` writes no response-function files).
Datasets: none. Data requests: none.
