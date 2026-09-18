# P071 — Alternative look-elsewhere treatments for the LZ model scan: research record

Simulated date 2026-09-15. Author profile: look-elsewhere methodologists. Category STAT (physics.data-an). Competes with P008.
Script: `output/code/P071_lee_alternatives.py` (run from the simulation root with `.venv/bin/python`). Result tables in
`output/work/P071/*.csv`, `results.json`; figures in `output/work/P071/figures/`; full console log `run_full.log`.

## 1. Motivation and framework

LZ (arXiv:2609.02823) reports a maximum local significance of 3.4σ over 616 models (293 distinguishable spectra) and a global
2.6σ from toy Monte Carlo of the minimum p-value over its own model space (supplement, "Log-Likelihood Function and the Look
Elsewhere Effect"; ≥ 30,000 toys per model for the local p-values, "Local significance" section). The implied effective number of
trials is N_eff = ln(1 − p_global)/ln(1 − p_local) = 13.9. P008 reproduced this with a simplified one-dimensional likelihood
(N_eff = 12.2 ± 0.5 ± 1.2, 2.64σ; 300,000 toys) and found that N_eff grows with threshold (4.8/6.0/8.3/12.2 at 2.0/2.5/3.0/3.4σ),
that Bonferroni over 293 spectra gives 1.29σ, and that the asymptotic Gross–Vitells estimate underestimates the calibrated N_eff by ~2.5.

This paper asks how the global significance depends on the *method* used to correct for the look-elsewhere effect (LEE) and on
the *definition of the search space*, using the same likelihood so that the comparisons are method-to-method:
(i) Bonferroni/Šidák bounds; (ii) Gross–Vitells with upcrossings/Euler characteristics counted in the toys, in the calibrated-Z
metric as well as the asymptotic one; (iii) eigenvalue ("effective number of tests") estimators from the correlation matrix of the
per-model statistics; (iv) the Bayesian model-marginalised trials factor of P027; (v) pre-registration counterfactuals; (vi) the
"corpus LEE": the search space enlarged by the model variants this corpus has tested since 2 September.

## 2. Machinery reused from P008 (all imported from `output/code/P008_lee_toys.py`; its `main()` is not run)

* One-dimensional extended likelihood in observed NR-equivalent energy E ∈ [5.4, 270] keV inside the NR band, bin width 0.5 keV.
  Background (Table I of the paper projected on E): accidentals 2.7 events (15 keV exponential), atmospheric-ν CEνNS 0.11 (computed
  shape), ⁸B 0.057 (below threshold), MSSI 0.0049 (flat above 50 keV), neutrons 0.05 (40 keV exponential). Total B = 2.922 events,
  0.00216 above 200 keV, 2.7 × 10⁻⁵ /keV at 248 keV. Mean 2.92 events per toy.
* Signal shapes: WimPyDD spectra (cache `output/work/P008/spectra_cache.npz`, 501 spectra: O₁, O₃–O₁₅ × isoscalar/isovector × 13
  masses; inelastic O₁/O₄ × s/v × {400, 1000, 4000} GeV × δ = 0…350 keV; O₁ˢ 1 TeV δ scan 0–400 keV in 10 keV steps), smeared with
  σ_E = 23 √(E/248 keV) keV and multiplied by the LZ efficiency (50% at 5.4 and 269.9 keV, plateau 0.96). 456 of the 460 LZ-like
  spectra are physical (P008: 287 distinct at the 1% CDF level; LZ: 293).
* Profile-likelihood statistic q₀ = 2[ln L(ŝ) − ln L(0)], ŝ ≥ 0, background normalisation fixed; background-only Poisson toys;
  toy-calibrated local p per model, p_m(q) = P(q₀,m ≥ q | H₀) from an independent calibration set; p_min = min_m p_m over a model
  subset; p_global(p_loc) = P(p_min ≤ p_loc); N_eff = ln(1 − p_global)/ln(1 − p_loc); Z one-sided Gaussian.

## 3. New ingredients

### 3.1 Extra spectra tested by the corpus (`extra_spectra_cache.npz`, 161 spectra, 158 physical)
* **Light mediators (P051):** the cached O₁ˢ 1 TeV δ-scan spectra for δ = 250…390 keV (15 values) multiplied in true energy by the
  propagator factor [m²/(m² + q²)]², q² = 2 m_N E_R (m_N = 122.3 GeV; q = 0.246 GeV at 248 keV), for m_med = 0.05, 0.1, 0.2, 0.3, 0.5,
  1 GeV: 90 spectra. (No new WimPyDD call.)
* **Isospin ratios (P032):** elastic O₁ at 1 TeV with c_n/c_p = r for 41 values r ∈ [−1, 1] (step 0.05), plus inelastic O₁ at
  δ = 300 and 350 keV with r = −0.7, 0, +0.5: 47 WimPyDD spectra. Convention c⁰ = c_p + c_n, c¹ = c_p − c_n (PAPER_GUIDE / P003);
  check: the proton-only (r = 0) spectrum has its high-energy M-response node at the 305 keV grid point (grid spacing ≈ 27 keV there),
  consistent with P032's 292 keV (proton-only) and far from 247 keV (neutron-only).
* **Halo and date variants (P018, P006):** O₁ˢ 1 TeV at δ = 300, 350, 380 keV for v_esc = 500/528/560/600 km/s, v₀ = 220/250 km/s
  (others at Baxter-2021 defaults), and day-of-year 167 (16 June) and 350 (December): 24 spectra, 21 physical (δ = 380 keV is
  kinematically closed for the three slowest halos).
* **Continuous δ scan (P021):** P008's 41-point O₁ˢ δ scan (40 physical). Total in the toys: 654 models = 456 LZ-like + 198 extras
  (40 scan + 90 light-mediator + 47 isospin + 21 halo/date).

### 3.2 Toys
Four configurations of 50,000 background-only toys each (the assignment cap): pair A = calibration seed 20260915 + evaluation seed
20260916; pair B = 20260917 + 20260918. Every model is calibrated and evaluated in every toy, so all subspaces share the same toys.
Statistical (binomial) error on N_eff at 3.4σ from 50,000 evaluation toys: ±0.7; A/B half-difference 0.6 (LZ-like), 0.7 (corpus).

### 3.3 Finite-calibration bias and its correction (`calibration_size_dependence.csv`)
With N_cal calibration toys the per-model threshold at p_loc = 3.37 × 10⁻⁴ rests on ≈ N_cal p_loc toys (17 at 50k, 50 at P008's 150k).
A minimum over M ≈ 456 *noisily* calibrated p-values is biased low, so p_global and N_eff are biased high. We re-calibrated all
models on disjoint sub-samples of each calibration set (8 × 6,250; 4 × 12,500; 2 × 25,000; 1 × 50,000) and evaluated the same
50,000 evaluation toys. LZ-like space, N_eff at 3.4σ (mean over pairs and replicates): 20.55, 17.91, 16.67, 15.89 for
N_cal = 6.25k, 12.5k, 25k, 50k (pair A: 19.3/17.1/15.9/15.3; B: 21.9/18.7/17.4/16.5). At 2.0/2.5/3.0σ the dependence is 1%, 2%, 8%
over the same range. Fitting a + b/N_cal and a + b/√N_cal and averaging (half-range as systematic):

| Z_local | N_eff(50k) | a (1/N) | a (1/√N) | **N_eff debiased** | bias at 50k | Z_global |
|---|---|---|---|---|---|---|
| 2.0 | 4.88 | 4.88 | 4.86 | **4.87 ± 0.01** | +0.4% | 1.25σ |
| 2.5 | 6.15 | 6.15 | 6.09 | **6.12 ± 0.03** | +0.5% | 1.78σ |
| 3.0 | 9.30 | 9.18 | 8.84 | **9.01 ± 0.17** | +3.2% | 2.25σ |
| 3.4 | 15.89 | 15.28 | 13.13 | **14.20 ± 1.08** | +11.9% | **2.59σ** |

Corpus space (654 models): 23.71/21.73/20.55/20.10 → **18.72 ± 0.85** at 3.4σ (bias +7.4%), 2.50σ. All numbers quoted below as
"debiased" use this extrapolation; the same procedure was applied to every search space (columns `N_eff_deb@Z` in `search_spaces.csv`).
P008's 150k-toy value (12.2) should carry a residual bias of +4% (1/N scaling) to +7% (1/√N), i.e. a debiased 11.4–11.7.

## 4. Results

### 4.1 Main LZ-like space (456 spectra; LZ's 293)
p_global(3.4σ) = 5.3 × 10⁻³ raw → N_eff = 15.9 (A 15.3, B 16.5); **debiased N_eff = 14.2 ± 1.1 (extrapolation) ± 0.7 (stat), Z_global = 2.59σ.**
LZ: 13.9 implied, 2.6σ; P008: 12.2, 2.64σ. Threshold dependence (debiased): 4.87, 6.12, 9.01, 14.2 at 2.0, 2.5, 3.0, 3.4σ (P008: 4.8,
6.0, 8.3, 12.2). Raw fine grid (`threshold_dependence.csv`, A+B pooled, 100k evaluation toys): N_eff = 3.5 (1.0σ), 3.9 (1.3σ), 4.0 (1.5σ),
5.4 (2.3σ), 10.6 (3.1σ); points above 3.4σ (up to 18 at 4.0σ) are increasingly calibration-biased and are not used.

### 4.2 Bonferroni and Šidák (`bonferroni_sidak.csv`), local 3.4σ
| M | 1 | 6 | 30 | 36 | 60 | 140 | 293 | 456 | 616 | 654 |
|---|---|---|---|---|---|---|---|---|---|---|
| Bonferroni Z | 3.40 | 2.87 | 2.32 | 2.25 | 2.05 | 1.67 | **1.29** | 1.02 | 0.81 | 0.77 |
| Šidák Z | 3.40 | 2.87 | 2.32 | 2.25 | 2.05 | 1.68 | 1.32 | 1.07 | 0.89 | 0.85 |

Šidák differs from Bonferroni by < 0.1σ. Both are lower bounds on p_global and are wrong by the factor M/N_eff = 293/14.2 ≈ 21 in p
(the "cluster size" below). Ratio of the exceedance count: at 3.4σ, given that at least one model exceeds, on average 32.7 of the 456
models exceed together (M/N_eff = 28.7 raw, 32.1 debiased); at 2.0/2.5/3.0σ the clusters hold 95/75/51 models. The trials are
clusters of spectra sharing an energy window, not spectra.

### 4.3 Eigenvalue ("effective number of tests") estimators (`eigenvalue_estimators.csv`; LZ-like space, pair A evaluation toys)
Estimators on the M × M correlation matrix (eigenvalues λ): Nyholt–Cheverud M_eff = 1 + (M − 1)(1 − Var(λ)/M); Li–Ji Σ[I(λ ≥ 1) + (λ − ⌊λ⌋)];
participation ratio (Σλ)²/Σλ² = M²/Σλ²; Galwey (Σ√λ)²/Σλ; Gao (number of components for 99.5% variance). Variables: (a) Gaussianised Z
by randomised PIT (the atom at q₀ = 0, 55–82% of toys per heavy model, spread uniformly so the marginal is exactly N(0,1)); (b) √q₀
(P008's metric); (c) tail indicators I(p ≤ α).

| variable | Nyholt | Li–Ji | participation ratio | Galwey | Gao 99.5% | λ_max |
|---|---|---|---|---|---|---|
| Gaussianised Z | 384 | 179 | 6.3 | 197 | 444 | 162 |
| √q₀ | 303 | 16 | 3.0 | 8.6 | 14 | 222 |
| I(p ≤ 0.1) | 352 | 59 | 4.4 | 46 | 193 | 176 |
| I(p ≤ 0.01) | 377 | 69 | 5.7 | 55 | 206 | 138 |
| I(p ≤ 0.001) | 396 | 85 | 7.5 | 61 | 172 | 120 |

The estimators span 3 to 444 (Z_global from 0.5σ to 3.2σ at local 3.4σ); Nyholt/Cheverud and Gao are dominated by the hundreds of
tiny eigenvalues of near-degenerate spectra and are meaningless here (P008 found 302). Only the participation ratio is in the right
range, and only when computed on tail indicators: 4.4/5.7/7.5 at α = 0.1/0.01/0.001 versus toy N_eff 3.9/5.4/10.6 at the same thresholds
(1.28/2.33/3.09σ), i.e. within ×1.15–0.7, and it reproduces the growth with threshold. None of these is a substitute for toys; the
eigenvalue methods answer a linear-correlation question, whereas the LEE is a joint-tail question.

### 4.4 Gross–Vitells with upcrossings counted in the toys (`gross_vitells.csv`)
Calibrated metric: Z_cal = Φ⁻¹(1 − p_m), marginally N(0,1) for q₀ > 0 by construction (−∞ for q₀ = 0). Asymptotic metric: √q₀.

*1D, O₁ˢ 1 TeV δ scan (41 points, 0–400 keV).* Mean upcrossings E[N_up(c₀)] of the calibrated field: 0.480, 0.311, 0.163, 0.0653,
0.0191, 0.00764 at c₀ = 0.5, 1.0, 1.5, 2.0, 2.5, 3.0; N_up e^{c₀²/2} = 0.54, 0.51, 0.50, 0.48, 0.43, 0.69 — Gaussian-like scaling to
c₀ = 2.5, then a 40% excess at 3.0 (excursion sets fragment into individual resolution windows). GV, P(max ≥ c) ≈ Φ̄(c) + N_up(c₀) e^{−(c²−c₀²)/2}:
N_eff(3.4σ) = 6.0, 5.7, 5.6, 5.4, 5.0, 7.3 from c₀ = 0.5…3.0, versus 10.1 directly in the same toys (raw; P008: 9.3): **GV underestimates
by ×1.7–2**. Asymptotic metric: GV 5.2, 4.7, 4.0, 3.9, 4.3, 5.8 versus 2.4 for the direct max of √q₀ in its own metric: in the
asymptotic metric GV *over*estimates its own field's tail by ×2 (the √q₀ field has a lighter joint tail than Gaussian, because
P(q₀ > 11.56) ≪ 3.4 × 10⁻⁴ for heavy models) while underestimating the calibrated answer by ×2 — the two errors that P008 saw combined as "~2.5".

*2D, the (m, δ) grids of O₁ˢ, O₁ᵛ, O₄ˢ, O₄ᵛ (4 × 3 masses × 8 splittings; the unphysical (400 GeV, 350 keV) cell set to −∞).*
Euler characteristic χ = V − E + F of the excursion set on the grid graph, summed over the four grids: E[χ] = 1.65, 1.03, 0.51, 0.19 at
c₀ = 0.5, 1.0, 1.5, 2.0. Fit E[χ(c)] = 4Φ̄(c) + (N₁ + N₂c)e^{−c²/2}: N₁ = 0.42, N₂ = 0.19. Extrapolated N_eff(3.4σ) = 13.8 (calibrated) versus
11.2 raw / 10.8 debiased directly (**+25–30%**); at 2.0σ 9.7 versus 3.1 (×3: at low thresholds the excursion set is one blob per grid
because the three masses are degenerate, and χ ≈ 1 per grid is not a Gaussian-field count). Asymptotic metric: 9.9 versus 2.6.
Conclusion: even with the counts taken from toys rather than from an asymptotic formula, GV brackets the toy answer only within a
factor 2 (−45% in 1D, +30% in 2D); the field is neither stationary nor Gaussian (single events in discrete windows), so the
Gaussian extrapolation e^{−c²/2} from low thresholds has no privileged status here.

### 4.5 Threshold dependence, cost in σ, extrapolation to 5σ (`extrapolation_5sigma.csv`)
Fits to the debiased N_eff(Z) at Z = 2, 2.5, 3, 3.4 (errors = max(half-range, 5%)): GV form N_eff = 1 + N e^{−c²/2}/Φ̄(c) ≈ 1 + N c√(2π)
(linear growth): N = 0.78, χ² = 56/3 (rejected); power law a c^k: a = 1.29, k = 1.81, χ² = 12; exponential a e^{bc}: a = 1.10, b = 0.72,
χ² = 6.6. (Raw 17-point fits give k = 1.79, b = 0.73.) The growth is faster than the Gaussian-field expectation because the joint tail is
made by one high-energy event landing in one of several resolution-wide windows, each lifting a different cluster of spectra; at low
thresholds any event above ~50 keV lifts all heavy models at once (cluster 95 of 456 at 2σ), at 3.4σ only the ~30 spectra whose
accepted spectrum peaks in the event's window (cluster 33).

LEE cost Z_local − Z_global (debiased toys): 0.75σ (2.0), 0.72σ (2.5), 0.75σ (3.0), **0.81σ (3.4)**; P008/LZ: 0.76–0.8σ. Although N_eff
triples between 2σ and 3.4σ the cost is flat, because ΔZ ≈ ln N_eff / Z_local (p_global ≈ N_eff p_local; ln 14.2/3.4 = 0.78).
Extrapolation to a 5σ local excess in the same space: N_eff(5σ) = 11 (GV form), 24 (power law), 39 (exponential), giving
Z_global = 4.51, 4.35, 4.24σ and **cost(5σ) = 0.49–0.76σ**; the Bonferroni bound (287 distinct spectra) gives cost 1.23σ, and a
saturation at 30 independent windows 0.70σ. A 5σ local excess in this model space would therefore remain ≥ 4.2σ global.

### 4.6 Pre-registration counterfactuals (`search_spaces.csv`; debiased, local 3.4σ)
| search space | models | N_eff | Z_global |
|---|---|---|---|
| (c) one model, O₆ˢ 1 TeV (L₁₀/L₄-like) | 1 | 0.84 ± 0.07 | 3.45σ |
| (c′) one model, O₁ˢ 1 TeV δ = 350 keV | 1 | 1.09 ± 0.07 | 3.38σ |
| (b1) six a-priori high-energy spectra: O₆ˢ, O₁₀ˢ, O₉ˢ, O₁₅ˢ at 1 TeV (L₁₀/L₄, L₂, L₁₆-like, q²-spin), O₁ˢ inelastic δ = 300, 350 keV | 6 | 3.0 ± 0.3 | 3.09σ |
| (b2) six spectra with the largest accepted fraction above 200 keV (all inelastic δ = 300–350 keV: O₁ᵛ 1000/4000/400 GeV, O₄ˢ 1000/4000, O₄ᵛ 1000) | 6 | 1.5 ± 0.2 | 3.29σ |
| (a) 14 isoscalar operators at 1 TeV + O₁ˢ/O₄ˢ inelastic 1 TeV × 8 δ (LZ's "20 L + 2 inelastic × 8 δ = 36" analogue) | 30 | 6.5 ± 0.8 | 2.85σ |
| (a′) same with isovector partners | 60 | 7.4 ± 0.9 | 2.81σ |
| heavy elastic m ≥ 100 GeV | 140 | 4.6 ± 0.6 | 2.96σ |
| inelastic O₁/O₄ grid | 92 | 10.8 ± 0.7 | 2.69σ |
| LZ-like full space | 456 | 14.2 ± 1.1 | 2.59σ |

Single-model N_eff below 1 reflects the discreteness of the calibrated p-values with ~3 events (P008: 0.7–1.4). The six inelastic
spectra of (b2) all select the same 300–350 keV window and cost only 0.1σ; the six *different* spectral families of (b1) cost 0.3σ.
A 30–36-spectrum pre-registration (one mass, isoscalar, plus the inelastic grid) would have reported 2.85σ, not 2.6σ; the assignment's
guess of 2.9–3.1σ holds only for ≤ 6–10 spectra.

### 4.7 The corpus LEE (`search_spaces.csv`, cumulative, debiased, local 3.4σ)
| space | models | N_eff | Z_global |
|---|---|---|---|
| LZ-like | 456 | 14.2 | 2.59σ |
| + continuous δ scan to 400 keV [P021] | 496 | 16.9 (+19%) | 2.53σ |
| + light mediators 6 m_med × 15 δ [P051] | 586 | 18.4 (+9%) | 2.50σ |
| + isospin ratios 41 r + 6 inelastic r [P032] | 633 | 18.7 (+1.6%) | 2.50σ |
| + halo/date variants [P018, P006] | 654 | 18.7 (+0.3%) | **2.50σ** |
| corpus extras alone | 198 | 15.2 | 2.57σ |

The 198 community variants carry as many trials on their own (15.2) as LZ's whole space, but they overlap almost entirely with it: the
union has 18.7, not 29. The δ scan beyond 350 keV adds the most (new windows at 360–390 keV, cf. P021's 3.6σ peak), the light mediators
a little (they reshape the same windows), isospin and halo variants essentially nothing (they change rates, hardly shapes). The corpus
enlarges N_eff by ×1.32 ± 0.1; applied to P008's 12.2 (→ 16.1, 2.55σ) or LZ's 13.9 (→ 18.3, 2.50σ): **the honest global significance of the
3.4σ event after two weeks of community reinterpretation is 2.50–2.55σ.** At P021's local 3.6σ the corpus space gives 2.62σ (raw 50k-toy
value; the +8% bias would make it ≈ 2.65σ). In our 1D likelihood the data (248 keV + 3 low-energy events) reach a calibrated local
3.94σ (O₄ˢ 4000 GeV δ = 350; P008 3.76 — calibration noise) and 3.16σ global in the LZ-like space, 2.89σ in the corpus space
(`data_own_global.csv`); the 198 extras do not beat the LZ-like maximum (best extra: isospin r = −0.7, δ = 350 keV, same window).

### 4.8 Bayesian comparison (results.json → `bayesian`)
P027: model-marginalised B_DM = 16.4 (uniform prior over 298 spectra) versus B_max = 171, so B_max/B_DM = 10.3. With uniform weights
B_DM = (1/M) Σ_m B_m, hence B_max/B_DM = M / Σ_m (B_m/B_max) ≡ M/M_share, where M_share = 298/10.3 = 29 is the number of spectra that
share the best model's evidence for the *observed* event. The frequentist N_eff has the same structure: N_eff = M/⟨#spectra exceeding |
any exceeds⟩ = 456/32 = 14.2 at 3.4σ, with the cluster size an *ensemble average* over background-only exceedances. The two agree
(10.3 vs 14.2) because the actual event is a typical exceedance: it sits in the 200–270 keV window where the heavy-operator and
δ ≥ 250 keV spectra pile up, so the observed sharing (29) matches the ensemble cluster (32). Used as a trials factor, 10.3 gives
Z_global = 2.70σ. The profile-likelihood analogue on LZ's own Tables S6/S7 (612 finite entries, q₀ = Z²): M/Σ e^{−(q_max−q_m)/2} = 7.6
(cluster 81 entries; 128 entries ≥ 3σ, 223 within 1σ of the maximum); on P008's data q₀ (asymptotic Z) 18.6; on calibrated Z 85 (the
calibrated Z of low-significance models is inflated by discreteness — not meaningful). The Sellke–Bayarri–Berger bound from our
global p (4.8 × 10⁻³) is 13.2 (P001: 14.7 from 2.6σ). Interpretation: the Bayesian factor is data-conditional and prior-weighted
(class priors halve it, P027), the frequentist one threshold-dependent (4.9 at 2σ, 14 at 3.4σ, ~24–39 at 5σ); their agreement at
3.4σ is structural, not a coincidence, and neither is "the" trials factor.

## 5. Figures
* `figures/fig1_globalZ_by_method.png` — global significance at local 3.4σ by method: Bonferroni/Šidák bounds (0.8–1.3σ), eigenvalue
  M_eff (2.0–2.8σ for the two least absurd variants), Gross–Vitells on the inelastic grids (2.60–2.71σ vs 2.69σ direct), Bayesian
  trials factor (2.70σ), toy min-p LZ-like (2.59σ this work, 2.64σ P008, 2.6σ LZ) and corpus space (2.50σ).
* `figures/fig2_neff_vs_threshold.png` — left: N_eff vs Z_local (raw A+B toys, debiased diamonds, P008 crosses, three fitted forms
  extrapolated to 5σ, Bonferroni 287); right: LEE cost Z_local − Z_global, flat at 0.7–0.8σ, extrapolations 0.5–0.8σ at 5σ.
* `figures/fig3_counterfactuals.png` — global significance for the pre-registration counterfactuals, LZ-like spaces and cumulative
  community additions (debiased), with LZ's 2.6σ and the local 3.4σ marked.

## 6. Validation and robustness
* Reproduction: at 2.0/2.5/3.0σ the debiased N_eff (4.87/6.12/9.01) matches P008 (4.8/6.0/8.3) within 1–8%; at 3.4σ 14.2 ± 1.1 vs
  12.2 ± 0.5 ± 1.2 (P008 carries a +4–7% residual calibration bias by the same scaling, so the debiased values are 14.2 vs ≈ 11.5, a
  1.5–2σ tension within the combined uncertainties). LZ's 13.9 lies between.
* Two independent toy pairs: N_eff(3.4σ) = 15.3 (A) and 16.5 (B) raw; sub-sample replicates scatter with std 2.0–2.5 at 6,250 toys.
* Calibration-size scaling: the 1/N and 1/√N extrapolations differ by 2.2 at 3.4σ (the quoted ±1.1) and by ≤ 0.35 below 3σ.
* Isospin convention check (§3.1); light-mediator spectra reproduce P051's qualitative behaviour (the best local Z stays at δ ≈ 380–390 keV).
* Not varied here (see P008): background composition (P008: N_eff 8.7–13.2 for ×3/÷2 high-energy background), resolution, efficiency edges.

## 7. Failed or abandoned approaches
* A first full run with the eigenvalue section in float64 died silently (the pipeline's exit status masked it); rerun with the
  calibration matrices released and float32 intermediates.
* Raw N_eff above 3.4σ from 50k-toy calibration (up to 18 at 4σ) was initially plotted as a rising LEE cost; identified as the
  finite-calibration bias and replaced by the sub-sample extrapolation. The assignment's cap of 5 × 10⁴ toys per configuration
  prevents the direct fix (more calibration toys).
* Nyholt/Cheverud and Gao estimators: uninformative (kept in the table as a warning).
* Gross–Vitells on the 2D grids at low thresholds (c₀ ≤ 1): χ ≈ one blob per grid; the fit is driven by c₀ = 1.5–2.0.

## 8. Discussion
The LEE correction for one event is robust in *method* only when the method is the toy min-p distribution itself: bounds are off
by ×20 in p, eigenvalue estimators by factors up to 30 either way, and Gross–Vitells by a factor 2 even with toy-counted upcrossings,
because the statistic field is a set of discrete energy windows lit by single events, not a Gaussian random field. The correction is
robust in *value* — 2.5–2.7σ — across LZ's toys, P008, this work and the Bayesian analogue, and the reason is structural: every
approach divides the model count by a cluster size of ~30 spectra sharing one energy window. What is *not* robust is the search-space
definition: a six-spectrum pre-registration would have reported 3.1–3.3σ, a 30-spectrum one 2.85σ, LZ's 293 gives 2.6σ, and the
community's ~200 post-hoc variants take it to 2.50σ. The trials factor grows with the local threshold (≈ Z^1.8) but the cost in σ does
not, so a 5σ local excess in the same space would survive as ≥ 4.2σ global; the LEE is a 0.7–0.8σ tax at every threshold, not a
growing one.

## 9. References
LZ Collaboration, arXiv:2609.02823 (2026). E. Gross, O. Vitells, Eur. Phys. J. C 70, 525 (2010). O. Vitells, E. Gross, Astropart.
Phys. 35, 230 (2011). G. Cowan, K. Cranmer, E. Gross, O. Vitells, Eur. Phys. J. C 71, 1554 (2011). D. R. Nyholt, Am. J. Hum. Genet.
74, 765 (2004). J. Li, L. Ji, Heredity 95, 221 (2005). T. Sellke, M. J. Bayarri, J. O. Berger, Am. Stat. 55, 62 (2001). D. Baxter
et al., Eur. Phys. J. C 81, 907 (2021). Corpus: P001, P006, P008, P016, P018, P021, P027, P032, P045, P051.

## 10. Tools and provenance (mirrors `output/provenance/P071.json`)
python 3.12.13 (`.venv/bin/python`); numpy 2.5.3 (default_rng, searchsorted, corrcoef, linalg.eigvalsh, polyfit, lstsq, trapezoid);
scipy 1.18.1 (stats.norm.sf/isf/pdf, special.expit); pandas 3.0.5; matplotlib 3.11.2 (Agg); WimPyDD 2.0.4 via lzcommon
(wd_halo with v0/vesc/day_of_year, wd_hamiltonian, wd_rate) for 71 new spectra; common/lzcommon.py (LZ, LSIG, OSIG, OSIG_DELTAS,
A_XE_MEAN, AMU_GEV, p_to_sigma, sigma_to_p); output/code/P008_lee_toys.py (imported: build_background, total_background, model_list,
compute_spectra, signal_pdfs, run_toys, q0_for_events, Calibrator, global_stats, neff_from_pglobal, sigma_E). Inputs read: PAPER_GUIDE,
dossier, ledger, P001/P008/P016/P021/P027/P045 papers, fulltext.tex LEE supplement (lines 462–500) and Local significance + Tables
S6/S7 (820–909), P008 script/results/provenance/details, lzcommon stats helpers, ENVIRONMENT_versions. Recalled knowledge (8 items):
Bonferroni/Šidák (certain); Gross–Vitells upcrossing formula and 2D Euler-characteristic form (certain/likely); Nyholt–Cheverud,
Li–Ji, Galwey, Gao estimators (likely); Sellke–Bayarri–Berger bound (certain); light-mediator propagator [m²/(m²+q²)]² (certain).
Commands: `SPECTRA_ONLY=1 .venv/bin/python output/code/P071_lee_alternatives.py` (extra WimPyDD cache, 60 s);
`.venv/bin/python output/code/P071_lee_alternatives.py > output/work/P071/run_raw.log 2>&1` (full run, 4 × 50,000 toys, ≈ 250 s).
WimPyDD-generated files: `output/work/P071/extra_spectra_cache.npz` (our cache of diff_rate outputs; no response-function files written).
