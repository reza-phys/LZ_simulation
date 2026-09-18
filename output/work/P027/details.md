# P027 — Bayesian model comparison across LZ's model space given one 248 keV event: research record

Simulated date 2026-09-08. Author profile: Bayesian statisticians in astroparticle physics. Category STAT.
Script: `output/code/P027_bayes_models.py` (run from the simulation root with `.venv/bin/python`). All numbers below are
in `output/work/P027/P027_results.json`, the CSV tables listed in §9, or `run_log.txt`.

## 1. Motivation and question

LZ tested 616 signal models (293 distinguishable spectra) against one 248 keV NR-band event and reduced the maximal local
3.4σ to a global 2.6σ with background-only toys (LEE). P001 converted the event into single-model Bayes factors
(B₁₀ ≈ 100–500 against the modelled background, ÷ ~14 for the trials), P008 reproduced the trials factor
(N_eff = 12.2 ± 0.5 ± 1.2), P003 computed the low-energy companions N_lo of every NREFT operator, and P016 built a
joint low/high-energy likelihood and Bayes factors per operator at 1000 GeV (L10 44.8, O6 18.3, …, O1 1.0). None of these
performed the Bayesian analogue of the LEE: a prior over the *models*, coupling marginalised model by model, the
posterior over the model space, and the model-marginalised Bayes factor B_DM = Σ_m π_m Z_m / Z_bkg. That is what we do
here, including (i) three coupling priors and two parametrisations, (ii) five model priors, (iii) a continuous prior on
the inelastic mass splitting δ up to the kinematic ceiling (the Bayesian version of P002's "248 keV is atypical for
δ ≤ 350 keV"), (iv) an optional in-window spectral shape factor for the event's position at 248 keV, and (v) posterior
predictive distributions for the P020 exposure.

## 2. Model space

| block | source | count |
|---|---|---|
| elastic O1, O3–O15, isoscalar/isovector, 13 masses (Table S6 grid) | P008 WimPyDD cache `output/work/P008/spectra_cache.npz` (time-averaged Baxter-2021 halo, 60-point 2–400 keV grid) | 364 |
| L10 ≡ (q²/m_N²)O4 − O6 (transverse-spin, P003), 13 masses | recomputed with WimPyDD 2.0.4 (`eft_hamiltonian` with q-dependent coefficient, `lz.wd_rate`, `A = 1/m_v²`) | 13 |
| inelastic O1^s, O1^v, O4^s, O4^v at 400/1000/4000 GeV, δ = 0…350 keV (Table S7 grid) | recomputed with WimPyDD (patched `wd_halo` v_min grid); O4^v on the LZ grid only (LZ: O4 s/v spectra indistinguishable) | 96 (92 physical) |
| δ scan O1^s, O1^v, O4^s at 400/1000/4000 GeV, 10 keV steps to the kinematic ceiling | WimPyDD | 283 extra points |

Energy grid for recomputed spectra: 28 log points 2–150 keV plus 5 keV steps 155–400 keV (78 points); spectra are
log-interpolated onto a 0.5 keV grid. Total 752 spectra; 469 physical LZ-grid models. **Distinguishable spectra:**
P008's 1 %-CDF degeneracy flag (287 of 456 distinct) plus LZ's own rule for L10 (masses ≥ 400 GeV degenerate → 11 L10
representatives) gives **298 distinguishable spectra** (LZ: 293): 240 elastic (of which 123 have m ≤ 50 GeV and cannot
produce a 248 keV recoil), 58 inelastic (O1^s 15, O1^v 18, O4^s 19, O4^v 6). Our operator basis is not LZ's Lagrangian
list L1–L20 (only L1 = O1, L2 = O10, L3 = O11, L4 = O6, L15 = O4 and L10 map directly; mapping recalled, likely), but it
spans the same shape families: unsuppressed M (O1, O11), q²-suppressed spin (O5^v, O6, O9, O10, O14, O15, L10), velocity
suppressed (O7, O8), Δ/Φ'' (O3, O12, O13), and Σ (O4).

Kinematics (annual-average halo, v_max = v_esc + v_E = 794.6 km/s): ceiling μv²_max/2 = 329.1 / 383.0 / 417.1 keV and
δ_max(248 keV) = 329.1 / 374.6 / 397.4 keV at 400 / 1000 / 4000 GeV (P002's June values are 341 / 387 / 409 keV). The
(400 GeV, 350 keV) point has zero rate, reproducing LZ's "not physical" dash.

Validation against P016/P003: for the 1000 GeV elastic operators the companion ratios N_lo recomputed from the P008 cache
agree with P016's (P003-cache) values to a median ratio 1.007 (range 0.995–1.037); L10 N_lo = 0.199 (P003: 0.20).
Table `P027_validation_Nlo.csv`.

## 3. Likelihood (P016's few-bin joint likelihood)

Data D: (i) high-energy bin, the 200–270 keV NR band: n_H = 1, b_H = 5.695 × 10⁻⁴ (P016's anchor that gives L10 its 3.4σ;
`P016_results.json: baseline_settings.b_H`); (ii) 20 low-energy NR-band bins (0.5σ_NR bins from −8σ to +2σ of the NR median,
digitised by P016 from Fig. 5's S1c < 250 phd panel): total b = 121.1, n = 121 (within ±1.5σ: b = 58.6, n = 42), with a
background scale nuisance θ ~ N(1, 0.3) (41-point grid, marginalised, not profiled); (iii) a 125–200 keV bin: b_M = 0.02,
n = 0. For a model m with efficiency-weighted true-energy window rates R_lo (5.4–55 keV), R_L2 (55–125), R_M (125–200),
R_hi (200–270) (P003 efficiency: 0.96 plateau, erf roll-offs σ = 3.4 keV at 5.4 keV and σ = 8 keV at 269.9 keV) define
fractions f_x = R_x / ΣR and companions N_x = R_x / R_hi. With total ROI signal μ the expected counts are μ f_top g_i in
low-energy bin i (g_i = Gaussian NR-band shape, f_top = f_lo + f_L2), μ f_M in the middle bin and s = μ f_hi in the
high-energy bin:

  L(μ)/L(0) = e^{−s} (1 + ρ s / b_H) · W(μ),  W(μ) = ⟨Π_i Pois(n_i | θb_i + μ f_top g_i) · Pois(0 | b_M + μ f_M)⟩_θ / ⟨Π_i Pois(n_i | θb_i) · e^{−b_M}⟩_θ .

ρ is the in-window **shape factor**: ρ_m = 70 keV × f_m^obs(248 keV), where f_m^obs is the model's observed-energy pdf
(Gaussian resolution σ = 23√(E/248) keV, efficiency at observed energy) normalised over 200–270 keV; a flat background in
the window is assumed (P008: 2.7 × 10⁻⁵/keV at 248 keV vs 0.0022/70 keV = 3.1 × 10⁻⁵/keV, i.e. flat to 15 %). ρ = 1
recovers P016's binned treatment (baseline). A variant uses σ√2 = 32.5 keV (stat ⊕ sys) → ρ_32.

## 4. Coupling priors and marginal likelihoods

Bayes factor of model m against background: B_m = ∫ π(x) [L/L(0)](x) dx, with

- **s-parametrisation** (assignment; P001/P016 compatible): x = s = expected 200–270 keV signal events, μ = s / f_hi.
  Priors (a) log-uniform on [10⁻³, 30]; (b) uniform on [0, 10]; (c) Jeffreys-like ∝ s^{−1/2} on [0, 10] (substitution s = t²).
  If f_hi = 0 (m ≤ 50 GeV) the model cannot produce any high-energy event: B_m = 0 for every s-prior.
- **μ-parametrisation**: x = μ = expected signal events in the whole 5.4–270 keV ROI, same three priors. Light models
  are then merely disfavoured by the low-energy null: B_m = 0.709 for every m ≤ 50 GeV model (and 0.72–0.86 for O1^s at
  100–1000 GeV).
- **P016's approximation** 1 + (1/b_H)∫π(s) s e^{−s} W(s) ds (ρ = 1) is also evaluated for cross-checking; it replaces
  ∫π e^{−s} W ds by 1 and therefore floors every B at 1. The exact integral shows that spectra with hundreds of companions
  are *disfavoured* relative to background under an s-prior (O1^s 1000 GeV: B = 0.062 log-uniform, 2 × 10⁻⁴ flat), because
  the prior forces s ≥ 10⁻³ high-energy events and hence ≥ 3 low-energy events.

Quadrature: 500-point log grid (log-uniform), 4001-point linear grid (uniform), 3001-point grid in t (Jeffreys); all
likelihoods in log space with `scipy.special.logsumexp` over θ.

**Companion-free reference** (f_hi = 1, ρ = 1): B = 170.8 (log-uniform), 175.6 (uniform), 246.3 (Jeffreys); P016-style 176.5
(P016: 176). P001's B₁₀ = 101 at b = 10⁻³ and 501 at 2 × 10⁻⁴ bracket this.

**Cross-check against P016's flat-prior table (1000 GeV)**, P016-style formula: O6^s 18.36 vs 18.34; O10^s 6.34 vs 6.34;
O4^s 1.56 vs 1.56; O1^s 1.00 vs 1.00; **L10 31.6 vs 44.8**. The L10 difference is a genuine correction: P016 approximated
L10's 55–200 keV companions by scaling O6's shape (N_M = 0.70), whereas the WimPyDD L10 spectrum peaks near 200 keV
(P003's double peak) and has N_M(125–200 keV) = 1.16 with N_L2 = 0.44; the empty 125–200 keV bin (b_M = 0.02) penalises this
by ~e^{−s N_M}. Exact flat-prior B for L10: 30.6. Table `P027_validation_B_vs_P016.csv`.

### 4.1 Per-model Bayes factors (selection; full table `P027_models.csv`, 1000 GeV elastic `P027_elastic_1000GeV.csv`)

| model | N_lo | N_M | f_hi | ρ | B log-u | B flat | B Jeffreys | B log-u, ρ | B_μ log-u |
|---|---|---|---|---|---|---|---|---|---|
| L10 1000 GeV | 0.20 | 1.16 | 0.356 | 0.91 | 71.4 | 30.6 | 66.5 | 65.1 | 71.7 |
| O6^s 1000 | 0.43 | 1.52 | 0.232 | 0.96 | 54.0 | 17.4 | 43.6 | 52.1 | 54.3 |
| O15^s 1000 | 1.93 | 0.54 | | 0.97 | 53.9 | 17.0 | 43.2 | 52.4 | 54.2 |
| O5^v 1000 | 1.05 | 1.33 | | 1.00 | 51.8 | 16.0 | 40.9 | 51.8 | 52.1 |
| O9^v / O9^s 1000 | 2.2 / 2.1 | 1.6 | | 0.87 / 0.85 | 45.7 / 45.2 | 12.4 / 12.1 | 33.9 / 33.3 | 39.7 / 38.4 | |
| O14^v / O14^s 1000 | 2.5 / 2.4 | 1.6 / 1.7 | | 0.86 / 0.84 | 43.4 / 43.0 | 11.2 / 11.0 | 31.4 / 30.9 | 37.4 / 36.2 | |
| O13^s 1000 | 4.9 | 0.77 | | 0.95 | 38.2 | 8.5 | 25.7 | 36.2 | 38.6 |
| O10^s 1000 | 3.1 | 2.2 | 0.101 | 0.89 | 30.2 | 5.4 | 18.1 | 27.1 | 30.6 |
| O11^v 1000 | 35 | 1.1 | 0.025 | 1.30 | 10.4 | 0.60 | 3.6 | 13.4 | 10.9 |
| O4^s 1000 (elastic) | 28 | 2.85 | 0.026 | 0.81 | 10.0 | 0.57 | 3.4 | 8.2 | 10.5 |
| O11^s 1000 | 266 | 7.3 | 0.0033 | 0.40 | 1.41 | 0.012 | 0.21 | 0.66 | 2.1 |
| O1^v 1000 | 493 | 1.9 | 0.0020 | 1.26 | 0.83 | 0.0047 | 0.11 | 1.02 | 1.57 |
| O1^s 1000 | 2826 | 10.0 | 3.4 × 10⁻⁴ | 0.36 | 0.062 | 2 × 10⁻⁴ | 0.015 | 0.033 | 0.86 |
| O1^s 100 GeV | 7.2 × 10⁴ | 32 | 1.4 × 10⁻⁵ | 0.23 | 0.000 | 10⁻⁴ | 0.002 | 0.000 | 0.72 |
| O6^s 200 / 100 GeV | 1.27 / 12.2 | 2.3 / 6.3 | 0.130 / 0.027 | 0.86 / 0.62 | 34.8 / 9.3 | 7.2 / 0.49 | 22.5 / 3.1 | 30.1 / 5.9 | 35.2 / 9.8 |
| L10 200 / 100 GeV | 0.60 / 6.1 | 1.7 / 4.2 | 0.234 / 0.061 | 0.82 / 0.61 | 52.3 / 18.4 | 16.3 / 2.0 | 41.6 / 8.6 | 43.0 / 11.3 | 52.6 / 18.8 |
| any m ≤ 50 GeV | ∞ | ∞ | 0 | 0 | 0 | 0 | 0 | 0 | 0.709 |
| O1^v 1000, δ = 350 | 0 | 0.012 | 0.989 | 1.40 | 168.8 | 171.6 | | 236.5 | 168.8 |
| O4^s 1000, δ = 350 | 0 | 0.072 | 0.933 | 1.23 | 159.3 | 152.9 | | 195.3 | |
| O1^s 1000, δ = 350 | 0 | 0.54 | 0.649 | 0.68 | 111.0 | 74.1 | | 76.2 | |
| O1^v 1000, δ = 300 | 0 | 0.32 | 0.709 | 1.32 | 126.2 | 95.9 | | 166.1 | |
| O1^s 1000, δ = 300 | 0 | 4.05 | 0.195 | 0.44 | 33.8 | 6.8 | | 15.3 | |
| O1^v / O1^s / O4^s 1000, δ = 250 | 0 | 0.78 / 6.2 / 1.7 | 0.36 / 0.12 / 0.30 | 1.29 / 0.40 / 0.88 | 80.0 / 22.5 / 58.1 | 38.3 / 3.0 / 20.2 | | 103.1 / 9.3 / 51.1 | |
| O1^s 1000, δ = 200 | 7.1 | 7.6 | 0.036 | 0.38 | 11.0 | 0.71 | | 4.4 | 11.5 |
| O1^v 1000, δ = 0 | 475 | 1.76 | 0.0021 | 1.26 | 0.87 | 0.005 | | 1.06 | |

Of the 117 distinguishable heavy (m ≥ 100 GeV) elastic spectra, 61 have B > 10 and 18 have B < 1 (median 11.1, log-uniform);
of the 58 inelastic spectra 47 have B > 10 and 8 have B > 100. Shape factors: heavy elastic ρ from 0.23 (O1^s 100 GeV;
248 keV sits 19 keV below the M-response node, P017) to 1.31 (O11^v); inelastic O1^s ρ = 0.36–0.57 for δ ≤ 340 keV, 0.68
at 350, 0.90 at 360, 1.24 at 370 keV (1000 GeV); O1^v 1.26–1.40; O4^s 0.8–1.2. The isovector M response has no node near
248 keV in our WimPyDD spectra (Z − N weighting), which is why O1^v tops the model list once ρ is used.

## 5. Model priors, B_DM and the posterior over models (table `P027_BDM_vs_prior.csv`)

Model priors over the 298 distinguishable spectra: (P1) uniform; (P2) uniform over the four classes {elastic-isoscalar
(147 incl. L10), elastic-isovector (93), inelastic O1 (33), inelastic O4 (25)} then uniform within; (P3) "physics-weighted":
inelastic (both operators) = elastic (both isospins) = ½, uniform within; (P4) uniform over the 469 raw physical models
(no degeneracy merging); (P5) uniform over the 175 heavy (m ≥ 100 GeV) spectra only.

| model prior | n | B_DM log-u s | uniform s | Jeffreys | log-u s, ρ | log-u μ | B_max/B_DM | N_eff(post) | models for 50 % / 90 % | prior mass with B = 0 | P(inel) | P(δ ≥ 300) | P(q²-spin) | P(SI-like) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 uniform spectra | 298 | **16.4** | 6.6 | 13.8 | 15.9 | 17.0 | 10.3 | 99 | 29 / 90 | 0.413 | 0.576 | 0.322 | 0.301 | < 10⁻⁴ |
| P2 uniform 4 classes | 298 | **28.7** | 13.8 | 26.7 | 28.9 | 29.2 | 5.9 | 66 | 16 / 64 | 0.252 | 0.848 | 0.463 | 0.107 | < 10⁻⁴ |
| P3 physics-weighted | 298 | **28.7** | 14.0 | 27.0 | 29.1 | 29.1 | 5.9 | 65 | 15 / 64 | 0.256 | 0.849 | 0.475 | 0.107 | < 10⁻⁴ |
| P4 raw models | 469 | 13.7 | 5.5 | 11.5 | 13.2 | 14.3 | 12.3 | 129 | 37 / 116 | 0.495 | 0.585 | 0.332 | 0.313 | < 10⁻⁴ |
| P5 heavy only | 175 | 28.0 | 11.2 | 23.5 | 27.1 | 28.5 | 6.0 | 99 | 29 / 90 | 0 | 0.576 | 0.322 | 0.301 | < 10⁻⁴ |

(B_max = 168.8 is O1^v 1000 GeV δ = 350 keV; with ρ the top model has 236.5. Posterior quantities in the last five
columns are for log-uniform s, ρ = 1; with ρ: N_eff(post) and class masses shift by < 20 %, e.g. P(inel) = 0.63 (P1),
P(q²-spin) = 0.27 (P1). L10 alone: 1.5 % (P1), 0.5 % (P3).)

Class-averaged Bayes factors (uniform within class, log-uniform s): elastic-isoscalar 8.4 (54 % of its members have B = 0),
elastic-isovector 9.0 (46 % zero), inelastic O1 48.4, inelastic O4 49.1 (`P027_class_table.csv`). The class priors P2/P3
give B_DM ≈ 29 simply because inelastic spectra are 19 % of the distinguishable spectra but 50 % of the class prior.

**Comparison with the frequentist LEE.** The single-model maximum (companion-free, 171–176) divided by P008's
N_eff = 12.2 or LZ's implied 13.9 gives 14.4 / 12.7; P001 quoted B/N_eff = 7.3–36 for b = 10⁻³–2 × 10⁻⁴. Our
model-marginalised B_DM = 16.4 (uniform over spectra) corresponds to a Bayesian trials factor B_max/B_DM = 10.3 (12.3 over
raw models), i.e. the mixture Occam factor is numerically close to the toy-MC N_eff, as P001 anticipated. It is smaller
(5.9) for class priors that up-weight the inelastic sector, and the total lies at or above the Sellke–Bayarri–Berger
bound 14.7 from the 2.6σ global p-value for every prior except uniform-s (6.6–14).

**Posterior over models.** Top-10 (P1, log-uniform s): O1^v 1000 GeV δ = 350 (3.4 %), O1^v 4000 δ = 350 (3.4 %), O4^s 1000
δ = 350 (3.3 %), O1^v 400 δ = 300 (3.0 %), O4^s 4000 δ = 350 (2.8 %), O1^v 1000 δ = 300 (2.6 %), O1^v 4000 δ = 300 (2.5 %),
O1^s 1000 δ = 350 (2.3 %), O4^s 400 δ = 300 (1.9 %), O4^s 1000 δ = 300 (1.7 %); L10 1000 GeV is 14th (1.5 %), O15^s 4000 GeV
18th (1.2 %). No single model exceeds 3.4 % (5.5 % under P2, 7.0 % with ρ under P3); half the posterior needs 15–29
models, 90 % needs 64–90; the exponential of the entropy is 65–99 models. SI-like elastic O1 (either isospin, any mass)
carries < 10⁻⁴ of the posterior under every s-prior (P016's floor of 1.0 for O1 was the approximation); under the
μ-parametrisation the 123 light models have B = 0.709 each and B_DM barely moves (17.0 vs 16.4).

## 6. Continuous δ: the Bayesian tuning penalty (tables `P027_inelastic_class_delta.csv`, `P027_delta_scan.csv`)

For each (operator, isospin, mass) we integrate B(δ) over δ with (i) a uniform prior on [0, δ_ceil(m)] (data-independent
kinematic ceiling μv²_max/2 = 329 / 383 / 417 keV; 10 keV trapezoid) and (ii) a log-uniform prior on [10 keV, δ_ceil], and
compare with the mean over LZ's physical grid points and with max_δ B (log-uniform s):

| op | m [GeV] | B uniform-δ | B log-δ | B LZ-grid mean | grid δ ≥ 300 mean | B_max (δ) | B_max/B_unif | δ posterior 16/50/84 % [keV] |
|---|---|---|---|---|---|---|---|---|
| O1^s | 400 | 14.8 | 5.2 | 10.4 | 40.0 | 149 (330) | 10.1 | 233 / 300 / 323 |
| O1^s | 1000 | 28.3 | 9.6 | 22.8 | 72.4 | 171 (380) | 6.0 | 270 / 349 / 375 |
| O1^s | 4000 | 34.7 | 11.7 | 18.5 | 54.2 | 171 (400) | 4.9 | 287 / 371 / 401 |
| O1^v | 400 | 45.9 | 17.8 | 41.3 | 147 | 170 (330) | 3.7 | 203 / 278 / 316 |
| O1^v | 1000 | 63.9 | 25.4 | 57.9 | 148 | 171 (380) | 2.7 | 218 / 309 / 362 |
| O1^v | 4000 | 71.9 | 29.0 | 58.5 | 144 | 171 (400) | 2.4 | 225 / 324 / 387 |
| O4^s | 400 | 38.2 | 20.7 | 35.0 | 94.6 | 159 (320) | 4.2 | 145 / 256 / 311 |
| O4^s | 1000 | 52.8 | 28.3 | 52.6 | 122 | 171 (370) | 3.2 | 167 / 296 / 356 |
| O4^s | 4000 | 60.5 | 32.2 | 50.9 | 109 | 171 (380) | 2.8 | 178 / 317 / 381 |

With the shape factor ρ: O1^s B_unif = 7.3 / 23.0 / 30.3, B_max = 111 / 240 / 233 at δ = 330 / 380 / 400 keV, tuning
penalty 15.2 / 10.5 / 7.7, δ posterior 68 % interval 243–325 / 318–379 / 340–406 keV (median 310 / 364 / 386) and only
18 / 32 / 37 % of the δ-prior volume has B > 10; O1^v: B_unif = 58 / 87 / 99, penalty 4.0 / 3.1 / 2.7, 62–72 % of δ-volume
with B > 10; O4^s: B_unif = 33 / 55 / 65, penalty 5.2 / 4.9 / 4.1, 74–88 % with B > 10. Class average over the nine
(op, τ, m) combinations: LZ grid 38.7 → uniform-δ 45.7 → log-δ 20.0 (with ρ: 40.6 → 50.8 → 21.1); B_max 171 (271 with ρ).

Interpretation. (1) Replacing LZ's 8-point grid by a continuous uniform δ prior does *not* penalise the inelastic class
(ratio 0.85–0.80): the grid already spends most of its weight at δ ≤ 200 keV where B ≤ 10, and the continuum adds the
high-B region 350 keV–δ_ceil that the grid truncates (P002, P008). (2) The genuine tuning penalty is B_max/B_unif = 2.4–10
(2.7–15 with ρ): the best-tuned δ is favoured by one order of magnitude over the class. (3) For the isoscalar O1 spectrum
the form-factor node at 266 keV makes 248 keV atypical (ρ ≈ 0.4) unless δ ≳ 350 keV, so the δ-posterior piles up within
~20 keV of the ceiling (median 364 keV, 68 % 318–379 keV at 1000 GeV), the Bayesian counterpart of P002's "typical only
for δ ≥ 385 keV" (P002 used the June halo, +12 keV). For O1^v and O4 no node intervenes and the δ posterior is broad
(68 % widths 140–200 keV). (4) B_DM with the continuous-δ class replacing the grid: 27.8 (P2), 27.8 (P3) [29.2 / 29.2 with ρ]
versus 28.7 / 28.7 with the grid — unchanged.

## 7. Posterior predictive for the next 6.76 t·yr (P020 exposure, k = 6.76/2.84 = 2.38; table `P027_posterior_predictive.csv`)

p(s | D, m) ∝ π(s) L(s); N ~ Poisson(k (s + b_H)); no annual-modulation weighting (P020 treats it).

| model | prior | s mean (16/50/84 %) | N mean | P(0) | P(1) | P(≥2) | P(≥3) |
|---|---|---|---|---|---|---|---|
| O1^v 1000 δ = 350 | log-u | 0.99 (0.17/0.68/1.81) | 2.35 | 0.30 | 0.21 | 0.49 | 0.35 |
| O1^v 1000 δ = 350 | flat | 1.97 (0.70/1.66/3.25) | 4.70 | 0.09 | 0.13 | 0.79 | 0.66 |
| O4^s 1000 δ = 350 | log-u | 0.93 | 2.22 | 0.31 | 0.21 | 0.48 | 0.33 |
| O1^s 1000 δ = 350 | log-u | 0.65 | 1.54 | 0.39 | 0.24 | 0.37 | 0.22 |
| O4^s 1000 δ = 300 | log-u | 0.49 | 1.16 | 0.46 | 0.25 | 0.29 | 0.16 |
| L10 1000 GeV | log-u | 0.42 (0.07/0.29/0.76) | 0.99 | 0.50 | 0.25 | 0.25 | 0.12 |
| L10 1000 GeV | flat | 0.83 (0.30/0.70/1.37) | 1.98 | 0.25 | 0.25 | 0.50 | 0.31 |
| O6^s 1000 GeV | log-u | 0.31 | 0.75 | 0.57 | 0.25 | 0.18 | 0.08 |
| model-averaged, P1 | log-u | 0.39 | 0.92 | 0.58 | 0.22 | 0.20 | 0.10 |
| model-averaged, P3 | log-u | 0.48 | 1.13 | 0.53 | 0.22 | 0.25 | 0.14 |
| background only (b_H bin) | — | 0 | 0.0014 | 0.999 | 0.001 | — | — |

The log-uniform prior pulls s toward small values (one event, no lower bound from the data), so the predictions are
below P020's plug-in ŝ = 1.0 (2.4 events, P(0) = 9 %); the flat prior reproduces P020 for the companion-free models (L10
flat: 2.0 events, P(0) = 25 %, because L10's companions shrink ŝ to 0.5 in P016). Under the model-averaged predictive,
zero events in 6.76 t·yr has probability 0.53–0.58 and is therefore *not* decisive against the DM mixture (it would
multiply the odds by ~0.55); ≥ 2 events (P = 0.20–0.25 under DM, 10⁻⁶ under the b_H background) would be.

## 8. Robustness, caveats, failed approaches

- **Coupling prior.** B_DM changes by ×2.5 between uniform-s [0, 10] (6.6) and log-uniform (16.4) under P1; Jeffreys is
  intermediate (13.8). The uniform prior is harsh on models with companions (it puts 90 % of its mass at s > 1, where
  even N_lo ≈ 1 costs a factor ~e^{−s}). Ranking of models is prior-independent.
- **s vs μ parametrisation.** Identical for heavy models (f_hi ≈ 0.1–1); differs only for light and SI-like models
  (B = 0 vs 0.7), moving B_DM by +0.6 (P1).
- **Shape factor.** ρ changes B_DM by −3 % (P1) to +1 % (P3) and reshuffles the top-10 (O1^v up, O1^s down); the σ√2
  variant (ρ_32) gives B_DM = 15.8 (P1). ρ assumes a flat background in the window (15 % accuracy) and one-dimensional
  energy information; LZ's 2D {S1c, log S2c} likelihood carries the same information implicitly.
- **b_H.** All B scale ≈ 1/b_H (P001); the anchor 5.7 × 10⁻⁴ is LZ-equivalent (P016). With P001's optimistic 2 × 10⁻⁴,
  everything is ×2.8 higher; with the whole-panel 0.0106, ×0.054.
- **Model space.** Our operator basis (298 distinguishable) is not LZ's Lagrangian list (293); 12 of LZ's 40 Lagrangians
  map onto our operators, and the L-list contains more heavy-mass q²-suppressed shapes (L6, L9, L11, L12, L16, L18–L20)
  whose B would be 30–70. A prior uniform over LZ's list would therefore shift mass from inelastic to elastic-spin
  operators (B_DM would rise a little, P(inel) fall to ~0.4).
- **Halo.** Annual-average Baxter-2021 halo (LZ convention); the June halo raises δ ceilings by 12–16 keV and would move
  the δ posteriors up by a similar amount (P002, P006).
- **Low-energy data.** 1D projection of Fig. 5 (P016); θ marginalised rather than profiled — the difference is < 2 % in B.
- **Failed/abandoned.** (i) A first run computed all 388 new WimPyDD spectra in a single 833 s pass, exceeding the
  600 s foreground limit; the script was made resumable (partial cache, `P027_BUDGET_S` environment variable) and
  re-run (158 s with the cache). (ii) Computing O4^v on the fine δ grid was dropped (O4 spectra cost ~3 s each; LZ
  states O4 s/v shapes are indistinguishable; the O4^v LZ-grid points confirm B within 1 % of O4^s).

## 9. Result files

`P027_models.csv` (752 rows: spectra classes, N_lo/N_L2/N_M, f_hi, ρ, all Bayes factors, posterior weights);
`P027_BDM_vs_prior.csv`; `P027_class_table.csv`; `P027_inelastic_class_delta.csv`; `P027_delta_scan.csv`;
`P027_elastic_1000GeV.csv`; `P027_posterior_predictive.csv`; `P027_validation_Nlo.csv`; `P027_validation_B_vs_P016.csv`;
`P027_results.json`; `P027_spectra.npz` (WimPyDD spectra); `run_log.txt`.

Figures (`figures/`): **Fig. 1** `P027_fig1_posterior_models.png` — posterior probability of the top-20 models under the
three main model priors (log-uniform s), coloured by class; **Fig. 2** `P027_fig2_BDM_vs_prior.png` — B_DM for five model
priors × five coupling priors/parametrisations, with P001's B/N_eff = 12.7 and the SBB cap 14.7; **Fig. 3**
`P027_fig3_inelastic_delta.png` — B(δ) for O1^s and O4^s at 400/1000/4000 GeV with and without ρ, LZ grid points and
kinematic ceilings; **Fig. 4** `P027_fig4_predictive.png` — posterior predictive P(N) for 6.76 t·yr for four models, the
model average and the background.

## 10. References

LZ Collaboration, arXiv:2609.02823 (2026). R. E. Kass, A. E. Raftery, J. Am. Stat. Assoc. 90, 773 (1995).
T. Sellke, M. J. Bayarri, J. O. Berger, Am. Stat. 55, 62 (2001). R. Trotta, Contemp. Phys. 49, 71 (2008).
D. Baxter et al., Eur. Phys. J. C 81, 907 (2021). I. Jeong, S. Kang, S. Scopel, G. Tomar, Comput. Phys. Commun. 276,
108342 (2022). N. Anand, A. L. Fitzpatrick, W. C. Haxton, Phys. Rev. C 89, 065501 (2014). Corpus: P001, P002, P003,
P008, P016, P017, P020.

## 11. Tools and provenance (mirrors `output/provenance/P027.json`)

Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (stats.norm/poisson, special.erf/logsumexp); pandas 3.0.5;
matplotlib 3.11.2 (Agg); WimPyDD 2.0.4 (eft_hamiltonian incl. q-dependent coefficient, diff_rate via lz.wd_rate,
streamed_halo_function via lz.wd_halo with the explicit v_min grid); common/lzcommon.py (LSIG, OSIG, LSIG_MASSES,
OSIG_DELTAS, M_NUCLEON_GEV, M_V_GEV, mu_red, m_nucleus_gev, vmax_kms, delta_max_kev, A_XE_MEAN, C_KMS, wd_*).
Local inputs: P016_results.json (digitised Fig. 5 bins, b_H), P016 operator table, P008 spectra cache and models.csv,
P003 operator table (via P016), LZ fulltext.tex LEE section (lines 485–498) and Tables S6–S7 (lines 820–909), papers
P001/P002/P003/P008/P016, dossier, ledger, PAPER_GUIDE, ENVIRONMENT_versions.txt.
Recalled knowledge (5): Kass–Raftery evidence scale (certain); Sellke–Bayarri–Berger bound (certain); Jeffreys prior
∝ s^{−1/2} for a Poisson mean (certain); Lagrangian ↔ operator map L1 = O1, L2 = O10, L3 = O11, L4 = O6, L15 = O4,
L10 ∝ (q²/m_N²)O4 − O6 (likely, via P003/P016); entropy-based effective number of models exp(H) (certain).
WimPyDD-generated files: none outside `output/` (WD.diff_rate does not write response-function files).
Datasets: none. Data requests: none.
