# P098 — Energy resolution at 250 keV from quanta statistics: does ±23 keV (9 %) follow from N_q fluctuations, recombination and the NR-band width?

Simulated arXiv date 2026-09-16 · physics.ins-det · RESP · LXe detector-response group.
Script: `output/code/P098_energy_resolution.py` (runs in ≈ 3 s from the simulation root; all numbers below are printed to
`output/work/P098/run_stdout.txt` and saved in `results.json`, `variance_budget.csv`, `sigma_E_vs_E.csv`, `er_budget_248keV.csv`).

## 1. Motivation and framework

The LZ paper states three times (abstract, l. 15; Results, l. 165; Summary, l. 332 of `inputs/LZ_arXiv_2609.02823_fulltext.tex`) that the
event is "consistent with a nuclear recoil of 248 ± 23 (stat) ± 23 (sys) keV". The ±23 keV therefore *is* the paper's own number,
not a corpus construction; the dossier (`output/00_evidence_dossier.md`, §1 and table row "NR energy") transcribes it verbatim and adds no
derivation. The paper gives no decomposition of either uncertainty, no resolution function, and no statement of the estimator used.
Corpus usage has diverged: P021, P038 and P057 smear spectra with σ_E = 11 √(E/248) keV (from P009's 2D likelihood width), P052 used a
23 keV (9.3 %) resolution as a bracketing variant and found it *improved* the reproduction of LZ's significance tables (rms 0.52 → 0.42σ),
and P043 showed that the quoted gains explain only 3.8 keV of the systematic, the rest requiring a ≈ 10 % light-yield uncertainty (P064:
δβ ≈ 0.085 in the yield slope).

This note builds the statistical part from first principles. Ingredients:

* mean yields: LZ Table S5 with the p(E) break, evaluated by nestpy 2.1.1 through `lz.nest_nr_yields` (`lz.NEST_NR_LZ`); and P009's
  "paper-contour" scale N_q = 11.32 E^1.112 (W = 13.46 eV implied by the keVee contour labels), with N_e from Table S5 and
  N_ph = N_q − N_e, which reproduces the paper's S1c(E) relation (P009, P024, P043, P064 all adopt it as baseline);
* quanta fluctuations: nestpy `NESTcalc.GetQuanta(yields, ρ = 2.9, W)` with the LZ_WS2024 width vector
  W = [0.404, 0.393, 0.0383, 0.497, 0.1906, 2.22, 0.3, 0.04311, 0.15505, 0.46894, −0.26564, 0, 0] (P024 verified its variance form to 0.3 %);
* detector layers (applied cumulatively, each toggled separately): (1) photon detection binomial with g₁ = 0.110 phd/photon [paper];
  (2) double-photoelectron emission P_dphe = 0.214 (nestpy LZ_WS2024 object; P009/P024 absorbed it into g₁ and did not simulate it);
  (3) single-phe resolution sPEres = 0.338 (nestpy object) per detected photon; (4) extraction binomial ε_ext = 0.80 [recalled, uncertain;
  0.726 from nestpy `CalculateG2` as variant] with SE size g₂/ε_ext so that ⟨S2c⟩ = g₂ N_e, g₂ = 34.5 phd/e [paper]; (5) SE-size Fano F = 4
  (nestpy object), Var(S2) = F·S2; (6) position-correction residual 2 % on S1c and S2c independently [assumption, P009/P024 baseline;
  0/3/4 % variants; P043 quotes 2–3 %]; (7) electron-lifetime correction uncertainty on S2c: drift 870 μs (dossier, Fig. 3), τ_e ≈ 6 ms
  [recalled, uncertain], 10 % on τ_e → 0.87/6 × 0.10 = 1.5 % on S2c (0 and 5 % variants).

Energy estimators for an NR at S1c, S2c:

* combined quanta ("N_q estimator"): N̂_q = S1c/g₁ + S2c/g₂, E_Nq = N_q⁻¹(N̂_q) on the chosen yield scale;
* S1-only: E_S1 = N_ph⁻¹(S1c/g₁); S2-only: E_S2 = N_e⁻¹(S2c/g₂);
* 2D Gaussian maximum likelihood in (S1c, log₁₀S2c): σ_E⁻² = m′ᵀ C⁻¹ m′ (Fisher information), with the mean vector derivative m′ from MC at
  244 and 252 keV and the covariance C from MC at 248 keV with the full detector; the optimal linear weights are w = C⁻¹m′/(m′ᵀC⁻¹m′).

Linearised, the N_q estimator has σ_E = σ(N̂_q)/(dN_q/dE) with dN_q/dE = β_eff N_q/E = 23.30 quanta/keV on the contour scale
(21.41 on Table S5). The analytic marginal variances (in quanta) are

| term | Var(N̂_q) | value (contour scale) |
|---|---|---|
| photon binomial | N_ph(1−g₁)/g₁ | 199.1² |
| double phe (extra over binomial) | [n_det P(1−P) + (1+P)² n_det(1−g₁/(1+P))]/g₁² − N_ph(1−g₁)/g₁, n_det = N_ph g₁/(1+P) | 125.3² |
| SPE resolution | (sPEres)² n_det/g₁² | 64.7² |
| extraction binomial | N_e(1−ε)/ε | 8.7² |
| SE Fano | F g₂ N_e/g₂² | 5.9² |
| position residual | σ_pos²(N_ph² + N_e²) | 98.1² |
| e-lifetime | (0.015 N_e)² | 4.5² |

## 2. Inputs

| input | value | source |
|---|---|---|
| g₁, g₂ | 0.110 ± 0.002 phd/photon, 34.5 ± 1.1 phd/e | paper l. 88 via `lz.LZ` |
| event | S1c = 540.1 phd, S2c = 9268 phd | paper l. 165 via `lz.LZ` |
| quoted energy | 248 ± 23 (stat) ± 23 (sys) keV | paper l. 15, 165, 332 |
| NR yields | Table S5 (`lz.NEST_NR_LZ`, incl. a = 0.0230, b = 0.0289, E₀ = 74.7) at 96.5 V/cm, ρ = 2.9 | paper Table S5; `lz.nest_nr_yields` |
| contour scale | N_q = 11.3245 E^1.11167, W = 13.4628 eV | P009 `results.json` |
| ER yields | Table S3 (`lz.NEST_ER_LZ`) via `lz.nest_er_yields` | paper Table S3 |
| width vector, P_dphe, sPEres, s2Fano | LZ_WS2024 object: see §1; 0.214; 0.338; 4.0 | nestpy 2.1.1 |
| W (NEST) | 13.44 eV (`NESTcalc.WorkFunction(2.9).Wq_eV`); 13.7 eV `lz.W_EV` (recalled) | nestpy; lzcommon |
| ε_ext, σ_pos, e-lifetime | 0.80; 0.02; 1.5 % on S2c | recalled/assumed (§1) |
| kinematics | v_max(16 June) = 809.1 km/s from `lz.vmax_kms(lz.v_earth_kms(day_of_year=167))`; `lz.delta_max_kev` | lzcommon |
| P021 peak slope | 0.31 keV of δ_peak per keV of E_obs | P021 |
| P043 systematic | 9.6 keV (g₁ 3.7, g₂ 0.8, N_q-scale 8.7) | P043 |
| recalled ER resolution | XENON1T σ/E = 0.317/√E(keV) + 0.0015 (Aprile et al. 2020) — likely; "≈ 4 % at 164/236 keV" in the assignment — uncertain | memory |

## 3. Results

### 3.1 Means, combined energy and inversion (script §1)

Event: N̂_ph = 540.1/0.110 = 4910.0, N̂_e = 9268/34.5 = 268.6, N̂_q = 5178.6. E_ee = W N̂_q = 70.95 keVee (W = 13.7), 69.60 (13.44),
69.72 (13.46; P009: 69.7).

| scale | N_ph(248) | N_e(248) | N_q(248) | S1c(248) | log₁₀S2c(248) | dN_q/dE | β_eff(N_q) | β_eff(S1) | β_eff(S2) | E_S1 | E_Nq | E_S2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Table S5 as printed | 4525.0 | 301.1 | 4826.1 | 497.8 | 4.0165 | 21.41 | 1.100 | 1.151 | 0.329 | 266.2 | 264.4 | 176.7 |
| paper contour (P009) | 4897.2 | 301.1 | 5198.3 | 538.7 | 4.0165 | 23.30 | 1.112 | 1.160 | 0.329 | 248.6 | 247.2 | 176.7 |

E_S1 and E_Nq reproduce P009 (266.2/264.4 and 248.5/247.2). The S2-only inversion of N_e = 268.6 gives 176.7 keV on either scale
(N_e is scale-independent by construction), because the charge yield with the p(E) break has d ln N_e/d ln E = 0.33 only.

### 3.2 Quanta fluctuations at fixed E = 248 keV (script §2; 120 000 `GetQuanta` samples, contour scale)

sd(N_ph) = 44.7, sd(N_e) = 17.6 (P024: 17.56), corr(N_ph, N_e) = −0.243, Cov = −190.9, sd(N_q) = **43.9 quanta**. Check:
Var(N_ph) + Var(N_e) + 2Cov = 1997.4 + 310.0 − 381.9 = 1925.5 = Var(N_q) (MC 1925.5). The anticorrelation removes 62 % of Var(N_e) from
N_q: recombination moves quanta between the channels and cancels exactly; only the ion/exciton Fano terms (W₀ = 0.404, W₁ = 0.393) and
the extraction-independent part survive. Intrinsic quanta fluctuations thus contribute σ_E = 43.9/23.30 = **1.88 keV (0.76 %)**.

### 3.3 Variance budget of the reconstructed energy (script §2; `variance_budget.csv`; Fig. 1 left)

N_q estimator, contour scale, layers added cumulatively; the marginal MC sd is √(Var_k − Var_{k−1}); with 120 000 samples the variance
is known to ≈ 0.4 %, so marginal terms below ≈ 16 quanta (0.7 keV) are not resolved by the MC and the analytic value is quoted.

| layer | σ(N̂_q) cum. [quanta] | marginal σ_E MC [keV] | analytic [keV] | cumulative σ_E [keV] | cum. % | 68 % half-width [keV] |
|---|---|---|---|---|---|---|
| quanta (NEST) | 43.9 | 1.88 | — | 1.88 | 0.76 | 1.87 |
| + photon binomial (g₁ = 0.110) | 203.4 | 8.52 | 8.54 | 8.73 | 3.52 | 8.67 |
| + double phe (P = 0.214) | 239.2 | 5.40 | 5.38 | 10.27 | 4.14 | 10.20 |
| + SPE resolution (0.338) | 247.7 | 2.77 | 2.78 | 10.63 | 4.29 | 10.54 |
| + extraction binomial (0.80) | 247.6 | (0.0) | 0.37 | 10.63 | 4.28 | 10.55 |
| + SE Fano (F = 4) | 248.8 | (1.06) | 0.25 | 10.68 | 4.31 | 10.60 |
| + position residual 2 % | 267.7 | 4.23 | 4.21 | 11.49 | 4.63 | 11.45 |
| + e-lifetime 1.5 % on S2c | 266.7 | (0.0) | 0.19 | **11.45** | **4.62** | 11.35 |

Table S5 scale (S1c = 498 phd at 248 keV): the same chain gives 2.05 → 9.20 → 10.76 → 11.12 → 11.13 → 11.16 → 11.95 → **11.94 keV**
(the smaller N_ph lowers σ(N̂_q) but dN_q/dE is smaller too). The S1-only estimator tracks the N_q estimator to within 0.2 keV at every
layer (11.64 keV full; S1c sd 29.3 phd = 5.4 %), because S2 carries only N_e/N_q = 5.8 % of the quanta.

Ranking: photon-detection binomial 8.5 keV (56 % of the variance), double-phe 5.4 (22 %), position residual 4.2 (14 %), SPE 2.8 (6 %),
NEST quanta 1.9 (3 %); every S2-side term (extraction, SE Fano, e-lifetime, and the recombination fluctuation, which cancels) is < 0.4 keV.
The energy resolution of a 248 keV NR in LZ is a *light-collection* number: σ_E/E ≈ 0.90 × √[(1−g₁)/(g₁N_ph)] × √(1 + dphe + …).

Variants (N_q estimator, full detector): σ_pos = 0/3/4 % → 10.68/12.41/13.58 keV; ε_ext = 0.726 → 11.46; no double-phe (P009/P024
convention) → 10.17; e-lifetime 0/5 % → 11.43/11.45. Our range for the statistical resolution is therefore **10.2–12.4 keV (4.1–5.0 %)**,
central 11.4 keV; P009's 10–11 keV and P024's 10.65 keV (slice sd_E) sit at the low edge because they omit double-phe.

**What ±23 keV would need.** σ_E = 23 keV requires σ(N̂_q) = 536 quanta, i.e. an S1c spread of 10.9 % at 540 phd (we have 5.4 %) or a
variance 4.0× the full model. No detector or NEST width parameter in our list can supply it: the position residual would have to be 9 %,
or g₁ would have to be 0.03. We note that 2 × 11.45 = 22.9 keV: the quoted "stat" coincides with a 95.4 % (2σ) half-width of our
resolution, which is the most economical reading if the paper's convention is not 1σ (hypothesis, not established).

### 3.4 The 2D estimator and the band-width ↔ resolution relation (script §2–3; Fig. 2)

Full-detector MC at 248 keV (contour): ⟨S1c⟩ = 538.7, sd 29.5 phd; ⟨log₁₀S2c⟩ = 4.0152 (P024: 4.0152), sd **0.0312 dex** (P024: 0.0313);
corr(S1c, log₁₀S2c) = −0.038. Derivatives: dS1c/dE = 2.564 phd/keV, dlog₁₀S2c/dE = 5.55 × 10⁻⁴ dex/keV. Fisher resolution:
σ_E(2D) = **11.19 keV**; S1-only Gaussian 11.51; S2-only 56.3 keV. Table S5 scale: 11.73 / 12.15 / 50.2 keV.
Optimal weights: 0.372 keV/phd on S1c and 84.7 keV/dex on log₁₀S2c (the N_q estimator implicitly uses 29.8 keV/dex). Applied to the
event: E_2D = 248 + 0.372(540.1 − 538.7) + 84.7(3.967 − 4.0152) = **244.4 keV** (P009's full likelihood: 245.8); the charge deficit pulls
the 2D estimate down by 4.1 keV, the N_q estimate by 1.35 keV.

Relation. At fixed energy the covariance is almost diagonal (corr −0.04; regression slope −3.7 × 10⁻⁵ dex/phd), so the band variable
log₁₀S2c and the energy variable S1c are statistically independent axes: σ(log₁₀S2c | S1c) = 0.03124 = σ(log₁₀S2c) = 0.03125. The band
width is set by the electron channel — 0.0312 dex × ln10 × 301 e = **21.7 electrons per band σ** — of which P024 attributes 54 % of the
variance to recombination binomial fluctuations that *cancel* in N_q (our Cov = −191 quanta² removes 62 % of Var(N_e)). Converted to
energy through the N_q estimator, one band σ is 21.7/23.30 = **0.93 keV**, i.e. 8 % of the energy resolution; through the 2D weight it is
84.7 × 0.0312 = 2.6 keV. Conversely the S1 axis, which carries the energy, contributes nothing to the band at fixed S1c. Band width and
energy resolution are therefore consistent but nearly decoupled: a 1.5σ band offset is a 32-electron, 1.4 keV (N_q) or 4 keV (2D)
effect, and no plausible band-width error can turn 11 keV into 23 keV (the whole band is 0.9 keV).

Flat-spectrum consistency: MC at fixed E gives sd(log₁₀S2c) = 0.0312; P024's slice |S1c − 540| < 10 for a flat spectrum gives 0.0313
(the mixing term 0.0061 in quadrature is what the slice adds; 0.0306 at fixed E in P024). The event is 1.545σ below the median in our
MC (P024 1.54; paper 1.5).

### 3.5 Energy dependence (script §4; `sigma_E_vs_E.csv`; Fig. 1 right)

| E [keV] | σ_E(N_q) [keV] | % | σ_E(S1) [keV] | 11√(E/248) | sd(log₁₀S2c) | sd(S1c)/S1c |
|---|---|---|---|---|---|---|
| 100 | 6.99 | 6.99 | 7.25 | 6.99 | 0.035 | 0.088 |
| 150 | 8.83 | 5.89 | 9.06 | 8.55 | 0.033 | 0.071 |
| 200 | 10.19 | 5.10 | 10.39 | 9.88 | 0.032 | 0.061 |
| 248 | 11.51 | 4.64 | 11.70 | 11.00 | 0.031 | 0.055 |
| 300 | 12.75 | 4.25 | 12.93 | 12.10 | 0.031 | 0.050 |
| 330 | 13.44 | 4.07 | 13.61 | 12.69 | 0.030 | 0.047 |

Power-law fit: σ_E = **11.5 (E/248)^0.544 keV** (S1-only: 11.7 (E/248)^0.525). The exponent follows from Var ∝ N_ph ∝ E^1.16 and
dN_q/dE ∝ E^0.11: γ = 0.58 − 0.11 = 0.47 for the counting terms, pulled up to 0.54 by the position residual (∝ E). The corpus law
11√(E/248) keV (P021, P038, P057) is within 5 % of the MC everywhere in 100–330 keV.

### 3.6 ER at 248 keV (script §6; `er_budget_248keV.csv`)

Table S3 β yields: N_ph = 10 996, N_e = 7456, N_q = 18 452 (E = W N_q = 248.0 keV with W = 13.44 eV). Quanta: sd(N_ph) = 660.5,
sd(N_e) = 659.5, corr = **−0.994**, sd(N_q) = 74.2 (Fano-like, 0.40 %). Cumulative σ_E: quanta 1.00 → photon binomial 4.13 → double phe
4.82 → SPE 5.02 → extraction 5.05 → SE Fano 5.04 → position 6.22 → e-lifetime **6.40 keV = 2.58 %**. The e-lifetime term matters here
(1.5 keV) because S2 carries 40 % of the quanta. Recalled measured combined-energy resolutions: XENON1T formula 2.16 % at 248 keV,
2.63 % at 164 keV, 2.21 % at 236 keV [likely]; the "≈ 4 % at the 164/236 keV lines" quoted in the assignment [uncertain] would imply
larger position/gain residuals (σ_pos ≈ 4–5 %) than our 2 %. Our 2.6 % sits between; the NR/ER resolution ratio at the same energy is
4.6/2.6 = 1.8 because the NR has 3.6× fewer photons and gains nothing from S2. (nestpy 2.1.1 uses the v2.4.0 single-skew ER fluctuation;
Table S4's double-skew form changes recombination fluctuations only, which cancel in N_q at the 99 % level shown by the correlation.)

### 3.7 S1-only, S2-only and the charge-poor tension in keV (script §7)

| estimator | event (contour) | event (Table S5) | MC at 248 keV: median, 16–84 % | σ |
|---|---|---|---|---|
| S1-only | 248.6 | 266.2 | 247.9, 236.4–259.7 | 11.6 keV |
| N_q | 247.2 | 264.4 | 247.9, 236.6–259.4 | 11.5 keV |
| S2-only | 176.7 | 176.7 | 243.7, 198.6–302.2 (1.9 % unresolvable) | 51.8 keV (linear 54.5) |
| 2D Gaussian | 244.4 | 260.7 | — | 11.2 keV |

E_S1 − E_S2 = 71.9 keV against √(11.5² + 54.5²) = 55.7 keV: **1.29σ**, the same information as the 1.54σ band offset expressed on an
energy axis where the S2 estimator is almost uninformative (β_eff(S2) = 0.33). In electrons: N_e(observed) = 268.6 vs the 248 keV model
300.2 → deficit 31.6 e = 1.35 keV in N_q, 4.1 keV in the 2D estimator. "If the event were an NR, S1 and S2 energies agree within
resolution": yes, at 1.3σ; and the agreement is so weak a test that it carries ≈ 1 keV of energy information.

### 3.8 Propagation to δ_max and the inelastic likelihood (script §8)

With v_max(16 June) = 809.1 km/s: δ_max(248 keV) = 341.1 / 386.6 / 409.4 keV for m = 400 / 1000 / 4000 GeV (P043: identical),
dδ_max/dE = 0.035 / 0.218 / 0.310. Resulting σ(δ_max):

| σ_E [keV] | 400 GeV | 1000 GeV | 4000 GeV | P021 peak (0.31 keV/keV) |
|---|---|---|---|---|
| 11.4 (this work, stat) | 0.40 | 2.50 | 3.55 | 3.5 |
| 11.2 (2D) | 0.39 | 2.44 | 3.47 | 3.5 |
| 14.9 (11.4 ⊕ P043's 9.6 scale) | 0.52 | 3.26 | 4.63 | 4.6 |
| 23 (paper "stat") | 0.80 | 5.02 | 7.13 | 7.1 |

For a spectrum with a sharp kinematic endpoint E_max below the event (flat below it), the smeared density at 248 keV is Φ((E_max − 248)/σ):
the ratio σ = 23 vs 11 is 1.0 / 1.56 / 4.3 / 20 for E_max = 248 / 240 / 230 / 220 keV, so a 23 keV smearing lets δ values whose window has
already slid 20–30 keV past the event keep a substantial likelihood; with 11 keV they are killed. This is why P021's collapse beyond
δ ≈ 385 keV is sharp and why P052's 23 keV variant redistributed significance among near-edge models.

## 4. Recommendation to the corpus

σ_E(248 keV, NR) = **11.4 keV (4.6 %) statistical**, range 10.2–12.4 keV over detector assumptions, S1-dominated; energy dependence
σ_E = 11.5 (E/248)^0.54 keV, for which 11√(E/248) is adequate. The paper's ±23 (sys) keV is a *scale* uncertainty (P043/P064: light-yield
slope, common to the whole spectrum) and must enter fits as a nuisance on the energy scale, not as a smearing; the ±23 (stat) cannot be
obtained from any combination of NEST quanta fluctuations, counting statistics, band width or position corrections and should not be
used as a resolution (P052's variant is to be read as a scale-plus-resolution bracket). A Gaussian with σ = 11.4 keV and a ±4 % scale
nuisance (P043) reproduces the S1c roll-off (P009: 11.8 keV), the band width and the 1.5σ statement simultaneously.

## 5. Validation and robustness

* Var(N_q) closes: 1997.4 + 310.0 − 2 × 190.9 = 1925.5 vs MC 1925.5.
* Analytic marginals agree with the MC to ≤ 0.03 keV for every term > 1 keV (8.52/8.54, 5.40/5.38, 2.77/2.78, 4.23/4.21).
* Band: sd(log₁₀S2c) = 0.0312 vs P024 0.0313; median 4.0152 vs P024 4.0152 (Fig. 4 digitised median 4.0162); event 1.545σ vs 1.54 (P024), 1.5 (paper).
* Energies: E_S1/E_Nq = 248.6/247.2 (contour), 266.2/264.4 (Table S5) vs P009 248.5/247.2 and 266.2/264.4; E_2D 244.4 vs P009's 245.8.
* δ_max(248) = 341.1/386.6/409.4 keV vs P043's 341.1/386.6/409.4.
* Energy dependence: the 100 keV point is a test of the inversion grid (a first run with a grid starting at 100 keV truncated the low tail
  and gave 4.3 keV; the grid was widened to 20–500 keV, giving 6.99 keV, and the fit exponent moved from a spurious 0.88 to 0.54).
* Seeds fixed (numpy 98, nestpy 98); the run is reproducible to the printed digits.

## 6. Failed or abandoned

* nestpy accessor names guessed from P009's description (`get_nr_er_width_parameters`, top-level `WorkFunction`) do not exist in 2.1.1;
  the width vector is the attribute `nr_er_width_parameters` and the work function is `NESTcalc.WorkFunction(ρ).Wq_eV`.
* First energy-dependence fit invalid (grid truncation, see §5); corrected.
* We did not attempt to reproduce LZ's double-skew (Table S4) ER fluctuation model: it does not affect the ER combined-energy resolution
  (recombination cancels) and P056 covers it.

## 7. Figures

* `figures/P098_budget_and_energy_dependence.png` — left: marginal σ_E of the N_q estimator per detector layer at 248 keV (contour scale)
  with cumulative values and LZ's ±23 keV; right: σ_E(E) for the N_q and S1-only estimators, the fitted power law and the corpus 11√(E/248) law.
* `figures/P098_fixed_energy_scatter.png` — 6000 full-detector MC events at exactly 248 keV in (S1c, log₁₀S2c) with the event and the NR
  mean loci on both yield scales: the cloud is elongated along S1c (energy axis, 29.5 phd) and narrow in log₁₀S2c (band axis, 0.031 dex),
  with no tilt (corr −0.04).

## 8. References

1. LZ Collaboration, arXiv:2609.02823 (2026) — event, g₁, g₂, Tables S3–S5, quoted energy.
2. M. Szydagis et al., "A Review of NEST Models for Liquid Xenon…", Front. Detect. Sci. Technol. 2, 1480975 (2024), arXiv:2211.10726 — NEST v2 yield and fluctuation model; nestpy.
3. B. Lenardo et al., IEEE Trans. Nucl. Sci. 62, 3387 (2015) — NR yield model behind Table S5.
4. D. S. Akerib et al. (LUX), Phys. Rev. D 95, 012008 (2017) — combined-energy estimator, W, recombination anticorrelation.
5. E. Aprile et al. (XENON1T), Eur. Phys. J. C 80, 785 (2020) — combined-energy resolution parametrisation (recalled).
6. J. Aalbers et al. (LZ), Phys. Rev. Lett. 131, 041002 (2023) — extraction efficiency ≈ 80 % (recalled).
7. Corpus: P009, P021, P024, P038, P043, P052, P056, P057, P064; dossier 00.

## 9. Tools and provenance (mirrors `output/provenance/P098.json`)

* Agent tools: Read ×15 (PAPER_GUIDE; P009, P024, P043, P064, P056, P052, P057 papers; fulltext.tex l. 84–123 and 586–690; lzcommon.py
  l. 1–75 and 255–290; P009 script l. 150–197; two P098 figures), Bash ×16 (dossier and tex greps; lzcommon, ledger, P021/P038 greps and
  environment versions; P009/P024 result and details greps; P043 budget; three nestpy probes (one failed on a missing `timeout`, one on
  the width accessor); four script runs (first failed: work directory missing) with output extraction; four word-count/JSON checks),
  Write ×4 (script, details.md, P098.json, P098.md), Edit ×16 (script ×6, paper ×9 trimming passes, JSON ×1).
* Software: python 3.12.13; nestpy 2.1.1 (detectors.LZ_WS2024, NESTcalc.GetYields/GetQuanta/WorkFunction, RandomGen.set_seed,
  interactions.NR/beta, default_nr_parameters); numpy 2.5.3 (default_rng binomial/normal, cov, polyfit, interp, gradient); scipy 1.18.1
  (stats.norm.cdf); pandas 3.0.5; matplotlib 3.11.2 (Agg); common/lzcommon.py (LZ, NEST_NR_LZ, NEST_ER_LZ, DRIFT_FIELD_VCM, W_EV,
  nest_nr_yields, nest_nr_params_vector, nest_er_yields, combined_energy_keV, delta_max_kev, vmax_kms, v_earth_kms).
* Recalled knowledge (6): ε_ext ≈ 0.80 (LZ SR1 80.5 %; uncertain); electron lifetime ≈ 6 ms in LZ SR3 (uncertain); XENON1T resolution
  σ/E = 0.317/√E + 0.0015 (likely); "≈ 4 % at 164/236 keV lines" (uncertain, from the assignment); W = 13.7 eV (lzcommon, likely);
  Fisher-information/optimal-linear-estimator formulas (certain).
* Datasets: none. Data requests: none. WimPyDD files: none.
