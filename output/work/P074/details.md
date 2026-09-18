# P074 — Long-range operators (q⁻², q⁻⁴) versus the LZ event: a systematic map of which mediator masses and operator structures can leave a lone 248 keV recoil

Research record (simulated date 2026-09-15). Script: `output/code/P074_long_range.py` (stages `contact`, `validate`, `dipole`, `analysis`); outputs in `output/work/P074/` (`P074_Nlo_table.csv`, `P074_Nlo_wide_n1.csv`, `P074_Nlo_wide_n2.csv`, `P074_mmed_min.csv`, `P074_survivors.csv`, `P074_operator_collapse.csv`, `P074_validation.csv`, `P074_dipole_massive_mediator.csv`, `P074_uv_summary.csv`, `P074_results.json`, `P074_contact_spectra.npz`, `P074_dipole_spectra.npz`, `P074_run.log`, `figures/`). All rates: WimPyDD 2.0.4 through `lzcommon.wd_rate` (natural xenon, shell-model responses), **Sun-frame** Baxter-2021 SHM from `lz.wd_halo()` (WimPyDD single-day halo, v_E = 250.6 km/s, no Earth orbital motion; v_max = 844 km/s on the explicit v_min grid), P003 efficiency model (plateau 0.96, 50 % at 5.4 and 269.9 keV, erf widths 3.4 / 8 keV), m_χ = 1000 GeV, exposure 2.84 t·yr.

## 1. Motivation and framework

P003 showed that a lone 248 keV recoil selects q²-suppressed spin operators (O6, O9, O10, O14, O15, O5^v: N_lo ≤ 5 low-energy events in 5.4–55 keV per 200–270 keV event) and excludes O1, O4, O7, O8, O11, O12 by one to three orders of magnitude. P031 matched these to relativistic contact operators and found the compatible ones need Λ = 1.5–62 GeV ≪ m_χ — i.e. a light mediator is implicit. P012 and P044 showed that the photon (m_med = 0) versions of the magnetic dipole, anapole and EDM fail (N_lo = 376, 27, 18 100), and P051 showed that a mediator lighter than q ≈ 0.25 GeV tilts inelastic O1 spectra toward low q. The obvious systematic question — which (operator, mediator mass) combinations retain the lone-event compatibility, and where exactly the contact regime ends — has not been answered. We answer it here for every NR operator of a spin-1/2 WIMP, for one and two propagators, and use the same machinery to (a) revisit "light-mediator SI dark matter" (O1 with a light scalar/vector), (b) decompose the photon-mediated magnetic dipole, and (c) give a UV-model summary.

## 2. Equations and conventions

**Propagator dressing.** For t-channel exchange of a mediator of mass m_med the contact Wilson coefficient becomes

  c_i(q) = c_i⁰ [ m_med² / (m_med² + q²) ]ⁿ,  n = 1 (one mediator), n = 2 (two mediators; e.g. dipole–dipole or a box with two propagators).

The recall of the propagator form is certain. In the long-range limit m_med → 0, c_i ∝ q^(−2n) (overall normalisation irrelevant for N_lo). For *elastic* scattering at fixed recoil energy q² = 2 m_A E_R, so the differential rate carries the exact factor

  P_n(E_R) = [ m_med² / (m_med² + 2 m_A E_R) ]^(2n)

per isotope; we evaluate it at the mean xenon mass A = 131.29 (m_A = 122.3 GeV) and multiply the WimPyDD contact spectrum. This is P051's factorisation (validated there at ≤ 6 % for inelastic kinematics); we re-validate it for the elastic case in §4 (≤ 3.7 % pointwise, ≤ 2.2 % in N_lo). At the reference energies: q(5.4 keV) = 0.036 GeV, q(30 keV) = 0.086 GeV, q(55 keV) = 0.116 GeV, q(248 keV) = 0.246 GeV.

**Couplings.** Isoscalar LZ/Anand unit coupling c_i^s = 1/m_v² (m_v = 246.2 GeV) = WimPyDD c⁰ = 2/m_v² (P003 convention via `lz.wd_c_from_anand`); "O1p" is the proton-only O1 (c_p = 1/m_v², c_n = 0), the elastic analogue of the dark-photon coupling used in P011/P051. c_i⁰ is the q → 0 coefficient; for n = 1 it equals g_χ g_N / m_med², so we also quote g_χ g_N = c_i^s m_med² (Anand normalisation, c_p = c_n = c^s).

**Observables.** Efficiency-weighted window rates R_lo (5.4–55 keV), R_hi (200–270 keV), R_100–200, R_roi (1–330 keV). N_lo = R_lo/R_hi (coupling-independent). Percentile of 248 keV = fraction of the efficiency-weighted ROI spectrum below 248 keV, in true energy and after Gaussian smearing σ(E) = 23 keV √(E/248 keV) (LZ's ±23 keV statistical error at 248 keV; √E scaling recalled/likely). Coupling for one event in 200–270 keV: c^s m_v² = (R_hi,unit × 2.84 t·yr)^(−1/2); "Λ-equivalent" Λ = (c^s)^(−1/2) (a bookkeeping scale for the dimension-6-normalised coefficient; for O6, O9, O10, O14, O15 the operators carry extra 1/m_N factors so Λ is not a physical mediator mass). Lone-event criterion: N_lo ≤ 5 (loose) and ≤ 3 (strict), P003's recalled 2024-tolerance (uncertain); P016's calibration of LZ's own profile likelihood tolerates 1.6–2.2 companions, so ≤ 3 is closer to what LZ's fit does.

**m_med,min and the contact-like mass.** On a 61-point log grid 0.01–10 GeV we log-interpolate the mass at which N_lo first drops below 5 (3), the mass at which N_lo is within ×1.5 (×2) of its contact value ("contact-like"), the log-midpoint transition mass between the contact and m → 0 values, and the local log-slope d ln N_lo/d ln m between 0.1 and 0.3 GeV.

**Power counting of the m → 0 limit.** The rate is |c(q)|² × (the operator's intrinsic q-power in the Anand response functions; recalled, certain/likely): O1 → M q⁰; O4 → (Σ′+Σ″) q⁰; O7 → Σ′ v⊥² q⁰; O8 → M v⊥² + Δ q²; O12 → Σ v⊥² + Φ q²; O3, O5, O9, O10, O11, O13, O14 → q²; O6 → Σ″ q⁴; O15 → (Σ′ + Φ″ v⊥²) q⁴. One q⁻² propagator divides the rate by q⁴, so only the q⁴ operators (O6, O15) return to a q⁰ contact-like shape (O6/q² has the Σ″ shape of O4; O15/q² an O12-like shape); q² operators overshoot to q⁻² and q⁰ operators to q⁻⁴. This corrects the naive expectation "O_i × q⁻² → the contact operator one q² lower", which holds at amplitude level but not for the rate.

**Analytic long-range SI.** For O1 with m_med → 0, dR/dE ∝ F²(E) η(v_min(E)) / E² (n = 1) or /E⁴ (n = 2). Hence N_lo(q⁻⁴) / N_lo(contact) = ⟨E⁻²⟩_lo / ⟨E⁻²⟩_hi with rate-weighted means over the efficiency-weighted windows. The pure kinematic ratio ∫dE/E² (no form factor, no halo, no efficiency) is (1/5.4 − 1/55)/(1/200 − 1/270) = 128.8 (111.7 with the efficiency).

**Photon-mediated magnetic dipole (P012 coefficients, recalled/likely, directdm-consistent per P031).** L = (μ_χ/2) χ̄σ^{μν}χF_{μν}: c1^N = eμ_χQ_N/(2m_χ), c5^N = 2eμ_χ m_N Q_N/q², c4^N = eμ_χ g_N/m_N, c6^N = −eμ_χ g_N m_N/q² (e = 0.30282, g_p = 5.5857, g_n = −3.8261). We compute full, charge (c1 + c5), spin (c4 + c6) and each coefficient alone at μ_χ = 1 μ_N (N_lo is μ-independent). If the photon is replaced by a massive kinetically-mixed vector of mass m, every coefficient acquires q²/(q² + m²), so the rate is multiplied by [q²/(q² + m²)]²; the heavy limit is the contact "dipole × charge/magnetic current" structure (P031's Q15-charge, L10-like).

## 3. Inputs

| input | source |
|---|---|
| WimPyDD contact spectra O1, O3–O15 (isoscalar), O1 proton-only, 1 TeV, 136 energies 1–330 keV | this work, `P074_contact_spectra.npz`; reproduce P003's N_lo (§4) |
| Halo | `lz.wd_halo()` Sun-frame Baxter SHM (v0 = 238, v_esc = 544, v_sun peculiar; explicit v_min grid) |
| Efficiency | P003 model via `lz.LZ` (plateau 0.96; 50 % at 5.4 / 269.9 keV) |
| Exposure 2.84 t·yr | LZ paper (`lz.LZ['exposure_tyr']`) |
| Event energy 248 ± 23 keV | LZ paper, Table I / abstract |
| Table S6 (L10, L16 = 3.4σ; others) and the Theory paragraph (NREFT basis) | `inputs/LZ_arXiv_2609.02823_fulltext.tex` l.35–46, 850–909 |
| P003 N_lo per operator; P012 photon dipole 376 and c1–c6; P044 photon anapole 26.9, EDM 18 100, O11×q² 2.19, O9×q² 0.024; P031 Λ list; P040 contact floor 350/1184; P051 light-mediator inelastic | corpus papers |
| Constants e, g_p, g_n, m_N, m_v | recalled (certain) / `lzcommon` |

## 4. Validation

*Contact spectra vs P003 (run log).* N_lo(1 TeV): O1^s 2754 (P003 2752), O3 12.7 (12.6), O4 28.1 (28), O5 14.2 (14.2), O6 0.435 (0.43), O7 32.4 (32–36), O8 156 (156), O9 2.07 (2.1), O10 3.09 (3.1), O11 259 (258), O12 100 (41–100), O13 4.91 (4.9), O14 2.35 (2.3), O15 1.91 (1.9); O1p 1245 (P012's photon-dipole table gives 1184 for the O1-proton-type charge piece at 200 GeV; P044's O8-charge/O1p-type numbers are consistent). Agreement to the quoted precision.

*Factorisation vs full WimPyDD q-dependent closures* (`P074_validation.csv`; ten (operator, m_med, n) cases, 11 energies each, plus window-integrated N_lo on an independent 6 keV grid):

| case | N_lo full WimPyDD | N_lo factorised | ratio | pointwise ratio range |
|---|---|---|---|---|
| O1s, 0.1 GeV, n=1 | 5.307e4 | 5.263e4 | 1.008 | 0.971–1.018 |
| O1s, 0.3 GeV, n=2 | 1.406e4 | 1.408e4 | 0.998 | 0.971–1.017 |
| O6s, 0.3 GeV, n=1 | 0.9436 | 0.9498 | 0.994 | 1.000–1.009 |
| O6s, 0.1 GeV, n=2 | 76.72 | 78.42 | 0.978 | 1.007–1.037 |
| O10s, 0.3 GeV, n=1 | 6.813 | 6.881 | 0.990 | 1.000–1.009 |
| O9s, 1 GeV, n=1 | 2.272 | 2.287 | 0.994 | 0.999–1.002 |
| O4s, 0.1 GeV, n=1 | 574 | 566.9 | 1.012 | 1.004–1.021 |
| O11s, 0.03 GeV, n=1 | 2.475e4 | 2.449e4 | 1.010 | 0.966–1.020 |
| O1p, 0.1 GeV, n=1 | 2.446e4 | 2.413e4 | 1.014 | 0.966–1.015 |
| O5s, 0.3 GeV, n=1 | 32.31 | 32.71 | 0.988 | 1.000–1.007 |

Max pointwise deviation 3.7 %, rms 1.2 % (110 points); max N_lo deviation 2.2 %. The residual is the isotope spread of q² (±4 %) and the 6 keV grid of the check. The factorisation is therefore exact for our purposes, and we use it for the full grid.

*Photon dipole.* Full N_lo = 375.2 vs P012's 375.8/376 (same coefficients, same code; the 0.2 % difference is the finer energy grid).

## 5. Results — N_lo(operator, m_med, n) at 1 TeV (Sun-frame halo)

n = 1 (`P074_Nlo_wide_n1.csv`; columns: m → 0 (q⁻² amplitude), 0.01, 0.03, 0.1, 0.3, 1, 3, 10 GeV, contact):

| op | q⁻² | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 | 10 | contact |
|---|---|---|---|---|---|---|---|---|---|
| O1s | 7.72e5 | 7.10e5 | 4.20e5 | 5.26e4 | 6222 | 3021 | 2783 | 2757 | 2754 |
| O3s | 1027 | 980 | 722 | 174 | 27.9 | 13.9 | 12.8 | 12.7 | 12.7 |
| O4s | 8283 | 7616 | 4495 | 567 | 65.7 | 31.0 | 28.5 | 28.2 | 28.1 |
| O5s | 2382 | 2228 | 1456 | 247 | 32.7 | 15.7 | 14.4 | 14.2 | 14.2 |
| O6s | 27.8 | 26.7 | 20.4 | 5.55 | 0.950 | 0.477 | 0.440 | 0.436 | 0.435 |
| O7s | 1.29e4 | 1.18e4 | 6732 | 740 | 77.3 | 35.8 | 32.7 | 32.4 | 32.4 |
| O8s | 4.75e4 | 4.37e4 | 2.59e4 | 3253 | 369 | 172 | 158 | 156 | 156 |
| O9s | 506 | 469 | 291 | 40.9 | 4.85 | 2.29 | 2.10 | 2.08 | 2.07 |
| O10s | 353 | 333 | 228 | 46.1 | 6.88 | 3.39 | 3.12 | 3.09 | 3.09 |
| O11s | 4.03e4 | 3.77e4 | 2.45e4 | 4114 | 568 | 283 | 262 | 259 | 259 |
| O12s | 1.46e4 | 1.37e4 | 9105 | 1631 | 226 | 110 | 101 | 100 | 100 |
| O13s | 417 | 397 | 291 | 69.0 | 10.9 | 5.39 | 4.97 | 4.92 | 4.91 |
| O14s | 577 | 536 | 332 | 46.5 | 5.50 | 2.59 | 2.38 | 2.36 | 2.35 |
| O15s | 99.6 | 96.4 | 77.0 | 23.3 | 4.13 | 2.09 | 1.93 | 1.91 | 1.91 |
| O1p | 3.48e5 | 3.20e5 | 1.90e5 | 2.41e4 | 2843 | 1368 | 1258 | 1246 | 1245 |

n = 2 (`P074_Nlo_wide_n2.csv`): O1s 4.86e8 / 3.94e8 / 1.09e8 / 1.10e6 / 1.41e4 / 3315 / 2813 / 2759 / 2754; O6s 6288 / 5384 / 2129 / 78.4 / 2.07 / 0.523 / 0.445 / 0.436 / 0.435; O9s 2.91e5 / 2.39e5 / 7.13e4 / 874 / 11.3 / 2.52 / 2.12 / 2.08 / 2.07; O10s 1.46e5 / 1.21e5 / 3.97e4 / 782 / 15.4 / 3.72 / 3.15 / 3.09 / 3.09; O14s 3.33e5 / 2.73e5 / 8.14e4 / 994 / 12.8 / 2.86 / 2.41 / 2.36 / 2.35; O15s 1.45e4 / 1.28e4 / 5920 / 308 / 8.94 / 2.29 / 1.95 / 1.91 / 1.91; O4s 5.74e6 → 154 (0.3) → 34.2 (1); O1p 2.23e8 → 6501 (0.3) → 1503 (1). Full table in the CSV; heat map `figures/P074_heatmap_Nlo.png`.

**Reading.** Every entry increases monotonically as m_med decreases: a light mediator only ever softens a spectrum. No operator that fails as a contact interaction (O1, O3, O4, O5, O7, O8, O11, O12, O1p) is rescued at any mediator mass; the compatible set is a subset of P003's contact set (O6, O9, O10, O14, O15, marginally O13), and each survivor has a minimum mediator mass.

### 5.1 Minimum mediator mass per operator (`P074_mmed_min.csv`)

| op | n | N_lo contact | N_lo (m→0) | m_min(N_lo ≤ 5) [GeV] | m_min(N_lo ≤ 3) | m contact-like (×1.5) | (×2) | transition mass | slope 0.1–0.3 GeV |
|---|---|---|---|---|---|---|---|---|---|
| O6s | 1 | 0.435 | 27.8 | 0.106 | 0.143 | 0.449 | 0.324 | 0.131 | −1.60 |
| O6s | 2 | 0.435 | 6288 | 0.218 | 0.259 | 0.661 | 0.491 | 0.112 | −3.30 |
| O15s | 1 | 1.91 | 99.6 | 0.258 | 0.420 | 0.446 | 0.322 | 0.137 | −1.57 |
| O15s | 2 | 1.91 | 1.45e4 | 0.404 | 0.619 | 0.658 | 0.489 | 0.119 | −3.22 |
| O9s | 1 | 2.07 | 506 | 0.293 | 0.494 | 0.469 | 0.342 | 0.111 | −1.94 |
| O9s | 2 | 2.07 | 2.91e5 | 0.446 | 0.719 | 0.685 | 0.512 | 0.103 | −3.95 |
| O14s | 1 | 2.35 | 577 | 0.324 | 0.621 | 0.469 | 0.342 | 0.111 | −1.94 |
| O14s | 2 | 2.35 | 3.33e5 | 0.488 | 0.894 | 0.685 | 0.511 | 0.103 | −3.95 |
| O10s | 1 | 3.09 | 353 | 0.412 | never | 0.455 | 0.330 | 0.119 | −1.73 |
| O10s | 2 | 3.09 | 1.46e5 | 0.608 | never | 0.669 | 0.497 | 0.104 | −3.57 |
| O13s | 1 | 4.91 | 417 | 2.34 | never | 0.453 | 0.328 | 0.126 | −1.68 |
| O13s | 2 | 4.91 | 1.18e5 | 3.31 | never | 0.666 | 0.496 | 0.109 | −3.46 |
| O1s | 1 | 2754 | 7.72e5 | never | never | 0.456 | 0.333 | 0.106 | −1.94 |
| O1s | 2 | 2754 | 4.86e8 | never | never | 0.667 | 0.497 | 0.099 | −3.96 |
| O4s | 1 | 28.1 | 8283 | never | never | 0.469 | 0.342 | 0.107 | −1.96 |
| O1p | 1 | 1245 | 3.48e5 | never | never | 0.460 | 0.336 | 0.107 | −1.94 |

(all fourteen operators in the CSV; the O3, O5, O7, O8, O11, O12 rows are "never" with contact-like masses 0.45–0.48 / 0.66–0.69 GeV.)

**The general rule.** The contact-like mass is remarkably operator-independent: N_lo is within 50 % of its contact value for m_med ≥ 0.45–0.48 GeV (n = 1) and ≥ 0.66–0.69 GeV (n = 2), i.e. m_med ≳ 1.8 q(248 keV) for one propagator and ≳ 2.7 q for two; within a factor 2 for m_med ≥ 0.32–0.35 / 0.49–0.52 GeV (≈ 1.3 q / 2.0 q). This is the "m_med ≳ 2–3 q ≈ 0.5–0.8 GeV" rule, now quantified: below ≈ 0.5 GeV every spectrum is measurably softer than its contact version; below the transition mass ≈ 0.10–0.14 GeV (q at 30–55 keV) the spectrum is essentially the long-range one. Because the momentum transfer spans only a factor 7 between the windows (0.036–0.25 GeV), there is no extended intermediate regime: the local slope between 0.1 and 0.3 GeV is −1.6 to −2.0 (n = 1) and −3.2 to −4.0 (n = 2) rather than the asymptotic −4n; for O1 the steepest local slopes are −2.27 (n = 1, at 0.11 GeV) and −4.75 (n = 2, at 0.10 GeV).

### 5.2 Survivors: position of 248 keV, LZ's empty band, couplings (`P074_survivors.csv`, n = 1)

| op | m_med [GeV] | N_lo | 248 keV percentile (true / smeared) | R(100–200)/R(200–270) | N_hi at unit coupling | c^s m_v² for 1 event in 200–270 | Λ = (c^s)^(−1/2) [GeV] | g_χg_N = c^s m² |
|---|---|---|---|---|---|---|---|---|
| O6 | contact | 0.435 | 93.5 / 92.6 % | 2.04 | 0.121 | 2.87 | 145 | — |
| O6 | 1 | 0.477 | 93.9 / 93.0 | 2.13 | 0.109 | 3.04 | 141 | 5.0e-5 |
| O6 | 0.3 | 0.950 | 95.9 / 95.2 | 2.77 | 0.046 | 4.68 | 114 | 6.9e-6 |
| O9 | contact | 2.07 | 96.5 / 95.5 | 2.07 | 1.19 | 0.916 | 257 | — |
| O9 | 1 | 2.29 | 96.7 / 95.8 | 2.14 | 1.07 | 0.967 | 250 | 1.6e-5 |
| O9 | 0.3 | 4.85 | 98.1 / 97.5 | 2.73 | 0.456 | 1.48 | 202 | 2.2e-6 |
| O10 | contact | 3.09 | 97.5 / 97.0 | 3.27 | 7.53 | 0.365 | 408 | — |
| O10 | 1 | 3.39 | 97.7 / 97.2 | 3.41 | 6.75 | 0.385 | 397 | 6.3e-6 |
| O14 | contact | 2.35 | 96.8 / 95.8 | 2.15 | 5.9e-7 | 1306 | 6.8 | — |
| O14 | 1 | 2.59 | 97.0 / 96.1 | 2.23 | 5.3e-7 | 1378 | 6.6 | 2.3e-2 |
| O15 | contact | 1.91 | 95.8 / 94.9 | 0.78 | 1.08 | 0.962 | 251 | — |
| O15 | 1 | 2.09 | 96.1 / 95.2 | 0.81 | 0.969 | 1.02 | 244 | 1.7e-5 |
| O15 | 0.3 | 4.13 | 97.8 / 97.2 | 1.03 | 0.409 | 1.56 | 197 | 2.3e-6 |

The mediator moves the event further into the tail of the accepted spectrum (O6: 93.5 → 95.9 %; O9: 96.5 → 98.1 %; O15: 95.8 → 97.8 % at 0.3 GeV) and fills LZ's empty 100–200 keV band: the expected 100–200 keV count per 200–270 keV event rises from 2.0 to 2.8 (O6, O9) — the elastic version of P051's inelastic argument (4.1 → 6.8). The coupling required for one 200–270 keV event grows as m_med falls because the propagator suppresses the high-q rate: for O6, c^s m_v² = 2.87 (contact) → 3.04 (1 GeV) → 4.68 (0.3 GeV), i.e. the q → 0 coefficient must be ×1.6 larger at 0.3 GeV; in terms of the product of couplings g_χ g_N = c^s m_med², one event needs 5 × 10⁻⁵ at 1 GeV and 7 × 10⁻⁶ at 0.3 GeV (O6), 1.6 × 10⁻⁵ / 2.2 × 10⁻⁶ (O9). The Λ column is the dimension-6 bookkeeping scale; P031's physical Λ for the pseudoscalar route to O6 (1.5 GeV) includes the m_q/m_N and 1/m_N² factors of the reduction.

### 5.3 Light-mediator spin-independent dark matter (O1, O1p)

The classic "light-mediator SI DM" (Fornengo–Panci–Regis 2011; Kaplinghat–Tulin–Yu 2014; recalled, likely) is O1 with c ∝ 1/(m² + q²). N_lo rises from 2754 (contact) to 3021 (1 GeV), 6222 (0.3 GeV), 5.3 × 10⁴ (0.1 GeV), 7.7 × 10⁵ (m → 0); with two propagators (n = 2) to 4.9 × 10⁸. The proton-only (dark-photon) version: 1245 → 2843 (0.3) → 3.5 × 10⁵. Scaling: N_lo ∝ m^(−1.94) (n = 1) and m^(−3.96) (n = 2) between 0.1 and 0.3 GeV, saturating below ≈ 0.05 GeV. The peak of the efficiency-weighted spectrum moves from 9.8 keV (contact) to the threshold (1 keV) for m ≤ 0.03 GeV; R(100–200)/R(200–270) rises from 11.2 to 23.5.

**Analytic check.** The rate-weighted means over the efficiency-weighted windows of the contact spectrum give ⟨E⁻²⟩_lo = 6.11 × 10⁻³ keV⁻² (E_eff,lo = 12.8 keV) and ⟨E⁻²⟩_hi = 2.18 × 10⁻⁵ keV⁻² (E_eff,hi = 214 keV), ratio 280; N_lo(q⁻⁴) predicted = 2754 × 280 = 7.70 × 10⁵ vs numerical 7.72 × 10⁵ (0.2 %; this checks the implementation, since the two are the same integral). The physical content is the decomposition: the pure kinematic ∫dE/E² ratio is 129 (112 with efficiency), while the shell-model form factor and the halo function shift the effective energies to 12.8 and 214 keV and raise the enhancement to 280 — the form factor already makes the contact spectrum fall steeply at high E, so the 1/E² tilt is felt at the *low* end (E_eff,lo = 12.8 keV, well below the window's geometric mean of 17 keV). For n = 2: ⟨E⁻⁴⟩ ratio 1.76 × 10⁵, predicted 4.84 × 10⁸ vs numerical 4.86 × 10⁸. Conclusion: light-mediator SI DM is excluded as the origin of a lone 248 keV recoil by a factor 10³–10⁹ in N_lo at every mediator mass, worse than the contact case (P003) and worse than the velocity-independent boosted-DM floor of P040 (350–1184), which a light mediator would also only raise.

### 5.4 Operator collapse in the m → 0 limit (`P074_operator_collapse.csv`)

| operator × propagator (n) | N_lo (m → 0) | same-operator contact | reference contact | ratio |
|---|---|---|---|---|
| O6 (n = 1): Σ″ q⁴/q⁴ → Σ″ q⁰ | 27.8 | 0.435 | O4 28.1 | 0.988 |
| O15 (n = 1): q⁴/q⁴ → q⁰ | 99.6 | 1.91 | O12 100.3 | 0.993 |
| O6 (n = 2): Σ″/q⁴ | 6288 | 0.435 | O4 28.1 | 224 |
| O10 (n = 1): Σ″ q²/q⁴ → Σ″/q² | 353 | 3.09 | O4 28.1 | 12.6 |
| O9 (n = 1): Σ′/q² | 506 | 2.07 | O4 28.1 | 18.0 |
| O11 (n = 1): M/q² | 4.03e4 | 259 | O1 2754 | 14.6 |
| O5 (n = 1): M v⊥²/q² + Δ | 2382 | 14.2 | O8 156 | 15.3 |
| O3 (n = 1) | 1027 | 12.7 | O7 32.4 | 31.7 |
| O14 (n = 1): Σ′ v⊥²/q² | 577 | 2.35 | O7 32.4 | 17.8 |
| O13 (n = 1) | 417 | 4.91 | O12 100 | 4.15 |
| O12, O4, O1 (n = 1): q⁻⁴ | 1.46e4, 8283, 7.72e5 | 100, 28.1, 2754 | — | — |

O6 with a massless mediator reproduces O4's N_lo to 1.2 % (the longitudinal Σ″ and the total Σ′ + Σ″ responses of xenon have the same spectral shape at this level), and O15/q² reproduces O12 to 0.7 %: the "one q² lower" collapse is exact for the q⁴ operators. For the q² operators (O9, O10, O11, O5, O14, O3, O13) a single q⁻² propagator over-corrects: the rate goes as q⁻², and N_lo lands 13–32× above the q⁰ reference contact operator — the P044 photon EDM (O11 amplitude/q², rate M/q²: 18 100 proton-only vs our isoscalar 4.0 × 10⁴) and photon anapole are instances. This is why the assignment's expectation "O6/q² is O4-like with N_lo ≈ 28" is confirmed while "O10/q² is O4-like" is not (353).

### 5.5 Photon-mediated magnetic dipole, decomposed (`P074_results.json` → `dipole`; figure `P074_dipole_decomposition.png`)

| piece | N_lo | share of R_lo | share of R_hi | share of dR/dE at 10 keV | at 248 keV |
|---|---|---|---|---|---|
| full (c1 + c5 + c4 + c6) | **375.2** | 1 | 1 | 1 | 1 |
| charge, c1 + c5 | 18 010 | 0.879 | 0.018 | 0.896 | 0.009 |
|  c5 alone (2eμm_N Q/q², O5) | 18 200 | 0.879 | 0.018 | 0.895 | 0.009 |
|  c1 alone (eμQ/2m_χ, O1) | 1245 | 0.0007 | 0.0002 | 0.0005 | 0.0001 |
| spin, c4 + c6 | 31.9 | 0.076 | 0.892 | 0.074 | 0.901 |
|  c4 alone | 29.6 | 0.206 | 2.60 | 0.136 | 2.73 |
|  c6 alone | 28.4 | 0.134 | 1.77 | 0.064 | 1.89 |

Charge–spin interference is +4.5 % of R_lo and +9.0 % of R_hi. Within the spin part c4 and c6 interfere destructively (the combination c4 + (q²/m_N²)c6 = 0 removes the longitudinal Σ″ piece, leaving the transverse L10 structure): the c4 + c6 rate is 0.22 (lo) / 0.20 (hi) of the sum of the separate rates. **Which term drives N_lo = 376.** The c5 (charge–dipole, rate ∝ Z² M v⊥²/q² ∝ 1/E_R) term supplies 88 % of the low-energy rate and only 1.8 % of the high-energy rate; the transverse spin term supplies 89 % of the high-energy rate. So N_lo(full) ≈ R_lo(charge)/R_hi(spin) = 0.879 × 3.25 × 10¹² / (0.892 × 8.67 × 10⁹) ≈ 369: the 248 keV event would be a spin–spin scatter and the 376 companions are charge scatters. The c1 term (∝ 1/m_χ) is irrelevant at 1 TeV (0.07 % of R_lo). This confirms P012's statement with the term-by-term shares.

**Massive dark-photon-mediated dipole** (`P074_dipole_massive_mediator.csv`): multiplying all four coefficients by q²/(q² + m²), N_lo = 375 (m = 0) → 351 (0.01 GeV) → 232 (0.03) → 40.5 (0.1) → 5.46 (0.3) → 2.64 (1) → 2.42 (3) → 2.40 (heavy limit; P031's Q15-charge contact structure gives 2.52). The charge share of R_hi stays 1.7–1.8 % at every m. The dipole portal becomes lone-event compatible for m ≥ 0.32 GeV (N_lo ≤ 5) or ≥ 0.64 GeV (≤ 3) — the same threshold as the operator map, because the charge piece is a q² operator (O5) dressed with q⁻².

### 5.6 UV summary (`P074_uv_summary.csv`)

| UV structure → NR route | N_lo contact | 0.3 GeV | m → 0 | verdict (N_lo ≤ 5) |
|---|---|---|---|---|
| scalar S–S or vector V–V mediator → O1 (light-mediator SI) | 2754 | 6222 | 7.7e5 | excluded at every m_med |
| kinetically mixed dark photon → O1 proton-only | 1245 | 2843 | 3.5e5 | excluded at every m_med |
| two propagators → O1 × q⁻⁴ | 2754 | 1.4e4 | 4.9e8 | excluded |
| pseudoscalar P–P → O6 (P031) | 0.435 | 0.95 | 27.8 | allowed for m_med ≥ 0.11 GeV (0.14 for ≤ 3) |
| S–P → O10 (P031) | 3.09 | 6.88 | 353 | allowed for m_med ≥ 0.41 GeV |
| DM tensor × quark axial current → O9 (P031) | 2.07 | 4.85 | 506 | allowed for m_med ≥ 0.29 GeV (0.49) |
| same with γ5 → O14 | 2.35 | 5.50 | 577 | allowed for m_med ≥ 0.32 GeV (0.62) |
| q⁴-spin O15 | 1.91 | 4.13 | 99.6 | allowed for m_med ≥ 0.26 GeV (0.42) |
| axial-vector A–A → O4 | 28.1 | 65.7 | 8283 | excluded at every m_med |
| V–A → O7; anapole-type A–V charge → O8; EDM-type → O11 | 32 / 156 / 259 | 77 / 369 / 568 | 1.3e4 / 4.8e4 / 4.0e4 | excluded |
| magnetic dipole via photon / massive dark photon | 2.40 (heavy) | 5.46 | 375 | allowed for m_med ≥ 0.32 GeV |

P031's pion/η poles for the isovector pseudoscalar route (O10/(m_π² + q²), N_lo 3.1 → 25.8) are a case of this map with m_med = m_π = 0.14 GeV: our O10 curve gives N_lo = 46 at 0.1 GeV and 6.9 at 0.3 GeV, bracketing P031's 25.8 (different isospin and pole structure). P051's inelastic case (a light mediator does not rescue δ ≈ 300 keV and pushes 248 keV to the 99.8th percentile) is the inelastic counterpart of §5.2.

## 6. Robustness

- Efficiency: P003 found ≤ 9 % variations of N_lo across efficiency variants for contact operators; the propagator factor is efficiency-independent, so the m_med thresholds shift by less than the grid spacing.
- Halo: N_lo is a ratio of rates for velocity-suppressed and unsuppressed windows; P003's mass dependence (200 GeV ≈ 2.9×, 4000 GeV ≈ 0.87×) carries over unchanged since the propagator does not depend on m_χ. Sun-frame vs annual mean differ by ≪ 10 % for elastic recoils below 270 keV at 1 TeV (v_min(270 keV, 1 TeV) = 210 km/s is far from the tail; P035 differences arise only near δ_max).
- Isotope spread: validated at ≤ 3.7 % pointwise.
- Criterion: with P016's stricter 1.6–2.2 companions, m_min moves up (O6 0.17–0.2 GeV; O9, O14, O15 only their contact values pass; O10, O13 fail as they do already as contact operators).

## 7. Failed or abandoned approaches

- The naive "O_i × q⁻² collapses onto the contact operator one q² lower" power counting was found wrong at the rate level for q² operators (§5.4) and replaced by explicit |c(q)|² counting; the script's note strings were corrected and the analysis stage re-run (the numbers did not change).
- A separate Σ″-only O4 Hamiltonian (WimPyDD response-keyed syntax) to isolate the longitudinal part was not built; the O6/q² ↔ O4 agreement to 1.2 % made it unnecessary.
- A 2024-exposure constant (4.2 t·yr) was defined in the script but not used; no 2024 count is quoted.

## 8. Discussion

The map says one thing in several ways: momentum transfer enters the LZ problem twice, once in the operator (q² per power) and once in the propagator (q⁻² per mediator), and a lone high-energy recoil needs the net power to be ≥ 0 — in practice +2 or +4 in the rate (P003's q⁴ spin operators). A mediator lighter than ≈ 0.5 GeV undoes part of the operator's q-weighting for every operator by the same factor (the contact-like mass is operator-independent to ±3 %), and a mediator lighter than ≈ 0.1 GeV undoes it completely (O6 → O4's 28, O15 → O12's 100). Since P031 found that the compatible contact operators need Λ = 1.5–62 GeV, the natural reading is a mediator in the 0.5–60 GeV window: light enough for the small Λ, heavy enough (≳ 2q) to leave the q⁴ weighting intact. Mediators below ≈ 0.3 GeV coupled to quarks are excluded as the origin unless the coupling is purely to spin with q⁴ weighting (O6 survives to 0.11 GeV). The dark-photon-mediated magnetic dipole joins the survivors only above 0.32 GeV; the photon itself (m = 0) fails through its charge term, as P012 said. None of this bears on whether the event is dark matter; it fixes the mediator-mass floor of every elastic interpretation.

## 9. Figures

- `figures/P074_heatmap_Nlo.png` — log₁₀ N_lo for every operator × mediator mass, n = 1 (left) and n = 2 (right); boxed cells N_lo ≤ 5.
- `figures/P074_Nlo_vs_mmed.png` — N_lo(m_med) on the fine grid for O1 (n = 1, 2), O4, O11, O10, O9, O6 (n = 1, 2), O15, O14; vertical lines at q(30 keV) and q(248 keV); shaded compatible region.
- `figures/P074_spectra_survivors.png` — efficiency-weighted spectra per ROI event for O6, O10, O9 (contact and 0.3/1 GeV) and O1 (contact, 0.1 GeV, q⁻²).
- `figures/P074_dipole_decomposition.png` — photon magnetic-dipole spectrum and its c1, c5, c4, c6 pieces at μ_χ = 1 μ_N.

## 10. References

LZ Collaboration, arXiv:2609.02823 (2026). A. L. Fitzpatrick, W. Haxton, E. Katz, N. Lubbers, Y. Xu, JCAP 02 (2013) 004. N. Anand, A. L. Fitzpatrick, W. C. Haxton, PRC 89, 065501 (2014). N. Fornengo, P. Panci, M. Regis, PRD 84, 115002 (2011) (recalled, likely). M. Kaplinghat, S. Tulin, H.-B. Yu, PRD 89, 035009 (2014) (recalled, likely). I. Jeong, S. Kang, S. Scopel, G. Tomar, CPC 276, 108342 (2022). D. Baxter et al., EPJC 81, 907 (2021). Corpus: P003, P012, P016, P031, P040, P044, P051.

## 11. Tools and provenance

Mirrors `output/provenance/P074.json`. Python 3.12.13 (`.venv/bin/python`, run from `/Users/reza/LZ_simulation`); WimPyDD 2.0.4 (`eft_hamiltonian` with closure-captured constant, propagator-dressed and q^(−2) coefficients; `diff_rate` via `lzcommon.wd_rate`; `streamed_halo_function` via `lzcommon.wd_halo`); numpy 2.5.3; scipy 1.18.1 (`special.erf`, `integrate.trapezoid`, `integrate.cumulative_trapezoid`); pandas 3.0.5; matplotlib 3.11.2 (Agg); `common/lzcommon.py`. Script stages and wall times: contact 47 s, validate 18 s, dipole 32 s, analysis 120 s and 72 s (re-run after correcting note strings and adding two columns). WimPyDD-generated files: none beyond its internal response-function cache (we used `diff_rate`, not `wimp_dd_rate`). Files read: PAPER_GUIDE.md; 00_evidence_dossier.md; results_ledger.csv (via pandas); papers P003, P012, P031, P040, P044, P051; code P044_anapole_edm.py, P051_light_mediator.py (l.1–200), P012_magnetic_dipole.py (l.222–281); lzcommon.py (l.300–386); WimPyDD/package.py (l.972–1011, 1250–1294); fulltext.tex (l.35–46, 850–911); P044 details.md (head), P044.json (head), P044_results_table.csv, P012_photon_dipole.csv; ENVIRONMENT_versions.txt (head). Recalled items: 9 (see JSON; 1 uncertain: the 3–5 event tolerance).
