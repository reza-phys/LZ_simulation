# P032 — research record
**Why the isovector O1 gets 1.3σ at δ = 0 while the isoscalar gets none: isospin, the M-response node and what isospin violation can buy**
Simulated date 2026-09-09 · hep-ph · nuclear-physics-minded DM phenomenologists · category IDM/EFT

Script: `output/code/P032_isovector.py` (run: `.venv/bin/python output/code/P032_isovector.py`; `--fast` reuses the cached WimPyDD basis spectra `output/work/P032/P032_basis_spectra.npz`). Full console output in `output/work/P032/run_log.txt`. All numbers below are from that run unless flagged.

## 1. Motivation

LZ's Tables S6/S7 give, for elastic spin-independent scattering at m ≥ 400 GeV, local significance 0.0σ for the isoscalar coupling (L1ˢ, L5ˢ, O1ˢ at δ = 0) but 0.3/1.3/1.2σ (L1ᵛ), 0.7/1.3/1.3σ (L5ᵛ) and 0.8/1.3/1.1σ (O1ᵛ, δ = 0) at 400/1000/4000 GeV for the isovector coupling. Since L1 and L5 differ only by a scalar constant (LZ LEE section, tex l. 495) and O1 at δ = 0 is the same operator, these are three readings of the *same* spectrum. P003 found N_lo (5.4–55 keV events per 200–270 keV event, 1 TeV) = 2752 (O1ˢ) vs 477 (O1ᵛ); P016 reproduced Z = 0.48 (O1ˢ) and 1.65 (O1ᵛ) with a few-bin likelihood; P021 noted O1ᵛ runs 0.2–0.6σ high relative to Table S7 and that the O1ᵛ profile Z is flat in δ (3.55–3.68). P011 found the proton-only node at 284–291 keV and a conversion factor ≈ 4 instead of (A/Z)² = 5.9; P017 located the isoscalar M node at 266.9 keV with an isotope spread 251–278 keV; P007 described the Higgsino as isovector-dominated (c¹/c⁰ = −1.17). This paper explains the isoscalar/isovector asymmetry quantitatively through the isovector M response and asks how far general isospin violation f_n/f_p can go.

## 2. Framework

### 2.1 Couplings and the quadratic form
O1 (NREFT, Fitzpatrick et al. 2013; Anand et al. 2014) with nucleon couplings c_p, c_n. WimPyDD convention (settled by P003 by digitising LZ Fig. 1; confirmed by P007): c⁰ = c_p + c_n, c¹ = c_p − c_n. With r ≡ f_n/f_p = c_n/c_p:

  c⁰ = c_p(1 + r), c¹ = c_p(1 − r);  r = +1 isoscalar (O1ˢ), r = −1 isovector (O1ᵛ), r = 0 proton-only, r = ∞ neutron-only, r = −Z/⟨N⟩ = −0.698 "xenophobic" (Feng, Kumar, Marfatia, Sanford 2011, recalled: the value −0.7 is the standard one, certain).

The differential rate is a quadratic form in the couplings (the nuclear responses W_M^{ττ′} enter bilinearly):

  dR/dE(c⁰, c¹) = (c⁰)² R₀₀(E) + 2 c⁰c¹ R₀₁(E) + (c¹)² R₁₁(E).   (1)

We compute three WimPyDD spectra per mass, (c⁰, c¹) = (1, 0), (0, 1), (1, 1) [GeV⁻²], obtain R₀₁ = [R(1,1) − R₀₀ − R₁₁]/2, and evaluate Eq. (1) for any r. Validation: a fourth directly computed spectrum at (0.3, 1.7) (i.e. r = −0.7) agrees with Eq. (1) to 3.2 × 10⁻¹³ (max relative deviation, 5–265 keV) — machine precision, as it must.

### 2.2 Nuclear response
For O1 only the M response enters: W_M^{ττ′}(q) per isotope, read from WimPyDD's `Xe.func_w[i](q)[nuclear_current['M'], τ, τ′]` (DMFormFactor-v6 one-body density matrices, harmonic-oscillator basis). Define proton/neutron pieces

  W_pp = (W⁰⁰ + 2W⁰¹ + W¹¹)/4,  W_nn = (W⁰⁰ − 2W⁰¹ + W¹¹)/4,  W_pn = (W⁰⁰ − W¹¹)/4.

For J = 0 (¹²⁸,¹³⁰,¹³²,¹³⁴,¹³⁶Xe) and J = ½ (¹²⁹Xe) only the M₀ multipole contributes and W is a single dyad, W^{ττ′} = F_τ F_τ′ (we verified W⁰¹²/(W⁰⁰W¹¹) = 0.995–1.055 at 100 keV); for ¹³¹Xe (J = 3/2) M₀ and M₂ both contribute and W⁰¹²/(W⁰⁰W¹¹) = 0.44 at 100 keV. **Consequence:** "the node" of a coupling combination is a zero of the amplitude Z F_p + rN F_n only for the dyad isotopes; in general (and for natural Xe) it is a *minimum* of the quadratic form c⁰²W⁰⁰ + 2c⁰c¹W⁰¹ + c¹²W¹¹. We therefore define nodes as local minima (P017's definition) and record their depth. For dyad isotopes the signed F_p, F_n are obtained by tracking the sign through their zeros (zeros of W_pn assigned to p or n by which of W_pp/W_pp(0), W_nn/W_nn(0) is the smaller); check: max|F_pF_n − W_pn|/max|W_pn| = 1.5–3.8 × 10⁻⁵ for all six dyad isotopes.

Normalised form factors f_p = F_p/Z, f_n = F_n/N (f(0) = 1); the ratio f_n/f_p = (Z/N) W_pn/W_pp is exact (signed) wherever W_pp ≠ 0. The isovector amplitude Z F_p − N F_n = Z² f_p − N² f_n vanishes where f_n/f_p = Z²/N² (0.52 for ¹²⁹Xe, 0.43 for ¹³⁶Xe); the general combination Z F_p + rN F_n = Z² f_p + rN² f_n vanishes where f_n/f_p = −Z²/(rN²).

### 2.3 Kinematics, halo, efficiency, windows
Baxter-2021 SHM, annual average, via `lz.wd_halo()` (explicit v_min grid to 844 km/s; P002 fix). Natural xenon, WimPyDD abundances (¹²⁴,¹²⁶Xe carry no density matrices; 7 active isotopes = 99.8 %). Rates in events/(t yr keV) for c_p = 1 GeV⁻² (all ratios coupling-independent). Energy grid 1 keV, 1–300 keV (elastic) and 40–300 keV (δ = 300 keV). Efficiency (P003/P016/P021 model): 0.96 plateau, erf edges at 5.4 keV (σ = 3.4 keV) and 269.9 keV (σ = 8 keV); LZ paper: 50 % points 5.4/269.9 keV, 96 % average 14–250 keV. Windows: 5.4–55 keV (2024 low-energy ROI), 55–125, 125–200 (P016 companion bins), 200–270 keV (high-energy window). N_lo = R(5.4–55)/R(200–270).

Node finder: local minima of dR/dE (or of the response) that are at least ×0.6 deeper than the function 15 keV on either side (rejects the ~1 % ripples from the v_min grid); parabolic refinement.

### 2.4 Local significance
P016's few-bin extended likelihood, re-implemented from `work/P016/P016_results.json`: 20 digitised Fig. 5 NR-band bins (−8σ … +2σ, background sum 121.1, data 121; ER-scale nuisance θ with σ = 0.3, profiled), a 125–200 keV bin (b = 0.02, n = 0), and the 200–270 keV bin with n = 1 and b_H = 5.695 × 10⁻⁴ (P016's L10 anchor). Signal per unit ŝ (events in 200–270 keV): N_lo + N_{55–125} in the top-panel bins (Gaussian shape in σ_NR), N_{125–200} in the middle bin. Z = √q₀ (Wilks), ŝ profiled on a log grid + bounded refinement. Compared with Table S6 L1ᵛ/L5ᵛ, Table S7 O1ᵛ (δ = 0) and the isoscalar zeros (`lz.LSIG`, `lz.OSIG`).

### 2.5 Inelastic reference
δ = 300 keV spectra (per Eq. (1)) for the three masses; percentile of 248 keV in the efficiency-weighted true-energy spectrum (as P002) and f(248)/f_peak of the accepted spectrum.

### 2.6 Validation of normalisation
Isoscalar WimPyDD rate for σ_p = σ_n = 10⁻⁴⁵ cm² (c_p = √(πσ/(ħc)²)/μ_p; WimPyDD c⁰ = 2c_p) vs `lz.dRdE_SI` (Helm): ratio 0.974 at 20 keV, 0.998 at 50 keV (1 TeV). Consistent with P003 (≤ 5 % below 100 keV).

## 3. Results

### 3.1 Why the isovector spectrum is different (Fig. 1, Fig. 3)
f_n falls faster with q than f_p (in the shell model the valence neutrons sit in higher orbitals, an effective neutron skin): for ¹²⁹Xe f_n/f_p = 0.948 (20 keV), 0.088 (100 keV), 0.172 (248 keV); ¹³²Xe 0.939, −0.369, −0.091; ¹³⁶Xe 0.926, −5.8, −0.53. Per-isotope zeros (dyad isotopes; ¹³¹Xe minima):

| A | F_p zeros [keV] | F_n zeros [keV] | isoscalar minima | isovector minima | proton-only | xenophobic (r = −0.698) |
|---|---|---|---|---|---|---|
| 128 | 114.5, 302.2 | 104.6, 258.8 | 109.1, 277.6 | 74.8, 190.1 | 114.4, 301.3 | 146.5, 379.9 |
| 129 | 110.7, 299.4 | 101.1, 257.2 | 105.4, 276.0 | 72.3, 185.3 | 110.7, 299.4 | 142.2, 371.2 |
| 130 | 111.1, 294.0 | 100.5, 252.2 | 105.2, 270.3 | 71.4, 185.9 | 111.1, 295.3 | 145.9, 373.2 |
| 131 | 107.8, 293.3 | 97.2, 249.0 | 101.9, 268.4 | 69.2, 181.8 | 107.8, 293.3 | 142.2, 367.6 |
| 132 | 108.2, 289.5 | 96.6, 244.2 | 101.7, 263.3 | 68.3, 182.9 | 108.3, 289.6 | 146.0, 368.8 |
| 134 | 103.7, 283.8 | 91.5, 239.4 | 96.7, 258.4 | 64.8, 179.5 | 103.7, 284.0 | 143.6, 358.3 |
| 136 | 101.7, 279.0 | 89.6, 231.0 | 94.8, 251.1 | 64.2, 173.2 | 101.7, 278.6 | 140.5, 352.4 |

Depths of the minima relative to F²(0): ≤ 10⁻⁵ (true zeros) for all but the xenophobic case (10⁻³–10⁻² : partial cancellation only) and ¹³¹Xe (10⁻⁵). The isoscalar spread 251–278 keV reproduces P017 (251–278); proton-only 279–301 keV brackets P011's 284–291 keV.

Natural-xenon minima of the 1 TeV rate (response-only minima in parentheses): isoscalar 102.0 / 266.8 keV (102.5 / 268.7); proton-only 108.1 / 292.0 (108.4 / 293.6); **isovector 68.7 / 181.6 (69.3 / 182.1)**; xenophobic r = −0.7: 143.7 / none below 300 (143.1 / 367.5); neutron-only 97.2 / 247.4; Higgsino (r = −12.76) 96.5 / 244.7. Node position vs r (response): node₂ = 226.0, 213.9, 182.1, 156.9, 373 (r = −0.74), 367.5 (−0.70), 339.1 (−0.5), 293.6 (0), 276.5 (0.5), 268.7 (1.0) keV; node₁ = 91.1, 86.8, 69.3, —, 148.5, 143.1, 123.7, 108.4, 104.3, 102.5 keV for the same r. The second node crosses the event energy (248 keV) at r ≈ −5 … −∞ (neutron-heavy couplings) and moves out of the ROI (> 340 keV) for −0.8 < r < −0.6.

**Ratio to the isoscalar rate at the same c_p** (ρ; 1 TeV, mass dependence < 2 %):

| coupling | ρ(q→0) = ((Z + r⟨N⟩)/A)² | ρ(5.4–55 keV) | ρ(200–270 keV) | ρ(248 keV) |
|---|---|---|---|---|
| proton-only r = 0 | 0.169 | 0.182 | 0.404 | 0.929 |
| xenophobic r = −0.7 | 1.7 × 10⁻⁶ (1.28 × 10⁻⁴ isotope-resolved) | 1.09 × 10⁻³ | 0.176 | 0.957 |
| isovector r = −1 | 0.0317 | 0.0219 | 0.127 | 0.988 |
| neutron-only (per c_n) | 0.347 | 0.329 | 0.160 | 0.065 |
| Higgsino c¹/c⁰ = −1.17 (per c_n) | 0.310 | 0.291 | 0.128 | 0.070 |

The isovector rate at 248 keV is 0.99 of the isoscalar at equal c_p (0.032 at q → 0): the isoscalar sits 19 keV below its node at 267 keV (P017), the isovector at the maximum of its lobe between 182 keV and ~300 keV. This 31-fold relative enhancement of the high-energy rate is the entire explanation of the Table S6/S7 asymmetry.

Isotope shares of dR/dE at 248 keV (1 TeV): isoscalar ¹²⁹Xe 49.0 %, ¹³¹Xe 23.5 %, ¹³²Xe 15.9 %, ¹³⁰Xe 4.8 %, ¹³⁴Xe 3.6 %, ¹²⁸Xe 3.3 %, ¹³⁶Xe −0.1 % (on its node at 251 keV); isovector ¹³²Xe 27.3 %, ¹³¹Xe 20.8 %, ¹²⁹Xe 18.6 %, ¹³⁶Xe 15.3 %, ¹³⁴Xe 14.0 %, ¹³⁰Xe 3.1 %, ¹²⁸Xe 1.0 %; proton-only 34.4/23.2/22.6/8.5/5.1 % (129/131/132/134/136); xenophobic 23.1/21.7/26.2/12.4/11.9 %. ¹³⁴Xe + ¹³⁶Xe carry 29 % of the isovector rate at 248 keV against 3.5 % of the isoscalar: their isovector strength at q → 0 is (N − Z)² = 26², 28² vs 21² for ¹²⁹Xe (×1.5–1.8 per nucleus relative to the A² ratio 1.08–1.11), and their isovector nodes (173–180 keV) are furthest below the event. At 20 keV the shares are nearly coupling-independent (isoscalar/isovector ¹²⁹Xe 25.9/22.0 %, ¹³²Xe 27.1/28.2 %, ¹³⁶Xe 9.2/11.9 %) except for the xenophobic case (¹²⁹Xe 45.6 %, ¹³⁶Xe 0.6 %, ¹²⁸Xe 4.0 %): the Z + rN cancellation is isotope-specific (Z − 0.7N = +2.2, +1.5, +0.8, +0.1, −0.6, −2.0, −3.4 for A = 128–136).

### 3.2 Scan in r = f_n/f_p (Fig. 2 left; `P032_r_scan.csv`, 201 values of r × 3 masses)

| r | N_lo 400 / 1000 / 4000 GeV | N_{55–200} (1 TeV) | node₁ / node₂ [keV] (1 TeV) |
|---|---|---|---|
| +1 isoscalar | 3851 / 2753 / 2385 | 92.2 | 102.0 / 266.8 |
| 0 proton-only | 1754 / 1245 / 1075 | 53.9 | 108.1 / 292.0 |
| −0.7 xenophobic | 24.0 / 17.0 / 14.7 | 11.7 | 143.7 / > 300 |
| −0.744 (N_lo minimum) | 10.35 / 7.28 / 6.27 | — | 148 / 373 |
| −1 isovector | 703 / 477 / 404 | 8.5 | 68.7 / 181.6 |
| −2 | 12 400 / 8 900 / 7 700 (read from Fig. 2) | — | 91 / 226 |
| ∞ neutron-only | 7905 / 5661 / 4909 | 156 | 97.2 / 247.4 |
| −12.76 Higgsino | 8760 / 6265 / 5429 | 168 | 96.5 / 244.7 |

P003's 1 TeV values (2752, 477) are reproduced (2753, 477.1). The minimum of N_lo(r) is at r_min = −0.7429 / −0.7435 / −0.7437 (400/1000/4000 GeV), **not** at −Z/⟨N⟩ = −0.698: the low-energy cancellation is limited by the isotope spread and by f_n ≠ f_p already at 20 keV (ρ(5.4–55) = 1.1 × 10⁻³ at r = −0.7 versus 1.7 × 10⁻⁶ for the mean-N estimate and 1.3 × 10⁻⁴ for the isotope-resolved q → 0 estimate Σ_i ab_i(Z + rN_i)²/Σ_i ab_i A_i²), and the optimum trades low-energy suppression against the high-energy rate. **N_lo ≤ 5 is reached for no r; N_lo ≤ 10 only for r ∈ [−0.76, −0.74] (1 TeV) and [−0.76, −0.72] (4 TeV), never at 400 GeV.** The best case still predicts 6–10 events in 5.4–55 keV per event in 200–270 keV. Compared with P003's recalled 2024 tolerance N_max = 3–5, isospin violation gets within a factor 1.3–3.5 of compatibility but not inside it; P016's actual likelihood tolerance (see 3.3) is looser.

Cross-section needed for one 200–270 keV event in 2.84 t yr, σ_p = c_p² μ_p²(ħc)²/π: isoscalar (σ_p = σ_n) 3.2 / 5.6 / 19 × 10⁻⁴⁴ cm² (400/1000/4000 GeV); proton-only 0.80 / 1.4 / 4.7 × 10⁻⁴³; xenophobic (r = −0.7) 1.9 / 3.2 / 11 × 10⁻⁴³; isovector 2.7 / 4.4 / 15 × 10⁻⁴³ cm². At r = −0.7 other targets are suppressed relative to A² by (Z + rN)²/A² = 0.011 (F), 0.0042 (Ar), 0.0020 (Ge), 8.9 × 10⁻⁵ (I), 2.7 × 10⁻⁴ (W) [at r_min = −0.7435: 0.0068, 0.0017, 4.3 × 10⁻⁴, 2.5 × 10⁻⁴, 1.8 × 10⁻³] — "xenophobic" DM is at least as germanium- and argon-phobic (pure arithmetic; `P032_other_targets_q0.csv`).

### 3.3 Local significance (Fig. 2 right; `P032_local_Z.csv`, `P032_implied_Nlo.csv`)

| r | N_lo (1 TeV) | Z 400 / 1000 / 4000 GeV | LZ Table S7 O1ᵛ or O1ˢ | LZ Table S6 L1 / L5 | ŝ, low-E events (1 TeV) |
|---|---|---|---|---|---|
| +1 | 2753 | 0.19 / 0.48 / 0.59 | 0.0 / 0.0 / 0.0 | 0 / 0 | 3.7 × 10⁻⁴, 1.01 |
| 0 | 1245 | 0.82 / 1.06 / 1.16 | — | — | 1.4 × 10⁻³, 1.77 |
| −0.7 | 17.0 | 2.72 / 2.82 / 2.87 | — | — | 0.080, 1.35 |
| −1 | 477 | 1.44 / 1.65 / 1.74 | 0.8 / 1.3 / 1.1 | 0.3, 0.7 / 1.3, 1.3 / 1.2, 1.3 | 4.7 × 10⁻³, 2.23 |

The pattern (isoscalar ≈ 0, isovector ≈ 1–1.7σ, rising with mass) is reproduced; the 1 TeV values match P016 (0.48, 1.65). Our isovector Z exceeds LZ's by 0.35 (1 TeV), 0.5–0.6 (4 TeV) and 0.6–1.1σ (400 GeV, where LZ's own three entries for the identical spectrum scatter between 0.3 and 0.8σ — an intrinsic scatter of ≥ 0.5σ in the toy-calibrated tables at this level). In our likelihood LZ's isovector significances correspond to N_lo = 878 (1 TeV, all three tables), 1850/3450/2110 (400 GeV) and 1200/1030/876 (4 TeV), i.e. 1.8× (1 TeV), 2.2–3× (4 TeV), 2.6–4.9× (400 GeV) WimPyDD's stock isovector N_lo. Equivalently, LZ's isovector spectrum has roughly half the accepted 200–270 keV rate per low-energy event that WimPyDD's stock response gives. Sensitivity: b_H × 0.5 / × 2 changes the isovector 1 TeV Z from 1.65 to 2.00 / 1.25 — the anchor alone could account for the offset if LZ's effective background density at the event were ≈ 2× P016's anchor, but that would also lower every other operator (P016 reproduces L10/O6/O10/O4 to 0.2σ), so a spectral cause is more likely.

Possible spectral causes (discussion, not computed): (i) LZ uses "one-body nuclear density matrices developed for DMFormFactor-v6 with modifications as described in LZ 2023" (tex l. 59); any change of the neutron vs proton radial distributions moves the isovector node — the isovector amplitude at 248 keV is a small difference of large terms (f_n/f_p ranges from +0.17 (¹²⁹Xe) to −0.53 (¹³⁶Xe) there), so a few-per-cent change of f_n − f_p can halve W¹¹ near the event while leaving W⁰⁰ almost unchanged; (ii) two-body (meson-exchange) currents, absent from WimPyDD, add a predominantly isoscalar piece of order few–30 % of the one-body M response (Hoferichter et al. 2016; recalled, likely), which matters most where the one-body isovector amplitude is small; (iii) LZ's 2D (S1c, log S2c) PDFs with the NR-band resolution versus our 1D true-energy proxy; (iv) the ≥ 0.5σ scatter among LZ's own identical-spectrum entries. We flag the isovector shape near the node as nuclear-structure fragile and the O1ᵛ significances (elastic and inelastic) as correspondingly uncertain at the ±0.5σ level.

### 3.4 Inelastic δ = 300 keV (Fig. 4; `P032_inelastic_delta300.csv`)
Percentile of 248 keV in the efficiency-weighted true-energy spectrum (400 / 1000 / 4000 GeV), accepted-spectrum peak and f(248)/f_peak at 1 TeV:

| r | percentile | peak [keV] | f(248)/f_peak | node₂ [keV] |
|---|---|---|---|---|
| +1 isoscalar | 99.5 / 99.5 / 99.5 % | 168 | 0.042 | 265.6 |
| 0 proton-only | 97.7 / 97.6 / 97.5 % | 179 | 0.175 | 291.1 |
| −0.7 xenophobic | 81.2 / 79.2 / 79.0 % | 231 | 0.91 | > 300 |
| −1 isovector | 66.9 / 70.0 / 71.0 % | 252 | 0.99 | 180.9 |
| −12.76 Higgsino | 98.0 / 98.4 / 98.4 % | 160 | 0.014 | 243.4 |
| ∞ neutron-only | 98.5 / 98.7 / 98.8 % | 160 | 0.011 | 246.0 |

The isoscalar 99.5 % reproduces P002 (99.5th percentile at 1 TeV). The form factor depends on q only, so the nodes are the same as in the elastic case; the kinematic onset is 114 / 92 / 84 keV. For pure isovector O1 the accepted spectrum *peaks at 252 keV*, i.e. at the event — which explains P021's flat O1ᵛ profile (3.55–3.68σ) and Table S7's uniformly higher O1ᵛ entries (0.8–1.7σ at δ ≤ 50 keV, 2.6 vs 0.8 at 100 keV, 3.4 vs 3.0 at 300 keV). The Higgsino, although "isovector-dominated" in the (c⁰, c¹) basis, is neutron-only-like in shape (r = −12.8): its node (245 keV) sits on the event, and 248 keV is at the 98.4th percentile with f/f_peak = 0.014 — worse than the isoscalar. Total accepted δ = 300 keV rate per c_p² (1 TeV): isoscalar 1.63 × 10¹³, proton-only 4.16 × 10¹², xenophobic 8.8 × 10¹¹, isovector 7.8 × 10¹¹ (ratio to isoscalar 0.048 vs 0.032 at q → 0).

### 3.5 LZ's "×3.2" vector conversion
LZ converts O1ˢ cross-section limits to the vector (Higgsino-like) coupling by (A/((A−Z) − (1 − 4 sin²θ_W)Z))² ≃ 3.2 (tex l. 808), a q → 0 statement. With the shell-model responses the isoscalar-equivalent/vector rate ratio is 1/ρ = 3.23 (q → 0; we reproduce 3.2 with sin²θ_W = 0.2312, recalled, certain), 3.43 in 5.4–55 keV, **7.8 in 200–270 keV and 14.2 at 248 keV** (neutron-only: 2.88, 3.04, 6.26, 15.4). Recast vector limits from a 248 keV event are therefore too strong by ×2.4 (window) to ×4.4 (event energy). P007's and P021's Higgsino counts used the actual couplings inside WimPyDD and are unaffected; only the recast of LZ's *isoscalar* intervals is.

## 4. Robustness and checks
- Bilinearity of Eq. (1): 3.2 × 10⁻¹³.
- Normalisation vs Helm/lzcommon: 0.974 (20 keV), 0.998 (50 keV).
- P003 N_lo reproduced (2753 vs 2752; 477.1 vs 477); P016 Z reproduced (0.476 vs 0.48; 1.65 vs 1.65); P017 nodes (102.0/266.8 vs 102.0/266.9; isotope spread 251–278); P002 percentile (99.5 %); P011 proton node (292 vs 284–291 keV, different δ/mass); LZ factor 3.2 (3.23).
- Dyad check: W⁰¹²/(W⁰⁰W¹¹) = 1.002, 0.994, 0.997, 0.950, 1.055, 1.006 for A = 128, 129, 130, 132, 134, 136 (0.44 for ¹³¹Xe); F_pF_n vs W_pn ≤ 3.8 × 10⁻⁵.
- Mass dependence of ρ: < 2 % between 400 and 4000 GeV; N_lo falls 1.6× from 400 to 4000 GeV for all r.
- b_H × 0.5 / × 2: isovector Z 2.00 / 1.25 (1 TeV).
- Node finder: depth criterion 0.6 at ±15 keV; the isovector minimum at 182 keV is ×10⁻³ deep, the xenophobic first minimum at 144 keV ×0.05 — both robust.
- Ripples of ~1 % in the WimPyDD spectra (v_min grid) are visible in Fig. 4 near the isoscalar peak; they do not affect window integrals.

## 5. Failed or abandoned approaches
- First attempt to locate per-isotope nodes as zeros of signed amplitudes obtained from √W with sign flips at "touching" minima: double-detections at shallow minima and the non-dyad ¹³¹Xe response gave spurious/missing nodes (e.g. no ~100 keV isoscalar node for ¹³¹Xe). Replaced by minima of the exact quadratic form plus an exact dyad-based sign assignment for J ≤ ½ isotopes.
- Second attempt using common zeros of the signed products Z W_pp + rN W_pn and Z W_pn + rN W_nn: exact for dyads but degenerate at r = 0 (Z F_p² has no sign change) and invalid for ¹³¹Xe; abandoned for the minima definition.
- A ×0.3 / ±12 keV depth criterion for minima rejected the genuine isoscalar node at 267 keV (the rate 12 keV above it is only 2.5× larger); loosened to ×0.6 / ±15 keV.

## 6. Figures
- `figures/P032_fig1_spectra_vs_r.png` — Left: O1 elastic spectra at 1 TeV for r = +1, 0, −0.7, −1, −2 at fixed c_p; right: ratio to the isoscalar spectrum. The isovector minima at 69 and 182 keV and the xenophobic minimum at 144 keV are visible; all curves cross near 248 keV at ratio ≈ 1.
- `figures/P032_fig2_Nlo_Z_vs_r.png` — Left: N_lo(r) for 400/1000/4000 GeV with the N_lo = 3 and 5 lines and r = −Z/⟨N⟩; right: local Z(r) from the P016 few-bin likelihood with LZ's Table S7 points at r = ±1.
- `figures/P032_fig3_nodes.png` — Left: positions of the first and second minima of the natural-Xe M response vs r (response-only curve, rate-based markers at 1 TeV) and the second-node position of each isotope; right: signed shell-model f_p (solid) and f_n (dashed) for ¹²⁹Xe, ¹³²Xe, ¹³⁶Xe.
- `figures/P032_fig4_inelastic_delta300.png` — Normalised accepted δ = 300 keV spectra at 1 TeV for r = +1, 0, −1 with the event energy marked; the isovector spectrum peaks at 252 keV.

## 7. Result tables (all in `output/work/P032/`)
`P032_r_scan.csv` (r grid × mass: window rates, N_lo, N_L2, N_M, ρ, nodes), `P032_isotope_nodes.csv`, `P032_isotope_shares.csv`, `P032_node_vs_r_response.csv`, `P032_local_Z.csv`, `P032_implied_Nlo.csv`, `P032_inelastic_delta300.csv`, `P032_sigma_p_one_event.csv`, `P032_other_targets_q0.csv`, `P032_summary.json`, `P032_basis_spectra.npz`, `run_log.txt`.

## 8. Extended discussion
1. *What LZ's 1.3σ means.* By P016's mechanism, Z depends on the spectrum only through ln N_lo; the isovector coupling gains ~1.2σ over the isoscalar because the shell-model isovector node sits 65 keV below the event while the isoscalar node sits 19 keV above it. No coupling in the (c_p, c_n) plane gives a lone high-energy recoil: the best case (r ≈ −0.74) still implies 6–10 low-energy events per high-energy event and a local significance of at most 2.9σ in our likelihood, i.e. O4-like (P003/P016: O4 N_lo = 28, 2.7σ) — and O4-like interactions were already judged to carry no evidential weight (P016 Bayes factors 1.0–1.6).
2. *Fragility.* The isovector shape at q ≈ 250 MeV depends on f_n − f_p, which the one-body shell model with a common oscillator parameter fixes only schematically; the neutron-skin thickness of xenon (~0.15 fm, recalled, uncertain) is not an input of DMFormFactor-v6. LZ's modified density matrices and two-body currents can plausibly move the isovector node by tens of keV and are the natural explanation of the 0.3–0.6σ (1.8× in N_lo) offset between WimPyDD-based reproductions (P016, P021, this work) and LZ's tables. A data release of LZ's own O1ᵛ spectrum would settle this (no data request filed: the headline does not depend on it).
3. *For inelastic model builders.* Pure isovector O1 is the one O1 coupling for which 248 keV is a typical recoil at δ = 300 keV (70th percentile, at the spectral peak); proton-only (dark photon, P011) puts it at the 97.6th percentile and the Higgsino at the 98.4th — the Higgsino's spectral shape is neutron-like and should be compared with O1ˢ, not O1ᵛ, entries of Table S7. Vector-coupling recasts of LZ's isoscalar limits using ×3.2 overstate the constraint by ×2.4–4.4 at the event energy.

## 9. References
LZ Collaboration, arXiv:2609.02823 (2026). A. L. Fitzpatrick, W. Haxton, E. Katz, N. Lubbers, Y. Xu, JCAP 02 (2013) 004. N. Anand, A. L. Fitzpatrick, W. C. Haxton, Phys. Rev. C 89, 065501 (2014). J. L. Feng, J. Kumar, D. Marfatia, D. Sanford, Phys. Lett. B 703, 124 (2011). I. Jeong, S. Kang, S. Scopel, G. Tomar, Comput. Phys. Commun. 276, 108342 (2022). M. Hoferichter, P. Klos, J. Menéndez, A. Schwenk, Phys. Rev. D 94, 063505 (2016). D. Baxter et al., Eur. Phys. J. C 81, 907 (2021). Corpus: P002, P003, P007, P011, P016, P017, P021.

## 10. Tools and provenance (mirrors `output/provenance/P032.json`)
- Agent tools: Read (PAPER_GUIDE; dossier; ledger ×2; P003/P011/P017/P021/P007/P016 papers; fulltext.tex l. 44–61, 462–501, 798–909; lzcommon l. 74–111, 300–372; P016 script l. 96–120, 120–270, 268–288; P003 script l. 28–117; P017 script l. 25–134; P016_results.json; figures ×7), Bash (grep/ls of tex, lzcommon, scripts, environment; WimPyDD `diff_rate` signature and `nuclear_current`; P021 provenance layout; dataviz palette; 5 script runs; 1 diagnostic on ¹³¹Xe dyad structure; table prints; word counts), Write (script, details, provenance, paper), Edit (~12 script fixes), Skill (dataviz).
- Software: python 3.12.13; WimPyDD 2.0.4 (`streamed_halo_function` via lzcommon, `eft_hamiltonian`, `diff_rate` incl. `isotopes_list`, `Xe.func_w`, `nuclear_current`); numpy 2.5.3; scipy 1.18.1 (special.erf, integrate.trapezoid/cumulative_trapezoid, optimize.minimize_scalar/brentq, stats.norm); pandas 3.0.5; matplotlib 3.11.2; common/lzcommon.py (wd_halo, wd_hamiltonian, wd_rate, dRdE_SI, LSIG, OSIG, XE_ISOTOPES, constants).
- Recalled knowledge: (ħc)² = 0.3894 × 10⁻²⁷ GeV² cm² (certain); sin²θ_W = 0.2312 (certain); xenophobic f_n/f_p ≈ −0.7 and its origin (Feng et al. 2011; certain); two-body-current size few–30 % of one-body SI response and predominantly isoscalar (likely); xenon neutron-skin ~0.15 fm (uncertain, discussion only); Wilks Z = √q₀ (certain); M-response multipole selection rules (M_J with even J ≤ 2J_nuc) (certain); L1/L5 ↔ O1 mapping (likely, P003; LZ states L1 and L5 differ by a constant).
- Datasets: none. Data requests: none. WimPyDD-generated files: none beyond `output/work/P032/P032_basis_spectra.npz` (our cache; WimPyDD `diff_rate` writes no response-function files).
