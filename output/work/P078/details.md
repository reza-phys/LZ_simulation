# P078 — research record
**Isospin violation and the isovector suppression factor for xenon at 250 keV: per-isotope amplitudes, the f_n/f_p scan and what it does to the high-energy/low-energy ratio**
Simulated date 2026-09-15 · nucl-th (cross-list hep-ph) · nuclear-theory group · category NUC · **competing with P032** (same question, independent method)

Script: `output/code/P078_isospin_violation.py` (run: `.venv/bin/python output/code/P078_isospin_violation.py`; `--recompute` rebuilds the per-isotope kernel cache `output/work/P078/kernels_O1_O4_1000gev.npz`, otherwise 15 s). Full console output: `output/work/P078/run_log.txt`. Every number below is from that run unless flagged.

## 1. Motivation and relation to P032
LZ (arXiv:2609.02823, Tables S6/S7) gives 0σ to the isoscalar O₁ (L1ˢ/L5ˢ/O1ˢ) but 0.8–1.3σ to the isovector O₁ at m ≥ 400 GeV. P032 explained this with WimPyDD's shell-model M response (isovector node at 182 keV vs isoscalar 267 keV) and scanned f_n/f_p, finding N_lo ≥ 7.3 (1 TeV, minimum at r = −0.744). We were asked to re-derive these results *independently* — per isotope, from the response coefficient functions rather than from Helm — and to extend them to (i) the amplitude and its nodes per isotope, (ii) R_hi/lo(r) for elastic and inelastic (δ = 250–380 keV) scattering, (iii) the xenophobic point at 248 keV, (iv) isotope-enriched/depleted targets, (v) N_lo(r) versus P032/P016/P030, and (vi) the spin-dependent O₄ with proton/neutron spin structure. We state at the end of each section where we agree or disagree with P032.

## 2. Framework
### 2.1 Couplings
O₁ (or O₄) with nucleon couplings c_p, c_n; r ≡ f_n/f_p = c_n/c_p. WimPyDD convention (P003, confirmed P007): c⁰ = c_p + c_n = c_p(1 + r), c¹ = c_p − c_n = c_p(1 − r). We work at c_p = 1 GeV⁻² (all ratios coupling independent); r = +1 isoscalar, 0 proton-only, −1 isovector, −0.7 "xenophobic" (Feng et al. 2011; recalled, certain), −Z/⟨N⟩ = −0.698 for natural xenon.

### 2.2 Per-isotope kernel factorisation (the method that differs from P032)
For the velocity-independent operators O₁ and O₄ WimPyDD's rate is exactly (P046/P062 method)

  dR/dE(E; r, m, δ, halo) = (1000 GeV/m) Σ_i [c⁰² K_i⁰⁰(E) + 2c⁰c¹ K_i⁰¹(E) + c¹² K_i¹¹(E)] η(v_min,i(E; m, δ)),   (1)

  v_min,i = |m_i E/μ_i + δ| / √(2 m_i E) × c   (WimPyDD `get_vmin`, c = 300 000 km/s, per-isotope nuclear mass m_i).

The kernels K_i^{ττ'} are obtained once per isotope (7 active isotopes; ¹²⁴Xe, ¹²⁶Xe carry no density matrices in WimPyDD, 0.18 % of natural Xe) from `WD.diff_rate(..., isotopes_list={0:[i]})` with a single 3000 km/s stream of unit δη (so η ≡ 1), for (c⁰, c¹) = (1,0), (0,1), (1,1) → K⁰⁰, K¹¹, K⁰¹ = [K(1,1) − K⁰⁰ − K¹¹]/2, on a 0.5 keV grid 0.5–400 keV (21 + 6 scans, 26 s). Any r, m, δ, halo *and any isotopic composition* is then a weighted sum; per-tonne rates for a composition with number fractions w_i use K_i × (w_i/ab_i) × (Σ ab_j m_j / Σ w_j m_j).

**Kernels are the response functions.** We verified K_i^{ττ'}(E)/W_{M,i}^{ττ'}(q(E)) = const_i to a relative spread of 3 × 10⁻¹⁶ (¹²⁹Xe: 4.815 × 10¹⁵; ¹³¹Xe 1.966 × 10¹⁵; ¹³⁶Xe 3.406 × 10¹⁵ in events/(t yr keV) per unit W; const_i ∝ ab_i/m_i). Nothing in the O₁ result therefore depends on anything but the DMFormFactor-v6 one-body density matrices and kinematics.

**Validation against direct WimPyDD calls** (natural Xe, full Hamiltonian at the given (c⁰,c¹)): 20 points — O₁ at (r, m, δ, halo) = (−0.7, 1000, 0, Sun-frame; E = 20/100/200/248 keV), (−1, 400, 0, annual; 10/150/248), (+1, 1000, 300, Sun-frame; 150/248/280), (−0.744, 4000, 350, annual; 200/248/300), (0.3, 1000, 250, annual; 120/248), O₄ at (−1, 1000, 0, Sun-frame; 20/100/248) and (0.5, 1000, 300, annual; 200/248): **max |factorised/direct − 1| = 9.2 × 10⁻¹⁵** (`validation_factorised_vs_direct.csv`). The 1/m scaling of the kernel was verified by P062 (exact) and is implied by the 400/4000 GeV validation points.

### 2.3 Halos, efficiency, windows
Baxter-2021 SHM through `lz.wd_halo` on an explicit 1200-point v_min grid to 844 km/s (P002 fix). Two halos: WimPyDD's single-day **Sun-frame** halo (`wd_halo()`, v_E = 250.6 km/s, v_max = 795.4 km/s; this is what P003/P016/P032 used and labelled "annual average", see P035) and a **12-day annual mean** (v_max 810.9 km/s). Elastic results differ by < 1 % between them; inelastic results at δ ≥ 350 keV do not, and we quote the annual mean as primary there. Efficiency: P003/P016/P032 model (0.96 plateau, erf edges at 5.4 keV with σ = 3.4 keV and 269.9 keV with σ = 8 keV) so that N_lo is directly comparable with P003/P016/P032. Windows (efficiency-weighted, trapezoid on the 0.5 keV grid): R_lo = R(5.4–55 keV), R_mid = R(55–200), R_hi = R(200–270). N_lo = R_lo/R_hi, N_mid = R_mid/R_hi, R_hi/lo = 1/N_lo. ρ(X) = rate(r)/rate(r = +1) at equal c_p. Percentile of 248 keV = fraction of the accepted 1–300 keV spectrum below 248 keV.

### 2.4 Response functions and signed amplitudes
For O₁ only the M response enters: W_M^{ττ'}(q) = `Xe.func_w[i](q)[nuclear_current['M'], τ, τ']`. Proton/neutron pieces W_pp = (W⁰⁰ + 2W⁰¹ + W¹¹)/4, W_nn = (W⁰⁰ − 2W⁰¹ + W¹¹)/4, W_pn = (W⁰⁰ − W¹¹)/4. **q → 0 check:** √(W_nn/W_pp) = 1.3696/1.3879/1.4064/1.4249/1.4434/1.4801/1.5172 for A = 128…136 against N/Z = 1.3704/1.3889/1.4074/1.4259/1.4444/1.4815/1.5185 (agreement 0.05–0.09 %: the M operator counts protons and neutrons, so F_p(0) = Z, F_n(0) = N up to a common normalisation; W⁰⁰(0)/A² = 0.0197 × (2J+1)). For J ≤ ½ (all but ¹³¹Xe) a single M₀ multipole contributes (selection rule: M_J with even J ≤ 2J_nuc; recalled, certain), so W^{ττ'} = F_τ F_τ' is a dyad; pointwise |W_pn²/(W_pp W_nn) − 1| ≤ 0.024 (5–200 keV, > 8 keV from the zeros; larger values only where the tabulated responses reach their ~10⁻⁶ noise floor). ¹³¹Xe (J = 3/2) has M₂ too; its "zeros" are minima (P032: depth 10⁻⁵) and we treat it with the quadratic form only.

Signed amplitudes for the six dyad isotopes: F_p = s_p √W_pp, F_n = s_n √W_nn, with sign flips at the zeros of W_pn = F_p F_n (sign changes of W_pn are exact zeros of F_p or F_n; the vanishing factor is the one with the smaller W_tt/W_tt(0), both being ~10⁻⁶ there). Consistency |F_p F_n − W_pn|/√(W_pp W_nn) ≤ 1.2 × 10⁻² (≤ 4 × 10⁻³ for A ≥ 129). Normalised f_τ = F_τ/F_τ(0). The coupling amplitude is A_r(q) = Z f_p + r N f_n and the quadratic form (1 + r)² W⁰⁰ + 2(1 − r²) W⁰¹ + (1 − r)² W¹¹ = 4 A_r² × norm; node = zero of A_r (dyads) or local minimum of the quadratic form (¹³¹Xe, natural Xe: minimum at least ×0.6 deeper than 15 keV on either side, parabolic refinement — P017/P032 definition).

For O₄ the response is Σ' + Σ'' (¹²⁹Xe and ¹³¹Xe only; all other isotopes have zero spin response). With the same pp/nn/pn decomposition, W_pn(0)/W_nn(0) = ⟨S_p⟩/⟨S_n⟩ (signed).

## 3. Results
### 3.1 Per-isotope amplitudes and nodes (Fig. 1; `isotope_nodes.csv`, `node_vs_r.csv`)
Zeros of F_p and F_n (keV; ¹³¹Xe: minima of W_pp, W_nn) and zeros of A_r for the key couplings:

| A | F_p zeros | F_n zeros | isoscalar r=+1 | isovector r=−1 | proton-only | xenophobic r=−0.7 (depth) |
|---|---|---|---|---|---|---|
| 128 | 114.5, 302.2 | 104.6, 258.8 | 109.1, 277.6 | 74.8, 190.1 | 114.4, 301.3 | 146.8, 380.3 (1.2e-3, 8e-4) |
| 129 | 110.7, 299.4 | 101.1, 257.2 | 105.4, 276.0 | 72.3, 185.3 | 110.7, 299.4 | 142.5, 371.6 (1.0e-3, 3e-4) |
| 130 | 111.1, 294.0 | 100.5, 252.2 | 105.2, 270.3 | 71.4, 185.9 | 111.1, 295.3 | 146.1, 373.5 (3e-3, 5e-4) |
| 131 | 107.4, 293.2 | 97.5, 249.1 | 101.9, 268.4 | 69.2, 181.8 | 107.8, 293.3 | 142.5, 367.9 (0.15, 0.21) |
| 132 | 108.2, 289.5 | 96.6, 244.2 | 101.7, 263.3 | 68.3, 182.9 | 108.3, 289.6 | 146.2, 369.1 (9e-3, 3e-3) |
| 134 | 103.7, 283.8 | 91.5, 239.4 | 96.7, 258.4 | 64.8, 179.5 | 103.7, 284.0 | 143.9, 358.6 (9e-4, 1e-4) |
| 136 | 101.7, 279.0 | 89.6, 231.0 | 94.8, 251.1 | 64.2, 173.2 | 101.7, 278.6 | 140.7, 352.7 (5e-4, 2e-4) |

(depth = F²_r(node)/F²_r(0); for the isoscalar, isovector and proton-only cases the minima are true zeros, depth ≤ 10⁻⁵; the xenophobic minima are partial cancellations only.) **Agreement with P032's table: every entry within 0.1–0.4 keV** (e.g. ¹²⁹Xe isovector 185.3 vs 185.3, ¹³⁶Xe isoscalar 251.1 vs 251.1) — expected, since both use the same WimPyDD responses; the independent sign construction confirms P032's node assignment.

Natural xenon (abundance/m_i weighted response): isoscalar 102.5/268.8 keV, isovector 69.3/182.2, proton-only 108.4/293.6, xenophobic 143.1/367.6; minima of the 1 TeV *rate*: 102.0/266.9, 68.6/181.6, 108.1/291.9, 143.7/366.0 (P032: 102.0/266.8, 68.7/181.6, 108.1/292.0, 143.7/> 300 — agree). Node vs r (response): node₂ = 226.1 (r = −2), 214.0 (−1.5), 182.2 (−1), 170.5 (−0.9), 163.8 (−0.85), then jumps to 381.3 (−0.8), 374.6 (−0.75), 367.6 (−0.7), 339.2 (−0.5), 293.6 (0), 276.6 (0.5), 268.8 (1), 261.7 (2) keV; node₁ = 91.2, 86.9, 69.3, 58.2, 49.8, 156.9, 149.8, 143.1, 123.7, 108.4, 104.3, 102.5, 100.8 keV. **The second node lies inside the 200–270 keV window only for r ≤ −1.21 or r ≥ 0.89; it crosses 248 keV exactly at r = −0.826 (from above) — and P032's statement "r ≈ −5…−∞" refers to the neutron-heavy branch beyond our scan; our crossing at −0.826 is where the node collapses from the high branch, so at r slightly below −0.83 the event sits on a node too.** For −0.82 < r < 0.89 the second node is above 270 keV (or absent below 400 keV).

**Helm versus shell model** (`lz.helm_F2`, A = 131): minima at 94.6 and 279.5 keV against the shell-model isoscalar 102.5/268.8 keV (natural) — Helm misplaces the second node by 11 keV and, having one form factor for protons and neutrons, has *no* isospin dependence at all: everything in this paper is invisible to a Helm calculation.

**Why 248 keV is special (per isotope).** f_n/f_p (signed) at 20/100/200/248 keV: ¹²⁸Xe 0.954/0.295/0.670/0.219; ¹²⁹Xe 0.949/0.092/0.629/0.166; ¹³⁰Xe 0.947/0.044/0.615/0.136; ¹³²Xe 0.940/−0.373/0.562/−0.100; ¹³⁴Xe 0.932/−2.09/0.513/−0.219; ¹³⁶Xe 0.927/−5.54/0.428/−0.502. The second zeros of F_n span 231–259 keV across the isotopes, i.e. **248 keV sits inside the band of second neutron-form-factor zeros**, 31–54 keV below the second proton zeros (279–302 keV). At the event energy xenon is therefore nearly *neutron-blind*: |f_n/f_p| = 0.10–0.50 with F_n changing sign between ¹³¹Xe and ¹³²Xe, so the 248 keV amplitude is proton-dominated for every r ∈ [−2, 2] (ρ(248) = 0.93, 0.96, 0.99, 1.18, 1.20 for r = 0, −0.7, −1, −2, +2; Fig. 2 middle). The isoscalar amplitude Z f_p + N f_n vanishes 19 keV above the event because there the small negative F_p is cancelled by the small positive F_n of the lighter isotopes; the isovector Z f_p − N f_n cannot vanish there (both terms have the same sign for A ≤ 131) — its node is at 182 keV where f_n/f_p = Z²/N² ≈ 0.5.

### 3.2 The f_n/f_p scan, elastic (Fig. 2; `r_scan.csv`, `key_points_1TeV.csv`)
1 TeV, Sun-frame halo (annual mean in parentheses where different):

| r | R_hi/lo | N_lo | N_mid | ρ(5.4–55) | ρ(200–270) | ρ(248) | nodes (rate) |
|---|---|---|---|---|---|---|---|
| +1 isoscalar | 3.65 × 10⁻⁴ | 2743 (2726) | 92.2 | 1 | 1 | 1 | 102.0/266.9 |
| 0 proton-only | 8.06 × 10⁻⁴ | 1240 (1232) | 53.9 | 0.183 | 0.404 | 0.929 | 108.1/291.9 |
| −0.7 xenophobic | 5.89 × 10⁻² | 16.97 (16.87) | 11.7 | 1.09 × 10⁻³ | 0.176 | 0.957 | 143.7/366.0 |
| −0.7436 (N_lo minimum) | 0.138 | **7.245 (7.199)** | 9.78 | 4.41 × 10⁻⁴ | 0.167 | 0.961 | 149.5/372.2 |
| −1 isovector | 2.11 × 10⁻³ | 474.6 (471.3) | 8.49 | 2.19 × 10⁻² | 0.127 | 0.989 | 68.6/181.6 |
| −2 | 1.19 × 10⁻⁴ | 8412 | 178 | 0.518 | 0.169 | 1.178 | 90.4/224.3 |
| +2 | 2.82 × 10⁻⁴ | 3542 | 111 | 2.47 | 1.916 | 1.201 | 100.2/259.6 |

Mass dependence (Sun-frame): N_lo(400/1000/4000 GeV) = 3837/2743/2376 (iso), 1748/1240/1071 (p-only), 23.97/16.97/14.65 (r = −0.7), 699/474.6/402.3 (vec), 12 330/8412/7145 (r = −2); **minimum 10.29/7.245/6.236 at r = −0.7431/−0.7436/−0.7439**. N_lo ≤ 10 only for r ∈ [−0.76, −0.73] (1 TeV) and [−0.77, −0.72] (4 TeV), never at 400 GeV; N_lo ≤ 5 for no r.

**Comparison with P032** (3851/2753/2385, 1754/1245/1075, 24.0/17.0/14.7, 703/477/404; minimum 10.35/7.28/6.27 at −0.7429/−0.7435/−0.7437): our values are 0.3–0.5 % lower throughout (0.5 keV grid and slightly different window sampling), r_min agrees to 10⁻³. We **confirm P032 quantitatively**, with an independent rate construction (per-isotope kernels × η instead of three natural-Xe spectra and the bilinear form). The isovector/isoscalar ratio at equal c_p: 0.0317 (q → 0), 0.0219 (5.4–55), 0.1265 (200–270), 0.9885 (248 keV) — P032: 0.032/0.022/0.127/0.99. Against P003 (2752, 477) and P030 (SHM 2569 for O₁, halo/efficiency treatment differs by 7 %) our isoscalar N_lo = 2743 (Sun-frame) / 2726 (annual mean).

**Maximum achievable enhancement of the high-energy rate.** Two different questions: (a) at *fixed c_p*, ρ(200–270) never exceeds 1 for r ≤ 1 (0.127–0.404 for the interesting couplings) and reaches 1.92 only trivially at r = +2 (more coherent charge); isospin violation never *raises* the absolute high-energy rate for a given proton coupling — it removes low-energy rate. (b) The *shape* ratio R_hi/lo is enhanced by up to **×378 at r = −0.7436** (R_hi/lo = 0.138 vs 3.65 × 10⁻⁴), ≥ ×100 for r ∈ [−0.80, −0.69] and ≥ ×10 for r ∈ [−0.94, −0.49]. The cross-section needed for one 200–270 keV event in 2.84 t yr (1 TeV; σ_p = c_p² μ_p² (ħc)²/π, (ħc)² = 0.3894 × 10⁻²⁷ GeV² cm² recalled, certain): 5.57 × 10⁻⁴⁴ (iso), 1.38 × 10⁻⁴³ (p-only), 3.17 × 10⁻⁴³ (r = −0.7), 3.34 × 10⁻⁴³ (r_min), 4.41 × 10⁻⁴³ (vec), 3.30 × 10⁻⁴³ (r = −2), 2.91 × 10⁻⁴⁴ cm² (r = +2). P032: 5.6/14/32/44 × 10⁻⁴⁴ — agree.

### 3.3 The xenophobic point at 248 keV (`xenophobic_isotope_interference.csv`)
At r = −0.7 the q → 0 coherent factors Z − 0.7N are +2.2, +1.5, +0.8, +0.1, −0.6, −2.0, −3.4 (A = 128…136): the low-energy rate is suppressed to ρ(5.4–55) = 1.09 × 10⁻³ (mean-N estimate 1.7 × 10⁻⁶ — the isotope spread and f_n ≠ f_p already at 20 keV limit the cancellation; P032 found the same 1.1 × 10⁻³). At 248 keV, with F_p(0) = Z and F_n(0) = N normalisation (units of nucleons):

| A | F_p(248) | −0.7 F_n(248) | amplitude | interference | share at 248 keV | share at 20 keV |
|---|---|---|---|---|---|---|
| 128 | −0.730 | +0.153 | −0.577 | destructive | 1.3 % | 4.0 % |
| 129 | −0.791 | +0.128 | −0.664 | destructive | 23.1 % | 45.6 % |
| 130 | −0.688 | +0.092 | −0.596 | destructive | 3.4 % | 5.3 % |
| 131 | (non-dyad) | | | | 21.7 % | 21.8 % |
| 132 | −0.620 | −0.063 | −0.683 | constructive | 26.2 % | 19.2 % |
| 134 | −0.602 | −0.136 | −0.739 | constructive | 12.4 % | 3.5 % |
| 136 | −0.500 | −0.267 | −0.767 | constructive | 11.9 % | 0.6 % |

The interference is destructive for A ≤ 131 (F_n still positive before its second zero) and *constructive* for A ≥ 132 (F_n already negative), and in every case the neutron term is 10–35 % of the proton term. Hence **the suppression at high q is ×160 weaker than at low q** (ρ(200–270)/ρ(5.4–55) = 0.176/1.09 × 10⁻³ = 161) and absent at the event energy itself (ρ(248) = 0.957). Destructive interference therefore *helps* the single-event interpretation in the only way isospin can: it removes the 5.4–55 keV population (N_lo 2743 → 17, → 7.2 at r_min) at a moderate cost in absolute rate (σ_p × 5.7 relative to the isoscalar for one event, ×6.0 at r_min). But it cannot make the event stand alone: even at r_min there are 7.2 companions in 5.4–55 keV and 9.8 in 55–200 keV per 200–270 keV event; by P016's likelihood (Z < 3σ for N_lo > 10.9) the best xenophobic tuning sits at ≈ 3σ, O₄-class (P016: O₄ with N_lo = 28 gets 2.7σ; P032 finds 2.8σ at r = −0.7). We agree with P032's verdict that "no f_n/f_p makes elastic SI compatible with a lone event".

### 3.4 Inelastic scattering: R_hi/lo, N_mid and the 248 keV percentile vs r (Fig. 2 right; `r_scan.csv`)
For δ ≥ 250 keV the 5.4–55 keV window is kinematically empty (onset 52–54 keV at δ = 250, 84–92 at 300, 138–154 at 350, 193–228 keV at 380, annual/Sun-frame), so R_hi/lo → ∞ trivially and the relevant companion population is 55–200 keV. 1 TeV, annual-mean halo (Sun-frame in parentheses):

| δ [keV] | quantity | r = +1 | 0 | −0.7 | −1 | best r (value) |
|---|---|---|---|---|---|---|
| 250 | N_mid | 7.94 | 4.10 | 1.03 | 1.91 | −0.756 (0.99) |
| 250 | pct(248) | 99.8 % | 98.7 % | 87.5 % | 85.5 % | |
| 300 | N_mid | 4.33 | 2.19 | 0.327 | 0.494 | −0.818 (0.246) |
| 300 | pct(248) | 99.5 (99.5) | 97.6 (97.6) | 79.4 (79.2) | 70.7 (70.0) | −0.95 (70.3 %) |
| 300 | ρ(200–270) | 1 | 0.425 | 0.210 | 0.165 | max 1.89 at +2 |
| 350 | N_mid | 0.893 | 0.495 | 0.0996 | 0.0167 | −1.05 (0.015) |
| 350 | pct(248) | 97.8 (97.0) | 92.5 (90.7) | 68.1 (65.8) | 51.0 (49.7) | −1.41 (40 %) |
| 350 | ρ(200–270) | 1 | 0.478 | 0.298 | 0.267 | max 1.83 at +2 |
| 380 | pct(248) | 75.1 (7.6) | 69.1 (40.5) | 39.2 (15.6) | 30.5 (12.2) | |
| 380 | ρ(200–270) | 1 | 0.965 (1.34) | 1.445 (4.99) | 1.779 (7.42) | **3.44 at r = −2 (19.2 Sun-frame)** |

Accepted-spectrum peaks at δ = 300 keV: 168 (iso), 178 (p), 231 (r = −0.7), 252 keV (vec); at 350 keV the xenophobic spectrum peaks at 244 keV with f(248)/f_peak = 0.997 and the isovector at 255 keV; at δ = 380 all couplings peak at 228–268 keV. Reading: (i) at δ = 300 keV our Sun-frame percentiles reproduce P032's 99.5/97.6/79.2/70.0 % exactly and the annual mean changes them by ≤ 0.7 %; (ii) the "best" r for putting 248 keV at a typical position moves with δ (−0.95 at 300, −1.4 at 350) — the neutron-heavy side r < −1 becomes competitive at large δ because its node₂ (214–226 keV) is then just *below* a window whose onset has moved up; (iii) **only at δ = 380 keV does isospin violation enhance the absolute 200–270 keV rate at fixed c_p** (×1.8 isovector, ×3.4 at r = −2, annual mean), because the isoscalar sits on its own node (252 keV) inside the narrow kinematic window; this number is strongly halo dependent (Sun-frame: ×7.4/×19, onset 228 vs 193 keV) and should be read with P035's warning in mind. **Agreement with P032:** complete at δ = 300; P032 did not treat 250/350/380 keV.

### 3.5 Enriched and depleted targets (Fig. 3; `enriched_targets.csv`, `enriched_target_test.csv`)
Per tonne, 1 TeV, relative to natural xenon at the same coupling. Elastic (Sun-frame):

| target | ρ_hi = R_hi/R_hi,nat (r = +1 / 0 / −0.7 / −1) | dR/dE(248)/natural | N_lo |
|---|---|---|---|
| ¹³⁶Xe-depleted (natural without ¹³⁶Xe) | 1.041 / 1.021 / 0.964 / 0.916 | 1.10 / 1.05 / 0.97 / 0.93 | 2634 / 1221 / 18.4 / 502 |
| ¹³⁴+¹³⁶Xe-depleted | 1.069 / 1.031 / 0.935 / 0.858 | 1.20 / 1.08 / 0.94 / 0.88 | 2568 / 1214 / 19.9 / 522 |
| ¹³⁶Xe-enriched (90 % ¹³⁶, 10 % ¹³⁴; 0νββ-grade, recalled likely) | 0.614 / 0.810 / 1.342 / 1.781 | **0.023** / 0.58 / 1.28 / 1.63 | 4458 / 1457 / 7.0 / 344 |
| ¹²⁹Xe-enriched (80 % + natural rest) | 1.231 / 1.134 / 0.910 / 0.746 | 1.71 / 1.26 / 0.91 / 0.77 | 2228 / 1117 / 24.6 / 556 |
| pure ¹²⁹Xe | 1.290 / 1.168 / 0.887 / 0.682 | 1.89 / 1.33 / 0.89 / 0.72 | 2126 / 1090 / 26.8 / 587 |
| pure ¹³²Xe | 0.844 / 0.879 / 0.948 / 0.989 | 0.59 / 0.83 / 0.97 / 1.01 | 3258 / 1403 / 14.9 / 499 |
| pure ¹³⁶Xe | 0.590 / 0.796 / 1.360 / 1.828 | −0.01 (node) / 0.56 / 1.30 / 1.66 | 4642 / 1480 / 6.75 / 339 |

δ = 300 keV (annual): ¹³⁶Xe-enriched ρ_hi = 0.79/1.04/1.68/2.15, dR/dE(248) ratio 0.027/0.75/1.60/1.99, node₂ 251.8 keV (isoscalar); ¹²⁹Xe-enriched 1.11/1.01/0.80/0.65; ¹³⁶Xe-depleted 1.02/1.00/0.93/0.88. Percentiles of 248 keV change by ≤ 3 % for every composition.

Findings: (a) depleting ¹³⁶Xe (or ¹³⁴+¹³⁶Xe) changes the natural-Xe spectrum by ≤ 9 % in the window and ≤ 20 % at 248 keV — LZ's result is insensitive to the small isotopic variations of commercial xenon; (b) a **¹³⁶Xe-enriched** target is the one composition with a qualitative signature: the isoscalar amplitude of ¹³⁶Xe has its zero at 251 keV, so at r = +1 the rate *at* 248 keV drops to 2 % of natural while for r = −0.7…−1 it *rises* ×1.3–1.6; (c) even so, integrated over 200–270 keV the enriched/natural ratio only moves between 0.61 (iso) and 1.78 (vec). A two-target counting test (equal exposures, each 200–270 keV event assigned to "natural" or "enriched") needs **124 high-energy events in total for a 3σ separation of r = +1 from r = −1, 233 for +1 vs −0.7, 1884 for −0.7 vs −1** (pure ¹²⁹Xe: 352, 1027, 2106). The low-energy window of *any* xenon target separates the same hypotheses with one high-energy event (N_lo 2743 vs 475 vs 17). **Isotope engineering is therefore not a viable decisive test of the isospin structure of a single 248 keV event**; it becomes informative only in a > 100-event future (P050 regime), and the isoscalar-on-node signature at 248 keV in ¹³⁶Xe would then be a striking spectral marker.

### 3.6 Spin-dependent O₄ with proton/neutron spin structure (Fig. 4; `O4_spin_structure.json`, `O4_r_scan.csv`)
From Σ' + Σ'' at q → 0: ⟨S_p⟩/⟨S_n⟩ = +0.0292 (¹²⁹Xe) and +0.0257 (¹³¹Xe); Σ'' carries 33.5 %/33.9 % of the q → 0 response (the 1/3 : 2/3 longitudinal/transverse split, as expected); isoscalar/isovector response ratio (1 + S_p/S_n)²/(1 − S_p/S_n)² = 1.124/1.108 at q → 0, 0.984/1.091 at 100 keV, 1.257/1.058 at 248 keV. No node of Σ' + Σ'' below 420 keV for either isotope or any r. The spin-cancellation ratio r* = −S_p/S_n = −0.029 (¹²⁹Xe), −0.026 (¹³¹Xe).

O₄ elastic, 1 TeV (Sun-frame): N_lo = 28.0 (iso; P003/P016: 28), 29.3 (vec), 29.5 (r = −0.7), 8.0 (proton-only), 28.4/29.0 (r = ±2); ρ(200–270) = 0.861 (vec), 0.409 (−0.7), 0.003 (p-only), 3.85/3.57 (±2). **Minimum N_lo = 3.61 at r = −0.0248**, i.e. at the spin-cancellation point, with N_lo ≤ 5 for r ∈ [−0.034, −0.014] and ≤ 10 for [−0.048, +0.006] — but the price is ρ(200–270) = 0.0021 (×480 more coupling for the same high-energy rate; ρ(5.4–55) = 2.7 × 10⁻⁴). This "spin-xenophobic" O₄ is the spin analogue of the M-response xenophobic point: c_p S_p + c_n S_n cancels at q → 0 while the different q dependence of the proton and neutron Σ responses spoils it at high q. It is the only place in the (c_p, c_n) plane of O₁ and O₄ where N_lo < 5 is reached, at 2 % tuning of c_n/c_p. Away from r ≈ 0, N_lo varies only 28–30.5. Shapes: isoscalar vs isovector O₄ normalised spectra differ by ≤ 5.7 % pointwise (KS distance 0.005; ratio 1.04 at 248 keV) — **LZ's statement (tex l. 496) that the O₄ isoscalar and isovector spectra are indistinguishable is confirmed**; for O₁ the same comparison gives KS 0.11 and a pointwise ratio 0.023 at 248 keV. Inelastic O₄ (δ = 300, Sun-frame): N_mid 1.09 (iso), 1.13 (−0.7), 1.12 (vec), 2.33 (p-only); percentile 88.1 % (iso/vec), 96.6 % (p-only); minimum N_mid 1.09 at r = +0.64 — no isospin lever at all for inelastic O₄.

## 4. Robustness and checks
- Kernel/response proportionality 3 × 10⁻¹⁶; factorised vs direct WimPyDD 9.2 × 10⁻¹⁵ over 20 points including per-isotope kinematics at δ = 350 keV and 4 TeV.
- q → 0: √(W_nn/W_pp) = N/Z to 0.05–0.09 %; dyad relation pointwise ≤ 2.4 % away from zeros; signed-product consistency ≤ 1.2 %.
- Sun-frame vs annual mean: elastic N_lo 2743 vs 2726 (0.6 %); δ = 300 percentiles ≤ 0.7 %; δ = 380 ρ_hi 3.4 vs 19 — halo-fragile, flagged.
- Reproductions: P032 nodes (≤ 0.4 keV), N_lo (0.3–0.5 %), r_min (10⁻³), ρ ratios (< 1 %), σ_p (< 2 %), δ = 300 percentiles (exact); P003 O₁ 2752/477 and O₄ 28; P017 isoscalar nodes 102.0/266.9 and isotope spread 251–278; P011 proton-only node 292 (P011: 284–291 at different δ/m); P002 99.5 %.
- Node finder depth criterion 0.6/15 keV inherited from P032; the xenophobic natural-Xe minima are ×0.05 deep, robust.
- Enriched-target mass correction (nuclei per tonne) ≤ 3 %; abundances from WimPyDD (¹²⁴Xe, ¹²⁶Xe omitted, 0.18 %).

## 5. Failed or abandoned approaches
- Locating the F_p, F_n zeros as *deep local minima* of W_pp, W_nn with a neighbour-depth criterion missed the second zeros of ¹²⁹Xe, ¹³⁰Xe, ¹³⁴Xe: the responses there are ~10⁻⁶ of W(0) and can be slightly negative (tabulation noise), so "depth" is ill defined. Replaced by sign changes of W_pn (exact for a dyad).
- A dyad test normalised to the q → 0 maximum (|W_pn² − W_pp W_nn|/max) was insensitive (¹³¹Xe passed); a pointwise ratio evaluated over 5–350 keV was dominated by noise near the zeros (¹³⁶Xe "failed" with deviation 20). Final: spin-based criterion (J ≤ ½) plus a pointwise check restricted to 5–200 keV and > 8 keV from any zero.
- Minimising N_lo for inelastic δ ≥ 250 keV is meaningless (window empty); replaced by N_mid.
- ρ(248) is undefined where the isoscalar reference vanishes at 248 keV (δ = 380 keV, Sun-frame onset 228 keV, node at 250 keV); flagged nan.

## 6. Figures (`output/work/P078/figures/`)
- `fig1_amplitudes_nodes.png` — left: signed shell-model f_p (solid) and f_n (dashed) for ¹²⁹Xe, ¹³²Xe, ¹³⁶Xe; middle: amplitude (F_p + rF_n)/A for r = +1, −1, −0.7 (the r = −1 and −0.7 curves have no zero near 248 keV; the isoscalar crosses zero at 251–276 keV depending on A); right: first (faint) and second (full) node position vs r for each isotope and for natural Xe, with the jump of the second node from 164 to 381 keV between r = −0.85 and −0.80.
- `fig2_rscan.png` — left: N_lo(r) for 400/1000/4000 GeV (Sun-frame) and 1 TeV annual mean, with N_lo = 5 and 10 lines; middle: ρ(200–270), ρ(248), ρ(5.4–55) and the shape-ratio enhancement vs r at 1 TeV; right: N_mid(r) for δ = 250/300/350/380 keV (annual mean) and elastic.
- `fig3_enriched_targets.png` — spectra per tonne at equal c_p for natural, ¹³⁶Xe-enriched and pure ¹²⁹Xe, r = +1 and −1: elastic (Sun-frame) and δ = 300 keV (annual mean); the ¹³⁶Xe isoscalar node at 251 keV is visible.
- `fig4_O4.png` — left: N_lo and ρ(200–270) vs r for O₄ (with O₁ for comparison) showing the spin-cancellation dip at r = −0.025; right: normalised O₄ spectra for r = +1, −1, 0.

## 7. Result tables (`output/work/P078/`)
`response_q0_checks.csv`, `isotope_nodes.csv`, `node_vs_r.csv`, `validation_factorised_vs_direct.csv`, `r_scan.csv` (401 r × {Sun-frame, annual} × {δ = 0, 250, 300, 350, 380 at 1 TeV; δ = 0, 300 at 400/4000 GeV}), `key_points_1TeV.csv`, `sigma_p_one_event.csv`, `xenophobic_isotope_interference.csv`, `enriched_targets.csv`, `enriched_target_test.csv`, `O4_spin_structure.json`, `O4_r_scan.csv`, `P078_summary.json`, `kernels_O1_O4_1000gev.npz` (our cache), `run_log.txt`.

## 8. Extended discussion
1. *Where we agree with P032.* Everything computed in common agrees to better than 0.5 % (N_lo, ρ), 0.4 keV (nodes), 10⁻³ (r_min) and exactly (δ = 300 percentiles). The isovector second node at 182 keV vs isoscalar 267 keV and the per-isotope spread (173–190 vs 251–278 keV) are confirmed with an independent construction of the signed amplitudes. The verdict "N_lo ≥ 6, never ≤ 5" stands.
2. *Where we add or differ.* (a) We identify the mechanism per isotope: 248 keV lies inside the band of second F_n zeros (231–259 keV), so the event energy is proton-dominated for *all* r — the isospin ratio changes the rate at 248 keV by ≤ 20 % over r ∈ [−2, 2] while changing the low-energy rate by ×2500. (b) P032 described the r-dependence of the second node as crossing 248 keV at "r ≈ −5…−∞"; we find that the node also collapses through 248 keV at r = −0.826 when it jumps between its two branches, so couplings just below −0.83 are *disfavoured* (event on a node), which sharpens the N_lo(r) minimum's lower edge. (c) P032's inelastic percentiles used the Sun-frame halo labelled "annual"; at δ = 300 this is harmless (≤ 0.7 %), but at 380 keV the Sun-frame halo exaggerates the isovector rate enhancement (×7.4 vs ×1.8 annual). (d) New: at δ = 380 keV isospin violation does raise the absolute high-energy rate at fixed c_p (×1.8–3.4) because the isoscalar sits on its 252 keV node; at δ ≤ 350 it never does. (e) New: the O₄ spin-cancellation point r = −0.025 gives N_lo = 3.6 < 5 — the only N_lo < 5 spot in the O₁/O₄ isospin planes — at ×480 rate cost and 2 % tuning. (f) New: isotope engineering is not a decisive test (124–1900 events).
3. *Nuclear-structure caveat.* All of this rests on DMFormFactor-v6 one-body density matrices with a common oscillator parameter; f_n − f_p at q ≈ 250 MeV — the neutron-skin-sensitive quantity — is only schematic, and two-body currents (Hoferichter et al. 2016; recalled, likely: few–30 % of the one-body isoscalar M response) are absent. Since the 248 keV amplitude is a small proton term (|f_p| ≈ 0.01) plus an even smaller neutron term, a 10 % change in the neutron radial distribution moves the F_n zero band by several keV and could shift the isoscalar node onto or away from the event; the *sign* of the effects here is robust, the ρ(248) values to ~20 %, and the node positions to perhaps ±10 keV (P032's LZ-vs-WimPyDD offset of 1.8× in N_lo is of this order).

## 9. References
LZ Collaboration, arXiv:2609.02823 (2026). A. L. Fitzpatrick, W. Haxton, E. Katz, N. Lubbers, Y. Xu, JCAP 02 (2013) 004. N. Anand, A. L. Fitzpatrick, W. C. Haxton, Phys. Rev. C 89, 065501 (2014). J. L. Feng, J. Kumar, D. Marfatia, D. Sanford, Phys. Lett. B 703, 124 (2011). I. Jeong, S. Kang, S. Scopel, G. Tomar, Comput. Phys. Commun. 276, 108342 (2022) [WimPyDD]. M. Hoferichter, P. Klos, J. Menéndez, A. Schwenk, Phys. Rev. D 94, 063505 (2016). D. Baxter et al., Eur. Phys. J. C 81, 907 (2021). Corpus: P002, P003, P007, P011, P016, P017, P030, P032, P035, P046, P050, P062.

## 10. Tools and provenance (mirrors `output/provenance/P078.json`)
- Agent tools: Read ×14 (PAPER_GUIDE; P032.md; P032 details; P062 script; P032 script; lzcommon l. 1–120, 295–405; run_log ×3; 4 figures), Bash ×15 (ls/grep of scripts, ledger rows, LZ tex l. 486–500 and isospin grep, P032 CSV/JSON, run logs; WimPyDD API inspection and timing; per-isotope minima diagnostic; 6 script runs; CSV/JSON post-processing; paper Results rewrite; word counts ×4; JSON check), Write ×5 (script, details, provenance, paper ×2), Edit (13 script fixes, 9 paper trims, 2 bookkeeping), Skill (dataviz).
- Software: python 3.12.13; WimPyDD 2.0.4 (`Xe.func_w`, `nuclear_current`, `eft_hamiltonian` via lzcommon, `diff_rate` with `isotopes_list`, `get_vmin` convention, `streamed_halo_function` via lzcommon); numpy 2.5.3; scipy 1.18.1 (special.erf, integrate.trapezoid/cumulative_trapezoid, optimize.minimize_scalar); matplotlib 3.11.2; common/lzcommon.py (wd, wd_halo, wd_hamiltonian, wd_rate, helm_F2, LZ, M_NUCLEON_GEV, VESC_KMS, OSIG/LSIG for comparison).
- Recalled knowledge: (ħc)² = 0.3894 × 10⁻²⁷ GeV² cm² (certain); xenophobic f_n/f_p ≈ −0.7 and its origin (Feng et al. 2011; certain); M_J multipole selection rule (even J ≤ 2J_nuc) (certain); 0νββ-grade ¹³⁶Xe enrichment ≈ 90 % with ¹³⁴Xe remainder (likely); two-body currents few–30 % of one-body isoscalar M (Hoferichter et al. 2016; likely, discussion only); Σ''/Σ' = 1/3 : 2/3 at q → 0 (certain, confirmed numerically).
- Datasets: none. Data requests: none. WimPyDD-generated files: none beyond our cache `output/work/P078/kernels_O1_O4_1000gev.npz` (`diff_rate` writes no response-function files).
