# P072 — Exothermic scattering of a metastable excited state as the origin of the LZ 248 keV event: kinematics, spectral shape, companions and the required relic fraction

Research record (simulated date 2026-09-15; hep-ph; IDM; competes with P058 §5.6). Script: `output/code/P072_exothermic_origin.py`
(runs in 16 s from the cached per-isotope tables, ~25 s cold). Tables in `output/work/P072/*.csv`, `P072_results.json`, `run_log.txt`;
figures in `output/work/P072/figures/`.

## 1. Motivation and question

P002/P007/P011/P021 read the 248 keV event as endothermic inelastic scattering χ₁N → χ₂N near the kinematic ceiling. P026 showed that both LZ-motivated
models leave f₂ = 0.42–0.50 of the dark matter in the excited state at freeze-out. If χ₂ is metastable (τ ≳ t_U), its down-scattering χ₂N → χ₁N releases
|δ| into the recoil with no velocity threshold. P058 treated the "event is a down-scatter" reading as a side result: 248 keV at the 97–99.7th percentile of
the exothermic spectrum, 2.2–9 companions in 125–200 keV per 200–270 keV event, four-bin Z = 2.7–3.2. We give an independent, fuller treatment:
(1) exothermic kinematics and shape over m = 0.3–10 TeV, |δ| = 100–500 keV, and which (m, |δ|) put 248 keV at the spectral peak or upper edge;
(2) whether heavy masses or low halo dispersion narrow the line enough to keep 125–200 keV empty, with a likelihood that uses the event's energy
(not only its bin); (3) the f₂σ product, the χ₁ up-scattering companion rate, the lifetime and the dark-photon window; (4) the modulation phase and the
16 June date; (5) the two-population (f₂, δ) plane; (6) a Bayes-factor verdict against the endothermic reading.

## 2. Framework and equations

Two-body kinematics (m_N nuclear mass, μ reduced mass, δ < 0 exothermic):

  v_min(E_R) = |m_N E_R/μ + δ| / √(2 m_N E_R)                                                   (1)
  E_±(v) = (μ²v²/m_N)[1 + |δ|/(μv²) ± √(1 + 2|δ|/(μv²))]  →  E* ± Δ(v),  E* = |δ|μ/m_N,  Δ(v) = (μ²v²/m_N)√(1 + 2|δ|/(μv²)) ≃ μv√(2μ|δ|)/m_N    (2)

Δ grows with μ (∝ μ^{3/2} for m ≪ … and saturating at v√(2m_N|δ|) for m → ∞): **a heavier DM particle broadens, not narrows, the exothermic window**;
only the DM speed v narrows it. O₁ rate:

  dR/dE = (ρ₀/m) Σ_i N_i (m_i/2μ_i²) |c|² F_i²(E) η(v_min,i(E)),  η(v_min) = ∫_{v>v_min} f(v)/v d³v.                 (3)

*Exact factorisation.* WimPyDD's per-stream kernel for O₁ is, isotope by isotope, a pure step function in v (the operator has no velocity dependence and
WimPyDD multiplies dσ/dE ∝ 1/v² by v²): K(E, v) = (1000 GeV/m) Σ_i w_i(E) θ(v − v_min,i(E; m, δ)), where w_i(E) is the single-isotope rate per unit δη at
m = 1000 GeV. We verified (i) single-stream values are independent of v and δ above threshold (v = 800–10⁴ km/s, δ = −300…+300: identical to 16 digits),
(ii) the normalisation scales exactly as 1/m (r·m = 4621.5 for m = 300, 1000, 4000, 10⁴ GeV), (iii) kernels rebuilt from the nine w_i reproduce P058's cached
WimPyDD kernels (iso 1 TeV −300, p 1 TeV −380, iso 4 TeV −350, iso 400 GeV +300, Higgsino 1 TeV −366; 641 energies × 1200 streams) with max relative
deviation 1.7–4.2×10⁻¹⁶. Hence dR/dE = (1000/m) Σ_i w_i(E) η(v_min,i(E)) is an *exact* WimPyDD spectrum for any (m, δ, halo), costing 27 × 641 diff_rate
calls once (5 s on 9 cores) instead of ~10⁵ per configuration. v_min uses WimPyDD's isotope masses 0.931·A and its factor 300 (`get_vmin`).

Halos: the streamed Baxter-2021 SHM (v₀ = 238, v_esc = 544 km/s, ρ₀ = 0.3 GeV/cm³) from `lz.wd_halo(day_of_year)` on 24 days; η(v) = Σ_{v_j ≥ v} δη_j
exactly as `diff_rate` sums streams; the annual mean is the 24-day average (P035/P058 convention; LZ used a time-averaged SHM). Analytic truncated
Maxwellians (`lz.eta0`, whose δη agrees with WimPyDD's to 10⁻⁸ in the sum) for the low-dispersion and dark-disk variants.

Detector: ε_ROI = 0.96 Φ((E−5.4)/2.5)[1−Φ((E−269.9)/11.5)] (Fig. S2; P011/P021 convention); empty high-energy region 600–1700 phd with
ε_HE = 0.955 Φ((E−271.6)/10.9)[1−Φ((E−670)/23)] (P038 edges, 1700 phd extrapolated as in P058); resolution σ_E = 11√(E/248) keV (P009/P021).
All ROI quantities are computed on the *observed*-energy accepted spectrum s_obs = S·(dR/dE ε_ROI), S the Gaussian smearing matrix (0.5–799.5 keV, 1 keV).

Statistics. Regions and data: 5.4–125 keV (n = 42, b = 58.6, NR-band acceptance 0.866; P016's digitised Fig. 5), 125–200 keV (0, 0.02), 600–1700 phd
(0, 0.606: 0.1 wall + 0.5 RFR MSSI in the HE SB, MSSI table, + 0.006 in 600–800 phd, P038), and the 200–270 keV region with one event at 248 keV over
b_H = 5.7×10⁻⁴ flat (P016 anchor). "Five-term" likelihood for a hypothesis H with per-unit-coupling counts N_X and accepted observed-energy density D(248):

  −ln L(κ) = Σ_{X ∈ lomid, gap, he} [μ_X − n_X ln μ_X] + (b_H + κN_hi) − ln(b_H/70 keV + κ D(248)),  μ_X = b_X + acc_X κ N_X.        (4)

The last term is the extended likelihood of the single event at its observed energy; replacing it by μ_hi − ln μ_hi gives P058's four-bin form ("Z₄").
Profile: Z = √(2[−ln L(0) − min_κ(−ln L)]). Bayes factors vs background: B = ∫π(μ_hi) L(κ = μ_hi/N_hi) dμ_hi / L(0) with π flat on μ_hi ∈ [0, 10]
(P001/P016 convention for the 248 keV signal), variants [0, 3], log-uniform [10⁻³, 10], without the 600–1700 phd term, and binned. Exothermic vs
endothermic: B_exo/B_endo at the best δ of each (1 TeV and all masses) and marginalised over δ with a flat prior (100–500 keV for both, or 250–390 keV
for the endothermic reading). Date factor p(t_obs|H)/p_flat = R_ROI(day 167)/⟨R_ROI⟩ from the 24-day curves (periodic cubic spline), as in P006.

## 3. Inputs

| Input | Value | Source |
|---|---|---|
| Exposure, ROI, efficiency, resolution | 2.84 t·yr; 5.4–269.9 keV at 50 %, plateau 0.96, erf σ 2.5/11.5 keV; σ_E = 11√(E/248) | LZ paper (abstract, Fig. S2); P009/P021 |
| High-energy edges | E50 = 271.6 / 423.2 keV (600 / 1000 phd), σ 10.9 / 14.5; d ln S1c/d ln E = 1.154; 1700 phd → 670 keV, σ 23 | P038; P058 extrapolation |
| Empty region 600–1700 phd | 0 observed; 0.606 expected | LZ MSSI table (tex lines 765–783), Fig. S4; P038 |
| Low-energy counts | 42 vs 58.6 (5.4–125 keV, ±1.5σ NR band); 0 vs 0.02 (125–200); 1 vs 5.7×10⁻⁴ (200–270) | P016, P021 |
| Halo | Baxter-2021 SHM, 24 days; June 16 (day 167); Sun frame | `lz.wd_halo`, `lz.eta0` |
| Couplings | isoscalar c_p = c_n = 1/m_v² (WimPyDD c⁰ = 2/m_v²), proton-only c_p = 1/m_v², Higgsino c_p = 6.20×10⁻⁷, c_n = −8.25×10⁻⁶ GeV⁻² | P003 convention; P007/P011/P058 |
| σ_N(unit) | c²μ_p²/π = 2.964×10⁻³⁸ cm² at 1 TeV | P058 |
| Freeze-out f₂ | 0.50 (dark photon), 0.42 (Higgsino) | P026 |
| Relic α_D | 0.0245 at 1 TeV | P011 |
| χ₂ → χ₁νν̄ width | Γ = 3G_eff²δ⁵/(120π³), G_eff = g_Dε tanθ_W g/(4cosθ_W m_Z²); σ_p = 16παα_Dε²μ_p²/m_A′⁴ | P011 eq. 3 |
| Constants | G_F, sin²θ_W = 0.2312, α, m_Z, ħ, t_U = 13.8 Gyr = 4.35×10¹⁷ s, AMU | recalled, certain |
| Dark-disk parameters | lag 10–100 km/s, σ 10–50 km/s, ρ_d/ρ₀ ≲ 0.2–1 | recalled (Read et al. 2008; Gaia bounds Schutz+2018/Buch+2019), uncertain |
| SPI line bound | f₂/τ_γ < 5×10⁻²³ s⁻¹ | P066 |

## 4. Results

### 4.1 Validation (`P072_validation_RROI.csv`, run_log Part A)

Rebuilt kernels vs P058's WimPyDD kernels: 1.7–4.2×10⁻¹⁶. R_ROI (exothermic ROI events per endothermic ROI event, f₂ = 1, 1 TeV, annual, true-energy
bins): iso 915.5 / 2.776×10⁴ / 8.122×10⁶ at δ = 300/350/380 (P058: 916 / 2.78×10⁴ / 8.11×10⁶); proton-only 859.1 (P058 859, P011 858) and 2.644×10⁶ at 380
(P058 2.64×10⁶); Higgsino 963.4 (P058 963). Percentile of 248 keV and companions at (iso, 1 TeV, 278/300/350/380): 0.997/0.996/0.993/0.991 and
6.21/5.79/4.80/4.20 (true energy) vs P058 0.997/0.996/0.993/0.992 and 6.3/5.9/4.9/4.3; in observed energy 5.76/5.39/4.51/3.98 (edge leakage).
Four-bin Z in P058's form: exo 2.93/2.96/3.02/3.04 vs P058 2.90/2.94/3.00/3.03; endo 3.02/3.07/3.18/2.33 vs 3.00/3.05/3.19/2.34. **Agreement ≤ 1 %
everywhere**, so every difference below is method, not machinery.

### 4.2 Kinematics and spectral shape (`P072_grid.csv`, `P072_248_at_peak_or_edge.csv`; Fig. 1 left, Fig. 2)

E* = 248 keV requires |δ| = 349 / 309 / 278 / 263 / 256 / 251 keV at m = 0.3 / 0.5 / 1 / 2 / 4 / 10 TeV; there Δ(250 km/s) = 138 / 165 / 196 / 208 / 223 /
230 keV. Δ increases monotonically with m (Eq. 2) and saturates at v√(2m_N|δ|) = 230 keV: the "heavy mass narrows the line" premise is false for exothermic
scattering; the window half-width is set by the halo speed alone. Accepted observed spectra (isoscalar; 1 TeV unless stated):

| |δ| [keV] | E* | mode | p16 / p50 / p84 | FWHM | percentile of 248 | N(125–200)/N(200–270) | N(5.4–125)/N(200–270) | N(600–1700 phd)/N(200–270) |
|---|---|---|---|---|---|---|---|---|
| 200 | 178 | 47 | 37 / 54 / 111 | 38 | 0.999 | 6.9 | 46 | 0.67 |
| 250 | 223 | 57 | 50 / 75 / 170 | 36 | 0.998 | 6.2 | 12 | 0.77 |
| 278 (E* = 248) | 248 | 62 | 60 / 137 / 184 | 36 | 0.997 | 5.7 | 5.1 | 0.84 |
| 300 | 267 | 158 | 70 / 150 / 190 | 76 | 0.996 | 5.4 | 2.9 | 0.91 |
| 350 | 312 | 164 | 132 / 165 / 200 | 78 | 0.993 | 4.5 | 0.75 | 1.11 |
| 400 | 356 | 170 | 143 / 173 / 205 | 78 | 0.990 | 3.6 | 0.23 | 1.41 |
| 450 | 401 | 176 | 149 / 179 / 211 | 78 | 0.986 | 2.8 | 0.09 | 1.89 |
| 500 | 446 | 184 | 156 / 185 / 216 | 76 | 0.980 | 2.1 | 0.04 | 2.7 |
| 300 GeV, 350 | 249 | 160 | 123 / 161 / 196 | 76 | 0.994 | 5.2 | 1.2 | 0.78 |
| 4 TeV, 256 | 248 | 58 | 53 / 78 / 176 | 37 | 0.997 | 5.9 | 8.7 | 0.86 |
| 10 TeV, 500 | 494 | 186 | 157 / 186 / 217 | 76 | 0.979 | 1.97 | 0.03 | 3.2 |

Two features: (i) for |δ| ≲ 280 keV at 1 TeV the accepted spectrum is *two-lobed*, with its mode at ≈ 60 keV below the 102 keV M-response node and a second
lobe at ≈ 160 keV, because E* sits at or below the 266 keV node and the F²-weight pulls the events down; (ii) for |δ| ≥ 300 keV the single lobe peaks at
158–186 keV and never moves past the 266 keV node whatever |δ| — the E* lobe above the node is removed by the 270 keV edge. **Over the whole grid
(6 masses × 41 splittings) the mode never exceeds 186 keV, the 84th percentile never exceeds 217 keV and the 95th never exceeds 237 keV; 248 keV lies at
the ≥ 97.9th percentile everywhere (99.7th at E* = 248).** There is no (m, |δ|) that puts 248 keV at the peak or even at the upper 84/95 % edge of the
accepted exothermic spectrum. The minimum companion ratio N(125–200)/N(200–270) is 1.97 (10 TeV, 500 keV); at E* = 248 it is 5.5–6.4.

### 4.3 Likelihood with the event's energy, and Bayes factors (`P072_grid.csv`, `P072_best_cases.csv`, `P072_vs_P058.csv`; Fig. 2 right)

1 TeV isoscalar, annual halo, flat prior μ_hi ∈ [0, 10] (B), date factor F = R(16 June)/⟨R⟩:

| hypothesis | |δ| | Z₄ (P058 form) | Z₅ (with density) | μ̂_hi [events] | μ̂_gap | B vs bkg | F | B·F |
|---|---|---|---|---|---|---|---|---|
| exo | 278 | 2.93 | 2.45 | 0.11 | 0.63 | 0.60 | 1.04 | 0.63 |
| exo | 300 | 2.96 | 2.50 | 0.12 | 0.66 | 0.77 | 1.03 | 0.79 |
| exo | 350 | 3.02 | 2.58 | 0.15 | 0.65 | 1.12 | 1.02 | 1.15 |
| exo | 400 | 3.05 | 2.65 | 0.16 | 0.59 | 1.49 | 1.03 | 1.54 |
| exo | 500 | 3.07 | 2.73 | 0.17 | 0.36 | 1.96 | 1.07 | 2.11 |
| endo | 300 | 3.07 | 2.63 | 0.17 | — | 1.48 | 1.43 | 2.12 |
| endo | 350 | 3.18 | 2.88 | 0.24 | — | 4.16 | 2.41 | 10.0 |
| endo | 380 | 2.33 | 2.28 | 0.02 | — | 0.084 | 3.92 | 0.33 |

The event density at 248 keV per 200–270 keV event is 0.0039–0.0053 keV⁻¹ for the exothermic spectra (vs 1/70 = 0.0143 for a flat bin) because the accepted
spectrum is falling steeply toward the 266 keV node and the 270 keV edge; using it costs the exothermic hypothesis 0.35–0.5σ relative to P058's binned Z
(2.93–3.07 → 2.45–2.73) and reduces the fitted signal to μ̂_hi = 0.11–0.17 events (the P016 mechanism: the profile fit buys the event by shrinking the signal
so that the 0.36–0.66 predicted 125–200 keV events are tolerated). The best exothermic Bayes factor against background is B = 1.96 (1 TeV, |δ| = 490 keV;
2.18 at 300 GeV / 500 keV), i.e. *no evidence*; at E* = 248 keV B = 0.60 (the hypothesis is mildly disfavoured relative to background). The best endothermic
values are B = 4.16 (1 TeV, 350 keV), 10.2 (300 GeV, 320 keV), 3.0 (4 TeV, 350), 2.9 (10 TeV, 360). Best-vs-best B(exo/endo) = 0.47 (1 TeV), 0.21 with the
date factor; over all masses (300 GeV endothermic optimum) 0.041 with the date. Marginalising with a flat prior over |δ| = 100–500 keV *for both* gives
B_exo = 0.83, B_endo = 0.69 (ratio 1.21), but this charges the endothermic reading for the 60 % of that range where it cannot make the event (δ < 250 keV
gives 10³–10⁴ low-energy companions; δ > 390 keV is kinematically forbidden); restricting the endothermic prior to its physical 250–390 keV range gives
B_endo = 1.71 (4.03 with date) and B(exo/endo) = 0.49 (0.22 with date). Prior variants at 1 TeV (best δ each): μ_max = 3 → 6.5 vs 13.9; log-uniform → 12.7
vs 19; no 600–1700 phd term → 12.5 vs 301 (the endothermic optimum then moves to 390 keV, P038's point); binned → 5.4 vs 10.8. **Every variant gives
B(exo/endo) = 0.04–0.67; the exothermic origin is disfavoured by a factor 2–25 relative to the endothermic reading and is never favoured over background.**
The 600–1700 phd region contributes: exothermic predictions of 0.84–2.7 events there per 200–270 keV event (e^{−0.84…−2.7} penalties at μ_hi = 1),
comparable to the endothermic 0.87 (350 keV) and decisive against endothermic 380 keV (42.7).

### 4.4 Coupling, χ₁ companions, lifetime, dark photon (`P072_coupling_lifetime.csv`)

One exothermic ROI event in 2.84 t·yr needs κf₂ = (c m_v²)² f₂ = 1.35×10⁻⁸ / 3.7×10⁻⁸ / 8.1×10⁻⁸ / 1.4×10⁻⁷ / 1.8×10⁻⁷ / 3.6×10⁻⁷ at |δ| = 200 / 250 / 300 /
350 / 380 / 450 keV (iso, 1 TeV), i.e. **σ_N f₂ = 4.0×10⁻⁴⁶ / 1.1×10⁻⁴⁵ / 2.4×10⁻⁴⁵ / 4.0×10⁻⁴⁵ / 5.3×10⁻⁴⁵ / 1.1×10⁻⁴⁴ cm²**, σ_N = 2×σ_N f₂ at f₂ = 0.5
(×4.4 at 4 TeV). This is 69 / 134 / 394 / 6500 / 1.2×10⁵ times below the coupling of the endothermic fit at the same δ. The χ₁ population (1 − f₂) then
up-scatters at κ(1−f₂)N_endo = 1.4×10⁻² / 3.9×10⁻³ / 1.1×10⁻³ / 3.6×10⁻⁵ / 1.2×10⁻⁷ events (200–380 keV), all far below LZ's tolerance in every bin:
the χ₁ companion rate is irrelevant. Lifetime: f₂(today) = f₂,fo e^{−t_U/τ}, so f₂ ≥ 0.9 / 0.5 / 0.1 f₂,fo needs τ ≥ 9.5 / 1.4 / 0.43 t_U = 4.1×10¹⁸ /
6.3×10¹⁷ / 1.9×10¹⁷ s; the hypothesis needs τ(χ₂) ≳ t_U, in the excluded window of P058 only if the *endothermic* fit is also imposed (here it is not).
Dark photon (proton-only, 1 TeV): σ_p f₂ = 1.8×10⁻⁴⁵ / 4.8×10⁻⁴⁵ / 1.0×10⁻⁴⁴ / 1.65×10⁻⁴⁴ / 2.1×10⁻⁴⁴ / 3.9×10⁻⁴⁴ cm² at |δ| = 200–450 keV. With τ = C/(σ_p m_A′⁴)
(ε²α_D tied to σ_p) the one-event condition f₂,fo σ_p e^{−t_U/τ} = σ_p f₂ has solutions only for **m_A′ ≤ 14.3 / 8.5 / 5.6 / 4.1 / 3.5 / 2.4 GeV**
(δ = 200–450; τ = t_U at the ceiling), and on the long-lived branch τ = 1.2×10²¹ s (m_A′/GeV)⁻⁴ at 300 keV with ε = 4.0×10⁻⁸ (α_D = 0.1) or 8.1×10⁻⁸
(relic α_D) at m_A′ = 1 GeV. P058's "m_A′ ≲ 1.4 GeV" used τ₀ at the endothermic-fit σ_p; with the 388× smaller exothermic-origin σ_p the ceiling is
1.4 × 388^{1/4}/e^{1/4} ≈ 5.6 GeV — a correction of P058. Higgsino χ₂: τ_νν̄ = 1.2×10⁶ s (P026) → f₂(today) = 0: the Higgsino cannot be the origin.
γ-line (P066): f₂/τ_γ < 5×10⁻²³ s⁻¹ with f₂ = 0.5, τ ≥ t_U requires BR_γ < 4.4×10⁻⁵ (τ/t_U); automatically satisfied by kinetic mixing (BR_γ = 0).

### 4.5 Annual modulation and the date (`P072_modulation.csv`; Fig. 4)

Exothermic ROI rate: a₁ = 2.3–5.5 % (|δ| = 150–450 keV, 1 TeV), maximum at day 145 (25 May); R(16 June)/⟨R⟩ = 1.02–1.05 (grid range 1.007–1.086).
The 125–200 keV rate alone modulates with *winter* maximum (a₁ = −0.7 to −3 %) for |δ| ≤ 300 keV. Endothermic: a₁ = 0.17–1.18, R(16 June)/⟨R⟩ =
1.17 / 1.43 / 2.41 at δ = 200 / 300 / 350 keV (grid 1.09–6.1; P006: 1.41 / 2.08 / 2.55 at 300/350/380). The date therefore favours the endothermic reading
by 0.72⁻¹–0.43⁻¹ at 300–350 keV; a down-scattering population is time-flat and would show no June clustering in LZ's future data (P034 test is void).

### 4.6 Two-population plane (`P072_two_population_plane.csv`; Fig. 3)

Spectrum = κ[(1 − f₂) endothermic + f₂ exothermic], κ profiled, 1 TeV. The maximum of B is at f₂ → 0, δ = 350 keV (B = 4.16, Z₅ = 2.88, μ̂_hi = 0.24,
μ̂_gap = 0.22). Along δ = 350 keV, B = 4.16 / 3.88 / 2.59 / 1.37 / 1.12 at f₂ = 0 / 6×10⁻⁶ / 6×10⁻⁵ / 6×10⁻⁴ / 1, halving at f₂ ≈ 1.8×10⁻⁴ (P058's fixed-κ
90 % limit 5.7×10⁻⁵); along δ = 300 keV 1.48 → 0.77; along δ = 380 keV the exothermic admixture *raises* B from 0.08 to 1.34 (the pure endothermic
380 keV spectrum is killed by the 600–1700 phd region, P038/P058), and for δ > 390 keV only the exothermic population exists (B = 1.5–2.0). With
f₂ ≥ 0.1 the best point is (0.1, 490 keV), B = 1.96, exothermic fraction of 200–270 keV events = 1.0. Nowhere in the plane does a mixture beat the
pure endothermic optimum; "one event at 248 keV and nothing else" is best mimicked by f₂ ≲ 10⁻⁴ at δ = 330–370 keV — i.e. by a χ₂ that has decayed.

### 4.7 Can a slow population make a line? (`P072_halo_variants.csv`; Fig. 1 right)

1 TeV, |δ| = 278 keV (E* = 248): SHM variants with v₀ = 150 / 100 / 50 km/s but v_E = 251 km/s give FWHM 74 / 72 / 69 keV, N(125–200)/N(200–270) =
5.7 / 6.1 / 6.6 and percentile 0.995 — the Earth's own speed sets the width, so halo dispersion is irrelevant. Only a population co-moving with the Sun
(dark disk) narrows it: lag 100 / 50 / 50 / 20 / 10 km/s with σ = 50 / 50 / 20 / 20 / 10 km/s → FWHM 64 / 60 / 43 / 40 / 32 keV, mode 188 / 200 / 218 / 230 /
242 keV, N(125–200)/N(200–270) = 1.85 / 0.92 / 0.15 / 0.029 / 0.0006, percentile 0.98 / 0.97 / 0.93 / 0.85 / 0.68, Z₅ = 2.9 / 3.1 / 3.4 / 3.6 / 3.8, B = 5 / 15 /
65 / 163 / 333. But f₂ is set by freeze-out and is the same in disk and halo, so the halo χ₂ population is unavoidable: adding the SHM at ρ_d/ρ₀ = 1 (0.25)
to the (lag 50, σ 20) disk gives N(125–200)/N(200–270) = 1.7 (3.5), percentile 0.983 (0.993), B = 7.5 (2.0). Gaia-era bounds (recalled, uncertain)
allow ρ_d/ρ₀ ≲ 0.2–0.3, so B ≲ 2: the dark-disk escape route is closed. The total exothermic rate is nearly halo-independent (1.85 / 1.45 / 1.1×10⁷ per
unit coupling for SHM / v₀ = 100 / disk), as expected from nσv ∝ v⁰.

## 5. Agreement and disagreement with P058

Agree (to ≤ 1 %): R_ROI, spectra, percentile 99.7 % at E* = 248, 2.2–9 (here 2.1–6.9 smeared) companions per 200–270 keV event, four-bin Z 2.9–3.0 in
their form, modulation 2–5 % with late-May phase. Disagree / extend: (i) using the event's energy inside the 200–270 keV bin lowers the exothermic
significance to Z = 2.45–2.73 (−0.35 to −0.5σ) and gives B ≤ 2 vs background, B(exo/endo) = 0.04–0.67 — a quantitative verdict P058 did not attempt;
(ii) P058's suggestion that P072 test the shape with LZ's full likelihood is met here with a five-term observed-energy likelihood; (iii) the mass
dependence: heavier χ broadens the window (Δ ∝ μ^{3/2}), so no multi-TeV escape exists; (iv) low-dispersion halos do not help, only a co-moving disk does,
and it is capped at B ≲ 2 by its density; (v) P058's dark-photon condition "m_A′ ≲ 1.4 GeV" for the event-as-exothermic case is corrected to m_A′ ≤ 5.6 GeV
(δ = 300 keV; 2.4–14 GeV over 450–200 keV) because the exothermic-origin σ_p is 388× smaller than the endothermic fit; (vi) the (f₂, δ) plane shows the mixture
never beats the pure endothermic optimum, and that at δ = 380 keV a χ₂ admixture *improves* the fit (B 0.08 → 1.3), consistent with P058's Δ(−2lnL) = −3.7.

## 6. Robustness

- Smearing: observed-energy bins vs true-energy bins change the companion ratios by 4–10 % (edge leakage: 5.76 vs 6.21 at 278 keV) and the four-bin Z
  by ≤ 0.03 (2.93 vs P058's 2.90).
- Halo frame: June/December/Sun-frame exothermic counts differ by ±4 % (P058; here Sun-frame vs annual 1.66 vs 1.68×10⁷ at 278 keV); the exothermic
  results are halo-insensitive; the endothermic denominators carry the halo dependence (×0.7–1.6 at 300 keV), which affects only the κ ratios in §4.4.
- Backgrounds (`P072_robustness.csv`, 1 TeV): b_gap ×0.5 or ×2 leaves B_exo and B_endo unchanged to 3 digits (the 125–200 keV penalty is the signal
  expectation e^{−μ_gap}, not the 0.02 background); b_he1700 ×0.5/×2 likewise. b_lomid ×0.5/×2 moves B_exo to 1.25/0.62 at 300 keV (±60 %, where
  N(5.4–125)/N(200–270) = 2.9), 1.29/1.05 at 350 keV and 1.97/1.95 at 500 keV, and B_endo by ≤ 4 %; the ordering exo < endo is unchanged.
- High-energy edge: replacing the extrapolated 1700 phd edge (b = 0.606) by P038's 1000 phd edge (b = 0.018) gives B_exo = 0.79/1.18/2.37 at 300/350/500 keV
  (+3 to +20 %) and B_endo = 1.53/4.54 (+3 to +9 %).
- Prior: B(exo/endo) ranges 0.04–0.67 across the flat/log/μ_max/no-HE variants (§4.3); the only ratio > 1 (1.21) arises from a flat δ prior extending the
  endothermic hypothesis into regions where it cannot produce the event.
- Isospin: proton-only (dark photon) gives fewer companions (2.1–2.9) and B_exo = 3.9 / 6.5 at δ = 300 / 380 vs B_endo = 7.3 / 3.3 — the same ordering at
  300 keV; at 380 keV the exothermic wins only because the endothermic proton-only 380 keV spectrum is also removed by the edge.

## 7. Failed or abandoned approaches

- Brute-force WimPyDD kernels for the 246-point (m, |δ|) grid (≈ 3×10⁵ diff_rate calls, ~3 h) — replaced by the exact per-isotope step factorisation
  after verifying it to 10⁻¹⁶ against P058's cache. A factorisation with the *mean* nucleus mass (P058's rejected shortcut) was not used.
- Passing `isotopes_list=[i]` (a list) to `diff_rate` fails; it must be a dict `{element_index: [isotope indices]}`.
- A flat δ prior over 100–500 keV for the endothermic hypothesis as the headline marginal comparison: it is prior-dominated (see §4.3) and reported only as a variant.
- Searching for (m, |δ|) with 248 keV at the mode or the 84/95th percentile: no crossing exists anywhere on the grid (`P072_248_at_peak_or_edge.csv`, empty lists).

## 8. Discussion and conclusion

A surviving χ₂ could produce a 248 keV recoil, but never *preferentially*: the F²-weighted, efficiency-truncated exothermic spectrum peaks at 60 or
160–190 keV for every mass and splitting, 248 keV is above its 98th percentile, and each 200–270 keV event comes with 2–7 events in LZ's empty
125–200 keV bin and 0.8–3 in the empty 600–1700 phd region. LZ's own data therefore give the hypothesis no support (B ≤ 2 against background) and
disfavour it by 2–25 relative to the endothermic reading it competes with, more once the June date is included. Heavier masses widen the window, low
halo dispersion is irrelevant because the Earth moves at 250 km/s, and a co-moving dark disk — the one way to make a line at E* — is capped by its density
at B ≲ 2. The required σ_N f₂ ≈ 2×10⁻⁴⁵ cm² (1 TeV, 300 keV) is 400× below the endothermic coupling, so the χ₁ population is silent; for a dark photon it
means m_A′ ≤ 5.6 GeV and ε ~ 10⁻⁸–10⁻⁷. The clean prediction of the hypothesis for LZ's next data is a time-flat population, 60–80 % of it below 200 keV.

## 9. Figures

- `figures/P072_fig1_spectra.png` — Left: accepted observed-energy spectra (unit area), isoscalar 1 TeV annual halo: χ₂ down-scatter at |δ| = 278 (E* = 248),
  300, 350, 450 keV vs χ₁ up-scatter at 350, 380 keV; grey band 125–200 keV (empty), dotted line the event. Right: |δ| = 278 keV for the SHM, a v₀ = 100 km/s
  halo and three co-moving dark-disk populations.
- `figures/P072_fig2_maps.png` — (m, |δ|) maps: N(125–200)/N(200–270), percentile of 248 keV, exothermic Bayes factor vs background with endothermic
  B = 2, 4, 8 contours; orange dots E* = 248 keV.
- `figures/P072_fig3_f2_delta_plane.png` — two-population plane at 1 TeV: Bayes factor vs background and the fitted 125–200 keV expectation; squares P058's f₂ limits.
- `figures/P072_fig4_modulation.png` — ROI rate/annual mean vs day of year: exothermic |δ| = 278, 300, 450 keV vs endothermic 350 keV; 16 June marked.

## 10. References

1. LZ Collaboration, arXiv:2609.02823 (2026). 2. D. Tucker-Smith, N. Weiner, PRD 64, 043502 (2001). 3. B. Batell, M. Pospelov, A. Ritz, PRD 79, 115019 (2009).
4. P. W. Graham, D. E. Kaplan, S. Rajendran, M. T. Walters, PRD 82, 063512 (2010). 5. J. I. Read, G. Lake, O. Agertz, V. P. Debattista, MNRAS 389, 1041 (2008) (dark disk).
6. I. Jeong, S. Kang, S. Scopel, G. Tomar, CPC 276, 108342 (2022) (WimPyDD). 7. D. Baxter et al., EPJC 81, 907 (2021). 8. R. E. Kass, A. E. Raftery, JASA 90, 773 (1995).
Corpus: dossier 00, P002, P006, P007, P009, P011, P016, P021, P026, P034, P038, P058, P066.

## 11. Tools and provenance (mirrors provenance/P072.json)

Agent tools: Read ×20 (PAPER_GUIDE, P058 paper + details + script, P011, P026, P038, P021, P066, P016, P034 papers, lzcommon.py, run_log, own figures ×7 over two rounds),
Bash ×27 (listings; ledger/tex greps and seds of tex lines 140–165, 440–465, 765–786; P002/P007 heads; P058 provenance/ledger-row inspection; WimPyDD source inspection for
`diff_rate`/`get_vmin`/`element`; four timing/factorisation test snippets; script runs ×8; pandas summary snippet; word counts; JSON check), Write ×5 (script, details, provenance,
paper twice), Edit ×29 (script ×21, details ×1, paper ×6, provenance ×1), Skill ×1 (dataviz; its JS validator skipped per PAPER_GUIDE).
Software: python 3.12.13; WimPyDD 2.0.4 (`diff_rate` with `isotopes_list={0:[i]}` single-stream calls, `streamed_halo_function` via `lz.wd_halo`, `eft_hamiltonian` via
`lz.wd_hamiltonian`, `Xe.mass/a/average_a`); numpy 2.5.3; scipy 1.18.1 (special.erf, stats.norm/poisson, optimize.minimize_scalar/brentq, interpolate.CubicSpline);
pandas 3.0.5; matplotlib 3.11.2; multiprocessing (fork, 9 workers); common/lzcommon.py (LZ, constants, eta0, wd, wd_halo, wd_hamiltonian, GEV_TO_CM2, M_V_GEV).
Recalled values (9): G_F, sin²θ_W, α, m_Z, ħ, AMU, t_U = 13.8 Gyr (certain); dark-disk lag/dispersion/density range (uncertain); nσv ∝ v⁰ for exothermic scattering (certain).
WimPyDD-generated files: none outside output/ (per-isotope tables cached in output/work/P072/cache/w_tables.npz).
