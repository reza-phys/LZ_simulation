# P049 — Muon-induced high-energy neutrons at 4850 ft: an independent estimate of untagged single scatters above 8 MeV and the meaning of the 41-minute-earlier muon

Research record (simulated date 2026-09-10). Script: `output/code/P049_muon_neutrons.py`; all numbers below are printed in
`output/work/P049/run_log.txt` and stored in `output/work/P049/P049_results.json`, `muon_rates.csv`, `sensitivity.csv`,
`spectrum_chain.csv`, `p013_mono_ss.csv`. Figures in `output/work/P049/figures/`.

## 1. Motivation and framework

LZ (arXiv:2609.02823, supplement "Neutrons", l. 793–798) treats muon-induced neutrons with a MUSUN + GEANT4 simulation of
2.1 × 10⁹ muons (1200 yr of LZ exposure) that produced no signal-like deposit, validated against GEANT4/FLUKA yields with a
factor-2 uncertainty; for rock neutrons of 50 MeV–1 GeV arriving without any accompanying particle the veto tagging efficiency
is 80.0 ± 4.0 %; the 90 % UL on untagged single scatters (SS) of any energy is 4.6 × 10⁻⁴ and the component is not in the
background model. The Discussion (l. 302) notes that the previous OD muon was 41 min before the event and the previous TPC muon
127 min before. P013 showed that muon-induced neutrons are the only neutron source with a hard enough SS spectrum to evade the
"companion population" argument (24–40 low-energy companions per 200–270 keV SS) and, taking LZ's UL at face value, obtained
0.6–1.6 × 10⁻⁵ window events. That number rests entirely on LZ's simulation. Here we rebuild the chain independently from recalled
muon fluxes and neutron yields, and quantify what the 41- and 127-minute gaps mean.

Chain: Φ_μ × Y_n × ρ × (spectrum) × (escape depth) × (direction) → neutron current out of the rock →
× receiving area of the water tank → × TPC/tank geometric fraction → × survival above 8.2 MeV through ≥ 2 m of water/GdLS →
× P(exactly one 200–270 keV scatter in the FV and nothing else | neutron enters the LXe) [P013 MC] → × untagged fraction.

## 2. Inputs

### 2.1 From the LZ paper
- 220 live days, 4.71 t FV (Data Analysis); `lz.LZ["live_days"]`.
- OD: 17 t GdLS (0.1 % Gd by mass) + 229 t water, PMTs in the water, detects neutrons, γ, muons (l. 83).
- Prompt veto: Skin > 2.5 phd within ±0.25 μs, OD > 4.5 phd within ±0.3 μs; delayed veto: Skin/OD > 300/200 keV within 600 μs (l. 143–144).
- Rock neutrons 50 MeV–1 GeV, tagging 80.0 ± 4.0 %; 2.1 × 10⁹ MUSUN muons = 1200 yr; factor-2 yield uncertainty; UL 4.6 × 10⁻⁴ (l. 793–798).
- Previous OD muon 41 min, previous TPC muon 127 min before the event (l. 302).

### 2.2 From the corpus
- P013: E_n,min(248 keV) = 8.20 MeV; mono-energetic transport MC (thr 3 keV, 4.74 t FV cylinder r = 63 cm, 4 < z < 135 cm, wall source,
  black-disk angular pattern) — P(SS in ROI | neutron enters LXe) and P(SS in 200–270 keV | enters LXe) at 8.5, 10, 15, 50, 100, 200 MeV
  (`output/work/P013/P013_results.json`, key `MC/mono`); muon-channel numbers 0.6–1.6 × 10⁻⁵, F(200–270 | SS) = 2.2–3.5 × 10⁻².

### 2.3 Recalled (flagged; central, range, reliability)
| item | value | range | reliability |
|---|---|---|---|
| muon flux Φ_μ at the 4850 ft level | 5.3 × 10⁻⁹ cm⁻² s⁻¹ (Majorana Demonstrator measurement 5.31 ± 0.17) | 4.4–5.5 × 10⁻⁹ | likely |
| mean muon energy | 320 GeV (Mei–Hime ⟨E⟩ = ε(1−e^{−bh})/(γ−2), ε = 693 GeV, b = 0.4 /km w.e., γ = 3.77, h = 4.3) | 280–350 | likely |
| neutron yield in rock Y_n | 3.5 × 10⁻⁴ n/(μ g cm⁻²) (Mei–Hime 4.14 × 10⁻⁶ E^0.74 → 3.0 × 10⁻⁴; assignment 4 × 10⁻⁴) | 2.5–5 × 10⁻⁴ | likely (×1.5) |
| production spectrum | (1−f_hard) Maxwellian T = 1.2 MeV + f_hard [E⁻¹ (1 MeV–knee), knee/E² (knee–3 GeV)], f_hard = 0.30, knee = 50 MeV | f_hard 0.2–0.4, knee 30–100 MeV | uncertain (×2 on f(>8.2)) |
| rock density | 2.7 g cm⁻³ | 2.6–2.9 | likely |
| escape depth still above 8.2 MeV, λ_esc | 20 / 30 / 40 cm for 8–30 / 30–100 / >100 MeV | (12,20,30)–(30,45,60) | uncertain (×1.5) |
| water tank | R = 3.8 m, H = 5.9 m (πR²H = 268 m³ ≈ 229 t water + 17 t GdLS + cryostat) | — | likely |
| GdLS acrylic-vessel envelope | R = 1.9 m, H = 3.9 m | (1.7, 3.4)–(2.1, 4.2) | uncertain |
| TPC active | R = 0.728 m, H = 1.456 m (as P013) | — | likely |
| muon zenith distribution | I ∝ cosⁿθ, n = 3.5 | 2.5–4.5 | likely |
| water + GdLS path to the TPC | 2.5 m | 2.0–3.0 | uncertain |
| effective attenuation length, water, E > 100 MeV (incl. cascade regeneration) | 1.0 m (nuclear interaction length 0.83 m) | 0.8–1.2 | uncertain |
| same, 50–100 MeV | 0.7 m | 0.55–0.9 | uncertain |
| n–p total cross section | 1.15 b (8 MeV), 0.94 (10), 0.69 (15), 0.48 (20), 0.38 (25), 0.30 (30), 0.17 (50), 0.075 (100), 0.043 (200), 0.034 b (1 GeV) | ±10 % | likely |
| n–¹⁶O nonelastic | 0.5 b (8–30 MeV), 0.4 (50), 0.3 b (≥ 100 MeV) | ±30 % | likely |
| GdLS (LAB) composition and cross sections | n_H = 6.3 × 10²², n_C = 3.8 × 10²² cm⁻³; σ_C,nonel(100 MeV) 0.22 b | — | likely |
| Gd capture time | ~30 μs in 0.1 % Gd LS; ~200 μs in water | — | likely |
| β-n emitters | ⁹Li 178 ms (51 %), ⁸He 119 ms (16 %), ¹⁷N 4.17 s (95 %), ¹⁶N 7.13 s (β-γ); ⁹Li yield in water 1.9 × 10⁻⁷ /(μ g cm⁻²); E_n ≤ 2 MeV | — | likely |
| ¹³⁷Xe | 3.82 min, β⁻ only (no neutron emission) | — | certain |

## 3. Muon rates and the 41/127-minute gaps

For I(θ) = I₀ cosⁿθ the flux through a horizontal plane is Φ_h = 2πI₀/(n+2); the rate through a vertical cylinder is
R = ∫ I(θ) A_proj(θ) dΩ with A_proj = πR² cosθ + 2RH sinθ, i.e. R = 2πI₀[πR²/(n+2) + 2RH ∫₀^{π/2} cosⁿθ sin²θ dθ].
Intervals between muons are exponential, P(gap ≥ t) = e^{−Rt}.

| volume | rate (h⁻¹) [range over Φ_μ, n] | top / sides | mean interval | P(gap ≥ 41 min) | P(gap ≥ 127 min) |
|---|---|---|---|---|---|
| water tank (3.8 m, 5.9 m) | 14.0 [11.2–15.4] | 8.66 / 5.34 | 4.3 min | 7.0 × 10⁻⁵ | 1.4 × 10⁻¹³ |
| GdLS envelope (1.9 m, 3.9 m) | 3.93 [3.1–4.4; geometry 3.1–4.7] | 2.16 / 1.77 | 15.3 min | 0.068 [0.039–0.119] | 2.4 × 10⁻⁴ |
| TPC active (0.728 m, 1.456 m) | 0.57 [0.45–0.63] | 0.32 / 0.25 | 105 min | 0.68 | 0.30 |

Interpretation. The 127-min TPC gap is entirely typical (P = 0.30 of gaps are longer). The 41-min "OD muon" gap is a typical-to-long
gap (7 %, range 4–12 %) if the OD muon tag effectively selects muons crossing the scintillator/cryostat region, but it is impossibly long
(7 × 10⁻⁵) for muons crossing anywhere in the 7.6 m water tank. Since a muon crossing metres of water emits ~10⁵ Cherenkov photons and is
certainly visible to the OD PMTs, we infer that LZ's muon classification carries an energy threshold well above water-Cherenkov light, i.e.
it is effectively a GdLS-crossing tag; either way the 41 min carry no information about the event. Also: MUSUN's 2.1 × 10⁹ muons per
1200 yr = 0.0555 μ s⁻¹, which at Φ_μ = 5.3 × 10⁻⁹ corresponds to a generation surface of ≈ 1050 m² (a ~30 m box around the cavern) —
consistent with the usual MUSUN set-up.

Physical relevance windows. A neutron thermalises and is captured in < 1 ms (Gd ~30 μs, water ~200 μs), so only muons within ≲ 1 ms of
the S1 could be causally related: P(tank-crossing muon within 1 ms) = 3.9 × 10⁻⁶ (2.3 × 10⁻⁶ within the 600 μs delayed window), and no
such muon was tagged (the event is in the science sample). A muon 41 min earlier is unrelated by 6 orders of magnitude in time scale.

LZ's UL decomposed. Zero signal-like events in 1200 yr → Poisson 90 % UL 2.303 events / 1200 yr; scaled to 220 d (0.6023 yr): 1.156 × 10⁻³;
× (1 − 0.80) untagged = 2.31 × 10⁻⁴; × 2 (stated yield uncertainty) = 4.62 × 10⁻⁴ — numerically the paper's 4.6 × 10⁻⁴. We therefore read
LZ's number as "90 % UL × untagged fraction × factor-2 yield safety".

## 4. Neutron production and escape from the rock

Production per unit wall area per cm of depth: Φ_μ Y_n ρ = 5.3 × 10⁻⁹ × 3.5 × 10⁻⁴ × 2.7 = 5.0 × 10⁻¹² n cm⁻³ s⁻¹. Within a 2 m shell
(the assignment's bookkeeping volume) that is 1.0 × 10⁻⁹ n cm⁻² s⁻¹ = 0.87 n m⁻² d⁻¹; for an illustrative 700 m² of cavern rock surface,
606 neutrons/day are produced within 2 m of the rock face, 105/day of them above 8.2 MeV. Most never emerge: the escape depth for a
neutron that must still be above 8.2 MeV is set by elastic moderation on Si/O (λ_tot 7–8 cm, ~9 % energy loss per collision) and
nonelastic removal (λ_nonel 14 cm at 14 MeV, ~35 cm at 100 MeV), giving λ_esc ≈ 20 / 30 / 40 cm for 8–30 / 30–100 / > 100 MeV.

Spectrum fractions (central parametrisation): f(> 1 MeV) = 0.75, f(> 8.2) = 0.173, f(> 10) = 0.160, f(> 30) = 0.092, f(> 50) = 0.060,
f(> 100) = 0.030, f(> 300) = 0.0092, f(> 1 GeV) = 0.0020. Corners (f_hard, knee): f(> 8.2) = 0.107–0.251, f(> 100) = 0.013–0.069.

Direction factor: 0.5 for the quasi-isotropic 8–30 MeV part (half the escaping neutrons head into the rock), rising logarithmically to 1
at ≥ 300 MeV (cascade neutrons are aligned with the downward muon; the ceiling is where the muon flux enters).

Current out of the rock above 8.2 MeV: J = Φ_μ Y_n ρ (dN/dE) λ_esc(E) g_dir(E) integrated = 1.52 × 10⁻¹¹ cm⁻² s⁻¹ (> 100 MeV: 5.3 × 10⁻¹²).
Cross-check against the whole cavern: with a view factor ≈ 1 in an enclosed cavity, J at the tank ≈ J at the wall.

## 5. Entry into the water tank and geometry

Tank top area 45.4 m², side 140.9 m²; the sides receive obliquely and partly shadowed flux, so A_eff = top + ½ side = 115.8 m².
Neutrons above 8.2 MeV entering the tank in 220 d: **334** (89.6 at 8–30 MeV, 125.6 at 30–100, 116.5 above 100 MeV).
Fraction aimed at the TPC: isotropic-current ratio A_TPC/A_tank = 0.043, downward-current ratio (πR²) = 0.037; adopted f_geo = 0.040
→ 13.4 neutrons aimed at the TPC.

## 6. Attenuation in water + GdLS (path 2.5 m, range 2–3 m)

Below 50 MeV the removal cross section is Σ = n_H σ_H(E) × (8.2/E) + n_O σ_O,nonel(E): for isotropic centre-of-mass n–p scattering E′ is
uniform on [0, E], so an H collision drops the neutron below 8.2 MeV with probability 8.2/E. λ_rem = 1/Σ (capped at 0.7 m). For 50–100 MeV
we use λ = 0.7 m and above 100 MeV λ = 1.0 m; both exceed the 0.83 m interaction length because hadronic cascades regenerate forward
neutrons (recalled/uncertain).

| E (MeV) | λ_rem (cm) | S(2.5 m) |
|---|---|---|
| 10 | 14.6 | 3.9 × 10⁻⁸ |
| 15 | 23.8 | 2.8 × 10⁻⁵ |
| 20 | 33.5 | 5.7 × 10⁻⁴ |
| 30 | 45.1 | 3.9 × 10⁻³ |
| 50 | 70 | 0.028 |
| ≥ 100 | 100 | 0.082 |

Sanity: the "every H collision halves the energy" picture gives λ_tot(10 MeV) = 8.9 cm → 28 collisions in 2.5 m → 2⁻²⁸ ≈ 3 × 10⁻⁹,
consistent with the 10 MeV row. Neutrons reaching the LXe still above 8.2 MeV in 220 d: **0.49** (0.002 from 8–30 MeV, 0.10 from 30–100,
0.38 from > 100 MeV): the answer is set by the > 50 MeV cascade component.

## 7. Single-scatter probability (P013 mono-energetic MC, per neutron entering the LXe)

| E (MeV) | P(SS in ROI) | P(SS in 200–270 keV) | F |
|---|---|---|---|
| 8.5 | 4.6 × 10⁻³ | 2.5 ± 0.3 × 10⁻⁴ | 0.054 |
| 10 | 4.4 × 10⁻³ | 2.6 ± 0.3 × 10⁻⁴ | 0.060 |
| 15 | 5.1 × 10⁻³ | 2.2 ± 0.3 × 10⁻⁴ | 0.042 |
| 50 | 6.5 × 10⁻³ | 1.2 ± 0.2 × 10⁻⁴ | 0.018 |
| 100 | 7.2 × 10⁻³ | 1.0 ± 0.2 × 10⁻⁴ | 0.014 |
| 200 | 9.0 × 10⁻³ | 1.5 ± 0.2 × 10⁻⁴ | 0.017 |

Interpolated in log E; above 200 MeV the 200 MeV values are used (P013's spectral MC shows the SS fraction changes slowly there).

## 8. Combined expectation (220 live days)

With f_untag = 0.20 (LZ): untagged SS of any energy **8.0 × 10⁻⁴** (LZ 90 % UL 4.6 × 10⁻⁴; ratio 1.7 — our chain is slightly above LZ's
own bound, i.e. if anything an overestimate); untagged SS in 200–270 keV **1.33 × 10⁻⁵**; effective F(200–270 | SS) = 0.017.

Sensitivity (one-at-a-time; `sensitivity.csv`):

| parameter | low → high | N_win | span |
|---|---|---|---|
| f_untag 0.04–0.24 | | 2.7 × 10⁻⁶ – 1.6 × 10⁻⁵ | ×6.0 |
| L_path 3.0–2.0 m | | 7.7 × 10⁻⁶ – 2.3 × 10⁻⁵ | ×3.0 |
| knee 30–100 MeV | | 9.0 × 10⁻⁶ – 2.2 × 10⁻⁵ | ×2.4 |
| λ_att(>100 MeV) 0.8–1.2 m | | 8.3 × 10⁻⁶ – 1.9 × 10⁻⁵ | ×2.3 |
| λ_esc | | 9.8 × 10⁻⁶ – 2.0 × 10⁻⁵ | ×2.0 |
| Y_n 2.5–5 × 10⁻⁴ | | 9.5 × 10⁻⁶ – 1.9 × 10⁻⁵ | ×2.0 |
| f_hard 0.2–0.4 | | 8.9 × 10⁻⁶ – 1.8 × 10⁻⁵ | ×2.0 |
| f_geo 0.037–0.065 | | 1.2 × 10⁻⁵ – 2.2 × 10⁻⁵ | ×1.8 |
| λ_att(50–100) | | 1.2 × 10⁻⁵ – 1.5 × 10⁻⁵ | ×1.3 |
| Φ_μ | | 1.1 × 10⁻⁵ – 1.4 × 10⁻⁵ | ×1.25 |

Log-quadrature of the half-spans: 1σ factor 4.0 → 68 % range 3.3 × 10⁻⁶ – 5.4 × 10⁻⁵. All-low / all-high corners: 1.4 × 10⁻⁷ / 1.9 × 10⁻⁴
(any energy: 8.5 × 10⁻⁶ / 1.1 × 10⁻²). The dominant uncertainties are the untagged fraction, the water path and the shape/attenuation of the
> 50 MeV component; the yield itself (LZ's factor 2) is only the fifth-largest.

## 9. Untagged fraction: a crude independent estimate

A neutron that makes the single FV scatter must cross 61 cm of GdLS twice (in and out) and the Skin twice, then thermalise. Visible prompt
GdLS light comes from n–p elastic recoils and C nonelastic interactions: P(no prompt light per crossing) = exp[−(n_H σ_H + n_C σ_C,nonel) × 61 cm]
= 0.31 (50 MeV), 0.45 (100), 0.51 (300), 0.53 (1 GeV). Skin: two ~6 cm LXe crossings with λ_tot = 19–29 cm; P(no Skin tag) = 0.54–0.81
(all or half of scatters tagged). Delayed miss (neutron leaves without a Gd capture within 600 μs, or capture γ below 200 keV): 0.3 (guess).
Product: 0.016–0.021 (50 MeV), 0.034–0.046 (100), 0.05–0.07 (≥ 300 MeV) — i.e. 2–7 %, versus LZ's 20 ± 4 %. LZ's figure is thus conservative
(it may include neutrons entering through the less-instrumented top/bottom); with f_untag = 0.04 our window expectation falls to 2.7 × 10⁻⁶
and the any-energy SS to 1.6 × 10⁻⁴, below LZ's UL.

Muon-accompanied neutrons: a muon crossing the OD (14 h⁻¹) is seen within ~30 ns; all its neutrons arrive within μs and are prompt- or
delayed-vetoed; muons crossing the rock beside the tank are the "rock neutron" case above, and their showers often enter the OD (LZ's 80 %
is for the sub-case where nothing else triggers, so the overall untagged fraction is lower still).

## 10. Delayed emitters: why 41 minutes cannot matter

β-delayed neutron emitters produced by muon spallation have T½ ≤ 7 s: 41 min is 13 800 (⁹Li), 20 700 (⁸He), 590 (¹⁷N), 345 (¹⁶N) half-lives.
Production is also tiny: ⁹Li in the 229 t of water in 220 d = 3.9 × 10⁻³ μ s⁻¹ × 500 g cm⁻² × 1.9 × 10⁻⁷ × T = 7.0 nuclei, 3.6 β-n decays,
and β-delayed neutrons carry ≤ 2 MeV → E_R,max = 0.0303 E_n ≤ 61 keV, far below 248 keV regardless of timing. In LXe, ¹³⁶Xe(n,γ)¹³⁷Xe
(3.8 min) is β⁻-only. No delayed channel connects a muon 41 min earlier to a 248 keV recoil.

## 11. Figures

- `figures/P049_fig1_spectra_attenuation.png` — left: production spectrum, current out of the rock (> 8.2 MeV) and the spectrum after 2.5 m
  of water (scaled); right: survival above 8.2 MeV versus water path for 10–500 MeV (100 and 500 MeV coincide, λ = 1.0 m), 2–3 m band shaded.
- `figures/P049_fig2_sensitivity.png` — one-at-a-time ranges of the window expectation; P013's 0.6–1.6 × 10⁻⁵ and LZ's 4.6 × 10⁻⁴ marked.
- `figures/P049_fig3_muon_intervals.png` — exponential waiting-time densities for the three volumes with 41 and 127 min marked.

## 12. Failed or abandoned

- First implementation of the H-removal term used (1 − 8.2/E) instead of 8.2/E per collision (understated removal at 10–30 MeV); fixed —
  the > 50 MeV component dominates, so the final numbers moved by < 4 %.
- The Mei–Hime analytic rock-neutron spectrum was not used: its coefficients could not be recalled reliably; replaced by the two-component
  parametrisation with a scanned hard fraction and knee.
- No attempt to model the cavern shape, the top/bottom OD tanks or the cryostat steel separately; all are inside the path-length range.

## 13. Extended discussion

Our independent chain reproduces LZ's simulated bound within a factor ~2 (8 × 10⁻⁴ vs UL 4.6 × 10⁻⁴ for SS of any energy) and gives
1.3 × 10⁻⁵ (3 × 10⁻⁶ – 5 × 10⁻⁵) untagged 200–270 keV single scatters in 220 d, in agreement with P013's 0.6–1.6 × 10⁻⁵ that was derived
from LZ's UL. The agreement is not circular: none of Φ_μ, Y_n, the spectrum, the escape depth or the attenuation is taken from LZ; only the
untagged fraction (20 %) and P013's SS probabilities are shared, and our own estimate of the untagged fraction is lower (2–7 %). Even the
all-high corner (1.9 × 10⁻⁴) leaves muon-induced neutrons at least 5000 times short of one expected window event. The 41-min and 127-min
gaps are statistically unremarkable for GdLS- and TPC-crossing muons respectively and are physically irrelevant (thermalisation < 1 ms; no
β-n emitter survives 41 min; β-delayed neutrons cannot reach 248 keV).

## 14. References

1. LZ Collaboration, arXiv:2609.02823 (2026).
2. D.-M. Mei and A. Hime, Phys. Rev. D 73, 053004 (2006) — muon flux, mean energy and neutron yield versus depth.
3. V. A. Kudryavtsev, Comput. Phys. Commun. 180, 339 (2009) — MUSIC/MUSUN.
4. H. M. Araújo, V. A. Kudryavtsev, N. J. C. Spooner, T. J. Sumner, Nucl. Instrum. Meth. A 545, 398 (2005) — Geant4/FLUKA neutron yields.
5. N. Abgrall et al. (Majorana), Astropart. Phys. 93, 70 (2017) — muon flux at the 4850 ft level of SURF.
6. V. Pěč, V. A. Kudryavtsev, H. M. Araújo, T. J. Sumner, Eur. Phys. J. C 84, 481 (2024) — muon-induced background in a next-generation LXe experiment.
7. Corpus: P013 (neutron origins; SS probabilities), P004, P022 (method/format).

## 15. Tools and provenance

Mirrors `output/provenance/P049.json`. Tools: Read (PAPER_GUIDE, dossier, ledger, P013/P004/P022, tex l. 80–89, 140–179, 296–327, 785–800,
P013_results.json, three figures), Bash (ledger/tex grep; P013 extraction; lzcommon inspection; P013 script grep; three script runs; word counts),
Write (script, details, provenance, paper), Edit x7 (removal-probability fix; figure label; four paper trims; provenance tool counts). Software: python 3.12.13, numpy 2.5.3,
scipy 1.18.1 (integrate.quad), pandas 3.0.5, matplotlib 3.11.2, common/lzcommon.py (LZ dict). 20 recalled items listed in §2.3.
Hand derivations: cylinder rate integral, exponential gap statistics, n–p removal probability, Poisson UL decomposition.
