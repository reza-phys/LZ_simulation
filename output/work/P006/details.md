# P006 — Sixteenth of June: the event date under inelastic dark matter and under background

Research record (simulated arXiv date 2026-09-03). Author profile: astroparticle theorists working on halo modelling and annual modulation. Category HALO; astro-ph.CO (cross-list hep-ph).

## 1. Motivation and framework

The LZ paper (arXiv:2609.02823) reports one nuclear-recoil-like event at 248 ± 23 ± 23 keV recorded 16 June 2023 21:22:39 UTC, and remarks in its Theory paragraph that inelastic models "can generate interaction rates with significant annual modulation peaking around June 2". The dossier (§3, point 3) lists the timing as a point in favour of dark matter (DM). We ask how much the *date alone* actually shifts the odds. For a single event the relevant quantity is the ratio of the normalised time densities

LR(t_ev) = p(t_ev | model) / p(t_ev | flat),   p(t | M) = R_M(t) L(t) / ∫_run R_M L dt,   p(t | flat) = L(t) / ∫_run L dt,

where R_M(t) is the model's rate in the region of interest (ROI) on day t and L(t) is the livetime density. Backgrounds are taken time-flat (calibration-related transients are not modelled here; that is P010's business). With uniform livetime, LR = R_M(t_ev) / ⟨R_M⟩_run, the rate on the event day divided by the livetime-weighted run mean.

The date is one draw from p(t|M). Its expected information for discriminating M from flat is the Kullback–Leibler divergence E_M[ln LR] = KL(p_M || p_flat), and its scatter is Var_M[ln LR]. For N events the sum Σ ln LR has mean N·E and variance N·Var; a 3σ discrimination in the frequentist sense (the model's expected statistic sits 3 sd of the flat-hypothesis distribution above the flat expectation) needs

N_3σ = 9 Var_flat[ln LR] / (E_M[ln LR] − E_flat[ln LR])²,   with E_flat[ln LR] = −KL(p_flat || p_M).

We also quote the Asimov-like N = 4.5 / KL(p_M||p_flat) (i.e. √(2 N KL) = 3), and verify with toy Monte Carlo.

## 2. Inputs

| Input | Value | Source |
|---|---|---|
| Run window | 27 March 2023 – 1 April 2024 = 371 calendar days | LZ Data Analysis paragraph (tex line 111) |
| Livetime | 220 live days → duty cycle 0.593 | same |
| Event time | 16 June 2023 21:22:39 UTC = 81.89 d after run start; day-of-year 167.89 | LZ Data Analysis paragraph (line 165) |
| Calibrations | AmBe deployment 8 June 2023, "one of three" NR calibrations; ⁵⁷Co 25 min before the event | LZ Discussion (lines 300–301) |
| ROI | NR 5.4–270 keV (50% efficiency points), plateau 0.96 | LZ paper; Fig. S2 |
| Efficiency roll-off | 0.92 @250, 0.75 @260, 0.50 @269.9, 0.25 @280, 0.09 @290 keV | read by eye from Fig. S2 inset (inputs/figures_png/FigS2_efficiency_werror.png) |
| Halo | Baxter-2021 SHM: v₀ = 238, v_esc = 544 km/s, v_⊙,pec = (11.1, 12.2, 7.3), ρ₀ = 0.3 GeV/cm³ | `lzcommon` constants (recalled, certain) |
| Earth orbital speed | 29.8 km/s; 15 km/s projection on solar motion; peak 2 June (day 153) | `lzcommon.v_earth_kms` (recalled, certain) |
| Spectra | WimPyDD 2.0.4 `diff_rate`, natural Xe, O1 isoscalar c⁰ = 1 GeV⁻² (arbitrary; only shapes matter), O11 isoscalar as q²-suppressed proxy; inelastic via `delta` | `lzcommon.wd_rate`, `wd_hamiltonian`, `wd_halo(day_of_year=…)` |
| Kinematics | `lzcommon.delta_max_kev`, `vmin_kms` | — |
| Independent Earth-velocity checks | wimprates 0.5.0 `v_earth(j2000)`; WimPyDD `v_earth_sun(λ)` with λ = 2π(d−80)/365 | packages |
| Bayes factor to combine with | B₁₀ = 7–37 after Occam penalty (100–500 before) | P001 headline |

Efficiency model used: ε(E) = 0.96 · [1 + e^{−(E−5.4 keV)/1 keV}]⁻¹ · Φ((269.9 keV − E)/15 keV). The Gaussian-CDF roll-off reproduces the five inset readings above to ±0.02. NR energy resolution is not applied (the modulation of the ROI-integrated rate is insensitive to a few-keV smearing).

## 3. Method

1. Earth speed |v_E(d)| for d = 1…365.25 in steps of 0.25 d with three models: lzcommon cosine, WimPyDD's `v_earth_sun`, wimprates `v_earth`.
2. For each model M and 42 days d (every 10 days plus 152.5, 153, 160, 167.89, 335) compute dR/dE on a 120-point energy grid (1–320 keV) with `wd_halo(day_of_year=d)`, multiply by ε(E), integrate over 5.4–270 keV (ROI) and 200–270 keV (the event's window); also dR/dE at 248 keV. Also compute the rate with the annual-average halo (`wd_halo()`).
3. Periodic cubic-spline interpolation in day-of-year (clipped at zero), evaluated on a 3-hour grid over the 371-day run.
4. Livetime models: (i) uniform, L = 220/371; (ii) three one-week gaps removed and the remaining density rescaled to 220 live days. **Assumption** (not published): gaps 10–17 April 2023 (D-D, guess), 5–12 June 2023 (anchored on the AmBe deployment of 8 June), 9–16 October 2023 (guess).
5. Time PDFs, LR at t_ev, P(|t − t_peak| < 30 d), KL divergences and variances, N_3σ, toy MC (4000 toys) for the ROI/uniform case.
6. δ_max(E_R, m, v_E + v_esc) on 16 June, at the annual-mean speed and in December, for E_R = 202, 248, 294 keV and m = 400, 1000, 4000 GeV; the number of days per year on which a 248 keV recoil is kinematically allowed for δ = 350–390 keV at m = 1000 GeV.
7. Post-processing (`P006_postprocess.py`): frequentist p-value of the date under flat, p_flat = P_flat[ln LR(t) ≥ ln LR(t_ev)] (fraction of uniform livetime whose predicted rate is at least the event-day rate), and the position of the observed ln LR within the model and flat distributions.

Models: O1 elastic m = 100 and 1000 GeV; O11 elastic m = 1000 GeV (momentum-suppressed proxy for dipole-like spectra; a full L10 mapping was not attempted); O1 inelastic m = 1000 GeV, δ = 100, 200, 250, 300, 350, 380 keV.

## 4. Results

### 4.1 Earth speed and modulation peak (`earth_speed_vs_doy.csv`, `figures/earth_speed.png`)

| model | peak day | peak date | amplitude (km/s) | mean (km/s) | v_E(16 June 21:22) | v_max = v_E + 544 |
|---|---|---|---|---|---|---|
| lzcommon cosine | 153.0 | 2 June | 15.00 | 250.55 | 265.06 | 809.06 |
| WimPyDD v_earth_sun | 152.0 | 1 June | 14.54 | 252.10 | 265.92 | 809.92 |
| wimprates v_earth | 152.75 | 1 June | 14.55 | 252.11 | 265.98 | 809.98 |

The three agree to ≤ 1.9 km/s everywhere; WimPyDD and wimprates agree to 0.06 km/s (same orbital model). The event is 15 days past the peak, at 0.97–0.99 of the peak speed excess. The rate calculations use WimPyDD's model.

### 4.2 Modulation of the in-ROI rate (`P006_results.csv`, `rate_vs_doy_*.csv`, `figures/rate_vs_doy.png`)

Modulation fraction f = (R_max − R_min)/(R_max + R_min); phase from a first-harmonic fit; peak day from the spline.

| model | f (5.4–270 keV) | f (200–270 keV) | peak day | cos-fit phase | June/Dec ratio (ROI) | R(16 June)/R(mean-v halo) |
|---|---|---|---|---|---|---|
| O1 elastic 100 GeV | 0.0070 | 0.200 | 334.5 (December) | 334.5 | 0.986 | 0.992 |
| O1 elastic 1000 GeV | 0.0244 | 0.0347 | 334.5 (December) | 334.5 | 0.953 | 0.974 |
| O11 elastic 1000 GeV | 0.0205 | 0.0349 | 334.5 (December) | 334.5 | 0.961 | 0.978 |
| O1 δ = 100 keV | 0.104 | 0.089 | 152 | 151.9 | 1.226 | 1.114 |
| O1 δ = 200 keV | 0.244 | 0.162 | 152 | 151.9 | 1.631 | 1.288 |
| O1 δ = 250 keV | 0.270 | 0.221 | 152 | 151.9 | 1.723 | 1.324 |
| O1 δ = 300 keV | 0.407 | 0.347 | 152 | 151.9 | 2.342 | 1.508 |
| O1 δ = 350 keV | 0.901 | 0.853 | 152 | 151.9 | 18.8 | 2.832 |
| O1 δ = 380 keV | 1.000 | 1.000 | 152 | 151.9 | ∞ (zero in winter) | 6.064 |

Notes. (a) For the elastic heavy-WIMP models the ROI-integrated rate is dominated by low-energy recoils with v_min ≲ 200 km/s, where the modulation is phase-reversed (December maximum; the classic low-v_min phase flip) — so the June date is *slightly disfavoured* for elastic SI at 1000 GeV. In the 200–270 keV window alone the 100 GeV elastic rate does peak in June with f = 0.20 because v_min(248 keV, 100 GeV) = 671 km/s. (b) The June/December ratio 2.34 for δ = 300 keV is consistent with the dossier's Helm-based 2.5. (c) For δ = 380 keV the 248 keV recoil is kinematically forbidden for 14% of the year (see §4.5), and the in-ROI rate vanishes for ~30% of the year.

### 4.3 Likelihood ratio of the date (`P006_results.csv`, `figures/LR_vs_delta.png`)

| model | LR ROI, uniform | LR ROI, gaps | LR 200–270 keV, uniform | P(±30 d of peak | M) | P(±30 d | flat) |
|---|---|---|---|---|---|
| O1 elastic 100 GeV | 0.993 | 0.993 | 1.195 | 0.163 | 0.162 |
| O1 elastic 1000 GeV | 0.977 | 0.976 | 1.032 | 0.166 | 0.162 |
| O11 elastic 1000 GeV | 0.981 | 0.980 | 1.033 | 0.165 | 0.162 |
| O1 δ = 100 keV | 1.099 | 1.101 | 1.085 | 0.178 | 0.162 |
| O1 δ = 200 keV | 1.239 | 1.246 | 1.157 | 0.200 | 0.162 |
| O1 δ = 250 keV | 1.267 | 1.274 | 1.216 | 0.205 | 0.162 |
| O1 δ = 300 keV | 1.406 | 1.418 | 1.344 | 0.227 | 0.162 |
| O1 δ = 350 keV | 2.078 | 2.124 | 1.956 | 0.335 | 0.162 |
| O1 δ = 380 keV | 2.545 | 2.627 | 2.545 | 0.410 | 0.162 |

The livetime model changes LR by < 4% (gaps remove some winter livetime and a week just before the event). LR is smaller than R(16 June)/R(mean-velocity halo) (1.41 vs 1.51 for δ = 300 keV) because the run-averaged rate exceeds the rate at the mean Earth speed (the rate is convex in v_E; Jensen's inequality), so normalising to the run rather than to an average halo costs ~7%.

Frequentist view (`P006_postprocess.json`): the event-day rate is exceeded on only 32.0 of the 371 run days (the peak of 2023 falls inside the run; that of 2024 does not), so under flat backgrounds p_flat = P[ln LR ≥ observed] = 0.0862, i.e. 1.36σ one-sided — identical for every June-peaked model because it is a rank statistic. Under the δ = 300 keV model the same tail has probability 0.122; under δ = 380 keV, 0.224. The observed ln LR = 0.341 (δ = 300 keV) sits 1.04 sd above the model expectation and 1.27 sd above the flat expectation — a thoroughly unremarkable draw under either hypothesis.

### 4.4 Information content of the date (`P006_results.csv`, `figures/N3sigma_vs_delta.png`)

ROI, uniform livetime. E_M and Var_M are the mean and variance of ln LR for one event drawn from the model; E_flat, Var_flat for one background event.

| model | E_M [nats] | sd_M | E_flat | sd_flat | N_3σ (analytic) | N_3σ (√(2NKL)=3) | toy z at N_3σ | N_3σ, 200–270 keV |
|---|---|---|---|---|---|---|---|---|
| O1 elastic 1000 GeV | 1.5×10⁻⁴ | 0.017 | −1.5×10⁻⁴ | 0.017 | 3.0×10⁴ | 3.0×10⁴ | — | 1.5×10⁴ |
| O1 δ = 100 keV | 0.0027 | 0.073 | −0.0027 | 0.073 | 1696 | 1692 | 2.92 | 2278 |
| O1 δ = 200 keV | 0.0150 | 0.172 | −0.0152 | 0.175 | 303 | 300 | 3.29 | 692 |
| O1 δ = 250 keV | 0.0186 | 0.191 | −0.0189 | 0.195 | 246 | 243 | 2.92 | 370 |
| O1 δ = 300 keV | 0.0432 | 0.287 | −0.0450 | 0.304 | 107 | 104 | 2.78 | 149 |
| O1 δ = 350 keV | 0.304 | 0.623 | −0.428 | 1.045 | 18 | 15 | 3.29 | 22 |
| O1 δ = 380 keV | 0.604 | 0.525 | −∞ (forbidden days) | — | — | 7.4 | — | — |

The toy z-values (4000 toys, median model statistic against the flat distribution) agree with the analytic N_3σ within toy resolution (±0.3σ). For one event (N = 1) the toy z is 0.29 (δ = 300) and 0.66 (δ = 350). Reading: the date of one event carries ~0.04 nats of expected information for δ = 300 keV with a scatter of 0.29 nats — the single realised value (0.34 nats) is dominated by noise, and ~100 events (~20 for δ = 350 keV) are needed before dates alone discriminate at 3σ. Only for δ ≥ 380 keV, where the recoil is forbidden in winter, does a handful of events (≈ 7) suffice, because a single winter event would then falsify the model outright.

### 4.5 Kinematic bonus: δ_max on 16 June (`P006_summary.json` → `kinematics`, `allowed_days_248keV_m1000`)

δ_max(E_R, m) = (v_max/c)√(2 m_N E_R) − m_N E_R/μ, with v_max = v_E + v_esc and wimprates' v_E (16 June 265.98; annual mean 252.11; December 237.35 km/s):

| E_R (keV) | m (GeV) | δ_max 16 June | δ_max annual mean | δ_max December |
|---|---|---|---|---|
| 248 | 400 | 341.8 | 330.4 | 318.3 |
| 248 | 1000 | 387.3 | 375.9 | 363.8 |
| 248 | 4000 | 410.1 | 398.7 | 386.6 |
| 202 | 1000 | 374.1 | 363.8 | 352.8 |
| 294 | 1000 | 394.8 | 382.4 | 369.2 |

The June date raises δ_max by +11.4 keV relative to the annual mean and +23.5 keV relative to December (m = 1000 GeV). With the lzcommon cosine speeds the numbers are 386.6/374.6/362.3 keV (dossier). Days per year on which a 248 keV recoil is allowed at m = 1000 GeV: δ = 350 keV, all; 370 keV, 244 d; 380 keV, 143.5 d; 385 keV, 82 d; 390 keV, none. The event date is inside the allowed window for all δ ≤ 387 keV.

### 4.6 Combination with P001

P001 finds B₁₀ = 7–37 (after a ~14-fold Occam penalty) for the event against modelled backgrounds. Multiplying by the date LR gives 10–52 for δ = 300 keV and 15–77 for δ = 350 keV; for elastic SI at 1000 GeV it becomes 7–36. The date does not change the qualitative conclusion of any earlier paper.

## 5. Validation and robustness

- Earth-speed models: three independent implementations agree to ≤ 1.9 km/s (§4.1); the peak day is 1–2 June in all.
- June/December ratio for δ = 300 keV: WimPyDD 2.34 vs dossier Helm estimate 2.5 (different form factor and velocity model).
- Livetime: uniform vs three assumed one-week gaps changes LR by ≤ 4%; the 60-day window probability under flat is 60/371 = 0.162 in both.
- Toy MC reproduces the analytic N_3σ within ±0.3σ.
- Interpolation: 42-day grid with periodic cubic spline; for δ = 380 keV the spline is clipped at zero; the modulation fraction of exactly 1.0 confirms the clipping is benign.
- Efficiency: the ROI-integrated inelastic rates are dominated by 100–270 keV where the efficiency is flat at 0.96, so the roll-off model has little effect on the *shape in time*; it matters only through the 250–270 keV tail for δ ≥ 350 keV.
- Grid: the rates on the event day were computed exactly (day 167.89 is in the day grid), not interpolated.

## 6. Failed or abandoned approaches

- A wait loop using `pgrep` to detect the end of the 17-minute WimPyDD run failed (process listing is blocked in the sandbox); replaced by polling for the output file.
- No attempt was made to map the LZ Lagrangian L10 (magnetic dipole) onto WimPyDD couplings; O11 (q²-suppressed) was used as a proxy and behaves like elastic SI in time (f = 2%).
- Modelling calibration-related background transients (activation after AmBe) in the time PDF was left to P010; here backgrounds are flat.

## 7. Discussion

The event date is compatible with an inelastic-DM origin and mildly favours it over time-flat backgrounds: for the δ = 300–350 keV, m = 1000 GeV models that maximise LZ's local significance, the date multiplies the odds by 1.4–2.1; for elastic SI (any mass ≥ 100 GeV) it does essentially nothing (LR 0.98–0.99, because the ROI rate peaks in December). The date's information is small in expectation (0.04–0.3 nats) and noisy (sd 0.3–0.6 nats), so LR ≈ 1.4 is not evidence in any useful sense: 32 of 371 run days would have given at least this LR under background, and one in eight events under the model itself would fall on a *less* favourable date. This is the single-event version of the standard result that annual modulation searches need O(100) events. The genuinely useful consequence of the date is kinematic: it extends the accessible δ range at 248 keV by 11 keV (to 387 keV at 1000 GeV), which matters for the Higgsino-like and dark-photon models that other papers (P002, P007) scan out to δ_max.

## 8. Figures

- `figures/earth_speed.png` — |v_E| vs day of year for the three models; event date marked.
- `figures/rate_vs_doy.png` — in-ROI WimPyDD O1 rate (normalised to annual mean) for elastic 1000 GeV and δ = 200, 300, 350, 380 keV; dotted line = peak (day 152), dashed = 16 June.
- `figures/LR_vs_delta.png` — LR vs δ for ROI/uniform, ROI/gaps and 200–270 keV/uniform.
- `figures/N3sigma_vs_delta.png` — events needed for 3σ from dates alone vs δ.

## 9. Result tables

`P006_results.csv` (all quantities per model and livetime), `P006_summary.json` (Earth-velocity, kinematics, allowed days, toy check, results), `P006_postprocess.json` (date p-values), `rate_vs_doy_*.csv` (R_ROI, R_200–270, dR/dE(248) vs day), `earth_speed_vs_doy.csv`.

## 10. References

1. LZ Collaboration, arXiv:2609.02823 (2026).
2. D. Tucker-Smith and N. Weiner, Phys. Rev. D 64, 043502 (2001) — inelastic DM.
3. K. Freese, J. Frieman and A. Gould, Phys. Rev. D 37, 3388 (1988) — annual modulation, phase reversal at low v_min.
4. K. Freese, M. Lisanti and C. Savage, Rev. Mod. Phys. 85, 1561 (2013) — annual modulation review.
5. J. Bramante, P. J. Fox, G. D. Kribs and A. Martin, Phys. Rev. D 94, 115026 (2016) — inelastic frontier.
6. D. Baxter et al., Eur. Phys. J. C 81, 907 (2021) — recommended halo parameters.
7. S. Kang, S. Scopel, G. Tomar and J.-H. Yoon, Comput. Phys. Commun. 279, 108423 (2022) — WimPyDD.
8. Corpus: P000 dossier; P001 (Bayes factors).

## 11. Tools and provenance (mirrors provenance/P006.json)

- Agent tools: Read ×7 (PAPER_GUIDE.md; 00_evidence_dossier.md; lzcommon.py; WimPyDD/package.py lines 6266–6335; FigS2_efficiency_werror.png; two P006 figures), Bash ×13 (greps of the LZ tex, ledger and environment listing, WimPyDD/wimprates signature and timing probes, two script runs, file listings, wait loops), Write ×5, Edit ×1, ToolSearch ×1.
- Software: Python 3.12.13; numpy 2.5.3; scipy 1.18.1 (stats.norm, interpolate.CubicSpline, integrate via numpy.trapezoid); pandas 3.0.5; matplotlib 3.11.2; WimPyDD 2.0.4 (`streamed_halo_function`, `diff_rate`, `eft_hamiltonian`, `v_earth_sun`); wimprates 0.5.0 (`v_earth`, `j2000_from_ymd`); numericalunits; `common/lzcommon.py` (`wd_halo`, `wd_hamiltonian`, `wd_rate`, `v_earth_kms`, `delta_max_kev`, `vmin_kms`, constants).
- Scripts: `output/code/P006_event_date.py` (`.venv/bin/python output/code/P006_event_date.py`, ~17 min); `output/code/P006_postprocess.py` (~5 s).
- Local inputs: LZ tex lines 50, 111, 165, 300–302, 452–460; `inputs/figures_png/FigS2_efficiency_werror.png`; dossier §§1–3, 6; `output/papers/P001.md` headline; `output/work/dossier/dossier_numbers.json` (cross-check of δ_max).
- Recalled knowledge (5): Earth orbital speed 29.8 km/s and 15 km/s projection, peak ≈ 2 June (certain); phase reversal of the modulation for v_min ≲ 200 km/s (certain); KL/expected-log-LR statistics and the N_3σ scaling (certain); Baxter-2021 SHM parameters via lzcommon (certain); references 2–7 above (likely for exact journal details).
- WimPyDD-generated files: none (no new files under WimPyDD/ after the run).
- Datasets: none. Data requests: none.
