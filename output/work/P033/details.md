# P033 — Geometric probability of a wall-MSSI topology with a 12 keV Compton scatter 27 cm inside the TPC: a photon-transport Monte Carlo

Simulated date 2026-09-09 · category BKG · physics.ins-det · author profile: detector-simulation specialists.
Script: `output/code/P033_mssi_geometry_mc.py` (stages `validation`, `grid --source {wall,bottom,top,cathode}`, `bulk`, `variants`, `summary`, `post`; run from the root with `.venv/bin/python`). Machine-readable results: `P033_summary.json`, `P033_post.json`, `P033_validation.json`, `P033_runs.csv`, `P033_kn_table.{csv,json}`, `P033_attenuation.json`, per-run `runs/*.json|npz`; figures in `figures/`.

## 1. Motivation and question

LZ's supplement decomposes the 248 keV candidate, if it were an MSSI, into a first scatter of 12 ± 2 keV at the event position (S1c 69 ± 17 phd, S2c 9268 phd) plus an S1-only second deposit of ≈ 471 phd: 77 ± 7 keV in the wall charge-dead layer or 204 (+65/−38) keV in the reverse-field region (RFR) (tex l. 733–734). The paper states that a ~12 keV scatter more than 20 cm from the TPC boundaries "is unlikely for any MSSI category" because the shallow-angle Compton scatter forces the γ to traverse > 60 cm of LXe (l. 735). P004 quantified this with a one-dimensional toy radial pdf and obtained a position penalty f_pos ∈ [10⁻⁴, 0.12] (central 0.003–0.05), raising the required wall-MSSI mismodelling factor from k ≈ 620 to ≳ 5 × 10³. That range spans three orders of magnitude. Here we replace the toy by a three-dimensional photon-transport Monte Carlo of the actual topology (one Compton deposit of 8–16 keV in the fiducial volume, all other deposits in a charge-dead region with the right total energy, nothing else in the active volume) and ask: what fraction of such histories put the FV deposit ≥ 25 cm from the wall and ≥ 20 cm above the cathode (the event is at 26.9 cm and 26.4 cm), how does that fraction depend on γ energy and source location, how does it compare with the concentration near the walls that LZ's Fig. S5 implies, and what does it do to P004's k_required?

## 2. Model

### 2.1 Geometry (cm)
Active LXe cylinder R = 72.8 (true wall), height H = 145.6 (z = 0 cathode, z = H gate); both recalled from Fig. 3/S5a axes (reconstructed r² axis ends at 72.8², drift axis 0–1049 μs ↔ z 145.6–0; P022; reliability likely). Charge-dead wall shell R − T_SHELL ≤ r < R with T_SHELL = 0.3 cm (paper: scalloped, "up to 3 mm"; the model's dead volume is 0.3 % of the active volume, which corresponds to a uniform shell of 0.11 cm — run as a variant). RFR: −13.75 ≤ z < 0, fully charge-dead (l. 186). Fiducial volume (simplified): r ≤ R − 8.0 (minimum stand-off to the true wall, l. 137), 9.0 ≤ z ≤ 132.8 (l. 139); variant with the mean stand-off 10.7 cm. Everything outside the LXe (r > R, z > H, z < −13.75) is "escape"; back-scattering from PTFE/skin/PMTs is not modelled.

### 2.2 Photon physics in LXe (ρ = 2.86 g cm⁻³, recalled)
- Compton scattering on free electrons, Klein–Nishina implemented exactly: total σ_KN(k) = 2πr_e²{(1+k)/k² [2(1+k)/(1+2k) − ln(1+2k)/k] + ln(1+2k)/(2k) − (1+3k)/(1+2k)²}, k = E/m_e c²; dσ/dT = π r_e² m_e c²/E² [ε + 1/ε − sin²θ], ε = E′/E, T = E − E′, cos θ = 1 − (E/E′ − 1) m_e c²/E. Numerical ∫dσ/dT over [0, T_max] reproduces σ_KN to 4 × 10⁻⁹. Analog sampling by Kahn's rejection method (validated: χ²/dof = 0.92 against dσ/dT in 40 bins with 4 × 10⁵ samples; sampled fraction with T ∈ [8, 16] keV at 1 MeV 0.00974 vs analytic 0.00960).
- Photoabsorption: local deposit of E. Mass attenuation coefficients μ_pe/ρ recalled from an XCOM-like table for Xe (values at 34.6 (K edge), 50, 100, 200, 300, 500, 1000, 2000, 3000 keV: 26, 9.7, 1.45, 0.21, 0.068, 0.0165, 0.0028, 0.00068, 0.00032 cm² g⁻¹; reliability likely, ±30 %). μ_C = n_e σ_KN (n_e from Z = 54, A = 131.29).
- Pair production: treated as local absorption (any pair event ruins the energy windows anyway); μ_pp/ρ = 0.0015/0.0040/0.0072 cm² g⁻¹ at 1.5/2.0/2.615 MeV (recalled, uncertain). The first grid runs at ≤ 1764 keV used a table 2× larger (0.003/0.0075/0.014); this changes μ by ≤ 3 % at ≤ 1461 keV and those runs were not repeated; the 1764 and 2615 keV runs and their variants were rerun with the corrected table.
- Resulting total attenuation lengths λ = 1/μ (cm): 0.22 (100 keV), 0.36 (122), 1.13 (200), 2.19 (295), 2.78 (352), 3.97 (500), 4.63 (609), 6.34 (1000), 6.93 (1173), 7.65 (1461), 8.21 (1764), 8.54 (2000), 9.06 (2615). Compton fraction of μ: 0.08 (100), 0.32 (200), 0.66 (352), 0.87 (609), 0.95 (1000), 0.94 (1461), 0.80 (2615). These lie between P004's table (0.30 at 122, 1.0 at 200, 3.7 at 500, 6.0 at 1000, 8.3 at 1764, 9.8 at 2615) and P029's shorter bracket (2.9 cm at 375 keV); the "long-λ" bracket of P004/P029 is run as a variant with all μ × 0.8.
- Photons below 25 keV are absorbed locally (range ≪ 1 mm). Xe K-fluorescence escape, Doppler broadening and electron binding are neglected (Compton at 8–16 keV energy transfer on bound electrons is slightly suppressed relative to free-electron KN; this makes our 12 keV probabilities conservative-high).
- The energy deposits in each region are converted to an S1 estimate with the paper's own conversions: 69 phd/12 keV for the FV deposit, 471 phd/77 keV = 6.1 phd keV⁻¹ in the wall shell, 471 phd/204 keV = 2.3 phd keV⁻¹ in the RFR (l. 733–734).

### 2.3 Sources
(a) **wall**: detector components behind the TPC wall — emission from r = R, z uniform in (0, H), direction cosine-law (Lambertian) about the inward normal (default; isotropic-hemisphere variant). (b) **bottom**: plane z = −13.75 (bottom grid/PMT array), Lambertian upward. (c) **top**: plane z = H (gate/anode/top array), Lambertian downward. (d) **cathode**: plane z = 0 (cathode wires), isotropic. (e) **bulk ²¹⁴Pb**: decay vertex uniform in the whole LXe (r < R, −13.75 < z < H; 7.6 t), γ isotropic; the β deposits its full energy locally at the vertex (range ≲ 1 mm): if the vertex is in the FV that deposit *is* the FV deposit (so the β itself must have 8–16 keV — P_β(8–16 keV) = 1.96 % for the 672 keV branch and 1.73 % for the 729 keV branch, allowed shape × non-relativistic Fermi function, Z = 83); if it is in a dead region it adds to the dead deposit; if it is in the active non-FV LXe the history is discarded (S2 outside the FV). Branches (recalled, likely): β to the 352 keV level (endpoint 672 keV) 42.5 %, to the 295 keV level (729 keV) 40 %; ground state (no γ, not MSSI) ≈ 9 %. The 242 + 53 keV cascade of the 295 level is neglected (a 53 keV local deposit destroys the 12 keV window anyway).
Lines for the surface sources: 352, 609, 1120, 1173, 1332, 1461, 1764, 2615 keV (²¹⁴Pb/²¹⁴Bi, ⁶⁰Co, ⁴⁰K, ²⁰⁸Tl; energies certain). A generic component mix weights lines by recalled per-decay intensities (0.356, 0.455, 0.149, 0.30, 0.30, 0.107, 0.153, 0.356; uncertain) assuming equal U-late/Th/K activities and ⁶⁰Co at 0.3; all results are also given per line. The four surface sources are combined with equal numbers of emitted photons ("equal mix"); the source dependence is quoted as the systematic.

### 2.4 Classification of a history (after transport)
- **wall12** (the event's wall-MSSI topology): exactly one FV deposit with 8 ≤ E_FV ≤ 16 keV, dead-shell total 60 ≤ E_wall ≤ 95 keV, E_RFR < 1 keV, no other active-volume deposit.
- **rfr12**: exactly one FV deposit 8–16 keV, 150 ≤ E_RFR ≤ 260 keV, E_wall < 1 keV.
- **mixed12**: 8–16 keV FV deposit with deposits in both dead regions and S1_est ∈ [480, 600] phd.
- **ss_roi** ("ordinary single-scatter ER"): exactly one deposit anywhere in the LXe, in the FV, 1.5 ≤ E_FV ≤ 75 keV (ROI ER energies).
- **wall_roi / rfr_roi** (general MSSI in the WS ROI): one FV deposit ≥ 1.5 keV plus dead deposits in one region with 3 ≤ S1_est ≤ 600 phd.
- **wall_any / rfr_any**: as above without the S1 cap (validation only).
Position class ("event class"): distance to the true wall d = R − r_FV ≥ 25 cm and z_FV ≥ 20 cm. Also recorded: d ≤ 10 cm ("near": within 2 cm of the 8 cm FV edge), a 10 × 10 cm box around the event, and the bottom corner d < 18 cm, z < 20 cm.

### 2.5 Transport and variance reduction (all unbiased; weights carried explicitly)
The ray-marcher moves each branch to the next boundary among the cylinders r = R, R − 0.3, R − 8 and the planes z = −13.75, 0, 9, 132.8, 145.6, processes the segment by region and iterates (≤ 80 segments):
1. *Active non-FV LXe*: any collision there produces an S2 outside the FV (or a second S2), so the history can never be the event; the branch is moved through with weight × e^{−μL} (survival biasing) and the collided part is dropped.
2. *FV, first FV collision not yet made*: split into a pass-through branch (weight × e^{−μL}) and a forced-collision branch with the collision point uniform along the FV chord and weight × μ e^{−μs} L (importance sampling of depth — deep scatters are sampled as often as shallow ones). At the collision the photo-/pair-absorption part (weight fraction (μ_pe + μ_pp)/μ) is recorded as a terminal deposit; the Compton part (μ_C/μ) either samples θ analog (mode **general**) or forces T uniform in [8, 16] keV with weight × (dσ/dT)(T) × 8 keV/σ_KN (mode **forced12**, exact KN weight).
3. *FV after the FV deposit*: pass-through only (a second FV deposit is multi-site).
4. *Dead regions (shell, RFR)*: split into pass-through (× e^{−μL}; terminal if the ray then leaves the LXe) and forced collision (× (1 − e^{−μL}), truncated-exponential position), absorption/Compton split as above, analog KN angle. Deposits accumulate per region; the number of dead-region collisions is unlimited.
5. Russian roulette below w = 10⁻¹⁴; photons are processed in chunks of 2 × 10⁴ to bound memory.
Mode **analog** (plain analog transport, collisions in active non-FV dropped) is used only to validate the weighted estimator.

## 3. Validation (`P033_validation.json`)
- Klein–Nishina: see §2.2 (χ²/dof 0.92; window fraction 0.00974 vs 0.00960; σ_KN numeric/analytic = 1 − 4 × 10⁻⁹).
- Analog (4 × 10⁶ wall-source photons at 1461 keV) versus weighted general mode (2 × 10⁵ photons), per emitted photon: ss_roi 4.5 ± 1.1 × 10⁻⁶ vs 3.91 ± 0.22 × 10⁻⁶ (pull +0.5σ; 18 vs 7788 records); wall_any 2.745 ± 0.083 × 10⁻⁴ vs 2.853 ± 0.046 × 10⁻⁴ (−1.1σ); rfr_any 7.65 ± 0.44 × 10⁻⁵ vs 8.09 ± 0.11 × 10⁻⁵ (−1.0σ); rfr_roi 2.5 ± 2.5 × 10⁻⁷ vs 0.93 ± 0.24 × 10⁻⁷ (+0.6σ); wall_roi: analog 0 records with 0.04 expected. Depth profile of wall_any in 1 cm bins from 8 to 22 cm: analog/weighted = 0.96, 0.92, 0.86, 1.17, 0.87, 1.16, 0.91, 0.84, 0.95, 1.05, 0.93, 0.95, 1.40, 0.85. The weighted estimator is unbiased within statistics and 20–400× more efficient.
- Physics cross-checks against P004's hand estimates: θ for a 12 keV deposit 6.39° at 1 MeV (P004: 6.4°), 4.36° at 1461 keV (4.4°), 2.43° at 2615 keV (2.4°); KN fraction T ∈ [10, 14] keV at 1 MeV 0.00480 (P004: 0.0048); P(interaction in 3 mm) 0.046 at 1 MeV (P004: 0.049).
- Normalisation cross-check against LZ's own model (§5): our wall-MSSI-in-ROI to single-scatter-ER ratio, scaled by LZ's detector-ER count, reproduces LZ's 0.0048 within ×1.15 after the dead-volume and veto corrections; ²¹⁴Pb gives 0.0038.

## 4. Results

### 4.1 Klein–Nishina and attenuation table (`P033_kn_table.csv`)
| E_γ (keV) | θ(8/12/16 keV) (°) | P(T∈8–16) | P(T∈60–95) | P(T∈150–260) | λ (cm) | P(interact in 3 mm) |
|---|---|---|---|---|---|---|
| 242 | 21.9/27.1/31.7 | 0.082 | 0.238 | 0 | 1.60 | 0.171 |
| 295 | 17.9/22.1/25.8 | 0.061 | 0.180 | 0.079 | 2.19 | 0.128 |
| 352 | 14.9/18.4/21.4 | 0.047 | 0.153 | 0.306 | 2.78 | 0.102 |
| 609 | 8.6/10.5/12.2 | 0.020 | 0.081 | 0.215 | 4.63 | 0.063 |
| 1000 | 5.2/6.4/7.4 | 0.0096 | 0.041 | 0.120 | 6.34 | 0.046 |
| 1173 | 4.4/5.4/6.3 | 0.0076 | 0.032 | 0.097 | 6.93 | 0.042 |
| 1461 | 3.6/4.4/5.0 | 0.0055 | 0.024 | 0.072 | 7.65 | 0.038 |
| 1764 | 2.9/3.6/4.2 | 0.0042 | 0.018 | 0.056 | 8.21 | 0.036 |
| 2615 | 2.0/2.4/2.8 | 0.0024 | 0.010 | 0.032 | 9.06 | 0.033 |
The "shallow-angle requirement": a 12 keV deposit from a ≥ 1 MeV γ needs θ ≤ 6.4°, so the outgoing γ continues essentially along its incoming direction and must cross the remaining chord of the TPC (P004's ≥ 54–75 cm) — the MC handles this automatically.

### 4.2 Main grid (forced12 mode; `P033_runs.csv`, `P033_summary.json`)
Per emitted photon, 5 × 10⁵ photons per line for wall and top, 1.25 × 10⁵ (bottom) and 1.5 × 10⁵ (cathode); weighted position-class records per line: 2600–9200 (wall), 1050–6100 (top), 140–440 (bottom), 75–440 (cathode).

**f_pos = P(d ≥ 25 cm and z ≥ 20 cm | wall12 class)**

| E_γ (keV) | wall | top | bottom | cathode |
|---|---|---|---|---|
| 352 | 8 × 10⁻⁶ | 1.7 × 10⁻⁴ | 0 (0 records) | 3 × 10⁻⁷ |
| 609 | 0.0022 | 0.0037 | 1.1 × 10⁻⁴ | 1.1 × 10⁻⁴ |
| 1120 | 0.0079 | 0.0165 | 0.0011 | 0.0012 |
| 1173 | 0.0145 | 0.0162 | 0.0012 | 0.0021 |
| 1332 | 0.0140 | 0.0203 | 0.0010 | 0.0021 |
| 1461 | 0.0107 | 0.0221 | 0.0017 | 0.0014 |
| 1764 | 0.0186 | 0.0289 | 0.0037 | 0.0040 |
| 2615 | 0.0257 | 0.0350 | 0.0055 | 0.0044 |
| line mix | **0.0137** | **0.0171** | **0.0018** | **0.0016** |

Equal mix of the four sources: **f_pos(wall12) = 0.0098** (P(wall12) = 2.29 × 10⁻⁹, P(wall12 ∧ pos) = 2.25 ± 0.06 × 10⁻¹¹ per emitted photon). Fraction within 2 cm of the FV edge (d ≤ 10 cm): 0.42–0.46 for every source and line ≥ 609 keV; d ≤ 15 cm: 0.83–0.87; median d = 10.3–10.7 cm. The depth profile is exponential from the FV edge with an effective length λ_eff = 4.22 (wall), 4.39 (top), 4.24 (bottom), 4.44 (cathode), 4.38 cm (equal mix) fitted over 10–40 cm — about 0.6 of the photon attenuation length because both the inbound and the outbound legs are attenuated. The density at the event's distance (26.9 cm) is a factor 90 (69–142 across sources) below the 8–9 cm bin. The 10 × 10 cm box around the event contains 5.5 × 10⁻⁴ of the wall12 class.

**RFR-like class (rfr12)**: P = 1.57 × 10⁻⁸ per photon (equal mix; 6.8× the wall12 class overall — the 13.75 cm RFR is a far more efficient second-scatter region than a 3 mm shell), but f_pos = 0.0021 (wall 0.0023, bottom 0.0021, cathode 1.6 × 10⁻⁴; top 0.59 with P = 8 × 10⁻¹³) and 81 % of the class has its FV deposit in the bottom corner (d < 18 cm, z < 20 cm); 89 % have z < 20 cm. At the event's position class the RFR-like and wall-like classes are comparable: rfr12/wall12 = 1.5 (equal mix), 2.25 (wall source), 104 (bottom source), 0.01 (top), 0.04 (cathode). The mixed class is 2 % of wall12.

**Bulk ²¹⁴Pb** (per excited-state decay anywhere in 7.6 t): P(wall12) = 4.6 × 10⁻⁸, P(wall12 ∧ pos) = 3.3 ± 0.8 × 10⁻¹¹ → f_pos = 7.2 × 10⁻⁴; λ_eff = 2.35 cm; edge-to-event density ratio 3700; rfr12 ∧ pos 5 × 10⁻¹⁵ (negligible). P(ss_roi) = 3.5 × 10⁻⁵, P(wall_roi) = 1.0 × 10⁻⁶ per excited decay.

**General mode** (analog angles; 2 × 10⁵ photons per line for wall/top, 5–6 × 10⁴ bottom/cathode): equal mix P(ss_roi) = 1.22 × 10⁻⁵, P(wall_roi) = 4.28 × 10⁻⁸ (ratio 3.5 × 10⁻³; per source 3.3–3.6 × 10⁻³), P(rfr_roi) = 2.35 × 10⁻⁷ (ratio 1.9 × 10⁻²; wall 2.8 × 10⁻², bottom 0.25, top 7 × 10⁻⁷, cathode 5 × 10⁻⁴). f_pos of the general wall-MSSI ROI population 0.011 (λ_eff 4.39 cm, 42 % at d ≤ 10 cm) — the same shape as the 12 keV class, i.e. the position penalty is not specific to the 12 + 77 keV decomposition. Energy-class fraction f_E = P(wall12)/P(wall_roi) = 0.053 (RFR: 0.067); P004's independently digitised neighbourhood fraction f_nb = 0.035 is consistent.

### 4.3 Variants (wall source; `P033_variants` in the summary)
| variant | E (keV) | f_pos(wall12) | P(wall12) relative |
|---|---|---|---|
| nominal | 609 / 1461 / 2615 | 0.0022 / 0.0107 / 0.0257 | 1 |
| all μ × 0.8 (λ +25 %) | 609 / 1461 / 2615 | 0.0063 / 0.033 / 0.051 | ×5.1 / ×3.3 / ×2.7 |
| isotropic hemisphere emission | 609 / 1461 / 2615 | 0.00053 / 0.0117 / 0.0145 | ×1.4 / ×1.05 / ×0.92 |
| dead shell 1.1 mm (0.3 % volume) | 1461 | 0.0144 | ×0.43 |
| FV stand-off 10.7 cm (mean) | 1461 | 0.040 | ×0.44 |
Longer attenuation lengths are the dominant systematic (×2–5 on f_pos); the angular distribution matters only at 609 keV; the shell thickness changes the rate, not the position distribution; a wider stand-off removes the near-edge population and raises the conditional f_pos to 4 % (but lowers the total by ×0.44, so the absolute expectation at the event position changes by only ×1.65).

## 5. Normalisation and the mismodelling factor (`P033_summary.json` → normalisation; `P033_post.json` → normalisation_alt)
(A) *LZ's own wall-MSSI expectation.* Science sample, 4.7 t ROI: 0.0048 (supplement table). Expected wall MSSI in P004's neighbourhood (S1c > 500 phd, ±2σ_NR) is 0.0048 × f_nb = 1.7 × 10⁻⁴; the position class adds f_pos:
- equal mix f_pos = 0.0098: N_pos = 1.65 × 10⁻⁶ → k_required(P ≥ 10 %) = −ln 0.9/N_pos = 6.4 × 10⁴; k(50 %) = 4.2 × 10⁵; k_required/k_allowed(95 %, P004: 1.64) = 3.9 × 10⁴;
- source extremes: f_pos 0.0016 (cathode) → k = 3.9 × 10⁵; 0.0171 (top) → 3.7 × 10⁴;
- own energy fraction f_E = 0.053 instead of f_nb: N_pos = 2.5 × 10⁻⁶, k = 4.2 × 10⁴;
- long-λ bracket (f_pos ×3): k ≈ 2 × 10⁴; stand-off 10.7 cm variant (f_pos 0.040 with the total ×0.44): k ≈ 1.6 × 10⁴ (obtained as 6.4 × 10⁴ × 0.0098/0.040/0.44 — the absolute position-class rate is what matters).
Every variant gives k_required ≥ 1.6 × 10⁴, versus the sideband limit k < 1.64 (P004): a ratio ≥ 10⁴. P004's f_pos ≤ 0.12 gave k ≥ 5 × 10³; the MC tightens this by a factor 3–13 and removes the three-decade ambiguity of the toy range.
(B) *Detector-ER normalisation (independent of LZ's MSSI simulation).* LZ expects 8.5 detector-ER single scatters in the science ROI (Table I) and 62.2 in the prompt-veto sample. With our ratio wall12_pos/ss_roi = 1.84 × 10⁻⁶ (equal mix; 3.1 × 10⁻⁷ bottom/cathode to 3.3 × 10⁻⁶ top): 1.6 × 10⁻⁵ events at the event's position class per run before dead-volume and veto corrections (×0.37 for a 0.3 %-volume shell, ×0.5 for the MSSI 94 % versus detector-ER 88 % tagging: 2.9 × 10⁻⁶), consistent with (A). Cross-check of the total: 8.5 × wall_roi/ss_roi = 0.030 (3 mm) → 0.011 (1.1 mm) → 0.0055 (veto ratio) vs LZ 0.0048 — agreement to 15 %, which validates the overall picture (dead-layer Compton scatter + escape). The RFR analogue does not agree: 8.5 × 0.019 = 0.16 (×0.06 untagged → 0.0098) vs LZ's 1 × 10⁻⁴; 73–81 % of our RFR-ROI and rfr12 histories have their FV deposit in the bottom corner d < 18 cm, z < 20 cm, exactly where LZ's real FV contour retreats to 18.2 cm from the wall (l. 138) — our uniform 8 cm stand-off overestimates the acceptance there; the residual factor (tagging of γ's escaping through the bottom array, the S2/ROI cuts on the FV deposit, the RFR light yield) is not resolved. We therefore normalise the RFR class with LZ's number: 1 × 10⁻⁴ × f_E(RFR) 0.067 × f_pos 0.0021 = 1.4 × 10⁻⁸ at the event's position class (bottom source alone: 1.3 × 10⁻⁸).
(C) *Bulk ²¹⁴Pb.* LZ's "internal β decays" (1341 ROI events) are dominated by ²¹⁴Pb ground-state decays; assuming 75 % of them are ²¹⁴Pb and a ground-state branch f_gs = 0.09 (recalled, uncertain), P(T_β ∈ 1.5–75 keV | Q = 1019 keV) = 0.095 gives 1.06 × 10⁴ ground-state decays in the FV, 1.07 × 10⁵ excited-state decays in the FV and 1.7 × 10⁵ in the 7.6 t of LXe. Then: wall MSSI in ROI 0.17 before veto → 0.0104 (×0.06 untagged) → 0.0038 (0.3 %-volume shell), again consistent with LZ's 0.0048 total; wall12 at the event's position class 5.6 × 10⁻⁶ before corrections → 1.2 × 10⁻⁷ after. Bulk ²¹⁴Pb has f_pos = 7 × 10⁻⁴ (λ = 2.2–2.8 cm) and is irrelevant at the event position.
(D) *Combined.* Expected MSSI of any category with a 12 keV FV scatter at d ≥ 25 cm, z ≥ 20 cm and the right S1: ≈ 2 × 10⁻⁶ per run (wall 1.6–2.5 × 10⁻⁶, RFR 1.4 × 10⁻⁸, ²¹⁴Pb ~10⁻⁷), i.e. 1 % of P004's neighbourhood expectation 1.7 × 10⁻⁴ and 2 × 10⁻⁴ of LZ's total MSSI 0.0049. P(≥ 1) ≈ 2 × 10⁻⁶.

## 6. Discussion
- The MC confirms the paper's qualitative statement and P004's mechanism, and pins the number: the FV-deposit density of 12 + 77 keV wall MSSI falls as e^{−d/4.3 cm} from the FV edge, 42–46 % of it lies within 2 cm of the edge, and only ≈ 1 % (0.16–1.7 % by source, 3–5 % in the longest-λ bracket) reaches the event's position class. P004's upper value 0.12 (single attenuated leg with λ = 9.8 cm) is excluded: the far-side chord and the requirement of no further active deposit are what the 1D toy missed; P004's lower value 10⁻⁴ applies only to ≤ 609 keV lines.
- Geometry alone does not prefer the wall over the RFR as the second-scatter site: for wall-emitted MeV γ's the RFR-like class is 2× more probable at the event position, for bottom-array γ's 100×. LZ's hit-pattern analysis disfavours the RFR interpretation (main text, S1 top/bottom asymmetry) and LZ's own RFR expectation in the 4.7 t ROI is 50× smaller than the wall one, so the RFR route is closed by data and normalisation, and the wall route by geometry.
- The "reverse" topology (β or absorption in a dead region, 12 keV Compton scatter of the γ in the FV) is included: the bulk ²¹⁴Pb runs start vertices in the shell and RFR, and the transport allows the dead-region scatter to precede the FV scatter (wall-first ordering). Since the S2 sits at the event position, the FV deposit must be the 12 keV one; no ordering escapes the two-leg attenuation.
- Model-derived rates reproduce LZ's wall-MSSI total to 15 % (detector γ) and 20 % (²¹⁴Pb), which lends credibility to the position distribution; the RFR total is over-predicted by ~100 in our simplified FV (bottom corner), so RFR numbers are quoted only as ratios normalised to LZ.
- Implication for the corpus: with N_pos ≈ 2 × 10⁻⁶, the wall-MSSI hypothesis requires a rate error k ≳ 1.6 × 10⁴–6 × 10⁴ against k < 1.64 allowed by the sidebands (P004); MSSI of any modelled category is a ≤ 10⁻⁵-level explanation of the event. This does not touch accidentals (P022), ER tails (P010) or artifacts.

## 7. Failed or abandoned approaches
- A single un-chunked transport of 2 × 10⁵ photons exhausted memory (branch populations grow at every FV/dead-region split); replaced by chunks of 2 × 10⁴ and per-run accumulators (no record hoarding).
- A first full run with all four sources in parallel exceeded the 10-minute foreground limit for the bottom source (photons entering through the RFR branch heavily); the bottom and cathode sources were run at 0.25–0.3 of the nominal statistics.
- The first pair-production table (2× too large) made λ(2615 keV) = 7.7 cm < λ(2 MeV); corrected and the 1764/2615 keV runs repeated (≤ 3 % effect at lower energies, not repeated).
- A per-line normalisation of the detector-γ mix by LZ's radioassay was not attempted (not in the paper); results are given per line and per source instead.

## 8. Figures
- `figures/P033_rz_map_and_depth.png` — Left/middle: probability density (log10, per cm²) of the FV deposit in (distance to true wall, z) for the wall-MSSI-like and RFR-MSSI-like 12 keV classes (equal mix of the four surface sources, line-weighted); dashed rectangle = simplified FV; grey lines = position-class cuts (25 cm, 20 cm); star = LZ event. Right: normalised density versus distance to the wall for the 12 + 77 keV class per source, for all wall MSSI in the ROI (general mode) and for the RFR 12 + 200 keV class; vertical line = event (26.9 cm).
- `figures/P033_kn_and_fpos.png` — Left: Klein–Nishina window probabilities (8–16, 60–95, 150–260 keV) and P(interaction in 3 mm) versus E_γ. Right: f_pos of the wall- and RFR-like 12 keV classes per line and source, with P004's f_pos range (light) and central range (dark) shaded.

## 9. Result tables
`P033_runs.csv` (all 79 runs: per-class probabilities, MC errors, record counts, position-class and near-edge fractions), `P033_kn_table.csv`, `P033_attenuation.json`, `P033_summary.json` (per-source and equal-mix aggregates, normalisations, variants), `P033_post.json` (depth-profile fits, corner fractions, alternative normalisations, ²¹⁴Pb budget), `P033_validation.json`.

## 10. References
1. LZ Collaboration, arXiv:2609.02823 (2026) — FV definition (l. 132–139), MSSI model (l. 184–196), supplement MSSI section and table (l. 710–783), Table I, prompt-sample table.
2. O. Klein and Y. Nishina, Z. Phys. 52, 853 (1929).
3. H. Kahn, "Applications of Monte Carlo", RAND RM-1237-AEC (1954) — Compton sampling method.
4. M. J. Berger et al., XCOM: Photon Cross Sections Database, NIST (recalled values).
5. I. Lux and L. Koblinger, Monte Carlo Particle Transport Methods (CRC, 1991) — survival biasing, forced collisions, splitting.
6. Corpus: P004 (k limits, f_nb, toy f_pos), P022 (drift geometry), P024 (NR PDF in MSSI contours), P029 (attenuation brackets).

## 11. Tools and provenance
Mirrors `output/provenance/P033.json`. Software: python 3.12.13, numpy 2.5.3 (vectorised transport, random Generator, histograms, polyfit), scipy 1.18.1 (imported via lzcommon only), pandas 3.0.5 (tables), matplotlib 3.11.2 (Agg figures), common/lzcommon.py (LZ constants: event position, ROI, background table). Files read: PAPER_GUIDE.md, 00_evidence_dossier.md, results_ledger.csv, papers P004/P022/P029/P024, work/P004/details.md §3.4, fulltext.tex l. 130–141, 184–197, 222–245, 518–582, 710–784, Fig. S5a and S6 PNGs, provenance/P004.json (format), lzcommon.py l. 1–75, ENVIRONMENT_versions.txt. Recalled items: 9 groups (TPC dimensions; LXe density; Xe photoelectric and pair coefficients; ²¹⁴Pb branches/endpoints and ground-state fraction; detector-γ line intensities; Kahn method; Fermi function; electron mass/r_e; internal-β composition). Commands: the stage list at the top of this file (validation; grid × 4 sources with --scale 1/1/0.25/0.3; bulk --scale 0.5; variants --scale 0.6; grid reruns --lines 2615,1764; variants --lines 2615; summary; post).
