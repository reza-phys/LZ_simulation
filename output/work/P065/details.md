# P065 — Magnetic inelastic dark matter (MiDM) at LZ: the transition-dipole scattering spectrum, the de-excitation photon and the surviving parameter space

Simulated date 2026-09-12 · hep-ph · MODEL · author profile: magnetic inelastic dark-matter model builders.
Script: `output/code/P065_midm.py` (run from the simulation root with `.venv/bin/python`; spectra cached in `P065_spectra_cache.npz`).
All numbers below are printed in `P065_run.log` or stored in the CSV/JSON files listed in §9.

## 1. Motivation and framework

Chang, Weiner and Yavin (2010) proposed *magnetic inelastic dark matter* (MiDM): a Majorana pair χ₁, χ₂ split by δ with a
transition magnetic moment, L = (μ_χ/2) χ̄₂ σ^{μν} χ₁ F_{μν} + h.c. Because the moment is purely off-diagonal, χ₁ has no
tree-level elastic coupling to the photon; scattering on nuclei is χ₁N → χ₂N through photon exchange with (i) the nuclear
charge (dipole–charge, long range, ∝ Z²) and (ii) the nuclear magnetic moments (dipole–dipole, ∝ spin). The excited state decays
by χ₂ → χ₁γ with Γ = μ_χ²δ³/π. The corpus has so far treated the photon-M1 transition as a sub-case of composite models
(P023: μ_tr = 2.1×10⁻⁴ μ_N for one event at δ = 300 keV, τ = 66 μs, single-site ceiling δ ≤ 326 keV) and folded it into the
de-excitation-photon analysis (P042: P(clean) = 0.9/0.5/0.1 at δ = 321/344/358 keV; the clean event demands τ > 0.43 μs). P012
showed that the *elastic* photon-coupled dipole is excluded by its Z²/E_R low-energy tail (N_lo = 376). Here we treat MiDM
as a model: spectrum and its charge/spin decomposition, the fitted moment, the lifetime and the P042 constraint, the relic density
from χ₁χ₂ → γ* → ff̄, the astrophysical de-excitation photons, and the target-scaling predictions.

## 2. Inputs

| Input | Value | Source |
|---|---|---|
| NREFT coefficients for a photon-coupled magnetic dipole | c₁^N = eμQ_N/(2m_χ), c₅^N = 2eμQ_N m_N/q², c₄^N = eμg_N/m_N, c₆^N = −eμg_N m_N/q² (Anand normalisation, WimPyDD c⁰ = c_p + c_n) | Fitzpatrick et al. 2012 / Anand et al. 2014 as implemented in P012 and P023 (recalled/likely; P023 reproduced P012's elastic normalisation to 0.4 %) |
| Inelastic kinematics | WimPyDD `delta` argument (v⊥² = v² − v_min², v_min = (m_N E_R/μ_N + δ)/√(2m_N E_R)) | WimPyDD 2.0.4 |
| g_p, g_n | 5.5857, −3.8261 | PDG (certain) |
| α, ħ, μ_N | 1/137.036; 6.5821×10⁻²⁵ GeV s; μ_N = e/(2m_p) = 0.1614 GeV⁻¹ = 1.0515×10⁻¹⁴ e·cm | PDG (certain) |
| Halo | Baxter-2021 SHM via `lz.wd_halo`: **Sun frame** (WimPyDD single-day halo, v_E = 250.6 km/s, v_max = 795 km/s), **June 16** (day 167, v_E = 265.1, v_max = 809 km/s), **annual** (12-day mean, only at δ = 300 keV for the P023 cross-check) | lzcommon; P035 labelling |
| Efficiency | erf model: plateau 0.96, 50 % at 5.4 and 269.9 keV, widths 3.4/8 keV | P003/P012/P023 model of LZ Fig. S2 |
| Exposure | 2.84 t·yr | LZ paper |
| m_χ | 1000 GeV throughout | assignment |
| P(clean | τ), P(TPC second site), P(delayed veto) | `output/work/P042/P042_topology_vs_tau_d{250,300,350,380}_wind.csv`, interpolated in ln τ, nearest-δ table | corpus P042 |
| Γ(χ₂ → χ₁γ) = μ²δ³/π | formula | Chang, Weiner, Yavin 2010 (recalled/likely; P023 checked it against the hydrogen 21 cm rate up to the spin-matrix factor) |
| Thermal target | 2.2×10⁻²⁶ cm³/s (self-conjugate); Dirac-like two-species target 4.4×10⁻²⁶ cm³/s | Steigman–Dasgupta–Beacom 2012 (certain) / bookkeeping (likely) |
| Σ_f N_c Q_f² at √s = 2 TeV | 8 (W⁺W⁻ omitted, +O(10 %)) | SM charges (certain) |
| Millicharge annihilation normalisation | σv_rel = πα²ε²N_cQ_f²/m² | standard QED (likely) |
| LZ-2024 low-energy search | 4.2 t·yr, 5.4–55 keV, tolerance ≤ 3–5 events | LZ PRL 135, 011802 via P003/P012 (certain/uncertain) |
| XENONnT / PandaX-4T | 3.1 / 1.54 t·yr, ROI edges ≲ 60–70 keV | via P005 (certain/uncertain) |
| MiDM literature moment | ~10⁻³ μ_N at ~100 GeV (DAMA-era fits) | Chang, Weiner, Yavin 2010 (uncertain) |
| P046 O₁ target ratios | W/Xe = 15.9/81.7/221/693, I/Xe = 0.591/0.295/0.142/0.0227 at 300/350/366/380 keV (annual halo, full windows) | corpus P046 |

Recalled items: 16 (`recalled_inputs.json`; 3 uncertain).

## 3. Method

* **Hamiltonians** (WimPyDD `eft_hamiltonian`, coefficient functions built as closures with *no* default arguments, per the P031
  pitfall): `midm` = {c₁, c₅, c₄, c₆}; `charge` = {c₁, c₅}; `spin` = {c₄, c₆}; `O1` = isoscalar contact O₁ at LZ unit coupling
  (WimPyDD c⁰ = 2/m_v²) for shape comparison. All at μ = 1 μ_N; rates scale as μ². Runtime patch (no file edited, as in P015/P046):
  the WimPyDD 18xW response files use `np` without importing it.
* **Grids**: E = 1–328 keV in 3 keV steps (110 points) for the LZ ROI; 335–995 keV in 20 keV steps for the rest of the kinematic
  window (target ratios). Spectra are interpolated to 0.5 keV before integrating. δ = 250–380 keV in 10 keV steps (+366) in the
  Sun frame; June 16 at 300 and 350 keV; annual at 300 keV; δ = 0 and 200 keV for the decomposition and the low-energy tail.
  One 110-point xenon spectrum costs 45–100 s, so the plan was kept to 60 spectra.
* **Counts**: N_ROI = ∫ε·dR/dE over E ≤ 330 keV; N_hi over 200–270 keV; N_lo over 5.4–55 keV (LZ-2024 window, same ε);
  μ(N = 1) = √(1/N(1 μ_N)); 68 % band from N = 0.3–2.4 (LZ's 1.0 (+1.4, −0.7) fit) → μ × 0.55–1.55.
* **Percentile of 248 keV**: CDF of the efficiency-folded spectrum within E ≤ 330 keV.
* **Lifetime**: τ = πħ/(μ²δ³); P(clean), P(TPC), P(delayed) from the P042 tables at τ(δ). June 16 curve: the Sun-frame μ(δ) is
  scaled by √(N_Sun/N_June) with the June/Sun rate ratio (1.529 at 300, 3.418 at 350 keV) interpolated in ln between the two
  computed points (a direct linear interpolation of P(clean) between two June points 50 keV apart was tried first and moved the
  ceiling the wrong way; abandoned).
* **Relic**: derivation in §5.4; Ω h² = 0.120 × (4.4×10⁻²⁶ cm³/s)/⟨σv⟩.

## 4. Kinematics of the window (1 TeV)

| δ [keV] | E₋ [keV] (Sun / June) | E₊ [keV] (Sun) | v_min(248 keV) [km/s] | max v⊥²/v_max² at 248 keV |
|---|---|---|---|---|
| 0 | 0 | 1365 | 339 | 0.818 |
| 200 | 32.5 | 976 | 582 | 0.463 |
| 250 | 57.6 | 862 | — | — |
| 300 | 97.5 / 90.4 | 733 | 704 | 0.216 |
| 350 | 170 / 152 | 571 | — | — |
| 366 | 213 | 500 | — | — |
| 380 | 284 | 404 | — | — |

The 5.4–55 keV window is kinematically empty for δ > 246 keV (Sun frame) / 251 keV (June); a 60 keV edge: 254/260 keV.
In the Sun frame δ_max(248 keV) ≈ 376 keV, so the δ = 380 keV Sun-frame row is beyond the ceiling (N(1 μ_N) = 0.17, μ(N = 1) = 2.4 μ_N); it is kept in the tables but not used in the paper.

## 5. Results

### 5.1 Spectrum, decomposition, position of 248 keV (`P065_xenon_scan.csv`, `P065_decomposition.csv`, `P065_shape_d300.csv`, Fig. 1)

**Fitted moment at δ = 300 keV.** N_ROI(1 μ_N) = 2.089×10⁷ (Sun frame) → μ_χ(N_ROI = 1) = 2.188×10⁻⁴ μ_N = 2.30×10⁻¹⁸ e·cm =
0.233 e/(2m_χ). June 16: N = 3.193×10⁷ (×1.529) → 1.770×10⁻⁴ μ_N. Annual mean: 2.108×10⁻⁴ μ_N versus P023's 2.107×10⁻⁴
(ratio 1.000): the P023 implementation is reproduced. μ(N_hi = 1) = 3.09×10⁻⁴ μ_N (Sun).

μ_χ(N_ROI = 1), Sun frame, in 10⁻⁴ μ_N: 0.84 (250), 0.99 (260), 1.18 (270), 1.43 (280), 1.75 (290), 2.19 (300), 2.81 (310),
3.74 (320), 5.26 (330), 8.03 (340), 14.1 (350), 32.9 (360), 76.0 (366), 181 (370). In e/(2m_χ): 0.089 (250) → 0.23 (300) →
1.5 (350) → 8.1 (366): above ≈ 345 keV the required moment exceeds the "natural" e/(2m_χ) of a TeV particle.

**Decomposition** (Sun frame; fractions of the full rate, which includes the c₄–c₆ (Σ′–Σ″) and c₄–c₅ (Σ′–Δ) interference):

| δ [keV] | charge part at 248 keV | dipole–dipole at 248 keV | interference at 248 keV | charge / spin share of N_ROI | N_lo per 200–270 keV event: full / charge / spin |
|---|---|---|---|---|---|
| 0 (elastic) | 0.9 % | 90.1 % | 9 % | — | 386 / 340 / 28.7 |
| 200 | 0.8 % | 90.2 % | 9 % | — | 0.223 / 0.156 / 0.019 |
| 300 | 0.6 % | 90.4 % | 9.0 % | 1.8 % / 88.6 % (interf. 9.6 %) | 0 / 0 / 0 |

The elastic row reproduces P012 (N_lo/hi = 376; their per-part ratios 18 000 and 32 correspond to our 340/0.018 ≈ 19 000 and
28.7/0.9 ≈ 32). At 248 keV the dipole–charge term is at the 1 % level *for every δ*: |c₅O₅|² ∝ Z² m_N² v⊥² F_M²/q² while
|c₄O₄ + c₆O₆|² ∝ g_N² Σ′ with no velocity factor; at q = 0.25 GeV the ratio is ≈ Z² v⊥² m_N²/q² × O(1) ≈ 10⁻², and the inelastic
kinematics reduce v⊥² further (max v⊥²/v_max² = 0.22 at δ = 300 versus 0.82 elastic). The long-range 1/E_R enhancement wins
only below ≈ 50 keV, i.e. in the elastic case. Consequently **the LZ event, in MiDM, is a spin–spin scatter on ¹²⁹Xe/¹³¹Xe**, and
the low-energy tail is *not* repopulated: at δ = 200 keV N_lo per high-energy event is 0.22 for MiDM against 7.65 for contact O₁
(the charge term supplies 70 % of that small tail), and for δ ≥ 250 keV the tail is kinematically absent (N_lo/hi = 2×10⁻⁸ at 250, 0 at ≥ 260).

**Shape and percentile.** MiDM/O₁ (normalised at 248 keV) = 0.07, 0.04, 0.10, 0.35 at 120, 150, 200, 300 keV: the MiDM spectrum
is *harder* than contact O₁ inside the window (Σ′ response versus the M response with its 266 keV node). Efficiency-folded,
δ = 300 keV: MiDM peaks at 196 keV with 68 % interval 160–241 keV and 248 keV at the **88.6th percentile**; contact O₁ peaks at
172 keV, 147–204 keV, 248 keV at the 99.5th percentile (P002: 99.5th). Fraction of the accepted rate below 200 keV: 0.49 (MiDM) vs
0.81 (O₁). Percentile of 248 keV in MiDM vs δ: 92.9 % (250), 88.6 (300), 85.0 (320), 77.4 (340), 69.2 (350), 52.0 (360),
32.2 (366), 17.5 (370); the event sits at the median for δ ≈ 360 keV (O₁: 380 keV, P021). Accepted peak: 181 → 196 → 247 → 256 keV
at δ = 250 → 300 → 350 → 366.

### 5.2 Lifetime and the de-excitation photon (`P065_lifetime_pclean.csv`, `P065_population_topology.csv`, Fig. 2)

τ = πħ/(μ²δ³). At δ = 300 keV, N = 1 (Sun): μ = 2.19×10⁻⁴ μ_N → τ = 61.4 μs (decay length 41 m at 663 km/s), P(clean) = 0.973,
P(second TPC site) = 0.016, P(OD delayed veto) = 0.010, P(prompt veto) = 0. June 16: τ = 94 μs, P(clean) = 0.982.
τ(δ), N = 1, Sun frame: 722, 460, 289, 178, 107, 61, 34, 17, 8.0, 3.1, 0.93, 0.16, 0.028, 0.0048 μs at δ = 250 … 370 keV.

| case | P(clean) = 0.9 | 0.5 | 0.1 |
|---|---|---|---|
| Sun frame, N = 1 | 321 keV | 342 | 355 |
| Sun frame, N = 0.3 (μ × 0.55) | 333 | 351 | 360 |
| Sun frame, N = 2.4 (μ × 1.55) | 308 | 333 | 348 |
| June-16 (scaled), N = 1 | 331 | 352 | 360 |
| June-16 (scaled), N = 0.3 / 2.4 | 344 / 319 | 358 / 344 | 365 / 357 |
| P042 (P023 annual moments), N = 1 | 321 | 344 | 358 |

Direct June points: δ = 350 keV, μ = 7.63×10⁻⁴ μ_N, τ = 3.18 μs, P(clean) = 0.574 (Sun: 0.93 μs, 0.20).
Reading: with the LZ rate fixing μ(δ), the clean single-site event excludes MiDM at δ > 355 keV (Sun) / 360 keV (June) at 90 % CL
and at δ > 342/352 keV at 50 %; the 68 % band on the fitted rate moves these by ±7 keV. P042's numbers are reproduced within 3 keV.

Population topology at N = 1 (P042 classes): δ = 300: 97.3 % clean, 1.6 % second TPC site (≈ 300 keV ER, 42 cm away, P042),
1.0 % OD delayed veto; 320: 90.9/5.6/3.3 %; 340: 57.0/25.0/17.5 %; 350: 20.1/59.1/20.4 %; never a prompt veto.

### 5.3 Elastic and low-energy constraints (`P065_low_energy_constraints.csv`)

*Elastic.* A pure transition moment has no χ₁χ₁γ vertex, so χ₁N → χ₁N vanishes at tree level; the leading elastic operator is
generated at one loop with two dipole insertions and χ₂ in the loop (a Rayleigh-type χ̄₁χ₁F² polarisability). Parametrically the
amplitude ratio to the tree-level inelastic amplitude is ~ (μq)²/(16π²) = 5×10⁻¹³ at q = 0.25 GeV (rate ratio 10⁻²⁵): no
direct-detection constraint from elastic scattering (order-of-magnitude estimate, flagged).

*Low-energy inelastic tail.* At μ(N_ROI = 1): expected 5.4–55 keV events in the LZ-2024 4.2 t·yr search = 1.2×10⁻⁸ (δ = 250),
0 (δ ≥ 260); XENONnT (3–60 keV, 3.1 t·yr) 4.5×10⁻⁵ (250), 1.9×10⁻⁹ (260), 0 (300); PandaX-4T half of that. The 2018 XENON1T
inelastic limit reaches only δ = 231 keV (P035). Hence **no low-energy dataset constrains MiDM for δ ≥ 250 keV**; the only
low-energy constraint would be on δ ≲ 240 keV, where MiDM's tail is 34× smaller than O₁'s anyway.

### 5.4 Relic density (`P065_relic.csv`)

**Derivation.** The transition current for χ₁(p₁)χ₂(p₂) → γ*(q = p₁+p₂) is J^ν = μ ū(p₁)σ^{μν}q_μ v(p₂) (vertex from
(μ/2)χ̄σ^{μν}χF_{μν} → μσ^{μν}q_μ). At threshold p₁ = p₂ = (m, 0), q = (2m, 0); only ν = i survives, σ^{0i} = iγ⁰γ^i, and
J^i = 4im² ξ†σ^iη, so Σ_spins J^iJ^{j*} = 32m⁴δ^{ij} (numerical Dirac-matrix check in the script: 32.0; the vector current
ūγ^iv gives 8m²δ^{ij}, check 8.0). The massless-fermion tensor Tr[k̸₁γ_ik̸₂γ_j] angular-averaged is (16/3)m²δ_{ij}. With the 1/4
spin average and photon propagator 1/s² = 1/(16m⁴): |M̄|² = 8e²Q_f²μ²m²; σv_rel = |M̄|²/(4m² · 8π) gives

  ⟨σv⟩(χ₁χ₂ → ff̄) = α μ_χ² Σ_f N_c Q_f²   (s-wave, velocity-independent, **mass-independent**).

The same steps for a vector current with charge g reproduce σv = πα α_χ Q_f²/m², i.e. the recalled millicharge formula, which
fixes the normalisation. χ₁χ₂ → γγ has no tree diagram (it would need a χ₂χ₂γ or χ₁χ₁γ vertex); χ₁χ₁ → γγ through χ₂ exchange is
O(μ⁴): σv ~ μ⁴m²/(4π) = 1.4×10⁻³⁰ cm³/s at the LZ moment (order of magnitude). Both χ₁ and χ₂ are populated at freeze-out
(δ ≪ T_f ≈ 40 GeV), so the system behaves like a Dirac pair annihilating only through χ₁χ₂: dn/dt = −(σv/2)n², and the thermal
target is 4.4×10⁻²⁶ cm³/s (likely; an O(1) Majorana convention factor would shift μ_th by ≤ √2).

**Numbers.** At μ(N_ROI = 1, 300 keV) = 2.19×10⁻⁴ μ_N: ⟨σv⟩ = 8.5×10⁻²⁸ cm³/s → Ω h² = 6.2, **52× over-abundant**; an extra
⟨σv⟩ = 4.3×10⁻²⁶ cm³/s from another channel (dark photon, Higgs portal, co-annihilation) or a non-thermal history is needed.
Ω h² at μ(N = 1): 42 (250), 6.2 (300), 1.07 (330), 0.46 (340), 0.15 (350), 0.027 (360), 0.005 (366).
The thermal moment is μ_th = 1.57×10⁻³ μ_N = 1.66×10⁻¹⁷ e·cm for any mass (compare the ~10⁻³ μ_N of DAMA-era MiDM fits,
recalled/uncertain). At μ_th LZ would have seen 52 events at δ = 300 keV, 8.9 at 330, 1.25 at 350: **thermal MiDM gives one LZ
event at δ = 351 keV (344–358 keV for N = 2.4–0.3)**, where τ = 0.74 μs (0.65–0.81) and P(clean) = 0.15 (0.16–0.14). The thermal
point is therefore disfavoured at ≈ 85 % CL by the clean event, independently of the halo (June: τ(351 keV) unchanged since μ_th is fixed).

### 5.5 Astrophysical 300 keV photons

With τ = 61 μs (or ≤ 1 ms anywhere in the fitted band) every χ₂ produced in the early Universe (P026: f₂ ≈ 0.4–0.5 at
decoupling for weak-strength transitions) decays long before BBN; f₂(today) = 0 and there is no relic line or continuum
(P026's Planck/SPI windows 10¹²–10²² s are irrelevant). A present-day line could only come from halo up-scattering
χ₁χ₁ → χ₂χ₂ (both legs flip; endothermic by 2δ, threshold v_rel = 464 km/s, available in the halo tail) followed by χ₂ → χ₁γ.
Its cross-section is O(μ⁴): σ ~ μ⁴m²/(4π) = 5×10⁻⁴¹ cm², giving a Galactic-centre flux ~10⁻¹⁸ ph cm⁻² s⁻¹ sr⁻¹, 13 orders below
SPI (~10⁻⁵ ph cm⁻² s⁻¹, recalled/uncertain). Nothing observable.

### 5.6 Target ratios (`P065_target_ratios.csv`)

Full-window rates (1–1000 keV, no efficiency, events per t·yr of element at μ = 1 μ_N or LZ unit O₁ coupling; Sun frame):

| δ [keV] | MiDM Xe | MiDM W | MiDM I | **W/Xe MiDM** | **I/Xe MiDM** | O₁ W/Xe (this work / P046 annual) | O₁ I/Xe (this work / P046) |
|---|---|---|---|---|---|---|---|
| 300 | 1.42×10⁷ | 1.79×10⁶ | 1.75×10⁸ | **0.126** | **12.4** | 16.5 / 15.9 | 0.585 / 0.591 |
| 350 | 7.13×10⁵ | 1.75×10⁵ | 5.19×10⁶ | **0.246** | **7.27** | 108 / 81.7 | 0.215 / 0.295 |
| 366 | 9.08×10⁴ | 8.54×10⁴ | 1.52×10⁵ | **0.941** | **1.68** | 360 / 221 | 0.030 / 0.142 |

Our O₁ ratios agree with P046 at 300 keV (1.04, 0.99) and drift at higher δ because P046 used the annual-mean halo (P046 itself
reports W/Xe(350) = 82 SHM vs 55 June; the Sun frame lies on the other side). The MiDM ratios are the *reverse* of O₁:
the rate is carried by the nuclear spin (¹²⁷I, J = 5/2, 100 % abundance, large ⟨S_p⟩; xenon 48 % odd isotopes; tungsten 14 % ¹⁸³W
with a small ⟨S_n⟩), not by A² or Z² (the naive coherent Z²/A vs A²/A scalings differ by only 4 %). A CaWO₄ δ-meter (P046) would
read *nothing* for MiDM (W/Xe = 0.13 instead of 16 at 300 keV; the MiDM/O₁ ratio-of-ratios is 0.008–0.002), whereas an iodine
target (NaI, CsI) would see 12× the xenon rate per tonne at δ = 300 keV, 7× at 350, 1.7× at 366. The tungsten spin response in
WimPyDD is a Gaussian placeholder (P046), so the W/Xe MiDM value is an order-of-magnitude number; the I/Xe value uses the
shell-model ¹²⁷I responses.

## 6. Validation and robustness

* Annual-mean μ(N = 1) at 300 keV reproduces P023 to 0.05 % (2.108 vs 2.107×10⁻⁴ μ_N).
* Elastic decomposition reproduces P012 (N_lo/hi 386 vs 376; part ratios 19 000/32 vs 18 000/32).
* Contact-O₁ percentile of 248 keV at δ = 300 keV: 99.5 %, identical to P002.
* P042's photon-M1 ceilings reproduced to ≤ 3 keV (321/342/355 vs 321/344/358) with independently computed moments and the Sun-frame halo.
* Dirac-matrix check of the annihilation tensors (32m⁴, 8m²) and of the millicharge normalisation.
* O₁ target ratios agree with P046 at 300 keV to 4 %.
* Halo: June 16 lowers μ(N = 1) by 0.81 (300) / 0.54 (350) and raises the P(clean) ceilings by 10 keV; the 68 % rate band moves them by ±7 keV.
* Grid: 3 keV steps with 0.5 keV interpolation; the kinematic edge E₋ is resolved to ≈ 1 keV (checked against `lz.E_R_range_keV`).

## 7. Failed or abandoned

* The first plan (71 spectra, 10 keV extended grid, δ up to 390 keV, June at four δ) exceeded the foreground time limit at 70–100 s per xenon spectrum; trimmed to 60 spectra with a 20 keV extended grid; δ = 390 dropped (beyond the Sun-frame ceiling).
* June-16 P(clean) ceilings from linear interpolation between δ = 300 and 350 keV moved the wrong way (306 keV); replaced by the scaled-μ(δ) construction.
* The tungsten target failed with `NameError: np` inside WimPyDD's 180W response file; fixed at runtime as in P015/P046 (no file edited).

## 8. Discussion

MiDM survives at LZ in a narrow but well-defined form. (i) Contrary to the intuition that the long-range Z² dipole–charge term
dominates a photon-coupled moment, at 248 keV it is a 1 % effect: the LZ event in MiDM is a spin–spin scatter, which is why the model
has no low-energy companions and why P012's elastic exclusion does not apply. (ii) The MiDM spectrum is harder than contact O₁, placing
248 keV at the 89th rather than the 99.5th percentile at δ = 300 keV; the event becomes median at δ ≈ 360 keV, but by then the
lifetime is 0.16 μs and the clean event excludes the model (P(clean) < 0.1 above 355 keV). (iii) The one-event moment
2.2×10⁻⁴ μ_N = 0.23 e/(2m_χ) is 7× below the thermal moment 1.6×10⁻³ μ_N; a thermal MiDM particle fits LZ only at δ = 351 keV,
where P(clean) = 0.15. So a thermal-relic MiDM reading of the event is disfavoured at 85 %, and a δ ≈ 300 keV reading requires an
additional annihilation channel providing 4.3×10⁻²⁶ cm³/s (dark photon or Higgs portal, both natural in MiDM UV completions) or an
asymmetric/non-thermal history. (iv) The decisive test is not another xenon exposure but a spin-sensitive target: iodine.

## 9. Files

* `P065_spectra_cache.npz` — 60 WimPyDD spectra (Xe/W/I; midm/charge/spin/O1; Sun/June/annual; δ).
* `P065_xenon_scan.csv` — counts, percentiles, moments per (model, halo, δ).
* `P065_decomposition.csv`, `P065_shape_d300.csv` — charge/spin decomposition, MiDM/O₁ shape ratio.
* `P065_lifetime_pclean.csv`, `P065_population_topology.csv` — τ(δ), P(clean), topology shares.
* `P065_low_energy_constraints.csv`, `P065_relic.csv`, `P065_target_ratios.csv`.
* `P065_summary.json`, `recalled_inputs.json`, `P065_run.log`.
* Figures: `figures/P065_fig1_spectra.png` (left: δ = 300 keV MiDM full/charge/spin vs contact O₁, LZ efficiency, event; right: efficiency-folded MiDM spectra at δ = 300/350/380 keV), `figures/P065_fig2_tau_pclean.png` (τ(δ) for N = 1 with 68 % band, June-scaled curve and direct June points, P042's 0.43 μs, thermal τ; P(clean|τ(δ)) on the right axis with the 0.9/0.5/0.1 crossings), `figures/P065_fig3_mu_vs_delta.png` (μ(N = 1) vs δ with band, June points, thermal moment, P012 elastic exclusion, P(clean) < 0.1 region).

## 10. References

LZ Collaboration, arXiv:2609.02823 (2026) · S. Chang, N. Weiner, I. Yavin, PRD 82, 125011 (2010) · D. Tucker-Smith, N. Weiner, PRD 64, 043502 (2001) · T. Banks, J.-F. Fox, N. Weiner, PRD 82, 075004 (2010) · V. Barger, W.-Y. Keung, D. Marfatia, PLB 696, 74 (2011) · N. Anand, A. L. Fitzpatrick, W. C. Haxton, PRC 89, 065501 (2014) · G. Barello, S. Chang, C. A. Newby, PRD 90, 094027 (2014) · G. Steigman, B. Dasgupta, J. F. Beacom, PRD 86, 023506 (2012) · I. Jeong, S. Kang, S. Scopel, G. Tomar (WimPyDD), CPC 276, 108342 (2022) · corpus: P002, P003, P011, P012, P021, P023, P026, P035, P042, P044, P046.

## 11. Tools and provenance

Agent tools: Read ×17 (PAPER_GUIDE, dossier, ledger p.1, P023/P042/P012/P044/P026/P011/P002/P021 papers, P023 script, lzcommon §8, P046 script excerpt, 3 figures), Bash ×19 (ledger/lzcommon greps, WimPyDD target inspection, P042/P023/P046 table inspection, timing test, 180W file, four script runs, interim-log check, table printing, three word-count checks, JSON validation; one further call with `sleep` was blocked by the harness and not executed), Write ×4 (script, details, provenance, paper), Edit ×20 (script 11, paper 9).
Software: python 3.12.13; WimPyDD 2.0.4 (`eft_hamiltonian` with q- and m_χ-dependent closures, `diff_rate` with `delta`, `streamed_halo_function`; Xe, W, I targets); numpy 2.5.3; scipy 1.18.1 (`special.erf`, `optimize.brentq`); pandas 3.0.5; matplotlib 3.11.2; common/lzcommon.py (`wd`, `wd_halo`, `wd_rate`, `wd_hamiltonian`, `wd_c_from_anand`, `E_R_range_keV`, `vmin_kms`, `v_earth_kms`, constants). Dirac-matrix annihilation check: numpy.
Script: `output/code/P065_midm.py`, run four times (`--budget 540/400/450/450`) to fill the cache, then analysis (16 s).
Local inputs: P042 topology CSVs (4), P023 cache (magnitude check only), lzcommon. Datasets: none. Data requests: none.
WimPyDD-generated files: none outside `output/work/P065/` (Hamiltonian names `P065_*` are in-memory; `WD.diff_rate` writes no response files).
