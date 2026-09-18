# Tool inventory (aggregated, whole process)

Final consolidated version (step 209; the whole run is complete: 100 papers, 100 provenance JSONs). Versions are taken from `environment/ENVIRONMENT_versions.txt` (generated 2026-09-16T15:23Z by `environment/setup.sh`) and from my own run of `.venv/bin/python environment/check_env.py` (process-log step 006), unless a command is quoted.

## 1. Environment

| Item | Value | Source |
|---|---|---|
| OS | macOS 26.6.2, Darwin 25.6.0 (xnu-12377.161.14~5/RELEASE_ARM64_T8132), arm64 (Apple Silicon MacBook Air) | ENVIRONMENT_versions.txt; `uname -a` (step 007) |
| Shell | zsh (`/bin/zsh`) | `echo $SHELL` (step 007) |
| Agent harness | Claude Code 2.1.272, model claude-fable-5-1, permission mode acceptEdits, macOS Seatbelt sandbox, network `allowedDomains: []`, WebFetch/WebSearch/MCP denied | ENVIRONMENT_versions.txt; `.claude/settings.json`; `run.sh` |
| Python interpreter | Python 3.12.13 at `.venv/bin/python` (installed from `requirements.lock.txt`) | ENVIRONMENT_versions.txt; check_env.py |
| TeX | pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026); BibTeX 0.99e; revtex4-2, siunitx, mhchem, cleveref, physics, tikz-feynman, feynmp, booktabs, natbib, amsmath, hyperref available | ENVIRONMENT_versions.txt; check_env.py |
| WimPyDD | 2.0.4 (source zip, sha256 30129efa…0eb002) in `WimPyDD/`; run from simulation root. Bundled experiments: DAMA_LIBRA_2019, LZ_2022, PICO60_2019, XENON_1T_2018; bundled targets incl. Xe (9 isotopes) | ENVIRONMENT_versions.txt; step 012, 016 |
| nestpy | 2.1.1 (`nestpy.__nest_version__` reports 0.0.0; the pip version is 2.1.1). Detectors: DetectorExample_XENON10, LUX_Run03, LZ_WS2022, LZ_WS2024 (g1=0.1122, central field 96.5 V/cm). NR parameter vector has 12 entries [α, β, γ, δ, ε, ζ, η, θ, ι, p, f1, f2]; no built-in (a, b, E0) power-law break | steps 013–014 |
| Not available | network; Fortran/C++ codes (micrOMEGAs, MadGraph, CLASS, DarkSUSY); ROOT; package installation; `/usr/bin/python3` (forbidden by PROMPT §2b) | PROMPT.md; ENVIRONMENT_versions.txt |

### Python packages (exact versions, from ENVIRONMENT_versions.txt)
numpy 2.5.3 · scipy 1.18.1 · sympy 1.14.0 · mpmath 1.3.0 · numba 0.67.0 · joblib 1.6.0 · pandas 3.0.5 · tabulate 0.10.0 · PyYAML 6.0.3 · h5py 3.16.0 · uproot 5.7.6 · awkward 2.13.0 · openpyxl 3.1.5 · iminuit 2.32.0 · statsmodels 0.15.0 · emcee 3.1.6 · corner 2.3.0 · dynesty 3.1.0 · lmfit 1.3.4 · pyhf 0.7.6 · uncertainties 3.2.3 · scikit-learn 1.9.1 · matplotlib 3.11.2 · seaborn 0.13.2 · astropy 8.0.1 (offline) · hepunits 2.4.6 · particle 1.0.1 · periodictable 2.1.0 · radioactivedecay 0.6.1 · wimprates 0.5.0 · directdm 2.2.2 · nestpy 2.1.1 · camb 2.0.4 · healpy 1.20.0 · pymupdf 1.28.2 · pdfplumber 0.11.10 · pillow 12.3.0 · opencv-python-headless 5.0.0.93 · tqdm 4.70.1

## 2. Tools and packages used, with papers (consolidated over P001–P100)

Generated from the 100 per-paper provenance JSONs by `output/code/common/inventory_tools.py` (step 209); the chronological batch-by-batch record follows in the Appendix. Coordinator-side tools: Claude Code 2.1.272 Read/Write/Edit/Bash/Agent/SendMessage/ToolSearch (process_log.md steps 001–212); `.venv/bin/python` for `code/P000_dossier_checks.py`, `code/common/lzcommon.py`, `code/common/ledger_tools.py`, `code/common/inventory_tools.py`; shell utilities `ls`, `wc`, `cat`, `head`, `sed`, `grep`, `mkdir`, `cp`.

### Software and packages (consolidated)

| Package / version | Papers (count) | Categories | Typical functions |
|---|---|---|---|
| python 3.12.13 | 100 | BKG 14, STAT 13, IDM 11, COMP 11, XEXP 8, PROJ 7, EFT 7, HALO 7, RESP 6, NUC 6, MODEL 6, EXO 4 | .venv/bin/python; .venv/bin/python from simulation root; .venv/bin/python from the simulation root |
| numpy 2.5.3 | 100 | BKG 14, STAT 13, IDM 11, COMP 11, XEXP 8, PROJ 7, EFT 7, HALO 7, RESP 6, NUC 6, MODEL 6, EXO 4 | arange, interp, trapezoid, cumsum, polyfit, log10; arange, linspace, trapezoid, argmax/argmin, isclose; arange, trapezoid, polyfit, linalg.lstsq, savez/load, cumsum |
| matplotlib 3.11.2 | 100 | BKG 14, STAT 13, IDM 11, COMP 11, XEXP 8, PROJ 7, EFT 7, HALO 7, RESP 6, NUC 6, MODEL 6, EXO 4 | Agg PNG figures; Agg PNG figures (3); Agg PNG figures (4) |
| common/lzcommon.py | 100 | BKG 14, STAT 13, IDM 11, COMP 11, XEXP 8, PROJ 7, EFT 7, HALO 7, RESP 6, NUC 6, MODEL 6, EXO 4 | GEV_TO_CM2 (GeV^-2 -> cm^2 -> fb conversion); GEV_TO_CM2 (GeV^-2 -> cm^2); LSIG, OSIG, LSIG_MASSES, OSIG_DELTAS, M_NUCLEON_GEV, M_V_GEV |
| scipy 1.18.1 | 93 | BKG 13, STAT 11, IDM 10, COMP 9, XEXP 8, PROJ 7, EFT 7, HALO 7, NUC 6, MODEL 6, RESP 5, EXO 4 | fft.rfft/irfft/next_fast_len (exact N-fold convolution); int; integrate.quad (Fermi-Dirac integrals, K moments, t(z), NFW ; integrate.quad (Feynman-parameter loop function; leptonic ph |
| pandas 3.0.5 | 88 | STAT 12, IDM 11, BKG 10, COMP 10, EFT 7, HALO 7, PROJ 6, RESP 6, MODEL 6, NUC 5, EXO 4, XEXP 4 | DataFrame tables -> CSV (inventory, f_gamma, expected by com; DataFrame tables and CSV output; ledger inspection; DataFrame tables, CSV I/O |
| WimPyDD 2.0.4 | 65 | IDM 10, STAT 9, XEXP 8, HALO 7, PROJ 7, EFT 7, MODEL 6, COMP 4, NUC 4, EXO 3 | Xe target (isotope indices), eft_hamiltonian (O4 isoscalar u; Xe target (mass, isotopes, abundance, func_w per-isotope nuc; Xe target object (a, z, spin, mass, abundance, func_w), nucl |
| none: derived by hand | 51 | COMP 10, BKG 8, IDM 8, MODEL 5, NUC 4, EFT 4, RESP 3, XEXP 2, HALO 2, EXO 2, STAT 2, PROJ 1 | A'-loop dark-Higgs width (loop function set to 1), 4-body h_; Bateman chain with purification removal (exact one-step upda; Compton kinematics, minimal path-length geometry, Poisson/li |
| nestpy 2.1.1 | 21 | BKG 6, RESP 5, NUC 3, STAT 3, EXO 2, XEXP 1, MODEL 1 | NESTcalc.GetYields (NR, LZ_WS2024) via lzcommon.nest_nr_yiel; NESTcalc.GetYields via lz.nest_nr_yields (248 keV NR yields,; NESTcalc.GetYields via lzcommon (NR with LZ Table S5 + p(E)  |
| Pillow 12.3.0 | 8 | BKG 4, STAT 3, EXO 1 | Image.open (only inside P052's header: Fig. 5 L10 digitisati; Image.open + colour-mask digitisation of the Fig. 5 L10 curv; Image.open for pixel-colour digitisation of Fig. 5 |
| PyMuPDF 1.28.2 | 7 | XEXP 2, EFT 2, IDM 1, NUC 1, BKG 1 | Document/Page.get_drawings (vector paths, colours, dashes, r; Page.get_drawings (vector-path digitisation of FigS7 curves ; Page.get_drawings vector-path digitisation of Fig. 6 (drawin |
| wimprates 0.5.0 | 4 | HALO 3, IDM 1 | earth_velocity, j2000 (validation of the from-scratch Earth-; earth_velocity, j2000 (validation of the geocentric lab velo; v_earth, earth_velocity, j2000_from_ymd (comparison only) |
| radioactivedecay 0.6.1 | 4 | BKG 4 | Nuclide('U-238').branching_fractions/progeny/decay_modes/hal; Nuclide.half_life for I-125, Xe-125, Xe-127, Xe-131m, Xe-129; Nuclide.half_life/decay_modes/branching_fractions/progeny (I |
| astropy 8.0.1 | 4 | IDM 2, PROJ 1, HALO 1 | Time, get_body_barycentric_posvel (Earth velocity), SkyCoord; Time; get_body_barycentric_posvel (built-in ephemeris) for t; bundled IERS-B table for UT1-UTC and polar motion (no downlo |
| numericalunits 1.28 | 3 | HALO 3 | km, s (unit conversion of wimprates output); km, s unit stripping for wimprates output; unit stripping for wimprates output |
| periodictable 2.1.0 | 3 | BKG 3 | Xe[124].abundance, Xe.mass; Xe[A].abundance for the nine stable isotopes; isotopic masses of Xe, F, Na, C, O, Ne, B, N, Al, P, Be, Mg, |
| iminuit 2.32.0 | 2 | STAT 2 | Minuit (migrad strategy 0, limits, fixed, minos, merrors) fo; Minuit.migrad (strategy 0, fallback 1), limits, fixed |
| sympy 1.14.0 | 1 | IDM 1 | Matrix.eigenvals, series (leading-order Higgsino splitting) |
| directdm 2.2.2 | 1 | EFT 1 | WC_3f(coeffs, DM_type).cNR(m,q), ._my_cNR(m), .ip; WC_5f(... |
| uncertainties 3.2.3 | 1 | RESP 1 | ufloat (tagged), wrap (numerical derivatives of the estimato |
| healpy 1.20.0 | 1 | COMP 1 | nside2npix, nside2pixarea, pix2ang (nside=128 D-map), mollvi |

### Agent tool calls (totals from provenance JSONs)

| Tool | Calls | Per paper |
|---|---|---|
| Read | 1722 | 17.2 |
| Bash | 1517 | 15.2 |
| Edit | 1371 | 13.7 |
| Write | 447 | 4.5 |
| Skill | 63 | 0.6 |
| ToolSearch | 24 | 0.2 |
| TaskStop | 21 | 0.2 |
| Monitor | 8 | 0.1 |
| **all** | 5173 | 51.7 |

The coordinator's own tool use (not counted above) is recorded step by step in `process_log.md`: ≈ 212 logged steps, of which ≈ 100 are agent launches/report handling, ≈ 60 Bash validations (`ledger_tools.py check/add`), and the rest Read/Write/Edit of the dossier, plan, guide, library, inventory, log and synthesis.

### Tool calls by taxonomy category

| Category | Papers | Read | Bash | Write | Edit | other |
|---|---|---|---|---|---|---|
| BKG | 14 | 239 | 185 | 60 | 213 | 6 |
| STAT | 13 | 273 | 196 | 59 | 142 | 16 |
| IDM | 11 | 203 | 202 | 51 | 134 | 22 |
| COMP | 11 | 159 | 147 | 48 | 171 | 8 |
| XEXP | 8 | 153 | 150 | 36 | 116 | 15 |
| EFT | 7 | 101 | 121 | 31 | 70 | 9 |
| HALO | 7 | 107 | 106 | 33 | 112 | 10 |
| PROJ | 7 | 144 | 105 | 31 | 86 | 12 |
| RESP | 6 | 112 | 90 | 28 | 96 | 5 |
| NUC | 6 | 91 | 77 | 26 | 66 | 5 |
| MODEL | 6 | 87 | 95 | 26 | 119 | 7 |
| EXO | 4 | 53 | 43 | 18 | 46 | 1 |

## 3. Local inputs and which papers used which parts

| File | Parts used | Papers |
|---|---|---|
| `inputs/LZ_arXiv_2609.02823_fulltext.tex` | whole paper + supplement (dossier, plan, P100); Tables I, S1–S7; Fig. 1–6 and S1–S7 captions; MSSI, veto, waveform, LEE, FV, neutron and Data-Release paragraphs (per paper, line ranges in each JSON) | dossier, plan, all papers |
| `inputs/figures_png/*.png` | Fig1 (P003 normalisation), Fig2/Fig4/Fig5/FigS1–S6 (pixel digitisation of bands, histograms, ER cloud: P004, P010, P022, P024, P033, P038, P041, P052, P056, P058), Fig3 (event position), Fig6/FigS7 (limits) | ≈ 25 papers |
| `inputs/arXiv_2609.02823_source/*.pdf` | vector paths of Fig. 1, 3, 6, S7 extracted with PyMuPDF (P003, P015, P045, P059, P070) | 7 papers |
| `output/00_evidence_dossier.md`, `01_landscape_and_plan.md`, `provenance/PAPER_GUIDE.md`, `results_ledger.csv` | framing, prior probabilities, conventions, earlier headlines | all papers |
| earlier corpus papers, work files and code | cited results, cached WimPyDD spectra/kernels (P003, P008, P011, P021, P027, P038, P050, P057, P058, P068 caches reused most), method snippets | all papers from P002 on |
| `WimPyDD/` package files (read only; WimPyC `Sun densities.tab`, response-function `.py` files) | targets, response functions, solar profile | P015, P037, P039, P046, P058, P065, P076, P094, P096 |
| `environment/ENVIRONMENT_versions.txt` | package versions | all papers |

### Local inputs by type (entries in provenance JSONs)

| Input type | Entries |
|---|---|
| corpus paper | 337 |
| dossier/plan/guide/ledger | 233 |
| corpus work file | 185 |
| corpus code | 146 |
| other | 144 |
| LZ tex | 70 |
| LZ figure PNG/PDF | 64 |

## 4. Recalled knowledge (from training memory, not from provided files)

### Recalled knowledge

Total items: 1134 in 100 papers (mean 11.3; min 0, max 49).
| Reliability | Items | Share |
|---|---|---|
| likely | 468 | 41 % |
| certain | 435 | 38 % |
| uncertain | 215 | 19 % |
| mixed/other | 16 | 1 % |

Most recall-dependent papers: P083 (49), P086 (30), P025 (24), P084 (23), P048 (22), P060 (21), P067 (21), P049 (20), P063 (20), P087 (20)

Failed or abandoned approaches recorded: 411.

The catalogue R01–R99 (Appendix, by batch) groups the items by topic with presumed sources and reliability; the per-paper JSONs hold the complete item-level lists. The most consequential *uncertain* items are external experimental limits (IceCube/ANTARES solar-WIMP limits in P076; H.E.S.S./Fermi/AMS/line anchors in P025, P084; LHC dijet/dilepton Z′ ceilings in P054; archival ROI edges in P059, P082; CRESST/PICO/SIMP exclusions in P046, P080), LZ geometry and light-yield numbers absent from the paper (P033, P042, P063, P073, P077), and low-energy atmospheric-neutrino and DSNB fluxes (P060, P087). No result at the event level (energy, band position, background exclusions) depends on an uncertain recalled number; model viability and calendar projections do (P100).

## 5. Datasets and data requests

- **Datasets used:** none fetched. One bundled WimPyDD experiment file (`WimPyDD/Experiments/DAMA_LIBRA_2019/modulation_amplitudes.tab`) was read by P028 as a local digitisation of DAMA/LIBRA-phase2 modulation amplitudes.
- **Data requests filed** (`output/data_requests/index.csv`): DR-001 (P012; also P045, P068) — LZ Data Release L10 normalisation and interval tables, important, pending; DR-002 (P052; also P090, P093) — event list and per-component 2D PDFs, important, pending; DR-003 (P093; also P077, P091) — waveform-level quantities of the candidate, important, pending. Results marked provisional: P012, P045, P052, P068 (scale only), P093, P099 (artefact kill condition). P100 recommends a further request for LZ's AmBe 200–330 keV band data (not filed).
- **WimPyDD-generated files** (outside `output/`, permitted by PROMPT §2b): P037 and P039 caused WimPyDD to (re)write response-function `.npy` files under `WimPyDD/WimPyC/Response_functions/spin_1_2/` (⁵⁶Fe, ¹H and other solar isotopes, `__init__.py`) at import/run time; P065/P094 patched the ¹⁸⁰W/⁴²Ca response modules in memory only. All other WimPyDD use went through `diff_rate`, which writes nothing; the per-paper caches listed below live under `output/work/`. Python byte-code caches (`__pycache__/*.pyc`) were also created inside `WimPyDD/` by normal imports (final check, step 211).

### WimPyDD-generated files declared

- P008: ['output/work/P008/spectra_cache.npz (our own cache of WD.diff_rate outputs; WimPyDD wrote no response-function files)']
- P027: ['output/work/P027/P027_spectra.npz (our cache of WimPyDD spectra; WD.diff_rate writes no files of its own)']
- P032: ['output/work/P032/P032_basis_spectra.npz (our own cache of WimPyDD diff_rate outputs; WimPyDD wrote no files of its own)']
- P037: ['WimPyDD/WimPyC/Response_functions/spin_1_2/56Fe_c_1_c_1.npy (rewritten by WimPyDD at import time, 2026-09-16 13:52:04, during the P037 run; not used by the an
- P038: ['output/work/P038/spectra_s_1000_full.npz (kernels x halos cache written by the script; no WimPyDD response-function files were written because diff_rate, not 
- P039: ['WimPyDD/WimPyC/Response_functions/spin_1_2/__init__.py', 'WimPyDD/WimPyC/Response_functions/spin_1_2/1H_c_1_c_1.npy', 'WimPyDD/WimPyC/Response_functions/spin_
- P044: ['none (WD.diff_rate via lzcommon.wd_rate; no response-function files written)']
- P047: ['none written by WimPyDD itself (diff_rate used, not wimp_dd_rate); our own cache output/work/P047/P047_isotope_spectra.npz']
- P058: ['none outside output/ (kernel caches: output/work/P058/cache/K_{iso,p,hig}_{400,1000,4000}_{m,p}{delta}.npz, 42 files)']
- P059: ['output/work/P059/cache/O1s_1000GeV_delta{0,50,100,150,200,250}_Eunit.npy', 'output/work/P059/cache/O1s_1000GeV_delta366_Efull.npy', 'output/work/P059/cache/L1
- P071: ['output/work/P071/extra_spectra_cache.npz (our cache of WD.diff_rate outputs for 161 corpus-variant spectra; WimPyDD wrote no response-function files)']
- P072: ['none outside output/ (per-isotope tables cached in output/work/P072/cache/w_tables.npz)']
- P075: ['none beyond output/work/P075/N_unit_grid_live.csv and output/work/P075/higgsino_grid_live.csv (script outputs; wd_rate uses diff_rate and writes no response-f
- P078: ['none beyond our own cache output/work/P078/kernels_O1_O4_1000gev.npz (WimPyDD diff_rate writes no response-function files)']
- P082: ['output/work/P082/cache/spectra_s_200.npz', 'output/work/P082/cache/spectra_s_300.npz', 'output/work/P082/cache/spectra_s_600.npz', 'output/work/P082/cache/spe
- P094: ['none outside output/ (31 cached spectra in output/work/P094/cache/*.npy)']

## 6. Unavailable tools and workarounds

| Wanted | Status | Workaround |
|---|---|---|
| Internet, HEPData, LZ Data Release | denied | numbers read from the paper's tables/figures; figures digitised (PyMuPDF vectors, Pillow pixels); three data requests filed; external limits recalled and flagged |
| NEST v2.4.x with LZ's (a, b, E0) charge-yield break; NEST ER skew model | nestpy 2.1.1 lacks the break and exposes no width accessors | p(E) implemented by hand in `lzcommon.nest_nr_yields`; skew-normal N_e model rebuilt in P056; `GetQuanta` used for widths (P098) |
| GEANT4 / full detector simulation | not available | weighted ray-march photon and neutron Monte Carlos in numpy (P033, P042, P063, P073, P077); analytic escape probabilities (P070) |
| micrOMEGAs / DarkSUSY relic and indirect codes | not available | own Boltzmann solvers (P054, P075, P086) and coupled-channel Sommerfeld solver (P084) |
| SRIM / stopping-power tables | not available | Lindhard/Lindhard–Scharff formulae and recalled ranges, flagged (P064, P067, P080, P096) |
| directdm 5/4-flavour classes | crash under numpy 2.5.3 | `WC_3flavor` and an in-script wrapper (P031) |
| astropy AltAz / IERS | unusable offline (leap-second config); `Time.ut1` needs `iers.conf.auto_max_age` set | analytic GMST horizon transform (P067, P092); explicit IERS config (P085) |
| `ps`, `pgrep`, `pkill`, `timeout`, `sysctl` | denied or absent in the sandbox | Bash until-loops, `TaskStop`, Python timing |
| Harness limits: 600 s foreground per command, 20 concurrent subagents, stream watchdog | — | WimPyDD kernel caching (14 papers), staged scripts with `--budget`/`--stage` flags, batch launches ≤ 10, three agents resumed by message |
| `node` (JavaScript) | present but not permitted | banned in PAPER_GUIDE after P046/P047 ran the dataviz palette validator |

---
# Appendix: batch-by-batch additions (chronological record, kept as written)

## Batch-1 additions (P001–P010; consolidated into the tables above at the end of the run)

### Tools/packages used in batch 1
| Tool / package | Version | Functions / modules used | Papers |
|---|---|---|---|
| scipy | 1.18.1 | stats.poisson/norm/gamma/chi2/truncnorm, integrate.quad, optimize.brentq/minimize_scalar | P001, P002, P004, P005, P006, P007, P008, P009, P010 |
| numpy | 2.5.3 | arrays, integration grids | all |
| matplotlib | 3.11.2 | Agg figures | all |
| pandas | 3.0.5 | tables/CSV | P002, P006, P007, P009, P010 |
| sympy | 1.14.0 | neutralino mass-matrix algebra (splitting formula) | P007 |
| WimPyDD | 2.0.4 | eft_hamiltonian (incl. q-dependent Wilson coefficients), diff_rate, streamed_halo_function (explicit v_min grid) | P002, P003, P005, P006, P007 |
| wimprates | 0.5.0 | v_earth, j2000_from_ymd, rate_wimp_std (validation) | P000 dossier, P006 |
| numericalunits | 1.28 | units for wimprates | P006 |
| nestpy | 2.1.1 | NESTcalc.GetYields (LZ Table S3/S5 parameters, hand-implemented p(E) break), GetQuanta fluctuations, detectors.LZ_WS2024 | P000, P009, P010 |
| pymupdf | 1.28.2 | vector-path and text extraction from Fig. 1, Fig. 6, Fig. S7 PDFs; WimPyDD manual text | P003, P007, coordinator (step 035) |
| Pillow | 12.3.0 | pixel digitisation of Fig. 4/5 PNGs | P004, P009, P010 |
| radioactivedecay | 0.6.1 | ¹²⁵Xe→¹²⁵I Bateman chain, half-lives | P010 |
| periodictable | 2.1.0 | ¹²⁴Xe abundance | P010 |
| common/lzcommon.py | corpus | constants, kinematics, halo, NEST wrappers, WimPyDD wrapper | all |
| Claude Code Agent (general-purpose subagents) | 2.1.272 | one subagent per paper; tool counts per paper in provenance/P0XX.json | P001–P010 |
| Skill (dataviz) | — | chart-style guidance | P001, P007 |
| Monitor / TaskStop / ToolSearch | 2.1.272 | background-job handling inside subagents | P002, P003, P006, P007, P008 |

### Local inputs used in batch 1
fulltext.tex (all sections; per-paper line ranges in the JSON files); Fig1_combined_recoils.pdf (P003 vector digitisation), Fig6_O1_L10_limit_stacked.pdf and FigS7_O1_as_SI.pdf (P007 digitisation); PNGs Fig1, Fig2, Fig4, Fig5, Fig6, FigS1a, FigS2, FigS7 (viewed/digitised); WimPyDD/package.py and test script (read-only, API); dossier and dossier_numbers.json; earlier papers via ledger.

### Recalled knowledge added in batch 1 (details per paper in provenance/P0XX.json)
| # | Item | Presumed source | Reliability | Papers |
|---|---|---|---|---|
| R06 | Sellke–Bayarri–Berger bound −1/(e p ln p); Kass–Raftery scale; Berger–Sellke | statistics literature | certain | P001 |
| R07 | Wilks/Cowan asymptotics; Gross–Vitells LEE | Cowan et al. 2011; Gross & Vitells 2010 | certain | P001, P008 |
| R08 | LXe γ attenuation lengths 0.3–9.8 cm (122–2615 keV); LZ TPC radius 72.8 cm | NIST XCOM-type tables; LZ instrument paper | likely | P004 |
| R09 | XENONnT 3.1 t·yr and PandaX-4T 1.54 t·yr exposures (certain); their ROI upper edges (uncertain) | XENON 2025 PRL; PandaX 2025 PRL | mixed | P005 |
| R10 | Anand et al. L10 NR reduction (q²O4 − m_N²O6 structure); L1/L5→O1, L2→O10, L3→O11, L4→O6, L15→O4; Anand c⁰=(c_p+c_n)/2 | Anand, Fitzpatrick, Haxton 2014 | likely | P003, P005, P012 |
| R11 | LZ 2024 low-energy search tolerance ≈ 3–5 signal events; 4.2 t·yr; arXiv 2410.17036 | LZ 2024 PRL | uncertain/likely | P003, P016 |
| R12 | G_F, sin²θ_W, m_Z, m_W, Z couplings, neutralino/chargino mass matrices; LEP chargino bound ≈ 92–103 GeV; thermal Higgsino 1.1 TeV; radiative chargino splitting ≈ 355 MeV; Hill–Solon loop σ ~1e-49–1e-48 cm² | PDG; SUSY literature | certain/likely | P007, P014 |
| R13 | W ≈ 13.7/13.5 eV; NEST v2 NR/ER yield functional forms; exciton-to-ion ratio α ≈ 0.06–0.2; 10–90% = ±1.2816σ | NEST papers; Dahl thesis | likely/certain | P009, P010 |
| R14 | ¹²⁴Xe 2νECEC T½ ≈ 1.1×10²² yr; KK branching ≈ 0.75 (uncertain); Te binding energies; ¹²⁵I/¹²⁵Xe decay schemes; LZ 2025 EC charge-yield reductions (uncertain) | XENON1T/LZ DEC papers; nuclear data | likely/uncertain | P010 |
| R15 | Earth orbital speed 29.8 km/s; 15 km/s projection; 2 June modulation peak; low-v_min phase reversal | Freese et al.; Lee, Lisanti, Safdi | certain | P006 |
| R16 | Xe M-response nodes near 100 and 265 keV | shell-model form factors (confirmed numerically with WimPyDD) | likely→confirmed | P002, P003 |

### Unavailable/deviations in batch 1
- Subagents P005 and P007 wrote scratch test files to the harness scratchpad outside `output/` (copied to `output/work/P005/scratch/`; P007's probe output was a transient file). Recorded in process_log steps 042, 044.
- `ps` is not permitted inside the sandbox (P003 noted); agents used file polling instead.
- WimPyDD default v_min grid truncation (P002) — fixed in `lzcommon.wd_halo` (step 047).

## Batch-2 additions (P011–P020)

### Tools/packages newly used or notable
| Tool / package | Version | Functions / modules used | Papers |
|---|---|---|---|
| WimPyDD | 2.0.4 | q- and m_χ-dependent Wilson coefficients (light mediators, L10), `diff_rate(sum_over_streams=False)` per-stream kernels, custom (v_min, δη) halo functions, targets I/W/Ge/Ar/F (tungsten response modules needed a runtime numpy injection: missing `import numpy as np` in `18xW_func_w.py`), `Xe.func_w` per-isotope nuclear responses, `nuclear_current` | P011, P012, P015, P017, P018, P020 |
| scipy | 1.18.1 | special.j1/jn_zeros (black-disk diffraction), ndimage.label (digitisation), stats.nbinom, interpolate.CubicSpline, integrate.quad, optimize.brentq/minimize_scalar | P013, P016, P020 |
| numpy | 2.5.3 | seeded 8×10⁶-neutron transport MC | P013 |
| Pillow | 12.3.0 | Fig. 5 panel digitisation (P016 top panel, P019 green curve) | P016, P019 |
| pymupdf | 1.28.2 | Fig. 6 bottom (P012, P017) and Fig. S7 (P015) vector digitisation | P012, P015, P017 |
| nestpy | 2.1.1 | (n,n′γ) hybrid loci (P013), NC-excitation loci (P019) via lzcommon | P013, P019 |
| periodictable | 2.1.0 | isotopic masses/abundances | P013 |
| radioactivedecay | 0.6.1 | (planned for P029) | — |
| Skill dataviz | — | figure styling | P014, P015, P017, P018, P020 |
| Claude Code Agent | 2.1.272 | one subagent per paper (tool counts in provenance/P0XX.json); one first run (P011) exceeded the 600 s foreground limit and was auto-backgrounded by the harness | P011–P020 |

### Local inputs newly used
Fig6_O1_L10_limit_stacked.pdf bottom panel (P012, P017); FigS7_O1_as_SI.pdf (P015); Fig5 PNG top panel and green curve (P016, P019); FigS2 PNG (P015, P020); Tables S1–S2 (P013); corpus caches: `work/P003/P003_spectra.npz` (P016, P027), `work/P007/*.csv` (P011, P014, P015, P017, P018).

### Recalled knowledge added in batch 2 (details in provenance/P0XX.json)
| # | Item | Presumed source | Reliability | Papers |
|---|---|---|---|---|
| R17 | Kinetic-mixing σ_p = 16π α α_D ε² μ²/m_A′⁴; ⟨σv⟩(χχ→A′A′) = πα_D²/m²; BaBar ε ≲ 1e-3 | dark-photon literature | certain/likely | P011 |
| R18 | Anand L10 reduction; photon-mediated dipole NR coefficients; nuclear magneton; tensor charges | Anand et al. 2014; Banks–Fox–Weiner | likely | P012 |
| R19 | (α,n) Q-values and endpoints (¹⁹F, ¹³C, ¹⁷O, ¹⁰B); Watt spectrum a=0.988 MeV, b=2.249 MeV⁻¹; ν̄=2.0; n–Xe cross-sections ~4 b (10 MeV), 2 b (100 MeV); black-disk R=1.25 A^{1/3} fm | nuclear data; Kudryavtsev; MUSUN | likely/uncertain | P013 |
| R20 | One-loop EW multiplet splittings (Higgsino 355 MeV asymptote, wino 165 MeV); chargino widths; 13 TeV DY cross-sections (≈200/23/0.67 fb at 300/500/1000 GeV); ATLAS/CMS disappearing-track and soft-lepton reach | Thomas–Wells; Cirelli et al.; LHC SUSY WG | likely | P014 |
| R21 | PICO-60 C₃F₈ 1404 kg·d; DEAP-3600 758 t·d; DAMA/LIBRA 2.46 t·yr; Q_I = 0.09; CRESST-II 52 kg·d; COSINE 0.2 t·yr; ANAIS 112.5 kg × 3 yr | experiment papers | likely/uncertain | P015 |
| R22 | 2pF parameters c=5.42 fm, a=0.57 fm for ¹³²Xe; charge radius 4.79 fm; Menéndez 2012 ⟨S_n⟩,⟨S_p⟩; two-body-current sizes | nuclear structure literature | likely/uncertain | P017 |
| R23 | Gaia-era v_esc 520–580 km/s, v0 220–250, ρ0 0.3–0.5; Lisanti k-tail form; SHM++ Sausage parameters | Evans, O'Hare & McCabe 2019; Lisanti et al. 2011 | likely | P018 |
| R24 | Atmospheric-ν flux ≈10.5 cm⁻²s⁻¹ above ~10 MeV and shape; Llewellyn-Smith NCE with Pauli blocking; k_F; NC excitation strengths; dipole-portal HNL formalism | Honda/Battistoni; Llewellyn Smith 1972 | likely/uncertain | P019 |
| R25 | Poisson–Gamma = negative binomial; Jeffreys prior; one-sided 90% = 2.706 | statistics | certain | P020 |

### Deviations/tool issues in batch 2
- WimPyDD tungsten response files lack `import numpy as np` (P015 worked around at runtime; WimPyDD not edited).
- WimPyDD bin-averaged η biases rates within ~20 keV of δ_max high by 2–6% (P018).
- P006 time-PDFs at δ = 350/380 keV computed before the v_min-grid fix (flagged by P020; noted in P006's ledger row).
- P011's probe script written to $TMPDIR (outside `output/`), transient.

## Batch-3 additions (P021–P030)

### Tools/packages newly used or notable
| Tool / package | Version | Functions / modules used | Papers |
|---|---|---|---|
| WimPyDD | 2.0.4 | per-stream kernels (`diff_rate(sum_over_streams=False)`) on explicit v_min grids (P021, P028, P030); 388 new spectra incl. O4 inelastic (P027, resumable cache); DAMA_LIBRA_2019 target with iodine quenching and bundled modulation-amplitude table (P028); q/m_χ-dependent Wilson coefficients for M1 transitions (P023) | P021, P023, P027, P028, P030 |
| scipy | 1.18.1 | integrate.solve_ivp (LSODA) Boltzmann equation (P026), special.logsumexp (P027), ndimage (P022), stats.skew (P024), special.j1 (P024) | P022, P024, P026, P027 |
| nestpy | 2.1.1 | GetQuanta with LZ_WS2024 width vector, CalculateG2, set_seed (P024) | P024 |
| radioactivedecay | 0.6.1 | Nuclide, Inventory.decay (Bateman chains to +8.39 d) | P029 |
| matplotlib | 3.11.2 | imread-based digitisation of Fig. 2/4/S1a (P024); viridis colour-map inversion for the Fig. S3 2D map (P022) | P022, P024 |
| Pillow | 12.3.0 | Fig. 3/5/S3 digitisation | P022 |
| Skill dataviz | — | figure styling | P021, P023, P025, P027, P028, P030 |
| Claude Code Agent | 2.1.272 | one subagent per paper; two first runs exceeded 600 s and were auto-backgrounded by the harness (P027 833 s; P011 earlier); `ps` denied by the sandbox (P027) | P021–P030 |
| QA subagent | 2.1.272 | footer-only condensation of P001–P020 (Edit ×48), report `provenance/QA_trim_batch1-2.md` | QA |

### Local inputs newly used
Fig. 3 PNG drift axis (P022: T_max ≈ 1049 μs, t_ev ≈ 859 μs); Fig. S3a/b PNGs (P022); Fig. 2 AmBe points (P024); Fig. S1a MSSI/L10 contours (P024); WimPyDD `Experiments/DAMA_LIBRA_2019/*.tab` (P028, treated as a bundled local dataset); corpus caches: P016 results and Fig.-5 bins (P021, P027), P008 spectra cache (P027), P007 grids (P021, P023, P025, P026), P011 outputs (P023, P025, P026), P018 halo code (P030).

### Recalled knowledge added in batch 3
| # | Item | Presumed source | Reliability | Papers |
|---|---|---|---|---|
| R26 | LZ TPC geometry: radius 72.8 cm, drift length 145.6 cm, active mass 7.0 t, LXe density 2.86 g/cm³, D_L ≈ 25 cm²/s; isolated-S2 rates from electrode emission (mHz–Hz) | LZ instrument papers | likely/uncertain | P022, P024, P029, P033 |
| R27 | QCD hadron hyperfine splittings (Δ–N, Σ*–Σ, B*–B, D*–D, J/ψ–η_c), HQET 1/m_Q scaling, Λ_QCD, glueball mass ≈ 7Λ, MiDM width Γ = μ²δ³/π | PDG; HQET literature; Chang–Weiner–Yavin | certain/likely | P023 |
| R28 | NEST NR variance (ω) form, 13-entry width vector, extraction efficiency ≈ 80%, AmBe ISO 8529-1 spectrum | NEST papers; ISO | likely/uncertain | P024 |
| R29 | Higgsino annihilation formulae (ADG/CFS), wino WW and line cross-sections, Hulthén Sommerfeld factor, Planck p_ann, f_eff, Fermi-LAT dSph / H.E.S.S. GC / AMS-02 p̄ anchors | Arkani-Hamed–Delgado–Giudice; Cirelli et al.; Planck 2018; Fermi/HESS/AMS | likely/uncertain | P025 |
| R30 | BBN photodissociation thresholds (D 2.22 MeV, ⁷Li ≈ 2.5, ⁴He 19.8 MeV), FIRAS μ ≤ 9×10⁻⁵ / y ≤ 1.5×10⁻⁵, Slatyer–Wu decaying-DM criterion, SPI sensitivity, NFW D-factor | COBE/FIRAS; Slatyer & Wu 2017; INTEGRAL | certain/uncertain | P026 |
| R31 | Kass–Raftery scale, SBB bound, Jeffreys prior, exp(entropy) effective model count | statistics | certain | P027 |
| R32 | NaI quenching Q_I ≈ 0.09, Q_Na ≈ 0.3; DAMA/LIBRA phase 152.5 d, rate ≈ 1 cpd/kg/keV, S_m errors 0.002–0.003 above 6 keVee; COSINE-100/ANAIS-112 exposures and backgrounds | DAMA/LIBRA, COSINE, ANAIS papers | likely/uncertain | P028 |
| R33 | Xe (n,γ) thermal/fast capture cross-sections; EC/IC emission schemes of ¹²⁵Xe, ¹²⁵I, ¹²⁷Xe, ¹²⁹ᵐXe, ¹³¹ᵐXe, ¹³³Xe, ¹³⁵Xe; LXe γ attenuation lengths (two brackets) | nuclear data (ENSDF/XCOM-like) | likely/uncertain | P029 |
| R34 | Sagittarius, S1, S2/shards, Gaia–Enceladus kinematics and density fractions; unbound DM fraction | Gaia-era literature (Myeong, O'Hare et al.) | likely/uncertain | P030 |
| R35 | Relativistic kinematics E* = δμ/m_N at the inelastic ceiling; FC single-event interval [0.105, 3.65] | derived; Feldman–Cousins 1998 | certain/likely | P021 |

### Deviations/tool issues in batch 3
- P022 wrote 4 probe scripts to a temp location outside `output/` (not recoverable from the scratchpad).
- P027's first run (833 s) exceeded the foreground limit and was auto-backgrounded; made resumable and re-run in the foreground.
- P006's δ ≥ 350 keV time-PDFs and P002/P006/P007 rates within 20 keV of δ_max carry the 2–6% WimPyDD bin-averaging bias (P018).
- P025 notes P011's thermal α_D is likely low by √2 (σ_eff = πα_D²/2m²); both conventions carried.

## Batch-4 and batch-5 additions (P031–P050)

### Tools/packages newly used or notable
| Tool / package | Version | Functions / modules used | Papers |
|---|---|---|---|
| directdm | 2.2.2 | WC_3flavor relativistic→NR matching and RG running (5/4-flavour classes crash under numpy 2.5.3: `np.delete` out-of-range; in-script wrapper) | P031, P044 (cross-check) |
| WimPyDD | 2.0.4 | per-isotope responses and `isotopes_list` (P032, P036, P047); WimPyC solar capture `wimp_capture`, `wimp_capture_geom`, `wimp_capture_annihilation` with Sun/White_Dwarf profiles (P039; densities normalised to M/R³); q-dependent Wilson coefficients with closures (P044); kernel caches (P034, P038, P046, P050); Fig.-1/6/S7-calibrated normalisations | P031–P050 |
| numpy | 2.5.3 | vectorised weighted photon-transport MC (P033), Woodcock tracking (P042), FFT N-fold convolutions (P034) | P033, P034, P042 |
| scipy | 1.18.1 | special.i0e (Marcum-Q, P041), fft (P034), stats.binom (P047), interpolate (P034) | P034, P041, P047 |
| astropy | 8.0.1 | Earth velocity, Galactic↔ICRS, SURF vertical frame (P042) | P042 |
| uncertainties | 3.2.3 | ufloat, wrap, error_components (P043) | P043 |
| nestpy | 2.1.1 | GetQuanta with the LZ_WS2024 width vector (P038, P043) | P038, P043 |
| Skill dataviz | — | figure styling | P033, P034, P037, P039, P042, P043, P046, P047, P048, P050 |
| `node` 26.4.0 | NOT permitted | dataviz palette validator executed by P046 (×3) and P047 (×1) before the guide banned it (step 108); cosmetic only | P046, P047 |
| Claude Code Agent | 2.1.272 | one subagent per paper; runs exceeding 600 s auto-backgrounded by the harness in P035, P042 (×2), P044, P045; `ps`/`pgrep` denied by the sandbox | P031–P050 |

### Local inputs newly used
Fig. S2 PNG inset (P038); Fig. S4 PNG (P038); Fig. S5a/S6 PNGs (P033); Fig. 4 PNG grey ER cloud (P041); veto tables S1–S2 (P042, P052); corpus caches: P021 scans (P038, P043, P050), P016 results (P032, P044), P017 responses (P040, P047), P007 grids (P037, P048), P014 tables (P048), P015 tables (P046), P038 acceptance (P043, P050), P023 lifetimes (P042).

### Recalled knowledge added in batches 4–5 (details in provenance/P0XX.json)
| # | Item | Presumed source | Reliability | Papers |
|---|---|---|---|---|
| R36 | directdm operator basis and normalisation; Anand L_i identifications; anapole/EDM NR reductions; EFT validity criteria | Bishara–Brod–Grinstein–Zupan; Anand et al.; Fitzpatrick et al. | likely | P031, P044 |
| R37 | Xe M multipole selection rules; xenophobic f_n/f_p = −0.7; neutron skin ≈ 0.15 fm; two-body-current isospin structure | nuclear theory literature | certain/uncertain | P032 |
| R38 | LXe photoelectric/pair coefficients; ²¹⁴Pb branches; U/Th/K/Co line mix; Lambertian wall emission | XCOM-like tables; ENSDF | likely/uncertain | P033 |
| R39 | Cosine-estimator variance Z = a₁√(N/2); Neyman–Pearson; XLZD-scale 60 t at 80% live | statistics; XLZD planning | certain/uncertain | P034, P050 |
| R40 | XENON1T 1 t·yr, ROI 4.9–40.9 keV, 1 TeV SI limit ≈ 8e-46 cm²; Poisson/FC constants | XENON1T 2018 PRL; Feldman–Cousins | likely/certain | P035, P045, P059 |
| R41 | ¹²⁹Xe*/¹³¹Xe* levels (39.6/80.2 keV), IC coefficients, inelastic/elastic SD structure-factor ratio R (Baudis et al. 2013); LZ 2024 SD-n limit ≈ 4e-41 cm² | nuclear data; Baudis et al.; LZ 2024 | certain/uncertain | P036 |
| R42 | SU(2)×U(1) multiplet Z couplings, one-loop splittings (Cirelli–Fornengo–Strumia), thermal masses of higher multiplets | Minimal DM literature | certain/uncertain | P037 |
| R43 | NS/WD benchmarks (1.5 M☉/12 km; 1 M☉/8000 km), Pauli/Fermi momenta, kinetic-heating T ≈ 1700 K, JWST NIRCam sensitivity, M4 WD luminosities, IceCube solar-WIMP scale | Baryakhtar et al. 2017; JWST docs; IceCube | likely/uncertain | P039 |
| R44 | Relativistic two-body kinematics; PDG proton spectrum; proton dipole form factor; Bringmann–Pospelov CRDM flux; CRDM exclusions; SURF rock composition | PDG; Bringmann & Pospelov 2019 | likely/uncertain | P040 |
| R45 | LXe D_T ≈ 50–60 cm²/s; electron lifetime ≳ 5–10 ms; extraction efficiency ≈ 0.8; ⁸³ᵐKr map amplitude ≲ 20%; O₂ attachment; α S2 sizes | LZ/XENON instrument papers | likely/uncertain | P041 |
| R46 | SURF coordinates, GMST, Skin/cryostat/OD geometry, veto light yields, S1 merging time (~100 ns), MiDM width | LZ instrument paper; astronomy formulae | certain/uncertain | P042 |
| R47 | W = 13.5 ± 0.2 eV; Gaussian propagation | NEST | likely/certain | P043 |
| R48 | Cowan boundary-aware asymptotics; Baxter 3σ threshold and PCL M_min = 0.16; flip-flopping under-coverage | Cowan et al. 2011; Baxter et al. 2021 | certain/likely | P045 |
| R49 | CaWO₄ light yields; ²⁰⁶Pb 103 keV recoils; ¹⁸⁰W α decay; radiogenic-neutron ceiling ≲ 10 MeV | CRESST papers | likely/uncertain | P046 |
| R50 | Xe enrichment practice (EXO-200/KamLAND-Zen/nEXO ¹³⁶Xe; ¹²⁹Xe MRI supply; centrifuge separation; costs) | 0νββ literature | likely/uncertain | P047 |
| R51 | 100 TeV Drell–Yan cross-sections, tracklet radii/backgrounds, ISR log, muon-collider σ(f f̄), FCC-ee W-parameter sensitivity, facility dates | FCC/MuC CDRs (recalled) | uncertain | P048 |
| R52 | SURF muon flux 5.3e-9 cm⁻²s⁻¹ and ⟨E⟩ ≈ 300 GeV; Mei–Hime yield 3.5e-4 n/(μ g cm⁻²); zenith cos^3.5; n–p/n–O cross-sections; Gd capture times; β-n emitters | Mei & Hime 2006; Kudryavtsev | likely/uncertain | P049 |
| R53 | XENONnT 4 t / PandaX-4T 2.7 t fiducial masses; XLZD 40–60 t from ~2032 | collaboration talks (recalled) | likely/uncertain | P050 |

### Deviations/tool issues in batches 4–5
- `node` executed by P046/P047 (dataviz validator) — not permitted; guide updated (step 108).
- WimPyDD `eft_hamiltonian` shared-default-argument pitfall found by P031; `lzcommon.wd_hamiltonian` fixed with closures (step 102); corpus audited, no result affected (step 103).
- `lz.wd_halo()` with no day = Sun-frame halo, not annual average (P035); docstring and guide updated (steps 098–099); rates within 20 keV of δ_max carry ×1.08–10 halo-frame ambiguity — stated in the synthesis.
- directdm 5/4-flavour classes incompatible with numpy 2.5.3 (P031).
- Harness auto-backgrounding of >600 s runs (P035, P042, P044, P045); all completed or were rerun in the foreground.

## Batch-6 and batch-7 additions (P051–P070; P057 appended below when reported)

### Tools/packages newly used or notable
| Tool / package | Version | Functions / use | Papers |
|---|---|---|---|
| WimPyDD | 2.0.4 | `diff_rate` per-isotope and per-stream kernels (P055, P062, P068), q-dependent light-mediator coefficients via closures (P051), δ < 0 exothermic kinematics (P058), L10 q²-coefficient closures and F/Xe/W/I targets (P065, P067, P068), flat-halo kernel trick (P067); 180W response file patched in memory (P065) | P051, P054, P055, P058, P059, P062, P065, P067, P068, P069 |
| nestpy | 2.1.1 | LZ_WS2024 GetYields/GetQuanta with LZ Table S3–S5 parameters (P053, P056, P064), ER width vector and skewness probes (P056), yields through `lzcommon.nest_nr_yields` (P052) | P052, P053, P056, P064 |
| iminuit | 2.32.0 | migrad/minos profile fits of the three-sample 2D likelihood (P052) | P052 |
| Pillow | 12.3.0 | colour-mask digitisation of the Fig. 5 L10 curve (P052) | P052 |
| pymupdf | 1.28.2 | vector extraction of Fig. 3 (event position, star glyph) and Fig. 6 (P059, P070) | P059, P070 |
| radioactivedecay | 0.6.1 | ²¹⁴Pb / ²³⁸U-chain branching and half-lives (P053, P063) | P053, P063 |
| healpy | 1.20.0 | Mollweide sky maps of the χ₂→χ₁γ line intensity (P066) | P066 |
| wimprates | 0.5.0 | `earth_velocity` validation of the from-scratch Earth-velocity frame (P055) | P055 |
| numericalunits | 1.28 | unit handling with wimprates (P055) | P055 |
| astropy | 8.0.1 | ephemeris and Galactic↔ICRS rotation (P067); AltAz frame unusable offline (leap-second config), analytic GMST used instead | P067 |
| scipy | 1.18.1 | solve_ivp Radau relic ODE (P054); qmc.LatinHypercube, spearmanr (P068); brentq root-finding everywhere; ndimage/skew (P056); special.erf/erfc/ndtr, exponential integral E2 (P070) | P051–P070 |
| Skill dataviz | — | figure styling | P054, P055, P058, P061, P062, P063, P066, P068, P069, P070 |
| Claude Code Agent | 2.1.272 | one subagent per paper; runs > 600 s auto-backgrounded by the harness in P052 (toys), P054 (×2, stopped), P055, P056 (×2), P058 (2364 s), P059 (×2), P062, P065, P067, P068 (stopped, rerun 567 s); agents P056, P065, P067 paused on background jobs and were resumed by coordinator messages | P051–P070 |
| none: derived by hand | — | analytic derivations (light-mediator propagator, P051; relic formulae and Z′ widths, P054; ⟨σv⟩ for MiDM, P065; posterior algebra, P061; Fisher/KL, P062; escape probability, P070) | P051, P053, P054, P061, P062, P063, P064, P065, P066 |

### Local inputs newly used
Fig. 3 PDF vector paths (P070) and Fig. 6 (P059); Fig. 2/4/5 PNGs re-digitised (P056); Fig. 4 and Fig. S4 PNGs (P058); Fig. S1a/b/c PNGs (P052); veto Tables S1–S2 and Eq. S1 (P052); LEE section and Tables S6/S7 (P052, P062); MSSI supplement table (P058, P063); corpus caches: P003/P012/P021 spectra (P052), P011 σ_p grid (P054, P058), P021 κ̂ scan (P054), P007 grids (P054, P068), P018 halo code (P068), P030 stream CSV and P006 date code (P055), P042 topology CSVs (P065), P043 systematics table (P068), P016 results (P052), P004/P022 Fig. 5 digitisations (P052), P033 wall-MSSI maps (P070), P046 multi-target code (P065, P067), P023 MiDM cache (P065); `WimPyDD/package.py` and the 180W response file (read only; P058, P065).

### Recalled knowledge added in batches 6–7 (details in provenance/P0XX.json)
| # | Item | Presumed source | Reliability | Papers |
|---|---|---|---|---|
| R54 | Dark-photon σ_p formula, χ₂→χ₁νν̄ width, A′ decay length; BaBar/NA48/beam-dump/LHCb ε bounds | Holdom; Tucker-Smith & Weiner; BaBar 2014/2017; NA48/2; BESST recasts; LHCb 2020 | certain / likely / uncertain (×3 in ε) | P051 |
| R55 | LZ TPC radius 72.8 cm and drift 145.6 cm, LXe density 2.86–2.9, ²¹⁴Pb decay scheme and β branches, XCOM attenuation lengths, Rn tag principle, Rn activity 1–5 μBq/kg, purification 1–4 t/day | LZ instrument/SR1 papers; ENSDF; NIST XCOM; LZ arXiv:2508.19117 | likely / uncertain | P053, P070 |
| R56 | Planck Ωh² = 0.120; Kolb–Turner / Gondolo–Gelmini relic formulae; g* = 86.25; Z′ widths and B−L charges; dijet g_q ceilings; B−L dilepton m/g′ ≳ 20 TeV; LEP-II 7 TeV; Z–Z′ mixing ≲ 1e-3; Hulthén Sommerfeld; Planck p_ann | PDG; Kolb & Turner; Gondolo & Gelmini 1991; ATLAS/CMS dijet+dilepton (recalled); LEP EWWG; Slatyer 2016 | certain / likely / uncertain | P054 |
| R57 | J2000 equatorial→Galactic matrix, obliquity, orbital speed, 2023 equinox, LLS Earth-velocity vectors; SHM++ Sausage parameters; Sgr / S1 / S2 / Helmi / retrograde-shard kinematics and fractions | Hipparcos; Almanac; Lee–Lisanti–Safdi 2013; Evans–O'Hare–McCabe 2019; Myeong 2018; O'Hare 2018/2020; Koppelman 2019 | certain (frames) / likely (speeds) / uncertain (signs, fractions) | P055 |
| R58 | NEST Var(N_e) = r(1−r)N_i + (σ_p N_i)², skew-normal N_e with skew switched off above N_q = 1e4; extraction efficiency ≈ 0.80; β endpoints ²¹²Pb/¹⁴C/²¹⁴Pb; Azzalini moments | NEST v2 source; LZ SR1; nuclear tables | likely / uncertain / certain | P056 |
| R59 | LZ SR1 (0.90 t·yr, g1 0.114), LUX 2021 EFT (0.092 t·yr), PandaX-II 2019 (0.148 t·yr), XENON1T 2018 (1.0 t·yr) exposures, ROI edges and plateau efficiencies; Poisson/PLR 90 % UL constants; Anand L10 reduction | respective publications (recalled); Cowan et al. 2011; Anand et al. 2014 | likely / uncertain / certain | P059, P069 |
| R60 | CEνNS cross-section and Q_W; Keil–Raffelt–Janka pinched spectrum; DSNB flux ≈ 30 cm⁻²s⁻¹ and Super-K limit; SN 3e53 erg, ⟨E⟩ 12–18 MeV; SN 2023ixf (M101, 6.4 Mpc, 19 May 2023); no Galactic SN / SNEWS alert since 1987A; Betelgeuse 0.2 kpc; Llewellyn-Smith NC form factors; IceCube diffuse flux; ν–N cross-sections (Gandhi et al.); SGR 1806-20 flare; GRB/solar-flare ν fluences; Super-K 22.5 kt and IBD σ | Freedman 1974; KRJ 2003; Beacom 2010; Bays 2012; Gandhi et al. 1998; Hurley 2005; Vogel & Beacom | certain / likely / uncertain (×2–3) | P060 |
| R61 | Kass–Raftery scale; Sellke–Bayarri–Berger bound; Lindley information; Gelman–Loken forking paths; Gamma–Poisson predictive; historical ~3σ single-event anomaly reference class | Kass & Raftery 1995; SBB 2001; Lindley 1956; Gelman & Loken 2014; field history | certain / likely | P061 |
| R62 | Inelastic kinematics E±, E*, v_min*; Fisher/KL and two-hypothesis N_3σ; Poisson-rate Fisher term; σ(a₁) ≈ √(2/N) | Tucker-Smith & Weiner 2001; Cowan et al. 2011; standard | certain | P062 |
| R63 | ²³⁸U SF ν̄ = 2.01, Terrell width 1.08, Watt a/b; prompt-γ multiplicity 6.5; ²³²Th SF branching; LZ PTFE ~1 t and U(early) 3–100 μBq/kg; Ti cryostat 2.2 t and U assays; R11410 PMT U(early) 1–3 mBq; (α,n) yield 1e-4 per chain decay in PTFE; XCOM attenuation Xe/LAB/Ti/SiO₂/water; LZ Skin/RFR/OCV geometry; OD/Skin thresholds; veto silent probabilities (estimated) | Holden & Zucker; Terrell 1957; SOURCES-4C; LZ radioassay EPJC 80, 1044; Ti assay Astropart. Phys. 96; LZ NIM A 953; Mei–Zhang–Hime 2009 | likely / uncertain / estimated | P063 |
| R64 | Lindhard L = kg/(1+kg) with ε and k; Lewin–Smith g(ε); W = 13.7/13.5 eV; NEST v2 NR yield functional forms; Thomas–Imel box model; Lenardo 2015 k ≈ 0.139; LUX D-D k ≈ 0.17; Aprile 2006 / Manzur 2010 / Plante 2011 L_eff; N_ex/N_i ≈ 1; AmBe endpoint 11 MeV | Lindhard 1963; Lewin & Smith 1996; NEST; Thomas & Imel 1987; Lenardo et al. 2015; LUX 2016; PRL 97/PRC 81/PRC 84; Dahl 2009 | certain / likely / uncertain | P064 |
| R65 | MiDM NREFT coefficients (c1, c5 ∝ 1/q²; c4, c6), Γ(χ₂→χ₁γ) = μ²δ³/π, nucleon g-factors, thermal ⟨σv⟩ 2.2e-26 and Dirac-like 4.4e-26, millicharge annihilation formula, ΣN_cQ_f² = 8, MiDM literature moment ~1e-3 μ_N, loop-elastic estimate | Fitzpatrick 2012 / Anand 2014; Chang–Weiner–Yavin 2010; Steigman–Dasgupta–Beacom 2012; PDG | certain / likely / uncertain | P065 |
| R66 | NFW r_s = 20 kpc, R☉ = 8.2 kpc, Einasto α = 0.17, Burkert r_c = 9 kpc; Planck 2018 cosmology; INTEGRAL/SPI narrow-line sensitivities (point 3e-5, diffuse 0.3–3e-5 ph cm⁻²s⁻¹); Teegarden & Watanabe 2006, Calore et al. 2023 line searches; COMPTEL band; COSI and e-ASTROGAM/AMEGO-X sensitivities; Gruber 1999 CXB parametrisation; decaying-DM isotropic intensity formula; positronium continuum; SPI instrumental lines | MW mass models; Planck 2018; Roques 2003; Vedrenne 2003; Wang 2007/2020; Diehl 2006; Tomsick 2019/2023; De Angelis 2017; Gruber 1999; Essig 2013; Siegert 2016; Weidenspointner 2003 | certain / likely / uncertain (×2–3) | P066 |
| R67 | Natural isotopes/abundances of Br and Ag; SRIM-scale recoil ranges (F in CF₄, S in CS₂, Xe in LXe, Ag/Br in emulsion, W in CaWO₄; exponent ~0.8); ideal-gas densities and molar masses; directional thresholds (1 mm gas, 100 nm NIT); 30° resolution / 70 % head-tail assumption; SURF coordinates; GMST formula; Cygnus apex; CYGNUS-1000 and NEWSdm designs; Radon-transform directional rate; columnar recombination | nuclear tables; SRIM systematics; DMTPC/DRIFT/NEWSdm papers; USNO; Vahsen et al. 2020; Gondolo 2002; Nygren 2013 | certain / likely / uncertain | P067 |
| R68 | Gaia-era priors: ρ₀ 0.3–0.5 GeV cm⁻³, v_esc 500–600 (central 528–580: Piffl 2014, Deason 2019, Monari 2018), v₀ 220–250, V☉ ± 5 km/s; Lisanti tail form with k ≈ 1.5–3.5; v₀–v_esc positive correlation; Latin hypercube; Gaussian copula (Iman–Conover); binned first-order Sobol estimator | de Salas & Widmark 2021; Piffl 2014; Deason 2019; Schönrich–Binney–Dehnen 2010; Lisanti 2011; McKay 1979; Iman & Conover 1982; Plischke 2010 | likely / certain | P068 |
| R69 | XENONnT SR0+SR1 3.1 t·yr and PandaX-4T Run0+Run1 1.54 t·yr (cited by LZ); XENON1T/LZ-SR1/LUX/PandaX-II exposures and ROI edges; fiducial masses 4.0/2.7 t; post-2023 untouched exposures (uncertain); run periods; Gamma–Poisson predictive; Asimov Z; CLs constants; trials-factor formula | PRL 135, 221003; PRL 134, 011805; PRL 121, 111302; PRL 131, 041002; PRL 118, 021303; conference reports 2024–25 (uncertain); Cowan et al. 2011 | certain / likely / uncertain | P069 |
| R70 | Operator↔Lagrangian map (O1↔L1, O4↔L15, O6↔L4, O10↔L2, O11↔L3); Chernoff half-χ² asymptotics; Baxter toy-p0 prescription; NEST S1–S2 anticorrelation as origin of the low-d signal tail | Anand et al. 2014; Chernoff 1954; Baxter et al. 2021; NEST | likely / certain | P052 |
| R71 | atomic mass unit, G_F, sin²θ_W, α, m_Z, ħ; t_U = 13.8 Gyr; ±1.5σ acceptance 0.866; exothermic σ ∝ 1/v; LZ-2024 tolerance ~3 events (uncertain); Planck photon-channel window (via P026) | PDG; Planck 2018; Graham et al. 2010 | certain / uncertain / likely | P058 |

### Datasets and data requests (batches 6–7)
- DR-002 filed by P052 (step 147): LZ Data Release science-sample event list and per-component 2D PDFs; priority important; status pending. P052's toy-calibrated significances are provisional pending DR-002.
- P068's d₁₀ scale inherits P012's provisional DR-001 factor (bands and ratios unaffected).
- No dataset was fetched; all inputs are the LZ paper files and corpus outputs.

### Deviations/tool issues in batches 6–7
- Harness auto-backgrounding of > 600 s runs: P052, P054 (×2), P055, P056 (×2), P058, P059 (×2), P062, P065, P067, P068; all completed, were stopped and rerun faster, or read from caches afterwards. P056, P065 and P067 paused on background jobs and were resumed by coordinator messages (prompts now say "foreground only").
- Scratch files written to `$TMPDIR` by P054 (timing test), P067 (run-1 copies), P068 (three tests) — outside `output/`; results unaffected.
- Agents deleted their own large caches after use (P056 195 MB `H3_store.npz`; P068 60 MB halo cache) — regenerable, noted in their details.md.
- `lzcommon.vmin_kms` returns negative values for δ < 0 below E* (P058); WimPyDD and `E_R_range_keV` are correct; note added to PAPER_GUIDE (step 148) rather than patching the frozen-in-use library.
- P053, P067, P068 reported that a cited paper (P033 or P055) "did not exist": those papers finished after the agents read the corpus; the agents substituted earlier papers. No result affected.
- P062 refuted the coordinator's proposed δ(m) "banana" hypothesis (δμ/m_N = const has the wrong sign); the paper reports the KL ridge instead.
- `ps`/`pgrep`/`pkill`/`timeout` denied or absent in the sandbox (P068); `sleep` blocked by the harness (P065).

#### P057 addendum (batch 6–7)
- Tools: WimPyDD 2.0.4 (`coeff_squared_list`, response cross terms, Sun-frame `streamed_halo_function`), scipy `cluster.hierarchy` (dendrograms), toy-MC N_3σ; Monitor tool used once to watch a cache directory; first run auto-backgrounded (completed).
- Local inputs: P050 annual-average inelastic spectra cache (reused, cited); P003/P012/P044/P050 scripts for conventions.
- R72 | NREFT response coefficient functions; Anand Lagrangian–operator identities (O6 = E_R × O10 structure); LZ-2024 SI limit ~few e-47 cm² at 1 TeV; 2024 tolerance 3–5 events; KL/Hellinger/Gaussian LLR statistics; Okabe–Ito palette | Fitzpatrick et al. 2012; Anand et al. 2014; LZ PRL 135, 011802 (recalled); standard statistics | certain / likely / uncertain | P057

## Batch-8 and batch-9 additions (P071–P088; P086, P089, P090 appended when reported)

### Tools/packages newly used or notable
| Tool / package | Version | Functions / use | Papers |
|---|---|---|---|
| WimPyDD | 2.0.4 | per-isotope kernel factorisation with `diff_rate(isotopes_list={element: [...]})` (P072, P078), δ < 0 exothermic kernels (P072), q-dependent light-mediator closures (P071, P074), F/Xe/W targets and `Xe.func_w`/`nuclear_current` responses (P078), per-stream kernels for solar capture (P076), spectra caches for global fits (P082), Higgsino O1 kernels (P085); `coeff_squared_list` (P057) | P071, P072, P074, P075, P076, P078, P079, P080, P081, P082, P085, P087, P088 |
| WimPyC (WimPyDD) | 2.0.4 | `wimp_capture`/`wimp_capture_geom` with the AGSS09ph Sun profile (`Sun densities.tab`) used to validate an own Gould-capture integral | P076 |
| astropy / astropy-iers-data | 8.0.1 | `Time`, `get_body_barycentric_posvel` (ERFA built-in ephemeris, offline), Galactic↔ICRS; `Time.ut1` requires `iers.conf.auto_max_age` set explicitly; AltAz unusable offline | P085 |
| wimprates / numericalunits | 0.5.0 / 1.28 | `earth_velocity` cross-check of the exact observer velocity | P085 |
| nestpy | 2.1.1 | LZ_WS2024 yields via `lzcommon` for CEνNS/NR placement (P087) and SIMP recoil signatures (P080) | P080, P087 |
| iminuit | 2.32.0 | used indirectly through P052's likelihood machinery (P090 in progress) | P090 |
| scipy | 1.18.1 | `solve_ivp` (DOP853 complex, coupled-channel Sommerfeld, P084; Radau relic ODE, P075), `linalg.solve`, `stats.beta/binomtest/poisson/gamma`, Kaplan–Meier by hand (P083), `cluster.hierarchy` (P057) | P071–P088 |
| multiprocessing (stdlib) | — | parallel per-isotope kernel generation (P072); parallel kernel workers (P058) | P058, P072 |
| Skill dataviz | — | figure styling (JS palette validator skipped per guide) | P072, P076, P081, P082, P083, P084, P085, P087, P088 |
| Claude Code Agent | 2.1.272 | one subagent per paper; > 600 s runs auto-backgrounded in P071 (memory crash, fixed with float32), P088 (300 s abort); 20-concurrent-subagent cap hit at batch 10a launch (P097, P098 delayed); stream-watchdog stalls in P094, P096, P097 (resumed by message) | P071–P098 |
| none: derived by hand | — | propagator factorisation (P074), Boltzmann/Griest–Seckel algebra (P075), tree-level multiplet annihilation and Hisano potentials (P084), Rutherford/stopping (P080), Kaplan–Meier (P083), Migdal dipole sum rules (P095) | P074, P075, P076, P080, P083, P084, P095 |

### Local inputs newly used
LZ tex: LEE supplement (P071), MSSI supplement table and veto Tables S1–S2 (P073, P090), Fig. 4/S1/S2/S4 captions (P072), Waveform Analysis section (P077, P093), Table I neutrino rows (P087), FV paragraph and sidebands (P079), Fig. S7 non-xenon curves via P015 (P082), bibliography entries (P082). Corpus caches: P008 toy code and spectra (P071), P058 kernels (P072), P021/P038/P059 spectra and MC efficiencies (P082), P011/P054/P007 coupling grids (P075, P076), P039 capture tables and response files (P076), P070 FV contours and MSSI maps (P079, P093), P050 spectra/background bins (P079, P087, P088), P027 model weights and cache (P081), P061 hyper-prior tables (P081, P083), P046/P047/P057/P062/P067 tables (P088), P032 tables and script (P078), P055 frame code (P085), `WimPyC/Sun densities.tab` (P076).

### Recalled knowledge added in batches 8–9 (details in provenance/P0XX.json)
| # | Item | Presumed source | Reliability | Papers |
|---|---|---|---|---|
| R73 | Bonferroni/Šidák; Gross–Vitells 1D/2D formulae; Nyholt–Cheverud, Li–Ji, Galwey, Gao eigenvalue M_eff estimators; SBB bound | Gross & Vitells 2010; Vitells & Gross 2011; Nyholt 2004; Li & Ji 2005; Galwey 2009 | certain / likely | P071 |
| R74 | Exothermic nσv velocity-independence; dark-disk lag/dispersion/density (uncertain); Bayes-factor prior conventions | Graham et al. 2010; Read et al. 2008; Kass & Raftery | certain / uncertain | P072 |
| R75 | LZ Skin thickness 4–8 cm, PTFE 1–3 cm, Ti 0.8 cm, vacuum/water gaps, OD GdLS 61 cm, Skin/OD thresholds (2.5/4.5 phd ↔ keV), PMT-array effective density; Klein–Nishina; XCOM attenuation anchors; material densities; Ti photoelectric Z^4.5 scaling; γ lines | LZ NIM A 953 (2020); OD paper; NIST XCOM; nuclear tables | likely / uncertain / certain | P073 |
| R76 | Propagator dressing of NR coefficients; operator q-power counting; photon-dipole NR coefficients; dark-photon proton-only coupling; 3–5 event low-energy tolerance (uncertain); light-mediator SI literature | Fitzpatrick et al. 2013; Fornengo–Panci–Regis 2011 | certain / likely / uncertain | P074 |
| R77 | M_Pl, s₀, ρ_c; g*(T) table; Higgsino freeze-out cross-section (ADG); Griest–Seckel; Gondolo–Gelmini collision term; Moroi–Randall non-thermal yield; BBN T_RH ≥ 4–5 MeV; ε ≲ 1e-3; χ–χ̄ oscillation; UV freeze-in | Kolb & Turner; Arkani-Hamed–Delgado–Giudice 2006; Griest & Seckel 1991; Gondolo & Gelmini 1991; Moroi & Randall 2000; Hannestad 2004; Tulin–Yu–Zurek 2012; Hall et al. 2010 | certain / likely | P075 |
| R78 | Solar constants; AGSS09ph fidelity; ρ_χ 0.3–0.4; evaporation mass 3–4 GeV; IceCube (2017–2022) and ANTARES solar-WIMP limits at 1 TeV (uncertain ×3); channel factors; muon stopping in the core; inelastic thermalisation stall | Serenelli et al. 2009; Gould 1987; IceCube EPJC 77, 146; ANTARES PLB 759; Nussinov–Wang–Yavin 2009; Menon et al. 2010 | certain / likely / uncertain | P076 |
| R79 | LXe optics (n = 1.69, Rayleigh 29–40 cm, PTFE R ≥ 0.95, PMT QE 0.30, TTS 9 ns FWHM), 100 MS/s DAQ, grid transparency; singlet/triplet lifetimes 2.2–4.3 / 21–27 ns; 45 ns recombination component; I_s/I_t for ER/NR/α/fission fragments; LUX/XMASS few-ns differences; ¹³¹ᵐXe 164 keV; LZ PSD paper arXiv:2603.26877 exists (content unknown) | Hitachi et al. 1983; Kubota 1979; Solovov 2004; Neves 2017; Hamamatsu; LUX 2018; XMASS 2018 | certain / likely / uncertain | P077 |
| R80 | Xenophobic f_n/f_p ≈ −0.7 (Feng et al. 2011); M_J multipole selection rule; 0νββ-grade ¹³⁶Xe enrichment ≈ 90 %; two-body currents few–30 %; Σ″/Σ′ 1/3:2/3 | Feng, Kumar, Marfatia, Sanford 2011; Hoferichter et al. 2016 | certain / likely | P078 |
| R81 | Asimov Z and 5σ p-value; TPC radius/drift; LXe density 2.86 | Cowan et al. 2011; LZ papers | certain / likely | P079 |
| R82 | Rutherford cross-section; SURF depth/rock composition; atmospheric column; Thomas–Fermi screening; Lindhard–Scharff stopping; MIP dE/dx in Xe; nuclearite σ and dE/dx; monopole effective charge (Ahlen–Kinoshita); Parker/MACRO bounds; FC 4.36; heliospheric/Earth B fields; LZ-2024 SI limit ~1e-47; SIMP exclusion regions (schematic) | textbooks; De Rújula & Glashow 1984; Ahlen & Kinoshita 1982; MACRO 2002; Feldman & Cousins 1998; Chuzhoy & Kolb 2009; Bramante et al. 2018 | certain / likely / uncertain | P080 |
| R83 | Poisson–Gamma = negative binomial; Exponential(1) posterior after one event under a log-uniform prior; calendar arithmetic; three-hypothesis Bayes algebra | standard Bayesian statistics | certain | P081 |
| R84 | Archival exposures/ROI edges (LZ SR1 0.90 t·yr, g1 0.114; XENON1T 1.0; LUX 3.35e4 kg·d; PandaX-II 54 t·d; XENONnT 3.1; PandaX-4T 1.54); PICO-60 CF₃I 1335 kg·d and PICO 2023 inelastic paper; CDMS-II 2011 inelastic search; absence of NaI/Ge > 200 keV analyses (statement of ignorance); CaWO₄ W fraction | LZ/XENON/LUX/PandaX/PICO/CDMS publications | certain / likely / uncertain | P082 |
| R85 | 45 historical anomalies (DAMA, CoGeNT, CDMS-Si, CRESST-II, XENON1T ER excess, 750 GeV diphoton, Fermi 130 GeV line, 3.5 keV line, BICEP2, OPERA, LEP/LHC Higgs hints, CDF top, IceCube PeV/Glashow, KM3-230213A, ANITA, MiniBooNE/LSND, EDGES, R_K, g−2, CDF W mass, X17, reactor/gallium anomalies, HdM 0νββ, DAMPE, CDF Wjj, top A_FB, HERA high-Q², ALEPH 4-jet, Θ⁺, AGASA, …) with outcomes and dates; 2 uncertain items excluded | respective publications (recalled) | 28 certain / 15 likely / 2 uncertain | P083 |
| R86 | EW constants; para-positronium normalisation; wino tree rate and Hisano Sommerfeld matrices; CFS one-loop splitting; wino resonance 2.3–2.5 TeV; Hulthén; Slatyer inelastic criterion; thermal masses of Y ≥ 1 multiplets (uncertain); Planck p_ann, f_eff; Fermi/H.E.S.S./AMS/line anchors (uncertain); EW Sudakov corrections | Hisano et al. 2005; Cirelli–Fornengo–Strumia 2006; Slatyer 2010/2016; Planck 2018; Fermi 2015; H.E.S.S. 2016/2018/2022; Cuoco 2017; Baumgart et al. | certain / likely / uncertain | P084 |
| R87 | IAU constants (GM☉, GM_E, AU, R_E); J2000 Galactic matrix; SURF coordinates; Schönrich peculiar velocity and errors; Baxter σ(v₀); Alenazi–Gondolo v_∞ mapping; focusing peak ~1 March (Lee–Lisanti–Peter–Safdi 2014); ERFA epv00 accuracy; Earth density; Higgsino loop σ ~1e-48 | IAU 2015; Hipparcos; SBD 2010; Baxter 2021; Alenazi & Gondolo 2006; LLPS 2014; SOFA; Hill & Solon 2014 | certain / likely | P085 |
| R88 | Low-q sin²θ_W = 0.23867; Helm parameters; Xe neutron skin; atmospheric-ν flux 10.5 cm⁻²s⁻¹ ± 25 % and spectral shape (Battistoni/Honda); SURF site factor; ⁸B/hep fluxes; DSNB flux/spectra; KRJ α = 2.5; seasonal 1–4 % and solar-cycle 5–10 % modulation; Billard discovery-limit definition | Billard–Strigari–Figueroa-Feliciano 2014; Battistoni 2005; Honda 2011; SNO/B16-GS98; Beacom 2010 | certain / likely / uncertain | P087 |
| R89 | XENONnT 4.0 t / 60 % live, PandaX-4T 2.7 t / 60 %, XLZD 60 t 400 keV ROI from 2032, LZ running to 2032, CRESST-scale 1–10 kg·yr, CaWO₄ W-band background 0.01–0.1 /kg·yr (placeholder), nEXO-class 5 t ¹³⁶Xe; Gamma–Poisson conjugacy; Asimov Z | collaboration papers/design books (recalled); Cowan et al. 2011 | likely / uncertain / certain | P088 |

### Datasets and data requests (batches 8–9)
- No dataset fetched. DR-001 (P012/P045) and DR-002 (P052) remain pending; P082 and P088 inherit no provisional dependence. DR-003 (P093, batch 10) filed at step 175.

### Deviations/tool issues in batches 8–9
- Harness auto-backgrounding: P071 (silent memory death of a 50k×654 float64 toy matrix; fixed with float32 and memory release), P088 (300 s abort, rerun 50–73 s). All other batch 8–9 runs completed in the foreground.
- Scratch files in `$TMPDIR`: P075 (WimPyDD timing test), P073 (three dbg73*.py diagnostics; location not stated). Results unaffected.
- 20-concurrent-subagent cap (harness) reached at the batch-10a launch: P097/P098 refused twice, launched after slots freed (steps 161–166).
- Stream-watchdog stalls ("no progress for 600 s") in P094, P096, P097 mid-task; resumed by coordinator messages (step 176).
- Agents reporting a cited paper as "not existing" because it finished later: P067/P068 (P055), P084 (P075), P088 (P081, P072), P093 (P073). No result affected; the synthesis notes the substitutions.
- astropy `Time.ut1` offline configuration issue (P085); `timeout`/`pkill`/`sysctl` unavailable in the sandbox (P098, P072); `sleep` blocked by the harness.
- P072 refuted the assignment's expectation that heavy masses narrow the exothermic line (they broaden it, Δ ∝ μ^{3/2}); P074 refuted the naive "q⁻² collapse" power counting for q²-rate operators; P084 did not reproduce the recalled "quintuplet resonance at 9–10 TeV"; P087 found P050's 2.5 keV erf turn-on and P019's F² ratio need correction; P098 found LZ's ±23 keV (stat) is not reproduced by quanta statistics (11.4 keV).

#### Batch-9 completion addendum (P086, P089) and batch-10a (P092–P098)
| Tool / package | Version | Functions / use | Papers |
|---|---|---|---|
| WimPyDD | 2.0.4 | q-dependent L1–L20 closures and 56 cached spectra (P089); per-isotope O1 kernels and the Sun-frame day halo (P092); per-stream kernels with δ < 0 and light targets H/C/O (P096, P094); composite SHM + disk kernels reusing P058/P068 caches (P097); Ca/W response files patched in memory (P094) | P089, P092, P094, P096, P097 |
| nestpy | 2.1.1 | GetQuanta at fixed energy for the variance budget (P098); ER yields for the χ₂e line placement (P094); NR+ER quanta addition (P095) | P094, P095, P098 |
| astropy / wimprates | 8.0.1 / 0.5.0 | eccentric-orbit Earth velocity check (P092) | P092 |
| scipy | 1.18.1 | quad/solve_ivp Radau/kve/brentq (P086 thermal history), cluster.hierarchy dendrograms (P089), stats.ncx2 non-central χ tail fractions (P097), erf/ndtr band integrals (P095, P098) | P086–P098 |
| numpy default_rng | 2.5.3 | 40 000-draw log-uniform Monte Carlo over corpus ranges (P093) | P093 |
| none: derived by hand | — | Dirac-bilinear NR reductions and the O16 identity (P089); dark-Higgs decay widths (P086); Migdal dipole sum rules and Stobbe check (P095); exothermic window width (P097); free-proton spin response (P096); Fisher-optimal energy estimator (P098) | P086, P089, P095, P096, P097, P098 |
| Claude Code Agent | 2.1.272 | stream-watchdog stalls in P094, P096, P097 (resumed by message); `timeout`/`tee` pitfalls (P097, P098) | P094, P096, P097, P098 |

Local inputs newly used: LZ Theory paragraph and L_i definitions (P089); ROI/Fig. 4/Table I ER counts (P094); tex event/veto/waveform/MSSI/neutron paragraphs (P093); P011 ε/m_A′ grids and P025 Sommerfeld grid (P086); P057 cache and P016 Z(N_lo) curve (P089); P055/P067/P034 scripts (P092); P070/P052/P004/P022 results (P093); P058 f₂ limits and kernels (P094, P096, P097); P068 kernels (P097); P024 band results and P036 script (P095); P009/P024/P043 work files (P098); WimPyDD C/H/O target tables (P096).

| # | Item | Presumed source | Reliability | Papers |
|---|---|---|---|---|
| R90 | g*(T) incl. QCD transition, ν decoupling; Planck 2018 cosmology and ΔN_eff < 0.3; A′ widths and R(s); inverse-decay thermalisation and freeze-in yields; KKM BBN envelope; FIRAS μ/y; Slatyer–Wu decaying-DM bound; Fradette et al. dark-photon cosmology; Bullet-cluster σ/m; Tulin–Yu–Zurek SIDM formulae; Higgs-portal limits | Kolb & Turner; Planck 2018; Kawasaki–Kohri–Moroi 2005; Fixsen 1996; Chluba; Slatyer & Wu 2017; Fradette et al. 2014; Tulin–Yu–Zurek 2013 | certain / likely / uncertain (×10 envelope) | P086 |
| R91 | Anand Table 1 structure and Dirac-bilinear reductions; Anand response coefficient functions; m_M = m_N; O16 = q²O12 + O15 (derived); Hellinger/KL; (N−Z)²/A² | Anand, Fitzpatrick, Haxton 2014; Fitzpatrick et al. 2013 | certain / likely | P089 |
| R92 | SURF coordinates and depth; ω_E; WGS84 radius; GMST; Earth/Sun escape speeds; 2023 perihelion/aphelion and e = 0.0167; Earth and crust composition; LZ elastic SI limit ~1e-45 at 1 TeV | IERS; USNO; almanac; McDonough & Sun 1995; LZ 2024 | certain / likely | P092 |
| R93 | Colouring theorem for superposed Poisson processes | Kingman 1993 | certain | P093 |
| R94 | Electron QED trace; ¹³⁶Xe 2νββ half-life 2.165e21 yr, Q = 2457.8 keV, abundance 8.86 %; Primakoff–Rosen spectrum; XENON1T ER resolution formula; ²¹⁴Pb β shape; Earth core/Pb abundance; ISM heavy-nucleus density; cosmic-ray flux; exposures and analysis ranges of nine experiments; XENON1T/nT/PandaX 2νββ analyses exist | EXO-200 2014; nuclear tables; Primakoff & Rosen 1959; XENON1T EPJC 80, 785; geophysics; collaboration papers | certain / likely / uncertain | P094 |
| R95 | Xe subshell binding energies; Slater's rules; H 1s photoionisation anchors; Stobbe formula; Migdal sudden-approximation formalism (Ibe et al. 2018; tabulated normalisation NOT used); Essig et al. photoabsorption relation; Kouvaris–Pradler bremsstrahlung form; Jackson dipole radiation; Firsov screening; electron CSDA range; Xe photoabsorption coefficients; Kα 29.7 keV, fluorescence yield 0.89; drift speed 1.5 mm/μs | X-ray Data Booklet; Slater 1930; Bethe & Salpeter; Ibe et al. 2018; Essig et al. 2020; Kouvaris & Pradler 2017; Jackson; NIST | certain / likely / uncertain | P095 |
| R96 | Borexino/KamLAND/JUNO/SNO+ masses, scintillator compositions and thresholds; ¹⁴C/¹²C 2.7e-18 (PC) to ~1e-17 (LAB); ¹⁴C decay data; Birks kB = 0.0098 cm/MeV; PSTAR proton stopping; carbon-ion stopping; Cecil light output; ¹³C spin structure; SD scaling; Cherenkov threshold | experiment papers; Borexino ¹⁴C paper; von Krosigk 2013; NIST PSTAR; Cecil–Anderson–Madey 1979; Engel–Pittel–Vogel 1992 | certain / likely / uncertain | P096 |
| R97 | Dark-disk lag/dispersion/density (Read et al. 2008/09; Purcell 2009; Bruch 2009), Gaia thin-disk surface-density limits (Schutz et al. 2018), non-central χ distribution, exothermic window width (derived), ρ₀ as a dynamical density | Read, Lake, Agertz, Debattista 2008; Purcell, Bullock, Kaplinghat 2009; Bruch et al. 2009; Schutz, Lin, Safdi, Wu 2018 | likely / uncertain / certain | P097 |
| R98 | LZ extraction efficiency 0.80, electron lifetime ~6 ms, XENON1T resolution formula, ~4 % line resolution, W = 13.7/13.44 eV, Fisher-optimal estimator | LZ SR1/SR3 papers; XENON1T 2020; LUX 2017; NEST | uncertain / likely / certain | P098 |

Deviations (batch 10a): watchdog stalls (P094, P096, P097) resumed; probe scripts in `$TMPDIR` (P094); P092/P094/P096 reported P085/P072/P075 "not existing" (timing); P096 could not use `WD.C13` directly (¹³C has no spin response table in WimPyDD, analytic estimate used); P094 found `lz.dRdE_SI` cannot take δ < 0 (used WimPyDD).

#### P090 addendum (batch 9)
- Tools: iminuit 2.32.0 via P052's likelihood engine (exec'd header with redirected log), scipy, pandas, matplotlib; script runtime 2 s; `timeout` unavailable (first attempt failed).
- Local inputs: LZ Tables I/S1/S2 and MSSI supplement; P052 script/results; P070 position profiles; P079 annulus ratios; P004/P033/P073 MSSI numbers; P071 N_eff table.
- R99 | √q₀ asymptotics; one-sided 95 % at Δ = 2.71; Šidák form; log-normal parametrisation of factor uncertainties | Cowan et al. 2011; standard statistics | certain | P090
