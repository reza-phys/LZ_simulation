# The LZ High-Energy Nuclear Recoil Event: A Simulated Community Response

## 0. Your role and the setting

It is **3 September 2026**. Yesterday (2 September 2026) the LUX-ZEPLIN (LZ) collaboration posted
*"Search for dark matter particle interactions in an extended nuclear recoil energy window with the
LUX-ZEPLIN (LZ) experiment"* (arXiv:2609.02823 [hep-ex]). It reports a single nuclear-recoil-like event at
~250 keV in a region with very low expected background. The global significance is modest, but the event
immediately drew the attention of the whole dark matter community.

You are going to play the role of **that entire research community (experimentalists,
phenomenologists, model builders, statisticians, nuclear theorists, astrophysicists and cosmologists)
over the first two weeks after the announcement (3–16 September 2026).** In that time you will study
the evidence carefully and then write **100 one-page research papers**. Each paper investigates one aspect of the
story, has a clearly recorded headline result, and is backed by detailed working files that contain the
full derivations, calculations and provenance.

This exercise will later be compared with the real literature that followed the announcement. For that
comparison to mean anything, you must produce **your own independent scientific work**:

- **Do not use the internet** in any form. No web search, web fetch, arXiv, INSPIRE, HEPData, Google
  Scholar, news, or social media. If you have tools, you may use them only for local computation
  (e.g. running Python to compute recoil spectra, rates, likelihoods, or significances).
- Your inputs are this prompt, the LZ paper, and your own prior physics knowledge. The paper is provided
  locally, and nothing else is needed:
  - `inputs/arXiv_2609.02823_source/`: the complete source as submitted to arXiv. It contains `main.tex` and
    the section `.tex` files, all tables, the supplement, `references.bib`/`main.bbl`, and every figure as its
    original full-resolution vector PDF.
  - `inputs/LZ_arXiv_2609.02823_fulltext.tex`: the same paper flattened into a single file with a cleaned
    bibliography, for easy reading.
  - `inputs/figures_png/`: PNG renders of the figures (200 dpi), for quick viewing.
- Do not claim knowledge of any result released after the LZ paper. If you use other experiments' results
  (XENONnT, PandaX-4T, DEAP-3600, PICO, CRESST, LHC searches, Fermi/H.E.S.S./CTA, Planck, etc.), use only
  what you actually know. State the numbers you are assuming and flag any uncertainty in them. **Never
  invent experimental data.**
- No external datasets are provided at the start, and this includes the LZ HEPData release. Work from the
  numbers, tables and figures in the paper. If a paper genuinely needs a public dataset, you may **formally
  request it** (see §4). The human overseeing this simulation decides whether to obtain and provide it. You
  must never try to fetch it yourself.
- **Full transparency about tools and information sources is mandatory** (see §3). Treat it as part of the
  scientific output, not as bookkeeping.

Scientific correctness comes first. Realism in what a community would actually write comes second. Be
specific: a paper whose headline is "further study is needed" is useless for the comparison.

---

## 1. Short summary of the LZ result (for orientation; the full paper text is authoritative)

**Dataset and detector.** The analysis uses 220 live days of LZ data (27 March 2023 – 1 April 2024), the same
run whose detector conditions and low-energy (S1c = 3–80 phd) WIMP search were described in LZ's 2024 result.
A fiducial mass of 4.71 ± 0.08 t (14.5% smaller than before, to suppress wall and MSSI backgrounds) gives an exposure of **2.84 t·yr**. The WIMP-search region of interest is extended from
S1c = 3–80 phd to **S1c = 3–600 phd**, with 10^2.75 < S2c < 10^4.15 phd. This corresponds to nuclear recoils of
**~5.4–270 keV** (50%-efficiency points), with an average NR efficiency of 96% between 14 and 250 keV.
Events are split into *science*, *prompt-veto* and *delayed-veto* samples and fit simultaneously.

**Detector response.** NEST v2.4.5 was retuned using tritium, ¹⁴C, ²¹²Pb, D-D neutron and AmBe calibrations.
Including AmBe (NR up to ~330 keV) required a new break in the NR charge-yield power law above
E₀ = 74.7 keV. The ER recombination-fluctuation model was also changed, to a double skew-Gaussian.

**Blinding.** Salting failed to cover the high-energy signal region, because the salt model was defined before
the AmBe calibration. The collaboration therefore states that this is a **non-blind analysis**. Analysis
selections were kept unchanged from the 2024 search.

**The event of interest.**
- S1c = 540.1 phd, S2c = 9268 phd.
- Interpreted as an elastic NR, the recoil energy is **E_R = 248 ± 23 (stat) ± 23 (sys) keV**.
- Recorded 16 June 2023, 21:22:39 UTC.
- Position in the S1–S2 plane: 1.5σ below the NR-band median and 6.7σ below the ER-band median.
- Location: 26.4 cm above the cathode, ~27 cm from the true TPC wall, well inside the fiducial volume.
- S2 pulse shape is consistent with a single-site point-like interaction, and the top/bottom S1 asymmetry is
  consistent with the reconstructed depth.
- S1 pulse-shape discrimination is inconclusive.
- The S1 hit pattern disfavours a reverse-field-region (RFR) MSSI but cannot exclude a wall MSSI.
- Interpreted as an ER, the energy is ~64 keVee, close to the ¹²⁴Xe/¹²⁵I double-vacancy lines at 64.3 and 67.3 keV.
- Nearby activity:
  - A ⁵⁷Co source was removed 25 min earlier, on the opposite side of the detector.
  - An AmBe calibration was performed on 8 June 2023.
  - A muon crossed the Outer Detector 41 min before, and the previous TPC muon was 127 min before.
- The radon tag could not be applied, because the detector was in the "mixed flow" state at the time.

**Backgrounds.**
- The science sample has 1710 events observed versus 1713 ± 39 fitted, dominated by ER backgrounds.
- In the highest-S1c panel of the NR-band projection (Fig. 5), the total integrated background is
  **0.0106 ± 0.0008 events**.
- Relevant components across the whole ROI:
  - atmospheric-ν CEνNS: 0.11 ± 0.02
  - ⁸B + hep ν: 0.057 ± 0.006
  - MSSI: (4.9 ± 4.9) × 10⁻³, with a 100% uncertainty for the charge-dead geometry
  - accidentals: 2.7 ± 0.6, concentrated at low energy
  - detector neutrons: fitted in [0, 0.118]
  - muon-induced neutrons: < 4.6 × 10⁻⁴ (90% CL)
- Neutrons need ≳ 8 MeV to give a 250 keV Xe recoil, and would be accompanied by lower-energy scatters.
- MSSI validation sidebands give p = 0.7, and goodness-of-fit tests are all at p > 0.05.

**Statistics.**
- The likelihood is unbinned and extended, with a two-sided profile likelihood ratio.
- **616 signal models** were tested (293 distinguishable spectra):
  - NREFT Lagrangians 𝓛₁–𝓛₂₀, isoscalar and isovector, at 13 masses from 10 to 4000 GeV
  - inelastic 𝒪₁ and 𝒪₄, isoscalar and isovector, at 400/1000/4000 GeV with mass splittings δ = 0–350 keV
- **Maximum local significance: 3.4σ.** It occurs for magnetic-moment-like 𝓛₁₀ at m ≳ 400 GeV, for 𝓛₁₆, and
  for inelastic 𝒪₁ᵛ and 𝒪₄ at δ ≈ 300–350 keV.
- **Global significance: 2.6σ** after the look-elsewhere effect, computed with toy Monte Carlo.
- Significance is ≈ 0 for all masses ≤ 50 GeV, for the SI-like 𝓛₁ˢ and 𝓛₅ˢ, and for elastic 𝒪₁ˢ (δ = 0).
- The best-fit 𝓛₁₀ˢ (1000 GeV) signal is 1.0 (+1.4, −0.7) events.
- Two-sided 90% CL intervals lift off zero for several models. The upper limits are world-leading for all
  models tested.

**Status.** LZ has continued taking data under the same field conditions since 1 April 2024. That data has not
yet been analysed for this signal region.

---

## 2. What you must do

Work through the four phases in order. Save each deliverable to disk as you go (layout in §5), with provenance recorded as in §3. **Complete
and save Phase 1 and Phase 2 before writing any paper, and do not revise them afterwards.** They are your
frozen forecast.

### Phase 1: Evidence dossier (`00_evidence_dossier.md`, ~2,000–3,000 words)

Read the whole LZ paper, including the supplement, tables and figures, and appraise it critically and independently:
- Every quantitative handle the paper provides, and what each one does and does not establish.
- The strongest points in favour of a dark matter interpretation, and the strongest points against it.
- The full list of possible explanations: statistical fluctuation, known backgrounds, mismodelled
  backgrounds, detector or reconstruction artifacts, calibration-related effects, new non-DM physics, and DM
  of various kinds.
- For each explanation, your initial probability estimate with reasoning (the probabilities should sum to 1).
- Open questions that the paper leaves unanswered, and the concrete calculations or measurements that would answer them.

### Phase 2: Research landscape and plan (`01_landscape_and_plan.md`)

1. **Build your own taxonomy** of the research directions the community would pursue in the first two
   weeks. You might consider statistical reinterpretation, background and detector explanations, particle-physics
   model building, nuclear-physics inputs, astrophysical and halo uncertainties, consistency with other
   experiments and targets, and complementary probes (indirect detection, colliders, cosmology,
   astrophysical objects). You might also consider non-DM exotic explanations, and prospects for decisive
   tests. These are only prompts. The taxonomy, its granularity, and anything missing from this list are
   your call.
2. **Forecast the real-world response.** Estimate:
   - how many papers citing the LZ result would appear on arXiv by 16 September 2026
   - their distribution across your categories
   - which topics would appear first
   - which ideas would attract the most papers
   - where the community consensus would land after two weeks
3. **Plan 100 papers.** For each, give an ID (P001–P100), a working title, a one-line question, a category, and a
   simulated posting date between 3 and 16 September 2026. Allocate papers roughly in proportion to your
   forecast of community attention, adjusted upward for directions you think are scientifically important.
   Competing or overlapping papers on the same popular idea are realistic and allowed; mark them. Order the
   papers chronologically so that later papers can build on, rebut, or refine earlier ones.

### Phase 3: Write the 100 papers (`papers/P001.md` … `papers/P100.md`)

Each paper has two layers:

1. **The detailed work: `work/P0XX/details.md`, with no length limit.** Write this first, as the work
   proceeds. It is the complete research record:
   - motivation and framework
   - every derivation and equation
   - all inputs, with where each came from
   - every intermediate and final number
   - validation checks and robustness variations
   - figures, saved in `work/P0XX/figures/` with captions
   - result tables, saved as CSV or JSON in `work/P0XX/`
   - approaches that failed
   - extended discussion
   - the full reference list

   Scripts go in `code/` (named `P0XX_*.py`, with shared utilities in `code/common/`).
2. **The paper: `papers/P0XX.md`, strictly one page.** It is a distillation of the detailed work, in the
   style of a concise arXiv letter. The body (Abstract through Conclusion) must be **at most ~550 words**, and the
   whole file, including header and footer, must fit on one printed page. Every number on the page must be
   traceable to `work/P0XX/details.md`, or to a script output referenced there. Do not put anything on the page
   that isn't backed by the detailed work.

Use this structure for the one-page paper:

```
# P0XX: <Title>
- Simulated arXiv date: 2026-09-DD
- Primary arXiv category: <hep-ph | hep-ex | astro-ph.CO | astro-ph.HE | nucl-th | physics.data-an | ...>
- Author profile: <kind of group writing this, e.g. "BSM phenomenology group", "former xenon-TPC
  experimentalists", "lattice/nuclear-structure theorists". No real names.>
- Category (from your taxonomy):
- Builds on / responds to: <LZ paper; P-numbers of earlier corpus papers, if any>

**Abstract.** (≤ 80 words)
**Question and approach.** (≤ 120 words: what is asked, the framework and method)
**Results.** (≤ 220 words: the key equation(s) and numbers; at most one small table or one figure reference)
**Caveats.** (≤ 80 words)
**Conclusion.** (≤ 50 words)
**References.** (≤ 6 essential works that you are confident exist and that predate Sept 2026, plus the LZ paper
               and corpus papers by P-number; the full list goes in details.md)

---
**HEADLINE RESULT:** <one sentence, specific and quantitative where possible>
**KEY NUMBERS:** <bullet list of the main quantitative outputs with units>
**RESULT TYPE:** computed (explicit calculation/code) | estimated (order-of-magnitude) | qualitative
**STANCE ON THE EVENT:** supports DM interpretation | favours background/artifact | statistical
                         reassessment (strengthens/weakens) | neutral (tool, projection, or constraint)
**CONFIDENCE:** high | medium | low, and one sentence explaining why
**DATA DEPENDENCE:** none | provisional pending DR-### (say which results would change)

**TOOLS (summary):** <one line naming every tool, package and version used, e.g. "python 3.12.13; numpy 2.5.3,
  scipy 1.18.1 (integrate.quad), WimPyDD 2.0.4; Read (table_local_significance.tex); 3 values recalled
  from memory". The complete record is in provenance/P0XX.json; see §3.>
**DETAILS:** work/P0XX/details.md · provenance/P0XX.json · code/P0XX_*.py · data requests: <none | DR-###>
```

The header, the result fields and the TOOLS/DETAILS lines are compact summaries and do not count toward
the ~550-word body limit. They must still fit on the same page.

Standards for the papers:
- **Do the physics.** Derive recoil spectra, rates, kinematic conditions (e.g. the allowed δ and mass ranges
  that can produce a 248 keV recoil), and likelihood or significance estimates. Also derive cross-section
  and coupling relations, relic abundances, and collider or indirect-detection rates. Where code
  would help and you can run it, run it, and save the scripts under `code/`. Report the numbers you actually
  obtained, and do not claim more precision than your method supports.
- Use the LZ paper's numbers (backgrounds, efficiencies, NEST parameters, S1c/S2c, g₁ = 0.110, g₂ = 34.5, exposure)
  wherever relevant.
- The one-page paper shows the essentials. Full derivations, all numbers, checks and figures live in
  `work/P0XX/details.md`, which must be complete enough for an expert to reproduce the result without
  asking you anything.
- Honest null or skeptical results are as valuable as positive ones. The corpus should reflect the
  genuine range of scientific opinion, including disagreement between papers.
- Each paper should make a distinct contribution. Avoid near-duplicates, except for deliberately marked
  competing papers.
- Keep the internal consistency of the corpus: later papers should use, or explicitly dispute, results of
  earlier ones.

### Phase 4: Results ledger and synthesis

1. **`results_ledger.csv`** and **`results_ledger.json`**, with one row per paper and these fields:
   `id, version, date, arxiv_category, taxonomy_category, title, question, headline_result, key_numbers,
   model_or_mechanism, stance, result_type, confidence, builds_on, agent_tools, software_and_packages,
   scripts, local_inputs, recalled_knowledge, datasets_used, data_requests, data_dependence`.
   Update the ledger after every batch of papers, so that it is always the source of truth.
2. **`99_synthesis.md`** (~2,000–3,000 words), containing:
   - Where the simulated community stands after two weeks.
   - Your updated probabilities for each explanation from Phase 1, and what changed them.
   - The 10 most important papers in the corpus and why.
   - The most viable DM scenarios, with preferred mass, coupling or cross-section, and δ ranges.
   - The most viable non-DM explanations.
   - The decisive tests: which experiment or analysis, the exposure needed, and the timeline.
   - A falsifiable prediction for LZ's next data release on this signal region.
   - A short self-assessment of which parts of your Phase 2 forecast you now think were wrong.

---

## 2b. Computing environment (provided, offline, fixed)

A research environment has been prepared for you. **Use only what is listed here.** Do not install, download
or build anything else. Do not use other interpreters, compilers or programs, such as `/usr/bin/python3`,
even if they happen to be present on the machine. Standard shell utilities for file handling (`ls`, `cat`,
`head`, `wc`, `grep`, `mkdir`, `cp`, `mv`) are fine, and must be logged like every other tool. If you need
something that is missing, work around it analytically or numerically with the tools provided, and record the
gap in the provenance (§3).

**Working directory.** You are running in the simulation root. Run every command from there. **All
deliverables go under `output/`**: wherever this prompt names `papers/`, `work/`, `code/`, `provenance/`,
`data_requests/` or the ledger and synthesis files, it means that path inside `output/`, e.g.
`output/papers/P001.md`.

**Protected, read-only areas.** `PROMPT.md`, `README.md`, `run.sh`, `inputs/`, `environment/`, `.venv/` and
`sim_logs/` cannot be modified, and attempts to write to them are blocked. If the human provides requested data, it
appears in `inputs/provided_data/DR-###/`.

**Independent tool log.** The harness automatically records every tool call in `sim_logs/tool_calls.jsonl`.
Your own provenance records (§3) must still be complete on their own; they will be cross-checked against this
log. All network access is blocked, and any attempt is logged.

**Python.** Use `.venv/bin/python` (Python 3.12) for every calculation, e.g.
`.venv/bin/python output/code/P012_recoil_spectra.py`. Exact versions of everything are in
`environment/ENVIRONMENT_versions.txt`; cite versions from that file in your provenance. The check run at setup
is in `environment/check_env_output.txt`. Run `.venv/bin/python environment/check_env.py` once yourself before
Phase 1, and log the output.

| Purpose | Packages |
|---|---|
| Numerics, symbolic algebra | numpy, scipy, sympy, mpmath, numba, joblib |
| Tables, data files | pandas, tabulate, PyYAML (HEPData YAML), h5py, uproot + awkward (ROOT files), openpyxl |
| Statistics and inference | iminuit, scipy.stats, statsmodels, emcee, corner, dynesty, lmfit, pyhf, uncertainties, scikit-learn |
| Plots | matplotlib (non-interactive, save to files), seaborn |
| Units, constants, astronomy | astropy (offline: no IERS/remote downloads), hepunits, particle (offline PDG tables) |
| Nuclear and decay data | periodictable (isotope masses, abundances), radioactivedecay (offline ICRP-107 decay data) |
| Direct-detection physics | **WimPyDD 2.0.4** (NREFT/inelastic rates and nuclear response functions, the code LZ used), **wimprates** (standard WIMP rates), **directdm** (relativistic → NR EFT matching), **nestpy 2.1.1** (NEST xenon response, includes `detectors.LZ_WS2024`) |
| Cosmology, sky maps | camb, healpy |
| Reading the paper's figures | PyMuPDF (`fitz`: extract vector paths and text from figure PDFs), pdfplumber, pillow, opencv-python-headless |

Notes:
- **WimPyDD** lives in `WimPyDD/` in the simulation root and must be run with the working directory equal to
  the simulation root (it resolves its data paths relative to the working directory). Treat
  `WimPyDD/` as a tool, not as an output location. Files it generates for new experiment or model definitions
  should be listed in the provenance.
- **nestpy** uses NEST's default parameters unless you pass the LZ-tuned values from the paper's Tables (the
  supplement lists the tuned ER and NR parameters). Say which one you used.
- Plot and cache directories are set automatically (`.cache/` in the simulation root).

**LaTeX.** LaTeX is available only if `environment/ENVIRONMENT_versions.txt` lists `pdflatex`. That file also
says which packages (revtex4-2, siunitx, mhchem, cleveref, physics, tikz-feynman, …) are present. The papers
themselves must be written in Markdown as specified in Phase 3. Where LaTeX is available, you may additionally
typeset a paper or figure with it, and record that in the provenance.

**Not available.** There is no network access, no Fortran compiler, no cmake and no ROOT. There is no
micrOMEGAs, MadGraph, CLASS or DarkSUSY. For relic abundances, collider rates and similar quantities, use
analytic or semi-analytic methods implemented in Python, and state the approximation.

---

## 3. Tool and provenance tracking (mandatory, crucial)

This record is as important as the physics. A reader must be able to reconstruct exactly **what tools
and what information produced every result**, for each paper and for the whole process. Record
everything, down to tools that seem too obvious to mention: the Read tool, shell utilities, a Python
standard-library module, `pdflatex`, a unit conversion recalled from memory. Do not summarise vaguely
("used Python"). Name the tool, package, version, function, file and purpose. If something was done
without tools (reasoning or algebra in text), say so explicitly.

Keep three records, updated as you go rather than reconstructed at the end:

1. **Per paper:** `provenance/P0XX.json` (machine-readable), mirrored as a "Tools and provenance" section at
   the end of `work/P0XX/details.md`, and summarised in the one-line TOOLS field on the paper page. Fields:
   - `agent_tools`: e.g. Read (which files), Bash (number of commands, and which), Write, Edit, Grep, subagents
   - `software`: interpreter and every package, with version and the specific functions or modules used,
     e.g. scipy 1.18.1 (integrate.quad, stats.poisson); pdflatex (TeX Live 2026); shell utilities; or
     "none: derived by hand"
   - `scripts_and_commands`: paths under `code/`, one line each on what they compute, and key commands run
   - `local_inputs`: exact files and the parts used, e.g. "table_local_significance.tex: 𝓛₁₀ row",
     "Fig4 PDF: vector paths extracted with PyMuPDF", outputs of other corpus papers
   - `recalled_knowledge`: every external number, formula or result taken from training knowledge rather
     than a provided file, with presumed source and a reliability flag (certain | likely | uncertain)
   - `datasets`: none, or the human-provided DR-### files used
   - `data_requests`: none, or DR-### with status (pending | provided | declined)
   - `failed_or_abandoned`: unavailable tools, errors, blocked actions, approaches dropped
2. **Whole process, chronological:** `provenance/process_log.md`. Append an entry for every step of work
   in every phase, including the dossier, the plan, each paper, the ledger and the synthesis. Each entry
   gives the timestamp or step number, the phase and paper ID, the agent tool used (Read / Write / Edit /
   Bash / Grep / Glob / subagent / …), and the exact file or command. It also gives the purpose, and the
   outcome, including errors and blocked actions.
3. **Whole process, aggregated:** `provenance/tool_inventory.md`, which summarises the following:
   - **Environment:** OS, shell, Python interpreter, TeX distribution. Take versions from
     `environment/ENVIRONMENT_versions.txt` and the output of `environment/check_env.py` (§2b). Any version
     you check by command (e.g. `.venv/bin/python -c "import numpy; print(numpy.__version__)"`,
     `pdflatex --version`) should be logged with the command used.
   - **Tools and packages:** every agent tool, program, library and package, with version, the specific
     functions or modules used, and the list of papers that used it.
   - **Local inputs:** every input file, and which papers used which parts of it.
   - **Recalled knowledge:** every piece of external knowledge taken from memory (experimental limits,
     exposures, astrophysical parameters, nuclear data, formulas from the literature), with presumed
     source, reliability flag, and the papers that relied on it.
   - **Datasets:** every data request and dataset, with its status.
   - **Unavailable tools:** every tool or package you wanted but could not use, and what you did instead.

Rules:
- Do not install packages or download anything. The environment is offline. Use what is already installed,
  and record any failed import or install attempt.
- Subagents, if used, must report their tool use back to you so it can be logged under the right paper.
- Never omit or smooth over a step to make the record look cleaner. Gaps in the record are treated as failures.

## 4. Data requests

Any paper may request a public dataset it needs: for example, another experiment's released data,
telescope or satellite data, a nuclear-data table, a published likelihood, or LZ's own HEPData release. You
**cannot obtain data yourself**. Instead, file a request. The human overseeing the simulation will decide,
case by case, whether to download it and provide it.

**Eligibility.** A request may only be for data that was **publicly available on or before 2 September 2026**,
together with the LZ paper's own data release. You may not request papers, analyses, news or commentary
about the LZ event published after its announcement.

**How to file.** Write `data_requests/DR-###.md` (numbered sequentially) and add a row to
`data_requests/index.csv`. Each request must be complete enough that someone unfamiliar with the project could
find and retrieve exactly the right data:

```
# DR-###: <short dataset name>
- Requested by: <paper ID(s)>            - Filed at step: <process_log step>
- Priority: essential (headline result depends on it) | important (materially improves result) | nice-to-have
- Dataset: <full name, experiment/instrument/collaboration, data product, release/version, date range>
- Exact content needed: <variables/columns, energy or mass ranges, units, event-level vs binned,
  which tables/figures/files, any selection>
- Where to find it: <archive/repository and best-known location (e.g. HEPData record, collaboration
  data-release page, NASA HEASARC/Fermi SSC, IAEA nuclear data, Zenodo, GitHub), identifiers such as arXiv ID,
  DOI or INSPIRE record of the associated paper. Mark any location recalled from memory as
  (unverified).>
- Expected format and size: <e.g. YAML/CSV/ROOT/FITS, approx. MB>
- Access or licence constraints, if known:
- Why it is needed: <the scientific question and why the paper's current inputs are insufficient>
- How it will be used: <the exact analysis step, and the script that will consume it>
- What could change: <which numbers or conclusions would be updated, and how much they might move>
- Fallback in the meantime: <what the paper does without it>
```

**While a request is pending, keep working.** Finish the paper with the best available fallback, such as
values read off the LZ paper's tables and figures, published summary numbers recalled from memory (flagged
as such in the provenance), or conservative approximations. Label each affected
result `provisional pending DR-###` in the paper, the ledger (`data_dependence`) and the provenance
records. A request never pauses the overall run. If the human later places data in
`inputs/provided_data/DR-###/` and asks you to continue, revise the affected papers as `P0XX_v2.md` (with `work/P0XX/details_v2.md`). Keep v1
unchanged, add a v2 ledger row, and log how the data changed the results.

## 5. Output layout

Everything you produce goes under `output/` in the simulation root. Nothing may be written anywhere else,
except `WimPyDD/` (tool-generated files) and `.cache/`.

```
output/
  00_evidence_dossier.md
  01_landscape_and_plan.md
  papers/P001.md … P100.md      (one page each; P0XX_v2.md only if revised with provided data)
  work/P001/ … P100/            (details.md: full research record; figures/; result tables)
  code/                         (P0XX_*.py scripts and common/ utilities, referenced from provenance)
  provenance/
    process_log.md              (chronological log of every step and tool call, whole process)
    tool_inventory.md           (aggregated tools / packages / versions / inputs / recalled knowledge)
    P001.json … P100.json       (per-paper provenance, machine-readable)
  data_requests/
    index.csv                   (id, requested_by, priority, dataset, where_to_find, status)
    DR-001.md …
  results_ledger.csv
  results_ledger.json
  99_synthesis.md
```

## 6. Execution notes

- Work autonomously and do not stop to ask questions. Where a choice is ambiguous, make a reasonable decision
  and document it.
- Write the papers in batches of about 10, in chronological order, and update the ledger after each batch.
- For each paper, do the work in order:
  1. calculations (scripts in `code/`)
  2. `work/P0XX/details.md`, written as you go
  3. `provenance/P0XX.json`
  4. the one-page `papers/P0XX.md`, distilled from the details
  5. the ledger row

  Check the page's body word count (≤ ~550) before moving on.
- If your context grows long, rely on the saved plan and ledger to keep continuity. Do not re-plan.
- Before Phase 1, record the environment in `provenance/tool_inventory.md` (versions of the software you can find).
- After each batch, update the paper files, `provenance/` (the log, the per-paper JSON and the inventory),
  `data_requests/`, and the ledger. Only then move on.
- Do not stop until all 100 papers, the ledger, the provenance records, the data-request index and the
  synthesis are complete. The synthesis must also include a **process-level tool summary**, covering which
  tools and packages were used, how often, and for which kinds of papers. It must also include a
  **data-request summary**: all requests ranked by priority, with the papers and results each would affect.
