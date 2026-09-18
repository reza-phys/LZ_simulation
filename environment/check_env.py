"""Check the LZ simulation's research environment.

Run from the simulation root:
    .venv/bin/python environment/check_env.py

It imports every provided package, prints its version, runs a few small physics
calculations, and reports whether LaTeX is available. It writes nothing except caches.
"""
import importlib
import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PACKAGES = [
    # (import name, distribution name, purpose)
    ("numpy", "numpy", "arrays, linear algebra"),
    ("scipy", "scipy", "integration, special functions, optimisation, statistics"),
    ("sympy", "sympy", "symbolic algebra (kinematics, operator matching)"),
    ("mpmath", "mpmath", "arbitrary-precision numerics"),
    ("pandas", "pandas", "tables, CSV/JSON ledgers"),
    ("matplotlib", "matplotlib", "plots (Agg backend, files only)"),
    ("seaborn", "seaborn", "statistical plots"),
    ("tabulate", "tabulate", "Markdown/LaTeX tables"),
    ("uncertainties", "uncertainties", "error propagation"),
    ("iminuit", "iminuit", "likelihood minimisation, profile likelihoods"),
    ("emcee", "emcee", "MCMC sampling"),
    ("corner", "corner", "posterior corner plots"),
    ("dynesty", "dynesty", "nested sampling, Bayesian evidence"),
    ("lmfit", "lmfit", "curve fitting"),
    ("statsmodels", "statsmodels", "statistical tests"),
    ("sklearn", "scikit-learn", "machine learning utilities"),
    ("numba", "numba", "fast toy Monte Carlo loops"),
    ("joblib", "joblib", "parallel loops"),
    ("tqdm", "tqdm", "progress bars"),
    ("astropy", "astropy", "units, constants, time/coordinates (Earth velocity, modulation)"),
    ("particle", "particle", "offline PDG particle data"),
    ("hepunits", "hepunits", "HEP unit conventions"),
    ("periodictable", "periodictable", "isotope masses and abundances"),
    ("radioactivedecay", "radioactivedecay", "offline ICRP-107 decay data (half-lives, chains)"),
    ("pyhf", "pyhf", "HistFactory likelihoods (e.g. HEPData-published)"),
    ("uproot", "uproot", "read ROOT files without ROOT"),
    ("awkward", "awkward", "jagged arrays for event data"),
    ("h5py", "h5py", "HDF5 files"),
    ("yaml", "PyYAML", "HEPData YAML tables"),
    ("openpyxl", "openpyxl", "Excel files"),
    ("pymupdf", "PyMuPDF", "PDF text/vector extraction (digitising figures)"),
    ("pdfplumber", "pdfplumber", "PDF tables and text"),
    ("PIL", "pillow", "images"),
    ("cv2", "opencv-python-headless", "image analysis (digitising plots)"),
    ("camb", "camb", "CMB/matter power spectra"),
    ("healpy", "healpy", "HEALPix sky maps"),
    ("wimprates", "wimprates", "standard WIMP recoil rates in xenon"),
    ("directdm", "directdm", "relativistic-to-NR EFT matching and running"),
    ("nestpy", "nestpy", "NEST liquid-xenon response (incl. LZ_WS2024 detector)"),
]

LATEX_FILES = ["revtex4-2.cls", "siunitx.sty", "mhchem.sty", "cleveref.sty", "physics.sty",
               "tikz-feynman.sty", "feynmp.sty", "booktabs.sty", "natbib.sty", "amsmath.sty", "hyperref.sty"]


def main():
    os.chdir(ROOT)
    print(f"python {sys.version.split()[0]} | {platform.platform()} | root={ROOT}")
    print(f"interpreter: {sys.executable}")
    failures = []
    for mod, dist, purpose in PACKAGES:
        try:
            importlib.import_module(mod)
            try:
                ver = md.version(dist)
            except md.PackageNotFoundError:
                ver = "?"
            print(f"OK    {mod:17s} {ver:12s} {purpose}")
        except Exception as exc:  # noqa: BLE001
            failures.append(mod)
            print(f"FAIL  {mod:17s} {type(exc).__name__}: {exc}")

    print("\n-- sample calculations --")

    def check(label, fn):
        try:
            print(f"OK    {label}: {fn()}")
        except Exception as exc:  # noqa: BLE001
            failures.append(label)
            print(f"FAIL  {label}: {type(exc).__name__}: {exc}")

    def nest_yield():
        import nestpy
        calc = nestpy.NESTcalc(nestpy.detectors.LZ_WS2024())
        y = calc.GetYields(nestpy.interactions.NR, 248.0, 2.888, 97.0)
        return (f"NEST default NR yields at 248 keV, 97 V/cm: "
                f"{y.PhotonYield:.0f} photons, {y.ElectronYield:.0f} electrons")

    def wimp_rate():
        import wimprates as wr
        r = wr.rate_wimp_std(250.0, 1000.0, 1e-45)
        return f"SI rate, m=1000 GeV, sigma=1e-45 cm2, E=250 keV: {r:.3e} (wimprates default units)"

    def decay():
        import radioactivedecay as rd
        return f"Pb-214 half-life {rd.Nuclide('Pb-214').half_life('m')} min"

    def astropy_offline():
        import astropy.utils.data as d
        import astropy.utils.iers as i
        if d.conf.allow_internet or i.conf.auto_download:
            raise RuntimeError("astropy network access is still enabled")
        return f"allow_internet={d.conf.allow_internet}, iers auto_download={i.conf.auto_download}"

    def wimpydd():
        if not os.path.isdir(os.path.join(ROOT, "WimPyDD")):
            raise RuntimeError("WimPyDD/ not found in the simulation root (run environment/setup.sh)")
        import WimPyDD as WD
        return f"WimPyDD imported; xenon isotopes {list(WD.Xe.isotopes)}"

    check("nestpy", nest_yield)
    check("wimprates", wimp_rate)
    check("radioactivedecay", decay)
    check("astropy offline config", astropy_offline)
    check("WimPyDD", wimpydd)

    print("\n-- LaTeX (optional) --")
    if shutil.which("pdflatex"):
        ver = subprocess.run(["pdflatex", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
        print(f"OK    pdflatex: {ver}")
        kpse = shutil.which("kpsewhich")
        for f in LATEX_FILES:
            found = kpse and subprocess.run([kpse, f], capture_output=True, text=True).stdout.strip()
            print(f"{'OK   ' if found else 'MISS '} {f}")
    else:
        print("NOT AVAILABLE  pdflatex (LaTeX typesetting unavailable; papers are Markdown, so this is optional)")

    print("\nAll checks passed." if not failures else f"\nFailures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
