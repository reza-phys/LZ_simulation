# LZ simulation: offline defaults for the research environment.
# setup.sh copies this file into .venv/lib/python3.12/site-packages/, so it runs at every
# start of .venv/bin/python. It keeps caches inside the simulation folder (the only place
# the Claude Code sandbox lets commands write) and turns off library-level network access.
import os

_site = os.path.dirname(os.path.abspath(__file__))
# site-packages -> python3.12 -> lib -> .venv -> simulation root
_root = os.path.abspath(os.path.join(_site, "..", "..", "..", ".."))
_cache = os.path.join(_root, ".cache")

os.environ.setdefault("XDG_CONFIG_HOME", os.path.join(_root, "environment", "xdg"))  # astropy.cfg lives here
os.environ.setdefault("XDG_CACHE_HOME", _cache)
os.environ.setdefault("ASTROPY_CACHE_DIR", os.path.join(_cache, "astropy"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_cache, "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(_cache, "numba"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PIP_NO_INDEX", "1")
os.environ.setdefault("UV_OFFLINE", "1")
