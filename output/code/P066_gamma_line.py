"""
P066 - A 300-380 keV line from the sky: Galactic-halo de-excitation line chi2 -> chi1 gamma
        from a long-lived excited dark-matter state, INTEGRAL/SPI, COMPTEL, COSI, and the CXB.

Run from the simulation root:  .venv/bin/python output/code/P066_gamma_line.py

Outputs (output/work/P066/):
  D_factors.csv                D per ROI (cone / box / all-sky / anti-centre) for NFW, Einasto, Burkert
  D_per_sr_directions.csv      D per sr along selected directions (morphology)
  validation.json              healpy-map D vs scipy cone integral; P026 comparison
  line_flux_table.csv          Phi(ROI) at f2/tau = 1e-22 s^-1 and the f2/tau bound per ROI / instrument
  sensitivities.csv            recalled sensitivities with reliability flags
  cxb_bound.csv                decaying-DM continuum vs CXB (Gruber 1999 recalled) -> f2/tau bounds
  corpus_models.csv            Higgsino / dark photon / MiDM / generic long-lived chi2 mapping
  doppler.json                 line width and dipole
  P066_results.json            everything headline-worthy
  figures/P066_fig1_f2_tau_plane.png, P066_fig2_flux_vs_roi.png, P066_fig3_cxb.png, P066_fig4_skymap.png
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import healpy as hp

sys.path.insert(0, "output/code")
from common import lzcommon as lz

OUT = "output/work/P066"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
t0 = time.time()
LOG = []
def log(s=""):
    print(s); LOG.append(str(s))

# ----------------------------------------------------------------------------------------------
# 0. Inputs (library + recalled, all flagged)
# ----------------------------------------------------------------------------------------------
RHO_SUN = lz.RHO0_GEV_CM3            # 0.3 GeV/cm^3 (lzcommon; Baxter et al. 2021 SHM)
V0 = lz.V0_KMS                       # 238 km/s
C_KMS = lz.C_KMS
KPC_CM = 3.0856775814913673e21       # recalled, certain
R_SUN = 8.2                          # kpc, recalled (likely; GRAVITY 8.18)
R_S = 20.0                           # kpc, NFW/Einasto scale radius (recalled standard, likely)
ALPHA_EIN = 0.17                     # Einasto shape (recalled, likely)
R_C_BUR = 9.0                        # kpc Burkert core (recalled, uncertain; Nesti & Salucci 2013 ~9.3)
R_MAX = 250.0                        # kpc halo truncation (assumption; NFW D changes <1% for 150-400)
M_CHI = 1000.0                       # GeV reference mass
T_U = 13.8e9 * 3.15576e7             # s, recalled certain (4.35e17 s)
H0_S = 67.4 * 1e5 / (3.0856775814913673e24)   # s^-1 (Planck 2018, recalled certain)
OM_M, OM_L = 0.315, 0.685
OMEGA_DM_H2 = 0.120
RHO_CRIT_H2 = 1.05368e-5             # GeV/cm^3 per h^2 (recalled certain)
RHO_DM_0 = OMEGA_DM_H2 * RHO_CRIT_H2  # 1.264e-6 GeV/cm^3 mean DM density today

RECALLED = [
 dict(item="NFW r_s = 20 kpc; R_sun = 8.2 kpc; rho_sun = 0.3 GeV/cm^3 (lzcommon)", presumed_source="standard Milky-Way fits; Baxter et al. 2021 via lzcommon", reliability="likely"),
 dict(item="Einasto alpha = 0.17 with r_s = 20 kpc; Burkert core r_c = 9 kpc", presumed_source="Aquarius/Via Lactea fits; Nesti & Salucci 2013", reliability="likely (Einasto) / uncertain (Burkert core radius)"),
 dict(item="Planck 2018: H0 = 67.4, Omega_m = 0.315, Omega_c h^2 = 0.120; rho_crit = 1.054e-5 h^2 GeV/cm^3; t_U = 13.8 Gyr", presumed_source="Planck 2018 / PDG", reliability="certain"),
 dict(item="INTEGRAL/SPI narrow-line 3 sigma sensitivity ~3e-5 ph cm^-2 s^-1 at 300 keV for 1 Ms (point source); energy resolution ~2 keV FWHM at 300 keV; effective area ~ 100 cm^2 at 300 keV", presumed_source="Roques et al. 2003, Vedrenne et al. 2003 (SPI instrument papers)", reliability="uncertain (factor 2-3)"),
 dict(item="SPI full-mission (>= 15 yr, > 100 Ms inner-Galaxy exposure) diffuse narrow-line detection threshold in the inner Galaxy ~(1-3)e-5 ph cm^-2 s^-1, anchored on 60Fe 1173/1332 keV (~3e-5 each, ~5 sigma), 26Al 1809 (3e-4 inner Galaxy), 511 keV bulge (1e-3, >50 sigma)", presumed_source="Wang et al. 2007/2020; Diehl et al. 2006; Siegert et al. 2016; Knoedlseder et al. 2005", reliability="uncertain (factor 3; 60Fe/26Al fluxes likely, threshold transfer to 300 keV uncertain)"),
 dict(item="SPI line searches: Teegarden & Watanabe 2006 (first year, 20 keV-8 MeV, diffuse-line 3 sigma limits ~1e-4); Calore et al. 2023 (16 yr, decaying-DM lines in |l|,|b| < 47.5 deg, flux limits ~1e-5-1e-4 per energy bin)", presumed_source="ApJ 646, 965 (2006); MNRAS 520, 4167 (2023)", reliability="uncertain (existence likely; numbers factor 3)"),
 dict(item="COMPTEL band 0.75-30 MeV; no sensitivity at 300 keV", presumed_source="Schoenfelder et al. 1993", reliability="certain"),
 dict(item="COSI (NASA SMEX, 0.2-5 MeV, launch ~2027): 2-yr all-sky 3 sigma narrow-line sensitivity ~1e-5 ph cm^-2 s^-1 near 511 keV; Ge energy resolution ~1% FWHM; angular resolution ~4 deg", presumed_source="Tomsick et al. 2019/2023 (COSI mission papers)", reliability="uncertain (factor 3)"),
 dict(item="e-ASTROGAM / AMEGO(-X) concepts: narrow-line 3 sigma sensitivity ~ (1-5)e-6 ph cm^-2 s^-1 at 0.3-1 MeV for a few-year survey", presumed_source="De Angelis et al. 2017; McEnery et al. 2019", reliability="uncertain (concept-level projections)"),
 dict(item="CXB/CGB above 60 keV (HEAO-1 A4 + SMM): E dN/dE = 0.0259(E/60)^-5.5 + 0.504(E/60)^-1.58 + 0.0288(E/60)^-1.05 keV cm^-2 s^-1 sr^-1 keV^-1; COMPTEL 1-10 MeV lies a factor 2-3 below the SMM extrapolation", presumed_source="Gruber et al. 1999, ApJ 520, 124; Weidenspointner et al. 2000", reliability="uncertain (factor 2 at 300 keV)"),
 dict(item="Decaying-DM isotropic intensity: dI/dE = c Gamma n0 / (4 pi E H(z)), 1+z = E0/E", presumed_source="standard (e.g. Essig et al. 2013); derived here", reliability="certain"),
 dict(item="Positronium continuum (3-photon) below 511 keV with f_Ps ~ 1 in the bulge; no known astrophysical narrow line between 300 and 380 keV in the diffuse sky (nearest: 7Be 478, e+e- 511; 44Ti 68/78; 57Co 122/136)", presumed_source="Siegert et al. 2016; nuclear line catalogues", reliability="likely"),
 dict(item="No strong SPI instrumental background line recalled in 300-380 keV (lines at 198, 439, 584, 882 keV etc. lie outside)", presumed_source="Weidenspointner et al. 2003 SPI background line catalogue", reliability="uncertain"),
 dict(item="Sun-frame speed 250.6 km/s (lzcommon V0 + peculiar); 1-D halo dispersion v0/sqrt(2); inner-halo dispersion up to ~250 km/s", presumed_source="lzcommon; Jeans arguments", reliability="likely"),
 dict(item="Dark-photon chi2 -> chi1 gamma vanishes for pure kinetic mixing (on-shell photon: mixing insertion ~ eps q^2 -> 0); tau_nunubar = 2.6e18 s (m_A'/GeV)^-4 at 300 keV; Higgsino tau_gamma ~ 0.06 s, tau_nunubar 1.2e6 s; photon-M1 tau = 66 us", presumed_source="corpus P011, P014, P023, P026", reliability="corpus (likely)"),
 dict(item="Exothermic bound: f2(today) <= 1.2e-3 (300 keV), 5.5e-5 (350), 3.8e-7 (380 keV) for any model at the LZ-fit cross-section; tau_tot <= 7.2e16, 4.8e16, 3.1e16 s with f2_fo = 0.42-0.50", presumed_source="corpus P011 sec. 3.7, P026 sec. 5.5", reliability="corpus (computed there)"),
]

# ----------------------------------------------------------------------------------------------
# 1. Density profiles (all normalised to rho(R_sun) = 0.3 GeV/cm^3)
# ----------------------------------------------------------------------------------------------
def rho_nfw(r):
    x = r / R_S
    xs = R_SUN / R_S
    rho_s = RHO_SUN * xs * (1 + xs) ** 2
    return rho_s / (x * (1 + x) ** 2)

def rho_einasto(r):
    f = lambda rr: np.exp(-2.0 / ALPHA_EIN * ((rr / R_S) ** ALPHA_EIN - 1.0))
    return RHO_SUN * f(r) / f(R_SUN)

def rho_burkert(r):
    f = lambda rr: 1.0 / ((1 + rr / R_C_BUR) * (1 + (rr / R_C_BUR) ** 2))
    return RHO_SUN * f(r) / f(R_SUN)

PROFILES = {"NFW": rho_nfw, "Einasto": rho_einasto, "Burkert": rho_burkert}

# ----------------------------------------------------------------------------------------------
# 2. Line-of-sight integral D(psi) = int rho ds  [GeV cm^-2 sr^-1 ... i.e. per sr]
# ----------------------------------------------------------------------------------------------
def s_max(cospsi):
    sinpsi2 = np.clip(1 - cospsi ** 2, 0, 1)
    return R_SUN * cospsi + np.sqrt(R_MAX ** 2 - R_SUN ** 2 * sinpsi2)

def D_los_quad(cospsi, rho):
    """scipy reference: GeV/cm^3 * kpc -> GeV/cm^2."""
    def f(s):
        r = np.sqrt(R_SUN ** 2 + s ** 2 - 2 * R_SUN * s * cospsi)
        return rho(max(r, 1e-8))
    sc = R_SUN * cospsi
    pts = [sc] if 0 < sc < s_max(cospsi) else None
    val = integrate.quad(f, 0, s_max(cospsi), limit=400, points=pts, epsrel=1e-8)[0]
    return val * KPC_CM

N_U = 600
def D_los_vec(cospsi, rho):
    """Vectorised over an array of cos(psi): log-spaced trapezoid on both sides of closest approach."""
    cospsi = np.atleast_1d(cospsi).astype(float)
    sc = R_SUN * cospsi                      # closest approach (may be negative -> behind us)
    smax = s_max(cospsi)
    out = np.zeros_like(cospsi)
    # segment A: from s=0 to s=sc (only if sc>0); variable u = sc - s in [1e-7, sc]
    # segment B: from s=sc (or 0) to smax; variable u = s - s0 in [1e-7, smax - s0]
    for seg in ("A", "B"):
        if seg == "A":
            L = np.where(sc > 0, sc, 0.0)
            s_ref = sc
            sign = -1.0
        else:
            s0 = np.where(sc > 0, sc, 0.0)
            L = smax - s0
            s_ref = s0
            sign = +1.0
        mask = L > 0
        if not mask.any():
            continue
        lu = np.linspace(np.log(1e-7), 0.0, N_U)          # log(u/L)
        u = L[mask, None] * np.exp(lu[None, :])
        s = s_ref[mask, None] + sign * u
        r = np.sqrt(np.maximum(R_SUN ** 2 + s ** 2 - 2 * R_SUN * s * cospsi[mask, None], 1e-16))
        integrand = rho(r) * u                             # du = u d(ln u)
        val = np.trapezoid(integrand, lu, axis=1)
        # add the tiny piece u in [0, 1e-7 L] analytically ~ rho(r_min)*1e-7 L (negligible)
        out[mask] += val
    return out * KPC_CM

# --- cone D-factor by 1-D azimuthal quad (reference) -------------------------------------------
def D_cone_quad(theta_deg, rho):
    f = lambda p: D_los_quad(np.cos(p), rho) * 2 * np.pi * np.sin(p)
    return integrate.quad(f, 0, np.radians(theta_deg), limit=200)[0]

# ----------------------------------------------------------------------------------------------
# 3. HEALPix map of D per sr and ROI integrals
# ----------------------------------------------------------------------------------------------
NSIDE = 128
NPIX = hp.nside2npix(NSIDE)
OMEGA_PIX = hp.nside2pixarea(NSIDE)      # sr
theta_pix, phi_pix = hp.pix2ang(NSIDE, np.arange(NPIX))
l_pix = np.degrees(phi_pix); l_pix = np.where(l_pix > 180, l_pix - 360, l_pix)
b_pix = 90.0 - np.degrees(theta_pix)
cospsi_pix = np.cos(np.radians(l_pix)) * np.cos(np.radians(b_pix))
psi_pix = np.degrees(np.arccos(np.clip(cospsi_pix, -1, 1)))

log("Building D-per-sr HEALPix maps (nside=%d, %d pixels, %.3f deg pixels)" % (NSIDE, NPIX, np.degrees(np.sqrt(OMEGA_PIX))))
maps = {}
for name, rho in PROFILES.items():
    m = np.zeros(NPIX)
    for i0 in range(0, NPIX, 8192):
        m[i0:i0 + 8192] = D_los_vec(cospsi_pix[i0:i0 + 8192], rho)
    maps[name] = m
    log("  %-8s done  t=%.1f s  D/sr at GC pixel %.3e, anticentre %.3e GeV cm^-2 sr^-1" % (name, time.time() - t0, m[psi_pix.argmin()], m[psi_pix.argmax()]))

ROIS = {
    "cone_5":  psi_pix < 5,
    "cone_10": psi_pix < 10,
    "cone_16": psi_pix < 16,
    "cone_30": psi_pix < 30,
    "box_10":  (np.abs(l_pix) < 10) & (np.abs(b_pix) < 10),
    "box_30":  (np.abs(l_pix) < 30) & (np.abs(b_pix) < 30),
    "box_47.5": (np.abs(l_pix) < 47.5) & (np.abs(b_pix) < 47.5),
    "all_sky": np.ones(NPIX, bool),
    "anticentre_cone_30": psi_pix > 150,
    "anticentre_cone_10": psi_pix > 170,
    "high_lat_|b|>30": np.abs(b_pix) > 30,
}
rows = []
for roi, mask in ROIS.items():
    omega = mask.sum() * OMEGA_PIX
    row = dict(ROI=roi, Omega_sr=omega)
    for name in PROFILES:
        row["D_" + name + "_GeVcm-2sr"] = float(maps[name][mask].sum() * OMEGA_PIX)
        row["Dbar_" + name + "_GeVcm-2sr-1"] = float(maps[name][mask].mean())
    rows.append(row)
Dtab = pd.DataFrame(rows)
Dtab.to_csv(os.path.join(OUT, "D_factors.csv"), index=False)
log("\nD-factors (GeV cm^-2 sr):")
log(Dtab[["ROI", "Omega_sr", "D_NFW_GeVcm-2sr", "D_Einasto_GeVcm-2sr", "D_Burkert_GeVcm-2sr"]].to_string(index=False, float_format=lambda x: "%.3e" % x))

# validation against the azimuthal quad and P026's 7.41e21 (10 deg cone)
val = {}
for th in (5, 10, 16, 30):
    q = D_cone_quad(th, rho_nfw)
    hpx = float(Dtab.loc[Dtab.ROI == "cone_%d" % th, "D_NFW_GeVcm-2sr"].iloc[0])
    val["NFW_cone_%d" % th] = dict(quad=q, healpix=hpx, ratio=hpx / q)
val["P026_D10_quoted"] = 7.41e21
val["P026_ratio_quad"] = val["NFW_cone_10"]["quad"] / 7.41e21
# r_max sensitivity
for rm in (150.0, 400.0):
    R_MAX_SAVE = R_MAX
    globals()["R_MAX"] = rm
    val["NFW_cone_10_rmax_%d" % int(rm)] = D_cone_quad(10, rho_nfw)
    globals()["R_MAX"] = R_MAX_SAVE
# vectorised vs quad along single directions
for psi in (0.5, 2, 10, 45, 90, 180):
    c = np.cos(np.radians(psi))
    val["los_psi_%g" % psi] = dict(quad=D_los_quad(c, rho_nfw), vec=float(D_los_vec(c, rho_nfw)[0]))
log("\nValidation: NFW cone D (quad vs healpix) and P026:")
for k, v in val.items():
    log("  %s: %s" % (k, v if not isinstance(v, dict) else {kk: ("%.4e" % vv) for kk, vv in v.items()}))
json.dump(val, open(os.path.join(OUT, "validation.json"), "w"), indent=1, default=float)

# directions (morphology)
dirs = [(0, 0), (0, 5), (0, 10), (10, 0), (30, 0), (0, 30), (60, 0), (90, 0), (0, 90), (180, 0)]
drows = []
for (l, b) in dirs:
    c = np.cos(np.radians(l)) * np.cos(np.radians(b))
    r = dict(l_deg=l, b_deg=b)
    for name, rho in PROFILES.items():
        if l == 0 and b == 0:
            # 1-deg cone average (the exact centre is log-singular for NFW)
            r["Dsr_" + name] = D_cone_quad(1.0, rho) / (2 * np.pi * (1 - np.cos(np.radians(1.0))))
        else:
            r["Dsr_" + name] = D_los_quad(c, rho)
    drows.append(r)
Ddir = pd.DataFrame(drows)
Ddir.to_csv(os.path.join(OUT, "D_per_sr_directions.csv"), index=False)
log("\nD per sr along directions (GeV cm^-2 sr^-1):")
log(Ddir.to_string(index=False, float_format=lambda x: "%.3e" % x))

# ----------------------------------------------------------------------------------------------
# 4. Sensitivities (recalled brackets) and line flux / f2-over-tau bounds
# ----------------------------------------------------------------------------------------------
SENS = [
 # instrument, ROI key, lo, hi (ph cm^-2 s^-1, 3 sigma narrow line at 300-380 keV), note, reliability
 ("SPI 1 Ms point-like (instrument paper)", "cone_5", 2e-5, 5e-5, "Roques+2003 narrow-line 3 sigma, 1e6 s", "uncertain x2"),
 ("SPI full mission, inner 10 deg", "cone_10", 3e-6, 3e-5, "60Fe/26Al anchored; coded-mask diffuse penalty", "uncertain x3"),
 ("SPI full mission, |l|,|b|<30 deg", "box_30", 1e-5, 1e-4, "wide-ROI systematics-limited", "uncertain x3"),
 ("SPI 16 yr DM-line search |l|,|b|<47.5", "box_47.5", 1e-5, 1e-4, "Calore+2023-like ROI", "uncertain x3"),
 ("COSI 2 yr survey, inner 10 deg", "cone_10", 3e-6, 3e-5, "Tomsick+2023 ~1e-5 at 511 keV", "uncertain x3"),
 ("e-ASTROGAM/AMEGO concept, inner 10 deg", "cone_10", 1e-6, 5e-6, "concept-level", "uncertain"),
]
pd.DataFrame(SENS, columns=["instrument", "ROI", "S_lo", "S_hi", "note", "reliability"]).to_csv(os.path.join(OUT, "sensitivities.csv"), index=False)

F2_OVER_TAU_REF = 1e-22   # s^-1 reference
def line_flux(D, f2_over_tau, m=M_CHI):
    return f2_over_tau * D / (4 * np.pi * m)

frows = []
for _, r in Dtab.iterrows():
    row = dict(ROI=r.ROI, Omega_sr=r.Omega_sr)
    for name in PROFILES:
        row["Phi_%s_at_1e-22" % name] = line_flux(r["D_%s_GeVcm-2sr" % name], F2_OVER_TAU_REF)
    frows.append(row)
Ftab = pd.DataFrame(frows)
# bounds per instrument
brows = []
for inst, roi, slo, shi, note, rel in SENS:
    D = float(Dtab.loc[Dtab.ROI == roi, "D_NFW_GeVcm-2sr"].iloc[0])
    D_e = float(Dtab.loc[Dtab.ROI == roi, "D_Einasto_GeVcm-2sr"].iloc[0])
    D_b = float(Dtab.loc[Dtab.ROI == roi, "D_Burkert_GeVcm-2sr"].iloc[0])
    for lab, S in (("optimistic", slo), ("conservative", shi)):
        brows.append(dict(instrument=inst, ROI=roi, sens_case=lab, S_ph_cm2_s=S,
                          f2_over_tau_max_NFW=4 * np.pi * M_CHI * S / D,
                          f2_over_tau_max_Einasto=4 * np.pi * M_CHI * S / D_e,
                          f2_over_tau_max_Burkert=4 * np.pi * M_CHI * S / D_b,
                          tau_min_f2_0p42_NFW=0.42 * D / (4 * np.pi * M_CHI * S),
                          tau_min_f2_1p2em3_NFW=1.2e-3 * D / (4 * np.pi * M_CHI * S)))
Btab = pd.DataFrame(brows)
Ftab.to_csv(os.path.join(OUT, "line_flux_table.csv"), index=False)
Btab.to_csv(os.path.join(OUT, "f2_tau_bounds.csv"), index=False)
log("\nLine flux at f2/tau = 1e-22 s^-1, m = 1 TeV (ph cm^-2 s^-1):")
log(Ftab.to_string(index=False, float_format=lambda x: "%.3e" % x))
log("\nBounds on f2/tau_gamma (s^-1) and tau_min (s):")
log(Btab.to_string(index=False, float_format=lambda x: "%.3e" % x))

# ----------------------------------------------------------------------------------------------
# 5. CXB continuum from redshifted decays
# ----------------------------------------------------------------------------------------------
def H_z(z):
    return H0_S * np.sqrt(OM_M * (1 + z) ** 3 + OM_L)

def cxb_gruber(E_keV):
    """Gruber et al. 1999 fit (E > 60 keV) in photons cm^-2 s^-1 sr^-1 keV^-1 (recalled, uncertain x2)."""
    x = E_keV / 60.0
    EI = 0.0259 * x ** -5.5 + 0.504 * x ** -1.58 + 0.0288 * x ** -1.05   # keV cm^-2 s^-1 sr^-1 keV^-1
    return EI / E_keV

def decay_spectrum(E_keV, E0_keV, f2_over_tau, m=M_CHI):
    """dI/dE [ph cm^-2 s^-1 sr^-1 keV^-1] for a rest-frame line at E0 from decays with rate f2/tau per DM particle."""
    E = np.atleast_1d(E_keV).astype(float)
    z = E0_keV / E - 1
    n0 = RHO_DM_0 / m                       # cm^-3 mean number density of DM
    c_cm = C_KMS * 1e5
    out = c_cm * f2_over_tau * n0 / (4 * np.pi * E * H_z(z))
    out[E > E0_keV] = 0.0
    return out

crows = []
E_grid = np.linspace(20, 400, 3801)
for E0 in (300.0, 350.0, 380.0):
    spec = decay_spectrum(E_grid, E0, 1e-22)
    cxb = cxb_gruber(E_grid)
    ratio = spec / cxb
    i = np.nanargmax(np.where(E_grid <= E0, ratio, 0))
    # bounds: decay <= 100% CXB (conservative) or <= 10% (residual after source modelling), and with CXB/2 (COMPTEL-like level)
    for crit, frac, cxb_scale in (("100%_Gruber", 1.0, 1.0), ("10%_Gruber", 0.1, 1.0), ("100%_half-Gruber", 1.0, 0.5), ("10%_half-Gruber", 0.1, 0.5)):
        bound = 1e-22 * frac * cxb_scale / ratio[i]
        crows.append(dict(E0_keV=E0, criterion=crit, E_max_ratio_keV=E_grid[i], z_at_Emax=E0 / E_grid[i] - 1,
                          f2_over_tau_max=bound, tau_min_f2_0p42=0.42 / bound,
                          decay_dIdE_at_E0_for_1em22=float(decay_spectrum(E0 - 1e-6, E0, 1e-22)[0]),
                          cxb_dIdE_at_E0=float(cxb_gruber(E0)), cxb_E2dNdE_at_E0=float(cxb_gruber(E0) * E0 ** 2)))
Ctab = pd.DataFrame(crows)
Ctab.to_csv(os.path.join(OUT, "cxb_bound.csv"), index=False)
log("\nCXB continuum bounds:")
log(Ctab.to_string(index=False, float_format=lambda x: "%.3e" % x))
# spectral edge: fraction of decay photons within E0/2..E0
spec300 = decay_spectrum(E_grid, 300.0, 1e-22)
frac_top_half = np.trapezoid(spec300[E_grid >= 150], E_grid[E_grid >= 150]) / np.trapezoid(spec300, E_grid)

# ----------------------------------------------------------------------------------------------
# 6. Corpus models
# ----------------------------------------------------------------------------------------------
D10 = float(Dtab.loc[Dtab.ROI == "cone_10", "D_NFW_GeVcm-2sr"].iloc[0])
D30 = float(Dtab.loc[Dtab.ROI == "box_30", "D_NFW_GeVcm-2sr"].iloc[0])
S10_lo, S10_hi = 3e-6, 3e-5
F2_FO = 0.42
mrows = []
def add(model, delta, f2_today, tau_gamma, note):
    ft = f2_today / tau_gamma if tau_gamma > 0 else 0.0
    phi10 = line_flux(D10, ft); phi30 = line_flux(D30, ft)
    mrows.append(dict(model=model, delta_keV=delta, f2_today=f2_today, tau_gamma_s=tau_gamma, f2_over_tau=ft,
                      Phi_cone10=phi10, Phi_box30=phi30, Phi10_over_SPI_lo=phi10 / S10_lo, Phi10_over_SPI_hi=phi10 / S10_hi,
                      detectable_SPI=("yes" if phi10 > S10_hi else ("marginal" if phi10 > S10_lo else "no")), note=note))

# Higgsino: tau_tot = 1.2e6 s (nu nubar) -> f2(today) = 0.42 exp(-t_U/1.2e6) = 0
f2_H = F2_FO * np.exp(-T_U / 1.2e6) if T_U / 1.2e6 < 700 else 0.0
add("Higgsino (P007/P014/P026)", 300, f2_H, 0.06, "tau_nunubar 1.2e6 s, tau_gamma ~0.06 s: no chi2 today")
# photon-M1 / MiDM-like (P023/P042)
add("photon-M1 composite / MiDM (P023, P042)", 300, 0.0, 66e-6, "tau = 66 us: chi2 decays in flight; f2(today) = 0")
# dark photon, pure kinetic mixing: BR_gamma = 0
add("dark photon, pure kinetic mixing (P011)", 300, 1.2e-3, np.inf, "BR_gamma = 0 (on-shell mixing vanishes); invisible nu nubar only")
# dark photon + UV dipole, at the exothermic boundary (f2 = 1.2e-3, tau_tot = 7.2e16 s), BR_gamma = 1e-3, 1e-2, 1e-1
EXO = {300: (1.2e-3, 7.2e16), 350: (5.5e-5, 4.8e16), 380: (3.8e-7, 3.1e16)}
for delta, (f2x, taux) in EXO.items():
    for BR in (1e-3, 1e-2, 1e-1):
        add("dark photon + dipole, exothermic boundary, BR_gamma=%g" % BR, delta, f2x, taux / BR,
            "f2(today) at P011/P026 exothermic bound; tau_gamma = tau_tot/BR_gamma")
# dark photon ignoring the exothermic bound (f2 = 0.5 relic, tau_nunubar = 2.6e18 s at m_A' = 1 GeV), BR_gamma 1e-3..1e-1
for BR in (1e-3, 1e-2, 1e-1):
    add("dark photon m_A'=1 GeV, f2=0.5, no exothermic bound, BR_gamma=%g" % BR, 300, 0.5 * np.exp(-T_U / 2.6e18), 2.6e18 / BR,
        "EXCLUDED by LZ exothermic (P011); shown for comparison")
# generic long-lived chi2 with only the photon channel: f2(today) = 0.42 exp(-t_U/tau)
for tau in (1e19, 1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26):
    add("generic long-lived chi2, photon channel only", 300, F2_FO * np.exp(-T_U / tau), tau,
        "f2(today) = 0.42 exp(-t_U/tau); exothermic bound violated if this chi is the LZ scatterer")
# generic with f2 capped at the exothermic bound (1.2e-3) and tau_gamma free
for tau in (1e18, 1e19, 1e20, 1e21):
    add("generic chi2, f2 capped at exothermic 1.2e-3", 300, 1.2e-3, tau, "f2 fixed by LZ; tau_gamma free (needs an extra fast invisible channel)")
Mtab = pd.DataFrame(mrows)
Mtab.to_csv(os.path.join(OUT, "corpus_models.csv"), index=False)
log("\nCorpus models:")
log(Mtab[["model", "delta_keV", "f2_today", "tau_gamma_s", "Phi_cone10", "detectable_SPI"]].to_string(index=False, float_format=lambda x: "%.3e" % x))

# SPI-excluded tau_gamma for the generic scenario (photon channel only; f2 = 0.42 e^{-t_U/tau})
def excluded_tau_generic(S, D=D10):
    # solve 0.42 exp(-t_U/tau) D/(4 pi m tau) = S for the upper root (tau > t_U)
    from scipy.optimize import brentq
    g = lambda lt: np.log(F2_FO * np.exp(-T_U / 10 ** lt) * D / (4 * np.pi * M_CHI * 10 ** lt)) - np.log(S)
    return 10 ** brentq(g, np.log10(T_U), 30)
tau_gen = {"SPI_lo": excluded_tau_generic(S10_lo), "SPI_hi": excluded_tau_generic(S10_hi),
           "COSI_lo": excluded_tau_generic(3e-6), "eASTROGAM_lo": excluded_tau_generic(1e-6)}
# BR_gamma limit at the exothermic boundary (delta = 300)
BR_lim = {lab: S * 4 * np.pi * M_CHI * EXO[300][1] / (EXO[300][0] * D10) for lab, S in (("SPI_lo", S10_lo), ("SPI_hi", S10_hi), ("eASTROGAM", 1e-6))}
BR_lim_350 = {lab: S * 4 * np.pi * M_CHI * EXO[350][1] / (EXO[350][0] * D10) for lab, S in (("SPI_lo", S10_lo), ("SPI_hi", S10_hi))}
log("\nGeneric photon-only chi2: SPI/COSI exclude tau_gamma < %s" % {k: "%.2e" % v for k, v in tau_gen.items()})
log("BR_gamma limits at the delta=300 keV exothermic boundary: %s ; delta=350: %s" % ({k: "%.2e" % v for k, v in BR_lim.items()}, {k: "%.2e" % v for k, v in BR_lim_350.items()}))
# maximum of f2(today)/tau along the physical curve and where it sits vs exothermic
tau_grid = np.logspace(16, 20, 401)
ft_curve = F2_FO * np.exp(-T_U / tau_grid) / tau_grid
i_max = ft_curve.argmax()
log("Physical curve f2 e^{-tU/tau}/tau peaks at tau = %.2e s (f2 = %.3f), f2/tau = %.2e s^-1 -> Phi10 = %.2e" % (tau_grid[i_max], F2_FO * np.exp(-T_U / tau_grid[i_max]), ft_curve[i_max], line_flux(D10, ft_curve[i_max])))

# ----------------------------------------------------------------------------------------------
# 7. Doppler width, dipole, recoil
# ----------------------------------------------------------------------------------------------
dop = {}
for E0 in (300.0, 350.0, 380.0):
    sig_los = V0 / np.sqrt(2)                        # 168 km/s isotropic SHM
    sig_E = E0 * sig_los / C_KMS
    v_sun = float(np.linalg.norm(lz.V_SUN_PEC + np.array([0, V0, 0])))
    dop[str(int(E0))] = dict(sigma_los_kms=sig_los, sigma_E_keV=sig_E, FWHM_keV=2.3548 * sig_E,
                             sigma_E_innerhalo_250kms_keV=E0 * 250 / C_KMS, FWHM_innerhalo_keV=2.3548 * E0 * 250 / C_KMS,
                             dipole_amplitude_keV=E0 * v_sun / C_KMS, dipole_peak_to_peak_keV=2 * E0 * v_sun / C_KMS,
                             earth_orbit_annual_amplitude_keV=E0 * lz.V_EARTH_ORBIT / C_KMS,
                             recoil_shift_eV=1e3 * E0 ** 2 / (2 * M_CHI * 1e6),   # keV^2/(GeV) -> eV : E0^2/(2 m) with m in keV
                             SPI_FWHM_keV=2.0, COSI_FWHM_keV_1pct=0.01 * E0, v_sun_kms=v_sun)
json.dump(dop, open(os.path.join(OUT, "doppler.json"), "w"), indent=1)
log("\nDoppler: %s" % json.dumps(dop["300"], indent=None))

# morphology numbers
GCsr = {n: float(Ddir.loc[(Ddir.l_deg == 0) & (Ddir.b_deg == 0), "Dsr_" + n].iloc[0]) for n in PROFILES}
ACsr = {n: float(Ddir.loc[(Ddir.l_deg == 180), "Dsr_" + n].iloc[0]) for n in PROFILES}
contrast = {n: GCsr[n] / ACsr[n] for n in PROFILES}
frac10 = {n: float(Dtab.loc[Dtab.ROI == "cone_10", "D_%s_GeVcm-2sr" % n].iloc[0] / Dtab.loc[Dtab.ROI == "all_sky", "D_%s_GeVcm-2sr" % n].iloc[0]) for n in PROFILES}
frac30 = {n: float(Dtab.loc[Dtab.ROI == "box_30", "D_%s_GeVcm-2sr" % n].iloc[0] / Dtab.loc[Dtab.ROI == "all_sky", "D_%s_GeVcm-2sr" % n].iloc[0]) for n in PROFILES}
log("GC(1 deg avg)/anticentre intensity contrast: %s ; fraction of all-sky flux in 10 deg cone: %s ; in 30 deg box: %s" % (
    {k: "%.1f" % v for k, v in contrast.items()}, {k: "%.3f" % v for k, v in frac10.items()}, {k: "%.3f" % v for k, v in frac30.items()}))

# ----------------------------------------------------------------------------------------------
# 8. Figures (palette from the dataviz reference instance; light mode)
# ----------------------------------------------------------------------------------------------
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4", green="#008300", violet="#4a3aa7", red="#e34948")
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": TXT2, "axes.labelcolor": TXT, "xtick.color": TXT2, "ytick.color": TXT2,
                     "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.6})

# Fig 1: (tau_gamma, f2_today) plane
fig, ax = plt.subplots(figsize=(7.2, 5.2))
tau = np.logspace(10, 28, 400)
S_D = lambda S: 4 * np.pi * M_CHI * S / D10
# SPI bracket
ax.fill_between(tau, tau * S_D(S10_lo), 1.5, color=C["orange"], alpha=0.12, lw=0)
ax.fill_between(tau, tau * S_D(S10_hi), 1.5, color=C["orange"], alpha=0.22, lw=0)
ax.plot(tau, tau * S_D(S10_hi), color=C["orange"], lw=2, label="SPI 10° line, S = 3×10⁻⁵ (conservative)")
ax.plot(tau, tau * S_D(S10_lo), color=C["orange"], lw=1.2, ls="--", label="SPI/COSI 10° line, S = 3×10⁻⁶ (optimistic)")
# CXB
cx100 = float(Ctab.loc[(Ctab.E0_keV == 300) & (Ctab.criterion == "100%_Gruber"), "f2_over_tau_max"].iloc[0])
cx10 = float(Ctab.loc[(Ctab.E0_keV == 300) & (Ctab.criterion == "10%_Gruber"), "f2_over_tau_max"].iloc[0])
ax.plot(tau, tau * cx100, color=C["violet"], lw=2, label="CXB continuum, 100 % of Gruber (1999)")
ax.plot(tau, tau * cx10, color=C["violet"], lw=1.2, ls="--", label="CXB continuum, 10 % residual")
# Planck (P026 windows): polygon in (tau, f2) from quoted windows
pl_f2 = [1e-3, 1e-2, 1e-1, 0.5]
pl_lo = [1e13, 4e12, 2e12, 1.6e12]
pl_hi = [7.9e15, 2e17, 2.5e18, 1.3e19]
poly_x = pl_lo + pl_hi[::-1]; poly_y = pl_f2 + pl_f2[::-1]
ax.fill(poly_x + [poly_x[0]], poly_y + [poly_y[0]], color=C["red"], alpha=0.18, lw=0)
ax.plot(pl_lo, pl_f2, color=C["red"], lw=1.5); ax.plot(pl_hi, pl_f2, color=C["red"], lw=1.5, label="Planck CMB (P026 windows)")
# exothermic (LZ) - f2 today > 1.2e-3 excluded for LZ-fit models at delta = 300 keV
ax.axhspan(1.2e-3, 1.5, xmin=0, xmax=1, facecolor="none", edgecolor=C["blue"], hatch="//", lw=0, alpha=0.18)
ax.axhline(1.2e-3, color=C["blue"], lw=1.8, label="LZ exothermic: f₂ > 1.2×10⁻³ excluded (δ = 300 keV, P011)")
# physical curve
ax.plot(tau, F2_FO * np.exp(-T_U / tau), color=TXT, lw=1.4, ls=":", label="f₂(today) = 0.42 e^(−t_U/τ), γ channel only")
# model points
for BR, mk, dy in ((1e-3, "o", -14), (1e-2, "s", -26), (1e-1, "D", -38)):
    ax.plot(EXO[300][1] / BR, EXO[300][0], marker=mk, ms=7, color=C["aqua"], mec="white", mew=1.2, ls="none")
    ax.annotate("BR_γ = %g" % BR, (EXO[300][1] / BR, EXO[300][0]), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=7.5, color=TXT2,
                arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6))
ax.plot([], [], marker="o", color=C["aqua"], ls="none", label="dark photon + UV dipole at the exothermic boundary")
ax.plot(np.logspace(20, 26, 7), F2_FO * np.exp(-T_U / np.logspace(20, 26, 7)), marker="^", ms=6, color=C["yellow"], mec="white", ls="none", label="generic long-lived χ₂ (f₂ = 0.42, τ_γ = 10²⁰–10²⁶ s)")
ax.annotate("Higgsino: τ_γ ≈ 0.06 s, f₂(today) = 0\nphoton-M1/MiDM: τ = 66 μs, f₂ = 0\n(off scale, no line)", xy=(2e10, 3e-6), fontsize=8, color=TXT2)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e10, 1e28); ax.set_ylim(1e-6, 1.2)
ax.set_xlabel("τ(χ₂ → χ₁ γ)  [s]"); ax.set_ylabel("f₂(today) = n(χ₂)/n(DM)")
ax.set_title("Excluded (τ_γ, f₂) for a 300 keV Galactic line, m_χ = 1 TeV, NFW", loc="left", color=TXT)
ax.grid(True, which="major"); ax.legend(fontsize=7.3, loc="lower right", frameon=False)
ax.text(3e13, 1.2e-4, "SPI excluded:\nabove / left of the orange lines", color=C["orange"], fontsize=8, ha="left")
ax.text(1.5e13, 3e-2, "Planck\n(P026)", color=C["red"], fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P066_fig1_f2_tau_plane.png"), dpi=170); plt.close(fig)

# Fig 2: line flux vs ROI (grouped horizontal bars) at f2/tau = 1e-22
fig, ax = plt.subplots(figsize=(7.2, 4.6))
roi_order = ["cone_5", "cone_10", "box_10", "cone_30", "box_30", "box_47.5", "all_sky", "anticentre_cone_30", "high_lat_|b|>30"]
labels = ["cone 5°", "cone 10°", "box |l|,|b|<10°", "cone 30°", "box <30°", "box <47.5°", "all sky", "anti-centre 30°", "|b| > 30°"]
y = np.arange(len(roi_order))
w = 0.26
for k, (name, col) in enumerate((("NFW", C["blue"]), ("Einasto", C["orange"]), ("Burkert", C["aqua"]))):
    vals = [float(Ftab.loc[Ftab.ROI == r, "Phi_%s_at_1e-22" % name].iloc[0]) for r in roi_order]
    ax.barh(y + (k - 1) * w, vals, height=w - 0.03, color=col, label=name, edgecolor="white", lw=0.8)
# sensitivity brackets
for roi_key, slo, shi, lab, col, side in (("cone_10", 3e-6, 3e-5, "SPI / COSI, 10° (3σ, recalled)", C["red"], "right"),
                                          ("box_30", 1e-5, 1e-4, "SPI, 30° box (recalled)", C["red"], "right"),
                                          ("cone_10", 1e-6, 5e-6, "e-ASTROGAM / AMEGO", C["violet"], "left")):
    yy = roi_order.index(roi_key)
    ax.plot([slo, shi], [yy + 0.45, yy + 0.45], color=col, lw=2.2, solid_capstyle="butt")
    if side == "right":
        ax.text(shi * 1.15, yy + 0.45, lab, va="center", fontsize=7.5, color=col)
    else:
        ax.text(slo / 1.15, yy + 0.45, lab, va="center", ha="right", fontsize=7.5, color=col)
ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
ax.set_xscale("log"); ax.set_xlim(3e-8, 3e-3)
ax.set_xlabel("line flux Φ  [ph cm⁻² s⁻¹]  at f₂/τ_γ = 10⁻²² s⁻¹, m_χ = 1 TeV")
ax.set_title("Galactic χ₂ → χ₁ γ line flux per region (Φ ∝ f₂/τ_γ) with recalled 3σ sensitivities", loc="left", fontsize=9.5)
ax.grid(True, axis="x"); ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P066_fig2_flux_vs_roi.png"), dpi=170); plt.close(fig)

# Fig 3: CXB vs decay continuum
fig, ax = plt.subplots(figsize=(7.0, 4.4))
E = np.linspace(30, 420, 2000)
ax.plot(E, E ** 2 * cxb_gruber(E), color=TXT, lw=2, label="CXB, Gruber et al. 1999 fit (recalled)")
ax.fill_between(E, 0.5 * E ** 2 * cxb_gruber(E), E ** 2 * cxb_gruber(E), color=TXT, alpha=0.08, lw=0)
for E0, col in ((300, C["blue"]), (350, C["orange"]), (380, C["aqua"])):
    sp = decay_spectrum(E, E0, cx100)
    ax.plot(E, E ** 2 * sp, color=col, lw=1.8, label="decay continuum, E₀ = %d keV, f₂/τ = %.1e s⁻¹ (CXB bound)" % (E0, cx100))
sp_spi = decay_spectrum(E, 300, 5.1e-23)
ax.plot(E, E ** 2 * sp_spi, color=C["blue"], lw=1.2, ls="--", label="E₀ = 300 keV at the SPI bound f₂/τ = 5×10⁻²³ s⁻¹")
ax.set_yscale("log"); ax.set_ylim(1e-3, 60); ax.set_xlim(30, 420)
ax.set_xlabel("E  [keV]"); ax.set_ylabel("E² dN/dE  [keV cm⁻² s⁻¹ sr⁻¹]")
ax.set_title("Redshifted χ₂ → χ₁ γ continuum versus the cosmic X-/γ-ray background", loc="left")
ax.grid(True); ax.legend(frameon=False, fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P066_fig3_cxb.png"), dpi=170); plt.close(fig)

# Fig 4: sky map of D per sr (NFW), log10, sequential blue ramp
seq = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("seqblue", seq)
plt.close("all")
hp.mollview(np.log10(maps["NFW"]), fig=99, cmap=cmap, title="log₁₀ D(l,b) [GeV cm⁻² sr⁻¹], NFW: line intensity ∝ f₂ D/(4π m τ_γ)",
            unit="log₁₀ D", min=21.85, max=23.2, cbar=True, notext=True, flip="astro")
fig = plt.figure(99)
hp.graticule(dpar=30, dmer=60, color=TXT2, alpha=0.4)
fig.savefig(os.path.join(FIG, "P066_fig4_skymap.png"), dpi=170); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# 9. Results JSON
# ----------------------------------------------------------------------------------------------
res = dict(
    D_factors={r.ROI: {n: r["D_%s_GeVcm-2sr" % n] for n in PROFILES} for _, r in Dtab.iterrows()},
    validation=val, contrast_GC_over_anticentre=contrast, fraction_allsky_in_cone10=frac10, fraction_allsky_in_box30=frac30,
    flux_at_1em22={r.ROI: {n: r["Phi_%s_at_1e-22" % n] for n in PROFILES} for _, r in Ftab.iterrows()},
    f2_over_tau_bounds=Btab.to_dict(orient="records"), cxb=Ctab.to_dict(orient="records"), decay_photon_fraction_in_top_half_energy=float(frac_top_half),
    generic_tau_excluded=tau_gen, BR_gamma_limit_delta300=BR_lim, BR_gamma_limit_delta350=BR_lim_350,
    physical_curve_peak=dict(tau=float(tau_grid[i_max]), f2=float(F2_FO * np.exp(-T_U / tau_grid[i_max])), f2_over_tau=float(ft_curve[i_max]), Phi10=float(line_flux(D10, ft_curve[i_max]))),
    doppler=dop, constants=dict(R_SUN=R_SUN, R_S=R_S, ALPHA_EIN=ALPHA_EIN, R_C_BUR=R_C_BUR, R_MAX=R_MAX, RHO_SUN=RHO_SUN, M_CHI=M_CHI, T_U=T_U, RHO_DM_0=RHO_DM_0, NSIDE=NSIDE),
    recalled_knowledge=RECALLED, runtime_s=time.time() - t0)
json.dump(res, open(os.path.join(OUT, "P066_results.json"), "w"), indent=1, default=float)
open(os.path.join(OUT, "run_log.txt"), "w").write("\n".join(LOG))
log("\nDone in %.1f s" % (time.time() - t0))
