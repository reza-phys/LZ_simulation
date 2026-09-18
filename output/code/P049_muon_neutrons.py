"""P049 -- Muon-induced high-energy neutrons at 4850 ft: an independent estimate of untagged
single scatters above 8 MeV in the LZ TPC, and the meaning of the 41-minute-earlier muon.

Run from the simulation root:  .venv/bin/python output/code/P049_muon_neutrons.py

Everything here is a back-of-the-envelope chain (flux x yield x spectrum x escape x geometry x
attenuation x single-scatter probability x untagged fraction) with every recalled input flagged in
RECALLED and scanned over a stated range.  The single-scatter probabilities per neutron entering the
LXe are taken from P013's mono-energetic transport MC table (output/work/P013/P013_results.json,
key MC/mono, threshold 3 keV, 4.74 t fiducial cylinder, wall source, black-disk angular pattern).
"""
import json, math, os, sys
import numpy as np
from scipy import integrate, special, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "output/code")
from common import lzcommon as lz

OUT = "output/work/P049"; FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
LOG = []
def log(s=""):
    print(s); LOG.append(s)

LIVE_D = lz.LZ["live_days"]                      # 220 d (paper)
T_LIVE = LIVE_D * 86400.0                         # s
UL_LZ = lz.LZ["muon_neutron_UL90"]                # 4.6e-4 SS of any energy (supplement "Neutrons")
TAG_ROCK = (0.80, 0.04)                           # rock-neutron tagging efficiency (supplement)
DELAYED_WINDOW_US = 600.0                         # delayed veto window (samples paragraph)

# ----------------------------------------------------------------------------------------------
# Recalled inputs (flagged).  Central value, (low, high) range, reliability.
# ----------------------------------------------------------------------------------------------
RECALLED = {
    "phi_mu": dict(val=5.3e-9, rng=(4.4e-9, 5.5e-9), unit="cm^-2 s^-1",
                   note="total muon flux at the SURF 4850 ft (Davis campus) level; Majorana Demonstrator measurement "
                        "5.31+-0.17e-9; Mei-Hime flat-overburden formula at 4.3 km w.e. gives 3.8e-9", rel="likely"),
    "E_mu_mean": dict(val=320.0, rng=(280.0, 350.0), unit="GeV",
                      note="mean muon energy at 4.3 km w.e., Mei-Hime <E>=eps(1-e^{-bh})/(gamma-2) with eps=693 GeV, "
                           "b=0.4/km w.e., gamma=3.77", rel="likely"),
    "Y_n": dict(val=3.5e-4, rng=(2.5e-4, 5.0e-4), unit="n per (mu g cm^-2)",
                note="muon-induced neutron yield in standard rock at ~300 GeV; Mei-Hime 4.14e-6 E^0.74 gives 3.0e-4, "
                     "FLUKA/GEANT4 spread and the assignment's 4e-4 bracketed", rel="likely (x1.5)"),
    "f_hard": dict(val=0.30, rng=(0.20, 0.40), unit="",
                   note="fraction of produced neutrons in the hard (cascade) component E^-1 (1-50 MeV) -> E^-2 (50 MeV-3 GeV); "
                        "the rest is an evaporation Maxwellian T=1.2 MeV; tuned so that f(>10 MeV)~0.15-0.25, f(>100 MeV)~0.02-0.04",
                   rel="uncertain (x2 on f(>8.3 MeV))"),
    "E_knee": dict(val=50.0, rng=(30.0, 100.0), unit="MeV", note="knee where the cascade spectrum steepens from E^-1 to E^-2",
                   rel="uncertain"),
    "rho_rock": dict(val=2.7, rng=(2.6, 2.9), unit="g cm^-3", note="Homestake rock density (~2.7-2.85)", rel="likely"),
    "lam_esc": dict(val=(20.0, 30.0, 40.0), rng=((12.0, 20.0, 30.0), (30.0, 45.0, 60.0)), unit="cm",
                    note="effective rock depth from which a neutron escapes still above 8.3 MeV, for 8-30 / 30-100 / >100 MeV: "
                         "elastic+nonelastic removal (lambda_tot 7-8 cm, lambda_nonel 14-35 cm in SiO2 rock)", rel="uncertain (x1.5)"),
    "tank": dict(val=(3.8, 5.9), rng=None, unit="m", note="LZ water tank radius 3.8 m, height 5.9 m (pi R^2 H = 268 m^3 vs 229 t water + 17 t GdLS + cryostat)",
                 rel="likely"),
    "gdls_env": dict(val=(1.9, 3.9), rng=((1.7, 3.4), (2.1, 4.2)), unit="m",
                     note="outer envelope of the GdLS acrylic vessels around the cryostat (61 cm thick side tanks)", rel="uncertain"),
    "tpc": dict(val=(0.728, 1.456), rng=None, unit="m", note="LZ active TPC radius 72.8 cm, height 145.6 cm (as in P013)", rel="likely"),
    "n_zenith": dict(val=3.5, rng=(2.5, 4.5), unit="", note="muon zenith distribution I ~ cos^n(theta) at 4.3 km w.e.", rel="likely"),
    "L_path": dict(val=2.5, rng=(2.0, 3.0), unit="m", note="water+GdLS path from tank surface to the TPC for neutrons aimed at it", rel="uncertain"),
    "lam_att_hi": dict(val=1.0, rng=(0.8, 1.2), unit="m",
                       note="effective attenuation length in water for E>100 MeV neutrons, including cascade regeneration "
                            "(nuclear interaction length in water 83 g/cm^2 = 0.83 m)", rel="uncertain"),
    "lam_att_mid": dict(val=0.7, rng=(0.55, 0.9), unit="m", note="same for 50-100 MeV", rel="uncertain"),
    "sigma_H": dict(val="0.94 b (10 MeV), 0.69 (15), 0.48 (20), 0.38 (25), 0.30 (30), 0.17 (50)", rng=None, unit="b",
                    note="n-p total cross section", rel="likely (+-10%)"),
    "sigma_O_nonel": dict(val="0.5 b (10-30 MeV), 0.4 (50), 0.3 (100 MeV)", rng=None, unit="b", note="n-16O nonelastic", rel="likely (+-30%)"),
    "GdLS": dict(val="LAB C18H30 rho=0.86; n_H 6.3e22, n_C 3.8e22 cm^-3; sigma_H(100 MeV) 0.075 b; sigma_C nonel 0.22 b, el 0.28 b",
                 rng=None, unit="", note="for the crude prompt-tag estimate", rel="likely"),
    "Gd_capture": dict(val="~30 us mean capture time in 0.1% Gd LS; ~200 us in pure water", rng=None, unit="us", note="", rel="likely"),
    "beta_n": dict(val="9Li 178 ms (beta-n 51%), 8He 119 ms (16%), 17N 4.17 s (95%), 16N 7.13 s (beta-gamma only); 9Li yield in water ~1.9e-7 /(mu g cm^-2); "
                   "beta-delayed neutron energies <= 2 MeV", rng=None, unit="", note="Super-K spallation yields; ENSDF", rel="likely"),
}

def rec(k): return RECALLED[k]["val"]

# ----------------------------------------------------------------------------------------------
# 1. Muon rates through cylinders and the 41 / 127 minute gaps
# ----------------------------------------------------------------------------------------------
def cyl_rate(phi_h, R, H, n):
    """Muon rate through a vertical cylinder for I(theta) = I0 cos^n theta.
    phi_h = flux through a horizontal plane = 2 pi I0 /(n+2).
    Rate = int I(theta) A_proj(theta) dOmega, A_proj = pi R^2 cos + 2 R H sin."""
    I0 = phi_h * (n + 2) / (2 * math.pi)
    top = 2 * math.pi * I0 * math.pi * R**2 / (n + 2)
    side_int = integrate.quad(lambda t: math.cos(t)**n * math.sin(t)**2, 0, math.pi / 2)[0]
    side = 2 * math.pi * I0 * 2 * R * H * side_int
    return top + side, top, side

log("=== 1. Muon rates and the 41 / 127 min gaps ===")
phi = rec("phi_mu"); n_z = rec("n_zenith")
geoms = {"water tank (R 3.8 m, H 5.9 m)": rec("tank") , "GdLS envelope (R 1.9 m, H 3.9 m)": rec("gdls_env"),
         "TPC active (R 0.728 m, H 1.456 m)": rec("tpc")}
rate_rows = []
for name, (R, H) in geoms.items():
    r, top, side = cyl_rate(phi, R * 100, H * 100, n_z)
    rows = dict(volume=name, R_m=R, H_m=H, rate_per_h=r * 3600, top_per_h=top * 3600, side_per_h=side * 3600,
                mean_interval_min=1 / r / 60,
                P_gap_ge_41min=math.exp(-r * 41 * 60), P_gap_ge_127min=math.exp(-r * 127 * 60),
                P_gap_le_41min=1 - math.exp(-r * 41 * 60))
    # range over flux and n
    rs = [cyl_rate(p, R * 100, H * 100, nn)[0] for p in RECALLED["phi_mu"]["rng"] for nn in RECALLED["n_zenith"]["rng"]]
    rows["rate_range_per_h"] = (min(rs) * 3600, max(rs) * 3600)
    rows["P_ge41_range"] = (math.exp(-max(rs) * 41 * 60), math.exp(-min(rs) * 41 * 60))
    rate_rows.append(rows)
    log(f"{name}: rate {r*3600:.2f}/h (top {top*3600:.2f}, sides {side*3600:.2f}); mean interval {1/r/60:.1f} min; "
        f"P(gap>=41 min) = {math.exp(-r*41*60):.2e}; P(gap>=127 min) = {math.exp(-r*127*60):.2e}; "
        f"rate range {rows['rate_range_per_h'][0]:.2f}-{rows['rate_range_per_h'][1]:.2f}/h")
# GdLS envelope geometry range
rs = [cyl_rate(phi, R * 100, H * 100, n_z)[0] for (R, H) in RECALLED["gdls_env"]["rng"]]
log(f"GdLS envelope geometry range: {rs[0]*3600:.2f}-{rs[1]*3600:.2f}/h -> P(gap>=41) {math.exp(-rs[1]*41*60):.3f}-{math.exp(-rs[0]*41*60):.3f}")
# MUSUN check: 2.1e9 muons = 1200 yr -> implied generation area
musun_rate = 2.1e9 / (1200 * 365.25 * 86400)
log(f"MUSUN: 2.1e9 muons / 1200 yr = {musun_rate:.4f} mu/s -> generation area {musun_rate/phi/1e4:.0f} m^2 at phi={phi:.1e}")
# Physical relevance windows
r_tank = cyl_rate(phi, 380, 590, n_z)[0]
log(f"P(a tank-crossing muon within 1 ms before the event) = {r_tank*1e-3:.1e}; within 600 us = {r_tank*600e-6:.1e}")
# LZ UL decomposition
ul_poisson = 2.302585 * (LIVE_D / 365.25) / 1200
log(f"90% UL from 0 events in 1200 yr scaled to 220 d: {ul_poisson:.3e}; x(1-0.8) = {ul_poisson*0.2:.3e}; x2 (yield uncertainty) = {ul_poisson*0.4:.3e} (paper: {UL_LZ:.1e})")

# ----------------------------------------------------------------------------------------------
# 2. Neutron production spectrum in rock and fractions above thresholds
# ----------------------------------------------------------------------------------------------
E_MIN_248 = 8.20   # MeV, P013 (natural Xe)
E_MIN_270 = 8.92
def spectrum(E, f_hard=rec("f_hard"), knee=rec("E_knee"), T_evap=1.2, E_lo=1.0, E_hi=3000.0):
    """dN/dE (normalised to 1 neutron) : evaporation Maxwellian (1-f_hard) + hard tail f_hard."""
    E = np.asarray(E, float)
    evap = 2 / math.sqrt(math.pi) * np.sqrt(E) / T_evap**1.5 * np.exp(-E / T_evap)
    # hard: 1/E on [E_lo, knee], knee/E^2 above, normalised on [E_lo, E_hi]
    norm = math.log(knee / E_lo) + knee * (1 / knee - 1 / E_hi)
    hard = np.where(E < knee, 1 / E, knee / E**2) / norm
    hard = np.where((E >= E_lo) & (E <= E_hi), hard, 0.0)
    return (1 - f_hard) * evap + f_hard * hard

def frac_above(E0, **kw):
    return integrate.quad(lambda e: spectrum(e, **kw), E0, 3000.0, limit=200, points=[kw.get("knee", rec("E_knee"))])[0]

log("\n=== 2. Production spectrum fractions ===")
fr = {}
for E0 in [1, E_MIN_248, 10, 20, 30, 50, 100, 300, 1000]:
    fr[E0] = frac_above(E0)
    log(f"f(E > {E0:g} MeV) = {fr[E0]:.4f}")
fr_range = {}
for fh in RECALLED["f_hard"]["rng"]:
    for kn in RECALLED["E_knee"]["rng"]:
        fr_range[(fh, kn)] = (frac_above(E_MIN_248, f_hard=fh, knee=kn), frac_above(100, f_hard=fh, knee=kn))
log("f(>8.2) and f(>100) over (f_hard, knee) corners: " + "; ".join(f"({a},{b}): {v[0]:.3f}/{v[1]:.4f}" for (a, b), v in fr_range.items()))

# Production per unit rock-wall area and neutrons/day within a 2 m shell (assignment's bookkeeping number)
Y = rec("Y_n"); rho = rec("rho_rock")
prod_col = phi * Y * rho     # neutrons cm^-3 s^-1 ... per cm of depth: n /(cm^2 s) per cm
log(f"\nProduction per unit wall area per cm depth: phi*Y*rho = {prod_col:.3e} n cm^-3 s^-1  -> within 2 m: {prod_col*200:.3e} n cm^-2 s^-1 = {prod_col*200*1e4*86400:.3f} n m^-2 d^-1")
A_cav = 700.0  # m^2 cavern rock surface surrounding the tank (illustrative)
log(f"For ~{A_cav:.0f} m^2 of cavern rock surface: {prod_col*200*1e4*86400*A_cav:.0f} n/day produced within 2 m, of which >8.2 MeV: {prod_col*200*1e4*86400*A_cav*fr[E_MIN_248]:.0f}/day")

# ----------------------------------------------------------------------------------------------
# 3. Escape from rock, entry into the water tank, attenuation and single scatters
# ----------------------------------------------------------------------------------------------
E_grid = np.geomspace(E_MIN_248, 3000.0, 400)

def lam_esc(E, vals=rec("lam_esc")):
    E = np.asarray(E, float)
    return np.where(E < 30, vals[0], np.where(E < 100, vals[1], vals[2]))

def direction_factor(E):
    """fraction of escaping neutrons emitted towards the cavern: 0.5 for the quasi-isotropic 8-30 MeV part,
    rising to 1 for the forward (muon-aligned) cascade part."""
    E = np.asarray(E, float)
    return np.clip(0.5 + 0.5 * np.log10(E / 30.0) / np.log10(300.0 / 30.0), 0.5, 1.0)

def wall_current(E, Y=Y, rho=rho, lam_vals=rec("lam_esc"), **spec_kw):
    """neutron current out of the rock, per cm^2 per s per MeV"""
    return phi * Y * rho * spectrum(E, **spec_kw) * lam_esc(E, lam_vals) * direction_factor(E)

R_t, H_t = rec("tank"); R_x, H_x = rec("tpc")
A_top_t = math.pi * R_t**2; A_side_t = 2 * math.pi * R_t * H_t
A_eff_tank = (A_top_t + 0.5 * A_side_t) * 1e4      # cm^2: sides receive a half-weight current (walls partly shadowed, oblique)
f_geo_iso = (2 * math.pi * R_x * H_x + 2 * math.pi * R_x**2) / (2 * math.pi * R_t * H_t + 2 * math.pi * R_t**2)
f_geo_down = (math.pi * R_x**2) / (math.pi * R_t**2)
f_geo = 0.5 * (f_geo_iso + f_geo_down)
log(f"\n=== 3. Entry into the tank, geometry ===")
log(f"Tank top area {A_top_t:.1f} m^2, side {A_side_t:.1f} m^2, effective receiving area {A_eff_tank/1e4:.1f} m^2")
log(f"TPC/tank geometric fraction: isotropic {f_geo_iso:.4f}, downward {f_geo_down:.4f}; adopted f_geo = {f_geo:.4f}")

J = wall_current(E_grid)                                       # cm^-2 s^-1 MeV^-1
N_in = np.trapezoid(J, E_grid) * A_eff_tank * T_LIVE          # neutrons > 8.2 MeV entering the tank in 220 d
def integ(f, Emin, Emax=3000.0):
    m = (E_grid >= Emin) & (E_grid <= Emax)
    return np.trapezoid(f[m], E_grid[m])
for lo, hi in [(E_MIN_248, 30), (30, 100), (100, 3000)]:
    log(f"  entering tank, {lo:g}-{hi:g} MeV: {integ(J, lo, hi)*A_eff_tank*T_LIVE:.1f} in 220 d")
log(f"Neutrons > 8.2 MeV entering the water tank in 220 d: {N_in:.1f}  (flux at the wall {np.trapezoid(J,E_grid):.2e} cm^-2 s^-1; "
    f">100 MeV: {integ(J,100):.2e})")

# --- attenuation in water: energy-dependent removal ---
def sigma_H(E):   # barn, n-p total, recalled
    Ek = np.array([8.0, 10, 15, 20, 25, 30, 50, 100, 200, 1000]); s = np.array([1.15, 0.94, 0.69, 0.48, 0.38, 0.30, 0.17, 0.075, 0.043, 0.034])
    return np.interp(np.log(E), np.log(Ek), s)
def sigma_O_nonel(E):
    Ek = np.array([8.0, 30, 50, 100, 1000]); s = np.array([0.5, 0.5, 0.4, 0.3, 0.3])
    return np.interp(np.log(E), np.log(Ek), s)
N_H_W, N_O_W = 6.69e22, 3.34e22

def lam_removal(E, lam_hi=rec("lam_att_hi"), lam_mid=rec("lam_att_mid")):
    """effective attenuation length (cm) for staying above 8.2 MeV while crossing water.
    E<50 MeV: H elastic removal (P(E'>8.2|E) = 1-8.2/E for isotropic CM n-p) + O nonelastic;
    50-100 MeV: lam_mid; >100 MeV: lam_hi (cascade regeneration included, recalled)."""
    E = np.asarray(E, float)
    p_remove = np.clip(E_MIN_248 / E, 0, 1)                    # isotropic CM n-p: E' uniform on [0,E] -> P(E' < 8.2) = 8.2/E
    Sigma = N_H_W * sigma_H(E) * 1e-24 * p_remove + N_O_W * sigma_O_nonel(E) * 1e-24
    lam_low = 1 / Sigma
    lam = np.where(E < 50, lam_low, np.where(E < 100, lam_mid * 100, lam_hi * 100))
    # continuity: never let the low-energy branch exceed lam_mid
    return np.minimum(lam, np.where(E < 50, lam_mid * 100, np.inf))

def survival(E, L_m=rec("L_path"), **kw):
    return np.exp(-L_m * 100 / lam_removal(E, **kw))

log("\n=== 4. Attenuation in water (path %.1f m) ===" % rec("L_path"))
for E0 in [10, 15, 20, 30, 50, 100, 200, 500, 1000]:
    log(f"  E = {E0:5g} MeV: lambda_rem = {float(lam_removal(E0)):6.1f} cm, survival S = {float(survival(E0)):.2e}")
log(f"  simple 'each H collision halves' picture: lambda_tot(10 MeV) = {1/(N_H_W*0.94e-24+N_O_W*1.5e-24):.1f} cm -> "
    f"{250/(1/(N_H_W*0.94e-24+N_O_W*1.5e-24)):.0f} collisions in 2.5 m; 2^-N = {2**(-250/(1/(N_H_W*0.94e-24+N_O_W*1.5e-24))):.1e}")

# --- single-scatter probabilities per neutron entering the LXe, from P013's mono-energetic MC ---
p13 = json.load(open("output/work/P013/P013_results.json"))
mono = p13["MC"]["mono"]
Em = np.array(sorted(float(k) for k in mono))
p_roi = np.array([mono[str(int(e)) if float(int(e)) == e else str(e)]["n_ss_roi"] / mono[str(int(e)) if float(int(e)) == e else str(e)]["N"] for e in Em])
p_win = np.array([mono[str(int(e)) if float(int(e)) == e else str(e)]["n_200_270"] / mono[str(int(e)) if float(int(e)) == e else str(e)]["N"] for e in Em])
p_win_err = np.sqrt(np.array([mono[str(int(e)) if float(int(e)) == e else str(e)]["n_200_270"] for e in Em])) / np.array([mono[str(int(e)) if float(int(e)) == e else str(e)]["N"] for e in Em])
def interp_log(E, y):
    E = np.asarray(E, float)
    return np.interp(np.log(np.clip(E, Em[0], Em[-1])), np.log(Em), y)
log("\n=== 5. P013 mono-energetic SS probabilities per neutron entering the LXe (thr 3 keV, 4.74 t FV) ===")
for e, a, b, c in zip(Em, p_roi, p_win, p_win_err):
    log(f"  {e:6.1f} MeV: P(SS in ROI) = {a:.2e}; P(SS 200-270 keV) = {b:.2e} +- {c:.1e}; F = {b/a:.3f}")

# --- combine ---
def expected(Y=Y, f_hard=rec("f_hard"), knee=rec("E_knee"), lam_vals=rec("lam_esc"), L_m=rec("L_path"),
             lam_hi=rec("lam_att_hi"), lam_mid=rec("lam_att_mid"), f_untag=1 - TAG_ROCK[0], phi_mu=phi, fgeo=f_geo, return_parts=False):
    Jw = phi_mu * Y * rho * spectrum(E_grid, f_hard=f_hard, knee=knee) * lam_esc(E_grid, lam_vals) * direction_factor(E_grid)
    N_tank = Jw * A_eff_tank * T_LIVE                        # per MeV
    S = survival(E_grid, L_m=L_m, lam_hi=lam_hi, lam_mid=lam_mid)
    N_lxe = N_tank * fgeo * S                                 # neutrons > 8.2 MeV reaching the LXe, per MeV
    n_lxe = np.trapezoid(N_lxe, E_grid)
    n_ss_any = np.trapezoid(N_lxe * interp_log(E_grid, p_roi), E_grid) * f_untag
    n_win = np.trapezoid(N_lxe * interp_log(E_grid, p_win), E_grid) * f_untag
    if return_parts:
        return dict(N_tank=np.trapezoid(N_tank, E_grid), N_aimed=np.trapezoid(N_tank, E_grid) * fgeo, N_lxe=n_lxe,
                    N_ss_any_untagged=n_ss_any, N_win_untagged=n_win,
                    N_lxe_by_band={f"{lo}-{hi}": float(np.trapezoid(N_lxe[(E_grid>=lo)&(E_grid<hi)], E_grid[(E_grid>=lo)&(E_grid<hi)])) for lo, hi in [(8.2,30),(30,100),(100,3000)]},
                    spec_tank=N_tank, spec_lxe=N_lxe)
    return n_win, n_ss_any

log("\n=== 6. Combined expectation, 220 live days ===")
C = expected(return_parts=True)
log(f"neutrons > 8.2 MeV entering tank {C['N_tank']:.1f}; aimed at TPC {C['N_aimed']:.2f}; reaching LXe above 8.2 MeV {C['N_lxe']:.3f} "
    f"(bands {C['N_lxe_by_band']})")
log(f"untagged SS of any energy: {C['N_ss_any_untagged']:.2e}  (LZ 90% UL 4.6e-4; ratio {C['N_ss_any_untagged']/UL_LZ:.2f})")
log(f"untagged SS in 200-270 keV: {C['N_win_untagged']:.2e}  (P013 from LZ UL: 0.6-1.6e-5)")
log(f"F(200-270 | SS) effective = {C['N_win_untagged']/C['N_ss_any_untagged']:.3f}")

# --- sensitivity scan: one-at-a-time and full corners ---
log("\n=== 7. Sensitivity (one-at-a-time, low/high) ===")
base_win, base_any = expected()
sens = []
params = {
    "phi_mu": ("phi_mu", RECALLED["phi_mu"]["rng"]),
    "Y_n": ("Y", RECALLED["Y_n"]["rng"]),
    "f_hard": ("f_hard", RECALLED["f_hard"]["rng"]),
    "E_knee": ("knee", RECALLED["E_knee"]["rng"]),
    "lam_esc": ("lam_vals", RECALLED["lam_esc"]["rng"]),
    "L_path": ("L_m", RECALLED["L_path"]["rng"]),
    "lam_att_hi": ("lam_hi", RECALLED["lam_att_hi"]["rng"]),
    "lam_att_mid": ("lam_mid", RECALLED["lam_att_mid"]["rng"]),
    "f_untag": ("f_untag", (0.04, 0.24)),
    "f_geo": ("fgeo", (f_geo_down, f_geo_iso * 1.5)),
}
for name, (kw, (lo, hi)) in params.items():
    wl, al = expected(**{kw: lo}); wh, ah = expected(**{kw: hi})
    sens.append(dict(parameter=name, low=lo if not isinstance(lo, tuple) else str(lo), high=hi if not isinstance(hi, tuple) else str(hi),
                     N_win_low=wl, N_win_high=wh, N_any_low=al, N_any_high=ah, span=max(wl, wh) / min(wl, wh)))
    log(f"  {name:12s}: N_win {wl:.2e} .. {wh:.2e} (span x{max(wl,wh)/min(wl,wh):.2f}); N_any {al:.2e} .. {ah:.2e}")
# stacked
lo_kw = dict(phi_mu=4.4e-9, Y=2.5e-4, f_hard=0.2, knee=30.0, lam_vals=RECALLED["lam_esc"]["rng"][0], L_m=3.0, lam_hi=0.8, lam_mid=0.55, f_untag=0.04)
hi_kw = dict(phi_mu=5.5e-9, Y=5.0e-4, f_hard=0.4, knee=100.0, lam_vals=RECALLED["lam_esc"]["rng"][1], L_m=2.0, lam_hi=1.2, lam_mid=0.9, f_untag=0.24)
w_lo, a_lo = expected(**lo_kw); w_hi, a_hi = expected(**hi_kw)
log(f"all-low corner: N_win {w_lo:.2e}, N_any {a_lo:.2e};  all-high corner: N_win {w_hi:.2e}, N_any {a_hi:.2e}")
# log-space quadrature of one-at-a-time half-spans (treat as independent lognormal factors)
half = np.array([0.5 * abs(math.log(s["N_win_high"] / s["N_win_low"])) for s in sens])
sig_ln = math.sqrt((half**2).sum())
log(f"log-quadrature 1-sigma factor on N_win: x/ {math.exp(sig_ln):.2f} -> 68% range {base_win/math.exp(sig_ln):.2e} .. {base_win*math.exp(sig_ln):.2e}")
dom = sorted(sens, key=lambda s: -s["span"])
log("dominant uncertainties (span): " + ", ".join(f"{s['parameter']} x{s['span']:.2f}" for s in dom[:5]))

# --- crude independent estimate of the untagged fraction ---
log("\n=== 8. Crude untagged fraction for a neutron that scatters once in the FV ===")
nH_ls, nC_ls = 6.3e22, 3.8e22
def p_no_prompt_gdls(E, t_cm=61.0):
    Sig = nH_ls * sigma_H(E) * 1e-24 + nC_ls * 0.22e-24     # visible: n-p elastic (proton recoil light) + C nonelastic
    return math.exp(-Sig * t_cm)
for E0 in [50, 100, 300, 1000]:
    p_in = p_no_prompt_gdls(E0); p_out = p_in
    lam_xe = 1 / (1.33e22 * (interp_xs := np.interp(np.log(E0), np.log([50, 100, 300, 1000]), [3.9, 3.6, 2.8, 2.6])) * 1e-24)
    p_skin = math.exp(-2 * 6.0 / lam_xe) ** 1  # two Skin crossings of ~6 cm; every scatter assumed tagged (upper bound on tagging)
    p_skin_half = math.exp(-2 * 6.0 / lam_xe * 0.5)  # half of Skin scatters below threshold
    p_delayed_miss = 0.3   # neutron leaves without thermalising in GdLS/water within 600 us or capture gamma below 200 keV: recalled guess
    log(f"  E={E0:5g} MeV: P(no prompt GdLS light in) {p_in:.2f} x out {p_out:.2f} x P(no Skin tag) {p_skin:.2f}-{p_skin_half:.2f} x P(delayed miss) {p_delayed_miss} "
        f"= {p_in*p_out*p_skin*p_delayed_miss:.3f}-{p_in*p_out*p_skin_half*p_delayed_miss:.3f} (LZ: 0.20 +- 0.04)")

# --- delayed emitters ---
log("\n=== 9. Delayed neutron emitters (why 41 min cannot matter) ===")
r_tank_s = r_tank
for iso, thalf_s in [("9Li", 0.178), ("8He", 0.119), ("17N", 4.17), ("16N", 7.13), ("137Xe", 229.0)]:
    n_half = 41 * 60 / thalf_s
    log(f"  {iso}: T1/2 = {thalf_s} s -> 41 min = {n_half:.0f} half-lives, survival 2^-{n_half:.0f} = {2.0**(-n_half) if n_half < 1000 else 0:.1e}")
path_g = 500.0  # g/cm^2 mean water path of a tank-crossing muon (~5 m)
Y_Li9 = 1.9e-7
n_Li9 = r_tank_s * path_g * Y_Li9 * T_LIVE
log(f"  9Li produced in the water in 220 d: {n_Li9:.1f} (rate {r_tank_s:.2e} mu/s x {path_g:.0f} g/cm^2 x {Y_Li9:.1e}); beta-n branch 0.51 -> {0.51*n_Li9:.1f} neutrons of E_n <= 1 MeV -> E_R <= {0.0303*1000:.0f} keV")
log(f"  beta-delayed neutrons E_n <= 2 MeV give E_R,max = {0.0303*2000:.0f} keV << 248 keV: kinematically irrelevant irrespective of timing")
log(f"  neutron capture time in GdLS ~30 us, in water ~200 us; the 600 us delayed window covers > 95% of Gd captures; "
    f"a neutron from a muon 41 min earlier would have been captured 41 min earlier.")

# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
c1, c2, c3, c4 = "#3b6ea8", "#c8553d", "#2a9d8f", "#7a6c5d"
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
Ep = np.geomspace(0.1, 3000, 600)
ax[0].loglog(Ep, spectrum(Ep) * Ep, color=c1, label="production (rock), E dN/dE")
Jp = wall_current(np.clip(Ep, 8.2, None)) * Ep; Jp[Ep < 8.2] = np.nan
ax[0].loglog(Ep, Jp / np.nanmax(Jp) * np.max(spectrum(Ep) * Ep) * 0.3, color=c2, label="current out of rock (>8.2 MeV, scaled)")
Sp = survival(np.clip(Ep, 8.2, None)); Sp[Ep < 8.2] = np.nan
ax[0].loglog(Ep, Jp * Sp / np.nanmax(Jp) * np.max(spectrum(Ep) * Ep) * 0.3, color=c3, label=f"after {rec('L_path'):.1f} m water (scaled)")
ax[0].axvline(8.2, color="k", ls=":", lw=1); ax[0].text(8.8, 0.5, "8.2 MeV\n(248 keV)", fontsize=8)
ax[0].set_xlabel("neutron energy E (MeV)"); ax[0].set_ylabel("E dN/dE (arb.)"); ax[0].set_ylim(1e-4, 1); ax[0].legend(fontsize=8, loc="lower left")
ax[0].set_title("Muon-induced neutron spectra", fontsize=10)
Lp = np.linspace(0, 3.5, 200)
for E0, col in zip([10, 20, 50, 100, 500], [c4, c1, c3, c2, "k"]):
    ax[1].semilogy(Lp, np.exp(-Lp * 100 / float(lam_removal(E0))), color=col, label=f"{E0} MeV (lambda = {float(lam_removal(E0)):.0f} cm)")
ax[1].axvspan(2.0, 3.0, color="0.9"); ax[1].axvline(rec("L_path"), color="k", ls="--", lw=1)
ax[1].set_ylim(1e-6, 1); ax[1].set_xlabel("water + GdLS path L (m)"); ax[1].set_ylabel("P(still above 8.2 MeV)")
ax[1].legend(fontsize=8); ax[1].set_title("Survival above the 248 keV threshold", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P049_fig1_spectra_attenuation.png"), dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.6))
labels = [s["parameter"] for s in dom][::-1]
lows = [min(s["N_win_low"], s["N_win_high"]) for s in dom][::-1]; highs = [max(s["N_win_low"], s["N_win_high"]) for s in dom][::-1]
y = np.arange(len(labels))
ax.barh(y, np.array(highs) - np.array(lows), left=lows, color=c1, alpha=0.8)
ax.axvline(base_win, color="k", lw=1.2, label=f"central {base_win:.1e}")
ax.axvline(UL_LZ * 0.022, color=c2, ls="--", label="P013 from LZ UL x F (0.6e-5)"); ax.axvline(UL_LZ * 0.035, color=c2, ls="--")
ax.axvline(UL_LZ, color=c3, ls=":", label="LZ UL, any energy 4.6e-4")
ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlabel("expected untagged 200-270 keV single scatters in 220 d")
ax.legend(fontsize=8, loc="lower right"); ax.set_title("One-at-a-time sensitivity of the window expectation", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P049_fig2_sensitivity.png"), dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.2))
t = np.linspace(0, 180, 600)
for rr, col in zip(rate_rows, [c1, c3, c2]):
    lam = rr["rate_per_h"] / 60
    ax.plot(t, lam * np.exp(-lam * t), color=col, label=f"{rr['volume'].split(' (')[0]}: mean {rr['mean_interval_min']:.0f} min")
ax.axvline(41, color="k", ls="--", lw=1); ax.text(42, ax.get_ylim()[1] * 0.6, "41 min (OD)", fontsize=8)
ax.axvline(127, color="k", ls=":", lw=1); ax.text(128, ax.get_ylim()[1] * 0.4, "127 min (TPC)", fontsize=8)
ax.set_yscale("log"); ax.set_ylim(1e-4, 0.3); ax.set_xlabel("time since previous muon (min)"); ax.set_ylabel("probability density (per min)")
ax.legend(fontsize=8); ax.set_title("Exponential waiting-time densities for muons", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P049_fig3_muon_intervals.png"), dpi=160); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# Save tables
# ----------------------------------------------------------------------------------------------
import pandas as pd
pd.DataFrame(rate_rows).to_csv(os.path.join(OUT, "muon_rates.csv"), index=False)
pd.DataFrame(sens).to_csv(os.path.join(OUT, "sensitivity.csv"), index=False)
pd.DataFrame(dict(E_MeV=Em, p_ss_roi=p_roi, p_ss_200_270=p_win, p_ss_200_270_err=p_win_err, F=p_win / p_roi)).to_csv(os.path.join(OUT, "p013_mono_ss.csv"), index=False)
spec_rows = pd.DataFrame(dict(E_MeV=E_grid, dNdE_prod=spectrum(E_grid), J_wall=J, lam_rem_cm=lam_removal(E_grid), S=survival(E_grid),
                              N_tank_perMeV=C["spec_tank"], N_lxe_perMeV=C["spec_lxe"]))
spec_rows.to_csv(os.path.join(OUT, "spectrum_chain.csv"), index=False)
results = dict(
    recalled={k: {kk: vv for kk, vv in v.items()} for k, v in RECALLED.items()},
    muon_rates=rate_rows, musun_rate_per_s=musun_rate, musun_area_m2=musun_rate / phi / 1e4,
    UL_decomposition=dict(poisson_0_in_1200yr_scaled=ul_poisson, x_untagged=ul_poisson * 0.2, x_untagged_x2=ul_poisson * 0.4, paper=UL_LZ),
    P_muon_within_1ms=r_tank * 1e-3, spectrum_fractions={str(k): v for k, v in fr.items()},
    spectrum_fraction_corners={f"fhard{a}_knee{b}": v for (a, b), v in fr_range.items()},
    production_per_cm2_s_per_cm=prod_col, geometry=dict(A_top_m2=A_top_t, A_side_m2=A_side_t, A_eff_m2=A_eff_tank / 1e4, f_geo_iso=f_geo_iso, f_geo_down=f_geo_down, f_geo=f_geo),
    chain={k: (v if not isinstance(v, np.ndarray) else None) for k, v in C.items() if k not in ("spec_tank", "spec_lxe")},
    survival={str(E0): float(survival(E0)) for E0 in [10, 15, 20, 30, 50, 100, 200, 500, 1000]},
    lam_removal_cm={str(E0): float(lam_removal(E0)) for E0 in [10, 15, 20, 30, 50, 100, 200, 500, 1000]},
    expected=dict(N_win_central=base_win, N_any_central=base_any, N_win_all_low=w_lo, N_win_all_high=w_hi, N_any_all_low=a_lo, N_any_all_high=a_hi,
                  sigma_ln=sig_ln, N_win_68=(base_win / math.exp(sig_ln), base_win * math.exp(sig_ln)), ratio_any_to_LZ_UL=base_any / UL_LZ),
    sensitivity=sens, Li9_in_water_220d=n_Li9,
)
def _default(o):
    if isinstance(o, (np.floating, np.integer)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, tuple): return list(o)
    return str(o)
json.dump(results, open(os.path.join(OUT, "P049_results.json"), "w"), indent=1, default=_default)
open(os.path.join(OUT, "run_log.txt"), "w").write("\n".join(LOG) + "\n")
log("\nsaved: " + OUT)
