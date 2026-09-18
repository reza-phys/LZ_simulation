"""
P077 -- S1 pulse-shape discrimination at 550 phd: what singlet/triplet ratios, photon statistics
and propagation in a 1.5 m TPC allow, and why LZ called it inconclusive.

Run from the simulation root:   .venv/bin/python output/code/P077_s1_psd.py

Sections
  0  recalled inputs (all flagged) -> work/P077/recalled_inputs.json
  1  optical photon Monte Carlo in an LZ-like cylinder -> propagation-time distributions,
     top/bottom split and collection efficiency vs depth   (work/P077/optics_vs_z.csv)
  2  S1 waveform toy: emission-time mixtures (singlet/triplet/recombination) + propagation + PMT TTS
     + 10 ns digitisation; statistics f_p, <t>, ideal template log-likelihood ratio
  3  ER/NR separation for the parameter brackets at 540 phd (event) and 275 phd (150 keV AmBe NR)
  4  scan of the effective mean-emission-time difference; what LZ's "not inconsistent" statement allows
  5  position dependence of the statistics and the residual after position-matched templates
  6  RFR (below cathode) vs z = 26.4 cm origin: arrival-time offset, rise time, top/bottom asymmetry
  7  single-event likelihood-ratio table
Figures in work/P077/figures/.
"""
import sys, os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "output/code")
from common import lzcommon as lz

OUT = "output/work/P077"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(20260915)
T0 = time.time()

# ----------------------------------------------------------------------------------------------
# 0. Inputs.  Paper numbers from lzcommon; everything else recalled and flagged.
# ----------------------------------------------------------------------------------------------
S1C_EVENT = lz.LZ["ev_S1c"]              # 540.1 phd (paper)
Z_EVENT = lz.LZ["ev_z_above_cathode_cm"]  # 26.4 cm (paper)
R_EVENT = 45.9                            # cm (dossier / Fig. 3 caption)
G1 = lz.LZ["g1"]

REC = {
    # geometry
    "TPC_radius_cm": (72.8, "LZ design papers (NIM A 953, 163047)", "likely"),
    "drift_length_cm": (145.6, "LZ design papers; P022 Fig.3 axis", "likely"),
    "RFR_depth_cm": (13.75, "paper (MSSI paragraph), via P022/P042", "certain"),
    # optics
    "n_LXe_175nm": (1.69, "Solovov et al. 2004 / Hitachi et al. 2005 refractive index of LXe at 175-178 nm", "likely"),
    "n_group_variant": (1.9, "group index estimated from LXe dispersion (Grace et al. 2017)", "uncertain"),
    "Rayleigh_length_cm": (35.0, "Ishida 1997 / Seidel 2002: 29-40 cm at 175 nm", "uncertain"),
    "PTFE_reflectance_LXe": (0.97, "Neves et al. 2017 / Kravitz et al. 2020 (>= 0.95 immersed)", "likely"),
    "array_plane_reflectance": (0.5, "assumed: PTFE reflector + PMT window mix", "uncertain"),
    "PMT_detection_prob_per_hit": (0.25, "QE ~0.30 at 175 nm (R11410) x photocathode coverage ~0.8", "uncertain"),
    "grid_transparency": (0.9, "woven-mesh optical transparency, assumed", "uncertain"),
    "PMT_TTS_sigma_ns": (1.5, "assignment baseline; R11410 datasheet TTS ~ 9 ns FWHM (sigma 3.8) as variant", "uncertain"),
    "digitiser_sample_ns": (10.0, "LZ DAQ 100 MS/s", "likely"),
    # scintillation timing
    "tau_singlet_ns": ((2.2, 4.3), "Hitachi et al. PRB 27, 5279 (1983); paper quotes 2-4 ns", "certain (range)"),
    "tau_triplet_ns": ((21.0, 27.0), "Hitachi 1983 (22/27 ns); paper quotes 21-28 ns", "certain (range)"),
    "tau_recombination_zero_field_ns": (45.0, "Kubota et al. PRB 20, 3486 (1979): 45 ns slow component for electrons at 0 V/cm", "likely"),
    "Is_It_ER_zero_field": (0.05, "Hitachi 1983 electrons (recombination dominated)", "likely"),
    "Is_It_ER_drift_field": ((0.4, 0.6), "assignment bracket; LUX PRD 97, 112002 (2018) at 180 V/cm", "uncertain"),
    "Is_It_NR": ((1.0, 3.0), "Hitachi fission fragments 1.6; Akimov 2002 / Dawson 2005 / Kwong 2010 Xe recoils ~1.5", "uncertain"),
    "empirical_dtau_eff_ns": ((1.0, 4.0), "LUX 2018 / XMASS 2018: ER-NR effective decay-time difference of a few ns", "uncertain"),
    # yields
    "Nph_150keV_NR": (2700, "P009 contour scale Nq = 11.32 E^1.112 minus ~10% electrons", "likely"),
}
with open(os.path.join(OUT, "recalled_inputs.json"), "w") as f:
    json.dump({k: {"value": v[0], "presumed_source": v[1], "reliability": v[2]} for k, v in REC.items()}, f, indent=1)

R_TPC = REC["TPC_radius_cm"][0]
L_DRIFT = REC["drift_length_cm"][0]
Z_BOT = -REC["RFR_depth_cm"][0]          # bottom PMT array plane (below the RFR)
Z_SURF = L_DRIFT + 0.5                   # liquid surface just above the gate
C_LXE = 29.9792 / REC["n_LXe_175nm"][0]  # cm/ns
LAM_R = REC["Rayleigh_length_cm"][0]
R_PTFE = REC["PTFE_reflectance_LXe"][0]
R_ARR = REC["array_plane_reflectance"][0]
EPS_DET = REC["PMT_detection_prob_per_hit"][0]
T_GRID = REC["grid_transparency"][0]
SIN_CRIT = 1.0 / REC["n_LXe_175nm"][0]
# The optical parameters are tuned (P077_TUNE=1 mode) so that the collection efficiency reproduces
# LZ's g1 = 0.110 phd/photon at the detector centre; the tuned values overwrite the defaults below.
TUNED = dict(R_PTFE=0.95, R_ARR=0.20, EPS_DET=0.16, T_GRID=0.90, N_GRIDS_TOP=2)
R_PTFE, R_ARR, EPS_DET, T_GRID = TUNED["R_PTFE"], TUNED["R_ARR"], TUNED["EPS_DET"], TUNED["T_GRID"]
N_GRIDS_TOP = TUNED["N_GRIDS_TOP"]      # gate + anode crossings on the way to the top array
REC["optics_tuned"] = (TUNED, "tuned in this work to reproduce g1 = 0.110 at the TPC centre (see details)", "derived")
TTS = REC["PMT_TTS_sigma_ns"][0]
DT_DIG = REC["digitiser_sample_ns"][0]

N_DET_EVENT = int(round(S1C_EVENT))                 # 540 detected photons
N_DET_AMBE = int(round(REC["Nph_150keV_NR"][0] * G1))  # ~ 297 phd for a 150 keV NR
N_PULSES = 10000

# ----------------------------------------------------------------------------------------------
# 1. Optical Monte Carlo
# ----------------------------------------------------------------------------------------------
def lambertian(normal, n):
    """cosine-weighted directions about `normal` (unit vector, shape (3,)) for n photons"""
    u1, u2 = rng.random(n), rng.random(n)
    r = np.sqrt(u1); phi = 2 * np.pi * u2
    x, y, zc = r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - u1)
    # build basis
    a = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(normal, a); t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    return (x[:, None] * t1 + y[:, None] * t2 + zc[:, None] * normal)

def lambertian_radial(pos, n):
    """cosine-weighted directions about the inward radial normal at the wall, per photon"""
    nx, ny = -pos[:, 0] / R_TPC, -pos[:, 1] / R_TPC
    u1, u2 = rng.random(n), rng.random(n)
    r = np.sqrt(u1); phi = 2 * np.pi * u2
    a, b, c = r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - u1)
    # tangent basis: t1 = azimuthal, t2 = z
    t1 = np.stack([-ny, nx, np.zeros(n)], axis=1)
    t2 = np.array([0, 0, 1.0])
    nrm = np.stack([nx, ny, np.zeros(n)], axis=1)
    return a[:, None] * t1 + b[:, None] * t2 + c[:, None] * nrm

def isotropic(n):
    ct = rng.uniform(-1, 1, n); ph = rng.uniform(0, 2 * np.pi, n); st = np.sqrt(1 - ct**2)
    return np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)

def optical_mc(z0, r0, n_ph=200000, c_lxe=C_LXE, lam_r=LAM_R, max_steps=400):
    """Track n_ph photons from (r0,0,z0). Returns arrival times (ns) and array flag (1=top,0=bottom)."""
    pos = np.tile(np.array([r0, 0.0, z0]), (n_ph, 1))
    d = isotropic(n_ph)
    t = np.zeros(n_ph)
    alive = np.ones(n_ph, bool)
    det_t, det_top = [], []
    for step in range(max_steps):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        p, v = pos[idx], d[idx]
        # distance to cylinder wall
        a = v[:, 0]**2 + v[:, 1]**2
        b = 2 * (p[:, 0] * v[:, 0] + p[:, 1] * v[:, 1])
        c = p[:, 0]**2 + p[:, 1]**2 - R_TPC**2
        disc = np.maximum(b**2 - 4 * a * c, 0)
        s_wall = np.where(a > 1e-12, (-b + np.sqrt(disc)) / (2 * np.maximum(a, 1e-12)), np.inf)
        # planes: bottom array, cathode grid (z=0), liquid surface
        with np.errstate(divide="ignore", invalid="ignore"):
            s_bot = np.where(v[:, 2] < 0, (Z_BOT - p[:, 2]) / v[:, 2], np.inf)
            s_top = np.where(v[:, 2] > 0, (Z_SURF - p[:, 2]) / v[:, 2], np.inf)
            s_cat = np.where(np.sign(v[:, 2]) * np.sign(p[:, 2]) < 0, (0.0 - p[:, 2]) / v[:, 2], np.inf)
        s_ray = rng.exponential(lam_r, idx.size)
        S = np.stack([s_wall, s_bot, s_top, s_cat, s_ray], axis=1)
        k = np.argmin(S, axis=1)
        s = S[np.arange(idx.size), k]
        pos[idx] = p + s[:, None] * v
        t[idx] += s / c_lxe
        u = rng.random(idx.size)
        # --- wall: PTFE Lambertian or absorbed
        m = k == 0
        if m.any():
            ii = idx[m]
            refl = u[m] < R_PTFE
            alive[ii[~refl]] = False
            pp = pos[ii[refl]]
            # nudge inward to avoid re-hitting
            rr = np.sqrt(pp[:, 0]**2 + pp[:, 1]**2)
            pos[ii[refl], 0] = pp[:, 0] * (R_TPC - 1e-3) / rr
            pos[ii[refl], 1] = pp[:, 1] * (R_TPC - 1e-3) / rr
            d[ii[refl]] = lambertian_radial(pos[ii[refl]], refl.sum())
        # --- bottom array plane (one shield-grid crossing, then PMT detection or diffuse reflection)
        m = k == 1
        if m.any():
            ii = idx[m]
            uu = u[m]
            p_det = T_GRID * EPS_DET
            det = uu < p_det
            refl = (~det) & (uu < p_det + T_GRID * (1 - EPS_DET) * R_ARR)
            det_t.append(t[ii[det]]); det_top.append(np.zeros(det.sum(), int))
            alive[ii[det]] = False
            alive[ii[~det & ~refl]] = False
            pos[ii[refl], 2] = Z_BOT + 1e-3
            d[ii[refl]] = lambertian(np.array([0, 0, 1.0]), refl.sum())
        # --- liquid surface: TIR or transmit to the top array (gas transit ~ 0.2 ns)
        m = k == 2
        if m.any():
            ii = idx[m]
            vz = d[ii, 2]
            sin_i = np.sqrt(np.maximum(1 - vz**2, 0))
            tir = sin_i > SIN_CRIT
            # TIR: specular reflection
            pos[ii[tir], 2] = Z_SURF - 1e-3
            d[ii[tir], 2] = -d[ii[tir], 2]
            # transmitted: cross gate + anode grids, reach the top array
            tr = ~tir
            uu = u[m][tr]
            tg = T_GRID ** N_GRIDS_TOP
            p_det = tg * EPS_DET
            det = uu < p_det
            refl = (~det) & (uu < p_det + tg * (1 - EPS_DET) * R_ARR * tg)   # reflected light re-crosses the grids
            jj = ii[tr]
            t[jj] += 0.2
            det_t.append(t[jj[det]]); det_top.append(np.ones(det.sum(), int))
            alive[jj[det]] = False
            alive[jj[~det & ~refl]] = False
            pos[jj[refl], 2] = Z_SURF - 1e-3
            d[jj[refl]] = lambertian(np.array([0, 0, -1.0]), refl.sum())
        # --- cathode grid crossing
        m = k == 3
        if m.any():
            ii = idx[m]
            absorbed = u[m] > T_GRID
            alive[ii[absorbed]] = False
            pos[ii[~absorbed], 2] += np.sign(d[ii[~absorbed], 2]) * 1e-3
        # --- Rayleigh scattering (isotropic redirection approximation)
        m = k == 4
        if m.any():
            ii = idx[m]
            d[ii] = isotropic(ii.size)
    alive_frac = alive.mean()
    tt = np.concatenate(det_t) if det_t else np.array([])
    top = np.concatenate(det_top) if det_top else np.array([])
    return tt, top, alive_frac

print("== 1. optical MC ==")
if os.environ.get("P077_TUNE"):
    # tuning mode: scan optical parameters, print collection (= g1 equivalent when EPS_DET includes QE) and timing at the centre
    grid = []
    for rp in (0.95, 0.97):
        for ra in (0.2, 0.4):
            for ed in (0.12, 0.16, 0.20, 0.25):
                for tg in (0.85, 0.90):
                    grid.append(dict(R_PTFE=rp, R_ARR=ra, EPS_DET=ed, T_GRID=tg))
    for g in grid:
        R_PTFE, R_ARR, EPS_DET, T_GRID = g["R_PTFE"], g["R_ARR"], g["EPS_DET"], g["T_GRID"]
        out = []
        for z in (73.0, Z_EVENT):
            tt, top, af = optical_mc(z, R_EVENT, n_ph=40000)
            out.append(f"z={z:5.1f} coll={tt.size/40000:.3f} f_top={top.mean():.2f} <t>={tt.mean():5.1f} med={np.median(tt):4.1f} sd={tt.std():4.1f}")
        print(g, " | ".join(out))
    sys.exit(0)
Z_GRID = [-7.0, -2.0, 2.0, 10.0, Z_EVENT, 40.0, 60.0, 73.0, 100.0, 120.0, 140.0]
optics = {}
rows = []
for z in Z_GRID:
    tt, top, af = optical_mc(z, R_EVENT, n_ph=200000)
    direct_b = max(z - Z_BOT, 0) / C_LXE
    optics[z] = (tt, top)
    coll = tt.size / 200000
    ftop = top.mean()
    tba = (top.sum() - (1 - top).sum()) / top.size
    tb = tt[top == 0]; tp = tt[top == 1]
    rows.append(dict(z_cm=z, collection=coll, f_top=ftop, TBA=tba, t_mean_ns=tt.mean(), t_median_ns=np.median(tt),
                     t_sd_ns=tt.std(), t_mean_bottom=tb.mean(), t_mean_top=tp.mean() if tp.size else np.nan,
                     t_p10_bottom=np.percentile(tb, 10), direct_bottom_ns=direct_b, unfinished=af))
    print(f"z={z:6.1f}  coll={coll:.3f} f_top={ftop:.3f} TBA={tba:+.3f}  <t>={tt.mean():5.2f} med={np.median(tt):5.2f} "
          f"sd={tt.std():5.2f}  <t>_bot={tb.mean():5.2f} <t>_top={tp.mean():5.2f}  direct_bot={direct_b:.2f}")
import csv
with open(os.path.join(OUT, "optics_vs_z.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
# variants at the event depth: group index 1.9, Rayleigh 30 cm, and a low-reflectivity "fast optics" case (PTFE R = 0.90)
tt_v1, top_v1, _ = optical_mc(Z_EVENT, R_EVENT, n_ph=100000, c_lxe=29.9792 / 1.9)
tt_v2, top_v2, _ = optical_mc(Z_EVENT, R_EVENT, n_ph=100000, lam_r=30.0)
_R_SAVE = R_PTFE; R_PTFE = 0.90
tt_v3, top_v3, _ = optical_mc(Z_EVENT, R_EVENT, n_ph=200000)
R_PTFE = _R_SAVE
optics["fast"] = (tt_v3, top_v3)
# no-propagation reference (emission + TTS only): delta-function propagation
optics["none"] = (np.zeros(1000), np.zeros(1000, int))
print(f"variants at z_event: n=1.9 -> <t>={tt_v1.mean():.2f} sd={tt_v1.std():.2f};  lambda_R=30 -> <t>={tt_v2.mean():.2f} sd={tt_v2.std():.2f};  "
      f"PTFE R=0.90 -> coll={tt_v3.size/200000:.3f} <t>={tt_v3.mean():.2f} sd={tt_v3.std():.2f} f_top={top_v3.mean():.2f}")
print(f"[{time.time()-T0:.0f}s]")

# ----------------------------------------------------------------------------------------------
# 2. S1 waveform toy
# ----------------------------------------------------------------------------------------------
def emission_times(n, R, tau_s, tau_t, f_rec=0.0, tau_rec=45.0):
    """singlet fraction f_s = R/(1+R); optional recombination delay on a fraction f_rec of the light"""
    fs = R / (1.0 + R)
    sing = rng.random(n) < fs
    t = np.where(sing, rng.exponential(tau_s, n), rng.exponential(tau_t, n))
    if f_rec > 0:
        rec = rng.random(n) < f_rec
        t = t + np.where(rec, rng.exponential(tau_rec, n), 0.0)
    return t

def mean_emission(R, tau_s, tau_t, f_rec=0.0, tau_rec=45.0):
    fs = R / (1 + R)
    return fs * tau_s + (1 - fs) * tau_t + f_rec * tau_rec

def simulate_pulses(model, n_det, z, n_pulses=N_PULSES, tts=TTS, poisson=True):
    """returns dict of arrays (n_pulses): photon times sorted per pulse are not stored; statistics only"""
    tt, top = optics[z]
    if poisson:
        N = rng.poisson(n_det, n_pulses)
    else:
        N = np.full(n_pulses, n_det)
    Ntot = N.sum()
    te = emission_times(Ntot, **model)
    tp = tt[rng.integers(0, tt.size, Ntot)]
    tj = rng.normal(0, tts, Ntot)
    t = te + tp + tj
    ev = np.repeat(np.arange(n_pulses), N)
    # per-pulse statistics
    order = np.lexsort((t, ev))
    t = t[order]; ev = ev[order]
    starts = np.concatenate([[0], np.cumsum(N)[:-1]])
    t_first = t[starts]
    # first photon as t0 (LZ-like pulse start); also the 5th photon
    t5 = t[np.minimum(starts + 4, starts + N - 1)]
    rel = t - np.repeat(t_first, N)
    rel5 = t - np.repeat(t5, N)
    stats = {}
    for W in (10, 20, 30, 50):
        stats[f"fp{W}"] = np.bincount(ev, weights=(rel <= W), minlength=n_pulses) / N
    stats["tmean"] = np.bincount(ev, weights=rel, minlength=n_pulses) / N
    stats["tmean5"] = np.bincount(ev, weights=rel5, minlength=n_pulses) / N + 0.0
    # rise time 10% -> 50% of cumulative and t10 (relative to true t=0), for §6
    cum_idx10 = starts + np.maximum(np.ceil(0.10 * N).astype(int) - 1, 0)
    cum_idx50 = starts + np.maximum(np.ceil(0.50 * N).astype(int) - 1, 0)
    stats["t10_abs"] = t[cum_idx10]
    stats["t50_abs"] = t[cum_idx50]
    stats["rise_10_50"] = t[cum_idx50] - t[cum_idx10]
    # 10 ns digitised waveform: bins relative to the first sample containing light
    b = np.floor(rel / DT_DIG).astype(int)
    stats["fp20_dig"] = np.bincount(ev, weights=(b <= 1), minlength=n_pulses) / N   # first two samples
    stats["tmean_dig"] = np.bincount(ev, weights=(b + 0.5) * DT_DIG, minlength=n_pulses) / N
    stats["N"] = N
    return stats, (t, ev, N, t_first)

def template_pdf(model, z, tts=TTS, n=2000000, edges=np.arange(-15, 600.5, 0.5)):
    tt, top = optics[z]
    te = emission_times(n, **model)
    t = te + tt[rng.integers(0, tt.size, n)] + rng.normal(0, tts, n)
    h, _ = np.histogram(t, bins=edges)
    p = (h + 0.5) / (h.sum() + 0.5 * h.size) / 0.5      # density with a floor
    return edges, np.log(p)

def ideal_llr(raw, logp_nr, logp_er, edges, shifts=np.arange(-6, 6.01, 0.5)):
    """sum_i ln p_NR(t_i - s) / p_ER(t_i - s'), each hypothesis profiled over its own t0 shift s"""
    t, ev, N, t_first = raw
    n_pulses = N.size
    def prof(logp):
        best = np.full(n_pulses, -np.inf)
        for s in shifts:
            k = np.clip(np.floor((t - s - edges[0]) / 0.5).astype(int), 0, logp.size - 1)
            ll = np.bincount(ev, weights=logp[k], minlength=n_pulses)
            best = np.maximum(best, ll)
        return best
    return prof(logp_nr) - prof(logp_er)

def separation(a, b):
    """d = |mu_a - mu_b| / sqrt((sa^2+sb^2)/2); also AUC and leakage of b past the a-median (50% a acceptance) and at 90% a acceptance"""
    d = abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()))
    # AUC: P(a > b) if mean a > mean b else P(a<b)
    sgn = 1 if a.mean() > b.mean() else -1
    aa, bb = sgn * a, sgn * b
    auc = (np.searchsorted(np.sort(bb), aa)).mean() / bb.size
    leak50 = (bb > np.median(aa)).mean()
    leak90 = (bb > np.percentile(aa, 10)).mean()
    return d, auc, leak50, leak90

# ---- parameter sets (all recalled/uncertain; see REC)
MODELS = {
    "NR_base":   dict(R=1.5, tau_s=3.0, tau_t=24.0),
    "NR_lo":     dict(R=1.0, tau_s=3.0, tau_t=27.0),     # slowest NR in bracket
    "NR_hi":     dict(R=3.0, tau_s=3.0, tau_t=21.0),     # fastest NR
    "ER_base":   dict(R=0.5, tau_s=3.0, tau_t=26.0),     # drift-field ER, no recombination delay
    "ER_fast":   dict(R=0.6, tau_s=3.0, tau_t=24.0),     # fastest ER in bracket
    "ER_slow":   dict(R=0.4, tau_s=3.0, tau_t=27.0),
    "ER_rec":    dict(R=0.5, tau_s=3.0, tau_t=26.0, f_rec=0.3, tau_rec=15.0),   # residual recombination delay
    "ER_zeroF":  dict(R=0.05, tau_s=2.2, tau_t=27.0, f_rec=0.9, tau_rec=45.0),  # Hitachi/Kubota zero-field electrons
}
for k, m in MODELS.items():
    print(f"{k:9s} <t_emit> = {mean_emission(**m):5.2f} ns")

print("\n== 2/3. pulse simulation at the event depth ==")
STATS = {}
RAW = {}
for k, m in MODELS.items():
    for ndet, tag in ((N_DET_EVENT, "540"), (N_DET_AMBE, "AmBe")):
        STATS[(k, tag)], RAW[(k, tag)] = simulate_pulses(m, ndet, Z_EVENT)
print(f"[{time.time()-T0:.0f}s]  N_det AmBe = {N_DET_AMBE}")

edges, LOGP = None, {}
for k in ("NR_base", "ER_base", "NR_lo", "ER_fast", "ER_zeroF", "NR_hi", "ER_slow", "ER_rec"):
    edges, LOGP[k] = template_pdf(MODELS[k], Z_EVENT)

def compare(nr, er, tag, stat_keys=("fp20", "tmean", "fp20_dig", "tmean_dig", "fp10", "fp30", "fp50")):
    out = {}
    for s in stat_keys:
        a, b = STATS[(nr, tag)][s], STATS[(er, tag)][s]
        d, auc, l50, l90 = separation(a, b)
        out[s] = dict(d=d, auc=auc, leak50=l50, leak90=l90, mu_nr=a.mean(), mu_er=b.mean(), sd_nr=a.std(), sd_er=b.std())
    # ideal LLR
    llr_nr = ideal_llr(RAW[(nr, tag)], LOGP[nr], LOGP[er], edges)
    llr_er = ideal_llr(RAW[(er, tag)], LOGP[nr], LOGP[er], edges)
    d, auc, l50, l90 = separation(llr_nr, llr_er)
    out["ideal_llr"] = dict(d=d, auc=auc, leak50=l50, leak90=l90, mu_nr=llr_nr.mean(), mu_er=llr_er.mean(),
                            sd_nr=llr_nr.std(), sd_er=llr_er.std(),
                            LR_at_NR_median=float(np.exp(np.median(llr_nr))), LR_at_ER_median=float(np.exp(np.median(llr_er))),
                            LR_midway=float(np.exp(0.5 * (np.median(llr_nr) + np.median(llr_er)))))
    return out, llr_nr, llr_er

PAIRS = [("NR_base", "ER_base"), ("NR_lo", "ER_fast"), ("NR_hi", "ER_slow"), ("NR_base", "ER_rec"), ("NR_base", "ER_zeroF")]
RESULTS = {}
LLR = {}
for nr, er in PAIRS:
    for tag in ("540", "AmBe"):
        res, lnr, ler = compare(nr, er, tag)
        RESULTS[f"{nr}|{er}|{tag}"] = res
        LLR[(nr, er, tag)] = (lnr, ler)
        dE = mean_emission(**MODELS[nr]) - mean_emission(**MODELS[er])
        print(f"{nr:8s} vs {er:8s} [{tag:4s}] d<t_emit>={dE:+6.2f} ns | fp20: d={res['fp20']['d']:.2f} leak50={res['fp20']['leak50']:.3f} "
              f"| tmean: d={res['tmean']['d']:.2f} | dig tmean d={res['tmean_dig']['d']:.2f} | ideal LLR d={res['ideal_llr']['d']:.2f} "
              f"LR(NRmed)={res['ideal_llr']['LR_at_NR_median']:.1f} LR(ERmed)={res['ideal_llr']['LR_at_ER_median']:.2g}")
print(f"[{time.time()-T0:.0f}s]")

# optics variants at 540 phd: fast optics (PTFE R = 0.90) and no propagation spread at all (emission + TTS only)
print("\n== 3b. optics variants (540 phd) ==")
OPT_VAR = {}
for okey in ("fast", "none"):
    for nr, er in (("NR_base", "ER_base"), ("NR_lo", "ER_fast")):
        _, lp_nr = template_pdf(MODELS[nr], okey, n=800000)
        _, lp_er = template_pdf(MODELS[er], okey, n=800000)
        st_nr, raw_nr = simulate_pulses(MODELS[nr], N_DET_EVENT, okey, n_pulses=4000)
        st_er, raw_er = simulate_pulses(MODELS[er], N_DET_EVENT, okey, n_pulses=4000)
        d_fp = separation(st_nr["fp20"], st_er["fp20"])[0]
        d_tm = separation(st_nr["tmean"], st_er["tmean"])[0]
        lnr = ideal_llr(raw_nr, lp_nr, lp_er, edges); ler = ideal_llr(raw_er, lp_nr, lp_er, edges)
        d_ll, _, l50, _ = separation(lnr, ler)
        OPT_VAR[f"{okey}|{nr}|{er}"] = dict(d_fp20=d_fp, d_tmean=d_tm, d_llr=d_ll, leak50=l50,
                                            fp20_ER=float(st_er["fp20"].mean()), fp20_NR=float(st_nr["fp20"].mean()),
                                            sd_fp20=float(st_er["fp20"].std()))
        print(f"optics={okey:4s} {nr} vs {er}: d_fp20={d_fp:.2f} d_tmean={d_tm:.2f} d_llr={d_ll:.2f} leak50={l50:.3f}  fp20 ER/NR={st_er['fp20'].mean():.3f}/{st_nr['fp20'].mean():.3f}")
print(f"[{time.time()-T0:.0f}s]")

# ----------------------------------------------------------------------------------------------
# 4. Scan of the effective mean-emission-time difference (NR faster than ER by Delta) at fixed ER_base
# ----------------------------------------------------------------------------------------------
print("\n== 4. scan of the NR singlet/triplet ratio at fixed lifetimes (ER fixed at I_s/I_t = 0.5, 3/26 ns) ==")
scan_rows = []
R_SCAN = [0.55, 0.6, 0.7, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0]
er_mean = mean_emission(**MODELS["ER_base"])
DELTAS = []
for Rnr in R_SCAN:
    m = dict(R=Rnr, tau_s=3.0, tau_t=26.0)
    dl = er_mean - mean_emission(**m)
    DELTAS.append(dl)
    row = dict(R_NR=Rnr, delta_ns=dl, tau_t_NR=m["tau_t"])
    for ndet, tag in ((N_DET_EVENT, "540"), (N_DET_AMBE, "AmBe"), (100, "100")):
        st, raw = simulate_pulses(m, ndet, Z_EVENT, n_pulses=4000)
        st_er, raw_er = simulate_pulses(MODELS["ER_base"], ndet, Z_EVENT, n_pulses=4000)
        d_fp, _, l50, _ = separation(st["fp20"], st_er["fp20"])
        d_tm, _, l50t, _ = separation(st["tmean"], st_er["tmean"])
        _, lp = template_pdf(m, Z_EVENT, n=600000)
        lnr = ideal_llr(raw, lp, LOGP["ER_base"], edges); ler = ideal_llr(raw_er, lp, LOGP["ER_base"], edges)
        d_ll, _, l50l, _ = separation(lnr, ler)
        row.update({f"d_fp20_{tag}": d_fp, f"d_tmean_{tag}": d_tm, f"d_llr_{tag}": d_ll, f"leak50_llr_{tag}": l50l,
                    f"LR_NRmed_{tag}": float(np.exp(np.median(lnr))), f"LR_ERmed_{tag}": float(np.exp(np.median(ler)))})
    scan_rows.append(row)
    print(f"Delta={dl:4.1f} ns tau_t={m['tau_t']:5.1f} | 540: d_fp={row['d_fp20_540']:.2f} d_tm={row['d_tmean_540']:.2f} d_llr={row['d_llr_540']:.2f} "
          f"LR(NRmed)={row['LR_NRmed_540']:.2f} | AmBe: d_llr={row['d_llr_AmBe']:.2f} LR={row['LR_NRmed_AmBe']:.2f} | 100phd d_llr={row['d_llr_100']:.2f}")
with open(os.path.join(OUT, "delta_scan.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(scan_rows[0].keys())); w.writeheader(); w.writerows(scan_rows)
print(f"[{time.time()-T0:.0f}s]")

# What does "AmBe 150 keV NRs not inconsistent with the ER template" allow?
# A sample of n_AmBe events with single-event separation d(AmBe) shifts the sample mean by d*sqrt(n) sigma_mean.
# "Not inconsistent" taken as a < 2 sigma shift for n_AmBe = 10 (bracket 5-30) -> d(AmBe) < 2/sqrt(n).
d_amb = np.array([r["d_llr_AmBe"] for r in scan_rows]); d_540 = np.array([r["d_llr_540"] for r in scan_rows]); dl = np.array(DELTAS)
ratio_540_amb = np.nanmean(d_540[d_amb > 0] / d_amb[d_amb > 0])
allowed = {}
k_lin = d_amb[0] / dl[0]          # small-difference slope d(AmBe) per ns
k_lin540 = d_540[0] / dl[0]
for n_amb in (5, 10, 30):
    dmax = 2.0 / np.sqrt(n_amb)
    dl_max = float(np.interp(dmax, d_amb, dl)) if dmax >= d_amb[0] else float(dmax / k_lin)
    d540_max = float(np.interp(dl_max, dl, d_540)) if dl_max >= dl[0] else float(k_lin540 * dl_max)
    allowed[n_amb] = dict(d_AmBe_max=dmax, delta_ns_max=dl_max, d_540_max=d540_max, LR_NRmedian_max=float(np.exp(d540_max**2 / 2)),
                          LR_ERmedian_min=float(np.exp(-d540_max**2 / 2)))
    print(f"n_AmBe={n_amb:2d}: d(AmBe)<{dmax:.2f} -> Delta<{dl_max:.2f} ns -> d(540)<{d540_max:.2f} -> LR at NR median < {np.exp(d540_max**2/2):.2f}")
print(f"mean ratio d(540)/d(AmBe) = {ratio_540_amb:.2f} (sqrt(540/{N_DET_AMBE}) = {np.sqrt(540/N_DET_AMBE):.2f})")

# ----------------------------------------------------------------------------------------------
# 5. Position dependence
# ----------------------------------------------------------------------------------------------
print("\n== 5. position dependence ==")
pos_rows = []
for z in Z_GRID:
    if z < 0:
        continue
    st_er, _ = simulate_pulses(MODELS["ER_base"], N_DET_EVENT, z, n_pulses=3000)
    st_nr, _ = simulate_pulses(MODELS["NR_base"], N_DET_EVENT, z, n_pulses=3000)
    pos_rows.append(dict(z_cm=z, fp20_ER=st_er["fp20"].mean(), fp20_ER_sd=st_er["fp20"].std(), fp20_NR=st_nr["fp20"].mean(),
                         tmean_ER=st_er["tmean"].mean(), tmean_ER_sd=st_er["tmean"].std(), tmean_NR=st_nr["tmean"].mean(),
                         rise_ER=st_er["rise_10_50"].mean(), rise_ER_sd=st_er["rise_10_50"].std()))
    print(f"z={z:6.1f}: fp20 ER={st_er['fp20'].mean():.4f}+-{st_er['fp20'].std():.4f} NR={st_nr['fp20'].mean():.4f} | "
          f"<t> ER={st_er['tmean'].mean():.2f}+-{st_er['tmean'].std():.2f} NR={st_nr['tmean'].mean():.2f} | rise ER={st_er['rise_10_50'].mean():.2f}+-{st_er['rise_10_50'].std():.2f}")
with open(os.path.join(OUT, "position_dependence.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(pos_rows[0].keys())); w.writeheader(); w.writerows(pos_rows)
zz = np.array([r["z_cm"] for r in pos_rows]); tmE = np.array([r["tmean_ER"] for r in pos_rows]); fpE = np.array([r["fp20_ER"] for r in pos_rows])
sdE = np.array([r["tmean_ER_sd"] for r in pos_rows]); sdfp = np.array([r["fp20_ER_sd"] for r in pos_rows])
i26 = int(np.argmin(abs(zz - Z_EVENT))); i100 = int(np.argmin(abs(zz - 100)))
shift_tm = tmE[i100] - tmE[i26]; shift_fp = fpE[i100] - fpE[i26]
ernr_tm = pos_rows[i26]["tmean_ER"] - pos_rows[i26]["tmean_NR"]; ernr_fp = pos_rows[i26]["fp20_NR"] - pos_rows[i26]["fp20_ER"]
slope_tm = np.gradient(tmE, zz)[i26]; slope_fp = np.gradient(fpE, zz)[i26]
posdep = dict(shift_tmean_26_to_100_ns=float(shift_tm), shift_fp20_26_to_100=float(shift_fp), ER_NR_tmean_diff_ns=float(ernr_tm),
              ER_NR_fp20_diff=float(ernr_fp), shift_over_ERNR_tmean=float(shift_tm / ernr_tm), shift_over_ERNR_fp=float(shift_fp / ernr_fp),
              slope_tmean_ns_per_cm=float(slope_tm), slope_fp20_per_cm=float(slope_fp),
              residual_pm5cm_in_sigma_tmean=float(abs(slope_tm) * 5 / sdE[i26]), residual_pm10cm_in_sigma_tmean=float(abs(slope_tm) * 10 / sdE[i26]),
              residual_pm5cm_in_sigma_fp=float(abs(slope_fp) * 5 / sdfp[i26]), residual_pm10cm_in_sigma_fp=float(abs(slope_fp) * 10 / sdfp[i26]),
              sigma_tmean_540=float(sdE[i26]), sigma_fp20_540=float(sdfp[i26]))
print(json.dumps(posdep, indent=1))

# ----------------------------------------------------------------------------------------------
# 6. RFR vs z = 26.4 cm origin
# ----------------------------------------------------------------------------------------------
print("\n== 6. RFR vs FV origin ==")
rfr = {}
for z in (-7.0, -2.0, 2.0, 10.0, Z_EVENT):
    st, _ = simulate_pulses(MODELS["ER_base"], N_DET_EVENT, z, n_pulses=4000)
    tt, top = optics[z]
    rfr[z] = dict(rise_10_50=float(st["rise_10_50"].mean()), rise_sd=float(st["rise_10_50"].std()), t10_abs=float(st["t10_abs"].mean()),
                  t50_abs=float(st["t50_abs"].mean()), f_top=float(top.mean()), TBA=float(2 * top.mean() - 1),
                  t_first_bottom_ns=float(np.percentile(tt[top == 0], 1)), direct_bottom_ns=max(z - Z_BOT, 0) / C_LXE,
                  fp20=float(st["fp20"].mean()), fp20_sd=float(st["fp20"].std()))
    print(f"z={z:6.1f}: rise10-50={rfr[z]['rise_10_50']:.2f}+-{rfr[z]['rise_sd']:.2f} ns  t10={rfr[z]['t10_abs']:.2f} t50={rfr[z]['t50_abs']:.2f}  "
          f"f_top={rfr[z]['f_top']:.3f} TBA={rfr[z]['TBA']:+.3f}  direct_bot={rfr[z]['direct_bottom_ns']:.2f} ns  fp20={rfr[z]['fp20']:.4f}")
d_rise = (rfr[Z_EVENT]["rise_10_50"] - rfr[-7.0]["rise_10_50"]) / np.sqrt(0.5 * (rfr[Z_EVENT]["rise_sd"]**2 + rfr[-7.0]["rise_sd"]**2))
sig_tba = 2 * np.sqrt(rfr[Z_EVENT]["f_top"] * (1 - rfr[Z_EVENT]["f_top"]) / N_DET_EVENT)
d_tba = (rfr[Z_EVENT]["TBA"] - rfr[-7.0]["TBA"]) / sig_tba
d_fp = (rfr[-7.0]["fp20"] - rfr[Z_EVENT]["fp20"]) / rfr[Z_EVENT]["fp20_sd"]
rfr_summary = dict(direct_path_offset_ns=(Z_EVENT - (-7.0)) / C_LXE, rise_diff_ns=rfr[Z_EVENT]["rise_10_50"] - rfr[-7.0]["rise_10_50"],
                   rise_separation_sigma=float(d_rise), TBA_RFR=rfr[-7.0]["TBA"], TBA_event=rfr[Z_EVENT]["TBA"], sigma_TBA_540=float(sig_tba),
                   TBA_separation_sigma=float(d_tba), fp20_separation_sigma=float(d_fp), c_lxe_cm_per_ns=C_LXE)
print(json.dumps(rfr_summary, indent=1))

# ----------------------------------------------------------------------------------------------
# 7. Single-event LR table (Gaussian approximation on the ideal LLR statistic, plus the direct exp(LLR))
# ----------------------------------------------------------------------------------------------
print("\n== 7. single-event LR table ==")
lr_table = []
for key, res in RESULTS.items():
    if not key.endswith("|540"):
        continue
    r = res["ideal_llr"]; d = r["d"]
    lr_table.append(dict(pair=key, d_ideal=d, LR_NR_median_direct=r["LR_at_NR_median"], LR_ER_median_direct=r["LR_at_ER_median"],
                         LR_midway_direct=r["LR_midway"], LR_NR_median_gauss=float(np.exp(d**2 / 2)), LR_ER_median_gauss=float(np.exp(-d**2 / 2)),
                         leak50=r["leak50"], leak90=r["leak90"], d_fp20=res["fp20"]["d"], d_tmean=res["tmean"]["d"], d_tmean_dig=res["tmean_dig"]["d"]))
    print(f"{key:28s} d={d:5.2f}  LR(NR med)={r['LR_at_NR_median']:8.2f}  LR(ER med)={r['LR_at_ER_median']:.3g}  LR(mid)={r['LR_midway']:.2f}  leak50={r['leak50']:.3f}")
for dd in (0.5, 1.0, 1.5, 2.0, 3.0):
    from math import erfc
    print(f"Gaussian d={dd}: LR at NR median = {np.exp(dd**2/2):.2f}, at ER median = {np.exp(-dd**2/2):.3f}, midway = 1; leak50 = {0.5*erfc(dd/np.sqrt(2)):.3f}")

# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
C_NR, C_ER, C_ZF, C_ACC = "#1f6f8b", "#c0392b", "#7d3c98", "#7f8c8d"
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, tag, ttl in ((axs[0], "540", "540 phd (event), f_p(20 ns)"), (axs[1], "AmBe", f"{N_DET_AMBE} phd (150 keV NR), f_p(20 ns)")):
    bins = np.linspace(0.05, 0.6, 111)
    ax.hist(STATS[("ER_base", tag)]["fp20"], bins, histtype="step", color=C_ER, lw=1.6, label="ER (drift field, I_s/I_t = 0.5)")
    ax.hist(STATS[("NR_base", tag)]["fp20"], bins, histtype="step", color=C_NR, lw=1.6, label="NR (I_s/I_t = 1.5)")
    ax.hist(STATS[("ER_zeroF", tag)]["fp20"], bins, histtype="step", color=C_ZF, lw=1.2, ls="--", label="ER zero-field (45 ns recomb.)")
    ax.hist(STATS[("NR_lo", tag)]["fp20"], bins, histtype="step", color=C_NR, lw=1.0, ls=":", label="NR slowest (I_s/I_t = 1, 27 ns)")
    ax.hist(STATS[("ER_fast", tag)]["fp20"], bins, histtype="step", color=C_ER, lw=1.0, ls=":", label="ER fastest (0.6, 24 ns)")
    ax.set_xlabel("prompt fraction f_p (0-20 ns after first photon)"); ax.set_ylabel("pulses / bin"); ax.set_title(ttl, fontsize=10)
axs[0].legend(fontsize=7, frameon=False)
ax = axs[2]
lnr, ler = LLR[("NR_base", "ER_base", "540")]
bins = np.linspace(-60, 60, 80)
ax.hist(ler, bins, histtype="step", color=C_ER, lw=1.6, label="ER pulses")
ax.hist(lnr, bins, histtype="step", color=C_NR, lw=1.6, label="NR pulses")
lnr2, ler2 = LLR[("NR_lo", "ER_fast", "540")]
ax.hist(ler2, bins, histtype="step", color=C_ER, lw=1.0, ls=":"); ax.hist(lnr2, bins, histtype="step", color=C_NR, lw=1.0, ls=":", label="closest bracket pair (dotted)")
ax.axvline(0, color=C_ACC, lw=0.8)
ax.set_xlabel("ideal template ln LR (NR/ER), 540 phd"); ax.set_title("full-template likelihood ratio", fontsize=10); ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P077_fig1_fp_distributions.png"), dpi=150); plt.close(fig)

# ROC
fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
ax = axs[0]
for (nr, er), col, ls, lab in (( ("NR_base", "ER_base"), "k", "-", "bracket centre"), (("NR_lo", "ER_fast"), C_NR, "--", "closest pair"),
                               (("NR_hi", "ER_slow"), C_ER, "--", "farthest pair"), (("NR_base", "ER_zeroF"), C_ZF, ":", "vs zero-field ER")):
    for tag, lw in (("540", 1.8), ("AmBe", 0.9)):
        a, b = LLR[(nr, er, tag)]
        thr = np.sort(np.concatenate([a, b]))[::20]
        acc = [(a > t).mean() for t in thr]; leak = [(b > t).mean() for t in thr]
        ax.plot(leak, acc, color=col, ls=ls, lw=lw, label=f"{lab} ({tag})" if True else None)
ax.plot([0, 1], [0, 1], color=C_ACC, lw=0.6)
ax.set_xscale("log"); ax.set_xlim(1e-3, 1); ax.set_xlabel("ER leakage (misidentified as NR)"); ax.set_ylabel("NR acceptance")
ax.set_title("ROC of the ideal template LR (thick 540 phd, thin AmBe)", fontsize=9); ax.legend(fontsize=6, frameon=False, ncol=2)
ax = axs[1]
ax.plot(DELTAS, [r["d_llr_540"] for r in scan_rows], "o-", color="k", label="540 phd, ideal LR")
ax.plot(DELTAS, [r["d_fp20_540"] for r in scan_rows], "s--", color="k", mfc="none", label="540 phd, f_p(20 ns)")
ax.plot(DELTAS, [r["d_llr_AmBe"] for r in scan_rows], "o-", color=C_NR, label=f"{N_DET_AMBE} phd (150 keV NR), ideal LR")
ax.plot(DELTAS, [r["d_llr_100"] for r in scan_rows], "o-", color=C_ACC, label="100 phd, ideal LR")
for n_amb, a in allowed.items():
    ax.axvline(a["delta_ns_max"], color=C_ER, lw=0.8, ls=":" if n_amb != 10 else "-")
ax.text(allowed[10]["delta_ns_max"] * 1.05, 6, "allowed by 'AmBe not\ninconsistent' (n=5,10,30)", color=C_ER, fontsize=7)
ax.set_xlabel("NR - ER difference in mean emission time  (ns)"); ax.set_ylabel("separation d (sigma)"); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_title("separation vs intrinsic timing difference", fontsize=9); ax.legend(fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P077_fig2_roc_and_scan.png"), dpi=150); plt.close(fig)

# depth dependence
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
zs = np.array([r["z_cm"] for r in rows]);
axs[0].plot(zs, [r["t_mean_ns"] for r in rows], "o-", color="k", label="<t_prop> all")
axs[0].plot(zs, [r["t_mean_bottom"] for r in rows], "s--", color=C_NR, label="bottom array")
axs[0].plot(zs, [r["t_mean_top"] for r in rows], "^--", color=C_ER, label="top array")
axs[0].plot(zs, [r["t_sd_ns"] for r in rows], "d:", color=C_ACC, label="sd(t_prop)")
axs[0].axvline(Z_EVENT, color=C_ACC, lw=0.7); axs[0].axvspan(-13.75, 0, color=C_ACC, alpha=0.15)
axs[0].set_xlabel("z above cathode (cm)  [shaded: RFR]"); axs[0].set_ylabel("ns"); axs[0].set_title("propagation time (optical MC)", fontsize=9); axs[0].legend(fontsize=7, frameon=False)
axs[1].plot(zs, [r["TBA"] for r in rows], "o-", color="k"); axs[1].axvline(Z_EVENT, color=C_ACC, lw=0.7); axs[1].axvspan(-13.75, 0, color=C_ACC, alpha=0.15)
axs[1].set_xlabel("z above cathode (cm)"); axs[1].set_ylabel("top-bottom asymmetry (T-B)/(T+B)"); axs[1].set_title(f"TBA vs depth; sigma(TBA) at 540 phd = {sig_tba:.3f}", fontsize=9)
ax = axs[2]
ax.errorbar(zz, tmE, yerr=sdE, fmt="o-", color=C_ER, label="ER <t> - t_first (540 phd) +- 1 sd")
ax.plot(zz, [r["tmean_NR"] for r in pos_rows], "s-", color=C_NR, label="NR")
ax.axvline(Z_EVENT, color=C_ACC, lw=0.7)
ax.set_xlabel("z above cathode (cm)"); ax.set_ylabel("mean arrival time after first photon (ns)"); ax.set_title("statistic vs depth (position dependence)", fontsize=9); ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P077_fig3_depth_dependence.png"), dpi=150); plt.close(fig)

# rise time RFR vs FV
fig, ax = plt.subplots(figsize=(5, 3.6))
for z, col, lab in ((-7.0, C_ACC, "RFR origin, z = -7 cm"), (Z_EVENT, C_NR, "event depth, z = 26.4 cm"), (100.0, C_ER, "z = 100 cm")):
    st, _ = simulate_pulses(MODELS["ER_base"], N_DET_EVENT, z, n_pulses=3000)
    ax.hist(st["rise_10_50"], np.linspace(5, 45, 81), histtype="step", lw=1.5, color=col, label=lab)
ax.set_xlabel("rise time 10% -> 50% of the S1 (ns), 540 detected photons"); ax.set_ylabel("pulses / bin"); ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P077_fig4_rise_time_rfr.png"), dpi=150); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# save
# ----------------------------------------------------------------------------------------------
summary = dict(N_det_event=N_DET_EVENT, N_det_AmBe=N_DET_AMBE, n_pulses=N_PULSES, z_event_cm=Z_EVENT,
               models={k: dict(v, mean_emission_ns=mean_emission(**v)) for k, v in MODELS.items()},
               optics_event=dict(t_mean=float(optics[Z_EVENT][0].mean()), t_sd=float(optics[Z_EVENT][0].std()), f_top=float(optics[Z_EVENT][1].mean()),
                                 variant_n1p9=dict(t_mean=float(tt_v1.mean()), t_sd=float(tt_v1.std())), variant_lamR30=dict(t_mean=float(tt_v2.mean()), t_sd=float(tt_v2.std()))),
               pair_results=RESULTS, optics_variants=OPT_VAR, delta_scan=scan_rows, allowed_by_LZ_statement=allowed, ratio_d540_dAmBe=float(ratio_540_amb),
               position_dependence=posdep, rfr=rfr, rfr_summary=rfr_summary, lr_table=lr_table, runtime_s=time.time() - T0)
def _conv(o):
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(str(type(o)))
with open(os.path.join(OUT, "P077_results.json"), "w") as f:
    json.dump(summary, f, indent=1, default=_conv)
print(f"\ndone in {time.time()-T0:.0f} s")
