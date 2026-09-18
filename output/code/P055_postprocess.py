"""
P055 post-processing (foreground, light):
  (a) first-harmonic phase/amplitude fits of the composite and pure rate time series saved by
      P055_gaia_substructures.py (rate_timeseries.json), replacing the 12-day-grid argmax peak day;
  (b) the elastic O1 dR/dE(248 keV) time dependence and run-window date likelihood ratio for the SHM and the
      SHM + substructure composites (the one channel to which the June-phased Sgr(+W)/Helmi(+W) clumps can
      contribute), using ONE WimPyDD per-stream kernel row (E = 248 keV, delta = 0) and analytic speed densities
      for isotropic streams; the anisotropic S1 and shards use the numerical grid on a REDUCED angular grid
      (48 x 32 instead of 96 x 64) to keep the runtime at ~1 min.
Run from the simulation root:  .venv/bin/python output/code/P055_postprocess.py
"""
import sys, os, json, math, time, datetime as dt
import numpy as np, pandas as pd
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
from numpy.polynomial.legendre import leggauss
T0 = time.time()
OUT = 'output/work/P055'
V0, VESC = lz.V0_KMS, lz.VESC_KMS
V_SUN = np.array([0.0, V0, 0.0]) + lz.V_SUN_PEC; VS_HAT = V_SUN / np.linalg.norm(V_SUN)
V_ORB = 29.79; VGRID = np.linspace(0.0, 1100.0, 1101); DV = 1.0; MASS = 1000.0
EQ2GAL = np.array([[-0.0548755604, -0.8734370902, -0.4838350155], [0.4941094279, -0.4448296300, 0.7469822445], [-0.8676661490, -0.1980763734, 0.4559837762]])
OBLIQ = math.radians(23.4393); YEAR = 2023
def doy_of(date): return (date - dt.datetime(YEAR, 1, 1)).total_seconds() / 86400.0 + 1.0
T_EQUINOX = doy_of(dt.datetime(2023, 3, 20, 21, 24)); T_EVENT = doy_of(dt.datetime(2023, 6, 16, 21, 22, 39)); OMEGA = 2 * math.pi / 365.25
def v_obs_vec(doy):
    lam = OMEGA * (doy - T_EQUINOX); v = V_ORB * np.array([math.sin(lam), -math.cos(lam), 0.0])
    x, y, z = v; v_eq = np.array([x, y * math.cos(OBLIQ) - z * math.sin(OBLIQ), y * math.sin(OBLIQ) + z * math.cos(OBLIQ)])
    return V_SUN + EQ2GAL @ v_eq
def cos_fit(t, y):
    X = np.column_stack([np.ones_like(t), np.cos(OMEGA * t), np.sin(OMEGA * t)])
    a0, a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(a0), float(math.hypot(a, b)), float((math.atan2(b, a) / OMEGA) % 365.25)

# ---------------- (a) rate phases from the saved time series ----------------
ts = json.load(open(f'{OUT}/rate_timeseries.json')); DOYS = np.array(ts['doy'])
rows = []
for k, v in ts.items():
    if k == 'doy': continue
    y = np.array(v)
    if y.mean() <= 0: continue
    a0, A, pk = cos_fit(DOYS, y / y.mean())
    rows.append(dict(series=k, a1=A, peak_doy_fit=pk, peak_doy_grid=float(DOYS[np.argmax(y)])))
dfa = pd.DataFrame(rows); dfa.to_csv(f'{OUT}/rate_phase_fits.csv', index=False, float_format='%.5g')
print(dfa[dfa.series.str.contains('SHM|GSE_comp|S1_comp|Vesc_retro_comp|Vesc_retro_pure')].to_string(index=False))

# ---------------- (b) elastic 248 keV: kernel row, speed densities, LR ----------------
WD = lz.wd()
GF, SW2 = 1.166e-5, 0.231; c_p = (GF / math.sqrt(2)) * (1 - 4 * SW2); c_n = -GF / math.sqrt(2)
HAM = lz.wd_hamiltonian('higgsino_Z_P055pp', {1: (c_p + c_n, c_p - c_n)})
K248 = np.array(WD.diff_rate(WD.Xe, HAM, MASS, 248.0, VGRID, np.ones_like(VGRID), j_chi=0.5, delta=0.0, sum_over_streams=False)) * 1000 * 365.25
def eta_from_f(f, w=VGRID):
    g = np.where(w > 0, f / np.where(w > 0, w, 1.0), 0.0); seg = 0.5 * (g[1:] + g[:-1]) * np.diff(w)
    return np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
def deta_from_eta(eta):
    de = np.zeros_like(eta); de[1:] = eta[:-1] - eta[1:]; return np.clip(de, 0.0, None)
def f_iso(mu, sigma, v_obs, w=VGRID):
    vl = float(np.linalg.norm(np.asarray(mu, float) - v_obs))
    f = w / (math.sqrt(2 * math.pi) * sigma * vl) * (np.exp(-(w - vl)**2 / (2 * sigma**2)) - np.exp(-(w + vl)**2 / (2 * sigma**2)))
    return f / float(np.trapezoid(f, w))
NTH, NPH = 48, 32
_x, _w = leggauss(NTH); _ph = (np.arange(NPH) + 0.5) * 2 * math.pi / NPH
def f_num(mu, sig, v_obs, w=VGRID):
    mu = np.asarray(mu, float); sig = np.asarray(sig, float); axis = mu - v_obs; e3 = axis / np.linalg.norm(axis)
    tmp = np.array([1.0, 0, 0]) if abs(e3[0]) < 0.9 else np.array([0, 1.0, 0]); e1 = np.cross(e3, tmp); e1 /= np.linalg.norm(e1); e2 = np.cross(e3, e1)
    ct = np.repeat(_x, NPH); st = np.sqrt(1 - ct**2); ph = np.tile(_ph, NTH)
    n = ct[:, None] * e3 + st[:, None] * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2); wgt = np.repeat(_w, NPH) * (2 * math.pi / NPH)
    vl = float(np.linalg.norm(axis)); sel = (w >= max(0, vl - 7 * sig.max())) & (w <= min(w[-1], vl + 7 * sig.max(), VESC + np.linalg.norm(v_obs) + 1))
    f = np.zeros_like(w); idx = np.where(sel)[0]
    for i0 in idx[::64]:
        i1 = min(i0 + 64, idx[-1] + 1); ww = w[i0:i1]
        U = ww[:, None, None] * n[None] + v_obs[None, None, :]
        g = np.exp(-0.5 * np.sum(((U - mu) / sig)**2, axis=-1)) * (np.sum(U**2, axis=-1) <= VESC**2)
        f[i0:i1] = ww**2 * (g @ wgt)
    return f / float(np.trapezoid(f, w))
CAT = [dict(key='Sgr_m', mu=(0, 0, -300), sig=(30, 30, 30), f=0.03), dict(key='Sgr_p', mu=(0, 0, 300), sig=(30, 30, 30), f=0.03),
       dict(key='S1', mu=(30, -297, -73), sig=(83, 27, 59), f=0.10), dict(key='S2', mu=(6, 164, -250), sig=(30, 20, 40), f=0.01),
       dict(key='Helmi_p', mu=(0, 150, 250), sig=(30, 30, 30), f=0.005), dict(key='Helmi_m', mu=(0, 150, -250), sig=(30, 30, 30), f=0.005),
       dict(key='Shards', mu=(0, -290, 0), sig=(60, 30, 60), f=0.01)]
d_start = dt.datetime(2023, 3, 27); t_run = np.linspace(0.0, 371.0, 371 * 8 + 1); doy_run = (doy_of(d_start) - 1.0 + t_run) % 365.25 + 1.0
def run_mean(vals):
    dd = np.concatenate([DOYS - 365.25, DOYS, DOYS + 365.25]); vv = np.concatenate([vals, vals, vals]); return float(np.mean(np.interp(doy_run, dd, vv)))
VOBS = np.array([v_obs_vec(d) for d in DOYS]); VE = np.linalg.norm(VOBS, axis=1); iJ = int(np.argmin(np.abs(DOYS - T_EVENT))); iDec = int(np.argmin(np.abs(DOYS - 336.0)))
rs = np.array([K248 @ deta_from_eta(np.asarray(lz.eta0(VGRID, v_e=v))) for v in VE])
a0, A, pk = cos_fit(DOYS, rs / rs.mean()); lr_s = rs[iJ] / run_mean(rs)
el = [dict(key='SHM', f=0.0, dRdE_248_June16=rs[iJ], a1_fit=A, peak_doy_fit=pk, June_Dec=rs[iJ] / rs[iDec], LR_16June=lr_s, LR_ratio_to_SHM=1.0)]
print('[%.0fs] SHM elastic 248 keV: a1 = %.4f, peak doy %.1f, LR = %.4f' % (time.time() - T0, A, pk, lr_s))
for c in CAT:
    sig = np.array(c['sig'], float)
    if np.ptp(sig) < 1e-9:
        rp = np.array([K248 @ deta_from_eta(eta_from_f(f_iso(c['mu'], float(sig[0]), VOBS[i]))) for i in range(len(DOYS))])
    else:
        rp = np.array([K248 @ deta_from_eta(eta_from_f(f_num(c['mu'], sig, VOBS[i]))) for i in range(len(DOYS))])
    f = c['f']; rc = (1 - f) * rs + f * rp
    a0c, Ac, pkc = cos_fit(DOYS, rc / rc.mean()); lr_c = rc[iJ] / run_mean(rc)
    if rp.mean() > 0:
        a0p, Ap, pkp = cos_fit(DOYS, rp / rp.mean())
    else:
        Ap, pkp = float('nan'), float('nan')
    el.append(dict(key=c['key'], f=f, dRdE_248_June16=rc[iJ], ratio_to_SHM_June16=rc[iJ] / rs[iJ], pure_over_SHM_June16=rp[iJ] / rs[iJ], a1_fit=Ac, peak_doy_fit=pkc,
                   June_Dec=rc[iJ] / rc[iDec], LR_16June=lr_c, LR_ratio_to_SHM=lr_c / lr_s, pure_a1_fit=Ap, pure_peak_doy_fit=pkp,
                   pure_June_Dec=float(rp[iJ] / rp[iDec]) if rp[iDec] > 0 else float('inf'), pure_max_over_min=float(rp.max() / rp.min()) if rp.min() > 0 else float('inf')))
    print('[%.0fs] %-8s ratio %.4f  pure/SHM %.3f  a1 %.4f peak %.1f  LR %.4f (x%.4f)  pure: a1 %.3f peak %.1f' % (time.time() - T0, c['key'], el[-1]['ratio_to_SHM_June16'], el[-1]['pure_over_SHM_June16'], Ac, pkc, lr_c, lr_c / lr_s, Ap, pkp))
pd.DataFrame(el).to_csv(f'{OUT}/elastic_248_dateLR.csv', index=False, float_format='%.5g')
json.dump(dict(runtime_s=time.time() - T0, angular_grid_reduced=[NTH, NPH], note='S1 and shards on a 48x32 angular grid; isotropic streams analytic'), open(f'{OUT}/postprocess_meta.json', 'w'), indent=1)
print('done %.0fs' % (time.time() - T0))
