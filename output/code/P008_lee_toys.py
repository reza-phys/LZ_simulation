"""
P008_lee_toys.py -- an independent look-elsewhere (trials-factor) estimate for the LZ 248 keV event.

Run from the simulation root:
    .venv/bin/python output/code/P008_lee_toys.py            # full run (200k toys + robustness variants)
    N_TOYS=40000 .venv/bin/python output/code/P008_lee_toys.py   # quick run

Outputs: output/work/P008/*.csv, *.json, spectra_cache.npz and output/work/P008/figures/*.png

Method (details in output/work/P008/details.md):
 1. Simplified one-dimensional extended likelihood in *observed* NR-equivalent recoil energy E on
    [5.4, 270] keV for the LZ science sample restricted to the NR band.  Background = accidentals
    (2.7, falling exponential), atmospheric-nu CEvNS (0.11, computed), 8B (0.057, below threshold),
    MSSI (0.0049, flat 50-270 keV), detector neutrons (0.05, (alpha,n)-like exponential).
    ER leakage inside the NR band above 100 keV is neglected (LZ Fig. 5: the ER tail sits at +1..+5 sigma).
 2. Signal shapes from WimPyDD (shell-model nuclear responses): O1,O3..O15 x {isoscalar, isovector}
    x 13 masses; inelastic O1/O4 x {s,v} x {400,1000,4000} GeV x 8 mass splittings; plus a fine
    delta scan (O1s, 1000 GeV) for the continuous-parameter (Gross-Vitells) comparison.
    All spectra smeared with sigma(E) = 23 keV x sqrt(E/248 keV) and multiplied by the LZ efficiency.
 3. Background-only toys (Poisson), profile-likelihood q0 for every model with the signal strength
    fitted (mu >= 0), toy-calibrated local p-values, min-p over model subspaces -> global p, N_eff.
 4. Model-space dependence, Bonferroni and Gross-Vitells comparisons, data local Z per model class.
Tools: python 3.12.13, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, WimPyDD 2.0.4 (via lzcommon
wd_hamiltonian / wd_rate / wd_halo), pandas 3.0.5, common/lzcommon.py.
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "output/code")
import numpy as np
import pandas as pd
from scipy import stats, special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import lzcommon as lz

WORK = "output/work/P008"
FIG = os.path.join(WORK, "figures")
os.makedirs(FIG, exist_ok=True)
RNG = np.random.default_rng(20260904)
N_TOYS = int(os.environ.get("N_TOYS", "200000"))      # total; half calibration, half evaluation
N_TOYS_VAR = int(os.environ.get("N_TOYS_VAR", str(max(20000, N_TOYS // 4))))
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

# ----------------------------------------------------------------------------------------------
# 1. Analysis grid, efficiency, resolution
# ----------------------------------------------------------------------------------------------
E_LO, E_HI = lz.LZ["E_50pct_low_keV"], 270.0          # 5.4 .. 270 keV (paper: 50% eff. points 5.4 / 269.9)
DE = 0.5
E_OBS = np.arange(E_LO, E_HI + 1e-9, DE)               # observed-energy grid (bin centres used as points)
E_TRUE = np.arange(0.5, 420.0, DE)                     # true-energy grid for smearing
SIG_248 = lz.LZ["ev_E_stat"]                           # 23 keV statistical resolution at 248 keV (paper)

def efficiency(E):
    """NR efficiency vs observed energy: 50% at 5.4 and 269.9 keV, plateau 0.96 (paper Fig. S2 caption)."""
    lo = special.expit((E - 5.4) / 1.5)
    hi = 1.0 - special.expit((E - 269.9) / 6.0)
    return 0.96 * lo * hi

def sigma_E(E):
    """Gaussian energy resolution, sigma = 23 keV at 248 keV, scaling as sqrt(E) (Poisson-like)."""
    return SIG_248 * np.sqrt(np.maximum(E, 0.1) / 248.0)

# smearing matrix  K[o, t] = N(E_o; E_t, sigma(E_t)) * dE_t   (true -> observed)
_K = stats.norm.pdf(E_OBS[:, None], loc=E_TRUE[None, :], scale=sigma_E(E_TRUE)[None, :]) * DE
EFF_OBS = efficiency(E_OBS)

def observed_pdf(rate_true):
    """Smear a true-energy spectrum (on E_TRUE) and apply efficiency; return unnormalised density on E_OBS."""
    return (_K @ rate_true) * EFF_OBS

def normalise(d):
    s = d.sum() * DE
    return d / s if s > 0 else d

# ----------------------------------------------------------------------------------------------
# 2. Background model (Table I of the paper, projected onto E inside the NR band)
# ----------------------------------------------------------------------------------------------
M_XE_MEV = lz.A_XE_MEAN * lz.AMU_GEV * 1e3   # ~122,300 MeV
SIN2W = 0.238                                # low-energy weak mixing angle (recalled, likely)
Z_XE, N_XE = 54, lz.A_XE_MEAN - 54

def atm_nu_flux_shape(Enu):
    """Shape of the total atmospheric-nu flux (all flavours) vs E_nu [MeV], arbitrary normalisation.
    Recalled (uncertain): dPhi/dE rises roughly linearly up to ~100 MeV, is flat to ~300 MeV,
    and falls ~E^-3 above (Battistoni et al. 2005 low-energy flux as used by Billard et al. 2014)."""
    Enu = np.asarray(Enu, float)
    return np.where(Enu < 100, Enu / 100.0, np.where(Enu < 300, 1.0, (300.0 / Enu) ** 3))

def atm_nu_recoil_true(E_keV, flux_index_low=1.0):
    """CEvNS recoil spectrum shape on Xe: dR/dE ∝ F^2(E) ∫_{Emin} Phi(Enu) (1 - M E/(2 Enu^2)) dEnu,
    Emin = sqrt(M E / 2).  Cross-section formula recalled (certain).  Q_w constant -> shape only."""
    out = np.zeros_like(E_keV)
    Enu = np.geomspace(5.0, 5000.0, 1500)
    phi = np.where(Enu < 100, (Enu / 100.0) ** flux_index_low, np.where(Enu < 300, 1.0, (300.0 / Enu) ** 3))
    for i, E in enumerate(E_keV):
        Emin = np.sqrt(M_XE_MEV * E * 1e-3 / 2.0)
        m = Enu > Emin
        integrand = phi[m] * (1.0 - M_XE_MEV * E * 1e-3 / (2.0 * Enu[m] ** 2))
        out[i] = np.trapezoid(integrand, Enu[m]) * lz.helm_F2(E, lz.A_XE_MEAN)
    return out

def build_background(tau_acc=15.0, tau_n=40.0, n_neutron=0.05, mssi_scale=1.0, atm_index=1.0):
    """Return dict name -> (expected counts in ROI, normalised observed-energy density on E_OBS)."""
    comps = {}
    # accidentals: falling exponential in observed energy (assumption; isolated-S1 spectrum falls steeply)
    comps["accidentals"] = (lz.LZ["bkg_expected"]["accidentals"][0], normalise(np.exp(-E_OBS / tau_acc)))
    # atmospheric-nu CEvNS: computed shape, smeared, efficiency
    comps["atm_nu"] = (lz.LZ["bkg_expected"]["atm_nu"][0], normalise(observed_pdf(atm_nu_recoil_true(E_TRUE, atm_index))))
    # 8B + hep: essentially below threshold; steep exponential at the lower edge (negligible above 8 keV)
    comps["B8_hep"] = (lz.LZ["bkg_expected"]["B8_hep"][0], normalise(np.exp(-(E_OBS - E_LO) / 1.0)))
    # MSSI: flat in NR-equivalent energy from ~50 keV (12 keV first scatter + 471 phd S1-only) to 270 keV
    comps["MSSI"] = (lz.LZ["bkg_expected"]["MSSI"][0] * mssi_scale, normalise(special.expit((E_OBS - 50.0) / 10.0)))
    # detector (alpha,n) neutrons: exponential true spectrum with scale tau_n (assumption), smeared, efficiency
    comps["neutrons"] = (n_neutron, normalise(observed_pdf(np.exp(-E_TRUE / tau_n))))
    return comps

def total_background(comps):
    B = sum(c[0] for c in comps.values())
    b = sum(c[0] * c[1] for c in comps.values())     # density (events / keV) on E_OBS
    return B, b

# ----------------------------------------------------------------------------------------------
# 3. Signal spectra from WimPyDD (cached)
# ----------------------------------------------------------------------------------------------
E_WD = np.geomspace(2.0, 400.0, 60)
MASSES = lz.LSIG_MASSES                                       # 13 masses, Table S6
OPS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
INEL_MASSES, INEL_DELTAS = [400, 1000, 4000], lz.OSIG_DELTAS
FINE_DELTAS = list(range(0, 401, 10))

def model_list():
    models = []
    for op in OPS:
        for tau in ("s", "v"):
            for m in MASSES:
                models.append(dict(name=f"O{op}{tau}_m{m}", cls="elastic", op=op, tau=tau, mass=m, delta=0.0))
    for op in (1, 4):
        for tau in ("s", "v"):
            for m in INEL_MASSES:
                for d in INEL_DELTAS:
                    models.append(dict(name=f"O{op}{tau}_m{m}_d{d}", cls="inelastic", op=op, tau=tau, mass=m, delta=float(d)))
    for d in FINE_DELTAS:
        models.append(dict(name=f"O1s_m1000_dfine{d}", cls="deltascan", op=1, tau="s", mass=1000, delta=float(d)))
    return models

def compute_spectra(models):
    cache = os.path.join(WORK, "spectra_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z["names"]) == [m["name"] for m in models]:
            log("loaded cached WimPyDD spectra", cache)
            return z["rates"]
    halo = lz.wd_halo()                                           # time-averaged Baxter-2021 SHM
    hams = {}
    rates = np.zeros((len(models), len(E_WD)))
    for i, m in enumerate(models):
        key = (m["op"], m["tau"])
        if key not in hams:
            c = (1e-3, 0.0) if m["tau"] == "s" else (0.0, 1e-3)
            hams[key] = lz.wd_hamiltonian(f"O{m['op']}{m['tau']}", {m["op"]: c})
        rates[i] = lz.wd_rate(hams[key], float(m["mass"]), E_WD, halo=halo, delta_kev=m["delta"])
        if i % 50 == 0:
            log(f"  WimPyDD spectrum {i}/{len(models)}: {m['name']}")
    np.savez(cache, names=np.array([m["name"] for m in models]), rates=rates, E_WD=E_WD)
    return rates

def signal_pdfs(models, rates):
    """Interpolate WimPyDD rates onto E_TRUE, smear, apply efficiency, normalise. Returns (F, keep, meta)."""
    F, keep, frac200, frac_roi = [], [], [], []
    for m, r in zip(models, rates):
        rt = np.interp(E_TRUE, E_WD, np.maximum(r, 0.0), left=0.0, right=0.0)
        # below the first WimPyDD grid point (2 keV) extrapolate flat (only matters for m <= 12 GeV shapes)
        rt[E_TRUE < E_WD[0]] = max(r[0], 0.0)
        tot_true = rt.sum() * DE
        d = observed_pdf(rt)
        s = d.sum() * DE
        ok = (tot_true > 0) and (s > 0) and (s / tot_true > 1e-6)
        keep.append(ok)
        F.append(d / s if ok else np.zeros_like(d))
        frac200.append((d[E_OBS > 200].sum() * DE / s) if ok else np.nan)
        frac_roi.append((s / tot_true) if tot_true > 0 else 0.0)
    return np.array(F), np.array(keep), np.array(frac200), np.array(frac_roi)

# ----------------------------------------------------------------------------------------------
# 4. Profile-likelihood test statistic q0, vectorised over toys x models
# ----------------------------------------------------------------------------------------------
def q0_for_events(E_events, F, b_dens, nmax_bisect=36):
    """E_events: (n_toys, n_pad) observed energies, NaN-padded.  F: (n_models, n_E) signal densities on E_OBS
    (integrate to 1).  b_dens: (n_E,) background density (integrates to B).  Returns q0 (n_toys, n_models)
    for  ln L(s) = -(B+s) + sum_i ln(b(E_i) + s f_m(E_i)),  s >= 0 (background normalisation fixed)."""
    idx = np.clip(np.round((np.nan_to_num(E_events, nan=E_LO) - E_LO) / DE).astype(int), 0, len(E_OBS) - 1)
    valid = np.isfinite(E_events)                                   # (T, P)
    b_i = b_dens[idx]                                               # (T, P)
    r = F[:, idx].transpose(1, 0, 2) / b_i[:, None, :]              # (T, M, P) = f/b
    r = np.where(valid[:, None, :], r, 0.0)
    g0 = r.sum(axis=2)                                              # dlnL/ds at s=0, +1
    has_root = g0 > 1.0
    n = valid.sum(axis=1)                                           # events per toy
    lo = np.zeros_like(g0)
    hi = np.broadcast_to(n[:, None].astype(float) + 1.0, g0.shape).copy()   # s_hat < n
    for _ in range(nmax_bisect):
        mid = 0.5 * (lo + hi)
        g = (r / (1.0 + mid[:, :, None] * r)).sum(axis=2)
        up = g > 1.0
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    s_hat = np.where(has_root, 0.5 * (lo + hi), 0.0)
    q0 = 2.0 * (-s_hat + np.log1p(s_hat[:, :, None] * r).sum(axis=2))
    return np.maximum(q0, 0.0).astype(np.float32), s_hat

def generate_toys(n_toys, B, b_dens, rng):
    n = rng.poisson(B, size=n_toys)
    pad = int(max(n.max(), 1))
    cdf = np.cumsum(b_dens) * DE
    cdf /= cdf[-1]
    E = np.full((n_toys, pad), np.nan)
    for k in range(1, pad + 1):
        rows = np.where(n >= k)[0]
        u = rng.random(len(rows))
        E[rows, k - 1] = np.interp(u, cdf, E_OBS)
    return E, n

def run_toys(n_toys, F, b_dens, B, rng, chunk=2500):
    Ev, n = generate_toys(n_toys, B, b_dens, rng)
    q = np.zeros((n_toys, F.shape[0]), dtype=np.float32)
    for a in range(0, n_toys, chunk):
        q[a:a + chunk], _ = q0_for_events(Ev[a:a + chunk], F, b_dens)
    return q, n

class Calibrator:
    """Toy-calibrated local p-values per model: p_m(q) = P(q0_m >= q | H0), from a calibration set."""
    def __init__(self, q_cal):
        self.sorted = np.sort(q_cal, axis=0)                     # (Ncal, M)
        self.N = q_cal.shape[0]
    def p(self, q):
        """q: (T, M) -> p (T, M).  p = (# cal >= q)/N; q<=0 -> 1.  Floor at 0.5/N."""
        T, M = q.shape
        out = np.empty((T, M))
        for m in range(M):
            k = np.searchsorted(self.sorted[:, m], q[:, m], side="left")   # cal values < q
            out[:, m] = (self.N - k) / self.N
        out = np.where(q <= 0, 1.0, out)
        return np.maximum(out, 0.5 / self.N)
    def q_threshold(self, p):
        """Per-model q0 threshold such that P(q0 >= q_thr) ~ p (empirical quantile)."""
        k = int(np.floor((1 - p) * self.N))
        return self.sorted[min(k, self.N - 1), :]

def neff_from_pglobal(p_glob, p_loc):
    return np.log(1 - p_glob) / np.log(1 - p_loc) if 0 < p_glob < 1 else np.nan

def global_stats(p_eval, subset, p_loc_list):
    """p_eval: (T, M) local p per eval toy.  subset: boolean mask over models.  Returns dict per p_loc."""
    pmin = p_eval[:, subset].min(axis=1)
    T = len(pmin)
    res = {}
    for p_loc in p_loc_list:
        k = int((pmin <= p_loc).sum())
        pg = k / T
        err = np.sqrt(max(k, 1)) / T
        res[p_loc] = dict(p_global=pg, p_global_err=err, n_exceed=k,
                          Z_global=float(lz.p_to_sigma(pg)) if pg > 0 else np.inf,
                          N_eff=neff_from_pglobal(pg, p_loc),
                          N_eff_err=(neff_from_pglobal(min(pg + err, 0.999), p_loc) - neff_from_pglobal(max(pg - err, 1e-9), p_loc)) / 2 if k > 0 else np.nan)
    return res, pmin

# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def main():
    out = {}
    P_LOC_LZ = float(lz.sigma_to_p(3.4))            # 3.37e-4
    P_LOC_LIST = [float(lz.sigma_to_p(z)) for z in (2.0, 2.5, 3.0, 3.4)]
    out["p_local_LZ_3.4sigma"] = P_LOC_LZ
    out["N_eff_implied_by_LZ_3.4_to_2.6"] = neff_from_pglobal(float(lz.sigma_to_p(2.6)), P_LOC_LZ)

    # --- background
    comps = build_background()
    B, b_dens = total_background(comps)
    out["background_total_ROI"] = B
    bt = []
    for k, (mu, pdf) in comps.items():
        bt.append(dict(component=k, expected=mu, frac_above_100keV=float(pdf[E_OBS > 100].sum() * DE),
                       frac_above_200keV=float(pdf[E_OBS > 200].sum() * DE), density_at_248_per_keV=float(mu * pdf[np.argmin(abs(E_OBS - 248))])))
    bt.append(dict(component="TOTAL", expected=B, frac_above_100keV=float(b_dens[E_OBS > 100].sum() * DE / B),
                   frac_above_200keV=float(b_dens[E_OBS > 200].sum() * DE / B), density_at_248_per_keV=float(b_dens[np.argmin(abs(E_OBS - 248))])))
    pd.DataFrame(bt).to_csv(os.path.join(WORK, "background_components.csv"), index=False)
    out["background_expected_above_200keV"] = float(b_dens[E_OBS > 200].sum() * DE)
    out["background_expected_200_to_270keV_within_pm23"] = float(b_dens[(E_OBS > 225) & (E_OBS < 271)].sum() * DE)
    out["background_density_at_248_per_keV"] = float(b_dens[np.argmin(abs(E_OBS - 248))])
    log("background total", B, "above 200 keV", out["background_expected_above_200keV"])

    # --- signals
    models = model_list()
    rates = compute_spectra(models)
    F, keep, frac200, frac_roi = signal_pdfs(models, rates)
    for m, k, f2, fr in zip(models, keep, frac200, frac_roi):
        m["physical"] = bool(k); m["frac_above_200keV"] = float(f2); m["frac_in_ROI_of_true_rate"] = float(fr)
    dfm = pd.DataFrame(models)
    # --- spectral degeneracy (max CDF distance) for Bonferroni-type counting
    C = np.cumsum(F, axis=1) * DE
    def distinct_count(mask, thr):
        idx = np.where(mask & keep)[0]
        reps = []
        for i in idx:
            if all(np.max(np.abs(C[i] - C[j])) > thr for j in reps):
                reps.append(i)
        return len(reps), reps
    dfm["distinct_rep_1pct"] = False
    n_dist, reps = distinct_count(dfm["cls"].isin(["elastic", "inelastic"]).values, 0.01)
    dfm.loc[reps, "distinct_rep_1pct"] = True
    out["n_models_LZlike_raw"] = int((dfm["cls"].isin(["elastic", "inelastic"])).sum())
    out["n_models_LZlike_physical"] = int((dfm["cls"].isin(["elastic", "inelastic"]) & keep).sum())
    out["n_models_LZlike_distinct_1pct"] = n_dist
    out["n_models_LZlike_distinct_2pct"] = distinct_count(dfm["cls"].isin(["elastic", "inelastic"]).values, 0.02)[0]
    out["n_models_LZlike_distinct_0.5pct"] = distinct_count(dfm["cls"].isin(["elastic", "inelastic"]).values, 0.005)[0]
    # mass degeneracy check 400 vs 1000 vs 4000 GeV
    dd = {}
    for op in (1, 4, 6, 10, 11):
        i4 = dfm.index[(dfm.name == f"O{op}s_m400")][0]; i1 = dfm.index[(dfm.name == f"O{op}s_m1000")][0]; i40 = dfm.index[(dfm.name == f"O{op}s_m4000")][0]
        i2 = dfm.index[(dfm.name == f"O{op}s_m200")][0]; i100 = dfm.index[(dfm.name == f"O{op}s_m100")][0]
        dd[f"O{op}s"] = dict(d_400_1000=float(np.max(abs(C[i4] - C[i1]))), d_1000_4000=float(np.max(abs(C[i1] - C[i40]))),
                            d_200_400=float(np.max(abs(C[i2] - C[i4]))), d_100_200=float(np.max(abs(C[i100] - C[i2]))),
                            d_s_v_1000=float(np.max(abs(C[i1] - C[dfm.index[(dfm.name == f"O{op}v_m1000")][0]]))))
    out["cdf_distance_checks"] = dd
    dfm.to_csv(os.path.join(WORK, "models.csv"), index=False)
    log("models:", out["n_models_LZlike_raw"], "physical", out["n_models_LZlike_physical"], "distinct(1%)", n_dist)

    Fk = F[keep]
    names_k = dfm.name.values[keep]
    cls_k = dfm.cls.values[keep]
    mass_k = dfm.mass.values[keep]
    delta_k = dfm.delta.values[keep]
    op_k = dfm.op.values[keep]
    tau_k = dfm.tau.values[keep]
    M = Fk.shape[0]

    # --- toys
    Ncal = Nev = N_TOYS // 2
    log(f"generating {Ncal} calibration toys ...")
    q_cal, n_cal = run_toys(Ncal, Fk, b_dens, B, RNG)
    log(f"generating {Nev} evaluation toys ...")
    q_ev, n_ev = run_toys(Nev, Fk, b_dens, B, RNG)
    cal = Calibrator(q_cal)
    p_ev = cal.p(q_ev)
    out["toys"] = dict(N_cal=Ncal, N_eval=Nev, mean_events=float(n_ev.mean()), max_events=int(n_ev.max()),
                       frac_toys_with_event_above_200keV=float(np.mean(n_ev > 0) * 0 + 0))  # placeholder fixed below
    # fraction of toys with >=1 event above 200 keV = 1-exp(-B_>200)
    out["toys"]["frac_toys_with_event_above_200keV"] = float(1 - np.exp(-out["background_expected_above_200keV"]))

    # --- asymptotic check: P(q0 > 0) and P(q0 > 1.64^2) per class vs Wilks (0.5, 0.05)
    asym = []
    for cname in ("elastic", "inelastic"):
        for heavy in (False, True):
            sel = (cls_k == cname) & ((mass_k >= 100) == heavy)
            if sel.sum() == 0: continue
            asym.append({"cls": cname, "heavy": heavy, "n_models": int(sel.sum()),
                         "P_q0_gt0": float((q_cal[:, sel] > 0).mean()), "wilks_gt0": 0.5,
                         "P_q0_gt_2.71": float((q_cal[:, sel] > 2.71).mean()), "wilks_gt_2.71": 0.05,
                         "P_q0_gt_11.56": float((q_cal[:, sel] > 11.56).mean()), "wilks_gt_11.56_3.4sigma": P_LOC_LZ})
    pd.DataFrame(asym).to_csv(os.path.join(WORK, "asymptotic_check.csv"), index=False)

    # --- model subspaces
    sub = {}
    is_lz = np.isin(cls_k, ["elastic", "inelastic"])
    sub["(c) full LZ-like space (elastic 14 ops x s/v x 13 m + inelastic O1/O4 x s/v x 3 m x 8 delta)"] = is_lz
    sub["(a) inelastic O1/O4 s+v (96 raw)"] = cls_k == "inelastic"
    sub["(a') inelastic O1/O4 isoscalar only (48 raw)"] = (cls_k == "inelastic") & (tau_k == "s")
    sub["(b) elastic, m >= 100 GeV (all operators, s+v)"] = (cls_k == "elastic") & (mass_k >= 100)
    sub["(b') elastic, m >= 100 GeV, isoscalar only"] = (cls_k == "elastic") & (mass_k >= 100) & (tau_k == "s")
    sub["elastic, all masses"] = cls_k == "elastic"
    sub["elastic, m <= 50 GeV"] = (cls_k == "elastic") & (mass_k <= 50)
    sub["single model: O1s 1000 GeV (sanity, N_eff=1)"] = names_k == "O1s_m1000"
    sub["single model: O6s 1000 GeV"] = names_k == "O6s_m1000"
    sub["O1s elastic, 13 masses"] = (cls_k == "elastic") & (op_k == 1) & (tau_k == "s")
    sub["O1s+O1v elastic, 13 masses"] = (cls_k == "elastic") & (op_k == 1)
    sub["inelastic O1s 1000 GeV, LZ delta grid (8)"] = (cls_k == "inelastic") & (op_k == 1) & (tau_k == "s") & (mass_k == 1000)
    for dmax in (100, 200, 300, 350, 400):
        sub[f"(d) continuous delta scan O1s 1000 GeV, 10 keV steps, delta <= {dmax}"] = (cls_k == "deltascan") & (delta_k <= dmax)
    sub["(c)+(d) full space + fine delta scan"] = is_lz | (cls_k == "deltascan")
    rows = []
    pmins = {}
    for k, mask in sub.items():
        res, pmin = global_stats(p_ev, mask, P_LOC_LIST)
        pmins[k] = pmin
        n_raw = int(mask.sum())
        n_dist = distinct_count(np.isin(np.arange(len(dfm)), np.where(keep)[0][mask]), 0.01)[0]
        row = dict(subspace=k, n_models_raw=n_raw, n_models_distinct_1pct=n_dist)
        for z, p_loc in zip((2.0, 2.5, 3.0, 3.4), P_LOC_LIST):
            r = res[p_loc]
            row.update({f"p_global@{z}": r["p_global"], f"Z_global@{z}": r["Z_global"], f"N_eff@{z}": r["N_eff"], f"N_eff_err@{z}": r["N_eff_err"]})
        row["Bonferroni_Z_global@3.4_raw"] = float(lz.p_to_sigma(min(1.0, n_raw * P_LOC_LZ))) if n_raw * P_LOC_LZ < 1 else 0.0
        row["Bonferroni_Z_global@3.4_distinct"] = float(lz.p_to_sigma(min(1.0, n_dist * P_LOC_LZ))) if n_dist * P_LOC_LZ < 1 else 0.0
        rows.append(row)
    dfs = pd.DataFrame(rows)
    dfs.to_csv(os.path.join(WORK, "subspace_neff.csv"), index=False)
    main_key = "(c) full LZ-like space (elastic 14 ops x s/v x 13 m + inelastic O1/O4 x s/v x 3 m x 8 delta)"
    out["main_result"] = dfs[dfs.subspace == main_key].iloc[0].to_dict()
    log("MAIN: N_eff@3.4 =", out["main_result"]["N_eff@3.4"], "Z_global@3.4 =", out["main_result"]["Z_global@3.4"])

    # --- Bonferroni for the LZ counts
    out["Bonferroni"] = {n: dict(p_global=min(1, n * P_LOC_LZ), Z_global=float(lz.p_to_sigma(min(1, n * P_LOC_LZ)))) for n in (14, 48, 96, 293, 616)}

    # --- correlation-based effective number (Nyholt/Cheverud M_eff) on Z = sqrt(q0) of the eval toys
    Zev = np.sqrt(q_ev[:, is_lz].astype(float))
    Zev = Zev[:, Zev.std(axis=0) > 0]
    corr = np.corrcoef(Zev, rowvar=False)
    lam = np.linalg.eigvalsh(np.nan_to_num(corr))
    Mtot = corr.shape[0]
    out["Nyholt_M_eff_full_space"] = float(1 + (Mtot - 1) * (1 - np.var(lam, ddof=1) / Mtot))
    out["Nyholt_M_total"] = int(Mtot)

    # --- Gross-Vitells on the fine delta scan (asymptotic Z = sqrt(q0) curves per toy)
    dsel = cls_k == "deltascan"
    Zd = np.sqrt(q_ev[:, dsel].astype(float))       # (T, n_delta)
    deltas_d = delta_k[dsel]
    gv = []
    for z0 in (0.5, 1.0, 1.5):
        up = ((Zd[:, 1:] > z0) & (Zd[:, :-1] <= z0)).sum(axis=1) + (Zd[:, 0] > z0)
        Nup = float(up.mean())
        for z in (2.0, 2.5, 3.0, 3.4):
            p_gv = float(stats.norm.sf(z) + Nup * np.exp(-(z ** 2 - z0 ** 2) / 2))
            p_direct = float((Zd.max(axis=1) >= z).mean())
            gv.append(dict(z0=z0, N_up_mean=Nup, z=z, p_global_GV=p_gv, p_global_direct_asymptoticZ=p_direct,
                           N_eff_GV=neff_from_pglobal(p_gv, float(stats.norm.sf(z))),
                           N_eff_direct=neff_from_pglobal(p_direct, float(stats.norm.sf(z))) if p_direct > 0 else np.nan))
    pd.DataFrame(gv).to_csv(os.path.join(WORK, "gross_vitells_deltascan.csv"), index=False)

    # --- the observed data in this simplified likelihood: 248 keV event + typical low-energy events
    acc_pdf = comps["accidentals"][1]
    cdf_acc = np.cumsum(acc_pdf) * DE
    def low_events(n):
        if n == 0: return []
        u = (np.arange(n) + 0.5) / n
        return list(np.interp(u, cdf_acc, E_OBS))
    data_rows = []
    for n_low in (0, 2, 3, 5):
        ev = np.array([[248.0] + low_events(n_low) + [np.nan] * (20 - 1 - n_low)])
        q_d, s_d = q0_for_events(ev, Fk, b_dens)
        p_d = cal.p(q_d)[0]
        for i in range(M):
            data_rows.append(dict(n_low=n_low, model=names_k[i], cls=cls_k[i], op=op_k[i], tau=tau_k[i], mass=mass_k[i], delta=delta_k[i],
                                  q0=float(q_d[0, i]), s_hat=float(s_d[0, i]), p_local=float(p_d[i]),
                                  Z_local=float(lz.p_to_sigma(p_d[i])) if p_d[i] < 0.5 else 0.0, Z_asymptotic=float(np.sqrt(q_d[0, i]))))
    dfd = pd.DataFrame(data_rows)
    dfd.to_csv(os.path.join(WORK, "data_local_Z.csv"), index=False)
    d3 = dfd[dfd.n_low == 3]
    out["data_localZ_n_low=3"] = dict(
        max_Z=float(d3.Z_local.max()), argmax=str(d3.loc[d3.Z_local.idxmax(), "model"]),
        p_min=float(d3.p_local.min()),
        O1s_by_mass={int(m): float(d3[(d3.model == f"O1s_m{m}")].Z_local.iloc[0]) for m in MASSES},
        O1v_by_mass={int(m): float(d3[(d3.model == f"O1v_m{m}")].Z_local.iloc[0]) for m in MASSES},
        O4s_by_mass={int(m): float(d3[(d3.model == f"O4s_m{m}")].Z_local.iloc[0]) for m in MASSES},
        O6s_by_mass={int(m): float(d3[(d3.model == f"O6s_m{m}")].Z_local.iloc[0]) for m in MASSES},
        O10s_by_mass={int(m): float(d3[(d3.model == f"O10s_m{m}")].Z_local.iloc[0]) for m in MASSES},
        O11s_by_mass={int(m): float(d3[(d3.model == f"O11s_m{m}")].Z_local.iloc[0]) for m in MASSES},
        O1s_inel_1000={int(d): float(d3[(d3.model == f"O1s_m1000_d{d}")].Z_local.iloc[0]) if (d3.model == f"O1s_m1000_d{d}").any() else None for d in INEL_DELTAS},
        O1v_inel_1000={int(d): float(d3[(d3.model == f"O1v_m1000_d{d}")].Z_local.iloc[0]) if (d3.model == f"O1v_m1000_d{d}").any() else None for d in INEL_DELTAS},
        O4s_inel_1000={int(d): float(d3[(d3.model == f"O4s_m1000_d{d}")].Z_local.iloc[0]) if (d3.model == f"O4s_m1000_d{d}").any() else None for d in INEL_DELTAS},
        n_models_Z_ge_3=int((d3[d3.cls != "deltascan"].Z_local >= 3.0).sum()),
        n_models_Z_le_0p5=int((d3[d3.cls != "deltascan"].Z_local <= 0.5).sum()))
    # global significance of the *data* in this simplified likelihood (own p_min against min-p distribution)
    for k in (main_key,):
        pm = pmins[k]
        p_min_data = float(d3[np.isin(d3.model, names_k[sub[k]])].p_local.min())
        pg = float((pm <= p_min_data).mean())
        out["data_global_in_simplified_likelihood"] = dict(p_min_data=p_min_data, Z_local_max=float(lz.p_to_sigma(p_min_data)),
                                                           p_global=pg, Z_global=float(lz.p_to_sigma(pg)) if pg > 0 else np.inf,
                                                           N_eff=neff_from_pglobal(pg, p_min_data))
    # dependence on n_low for the max-Z and for the O1s 100 GeV model
    out["data_maxZ_vs_n_low"] = {int(n): float(dfd[(dfd.n_low == n) & (dfd.cls != "deltascan")].Z_local.max()) for n in (0, 2, 3, 5)}

    # --- analytic single-event picture: local p of a heavy model ~ background expected where f/b >= (f/b)(248)
    i6 = list(names_k).index("O6s_m1000")
    lr = Fk[i6] / b_dens
    lr_obs = lr[np.argmin(abs(E_OBS - 248))]
    out["single_event_picture_O6s_1000"] = dict(f_over_b_at_248=float(lr_obs),
                                                Z_asymptotic_formula=float(np.sqrt(2 * (np.log(lr_obs) - 1))) if lr_obs > np.e else 0.0,
                                                bkg_expected_where_LR_ge_obs=float(b_dens[lr >= lr_obs].sum() * DE),
                                                p_local_toy=float(d3[d3.model == "O6s_m1000"].p_local.iloc[0]))
    # union of signal-dominated regions over all heavy models at the observed LR of each -> N_eff proxy
    heavy = is_lz & (mass_k >= 100)
    reg = np.zeros(len(E_OBS), bool)
    for i in np.where(heavy)[0]:
        lr = Fk[i] / b_dens
        thr = lr[np.argmin(abs(E_OBS - 248))]
        if thr > 1: reg |= lr >= thr
    out["single_event_picture_union"] = dict(bkg_expected_in_union_of_heavy_signal_regions=float(b_dens[reg].sum() * DE),
                                             union_region_keV=[float(E_OBS[reg].min()) if reg.any() else None, float(E_OBS[reg].max()) if reg.any() else None])

    # --- robustness variants (fewer toys)
    variants = {
        "nominal": dict(),
        "accidentals tau=10 keV": dict(tau_acc=10.0),
        "accidentals tau=20 keV": dict(tau_acc=20.0),
        "high-E bkg up: neutrons 0.118 tau=60, MSSI x2": dict(n_neutron=0.118, tau_n=60.0, mssi_scale=2.0),
        "high-E bkg down: neutrons 0.02, MSSI x0.5": dict(n_neutron=0.02, mssi_scale=0.5),
        "atm-nu flux flat below 100 MeV": dict(atm_index=0.0),
    }
    vrows = []
    for vname, kw in variants.items():
        cv = build_background(**kw)
        Bv, bv = total_background(cv)
        rng_v = np.random.default_rng(1234)
        qc, _ = run_toys(N_TOYS_VAR // 2, Fk, bv, Bv, rng_v)
        qe, _ = run_toys(N_TOYS_VAR // 2, Fk, bv, Bv, rng_v)
        pv = Calibrator(qc).p(qe)
        res, _ = global_stats(pv, is_lz, [P_LOC_LZ, float(lz.sigma_to_p(3.0))])
        # data local max Z in this variant
        ev = np.array([[248.0] + low_events(3) + [np.nan] * 16])
        qd, _ = q0_for_events(ev, Fk, bv)
        pdv = Calibrator(qc).p(qd)[0]
        vrows.append(dict(variant=vname, B_total=Bv, B_above_200=float(bv[E_OBS > 200].sum() * DE),
                          N_eff_at_3p4=res[P_LOC_LZ]["N_eff"], N_eff_err_at_3p4=res[P_LOC_LZ]["N_eff_err"], Z_global_at_3p4=res[P_LOC_LZ]["Z_global"],
                          N_eff_at_3p0=res[float(lz.sigma_to_p(3.0))]["N_eff"], Z_global_at_3p0=res[float(lz.sigma_to_p(3.0))]["Z_global"],
                          data_maxZ_local=float(lz.p_to_sigma(pdv[is_lz].min())), n_toys=N_TOYS_VAR))
        log("variant", vname, "N_eff@3.4 =", vrows[-1]["N_eff_at_3p4"])
    pd.DataFrame(vrows).to_csv(os.path.join(WORK, "robustness.csv"), index=False)

    # --- figures
    # Fig 1: distribution of max local Z over the full space, vs single model; LZ 3.4 line
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    Zmax_full = lz.p_to_sigma(np.clip(pmins[main_key], 1e-12, 1)); Zmax_full = np.where(pmins[main_key] >= 0.5, 0, Zmax_full)
    Zsingle = lz.p_to_sigma(np.clip(pmins["single model: O6s 1000 GeV"], 1e-12, 1)); Zsingle = np.where(pmins["single model: O6s 1000 GeV"] >= 0.5, 0, Zsingle)
    Zinel = lz.p_to_sigma(np.clip(pmins["(a) inelastic O1/O4 s+v (96 raw)"], 1e-12, 1)); Zinel = np.where(pmins["(a) inelastic O1/O4 s+v (96 raw)"] >= 0.5, 0, Zinel)
    zz = np.linspace(0, 4.5, 91)
    for Zs, lab, c in ((Zmax_full, f"max over full model space ({int(is_lz.sum())} models)", "C3"),
                       (Zinel, "max over inelastic O1/O4 (96)", "C1"), (Zsingle, "single model (O6s, 1 TeV)", "C0")):
        surv = [(Zs >= z).mean() for z in zz]
        ax.plot(zz, surv, color=c, label=lab)
    ax.plot(zz, stats.norm.sf(zz), "k--", lw=1, label="one-sided Gaussian tail p(Z)")
    ax.axvline(3.4, color="gray", ls=":"); ax.text(3.42, 0.3, "LZ local 3.4σ", rotation=90, va="center", fontsize=8)
    ax.axhline(float(lz.sigma_to_p(2.6)), color="gray", ls="-.", lw=0.8); ax.text(0.1, float(lz.sigma_to_p(2.6)) * 1.3, "LZ global 2.6σ (p=4.7e-3)", fontsize=8)
    ax.set_yscale("log"); ax.set_ylim(1e-5, 1.5); ax.set_xlabel("local significance threshold Z"); ax.set_ylabel("P(max local Z ≥ Z | background only)")
    ax.legend(fontsize=7.5, loc="lower left"); ax.set_title("P008: background-only toys, simplified 1D likelihood", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_maxZ_distribution.png"), dpi=160); plt.close(fig)

    # Fig 2: N_eff vs model-space size
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    pts = dfs[~dfs.subspace.str.startswith("(d)") & ~dfs.subspace.str.contains("fine")]
    ax.scatter(pts.n_models_raw, pts["N_eff@3.4"], color="C3", label="N_eff at local 3.4σ (toys)")
    ax.scatter(pts.n_models_raw, pts["N_eff@3.0"], color="C0", marker="s", label="N_eff at local 3.0σ (toys)")
    for _, r in pts.iterrows():
        ax.annotate(r.subspace.split(" (")[0][:28], (r.n_models_raw, r["N_eff@3.4"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    xx = np.geomspace(1, 700, 50)
    ax.plot(xx, xx, "k--", lw=1, label="Bonferroni (N_eff = N_models)")
    ax.axhline(out["N_eff_implied_by_LZ_3.4_to_2.6"], color="gray", ls=":", label=f"LZ implied N_eff = {out['N_eff_implied_by_LZ_3.4_to_2.6']:.1f}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("number of models in the tested space"); ax.set_ylabel("effective number of independent trials N_eff")
    ax.legend(fontsize=7.5); ax.set_title("P008: trials factor vs model space", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_neff_vs_modelspace.png"), dpi=160); plt.close(fig)

    # Fig 3: background and representative signal shapes
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, (mu, pdf) in comps.items():
        ax.plot(E_OBS, mu * pdf, lw=1, label=f"{k} ({mu:.3g} ev)")
    ax.plot(E_OBS, b_dens, "k", lw=1.5, label=f"total background ({B:.2f} ev)")
    for nm, c in (("O1s_m1000", "C3"), ("O1s_m30", "C8"), ("O6s_m1000", "C1"), ("O1s_m1000_d300", "C2"), ("O4s_m1000", "C4")):
        i = list(names_k).index(nm)
        ax.plot(E_OBS, Fk[i], "--", color=c, lw=1, label=f"signal shape {nm} (1 ev)")
    ax.axvline(248, color="gray", ls=":")
    ax.set_yscale("log"); ax.set_ylim(1e-7, 1); ax.set_xlabel("observed NR-equivalent energy [keV]"); ax.set_ylabel("events / keV in 2.84 t·yr")
    ax.legend(fontsize=6.5, ncol=2); ax.set_title("P008: simplified 1D model (smeared, efficiency applied)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_spectra_background.png"), dpi=160); plt.close(fig)

    # Fig 4: N_eff vs delta_max for the continuous scan + GV
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    dpts = dfs[dfs.subspace.str.startswith("(d)")]
    dm = [int(s.split("<= ")[1]) for s in dpts.subspace]
    ax.plot(dm, dpts["N_eff@3.4"], "o-", color="C3", label="N_eff at 3.4σ, toy-calibrated (10 keV δ steps)")
    ax.plot(dm, dpts["N_eff@3.0"], "s-", color="C0", label="N_eff at 3.0σ")
    ax.plot(dm, dpts.n_models_raw, "k--", lw=1, label="Bonferroni")
    ax.set_xlabel("δ_max of the scan [keV] (O1 isoscalar, 1000 GeV)"); ax.set_ylabel("N_eff"); ax.legend(fontsize=7.5)
    ax.set_title("P008: trials factor of a continuous mass-splitting scan", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_neff_vs_deltamax.png"), dpi=160); plt.close(fig)

    out["runtime_s"] = time.time() - T0
    json.dump(out, open(os.path.join(WORK, "results.json"), "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    log("done")
    print(json.dumps(out["main_result"], indent=1, default=str))
    print("N_eff implied by LZ:", out["N_eff_implied_by_LZ_3.4_to_2.6"])
    print("data local Z summary:", json.dumps(out["data_localZ_n_low=3"], indent=1, default=str))
    print("data global (simplified):", out["data_global_in_simplified_likelihood"])
    print("Nyholt M_eff:", out["Nyholt_M_eff_full_space"], "of", out["Nyholt_M_total"])

if __name__ == "__main__":
    main()
