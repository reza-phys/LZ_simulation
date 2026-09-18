"""
P071_lee_alternatives.py -- alternative look-elsewhere treatments for the LZ model scan.

Run from the simulation root:
    .venv/bin/python output/code/P071_lee_alternatives.py                 # full run (4 x 50k toys)
    N_TOYS=10000 .venv/bin/python output/code/P071_lee_alternatives.py    # quick run
    SPECTRA_ONLY=1 .venv/bin/python output/code/P071_lee_alternatives.py  # only build the extra WimPyDD cache

Re-uses P008's one-dimensional extended likelihood, background model, WimPyDD spectra cache
(output/work/P008/spectra_cache.npz), toy generator, profile-likelihood q0 and toy calibration
(imported from output/code/P008_lee_toys.py; its main() is not executed).  New here:
  1. Bonferroni / Sidak bounds; eigenvalue-based effective numbers (Nyholt-Cheverud, Li-Ji, participation
     ratio, Galwey, Gao) from the correlation matrix of calibrated per-model Z's and of tail indicators;
  2. Gross-Vitells with upcrossings / Euler characteristic counted in the toys, in the *calibrated*-Z metric
     (marginally N(0,1) by construction) as well as the asymptotic sqrt(q0) metric, 1D (delta scan) and 2D
     ((m, delta) inelastic grids), extrapolated from low thresholds to 3.4 sigma and compared with direct toys;
  3. threshold dependence N_eff(Z_local), the LEE cost in sigma, and its extrapolation to 5 sigma;
  4. pre-registration counterfactual search spaces and the "corpus LEE": LZ's space enlarged by the
     variants tested in this corpus (continuous delta scan [P021], light mediators [P051], isospin ratios
     [P032], halo/date variants [P018, P006]);
  5. the Bayesian trials factor (P027) versus a likelihood-weighted effective model count.
Toys: <= 50,000 per configuration (two independent calibration/evaluation pairs).
Outputs: output/work/P071/*.csv, results.json, extra_spectra_cache.npz, figures/*.png
Tools: python 3.12.13, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, matplotlib 3.11.2, WimPyDD 2.0.4 (via lzcommon),
       common/lzcommon.py, output/code/P008_lee_toys.py.
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
import P008_lee_toys as p8

WORK = "output/work/P071"
FIG = os.path.join(WORK, "figures")
os.makedirs(FIG, exist_ok=True)
N_TOYS = int(os.environ.get("N_TOYS", "50000"))          # per configuration (hard cap 5e4)
assert N_TOYS <= 50000
SPECTRA_ONLY = os.environ.get("SPECTRA_ONLY", "0") == "1"
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
Zs = lambda p: float(lz.p_to_sigma(p)) if 0 < p < 0.5 else (0.0 if p >= 0.5 else np.inf)
P_LZ = float(lz.sigma_to_p(3.4))
Z_LIST = (2.0, 2.5, 3.0, 3.4, 3.6)
P_LIST = [float(lz.sigma_to_p(z)) for z in Z_LIST]
neff = p8.neff_from_pglobal

# palette (dataviz skill default categorical slots; single-hue use where one series)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_VIOLET, C_RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"
C_MUTED, C_GRID, C_INK2 = "#898781", "#e1e0d9", "#52514e"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": C_INK2, "xtick.color": C_INK2,
                     "ytick.color": C_INK2, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})

# ----------------------------------------------------------------------------------------------
# 1. Background and LZ-like signal spectra (P008 machinery, cached WimPyDD spectra)
# ----------------------------------------------------------------------------------------------
comps = p8.build_background()
B, b_dens = p8.total_background(comps)
models = p8.model_list()
rates = p8.compute_spectra(models)                       # loads output/work/P008/spectra_cache.npz
F, keep, frac200, frac_roi = p8.signal_pdfs(models, rates)
dfm = pd.DataFrame(models)
dfm["physical"] = keep; dfm["frac_above_200keV"] = frac200
E_WD, E_TRUE, E_OBS, DE = p8.E_WD, p8.E_TRUE, p8.E_OBS, p8.DE
log("P008 space loaded:", len(models), "models,", int(keep.sum()), "physical; background", round(B, 3))

# ----------------------------------------------------------------------------------------------
# 2. Extra spectra tested by the corpus ("community" additions)
# ----------------------------------------------------------------------------------------------
M_N_GEV = lz.A_XE_MEAN * lz.AMU_GEV                      # ~122.3 GeV nuclear mass
def propagator(E_keV, m_med_gev):
    q2 = 2.0 * M_N_GEV * E_keV * 1e-6                    # GeV^2
    return (m_med_gev ** 2 / (m_med_gev ** 2 + q2)) ** 2

LM_MED = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]                 # P051: six mediator masses
LM_DELTAS = list(range(250, 400, 10))                    # 15 mass splittings
ISO_R = [round(x, 2) for x in np.linspace(-1, 1, 41)]    # P032: 41 ratios c_n/c_p at 1 TeV, elastic O1
ISO_INEL = [(-0.7, 300), (0.0, 300), (0.5, 300), (-0.7, 350), (0.0, 350), (0.5, 350)]
HALOS = {"vesc500": dict(vesc=500.0), "vesc528": dict(vesc=528.0), "vesc560": dict(vesc=560.0), "vesc600": dict(vesc=600.0),
         "v0_220": dict(v0=220.0), "v0_250": dict(v0=250.0), "june": dict(day_of_year=167), "december": dict(day_of_year=350)}
HALO_DELTAS = [300, 350, 380]

def extra_models():
    ex = []
    for mm in LM_MED:
        for d in LM_DELTAS:
            ex.append(dict(name=f"LM_m{mm}_d{d}", cls="lightmed", op=1, tau="s", mass=1000, delta=float(d), m_med=mm))
    for r in ISO_R:
        ex.append(dict(name=f"ISO_r{r:+.2f}_m1000", cls="isospin", op=1, tau="r", mass=1000, delta=0.0, r=r))
    for r, d in ISO_INEL:
        ex.append(dict(name=f"ISO_r{r:+.2f}_m1000_d{d}", cls="isospin", op=1, tau="r", mass=1000, delta=float(d), r=r))
    for h in HALOS:
        for d in HALO_DELTAS:
            ex.append(dict(name=f"HALO_{h}_d{d}", cls="halo", op=1, tau="s", mass=1000, delta=float(d), halo=h))
    return ex

def compute_extra(ex):
    cache = os.path.join(WORK, "extra_spectra_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z["names"]) == [m["name"] for m in ex]:
            log("loaded cached extra spectra", cache); return z["rates"]
    out = np.zeros((len(ex), len(E_WD)))
    names = [m["name"] for m in models]
    halo0 = lz.wd_halo()
    ham_s = lz.wd_hamiltonian("P071_O1s", {1: (1e-3, 0.0)})
    halos = {}
    for i, m in enumerate(ex):
        if m["cls"] == "lightmed":
            base = rates[names.index(f"O1s_m1000_dfine{int(m['delta'])}")]
            out[i] = np.maximum(base, 0) * propagator(E_WD, m["m_med"])
        elif m["cls"] == "isospin":
            cp, cn = 1e-3, m["r"] * 1e-3
            ham = lz.wd_hamiltonian(f"P071_O1_r{m['r']:+.2f}", {1: (cp + cn, cp - cn)})   # WimPyDD c0=cp+cn, c1=cp-cn
            out[i] = lz.wd_rate(ham, 1000.0, E_WD, halo=halo0, delta_kev=m["delta"])
        elif m["cls"] == "halo":
            if m["halo"] not in halos:
                halos[m["halo"]] = lz.wd_halo(**HALOS[m["halo"]])
            out[i] = lz.wd_rate(ham_s, 1000.0, E_WD, halo=halos[m["halo"]], delta_kev=m["delta"])
        if i % 20 == 0: log(f"  extra spectrum {i}/{len(ex)}: {m['name']}")
    np.savez(cache, names=np.array([m["name"] for m in ex]), rates=out, E_WD=E_WD)
    return out

ex_models = extra_models()
ex_rates = compute_extra(ex_models)
# isospin convention check: proton-only (r=0) second M-node should sit near 292 keV (P032), neutron-only near 247 keV
i_r0 = [m["name"] for m in ex_models].index("ISO_r+0.00_m1000")
r0 = ex_rates[i_r0]; sel = (E_WD > 150) & (E_WD < 380)
node_r0 = float(E_WD[sel][np.argmin(r0[sel])])
log("isospin check: proton-only (r=0) high-energy node at", round(node_r0, 1), "keV (P032: 292 proton-only / 247 neutron-only)")
if SPECTRA_ONLY:
    sys.exit(0)
Fx, keepx, frac200x, _ = p8.signal_pdfs(ex_models, ex_rates)
dfx = pd.DataFrame(ex_models); dfx["physical"] = keepx; dfx["frac_above_200keV"] = frac200x
log("extra spectra:", len(ex_models), "physical", int(keepx.sum()))

# ----------------------------------------------------------------------------------------------
# 3. Combined model set and toys
# ----------------------------------------------------------------------------------------------
ALL = pd.concat([dfm, dfx], ignore_index=True)
FALL = np.vstack([F, Fx]); KEEP = np.concatenate([keep, keepx])
Fk = FALL[KEEP]; A = ALL[KEEP].reset_index(drop=True)
M = Fk.shape[0]
cls, names, mass, delta, op, tau = (A[c].values for c in ("cls", "name", "mass", "delta", "op", "tau"))
log("total models in toys:", M)

def make_pair(seed_cal, seed_ev):
    q_cal, _ = p8.run_toys(N_TOYS, Fk, b_dens, B, np.random.default_rng(seed_cal), chunk=2000)
    q_ev, n_ev = p8.run_toys(N_TOYS, Fk, b_dens, B, np.random.default_rng(seed_ev), chunk=2000)
    cal = p8.Calibrator(q_cal)
    return cal, q_ev, cal.p(q_ev), n_ev, q_cal

log(f"toys: pair A ({N_TOYS} cal + {N_TOYS} eval) ...")
calA, q_evA, p_evA, n_evA, q_calA = make_pair(20260915, 20260916)
log("toys: pair B ...")
calB, q_evB, p_evB, n_evB, q_calB = make_pair(20260917, 20260918)
log("toys done")

# ----------------------------------------------------------------------------------------------
# 4. Search spaces (LZ-like, counterfactual, corpus)
# ----------------------------------------------------------------------------------------------
is_lz = np.isin(cls, ["elastic", "inelastic"])
is_scan = cls == "deltascan"
heavy1000 = (cls == "elastic") & (mass == 1000)
SIX_APRIORI = ["O6s_m1000", "O10s_m1000", "O9s_m1000", "O15s_m1000", "O1s_m1000_d300", "O1s_m1000_d350"]
top6 = A[is_lz].sort_values("frac_above_200keV", ascending=False).name.values[:6]
spaces = {
    "(c) single model O6s 1 TeV [L10/L4-like]": names == "O6s_m1000",
    "(c') single model O1s 1 TeV delta=350": names == "O1s_m1000_d350",
    "(b1) 6 a-priori high-energy spectra": np.isin(names, SIX_APRIORI),
    "(b2) 6 spectra with largest fraction above 200 keV": np.isin(names, top6),
    "(a) 14 isoscalar ops at 1 TeV + O1s/O4s inelastic 1 TeV x 8 delta (30)": (heavy1000 & (tau == "s")) | ((cls == "inelastic") & (mass == 1000) & (tau == "s")),
    "(a') 28 s+v ops at 1 TeV + O1/O4 s+v inelastic 1 TeV x 8 delta (60)": heavy1000 | ((cls == "inelastic") & (mass == 1000)),
    "heavy elastic m>=100 GeV (140)": (cls == "elastic") & (mass >= 100),
    "inelastic O1/O4 grid (96)": cls == "inelastic",
    "LZ-like full space (456; LZ 293 distinct)": is_lz,
    "+ continuous delta scan [P021] (497)": is_lz | is_scan,
    "+ light mediators [P051]": is_lz | is_scan | (cls == "lightmed"),
    "+ isospin ratios [P032]": is_lz | is_scan | (cls == "lightmed") | (cls == "isospin"),
    "+ halo/date variants [P018,P006] = corpus space": is_lz | is_scan | (cls == "lightmed") | (cls == "isospin") | (cls == "halo"),
    "corpus extras only (scan+lightmed+isospin+halo)": ~is_lz,
}
rows = []
for k, mask in spaces.items():
    row = dict(space=k, n_models=int(mask.sum()))
    for tag, p_ev in (("A", p_evA), ("B", p_evB)):
        res, pmin = p8.global_stats(p_ev, mask, P_LIST)
        for z, pl in zip(Z_LIST, P_LIST):
            row[f"p_global@{z}_{tag}"] = res[pl]["p_global"]; row[f"N_eff@{z}_{tag}"] = res[pl]["N_eff"]; row[f"Z_global@{z}_{tag}"] = res[pl]["Z_global"]
    for z in Z_LIST:
        pa, pb = row[f"p_global@{z}_A"], row[f"p_global@{z}_B"]
        pg = 0.5 * (pa + pb)
        row[f"p_global@{z}"] = pg; row[f"Z_global@{z}"] = Zs(pg); row[f"N_eff@{z}"] = neff(pg, float(lz.sigma_to_p(z)))
        row[f"N_eff_spread@{z}"] = abs(row[f"N_eff@{z}_A"] - row[f"N_eff@{z}_B"]) / 2
        row[f"N_eff_binom@{z}"] = np.sqrt(max(pg * 2 * N_TOYS, 1)) / (2 * N_TOYS) / float(lz.sigma_to_p(z))
    row["Bonferroni_Z@3.4"] = Zs(min(1.0, row["n_models"] * P_LZ))
    rows.append(row)
dfs = pd.DataFrame(rows)
dfs.to_csv(os.path.join(WORK, "search_spaces.csv"), index=False)
main = dfs[dfs.space.str.startswith("LZ-like")].iloc[0]
log("MAIN LZ-like: N_eff@3.4 =", round(main["N_eff@3.4"], 2), "(A", round(main["N_eff@3.4_A"], 2), "B", round(main["N_eff@3.4_B"], 2), ") Z_global =", round(main["Z_global@3.4"], 3))
corpus = dfs[dfs.space.str.contains("corpus space")].iloc[0]
log("CORPUS: N_eff@3.4 =", round(corpus["N_eff@3.4"], 2), "Z_global =", round(corpus["Z_global@3.4"], 3), " @3.6:", round(corpus["Z_global@3.6"], 3))
print(dfs[["space", "n_models", "N_eff@3.4", "N_eff_spread@3.4", "Z_global@3.4", "Z_global@3.6", "Bonferroni_Z@3.4"]].to_string())

# ----------------------------------------------------------------------------------------------
# 4b. Calibration-size dependence: a minimum over M noisily calibrated p-values is biased low, so N_eff is
#     biased high for finite N_cal. Re-calibrate with disjoint sub-samples (6.25k .. 50k) and extrapolate.
# ----------------------------------------------------------------------------------------------
ZC = (2.0, 2.5, 3.0, 3.4)
def neff_vs_ncal(q_cal, q_ev, mask, sizes=(6250, 12500, 25000, 50000)):
    rows = []
    qe = q_ev[:, mask]
    for n in sizes:
        nrep = N_TOYS // n
        vals = {z: [] for z in ZC}
        for r in range(nrep):
            sub = p8.Calibrator(q_cal[r * n:(r + 1) * n][:, mask])
            pmin = sub.p(qe).min(axis=1)
            for z in ZC:
                pl = float(lz.sigma_to_p(z)); pg = float((pmin <= pl).mean())
                vals[z].append(neff(pg, pl) if pg > 0 else np.nan)
        row = dict(N_cal=n, n_rep=nrep)
        for z in ZC:
            row[f"N_eff@{z}"] = float(np.nanmean(vals[z])); row[f"N_eff_sem@{z}"] = float(np.nanstd(vals[z]) / np.sqrt(max(nrep, 1)))
        rows.append(row)
    return pd.DataFrame(rows)

SIZES = (6250, 12500, 25000, 50000)
recs = []                       # one record per (pair, N_cal, replicate, space, Z)
for tag, (qc, qe) in (("A", (q_calA, q_evA)), ("B", (q_calB, q_evB))):
    for n in SIZES:
        nrep = N_TOYS // n
        for r in range(nrep):
            p_sub = p8.Calibrator(qc[r * n:(r + 1) * n]).p(qe)          # calibrate all models on a disjoint sub-sample
            for lab, mask in spaces.items():
                pmin = p_sub[:, mask].min(axis=1)
                for z in ZC:
                    pl = float(lz.sigma_to_p(z)); pg = float((pmin <= pl).mean())
                    recs.append(dict(pair=tag, N_cal=n, rep=r, space=lab, Z=z, N_eff=neff(pg, pl) if pg > 0 else np.nan))
        log(f"  calibration-size study: pair {tag}, N_cal = {n} ({nrep} replicates) done")
dfc_all = pd.DataFrame(recs)
dfc = dfc_all.groupby(["space", "pair", "N_cal", "Z"]).N_eff.agg(["mean", "std", "count"]).reset_index()
dfc.to_csv(os.path.join(WORK, "calibration_size_dependence.csv"), index=False)

def extrapolate(g):
    """g: N_cal-sorted frame with mean N_eff; fit a + b/N_cal and a + b/sqrt(N_cal); average, half-range as systematic."""
    x = g.N_cal.values.astype(float); yv = g.N_eff.values
    b1, a1 = np.polyfit(1 / x, yv, 1); b2, a2 = np.polyfit(1 / np.sqrt(x), yv, 1); nd = 0.5 * (a1 + a2)
    return dict(N_eff_50k=float(yv[-1]), N_eff_inf_1overN=float(a1), N_eff_inf_1oversqrtN=float(a2), N_eff_debiased=float(nd),
                N_eff_debiased_halfrange=float(abs(a1 - a2) / 2), bias_at_50k_pct=float(100 * (yv[-1] / nd - 1)))
deb = {}
for lab in spaces:
    deb[lab] = {}
    for z in ZC:
        g = dfc_all[(dfc_all.space == lab) & (dfc_all.Z == z)].groupby("N_cal").N_eff.mean().reset_index().sort_values("N_cal")
        if g.N_eff.isna().any() or len(g) < 4:
            deb[lab][z] = dict(N_eff_debiased=np.nan, Z_global_debiased=np.nan); continue
        deb[lab][z] = extrapolate(g)
        deb[lab][z]["Z_global_debiased"] = Zs(1 - (1 - float(lz.sigma_to_p(z))) ** deb[lab][z]["N_eff_debiased"])
for z in ZC:
    dfs[f"N_eff_deb@{z}"] = [deb[k][z]["N_eff_debiased"] for k in dfs.space]
    dfs[f"N_eff_deb_halfrange@{z}"] = [deb[k][z].get("N_eff_debiased_halfrange", np.nan) for k in dfs.space]
    dfs[f"Z_global_deb@{z}"] = [deb[k][z]["Z_global_debiased"] for k in dfs.space]
dfs.to_csv(os.path.join(WORK, "search_spaces.csv"), index=False)
deb_main = deb[main["space"]]; deb_corp = deb[corpus["space"]]
print("calibration-size dependence (LZ-like / corpus; mean over pairs and replicates):")
print(dfc_all[dfc_all.space.isin([main["space"], corpus["space"]])].groupby(["space", "N_cal", "Z"]).N_eff.mean().unstack("Z").to_string())
print("debiased LZ-like:", json.dumps(deb_main, indent=1, default=float))
print("debiased corpus:", json.dumps(deb_corp, indent=1, default=float))
print(dfs[["space", "n_models", "N_eff@3.4", "N_eff_deb@3.4", "N_eff_deb_halfrange@3.4", "Z_global@3.4", "Z_global_deb@3.4", "Z_global_deb@3.0"]].to_string())
ratio_corpus = deb_corp[3.4]["N_eff_debiased"] / deb_main[3.4]["N_eff_debiased"]
honest = {lab_: dict(N_eff=float(base * ratio_corpus), Z_global=Zs(1 - (1 - P_LZ) ** (base * ratio_corpus)))
          for lab_, base in (("this_work_debiased", deb_main[3.4]["N_eff_debiased"]), ("P008_12.2", 12.2), ("LZ_13.9", 13.86))}
print("corpus/LZ-like N_eff ratio", ratio_corpus, "honest global:", honest)
del q_calA, q_calB, calB, recs, p_sub
import gc; gc.collect()
log("calibration matrices released")

# ----------------------------------------------------------------------------------------------
# 5. Bonferroni / Sidak bounds
# ----------------------------------------------------------------------------------------------
bonf = []
for Mb in (1, 6, 30, 36, 60, 140, 293, 456, 616, int((~is_lz).sum() + is_lz.sum())):
    for z in (3.4, 3.6, 5.0):
        pl = float(lz.sigma_to_p(z))
        bonf.append(dict(M=Mb, Z_local=z, p_Bonferroni=min(1, Mb * pl), Z_Bonferroni=Zs(min(1, Mb * pl)),
                         p_Sidak=1 - (1 - pl) ** Mb, Z_Sidak=Zs(1 - (1 - pl) ** Mb)))
dfb = pd.DataFrame(bonf); dfb.to_csv(os.path.join(WORK, "bonferroni_sidak.csv"), index=False)

# ----------------------------------------------------------------------------------------------
# 6. Correlation / eigenvalue effective numbers (LZ-like space, pair A)
# ----------------------------------------------------------------------------------------------
def eigen_estimators(X):
    """X: (T, M) variables. Returns dict of eigenvalue-based effective numbers of tests."""
    X = X[:, X.std(axis=0) > 0]
    Mm = X.shape[1]
    C = np.corrcoef(X, rowvar=False)
    lam = np.clip(np.linalg.eigvalsh(np.nan_to_num(C)), 0, None)
    lam_sorted = np.sort(lam)[::-1]
    cum = np.cumsum(lam_sorted) / lam_sorted.sum()
    return dict(M=int(Mm),
                Nyholt_Cheverud=float(1 + (Mm - 1) * (1 - np.var(lam, ddof=1) / Mm)),
                Li_Ji=float(np.sum((lam >= 1).astype(float) + (lam - np.floor(lam)))),
                participation_ratio=float(lam.sum() ** 2 / np.sum(lam ** 2)),
                Galwey=float(np.sum(np.sqrt(lam)) ** 2 / lam.sum()),
                Gao_99p5=int(np.searchsorted(cum, 0.995) + 1),
                lambda_max=float(lam_sorted[0]), top5_variance_fraction=float(cum[4]))

rng = np.random.default_rng(7)
qA = q_evA[:, is_lz]; pA = p_evA[:, is_lz].astype(np.float32)       # float32 throughout to limit memory
# (i) randomised-PIT Gaussianisation: p is discrete with an atom at q0 = 0 (p = 1); spread the atom uniformly
p0 = (qA <= 0).mean(axis=0, dtype=np.float32)                  # P(q0 = 0) per model
u = np.where(qA <= 0, (1 - p0)[None, :] + rng.random(qA.shape, dtype=np.float32) * p0[None, :], pA)
u = np.clip(u, 0.5 / N_TOYS, 1 - 1e-6).astype(np.float32)
Zg = stats.norm.isf(u).astype(np.float32); del u
eig = {"gaussianised_Z (randomised PIT)": eigen_estimators(Zg)}
del Zg; gc.collect()
eig["asymptotic sqrt(q0) [P008 metric]"] = eigen_estimators(np.sqrt(qA))
for alpha in (0.1, 0.01, 1e-3):
    eig[f"tail indicator I(p<={alpha})"] = eigen_estimators((pA <= alpha).astype(np.float32))
    gc.collect()
# exceedance cluster sizes: mean number of models jointly exceeding, given at least one does
clus = {}
for z in (2.0, 2.5, 3.0, 3.4):
    pl = float(lz.sigma_to_p(z)); ex_ = (pA <= pl)
    any_ = ex_.any(axis=1)
    clus[z] = dict(mean_models_exceeding_given_any=float(ex_[any_].sum(axis=1).mean()) if any_.any() else np.nan,
                   M_over_Neff=float(is_lz.sum() / main[f"N_eff@{z}"]), n_toys_any=int(any_.sum()))
eigrows = [dict(variable=k, **v) for k, v in eig.items()]
for r_ in eigrows:
    for key in ("Nyholt_Cheverud", "Li_Ji", "participation_ratio", "Galwey", "Gao_99p5"):
        r_[f"Z_global@3.4_if_{key}"] = Zs(min(1, r_[key] * P_LZ))
dfe = pd.DataFrame(eigrows); dfe.to_csv(os.path.join(WORK, "eigenvalue_estimators.csv"), index=False)
print(dfe[["variable", "M", "Nyholt_Cheverud", "Li_Ji", "participation_ratio", "Galwey", "Gao_99p5", "lambda_max"]].to_string())
print("cluster sizes:", json.dumps(clus, indent=0, default=float))

# ----------------------------------------------------------------------------------------------
# 7. Gross-Vitells with upcrossings / Euler characteristic counted in the toys
# ----------------------------------------------------------------------------------------------
def zfield(p, q):
    """calibrated Z per toy/model; q0 = 0 -> -inf (below any positive threshold)."""
    Zc = stats.norm.isf(np.clip(p, 1e-12, 1))
    return np.where(q <= 0, -np.inf, Zc)

def gv_1d(Zf, thresholds=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), targets=(2.0, 2.5, 3.0, 3.4)):
    """Zf: (T, n) ordered scan. Upcrossings at c0 -> GV extrapolation; compare with direct max."""
    out = []
    zmax = Zf.max(axis=1)
    for c0 in thresholds:
        above = Zf > c0
        nup = (above[:, 1:] & ~above[:, :-1]).sum(axis=1) + above[:, 0]
        Nup = float(nup.mean())
        for c in targets:
            p_gv = float(stats.norm.sf(c) + Nup * np.exp(-(c ** 2 - c0 ** 2) / 2))
            p_dir = float((zmax >= c).mean())
            out.append(dict(c0=c0, N_up=Nup, c=c, p_GV=p_gv, N_eff_GV=neff(p_gv, float(stats.norm.sf(c))),
                            p_direct=p_dir, N_eff_direct=neff(p_dir, float(stats.norm.sf(c))) if p_dir > 0 else np.nan))
    return pd.DataFrame(out)

def euler_char_grid(Zf, shape, c):
    """Zf: (T, n) with n = prod(shape) laid out row-major on a 2D grid; Euler characteristic
    chi = V - E + F of the excursion set {Z > c} using 4-neighbour edges and 2x2 faces."""
    Aabove = (Zf > c).reshape(Zf.shape[0], *shape)
    V = Aabove.sum(axis=(1, 2))
    Eh = (Aabove[:, :, 1:] & Aabove[:, :, :-1]).sum(axis=(1, 2))
    Ev = (Aabove[:, 1:, :] & Aabove[:, :-1, :]).sum(axis=(1, 2))
    Fc = (Aabove[:, 1:, 1:] & Aabove[:, :-1, 1:] & Aabove[:, 1:, :-1] & Aabove[:, :-1, :-1]).sum(axis=(1, 2))
    return V - Eh - Ev + Fc

def gv_2d(Zf_list, shape, thresholds=(0.5, 1.0, 1.5, 2.0), targets=(2.0, 2.5, 3.0, 3.4)):
    """Sum of Euler characteristics over several (m, delta) grids (one per operator/isospin);
    fit E[chi(c)] = Phi_bar(c) * n_grids + (N1 + N2 c) exp(-c^2/2); extrapolate."""
    chi = {c0: float(sum(euler_char_grid(Zf, shape, c0).mean() for Zf in Zf_list)) for c0 in thresholds}
    ngrid = len(Zf_list)
    cc = np.array(thresholds); y = np.array([chi[c] - ngrid * stats.norm.sf(c) for c in thresholds]) * np.exp(cc ** 2 / 2)
    Amat = np.vstack([np.ones_like(cc), cc]).T
    (N1, N2), *_ = np.linalg.lstsq(Amat, y, rcond=None)
    zmax = np.max(np.hstack(Zf_list), axis=1)
    out = []
    for c in targets:
        p_gv = float(ngrid * stats.norm.sf(c) + (N1 + N2 * c) * np.exp(-c ** 2 / 2))
        p_dir = float((zmax >= c).mean())
        out.append(dict(c=c, p_GV=p_gv, N_eff_GV=neff(min(p_gv, 0.99), float(stats.norm.sf(c))), p_direct=p_dir,
                        N_eff_direct=neff(p_dir, float(stats.norm.sf(c))) if p_dir > 0 else np.nan))
    return pd.DataFrame(out), dict(N1=float(N1), N2=float(N2), chi=chi, n_grids=ngrid)

gvres = {}
# 1D: fine delta scan O1s 1 TeV (ordered in delta)
sc_idx = np.where(is_scan)[0]; sc_idx = sc_idx[np.argsort(delta[sc_idx])]
Zcal_scan = zfield(p_evA[:, sc_idx], q_evA[:, sc_idx]); Zasy_scan = np.sqrt(q_evA[:, sc_idx].astype(float))
g1c = gv_1d(Zcal_scan); g1c["metric"] = "calibrated Z"; g1c["scan"] = "delta scan O1s 1 TeV (41 pts, 0-400 keV)"
g1a = gv_1d(Zasy_scan); g1a["metric"] = "asymptotic sqrt(q0)"; g1a["scan"] = g1c["scan"].iloc[0]
# 1D: LZ's 8-point delta grid per (operator, isospin, mass): 12 scans of 8 points -> treat each as a 1D scan and combine via sum of upcrossings
def grid_block(opv, tauv, mv):
    idx = [np.where((cls == "inelastic") & (op == opv) & (tau == tauv) & (mass == mv) & (delta == d))[0] for d in lz.OSIG_DELTAS]
    return idx
# 2D: (m, delta) grids for each of O1s, O1v, O4s, O4v; the unphysical (400 GeV, 350 keV) cell is filled with -inf
Zc_blocks, Za_blocks = [], []
for opv in (1, 4):
    for tauv in ("s", "v"):
        Zc = np.full((N_TOYS, 3, 8), -np.inf); Za = np.zeros((N_TOYS, 3, 8))
        for i_m, mv in enumerate((400, 1000, 4000)):
            for i_d, d in enumerate(lz.OSIG_DELTAS):
                j = np.where((cls == "inelastic") & (op == opv) & (tau == tauv) & (mass == mv) & (delta == d))[0]
                if len(j):
                    Zc[:, i_m, i_d] = zfield(p_evA[:, j[0]], q_evA[:, j[0]]); Za[:, i_m, i_d] = np.sqrt(q_evA[:, j[0]].astype(float))
        Zc_blocks.append(Zc.reshape(N_TOYS, 24)); Za_blocks.append(Za.reshape(N_TOYS, 24))
g2c, fit2c = gv_2d(Zc_blocks, (3, 8)); g2c["metric"] = "calibrated Z"; g2c["scan"] = "inelastic (m,delta) grids O1s/O1v/O4s/O4v (4 x 3 x 8)"
g2a, fit2a = gv_2d(Za_blocks, (3, 8)); g2a["metric"] = "asymptotic sqrt(q0)"; g2a["scan"] = g2c["scan"].iloc[0]
dfg = pd.concat([g1c, g1a, g2c, g2a], ignore_index=True)
dfg.to_csv(os.path.join(WORK, "gross_vitells.csv"), index=False)
gvres["fit_2d_calibrated"] = fit2c; gvres["fit_2d_asymptotic"] = fit2a
print("GV 1D calibrated:\n", g1c[g1c.c == 3.4].to_string()); print("GV 1D asymptotic:\n", g1a[g1a.c == 3.4].to_string())
print("GV 2D calibrated:\n", g2c.to_string()); print("GV 2D asymptotic:\n", g2a.to_string())
# upcrossing scaling check: does N_up(c0) follow exp(-c0^2/2)?
up_scaling = {}
for lab, g in (("calibrated", g1c), ("asymptotic", g1a)):
    gg = g.drop_duplicates("c0")
    up_scaling[lab] = {float(c0): dict(N_up=float(n), N_up_times_exp=float(n * np.exp(c0 ** 2 / 2))) for c0, n in zip(gg.c0, gg.N_up)}
gvres["upcrossing_scaling_1d"] = up_scaling

# ----------------------------------------------------------------------------------------------
# 8. Threshold dependence: N_eff(Z), cost in sigma, extrapolation to 5 sigma
# ----------------------------------------------------------------------------------------------
zgrid = np.round(np.arange(1.0, 4.01, 0.1), 2)
pmin_lz = np.concatenate([p_evA[:, is_lz].min(axis=1), p_evB[:, is_lz].min(axis=1)])
pmin_corpus = np.concatenate([p_evA.min(axis=1), p_evB.min(axis=1)])
thr = []
for z in zgrid:
    pl = float(lz.sigma_to_p(z))
    for lab, pm in (("LZ-like", pmin_lz), ("corpus", pmin_corpus)):
        k = int((pm <= pl).sum()); pg = k / len(pm)
        thr.append(dict(space=lab, Z_local=z, p_local=pl, p_global=pg, n_exceed=k, Z_global=Zs(pg) if pg > 0 else np.nan,
                        N_eff=neff(pg, pl) if pg > 0 else np.nan, N_eff_err=np.sqrt(max(k, 1)) / len(pm) / pl,
                        cost_sigma=(z - Zs(pg)) if pg > 0 else np.nan))
dft = pd.DataFrame(thr); dft.to_csv(os.path.join(WORK, "threshold_dependence.csv"), index=False)
# fits: primary on the debiased N_eff(Z) at Z = 2, 2.5, 3, 3.4 (LZ-like); secondary on the raw A+B curve, Z in [2, 3.6]
# (i) Gross-Vitells form: p_g = Phi_bar(c) + N exp(-c^2/2)  ->  N_eff(c) = 1 + N exp(-c^2/2)/Phi_bar(c)  (~ 1 + N c sqrt(2 pi))
# (ii) power law N_eff = a c^k;  (iii) exponential N_eff = a exp(b c)
gfun = lambda c: np.exp(-c ** 2 / 2) / stats.norm.sf(c)
def fit_forms(z_, n_, e_):
    N_gv = float(np.sum((n_ - 1) * gfun(z_) / e_ ** 2) / np.sum(gfun(z_) ** 2 / e_ ** 2))
    k_pl, lna = np.polyfit(np.log(z_), np.log(n_), 1, w=n_ / e_)
    b_ex, lna_ex = np.polyfit(z_, np.log(n_), 1, w=n_ / e_)
    f = dict(GV_form_N=N_gv, powerlaw_a=float(np.exp(lna)), powerlaw_k=float(k_pl), exp_a=float(np.exp(lna_ex)), exp_b=float(b_ex), n_points=int(len(z_)))
    f["chi2_GV"] = float(np.sum(((1 + N_gv * gfun(z_) - n_) / e_) ** 2))
    f["chi2_power"] = float(np.sum(((np.exp(lna) * z_ ** k_pl - n_) / e_) ** 2))
    f["chi2_exp"] = float(np.sum(((np.exp(lna_ex) * np.exp(b_ex * z_) - n_) / e_) ** 2))
    return f
zz = np.array(ZC); nn = np.array([deb_main[z]["N_eff_debiased"] for z in ZC])
ee = np.array([max(deb_main[z]["N_eff_debiased_halfrange"], 0.05 * deb_main[z]["N_eff_debiased"]) for z in ZC])
d = dft[(dft.space == "LZ-like") & (dft.Z_local >= 2.0) & (dft.Z_local <= 3.6)]
fits = fit_forms(zz, nn, ee); fits_raw = fit_forms(d.Z_local.values, d.N_eff.values, d.N_eff_err.values)
def neff_model(c, kind, f=None):
    f = fits if f is None else f
    if kind == "GV": return 1 + f["GV_form_N"] * gfun(c)
    if kind == "power": return f["powerlaw_a"] * c ** f["powerlaw_k"]
    if kind == "exp": return f["exp_a"] * np.exp(f["exp_b"] * c)
ext = []
for c in (3.4, 4.0, 4.5, 5.0):
    pl = float(lz.sigma_to_p(c)); row = dict(Z_local=c, p_local=pl)
    for kind in ("GV", "power", "exp"):
        ne = float(neff_model(c, kind)); pg = 1 - (1 - pl) ** ne
        row[f"N_eff_{kind}"] = ne; row[f"Z_global_{kind}"] = Zs(pg); row[f"cost_{kind}"] = c - Zs(pg)
        ner = float(neff_model(c, kind, fits_raw)); pgr = 1 - (1 - pl) ** ner
        row[f"N_eff_{kind}_raw"] = ner; row[f"cost_{kind}_raw"] = c - Zs(pgr)
    for Msat, lab in ((287, "Bonferroni_287"), (30, "saturation_30_windows")):
        pg = 1 - (1 - pl) ** Msat; row[f"Z_global_{lab}"] = Zs(pg); row[f"cost_{lab}"] = c - Zs(pg)
    ext.append(row)
dfx5 = pd.DataFrame(ext); dfx5.to_csv(os.path.join(WORK, "extrapolation_5sigma.csv"), index=False)
print("fits (debiased):", fits); print("fits (raw):", fits_raw); print(dfx5.to_string())
# resolution-window count: number of 2-sigma-wide resolution windows in the ROI (sigma = 23 sqrt(E/248) keV)
n_windows = float(np.trapezoid(1 / (2 * p8.sigma_E(E_OBS)), E_OBS))

# ----------------------------------------------------------------------------------------------
# 9. Local Z of the data (248 keV event + 3 low-energy events) for all models, incl. corpus extras
# ----------------------------------------------------------------------------------------------
acc_pdf = comps["accidentals"][1]; cdf_acc = np.cumsum(acc_pdf) * DE
low = list(np.interp((np.arange(3) + 0.5) / 3, cdf_acc, E_OBS))
ev = np.array([[248.0] + low + [np.nan] * 16])
q_d, s_d = p8.q0_for_events(ev, Fk, b_dens)
p_d = calA.p(q_d)[0]
dd = pd.DataFrame(dict(model=names, cls=cls, mass=mass, delta=delta, q0=q_d[0], s_hat=s_d[0], p_local=p_d,
                       Z_local=[Zs(p) for p in p_d], Z_asymptotic=np.sqrt(q_d[0])))
dd.to_csv(os.path.join(WORK, "data_local_Z_all_models.csv"), index=False)
# global significance of the data's own best model in each space (own p_min vs. min-p distribution)
own = []
for k, mask in spaces.items():
    pmd = float(dd.p_local.values[mask].min()); best = names[mask][np.argmin(dd.p_local.values[mask])]
    pm = np.concatenate([p_evA[:, mask].min(axis=1), p_evB[:, mask].min(axis=1)])
    pg = float((pm <= pmd).mean())
    own.append(dict(space=k, best_model=best, Z_local_max=Zs(pmd), p_global=pg, Z_global=Zs(pg) if pg > 0 else np.inf))
dfo = pd.DataFrame(own); dfo.to_csv(os.path.join(WORK, "data_own_global.csv"), index=False)
print(dfo.to_string())
print("extras local Z (top):\n", dd[~np.isin(dd.cls, ["elastic", "inelastic"])].sort_values("Z_local", ascending=False).head(12).to_string())

# ----------------------------------------------------------------------------------------------
# 10. Bayesian trials factor (P027) vs likelihood-weighted effective model counts
# ----------------------------------------------------------------------------------------------
def flatten(x):
    if isinstance(x, dict): return [v for xx in x.values() for v in flatten(xx)]
    if isinstance(x, (list, tuple, np.ndarray)): return [v for xx in x for v in flatten(xx)]
    return [x]
lzZ = np.array([v for v in flatten(lz.LSIG) + flatten(lz.OSIG) if isinstance(v, (int, float)) and np.isfinite(v)], float)
def n_like(Zarr):
    """M / sum_m exp(-(Zmax^2 - Z_m^2)/2): equal-weight profile-likelihood analogue of B_max/B_mean."""
    q = Zarr ** 2; return float(len(q) / np.sum(np.exp(-(q.max() - q) / 2)))
bay = dict(LZ_tables_n_entries=int(len(lzZ)), LZ_tables_Zmax=float(lzZ.max()),
           N_like_LZ_tables_all=n_like(lzZ), N_like_LZ_tables_Zgt0=n_like(lzZ[lzZ > 0]),
           N_like_LZ_tables_scaled_293=float(293 / 616 * n_like(lzZ)),
           N_like_P008_data_LZlike_asymptoticZ=n_like(dd.Z_asymptotic.values[is_lz]),
           N_like_P008_data_LZlike_calibratedZ=n_like(dd.Z_local.values[is_lz]),
           N_like_P008_data_corpus_calibratedZ=n_like(dd.Z_local.values),
           n_models_within_1sigma_of_max_LZ=int((lzZ >= lzZ.max() - 1.0).sum()),
           n_models_Z_ge_3_LZ=int((lzZ >= 3.0).sum()),
           P027_Bmax_over_BDM=10.3, P027_BDM_uniform=16.4, P027_Bmax=171, P008_Neff=12.2, LZ_implied_Neff=13.9,
           this_work_Neff_3p4=float(main["N_eff@3.4"]), this_work_Neff_3p4_debiased=deb_main[3.4]["N_eff_debiased"],
           M_over_Neff_debiased=float(is_lz.sum() / deb_main[3.4]["N_eff_debiased"]),
           M_over_Bayesian_factor_P027=float(298 / 10.3), M_over_N_like_LZ_tables=float(612 / n_like(lzZ)),
           SBB_bound_from_global_p=float(-1 / (np.e * float(lz.sigma_to_p(main["Z_global@3.4"])) * np.log(float(lz.sigma_to_p(main["Z_global@3.4"]))))))
bay["Z_global_if_Bayesian_trials_factor"] = Zs(min(1, 10.3 * P_LZ))
print("Bayesian comparison:", json.dumps(bay, indent=1))

# ----------------------------------------------------------------------------------------------
# 11. Figures
# ----------------------------------------------------------------------------------------------
def style(ax):
    ax.grid(axis="x", color=C_GRID, lw=0.6); ax.set_axisbelow(True)

# Fig 1: global Z by method (local 3.4 sigma)
sub_gv2c = g2c[g2c.c == 3.4].iloc[0]; sub_gv2a = g2a[g2a.c == 3.4].iloc[0]
inel = dfs[dfs.space.str.startswith("inelastic")].iloc[0]
eg = dfe.set_index("variable")
methods = [
    ("Bonferroni, 616 models", Zs(616 * P_LZ), "bound"),
    ("Bonferroni, 293 distinct spectra", Zs(293 * P_LZ), "bound"),
    ("Šidák, 293", Zs(1 - (1 - P_LZ) ** 293), "bound"),
    ("Eigenvalue M_eff (Li–Ji, tail indicator p≤0.01)", Zs(min(1, eg.loc["tail indicator I(p<=0.01)", "Li_Ji"] * P_LZ)), "eigen"),
    ("Eigenvalue M_eff (participation ratio, tail p≤0.001)", Zs(min(1, eg.loc["tail indicator I(p<=0.001)", "participation_ratio"] * P_LZ)), "eigen"),
    ("Gross–Vitells, inelastic grids, asymptotic Z", Zs(sub_gv2a.p_GV), "gv"),
    ("Gross–Vitells, inelastic grids, calibrated Z", Zs(sub_gv2c.p_GV), "gv"),
    ("Toy min-p, inelastic grids (direct, debiased)", float(deb[inel["space"]][3.4]["Z_global_debiased"]), "toy"),
    ("Bayesian trials factor B_max/B_DM = 10.3 [P027]", bay["Z_global_if_Bayesian_trials_factor"], "bayes"),
    ("Toy min-p, LZ-like space, this work (N_cal → ∞)", float(deb_main[3.4]["Z_global_debiased"]), "toy"),
    ("Toy min-p, LZ-like space [P008, 300k toys]", 2.64, "toy"),
    ("LZ toy MC, 293 spectra", 2.6, "lz"),
    ("Toy min-p, corpus space, this work (N_cal → ∞)", float(deb_corp[3.4]["Z_global_debiased"]), "toy"),
]
colmap = dict(bound=C_MUTED, eigen=C_VIOLET, gv=C_AQUA, toy=C_BLUE, bayes=C_ORANGE, lz=C_YELLOW)
fig, ax = plt.subplots(figsize=(7.6, 5.6))
y = np.arange(len(methods))[::-1]
for yi, (lab, zval, kind) in zip(y, methods):
    ax.barh(yi, zval, color=colmap[kind], height=0.62)
    ax.text(zval + 0.04, yi, f"{zval:.2f}σ", va="center", fontsize=8, color=C_INK2)
ax.set_yticks(y); ax.set_yticklabels([m[0] for m in methods], fontsize=8)
ax.axvline(3.4, color=C_MUTED, ls=":", lw=1); ax.text(3.41, y[0] + 0.5, "local 3.4σ", fontsize=7.5, color=C_MUTED)
ax.set_xlim(0, 3.9); ax.set_xlabel("global significance at local 3.4σ [σ]"); style(ax)
ax.set_title("P071: global significance at local 3.4σ, by LEE method", fontsize=10, loc="left")
handles = [plt.Rectangle((0, 0), 1, 1, color=colmap[k]) for k in ("bound", "eigen", "gv", "toy", "bayes", "lz")]
ax.legend(handles, ["bounds", "eigenvalue M_eff", "Gross–Vitells", "toy min-p", "Bayesian", "LZ"], fontsize=7.5,
          loc="upper center", bbox_to_anchor=(0.35, -0.13), frameon=False, ncol=6)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_globalZ_by_method.png"), dpi=170); plt.close(fig)

# Fig 2: N_eff vs threshold and the LEE cost
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.9))
dl = dft[dft.space == "LZ-like"]; dc = dft[dft.space == "corpus"]
ax = axs[0]
ax.errorbar(dl.Z_local, dl.N_eff, yerr=dl.N_eff_err, fmt="o", ms=3.5, color=C_BLUE, lw=1, label="LZ-like space (toys, A+B)")
ax.plot(dc.Z_local, dc.N_eff, "s", ms=3, color=C_ORANGE, label="corpus space (toys)")
cc = np.linspace(1.0, 5.0, 200)
ax.errorbar(zz, nn, yerr=ee, fmt="D", ms=5, color=C_BLUE, mfc="white", lw=1, label="LZ-like, extrapolated to N_cal → ∞")
ax.plot(cc, neff_model(cc, "GV"), "-", color=C_AQUA, lw=1.5, label=f"GV form 1 + N e^(-c²/2)/Φ̄(c), N = {fits['GV_form_N']:.2f}")
ax.plot(cc, neff_model(cc, "power"), "--", color=C_VIOLET, lw=1.5, label=f"power law a c^k, k = {fits['powerlaw_k']:.2f}")
ax.plot(cc, neff_model(cc, "exp"), ":", color=C_RED, lw=1.5, label=f"exponential a e^(bc), b = {fits['exp_b']:.2f}")
ax.axhline(287, color=C_MUTED, lw=0.8, ls="-."); ax.text(3.75, 287 * 0.6, "Bonferroni: 287 distinct", fontsize=7, color=C_MUTED)
for zref, nref in ((2.0, 4.8), (2.5, 6.0), (3.0, 8.3), (3.4, 12.2)):
    ax.plot(zref, nref, "x", color=C_INK2, ms=6, mew=1.2)
ax.plot([], [], "x", color=C_INK2, label="P008 (300k toys)")
ax.set_yscale("log"); ax.set_ylim(1, 1000); ax.set_xlim(1.0, 5.05)
ax.set_xlabel("local significance threshold Z_local [σ]"); ax.set_ylabel("effective number of trials N_eff")
ax.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93)); ax.grid(color=C_GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("N_eff grows with threshold\n(raw points above 3.4σ are calibration-biased)", fontsize=8.5, loc="left")
ax = axs[1]
sel_l = (dl.Z_local >= 1.5) & (dl.Z_local <= 3.4); sel_c = (dc.Z_local >= 1.5) & (dc.Z_local <= 3.4)
ax.plot(dl.Z_local[sel_l], dl.cost_sigma[sel_l], "o", ms=3.5, color=C_BLUE, label="toys, LZ-like (raw)")
ax.plot(dc.Z_local[sel_c], dc.cost_sigma[sel_c], "s", ms=3, color=C_ORANGE, label="toys, corpus (raw)")
ax.plot(zz, zz - np.array([deb_main[z]["Z_global_debiased"] for z in ZC]), "D", ms=5, color=C_BLUE, mfc="white", label="LZ-like, N_cal → ∞")
ax.plot(zz, zz - np.array([deb_corp[z]["Z_global_debiased"] for z in ZC]), "D", ms=5, color=C_ORANGE, mfc="white", label="corpus, N_cal → ∞")
for kind, col, ls in (("GV", C_AQUA, "-"), ("power", C_VIOLET, "--"), ("exp", C_RED, ":")):
    pg = 1 - (1 - stats.norm.sf(cc)) ** neff_model(cc, kind)
    ax.plot(cc, cc - stats.norm.isf(np.clip(pg, 1e-15, 0.5)), ls, color=col, lw=1.5, label=f"{kind} fit")
pgb = 1 - (1 - stats.norm.sf(cc)) ** 287
ax.plot(cc, cc - stats.norm.isf(np.clip(pgb, 1e-15, 0.5)), "-.", color=C_MUTED, lw=0.8, label="Bonferroni 287")
ax.set_xlim(1.5, 5.05); ax.set_ylim(0, 1.6); ax.set_xlabel("local significance Z_local [σ]"); ax.set_ylabel("LEE cost  Z_local − Z_global [σ]")
ax.legend(fontsize=6.5, frameon=False, loc="lower right", ncol=2); ax.grid(color=C_GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("...but the cost in σ stays near 0.7σ", fontsize=9.5, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_neff_vs_threshold.png"), dpi=170); plt.close(fig)

# Fig 3: counterfactual search spaces
order = ["(c) single model O6s 1 TeV [L10/L4-like]", "(b1) 6 a-priori high-energy spectra", "(b2) 6 spectra with largest fraction above 200 keV",
         "(a) 14 isoscalar ops at 1 TeV + O1s/O4s inelastic 1 TeV x 8 delta (30)", "(a') 28 s+v ops at 1 TeV + O1/O4 s+v inelastic 1 TeV x 8 delta (60)",
         "heavy elastic m>=100 GeV (140)", "inelastic O1/O4 grid (96)", "LZ-like full space (456; LZ 293 distinct)",
         "+ continuous delta scan [P021] (497)", "+ light mediators [P051]", "+ isospin ratios [P032]", "+ halo/date variants [P018,P006] = corpus space"]
short = ["single model (1)", "6 a-priori spectra", "6 highest high-E fraction", "1 TeV isoscalar + inelastic (30)", "1 TeV s+v + inelastic (60)",
         "heavy elastic (140)", "inelastic grid (96)", "LZ-like space (456 / 293)", "+ δ scan [P021]", "+ light mediators [P051]",
         "+ isospin ratios [P032]", "+ halo/date variants = corpus"]
ds = dfs.set_index("space").loc[order]
fig, ax = plt.subplots(figsize=(7.6, 5.0))
y = np.arange(len(order))[::-1]
cols = [C_AQUA] * 5 + [C_BLUE] * 3 + [C_ORANGE] * 4
for yi, (z, ne, nm, c) in enumerate(zip(ds["Z_global_deb@3.4"], ds["N_eff_deb@3.4"], ds["n_models"], cols)):
    ax.barh(y[yi], z, color=c, height=0.62)
    ax.text(z + 0.03, y[yi], f"{z:.2f}σ   N_eff = {ne:.1f}", va="center", fontsize=7.8, color=C_INK2)
ax.set_yticks(y); ax.set_yticklabels(short, fontsize=8)
ax.axvline(3.4, color=C_MUTED, ls=":", lw=1); ax.axvline(2.6, color=C_YELLOW, ls="--", lw=1)
ax.text(3.43, y[0] + 0.45, "local 3.4σ", fontsize=7.5, color=C_MUTED); ax.text(2.2, y[0] + 0.45, "LZ global 2.6σ", fontsize=7.5, color=C_INK2)
ax.set_xlim(0, 4.7); ax.set_xlabel("global significance at local 3.4σ [σ]  (toy min-p, extrapolated to N_cal → ∞)"); style(ax)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_AQUA, C_BLUE, C_ORANGE)]
ax.legend(handles, ["pre-registration counterfactuals", "LZ-like spaces", "community additions (cumulative)"], fontsize=7.5, frameon=False,
          loc="upper center", bbox_to_anchor=(0.4, -0.14), ncol=3)
ax.set_title("P071: what the search space does to the same 3.4σ event", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_counterfactuals.png"), dpi=170); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# 12. Results JSON
# ----------------------------------------------------------------------------------------------
out = dict(
    N_TOYS_per_configuration=N_TOYS, configurations=4, models_in_toys=int(M), models_LZlike=int(is_lz.sum()), models_extra=int((~is_lz).sum()),
    extra_breakdown={k: int((cls == k).sum()) for k in ("deltascan", "lightmed", "isospin", "halo")},
    background_total=float(B), background_above_200keV=float(b_dens[E_OBS > 200].sum() * DE), isospin_node_r0_keV=node_r0,
    mean_events_per_toy=float(np.mean(np.concatenate([n_evA, n_evB]))),
    main_LZlike={k: (float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v) for k, v in main.items()},
    corpus={k: (float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v) for k, v in corpus.items()},
    eigenvalue_estimators=eig, exceedance_clusters=clus, gross_vitells=gvres,
    gv_1d_at_3p4={m_: {str(c0): dict(N_eff_GV=float(g[(g.c == 3.4) & (g.c0 == c0)].N_eff_GV.iloc[0]), N_eff_direct=float(g[(g.c == 3.4) & (g.c0 == c0)].N_eff_direct.iloc[0]))
                        for c0 in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)} for m_, g in (("calibrated", g1c), ("asymptotic", g1a))},
    calibration_size_dependence=dfc.to_dict(orient="records"), debiased=deb, corpus_over_LZlike_Neff_ratio=float(ratio_corpus),
    honest_global_after_corpus=honest, threshold_fits_raw=fits_raw,
    gv_2d_at_3p4=dict(calibrated=dict(N_eff_GV=float(sub_gv2c.N_eff_GV), N_eff_direct=float(sub_gv2c.N_eff_direct)),
                      asymptotic=dict(N_eff_GV=float(sub_gv2a.N_eff_GV), N_eff_direct=float(sub_gv2a.N_eff_direct))),
    threshold_fits=fits, n_resolution_windows_in_ROI=n_windows, extrapolation=dfx5.to_dict(orient="records"),
    bayesian=bay, top6_by_frac_above_200=list(map(str, top6)), six_apriori=SIX_APRIORI,
    data_own_global=dfo.to_dict(orient="records"), runtime_s=time.time() - T0)
json.dump(out, open(os.path.join(WORK, "results.json"), "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
log("done")
