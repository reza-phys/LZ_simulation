"""
P004 -- Wall MSSI as the origin of the LZ 248 keV event: the mismodelling factor k required
and what the paper's own MSSI sidebands allow.

Run from the simulation root:  .venv/bin/python output/code/P004_wall_mssi.py

Sections
 1. Digitise the MSSI (and total-model) histograms of Fig. 5 bottom panel (S1c > 500 phd) by
    pixel colour, to obtain the fraction f_nb of wall MSSI that lands at S1c>500 phd and within
    +-2 sigma of the NR median (the event sits at -1.5 sigma).  Cross-check: the digitised total
    model must integrate to ~0.0106 (Fig. 5 caption).
 2. Poisson likelihood for a common wall-MSSI scale factor k over the wall bins of Table S? (MSSI
    comparison table): WS ROI 5.4 t, HE SB 4.7 t, HE SB 5.4 t; science and prompt.  MLE, likelihood-
    ratio and Bayesian (flat prior) upper limits; variants (science only, prompt only, 20 % wall/RFR
    split nuisance, prompt-bin contamination, pessimistic misclassification).
 3. P(>=1 wall MSSI | k) in the ROI and in the event's neighbourhood; k needed for 10 %/50 %;
    Bayes factor "k free within sideband posterior" vs "k = 1".
 4. Topology: Compton kinematics (Klein-Nishina) + LXe attenuation for a 12 keV deposit 27 cm from
    the wall followed by absorption in a <= 3 mm dead layer; suppression relative to a generic
    wall MSSI; position penalty f_pos.
 5. Veto-efficiency propagation (94 +- 2 %).
All numbers are written to output/work/P004/*.json|csv and the figure to output/work/P004/figures/.
"""
import json, os, sys
import numpy as np
from scipy import stats, optimize, integrate
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "output/code")
from common import lzcommon as lz  # noqa: E402  (used for LZ numbers and poisson helper)

OUT = "output/work/P004"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
R = {}  # results dictionary

# --------------------------------------------------------------------------------------
# 0. Inputs from the LZ paper (arXiv:2609.02823)
# --------------------------------------------------------------------------------------
# Supplement Table "Comparison of simulated and observed MSSI counts" (tab:MSSI_comp)
# (region, type, volume): (sim_sci, obs_sci, sim_prompt, obs_prompt)
TABLE = {
    ("WS ROI", "Wall", 4.7): (0.0048, None, 0.09, None),   # blind search bin: excluded from fit
    ("WS ROI", "RFR", 4.7): (0.0001, None, 0.10, None),
    ("WS ROI", "Wall", 5.4): (0.03, 0, 0.1, 1),
    ("WS ROI", "RFR", 5.4): (0.003, 0, 1.9, 2),
    ("HE SB", "Wall", 4.7): (0.1, 0, 0.3, 1),
    ("HE SB", "RFR", 4.7): (0.5, 0, 0.5, 0),
    ("HE SB", "Wall", 5.4): (0.5, 0, 2.2, 0),
    ("HE SB", "RFR", 5.4): (21.5, 18, 5.2, 3),
}
WALL_SCI_47 = 0.0048          # predicted wall MSSI, science sample, 4.7 t, WS ROI
MSSI_SCI_TOTAL = 4.9e-3       # Table I (science sample), pre-fit
VETO_EFF, VETO_EFF_ERR = 0.94, 0.02
FIG5_TOTAL_BOTTOM = 0.0106    # Fig. 5 caption, S1c>500 phd panel
EVENT_SIGMA_NR = -1.5         # event location relative to NR median in units of sigma_NR
E1_KEV, E1_ERR = 12.0, 2.0    # MSSI decomposition of the event (supplement)
E2_WALL_KEV, E2_ERR = 77.0, 7.0
D_WALL_CM = 26.9              # distance of the event from the true wall
D_CATHODE_CM = 26.4
R_TPC_CM = 72.8               # TPC radius (recalled, likely: LZ active radius 72.8 cm)
FV_STANDOFF_CM = 8.0          # minimum stand-off of FV from true wall
FV_MEAN_STANDOFF_CM = 10.7
DEAD_LAYER_CM = 0.3
RFR_DEPTH_CM = 13.75

# --------------------------------------------------------------------------------------
# 1. Digitise Fig. 5 bottom panel
# --------------------------------------------------------------------------------------
def digitise_fig5(png="inputs/figures_png/Fig5_NR_distance_3panels_no_sig.png"):
    im = np.asarray(Image.open(png).convert("RGB")).astype(int)
    # frame of the bottom panel found from long dark lines (see details.md): rows 2103..2905, cols 219..2409
    y_top, y_bot = 2103, 2905      # frame rows of the bottom panel
    x_l, x_r = 219, 2409           # frame columns: -8 sigma and +8 sigma (x tick marks at 492,766,...,2135 = -6..+6)
    # y tick marks (detected as short dark segments on the left frame): rows 2131, 2324, 2518, 2711 for
    # 10^2, 10^0, 10^-2, 10^-4  -> 96.75 px per decade; 10^-6 at 2905 (frame bottom)
    Y_TICK_2, PX_PER_DECADE = 2131.0, (2711.0 - 2131.0) / 6.0
    def y_to_val(y):  # log10 value
        return 2.0 - (y - Y_TICK_2) / PX_PER_DECADE
    def x_to_sig(x):
        return -8.0 + 16.0 * (x - x_l) / (x_r - x_l)
    colours = {"MSSI": (2, 81, 128), "Total": (0, 0, 255), "Accidentals": (255, 220, 61),
               "NRs": (124, 174, 0), "ERs": (0, 194, 249), "Internal": (255, 0, 255)}
    edges = np.arange(-8, 8.01, 0.5)
    out = {}
    for name, col in colours.items():
        col = np.array(col)
        sub = im[y_top + 3:y_bot - 2, x_l + 3:x_r - 2]
        mask = (np.abs(sub - col).sum(axis=2) < 60)
        vals = np.full(len(edges) - 1, np.nan)
        for i in range(len(edges) - 1):
            # sample the middle 60 % of each bin (avoid vertical step lines)
            xa = int(x_l + (edges[i] + 8 + 0.1) / 16 * (x_r - x_l))
            xb = int(x_l + (edges[i] + 8 + 0.4) / 16 * (x_r - x_l))
            ys = []
            for x in range(xa, xb):
                yy = np.where(mask[:, x - (x_l + 3)])[0]
                if len(yy):
                    ys.append(yy.min() + y_top + 3)   # top-most pixel of the line = the histogram level
            if ys:
                vals[i] = 10 ** y_to_val(np.median(ys))
        out[name] = vals
    return edges, out

edges, hist = digitise_fig5()
centres = 0.5 * (edges[1:] + edges[:-1])
tot = np.nan_to_num(hist["Total"])
mssi = np.nan_to_num(hist["MSSI"])
# Where the total line is hidden behind another line of identical level the top-most pixel logic still
# finds the blue line; where the MSSI line is hidden under the total (not the case here) it would be lost.
R["fig5_digitised"] = {"sigma_bin_low": edges[:-1].tolist(), "total": tot.tolist(), "mssi": mssi.tolist(),
                       "accidentals": np.nan_to_num(hist["Accidentals"]).tolist(),
                       "nrs": np.nan_to_num(hist["NRs"]).tolist()}
tot_int = tot.sum()          # "events / bin width" with 0.5 sigma bins: values are per bin
R["fig5_total_integral_digitised"] = float(tot_int)
R["fig5_total_caption"] = FIG5_TOTAL_BOTTOM
R["fig5_total_ratio"] = float(tot_int / FIG5_TOTAL_BOTTOM)
# the y label is "Events / bin width" -- if per unit sigma, integral = sum*0.5.  Test both:
R["fig5_total_integral_times_binwidth"] = float(tot_int * 0.5)

def frac_window(arr, lo, hi):
    sel = (edges[:-1] >= lo - 1e-9) & (edges[1:] <= hi + 1e-9)
    return arr[sel].sum()

mssi_panel = mssi.sum()
mssi_pm2 = frac_window(mssi, -2.0, 2.0)
mssi_m3p1 = frac_window(mssi, -3.0, 1.0)      # asymmetric window around -1.5 sigma (event bin +- 2 bins)
mssi_bin_event = frac_window(mssi, -2.0, -1.5)  # the bin containing the event (-1.5 is the bin edge; plotted at -1.75)
# decide the normalisation of "events / bin width": use the total-model cross-check
norm = 1.0 if abs(R["fig5_total_ratio"] - 1) < abs(R["fig5_total_integral_times_binwidth"] / FIG5_TOTAL_BOTTOM - 1) else 0.5
R["fig5_norm_factor_used"] = norm
R["mssi_S1c_gt500_counts"] = float(mssi_panel * norm)
R["mssi_S1c_gt500_pm2sig_counts"] = float(mssi_pm2 * norm)
R["mssi_S1c_gt500_m3p1_counts"] = float(mssi_m3p1 * norm)
R["mssi_event_bin_counts"] = float(mssi_bin_event * norm)
R["total_model_pm2sig_counts"] = float(frac_window(tot, -2, 2) * norm)
# fractions of the whole-ROI wall MSSI (0.0048; RFR 0.0001 is negligible and in any case follows the same PDF here)
f_panel = mssi_panel * norm / MSSI_SCI_TOTAL
f_nb = mssi_pm2 * norm / MSSI_SCI_TOTAL
f_nb_alt = mssi_m3p1 * norm / MSSI_SCI_TOTAL
R["f_S1c_gt500"] = float(f_panel)
R["f_nb_pm2sigma"] = float(f_nb)
R["f_nb_m3p1sigma"] = float(f_nb_alt)
# reading-uncertainty range: pixel-level +-1 px in log space is ~ +-0.01 dex (negligible); the dominant
# systematic is which bins are attributed (line overlaps) -> quote a factor-1.5 range, plus the Fig. S1a
# contour reading as an independent estimate (see details.md): f_nb in [f_nb/1.5, 1.5 f_nb] and cap 0.15.
F_NB_RANGE = (f_nb / 1.5, min(1.5 * f_nb, 0.15))
R["f_nb_range"] = [float(F_NB_RANGE[0]), float(F_NB_RANGE[1])]

# --------------------------------------------------------------------------------------
# 2. Sideband likelihood for k
# --------------------------------------------------------------------------------------
wall_bins = [(key, v) for key, v in TABLE.items() if key[1] == "Wall" and v[1] is not None]
sci_pred = np.array([v[0] for _, v in wall_bins]); sci_obs = np.array([v[1] for _, v in wall_bins])
pr_pred = np.array([v[2] for _, v in wall_bins]); pr_obs = np.array([v[3] for _, v in wall_bins])
rfr_bins = [(key, v) for key, v in TABLE.items() if key[1] == "RFR" and v[1] is not None]
rfr_sci_pred = np.array([v[0] for _, v in rfr_bins]); rfr_sci_obs = np.array([v[1] for _, v in rfr_bins])
rfr_pr_pred = np.array([v[2] for _, v in rfr_bins]); rfr_pr_obs = np.array([v[3] for _, v in rfr_bins])
R["wall_bins"] = [{"region": k[0], "volume_t": k[2], "sim_sci": v[0], "obs_sci": v[1],
                   "sim_prompt": v[2], "obs_prompt": v[3]} for k, v in wall_bins]
R["wall_sum_pred_sci"] = float(sci_pred.sum()); R["wall_sum_obs_sci"] = int(sci_obs.sum())
R["wall_sum_pred_prompt"] = float(pr_pred.sum()); R["wall_sum_obs_prompt"] = int(pr_obs.sum())

kgrid = np.linspace(1e-4, 12, 24001)

def nll_poisson(k, pred, obs, bkg=0.0):
    mu = k * pred + bkg
    return np.sum(mu - obs * np.log(mu))

def scan(pred, obs, bkg=0.0, label=""):
    nll = np.array([nll_poisson(k, pred, obs, bkg) for k in kgrid])
    nll -= nll.min()
    kml = kgrid[nll.argmin()]
    # likelihood-ratio one-sided upper limits: -2 dlnL = 1 (68%: 0.99 -> use chi2 1 dof one-sided CL)
    def lr_ul(cl):
        thr = stats.chi2.ppf(2 * cl - 1, 1) / 2  # one-sided: P(q<thr)=2cl-1 -> delta lnL
        above = kgrid[(kgrid > kml) & (nll > thr)]
        return float(above.min()) if len(above) else np.inf
    # Bayesian flat prior on k>=0
    L = np.exp(-nll); post = L / np.trapezoid(L, kgrid); cdf = np.concatenate([[0], np.cumsum(0.5 * (post[1:] + post[:-1]) * np.diff(kgrid))])
    def bayes_ul(cl):
        return float(np.interp(cl, cdf, kgrid))
    # exact Poisson (Neyman one-sided, total counts) as a cross-check where counts are summed
    res = {"label": label, "k_ml": float(kml),
           "ul68_LR": lr_ul(0.68), "ul90_LR": lr_ul(0.90), "ul95_LR": lr_ul(0.95),
           "ul68_Bayes": bayes_ul(0.68), "ul90_Bayes": bayes_ul(0.90), "ul95_Bayes": bayes_ul(0.95),
           "nll_at_k1": float(np.interp(1.0, kgrid, nll)), "sum_pred": float(np.sum(pred)), "sum_obs": int(np.sum(obs))}
    return res, nll, post

variants = {}
allpred = np.concatenate([sci_pred, pr_pred]); allobs = np.concatenate([sci_obs, pr_obs])
variants["all_wall"], nll_all, post_all = scan(allpred, allobs, 0.0, "all 6 wall bins (sci+prompt)")
variants["science_only"], nll_sci, post_sci = scan(sci_pred, sci_obs, 0.0, "3 wall science bins")
variants["prompt_only"], nll_pr, post_pr = scan(pr_pred, pr_obs, 0.0, "3 wall prompt bins")
# prompt-bin contamination by non-MSSI (detector gamma singles, accidentals): add 0.1 events per prompt bin
contam = np.concatenate([np.zeros(3), 0.1 * np.ones(3)])
variants["prompt_contam_0p1"], _, _ = scan(allpred, allobs, contam, "all wall bins, +0.1 non-MSSI per prompt bin")
contam2 = np.concatenate([np.zeros(3), 0.3 * np.ones(3)])
variants["prompt_contam_0p3"], _, _ = scan(allpred, allobs, contam2, "all wall bins, +0.3 non-MSSI per prompt bin")

# 20 % uncertainty on the simulated wall/RFR split: wall_i -> wall_i (1+eps), eps ~ N(0,0.2), profiled
def nll_split(k, eps):
    pred = allpred * (1 + eps)
    return nll_poisson(k, pred, allobs) + 0.5 * (eps / 0.2) ** 2
nll_s = np.array([optimize.minimize_scalar(lambda e: nll_split(k, e), bounds=(-0.8, 2.0), method="bounded").fun for k in kgrid[::10]])
kg10 = kgrid[::10]; nll_s -= nll_s.min(); kml_s = kg10[nll_s.argmin()]
def ul_from(nll, kg, kml, cl):
    thr = stats.chi2.ppf(2 * cl - 1, 1) / 2
    ab = kg[(kg > kml) & (nll > thr)]
    return float(ab.min()) if len(ab) else np.inf
Ls = np.exp(-nll_s); ps = Ls / np.trapezoid(Ls, kg10); cdfs = np.concatenate([[0], np.cumsum(0.5 * (ps[1:] + ps[:-1]) * np.diff(kg10))])
variants["split_20pct"] = {"label": "all wall bins, 20% wall/RFR split nuisance profiled", "k_ml": float(kml_s),
                           "ul68_LR": ul_from(nll_s, kg10, kml_s, .68), "ul90_LR": ul_from(nll_s, kg10, kml_s, .90),
                           "ul95_LR": ul_from(nll_s, kg10, kml_s, .95),
                           "ul68_Bayes": float(np.interp(.68, cdfs, kg10)), "ul90_Bayes": float(np.interp(.90, cdfs, kg10)),
                           "ul95_Bayes": float(np.interp(.95, cdfs, kg10))}
# pessimistic mis-classification extreme: 20 % of the OBSERVED RFR events were really wall events
pess_obs = np.concatenate([sci_obs + 0.2 * rfr_sci_obs, pr_obs + 0.2 * rfr_pr_obs])
variants["pessimistic_obs_misclass"], _, _ = scan(allpred, pess_obs, 0.0, "20% of observed RFR events re-assigned to wall (extreme)")
# a k=1 check: exact Poisson p-value for the observed wall totals under k=1
variants["all_wall"]["p_obs_le_2_given_k1"] = float(stats.poisson.cdf(2, allpred.sum()))
variants["all_wall"]["p_obs_0_sci_given_k1"] = float(stats.poisson.pmf(0, sci_pred.sum()))
R["k_variants"] = variants
# the paper's own 100 % uncertainty: Gaussian(1,1) truncated at 0 -> 95 % one-sided UL
R["k_paper_prior_95"] = float(stats.truncnorm.ppf(0.95, 0, np.inf, loc=1, scale=1))

# --------------------------------------------------------------------------------------
# 3. P(>=1) as a function of k; k required; Bayes factor
# --------------------------------------------------------------------------------------
def p_ge1(mu):
    return 1 - np.exp(-mu)
mu_roi = WALL_SCI_47 * kgrid
R["P_ge1_ROI_k1"] = float(p_ge1(WALL_SCI_47))
R["P_ge1_nb_k1"] = float(p_ge1(WALL_SCI_47 * f_nb))
R["P_ge1_nb_k1_range"] = [float(p_ge1(WALL_SCI_47 * F_NB_RANGE[0])), float(p_ge1(WALL_SCI_47 * F_NB_RANGE[1]))]
def k_for(P, f):
    return -np.log(1 - P) / (WALL_SCI_47 * f)
R["k_required"] = {
    "ROI_10pct": float(k_for(0.10, 1.0)), "ROI_50pct": float(k_for(0.50, 1.0)),
    "S1c_gt500_10pct": float(k_for(0.10, f_panel)), "S1c_gt500_50pct": float(k_for(0.50, f_panel)),
    "nb_10pct": float(k_for(0.10, f_nb)), "nb_50pct": float(k_for(0.50, f_nb)),
    "nb_10pct_range": [float(k_for(0.10, F_NB_RANGE[1])), float(k_for(0.10, F_NB_RANGE[0]))],
    "nb_50pct_range": [float(k_for(0.50, F_NB_RANGE[1])), float(k_for(0.50, F_NB_RANGE[0]))],
}
k95 = variants["all_wall"]["ul95_LR"]
R["ratio_k_required_nb10_over_k95"] = float(R["k_required"]["nb_10pct"] / k95)
R["ratio_k_required_ROI10_over_k95"] = float(R["k_required"]["ROI_10pct"] / k95)
# P(>=1) at the allowed k (95 %)
R["P_ge1_nb_at_k95"] = float(p_ge1(WALL_SCI_47 * f_nb * k95))
R["P_ge1_ROI_at_k95"] = float(p_ge1(WALL_SCI_47 * k95))
R["P_ge1_nb_at_k_pess"] = float(p_ge1(WALL_SCI_47 * f_nb * variants["pessimistic_obs_misclass"]["ul95_LR"]))
# Bayes factor: H_free: k ~ posterior from sidebands (flat prior); H_1: k=1; data: exactly 1 event in the neighbourhood
def like_event(k, f=f_nb):
    mu = WALL_SCI_47 * f * k
    return mu * np.exp(-mu)
BF_free_vs_1 = np.trapezoid(like_event(kgrid) * post_all, kgrid) / like_event(1.0)
# with the LZ prior (truncated normal 1 +- 1) instead of the sideband posterior
prior_lz = stats.truncnorm.pdf(kgrid, 0, np.inf, loc=1, scale=1)
BF_lzprior_vs_1 = np.trapezoid(like_event(kgrid) * prior_lz, kgrid) / like_event(1.0)
# sideband-updated LZ prior
post_lz = prior_lz * np.exp(-nll_all); post_lz /= np.trapezoid(post_lz, kgrid)
BF_lzpost_vs_1 = np.trapezoid(like_event(kgrid) * post_lz, kgrid) / like_event(1.0)
R["bayes_factors"] = {"k_free_sideband_posterior_vs_k1": float(BF_free_vs_1),
                      "k_LZ_prior_trunc_normal_vs_k1": float(BF_lzprior_vs_1),
                      "k_LZ_prior_x_sideband_vs_k1": float(BF_lzpost_vs_1),
                      "posterior_mean_k_flat": float(np.trapezoid(kgrid * post_all, kgrid)),
                      "posterior_mean_k_LZprior_x_SB": float(np.trapezoid(kgrid * post_lz, kgrid))}
# Odds of "wall MSSI" vs "as-modelled everything else" for the neighbourhood: mu_other in the +-2 sigma window
mu_other_nb = R["total_model_pm2sig_counts"] - R["mssi_S1c_gt500_pm2sig_counts"]
R["mu_other_backgrounds_pm2sig"] = float(mu_other_nb)
R["mssi_share_of_pm2sig_model_k1"] = float(R["mssi_S1c_gt500_pm2sig_counts"] / R["total_model_pm2sig_counts"])
R["mssi_share_of_pm2sig_model_k95"] = float(k95 * R["mssi_S1c_gt500_pm2sig_counts"] / (k95 * R["mssi_S1c_gt500_pm2sig_counts"] + mu_other_nb))

# --------------------------------------------------------------------------------------
# 4. Topology: Compton kinematics and attenuation
# --------------------------------------------------------------------------------------
ME = 510.999  # keV, electron mass (certain)
def T_of_theta(E, th):
    Ep = E / (1 + (E / ME) * (1 - np.cos(th)))
    return E - Ep
def theta_of_T(E, T):
    Ep = E - T
    c = 1 - (E / Ep - 1) * ME / E
    return np.arccos(np.clip(c, -1, 1))
def kn_dsdo(E, th):  # Klein-Nishina, arbitrary units (r_e^2/2 dropped) -- formula certain
    Ep = E / (1 + (E / ME) * (1 - np.cos(th)))
    r = Ep / E
    return r ** 2 * (r + 1 / r - np.sin(th) ** 2)
def kn_frac_T(E, Tlo, Thi):
    tot, _ = integrate.quad(lambda t: kn_dsdo(E, t) * np.sin(t), 0, np.pi)
    a, b = theta_of_T(E, Tlo), theta_of_T(E, Thi)
    if Thi >= T_of_theta(E, np.pi):
        b = np.pi
    part, _ = integrate.quad(lambda t: kn_dsdo(E, t) * np.sin(t), a, b)
    return part / tot
# Recalled LXe total attenuation lengths (density 2.9 g/cm3), NIST-XCOM-like values, reliability "likely" (+-20 %):
#  E [keV]: 122 -> 0.3 cm (paper: <4 mm), 200 -> 1.0, 300 -> 2.0, 500 -> 3.7, 662 -> 4.7, 1000 -> 6.0, 1500 -> 7.6,
#  1764 -> 8.3, 2615 -> 9.8
ATT_E = np.array([122, 200, 300, 500, 662, 1000, 1500, 1764, 2615.])
ATT_L = np.array([0.30, 1.0, 2.0, 3.7, 4.7, 6.0, 7.6, 8.3, 9.8])
def lam(E):
    return np.exp(np.interp(np.log(E), np.log(ATT_E), np.log(ATT_L)))

# minimal in-LXe path lengths for the event topology.
# Ordering B (wall first): dead-layer scatter -> >=26.9 cm to the event -> 12 keV forward scatter -> exit.
# A straight line through a point at r=45.9 cm in a cylinder of R=72.8 cm has horizontal chord >= 2 sqrt(R^2-r^2);
# the shortest escape is through the cathode (26.4 cm below) -> total >= min over dip angle, computed here:
r_ev = R_TPC_CM - D_WALL_CM
def total_path(dz):
    leg1 = np.hypot(D_WALL_CM, dz)               # wall (dead layer) -> event, horizontal distance >= 26.9
    leg2 = D_CATHODE_CM * np.sqrt(1 + (D_WALL_CM / dz) ** 2)   # event -> cathode plane along the same line
    return leg1 + leg2
res = optimize.minimize_scalar(total_path, bounds=(1, 200), method="bounded")
L_MIN_GEOM = float(res.fun)                      # ~75 cm (plus 13.75 cm RFR to leave without a third S1)
L_CHORD = float(2 * np.sqrt(R_TPC_CM ** 2 - r_ev ** 2))
L_CONS = 2 * D_WALL_CM                           # 53.8 cm: naive "in and out" bound
L_GENERIC = 2 * FV_MEAN_STANDOFF_CM              # ~21 cm for a generic FV-edge wall MSSI
R["paths_cm"] = {"L_conservative_in_out": L_CONS, "L_min_geometric_via_cathode": L_MIN_GEOM,
                 "dz_at_min": float(res.x), "L_horizontal_chord": L_CHORD, "L_generic_wall_MSSI": L_GENERIC,
                 "RFR_extra": RFR_DEPTH_CM}
topo = []
for E in [352., 500., 609., 1000., 1461., 1764., 2615.]:
    l = float(lam(E))
    th12 = float(np.degrees(theta_of_T(E, E1_KEV)))
    th77 = float(np.degrees(theta_of_T(E, E2_WALL_KEV))) if E2_WALL_KEV < T_of_theta(E, np.pi) else float("nan")
    f12 = kn_frac_T(E, E1_KEV - E1_ERR, E1_KEV + E1_ERR)
    Tmax = T_of_theta(E, np.pi)
    f77 = kn_frac_T(E, E2_WALL_KEV - E2_ERR, min(E2_WALL_KEV + E2_ERR, Tmax)) if E2_WALL_KEV - E2_ERR < Tmax else 0.0
    p_dead = 1 - np.exp(-DEAD_LAYER_CM / l)      # interaction probability in a 3 mm dead layer
    p_cons = np.exp(-L_CONS / l); p_geom = np.exp(-L_MIN_GEOM / l); p_gen = np.exp(-L_GENERIC / l)
    topo.append({"E_keV": E, "lambda_cm": l, "theta_12keV_deg": th12, "theta_77keV_deg": th77,
                 "KN_frac_T_10_14keV": f12, "KN_frac_T_70_84keV": f77, "P_interact_3mm": p_dead,
                 "P_traverse_54cm": p_cons, "P_traverse_75cm": p_geom, "P_traverse_generic_21cm": p_gen,
                 "suppression_vs_generic_54cm": p_cons / p_gen, "suppression_vs_generic_75cm": p_geom / p_gen})
R["topology"] = topo
# position penalty f_pos: fraction of wall-MSSI FV scatters at distance d >= 25 cm from the true wall,
# toy radial pdf  p(d) ∝ (R-d) exp(-n d/λ), n=1 (one attenuated leg) or 2 (both legs), d >= 8 cm
def f_pos(l, n, dmin=FV_STANDOFF_CM, dcut=25.0):
    f = lambda d: (R_TPC_CM - d) * np.exp(-n * d / l)
    num, _ = integrate.quad(f, dcut, R_TPC_CM); den, _ = integrate.quad(f, dmin, R_TPC_CM)
    return num / den
fpos = {f"lambda_{l}cm_n{n}": float(f_pos(l, n)) for l in [3.7, 6.0, 8.3, 9.8] for n in [1, 2]}
R["f_pos"] = fpos
R["f_pos_range"] = [min(fpos.values()), max(fpos.values())]
# k required once the position penalty is included (P=10 % in the neighbourhood AND at d>=25 cm)
R["k_required_nb10_with_fpos_range"] = [float(k_for(0.10, f_nb) / max(fpos.values())), float(k_for(0.10, f_nb) / min(fpos.values()))]

# --------------------------------------------------------------------------------------
# 5. Veto efficiency propagation
# --------------------------------------------------------------------------------------
untag = 1 - VETO_EFF
R["veto"] = {"untagged_fraction": untag, "untagged_fraction_err": VETO_EFF_ERR,
             "relative_err_on_science_prediction": VETO_EFF_ERR / untag,
             "wall_sci_47_range_1sigma": [WALL_SCI_47 * (untag - VETO_EFF_ERR) / untag, WALL_SCI_47 * (untag + VETO_EFF_ERR) / untag],
             "implied_eff_from_table_47_wall": TABLE[("WS ROI", "Wall", 4.7)][2] / (TABLE[("WS ROI", "Wall", 4.7)][2] + WALL_SCI_47),
             "implied_eff_HE_SB_54_wall": 2.2 / 2.7, "implied_eff_WS_ROI_54_wall": 0.1 / 0.13}
# k_sci (the science-sample normalisation, insensitive to the veto efficiency) from science bins only:
R["veto"]["k_sci_ul90_from_science_bins_only"] = variants["science_only"]["ul90_LR"]
R["veto"]["k_sci_ul95_from_science_bins_only"] = variants["science_only"]["ul95_LR"]
# untagged fraction needed for the event if k=1 (prompt-normalised total wall MSSI 0.0948 in the 4.7 t ROI):
tot_wall_47 = TABLE[("WS ROI", "Wall", 4.7)][2] + WALL_SCI_47
R["veto"]["total_wall_MSSI_47_ROI"] = tot_wall_47
R["veto"]["untagged_fraction_needed_P10_nb"] = float(-np.log(0.9) / (tot_wall_47 * f_nb))

# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
ax[0].step(edges[:-1], np.where(tot > 0, tot, np.nan), where="post", color="tab:blue", label="total model (digitised)")
ax[0].step(edges[:-1], np.where(mssi > 0, mssi, np.nan), where="post", color="#025180", label="MSSI (digitised)")
ax[0].axvline(EVENT_SIGMA_NR, color="k", ls=":", label="event (-1.5σ)")
ax[0].axvspan(-2, 2, color="0.9", zorder=0)
ax[0].set_yscale("log"); ax[0].set_xlabel(r"$(\log_{10}S2c-\mu_{NR})/\sigma_{NR}$"); ax[0].set_ylabel("events / 0.5σ bin, S1c>500 phd")
ax[0].set_title(f"Fig.5 bottom panel: total {tot_int*norm:.4f} (caption 0.0106)\nMSSI in ±2σ: {mssi_pm2*norm:.1e} → f_nb={f_nb:.3f}", fontsize=9)
ax[0].legend(fontsize=8)
ax[1].plot(kgrid, nll_all, label="all 6 wall bins")
ax[1].plot(kgrid, nll_sci, "--", label="science bins only")
ax[1].plot(kgrid, nll_pr, ":", label="prompt bins only")
ax[1].plot(kg10, nll_s, "-.", label="+20% split nuisance")
for cl, thr in [(0.90, stats.chi2.ppf(0.8, 1) / 2), (0.95, stats.chi2.ppf(0.9, 1) / 2)]:
    ax[1].axhline(thr, color="0.5", lw=0.8); ax[1].text(11.5, thr + 0.05, f"{int(cl*100)}% (one-sided)", ha="right", fontsize=8)
ax[1].axvline(1, color="k", ls=":"); ax[1].set_xlim(0, 12); ax[1].set_ylim(0, 6)
ax[1].set_xlabel("wall-MSSI scale factor k"); ax[1].set_ylabel(r"$-\Delta\ln L$"); ax[1].legend(fontsize=8)
ax[1].set_title(f"Sidebands: k_ML={variants['all_wall']['k_ml']:.2f}, k<{variants['all_wall']['ul90_LR']:.2f} (90%), <{k95:.2f} (95%)", fontsize=9)
kk = np.logspace(-1, 4, 400)
ax[2].plot(kk, p_ge1(WALL_SCI_47 * kk), label="anywhere in WS ROI (0.0048 k)")
ax[2].fill_between(kk, p_ge1(WALL_SCI_47 * F_NB_RANGE[0] * kk), p_ge1(WALL_SCI_47 * F_NB_RANGE[1] * kk), alpha=0.3, color="tab:red")
ax[2].plot(kk, p_ge1(WALL_SCI_47 * f_nb * kk), color="tab:red", label=f"S1c>500 & |Δ|<2σ (f_nb={f_nb:.3f})")
ax[2].plot(kk, p_ge1(WALL_SCI_47 * f_nb * max(fpos.values()) * kk), color="tab:purple", ls="--", label="… and ≥25 cm from wall (f_pos max)")
ax[2].axvspan(1e-1, k95, color="green", alpha=0.15, label=f"allowed by sidebands (k<{k95:.1f}, 95%)")
ax[2].axvspan(1e-1, variants["pessimistic_obs_misclass"]["ul95_LR"], color="orange", alpha=0.12, label="pessimistic mis-classification extreme")
ax[2].axhline(0.1, color="0.5", lw=0.8); ax[2].axhline(0.5, color="0.5", lw=0.8)
ax[2].set_xscale("log"); ax[2].set_xlabel("k"); ax[2].set_ylabel("P(≥1 wall MSSI)"); ax[2].legend(fontsize=7, loc="upper left")
ax[2].set_title(f"k for 10% in neighbourhood: {R['k_required']['nb_10pct']:.0f}  (k95 sideband: {k95:.1f})", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "P004_k_likelihood_and_Pge1.png"), dpi=150)

# --------------------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------------------
with open(os.path.join(OUT, "P004_results.json"), "w") as f:
    json.dump(R, f, indent=1, default=float)
import csv
with open(os.path.join(OUT, "P004_k_limits.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["variant", "label", "k_ml", "ul68_LR", "ul90_LR", "ul95_LR", "ul68_Bayes", "ul90_Bayes", "ul95_Bayes"])
    for name, v in variants.items():
        w.writerow([name, v["label"], f"{v['k_ml']:.3f}"] + [f"{v[x]:.3f}" for x in ["ul68_LR", "ul90_LR", "ul95_LR", "ul68_Bayes", "ul90_Bayes", "ul95_Bayes"]])
with open(os.path.join(OUT, "P004_topology.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(topo[0].keys())); w.writeheader(); [w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in t.items()}) for t in topo]
with open(os.path.join(OUT, "P004_fig5_digitised.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["sigma_low", "sigma_high", "total", "mssi", "accidentals", "nrs"])
    for i in range(len(centres)):
        w.writerow([edges[i], edges[i + 1], f"{tot[i]:.3e}", f"{mssi[i]:.3e}", f"{np.nan_to_num(hist['Accidentals'])[i]:.3e}", f"{np.nan_to_num(hist['NRs'])[i]:.3e}"])

# console summary
print(json.dumps({k: R[k] for k in ["fig5_total_integral_digitised", "fig5_total_ratio", "fig5_norm_factor_used",
                                    "mssi_S1c_gt500_counts", "mssi_S1c_gt500_pm2sig_counts", "mssi_event_bin_counts",
                                    "total_model_pm2sig_counts", "f_S1c_gt500", "f_nb_pm2sigma", "f_nb_m3p1sigma", "f_nb_range",
                                    "k_paper_prior_95", "P_ge1_ROI_k1", "P_ge1_nb_k1", "k_required",
                                    "ratio_k_required_nb10_over_k95", "ratio_k_required_ROI10_over_k95", "P_ge1_nb_at_k95",
                                    "P_ge1_ROI_at_k95", "P_ge1_nb_at_k_pess", "bayes_factors", "mu_other_backgrounds_pm2sig",
                                    "mssi_share_of_pm2sig_model_k1", "mssi_share_of_pm2sig_model_k95", "paths_cm", "f_pos_range",
                                    "k_required_nb10_with_fpos_range", "veto"]}, indent=1, default=float))
for name, v in variants.items():
    print(f"{name:28s} k_ML={v['k_ml']:.3f}  LR UL 68/90/95 = {v['ul68_LR']:.2f}/{v['ul90_LR']:.2f}/{v['ul95_LR']:.2f}   Bayes 68/90/95 = {v['ul68_Bayes']:.2f}/{v['ul90_Bayes']:.2f}/{v['ul95_Bayes']:.2f}")
print("digitised bins (sigma_low, total, mssi):")
for i in range(len(centres)):
    print(f"  {edges[i]:5.1f}  {tot[i]:.2e}  {mssi[i]:.2e}")
for t in topo:
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in t.items()})
