"""
P006 -- Sixteenth of June: the event date under inelastic dark matter and under background.

Computes, for the single LZ event recorded 16 June 2023 21:22:39 UTC (day-of-year 167):
  1. Earth speed vs day-of-year (three models), modulation peak day, v_max(t) = v_E + v_esc.
  2. WimPyDD in-ROI rates through the year for O1 (SI-like) elastic and inelastic models,
     the normalised time PDF p(t|model) over the run, modulation fraction and phase.
  3. Likelihood ratio LR = p(16 June|model)/p(16 June|flat) under two livetime models, and the
     probability that one event falls within +-30 d of the modulation peak.
  4. Information content of the date: E[log LR], Var[log LR] under model and under flat,
     N_events needed for 3 sigma discrimination (analytic + toy MC).
  5. delta_max on 16 June vs the annual mean (kinematic bonus).
Run from the simulation root:  .venv/bin/python output/code/P006_event_date.py
Outputs: output/work/P006/*.csv, *.json, figures/*.png
"""
from __future__ import annotations
import sys, os, json, math, datetime as dt
import numpy as np
from scipy import integrate, optimize, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "output/code")
from common import lzcommon as lz  # noqa: E402

OUT = "output/work/P006"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(6)
T_YR = 365.25

# ----------------------------------------------------------------------------
# 0. Calendar of the run (LZ Data Analysis paragraph: 220 live days, 27 Mar 2023 -- 1 Apr 2024)
# ----------------------------------------------------------------------------
d_start = dt.date(2023, 3, 27)
d_end = dt.date(2024, 4, 1)
d_event = dt.date(2023, 6, 16)
RUN_DAYS = (d_end - d_start).days             # 371 calendar days
t_event = (d_event - d_start).days + (21 + 22 / 60 + 39 / 3600) / 24.0   # days since run start
doy_start = d_start.timetuple().tm_yday       # 86
def doy_of(t_run):
    """day-of-year (1-based, continuous) for time t_run days after run start (2023 basis)."""
    return (doy_start + t_run - 1) % T_YR + 1
doy_event = doy_of(t_event)                   # ~167.9 (16 June, 21:22 UTC)
LIVE_DAYS = lz.LZ["live_days"]

# ----------------------------------------------------------------------------
# 1. Earth speed vs day-of-year: three models
# ----------------------------------------------------------------------------
WD = lz.wd()
import wimprates as wr, numericalunits as nu

def vE_lz(doy):
    return lz.v_earth_kms(doy)

def vE_wd(doy):
    lam = (doy - 80) * 2 * np.pi / 365.0
    v = np.array([0.0, lz.V0_KMS, 0.0]) + lz.V_SUN_PEC + np.array(WD.v_earth_sun(lam))
    return float(np.linalg.norm(v))

def vE_wr(doy, year=2023):
    d = dt.date(year, 1, 1) + dt.timedelta(days=float(doy) - 1)
    t = wr.j2000_from_ymd(d.year, d.month, d.day) + (doy - math.floor(doy))
    return float(wr.v_earth(t) / (nu.km / nu.s))

doys = np.arange(1, 366.25, 0.25)
vE_tab = {"lz_cosine": np.array([vE_lz(d) for d in doys]),
          "wimpydd": np.array([vE_wd(d) for d in doys]),
          "wimprates": np.array([vE_wr(d) for d in doys])}
earth = {}
for k, v in vE_tab.items():
    ipk = int(np.argmax(v))
    earth[k] = dict(peak_doy=float(doys[ipk]), v_max_kms=float(v[ipk]), v_min_kms=float(v.min()),
                    mean_kms=float(v.mean()), amplitude_kms=float((v.max() - v.min()) / 2),
                    v_event_kms=float({"lz_cosine": vE_lz, "wimpydd": vE_wd, "wimprates": vE_wr}[k](doy_event)),
                    vmax_event_kms=float({"lz_cosine": vE_lz, "wimpydd": vE_wd, "wimprates": vE_wr}[k](doy_event) + lz.VESC_KMS))
    pk = dt.date(2023, 1, 1) + dt.timedelta(days=float(doys[ipk]) - 1)
    earth[k]["peak_date"] = pk.isoformat()
np.savetxt(os.path.join(OUT, "earth_speed_vs_doy.csv"),
           np.c_[doys, vE_tab["lz_cosine"], vE_tab["wimpydd"], vE_tab["wimprates"]], delimiter=",",
           header="doy,vE_lz_cosine_kms,vE_wimpydd_kms,vE_wimprates_kms", comments="")

# ----------------------------------------------------------------------------
# 2. Efficiency (Fig. S2 of the LZ supplement, read by eye) and models
# ----------------------------------------------------------------------------
def eff(E):
    """LZ NR signal efficiency: 50% at 5.4 keV (logistic, width 1 keV), plateau 0.96,
    high-energy roll-off 0.96*Phi((269.9-E)/15 keV): reproduces 0.92@250, 0.75@260, 0.50@269.9,
    0.25@280, 0.09@290 keV read from the Fig. S2 inset."""
    E = np.asarray(E, float)
    lo = 1.0 / (1.0 + np.exp(-(E - lz.LZ["E_50pct_low_keV"]) / 1.0))
    hi = stats.norm.cdf((lz.LZ["E_50pct_high_keV"] - E) / 15.0)
    return lz.LZ["eff_plateau"] * lo * hi

E_GRID = np.concatenate([np.linspace(1.0, 20.0, 20), np.linspace(21.0, 320.0, 100)])
ROI = (E_GRID >= 5.4) & (E_GRID <= 270.0)
HIGH = (E_GRID >= 200.0) & (E_GRID <= 270.0)

HAM_O1 = lz.wd_hamiltonian("O1s", {1: (1.0, 0.0)})
HAM_O11 = lz.wd_hamiltonian("O11s", {11: (1.0, 0.0)})   # q^2-suppressed proxy for dipole-like spectra
MODELS = [("O1 el m=100", HAM_O1, 100.0, 0.0), ("O1 el m=1000", HAM_O1, 1000.0, 0.0),
          ("O11 el m=1000", HAM_O11, 1000.0, 0.0)]
MODELS += [(f"O1 d={d:d} m=1000", HAM_O1, 1000.0, float(d)) for d in (100, 200, 250, 300, 350, 380)]

# day grid: every 10 days through the year plus the event day and the modulation peak
DAY_GRID = np.unique(np.round(np.concatenate([np.arange(1.0, 366.0, 10.0), [152.5, 153.0, 160.0, doy_event, 335.0]]), 3))
print(f"day grid: {len(DAY_GRID)} days; energy grid {len(E_GRID)} points")

halos = {d: lz.wd_halo(day_of_year=float(d)) for d in DAY_GRID}
halo_avg = lz.wd_halo()   # annual average (WimPyDD: v_earth_sun = 0)

def integ(r, mask):
    return float(np.trapezoid((r * eff(E_GRID))[mask], E_GRID[mask]))

rates = {}   # name -> dict(day-> (R_roi, R_high, dRdE at 248))
spectra_event_day = {}
for name, ham, m, delta in MODELS:
    rows = []
    for d in DAY_GRID:
        r = np.clip(lz.wd_rate(ham, m, E_GRID, halo=halos[d], delta_kev=delta), 0, None)
        r248 = float(np.interp(248.0, E_GRID, r))
        rows.append((d, integ(r, ROI), integ(r, HIGH), r248))
        if abs(d - doy_event) < 1e-2:
            spectra_event_day[name] = r
            R_ev = rows[-1][1]
    r_avg = np.clip(lz.wd_rate(ham, m, E_GRID, halo=halo_avg, delta_kev=delta), 0, None)
    rates[name] = dict(table=np.array(rows), R_roi_avg_halo=integ(r_avg, ROI), R_high_avg_halo=integ(r_avg, HIGH))
    print(f"  done {name:16s}  R_ROI(16 Jun)/R_ROI(avg halo) = {R_ev / max(rates[name]['R_roi_avg_halo'], 1e-300):.3f}")

# periodic interpolation of R(doy): cubic spline on a periodic extension, clipped at zero
from scipy.interpolate import CubicSpline
def periodic_interp(days, vals):
    dd = np.concatenate([days - T_YR, days, days + T_YR])
    vv = np.concatenate([vals, vals, vals])
    cs = CubicSpline(dd, vv)
    return lambda x: np.clip(cs(np.asarray(x) % T_YR + (T_YR if False else 0)), 0.0, None)

# ----------------------------------------------------------------------------
# 3. Livetime models and time PDFs
# ----------------------------------------------------------------------------
t_fine = np.linspace(0.0, RUN_DAYS, 371 * 8 + 1)     # days since run start (3 h steps)
def livetime_uniform(t):
    return np.full_like(np.asarray(t, float), LIVE_DAYS / RUN_DAYS)

# ASSUMPTION (not published): three ~1-week calibration gaps. The AmBe campaign is anchored on
# the paper's "AmBe deployment on 8 June 2023" (Discussion); the other two NR campaigns are placed
# at the start of the run (D-D, Apr 2023) and mid-run (Oct 2023) as a plausible guess.
GAPS = [(dt.date(2023, 4, 10), dt.date(2023, 4, 17)), (dt.date(2023, 6, 5), dt.date(2023, 6, 12)),
        (dt.date(2023, 10, 9), dt.date(2023, 10, 16))]
def livetime_gaps(t):
    t = np.asarray(t, float)
    w = np.ones_like(t)
    for a, b in GAPS:
        w[(t >= (a - d_start).days) & (t < (b - d_start).days)] = 0.0
    n_gap = sum((b - a).days for a, b in GAPS)
    return w * LIVE_DAYS / (RUN_DAYS - n_gap)

LIVETIMES = {"uniform": livetime_uniform, "gaps": livetime_gaps}

def kl_and_var(p, q, t):
    """E_p[ln(p/q)] and Var_p[ln(p/q)] on grid t (p, q normalised densities on the support of p)."""
    m = (p > 0)
    l = np.zeros_like(p); l[m] = np.log(p[m] / q[m])
    mean = np.trapezoid(p * l, t)
    var = np.trapezoid(p * (l - mean) ** 2, t)
    return float(mean), float(var)

results = []
pdf_store = {}
for name, ham, m, delta in MODELS:
    tab = rates[name]["table"]
    R_of_doy = periodic_interp(tab[:, 0], tab[:, 1])
    Rh_of_doy = periodic_interp(tab[:, 0], tab[:, 2])
    R_t = R_of_doy(doy_of(t_fine))
    Rh_t = Rh_of_doy(doy_of(t_fine))
    # modulation: on the year
    Ryr = R_of_doy(doys)
    Rmax, Rmin = Ryr.max(), Ryr.min()
    modfrac = (Rmax - Rmin) / (Rmax + Rmin) if Rmax > 0 else float("nan")
    peak_doy = float(doys[np.argmax(Ryr)])
    # cosine-fit phase (first harmonic)
    if Rmax > 0:
        A0 = Ryr.mean()
        c1 = 2 * np.mean(Ryr * np.cos(2 * np.pi * (doys - 1) / T_YR)); s1 = 2 * np.mean(Ryr * np.sin(2 * np.pi * (doys - 1) / T_YR))
        phase_doy = float((np.arctan2(s1, c1) * T_YR / (2 * np.pi)) % T_YR + 1)
        A1_over_A0 = float(math.hypot(c1, s1) / A0)
    else:
        phase_doy = A1_over_A0 = float("nan")
    Rh_yr = Rh_of_doy(doys)
    modfrac_high = (Rh_yr.max() - Rh_yr.min()) / (Rh_yr.max() + Rh_yr.min()) if Rh_yr.max() > 0 else float("nan")
    frac_days_allowed = float(np.mean(Ryr > 1e-6 * Rmax)) if Rmax > 0 else 0.0
    row = dict(model=name, m_gev=m, delta_kev=delta, mod_fraction_roi=float(modfrac), mod_fraction_200_270=float(modfrac_high),
               peak_doy=peak_doy, cosfit_phase_doy=phase_doy, A1_over_A0=A1_over_A0, frac_year_allowed=frac_days_allowed,
               R_roi_event_over_avg_halo=float(R_of_doy(doy_event) / rates[name]["R_roi_avg_halo"]) if rates[name]["R_roi_avg_halo"] > 0 else float("inf"),
               R_roi_event_over_dec=float(R_of_doy(doy_event) / R_of_doy(336.0)) if R_of_doy(336.0) > 0 else float("inf"))
    for lname, L in LIVETIMES.items():
        Lt = L(t_fine)
        norm_flat = np.trapezoid(Lt, t_fine)
        p_flat = Lt / norm_flat
        for tag, RR in (("roi", R_t), ("200_270", Rh_t)):
            w = RR * Lt
            Z = np.trapezoid(w, t_fine)
            p_mod = w / Z
            p_ev_mod = float(np.interp(t_event, t_fine, p_mod))
            p_ev_flat = float(np.interp(t_event, t_fine, p_flat))
            LR = p_ev_mod / p_ev_flat
            # probability of +-30 d around the peak (peak within the run: first occurrence)
            t_peak = ((peak_doy - doy_start) % T_YR)
            win = (t_fine >= t_peak - 30) & (t_fine <= t_peak + 30)
            P30_mod = float(np.trapezoid(p_mod[win], t_fine[win]))
            P30_flat = float(np.trapezoid(p_flat[win], t_fine[win]))
            KL_m, V_m = kl_and_var(p_mod, p_flat, t_fine)              # under model: E ln(p_m/p_f)
            KL_f, V_f = kl_and_var(p_flat, np.where(p_mod > 0, p_mod, np.nan), t_fine) if np.all(p_mod[p_flat > 0] > 0) else (float("inf"), float("inf"))
            # analytic N for 3 sigma: separation of sum(lnLR) means in units of the flat-hypothesis sd
            sep = KL_m + KL_f   # E_m - E_f  (E_f = -KL_f)
            N3 = 9.0 * V_f / sep ** 2 if np.isfinite(V_f) else float("nan")
            # alternative: sqrt(2 N KL_m) = 3  (Asimov-like)
            N3_asimov = 4.5 / KL_m if KL_m > 0 else float("inf")
            row.update({f"LR_{tag}_{lname}": LR, f"pev_mod_{tag}_{lname}": p_ev_mod, f"pev_flat_{lname}": p_ev_flat,
                        f"P30_mod_{tag}_{lname}": P30_mod, f"P30_flat_{lname}": P30_flat,
                        f"ElnLR_model_{tag}_{lname}": KL_m, f"VarlnLR_model_{tag}_{lname}": V_m,
                        f"ElnLR_flat_{tag}_{lname}": -KL_f, f"VarlnLR_flat_{tag}_{lname}": V_f,
                        f"N3sigma_{tag}_{lname}": N3, f"N3sigma_asimov_{tag}_{lname}": N3_asimov})
            if lname == "uniform" and tag == "roi":
                pdf_store[name] = (p_mod, p_flat)
    results.append(row)

# toy-MC check of N_3sigma for the ROI/uniform case: draw N events from p_mod and from p_flat
def toy_N3(p_mod, p_flat, t, N, ntoy=4000):
    cdf_m = np.cumsum(p_mod) * (t[1] - t[0]); cdf_m /= cdf_m[-1]
    cdf_f = np.cumsum(p_flat) * (t[1] - t[0]); cdf_f /= cdf_f[-1]
    lr = np.zeros_like(p_mod); m = p_mod > 0; lr[m] = np.log(p_mod[m] / p_flat[m]); lr[~m] = -50.0
    def draw(cdf):
        u = rng.random((ntoy, N)); return np.interp(u, cdf, t)
    S_m = np.sum(np.interp(draw(cdf_m), t, lr), axis=1)
    S_f = np.sum(np.interp(draw(cdf_f), t, lr), axis=1)
    # p-value of median model statistic under the flat distribution
    med = np.median(S_m); p = np.mean(S_f >= med)
    return float(stats.norm.isf(max(p, 0.5 / ntoy)))

toy = {}
for row in results:
    name = row["model"]
    N3 = row["N3sigma_roi_uniform"]
    if np.isfinite(N3) and N3 < 5000:
        N = max(1, int(round(N3)))
        z = toy_N3(*pdf_store[name], t_fine, N)
        toy[name] = dict(N=N, z_at_N=z)
        # also z for N = 1
        toy[name]["z_at_N1"] = toy_N3(*pdf_store[name], t_fine, 1)

# ----------------------------------------------------------------------------
# 4. delta_max kinematics: 16 June vs annual mean vs December
# ----------------------------------------------------------------------------
kin = {}
for k, f in (("lz_cosine", vE_lz), ("wimprates", vE_wr)):
    vJ, vA, vD = f(doy_event), (vE_tab[k].mean()), f(336.0)
    kin[k] = {}
    for E in (202.0, 248.0, 294.0):
        kin[k][f"E{int(E)}"] = {"delta_max_16June_keV": {m: lz.delta_max_kev(E, m, v_kms=vJ + lz.VESC_KMS) for m in (400, 1000, 4000)},
                                "delta_max_annual_mean_keV": {m: lz.delta_max_kev(E, m, v_kms=vA + lz.VESC_KMS) for m in (400, 1000, 4000)},
                                "delta_max_December_keV": {m: lz.delta_max_kev(E, m, v_kms=vD + lz.VESC_KMS) for m in (400, 1000, 4000)}}
    kin[k]["vE_16June"], kin[k]["vE_mean"], kin[k]["vE_Dec"] = vJ, vA, vD
# days of the year on which a 248 keV recoil is kinematically allowed for given delta (m=1000)
allowed = {}
for delta in (350, 370, 380, 385, 390):
    vmin = lz.vmin_kms(248.0, 1000.0, delta_kev=delta)
    ok = vE_tab["wimprates"] + lz.VESC_KMS > vmin
    allowed[delta] = dict(vmin_kms=float(vmin), frac_year=float(ok.mean()), n_days=float(ok.mean() * T_YR),
                          allowed_on_16June=bool(vE_wr(doy_event) + lz.VESC_KMS > vmin))

# ----------------------------------------------------------------------------
# 5. Save tables
# ----------------------------------------------------------------------------
import pandas as pd
df = pd.DataFrame(results)
df.to_csv(os.path.join(OUT, "P006_results.csv"), index=False)
for name in rates:
    tab = rates[name]["table"]
    np.savetxt(os.path.join(OUT, f"rate_vs_doy_{name.replace(' ', '_').replace('=', '')}.csv"), tab, delimiter=",",
               header="doy,R_roi_arb,R_200_270_arb,dRdE_248_arb", comments="")
summary = dict(run=dict(start=d_start.isoformat(), end=d_end.isoformat(), calendar_days=RUN_DAYS, live_days=LIVE_DAYS,
                        duty_cycle=LIVE_DAYS / RUN_DAYS, t_event_days_after_start=t_event, doy_event=doy_event,
                        gaps_assumed=[(a.isoformat(), b.isoformat()) for a, b in GAPS]),
               earth_velocity=earth, kinematics=kin, allowed_days_248keV_m1000=allowed,
               efficiency_model="0.96*logistic((E-5.4)/1 keV)*Phi((269.9-E)/15 keV)",
               toy_check=toy, results=results)
json.dump(summary, open(os.path.join(OUT, "P006_summary.json"), "w"), indent=1, default=float)

# ----------------------------------------------------------------------------
# 6. Figures
# ----------------------------------------------------------------------------
# Fig 1: rate vs day-of-year for several models, normalised to annual mean; event date marked
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for name in ("O1 el m=1000", "O1 d=200 m=1000", "O1 d=300 m=1000", "O1 d=350 m=1000", "O1 d=380 m=1000"):
    tab = rates[name]["table"]
    f = periodic_interp(tab[:, 0], tab[:, 1])
    y = f(doys); y = y / y.mean() if y.mean() > 0 else y
    ax.plot(doys, y, label=name.replace("O1 ", "").replace("d=", "δ=").replace(" m=1000", " keV") if "d=" in name else "elastic SI, 1000 GeV")
ax.axvline(doy_event, color="k", ls="--", lw=1); ax.text(doy_event + 3, ax.get_ylim()[1] * 0.92, "16 June", fontsize=9)
ax.axvline(earth["wimprates"]["peak_doy"], color="grey", ls=":", lw=1)
ax.set_xlabel("day of year"); ax.set_ylabel("in-ROI rate / annual mean"); ax.set_yscale("log"); ax.set_ylim(0.05, 20)
ax.legend(fontsize=8, ncol=2); ax.set_title("WimPyDD O1 rate in 5.4–270 keV (LZ efficiency), m = 1000 GeV")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "rate_vs_doy.png"), dpi=150); plt.close(fig)

# Fig 2: LR vs delta (both livetimes, ROI and 200-270)
sub = df[df.model.str.contains("d=") | (df.model == "O1 el m=1000")].copy()
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(sub.delta_kev, sub.LR_roi_uniform, "o-", label="ROI 5.4–270 keV, uniform livetime")
ax.plot(sub.delta_kev, sub.LR_roi_gaps, "s--", label="ROI, with calibration gaps")
ax.plot(sub.delta_kev, sub["LR_200_270_uniform"], "^-", label="200–270 keV, uniform livetime")
ax.axhline(1, color="k", lw=0.8); ax.set_xlabel("δ [keV]  (m = 1000 GeV, O1)"); ax.set_ylabel("LR = p(16 June | model) / p(16 June | flat)")
ax.set_yscale("log"); ax.legend(fontsize=8); ax.set_title("Date likelihood ratio versus mass splitting")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "LR_vs_delta.png"), dpi=150); plt.close(fig)

# Fig 3: Earth speed models
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for k, v in vE_tab.items():
    ax.plot(doys, v, label=k)
ax.axvline(doy_event, color="k", ls="--", lw=1); ax.set_xlabel("day of year"); ax.set_ylabel("v_E [km/s]"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "earth_speed.png"), dpi=150); plt.close(fig)

# Fig 4: information content: N_3sigma vs delta
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(sub.delta_kev, sub.N3sigma_roi_uniform, "o-", label="ROI, analytic (z-score)")
ax.plot(sub.delta_kev, sub.N3sigma_asimov_roi_uniform, "s--", label="ROI, sqrt(2 N KL) = 3")
ax.plot(sub.delta_kev, sub["N3sigma_200_270_uniform"], "^-", label="200–270 keV, analytic")
ax.set_yscale("log"); ax.set_xlabel("δ [keV]"); ax.set_ylabel("events needed for 3σ from dates alone"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "N3sigma_vs_delta.png"), dpi=150); plt.close(fig)

# ----------------------------------------------------------------------------
# 7. Print summary
# ----------------------------------------------------------------------------
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
cols = ["model", "mod_fraction_roi", "mod_fraction_200_270", "peak_doy", "cosfit_phase_doy", "frac_year_allowed",
        "R_roi_event_over_avg_halo", "R_roi_event_over_dec", "LR_roi_uniform", "LR_roi_gaps", "LR_200_270_uniform",
        "P30_mod_roi_uniform", "P30_flat_uniform", "ElnLR_model_roi_uniform", "VarlnLR_model_roi_uniform",
        "ElnLR_flat_roi_uniform", "VarlnLR_flat_roi_uniform", "N3sigma_roi_uniform", "N3sigma_asimov_roi_uniform", "N3sigma_200_270_uniform"]
print(df[cols].to_string(float_format=lambda x: f"{x:.4g}"))
print("\nEarth velocity models:"); print(json.dumps(earth, indent=1))
print("\nEvent: t =", round(t_event, 3), "d after start; doy =", round(doy_event, 3), "; duty cycle", round(LIVE_DAYS / RUN_DAYS, 4))
print("\ndelta_max (keV), wimprates Earth speed:"); print(json.dumps(kin["wimprates"], indent=1, default=float))
print("\nallowed days for 248 keV at m=1000:"); print(json.dumps(allowed, indent=1))
print("\ntoy check:"); print(json.dumps(toy, indent=1))
