"""
P006 post-processing (fast; reads output/work/P006/rate_vs_doy_*.csv written by P006_event_date.py).
For each model: the frequentist p-value of the date under the flat (uniform-livetime) hypothesis,
p_flat = P_flat[ ln LR(t) >= ln LR(16 June) ] = livetime fraction of the run whose predicted rate is at
least the 16 June rate; the z-score of the observed ln LR under the model and under flat; and the
combination with P001's Bayes factors.  Run from the root: .venv/bin/python output/code/P006_postprocess.py
"""
import sys, os, glob, json, math, datetime as dt
import numpy as np
from scipy import stats
from scipy.interpolate import CubicSpline
sys.path.insert(0, "output/code")

OUT = "output/work/P006"
T_YR = 365.25
d_start, d_end, d_event = dt.date(2023, 3, 27), dt.date(2024, 4, 1), dt.date(2023, 6, 16)
RUN = (d_end - d_start).days
t_ev = (d_event - d_start).days + (21 + 22 / 60 + 39 / 3600) / 24
doy_start = d_start.timetuple().tm_yday
doy_of = lambda t: (doy_start + t - 1) % T_YR + 1
t = np.linspace(0, RUN, RUN * 24 + 1)          # hourly grid
res = json.load(open(os.path.join(OUT, "P006_summary.json")))
rows = {r["model"]: r for r in res["results"]}

def periodic(days, vals):
    cs = CubicSpline(np.concatenate([days - T_YR, days, days + T_YR]), np.concatenate([vals] * 3))
    return lambda x: np.clip(cs(np.asarray(x)), 0, None)

out = {}
for f in sorted(glob.glob(os.path.join(OUT, "rate_vs_doy_*.csv"))):
    name = os.path.basename(f)[len("rate_vs_doy_"):-4]
    tab = np.loadtxt(f, delimiter=",", skiprows=1)
    R = periodic(tab[:, 0], tab[:, 1])(doy_of(t))
    Rbar = R.mean()                              # uniform livetime -> run mean
    lnLR = np.where(R > 0, np.log(np.where(R > 0, R, 1) / Rbar), -np.inf)
    lnLR_ev = float(np.interp(t_ev, t, lnLR))
    p_flat = float(np.mean(lnLR >= lnLR_ev))     # fraction of (uniform) livetime at least as "peaked"
    # under the model, the date distribution is p_m = R/sum(R): P_m[lnLR >= obs]
    p_model_ge = float(np.sum(R[lnLR >= lnLR_ev]) / R.sum())
    # sd-scaled position of the observation
    Em, Vm = rows[[k for k in rows if k.replace(' ', '_').replace('=', '') == name][0]]["ElnLR_model_roi_uniform"], rows[[k for k in rows if k.replace(' ', '_').replace('=', '') == name][0]]["VarlnLR_model_roi_uniform"]
    Ef, Vf = rows[[k for k in rows if k.replace(' ', '_').replace('=', '') == name][0]]["ElnLR_flat_roi_uniform"], rows[[k for k in rows if k.replace(' ', '_').replace('=', '') == name][0]]["VarlnLR_flat_roi_uniform"]
    # days in the run with rate >= the event-day rate (contiguous window around the 2023 peak)
    n_days_ge = float(np.mean(lnLR >= lnLR_ev) * RUN)
    out[name] = dict(lnLR_event=lnLR_ev, LR_event=math.exp(lnLR_ev), p_flat_ge=p_flat, z_one_sided_flat=float(stats.norm.isf(p_flat)),
                     p_model_ge=p_model_ge, days_in_run_with_rate_ge_event=n_days_ge,
                     z_of_obs_under_model=(lnLR_ev - Em) / math.sqrt(Vm) if Vm > 0 else float("nan"),
                     z_of_obs_under_flat=(lnLR_ev - Ef) / math.sqrt(Vf) if np.isfinite(Vf) and Vf > 0 else float("nan"),
                     R_event_over_run_max=float(np.interp(t_ev, t, R) / R.max()))
# combination with P001 (Bayes factor after Occam penalty 7-37; before penalty 100-500)
for name, d in out.items():
    d["B10_P001_occam_7_37_times_LR"] = [7 * d["LR_event"], 37 * d["LR_event"]]
json.dump(out, open(os.path.join(OUT, "P006_postprocess.json"), "w"), indent=1)
for name, d in out.items():
    print(f"{name:18s} LR={d['LR_event']:.3f}  p_flat(>=)={d['p_flat_ge']:.4f} ({d['z_one_sided_flat']:.2f} sigma one-sided)  "
          f"days>=event: {d['days_in_run_with_rate_ge_event']:.1f}  P_model(>=)={d['p_model_ge']:.3f}  z_obs|model={d['z_of_obs_under_model']:.2f}  z_obs|flat={d['z_of_obs_under_flat']:.2f}  R_ev/R_max={d['R_event_over_run_max']:.3f}")
