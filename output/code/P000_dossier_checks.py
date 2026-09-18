"""
P000_dossier_checks.py -- quantitative checks supporting the Phase-1 evidence dossier.

Run from the simulation root:  .venv/bin/python output/code/P000_dossier_checks.py
Outputs: output/work/dossier/dossier_numbers.json and printed summary.
Tools: python 3.12.13, numpy 2.5.3, scipy 1.18.1 (special.erf, stats.poisson/norm, integrate.quad),
       wimprates 0.5.0 (rate_wimp_std, cross-check), nestpy 2.1.1 (NESTcalc.GetYields with LZ-tuned
       parameters from Tables S3-S5, LZ_WS2024 detector), common/lzcommon.py.
"""
import sys, json, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "output/code")
import numpy as np
from scipy import integrate
from common import lzcommon as lz
import wimprates as wr

out = {}
os.makedirs("output/work/dossier", exist_ok=True)

# 1. Halo kinematics on the event date (16 June 2023 = day-of-year 167)
vE_avg, vE_jun, vE_dec = lz.v_earth_kms(), lz.v_earth_kms(167), lz.v_earth_kms(153 + 182.6)
out["v_earth_avg_kms"], out["v_earth_16June_kms"], out["v_earth_Dec_kms"] = vE_avg, vE_jun, vE_dec
out["vmax_16June_kms"], out["vmax_Dec_kms"] = lz.vmax_kms(vE_jun), lz.vmax_kms(vE_dec)
# wimprates cross-check of Earth speed on the event date
t_ev = wr.j2000_from_ymd(2023, 6, 16.89)
import numericalunits as nu
out["wimprates_v_earth_16June_kms"] = float(wr.v_earth(t_ev) / (nu.km / nu.s))

# 2. Elastic kinematics: minimum WIMP mass to give 248 keV (and 248-46 keV low edge)
out["m_chi_min_elastic_248keV_GeV"] = lz.m_chi_min_gev(248, v_kms=lz.vmax_kms(vE_jun))
out["m_chi_min_elastic_202keV_GeV"] = lz.m_chi_min_gev(202, v_kms=lz.vmax_kms(vE_jun))
out["vmin_248keV_kms"] = {m: lz.vmin_kms(248, m) for m in [50, 100, 200, 400, 1000, 4000]}

# 3. Inelastic kinematics: delta_max for 248 keV
out["delta_max_248keV_keV"] = {m: {"June": lz.delta_max_kev(248, m, v_kms=lz.vmax_kms(vE_jun)),
                                   "Dec": lz.delta_max_kev(248, m, v_kms=lz.vmax_kms(vE_dec)),
                                   "avg": lz.delta_max_kev(248, m)} for m in [400, 1000, 4000, 1e6]}
# for the +-1 sigma energy window
out["delta_max_202keV_1000GeV"] = lz.delta_max_kev(202, 1000, v_kms=lz.vmax_kms(vE_jun))
out["delta_max_294keV_1000GeV"] = lz.delta_max_kev(294, 1000, v_kms=lz.vmax_kms(vE_jun))

# 4. Event signal arithmetic
Nph, Ne = 540.1 / lz.LZ["g1"], 9268.0 / lz.LZ["g2"]
out["event_Nph"], out["event_Ne"], out["event_Nq"] = Nph, Ne, Nph + Ne
out["event_Eee_W13.7_keV"] = lz.combined_energy_keV(540.1, 9268)
out["event_Eee_W13.5_keV"] = lz.combined_energy_keV(540.1, 9268, W_eV=13.5)
out["event_Enr_from_alpha_beta_keV"] = ((Nph + Ne) / lz.NEST_NR_LZ["alpha"]) ** (1 / lz.NEST_NR_LZ["beta"])
# NEST-tuned NR band centre near the event
nest = {}
for E in [150, 200, 225, 248, 270, 300]:
    nph, ne = lz.nest_nr_yields(E)
    nph0, ne0 = lz.nest_nr_yields(E, params=lz.NEST_NR_DEFAULT)
    nest[E] = dict(S1c_LZ=nph * lz.LZ["g1"], log10S2c_LZ=float(np.log10(ne * lz.LZ["g2"])),
                   S1c_default=nph0 * lz.LZ["g1"], log10S2c_default=float(np.log10(ne0 * lz.LZ["g2"])))
out["nest_nr_band_centre"] = nest
# ER (beta) yields at the double-vacancy line energies with LZ ER parameters
er = {}
for E in [64.3, 67.3, 70.9]:
    nph, ne = lz.nest_er_yields(E, params=lz.NEST_ER_LZ)
    er[E] = dict(S1c=nph * lz.LZ["g1"], log10S2c=float(np.log10(ne * lz.LZ["g2"])))
out["nest_er_beta_at_DEC_energies"] = er

# 5. Statistics
p_panel = lz.poisson_p_at_least(1, 0.0106)
out["P_ge1_given_0.0106"] = p_panel
out["sigma_one_sided_panel"] = lz.p_to_sigma(p_panel)
out["p_local_3.4sigma"] = lz.sigma_to_p(3.4)
out["p_global_2.6sigma"] = lz.sigma_to_p(2.6)
out["effective_independent_trials"] = float(np.log(1 - lz.sigma_to_p(2.6)) / np.log(1 - lz.sigma_to_p(3.4)))
# Poisson: probability that a background of mu gives >=1 event in the region where the event sits
for mu in [0.0106, 0.02, 0.05, 0.1]:
    out[f"P_ge1_mu={mu}"] = lz.poisson_p_at_least(1, mu)
# Bayes factor for a single event: likelihood ratio (s+b)/b for a best-fit s=1 signal with b=0.0106 in the box
out["LR_s1_vs_b_in_box"] = (1.0 + 0.0106) / 0.0106

# 6. SI rate sanity: our library vs wimprates, and high-energy fraction
cmp = {}
for E in [10, 50, 100, 200, 250]:
    cmp[E] = dict(lzcommon=lz.dRdE_SI(E, 1000, 1e-45), wimprates=float(wr.rate_wimp_std(E, mw=1000, sigma_nucleon=1e-45)))
out["SI_rate_1000GeV_1e-45_per_t_yr_keV"] = cmp
R_all = integrate.quad(lambda E: lz.dRdE_SI(E, 1000, 1e-45), 5.4, 270, limit=200)[0]
R_hi = integrate.quad(lambda E: lz.dRdE_SI(E, 1000, 1e-45), 200, 270, limit=200)[0]
out["SI_1000GeV_rate_5.4-270_per_t_yr"] = R_all
out["SI_1000GeV_fraction_above_200keV"] = R_hi / R_all
out["SI_1000GeV_events_below_200_per_event_above_200"] = (R_all - R_hi) / R_hi
# same for inelastic delta=300 keV: fraction above 200 keV
R_all_i = integrate.quad(lambda E: lz.dRdE_SI(E, 1000, 1e-40, delta_kev=300, v_e=vE_jun), 5.4, 270, limit=200)[0]
R_hi_i = integrate.quad(lambda E: lz.dRdE_SI(E, 1000, 1e-40, delta_kev=300, v_e=vE_jun), 200, 270, limit=200)[0]
out["inelastic_d300_1000GeV_fraction_above_200keV"] = R_hi_i / R_all_i if R_all_i > 0 else None
Emin_i, Emax_i = lz.E_R_range_keV(1000, lz.vmax_kms(vE_jun), delta_kev=300)
out["inelastic_d300_1000GeV_Erange_keV_June"] = [Emin_i, Emax_i]
# modulation of inelastic delta=300 rate: June vs December total rate in ROI
R_dec_i = integrate.quad(lambda E: lz.dRdE_SI(E, 1000, 1e-40, delta_kev=300, v_e=vE_dec), 5.4, 270, limit=200)[0]
out["inelastic_d300_1000GeV_June_over_Dec_rate"] = R_all_i / R_dec_i if R_dec_i > 0 else float("inf")

# 7. Neutron kinematics: minimum neutron energy for 248 keV Xe recoil (elastic, head-on)
A = lz.A_XE_MEAN
out["E_n_min_for_248keV_MeV"] = 248e-3 * (1 + A) ** 2 / (4 * A)

# 8. Exposure check
out["exposure_check_tyr"] = lz.LZ["exposure_check_tyr"]

json.dump(out, open("output/work/dossier/dossier_numbers.json", "w"), indent=1, default=float)
for k, v in out.items():
    print(k, "=", v)
