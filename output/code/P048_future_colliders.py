"""
P048 - Chargino lifetimes, disappearing tracks and the road to a thermal Higgsino:
HL-LHC, FCC-hh and a muon collider after LZ.

Run from the simulation root:  .venv/bin/python output/code/P048_future_colliders.py

Inputs from the corpus: P014 chargino lifetimes (c tau = 0.71 cm at 1 TeV), P014 recalled 13 TeV
Drell-Yan anchors, P014 tree-level Delta m(+-)/delta coefficients, P007 fit points, P037 Y=1 triplet.
Every recalled number is flagged in RECALLED below and written to recalled_knowledge.json.

Parts
  A  cross-sections: 13 -> 14 TeV and 13 -> 100 TeV by tau-scaling of the P014 anchor fit;
     tree-level mu+mu- -> chi+chi-, chi1chi2 (doublet) and the Y=1 triplet at sqrt(s)=10 TeV
  B  beta*gamma_T distributions (toy Drell-Yan) and tracklet survival; wino calibration of eps_kin;
     expected tracklets vs mass and the mass reach for HL-LHC, FCC-hh, MuC
  C  mono-photon at the muon collider (collinear ISR estimate) and the nu nu gamma background
  D  delta-blindness: chi2 -> chi1 gamma lab-frame spectrum, decay lengths, Delta m(+-) -> delta bound,
     W-parameter reach of FCC-ee/CEPC/ILC
  E  identification logic, timeline table, figures
"""
import sys, os, json
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import lzcommon as lz

OUT = 'output/work/P048'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(48)
LOG = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s); LOG.append(s)

# ----------------------------------------------------------------------------------------------
# Recalled inputs (flagged)
# ----------------------------------------------------------------------------------------------
RECALLED = []
def rec(item, src, rel):
    RECALLED.append(dict(item=item, presumed_source=src, reliability=rel))

ALPHA_MZ = 1/127.95;  S2W = 0.2312;  C2W = 1 - S2W;  MZ = 91.1876; GZ = 2.4952; MW = 80.377
GF = 1.1664e-5; G2 = np.sqrt(4*np.pi*ALPHA_MZ/S2W)   # SU(2) coupling from alpha(mZ), s_W^2
rec("alpha(mZ)=1/127.95, sin^2 theta_W=0.2312, mZ=91.1876, Gamma_Z=2.4952, mW=80.377 GeV, G_F", "PDG", "certain")
GEV2_TO_FB = lz.GEV_TO_CM2 * 1e39   # 1 GeV^-2 = 3.894e11 fb
HBAR_C_M = 1.973e-16 * 1.0          # m GeV (hbar c = 1.973e-16 m GeV)
M_E_KEV = 510.999

# P014 13 TeV NLO+NLL pure-wino C1N2 anchors (recalled there; factor ~1.5)
WINO_ANCH_M = np.array([100, 200, 300, 400, 500, 600, 800, 1000.])
WINO_ANCH_FB = np.array([12000, 1800, 390, 120, 46, 20, 4.6, 1.3])
rec("13 TeV NLO+NLL pure-wino chi1+-chi2^0 cross-sections 12 pb ... 1.3 fb for 100-1000 GeV (P014 table)",
    "LHC SUSY xsec WG via corpus P014", "likely (factor ~1.5)")
rec("Higgsino chi+-chi0 channels = 1/2 of wino C1N2 at equal mass; chi+chi- + chi1chi2 add ~50%", "P014 (SU(2) Wigner-Eckart)", "likely")
rec("Parton-luminosity tau-scaling: sigma(m, s2) = sigma(m sqrt(s1/s2), s1) x s1/s2 (neglects PDF Q^2 evolution)",
    "collinear factorisation; own derivation", "likely (x/÷1.5 at fixed tau)")
rec("Assignment/literature 100 TeV Higgsino sigma(chi+chi- + chi+-chi0) ~ 30 fb at 1 TeV; Low-Wang 2014 Higgsino total ~60 fb",
    "Low & Wang JHEP 08 (2014) 161; assignment", "uncertain (factor 3)")

coef = np.polyfit(np.log(WINO_ANCH_M), np.log(WINO_ANCH_FB), 2)
def sig_wino_13(m):          # fb, C1N2 pure wino at 13 TeV (quadratic in ln m; extrapolated beyond 1 TeV)
    return np.exp(np.polyval(coef, np.log(m)))
def slope_13(m):
    return 2*coef[0]*np.log(m) + coef[1]
def sig_H_cn(m, sqrt_s_TeV):
    """Higgsino chi+-chi0 cross-section [fb] at collider energy sqrt(s) by tau-scaling of the 13 TeV fit."""
    r = 13.0/sqrt_s_TeV
    m_eq = m*r
    m_eq = np.clip(m_eq, 60, None)   # do not extrapolate below 60 GeV
    return 0.5*sig_wino_13(m_eq)*r**2
def sig_H_tot(m, sqrt_s):   # all four channels
    return 1.5*sig_H_cn(m, sqrt_s)
def sig_wino_tot(m, sqrt_s):  # C1N2 + C1C1 ~ 1.5 x C1N2 (P014 convention)
    r = 13.0/sqrt_s; return 1.5*sig_wino_13(np.clip(m*r, 60, None))*r**2

# ----------------------------------------------------------------------------------------------
# Part A1: hadron-collider cross-sections and pair counts
# ----------------------------------------------------------------------------------------------
log("="*100); log("PART A1: Drell-Yan cross-sections by tau-scaling of the P014 13 TeV anchors")
masses = np.array([300, 500, 800, 1000, 1100, 1500, 2000, 3000.])
rows = []
for m in masses:
    s13 = sig_H_cn(m, 13); s14 = sig_H_cn(m, 14); s100 = sig_H_cn(m, 100)
    rows.append(dict(m_GeV=m, sigma_H_cn_13TeV_fb=s13, sigma_H_cn_14TeV_fb=s14, ratio_14_13=s14/s13,
                     sigma_H_cn_100TeV_fb=s100, ratio_100_13=s100/s13,
                     sigma_H_tot_14TeV_fb=1.5*s14, sigma_H_tot_100TeV_fb=1.5*s100,
                     sigma_H_100TeV_low_fb=1.5*s100/3, sigma_H_100TeV_high_fb=1.5*s100*3,
                     pairs_HLLHC_3ab=1.5*s14*3000, pairs_FCChh_30ab=1.5*s100*30000,
                     slope_13=slope_13(m), slope_100=slope_13(max(m*0.13, 60)),
                     clipped_100TeV=(m*0.13 < 60)))
dfA = pd.DataFrame(rows); dfA.to_csv(f'{OUT}/hadron_xsec_pairs.csv', index=False)
log("NOTE: 100 TeV values with clipped_100TeV=True (m < 460 GeV) use the 13 TeV fit frozen at 60 GeV and are lower bounds; not used for any reach statement.")
log(dfA[['m_GeV','sigma_H_cn_13TeV_fb','ratio_14_13','sigma_H_cn_100TeV_fb','ratio_100_13','pairs_HLLHC_3ab','pairs_FCChh_30ab','slope_100']].to_string(float_format=lambda x: f'{x:.4g}'))
log(f"check vs P014 13 TeV anchors: sig_H_cn(300,500,1000) = {sig_H_cn(300,13):.1f}, {sig_H_cn(500,13):.1f}, {sig_H_cn(1000,13):.3f} fb (P014: 201, 23, 0.67)")
log(f"100 TeV, 1 TeV: chi+-chi0 {sig_H_cn(1000,100):.1f} fb; chi+chi- + chi+-chi0 ~ {sig_H_cn(1000,100)*(1+1/3):.1f} fb (assignment recalled 30 fb, band /3..x3)")

# ----------------------------------------------------------------------------------------------
# Part A2: muon collider tree-level cross-sections
# ----------------------------------------------------------------------------------------------
log("="*100); log("PART A2: mu+mu- -> chi chi at tree level (s-channel gamma + Z)")
rec("Tree-level e+e-/mu+mu- -> f fbar with s-channel gamma+Z: sigma = (4 pi alpha^2/3s) beta {(3-beta^2)/2 [Q^2 - 2 Q v_l v_f Re chi + (v_l^2+a_l^2) v_f^2 |chi|^2] + beta^2 (v_l^2+a_l^2) a_f^2 |chi|^2}, chi = s/(4 s_W^2 c_W^2 (s - mZ^2 + i mZ Gamma_Z)), v = g_L+g_R, a = g_L-g_R with g_L = T3 - Q s_W^2",
    "standard textbook (Peskin-Schroeder ch. 21; LEP2 literature)", "certain (massless limit reproduces R-ratio formula)")
rec("Pure-Higgsino chargino is vector-like under the Z: g_L = g_R = 1/2 - s_W^2; neutral Dirac Higgsino g_L=g_R=-1/2; Y=1 triplet: (T3,Q) = (1,2),(0,1),(-1,0)",
    "SU(2)xU(1) quantum numbers", "certain")
V_MU = -0.5 + 2*S2W; A_MU = -0.5
def chi_Z(s):
    return s/(4*S2W*C2W*(s - MZ**2 + 1j*MZ*GZ))
def sigma_ff(s, m, Q, gL, gR, alpha=ALPHA_MZ, ncol=1):
    """sigma(l+l- -> f fbar) in fb for a Dirac fermion of mass m, charge Q, Z couplings gL, gR."""
    if s <= 4*m**2: return 0.0
    beta = np.sqrt(1 - 4*m**2/s); v = gL + gR; a = gL - gR; c = chi_Z(s)
    vv = Q**2 - 2*Q*V_MU*v*c.real + (V_MU**2 + A_MU**2)*v**2*abs(c)**2
    aa = (V_MU**2 + A_MU**2)*a**2*abs(c)**2
    return ncol*(4*np.pi*alpha**2/(3*s))*beta*((3 - beta**2)/2*vv + beta**2*aa)*GEV2_TO_FB

def muc_xsecs(m, sqrt_s=10000.):
    s = sqrt_s**2
    d = dict(m_GeV=m, sqrt_s_TeV=sqrt_s/1000)
    d['doublet_chipm_fb'] = sigma_ff(s, m, 1, 0.5 - S2W, 0.5 - S2W)
    d['doublet_chi1chi2_fb'] = sigma_ff(s, m, 0, -0.5, -0.5)        # Dirac psi0 = (chi1 + i chi2)/sqrt2: pure off-diagonal current
    d['doublet_total_fb'] = d['doublet_chipm_fb'] + d['doublet_chi1chi2_fb']
    d['tripletY1_Q1_fb'] = sigma_ff(s, m, 1, -S2W, -S2W)
    d['tripletY1_Q2_fb'] = sigma_ff(s, m, 2, 1 - 2*S2W, 1 - 2*S2W)
    d['tripletY1_chi1chi2_fb'] = sigma_ff(s, m, 0, -1.0, -1.0)
    d['tripletY1_total_fb'] = d['tripletY1_Q1_fb'] + d['tripletY1_Q2_fb'] + d['tripletY1_chi1chi2_fb']
    d['tripletY1_charged_fb'] = d['tripletY1_Q1_fb'] + d['tripletY1_Q2_fb']
    d['ratio_triplet_over_doublet_total'] = d['tripletY1_total_fb']/d['doublet_total_fb']
    d['ratio_triplet_over_doublet_charged'] = d['tripletY1_charged_fb']/d['doublet_chipm_fb']
    d['betagamma_chargino'] = np.sqrt(s/4 - m**2)/m
    d['sigma_point_fb'] = 4*np.pi*ALPHA_MZ**2/(3*s)*GEV2_TO_FB
    return d
muc_rows = [muc_xsecs(m) for m in [300, 500, 800, 1000, 1100, 1500, 2000, 3000, 4000, 4500]]
muc_rows += [muc_xsecs(1100, 3000.), muc_xsecs(1100, 14000.), muc_xsecs(1100, 30000.)]
dfM = pd.DataFrame(muc_rows); dfM.to_csv(f'{OUT}/muon_collider_xsec.csv', index=False)
log(dfM[['m_GeV','sqrt_s_TeV','doublet_chipm_fb','doublet_chi1chi2_fb','doublet_total_fb','tripletY1_total_fb','ratio_triplet_over_doublet_total','betagamma_chargino']].to_string(float_format=lambda x: f'{x:.4g}'))
m11 = muc_xsecs(1100)
# coupling-factor decomposition at 10 TeV for the chargino
s = 1e8; c = chi_Z(s); v = 1 - 2*S2W
log(f"chargino coupling factor at 10 TeV: photon 1, interference {-2*1*V_MU*v*c.real:.4f}, Z^2 {(V_MU**2+A_MU**2)*v**2*abs(c)**2:.4f}; |chi|->{abs(c):.4f}; sigma_point = {m11['sigma_point_fb']:.3f} fb")
log(f"1.1 TeV at 10 TeV: chi+chi- {m11['doublet_chipm_fb']:.3f} fb, chi1chi2 {m11['doublet_chi1chi2_fb']:.3f} fb -> {m11['doublet_total_fb']*1e4:.0f} pairs in 10 ab^-1; beta gamma = {m11['betagamma_chargino']:.2f}")
# massless-limit check of the formula: R-ratio style
chk = sigma_ff(s, 0.0, -1, -0.5 + S2W, S2W)/m11['sigma_point_fb']
chk_analytic = 1 + 2*V_MU**2*c.real + (V_MU**2 + A_MU**2)**2*abs(c)**2
log(f"check: mu+mu- -> e+e- (massless) / sigma_point at 10 TeV = {chk:.4f} (analytic 1 + 2 v^2 Re chi + (v^2+a^2)^2 |chi|^2 = {chk_analytic:.4f})")

# ----------------------------------------------------------------------------------------------
# Part B: beta*gamma_T distributions, tracklet survival, reach
# ----------------------------------------------------------------------------------------------
log("="*100); log("PART B: toy Drell-Yan kinematics, tracklet survival, expected tracklets and reach")
rec("Tracklet radii: ATLAS Run-2 pixel tracklet needs 4 hits, r >~ 12 cm; ITk/FCC-hh 'short tracklet' proposals r ~ 5-10 cm; CMS-like tracks r >~ 30 cm; MuC vertex detector innermost layer ~3 cm with 10-degree nozzles",
    "ATLAS EPJC 82 (2022) 606; Fukuda et al. 2018; Saito et al. 2019; FCC-hh CDR; MuC detector concept (Capdevilla et al. 2021)", "uncertain")
rec("ATLAS 136 fb^-1 disappearing-track wino exclusion 660 GeV (tau=0.2 ns), ~5 selected events for exclusion; fake-tracklet backgrounds O(few-10) events",
    "ATLAS EPJC 82 (2022) 606 via P014", "likely (masses) / uncertain (event counts)")
rec("Projected tracklet backgrounds: HL-LHC 3 ab^-1 ~ 30 (10-100) events; FCC-hh 30 ab^-1 ~ 100 (10-1000, pile-up 1000); MuC ~ 1-10 after BIB rejection",
    "scaling of ATLAS Run-2 counts; Saito et al. 2019; Capdevilla et al. 2021", "uncertain")
rec("Fixed-order angular distribution for vector-coupled fermion pairs d sigma/d cos theta ~ 2 - beta^2 sin^2 theta; pair-mass spectrum from dL/dtau ~ tau^-a with a = -slope/2",
    "own derivation from the sigma_ff formula; luminosity power law fitted to the P014 anchor slopes", "likely")

# P014 Higgsino c tau vs mass (cm) - interpolate in log-log
P014_M = np.array([300, 500, 1000, 2000, 4000.]); P014_CT = np.array([0.9531, 0.8109, 0.7117, 0.6644, 0.6413])
def ctau_H(m):
    return np.exp(np.interp(np.log(np.clip(m, 300, 4000)), np.log(P014_M), np.log(P014_CT)))
CT_WINO_660 = 6.823   # cm, P014 chargino_lifetimes.csv
CT_TRIPLET_Y1 = 0.7117*(342.2/511.)**3/2.   # Q=1 -> Q=0 of the Y=1 triplet: kappa^2 = 2, Delta m 511 MeV (P037)
log(f"c tau: Higgsino 1.1 TeV {ctau_H(1100):.3f} cm; Y=1 triplet Q=1 state (Delta m 511 MeV, kappa^2=2) {CT_TRIPLET_Y1:.3f} cm")

def sample_betagamma_T(m, sqrt_s_TeV, n=200000, slope=None):
    """Toy Drell-Yan: pair mass M with p(M) ~ M^-(2a+1) beta_M (3-beta_M^2)/2, a = -slope/2; cos theta* ~ 2 - beta^2 sin^2;
    returns transverse beta*gamma = p_T/m = (M beta_M / 2m) sin theta*. Pair p_T (ISR) neglected."""
    if slope is None:
        slope = slope_13(max(m*13/sqrt_s_TeV, 60))
    a = -slope/2
    s = (sqrt_s_TeV*1000)**2
    Mgrid = np.linspace(2*m*(1+1e-6), min(np.sqrt(s), 20*m), 4000)
    bM = np.sqrt(1 - 4*m**2/Mgrid**2)
    pM = Mgrid**(-(2*a+1))*bM*(3 - bM**2)/2
    cdf = np.cumsum(pM); cdf /= cdf[-1]
    M = np.interp(rng.random(n), cdf, Mgrid)
    bM = np.sqrt(1 - 4*m**2/M**2)
    # rejection sample cos theta*
    ct = np.empty(n); filled = 0
    while filled < n:
        c_try = rng.uniform(-1, 1, 2*n); u = rng.uniform(0, 2, 2*n)
        w = 2 - bM[np.minimum(np.arange(2*n) % n, n-1)]**2*(1 - c_try**2)
        ok = u < w
        take = c_try[ok][:n-filled]; ct[filled:filled+len(take)] = take; filled += len(take)
    bgT = (M*bM/(2*m))*np.sqrt(1 - ct**2)
    return bgT, a

def surv_frac(bgT, ctau, rmin):
    return np.mean(np.exp(-rmin/np.maximum(bgT*ctau, 1e-12)))

# calibration: wino 660 GeV at 13 TeV, r_min = 12 cm
bg_w, a_w = sample_betagamma_T(660, 13)
P_w = surv_frac(bg_w, CT_WINO_660, 12.0)
N_prod_w = sig_wino_tot(660, 13)*136.   # produced C1N2+C1C1 events
n_ch_per_event = 1.5/1.5*(1*1.0 + 1*2.0)/2   # C1N2 (1 chargino) and C1C1 (2) with sigma ratio 1:0.5 -> weighted 4/3
n_ch_per_event = (1.0*1 + 0.5*2)/1.5
eps_kin = 5.0/(N_prod_w*n_ch_per_event*P_w)
log(f"wino calibration (660 GeV, 13 TeV): a={a_w:.2f}, median betagamma_T={np.median(bg_w):.2f}, P(r_T>12 cm)={P_w:.3f}, produced {N_prod_w:.0f} events, "
    f"eps_kin = 5/(N n_ch P) = {eps_kin:.4f}")
rec("eps_kin calibration assumes the ATLAS wino exclusion corresponds to ~5 selected signal events and 1.5 x C1N2 produced", "P014 toy", "uncertain (factor 2)")

# beta gamma_T distributions for the Higgsino at the three machines
quant_rows = []
bg_store = {}
for (lab, sqs) in [('HL-LHC 14 TeV', 14), ('FCC-hh 100 TeV', 100)]:
    for m in [300, 500, 1000, 1100, 1500, 2000]:
        bg, a = sample_betagamma_T(m, sqs); bg_store[(lab, m)] = bg
        q = np.percentile(bg, [16, 50, 84, 99])
        quant_rows.append(dict(machine=lab, m_GeV=m, a_lum=a, bgT_p16=q[0], bgT_p50=q[1], bgT_p84=q[2], bgT_p99=q[3],
                               frac_bgT_gt_2=np.mean(bg > 2), frac_bgT_gt_3=np.mean(bg > 3)))
dfQ = pd.DataFrame(quant_rows); dfQ.to_csv(f'{OUT}/betagamma_T_quantiles.csv', index=False)
log(dfQ.to_string(float_format=lambda x: f'{x:.3g}'))

# survival table for the Higgsino (c tau at each mass) at r_min = 5, 10, 12, 30 cm
surv_rows = []
for (lab, sqs) in [('HL-LHC 14 TeV', 14), ('FCC-hh 100 TeV', 100)]:
    for m in [300, 500, 1000, 1100, 1500, 2000]:
        bg = bg_store[(lab, m)]; ct = ctau_H(m)
        d = dict(machine=lab, m_GeV=m, ctau_cm=ct)
        for r in [5, 10, 12, 30]:
            d[f'P_rT_gt_{r}cm'] = surv_frac(bg, ct, r)
        d['P_rT_gt_5cm_tripletY1'] = surv_frac(bg, CT_TRIPLET_Y1, 5)
        surv_rows.append(d)
# muon collider: fixed beta gamma, angular distribution 2 - beta^2 sin^2 theta, transverse decay length = betagamma ctau sin theta
def muc_survival(m, ctau, rmin, sqrt_s=10000., theta_min_deg=10.):
    s = sqrt_s**2; beta = np.sqrt(1 - 4*m**2/s); bg = beta/np.sqrt(1 - beta**2)
    th = np.linspace(np.radians(theta_min_deg), np.pi - np.radians(theta_min_deg), 4001)
    w = (2 - beta**2*np.sin(th)**2)*np.sin(th)
    P = np.exp(-rmin/(bg*ctau*np.sin(th)))
    return np.trapezoid(w*P, th)/np.trapezoid(w, th), np.trapezoid(w, th)/np.trapezoid((2 - beta**2*np.sin(th)**2)*np.sin(th), np.linspace(1e-6, np.pi-1e-6, 4001))
for m in [300, 500, 1000, 1100, 1500, 2000, 3000, 4000]:
    ct = ctau_H(m); d = dict(machine='MuC 10 TeV', m_GeV=m, ctau_cm=ct)
    for r in [5, 10, 12, 30]:
        d[f'P_rT_gt_{r}cm'] = muc_survival(m, ct, r)[0]
    d['P_rT_gt_5cm_tripletY1'] = muc_survival(m, CT_TRIPLET_Y1, 5)[0]
    d['betagamma'] = np.sqrt(1e8/4 - m**2)/m; d['acceptance_theta_gt_10deg'] = muc_survival(m, ct, 5)[1]
    surv_rows.append(d)
dfS = pd.DataFrame(surv_rows); dfS.to_csv(f'{OUT}/tracklet_survival.csv', index=False)
log(dfS.to_string(float_format=lambda x: f'{x:.3g}'))

# expected tracklets and reach vs mass
mgrid = np.concatenate([np.arange(250, 1000, 25), np.arange(1000, 4600, 50)]).astype(float)
def n_charginos_hadron(m, sqs, lumi_fb):
    # chi+-chi0 (1 chargino) with sigma_cn, chi+chi- (2 charginos) ~ sigma_cn/3, chi1chi2 (0) ~ sigma_cn/6  [total = 1.5 sigma_cn]
    scn = sig_H_cn(m, sqs); return lumi_fb*(scn*1 + scn/3*2)
rec("Higgsino channel split of the extra 50%: chi+chi- ~ 1/3 and chi1chi2 ~ 1/6 of the chi+-chi0 rate", "own estimate from Z/W couplings", "uncertain")
BKG = {'HL-LHC 14 TeV': (30., 10., 100.), 'FCC-hh 100 TeV': (100., 10., 1000.), 'MuC 10 TeV': (3., 1., 30.)}
LUMI = {'HL-LHC 14 TeV': 3000., 'FCC-hh 100 TeV': 30000., 'MuC 10 TeV': 10000.}
RMINS = {'HL-LHC 14 TeV': [12, 5], 'FCC-hh 100 TeV': [10, 5], 'MuC 10 TeV': [10, 5]}
reach_curves = {}
reach_rows = []
for lab in LUMI:
    for r in RMINS[lab]:
        S = []
        for m in mgrid:
            ct = ctau_H(m)
            if lab == 'MuC 10 TeV':
                if m >= 5000: S.append(0); continue
                nch = 2*muc_xsecs(m)['doublet_chipm_fb']*LUMI[lab]
                P = muc_survival(m, ct, r)[0]
                S.append(nch*P*0.5)     # 50% tracklet reconstruction/quality efficiency at a lepton collider (assumption)
            else:
                sqs = 14 if 'HL' in lab else 100
                bg, _ = sample_betagamma_T(m, sqs, n=60000)
                S.append(n_charginos_hadron(m, sqs, LUMI[lab])*surv_frac(bg, ct, r)*eps_kin)
        S = np.array(S); reach_curves[(lab, r)] = S
        for B, tag in zip(BKG[lab], ['central', 'low', 'high']):
            excl = S >= np.maximum(2*np.sqrt(B), 3.0); disc = S >= np.maximum(5*np.sqrt(B), 5.0)
            m_ex = mgrid[excl].max() if excl.any() else np.nan; m_di = mgrid[disc].max() if disc.any() else np.nan
            reach_rows.append(dict(machine=lab, r_min_cm=r, bkg_events=B, bkg_case=tag, reach_95CL_GeV=m_ex, reach_5sigma_GeV=m_di,
                                   S_at_1100=np.interp(1100, mgrid, S), S_at_300=np.interp(300, mgrid, S), S_at_500=np.interp(500, mgrid, S)))
rec("MuC tracklet reconstruction efficiency 50% after BIB-driven quality cuts", "assumption motivated by Capdevilla et al. 2021", "uncertain")
dfR = pd.DataFrame(reach_rows); dfR.to_csv(f'{OUT}/reach_vs_mass.csv', index=False)
log(dfR.to_string(float_format=lambda x: f'{x:.4g}'))
# cross-section band: rescale S by the recalled cross-section uncertainty (x/÷2 HL-LHC, x/÷3 FCC-hh, ±10% MuC tree level)
XS_BAND = {'HL-LHC 14 TeV': (0.5, 2.0), 'FCC-hh 100 TeV': (1/3, 3.0), 'MuC 10 TeV': (0.9, 1.1)}
band_rows = []
for (lab, r), S in reach_curves.items():
    B = BKG[lab][0]
    for f, tag in zip(XS_BAND[lab], ['xsec_low', 'xsec_high']):
        Sf = S*f
        excl = Sf >= np.maximum(2*np.sqrt(B), 3.0); disc = Sf >= np.maximum(5*np.sqrt(B), 5.0)
        band_rows.append(dict(machine=lab, r_min_cm=r, xsec_factor=f, case=tag, bkg_events=B,
                              reach_95CL_GeV=mgrid[excl].max() if excl.any() else np.nan,
                              reach_5sigma_GeV=mgrid[disc].max() if disc.any() else np.nan, S_at_1100=np.interp(1100, mgrid, Sf)))
dfRB = pd.DataFrame(band_rows); dfRB.to_csv(f'{OUT}/reach_xsec_band.csv', index=False)
log("reach under cross-section rescaling:"); log(dfRB.to_string(float_format=lambda x: f'{x:.4g}'))
pd.DataFrame({'m_GeV': mgrid, **{f'{k[0]}_r{k[1]}cm': v for k, v in reach_curves.items()}}).to_csv(f'{OUT}/expected_tracklets_vs_mass.csv', index=False)

# ----------------------------------------------------------------------------------------------
# Part C: mono-photon at the muon collider (collinear ISR estimate) and nu nu gamma background
# ----------------------------------------------------------------------------------------------
log("="*100); log("PART C: mono-photon at 10 TeV MuC")
rec("Collinear ISR: dP/dx dtheta^2 = (alpha/2pi)(1+(1-x)^2)/x / theta^2, integrated over 10-170 degrees -> ln(1/theta_min^2) ~ 3.5 per beam",
    "Weizsaecker-Williams / electron structure function", "likely (factor ~2 for wide-angle photons)")
rec("sigma(l+l- -> nu nubar, 3 flavours) at multi-TeV ~ 50-100 pb, dominated by t-channel W exchange (asymptotically G_F^2 mW^2/pi ~ 110 pb scale)",
    "standard result; own estimate", "uncertain (factor 2)")
rec("Literature MuC reach: 10 TeV, 10 ab^-1: mono-photon 95% CL reach covers/nears the 1.1 TeV Higgsino (Han-Liu-Wang-Wang 2020); disappearing tracks discover it at 5 sigma (Capdevilla-Meloni-Simoniello-Zurita 2021); 3 TeV MuC excludes it",
    "arXiv:2009.11287; JHEP 06 (2021) 133", "likely (qualitative), uncertain (numbers)")
def sigma_monophoton(m, sqrt_s=10000., Egamma_min=100., theta_min_deg=10., lumi_ab=10.):
    s = sqrt_s**2; xmin = 2*Egamma_min/sqrt_s; xmax = 1 - 4*m**2/s
    L = np.log(1/np.radians(theta_min_deg)**2)
    def integrand(x):
        return 2*(ALPHA_MZ/(2*np.pi))*L*(1 + (1-x)**2)/x*(sigma_ff(s*(1-x), m, 1, 0.5 - S2W, 0.5 - S2W) + sigma_ff(s*(1-x), m, 0, -0.5, -0.5))
    sig, _ = integrate.quad(integrand, xmin, xmax*0.999, limit=200)
    # background: nu nubar gamma with the same photon cuts, sigma(nu nu) taken constant = 70 pb (uncertain)
    def bkg_int(x):
        return 2*(ALPHA_MZ/(2*np.pi))*L*(1 + (1-x)**2)/x*70e3
    bkg, _ = integrate.quad(bkg_int, xmin, xmax*0.999)
    return sig, bkg, sig*lumi_ab*1000, bkg*lumi_ab*1000
mono_rows = []
for m in [500, 1000, 1100, 1500, 2000]:
    sg, bk, S, B = sigma_monophoton(m)
    mono_rows.append(dict(m_GeV=m, sigma_gamma_signal_fb=sg, sigma_gamma_bkg_fb=bk, S_10ab=S, B_10ab=B, S_over_sqrtB=S/np.sqrt(B),
                          frac_of_pair_xsec=sg/muc_xsecs(m)['doublet_total_fb']))
dfC = pd.DataFrame(mono_rows); dfC.to_csv(f'{OUT}/monophoton_muc.csv', index=False)
log(dfC.to_string(float_format=lambda x: f'{x:.4g}'))

# ----------------------------------------------------------------------------------------------
# Part D: delta-blindness
# ----------------------------------------------------------------------------------------------
log("="*100); log("PART D: delta-blindness of colliders")
DELTA = 366.0  # keV, P007 best fit at 1 TeV
TAU_G = 0.059; TAU_NN = 4.57e5   # s, P014 chi2_decays.csv (1 TeV)
rec("chi2 lifetimes: tau(chi2->chi1 gamma) ~ 0.06 s (loop dipole, uncertain O(1)), tau(chi2->chi1 nu nubar) = 4.6e5 s (P014)", "corpus P014", "corpus (photon channel uncertain)")
def lab_photon_range(gamma, delta=DELTA):
    beta = np.sqrt(1 - 1/gamma**2); return gamma*(1-beta)*delta, gamma*(1+beta)*delta, gamma*delta
d_rows = []
for gam, lab in [(1.5, 'slow'), (3.0, 'typical hadron collider'), (4.55, 'MuC 10 TeV, 1.1 TeV'), (30, 'extreme tail')]:
    lo, hi, mean = lab_photon_range(gam)
    beta = np.sqrt(1 - 1/gam**2); L_g = gam*beta*3e8*TAU_G; L_nn = gam*beta*3e8*TAU_NN
    d_rows.append(dict(gamma=gam, case=lab, E_lab_min_keV=lo, E_lab_max_keV=hi, E_lab_mean_keV=mean,
                       decay_length_gamma_channel_m=L_g, decay_length_nunu_channel_m=L_nn,
                       P_decay_within_10m_gamma=10/L_g, P_decay_within_10m_nunu=10/L_nn))
dfD = pd.DataFrame(d_rows); dfD.to_csv(f'{OUT}/chi2_lab_decay.csv', index=False)
log(dfD.to_string(float_format=lambda x: f'{x:.3g}'))
# chi2 yield and in-detector decays
n_chi2_muc = m11['doublet_chi1chi2_fb']*1e4
n_chi2_fcc = (0.5*sig_H_cn(1100, 100) + sig_H_cn(1100, 100)/6)*30000
log(f"chi2 produced: MuC 10 ab^-1 {n_chi2_muc:.0f}; FCC-hh 30 ab^-1 {n_chi2_fcc:.3g}; in-detector (10 m) photon decays: MuC {n_chi2_muc*10/(4.55*0.98*3e8*TAU_G):.2e}, FCC-hh {n_chi2_fcc*10/(3*0.94*3e8*TAU_G):.2e}")
log(f"e+e- channel closed: delta = {DELTA} keV < 2 m_e = {2*M_E_KEV:.0f} keV")

# Delta m(+-) -> delta bound. Tree coefficients from P014 (tree piece / delta): 0.096e-3 .. 2.126e-3
rec("P014 tree-level (Delta m+- - Delta m_rad)/delta = (0.1-2.1)e-3 depending on gaugino hierarchy and tan beta; Delta m_rad theory uncertainty +-10 MeV (scheme), two-loop could reduce to ~+-3 MeV",
    "corpus P014", "corpus")
TREE_COEF = {'wino-only tanb=2': 0.096e-3, 'M2=2M1 tanb=2': 0.395e-3, 'bino-only tanb=2': 0.885e-3, 'M2=-2M1 tanb=2': 2.126e-3, 'bino-only tanb=10': 0.595e-3}
dm_rows = []
for th_err_MeV, exp_err_MeV, tag in [(10, 0, 'one-loop theory only'), (3, 3, 'two-loop + MuC c tau (1%)'), (1, 1, 'aspirational')]:
    tot = np.hypot(th_err_MeV, exp_err_MeV)
    for k, cft in TREE_COEF.items():
        dm_rows.append(dict(case=tag, hierarchy=k, sigma_Dm_MeV=tot, delta_bound_GeV=tot*1e-3/cft, ratio_to_LZ_delta=(tot*1e-3/cft)/(DELTA*1e-6)))
dfDm = pd.DataFrame(dm_rows); dfDm.to_csv(f'{OUT}/deltam_to_delta_bound.csv', index=False)
log(dfDm.to_string(float_format=lambda x: f'{x:.3g}'))
# MuC c tau measurement: with N tracklets and known betagamma, sigma(ctau)/ctau ~ 1/sqrt(N) (exponential fit); Delta m ~ ctau^-1/3
N_trk_muc = reach_curves[('MuC 10 TeV', 5)][np.argmin(abs(mgrid - 1100))]
log(f"MuC 1.1 TeV: expected tracklets (r>5 cm, 50% eff) = {N_trk_muc:.0f} -> sigma(ctau)/ctau ~ {1/np.sqrt(N_trk_muc):.3f}, sigma(Delta m)/Delta m ~ {1/3/np.sqrt(N_trk_muc):.4f} -> {342*1/3/np.sqrt(N_trk_muc):.2f} MeV")

# W parameter (FCC-ee etc.)
rec("Oblique W parameter from a heavy vector-coupled fermion multiplet: W = g^2 mW^2 Sum(T3^2)/(60 pi^2 m^2) (per Dirac fermion, from the q^4 term of the vacuum polarisation, alpha/(15 pi) q^2/m^2); FCC-ee/CEPC projected sensitivity |W| ~ 3e-5 (LEP ~ 1e-3)",
    "Barbieri-Pomarol-Rattazzi-Strumia 2004 definitions; own one-loop expansion; Di Luzio-Groeber-Panico JHEP 2018 for sensitivity", "likely (formula) / uncertain (sensitivity)")
def W_param(m, sumT3sq):
    return G2**2*MW**2*sumT3sq/(60*np.pi**2*m**2)
W_rows = []
for m in [300, 500, 1100, 2000]:
    W_rows.append(dict(m_GeV=m, W_doublet=W_param(m, 0.5), W_tripletY1=W_param(m, 2.0)))
dfW = pd.DataFrame(W_rows); dfW.to_csv(f'{OUT}/W_parameter.csv', index=False)
log(dfW.to_string())
for sens in [3e-5, 1e-5]:
    log(f"FCC-ee sensitivity |W|<{sens:.0e}: doublet reach m < {np.sqrt(G2**2*MW**2*0.5/(60*np.pi**2*sens)):.0f} GeV; Y=1 triplet {np.sqrt(G2**2*MW**2*2/(60*np.pi**2*sens)):.0f} GeV")

# ----------------------------------------------------------------------------------------------
# Part E: identification logic and timeline
# ----------------------------------------------------------------------------------------------
log("="*100); log("PART E: identification and timeline")
rec("P007: delta(N=1) = 366 (1 TeV), 375 (2 TeV) keV; N falls x5 per 10 keV near delta_max; P018 halo systematic on delta -17/+11 keV",
    "corpus P007/P018", "corpus")
ddelta_dlnm = (375 - 366)/np.log(2)     # keV per e-fold in m
for frac in [0.05, 0.2]:
    log(f"mass error {frac*100:.0f}% -> delta shift {ddelta_dlnm*frac:.2f} keV (vs halo systematic ~10-17 keV)")
for N in [1, 4, 10, 30]:
    log(f"N_DD = {N}: statistical sigma(delta) = {10/np.log(5)/np.sqrt(N):.1f} keV")
# collider multiplet discrimination via sigma_prod: ratio triplet/doublet at 10 TeV (total) and (charged only)
log(f"10 TeV, 1.1 TeV: sigma total doublet {m11['doublet_total_fb']:.3f} fb vs Y=1 triplet {m11['tripletY1_total_fb']:.3f} fb (ratio {m11['ratio_triplet_over_doublet_total']:.2f}); charged-only ratio {m11['ratio_triplet_over_doublet_charged']:.2f}")
log(f"sigma_prod to 20% discriminates doublet from Y=1 triplet at {np.log(m11['ratio_triplet_over_doublet_total'])/0.2:.1f} sigma (log-ratio / 0.2)")

timeline = [
    dict(facility='HL-LHC (14 TeV, 3 ab^-1)', years='2030-2041 (recalled, likely)', doublet_reach='disappearing tracks: ' +
         f"{dfR[(dfR.machine=='HL-LHC 14 TeV')&(dfR.r_min_cm==12)&(dfR.bkg_case=='central')].reach_95CL_GeV.iloc[0]:.0f} GeV (12 cm), {dfR[(dfR.machine=='HL-LHC 14 TeV')&(dfR.r_min_cm==5)&(dfR.bkg_case=='central')].reach_95CL_GeV.iloc[0]:.0f} GeV (5 cm); monojet ~200 GeV (recalled)",
         tripletY1_reach='c tau 0.1 cm: no tracklets; monojet-like only', verdict='below or at the LZ floor (259 GeV); cannot test the fit'),
    dict(facility='FCC-ee / CEPC / ILC (Z pole, WW, 250-500 GeV)', years='2045-2065 (recalled, uncertain)',
         doublet_reach=f"W parameter: m < {np.sqrt(G2**2*MW**2*0.5/(60*np.pi**2*3e-5)):.0f} GeV; direct pair production < sqrt(s)/2 = 125-250 GeV",
         tripletY1_reach=f"W: m < {np.sqrt(G2**2*MW**2*2/(60*np.pi**2*3e-5)):.0f} GeV", verdict='no sensitivity to the LZ region'),
    dict(facility='FCC-hh (100 TeV, 30 ab^-1)', years='2070s (recalled, likely)', doublet_reach='disappearing tracks: ' +
         f"{dfR[(dfR.machine=='FCC-hh 100 TeV')&(dfR.r_min_cm==10)&(dfR.bkg_case=='central')].reach_95CL_GeV.iloc[0]:.0f} GeV (10 cm), {dfR[(dfR.machine=='FCC-hh 100 TeV')&(dfR.r_min_cm==5)&(dfR.bkg_case=='central')].reach_95CL_GeV.iloc[0]:.0f} GeV (5 cm) at 95% CL; monojet ~0.9 TeV (recalled)",
         tripletY1_reach='monojet/soft pions; 4x production; no tracklets', verdict='exclusion of the thermal Higgsino plausible, 5 sigma marginal; measures m only if discovered'),
    dict(facility='Muon collider (10 TeV, 10 ab^-1)', years='2050s at the earliest (recalled, uncertain)', doublet_reach=
         f"{m11['doublet_total_fb']*1e4:.0f} pairs; {N_trk_muc:.0f} tracklets r>5 cm at 1.1 TeV; 5 sigma to ~{dfR[(dfR.machine=='MuC 10 TeV')&(dfR.r_min_cm==5)&(dfR.bkg_case=='central')].reach_5sigma_GeV.iloc[0]:.0f} GeV; m, sigma_prod (multiplet), c tau (Delta m+-)",
         tripletY1_reach=f"{m11['tripletY1_total_fb']*1e4:.0f} pairs, mono-photon/soft pions; no tracklets (c tau 0.1 cm)", verdict='full coverage of every P007 fit point; measures m, Y, Delta m+-; blind to delta'),
]
dfT = pd.DataFrame(timeline); dfT.to_csv(f'{OUT}/timeline_table.csv', index=False)
log(dfT.to_string())

# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
C = {'HL-LHC 14 TeV': '#2a78d6', 'FCC-hh 100 TeV': '#eb6834', 'MuC 10 TeV': '#1baf7a'}
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
for (lab, r), S in reach_curves.items():
    ls = '-' if r == RMINS[lab][0] else '--'
    ax.plot(mgrid, np.maximum(S, 1e-3), ls, color=C[lab], lw=2, label=f'{lab}, r > {r} cm')
for lab in LUMI:
    B = BKG[lab][0]; ax.axhline(5*np.sqrt(B), color=C[lab], lw=0.8, alpha=0.5)
    ax.text(4400, 5*np.sqrt(B)*1.15, f'5√B, B={B:.0f}', color=C[lab], fontsize=7, ha='right')
ax.axvspan(100, 259, color='#e0e0dc', alpha=0.6, lw=0); ax.text(120, 3e5, 'LZ floor\nm ≥ 259 GeV', fontsize=7.5, color='#52514e')
ax.axvline(1100, color='#52514e', lw=1, ls=':'); ax.text(1130, 3e5, 'thermal Higgsino 1.1 TeV', fontsize=7.5, rotation=90, va='top', color='#52514e')
for mfit in [300, 500, 1000, 2000, 4000]:
    ax.plot(mfit, 1.5e-3, marker='^', color='#0b0b0b', ms=5, clip_on=False)
ax.text(2300, 2.2e-3, 'P007 fit points', fontsize=7.5)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(100, 4600); ax.set_ylim(1e-3, 1e6)
ax.set_xlabel('Higgsino mass m [GeV]'); ax.set_ylabel('expected disappearing-track events (cτ from P014)')
ax.set_title('LZ-fitting Higgsino (Δm± = 342 MeV, cτ = 0.71 cm): tracklet yields at future colliders', fontsize=9.5)
ax.grid(alpha=0.25, which='both', lw=0.4); ax.legend(fontsize=7.5, loc='upper right', ncol=1, framealpha=0.9)
fig.tight_layout(); fig.savefig(f'{FIG}/P048_fig1_reach_vs_mass.png'); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=150)
ax = axs[0]
bgs = np.logspace(-1, 1.3, 200)
for r, ls in [(5, '-'), (10, '--'), (12, ':'), (30, '-.')]:
    ax.plot(bgs, np.exp(-r/(bgs*0.7117)), ls, color='#2a78d6', lw=1.8, label=f'Higgsino cτ = 0.71 cm, r > {r} cm')
ax.plot(bgs, np.exp(-12/(bgs*6.82)), '-', color='#eb6834', lw=1.8, label='wino cτ = 6.8 cm, r > 12 cm')
ax.plot(bgs, np.exp(-5/(bgs*CT_TRIPLET_Y1)), '-', color='#4a3aa7', lw=1.8, label=f'Y=1 triplet cτ = {CT_TRIPLET_Y1:.2f} cm, r > 5 cm')
ax.axvline(m11['betagamma_chargino'], color='#1baf7a', lw=1.2); ax.text(m11['betagamma_chargino']*1.08, 3e-3, 'MuC 10 TeV, 1.1 TeV:\nβγ = 4.4', color='#1baf7a', fontsize=6.8, va='center')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-6, 1); ax.set_xlim(0.1, 20)
ax.set_xlabel('transverse βγ = p_T / m'); ax.set_ylabel('P(decay radius > r)'); ax.legend(fontsize=6.5, loc='lower right'); ax.grid(alpha=0.25, which='both', lw=0.4)
ax.set_title('tracklet survival', fontsize=9.5)
ax = axs[1]
for lab, ls in [('HL-LHC 14 TeV', '-'), ('FCC-hh 100 TeV', '--')]:
    for m, al in [(300, 0.45), (1100, 1.0)]:
        bg = bg_store[(lab, m)]
        h, e = np.histogram(np.log10(bg), bins=60, range=(-1.5, 1.5), density=True)
        ax.step(10**(0.5*(e[1:]+e[:-1])), h, ls, color=C[lab], alpha=al, lw=1.6, label=f'{lab}, m = {m} GeV')
ax.axvline(m11['betagamma_chargino'], color='#1baf7a', lw=1.5, label='MuC 10 TeV, m = 1.1 TeV (fixed)')
ax.set_xscale('log'); ax.set_xlim(0.03, 30); ax.set_xlabel('transverse βγ of the chargino'); ax.set_ylabel('density per dex')
ax.set_title('toy Drell–Yan βγ_T distributions', fontsize=9.5); ax.legend(fontsize=6.5); ax.grid(alpha=0.25, which='both', lw=0.4)
fig.tight_layout(); fig.savefig(f'{FIG}/P048_fig2_survival_betagamma.png'); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# Save summary
# ----------------------------------------------------------------------------------------------
summary = dict(
    xsec_14TeV_cn_fb={int(m): float(sig_H_cn(m, 14)) for m in masses}, xsec_100TeV_cn_fb={int(m): float(sig_H_cn(m, 100)) for m in masses},
    muc_1100=m11, eps_kin=float(eps_kin), wino_calib=dict(P=float(P_w), a=float(a_w), N_prod=float(N_prod_w), median_bgT=float(np.median(bg_w))),
    ctau_H_1100_cm=float(ctau_H(1100)), ctau_tripletY1_cm=float(CT_TRIPLET_Y1), N_trk_muc_1100_r5=float(N_trk_muc),
    n_chi2_muc=float(n_chi2_muc), n_chi2_fcc=float(n_chi2_fcc), ddelta_dlnm_keV=float(ddelta_dlnm),
    W_doublet_1100=float(W_param(1100, 0.5)), W_reach_doublet_sens3em5_GeV=float(np.sqrt(G2**2*MW**2*0.5/(60*np.pi**2*3e-5))))
json.dump(summary, open(f'{OUT}/P048_summary.json', 'w'), indent=1, default=float)
json.dump(RECALLED, open(f'{OUT}/recalled_knowledge.json', 'w'), indent=1)
open(f'{OUT}/run_log.txt', 'w').write('\n'.join(LOG))
log(f"recalled items: {len(RECALLED)}; done.")
