"""P073 -- What the vetoes know: MSSI tagging efficiency (wall/RFR split), the topology-specific
prompt-veto efficiency for the 12 keV (FV) + 77 keV (wall shell) decomposition of the LZ event,
the likelihood ratio carried by the event's silence in the Skin and OD, and the delayed-sample excess.

Run from the simulation root:  .venv/bin/python output/code/P073_veto_information.py
Outputs: output/work/P073/*.csv, *.json, figures/*.png
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy import stats, optimize

sys.path.insert(0, 'output/code')
from common import lzcommon as lz

OUT = 'output/work/P073'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(73)
T0 = time.time()

# ----------------------------------------------------------------------------------------------
# 0. Paper inputs (all from inputs/LZ_arXiv_2609.02823_fulltext.tex)
# ----------------------------------------------------------------------------------------------
# Samples paragraph (l. 141-147)
PROMPT_SKIN_PHD, PROMPT_SKIN_WIN_US = 2.5, 0.25   # > 2.5 phd within +-0.25 us
PROMPT_OD_PHD, PROMPT_OD_WIN_US = 4.5, 0.30       # > 4.5 phd within +-0.30 us
DELAYED_SKIN_KEV, DELAYED_OD_KEV, DELAYED_WIN_US = 300., 200., 600.
F_RAND_PROMPT, F_RAND_DELAYED = 1e-4, 0.0286      # random-coincidence veto fractions
# Table S1 (prompt sample) / Table I (science): expected counts
MSSI_SCI, MSSI_PROMPT, MSSI_DELAYED = 4.9e-3, 7.7e-2, 1.4e-4
DETER_SCI, DETER_PROMPT = 8.5, 62.2
LAM_PG, LAM_MSSI, LAM_MSSI_ERR = 0.88, 0.94, 0.02
LAM_PN, LAM_DN = 0.05, 0.87
N_SCI, N_PROMPT, N_DELAYED = 1710, 66, 55
FIT_DELAYED, FIT_DELAYED_ERR = 50.4, 2.0
FIT_PROMPT, FIT_PROMPT_ERR = 69.4, 11.7
# Supplement MSSI comparison table (l. 765-783): simulated / observed, science / prompt
MSSI_TABLE = pd.DataFrame([
    # region, type, vol, sim_sci, obs_sci, sim_prompt, obs_prompt
    ('WS ROI', 'Wall', 4.7, 0.0048, np.nan, 0.09, np.nan),
    ('WS ROI', 'RFR', 4.7, 0.0001, np.nan, 0.10, np.nan),
    ('WS ROI', 'Wall', 5.4, 0.03, 0, 0.1, 1),
    ('WS ROI', 'RFR', 5.4, 0.003, 0, 1.9, 2),
    ('HE SB', 'Wall', 4.7, 0.1, 0, 0.3, 1),
    ('HE SB', 'RFR', 4.7, 0.5, 0, 0.5, 0),
    ('HE SB', 'Wall', 5.4, 0.5, 0, 2.2, 0),
    ('HE SB', 'RFR', 5.4, 21.5, 18, 5.2, 3),
], columns=['region', 'type', 'vol_t', 'sim_sci', 'obs_sci', 'sim_prompt', 'obs_prompt'])
# Table S2 (delayed sample) expected components
DELAYED_EXPECTED = dict(internal_beta=39.5, nu_ER=4.1, Xe136=3.2, CH3T_C14=1.6, Xe124=0.62, Kr83m=0.50,
                        I125=0.26, detector_ER=0.25, accidentals=0.080, Xe127_125=0.045, atm_nu=0.0033,
                        B8_hep=0.0017, MSSI=1.4e-4)

# ----------------------------------------------------------------------------------------------
# 1. Tagging efficiencies from the paper's own numbers
# ----------------------------------------------------------------------------------------------
def binom_interval(k, n, cl=0.68):
    """Clopper-Pearson interval for efficiency k/n (k tagged of n)."""
    a = (1 - cl) / 2
    lo = 0.0 if k == 0 else stats.beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a, k + 1, n - k)
    return lo, hi

rows = []
for _, r in MSSI_TABLE.iterrows():
    eps_sim = r.sim_prompt / (r.sim_prompt + r.sim_sci)
    if np.isnan(r.obs_sci):
        eps_dat, lo, hi, ll90 = np.nan, np.nan, np.nan, np.nan
    else:
        n = int(r.obs_sci + r.obs_prompt)
        k = int(r.obs_prompt)
        eps_dat = k / n if n > 0 else np.nan
        lo, hi = binom_interval(k, n) if n > 0 else (np.nan, np.nan)
        ll90 = (0.0 if k == 0 else stats.beta.ppf(0.10, k, n - k + 1)) if n > 0 else np.nan
    rows.append(dict(region=r.region, type=r.type, vol_t=r.vol_t, sim_sci=r.sim_sci, sim_prompt=r.sim_prompt,
                     eps_sim=eps_sim, untagged_sim=1 - eps_sim, obs_sci=r.obs_sci, obs_prompt=r.obs_prompt,
                     eps_data=eps_dat, eps_data_68lo=lo, eps_data_68hi=hi, eps_data_90LL=ll90))
EFF = pd.DataFrame(rows)

# aggregated categories
def agg(mask, label):
    s = MSSI_TABLE[mask]
    eps_sim = s.sim_prompt.sum() / (s.sim_prompt.sum() + s.sim_sci.sum())
    o = s.dropna(subset=['obs_sci'])
    k, n = int(o.obs_prompt.sum()), int(o.obs_prompt.sum() + o.obs_sci.sum())
    lo, hi = binom_interval(k, n) if n else (np.nan, np.nan)
    ll90 = (0.0 if k == 0 else stats.beta.ppf(0.10, k, n - k + 1)) if n else np.nan
    # Poisson probability of the observed split given the simulated split (validation power)
    return dict(category=label, sim_sci=s.sim_sci.sum(), sim_prompt=s.sim_prompt.sum(), eps_sim=eps_sim,
                obs_sci=o.obs_sci.sum(), obs_prompt=o.obs_prompt.sum(), n_obs=n, eps_data=(k / n if n else np.nan),
                eps_68lo=lo, eps_68hi=hi, eps_90LL=ll90)
AGG = pd.DataFrame([
    agg((MSSI_TABLE.region == 'WS ROI') & (MSSI_TABLE.vol_t == 4.7), 'WS ROI 4.7 t (wall+RFR) [search bin]'),
    agg((MSSI_TABLE.type == 'Wall') & (MSSI_TABLE.region == 'WS ROI') & (MSSI_TABLE.vol_t == 4.7), 'WS ROI 4.7 t wall'),
    agg(MSSI_TABLE.type == 'Wall', 'all wall bins'),
    agg((MSSI_TABLE.type == 'Wall') & ~((MSSI_TABLE.region == 'WS ROI') & (MSSI_TABLE.vol_t == 4.7)), 'wall sidebands (6 bins)'),
    agg((MSSI_TABLE.type == 'RFR') & ~((MSSI_TABLE.region == 'HE SB') & (MSSI_TABLE.vol_t == 5.4)), 'RFR excl. HE-SB 5.4 t'),
    agg((MSSI_TABLE.type == 'RFR') & (MSSI_TABLE.region == 'HE SB') & (MSSI_TABLE.vol_t == 5.4), 'HE-SB 5.4 t RFR (214Pb)'),
])
# Tables S1/I implied efficiencies
eps_tables_S1 = MSSI_PROMPT / (MSSI_PROMPT + MSSI_SCI)          # 0.077/(0.077+0.0049)
eps_detER = DETER_PROMPT / (DETER_PROMPT + DETER_SCI)            # 62.2/(62.2+8.5)
prompt_from_lambda = MSSI_SCI * LAM_MSSI / (1 - LAM_MSSI)        # what lambda=0.94 implies for the prompt sample
ratio_table_vs_S1 = (0.09 + 0.10) / MSSI_PROMPT
# how well do the HE-SB 5.4 t RFR data test the simulated split? likelihood ratio eps_sim vs eps_ML
k, n = 3, 21
p_sim = EFF.eps_sim.iloc[7]
p_split = stats.binomtest(k, n, p_sim).pvalue
P1 = dict(eps_S1_tables=eps_tables_S1, eps_detER_tables=eps_detER, lambda_PG=LAM_PG, lambda_MSSI=LAM_MSSI,
          prompt_MSSI_from_lambda=prompt_from_lambda, prompt_MSSI_table_sum_4p7=0.19, ratio_table_vs_S1=ratio_table_vs_S1,
          eps_WSROI_4p7_table=float(AGG.eps_sim.iloc[0]), eps_WSROI_4p7_wall_table=float(AGG.eps_sim.iloc[1]),
          untagged_wall_table=1 - float(AGG.eps_sim.iloc[1]), untagged_lambda=1 - LAM_MSSI,
          HESB_RFR_binom_p=p_split, eps_wall_sidebands_data_90LL=float(AGG.eps_90LL.iloc[3]),
          eps_RFR_excl_data_90LL=float(AGG.eps_90LL.iloc[4]))
EFF.to_csv(os.path.join(OUT, 'P073_tagging_efficiency_table.csv'), index=False)
AGG.to_csv(os.path.join(OUT, 'P073_tagging_efficiency_aggregated.csv'), index=False)
print('--- Part 1: efficiencies from the paper ---')
print(EFF[['region', 'type', 'vol_t', 'eps_sim', 'obs_sci', 'obs_prompt', 'eps_data', 'eps_data_90LL']].to_string())
print(AGG.to_string())
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in P1.items()})

# ----------------------------------------------------------------------------------------------
# 2. Topology-specific tagging: photon transport of the outgoing gamma
# ----------------------------------------------------------------------------------------------
ME = 510.999      # keV (certain)
R_E2 = 7.9407e-26 # cm^2, classical electron radius squared (certain)
NA = 6.02214e23

def sigma_kn(E):
    """Klein-Nishina total cross-section per electron, cm^2 (E in keV)."""
    k = np.asarray(E, float) / ME
    t = np.log(1 + 2 * k)
    a = (1 + k) / k**2 * (2 * (1 + k) / (1 + 2 * k) - t / k) + t / (2 * k) - (1 + 3 * k) / (1 + 2 * k)**2
    return 2 * np.pi * R_E2 * a

# Materials: density (g/cm3), Z/A -> electron density; photoelectric handled for Xe (corpus anchors) and Ti (scaled)
RHO_LXE = 2.86                         # P070 convention (recalled, likely)
RHO_PMT0 = 0.8   # effective density of the PMT arrays (glass/Kovar/PTFE plate/Ti truss + vacuum): recalled/uncertain, bracketed 0.4-1.2
NE = dict(lxe=RHO_LXE * NA * 54 / 131.29, ptfe=2.2 * NA * 48 / 100.02, ti=4.5 * NA * 22 / 47.87,
          water=1.0 * NA * 10 / 18.015, gdls=0.86 * NA * 0.571, pmt=RHO_PMT0 * NA * 0.50, none=0.0)
# LXe total attenuation-length anchors (cm) from the corpus (P004 122 keV; P029 164-375 keV; P042 300 keV; P033 352/1000/2615 keV)
XE_ANCH_E = np.array([122., 164., 203., 243., 300., 352., 375., 1000., 2615.])
XE_ANCH_LAM = np.array([0.30, 0.62, 0.99, 1.42, 2.30, 2.75, 2.9, 6.3, 9.1])
_mu_tot = 1 / XE_ANCH_LAM
_mu_kn = NE['lxe'] * sigma_kn(XE_ANCH_E)
_mu_pe = np.clip(_mu_tot - _mu_kn, 1e-4, None)  # photoelectric (+ pair at 2.6 MeV) as 'absorption'
def mu_pe_xe(E, scale=1.0):
    E = np.asarray(E, float)
    return scale * np.exp(np.interp(np.log(E), np.log(XE_ANCH_E), np.log(_mu_pe)))
# Ti photoelectric: per-atom sigma_pe ~ Z^4.5 (recalled, likely); atoms/cm3 Ti 5.66e22 vs Xe 1.31e22
TI_PE_SCALE = (22 / 54)**4.5 * (4.5 / 47.87) / (RHO_LXE / 131.29)

def mu_material(E, mat, xe_scale=1.0, pmt_scale=1.0):
    """(mu_total, mu_absorption) per cm for material index array."""
    E = np.asarray(E, float)
    mu_c = np.zeros_like(E); mu_a = np.zeros_like(E)
    skn = sigma_kn(np.clip(E, 1, None))
    for code, name in MAT_NAMES.items():
        m = mat == code
        if not m.any():
            continue
        ne = NE[MAT_BASE[name]]
        fac = xe_scale if MAT_BASE[name] == 'lxe' else (pmt_scale if MAT_BASE[name] == 'pmt' else 1.0)
        mu_c[m] = ne * skn[m] * fac
        if MAT_BASE[name] == 'lxe':
            mu_a[m] = mu_pe_xe(E[m], xe_scale)
        elif MAT_BASE[name] == 'ti':
            mu_a[m] = mu_pe_xe(E[m]) * TI_PE_SCALE
    return mu_c + mu_a, mu_a

# material codes: 0 none/gas/vacuum/outside, 1 TPC active LXe, 2 wall dead shell LXe, 3 RFR LXe (+ gap to PMTs),
# 4 Skin LXe, 5 PTFE, 6 Ti, 7 water, 8 GdLS, 9 PMT array
MAT_NAMES = {1: 'tpc', 2: 'shell', 3: 'rfr', 4: 'skin', 5: 'ptfe', 6: 'ti', 7: 'water', 8: 'gdls', 9: 'pmt'}
MAT_BASE = dict(tpc='lxe', shell='lxe', rfr='lxe', skin='lxe', ptfe='ptfe', ti='ti', water='water', gdls='gdls', pmt='pmt')

R_TPC, H_TPC, T_SHELL = 72.8, 145.6, 0.3            # cm (P004/P033 recalled; shell = paper's 'up to 3 mm')
GEO0 = dict(t_ptfe=2.0, t_skin=6.0, od_side=61.0, od_top=45.0, od_bot=45.0, rfr=13.75, gap_pmt=2.25,
            pmt_thick=15.0, skin_bot=25.0, ti=0.8, vac=7.0, water=4.0, gas_top=4.4, r_world=260.0, rho_pmt=RHO_PMT0)

def region(x, y, z, g):
    """Material code at points (vectorised), for geometry dict g. z = 0 at the cathode."""
    r = np.hypot(x, y)
    mat = np.zeros(r.shape, dtype=np.int8)
    r_ptfe = R_TPC + g['t_ptfe']; r_skin = r_ptfe + g['t_skin']
    r_icv = r_skin + g['ti']; r_ocv = r_icv + g['vac']; r_ocv2 = r_ocv + g['ti']
    r_w = r_ocv2 + g['water']; r_od = r_w + g['od_side']
    z_rfr = -g['rfr']; z_gap = z_rfr - g['gap_pmt']; z_pmtb = z_gap - g['pmt_thick']; z_skb = z_pmtb - g['skin_bot']
    z_icvb = z_skb - g['ti']; z_ocvb = z_icvb - g['vac'] - g['ti']; z_wb = z_ocvb - g['water']; z_odb = z_wb - g['od_bot']
    z_liq = H_TPC + 0.5; z_gas = H_TPC + g['gas_top']; z_pmtt = z_gas + g['pmt_thick']
    z_icvt = z_pmtt + 5.0; z_ocvt = z_icvt + g['ti'] + g['vac']; z_wt = z_ocvt + g['ti'] + g['water']; z_odt = z_wt + g['od_top']
    # inner LXe column (r < r_ptfe region handled: TPC / shell / RFR / PMT arrays / bottom skin)
    inner = r < R_TPC
    mat[inner & (z > 0) & (z < H_TPC) & (r < R_TPC - T_SHELL)] = 1
    mat[inner & (z > 0) & (z < H_TPC) & (r >= R_TPC - T_SHELL)] = 2
    mat[inner & (z <= 0) & (z > z_gap)] = 3
    # PMT arrays span the full inner+skin radius
    mat[(r < r_skin) & (z <= z_gap) & (z > z_pmtb)] = 9
    mat[(r < r_skin) & (z <= z_pmtb) & (z > z_skb)] = 4          # bottom Skin dome
    mat[(r < r_skin) & (z >= z_gas) & (z < z_pmtt)] = 9          # top PMT array (gas above liquid is 'none')
    # side: PTFE, Skin (up to liquid level), then Ti / vacuum / Ti / water / GdLS / water
    side = (r >= R_TPC) & (r < r_ptfe) & (z > z_gap) & (z < z_liq)
    mat[side] = 5
    skin = (r >= r_ptfe) & (r < r_skin) & (z > z_skb) & (z < z_liq)
    mat[skin] = 4
    # cryostat + OD shells as coaxial cylinders with flat end caps
    in_icv = (r < r_icv) & (z > z_icvb) & (z < z_icvt + g['ti'])
    in_icv_inner = (r < r_skin) & (z > z_skb) & (z < z_icvt)
    mat[in_icv & ~in_icv_inner & (mat == 0)] = 6
    in_ocv = (r < r_ocv2) & (z > z_ocvb) & (z < z_ocvt + g['ti'])
    in_ocv_inner = (r < r_ocv) & (z > z_ocvb + g['ti']) & (z < z_ocvt)
    mat[in_ocv & ~in_ocv_inner & (mat == 0)] = 6
    in_w1 = (r < r_w) & (z > z_wb) & (z < z_wt)
    mat[in_w1 & ~in_ocv & (mat == 0)] = 7
    in_od = (r < r_od) & (z > z_odb) & (z < z_odt)
    mat[in_od & ~in_w1 & (mat == 0)] = 8
    mat[~in_od & (mat == 0)] = 7                                   # outer water (silent)
    mat[(r > g['r_world']) | (z < z_odb - 40) | (z > z_odt + 40)] = 0
    return mat

def sample_kn_cos(E):
    """Klein-Nishina polar-angle cosines for photon energies E (keV), rejection sampling."""
    E = np.asarray(E, float); k = E / ME
    out = np.empty_like(E); todo = np.ones(E.shape, bool)
    while todo.any():
        n = todo.sum()
        c = rng.uniform(-1, 1, n); kk = k[todo]
        ratio = 1 / (1 + kk * (1 - c))
        f = ratio**2 * (ratio + 1 / ratio - (1 - c**2))
        acc = rng.uniform(0, 2, n) < f
        idx = np.flatnonzero(todo)[acc]
        out[idx] = c[acc]; todo[idx] = False
    return out

def rotate(u, cos_t, phi):
    """Rotate unit vectors u (N,3) by polar angle acos(cos_t) and azimuth phi."""
    sin_t = np.sqrt(np.clip(1 - cos_t**2, 0, 1))
    a = np.where((np.abs(u[:, 2]) < 0.9)[:, None], np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    v = np.cross(u, a); v /= np.linalg.norm(v, axis=1)[:, None]
    w = np.cross(u, v)
    return cos_t[:, None] * u + sin_t[:, None] * (np.cos(phi)[:, None] * v + np.sin(phi)[:, None] * w)

def transport(pos, u, E, g, xe_scale=1.0, ds=0.25, e_cut=5.0, max_steps=3000):
    """Ray-march photons; returns deposits per class and exit route.
    Deposits: dict of arrays (tpc, shell, rfr, skin, gdls, silent). route (by the photon's final position:
    absorption point or escape point): 1 side (z inside the TPC height), 2 top corner (above the liquid), 3 bottom."""
    N = len(E)
    pos = pos.copy(); u = u.copy(); E = E.copy()
    dep = {k: np.zeros(N) for k in ['tpc', 'shell', 'rfr', 'skin', 'gdls', 'silent']}
    alive = np.ones(N, bool)
    pmt_scale = g.get('rho_pmt', RHO_PMT0) / RHO_PMT0
    CLASS = {1: 'tpc', 2: 'shell', 3: 'rfr', 4: 'skin', 8: 'gdls', 5: 'silent', 6: 'silent', 7: 'silent', 9: 'silent'}
    for _ in range(max_steps):
        ia = np.flatnonzero(alive)
        if ia.size == 0:
            break
        p = pos[ia]; uu = u[ia]; Ee = E[ia]
        mat = region(p[:, 0], p[:, 1], p[:, 2], g)
        rr = np.hypot(p[:, 0], p[:, 1])
        # kill escapers (outside world)
        dead = (mat == 0) & ((rr > g['r_world']) | (p[:, 2] < -200) | (p[:, 2] > 400))
        alive[ia[dead]] = False
        mu_t, mu_a = mu_material(Ee, mat, xe_scale, pmt_scale)
        pint = 1 - np.exp(-mu_t * ds)
        hit = rng.uniform(size=ia.size) < pint
        # move everyone one step
        pos[ia] = p + uu * ds
        if hit.any():
            ih = ia[hit]; mh = mat[hit]; Eh = Ee[hit]
            cls = np.array([CLASS.get(int(m), 'silent') for m in mh])
            absorb = rng.uniform(size=ih.size) < (mu_a[hit] / mu_t[hit])
            # photoabsorption
            for c in np.unique(cls[absorb]):
                sel = absorb & (cls == c)
                np.add.at(dep[c], ih[sel], Eh[sel])
            alive[ih[absorb]] = False
            # Compton
            ic = ih[~absorb]
            if ic.size:
                Ec = Eh[~absorb]; cc = cls[~absorb]
                cos_t = sample_kn_cos(Ec)
                Ep = Ec / (1 + Ec / ME * (1 - cos_t))
                T = Ec - Ep
                for c in np.unique(cc):
                    sel = cc == c
                    np.add.at(dep[c], ic[sel], T[sel])
                u[ic] = rotate(u[ic], cos_t, rng.uniform(0, 2 * np.pi, ic.size))
                E[ic] = Ep
                low = Ep < e_cut
                if low.any():
                    for c in np.unique(cc[low]):
                        sel = low & (cc == c)
                        np.add.at(dep[c], ic[sel], Ep[sel])
                    alive[ic[low]] = False
    # route from the final position (absorption or escape point)
    zf = pos[:, 2]
    route = np.where(zf > H_TPC, 2, np.where(zf < -g['rfr'] - g['gap_pmt'], 3, 1)).astype(np.int8)
    return dep, route

def lam_xe(E, xe_scale=1.0):
    mu_t, _ = mu_material(np.atleast_1d(E), np.ones(np.size(E), np.int8), xe_scale)
    return 1 / mu_t

VERTEX = np.array([45.9, 0.0, 26.4])   # paper (Fig. 3 caption)
E_FV, E_WALL = 12.0, 77.0               # paper decomposition (supplement l. 733-734)

def cyl_hit(p, u, R):
    """Forward distance to |p+tu|_xy = R (t > 0)."""
    a = u[:, 0]**2 + u[:, 1]**2
    b = 2 * (p[:, 0] * u[:, 0] + p[:, 1] * u[:, 1])
    c = p[:, 0]**2 + p[:, 1]**2 - R**2
    disc = b**2 - 4 * a * c
    t = (-b + np.sqrt(np.clip(disc, 0, None))) / (2 * a)
    return t

def initial_states(E0, N, ordering, g, xe_scale=1.0):
    """Build the outgoing photon (position, direction, energy) and history weight for one ordering.
    A: 12 keV at the vertex first, then 77 keV Compton in the wall shell, gamma leaves the shell outward.
    B: 77 keV in the wall shell first, then 12 keV at the vertex; gamma continues across the TPC."""
    if ordering == 'A':
        # isotropic lines through the vertex; keep those whose forward ray hits the side wall shell
        cz = rng.uniform(-1, 1, 4 * N); ph = rng.uniform(0, 2 * np.pi, 4 * N)
        s = np.sqrt(1 - cz**2)
        u = np.stack([s * np.cos(ph), s * np.sin(ph), cz], 1)
        p = np.repeat(VERTEX[None, :], 4 * N, 0)
        t_out = cyl_hit(p, u, R_TPC - T_SHELL)
        z_hit = p[:, 2] + t_out * u[:, 2]
        ok = (z_hit > 0) & (z_hit < H_TPC)
        u, p, t_out, z_hit = u[ok][:N], p[ok][:N], t_out[ok][:N], z_hit[ok][:N]
        n = len(u)
        E1 = E0 - E_FV
        lam1 = lam_xe(E1, xe_scale)[0]; lam0 = lam_xe(E0, xe_scale)[0]
        # upstream distance inside the active TPC (backwards along -u)
        t_back_cyl = cyl_hit(p, -u, R_TPC)
        with np.errstate(divide='ignore'):
            t_back_z = np.where(u[:, 2] > 0, p[:, 2] / u[:, 2], (H_TPC - p[:, 2]) / (-u[:, 2]))
        t_in = np.minimum(t_back_cyl, t_back_z)
        entry_bottom = (t_back_z < t_back_cyl) & (u[:, 2] > 0)
        # radial cosine at the wall
        hit = p + t_out[:, None] * u
        rhat = np.stack([hit[:, 0], hit[:, 1], np.zeros(n)], 1) / np.hypot(hit[:, 0], hit[:, 1])[:, None]
        cos_psi = np.clip(np.abs((u * rhat).sum(1)), 0.02, 1)
        p_shell = 1 - np.exp(-T_SHELL / (lam1 * cos_psi))
        w = np.exp(-t_in / lam0 - t_out / lam1) * p_shell
        w *= np.where(entry_bottom, np.exp(-g['rfr'] / lam0), 1.0)   # silent RFR crossing for bottom entries
        # interaction point inside the shell (uniform along the in-shell path, capped at the outer wall)
        depth = rng.uniform(0, 1, n) * np.minimum(T_SHELL / cos_psi, 3.0)
        pos = hit + depth[:, None] * u
        rr = np.hypot(pos[:, 0], pos[:, 1]); over = rr > R_TPC - 1e-3
        pos[over] *= ((R_TPC - 1e-3) / rr[over])[:, None] * np.array([1, 1, 0]) + np.array([0, 0, 1])
        # Compton with fixed 77 keV deposit
        E2 = E1 - E_WALL
        cos_t = 1 - ME * (1 / E2 - 1 / E1)
        u2 = rotate(u, np.full(n, cos_t), rng.uniform(0, 2 * np.pi, n))
        return pos, u2, np.full(n, E2), w, dict(theta_wall_deg=float(np.degrees(np.arccos(cos_t))))
    else:
        # shell points uniform over the side wall; ray to the vertex
        n = N
        zs = rng.uniform(0, H_TPC, n); ph = rng.uniform(0, 2 * np.pi, n)
        rs = R_TPC - T_SHELL / 2
        sp = np.stack([rs * np.cos(ph), rs * np.sin(ph), zs], 1)
        d = VERTEX[None, :] - sp
        dist = np.linalg.norm(d, axis=1)
        u = d / dist[:, None]
        E1 = E0 - E_WALL
        lam1 = lam_xe(E1, xe_scale)[0]
        w = np.exp(-dist / lam1)                        # 1/d^2 flux x d^2 (uniform shell-area sampling) -> survival only
        E2 = E1 - E_FV
        lam2 = lam_xe(E2, xe_scale)[0]
        cos_t = 1 - ME * (1 / E2 - 1 / E1)
        u2 = rotate(u, np.full(n, cos_t), rng.uniform(0, 2 * np.pi, n))
        p = np.repeat(VERTEX[None, :], n, 0)
        # analytic silent crossing of the active TPC: weight by survival, start at the exit point
        t_cyl = cyl_hit(p, u2, R_TPC - T_SHELL)
        with np.errstate(divide='ignore'):
            t_z = np.where(u2[:, 2] > 0, (H_TPC - p[:, 2]) / u2[:, 2], np.where(u2[:, 2] < 0, -p[:, 2] / u2[:, 2], np.inf))
        t_exit = np.minimum(t_cyl, t_z)
        w = w * np.exp(-t_exit / lam2)
        pos = p + (t_exit + 1e-3)[:, None] * u2
        return pos, u2, np.full(n, E2), w, dict(theta_fv_deg=float(np.degrees(np.arccos(cos_t))),
                                                f_exit_top=float(np.average(t_z < t_cyl, weights=w) if w.sum() > 0 else np.nan),
                                                f_exit_top_up=float(np.average((t_z < t_cyl) & (u2[:, 2] > 0), weights=w)))

def classify(dep, route, w, thr_skin, thr_od, allow_extra_dead=False):
    """Weighted outcome fractions conditional on the observed topology surviving: no further deposit in any
    LXe seen by the TPC PMTs (active TPC -> extra S2, fails SS; wall shell or RFR -> extra S1-only light beyond the
    471 phd budget). allow_extra_dead=True keeps histories with extra shell/RFR light (cross-check only)."""
    tpc = dep['tpc'] > 1.0
    rfr = dep['rfr'] > 1.0
    shell = dep['shell'] > 1.0
    keep = ~tpc & (~rfr & ~shell if not allow_extra_dead else np.ones_like(tpc))
    skin_v = keep & (dep['skin'] >= thr_skin)
    od_v = keep & (dep['gdls'] >= thr_od) & ~skin_v
    silent = keep & ~skin_v & ~od_v
    W = w.sum(); Wk = w[keep].sum()
    def frac(m, den):
        return w[m].sum() / den if den > 0 else np.nan
    eps = (w[skin_v].sum() + w[od_v].sum()) / Wk
    # effective-sample-size error on eps
    neff = Wk**2 / (w[keep]**2).sum()
    eps_err = np.sqrt(eps * (1 - eps) / neff)
    out = dict(f_tpc=frac(tpc, W), f_rfr=frac(rfr & ~tpc, W), f_shell=frac(shell & ~tpc & ~rfr, W), f_keep=Wk / W,
               eps=eps, eps_err=eps_err, untagged=1 - eps,
               f_skin_veto=frac(skin_v, Wk), f_od_veto=frac(od_v, Wk),
               silent_side=frac(silent & (route == 1), Wk), silent_top=frac(silent & (route == 2), Wk),
               silent_bottom=frac(silent & (route == 3), Wk),
               # sub-threshold information among silent events
               silent_subthr_skin=frac(silent & (dep['skin'] > 0), w[silent].sum()),
               silent_subthr_od=frac(silent & (dep['gdls'] > 0), w[silent].sum()),
               silent_any_subthr=frac(silent & ((dep['skin'] > 0) | (dep['gdls'] > 0)), w[silent].sum()),
               neff=neff)
    return out

E0_LIST = [609., 1120., 1250., 1461., 1764., 2615.]   # 214Bi, 214Bi, 60Co proxy, 40K, 214Bi, 208Tl (recalled, certain lines)
THR_SKIN0, THR_OD0 = 5.0, 50.0    # keV-equivalent of 2.5 phd (Skin) and 4.5 phd (OD): recalled/uncertain, bracketed
N_MC = 30000

def run_set(g, xe_scale=1.0, thr_skin=THR_SKIN0, thr_od=THR_OD0, N=N_MC, energies=E0_LIST, orderings=('A', 'B'), keep_raw=False):
    res = []; raw = {}
    for E0 in energies:
        for od in orderings:
            pos, u, E, w, extra = initial_states(E0, N, od, g, xe_scale)
            dep, route = transport(pos, u, E, g, xe_scale)
            c = classify(dep, route, w, thr_skin, thr_od)
            c.update(E0=E0, ordering=od, E_out=float(E[0]), **extra)
            res.append(c)
            if keep_raw:
                raw[(E0, od)] = (dep, route, w)
    df = pd.DataFrame(res)
    return (df, raw) if keep_raw else df

def mix(df, col='eps'):
    """Equal-weight mix over energies and orderings of a per-run quantity (eps, etc.)."""
    return float(df[col].mean())

print('\n--- Part 2: topology-specific tagging MC (central geometry) ---')
print('lambda_LXe (cm) at 300/609/1000/1461/2615 keV:', np.round(lam_xe(np.array([300., 609., 1000., 1461., 2615.])), 2))
print('sigma_KN(1 MeV) = %.4f b' % (sigma_kn(1000.) * 1e24))
CEN, RAW = run_set(GEO0, keep_raw=True)
print(CEN[['E0', 'ordering', 'E_out', 'f_tpc', 'f_rfr', 'f_shell', 'eps', 'eps_err', 'f_skin_veto', 'f_od_veto', 'silent_side', 'silent_top',
           'silent_bottom', 'silent_any_subthr']].to_string(float_format=lambda v: '%.4f' % v))
CEN.to_csv(os.path.join(OUT, 'P073_topology_tagging_central.csv'), index=False)
eps_topo = mix(CEN); unt_topo = 1 - eps_topo
eps_A = CEN[CEN.ordering == 'A'].eps.mean(); eps_B = CEN[CEN.ordering == 'B'].eps.mean()
eps_allow = np.mean([classify(*RAW[(E0, od)], THR_SKIN0, THR_OD0, allow_extra_dead=True)['eps'] for E0 in E0_LIST for od in ('A', 'B')])
print('mix eps_topo = %.4f (A %.4f, B %.4f); untagged %.4f; if extra shell/RFR light allowed: %.4f' % (eps_topo, eps_A, eps_B, unt_topo, eps_allow))
print('elapsed %.0f s' % (time.time() - T0))

# validation of the transport against analytic slab attenuation (radial 1372 keV photons)
def _slab_check(E=1372., n=15000):
    out = {}
    pos = np.tile([93.5, 0., 50.], (n, 1)); u = np.tile([1., 0., 0.], (n, 1))
    dep, _ = transport(pos, u, np.full(n, E), GEO0)
    mu = NE['gdls'] * sigma_kn(E)
    out['gdls_lambda_cm'] = 1 / mu; out['gdls_P0_analytic'] = float(np.exp(-61 * mu)); out['gdls_P0_mc_with_backscatter'] = float(np.mean(dep['gdls'] == 0))
    dep, _ = transport(pos, u, np.full(n, E), {**GEO0, 'r_world': 155.0})
    out['gdls_P0_mc_no_backscatter'] = float(np.mean(dep['gdls'] == 0))
    pos = np.tile([74.85, 0., 50.], (n, 1))
    dep, _ = transport(pos, u, np.full(n, E), GEO0)
    lam = lam_xe(E)[0]
    out['skin_lambda_cm'] = float(lam); out['skin_P0_analytic'] = float(np.exp(-6 / lam)); out['skin_P0_mc_with_backscatter'] = float(np.mean(dep['skin'] == 0))
    out['skin0_and_od_below_50keV_radial'] = float(np.mean((dep['skin'] == 0) & (dep['gdls'] < 50)))
    return out
VALID = _slab_check()
print('slab validation:', {k: round(v, 4) for k, v in VALID.items()})

# central value: ordering A (the SS-compatible wall class is dominated by A; B survivors are a rare sub-class that exits
# through the cathode and is mostly excluded by RFR light). The equal A/B mix and B alone enter the range.
unt_central = 1 - eps_A
eps_err_A = float(np.sqrt((CEN[CEN.ordering == 'A'].eps_err**2).sum()) / (CEN.ordering == 'A').sum())

# threshold scan on the stored central histories (no re-transport needed)
thr_rows = []
for ts in [2.0, 5.0, 10.0, 20.0]:
    for to in [20.0, 50.0, 100.0, 200.0]:
        eA = [classify(*RAW[(E0, 'A')], ts, to)['eps'] for E0 in E0_LIST]
        eB = [classify(*RAW[(E0, 'B')], ts, to)['eps'] for E0 in E0_LIST]
        thr_rows.append(dict(thr_skin_keV=ts, thr_od_keV=to, eps_A=np.mean(eA), untagged_A=1 - np.mean(eA),
                             eps_mix=np.mean(eA + eB), untagged_mix=1 - np.mean(eA + eB)))
THR = pd.DataFrame(thr_rows)
THR.to_csv(os.path.join(OUT, 'P073_threshold_scan.csv'), index=False)
print('untagged (ordering A) vs thresholds:')
print(THR.pivot(index='thr_skin_keV', columns='thr_od_keV', values='untagged_A').to_string(float_format=lambda v: '%.4f' % v))

# geometry / attenuation variants (mix over energies; fewer photons)
VARIANTS = {
    'central': (GEO0, 1.0),
    'skin 4 cm': ({**GEO0, 't_skin': 4.0}, 1.0),
    'skin 8 cm': ({**GEO0, 't_skin': 8.0}, 1.0),
    'PTFE 1 cm': ({**GEO0, 't_ptfe': 1.0}, 1.0),
    'PTFE 3 cm': ({**GEO0, 't_ptfe': 3.0}, 1.0),
    'OD top/bottom 30 cm': ({**GEO0, 'od_top': 30.0, 'od_bot': 30.0}, 1.0),
    'OD top/bottom 60 cm': ({**GEO0, 'od_top': 60.0, 'od_bot': 60.0}, 1.0),
    'Xe mu x0.8': (GEO0, 0.8),
    'Xe mu x1.25': (GEO0, 1.25),
    'PMT arrays rho 0.4': ({**GEO0, 'rho_pmt': 0.4}, 1.0),
    'PMT arrays rho 1.2': ({**GEO0, 'rho_pmt': 1.2}, 1.0),
    'no water gap, thin PMT arrays (5 cm)': ({**GEO0, 'water': 0.5, 'pmt_thick': 5.0}, 1.0),
    'top corner weakest: PMT rho 1.2 + OD top 30 cm + skin 4 cm': ({**GEO0, 'rho_pmt': 1.2, 'od_top': 30.0, 'od_bot': 30.0, 't_skin': 4.0}, 1.0),
    'top corner strongest: PMT rho 0.4 + OD top 60 cm + skin 8 cm': ({**GEO0, 'rho_pmt': 0.4, 'od_top': 60.0, 'od_bot': 60.0, 't_skin': 8.0}, 1.0),
}
var_rows = []
for name, (g, xs) in VARIANTS.items():
    if name == 'central':
        df = CEN
    else:
        df = run_set(g, xs, N=10000)
    dA = df[df.ordering == 'A']; dB = df[df.ordering == 'B']
    var_rows.append(dict(variant=name, eps_mix=mix(df), untagged_mix=1 - mix(df),
                         eps_A=dA.eps.mean(), untagged_A=1 - dA.eps.mean(), mc_err_A=float(np.sqrt((dA.eps_err**2).sum()) / len(dA)),
                         eps_B=dB.eps.mean(), untagged_B=1 - dB.eps.mean(), mc_err_B=float(np.sqrt((dB.eps_err**2).sum()) / len(dB)),
                         untagged_side_A=dA.silent_side.mean(), untagged_top_A=dA.silent_top.mean(), untagged_bottom_A=dA.silent_bottom.mean()))
    print('%-58s untagged A %.4f +- %.4f = side %.4f + top %.4f + bottom %.4f | B %.4f +- %.4f | mix %.4f [%.0f s]' % (
        name, var_rows[-1]['untagged_A'], var_rows[-1]['mc_err_A'], var_rows[-1]['untagged_side_A'], var_rows[-1]['untagged_top_A'],
        var_rows[-1]['untagged_bottom_A'], var_rows[-1]['untagged_B'], var_rows[-1]['mc_err_B'], var_rows[-1]['untagged_mix'], time.time() - T0))
VAR = pd.DataFrame(var_rows)
VAR.to_csv(os.path.join(OUT, 'P073_topology_variants.csv'), index=False)
# range: geometry/attenuation variants (A), thresholds (A), ordering B and the equal mix
unt_range = (float(min(VAR.untagged_A.min(), VAR.untagged_B.min(), THR.untagged_A.min(), THR.untagged_mix.min())),
             float(max(VAR.untagged_A.max(), THR.untagged_A.max())))
print('central untagged (A) = %.4f +- %.4f (MC); range %.4f-%.4f; equal A/B mix %.4f' % (unt_central, eps_err_A, *unt_range, unt_topo))

# RFR variant of the event decomposition: 204 keV in the RFR below the cathode, gamma continues downward/outward
def rfr_topology(E0, N, g, xe_scale=1.0):
    """Ordering A analogue: 12 keV at the vertex, then 204 keV Compton at a point in the RFR; transport onward."""
    cz = rng.uniform(-1, 0, 3 * N); ph = rng.uniform(0, 2 * np.pi, 3 * N)
    s = np.sqrt(1 - cz**2)
    u = np.stack([s * np.cos(ph), s * np.sin(ph), cz], 1)
    p = np.repeat(VERTEX[None, :], 3 * N, 0)
    t_cath = -p[:, 2] / u[:, 2]
    hit = p + t_cath[:, None] * u
    ok = np.hypot(hit[:, 0], hit[:, 1]) < R_TPC
    u, p, t_cath, hit = u[ok][:N], p[ok][:N], t_cath[ok][:N], hit[ok][:N]
    n = len(u)
    E1 = E0 - E_FV; lam1 = lam_xe(E1, xe_scale)[0]
    depth = rng.exponential(lam1, n)
    path_rfr = np.minimum(depth, g['rfr'] / np.abs(u[:, 2]) * 0.999)
    pos = hit + path_rfr[:, None] * u
    w = np.exp(-t_cath / lam1) * (1 - np.exp(-g['rfr'] / (np.abs(u[:, 2]) * lam1)))
    E2 = E1 - 204.0
    cos_t = 1 - ME * (1 / E2 - 1 / E1)
    u2 = rotate(u, np.full(n, cos_t), rng.uniform(0, 2 * np.pi, n))
    dep, route = transport(pos, u2, np.full(n, E2), g, xe_scale)
    # the 204 keV deposit is already applied; any further RFR/TPC deposit changes the observed S1/S2 -> excluded
    c = classify(dep, route, w, THR_SKIN0, THR_OD0)
    c['eps_if_extra_RFR_light_allowed'] = classify(dep, route, w, THR_SKIN0, THR_OD0, allow_extra_dead=True)['eps']
    return c
rfr_rows = []
for E0 in [1120., 1461., 2615.]:
    c = rfr_topology(E0, 12000, GEO0); c['E0'] = E0; rfr_rows.append(c)
RFR = pd.DataFrame(rfr_rows)
RFR.to_csv(os.path.join(OUT, 'P073_rfr_topology.csv'), index=False)
print('RFR decomposition (12 keV + 204 keV): eps by E0', RFR[['E0', 'eps', 'eps_err', 'f_rfr', 'f_keep', 'silent_bottom', 'silent_side', 'eps_if_extra_RFR_light_allowed']].round(4).to_dict('records'))

# ----------------------------------------------------------------------------------------------
# 3. Likelihood ratios from the silent vetoes
# ----------------------------------------------------------------------------------------------
P_sci_DM = (1 - F_RAND_PROMPT) * (1 - F_RAND_DELAYED)
def p_science(eps_prompt, eps_delayed=0.0):
    return (1 - eps_prompt) * (1 - F_RAND_PROMPT) * (1 - eps_delayed) * (1 - F_RAND_DELAYED)
HYP = {
    'DM / atmospheric nu / accidental (uncorrelated with vetoes)': p_science(0.0),
    'wall MSSI, paper lambda_MSSI = 0.94': p_science(LAM_MSSI),
    'wall MSSI, MSSI-table WS ROI 4.7 t wall (0.949)': p_science(P1['eps_WSROI_4p7_wall_table']),
    'wall MSSI, event topology 12+77 keV (this work, ordering A, central)': p_science(eps_A),
    'wall MSSI, event topology, equal A/B mix': p_science(eps_topo),
    'wall MSSI, event topology, least-tagged variant': p_science(1 - unt_range[1]),
    'wall MSSI, event topology, most-tagged variant': p_science(1 - unt_range[0]),
    'RFR MSSI, MSSI-table WS ROI 4.7 t (0.999)': p_science(EFF.eps_sim.iloc[1]),
    'RFR MSSI, event topology 12+204 keV (this work)': p_science(RFR.eps.mean()),
    '214Pb RFR MSSI, gamma contained (HE-SB 5.4 t, 0.195)': p_science(EFF.eps_sim.iloc[7]),
    'detector-ER single scatter (lambda_PG = 0.88)': p_science(LAM_PG),
    'neutron single scatter (lambda_PN = 0.05, lambda_DN = 0.87)': (1 - LAM_PN) * (1 - F_RAND_PROMPT) * (1 - LAM_DN) * (1 - F_RAND_DELAYED),
}
LR = pd.DataFrame([dict(hypothesis=k, P_science=v, LR_vs_DM=v / P_sci_DM) for k, v in HYP.items()])
LR.to_csv(os.path.join(OUT, 'P073_veto_silence_LR.csv'), index=False)
print('\n--- Part 3: veto-silence likelihood ratios ---')
print(LR.to_string(float_format=lambda v: '%.4g' % v))
extra_factor_vs_094 = unt_central / (1 - LAM_MSSI)
extra_factor_vs_table = unt_central / P1['untagged_wall_table']
extra_range_vs_table = (unt_range[0] / P1['untagged_wall_table'], unt_range[1] / P1['untagged_wall_table'])
# combination with P004 (k_required 619 for 10 % in the neighbourhood, already including the 5.1 % untagged of 0.0048)
K_P004 = 619.; K_P004_RANGE = (413., 929.); K_P033 = 6.4e4
k_req_new = K_P004 / extra_factor_vs_table
k_req_new_range = (K_P004_RANGE[0] / extra_range_vs_table[1], K_P004_RANGE[1] / extra_range_vs_table[0])
k_req_P033_new = K_P033 / extra_factor_vs_table
print('extra factor on wall-MSSI science expectation for the event topology: %.3f (vs 0.94: %.3f); range %.3f-%.3f' %
      (extra_factor_vs_table, extra_factor_vs_094, *extra_range_vs_table))
print('P004 k_required 619 -> %.0f (%.0f-%.0f); P033 6.4e4 -> %.2e' % (k_req_new, *k_req_new_range, k_req_P033_new))

# ----------------------------------------------------------------------------------------------
# 4. Delayed-sample excess: 55 observed vs 50.4 +- 2.0 fitted
# ----------------------------------------------------------------------------------------------
naive_z = (N_DELAYED - FIT_DELAYED) / FIT_DELAYED_ERR
p_pois = stats.poisson.sf(N_DELAYED - 1, FIT_DELAYED)              # P(N >= 55 | mu = 50.4)
# marginalise over the fitted-mean uncertainty (Gaussian)
mus = np.linspace(FIT_DELAYED - 5 * FIT_DELAYED_ERR, FIT_DELAYED + 5 * FIT_DELAYED_ERR, 2001)
wts = stats.norm.pdf(mus, FIT_DELAYED, FIT_DELAYED_ERR); wts /= wts.sum()
p_marg = float((stats.poisson.sf(N_DELAYED - 1, mus) * wts).sum())
z_marg = stats.norm.isf(p_marg)
z_gauss_comb = (N_DELAYED - FIT_DELAYED) / np.sqrt(FIT_DELAYED + FIT_DELAYED_ERR**2)
# expectation from random coincidences of the science-sample population alone
n_delayed_from_random = N_SCI * F_RAND_DELAYED / (1 - F_RAND_DELAYED - F_RAND_PROMPT)
sum_S2 = sum(DELAYED_EXPECTED.values())
# inelastic chi2-photon population (P042): 0.0094 (delta = 300 keV, tau = 66 us) to 0.70 (delta = 350 keV, 1.4 us) delayed events per clean event
p042_delayed_per_clean = (0.0094, 0.70)
n_clean_needed = [(N_DELAYED - FIT_DELAYED) / x for x in p042_delayed_per_clean]
# neutron reading of the excess: lambda_DN = 0.87 -> science-sample neutrons if the entire excess were neutrons
n_neut_total = (N_DELAYED - FIT_DELAYED) / LAM_DN
n_neut_science = n_neut_total * (1 - LAM_DN) * (1 - LAM_PN)
# implied veto-pulse rates from the random-coincidence fractions
rate_prompt_hz = F_RAND_PROMPT / ((2 * PROMPT_SKIN_WIN_US + 2 * PROMPT_OD_WIN_US) * 1e-6)   # if windows add (Skin+OD)
rate_prompt_hz_single = F_RAND_PROMPT / (2 * PROMPT_OD_WIN_US * 1e-6)
rate_delayed_hz = F_RAND_DELAYED / (DELAYED_WIN_US * 1e-6)
P4 = dict(naive_z=naive_z, p_poisson_fixed_mu=p_pois, p_marginalised=p_marg, z_marginalised=z_marg, z_gauss_combined=z_gauss_comb,
          n_delayed_from_random_of_science=n_delayed_from_random, sum_table_S2=sum_S2, excess=N_DELAYED - FIT_DELAYED,
          p042_delayed_per_clean=p042_delayed_per_clean, n_clean_events_needed_for_excess=n_clean_needed,
          neutron_total_if_excess_is_neutrons=n_neut_total, neutron_science_if_excess_is_neutrons=n_neut_science,
          implied_prompt_veto_rate_hz=(rate_prompt_hz, rate_prompt_hz_single), implied_delayed_veto_rate_hz=rate_delayed_hz,
          P_DM_event_randomly_vetoed=F_RAND_PROMPT + F_RAND_DELAYED, signal_in_prompt_TableS1=1.0e-4, signal_in_delayed_TableS2=3.1e-2,
          expected_signal_in_delayed_from_fraction=1.0 * F_RAND_DELAYED + 1.0 * 0.0)  # the 3.1e-2 includes the 1.0 best fit x 0.0286
print('\n--- Part 4/5: delayed excess and signal side ---')
print({k: (np.round(v, 4) if isinstance(v, (float, np.floating)) else v) for k, v in P4.items()})

# ----------------------------------------------------------------------------------------------
# 5. Recommendation numbers: sub-threshold Skin/OD search around the event
# ----------------------------------------------------------------------------------------------
sub_skin = mix(CEN, 'silent_subthr_skin'); sub_od = mix(CEN, 'silent_subthr_od'); sub_any = mix(CEN, 'silent_any_subthr')
# random probability of a sub-threshold pulse in +-1 us if lowering thresholds raises the pulse rate by x5 / x20
for mult in (5, 20):
    pass
P_rand_sub = {m: rate_prompt_hz * m * 2e-6 for m in (1, 5, 20)}
LR_subthr_found = {m: sub_any / P_rand_sub[m] for m in (5, 20)}
REC = dict(silent_with_subthreshold_skin=sub_skin, silent_with_subthreshold_od=sub_od, silent_with_any_subthreshold=sub_any,
           P_random_subthreshold_pm1us=P_rand_sub, LR_wallMSSI_vs_DM_if_subthreshold_pulse_found=LR_subthr_found)
print('recommendation numbers:', {k: (np.round(v, 4) if isinstance(v, float) else v) for k, v in REC.items()})

# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
C_SIM, C_DAT, C_TOP, C_REF = '#4C72B0', '#DD8452', '#55A868', '#8172B3'

fig, ax = plt.subplots(figsize=(8.4, 4.2))
labels = [f"{r.region}\n{r.type} {r.vol_t} t" for _, r in EFF.iterrows()]
x = np.arange(len(EFF))
ax.bar(x, EFF.eps_sim, color=C_SIM, width=0.6, label='simulated (paper MSSI table): $N_p/(N_p+N_s)$')
m = ~EFF.eps_data.isna()
ax.errorbar(x[m], EFF.eps_data[m], yerr=[EFF.eps_data[m] - EFF.eps_data_68lo[m], EFF.eps_data_68hi[m] - EFF.eps_data[m]],
            fmt='o', color=C_DAT, capsize=3, label='observed counts (68% Clopper-Pearson)')
xt = len(EFF) + np.arange(4)
vals = [eps_detER, eps_tables_S1, eps_A, RFR.eps.mean()]
cols = [C_REF, C_REF, C_TOP, C_TOP]
ax.bar(xt, vals, color=cols, width=0.6)
ax.errorbar([xt[2]], [eps_A], yerr=[[eps_A - (1 - unt_range[1])], [(1 - unt_range[0]) - eps_A]], fmt='none', color='k', capsize=3)
ax.axhline(LAM_MSSI, color='k', ls='--', lw=1); ax.text(len(EFF) + 3.6, LAM_MSSI - 0.045, r'$\lambda_{\rm MSSI}=0.94$', ha='right')
ax.set_xticks(list(x) + list(xt))
ax.set_xticklabels(labels + ['det. ERs\nTables I/S1', 'MSSI\nTables I/S1', 'event topo.\nwall 12+77', 'event topo.\nRFR 12+204'],
                   fontsize=7.5)
ax.set_ylabel('prompt-veto tagging efficiency $\\varepsilon$'); ax.set_ylim(0, 1.05)
ax.legend(loc='lower left', fontsize=8, frameon=False)
ax.set_title('MSSI prompt-veto tagging efficiency: paper table, data check, and the event topology', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P073_fig1_tagging_efficiency.png'), dpi=160); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6))
ax = axs[0]
for od, mk in (('A', 'o'), ('B', 's')):
    d = CEN[CEN.ordering == od]
    ax.errorbar(d.E0, 100 * d.untagged, yerr=100 * d.eps_err, fmt=mk + '-', label=f'ordering {od}', color=C_TOP if od == 'A' else C_SIM)
ax.axhline(100 * (1 - LAM_MSSI), color='k', ls='--', lw=1, label='paper: 6% (all WS-ROI MSSI)')
ax.axhline(100 * P1['untagged_wall_table'], color=C_REF, ls=':', lw=1.2, label='MSSI table: 5.1% (WS-ROI wall)')
ax.axhspan(100 * unt_range[0], 100 * unt_range[1], color=C_TOP, alpha=0.15, label='range (variants, thresholds, ordering)')
ax.axhline(100 * unt_central, color=C_TOP, lw=1.5, label='central (ordering A mix): %.1f%%' % (100 * unt_central))
ax.set_xlabel(r'initial $\gamma$ energy $E_0$ (keV)'); ax.set_ylabel('untagged fraction $1-\\varepsilon$ (%)')
ax.set_ylim(0, 8); ax.legend(fontsize=7, frameon=False); ax.set_title('Untagged fraction, event topology', fontsize=9.5)
ax = axs[1]
lr = LR.iloc[[1, 2, 3, 8, 9, 10, 11]]
names = ['wall MSSI\n$\\lambda$=0.94', 'wall MSSI\ntable 0.949', 'wall MSSI\nevent topo.', 'RFR MSSI\nevent topo.',
         '$^{214}$Pb RFR\ncontained', 'detector ER\nSS', 'neutron SS']
ax.barh(np.arange(len(lr)), lr.LR_vs_DM, color=[C_REF, C_REF, C_TOP, C_TOP, C_DAT, C_DAT, C_DAT])
ax.set_xscale('log'); ax.set_yticks(np.arange(len(lr))); ax.set_yticklabels(names, fontsize=7.5)
ax.set_xlabel('P(science sample | H) / P(science sample | DM)'); ax.invert_yaxis()
ax.set_title('Likelihood ratio from the silent vetoes', fontsize=9.5)
for i, v in enumerate(lr.LR_vs_DM):
    ax.text(v * 1.15, i, '%.3g' % v, va='center', fontsize=7.5)
ax.set_xlim(3e-3, 3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P073_fig2_untagged_and_LR.png'), dpi=160); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# Summary JSON
# ----------------------------------------------------------------------------------------------
def _j(o):
    if isinstance(o, (np.floating, float)): return float(o)
    if isinstance(o, (np.integer, int)): return int(o)
    if isinstance(o, dict): return {str(k): _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_j(v) for v in o]
    return o
SUMMARY = dict(
    part1_paper_efficiencies=P1,
    part2_topology=dict(untagged_central_A=unt_central, mc_err_A=eps_err_A, eps_mix=eps_topo, untagged_mix=unt_topo, eps_A=eps_A, eps_B=eps_B,
                        untagged_range=unt_range,
                        eps_if_extra_dead_light_allowed=eps_allow, slab_validation=VALID,
                        variants=VAR.to_dict('records'), threshold_scan=THR.to_dict('records'),
                        per_energy=CEN[['E0', 'ordering', 'eps', 'eps_err', 'f_tpc', 'silent_side', 'silent_top', 'silent_bottom']].to_dict('records'),
                        rfr_topology_eps=RFR.eps.mean(), rfr_per_energy=RFR[['E0', 'eps', 'silent_bottom', 'silent_side']].to_dict('records'),
                        lambda_LXe_cm={str(e): float(lam_xe(e)[0]) for e in [300., 609., 1000., 1461., 2615.]},
                        theta_wall_deg={str(E0): float(CEN[(CEN.E0 == E0) & (CEN.ordering == 'A')].theta_wall_deg.iloc[0]) for E0 in E0_LIST},
                        skin_thr_keV=THR_SKIN0, od_thr_keV=THR_OD0, N_per_run=N_MC),
    part3_LR=dict(P_science_DM=P_sci_DM, LR_table=LR.to_dict('records'), extra_factor_vs_table=extra_factor_vs_table,
                  extra_factor_vs_094=extra_factor_vs_094, extra_factor_range=extra_range_vs_table,
                  k_required_P004_updated=(k_req_new, k_req_new_range), k_required_P033_updated=k_req_P033_new),
    part4_delayed=P4, part5_recommendation=REC, runtime_s=time.time() - T0)
with open(os.path.join(OUT, 'P073_summary.json'), 'w') as f:
    json.dump(_j(SUMMARY), f, indent=1)
print('\nwrote', os.path.join(OUT, 'P073_summary.json'), ' total %.0f s' % (time.time() - T0))
