"""
P078 -- Isospin violation and the isovector suppression factor for xenon at 250 keV: per-isotope amplitudes,
        the f_n/f_p scan and what it does to the high-energy/low-energy ratio.

Run from the simulation root:   .venv/bin/python output/code/P078_isospin_violation.py  [--recompute]

Method (independent of P032, which used three full natural-Xe WimPyDD spectra per mass and a quadratic form).
  For the velocity-independent operators O1 and O4 WimPyDD's rate factorises exactly per isotope (P046/P062):
      dR/dE(E; r, m, delta, halo) = (1000 GeV/m) * sum_i [c0^2 K_i^00(E) + 2 c0 c1 K_i^01(E) + c1^2 K_i^11(E)] * eta(v_min,i(E; m, delta)),
  with c0 = c_p (1 + r), c1 = c_p (1 - r), r = f_n/f_p = c_n/c_p (WimPyDD c^0 = c_p + c_n, c^1 = c_p - c_n; P003 convention),
  and K_i^{tt'}(E) obtained once per isotope from a single 3000 km/s stream with delta_eta = 1 (so eta = 1).  We verify
  numerically that K_i^{tt'}(E) = const_i * W_i^{tt'}(q(E)) (WimPyDD's shell-model response function), so the kernels ARE
  the response functions and everything below is a statement about DMFormFactor-v6 one-body density matrices.
  Halos (Baxter-2021 SHM via lzcommon): WimPyDD's single-day SUN-FRAME halo (v_E = 250.6 km/s; what P003/P032 used and
  called "annual average") and a 12-day ANNUAL MEAN.  Isotope re-weighting of the kernels gives enriched/depleted targets.
Parts
  1. M-response per isotope: W_pp, W_nn, W_pn; signed F_p, F_n (dyad isotopes) and the amplitude Z f_p + r N f_n;
     nodes vs E_R and vs r; natural-Xe minima; comparison with P032 and with the Helm form factor.
  2. Kernels for O1 and O4 (M, Sigma', Sigma'') + validation against direct WimPyDD calls.
  3. Scan r in [-2, 2]: R_hi/lo = R(200-270)/R(5.4-55), N_lo, N_mid = R(55-200)/R(200-270), rho's, percentile of
     248 keV, for elastic and delta = 250, 300, 350, 380 keV at 1 TeV (400/4000 GeV for elastic), both halos.
  4. The xenophobic point r = -0.7 (and the N_lo minimum): per-isotope interference at q -> 0 and at 248 keV.
  5. Enriched / depleted targets at 248 keV and in the windows; N_3sigma for an enriched-vs-natural rate ratio test.
  6. O4 (spin-dependent) with proton/neutron spin structure: S_p/S_n per isotope, r scan, isoscalar vs isovector shapes.
Outputs -> output/work/P078/*.csv|json|npz, figures -> output/work/P078/figures/, run_log.txt
"""
import sys, os, math, json, time, csv
import numpy as np
from scipy import special, integrate, optimize
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'output/work/P078'; FIG = OUT + '/figures'
os.makedirs(FIG, exist_ok=True)
T0 = time.time()
LOG = open(f'{OUT}/run_log.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()
RECOMPUTE = '--recompute' in sys.argv

WD = lz.wd(); Xe = WD.Xe; NC = WD.nuclear_current
Z_XE = 54
M_K = 1000.0                      # kernel reference mass (kernels scale exactly as 1/m; verified below)
ISO_IDX = [i for i in range(Xe.n_isotopes) if Xe.func_w[i](0.05)[NC['M'], 0, 0] > 0]     # 7 active isotopes
A_ISO = [int(Xe.a[i]) for i in ISO_IDX]; N_ISO = [a - Z_XE for a in A_ISO]
AB = np.array([Xe.abundance[i] for i in ISO_IDX]); MN_I = np.array([Xe.mass[i] for i in ISO_IDX], float)
AB_N = AB / AB.sum()
say(f'active isotopes A={A_ISO}, abundances={np.round(AB,5).tolist()} (sum {AB.sum():.4f}); masses {np.round(MN_I,2).tolist()} GeV')
E_EVENT = 248.0

# efficiency model shared with P003/P016/P032 (0.96 plateau; erf edges at 5.4 keV, sigma 3.4; 269.9 keV, sigma 8)
def efficiency(E):
    E = np.asarray(E, float)
    return 0.96 * 0.5 * (1 + special.erf((E - 5.4) / (math.sqrt(2) * 3.4))) * 0.5 * (1 - special.erf((E - 269.9) / (math.sqrt(2) * 8.0)))
E_GRID = np.arange(0.5, 400.0 + 0.25, 0.5)          # true recoil energy, 0.5 keV steps
EFF = efficiency(E_GRID)

def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)

# ==================================================================================================
# Part 1: M response functions per isotope
# ==================================================================================================
def q_of(E, mN):
    return np.sqrt(2.0 * mN * np.asarray(E, float) * 1e-6)      # GeV

def W_cur(i, cur, E):
    mN = Xe.mass[i]; fw = Xe.func_w[i]
    out = np.zeros((len(E), 2, 2))
    for k, qq in enumerate(q_of(E, mN)):
        out[k] = fw(float(qq))[NC[cur]]
    return out          # [E, tau, tau']

E_RESP = np.arange(0.25, 420.0 + 0.1, 0.25)
RESP = {}
for j, i in enumerate(ISO_IDX):
    W = W_cur(i, 'M', E_RESP)
    W00, W01, W11 = W[:, 0, 0], W[:, 0, 1], W[:, 1, 1]
    Wpp = 0.25 * (W00 + 2 * W01 + W11); Wnn = 0.25 * (W00 - 2 * W01 + W11); Wpn = 0.25 * (W00 - W11)
    RESP[A_ISO[j]] = dict(W00=W00, W01=W01, W11=W11, Wpp=Wpp, Wnn=Wnn, Wpn=Wpn, mN=Xe.mass[i], ab=AB[j], N=N_ISO[j], idx=i)

def local_minima(E, y, lo, hi, depth=0.6, half=15.0):
    out = []
    for k in range(1, len(E) - 1):
        if lo < E[k] < hi and y[k] < y[k - 1] and y[k] <= y[k + 1]:
            yl = np.interp(E[k] - half, E, y); yr = np.interp(E[k] + half, E, y)
            if y[k] < depth * yl and y[k] < depth * yr:
                y0, y1, y2 = y[k - 1], y[k], y[k + 1]; d = y0 - 2 * y1 + y2
                out.append(float(E[k] + (0.5 * (y0 - y2) / d * (E[k] - E[k - 1]) if d > 0 else 0.0)))
    return out

def zeros_of(E, y, lo=0.0, hi=1e9):
    out = []
    for k in range(len(E) - 1):
        if lo < E[k] < hi and y[k] * y[k + 1] < 0:
            out.append(float(E[k] - y[k] * (E[k + 1] - E[k]) / (y[k + 1] - y[k])))
    return out

# signed proton / neutron form factors for the dyad isotopes (J = 0, 1/2):  W^{tt'} = F_t F_t'.
# Independent sign construction (differs from P032): F_p = s_p sqrt(W_pp) with s_p flipping at every deep minimum of
# W_pp (W_pp < 1e-3 W_pp(0) and both neighbours 5 keV away larger), F_n = W_pn / F_p (exact for a dyad) with the
# near-zero region of F_p bridged by sign continuity of F_n through sqrt(W_nn).
q0_rows = []
for A, d in RESP.items():
    Wpp, Wnn, Wpn = d['Wpp'], d['Wnn'], d['Wpn']
    # dyad test: pointwise W_pn^2/(W_pp W_nn) = 1 for a single-multipole (J = 0, 1/2) response; evaluated away from the
    # zeros (10-90 and 120-230 keV), where the tabulated responses are far above their ~1e-6 noise floor
    zpn = zeros_of(E_RESP, Wpn, 5, 420)
    away = (E_RESP > 5) & (E_RESP < 200)          # above ~200 keV the tabulated responses approach their ~1e-6 noise floor
    for zc in zpn: away &= np.abs(E_RESP - zc) > 8.0
    dyad = float(np.max(np.abs(Wpn[away] ** 2 / (Wpp[away] * Wnn[away]) - 1.0)))
    d['dyad_dev'] = dyad; d['away'] = away
    # a single M_0 multipole (hence an exact dyad) requires J <= 1/2 (M_J with even J <= 2 J_nuc); 131Xe (J = 3/2) also has M_2
    d['is_dyad'] = bool(Xe.spin[d['idx']] <= 0.5)
    ratioNZ = math.sqrt(Wnn[0] / Wpp[0]); ratio_pn = Wpn[0] / Wpp[0]
    q0_rows.append(dict(A=A, N=d['N'], ab=d['ab'], Wpp0=Wpp[0], Wnn0=Wnn[0], Wpn0=Wpn[0], sqrt_Wnn_over_Wpp_q0=ratioNZ,
                        N_over_Z=d['N'] / Z_XE, Wpn_over_Wpp_q0=ratio_pn, W00_over_A2=d['W00'][0] / (A ** 2),
                        max_dyad_deviation_below300=dyad, is_dyad=d['is_dyad']))
    if d['is_dyad']:
        # W_pn = F_p F_n changes sign exactly where F_p or F_n vanishes; the vanishing factor is the one whose
        # normalised square W_tt/W_tt(0) is the smaller at the crossing (both are ~1e-6 of W(0) there: true zeros).
        mins_p, mins_n = [], []
        for zc in zeros_of(E_RESP, Wpn, 30, 420):
            (mins_p if np.interp(zc, E_RESP, Wpp) / Wpp[0] < np.interp(zc, E_RESP, Wnn) / Wnn[0] else mins_n).append(zc)
        sp = np.ones_like(E_RESP)
        for z in mins_p: sp[E_RESP > z] *= -1
        Fp = sp * np.sqrt(np.abs(Wpp))
        sn = np.ones_like(E_RESP)
        for z in mins_n: sn[E_RESP > z] *= -1
        Fn = sn * np.sqrt(np.abs(Wnn))
        # local sign consistency: |F_p F_n - W_pn| relative to the local scale sqrt(W_pp W_nn), 5-350 keV
        # (floor 1e-4 sqrt(W_pp(0) W_nn(0)) keeps the ~1e-6 tabulation noise at the zeros from dominating the ratio)
        okE = d['away']
        cons = float(np.max(np.abs(Fp * Fn - Wpn)[okE] / np.maximum(np.sqrt(np.abs(Wpp * Wnn))[okE], 1e-4 * math.sqrt(Wpp[0] * Wnn[0]))))
        d.update(Fp=Fp / Fp[0] * Z_XE, Fn=Fn / Fn[0] * d['N'], zeros_p=mins_p, zeros_n=mins_n, sign_consistency=cons)
        # normalised f_p, f_n (f(0) = 1) and their ratio
        d['fp'] = Fp / Fp[0]; d['fn'] = Fn / Fn[0]
    else:
        d.update(Fp=None, Fn=None, zeros_p=local_minima(E_RESP, Wpp, 30, 420), zeros_n=local_minima(E_RESP, Wnn, 30, 420), sign_consistency=float('nan'))
write_csv(f'{OUT}/response_q0_checks.csv', q0_rows)
say('q->0 checks (sqrt(W_nn/W_pp) should equal N/Z if M counts nucleons):', [(r['A'], round(r['sqrt_Wnn_over_Wpp_q0'], 4), round(r['N_over_Z'], 4), round(r['W00_over_A2'], 4), round(r['max_dyad_deviation_below300'], 4)) for r in q0_rows])
say('pointwise dyad deviation max|W_pn^2/(W_pp W_nn) - 1| (5-200 keV, >8 keV from zeros):', {A: f"{d['dyad_dev']:.1e}" for A, d in RESP.items()})
say('sign-consistency |F_p F_n - W_pn| / sqrt(W_pp W_nn) (dyads, same range):', {A: f"{d['sign_consistency']:.1e}" for A, d in RESP.items() if d['is_dyad']})

def F2_iso(A, r):
    """quadratic form for coupling ratio r with c_p = 1: (1+r)^2 W00 + 2(1+r)(1-r) W01 + (1-r)^2 W11  (= 4 |Z f_p + r N f_n|^2 x norm)"""
    d = RESP[A]; c0, c1 = 1 + r, 1 - r
    return c0 ** 2 * d['W00'] + 2 * c0 * c1 * d['W01'] + c1 ** 2 * d['W11']

def amp_iso(A, r):
    """signed amplitude F_p + r F_n (F_p(0) = Z, F_n(0) = N) for dyad isotopes"""
    d = RESP[A]
    return d['Fp'] + r * d['Fn']

R_KEY = {'isoscalar r=+1': 1.0, 'proton-only r=0': 0.0, 'xenophobic r=-0.7': -0.7, 'isovector r=-1': -1.0, 'r=-2': -2.0, 'r=+2': 2.0}
iso_rows = []
for A, d in RESP.items():
    row = dict(A=A, N=d['N'], ab=d['ab'], is_dyad=d['is_dyad'], zeros_Fp=[round(x, 1) for x in d['zeros_p']], zeros_Fn=[round(x, 1) for x in d['zeros_n']])
    for lab, r in R_KEY.items():
        y = F2_iso(A, r); mins = local_minima(E_RESP, y, 40, 420)
        row[f'minima_{lab}'] = [round(x, 1) for x in mins]
        row[f'depth_{lab}'] = [float(f"{np.interp(x, E_RESP, y) / y[0]:.1e}") for x in mins]
        if d['is_dyad']:
            row[f'zeros_amp_{lab}'] = [round(x, 1) for x in zeros_of(E_RESP, amp_iso(A, r), 40, 420)]
        row[f'F2_248_over_F2iso0_{lab}'] = float(np.interp(E_EVENT, E_RESP, y) / F2_iso(A, 1.0)[0])
    if d['is_dyad']:
        for E0 in (20.0, 100.0, 200.0, 248.0):
            row[f'fn_over_fp_{int(E0)}'] = float(np.interp(E0, E_RESP, d['fn']) / np.interp(E0, E_RESP, d['fp']))
            row[f'fp_{int(E0)}'] = float(np.interp(E0, E_RESP, d['fp'])); row[f'fn_{int(E0)}'] = float(np.interp(E0, E_RESP, d['fn']))
    iso_rows.append(row)
write_csv(f'{OUT}/isotope_nodes.csv', iso_rows)
for row in iso_rows:
    say(f"  {row['A']}Xe  zeros F_p {row['zeros_Fp']}  F_n {row['zeros_Fn']}  | minima iso {row['minima_isoscalar r=+1']}  vec {row['minima_isovector r=-1']}  "
        f"p-only {row['minima_proton-only r=0']}  xeno {row['minima_xenophobic r=-0.7']} (depth {row['depth_xenophobic r=-0.7']})")

# node position of each isotope's amplitude vs r (second node, > 150 keV, and first node)
R_FINE = np.round(np.arange(-2.0, 2.0001, 0.01), 3)
node_vs_r = []
for r in R_FINE:
    row = dict(r=r)
    for A in RESP:
        mins = local_minima(E_RESP, F2_iso(A, r), 40, 420)
        n1 = [x for x in mins if x < 160]; n2 = [x for x in mins if x >= 160]
        row[f'node1_{A}'] = n1[0] if n1 else float('nan'); row[f'node2_{A}'] = n2[0] if n2 else float('nan')
    # natural xenon response-only minimum (abundance / m_N weighted, as in the rate kernel)
    ynat = sum(RESP[A]['ab'] / RESP[A]['mN'] * F2_iso(A, r) for A in RESP)
    mins = local_minima(E_RESP, ynat, 40, 420); n1 = [x for x in mins if x < 160]; n2 = [x for x in mins if x >= 160]
    row['node1_nat'] = n1[0] if n1 else float('nan'); row['node2_nat'] = n2[0] if n2 else float('nan')
    node_vs_r.append(row)
write_csv(f'{OUT}/node_vs_r.csv', node_vs_r)
def node_at(r, key):
    return next(x[key] for x in node_vs_r if abs(x['r'] - r) < 1e-6)
say('natural-Xe response minima (keV): iso', node_at(1.0, 'node1_nat'), node_at(1.0, 'node2_nat'), ' vec', node_at(-1.0, 'node1_nat'), node_at(-1.0, 'node2_nat'),
    ' p-only', node_at(0.0, 'node1_nat'), node_at(0.0, 'node2_nat'), ' xeno', node_at(-0.7, 'node1_nat'), node_at(-0.7, 'node2_nat'))
# r for which the natural-Xe second node sits exactly on 248 keV
n2 = np.array([x['node2_nat'] for x in node_vs_r]); rr = np.array([x['r'] for x in node_vs_r])
cross = [float(rr[k] + (248 - n2[k]) * (rr[k + 1] - rr[k]) / (n2[k + 1] - n2[k])) for k in range(len(rr) - 1) if np.isfinite(n2[k]) and np.isfinite(n2[k + 1]) and (n2[k] - 248) * (n2[k + 1] - 248) < 0]
say('r values at which the natural-Xe second minimum crosses 248 keV:', [round(c, 3) for c in cross])

# Helm comparison (lzcommon helm_F2): zeros for A = 131
helm = np.array([lz.helm_F2(e, 131) for e in E_RESP])
helm_min = local_minima(E_RESP, helm, 40, 420)
say('Helm F^2 (A=131) minima:', [round(x, 1) for x in helm_min], ' vs shell-model isoscalar natural-Xe', node_at(1.0, 'node1_nat'), node_at(1.0, 'node2_nat'))

# ==================================================================================================
# Part 2: kernels
# ==================================================================================================
STREAM = (np.array([3000.0]), np.array([1.0]))
HAMS = {'00': lz.wd_hamiltonian('P078_c0', {1: (1.0, 0.0)}), '11': lz.wd_hamiltonian('P078_c1', {1: (0.0, 1.0)}), 'pp': lz.wd_hamiltonian('P078_c0c1', {1: (1.0, 1.0)})}
HAMS4 = {'00': lz.wd_hamiltonian('P078_O4c0', {4: (1.0, 0.0)}), '11': lz.wd_hamiltonian('P078_O4c1', {4: (0.0, 1.0)}), 'pp': lz.wd_hamiltonian('P078_O4c0c1', {4: (1.0, 1.0)})}
CACHE = f'{OUT}/kernels_O1_O4_1000gev.npz'
if os.path.exists(CACHE) and not RECOMPUTE:
    z = np.load(CACHE); KER = {k: z[k] for k in z.files if k != 'E_GRID'}
    assert np.array_equal(z['E_GRID'], E_GRID); say('kernels loaded from cache')
else:
    KER = {}
    for op, H in (('O1', HAMS), ('O4', HAMS4)):
        for j, i in enumerate(ISO_IDX):
            if op == 'O4' and A_ISO[j] not in (129, 131):
                for key in ('00', '11', '01'): KER[f'{op}_{key}_{A_ISO[j]}'] = np.zeros_like(E_GRID)
                continue
            k00 = np.atleast_1d(lz.wd_rate(H['00'], M_K, E_GRID, halo=STREAM, isotopes_list={0: [i]}))
            k11 = np.atleast_1d(lz.wd_rate(H['11'], M_K, E_GRID, halo=STREAM, isotopes_list={0: [i]}))
            kpp = np.atleast_1d(lz.wd_rate(H['pp'], M_K, E_GRID, halo=STREAM, isotopes_list={0: [i]}))
            KER[f'{op}_00_{A_ISO[j]}'] = k00; KER[f'{op}_11_{A_ISO[j]}'] = k11; KER[f'{op}_01_{A_ISO[j]}'] = 0.5 * (kpp - k00 - k11)
            say(f'  kernel {op} {A_ISO[j]}Xe done ({time.time()-T0:.0f} s)')
    np.savez(CACHE, E_GRID=E_GRID, **KER)
# kernel = const * W(q): check
for A in (129, 131, 136):
    d = RESP[A]
    for key, wk in (('00', 'W00'), ('11', 'W11'), ('01', 'W01')):
        K = KER[f'O1_{key}_{A}']; Wq = np.interp(E_GRID, E_RESP, d[wk])
        ok = np.abs(Wq) > 1e-3 * np.abs(Wq).max()
        ratio = K[ok] / Wq[ok]
        say(f'  K^{key}/W^{key} for {A}Xe: mean {ratio.mean():.6e}, rel. spread {ratio.std()/abs(ratio.mean()):.1e}  (const_i should be prop. to ab_i/m_i: {d["ab"]/d["mN"]:.5e})')

# halos
class Halo:
    def __init__(self, vmin, deta, label):
        self.vmin = np.asarray(vmin); self.deta = np.asarray(deta); self.label = label
        self.cum = np.cumsum(self.deta[::-1])[::-1]
        nz = np.nonzero(self.deta > 0)[0]; self.vmax = float(self.vmin[nz.max()]) if nz.size else 0.0
    def eta(self, v):
        idx = np.searchsorted(self.vmin, v, side='right'); out = np.zeros(np.shape(v)); ok = idx < len(self.cum); out[ok] = self.cum[idx[ok]]; return out
VGRID = np.linspace(0.0, lz.VESC_KMS + 300.0, 1200)
HALOS = {'sunframe': Halo(VGRID, lz.wd_halo(vmin=VGRID)[1], 'Sun-frame (v_E = 250.6 km/s)'),
         'annual': Halo(VGRID, np.mean([lz.wd_halo(day_of_year=d, vmin=VGRID)[1] for d in 15.0 + 365.25 / 12 * np.arange(12)], axis=0), '12-day annual mean')}
say('halo v_max: sunframe %.1f, annual %.1f km/s' % (HALOS['sunframe'].vmax, HALOS['annual'].vmax))

def vmin_iso(E, mN, m, delta):
    mu = m * mN / (m + mN)
    return 1.0 / np.sqrt(2.0 * mN * E) * np.abs(mN * E / mu + delta) * 300.0          # WimPyDD get_vmin (E, delta in keV; masses GeV)

def spectrum(r, m, delta, halo, op='O1', weights=None, E=E_GRID, cp=1.0):
    """dR/dE [events/(t yr keV)] for c_p = cp, c_n = r c_p.  weights: isotope number fractions (default natural)."""
    c0, c1 = cp * (1 + r), cp * (1 - r)
    out = np.zeros(np.shape(E))
    if weights is None: weights = AB
    mass_corr = float(np.sum(AB * MN_I) / np.sum(np.asarray(weights) * MN_I))      # nuclei per tonne relative to natural Xe
    for j, A in enumerate(A_ISO):
        w = weights[j] / AB[j] * mass_corr
        if w == 0: continue
        Kr = c0 ** 2 * KER[f'{op}_00_{A}'] + 2 * c0 * c1 * KER[f'{op}_01_{A}'] + c1 ** 2 * KER[f'{op}_11_{A}']
        if E is not E_GRID: Kr = np.interp(E, E_GRID, Kr, left=0.0, right=0.0)
        out += w * Kr * halo.eta(vmin_iso(E, MN_I[j], m, delta))
    return out * (M_K / m)

def spectrum_cn(c0, c1, m, delta, halo, op='O1'):
    out = np.zeros_like(E_GRID)
    for j, A in enumerate(A_ISO):
        Kr = c0 ** 2 * KER[f'{op}_00_{A}'] + 2 * c0 * c1 * KER[f'{op}_01_{A}'] + c1 ** 2 * KER[f'{op}_11_{A}']
        out += Kr * halo.eta(vmin_iso(E_GRID, MN_I[j], m, delta))
    return out * (M_K / m)

# validation against direct WimPyDD calls (natural Xe, several r, m, delta, halo)
VAL = []
for (r, m, dlt, hk, Es, op) in [(-0.7, 1000., 0., 'sunframe', [20., 100., 200., 248.], 'O1'), (-1.0, 400., 0., 'annual', [10., 150., 248.], 'O1'),
                                 (1.0, 1000., 300., 'sunframe', [150., 248., 280.], 'O1'), (-0.744, 4000., 350., 'annual', [200., 248., 300.], 'O1'),
                                 (0.3, 1000., 250., 'annual', [120., 248.], 'O1'), (-1.0, 1000., 0., 'sunframe', [20., 100., 248.], 'O4'), (0.5, 1000., 300., 'annual', [200., 248.], 'O4')]:
    h = HALOS[hk]; c0, c1 = 1 + r, 1 - r
    H = lz.wd_hamiltonian(f'P078_val_{op}_{abs(hash((r, m, dlt)))%10000}', {1 if op == 'O1' else 4: (c0, c1)})
    direct = np.atleast_1d(lz.wd_rate(H, m, Es, halo=(h.vmin, h.deta), delta_kev=dlt))
    fact = spectrum(r, m, dlt, h, op=op, E=np.array(Es))
    for e, a, b in zip(Es, direct, fact):
        VAL.append(dict(op=op, r=r, m=m, delta=dlt, halo=hk, E=e, direct=float(a), factorised=float(b), ratio=float(b / a) if a else float('nan')))
write_csv(f'{OUT}/validation_factorised_vs_direct.csv', VAL)
say('validation factorised/direct:', [(v['op'], v['r'], v['m'], v['delta'], v['E'], round(v['ratio'], 4)) for v in VAL])
say(f'  max |ratio - 1| = {max(abs(v["ratio"] - 1) for v in VAL if np.isfinite(v["ratio"])):.2e}')

# ==================================================================================================
# Part 3: the r scan
# ==================================================================================================
def win(s, a, b):
    mk = (E_GRID >= a) & (E_GRID <= b)
    return float(np.trapezoid((s * EFF)[mk], E_GRID[mk]))

def summarise(s):
    Rlo, Rmid, Rhi = win(s, 5.4, 55), win(s, 55, 200), win(s, 200, 270)
    f = s * EFF; cum = integrate.cumulative_trapezoid(f, E_GRID, initial=0.0)
    tot = cum[-1]
    mk = (E_GRID >= 1) & (E_GRID <= 300)
    pct = float(np.interp(248.0, E_GRID, cum) / cum[np.searchsorted(E_GRID, 300.0)]) if tot > 0 else float('nan')
    mins = local_minima(E_GRID, s, 40, 400); n1 = [x for x in mins if x < 160]; n2 = [x for x in mins if x >= 160]
    peak = float(E_GRID[np.argmax(f * mk)])
    return dict(R_lo=Rlo, R_mid=Rmid, R_hi=Rhi, R_hi_over_lo=Rhi / Rlo if Rlo > 0 else float('inf'), N_lo=Rlo / Rhi if Rhi > 0 else float('inf'),
                N_mid=Rmid / Rhi if Rhi > 0 else float('inf'), dRdE_248=float(np.interp(248.0, E_GRID, s)), pct_248_in_1_300=pct,
                node1=n1[0] if n1 else float('nan'), node2=n2[0] if n2 else float('nan'), peak_accepted=peak,
                f248_over_fpeak=float(np.interp(248.0, E_GRID, f) / f[mk].max()) if f[mk].max() > 0 else float('nan'),
                onset=float(E_GRID[np.argmax(s > 1e-12 * s.max())]) if s.max() > 0 else float('nan'))

R_SCAN = np.round(np.arange(-2.0, 2.0001, 0.01), 3)
DELTAS = [0.0, 250.0, 300.0, 350.0, 380.0]
SCAN = []
for hk in ('sunframe', 'annual'):
    h = HALOS[hk]
    for m in (400.0, 1000.0, 4000.0):
        for dlt in DELTAS:
            if m != 1000.0 and dlt not in (0.0, 300.0): continue
            ref = summarise(spectrum(1.0, m, dlt, h))
            if ref['R_hi'] <= 0: continue
            for r in R_SCAN:
                s = spectrum(r, m, dlt, h); su = summarise(s)
                ref_ok = ref['dRdE_248'] > 1e-3 * ref['R_hi'] / 70.0      # isoscalar reference not on its node / below onset at 248 keV
                su.update(halo=hk, m=m, delta=dlt, r=r, rho_lo=su['R_lo'] / ref['R_lo'] if ref['R_lo'] > 0 else float('nan'), rho_hi=su['R_hi'] / ref['R_hi'],
                          rho_248=su['dRdE_248'] / ref['dRdE_248'] if ref_ok else float('nan'),
                          ratio_enh=su['R_hi_over_lo'] / ref['R_hi_over_lo'] if np.isfinite(ref['R_hi_over_lo']) and ref['R_hi_over_lo'] > 0 else float('nan'))
                SCAN.append(su)
    say(f'scan {hk} done ({time.time()-T0:.0f} s)')
write_csv(f'{OUT}/r_scan.csv', SCAN)

def pick(hk, m, dlt, r=None):
    rows = [x for x in SCAN if x['halo'] == hk and x['m'] == m and x['delta'] == dlt]
    if r is None: return rows
    return min(rows, key=lambda x: abs(x['r'] - r))

KEY = []
for hk in ('sunframe', 'annual'):
    for dlt in DELTAS:
        rows = pick(hk, 1000.0, dlt)
        if not rows: continue
        # elastic: minimise N_lo (5.4-55 keV companions).  Inelastic (delta >= 250 keV): the 5.4-55 keV window is
        # kinematically empty, so the relevant companion population is 55-200 keV -> minimise N_mid instead.
        qty = 'N_lo' if dlt == 0.0 else 'N_mid'
        fin = [x for x in rows if np.isfinite(x[qty])]
        best = min(fin, key=lambda x: x[qty]) if fin else None
        if best is not None:
            f = lambda r: summarise(spectrum(r, 1000.0, dlt, HALOS[hk]))[qty]
            res = optimize.minimize_scalar(f, bounds=(best['r'] - 0.03, best['r'] + 0.03), method='bounded', options={'xatol': 1e-5})
            r_min, N_min = float(res.x), float(res.fun)
        else:
            r_min, N_min = float('nan'), float('nan')
        maxenh = max(rows, key=lambda x: x['rho_hi'])
        maxratio = max([x for x in rows if np.isfinite(x['ratio_enh'])], key=lambda x: x['ratio_enh']) if any(np.isfinite(x['ratio_enh']) for x in rows) else None
        for lab, r in (('iso', 1.0), ('p-only', 0.0), ('xeno', -0.7), (f'{qty}-min', r_min), ('vec', -1.0), ('r=-2', -2.0), ('r=+2', 2.0)):
            x = pick(hk, 1000.0, dlt, r) if not lab.endswith('-min') else None
            if lab.endswith('-min'):
                if not np.isfinite(r_min): continue
                x = summarise(spectrum(r_min, 1000.0, dlt, HALOS[hk])); ref = pick(hk, 1000.0, dlt, 1.0)
                x.update(halo=hk, m=1000.0, delta=dlt, r=r_min, rho_lo=x['R_lo'] / ref['R_lo'] if ref['R_lo'] > 0 else float('nan'), rho_hi=x['R_hi'] / ref['R_hi'], rho_248=x['dRdE_248'] / ref['dRdE_248'],
                         ratio_enh=x['R_hi_over_lo'] / ref['R_hi_over_lo'] if ref['R_hi_over_lo'] > 0 else float('nan'))
            KEY.append(dict(label=lab, **x, r_Nlo_min=r_min, Nlo_min=N_min, r_max_rho_hi=maxenh['r'], max_rho_hi=maxenh['rho_hi'],
                            r_max_ratio_enh=maxratio['r'] if maxratio else float('nan'), max_ratio_enh=maxratio['ratio_enh'] if maxratio else float('nan')))
write_csv(f'{OUT}/key_points_1TeV.csv', KEY)
for x in KEY:
    say(f"[{x['halo']:8s} delta={x['delta']:3.0f}] {x['label']:7s} r={x['r']:+.3f}: R_hi/lo={x['R_hi_over_lo']:.3e} N_lo={x['N_lo']:.4g} N_mid={x['N_mid']:.3g} rho_lo={x['rho_lo']:.3e} rho_hi={x['rho_hi']:.3f} rho_248={x['rho_248']:.3f} "
        f"nodes {x['node1']:.1f}/{x['node2']:.1f} pct248={x['pct_248_in_1_300']:.3f} peak={x['peak_accepted']:.0f} f248/fpk={x['f248_over_fpeak']:.3f} onset={x['onset']:.0f} | r_min={x['r_Nlo_min']:.4f} Nlo_min={x['Nlo_min']:.3f} max rho_hi {x['max_rho_hi']:.2f} at r={x['r_max_rho_hi']:+.2f}; max ratio enh {x['max_ratio_enh']:.3g} at r={x['r_max_ratio_enh']:+.2f}")
# mass dependence (elastic, sunframe) for the P032 comparison
for m in (400.0, 1000.0, 4000.0):
    for r in (1.0, 0.0, -0.7, -1.0, -2.0):
        x = pick('sunframe', m, 0.0, r); say(f"  N_lo elastic sunframe m={m:.0f} r={r:+.1f}: {x['N_lo']:.4g}  (nodes {x['node1']:.1f}/{x['node2']:.1f}, rho_248 {x['rho_248']:.3f})")
    fin = [x for x in pick('sunframe', m, 0.0) if np.isfinite(x['N_lo'])]; b = min(fin, key=lambda x: x['N_lo'])
    f = lambda r: summarise(spectrum(r, m, 0.0, HALOS['sunframe']))['N_lo']
    res = optimize.minimize_scalar(f, bounds=(b['r'] - 0.03, b['r'] + 0.03), method='bounded', options={'xatol': 1e-5})
    say(f'  N_lo minimum m={m:.0f} sunframe: r_min={res.x:.4f}, N_lo={res.fun:.3f}   (P032: -0.7429/-0.7435/-0.7437, 10.35/7.28/6.27)')
    rows = pick('sunframe', m, 0.0); r_le10 = [x['r'] for x in rows if x['N_lo'] <= 10]; r_le5 = [x['r'] for x in rows if x['N_lo'] <= 5]
    say(f'    N_lo <= 10 for r in [{min(r_le10) if r_le10 else "-"}, {max(r_le10) if r_le10 else "-"}], <= 5: {"none" if not r_le5 else (min(r_le5), max(r_le5))}')
# the (2r/(1+r))-type ratio: how much of the isovector's "suppression" is form factor vs coherence
x = pick('sunframe', 1000.0, 0.0, -1.0)
say(f"isovector/isoscalar at equal c_p (1 TeV, sunframe): q->0 {((Z_XE - np.sum(AB_N*np.array(N_ISO)))/(Z_XE + np.sum(AB_N*np.array(N_ISO))))**2:.4f}  5.4-55: {x['rho_lo']:.4f}  200-270: {x['rho_hi']:.4f}  248: {x['rho_248']:.4f}")

# ==================================================================================================
# Part 4: xenophobic point -- per-isotope interference
# ==================================================================================================
XENO = []
h = HALOS['sunframe']
for r in (-0.7, -0.744, -1.0, 0.0, 1.0):
    tot248 = spectrum(r, 1000.0, 0.0, h, E=np.array([248.0]))[0]; tot20 = spectrum(r, 1000.0, 0.0, h, E=np.array([20.0]))[0]
    for j, A in enumerate(A_ISO):
        d = RESP[A]; w = np.zeros(len(A_ISO)); w[j] = AB[j]
        s248 = spectrum(r, 1000.0, 0.0, h, weights=w, E=np.array([248.0]))[0] * float(np.sum(w * MN_I) / np.sum(AB * MN_I))   # undo mass_corr -> natural-Xe share
        s20 = spectrum(r, 1000.0, 0.0, h, weights=w, E=np.array([20.0]))[0] * float(np.sum(w * MN_I) / np.sum(AB * MN_I))
        row = dict(r=r, A=A, N=d['N'], ab=d['ab'], Z_plus_rN=Z_XE + r * d['N'], share_20=s20 / tot20, share_248=s248 / tot248)
        if d['is_dyad']:
            Fp248 = float(np.interp(248, E_RESP, d['Fp'])); Fn248 = float(np.interp(248, E_RESP, d['Fn']))
            Fp20 = float(np.interp(20, E_RESP, d['Fp'])); Fn20 = float(np.interp(20, E_RESP, d['Fn']))
            row.update(Fp_248=Fp248, rFn_248=r * Fn248, amp_248=Fp248 + r * Fn248, interference_248='constructive' if Fp248 * r * Fn248 > 0 else 'destructive',
                       amp2_248_over_iso=(Fp248 + r * Fn248) ** 2 / (Fp248 + Fn248) ** 2 if (Fp248 + Fn248) != 0 else float('inf'),
                       Fp_20=Fp20, rFn_20=r * Fn20, amp_20=Fp20 + r * Fn20, interference_20='constructive' if Fp20 * r * Fn20 > 0 else 'destructive',
                       amp2_20_over_iso=(Fp20 + r * Fn20) ** 2 / (Fp20 + Fn20) ** 2)
        XENO.append(row)
write_csv(f'{OUT}/xenophobic_isotope_interference.csv', XENO)
say('xenophobic r=-0.7 per isotope at 248 keV: (A, F_p, rF_n, amplitude, interference, share_248, share_20)')
for x in XENO:
    if x['r'] == -0.7: say('  ', x['A'], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items() if k in ('Fp_248', 'rFn_248', 'amp_248', 'interference_248', 'share_248', 'share_20', 'Z_plus_rN')})
# cross-section for one 200-270 keV event in 2.84 t yr vs r (1 TeV, sunframe), and N_lo, rho at the same time
HBARC2 = 0.3894e-27   # GeV^2 cm^2 (recalled, certain)
mu_p = 1000.0 * lz.M_NUCLEON_GEV / (1000.0 + lz.M_NUCLEON_GEV)
SIG = []
for r in (1.0, 0.0, -0.7, -0.744, -1.0, -2.0, 2.0):
    x = pick('sunframe', 1000.0, 0.0, r) if r != -0.744 else [k for k in KEY if k['label'] == 'N_lo-min' and k['halo'] == 'sunframe' and k['delta'] == 0.0][0]
    cp2 = 1.0 / (x['R_hi'] * lz.LZ['exposure_tyr'])
    SIG.append(dict(r=r, sigma_p_one_event_cm2=cp2 * mu_p ** 2 / math.pi * HBARC2, sigma_n_cm2=cp2 * r ** 2 * mu_p ** 2 / math.pi * HBARC2, N_lo=x['N_lo'], rho_hi=x['rho_hi'], rho_248=x['rho_248'], rho_lo=x['rho_lo']))
write_csv(f'{OUT}/sigma_p_one_event.csv', SIG)
say('sigma_p for one 200-270 keV event (2.84 t yr, 1 TeV):', [(s['r'], f"{s['sigma_p_one_event_cm2']:.2e}") for s in SIG])

# ==================================================================================================
# Part 5: enriched / depleted targets
# ==================================================================================================
def comp(**frac):
    w = np.zeros(len(A_ISO))
    for A, f in frac.items(): w[A_ISO.index(int(A[1:]))] = f
    return w / w.sum()
nat_wo136 = AB.copy(); nat_wo136[A_ISO.index(136)] = 0; nat_wo136 /= nat_wo136.sum()
nat_wo134136 = AB.copy(); nat_wo134136[A_ISO.index(136)] = 0; nat_wo134136[A_ISO.index(134)] = 0; nat_wo134136 /= nat_wo134136.sum()
TARGETS = {'natural': AB / AB.sum(), '136Xe-depleted': nat_wo136, '134+136Xe-depleted': nat_wo134136,
           '136Xe-enriched (90/10 136/134)': comp(A136=0.9, A134=0.1), '129Xe-enriched (80 % + natural rest)': 0.8 * comp(A129=1.0) + 0.2 * (AB / AB.sum()),
           'pure 129Xe': comp(A129=1.0), 'pure 131Xe': comp(A131=1.0), 'pure 132Xe': comp(A132=1.0), 'pure 136Xe': comp(A136=1.0)}
ENR = []
for hk in ('sunframe', 'annual'):
    h = HALOS[hk]
    for dlt in (0.0, 300.0, 350.0):
        refs = {r: summarise(spectrum(r, 1000.0, dlt, h)) for r in (1.0, 0.0, -0.7, -1.0)}
        for tname, w in TARGETS.items():
            for r in (1.0, 0.0, -0.7, -1.0):
                s = spectrum(r, 1000.0, dlt, h, weights=w); su = summarise(s)
                ENR.append(dict(halo=hk, delta=dlt, target=tname, r=r, **su, R_hi_over_natural=su['R_hi'] / refs[r]['R_hi'] if refs[r]['R_hi'] > 0 else float('nan'),
                                dRdE248_over_natural=su['dRdE_248'] / refs[r]['dRdE_248'] if refs[r]['dRdE_248'] > 0 else float('nan'),
                                R_lo_over_natural=su['R_lo'] / refs[r]['R_lo'] if refs[r]['R_lo'] > 0 else float('nan'),
                                R_mid_over_natural=su['R_mid'] / refs[r]['R_mid'] if refs[r]['R_mid'] > 0 else float('nan')))
write_csv(f'{OUT}/enriched_targets.csv', ENR)
say('enriched targets (sunframe, delta=0): target, r, R_hi/natural, dRdE248/natural, N_lo, nodes')
for x in ENR:
    if x['halo'] == 'sunframe' and x['delta'] == 0.0:
        say(f"  {x['target']:38s} r={x['r']:+.1f}: R_hi/nat {x['R_hi_over_natural']:.3f}  248/nat {x['dRdE248_over_natural']:.3f}  N_lo {x['N_lo']:.4g}  nodes {x['node1']:.1f}/{x['node2']:.1f}  R_hi/lo {x['R_hi_over_lo']:.3e}")
say('enriched targets (annual, delta=300): target, r, R_hi/natural, dRdE248/natural, pct248, peak')
for x in ENR:
    if x['halo'] == 'annual' and x['delta'] == 300.0:
        say(f"  {x['target']:38s} r={x['r']:+.1f}: R_hi/nat {x['R_hi_over_natural']:.3f}  248/nat {x['dRdE248_over_natural']:.3f}  pct248 {x['pct_248_in_1_300']:.3f}  peak {x['peak_accepted']:.0f}  node2 {x['node2']:.1f}")

# decisive-test arithmetic: two targets (natural and enriched) with equal exposure; each 200-270 keV event falls into
# 'natural' or 'enriched' with probability (1, rho)/(1+rho), rho = R_hi(enr)/R_hi(nat) at the same coupling.  Events for a
# 3 sigma separation of hypotheses r1 vs r2 (mean log-likelihood-ratio vs its s.d., symmetrised, as P062).
def n_sep(p, q, z=3.0):
    l = np.log(p / q); kpq = np.sum(p * l); vp = np.sum(p * l ** 2) - kpq ** 2; kqp = -np.sum(q * l); vq = np.sum(q * l ** 2) - kqp ** 2
    return float((z * (math.sqrt(max(vp, 0)) + math.sqrt(max(vq, 0))) / (kpq + kqp)) ** 2)
TEST = []
for hk in ('sunframe', 'annual'):
    for dlt in (0.0, 300.0):
        for tname in TARGETS:
            if tname == 'natural': continue
            rho = {x['r']: x['R_hi_over_natural'] for x in ENR if x['halo'] == hk and x['delta'] == dlt and x['target'] == tname}
            for (r1, r2) in ((1.0, -1.0), (1.0, -0.7), (-0.7, -1.0), (1.0, 0.0)):
                p = np.array([1.0, rho[r1]]) / (1 + rho[r1]); q = np.array([1.0, rho[r2]]) / (1 + rho[r2])
                TEST.append(dict(halo=hk, delta=dlt, target=tname, r1=r1, r2=r2, rho1=rho[r1], rho2=rho[r2], N_3sigma_total_events=n_sep(p, q)))
write_csv(f'{OUT}/enriched_target_test.csv', TEST)
for x in TEST:
    if x['halo'] == 'sunframe' and x['delta'] == 0.0 and x['target'] in ('136Xe-enriched (90/10 136/134)', 'pure 129Xe', '136Xe-depleted'):
        say(f"  test {x['target']:32s} r {x['r1']:+.1f} vs {x['r2']:+.1f}: rho {x['rho1']:.3f} vs {x['rho2']:.3f} -> N_3sigma = {x['N_3sigma_total_events']:.0f} high-energy events (both targets)")

# ==================================================================================================
# Part 6: O4 spin-dependent
# ==================================================================================================
SPIN = []
for A in (129, 131):
    i = RESP[A]['idx']
    tot = {}
    for cur in ('Sigma_prime', 'Sigma_prime_prime'):
        W = W_cur(i, cur, E_RESP); tot[cur] = W
    W = tot['Sigma_prime'] + tot['Sigma_prime_prime']
    Wpp = 0.25 * (W[:, 0, 0] + 2 * W[:, 0, 1] + W[:, 1, 1]); Wnn = 0.25 * (W[:, 0, 0] - 2 * W[:, 0, 1] + W[:, 1, 1]); Wpn = 0.25 * (W[:, 0, 0] - W[:, 1, 1])
    Sp_over_Sn = Wpn[0] / Wnn[0]           # = S_p/S_n (signed) at q -> 0 (Sigma'+Sigma'' at q=0 is prop. to <S_t><S_t'>)
    r_cancel = -Sp_over_Sn                 # c_n/c_p at which c_p S_p + c_n S_n = 0  ->  r* = -S_p/S_n
    RESP[A]['O4'] = dict(Wpp=Wpp, Wnn=Wnn, Wpn=Wpn, W=W, Spp=tot['Sigma_prime'], Sppp=tot['Sigma_prime_prime'])
    dyad = float(np.max(np.abs(Wpn ** 2 - Wpp * Wnn)[E_RESP < 300] / np.max((Wpp * Wnn)[E_RESP < 300])))
    # O4 isoscalar / isovector quadratic forms and minima
    def F2_O4(r):
        c0, c1 = 1 + r, 1 - r
        return c0 ** 2 * W[:, 0, 0] + 2 * c0 * c1 * W[:, 0, 1] + c1 ** 2 * W[:, 1, 1]
    mins = {lab: [round(x, 1) for x in local_minima(E_RESP, F2_O4(r), 40, 420)] for lab, r in (('iso', 1.0), ('vec', -1.0), ('p', 0.0), ('n', 1e6))}
    Wp_frac_q0 = Wpp[0] / (Wpp[0] + Wnn[0] + 2 * Wpn[0])
    SPIN.append(dict(A=A, Sp_over_Sn_q0=Sp_over_Sn, r_cancel=r_cancel, proton_fraction_isoscalar_q0=Wp_frac_q0, dyad_dev=dyad, minima=mins,
                     iso_over_vec_q0=F2_O4(1.0)[0] / F2_O4(-1.0)[0], iso_over_vec_248=float(np.interp(248, E_RESP, F2_O4(1.0)) / np.interp(248, E_RESP, F2_O4(-1.0))),
                     iso_over_vec_100=float(np.interp(100, E_RESP, F2_O4(1.0)) / np.interp(100, E_RESP, F2_O4(-1.0))),
                     Sigma_pp_share_q0=tot['Sigma_prime_prime'][0, 0, 0] / (tot['Sigma_prime'][0, 0, 0] + tot['Sigma_prime_prime'][0, 0, 0])))
json.dump([{k: (v if not isinstance(v, np.floating) else float(v)) for k, v in s.items()} for s in SPIN], open(f'{OUT}/O4_spin_structure.json', 'w'), indent=1)
for s in SPIN:
    say(f"  O4 {s['A']}Xe: S_p/S_n(q->0) = {s['Sp_over_Sn_q0']:+.4f}, cancellation r* = {s['r_cancel']:+.4f}, iso/vec F2 ratio q0 {s['iso_over_vec_q0']:.3f}, 100 keV {s['iso_over_vec_100']:.3f}, 248 keV {s['iso_over_vec_248']:.3f}; minima {s['minima']}; Sigma'' share {s['Sigma_pp_share_q0']:.3f}; dyad dev {s['dyad_dev']:.3f}")
O4SCAN = []
for hk in ('sunframe', 'annual'):
    h = HALOS[hk]
    for dlt in (0.0, 300.0, 350.0):
        ref = summarise(spectrum(1.0, 1000.0, dlt, h, op='O4'))
        for r in np.round(np.concatenate([np.arange(-2, 2.001, 0.02), np.arange(-0.1, 0.1001, 0.002)]), 4):
            s = spectrum(r, 1000.0, dlt, h, op='O4'); su = summarise(s)
            su.update(halo=hk, delta=dlt, r=r, rho_hi=su['R_hi'] / ref['R_hi'], rho_lo=su['R_lo'] / ref['R_lo'] if ref['R_lo'] > 0 else float('nan'), rho_248=su['dRdE_248'] / ref['dRdE_248'],
                      ratio_enh=su['R_hi_over_lo'] / ref['R_hi_over_lo'] if ref['R_hi_over_lo'] > 0 else float('nan'))
            O4SCAN.append(su)
O4SCAN = sorted(O4SCAN, key=lambda x: (x['halo'], x['delta'], x['r']))
write_csv(f'{OUT}/O4_r_scan.csv', O4SCAN)
for hk in ('sunframe',):
    for dlt in (0.0, 300.0):
        rows = [x for x in O4SCAN if x['halo'] == hk and x['delta'] == dlt]
        for r in (1.0, 0.0, -0.7, -1.0, 2.0, -2.0):
            x = min(rows, key=lambda y: abs(y['r'] - r))
            say(f"  O4 {hk} delta={dlt:.0f} r={r:+.1f}: N_lo {x['N_lo']:.4g} N_mid {x['N_mid']:.3g} R_hi/lo {x['R_hi_over_lo']:.3e} rho_hi {x['rho_hi']:.3f} rho_248 {x['rho_248']:.3f} nodes {x['node1']:.1f}/{x['node2']:.1f} pct248 {x['pct_248_in_1_300']:.3f} ratio_enh {x['ratio_enh']:.3f}")
        qty = 'N_lo' if dlt == 0.0 else 'N_mid'
        fin = [x for x in rows if np.isfinite(x[qty])]
        b = min(fin, key=lambda x: x[qty]); w = max(fin, key=lambda x: x[qty])
        f = lambda r: summarise(spectrum(r, 1000.0, dlt, HALOS[hk], op='O4'))[qty]
        res = optimize.minimize_scalar(f, bounds=(b['r'] - 0.004, b['r'] + 0.004), method='bounded', options={'xatol': 1e-6})
        xb = summarise(spectrum(float(res.x), 1000.0, dlt, HALOS[hk], op='O4')); ref = summarise(spectrum(1.0, 1000.0, dlt, HALOS[hk], op='O4'))
        r5 = [x['r'] for x in fin if x[qty] <= 5]; r10 = [x['r'] for x in fin if x[qty] <= 10]
        say(f"    O4 {qty} over r in [-2,2]: min {res.fun:.3g} at r={res.x:+.4f} (rho_hi there {xb['R_hi']/ref['R_hi']:.4f}, rho_lo {xb['R_lo']/ref['R_lo'] if ref['R_lo']>0 else float('nan'):.2e}); max {w[qty]:.3g} at r={w['r']:+.3f}; "
            f"{qty}<=5 for r in [{min(r5) if r5 else '-'}, {max(r5) if r5 else '-'}], <=10 for r in [{min(r10) if r10 else '-'}, {max(r10) if r10 else '-'}]")
        O4MIN = dict(halo=hk, delta=dlt, qty=qty, r_min=float(res.x), min_value=float(res.fun), rho_hi_at_min=xb['R_hi'] / ref['R_hi'], r_range_le5=(min(r5), max(r5)) if r5 else None, r_range_le10=(min(r10), max(r10)) if r10 else None)
        SPIN.append(O4MIN)
# shape distinguishability iso vs vec (LZ tex l. 496 says O4 iso/vec spectra indistinguishable)
for op in ('O4', 'O1'):
    s1 = spectrum(1.0, 1000.0, 0.0, HALOS['sunframe'], op=op); s2 = spectrum(-1.0, 1000.0, 0.0, HALOS['sunframe'], op=op)
    mk = (E_GRID >= 5) & (E_GRID <= 270)
    n1 = s1[mk] / np.trapezoid(s1[mk], E_GRID[mk]); n2 = s2[mk] / np.trapezoid(s2[mk], E_GRID[mk])
    ks = float(np.max(np.abs(integrate.cumulative_trapezoid(n1, E_GRID[mk], initial=0) - integrate.cumulative_trapezoid(n2, E_GRID[mk], initial=0))))
    say(f'  {op} isoscalar vs isovector normalised shapes (5-270 keV, elastic): KS distance {ks:.4f}, max |ratio-1| {np.max(np.abs(n1/n2-1)):.3f}, ratio at 248 {float(np.interp(248, E_GRID[mk], n1/n2)):.3f}')

# ==================================================================================================
# Figures (dataviz reference palette; fixed categorical order)
# ==================================================================================================
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, RED, GREEN = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7', '#e34948', '#008300'
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'grid.color': GRID, 'axes.grid': True, 'grid.linewidth': 0.6, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
ISOCOL = {128: MAGENTA, 129: BLUE, 130: YELLOW, 131: GREEN, 132: ORANGE, 134: RED, 136: VIOLET}

# Fig 1: signed amplitudes per isotope and node positions vs r
fig, axs = plt.subplots(1, 3, figsize=(13, 4.1))
ax = axs[0]
for A in (129, 132, 136):
    d = RESP[A]
    ax.plot(E_RESP, d['fp'], color=ISOCOL[A], lw=2, label=f'$^{{{A}}}$Xe $f_p$')
    ax.plot(E_RESP, d['fn'], color=ISOCOL[A], lw=2, ls='--', label=f'$^{{{A}}}$Xe $f_n$')
ax.axhline(0, color=MUTED, lw=0.8); ax.axvline(248, color=MUTED, lw=1, ls=':')
ax.set_xlim(0, 400); ax.set_ylim(-0.12, 0.35); ax.set_xlabel('$E_R$ [keV]'); ax.set_ylabel('$f(q)/f(0)$ (signed, shell model)')
ax.set_title('proton and neutron M form factors', loc='left', color=INK); ax.legend(fontsize=7, ncol=2)
ax = axs[1]
for A in (129, 132, 136):
    d = RESP[A]
    for r, ls in ((1.0, '-'), (-1.0, '--'), (-0.7, ':')):
        a = amp_iso(A, r) / (Z_XE + d['N'])
        ax.plot(E_RESP, a, color=ISOCOL[A], lw=1.8, ls=ls, label=f'$^{{{A}}}$Xe r={r:+.1f}' if A == 129 or r == 1.0 else None)
ax.axhline(0, color=MUTED, lw=0.8); ax.axvline(248, color=MUTED, lw=1, ls=':'); ax.text(250, 0.11, 'event', color=INK2, fontsize=8)
ax.set_xlim(0, 400); ax.set_ylim(-0.06, 0.13); ax.set_xlabel('$E_R$ [keV]'); ax.set_ylabel('$(F_p + rF_n)/A$')
ax.set_title('amplitude $Z f_p + r N f_n$: r = +1 (solid), −1 (dashed), −0.7 (dotted)', loc='left', color=INK, fontsize=8.5); ax.legend(fontsize=7)
ax = axs[2]
rr = np.array([x['r'] for x in node_vs_r])
for A in RESP:
    ax.plot(rr, [x[f'node2_{A}'] for x in node_vs_r], color=ISOCOL[A], lw=1.2, label=f'$^{{{A}}}$Xe')
    ax.plot(rr, [x[f'node1_{A}'] for x in node_vs_r], color=ISOCOL[A], lw=1.2, alpha=0.5)
ax.plot(rr, [x['node2_nat'] for x in node_vs_r], color=INK, lw=2.2, label='natural Xe (minimum)')
ax.plot(rr, [x['node1_nat'] for x in node_vs_r], color=INK, lw=2.2, alpha=0.5)
ax.axhline(248, color=MUTED, lw=1, ls=':'); ax.text(-1.95, 252, 'event 248 keV', color=INK2, fontsize=8)
ax.set_xlabel('r = $f_n/f_p$'); ax.set_ylabel('node position [keV]'); ax.set_ylim(50, 400); ax.set_title('M-response nodes vs r (first: faint, second: full)', loc='left', color=INK)
ax.legend(fontsize=6.5, ncol=2, loc='upper right')
fig.tight_layout(); fig.savefig(f'{FIG}/fig1_amplitudes_nodes.png', dpi=150); plt.close(fig)

# Fig 2: R_hi/lo vs r (elastic, 3 masses) and rho_hi, rho_248, ratio enhancement vs r; inelastic N_mid vs r
fig, axs = plt.subplots(1, 3, figsize=(13, 4.1))
ax = axs[0]
for m, col in ((400.0, BLUE), (1000.0, ORANGE), (4000.0, AQUA)):
    rows = pick('sunframe', m, 0.0); ax.plot([x['r'] for x in rows], [x['N_lo'] for x in rows], color=col, lw=2, label=f'{m:.0f} GeV, Sun-frame')
rows = pick('annual', 1000.0, 0.0); ax.plot([x['r'] for x in rows], [x['N_lo'] for x in rows], color=ORANGE, lw=1.2, ls='--', label='1000 GeV, annual mean')
ax.set_yscale('log'); ax.set_xlabel('r = $f_n/f_p$'); ax.set_ylabel('$N_{lo}$ = R(5.4–55) / R(200–270 keV)'); ax.set_title('elastic O1: low-energy companions per high-energy event', loc='left', color=INK, fontsize=8.5)
for y, lab in ((5, '5'), (10, '10')): ax.axhline(y, color=MUTED, lw=0.8, ls=':'); ax.text(1.7, y * 1.15, lab, color=INK2, fontsize=8)
ax.legend(fontsize=7)
ax = axs[1]
rows = pick('sunframe', 1000.0, 0.0)
ax.plot([x['r'] for x in rows], [x['rho_hi'] for x in rows], color=BLUE, lw=2, label='ρ(200–270 keV): rate / isoscalar at equal $c_p$')
ax.plot([x['r'] for x in rows], [x['rho_248'] for x in rows], color=ORANGE, lw=2, label='ρ(248 keV)')
ax.plot([x['r'] for x in rows], [x['rho_lo'] for x in rows], color=AQUA, lw=2, label='ρ(5.4–55 keV)')
ax.plot([x['r'] for x in rows], [x['ratio_enh'] for x in rows], color=VIOLET, lw=2, label='(R$_{hi}$/R$_{lo}$) / isoscalar value')
ax.set_yscale('log'); ax.set_ylim(1e-4, 1e4); ax.set_xlabel('r = $f_n/f_p$'); ax.set_title('elastic, 1 TeV: suppression is energy dependent', loc='left', color=INK, fontsize=8.5); ax.legend(fontsize=7, loc='lower right')
ax = axs[2]
for dlt, col in ((250.0, BLUE), (300.0, ORANGE), (350.0, AQUA), (380.0, VIOLET)):
    rows = pick('annual', 1000.0, dlt)
    if rows: ax.plot([x['r'] for x in rows], [x['N_mid'] for x in rows], color=col, lw=2, label=f'δ = {dlt:.0f} keV')
rows = pick('annual', 1000.0, 0.0); ax.plot([x['r'] for x in rows], [x['N_mid'] for x in rows], color=INK2, lw=1.2, ls='--', label='elastic')
ax.set_yscale('log'); ax.set_xlabel('r = $f_n/f_p$'); ax.set_ylabel('$N_{mid}$ = R(55–200) / R(200–270 keV)'); ax.set_title('inelastic O1 (annual-mean halo, 1 TeV): mid-energy companions', loc='left', color=INK, fontsize=8.5)
ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{FIG}/fig2_rscan.png', dpi=150); plt.close(fig)

# Fig 3: enriched targets: spectra at delta = 0 and 300 for natural, 136-enriched, pure 129 (r = +1, -1)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.1))
for ax, dlt, hk in ((axs[0], 0.0, 'sunframe'), (axs[1], 300.0, 'annual')):
    h = HALOS[hk]
    for tname, col in (('natural', BLUE), ('136Xe-enriched (90/10 136/134)', ORANGE), ('pure 129Xe', AQUA)):
        for r, ls in ((1.0, '-'), (-1.0, '--')):
            s = spectrum(r, 1000.0, dlt, h, weights=TARGETS[tname]); ref = spectrum(1.0, 1000.0, dlt, h)
            ax.plot(E_GRID, np.clip(s, 1e-30, None) / ref.max(), color=col, lw=2 if r == 1 else 1.4, ls=ls, label=f'{tname.split(" (")[0]}, r={r:+.0f}')
    ax.axvline(248, color=MUTED, lw=1, ls=':'); ax.set_yscale('log'); ax.set_xlim(0 if dlt == 0 else 60, 320)
    ax.set_ylim(1e-6, 3); ax.set_xlabel('$E_R$ [keV]'); ax.set_ylabel('dR/dE per tonne / max(natural isoscalar)')
    ax.set_title(f'{"elastic, Sun-frame" if dlt == 0 else "δ = 300 keV, annual mean"}, 1 TeV, equal $c_p$', loc='left', color=INK, fontsize=8.5); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{FIG}/fig3_enriched_targets.png', dpi=150); plt.close(fig)

# Fig 4: O4 -- N_lo and rho vs r, and spectra iso/vec
fig, axs = plt.subplots(1, 2, figsize=(10, 4.1))
ax = axs[0]
rows = [x for x in O4SCAN if x['halo'] == 'sunframe' and x['delta'] == 0.0]
ax.plot([x['r'] for x in rows], [x['N_lo'] for x in rows], color=BLUE, lw=2, label='$N_{lo}$ (O4, elastic)')
ax.plot([x['r'] for x in rows], [x['rho_hi'] for x in rows], color=ORANGE, lw=2, label='ρ(200–270 keV) vs isoscalar')
rows1 = pick('sunframe', 1000.0, 0.0); ax.plot([x['r'] for x in rows1], [x['N_lo'] for x in rows1], color=INK2, lw=1.2, ls='--', label='$N_{lo}$ (O1, elastic)')
ax.set_yscale('log'); ax.set_xlabel('r = $c_n/c_p$'); ax.set_title('spin-dependent O4 vs isospin ratio (1 TeV, Sun-frame)', loc='left', color=INK, fontsize=8.5); ax.legend(fontsize=7)
ax = axs[1]
for r, col in ((1.0, BLUE), (-1.0, ORANGE), (0.0, AQUA)):
    s = spectrum(r, 1000.0, 0.0, HALOS['sunframe'], op='O4'); mk = (E_GRID >= 1) & (E_GRID <= 300)
    ax.plot(E_GRID[mk], s[mk] / np.trapezoid(s[mk], E_GRID[mk]), color=col, lw=2, label=f'O4 r={r:+.0f}')
ax.axvline(248, color=MUTED, lw=1, ls=':'); ax.set_yscale('log'); ax.set_xlabel('$E_R$ [keV]'); ax.set_ylabel('normalised dR/dE [keV$^{-1}$]')
ax.set_title('O4 shapes: isoscalar, isovector, proton-only', loc='left', color=INK, fontsize=8.5); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{FIG}/fig4_O4.png', dpi=150); plt.close(fig)

# summary JSON
def _clean(o):
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_, bool)): return bool(o)
    if isinstance(o, np.ndarray): return _clean(o.tolist())
    return o
summary = dict(isotopes=A_ISO, abundances=AB.tolist(), q0_checks=q0_rows, isotope_nodes=iso_rows, natural_nodes={lab: (node_at(r, 'node1_nat'), node_at(r, 'node2_nat')) for lab, r in R_KEY.items()},
               r_crossing_248=cross, helm_minima_A131=helm_min, validation=VAL, key_points_1TeV=KEY, sigma_p_one_event=SIG, xenophobic=XENO, enriched=ENR, enriched_test=TEST,
               O4_spin=SPIN, halo_vmax={k: HALOS[k].vmax for k in HALOS}, runtime_s=time.time() - T0)
json.dump(_clean(summary), open(f'{OUT}/P078_summary.json', 'w'), indent=1)
say('all done (%.0f s)' % (time.time() - T0))
