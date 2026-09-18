#!/usr/bin/env python
"""P017 - Xenon nuclear response at q ~ 250 MeV: form-factor nodes, isotope mixtures and the
nuclear-structure uncertainty on the couplings inferred from the LZ 248 keV event.

Run from the simulation root:  .venv/bin/python output/code/P017_nuclear_response.py
Outputs: output/work/P017/*.csv|json, output/work/P017/figures/*.png, output/work/P017/run_log.txt

Parts
 1. M (SI) response: WimPyDD shell model (DMFormFactor one-body density matrices) per isotope vs Helm
    (Lewin-Smith) vs two-parameter Fermi (2pF) density transformed numerically; nodes; isotope spread;
    shell/Helm ratio at fixed E_R.
 2. Spin responses Sigma', Sigma'' for 129Xe, 131Xe (isoscalar/isovector -> proton/neutron), q^4-weighted
    L10 kernel; isotope shares of the L10/O4/O6 rates at 248 keV.
 3. Rate sensitivity in the 225-271 keV window (and Gaussian-smeared by 23 keV) for O1 elastic (1000 GeV),
    O1 inelastic (delta = 300 keV) and L10/O4/O6 under shell-model, Helm and 2pF (and thin-shell) responses.
 4. Propagation to LZ's two-sided intervals (P007 digitisation of Fig. 6 top; own digitisation of Fig. 6
    bottom for L10).
"""
import sys, os, json, math
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
from scipy import integrate, optimize, special
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import lzcommon as lz

OUT = 'output/work/P017'; FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
LOG = open(os.path.join(OUT, 'run_log.txt'), 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()

WD = lz.wd()
xe = WD.Xe
NC = WD.nuclear_current
HBARC = lz.HBARC_GEV_FM          # GeV fm
MN = lz.M_NUCLEON_GEV
E_EVENT, DE_EVENT = 248.0, 23.0   # keV (stat)
WIN = (E_EVENT - DE_EVENT, E_EVENT + DE_EVENT)

# ---------------------------------------------------------------------------------------------
# Isotope bookkeeping.  WimPyDD carries 9 isotopes but 124Xe and 126Xe have no density matrices
# (all responses identically zero); the 7 remaining carry 99.8 % of natural xenon.
# ---------------------------------------------------------------------------------------------
ISO = []
for k, (m, name, fw, ab) in enumerate(zip(xe.mass, xe.isotopes, xe.func_w, xe.abundance)):
    A = int(name[:3]); w0 = fw(1e-6)
    active = w0[NC['M'], 0, 0] > 0
    J = {129: 0.5, 131: 1.5}.get(A, 0.0)
    ISO.append(dict(idx=k, A=A, name=name, m=float(m), fw=fw, ab=float(ab), active=bool(active), J=J,
                    WM0=float(w0[NC['M'], 0, 0])))
ACT = [d for d in ISO if d['active']]
log('active isotopes:', [(d['A'], round(d['ab'], 5)) for d in ACT], ' inactive:', [d['A'] for d in ISO if not d['active']])
log('W_M^00(0) per isotope and A^2(2J+1)/(16 pi):', [(d['A'], round(d['WM0'], 2), round(d['A']**2 * (2*d['J']+1) / (16*math.pi), 2)) for d in ACT])

def q_gev(E_keV, m_gev):
    return np.sqrt(2.0 * m_gev * np.asarray(E_keV, float) * 1e-6)

M_XE_MEAN = lz.m_nucleus_gev(lz.A_XE_MEAN)
log(f'q(248 keV, <A>=131.29) = {1e3*q_gev(248, M_XE_MEAN):.1f} MeV; q(225) = {1e3*q_gev(225, M_XE_MEAN):.1f}, q(271) = {1e3*q_gev(271, M_XE_MEAN):.1f} MeV')
log(f'q(248 keV) per isotope [MeV]: ' + ', '.join(f"{d['A']}:{1e3*q_gev(248, d['m']):.1f}" for d in ACT))

# ---------------------------------------------------------------------------------------------
# Form factors
# ---------------------------------------------------------------------------------------------
def helm_F2_q(q, A):
    """Helm |F|^2 as a function of q [GeV] (same parametrisation as lzcommon.helm_F2)."""
    q_fm = np.asarray(q, float) / HBARC
    s = 0.9; c = 1.23 * A ** (1/3) - 0.60; a = 0.52
    rn = math.sqrt(c*c + 7/3*math.pi**2*a*a - 5*s*s)
    x = q_fm * rn
    with np.errstate(divide='ignore', invalid='ignore'):
        j1 = np.where(x > 1e-6, (np.sin(x) - x*np.cos(x)) / np.maximum(x, 1e-12)**2, x/3)
        F = np.where(x > 1e-6, 3*j1/np.maximum(x, 1e-12), 1.0)
    return F**2 * np.exp(-(q_fm*s)**2), rn

def helm_F_signed(q, A):
    q_fm = np.asarray(q, float) / HBARC
    s = 0.9; c = 1.23 * A ** (1/3) - 0.60; a = 0.52
    rn = math.sqrt(c*c + 7/3*math.pi**2*a*a - 5*s*s); x = q_fm*rn
    j1 = (np.sin(x) - x*np.cos(x)) / x**2
    return 3*j1/x * np.exp(-(q_fm*s)**2/2)

# 2pF: rho(r) = rho0 / (1 + exp((r-c)/a)); c = 5.42 fm, a = 0.57 fm for 132Xe (recalled, likely), c scaled A^{1/3}
C2PF_132, A2PF = 5.42, 0.57
def fermi_F(q, A, c132=C2PF_132, a=A2PF):
    c = c132 * (A/132.0) ** (1/3)
    r = np.linspace(0, 20.0, 4001)
    rho = 1.0 / (1.0 + np.exp((r - c) / a))
    norm = integrate.trapezoid(rho * r**2, r)
    q_fm = np.atleast_1d(np.asarray(q, float)) / HBARC
    out = np.empty_like(q_fm)
    for i, qq in enumerate(q_fm):
        x = qq * r
        j0 = np.where(x > 1e-8, np.sin(x)/np.maximum(x, 1e-12), 1.0)
        out[i] = integrate.trapezoid(rho * r**2 * j0, r) / norm
    return out if out.size > 1 else float(out[0])

def rms_radius_2pF(A):
    c = C2PF_132 * (A/132.0) ** (1/3); r = np.linspace(0, 20, 4001)
    rho = 1/(1+np.exp((r-c)/A2PF))
    return math.sqrt(integrate.trapezoid(rho*r**4, r)/integrate.trapezoid(rho*r**2, r))

def helm_rms(A):
    s = 0.9; c = 1.23*A**(1/3) - 0.60; a = 0.52
    rn2 = c*c + 7/3*math.pi**2*a*a - 5*s*s
    return math.sqrt(rn2 + 3*s*s)   # <r^2> = 3/5 rn^2 ... (Helm: <r^2> = 3/5 rn^2 + 3 s^2) -> use proper
def helm_rms_proper(A):
    s = 0.9; c = 1.23*A**(1/3) - 0.60; a = 0.52
    rn2 = c*c + 7/3*math.pi**2*a*a - 5*s*s
    return math.sqrt(0.6*rn2 + 3*s*s)
log(f'rms radii 132Xe: 2pF {rms_radius_2pF(132):.3f} fm, Helm {helm_rms_proper(132):.3f} fm (recalled charge radius 132Xe ~ 4.79 fm, likely)')

def shell_W(d, q, key, tau=0, taup=0):
    """WimPyDD response W_key^{tau tau'}(q) for isotope dict d, q array in GeV."""
    q = np.atleast_1d(np.asarray(q, float))
    return np.array([d['fw'](float(qq))[NC[key], tau, taup] for qq in q])

def shell_F2(d, q, key='M'):
    return shell_W(d, q, key) / shell_W(d, np.array([1e-6]), key)[0]

# ---------------------------------------------------------------------------------------------
# Part 1: M response vs E_R, nodes, isotope spread, ratios
# ---------------------------------------------------------------------------------------------
E_grid = np.linspace(1.0, 420.0, 4191)     # keV, 0.1 keV steps
wA2 = np.array([d['ab'] * d['A']**2 for d in ACT]); wA2 /= wA2.sum()

F2_shell_iso, F2_helm_iso, F2_2pf_iso = {}, {}, {}
for d in ACT:
    q = q_gev(E_grid, d['m'])
    F2_shell_iso[d['A']] = shell_F2(d, q, 'M')
    F2_helm_iso[d['A']] = helm_F2_q(q, d['A'])[0]
    F2_2pf_iso[d['A']] = fermi_F(q, d['A'])**2
F2_shell = sum(w * F2_shell_iso[d['A']] for w, d in zip(wA2, ACT))
F2_helm = sum(w * F2_helm_iso[d['A']] for w, d in zip(wA2, ACT))
F2_2pf = sum(w * F2_2pf_iso[d['A']] for w, d in zip(wA2, ACT))
# single-isotope-like reference at <A>: Helm with A=131.29 (as lzcommon.dRdE_SI does isotope by isotope anyway)

def local_minima(E, y, Emin=30.0):
    y = np.asarray(y); idx = np.where((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:]))[0] + 1
    return [float(E[i]) for i in idx if E[i] > Emin]

nodes = []
for d in ACT:
    for lab, F2 in (('shell', F2_shell_iso[d['A']]), ('Helm', F2_helm_iso[d['A']]), ('2pF', F2_2pf_iso[d['A']])):
        mins = local_minima(E_grid, F2)
        q1 = 1e3*q_gev(mins[0], d['m']) if len(mins) > 0 else np.nan
        q2 = 1e3*q_gev(mins[1], d['m']) if len(mins) > 1 else np.nan
        nodes.append(dict(A=d['A'], model=lab, E_node1_keV=mins[0] if mins else np.nan, E_node2_keV=mins[1] if len(mins) > 1 else np.nan,
                          q_node1_MeV=q1, q_node2_MeV=q2, F2_at_node2=float(np.interp(mins[1], E_grid, F2)) if len(mins) > 1 else np.nan))
for lab, F2 in (('shell', F2_shell), ('Helm', F2_helm), ('2pF', F2_2pf)):
    mins = local_minima(E_grid, F2)
    nodes.append(dict(A=0, model=lab, E_node1_keV=mins[0] if mins else np.nan, E_node2_keV=mins[1] if len(mins) > 1 else np.nan,
                      q_node1_MeV=1e3*q_gev(mins[0], M_XE_MEAN) if mins else np.nan, q_node2_MeV=1e3*q_gev(mins[1], M_XE_MEAN) if len(mins) > 1 else np.nan,
                      F2_at_node2=float(np.interp(mins[1], E_grid, F2)) if len(mins) > 1 else np.nan))
nodes = pd.DataFrame(nodes); nodes.to_csv(os.path.join(OUT, 'P017_M_nodes.csv'), index=False, float_format='%.4g')
log('\n== M-response nodes (A=0 : isotope-weighted natural Xe) ==\n' + nodes.to_string(index=False))
for lab in ('shell', 'Helm', '2pF'):
    sub = nodes[(nodes.model == lab) & (nodes.A > 0)]
    log(f'{lab}: node-2 spread over isotopes {sub.E_node2_keV.min():.1f}-{sub.E_node2_keV.max():.1f} keV; node-1 {sub.E_node1_keV.min():.1f}-{sub.E_node1_keV.max():.1f} keV')

E_pts = [50, 100, 150, 200, 225, 248, 265, 270, 300]
rat = pd.DataFrame(dict(E_keV=E_pts,
                        q_MeV=[1e3*q_gev(e, M_XE_MEAN) for e in E_pts],
                        F2_shell=[np.interp(e, E_grid, F2_shell) for e in E_pts],
                        F2_helm=[np.interp(e, E_grid, F2_helm) for e in E_pts],
                        F2_2pf=[np.interp(e, E_grid, F2_2pf) for e in E_pts]))
rat['shell_over_helm'] = rat.F2_shell / rat.F2_helm; rat['twopf_over_helm'] = rat.F2_2pf / rat.F2_helm; rat['shell_over_2pf'] = rat.F2_shell / rat.F2_2pf
rat.to_csv(os.path.join(OUT, 'P017_M_ratios.csv'), index=False, float_format='%.4g')
log('\n== isotope-weighted M form factor and ratios ==\n' + rat.to_string(index=False))
# 132Xe alone (single isotope, no smearing by mixture)
d132 = [d for d in ACT if d['A'] == 132][0]
r132 = pd.DataFrame(dict(E_keV=E_pts, F2_shell_132=[np.interp(e, E_grid, F2_shell_iso[132]) for e in E_pts],
                         F2_helm_132=[np.interp(e, E_grid, F2_helm_iso[132]) for e in E_pts],
                         F2_2pf_132=[np.interp(e, E_grid, F2_2pf_iso[132]) for e in E_pts]))
r132['shell_over_helm_132'] = r132.F2_shell_132 / r132.F2_helm_132
log('\n== 132Xe alone ==\n' + r132.to_string(index=False))
# depth of the node: mixture vs single isotope
log(f"node-2 depth: F2_shell(natural) min = {F2_shell[(E_grid>200)&(E_grid<320)].min():.3e} vs 132Xe alone {F2_shell_iso[132][(E_grid>200)&(E_grid<320)].min():.3e}; "
    f"F2(248) natural = {np.interp(248,E_grid,F2_shell):.3e}, 132Xe = {np.interp(248,E_grid,F2_shell_iso[132]):.3e}")

# ---------------------------------------------------------------------------------------------
# Part 2: spin responses for 129Xe and 131Xe
# ---------------------------------------------------------------------------------------------
def pn_combos(d, q, key):
    """Return W^{00}, W^{11}, W^{01}, and proton-only / neutron-only combinations for WimPyDD's
    convention c^0 = c_p + c_n, c^1 = c_p - c_n:  W_pp = W00 + W01 + W10 + W11,  W_nn = W00 - W01 - W10 + W11."""
    W00 = shell_W(d, q, key, 0, 0); W11 = shell_W(d, q, key, 1, 1); W01 = shell_W(d, q, key, 0, 1); W10 = shell_W(d, q, key, 1, 0)
    return dict(W00=W00, W11=W11, W01=W01, Wpp=W00 + W01 + W10 + W11, Wnn=W00 - W01 - W10 + W11)

spin = {}
E_sp = np.linspace(0.5, 420.0, 840)
for A in (129, 131):
    d = [x for x in ACT if x['A'] == A][0]; q = q_gev(E_sp, d['m'])
    spin[A] = dict(E=E_sp, q=q, Sp=pn_combos(d, q, 'Sigma_prime'), Spp=pn_combos(d, q, 'Sigma_prime_prime'),
                   Delta=pn_combos(d, q, 'Delta'), Phipp=pn_combos(d, q, 'Phi_prime_prime'), M=pn_combos(d, q, 'M'))
    sp0 = spin[A]['Sp']; spp0 = spin[A]['Spp']
    log(f"\n{A}Xe at q->0: Sigma' W00={sp0['W00'][0]:.4e} W11={sp0['W11'][0]:.4e} W01={sp0['W01'][0]:.4e}; Wnn={sp0['Wnn'][0]:.4e} Wpp={sp0['Wpp'][0]:.4e} (nn/pp={sp0['Wnn'][0]/sp0['Wpp'][0]:.0f});"
        f" Sigma'' W00={spp0['W00'][0]:.4e}, Sigma'/Sigma''={sp0['W00'][0]/spp0['W00'][0]:.3f}")
    log(f"   implied |<S_n>|/|<S_p>| = {math.sqrt(sp0['Wnn'][0]/sp0['Wpp'][0]):.1f}; sign of <S_p><S_n>: {'same' if sp0['W00'][0] > sp0['W11'][0] else 'opposite'} (W00>W11 <=> S_p S_n > 0)")
# ratio of q->0 neutron responses 129/131 (per 2J+1 normalisation) vs recalled <S_n>: 0.329 (129Xe), -0.272 (131Xe) (Menendez et al 2012, likely)
r_wd = (spin[129]['Spp']['Wnn'][0]/2.0) / (spin[131]['Spp']['Wnn'][0]/4.0)
r_rec = (3*0.329**2) / ((5/3)*0.272**2)
log(f"neutron Sigma'' (0) ratio 129/131 with (2J+1) removed: WimPyDD {r_wd:.2f} vs (J+1)/J <S_n>^2 from recalled Menendez-2012 spins {r_rec:.2f}")

# spin nodes and q^4-weighted kernel
spin_rows = []
for A in (129, 131):
    s = spin[A]; q = s['q']
    for key, lab in (('Sp', "Sigma'"), ('Spp', "Sigma''")):
        for comb in ('W00', 'Wnn', 'Wpp'):
            y = s[key][comb]; mins = local_minima(E_sp, y, 20.0)
            y0 = y[0]; ratio248 = np.interp(248, E_sp, y) / y0
            k = q**4 * y; kpeak = E_sp[np.argmax(k)]
            spin_rows.append(dict(A=A, response=lab, combo=comb, minima_keV=';'.join(f'{m:.0f}' for m in mins) if mins else 'none <420 keV',
                                  W_over_W0_at_248=ratio248, W_over_W0_at_150=np.interp(150, E_sp, y)/y0, W_over_W0_at_400=np.interp(400, E_sp, y)/y0,
                                  q4W_peak_keV=kpeak, q4W_248_over_peak=np.interp(248, E_sp, k)/k.max()))
spin_df = pd.DataFrame(spin_rows); spin_df.to_csv(os.path.join(OUT, 'P017_spin_responses.csv'), index=False, float_format='%.4g')
log('\n== spin responses ==\n' + spin_df.to_string(index=False))
# save the curves
np.savez(os.path.join(OUT, 'P017_response_curves.npz'), E_grid=E_grid, F2_shell=F2_shell, F2_helm=F2_helm, F2_2pf=F2_2pf,
         **{f'F2_shell_{A}': F2_shell_iso[A] for A in F2_shell_iso}, **{f'F2_helm_{A}': F2_helm_iso[A] for A in F2_helm_iso},
         E_sp=E_sp, **{f'{A}_{k}_{c}': spin[A][k][c] for A in spin for k in ('Sp', 'Spp', 'Delta', 'Phipp', 'M') for c in ('W00', 'W11', 'W01', 'Wnn', 'Wpp')})

# Other responses at the event: Delta, Phi'' (isotope-weighted, normalised at q->0) for context
ctx = []
for key in ('M', 'Sigma_prime', 'Sigma_prime_prime', 'Delta', 'Phi_prime_prime'):
    num = 0.0; den = 0.0
    for d in ACT:
        w0 = shell_W(d, np.array([1e-6]), key)[0]
        if w0 <= 0: continue
        wq = shell_W(d, q_gev(np.array([248.0]), d['m']), key)[0]
        num += d['ab'] * wq / (2*d['J']+1); den += d['ab'] * w0 / (2*d['J']+1)
    ctx.append(dict(response=key, W248_over_W0_natural=num/den))
ctx = pd.DataFrame(ctx); ctx.to_csv(os.path.join(OUT, 'P017_all_responses_at_248.csv'), index=False, float_format='%.4g')
log('\n== isoscalar responses at 248 keV relative to q->0 (abundance-weighted) ==\n' + ctx.to_string(index=False))

# ---------------------------------------------------------------------------------------------
# Part 3: rates.  WimPyDD spectra per isotope, then phenomenological re-weighting of the q-dependence.
# ---------------------------------------------------------------------------------------------
halo = lz.wd_halo()                       # Baxter-2021 SHM, time-averaged (P007: reproduces LZ's intervals best)
c0u, c1u = lz.wd_c_from_anand(1.0 / lz.M_V_GEV**2)   # LZ unit coupling
H = {
    'O1': lz.wd_hamiltonian('P017_O1s', {1: (c0u, c1u)}),
    'O4': lz.wd_hamiltonian('P017_O4s', {4: (c0u, c1u)}),
    'O6': lz.wd_hamiltonian('P017_O6s', {6: (c0u, c1u)}),
    'L10': WD.eft_hamiltonian('P017_L10like', {(4, 'q2'): lambda q, A=c0u: [A * q**2 / MN**2, 0.0], 6: lambda A=c0u: [-A, 0.0]}),
}
RESP_OF = {'O1': ('M',), 'O4': ('Sigma_prime', 'Sigma_prime_prime'), 'O6': ('Sigma_prime_prime',), 'L10': ('Sigma_prime',)}
MCHI = 1000.0
E_rate = np.arange(2.0, 420.1, 2.0)

def wd_iso_rate(ham, d, E, delta=0.0):
    return np.array([WD.diff_rate(xe, ham, MCHI, float(e), halo[0], halo[1], delta=delta, isotopes_list={0: [d['idx']]}) for e in E]) * 1000.0 * 365.25

def pheno_F2(d, q, kind):
    if kind == 'Helm': return helm_F2_q(q, d['A'])[0]
    if kind == '2pF': return fermi_F(q, d['A'])**2
    if kind == 'thinshell':          # Lewin-Smith SD recommendation: j0^2(q r_n)
        rn = helm_F2_q(q, d['A'])[1]; x = q / HBARC * rn
        return np.where(x > 1e-8, np.sin(x)/np.maximum(x, 1e-12), 1.0)**2
    raise ValueError(kind)

cases = [('O1_el', 'O1', 0.0), ('O1_inel300', 'O1', 300.0), ('L10', 'L10', 0.0), ('O4', 'O4', 0.0), ('O6', 'O6', 0.0)]
spectra = {}
for cname, op, delta in cases:
    tot = {'shell': np.zeros_like(E_rate)}
    kinds = ('Helm', '2pF') if op == 'O1' else ('Helm', 'thinshell')
    for k in kinds: tot[k] = np.zeros_like(E_rate)
    for d in ACT:
        r_i = wd_iso_rate(H[op], d, E_rate, delta)
        q = q_gev(E_rate, d['m'])
        # shell-model q-dependence of the relevant (isoscalar) response(s), normalised at q->0
        Wq = sum(shell_W(d, q, key) for key in RESP_OF[op]); W0 = sum(shell_W(d, np.array([1e-6]), key)[0] for key in RESP_OF[op])
        if W0 <= 0: continue
        F2s = Wq / W0
        tot['shell'] += r_i
        for k in kinds:
            with np.errstate(divide='ignore', invalid='ignore'):
                scale = np.where(F2s > 0, pheno_F2(d, q, k) / F2s, 0.0)
            tot[k] += r_i * scale
    spectra[cname] = tot
    log(f'{cname}: shell rate at 10/50/248 keV = {np.interp(10,E_rate,tot["shell"]):.4g} / {np.interp(50,E_rate,tot["shell"]):.4g} / {np.interp(248,E_rate,tot["shell"]):.4g} /t/yr/keV')

# validation: Helm re-weighted WimPyDD O1 vs lzcommon.dRdE_SI (Helm, own kinematics) with sigma_n from c_n = 1/m_v^2
mu_n = lz.mu_red(MCHI, MN); sigma_n = (1/lz.M_V_GEV**2)**2 * mu_n**2 / math.pi * lz.GEV_TO_CM2
log(f'\nsigma_n(unit coupling, 1000 GeV) = {sigma_n:.3e} cm^2')
val = []
for e in (10, 50, 100, 150, 200, 225, 248, 270):
    helm_lz = lz.dRdE_SI(e, MCHI, sigma_n)
    val.append(dict(E_keV=e, WD_shell=np.interp(e, E_rate, spectra['O1_el']['shell']), WD_reweighted_Helm=np.interp(e, E_rate, spectra['O1_el']['Helm']),
                    lzcommon_Helm=helm_lz, reweighted_over_lzcommon=np.interp(e, E_rate, spectra['O1_el']['Helm'])/helm_lz,
                    shell_over_lzHelm=np.interp(e, E_rate, spectra['O1_el']['shell'])/helm_lz))
val = pd.DataFrame(val); val.to_csv(os.path.join(OUT, 'P017_validation_helm.csv'), index=False, float_format='%.4g')
log('== validation: re-weighted WimPyDD (Helm) vs lzcommon Helm SI ==\n' + val.to_string(index=False))

# window integrals, plain and Gaussian-smeared (sigma = 23 keV)
def smear(E, y, sig=DE_EVENT):
    dE = E[1] - E[0]; k = np.arange(-int(4*sig/dE), int(4*sig/dE) + 1) * dE
    g = np.exp(-0.5*(k/sig)**2); g /= g.sum()
    return np.convolve(y, g, mode='same')
def win_int(E, y, lo=WIN[0], hi=WIN[1]):
    m = (E >= lo) & (E <= hi); return integrate.trapezoid(y[m], E[m])

rows = []
for cname, tot in spectra.items():
    base = win_int(E_rate, tot['shell']); base_s = win_int(E_rate, smear(E_rate, tot['shell']))
    for k, y in tot.items():
        rows.append(dict(case=cname, response=k, R_window_per_t_yr=win_int(E_rate, y), ratio_to_shell=win_int(E_rate, y)/base,
                         R_window_smeared=win_int(E_rate, smear(E_rate, y)), ratio_to_shell_smeared=win_int(E_rate, smear(E_rate, y))/base_s,
                         dRdE_248=np.interp(248, E_rate, y), dRdE_230=np.interp(230, E_rate, y), dRdE_270=np.interp(270, E_rate, y),
                         max_over_min_230_270=(lambda m: y[m].max()/max(y[m].min(), 1e-300))((E_rate >= 230) & (E_rate <= 270)),
                         coupling2_factor_if_true=base/max(win_int(E_rate, y), 1e-300)))
win = pd.DataFrame(rows); win.to_csv(os.path.join(OUT, 'P017_window_rates.csv'), index=False, float_format='%.4g')
log('\n== 225-271 keV window rates (unit coupling, 1000 GeV) ==\n' + win.to_string(index=False))
for cname, tot in spectra.items():
    y = tot['shell']; m = (E_rate >= 230) & (E_rate <= 270)
    log(f'{cname}: shell dR/dE max/min over 230-270 keV = {y[m].max()/y[m].min():.3g}; dR/dE(230)/dR/dE(248) = {np.interp(230,E_rate,y)/np.interp(248,E_rate,y):.3g}; (270)/(248) = {np.interp(270,E_rate,y)/np.interp(248,E_rate,y):.3g}')

# isotope shares at 248 keV for the spin operators (and O1 for reference)
share = []
for cname, op, delta in cases:
    r = {d['A']: float(WD.diff_rate(xe, H[op], MCHI, 248.0, halo[0], halo[1], delta=delta, isotopes_list={0: [d['idx']]})) for d in ACT}
    s = sum(r.values())
    share.append(dict(case=cname, **{f'f_{A}': (r[A]/s if s > 0 else np.nan) for A in r}))
share = pd.DataFrame(share); share.to_csv(os.path.join(OUT, 'P017_isotope_shares_248.csv'), index=False, float_format='%.4g')
log('\n== isotope shares of dR/dE at 248 keV ==\n' + share.to_string(index=False))
for E0 in (50.0, 150.0):
    r = {A: float(WD.diff_rate(xe, H['L10'], MCHI, E0, halo[0], halo[1], isotopes_list={0: [d['idx']]})) for A, d in ((d['A'], d) for d in ACT if d['J'] > 0)}
    log(f'L10 129Xe share at {E0:.0f} keV: {r[129]/(r[129]+r[131]):.3f}')

# ---------------------------------------------------------------------------------------------
# Part 4: propagation to LZ's intervals
# ---------------------------------------------------------------------------------------------
# 4a. O1 inelastic, P007 digitised Fig. 6 top; ratio shell/Helm in the window for delta = 250, 300, 350
p7 = pd.read_csv('output/work/P007/lz_intervals_digitised.csv')
prop = []
for delta in (250.0, 300.0, 350.0):
    tot = {'shell': np.zeros_like(E_rate), 'Helm': np.zeros_like(E_rate), '2pF': np.zeros_like(E_rate)}
    for d in ACT:
        r_i = wd_iso_rate(H['O1'], d, E_rate, delta); q = q_gev(E_rate, d['m'])
        F2s = shell_F2(d, q, 'M'); tot['shell'] += r_i
        for k in ('Helm', '2pF'):
            with np.errstate(divide='ignore', invalid='ignore'):
                tot[k] += r_i * np.where(F2s > 0, pheno_F2(d, q, k)/F2s, 0.0)
    spectra[f'O1_inel{int(delta)}'] = tot
    row = p7[p7.delta_keV == int(delta)].iloc[0]
    rw = {k: win_int(E_rate, tot[k]) for k in tot}; rs = {k: win_int(E_rate, smear(E_rate, tot[k])) for k in tot}
    # full-ROI (efficiency-weighted) integrals as the more faithful proxy for what the PLR sees
    eff = 0.96 * 0.5*(1+special.erf((E_rate-5.4)/(math.sqrt(2)*2.5))) * 0.5*(1-special.erf((E_rate-269.9)/(math.sqrt(2)*11.5)))
    rroi = {k: integrate.trapezoid(tot[k]*eff, E_rate) for k in tot}
    for k in ('Helm', '2pF'):
        f_win, f_sm, f_roi = rw['shell']/rw[k], rs['shell']/rs[k], rroi['shell']/rroi[k]
        prop.append(dict(delta_keV=delta, alt_response=k, factor_window=f_win, factor_smeared=f_sm, factor_fullROI=f_roi,
                         LZ_lower=row.c1s_mv2_sq_lower, LZ_upper=row.c1s_mv2_sq_upper,
                         lower_if_alt_fullROI=row.c1s_mv2_sq_lower*f_roi, upper_if_alt_fullROI=row.c1s_mv2_sq_upper*f_roi,
                         N_unit_shell_ROI_2p84tyr=rroi['shell']*2.84, N_unit_alt_ROI_2p84tyr=rroi[k]*2.84))
prop = pd.DataFrame(prop); prop.to_csv(os.path.join(OUT, 'P017_O1_interval_propagation.csv'), index=False, float_format='%.4g')
log('\n== O1 inelastic: coupling^2 factor (shell/alt) and shifted LZ intervals ==\n' + prop.to_string(index=False))

# 4b. L10: digitise Fig. 6 bottom (own digitisation, same method as P007)
import pymupdf
page = pymupdf.open('inputs/arXiv_2609.02823_source/Fig6_O1_L10_limit_stacked.pdf')[0]
dr = page.get_drawings()
xt = np.array([73.1, 231.9, 390.7]); xv = np.array([1, 2, 3]); sx, ix = np.polyfit(xt, xv, 1)
yt = np.array([629.1, 536.6, 444.0, 351.4]); yv = np.array([-1, 0, 1, 2]); sy, iy = np.polyfit(yt, yv, 1)
def conv(idx):
    pts = set()
    for it in dr[idx]['items']:
        if it[0] == 'l': pts.add((it[1].x, it[1].y)); pts.add((it[2].x, it[2].y))
    pts = sorted(pts)
    return [(10**(sx*x+ix), 10**(sy*y+iy)) for x, y in pts]
l10 = dict(upper=conv(349), lower=conv(350), median=conv(346))
json.dump(l10, open(os.path.join(OUT, 'P017_L10_interval_digitised.json'), 'w'), indent=1)
# what does the bottom y-axis label say?  dump all text tokens left of the bottom panel
ylab = [(round(0.5*(w[0]+w[2]), 1), round(0.5*(w[1]+w[3]), 1), w[4]) for w in page.get_text('words') if 0.5*(w[0]+w[2]) < 40]
log('\nFig. 6 y-axis label tokens (x<40 pt): top panel ' + str([t for t in ylab if t[1] < 330]) + ' | bottom panel ' + str([t for t in ylab if t[1] >= 330]))
def at_mass(curve, m):
    xs = np.log10([p[0] for p in curve]); ys = np.log10([p[1] for p in curve]); o = np.argsort(xs)
    return 10**np.interp(math.log10(m), xs[o], ys[o])
log('== Fig. 6 bottom (L10^s) digitised; axis value v (either (d10 m_v^2)^2 or d10 m_v^2, see label tokens) ==')
for m in (100, 200, 400, 1000, 4000):
    lo = at_mass(l10['lower'], m) if m >= 200 else np.nan
    log(f'  m={m}: upper {at_mass(l10["upper"], m):.3g}, median {at_mass(l10["median"], m):.3g}, lower {lo:.3g}')
log('  lower-curve x range (GeV): %.1f - %.1f' % (min(p[0] for p in l10['lower']), max(p[0] for p in l10['lower'])))
# consistency: unit-coupling L10 events in the ROI at 1000 GeV.  P003 found LZ's L10^s curve = 247.5 x our
# (q^2/mN^2)O4 - O6 combination at A = 1/m_v^2, i.e. 247.5/4 = 61.9 at our A = 2/m_v^2.
eff = 0.96 * 0.5*(1+special.erf((E_rate-5.4)/(math.sqrt(2)*2.5))) * 0.5*(1-special.erf((E_rate-269.9)/(math.sqrt(2)*11.5)))
P003_SCALE = 247.5 / 4.0
N_unit_L10 = integrate.trapezoid(spectra['L10']['shell']*eff, E_rate) * 2.84 * P003_SCALE
up1000, lo1000, med1000 = at_mass(l10['upper'], 1000), at_mass(l10['lower'], 1000), at_mass(l10['median'], 1000)
log(f'  L10 unit-coupling ROI events at 1000 GeV (P003 scale x{P003_SCALE:.1f}, 2.84 t yr, eff. model): {N_unit_L10:.3g}')
log(f'  if axis = (d10 m_v^2)^2: edges -> {N_unit_L10*lo1000:.3g} .. {N_unit_L10*up1000:.3g} events (median {N_unit_L10*med1000:.3g})')
log(f'  if axis = d10 m_v^2     : edges -> {N_unit_L10*lo1000**2:.3g} .. {N_unit_L10*up1000**2:.3g} events (median {N_unit_L10*med1000**2:.3g});  1-event PLR interval ~ [0.1, 3.7], P007 found 0.16-0.29 / 3.2-3.8 for O1')
l10_prop = []
for k in ('Helm', 'thinshell'):
    rroi_s = integrate.trapezoid(spectra['L10']['shell']*eff, E_rate); rroi_k = integrate.trapezoid(spectra['L10'][k]*eff, E_rate)
    f = rroi_s / rroi_k
    fw = win_int(E_rate, spectra['L10']['shell'])/win_int(E_rate, spectra['L10'][k])
    fs = win_int(E_rate, smear(E_rate, spectra['L10']['shell']))/win_int(E_rate, smear(E_rate, spectra['L10'][k]))
    l10_prop.append(dict(alt_response=k, factor_coupling2_fullROI=f, factor_coupling2_window=fw, factor_coupling2_smeared=fs,
                         LZ_upper_1000=up1000, LZ_lower_1000=lo1000, upper_if_alt_linear_axis=up1000*math.sqrt(f), lower_if_alt_linear_axis=lo1000*math.sqrt(f)))
l10_prop = pd.DataFrame(l10_prop); l10_prop.to_csv(os.path.join(OUT, 'P017_L10_interval_propagation.csv'), index=False, float_format='%.4g')
log(l10_prop.to_string(index=False))

# ---------------------------------------------------------------------------------------------
# Robustness: radius vs shape.  2pF with c tuned so that its rms radius equals Helm's; Helm with s = 0.8 / 1.0 fm.
# ---------------------------------------------------------------------------------------------
rob = []
d = d132; q = q_gev(E_grid, d['m'])
def node2(F2):
    mins = local_minima(E_grid, F2); return mins[1] if len(mins) > 1 else np.nan
rob.append(dict(variant='shell model 132Xe', rms_fm=np.nan, E_node2_keV=node2(F2_shell_iso[132]), F2_248=np.interp(248, E_grid, F2_shell_iso[132])))
rob.append(dict(variant='Helm s=0.9 (default)', rms_fm=helm_rms_proper(132), E_node2_keV=node2(F2_helm_iso[132]), F2_248=np.interp(248, E_grid, F2_helm_iso[132])))
for s in (0.8, 1.0):
    q_fm = q/HBARC; c = 1.23*132**(1/3) - 0.60; a = 0.52; rn = math.sqrt(c*c + 7/3*math.pi**2*a*a - 5*s*s); x = q_fm*rn
    F2 = (3*(np.sin(x) - x*np.cos(x))/x**3)**2 * np.exp(-(q_fm*s)**2)
    rob.append(dict(variant=f'Helm s={s}', rms_fm=math.sqrt(0.6*rn*rn + 3*s*s), E_node2_keV=node2(F2), F2_248=np.interp(248, E_grid, F2)))
rob.append(dict(variant='2pF c=5.42 a=0.57 (default)', rms_fm=rms_radius_2pF(132), E_node2_keV=node2(F2_2pf_iso[132]), F2_248=np.interp(248, E_grid, F2_2pf_iso[132])))
target_rms = helm_rms_proper(132)
def rms_c(c):
    r = np.linspace(0, 20, 4001); rho = 1/(1+np.exp((r-c)/A2PF)); return math.sqrt(integrate.trapezoid(rho*r**4, r)/integrate.trapezoid(rho*r**2, r))
c_match = optimize.brentq(lambda c: rms_c(c) - target_rms, 5.0, 6.0)
F2m = fermi_F(q, 132, c132=c_match)**2
rob.append(dict(variant=f'2pF c={c_match:.3f} (rms matched to Helm)', rms_fm=target_rms, E_node2_keV=node2(F2m), F2_248=np.interp(248, E_grid, F2m)))
for a in (0.50, 0.65):
    r = np.linspace(0, 20, 4001); rho = 1/(1+np.exp((r-C2PF_132)/a)); norm = integrate.trapezoid(rho*r**2, r)
    F = np.array([integrate.trapezoid(rho*r**2*np.where(qq*r/HBARC > 1e-8, np.sin(qq*r/HBARC)/np.maximum(qq*r/HBARC, 1e-12), 1.0), r)/norm for qq in q])
    rob.append(dict(variant=f'2pF c=5.42 a={a}', rms_fm=math.sqrt(integrate.trapezoid(rho*r**4, r)/norm), E_node2_keV=node2(F**2), F2_248=np.interp(248, E_grid, F**2)))
rob = pd.DataFrame(rob); rob.to_csv(os.path.join(OUT, 'P017_robustness_132Xe.csv'), index=False, float_format='%.4g')
log('\n== robustness (132Xe): node-2 position and F^2(248 keV) vs radius/shape parameters ==\n' + rob.to_string(index=False))
summ_extra = dict(L10_axis_note='see run_log: label tokens', L10_N_unit_ROI_P003scale=float(N_unit_L10), c2pF_rms_matched=float(c_match))

# summary JSON
summ = dict(q_248_MeV=float(1e3*q_gev(248, M_XE_MEAN)), q_window_MeV=[float(1e3*q_gev(WIN[0], M_XE_MEAN)), float(1e3*q_gev(WIN[1], M_XE_MEAN))],
            nodes_natural={lab: dict(E1=float(nodes[(nodes.A == 0) & (nodes.model == lab)].E_node1_keV.iloc[0]), E2=float(nodes[(nodes.A == 0) & (nodes.model == lab)].E_node2_keV.iloc[0])) for lab in ('shell', 'Helm', '2pF')},
            node2_isotope_spread_keV={lab: [float(nodes[(nodes.model == lab) & (nodes.A > 0)].E_node2_keV.min()), float(nodes[(nodes.model == lab) & (nodes.A > 0)].E_node2_keV.max())] for lab in ('shell', 'Helm', '2pF')},
            shell_over_helm_F2={int(e): float(v) for e, v in zip(rat.E_keV, rat.shell_over_helm)},
            twopf_over_helm_F2={int(e): float(v) for e, v in zip(rat.E_keV, rat.twopf_over_helm)},
            window_ratios={r.case + '/' + r.response: float(r.ratio_to_shell) for r in win.itertuples()},
            window_ratios_smeared={r.case + '/' + r.response: float(r.ratio_to_shell_smeared) for r in win.itertuples()},
            isotope_shares_248=share.set_index('case').to_dict(orient='index'),
            L10_interval_1000GeV=dict(upper=float(up1000), lower=float(lo1000), median=float(at_mass(l10['median'], 1000)), N_unit_ROI=float(N_unit_L10)),
            sigma_n_unit_cm2=float(sigma_n), **summ_extra)
json.dump(summ, open(os.path.join(OUT, 'P017_summary.json'), 'w'), indent=1, default=float)

# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
C = dict(shell='#2a78d6', Helm='#eb6834', twopf='#1baf7a', thin='#4a3aa7', a129='#2a78d6', a131='#eb6834', ink='#0b0b0b', ink2='#52514e', grid='#e6e5e1')
plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': C['ink2'], 'axes.labelcolor': C['ink'], 'xtick.color': C['ink2'], 'ytick.color': C['ink2'],
                     'axes.grid': True, 'grid.color': C['grid'], 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'lines.linewidth': 1.8, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
def event_band(ax):
    ax.axvspan(WIN[0], WIN[1], color='#eda100', alpha=0.15, lw=0); ax.axvline(E_EVENT, color='#eda100', lw=1.2)

# Fig 1: M response
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
a1.semilogy(E_grid, F2_shell, color=C['shell'], label='shell model (WimPyDD / DMFormFactor), natural Xe')
a1.semilogy(E_grid, F2_helm, color=C['Helm'], label='Helm (Lewin-Smith)')
a1.semilogy(E_grid, F2_2pf, color=C['twopf'], label='2pF density, numerical transform')
for A, ls in ((129, ':'), (136, '--')):
    a1.semilogy(E_grid, F2_shell_iso[A], color=C['shell'], lw=0.9, ls=ls, alpha=0.8, label=f'shell model, {A}Xe only')
event_band(a1); a1.set_ylim(1e-5, 1.5); a1.set_ylabel(r'$W_M^{00}(q)/W_M^{00}(0)$  (abundance$\times A^2$ weighted)')
a1.legend(loc='lower left', fontsize=8); a1.set_title('Spin-independent (M) response of xenon vs recoil energy', loc='left', color=C['ink'])
a1.text(E_EVENT+2, 0.8, '248 ± 23 keV', color='#9a6a00', fontsize=8)
a2.semilogy(E_grid, F2_shell/F2_helm, color=C['shell'], label='shell / Helm'); a2.semilogy(E_grid, F2_2pf/F2_helm, color=C['twopf'], label='2pF / Helm')
a2.axhline(1, color=C['ink2'], lw=0.8); event_band(a2); a2.set_ylim(0.05, 20); a2.set_ylabel('ratio'); a2.set_xlabel('recoil energy $E_R$ [keV]'); a2.legend(loc='upper left', fontsize=8)
a2.set_xlim(0, 420); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P017_fig1_M_response.png'), dpi=150); plt.close(fig)

# Fig 2: spin responses and L10 kernel
fig, ax = plt.subplots(1, 3, figsize=(11, 3.7))
for A, col in ((129, C['a129']), (131, C['a131'])):
    s = spin[A]
    ax[0].semilogy(E_sp, s['Sp']['Wnn']/s['Sp']['Wnn'][0], color=col, label=f"{A}Xe  Σ' (nn)")
    ax[0].semilogy(E_sp, s['Spp']['Wnn']/s['Spp']['Wnn'][0], color=col, ls='--', label=f"{A}Xe  Σ'' (nn)")
    ax[1].semilogy(E_sp, s['Sp']['Wpp']/s['Sp']['Wpp'][0], color=col, label=f"{A}Xe  Σ' (pp)")
    ax[1].semilogy(E_sp, s['Spp']['Wpp']/s['Spp']['Wpp'][0], color=col, ls='--', label=f"{A}Xe  Σ'' (pp)")
    k = s['q']**4 * s['Sp']['W00']; ax[2].plot(E_sp, k/k.max(), color=col, label=f"{A}Xe  q⁴ Σ' (00)")
d = [x for x in ACT if x['A'] == 131][0]; qh = q_gev(E_sp, d['m'])
ax[0].semilogy(E_sp, helm_F2_q(qh, 131)[0], color=C['Helm'], lw=1.0, label='Helm |F|² (131)'); ax[0].semilogy(E_sp, pheno_F2(d, qh, 'thinshell'), color=C['thin'], lw=1.0, label='thin shell j₀²')
for a in ax: event_band(a); a.set_xlabel('$E_R$ [keV]'); a.set_xlim(0, 420)
ax[0].set_ylim(1e-3, 2); ax[1].set_ylim(1e-3, 2); ax[0].set_ylabel('response / value at q→0'); ax[0].set_title('neutron spin responses', loc='left'); ax[1].set_title('proton spin responses', loc='left')
ax[2].set_title("L10 kernel  q⁴ W_Σ'(q) (normalised)", loc='left'); ax[2].set_ylim(0, 1.1)
for a in ax: a.legend(fontsize=7.5, loc='lower left' if a is not ax[2] else 'upper left')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P017_fig2_spin_responses.png'), dpi=150); plt.close(fig)

# Fig 3: spectra under the three response models
fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
for a, (cname, title) in zip(ax, (('O1_el', 'O1 elastic, 1000 GeV'), ('O1_inel300', 'O1 inelastic, δ = 300 keV, 1000 GeV'), ('L10', 'L10-like  (q²/m_N²)O4 − O6, 1000 GeV'))):
    tot = spectra[cname]
    a.semilogy(E_rate, tot['shell'], color=C['shell'], label='shell model')
    a.semilogy(E_rate, tot['Helm'], color=C['Helm'], label='Helm-shaped')
    k3 = '2pF' if '2pF' in tot else 'thinshell'; a.semilogy(E_rate, tot[k3], color=C['twopf'] if k3 == '2pF' else C['thin'], label=k3 if k3 == '2pF' else 'thin-shell j₀²')
    a.semilogy(E_rate, smear(E_rate, tot['shell']), color=C['shell'], ls=':', lw=1.2, label='shell, σ = 23 keV smeared')
    event_band(a); a.set_title(title, loc='left', fontsize=9.5); a.set_xlabel('$E_R$ [keV]'); a.set_xlim(0, 350)
    m = (E_rate > 150) & (E_rate < 350); a.set_ylim(tot['shell'][m].min()/30, tot['shell'][m].max()*30); a.legend(fontsize=7.5, loc='lower left')
ax[0].set_ylabel('dR/dE [events / (t yr keV)], unit coupling'); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P017_fig3_spectra.png'), dpi=150); plt.close(fig)

# Fig 4: node positions per isotope
fig, a = plt.subplots(figsize=(6.4, 3.4))
for lab, col, off in (('shell', C['shell'], -0.15), ('Helm', C['Helm'], 0.0), ('2pF', C['twopf'], 0.15)):
    sub = nodes[(nodes.model == lab) & (nodes.A > 0)]
    a.scatter(sub.E_node2_keV, np.arange(len(sub)) + off, color=col, s=28, label=f'{lab}: 2nd node'); a.scatter(sub.E_node1_keV, np.arange(len(sub)) + off, color=col, s=28, marker='s', alpha=0.6)
a.set_yticks(np.arange(len(ACT))); a.set_yticklabels([f"{d['A']}Xe ({100*d['ab']:.1f} %)" for d in ACT]); event_band(a)
a.set_xlabel('$E_R$ of M-response minimum [keV]  (squares: first node, circles: second node)'); a.legend(fontsize=8, loc='center'); a.set_xlim(60, 320)
a.set_title('Form-factor node positions by isotope', loc='left'); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P017_fig4_nodes.png'), dpi=150); plt.close(fig)

# Fig 5: window-rate ratios
fig, a = plt.subplots(figsize=(6.4, 3.2))
lab_case = {'O1_el': 'O1 el.', 'O1_inel300': 'O1 δ=300', 'L10': 'L10', 'O4': 'O4', 'O6': 'O6'}
xs = np.arange(len(cases)); wdt = 0.36
for j, (k, col, name) in enumerate((('Helm', C['Helm'], 'Helm-shaped / shell'), ('alt', C['twopf'], '2pF (O1) or thin-shell (spin) / shell'))):
    vals = []
    for cname, op, _ in cases:
        kk = k if k == 'Helm' else ('2pF' if op == 'O1' else 'thinshell')
        vals.append(win[(win.case == cname) & (win.response == kk)].ratio_to_shell.iloc[0])
    a.bar(xs + (j-0.5)*wdt, vals, wdt*0.92, color=col, label=name)
    for x, v in zip(xs, vals): a.text(x + (j-0.5)*wdt, v*1.08, f'{v:.2f}', ha='center', fontsize=7.5, color=C['ink2'])
a.set_yscale('log'); a.axhline(1, color=C['ink2'], lw=0.8); a.set_xticks(xs); a.set_xticklabels([lab_case[c[0]] for c in cases]); a.set_ylim(0.005, 40)
a.set_ylabel('rate in 225–271 keV, alt. / shell model'); a.legend(fontsize=8, loc='upper right'); a.set_title('Window-rate ratio (inferred coupling² scales as its inverse)', loc='left', fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P017_fig5_window_ratios.png'), dpi=150); plt.close(fig)
log('\nfigures written to', FIG)
LOG.close()
