#!/usr/bin/env python
"""
P084 -- Sommerfeld-enhanced annihilation of the electroweak multiplets that fit the LZ event,
and the CMB / dwarf-spheroidal / Galactic-centre / antiproton constraints on them.

Run from the simulation root:  .venv/bin/python output/code/P084_sommerfeld_indirect.py

Physics summary (details in output/work/P084/details.md):
 * Fermion SU(2)_L n-plets with hypercharge Y = T = (n-1)/2 (the neutral component is the T3 = -Y state):
   (2,1/2) Higgsino-like doublet, (3,1) triplet, (4,3/2) quadruplet, (5,2) quintuplet -- the P037 LZ fits.
   Real multiplets (3,0) wino and (5,0) MDM quintuplet are computed as literature checks of the solver.
 * Tree-level s-wave annihilation of the neutral Majorana state chi1 (delta -> 0 limit) into W+W-, ZZ,
   Z gamma, gamma gamma from the SU(2) x U(1) gauge vertices, using the positronium-calibrated threshold
   formula  sigma v(chi1 chi1 -> V V') = |sum_j a_j|^2 / (32 pi m^2) (x 1/2 for identical bosons), where a_j is
   the product of the two gauge couplings of each t/u-channel diagram (P025 calibrated the same formula on
   the wino, 2 pi alpha_2^2/m^2, and the Higgsino, pi alpha_2^2/8m^2).
 * Sommerfeld enhancement: multi-channel radial Schroedinger equation in the two-body Q_tot = 0, 1S0 sector
   {psi^(Q) psibar^(Q)}, Q = 0..n-1, with W-exchange (Yukawa, m_W) off-diagonal, Z-exchange (Yukawa, m_Z) and
   photon (Coulomb) diagonal potentials, and the one-loop charged-neutral mass splittings 2 Delta m_Q as
   channel thresholds.  Solved numerically (inward integration from r_max = 12/m_W of n+1 basis solutions,
   regularity at the origin imposed by a linear solve), NOT the Hulthen approximation (Hulthen is used only
   as a single-channel validation of the solver).
 * Enhanced cross-sections:  sigma v_X = |sum_k s_k a_k^X|^2 / (32 pi m^2), s_k = channel-k wavefunction
   derivative at the origin for unit incoming flux in the neutral channel.  The gamma gamma / Z gamma lines
   arise from the charged components (s_k, k>=1) -- the Hisano et al. mechanism.
All experimental limits are recalled and flagged in details.md / provenance.
"""
import os, sys, json, time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.getcwd()
OUT = os.path.join(ROOT, "output/work/P084")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
LOG = open(os.path.join(OUT, "run_log.txt"), "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


# ----------------------------------------------------------------------------------------------
# Constants (recalled, certain unless stated)
# ----------------------------------------------------------------------------------------------
SW2 = 0.2312                     # sin^2 theta_W (MSbar at m_Z)
CW = np.sqrt(1.0 - SW2)
ALPHA2 = 1.0 / 29.6              # alpha_2(m_Z)  (g = 0.6516), as P025
G2 = np.sqrt(4 * np.pi * ALPHA2)
ALPHA_EM_0 = 1.0 / 137.036       # alpha(0): long-range Coulomb potential
ALPHA_EM_MZ = ALPHA2 * SW2       # alpha(m_Z) = alpha_2 s_W^2 = 1/128: annihilation vertices
E_MZ = np.sqrt(4 * np.pi * ALPHA_EM_MZ)
M_W, M_Z = 80.377, 91.1876       # GeV
GEV2_TO_CM3S = 1.1673e-17        # 1 GeV^-2 = 1.1673e-17 cm^3/s  (hbar^2 c^3)
SIN2_HALF_TW = (1 - CW) / 2      # sin^2(theta_W/2)

log("=== P084 run", time.strftime("%Y-%m-%d %H:%M:%S"))
log(f"alpha_2 = {ALPHA2:.5f}, g = {G2:.4f}, c_W = {CW:.4f}, alpha(0) = {ALPHA_EM_0:.6f}, alpha(mZ) = {ALPHA_EM_MZ:.6f}")


def delta_m_gev(Q, Y):
    """One-loop EW mass splitting M_Q - M_0 in the heavy-mass limit (Cirelli-Fornengo-Strumia 2006; recalled,
    likely; re-derived in details.md Sec. 2.3):  alpha_2 m_W [ Q^2 sin^2(theta_W/2) + Q Y (1-c_W)/c_W ]."""
    return ALPHA2 * M_W * (Q**2 * SIN2_HALF_TW + Q * Y * (1 - CW) / CW)


# ----------------------------------------------------------------------------------------------
# Multiplet systems: potential matrices, thresholds, annihilation vectors
# ----------------------------------------------------------------------------------------------
def dirac_multiplet(n, Y):
    """Complex (Dirac) n-plet with hypercharge Y = T = (n-1)/2: neutral state is T3 = -T.
    Channels k = Q = 0..n-1 : psi^(Q) psibar^(Q), 1S0.  Returns a dict used by the solver."""
    T = (n - 1) / 2
    assert abs(Y - T) < 1e-9, "only maximal-hypercharge multiplets (neutral = bottom of multiplet)"
    ks = np.arange(n)
    T3 = -T + ks
    Q = T3 + Y
    gZ = (G2 / CW) * (T3 - Q * SW2)                      # Z coupling of psi^(Q)
    Cup2 = (T - T3) * (T + T3 + 1)                       # |<T3+1|T+|T3>|^2 = (n-1-k)(k+1)
    Cdn2 = (T + T3) * (T - T3 + 1)                       # |<T3-1|T-|T3>|^2 = k (n-k)
    Dc = np.diag(ALPHA_EM_0 * Q**2)                      # photon Coulomb (attractive, particle-antiparticle)
    Dz = np.diag(gZ**2 / (4 * np.pi))                    # Z Yukawa, diagonal
    W = np.zeros((n, n))
    for k in range(n - 1):
        W[k, k + 1] = W[k + 1, k] = (ALPHA2 / 2) * Cup2[k]   # single W exchange psi^(k) psibar^(k) <-> psi^(k+1) psibar^(k+1)
    thr = 2 * np.array([delta_m_gev(q, Y) for q in Q])
    a = {
        "WW": (G2**2 / 2) * (Cdn2 + Cup2),
        "ZZ": 2 * gZ**2,
        "gg": 2 * E_MZ**2 * Q**2,
        "Zg": 2 * E_MZ * Q * gZ,
    }
    ident = {"WW": 1.0, "ZZ": 0.5, "gg": 0.5, "Zg": 1.0}
    return dict(name=f"({n},{Y:g})", n=n, Y=Y, Q=Q, T3=T3, gZ=gZ, Dc=Dc, Dz=Dz, W=W, thr=thr, a=a,
                a_norm=a, ident=ident, real=False, tree_a0=None)


def real_multiplet(n):
    """Real (Y=0, Majorana) n-plet, n odd: channels Q = 0..T  {chi0 chi0 (identical), psi^(Q) psibar^(Q)}.
    Identical-particle normalisation of |chi0 chi0>: potential column x sqrt2, annihilation a_0 -> a_0/sqrt2
    (Hisano-Matsumoto-Nojiri-Saito 2005 convention; reproduces their wino matrices)."""
    T = (n - 1) // 2
    Q = np.arange(T + 1)
    gZ = G2 * CW * Q
    Cup2 = (T - Q) * (T + Q + 1.0)
    Cdn2 = (T + Q) * (T - Q + 1.0)
    Dc = np.diag(ALPHA_EM_0 * Q**2)
    Dz = np.diag(gZ**2 / (4 * np.pi))
    W = np.zeros((T + 1, T + 1))
    for k in range(T):
        W[k, k + 1] = W[k + 1, k] = (ALPHA2 / 2) * Cup2[k] * (np.sqrt(2) if k == 0 else 1.0)
    thr = 2 * np.array([delta_m_gev(q, 0.0) for q in Q])
    a = {"WW": (G2**2 / 2) * (Cdn2 + Cup2), "ZZ": 2 * gZ**2, "gg": 2 * E_MZ**2 * Q**2, "Zg": 2 * E_MZ * Q * gZ}
    a_norm = {key: v.copy() for key, v in a.items()}
    for key in a_norm:
        a_norm[key][0] /= np.sqrt(2)
    ident = {"WW": 1.0, "ZZ": 0.5, "gg": 0.5, "Zg": 1.0}
    return dict(name=f"({n},0)", n=n, Y=0.0, Q=Q, T3=Q, gZ=gZ, Dc=Dc, Dz=Dz, W=W, thr=thr, a=a,
                a_norm=a_norm, ident=ident, real=True)


def tree_sigmav(sys_, m):
    """Tree-level s-wave sigma v (cm^3/s) of the neutral Majorana state into each final state."""
    out = {}
    for X, av in sys_["a"].items():
        mV = {"WW": (M_W, M_W), "ZZ": (M_Z, M_Z), "gg": (0, 0), "Zg": (M_Z, 0)}[X]
        beta = np.sqrt(max(0.0, 1 - ((mV[0] + mV[1]) / (2 * m))**2))
        out[X] = av[0]**2 / (32 * np.pi * m**2) * sys_["ident"][X] * beta * GEV2_TO_CM3S
    out["tot"] = out["WW"] + out["ZZ"]
    return out


# ----------------------------------------------------------------------------------------------
# Multi-channel Sommerfeld solver
# ----------------------------------------------------------------------------------------------
def sommerfeld(sys_, m, v_rel, rmax_over_mW=12.0, r0_over_m=0.02, rtol=1e-10, return_s=False):
    """Solve  u'' = m [ V(r) + diag(thr) - E ] u  (reduced mass m/2; E = m v_rel^2 / 4) for the 1S0 channels.
    Inward integration from r_max of (n+1) basis solutions (closed: decaying; open j>0: outgoing; open 0: e^{+-ikr});
    regularity u(r0)=0 and unit incoming flux in channel 0 fix the physical solution; s_k = u_k'(r0).
    Returns dict of enhancement factors S_X = |sum_k s_k a_k^X|^2 / |a_0^X|^2 (a_0^X = 0 -> absolute |sum|^2)."""
    n = len(sys_["thr"])
    thr = sys_["thr"]
    E = m * v_rel**2 / 4
    Dc, Dz, W = sys_["Dc"], sys_["Dz"], sys_["W"]
    diag_term = m * (thr - E)
    rmax = rmax_over_mW / M_W
    r0 = r0_over_m / m
    open_ch = [j for j in range(n) if E > thr[j]]
    closed_ch = [j for j in range(n) if E <= thr[j]]
    kk = np.sqrt(np.abs(m * (E - thr)))          # k_j (open) or kappa_j (closed)
    nb = n + 1
    Y0 = np.zeros((2 * n, nb), dtype=complex)
    cols = []
    # closed channels: decaying
    for j in closed_ch:
        c = len(cols)
        Y0[j, c] = 1.0
        Y0[n + j, c] = -kk[j]
        cols.append(("closed", j))
    # open channels j>0: outgoing only
    for j in open_ch:
        if j == 0:
            continue
        c = len(cols)
        Y0[j, c] = 1.0
        Y0[n + j, c] = 1j * kk[j]
        cols.append(("out", j))
    # open channel 0: e^{+ikr}, e^{-ikr}
    k0 = kk[0]
    c_out = len(cols); Y0[0, c_out] = np.exp(1j * k0 * rmax); Y0[n, c_out] = 1j * k0 * np.exp(1j * k0 * rmax); cols.append(("out", 0))
    c_in = len(cols); Y0[0, c_in] = np.exp(-1j * k0 * rmax); Y0[n, c_in] = -1j * k0 * np.exp(-1j * k0 * rmax); cols.append(("in", 0))
    assert len(cols) == nb
    mDc, mDz, mW = m * Dc, m * Dz, m * W

    def rhs(r, y):
        Yv = y.reshape(2 * n, nb)
        M = -(mDc + mDz * np.exp(-M_Z * r) + mW * np.exp(-M_W * r)) / r
        M[np.diag_indices(n)] += diag_term
        out = np.empty_like(Yv)
        out[:n] = Yv[n:]
        out[n:] = M @ Yv[:n]
        return out.ravel()

    sol = solve_ivp(rhs, (rmax, r0), Y0.ravel(), method="DOP853", rtol=rtol, atol=1e-14)
    if not sol.success:
        raise RuntimeError(sol.message)
    Yend = sol.y[:, -1].reshape(2 * n, nb)
    U, dU = Yend[:n], Yend[n:]
    scale = np.max(np.abs(U), axis=0)
    scale[scale == 0] = 1.0
    A = np.zeros((nb, nb), dtype=complex)
    A[:n, :] = U / scale
    A[n, c_in] = 1.0 / scale[c_in]
    b = np.zeros(nb, dtype=complex)
    b[n] = 1j / (2 * k0)          # incoming amplitude of sin(kr)/k
    x = np.linalg.solve(A, b) / scale
    s = dU @ x                    # u_k'(r0): free solution gives s = (1,0,...)
    res = {}
    for X, av in sys_["a_norm"].items():
        amp = np.sum(s * av)
        res["S_" + X] = abs(amp)**2 / av[0]**2 if av[0] != 0 else abs(amp)**2
    tree = sys_["a"]
    # total (WW+ZZ) enhancement weighted by tree cross-sections (identical-boson factor included)
    num = res["S_WW"] * tree["WW"][0]**2 + res["S_ZZ"] * 0.5 * tree["ZZ"][0]**2
    den = tree["WW"][0]**2 + 0.5 * tree["ZZ"][0]**2
    res["S_tot"] = num / den
    # absolute line cross-section (cm^3/s): gamma gamma + 1/2 Z gamma, from the charged components
    line = (res["S_gg"] * 0.5 + 0.5 * res["S_Zg"]) / (32 * np.pi * m**2) * GEV2_TO_CM3S   # a_0 = 0 -> S_ holds |amp|^2
    res["sigv_line"] = line
    res["sigv_gg"] = res["S_gg"] * 0.5 / (32 * np.pi * m**2) * GEV2_TO_CM3S
    res["nfev"] = sol.nfev
    if return_s:
        res["s"] = s
    return res


# ----------------------------------------------------------------------------------------------
# 1. Validation of the solver
# ----------------------------------------------------------------------------------------------
log("\n--- 1. Solver validation ---")
VAL = {}


def single_channel(alpha, mphi, coulomb=False):
    """One-channel toy system: V = -alpha e^{-mphi r}/r (Yukawa with mass mphi placed in the m_W slot) or Coulomb."""
    s = dict(name="toy", n=1, Q=np.array([0]), thr=np.array([0.0]), Dc=np.array([[alpha if coulomb else 0.0]]),
             Dz=np.zeros((1, 1)), W=np.array([[0.0 if coulomb else alpha]]), a={"WW": np.array([1.0])},
             a_norm={"WW": np.array([1.0])}, ident={"WW": 1.0}, real=False)
    s["a"].update({"ZZ": np.array([0.0]), "gg": np.array([0.0]), "Zg": np.array([0.0])})
    s["a_norm"] = s["a"]
    s["ident"].update({"ZZ": .5, "gg": .5, "Zg": 1.})
    return s


# (a) Coulomb: exact S = 2 pi eta/(1-e^{-2 pi eta}), eta = alpha / v_rel   (r_max large: Coulomb is long-range)
m_t = 1000.0
for alpha, v in [(0.01, 0.1), (0.03, 0.05), (0.03, 0.02)]:
    s_c = single_channel(alpha, 0.0, coulomb=True)
    exact = 2 * np.pi * alpha / v / (1 - np.exp(-2 * np.pi * alpha / v))
    num = sommerfeld(s_c, m_t, v, rmax_over_mW=12.0 * 40)["S_WW"]   # r_max = 480/m_W: k r_max >> 1
    log(f"Coulomb alpha={alpha} v={v}: numerical S = {num:.4f}, exact {exact:.4f}, ratio {num/exact:.4f}")
    VAL[f"coulomb_a{alpha}_v{v}"] = dict(numerical=num, exact=exact)

# (b) Yukawa single channel vs the Hulthen closed form (Cassel 2010; Slatyer 2010) -- approximate (10-20 %)
def hulthen(alpha, mphi, m, v_rel):
    ev = (v_rel / 2) / alpha
    ephi = mphi / (alpha * m)
    estar = np.pi**2 * ephi / 6
    x = 2 * np.pi * ev / estar
    arg = 1 / estar - ev**2 / estar**2
    cos_term = np.cos(2 * np.pi * np.sqrt(arg)) if arg >= 0 else np.cosh(2 * np.pi * np.sqrt(-arg))
    return (np.pi / ev) * np.sinh(x) / (np.cosh(x) - cos_term)


for alpha, mY, v in [(0.03, M_W, 1e-3), (0.05, M_W, 1e-3), (0.03, M_W, 1e-2)]:
    s_y = single_channel(alpha, M_W)
    num = sommerfeld(s_y, 2000.0, v)["S_WW"]
    hul = hulthen(alpha, mY, 2000.0, v)
    log(f"Yukawa alpha={alpha}, m=2 TeV, v={v}: numerical S = {num:.4f}, Hulthen {hul:.4f}, ratio {num/hul:.3f}")
    VAL[f"yukawa_a{alpha}_v{v}"] = dict(numerical=num, hulthen=hul)

# (c) r0 and r_max convergence for the (5,2) quintuplet at 1 TeV (worst case: 5 channels, deep thresholds)
q52 = dirac_multiplet(5, 2.0)
ref = sommerfeld(q52, 1000.0, 1e-3)["S_tot"]
for kw in [dict(r0_over_m=0.005), dict(r0_over_m=0.08), dict(rmax_over_mW=18.0), dict(rmax_over_mW=8.0), dict(rtol=1e-8)]:
    alt = sommerfeld(q52, 1000.0, 1e-3, **kw)["S_tot"]
    log(f"(5,2) 1 TeV v=1e-3 S_tot = {ref:.5f}; with {kw}: {alt:.5f} (ratio {alt/ref:.5f})")
    VAL["conv_" + "_".join(f"{a}{b}" for a, b in kw.items())] = dict(ref=ref, alt=alt)

# (d) Wino (3,0): literature check -- first resonance recalled at 2.3-2.5 TeV (Hisano et al. 2005 and later)
wino = real_multiplet(3)
log(f"wino channels Q={wino['Q']}, thresholds 2 dm = {wino['thr']} GeV, V_off = {wino['W'][0,1]/ALPHA2:.4f} alpha_2 (Hisano: sqrt2 = 1.4142)")
log(f"wino Z-Yukawa strength charged = {wino['Dz'][1,1]/ALPHA2:.4f} alpha_2 (Hisano: c_W^2 = {CW**2:.4f})")
log(f"wino annihilation vector a_WW/g^2 (state-normalised) = {wino['a_norm']['WW']/G2**2}  (Hisano Gamma ~ [[2,sqrt2],[sqrt2,1]] -> sqrt2 : 1)")

# (e) Majorana-basis 3-channel doublet with finite delta (chi1chi1, chi2chi2, chi+chi-) vs Dirac 2-channel limit
def doublet_majorana(delta_gev):
    d = dirac_multiplet(2, 0.5)
    aZ = d["Dz"][0, 0]
    w = d["W"][0, 1] / np.sqrt(2)
    s = dict(name="(2,1/2) Majorana basis", n=3, Q=np.array([0, 0, 1]),
             Dc=np.diag([0.0, 0.0, d["Dc"][1, 1]]),
             Dz=np.array([[0.0, aZ, 0.0], [aZ, 0.0, 0.0], [0.0, 0.0, d["Dz"][1, 1]]]),
             W=np.array([[0.0, 0.0, w], [0.0, 0.0, w], [w, w, 0.0]]),
             thr=np.array([0.0, 2 * delta_gev, d["thr"][1]]), real=False, ident=d["ident"])
    a = {}
    for X, av in d["a"].items():
        a[X] = np.array([av[0] / np.sqrt(2), av[0] / np.sqrt(2), av[1]])
    s["a"] = a
    s["a_norm"] = a
    return s


dbl = dirac_multiplet(2, 0.5)
for mtest in [1000.0, 3000.0]:
    r_dirac = sommerfeld(dbl, mtest, 1e-3)
    r_maj0 = sommerfeld(doublet_majorana(0.0), mtest, 1e-3)
    r_maj = sommerfeld(doublet_majorana(0.366e-3), mtest, 1e-3)
    r_maj_v0 = sommerfeld(doublet_majorana(0.366e-3), mtest, 1e-6)
    r_dirac_v0 = sommerfeld(dbl, mtest, 1e-6)
    log(f"doublet m={mtest:.0f}: S_tot Dirac {r_dirac['S_tot']:.5f} | Majorana delta=0 {r_maj0['S_tot']:.5f} | "
        f"Majorana delta=366 keV {r_maj['S_tot']:.5f} (v=1e-3); v=1e-6: Dirac {r_dirac_v0['S_tot']:.5f}, Maj(366 keV) {r_maj_v0['S_tot']:.5f}")
    VAL[f"majorana_check_m{mtest:.0f}"] = dict(dirac=r_dirac["S_tot"], majorana_delta0=r_maj0["S_tot"],
                                               majorana_delta366=r_maj["S_tot"], dirac_v1e6=r_dirac_v0["S_tot"],
                                               majorana_delta366_v1e6=r_maj_v0["S_tot"])

# ----------------------------------------------------------------------------------------------
# 2. Multiplets, tree-level cross-sections
# ----------------------------------------------------------------------------------------------
log("\n--- 2. Tree-level cross-sections of the neutral Majorana state (delta -> 0) ---")
MULTS = {"(2,1/2)": dirac_multiplet(2, 0.5), "(3,1)": dirac_multiplet(3, 1.0),
         "(4,3/2)": dirac_multiplet(4, 1.5), "(5,2)": dirac_multiplet(5, 2.0)}
LABEL = {"(2,1/2)": "doublet Y=1/2 (Higgsino-like)", "(3,1)": "triplet Y=1", "(4,3/2)": "quadruplet Y=3/2", "(5,2)": "quintuplet Y=2"}
# thermal masses (recalled; doublet certain, others uncertain +-30 %, from P037 / MDM literature)
M_TH = {"(2,1/2)": 1.1, "(3,1)": 2.0, "(4,3/2)": 2.4, "(5,2)": 4.5}
# delta(N=1) [keV] from P037 (1 TeV) and P007/P037 (2 TeV, 4 TeV); interpolations flagged in details.md
DELTA_N1 = {("(2,1/2)", 1.0): 365.8, ("(2,1/2)", 1.1): 367.0, ("(2,1/2)", 2.0): 375.0,
            ("(3,1)", 1.0): 374.1, ("(3,1)", 2.0): 384.7,
            ("(4,3/2)", 1.0): 379.2, ("(4,3/2)", 2.0): 390.8, ("(4,3/2)", 2.4): 392.0,
            ("(5,2)", 1.0): 382.7, ("(5,2)", 2.0): 393.0, ("(5,2)", 4.5): 398.0}
BENCH = {"(2,1/2)": [1.0, 1.1, 2.0], "(3,1)": [1.0, 2.0], "(4,3/2)": [1.0, 2.0, 2.4], "(5,2)": [1.0, 2.0, 4.5]}

tree_rows = []
for key, S_ in MULTS.items():
    log(f"{key}: Q = {S_['Q']}, g_Z/(g/c_W) = {np.round(S_['gZ']/(G2/CW),4)}, Delta m(Q) [GeV] = {np.round(S_['thr']/2,4)}")
    log(f"   a_WW/g^2 = {np.round(S_['a']['WW']/G2**2,3)}, a_ZZ/g^2 = {np.round(S_['a']['ZZ']/G2**2,3)}, "
        f"a_gg/e^2 = {np.round(S_['a']['gg']/E_MZ**2,3)}, W_off/alpha2 = {np.round(np.diag(S_['W'],1)/ALPHA2,3)}, "
        f"alpha_Z(neutral) = {S_['Dz'][0,0]:.4f}")
    for mTeV in [1.0, 1.1, 2.0, 2.4, 4.5]:
        t = tree_sigmav(S_, 1000 * mTeV)
        tree_rows.append(dict(multiplet=key, mass_TeV=mTeV, sigv_WW=t["WW"], sigv_ZZ=t["ZZ"], sigv_tot=t["tot"],
                              sigv_gg_tree=t["gg"], sigv_Zg_tree=t["Zg"]))
        if mTeV in (1.0, M_TH[key]):
            log(f"   m = {mTeV} TeV: <sv>_WW = {t['WW']:.3e}, ZZ = {t['ZZ']:.3e}, total = {t['tot']:.3e} cm^3/s; "
                f"gg, Zg tree = {t['gg']:.1e}, {t['Zg']:.1e}")
import pandas as pd
pd.DataFrame(tree_rows).to_csv(os.path.join(OUT, "P084_tree_xsec.csv"), index=False)
# analytic formulae check
for key, S_ in MULTS.items():
    n = S_["n"]; Y = S_["Y"]
    ww_formula = np.pi * ALPHA2**2 * (n - 1)**2 / 8 / 1e6 * GEV2_TO_CM3S
    zz_formula = np.pi * ALPHA2**2 * Y**4 / CW**4 / 1e6 * GEV2_TO_CM3S
    t = tree_sigmav(S_, 1000.0)
    log(f"   formula check {key}: WW pi a2^2 (n-1)^2/(8 m^2) = {ww_formula:.3e} (code {t['WW']:.3e}, beta factor); "
        f"ZZ pi a2^2 Y^4/(c_W^4 m^2) = {zz_formula:.3e} (code {t['ZZ']:.3e})")

# ----------------------------------------------------------------------------------------------
# 3. Sommerfeld factors at the benchmark masses and velocities
# ----------------------------------------------------------------------------------------------
log("\n--- 3. Sommerfeld factors at benchmarks ---")
V_MW, V_DW, V_CMB = 1e-3, 3e-5, 1e-6
bench_rows = []
t0 = time.time()
for key, S_ in MULTS.items():
    for mTeV in BENCH[key]:
        m = 1000 * mTeV
        rMW = sommerfeld(S_, m, V_MW); rMWlo = sommerfeld(S_, m, 3e-4); rMWhi = sommerfeld(S_, m, 2e-3)
        rDW = sommerfeld(S_, m, V_DW); rDW2 = sommerfeld(S_, m, 1e-4)
        rCMB = sommerfeld(S_, m, V_CMB)
        t = tree_sigmav(S_, m)
        row = dict(multiplet=key, mass_TeV=mTeV, delta_N1_keV=DELTA_N1.get((key, mTeV), np.nan),
                   sigv_tree_WW=t["WW"], sigv_tree_ZZ=t["ZZ"], sigv_tree_tot=t["tot"],
                   S_tot_MW=rMW["S_tot"], S_WW_MW=rMW["S_WW"], S_ZZ_MW=rMW["S_ZZ"],
                   S_tot_MW_v3em4=rMWlo["S_tot"], S_tot_MW_v2em3=rMWhi["S_tot"],
                   S_tot_dwarf=rDW["S_tot"], S_tot_dwarf_v1em4=rDW2["S_tot"], S_tot_CMB=rCMB["S_tot"],
                   sigv_MW_tot=t["tot"] * rMW["S_tot"], sigv_dwarf_tot=t["tot"] * rDW["S_tot"], sigv_CMB_tot=t["tot"] * rCMB["S_tot"],
                   sigv_line_MW=rMW["sigv_line"], sigv_gg_MW=rMW["sigv_gg"], sigv_line_CMB=rCMB["sigv_line"])
        bench_rows.append(row)
        log(f"{key} m={mTeV} TeV: S_tot(MW 1e-3) = {rMW['S_tot']:.3f} [v=3e-4: {rMWlo['S_tot']:.3f}, 2e-3: {rMWhi['S_tot']:.3f}], "
            f"S_WW = {rMW['S_WW']:.3f}, S_ZZ = {rMW['S_ZZ']:.3f}; S(dwarf 3e-5) = {rDW['S_tot']:.3f}; S(CMB 1e-6) = {rCMB['S_tot']:.3f}; "
            f"<sv>_MW = {t['tot']*rMW['S_tot']:.3e}; line(gg+Zg/2) = {rMW['sigv_line']:.3e} cm^3/s  [{rMW['nfev']} fev]")
log(f"(benchmarks took {time.time()-t0:.1f} s)")
bench = pd.DataFrame(bench_rows)
bench.to_csv(os.path.join(OUT, "P084_sommerfeld_benchmarks.csv"), index=False)

# S(v) curves at 1 TeV and the thermal mass
log("\n--- 3b. S(v) curves ---")
vgrid = np.logspace(-6, np.log10(0.3), 22)
sv_rows = []
for key, S_ in MULTS.items():
    for mTeV in sorted(set([1.0, M_TH[key]])):
        for v in vgrid:
            r = sommerfeld(S_, 1000 * mTeV, v)
            sv_rows.append(dict(multiplet=key, mass_TeV=mTeV, v_rel=v, S_tot=r["S_tot"], S_WW=r["S_WW"], S_ZZ=r["S_ZZ"], sigv_line=r["sigv_line"]))
svdf = pd.DataFrame(sv_rows)
svdf.to_csv(os.path.join(OUT, "P084_S_vs_v.csv"), index=False)
for key in MULTS:
    d = svdf[(svdf.multiplet == key) & (svdf.mass_TeV == 1.0)]
    log(f"{key} 1 TeV: S_tot at v = 1e-6 / 1e-4 / 1e-3 / 1e-2 / 0.1 / 0.3 : " +
        " / ".join(f"{np.interp(np.log10(vv), np.log10(d.v_rel), d.S_tot):.3f}" for vv in [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.3]) +
        f"; v_thr(charged) = {np.sqrt(4*MULTS[key]['thr'][1]/1000):.3f}")

# ----------------------------------------------------------------------------------------------
# 4. Mass scan: resonance structure
# ----------------------------------------------------------------------------------------------
log("\n--- 4. Mass scan 0.5-12 TeV ---")
mgrid = np.logspace(np.log10(500), np.log10(12000), 48)
scan_rows = []
t0 = time.time()
SCAN_SYS = dict(MULTS); SCAN_SYS["(3,0)"] = wino; SCAN_SYS["(5,0)"] = real_multiplet(5)
for key, S_ in SCAN_SYS.items():
    for m in mgrid:
        rMW = sommerfeld(S_, m, V_MW)
        rS = sommerfeld(S_, m, V_CMB)
        t = tree_sigmav(S_, m)
        scan_rows.append(dict(multiplet=key, mass_GeV=m, S_tot_MW=rMW["S_tot"], S_tot_sat=rS["S_tot"],
                              S_WW_MW=rMW["S_WW"], sigv_tree_tot=t["tot"], sigv_MW_tot=t["tot"] * rMW["S_tot"],
                              sigv_sat_tot=t["tot"] * rS["S_tot"], sigv_line_MW=rMW["sigv_line"], sigv_line_sat=rS["sigv_line"]))
    log(f"  {key} scanned ({time.time()-t0:.0f} s elapsed)")
scan = pd.DataFrame(scan_rows)
scan.to_csv(os.path.join(OUT, "P084_mass_scan.csv"), index=False)
RES = {}
for key in SCAN_SYS:
    d = scan[scan.multiplet == key].reset_index(drop=True)
    Ssat = d.S_tot_sat.values
    peaks = [i for i in range(1, len(d) - 1) if Ssat[i] > Ssat[i - 1] and Ssat[i] > Ssat[i + 1]]
    RES[key] = [(float(d.mass_GeV[i]), float(Ssat[i]), float(d.S_tot_MW[i])) for i in peaks]
    log(f"{key}: saturated-S local maxima (grid spacing 6.6 %): " +
        "; ".join(f"m = {mm/1000:.2f} TeV (S_sat = {ss:.0f}, S_MW = {sm:.0f})" for mm, ss, sm in RES[key]))
    log(f"     S_MW at 1/2/3 TeV: {np.interp(1000, d.mass_GeV, d.S_tot_MW):.2f} / {np.interp(2000, d.mass_GeV, d.S_tot_MW):.2f} / {np.interp(3000, d.mass_GeV, d.S_tot_MW):.2f}; "
        f"S_sat: {np.interp(1000, d.mass_GeV, Ssat):.2f} / {np.interp(2000, d.mass_GeV, Ssat):.2f} / {np.interp(3000, d.mass_GeV, Ssat):.2f}")
# wino numbers for the literature check
dw = scan[scan.multiplet == "(3,0)"]
log(f"wino 3 TeV: <sv>_tree = {np.interp(3000, dw.mass_GeV, dw.sigv_tree_tot):.2e}, MW-enhanced = {np.interp(3000, dw.mass_GeV, dw.sigv_MW_tot):.2e}, "
    f"line(gg+Zg/2) = {np.interp(3000, dw.mass_GeV, dw.sigv_line_MW):.2e} cm^3/s")

# ----------------------------------------------------------------------------------------------
# 5. Limits (all recalled), thermal fractions and verdict table
# ----------------------------------------------------------------------------------------------
log("\n--- 5. Limits and verdicts ---")
# anchors at 1 TeV for WW-like final states [cm^3/s], scaling exponent with mass, (low, high) band, reliability
LIMITS = {
    "Fermi_dSph": dict(anchor=1e-25, slope=1.0, band=(5e-26, 2e-25), rel="likely (x2)", S="dwarf"),
    "HESS_GC":    dict(anchor=2e-26, slope=0.5, band=(5e-27, 4e-26), rel="uncertain (Einasto; cored x10 weaker)", S="MW"),
    "AMS_pbar":   dict(anchor=3e-26, slope=1.0, band=(1e-26, 1e-25), rel="uncertain (propagation x3-10)", S="MW"),
    "Planck_CMB": dict(anchor=3.5e-28 * 1000 / 0.35, slope=1.0, band=(3.5e-28 * 1000 / 0.35, 3.2e-28 * 1000 / 0.20), rel="likely (p_ann 3.2-3.5e-28, f_eff 0.2-0.35)", S="CMB"),
}
LINE_LIMIT = dict(anchor=2e-28, slope=1.0, band=(1e-28, 4e-28), rel="uncertain (H.E.S.S. Einasto line)", S="MW")


def limit_at(L, m_gev):
    f = (m_gev / 1000.0)**L["slope"]
    return L["anchor"] * f, L["band"][0] * f, L["band"][1] * f


verdict_rows = []
for _, b in bench.iterrows():
    key, mTeV = b.multiplet, b.mass_TeV
    m = 1000 * mTeV
    f_th = min(1.0, (mTeV / M_TH[key])**2)
    if key == "(2,1/2)" and mTeV == 1.0:
        f_th = 0.80   # corpus: P007 0.83, P025 0.73 -> 0.8 (assignment)
    dd_shift = -10.0 * np.log(1 / f_th) / np.log(5.0) if f_th < 1 else 0.0   # N falls x5 per 10 keV near delta_max (P007/P037)
    row = dict(multiplet=key, mass_TeV=mTeV, delta_N1_keV=b.delta_N1_keV, f_thermal=f_th, delta_N1_subcomponent_keV=b.delta_N1_keV + dd_shift,
               sigv_tree_tot=b.sigv_tree_tot, S_MW=b.S_tot_MW, S_dwarf=b.S_tot_dwarf, S_CMB=b.S_tot_CMB,
               sigv_MW=b.sigv_MW_tot, sigv_line_MW=b.sigv_line_MW)
    worst = 0.0; worst_name = ""; worst_lo = 0.0
    for name, L in LIMITS.items():
        anchor, lo, hi = limit_at(L, m)
        sv = {"MW": b.sigv_MW_tot, "dwarf": b.sigv_dwarf_tot, "CMB": b.sigv_CMB_tot}[L["S"]]
        ratio = sv / anchor
        row[f"ratio_{name}"] = ratio
        row[f"ratio_{name}_range"] = f"{sv/hi:.2g}-{sv/lo:.2g}"
        row[f"ratio_{name}_thermal"] = ratio * f_th**2
        if ratio > worst:
            worst, worst_name, worst_lo = ratio, name, sv / hi
    anchor, lo, hi = limit_at(LINE_LIMIT, m)
    row["ratio_line"] = b.sigv_line_MW / anchor
    row["ratio_line_range"] = f"{b.sigv_line_MW/hi:.2g}-{b.sigv_line_MW/lo:.2g}"
    row["ratio_line_thermal"] = row["ratio_line"] * f_th**2
    if row["ratio_line"] > worst:
        worst, worst_name, worst_lo = row["ratio_line"], "line", b.sigv_line_MW / hi
    row["worst_probe"] = worst_name
    row["worst_ratio"] = worst
    if worst_lo > 1:
        v_full = "EXCLUDED (beyond band)"
    elif worst > 1:
        v_full = "excluded at anchor (marginal within band)"
    elif worst > 0.5:
        v_full = "allowed, marginal (<x2 headroom)"
    else:
        v_full = "allowed"
    row["verdict_full_density"] = v_full
    wt = worst * f_th**2
    row["worst_ratio_thermal"] = wt
    row["verdict_thermal_fraction"] = ("excluded" if wt > 1 else ("marginal" if wt > 0.5 else "allowed")) + f" (f={f_th:.2f})"
    # dilution needed to evade the worst anchor limit (indirect ~ f^2) and the LZ-implied delta(N=1) at that fraction (rate ~ f)
    f_max = min(1.0, 1.0 / np.sqrt(worst)) if worst > 0 else 1.0
    row["f_max_evade_anchor"] = f_max
    row["f_max_evade_band_high"] = min(1.0, 1.0 / np.sqrt(worst_lo)) if worst_lo > 0 else 1.0
    row["delta_N1_at_f_max_keV"] = b.delta_N1_keV - 10.0 * np.log(1 / f_max) / np.log(5.0)
    verdict_rows.append(row)
    log(f"{key} {mTeV} TeV (delta={b.delta_N1_keV:.0f} keV): <sv>_MW = {b.sigv_MW_tot:.2e} (S_MW {b.S_tot_MW:.2f}, S_dw {b.S_tot_dwarf:.2f}, S_CMB {b.S_tot_CMB:.2f}); "
        f"ratios Fermi {row['ratio_Fermi_dSph']:.2f} [{row['ratio_Fermi_dSph_range']}], HESS {row['ratio_HESS_GC']:.2f} [{row['ratio_HESS_GC_range']}], "
        f"AMS {row['ratio_AMS_pbar']:.2f} [{row['ratio_AMS_pbar_range']}], Planck {row['ratio_Planck_CMB']:.3f} [{row['ratio_Planck_CMB_range']}], "
        f"line {row['ratio_line']:.2f} [{row['ratio_line_range']}] -> {v_full}; thermal f={f_th:.2f}: worst x{wt:.2g} -> {row['verdict_thermal_fraction']}; "
        f"delta(N=1) -> {row['delta_N1_subcomponent_keV']:.0f} keV; evasion needs f <= {row['f_max_evade_anchor']:.3f} "
        f"(<= {row['f_max_evade_band_high']:.3f} at the loose band edge) -> delta(N=1) = {row['delta_N1_at_f_max_keV']:.0f} keV")
verd = pd.DataFrame(verdict_rows)
# S along the +-30 % thermal-mass band of the (3,1) triplet and the doublet
for key, ms in [("(3,1)", [1.4, 1.7, 2.0, 2.3, 2.6]), ("(2,1/2)", [1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0])]:
    d = scan[scan.multiplet == key]
    log(f"{key} S_MW along mass band: " + ", ".join(f"{mt} TeV: {np.exp(np.interp(np.log(1000*mt), np.log(d.mass_GeV), np.log(d.S_tot_MW))):.1f}" for mt in ms))
verd.to_csv(os.path.join(OUT, "P084_verdict_table.csv"), index=False)

# Mass at which each multiplet (full density, MW S) crosses the anchor limits, and resonance proximity
log("\n--- 5b. Exclusion masses (full local density, anchor limits) ---")
EXCL = {}
for key in MULTS:
    d = scan[scan.multiplet == key].reset_index(drop=True)
    out = {}
    for name in ["Fermi_dSph", "HESS_GC", "Planck_CMB"]:
        L = LIMITS[name]
        sv = d.sigv_sat_tot if L["S"] in ("CMB", "dwarf") else d.sigv_MW_tot
        lim = np.array([limit_at(L, mm)[0] for mm in d.mass_GeV])
        ratio = sv.values / lim
        excluded = d.mass_GeV.values[ratio > 1]
        out[name] = dict(min_allowed_mass_TeV=(float(excluded.max()) / 1000 if len(excluded) else None), ratio_1TeV=float(np.interp(1000, d.mass_GeV, ratio)),
                         n_excluded_points=int((ratio > 1).sum()), allowed_islands=int(((ratio[1:] > 1) != (ratio[:-1] > 1)).sum()))
    EXCL[key] = out
    log(f"{key}: " + "; ".join(f"{k}: ratio(1 TeV) {v['ratio_1TeV']:.2f}, largest excluded grid mass {v['min_allowed_mass_TeV']} TeV, {v['n_excluded_points']}/48 grid points excluded, {v['allowed_islands']} band crossings" for k, v in out.items()))

# ----------------------------------------------------------------------------------------------
# 6. Figures
# ----------------------------------------------------------------------------------------------
COL = {"(2,1/2)": "#2a78d6", "(3,1)": "#eb6834", "(4,3/2)": "#1baf7a", "(5,2)": "#eda100"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.edgecolor": "#c3c2b7", "figure.facecolor": "white"})

# Fig 1: S vs mass
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
for ax, col, ttl in [(axes[0], "S_tot_MW", "Milky-Way halo, v_rel = 10$^{-3}$ c"), (axes[1], "S_tot_sat", "saturated (dwarfs, CMB), v_rel = 10$^{-6}$ c")]:
    for key in MULTS:
        d = scan[scan.multiplet == key]
        ax.plot(d.mass_GeV / 1000, d[col], color=COL[key], lw=2, label=f"{key} {LABEL[key].split(' (')[0]}")
        ax.plot([1.0], [np.interp(1000, d.mass_GeV, d[col])], "o", color=COL[key], ms=6, mec="white")
        ax.plot([M_TH[key]], [np.interp(1000 * M_TH[key], d.mass_GeV, d[col])], "s", color=COL[key], ms=6, mec="white")
    d = scan[scan.multiplet == "(3,0)"]
    ax.plot(d.mass_GeV / 1000, d[col], color="#9a9891", lw=1.2, ls="--", label="(3,0) wino (solver check)")
    ax.axvspan(2.3, 2.5, color="#9a9891", alpha=0.18, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("multiplet mass [TeV]"); ax.set_title(ttl, fontsize=9, loc="left")
    ax.set_xticks([0.5, 1, 2, 5, 10]); ax.set_xticklabels(["0.5", "1", "2", "5", "10"])
axes[0].set_ylabel("Sommerfeld factor S (WW+ZZ)")
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
axes[1].text(2.4, axes[1].get_ylim()[1] * 0.5, "recalled wino\nresonance 2.3-2.5 TeV", fontsize=7, color="#52514e", ha="center", va="top")
axes[0].text(0.52, axes[0].get_ylim()[0] * 1.3, "circles: 1 TeV LZ fit (P037); squares: recalled thermal mass", fontsize=7, color="#52514e")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "P084_fig1_S_vs_mass.png"), dpi=170)
plt.close(fig)

# Fig 2: enhanced <sigma v> vs mass with limit bands
fig, ax = plt.subplots(figsize=(6.6, 5.0))
mm = scan[scan.multiplet == "(2,1/2)"].mass_GeV.values
for name, L, colr, lab, ytxt in [("Fermi_dSph", LIMITS["Fermi_dSph"], "#e6e5e1", "Fermi-LAT dSph (WW, recalled)", 1.0),
                                 ("HESS_GC", LIMITS["HESS_GC"], "#d6d5cf", "H.E.S.S. GC Einasto (WW, recalled)", 1.0),
                                 ("Planck_CMB", LIMITS["Planck_CMB"], "#e6e5e1", "Planck CMB, f_eff = 0.2-0.35 (recalled)", 1.0)]:
    lo = np.array([limit_at(L, x)[1] for x in mm]); hi = np.array([limit_at(L, x)[2] for x in mm])
    ax.fill_between(mm / 1000, lo, hi, color=colr, alpha=0.9, lw=0)
    ax.text(0.52, np.sqrt(lo[0] * hi[0]), lab, fontsize=6.8, color="#52514e", va="center")
for key in MULTS:
    d = scan[scan.multiplet == key]
    ax.plot(d.mass_GeV / 1000, d.sigv_MW_tot, color=COL[key], lw=2, label=f"{key} {LABEL[key].split(' (')[0]}: MW-enhanced, full local density")
    f2 = np.minimum(1.0, (d.mass_GeV / 1000 / M_TH[key])**2)**2
    ax.plot(d.mass_GeV / 1000, d.sigv_MW_tot * f2, color=COL[key], lw=1.2, ls=":")
    ax.plot([1.0], [np.interp(1000, d.mass_GeV, d.sigv_MW_tot)], "o", color=COL[key], ms=6, mec="white")
ax.plot([], [], color="#52514e", ls=":", lw=1.2, label=r"dotted: same $\times f_{\rm thermal}^2$ (thermal sub-component below M$_{th}$)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.5, 12); ax.set_ylim(1e-27, 1e-21)
ax.set_xticks([0.5, 1, 2, 5, 10]); ax.set_xticklabels(["0.5", "1", "2", "5", "10"])
ax.set_xlabel("multiplet mass [TeV]"); ax.set_ylabel(r"$\langle\sigma v\rangle$(WW+ZZ) today [cm$^3$ s$^{-1}$]")
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
ax.text(1.0, 1.4e-27, "circles: 1 TeV LZ fits (P037)", fontsize=7, color="#52514e", ha="center")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "P084_fig2_sigv_vs_limits.png"), dpi=170)
plt.close(fig)

# Fig 3: S(v) at 1 TeV
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for key in MULTS:
    d = svdf[(svdf.multiplet == key) & (svdf.mass_TeV == 1.0)]
    ax.plot(d.v_rel, d.S_tot, color=COL[key], lw=2, label=f"{key}")
    ax.axvline(np.sqrt(4 * MULTS[key]["thr"][1] / 1000), color=COL[key], lw=0.8, ls=":")
for v, lab in [(1e-3, "MW"), (3e-5, "dwarfs"), (1e-6, "CMB")]:
    ax.axvline(v, color="#c3c2b7", lw=0.8); ax.text(v, ax.get_ylim()[1] if False else 1.02, lab, fontsize=7, color="#52514e", ha="center", transform=ax.get_xaxis_transform())
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("relative velocity v_rel / c"); ax.set_ylabel("S (WW+ZZ), m = 1 TeV")
ax.legend(frameon=False, fontsize=7.5, loc="center left")
ax.text(0.06, 0.9, "dotted: charged-pair threshold\n$v_{thr} = (8\\Delta m_1/m)^{1/2}$", fontsize=7, color="#52514e", transform=ax.transAxes)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "P084_fig3_S_vs_v.png"), dpi=170)
plt.close(fig)

# ----------------------------------------------------------------------------------------------
# 7. Result JSON
# ----------------------------------------------------------------------------------------------
results = dict(constants=dict(alpha2=ALPHA2, sw2=SW2, alpha0=ALPHA_EM_0, alpha_mZ=ALPHA_EM_MZ, mW=M_W, mZ=M_Z, gev2_to_cm3s=GEV2_TO_CM3S),
               validation=VAL, thermal_masses_TeV=M_TH, limits=LIMITS | {"HESS_line": LINE_LIMIT},
               splittings_GeV={k: list(np.round(S_["thr"] / 2, 4)) for k, S_ in SCAN_SYS.items()},
               resonances=RES, exclusion_scan=EXCL,
               benchmarks=bench.to_dict(orient="records"), verdicts=verd.to_dict(orient="records"))
with open(os.path.join(OUT, "P084_results.json"), "w") as f:
    json.dump(results, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
log("\nwrote", os.listdir(OUT))
log("=== done", time.strftime("%H:%M:%S"))
