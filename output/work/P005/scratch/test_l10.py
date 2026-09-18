import sys, time, numpy as np
sys.path.insert(0, 'output/code')
from common import lzcommon as lz

mN = lz.M_NUCLEON_GEV
mM = mN  # cancels in the shape

halo = lz.wd_halo()
print("halo ok", len(halo[0]))

O1 = lz.wd_hamiltonian('o1s', {1: (1.0, 0.0)})
t = time.time()
E = np.array([60, 100, 150, 200, 250, 300])
r = lz.wd_rate(O1, 1000., E, halo=halo, delta_kev=300.)
print("O1s d=300 (1/m_v^2 units -> scale (1/246.2^2)^2)", r * (1/246.2**2)**2, time.time()-t, "s")

# L10 reduction, Anand et al. Table 1 (recalled, likely). isoscalar: cn=cp -> WimPyDD c0=2cp, c1=0
def mk_L10(sign46):
    def c1(q, mchi):
        return [q**2/(2*mchi*mM), 0.]
    def c5(q):
        return [2*mN/mM + 0*q, 0.]
    def c4(q):
        return [sign46*(-2.)*(mN/mM)*q**2/mN**2, 0.]
    def c6(q):
        return [sign46*2.*mN/mM + 0*q, 0.]
    WD = lz.wd()
    return WD.eft_hamiltonian('L10_%d' % sign46, {(1, 'q2'): c1, 5: c5, (4, 'q2'): c4, 6: c6})

Eg = np.array([5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 125, 150, 175, 200, 210, 225, 250, 270, 300, 350], float)
for s in (+1, -1):
    H = mk_L10(s)
    t = time.time()
    r = lz.wd_rate(H, 1000., Eg, halo=halo)
    print("sign", s, "time", time.time()-t)
    for e, v in zip(Eg, r):
        print(f"  {e:6.0f}  {v:.4e}")
# Individual pieces for insight
for nm, H in [("O5 only", lz.wd()._WD if False else None)]:
    pass
H5 = lz.wd().eft_hamiltonian('o5', {5: lambda q: [2*mN/mM + 0*q, 0.]})
H46 = lz.wd().eft_hamiltonian('o46', {(4, 'q2'): lambda q: [-2.*(mN/mM)*q**2/mN**2, 0.], 6: lambda q: [2.*mN/mM + 0*q, 0.]})
H4 = lz.wd().eft_hamiltonian('o4', {4: lambda q: [1. + 0*q, 0.]})
H6 = lz.wd().eft_hamiltonian('o6', {6: lambda q: [1. + 0*q, 0.]})
for nm, H in [("O5", H5), ("q2O4-O6", H46), ("O4", H4), ("O6", H6)]:
    r = lz.wd_rate(H, 1000., Eg, halo=halo)
    print(nm, " ".join(f"{v:.3e}" for v in r))
