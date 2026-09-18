"""P034 post-processing: derived numbers from output/work/P034/P034_results.json and degradations.csv.
Run from the simulation root: .venv/bin/python output/code/P034_postprocess.py"""
import json, numpy as np, pandas as pd
R = json.load(open('output/work/P034/P034_results.json'))
inp = R['inputs']; EXPO = inp['exposure_LZ']; TY = inp['tyr_per_year_LZ']
out = {}
# 1. Median Z against a flat alternative at LZ milestones (exposures incl. the first 2.84 t yr)
milestones = {'Sep2026_6.76+2.84': 6.76 + EXPO, 'end2027_10.47': 10.468, '1000_live_days_12.9': 1000 / 365.25 * 4.71,
              'end2028': EXPO + TY * (2029.0 - 2024.25)}
out['Z_at_milestones'] = {}
for tag, E in milestones.items():
    row = {}
    for d, T in R['tests_vs_flat'].items():
        sc = T['llr']['scan']; N = np.array(sc['N'], float); Z = np.array(sc['Z'], float)
        for mu, mtag in ((1.0, 'best'), (2.36, 'hi68'), (0.30, 'lo68')):
            n = E / EXPO * mu
            row[f'delta{int(float(d))}_{mtag}'] = dict(N_expected=round(n, 2), Z_median=round(float(np.interp(n, N, Z)), 2))
    out['Z_at_milestones'][tag] = dict(E_tyr=round(E, 2), **row)
# 2. ROI extension: no roll-off variant vs nominal -> more events per exposure but flatter PDF
deg = pd.read_csv('output/work/P034/degradations.csv')
eff = deg[deg.kind == 'efficiency']
out['roi_extension'] = []
for d in sorted(eff.delta.unique()):
    nom = eff[(eff.variant == 'nominal') & (eff.delta == d)].iloc[0]
    for v in ('none', 'edge290', 'edge250'):
        alt = eff[(eff.variant == v) & (eff.delta == d)].iloc[0]
        rate_ratio = alt.N_unit_per_2p84 / nom.N_unit_per_2p84
        E_nom = nom.N3_llr * EXPO; E_alt = alt.N3_llr * EXPO / rate_ratio
        out['roi_extension'].append(dict(delta=float(d), variant=v, rate_ratio=round(float(rate_ratio), 3), N3_nominal=round(float(nom.N3_llr), 1),
                                         N3_variant=round(float(alt.N3_llr), 1), E3_nominal_tyr=round(float(E_nom), 1), E3_variant_tyr_same_coupling=round(float(E_alt), 2),
                                         speedup=round(float(E_nom / E_alt), 2)))
# 3. halo shape: N3 ratios relative to std, and equivalent delta shift
halo = deg[deg.kind == 'halo']
out['halo_shape'] = []
for v in ('vesc528', 'vesc560', 'v0_220', 'v0_250'):
    for d in sorted(halo.delta.unique()):
        s = halo[(halo.variant == 'std') & (halo.delta == d)].iloc[0]; a = halo[(halo.variant == v) & (halo.delta == d)].iloc[0]
        out['halo_shape'].append(dict(variant=v, delta=float(d), N3_llr=round(float(a.N3_llr), 1), N3_over_std=round(float(a.N3_llr / s.N3_llr), 2),
                                      a1=round(float(a.a1), 3), log10_rate_over_std=round(float(np.log10(a.N_unit_per_2p84 / s.N_unit_per_2p84)), 2)))
# 4. gaps summary
gap = deg[deg.kind == 'gap']
out['gap_summary'] = [dict(variant=r.variant, delta=float(r.delta), N3_llr=round(float(r.N3_llr), 1), N3_cos=round(float(r.N3_cos), 1),
                           cos_bias_per_sqrtN=round(float(r.cos_null_bias_per_sqrtN), 3)) for r in gap.itertuples()]
# 5. exposures with band, compact table
exp = pd.read_csv('output/work/P034/required_exposure.csv')
tab = []
for d in sorted(exp.delta.unique()):
    row = dict(delta=float(d))
    for test in ('LLR', 'cosine', 'window May-Aug', 'window optimal'):
        for sig in (3, 5):
            b = exp[(exp.delta == d) & (exp.test == test) & (exp.sigma == sig) & (exp.rate == 'best')].iloc[0]
            lo = exp[(exp.delta == d) & (exp.test == test) & (exp.sigma == sig) & (exp.rate == 'hi68')].iloc[0]
            hi = exp[(exp.delta == d) & (exp.test == test) & (exp.sigma == sig) & (exp.rate == 'lo68')].iloc[0]
            row[f'{test}_{sig}s'] = dict(N=round(float(b.N), 1), E_tyr=round(float(b.E_tyr), 1), E_band=[round(float(lo.E_tyr), 1), round(float(hi.E_tyr), 1)],
                                         LZ_years=round(float(b.LZ_years), 1), year=round(float(b.year_reached), 1), det60t_years=round(float(b.det60t_years), 2))
    tab.append(row)
out['exposure_table'] = tab
json.dump(out, open('output/work/P034/derived_numbers.json', 'w'), indent=1)
for k in ('Z_at_milestones',):
    for tag, v in out[k].items():
        print(tag, v['E_tyr'], {kk: vv['Z_median'] for kk, vv in v.items() if kk != 'E_tyr' and kk.endswith('best')},
              {kk: vv['Z_median'] for kk, vv in v.items() if kk != 'E_tyr' and kk.endswith('hi68')})
for r in out['roi_extension']:
    if r['variant'] == 'none': print(r)
for r in out['halo_shape']:
    if r['delta'] in (350.0, 366.0): print(r)
for r in tab:
    print(r['delta'], 'LLR3', r['LLR_3s'], 'LLR5', r['LLR_5s']['E_tyr'], r['LLR_5s']['E_band'], 'win3', r['window May-Aug_3s']['E_tyr'], 'opt3', r['window optimal_3s']['E_tyr'])
