"""Exploratory, post hoc (not preregistered): is EDSP-tuned demonstrably equivalent to the best method on a graph?

For every graph of suites E1, E2 (+ B5), E2U, E3 (+ coauthor, B6, A18 phase 1) and A13: the best non-EDSP method by mean
over the splits it shares with EDSP-tuned; TOST with margin ±1 point of the headline metric (×100), two one-sided
Nadeau–Bengio corrected t tests at α = 0.05 (n_test / n_train: 0.5 Platonov graphs, 1 fraud graphs, 1/3 otherwise).
Label-noise (E6) records are excluded. No multiplicity correction. Output: results/aggregates/tost/TOST.md.
"""
from pathlib import Path
import json,glob,collections,sys
import numpy as np
from scipy import stats
SETS={'E1':['results/runs/e1'],'E2':['results/runs/e2','results/runs/b5'],'E2U':['results/runs/e2u'],
      'E3':['results/runs/e3','results/runs/e3-coauthor','results/runs/b6','results/runs/a18-phase1'],
      'A13':['results/runs/native-mixed']}
PLAT={'roman-empire','amazon-ratings','minesweeper','tolokers','questions','chameleon-filtered','squirrel-filtered'}
FRAUD={'amazon-fraud','yelpchi'}
def ratio(d): return 0.5 if d in PLAT else 1.0 if d in FRAUD else 1/3
out=[]
for name,dirs in SETS.items():
    by=collections.defaultdict(dict); suite_of={}
    for d in dirs:
        for f in glob.glob(d+'/**/*.jsonl',recursive=True):
            if 'ACQ' in f: continue
            for l in open(f):
                r=json.loads(l)
                if r.get('method') in ('NOT-RUN','DATASET-STATS') or 'split' not in r: continue
                if r.get('label_noise') or '~' in r.get('dataset',''): continue
                if name=='E3' and r.get('suite')!='E3': continue
                if name=='E3' and r['dataset']=='amazon-fraud' and str(r['split'])=='rand8': continue
                v=r['metrics'].get(r['headline_metric'])
                if v is None or not np.isfinite(v): continue
                by[(r['dataset'],r['method'])][str(r['split'])]=100*v
    for g in sorted({d for d,_ in by}):
        e=by.get((g,'EDSP-tuned'))
        if not e: continue
        others={m:v for (d,m),v in by.items() if d==g and not m.startswith('EDSP') and len(set(v)&set(e))>=3}
        if not others: continue
        best=max(others,key=lambda m:np.mean([others[m][s] for s in set(others[m])&set(e)]))
        c=sorted(set(e)&set(others[best])); dd=np.array([e[s]-others[best][s] for s in c]); k=len(dd)
        se=np.sqrt((1/k+ratio(g))*dd.var(ddof=1)); df=k-1
        if se==0: p=0.0 if abs(dd.mean())<1 else 1.0
        else: p=max(stats.t.sf((dd.mean()+1)/se,df),stats.t.cdf((dd.mean()-1)/se,df))
        out.append((name,g,best,k,dd.mean(),dd.std(ddof=1),p))
lines=['| suite | graph | best method | splits | Δ EDSP − best | sd | TOST p | equivalent within ±1 |','|---|---|---|---|---|---|---|---|']
for x in out:
    lines.append(f'| {x[0]} | {x[1]} | {x[2]} | {x[3]} | {x[4]:+.2f} | {x[5]:.2f} | {x[6]:.3g} | {"yes" if x[6]<0.05 else "no"} |')
Path('results/aggregates/tost').mkdir(parents=True,exist_ok=True)
Path('results/aggregates/tost/TOST.md').write_text('# EDSP-tuned vs the best method per graph: TOST (exploratory, post hoc)\n\n'+'\n'.join(lines)+'\n')
for x in out:
    print(f"{x[0]:4s} {x[1]:28s} best={x[2]:22s} n={x[3]:2d} Δ={x[4]:+6.2f} sd={x[5]:5.2f} TOST p={x[6]:.3g} {'EQUIV' if x[6]<0.05 else ''}")
