# H6 — finite label trust under label noise (PREREGISTRATION A16)

**Verdict: NOT SUPPORTED.** Primary conditions (EDSP-TL vs EDSP-default, win / tie / loss of 14 graphs, Holm): uniform:0.3 → 1/13/0 (n=14); pairwise:0.3 → 1/13/0 (n=14)

## EDSP-TL vs EDSP-default by condition (win / tie / loss; mean Δ ×100 over graphs)

| condition | graphs | win | tie | loss | mean Δ | EDSP-TL vs LGC (w/t/l) | EDSP-TL vs harmonic (w/t/l) |
|---|---|---|---|---|---|---|---|
| pairwise:0.1 | 14 | 2 | 12 | 0 | +0.04 | 0/1/13 | 1/9/4 |
| pairwise:0.2 | 14 | 2 | 12 | 0 | +0.03 | 0/0/14 | 1/8/5 |
| pairwise:0.3 | 14 | 1 | 13 | 0 | +0.00 | 0/0/14 | 0/9/5 |
| pairwise:0.4 | 14 | 1 | 13 | 0 | -0.00 | 0/3/11 | 0/9/5 |
| pairwise:0.5 | 14 | 1 | 13 | 0 | +0.01 | 0/12/2 | 0/13/1 |
| uniform:0 | 14 | 2 | 12 | 0 | +0.05 | 0/1/13 | 1/9/4 |
| uniform:0.1 | 14 | 2 | 12 | 0 | +0.04 | 0/1/13 | 1/8/5 |
| uniform:0.2 | 14 | 2 | 12 | 0 | +0.03 | 0/0/14 | 1/6/7 |
| uniform:0.3 | 14 | 1 | 13 | 0 | +0.03 | 0/0/14 | 0/5/9 |
| uniform:0.4 | 14 | 1 | 13 | 0 | +0.01 | 0/2/12 | 0/5/9 |
| uniform:0.5 | 14 | 0 | 14 | 0 | -0.00 | 0/2/12 | 0/7/7 |

## Per graph, primary conditions (Δ ×100, Holm p)

| graph | uniform:0.3 | pairwise:0.3 |
|---|---|---|
| amazon-computers | -0.01 (p=1, tie) | -0.03 (p=1, tie) |
| amazon-photo | -0.03 (p=1, tie) | -0.02 (p=1, tie) |
| amazon-ratings | -0.01 (p=1, tie) | -0.00 (p=1, tie) |
| chameleon-filtered | +0.13 (p=nan, tie) | +0.00 (p=nan, tie) |
| citeseer | +0.01 (p=1, tie) | -0.01 (p=1, tie) |
| coauthor-cs | -0.01 (p=nan, tie) | -0.03 (p=1, tie) |
| coauthor-physics | -0.01 (p=1, tie) | -0.02 (p=1, tie) |
| cora | +0.00 (p=nan, tie) | -0.03 (p=1, tie) |
| pubmed | +0.00 (p=1, tie) | -0.04 (p=1, tie) |
| questions | +0.01 (p=0.125, tie) | +0.01 (p=0.156, tie) |
| roman-empire | -0.01 (p=1, tie) | -0.02 (p=1, tie) |
| squirrel-filtered | +0.02 (p=nan, tie) | +0.00 (p=nan, tie) |
| tolokers | +0.00 (p=nan, tie) | +0.00 (p=nan, tie) |
| wikics | +0.25 (p=0.00262, win) | +0.25 (p=0.00216, win) |

## Fixed-trust sweep: mean Δ ×100 of EDSP-TL=T vs EDSP-default (over graphs and splits)

| condition | T=0.5 | T=1 | T=2 | T=4 |
|---|---|---|---|---|
| pairwise:0.1 | +0.032 | +0.016 | +0.006 | -0.002 |
| pairwise:0.2 | -0.007 | -0.011 | -0.001 | -0.003 |
| pairwise:0.3 | -0.026 | -0.005 | -0.003 | +0.003 |
| pairwise:0.4 | -0.029 | -0.010 | -0.004 | -0.002 |
| pairwise:0.5 | +0.014 | +0.019 | +0.004 | +0.001 |
| uniform:0 | +0.057 | +0.020 | -0.001 | +0.000 |
| uniform:0.1 | +0.039 | +0.014 | +0.000 | +0.003 |
| uniform:0.2 | +0.006 | +0.003 | -0.004 | -0.002 |
| uniform:0.3 | -0.007 | -0.011 | -0.010 | -0.001 |
| uniform:0.4 | -0.024 | -0.011 | -0.005 | -0.000 |
| uniform:0.5 | -0.035 | -0.021 | -0.015 | -0.005 |

## T_L chosen by validation (EDSP-TL), counts over splits

- pairwise:0.1: T_L=0.5: 61, T_L=1.0: 13, T_L=2.0: 7, T_L=4.0: 2, T_L=inf: 70
- pairwise:0.2: T_L=0.5: 49, T_L=1.0: 11, T_L=2.0: 13, T_L=4.0: 4, T_L=inf: 76
- pairwise:0.3: T_L=0.5: 47, T_L=1.0: 12, T_L=2.0: 7, T_L=4.0: 4, T_L=inf: 83
- pairwise:0.4: T_L=0.5: 43, T_L=1.0: 15, T_L=2.0: 9, T_L=4.0: 6, T_L=inf: 80
- pairwise:0.5: T_L=0.5: 57, T_L=1.0: 15, T_L=2.0: 7, T_L=4.0: 6, T_L=inf: 68
- uniform:0: T_L=0.5: 61, T_L=1.0: 8, T_L=2.0: 2, T_L=4.0: 5, T_L=inf: 77
- uniform:0.1: T_L=0.5: 52, T_L=1.0: 10, T_L=2.0: 6, T_L=4.0: 3, T_L=inf: 82
- uniform:0.2: T_L=0.5: 50, T_L=1.0: 11, T_L=2.0: 6, T_L=4.0: 6, T_L=inf: 80
- uniform:0.3: T_L=0.5: 41, T_L=1.0: 11, T_L=2.0: 6, T_L=4.0: 6, T_L=inf: 89
- uniform:0.4: T_L=0.5: 36, T_L=1.0: 10, T_L=2.0: 11, T_L=4.0: 5, T_L=inf: 91
- uniform:0.5: T_L=0.5: 31, T_L=1.0: 12, T_L=2.0: 11, T_L=4.0: 7, T_L=inf: 92
