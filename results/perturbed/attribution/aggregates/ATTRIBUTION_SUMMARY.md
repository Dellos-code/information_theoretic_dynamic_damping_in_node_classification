# A11 — whose win is it in MIX? (PERTURBED-REAL, not real-world results)

EDSP-tuned vs the better (by mean) of `signed_vote+HL` / `signed_lgc+HL` on the same re-wired graphs; ×100, mean over (seed, split) pairs; Wilcoxon over pairs, Holm over the conditions of a base within each family. Win = EDSP's propagation adds value beyond H-LEARNED's polarity.

| base | family | condition | pairs | EDSP-tuned | signed_vote+HL | signed_lgc+HL | Δ | p (Holm) | verdict | determinism (max |Δ| vs A4/A8) |
|---|---|---|---|---|---|---|---|---|---|---|
| amazon-ratings | loc = 0.8 | beta:m=0.2,c=0.5 | 30 | 60.80 | 58.01 | 61.87 | -1.07 | 5.2e-06 | loss | 0.00e+00 over 30 |
| amazon-ratings | loc = 0 | beta:m=0.2,c=0.5,loc=0 | 30 | 37.20 | 37.16 | 43.25 | -6.05 | 1.73e-06 | loss | 0.00e+00 over 30 |
| amazon-ratings | loc = 0.8 | beta:m=0.5,c=0.5 | 30 | 63.01 | 59.20 | 62.31 | +0.70 | 3.76e-05 | win | 0.00e+00 over 30 |
| amazon-ratings | loc = 0 | beta:m=0.5,c=0.5,loc=0 | 30 | 60.65 | 30.47 | 32.48 | +28.18 | 3.73e-09 | win | 0.00e+00 over 30 |
| amazon-ratings | loc = 0.8 | global:h=0 | 30 | 76.13 | 71.29 | 77.04 | -0.90 | 5.2e-06 | loss | 0.00e+00 over 30 |
| wiki-editor | loc = 0.8 | beta:m=0.2,c=0.5 | 30 | 91.60 | 89.28 | 90.92 | +0.68 | 2.42e-05 | win | 0.00e+00 over 30 |
| wiki-editor | loc = 0 | beta:m=0.2,c=0.5,loc=0 | 30 | 89.58 | 86.70 | 90.28 | -0.70 | 1.38e-05 | loss | 0.00e+00 over 30 |
| wiki-editor | loc = 0.8 | beta:m=0.5,c=0.5 | 30 | 88.46 | 86.34 | 88.97 | -0.51 | 0.00246 | loss | 2.48e-04 over 30 |
| wiki-editor | loc = 0 | beta:m=0.5,c=0.5,loc=0 | 30 | 71.02 | 71.26 | 70.66 | -0.25 | 0.0595 | tie | 4.95e-04 over 30 |
| wiki-editor | loc = 0.8 | beta:m=0.5,c=5 | 30 | 71.55 | 71.50 | 71.38 | +0.05 | 0.56 | tie | 2.48e-04 over 30 |
| wiki-editor | loc = 0 | beta:m=0.5,c=5,loc=0 | 30 | 61.38 | 62.37 | 62.46 | -1.08 | 1.38e-05 | loss | 4.95e-04 over 30 |
| wiki-editor | loc = 0.8 | beta:m=0.8,c=0.5 | 30 | 91.17 | 89.48 | 91.66 | -0.49 | 3.14e-05 | loss | 2.48e-04 over 30 |
| wiki-editor | loc = 0 | beta:m=0.8,c=0.5,loc=0 | 30 | 87.75 | 85.89 | 89.56 | -1.81 | 1.38e-05 | loss | 2.48e-04 over 30 |
| wiki-editor | loc = 0.8 | community:hi=0.9,lo=0.1 | 30 | 70.87 | 70.15 | 70.52 | +0.35 | 0.0332 | win | 9.90e-04 over 30 |
| wiki-editor | loc = 0 | community:hi=0.9,lo=0.1,loc=0 | 30 | 70.36 | 70.11 | 70.15 | +0.22 | 0.0634 | tie | 9.90e-04 over 30 |
| wiki-editor | loc = 0.8 | global:h=0.1 | 30 | 95.81 | 92.90 | 95.61 | +0.20 | 2.42e-05 | win | 0.00e+00 over 30 |
| wiki-editor | loc = 0.8 | global:h=0.2 | 30 | 91.96 | 88.40 | 90.92 | +1.03 | 2.42e-05 | win | 0.00e+00 over 30 |
| wiki-editor | loc = 0.8 | global:h=0.3 | 30 | 85.35 | 82.19 | 83.18 | +2.17 | 2.42e-05 | win | 0.00e+00 over 30 |
| wiki-editor | loc = 0.8 | global:h=0.4 | 30 | 73.72 | 73.25 | 69.93 | +0.47 | 0.000422 | win | 0.00e+00 over 30 |
| wiki-editor | loc = 0.8 | logistic | 30 | 83.42 | 81.86 | 84.06 | -0.63 | 0.000605 | loss | 0.00e+00 over 30 |
| wiki-editor | loc = 0 | logistic:loc=0 | 30 | 66.26 | 67.11 | 65.41 | -0.84 | 1.38e-05 | loss | 0.00e+00 over 30 |
| wiki-editor | loc = 0.8 | permutation:h=0.1 | 30 | 95.87 | 92.89 | 95.69 | +0.18 | 0.000161 | win | 0.00e+00 over 30 |
| wiki-editor | loc = 0.8 | permutation:h=0.3 | 30 | 85.46 | 82.25 | 83.17 | +2.30 | 2.42e-05 | win | 0.00e+00 over 30 |
| wiki-editor | loc = 0.8 | regional:s=20,hi=0.9,lo=0.1 | 30 | 89.44 | 87.45 | 89.53 | -0.09 | 0.56 | tie | 2.48e-04 over 30 |
| wiki-editor | loc = 0 | regional:s=20,hi=0.9,lo=0.1,loc=0 | 30 | 82.35 | 81.04 | 82.89 | -0.54 | 0.015 | loss | 4.95e-04 over 30 |
| wiki-editor | loc = 0.8 | uniform | 30 | 77.34 | 76.60 | 77.86 | -0.52 | 0.000737 | loss | 0.00e+00 over 30 |
| wiki-editor | loc = 0 | uniform:loc=0 | 30 | 64.14 | 65.19 | 63.58 | -1.05 | 1.38e-05 | loss | 0.00e+00 over 30 |
