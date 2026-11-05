# PR-1b findings

These toy-graph findings are generated from recorded runs in JSONL form.

| Case | EDSP-canonical | EDSP-GS (id) | EDSP-GS (seeded random, seed=0) | EDSP-damped(beta=0.5) |
|---|---|---|---|---|
| (a) one homophilic edge, p=0.5 | converged in 261; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=9.341e-07; V=[[0.7773, 0.2227],  [0.2227, 0.7773]] | converged in 101; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=8.337e-07; V=[[0.7773, 0.2227],  [0.2227, 0.7773]] | converged in 132; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=9.193e-07; V=[[0.7773, 0.2227],  [0.2227, 0.7773]] | converged in 5; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=1.921e-08; V=[[0.7773, 0.2227],  [0.2227, 0.7773]] |
| (a) one homophilic edge, p=0.9 | max_iter in 1000; osc=True; amp=0.7521601134; ||Φ(V)-V||₁=7.522e-01; V=[[0.8989, 0.1011],  [0.1011, 0.8989]] | converged in 7; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=4.896e-08; V=[[0.5228, 0.4772],  [0.1011, 0.8989]] | converged in 8; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=2.946e-08; V=[[0.5228, 0.4772],  [0.1011, 0.8989]] | converged in 8; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=3.109e-08; V=[[0.7466, 0.2534],  [0.2534, 0.7466]] |
| (b) heterophilic 4-cycle, p=0.3 | converged in 321; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=9.357e-07; V=[[0.7763, 0.2237],  [0.7763, 0.2237],  [0.7763, 0.2237],  [0.7763, 0.2237]] | converged in 117; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=9.188e-07; V=[[0.7763, 0.2237],  [0.7763, 0.2237],  [0.7763, 0.2237],  [0.7763, 0.2237]] | converged in 153; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=6.872e-07; V=[[0.7763, 0.2237],  [0.7763, 0.2237],  [0.7763, 0.2237],  [0.7763, 0.2237]] | converged in 5; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=9.882e-09; V=[[0.7763, 0.2237],  [0.7763, 0.2237],  [0.7763, 0.2237],  [0.7763, 0.2237]] |
| (b) heterophilic 4-cycle, p=0.5 | max_iter in 1000; osc=True; amp=0.6332320112; ||Φ(V)-V||₁=6.332e-01; V=[[0.8898, 0.1102],  [0.8898, 0.1102],  [0.8898, 0.1102],  [0.8898, 0.1102]] | converged in 20; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=2.821e-07; V=[[0.5732, 0.4268],  [0.8898, 0.1102],  [0.5732, 0.4268],  [0.8898, 0.1102]] | converged in 19; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=1.564e-07; V=[[0.5732, 0.4268],  [0.8898, 0.1102],  [0.5732, 0.4268],  [0.8898, 0.1102]] | converged in 6; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=1.321e-07; V=[[0.7562, 0.2438],  [0.7562, 0.2438],  [0.7562, 0.2438],  [0.7562, 0.2438]] |
| (b) heterophilic 4-cycle, p=0.7 | max_iter in 1000; osc=True; amp=0.7577581928; ||Φ(V)-V||₁=7.578e-01; V=[[0.8991, 0.1009],  [0.8991, 0.1009],  [0.8991, 0.1009],  [0.8991, 0.1009]] | converged in 11; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=8.288e-08; V=[[0.5203, 0.4797],  [0.8991, 0.1009],  [0.5203, 0.4797],  [0.8991, 0.1009]] | converged in 7; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=5.016e-07; V=[[0.5203, 0.4797],  [0.8991, 0.1009],  [0.5203, 0.4797],  [0.8991, 0.1009]] | converged in 8; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=3.741e-08; V=[[0.7461, 0.2539],  [0.7461, 0.2539],  [0.7461, 0.2539],  [0.7461, 0.2539]] |
| (b) heterophilic 4-cycle, p=0.9 | max_iter in 1000; osc=True; amp=0.7959321616; ||Φ(V)-V||₁=7.959e-01; V=[[0.9, 0.1],  [0.9, 0.1],  [0.9, 0.1],  [0.9, 0.1]] | converged in 8; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=5.892e-09; V=[[0.502, 0.498],  [0.9  , 0.1  ],  [0.502, 0.498],  [0.9  , 0.1  ]] | converged in 3; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=3.368e-09; V=[[0.502, 0.498],  [0.9  , 0.1  ],  [0.502, 0.498],  [0.9  , 0.1  ]] | converged in 8; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=1.364e-07; V=[[0.7416, 0.2584],  [0.7416, 0.2584],  [0.7416, 0.2584],  [0.7416, 0.2584]] |
| (c) one-hot priors, one homophilic edge, p=0.9 | max_iter in 1000; osc=True; amp=0.9365404153; ||Φ(V)-V||₁=9.365e-01; V=[[0.9987, 0.0013],  [0.0013, 0.9987]] | converged in 9; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=8.530e-08; V=[[0.5305, 0.4695],  [0.0013, 0.9987]] | converged in 9; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=1.375e-06; V=[[0.5305, 0.4695],  [0.0013, 0.9987]] | converged in 7; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=8.489e-08; V=[[0.8185, 0.1815],  [0.1815, 0.8185]] |
| (e7) unlabelled heterophilic 4-cycle, one non-uniform start, p=0.9 | converged in 79; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=8.777e-07; V=[[0.9, 0.1],  [0.1, 0.9],  [0.9, 0.1],  [0.1, 0.9]] | converged in 41; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=4.578e-07; V=[[0.9, 0.1],  [0.1, 0.9],  [0.9, 0.1],  [0.1, 0.9]] | converged in 51; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=6.565e-07; V=[[0.9, 0.1],  [0.1, 0.9],  [0.9, 0.1],  [0.1, 0.9]] | converged in 145; osc=False; amp=0.0000000000; ||Φ(V)-V||₁=1.758e-06; V=[[0.9, 0.1],  [0.1, 0.9],  [0.9, 0.1],  [0.1, 0.9]] |

Run command:

```text
uv run python scripts/findings_pr1b.py --write-docs
```

Records file: `results/runs/findings_pr1b.jsonl`

Case (a), p=0.9 has at least three fixed points under canonical Φ:
- Symmetric fixed point (captured by damped): V0≈[0.7466, 0.2534], V1≈[0.2534, 0.7466]
- Asymmetric mirror pair (GS-id picks one by node order): V0≈[0.5228, 0.4772], V1≈[0.1011, 0.8989]
