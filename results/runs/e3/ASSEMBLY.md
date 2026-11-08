# Assembly of run records

Sources (76 refs): main, origin/results/e3-42368756508, origin/results/e3-42369205852, origin/results/e3-42382541370, origin/results/e3-42385833158, origin/results/e3-42405479377, origin/results/e3-42458905802, origin/results/e3-42458910722, origin/results/e3-42458915368, origin/results/e3-42458919783, origin/results/e3-42467691919, origin/results/e3-42467696059, origin/results/e3-42467700145, origin/results/e3-42467704159, origin/results/e3-42467708185, origin/results/e3-42467712413, origin/results/e3-42467716806, origin/results/e3-42467721355, origin/results/e3-42467726017, origin/results/e3-42467730714, origin/results/e3-42467735474, origin/results/e3-42467740065, origin/results/e3-42467744211, origin/results/e3-42467744211-rescue-42480240033, origin/results/e3-42467748518, origin/results/e3-42467753096, origin/results/e3-42467753096-rescue-42480234168, origin/results/e3-42467757225, origin/results/e3-42467761717, origin/results/e3-42467765814, origin/results/e3-42471109374, origin/results/e3-42471113519, origin/results/e3-42471117905, origin/results/e3-42471122471, origin/results/e3-42471122471-rescue-42489101397, origin/results/e3-42471127113, origin/results/e3-42471131357, origin/results/e3-42471135631, origin/results/e3-42471140026, origin/results/e3-42471144610, origin/results/e3-42471148953, origin/results/e3-42471153452, origin/results/e3-42471157812, origin/results/e3-42474871330, origin/results/e3-42474874953, origin/results/e3-42474878660, origin/results/e3-42474883039, origin/results/e3-42475954784, origin/results/e3-42475959824, origin/results/e3-42475964446, origin/results/e3-42475969615, origin/results/e3-42475974592, origin/results/e3-42475979970, origin/results/e3-42477345347, origin/results/e3-42477350460, origin/results/e3-42480358577, origin/results/e3-42480364987, origin/results/e3-42489276896, origin/results/e3-42489282373, origin/results/e3-42489288203, origin/results/e3-42489293801, origin/results/e3-42489299538, origin/results/e3-42489304680, origin/results/e3-42489310089, origin/results/e3-42490821424, origin/results/e3-42491930290, origin/results/e3-42491934661, origin/results/e3-42491938906, origin/results/e3-42491943224, origin/results/e3-42491947382, origin/results/e3-42491951606, origin/results/e3-42491955744, origin/results/e3-42493148133, origin/results/e3-42493155478, origin/results/e3-42498145946, origin/results/e3-42498152359

Scored records: 1807; datasets: 14; conflicting duplicates: 28; NOT-RUN records kept: 0 (of 0; the rest were superseded by a scored re-run).

| dataset | method: splits |
|---|---|
| amazon-computers | EDSP-default: 10, EDSP-tuned: 10, bwgnn: 10, correct_and_smooth: 10, feature_lr: 10, gat: 10, gcn: 10, gprgnn: 10, majority: 10, mlp: 10, sage: 10, train_prior: 10 |
| amazon-fraud | EDSP-default: 10, EDSP-tuned: 10, bwgnn: 10, correct_and_smooth: 10, feature_lr: 10, majority: 10, mlp: 10, train_prior: 10 |
| amazon-photo | EDSP-default: 10, EDSP-tuned: 10, bwgnn: 10, correct_and_smooth: 10, feature_lr: 10, gat: 10, gcn: 10, gprgnn: 10, majority: 10, mlp: 10, sage: 10, train_prior: 10 |
| amazon-ratings | EDSP-default: 10, EDSP-tuned: 10, bwgnn: 10, correct_and_smooth: 10, feature_lr: 10, gat: 10, gcn: 10, gprgnn: 10, majority: 10, mlp: 10, sage: 10, train_prior: 10 |
| chameleon-filtered | EDSP-default: 10, EDSP-tuned: 10, bwgnn: 10, correct_and_smooth: 10, feature_lr: 10, gat: 10, gcn: 10, gprgnn: 10, majority: 10, mlp: 10, sage: 10, train_prior: 10 |
| citeseer | EDSP-default: 11, EDSP-tuned: 11, bwgnn: 11, correct_and_smooth: 11, feature_lr: 11, gat: 11, gcn: 11, gprgnn: 11, majority: 11, mlp: 11, sage: 11, train_prior: 11 |
| cora | EDSP-default: 11, EDSP-tuned: 11, bwgnn: 11, correct_and_smooth: 11, feature_lr: 11, gat: 11, gcn: 11, gprgnn: 11, majority: 11, mlp: 11, sage: 11, train_prior: 11 |
| minesweeper | EDSP-default: 10, EDSP-tuned: 10, bwgnn: 10, correct_and_smooth: 10, feature_lr: 10, gat: 10, gcn: 10, gprgnn: 10, majority: 10, mlp: 10, sage: 10, train_prior: 10 |
| pubmed | EDSP-default: 11, EDSP-tuned: 11, bwgnn: 11, correct_and_smooth: 11, feature_lr: 11, gat: 11, gcn: 11, gprgnn: 11, majority: 11, mlp: 11, sage: 11, train_prior: 11 |
| questions | EDSP-default: 10, EDSP-tuned: 10, bwgnn: 10, correct_and_smooth: 10, feature_lr: 10, gat: 10, gat(h64): 10, gcn: 10, gprgnn: 10, majority: 10, mlp: 10, sage: 10, train_prior: 10 |
| roman-empire | EDSP-default: 10, EDSP-tuned: 10, bwgnn: 10, correct_and_smooth: 10, feature_lr: 10, gat: 10, gcn: 10, gprgnn: 10, majority: 10, mlp: 10, sage: 10, train_prior: 10 |
| squirrel-filtered | EDSP-default: 10, EDSP-tuned: 10, bwgnn: 10, correct_and_smooth: 10, feature_lr: 10, gat: 10, gcn: 10, gprgnn: 10, majority: 10, mlp: 10, sage: 10, train_prior: 10 |
| tolokers | EDSP-default: 10, EDSP-tuned: 10, bwgnn: 10, correct_and_smooth: 10, feature_lr: 10, gat: 1, gat(h64): 10, gcn: 10, gprgnn: 10, majority: 10, mlp: 10, sage: 10, train_prior: 10 |
| wikics | EDSP-default: 20, EDSP-tuned: 20, bwgnn: 20, correct_and_smooth: 20, feature_lr: 20, gat: 20, gcn: 20, gprgnn: 20, majority: 20, mlp: 20, sage: 20, train_prior: 20 |

## Left out on purpose (--drop)

- amazon-fraud:gcn: 4 records

## Conflicting duplicates (first kept)

| dataset | split | method | kept from | dropped from | kept | dropped |
|---|---|---|---|---|---|---|
| amazon-fraud | rand7 | EDSP-default | main | origin/results/e3-42382541370 | 0.933009 | 0.932272 |
| amazon-fraud | rand8 | EDSP-default | main | origin/results/e3-42382541370 | 0.943926 | 0.946892 |
| amazon-fraud | rand6 | mlp | main | origin/results/e3-42385833158 | 0.969367 | 0.927455 |
| amazon-fraud | rand7 | EDSP-default | main | origin/results/e3-42385833158 | 0.933009 | 0.932272 |
| amazon-fraud | rand7 | EDSP-tuned | main | origin/results/e3-42385833158 | 0.964454 | 0.964093 |
| amazon-fraud | rand7 | feature_lr | main | origin/results/e3-42385833158 | 0.964375 | 0.964000 |
| amazon-fraud | rand7 | correct_and_smooth | main | origin/results/e3-42385833158 | 0.952369 | 0.952376 |
| amazon-fraud | rand7 | mlp | main | origin/results/e3-42385833158 | 0.962752 | 0.915041 |
| amazon-fraud | rand8 | EDSP-default | main | origin/results/e3-42385833158 | 0.943926 | 0.946892 |
| amazon-fraud | rand8 | EDSP-tuned | main | origin/results/e3-42385833158 | 0.968648 | 0.977539 |
| amazon-fraud | rand8 | feature_lr | main | origin/results/e3-42385833158 | 0.969344 | 0.977842 |
| amazon-fraud | rand8 | correct_and_smooth | main | origin/results/e3-42385833158 | 0.942767 | 0.964386 |
| amazon-fraud | rand8 | mlp | main | origin/results/e3-42385833158 | 0.974057 | 0.947440 |
| amazon-fraud | rand9 | EDSP-default | main | origin/results/e3-42385833158 | 0.928621 | 0.927851 |
| amazon-fraud | rand9 | EDSP-tuned | main | origin/results/e3-42385833158 | 0.953094 | 0.951631 |
| amazon-fraud | rand9 | feature_lr | main | origin/results/e3-42385833158 | 0.954559 | 0.952353 |
| amazon-fraud | rand9 | correct_and_smooth | main | origin/results/e3-42385833158 | 0.951960 | 0.954333 |
| amazon-fraud | rand9 | mlp | main | origin/results/e3-42385833158 | 0.964422 | 0.916540 |
| amazon-computers | rand4 | gat | origin/results/e3-42369205852 | origin/results/e3-42458910722 | 0.924727 | 0.924000 |
| amazon-computers | rand6 | gat | origin/results/e3-42369205852 | origin/results/e3-42458919783 | 0.919636 | 0.916727 |
| tolokers | 6 | EDSP-tuned | origin/results/e3-42369205852 | origin/results/e3-42467757225 | 0.748707 | 0.748707 |
| questions | 8 | mlp | origin/results/e3-42369205852 | origin/results/e3-42471148953 | 0.716177 | 0.716177 |
| questions | 8 | gcn | origin/results/e3-42369205852 | origin/results/e3-42471148953 | 0.736555 | 0.741200 |
| questions | 4 | gprgnn | origin/results/e3-42471157812 | origin/results/e3-42475959824 | 0.753632 | 0.753628 |
| questions | 3 | gcn | origin/results/e3-42369205852 | origin/results/e3-42491934661 | 0.736525 | 0.734593 |
| questions | 6 | gcn | origin/results/e3-42471148953 | origin/results/e3-42491943224 | 0.739055 | 0.737543 |
| questions | 6 | sage | origin/results/e3-42471148953 | origin/results/e3-42491943224 | 0.746804 | 0.751248 |
| questions | 8 | gcn | origin/results/e3-42369205852 | origin/results/e3-42491951606 | 0.736555 | 0.741200 |
