#!/bin/bash
# Experiments from the Fed-PRIME paper. Results go to results/paper.jsonl.
# Usage: bash run_paper_experiments.sh table1|eta|noniid|scale   (then: python summarize.py results/paper.jsonl)
SEEDS="0 1 2"
R=250
run() { python main.py --rounds $R --eval_every 50 --out results/paper.jsonl "$@"; }

case $1 in
  table1)   # Table 1: both datasets, three training missing types, all test scenarios
    for f in feats8 feats_imdb8; do for m in text image both; do for s in $SEEDS; do
      run --feats $f --missing $m --seed $s; done; done; done ;;
  eta)      # Fig. 4: Food-101, missing both, different missing rates
    for e in 0 0.25 0.5 0.75 1.0; do for s in $SEEDS; do
      run --feats feats8 --missing both --eta $e --seed $s; done; done ;;
  noniid)   # Table 4: Food-101, Dirichlet alpha = 0.1
    for m in text image both; do for s in $SEEDS; do
      run --feats feats8 --missing $m --alpha 0.1 --seed $s; done; done ;;
  scale)    # Supplement F: 100 clients, 10% to 50% take part each round
    for p in 0.1 0.2 0.3 0.4 0.5; do for s in $SEEDS; do
      run --feats feats8 --missing both --clients 100 --participation $p --seed $s; done; done ;;
  *) echo "usage: bash run_paper_experiments.sh table1|eta|noniid|scale" ;;
esac
