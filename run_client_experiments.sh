#!/bin/bash
# How does federated learning behave as the number of clients grows? Results go to results/clients.jsonl.
# 1 client = centralized training. Same data, same 250 rounds, 1 local epoch; only the number of clients changes.
# eta 0 isolates the effect of federation; eta 0.7 adds the paper's missing-modality setting.
# Usage: bash run_client_experiments.sh   (then: python summarize.py results/clients.jsonl)
SEEDS="0 1 2"
for f in feats8 feats_imdb8; do
  for e in 0 0.7; do
    for c in 1 5 10 20 30 50 100; do
      for s in $SEEDS; do
        python main.py --feats $f --missing both --eta $e --clients $c --rounds 250 --eval_every 10 \
          --seed $s --out results/clients.jsonl
      done
    done
  done
done
