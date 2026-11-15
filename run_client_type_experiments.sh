#!/bin/bash
# Experiments with fixed client types (image-only / text-only / full clients), 20 clients, 250 rounds, 3 seeds.
# --shares gives the relative weights of image-only,text-only,full clients; "aware" = modality-aware aggregation.
# Usage:  bash run_client_type_experiments.sh baselines|aggregation|shares|imbalance|all
#         then: python summarize.py results/client_types.jsonl
# Finished runs are skipped, so the script can be started again after an interruption.
OUT=results/client_types.jsonl
SEEDS=${SEEDS:-"0 1 2"}
R=${R:-250}

run() {   # run <features> <seed> <main.py options...>
  local feats=$1 seed=$2; shift 2
  local tag="$feats $seed $*"
  if grep -qxF "$tag" results/.client_types_done 2>/dev/null; then echo "skip: $tag"; return; fi
  python main.py --feats $feats --rounds $R --eval_every 10 --seed $seed --out $OUT "$@" > /dev/null \
    && echo "$tag" >> results/.client_types_done && echo "done: $tag"
}

baselines() {     # centralized (one client, all data) and 20 clients that all have both inputs
  for s in $SEEDS; do
    run feats8 $s --clients 1 --eta 0
    run feats8 $s --clients 20 --eta 0
  done
}
aggregation() {   # equal thirds, plain FedAvg against modality-aware averaging, both datasets
  for f in feats8 feats_imdb8; do for s in $SEEDS; do for a in fedavg aware; do
    run $f $s --shares 1,1,1 --aggregation $a
  done; done; done
}
shares() {        # share of full clients: 0%, 10%, 33%, 67%, 100% (the rest split equally)
  for w in 1,1,0 0.45,0.45,0.1 1,1,1 0.165,0.165,0.67 0,0,1; do for s in $SEEDS; do for a in fedavg aware; do
    run feats8 $s --shares $w --aggregation $a
  done; done; done
}
imbalance() {     # one client type dominates
  for w in 0.7,0.15,0.15 0.15,0.7,0.15 0.15,0.15,0.7; do for s in $SEEDS; do for a in fedavg aware; do
    run feats8 $s --shares $w --aggregation $a
  done; done; done
}

case $1 in
  baselines) baselines ;;
  aggregation) aggregation ;;
  shares) shares ;;
  imbalance) imbalance ;;
  all) baselines; aggregation; shares; imbalance ;;
  *) echo "usage: bash run_client_type_experiments.sh baselines|aggregation|shares|imbalance|all" ;;
esac
