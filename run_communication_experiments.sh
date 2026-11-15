#!/bin/bash
# Communication experiments: how little traffic does it take to reach the accuracy of plain FedAvg?
# Both datasets, 20 clients, 3 seeds, two settings:
#   per-sample : 70% of the samples miss an input ("both" pattern), as in Fed-PRIME
#   types      : fixed client types, one third image-only, one third text-only, one third full
# Blocks:  reference (plain FedAvg), compression, epochs, participation, combined (types setting only).
# Usage:   bash run_communication_experiments.sh reference|compression|epochs|participation|combined|all
#          then: python summarize.py results/communication.jsonl        (--target 0.98 is the default target)
# Evaluates every 5 rounds so that the rounds needed to reach the target can be read off.
# Finished runs are skipped, so after an interruption (sleep, closed terminal) the same command continues where it stopped.
# A run that was cut off in the middle is not recorded and starts again from round 1.
OUT=results/communication.jsonl
SEEDS=${SEEDS:-"0 1 2"}
R=${R:-250}
mkdir -p results

run() {   # run <features> <seed> <rounds> <main.py options...>
  local feats=$1 seed=$2 rounds=$3; shift 3
  local tag="$feats $seed $rounds $*"
  if grep -qxF "$tag" results/.communication_done 2>/dev/null; then echo "skip: $tag"; return; fi
  python main.py --feats $feats --rounds $rounds --eval_every 5 --seed $seed --out $OUT "$@" > /dev/null \
    && echo "$tag" >> results/.communication_done && echo "done: $tag" \
    || echo "FAILED: $tag"
}

settings() {   # prints the two settings, one per line
  echo "--eta 0.7 --missing both"
  echo "--shares 1,1,1"
}

each() {       # each <function>: call <function> <features> <seed> <setting options> for every combination
  for f in feats8 feats_imdb8; do for s in $SEEDS; do
    settings | while read -r setting; do $1 $f $s $setting; done
  done; done
}

reference()     { run $1 $2 $R "${@:3}"; }
compression()   {
  run $1 $2 $R "${@:3}" --compress int8
  run $1 $2 $R "${@:3}" --compress topk --topk 0.1
  run $1 $2 $R "${@:3}" --compress topk --topk 0.01
  run $1 $2 $R "${@:3}" --compress topk --topk 0.1 --error_feedback
  run $1 $2 $R "${@:3}" --compress topk --topk 0.01 --error_feedback
}
epochs()        {   # the same amount of local computation in total: rounds x epochs = R
  run $1 $2 $((R / 2)) "${@:3}" --local_epochs 2
  run $1 $2 $((R / 5)) "${@:3}" --local_epochs 5
  run $1 $2 $((R / 10)) "${@:3}" --local_epochs 10
}
participation() { for p in 0.1 0.2 0.3 0.5; do run $1 $2 $R "${@:3}" --participation $p; done; }
combined()      {   # only where clients lack a modality, so that partial uploads matter
  [[ "$3" == "--shares" ]] || return
  run $1 $2 $R "${@:3}" --aggregation aware
  run $1 $2 $R "${@:3}" --aggregation aware --compress int8
  run $1 $2 $R "${@:3}" --aggregation aware --compress topk --topk 0.1 --error_feedback
  run $1 $2 $((R / 2)) "${@:3}" --aggregation aware --compress int8 --local_epochs 2
}

case $1 in
  reference|compression|epochs|participation|combined) each $1 ;;
  all) for b in reference compression combined participation epochs; do each $b; done ;;
  *) echo "usage: bash run_communication_experiments.sh reference|compression|epochs|participation|combined|all" ;;
esac
