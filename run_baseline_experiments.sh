#!/bin/bash
# Baselines on both datasets, 20 clients, 250 rounds, 3 seeds, in two settings:
#   per-sample : Fed-PRIME protocol, 70% of the samples miss an input ("both" pattern)
#   types      : fixed client types, one third image-only, one third text-only, one third full
# Methods: fedavg (the main setup), fedprox, modality dropout, imputation, aware (types only), local-only,
#          fedmsplit-style, mfedmc-style.
# Usage:  bash run_baseline_experiments.sh per-sample|types|all      then: python summarize.py results/baselines.jsonl
# Finished runs are skipped, so the script can be started again after an interruption.
OUT=results/baselines.jsonl
SEEDS=${SEEDS:-"0 1 2"}
R=${R:-250}
mkdir -p results

run() {   # run <features> <seed> <main.py options...>
  local feats=$1 seed=$2; shift 2
  local tag="$feats $seed $*"
  if grep -qxF "$tag" results/.baselines_done 2>/dev/null; then echo "skip: $tag"; return; fi
  python main.py --feats $feats --rounds $R --eval_every 50 --seed $seed --out $OUT "$@" > /dev/null \
    && echo "$tag" >> results/.baselines_done && echo "done: $tag"
}

methods() {   # methods <features> <setting options...>   (the setting is the same for every method)
  local feats=$1; shift
  for s in $SEEDS; do
    run $feats $s "$@"                                          # fedavg
    run $feats $s "$@" --mu 0.01                                # fedprox
    run $feats $s "$@" --mu 0.1                                 # fedprox, stronger
    run $feats $s "$@" --modality_dropout 0.3                   # modality dropout
    run $feats $s "$@" --impute                                 # imputation
    run $feats $s "$@" --method local                           # no sharing
    run $feats $s "$@" --method fedmsplit --mu 0.01 --temp 0.1  # similarity-weighted block averaging
    run $feats $s "$@" --method mfedmc --gamma 1 --participation 0.5   # selected encoders from half of the clients
  done
}

per_sample() { for f in feats8 feats_imdb8; do methods $f --eta 0.7 --missing both; done; }
types() {
  for f in feats8 feats_imdb8; do
    methods $f --shares 1,1,1
    for s in $SEEDS; do run $f $s --shares 1,1,1 --aggregation aware; done
  done
}

case $1 in
  per-sample) per_sample ;;
  types) types ;;
  all) per_sample; types ;;
  *) echo "usage: bash run_baseline_experiments.sh per-sample|types|all" ;;
esac
