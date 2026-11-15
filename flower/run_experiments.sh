#!/bin/bash
# Flower versions of the simulator experiments. Run inside flower/ with the Flower virtual environment active.
# Results are appended to ../results/flower.jsonl; compare with the simulator using summarize.py.
# Usage:  bash run_experiments.sh table1|eta|noniid|scale|clients
# Smaller test run:  SEEDS="0" R=20 bash run_experiments.sh table1
SEEDS=${SEEDS:-"0 1 2"}
R=${R:-250}
ROOT=$(cd .. && pwd)

# run <clients> <features> <missing pattern> <eta> <seed> [extra run-config entries]
# done_already <clients> <features> <missing> <eta> <seed> [extra...]: succeeds if the run is already in flower.jsonl
done_already() {
  python3 - "$ROOT/results/flower.jsonl" "$R" "$@" <<'PY'
import json, sys
path, rounds, clients, feats, missing, eta, seed, *extra = sys.argv[1:]
kv = dict(e.split("=") for e in extra)
alpha = float(kv["alpha"]) if "alpha" in kv else None
part = float(kv.get("participation", 1.0))
try:
    for line in open(path):
        a = json.loads(line)["args"]
        if (a["feats"], a["missing"], a["clients"], a["rounds"], a["seed"]) == (feats, missing, int(clients), int(rounds), int(seed)) \
           and abs(a["eta"] - float(eta)) < 1e-9 and a["alpha"] == alpha and abs(a["participation"] - part) < 1e-9:
            sys.exit(0)
except FileNotFoundError:
    pass
sys.exit(1)
PY
}

run() {
  local clients=$1 feats=$2 missing=$3 eta=$4 seed=$5
  shift 5
  if done_already "$clients" "$feats" "$missing" "$eta" "$seed" "$@"; then
    echo "skip (already done): clients=$clients $feats $missing eta=$eta seed=$seed $*"
    return
  fi
  flwr run . --stream \
    --federation-config "num-supernodes=$clients client-resources-num-cpus=1" \
    --run-config "num-server-rounds=$R feats=\"$feats\" missing=\"$missing\" eta=$eta seed=$seed repo-root=\"$ROOT\" $*"
}

case $1 in
  table1)   # both datasets, three missing patterns, 20 clients, eta 0.7
    for f in feats8 feats_imdb8; do for m in text image both; do for s in $SEEDS; do
      run 20 $f $m 0.7 $s; done; done; done ;;
  eta)      # Food-101, missing both, other missing rates (0.7 is in table1)
    for e in 0 0.25 0.5 0.75 1.0; do for s in $SEEDS; do
      run 20 feats8 both $e $s; done; done ;;
  noniid)   # Food-101, Dirichlet alpha 0.1
    for m in text image both; do for s in $SEEDS; do
      run 20 feats8 $m 0.7 $s alpha=0.1; done; done ;;
  scale)    # 100 clients, 10% to 50% take part in each round
    for p in 0.1 0.2 0.3 0.4 0.5; do for s in $SEEDS; do
      run 100 feats8 both 0.7 $s participation=$p; done; done ;;
  clients)  # number of clients, both datasets, eta 0 and 0.7
    for f in feats8 feats_imdb8; do for e in 0 0.7; do for c in 1 5 10 20 30 50 100; do for s in $SEEDS; do
      run $c $f both $e $s eval-every=10; done; done; done; done ;;
  *) echo "usage: bash run_experiments.sh table1|eta|noniid|scale|clients" ;;
esac
