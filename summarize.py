"""Average result files over seeds. One row per setting, one column per test scenario (mean +- std, in %).
Shows F1-macro for MM-IMDb and F1-micro (= accuracy) for Food-101, like the paper.
Usage: python summarize.py results/paper.jsonl results/clients.jsonl results/flower.jsonl [--target 0.98]
Setting columns that have the same value in every row are left out. Runs from the simulator and from Flower share a
table ("framework"). The same run listed in two files counts once (the first file wins).

Traffic to a target: the target is --target (default 0.98) times the final "missing" score of plain FedAvg
(one local epoch, all clients, no compression) in the same dataset and missing-data setting. "rounds to target" is the first
evaluated round at which a run reaches it, "MB to target" the traffic up to that round, "x less" the ratio to plain FedAvg.
A run that never reaches it is left out of the mean; "(2/3)" means that 2 of 3 seeds reached it.
Needs runs that logged history and a plain FedAvg run in the same table."""
import json
import sys
from collections import defaultdict
import numpy as np

SETTING = ["feats", "missing", "eta", "shares", "aggregation", "method", "mu", "impute", "clients", "participation", "alpha",
           "modality_dropout", "local_epochs", "compress", "topk", "error_feedback", "rounds", "framework"]
DEFAULTS = {"framework": "simulator", "shares": "-", "aggregation": "fedavg", "method": "fedavg", "mu": 0.0, "impute": False,
            "compress": "none", "topk": None, "error_feedback": False, "local_epochs": 1}
SCENARIOS = ["full", "image_only", "text_only", "missing", "miss_both"]
REF = ["feats", "missing", "eta", "shares", "clients", "alpha"]      # runs with the same values share a plain-FedAvg reference

argv = sys.argv[1:]
target_share = 0.98
if "--target" in argv:
    i = argv.index("--target")
    target_share = float(argv[i + 1])
    del argv[i:i + 2]

groups, seen = defaultdict(list), set()
for path in argv or ["results/paper.jsonl"]:
    for line in open(path):
        run = json.loads(line)
        if "f1_micro" not in run["results"]["full"]:      # skip old runs that stored "accuracy"
            continue
        args = {**DEFAULTS, **run["args"]}
        args["shares"] = args["shares"] or "-"
        if args["compress"] != "topk":
            args["topk"] = None
        if args["compress"] == "none":
            args["error_feedback"] = False
        key = tuple(args.get(k) for k in SETTING)
        if (key, args["seed"]) in seen:
            continue
        seen.add((key, args["seed"]))
        groups[key].append(run)

col = {k: i for i, k in enumerate(SETTING)}
shown = [k for k in SETTING if k == "feats" or len({key[col[k]] for key in groups}) > 1]


def metric_of(key):
    return "f1_macro" if "imdb" in key[0] else "f1_micro"


def is_plain(key):
    return (key[col["method"]] == "fedavg" and key[col["aggregation"]] == "fedavg" and key[col["mu"]] == 0
            and not key[col["impute"]] and key[col["modality_dropout"]] == 0 and key[col["compress"]] == "none"
            and key[col["local_epochs"]] == 1 and key[col["participation"]] == 1.0)


def reached(run, metric, target):
    for h in run.get("history", []):
        if h["results"]["missing"][metric] >= target:
            return h["round"]
    return None


def to_target(key, runs, target):
    rounds = [reached(r, metric_of(key), target) for r in runs]
    ok = [x for x in rounds if x is not None]
    return ok, len(rounds)


# plain FedAvg reference per setting: target score and MB needed to reach it
reference = {}
for key, runs in groups.items():
    if is_plain(key) and all(r.get("history") for r in runs):
        metric = metric_of(key)
        target = target_share * np.mean([r["results"]["missing"][metric] for r in runs])
        ok, n = to_target(key, runs, target)
        if ok:
            reference[tuple(key[col[k]] for k in REF)] = (target, np.mean(ok) * runs[0]["bytes_per_round"] / 1e6)

print(" | ".join(shown) + " | seeds | " + " | ".join(SCENARIOS) + " | MB/round | s/round | rounds to target | MB to target | x less")
for key, runs in sorted(groups.items(), key=lambda kv: tuple(-1 if v is None else v for v in kv[0])):
    metric = metric_of(key)
    cells = []
    for sc in SCENARIOS:
        v = [100 * r["results"][sc][metric] for r in runs if sc in r["results"]]
        cells.append(f"{np.mean(v):.1f}±{np.std(v):.1f}" if v else "-")
    mb = runs[0]["bytes_per_round"] / 1e6
    sec = np.mean([r.get("seconds_per_round", 0) for r in runs])
    ref = reference.get(tuple(key[col[k]] for k in REF))
    conv = "- | - | -"
    if ref and all(r.get("history") for r in runs):
        ok, n = to_target(key, runs, ref[0])
        if ok:
            rounds, megabytes = np.mean(ok), np.mean(ok) * mb
            tag = "" if len(ok) == n else f" ({len(ok)}/{n})"
            conv = f"{rounds:.0f}{tag} | {megabytes:,.0f} | {ref[1] / megabytes:.1f}"
    print(" | ".join(str(key[col[k]]) for k in shown) + f" | {len(runs)} | " + " | ".join(cells) + f" | {mb:.1f} | {sec:.2f} | {conv}")
