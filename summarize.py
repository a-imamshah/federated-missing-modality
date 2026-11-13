"""Average a results file over seeds. One row per setting, one column per test scenario (mean +- std, in %).
Shows F1-macro for MM-IMDb and F1-micro (= accuracy) for Food-101, like the paper.
Usage: python summarize.py results/paper.jsonl   (or results/clients.jsonl)
For runs that logged history: round at which the "missing" scenario first reaches 95% of its final score,
and the total traffic (MB) up to that round."""
import json
import sys
from collections import defaultdict
import numpy as np

SETTING = ["feats", "missing", "eta", "clients", "participation", "alpha", "modality_dropout", "rounds"]
SCENARIOS = ["full", "image_only", "text_only", "missing", "miss_both"]

groups = defaultdict(list)
for line in open(sys.argv[1] if len(sys.argv) > 1 else "results/paper.jsonl"):
    run = json.loads(line)
    if "f1_micro" not in run["results"]["full"]:      # skip old runs that stored "accuracy"
        continue
    key = tuple(run["args"].get(k) for k in SETTING)
    groups[key].append(run)


def round_to_95(run, metric):
    scores = [(h["round"], h["results"]["missing"][metric]) for h in run.get("history", [])]
    if not scores:
        return None
    target = 0.95 * scores[-1][1]
    return next(r for r, s in scores if s >= target)


print(" | ".join(SETTING) + " | seeds | " + " | ".join(SCENARIOS) + " | MB/round | s/round | rounds to 95% | MB to 95%")
for key, runs in sorted(groups.items(), key=lambda kv: tuple(-1 if v is None else v for v in kv[0])):
    metric = "f1_macro" if "imdb" in key[0] else "f1_micro"
    cells = []
    for sc in SCENARIOS:
        v = [100 * r["results"][sc][metric] for r in runs if sc in r["results"]]
        cells.append(f"{np.mean(v):.1f}±{np.std(v):.1f}" if v else "-")
    mb = runs[0]["bytes_per_round"] / 1e6
    sec = np.mean([r.get("seconds_per_round", 0) for r in runs])
    rounds95 = [x for x in (round_to_95(r, metric) for r in runs) if x is not None]
    conv = f"{np.mean(rounds95):.0f} | {np.mean(rounds95) * mb:,.0f}" if rounds95 else "- | -"
    print(" | ".join(str(k) for k in key) + f" | {len(runs)} | " + " | ".join(cells) + f" | {mb:.1f} | {sec:.2f} | {conv}")
