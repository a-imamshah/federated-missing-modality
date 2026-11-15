"""Run one federated experiment on the cached CLIP features."""
import argparse
import json
import time
import torch

from data import (load_split, split_train_val, split_clients, make_missing, make_scenarios,
                  apply_client_types, make_scenarios_by_shares)
from baselines import make_strategy
from federated import run_round
from model import Net
from utils import evaluate, evaluate_average, traffic_per_round, print_results

parser = argparse.ArgumentParser()
parser.add_argument("--feats", default="feats8", help="folder with the extracted features")
parser.add_argument("--clients", type=int, default=20)
parser.add_argument("--rounds", type=int, default=50)
parser.add_argument("--local_epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--alpha", type=float, default=None, help="Dirichlet alpha for a non-IID split (single-label only); default = IID")
parser.add_argument("--participation", type=float, default=1.0, help="fraction of clients that train in each round")
parser.add_argument("--eta", type=float, default=0.7, help="fraction of samples with a missing input")
parser.add_argument("--missing", choices=["image", "text", "both"], default="both")
parser.add_argument("--shares", default=None, help="fixed client types: relative weights of image-only,text-only,full clients, e.g. 1,1,1 "
                    "(20 clients -> 7/7/6). Replaces the per-sample missing rate (--eta and --missing are ignored)")
parser.add_argument("--aggregation", choices=["fedavg", "aware"], default="fedavg",
                    help="aware: average each part of the model only over the clients that train it, and upload only those parts")
parser.add_argument("--method", choices=["fedavg", "local", "fedmsplit", "mfedmc"], default="fedavg",
                    help="baselines: local = no sharing, fedmsplit = similarity-weighted block averaging, mfedmc = upload selected encoders only")
parser.add_argument("--mu", type=float, default=0.0, help="FedProx penalty (0 = off); fedmsplit pulls towards the client's own average")
parser.add_argument("--impute", action="store_true", help="baseline: predict the missing CLIP vector from the present one")
parser.add_argument("--temp", type=float, default=0.1, help="fedmsplit: softmax temperature of the client similarity")
parser.add_argument("--gamma", type=int, default=1, help="mfedmc: number of encoders each selected client uploads")
parser.add_argument("--modality_dropout", type=float, default=0.0, help="0 = plain FedAvg with zero-masking")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--eval_every", type=int, default=5)
parser.add_argument("--out", default="results.jsonl")
args = parser.parse_args()

torch.manual_seed(args.seed)

# Data: train / validation / test, with missing inputs chosen for the training samples
train, val = split_train_val(load_split(args.feats, "train"), val_fraction=0.1)
test = load_split(args.feats, "test")
if args.shares:
    shares = [float(x) for x in args.shares.split(",")]
    clients = apply_client_types(split_clients(train, args.clients, args.alpha), shares)
    val_masks = make_scenarios_by_shares(len(val.y), shares)
    test_masks = make_scenarios_by_shares(len(test.y), shares)
    args.eta, args.missing = None, "types"                  # not used in this mode; recorded as such
else:
    has_img, has_txt = make_missing(len(train.y), args.eta, args.missing)
    train = train._replace(has_img=has_img, has_txt=has_txt)
    clients = split_clients(train, args.clients, args.alpha)
    val_masks = make_scenarios(len(val.y), args.eta, args.missing)
    test_masks = make_scenarios(len(test.y), args.eta, args.missing)

# single-label: y holds class indices; multi-label: y is a multi-hot matrix with one column per genre
num_classes = train.y.shape[1] if train.y.ndim == 2 else int(train.y.max()) + 1
model = Net(dim=train.img.shape[1], num_classes=num_classes, impute=args.impute)
strategy = make_strategy(model, clients, args)     # None: one global model

train_seconds = 0.0
history = []
for round_number in range(1, args.rounds + 1):
    start = time.perf_counter()
    if strategy:
        strategy.round(clients, args)
    else:
        run_round(model, clients, args)
    train_seconds += time.perf_counter() - start
    if round_number % args.eval_every == 0 or round_number == args.rounds:
        results = (evaluate_average(strategy.models, val, test, val_masks, test_masks) if strategy
                   else evaluate(model, val, test, val_masks, test_masks))
        history.append({"round": round_number, "results": results})
        scores = "  ".join(f"{name} {r['f1_micro']:.3f}" for name, r in results.items())
        print(f"round {round_number:>3} | F1-micro: {scores}", flush=True)

print()
print_results(results)
if strategy:
    bytes_round = strategy.bytes_per_round(args.rounds)
else:
    bytes_round = traffic_per_round(model, clients, args.participation, partial_upload=args.aggregation == "aware")
seconds_round = train_seconds / args.rounds
print(f"\nbytes per round: {bytes_round:,}   seconds per round: {seconds_round:.2f}")
with open(args.out, "a") as f:
    f.write(json.dumps({"args": vars(args), "bytes_per_round": bytes_round, "seconds_per_round": seconds_round, "results": results, "history": history}) + "\n")
