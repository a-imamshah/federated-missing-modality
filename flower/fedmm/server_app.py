"""ServerApp: FedAvg over the clients, and evaluation of the global model on the test set."""
import json
import os
import time

import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fedmm.task import get_setup

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    num_clients = len(list(grid.get_node_ids()))
    setup = get_setup(cfg, num_clients)

    from model import Net
    from utils import bytes_per_round, evaluate, print_results

    rounds, eval_every = int(cfg["num-server-rounds"]), int(cfg["eval-every"])
    participation = float(cfg["participation"])
    selected = max(int(num_clients * participation), 1)          # clients that train in each round

    torch.manual_seed(int(cfg["seed"]))
    model = Net(dim=setup.dim, num_classes=setup.num_classes)
    history, eval_seconds = [], [0.0]

    def global_evaluate(server_round: int, arrays: ArrayRecord):
        """Scores the global model on the test set, as main.py does (temperature fitted on the validation set)."""
        if server_round == 0 or not (server_round % eval_every == 0 or server_round == rounds):
            return None
        start = time.perf_counter()
        scored = Net(dim=setup.dim, num_classes=setup.num_classes)
        scored.load_state_dict(arrays.to_torch_state_dict())
        results = evaluate(scored, setup.val, setup.test, setup.val_masks, setup.test_masks)
        history.append({"round": server_round, "results": results})
        eval_seconds[0] += time.perf_counter() - start
        scores = {f"{scenario}_{metric}": value for scenario, r in results.items() for metric, value in r.items()}
        return MetricRecord(scores)

    strategy = FedAvg(fraction_train=participation, fraction_evaluate=0.0, min_train_nodes=1, min_available_nodes=1)
    start = time.perf_counter()
    result = strategy.start(grid=grid, initial_arrays=ArrayRecord(model.state_dict()), num_rounds=rounds,
                            evaluate_fn=global_evaluate)
    seconds_per_round = (time.perf_counter() - start - eval_seconds[0]) / rounds

    final = history[-1]["results"]
    print_results(final)
    payload = sum(len(array.data) for array in result.arrays.values())     # serialized weights, one direction
    print(f"\nbytes per round: {bytes_per_round(model, selected):,}   seconds per round: {seconds_per_round:.2f}")

    record = {"args": {"framework": "flower", "feats": cfg["feats"], "clients": num_clients, "rounds": rounds,
                       "local_epochs": cfg["local-epochs"], "lr": cfg["learning-rate"], "batch_size": cfg["batch-size"],
                       "alpha": float(cfg["alpha"]) or None, "participation": participation,
                       "eta": float(cfg["eta"]), "missing": cfg["missing"], "modality_dropout": cfg["modality-dropout"],
                       "seed": int(cfg["seed"]), "eval_every": eval_every, "out": cfg["out"]},
              "bytes_per_round": bytes_per_round(model, selected), "payload_bytes_per_round": payload * 2 * selected,
              "seconds_per_round": seconds_per_round, "results": final, "history": history}
    out = os.path.join(cfg["repo-root"], cfg["out"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(record) + "\n")
