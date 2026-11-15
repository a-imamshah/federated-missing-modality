"""Data and settings shared by the ServerApp and the ClientApp.

The app reuses data.py, model.py, federated.py and utils.py from the repository root, so the simulator and the
Flower version run the same code. The root folder is passed in the run config (repo-root)."""
import sys
from functools import lru_cache
from types import SimpleNamespace


def use_repo(root):
    """Make the repository root importable and return it."""
    if not root:
        raise ValueError('Pass the repository root: --run-config "repo-root=\\"$(cd .. && pwd)\\""')
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


@lru_cache(maxsize=2)
def _build(root, feats, seed, eta, missing, num_clients, alpha):
    """Same data setup and random order as main.py, so every process (server and clients) gets identical splits."""
    use_repo(root)
    import torch
    from data import load_split, make_missing, make_scenarios, split_clients, split_train_val

    torch.manual_seed(seed)
    train, val = split_train_val(load_split(f"{root}/{feats}", "train"), val_fraction=0.1)
    test = load_split(f"{root}/{feats}", "test")
    has_img, has_txt = make_missing(len(train.y), eta, missing)
    train = train._replace(has_img=has_img, has_txt=has_txt)
    clients = split_clients(train, num_clients, alpha)
    val_masks = make_scenarios(len(val.y), eta, missing)
    test_masks = make_scenarios(len(test.y), eta, missing)
    num_classes = train.y.shape[1] if train.y.ndim == 2 else int(train.y.max()) + 1
    return SimpleNamespace(clients=clients, val=val, test=test, val_masks=val_masks, test_masks=test_masks,
                           dim=train.img.shape[1], num_classes=num_classes)


def get_setup(run_config, num_clients):
    """Data splits for this run. alpha = 0 means an IID split."""
    return _build(use_repo(run_config["repo-root"]), run_config["feats"], int(run_config["seed"]),
                  float(run_config["eta"]), run_config["missing"], num_clients, float(run_config["alpha"]) or None)
