"""Sanity checks that clients do not share data or weights. Run: python check_simulator.py"""
import copy
import types
import torch

from data import load_split, split_train_val, split_clients
from federated import train_client, average
from model import Net

torch.manual_seed(0)
train, _ = split_train_val(load_split("feats8", "train"), 0.1)

# 1. Every training row goes to exactly one client (rows are tracked by an id stored in the image column).
tracked = train._replace(img=torch.arange(len(train.y)).float()[:, None])
for alpha in (None, 0.1):
    parts = [set(c.img[:, 0].long().tolist()) for c in split_clients(tracked, 20, alpha)]
    assert sum(len(p) for p in parts) == len(train.y) == len(set().union(*parts)), "clients overlap"
    print(f"1. alpha={alpha}: client data is disjoint and complete, sizes {sorted(len(p) for p in parts)}")

# 2. Training a client leaves the global model untouched, and clients do not share weights.
clients = split_clients(train, 20)
args = types.SimpleNamespace(lr=1e-3, local_epochs=1, batch_size=64, modality_dropout=0.0)
model = Net()
before = copy.deepcopy(model.state_dict())
states = [train_client(model, c, args) for c in clients[:3]]
assert all(torch.equal(before[k], v) for k, v in model.state_dict().items()), "global model changed"
assert not torch.equal(states[0]["classifier.weight"], states[1]["classifier.weight"]), "clients identical"
assert all(states[0][k].data_ptr() != model.state_dict()[k].data_ptr() for k in before), "memory shared"
print("2. global model unchanged by local training; client weights differ and use separate memory")

# 3. The average is the sample-weighted mean.
sizes = [len(c.y) for c in clients[:3]]
avg = average(states, sizes)["classifier.bias"]
expected = (torch.stack([s["classifier.bias"] for s in states]) * torch.tensor(sizes).float()[:, None]).sum(0) / sum(sizes)
assert torch.allclose(avg, expected, atol=1e-6), "average is wrong"
print("3. FedAvg equals the weighted mean of the client weights")
