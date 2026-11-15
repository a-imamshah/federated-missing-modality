"""Sanity checks that clients do not share data or weights. Run: python check_simulator.py"""
import copy
import types
import torch

from data import load_split, split_train_val, split_clients, apply_client_types, make_scenarios_by_shares
from federated import train_client, average, average_by_modality
from utils import traffic_per_round
from baselines import make_strategy
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

# 4. Fixed client types: 20 clients with shares 1,1,1 become 7 image-only, 7 text-only and 6 full clients.
typed = apply_client_types(split_clients(train, 20), (1, 1, 1))
kinds = ["image-only" if not c.has_txt.any() else "text-only" if not c.has_img.any() else "full" for c in typed]
assert sorted((kinds.count(k) for k in set(kinds))) == [6, 7, 7], kinds
assert all(c.has_img.all() for c, k in zip(typed, kinds) if k != "text-only")
assert all(c.has_txt.all() for c, k in zip(typed, kinds) if k != "image-only")
masks = make_scenarios_by_shares(1000, (1, 1, 1))["missing"]
assert int(masks[0].sum()) in (666, 667) and int(masks[1].sum()) in (666, 667), (masks[0].sum(), masks[1].sum())
print("4. client types:", {k: kinds.count(k) for k in set(kinds)})

# 5. An image-only client cannot change the text adapter, and modality-aware averaging ignores it for that part.
img_client = next(c for c, k in zip(typed, kinds) if k == "image-only")
txt_client = next(c for c, k in zip(typed, kinds) if k == "text-only")
model = Net()
s_img, s_txt = train_client(model, img_client, args), train_client(model, txt_client, args)
assert all(torch.equal(s_img[k], model.state_dict()[k]) for k in s_img if k.startswith("txt_adapter.")), "text adapter changed"
assert all(torch.equal(s_txt[k], model.state_dict()[k]) for k in s_txt if k.startswith("img_adapter.")), "image adapter changed"
aware = average_by_modality([s_img, s_txt], [img_client, txt_client], model.state_dict())
assert all(torch.allclose(aware[k], s_txt[k], atol=1e-6) for k in aware if k.startswith("txt_adapter.")), "text adapter should come from the text client only"
assert all(torch.allclose(aware[k], s_img[k], atol=1e-6) for k in aware if k.startswith("img_adapter.")), "image adapter should come from the image client only"
print("5. unimodal clients leave the other adapter unchanged; modality-aware average uses only the clients that train it")

# 6. Partial uploads reduce traffic by about a quarter for each unimodal client.
full_bytes = traffic_per_round(model, typed, 1.0, partial_upload=False)
part_bytes = traffic_per_round(model, typed, 1.0, partial_upload=True)
print(f"6. MB per round: full uploads {full_bytes / 1e6:.1f}, partial uploads {part_bytes / 1e6:.1f}")
assert part_bytes < full_bytes

# 7. FedProx: a large penalty keeps the trained model close to the model it started from.
def distance(a, b):
    return sum(((a[k] - b[k]) ** 2).sum() for k in a).sqrt().item()

model = Net()
free = train_client(model, typed[0], types.SimpleNamespace(lr=1e-3, local_epochs=2, batch_size=64, modality_dropout=0.0, mu=0.0))
tied = train_client(model, typed[0], types.SimpleNamespace(lr=1e-3, local_epochs=2, batch_size=64, modality_dropout=0.0, mu=100.0))
d_free, d_tied = distance(free, model.state_dict()), distance(tied, model.state_dict())
assert d_tied < d_free, (d_tied, d_free)
print(f"7. FedProx: distance moved {d_free:.3f} without the penalty, {d_tied:.3f} with mu=100")

# 8. A missing input is never used, with zero-masking and with imputation.
for impute in (False, True):
    net = Net(impute=impute).eval()
    img, txt = torch.randn(8, 512), torch.randn(8, 512)
    ones, zeros = torch.ones(8), torch.zeros(8)
    assert torch.allclose(net(img, txt, ones, zeros), net(img, torch.randn(8, 512), ones, zeros), atol=1e-6), "masked text leaks"
    assert torch.allclose(net(img, txt, zeros, ones), net(torch.randn(8, 512), txt, zeros, ones), atol=1e-6), "masked image leaks"
print("8. masked inputs do not influence the output (zero-masking and imputation)")

# 9. Per-client baselines: one round on 2 image-only, 2 text-only and 2 full clients.
pick = [i for k in ("image-only", "text-only", "full") for i in [j for j, kk in enumerate(kinds) if kk == k][:2]]
few = [typed[i] for i in pick]
base = types.SimpleNamespace(lr=1e-3, local_epochs=1, batch_size=64, modality_dropout=0.0, mu=0.01, temp=0.1, gamma=1,
                             participation=0.5)
for method in ("local", "fedmsplit", "mfedmc"):
    a = types.SimpleNamespace(**vars(base), method=method)
    strategy = make_strategy(Net(), few, a)
    for _ in range(2):
        strategy.round(few, a)
    assert len(strategy.models) == len(few)
    assert not torch.equal(strategy.models[0].state_dict()["classifier.weight"], strategy.models[5].state_dict()["classifier.weight"]), "client models identical"
    print(f"9. {method}: {len(strategy.models)} client models, {strategy.bytes_per_round(2) / 1e6:.1f} MB per round")
