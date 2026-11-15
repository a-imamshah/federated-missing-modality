"""Federated training: local training on each client and averaging on the server."""
import copy
import torch

from utils import loss_fn


def drop_one_modality(has_img, has_txt, p):
    """Modality dropout. Each sample that has both inputs loses one of them (picked at random) with probability p."""
    both = (has_img == 1) & (has_txt == 1)
    drop = both & (torch.rand(len(both)) < p)
    hide_img = drop & (torch.rand(len(both)) < 0.5)
    has_img[hide_img] = 0
    has_txt[drop & ~hide_img] = 0


def train_client(global_model, data, args):
    """Copy the global model, train the copy on this client's data, return its weights."""
    model = copy.deepcopy(global_model)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mu = getattr(args, "mu", 0.0)      # FedProx: penalty for moving away from the model the client started from
    anchor = [p.detach().clone() for p in model.parameters()] if mu > 0 else None
    for _ in range(args.local_epochs):
        order = torch.randperm(len(data.y))
        for start in range(0, len(order), args.batch_size):
            b = order[start:start + args.batch_size]
            has_img, has_txt = data.has_img[b], data.has_txt[b]
            if args.modality_dropout > 0:
                drop_one_modality(has_img, has_txt, args.modality_dropout)
            loss = loss_fn(model(data.img[b], data.txt[b], has_img, has_txt), data.y[b])
            if anchor is not None:
                loss = loss + mu / 2 * sum(((p - q) ** 2).sum() for p, q in zip(model.parameters(), anchor))
            if getattr(model, "impute", False):
                loss = loss + model.recon_loss(data.img[b], data.txt[b], has_img, has_txt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.state_dict()


def average(states, sizes):
    """FedAvg: average the clients' weights, each weighted by its number of samples."""
    total = sum(sizes)
    return {k: sum(s[k] * n for s, n in zip(states, sizes)) / total for k in states[0]}


def average_by_modality(states, clients, global_state):
    """Modality-aware FedAvg. Each part of the model is averaged only over the clients that train it:
    the image adapter over samples that have an image, the text adapter over samples that have text,
    everything else over all samples. Weights are the number of samples that use the part."""
    out = {}
    for k in states[0]:
        if k.startswith("img_adapter."):
            sizes = [float(c.has_img.sum()) for c in clients]
        elif k.startswith("txt_adapter."):
            sizes = [float(c.has_txt.sum()) for c in clients]
        else:
            sizes = [float(len(c.y)) for c in clients]
        total = sum(sizes)
        out[k] = global_state[k] if total == 0 else sum(s[k] * n for s, n in zip(states, sizes)) / total
    return out


def run_round(global_model, clients, args):
    """One round: a random fraction (args.participation) of the clients trains, then the server averages."""
    k = max(1, round(args.participation * len(clients)))
    chosen = [clients[i] for i in torch.randperm(len(clients))[:k]]
    states = [train_client(global_model, c, args) for c in chosen]
    if getattr(args, "aggregation", "fedavg") == "aware":
        global_model.load_state_dict(average_by_modality(states, chosen, global_model.state_dict()))
    else:
        global_model.load_state_dict(average(states, [len(c.y) for c in chosen]))
