"""Federated training: local training on each client and averaging on the server."""
import copy
import math
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


_residual = {}   # error feedback: what each client could not send yet (one process runs one experiment)


def uploaded_keys(state, client, partial):
    """The weights a client uploads. With partial uploads a client without images (text) leaves out the image (text) adapter."""
    skip = []
    if partial and not client.has_img.any():
        skip.append("img_adapter.")
    if partial and not client.has_txt.any():
        skip.append("txt_adapter.")
    return [k for k in state if not any(k.startswith(p) for p in skip)]


def compress_state(base, state, client, cid, args):
    """What the server receives from a client when uploads are compressed: the global weights plus the compressed update
    (update = trained weights - weights the client started from). int8: every tensor is rounded to 255 levels.
    topk: only the largest args.topk share of all entries is sent. With args.error_feedback the part that was not
    sent is added to the client's next update."""
    keys = uploaded_keys(state, client, args.aggregation == "aware")
    delta = {k: state[k] - base[k] for k in keys}
    if args.error_feedback:
        delta = {k: d + _residual.get(cid, {}).get(k, 0) for k, d in delta.items()}
    if args.compress == "int8":
        sent = {}
        for k, d in delta.items():
            scale = d.abs().max().clamp(min=1e-12) / 127
            sent[k] = torch.round(d / scale).clamp(-127, 127) * scale
    else:
        flat = torch.cat([d.flatten() for d in delta.values()])
        keep = flat.abs().topk(max(1, math.ceil(args.topk * flat.numel()))).indices
        out = torch.zeros_like(flat)
        out[keep] = flat[keep]
        sent, pos = {}, 0
        for k, d in delta.items():
            sent[k] = out[pos:pos + d.numel()].view_as(d)
            pos += d.numel()
    if args.error_feedback:
        _residual[cid] = {k: delta[k] - sent[k] for k in delta}
    return {**state, **{k: base[k] + sent[k] for k in keys}}


def run_round(global_model, clients, args):
    """One round: a random fraction (args.participation) of the clients trains, then the server averages."""
    k = max(1, round(args.participation * len(clients)))
    ids = torch.randperm(len(clients))[:k].tolist()
    chosen = [clients[i] for i in ids]
    states = [train_client(global_model, c, args) for c in chosen]
    if getattr(args, "compress", "none") != "none":
        base = {n: v.clone() for n, v in global_model.state_dict().items()}
        states = [compress_state(base, st, c, i, args) for st, c, i in zip(states, chosen, ids)]
    if getattr(args, "aggregation", "fedavg") == "aware":
        global_model.load_state_dict(average_by_modality(states, chosen, global_model.state_dict()))
    else:
        global_model.load_state_dict(average(states, [len(c.y) for c in chosen]))
