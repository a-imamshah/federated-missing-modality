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
    for _ in range(args.local_epochs):
        order = torch.randperm(len(data.y))
        for start in range(0, len(order), args.batch_size):
            b = order[start:start + args.batch_size]
            has_img, has_txt = data.has_img[b], data.has_txt[b]
            if args.modality_dropout > 0:
                drop_one_modality(has_img, has_txt, args.modality_dropout)
            loss = loss_fn(model(data.img[b], data.txt[b], has_img, has_txt), data.y[b])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.state_dict()


def average(states, sizes):
    """FedAvg: average the clients' weights, each weighted by its number of samples."""
    total = sum(sizes)
    return {k: sum(s[k] * n for s, n in zip(states, sizes)) / total for k in states[0]}


def run_round(global_model, clients, args):
    """One round: a random fraction (args.participation) of the clients trains, then the server averages."""
    k = max(1, round(args.participation * len(clients)))
    chosen = [clients[i] for i in torch.randperm(len(clients))[:k]]
    states = [train_client(global_model, c, args) for c in chosen]
    global_model.load_state_dict(average(states, [len(c.y) for c in chosen]))
