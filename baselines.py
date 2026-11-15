"""Baselines that keep one model per client: local-only, FedMSplit-style and MFedMC-style.
All three are simplified versions of the methods' main mechanism, applied to the small model on frozen CLIP features;
they are not the original implementations. Each strategy has .models (one per client), .round(clients, args) and
.bytes_per_round(rounds). The final scores are the average over the client models."""
import copy
import torch

from federated import train_client
from utils import loss_fn, traffic_per_round

IMG, TXT = "img_adapter.", "txt_adapter."
BLOCKS = [IMG, TXT, "fusion.", "classifier."]


def holds(client, block):
    """Does this client have the data that trains the block? (image adapter: images, text adapter: text)"""
    return bool(client.has_img.any()) if block == IMG else bool(client.has_txt.any()) if block == TXT else True


def block_keys(state, block):
    return [k for k in state if k.startswith(block)]


class Local:
    """Local-only: every client trains its own model and nothing is shared."""
    def __init__(self, model, clients, args):
        self.models = [copy.deepcopy(model) for _ in clients]

    def round(self, clients, args):
        for m, c in zip(self.models, clients):
            m.load_state_dict(train_client(m, c, args))

    def bytes_per_round(self, rounds):
        return 0


class SimilarityAggregation:
    """FedMSplit-style. Every client keeps its own model. The server builds, for each client and each block, a weighted
    average of the blocks of the clients that have that block. Weights = softmax(cosine similarity / temp) x samples,
    so similar clients count more. Each client then trains from its own average (with the FedProx term, args.mu,
    pulling it towards that average). All clients take part in every round; uploads are only the blocks a client trains."""
    def __init__(self, model, clients, args):
        self.models = [copy.deepcopy(model) for _ in clients]
        self.model_for_bytes = model

    def round(self, clients, args):
        states = [train_client(m, c, args) for m, c in zip(self.models, clients)]
        sizes = torch.tensor([float(len(c.y)) for c in clients])
        new = [dict(s) for s in states]
        for block in BLOCKS:
            keys = block_keys(states[0], block)
            flat = torch.stack([torch.cat([s[k].flatten() for k in keys]) for s in states])
            members = [j for j, c in enumerate(clients) if holds(c, block)]
            unit = torch.nn.functional.normalize(flat, dim=1)
            sim = unit @ unit[members].T                                   # clients x members
            w = torch.softmax(sim / args.temp, dim=1) * sizes[members]
            agg = (w / w.sum(1, keepdim=True)) @ flat[members]             # clients x parameters
            for i in range(len(clients)):
                pos = 0
                for k in keys:
                    n = states[0][k].numel()
                    new[i][k] = agg[i, pos:pos + n].view_as(states[0][k])
                    pos += n
        for m, s in zip(self.models, new):
            m.load_state_dict(s)

    def bytes_per_round(self, rounds):
        return self._bytes

    def set_bytes(self, clients, args):
        self._bytes = traffic_per_round(self.model_for_bytes, clients, 1.0, partial_upload=True)


class SelectiveUpload:
    """MFedMC-style. The two modality encoders (adapters) are shared and averaged on the server; the fusion layer and
    classifier stay local on every client. After local training the server keeps the clients with the lowest local loss
    (a fraction args.participation); each of them uploads at most args.gamma encoders it has, the least recently uploaded
    first. Every client downloads the encoders it has."""
    def __init__(self, model, clients, args):
        self.models = [copy.deepcopy(model) for _ in clients]
        state = model.state_dict()
        self.shared = {k: v.clone() for b in (IMG, TXT) for k, v in state.items() if k.startswith(b)}
        self.size = {b: 4 * sum(v.numel() for k, v in state.items() if k.startswith(b)) for b in (IMG, TXT)}
        self.last_upload = {b: [-1] * len(clients) for b in (IMG, TXT)}
        self.t, self.bytes = 0, 0

    @torch.no_grad()
    def _loss(self, m, c):
        m.eval()
        return loss_fn(m(c.img, c.txt, c.has_img, c.has_txt), c.y).item()

    def round(self, clients, args):
        self.t += 1
        losses = []
        for m, c in zip(self.models, clients):
            m.load_state_dict(train_client(m, c, args))
            losses.append(self._loss(m, c))
        k = max(1, round(args.participation * len(clients)))
        chosen = sorted(range(len(clients)), key=lambda i: losses[i])[:k]
        uploads = {IMG: [], TXT: []}
        for i in chosen:
            have = sorted((b for b in (IMG, TXT) if holds(clients[i], b)), key=lambda b: self.last_upload[b][i])
            for b in have[:args.gamma]:
                uploads[b].append(i)
                self.last_upload[b][i] = self.t
                self.bytes += self.size[b]
        for b, idx in uploads.items():
            if not idx:
                continue
            weights = [float(clients[i].has_img.sum() if b == IMG else clients[i].has_txt.sum()) for i in idx]
            for key in self.shared:
                if key.startswith(b):
                    self.shared[key] = sum(self.models[i].state_dict()[key] * w for i, w in zip(idx, weights)) / sum(weights)
        for c in clients:                                                  # downloads
            self.bytes += sum(self.size[b] for b in (IMG, TXT) if holds(c, b))
        for m in self.models:
            m.load_state_dict({**m.state_dict(), **self.shared})

    def bytes_per_round(self, rounds):
        return int(self.bytes / rounds)


def make_strategy(model, clients, args):
    """Returns None for the global-model methods (plain FedAvg, FedProx, modality dropout, imputation, aware averaging)."""
    if args.method == "local":
        return Local(model, clients, args)
    if args.method == "fedmsplit":
        s = SimilarityAggregation(model, clients, args)
        s.set_bytes(clients, args)
        return s
    if args.method == "mfedmc":
        return SelectiveUpload(model, clients, args)
    return None
