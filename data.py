"""Loading the cached CLIP features, splitting them across clients, and hiding modalities."""
from collections import namedtuple
import numpy as np
import torch

# One row per sample. has_img / has_txt are 1.0 if that input is available, 0.0 if it is missing.
Data = namedtuple("Data", "img txt y has_img has_txt")


def load_split(folder, split):
    """Read {split}_img.npy, {split}_txt.npy, {split}_y.npy written by the Kaggle extraction script."""
    img = torch.from_numpy(np.load(f"{folder}/{split}_img.npy").astype("float32"))
    txt = torch.from_numpy(np.load(f"{folder}/{split}_txt.npy").astype("float32"))
    y = torch.from_numpy(np.load(f"{folder}/{split}_y.npy"))
    ones = torch.ones(len(y))
    return Data(img, txt, y, ones, ones.clone())


def subset(data, idx):
    return Data(*[t[idx] for t in data])


def split_train_val(data, val_fraction):
    """Hold out part of the training set. The validation part is only used to fit the temperature."""
    perm = torch.randperm(len(data.y))
    n_val = int(val_fraction * len(perm))
    return subset(data, perm[n_val:]), subset(data, perm[:n_val])


def split_clients(data, num_clients, alpha=None):
    """Split the training samples across clients. alpha=None: random (IID) split.
    alpha=0.1 (single-label only): non-IID. Each class is shared out among the clients with
    proportions drawn from a Dirichlet(alpha) distribution; a small alpha gives very uneven clients."""
    if alpha is None:
        perm = torch.randperm(len(data.y))
        return [subset(data, idx) for idx in torch.tensor_split(perm, num_clients)]
    parts = [[] for _ in range(num_clients)]
    for c in data.y.unique():
        idx = (data.y == c).nonzero().flatten()
        idx = idx[torch.randperm(len(idx))]
        share = torch.distributions.Dirichlet(torch.full((num_clients,), alpha)).sample()
        cuts = (share.cumsum(0)[:-1] * len(idx)).long()
        for part, chunk in zip(parts, torch.tensor_split(idx, cuts)):
            part.append(chunk)
    return [subset(data, torch.cat(p)) for p in parts]


def make_missing(n, eta, mode):
    """Choose which inputs are missing. Returns (has_img, has_txt) for n samples.
    mode "image": a fraction eta of the samples lose their image.
    mode "text":  a fraction eta of the samples lose their text.
    mode "both":  eta/2 of the samples lose the image and a different eta/2 lose the text."""
    has_img, has_txt = torch.ones(n), torch.ones(n)
    order = torch.randperm(n)
    k = int(eta * n)
    if mode == "image":
        has_img[order[:k]] = 0
    elif mode == "text":
        has_txt[order[:k]] = 0
    else:
        half = k // 2
        has_img[order[:half]] = 0
        has_txt[order[half:2 * half]] = 0
    return has_img, has_txt


def make_scenarios(n, eta, mode):
    """The ways a validation or test sample can arrive: with both inputs, image only, text only,
    with inputs missing in the same way as in training ("missing"), and with the "both" pattern
    ("miss_both", left out when training already used that pattern because it would be the same)."""
    ones, zeros = torch.ones(n), torch.zeros(n)
    scenarios = {"full": (ones, ones), "image_only": (ones, zeros), "text_only": (zeros, ones),
                 "missing": make_missing(n, eta, mode)}
    if mode != "both":
        scenarios["miss_both"] = make_missing(n, eta, "both")
    return scenarios


def _counts(shares, n):
    """Split n items among the entries of shares (relative weights); the counts add up to n."""
    w = torch.tensor(shares, dtype=torch.float64)
    w = w / w.sum() * n
    c = w.floor().long()
    for i in (w - c).argsort(descending=True)[:n - int(c.sum())]:
        c[i] += 1
    return c.tolist()


def apply_client_types(clients, shares):
    """Give every client one fixed modality set. shares = relative weights of (image-only, text-only, full) clients,
    for example (1, 1, 1): 20 clients become 7 / 7 / 6. An image-only client has no text for any of its samples."""
    n_img_only, n_txt_only, _ = _counts(shares, len(clients))
    typed = []
    for i, c in enumerate(clients):
        if i < n_img_only:
            c = c._replace(has_txt=torch.zeros_like(c.has_txt))
        elif i < n_img_only + n_txt_only:
            c = c._replace(has_img=torch.zeros_like(c.has_img))
        typed.append(c)
    return typed


def make_scenarios_by_shares(n, shares):
    """Test scenarios when clients have fixed types. "missing": every sample is image-only, text-only or full
    in the same proportions as the client types."""
    ones, zeros = torch.ones(n), torch.zeros(n)
    n_img_only, n_txt_only, _ = _counts(shares, n)
    order = torch.randperm(n)
    has_img, has_txt = torch.ones(n), torch.ones(n)
    has_txt[order[:n_img_only]] = 0
    has_img[order[n_img_only:n_img_only + n_txt_only]] = 0
    return {"full": (ones, ones), "image_only": (ones, zeros), "text_only": (zeros, ones),
            "missing": (has_img, has_txt)}
