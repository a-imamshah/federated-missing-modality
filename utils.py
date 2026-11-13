"""Loss, metrics, temperature scaling and evaluation. Works for single-label (y = class index, shape [N])
and multi-label (y = multi-hot floats, shape [N, C]); the mode is read from the shape of y."""
import torch
import torch.nn.functional as F


def loss_fn(logits, y):
    if y.ndim == 2:
        return F.binary_cross_entropy_with_logits(logits, y)
    return F.cross_entropy(logits, y)


@torch.no_grad()
def get_logits(model, data, masks):
    model.eval()
    return model(data.img, data.txt, *masks)


def f1_scores(logits, y):
    """Micro and macro F1. Multi-label: a genre is predicted when its probability is above 0.5.
    Single-label: the predicted class is the argmax (then micro F1 equals accuracy)."""
    if y.ndim == 2:
        pred, target = (logits > 0).float(), y
    else:
        pred = F.one_hot(logits.argmax(dim=1), logits.shape[1]).float()
        target = F.one_hot(y, logits.shape[1]).float()
    tp, fp, fn = (pred * target).sum(0), (pred * (1 - target)).sum(0), ((1 - pred) * target).sum(0)
    micro = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())
    per_class = torch.where(tp > 0, 2 * tp / (2 * tp + fp + fn).clamp(min=1), torch.zeros_like(tp))
    return micro.item(), per_class.mean().item()


def ece(logits, y, bins=15):
    """Expected calibration error: gap between confidence and how often it is right, averaged over confidence bins.
    Single-label: confidence of the top class. Multi-label: probability of every (sample, genre) pair vs how often it is positive."""
    if y.ndim == 2:
        conf, correct = torch.sigmoid(logits).flatten(), y.flatten()
    else:
        conf, pred = F.softmax(logits, dim=1).max(dim=1)
        correct = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            total += in_bin.float().mean().item() * abs(correct[in_bin].mean().item() - conf[in_bin].mean().item())
    return total


def fit_temperature(logits, y):
    """Find the temperature T that minimises the loss of logits / T."""
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(logits / log_t.exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_t.exp().item()


def evaluate(model, val, test, val_masks, test_masks):
    """For each scenario: fit a temperature on validation, then score the test set."""
    results = {}
    for name in test_masks:
        val_logits = get_logits(model, val, val_masks[name])
        test_logits = get_logits(model, test, test_masks[name])
        temperature = fit_temperature(val_logits, val.y)
        micro, macro = f1_scores(test_logits, test.y)
        results[name] = {"f1_micro": micro, "f1_macro": macro,
                         "ece": ece(test_logits, test.y),
                         "ece_scaled": ece(test_logits / temperature, test.y),
                         "temperature": temperature}
    return results


def bytes_per_round(model, num_clients):
    """Every client downloads and uploads all trainable weights (float32, 4 bytes each) once per round."""
    return sum(p.numel() for p in model.parameters()) * 4 * 2 * num_clients


def print_results(results):
    print(f"{'scenario':<12}{'F1-micro':>10}{'F1-macro':>10}{'ECE':>8}{'ECE scaled':>12}{'T':>7}")
    for name, r in results.items():
        print(f"{name:<12}{r['f1_micro']:>10.3f}{r['f1_macro']:>10.3f}{r['ece']:>8.3f}{r['ece_scaled']:>12.3f}{r['temperature']:>7.2f}")
