"""Forecasters: load of the hall H minutes ahead.

Every model gets the last 60 minutes of load (up to minute t - H) and predicts the load at minute t.

- persistence: just repeats d[t - H]
- ridge: linear regression on the 60 values with L2 penalty
- GRU: small recurrent net (Cho et al., 2014), 64 hidden units

Ridge and GRU predict the change z[t] - z[t - H] of the normalised load, and we add it to the
last known value. This way one model works for halls with very different load levels.

GRU is trained with MSE or with the asymmetric loss

    L = mean( lam * max(d - y, 0) + max(y - d, 0) )

d - real load, y - forecast. Underestimating costs lam times more than overestimating.
lam = 1 is plain MAE. The minimum of this loss is the quantile tau = lam / (1 + lam)
(Koenker & Bassett, 1978), e.g. for lam = 8 the load goes above the forecast in ~1 of 9 minutes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

L = 60      # history length, minutes


# ---- windows
def make_windows(S, mu, sd, H, t_from, t_to, stride=1):
    """Cut the series into (input, target) pairs.

    S - load [series x minutes], mu/sd - mean and std of the training part of each series,
    t_from..t_to - which target minutes to use. Input is z[t-H-L+1 .. t-H].
    Returns X [n, L], y [n] (change z[t] - z[t-H]), series index, target minute.
    """
    Xs, ys, ids, ts = [], [], [], []
    for i, x in enumerate(S):
        z = (x - mu[i]) / sd[i]
        t = np.arange(max(t_from, L + H - 1), t_to, stride)
        if len(t) == 0:
            continue
        idx = (t - H)[:, None] - np.arange(L - 1, -1, -1)[None, :]
        Xs.append(z[idx].astype(np.float32))
        ys.append((z[t] - z[t - H]).astype(np.float32))
        ids.append(np.full(len(t), i))
        ts.append(t)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ids), np.concatenate(ts)


def to_level(S, mu, sd, ids, t, dz, H):
    """Predicted change -> load in busy servers."""
    yhat = np.full(S.shape, np.nan)
    z_last = (S[ids, t - H] - mu[ids]) / sd[ids]
    yhat[ids, t] = (z_last + dz) * sd[ids] + mu[ids]
    return yhat


def persistence(S, H):
    """Forecast for t = load at t - H."""
    return np.c_[np.full((S.shape[0], H), np.nan), S[:, :-H]]


# ---- ridge
def ridge(Xtr, ytr, Xva, yva, alphas=(0.1, 1, 10, 100, 1000)):
    """Ridge regression on 60 lags + intercept.

    Neighbouring minutes are almost the same, so plain least squares gives huge weights with
    opposite signs. The penalty keeps them small. alpha is picked by MAE on validation.
    """
    A = np.c_[Xtr, np.ones(len(Xtr))].astype(np.float64)
    Av = np.c_[Xva, np.ones(len(Xva))].astype(np.float64)
    G, b = A.T @ A, A.T @ ytr.astype(np.float64)
    best = None
    for a in alphas:
        R = a * np.eye(G.shape[0])
        R[-1, -1] = 0                       # don't penalise the intercept
        w = np.linalg.solve(G + R, b)
        mae = np.mean(np.abs(Av @ w - yva))
        if best is None or mae < best[0]:
            best = (mae, w, a)
    return best[1], {"alpha": best[2], "val_mae": float(best[0])}


# ---- GRU
class GRUForecaster(nn.Module):
    """1-layer GRU, reads the 60 minutes one by one, last hidden state -> forecast."""

    def __init__(self, hidden=64):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                   # x: [batch, 60]
        h, _ = self.gru(x.unsqueeze(-1))
        return self.head(h[:, -1]).squeeze(-1)


def loss_fn(kind):
    """kind = 'mse' or a number lam >= 1 (asymmetric loss)."""
    if kind == "mse":
        return lambda p, y: nn.functional.mse_loss(p, y)
    lam = float(kind)

    def asym(p, y):
        e = y - p                                       # e > 0 means the forecast is too low
        return torch.mean(torch.where(e > 0, lam * e, -e))
    return asym


@torch.no_grad()
def _mean_loss(model, X, y, f, device, batch=8192):
    model.eval()
    tot = 0.0
    for k in range(0, len(X), batch):
        xb, yb = X[k:k + batch].to(device), y[k:k + batch].to(device)
        tot += f(model(xb), yb).item() * len(xb)
    return tot / len(X)


def train_gru(kind, Xtr, ytr, Xva, yva, device, lrs=(3e-4, 1e-3, 3e-3), epochs=8, batch=512, patience=2,
              seed=0, log=print):
    """Train one GRU with loss `kind`.

    We try 3 learning rates. For each one we stop when the validation loss hasn't improved for
    `patience` epochs and keep the best epoch. Then we keep the lr with the lowest validation loss.
    The seed sets the initial weights and batch order (different seed = different training run).
    """
    f = loss_fn(kind)
    Xtr_t, ytr_t, Xva_t, yva_t = map(torch.tensor, (Xtr, ytr, Xva, yva))
    best = (np.inf, None, None, 0)
    for lr in lrs:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = GRUForecaster().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        best_lr, best_ep, wait, state = np.inf, 0, 0, None
        gen = torch.Generator().manual_seed(seed)
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(Xtr_t), generator=gen)
            for k in range(0, len(perm), batch):
                b = perm[k:k + batch]
                loss = f(model(Xtr_t[b].to(device)), ytr_t[b].to(device))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            va = _mean_loss(model, Xva_t, yva_t, f, device)
            if va < best_lr - 1e-6:
                best_lr, best_ep, wait = va, ep + 1, 0
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        log(f"    {kind}, seed {seed}, lr {lr:g}: val loss {best_lr:.4f} (epoch {best_ep})")
        if best_lr < best[0]:
            best = (best_lr, state, lr, best_ep)
    model = GRUForecaster().to(device)
    model.load_state_dict(best[1])
    return model, {"val_loss": best[0], "lr": best[2], "best_epoch": best[3]}


@torch.no_grad()
def predict(model, X, device, batch=8192):
    model.eval()
    return np.concatenate([model(torch.tensor(X[k:k + batch]).to(device)).cpu().numpy()
                           for k in range(0, len(X), batch)])
