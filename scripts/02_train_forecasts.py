"""Step 2: train the forecasters for one horizon H and save forecasts for every hall and minute.

  python scripts/02_train_forecasts.py --series data/series.npz --H 5 --seeds 0,1,2,3,4,5,6,7,8,9 --out results/H5

Split in time: first 60 % - training, next 20 % - validation (lr, early stopping, ridge alpha,
and later the reserves), last 20 % - test. Models only look at the past.

Saves forecasts.npz (persistence, ridge, gru_mse_s<seed>, gru_l<lambda>_s<seed>) and train_log.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dccontrol import forecast as fc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--series", required=True)
ap.add_argument("--H", type=int, default=5)
ap.add_argument("--seeds", default="0,1,2,3,4")
ap.add_argument("--lams", default="1,2,4,8", help="lambda values for the asymmetric loss")
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--stride", type=int, default=10, help="use every 10th training window (they overlap a lot)")
ap.add_argument("--train-share", type=float, default=0.6)
ap.add_argument("--val-share", type=float, default=0.2)
ap.add_argument("--max-series", type=int, default=0, help="only first N halls, for quick tests")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
logf = open(os.path.join(a.out, "train_log.txt"), "w")
T0 = time.time()


def log(*msg):
    s = " ".join(str(x) for x in msg)
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


S = np.load(a.series)["load"].astype(np.float64)
if a.max_series:
    S = S[:a.max_series]
n_s, T = S.shape
H = a.H
n_tr, n_va = int(a.train_share * T), int((a.train_share + a.val_share) * T)
log(f"{n_s} halls, {T} min, H = {H}; train 0-{n_tr}, val {n_tr}-{n_va}, test {n_va}-{T}")

# normalise each hall by mean/std of its own training part, so one network fits all halls
mu, sd = S[:, :n_tr].mean(1), S[:, :n_tr].std(1) + 1e-6
Xtr, ytr, _, _ = fc.make_windows(S, mu, sd, H, 0, n_tr, stride=a.stride)
Xva, yva, _, _ = fc.make_windows(S, mu, sd, H, n_tr, n_va, stride=5)
Xall, _, ids, ts = fc.make_windows(S, mu, sd, H, n_tr, T, stride=1)        # val + test, every minute
device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"train windows {len(Xtr)}, val {len(Xva)}, to predict {len(Xall)}, device: {device}")

F = {"persistence": fc.persistence(S, H)}
w, rinfo = fc.ridge(Xtr, ytr, Xva, yva)
F["ridge"] = fc.to_level(S, mu, sd, ids, ts, np.c_[Xall, np.ones(len(Xall))] @ w, H)
info = {"ridge": rinfo, "H": H, "n_tr": n_tr, "n_va": n_va, "T": T}
log(f"ridge: {rinfo}")

for seed in [int(v) for v in a.seeds.split(",")]:
    for kind in ["mse"] + a.lams.split(","):
        t0 = time.time()
        model, inf = fc.train_gru(kind, Xtr, ytr, Xva, yva, device, epochs=a.epochs, seed=seed, log=log)
        name = f"gru_mse_s{seed}" if kind == "mse" else f"gru_l{kind}_s{seed}"
        F[name] = fc.to_level(S, mu, sd, ids, ts, fc.predict(model, Xall, device), H)
        inf["seconds"] = round(time.time() - t0, 1)
        info[name] = inf
        log(f"  {name}: {inf}  ({(time.time() - T0) / 60:.0f} min so far)")

np.savez_compressed(os.path.join(a.out, "forecasts.npz"), load=S.astype(np.float32), n_tr=n_tr, n_va=n_va, H=H,
                    **{k: v.astype(np.float32) for k, v in F.items()})
json.dump(info, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)
log(f"finished in {(time.time() - T0) / 60:.0f} min, saved to {a.out}")
