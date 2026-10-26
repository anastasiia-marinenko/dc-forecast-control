"""Step 3: run every forecaster through the decision module and the hall simulator.

  python scripts/03_evaluate.py --run results/H5

A) RQ1, energy vs reliability: for each shortage limit (0.1 ... 5 %) m and s are tuned on the
   test part itself, so every model is shown at its best and only energy differs.
B) RQ2, transfer: m and s are tuned on validation and then used as is on the test part
   (like an operator would do). Here limits can be broken and we check by how much.

Overheating limit is always 0.1 %. Results are saved per hall, so models can be compared hall by hall:
  tradeoff.csv (A), transfer.csv (B), reference.csv (all servers on), accuracy.csv (errors, idle servers)
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dccontrol.calibrate import always_on, calibrate, run  # noqa: E402
from dccontrol.hall import active_servers  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="folder with forecasts.npz")
ap.add_argument("--group", type=int, default=20)
ap.add_argument("--ramp", type=float, default=0.1, help="max change of cooling power per minute (share of max)")
ap.add_argument("--v-grid", default="0.001,0.0025,0.005,0.01,0.02,0.05")
ap.add_argument("--theta-max", type=float, default=0.001)
a = ap.parse_args()
T0 = time.time()

R = np.load(os.path.join(a.run, "forecasts.npz"))
S = R["load"].astype(np.float64)
n_tr, n_va, H = int(R["n_tr"]), int(R["n_va"]), int(R["H"])
n_s, T = S.shape
g = a.group
F = {k: R[k].astype(np.float64) for k in R.files if k not in ("load", "n_tr", "n_va", "H")}
F["ideal"] = S.copy()                        # perfect forecast = best possible case
# forecasts on validation start only after 60 + H minutes of history
VA, TE = slice(n_tr + 60 + H, n_va), slice(n_va, T)
VGRID = [float(v) for v in a.v_grid.split(",")]
names = list(F)
print(f"{n_s} halls, {len(names)} models, H = {H}; val {VA.stop - VA.start} min, test {T - n_va} min")


def one_hall(i, name):
    d_va, d_te = S[i, VA], S[i, TE]
    y_va, y_te = F[name][i, VA], F[name][i, TE]
    trade, trans = [], []
    for v in VGRID:
        # A) tuned on test
        r = calibrate(d_te, y_te, g, H, a.ramp, v, a.theta_max)
        trade.append({"hall": i, "model": name, "v_max": v, **r})
        # B) tuned on validation, then used on test
        c = calibrate(d_va, y_va, g, H, a.ramp, v, a.theta_max)
        t = run(d_te, y_te, g, H, c["m"], c["s"], a.ramp)
        trans.append({"hall": i, "model": name, "v_max": v, "m": c["m"], "s": c["s"], "V_val": c["V"],
                      "Theta_val": c["Theta"], "E": t["E"], "E_server": t["E_server"], "E_cool": t["E_cool"],
                      "V": t["V"], "Theta": t["Theta"]})
    return trade, trans


out = Parallel(n_jobs=-1, batch_size=8)(delayed(one_hall)(i, n) for n in names for i in range(n_s))
pd.DataFrame([r for tr, _ in out for r in tr]).to_csv(os.path.join(a.run, "tradeoff.csv"), index=False)
pd.DataFrame([r for _, tn in out for r in tn]).to_csv(os.path.join(a.run, "transfer.csv"), index=False)
print(f"A and B done ({(time.time() - T0) / 60:.1f} min)")

ref = [{"hall": i, **always_on(S[i, TE], g, H, a.ramp, a.theta_max)} for i in range(n_s)]
pd.DataFrame(ref).to_csv(os.path.join(a.run, "reference.csv"), index=False)

# forecast errors on test + where the extra servers go
d = S[:, TE]
change = d - S[:, n_va - H:T - H]                     # change of load over the last H minutes
big = np.abs(change) >= np.quantile(np.abs(change), 0.95, axis=1, keepdims=True)
rising = big & (change > 0)                          # top 5 % changes that went up
calm = ~big                                          # everything else
trade = pd.read_csv(os.path.join(a.run, "tradeoff.csv"))
m1 = trade[trade.v_max == 0.01].set_index(["model", "hall"]).m
rows = []
for name in names:
    y = F[name][:, TE]
    e = d - y
    idle_calm, short_rise = [], []
    for i in range(n_s):
        n = active_servers(np.ascontiguousarray(y[i]), float(m1.loc[(name, i)]), g)
        idle_calm.append(np.mean(np.maximum(n - d[i], 0)[calm[i]]))       # idle servers in calm minutes
        short_rise.append(np.mean((d[i] > n + 1e-9)[rising[i]]))          # shortage when load jumps up
    rows.append({"model": name, "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(-e)),
                 "under_share": float(np.mean(e > 0)), "idle_calm": float(np.mean(idle_calm)),
                 "shortage_in_rises": float(np.mean(short_rise)), "m_at_1pct": float(m1.loc[name].mean())})
pd.DataFrame(rows).to_csv(os.path.join(a.run, "accuracy.csv"), index=False)
print(f"done in {(time.time() - T0) / 60:.1f} min, results in {a.run}")
