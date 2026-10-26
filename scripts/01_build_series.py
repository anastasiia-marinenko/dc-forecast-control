"""Step 1: per-minute load of each hall.

  --trace machine_usage.tar.gz   from the raw trace (20-30 min)
  --from-npz services.npz        take ready series from a file (key g20)
  --synthetic                    fake data, just to test the code

Saves an .npz with `load` [halls x minutes] (busy servers) and `group` (machines per hall).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dccontrol.data import clean_machines, load_machine_minutes, make_groups, synthetic_machines  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--trace", default="")
ap.add_argument("--from-npz", default="")
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--group", type=int, default=20)
ap.add_argument("--out", default="data/series.npz")
a = ap.parse_args()

if a.from_npz:
    S = np.load(a.from_npz)[f"g{a.group}"]
    print(f"loaded series from {a.from_npz}")
else:
    if a.synthetic:
        cpu = synthetic_machines(400)
        print("using synthetic data (only for testing)")
    else:
        raw, _ = load_machine_minutes(a.trace)
        print(f"{len(raw)} machines in the trace")
        cpu = clean_machines(raw)
    S = make_groups(cpu, a.group)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
np.savez_compressed(a.out, load=S.astype(np.float32), group=a.group)
print(f"{S.shape[0]} halls x {a.group} machines, {S.shape[1]} min ({S.shape[1] / 1440:.2f} days), "
      f"avg load {S.mean():.2f} busy servers, saved to {a.out}")
