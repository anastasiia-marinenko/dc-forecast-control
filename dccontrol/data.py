"""Load data from the Alibaba cluster trace v2018 (https://github.com/alibaba/clusterdata).

machine_usage.csv has no header, columns are:
    machine_id, time_stamp (s), cpu_util_percent, mem_util_percent, mem_gps, mkpi,
    net_in, net_out, disk_io_percent
It's about 4000 machines over 8 days, 1.7 GB zipped. We read it straight from the .tar.gz,
so there's no need to unpack 270M rows to disk.

Hall load = sum of cpu% of its machines / 100, i.e. how many fully busy servers it needs.
One group of 20 machines = one small hall.
"""
from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd

TRACE_URL = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_usage.tar.gz"
MINUTES = 8 * 1440          # 8 days


def load_machine_minutes(tar_path: str, chunksize: int = 5_000_000, max_minutes: int = MINUTES,
                         log=print) -> tuple[np.ndarray, list[str]]:
    """Mean cpu% of each machine in each minute.

    Returns a [machines x minutes] matrix (NaN if a machine sent nothing that minute) and machine ids.
    Samples come every 10-60 s, so one cell = mean of the samples inside that minute.
    """
    index = pd.Index([], dtype=object)
    sums = np.zeros((4200, max_minutes), np.float64)     # sum of cpu% per cell
    cnts = np.zeros((4200, max_minutes), np.float32)     # how many samples per cell
    rows = 0
    # tar prints the csv to stdout and pandas reads it in chunks
    proc = subprocess.Popen(["tar", "-xzOf", tar_path], stdout=subprocess.PIPE)
    for ch in pd.read_csv(proc.stdout, header=None, usecols=[0, 1, 2], names=["m", "t", "cpu"],
                          chunksize=chunksize, dtype={"m": str, "t": float, "cpu": float}):
        ch = ch[(ch.cpu >= 0) & (ch.cpu <= 100) & (ch.t >= 0)]          # skip broken rows
        ch = ch.assign(mi=(ch.t // 60).astype(np.int64))                  # seconds -> minute
        ch = ch[ch.mi < max_minutes]
        new_ids = pd.Index(ch.m.unique()).difference(index)
        if len(new_ids):                                                  # more machines than rows -> add rows
            index = index.append(new_ids)
            while len(index) > sums.shape[0]:
                sums = np.vstack([sums, np.zeros((500, max_minutes))])
                cnts = np.vstack([cnts, np.zeros((500, max_minutes), np.float32)])
        ch = ch.assign(code=index.get_indexer(ch.m))
        agg = ch.groupby(["code", "mi"]).cpu.agg(["sum", "count"]).reset_index()
        np.add.at(sums, (agg.code.to_numpy(), agg.mi.to_numpy()), agg["sum"].to_numpy())
        np.add.at(cnts, (agg.code.to_numpy(), agg.mi.to_numpy()), agg["count"].to_numpy())
        rows += len(ch)
        log(f"  {rows / 1e6:.0f}M rows read, {len(index)} machines so far")
    proc.wait()
    n = len(index)
    with np.errstate(invalid="ignore", divide="ignore"):
        cpu = (sums[:n] / cnts[:n]).astype(np.float32)
    return cpu, list(index)


def clean_machines(cpu: np.ndarray, max_missing: float = 0.05, min_machines: int = 400, log=print) -> np.ndarray:
    """Remove gaps.

    1) keep the longest stretch of minutes where at least half of the machines have data
       (start and end of the trace are almost empty)
    2) drop machines that miss more than max_missing of those minutes
    3) fill what's left with the previous value of the same machine
    """
    have = ~np.isnan(cpu)
    good = have.mean(0) >= 0.5
    best, start = (0, 0, 0), None
    for t, ok in enumerate(np.append(good, False)):
        if ok and start is None:
            start = t
        if not ok and start is not None:
            if t - start > best[0]:
                best = (t - start, start, t)
            start = None
    _, t0, t1 = best
    log(f"using minutes {t0}..{t1} ({(t1 - t0) / 1440:.2f} days)")
    x = cpu[:, t0:t1]
    miss = np.isnan(x).mean(1)
    thr = max_missing
    while (miss <= thr).sum() < min_machines and thr < 0.5:   # only if too few machines are left
        thr += 0.05
    ok = miss <= thr
    x = pd.DataFrame(x[ok].T).ffill().bfill().to_numpy().T
    log(f"kept {ok.sum()} of {len(cpu)} machines (missing <= {thr:.2f})")
    return x.astype(np.float32)


def make_groups(cpu: np.ndarray, group: int, seed: int = 0) -> np.ndarray:
    """Shuffle machines, cut them into groups of `group` and return the load of each group
    in busy servers (sum of cpu% / 100)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(cpu))
    n = len(idx) // group
    return np.stack([cpu[idx[i * group:(i + 1) * group]].sum(0) / 100.0 for i in range(n)]).astype(np.float32)


def synthetic_machines(n: int = 400, minutes: int = MINUTES, seed: int = 0) -> np.ndarray:
    """Fake machines (flat, daily cycle, bursts). Just to test that the code runs."""
    rng = np.random.default_rng(seed)
    t = np.arange(minutes)
    out = np.empty((n, minutes), np.float32)
    for i in range(n):
        x = rng.uniform(15, 45) + rng.normal(0, 1.5, minutes)
        if i % 3 >= 1:
            x += rng.uniform(5, 20) * np.sin(2 * np.pi * (t / 1440 - rng.uniform()))
        if i % 3 == 2:
            for s in rng.integers(0, minutes - 60, size=rng.integers(20, 80)):
                x[s:s + rng.integers(3, 40)] += rng.uniform(15, 45)
        out[i] = np.clip(x, 0, 100)
    return out
