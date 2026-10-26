"""Pick the server reserve m and the temperature setback s for one hall and one forecaster.

Limits: shortage V <= v_max (default 1 %) and overheating Theta <= theta_max (default 0.1 %).
First we take the smallest m that keeps V under the limit (V doesn't depend on s, and bigger m
only adds idle servers). Then, with this m, the smallest s that keeps Theta under the limit
(colder air = lower COP = more energy). If no s works, we use the largest one and set thermal_ok=False.
"""
from __future__ import annotations

import numpy as np

from .hall import MARGINS, SETBACKS, shortage_share, simulate


def calibrate(d, yhat, g, H, ramp=0.1, v_max=0.01, theta_max=0.001):
    d = np.ascontiguousarray(d, np.float64)
    yhat = np.ascontiguousarray(yhat, np.float64)
    m = MARGINS[-1]
    for mm in MARGINS:                                   # grid goes up, so the first one that works is the smallest
        if shortage_share(d, yhat, mm, g) <= v_max + 1e-12:
            m = mm
            break
    for s in SETBACKS:                                   # same here: first s that works is the cheapest
        es, ec, v, th = simulate(d, yhat, g, m, s, H, ramp, False)
        if th <= theta_max + 1e-12:
            return {"m": float(m), "s": float(s), "E": es + ec, "E_server": es, "E_cool": ec,
                    "V": v, "Theta": th, "thermal_ok": True}
    return {"m": float(m), "s": float(s), "E": es + ec, "E_server": es, "E_cool": ec,
            "V": v, "Theta": th, "thermal_ok": False}


def run(d, yhat, g, H, m, s, ramp=0.1):
    """Run the hall with fixed m and s (no tuning)."""
    es, ec, v, th = simulate(np.ascontiguousarray(d, np.float64), np.ascontiguousarray(yhat, np.float64),
                             g, m, s, H, ramp, False)
    return {"m": float(m), "s": float(s), "E": es + ec, "E_server": es, "E_cool": ec, "V": v, "Theta": th}


def always_on(d, g, H, ramp=0.1, theta_max=0.001):
    """Baseline: all g servers are always on, cooling follows the real load."""
    d = np.ascontiguousarray(d, np.float64)
    for s in SETBACKS:
        es, ec, v, th = simulate(d, d, g, 0.0, s, H, ramp, True)
        if th <= theta_max + 1e-12:
            break
    return {"E": es + ec, "E_server": es, "E_cool": ec, "Theta": th, "s": float(s)}
