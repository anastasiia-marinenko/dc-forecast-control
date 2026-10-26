"""Decision module + simple model of a hall with two zones.

g servers in two zones:
  A (60 %) - hot aisle is closed, so supply air can be only 3 K below the zone temperature
  B (40 %) - air mixes, supply air has to be 9 K colder
Two zones is the simplest case where it matters *where* tasks run.

Each minute the module only knows the forecast y for H minutes ahead and decides:
  1) servers: n = ceil((1 + m) * y), m - server reserve (can be < 0 if a model overestimates a lot)
  2) placement: fill zone A first, the rest goes to B. If load > n, that minute is a shortage (V)
  3) cooling: power can change by at most `ramp` of max per minute, so we raise it in advance
     for the heat the forecast expects, and lower it only when the heat actually drops.
     Target is 27 - s degrees (s - setback). Zone above 27 C = overheating minute (Theta).

Server: P = 100 W + 100 W * utilisation, switched off = 0.
Cooling: removed heat / COP, COP = 0.0068 Ts^2 + 0.0008 Ts + 0.458 (Moore et al., 2005),
Ts - supply air temperature. Zone temperature: C dT/dt = P_IT - Q.

The numbers are reasonable but not from a real data center - it's a test bench to compare
forecasters, not a prediction of real savings.
"""
from __future__ import annotations

import numpy as np
from numba import njit

P_IDLE, P_MAX = 100.0, 200.0      # W per server, idle server takes half of peak
T_MAX = 27.0                      # C, ASHRAE recommended upper limit
T_FLOOR = 18.0                    # C, never cool below this
DT = 60.0                         # s, one step = 1 minute
C_PER_SERVER = 6000.0             # J/K per server, full zone without cooling heats ~2 K/min
TAU = 300.0                       # s, how fast cooling fixes a temperature error
SUPPLY_DELTA = (3.0, 9.0)         # K, supply air below zone temperature (A, B)
SHARE_A = 0.6                     # part of servers in zone A
QMAX_FACTOR = 1.2                 # cooling capacity / max heat of a zone

MARGINS = np.round(np.arange(-0.5, 2.0001, 0.02), 2)     # values of m we try
SETBACKS = np.round(np.arange(0.0, 9.0001, 0.25), 2)     # values of s we try, K


@njit(cache=True)
def cop(t_supply):
    return 0.0068 * t_supply * t_supply + 0.0008 * t_supply + 0.458


@njit(cache=True)
def zone_power(load, n, nA):
    """Server power (W) in zones A and B, n servers on, A filled first."""
    na = min(n, nA)
    nb = n - na
    busy_a = min(load, na)
    busy_b = min(max(load - na, 0.0), nb)
    return na * P_IDLE + busy_a * (P_MAX - P_IDLE), nb * P_IDLE + busy_b * (P_MAX - P_IDLE)


@njit(cache=True)
def active_servers(yhat, m, g):
    """n_t = ceil((1 + m) * yhat_t), between 1 and g."""
    n = np.empty(yhat.shape[0], np.int64)
    for t in range(yhat.shape[0]):
        v = np.ceil((1.0 + m) * max(yhat[t], 0.0) - 1e-9)
        n[t] = int(min(max(v, 1.0), g))
    return n


@njit(cache=True)
def shortage_share(d, yhat, m, g):
    """V = share of minutes where load > servers on. Only depends on m."""
    n = active_servers(yhat, m, g)
    v = 0
    for t in range(d.shape[0]):
        if d[t] > n[t] + 1e-9:
            v += 1
    return v / d.shape[0]


@njit(cache=True)
def simulate(d, yhat, g, m, s, H, ramp, always_on):
    """Simulate the hall.

    d[t] - real load (busy servers), yhat[t] - forecast for t made H minutes before.
    Returns server kWh, cooling kWh, V, Theta.
    """
    T = d.shape[0]
    nA = int(round(SHARE_A * g))
    nB = g - nA
    C = (C_PER_SERVER * nA, C_PER_SERVER * nB)
    qmax = (QMAX_FACTOR * nA * P_MAX, QMAX_FACTOR * nB * P_MAX)
    t_ref = T_MAX - s                                   # temperature we aim for
    if always_on:
        n = np.full(T, g, np.int64)
    else:
        n = active_servers(yhat, m, g)
    temp = np.array([t_ref, t_ref])
    p0a, p0b = zone_power(d[0], n[0], nA)
    q = np.array([p0a, p0b])                            # cooling power per zone, W
    e_srv = 0.0
    e_cool = 0.0
    short = 0
    hot = 0
    for t in range(T):
        pa, pb = zone_power(d[t], n[t], nA)            # heat right now
        e_srv += pa + pb
        if d[t] > n[t] + 1e-9:
            short += 1
        tp = min(t + H, T - 1)                          # heat the forecast expects at t + H
        fa, fb = zone_power(yhat[tp], g if always_on else n[tp], nA)
        over = False
        for z in range(2):
            p_fore = fa if z == 0 else fb
            p_now = pa if z == 0 else pb
            # get ready for a rise early, follow drops when they happen,
            # plus fix the gap between zone temperature and target
            target = max(p_now, p_fore) + C[z] / TAU * (temp[z] - t_ref)
            step = ramp * qmax[z]
            qz = min(max(target, q[z] - step), q[z] + step)       # can't change faster than ramp
            qz = min(max(qz, 0.0), qmax[z])
            q[z] = qz
            e_cool += qz / cop(temp[z] - SUPPLY_DELTA[z])
            temp[z] = max(T_FLOOR, temp[z] + DT / C[z] * (p_now - qz))
            if temp[z] > T_MAX + 1e-9:
                over = True
        if over:
            hot += 1
    return e_srv * DT / 3.6e6, e_cool * DT / 3.6e6, short / T, hot / T
