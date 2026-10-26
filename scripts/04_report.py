"""Step 4: statistics for all runs + a short text summary (numbers.txt).

  python scripts/04_report.py --runs results/H5 results/H10 results/H15 --out results/report

We compare models hall by hall (all models run on the same halls, so it's a paired comparison).
For GRU variants the energy of each hall is first averaged over the training runs (seeds).

- saving of A vs B: 100 * (1 - E_A / E_B), averaged over halls
- 95 % CI: bootstrap, 5000 resamples of halls, 2.5 and 97.5 percentiles of the mean
- p-value: Wilcoxon signed-rank test on paired hall energies
- we also check the saving separately for every seed
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--v-main", type=float, default=0.01)
ap.add_argument("--theta-max", type=float, default=0.001)
ap.add_argument("--out", default="results/report")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(0)
FAMS = ["persistence", "ridge", "gru_mse", "gru_l1", "gru_l2", "gru_l4", "gru_l8", "ideal"]


def family(model):
    return model.rsplit("_s", 1)[0] if model.startswith("gru_") else model


def seed_of(model):
    return int(model.rsplit("_s", 1)[1]) if model.startswith("gru_") else None


def boot_ci(x, n=5000):
    x = np.asarray(x)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def compare(EA, EB):
    """Compare two models hall by hall (Series indexed by hall)."""
    saving = 100 * (1 - EA / EB)
    p = float(wilcoxon(EA, EB).pvalue) if (EA != EB).any() else 1.0
    return {"saving_pct": float(saving.mean()), "ci95": boot_ci(saving.values), "p": p,
            "halls_better": int((saving > 0).sum()), "halls": int(len(saving))}


summary, lines = {}, []


def say(s=""):
    print(s)
    lines.append(s)


for run in a.runs:
    trade = pd.read_csv(os.path.join(run, "tradeoff.csv"))
    trans = pd.read_csv(os.path.join(run, "transfer.csv"))
    acc = pd.read_csv(os.path.join(run, "accuracy.csv"))
    ref = pd.read_csv(os.path.join(run, "reference.csv")).set_index("hall").E
    info = json.load(open(os.path.join(run, "train_log.json")))
    H = int(info["H"])
    days = (info["T"] - info["n_va"]) / 1440          # test part, days
    for df in (trade, trans, acc):
        df["family"] = df.model.map(family)
        df["seed"] = df.model.map(seed_of)
    fams = [f for f in FAMS if f in set(trade.family)]
    seeds = sorted(trade.seed.dropna().unique().astype(int))
    S = {"H": H, "seeds": len(seeds)}
    say(f"===== H = {H} min, {len(seeds)} runs per GRU variant =====")

    # ---- RQ1: energy at the same reliability (m, s tuned on test)
    t1 = trade[trade.v_max == a.v_main]
    E = t1.groupby(["family", "hall"]).E.mean().unstack(0)            # energy per hall, mean over seeds
    S["energy_kwh_per_day"] = {f: float(E[f].mean() / days) for f in fams}
    S["always_on_kwh_per_day"] = float(ref.mean() / days)
    S["saving_vs_always_on_pct"] = {f: float(100 * (1 - E[f].mean() / ref.mean())) for f in fams}
    S["rq1_vs_mse"] = {f: compare(E[f], E["gru_mse"]) for f in fams if f not in ("gru_mse",)}
    per_seed = {}
    for f in [f for f in fams if f.startswith("gru_l")]:
        vals = []
        for s in seeds:
            ea = t1[t1.model == f"{f}_s{s}"].set_index("hall").E
            eb = t1[t1.model == f"gru_mse_s{s}"].set_index("hall").E
            vals.append(float(100 * (1 - ea.mean() / eb.mean())))
        per_seed[f] = {"per_seed_pct": vals, "positive": int(sum(v > 0 for v in vals)),
                       "min": float(min(vals)), "max": float(max(vals))}
    S["rq1_per_seed"] = per_seed
    mse_seed_E = [t1[t1.model == f"gru_mse_s{s}"].E.mean() for s in seeds]
    S["mse_seed_spread_pct"] = float(100 * np.ptp(mse_seed_E) / np.mean(mse_seed_E))
    # how much more energy than with the ideal forecast, for every shortage limit
    curve = {}
    for v, tv in trade.groupby("v_max"):
        Ev = tv.groupby(["family", "hall"]).E.mean().unstack(0)
        curve[str(v)] = {f: float(100 * (Ev[f] / Ev["ideal"] - 1).mean()) for f in fams}
        curve[str(v)]["saving_vs_mse"] = {f: float(100 * (1 - Ev[f] / Ev["gru_mse"]).mean()) for f in fams}
    S["tradeoff_curve"] = curve
    say(f"RQ1 (V <= {a.v_main:.0%}, Theta <= {a.theta_max:.1%}, m and s tuned on test):")
    say(f"  vs all servers on: saves {min(S['saving_vs_always_on_pct'][f] for f in fams if f != 'ideal'):.1f}-"
        f"{max(S['saving_vs_always_on_pct'][f] for f in fams if f != 'ideal'):.1f} %")
    for f in fams:
        if f == "gru_mse":
            say(f"  {f:12s} (baseline)")
            continue
        c = S["rq1_vs_mse"][f]
        extra = f", {per_seed[f]['positive']}/{len(seeds)} runs better" if f in per_seed else ""
        say(f"  {f:12s} {c['saving_pct']:+.2f} % vs GRU-MSE (CI {c['ci95'][0]:+.2f}..{c['ci95'][1]:+.2f}), "
            f"{c['halls_better']}/{c['halls']} halls better, p = {c['p']:.1g}{extra}")
    say(f"  GRU-MSE runs differ in energy by {S['mse_seed_spread_pct']:.2f} %")
    S["rq1_vs_ridge"] = {f: compare(E[f], E["ridge"]) for f in fams if f.startswith("gru_l")}
    for f, c in S["rq1_vs_ridge"].items():
        say(f"  {f:12s} vs ridge: {c['saving_pct']:+.2f} %, {c['halls_better']}/{c['halls']} halls better, p = {c['p']:.1g}")
    say("  extra energy vs ideal forecast, %:")
    for v in sorted(curve, key=float):
        say(f"    V <= {float(v):.2%}: " + ", ".join(f"{f} {curve[v][f]:.2f}" for f in fams if f != "ideal"))

    # ---- why: reserve and idle servers
    am = acc.groupby("family")[["mae", "bias", "under_share", "idle_calm", "shortage_in_rises", "m_at_1pct"]].mean()
    S["accuracy"] = am.round(4).to_dict("index")
    say("  errors and servers on test (mean over runs):")
    say(am.round(3).to_string())

    # ---- does a more accurate model use less energy? one point = one trained model
    ind = t1.groupby("model").E.mean()
    mae = acc.set_index("model").mae
    models = [m for m in ind.index if m in mae.index and m != "ideal"]
    rho_m, p_m = spearmanr(mae[models], ind[models])
    fam_e = E.mean()
    fam_mae = acc.groupby("family").mae.mean()
    fl = [f for f in fams if f != "ideal"]
    rho_f, p_f = spearmanr(fam_mae[fl], fam_e[fl])
    S["accuracy_vs_energy"] = {"spearman_models": float(rho_m), "p_models": float(p_m), "n_models": len(models),
                               "spearman_families": float(rho_f), "p_families": float(p_f),
                               "rank_by_mae": list(fam_mae[fl].sort_values().index),
                               "rank_by_energy": list(fam_e[fl].sort_values().index)}
    say(f"  Spearman(MAE, energy): {rho_m:.2f} for {len(models)} models (p = {p_m:.2g}), "
        f"{rho_f:.2f} for {len(fl)} variants (p = {p_f:.2g})")
    say(f"  by MAE   : {S['accuracy_vs_energy']['rank_by_mae']}")
    say(f"  by energy: {S['accuracy_vs_energy']['rank_by_energy']}")

    # ---- RQ2: tuned on validation, used on test
    tr = trans[trans.v_max == a.v_main].copy()
    tr["both_ok"] = (tr.V <= a.v_main + 1e-12) & (tr.Theta <= a.theta_max + 1e-12)
    g2 = tr.groupby("family")
    Et = tr.groupby(["family", "hall"]).E.mean().unstack(0)
    S["rq2"] = {f: {"V_val_pct": float(100 * g2.get_group(f).V_val.mean()),
                    "V_test_pct": float(100 * g2.get_group(f).V.mean()),
                    "Theta_test_pct": float(100 * g2.get_group(f).Theta.mean()),
                    "halls_both_ok_pct": float(100 * g2.get_group(f).both_ok.mean()),
                    "energy_vs_mse": compare(Et[f], Et["gru_mse"]) if f != "gru_mse" else None,
                    "Theta_runs_range_pct": [float(100 * x) for x in
                                             (g2.get_group(f).groupby("model").Theta.mean().min(),
                                              g2.get_group(f).groupby("model").Theta.mean().max())]}
                for f in fams}
    calib = {}
    for v, tv in trans.groupby("v_max"):
        calib[str(v)] = {f: {"V_test_pct": float(100 * tv[tv.family == f].V.mean()),
                             "Theta_test_pct": float(100 * tv[tv.family == f].Theta.mean())} for f in fams}
    S["rq2_calibration_curve"] = calib
    say(f"RQ2 (m, s tuned on validation for V <= {a.v_main:.0%}, then used on test):")
    for f in fams:
        r = S["rq2"][f]
        e = r["energy_vs_mse"]
        say(f"  {f:12s} V {r['V_val_pct']:.2f} % -> {r['V_test_pct']:.2f} %, Theta {r['Theta_test_pct']:.3f} %, "
            f"both limits ok in {r['halls_both_ok_pct']:.0f} % halls"
            + (f", energy vs GRU-MSE {e['saving_pct']:+.2f} %" if e else ""))
    # overheating for each training run
    runs_th = {f: [float(100 * x) for x in g2.get_group(f).groupby("model").Theta.mean().values]
               for f in fams if f.startswith("gru_")}
    S["rq2_theta_per_run"] = runs_th
    for f, v in runs_th.items():
        say(f"  {f:12s} overheating by run, %: {', '.join(f'{x:.3f}' for x in v)}; "
            f"{sum(x <= 100 * a.theta_max + 1e-9 for x in v)}/{len(v)} runs ok")

    # safety factor: tune for a stricter V on validation, then check against the real limit on test
    safety = {}
    say(f"  safety factor (stricter V when tuning, checked against V <= {a.v_main:.0%}, Theta <= {a.theta_max:.1%}):")
    for v in sorted(trans.v_max.unique()):
        if v > a.v_main + 1e-12:
            continue
        tv = trans[trans.v_max == v].copy()
        tv["ok"] = (tv.V <= a.v_main + 1e-12) & (tv.Theta <= a.theta_max + 1e-12)
        Ev = tv.groupby(["family", "hall"]).E.mean().unstack(0)
        row = {f: {"halls_both_ok_pct": float(100 * tv[tv.family == f].ok.mean()),
                   "V_test_pct": float(100 * tv[tv.family == f].V.mean()),
                   "Theta_test_pct": float(100 * tv[tv.family == f].Theta.mean()),
                   "energy_vs_mse_pct": float(100 * (1 - Ev[f].mean() / Ev["gru_mse"].mean()))} for f in fams}
        safety[str(v)] = row
        say(f"    factor {a.v_main / v:g} (target {v:.2%}): " + ", ".join(
            f"{f} {row[f]['halls_both_ok_pct']:.0f} % halls / E {row[f]['energy_vs_mse_pct']:+.2f}"
            for f in fams if f != "ideal"))
    S["rq2_safety_factor"] = safety
    summary[f"H{H}"] = S
    say()

json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
open(os.path.join(a.out, "numbers.txt"), "w").write("\n".join(lines) + "\n")
print("saved to", a.out)
