"""Step 5: figures (English and Ukrainian labels, 300 dpi).

  python scripts/05_figures.py --runs results/H5 results/H10 results/H15 --report results/report --out results/figures

fig_energy   - energy saving vs GRU-MSE for each shortage limit, one panel per horizon (RQ1)
fig_transfer - reserves tuned on validation and used on test: overheating per run and
               halls that kept both limits with a safety factor (RQ2)
fig_accuracy - MAE vs energy for every trained model
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--report", required=True)
ap.add_argument("--out", default="results/figures")
ap.add_argument("--v-main", type=float, default=0.01)
ap.add_argument("--theta-max", type=float, default=0.001)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

LINES = ["ridge", "gru_l1", "gru_l2", "gru_l4", "gru_l8"]
COLOR = {"persistence": "#4d4d4d", "ridge": "#7f7f7f", "gru_mse": "#1f6fb4", "gru_l1": "#8fbfe3",
         "gru_l2": "#f2b84b", "gru_l4": "#e2702c", "gru_l8": "#b8261d"}
MARK = {"persistence": "x", "ridge": "s", "gru_mse": "o", "gru_l1": "^", "gru_l2": "v", "gru_l4": "D", "gru_l8": "P"}
TXT = {
    "en": {"persistence": "Persistence", "ridge": "Ridge regression", "gru_mse": "GRU, MSE",
           "gru_l1": "GRU, λ = 1 (MAE)", "gru_l2": "GRU, λ = 2", "gru_l4": "GRU, λ = 4", "gru_l8": "GRU, λ = 8",
           "H": "forecast {} min ahead", "vmax": "Allowed shortage minutes, %",
           "sav": "Energy saving\nvs GRU, MSE, %", "base": "GRU, MSE", "better": "less energy",
           "a": "(a) Overheating on new data, 5 min ahead", "runs": "each dot = one training run",
           "theta": "Overheating minutes, %", "limit": "limit 0.1 %",
           "b": "(b) Halls meeting both limits, {} min ahead", "c": "(c) Halls meeting both limits, {} min ahead",
           "factor": "Shortage target used for tuning (safety factor)", "share": "Halls, %",
           "mae": "Forecast error MAE, servers", "saving": "Energy saving\nvs GRU, MSE, %",
           "acc": "H = 5 min, shortage ≤ 1 %: each point is one trained model"},
    "uk": {"persistence": "Інерційний прогноз", "ridge": "Гребенева регресія", "gru_mse": "GRU, MSE",
           "gru_l1": "GRU, λ = 1 (MAE)", "gru_l2": "GRU, λ = 2", "gru_l4": "GRU, λ = 4", "gru_l8": "GRU, λ = 8",
           "H": "прогноз на {} хв", "vmax": "Допустима частка хвилин нестачі, %",
           "sav": "Економія енергії\nвідносно GRU, MSE, %", "base": "GRU, MSE", "better": "менше енергії",
           "a": "(а) Перегрів на нових даних, прогноз на 5 хв", "runs": "кожна точка — один запуск навчання",
           "theta": "Хвилини перегріву, %", "limit": "межа 0,1 %",
           "b": "(б) Зали з обома вимогами, прогноз на {} хв", "c": "(в) Зали з обома вимогами, прогноз на {} хв",
           "factor": "Ціль нестачі під час калібрування (коефіцієнт запасу)", "share": "Залів, %",
           "mae": "Похибка прогнозу MAE, серверів", "saving": "Економія енергії\nвідносно GRU, MSE, %",
           "acc": "H = 5 хв, нестача ≤ 1 %: кожна точка — одна навчена модель"},
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})


def fam(m):
    return m.rsplit("_s", 1)[0] if m.startswith("gru_") else m


runs = {}
for r in a.runs:
    runs[int(json.load(open(os.path.join(r, "train_log.json")))["H"])] = r
summary = json.load(open(os.path.join(a.report, "summary.json")))
Hs = sorted(runs)


def save(fig, name, lang):
    fig.savefig(os.path.join(a.out, f"{name}_{lang}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


for lang in ("en", "uk"):
    t = TXT[lang]
    num = (lambda x: f"{x:g}".replace(".", ",")) if lang == "uk" else (lambda x: f"{x:g}")

    # ---- RQ1
    fig, axes = plt.subplots(1, len(Hs), figsize=(3.1 * len(Hs), 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, H in zip(axes, Hs):
        cur = summary[f"H{H}"]["tradeoff_curve"]
        vs = sorted(cur, key=float)
        x = [100 * float(v) for v in vs]
        ax.axhline(0, color=COLOR["gru_mse"], lw=2, label=t["base"])
        for f in LINES:
            ax.plot(x, [cur[v]["saving_vs_mse"][f] for v in vs], marker=MARK[f], ms=4.5, lw=1.7,
                    color=COLOR[f], label=t[f])
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([num(v) for v in x])
        ax.minorticks_off()
        ax.set_title(t["H"].format(H), fontsize=10)
        ax.set_xlabel(t["vmax"], fontsize=8.5)
    axes[0].set_ylabel(t["sav"])
    axes[0].annotate("", xy=(0.80, 0.42), xytext=(0.80, 0.18), xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="->", color="0.3", lw=1.2))
    axes[0].text(0.77, 0.30, t["better"], transform=axes[0].transAxes, fontsize=8, color="0.3", va="center",
                 ha="right")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_energy", lang)

    # ---- RQ2
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ax = axes[0]
    fams_th = ["gru_mse", "gru_l1", "gru_l2", "gru_l4", "gru_l8"]
    rng = np.random.default_rng(1)
    for k, f in enumerate(fams_th):
        v = np.array(summary[f"H{Hs[0]}"]["rq2_theta_per_run"][f])
        ax.scatter(k + rng.uniform(-0.12, 0.12, len(v)), np.maximum(v, 0.002), s=26, color=COLOR[f],
                   marker=MARK[f], edgecolor="k", linewidth=0.3, zorder=3)
        ax.plot([k - 0.25, k + 0.25], [np.mean(v)] * 2, color="k", lw=1.2, zorder=4)
    ax.axhline(100 * a.theta_max, color="#c0392b", lw=1.2, ls="--")
    ax.text(len(fams_th) - 0.5, 100 * a.theta_max * 1.12, t["limit"], color="#c0392b", ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_yticks([0.003, 0.01, 0.03, 0.1, 0.3, 1])
    ax.set_yticklabels([num(v) for v in (0.003, 0.01, 0.03, 0.1, 0.3, 1)])
    ax.minorticks_off()
    ax.set_xticks(range(len(fams_th)))
    ax.set_xticklabels([t[f].replace("GRU, ", "") for f in fams_th], fontsize=8)
    ax.set_ylabel(t["theta"])
    ax.set_title(t["a"], fontsize=9.5)
    ax.text(0.02, 0.02, t["runs"], transform=ax.transAxes, fontsize=7.5, color="0.35")
    ax.grid(axis="x", visible=False)
    for ax, H, key in ((axes[1], Hs[0], "b"), (axes[2], Hs[1] if len(Hs) > 1 else Hs[0], "c")):
        sf = summary[f"H{H}"]["rq2_safety_factor"]
        vs = sorted(sf, key=float, reverse=True)                 # 1, 0.5, 0.25, 0.1 %
        fac = [a.v_main / float(v) for v in vs]
        for f in ["ridge", "gru_mse", "gru_l4", "gru_l8"]:
            ax.plot(fac, [sf[v][f]["halls_both_ok_pct"] for v in vs], marker=MARK[f], ms=5, lw=1.8 if f != "ridge" else 1.3,
                    color=COLOR[f], ls="--" if f == "ridge" else "-", label=t[f])
        ax.set_xscale("log")
        ax.set_xticks(fac)
        ax.set_xticklabels([f"{num(100 * float(v))} %\n×{num(f)}" for v, f in zip(vs, fac)], fontsize=8)
        ax.minorticks_off()
        ax.set_ylim(0, 100)
        ax.set_ylabel(t["share"])
        ax.set_title(t[key].format(H), fontsize=9.5)
    h, lab = axes[2].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.62, -0.09))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    p1, p2 = axes[1].get_position(), axes[2].get_position()
    fig.text((p1.x0 + p2.x1) / 2, 0.035, t["factor"], ha="center", fontsize=8.5)
    save(fig, "fig_transfer", lang)

    # ---- MAE vs energy, H = 5
    r = runs[Hs[0]]
    trade = pd.read_csv(os.path.join(r, "tradeoff.csv"))
    acc = pd.read_csv(os.path.join(r, "accuracy.csv")).set_index("model")
    E = trade[trade.v_max == a.v_main].groupby("model").E.mean()
    mse_mean = E[[m for m in E.index if fam(m) == "gru_mse"]].mean()
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for f in ["persistence", "ridge", "gru_mse", "gru_l1", "gru_l2", "gru_l4", "gru_l8"]:
        ms = [m for m in E.index if fam(m) == f]
        ax.scatter(acc.loc[ms, "mae"], 100 * (1 - E[ms] / mse_mean), s=24 if len(ms) > 1 else 46,
                   marker=MARK[f], color=COLOR[f], label=t[f], edgecolor="k" if MARK[f] != "x" else None,
                   linewidth=0.3)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel(t["mae"])
    ax.set_ylabel(t["saving"])
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    if lang == "uk":
        from matplotlib.ticker import FuncFormatter
        fmt = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",").replace("-", "−"))
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
    fig.tight_layout()
    save(fig, "fig_accuracy", lang)
print("figures saved to", a.out)
