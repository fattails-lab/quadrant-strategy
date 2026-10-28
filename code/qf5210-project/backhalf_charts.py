"""Charts for the deck's back half: A2 robustness bars, A7 ES bars, A9 spread comparison."""
import warnings, os
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QP = os.path.dirname(os.path.abspath(__file__))
IR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "idx-report")
COL = {"E1": "#1f3864", "DIP": "#445e2e", "IDX": "#b05a0b", "Wheel": "#5b2d90", "BOOK": "#1f3864"}

# ---------- A2: Sharpe / NW / drop-best grouped bars ----------
# values are read from the current book study (dip_swap.json), never hardcoded,
# so the charts cannot silently go stale against the reported numbers
import json
D = json.load(open(os.path.join(QP, "dip_swap.json")))
names = ["E1", "DIP", "IDX", "Wheel", "BOOK"]
def _bk(k, key):
    return (D["book"] if k == "BOOK" else D["sleeves_common"][k])[key]
raw = [_bk(k, "sharpe") for k in names]
nw = [_bk(k, "nw") for k in names]
drop = [_bk(k, "drop") for k in names]
fig, ax = plt.subplots(figsize=(6.4, 3.1))
x = np.arange(len(names)); wdt = 0.27
for i, (vals, lab, al) in enumerate([(raw, "Sharpe", 1.0), (nw, "Newey–West", 0.62), (drop, "Drop best week", 0.30)]):
    cols = [COL[n] for n in names]
    ax.bar(x + (i - 1) * wdt, vals, wdt, label=lab, color=cols, alpha=al, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("Sharpe", fontsize=10); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
ax.legend(fontsize=8.5, frameon=False, loc="upper left")
ax.set_title("No sleeve lives off its best week", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(QP, "chart_a2.png"), dpi=150); plt.close(fig)

# ---------- A7: ES hist vs EVT horizontal bars ----------
es_h = [D["vares"][k]["ES_hist%"] for k in names]
es_e = [D["vares"][k]["ES_evt%"] for k in names]
fig, ax = plt.subplots(figsize=(7.4, 2.6))
y = np.arange(len(names))[::-1]; hgt = 0.36
for yi, n, a, b in zip(y, names, es_h, es_e):
    ax.barh(yi + hgt / 2, a, hgt, color=COL[n], alpha=0.55)
    ax.barh(yi - hgt / 2, b, hgt, color=COL[n])
    ax.text(a + 0.5, yi + hgt / 2, f"{a:.1f}", va="center", fontsize=9)
    ax.text(b + 0.5, yi - hgt / 2, f"{b:.1f}", va="center", fontsize=9, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel("1-week 99% Expected Shortfall (% of sleeve capital)", fontsize=9.5)
ax.grid(axis="x", alpha=0.3); ax.set_axisbelow(True)
ax.set_title("Expected Shortfall: historical (light) vs EVT tail fit (solid)", fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(QP, "chart_a7.png"), dpi=150); plt.close(fig)

# ---------- A9: spread that comes back vs one that does not ----------
ip = pd.read_csv(os.path.join(IR, "idx_prices.csv"), index_col=0, parse_dates=True)
gp = pd.read_csv(os.path.join(IR, "gold_prices.csv"), index_col=0, parse_dates=True)
a = np.log(ip["ANTM.JK"].dropna()); b = np.log(gp["GC=F"].dropna())
j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
j = j[j.index >= "2016-01-01"]
beta, alpha = np.polyfit(j["b"], j["a"], 1)
spread = j["a"] - (alpha + beta * j["b"])
spread = (spread - spread.mean()) / spread.std()
ss = pd.read_csv(os.path.join(QP, "design-a", "s_scores.csv"), index_col=0, parse_dates=True)["AAPL"].dropna()
fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.05))
axes[0].plot(spread.index, spread.values, color="#b05a0b", lw=0.9)
axes[0].axhline(0, color="gray", lw=0.8, ls=":")
axes[0].set_title("ANTM vs gold spread: never comes back\n(Engle–Granger p = 0.19)", fontsize=9.5)
axes[1].plot(ss.index, ss.values, color="#1f3864", lw=0.7)
axes[1].axhline(0, color="gray", lw=0.8, ls=":")
axes[1].axhline(1.5, color="#c0392b", lw=0.7, ls="--"); axes[1].axhline(-1.5, color="#c0392b", lw=0.7, ls="--")
axes[1].set_title("E1 residual (AAPL s-score): snaps back\n(half-life ≈ 5 days; entry bands ±1.5)", fontsize=9.5)
for ax in axes:
    ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig(os.path.join(QP, "chart_a9.png"), dpi=150); plt.close(fig)
print("chart_a2.png, chart_a7.png, chart_a9.png saved")
