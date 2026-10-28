"""Final headline run for the group-facing prototype + charts + sanity checks."""
import warnings, sys, json
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from engine import run_backtest

SG = ["D05.SI", "O39.SI", "U11.SI", "Z74.SI", "S68.SI", "C09.SI", "F34.SI", "G13.SI", "Y92.SI"]
US34 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "BAC", "WFC", "XOM",
        "CVX", "UNH", "LLY", "HD", "PG", "COST", "V", "MA", "NFLX", "AVGO",
        "CRM", "ORCL", "ABBV", "MRK", "PEP"]
px_big = pd.read_csv("prices_big.csv", index_col=0, parse_dates=True)
px = px_big[[c for c in px_big.columns if c in SG + US34]]
px.to_csv("prices_daily_close.csv")

CFG = dict(M=3, S_OPEN=1.5, S_CLOSE=0.50, S_STOP=3.5, HL_MAX=30.0, WIN=60, COST_BPS=10.0)
out, ret_net, ret_gross, tr, s_hist = run_backtest(px, **CFG)
print("HEADLINE:", json.dumps(out, indent=1))

# yearly breakdown (net)
eq = (1 + ret_net).cumprod()
rows = []
for y, r in ret_net.groupby(ret_net.index.year):
    av = r.std(ddof=1) * np.sqrt(252)
    rows.append({"year": int(y), "ret%": round(float((1 + r).prod() - 1) * 100, 2),
                 "sharpe": round(float(r.mean() * 252 / av), 2) if av > 0 else None,
                 "vol%": round(float(av) * 100, 2)})
yr = pd.DataFrame(rows)
print(yr.to_string(index=False))

# sub-period 2024+
r24 = ret_net[ret_net.index >= "2024-01-01"]
av = r24.std(ddof=1) * np.sqrt(252)
print(f"2024-now: ret {(float((1+r24).prod()-1)*100):.2f}%  Sharpe {float(r24.mean()*252/av):.2f}  vol {float(av)*100:.2f}%")

# ---------- sanity checks ----------
assert ret_net.notna().all(), "NaN in net returns"
recon = float((1 + ret_net).prod())
print(f"Sanity: equity recon {recon:.4f} vs metrics total {out['net']['total%']}%")
# trade-level: every closed trade must have exit >= entry, days >= 1
trc = tr[tr["days"] > 0]
print(f"Sanity: {len(tr)} trades, {len(trc)} with days>=1, bad entries: {int((trc['exit'] < trc['entry']).sum())}")
# SG vs US trade split
tr["sg"] = tr["ticker"].str.endswith(".SI")
print("Trades by market:", tr.groupby("sg").agg(n=("pnl", "size"), win=("pnl", lambda p: round(float((p > 0).mean()) * 100, 1)), avg_bps=("pnl", lambda p: round(float(p.mean()) * 1e4, 1))).to_dict("index"))

# ---------- charts ----------
fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
axes[0].plot(eq.index, eq.values, lw=1.2, color="#1a5276")
axes[0].set_ylabel("Growth of $1")
axes[0].set_title("E1 equity curve (net of costs: 10 bps per weight change), 2019–2026")
axes[0].grid(alpha=0.3)
dd = eq / eq.cummax() - 1
axes[1].fill_between(dd.index, dd.values * 100, 0, color="#c0392b", alpha=0.55)
axes[1].set_ylabel("Drawdown %")
axes[1].grid(alpha=0.3)
fig.savefig("chart_equity.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# s-score example: DBS 2025-2026 with bands + trade shading
s = s_hist["D05.SI"].loc["2025-01-01":"2026-10-09"]
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(s.index, s.values, lw=1.0, color="#1a5276", label="DBS s-score")
for lvl, c, ls in [(1.5, "#c0392b", "--"), (-1.5, "#c0392b", "--"), (0.5, "grey", ":"), (-0.5, "grey", ":")]:
    ax.axhline(lvl, color=c, ls=ls, lw=0.9)
ax.axhline(0, color="black", lw=0.7)
dbs_tr = tr[(tr["ticker"] == "D05.SI") & (tr["entry"] >= "2025-01-01")]
for _, r_ in dbs_tr.iterrows():
    ax.axvspan(pd.Timestamp(r_["entry"]), pd.Timestamp(r_["exit"]),
               color="#f39c12" if r_["dir"] < 0 else "#27ae60", alpha=0.18)
ax.set_ylabel("s-score")
ax.set_title("Example signal — DBS (D05.SI) s-score, 2025–2026  (shaded = open positions, green long / orange short)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)
fig.savefig("chart_sscore.png", dpi=160, bbox_inches="tight")
plt.close(fig)

summary = {"config": CFG, "headline": out, "yearly": rows,
           "universe": list(px.columns)}
with open("final_summary.json", "w") as f:
    json.dump(summary, f, indent=1)
ret_net.to_csv("final_returns_net.csv", header=["ret"])
tr.to_csv("final_trades.csv", index=False)
print("FINAL RUN DONE")
