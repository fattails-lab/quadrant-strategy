"""
Strategy 2 WEEKLY variant: gate checked at each week-end (Friday) close; baskets rebalanced
to equal weights weekly while invested; same baskets, gates, and 25bps cost model as the monthly run.
Outputs every number the group report needs + a fresh equity chart.
"""
import warnings, os, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
BASKETS = {
    "Coal": (["ADRO.JK", "PTBA.JK", "ITMG.JK", "BYAN.JK"], "BTU"),
    "CPO": (["AALI.JK", "LSIP.JK", "TAPG.JK"], "ZL=F"),
    "Nickel": (["ANTM.JK", "INCO.JK", "MDKA.JK", "NCKL.JK", "MBMA.JK"], "JJN"),
}
px = pd.read_csv(os.path.join(OUT, "idx_prices.csv"), index_col=0, parse_dates=True)
wk = px.resample("W-FRI").last()
weeks = wk.index

def run_basket(names, gate=None):
    port_val = 1.0; holdings = {}
    out = []
    for i in range(1, len(weeks)):
        rp, rn = wk.loc[weeks[i - 1]], wk.loc[weeks[i]]
        total_prev = port_val
        val_hold = sum(sh * rp[t] for t, sh in holdings.items() if pd.notna(rp.get(t)))
        on = True if gate is None else bool(gate.iloc[i - 1])
        avail = [n for n in names if pd.notna(rp.get(n)) and rp.get(n, 0) > 0]
        if on and avail:
            cur = {t: (holdings.get(t, 0.0) * rp[t] if pd.notna(rp.get(t)) else 0.0)
                   for t in set(holdings) | set(avail)}
            tgt0 = {n: total_prev / len(avail) for n in avail}
            turn = sum(abs(tgt0.get(t, 0.0) - cur.get(t, 0.0)) for t in set(tgt0) | set(cur))
            cost = 0.0025 * turn
            investable = max(total_prev - cost, 0.0)
            holdings = {n: (investable / len(avail)) / rp[n] for n in avail}
            cash_after = 0.0
        else:
            cost = 0.0025 * val_hold
            holdings = {}
            cash_after = total_prev - cost
        val_now = cash_after
        for t, sh in holdings.items():
            p = rn.get(t)
            if pd.isna(p):
                p = rp.get(t, 0.0)
            val_now += sh * p
        out.append({"week": weeks[i], "ret": val_now / total_prev - 1 if total_prev > 0 else 0.0,
                    "invested": 1.0 if holdings else 0.0})
        port_val = val_now
    df = pd.DataFrame(out).set_index("week")
    return df["ret"], df["invested"]

def gate_proxy(proxy):
    s = px[proxy].dropna()
    g = (s > s.rolling(200).mean())
    return g.resample("W-FRI").last().reindex(weeks).ffill().fillna(False)

def gate_own(names):
    # normalised equal-weight index built from weekly returns; averaging raw
    # share prices would jump whenever a member enters or leaves the basket
    lvl = (1 + wk[names].pct_change().mean(axis=1).fillna(0)).cumprod()
    return (lvl > lvl.rolling(40).mean()).fillna(False)  # 40-week ~ 10-month

def stats_w(r, label=""):
    r = r.dropna()
    eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
    av = r.std(ddof=1) * np.sqrt(52); ar = r.mean() * 52
    s = {"sharpe": round(float(ar / av), 2) if av > 0 else None, "cagr%": round(float(ar) * 100, 2),
         "vol%": round(float(av) * 100, 2), "max_dd%": round(float(dd.min()) * 100, 2),
         "total%": round(float(eq.iloc[-1] - 1) * 100, 1)}
    print(f"  {label:34s} Sh {s['sharpe']:>5}  CAGR {s['cagr%']:>6}%  vol {s['vol%']:>5}%  "
          f"maxDD {s['max_dd%']:>7}%  total {s['total%']}%", flush=True)
    return s

res = {}; gated = {}; ungated = {}; cur = {}
for name, (members, proxy) in BASKETS.items():
    if proxy and proxy in px.columns and px[proxy].dropna().shape[0] > 250:
        g = gate_proxy(proxy); src = proxy
    else:
        g = gate_own(members); src = "own-trend 40wk"
    rg, inv = run_basket(members, gate=g)
    ru, _ = run_basket(members, gate=None)
    rg = rg[rg.index >= "2016-01-01"]; ru = ru[ru.index >= "2016-01-01"]; inv = inv[inv.index >= "2016-01-01"]
    res[f"{name}_gated"] = stats_w(rg, f"{name} GATED ({src})")
    res[f"{name}_ungated"] = stats_w(ru, f"{name} ungated")
    res[f"{name}_invested%"] = round(float(inv.mean()) * 100, 1)
    print(f"    invested {inv.mean()*100:.0f}% of weeks")
    gated[name] = rg; ungated[name] = ru
    cur[name] = "ON" if bool(g.iloc[-1]) else "OFF"
    print(f"    CURRENT {name}: {cur[name]}")

book_g = pd.concat(gated.values(), axis=1).mean(axis=1)
book_u = pd.concat(ungated.values(), axis=1).mean(axis=1)
res["Book_gated"] = stats_w(book_g, "Combined 3-sleeve GATED")
res["Book_ungated"] = stats_w(book_u, "Combined 3-sleeve ungated")
ihsg = wk["^JKSE"].pct_change().dropna(); ihsg = ihsg[ihsg.index >= "2016-01-01"]
res["IHSG"] = stats_w(ihsg, "IHSG")
gro = pd.concat([ungated["Coal"], ungated["Nickel"]], axis=1).mean(axis=1)
bal = pd.concat([gated["Coal"], ungated["Nickel"]], axis=1).mean(axis=1)
res["Growth"] = stats_w(gro, "GROWTH (coal+nickel ungated)")
res["Balanced"] = stats_w(bal, "BALANCED (coal gated+nickel)")
for lab, (a, b) in {"2020": ("2020-01-01", "2020-12-31"), "2021-22": ("2021-01-01", "2022-12-31"),
                    "2023-24": ("2023-01-01", "2024-12-31"), "2025-26": ("2025-01-01", "2026-12-31")}.items():
    res[f"bookg_{lab}"] = stats_w(book_g[(book_g.index >= a) & (book_g.index <= b)], f"Gated book {lab}")

fig, ax = plt.subplots(figsize=(9, 4.2))
for s, lab, c in [(book_g, "Commodity book, gated (weekly trigger)", "#229954"),
                  (book_u, "Commodity book, ungated", "#b9770e"), (ihsg, "IHSG", "#1a5276")]:
    eq = (1 + s).cumprod()
    ax.plot(eq.index, eq.values, lw=1.1, color=c, label=lab)
ax.set_yscale("log"); ax.set_ylabel("Growth of Rp1 (log scale)")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
ax.set_title("Indonesian commodity baskets vs IHSG, 2016-2026 (weekly, net of costs)")
fig.savefig(os.path.join(OUT, "chart_idx_weekly.png"), dpi=160, bbox_inches="tight")
print("chart saved")
book_g.to_csv(os.path.join(OUT, "strat2_book_weekly.csv"), header=["ret"])
gro.to_csv(os.path.join(OUT, "strat2_growth_weekly.csv"), header=["ret"])
with open(os.path.join(OUT, "strat2_weekly_results.json"), "w") as f:
    json.dump({"stats": res, "current": cur}, f, indent=1)
print("\nWEEKLY DONE")
