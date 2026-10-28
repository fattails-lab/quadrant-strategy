"""
Strategy 2 backtest: commodity-gated Indonesian equity baskets (monthly).
  Coal basket (ADRO, PTBA, ITMG, BYAN) gated on Peabody (BTU) 200d trend.
  CPO basket (AALI, LSIP, TAPG) gated on soybean oil futures (ZL=F) 200d trend (CPO proxy).
  Nickel basket (ANTM, INCO, MDKA + NCKL, MBMA when listed) gated on nickel proxy if available, else own-basket trend.
Baskets: monthly-rebalanced equal weight with share accounting; gate set at month-end, earns next month.
Costs 25 bps per side on turnover. Benchmarks: ungated baskets, IHSG (^JKSE). All returns IDR.
"""
import warnings, os, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

BASKETS = {
    "Coal": (["ADRO.JK", "PTBA.JK", "ITMG.JK", "BYAN.JK"], "BTU"),
    "CPO": (["AALI.JK", "LSIP.JK", "TAPG.JK"], "ZL=F"),
    "Nickel": (["ANTM.JK", "INCO.JK", "MDKA.JK", "NCKL.JK", "MBMA.JK"], "JJN"),
}
tickers = sorted({t for v in BASKETS.values() for t in v[0]} | {v[1] for v in BASKETS.values()} | {"^JKSE"})
print("Downloading:", tickers, flush=True)
px = yf.download(tickers, start="2015-01-01", end="2026-10-10", progress=False, auto_adjust=True)["Close"]
px.index = pd.to_datetime(px.index).tz_localize(None)
for t in tickers:
    if t not in px.columns or px[t].dropna().empty:
        print(f"  MISSING: {t}")
print(f"Panel: {len(px)} rows, {px.index[0].date()} -> {px.index[-1].date()}", flush=True)
px.to_csv(os.path.join(OUT, "idx_prices.csv"))

# month-end prices
me = px.resample("M").last()
months = me.index

# Clean implementation with explicit portfolio value accounting:
def run_basket(names, gate=None):
    """gate: Series indexed by month-end with 0/1 (signal at end of m for month m+1), or None = always on."""
    port_val = 1.0; holdings = {}  # ticker -> shares
    out = []
    for i in range(1, len(months)):
        m_prev, m_now = months[i - 1], months[i]
        rp, rn = me.loc[m_prev], me.loc[m_now]
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
        out.append({"month": m_now, "ret": val_now / total_prev - 1 if total_prev > 0 else 0.0,
                    "invested": 1.0 if holdings else 0.0})
        port_val = val_now
    df = pd.DataFrame(out).set_index("month")
    return df["ret"], df["invested"]

def gate_from_proxy(proxy, basket_names=None):
    """200-day trend gate on daily proxy prices, sampled at month-ends."""
    if proxy in px.columns and px[proxy].dropna().shape[0] > 250:
        s = px[proxy].dropna()
        ma = s.rolling(200).mean()
        g = (s > ma)
        return g.resample("M").last().reindex(months).ffill().fillna(False), proxy
    # fallback: basket's own month-end price index trend (10-month MA on EW index proxy)
    idx_level = (1 + me[basket_names].pct_change().mean(axis=1).fillna(0)).cumprod()  # normalised EW index (raw price average jumps on basket changes)
    g = idx_level > idx_level.rolling(10).mean()
    return g.fillna(False), "own-basket trend (proxy unavailable)"

def stats_m(r, label=""):
    r = r.dropna()
    eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
    av = r.std(ddof=1) * np.sqrt(12); ar = r.mean() * 12
    s = {"sharpe": round(float(ar / av), 2) if av > 0 else None, "cagr%": round(float(ar) * 100, 2),
         "vol%": round(float(av) * 100, 2), "max_dd%": round(float(dd.min()) * 100, 2),
         "total%": round(float(eq.iloc[-1] - 1) * 100, 1)}
    print(f"  {label:34s} Sh {s['sharpe']:>5}  CAGR {s['cagr%']:>6}%  vol {s['vol%']:>5}%  "
          f"maxDD {s['max_dd%']:>7}%  total {s['total%']}%", flush=True)
    return s

res = {}
gated_series = {}
for name, (members, proxy) in BASKETS.items():
    g, used = gate_from_proxy(proxy, members)
    print(f"{name}: gate source = {used}")
    rg, inv = run_basket(members, gate=g)
    ru, _ = run_basket(members, gate=None)
    rg = rg[rg.index >= "2016-01-01"]; ru = ru[ru.index >= "2016-01-01"]
    inv = inv[inv.index >= "2016-01-01"]
    res[f"{name}_gated"] = stats_m(rg, f"{name} GATED ({used})")
    res[f"{name}_ungated"] = stats_m(ru, f"{name} buy & hold basket")
    res[f"{name}_invested%"] = round(float(inv.mean()) * 100, 1)
    print(f"    time invested: {inv.mean()*100:.0f}%")
    gated_series[name] = rg

book = pd.concat(gated_series.values(), axis=1).mean(axis=1)
res["Book"] = stats_m(book, "COMBINED gated book (EW sleeves)")
ihsg = me["^JKSE"].pct_change().dropna()
ihsg = ihsg[ihsg.index >= "2016-01-01"]
res["IHSG"] = stats_m(ihsg, "IHSG buy & hold")
for lab, (a, b) in {"2020 COVID yr": ("2020-01-01", "2020-12-31"), "2021-22 supercycle": ("2021-01-01", "2022-12-31"),
                    "2023-24 normalisation": ("2023-01-01", "2024-12-31"), "2025-26": ("2025-01-01", "2026-12-31")}.items():
    res[f"book_{lab}"] = stats_m(book[(book.index >= a) & (book.index <= b)], f"Book {lab}")
# current gates
cur = {}
for name, (members, proxy) in BASKETS.items():
    g, used = gate_from_proxy(proxy, members)
    cur[name] = {"gate": "ON" if bool(g.iloc[-1]) else "OFF", "source": used,
                 "members": members}
    print(f"CURRENT {name}: {cur[name]['gate']} ({used})")
book.to_csv(os.path.join(OUT, "strat2_book_monthly.csv"), header=["ret"])
with open(os.path.join(OUT, "strat2_results.json"), "w") as f:
    json.dump({"stats": res, "current": cur}, f, indent=1)
print("\nSTRAT2 DONE")
