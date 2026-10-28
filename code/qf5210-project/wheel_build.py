"""
BTC options wheel backtest (Sleeve 2), synthetic options priced off Deribit DVOL.
Weekly cycle (Fridays). State machine per the group's design:
  regime: bull if BTC close >= 200-day MA else bear.
  holding BTC: bull -> sell 1-week 25-delta covered call; bear -> hold, no option.
  flat:        bear -> sell 1-week 25-delta cash-secured put (full cash deployment);
               bull -> stay in cash.
Options priced with Black-Scholes at sigma = DVOL (30d IV; see limitations), T = 7/365,
premium haircut 5% (bid side). Cash earns ^IRX. Fractional BTC. Start $100k, Apr 2021.
Ablation: always-wheel (regime ignored). Benchmark: BTC buy and hold.
"""
import warnings, os, json, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
from scipy.optimize import brentq

BASE = os.path.dirname(os.path.abspath(__file__))

# ---------- DVOL ----------
dvol_path = os.path.join(BASE, "dvol_daily.csv")
if not os.path.exists(dvol_path):
    rows = []
    start = int(pd.Timestamp("2021-03-20").timestamp() * 1000)
    end = int(pd.Timestamp("2026-10-10").timestamp() * 1000)
    step = 150 * 24 * 3600 * 1000
    t = start
    while t < end:
        url = ("https://www.deribit.com/api/v2/public/get_volatility_index_data"
               f"?currency=BTC&start_timestamp={t}&end_timestamp={min(t + step, end)}&resolution=1D")
        d = requests.get(url, timeout=30).json()["result"]["data"]
        rows += d
        t += step
        time.sleep(0.3)
    dv = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])
    dv["date"] = pd.to_datetime(dv["ts"], unit="ms").dt.normalize()
    dv = dv.drop_duplicates("date").set_index("date")["close"]
    dv.to_csv(dvol_path)
    print(f"DVOL fetched: {len(dv)} days {dv.index[0].date()} -> {dv.index[-1].date()}")
else:
    dv = pd.read_csv(dvol_path, index_col=0, parse_dates=True).iloc[:, 0]
    print(f"DVOL loaded: {len(dv)} days")

px = pd.read_csv(os.path.join(BASE, "cta_futures_prices.csv"), index_col=0, parse_dates=True)
btc = px["BTC-USD"].dropna()
irx = px["^IRX"].ffill() / 100.0
ma200 = btc.rolling(200).mean()
dvol_d = dv.reindex(btc.index).ffill()

def bs_price(S, K, T, r, sig, kind):
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def strike_for_delta(S, T, r, sig, target, kind):
    def delta(K):
        d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
        return norm.cdf(d1) - 1 if kind == "put" else norm.cdf(d1)
    lo, hi = (S * 0.3, S) if kind == "put" else (S, S * 3.0)
    return brentq(lambda K: delta(K) - target, lo, hi, xtol=S * 1e-6)

def run_wheel(use_regime=True):
    fridays = btc.resample("W-FRI").last().index
    fridays = [f for f in fridays if f >= pd.Timestamp("2021-04-02") and pd.notna(dvol_d.get(f, np.nan))
               and pd.notna(ma200.get(f, np.nan))]
    cash, qty, opt = 100_000.0, 0.0, None
    stats = {"puts_sold": 0, "puts_assigned": 0, "calls_sold": 0, "calls_away": 0, "premium": 0.0}
    rows = []
    for i, f in enumerate(fridays):
        S = float(btc.loc[:f].iloc[-1])
        r = float(irx.loc[:f].iloc[-1])
        # settle last week's option at today's close
        if opt is not None:
            kind, K, q, _ = opt
            if kind == "put" and S < K:
                cash -= K * q; qty += q; stats["puts_assigned"] += 1
            elif kind == "call" and S > K:
                cash += K * q; qty = 0.0; stats["calls_away"] += 1
            opt = None
        # weekly interest on cash
        cash *= (1 + r * 7 / 365)
        bull = S >= float(ma200.loc[:f].iloc[-1])
        sig = float(dvol_d.loc[:f].iloc[-1]) / 100.0
        T = 7 / 365
        if qty > 0:
            if bull or not use_regime:
                K = strike_for_delta(S, T, r, sig, 0.25, "call")
                prem = bs_price(S, K, T, r, sig, "call") * 0.95 * qty
                cash += prem; stats["premium"] += prem; stats["calls_sold"] += 1
                opt = ("call", K, qty, f)
        else:
            if (not bull) or not use_regime:
                K = strike_for_delta(S, T, r, sig, -0.25, "put")
                q = cash / K
                prem = bs_price(S, K, T, r, sig, "put") * 0.95 * q
                cash += prem; stats["premium"] += prem; stats["puts_sold"] += 1
                opt = ("put", K, q, f)
        rows.append({"week": f, "equity": cash + qty * S, "cash": cash, "qty": qty,
                     "btc": S, "holding": qty > 0})
    return pd.DataFrame(rows).set_index("week"), stats

def stats_block(eq, label):
    r = eq.pct_change().dropna()
    e = eq / eq.iloc[0]
    dd = e / e.cummax() - 1
    av = r.std(ddof=1) * np.sqrt(52); ar = r.mean() * 52
    print(f"  {label:26s} Sh {ar/av:5.2f}  CAGR {ar*100:6.2f}%  vol {av*100:5.2f}%  "
          f"maxDD {dd.min()*100:6.2f}%  final ${eq.iloc[-1]:,.0f}")
    return {"sharpe": round(float(ar / av), 2), "cagr%": round(float(ar) * 100, 2),
            "vol%": round(float(av) * 100, 2), "max_dd%": round(float(dd.min()) * 100, 2),
            "final$": round(float(eq.iloc[-1]))}

res = {}
print("\n--- BTC wheel (synthetic, DVOL-priced) ---")
wdf, wst = run_wheel(True)
res["wheel"] = stats_block(wdf["equity"], "Regime wheel (group design)")
print(f"    puts sold {wst['puts_sold']}, assigned {wst['puts_assigned']} "
      f"({wst['puts_assigned']/max(wst['puts_sold'],1)*100:.0f}%); calls sold {wst['calls_sold']}, "
      f"called away {wst['calls_away']}; premium collected ${wst['premium']:,.0f}; "
      f"weeks holding BTC {wdf['holding'].mean()*100:.0f}%")
adf, ast = run_wheel(False)
res["wheel_always"] = stats_block(adf["equity"], "Always wheel (no regime)")
print("    ALWAYS premium collected ${:,.0f}; puts sold {}, assigned {}; calls sold {}, away {}".format(
    ast["premium"], ast["puts_sold"], ast["puts_assigned"], ast["calls_sold"], ast["calls_away"]))
fr = wdf.index
bh = 100_000 * wdf["btc"] / wdf["btc"].iloc[0]
res["bh"] = stats_block(bh, "BTC buy and hold")
print("--- by year (total return) ---")
for nm, eq in [("wheel", wdf["equity"]), ("always", adf["equity"]), ("B&H", bh)]:
    line = "  " + nm.ljust(7)
    for y in range(2021, 2027):
        e = eq[eq.index.year == y]
        if len(e) > 3:
            line += f" {y}: {(e.iloc[-1]/e.iloc[0]-1)*100:+6.1f}%"
    print(line)
wdf["equity"].to_csv(os.path.join(BASE, "returns_wheel.csv"))
adf["equity"].to_csv(os.path.join(BASE, "returns_wheel_always.csv"))
with open(os.path.join(BASE, "wheel_summary.json"), "w") as fo:
    json.dump({"metrics": res, "trade_stats": wst, "trade_stats_always": ast}, fo, indent=1)
print("\nWHEEL BUILD DONE")
