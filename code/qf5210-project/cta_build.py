"""
CTA futures build (Sleeve 4): multi-horizon trend, 9 futures markets, weekly execution.
Signal per market: mean of sign(63d), sign(126d), sign(252d) returns.
Sizing: risk parity per market: pos_i = signal_i * K / (sigma_i * sqrt(N)), K = 0.11 fixed ex ante,
sigma_i = 60d annualised vol floored at 5%. Rebalance Fridays, 3 bps on notional turnover.
"""
import warnings, os, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
px = pd.read_csv(os.path.join(BASE, "cta_futures_prices.csv"), index_col=0, parse_dates=True)
MKTS = ["ES=F", "ZN=F", "6E=F", "6J=F", "GC=F", "SI=F", "CL=F", "ZC=F", "NG=F"]
P = px[MKTS].ffill()
ret = P.pct_change().fillna(0.0)
N = len(MKTS)
K = 0.11

def signal_series(prices):
    sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for h in (63, 126, 252):
        sig += np.sign(prices / prices.shift(h) - 1.0)
    return (sig / 3.0).fillna(0.0)

SIG = signal_series(P)
VOL = (ret.rolling(60).std() * np.sqrt(252)).clip(lower=0.05)

fridays = P.resample("W-FRI").last().index
fridays = [f for f in fridays if f >= pd.Timestamp("2016-01-01")]
daily_pnl = pd.Series(0.0, index=P.index)
pos_prev = pd.Series(0.0, index=MKTS)
weekly_rows = []
for i in range(1, len(fridays)):
    f0, f1 = fridays[i - 1], fridays[i]
    sig = SIG.loc[:f0].iloc[-1]
    vol = VOL.loc[:f0].iloc[-1]
    pos = sig * K / (vol * np.sqrt(N))
    turn = float((pos - pos_prev).abs().sum())
    cost = 0.0003 * turn
    days = ret.loc[(ret.index > f0) & (ret.index <= f1)]
    pnl = float((days * pos).sum().sum()) - cost
    weekly_rows.append({"week": f1, "ret": pnl, "turnover": turn})
    pos_prev = pos
cta = pd.DataFrame(weekly_rows).set_index("week")["ret"]
cta.to_csv(os.path.join(BASE, "returns_CTA.csv"))
print(f"CTA weeks: {len(cta)}, {cta.index[0].date()} -> {cta.index[-1].date()}, "
      f"avg weekly turnover {pd.DataFrame(weekly_rows)['turnover'].mean():.2f}x")

def stats(r, label):
    r = r.dropna()
    eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
    av = r.std(ddof=1) * np.sqrt(52); ar = r.mean() * 52
    print(f"  {label:22s} Sh {ar/av:5.2f}  CAGR {ar*100:6.2f}%  vol {av*100:5.2f}%  "
          f"maxDD {dd.min()*100:6.2f}%  total {(eq.iloc[-1]-1)*100:8.1f}%")
    return {"sharpe": round(float(ar / av), 2), "cagr%": round(float(ar) * 100, 2),
            "vol%": round(float(av) * 100, 2), "max_dd%": round(float(dd.min()) * 100, 2),
            "total%": round(float(eq.iloc[-1] - 1) * 100, 1)}

res = {}
print("\n--- CTA futures ---")
res["full"] = stats(cta, "Full 2016-2026")
res["since2020"] = stats(cta[cta.index >= "2020-01-01"], "Since 2020")
res["since2024"] = stats(cta[cta.index >= "2024-01-01"], "Since 2024")
print("--- by year ---")
for y in range(2016, 2027):
    ry = cta[cta.index.year == y]
    if len(ry) > 5:
        print(f"  {y}: total {(np.prod(1+ry)-1)*100:7.1f}%   Sharpe {ry.mean()/ry.std()*np.sqrt(52):5.2f}")
print("--- stress windows ---")
for nm, a, b in [("COVID crash 19Feb-23Mar20", "2020-02-19", "2020-03-23"),
                 ("2022 bear (full year)", "2022-01-01", "2022-12-31"),
                 ("Aug 2024 yen shock", "2024-07-26", "2024-08-09"),
                 ("Apr 2025 tariff shock", "2025-04-01", "2025-04-30")]:
    w = cta[(cta.index > a) & (cta.index <= b)]
    print(f"  {nm}: {(np.prod(1+w)-1)*100:+.1f}% over {len(w)} weeks")
# correlations on common windows
e1 = pd.read_csv(os.path.join(BASE, "design-a", "final_returns_net.csv"), index_col=0, parse_dates=True).iloc[:, 0]
e1w = (1 + e1).resample("W-FRI").prod() - 1
j = pd.concat([cta.rename("CTA"), e1w.rename("E1")], axis=1, join="inner").dropna()
print(f"\ncorr(CTA, E1) weekly common window {j.index[0].date()}->{j.index[-1].date()}: {j.corr().iloc[0,1]:.2f}")
spx = px["ES=F"].resample("W-FRI").last().pct_change()
j2 = pd.concat([cta.rename("CTA"), spx.rename("ES")], axis=1, join="inner").dropna()
print(f"corr(CTA, S&P futures) full: {j2.corr().iloc[0,1]:.2f}")
with open(os.path.join(BASE, "cta_summary.json"), "w") as f:
    json.dump(res, f, indent=1)
print("\nCTA BUILD DONE")
