"""
Robustness battery for the FINAL book (E1 44% / E2mh 35% / JP hedged 21%, risk parity, VT12):
  1. Engine 1 parameter grid (M, s_open) + winner robustness.
  2. Cost stress: whole book at 20 bps/side.
  3. Sub-period performance (unlevered + VT12).
  4. Stress windows: COVID crash, 2022 bear, Aug 2024 yen shock, Apr 2025 tariff shock.
  5. Asset removal: drop JP, drop E1, drop E2.
  6. Book-level brief metrics: win rate (daily), drawdown episode count, etc.
"""
import warnings, sys, os, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "design-a"))
import numpy as np
import pandas as pd
from engine import run_backtest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "final")
os.makedirs(OUT, exist_ok=True)

def stats(r, label=""):
    r = r.dropna()
    eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
    av = r.std(ddof=1) * np.sqrt(252); ar = r.mean() * 252
    s = {"sharpe": round(float(ar / av), 2) if av > 0 else None,
         "ann_ret%": round(float(ar) * 100, 2), "ann_vol%": round(float(av) * 100, 2),
         "max_dd%": round(float(dd.min()) * 100, 2), "total%": round(float(eq.iloc[-1] - 1) * 100, 1)}
    print(f"  {label:36s} Sh {s['sharpe']:>5}  ret {s['ann_ret%']:>7}%  vol {s['ann_vol%']:>6}%  "
          f"maxDD {s['max_dd%']:>7}%  total {s['total%']}%", flush=True)
    return s

def load_ret(p):
    return pd.read_csv(p, index_col=0, parse_dates=True)["ret"]

def ts_mom_multi(px, lookbacks, vol_win=60, rebal_every=5, cost=10e-4):
    R = px.pct_change().fillna(0.0)
    dates = px.index; T = len(px)
    rets = pd.Series(0.0, index=dates)
    w = np.zeros(px.shape[1]); w_prev = np.zeros(px.shape[1])
    rv = R.rolling(vol_win).std() * np.sqrt(252)
    start = max(max(lookbacks), vol_win) + 1
    for t in range(start, T - 1):
        if (t - start) % rebal_every == 0:
            sig = np.mean([np.sign(px.iloc[t].values / px.iloc[t - lb].values - 1)
                           for lb in lookbacks], axis=0)
            iv = 1.0 / np.where(rv.iloc[t].values > 0.05, rv.iloc[t].values, np.nan)
            raw = np.where(np.isfinite(sig * iv), sig * iv, 0.0)
            w_new = raw / np.abs(raw).sum() if np.abs(raw).sum() > 0 else raw
            rets.iloc[t + 1] = -np.abs(w_new - w_prev).sum() * cost
            w = w_new; w_prev = w_new.copy()
        rets.iloc[t + 1] += float(w @ R.iloc[t + 1].values)
    return rets.loc[dates[start + 1:]]

res = {}
px34 = pd.read_csv(os.path.join(BASE, "design-a", "prices_daily_close.csv"), index_col=0, parse_dates=True)

print("--- 1. Engine 1 parameter grid (net, 10bps) ---", flush=True)
grid = {}
for M, so in [(2, 1.5), (3, 1.25), (3, 1.5), (3, 1.75), (4, 1.5), (5, 1.5)]:
    _, r, *_ = run_backtest(px34, M=M, S_OPEN=so, S_CLOSE=0.50, S_STOP=3.5,
                            HL_MAX=30.0, WIN=60, COST_BPS=10.0)
    grid[f"M{M}_s{so}"] = stats(r, f"E1 M={M} s_open={so}")
res["E1_grid"] = grid

print("--- 2. Cost stress at 20 bps ---", flush=True)
_, r_e1_20, *_ = run_backtest(px34, M=3, S_OPEN=1.5, S_CLOSE=0.50, S_STOP=3.5,
                              HL_MAX=30.0, WIN=60, COST_BPS=20.0)
stats(r_e1_20, "E1 at 20bps")
r_e2_20 = ts_mom_multi(px34, (63, 126, 252), cost=20e-4)
stats(r_e2_20, "E2mh at 20bps")
eq44 = pd.read_csv(os.path.join(BASE, "v2", "prices_eq44.csv"), index_col=0, parse_dates=True)
JP = ["7203.T", "6758.T", "9984.T", "8035.T", "6857.T", "7011.T", "6501.T", "8306.T", "9983.T", "7974.T"]
r_jp_20 = ts_mom_multi(eq44[JP], (126,), cost=20e-4) + 0.03 / 252.0
stats(r_jp_20, "JP hedged at 20bps")
W = {"E1": 0.44, "E2": 0.35, "JP": 0.21}
idx = r_e1_20.index.intersection(r_e2_20.index).intersection(r_jp_20.index)
book20 = W["E1"] * r_e1_20.loc[idx] + W["E2"] * r_e2_20.loc[idx] + W["JP"] * r_jp_20.loc[idx]
res["book20_unlev"] = stats(book20, "BOOK 20bps unlevered")

print("--- 3-5. Final book: sub-periods, stress, removal ---", flush=True)
book = load_ret(os.path.join(BASE, "v3", "returns_book_v3.csv"))
bookvt = load_ret(os.path.join(BASE, "v3", "returns_book_v3_vt12.csv"))
for lab, (a, b) in {"2020-21": ("2020-05-01", "2021-12-31"), "2022": ("2022-01-01", "2022-12-31"),
                    "2023": ("2023-01-01", "2023-12-31"), "2024": ("2024-01-01", "2024-12-31"),
                    "2025": ("2025-01-01", "2025-12-31"), "2026 YTD": ("2026-01-01", "2026-12-31")}.items():
    res[f"sub_{lab}"] = stats(bookvt[(bookvt.index >= a) & (bookvt.index <= b)], f"VT12 {lab}")
for lab, (a, b) in {"COVID crash Feb-Mar 2020": ("2020-02-19", "2020-03-23"),
                    "2022 bear (Jan-Oct)": ("2022-01-03", "2022-10-14"),
                    "Yen shock Aug 2024": ("2024-07-31", "2024-08-07"),
                    "Tariff shock Apr 2025": ("2025-04-02", "2025-04-09")}.items():
    seg = bookvt[(bookvt.index >= a) & (bookvt.index <= b)]
    tot = float((1 + seg).prod() - 1) * 100
    print(f"  {lab:28s} book VT12 return in window: {tot:+.2f}%", flush=True)
    res[f"stress_{lab}"] = {"window%": round(tot, 2)}
E1 = load_ret(os.path.join(BASE, "design-a", "final_returns_net.csv"))
E2 = load_ret(os.path.join(BASE, "v3", "returns_E2mh.csv"))
JPh = load_ret(os.path.join(BASE, "v3", "returns_JP_hedged.csv"))
i2 = E1.index.intersection(E2.index).intersection(JPh.index)
rem = {}
rem["no_JP"] = stats(0.56 * E1.loc[i2] + 0.44 * E2.loc[i2], "Book without Japan (E1/E2 RP)")
rem["no_E1"] = stats(0.63 * E2.loc[i2] + 0.37 * JPh.loc[i2], "Book without Engine 1 (E2/JP RP)")
rem["no_E2"] = stats(0.68 * E1.loc[i2] + 0.32 * JPh.loc[i2], "Book without Engine 2 (E1/JP RP)")
res["removal"] = rem

print("--- 6. Book brief metrics ---", flush=True)
eq = (1 + bookvt).cumprod(); dd = eq / eq.cummax() - 1
in_dd = dd < -0.02
episodes = int(((in_dd) & (~in_dd.shift(1).fillna(False))).sum())
print(f"  VT12 book: daily win rate {(bookvt > 0).mean() * 100:.1f}%, DD episodes (>2% deep): {episodes}, "
      f"max DD {dd.min() * 100:.2f}%, trading days {len(bookvt)}", flush=True)
res["book_metrics"] = {"daily_win%": round(float((bookvt > 0).mean()) * 100, 1),
                       "dd_episodes_2pct": episodes, "days": int(len(bookvt))}
book.to_csv(os.path.join(OUT, "final_book_unlev.csv"), header=["ret"])
with open(os.path.join(OUT, "battery.json"), "w") as f:
    json.dump(res, f, indent=1)
print("\nBATTERY DONE")
