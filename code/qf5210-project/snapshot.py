"""Live-book snapshot: what both engines hold as of the latest data date."""
import warnings, sys, json, os
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "design-a"))
import numpy as np
import pandas as pd
from engine import run_backtest

BASE = os.path.dirname(os.path.abspath(__file__))
px = pd.read_csv(os.path.join(BASE, "design-a", "prices_daily_close.csv"),
                 index_col=0, parse_dates=True)
asof = px.index[-1]
print("As of:", asof.date())

# Engine 1: Design A final positions (headline config)
out, *_ = run_backtest(px, M=3, S_OPEN=1.5, S_CLOSE=0.50, S_STOP=3.5, HL_MAX=30.0, WIN=60, COST_BPS=10.0)
fp = out["final_positions"]
print(f"\nENGINE 1 (stat arb): {len(fp)} open positions")
for p in fp:
    print(f"  {'LONG ' if p['dir']>0 else 'SHORT'} {p['ticker']:8s} weight {p['weight%']:>5}%  s={p['s']}")

# Engine 2: TS momentum current book (63/126/252d sign blend, as backtested in
# cta_build.py; vol-scaled, gross 1)
R = px.pct_change()
sig = sum(np.sign(px.iloc[-1].values / px.iloc[-k - 1].values - 1) for k in (63, 126, 252)) / 3
rv = (R.rolling(60).std() * np.sqrt(252)).iloc[-1].values
iv = np.where(rv > 0.05, 1.0 / np.where(rv > 0.05, rv, 1.0), 0.0)
raw = sig * iv
w = raw / np.abs(raw).sum()
ts_book = [{"ticker": t, "dir": int(np.sign(s)), "weight%": round(float(abs(wi)) * 100, 1),
            "ret126%": round(float(px.iloc[-1][t] / px.iloc[-127][t] - 1) * 100, 1)}
           for t, s, wi in zip(px.columns, sig, w) if wi != 0]
ts_book.sort(key=lambda d: -d["weight%"])
print(f"\nENGINE 2 (TS momentum): {sum(1 for b in ts_book if b['dir']>0)} long / "
      f"{sum(1 for b in ts_book if b['dir']<0)} short")
for b in ts_book:
    print(f"  {'LONG ' if b['dir']>0 else 'SHORT'} {b['ticker']:8s} weight {b['weight%']:>5}%  6m ret {b['ret126%']:+.1f}%")

snap = {"as_of": str(asof.date()), "engine1_positions": fp, "engine2_book": ts_book}
with open(os.path.join(BASE, "snapshot.json"), "w") as f:
    json.dump(snap, f, indent=1)
print("\nSNAPSHOT SAVED")
