"""Performance metrics and small output helpers shared by every sleeve."""
import csv
import json

import numpy as np


def perf_stats(returns, periods_per_year):
    """Annualised return, vol, Sharpe, max drawdown and total return of a periodic return series.

    ann_ret is the arithmetic mean return times periods per year. It is the "CAGR" column on
    slides 5, 6 and 8.
    """
    r = returns.dropna()
    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1
    ann_ret = r.mean() * periods_per_year
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    return {
        "sharpe": round(float(ann_ret / ann_vol), 2) if ann_vol > 0 else None,
        "ann_ret%": round(float(ann_ret) * 100, 2),
        "ann_vol%": round(float(ann_vol) * 100, 2),
        "max_dd%": round(float(drawdown.min()) * 100, 2),
        "total%": round(float(equity.iloc[-1] - 1) * 100, 2),
    }


def report(returns, label, periods_per_year=52):
    """perf_stats plus a one-line printout."""
    s = perf_stats(returns, periods_per_year)
    print(f"  {label:40s} Sh {s['sharpe']!s:>5}  ret {s['ann_ret%']:>6}%  vol {s['ann_vol%']:>5}%  "
          f"maxDD {s['max_dd%']:>7}%  total {s['total%']}%")
    return s


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def write_rows(rows, path):
    """Write a list of dicts to CSV (empty file if there are no rows)."""
    with open(path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
