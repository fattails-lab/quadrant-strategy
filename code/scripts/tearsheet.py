"""Tear sheet for a window, per sleeve and for the book (slide 9's Friday report).

    python scripts/tearsheet.py                          # 1 to 30 Oct 2026, the live window
    python scripts/tearsheet.py 2026-10-01 2026-10-30

Daily equity from the sleeves' outputs, so run them first. Writes outputs/book/tearsheet_<start>_<end>.csv
and .json. Trade counts come from the E1 and dip trade logs; the wheel and short IDR have no trade log
beyond their weekly and daily equity.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from book import config, portfolio  # noqa: E402


def trades_in_window(start, end):
    """E1 and dip trades closed inside the window: count, win rate, average P&L."""
    out = {}
    e1 = pd.read_csv(config.OUTPUTS / "e1" / "final_trades.csv", parse_dates=["exit"])
    e1 = e1[(e1["exit"] >= start) & (e1["exit"] <= end)]
    out["e1"] = {"trades": int(len(e1)), "win_rate%": round(float((e1["pnl"] > 0).mean() * 100), 1) if len(e1) else None,
                 "avg_pnl_bps": round(float(e1["pnl"].mean() * 1e4), 1) if len(e1) else None}
    dip = pd.read_csv(config.OUTPUTS / "dip" / "trades_B_100k.csv", parse_dates=["exit_date"])
    dip = dip[(dip["exit_date"] >= start) & (dip["exit_date"] <= end)]
    out["dip"] = {"trades": int(len(dip)), "win_rate%": round(float((dip["net_pnl"] > 0).mean() * 100), 1) if len(dip) else None,
                  "avg_ret%": round(float(dip["ret_pct"].mean()), 2) if len(dip) else None}
    return out


def tearsheet(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    eq = portfolio.daily_equity().dropna()
    w = pd.Series(config.BOOK_WEIGHTS)
    base = eq[eq.index < start].iloc[-1]
    win = eq[(eq.index >= start) & (eq.index <= end)] / base            # growth of 1 from the last close before the window
    win["book"] = (win[w.index] * w).sum(axis=1)                        # fixed weights inside the month, no rebalance
    days = (end - start).days + 1          # calendar days from the close before `start` to `end`, 30 for October
    rows = {}
    for k in win:
        g = win[k]
        r = g.pct_change().dropna()
        dd = (g / g.cummax() - 1).min()
        ret = g.iloc[-1] - 1
        rows[k] = {"weight%": float(w.get(k, np.nan) * 100) if k in w.index else None,
                   "return%": round(ret * 100, 2), "annualised%": round(((1 + ret) ** (365 / days) - 1) * 100, 1),
                   "vol%": round(r.std() * np.sqrt(252) * 100, 1),
                   "sharpe": round(r.mean() / r.std() * np.sqrt(252), 2) if r.std() > 0 else None,
                   "max_dd%": round(dd * 100, 2), "best_day%": round(r.max() * 100, 2), "worst_day%": round(r.min() * 100, 2),
                   "worst_day": str(r.idxmin().date()), "up_days%": round((r > 0).mean() * 100, 0), "trading_days": int(len(r))}
    for k, t in trades_in_window(start, end).items():
        rows[k].update(t)
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    a, b = (sys.argv[1:3] + ["2026-10-01", "2026-10-30"])[:2] if len(sys.argv) >= 3 else ("2026-10-01", "2026-10-30")
    ts = tearsheet(a, b)
    pd.set_option("display.width", 200)
    print(f"Tear sheet {a} to {b}\n")
    print(ts.to_string())
    out = config.output_dir("book")
    ts.to_csv(out / f"tearsheet_{a}_{b}.csv")
    ts.to_json(out / f"tearsheet_{a}_{b}.json", orient="index", indent=1)
