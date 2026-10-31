"""Return of each sleeve and of the book over a live window, annualised. Run the sleeves first.

    python scripts/evaluate_window.py                                  # October 2026
    python scripts/evaluate_window.py --start 2026-10-14 --end 2026-10-30

Calendar window, from daily closes: from the last close before --start to the last close on or before
--end. The wheel is marked to market daily, open option included. The book holds config.BOOK_WEIGHTS at
the start of the window (it rebalances quarterly, so not inside a month).
Annualised = (1 + r) ** (365 / days) - 1, days being the calendar days between the two closes.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from book import config  # noqa: E402
from book.metrics import save_json  # noqa: E402
from book.portfolio import window_returns  # noqa: E402


def evaluate(start, end, weights=None):
    weights = pd.Series(weights or config.BOOK_WEIGHTS, dtype=float)
    r = window_returns(start, end)
    r["book"] = (r[weights.index] * weights).sum() / weights.sum()
    days = (min(pd.Timestamp(end), pd.Timestamp.today().normalize()) - pd.Timestamp(start)).days + 1
    out = pd.DataFrame({"return%": r * 100, "annualised%": ((1 + r) ** (365 / days) - 1) * 100,
                        "weight%": (weights / weights.sum() * 100).reindex(r.index)}).round(2)
    return out, days


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-10-01")
    ap.add_argument("--end", default="2026-10-30")
    a = ap.parse_args()
    out, days = evaluate(a.start, a.end)
    print(f"{a.start} to {a.end} ({days} days)\n")
    print(out.to_string())
    save_json({"start": a.start, "end": a.end, "days": days, "summary": out.to_dict(orient="index")},
              config.output_dir("book") / f"window_{a.start}_{a.end}.json")
