"""Monte Carlo forward test (Appendix A10). Writes outputs/montecarlo/.

    python scripts/run_montecarlo.py                  # wheel forward test + book (if sleeve outputs exist)
    python scripts/run_montecarlo.py --calibration    # also the walk-forward calibration (~1-2 min)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from book import config, montecarlo as mc, portfolio  # noqa: E402
from book.metrics import save_json  # noqa: E402
from book.sleeves import wheel  # noqa: E402


def wheel_forward(out_dir):
    mkt = wheel.load_market()
    hist, live = wheel.run_wheel(mkt, use_regime=False)
    last_friday = hist.index[-1]
    print(f"Live state at {last_friday.date()}: cash ${live.cash:,.0f}, HYPE {live.qty:.3f}, open option {live.opt}")

    tables, info = [], {}
    for sc in mc.SCENARIOS:
        paths, bh, info = mc.wheel_forward(mkt, live, last_friday, scenario=sc)
        np.save(out_dir / f"wheel_paths_{sc}.npy", paths)
        t = mc.summarise(paths, config.MC_HORIZONS_WEEKS).assign(scenario=sc, asset="wheel")
        tb = mc.summarise(bh, config.MC_HORIZONS_WEEKS).assign(scenario=sc, asset="hype")
        tables += [t, tb]
        print(f"\n--- {sc} ---\n{t.to_string(index=False)}")
    summary = pd.concat(tables)
    summary.to_csv(out_dir / "wheel_forward_summary.csv", index=False)

    # the historical record over the same horizons, for comparison
    eq = hist["equity"]
    backtest_cagr = mc.annualised(np.array([eq.iloc[-1] / eq.iloc[0]]), len(eq) - 1) * 100
    hist_rows = [{"weeks": h, "median%": (eq.shift(-h) / eq - 1).median() * 100,
                  "p5%": (eq.shift(-h) / eq - 1).quantile(0.05) * 100,
                  "P(gain)%": ((eq.shift(-h) / eq - 1).dropna() > 0).mean() * 100,
                  "annualised%": backtest_cagr}
                 for h in config.MC_HORIZONS_WEEKS]
    save_json({"live_state": {"date": str(last_friday.date()), "spot": float(hist["spot"].iloc[-1]), "cash": live.cash, "qty": live.qty,
                              "open_option": list(live.opt) if live.opt else None},
               "n_paths": config.MC_PATHS, "block_days": config.MC_BLOCK_DAYS, "seed": config.MC_SEED,
               "historical": hist_rows, **{k: (float(v) if isinstance(v, (np.floating, float)) else v)
                                           for k, v in info.items()}},
              out_dir / "wheel_forward_info.json")
    return mkt


def book_forward(out_dir):
    """Joint bootstrap of the book's sleeves' weekly returns, if their outputs are present."""
    missing = [str(p) for k, p in portfolio.SLEEVE_FILES.items() if k in config.BOOK_WEIGHTS and not p.exists()]
    if missing:
        print("\nBook forward test skipped, missing sleeve outputs:\n  " + "\n  ".join(missing))
        return
    weekly = portfolio.weekly_returns(list(config.BOOK_WEIGHTS)).dropna()
    print(f"\nBook: {len(weekly)} common weeks {weekly.index[0].date()} to {weekly.index[-1].date()}")
    tables = []
    for sc in mc.SCENARIOS:
        paths = mc.book_forward(weekly, scenario=sc)
        np.save(out_dir / f"book_paths_{sc}.npy", paths)
        t = mc.summarise(paths, config.MC_HORIZONS_WEEKS).assign(scenario=sc)
        tables.append(t)
        print(f"--- book {sc} ---\n{t.to_string(index=False)}")
    pd.concat(tables).to_csv(out_dir / "book_forward_summary.csv", index=False)


if __name__ == "__main__":
    out_dir = config.output_dir("montecarlo")
    mkt = wheel_forward(out_dir)
    book_forward(out_dir)
    if "--calibration" in sys.argv:
        cal = mc.wheel_calibration(mkt)
        cal.to_csv(out_dir / "wheel_calibration.csv", index=False)
        print("\nWalk-forward calibration (wheel):\n" + mc.calibration_summary(cal).to_string())
