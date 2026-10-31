"""Weekly sleeve returns and the book (slide 7). Reads the sleeves' outputs, so run them first."""
import pandas as pd

from book import config

SLEEVE_FILES = {
    "e1": config.OUTPUTS / "e1" / "final_returns_net.csv",            # daily returns
    "dip": config.OUTPUTS / "dip" / "equity_B_100k.csv",              # daily equity
    "wheel": config.OUTPUTS / "wheel" / "returns_wheel_always.csv",   # weekly equity
    "fx": config.OUTPUTS / "fx" / "equity_daily.csv",                 # daily equity
}
DAILY_FILES = {**SLEEVE_FILES, "wheel": config.OUTPUTS / "wheel" / "equity_daily_always.csv"}


def daily_equity(sleeves=None):
    """Daily equity (growth of 1) per sleeve; the wheel is marked to market with its open option."""
    sleeves = sleeves or [k for k, p in DAILY_FILES.items() if p.exists()]
    cols = {}
    for k in sleeves:
        df = pd.read_csv(DAILY_FILES[k], index_col=0, parse_dates=True)
        cols[k] = (1 + df["ret"]).cumprod() if k == "e1" else df["equity"] / df["equity"].iloc[0]
    return pd.DataFrame(cols)


def window_returns(start, end, sleeves=None):
    """Return of each sleeve from the last close before `start` to the last close on or before `end`."""
    eq = daily_equity(sleeves)
    out = {}
    for k in eq:
        s = eq[k].dropna()
        out[k] = s[s.index <= pd.Timestamp(end)].iloc[-1] / s[s.index < pd.Timestamp(start)].iloc[-1] - 1
    return pd.Series(out)


def weekly_returns(sleeves=None):
    """Friday-to-Friday returns, one column per sleeve. Defaults to every sleeve whose output exists."""
    sleeves = sleeves or [k for k, p in SLEEVE_FILES.items() if p.exists()]
    cols = {}
    for k in sleeves:
        p = SLEEVE_FILES[k]
        if k == "e1":
            r = pd.read_csv(p, index_col=0, parse_dates=True)["ret"]
            cols[k] = (1 + r).resample("W-FRI").prod() - 1
        elif k in ("dip", "fx"):
            eq = pd.read_csv(p, index_col=0, parse_dates=True)["equity"]
            cols[k] = eq.resample("W-FRI").last().pct_change()
        else:
            cols[k] = pd.read_csv(p, index_col=0, parse_dates=True)["equity"].pct_change()
    return pd.DataFrame(cols)


def book_returns(weekly=None, weights=None):
    """Book weekly returns, rebalanced weekly to `weights`, over the weeks every sleeve in it has data."""
    weights = pd.Series(weights or config.BOOK_WEIGHTS, dtype=float)
    weights = weights / weights.sum()
    R = (weekly if weekly is not None else weekly_returns(list(weights.index)))[weights.index].dropna()
    return R @ weights
