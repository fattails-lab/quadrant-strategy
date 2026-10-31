"""Sleeve 4 · Short IDR (long USD/IDR), added 31 Oct 2026.

Short the rupiah against the dollar at 1x notional, held through rolling one-month forwards, but only while
the dollar is rising: the position is on when the dollar index's config.FX_MOM_DAYS-day return is positive
at the close, and applied from the next day. The forward costs the gap between Indonesian and US short rates
(config.FX_CARRY a year, charged daily while on), config.FX_ROLL_BPS at each monthly roll, and
config.FX_SWITCH_BPS at each entry or exit. Returns are in USD; idle cash earns nothing.

The premium is the other side of EM carry: the rupiah has weakened against the dollar in most years, but
the forward prices that in, so an always-on short earns about zero after carry. The rupiah falls fastest
when the dollar is already strong, which is the only time the sleeve pays the carry.
"""
import numpy as np
import pandas as pd

from book import config
from book.metrics import report, save_json


def load_fx():
    px = pd.read_csv(config.data_dir("fx") / config.FX_PRICE_FILE, index_col=0, parse_dates=True)
    fx = px[config.FX_TICKER].dropna()
    return fx[fx.pct_change().abs().fillna(0) < 0.05]   # drop the odd bad tick in Yahoo's IDR series


def load_dxy():
    px = pd.read_csv(config.data_dir("fx") / config.FX_DXY_FILE, index_col=0, parse_dates=True)
    return px[config.FX_DXY_TICKER].dropna()


def position(fx, dxy, days=config.FX_MOM_DAYS):
    """1 while the dollar index's `days`-day return is positive at the close, applied the next day; else 0."""
    on = (dxy.pct_change(days) > 0).astype(float)
    return on.reindex(fx.index, method="ffill").fillna(0).shift(1).fillna(0)


def daily_returns(fx, pos):
    gap = fx.index.to_series().diff().dt.days.fillna(0)
    r = pos * (fx.pct_change().fillna(0) - config.FX_CARRY * gap / 365)
    roll = fx.index.to_series().dt.month.diff().fillna(0) != 0
    r[roll] -= pos[roll] * config.FX_ROLL_BPS / 1e4
    return r - pos.diff().abs().fillna(0) * config.FX_SWITCH_BPS / 1e4


def daily_equity(fx, pos, start=config.FX_START, capital=config.FX_CAPITAL):
    r = daily_returns(fx, pos)
    r = r[r.index >= pd.Timestamp(start)]
    r.iloc[0] = 0.0
    return capital * (1 + r).cumprod()


def run():
    """Short IDR with the dollar filter from 2015; outputs in outputs/fx/."""
    out_dir = config.output_dir("fx")
    fx, dxy = load_fx(), load_dxy()
    pos = position(fx, dxy)
    eq = daily_equity(fx, pos)
    always = daily_equity(fx, pd.Series(1.0, fx.index))
    weekly = eq.resample("W-FRI").last()
    res = {"short_idr": report(weekly.pct_change(), "Short IDR, dollar filter"),
           "since_2025": report(weekly[weekly.index >= "2025-01-01"].pct_change(), "Short IDR, 2025 on"),
           "always_on": report(always.resample("W-FRI").last().pct_change(), "Short IDR, always on (no filter)")}
    for y, e in eq.groupby(eq.index.year):
        prev = eq[eq.index < e.index[0]]
        base = prev.iloc[-1] if len(prev) else e.iloc[0]
        res.setdefault("yearly%", {})[str(y)] = round(float(e.iloc[-1] / base - 1) * 100, 2)
    print("  yearly:", res["yearly%"])
    p = pos[pos.index >= config.FX_START]
    yrs = (p.index[-1] - p.index[0]).days / 365.25
    print(f"  dollar filter: on {p.mean() * 100:.0f}% of days, {p.diff().abs().sum() / yrs:.1f} switches a year")
    eq.to_csv(out_dir / "equity_daily.csv", index_label="date", header=["equity"])
    p.astype(int).to_csv(out_dir / "position.csv", index_label="date", header=["short_on"])
    save_json({"config": {"carry": config.FX_CARRY, "roll_bps": config.FX_ROLL_BPS, "switch_bps": config.FX_SWITCH_BPS,
                          "mom_days": config.FX_MOM_DAYS, "start": config.FX_START},
               "metrics": res, "last": {"date": str(fx.index[-1].date()), "usdidr": float(fx.iloc[-1]),
                                        "dxy": float(dxy.iloc[-1]), "short_on": int(p.iloc[-1])},
               "time_on%": round(float(p.mean()) * 100, 1), "switches_per_year": round(float(p.diff().abs().sum() / yrs), 1),
               "n_rolls": int((eq.index.to_series().dt.month.diff().fillna(0) != 0).sum()),
               "usdidr_change%": round(float(fx.iloc[-1] / fx[fx.index >= config.FX_START].iloc[0] - 1) * 100, 1),
               "ann_drift%": round(float(np.log(fx.iloc[-1] / fx[fx.index >= config.FX_START].iloc[0])
                                         / ((fx.index[-1] - pd.Timestamp(config.FX_START)).days / 365)) * 100, 2)},
              out_dir / "fx_summary.json")
    return {"equity": eq, "metrics": res}
