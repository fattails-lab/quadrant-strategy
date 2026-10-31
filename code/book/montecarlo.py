"""Monte Carlo forward test (Appendix A10): how the next N weeks could go, starting from today.

Method
  Paths   stationary block bootstrap of history (Politis-Romano). Blocks start at random days and
          have geometric lengths (mean config.MC_BLOCK_DAYS), so volatility clusters and
          crash-rebound sequences stay intact. For the wheel, each day's HYPE move is first divided
          by the previous day's vol proxy (30-day realised vol x IV_RV_MULT) and the bootstrapped
          moves are rescaled by the simulated vol (filtered historical simulation). Without this,
          high-vol stretches would be priced at today's vol and paths would misprice the options. The simulation
          continues from the live end state (cash, HYPE held, the open option).
  Noise   the inputs we only estimated are redrawn on every path: option pricing at 0.83x to 1.0x
          the vol proxy and fills at 85% to 95% of model price.
  Shock   the worst 7 days in the sample are replayed at a random point inside the horizon, on top
          of the bootstrapped path.

Two entry points:
  wheel_forward()  reruns the actual wheel engine on simulated HYPE / vol paths;
  book_forward()   bootstraps the three sleeves' weekly returns jointly (keeps their correlation).
"""
import numpy as np
import pandas as pd

from book import config
from book.sleeves import wheel as wheel_sleeve

SCENARIOS = ("base", "noise", "shock")


def bootstrap_index(n_hist, n_out, mean_block, rng):
    """Indices for a stationary block bootstrap of length n_out from a sample of n_hist rows."""
    idx = []
    while len(idx) < n_out:
        start = rng.integers(0, n_hist)
        length = rng.geometric(1 / mean_block)
        idx.extend(((start + np.arange(length)) % n_hist).tolist())
    return np.array(idx[:n_out])


def annualised(growth, weeks):
    """Compounded annual rate implied by growth-of-1 outcomes over `weeks` (mean of log growth).

    Annualising the median instead would overstate a short-vol strategy, whose median beats its
    mean because the losses are rare but large.
    """
    return np.exp(np.log(growth).mean() * 52 / weeks) - 1


def summarise(paths, horizons):
    """Distribution of cumulative return at each horizon. paths: (n_paths, n_weeks) growth of 1."""
    rows = []
    for h in horizons:
        r = paths[:, h - 1] - 1
        tail = r[r <= np.quantile(r, 0.05)]
        rows.append({"weeks": h, "median%": np.median(r) * 100, "p5%": np.quantile(r, 0.05) * 100,
                     "p95%": np.quantile(r, 0.95) * 100, "es5%": tail.mean() * 100,
                     "P(gain)%": (r > 0).mean() * 100, "P(loss>10%)%": (r < -0.10).mean() * 100,
                     "annualised%": annualised(r + 1, h) * 100})
    return pd.DataFrame(rows).round(1)


# ---------- wheel: rerun the engine on simulated markets ----------
def _worst_week(spot_lr, iv_lr):
    """The 7 consecutive days with the lowest spot log return, as (spot, iv) daily log changes."""
    roll = pd.Series(spot_lr).rolling(7).sum().values
    end = int(np.nanargmin(roll))
    return spot_lr[end - 6:end + 1], iv_lr[end - 6:end + 1]


def wheel_forward(mkt, live_state, last_friday, horizon_weeks=max(config.MC_HORIZONS_WEEKS),
                  n_paths=config.MC_PATHS, scenario="base", seed=config.MC_SEED):
    """Simulate the always-on wheel forward from its live state.

    Returns (growth paths (n_paths, horizon_weeks) of the sleeve, same for buy-and-hold, info).
    """
    rng = np.random.default_rng(seed)
    spot, iv = mkt["spot"], mkt["iv"].ffill()
    daily_iv = iv.shift(1) / 100 / np.sqrt(365)   # yesterday's implied daily vol
    sample = pd.DataFrame({"spot": np.log(spot).diff(), "iv": np.log(iv).diff(),
                           "z": np.log(spot).diff() / daily_iv}).dropna()
    spot_lr, iv_lr, z = sample["spot"].values, sample["iv"].values, sample["z"].values
    shock_spot, shock_iv = _worst_week(spot_lr, iv_lr)

    last_day = spot.index[-1]
    future = pd.date_range(last_day + pd.Timedelta(days=1), last_friday + pd.Timedelta(weeks=horizon_weeks))
    n = len(future)
    tail = spot.iloc[-250:]                         # enough history for the 200-day MA
    rf = float(mkt["irx"].iloc[-1])                 # hold the T-bill rate flat over the horizon
    spot0 = float(spot.loc[:last_friday].iloc[-1])
    eq0 = live_state.cash + live_state.qty * spot0

    sleeve, bh = np.empty((n_paths, horizon_weeks)), np.empty((n_paths, horizon_weeks))
    for k in range(n_paths):
        idx = bootstrap_index(len(sample), n, config.MC_BLOCK_DAYS, rng)
        r_v = iv_lr[idx].copy()
        iv_mult, haircut = 1.0, config.PREMIUM_HAIRCUT
        if scenario == "noise":
            iv_mult, haircut = rng.uniform(0.83, 1.0), rng.uniform(0.85, 0.95)
        elif scenario == "shock":
            at = rng.integers(0, n - 7)
            r_v[at:at + 7] += shock_iv

        # filtered historical simulation: standardised moves rescaled by the simulated vol,
        # so realised vol stays tied to the vol the options are priced at
        path_iv = float(iv.iloc[-1]) * np.exp(np.cumsum(r_v))
        prev_iv = np.r_[float(iv.iloc[-1]), path_iv[:-1]] / 100 / np.sqrt(365)
        r_s = z[idx] * prev_iv
        if scenario == "shock":
            r_s[at:at + 7] += shock_spot
        path_spot = pd.Series(float(spot.iloc[-1]) * np.exp(np.cumsum(r_s)), index=future)
        full_spot = pd.concat([tail, path_spot])
        sim = wheel_sleeve.market_frame(full_spot, pd.Series(rf, index=full_spot.index),
                                        pd.concat([iv.iloc[-250:], pd.Series(path_iv, index=future)]))
        weeks, _ = wheel_sleeve.run_wheel(sim, use_regime=False, state=live_state,
                                          start=last_friday + pd.Timedelta(days=1),
                                          iv_mult=iv_mult, haircut=haircut)
        sleeve[k] = weeks["equity"].values[:horizon_weeks] / eq0
        bh[k] = weeks["spot"].values[:horizon_weeks] / spot0

    info = {"start": str(last_friday.date()), "shock_spot%": (np.exp(shock_spot.sum()) - 1) * 100,
            "sample": f"{sample.index[0].date()} to {sample.index[-1].date()}"}
    return sleeve, bh, info


# ---------- book: bootstrap the sleeves' weekly returns jointly ----------
def book_forward(weekly_returns, weights=None, horizon_weeks=max(config.MC_HORIZONS_WEEKS),
                 n_paths=config.MC_PATHS, scenario="base", mean_block_weeks=4, seed=config.MC_SEED):
    """Forward paths of the weighted book from a frame of weekly sleeve returns (one column per sleeve).

    Whole weeks are resampled together, so cross-sleeve correlation is kept. "noise" halves each
    sleeve's mean return (edge decay); "shock" replays the book's worst historical week.
    Returns growth paths (n_paths, horizon_weeks).
    """
    rng = np.random.default_rng(seed)
    weights = weights or config.BOOK_WEIGHTS
    R = weekly_returns[list(weights)].dropna()
    w = np.array(list(weights.values()))
    if scenario == "noise":
        R = R - 0.5 * R.mean()
    X = R.values
    worst = X[int(np.argmin(X @ w))]
    paths = np.empty((n_paths, horizon_weeks))
    for k in range(n_paths):
        sim = X[bootstrap_index(len(X), horizon_weeks, mean_block_weeks, rng)].copy()
        if scenario == "shock":
            sim[rng.integers(0, horizon_weeks)] = worst
        paths[k] = np.cumprod(1 + sim @ w)   # weekly rebalanced to the target weights
    return paths


# ---------- does the forecast work? walk-forward calibration ----------
def wheel_calibration(mkt, first="2025-07-04", every_weeks=4, horizons=config.MC_HORIZONS_WEEKS,
                      n_paths=300):
    """At past Fridays, forecast with data up to that day only, then score the realised outcome.

    Returns one row per (start, horizon) with the realised return and its percentile in the forecast.
    A calibrated forecast puts ~90% of outcomes inside its 5-95% band and ~5% below the 5th percentile.
    """
    full, _ = wheel_sleeve.run_wheel(mkt, use_regime=False)
    eq = full["equity"]
    starts = [f for f in eq.index[eq.index >= first][::every_weeks]
              if eq.index.get_loc(f) + max(horizons) < len(eq)]
    rows = []
    for f in starts:
        past = {k: v.loc[:f] for k, v in mkt.items()}
        _, state = wheel_sleeve.run_wheel(past, use_regime=False)
        paths, _, _ = wheel_forward(past, state, f, max(horizons), n_paths, seed=int(f.strftime("%Y%m%d")))
        for h in horizons:
            realised = eq.iloc[eq.index.get_loc(f) + h] / eq.loc[f] - 1
            sim = paths[:, h - 1] - 1
            rows.append({"start": f.date(), "weeks": h, "realised%": realised * 100,
                         "forecast_median%": np.median(sim) * 100,
                         "pctile": (sim < realised).mean() * 100})
    return pd.DataFrame(rows)


def calibration_summary(cal):
    g = cal.groupby("weeks")["pctile"]
    return pd.DataFrame({"n": g.size(),
                         "inside 5-95%": g.apply(lambda p: ((p >= 5) & (p <= 95)).mean() * 100),
                         "below 5%": g.apply(lambda p: (p < 5).mean() * 100),
                         "above 95%": g.apply(lambda p: (p > 95).mean() * 100),
                         "median pctile": g.median()}).round(1)
