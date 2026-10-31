"""Sleeve 2 · Crypto options wheel (slide 5): BTC from Apr 2021; 70% HYPE / 30% BTC from Jan 2025.

Weekly cycle on Fridays with synthetic 1-week 25-delta options, priced by Black-Scholes. The BTC leg
(2 Apr 2021 to 3 Jan 2025) prices at Deribit's DVOL. HYPE has no implied-vol index, so the HYPE leg
prices at 30-day realised vol x config.IV_RV_MULT. Premium is taken at 95% of model price. Cash earns
the 13-week T-bill rate. USD 100K in Apr 2021. From 3 Jan 2025 the sleeve holds two wheels, HYPE and
BTC, at config.WHEEL_MIX, rebalanced at each quarter start; values are marked to market daily.

  State         Always-on wheel (adopted)    Regime wheel (rejected, A3)
  Flat          sell cash-secured put        sell put only in bear (spot < 200d MA)
  Holding HYPE  sell covered call            sell call only in bull
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from book import config
from book.metrics import report, save_json


# ---------- data ----------
def realised_vol(spot, window=config.IV_WINDOW):
    """Annualised close-to-close vol in percent (crypto trades 365 days a year)."""
    return np.log(spot).diff().rolling(window).std() * np.sqrt(365) * 100


def load_market():
    """Daily spot close, T-bill rate, 200-day MA and the implied-vol proxy (percent)."""
    px = pd.read_csv(config.data_dir("wheel") / config.WHEEL_PRICE_FILE, index_col=0, parse_dates=True)
    spot = px[config.WHEEL_TICKER].dropna()
    irx = px["^IRX"].ffill().reindex(spot.index).ffill() / 100.0
    print(f"{config.WHEEL_TICKER}: {len(spot)} days {spot.index[0].date()} -> {spot.index[-1].date()}")
    return market_frame(spot, irx, realised_vol(spot) * config.IV_RV_MULT)


def load_btc_market():
    """BTC daily close, T-bill rate and Deribit DVOL (percent), for the 2021-2024 leg."""
    d = config.data_dir("wheel")
    px = pd.read_csv(d / config.WHEEL_BTC_FILE, index_col=0, parse_dates=True)
    spot = px[config.WHEEL_BTC_TICKER].dropna()
    irx = px["^IRX"].ffill().reindex(spot.index).ffill().bfill() / 100.0
    dvol = pd.read_csv(d / config.WHEEL_DVOL_FILE, index_col=0, parse_dates=True).iloc[:, 0]
    return market_frame(spot, irx, dvol.reindex(spot.index).ffill())


def chain(btc_series, hype_series, switch=config.WHEEL_SWITCH):
    """BTC leg up to the switch Friday, then the HYPE leg rescaled to start from the BTC leg's equity."""
    sw = pd.Timestamp(switch)
    head = btc_series[btc_series.index <= sw]
    scale = float(btc_series.asof(sw)) / config.WHEEL_CAPITAL
    tail = hype_series[hype_series.index > sw] * scale
    return pd.concat([head, tail])


def blend(daily, weights, start):
    """Quarterly-rebalanced mix of daily equity series from `start` (growth of 1 at `start`)."""
    D = pd.DataFrame(daily).ffill()
    D = D[D.index >= pd.Timestamp(start)].dropna()
    w = pd.Series(weights)
    out, level = [], 1.0
    for _, q in D.groupby(D.index.to_period("Q")):
        base = D.loc[:q.index[0]].iloc[-2] if len(D.loc[:q.index[0]]) > 1 else q.iloc[0]
        v = level * (q / base * w).sum(axis=1)
        out.append(v)
        level = v.iloc[-1]
    return pd.concat(out)


def market_frame(spot, irx, iv):
    return {"spot": spot, "irx": irx, "ma200": spot.rolling(200).mean(), "iv": iv}


# ---------- option pricing ----------
def bs_price(S, K, T, r, sig, kind):
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def strike_for_delta(S, T, r, sig, target, kind):
    """Strike whose Black-Scholes delta equals target (e.g. 0.25 call, -0.25 put)."""
    def delta(K):
        d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
        return norm.cdf(d1) - 1 if kind == "put" else norm.cdf(d1)
    lo, hi = (S * 0.3, S) if kind == "put" else (S, S * 3.0)
    return brentq(lambda K: delta(K) - target, lo, hi, xtol=S * 1e-6)


# ---------- engine ----------
@dataclass
class WheelState:
    cash: float = config.WHEEL_CAPITAL
    qty: float = 0.0
    opt: tuple = None   # (kind, strike, quantity) of the option written last Friday
    stats: dict = field(default_factory=lambda: {"puts_sold": 0, "puts_assigned": 0, "calls_sold": 0,
                                                 "calls_away": 0, "premium": 0.0})

    def copy(self):
        return WheelState(self.cash, self.qty, self.opt, dict(self.stats))


def step_week(st, S, r, bull, sig, use_regime, haircut=config.PREMIUM_HAIRCUT):
    """One Friday: settle last week's option, accrue a week of interest, write this week's option."""
    # 1) settle at today's close
    if st.opt is not None:
        kind, K, q = st.opt
        if kind == "put" and S < K:
            st.cash -= K * q
            st.qty += q
            st.stats["puts_assigned"] += 1
        elif kind == "call" and S > K:
            st.cash += K * q
            st.qty = 0.0
            st.stats["calls_away"] += 1
        st.opt = None

    # 2) a week of T-bill interest on cash
    st.cash *= (1 + r * 7 / 365)

    # 3) write this week's option
    T = config.OPTION_T
    if st.qty > 0:
        if bull or not use_regime:
            K = strike_for_delta(S, T, r, sig, config.OPTION_DELTA, "call")
            prem = bs_price(S, K, T, r, sig, "call") * haircut * st.qty
            st.cash += prem
            st.stats["premium"] += prem
            st.stats["calls_sold"] += 1
            st.opt = ("call", K, st.qty)
    elif (not bull) or not use_regime:
        K = strike_for_delta(S, T, r, sig, -config.OPTION_DELTA, "put")
        q = st.cash / K  # fully cash-secured
        prem = bs_price(S, K, T, r, sig, "put") * haircut * q
        st.cash += prem
        st.stats["premium"] += prem
        st.stats["puts_sold"] += 1
        st.opt = ("put", K, q)


def run_wheel(mkt, use_regime, state=None, start=config.WHEEL_START, iv_mult=1.0,
              haircut=config.PREMIUM_HAIRCUT):
    """Weekly wheel from `start` (or from `state`, to continue a live book).

    iv_mult scales the vol before pricing (1.0 = the frozen spec). The 200-day MA is only required
    by the regime wheel. Returns (weekly frame, final state).
    """
    spot, irx, ma200, iv = mkt["spot"], mkt["irx"], mkt["ma200"], mkt["iv"]
    iv = iv.ffill()
    fridays = [f for f in spot.resample("W-FRI").last().index
               if f >= pd.Timestamp(start) and f <= spot.index[-1]
               and pd.notna(iv.asof(f)) and (not use_regime or pd.notna(ma200.asof(f)))]
    st = WheelState() if state is None else state.copy()
    rows = []
    for f in fridays:
        S = float(spot.loc[:f].iloc[-1])
        r = float(irx.loc[:f].iloc[-1])
        bull = S >= float(ma200.loc[:f].iloc[-1]) if use_regime else True
        sig = float(iv.loc[:f].iloc[-1]) / 100.0 * iv_mult
        step_week(st, S, r, bull, sig, use_regime, haircut)
        rows.append({"week": f, "equity": st.cash + st.qty * S, "cash": st.cash, "qty": st.qty,
                     "spot": S, "holding": st.qty > 0, "opt": st.opt, "r": r, "sig": sig})
    return pd.DataFrame(rows).set_index("week"), st


def daily_equity(mkt, weeks):
    """Mark the weekly book to market every day: cash plus coins, less the open short option.

    The weekly `equity` column books the premium on the day of the sale; this series also carries the
    option's remaining value, so it is what an account statement would show on any day.
    """
    spot = mkt["spot"]
    fridays = list(weeks.index)
    out = {}
    for i, f in enumerate(fridays):
        w = weeks.loc[f]
        end = fridays[i + 1] if i + 1 < len(fridays) else spot.index[-1] + pd.Timedelta(days=1)
        days = spot.index[(spot.index >= f) & (spot.index < end)]
        for d in days:
            S = float(spot.loc[d])
            eq = w["cash"] * (1 + w["r"] * (d - f).days / 365) + w["qty"] * S
            if w["opt"] is not None:
                kind, K, q = w["opt"]
                T = max((f + pd.Timedelta(days=7) - d).days, 0) / 365
                if T > 0:
                    eq -= bs_price(S, K, T, w["r"], w["sig"], kind) * q
                else:
                    eq -= max(K - S, 0) * q if kind == "put" else max(S - K, 0) * q
            out[d] = eq
    return pd.Series(out, name="equity")


# ---------- entry point ----------
def run():
    """Always-on and regime wheels, BTC then HYPE, vs buy-and-hold; outputs in outputs/wheel/."""
    out_dir = config.output_dir("wheel")
    btc_mkt, mkt = load_btc_market(), load_market()
    sw = pd.Timestamp(config.WHEEL_SWITCH)
    legs = {}
    for name, regime in [("always", False), ("regime", True)]:
        b_df, b_st = run_wheel(btc_mkt, use_regime=regime, start=config.WHEEL_BTC_START)
        h_df, h_st = run_wheel(mkt, use_regime=regime, start=config.WHEEL_SWITCH)
        b_day = daily_equity(btc_mkt, b_df)
        h_day = daily_equity(mkt, h_df) if len(h_df) else pd.Series(dtype=float)
        h_day = h_day.reindex(b_day.index[b_day.index >= sw]).fillna(config.WHEEL_CAPITAL) if regime else h_day
        mix = blend({"hype": h_day, "btc": b_day}, config.WHEEL_MIX, sw)
        daily = pd.concat([b_day[b_day.index < sw], mix * float(b_day.asof(sw))])
        weekly = pd.concat([b_df["equity"][b_df.index <= sw],
                            daily.resample("W-FRI").last()[lambda x: x.index > sw]])
        legs[name] = {"btc": (b_df, b_st), "hype": (h_df, h_st), "equity": weekly, "daily": daily,
                      "b_day": b_day, "h_day": h_day}
    always = legs["always"]
    daily = always["daily"]
    coin = blend({"hype": mkt["spot"], "btc": btc_mkt["spot"]}, config.WHEEL_MIX, sw)
    spot = pd.concat([btc_mkt["spot"][btc_mkt["spot"].index < sw] / btc_mkt["spot"].asof(pd.Timestamp(config.WHEEL_BTC_START)),
                      coin * btc_mkt["spot"].asof(sw) / btc_mkt["spot"].asof(pd.Timestamp(config.WHEEL_BTC_START))]) * config.WHEEL_CAPITAL
    bh_equity = spot.reindex(always["equity"].index, method="ffill")

    # trend rule, per leg: a coin below its own 200-day average at a Friday close runs its leg the next week
    # at WHEEL_TREND_KEEP of size, the rest in T-bills; the legs are blended 70/30 after the rule
    tb = btc_mkt["irx"]

    def trend(day, px, keep):
        sma = px.rolling(config.WHEEL_TREND_DAYS).mean()
        on_fri = (px > sma).where(sma.notna(), True).resample("W-FRI").last().astype(bool)
        on_d = on_fri.shift(1).reindex(day.index, method="ffill").fillna(True).astype(bool)
        gap = day.index.to_series().diff().dt.days.fillna(1)
        r = day.pct_change().fillna(0)
        r = r.where(on_d, keep * r + (1 - keep) * tb.reindex(day.index, method="ffill") * gap / 365)
        return float(day.iloc[0]) * (1 + r).cumprod(), on_fri

    b_f, b_on = trend(always["b_day"], btc_mkt["spot"], config.WHEEL_TREND_KEEP["btc"])
    h_f, h_on = trend(always["h_day"], mkt["spot"], config.WHEEL_TREND_KEEP["hype"])
    mix_f = blend({"hype": h_f, "btc": b_f}, config.WHEEL_MIX, sw)
    filt_d = pd.concat([b_f[b_f.index < sw], mix_f * float(b_f.asof(sw))])
    raw_w = always["equity"]
    filt_w = filt_d.resample("W-FRI").last().reindex(raw_w.index, method="ffill")
    on_w = pd.DataFrame({"btc_on": b_on, "hype_on": h_on}).shift(1).reindex(raw_w.index)
    on_w["hype_on"] = on_w["hype_on"].where(on_w.index > sw)
    for leg in ("btc_on", "hype_on"):
        x = on_w[leg].dropna().astype(bool)
        print(f"  trend rule {leg[:-3].upper()}: out {(~x).mean() * 100:.0f}% of weeks, {int(x.astype(int).diff().abs().sum())} switches")

    res = {}
    for key, eq, label in [("wheel_raw", raw_w, "Always-on wheel, no switch"),
                           ("wheel", legs["regime"]["equity"], "Regime wheel (first design)"),
                           ("wheel_always", filt_w, "Always-on wheel, per-leg 200d rule (adopted)"),
                           ("bh", bh_equity, "Buy and hold, BTC then 70/30 HYPE/BTC")]:
        res[key] = report(eq.pct_change(), label)
        res[key]["final$"] = round(float(eq.iloc[-1]))
    stats = {}
    for leg in ("btc", "hype"):
        df, st = always[leg]
        stats[leg] = st.stats
        print(f"  ALWAYS {leg.upper()}: puts sold {st.stats['puts_sold']}, assigned {st.stats['puts_assigned']}; "
              f"calls sold {st.stats['calls_sold']}, called away {st.stats['calls_away']}; premium ${st.stats['premium']:,.0f}")
    eq = always["equity"]
    print("  always " + " ".join(f"{y}: {(e.iloc[-1] / eq[eq.index < e.index[0]].iloc[-1] - 1) * 100 if (eq.index < e.index[0]).any() else (e.iloc[-1] / e.iloc[0] - 1) * 100:+6.1f}%"
                                 for y, e in eq.groupby(eq.index.year)))

    legs["regime"]["equity"].to_csv(out_dir / "returns_wheel.csv", header=["equity"])
    raw_w.to_csv(out_dir / "returns_wheel_raw.csv", header=["equity"])
    filt_w.to_csv(out_dir / "returns_wheel_always.csv", header=["equity"])
    filt_d.to_csv(out_dir / "equity_daily_always.csv", index_label="date", header=["equity"])
    on_w.to_csv(out_dir / "trend_switch.csv")
    bh_equity.to_csv(out_dir / "returns_bh.csv", header=["equity"])
    save_json({"metrics": res, "trade_stats_always": stats, "switch": config.WHEEL_SWITCH},
              out_dir / "wheel_summary.json")
    return {"market": mkt, "btc_market": btc_mkt, "always": always["hype"][0], "always_state": always["hype"][1],
            "equity": always["equity"], "metrics": res}
