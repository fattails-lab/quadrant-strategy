"""Every book-level number and chart the deck uses. Run after the sleeves and fetch_data.py book.

    python scripts/book_report.py

Writes outputs/book/deck_stats.json and the charts in assets/deck/img/. CAGR here is geometric
(compounded), not the arithmetic ann_ret% of the sleeve summaries.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from arch import arch_model  # noqa: E402
from scipy.stats import genpareto  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from book import config, portfolio  # noqa: E402
from book.metrics import save_json  # noqa: E402

WLABEL = "/".join(str(int(round(config.BOOK_WEIGHTS[k] * 100))) for k in ("e1", "dip", "fx", "wheel"))
from book.sleeves import fx_idr, wheel  # noqa: E402

IMG = config.ROOT / "assets" / "deck" / "img"
NAVY, PURPLE, GREEN, GREY, ORANGE = "#1F3864", "#5B2C8F", "#4A6B2A", "#7f7f7f", "#B25A0E"
OCT = ("2026-10-01", "2026-10-30")   # the live window, to the 30 Oct close


def stats(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    g = eq.iloc[-1]
    return {"sharpe": r.mean() / r.std() * np.sqrt(52), "cagr%": (g ** (52 / len(r)) - 1) * 100,
            "vol%": r.std() * np.sqrt(52) * 100, "max_dd%": (eq / eq.cummax() - 1).min() * 100,
            "total%": (g - 1) * 100}


def nw_sharpe(r, lags=4):
    """Sharpe with a Newey-West (Bartlett) long-run variance instead of the plain variance."""
    x = (r - r.mean()).values
    lrv = x @ x / len(x)
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / len(x)
    return r.mean() / np.sqrt(lrv) * np.sqrt(52)


def window(r, start, end):
    w = r[(r.index >= pd.Timestamp(start)) & (r.index <= pd.Timestamp(end))]
    return float(((1 + w).prod() - 1) * 100), float((((1 + w).prod()) ** (52 / len(w)) - 1) * 100)


def tail(r, q=0.99):
    """One-week 99% VaR and ES of losses (% of capital): historical, and EVT (GPD over the 90% loss quantile)."""
    loss = -r.dropna().values * 100
    var_h = np.quantile(loss, q)
    es_h = loss[loss >= var_h].mean()
    u = np.quantile(loss, 0.90)
    exc = loss[loss > u] - u
    xi, _, beta = genpareto.fit(exc, floc=0)
    p_u = len(exc) / len(loss)
    var_e = u + beta / xi * (((1 - q) / p_u) ** -xi - 1)
    es_e = (var_e + beta - xi * u) / (1 - xi)
    return {"var_hist": var_h, "es_hist": es_h, "var_evt": var_e, "es_evt": es_e, "xi": xi, "n_exceed": len(exc)}


def main():
    out = config.output_dir("book")
    S = list(config.BOOK_WEIGHTS)                       # e1, dip, wheel, fx
    R = portfolio.weekly_returns(S)
    bench = pd.read_csv(config.data_dir("book") / "benchmarks.csv", index_col=0, parse_dates=True)
    hype = pd.read_csv(config.data_dir("wheel") / config.WHEEL_PRICE_FILE, index_col=0,
                       parse_dates=True)[config.WHEEL_TICKER]
    usdidr = fx_idr.load_fx()
    wk = lambda s: s.dropna().resample("W-FRI").last().pct_change()  # noqa: E731
    coin = pd.read_csv(config.OUTPUTS / "wheel" / "returns_bh.csv", index_col=0, parse_dates=True)["equity"]
    R["spx"], R["gold"], R["hype"], R["usdidr"] = wk(bench["^GSPC"]), wk(bench["GC=F"]), wk(hype), wk(usdidr)
    R["coin"] = coin.pct_change()            # the wheel's coin: BTC to 3 Jan 2025, HYPE after
    R["book"] = portfolio.book_returns(R)
    C = R.dropna(subset=S + ["book"])
    C = C[C.index <= pd.Timestamp(OCT[1])]
    res = {"window": [str(C.index[0].date()), str(C.index[-1].date())], "n_weeks": len(C),
           "weights": config.BOOK_WEIGHTS}

    res["stats"] = {k: stats(C[k]) for k in S + ["book", "spx", "coin", "usdidr"]}
    res["by_year"] = {str(y): {k: stats(g[k]) for k in S + ["book", "spx"]} for y, g in C.groupby(C.index.year)}
    res["corr"] = C[S + ["spx"]].corr().round(2).to_dict()
    res["robust"] = {k: {"sharpe": stats(C[k])["sharpe"], "nw": nw_sharpe(C[k]),
                         "no_best_week": stats(C[k].drop(C[k].idxmax()))["sharpe"]} for k in S + ["book", "spx"]}

    # the live window: calendar October from daily closes, book held at its weights from 30 Sep
    oct_r = portfolio.window_returns(*OCT, sleeves=S)
    oct_r["book"] = sum(oct_r[k] * w for k, w in config.BOOK_WEIGHTS.items())
    days = (pd.Timestamp(OCT[1]) - pd.Timestamp(OCT[0])).days + 1
    res["october"] = {k: [float(v * 100), float(((1 + v) ** (365 / days) - 1) * 100)] for k, v in oct_r.items()}
    res["october"]["usdidr"] = [float((usdidr[:OCT[1]].iloc[-1] / usdidr[:"2026-09-30"].iloc[-1] - 1) * 100), None]
    res["october"]["spx"] = [float((bench["^GSPC"].dropna()[:OCT[1]].iloc[-1] / bench["^GSPC"].dropna()[:"2026-09-30"].iloc[-1] - 1) * 100), None]

    # book drawdown and when the brake (-5% halve, -8% stop) would have fired
    eq = (1 + C["book"]).cumprod()
    dd = eq / eq.cummax() - 1
    first = lambda m: str(dd[m].index.min().date()) if m.any() else None  # noqa: E731
    res["brake"] = {"halve_first": first(dd <= config.BRAKE_HALVE_DD), "stop_first": first(dd <= config.BRAKE_STOP_DD),
                    "max_dd_date": str(dd.idxmin().date()), "weeks_below_-5%": int((dd <= -0.05).sum())}
    res["worst_weeks"] = {k: {"date": str(C[k].idxmin().date()), "ret%": C[k].min() * 100} for k in S + ["book"]}
    cols = S + ["book", "spx", "coin", "usdidr"]
    res["stress"] = {"crypto_winter_2022": {k: window(C[k], "2022-01-01", "2022-12-31")[0] for k in cols},
                     "yen_unwind_aug_2024": {k: window(C[k], "2024-07-27", "2024-08-09")[0] for k in cols},
                     "hype_crash_mar_2025": {k: window(C[k], "2025-03-01", "2025-03-07")[0] for k in cols},
                     "tariff_apr_2025": {k: window(C[k], "2025-04-01", "2025-04-11")[0] for k in cols},
                     "chip_selloff_oct_2026": {k: window(C[k], "2026-10-10", "2026-10-16")[0] for k in cols}}

    # weights sensitivity
    vol = C[S].std()
    iv = (1 / vol) / (1 / vol).sum()
    alts = {"chosen": config.BOOK_WEIGHTS, "inverse_vol": iv.to_dict(), "equal": {k: 1 for k in S},
            "fx_in_idx_slot": {"e1": 40, "dip": 20, "wheel": 15, "fx": 25},
            "no_fx_40_20_40": {"e1": 40, "dip": 20, "wheel": 40}}
    res["weights_alt"] = {}
    for k, w in alts.items():
        b = portfolio.book_returns(C, w)
        o = portfolio.window_returns(*OCT, sleeves=list(w))
        o_book = sum(o[s] * v for s, v in w.items()) / sum(w.values())
        res["weights_alt"][k] = {"w": {s: v / sum(w.values()) for s, v in w.items()}, **stats(b),
                                 "oct%": float(o_book * 100), "oct_ann%": float(((1 + o_book) ** (365 / days) - 1) * 100)}

    # factor betas (weekly OLS on SPX, the wheel's coin, gold, USD/IDR)
    F = ["spx", "coin", "gold", "usdidr"]
    X = sm.add_constant(C[F].dropna())
    res["factors"] = {}
    for k in S + ["book"]:
        m = sm.OLS(C.loc[X.index, k], X).fit()
        res["factors"][k] = {f: {"b": m.params[f], "t": m.tvalues[f]} for f in F} | {"r2": m.rsquared}
    res["factors_n"] = len(X)

    res["tail"] = {k: tail(C[k]) for k in S + ["book"]}
    res["garch"] = garch_exhibit()
    res["coint"] = cointegration_exhibit()
    save_json(res, out / "deck_stats.json")
    charts(C, res)
    appendix_charts(res)
    dip_chart()
    fx_chart()
    print(f"wrote {out / 'deck_stats.json'}")
    return res


def garch_exhibit():
    """GARCH(1,1) on HYPE daily returns, fitted on 2025; out of sample in 2026 against the pricing vol."""
    mkt = wheel.load_market()
    r = np.log(mkt["spot"]).diff().dropna() * 100
    fit_end = "2025-12-31"
    am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(last_obs=fit_end, disp="off")
    fc = res.forecast(horizon=7, start=pd.Timestamp("2026-01-01"), reindex=False)
    garch = np.sqrt(fc.variance.mean(axis=1) * 365)                        # annualised %, next 7 days
    realised = r.rolling(7).std().shift(-7) * np.sqrt(365)                 # what the next 7 days delivered
    df = pd.DataFrame({"pricing": mkt["iv"], "garch": garch, "realised": realised}).dropna()
    df.to_csv(config.output_dir("book") / "garch_vol.csv")
    p = res.params
    out = {"alpha": p["alpha[1]"], "beta": p["beta[1]"], "nu": p["nu"], "fit": f"{r.index[0].date()} to {fit_end}",
           "oos": f"{df.index[0].date()} to {df.index[-1].date()}",
           "mean_pricing": df["pricing"].mean(), "mean_garch": df["garch"].mean(), "mean_realised": df["realised"].mean(),
           "pricing_above_realised%": (df["pricing"] > df["realised"]).mean() * 100}
    fig, ax = plt.subplots(figsize=(11.0, 3.4), dpi=150)
    ax.plot(df.index, df["realised"], color=GREY, lw=1, label="Realised over the next 7 days")
    ax.plot(df.index, df["garch"], color=NAVY, lw=1.6, label="GARCH(1,1) 7-day forecast")
    ax.plot(df.index, df["pricing"], color=PURPLE, lw=1.6, label="Pricing vol (30-day realised)")
    ax.set_ylabel("annualised vol, %")
    ax.set_title("HYPE volatility out of sample, 2026: what the wheel priced at vs GARCH vs what followed")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(IMG / "garch_vol.png")
    plt.close(fig)
    return out


def cointegration_exhibit():
    """Classic stock pairs (Engle-Granger on log prices) vs E1's factor residuals (ADF), 2019 to now."""
    from statsmodels.tsa.stattools import adfuller, coint

    from book.sleeves import e1_pca

    px = e1_pca.load_prices()
    lp = np.log(px)
    pairs = [("D05.SI", "O39.SI"), ("O39.SI", "U11.SI"), ("JPM", "BAC"), ("XOM", "CVX"), ("V", "MA"), ("PG", "PEP")]
    rows, spreads = [], {}
    for y, x in pairs:
        beta = sm.OLS(lp[y], sm.add_constant(lp[x])).fit().params[x]
        spread = lp[y] - beta * lp[x]
        b = sm.OLS(spread.values[1:], sm.add_constant(spread.values[:-1])).fit().params[1]
        rows.append({"pair": f"{y} vs {x}", "beta": beta, "eg_p": coint(lp[y], lp[x])[1],
                     "half_life": np.log(2) / -np.log(b) if 0 < b < 1 else np.inf})
        spreads[f"{y} vs {x}"] = spread - spread.mean()

    # E1 residuals: the cumulative factor residual of every name in 60-day windows, every 20 days
    R = px.pct_change().dropna().values
    rej, hls, last = [], [], None
    for t in range(config.E1_CONFIG["window"], len(R), 20):
        W = R[t - config.E1_CONFIG["window"] + 1:t + 1]
        mu, sd = W.mean(axis=0), W.std(axis=0, ddof=1)
        Z = (W - mu) / sd
        C = Z.T @ Z / (len(W) - 1)
        ev, evec = np.linalg.eigh(C)
        V = evec[:, np.argsort(ev)[::-1][:config.E1_CONFIG["n_factors"]]]
        F = Z @ V
        X = np.cumsum((Z - F @ np.linalg.solve(F.T @ F, F.T @ Z)) * sd, axis=0)
        ps = [adfuller(X[:, j], maxlag=1, autolag=None)[1] for j in range(X.shape[1])]
        rej.append(np.mean(np.array(ps) < 0.05))
        for j in range(X.shape[1]):
            b = np.polyfit(X[:-1, j], X[1:, j], 1)[0]
            if 0 < b < 1:
                hls.append(np.log(2) / -np.log(b))
        last = (px.index[t + 1], X, ps)
    res = {"pairs": rows, "windows": len(rej), "adf_reject_share%": float(np.mean(rej) * 100),
           "half_life_median": float(np.median(hls)), "half_life_iqr": [float(np.percentile(hls, 25)), float(np.percentile(hls, 75))],
           "last_window": str(last[0].date()), "last_reject": int(sum(p < 0.05 for p in last[2]))}

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.047), dpi=150)
    worst = max(rows, key=lambda r: r["eg_p"])["pair"]
    axes[0].plot(spreads[worst].index, spreads[worst], color="#B25A0E", lw=1)
    names_sg = {"D05.SI": "DBS", "O39.SI": "OCBC", "U11.SI": "UOB"}
    pretty = " vs ".join(names_sg.get(t, t) for t in worst.split(" vs "))
    axes[0].set_title(f"{pretty} spread, 2019–2026", fontsize=8)
    j = list(px.columns).index("D05.SI")
    axes[1].plot(range(len(last[1])), last[1][:, j] * 100, color=NAVY, lw=1.4)
    axes[1].set_title(f"DBS E1 residual, 60 days to {last[0]:%d %b %Y}", fontsize=8)
    axes[1].set_ylabel("residual, %", fontsize=7)
    for ax in axes:
        ax.axhline(0, color=GREY, lw=.8)
        ax.grid(alpha=.3)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(IMG / "cointegration.png")
    plt.close(fig)
    return res


def appendix_charts(res):
    """A2 robustness bars and A7 expected-shortfall bars, in the original deck's style."""
    names = [("e1", "E1", NAVY), ("dip", "DIP", GREEN), ("wheel", "Wheel", PURPLE), ("fx", "Short IDR", ORANGE),
             ("book", "BOOK", NAVY)]
    fig, ax = plt.subplots(figsize=(6.4, 3.1), dpi=150)
    for i, (k, lab, c) in enumerate(names):
        r = res["robust"][k]
        for j, (key, a, leg) in enumerate([("sharpe", 1.0, "Sharpe"), ("nw", 0.6, "Newey–West"), ("no_best_week", 0.3, "Drop best week")]):
            ax.bar(i + (j - 1) * 0.27, r[key], width=0.26, color=c, alpha=a, label=leg if i == 0 else None)
    ax.set_xticks(range(len(names)), [n[1] for n in names])
    ax.axhline(0, color=GREY, lw=.8)
    ax.set_ylabel("Sharpe")
    ax.set_title("No sleeve lives off its best week (short IDR is the exception)", pad=24)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(IMG / "robustness.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 2.6), dpi=150)
    for i, (k, lab, c) in enumerate(names):
        t = res["tail"][k]
        y = len(names) - 1 - i
        ax.barh(y + 0.18, t["es_hist"], height=0.34, color=c, alpha=0.55)
        ax.barh(y - 0.18, t["es_evt"], height=0.34, color=c)
        ax.text(t["es_hist"] + 0.3, y + 0.19, f"{t['es_hist']:.1f}", va="center", fontsize=7.5)
        ax.text(t["es_evt"] + 0.3, y - 0.19, f"{t['es_evt']:.1f}", va="center", fontsize=7.5, weight="bold")
    ax.set_yticks(range(len(names)), [n[1] for n in names][::-1])
    ax.set_xlim(0, max(res["tail"][k]["es_hist"] for k, *_ in names) * 1.12)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("1-week 99% Expected Shortfall (% of sleeve capital)", fontsize=8)
    ax.set_title("Expected Shortfall: historical (light) vs EVT tail fit (solid)", fontsize=9)
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    fig.savefig(IMG / "tail_es.png")
    plt.close(fig)


def dip_chart():
    """Dip sleeve equity and drawdown, 2015 to now (slide 6)."""
    eq = pd.read_csv(config.OUTPUTS / "dip" / "equity_B_100k.csv", index_col="date", parse_dates=True)["equity"]
    dd = (eq / eq.cummax() - 1) * 100
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.0, 4.6), dpi=150, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(eq.index, eq, color="#2F4A1A", lw=1)
    a1.set_ylabel("Equity (USD 100K start)", fontsize=8)
    a1.set_title(f"Sleeve 4, dip buying (Nasdaq-100), equity curve 2015 to {eq.index[-1]:%Y}, net of commissions", fontsize=9)
    a2.fill_between(dd.index, dd, 0, color="#C0392B", alpha=.55)
    a2.set_ylabel("Drawdown %", fontsize=8)
    for a in (a1, a2):
        a.grid(alpha=.3)
        a.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(IMG / "dip_equity.png")
    plt.close(fig)


def fx_chart():
    """Short-IDR sleeve vs USD/IDR spot, 2015 to now (slide 6)."""
    eq = pd.read_csv(config.OUTPUTS / "fx" / "equity_daily.csv", index_col=0, parse_dates=True)["equity"]
    fx = fx_idr.load_fx()
    fx = fx[fx.index >= eq.index[0]]
    fig, ax = plt.subplots(figsize=(8.227, 4.14), dpi=150)
    ax.plot(eq.index, eq / eq.iloc[0], color=ORANGE, lw=1.4, label="Short IDR sleeve (after carry and roll costs)")
    ax.plot(fx.index, fx / fx.iloc[0], color=GREY, lw=1.0, label="USD/IDR spot (no carry)")
    ax.axhline(1, color=GREY, lw=.6)
    ax.set_ylabel("Growth of 1 (USD)")
    ax.set_title(f"Short IDR vs USD/IDR spot, {eq.index[0]:%Y} to {eq.index[-1]:%Y} (daily)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(IMG / "fx_equity.png")
    plt.close(fig)


def charts(C, res):
    g = (1 + C[["e1", "dip", "wheel", "fx", "book", "spx"]].fillna(0)).cumprod()
    label = "/".join(f"{w * 100:.0f}" for w in config.BOOK_WEIGHTS.values())

    fig, ax = plt.subplots(figsize=(12.7, 1.847), dpi=150)
    ax.plot(g.index, g["book"] * 100, color="black", lw=3, label=f"Book, E1 / dip / short IDR / wheel {WLABEL}")
    ax.plot(g.index, g["spx"] * 100, color=GREY, lw=1.8, ls="--", label="S&P 500")
    ax.set_yticks([100, 200, 300])
    ax.tick_params(labelsize=14)
    ax.legend(loc="upper left", fontsize=14, frameon=False)
    ax.grid(alpha=.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG / "book_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=150)
    ax.plot(g.index, g["e1"], color=NAVY, lw=1.3, label="E1 (sleeve)")
    ax.plot(g.index, g["wheel"], color=PURPLE, lw=1.3, label="Wheel, BTC then HYPE (sleeve)")
    ax.plot(g.index, g["dip"], color=GREEN, lw=1.3, label="DIP (sleeve)")
    ax.plot(g.index, g["fx"], color=ORANGE, lw=1.3, label="Short IDR (sleeve)")
    ax.plot(g.index, g["book"], color="black", lw=2.6, label=f"BOOK {label}")
    ax.plot(g.index, g["spx"], color=GREY, lw=1.3, ls="--", label="SPX (benchmark)")
    ax.set_yscale("log")
    ax.set_title(f"Four sleeves and the book, {C.index[0]:%b %Y} to {C.index[-1]:%b %Y} (weekly, net of costs)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(IMG / "book_sleeves.png")
    plt.close(fig)

    w = pd.read_csv(config.OUTPUTS / "wheel" / "returns_wheel_always.csv", index_col=0, parse_dates=True)["equity"]
    rg = pd.read_csv(config.OUTPUTS / "wheel" / "returns_wheel.csv", index_col=0, parse_dates=True)["equity"]
    bh = pd.read_csv(config.OUTPUTS / "wheel" / "returns_bh.csv", index_col=0, parse_dates=True)["equity"]
    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=150)
    ax.plot(w.index, w / 1000, color=NAVY, lw=1.8, label="Wheel (adopted)")
    ax.plot(bh.index, bh / bh.iloc[0] * 100, color=GREY, lw=1.3, label="Buy and hold, BTC then HYPE")
    ax.set_yscale("log")
    ax.set_ylabel("$ thousands, log scale")
    ax.set_title(f"Wheel backtest, BTC to Jan 2025 then HYPE, {w.index[0]:%b %Y} to {w.index[-1]:%b %Y} ($100K start)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(IMG / "wheel_equity.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
