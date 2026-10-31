"""Sleeve 1 · E1 PCA statistical arbitrage (slide 4).

Avellaneda-Lee style. Each day, on a rolling 60-day window:
  1. standardise returns and take the top 3 PCA factors;
  2. regress each name on the factors and model its cumulative residual as an OU process;
  3. the s-score is the residual's distance from its OU mean, in equilibrium standard deviations;
  4. open at |s| > 1.5, close at |s| < 0.5, stop at |s| > 3.5, only if half-life <= 30 days;
  5. weight open names by inverse OU volatility; charge 10 bps on every change in weight.
P&L is the next day's residual return, so the sleeve is factor-neutral by construction.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from book import config
from book.metrics import perf_stats, save_json


# ---------- engine: one window ----------
def fit_window(W, n_factors):
    """Fit PCA factors and an OU model to the residuals of one return window (rows = days).

    Returns None if any name has zero variance in the window (the day is skipped).
    """
    mu, sd = W.mean(axis=0), W.std(axis=0, ddof=1)
    sd = np.where(sd == 0, np.nan, sd)
    Z = (W - mu) / sd
    if np.isnan(Z).any():
        return None

    # PCA on the correlation matrix, then residuals of each name against the factors
    C = (Z.T @ Z) / (len(W) - 1)
    evals, evecs = np.linalg.eigh(C)
    top = np.argsort(evals)[::-1][:n_factors]
    V = evecs[:, top]
    F = Z @ V
    B = np.linalg.solve(F.T @ F, F.T @ Z)
    X = np.cumsum((Z - F @ B) * sd, axis=0)  # cumulative residual, in return units

    # AR(1) on the cumulative residual: x[k+1] = a + b x[k]  ->  OU mean, sigma, half-life
    x0, x1 = X[:-1], X[1:]
    n = len(x0)
    mx0, mx1 = x0.mean(axis=0), x1.mean(axis=0)
    cov = ((x0 - mx0) * (x1 - mx1)).sum(axis=0) / n
    var0 = ((x0 - mx0) ** 2).sum(axis=0) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(var0 > 0, cov / np.where(var0 > 0, var0, 1.0), np.nan)
        a = mx1 - b * mx0
        sd_z = (x1 - (a + b * x0)).std(axis=0, ddof=1)
        valid = (b > 0) & (b < 1)
        kappa = np.where(valid, -np.log(np.where(valid, b, 1.0)), np.nan)
        m_eq = np.where(valid, a / (1 - np.where(valid, b, 0.0)), np.nan)
        sig_eq = np.where(valid, sd_z / np.sqrt(1 - np.where(valid, b, 0.0) ** 2), np.nan)
        half_life = np.where(valid & (kappa > 0), np.log(2) / np.where(kappa > 0, kappa, np.nan), np.nan)
        s = np.where(valid, (X[-1] - m_eq) / np.where(sig_eq > 0, sig_eq, np.nan), np.nan)

    return {"mu": mu, "sd": sd, "V": V, "B": B, "var_exp": evals[top].sum() / evals.sum(),
            "s": s, "valid": valid, "sig_eq": sig_eq, "half_life": half_life}


def next_positions(pos, s, tradable, s_open, s_close, s_stop):
    """Apply the open / close / stop rules to the current s-scores. pos is -1, 0 or +1 per name."""
    new_pos = pos.copy()
    for i in range(len(pos)):
        if not np.isfinite(s[i]):
            continue
        if pos[i] == 0 and tradable[i]:
            if s[i] < -s_open:
                new_pos[i] = 1
            elif s[i] > s_open:
                new_pos[i] = -1
        elif pos[i] == 1 and (s[i] > -s_close or s[i] < -s_stop):
            new_pos[i] = 0
        elif pos[i] == -1 and (s[i] < s_close or s[i] > s_stop):
            new_pos[i] = 0
    return new_pos


def inverse_vol_weights(open_mask, sig_eq):
    """Inverse OU-volatility weights over the open names, summing to 1."""
    w = np.zeros(len(open_mask))
    if open_mask.any():
        inv = np.where(open_mask & np.isfinite(sig_eq) & (sig_eq > 0),
                       1.0 / np.where(sig_eq > 0, sig_eq, 1.0), 0.0)
        if inv.sum() > 0:
            w = inv / inv.sum()
    return w


def next_residual(r_next, fit):
    """Next day's factor residual return for each name, using today's fitted factors."""
    z = (r_next - fit["mu"]) / fit["sd"]
    f = z @ fit["V"]
    return (z - fit["B"].T @ f) * fit["sd"]


# ---------- engine: full backtest ----------
def backtest(px, window=60, n_factors=15, s_open=1.25, s_close=0.50, s_stop=3.5,
             hl_max=30.0, cost_bps=10.0):
    """Run E1 over a price frame. Returns (summary, net returns, gross returns, trades, s-score history)."""
    R = px.pct_change().dropna().values
    dates = px.index[1:]
    T, N = R.shape
    names = list(px.columns)

    pos = np.zeros(N, dtype=int)       # -1 / 0 / +1
    pos_w = np.zeros(N)                # |weight| of open positions, for reporting
    wpos_prev = np.zeros(N)            # signed weights held into the current day
    entry_idx = np.full(N, -1)
    trade_pnl = np.zeros(N)
    trade_days = np.zeros(N, dtype=int)
    trades = []
    port_net, port_gross = np.zeros(T), np.zeros(T)
    s_hist = np.full((T, N), np.nan)
    n_open_hist = np.zeros(T, dtype=int)
    var_exp = np.nan

    for t in range(window, T - 1):
        fit = fit_window(R[t - window + 1: t + 1], n_factors)
        if fit is None:
            continue
        var_exp = fit["var_exp"]
        s = fit["s"]
        s_hist[t] = s

        # signals at close of t, held into t+1
        tradable = fit["valid"] & (fit["half_life"] <= hl_max) & np.isfinite(s)
        new_pos = next_positions(pos, s, tradable, s_open, s_close, s_stop)
        open_mask = new_pos != 0
        wpos = inverse_vol_weights(open_mask, fit["sig_eq"]) * new_pos
        n_open_hist[t + 1] = int(open_mask.sum())

        # P&L on day t+1; cost on every change in signed weight (entries, exits, daily rebalancing)
        eps_next = next_residual(R[t + 1], fit)
        gross = float(np.sum(wpos[open_mask] * eps_next[open_mask]))
        c_vec = np.abs(wpos - wpos_prev) * cost_bps / 1e4

        # trade bookkeeping
        for i in range(N):
            if open_mask[i]:
                trade_pnl[i] += wpos[i] * eps_next[i]
                trade_days[i] += 1
                pos_w[i] = abs(wpos[i])
                if pos[i] == 0:
                    entry_idx[i] = t + 1
            trade_pnl[i] -= c_vec[i]
        for i in range(N):
            if pos[i] != 0 and new_pos[i] == 0:
                trades.append({"ticker": names[i], "dir": int(pos[i]),
                               "entry": str(dates[entry_idx[i]].date()) if entry_idx[i] >= 0 else "",
                               "exit": str(dates[t].date()), "days": int(trade_days[i]),
                               "pnl": float(trade_pnl[i])})
                trade_pnl[i], trade_days[i], entry_idx[i], pos_w[i] = 0.0, 0, -1, 0.0

        port_gross[t + 1] = gross
        port_net[t + 1] = gross - float(c_vec.sum())
        wpos_prev = wpos
        pos = new_pos

    # positions still open at the end
    for i in range(N):
        if pos[i] != 0 and entry_idx[i] >= 0:
            trades.append({"ticker": names[i], "dir": int(pos[i]),
                           "entry": str(dates[entry_idx[i]].date()),
                           "exit": str(dates[-1].date()), "days": int(trade_days[i]),
                           "pnl": float(trade_pnl[i]), "open_at_end": True})

    ret_net = pd.Series(port_net[window + 1:], index=dates[window + 1:])
    ret_gross = pd.Series(port_gross[window + 1:], index=dates[window + 1:])
    tr = pd.DataFrame(trades)
    last_s = s_hist[T - 2] if T >= 2 else np.full(N, np.nan)
    summary = {
        "net": perf_stats(ret_net, 252),
        "gross": perf_stats(ret_gross, 252),
        "final_positions": [{"ticker": names[i], "dir": int(pos[i]),
                             "weight%": round(float(pos_w[i]) * 100, 1),
                             "s": round(float(last_s[i]), 2) if np.isfinite(last_s[i]) else None}
                            for i in range(N) if pos[i] != 0],
        "n_trades": int(len(tr)),
        "win_rate%": round(float((tr["pnl"] > 0).mean()) * 100, 1) if len(tr) else None,
        "avg_hold_d": round(float(tr["days"].mean()), 1) if len(tr) else None,
        "avg_open": round(float(n_open_hist[window + 1:].mean()), 1),
        "var_expl%": round(float(var_exp) * 100, 1),
    }
    return summary, ret_net, ret_gross, tr, pd.DataFrame(s_hist, index=dates, columns=names)


# ---------- data ----------
def load_prices():
    px_big = pd.read_csv(config.data_dir("e1") / "prices_big.csv", index_col=0, parse_dates=True)
    return px_big[[c for c in px_big.columns if c in config.E1_SG + config.E1_US]]


# ---------- reporting ----------
def yearly_table(ret):
    rows = []
    for y, r in ret.groupby(ret.index.year):
        vol = r.std(ddof=1) * np.sqrt(252)
        rows.append({"year": int(y), "ret%": round(float((1 + r).prod() - 1) * 100, 2),
                     "sharpe": round(float(r.mean() * 252 / vol), 2) if vol > 0 else None,
                     "vol%": round(float(vol) * 100, 2)})
    return rows


def sanity_checks(ret, out, trades):
    assert ret.notna().all(), "NaN in net returns"
    print(f"Equity recon {float((1 + ret).prod()):.4f} vs metrics total {out['net']['total%']}%")
    closed = trades[trades["days"] > 0]
    print(f"{len(trades)} trades, {len(closed)} with days>=1, "
          f"bad entries: {int((closed['exit'] < closed['entry']).sum())}")
    trades["sg"] = trades["ticker"].str.endswith(".SI")
    print("Trades by market (sg=True is SGX):", trades.groupby("sg").agg(
        n=("pnl", "size"),
        win=("pnl", lambda p: round(float((p > 0).mean()) * 100, 1)),
        avg_bps=("pnl", lambda p: round(float(p.mean()) * 1e4, 1))).to_dict("index"))


def plot_equity(ret, path):
    eq = (1 + ret).cumprod()
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    axes[0].plot(eq.index, eq.values, lw=1.2, color="#1a5276")
    axes[0].set_ylabel("Growth of $1")
    axes[0].set_title("E1 equity curve (net of costs: 10 bps per weight change), 2019–2026")
    axes[0].grid(alpha=0.3)
    dd = eq / eq.cummax() - 1
    axes[1].fill_between(dd.index, dd.values * 100, 0, color="#c0392b", alpha=0.55)
    axes[1].set_ylabel("Drawdown %")
    axes[1].grid(alpha=0.3)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def plot_sscore(s_hist, trades, path, ticker="D05.SI"):
    """Example signal: s-score with entry/exit bands, open positions shaded."""
    s = s_hist[ticker].loc["2025-01-01":"2026-10-09"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(s.index, s.values, lw=1.0, color="#1a5276", label="DBS s-score")
    for lvl, c, ls in [(1.5, "#c0392b", "--"), (-1.5, "#c0392b", "--"), (0.5, "grey", ":"), (-0.5, "grey", ":")]:
        ax.axhline(lvl, color=c, ls=ls, lw=0.9)
    ax.axhline(0, color="black", lw=0.7)
    for _, t in trades[(trades["ticker"] == ticker) & (trades["entry"] >= "2025-01-01")].iterrows():
        ax.axvspan(pd.Timestamp(t["entry"]), pd.Timestamp(t["exit"]),
                   color="#f39c12" if t["dir"] < 0 else "#27ae60", alpha=0.18)
    ax.set_ylabel("s-score")
    ax.set_title("Example signal — DBS (D05.SI) s-score, 2025–2026  (shaded = open positions, green long / orange short)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


# ---------- entry point ----------
def run():
    """Headline run: backtest, yearly table, sanity checks, charts, outputs in outputs/e1/."""
    out_dir = config.output_dir("e1")
    px = load_prices()
    px.to_csv(out_dir / "prices_daily_close.csv")

    out, ret, _, trades, s_hist = backtest(px, **config.E1_CONFIG)
    print("HEADLINE:", json.dumps(out, indent=1))

    yearly = yearly_table(ret)
    print(pd.DataFrame(yearly).to_string(index=False))
    r24 = ret[ret.index >= "2024-01-01"]
    vol24 = r24.std(ddof=1) * np.sqrt(252)
    print(f"2024-now: ret {float((1 + r24).prod() - 1) * 100:.2f}%  "
          f"Sharpe {float(r24.mean() * 252 / vol24):.2f}  vol {float(vol24) * 100:.2f}%")

    sanity_checks(ret, out, trades)
    plot_equity(ret, out_dir / "chart_equity.png")
    plot_sscore(s_hist, trades, out_dir / "chart_sscore.png")

    save_json({"config": config.E1_CONFIG, "headline": out, "yearly": yearly, "universe": list(px.columns)},
              out_dir / "final_summary.json")
    ret.to_csv(out_dir / "final_returns_net.csv", header=["ret"])
    trades.to_csv(out_dir / "final_trades.csv", index=False)
    return {"summary": out, "returns": ret, "trades": trades, "s_hist": s_hist}
