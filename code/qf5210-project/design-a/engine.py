"""
Design A engine: PCA statistical arbitrage (Avellaneda-Lee style), parameterised.
Returns metrics + daily net/gross returns + trades + s-score history.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf

def load_prices(tickers, start, end, min_history_date="2019-06-01"):
    px = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)["Close"]
    px = px.dropna(axis=1, how="all")
    fv = px.apply(lambda c: c.first_valid_index())
    cutoff = pd.Timestamp(min_history_date)
    keep = [c for c in px.columns if fv[c] is not None and fv[c] <= cutoff]
    px = px[keep].dropna(how="any")
    return px

def run_backtest(px, WIN=60, M=15, S_OPEN=1.25, S_CLOSE=0.50, S_STOP=3.5,
                 HL_MAX=30.0, COST_BPS=10.0):
    R = px.pct_change().dropna().values
    dates = px.index[1:]
    T, N = R.shape
    names = list(px.columns)
    pos = np.zeros(N, dtype=int); pos_w = np.zeros(N)
    entry_idx = np.full(N, -1)
    trade_pnl = np.zeros(N); trade_days = np.zeros(N, dtype=int)
    trades = []
    port_net = np.zeros(T); port_gross = np.zeros(T)
    s_hist = np.full((T, N), np.nan)
    n_open_hist = np.zeros(T, dtype=int)
    var_exp = np.nan
    wpos_prev = np.zeros(N)                       # signed weights held into the current day

    for t in range(WIN, T - 1):
        W = R[t - WIN + 1: t + 1]
        mu, sd = W.mean(axis=0), W.std(axis=0, ddof=1)
        sd = np.where(sd == 0, np.nan, sd)
        Z = (W - mu) / sd
        if np.isnan(Z).any():
            continue
        C = (Z.T @ Z) / (WIN - 1)
        evals, evecs = np.linalg.eigh(C)
        idx = np.argsort(evals)[::-1][:M]
        V = evecs[:, idx]
        var_exp = evals[idx].sum() / evals.sum()
        F = Z @ V
        B = np.linalg.solve(F.T @ F, F.T @ Z)
        E = Z - F @ B
        EPS = E * sd
        X = np.cumsum(EPS, axis=0)

        x0, x1 = X[:-1], X[1:]
        n = len(x0)
        mx0, mx1 = x0.mean(axis=0), x1.mean(axis=0)
        cov = ((x0 - mx0) * (x1 - mx1)).sum(axis=0) / n
        var0 = ((x0 - mx0) ** 2).sum(axis=0) / n
        with np.errstate(divide="ignore", invalid="ignore"):
            b = np.where(var0 > 0, cov / np.where(var0 > 0, var0, 1.0), np.nan)
            a = mx1 - b * mx0
            resid = x1 - (a + b * x0)
            sd_z = resid.std(axis=0, ddof=1)
            valid = (b > 0) & (b < 1)
            kappa = np.where(valid, -np.log(np.where(valid, b, 1.0)), np.nan)
            m_eq = np.where(valid, a / (1 - np.where(valid, b, 0.0)), np.nan)
            sig_eq = np.where(valid, sd_z / np.sqrt(1 - np.where(valid, b, 0.0) ** 2), np.nan)
            hl = np.where(valid & (kappa > 0), np.log(2) / np.where(kappa > 0, kappa, np.nan), np.nan)
            s = np.where(valid, (X[-1] - m_eq) / np.where(sig_eq > 0, sig_eq, np.nan), np.nan)
        s_hist[t] = s

        tradable = valid & (hl <= HL_MAX) & np.isfinite(s)
        new_pos = pos.copy()
        for i in range(N):
            if not np.isfinite(s[i]):
                continue
            if pos[i] == 0 and tradable[i]:
                if s[i] < -S_OPEN: new_pos[i] = 1
                elif s[i] > S_OPEN: new_pos[i] = -1
            elif pos[i] == 1 and (s[i] > -S_CLOSE or s[i] < -S_STOP):
                new_pos[i] = 0
            elif pos[i] == -1 and (s[i] < S_CLOSE or s[i] > S_STOP):
                new_pos[i] = 0

        open_mask = new_pos != 0
        w = np.zeros(N)
        if open_mask.any():
            inv = np.where(open_mask & np.isfinite(sig_eq) & (sig_eq > 0),
                           1.0 / np.where(sig_eq > 0, sig_eq, 1.0), 0.0)
            if inv.sum() > 0:
                w = inv / inv.sum()
        n_open_hist[t + 1] = int(open_mask.sum())

        z_next = (R[t + 1] - mu) / sd
        f_next = z_next @ V
        eps_next = (z_next - B.T @ f_next) * sd

        wpos = w * new_pos                              # signed weights held into t+1
        gross = float(np.sum(wpos[open_mask] * eps_next[open_mask]))
        # costs on EVERY change in signed weight: entries, exits and daily rebalancing
        # of continuing positions (weights are renormalised each day over the open set)
        c_vec = np.abs(wpos - wpos_prev) * COST_BPS / 1e4
        cost = float(c_vec.sum())
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
                trade_pnl[i] = 0.0; trade_days[i] = 0; entry_idx[i] = -1; pos_w[i] = 0.0
        port_gross[t + 1] = gross
        port_net[t + 1] = gross - cost
        wpos_prev = wpos
        pos = new_pos

    for i in range(N):
        if pos[i] != 0 and entry_idx[i] >= 0:
            trades.append({"ticker": names[i], "dir": int(pos[i]),
                           "entry": str(dates[entry_idx[i]].date()),
                           "exit": str(dates[-1].date()), "days": int(trade_days[i]),
                           "pnl": float(trade_pnl[i]), "open_at_end": True})

    def stats(r):
        eq = (1 + r).cumprod()
        dd = eq / eq.cummax() - 1
        ar, av = r.mean() * 252, r.std(ddof=1) * np.sqrt(252)
        return {"ann_ret%": round(float(ar) * 100, 2), "ann_vol%": round(float(av) * 100, 2),
                "sharpe": round(float(ar / av), 2) if av > 0 else None,
                "max_dd%": round(float(dd.min()) * 100, 2),
                "total%": round(float(eq.iloc[-1] - 1) * 100, 2)}
    ret_net = pd.Series(port_net[WIN + 1:], index=dates[WIN + 1:])
    ret_gross = pd.Series(port_gross[WIN + 1:], index=dates[WIN + 1:])
    tr = pd.DataFrame(trades)
    last_s = s_hist[T - 2] if T >= 2 else np.full(N, np.nan)
    final_positions = [{"ticker": names[i], "dir": int(pos[i]),
                        "weight%": round(float(pos_w[i]) * 100, 1),
                        "s": round(float(last_s[i]), 2) if np.isfinite(last_s[i]) else None}
                       for i in range(N) if pos[i] != 0]
    out = {"net": stats(ret_net), "gross": stats(ret_gross),
           "final_positions": final_positions,
           "n_trades": int(len(tr)),
           "win_rate%": round(float((tr["pnl"] > 0).mean()) * 100, 1) if len(tr) else None,
           "avg_hold_d": round(float(tr["days"].mean()), 1) if len(tr) else None,
           "avg_open": round(float(n_open_hist[WIN + 1:].mean()), 1),
           "var_expl%": round(float(var_exp) * 100, 1)}
    return out, ret_net, ret_gross, tr, pd.DataFrame(s_hist, index=dates, columns=names)
