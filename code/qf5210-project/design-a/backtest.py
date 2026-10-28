"""
Design A prototype: PCA statistical arbitrage (Avellaneda-Lee style)
QF5210 Group Project - for group discussion, NOT submission.

Method (rolling 60d window, walk-forward, signals at close t earn t+1 residuals):
  1. PCA on correlation matrix of window returns -> top M eigenportfolios as factors
  2. Regress each stock's standardized returns on factor scores -> residuals
  3. Cumulate residuals -> X_i; fit OU via AR(1): X_{n+1} = a + b X_n + z
     kappa=-ln b, m=a/(1-b), sigma_eq=sd(z)/sqrt(1-b^2), s=(X_last-m)/sigma_eq
  4. Rules: open long s<-1.25, open short s>+1.25, close long s>-0.50, close short s<+0.50,
     risk stop |s|>3.5; only names with OU half-life <= 30 days are tradable
  5. Open positions accrue NEXT-day out-of-sample residual returns (loadings fixed from window)
  6. Equal-risk weights (w ~ 1/sigma_eq, gross normalised to 1); costs 10 bps per side on |w|
"""
import warnings, json, os
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "design-a")
os.makedirs(OUT, exist_ok=True)

SG = ["D05.SI", "O39.SI", "U11.SI", "Z74.SI", "S68.SI", "C09.SI", "F34.SI", "G13.SI", "Y92.SI"]
US = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "BAC", "WFC", "XOM",
      "CVX", "UNH", "LLY", "HD", "PG", "COST", "V", "MA", "NFLX", "AVGO",
      "CRM", "ORCL", "ABBV", "MRK", "PEP"]
TICKERS = SG + US

START, END = "2019-01-01", "2026-10-10"
WIN = 60          # estimation window (trading days)
M = 15            # number of PCA factors
S_OPEN, S_CLOSE, S_STOP = 1.25, 0.50, 3.5
HL_MAX = 30.0     # max OU half-life (days) to be tradable
COST_BPS = 10.0   # per side, on position weight

# ---------------------------------------------------------------- data
print("Downloading prices ...")
px = yf.download(TICKERS, start=START, end=END, progress=False, auto_adjust=True)["Close"]
px = px.dropna(axis=1, how="all")
first_valid = px.apply(lambda c: c.first_valid_index())
cutoff = pd.Timestamp("2019-06-01")
keep = [c for c in px.columns if first_valid[c] is not None and first_valid[c] <= cutoff]
dropped = [c for c in px.columns if c not in keep]
px = px[keep].dropna(how="any")   # align SG+US calendars (intersection) - prototype simplification
print(f"Universe kept ({len(keep)}): {keep}")
print(f"Dropped (insufficient history): {dropped}")
print(f"Aligned trading days: {len(px)}  {px.index[0].date()} -> {px.index[-1].date()}")
px.to_csv(os.path.join(OUT, "prices_daily_close.csv"))

R = px.pct_change().dropna().values          # (T, N) simple returns
dates = px.index[1:]
T, N = R.shape
names = list(px.columns)

# ---------------------------------------------------------------- engine
pos = np.zeros(N, dtype=int)                  # -1 short, 0 flat, +1 long
pos_w = np.zeros(N)                           # last held weight per position (exit cost)
entry_idx = np.full(N, -1)
trade_pnl = np.zeros(N)
trade_days = np.zeros(N, dtype=int)
trades = []
port_ret = np.zeros(T)
s_hist = np.full((T, N), np.nan)
n_open_hist = np.zeros(T, dtype=int)
var_explained_last = None

for t in range(WIN, T - 1):
    W = R[t - WIN + 1: t + 1]                 # window ending at t (data <= t only)
    mu, sd = W.mean(axis=0), W.std(axis=0, ddof=1)
    sd = np.where(sd == 0, np.nan, sd)
    Z = (W - mu) / sd
    if np.isnan(Z).any():
        continue
    C = (Z.T @ Z) / (WIN - 1)
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1][:M]
    V = evecs[:, idx]                          # (N, M)
    var_explained_last = evals[idx].sum() / evals.sum()
    F = Z @ V                                  # (WIN, M) factor scores
    B = np.linalg.solve(F.T @ F, F.T @ Z)      # (M, N) loadings
    E = Z - F @ B                              # standardized residuals
    EPS = E * sd                               # residuals in return units
    X = np.cumsum(EPS, axis=0)                 # cumulated idiosyncratic path

    # OU fit per name via AR(1) on X (vectorised OLS per column)
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

    # ---- signal state machine (decisions at close of t) ----
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

    # ---- weights among positions open for t+1 (equal risk, gross = 1) ----
    open_mask = new_pos != 0
    w = np.zeros(N)
    if open_mask.any():
        inv = np.where(open_mask & np.isfinite(sig_eq) & (sig_eq > 0),
                       1.0 / np.where(sig_eq > 0, sig_eq, 1.0), 0.0)
        if inv.sum() > 0:
            w = inv / inv.sum()
    n_open_hist[t + 1] = int(open_mask.sum())

    # ---- out-of-sample residual for day t+1 (window stats/loadings fixed) ----
    z_next = (R[t + 1] - mu) / sd
    f_next = z_next @ V
    eps_next = (z_next - B.T @ f_next) * sd

    # ---- accrue P&L on positions held into t+1; entry costs on new opens ----
    day = float(np.sum(w[open_mask] * new_pos[open_mask] * eps_next[open_mask]))
    for i in range(N):
        if open_mask[i]:
            trade_pnl[i] += w[i] * new_pos[i] * eps_next[i]
            trade_days[i] += 1
            pos_w[i] = w[i]
            if pos[i] == 0:                    # opened at close t
                entry_idx[i] = t + 1
                c = w[i] * COST_BPS / 1e4
                trade_pnl[i] -= c
                day -= c
    # ---- positions closed at close of t: exit cost + record trade ----
    for i in range(N):
        if pos[i] != 0 and new_pos[i] == 0:
            c = pos_w[i] * COST_BPS / 1e4
            trade_pnl[i] -= c
            day -= c
            trades.append({"ticker": names[i], "dir": int(pos[i]),
                           "entry": str(dates[entry_idx[i]].date()) if entry_idx[i] >= 0 else "",
                           "exit": str(dates[t].date()), "days": int(trade_days[i]),
                           "pnl": float(trade_pnl[i])})
            trade_pnl[i] = 0.0; trade_days[i] = 0; entry_idx[i] = -1; pos_w[i] = 0.0
    port_ret[t + 1] = day
    pos = new_pos

# close still-open trades at end (marked, for statistics)
for i in range(N):
    if pos[i] != 0 and entry_idx[i] >= 0:
        trades.append({"ticker": names[i], "dir": int(pos[i]),
                       "entry": str(dates[entry_idx[i]].date()),
                       "exit": str(dates[-1].date()), "days": int(trade_days[i]),
                       "pnl": float(trade_pnl[i]), "open_at_end": True})

# ---------------------------------------------------------------- metrics
ret = pd.Series(port_ret[WIN + 1:], index=dates[WIN + 1:])
eq = (1 + ret).cumprod()
dd = eq / eq.cummax() - 1
ann_ret = ret.mean() * 252
ann_vol = ret.std(ddof=1) * np.sqrt(252)
sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
tr = pd.DataFrame(trades)
metrics = {
    "period": f"{ret.index[0].date()} to {ret.index[-1].date()}",
    "n_names": N, "names": names,
    "ann_return_pct": round(float(ann_ret) * 100, 2),
    "ann_vol_pct": round(float(ann_vol) * 100, 2),
    "sharpe": round(float(sharpe), 2),
    "max_drawdown_pct": round(float(dd.min()) * 100, 2),
    "n_trades": int(len(tr)),
    "win_rate_pct": round(float((tr["pnl"] > 0).mean()) * 100, 1) if len(tr) else None,
    "avg_holding_days": round(float(tr["days"].mean()), 1) if len(tr) else None,
    "avg_profit_per_trade_bps": round(float(tr["pnl"].mean()) * 1e4, 1) if len(tr) else None,
    "total_return_pct": round(float(eq.iloc[-1] - 1) * 100, 2),
    "avg_open_positions": round(float(n_open_hist[WIN + 1:].mean()), 1),
    "factor_var_explained_pct": round(float(var_explained_last) * 100, 1),
    "params": {"window": WIN, "factors": M, "s_open": S_OPEN, "s_close": S_CLOSE,
               "s_stop": S_STOP, "hl_max": HL_MAX, "cost_bps_per_side": COST_BPS},
}
print(json.dumps({k: v for k, v in metrics.items() if k != "names"}, indent=1))

ret.to_csv(os.path.join(OUT, "portfolio_returns.csv"), header=["ret"])
eq.to_csv(os.path.join(OUT, "equity_curve.csv"), header=["equity"])
dd.to_csv(os.path.join(OUT, "drawdown.csv"), header=["dd"])
tr.to_csv(os.path.join(OUT, "trades.csv"), index=False)
pd.DataFrame(s_hist, index=dates, columns=names).to_csv(os.path.join(OUT, "s_scores.csv"))
with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=1)
print("Saved outputs to", OUT)
