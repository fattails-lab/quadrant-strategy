"""A8: DVOL (implied) vs GARCH(1,1) forecast vol for BTC. Fit on all data through 2022-12, OOS 2023+."""
import warnings, os, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QP = os.path.dirname(os.path.abspath(__file__))
px = pd.read_csv(os.path.join(QP, "cta_futures_prices.csv"), index_col=0, parse_dates=True)
btc = px["BTC-USD"].dropna()
r = np.log(btc / btc.shift(1)).dropna() * 100.0  # % log returns
dvol = pd.read_csv(os.path.join(QP, "dvol_daily.csv"), index_col=0, parse_dates=True).iloc[:, 0]
dvol.index = pd.to_datetime(dvol.index).normalize()
r.index = pd.to_datetime(r.index).normalize()

train = r.loc[:"2022-12-31"]
def nll(params, x):
    w, a, b = params
    if w <= 0 or a < 0 or b < 0 or a + b >= 0.9999:
        return 1e10
    h = np.empty(len(x)); h[0] = np.var(x)
    for i in range(1, len(x)):
        h[i] = w + a * x[i - 1] ** 2 + b * h[i - 1]
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + x ** 2 / h)
res = minimize(nll, x0=[0.05, 0.08, 0.90], args=(train.values,), method="Nelder-Mead")
w, a, b = res.x
print(f"GARCH(1,1) fit {train.index[0].date()}->{train.index[-1].date()}: omega {w:.4f}, alpha {a:.3f}, beta {b:.3f}, "
      f"persistence {a + b:.3f}, unconditional vol {np.sqrt(w / (1 - a - b) * 365):.1f}% ann.")

# OOS one-step forecasts 2023+
full = r.values; idx = r.index
h = np.empty(len(full)); h[0] = np.var(train)
for i in range(1, len(full)):
    h[i] = w + a * full[i - 1] ** 2 + b * h[i - 1]
fc = pd.Series(np.sqrt(h) * np.sqrt(365), index=idx)  # annualised %
fc = fc.loc["2023-01-01":]
dv = dvol.reindex(fc.index).ffill()
real_trail = (r.rolling(30).std() * np.sqrt(365)).reindex(fc.index)
real_fwd = (r.iloc[::-1].rolling(30).std().iloc[::-1] * np.sqrt(365)).reindex(fc.index)
D = pd.concat([dv.rename("DVOL"), fc.rename("GARCH"), real_trail.rename("RT"), real_fwd.rename("RF")], axis=1).dropna()
stats = {
    "garch_params": {"omega": round(float(w), 4), "alpha": round(float(a), 3), "beta": round(float(b), 3)},
    "oos_from": str(D.index[0].date()), "oos_to": str(D.index[-1].date()), "n": int(len(D)),
    "mean_DVOL%": round(float(D["DVOL"].mean()), 1),
    "mean_GARCH%": round(float(D["GARCH"].mean()), 1),
    "mean_realised_fwd%": round(float(D["RF"].mean()), 1),
    "gap_DVOL_minus_GARCH%": round(float((D["DVOL"] - D["GARCH"]).mean()), 1),
    "pct_days_DVOL_gt_GARCH": round(float((D["DVOL"] > D["GARCH"]).mean() * 100), 1),
    "pct_days_DVOL_gt_realised_fwd": round(float((D["DVOL"] > D["RF"]).mean() * 100), 1),
}
print(stats)

fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(D.index, D["DVOL"], color="#1f3864", lw=1.4, label="DVOL implied vol (what option buyers pay)")
ax.plot(D.index, D["GARCH"], color="#c55a11", lw=1.3, label="GARCH(1,1) forecast vol (fitted to Dec 2022, out-of-sample after)")
ax.plot(D.index, D["RT"], color="gray", lw=1.0, ls="--", label="Realised vol (trailing 30d)")
ax.set_ylabel("Annualised vol (%)"); ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc="upper right")
ax.set_title("BTC volatility: implied vs GARCH forecast vs realised, Jan 2023 to Oct 2026", fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(QP, "chart_garch.png"), dpi=150); plt.close(fig)
json.dump(stats, open(os.path.join(QP, "garch_exhibit.json"), "w"), indent=1)
print("chart_garch.png + garch_exhibit.json saved")
