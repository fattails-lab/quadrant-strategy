"""Sleeve 4 swap: dip-buy strategy replaces CTA.
Weekly returns from the verified rebuild's Run B equity curve ($100K, 2015-2026);
book study, attribution, VaR/ES and weight sensitivity re-run with DIP in place of CTA."""
import warnings, os, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as st

QP = os.path.dirname(os.path.abspath(__file__))
IR = os.path.join(QP, "..", "idx-report")
DIPD = os.path.join(QP, "..", "..", "data")
out = {}

eq = pd.read_csv(os.path.join(DIPD, "equity_B_100k.csv"), index_col=0, parse_dates=True)
dip_w = eq["equity"].resample("W-FRI").last().pct_change().dropna()
dip_w.rename("ret").to_csv(os.path.join(QP, "returns_DIP.csv"))
out["dip_weekly_rows"] = int(len(dip_w))

def stats(r):
    r = r.dropna()
    e = (1 + r).cumprod(); dd = e / e.cummax() - 1
    mu, sd = r.mean(), r.std(ddof=1)
    sh = mu * 52 / (sd * np.sqrt(52))
    n = len(r)
    u = r - mu
    var = (u ** 2).mean()
    for l in range(1, 9):
        var += 2 * (1 - l / 9) * (u.iloc[l:].values * u.iloc[:-l].values).mean()
    nw = mu / np.sqrt(max(var, 1e-12)) * np.sqrt(52)
    r2 = r.drop(index=r.idxmax())
    drop = r2.mean() * 52 / (r2.std(ddof=1) * np.sqrt(52))
    return {"sharpe": round(float(sh), 2), "nw": round(float(nw), 2), "drop": round(float(drop), 2),
            "geoCAGR%": round(float(e.iloc[-1] ** (52 / n) - 1) * 100, 2),
            "vol%": round(float(sd * np.sqrt(52)) * 100, 2), "maxDD%": round(float(dd.min()) * 100, 2),
            "total%": round(float(e.iloc[-1] - 1) * 100, 1)}

full = dip_w[dip_w.index >= "2016-01-01"]
out["dip_full_weekly"] = stats(full)
out["dip_years"] = {int(y): round(float(np.prod(1 + g) - 1) * 100, 1) for y, g in dip_w.groupby(dip_w.index.year) if y >= 2020}
# stress windows from the DAILY equity curve
d = eq["equity"]
def win(a, b):
    s = d.loc[a:b]
    return round(float(s.iloc[-1] / s.iloc[0] - 1) * 100, 1)
out["dip_stress_daily"] = {"covid_Feb19_Mar23_2020": win("2020-02-19", "2020-03-23"),
                           "2022_full": win("2021-12-31", "2022-12-30"),
                           "aug2024": win("2024-08-01", "2024-08-09"),
                           "apr2025": win("2025-04-02", "2025-04-09")}
out["dip_capuse_2022%"] = round(float(eq["capital_usage_pct"].loc["2022-01-01":"2022-12-31"].mean()), 1)
out["dip_capuse_common%"] = round(float(eq["capital_usage_pct"].loc["2021-04-01":"2026-10-09"].mean()), 1)
print("DIP weekly full:", out["dip_full_weekly"])
print("DIP years:", out["dip_years"], "| stress:", out["dip_stress_daily"],
      "| capuse 2022:", out["dip_capuse_2022%"], "common:", out["dip_capuse_common%"])

# ---------- book with DIP ----------
e1d = pd.read_csv(os.path.join(QP, "design-a", "final_returns_net.csv"), index_col=0, parse_dates=True).iloc[:, 0]
e1 = (1 + e1d).resample("W-FRI").prod() - 1
idx = pd.read_csv(os.path.join(IR, "strat2_growth_weekly.csv"), index_col=0, parse_dates=True).iloc[:, 0]
wh = pd.read_csv(os.path.join(QP, "returns_wheel_always.csv"), index_col=0, parse_dates=True).iloc[:, 0].pct_change()
S = pd.concat([e1.rename("E1"), dip_w.rename("DIP"), idx.rename("IDX"), wh.rename("Wheel")],
              axis=1, join="inner").dropna()
S = S[S.index >= "2021-04-02"]
out["window"] = f"{S.index[0].date()} -> {S.index[-1].date()}, {len(S)} weeks"
out["sleeves_common"] = {c: stats(S[c]) for c in S.columns}
out["corr"] = {f"{a}-{b}": round(float(S[a].corr(S[b])), 2)
               for i, a in enumerate(S.columns) for b in S.columns[i + 1:]}
print("common-window sleeves:", json.dumps(out["sleeves_common"], indent=None))
print("corr:", out["corr"])

def run_book(W):
    cap = {k: W[k] * 100.0 for k in W}; rows = []
    for i, (wk, row) in enumerate(S.iterrows()):
        if i > 0 and wk.month in (1, 4, 7, 10) and S.index[i - 1].month != wk.month:
            tot = sum(cap.values()); cap = {k: W[k] * tot for k in W}
        tot0 = sum(cap.values())
        for k in W: cap[k] *= (1 + row[k])
        rows.append(sum(cap.values()) / tot0 - 1)
    return pd.Series(rows, index=S.index)

W = {"E1": .40, "DIP": .20, "IDX": .25, "Wheel": .15}
book = run_book(W)
book.rename("ret").to_csv(os.path.join(QP, "returns_book4_dip.csv"))
out["book"] = stats(book)
out["book_2022%"] = round(float(np.prod(1 + book[book.index.year == 2022]) - 1) * 100, 1)
def wsum(a, b):
    seg = book[(book.index >= a) & (book.index <= b)]
    return round(float(np.prod(1 + seg) - 1) * 100, 1)
out["book_stress"] = {"aug2024_wk_0809": wsum("2024-08-03", "2024-08-09"),
                      "apr2025_0404_0411": wsum("2025-04-04", "2025-04-11")}
print("BOOK(dip):", out["book"], "| 2022:", out["book_2022%"], "| stress:", out["book_stress"])

# vol-target overlay retest (same method as book_study.py: trailing 26 weekly
# returns up to t-1 only, leverage clipped to [0, 2]; no lookahead)
tv = book.rolling(26).std().shift(1) * np.sqrt(52)
lev = (0.12 / tv).clip(0, 2)
bookvt = (book * lev.reindex(book.index).ffill().fillna(1.0)).dropna()
out["book_vt12"] = stats(bookvt)

# weight sensitivity (risk parity from sleeve vols)
vols = S.std() * np.sqrt(52)
inv = (1 / vols) / (1 / vols).sum()
out["parity_weights"] = {k: round(float(v), 2) for k, v in inv.items()}
for lab, w in [("parity", {k: float(inv[k]) for k in inv.index}),
               ("equal", {k: 0.25 for k in S.columns}),
               ("e1heavy", {"E1": .50, "DIP": .15, "IDX": .20, "Wheel": .15})]:
    b = run_book(w); stt = stats(b)
    stt["2022%"] = round(float(np.prod(1 + b[b.index.year == 2022]) - 1) * 100, 1)
    out[f"sens_{lab}"] = stt
print("parity weights:", out["parity_weights"])
for k in ("sens_parity", "sens_equal", "sens_e1heavy"):
    print(k, out[k])

# ---------- factor attribution ----------
px = pd.read_csv(os.path.join(QP, "cta_futures_prices.csv"), index_col=0, parse_dates=True)
ip = pd.read_csv(os.path.join(IR, "idx_prices.csv"), index_col=0, parse_dates=True)
fac = pd.DataFrame({"SPX": px["ES=F"].resample("W-FRI").last().pct_change(),
                    "BTC": px["BTC-USD"].resample("W-FRI").last().pct_change(),
                    "Gold": px["GC=F"].resample("W-FRI").last().pct_change(),
                    "Coal": ip["BTU"].resample("W-FRI").last().pct_change()}).reindex(S.index)
F = pd.concat([S, book.rename("BOOK"), fac], axis=1, join="inner").dropna()
out["factors"] = {}
for c in ["E1", "DIP", "IDX", "Wheel", "BOOK"]:
    m = sm.OLS(F[c], sm.add_constant(F[["SPX", "BTC", "Gold", "Coal"]])).fit()
    out["factors"][c] = {"betas": {k: [round(float(m.params[k]), 2), round(float(m.tvalues[k]), 1)]
                                   for k in ["SPX", "BTC", "Gold", "Coal"]},
                         "R2": round(float(m.rsquared), 2)}
    print("attr", c, out["factors"][c])

# ---------- VaR / ES ----------
def var_es(r):
    r = r.dropna().values
    var_h = -np.quantile(r, 0.01); es_h = -r[r <= -var_h].mean()
    L = -r; u = np.quantile(L, 0.90); exc = L[L > u] - u
    xi, _, beta = st.genpareto.fit(exc, floc=0)
    n, Nu = len(L), len(exc)
    var_e = u + (beta / xi) * (((1 - 0.99) * n / Nu) ** (-xi) - 1) if abs(xi) > 1e-6 else u + beta * np.log(n / Nu / 0.01)
    es_e = (var_e + beta - xi * u) / (1 - xi) if xi < 1 else float("nan")
    return {"VaR_hist%": round(float(var_h) * 100, 2), "ES_hist%": round(float(es_h) * 100, 2),
            "VaR_evt%": round(float(var_e) * 100, 2), "ES_evt%": round(float(es_e) * 100, 2), "xi": round(float(xi), 2)}
out["vares"] = {c: var_es(S[c]) for c in S.columns}
out["vares"]["BOOK"] = var_es(book)
print("VaR/ES:", json.dumps(out["vares"]))

# ---------- chart ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
dd = d / d.cummax() - 1
fig, axes = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
axes[0].plot(d.index, d.values, color="#445e2e", lw=1.2)
axes[0].set_ylabel("Equity (USD 100K start)")
axes[0].set_title("Sleeve 4 — Dip buying (Nasdaq-100), equity curve 2015–2026, net of commissions", fontsize=11)
axes[0].grid(alpha=0.3)
axes[1].fill_between(dd.index, dd.values * 100, 0, color="#c0392b", alpha=0.55)
axes[1].set_ylabel("Drawdown %"); axes[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(QP, "chart_dip.png"), dpi=150); plt.close(fig)

json.dump(out, open(os.path.join(QP, "dip_swap.json"), "w"), indent=1)
print("DIP SWAP STUDY DONE")
