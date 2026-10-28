"""
Lecture exhibits for the deck appendix (A6-A9):
 A6 factor attribution (OLS of sleeve weekly returns on SPX/BTC/Gold/Coal, common window)
 A7 VaR/ES: historical + EVT (POT/GPD) 1-week 99% per sleeve and book
 A9a Engle-Granger: IDX miners vs their commodity (daily log prices, 2016-2026)
 A9b E1 diagnostics: ADF rejection rate + OU half-life distribution from saved s-scores
 A9c CTA robustness: EWMA vol sizing variant vs 60d baseline
"""
import warnings, os, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from scipy import stats as st

QP = os.path.dirname(os.path.abspath(__file__))
IR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "idx-report")
out = {}

# ---------- assemble sleeves (common window) ----------
e1d = pd.read_csv(os.path.join(QP, "design-a", "final_returns_net.csv"), index_col=0, parse_dates=True).iloc[:, 0]
e1 = (1 + e1d).resample("W-FRI").prod() - 1
cta = pd.read_csv(os.path.join(QP, "returns_CTA.csv"), index_col=0, parse_dates=True).iloc[:, 0]
idx = pd.read_csv(os.path.join(IR, "strat2_growth_weekly.csv"), index_col=0, parse_dates=True).iloc[:, 0]
wh = pd.read_csv(os.path.join(QP, "returns_wheel_always.csv"), index_col=0, parse_dates=True).iloc[:, 0].pct_change()
S = pd.concat([e1.rename("E1"), cta.rename("CTA"), idx.rename("IDX"), wh.rename("Wheel")],
              axis=1, join="inner").dropna()
S = S[S.index >= "2021-04-02"]
book = pd.read_csv(os.path.join(QP, "returns_book4.csv"), index_col=0, parse_dates=True).iloc[:, 0]
book = book.reindex(S.index)

# ---------- A6 factor attribution ----------
px = pd.read_csv(os.path.join(QP, "cta_futures_prices.csv"), index_col=0, parse_dates=True)
ip = pd.read_csv(os.path.join(IR, "idx_prices.csv"), index_col=0, parse_dates=True)
fac = pd.DataFrame({
    "SPX": px["ES=F"].resample("W-FRI").last().pct_change(),
    "BTC": px["BTC-USD"].resample("W-FRI").last().pct_change(),
    "Gold": px["GC=F"].resample("W-FRI").last().pct_change(),
    "Coal": ip["BTU"].resample("W-FRI").last().pct_change(),
}).reindex(S.index)
F = pd.concat([S, book.rename("BOOK"), fac], axis=1, join="inner").dropna()
print(f"Attribution window: {F.index[0].date()} -> {F.index[-1].date()}, n={len(F)}")
out["factors"] = {}
print("\n--- A6 betas (t-stat in parens), R2 ---")
for c in ["E1", "CTA", "IDX", "Wheel", "BOOK"]:
    X = sm.add_constant(F[["SPX", "BTC", "Gold", "Coal"]])
    m = sm.OLS(F[c], X).fit()
    b = {k: [round(float(m.params[k]), 2), round(float(m.tvalues[k]), 1)] for k in ["SPX", "BTC", "Gold", "Coal"]}
    out["factors"][c] = {"betas": b, "R2": round(float(m.rsquared), 2), "alpha_wk%": round(float(m.params["const"]) * 100, 3)}
    print(f"  {c:6s} SPX {b['SPX'][0]:+.2f} ({b['SPX'][1]:+.1f})  BTC {b['BTC'][0]:+.2f} ({b['BTC'][1]:+.1f})  "
          f"Gold {b['Gold'][0]:+.2f} ({b['Gold'][1]:+.1f})  Coal {b['Coal'][0]:+.2f} ({b['Coal'][1]:+.1f})  R2 {m.rsquared:.2f}")

# ---------- A7 VaR / ES ----------
def var_es(r, p=0.99):
    r = r.dropna().values
    var_h = -np.quantile(r, 1 - p)
    es_h = -r[r <= -var_h].mean()
    L = -r
    u = np.quantile(L, 0.90)
    exc = L[L > u] - u
    xi, _, beta = st.genpareto.fit(exc, floc=0)
    n, Nu = len(L), len(exc)
    var_e = u + (beta / xi) * (((1 - p) * n / Nu) ** (-xi) - 1) if abs(xi) > 1e-6 else u + beta * np.log(n / Nu / (1 - p))
    es_e = (var_e + beta - xi * u) / (1 - xi) if xi < 1 else float("nan")
    return {"VaR_hist%": round(float(var_h) * 100, 2), "ES_hist%": round(float(es_h) * 100, 2),
            "VaR_evt%": round(float(var_e) * 100, 2), "ES_evt%": round(float(es_e) * 100, 2),
            "xi": round(float(xi), 2)}
out["vares"] = {}
print("\n--- A7 1-week 99% VaR / ES (losses, %) ---")
for c in ["E1", "CTA", "IDX", "Wheel"]:
    out["vares"][c] = var_es(S[c])
    print(f"  {c:6s} {out['vares'][c]}")
out["vares"]["BOOK"] = var_es(book)
print(f"  BOOK   {out['vares']['BOOK']}")

# ---------- A9a Engle-Granger ----------
print("\n--- A9a Engle-Granger (daily, 2016-2026): name vs commodity ---")
out["eg"] = {}
pairs = [("ADRO.JK", "BTU", ip, "BTU"), ("PTBA.JK", "BTU", ip, "BTU"), ("ITMG.JK", "BTU", ip, "BTU"),
         ("BYAN.JK", "BTU", ip, "BTU"),
         ("ANTM.JK", "Gold", ip, None), ("MDKA.JK", "Gold", ip, None)]
gp = pd.read_csv(os.path.join(IR, "gold_prices.csv"), index_col=0, parse_dates=True)
for name, kind, src, col in pairs:
    if kind == "BTU":
        a, b = np.log(src[name].dropna()), np.log(src["BTU"].dropna())
    else:
        a, b = np.log(src[name].dropna()), np.log(gp["GC=F"].dropna())
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    j = j[j.index >= "2016-01-01"]
    score, pval, _ = coint(j["a"], j["b"])
    X = sm.add_constant(j["b"])
    resid = sm.OLS(j["a"], X).fit().resid
    rv = resid.values
    phi = np.polyfit(rv[:-1], rv[1:], 1)[0]
    hl = float(np.log(2) / -np.log(phi)) if 0 < phi < 1 else float("nan")
    out["eg"][f"{name} vs {kind}"] = {"coint_p": round(float(pval), 3),
                                      "half_life_days": (round(hl) if hl == hl and hl <= 500 else ">500"),
                                      "beta": round(float(sm.OLS(j['a'], X).fit().params[1]), 2)}
    print(f"  {name:9s} vs {kind:4s}: EG p={pval:.3f}  beta {out['eg'][f'{name} vs {kind}']['beta']:+.2f}  spread half-life {hl:.0f}d")

# ---------- A9b E1 diagnostics ----------
ss = pd.read_csv(os.path.join(QP, "design-a", "s_scores.csv"), index_col=0, parse_dates=True)
rej, hls, tested = 0, [], 0
for c in ss.columns:
    x = ss[c].dropna()
    if len(x) < 300:
        continue
    tested += 1
    adf_p = adfuller(x, regression="n", autolag="AIC")[1]
    rej += adf_p < 0.05
    xv = x.values
    phi = np.polyfit(xv[:-1], xv[1:], 1)[0]
    if 0 < phi < 1:
        hls.append(np.log(2) / -np.log(phi))
hls = np.array(hls)
out["e1_diag"] = {"names": int(tested), "adf_reject%": round(float(rej / tested * 100), 1),
                  "hl_median_d": round(float(np.median(hls)), 1),
                  "hl_p25_d": round(float(np.percentile(hls, 25)), 1),
                  "hl_p75_d": round(float(np.percentile(hls, 75)), 1),
                  "hl_le30%": round(float((hls <= 30).mean() * 100), 1)}
print(f"\n--- A9b E1 s-score diagnostics: {out['e1_diag']}")

# ---------- A9c CTA EWMA variant ----------
MKTS = ["ES=F", "ZN=F", "6E=F", "6J=F", "GC=F", "SI=F", "CL=F", "ZC=F", "NG=F"]
P = px[MKTS].ffill(); R = P.pct_change().fillna(0.0)
SIG = pd.DataFrame(0.0, index=P.index, columns=P.columns)
for h in (63, 126, 252):
    SIG += np.sign(P / P.shift(h) - 1.0)
SIG = (SIG / 3.0).fillna(0.0)
VOLE = (np.sqrt((R ** 2).ewm(halflife=30).mean()) * np.sqrt(252)).clip(lower=0.05)
fridays = [f for f in P.resample("W-FRI").last().index if f >= pd.Timestamp("2016-01-01")]
pos_prev = pd.Series(0.0, index=MKTS); rows = []
for i in range(1, len(fridays)):
    f0, f1 = fridays[i - 1], fridays[i]
    pos = SIG.loc[:f0].iloc[-1] * 0.11 / (VOLE.loc[:f0].iloc[-1] * np.sqrt(len(MKTS)))
    cost = 0.0003 * float((pos - pos_prev).abs().sum())
    days = R.loc[(R.index > f0) & (R.index <= f1)]
    rows.append({"week": f1, "ret": float((days * pos).sum().sum()) - cost})
    pos_prev = pos
ev = pd.DataFrame(rows).set_index("week")["ret"]
def shp(r):
    return float(r.mean() * 52 / (r.std(ddof=1) * np.sqrt(52)))
out["cta_ewma"] = {"sharpe_full": round(shp(ev), 2),
                   "ret_2022%": round(float(np.prod(1 + ev[ev.index.year == 2022]) - 1) * 100, 1)}
print(f"\n--- A9c CTA EWMA-vol variant: Sharpe {out['cta_ewma']['sharpe_full']} (base 0.34), "
      f"2022 {out['cta_ewma']['ret_2022%']}% (base +17.4%)")

with open(os.path.join(QP, "lecture_exhibits.json"), "w") as f:
    json.dump(out, f, indent=1)
print("\nLECTURE EXHIBITS DONE")
