"""
Book combination study for the deck (slide 7 + appendix robustness).
Common window Apr 2021-Oct 2026 (wheel data start). Sleeves: E1 (daily->weekly),
CTA (weekly), IDX growth 2-sleeve (weekly), Wheel always-on (weekly from equity).
Book: E1 40 / CTA 20 / IDX 25 / Wheel 15, quarterly rebalanced; VT12 variant on top.
Robustness per sleeve: Newey-West Sharpe (Bartlett, L=8), drop-best-week Sharpe.
"""
import warnings, os, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QP = os.path.dirname(os.path.abspath(__file__))
IR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "idx-report")

e1d = pd.read_csv(os.path.join(QP, "design-a", "final_returns_net.csv"), index_col=0, parse_dates=True).iloc[:, 0]
e1 = (1 + e1d).resample("W-FRI").prod() - 1
cta = pd.read_csv(os.path.join(QP, "returns_CTA.csv"), index_col=0, parse_dates=True).iloc[:, 0]
idx = pd.read_csv(os.path.join(IR, "strat2_growth_weekly.csv"), index_col=0, parse_dates=True).iloc[:, 0]
wh_eq = pd.read_csv(os.path.join(QP, "returns_wheel_always.csv"), index_col=0, parse_dates=True).iloc[:, 0]
wh = wh_eq.pct_change()

S = pd.concat([e1.rename("E1"), cta.rename("CTA"), idx.rename("IDX"), wh.rename("Wheel")], axis=1, join="inner").dropna()
S = S[S.index >= "2021-04-02"]
print(f"Common window: {S.index[0].date()} -> {S.index[-1].date()}, {len(S)} weeks")

def nw_sharpe(r, L=8):
    r = r.dropna(); n = len(r); mu = r.mean(); e = r - mu
    var = (e ** 2).mean()
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        var += 2 * w * (e[l:].values * e[:-l].values).mean()
    return float(mu / np.sqrt(var) * np.sqrt(52))

def stats(r, label=""):
    r = r.dropna()
    eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
    av = r.std(ddof=1) * np.sqrt(52); ar = r.mean() * 52
    yrs = len(r) / 52
    geo = float(eq.iloc[-1] ** (1 / yrs) - 1) * 100
    out = {"sharpe": round(float(ar / av), 2), "geo_cagr%": round(geo, 2), "vol%": round(float(av) * 100, 2),
           "max_dd%": round(float(dd.min()) * 100, 2), "total%": round(float(eq.iloc[-1] - 1) * 100, 1),
           "nw_sharpe": round(nw_sharpe(r), 2)}
    rb = r.drop(r.idxmax())
    out["drop_best_sharpe"] = round(float(rb.mean() * 52 / (rb.std(ddof=1) * np.sqrt(52))), 2)
    print(f"  {label:10s} Sh {out['sharpe']:5.2f} (NW {out['nw_sharpe']:5.2f}, drop-best {out['drop_best_sharpe']:5.2f})  "
          f"geoCAGR {out['geo_cagr%']:6.2f}%  vol {out['vol%']:5.2f}%  maxDD {out['max_dd%']:6.2f}%  total {out['total%']:8.1f}%")
    return out

res = {"sleeves_common": {}}
print("--- sleeves on common window ---")
for c in S.columns:
    res["sleeves_common"][c] = stats(S[c], c)
print("\n--- correlations ---")
C = S.corr().round(2)
print(C.to_string())
res["corr"] = {a: {b: float(C.loc[a, b]) for b in C.columns} for a in C.index}

W = {"E1": 0.40, "CTA": 0.20, "IDX": 0.25, "Wheel": 0.15}
def run_book(vt=False):
    cap = {k: W[k] * 100.0 for k in W}
    out = []
    hist = []
    for i, (wk, row) in enumerate(S.iterrows()):
        if i > 0 and wk.month != S.index[i - 1].month and wk.month in (1, 4, 7, 10) and \
           (S.index[i - 1].month != wk.month):
            tot = sum(cap.values())
            cap = {k: W[k] * tot for k in W}
        lev = 1.0
        if vt and len(hist) >= 26:
            rv = np.std(hist[-26:], ddof=1) * np.sqrt(52)
            lev = float(np.clip(0.12 / rv, 0.0, 2.0)) if rv > 0 else 1.0
        tot0 = sum(cap.values())
        for k in W:
            cap[k] *= (1 + lev * row[k])
        out.append({"week": wk, "ret": sum(cap.values()) / tot0 - 1})
        hist.append(out[-1]["ret"])
    return pd.DataFrame(out).set_index("week")["ret"]

book = run_book(False)
book_vt = run_book(True)
print("\n--- book ---")
res["book"] = stats(book, "Book 40/20/25/15")
res["book_vt12"] = stats(book_vt, "Book + VT12")
res["book_2022"] = round(float(np.prod(1 + book[book.index.year == 2022]) - 1) * 100, 1)
print(f"  Book 2022 total: {res['book_2022']}%")
for nm, a, b in [("Aug 2024 yen shock", "2024-07-26", "2024-08-09"), ("Apr 2025 tariff shock", "2025-04-01", "2025-04-30")]:
    w = book[(book.index > a) & (book.index <= b)]
    print(f"  Book {nm}: {(np.prod(1+w)-1)*100:+.1f}%")
    res[f"book_{nm[:3]}"] = round(float(np.prod(1 + w) - 1) * 100, 1)

fig, ax = plt.subplots(figsize=(10.5, 4.6))
cols = {"E1": "#1f3864", "CTA": "#548235", "IDX": "#c55a11", "Wheel": "#7030a0", "Book": "#000000"}
for c in S.columns:
    (1 + S[c]).cumprod().plot(ax=ax, color=cols[c], lw=1.2, label=f"{c} (sleeve)")
(1 + book).cumprod().plot(ax=ax, color=cols["Book"], lw=2.2, label="BOOK 40/20/25/15")
ax.set_yscale("log"); ax.grid(alpha=0.25); ax.legend(fontsize=9, loc="upper left")
ax.set_title("Four sleeves and the combined book, common window Apr 2021 to Oct 2026 (weekly, net of costs)", fontsize=11)
ax.set_xlabel("")
plt.tight_layout(); plt.savefig(os.path.join(QP, "chart_book.png"), dpi=150)
book.to_csv(os.path.join(QP, "returns_book4.csv"))
with open(os.path.join(QP, "book_study.json"), "w") as f:
    json.dump(res, f, indent=1)
print("\nBOOK STUDY DONE")
