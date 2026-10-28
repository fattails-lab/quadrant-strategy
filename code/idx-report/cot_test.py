"""
Strategy 3 test: CFTC (COT) positioning overlay on gold trend.
  Signal: COMEX gold managed-money net long as % of open interest (disaggregated COT),
          rolling 156-report percentile. Publication lag enforced: a Tuesday report is
          usable at Friday closes at least 7 days later (published Friday, after IDX close).
  Testbeds: (1) gold itself (GLD, trend gate 40wk MA), (2) IDX gold-miner basket
          (ANTM, BRMS, PSAB, MDKA, ARCI) gated on gold's 40wk trend, weekly machinery.
  Overlays: Brake A pctile>=90 -> flat; Brake A-half -> 0.5x; Brake B pctile>=90 AND
          gold 10wk momentum <= 0 -> flat; sensitivity at 85/95; contrarian diagnostic.
"""
import warnings, os, io, json, zipfile, glob
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------- 1. Parse COT ----------
frames = []
for z in sorted(glob.glob(os.path.join(OUT, "cot", "deacot*.zip"))):
    if os.path.getsize(z) < 500_000:
        print(f"skip {os.path.basename(z)} (too small: challenge page)")
        continue
    with zipfile.ZipFile(z) as zf:
        name = zf.namelist()[0]
        raw = zf.read(name).decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(raw), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    g = df[df["CFTC Contract Market Code"].astype(str).str.strip() == "088691"]
    out = pd.DataFrame({"date": pd.to_datetime(g["As of Date in Form YYYY-MM-DD"]),
                        "oi": g["Open Interest (All)"].astype(float),
                        "mm_net": g["Noncommercial Positions-Long (All)"].astype(float)
                                  - g["Noncommercial Positions-Short (All)"].astype(float)})
    frames.append(out)
    print(f"{os.path.basename(z)}: {len(out)} gold reports")
cot = pd.concat(frames).drop_duplicates("date").sort_values("date").reset_index(drop=True)
cot["net_pct_oi"] = cot["mm_net"] / cot["oi"] * 100
# rolling percentile over trailing 156 reports (min 104)
v = cot["net_pct_oi"].values
pct = np.full(len(v), np.nan)
for i in range(len(v)):
    lo = max(0, i - 155)
    w = v[lo:i + 1]
    if len(w) >= 104:
        pct[i] = (w < v[i]).mean() * 100
cot["pctile"] = pct
print(f"COT series: {cot['date'].min().date()} -> {cot['date'].max().date()}, {len(cot)} reports")
print(f"Latest: net {cot['mm_net'].iloc[-1]:,.0f} contracts, {cot['net_pct_oi'].iloc[-1]:.1f}% of OI, "
      f"percentile {cot['pctile'].iloc[-1]:.0f}")
cot.to_csv(os.path.join(OUT, "cot_gold.csv"), index=False)

# ---------- 2. Prices / weekly frames ----------
gp = pd.read_csv(os.path.join(OUT, "gold_prices.csv"), index_col=0, parse_dates=True)
wk = gp.resample("W-FRI").last()
weeks = wk.index
gld = wk["GLD"]; gc = wk["GC=F"]

# lagged COT aligned to Fridays: usable report must be dated <= Friday - 7 days
fdf = pd.DataFrame({"week": weeks, "limit": weeks - pd.Timedelta(days=7)})
cdf = cot[["date", "net_pct_oi", "pctile"]].sort_values("date")
merged = pd.merge_asof(fdf.sort_values("limit"), cdf, left_on="limit", right_on="date",
                       direction="backward").set_index("week").reindex(weeks)
pctile_w = merged["pctile"]
print(f"COT coverage on Fridays: {pctile_w.notna().mean()*100:.0f}% from {weeks[0].date()}")

def stats_w(r, label=""):
    r = r.dropna()
    eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
    av = r.std(ddof=1) * np.sqrt(52); ar = r.mean() * 52
    s = {"sharpe": round(float(ar / av), 2) if av > 0 else None, "cagr%": round(float(ar) * 100, 2),
         "vol%": round(float(av) * 100, 2), "max_dd%": round(float(dd.min()) * 100, 2),
         "total%": round(float(eq.iloc[-1] - 1) * 100, 1)}
    print(f"  {label:38s} Sh {s['sharpe']:>5}  CAGR {s['cagr%']:>6}%  vol {s['vol%']:>5}%  "
          f"maxDD {s['max_dd%']:>7}%  total {s['total%']}%", flush=True)
    return s

def apply_scale(r, s, cost_rate):
    """r: weekly returns of the gated strategy; s: exposure scale decided at prior week-end."""
    s = s.reindex(r.index).ffill().fillna(1.0)
    ds = s.diff().abs().fillna(0.0)
    return s * r - ds * cost_rate, float((s < 1.0).mean() * 100)

# ---------- 3. Testbed 1: gold itself ----------
gld_ret = gld.pct_change().fillna(0.0)
trend = (gld > gld.rolling(40).mean()).astype(float)
pos = trend.shift(1).fillna(0.0)
base_gold = pos * gld_ret - pos.diff().abs().fillna(0.0) * 0.0010
base_gold = base_gold[base_gold.index >= "2016-01-01"]
mom10 = gld.pct_change(10)
res = {}
print("\n--- Testbed 1: GLD trend (40wk) +/- COT overlay ---")
res["gold_base"] = stats_w(base_gold, "Gold trend only")
p = pctile_w.reindex(base_gold.index)
p = p.fillna(0.0)  # no COT history yet -> treat as not braked (same as the other brakes)
brakeA = (p < 90).astype(float)
brakeA_half = np.where(p >= 90, 0.5, 1.0)
brakeB = np.where((p >= 90) & (mom10.reindex(base_gold.index) <= 0), 0.0, 1.0)
for nm, sc, lab in [("gold_A", brakeA, "Brake A: flat if pctile>=90"),
                    ("gold_Ahalf", pd.Series(brakeA_half, index=base_gold.index), "Brake A-half: 0.5x if >=90"),
                    ("gold_B", pd.Series(brakeB, index=base_gold.index), "Brake B: flat if >=90 & mom<=0"),
                    ("gold_A85", (p < 85).astype(float), "Brake A at 85"),
                    ("gold_A95", (p < 95).astype(float), "Brake A at 95")]:
    rr, cut = apply_scale(base_gold, pd.Series(np.asarray(sc, dtype=float), index=base_gold.index), 0.0010)
    res[nm] = stats_w(rr, lab)
    res[nm]["weeks_cut%"] = round(cut, 1)
contra = np.where(p >= 90, 0.0, 1.0)
rr, _ = apply_scale(gld_ret[gld_ret.index >= "2016-01-01"],
                    pd.Series(contra, index=gld_ret[gld_ret.index >= "2016-01-01"].index), 0.0010)
res["gold_contra"] = stats_w(rr, "Contrarian only (long unless >=90, NO trend)")
res["gold_bh"] = stats_w(gld_ret[gld_ret.index >= "2016-01-01"], "GLD buy & hold (ref)")

# ---------- 4. Testbed 2: IDX gold miners ----------
def run_basket_wk(frame, names, gate):
    port_val = 1.0; holdings = {}
    out = []
    idxs = frame.index
    for i in range(1, len(idxs)):
        rp, rn = frame.loc[idxs[i - 1]], frame.loc[idxs[i]]
        total_prev = port_val
        val_hold = sum(sh * rp[t] for t, sh in holdings.items() if pd.notna(rp.get(t)))
        on = bool(gate.iloc[i - 1])
        avail = [n for n in names if pd.notna(rp.get(n)) and rp.get(n, 0) > 0]
        if on and avail:
            cur = {t: (holdings.get(t, 0.0) * rp[t] if pd.notna(rp.get(t)) else 0.0)
                   for t in set(holdings) | set(avail)}
            tgt0 = {n: total_prev / len(avail) for n in avail}
            turn = sum(abs(tgt0.get(t, 0.0) - cur.get(t, 0.0)) for t in set(tgt0) | set(cur))
            cost = 0.0025 * turn
            investable = max(total_prev - cost, 0.0)
            holdings = {n: (investable / len(avail)) / rp[n] for n in avail}
            cash_after = 0.0
        else:
            cost = 0.0025 * val_hold
            holdings = {}
            cash_after = total_prev - cost
        val_now = cash_after
        for t, sh in holdings.items():
            pp = rn.get(t)
            if pd.isna(pp):
                pp = rp.get(t, 0.0)
            val_now += sh * pp
        out.append({"week": idxs[i], "ret": val_now / total_prev - 1 if total_prev > 0 else 0.0})
        port_val = val_now
    return pd.DataFrame(out).set_index("week")["ret"]

MINERS = ["ANTM.JK", "BRMS.JK", "PSAB.JK", "MDKA.JK", "ARCI.JK"]
gold_gate = (gc > gc.rolling(40).mean()).reindex(weeks).fillna(False)
miners_base = run_basket_wk(wk, MINERS, gold_gate)
miners_base = miners_base[miners_base.index >= "2016-01-01"]
miners_ungated = run_basket_wk(wk, MINERS, pd.Series(True, index=weeks))
miners_ungated = miners_ungated[miners_ungated.index >= "2016-01-01"]
print("\n--- Testbed 2: IDX gold miners, gold-trend gate +/- COT overlay ---")
res["miners_base"] = stats_w(miners_base, "Miners + gold trend gate")
res["miners_ungated"] = stats_w(miners_ungated, "Miners ungated (ref)")
pm = pctile_w.reindex(miners_base.index).fillna(0.0)
for nm, sc, lab in [("miners_A", (pm < 90).astype(float), "Miners + Brake A"),
                    ("miners_Ahalf", pd.Series(np.where(pm >= 90, 0.5, 1.0), index=miners_base.index), "Miners + Brake A-half"),
                    ("miners_B", pd.Series(np.where((pm >= 90) & (mom10.reindex(miners_base.index) <= 0), 0.0, 1.0), index=miners_base.index), "Miners + Brake B")]:
    rr, cut = apply_scale(miners_base, pd.Series(np.asarray(sc, dtype=float), index=miners_base.index), 0.0025)
    res[nm] = stats_w(rr, lab)
    res[nm]["weeks_cut%"] = round(cut, 1)

with open(os.path.join(OUT, "cot_test_results.json"), "w") as f:
    json.dump(res, f, indent=1)
print("\nCOT TEST DONE")
