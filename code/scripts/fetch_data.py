"""Download the inputs for E1, dip and the HYPE wheel from yfinance into data/.

    python scripts/fetch_data.py              # everything, up to today
    python scripts/fetch_data.py wheel e1     # just these

The dip universe is the current Nasdaq-100 list from Wikipedia (survivorship-biased, slide 9).
"""
import io
import json
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from book import config  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (QF5210 coursework)"}


def _yf():
    import yfinance as yf   # only needed here, so not in requirements.txt
    return yf


def hyperliquid_daily(coin, start):
    """Daily closes (UTC) from Hyperliquid's public info API."""
    req = {"coin": coin, "interval": "1d", "startTime": int(pd.Timestamp(start).timestamp() * 1000)}
    rows = requests.post("https://api.hyperliquid.xyz/info", json={"type": "candleSnapshot", "req": req},
                         timeout=30).json()
    return pd.Series({pd.to_datetime(c["t"], unit="ms").normalize(): float(c["c"]) for c in rows})


def fetch_wheel():
    yf = _yf()
    d = config.data_dir("wheel")
    d.mkdir(parents=True, exist_ok=True)
    cols = {}
    for t in (config.WHEEL_TICKER, "^IRX"):
        cols[t] = yf.Ticker(t).history(start="2020-09-01", auto_adjust=False)["Close"]
        cols[t].index = cols[t].index.tz_localize(None).normalize()
    px = pd.DataFrame(cols)
    # yfinance skips some HYPE days (30 Oct 2026, for one); fill them from Hyperliquid's daily candles
    hl = hyperliquid_daily("HYPE", px.index[0])
    s = px[config.WHEEL_TICKER]
    first = s.first_valid_index()
    gaps = s.index[(s.index > first) & s.isna()]
    px.loc[gaps, config.WHEEL_TICKER] = hl.reindex(gaps)
    print(f"wheel: filled {px.loc[gaps, config.WHEEL_TICKER].notna().sum()} missing days from Hyperliquid")
    px.index.name = "Date"
    px.to_csv(d / config.WHEEL_PRICE_FILE)
    print(f"wheel: {config.WHEEL_TICKER} {px[config.WHEEL_TICKER].dropna().index[0].date()} -> {px.index[-1].date()}")


def fetch_e1():
    yf = _yf()
    d = config.data_dir("e1")
    d.mkdir(parents=True, exist_ok=True)
    # days when both SGX and the US traded, from 2019: this reproduces the original returns exactly
    px = yf.download(config.E1_SG + config.E1_US, start="2019-01-01", auto_adjust=True,
                     progress=False)["Close"].dropna()
    px.to_csv(d / "prices_big.csv")
    print(f"e1: {px.shape[1]} tickers {px.index[0].date()} -> {px.index[-1].date()}")


def fetch_dip():
    yf = _yf()
    base = config.data_dir("dip")
    bars_dir = base / "data"
    bars_dir.mkdir(parents=True, exist_ok=True)
    html = requests.get("https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies", headers=UA, timeout=30).text
    table = next(t for t in pd.read_html(io.StringIO(html)) if "Ticker" in t.columns and len(t) > 90)
    syms = sorted(table["Ticker"].astype(str))
    json.dump(syms, open(base / "universe.json", "w"), indent=1)
    raw = yf.download([s.replace(".", "-") for s in syms], start="2014-01-01", auto_adjust=True,
                      progress=False, group_by="ticker")
    sources = {}
    for s in syms:
        try:
            b = raw[s.replace(".", "-")].dropna(subset=["Close"])
        except KeyError:
            b = pd.DataFrame()
        if b.empty:
            sources[s] = "FAILED"
            continue
        b = b.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        b.index = b.index.strftime("%Y-%m-%d")
        b.index.name = "date"
        b.to_csv(bars_dir / f"{s}.csv")
        sources[s] = "yfinance"
    json.dump(sources, open(bars_dir / "_sources.json", "w"), indent=1)
    print(f"dip: {sum(v != 'FAILED' for v in sources.values())}/{len(syms)} symbols")


def fetch_benchmarks():
    """S&P 500, BTC and gold daily closes, for the deck's benchmark and factor exhibits."""
    yf = _yf()
    d = config.data_dir("book")
    d.mkdir(parents=True, exist_ok=True)
    px = yf.download(["^GSPC", "BTC-USD", "GC=F"], start="2020-09-01", auto_adjust=True, progress=False)["Close"]
    px.to_csv(d / "benchmarks.csv")
    print(f"benchmarks: {list(px.columns)} -> {px.index[-1].date()}")


def fetch_fx():
    yf = _yf()
    d = config.data_dir("fx")
    d.mkdir(parents=True, exist_ok=True)
    s = yf.Ticker(config.FX_TICKER).history(start="2014-06-01", auto_adjust=False)["Close"]
    s.index = s.index.tz_localize(None).normalize()
    s.rename(config.FX_TICKER).to_frame().rename_axis("Date").to_csv(d / config.FX_PRICE_FILE)
    print(f"fx: {config.FX_TICKER} {s.index[0].date()} -> {s.index[-1].date()}")
    x = yf.Ticker(config.FX_DXY_TICKER).history(start="2014-06-01", auto_adjust=False)["Close"]
    x.index = x.index.tz_localize(None).normalize()
    x.rename(config.FX_DXY_TICKER).to_frame().rename_axis("Date").to_csv(d / config.FX_DXY_FILE)
    print(f"fx: {config.FX_DXY_TICKER} {x.index[0].date()} -> {x.index[-1].date()}")


FETCHERS = {"wheel": fetch_wheel, "e1": fetch_e1, "dip": fetch_dip, "fx": fetch_fx, "book": fetch_benchmarks}

if __name__ == "__main__":
    for name in sys.argv[1:] or FETCHERS:
        FETCHERS[name]()
