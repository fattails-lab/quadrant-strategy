"""Sleeve 3 · Nasdaq-100 dip buying (slide 6).

Our rebuild of a published dip-buying rule set, "Buying Short-Term Dips in Stocks" (Cracking Markets), verified against its reported results.

  Universe   current Nasdaq-100 constituents (survivorship-biased, slide 9)
  Trend      close > 200-day SMA
  Signal     1-day return < -3% and ATR(5)/close > 3%; ATR(5) = simple mean of the last 5 true ranges
  Entry      next day only, limit = signal close - 0.9 x ATR(5); fill at limit if low <= limit, or at
             the open if it gaps below; unfilled orders expire
  Ranking    ATR(5)/close descending (ties by symbol); max 10 positions, one per symbol; slots counted
             at start of day; an unfilled order uses no slot
  Sizing     prior close equity / 10, whole shares, capped at cash (our reconstruction)
  Exits      from the day after entry, in order: target = signal close + 0.5 x ATR(5) intraday;
             close > prior day's high; close of day 11 held. No stop loss
  Costs      IBKR tiered: $0.0035/share, min $0.35, max 1% of value, on entry and exit
"""
import csv
import json
import math
from datetime import date

from book import config
from book.metrics import save_json, write_rows


# ---------- data ----------
def sma(values, n):
    """Simple moving average via a running sum; None until n values are available."""
    out, run = [None] * len(values), 0.0
    for i, v in enumerate(values):
        run += v
        if i >= n:
            run -= values[i - n]
        if i >= n - 1:
            out[i] = run / n
    return out


def atr(bars, n):
    """Simple mean of the last n true ranges, including today."""
    trs = []
    for i, b in enumerate(bars):
        if i == 0:
            trs.append(b["high"] - b["low"])
        else:
            pc = bars[i - 1]["close"]
            trs.append(max(b["high"] - b["low"], abs(b["high"] - pc), abs(b["low"] - pc)))
    return [sum(trs[i - n + 1:i + 1]) / n if i >= n - 1 else None for i in range(len(bars))]


def load_data():
    """Daily OHLC bars plus SMA(200) and ATR(5) for every symbol that downloaded successfully."""
    base = config.data_dir("dip")
    bars_dir = base / "data"
    syms = json.load(open(base / "universe.json"))
    sources = json.load(open(bars_dir / "_sources.json"))
    data = {}
    for sym in syms:
        path = bars_dir / f"{sym}.csv"
        if sources.get(sym) in (None, "FAILED") or not path.exists():
            continue
        with open(path) as f:
            bars = [{"date": r["date"], "open": float(r["open"]), "high": float(r["high"]),
                     "low": float(r["low"]), "close": float(r["close"]),
                     "volume": float(r["volume"] or 0)} for r in csv.DictReader(f)]
        bars.sort(key=lambda b: b["date"])
        data[sym] = {"bars": bars,
                     "idx": {b["date"]: i for i, b in enumerate(bars)},
                     "sma200": sma([b["close"] for b in bars], 200),
                     "atr5": atr(bars, 5)}
    return data


def commission(shares, value):
    return min(max(config.COMM_PER_SHARE * shares, config.COMM_MIN), config.COMM_MAX_PCT * value)


# ---------- engine ----------
class DipBacktest:
    """Day-by-day simulation. Each day: fill entries, process exits, scan for signals, mark to market."""

    def __init__(self, data, start, end, capital):
        self.data, self.start, self.end, self.capital = data, start, end, capital
        self.cash = capital
        self.positions = {}          # sym -> open position
        self.pending = {}            # entry date -> [limit orders]
        self.trades = []
        self.equity_rows = []
        self.equity_prev_close = capital
        self.last_close = {}         # sym -> last known close, for marking on days without a bar

    def run(self):
        days = sorted({b["date"] for d in self.data.values() for b in d["bars"]
                       if self.start <= b["date"] <= self.end})
        for day in days:
            self._fill_entries(day)
            self._process_exits(day)
            self._scan_signals(day)
            self._mark_to_market(day)
        return self

    def _fill_entries(self, day):
        orders = [o for o in self.pending.pop(day, []) if o["sym"] not in self.positions]
        orders.sort(key=lambda o: (-o["atr_ratio"], o["sym"]))
        slots = config.DIP_MAX_POS - len(self.positions)
        for o in orders:
            if slots <= 0:
                break
            d = self.data[o["sym"]]
            i = d["idx"].get(day)
            if i is None:
                continue
            bar = d["bars"][i]
            if bar["low"] > o["limit"]:
                continue  # not filled: order dies, no slot used
            fill = bar["open"] if bar["open"] < o["limit"] else o["limit"]
            shares = int((self.equity_prev_close / config.DIP_MAX_POS) // fill)
            while shares > 0 and shares * fill + commission(shares, shares * fill) > self.cash:
                shares -= 1
            if shares <= 0:
                continue
            comm = commission(shares, shares * fill)
            self.cash -= shares * fill + comm
            self.positions[o["sym"]] = {**o, "entry_date": day, "entry_idx": i, "entry_price": fill,
                                        "shares": shares, "entry_comm": comm}
            slots -= 1

    def _process_exits(self, day):
        for sym in list(self.positions):
            p = self.positions[sym]
            if p["entry_date"] >= day:
                continue  # exits start the day after entry
            d = self.data[sym]
            i = d["idx"].get(day)
            if i is None:
                continue
            bar = d["bars"][i]
            held = i - p["entry_idx"] + 1
            if bar["high"] >= p["target"]:
                exit_price = bar["open"] if bar["open"] > p["target"] else p["target"]
                reason = "profit_target"
            elif bar["close"] > d["bars"][i - 1]["high"]:
                exit_price, reason = bar["close"], "rebound"
            elif held > config.DIP_MAX_HOLD:
                exit_price, reason = bar["close"], "time"
            else:
                continue
            proceeds = p["shares"] * exit_price
            comm = commission(p["shares"], proceeds)
            self.cash += proceeds - comm
            cost_basis = p["shares"] * p["entry_price"] + p["entry_comm"]
            net = (proceeds - comm) - cost_basis
            self.trades.append({
                "symbol": sym, "signal_date": p["signal_date"],
                "entry_date": p["entry_date"], "entry_price": round(p["entry_price"], 4),
                "shares": p["shares"], "exit_date": day,
                "exit_price": round(exit_price, 4), "exit_reason": reason,
                "holding_days": held, "entry_comm": round(p["entry_comm"], 4),
                "exit_comm": round(comm, 4), "net_pnl": round(net, 2),
                "ret_pct": round(100.0 * net / cost_basis, 3),
            })
            del self.positions[sym]

    def _scan_signals(self, day):
        for sym, d in self.data.items():
            i = d["idx"].get(day)
            if i is None or i < 1 or i + 1 >= len(d["bars"]):
                continue
            sma200, atr5 = d["sma200"][i], d["atr5"][i]
            if sma200 is None or atr5 is None:
                continue
            c, pc = d["bars"][i]["close"], d["bars"][i - 1]["close"]
            if c > sma200 and (c / pc - 1.0) < config.DIP_DROP and 100.0 * atr5 / c > config.DIP_ATR_PCT:
                entry_date = d["bars"][i + 1]["date"]
                if entry_date <= self.end:
                    self.pending.setdefault(entry_date, []).append({
                        "sym": sym, "signal_date": day, "signal_close": c,
                        "atr5_sig": atr5, "atr_ratio": atr5 / c,
                        "limit": c - config.DIP_LIMIT_ATR * atr5, "target": c + config.DIP_TARGET_ATR * atr5,
                    })

    def _mark_to_market(self, day):
        for sym, d in self.data.items():
            i = d["idx"].get(day)
            if i is not None:
                self.last_close[sym] = d["bars"][i]["close"]
        invested = 0.0
        for sym, p in self.positions.items():
            invested += p["shares"] * self.last_close[sym]
        equity = self.cash + invested
        self.equity_rows.append({"date": day, "equity": round(equity, 2),
                                 "cash": round(self.cash, 2), "invested": round(invested, 2),
                                 "n_positions": len(self.positions),
                                 "capital_usage_pct": round(100.0 * invested / equity, 3) if equity else 0.0})
        self.equity_prev_close = equity


def summarise(bt, label):
    """Summary in the article's terms: calendar-day CAGR, daily Sharpe (x sqrt 252), drawdown on closes."""
    eq = [bt.capital] + [r["equity"] for r in bt.equity_rows]
    rets = [eq[i] / eq[i - 1] - 1.0 for i in range(1, len(eq))]
    years = (date.fromisoformat(bt.end) - date.fromisoformat(bt.start)).days / 365.25
    final = eq[-1]
    peak, maxdd = eq[0], 0.0
    for v in eq:
        peak = max(peak, v)
        maxdd = min(maxdd, v / peak - 1.0)
    sharpe = 0.0
    if rets:
        m = sum(rets) / len(rets)
        var = sum((r - m) ** 2 for r in rets) / (len(rets) - 1) if len(rets) > 1 else 0.0
        sharpe = (m / math.sqrt(var)) * math.sqrt(252) if var > 0 else 0.0
    trades, rows = bt.trades, bt.equity_rows
    n = len(trades)
    return {
        "label": label, "start": bt.start, "end": bt.end, "capital": bt.capital,
        "final_equity": round(final, 2), "total_return_pct": round(100 * (final / bt.capital - 1), 2),
        "cagr_pct": round(100 * ((final / bt.capital) ** (1.0 / years) - 1.0), 2),
        "max_dd_pct": round(100 * maxdd, 2),
        "sharpe": round(sharpe, 3), "n_trades": n,
        "win_rate_pct": round(100 * sum(1 for t in trades if t["net_pnl"] > 0) / n, 1) if n else 0.0,
        "avg_holding_days": round(sum(t["holding_days"] for t in trades) / n, 2) if n else 0.0,
        "avg_capital_usage_pct": round(sum(r["capital_usage_pct"] for r in rows) / len(rows), 2) if rows else 0.0,
        "total_net_pnl": round(sum(t["net_pnl"] for t in trades), 2),
        "total_commissions": round(sum(t["entry_comm"] + t["exit_comm"] for t in trades), 2),
        "open_positions_at_end": len(bt.positions),
        "trading_days": len(rows),
        "exit_mix": {r: sum(1 for t in trades if t["exit_reason"] == r)
                     for r in ("profit_target", "rebound", "time")},
    }


# ---------- entry point ----------
def run():
    """The three standard runs (since 2024 at 10K and 100K, long run from 2015); outputs in outputs/dip/."""
    out_dir = config.output_dir("dip")
    data = load_data()
    print(f"loaded {len(data)} symbols")
    results = []
    for label, start, end, capital in config.DIP_RUNS:
        bt = DipBacktest(data, start, end, capital).run()
        stats = summarise(bt, label)
        results.append(stats)
        write_rows(bt.trades, out_dir / f"trades_{label}.csv")
        write_rows(bt.equity_rows, out_dir / f"equity_{label}.csv")
        print(json.dumps(stats, indent=1))
    save_json(results, out_dir / "stats.json")
    return results
