"""Paths and frozen parameters for every sleeve (slide 12, Appendix A1).

Change a value here and it changes everywhere. Anything changed during the live window goes in the
freeze log with a reason.
"""
from pathlib import Path

# ---------- paths ----------
# Package adaptation (submission package only): the code sits one level deeper here
# (under code/), so ROOT resolves one level higher, to the package root.
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"


def data_dir(sleeve):
    return DATA / sleeve


def output_dir(sleeve):
    path = OUTPUTS / sleeve
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------- book (slide 7) ----------
# 31 Oct 2026: the commodity-stock sleeve was replaced by short IDR; the wheel adds HYPE from Jan 2025.
BOOK_WEIGHTS = {"e1": 0.20, "dip": 0.15, "wheel": 0.50, "fx": 0.15}
BRAKE_HALVE_DD = -0.05   # halve gross exposure at -5% book drawdown
BRAKE_STOP_DD = -0.08    # stop and review at -8%

# ---------- Sleeve 1: E1 PCA stat arb (slide 4) ----------
E1_SG = ["D05.SI", "O39.SI", "U11.SI", "Z74.SI", "S68.SI", "C09.SI", "F34.SI", "G13.SI", "Y92.SI"]
E1_US = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "BAC", "WFC", "XOM",
         "CVX", "UNH", "LLY", "HD", "PG", "COST", "V", "MA", "NFLX", "AVGO",
         "CRM", "ORCL", "ABBV", "MRK", "PEP"]
E1_CONFIG = dict(window=60, n_factors=3, s_open=1.5, s_close=0.50, s_stop=3.5,
                 hl_max=30.0, cost_bps=10.0)

# ---------- Sleeve 2: crypto options wheel (slide 5) ----------
# BTC only from Apr 2021 to 3 Jan 2025 (priced off Deribit DVOL). From then, 70% HYPE (the coin the group's
# tokenomics screen picks, priced off realised vol) and 30% BTC, rebalanced quarterly.
WHEEL_BTC_TICKER = "BTC-USD"
WHEEL_BTC_FILE = "btc_daily.csv"
WHEEL_DVOL_FILE = "dvol_daily.csv"
WHEEL_BTC_START = "2021-04-02"
WHEEL_SWITCH = "2025-01-03"      # last BTC-only Friday; from here the wheel runs HYPE and BTC legs
WHEEL_MIX = {"hype": 0.70, "btc": 0.30}   # from the switch, rebalanced quarterly with the book
# Trend rule (31 Oct 2026), per leg: a coin that closes a Friday below its own 200-day average has its leg
# run the next week at WHEEL_TREND_KEEP of size, the rest in T-bills; back to full size once it closes above.
# BTC halves; HYPE, the altcoin, exits fully. A coin with fewer than 200 days of history runs at full size.
WHEEL_TREND_DAYS = 200
WHEEL_TREND_KEEP = {"btc": 0.5, "hype": 0.0}
WHEEL_TICKER = "HYPE32196-USD"   # Hyperliquid on yfinance, daily closes from 29 Nov 2024
WHEEL_PRICE_FILE = "hype_daily.csv"
WHEEL_CAPITAL = 100_000.0
WHEEL_START = WHEEL_SWITCH
# HYPE has no implied-vol index like Deribit's DVOL, so options are priced at the 30-day realised vol
# times this multiple. 1.0 assumes no volatility risk premium (BTC's DVOL ran at a median 1.18x).
IV_WINDOW = 30
IV_RV_MULT = 1.0
OPTION_DELTA = 0.25
OPTION_T = 7 / 365
PREMIUM_HAIRCUT = 0.95   # sell at 95% of model price (bid side)

# ---------- Sleeve 3: Nasdaq-100 dip buying (slide 6) ----------
DIP_MAX_POS = 10
DIP_DROP = -0.03         # 1-day return threshold
DIP_ATR_PCT = 3.0        # 100 * ATR(5) / close threshold
DIP_LIMIT_ATR = 0.9      # entry limit = close - 0.9 * ATR
DIP_TARGET_ATR = 0.5     # profit target = close + 0.5 * ATR
DIP_MAX_HOLD = 10        # exit at close once held > 10 days
COMM_PER_SHARE, COMM_MIN, COMM_MAX_PCT = 0.0035, 0.35, 0.01
DIP_RUNS = [             # (label, start, end, capital)
    ("A_10k", "2024-01-01", "2026-10-30", 10_000),
    ("A_100k", "2024-01-01", "2026-10-30", 100_000),
    ("B_100k", "2015-01-01", "2026-10-30", 100_000),
]

# ---------- Sleeve 4: short IDR (added 31 Oct 2026) ----------
FX_TICKER = "USDIDR=X"
FX_PRICE_FILE = "usdidr_daily.csv"
FX_START = "2015-01-01"
FX_CAPITAL = 100_000.0
# Cost of holding long USD/IDR forwards: the IDR-minus-USD short-rate gap, as a flat annual rate. Bank
# Indonesia's policy rate sat about 2 to 7 points above US bills over 2015 to 2026; 3.0% is a mid estimate.
FX_CARRY = 0.030
FX_ROLL_BPS = 3.0        # bid-ask on each monthly forward roll
# Dollar filter (31 Oct 2026): hold the short only while the dollar index's FX_MOM_DAYS-day return is
# positive, decided at each close and applied the next day. Each entry or exit pays FX_SWITCH_BPS.
FX_DXY_TICKER = "DX-Y.NYB"
FX_DXY_FILE = "dxy_daily.csv"
FX_MOM_DAYS = 21
FX_SWITCH_BPS = 5.0

# ---------- Monte Carlo forward test (Appendix A10) ----------
MC_PATHS = 2000             # paths for the statistics
MC_PLOT_PATHS = 100         # paths drawn on the chart
MC_HORIZONS_WEEKS = (4, 13)   # the assessed 1-month window, and about 3 months
MC_BLOCK_DAYS = 20            # mean block length for the stationary bootstrap
MC_SEED = 5210
