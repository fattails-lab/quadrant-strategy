"""Render the Monte Carlo charts (Appendix A10) from outputs/montecarlo/.

    python scripts/run_montecarlo.py --calibration
    python scripts/build_mc_slide.py
    python scripts/build_deck.py

Writes outputs/montecarlo/slide_A10.pdf / .png (a standalone preview of the slide) and
assets/deck/img/mc_fan.png, the fan chart that slide 22 of the HTML deck uses.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from book import config, montecarlo as mc  # noqa: E402

# deck style (sampled from the existing slides)
NAVY, INK, GREY = "#1F3864", "#262626", "#6B6B6B"
PURPLE, PURPLE_BG = "#5B2C8F", "#E9DDF3"
RED = "#C0392B"
FONT = "Helvetica Neue"
W, H = 13.333, 7.5            # inches, 960 x 540 pt like the deck
PAGE_NO = 22

MC_DIR = config.OUTPUTS / "montecarlo"

plt.rcParams.update({"font.family": FONT, "font.size": 10, "axes.edgecolor": "#444444"})


def px(x, y):
    """Deck pixel coordinates (1334 x 750, origin top-left) to figure fraction."""
    return x / 1334, 1 - y / 750


def text(fig, x, y, s, **kw):
    fig.text(*px(x, y), s, va="top", ha=kw.pop("ha", "left"), **kw)


def card(fig, x, y, w, h, big, small):
    x0, y0 = px(x, y + h)
    fig.patches.append(FancyBboxPatch((x0, y0), w / 1334, h / 750, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, fc=PURPLE_BG, ec="none", zorder=0))
    text(fig, x + w / 2, y + 22, big, ha="center", size=17, weight="bold", color=PURPLE)
    text(fig, x + w / 2, y + 52, small, ha="center", size=8.5, color=INK)


def load():
    s = pd.read_csv(MC_DIR / "wheel_forward_summary.csv")
    info = json.load(open(MC_DIR / "wheel_forward_info.json"))
    cal = mc.calibration_summary(pd.read_csv(MC_DIR / "wheel_calibration.csv"))
    paths = {sc: np.load(MC_DIR / f"wheel_paths_{sc}.npy") for sc in mc.SCENARIOS}
    return s, info, cal, paths


def get(s, sc, weeks, col, asset="wheel"):
    return float(s[(s.scenario == sc) & (s.weeks == weeks) & (s.asset == asset)][col].iloc[0])


def plot_fan(ax, paths, info, live, h_short):
    """Base-case paths with their 5-95% band and median, plus the shock scenario's median and 5th percentile."""
    base = (paths["base"] - 1) * 100
    weeks = np.arange(0, base.shape[1] + 1)
    with_start = np.c_[np.zeros(len(base)), base]
    for p in with_start[:config.MC_PLOT_PATHS]:
        ax.plot(weeks, p, color=PURPLE, alpha=0.12, lw=0.7)
    lo, med, hi = np.percentile(with_start, [5, 50, 95], axis=0)
    ax.fill_between(weeks, lo, hi, color=PURPLE, alpha=0.13, lw=0, label="Base: 5 to 95% band")
    ax.plot(weeks, med, color=PURPLE, lw=2.2, label="Base: median")
    shock = np.c_[np.zeros(len(paths["shock"])), (paths["shock"] - 1) * 100]
    ax.plot(weeks, np.median(shock, axis=0), color=RED, lw=1.6, ls="--", label="Shock: median")
    ax.plot(weeks, np.percentile(shock, 5, axis=0), color=RED, lw=1.0, ls=":", label="Shock: 5th percentile")
    ax.axvline(h_short, color=NAVY, lw=1, ls="--")
    ax.text(h_short + 0.15, ax.get_ylim()[1] * 0.92, "assessed\nwindow", color=NAVY, fontsize=8, va="top")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlim(0, weeks[-1])
    ax.set_xlabel(f"Weeks after {live['date']}", fontsize=9)
    ax.set_ylabel("Sleeve return (%)", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower left", frameon=False)
    ax.set_title(f"{config.MC_PLOT_PATHS} of {info['n_paths']:,} simulated paths, always-on wheel from today's position",
                 fontsize=9.5, color=INK)


def save_fan_chart():
    """The fan chart alone, for the HTML deck (assets/deck/img/mc_fan.png)."""
    _, info, _, paths = load()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    plot_fan(ax, paths, info, info["live_state"], config.MC_HORIZONS_WEEKS[0])
    fig.savefig(config.ROOT / "assets" / "deck" / "img" / "mc_fan.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build():
    s, info, cal, paths = load()
    h_short, h_long = config.MC_HORIZONS_WEEKS
    hist = {r["weeks"]: r for r in info["historical"]}
    live = info["live_state"]

    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")

    # header
    text(fig, 55, 32, "APPENDIX A10 · MONTE CARLO FORWARD TEST", size=10.5, weight="bold", color=NAVY)
    text(fig, 55, 58, f"The wheel, the next {h_short} and {h_long} weeks: {info['n_paths']:,} forward paths",
         size=25, weight="bold", color=INK)
    fig.add_artist(plt.Line2D([px(55, 0)[0], px(1278, 0)[0]], [px(0, 123)[1]] * 2, color=NAVY, lw=3.2))

    # stat cards
    cards = [
        (f"{get(s, 'base', h_short, 'P(gain)%'):.0f}%", f"Paths up after {h_short} weeks"),
        (f"{get(s, 'base', h_short, 'median%'):+.1f}%", f"Median {h_short}-week return"),
        (f"{get(s, 'base', h_short, 'p5%'):+.1f}%".replace("-", "−"), f"5th percentile, {h_short} weeks"),
        (f"{get(s, 'base', h_long, 'annualised%'):+.0f}%/yr", f"Compounded, {h_long}-week paths"),
        (f"{cal.loc[h_short, 'inside 5-95%']:.0f}%", "Past outcomes inside 5–95% band"),
    ]
    for i, (big, small) in enumerate(cards):
        card(fig, 55 + i * 248, 137, 231, 78, big, small)

    # fan chart
    ax = fig.add_axes([px(105, 0)[0], px(0, 600)[1], (745 - 105) / 1334, (600 - 268) / 750])
    plot_fan(ax, paths, info, live, h_short)

    # results table
    tx = 785
    cols = [("Scenario", 0, "left"), (f"{h_short}wk\nmedian", 128, "right"), (f"{h_short}wk\nP(up)", 186, "right"),
            (f"{h_long}wk\nmedian", 258, "right"), (f"{h_long}wk\n5th pct", 330, "right"),
            (f"{h_long}wk\nP(up)", 382, "right"), ("Per year\n(compounded)", 472, "right")]
    rows = [(sc.capitalize() if sc != "base" else "Base", [get(s, sc, h_short, "median%"), get(s, sc, h_short, "P(gain)%"),
             get(s, sc, h_long, "median%"), get(s, sc, h_long, "p5%"), get(s, sc, h_long, "P(gain)%"),
             get(s, sc, h_long, "annualised%")])
            for sc in mc.SCENARIOS]
    rows.append(("History", [hist[h_short]["median%"], hist[h_short]["P(gain)%"], hist[h_long]["median%"],
                             hist[h_long]["p5%"], hist[h_long]["P(gain)%"], hist[h_long]["annualised%"]]))
    y = 236
    fig.patches.append(plt.Rectangle(px(tx - 6, y + 34), 484 / 1334, 34 / 750, transform=fig.transFigure, fc=NAVY, ec="none"))
    for name, dx, ha in cols:
        text(fig, tx + dx, y + 4, name, size=8.2, weight="bold", color="white", ha=ha, linespacing=1.1)
    for i, (name, vals) in enumerate(rows):
        yy = y + 34 + i * 25
        if i % 2:
            fig.patches.append(plt.Rectangle(px(tx - 6, yy + 25), 484 / 1334, 25 / 750, transform=fig.transFigure,
                                             fc="#F2F2F2", ec="none"))
        bold = "bold" if name == "Base" else "normal"
        text(fig, tx, yy + 6, name, size=9, color=INK, weight=bold)
        fmt = [f"{vals[0]:+.1f}%", f"{vals[1]:.0f}%", f"{vals[2]:+.1f}%", f"{vals[3]:+.1f}%", f"{vals[4]:.0f}%",
               f"{vals[5]:+.0f}%"]
        for (_, dx, ha), v in zip(cols[1:], fmt):
            text(fig, tx + dx, yy + 6, v.replace("-", "−"), size=9, color=INK, ha=ha, weight=bold)

    # method and calibration
    shock_pct = info["shock_spot%"]
    first, last = (pd.Timestamp(d).strftime("%B %-d, %Y") for d in info["sample"].split(" to "))
    body = (
        f"Paths. Daily HYPE moves and vol changes were block-bootstrapped over all available data, "
        f"{first} through {last}, with blocks averaging {config.MC_BLOCK_DAYS} days. Each move is scaled by "
        f"the previous day's 30-day realized vol, so realized volatility stays tied to the volatility the options are priced at. "
        f"Every path starts from the current book: short a \\${live['open_option'][1]:,.0f} put with HYPE "
        f"at \\${live['spot']:,.2f}.\n\n"
        f"Noise. Every path draws its own option pricing, at 0.83 to 1.0 × the vol proxy, and fills at 85 to 95% of "
        f"model price.\n\n"
        f"Shock. The largest weekly loss in the sample (−{abs(shock_pct):.0f}%) is replayed once, "
        f"at a random point within the horizon.\n\n"
        f"Calibration. Forecasting from {int(cal.loc[h_short, 'n'])} past dates, using only the data available "
        f"then, put {cal.loc[h_short, 'inside 5-95%']:.0f}% of four-week outcomes inside the 5–95% band (target 90%) "
        f"and {cal.loc[h_long, 'inside 5-95%']:.0f}% of thirteen-week ones. {cal.loc[h_long, 'below 5%']:.0f}% of "
        f"thirteen-week outcomes fell below the 5th percentile, so the thirteen-week downside is understated."
    )
    import textwrap
    wrapped = "\n".join("\n".join(textwrap.wrap(p, 95)) if p else "" for p in body.split("\n"))
    text(fig, tx - 6, 380, wrapped, size=8.1, color=INK, linespacing=1.3)

    # takeaway box
    x0, y0 = px(55, 715)
    fig.patches.append(FancyBboxPatch((x0, y0), (1278 - 55) / 1334, 58 / 750, boxstyle="round,pad=0,rounding_size=0.01",
                                      transform=fig.transFigure, fc=PURPLE_BG, ec="none"))
    take = (f"Still working, most likely: {get(s, 'base', h_short, 'P(gain)%'):.0f}% of paths higher after {h_short} weeks, "
            f"{get(s, 'base', h_long, 'annualised%'):.0f}% a year compounded (backtest {hist[h_long]['annualised%']:.0f}%). "
            f"The risk is a crash week: {get(s, 'shock', h_long, 'median%'):.0f}% median at {h_long} weeks."
            .replace(": -", ": −"))
    text(fig, 75, 675, take, size=10, color=INK, weight="bold")

    text(fig, 1300, 718, str(PAGE_NO), size=9, color=GREY, ha="right")

    fig.savefig(MC_DIR / "slide_A10.pdf")
    fig.savefig(MC_DIR / "slide_A10.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    build()
    save_fan_chart()
