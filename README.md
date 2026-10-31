# QF5210 Group Project — Submission Package (finalised 28 Oct 2026)

**Group:** Fat Tails
**Project:** The Quadrant Strategy — Four Market Structures. Four sleeves, weights 40/20/25/15:
E1 PCA mean reversion (40) / Nasdaq-100 dip-buying (20) / IDX coal+nickel (25) /
BTC options wheel (15).
**Canonical book (full-turnover costing, rerun 22 Oct 2026):** Sharpe 1.54, CAGR 15.39%,
max DD −8.12%, total +121.0% (100 → 221), 2022 +14.1%. Historical window:
9 Apr 2021 to 9 Oct 2026, 288 weekly observations. The October 2026 live test is
a separate evaluation ending 30 Oct 2026 and is not blended into these figures.
**Submission deadline:** Sat 31 Oct 2026, 23:59 SGT (Canvas). Presentation Wed 4 Nov, 20:30–20:40.

## Contents
- `slides/` — editable PowerPoint (21 slides; figure audit 23 Oct 2026, claim-accuracy
  passes 28 Oct 2026) + PDF reading copy. Deck MD5 1275c37e28e5fa645aa7c728188ad676.
- `code/` — the canonical post-swap source set (16 .py files, from
  QF5210_source_code_postswap.zip, 22 Oct 2026). In this package, dip_build.py's
  three workspace root constants were made package-relative (28 Oct 2026, no other
  change) and its input files are included in place, so the book construction runs
  from the package itself (see Reproduction).
- `data/` — the brief's data deliverable plus supporting series:
  portfolio_daily_closes.csv (daily closing prices for the full portfolio asset list),
  reference_series.csv, sleeve_and_book_weekly_returns.csv (288 weeks, post-swap book),
  and the series/summaries the deck's figures trace to: dip_swap.json (canonical
  book/sleeve stats), returns_book4_dip.csv, returns_DIP.csv, returns_SPX.csv,
  returns_wheel*.csv, returns_CTA.csv, E1 final_returns_net.csv / final_trades.csv /
  final_summary.json, wheel/cta/garch/lecture summary JSONs, dvol_daily.csv,
  cta_futures_prices.csv. sleeve_and_book_weekly_returns.csv was regenerated on
  28 Oct 2026 from the canonical components (an earlier copy carried the pre-rerun,
  pre-full-costing book and has been replaced); strat2_growth_weekly.csv and
  equity_B_100k.csv are the IDX and DIP inputs to the book construction.
  Note: book_study.json and lecture_exhibits.json are superseded CTA-era records
  kept as the A3 audit trail; the canonical book study is dip_swap.json.
  code/qf5210-project/final/battery.py likewise documents a retired book
  composition and is kept as a record.
- `video/` — QF5210_Quadrant_Video_60s_FINAL.mp4, the 1-minute pitch video (60.0 s);
  the same file is the one whose link goes into the registration spreadsheet.
- `fonts/` — Baloo2.ttf, the display font used on the title slide; install it before
  editing slide 1 so the typography renders as designed.
- `requirements.txt` — Python dependencies for the code, at the versions of the
  environment that produced the canonical results.

## Reproduction
The single canonical book series is `data/returns_book4_dip.csv`. To rebuild it
from the packaged code and inputs:

    python code/qf5210-project/dip_build.py

Expected output: book Sharpe 1.54, Newey-West 1.75, CAGR 15.39%, vol 9.64%,
max DD −8.12%, total +121.0%, 2022 +14.1% (dip_swap.json, regenerated in place).
Verified 28 Oct 2026 in a clean directory outside the original workspace: the
rebuilt returns_book4_dip.csv matched the canonical series on all 288 weeks
(max abs difference 0.0). Scope of this verification: the portfolio aggregation
from the supplied sleeve input series. The sleeve engines
(design-a/final_run.py, wheel_build.py, idx-report/strat2_weekly.py)
are included as the record of how those inputs were produced; the DIP equity
curve (data/equity_B_100k.csv) is supplied as an input series, with its design
specified on slide 7 and A1; a full upstream rerun from raw prices is not
packaged end to end.
Python dependencies are listed in requirements.txt. Run with the pinned versions
(pandas 2.1.4): pandas 3 changes NaN handling in pct_change, which alters the
IHSG comparator printed by code/idx-report/strat2_weekly.py (0.35 / 5.61%
instead of slide 6's 0.25 / 4.1%).

## Group members and contributions
| Member | Matric no. | Contribution |
| --- | --- | --- |
| Andrew Wiraatmaja | A0200755M | Introduction and idea behind making this portfolio; helped reviewing the backtesting results + codes, and helped create the final video for submission. |
| Edward Sujono | A0353761B | Originated the BTC options wheel sleeve, contributed to IDX coal and nickel sleeve, reviewing codes prior to submission. |
| Michael Alexander Japardi | A0357883J | Built the Nasdaq-100 dip buying sleeve, contributed ideas to compositions of each sleeves and combining them into the portfolio. Also tested the allocations and set the 40/20/25/15 weights, ran the backtest + live-test, produced the powerpoint and video drafts, the submission package, and the group's data entries. |
| Victoria Vivian Chan | A0352265E | Assisted with PCA mean reversion and data analysis, testing components of the mean reversion sleeve. |
| Wielly Halim | A0355248X | Backtesting and data analysis of PCA mean reversion, IDX coal and nickel sleeve; helped revise the presentation slides. |

## Live window
The live test period runs 1 to 30 Oct 2026. October test figures are the final
(post-swap) book's returns computed over the whole window from prices; the
brake was not applied to them. The final October tear sheet (the brief's
metrics, per sleeve and for the book) is reported in the registration
spreadsheet and at the presentation on 4 Nov 2026.
