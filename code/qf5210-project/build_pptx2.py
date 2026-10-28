"""PPTX v2: design upgrade (sleeve colour system, kickers, stat chips, verdict strips)
+ lecture appendix A6-A9. Content mirrors deck.tex + lecture_exhibits.json + garch_exhibit.json."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

BASE = os.path.expanduser("~/workspace/qf5210-project")
IDX = os.path.expanduser("~/workspace/idx-report")
NAVY = RGBColor(0x1F, 0x38, 0x64)
PURPLE = RGBColor(0x5B, 0x2D, 0x90)
ORANGE = RGBColor(0xB0, 0x5A, 0x0B)
GREEN = RGBColor(0x44, 0x5E, 0x2E)
DARK = RGBColor(0x26, 0x26, 0x26)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xF2, 0xF2, 0xF2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TINT = {NAVY: RGBColor(0xD9, 0xE1, 0xF2), PURPLE: RGBColor(0xE9, 0xDE, 0xF5),
        ORANGE: RGBColor(0xFB, 0xE7, 0xCE), GREEN: RGBColor(0xE2, 0xED, 0xDA),
        GRAY: RGBColor(0xEC, 0xEC, 0xEC)}
FONT = "Calibri"
prs = Presentation()
prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
MX, CW = 0.55, 12.233

def _rich(p, text, size, bold_all=False, color=DARK):
    for i, seg in enumerate(text.split("**")):
        if not seg:
            continue
        r = p.add_run(); r.text = seg
        r.font.size = Pt(size); r.font.name = FONT
        r.font.bold = bold_all or (i % 2 == 1); r.font.color.rgb = color

def tb(s, x, y, w, h, lines, size=14.5, align=PP_ALIGN.LEFT):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, ln in enumerate(lines):
        if isinstance(ln, str):
            ln = (ln, {})
        text, o = ln
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = o.get("align", align)
        p.space_after = Pt(o.get("space_after", 7)); p.line_spacing = 1.06
        _rich(p, text, o.get("size", size), o.get("bold", False), o.get("color", DARK))
    return box

def bullets(s, x, y, w, h, items, size=14.5):
    return tb(s, x, y, w, h, [("•  " + it, {}) for it in items], size=size)

def header(s, kicker, title, colour=NAVY):
    tb(s, MX, 0.30, CW, 0.3, [(kicker.upper(), {"size": 11, "bold": True, "color": colour, "space_after": 0})])
    tb(s, MX, 0.56, CW, 0.62, [(title, {"size": 26, "bold": True, "color": DARK, "space_after": 0})])
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(MX), Inches(1.20), Inches(CW), Pt(3.5))
    bar.fill.solid(); bar.fill.fore_color.rgb = colour; bar.line.fill.background(); bar.shadow.inherit = False

def chips(s, items, colour, y=1.36, vsize=18):
    n = len(items); gap = 0.18
    w = (CW - gap * (n - 1)) / n
    made = []
    for i, (val, lab) in enumerate(items):
        x = MX + i * (w + gap)
        sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.80))
        made.append(sh)
        sh.fill.solid(); sh.fill.fore_color.rgb = TINT[colour]; sh.line.fill.background(); sh.shadow.inherit = False
        tf = sh.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.08); tf.margin_top = Inches(0.05); tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        _rich(p, val, vsize, True, colour)
        p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
        _rich(p2, lab, 9.5, False, GRAY)
    return made

def verdict(s, text, colour):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(MX), Inches(6.58), Inches(CW), Inches(0.58))
    sh.fill.solid(); sh.fill.fore_color.rgb = TINT[colour]; sh.line.fill.background(); sh.shadow.inherit = False
    tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.18)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    _rich(p, text, 13, False, DARK)

def pointer(s, text):
    tb(s, MX, 6.26, CW, 0.3, [(text, {"size": 10.5, "color": GRAY, "space_after": 0})])

def table(s, x, y, w, rows, col_ws, font=13.5, header_row=True, bold_rows=(), text_table=False,
          hcolor=NAVY, bold_cells=(), cell_colors={}, cell_fills={}):
    """Drawn grid (rectangles + text boxes) instead of a native table shape:
    some preview renderers drop native tables; shapes render everywhere."""
    import math
    line_h = font * 1.24 / 72.0
    # estimate wrapped lines per cell
    heights = []
    for ri, row in enumerate(rows):
        mx_lines = 1
        for ci, val in enumerate(row):
            txt = str(val)
            cpl = max(5.0, (col_ws[ci] - 0.16) * 132.0 / font)
            mx_lines = max(mx_lines, math.ceil(len(txt) / cpl))
        heights.append(max(0.30, mx_lines * line_h + 0.075))
    cy = y
    for ri, row in enumerate(rows):
        cx = x
        for ci, val in enumerate(row):
            cw_in = col_ws[ci]
            rect = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(cx), Inches(cy), Inches(cw_in), Inches(heights[ri]))
            rect.shadow.inherit = False
            if ri == 0 and header_row:
                rect.fill.solid(); rect.fill.fore_color.rgb = hcolor
            elif (ri, ci) in cell_fills:
                rect.fill.solid(); rect.fill.fore_color.rgb = cell_fills[(ri, ci)]
            else:
                rect.fill.solid(); rect.fill.fore_color.rgb = WHITE if ri % 2 else LIGHT
            rect.line.color.rgb = WHITE; rect.line.width = Pt(1.0)
            box = s.shapes.add_textbox(Inches(cx + 0.07), Inches(cy), Inches(cw_in - 0.14), Inches(heights[ri]))
            tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            if text_table or ci == 0:
                p.alignment = PP_ALIGN.LEFT
            elif ci > 0:
                p.alignment = PP_ALIGN.RIGHT
            r = p.add_run(); r.text = str(val)
            r.font.name = FONT; r.font.size = Pt(font)
            if ri == 0 and header_row:
                r.font.bold = True; r.font.color.rgb = WHITE
            else:
                r.font.color.rgb = cell_colors.get((ri, ci), DARK)
                if ri in bold_rows or (ri, ci) in bold_cells:
                    r.font.bold = True
            cx += cw_in
        cy += heights[ri]
    return None

def pic(s, path, x, y, w, caption=None):
    p = s.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w))
    if caption:
        tb(s, x, y + p.height / 914400 + 0.08, w, 0.9, [(caption, {"size": 10.5, "color": GRAY})], size=10.5)
    return p

def pagenum(s, n):
    tb(s, 12.55, 7.14, 0.5, 0.3, [(str(n), {"size": 10, "color": GRAY, "align": PP_ALIGN.RIGHT})], size=10)

def new_slide(kicker, title, colour=NAVY):
    s = prs.slides.add_slide(BLANK)
    header(s, kicker, title, colour)
    return s

def speaker(s, label, colour=NAVY):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.30), Inches(0.20), Inches(2.48), Inches(0.38))
    sh.fill.solid(); sh.fill.fore_color.rgb = TINT[colour]; sh.line.fill.background(); sh.shadow.inherit = False
    tf = sh.text_frame; tf.word_wrap = False
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    _rich(p, label, 10.5, True, colour)

def note(s, text):
    ns = s.notes_slide
    ns.placeholders[1].text_frame.text = text

# ================= 1. TITLE — The Quadrant Strategy (Baloo 2 typography, approved 23 Oct) =================
from lxml import etree as _etree
HA = os.path.expanduser("~/workspace/qf5210-project/hero_preview")
H_BODY = RGBColor(0x40, 0x40, 0x40); H_CREAM = RGBColor(0xFA, 0xF5, 0xEC)
H_LINEC = RGBColor(0xE3, 0xD9, 0xC6); H_DIVC = RGBColor(0xDC, 0xD0, 0xB4); H_SHADOW = RGBColor(0xEA, 0xE0, 0xCB)
H_GOLD = RGBColor(0xC0, 0x8A, 0x1E); H_CORAL = RGBColor(0xC0, 0x39, 0x2B)
H_DISPLAY = "Baloo 2"

def h_rect(s, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, adj=None, lw=1.0):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if adj is not None:
        try: sh.adjustments[0] = adj
        except Exception: pass
    if fill is None: sh.fill.background()
    else: sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None: sh.line.fill.background()
    else: sh.line.color.rgb = line; sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    return sh

def h_tb(s, x, y, w, h, paras, align=PP_ALIGN.LEFT):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, runs in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = runs[0][1].get("align", align)
        p.line_spacing = runs[0][1].get("ls", 1.06)
        p.space_after = Pt(runs[0][1].get("sa", 0))
        for text, o in runs:
            r = p.add_run(); r.text = text
            r.font.size = Pt(o.get("size", 12)); r.font.bold = o.get("bold", False)
            r.font.italic = o.get("italic", False)
            r.font.name = o.get("font", FONT); r.font.color.rgb = o.get("color", H_BODY)
            if o.get("spc"):
                r.font._rPr.set("spc", str(o["spc"]))
    return box

s = prs.slides.add_slide(BLANK)
h_rect(s, 0, 0, 13.333, 7.5, H_CREAM)
# corner labels (overline QF5210 / GROUP PROJECT, as corner labels)
h_tb(s, 0.55, 0.30, 3.0, 0.4, [[("QF5210", {"size": 16, "bold": True, "font": H_DISPLAY, "color": NAVY})]])
h_tb(s, 9.28, 0.30, 3.5, 0.4, [[("GROUP PROJECT", {"size": 16, "bold": True, "font": H_DISPLAY, "color": NAVY, "align": PP_ALIGN.RIGHT})]])
# hero illustration video (right): Quadrant emblem; the wheel turns in quarter-turns
h_rect(s, 8.42, 1.20, 4.43, 5.54, H_SHADOW)
illo_movie = s.shapes.add_movie(os.path.join(HA, "illustration_loop.mp4"),
                                Inches(8.35), Inches(1.12), Inches(4.43), Inches(5.54),
                                poster_frame_image=os.path.join(HA, "illustration_poster.png"))
h_rect(s, 8.35, 1.12, 4.43, 5.54, None, line=H_LINEC, lw=1.0)
# left column: eyebrow + oversized display words
h_tb(s, 0.57, 1.24, 6.5, 0.5, [[("THE", {"size": 21, "bold": True, "font": H_DISPLAY, "color": NAVY, "spc": 900})]])
h_tb(s, 0.50, 1.62, 7.75, 1.55, [[("Quadrant", {"size": 92, "bold": True, "font": H_DISPLAY, "color": NAVY, "ls": 1.0})]])
h_tb(s, 0.50, 3.12, 7.75, 1.55, [
    [("Strategy", {"size": 92, "bold": True, "font": H_DISPLAY, "color": H_GOLD, "ls": 1.0}),
     (".", {"size": 92, "bold": True, "font": H_DISPLAY, "color": H_CORAL, "ls": 1.0})]])
h_tb(s, 0.57, 4.72, 6.6, 0.42, [[("Four Market Structures", {"size": 18, "bold": True, "color": NAVY})]])
h_tb(s, 0.57, 5.08, 6.9, 0.32, [[("One book, four engines \u2014 each harvesting a different named premium.", {"size": 12.5, "color": H_BODY})]])
_stat_runs = []
for _v, _l in [("1.54", "Sharpe"), ("15.4%", "CAGR"), ("\u22128.1%", "Max DD"), ("+14.1%", "in 2022")]:
    if _stat_runs: _stat_runs.append(("   \u00b7   ", {"size": 12, "color": GRAY}))
    _stat_runs.append((_v + " ", {"size": 13, "bold": True, "color": NAVY}))
    _stat_runs.append((_l, {"size": 12, "color": GRAY}))
h_tb(s, 0.57, 5.44, 6.6, 0.32, [_stat_runs])
# chart-loop card (slim): the book's equity loop
h_rect(s, 0.65, 5.94, 7.30, 1.02, H_SHADOW, shape=MSO_SHAPE.ROUNDED_RECTANGLE, adj=0.10)
h_rect(s, 0.57, 5.86, 7.30, 1.02, WHITE, line=H_LINEC, shape=MSO_SHAPE.ROUNDED_RECTANGLE, adj=0.10, lw=1.0)
chart_movie = s.shapes.add_movie(os.path.join(HA, "chart_loop.mp4"),
                                 Inches(0.70), Inches(5.945), Inches(2.55), Inches(0.86),
                                 poster_frame_image=os.path.join(HA, "poster_chart.png"))
h_tb(s, 3.45, 6.02, 4.30, 0.75, [
    [("The book, animated", {"size": 10.5, "bold": True, "color": NAVY, "sa": 2})],
    [("Equity curve, Apr 2021 \u2014 Oct 2026, net of costs.  ", {"size": 9.5, "color": GRAY}),
     ("+121.0% total", {"size": 9.5, "bold": True, "color": NAVY})]])
# bottom row: names left, course line right
h_tb(s, 0.57, 7.06, 8.2, 0.3, [[("Andrew Wiraatmaja \u00b7 Edward Sujono \u00b7 Michael Japardi \u00b7 Victoria Chan \u00b7 Wielly Halim", {"size": 10.5, "color": H_BODY})]])
h_tb(s, 8.35, 7.06, 4.43, 0.3, [[("QF5210 \u00b7 Financial Time Series \u00b7 Oct 2026", {"size": 9.5, "color": GRAY, "align": PP_ALIGN.RIGHT})]])
_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
# autoplay + loop: edit the native timing python-pptx generated for the movies
# (hand-rolled timing trees are rejected by PowerPoint; native shape is required)
_timing_el = s._element.find(f"{{{_P}}}timing")
if _timing_el is not None:
    for _mn in _timing_el.iter(f"{{{_P}}}cMediaNode"):
        _ctn = _mn.find(f"{{{_P}}}cTn")
        _ctn.set("repeatCount", "indefinite")
        _cond = _ctn.find(f"{{{_P}}}stCondLst/{{{_P}}}cond")
        if _cond is not None:
            _cond.set("delay", "0")
note(s, "Speaker 1 of 5: Andrew. You open the deck (title), then slides 2 (Hook) and 3 (Why a book). About 2 minutes for your three slides. Hand over to Edward for Sleeve 1. Title slide: 'The Quadrant Strategy' in the Baloo 2 display treatment (approved 23 Oct); both videos on it (illustration with rotating quadrant wheel, chart card) autoplay and loop in PowerPoint slideshow mode.")


# ================= 2. THE HOOK =================
s = new_slide("The hook · Why this book, why now", "Tariffs, a yen unwind, a crypto winter")
speaker(s, "SPEAKER 1 · ANDREW")
note(s, "Speaker 1 of 5: Andrew (slides 2-3). Hook: target about 50 seconds. Five regimes, one line each; land the closing line at the bottom.")
tb(s, MX, 1.34, CW, 0.4, [("Five regimes in five years. Each paid a different premium and punished a different flaw. Here is which sleeve answered each one.", {"size": 14, "color": GRAY})])
_hook = [
    (NAVY, "2021–22 · Crypto winter", "BTC fell 74% peak-to-trough; buy-and-hold wore all of it.", "Wheel: max drawdown −43%, and USD 476K of modelled premium across the window."),
    (GREEN, "2022 · The rate shock", "Stocks and bonds fell together; the classic hedge failed.", "IDX carried the year: coal and nickel +98.7% on the energy shock. E1 −2.2%, the wheel bled, dip sat in cash (−1.1%). Book: +14.1%."),
    (NAVY, "Aug 2024 · The yen unwind", "A three-day global deleveraging as the carry trade burst.", "Book: +0.4%. Nothing in the book trades the yen; nothing had to."),
    (PURPLE, "Apr 2025 · The Trump tariff shock", "Policy whiplash: tariffs announced, paused, re-announced within weeks.", "The dip sleeve bought the tariff dip (+9.3% in a week). Book: +1.1%."),
    (ORANGE, "2023–26 · The gold supercycle", "Speculative positioning pinned at the 79th percentile (Oct 2026).", "IDX loads on gold (β +0.38); the gold basket and COT brake are design variants outside the book."),
]
for i, (col, lab, mid, ans) in enumerate(_hook):
    yy = 1.80 + i * 0.93
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(MX), Inches(yy), Inches(CW), Inches(0.84))
    sh.fill.solid(); sh.fill.fore_color.rgb = TINT[col]; sh.line.fill.background(); sh.shadow.inherit = False
    tb(s, MX + 0.2, yy + 0.10, 3.3, 0.6, [(lab, {"size": 14, "bold": True, "color": col, "space_after": 0})])
    tb(s, MX + 3.6, yy + 0.07, 4.9, 0.72, [(mid, {"size": 12, "space_after": 0})], size=12)
    tb(s, MX + 8.65, yy + 0.07, 3.43, 0.72, [(ans, {"size": 12, "space_after": 0})], size=12)
tb(s, MX, 6.46, CW, 0.4, [("No single strategy survived all five regimes. The book is built so no single headline can fire every sleeve at once.", {"size": 14, "bold": True, "align": PP_ALIGN.CENTER})])
pagenum(s, 2)

# ================= 3. WHY A BOOK =================
s = new_slide("The book · Why four engines", "Why a book, not a strategy")
speaker(s, "SPEAKER 1 · ANDREW")
note(s, "Speaker 1 of 5: Andrew (last of your pair). Target about 50 seconds: correlations -0.09 to +0.23 are the point. Hand over to Edward.")
table(s, MX, 1.42, CW,
      [["Sleeve", "Premium", "Market", "What kills it"],
       ["1. Mean reversion", "Relative value", "SGX + US equities", "Crowding; regime change"],
       ["2. BTC wheel", "Volatility", "Bitcoin options", "A full bear market, held naked"],
       ["3. IDX commodities", "Commodity beta", "Indonesian equities", "Commodity downcycles; IDR"],
       ["4. Dip buying", "Short-term reversal", "Nasdaq-100 equities", "A bear that keeps dipping; no stop loss"]],
      [2.6, 2.2, 3.1, 4.33], font=15, text_table=True,
      cell_colors={(1, 0): NAVY, (2, 0): PURPLE, (3, 0): ORANGE, (4, 0): GREEN})
tb(s, MX, 3.72, CW, 1.5,
   [("Measured on the common window (Apr 2021 to Oct 2026), every pair of sleeves correlates between **−0.09 and +0.23**. The four signals are differentiated, not independent — the wheel carries BTC beta and IDX loads on gold — so the book can be more than the sum of its parts, if it is sized honestly.", {}),
    ("The shared flaw, stated before the results: sleeves 2, 3 and 4 are all long risk. What stands against a crash is Sleeve 1’s near-zero measured factor exposure (R² 0.01), Sleeve 4’s trend filter (it held −1.1% in 2022 by sitting mostly in cash), and the book brake. The portfolio slide tests whether that is enough.", {})], size=15)
chips(s, [("1.54", "Book Sharpe (common window)"), ("≤ 0.23", "Highest pairwise sleeve correlation"),
          ("+14.1%", "Book return in 2022"), ("−8.1%", "Book max drawdown")], NAVY, y=5.42)
pagenum(s, 3)

# ================= 3. E1 =================
s = new_slide("Sleeve 1 · Relative value", "Mean reversion engine (PCA statistical arbitrage)", NAVY)
speaker(s, "SPEAKER 2 · EDWARD")
note(s, "Speaker 2 of 5: Edward (slides 4-5). E1: target about 50 seconds. The costs line matters: 10 bps on every weight change. Since-2024 Sharpe 0.06 is the disclosed flaw.")
chips(s, [("0.67", "Sharpe, net (2019–2026)"), ("−9.1%", "Max drawdown"), ("1,541", "Trades · 59.1% win rate"), ("0.06", "Sharpe since 2024 — decaying")], NAVY)
tb(s, MX, 2.36, 6.9, 1.1,
   [("**Frozen spec.** Residuals from a 3-factor PCA model (60-day window) on 34 names (9 SGX + 25 US). Trade the OU s-score: open at ±1.5, close at ∓0.5, stop at |s| > 3.5, half-life ≤ 30 days. Costs: 10 bps on every change in position weight (entries, exits and daily rebalancing).", {})], size=14)
table(s, MX, 3.5, 6.9,
      [["Net results, Apr 2019 to Oct 2026", ""],
       ["Return", "5.6%/yr"], ["Sharpe", "0.67"], ["Volatility", "8.5%"],
       ["Max drawdown", "−9.1%"], ["Trades / win rate", "1,541 / 59.1%"],
       ["Avg hold / profit per trade", "11.8 days / +2.7 bps"]],
      [4.6, 2.3], font=13.5, hcolor=NAVY)
pic(s, os.path.join(BASE, "design-a", "chart_equity.png"), 7.75, 2.36, 5.03,
    "Equity curve and drawdown, net of costs. The engine made money through the 2020 crash; 2022 was a small loss (−2.2%) once every rebalance is charged, and its flat years are recent.")
pointer(s, "Lecture exhibits: factor attribution (A6) · cointegration and residual diagnostics (A9)")
verdict(s, "**The flaw, on the record:** the edge is real and decaying, so this engine is the book's ballast at 40%, not its return promise.", NAVY)
pagenum(s, 4)

# ================= 4. WHEEL =================
s = new_slide("Sleeve 2 · Volatility premium", "Bitcoin options wheel", PURPLE)
speaker(s, "SPEAKER 2 · EDWARD", PURPLE)
note(s, "Speaker 2 of 5: Edward (last of your pair). Wheel: about 50 seconds. The regime wheel is the kill story: -60.7% in 2022 vs -32.0% always-on. Hand over to Wielly.")
chips(s, [("0.65 vs 0.38", "Sharpe vs buy-and-hold"), ("$476K", "Modelled gross premium, 289 sales"), ("−43% vs −74%", "Max DD vs buy-and-hold"), ("≈22%", "1-wk 99% ES — the tail (A7)")], PURPLE)
tb(s, MX, 2.36, 6.9, 1.05,
   [("**Mechanism.** Weekly cycle, 25-delta options, 1 week to expiry, priced off Deribit's DVOL index. Flat: sell cash-secured puts. Holding: sell covered calls. Cash earns the T-bill rate. Apr 2021 to Oct 2026, USD 100K.", {})], size=14)
table(s, MX, 3.48, 6.9,
      [["", "Sharpe", "CAGR", "Max DD"],
       ["Regime wheel (first design)", "0.48", "11.3%", "−63.3%"],
       ["Always-on wheel (adopted)", "0.65", "17.2%", "−43.4%"],
       ["BTC buy and hold", "0.38", "6.2%", "−74.3%"]],
      [2.9, 1.3, 1.3, 1.4], font=13.5, bold_rows=(2,), hcolor=PURPLE)
tb(s, MX, 4.95, 6.9, 1.3,
   [("**What the test killed:** the first design sold puts only in bear regimes and calls only in bull regimes. In 2022 it returned −60.7%, close to buy-and-hold (−64%): once assigned, it sold nothing and rode the bear naked. The plain wheel kept harvesting calls through the decline and lost −32.0%.", {})], size=13.5)
pic(s, os.path.join(BASE, "chart_wheel.png"), 7.75, 2.36, 5.03,
    "Adopted vs rejected vs buy-and-hold. The regime overlay is kept in the rejection log (A3), not buried.")
pointer(s, "Lecture exhibits: implied vs GARCH volatility (A8) · tail risk, EVT (A7)")
verdict(s, "**Always-on, never idle:** the regime filter was the flaw. Short volatility by construction, so it stays the smallest sleeve at 15%. Its 17.2% CAGR is the wheel's own window from 2 Apr 2021; on the book's common window it is 16.7%.", PURPLE)
pagenum(s, 5)

# ================= 5. IDX =================
s = new_slide("Sleeve 3 · Commodity beta", "IDX commodity sleeve (weekly signals)", ORANGE)
speaker(s, "SPEAKER 3 · WIELLY", ORANGE)
note(s, "Speaker 3 of 5: Wielly (slides 6-7). IDX: about 50 seconds. Selection is the alpha; gates are insurance. Gates today: coal OFF, nickel OFF.")
chips(s, [("1.11 vs 0.25", "Sharpe vs IHSG benchmark"), ("30.2%", "CAGR (IDR)"), ("+0.38", "IDX beta to gold (A6)"), ("OFF · OFF", "Design-variant gates, 9 Oct")], ORANGE)
tb(s, MX, 2.36, 6.9, 1.05,
   [("**Design.** Equal-weight Indonesian commodity baskets, weekly, 25 bps per side, IDR. Coal: ADRO, PTBA, ITMG, BYAN. Nickel: ANTM, INCO, MDKA, NCKL, MBMA. Gold miners: ANTM, BRMS, PSAB, MDKA, ARCI.", {})], size=14)
table(s, MX, 3.48, 6.9,
      [["", "Sharpe", "CAGR", "Max DD"],
       ["Coal + nickel (growth)", "1.11", "30.2%", "−41.8%"],
       ["IHSG benchmark", "0.25", "4.1%", "−38.4%"]],
      [2.9, 1.3, 1.3, 1.4], font=13.5, hcolor=ORANGE)
tb(s, MX, 4.52, 6.9, 1.6,
   [("**Findings:** (1) the alpha is **asset selection**: IHSG is bank-heavy and dilutes the commodity torque. (2) Trend gates are **insurance, not alpha** (coal drawdown −51% → −22%, no Sharpe added). (3) In the gated variant, a COT positioning brake lifted the gold-miner basket's Sharpe 0.81 → 0.86; the naive version failed (A3).", {})], size=13.5)
pic(s, os.path.join(IDX, "chart_idx_weekly.png"), 7.75, 2.36, 5.03,
    "Weekly books vs IHSG, 2016 to 2026. Returns in IDR; SGD investors earn less after FX. The book series is the ungated coal + nickel growth baskets; gates and the gold basket are design variants.")
pointer(s, "CAGR here is the annualised mean weekly return; geometric: IDX 30.3%, IHSG 2.8%.  ·  Exhibits: A6 · A9")
verdict(s, "**Selection is the alpha:** concentrated commodity torque the benchmark dilutes, with gates that insure rather than pretend to predict.", ORANGE)
pagenum(s, 6)

# ================= 6. DIP =================
s = new_slide("Sleeve 4 · Short-term reversal", "Dip buying, Nasdaq-100", GREEN)
speaker(s, "SPEAKER 3 · WIELLY", GREEN)
note(s, "Speaker 3 of 5: Wielly (last of your pair). Dip: about 50 seconds. No stop loss by design; the 200-day filter is the defence. Hand over to Michael.")
chips(s, [("1.07", "Sharpe, long run 2015–2026"), ("1.70", "Sharpe since 2024"), ("−1.1%", "2022: sat the bear out"), ("≈13%", "Average capital use")], GREEN)
tb(s, MX, 2.36, 6.9, 1.3,
   [("**Design.** Our rebuild of a published dip-buying rule set, verified against its reported results. Nasdaq-100 names above their 200-day average; a one-day drop over 3% at high ATR triggers a next-day limit buy 0.9×ATR below the close. Exits: a +0.5×ATR target, a close above the prior day's high, or day 11. Max 10 positions. No stop loss.", {})], size=14)
table(s, MX, 3.94, 6.9,
      [["Long run 2015–2026", "Sharpe 1.07 daily (1.50 weekly), CAGR 14.8%"],
       ["Trades / win rate / hold", "853 / 69.9% / 4.7 days"],
       ["Max drawdown", "−11.3%"],
       ["COVID crash, Feb–Mar 2020", "−3.7%"],
       ["Since 2024 (published test window)", "Sharpe 1.70, CAGR 39.1% (to 14 Oct 2026)"]],
      [3.1, 3.8], font=13.5, header_row=False, bold_rows=(1,), hcolor=GREEN)
pic(s, os.path.join(BASE, "chart_dip.png"), 7.75, 2.36, 5.03,
    "Equity curve and drawdown, 2015 to 2026, net of IBKR commissions. The flat stretches are the design working: no qualifying dips, no trades. Universe: today's constituents applied backwards (slide 10).")
pointer(s, "Lecture exhibits: factor attribution (A6) · tail risk, EVT (A7)")
verdict(s, "**A long-only sleeve that survives bears by absence:** the 200-day filter keeps it in cash when dips turn into downtrends. Its risk is the mirror image: a crash that starts at all-time highs.", GREEN)
pagenum(s, 7)

# ================= 7. PORTFOLIO =================
s = new_slide("Portfolio · Construction", "The whole falls less than every part", NAVY)
speaker(s, "SPEAKER 4 · MICHAEL")
note(s, "Speaker 4 of 5: Michael (slides 8-9). Portfolio: about 50 seconds. Book Sharpe 1.54 against sleeves 0.22 to 1.73 and the SPX at 0.84 with a -24.8% drawdown; correlations near zero, VT overlay kept out.")
chips(s, [("1.54", "Book Sharpe (common window)"), ("+14.1%", "2022: BTC −64%, equities −19%"), ("−8.1%", "Book max drawdown"), ("+0.07", "Book equity beta (A6)")], NAVY)
tb(s, MX, 2.36, 6.9, 0.8,
   [("Allocation, quarterly rebalanced: **E1 40%   Dip 20%   IDX 25%   Wheel 15%**. Sleeves are aggregated in native currencies (IDX in IDR, unhedged): a model composite, not a single-currency fund return.", {})], size=14)
table(s, MX, 3.2, 6.9,
      [["", "Sharpe*", "CAGR", "Vol", "Max DD"],
       ["E1", "0.22", "1.5%", "8.2%", "−8.4%"],
       ["DIP", "1.73", "20.8%", "11.3%", "−10.9%"],
       ["IDX", "1.04", "26.3%", "25.5%", "−35.3%"],
       ["Wheel", "0.65", "16.7%", "32.4%", "−43.4%"],
       ["BOOK", "1.54", "15.4%", "9.6%", "−8.1%"],
       ["SPX", "0.84", "12.7%", "15.7%", "−24.8%"]],
      [1.3, 1.4, 1.4, 1.3, 1.5], font=12, bold_rows=(5,), hcolor=NAVY)
tb(s, MX, 5.50, 6.9, 1.0,
   [("Stress record: 2022 +14.1%; Aug 2024 +0.4%; Apr 2025 +1.1%. The 12% vol-target overlay (Sharpe 1.45, DD −10.7%) levers the book up from 9.6% vol: CAGR 19.0%, deeper drawdown. Kept out.", {})], size=13.5)
pic(s, os.path.join(BASE, "chart_book.png"), 7.75, 2.36, 5.03,
    "Correlations: E1–DIP −0.09, E1–IDX −0.03, E1–Wheel −0.01, DIP–IDX −0.08, DIP–Wheel +0.16, IDX–Wheel +0.23. The grey dashed line is the S&P 500: +94.3% over the window with a −24.8% drawdown, against the book's +121.0% and −8.1%. The dip sleeve trades E1’s market and still correlates at −0.09: it deploys only ≈13% of its capital on average, with positions open on about half of days. Lecture exhibits: A5 · A7 · A6.")
pointer(s, "*Sharpe = annualised mean weekly return ÷ standard deviation, no risk-free rate deducted (all rows); net of T-bills (avg 3.4%): book 1.18, SPX 0.63.")
verdict(s, "**Diversification is still the strategy at book level:** Sharpe 1.54, max drawdown −8.1%, positive in all three stress windows; 2022 is IDX-driven (+98.7%), without which the other sleeves lose about 7%. SPX: Sharpe 0.84, drawdown −24.8%, −19.4% in 2022.", NAVY)
pagenum(s, 8)

# ================= 8. LIVE PLAN =================
s = new_slide("Live window · Operations", "Operating plan")
speaker(s, "SPEAKER 4 · MICHAEL")
note(s, "Speaker 4 of 5: Michael (last of your pair). Live plan: about 50 seconds. Discipline before performance; the window grades discipline, not luck. Hand over to Victoria.")
tb(s, MX, 1.42, 5.9, 0.6, [("**Discipline before performance.**", {"size": 16})], size=16)
bullets(s, MX, 2.0, 5.9, 3.6,
        ["Rules frozen 1 Oct; the freeze log at right records two changes made inside the window, with reasons.",
         "Weekly tear sheet per sleeve and for the book, produced at every Friday close in the window.",
         "Book brake: halve gross at −5% drawdown; stop and review at −8% (backtests exclude the brake). October crossed −5% in the week ending 16 Oct; test returns are unbraked."], size=14.5)
tb(s, 6.85, 1.42, 5.9, 4.6,
   [("**Reporting, in the brief's own metrics:** annual return, Sharpe, maximum drawdown and drawdown count, win rate, volatility, number of trades, average holding period, profit per trade.", {}),
    ("Reported per sleeve and for the book, with trade counts beside every ratio. A few weeks of live data can validate operations and discipline, not a premium, and we will not present it as if it does.", {}),
    ("**Freeze log:** frozen 1 Oct; two changes inside the window, both on pre-window evidence: (1) 15 Oct, Sleeve 4 CTA futures replaced by dip buying (verified 1.07 vs 0.34, A3); (2) 22 Oct, E1 costing rerun at full turnover (returns fell). **Brake, quantified:** at or below −5% in 10 weeks (5 episodes); −8% breached once (5 Jun 2026, −8.12%); return effect not simulated. The swap was no response to the live loss: the dip sleeve fell in October too.", {})], size=12.5)

tb(s, MX, 4.00, CW, 0.3, [("The operating rhythm", {"size": 13, "bold": True, "color": NAVY, "space_after": 0})])
_cad = [("DAILY", NAVY, WHITE, "E1 s-scores and dip triggers checked. Orders only from frozen rules; every fill logged."),
        ("WEEKLY", NAVY, WHITE, "IDX coal and nickel baskets rebalanced; gate status logged. The wheel rolls the next 25-delta BTC contracts."),
        ("QUARTERLY", NAVY, WHITE, "Book rebalanced back to 40 / 20 / 25 / 15. The weights are not negotiable in between."),
        ("FRIDAY CLOSE", NAVY, WHITE, "Tear sheet per sleeve and for the book, in the brief's metrics, trade counts beside every ratio.")]
for _i, (_lab, _col, _fill, _body) in enumerate(_cad):
    _x = MX + _i * 3.084
    _r = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(_x), Inches(4.32), Inches(2.977), Inches(1.60))
    try: _r.adjustments[0] = 0.07
    except Exception: pass
    _r.fill.solid(); _r.fill.fore_color.rgb = _fill; _r.line.color.rgb = RGBColor(0xD8, 0xD8, 0xD8); _r.line.width = Pt(0.75); _r.shadow.inherit = False
    tb(s, _x + 0.14, 4.46, 2.70, 0.3, [(_lab, {"size": 11.5, "bold": True, "color": _col, "space_after": 0})])
    tb(s, _x + 0.14, 4.78, 2.70, 1.05, [(_body, {"size": 10.5, "space_after": 0})], size=10.5)
tb(s, MX, 5.99, 4.6, 0.28, [("The brake, as a gauge", {"size": 13, "bold": True, "color": NAVY, "space_after": 0})])
tb(s, MX + 8.35, 6.01, 3.05, 0.26, [("\u22128.1% backtest max (unbraked) \u25bc", {"size": 10, "color": GRAY, "align": PP_ALIGN.CENTER, "space_after": 0})])
for _gx, _gw, _gf, _gl in [(MX, 6.115, RGBColor(0xD5, 0xE8, 0xD4), "0% to \u22125% \u00b7 full gross"),
                          (MX + 6.115, 3.669, RGBColor(0xFB, 0xE7, 0xCE), "\u22125% \u00b7 halve gross"),
                          (MX + 9.784, 2.446, RGBColor(0xF5, 0xD5, 0xD5), "\u22128% \u00b7 stop and review")]:
    _r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(_gx), Inches(6.25), Inches(_gw), Inches(0.30))
    _r.fill.solid(); _r.fill.fore_color.rgb = _gf; _r.line.color.rgb = WHITE; _r.line.width = Pt(1.0); _r.shadow.inherit = False
    tb(s, _gx + 0.1, 6.305, _gw - 0.2, 0.24, [(_gl, {"size": 10.5, "bold": True, "align": PP_ALIGN.CENTER, "space_after": 0})])
_mk = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(MX + CW * 0.81), Inches(6.16), Inches(0.028), Inches(0.40))
_mk.fill.solid(); _mk.fill.fore_color.rgb = NAVY; _mk.line.fill.background(); _mk.shadow.inherit = False
verdict(s, "**The window grades discipline, not luck:** frozen rules, Friday tear sheets, and a brake whose −5% line the book crossed in the week ending 16 Oct.", NAVY)
pagenum(s, 9)

# ================= 9. LIMITATIONS =================
s = new_slide("Honesty · Limitations", "Limitations, stated plainly")
speaker(s, "SPEAKER 5 · VICTORIA")
note(s, "Speaker 5 of 5: Victoria (slides 10-11). Limitations: about 50 seconds. State them plainly; do not apologise for them.")
bullets(s, MX, 1.42, 5.9, 2.5,
        ["**Wheel:** no option quotes were used. Premiums are Black–Scholes at one flat vol (Deribit's 30-day DVOL) on 1-week 25-delta options: no skew, no bid–ask, no term structure, less a 5% haircut. The $476K premium and the A8 vol gap are model outputs that likely flatter calm markets. Venue and assignment mechanics simplified.",
         "**E1:** the 34-name universe (9 SGX, 25 US) is today's large-cap list applied backwards; decay documented, not explained away; SGX borrow assumed at modelled cost.",
         "**Dip:** today's Nasdaq-100 applied backwards, so names that fell out after a decline are absent; a buy-the-dip rule is the design most flattered by this, so read its 1.73 Sharpe as an upper bound. Sizing is reconstructed. Idle cash earns nothing in the backtest."], size=12.5)
bullets(s, 6.85, 1.42, 5.9, 2.5,
        ["**IDX:** baskets are today's known winners (survivorship flatters levels; same-basket comparisons are insulated). The nickel basket held three names until Apr 2023 (NCKL and MBMA listed later), so 2022's +98.7% is a seven-stock result. IDR unhedged; coal and nickel are one cycle, not two.",
         "**Book:** the window starts Apr 2021 (BTC options history); no 2008/2020-style crash for the full book, and the live window is weeks long."], size=12.5)
tb(s, MX, 4.26, CW, 0.35, [("Where each sleeve is most exposed", {"size": 14, "bold": True, "color": NAVY, "space_after": 0})])
_sev_cols = ["Synthetic / estimated inputs", "Costs & frictions", "Edge decay", "Tail / regime break"]
_sev = [(NAVY, "E1 · Mean reversion", ["Med", "Med", "High", "Low"]),
        (PURPLE, "Wheel · BTC options", ["High", "Med", "Low", "High"]),
        (ORANGE, "IDX · Commodities", ["High", "Med", "Low", "Med"]),
        (GREEN, "Dip · NDX reversal", ["High", "Low", "Med", "High"])]
_sev_fill = {"High": RGBColor(0xC0, 0x39, 0x2B), "Med": RGBColor(0xE6, 0x9C, 0x2E), "Low": RGBColor(0xA9, 0xBC, 0x9B)}
for ci, clab in enumerate(_sev_cols):
    tb(s, 2.62 + ci * 2.42, 4.64, 2.3, 0.3, [(clab, {"size": 10.5, "color": GRAY, "align": PP_ALIGN.CENTER, "space_after": 0})])
for ri, (col, lab, sev) in enumerate(_sev):
    yy = 5.00 + ri * 0.40
    tb(s, MX, yy, 2.0, 0.3, [(lab, {"size": 11.5, "bold": True, "color": col, "space_after": 0})])
    for ci, sv in enumerate(sev):
        cx = 2.62 + ci * 2.42 + 1.03
        ov = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx), Inches(yy + 0.02), Inches(0.24), Inches(0.24))
        ov.fill.solid(); ov.fill.fore_color.rgb = _sev_fill[sv]; ov.line.fill.background(); ov.shadow.inherit = False
tb(s, MX, 6.54, CW, 0.3, [("Severity:  ", {"size": 10.5, "color": GRAY, "space_after": 0})])
for i, sv in enumerate(["High", "Med", "Low"]):
    ov = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(1.45 + i * 1.15), Inches(6.56), Inches(0.17), Inches(0.17))
    ov.fill.solid(); ov.fill.fore_color.rgb = _sev_fill[sv]; ov.line.fill.background(); ov.shadow.inherit = False
    tb(s, 1.68 + i * 1.15, 6.52, 0.9, 0.3, [(sv, {"size": 10.5, "color": GRAY, "space_after": 0})])
pagenum(s, 10)

# ================= 10. KILL CRITERIA =================
s = new_slide("Risk · Pre-commitment", "Kill criteria, pre-committed")
speaker(s, "SPEAKER 5 · VICTORIA")
note(s, "Speaker 5 of 5: Victoria (last of your pair). Kill criteria: about 50 seconds, then hand to the group close on slide 12.")
table(s, MX, 1.42, CW,
      [["Sleeve", "Kill rule"],
       ["E1", "Stop if live profit per trade is negative after 200 trades, or if the freeze log shows a parameter change."],
       ["Wheel", "Stop if cumulative assignment losses exceed the trailing two years of premium collected."],
       ["IDX", "Stop if the sleeve's live drawdown exceeds −42%, beyond its full-history backtest maximum (−41.8% since 2016; −35.3% on the common window)."],
       ["Dip", "Stop if sleeve drawdown exceeds −13%, beyond its backtest maximum of −11.3% (July 2026, long run and since 2024 alike): the design has no stop loss, so the drawdown limit is the stop."],
       ["Book", "The −8% brake is the kill criterion for the combination itself."]],
      [1.6, 10.63], font=14.5, text_table=True)
tb(s, MX, 4.30, CW, 0.35, [("The book brake, on one line", {"size": 14, "bold": True, "color": NAVY, "space_after": 0})])
for _x, _w, _fill, _lab in [(MX, 5.10, RGBColor(0xD5, 0xE8, 0xD4), "0% to −5%: full gross"),
                            (MX + 5.10, 3.06, RGBColor(0xFB, 0xE7, 0xCE), "−5%: halve gross"),
                            (MX + 8.16, 4.07, RGBColor(0xF5, 0xD5, 0xD5), "−8%: stop and review")]:
    _r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(_x), Inches(4.70), Inches(_w), Inches(0.44))
    _r.fill.solid(); _r.fill.fore_color.rgb = _fill; _r.line.color.rgb = WHITE; _r.line.width = Pt(1.0); _r.shadow.inherit = False
    _tf = _r.text_frame; _tf.word_wrap = True
    _tf.margin_left = _tf.margin_right = Inches(0.1)
    _p = _tf.paragraphs[0]; _p.alignment = PP_ALIGN.CENTER
    _rich(_p, _lab, 12, True, DARK)
tb(s, MX, 5.28, CW, 0.35, [("Backtest maximum drawdown without the brake: −8.1% (Apr 2021 to Oct 2026). The −8% stop sits just inside it, so a live repeat is cut at the stop, not after it.", {"size": 12, "color": GRAY})])
tb(s, MX, 5.85, CW, 1.1,
   [("Four premia, four market structures, one book: sized by damage, frozen in advance,", {"size": 17, "align": PP_ALIGN.CENTER, "space_after": 2}),
    ("and reported the way we would want to read it.", {"size": 17, "align": PP_ALIGN.CENTER})], size=17)
pagenum(s, 11)

# ================= 12. CONCLUSION =================
s = new_slide("Conclusion", "What we built, and what would change our minds")
note(s, "Group close: all five speakers, one takeaway each, about 40 seconds total. Andrew: the hook, answered. Edward: Sleeve 1. Wielly: Sleeves 3 and 4. Michael: the book in numbers. Victoria: the kill discipline. End on the verdict strip, then Q&A. Appendix A1-A9 is backup for Q&A, not presented.")
_takeaways = [
    ("1 · Four premia, no single dominant driver.",
     "Pairwise correlations −0.09 to +0.23; the book's equity beta is +0.07; but three sleeves are long risk (slide 3). Diversification here is measured in sample, over 288 weeks."),
    ("2 · The book falls less than its parts.",
     "Sharpe 1.54 vs 0.22 to 1.73 for the sleeves; maximum drawdown −8.1%, shallower than every sleeve; +14.1% in 2022. The dip sleeve rivals it, but deploys only ≈13% of capital on average (in cash on about half of days): it cannot be the whole book."),
    ("3 · Four designs dropped; the record is public.",
     "Three designs died on their own evidence, and a fourth, the working CTA futures sleeve, was still substituted when the dip strategy verified stronger (A3)."),
    ("4 · Sized by damage, frozen in advance.",
     "EVT tails set the weights' conscience: the wheel is smallest because its shortfall is largest (A7). The brake is pre-committed (slide 11)."),
]
for _i, (_h, _sup) in enumerate(_takeaways):
    _yy = 1.42 + _i * 1.05
    tb(s, MX, _yy, 6.9, 0.35, [(_h, {"size": 15, "bold": True, "color": NAVY, "space_after": 0})])
    tb(s, MX, _yy + 0.34, 6.9, 0.62, [(_sup, {"size": 12, "color": GRAY, "space_after": 0})], size=12)
tb(s, MX, 5.62, 6.9, 0.62,
   [("**What would change our minds:** a sleeve breaching its kill rule in the live window. We stop it, and we report that too.", {})], size=13)
pic(s, os.path.join(BASE, "chart_title.png"), 7.75, 1.50, 5.03)
tb(s, 7.75, 2.44, 5.03, 0.3, [("The book: Apr 2021 to Oct 2026, 100 → 221.", {"size": 11, "color": GRAY, "space_after": 0})])
tb(s, 7.75, 2.86, 5.03, 0.32, [("THE HOOK, ANSWERED", {"size": 11, "bold": True, "color": NAVY, "space_after": 0})])
_hookback = [(NAVY, "Crypto winter", "Wheel −43% vs −74%"),
             (GREEN, "Rate shock 2022", "Book +14.1%"),
             (NAVY, "Yen unwind", "Book +0.4%"),
             (PURPLE, "Tariff shock", "Book +1.1%"),
             (ORANGE, "Gold supercycle", "Indirect only: IDX gold β +0.38")]
for _i, (_col, _lab, _out) in enumerate(_hookback):
    _yy = 3.24 + _i * 0.56
    _r = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.75), Inches(_yy), Inches(5.03), Inches(0.46))
    _r.fill.solid(); _r.fill.fore_color.rgb = TINT[_col]; _r.line.fill.background(); _r.shadow.inherit = False
    tb(s, 7.95, _yy + 0.10, 2.3, 0.3, [(_lab, {"size": 12, "bold": True, "color": _col, "space_after": 0})])
    tb(s, 10.15, _yy + 0.10, 2.45, 0.3, [(_out, {"size": 12, "bold": True, "align": PP_ALIGN.RIGHT, "space_after": 0})])
verdict(s, "**The claim of this project is not that we found four edges.** It is that we built a book that can survive being wrong about one of them.", NAVY)
pagenum(s, 12)

# ================= A1 =================
s = new_slide("Appendix A1 · Specifications", "Frozen specification sheet")
table(s, MX, 1.42, CW,
      [["Sleeve", "Specification"],
       ["E1", "3 PCA factors, 60d window, s open ±1.5 / close ∓0.5, stop |s| > 3.5, OU half-life ≤ 30d; 34 names (9 SGX, 25 US); 10 bps on every change in position weight (entries, exits, daily rebalancing)."],
       ["Wheel", "Weekly Fri cycle; 25-delta 1-week options; puts when flat, calls when holding; full cash cover; T-bill on cash; 5% premium haircut; DVOL pricing (30d IV)."],
       ["IDX", "EW baskets, weekly; coal (ADRO, PTBA, ITMG, BYAN), nickel (ANTM, INCO, MDKA, NCKL, MBMA), gold (ANTM, BRMS, PSAB, MDKA, ARCI); sleeve gates: coal on BTU trend, nickel on own 40wk trend, gold on gold 40wk trend + COT Brake B; 25 bps/side; IDR. Book series: ungated coal + nickel growth baskets; gates and the gold basket are design variants (A3)."],
       ["Dip", "Nasdaq-100 (101 current constituents); close > 200d SMA; signal: 1-day drop > 3% and ATR(5)/close > 3%; next-day limit at signal close − 0.9×ATR(5); rank ATR(5)/close desc; max 10 positions; size = prior-close equity / 10 (reconstructed); exits: +0.5×ATR target, close > prior day high, or day 11; no stop; IBKR tiered commissions."],
       ["Book", "40/20/25/15, quarterly rebalance; brake at −5% (halve) and −8% (stop) book drawdown."]],
      [1.4, 10.83], font=12.5, text_table=True)
for _i, (_lab, _sub) in enumerate([("FROZEN", "Final specification, fixed Oct 2026; two changes logged (slide 9)"),
                                   ("LIVE WINDOW", "Oct 2026; final tear sheet at the 30 Oct close"),
                                   ("NO SILENT CHANGES", "Any change enters the freeze log, with reasons")]):
    _x = MX + _i * 4.20
    _r = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(_x), Inches(4.85), Inches(3.83), Inches(0.78))
    _r.fill.solid(); _r.fill.fore_color.rgb = TINT[NAVY]; _r.line.fill.background(); _r.shadow.inherit = False
    _tf = _r.text_frame; _tf.word_wrap = True
    _tf.margin_left = _tf.margin_right = Inches(0.12); _tf.margin_top = Inches(0.07)
    _p = _tf.paragraphs[0]; _p.alignment = PP_ALIGN.CENTER; _rich(_p, _lab, 12.5, True, NAVY)
    _p2 = _tf.add_paragraph(); _p2.alignment = PP_ALIGN.CENTER; _rich(_p2, _sub, 9.5, False, GRAY)
    if _i < 2:
        tb(s, _x + 3.83, 5.02, 0.37, 0.4, [("→", {"size": 16, "bold": True, "color": NAVY, "align": PP_ALIGN.CENTER, "space_after": 0})])
pagenum(s, 13)

# ================= A2 =================
s = new_slide("Appendix A2 · Robustness", "Robustness (common window, weekly)")
table(s, MX, 1.5, 5.9,
      [["", "Sharpe", "Newey–West", "Drop-best"],
       ["E1", "0.22", "0.30", "0.14"],
       ["DIP", "1.73", "1.81", "1.73"],
       ["IDX", "1.04", "0.98", "0.98"],
       ["Wheel", "0.65", "0.69", "0.59"],
       ["BOOK", "1.54", "1.75", "1.48"],
       ["SPX", "0.84", "0.98", "0.78"]],
      [1.0, 1.05, 1.85, 2.0], font=13.5, bold_rows=(5,))
pic(s, os.path.join(BASE, "chart_a2.png"), 6.95, 1.5, 5.83)
tb(s, MX, 4.55, CW, 2.2,
   [("No sleeve, and not the book, depends on its single best week. Newey–West Sharpe is above raw Sharpe for every row except IDX, consistent with mild negative weekly autocorrelation in the others.", {}),
    ("Parameter sensitivity: E1 Sharpe peaks at the frozen 3 factors (0.43/0.67/0.48/0.26 for 2/3/4/5 factors), so the frozen spec is the in-sample best on that axis. s-entry 1.75 gives 0.99 against the frozen 1.5 (drawdown −14.8% vs −9.1%); it was measured and deliberately not taken.", {})], size=14)
pagenum(s, 14)

# ================= A3 =================
s = new_slide("Appendix A3 · Rejection log", "What we tested and killed")
table(s, MX, 1.42, CW,
      [["Candidate", "Evidence"],
       ["CTA futures trend (built, then substituted)", "Verified as designed (Sharpe 0.34; 2022 +17.4%), then replaced as Sleeve 4 by the dip strategy on the group's direction, for its stronger verified record (1.07 long run, 1.70 since 2024). Cost: the book no longer shorts bonds or the dollar."],
       ["BTC regime wheel", "Sharpe 0.48 vs 0.65 always-on; 2022 −60.7% vs −32.0%. Replaced by the plain wheel."],
       ["COT Brake A (gold)", "Exit at 90th positioning percentile: Sharpe 0.54 vs 0.56 (gold), 0.67 vs 0.81 (miners); out of the market 32% of weeks. Brake B (percentile + momentum) adopted instead."],
       ["CPO basket (IDX)", "Sharpe 0.24 to 0.40 in every variant, drawdowns −53% to −68%. Dropped; gold miners took the third-basket slot."],
       ["Pairs trading", "Set aside by the group as a well-trodden design before testing; the PCA engine generalises it."]],
      [2.7, 9.53], font=13, text_table=True)
chips(s, [("5", "Entries in this log"), ("4", "Killed or substituted"), ("1", "Set aside untested (pairs)"), ("0", "Killed designs still in the book")], NAVY, y=5.42)
pagenum(s, 15)

# ================= A4 =================
s = new_slide("Appendix A4 · Data & protocols", "Data and anti-look-ahead protocols")
tb(s, MX, 1.42, 5.9, 4.9,
   [("**Sources.** Prices: Yahoo Finance (equities, futures, BTC). Options: Deribit DVOL volatility index (daily, Mar 2021 onwards), options priced by Black–Scholes at 25 delta. Positioning: CFTC Commitments of Traders, COMEX gold, legacy noncommercial net as % of open interest. Rates: 13-week T-bill (^IRX).", {})], size=14.5)
tb(s, 6.85, 1.42, 5.9, 4.9,
   [("**Anti-look-ahead protocols.** COT: a Tuesday report is used only at Friday evaluations at least 7 days later (it is published Friday, after the IDX close). All signals are computed on data dated on or before the decision close; execution is at the next session or the same close as stated per sleeve. Point-in-time caveat: IDX baskets and the E1 universe are today's constituents; the limitation is disclosed on slide 10, not buried here.", {})], size=14.5)
for _i, (_lab, _sub) in enumerate([("SOURCES", "Yahoo prices · Deribit DVOL · CFTC COT · T-bill rate"),
                                   ("PROTOCOLS", "COT lagged 7+ days · signals on or before the decision close · costs per sleeve"),
                                   ("OUTPUT", "Friday tear sheet: the brief's nine metrics, per sleeve and for the book")]):
    _x = MX + _i * 4.20
    _r = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(_x), Inches(4.85), Inches(3.83), Inches(0.95))
    _r.fill.solid(); _r.fill.fore_color.rgb = TINT[NAVY]; _r.line.fill.background(); _r.shadow.inherit = False
    _tf = _r.text_frame; _tf.word_wrap = True
    _tf.margin_left = _tf.margin_right = Inches(0.12); _tf.margin_top = Inches(0.07)
    _p = _tf.paragraphs[0]; _p.alignment = PP_ALIGN.CENTER; _rich(_p, _lab, 12.5, True, NAVY)
    _p2 = _tf.add_paragraph(); _p2.alignment = PP_ALIGN.CENTER; _rich(_p2, _sub, 9.5, False, GRAY)
    if _i < 2:
        tb(s, _x + 3.83, 5.15, 0.37, 0.4, [("→", {"size": 16, "bold": True, "color": NAVY, "align": PP_ALIGN.CENTER, "space_after": 0})])
pagenum(s, 16)

# ================= A5 =================
s = new_slide("Appendix A5 · Allocation", "Why these weights (sensitivity)")
tb(s, MX, 1.42, CW, 0.9,
   [("Weights were set by design and frozen for the live window; the study validates the choice rather than fitting it. Anchor: risk parity on measured sleeve vols (E1 8.2%, Dip 11.3%, IDX 25.5%, Wheel 32.4%) gives ≈44/31/14/11, adjusted for E1's alpha decay, the dip sleeve’s long-only equity exposure, and the wheel’s short-vol tail.", {})], size=14)
for _i, (_lab, _pct, _col) in enumerate([("E1", 40, NAVY), ("IDX", 25, ORANGE), ("Dip", 20, GREEN), ("Wheel", 15, PURPLE)]):
    _yy = 2.44 + _i * 0.40
    tb(s, MX, _yy + 0.02, 1.3, 0.3, [(_lab, {"size": 12.5, "bold": True, "color": _col, "space_after": 0})])
    _r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(2.0), Inches(_yy), Inches(_pct / 40 * 6.2), Inches(0.30))
    _r.fill.solid(); _r.fill.fore_color.rgb = _col; _r.line.fill.background(); _r.shadow.inherit = False
    tb(s, 2.0 + _pct / 40 * 6.2 + 0.15, _yy + 0.02, 0.8, 0.3, [(f"{_pct}%", {"size": 12.5, "bold": True, "space_after": 0})])
table(s, 1.9, 4.18, 9.5,
      [["Weights", "Sharpe", "CAGR", "Vol", "Max DD", "2022"],
       ["Chosen 40/20/25/15", "1.54", "15.4%", "9.6%", "−8.1%", "+14.1%"],
       ["Risk parity ≈44/31/14/11", "1.78", "13.8%", "7.4%", "−4.9%", "+6.2%"],
       ["Equal weight 25 each", "1.48", "18.4%", "12.0%", "−10.5%", "+10.8%"],
       ["E1-heavy 50/15/20/15", "1.42", "13.0%", "8.9%", "−7.4%", "+9.8%"]],
      [2.9, 1.3, 1.3, 1.2, 1.4, 1.4], font=14, bold_rows=(1,))
tb(s, MX, 6.02, CW, 1.0,
   [("The optimum is no longer flat: parity reaches 1.78 by loading the low-volatility dip sleeve to 31%. The chosen weights accept 1.54 in exchange for the best crisis year (+14.1% in 2022 vs parity’s +6.2%) and less riding on two equity sleeves. Either is defensible; the trade-off is disclosed, not hidden.", {})], size=13)
pagenum(s, 17)

# ================= A6 FACTOR ATTRIBUTION =================
s = new_slide("Appendix A6 · Lecture: factor models", "Factor attribution: the premia are different, measurably")
table(s, 1.65, 1.55, 10.0,
      [["Weekly return betas", "SPX", "BTC", "Gold", "Coal", "R²"],
       ["E1", "+0.03", "−0.00", "−0.02", "−0.01", "0.01"],
       ["DIP", "+0.16", "+0.00", "+0.08", "+0.00", "0.08"],
       ["IDX", "+0.03", "+0.08", "+0.38", "+0.07", "0.17"],
       ["Wheel", "+0.16", "+0.47", "−0.02", "+0.02", "0.66"],
       ["BOOK", "+0.07", "+0.09", "+0.10", "+0.02", "0.41"]],
      [2.8, 1.4, 1.4, 1.4, 1.4, 1.6], font=15, bold_rows=(5,),
      bold_cells={(2, 1), (3, 2), (3, 3), (3, 4), (4, 2), (5, 2), (5, 3), (5, 4)},
      cell_fills={(1, 1): RGBColor(0xE4, 0xE9, 0xF4),
                  (2, 1): RGBColor(0xC7, 0xD3, 0xEA), (2, 3): RGBColor(0xE4, 0xE9, 0xF4),
                  (3, 1): RGBColor(0xE4, 0xE9, 0xF4), (3, 2): RGBColor(0xE4, 0xE9, 0xF4), (3, 3): RGBColor(0x9D, 0xB3, 0xD9), (3, 4): RGBColor(0xE4, 0xE9, 0xF4),
                  (4, 1): RGBColor(0xC7, 0xD3, 0xEA), (4, 2): RGBColor(0x9D, 0xB3, 0xD9),
                  (5, 1): RGBColor(0xE4, 0xE9, 0xF4), (5, 2): RGBColor(0xE4, 0xE9, 0xF4), (5, 3): RGBColor(0xC7, 0xD3, 0xEA)})
tb(s, MX, 4.15, CW, 1.9,
   [("OLS of each sleeve's weekly returns on S&P 500, BTC, gold and coal futures, common window (Apr 2021 to Oct 2026, n = 288). **Bold: |t| > 2.6.** Cell shading deepens with |β|.", {"size": 12.5, "color": GRAY}),
    ("**E1 loads on nothing** (R² = 0.01): near-zero measured factor exposure, not neutrality by claim. The **book's equity beta is +0.07**: about +0.03 from the dip sleeve (+0.16 at a 20% weight), +0.02 from the wheel (+0.16 at 15%), the rest from E1 and IDX. The book's largest betas are to gold (+0.10) and BTC (+0.09). IDX loads on gold (+0.38) more than coal (+0.07): its nickel names (ANTM and MDKA are also gold miners) dominated the window.", {})], size=14)
pagenum(s, 18)

# ================= A7 VAR / ES =================
s = new_slide("Appendix A7 · Lecture: risk & EVT (L7)", "Tail risk: one-week 99% VaR and Expected Shortfall")
table(s, 1.65, 1.55, 10.0,
      [["Losses, % of sleeve capital", "VaR (hist)", "ES (hist)", "VaR (EVT)", "ES (EVT)", "Tail ξ"],
       ["E1 (40%)", "2.8", "3.8", "3.1", "3.8", "−0.03"],
       ["DIP (20%)", "4.1", "4.6", "3.6", "4.4", "−0.20"],
       ["IDX (25%)", "7.6", "8.3", "7.7", "8.2", "−0.53"],
       ["Wheel (15%)", "15.9", "20.8", "15.9", "22.0", "+0.12"],
       ["BOOK", "4.5", "5.1", "3.5", "5.2", "+0.26"]],
      [3.4, 1.3, 1.3, 1.3, 1.3, 1.4], font=15, bold_rows=(5,))
tb(s, MX, 3.68, 7.3, 0.3,
   [("Historical quantiles on weekly returns plus an EVT peaks-over-threshold fit (GPD, 90% threshold, ≈29 exceedances; ξ is indicative, not precise), common window.", {"size": 11.5, "color": GRAY})], size=11.5)
pic(s, os.path.join(BASE, "chart_a7.png"), MX, 4.14, 7.3)
tb(s, 8.15, 3.98, 4.63, 2.5,
   [("The **wheel owns the tail**, a 1-week ES of about 21 to 22% of its own capital, and is the smallest sleeve for exactly that reason.", {"space_after": 9}),
    ("The EVT fit and history now agree on the book (ES ≈ 5.1 to 5.2%): the dip sleeve’s truncated tail (ξ −0.20; its exits cap the holding period, not the loss) thins the book’s tail versus the earlier CTA-futures Sleeve 4 (A3). The −8% brake still sits beyond one very bad week.", {})], size=13.5)
pagenum(s, 19)

# ================= A8 GARCH =================
s = new_slide("Appendix A8 · Lecture: ARCH / GARCH", "The wheel's premium, measured: implied vs forecast vs realised vol")
pic(s, os.path.join(BASE, "chart_garch.png"), MX, 1.5, 8.3)
tb(s, 9.05, 1.55, 3.73, 3.6,
   [("**GARCH(1,1) on BTC daily returns**, fitted Jan 2015 to Dec 2022: α 0.055, β 0.945, persistence ≈ 1.00. Volatility shocks barely decay, which is why a premium seller respects vol regimes.", {"space_after": 9}),
    ("**Out-of-sample, Jan 2023 to Sep 2026 (means):** DVOL implied 49.7%, GARCH forecast 49.7%, vol that subsequently arrived 44.4%.", {"space_after": 9}),
    ("Implied exceeded subsequent realised vol on **69% of days**.", {})], size=13)
tb(s, MX, 5.35, CW, 1.1,
   [("Implied and GARCH agree as forecasts; both sit above what volatility delivered. The ≈5-vol-point average gap is the implied-minus-realised vol spread — the premium, quoted in vol points. The wheel sells that gap and nets assignment losses against it in the backtest (Sleeve 2); the chart's 30-day IV input flatters calm-market levels, as disclosed.", {})], size=13.5)
pagenum(s, 20)

# ================= A9 COINTEGRATION =================
s = new_slide("Appendix A9 · Lecture: cointegration", "Cointegration exhibits: beta without an anchor")
table(s, MX, 1.5, 7.3,
      [["Pair", "β", "EG p", "Spread HL"],
       ["ADRO vs coal (BTU)", "0.39", "0.95", ">500d"],
       ["PTBA vs coal (BTU)", "0.24", "0.83", "312d"],
       ["ITMG vs coal (BTU)", "0.49", "0.79", "325d"],
       ["BYAN vs coal (BTU)", "0.55", "0.73", ">500d"],
       ["ANTM vs gold", "1.40", "0.19", "153d"],
       ["MDKA vs gold", "1.47", "0.89", "497d"]],
      [3.4, 0.9, 1.1, 1.9], font=13.5)
tb(s, MX, 3.95, 7.3, 0.55,
   [("Daily log prices, Jan 2016 to Oct 2026. β from OLS of log miner on log commodity; EG p is the Engle–Granger cointegration p-value; HL is the spread's AR(1) half-life.", {"size": 11, "color": GRAY})], size=11)
pic(s, os.path.join(BASE, "chart_a9.png"), MX, 4.58, 7.3)
tb(s, 8.1, 1.5, 4.68, 3.9,
   [("**Engle–Granger verdict: no pair cointegrates** (all p ≥ 0.19; spread half-lives of 5 months to effectively never). The miners carry commodity *return beta* without a price-level anchor, so Sleeve 3's design research used trend gates, not spreads against the commodity.", {"space_after": 9}),
    ("**Contrast, Sleeve 1:** ADF rejects a unit root for **all 34** factor-residual series; residual half-lives cluster at 5 to 6 days, inside the engine's 30-day admission filter. That is what a tradeable spread looks like.", {"space_after": 9}),
    ("**Sleeve 4’s filter is the same logic used defensively:** dip entries require a close above the 200-day average, so in 2022 the sleeve returned −1.1% while sitting mostly in cash (average capital use 10%). An uptrend filter is a crude but effective regime gate.", {})], size=13)
pagenum(s, 21)

prs.core_properties.title = "The Quadrant Strategy — Four Market Structures · QF5210 Group Project"
prs.core_properties.author = "QF5210 Group"
out = os.path.join(BASE, "QF5210_Presentation_Slides.pptx")
prs.save(out)
print("saved", out)
