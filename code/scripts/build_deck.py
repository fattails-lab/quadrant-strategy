"""Print assets/deck/deck.html to the two deck PDFs with headless Chrome.

    python scripts/build_deck.py

Writes assets/QF5210_Presentation_Slides.pdf (slides 1-22) and
assets/QF5210_Presentation_Slides_with_MC.pdf (slides 1-22, with the Monte Carlo appendix A10).
Slide 23's numbers are typed into deck.html; after rerunning the Monte Carlo, update them from
outputs/montecarlo/wheel_forward_summary.csv and assets/deck/img/mc_fan.png is refreshed by
scripts/build_mc_slide.py.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf
from pypdf import PdfReader, PdfWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from book import config  # noqa: E402

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DECK_HTML = config.ROOT / "assets" / "deck" / "deck.html"
MAIN_PDF = config.ROOT / "assets" / "QF5210_Presentation_Slides.pdf"
MC_PDF = config.ROOT / "assets" / "QF5210_Presentation_Slides_with_MC.pdf"
MAIN_SLIDES = 21


def print_html(out):
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=15000", f"--print-to-pdf={out}", DECK_HTML.as_uri()],
                   check=True, capture_output=True)


def write_pages(reader, n, out):
    writer = PdfWriter()
    for page in reader.pages[:n]:
        writer.add_page(page)
    with open(out, "wb") as f:
        writer.write(f)
    # no compress(): re-encoding to JPEG flattens the card shadows into grey boxes
    print(f"Wrote {out.relative_to(config.ROOT)} ({n} pages, {out.stat().st_size / 1e6:.1f} MB)")


def compress(path, quality=85):
    """Re-encode the chart images as JPEG and drop unused objects (about 6 MB -> 1.4 MB)."""
    doc = pymupdf.open(path)
    doc.rewrite_images(dpi_threshold=None, dpi_target=0, quality=quality, lossy=True, lossless=True,
                       bitonal=False, color=True, gray=True, set_to_gray=False)
    tmp = path.with_suffix(".tmp.pdf")
    doc.save(tmp, garbage=4, deflate=True)
    doc.close()
    tmp.replace(path)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        full = Path(tmp) / "deck.pdf"
        print_html(full)
        reader = PdfReader(full)
        write_pages(reader, MAIN_SLIDES, MAIN_PDF)
        write_pages(reader, len(reader.pages), MC_PDF)
