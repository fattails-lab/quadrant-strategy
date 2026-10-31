"""Run one sleeve or all of them.

    python scripts/run_sleeves.py            # all
    python scripts/run_sleeves.py wheel dip  # just these
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from book.sleeves import dip, e1_pca, fx_idr, wheel  # noqa: E402

SLEEVES = {"e1": e1_pca.run, "wheel": wheel.run, "dip": dip.run, "fx": fx_idr.run}

if __name__ == "__main__":
    for name in sys.argv[1:] or SLEEVES:
        print(f"\n===== {name} =====")
        SLEEVES[name]()
