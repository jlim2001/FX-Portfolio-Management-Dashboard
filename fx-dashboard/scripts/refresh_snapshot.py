"""Refresh the bundled offline snapshot (data/market_snapshot.json) from the live sources.

    python scripts/refresh_snapshot.py [--source auto|yahoo|ecb] [--years 3]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fxrisk.market import load_market, save_snapshot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="auto", choices=["auto", "yahoo", "ecb"])
ap.add_argument("--years", type=int, default=3)
a = ap.parse_args()
m, warnings = load_market(a.source, a.years)
for w in warnings:
    print("warning:", w)
if m.source.startswith("Bundled"):
    sys.exit("No live source available; snapshot left unchanged.")
save_snapshot(m)
print(f"Saved {len(m.hist)} closes {m.hist.index[0]} → {m.hist.index[-1]}, live {m.live_time} from {m.source}")
