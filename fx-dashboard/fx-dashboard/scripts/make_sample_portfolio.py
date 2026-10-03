"""One-off generator for data/portfolio.csv (the sample "existing" book).

Entry prices are the market close on each trade date with a small random
execution offset (+/- up to 12bp), rounded to the pair's quoting precision.
Trades dated today are struck against the live quote. Run once; the CSV is
then treated as the system of record and read by the dashboard.
"""
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
mkt = json.loads((ROOT / "data/market_snapshot.json").read_text())
dates = mkt["dates"]

def usd_per(ccy, idx=None):
    if ccy == "USD":
        return 1.0
    return mkt["live"][ccy] if idx is None else mkt["usd_per"][ccy][idx]

def decimals(pair):
    if "JPY" in pair:
        return 3
    if any(c in pair for c in ("MXN", "ZAR", "NOK", "SEK")):
        return 4
    return 5

#        id      trade date    pair      side   notional (base ccy)  book
TRADES = [
    ("T001", "2026-03-16", "EURUSD", "BUY", 25_000_000, "Macro"),
    ("T002", "2026-04-21", "USDJPY", "SELL", 20_000_000, "Macro"),
    ("T003", "2026-05-11", "GBPUSD", "SELL", 15_000_000, "Relative Value"),
    ("T004", "2026-06-02", "AUDUSD", "BUY", 20_000_000, "Commodity FX"),
    ("T005", "2026-06-18", "USDCAD", "BUY", 18_000_000, "Commodity FX"),
    ("T006", "2026-07-07", "EURGBP", "BUY", 12_000_000, "Relative Value"),
    ("T007", "2026-07-22", "USDMXN", "SELL", 10_000_000, "EM Carry"),
    ("T008", "2026-08-12", "AUDNZD", "BUY", 15_000_000, "Relative Value"),
    ("T009", "2026-08-27", "USDCHF", "BUY", 12_000_000, "Macro"),
    ("T010", "2026-09-03", "EURUSD", "BUY", 10_000_000, "Macro"),
    ("T011", "2026-09-09", "EURNOK", "SELL", 8_000_000, "Relative Value"),
    ("T012", "2026-09-17", "USDZAR", "SELL", 6_000_000, "EM Carry"),
    ("T013", "2026-09-24", "USDSEK", "BUY", 10_000_000, "Relative Value"),
    ("T014", "2026-09-29", "EURUSD", "SELL", 5_000_000, "Macro"),
    ("T015", "2026-10-01", "EURJPY", "BUY", 10_000_000, "Macro"),
    ("T016", "2026-10-02", "USDSGD", "SELL", 15_000_000, "EM Carry"),
]

random.seed(7)
rows = []
for tid, d, pair, side, notional, book in TRADES:
    base, quote = pair[:3], pair[3:]
    idx = dates.index(d) if d in dates else None  # None -> today, use live
    px = usd_per(base, idx) / usd_per(quote, idx)
    px *= 1 + random.uniform(-0.0012, 0.0012)
    rows.append([tid, d, pair, side, notional, f"{px:.{decimals(pair)}f}", book])

out = ROOT / "data/portfolio.csv"
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["trade_id", "trade_date", "currency_pair", "side", "notional", "entry_price", "book"])
    w.writerows(rows)
print(out.read_text())
