"""Read and validate a portfolio file.

Expected columns (case-insensitive, any order, extra columns ignored):
trade_id, trade_date (YYYY-MM-DD), currency_pair (EURUSD, EUR/USD, eur-usd),
side (BUY/SELL), notional (positive, in the base currency; "1,000,000" and
"1_000_000" accepted), entry_price (quote per base). Optional: book.

Two stages:
  load_portfolio()      format checks that need no market data.
  check_against_market() rejects future-dated trades and entry prices that are
                         clearly wrong (e.g. an inverted USDJPY quote).
"""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["trade_id", "trade_date", "currency_pair", "side", "notional", "entry_price"]
SAMPLE = Path(__file__).resolve().parents[1] / "data" / "portfolio.csv"
COLUMNS = REQUIRED + ["book", "base", "quote", "N"]

# Entry price vs the market close on the trade date.
OFF_MARKET_REJECT = 0.25   # > 25% away: almost certainly inverted or mistyped
OFF_MARKET_WARN = 0.03     # > 3% away: accepted, but flagged as off-market


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def _read_bytes(src) -> bytes:
    if isinstance(src, (str, Path)):
        return Path(src).read_bytes()
    data = src.read()  # file-like: Streamlit UploadedFile, BytesIO, StringIO
    return data.encode() if isinstance(data, str) else data


def _decode(raw: bytes) -> tuple[str | None, str | None]:
    """Return (text, None) or (None, reason the file is not a text CSV)."""
    if raw[:4] == b"PK\x03\x04":
        return None, "This looks like an Excel workbook (.xlsx), not a CSV. In Excel, use File → Save As → CSV."
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16"), None
    if b"\x00" in raw[:4096]:
        return None, "This is a binary file, not a text CSV."
    try:
        return raw.decode("utf-8-sig"), None  # utf-8-sig drops the BOM Excel adds to CSV exports
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace"), None  # older Windows/Excel exports


def _delimiter(header: str) -> str:
    """Comma by default; semicolon or tab when the header clearly uses one (European Excel)."""
    counts = {d: header.count(d) for d in (",", ";", "\t")}
    best = max(counts, key=counts.get)
    return best if counts[best] > counts[","] else ","


def load_portfolio(src=SAMPLE, currencies: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Return (clean trades, error messages). Never raises on bad input; bad rows are dropped."""
    text, reason = _decode(_read_bytes(src))
    if reason:
        return _empty(), [reason]
    if not text.strip():
        return _empty(), ["The file is empty."]
    sep = _delimiter(text.lstrip().splitlines()[0])
    try:
        # The csv module (not pandas) so each row keeps its real line number and a
        # row with extra values is reported instead of silently truncated.
        reader = csv.reader(io.StringIO(text), delimiter=sep)
        lines = [(reader.line_num, row) for row in reader if any(c.strip() for c in row)]
    except csv.Error as exc:
        return _empty(), [f"The file is not a well-formed CSV: {exc}."]
    (_, header), body = lines[0], lines[1:]
    header = [h.strip().lower() for h in header]
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        found = ", ".join(h for h in header[:8] if h) or "none"
        hint = " If this is a semicolon- or tab-separated export, check the header row." if len(header) == 1 else ""
        return _empty(), [f"Missing column(s): {', '.join(missing)}. Columns found: {found}.{hint}"]

    errors, rows, line_nos = [], [], []
    for n, row in body:
        if len(row) > len(header) and any(c.strip() for c in row[len(header):]):
            errors.append(f"Row {n}: has {len(row)} values but the header has {len(header)} columns.")
            continue
        rows.append((row + [""] * len(header))[:len(header)])
        line_nos.append(n)
    if not rows:
        return _empty(), errors
    df = pd.DataFrame(rows, columns=header)
    df = df.loc[:, ~df.columns.duplicated()]  # a repeated column name: the first one wins

    known = {"USD", *(currencies or [])}
    df = df.apply(lambda s: s.str.strip())
    df["currency_pair"] = df.currency_pair.str.upper().str.replace(r"[^A-Z]", "", regex=True)
    df["side"] = df.side.str.upper()
    df["notional"] = pd.to_numeric(df.notional.str.replace(r"[,_]", "", regex=True), errors="coerce")
    # Semicolon-separated files come from locales that write decimals as 1,10.
    price = df.entry_price.str.replace(",", ".", regex=False) if sep == ";" else df.entry_price
    df["entry_price"] = pd.to_numeric(price, errors="coerce")
    dates = pd.to_datetime(df.trade_date, format="%Y-%m-%d", errors="coerce")
    df["base"], df["quote"] = df.currency_pair.str[:3], df.currency_pair.str[3:]
    if "book" not in df.columns:
        df["book"] = ""
    df["book"] = df.book.replace("", "Unassigned")

    keep, seen = [], set()
    for i, r in df.iterrows():
        label = f"Row {line_nos[i]} ({r.trade_id or 'no id'})"  # line number in the file
        if not r.trade_id:
            problem = "trade_id is blank"
        elif r.trade_id in seen:
            # Ids index the exposure matrix; a duplicate would merge two trades' risk.
            problem = f"trade_id {r.trade_id} is used more than once (first occurrence kept)"
        elif len(r.currency_pair) != 6 or r.base not in known or r.quote not in known or r.base == r.quote:
            problem = f"'{r.currency_pair or 'blank'}' is not a supported pair"
        elif r.side not in ("BUY", "SELL"):
            problem = "side must be BUY or SELL"
        elif not (r.notional > 0 and math.isfinite(r.notional)):
            problem = "notional must be a positive number (the side gives the direction)"
        elif not (r.entry_price > 0 and math.isfinite(r.entry_price)):
            problem = "entry_price must be a positive number"
        elif pd.isna(dates[i]):
            problem = "trade_date must be a real date in YYYY-MM-DD format"
        else:
            problem = None
        if problem:
            errors.append(f"{label}: {problem}.")
        else:
            keep.append(i)
            seen.add(r.trade_id)
    errors.sort(key=lambda e: int(e.split()[1].rstrip(":")))
    out = df.loc[keep].reset_index(drop=True)
    out["N"] = np.where(out.side == "BUY", 1.0, -1.0) * out.notional  # signed base notional
    return out[COLUMNS], errors


def check_against_market(book: pd.DataFrame, m) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (accepted trades, errors, warnings) after checks that need market data.

    m is a fxrisk.market.MarketData. The entry price is compared with the close
    on the trade date, or the nearest earlier close. Trades before the history
    starts are compared with the earliest close, using the reject threshold only.
    """
    if book.empty:
        return book, [], []
    today = m.live_time.strftime("%Y-%m-%d")
    idx = m.hist.index
    errors, warnings, keep = [], [], []
    for i, r in book.iterrows():
        label = f"{r.trade_id} ({r.currency_pair})"
        if r.trade_date > today:
            errors.append(f"{label}: trade_date {r.trade_date} is after today's prices ({today}).")
            continue
        j = idx.searchsorted(r.trade_date, side="right") - 1
        before_history = j < 0
        u = m.rates_at(max(j, 0)) if r.trade_date <= idx[-1] else m.live
        ref = m.spot(r.currency_pair, u)
        gap = r.entry_price / ref - 1
        if abs(gap) > OFF_MARKET_REJECT:
            hint = " The price may be inverted." if abs(r.entry_price * ref - 1) < OFF_MARKET_REJECT else ""
            errors.append(f"{label}: entry_price {r.entry_price:g} is {gap:+.0%} from the market "
                          f"({ref:.5g}) on {r.trade_date}.{hint}")
            continue
        if abs(gap) > OFF_MARKET_WARN and not before_history:
            warnings.append(f"{label}: entry_price {r.entry_price:g} is {gap:+.1%} from the {r.trade_date} close "
                            f"({ref:.5g}). Check it was booked correctly.")
        keep.append(i)
    return book.loc[keep].reset_index(drop=True), errors, warnings
