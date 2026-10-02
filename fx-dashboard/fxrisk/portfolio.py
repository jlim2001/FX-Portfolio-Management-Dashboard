"""Read and validate a portfolio file.

Expected columns (case-insensitive): trade_id, trade_date (YYYY-MM-DD),
currency_pair (e.g. EURUSD or EUR/USD), side (BUY/SELL), notional (in the
base currency, positive), entry_price (quote per base). Optional: book.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["trade_id", "trade_date", "currency_pair", "side", "notional", "entry_price"]
SAMPLE = Path(__file__).resolve().parents[1] / "data" / "portfolio.csv"


def load_portfolio(src=SAMPLE, currencies: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Return (clean trades, row-level error messages). Invalid rows are dropped, not fatal."""
    df = pd.read_csv(src, dtype=str).rename(columns=str.lower)
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return pd.DataFrame(), [f"Missing column(s): {', '.join(missing)}."]
    known = {"USD", *(currencies or [])}
    df = df.apply(lambda s: s.str.strip())
    df["currency_pair"] = df.currency_pair.str.upper().str.replace(r"[^A-Z]", "", regex=True)
    df["side"] = df.side.str.upper()
    df["notional"] = pd.to_numeric(df.notional.str.replace(r"[,_]", "", regex=True), errors="coerce")
    df["entry_price"] = pd.to_numeric(df.entry_price, errors="coerce")
    df["base"], df["quote"] = df.currency_pair.str[:3], df.currency_pair.str[3:]
    if "book" not in df.columns:
        df["book"] = "Unassigned"
    df["book"] = df.book.fillna("Unassigned")

    errors, keep = [], []
    for i, r in df.iterrows():
        row = i + 2  # header is row 1
        problem = None
        if len(r.currency_pair) != 6 or r.base not in known or r.quote not in known or r.base == r.quote:
            problem = f"'{r.currency_pair}' is not a supported pair"
        elif r.side not in ("BUY", "SELL"):
            problem = "side must be BUY or SELL"
        elif not r.notional > 0:
            problem = "notional must be a positive number"
        elif not r.entry_price > 0:
            problem = "entry_price must be a positive number"
        elif pd.isna(pd.to_datetime(r.trade_date, format="%Y-%m-%d", errors="coerce")):
            problem = "trade_date must be YYYY-MM-DD"
        if problem:
            errors.append(f"Row {row} ({r.trade_id}): {problem}.")
        else:
            keep.append(i)
    out = df.loc[keep].reset_index(drop=True)
    out["N"] = np.where(out.side == "BUY", 1.0, -1.0) * out.notional  # signed base notional
    return out[REQUIRED + ["book", "base", "quote", "N"]], errors
