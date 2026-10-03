"""Edge-case portfolios in tests/portfolios/: parsing, validation and calculation checks.

Each file isolates one behaviour; see tests/portfolios/README.md. Runs offline
against the bundled market snapshot (previous close 2026-10-01).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxrisk import risk as rk
from fxrisk.market import load_snapshot
from fxrisk.portfolio import check_against_market, load_portfolio

DIR = Path(__file__).parent / "portfolios"
METHODS = ["hs", "normal", "ewma"]


@pytest.fixture(scope="module")
def mkt():
    return load_snapshot()


def load(name, mkt):
    book, errors = load_portfolio(DIR / name, mkt.currencies)
    book, mkt_errors, warnings = check_against_market(book, mkt)
    return book, errors + mkt_errors, warnings


def full_run(book, mkt, method="hs"):
    """Every engine function the dashboard calls. Returns the analysis dict."""
    pos, X = rk.value_positions(book, mkt)
    cfg = rk.RiskConfig(method)
    a = rk.analyse(pos, X, mkt, cfg)
    a["positions"], a["X"] = pos, X
    a["backtest"] = rk.backtest(a["exposure"], mkt, cfg, 250)
    a["hedges"] = rk.best_hedges(a["pos"], a["exposure"], mkt, cfg)
    a["history"] = rk.pnl_history(book, mkt)
    return a


def assert_all_finite(a):
    items = a["pos"]["items"]
    assert np.isfinite(items.drop(columns="marginal_per_mm").values).all(), items
    assert np.isfinite([a["pos"]["var"], a["pos"]["es"]]).all()
    assert np.isfinite(a["history"]).all()
    assert np.isfinite(a["hedges"][["hedge_usd", "var_after", "reduction"]].values).all()


# ---------- calculation checks ----------

@pytest.mark.parametrize("method", METHODS)
def test_single_trade(mkt, method):
    """With one trade, portfolio VaR = standalone = component = incremental, and 100% of VaR."""
    book, errors, _ = load("single_trade.csv", mkt)
    assert not errors and len(book) == 1
    a = full_run(book, mkt, method)
    it = a["pos"]["items"].iloc[0]
    var = a["pos"]["var"]
    assert var > 0
    assert it.standalone == pytest.approx(var)
    assert it.component == pytest.approx(var)
    assert it.incremental == pytest.approx(var)
    assert it.pct_of_var == pytest.approx(1.0)
    # Hand calculation: 10mm EUR bought at 1.10, P&L already in USD.
    assert a["positions"].pnl_itd.iloc[0] == pytest.approx(10_000_000 * (mkt.live["EUR"] - 1.10))
    assert a["exposure"]["EUR"] == pytest.approx(10_000_000 * mkt.live["EUR"])
    assert (a["exposure"].drop("EUR") == 0).all()
    assert_all_finite(a)


@pytest.mark.parametrize("method", METHODS)
def test_fully_hedged_book_has_zero_risk(mkt, method):
    """Buy and sell the same amount at the same price: zero P&L, exposure and VaR. No NaNs."""
    book, errors, _ = load("fully_hedged.csv", mkt)
    assert not errors and len(book) == 2
    a = full_run(book, mkt, method)
    assert a["positions"].pnl_itd.sum() == pytest.approx(0, abs=1e-6)
    assert a["positions"].pnl_day.sum() == pytest.approx(0, abs=1e-6)
    assert np.allclose(a["exposure"], 0, atol=1e-6)
    assert a["pos"]["var"] == pytest.approx(0, abs=1e-6) and not np.signbit(a["pos"]["var"])
    assert np.allclose(a["pos"]["items"].component, 0, atol=1e-6)
    # Each leg still has standalone risk; they offset exactly. Parametric VaR is
    # symmetric, so long and short legs match. HS is not: the long leg's VaR comes
    # from the worst EUR days, the short leg's from the best ones.
    st = a["pos"]["items"].standalone
    assert (st > 0).all()
    if method == "hs":
        assert st.iloc[0] != pytest.approx(st.iloc[1])
    else:
        assert st.iloc[0] == pytest.approx(st.iloc[1])
    assert a["backtest"]["exceptions"] == 0
    assert_all_finite(a)


def test_inverse_pair_is_the_same_trade(mkt):
    """BUY 10mm EURUSD @1.10 = SELL 11mm USDEUR @1/1.10: same P&L, exposure and risk."""
    book, errors, _ = load("inverse_pair_equivalence.csv", mkt)
    assert not errors
    a = full_run(book, mkt)
    p, X, items = a["positions"], a["X"], a["pos"]["items"]
    assert p.loc["A", "pnl_itd"] == pytest.approx(p.loc["B", "pnl_itd"], rel=1e-9)
    assert p.loc["A", "pnl_day"] == pytest.approx(p.loc["B", "pnl_day"], rel=1e-9)
    np.testing.assert_allclose(X.loc["A"], X.loc["B"], rtol=1e-9, atol=1e-6)
    assert items.loc["A", "standalone"] == pytest.approx(items.loc["B", "standalone"], rel=1e-9)
    # Identical trades split VaR equally and are perfectly correlated: VaR doubles.
    assert items.loc["A", "component"] == pytest.approx(items.loc["B", "component"], rel=1e-9)
    assert a["pos"]["var"] == pytest.approx(2 * items.loc["A", "standalone"], rel=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_cross_equals_its_usd_legs(mkt, method):
    """BUY EURGBP = BUY EURUSD + SELL GBPUSD at matching amounts and prices (K = K1/K2)."""
    book, errors, _ = load("cross_vs_legs.csv", mkt)
    assert not errors
    pos, X = rk.value_positions(book, mkt)
    np.testing.assert_allclose(X.loc["X1"], X.loc["L1"] + X.loc["L2"], rtol=1e-9, atol=1e-6)
    assert pos.loc["X1", "pnl_itd"] == pytest.approx(pos.loc[["L1", "L2"], "pnl_itd"].sum(), rel=1e-9)
    assert pos.loc["X1", "pnl_day"] == pytest.approx(pos.loc[["L1", "L2"], "pnl_day"].sum(), rel=1e-9)
    cfg = rk.RiskConfig(method)
    var_cross = rk.risk_report(X.loc[["X1"]], mkt.returns, cfg)["var"]
    var_legs = rk.risk_report(X.loc[["L1", "L2"]], mkt.returns, cfg)["var"]
    assert var_cross == pytest.approx(var_legs, rel=1e-9)


def test_trade_opened_today(mkt):
    """A trade dated after the last close counts its whole P&L as today's; an older one does not."""
    book, errors, _ = load("opened_today.csv", mkt)
    assert not errors
    p = rk.value_positions(book, mkt)[0]
    assert bool(p.loc["NEW", "opened_today"]) and not bool(p.loc["OLD", "opened_today"])
    assert p.loc["NEW", "pnl_day"] == pytest.approx(p.loc["NEW", "pnl_itd"])
    assert p.loc["OLD", "pnl_day"] != pytest.approx(p.loc["OLD", "pnl_itd"])
    # Same pair, notional and entry: identical P&L since inception.
    assert p.loc["NEW", "pnl_itd"] == pytest.approx(p.loc["OLD", "pnl_itd"])


def test_trade_before_history_start(mkt):
    """A trade older than the price history still values, and its P&L history starts at the first close."""
    book, errors, warnings = load("before_history.csv", mkt)
    assert not errors and not warnings  # no off-market warning without a trade-date price
    a = full_run(book, mkt)
    h = a["history"]
    assert h.index[0] == mkt.hist.index[0]
    assert h.iloc[-1] == pytest.approx(a["positions"].pnl_itd.iloc[0])
    assert_all_finite(a)


def test_scaling_and_sign_flip(mkt):
    """VaR is homogeneous: 2x notionals = 2x VaR. Flipping every side keeps parametric VaR."""
    book, _ = load_portfolio(currencies=mkt.currencies)
    _, X = rk.value_positions(book, mkt)
    double = book.assign(notional=book.notional * 2, N=book.N * 2)
    flipped = book.assign(side=book.side.map({"BUY": "SELL", "SELL": "BUY"}), N=-book.N)
    _, X2 = rk.value_positions(double, mkt)
    _, Xf = rk.value_positions(flipped, mkt)
    for method in METHODS:
        cfg = rk.RiskConfig(method)
        base = rk.risk_report(X, mkt.returns, cfg)["var"]
        assert rk.risk_report(X2, mkt.returns, cfg)["var"] == pytest.approx(2 * base)
        if method != "hs":  # HS is asymmetric: flipping swaps the loss and gain tails
            assert rk.risk_report(Xf, mkt.returns, cfg)["var"] == pytest.approx(base)
    # For HS, VaR of the flipped book is the k-th best P&L of the original.
    r = rk.risk_report(X, mkt.returns, rk.RiskConfig("hs"))
    kth_best = np.sort(r["pnl"].values)[::-1][r["k"] - 1]
    assert rk.risk_report(Xf, mkt.returns, rk.RiskConfig("hs"))["var"] == pytest.approx(kth_best)


# ---------- parsing and validation ----------

def test_messy_but_valid_file_parses(mkt):
    """BOM, CRLF, odd header case/spacing, reordered + extra columns, blank line, varied formats."""
    book, errors, _ = load("messy_but_valid.csv", mkt)
    assert errors == []
    b = book.set_index("trade_id")
    assert list(b.index) == ["F1", "F2", "F3"]
    assert list(b.currency_pair) == ["EURUSD", "USDJPY", "EURGBP"]
    assert list(b.side) == ["BUY", "SELL", "BUY"]
    assert list(b.notional) == [10_000_000, 5_000_000, 2_000_000]
    assert list(b.entry_price) == [1.10, 150.0, 0.85]
    assert list(b.N) == [10_000_000, -5_000_000, 2_000_000]
    assert list(b.book) == ["Messy", "Messy", "Unassigned"]


def test_invalid_rows_are_rejected_individually(mkt):
    book, errors, _ = load("invalid_rows.csv", mkt)
    assert list(book.trade_id) == ["OK1"] and book.currency_pair.iloc[0] == "EURUSD"
    text = "\n".join(errors)
    for expected in ["'USDTRY' is not a supported pair", "'USDUSD' is not a supported pair",
                     "'EURUSDX' is not a supported pair", "'blank' is not a supported pair",
                     "side must be BUY or SELL", "trade_id is blank",
                     "trade_id OK1 is used more than once"]:
        assert expected in text, expected
    assert text.count("notional must be a positive number") == 2      # negative, and text
    assert text.count("entry_price must be a positive number") == 1   # blank
    assert text.count("trade_date must be a real date") == 2          # 01/06/2026, 2026-02-30
    assert len(errors) == 12
    # The duplicate id must not leak GBP exposure into the kept EURUSD trade.
    _, X = rk.value_positions(book, mkt)
    assert X.loc["OK1", "GBP"] == 0


def test_all_zeros_rejects_every_row(mkt):
    book, errors, _ = load("all_zeros.csv", mkt)
    assert book.empty and len(errors) == 4
    assert sum("notional" in e for e in errors) == 3 and sum("entry_price" in e for e in errors) == 1


@pytest.mark.parametrize("name, message", [
    ("empty_file.csv", "The file is empty."),
    ("header_only.csv", None),
    ("missing_column.csv", "Missing column(s): entry_price. Columns found: trade_id, trade_date"),
])
def test_files_with_no_usable_trades(mkt, name, message):
    book, errors, warnings = load(name, mkt)
    assert book.empty and list(book.columns)[:6] == ["trade_id", "trade_date", "currency_pair", "side",
                                                     "notional", "entry_price"]
    assert warnings == []
    if message:
        assert len(errors) == 1 and errors[0].startswith(message)
    else:
        assert errors == []


def test_missing_book_column_defaults(mkt):
    book, errors, _ = load("no_book_column.csv", mkt)
    assert not errors and list(book.book) == ["Unassigned"]


def test_market_sanity_checks(mkt):
    """Future-dated and clearly wrong prices are rejected; the good trade is kept."""
    book, errors, warnings = load("market_sanity.csv", mkt)
    assert list(book.trade_id) == ["G1"]
    text = "\n".join(errors)
    assert "P1 (USDJPY)" in text and "may be inverted" in text      # 0.00667 = 1/150
    assert "P2 (EURUSD)" in text and "may be inverted" not in text.split("P2")[1]  # 110 = mistyped
    assert "F1 (EURUSD): trade_date 2027-01-15 is after today's prices" in text
    assert len(errors) == 3


def test_off_market_warning_but_accepted(mkt):
    """An entry a few % off the trade-date close is kept but flagged."""
    d = mkt.hist.index[-30]
    close = mkt.spot("EURUSD", mkt.rates_at(len(mkt.hist) - 30))
    csv = ("trade_id,trade_date,currency_pair,side,notional,entry_price\n"
           f"W1,{d},EURUSD,BUY,1000000,{close * 1.05:.5f}\n"
           f"W2,{d},EURUSD,BUY,1000000,{close * 1.001:.5f}\n")
    import io
    book, _ = load_portfolio(io.StringIO(csv), mkt.currencies)
    book, errors, warnings = check_against_market(book, mkt)
    assert list(book.trade_id) == ["W1", "W2"] and not errors
    assert len(warnings) == 1 and warnings[0].startswith("W1") and "+5.0%" in warnings[0]


def test_sample_book_passes_market_checks(mkt):
    book, errors = load_portfolio(currencies=mkt.currencies)
    book, mkt_errors, warnings = check_against_market(book, mkt)
    assert len(book) == 16 and not errors and not mkt_errors and not warnings


# ---------- malformed files (must never raise) ----------

@pytest.mark.parametrize("name, trades, message", [
    ("excel_renamed.csv", 0, "This looks like an Excel workbook"),
    ("not_a_portfolio.csv", 0, "Missing column(s): trade_id"),
    ("ragged_rows.csv", 0, "Row 2: has 9 values but the header has 6 columns."),
    ("infinite_values.csv", 0, "Row 2 (A): notional must be a positive number"),
    ("semicolon_separated.csv", 1, None),   # ; delimiter and 1,10 decimal comma both understood
    ("utf16_encoded.csv", 1, None),
])
def test_malformed_files(mkt, name, trades, message):
    book, errors, _ = load(name, mkt)
    assert len(book) == trades
    if message:
        assert errors and errors[0].startswith(message), errors
    else:
        assert errors == []
    if trades:
        assert book.entry_price.iloc[0] == pytest.approx(1.10)


def test_row_numbers_match_file_lines(mkt):
    """Errors cite the line in the file, even after blank lines."""
    import io
    csv = ("trade_id,trade_date,currency_pair,side,notional,entry_price\n"
           "A,2026-06-01,EURUSD,BUY,1000000,1.10\n"
           "\n\n"
           "B,2026-06-01,EURUSD,HOLD,1000000,1.10\n")
    _, errors = load_portfolio(io.StringIO(csv), mkt.currencies)
    assert errors == ["Row 5 (B): side must be BUY or SELL."]
