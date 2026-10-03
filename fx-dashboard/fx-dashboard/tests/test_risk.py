"""Tests for valuation and risk. Run offline against the bundled snapshot:

    python -m pytest -q
"""
import io
import math

import numpy as np
import pandas as pd
import pytest

from fxrisk.market import load_snapshot
from fxrisk.portfolio import load_portfolio
from fxrisk import risk as rk

METHODS = ["hs", "normal", "ewma"]


@pytest.fixture(scope="module")
def mkt():
    return load_snapshot()


@pytest.fixture(scope="module")
def book(mkt):
    b, errors = load_portfolio(currencies=mkt.currencies)
    assert not errors
    return b


@pytest.fixture(scope="module")
def valued(book, mkt):
    return rk.value_positions(book, mkt)


# ---------- valuation ----------

def test_pnl_hand_calculations(valued, book, mkt):
    pos, _ = valued
    u = mkt.live
    for pair, expected in [
        ("EURUSD", lambda t: t.N * (u.EUR - t.entry_price)),                       # USD quote: P&L already USD
        ("USDJPY", lambda t: t.N * (u.USD / u.JPY - t.entry_price) * u.JPY),       # JPY P&L converted at spot
        ("EURGBP", lambda t: t.N * (u.EUR / u.GBP - t.entry_price) * u.GBP),       # cross
    ]:
        t = book[book.currency_pair == pair].iloc[0]
        assert pos.loc[t.trade_id, "pnl_itd"] == pytest.approx(expected(t), rel=1e-12)


def test_daily_pnl(valued, book, mkt):
    pos, _ = valued
    old = pos[~pos.opened_today]
    at_prev = (old.N * (mkt.prev[old.base].values - old.entry_price * mkt.prev[old.quote].values))
    np.testing.assert_allclose(old.pnl_day, old.pnl_itd - at_prev)
    new = pos[pos.opened_today]
    assert len(new) >= 1 and np.allclose(new.pnl_day, new.pnl_itd)


def test_scenario_pnl_is_exact_full_revaluation(valued, book, mkt):
    """X @ shock must equal repricing every trade at shocked rates (spot is linear in u)."""
    pos, X = valued
    rng = np.random.default_rng(0)
    shock = pd.Series(rng.normal(0, 0.05, len(mkt.currencies)), index=mkt.currencies)
    u1 = mkt.live.copy()
    u1[mkt.currencies] *= 1 + shock
    repriced = book.N.values * (u1[book.base].values - book.entry_price.values * u1[book.quote].values)
    np.testing.assert_allclose(repriced - pos.pnl_itd.values, X.values @ shock.values, rtol=1e-10, atol=1e-6)


def test_history_ends_at_itd(book, mkt, valued):
    h = rk.pnl_history(book, mkt)
    assert h.iloc[-1] == pytest.approx(valued[0].pnl_itd.sum())


# ---------- VaR primitives ----------

def test_hs_var_definition():
    pnl = -np.arange(1, 501, dtype=float)[::-1]  # losses 1..500
    v, es, k = rk.hs_var(pnl, 0.99)
    assert k == 5 and v == 496 and es == pytest.approx(np.mean([500, 499, 498, 497, 496]))
    assert rk.tail_count(500, 0.99) == 5  # guards the 500*(1-0.99) = 5.000000000000004 float trap
    assert rk.tail_count(250, 0.975) == 7


def test_normal_var_matches_closed_form(valued, mkt):
    _, X = valued
    cfg = rk.RiskConfig("normal", 0.99, 1, 500)
    r = rk.risk_report(X, mkt.returns, cfg)
    R = mkt.returns.iloc[-500:].values
    sd = np.sqrt(np.mean((R @ X.values.sum(0)) ** 2))  # zero-mean sigma of portfolio P&L
    assert r["var"] == pytest.approx(2.3263478740 * sd, rel=1e-8)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("horizon", [1, 10])
def test_euler_additivity(valued, mkt, method, horizon):
    _, X = valued
    r = rk.risk_report(X, mkt.returns, rk.RiskConfig(method, 0.99, horizon, 500))
    assert r["items"].component.sum() == pytest.approx(r["var"], rel=1e-10)
    assert r["items"].component_es.sum() == pytest.approx(r["hs_es"], rel=1e-10)


@pytest.mark.parametrize("method", ["normal", "ewma"])
def test_parametric_marginal_var_matches_finite_difference(valued, mkt, method):
    """Component VaR_i = d VaR / d (size_i) evaluated at size 1, checked numerically."""
    _, X = valued
    cfg = rk.RiskConfig(method, 0.99, 1, 500)
    r = rk.risk_report(X, mkt.returns, cfg)
    h = 1e-5
    for i, tid in enumerate(X.index[:6]):
        up, dn = X.copy(), X.copy()
        up.iloc[i] *= 1 + h
        dn.iloc[i] *= 1 - h
        fd = (rk.risk_report(up, mkt.returns, cfg)["var"] - rk.risk_report(dn, mkt.returns, cfg)["var"]) / (2 * h)
        assert r["items"].loc[tid, "component"] == pytest.approx(fd, rel=1e-5)


@pytest.mark.parametrize("method", METHODS)
def test_incremental_var_is_with_minus_without(valued, mkt, method):
    _, X = valued
    cfg = rk.RiskConfig(method, 0.99, 1, 500)
    full = rk.risk_report(X, mkt.returns, cfg)["var"]
    tid = X.index[0]
    without = rk.risk_report(X.drop(tid), mkt.returns, cfg)["var"]
    assert rk.risk_report(X, mkt.returns, cfg)["items"].loc[tid, "incremental"] == pytest.approx(full - without)


def test_currency_and_position_views_agree(valued, mkt, book):
    pos, X = valued
    a = rk.analyse(pos, X, mkt, rk.RiskConfig())
    assert a["pos"]["var"] == pytest.approx(a["ccy"]["var"])
    assert a["ccy"]["items"].component.sum() == pytest.approx(a["pos"]["var"])
    np.testing.assert_allclose(a["pos"]["items"].marginal_per_mm * pos.usd_notional / 1e6, a["pos"]["items"].component)


def test_best_hedge_is_variance_minimising(valued, mkt):
    pos, X = valued
    cfg = rk.RiskConfig("normal")
    a = rk.analyse(pos, X, mkt, cfg)
    h = rk.best_hedges(a["pos"], a["exposure"], mkt, cfg)
    S, e = a["pos"]["S"], a["exposure"].values
    for c in h.index[:3]:
        j = list(a["exposure"].index).index(c)
        var = lambda d: (e + d * np.eye(len(e))[j]) @ S @ (e + d * np.eye(len(e))[j])
        d0 = h.loc[c, "hedge_usd"]
        assert var(d0) <= min(var(d0 * 1.01), var(d0 * 0.99))
        assert h.loc[c, "var_after"] == pytest.approx(2.3263478740 * math.sqrt(var(d0)), rel=1e-6)


# ---------- backtest ----------

def test_backtest_kupiec(valued, mkt):
    pos, X = valued
    bt = rk.backtest(X.sum(), mkt, rk.RiskConfig(), 250)
    assert bt["n"] == 250 and bt["exceptions"] == int(bt["series"].breach.sum())
    x, n, p = bt["exceptions"], 250, 0.01
    lr = -2 * ((n - x) * math.log(1 - p) + x * math.log(p)
               - ((n - x) * math.log(1 - x / n) + (x * math.log(x / n) if x else 0)))
    assert bt["lr"] == pytest.approx(lr)


# ---------- input validation ----------

def test_portfolio_validation(mkt):
    csv = io.StringIO(
        "trade_id,trade_date,currency_pair,side,notional,entry_price\n"
        "A,2026-01-05,EUR/USD,buy,\"1,000,000\",1.10\n"
        "B,2026-01-05,EURXYZ,BUY,1000000,1.10\n"
        "C,2026-01-05,USDJPY,HOLD,1000000,150\n"
        "D,2026-01-05,USDJPY,SELL,-5,150\n"
        "E,05/01/2026,USDJPY,SELL,5,150\n"
    )
    b, errors = load_portfolio(csv, mkt.currencies)
    assert list(b.trade_id) == ["A"] and b.N.iloc[0] == 1_000_000 and b.currency_pair.iloc[0] == "EURUSD"
    assert len(errors) == 4
