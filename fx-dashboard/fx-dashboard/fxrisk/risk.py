"""Valuation and risk for a book of FX spot trades.

A spot trade is two cash legs: +N BASE and -N*K QUOTE (N signed base notional,
K entry price). Its USD value is

    V = N*u[BASE] - N*K*u[QUOTE]          (= N*(S - K)*u[QUOTE], the USD MTM)

with u[c] = USD per unit of c. Risk factors are the simple daily returns R[c]
of u[c]. V is linear in u, so the scenario P&L

    dV = sum_c x[c] * R[c],   x[BASE] = N*u[BASE],  x[QUOTE] = -N*K*u[QUOTE]

is an exact full revaluation for spot, not a delta approximation. The USD leg
carries no risk for a USD-reporting book and is dropped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .market import MarketData

EPS = 1e-9


@dataclass(frozen=True)
class RiskConfig:
    method: str = "hs"        # "hs" historical simulation | "normal" | "ewma" (parametric)
    conf: float = 0.99
    horizon: int = 1          # days; sqrt(h) scaling for every method
    window: int = 500         # look-back in daily returns
    lam: float = 0.94         # EWMA decay (RiskMetrics)

    @property
    def scale(self) -> float:
        return math.sqrt(self.horizon)


# ---------- valuation ----------

def _usd_value(book: pd.DataFrame, u: pd.Series) -> np.ndarray:
    return (book.N * (u[book.base].values - book.entry_price * u[book.quote].values)).values


def value_positions(book: pd.DataFrame, m: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-trade valuation and the trade x currency USD exposure matrix."""
    pos = book.copy()
    pos["spot"] = [m.spot(p) for p in pos.currency_pair]
    pos["prev_close"] = [m.spot(p, m.prev) for p in pos.currency_pair]
    pos["pnl_itd"] = _usd_value(book, m.live)
    pos["opened_today"] = pos.trade_date > m.prev_date
    # Daily P&L = change in USD value since the last close. A trade opened after
    # that close has no prior value, so its daily P&L is its full P&L.
    pos["pnl_day"] = np.where(pos.opened_today, pos.pnl_itd, pos.pnl_itd - _usd_value(book, m.prev))
    pos["usd_notional"] = pos.N.abs() * m.live[pos.base].values
    sign = np.sign(pos.N)
    pos["move_pct"] = sign * (pos.spot / pos.entry_price - 1)

    X = pd.DataFrame(0.0, index=pos.trade_id, columns=m.currencies)
    for r in pos.itertuples():
        if r.base != "USD":
            X.loc[r.trade_id, r.base] += r.N * m.live[r.base]
        if r.quote != "USD":
            X.loc[r.trade_id, r.quote] -= r.N * r.entry_price * m.live[r.quote]
    return pos.set_index("trade_id"), X


# ---------- VaR primitives ----------

def tail_count(n: int, conf: float) -> int:
    """VaR is the k-th worst scenario, ES the mean of the k worst."""
    return max(1, math.ceil(n * (1 - conf) - EPS))


def hs_var(pnl: np.ndarray, conf: float) -> tuple[float, float, int]:
    s = np.sort(pnl)
    k = tail_count(len(s), conf)
    return -s[k - 1], -s[:k].mean(), k


def covariance(R: np.ndarray, lam: float | None = None) -> np.ndarray:
    """Zero-mean covariance; equal weights, or EWMA weights lam^(age) when lam is given."""
    n = len(R)
    w = lam ** np.arange(n - 1, -1, -1) if lam else np.ones(n)
    w = w / w.sum()
    return (R * w[:, None]).T @ R


def risk_report(X: pd.DataFrame, R: pd.DataFrame, cfg: RiskConfig) -> dict:
    """Portfolio VaR/ES and per-item standalone, component, incremental VaR and component ES.

    X: items x currencies USD exposures. R: scenarios x currencies returns.
    """
    R = R.iloc[-cfg.window:]
    Rv, Xv, s = R.values, X.values, cfg.scale
    e = Xv.sum(axis=0)
    Pi = Rv @ Xv.T                       # scenario P&L, scenarios x items
    Pp = Pi.sum(axis=1)

    # Historical simulation is always computed: it supplies the P&L distribution,
    # the ES decomposition and the HS figures shown next to the parametric ones.
    v_hs, es_hs, k = hs_var(Pp, cfg.conf)
    order = np.argsort(Pp, kind="stable")
    S = covariance(Rv, cfg.lam if cfg.method == "ewma" else None)
    z = norm.ppf(cfg.conf)
    sigma = math.sqrt(max(e @ S @ e, 0.0))
    v_n, es_n = z * sigma * s, sigma * norm.pdf(z) / (1 - cfg.conf) * s

    if cfg.method == "hs":
        var, es = v_hs * s + 0.0, es_hs * s + 0.0  # + 0.0 turns -0.0 into 0.0 for display
        standalone = np.array([hs_var(Pi[:, i], cfg.conf)[0] for i in range(Pi.shape[1])]) * s
        incremental = var - np.array([hs_var(Pp - Pi[:, i], cfg.conf)[0] for i in range(Pi.shape[1])]) * s
        # Smoothed Euler estimator: average each item's P&L over the scenarios
        # ranked k-m..k+m around the VaR scenario, rescaled so the components
        # sum to portfolio VaR exactly (a single-scenario estimate is too noisy).
        m = max(1, k // 2)
        band = order[max(0, k - 1 - m): k + m]
        pp_band = Pp[band].mean()
        # A fully hedged book has zero P&L in every scenario: nothing to allocate.
        component = Pi[band].mean(axis=0) / pp_band * var if abs(pp_band) > EPS else np.zeros(Pi.shape[1])
    else:
        var, es = v_n, es_n
        quad = lambda A: np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", A, S, A), 0))
        standalone = z * quad(Xv) * s
        incremental = var - z * quad(e[None, :] - Xv) * s
        component = z * (Xv @ S @ e) / sigma * s if sigma else np.zeros(len(Xv))

    items = pd.DataFrame({
        "standalone": standalone, "component": component, "incremental": incremental,
        "component_es": -Pi[order[:k]].mean(axis=0) * s,  # exact HS ES decomposition
    }, index=X.index)
    items["pct_of_var"] = items.component / var if var else 0.0
    return {
        "var": var, "es": es, "hs_var": v_hs * s, "hs_es": es_hs * s, "normal_var": v_n, "normal_es": es_n,
        "k": k, "sigma": sigma, "S": S, "items": items,
        "pnl": pd.Series(Pp, index=R.index), "window_start": R.index[0], "window_end": R.index[-1],
        "var_scenario_date": R.index[order[k - 1]],
    }


def analyse(pos: pd.DataFrame, X: pd.DataFrame, m: MarketData, cfg: RiskConfig) -> dict:
    """Position-level and currency-level risk for the book."""
    R = m.returns
    by_pos = risk_report(X, R, cfg)
    # Marginal VaR: change in portfolio VaR per +$1mm of USD notional added to
    # this position in its current direction. By Euler, component = size x marginal.
    by_pos["items"]["marginal_per_mm"] = by_pos["items"].component / pos.usd_notional * 1e6
    exposure = X.sum(axis=0)
    by_ccy = risk_report(pd.DataFrame(np.diag(exposure.values), index=exposure.index, columns=exposure.index), R, cfg)
    return {"pos": by_pos, "ccy": by_ccy, "exposure": exposure}


def compare_methods(X: pd.DataFrame, m: MarketData, cfg: RiskConfig) -> pd.DataFrame:
    rows = []
    for name, method in [("Historical simulation", "hs"), ("Parametric, equal weights", "normal"),
                         (f"Parametric, EWMA (λ={cfg.lam})", "ewma")]:
        r = risk_report(X, m.returns, RiskConfig(method, cfg.conf, cfg.horizon, cfg.window, cfg.lam))
        rows.append({"Method": name, "VaR": r["var"], "Expected shortfall": r["es"]})
    return pd.DataFrame(rows).set_index("Method")


# ---------- hedging ----------

def best_hedges(report: dict, exposure: pd.Series, m: MarketData, cfg: RiskConfig) -> pd.DataFrame:
    """For each currency, the trade vs USD that minimises parametric portfolio variance.

    Minimising (e + d*1_c)' S (e + d*1_c) over d gives d = -(S e)_c / S_cc and a
    variance reduction of (S e)_c^2 / S_cc.
    """
    S, e = report["S"], exposure.values
    Se, d = S @ e, np.diag(S)
    sigma = math.sqrt(max(e @ S @ e, 0.0))
    after = np.sqrt(np.maximum(sigma**2 - Se**2 / d, 0))
    z, s = norm.ppf(cfg.conf), cfg.scale
    out = pd.DataFrame({
        "hedge_usd": -Se / d,
        "hedge_ccy": -Se / d / m.live[exposure.index].values,
        "var_before": z * sigma * s, "var_after": z * after * s,
        "reduction": 1 - after / sigma if sigma else 0.0,
    }, index=exposure.index)
    return out.sort_values("reduction", ascending=False)


# ---------- backtesting ----------

def backtest(exposure: pd.Series, m: MarketData, cfg: RiskConfig, days: int = 250) -> dict:
    """Rolling 1-day VaR of today's book vs the P&L it would have made each day (hypothetical P&L)."""
    R = m.returns
    e = exposure[R.columns].values
    P = R.values @ e
    n = min(days, len(P) - cfg.window)
    z = norm.ppf(cfg.conf)
    var = np.empty(n)
    for j, t in enumerate(range(len(P) - n, len(P))):
        if cfg.method == "hs":
            var[j] = hs_var(P[t - cfg.window:t], cfg.conf)[0]
        else:
            S = covariance(R.values[t - cfg.window:t], cfg.lam if cfg.method == "ewma" else None)
            var[j] = z * math.sqrt(max(e @ S @ e, 0.0))
    df = pd.DataFrame({"pnl": P[-n:], "var": var}, index=R.index[-n:])
    df["breach"] = df.pnl < -df["var"]
    x, p = int(df.breach.sum()), 1 - cfg.conf
    # Kupiec proportion-of-failures test; ll(0) = ll(1) = 0 as limits.
    ll = lambda q: 0.0 if q <= 0 or q >= 1 else (n - x) * math.log(1 - q) + x * math.log(q)
    lr = -2 * (ll(p) - ll(x / n))
    pval = float(chi2.sf(lr, 1))
    if abs(cfg.conf - 0.99) < 1e-9 and n == 250:  # Basel traffic light
        zone = "green" if x <= 4 else "yellow" if x <= 9 else "red"
    else:
        zone = "green" if pval >= 0.05 else "yellow" if pval >= 1e-4 else "red"
    return {"series": df, "n": n, "exceptions": x, "expected": n * p, "lr": lr, "p_value": pval, "zone": zone}


# ---------- P&L history ----------

def pnl_history(book: pd.DataFrame, m: MarketData) -> pd.Series:
    """Inception-to-date USD P&L of the book over time; trades enter on their trade date."""
    if book.empty:
        return pd.Series(dtype=float)
    first = book.trade_date.min()
    out = {}
    for i, d in enumerate(m.hist.index):
        if d < first:
            continue
        live_trades = book[book.trade_date <= d]
        out[d] = _usd_value(live_trades, m.rates_at(i)).sum()
    # Final point: live prices (dated today, after the last close).
    out[m.live_time.strftime("%Y-%m-%d")] = _usd_value(book, m.live).sum()
    return pd.Series(out)


def correlation(S: np.ndarray, labels) -> pd.DataFrame:
    sd = np.sqrt(np.diag(S))
    return pd.DataFrame(S / np.outer(sd, sd), index=labels, columns=labels)
