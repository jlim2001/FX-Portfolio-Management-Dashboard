"""Current and historical FX rates.

Every rate is held as "USD per 1 unit of currency" (EUR -> 1.17, JPY -> 0.0064,
USD -> 1). Any pair price is BASE/QUOTE = usd_per[BASE] / usd_per[QUOTE], so
crosses (EURGBP, AUDNZD, ...) are triangulated from the same USD legs that
drive the risk factors.

Source chain (first one that works wins):
  1. Yahoo Finance chart API: daily closes plus the latest intraday quote.
  2. Frankfurter / ECB reference rates: daily 16:00 CET fixings.
  3. The bundled snapshot in data/market_snapshot.json (offline / rate-limited).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

CURRENCIES = ["EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "NOK", "SEK", "MXN", "ZAR", "SGD"]
# Yahoo quotes these as XXXUSD (USD per unit); all others are USDXXX and get inverted.
USD_QUOTED = {"EUR", "GBP", "AUD", "NZD"}
SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "market_snapshot.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (fx-dashboard)"}


@dataclass
class MarketData:
    hist: pd.DataFrame      # close dates x currencies, USD per unit (no USD column)
    live: pd.Series         # latest USD per unit, includes USD = 1
    live_time: pd.Timestamp
    source: str

    @property
    def currencies(self) -> list[str]:
        return list(self.hist.columns)

    @property
    def prev(self) -> pd.Series:
        """Last official close (the reference for daily P&L)."""
        return pd.concat([self.hist.iloc[-1], pd.Series({"USD": 1.0})])

    @property
    def prev_date(self) -> str:
        return self.hist.index[-1]

    @property
    def returns(self) -> pd.DataFrame:
        """Simple daily returns of USD-per-unit rates: the risk factors."""
        return self.hist.pct_change().iloc[1:]

    def rates_at(self, i: int) -> pd.Series:
        return pd.concat([self.hist.iloc[i], pd.Series({"USD": 1.0})])

    def spot(self, pair: str, u: pd.Series | None = None) -> float:
        u = self.live if u is None else u
        return float(u[pair[:3]] / u[pair[3:]])


# ---------- sources ----------

def _yahoo_series(ccy: str, years: int) -> tuple[pd.Series, float, int]:
    ticker = f"{ccy}USD=X" if ccy in USD_QUOTED else f"{ccy}=X"
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                     params={"range": f"{years}y", "interval": "1d"}, headers=HEADERS, timeout=20)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    # Bars are stamped at London midnight; shift by the exchange offset so each
    # stamp lands on its own trading date rather than the prior UTC day.
    offset = int(res["meta"].get("gmtoffset", 0))
    dates = pd.to_datetime([t + offset for t in res["timestamp"]], unit="s").strftime("%Y-%m-%d")
    closes = pd.Series(res["indicators"]["quote"][0]["close"], index=dates, dtype="float64")
    closes = closes[~closes.index.duplicated(keep="last")].dropna()
    live = float(res["meta"]["regularMarketPrice"])
    if ccy not in USD_QUOTED:
        closes, live = 1.0 / closes, 1.0 / live
    return closes, live, int(res["meta"]["regularMarketTime"])


def fetch_yahoo(years: int = 3) -> MarketData:
    raw, live, stamps = {}, {}, []
    for ccy in CURRENCIES:
        raw[ccy], live[ccy], ts = _yahoo_series(ccy, years)
        stamps.append(ts)
    live_time = pd.Timestamp(min(stamps), unit="s", tz="UTC")
    # The bar for the live quote's own session is still moving (or, at a weekend,
    # equals the live quote): keep only earlier closes, so "previous close" is the
    # close before the session the live quote belongs to.
    session = (live_time + pd.Timedelta(hours=1)).strftime("%Y-%m-%d")  # London date of the quote
    hist = {c: s[s.index < session] for c, s in raw.items()}
    return MarketData(_clean(pd.DataFrame(hist)), pd.Series({**live, "USD": 1.0}), live_time,
                      "Yahoo Finance (daily closes; live = latest intraday quote)")


def fetch_ecb(years: int = 3) -> MarketData:
    start = (dt.date.today() - dt.timedelta(days=365 * years + 7)).isoformat()
    r = requests.get(f"https://api.frankfurter.dev/v1/{start}..",
                     params={"base": "USD", "symbols": ",".join(CURRENCIES)}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = 1.0 / pd.DataFrame(r.json()["rates"]).T.astype("float64").sort_index()  # ccy per USD -> USD per ccy
    df = df[CURRENCIES]
    last = df.index[-1]
    return MarketData(_clean(df.iloc[:-1]), pd.Series({**df.iloc[-1].to_dict(), "USD": 1.0}),
                      pd.Timestamp(f"{last} 14:00", tz="UTC"),
                      "ECB reference rates via Frankfurter (live = latest 16:00 CET fixing)")


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_index()
    frame = frame[pd.to_datetime(frame.index).dayofweek < 5]
    # Bad ticks: isolated spikes that revert next day (seen in Yahoo ZAR). Points
    # >8% from the centred 5-day median are treated as missing, so a bad print
    # never becomes a fake VaR scenario.
    med = frame.rolling(5, center=True, min_periods=3).median()
    spikes = (frame / med - 1).abs() > 0.08
    for ccy in frame.columns[spikes.any()]:
        log.warning("%s: removed bad prints on %s", ccy, list(frame.index[spikes[ccy]]))
    # Holidays differ across tickers: carry the last close over gaps of <= 3 days.
    return frame.mask(spikes).ffill(limit=3).dropna()


# ---------- snapshot ----------

def save_snapshot(m: MarketData, path: Path = SNAPSHOT) -> None:
    path.write_text(json.dumps({
        "source": m.source, "live_time": m.live_time.isoformat(),
        "dates": list(m.hist.index),
        "usd_per": {c: [round(float(v), 10) for v in m.hist[c]] for c in m.currencies},
        "live": {c: round(float(m.live[c]), 10) for c in m.currencies},
    }, separators=(",", ":")))


def load_snapshot(path: Path = SNAPSHOT) -> MarketData:
    d = json.loads(Path(path).read_text())
    hist = pd.DataFrame(d["usd_per"], index=d["dates"])
    return MarketData(hist, pd.Series({**d["live"], "USD": 1.0}), pd.Timestamp(d["live_time"]),
                      f"Bundled snapshot: {d['source']}")


def load_market(source: str = "auto", years: int = 3) -> tuple[MarketData, list[str]]:
    """Return market data and a list of warnings describing any fallbacks taken."""
    chain = {"auto": [fetch_yahoo, fetch_ecb], "yahoo": [fetch_yahoo], "ecb": [fetch_ecb], "snapshot": []}[source]
    warnings = []
    for fn in chain:
        try:
            return fn(years), warnings
        except Exception as exc:  # network, rate limit, schema change
            warnings.append(f"{fn.__name__.replace('fetch_', '').upper()} unavailable ({type(exc).__name__}: {exc})")
    return load_snapshot(), warnings
