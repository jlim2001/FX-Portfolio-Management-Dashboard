# FX Portfolio Risk Dashboard

A Streamlit dashboard for an FX portfolio manager to monitor P&L and risk on a book of FX spot trades. Everything is written in Python (pandas, NumPy, SciPy, Plotly).

**Live dashboard:** https://fx-portfolio-management-dashboard-3crh3ryzox35uajehepqjp.streamlit.app/

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
python -m pytest -q           # 51 tests, run offline against the bundled snapshot
```

The dashboard opens blank. Upload a trade CSV in the sidebar, or click **Or try it with the sample book**. The landing page lists the expected columns and offers a template CSV to download.

**Walkthrough notebook:** [`notebooks/fx_risk_walkthrough.ipynb`](notebooks/fx_risk_walkthrough.ipynb) repeats every dashboard calculation step by step:
- by hand in NumPy, with the formulas;
- with the dashboard's own functions;
- checking that the two agree (46 checks on the sample book).

Set `PORTFOLIO` in its first cell to any CSV, including the files in `tests/portfolios/`. It needs `pip install matplotlib jupyter` in addition to the requirements.

**Deploying to Streamlit Community Cloud (free):**
1. Push this folder to a GitHub repository.
2. At share.streamlit.io, choose *Create app*, then pick the repository, the branch and `app.py`.
3. Deploy. You get a public `https://<name>.streamlit.app` link.

`requirements.txt` and `.streamlit/config.toml` are already set up for this.

## What the dashboard shows

| Area | Contents |
|---|---|
| Headline | P&L since inception, P&L today, VaR, expected shortfall, VaR-limit usage, gross notional. A banner appears when usage passes 85% or 100%. |
| Positions | Every trade: entry price, live spot, move since entry, USD notional, daily and inception P&L, standalone and component VaR. A by-book summary follows. |
| Risk decomposition | Component VaR by trade and by currency. Position table with standalone, component, marginal (per $1mm) and incremental VaR, plus component ES. Diversification benefit. Scenario P&L histogram with HS VaR, HS ES and normal VaR marked. VaR under all three methods side by side. |
| Exposure & hedging | Net USD-equivalent exposure per currency, including the implied USD leg. Best single-currency hedges ranked by VaR reduction. |
| VaR backtest | 250 days of today's book's hypothetical P&L against the prior day's VaR. Exception count, Kupiec test, Basel traffic-light zone. |
| P&L history | Book P&L since the first trade, built from the real trade dates. Daily P&L bars, best and worst day, max drawdown, hit rate. |
| What-if trade | Prices a proposed trade at live mid. Shows the change in VaR, ES and limit usage, the trade's own component VaR, and before/after tables of currency exposure and component VaR by currency. |
| Market | Live spot, previous close, 1-day and 1-month change, 3-month and 1-year realised vol, and the correlation matrix of the risk factors. |

**Sidebar controls:**
- Which price source is in use right now, and why. Green means Yahoo is live, or the market is closed for the weekend. Amber means it fell back to ECB, with the reason. Red means it is using the bundled snapshot. There is also a refresh button.
- Portfolio CSV upload, with row-level validation messages.
- Book filter.
- VaR method, confidence (95 / 97.5 / 99%), horizon (1 / 5 / 10 days), look-back (250 / 500 / 750 days) and EWMA λ.
- VaR limit, with the measure it is set on (default $1mm on 99% 1-day historical VaR over 500 days). Limit usage always uses that measure, so changing the analysis settings never changes it. Viewing 95% VaR does not make the book look further from its limit.

## Data

**Portfolio:** the user uploads a CSV; nothing loads by default. `data/portfolio.csv` is a sample book, available from the landing page. Columns are `trade_id, trade_date, currency_pair, side, notional, entry_price, book`. Notional is in the base currency. The sample book has 16 trades across 4 books:
- majors, crosses (EURGBP, AUDNZD, EURNOK, EURJPY) and EM pairs (MXN, ZAR, SGD);
- longs and shorts, including a partial unwind (T014);
- a trade dated yesterday and one dated today.

Entry prices are the actual close on each trade date plus a small random execution offset (`scripts/make_sample_portfolio.py`). Any CSV with these columns can be uploaded in the app. Pairs can be written `EURUSD`, `EUR/USD` or `eur-usd`, and notionals as `1,000,000` or `1_000_000`.

Uploads are validated row by row, and each bad row is rejected with its own message. Rows are rejected for:
- an unsupported or malformed pair;
- a side other than BUY or SELL;
- a notional or price that is not positive;
- an invalid date;
- a blank or duplicate `trade_id`;
- a trade dated after today's prices;
- an entry price more than 25% from the market on the trade date. An inverted quote gets a hint saying so.

Entry prices more than 3% off-market are accepted but flagged.

A file that can't be used never crashes the app. Instead the main page names the problem and shows the expected format. Cases covered:
- an Excel workbook renamed to .csv;
- a binary file;
- prose text;
- missing columns (the message lists the columns that were found);
- rows with extra values;
- every row invalid.

Semicolon-separated files with decimal commas, UTF-16 and Windows-1252 encodings are read correctly. Error messages give the row's real line number in the file. `tests/portfolios/` holds edge-case files that exercise each of these.

**Market data:** `fxrisk/market.py`. Three years of daily closes plus a live quote for 12 currencies against USD. The app always uses Yahoo Finance. It falls back only if Yahoo fails, and the sidebar says which source is in use and why. Results are cached for 15 minutes. The order is:
1. **Yahoo Finance** chart API: daily closes plus the latest intraday quote.
2. **ECB reference rates** via the Frankfurter API: daily 16:00 CET fixings.
3. **Bundled snapshot** `data/market_snapshot.json`, so the app still works offline or if both APIs are blocked. Refresh it with `python scripts/refresh_snapshot.py`.

Two data issues are handled explicitly:
- Yahoo stamps daily bars at London midnight. Read naively in UTC, every date lands one day early. Timestamps are shifted by the exchange offset.
- Yahoo ZAR has isolated spike-and-revert bad ticks (Nov 2024 and Jan 2025, about 20%). Any point more than 8% from its centred 5-day median is treated as missing and forward-filled. Otherwise those ticks would become fake VaR scenarios.

## Methodology

### Valuation and P&L
All rates are stored as **u(c) = USD per one unit of currency c**, with u(USD) = 1. A pair BASE/QUOTE is priced as u(BASE)/u(QUOTE), so crosses are triangulated from the same USD rates that drive risk.

A spot trade with signed base notional N and entry price K is two cash legs: +N BASE and −N·K QUOTE. Its USD value is

```
V = N·u(BASE) − N·K·u(QUOTE)  =  N·(S − K)·u(QUOTE)
```

- **P&L since inception** is V at live rates. Quote-currency P&L is converted to USD at the live rate.
- **P&L today** is V at live rates minus V at the previous official close. A trade booked after that close counts its full P&L as today's.

### Risk factors and exposures
The risk factors are the 12 simple daily returns R(c) of u(c). Each trade maps to USD exposures x(BASE) = N·u(BASE) and x(QUOTE) = −N·K·u(QUOTE). The USD leg carries no risk for a USD-reporting book. Because V is linear in u, the scenario P&L Σ x(c)·R(c) is an **exact full revaluation** for spot, not a delta approximation (a test checks this). Exposures net across trades, so EURUSD, EURGBP and EURJPY all contribute to one EUR exposure.

### VaR and expected shortfall
| | Default | Notes |
|---|---|---|
| Method | Historical simulation (HS) | Parametric normal (equal-weight) and EWMA (RiskMetrics, λ = 0.94) are also available. |
| Confidence | 99% | 95% and 97.5% are also available. |
| Horizon | 1 day | 5 and 10 days are scaled by √h. |
| Look-back | 500 days (about 2 years) | 250 and 750 are also available. |

**Why HS by default:** it needs no distributional assumption, and FX returns have fat tails. On this book, HS VaR is about 23% above normal VaR. Two years of data covers several volatile episodes (the 2024 yen carry unwind, the April 2025 tariff shock) while staying reasonably current.

- **HS:** today's exposures are applied to each of the last *n* daily moves. VaR is the k-th worst P&L, with k = ⌈n·(1−α)⌉ (the 5th worst of 500 at 99%). ES is the mean of the k worst. The code guards against the float trap where 500·(1−0.99) = 5.000000000000004 would round up to 6.
- **Parametric:** VaR = z_α·√(eᵀΣe) and ES = σ·φ(z_α)/(1−α), with a zero-mean covariance Σ that is either equal-weight or EWMA.

### Decomposition (position level and currency level)
- **Standalone VaR**: the VaR of the trade on its own.
- **Component VaR** (Euler allocation; components sum exactly to portfolio VaR):
  - Parametric: CVaRᵢ = z·xᵢᵀΣe / σ_p.
  - HS: the derivative is estimated with a smoothed estimator. Each trade's P&L is averaged over the scenarios ranked k−m to k+m around the VaR scenario (m = ⌊k/2⌋), then rescaled so the components add up to VaR. A single scenario is too noisy to use.
- **Marginal VaR**: ∂VaR/∂(position size). It is reported as the change in portfolio VaR per +$1mm USD notional in the trade's current direction, equal to CVaRᵢ / notionalᵢ × 1mm. A test checks it against a finite-difference derivative.
- **Incremental VaR**: VaR(book) − VaR(book without the trade). This is an exact recalculation, which is what a PM needs before closing a trade.
- **Component ES**: each trade's mean P&L across the k tail scenarios. It sums exactly to HS ES.

### Other analytics
- **Best hedge:** for each currency c, the USD trade d = −(Σe)_c/Σ_cc minimises portfolio variance. The variance falls by (Σe)_c²/Σ_cc.
- **Backtest:** each of the last 250 days gets a VaR computed only from data available before that day. That VaR is compared with the P&L today's book would have made that day.
  - **Kupiec POF:** a likelihood-ratio test, χ²(1), of whether the exception count matches the confidence level.
  - **Basel traffic light:** green 0–4 exceptions, yellow 5–9, red 10+, at 99% over 250 days. For other settings the zone falls back to the Kupiec p-value.

## Assumptions
- Positions are FX spot held at mid. There are no bid/ask spreads, transaction costs or funding costs.
- The reporting currency is USD. A rate that is missing for 3 days or fewer is forward-filled.
- Each trade is held separately. Offsetting trades (T014 against T001 and T010) net through exposures rather than through FIFO matching of realised P&L, so all P&L is unrealised MTM.
- Daily P&L uses the source's last close as the official close: London-midnight bars from Yahoo, or the 16:00 CET fixing from ECB. "Live" is the latest quote, and each currency's quote time can differ by a few minutes. The app shows the earliest quote time.
- Returns are i.i.d. for √h scaling, and parametric VaR assumes zero mean.

## Limitations and known issues
- **No carry or forward points.** Rolled spot positions accrue the interest differential, which matters for carry trades such as short USDMXN or USDZAR. Neither P&L nor risk includes it.
- **Yahoo's chart API is unofficial.** It can rate-limit or block cloud IPs, including Streamlit Cloud. The app then falls back to ECB fixings (one update per day, so "P&L today" compares fixing to fixing) or to the bundled snapshot. The sidebar always shows which source is in use.
- **√h scaling** ignores volatility clustering and autocorrelation. A 10-day figure is a rough guide only.
- **Equal-weight HS reacts slowly** to volatility regime changes. EWMA reacts quickly but assumes normality. Filtered HS would combine the strengths of both (see below).
- **HS component VaR is an estimate.** It depends on a handful of tail scenarios, so it can move noticeably when one day enters or leaves the window. Component ES is more stable and is shown alongside.
- **The backtest uses hypothetical P&L.** It tests today's book over history, which validates the model, not past trading.
- **Best-hedge suggestions can be proxy hedges.** A suggestion such as selling SGD against a EUR long works only through correlation and carries basis risk.
- **Coverage is 12 currencies plus USD.** Uploaded trades in other currencies are rejected with a message. Trade dates have no time of day.
- **No persistence.** Uploaded portfolios and what-if trades last only for the browser session.

## What I would add with more time
1. **More products:** FX forwards and NDFs (forward points, discounting, carry P&L), then vanilla options with delta, gamma and vega, and full-revaluation VaR.
2. **Filtered historical simulation:** scale historical returns by current EWMA volatility. Add stressed VaR on a fixed crisis window and Monte Carlo VaR.
3. **P&L attribution:** split each day's P&L into spot move, carry and new trades, by book and by currency.
4. **FX Style Analytics:** get style exposure of current book and monitor how styles are performing.
5. **Data and trade capture:** a database (Postgres or SQLite) for trades with an audit trail, end-of-day risk snapshots for trend charts, a licensed real-time feed (Refinitiv or Bloomberg) and a check that compares the two sources.
6. **Limits and alerts:** per-book VaR and stop-loss limits, notional limits per currency, and email or Slack alerts on breaches.
7. **More analytics:** factor or PCA views (USD factor, risk-on/risk-off), liquidity-adjusted VaR using bid/ask and market depth, and an optimiser for hedges using several currencies under cost constraints.
8. **Engineering:** login with per-user books, background price refresh, and CI running the test suite.

## Project layout
```
app.py                       Streamlit dashboard (UI only; all maths lives in fxrisk/)
fxrisk/market.py             data sources, cleaning, snapshot, MarketData container
fxrisk/portfolio.py          CSV loading and validation
fxrisk/risk.py               valuation, VaR/ES, decompositions, hedges, backtest, history
notebooks/fx_risk_walkthrough.ipynb  every dashboard calculation step by step, checked against fxrisk
data/portfolio.csv           sample book (opt-in from the landing page; also the template download)
data/market_snapshot.json    offline market data fallback
scripts/refresh_snapshot.py  refresh the snapshot from live sources
scripts/make_sample_portfolio.py  how the sample book was generated
tests/test_risk.py           hand calculations, Euler additivity, finite-difference marginal VaR,
                             exact-revaluation check, hedge optimality, Kupiec, input validation
tests/test_portfolios.py     edge-case portfolio files: one trade, fully hedged, inverse pairs,
tests/portfolios/            crosses vs legs, malformed and invalid files (see its README)
```
