# Test portfolios

Edge-case trade files, each isolating one behaviour. `tests/test_portfolios.py` checks the expected result for every file. You can also upload any of them in the dashboard sidebar to see the behaviour directly.

Prices are round numbers such as EURUSD 1.10, so most files raise an "off-market" warning when uploaded. The warning is expected: it means the price is a few % from the real close on that date. Trades are still accepted unless the price is more than 25% away.

## Calculation checks

| File | What it contains | Expected result |
|---|---|---|
| `single_trade.csv` | One EURUSD long, 10mm at 1.10 | Portfolio VaR = standalone = component = incremental VaR, with 100% of VaR. P&L = 10mm × (spot − 1.10). Exposure is EUR only. |
| `fully_hedged.csv` | Buy and sell 10mm EURUSD at the same price | P&L, exposure, VaR, component VaR and every stress result are exactly 0, with no NaNs. Each leg still has standalone VaR on its own. |
| `inverse_pair_equivalence.csv` | BUY 10mm EURUSD @ 1.10 and SELL 11mm USDEUR @ 1/1.10 | These are the same trade written two ways. P&L, daily P&L, exposures and standalone VaR are identical, and the two trades split VaR 50/50. |
| `cross_vs_legs.csv` | BUY EURGBP, versus BUY EURUSD plus SELL GBPUSD at matching amounts | The cross equals its two USD legs: same exposures, same P&L, and the same VaR under every method. |
| `opened_today.csv` | Two identical USDJPY trades, one dated before the last close and one after | The new trade's daily P&L equals its inception P&L. The old trade's does not. Both have the same inception P&L. |
| `before_history.csv` | A 2019 trade, older than the 3-year price history | Values normally. The P&L history starts at the first available close. No off-market check is possible, so none is raised. |

`test_scaling_and_sign_flip` uses the sample book rather than a file. It checks two things:
- Doubling every notional doubles VaR.
- Flipping every side leaves parametric VaR unchanged. Under historical simulation it gives the k-th best P&L instead of the k-th worst.

## Parsing and validation checks

| File | What it contains | Expected result |
|---|---|---|
| `messy_but_valid.csv` | A byte-order mark, Windows line endings, upper-case and padded headers, reordered and extra columns, a blank line, the pair formats `EUR/USD`, `usd-jpy` and ` eur gbp `, sides `buy` and ` Sell `, notionals `"10,000,000"` and `5_000_000`, and a blank book | All 3 trades parse with the correct values. The blank book becomes "Unassigned". |
| `invalid_rows.csv` | 1 good row and 12 bad ones: an unsupported currency (TRY), USDUSD, a 7-letter pair, a blank pair, side HOLD, a negative notional, a text notional, a blank price, a DD/MM/YYYY date, 30 February, a blank trade_id, and a duplicate trade_id | Only the good row is kept. Each bad row gets its own message. The duplicate id does not merge exposure into the kept trade. |
| `all_zeros.csv` | Zero notionals and zero prices | Every row is rejected. The dashboard shows "No trades to show". |
| `empty_file.csv` | A 0-byte file | "The file is empty." No crash. |
| `header_only.csv` | Column names but no rows | An empty book with no errors. |
| `missing_column.csv` | No `entry_price` column | "Missing column(s): entry_price." |
| `no_book_column.csv` | No optional `book` column | The trade goes into book "Unassigned". |
| `market_sanity.csv` | A good trade, USDJPY entered inverted (0.00667), EURUSD entered as 110, and a trade dated 2027 | Only the good trade is kept. The inverted price is rejected with a hint that it "may be inverted". The future-dated trade is rejected. |

Run them with:

```bash
python -m pytest tests/test_portfolios.py -v
```
