"""FX portfolio P&L and risk dashboard.

    streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fxrisk import risk as rk
from fxrisk.market import USD_QUOTED, load_market
from fxrisk.portfolio import REQUIRED, SAMPLE, check_against_market, load_portfolio

st.set_page_config(page_title="FX Portfolio Risk", layout="wide", initial_sidebar_state="expanded")

GAIN, LOSS, ACCENT, MUTED, GRID = "#13795b", "#c2362b", "#2453c4", "#8a93a6", "rgba(138,147,166,0.22)"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
  [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums; font-size: 1.55rem;}
  [data-testid="stMetricLabel"] p {text-transform: uppercase; letter-spacing: .06em; font-size: .72rem;}
  .stTabs [data-baseweb="tab-list"] {gap: .25rem; flex-wrap: wrap;}
  .zone {display:inline-block; padding:.15rem .6rem; border-radius:999px; font-weight:600; font-size:.8rem;}
  .zone.green {background:rgba(19,121,91,.14); color:#13795b;}
  .zone.yellow {background:rgba(214,150,0,.16); color:#a06c00;}
  .zone.red {background:rgba(194,54,43,.14); color:#c2362b;}
</style>""", unsafe_allow_html=True)


# ---------- formatting helpers ----------

def usd(v: float, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    s = f"{abs(v):,.0f}"
    sign = "-" if v < 0 else ("+" if signed and v > 0 else "")
    return f"{sign}${s}"


def md(text: str) -> str:
    """Escape $ so Streamlit markdown doesn't treat dollar amounts as LaTeX."""
    return text.replace("$", r"\$")


def mm(v: float) -> str:
    return f"{v / 1e6:+,.1f}mm" if v else "0"


def pnl_color(v):
    if isinstance(v, (int, float)) and not np.isnan(v):
        return f"color: {GAIN}" if v > 0 else f"color: {LOSS}" if v < 0 else ""
    return ""


def styled(df: pd.DataFrame, fmt: dict, color_cols=()):
    s = df.style.format(fmt, na_rep="–")
    if color_cols:
        s = s.map(pnl_color, subset=list(color_cols))
    return s


def market_pair(c: str) -> str:
    return f"{c}USD" if c in USD_QUOTED else f"USD{c}"


def price_fmt(pair: str) -> str:
    return "{:,.3f}" if "JPY" in pair else "{:,.4f}" if any(c in pair for c in ("MXN", "ZAR", "NOK", "SEK")) else "{:,.5f}"


def fig_layout(fig: go.Figure, height=340, **kw) -> go.Figure:
    layout = dict(height=height, margin=dict(l=8, r=8, t=44, b=8), paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)", font=dict(size=12), hoverlabel=dict(font_size=12),
                  title_x=0, title_y=0.98, title_yanchor="top",
                  legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1))
    fig.update_layout(**{**layout, **kw})
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=MUTED)
    return fig


METHODS = {"Historical simulation": "hs", "Parametric (normal)": "normal", "Parametric (EWMA)": "ewma"}
METHOD_SHORT = {"hs": "historical", "normal": "normal", "ewma": "EWMA"}


def pct_label(x: float) -> str:
    return f"{x:.1%}".replace(".0%", "%")


# ---------- data ----------

@st.cache_data(ttl=900, show_spinner="Loading market data…")
def get_market(source: str):
    return load_market(source)


with st.sidebar:
    st.header("Market data")
    src_label = st.selectbox("Source", ["Live (Yahoo, then ECB)", "Yahoo Finance", "ECB reference rates", "Bundled snapshot"],
                             key="src")
    source = {"Live (Yahoo, then ECB)": "auto", "Yahoo Finance": "yahoo", "ECB reference rates": "ecb",
              "Bundled snapshot": "snapshot"}[src_label]
    if st.button("Refresh prices", use_container_width=True):
        get_market.clear()

mkt, mkt_warnings = get_market(source)

with st.sidebar:
    st.caption(f"{mkt.source}  \nPrices as of **{mkt.live_time:%d %b %Y %H:%M} UTC** · "
               f"previous close {mkt.prev_date} · {len(mkt.hist)} daily closes from {mkt.hist.index[0]}")
    for w in mkt_warnings:
        st.warning(w, icon="⚠️")

    st.header("Portfolio")
    upload = st.file_uploader("Upload a trade file (CSV)", type="csv",
                              help="Columns: " + ", ".join(REQUIRED) + ", book (optional). Notional is in the base currency.")
    book_all, load_errors = load_portfolio(upload if upload else SAMPLE, mkt.currencies)
    book_all, mkt_errors, mkt_warnings = check_against_market(book_all, mkt)
    load_errors += mkt_errors
    st.caption(f"{'Uploaded file' if upload else 'Sample book (data/portfolio.csv)'} · {len(book_all)} trades loaded")
    if load_errors:
        with st.expander(f"{len(load_errors)} row(s) rejected", expanded=True):
            for e in load_errors:
                st.error(e)
    if mkt_warnings:
        with st.expander(f"{len(mkt_warnings)} off-market price(s)", expanded=False):
            for w in mkt_warnings:
                st.warning(w)
    books = sorted(book_all.book.unique()) if len(book_all) else []
    sel_books = st.multiselect("Books", books, default=books, key="books")

    st.header("Risk settings")
    method_label = st.radio("VaR method", list(METHODS), key="method")
    method = METHODS[method_label]
    c1, c2 = st.columns(2)
    conf = c1.selectbox("Confidence", [0.95, 0.975, 0.99], index=2, format_func=pct_label, key="conf")
    horizon = c2.selectbox("Horizon (days)", [1, 5, 10], index=0, key="horizon")
    window = st.select_slider("Look-back (trading days)", [250, 500, 750], value=500, key="window")
    lam = st.slider("EWMA decay λ", 0.90, 0.99, 0.94, 0.01, key="lam") if method == "ewma" else 0.94

    st.header("VaR limit")
    limit = st.number_input("Limit (USD)", min_value=0, value=1_000_000, step=50_000, key="limit")
    with st.expander("Measured on", expanded=False):
        st.caption("A limit is set on one VaR measure. Limit usage always uses this measure, so changing the "
                   "risk settings above does not change it.")
        lim_method_label = st.selectbox("Method", list(METHODS), index=0, key="lim_method")
        l1, l2 = st.columns(2)
        lim_conf = l1.selectbox("Confidence", [0.95, 0.975, 0.99], index=2, format_func=pct_label, key="lim_conf")
        lim_horizon = l2.selectbox("Horizon (days)", [1, 5, 10], index=0, key="lim_horizon")
        lim_window = st.select_slider("Look-back (trading days)", [250, 500, 750], value=500, key="lim_window")

cfg = rk.RiskConfig(method, conf, horizon, window, lam)
lim_cfg = rk.RiskConfig(METHODS[lim_method_label], lim_conf, lim_horizon, lim_window, 0.94)
lim_label = f"{pct_label(lim_conf)} {lim_horizon}-day {METHOD_SHORT[lim_cfg.method]} VaR"
book = book_all[book_all.book.isin(sel_books)].reset_index(drop=True) if len(book_all) else book_all
conf_s = pct_label(conf)
var_label = f"{conf_s} {horizon}d"

st.title("FX Portfolio Risk")
if book.empty:
    st.info("No trades to show. Upload a trade file or select at least one book in the sidebar.")
    st.stop()

pos, X = rk.value_positions(book, mkt)
A = rk.analyse(pos, X, mkt, cfg)
R_pos, R_ccy, exposure = A["pos"], A["ccy"], A["exposure"]
items = R_pos["items"]
# The limit is always measured on its own fixed definition, not the analysis settings.
limit_var = R_pos["var"] if lim_cfg == cfg else rk.risk_report(X, mkt.returns, lim_cfg)["var"]

st.caption(f"{len(pos)} trades across {pos.book.nunique()} book(s) · {method_label}, {var_label}, "
           f"{R_pos['window_start']} → {R_pos['window_end']} ({window} scenarios)")

# ---------- headline ----------

k = st.columns(6)
k[0].metric("P&L since inception", usd(pos.pnl_itd.sum(), True))
k[1].metric("P&L today", usd(pos.pnl_day.sum(), True), help=f"Live prices vs the {mkt.prev_date} close. "
            "Trades booked since that close count their full P&L.")
k[2].metric(f"VaR {var_label}", usd(R_pos["var"]))
k[3].metric(f"Expected shortfall", usd(R_pos["es"]), help="Average loss beyond VaR at the same confidence.")
util = limit_var / limit if limit else np.nan
k[4].metric("VaR limit used", f"{util:.0%}" if limit else "–",
            delta=(f"{usd(limit - limit_var)} headroom" if util <= 1 else f"{usd(limit_var - limit)} over") if limit else None,
            delta_color="normal" if util <= 1 else "inverse",
            help=md(f"{usd(limit_var)} of {lim_label} against a {usd(limit)} limit. The limit measure is set under "
                    "VaR limit in the sidebar and does not follow the risk settings."))
k[5].metric("Gross notional", f"${pos.usd_notional.sum() / 1e6:,.0f}mm",
            help="Sum of absolute base-currency notionals in USD.")
if limit:
    st.caption(md(f"Limit: {usd(limit)} on {lim_label} ({lim_cfg.window}-day look-back). "
                  f"Current {lim_label}: {usd(limit_var)}."))
if limit and util > 1:
    st.error(md(f"{lim_label} of {usd(limit_var)} is above the {usd(limit)} limit. "
                "The Exposure & hedging tab lists the single trades that cut risk most."))
elif limit and util > 0.85:
    st.warning(md(f"{lim_label} is at {util:.0%} of the {usd(limit)} limit."))

tabs = st.tabs(["Positions", "Risk decomposition", "Exposure & hedging", "Stress tests", "VaR backtest",
                "P&L history", "What-if trade", "Market"])

# ---------- positions ----------

with tabs[0]:
    tbl = pd.DataFrame({
        "Date": pos.trade_date, "Book": pos.book, "Pair": pos.currency_pair, "Side": pos.side,
        "Notional": pos.notional, "Entry": pos.entry_price, "Spot": pos.spot, "Move": pos.move_pct,
        "USD notional": pos.usd_notional, "P&L today": pos.pnl_day, "P&L since inception": pos.pnl_itd,
        "Standalone VaR": items.standalone, "Component VaR": items.component, "% of VaR": items.pct_of_var,
    })
    fmt = {"Notional": "{:,.0f}", "USD notional": "{:,.0f}", "P&L today": "{:+,.0f}", "P&L since inception": "{:+,.0f}",
           "Standalone VaR": "{:,.0f}", "Component VaR": "{:,.0f}", "% of VaR": "{:.1%}", "Move": "{:+.2%}"}
    s = styled(tbl, fmt, ["P&L today", "P&L since inception", "Move"])
    for col in ("Entry", "Spot"):
        s = s.format({col: lambda v, _c=col: f"{v:,.5g}" if v < 10 else f"{v:,.3f}"}, subset=[col])
    st.dataframe(s, use_container_width=True, height=min(38 * (len(tbl) + 1) + 4, 640))
    st.caption("Move is the spot change since entry in the trade's favour. Notional is in the base currency. "
               "P&L is in USD, with quote-currency P&L converted at the live rate.")

    st.subheader("By book")
    g = pos.assign(component=items.component).groupby("book")
    by_book = pd.DataFrame({"Trades": g.size(), "USD notional": g.usd_notional.sum(), "P&L today": g.pnl_day.sum(),
                            "P&L since inception": g.pnl_itd.sum(), "Component VaR": g.component.sum()})
    by_book["Standalone VaR"] = [rk.risk_report(X.loc[pos.book == b], mkt.returns, cfg)["var"] for b in by_book.index]
    by_book["% of VaR"] = by_book["Component VaR"] / R_pos["var"]
    st.dataframe(styled(by_book, {"USD notional": "{:,.0f}", "P&L today": "{:+,.0f}", "P&L since inception": "{:+,.0f}",
                                  "Component VaR": "{:,.0f}", "Standalone VaR": "{:,.0f}", "% of VaR": "{:.1%}"},
                        ["P&L today", "P&L since inception"]), use_container_width=True)

# ---------- risk decomposition ----------

with tabs[1]:
    diversification = items.standalone.sum() - R_pos["var"]
    c = st.columns(4)
    c[0].metric(f"Portfolio VaR", usd(R_pos["var"]))
    c[1].metric("Sum of standalone VaRs", usd(items.standalone.sum()))
    c[2].metric("Diversification benefit", usd(diversification),
                f"{diversification / items.standalone.sum():.0%} of undiversified" if items.standalone.sum() else None,
                delta_color="off")
    c[3].metric("VaR scenario date" if method == "hs" else "Portfolio daily σ",
                R_pos["var_scenario_date"] if method == "hs" else usd(R_pos["sigma"]),
                help="The historical day whose P&L sets HS VaR." if method == "hs" else "Standard deviation of 1-day P&L.")

    left, right = st.columns(2)
    with left:
        srt = items.component.sort_values()
        lbl = [f"{t} {pos.loc[t, 'currency_pair']} {pos.loc[t, 'side'][0]}" for t in srt.index]
        fig = go.Figure(go.Bar(x=srt.values, y=lbl, orientation="h", marker_color=[LOSS if v > 0 else GAIN for v in srt],
                               hovertemplate="%{y}<br>Component VaR $%{x:,.0f}<extra></extra>"))
        st.plotly_chart(fig_layout(fig, 460, title=f"Component VaR by trade ({var_label})"), use_container_width=True)
        st.caption("Bars to the right add risk. Bars to the left offset other trades and reduce portfolio VaR.")
    with right:
        cc = R_ccy["items"].component.sort_values()
        cc = cc[cc.abs() > 1]
        fig = go.Figure(go.Bar(x=cc.values, y=cc.index, orientation="h", marker_color=[LOSS if v > 0 else GAIN for v in cc],
                               hovertemplate="%{y}<br>Component VaR $%{x:,.0f}<extra></extra>"))
        st.plotly_chart(fig_layout(fig, 460, title="Component VaR by currency (vs USD)"), use_container_width=True)
        st.caption("The same VaR split by risk factor. Both views add up to portfolio VaR.")

    st.subheader("Position risk")
    rt = pd.DataFrame({
        "Pair": pos.currency_pair, "Side": pos.side, "USD notional": pos.usd_notional,
        "Standalone VaR": items.standalone, "Component VaR": items.component, "% of VaR": items.pct_of_var,
        "Marginal VaR per $1mm": items.marginal_per_mm, "Incremental VaR": items.incremental,
        "Component ES": items.component_es,
    })
    st.dataframe(styled(rt, {"USD notional": "{:,.0f}", "Standalone VaR": "{:,.0f}", "Component VaR": "{:+,.0f}",
                             "% of VaR": "{:.1%}", "Marginal VaR per $1mm": "{:+,.0f}", "Incremental VaR": "{:+,.0f}",
                             "Component ES": "{:+,.0f}"}), use_container_width=True,
                 height=min(38 * (len(rt) + 1) + 4, 640))
    with st.expander("How to read these columns"):
        st.markdown(f"""
- **Standalone VaR**: the trade's VaR on its own, ignoring the rest of the book.
- **Component VaR**: the trade's share of portfolio VaR (Euler allocation). The column sums to portfolio VaR. A negative value means the trade hedges the book.
- **Marginal VaR per $1mm**: the change in portfolio VaR if you add $1mm USD notional to the trade in the same direction. It equals component VaR ÷ USD notional × 1mm.
- **Incremental VaR**: portfolio VaR minus the VaR of the book without this trade. This is the exact effect of closing the trade, not a linear estimate.
- **Component ES**: the trade's average P&L across the {R_pos['k']} worst historical scenarios. The column sums exactly to historical-simulation ES.
""")

    left, right = st.columns([3, 2])
    with left:
        p = R_pos["pnl"] * cfg.scale
        fig = go.Figure(go.Histogram(x=p.values, nbinsx=60, marker_color=ACCENT, opacity=0.75,
                                     hovertemplate="P&L $%{x:,.0f}<br>%{y} days<extra></extra>", name="Scenario P&L", showlegend=False))
        top = np.histogram(p.values, bins=60)[0].max() * 1.08
        for x, name, color, dash in [(-R_pos["hs_var"], f"HS VaR {usd(R_pos['hs_var'])}", LOSS, "solid"),
                                     (-R_pos["hs_es"], f"HS ES {usd(R_pos['hs_es'])}", "#7a1f17", "dot"),
                                     (-R_pos["normal_var"], f"Normal VaR {usd(R_pos['normal_var'])}", "#c98500", "dash")]:
            fig.add_scatter(x=[x, x], y=[0, top], mode="lines", name=name, line=dict(color=color, dash=dash, width=2),
                            hovertemplate=f"{name}<extra></extra>")
        fig.add_annotation(x=0, y=1.0, xref="paper", yref="paper", text="← losses", showarrow=False,
                           xanchor="left", yanchor="top", font=dict(size=11, color=MUTED))
        fig.update_xaxes(title="Scenario P&L (USD)")
        fig.update_yaxes(title="Days")
        st.plotly_chart(fig_layout(fig, 380, title=f"Today's book revalued over {window} historical days"
                                   + (f" (×√{horizon})" if horizon > 1 else ""), showlegend=True,
                                   legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.99,
                                               bgcolor="rgba(255,255,255,0.75)")), use_container_width=True)
    with right:
        st.markdown(f"**VaR by method** ({var_label})")
        cmp = rk.compare_methods(X, mkt, cfg)
        st.dataframe(styled(cmp, {"VaR": "{:,.0f}", "Expected shortfall": "{:,.0f}"}), use_container_width=True)
        p1 = R_pos["pnl"]
        st.caption(f"Scenario P&L skew {p1.skew():.2f}, excess kurtosis {p1.kurt():.2f}. "
                   "Fat tails push historical VaR above the normal estimate. EWMA weights recent volatility more heavily.")

# ---------- exposure & hedging ----------

with tabs[2]:
    left, right = st.columns([3, 2])
    ex = exposure[exposure.abs() > 1].sort_values()
    with left:
        fig = go.Figure(go.Bar(x=ex.values / 1e6, y=ex.index, orientation="h", marker_color=[GAIN if v > 0 else LOSS for v in ex],
                               text=[mm(v) for v in ex.values], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:,.1f}mm USD<extra></extra>"))
        pad = max(ex.abs().max() / 1e6 * 0.25, 1)
        fig.update_xaxes(title="USD mm (long + / short −)", range=[min(ex.min() / 1e6, 0) - pad, max(ex.max() / 1e6, 0) + pad])
        st.plotly_chart(fig_layout(fig, 420, title="Net currency exposure vs USD", showlegend=False), use_container_width=True)
    with right:
        vol = mkt.returns.iloc[-window:].std() * np.sqrt(252)
        et = pd.DataFrame({"Net USD exposure": exposure, "Ann. vol": vol, "Standalone VaR": R_ccy["items"].standalone,
                           "Component VaR": R_ccy["items"].component})
        et.loc["USD (implied)"] = [-exposure.sum(), 0.0, 0.0, 0.0]
        st.dataframe(styled(et, {"Net USD exposure": "{:+,.0f}", "Ann. vol": "{:.1%}", "Standalone VaR": "{:,.0f}",
                                 "Component VaR": "{:+,.0f}"}), use_container_width=True, height=490)
        st.caption("USD (implied) is the net USD cash position that balances the currency legs.")

    st.subheader("Best single-trade hedges")
    hedges = rk.best_hedges(R_pos, exposure, mkt, cfg)
    ht = pd.DataFrame({
        "Trade": [f"{'Buy' if r.hedge_usd > 0 else 'Sell'} {abs(r.hedge_ccy) / 1e6:,.1f}mm {c} vs USD" for c, r in hedges.iterrows()],
        "USD amount": hedges.hedge_usd, "VaR after": hedges.var_after, "VaR reduction": hedges.reduction,
    }, index=hedges.index)
    st.dataframe(styled(ht.head(6), {"USD amount": "{:+,.0f}", "VaR after": "{:,.0f}", "VaR reduction": "{:.1%}"}),
                 use_container_width=True)
    st.caption(md(f"Each row is the trade in one currency against USD that most reduces portfolio variance: "
               f"−(Σe)ᵢ / Σᵢᵢ under the {'EWMA' if method == 'ewma' else 'equal-weight'} covariance. "
               f"VaR shown is parametric ({usd(R_pos['normal_var'])} before hedging). Proxy hedges such as SGD for "
               "EUR work through correlation, so check them against the stress tests."))

# ---------- stress ----------

with tabs[3]:
    summary, by_trade = rk.stress_tests(X, mkt)
    srt = summary["P&L"].sort_values()
    fig = go.Figure(go.Bar(x=srt.values, y=srt.index, orientation="h", marker_color=[GAIN if v > 0 else LOSS for v in srt],
                           hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>"))
    fig.add_vline(x=-R_pos["var"], line_dash="dot", line_color=MUTED)
    fig.add_annotation(x=-R_pos["var"], y=1.02, yref="paper", text=f"−VaR {var_label}", showarrow=False,
                       font=dict(size=11, color=MUTED))
    fig.update_yaxes(tickfont=dict(size=12))
    st.plotly_chart(fig_layout(fig, 360, title="Scenario P&L, today's book", showlegend=False), use_container_width=True)
    st.dataframe(styled(summary[["Type", "Period", "P&L", "Shocks"]], {"P&L": "{:+,.0f}"}, ["P&L"]),
                 use_container_width=True)
    st.caption("Historical episodes apply each currency's actual move between the two closes. "
               "Hypothetical shocks move USD per unit of each currency. Revaluation is exact for spot.")
    pick = st.selectbox("Breakdown by trade", summary.index, key="stress_pick")
    bt_df = pd.DataFrame({"Pair": pos.currency_pair, "Side": pos.side, "Book": pos.book, "Scenario P&L": by_trade[pick]})
    st.dataframe(styled(bt_df.sort_values("Scenario P&L"), {"Scenario P&L": "{:+,.0f}"}, ["Scenario P&L"]),
                 use_container_width=True)

# ---------- backtest ----------

with tabs[4]:
    bt = rk.backtest(exposure, mkt, rk.RiskConfig(method, conf, 1, window, lam), 250)
    if bt["n"] < 20:
        st.info(f"A {window}-day look-back leaves only {bt['n']} days to test. Choose a shorter look-back.")
    else:
        s = bt["series"]
        c = st.columns(4)
        c[0].metric("Days tested", bt["n"])
        c[1].metric("Exceptions", f"{bt['exceptions']}", f"{bt['expected']:.1f} expected", delta_color="off")
        c[2].metric("Kupiec p-value", f"{bt['p_value']:.2f}", help="Probability of seeing this many exceptions if the "
                    "model is correct. Below 0.05 means the model is mis-calibrated.")
        c[3].markdown(f"<div style='padding-top:.4rem'>Basel zone</div><span class='zone {bt['zone']}'>{bt['zone'].title()}</span>",
                      unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_bar(x=s.index, y=s.pnl, marker_color=np.where(s.breach, LOSS, "rgba(138,147,166,0.55)"),
                    name="Hypothetical P&L", hovertemplate="%{x}<br>P&L $%{y:,.0f}<extra></extra>")
        fig.add_scatter(x=s.index, y=-s["var"], mode="lines", line=dict(color=ACCENT, width=2), name=f"−VaR {conf_s} 1d",
                        hovertemplate="%{x}<br>−VaR $%{y:,.0f}<extra></extra>")
        st.plotly_chart(fig_layout(fig, 380, title="Walk-forward backtest: each day's P&L vs VaR from the prior window only"), use_container_width=True)
        st.caption("Each day's VaR uses only data available before that day. P&L is what today's positions would "
                   "have made: a test of the model, not of past trading.")

# ---------- history ----------

with tabs[5]:
    h = rk.pnl_history(book, mkt)
    d = h.diff().fillna(h.iloc[0])
    fig = go.Figure()
    fig.add_scatter(x=h.index, y=h.values, mode="lines", line=dict(color=ACCENT, width=2), fill="tozeroy",
                    fillcolor="rgba(36,83,196,0.08)", name="P&L since inception",
                    hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>")
    fig.add_scatter(x=[h.index[-1]], y=[h.iloc[-1]], mode="markers+text", marker=dict(size=9, color=ACCENT),
                    text=[f"Live {usd(h.iloc[-1], True)}"], textposition="top left", showlegend=False, hoverinfo="skip")
    st.plotly_chart(fig_layout(fig, 340, title="Book P&L since first trade (USD)", showlegend=False), use_container_width=True)
    fig = go.Figure(go.Bar(x=d.index, y=d.values, marker_color=[GAIN if v >= 0 else LOSS for v in d],
                           hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>"))
    st.plotly_chart(fig_layout(fig, 240, title="Daily P&L (USD)"), use_container_width=True)
    c = st.columns(4)
    c[0].metric("Best day", usd(d.max(), True), d.idxmax(), delta_color="off")
    c[1].metric("Worst day", usd(d.min(), True), d.idxmin(), delta_color="off")
    peak = h.cummax()
    c[2].metric("Max drawdown", usd((h - peak).min()), help="Largest fall from a running peak in cumulative P&L.")
    c[3].metric("Hit rate", f"{(d > 0).mean():.0%}", help="Share of days with positive P&L.")
    st.caption("Uses daily closes and the actual trade dates, so the book grows as trades are added. The last point uses live prices.")

# ---------- what-if ----------

with tabs[6]:
    st.markdown("Test a trade against the current book before you execute it. It is priced at the live mid.")
    ccys = ["USD"] + mkt.currencies
    with st.form("whatif"):
        c = st.columns([1, 1, 1, 2, 1])
        base = c[0].selectbox("Base", ccys, index=1, key="wi_base")
        quote = c[1].selectbox("Quote", ccys, index=0, key="wi_quote")
        side = c[2].selectbox("Side", ["BUY", "SELL"], key="wi_side")
        notional = c[3].number_input("Notional (base currency)", min_value=0, value=10_000_000, step=1_000_000, key="wi_notional")
        c[4].markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        go_ = c[4].form_submit_button("Test trade", use_container_width=True)
    if base == quote:
        st.error("Choose two different currencies.")
    elif go_ or "wi_last" in st.session_state:
        st.session_state["wi_last"] = True
        spot = mkt.spot(base + quote)
        new = pd.DataFrame([{"trade_id": "WHAT-IF", "trade_date": mkt.live_time.strftime("%Y-%m-%d"),
                             "currency_pair": base + quote, "side": side, "notional": float(notional), "entry_price": spot,
                             "book": "What-if", "base": base, "quote": quote,
                             "N": (1 if side == "BUY" else -1) * float(notional)}])
        book2 = pd.concat([book, new], ignore_index=True)
        pos2, X2 = rk.value_positions(book2, mkt)
        A2 = rk.analyse(pos2, X2, mkt, cfg)
        st2, _ = rk.stress_tests(X2, mkt)
        st1 = summary["P&L"]
        it = A2["pos"]["items"].loc["WHAT-IF"]
        c = st.columns(4)
        c[0].metric(f"VaR {var_label}", usd(A2["pos"]["var"]), usd(A2["pos"]["var"] - R_pos["var"], True), delta_color="inverse")
        c[1].metric("Expected shortfall", usd(A2["pos"]["es"]), usd(A2["pos"]["es"] - R_pos["es"], True), delta_color="inverse")
        c[2].metric("Worst stress", usd(st2["P&L"].min()), usd(st2["P&L"].min() - st1.min(), True))
        lim2 = rk.risk_report(X2, mkt.returns, lim_cfg)["var"]
        c[3].metric("Limit used", f"{lim2 / limit:.0%}" if limit else "–",
                    f"{(lim2 - limit_var) / limit:+.0%}" if limit else None, delta_color="inverse",
                    help=md(f"Measured on {lim_label}, the limit's own definition."))
        st.caption(md(f"{side.title()} {notional:,.0f} {base} vs {quote} at {spot:,.5g} "
                   f"(${pos2.loc['WHAT-IF', 'usd_notional']:,.0f} USD notional). "
                   f"Standalone VaR {usd(it.standalone)}, component VaR {usd(it.component, True)} "
                   f"({it.pct_of_var:.0%} of the new total)."))
        cmp = pd.DataFrame({"Before": exposure, "After": A2["exposure"]})
        cmp["Change"] = cmp.After - cmp.Before
        cmp = cmp[(cmp.abs() > 1).any(axis=1)]
        left, right = st.columns(2)
        left.markdown("**Net currency exposure (USD)**")
        left.dataframe(styled(cmp, {"Before": "{:+,.0f}", "After": "{:+,.0f}", "Change": "{:+,.0f}"}), use_container_width=True)
        sc = pd.DataFrame({"Before": st1, "After": st2["P&L"]})
        sc["Change"] = sc.After - sc.Before
        right.markdown("**Stress P&L (USD)**")
        right.dataframe(styled(sc, {"Before": "{:+,.0f}", "After": "{:+,.0f}", "Change": "{:+,.0f}"}, ["Change"]),
                        use_container_width=True)

# ---------- market ----------

with tabs[7]:
    rets = mkt.returns
    rows = []
    for c_ in mkt.currencies:
        pair = market_pair(c_)
        inv = c_ not in USD_QUOTED
        f = (lambda v: 1 / v) if inv else (lambda v: v)
        live, prev = f(mkt.live[c_]), f(mkt.hist[c_].iloc[-1])
        m1 = f(mkt.hist[c_].iloc[-22])
        rows.append({"Pair": pair, "Spot": live, "Prev close": prev, "1d": live / prev - 1, "1m": live / m1 - 1,
                     "Vol 3m": rets[c_].iloc[-63:].std() * np.sqrt(252), "Vol 1y": rets[c_].iloc[-252:].std() * np.sqrt(252),
                     "Exposure": exposure[c_]})
    mt = pd.DataFrame(rows).set_index("Pair")
    left, right = st.columns([5, 4])
    with left:
        st.markdown("**Spot rates and realised volatility**")
        s = styled(mt, {"1d": "{:+.2%}", "1m": "{:+.2%}", "Vol 3m": "{:.1%}", "Vol 1y": "{:.1%}", "Exposure": "{:+,.0f}"})
        for col in ("Spot", "Prev close"):
            s = s.format({col: lambda v: f"{v:,.5g}" if v < 10 else f"{v:,.3f}"}, subset=[col])
        st.dataframe(s, use_container_width=True, height=490)
        st.caption("Moves are in market quoting convention. Vol is annualised from daily returns. Exposure is net USD-equivalent.")
    with right:
        corr = rk.correlation(R_pos["S"], mkt.currencies)
        fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                   colorscale=[[0, "#c2362b"], [0.5, "#f0efec"], [1, "#2a78d6"]],
                                   text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(size=9),
                                   hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>", colorbar=dict(thickness=10)))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_layout(fig, 470, title=f"Correlation of USD returns ({'EWMA' if method == 'ewma' else f'{window}d'})"),
                        use_container_width=True)

with st.expander("Methodology"):
    st.markdown(f"""
**Valuation.** Each trade is two cash legs: +N base and −N×entry quote. Its USD value is
N·u(base) − N·K·u(quote), where u(c) is USD per unit of c. This equals N·(S − K) converted to USD at the live quote rate.
Daily P&L is the change in that value since the {mkt.prev_date} close.

**Risk factors.** Daily returns of each currency against USD. Crosses such as EURGBP split into their EUR and GBP legs, so
offsetting exposures net across trades. For spot, scenario P&L = Σ exposure × return is exact. It is not a delta approximation.

**VaR.** Historical simulation revalues today's book over the last {window} daily moves. VaR is the {R_pos['k']}th worst
outcome and ES is the mean of the {R_pos['k']} worst. Parametric VaR = z·√(e′Σe) with a zero-mean covariance, either
equal-weight or EWMA (λ = {lam}). Multi-day horizons scale by √h.

**Decomposition.** Component VaR uses Euler allocation. Parametric: z·xᵢ′Σe/σ. Historical: each trade's average P&L in
scenarios ranked near the VaR scenario, rescaled to sum to VaR. Marginal VaR = component ÷ notional.
Incremental VaR is a full with/without recalculation.
""")
