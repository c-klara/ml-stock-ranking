"""
Streamlit front end (week-1 skeleton).

Shows that the pipeline works end to end:
database -> features -> chart. Later weeks add model pages.

Run from the project root:  streamlit run app/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the src package importable when Streamlit runs this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DB_PATH  # noqa: E402
from src.database import read_table  # noqa: E402

st.set_page_config(page_title="ML Stock Ranking", layout="wide")
st.title("ML Stock Ranking")
st.caption("Week-1 skeleton: data pipeline and first feature")

# Stop with a helpful message if the data scripts have not been run yet
if not DB_PATH.exists():
    st.error("No database found. Run `python -m src.fetch_data` "
             "and `python -m src.features` first.")
    st.stop()


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prices and features once and keep them in memory."""
    prices = read_table("prices", parse_dates=["date"])
    features = read_table("features", parse_dates=["month"])
    return prices, features


prices, features = load_data()

# --- Section 1: price history of one stock -------------------------------
st.header("Price history")
tickers = sorted(prices["ticker"].unique())
ticker = st.selectbox("Select a stock", tickers)
stock = prices[prices["ticker"] == ticker].set_index("date")["close"]
st.line_chart(stock)

# --- Section 2: first evidence — does momentum predict next month? --------
st.header("Does momentum predict next month's relative return?")
st.write(
    "Each month, stocks are sorted into ten groups by 12-month momentum. "
    "The chart shows each group's average next-month return relative to "
    "the average stock. A rising pattern means momentum carries a signal."
)

data = features.dropna(subset=["target"]).copy()
data["decile"] = data.groupby("month")["mom_12_1"].transform(
    lambda x: pd.qcut(x, 10, labels=False, duplicates="drop") + 1
)
by_decile = data.groupby("decile")["target"].mean() * 100  # in percent
by_decile.index = by_decile.index.astype(int).astype(str)
st.bar_chart(by_decile, x_label="Momentum decile (1 = lowest)",
             y_label="Avg. next-month excess return (%)")

# --- Section 3: monthly ranking ------------------------------------------
st.header("Momentum ranking by month")
months = sorted(features["month"].unique(), reverse=True)
month = st.selectbox("Select a month", months,
                     format_func=lambda m: pd.Timestamp(m).strftime("%Y-%m"))
ranking = (features[features["month"] == month]
           .sort_values("mom_12_1", ascending=False)
           [["ticker", "mom_12_1", "rev_1m"]])
left, right = st.columns(2)
left.subheader("Top 10")
left.dataframe(ranking.head(10), hide_index=True)
right.subheader("Bottom 10")
right.dataframe(ranking.tail(10), hide_index=True)
