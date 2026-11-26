"""
Model-quality page (owner: Andri).

Shows how reliable each model's forecasts are:
1. a scorecard table with four measures per model, compared with
   simply guessing 0% every month, and
2. a chart of forecast vs. actual return for one stock the user picks.

Data source: the `predictions` table (written by src/train.py).
Until that table exists, the page builds clearly labelled demo data
so the layout can be developed and tested.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make the src package importable when Streamlit runs this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import DB_PATH  # noqa: E402
from src.database import read_table  # noqa: E402

# Expected columns of the predictions table. Agree on these with Raphael
# (src/train.py); if his names differ, only this mapping needs to change.
COL_MONTH = "month"          # month the forecast is made for
COL_TICKER = "ticker"        # stock symbol
COL_MODEL = "model"          # model name, e.g. "Random forest"
COL_PRED = "prediction"      # forecast of next-month excess return
COL_ACTUAL = "actual"        # realised next-month excess return

DEMO_MODELS = ["Linear regression", "Random forest", "Gradient boosting"]


def table_exists(name: str) -> bool:
    """Return True if the given table exists in the project database."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
    return row is not None


def make_demo_predictions() -> pd.DataFrame:
    """
    Build fake forecasts for 2020-2025 from the real targets.

    Each demo model predicts a small share of the true return plus noise,
    which mimics the weak signal real models usually find. Only for
    building the page; the numbers mean nothing.
    """
    features = read_table("features", parse_dates=["month"])
    data = features.dropna(subset=["target"])
    data = data[data["month"] >= "2020-01-01"]

    rng = np.random.default_rng(seed=42)  # fixed seed: same demo every run
    frames = []
    for i, model in enumerate(DEMO_MODELS):
        signal = 0.05 * (i + 1)  # each demo model is a bit "better"
        noise = rng.normal(0, data["target"].std() * 0.3, len(data))
        frames.append(pd.DataFrame({
            COL_MONTH: data["month"].values,
            COL_TICKER: data["ticker"].values,
            COL_MODEL: model,
            COL_PRED: signal * data["target"].values + noise,
            COL_ACTUAL: data["target"].values,
        }))
    return pd.concat(frames, ignore_index=True)


@st.cache_data
def load_predictions() -> tuple[pd.DataFrame, bool]:
    """Load real predictions if available, otherwise demo data. Returns (data, is_demo)."""
    if table_exists("predictions"):
        return read_table("predictions", parse_dates=[COL_MONTH]), False
    return make_demo_predictions(), True


def scorecard(pred: pd.Series, actual: pd.Series) -> dict:
    """
    Compute the four quality measures for one set of forecasts.

    - MAE: average absolute error (in percentage points).
    - RMSE: like MAE, but punishes large misses more.
    - Out-of-sample R²: share of error removed compared with always
      guessing 0%. Above 0 means the model beats the 0% guess.
    - Hit rate: how often the forecast has the right sign (up or down).

    Placeholder until Cornelius's src/evaluate.py exists; then this
    page should call his function instead, so the numbers match.
    """
    error = actual - pred
    return {
        "MAE (%)": error.abs().mean() * 100,
        "RMSE (%)": np.sqrt((error ** 2).mean()) * 100,
        "Out-of-sample R²": 1 - (error ** 2).sum() / (actual ** 2).sum(),
        "Hit rate (%)": (np.sign(pred) == np.sign(actual)).mean() * 100,
    }


# --- Page layout ------------------------------------------------------------
st.set_page_config(page_title="Model quality", layout="wide")
st.title("Model quality")
st.write(
    "How reliable are the forecasts? Every model was tested on 2020–2025, "
    "a period it never saw during training."
)

if not DB_PATH.exists():
    st.error("No database found. Run `python -m src.fetch_data` "
             "and `python -m src.features` first.")
    st.stop()

preds, is_demo = load_predictions()
if is_demo:
    st.warning("Demo data: the real forecasts (table `predictions`) "
               "do not exist yet. These numbers are fake.")

# --- Section 1: scorecard ---------------------------------------------------
st.header("Scorecard")
rows = []
for model, group in preds.groupby(COL_MODEL):
    rows.append({"Model": model, **scorecard(group[COL_PRED], group[COL_ACTUAL])})

# Benchmark: always forecast 0%. Its R² is 0 by definition; a hit rate
# is not meaningful because 0% has no direction.
actual_all = preds.drop_duplicates([COL_MONTH, COL_TICKER])[COL_ACTUAL]
baseline = scorecard(pd.Series(0.0, index=actual_all.index), actual_all)
baseline["Hit rate (%)"] = np.nan
rows.append({"Model": "Always 0% (benchmark)", **baseline})

table = pd.DataFrame(rows).set_index("Model")
st.dataframe(table.style.format("{:.2f}", na_rep="–"))
st.caption("Lower MAE/RMSE is better. R² above 0 means the model beats "
           "guessing 0%. A hit rate above 50% means the direction is "
           "right more often than not.")

# --- Section 2: forecast vs. actual for one stock ---------------------------
st.header("Forecast vs. actual return")
left, right = st.columns(2)
model = left.selectbox("Model", sorted(preds[COL_MODEL].unique()))
ticker = right.selectbox("Stock", sorted(preds[COL_TICKER].unique()))

stock = preds[(preds[COL_MODEL] == model) & (preds[COL_TICKER] == ticker)]
chart = (stock.set_index(COL_MONTH)[[COL_PRED, COL_ACTUAL]]
         .sort_index()
         .rename(columns={COL_PRED: "Forecast", COL_ACTUAL: "Actual"}) * 100)
st.line_chart(chart, y_label="Next-month excess return (%)",
              color=["#4C9BE8", "#F28E2B"])
st.caption("Each point is one month. The closer the two lines, the "
           "better the forecast for this stock.")
