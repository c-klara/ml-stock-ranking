"""
Step 2 of the pipeline: turn daily prices into monthly stock characteristics.

Every feature for month t uses ONLY information available at the end of
month t. The target is the return in month t+1. Mixing these up
(look-ahead bias) would make any backtest worthless.

Week-1 features
- mom_12_1 : return from month t-12 to t-1 (skips the latest month)
- rev_1m   : return in month t (short-term reversal)

Target
- target   : return in month t+1 minus the average return of all stocks that
             month, so the model learns which stocks beat the others,
             not whether the market as a whole rises

Output table: features (ticker, month, mom_12_1, rev_1m, mom_12_1_rank,
              rev_1m_rank, target)

Run from the project root:  python -m src.features
"""

import pandas as pd

from src.database import read_table, write_table

FEATURES = ["mom_12_1", "rev_1m"]


def monthly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert daily long-format prices into a wide table of month-end prices."""
    wide = prices.pivot(index="date", columns="ticker", values="close")
    return wide.resample("ME").last()


def rank_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Within each month, replace values by their rank scaled to [-1, 1].

    This makes features comparable across months and neutralises outliers;
    it is standard practice in machine-learning asset pricing studies.
    """
    ranks = df.rank(axis=1)
    n = df.notna().sum(axis=1)
    return 2 * (ranks - 1).div(n - 1, axis=0) - 1


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute features and target for every stock and month."""
    mp = monthly_prices(prices)
    returns = mp.pct_change(fill_method=None)

    raw = {
        # Price one month ago divided by price twelve months ago
        "mom_12_1": mp.shift(1) / mp.shift(12) - 1,
        # Return over the most recent month
        "rev_1m": returns,
    }

    # Next month's return, minus the cross-sectional average that month
    next_ret = returns.shift(-1)
    target = next_ret.sub(next_ret.mean(axis=1), axis=0)

    # Assemble one long table: one row per (month, ticker)
    columns = {name: df.stack() for name, df in raw.items()}
    columns.update({f"{name}_rank": rank_normalize(df).stack() for name, df in raw.items()})
    columns["target"] = target.stack()
    table = pd.DataFrame(columns).reset_index()
    table = table.rename(columns={"date": "month"})

    # Drop rows without a complete feature set (e.g. the first 12 months)
    return table.dropna(subset=FEATURES)


def main() -> None:
    """Load prices from the database, build features, and store them."""
    prices = read_table("prices", parse_dates=["date"])
    features = build_features(prices)
    write_table(features, "features")
    print(f"Stored {len(features):,} feature rows "
          f"({features['month'].min():%Y-%m} to {features['month'].max():%Y-%m})")


if __name__ == "__main__":
    main()
