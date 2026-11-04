"""
Step 1 of the pipeline: download raw data and store it in SQLite.

Sources
- Daily adjusted prices and volumes: Yahoo Finance via the yfinance API
- Daily Fama-French factors (market, size, value, risk-free rate):
  Kenneth French's data library

Output tables
- prices  (ticker, date, close, volume)
- factors (date, mkt_rf, smb, hml, rf)

Run from the project root:  python -m src.fetch_data
"""

import io
import zipfile

import pandas as pd
import requests
import yfinance as yf

from src.config import (
    BATCH_SIZE,
    END_DATE,
    FF_FACTORS_URL,
    N_STOCKS,
    SP500_URL,
    START_DATE,
)
from src.database import write_table

# A plain browser-style header: Wikipedia refuses requests without one
HEADERS = {"User-Agent": "Mozilla/5.0 (student research project)"}


def get_sp500_tickers() -> list[str]:
    """
    Return the current S&P 500 tickers in Yahoo Finance format.

    Note: using today's constituents introduces survivorship bias
    (firms that dropped out of the index are missing). We accept and
    document this limitation for version 1.
    """
    html = requests.get(SP500_URL, headers=HEADERS, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    # Yahoo writes share classes with a dash (BRK-B), Wikipedia with a dot (BRK.B)
    return table["Symbol"].str.replace(".", "-", regex=False).tolist()


def download_prices(tickers: list[str]) -> pd.DataFrame:
    """
    Download daily adjusted close prices and volumes in batches.

    Returns a long-format table with one row per stock per trading day.
    """
    frames = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        print(f"Downloading tickers {i + 1}-{i + len(batch)} of {len(tickers)}")
        data = yf.download(
            batch,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,  # adjust for splits and dividends
            progress=False,
            threads=True,
        )
        # Reshape from wide (one column per ticker) to long format
        close = data["Close"].stack().rename("close")
        volume = data["Volume"].stack().rename("volume")
        frames.append(pd.concat([close, volume], axis=1))

    prices = pd.concat(frames).reset_index()
    prices.columns = ["date", "ticker", "close", "volume"]
    return prices.dropna(subset=["close"])


def select_universe(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Keep the n most liquid stocks, measured by median daily traded value.

    Limitation: liquidity is measured over the whole sample, which uses
    information from the future. Acceptable for a v1 universe filter;
    disclosed in the app's methodology page.
    """
    traded_value = prices["close"] * prices["volume"]
    liquidity = traded_value.groupby(prices["ticker"]).median()
    keep = liquidity.nlargest(n).index
    return prices[prices["ticker"].isin(keep)]


def download_ff_factors() -> pd.DataFrame:
    """
    Download daily Fama-French three-factor data and convert percent to decimals.

    The file is a zipped CSV with text above and below the data; we keep
    only the lines that start with an 8-digit date (YYYYMMDD).
    """
    response = requests.get(FF_FACTORS_URL, headers=HEADERS, timeout=60)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        raw = zf.read(zf.namelist()[0]).decode("latin-1")

    rows = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5 and parts[0].isdigit() and len(parts[0]) == 8:
            rows.append(parts)

    factors = pd.DataFrame(rows, columns=["date", "mkt_rf", "smb", "hml", "rf"])
    factors["date"] = pd.to_datetime(factors["date"], format="%Y%m%d")
    value_cols = ["mkt_rf", "smb", "hml", "rf"]
    factors[value_cols] = factors[value_cols].astype(float) / 100  # percent -> decimal
    return factors[(factors["date"] >= START_DATE) & (factors["date"] <= END_DATE)]


def main() -> None:
    """Run the full download and store both tables in the database."""
    tickers = get_sp500_tickers()
    prices = select_universe(download_prices(tickers), N_STOCKS)
    write_table(prices, "prices")
    print(f"Stored {len(prices):,} price rows for {prices['ticker'].nunique()} stocks")

    factors = download_ff_factors()
    write_table(factors, "factors")
    print(f"Stored {len(factors):,} factor rows")


if __name__ == "__main__":
    main()
