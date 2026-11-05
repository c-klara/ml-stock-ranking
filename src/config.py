"""
Central configuration for the project.

All scripts import their settings from here, so a parameter
(for example the sample period) only ever needs to be changed in one place.
"""

from pathlib import Path

# Project root = the folder that contains src/, app/ and data/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Location of the SQLite database that every component reads from and writes to
DB_PATH = PROJECT_ROOT / "data" / "market.db"

# Sample period. Prices start one year early because the 12-month
# momentum feature needs a full year of history before the first prediction.
START_DATE = "2004-01-01"
END_DATE = "2025-12-31"

# Number of stocks in the universe. 600 > 503 S&P 500 members, so all are kept;
# the download always fetches all members, so a lower value only filters afterwards.
N_STOCKS = 600

# Download prices in batches to stay within Yahoo Finance rate limits
BATCH_SIZE = 50

# Daily Fama-French three-factor data (market, size, value) from Kenneth French's library
FF_FACTORS_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)

# Wikipedia page that lists the current S&P 500 constituents
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
