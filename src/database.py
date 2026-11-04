"""
Thin helper layer around the SQLite database.

Every component (data download, features, models, app) reads and writes
through these two functions, so the storage format is defined in one place.
"""

import sqlite3

import pandas as pd

from src.config import DB_PATH


def get_connection() -> sqlite3.Connection:
    """Open a connection to the project database, creating the folder if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def write_table(df: pd.DataFrame, table: str) -> None:
    """
    Store a DataFrame as a database table, replacing any previous version.

    Replacing (rather than appending) keeps every run reproducible:
    re-running a script always produces the same table.
    """
    with get_connection() as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)


def read_table(table: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Load a complete table from the database into a DataFrame."""
    with get_connection() as conn:
        return pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=parse_dates)
