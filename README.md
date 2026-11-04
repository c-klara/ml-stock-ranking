# ML Stock Ranking — Group 05.01

Fundamentals and Methods of Computer Science, HSG, Herbstsemester 2026.

We train our own machine-learning models to rank about 300 US stocks by
expected next-month return, and test whether a portfolio built on those
rankings beats a linear model and a simple momentum strategy.

## Setup

```bash
pip install -r requirements.txt
python -m src.fetch_data     # download prices and factors into data/market.db (10-20 min)
python -m src.features       # build monthly features and the target
streamlit run app/app.py     # launch the app
```

Run all commands from this folder (the project root).

## Structure

```
src/config.py      all settings (period, universe size, data URLs)
src/database.py    read/write helpers for the SQLite database
src/fetch_data.py  step 1: download prices (yfinance) and factors (Kenneth French)
src/features.py    step 2: monthly features and target
app/app.py         Streamlit application
data/              SQLite database (not committed to Git)
log/               contribution and GenAI log
```

## Database tables

| Table    | Columns                                                   | Owner    |
|----------|-----------------------------------------------------------|----------|
| prices   | ticker, date, close, volume                               | Member 1 |
| factors  | date, mkt_rf, smb, hml, rf                                | Member 1 |
| features | ticker, month, feature columns, *_rank columns, target    | Member 2 |

Still to come: `predictions` (Member 3) and `backtest` (Member 4).

## Known limitations (v1)

- Universe uses today's S&P 500 members (survivorship bias).
- Universe filter uses liquidity measured over the full sample.
- Transaction costs are ignored.
