"""
02_collect_prices.py  [STOXX 50 PROXY]
---------------------------------------
Bootstrap price data from the STOXX 600 analysis, then add ^STOXX50E
as the benchmark for the market model.

Strategy:
  1. Load existing prices_raw.parquet from main analysis (all STOXX 600 tickers).
  2. Download ^STOXX50E (EURO STOXX 50 index) if not already present.
  3. Save the combined prices to stoxx50/data/intermediate/prices_raw.parquet.

This avoids re-downloading thousands of price series already collected.

Input:  <project_root>/data/intermediate/prices_raw.parquet (STOXX 600 prices cache)
Output: stoxx50/data/intermediate/prices_raw.parquet
"""

import random
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

SX50_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

SX600_PRICES = PROJECT_ROOT / "data" / "intermediate" / "prices_raw.parquet"
OUT_PATH = SX50_ROOT / "data" / "intermediate" / "prices_raw.parquet"

BENCHMARK_TICKER = "^STOXX50E"  # ← EURO STOXX 50 index
START_DATE = "2013-01-01"
END_DATE = "2026-02-28"

print(f"yfinance version: {yf.__version__}")

# ----------------------------------------------------------
# 1. LOAD EXISTING STOXX 600 PRICES
# ----------------------------------------------------------
if not SX600_PRICES.exists():
    raise FileNotFoundError(
        f"STOXX 600 prices not found at {SX600_PRICES}. "
        "Run the main analysis pipeline (scripts/preparation/02_collect_prices.py) first."
    )

print(f"Loading existing STOXX 600 prices from {SX600_PRICES} ...")
prices = pd.read_parquet(SX600_PRICES)
print(f"  {len(prices):,} rows, {prices['ticker'].nunique()} tickers")

existing_tickers = set(prices["ticker"].unique())
print(f"  Tickers already present: {len(existing_tickers)}")

# ----------------------------------------------------------
# 2. ADD ^STOXX50E BENCHMARK IF MISSING
# ----------------------------------------------------------
if BENCHMARK_TICKER in existing_tickers:
    print(f"\n  {BENCHMARK_TICKER} already present in prices. No download needed.")
    prices_combined = prices.copy()
else:
    print(f"\n  {BENCHMARK_TICKER} not found — downloading from Yahoo Finance …")

    pause = random.uniform(5, 10)
    print(f"  Waiting {pause:.0f}s ...")
    time.sleep(pause)

    success = False
    for attempt in range(3):
        try:
            raw = yf.download(
                BENCHMARK_TICKER,
                start=START_DATE,
                end=END_DATE,
                progress=False,
                auto_adjust=False,
            )

            if raw.empty:
                raise ValueError(f"Empty response for {BENCHMARK_TICKER}")

            # Flatten multi-index if needed
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)

            close_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
            bm_df = raw[[close_col, "Volume"]].reset_index()
            bm_df.columns = ["date", "close", "volume"]
            bm_df["ticker"] = BENCHMARK_TICKER
            bm_df = bm_df.dropna(subset=["close"])

            print(f"  Downloaded {len(bm_df)} rows for {BENCHMARK_TICKER}")
            prices_combined = pd.concat([prices, bm_df], ignore_index=True)
            success = True
            break

        except Exception as e:
            wait_retry = 60 * (attempt + 1)
            print(f"  Attempt {attempt + 1}/3 failed — waiting {wait_retry}s — {e}")
            time.sleep(wait_retry)

    if not success:
        raise RuntimeError(
            f"Could not download {BENCHMARK_TICKER}. "
            "Check Yahoo Finance ticker and retry."
        )

# ----------------------------------------------------------
# 3. SAVE
# ----------------------------------------------------------
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
prices_combined.to_parquet(OUT_PATH, index=False)
print(
    f"\n✅ Saved {len(prices_combined):,} rows, "
    f"{prices_combined['ticker'].nunique()} tickers "
    f"→ stoxx50/data/intermediate/prices_raw.parquet"
)
