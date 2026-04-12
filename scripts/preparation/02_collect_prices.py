"""
02_collect_prices.py
--------------------
Download historical prices (Adj Close + Volume) via Yahoo Finance
for all ADD event tickers and stable tickers.

Input:  data/intermediate/events.csv, data/intermediate/panel_composition.parquet
Output: data/intermediate/prices_raw.parquet
"""

import random
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]

print(f"yfinance version: {yf.__version__}")

# ----------------------------------------------------------
# 1. LOAD TICKERS
# ----------------------------------------------------------
events = pd.read_csv(ROOT / "data" / "intermediate" / "events.csv")
panel = pd.read_parquet(ROOT / "data" / "intermediate" / "panel_composition.parquet")

add_tickers = events[events["event_type"] == "ADD"]["ticker"].unique().tolist()
stable_tickers = (
    panel.groupby("ticker")["date"].count().loc[lambda x: x >= 12].index.tolist()
)

all_tickers = list(set(add_tickers + stable_tickers + ["^STOXX"]))
print(f"Tickers to download: {len(all_tickers)}")

# ----------------------------------------------------------
# 2. BATCH DOWNLOAD
# ----------------------------------------------------------
BATCH_SIZE = 25
START_DATE = "2013-01-01"
END_DATE = "2026-02-28"

all_data = []
failed_tickers = []

for i in range(0, len(all_tickers), BATCH_SIZE):
    batch = all_tickers[i : i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    n_batches = -(-len(all_tickers) // BATCH_SIZE)

    pause = random.uniform(15, 25)
    print(f"[{batch_num}/{n_batches}] Waiting {pause:.0f}s ...")
    time.sleep(pause)

    print(f"[{batch_num}/{n_batches}] Downloading {len(batch)} tickers ...")

    success = False
    for attempt in range(3):
        try:
            raw = yf.download(
                batch,
                start=START_DATE,
                end=END_DATE,
                progress=False,
                auto_adjust=False,
            )

            if raw.empty:
                print(f"[{batch_num}/{n_batches}] Empty response, skipping batch")
                break

            close = raw["Adj Close"].stack(future_stack=True).reset_index()
            close.columns = ["date", "ticker", "close"]

            volume = raw["Volume"].stack(future_stack=True).reset_index()
            volume.columns = ["date", "ticker", "volume"]

            df = close.merge(volume, on=["date", "ticker"]).dropna(
                subset=["close", "volume"]
            )

            all_data.append(df)
            print(
                f"[{batch_num}/{n_batches}] Done — {len(df)} rows, {df['ticker'].nunique()} valid tickers"
            )
            success = True
            break

        except Exception as e:
            wait_retry = 60 * (attempt + 1)
            print(
                f"[{batch_num}/{n_batches}] Attempt {attempt + 1}/3 — waiting {wait_retry}s — {e}"
            )
            time.sleep(wait_retry)

    if not success:
        failed_tickers.extend(batch)
        print(f"[{batch_num}/{n_batches}] Batch permanently failed")

# ----------------------------------------------------------
# 3. EXPORT
# ----------------------------------------------------------
out_dir = ROOT / "data" / "intermediate"

if failed_tickers:
    pd.Series(failed_tickers, name="ticker").to_csv(
        out_dir / "failed_tickers.csv", index=False
    )
    print(
        f"Failed tickers: {len(failed_tickers)} — see data/intermediate/failed_tickers.csv"
    )

if not all_data:
    print("No data retrieved — check connection and tickers")
    raise SystemExit(1)

prices = (
    pd.concat(all_data, ignore_index=True)
    .dropna(subset=["close", "volume"])
    .sort_values(["ticker", "date"])
    .reset_index(drop=True)
)
prices["date"] = pd.to_datetime(prices["date"])

prices.to_parquet(out_dir / "prices_raw.parquet", index=False)

print(f"Export complete : {len(prices)} rows, {prices['ticker'].nunique()} tickers")
print(
    f"Date range     : {prices['date'].min().date()} -> {prices['date'].max().date()}"
)
