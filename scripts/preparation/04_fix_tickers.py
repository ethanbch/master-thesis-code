"""
04_fix_tickers.py
-----------------
Re-download missing tickers identified by 03_check_coverage.py
after applying RIC→Yahoo remapping. Merge with existing prices.

Input:  data/intermediate/missing_add_tickers.csv, data/intermediate/prices_raw.parquet
Output: data/intermediate/prices_raw.parquet (enriched)
"""

import random
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
data_dir = ROOT / "data" / "intermediate"

# ----------------------------------------------------------
# 1. LOAD MISSING TICKERS
# ----------------------------------------------------------
missing = pd.read_csv(data_dir / "missing_add_tickers.csv")
prices_existing = pd.read_parquet(data_dir / "prices_raw.parquet")

tickers_missing = missing["ticker"].unique().tolist()
print(f"Missing ADD tickers to fix: {len(tickers_missing)}")

# ----------------------------------------------------------
# 2. TICKER REMAPPING
# ----------------------------------------------------------
swiss_remap = {t: t.replace(".S", ".SW") for t in tickers_missing if t.endswith(".S")}

german_remap = {
    "SAPG.DE": "SAP.DE",
    "BMWG.DE": "BMW.DE",
    "CBKG.DE": "CBK.DE",
    "ALVG.DE": "ALV.DE",
    "LXSG.DE": "LXS.DE",
    "HOTG.DE": "HOT.DE",
    "NAFG.DE": "NDA.DE",
    "HAGG.DE": "HAG.DE",
    "HFGG.DE": "HFG.DE",
    "TEGG.DE": "TEG.DE",
    "TKAG.DE": "TKA.DE",
    "AG1G.DE": "AG1.DE",
    "SDFGn.DE": "SDF.DE",
    "AFXG.DE": "AFX.DE",
    "GBFG.DE": "GBF.DE",
    "FTKn.DE": "FTK.DE",
    "NDXG.DE": "NDX1.DE",
    "PUMG.DE": "PUM.DE",
    "JENGN.DE": "JEN.DE",
    "BOSSn.DE": "BOSS.DE",
    "AIXGn.DE": "AIXA.DE",
    "AMV0n.DE": "AM3D.DE",
}

irish_remap = {t: t.replace(".I", ".IR") for t in tickers_missing if t.endswith(".I")}
austrian_remap = {
    t: t.replace(".VI", ".VIE") for t in tickers_missing if t.endswith(".VI")
}
luxembourg_remap = {t: t.replace(".PA", ".PA") for t in tickers_missing if "SESFd" in t}
luxembourg_remap["SESFd.PA"] = "SESG.PA"

all_remaps = {
    **swiss_remap,
    **german_remap,
    **irish_remap,
    **austrian_remap,
    **luxembourg_remap,
}

# Build download list
download_map = {}
for ric in tickers_missing:
    yahoo = all_remaps.get(ric, ric)
    download_map[yahoo] = ric

yahoo_tickers = list(download_map.keys())
print(f"Tickers to attempt after remapping: {len(yahoo_tickers)}")
print(f"Swiss remapped  : {len(swiss_remap)}")
print(f"German remapped : {len(german_remap)}")
print(f"Irish remapped  : {len(irish_remap)}")

# ----------------------------------------------------------
# 3. DOWNLOAD REMAPPED TICKERS
# ----------------------------------------------------------
BATCH_SIZE = 25
START_DATE = "2013-01-01"
END_DATE = "2026-02-28"

all_data = []
failed_tickers = []

batches = [
    yahoo_tickers[i : i + BATCH_SIZE] for i in range(0, len(yahoo_tickers), BATCH_SIZE)
]

for batch_num, batch in enumerate(batches, start=1):
    n_batches = len(batches)

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
                print(f"[{batch_num}/{n_batches}] Empty response, skipping")
                break

            close = raw["Adj Close"].stack(future_stack=True).reset_index()
            close.columns = ["date", "ticker", "close"]

            volume = raw["Volume"].stack(future_stack=True).reset_index()
            volume.columns = ["date", "ticker", "volume"]

            df = close.merge(volume, on=["date", "ticker"]).dropna(
                subset=["close", "volume"]
            )

            # Remap yahoo ticker back to original RIC
            df["ticker"] = (
                df["ticker"]
                .map({v: k for k, v in download_map.items()})
                .fillna(df["ticker"])
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
# 4. MERGE WITH EXISTING PRICES
# ----------------------------------------------------------
if all_data:
    new_prices = pd.concat(all_data, ignore_index=True).dropna(
        subset=["close", "volume"]
    )
    new_prices["date"] = pd.to_datetime(new_prices["date"])

    print(
        f"\nNew data retrieved: {len(new_prices)} rows, {new_prices['ticker'].nunique()} tickers"
    )

    tickers_to_replace = new_prices["ticker"].unique()
    prices_existing = prices_existing[
        ~prices_existing["ticker"].isin(tickers_to_replace)
    ]

    prices_final = (
        pd.concat([prices_existing, new_prices], ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    prices_final.to_parquet(data_dir / "prices_raw.parquet", index=False)
    print(
        f"Export complete : {len(prices_final)} rows, {prices_final['ticker'].nunique()} tickers"
    )
    print(
        f"Date range     : {prices_final['date'].min().date()} -> {prices_final['date'].max().date()}"
    )
else:
    print("No new data retrieved")

if failed_tickers:
    pd.Series(failed_tickers, name="ticker").to_csv(
        data_dir / "failed_tickers_fix.csv", index=False
    )
    print(
        f"Still failing after remap: {len(failed_tickers)} — see data/intermediate/failed_tickers_fix.csv"
    )
