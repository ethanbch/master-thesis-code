"""
04_fix_tickers.py  [STOXX 50 PROXY]
-------------------------------------
Re-download missing tickers identified by 03_check_coverage.py
after applying RIC→Yahoo remapping. Merge with existing prices.

Input:  stoxx50/data/intermediate/missing_add_tickers.csv,
        stoxx50/data/intermediate/prices_raw.parquet
Output: stoxx50/data/intermediate/prices_raw.parquet (enriched)
"""

import random
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

SX50_ROOT = Path(__file__).resolve().parents[2]
data_dir = SX50_ROOT / "data" / "intermediate"

# ----------------------------------------------------------
# 1. LOAD MISSING TICKERS
# ----------------------------------------------------------
missing_path = data_dir / "missing_add_tickers.csv"
if not missing_path.exists():
    print("No missing_add_tickers.csv found — nothing to fix.")
    raise SystemExit(0)

missing = pd.read_csv(missing_path)
prices_existing = pd.read_parquet(data_dir / "prices_raw.parquet")

tickers_missing = missing["ticker"].unique().tolist()
print(f"Missing ADD tickers to fix: {len(tickers_missing)}")

if len(tickers_missing) == 0:
    print("All tickers already covered. Nothing to do.")
    raise SystemExit(0)

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
luxembourg_remap = {"SESFd.PA": "SESG.PA"}

all_remaps = {
    **swiss_remap,
    **german_remap,
    **irish_remap,
    **austrian_remap,
    **luxembourg_remap,
}

download_map = {}
for ric in tickers_missing:
    yahoo = all_remaps.get(ric, ric)
    download_map[yahoo] = ric

yahoo_tickers = list(download_map.keys())
print(f"Tickers to attempt after remapping: {len(yahoo_tickers)}")

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
                batch, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False
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
            df["ticker"] = (
                df["ticker"]
                .map({v: k for k, v in download_map.items()})
                .fillna(df["ticker"])
            )

            all_data.append(df)
            print(
                f"[{batch_num}/{n_batches}] Done — {len(df)} rows, "
                f"{df['ticker'].nunique()} valid tickers"
            )
            success = True
            break

        except Exception as e:
            wait_retry = 60 * (attempt + 1)
            print(
                f"[{batch_num}/{n_batches}] Attempt {attempt+1}/3 — waiting {wait_retry}s — {e}"
            )
            time.sleep(wait_retry)

    if not success:
        failed_tickers.extend(batch)

# ----------------------------------------------------------
# 4. MERGE WITH EXISTING PRICES
# ----------------------------------------------------------
if all_data:
    new_prices = pd.concat(all_data, ignore_index=True).dropna(
        subset=["close", "volume"]
    )
    new_prices["date"] = pd.to_datetime(new_prices["date"])

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
        f"\n✅ Export: {len(prices_final)} rows, "
        f"{prices_final['ticker'].nunique()} tickers"
    )
else:
    print("No new data retrieved.")

if failed_tickers:
    pd.Series(failed_tickers, name="ticker").to_csv(
        data_dir / "failed_tickers_fix.csv", index=False
    )
    print(
        f"Still failing: {len(failed_tickers)} — see stoxx50/data/intermediate/failed_tickers_fix.csv"
    )
