"""
04_fix_tickers.py
-----------------
Re-download missing tickers identified by 03_check_coverage.py
after applying RIC→Yahoo remapping. Merge with existing prices.

Input:  data/intermediate/missing_add_tickers.csv, data/intermediate/prices_raw.parquet
Output: data/intermediate/prices_raw.parquet (enriched)
"""

import random
import re
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
data_dir = ROOT / "data" / "intermediate"

# ----------------------------------------------------------
# 1. LOAD MISSING TICKERS
# ----------------------------------------------------------
missing = pd.read_csv(
    data_dir / "missing_add_tickers.csv", dtype={"ticker": str, "ric": str}
)
prices_existing = pd.read_parquet(data_dir / "prices_raw.parquet")

# Supprimer les artefacts PDF : tickers NaN ou purement numériques (ex: "1.7", "2.0")
missing = missing.dropna(subset=["ticker"])
missing = missing[~missing["ticker"].str.fullmatch(r"\d+(\.\d+)?")]

tickers_missing = missing["ticker"].unique().tolist()
print(f"Missing ADD tickers to fix: {len(tickers_missing)}")

# ----------------------------------------------------------
# 2. TICKER REMAPPING
# ----------------------------------------------------------


def _apply_regex_remaps(ticker: str) -> str:
    """Applique les règles de traduction RIC → Yahoo Finance par regex."""
    if not isinstance(ticker, str):
        return ticker
    # Suisse : .S → .SW
    ticker = re.sub(r"\.S$", ".SW", ticker)
    # Nordiques : classe B/A avant .ST, .CO, .HE, .OL
    # ex: CARLB.CO → CARL-B.CO, BETSb.ST → BETS-B.ST
    ticker = re.sub(
        r"([A-Za-z])([BbAa])\.(ST|CO|HE|OL)$",
        lambda m: f"{m.group(1)}-{m.group(2).upper()}.{m.group(3)}",
        ticker,
    )
    return ticker


MANUAL_REMAP = {
    # France
    "RENA.PA": "RNO.PA",
    "AIRP.PA": "AI.PA",
    "CARR.PA": "CA.PA",
    "PUBP.PA": "PUB.PA",
    "ERMT.PA": "ERA.PA",
    "EPED.PA": "FGR.PA",  # Faurecia → Forvia
    "AIRF.PA": "AF.PA",  # Air France-KLM
    "LTEN.PA": "ATE.PA",  # Alten
    "JCDX.PA": "DEC.PA",  # JCDecaux
    "NEXI.PA": "NXI.PA",  # Nexity
    "BICP.PA": "BB.PA",  # Société BIC
    "SOIT.PA": "SOI.PA",  # Soitec
    "SOPR.PA": "SOP.PA",  # Sopra Steria
    "RUBF.PA": "RUI.PA",  # Rubis
    "NEXS.PA": "NEX.PA",  # Nexans
    "ISOS.PA": "IPS.PA",  # Ipsos
    "VLLP.PA": "VK.PA",  # Vallourec
    "UBIP.PA": "UBI.PA",  # Ubisoft
    "VLOF.PA": "FR.PA",  # Valeo
    # Allemagne
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
    "BOSSn.DE": "BOSS.DE",
    "AIXGn.DE": "AIXA.DE",
    "AMV0n.DE": "AM3D.DE",
    "JENGN.DE": "JEN.DE",
    "FRAG.DE": "FRA.DE",
    "MORG.DE": "MOR.DE",
    "BASFN.DE": "BAS.DE",
    "DPWGN.DE": "DHL.DE",
    "LHAG.DE": "LHA.DE",
    "BASFn.DE": "BAS.DE",
    "DPWGn.DE": "DHL.DE",
    "SOWG.DE": "SOW.DE",
    "DUEG.DE": "DUE.DE",
    "DEQGn.DE": "DEQ.DE",
    "WCHG.DE": "WCH.DE",  # Wacker Chemie
    "EVDG.DE": "EVD.DE",  # CTS Eventim
    "NEKG.DE": "NEM.DE",  # Nemetschek
    "RAAG.DE": "RAA.DE",  # Rational AG
    "COKG.DE": "COK.DE",  # Cancom
    "HYQGn.DE": "HYQ.DE",  # Hypoport
    "TLXGn.DE": "TLX.DE",  # Talanx
    "S92G.DE": "S92.DE",  # SMA Solar Technology
    "GLJn.DE": "GLJ.DE",  # Grenke
    "FNTGn.DE": "FNTN.DE",  # Freenet
    # Grande-Bretagne
    "WISEa.L": "WISE.L",  # Wise (class A)
    "GFTU_u.L": "GFTU.L",  # Grafton Group (units)
    "ECM.L": "RS1.L",  # Electrocomponents → RS Group
    "TCAPI.L": "TCAP.L",  # TP ICAP
    "VTYV.L": "VTY.L",  # Vistry Group
    "SCTS.L": "SCT.L",  # Softcat
    "CBRO.L": "CBG.L",  # Close Brothers Group
    "VCTX.L": "VCT.L",  # Victrex
    "BEZG.L": "BEZ.L",  # Beazley
    "BALF.L": "BBY.L",  # Balfour Beatty
    "TRNT.L": "TRN.L",  # Trainline
    "SHCS.L": "SHC.L",  # Shaftesbury Capital
    "HAYS.L": "HAS.L",  # Hays
    "PLUSP.L": "PLUS.L",  # Plus500
    "JUSTJ.L": "JUST.L",  # Just Group
    "PAGPA.L": "PAG.L",  # Paragon Banking Group
    "PAFR.L": "PAF.L",  # Pan African Resources
    "HOCM.L": "HOC.L",  # Hochschild Mining
    "OSBO.L": "OSB.L",  # OSB Group (ex-OneSavings Bank)
    "DC.L": "CURY.L",  # Dixons Carphone → Currys
    "GPOR.L": "GPE.L",  # Great Portland Estates
    # Irlande
    "GL9.I": "GL9.IR",
    "CRH.I": "CRH.L",
    # Autriche
    "VIGR.VI": "VIG.VI",
    # Luxembourg
    "SESFd.PA": "SESG.PA",
}

# Appliquer d'abord le dictionnaire, puis les regex
remapped_tickers = []
for ric in tickers_missing:
    yahoo = MANUAL_REMAP.get(ric, ric)
    yahoo = _apply_regex_remaps(yahoo)
    remapped_tickers.append(yahoo)

# Build download list
download_map = {}
for ric, yahoo in zip(tickers_missing, remapped_tickers):
    download_map[yahoo] = ric

yahoo_tickers = list(download_map.keys())
n_changed = sum(1 for r, y in zip(tickers_missing, remapped_tickers) if r != y)
print(
    f"Tickers to attempt after remapping: {len(yahoo_tickers)} ({n_changed} remapped)"
)

# ----------------------------------------------------------
# 3. DOWNLOAD REMAPPED TICKERS
# ----------------------------------------------------------
BATCH_SIZE = 50
START_DATE = "2013-01-01"
END_DATE = "2026-02-28"

all_data = []
failed_tickers = []

batches = [
    yahoo_tickers[i : i + BATCH_SIZE] for i in range(0, len(yahoo_tickers), BATCH_SIZE)
]

for batch_num, batch in enumerate(batches, start=1):
    n_batches = len(batches)

    pause = random.uniform(3, 6)
    print(f"[{batch_num}/{n_batches}] Waiting {pause:.1f}s ...")
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
                ignore_tz=True,
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
            wait_retry = 15 * (attempt + 1)
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
