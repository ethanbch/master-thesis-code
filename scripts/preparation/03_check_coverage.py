"""
03_check_coverage.py
--------------------
Check coverage of ADD event tickers in downloaded prices.
Apply RIC → Yahoo Finance remapping. Export missing tickers list.

Input:  data/intermediate/events.csv, data/intermediate/prices_raw.parquet,
        data/intermediate/panel_composition.parquet
Output: data/intermediate/events.csv (updated with ric column),
        data/intermediate/missing_add_tickers.csv
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------
# 1. DEFINE REMAPPING (RIC -> Yahoo Finance)
# ----------------------------------------------------------


def _apply_regex_remaps(ticker: str) -> str:
    """Applique les règles de traduction RIC → Yahoo Finance par regex.

    Règles :
    - Suisse : .S → .SW
    - Nordiques : classe B/A avant .ST, .CO, .HE, .OL (ex: BETSb.ST → BETS-B.ST)
    """
    if not isinstance(ticker, str):
        return ticker
    # Suisse : .S → .SW
    ticker = re.sub(r"\.S$", ".SW", ticker)
    # Nordiques : classe B/A juste avant le suffixe de bourse
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
    "BASFn.DE": "BAS.DE",
    "DPWGN.DE": "DHL.DE",
    "DPWGn.DE": "DHL.DE",
    "LHAG.DE": "LHA.DE",
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

# ----------------------------------------------------------
# 2. LOAD DATA AND APPLY REMAP
# ----------------------------------------------------------
data_dir = ROOT / "data" / "intermediate"
events = pd.read_csv(data_dir / "events.csv", dtype={"ticker": str, "ric": str})
prices = pd.read_parquet(data_dir / "prices_raw.parquet")
panel = pd.read_parquet(data_dir / "panel_composition.parquet")

# Supprimer les tickers artefacts (NaN ou purement numériques, ex: "1.7")
events = events.dropna(subset=["ticker"])
events = events[~events["ticker"].str.fullmatch(r"\d+(\.\d+)?")].reset_index(drop=True)

# Preserve original RIC, apply manual remap then regex rules
if "ric" not in events.columns:
    events["ric"] = events["ticker"]
events["ticker"] = events["ric"].replace(MANUAL_REMAP).apply(_apply_regex_remaps)

if "ric" not in panel.columns:
    panel["ric"] = panel["ticker"]
panel["ticker"] = panel["ric"].replace(MANUAL_REMAP).apply(_apply_regex_remaps)

# Save updated events
events.to_csv(data_dir / "events.csv", index=False)
n_remapped = (events["ticker"] != events["ric"]).sum()
print(f"Remapping applied: {n_remapped} tickers updated in events.csv")

# ----------------------------------------------------------
# 3. COVERAGE OF ADD EVENTS (treated group)
# ----------------------------------------------------------
tickers_with_data = set(prices["ticker"].unique())
add_events = events[events["event_type"] == "ADD"].copy()
add_events["has_data"] = add_events["ticker"].isin(tickers_with_data)

n_add = len(add_events)
n_add_ok = add_events["has_data"].sum()
n_add_miss = n_add - n_add_ok

print("\n" + "=" * 55)
print("ADD EVENTS COVERAGE (treated group)")
print("=" * 55)
print(f"Total ADD events     : {n_add}")
print(f"With price data      : {n_add_ok} ({100 * n_add_ok / n_add:.1f}%)")
print(f"Missing              : {n_add_miss} ({100 * n_add_miss / n_add:.1f}%)")

# ----------------------------------------------------------
# 4. COVERAGE BY COUNTRY
# ----------------------------------------------------------
print("\n" + "=" * 55)
print("COVERAGE BY COUNTRY (ADD events)")
print("=" * 55)
coverage_by_country = (
    add_events.groupby("Country")["has_data"]
    .agg(total="count", with_data="sum")
    .assign(coverage_pct=lambda df: 100 * df["with_data"] / df["total"])
    .sort_values("coverage_pct", ascending=False)
)
print(coverage_by_country.to_string())

# ----------------------------------------------------------
# 5. COVERAGE BY EXCHANGE SUFFIX
# ----------------------------------------------------------
add_events["suffix"] = add_events["ticker"].str.extract(r"\.([A-Z]+)$")

print("\n" + "=" * 55)
print("COVERAGE BY EXCHANGE SUFFIX (ADD events)")
print("=" * 55)
coverage_by_suffix = (
    add_events.groupby("suffix")["has_data"]
    .agg(total="count", with_data="sum")
    .assign(coverage_pct=lambda df: 100 * df["with_data"] / df["total"])
    .sort_values("coverage_pct", ascending=False)
)
print(coverage_by_suffix.to_string())

# ----------------------------------------------------------
# 6. MISSING ADD TICKERS
# ----------------------------------------------------------
missing_add = add_events[~add_events["has_data"]][
    ["date", "ric", "ticker", "name", "Country"]
]
print("\n" + "=" * 55)
print("MISSING ADD TICKERS")
print("=" * 55)
print(missing_add.to_string(index=False))
missing_add.to_csv(data_dir / "missing_add_tickers.csv", index=False)

# ----------------------------------------------------------
# 7. DATA QUALITY
# ----------------------------------------------------------
print("\n" + "=" * 55)
print("DATA QUALITY — AVAILABLE TICKERS")
print("=" * 55)
obs_per_ticker = prices.groupby("ticker")["date"].count()
print(f"Median obs per ticker  : {obs_per_ticker.median():.0f} days")
print(f"Min obs per ticker     : {obs_per_ticker.min()} days")
print(f"Tickers < 252 days     : {(obs_per_ticker < 252).sum()} (less than 1 year)")
print(f"Tickers >= 252 days    : {(obs_per_ticker >= 252).sum()}")
print(f"Tickers >= 2520 days   : {(obs_per_ticker >= 2520).sum()} (10 years)")

# ----------------------------------------------------------
# 8. GEOGRAPHIC BIAS
# ----------------------------------------------------------
print("\n" + "=" * 55)
print("GEOGRAPHIC BIAS — panel vs prices coverage")
print("=" * 55)
panel["has_data"] = panel["ticker"].isin(tickers_with_data)
geo_bias = (
    panel.groupby("Country")["has_data"]
    .agg(total="count", with_data="sum")
    .assign(coverage_pct=lambda df: 100 * df["with_data"] / df["total"])
    .sort_values("coverage_pct", ascending=False)
)
print(geo_bias.to_string())

print("\nOutputs saved: data/intermediate/missing_add_tickers.csv")
