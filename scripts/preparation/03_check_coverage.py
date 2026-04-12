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

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------
# 1. DEFINE REMAPPING (RIC -> Yahoo Finance)
# ----------------------------------------------------------
TICKER_REMAP = {
    # Swiss : .S -> .SW
    **{
        t: t.replace(".S", ".SW")
        for t in [
            "AMS.S",
            "BANB.S",
            "BCHN.S",
            "DKSH.S",
            "DOKA.S",
            "EFGN.S",
            "HUBN.S",
            "IFCN.S",
            "INRN.S",
            "MOBN.S",
            "SFSN.S",
            "SUN.S",
            "VATN.S",
            "VZN.S",
            "COTNE.S",
            "HUBN.S",
        ]
    },
    # Irish : .I -> .IR
    "GL9.I": "GL9.IR",
    # Austrian : .VI -> .VIE
    "VOES.VI": "VOES.VI",
    "VIGR.VI": "VIG.VI",
    # German manual remaps
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
}

# ----------------------------------------------------------
# 2. LOAD DATA AND APPLY REMAP
# ----------------------------------------------------------
data_dir = ROOT / "data" / "intermediate"
events = pd.read_csv(data_dir / "events.csv")
prices = pd.read_parquet(data_dir / "prices_raw.parquet")
panel = pd.read_parquet(data_dir / "panel_composition.parquet")

# Preserve original RIC, apply remap to ticker column
if "ric" not in events.columns:
    events["ric"] = events["ticker"]
events["ticker"] = events["ric"].replace(TICKER_REMAP)

if "ric" not in panel.columns:
    panel["ric"] = panel["ticker"]
panel["ticker"] = panel["ric"].replace(TICKER_REMAP)

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
