"""
01_panel_composition.py  [STOXX 50 PROXY]
-----------------------------------------
Build the composition panel from STOXX 600 monthly CSV files,
and detect ADD/DELETE events using rank threshold = 50.

Stocks crossing rank 50 within the STOXX 600 universe are used
as a proxy for EURO STOXX 50 (SX5E) inclusions/exclusions.

Input:  <project_root>/data/raw/inclusions/slpublic_sxxp_*.csv
Output: stoxx50/data/intermediate/panel_composition.parquet
        stoxx50/data/intermediate/events.csv
"""

import glob
from pathlib import Path

import pandas as pd

SX50_ROOT = Path(__file__).resolve().parents[2]  # stoxx50/
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # master-thesis-code/

RANK_THRESHOLD = 50  # ← key difference vs STOXX 600 analysis

# ============================================================
# 1. CHARGEMENT ET MERGE DE TOUS LES FICHIERS
# ============================================================

folder = PROJECT_ROOT / "data" / "raw" / "inclusions"
files = sorted(glob.glob(str(folder / "slpublic_sxxp_*.csv")))
print(f">>> {len(files)} fichiers trouvés dans {folder}")

dfs = []
for f in files:
    df = pd.read_csv(f, sep=";", dtype=str)
    dfs.append(df)

panel = pd.concat(dfs, ignore_index=True)

# Nettoyage des types
panel["Creation_Date"] = pd.to_datetime(panel["Creation_Date"], format="%Y%m%d")
panel["Rank (FINAL)"] = pd.to_numeric(panel["Rank (FINAL)"], errors="coerce")
panel["Rank (PREVIOUS)"] = pd.to_numeric(panel["Rank (PREVIOUS)"], errors="coerce")

panel = panel[
    [
        "Creation_Date",
        "ISIN",
        "RIC",
        "Instrument_Name",
        "Country",
        "Exchange",
        "Index Membership",
        "Rank (FINAL)",
        "Rank (PREVIOUS)",
    ]
].rename(
    columns={
        "Creation_Date": "date",
        "RIC": "ticker",
        "Instrument_Name": "name",
        "Rank (FINAL)": "rank_final",
        "Rank (PREVIOUS)": "rank_prev",
    }
)

print(f">>> Panel : {panel.shape[0]} lignes, {panel['date'].nunique()} snapshots")
print(panel.head())

# ============================================================
# 2. DÉTECTION DES ÉVÉNEMENTS  (seuil = RANK_THRESHOLD)
# ============================================================

valid = panel.dropna(subset=["rank_final", "rank_prev"])
events = valid[
    (
        (valid["rank_final"] <= RANK_THRESHOLD) & (valid["rank_prev"] > RANK_THRESHOLD)
    )  # ADD
    | (
        (valid["rank_prev"] <= RANK_THRESHOLD) & (valid["rank_final"] > RANK_THRESHOLD)
    )  # DELETE
].copy()

events["event_type"] = events.apply(
    lambda r: (
        "ADD"
        if pd.notna(r["rank_final"]) and r["rank_final"] <= RANK_THRESHOLD
        else "DELETE"
    ),
    axis=1,
)

events = events[
    [
        "date",
        "ISIN",
        "ticker",
        "name",
        "Country",
        "event_type",
        "rank_final",
        "rank_prev",
    ]
]
events = events.sort_values("date").reset_index(drop=True)

print(f"\n>>> {len(events)} événements détectés (seuil = {RANK_THRESHOLD})")
print(events["event_type"].value_counts())
print(events.head(20))

# ============================================================
# 3. EXPORT
# ============================================================

out_dir = SX50_ROOT / "data" / "intermediate"
out_dir.mkdir(parents=True, exist_ok=True)
panel.to_parquet(out_dir / "panel_composition.parquet", index=False)
events.to_csv(out_dir / "events.csv", index=False)
print(
    f"\n✅ Exports :"
    f"\n   stoxx50/data/intermediate/panel_composition.parquet"
    f"\n   stoxx50/data/intermediate/events.csv"
)
