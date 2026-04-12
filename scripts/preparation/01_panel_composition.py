"""
01_panel_composition.py
-----------------------
Build the composition panel from STOXX monthly CSV files,
and detect ADD/DELETE events by rank transition.

Input:  data/raw/inclusions/slpublic_sxxp_*.csv
Output: data/intermediate/panel_composition.parquet
        data/intermediate/events.csv
"""

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# 1. CHARGEMENT ET MERGE DE TOUS LES FICHIERS
# ============================================================

folder = ROOT / "data" / "raw" / "inclusions"
files = sorted(glob.glob(str(folder / "slpublic_sxxp_*.csv")))
print(f">>> {len(files)} fichiers trouvés")

dfs = []
for f in files:
    df = pd.read_csv(f, sep=";", dtype=str)
    dfs.append(df)

# Merge en un seul DataFrame
panel = pd.concat(dfs, ignore_index=True)

# Nettoyage des types
panel["Creation_Date"] = pd.to_datetime(panel["Creation_Date"], format="%Y%m%d")
panel["Rank (FINAL)"] = pd.to_numeric(panel["Rank (FINAL)"], errors="coerce")
panel["Rank (PREVIOUS)"] = pd.to_numeric(panel["Rank (PREVIOUS)"], errors="coerce")

# Colonnes utiles uniquement
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
# 2. DÉTECTION DES ÉVÉNEMENTS (INCLUSIONS / EXCLUSIONS)
# ============================================================

valid = panel.dropna(subset=["rank_final", "rank_prev"])
events = valid[
    ((valid["rank_final"] <= 600) & (valid["rank_prev"] > 600))  # ADD
    | ((valid["rank_prev"] <= 600) & (valid["rank_final"] > 600))  # DELETE
].copy()

events["event_type"] = events.apply(
    lambda r: (
        "ADD" if pd.notna(r["rank_final"]) and r["rank_final"] <= 600 else "DELETE"
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

print(f"\n>>> {len(events)} événements détectés")
print(events["event_type"].value_counts())
print(events.head(20))

# ============================================================
# 3. EXPORT
# ============================================================

out_dir = ROOT / "data" / "intermediate"
out_dir.mkdir(parents=True, exist_ok=True)
panel.to_parquet(out_dir / "panel_composition.parquet", index=False)
events.to_csv(out_dir / "events.csv", index=False)
print(
    "\n✅ Exports : data/intermediate/panel_composition.parquet + data/intermediate/events.csv"
)
