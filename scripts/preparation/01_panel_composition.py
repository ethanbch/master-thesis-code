"""
01_panel_composition.py
-----------------------
Build the composition panel from STOXX monthly CSV files,
and detect ADD/DELETE events by rank transition.

Input:  data/raw/inclusions/slpublic_sxxp_*.csv
Output: data/intermediate/panel_composition.parquet
        data/intermediate/events.csv

NOTE — Historique des formats CSV
---------------------------------
Certains fichiers (revues trimestrielles, mois 01/04/07/10, de 2015 à
début 2021) proviennent d'exports PDF mal parsés :
  • La colonne ``Rank (FINAL)`` ne contient que des chiffres 0-9
    (artefact du parsing PDF), rendant les rangs finaux inutilisables.
  • Des lignes parasites « Page » (en-têtes/pieds de page PDF) se
    glissent dans les données.
  • La colonne ``Rank (PREVIOUS)`` reste correcte dans tous les cas.
À partir de nov-2023, une colonne ``FF Mcap (MEUR)`` est ajoutée mais
n'affecte pas le parsing par nom de colonne.
"""

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# Seuil : si Rank (FINAL) a moins de RANK_UNIQUE_THRESHOLD valeurs
# uniques, on considère que la colonne est corrompue (artefact PDF).
RANK_UNIQUE_THRESHOLD = 50


# ============================================================
# HELPERS
# ============================================================


def _is_rank_corrupted(series: pd.Series) -> bool:
    """Détecte si une série de rangs est corrompue (trop peu de valeurs uniques)."""
    numeric = pd.to_numeric(series, errors="coerce")
    return int(numeric.nunique()) < RANK_UNIQUE_THRESHOLD


def _remove_pdf_artifacts(df: pd.DataFrame, filepath: str) -> pd.DataFrame:
    """Supprime les lignes parasites issues du parsing PDF.

    Deux types d'artefacts sont présents dans les fichiers pré-nov 2023 :
    1. Lignes « Page » : en-têtes/pieds de page PDF contenant "Page" dans
       le nom de l'instrument.
    2. Lignes à rang 0-9 dupliqué : les numéros de page du PDF sont parsés
       comme ``Rank (FINAL)`` = 0–9. Le vrai stock classé 1–9 a un
       ``Rank (PREVIOUS)`` proche de son rang ; les artefacts ont un
       ``Rank (PREVIOUS)`` très éloigné (séquentiel dans le PDF).
    """
    fname = Path(filepath).name
    n_before = len(df)

    # 1. Lignes « Page »
    page_mask = df["Instrument_Name"].str.contains("Page", na=False)
    df = df[~page_mask].copy()

    # 2. Rang 0 : toujours un artefact (les vrais rangs commencent à 1)
    df = df[df["Rank (FINAL)"] != 0].copy()

    # 3. Rangs 1-9 dupliqués : garder uniquement le stock authentique
    #    (celui dont |rank_final - rank_prev| est le plus petit)
    for rank_val in range(1, 10):
        mask = df["Rank (FINAL)"] == rank_val
        if mask.sum() > 1:
            subset = df[mask].copy()
            subset["_diff"] = (subset["Rank (FINAL)"] - subset["Rank (PREVIOUS)"]).abs()
            keep_idx = subset["_diff"].idxmin()
            drop_idx = subset.index.difference([keep_idx])
            df = df.drop(drop_idx)

    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"  ⚠ {fname}: {n_removed} lignes artefacts PDF supprimées")

    return df


def _load_and_clean_file(filepath: str) -> pd.DataFrame:
    """Charge un fichier CSV STOXX, nettoie les artefacts PDF et
    corrige les rangs corrompus le cas échéant.

    Returns
    -------
    DataFrame avec les colonnes normalisées et les types sécurisés.
    """
    df = pd.read_csv(filepath, sep=";", dtype=str)

    # --- Conversion numérique des rangs ---
    df["Rank (FINAL)"] = pd.to_numeric(df["Rank (FINAL)"], errors="coerce")
    df["Rank (PREVIOUS)"] = pd.to_numeric(df["Rank (PREVIOUS)"], errors="coerce")

    # --- Suppression de tous les artefacts PDF ---
    df = _remove_pdf_artifacts(df, filepath)

    # --- Détection et correction des rangs totalement corrompus ---
    # Dans certains fichiers (revues trimestrielles 2015-2021), Rank (FINAL)
    # ne contient que des chiffres 0-9 même après nettoyage des artefacts ;
    # on le remplace par Rank (PREVIOUS) qui est fiable.
    if _is_rank_corrupted(df["Rank (FINAL)"]):
        print(
            f"  ⚠ {Path(filepath).name}: Rank (FINAL) corrompu "
            f"({int(df['Rank (FINAL)'].nunique())} valeurs uniques) "
            f"→ remplacé par Rank (PREVIOUS)"
        )
        df["Rank (FINAL)"] = df["Rank (PREVIOUS)"]

    # --- Conversion en int nullable (Int64) pour un typage propre ---
    for col in ("Rank (FINAL)", "Rank (PREVIOUS)"):
        df[col] = df[col].astype("Int64")

    return df


# ============================================================
# 1. CHARGEMENT ET MERGE DE TOUS LES FICHIERS
# ============================================================

folder = ROOT / "data" / "raw" / "inclusions"
files = sorted(glob.glob(str(folder / "slpublic_sxxp_*.csv")))
print(f">>> {len(files)} fichiers trouvés")

dfs = []
for f in files:
    dfs.append(_load_and_clean_file(f))

# Merge en un seul DataFrame
panel = pd.concat(dfs, ignore_index=True)

# Nettoyage des types
panel["Creation_Date"] = pd.to_datetime(panel["Creation_Date"], format="%Y%m%d")

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

# On ne garde que les lignes avec les deux rangs renseignés
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
