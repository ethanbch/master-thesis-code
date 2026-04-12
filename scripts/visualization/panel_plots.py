"""
panel_plots.py
--------------
Generate a 3x2 summary figure of the panel data.

Input:  data/intermediate/panel_monthly.parquet,
        data/intermediate/events.csv,
        data/intermediate/panel_composition.parquet
Output: figures/panel_overview.png
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def log(msg: str):
    print(f"[plots] {msg}", flush=True)


def main(show: bool = False):
    t0 = time.time()

    log("Chargement de data/intermediate/panel_monthly.parquet...")
    panel = pd.read_parquet(ROOT / "data" / "intermediate" / "panel_monthly.parquet")
    log(f"panel chargé: {len(panel):,} lignes, colonnes={panel.columns.tolist()}")

    log("Chargement de data/intermediate/events.csv...")
    events = pd.read_csv(ROOT / "data" / "intermediate" / "events.csv")
    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    log(f"events chargés: {len(events):,} lignes")

    if "R2_raw" not in panel.columns:
        raise KeyError(
            f"Colonne 'R2_raw' introuvable dans panel_monthly.parquet. Colonnes: {panel.columns.tolist()}"
        )
    panel = panel[panel["R2_raw"] > 0].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"])
    log(f"panel après filtre R2_raw>0: {len(panel):,} lignes")

    if panel.empty:
        raise ValueError("Le panel est vide après filtrage. Impossible de tracer.")

    log("Création de la figure 3x2...")
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        "STOXX Europe 600 — Market Efficiency Panel (2013–2026)",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    # 1. Mean Synchronicity over time
    log("Trace 1/6: Mean Synchronicity")
    ax = axes[0, 0]
    monthly_synch = panel.groupby("date")["Synchronicity"].mean()
    ax.plot(monthly_synch.index, monthly_synch.values, linewidth=1.5, color="#1f77b4")
    ax.axhline(
        monthly_synch.mean(),
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Mean = {monthly_synch.mean():.2f}",
    )
    ax.set_title("Mean Synchronicity (log R²/1−R²) over time")
    ax.set_ylabel("Synchronicity")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)

    # 2. Mean R² over time
    log("Trace 2/6: Mean R²")
    ax = axes[0, 1]
    monthly_r2 = panel.groupby("date")["R2_raw"].mean()
    ax.plot(monthly_r2.index, monthly_r2.values, linewidth=1.5, color="#ff7f0e")
    ax.axhline(
        monthly_r2.mean(),
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Mean R² = {monthly_r2.mean():.3f}",
    )
    ax.set_title("Mean R² (raw) over time")
    ax.set_ylabel("R²")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)

    # 3. Mean Idiosyncratic Volatility over time
    log("Trace 3/6: Mean Idiosyncratic Volatility")
    ax = axes[1, 0]
    monthly_vol = panel.groupby("date")["Idio_Vol"].mean()
    ax.plot(monthly_vol.index, monthly_vol.values, linewidth=1.5, color="#2ca02c")
    ax.axhline(
        monthly_vol.mean(),
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Mean = {monthly_vol.mean():.4f}",
    )
    ax.set_title("Mean Idiosyncratic Volatility over time")
    ax.set_ylabel("Std of residuals")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)

    # 4. Number of ADD events per month
    log("Trace 4/6: Nombre de ADD")
    ax = axes[1, 1]
    add_counts = (
        events[events["event_type"] == "ADD"]
        .groupby("date")
        .size()
        .reset_index(name="n_add")
    )
    ax.bar(
        add_counts["date"], add_counts["n_add"], width=20, color="#9467bd", alpha=0.8
    )
    ax.set_title("Number of ADD events per rebalancing date")
    ax.set_ylabel("Count")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    # 5. Distribution of R²
    log("Trace 5/6: Distribution R²")
    ax = axes[2, 0]
    ax.hist(panel["R2_raw"], bins=80, color="#8c564b", edgecolor="white", linewidth=0.3)
    ax.axvline(
        panel["R2_raw"].median(),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Median = {panel['R2_raw'].median():.3f}",
    )
    ax.set_title("Distribution of R² (all tickers, all months)")
    ax.set_xlabel("R²")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Synchronicity by country (boxplot)
    log("Trace 6/6: Boxplot par pays")
    ax = axes[2, 1]
    panel_comp = pd.read_parquet(
        ROOT / "data" / "intermediate" / "panel_composition.parquet"
    )[["ticker", "Country"]].drop_duplicates("ticker")
    panel_country = panel.merge(panel_comp, on="ticker", how="left")

    top_countries = ["GB", "FR", "DE", "SE", "CH", "IT", "NO", "ES", "NL", "DK"]
    data_box = [
        panel_country[panel_country["Country"] == c]["Synchronicity"].dropna().values
        for c in top_countries
    ]
    bp = ax.boxplot(
        data_box, tick_labels=top_countries, patch_artist=True, showfliers=False
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#aec7e8")
    ax.set_title("Synchronicity distribution by country")
    ax.set_ylabel("Synchronicity")
    ax.set_xlabel("Country")
    ax.grid(True, alpha=0.3, axis="y")

    log("Mise en page + sauvegarde...")
    plt.tight_layout()
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    output_path = fig_dir / "panel_overview.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    log(f"Saved: {output_path}")

    elapsed = time.time() - t0
    log(f"Terminé en {elapsed:.1f}s")

    if show:
        log("Affichage interactif activé (--show)")
        plt.show()
    else:
        log("Mode non-interactif: pas de plt.show() (utilise --show pour afficher)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot panel overview charts.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Affiche la figure à l'écran (sinon sauvegarde uniquement).",
    )
    args = parser.parse_args()
    main(show=args.show)
