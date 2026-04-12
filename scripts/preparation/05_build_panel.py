"""
05_build_panel.py
-----------------
Build the monthly panel of market efficiency metrics.
For each (ticker, month), run OLS market model and compute:
R², Synchronicity, Idiosyncratic Volatility, Amihud, Turnover.

Input:  data/intermediate/prices_raw.parquet
Output: data/intermediate/panel_monthly.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

MIN_OBS = 15
MARKET_TICKER = "^STOXX"

# ============================================================
# 1. CHARGEMENT
# ============================================================
print(">>> Chargement des données...")
prices = pd.read_parquet(ROOT / "data" / "intermediate" / "prices_raw.parquet")

# Harmonisation des noms de colonnes
colmap = {c.lower(): c for c in prices.columns}

if "date" not in colmap or "ticker" not in colmap:
    raise KeyError(
        f"Colonnes attendues manquantes. Colonnes disponibles: {prices.columns.tolist()}"
    )

if "close" in colmap:
    close_col = colmap["close"]
elif "adj close" in colmap:
    close_col = colmap["adj close"]
else:
    raise KeyError(
        f"Aucune colonne de prix trouvée. Colonnes: {prices.columns.tolist()}"
    )

if "volume" in colmap:
    volume_col = colmap["volume"]
else:
    raise KeyError(
        f"Aucune colonne de volume trouvée. Colonnes: {prices.columns.tolist()}"
    )

date_col = colmap["date"]
ticker_col = colmap["ticker"]

if date_col != "date":
    prices = prices.rename(columns={date_col: "date"})
if ticker_col != "ticker":
    prices = prices.rename(columns={ticker_col: "ticker"})
if close_col != "Close":
    prices = prices.rename(columns={close_col: "Close"})
if volume_col != "Volume":
    prices = prices.rename(columns={volume_col: "Volume"})

prices["date"] = pd.to_datetime(prices["date"])
prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")
prices["Volume"] = pd.to_numeric(prices["Volume"], errors="coerce")
# Sanitisation stricte : supprime les prix et volumes invalides ou nuls
# (évite les RuntimeWarning sur np.log et les valeurs d'Amihud négatives)
prices = prices.dropna(subset=["date", "ticker", "Close", "Volume"]).copy()
prices = prices[(prices["Close"] > 0) & (prices["Volume"] > 0)].copy()
prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

if MARKET_TICKER not in prices["ticker"].unique():
    raise ValueError(
        f"Benchmark {MARKET_TICKER} absent de prices_raw.parquet. "
        "Vérifiez 02_collect_prices.py."
    )

print(f">>> {len(prices):,} observations, {prices['ticker'].nunique()} tickers")
print(f">>> Fenêtre : {prices['date'].min().date()} → {prices['date'].max().date()}")

# ============================================================
# 2. LOG-RETURNS JOURNALIERS
# ============================================================
print(">>> Calcul des log-returns journaliers...")

prices = prices.sort_values(["ticker", "date"])
prices["log_ret"] = prices.groupby("ticker")["Close"].transform(
    lambda s: np.log(s / s.shift(1))
)
# Remplace les infinis résiduels (ex: prix 0 ayant survécu) par NaN
prices["log_ret"] = prices["log_ret"].replace([np.inf, -np.inf], np.nan)
prices = prices.dropna(subset=["log_ret"])

# ============================================================
# 3. EXTRACTION DU BENCHMARK
# ============================================================
market = (
    prices[prices["ticker"] == MARKET_TICKER][["date", "log_ret"]]
    .rename(columns={"log_ret": "mkt_ret"})
    .set_index("date")
)
market = market[~market.index.duplicated(keep="first")]

# ============================================================
# 4. BOUCLE MENSUELLE PAR TICKER
# ============================================================
print(">>> Construction du panel mensuel (OLS par (ticker, mois))...")

prices["month"] = prices["date"].dt.to_period("M")
stocks = prices[prices["ticker"] != MARKET_TICKER].copy()

results = []

grouped = stocks.groupby(["ticker", "month"])
n_groups = len(grouped)
print(f"    {n_groups:,} groupes (ticker × mois) à traiter...")

for i, ((ticker, month), grp) in enumerate(grouped):
    if i % 10000 == 0:
        print(f"    {i:,}/{n_groups:,}...", end="\r")

    n = len(grp)
    if n < MIN_OBS:
        continue

    grp_idx = grp.set_index("date")
    merged = grp_idx[["log_ret", "Close", "Volume"]].join(market, how="inner")
    merged = merged.dropna()

    if len(merged) < MIN_OBS:
        continue

    y = merged["log_ret"].values
    x = merged["mkt_ret"].values

    n = len(y)
    X = np.column_stack([np.ones(n), x])
    try:
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        continue

    y_hat = X @ coeffs
    residuals = y - y_hat
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2 = max(0.0, min(1.0, r2))

    r2_clipped = np.clip(r2, 0.001, 0.999)
    synchronicity = np.log(r2_clipped / (1 - r2_clipped))

    idio_vol = float(np.std(residuals, ddof=1))

    dollar_volume = merged["Close"] * merged["Volume"]
    amihud_raw = np.abs(merged["log_ret"]) / dollar_volume
    amihud_raw = amihud_raw.replace([np.inf, -np.inf], np.nan)
    amihud = float(amihud_raw.mean()) * 1e6

    avg_volume = float(merged["Volume"].mean())
    date_ref = month.to_timestamp()

    results.append(
        {
            "date": date_ref,
            "ticker": ticker,
            "R2_raw": round(r2, 6),
            "Synchronicity": round(synchronicity, 6),
            "Idio_Vol": round(idio_vol, 6),
            "Amihud": round(amihud, 6) if not np.isnan(amihud) else np.nan,
            "Avg_Volume": round(avg_volume, 2),
            "N_obs": len(merged),
        }
    )

print(f"\n>>> {len(results):,} observations mensuelles construites")

# ============================================================
# 5. EXPORT
# ============================================================
panel = pd.DataFrame(results)
panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

print(panel.describe())
print(panel.head(5))

out_path = ROOT / "data" / "intermediate" / "panel_monthly.parquet"
panel.to_parquet(out_path, index=False)
print(f"\n✅ Export : {out_path}")
print(f"   Colonnes : {panel.columns.tolist()}")
print(f"   Tickers  : {panel['ticker'].nunique()}")
print(f"   Fenêtre  : {panel['date'].min().date()} → {panel['date'].max().date()}")
