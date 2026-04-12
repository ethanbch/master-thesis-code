"""
06_build_features.py
--------------------
For each ADD event, compute pre-event features (12-month window)
for the treated firm and every potential control ticker.

Features: Log_MarketCap, Momentum_12m, Volatility_pre

Input:  data/intermediate/prices_raw.parquet, data/intermediate/events.csv
Output: data/intermediate/features_at_event.csv


Due to the unavailability of historical Shares Outstanding data for the
entire sample, we follow standard market microstructure literature
and rely on the logarithmic Dollar Volume ($\ln(Price \times Volume)$) as
a proxy for firm size and market capitalization. As demonstrated by
Brennan, Chordia, and Subrahmanyam (1998) and Amihud (2002), Dollar
Volume exhibits a near-perfect correlation with market capitalization
and acts as a robust control for both firm size and liquidity
characteristics in cross-sectional asset pricing models.
"""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PRICES_PATH = ROOT / "data" / "intermediate" / "prices_raw.parquet"
EVENTS_PATH = ROOT / "data" / "intermediate" / "events.csv"
OUTPUT_PATH = ROOT / "data" / "intermediate" / "features_at_event.csv"
MIN_OBS = 200
WINDOW_DAYS = 365

# ── Load data ───────────────────────────────────────────────
print("Loading prices …")
prices = pd.read_parquet(PRICES_PATH)

# Normalise column names
colmap = {c.lower(): c for c in prices.columns}
rename = {}
for target, variants in [
    ("close", ["close", "adj close", "adj_close"]),
    ("volume", ["volume"]),
    ("date", ["date"]),
    ("ticker", ["ticker"]),
]:
    for v in variants:
        if v in colmap:
            rename[colmap[v]] = target
            break
prices.rename(columns=rename, inplace=True)

prices["date"] = pd.to_datetime(prices["date"])
prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce")
# Sanitisation stricte : supprime prix nuls/négatifs et volumes invalides
prices = prices[(prices["close"] > 0) & (prices["volume"] > 0)].copy()
prices.sort_values(["ticker", "date"], inplace=True)

print("Loading events …")
events = pd.read_csv(EVENTS_PATH)
events["date"] = pd.to_datetime(events["date"])
add_events = events[events["event_type"] == "ADD"].copy()
print(f"  {len(add_events)} ADD events to process.\n")

# ── Pre-compute daily log returns ───────────────────────────
prices["log_ret"] = prices.groupby("ticker")["close"].transform(
    lambda s: np.log(s / s.shift(1))
)
# Remplace les infinis résiduels par NaN puis les supprime
prices["log_ret"] = prices["log_ret"].replace([np.inf, -np.inf], np.nan)
prices = prices.dropna(subset=["log_ret"]).copy()


# ── Helper: compute features for all tickers in a window ────
def compute_features(prices_df, window_start, window_end):
    """Return a DataFrame with one row per ticker."""
    mask = (prices_df["date"] >= window_start) & (prices_df["date"] <= window_end)
    sub = prices_df.loc[mask].copy()

    agg = sub.groupby("ticker").agg(
        n_obs=("close", "size"),
        first_close=("close", "first"),
        last_close=("close", "last"),
        mean_volume=("volume", "mean"),
        vol_ret=("log_ret", "std"),
    )

    agg = agg[agg["n_obs"] >= MIN_OBS].copy()

    agg["Log_MarketCap"] = np.log(agg["last_close"] * agg["mean_volume"])
    agg["Momentum_12m"] = np.log(agg["last_close"] / agg["first_close"])
    agg["Volatility_pre"] = agg["vol_ret"]

    feat_cols = ["Log_MarketCap", "Momentum_12m", "Volatility_pre"]
    agg = agg[feat_cols].replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols)
    return agg.reset_index()


# ── Main loop over ADD events ──────────────────────────────
all_rows = []
n_events = len(add_events)

for i, (_, ev) in enumerate(add_events.iterrows(), 1):
    event_date = ev["date"]
    treated_ticker = ev["ticker"]
    window_start = event_date - timedelta(days=WINDOW_DAYS)
    window_end = event_date - timedelta(days=1)

    feat = compute_features(prices, window_start, window_end)
    if feat.empty:
        continue

    feat["event_date"] = event_date
    feat["ticker_treated"] = treated_ticker
    feat["treated"] = (feat["ticker"] == treated_ticker).astype(int)

    all_rows.append(feat)

    if i % 20 == 0 or i == n_events:
        print(
            f"  [{i:>3}/{n_events}] event {event_date.date()} | "
            f"{treated_ticker:12s} | pool size {len(feat)}"
        )

# ── Concat & export ─────────────────────────────────────────
result = pd.concat(all_rows, ignore_index=True)
result = result[
    [
        "event_date",
        "ticker_treated",
        "ticker",
        "Log_MarketCap",
        "Momentum_12m",
        "Volatility_pre",
        "treated",
    ]
]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_PATH, index=False)
print(f"\nDone — {len(result):,} rows written to {OUTPUT_PATH}")
print(f"  Unique events: {result['event_date'].nunique()}")
print(f"  Unique tickers in pool: {result['ticker'].nunique()}")
