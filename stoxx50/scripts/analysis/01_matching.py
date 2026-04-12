"""
01_matching.py  [STOXX 50 PROXY]
----------------------------------
Propensity Score Matching (PSM) for each ADD event (rank ≤ 50 proxy).

Input:  stoxx50/data/intermediate/features_at_event.csv
Output: stoxx50/data/results/matched_pairs.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SX50_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = SX50_ROOT / "data" / "intermediate" / "features_at_event.csv"
OUTPUT_PATH = SX50_ROOT / "data" / "results" / "matched_pairs.csv"

FEATURES = ["Log_MarketCap", "Momentum_12m", "Volatility_pre"]
CALIPER = 0.01


# ── Helper: Standardized Mean Difference ────────────────────
def smd(treated_val, control_val, pooled_std):
    if pooled_std == 0:
        return 0.0
    return (treated_val - control_val) / pooled_std


# ── Load data ───────────────────────────────────────────────
print("Loading features …")
df = pd.read_csv(INPUT_PATH)
df["event_date"] = pd.to_datetime(df["event_date"])
print(f"  {len(df):,} rows, {df['event_date'].nunique()} unique events.\n")

# ── Identify unique events ──────────────────────────────────
unique_events = (
    df[["event_date", "ticker_treated"]]
    .drop_duplicates()
    .sort_values("event_date")
    .reset_index(drop=True)
)
n_events = len(unique_events)
print(f"Processing {n_events} events …\n")

# ── Main matching loop ──────────────────────────────────────
results = []
n_valid = 0
n_attempted = 0

for i, row in unique_events.iterrows():
    ev_date = row["event_date"]
    tk_treated = row["ticker_treated"]
    n_attempted += 1

    mask = (df["event_date"] == ev_date) & (df["ticker_treated"] == tk_treated)
    sub = df.loc[mask].copy()

    treated = sub[sub["treated"] == 1]
    controls = sub[sub["treated"] == 0]

    if len(treated) == 0 or len(controls) < 2:
        continue

    scaler = StandardScaler()
    X_all = sub[FEATURES].values
    X_scaled = scaler.fit_transform(X_all)
    y_all = sub["treated"].values

    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        lr.fit(X_scaled, y_all)
        scores = lr.predict_proba(X_scaled)[:, 1]
    except Exception as e:
        print(f"  [{i+1:>3}/{n_events}] SKIP {tk_treated} ({ev_date.date()}): {e}")
        continue

    sub = sub.copy()
    sub["pscore"] = scores

    treated_sub = sub[sub["treated"] == 1]
    control_sub = sub[sub["treated"] == 0]

    score_treated = treated_sub["pscore"].values[0]
    control_scores = control_sub["pscore"].values
    distances = np.abs(control_scores - score_treated)
    best_idx = np.argmin(distances)
    best_distance = distances[best_idx]
    best_control = control_sub.iloc[best_idx]

    match_valid = best_distance <= CALIPER

    treated_feats = treated_sub[FEATURES].values[0]
    control_feats = best_control[FEATURES].values
    pooled_stds = sub[FEATURES].std().values

    smd_vals = [
        smd(treated_feats[j], control_feats[j], pooled_stds[j])
        for j in range(len(FEATURES))
    ]

    smd_warning = any(abs(s) > 0.1 for s in smd_vals)

    results.append(
        {
            "event_date": ev_date,
            "ticker_treated": tk_treated,
            "ticker_control": best_control["ticker"],
            "score_treated": score_treated,
            "score_control": best_control["pscore"],
            "distance": best_distance,
            "SMD_MarketCap": smd_vals[0],
            "SMD_Momentum": smd_vals[1],
            "SMD_Vol": smd_vals[2],
            "match_valid": match_valid,
        }
    )

    if match_valid:
        n_valid += 1

    if n_attempted % 10 == 0 or n_attempted == n_events:
        print(
            f"  [{n_attempted:>3}/{n_events}] "
            f"{ev_date.date()} | {tk_treated:12s} | "
            f"control={best_control['ticker']:12s} | "
            f"dist={best_distance:.6f} | "
            f"valid={match_valid}" + (" ⚠ SMD>0.1" if smd_warning else "")
        )

# ── Export ──────────────────────────────────────────────────
out = pd.DataFrame(results)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT_PATH, index=False)

rejection_rate = (n_attempted - n_valid) / n_attempted * 100 if n_attempted > 0 else 0

print(f"\n{'='*60}")
print(f"Total events attempted : {n_attempted}")
print(f"Valid matches          : {n_valid}")
print(f"Rejected (caliper)     : {n_attempted - n_valid}")
print(f"Rejection rate         : {rejection_rate:.1f}%")
print(f"\nOutput written to {OUTPUT_PATH}")
print(f"  {len(out)} rows")
