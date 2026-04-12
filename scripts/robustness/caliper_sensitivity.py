"""
caliper_sensitivity.py
----------------------
Sensitivity analysis: re-run matching + static DiD for three caliper values.

Input:  data/intermediate/features_at_event.csv, data/intermediate/panel_monthly.parquet
Output: data/results/robustness_caliper_results.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = ROOT / "data" / "intermediate" / "features_at_event.csv"
PANEL_PATH = ROOT / "data" / "intermediate" / "panel_monthly.parquet"
OUT_CSV = ROOT / "data" / "results" / "robustness_caliper_results.csv"

FEATURES = ["Log_MarketCap", "Momentum_12m", "Volatility_pre"]
CALIPERS = [0.005, 0.01, 0.05]
DID_WINDOW = 12

# ── Load data ───────────────────────────────────────────────
print("Loading features …")
feat_df = pd.read_csv(FEATURES_PATH)
feat_df["event_date"] = pd.to_datetime(feat_df["event_date"])

print("Loading panel …")
panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])

panel_by_ticker = {tk: grp.copy() for tk, grp in panel.groupby("ticker")}

unique_events = (
    feat_df[["event_date", "ticker_treated"]]
    .drop_duplicates()
    .sort_values("event_date")
    .reset_index(drop=True)
)
print(f"  {len(unique_events)} unique ADD events\n")


def run_matching(caliper: float) -> pd.DataFrame:
    """Run PSM for all events and return matched pairs for given caliper."""
    results = []
    for _, row in unique_events.iterrows():
        ev_date = row["event_date"]
        tk_treated = row["ticker_treated"]

        mask = (feat_df["event_date"] == ev_date) & (
            feat_df["ticker_treated"] == tk_treated
        )
        sub = feat_df.loc[mask].copy()
        treated = sub[sub["treated"] == 1]
        controls = sub[sub["treated"] == 0]

        if len(treated) == 0 or len(controls) < 2:
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(sub[FEATURES].values)
        y_all = sub["treated"].values

        try:
            lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
            lr.fit(X_scaled, y_all)
            scores = lr.predict_proba(X_scaled)[:, 1]
        except Exception:
            continue

        sub = sub.copy()
        sub["pscore"] = scores

        score_treated = sub.loc[sub["treated"] == 1, "pscore"].values[0]
        ctrl_sub = sub[sub["treated"] == 0]
        distances = np.abs(ctrl_sub["pscore"].values - score_treated)
        best_idx = np.argmin(distances)
        best_dist = distances[best_idx]
        best_ctrl = ctrl_sub.iloc[best_idx]

        results.append(
            {
                "event_date": ev_date,
                "ticker_treated": tk_treated,
                "ticker_control": best_ctrl["ticker"],
                "distance": best_dist,
                "match_valid": best_dist <= caliper,
            }
        )

    return pd.DataFrame(results)


def build_stacked_panel(matches: pd.DataFrame) -> pd.DataFrame:
    """Build stacked DiD panel for valid matched pairs."""
    rows = []
    pair_id = 0
    for _, m in matches[matches["match_valid"]].iterrows():
        ev_date = m["event_date"]
        date_start = ev_date - pd.DateOffset(months=DID_WINDOW)
        date_end = ev_date + pd.DateOffset(months=DID_WINDOW)

        for ticker, treat_val in [
            (m["ticker_treated"], 1),
            (m["ticker_control"], 0),
        ]:
            if ticker not in panel_by_ticker:
                continue
            sub = panel_by_ticker[ticker]
            sub = sub[(sub["date"] >= date_start) & (sub["date"] <= date_end)].copy()
            if len(sub) < 6:
                continue
            sub["Treat"] = treat_val
            sub["Post"] = (sub["date"] >= ev_date).astype(int)
            sub["Treat_Post"] = sub["Treat"] * sub["Post"]
            sub["entity"] = f"{pair_id}_{ticker}"
            sub["time_id"] = (sub["date"].dt.year - ev_date.year) * 12 + (
                sub["date"].dt.month - ev_date.month
            )
            rows.append(sub)
        pair_id += 1

    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    return df.dropna(subset=["Synchronicity", "Amihud", "Turnover"])


def run_did(df: pd.DataFrame) -> dict:
    """Estimate static DiD and return Treat_Post statistics."""
    df_idx = df.set_index(["entity", "time_id"])
    y = df_idx["Synchronicity"]
    X = df_idx[["Treat_Post", "Amihud", "Turnover"]]
    mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    ci = res.conf_int().loc["Treat_Post"]
    return {
        "beta": res.params["Treat_Post"],
        "se": res.std_errors["Treat_Post"],
        "t_stat": res.tstats["Treat_Post"],
        "p_value": res.pvalues["Treat_Post"],
        "ci_lower": ci.iloc[0],
        "ci_upper": ci.iloc[1],
    }


# ── Main loop over calipers ────────────────────────────────
summary_rows = []

for caliper in CALIPERS:
    tag = "(baseline)" if caliper == 0.01 else ""
    print(f"{'='*60}")
    print(f"  Caliper = {caliper} {tag}")
    print(f"{'='*60}")

    matches = run_matching(caliper)
    n_valid = matches["match_valid"].sum()
    n_attempted = len(matches)
    rej_rate = (n_attempted - n_valid) / n_attempted * 100 if n_attempted else 0
    print(
        f"  Pairs: {n_valid} valid / {n_attempted} attempted ({rej_rate:.1f}% rejected)"
    )

    df_panel = build_stacked_panel(matches)
    print(
        f"  Stacked panel: {len(df_panel):,} rows, {df_panel['entity'].nunique()} entities"
    )

    stats = run_did(df_panel)

    print(
        f"  β_DiD = {stats['beta']:+.4f}  SE = {stats['se']:.4f}"
        f"  t = {stats['t_stat']:.3f}  p = {stats['p_value']:.4f}"
        f"  95% CI [{stats['ci_lower']:+.4f}, {stats['ci_upper']:+.4f}]"
    )
    print()

    summary_rows.append(
        {
            "caliper": caliper,
            "baseline": caliper == 0.01,
            "N_pairs": int(n_valid),
            "beta_DiD": round(stats["beta"], 6),
            "std_err": round(stats["se"], 6),
            "t_stat": round(stats["t_stat"], 4),
            "p_value": round(stats["p_value"], 4),
            "ci_lower": round(stats["ci_lower"], 6),
            "ci_upper": round(stats["ci_upper"], 6),
        }
    )

# ── Export ──────────────────────────────────────────────────
results_df = pd.DataFrame(summary_rows)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUT_CSV, index=False)

print("=" * 60)
print("  ROBUSTNESS SUMMARY — Caliper sensitivity (Y = Synchronicity)")
print("=" * 60)
print(
    results_df[["caliper", "N_pairs", "beta_DiD", "std_err", "p_value"]].to_string(
        index=False
    )
)
print(f"\nSaved: {OUT_CSV}")
