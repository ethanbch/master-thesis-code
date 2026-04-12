"""
02_did_estimation.py
--------------------
Difference-in-Differences estimation using PanelOLS.

Input:  data/intermediate/panel_monthly.parquet, data/results/matched_pairs.csv
Output: data/results/did_results_synchronicity.csv,
        data/results/did_results_idiovol.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]

PANEL_PATH = ROOT / "data" / "intermediate" / "panel_monthly.parquet"
MATCHES_PATH = ROOT / "data" / "results" / "matched_pairs.csv"
OUT_SYNCH = ROOT / "data" / "results" / "did_results_synchronicity.csv"
OUT_IVOL = ROOT / "data" / "results" / "did_results_idiovol.csv"

WINDOW = 12  # months before / after event


# ── Load data ───────────────────────────────────────────────
print("Loading panel …")
panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
print(f"  Panel: {len(panel):,} rows, {panel['ticker'].nunique()} tickers")

# Winsorize Amihud at 1%/99% — keep parquet untouched, apply here before OLS
_p01 = panel["Amihud"].quantile(0.01)
_p99 = panel["Amihud"].quantile(0.99)
panel["Amihud"] = panel["Amihud"].clip(lower=_p01, upper=_p99)
print(f"  Amihud winsorized at [{_p01:.4f}, {_p99:.4f}]")

_iv_p01 = panel["Idio_Vol"].quantile(0.01)
_iv_p99 = panel["Idio_Vol"].quantile(0.99)
panel["Idio_Vol"] = panel["Idio_Vol"].clip(lower=_iv_p01, upper=_iv_p99)
print(f"  Idio_Vol winsorized at [{_iv_p01:.4f}, {_iv_p99:.4f}]")

print("Loading matched pairs …")
matches = pd.read_csv(MATCHES_PATH)
matches["event_date"] = pd.to_datetime(matches["event_date"])
matches = matches[matches["match_valid"] == True].copy()
print(f"  Valid pairs: {len(matches)}\n")

# ── Build stacked DiD panel ─────────────────────────────────
print("Building stacked DiD panel …")
stacked_rows = []
pair_id = 0
skipped = 0

for _, m in matches.iterrows():
    ev_date = m["event_date"]
    tk_treat = m["ticker_treated"]
    tk_ctrl = m["ticker_control"]

    date_start = ev_date - pd.DateOffset(months=WINDOW)
    date_end = ev_date + pd.DateOffset(months=WINDOW)

    for ticker, treat_val in [(tk_treat, 1), (tk_ctrl, 0)]:
        sub = panel[
            (panel["ticker"] == ticker)
            & (panel["date"] >= date_start)
            & (panel["date"] <= date_end)
        ].copy()

        if len(sub) < 6:
            skipped += 1
            continue

        sub["pair_id"] = pair_id
        sub["Treat"] = treat_val
        sub["Post"] = (sub["date"] >= ev_date).astype(int)
        sub["Treat_Post"] = sub["Treat"] * sub["Post"]
        sub["event_date"] = ev_date

        sub["rel_month"] = (sub["date"].dt.year - ev_date.year) * 12 + (
            sub["date"].dt.month - ev_date.month
        )

        sub["entity"] = f"{pair_id}_{ticker}"
        sub["time_id"] = sub["rel_month"]

        stacked_rows.append(sub)

    pair_id += 1

df = pd.concat(stacked_rows, ignore_index=True)
print(f"  Stacked panel: {len(df):,} rows, {df['entity'].nunique()} entities")
print(f"  Pairs skipped (too few obs): {skipped}")
print(f"  Rel months range: [{df['rel_month'].min()}, {df['rel_month'].max()}]\n")

df = df.dropna(subset=["Synchronicity", "Idio_Vol", "Amihud", "Avg_Volume"])
print(f"  After dropping NaN: {len(df):,} rows\n")

df = df.set_index(["entity", "time_id"])


# ── Estimation function ────────────────────────────────────
def run_panel_did(data, dep_var, label):
    """Run PanelOLS DiD with entity + time FE, clustered SE."""
    print(f"{'='*60}")
    print(f"  Dependent variable: {label} ({dep_var})")
    print(f"{'='*60}")

    y = data[dep_var]
    X = data[["Treat_Post", "Amihud", "Avg_Volume"]]

    mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    print(res.summary)
    print()

    params = res.params
    std_errors = res.std_errors
    tstats = res.tstats
    pvalues = res.pvalues
    ci = res.conf_int()

    results = pd.DataFrame(
        {
            "variable": params.index,
            "coef": params.values,
            "std_err": std_errors.values,
            "t_stat": tstats.values,
            "p_value": pvalues.values,
            "ci_lower": ci.iloc[:, 0].values,
            "ci_upper": ci.iloc[:, 1].values,
        }
    )

    return results


# ── Run estimations ─────────────────────────────────────────
res_synch = run_panel_did(df, "Synchronicity", "Synchronicity")
res_ivol = run_panel_did(df, "Idio_Vol", "Idiosyncratic Volatility")

# ── Export ──────────────────────────────────────────────────
OUT_SYNCH.parent.mkdir(parents=True, exist_ok=True)
res_synch.to_csv(OUT_SYNCH, index=False)
res_ivol.to_csv(OUT_IVOL, index=False)

print(f"\nResults exported:")
print(f"  {OUT_SYNCH}")
print(f"  {OUT_IVOL}")
