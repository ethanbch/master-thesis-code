"""
03_event_study.py
-----------------
Dynamic DiD (event-study) estimation and coefficient plots.

Input:  data/intermediate/panel_monthly.parquet, data/results/matched_pairs.csv
Output: figures/event_study_synchronicity.png, figures/event_study_idiovol.png
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]

PANEL_PATH = ROOT / "data" / "intermediate" / "panel_monthly.parquet"
MATCHES_PATH = ROOT / "data" / "results" / "matched_pairs.csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

WINDOW = 6
REF_PERIOD = -1


# ── Load data ───────────────────────────────────────────────
print("Loading panel …")
panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])

print("Loading matched pairs …")
matches = pd.read_csv(MATCHES_PATH)
matches["event_date"] = pd.to_datetime(matches["event_date"])
matches = matches[matches["match_valid"] == True].copy()
print(f"  Valid pairs: {len(matches)}\n")


# ── Build stacked event-study panel ────────────────────────
print("Building stacked event-study panel …")
stacked_rows = []
pair_id = 0

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

        if len(sub) < 4:
            continue

        sub["pair_id"] = pair_id
        sub["Treat"] = treat_val
        sub["event_date"] = ev_date

        sub["tau"] = (sub["date"].dt.year - ev_date.year) * 12 + (
            sub["date"].dt.month - ev_date.month
        )

        sub = sub[(sub["tau"] >= -WINDOW) & (sub["tau"] <= WINDOW)].copy()
        sub["entity"] = f"{pair_id}_{ticker}"
        stacked_rows.append(sub)

    pair_id += 1

df = pd.concat(stacked_rows, ignore_index=True)
df = df.dropna(subset=["Synchronicity", "Idio_Vol"])
print(f"  Stacked panel: {len(df):,} rows, {df['entity'].nunique()} entities\n")

# ── Create interaction dummies ──────────────────────────────
taus = sorted(df["tau"].unique())
taus_used = [t for t in taus if t != REF_PERIOD]

for t in taus_used:
    col = f"D_tau_{t}" if t < 0 else f"D_tau_p{t}"
    df[col] = ((df["Treat"] == 1) & (df["tau"] == t)).astype(int)

dummy_cols = [f"D_tau_{t}" if t < 0 else f"D_tau_p{t}" for t in taus_used]

df = df.set_index(["entity", "tau"])


# ── Estimation function ────────────────────────────────────
def estimate_event_study(data, dep_var, label, dummy_cols, taus_used):
    """Estimate dynamic DiD and return coefficients + CI."""
    print(f"{'='*60}")
    print(f"  Event study: {label} ({dep_var})")
    print(f"{'='*60}")

    y = data[dep_var]
    X = data[dummy_cols]

    mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    print(res.summary)
    print()

    coefs = res.params[dummy_cols].values
    ci = res.conf_int().loc[dummy_cols]
    ci_lower = ci.iloc[:, 0].values
    ci_upper = ci.iloc[:, 1].values

    result_taus = []
    result_coefs = []
    result_lower = []
    result_upper = []

    idx = 0
    for t in sorted(taus_used + [REF_PERIOD]):
        result_taus.append(t)
        if t == REF_PERIOD:
            result_coefs.append(0.0)
            result_lower.append(0.0)
            result_upper.append(0.0)
        else:
            result_coefs.append(coefs[idx])
            result_lower.append(ci_lower[idx])
            result_upper.append(ci_upper[idx])
            idx += 1

    return result_taus, result_coefs, result_lower, result_upper


# ── Plot function ───────────────────────────────────────────
def plot_event_study(taus, coefs, ci_lower, ci_upper, label, filename):
    """Plot β_τ with 95% CI."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(taus, coefs, "o-", color="#2c3e50", linewidth=2, markersize=6, zorder=3)
    ax.fill_between(taus, ci_lower, ci_upper, alpha=0.2, color="#3498db", zorder=2)

    ax.axhline(y=0, color="grey", linewidth=0.8, linestyle="-")
    ax.axvline(
        x=-0.5,
        color="red",
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
        label="Treatment onset",
    )

    ax.set_xlabel("Event time τ (months)", fontsize=12)
    ax.set_ylabel(f"β_τ  ({label})", fontsize=12)
    ax.set_title(
        f"Event Study — {label}\n" f"Dynamic DiD with Entity & Time FE, clustered SE",
        fontsize=13,
    )
    ax.set_xticks(taus)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}\n")


# ── Run both estimations ───────────────────────────────────
taus_s, coefs_s, lo_s, hi_s = estimate_event_study(
    df, "Synchronicity", "Synchronicity", dummy_cols, taus_used
)
plot_event_study(
    taus_s, coefs_s, lo_s, hi_s, "Synchronicity", "event_study_synchronicity.png"
)

taus_v, coefs_v, lo_v, hi_v = estimate_event_study(
    df, "Idio_Vol", "Idiosyncratic Volatility", dummy_cols, taus_used
)
plot_event_study(
    taus_v, coefs_v, lo_v, hi_v, "Idiosyncratic Volatility", "event_study_idiovol.png"
)

print("Done.")
