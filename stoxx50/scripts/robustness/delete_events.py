"""
delete_events.py  [STOXX 50 PROXY]
------------------------------------
Replicates the full PSM + DiD pipeline for DELETE events (rank > 50 proxy).

ADD comparison stats are read dynamically from stoxx50 DiD results.

Input:  stoxx50/data/intermediate/events.csv,
        stoxx50/data/intermediate/prices_raw.parquet,
        stoxx50/data/intermediate/panel_monthly.parquet,
        stoxx50/data/results/did_results_synchronicity.csv (optional, for comparison)
Output: stoxx50/data/results/delete_matched_pairs.csv,
        stoxx50/data/results/delete_did_results.csv,
        stoxx50/data/results/add_vs_delete_comparison.csv,
        stoxx50/figures/event_study_delete_synchronicity.png
"""

import warnings
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SX50_ROOT = Path(__file__).resolve().parents[2]

EVENTS_PATH = SX50_ROOT / "data" / "intermediate" / "events.csv"
PRICES_PATH = SX50_ROOT / "data" / "intermediate" / "prices_raw.parquet"
PANEL_PATH = SX50_ROOT / "data" / "intermediate" / "panel_monthly.parquet"
OUT_MATCHES = SX50_ROOT / "data" / "results" / "delete_matched_pairs.csv"
OUT_DID = SX50_ROOT / "data" / "results" / "delete_did_results.csv"
OUT_FIG = SX50_ROOT / "figures" / "event_study_delete_synchronicity.png"
DID_SYNCH_PATH = SX50_ROOT / "data" / "results" / "did_results_synchronicity.csv"
DID_IVOL_PATH = SX50_ROOT / "data" / "results" / "did_results_idiovol.csv"
(SX50_ROOT / "figures").mkdir(exist_ok=True)

FEATURES = ["Log_MarketCap", "Momentum_12m", "Volatility_pre"]
CALIPER = 0.01
MIN_OBS = 200
WINDOW_DAYS = 365
DID_WINDOW = 12
ES_WINDOW = 6
REF_PERIOD = -1


# ── Read ADD β from stoxx50 DiD results (if available) ──────
def read_add_beta(path, dep_var_label):
    if path.exists():
        df = pd.read_csv(path)
        row = df[df["variable"] == "Treat_Post"]
        if len(row) > 0:
            return float(row["coef"].values[0]), float(row["p_value"].values[0])
    return None, None


ADD_BETA_SYNCH, ADD_P_SYNCH = read_add_beta(DID_SYNCH_PATH, "Synchronicity")
ADD_BETA_IVOL, ADD_P_IVOL = read_add_beta(DID_IVOL_PATH, "IVol")

print("ADD (SX50 proxy) reference betas:")
print(f"  Synchronicity: β = {ADD_BETA_SYNCH}  (p = {ADD_P_SYNCH})")
print(f"  IVol         : β = {ADD_BETA_IVOL}  (p = {ADD_P_IVOL})")
print("  (None means 02_did_estimation.py has not been run yet)\n")


# ═══════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 1 — Loading data")
print("=" * 60)

events = pd.read_csv(EVENTS_PATH)
events["date"] = pd.to_datetime(events["date"])
delete_events = events[events["event_type"] == "DELETE"].copy()
print(f"  DELETE events: {len(delete_events)}")

prices = pd.read_parquet(PRICES_PATH)
prices["date"] = pd.to_datetime(prices["date"])
prices.sort_values(["ticker", "date"], inplace=True)
prices["log_ret"] = prices.groupby("ticker")["close"].transform(
    lambda s: np.log(s / s.shift(1))
)
print(f"  Prices: {len(prices):,} rows, {prices['ticker'].nunique()} tickers")

panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
print(f"  Panel: {len(panel):,} rows\n")


# ═══════════════════════════════════════════════════════════
# STEP 2 — PSM FEATURES FOR DELETE EVENTS
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 2 — Building PSM features for DELETE events")
print("=" * 60)


def compute_features(prices_df, window_start, window_end):
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
    return agg[["Log_MarketCap", "Momentum_12m", "Volatility_pre"]].reset_index()


all_features = []
n_events = len(delete_events)

for i, (_, ev) in enumerate(delete_events.iterrows(), 1):
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
    all_features.append(feat)

    if i % 10 == 0 or i == n_events:
        print(f"  [{i:>3}/{n_events}] {event_date.date()} | {treated_ticker}")

features_df = pd.concat(all_features, ignore_index=True)
print(f"  Features: {len(features_df):,} rows\n")


# ═══════════════════════════════════════════════════════════
# STEP 3 — PSM MATCHING
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print(f"  STEP 3 — PSM matching (caliper={CALIPER})")
print("=" * 60)

unique_events = (
    features_df[["event_date", "ticker_treated"]]
    .drop_duplicates()
    .sort_values("event_date")
    .reset_index(drop=True)
)

match_results = []
n_valid = 0

for idx, row in unique_events.iterrows():
    ev_date = row["event_date"]
    tk_treated = row["ticker_treated"]

    mask = (features_df["event_date"] == ev_date) & (
        features_df["ticker_treated"] == tk_treated
    )
    sub = features_df.loc[mask].copy()
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

    treated_sub = sub[sub["treated"] == 1]
    control_sub = sub[sub["treated"] == 0]

    score_treated = treated_sub["pscore"].values[0]
    distances = np.abs(control_sub["pscore"].values - score_treated)
    best_idx = np.argmin(distances)
    best_distance = distances[best_idx]
    best_control = control_sub.iloc[best_idx]

    match_valid = best_distance <= CALIPER

    treated_feats = treated_sub[FEATURES].values[0]
    control_feats = best_control[FEATURES].values
    pooled_stds = sub[FEATURES].std().values
    smd_vals = [
        (
            (treated_feats[j] - control_feats[j]) / pooled_stds[j]
            if pooled_stds[j] > 0
            else 0.0
        )
        for j in range(len(FEATURES))
    ]

    match_results.append(
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

matches_df = pd.DataFrame(match_results)
OUT_MATCHES.parent.mkdir(parents=True, exist_ok=True)
matches_df.to_csv(OUT_MATCHES, index=False)

n_attempted = len(matches_df)
rejection_rate = (n_attempted - n_valid) / n_attempted * 100 if n_attempted else 0
print(f"  Total attempted : {n_attempted}")
print(f"  Valid matches   : {n_valid}")
print(f"  Rejection rate  : {rejection_rate:.1f}%")
print(f"  Saved: {OUT_MATCHES}\n")


# ═══════════════════════════════════════════════════════════
# STEP 4 — STATIC DiD
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 4 — Static DiD (DELETE pairs)")
print("=" * 60)

valid_matches = matches_df[matches_df["match_valid"]].copy()
print(f"  Using {len(valid_matches)} valid pairs\n")


def build_stacked_panel(matches_iter, panel_df, window_months):
    rows = []
    pair_id = 0
    for _, m in matches_iter.iterrows():
        ev_date = m["event_date"]
        tk_treat = m["ticker_treated"]
        tk_ctrl = m["ticker_control"]
        date_start = ev_date - pd.DateOffset(months=window_months)
        date_end = ev_date + pd.DateOffset(months=window_months)

        for ticker, treat_val in [(tk_treat, 1), (tk_ctrl, 0)]:
            sub = panel_df[
                (panel_df["ticker"] == ticker)
                & (panel_df["date"] >= date_start)
                & (panel_df["date"] <= date_end)
            ].copy()
            if len(sub) < 6:
                continue
            sub["pair_id"] = pair_id
            sub["Treat"] = treat_val
            sub["Post"] = (sub["date"] >= ev_date).astype(int)
            sub["Treat_Post"] = sub["Treat"] * sub["Post"]
            sub["entity"] = f"{pair_id}_{ticker}"
            sub["tau"] = (sub["date"].dt.year - ev_date.year) * 12 + (
                sub["date"].dt.month - ev_date.month
            )
            sub["time_id"] = sub["tau"]
            rows.append(sub)
        pair_id += 1

    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    return df.dropna(subset=["Synchronicity", "Idio_Vol", "Amihud", "Turnover"])


df_static = build_stacked_panel(valid_matches, panel, DID_WINDOW)
print(
    f"  Stacked panel: {len(df_static):,} rows, {df_static['entity'].nunique()} entities"
)

df_static_idx = df_static.set_index(["entity", "time_id"])
did_results_all = []

for dep_var, label in [("Synchronicity", "Synchronicity"), ("Idio_Vol", "Idio_Vol")]:
    y = df_static_idx[dep_var]
    X = df_static_idx[["Treat_Post", "Amihud", "Turnover"]]
    mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    print(f"\n  --- {label} ---")
    print(res.summary)

    for var in res.params.index:
        ci = res.conf_int().loc[var]
        did_results_all.append(
            {
                "dep_var": label,
                "variable": var,
                "coef": res.params[var],
                "std_err": res.std_errors[var],
                "t_stat": res.tstats[var],
                "p_value": res.pvalues[var],
                "ci_lower": ci.iloc[0],
                "ci_upper": ci.iloc[1],
            }
        )

did_df = pd.DataFrame(did_results_all)
did_df.to_csv(OUT_DID, index=False)
print(f"\n  Saved: {OUT_DID}\n")

del_row_synch = did_df[
    (did_df["dep_var"] == "Synchronicity") & (did_df["variable"] == "Treat_Post")
]
del_row_ivol = did_df[
    (did_df["dep_var"] == "Idio_Vol") & (did_df["variable"] == "Treat_Post")
]
del_beta_synch = float(del_row_synch["coef"].values[0])
del_p_synch = float(del_row_synch["p_value"].values[0])
del_beta_ivol = float(del_row_ivol["coef"].values[0])
del_p_ivol = float(del_row_ivol["p_value"].values[0])


# ═══════════════════════════════════════════════════════════
# STEP 5 — DYNAMIC DiD (EVENT STUDY)
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 5 — Dynamic DiD (event study, DELETE)")
print("=" * 60)

df_es = build_stacked_panel(valid_matches, panel, ES_WINDOW)
df_es = df_es[(df_es["tau"] >= -ES_WINDOW) & (df_es["tau"] <= ES_WINDOW)].copy()
print(f"  Event-study panel: {len(df_es):,} rows")

taus = sorted(df_es["tau"].unique())
taus_used = [t for t in taus if t != REF_PERIOD]

for t in taus_used:
    col = f"D_tau_{t}" if t < 0 else f"D_tau_p{t}"
    df_es[col] = ((df_es["Treat"] == 1) & (df_es["tau"] == t)).astype(int)

dummy_cols = [f"D_tau_{t}" if t < 0 else f"D_tau_p{t}" for t in taus_used]
df_es_idx = df_es.set_index(["entity", "tau"])

y = df_es_idx["Synchronicity"]
X = df_es_idx[dummy_cols]
mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
res = mod.fit(cov_type="clustered", cluster_entity=True)
print(res.summary)

coefs = res.params[dummy_cols].values
ci = res.conf_int().loc[dummy_cols]
ci_lower = ci.iloc[:, 0].values
ci_upper = ci.iloc[:, 1].values

result_taus, result_coefs, result_lower, result_upper = [], [], [], []
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

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(result_taus, result_coefs, "o-", color="#2c3e50", linewidth=2, markersize=6)
ax.fill_between(result_taus, result_lower, result_upper, alpha=0.2, color="#e74c3c")
ax.axhline(y=0, color="grey", linewidth=0.8)
ax.axvline(
    x=-0.5,
    color="red",
    linewidth=1.5,
    linestyle="--",
    alpha=0.7,
    label="Exclusion onset",
)
ax.set_xlabel("Event time τ (months)", fontsize=12)
ax.set_ylabel("β_τ  (Synchronicity)", fontsize=12)
ax.set_title(
    "Event Study — DELETE Events (EURO STOXX 50 Proxy)\n"
    "Dynamic DiD with Entity & Time FE, clustered SE",
    fontsize=13,
)
ax.set_xticks(result_taus)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {OUT_FIG}\n")


# ═══════════════════════════════════════════════════════════
# STEP 6 — COMPARISON TABLE: ADD vs DELETE
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 6 — ADD vs DELETE comparison")
print("=" * 60)

comparison_rows = [
    {
        "Event Type": "ADD (SX50 proxy inclusion)",
        "N_pairs": "—" if ADD_BETA_SYNCH is None else "(see matched_pairs.csv)",
        "β_DiD_Synch": ADD_BETA_SYNCH,
        "p_Synch": ADD_P_SYNCH,
        "β_DiD_IVol": ADD_BETA_IVOL,
        "p_IVol": ADD_P_IVOL,
    },
    {
        "Event Type": "DELETE (SX50 proxy exclusion)",
        "N_pairs": n_valid,
        "β_DiD_Synch": round(del_beta_synch, 4),
        "p_Synch": round(del_p_synch, 4),
        "β_DiD_IVol": round(del_beta_ivol, 4),
        "p_IVol": round(del_p_ivol, 4),
    },
]

comparison = pd.DataFrame(comparison_rows)
comparison_path = SX50_ROOT / "data" / "results" / "add_vs_delete_comparison.csv"
comparison.to_csv(comparison_path, index=False)

print(comparison.to_string(index=False))
print(f"\n  Saved: {comparison_path}")
print("\nDone.")
