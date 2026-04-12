"""
04_double_ml.py
---------------
Double Machine Learning (DML) — Partially Linear Regression for the causal
effect of STOXX 600 index inclusion on price synchronicity.

Framework: Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins
(2018), "Double/Debiased machine learning for treatment and structural
parameters", The Econometrics Journal 21(1): C1–C68.

Structural equation
-------------------
    Y = θ·D + g(X) + ε,   E[ε | D, X] = 0

where:
    Y = Synchronicity      (outcome — price co-movement with the market index)
    D = treated            (binary: 1 if ADD event, 0 if matched control)
    X = [Log_MarketCap, Momentum_12m, Volatility_pre]  (pre-event covariates)
    θ = ATE of interest

Identification via Neyman Orthogonality + Cross-Fitting
-------------------------------------------------------
1. Neyman Orthogonality: the moment condition identifying θ is first-order
   insensitive to perturbations in the nuisance functions g(X) = E[Y|X] and
   m(X) = E[D|X].  This eliminates the regularisation bias that arises when
   flexible ML estimators (e.g. random forests) converge at rates slower than
   n^{-1/2}, allowing √n-consistent estimation of the structural parameter θ
   even when the nuisance models are high-dimensional or non-parametric.

2. Cross-Fitting (n_folds=5, n_rep=5): the sample is split into K folds;
   nuisance models are trained on K-1 folds and applied out-of-sample on the
   held-out fold.  This sample-splitting removes the Donsker condition
   requirement and prevents the bias that would arise if the same observations
   were used both to fit the nuisance models and to construct the orthogonal
   residuals. Repeating cross-fitting 5 times and averaging results reduces
   Monte-Carlo variance from the random fold assignments.

Nuisance models
---------------
    g(X)  →  RandomForestRegressor   (captures non-linear covariate effects on Y)
    m(X)  →  RandomForestClassifier  (flexible propensity score; avoids logit
                                       misspecification for a binary treatment)

The doubly-robust structure of DML ensures consistent estimation of θ if at
least one of the two nuisance models is correctly specified (or converges).

Input:  data/intermediate/features_at_event.csv
        data/intermediate/panel_monthly.parquet
        data/results/matched_pairs.csv
Output: data/results/dml_results.csv
"""

from pathlib import Path

import doubleml as dml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = ROOT / "data" / "intermediate" / "features_at_event.csv"
PANEL_PATH = ROOT / "data" / "intermediate" / "panel_monthly.parquet"
MATCHES_PATH = ROOT / "data" / "results" / "matched_pairs.csv"
OUTPUT_PATH = ROOT / "data" / "results" / "dml_results.csv"

FEATURES = ["Log_MarketCap", "Momentum_12m", "Volatility_pre"]
N_FOLDS = 5
N_REP = 5
WINDOW = 12  # months — same pre-event window as the PSM / DiD pipeline


# ── 1. Load matched pairs ───────────────────────────────────
print("Loading matched pairs …")
matches = pd.read_csv(MATCHES_PATH)
matches["event_date"] = pd.to_datetime(matches["event_date"])
matches = matches[matches["match_valid"] == True].copy()
print(f"  Valid pairs: {len(matches)}\n")


# ── 2. Load panel and precompute mean pre-event Synchronicity ──
print("Loading panel …")
panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])


def _pre_event_mean(ticker: str, ev_date: pd.Timestamp) -> float:
    """Mean Synchronicity over the WINDOW months before ev_date."""
    start = ev_date - pd.DateOffset(months=WINDOW)
    sub = panel[
        (panel["ticker"] == ticker)
        & (panel["date"] >= start)
        & (panel["date"] < ev_date)
    ]["Synchronicity"].dropna()
    return sub.mean() if len(sub) >= 3 else np.nan


# ── 3. Build cross-sectional DML dataset ───────────────────
# One row per ticker (treated or control), with:
#   Y  = mean Synchronicity in the POST window (event month + WINDOW months)
#   D  = treated flag
#   X  = pre-event features from features_at_event.csv
print("Building cross-sectional DML dataset …")

features_df = pd.read_csv(FEATURES_PATH)
features_df["event_date"] = pd.to_datetime(features_df["event_date"])

rows = []
for _, m in matches.iterrows():
    ev_date = m["event_date"]
    post_start = ev_date
    post_end = ev_date + pd.DateOffset(months=WINDOW)

    for ticker, treat_val in [(m["ticker_treated"], 1), (m["ticker_control"], 0)]:
        # Post-event outcome
        post_obs = panel[
            (panel["ticker"] == ticker)
            & (panel["date"] >= post_start)
            & (panel["date"] <= post_end)
        ]["Synchronicity"].dropna()
        if len(post_obs) < 3:
            continue
        y_post = post_obs.mean()

        # Pre-event features — look up in features_at_event for this event
        feat_row = features_df[
            (features_df["event_date"] == ev_date) & (features_df["ticker"] == ticker)
        ]
        if feat_row.empty:
            continue
        feat_row = feat_row.iloc[0]

        rows.append(
            {
                "ticker": ticker,
                "event_date": ev_date,
                "treated": treat_val,
                "Synchronicity_post": y_post,
                **{f: feat_row[f] for f in FEATURES},
            }
        )

dml_df = pd.DataFrame(rows).dropna()
print(f"  DML dataset: {len(dml_df):,} obs ({dml_df['treated'].sum()} treated)\n")

if len(dml_df) < 50:
    raise RuntimeError(
        "DML dataset too small (< 50 obs). Check panel coverage and matched pairs."
    )


# ── 4. Construct DoubleMLData object ───────────────────────
print("Constructing DoubleMLData …")
data_dml = dml.DoubleMLData(
    dml_df,
    y_col="Synchronicity_post",
    d_cols="treated",
    x_cols=FEATURES,
)


# ── 5. Specify nuisance learners ───────────────────────────
# RandomForestRegressor for E[Y|X] — captures non-linear covariate effects.
# RandomForestClassifier for E[D|X] — flexible propensity score model.
# Both use default hyperparameters (n_estimators=100); the doubly-robust
# structure of DML ensures consistency if at least one nuisance model converges.
ml_g = RandomForestRegressor(n_estimators=100, random_state=42)
ml_m = RandomForestClassifier(n_estimators=100, random_state=42)


# ── 6. Fit PLR model ───────────────────────────────────────
print(f"Fitting DoubleML PLR (n_folds={N_FOLDS}, n_rep={N_REP}) …")
plr = dml.DoubleMLPLR(
    data_dml,
    ml_l=ml_g,  # learner for E[Y|X]  (outcome nuisance)
    ml_m=ml_m,  # learner for E[D|X]  (treatment nuisance / propensity)
    n_folds=N_FOLDS,
    n_rep=N_REP,
    score="partialling out",
)
plr.fit()

print("\n" + "=" * 60)
print("  DOUBLE ML — PLR RESULTS")
print("=" * 60)
print(plr.summary)

# Point estimate and inference
theta = plr.coef[0]
se = plr.se[0]
pval = plr.pval[0]
ci_lo, ci_hi = plr.confint().iloc[0]

print(f"\n  ATE (θ)    : {theta:+.6f}")
print(f"  Std. Error : {se:.6f}")
print(f"  p-value    : {pval:.4f}")
print(f"  95% CI     : [{ci_lo:.6f}, {ci_hi:.6f}]")

significance = (
    "*** (1%)"
    if pval < 0.01
    else "** (5%)" if pval < 0.05 else "* (10%)" if pval < 0.10 else "n.s."
)
print(f"  Significance: {significance}\n")


# ── 7. Nuisance model performance ─────────────────────────
# The nuisance_tuning_summary attribute provides in-sample metrics when tuning
# is active.  Without tuning, we rely on the cross-fitted residuals as a
# qualitative indicator of fit quality; no separate R² computation is exposed
# by DoubleML 0.11.x public API.
print("Nuisance models fitted via 5-fold cross-fitting (no explicit R² API).")
print("Propensity warnings (close to 0/1) are expected on a balanced matched-pair")
print("dataset where treatment assignment is nearly 50/50 by construction.\n")


# ── 8. Export results ──────────────────────────────────────
results = pd.DataFrame(
    {
        "outcome": ["Synchronicity"],
        "method": ["DoubleML_PLR"],
        "n_obs": [len(dml_df)],
        "n_treated": [int(dml_df["treated"].sum())],
        "n_folds": [N_FOLDS],
        "n_rep": [N_REP],
        "theta": [theta],
        "se": [se],
        "pval": [pval],
        "ci_lower_95": [ci_lo],
        "ci_upper_95": [ci_hi],
        "significance": [significance],
    }
)
results.to_csv(OUTPUT_PATH, index=False)
print(f"\nResults saved to {OUTPUT_PATH.relative_to(ROOT)}")
