"""
placebo_dml.py
--------------
Permutation (randomization inference) test for the Double ML estimate.

Rationale
---------
The internal guarantees of DML (Neyman orthogonality + cross-fitting) protect
against regularisation bias but do not exclude the possibility that the observed
θ = +0.119 arises by chance from the particular pairing of treated and control
firms.  A permutation test directly addresses this: under the sharp null
hypothesis H₀: θ = 0, randomly shuffling the treatment labels within each
matched pair destroys the causal signal while preserving the covariate structure
and the by-pair matching, yielding an exact finite-sample reference distribution.

Procedure
---------
For each permutation iteration:
  1. For every matched pair, flip a fair coin: with probability 0.5, swap the
     `treated` labels between the two observations. This preserves the
     within-pair balance (each pair always contributes exactly one treated and
     one control observation) while destroying the true treatment assignment.
  2. Re-fit the DoubleML PLR on the permuted dataset (n_folds=5, n_rep=1 for
     speed; fewer reps are sufficient for the reference distribution).
  3. Record the permuted θ̃.

The empirical p-value is the share of permuted θ̃ values that are ≥ θ_obs
(one-sided, since the observed effect is positive).

Reference: Good (2005), "Permutation, Parametric and Bootstrap Tests of
Hypotheses", Springer.

Input:   data/results/matched_pairs.csv
         data/intermediate/features_at_event.csv
         data/intermediate/panel_monthly.parquet
Output:  data/results/placebo_dml_results.csv
         figures/placebo_dml_distribution.png
"""

import warnings
from pathlib import Path

import doubleml as dml_lib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = ROOT / "data" / "intermediate" / "features_at_event.csv"
PANEL_PATH = ROOT / "data" / "intermediate" / "panel_monthly.parquet"
MATCHES_PATH = ROOT / "data" / "results" / "matched_pairs.csv"
OUT_CSV = ROOT / "data" / "results" / "placebo_dml_results.csv"
OUT_FIG = ROOT / "figures" / "placebo_dml_distribution.png"
(ROOT / "figures").mkdir(exist_ok=True)

FEATURES = ["Log_MarketCap", "Momentum_12m", "Volatility_pre"]
TRUE_THETA = 0.1187  # observed DML estimate from 04_double_ml.py
N_ITER = 200  # 200 permutations — sufficient for p-value precision ±0.03
N_FOLDS = 5
N_REP = 1  # 1 rep per permutation for speed (200 × 5-fold fits)
WINDOW = 12
RNG_SEED = 42


# ── 1. Build the base DML dataset (same logic as 04_double_ml.py) ─────────
print("Loading data …")
matches = pd.read_csv(MATCHES_PATH)
matches["event_date"] = pd.to_datetime(matches["event_date"])
matches = matches[matches["match_valid"] == True].copy().reset_index(drop=True)
print(f"  Valid pairs: {len(matches)}")

panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])

features_df = pd.read_csv(FEATURES_PATH)
features_df["event_date"] = pd.to_datetime(features_df["event_date"])

print("Building cross-sectional DML dataset …")
rows = []
for _, m in matches.iterrows():
    ev_date = m["event_date"]
    post_end = ev_date + pd.DateOffset(months=WINDOW)

    for ticker, treat_val in [(m["ticker_treated"], 1), (m["ticker_control"], 0)]:
        post_obs = panel[
            (panel["ticker"] == ticker)
            & (panel["date"] >= ev_date)
            & (panel["date"] <= post_end)
        ]["Synchronicity"].dropna()
        if len(post_obs) < 3:
            continue

        feat_row = features_df[
            (features_df["event_date"] == ev_date) & (features_df["ticker"] == ticker)
        ]
        if feat_row.empty:
            continue

        rows.append(
            {
                "pair_id": _,
                "ticker": ticker,
                "event_date": ev_date,
                "treated": treat_val,
                "Synchronicity_post": post_obs.mean(),
                **{f: feat_row.iloc[0][f] for f in FEATURES},
            }
        )

base_df = pd.DataFrame(rows).dropna().reset_index(drop=True)
n_obs = len(base_df)
n_treated = int(base_df["treated"].sum())
print(
    f"  Base dataset: {n_obs:,} obs ({n_treated} treated, {n_obs - n_treated} control)\n"
)


# ── 2. Fit observed model ──────────────────────────────────
def fit_dml(df: pd.DataFrame, seed: int, n_rep: int = N_REP) -> float:
    """Fit DoubleML PLR and return the ATE estimate θ."""
    data_obj = dml_lib.DoubleMLData(
        df,
        y_col="Synchronicity_post",
        d_cols="treated",
        x_cols=FEATURES,
    )
    plr = dml_lib.DoubleMLPLR(
        data_obj,
        ml_l=RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=-1),
        ml_m=RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1),
        n_folds=N_FOLDS,
        n_rep=n_rep,
        score="partialling out",
    )
    plr.fit()
    return float(plr.coef[0])


print("Fitting observed DML model (n_rep=5 for reference) …")
theta_obs = fit_dml(base_df, seed=42, n_rep=5)
print(f"  Observed θ = {theta_obs:+.6f}  (stored TRUE_THETA = {TRUE_THETA})\n")


# ── 3. Permutation loop ────────────────────────────────────
rng = np.random.default_rng(RNG_SEED)
permuted_thetas = []

print(f"Running {N_ITER} permutation iterations (n_folds={N_FOLDS}, n_rep={N_REP}) …")
print("(50-tree forests, n_jobs=-1 — expect ~5–10 min total)\n")

for it in range(N_ITER):
    perm_df = base_df.copy()

    # Within each matched pair, flip labels with probability 0.5
    for pid in perm_df["pair_id"].unique():
        mask = perm_df["pair_id"] == pid
        if rng.random() < 0.5:
            # Swap: 1→0 and 0→1 within the pair
            perm_df.loc[mask, "treated"] = 1 - perm_df.loc[mask, "treated"].values

    try:
        theta_perm = fit_dml(perm_df, seed=int(rng.integers(1_000_000)))
    except Exception:
        theta_perm = np.nan

    permuted_thetas.append(theta_perm)

    if (it + 1) % 10 == 0 or it == 0:
        valid_so_far = [t for t in permuted_thetas if not np.isnan(t)]
        emp_p = (
            np.mean([t >= theta_obs for t in valid_so_far]) if valid_so_far else np.nan
        )
        print(
            f"  [{it+1:>3}/{N_ITER}]  θ_perm = {theta_perm:+.4f}  (running emp. p = {emp_p:.3f})"
        )


# ── 4. Summary statistics ──────────────────────────────────
thetas = np.array([t for t in permuted_thetas if not np.isnan(t)])
empirical_p = float(np.mean(thetas >= theta_obs))

print(f"\n{'='*60}")
print(f"  PERMUTATION TEST — DML (DoubleML PLR)")
print(f"{'='*60}")
print(f"  Observed θ            = {theta_obs:+.6f}")
print(f"  Permuted θ — mean     = {thetas.mean():+.4f}")
print(f"  Permuted θ — median   = {np.median(thetas):+.4f}")
print(f"  Permuted θ — std      = {thetas.std():.4f}")
print(f"  Permuted θ — p95      = {np.percentile(thetas, 95):+.4f}")
print(f"  Valid iterations      = {len(thetas)} / {N_ITER}")
print(f"  Empirical p-value     = {empirical_p:.4f}  (share of θ_perm ≥ θ_obs)")
sig = (
    "*** (1%)"
    if empirical_p < 0.01
    else (
        "** (5%)" if empirical_p < 0.05 else "* (10%)" if empirical_p < 0.10 else "n.s."
    )
)
print(f"  Significance          : {sig}")
print(f"{'='*60}\n")


# ── 5. Save results ────────────────────────────────────────
out = pd.DataFrame(
    {
        "iteration": range(N_ITER),
        "theta_permuted": permuted_thetas,
    }
)
out.to_csv(OUT_CSV, index=False)
print(f"Results saved to {OUT_CSV.relative_to(ROOT)}")


# ── 6. Plot ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

ax.hist(
    thetas,
    bins=40,
    color="#b0c4de",
    edgecolor="white",
    linewidth=0.5,
    density=True,
    alpha=0.85,
    label=f"Permuted θ distribution ({len(thetas)} iters)",
)
ax.axvline(
    theta_obs,
    color="#c0392b",
    linewidth=2,
    linestyle="--",
    label=f"Observed θ = {theta_obs:+.4f}",
)
ax.axvline(
    thetas.mean(),
    color="#2c3e50",
    linewidth=1.2,
    linestyle=":",
    label=f"Mean permuted θ = {thetas.mean():+.4f}",
)

ax.set_xlabel("θ (ATE — DoubleML PLR)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title(
    f"Permutation Test — DoubleML PLR ({len(thetas)} iterations)\n"
    f"Empirical p-value = {empirical_p:.3f}  ({sig})",
    fontsize=13,
)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved:  {OUT_FIG.relative_to(ROOT)}")
