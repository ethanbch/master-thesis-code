"""
placebo_test.py
---------------
Placebo (randomization inference) test for the DiD estimate.

Input:  data/intermediate/panel_monthly.parquet, data/results/matched_pairs.csv
Output: data/results/placebo_results.csv, figures/placebo_distribution.png
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

PANEL_PATH = ROOT / "data" / "intermediate" / "panel_monthly.parquet"
MATCHES_PATH = ROOT / "data" / "results" / "matched_pairs.csv"
OUT_CSV = ROOT / "data" / "results" / "placebo_results.csv"
OUT_FIG = ROOT / "figures" / "placebo_distribution.png"
(ROOT / "figures").mkdir(exist_ok=True)

TRUE_BETA = -0.115
N_ITER = 500
WINDOW = 12
RNG_SEED = 42

# ── Load data ───────────────────────────────────────────────
print("Loading panel …")
panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
print(f"  Panel: {len(panel):,} rows, {panel['ticker'].nunique()} tickers")

print("Loading matched pairs …")
matches = pd.read_csv(MATCHES_PATH)
matches["event_date"] = pd.to_datetime(matches["event_date"])
matches = matches[matches["match_valid"]].copy().reset_index(drop=True)
n_pairs = len(matches)
print(f"  Valid pairs: {n_pairs}\n")

panel_by_ticker = {tk: grp for tk, grp in panel.groupby("ticker")}


def build_stacked_panel(placebo_dates: np.ndarray) -> pd.DataFrame:
    """Build the stacked DiD panel using given placebo event dates."""
    rows = []
    for i, m in matches.iterrows():
        pdate = pd.Timestamp(placebo_dates[i])
        tk_treat = m["ticker_treated"]
        tk_ctrl = m["ticker_control"]

        date_start = pdate - pd.DateOffset(months=WINDOW)
        date_end = pdate + pd.DateOffset(months=WINDOW)

        for ticker, treat_val in [(tk_treat, 1), (tk_ctrl, 0)]:
            if ticker not in panel_by_ticker:
                continue
            tk_data = panel_by_ticker[ticker]
            sub = tk_data[
                (tk_data["date"] >= date_start) & (tk_data["date"] <= date_end)
            ].copy()

            if len(sub) < 6:
                continue

            sub["Treat"] = treat_val
            sub["Post"] = (sub["date"] >= pdate).astype(int)
            sub["Treat_Post"] = sub["Treat"] * sub["Post"]
            sub["entity"] = f"{i}_{ticker}"
            sub["time_id"] = (sub["date"].dt.year - pdate.year) * 12 + (
                sub["date"].dt.month - pdate.month
            )
            rows.append(sub)

    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=["Synchronicity", "Amihud", "Turnover"])
    return df


def draw_placebo_dates(rng: np.random.Generator) -> np.ndarray:
    """For each pair, draw a date in [event_date-12m, event_date-2m]."""
    dates = []
    for _, m in matches.iterrows():
        ev = m["event_date"]
        lo = ev - pd.DateOffset(months=12)
        hi = ev - pd.DateOffset(months=2)
        candidates = pd.date_range(lo.replace(day=1), hi.replace(day=1), freq="MS")
        if len(candidates) == 0:
            dates.append(lo)
        else:
            dates.append(candidates[rng.integers(len(candidates))])
    return np.array(dates)


# ── Main loop ───────────────────────────────────────────────
rng = np.random.default_rng(RNG_SEED)
results = []

print(f"Running {N_ITER} placebo iterations …\n")

for it in range(N_ITER):
    placebo_dates = draw_placebo_dates(rng)
    df = build_stacked_panel(placebo_dates)

    if df.empty or df["entity"].nunique() < 10:
        results.append({"iteration": it, "beta_placebo": np.nan, "p_value": np.nan})
        continue

    df = df.set_index(["entity", "time_id"])

    try:
        y = df["Synchronicity"]
        X = df[["Treat_Post", "Amihud", "Turnover"]]
        mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        beta = res.params["Treat_Post"]
        pval = res.pvalues["Treat_Post"]
    except Exception:
        beta, pval = np.nan, np.nan

    results.append({"iteration": it, "beta_placebo": beta, "p_value": pval})

    if (it + 1) % 50 == 0 or it == 0:
        print(f"  [{it+1:>3}/{N_ITER}]  β_placebo = {beta:+.4f}  (p = {pval:.3f})")

# ── Results ─────────────────────────────────────────────────
res_df = pd.DataFrame(results)
res_df.to_csv(OUT_CSV, index=False)

betas = res_df["beta_placebo"].dropna()
empirical_p = (betas < TRUE_BETA).mean()

print(f"\n{'='*60}")
print(f"  Placebo test — {len(betas)} valid iterations out of {N_ITER}")
print(f"  True β_DiD           = {TRUE_BETA:.4f}")
print(f"  Mean β_placebo       = {betas.mean():+.4f}")
print(f"  Median β_placebo     = {betas.median():+.4f}")
print(f"  Std β_placebo        = {betas.std():.4f}")
print(f"  Empirical p-value    = {empirical_p:.4f}")
print(f"    (share of placebo β < true β)")
print(f"{'='*60}\n")

# ── Plot ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

ax.hist(
    betas,
    bins=40,
    color="#b0c4de",
    edgecolor="white",
    linewidth=0.5,
    density=True,
    alpha=0.85,
    label="Placebo β distribution",
)
ax.axvline(
    TRUE_BETA,
    color="#c0392b",
    linewidth=2,
    linestyle="--",
    label=f"True β = {TRUE_BETA:.3f}",
)
ax.axvline(
    betas.mean(),
    color="#2c3e50",
    linewidth=1.2,
    linestyle=":",
    label=f"Mean placebo β = {betas.mean():.3f}",
)

ax.set_xlabel("β (Treat × Post)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title(
    f"Placebo Test — Distribution of β_placebo ({len(betas)} iterations)\n"
    f"Empirical p-value = {empirical_p:.3f}",
    fontsize=13,
)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved: {OUT_FIG}")
print(f"Results saved: {OUT_CSV}")
