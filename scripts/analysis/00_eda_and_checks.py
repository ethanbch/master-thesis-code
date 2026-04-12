"""
00_eda_and_checks.py
--------------------
Exploratory Data Analysis (EDA) and data quality checks.
Run this script BEFORE 01_matching.py to validate cleaned data.

Inputs:
  data/intermediate/panel_monthly.parquet  — monthly efficiency metrics
  data/intermediate/features_at_event.csv  — PSM control variables

Outputs (figures/):
  psm_features_dist.png        — KDE distributions by treated/control
  efficiency_metrics_boxplots.png — boxplots of efficiency metrics
  correlation_matrix.png       — Pearson correlation heatmap
  events_per_year.png          — ADD events count by year
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Academic style (publication-ready)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    }
)

# ============================================================
# 1. LOADING
# ============================================================
print("=" * 60)
print("DATA LOADING")
print("=" * 60)

panel = pd.read_parquet(ROOT / "data" / "intermediate" / "panel_monthly.parquet")
panel["date"] = pd.to_datetime(panel["date"])

feat = pd.read_csv(ROOT / "data" / "intermediate" / "features_at_event.csv")
feat["event_date"] = pd.to_datetime(feat["event_date"])

print(f"panel_monthly    : {panel.shape[0]:,} rows × {panel.shape[1]} columns")
print(f"features_at_event: {feat.shape[0]:,} rows × {feat.shape[1]} columns")

# ============================================================
# 2. STATISTICAL CHECKS
# ============================================================
print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS — panel_monthly")
print("=" * 60)
panel_cols = ["R2_raw", "Synchronicity", "Idio_Vol", "Amihud", "Avg_Volume"]
print(panel[panel_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

# Residual NaNs and Infs
nan_count = panel[panel_cols].isna().sum()
inf_count = panel[panel_cols].isin([np.inf, -np.inf]).sum()
print("\nResidual NaNs:")
print(nan_count[nan_count > 0].to_string() or "  None")
print("Residual Infs:")
print(inf_count[inf_count > 0].to_string() or "  None")

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS — features_at_event")
print("=" * 60)
feat_cols = ["Log_MarketCap", "Momentum_12m", "Volatility_pre"]
print(feat[feat_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

nan_count_f = feat[feat_cols].isna().sum()
inf_count_f = feat[feat_cols].isin([np.inf, -np.inf]).sum()
print("\nResidual NaNs:")
print(nan_count_f[nan_count_f > 0].to_string() or "  None")
print("Residual Infs:")
print(inf_count_f[inf_count_f > 0].to_string() or "  None")

print(f"\nUnique ADD events: {feat['event_date'].nunique()}")
print(f"Unique treated tickers: {feat['ticker_treated'].nunique()}")
print(
    f"Average control pool size: "
    f"{feat.groupby('event_date').size().mean():.0f} tickers/event"
)

# ============================================================
# 3. FIGURE 1 — PSM Distributions (treated vs control)
# ============================================================
print("\n[1/4] Generating psm_features_dist.png ...")

# Deduplicate: one row per (event_date, ticker) for KDEs
feat_dedup = feat.drop_duplicates(subset=["event_date", "ticker"])
treated = feat_dedup[feat_dedup["treated"] == 1]
control = feat_dedup[feat_dedup["treated"] == 0]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(
    "Matching Variable Distributions (PSM)\nTreated Group vs Control Pool",
    fontsize=13,
    y=1.02,
)

psm_vars = {
    "Log_MarketCap": "Log Market Cap",
    "Momentum_12m": "12-month Momentum",
    "Volatility_pre": "Pre-event Volatility",
}

for ax, (col, label) in zip(axes, psm_vars.items()):
    # Clip to 1-99 percentiles for display (without altering data)
    lo, hi = feat_dedup[col].quantile([0.01, 0.99])
    data_c = control[col].clip(lo, hi)
    data_t = treated[col].clip(lo, hi)

    sns.kdeplot(data_c, ax=ax, label="Control", fill=True, alpha=0.35, color="#4878CF")
    sns.kdeplot(
        data_t, ax=ax, label="Treated (ADD)", fill=True, alpha=0.5, color="#D65F5F"
    )
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=9)

plt.tight_layout()
out = FIG_DIR / "psm_features_dist.png"
fig.savefig(out)
plt.close(fig)
print(f"   → Saved: {out}")

# ============================================================
# 4. FIGURE 2 — Efficiency Metrics Boxplots
# ============================================================
print("[2/4] Generating efficiency_metrics_boxplots.png ...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    "Market Efficiency Metrics Distribution\n(monthly panel)", fontsize=13, y=1.02
)

# Synchronicity: linear scale
ax_sync = axes[0]
display = panel["Synchronicity"].clip(
    panel["Synchronicity"].quantile(0.01), panel["Synchronicity"].quantile(0.99)
)
sns.boxplot(y=display, ax=ax_sync, color="#4878CF", fliersize=2, linewidth=0.8)
ax_sync.set_title("Synchronicity", fontsize=11)
ax_sync.set_ylabel("Value", fontsize=10)
ax_sync.set_xlabel("")

# Idio_Vol: linear scale
ax_idio = axes[1]
display_i = panel["Idio_Vol"].clip(0, panel["Idio_Vol"].quantile(0.99))
sns.boxplot(y=display_i, ax=ax_idio, color="#6ACC65", fliersize=2, linewidth=0.8)
ax_idio.set_title("Idio_Vol", fontsize=11)
ax_idio.set_ylabel("Value", fontsize=10)
ax_idio.set_xlabel("")

# Amihud: log scale (positive values only)
ax_ami = axes[2]
amihud_pos = panel["Amihud"].dropna()
amihud_pos = amihud_pos[amihud_pos > 0]
amihud_clipped = amihud_pos.clip(amihud_pos.quantile(0.01), amihud_pos.quantile(0.99))
sns.boxplot(
    y=np.log10(amihud_clipped), ax=ax_ami, color="#D65F5F", fliersize=2, linewidth=0.8
)
ax_ami.set_title("Amihud (log₁₀ scale)", fontsize=11)
ax_ami.set_ylabel("log₁₀(Amihud)", fontsize=10)
ax_ami.set_xlabel("")

plt.tight_layout()
out = FIG_DIR / "efficiency_metrics_boxplots.png"
fig.savefig(out)
plt.close(fig)
print(f"   → Saved: {out}")

# ============================================================
# 5. FIGURE 3 — Correlation Matrix
# ============================================================
print("[3/4] Generating correlation_matrix.png ...")

# Winsorize at 1%-99% to prevent correlation spikes from outliers
corr_df = panel[panel_cols].copy()
for col in panel_cols:
    lo, hi = corr_df[col].quantile([0.01, 0.99])
    corr_df[col] = corr_df[col].clip(lo, hi)

corr_matrix = corr_df.corr(method="pearson")

# Clean labels
label_map = {
    "R2_raw": "R²",
    "Synchronicity": "Synchronicity",
    "Idio_Vol": "Idio Vol",
    "Amihud": "Amihud",
    "Avg_Volume": "Avg Volume",
}
corr_matrix = corr_matrix.rename(index=label_map, columns=label_map)

fig, ax = plt.subplots(figsize=(7, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # upper triangle mask
sns.heatmap(
    corr_matrix,
    ax=ax,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
ax.set_title(
    "Pearson Correlation Matrix\n(efficiency metrics, winsorized p1-p99)", fontsize=11
)
plt.tight_layout()
out = FIG_DIR / "correlation_matrix.png"
fig.savefig(out)
plt.close(fig)
print(f"   → Saved: {out}")

# ============================================================
# 6. FIGURE 4 — ADD Events per Year
# ============================================================
print("[4/4] Generating events_per_year.png ...")

# An ADD event = a row where treated == 1 in features_at_event
add_events = feat[feat["treated"] == 1].copy()
add_events["year"] = add_events["event_date"].dt.year
events_per_year = add_events.groupby("year").size().reset_index(name="count")

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(
    events_per_year["year"],
    events_per_year["count"],
    color="#4878CF",
    edgecolor="white",
    linewidth=0.6,
)
# Annotate each bar
for bar, val in zip(bars, events_per_year["count"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        str(val),
        ha="center",
        va="bottom",
        fontsize=9,
    )
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Number of Inclusions (ADD)", fontsize=11)
ax.set_title("Temporal Distribution of STOXX Europe 600 Inclusion Events", fontsize=12)
ax.set_xticks(events_per_year["year"])
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
out = FIG_DIR / "events_per_year.png"
fig.savefig(out)
plt.close(fig)
print(f"   → Saved: {out}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)
print(f"4 plots saved in: {FIG_DIR}/")
print("  • psm_features_dist.png")
print("  • efficiency_metrics_boxplots.png")
print("  • correlation_matrix.png")
print("  • events_per_year.png")
