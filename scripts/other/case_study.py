from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(".")
events = pd.read_csv(root / "data/intermediate/events.csv")
events["date"] = pd.to_datetime(events["date"])
panel = pd.read_parquet(root / "data/intermediate/panel_monthly.parquet")
panel["date"] = pd.to_datetime(panel["date"])

required = ["ticker", "date", "Synchronicity", "Idio_Vol"]
for c in required:
    if c not in panel.columns:
        raise RuntimeError(f"Missing column in panel_monthly: {c}")


def extract_event_window(ev_row, window=12):
    tk = ev_row["ticker"]
    ev_date = pd.Timestamp(ev_row["date"])
    sub = panel[panel["ticker"] == tk].copy()
    if sub.empty:
        return None
    sub["tau"] = (sub["date"].dt.year - ev_date.year) * 12 + (
        sub["date"].dt.month - ev_date.month
    )
    sub = sub[(sub["tau"] >= -window) & (sub["tau"] <= window)].copy()
    if sub.empty:
        return None

    pre = sub[sub["tau"] <= -1]
    post = sub[sub["tau"] >= 0]

    syn_pre = pre["Synchronicity"].mean()
    syn_post = post["Synchronicity"].mean()
    iv_pre = pre["Idio_Vol"].mean()
    iv_post = post["Idio_Vol"].mean()

    n_pre = pre["Synchronicity"].notna().sum()
    n_post = post["Synchronicity"].notna().sum()

    if min(n_pre, n_post) < 8:
        return None
    if pd.isna(syn_pre) or pd.isna(syn_post) or pd.isna(iv_pre) or pd.isna(iv_post):
        return None

    out = {
        "event_type": ev_row["event_type"],
        "ticker": tk,
        "name": ev_row.get("name", ""),
        "country": ev_row.get("Country", ""),
        "event_date": ev_date,
        "n_pre": int(n_pre),
        "n_post": int(n_post),
        "synch_pre": float(syn_pre),
        "synch_post": float(syn_post),
        "delta_synch": float(syn_post - syn_pre),
        "idio_pre": float(iv_pre),
        "idio_post": float(iv_post),
        "delta_idio": float(iv_post - iv_pre),
        "window_df": sub[["tau", "Synchronicity", "Idio_Vol"]].copy(),
    }
    return out


candidates = []
for _, ev in events.iterrows():
    x = extract_event_window(ev)
    if x is not None:
        candidates.append(x)

if not candidates:
    raise RuntimeError("No valid case-study candidates found.")

cand_df = pd.DataFrame(
    [{k: v for k, v in c.items() if k != "window_df"} for c in candidates]
)

# ADD: prioritize strong increase in synchronicity and decrease in idio volatility
add_cand = cand_df[cand_df["event_type"] == "ADD"].copy()
add_cand["score"] = add_cand["delta_synch"] - 5 * add_cand["delta_idio"]
add_sel_row = add_cand.sort_values("score", ascending=False).iloc[0]

# DELETE: prioritize strong decrease in synchronicity
# (delta_synch negative => larger -delta_synch is better)
del_cand = cand_df[cand_df["event_type"] == "DELETE"].copy()
del_cand["score"] = -del_cand["delta_synch"] - 5 * del_cand["delta_idio"]
del_sel_row = del_cand.sort_values("score", ascending=False).iloc[0]

sel_keys = set(
    [
        (add_sel_row["ticker"], pd.Timestamp(add_sel_row["event_date"])),
        (del_sel_row["ticker"], pd.Timestamp(del_sel_row["event_date"])),
    ]
)

selected = []
for c in candidates:
    key = (c["ticker"], pd.Timestamp(c["event_date"]))
    if key in sel_keys:
        selected.append(c)

selected = sorted(selected, key=lambda x: x["event_type"])

# Export summary table
summary_rows = []
for s in selected:
    summary_rows.append(
        {
            "event_type": s["event_type"],
            "ticker": s["ticker"],
            "name": s["name"],
            "country": s["country"],
            "event_date": s["event_date"].date().isoformat(),
            "n_pre": s["n_pre"],
            "n_post": s["n_post"],
            "synch_pre": round(s["synch_pre"], 4),
            "synch_post": round(s["synch_post"], 4),
            "delta_synch": round(s["delta_synch"], 4),
            "idio_pre": round(s["idio_pre"], 6),
            "idio_post": round(s["idio_post"], 6),
            "delta_idio": round(s["delta_idio"], 6),
        }
    )

summary = pd.DataFrame(summary_rows)
out_csv = root / "data/results/case_study_add_delete.csv"
summary.to_csv(out_csv, index=False)

# Build figure (2x2): each row event type, columns Synchronicity / Idio_Vol
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

for r, s in enumerate(selected):
    dfw = s["window_df"].groupby("tau", as_index=False).mean().sort_values("tau")

    ax1 = axes[r, 0]
    ax1.plot(
        dfw["tau"], dfw["Synchronicity"], marker="o", linewidth=1.8, color="#1f77b4"
    )
    ax1.axvline(-0.5, linestyle="--", color="red", alpha=0.7)
    ax1.axhline(s["synch_pre"], linestyle=":", color="gray", alpha=0.7)
    ax1.set_title(f"{s['event_type']} {s['ticker']} — Synchronicity")
    ax1.set_ylabel("Logit($R^2$)")
    ax1.grid(alpha=0.25)

    ax2 = axes[r, 1]
    ax2.plot(dfw["tau"], dfw["Idio_Vol"], marker="o", linewidth=1.8, color="#2ca02c")
    ax2.axvline(-0.5, linestyle="--", color="red", alpha=0.7)
    ax2.axhline(s["idio_pre"], linestyle=":", color="gray", alpha=0.7)
    ax2.set_title(f"{s['event_type']} {s['ticker']} — Idio_Vol")
    ax2.grid(alpha=0.25)

for ax in axes[-1, :]:
    ax.set_xlabel("Months relative to event ($\\tau$)")

fig.suptitle(
    "Illustrative One-Year Dynamics Around Index Membership Events", fontsize=14
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out_fig = root / "figures/case_study_add_delete_dynamics.png"
fig.savefig(out_fig, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Selected case studies:")
print(summary.to_string(index=False))
print("\nSaved:", out_csv)
print("Saved:", out_fig)
