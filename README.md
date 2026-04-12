ATTENTION PROBLEMES A REGLER : PROBLEME DE PARSING POUR TOUS LES TRUCS AVANT NOVEMBRE 2023, CA A LAIR DE PRENDRE POUR LE RANK FINAL LE CHIFFRE APRES LA VIRGULE DE LA COLONNE FF MCAP DONC IL FAUT CORRIGER CELA
LES RESULTATS QUON A LA CE SONT LES RESULTATS POUR LE TRUC QUI A TOURNE POUR LES DATES A PARTIR DE JANVIER 2024. IL FAUT REFAIRE LE SCRIPT TEST.PY POUR QUIL PARSE PARFAITEMENT ET ENSUITE SEULEMENT ON POURRA RELANCER MAKE ALL ET ON SERA BIEN

JE PENSE QUE POUR LA SUITE IL FAUDRAIT VOIR LEVOLUTION DU BETA ENTRE STOXX600 ET STOXX50 ET REFLECHIR A SI ON POURRAIT INCLURE DU ML OU MEME DU DL

# Passive Ownership & Price Efficiency — STOXX Europe 600

Master's thesis empirical pipeline: causal impact of passive ownership shocks
(index inclusions/exclusions) on price efficiency in European equity markets.

## Project Structure

```
├── scripts/
│   ├── preparation/              Data pipeline (run in order)
│   │   ├── 01_panel_composition.py   Build composition panel & detect events
│   │   ├── 02_collect_prices.py      Download prices via Yahoo Finance
│   │   ├── 03_check_coverage.py      Check ticker coverage & RIC remapping
│   │   ├── 04_fix_tickers.py         Re-download missing tickers
│   │   ├── 05_build_panel.py         Build monthly efficiency metrics panel
│   │   └── 06_build_features.py      Compute pre-event PSM features
│   │
│   ├── analysis/                 Main analysis (run in order)
│   │   ├── 01_matching.py            Propensity Score Matching
│   │   ├── 02_did_estimation.py      Static DiD (PanelOLS)
│   │   └── 03_event_study.py         Dynamic DiD + coefficient plots
│   │
│   ├── robustness/               Robustness checks
│   │   ├── placebo_test.py           Randomization inference (500 iterations)
│   │   ├── delete_events.py          Full pipeline for DELETE events
│   │   └── caliper_sensitivity.py    Caliper sensitivity analysis
│   │
│   └── visualization/            Descriptive plots
│       └── panel_plots.py            Panel overview (3×2 figure)
│
├── notebooks/
│   └── exploration.ipynb         Exploratory analysis
│
├── data/
│   ├── raw/                      Input data (read-only)
│   │   └── inclusions/               STOXX 600 monthly composition CSVs
│   ├── intermediate/             Pipeline intermediates
│   │   ├── panel_composition.parquet
│   │   ├── events.csv
│   │   ├── prices_raw.parquet
│   │   ├── panel_monthly.parquet
│   │   └── features_at_event.csv
│   └── results/                  Final outputs
│       ├── matched_pairs.csv
│       ├── did_results_*.csv
│       ├── placebo_results.csv
│       └── robustness_caliper_results.csv
│
├── figures/                      Generated figures
├── Makefile                      Pipeline runner
├── pyproject.toml                Dependencies
└── README.md
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run the full pipeline
make all

# Or run individual stages
make prep          # Data preparation (steps 01–06)
make analysis      # Matching → DiD → Event study
make robustness    # Placebo, DELETE events, caliper sensitivity
make plots         # Panel overview figure
```

## Methodology

### Data Source

Index composition data from STOXX monthly public constituent files
(`slpublic_sxxp_*.csv`), covering the STOXX Europe 600 index.
**26 monthly snapshots** (2024–2026), ~48,000 firm-date observations.

| Field             | Description                                            |
| ----------------- | ------------------------------------------------------ |
| `Rank (FINAL)`    | Rank of the security in the current snapshot           |
| `Rank (PREVIOUS)` | Rank of the security in the preceding snapshot         |
| `ISIN`            | International Securities Identification Number         |
| `RIC`             | Reuters Instrument Code (used as Yahoo Finance ticker) |

### Event Identification

- **ADD:** rank transitions from above 600 to 600 or below
  ($\text{rank\_prev} > 600 \land \text{rank\_final} \leq 600$)
- **DELETE:** rank transitions from 600 or below to above 600
  ($\text{rank\_prev} \leq 600 \land \text{rank\_final} > 600$)

Observations with `NaN` in either rank are excluded (IPOs, delistings).
Final sample: **302 ADDs, 295 DELETEs**.

### Identification Strategy

1. **Propensity Score Matching** — LogisticRegression on Log_MarketCap,
   Momentum_12m, Volatility_pre; nearest-neighbor; caliper = 0.01
2. **Stacked DiD** — PanelOLS with entity + time FE, clustered SE at entity level
3. **Event Study** — Dynamic DiD with τ ∈ [−6, +6], reference τ = −1

### Key Results

| Specification  | β_DiD (Synchronicity) | p-value |
| -------------- | --------------------- | ------- |
| ADD (baseline) | −0.115                | 0.174   |
| DELETE         | −0.118                | 0.196   |
| Placebo (emp.) | —                     | 0.196   |

Caliper sensitivity: results stable across [0.005, 0.01, 0.05].

The caliper of 0.01 yields 196 valid matched pairs (64.9% of ADD events).
Propensity score distances are highly concentrated (P75 = 0.00004, P95 = 0.0016),
indicating near-perfect covariate balance for retained pairs.
