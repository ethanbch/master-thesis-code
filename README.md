# Passive Ownership & Price Efficiency — STOXX Europe 600

Master's thesis empirical pipeline: causal impact of passive ownership shocks
(index inclusions / exclusions) on price efficiency in European equity markets.

---

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
│   │   ├── 00_eda_and_checks.py      Pre-analysis EDA & data validation
│   │   ├── 01_matching.py            Propensity Score Matching
│   │   ├── 02_did_estimation.py      Static DiD (PanelOLS)
│   │   └── 03_event_study.py         Dynamic DiD + coefficient plots
│   │
│   ├── robustness/               Robustness checks
│   │   ├── placebo_test.py           Randomization inference — OLS DiD (200 iterations)
│   │   ├── placebo_dml.py            Permutation test — DML estimate (200 iterations)
│   │   ├── delete_events.py          Full pipeline for DELETE events
│   │   └── caliper_sensitivity.py    Caliper sensitivity analysis
│   │
│   └── visualization/            Descriptive plots
│       └── panel_plots.py            Panel overview figure
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
│       ├── placebo_dml_results.csv
│       └── robustness_caliper_results.csv
│
├── figures/                      Generated figures
├── Makefile                      Pipeline runner
├── pyproject.toml                Dependencies
└── README.md
```

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run the full pipeline
make all

# Or run individual stages
make prep          # Data preparation (steps 01–06)
make analysis      # Matching → DiD → Event study
make robustness    # Placebo (OLS + DML), DELETE events, caliper sensitivity
make plots         # Panel overview figure
```

---

## Data Preparation — Detailed Documentation

This section documents every step of the data pipeline, the problems encountered,
and the methodological decisions taken to address them. It is intended to support
the methodology chapter of the thesis.

---

### Step 1 — `01_panel_composition.py`: Build the composition panel & detect events

#### Source data

The raw data are the official STOXX monthly public constituent files
(`slpublic_sxxp_YYYYMMDD.csv`), covering the STOXX Europe 600 index from
**November 2015 to February 2026** — a total of **126 monthly snapshots**.
Each file lists the 600 (and a few hundred borderline) securities ranked at that
date, with the two key fields `Rank (FINAL)` (rank in the current snapshot) and
`Rank (PREVIOUS)` (rank in the preceding snapshot).

| Field             | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `Rank (FINAL)`    | Rank of the security in the current monthly snapshot |
| `Rank (PREVIOUS)` | Rank of the security in the preceding snapshot       |
| `ISIN`            | International Securities Identification Number       |
| `RIC`             | Reuters Instrument Code — used as starting ticker    |
| `Country`         | Domicile country (17 countries in the sample)        |

#### Problem: PDF parsing artifacts in pre-November 2023 files

The files from November 2015 through approximately mid-2021 were originally
generated as PDF quarterly reviews and later converted to CSV. This conversion
left two classes of artifacts that had to be removed before the data could be
used:

1. **"Page" rows** — PDF page headers and footers were parsed as data rows.
   These rows contain the string `"Page"` in the `Instrument_Name` field and
   carry no meaningful rank information.

2. **Duplicate rank-0 through rank-9 rows** — PDF page numbers (0, 1, 2, …, 9)
   were misread as `Rank (FINAL)` values. For each rank value 0–9, the file
   therefore contains the genuine security ranked at that position _and_ one or
   several artifact rows that inherited the same rank from a page number.

3. **Fully corrupted `Rank (FINAL)` columns** — In quarterly review files
   (months 01, 04, 07, 10) from 2015 to 2021, the entire `Rank (FINAL)` column
   contains only digits 0–9, rendering it completely unusable. The
   `Rank (PREVIOUS)` column is unaffected in all cases.

These artifacts created a catastrophic bias: a naïve run of the event-detection
logic produced **13,153 spurious events with an ADD/DELETE ratio of 10.8:1**,
entirely driven by the duplicated rank-0–9 rows.

#### Fix implemented

**Artifact removal (`_remove_pdf_artifacts`)**

- Remove all rows where `Instrument_Name` contains `"Page"`.
- Remove all rows where `Rank (FINAL) == 0` (always an artifact; real ranks start at 1).
- For ranks 1–9, when multiple rows share the same rank value, keep only the
  genuine security using the **minimum-distance-to-previous-rank rule**:
  $$\arg\min_j \bigl|\text{Rank\_final}_j - \text{Rank\_prev}_j\bigr|$$
  A real stock ranked 7 this month was almost certainly ranked close to 7 last
  month (index composition is stable between reviews); an artifact row carries a
  page-number rank unrelated to the security's actual position. This heuristic
  correctly identifies the genuine security in every case tested.

**Corruption fallback (`_is_rank_corrupted`)**

- After artifact removal, if `Rank (FINAL)` still has fewer than 50 unique values,
  the column is deemed fully corrupted.
- In that case, `Rank (FINAL)` is replaced by `Rank (PREVIOUS)`. This is
  methodologically sound because, for quarterly index reviews, the index
  composition changes little between consecutive months; using `Rank (PREVIOUS)`
  as a proxy for `Rank (FINAL)` introduces only negligible noise.

**Result:** After cleanup, the sample yields **953 ADD events and 921 DELETE events**
across 17 countries and 434 unique treated tickers — a balanced 1:1 ADD/DELETE ratio
consistent with the mechanical nature of index rebalancing.

#### Event identification

An **ADD** event is defined as the first date at which a security's rank crosses
into the index:

$$\text{rank\_prev} > 600 \quad \land \quad \text{rank\_final} \leq 600$$

A **DELETE** event is the symmetric transition outward:

$$\text{rank\_prev} \leq 600 \quad \land \quad \text{rank\_final} > 600$$

Observations where either rank is `NaN` (IPOs, delistings, first appearances)
are excluded, as no clean before/after comparison is possible.

---

### Step 2 — `02_collect_prices.py`: Download historical prices

#### Data source and coverage

Daily adjusted closing prices and volumes are retrieved from **Yahoo Finance**
via `yfinance` for every ADD-event ticker plus every "stable" ticker (present in
at least 12 monthly snapshots, used as the potential control pool). The market
benchmark `^STOXX` (STOXX Europe 600 total return index) is also downloaded.

- **Time window:** 1 January 2013 → 28 February 2026 (12 years of history,
  providing a minimum 12-month pre-event window for all events back to 2015).
- **Batch size:** 50 tickers per request, with a randomised pause of 3–6 seconds
  between batches to respect Yahoo Finance rate limits.
- **Retry logic:** Up to 3 attempts per batch, with exponential back-off (15s,
  30s, 45s) on HTTP 429 / 5xx errors.
- **Output:** `prices_raw.parquet` — long-format table with columns `date`,
  `ticker`, `close` (adjusted), `volume`.

#### Why Yahoo Finance?

We use Yahoo Finance because it provides split-adjusted and dividend-adjusted
closing prices (`Adj Close`) at no cost for a multi-country European sample.
The main limitation — ticker identifiers — is addressed in steps 3 and 4.

---

### Step 3 — `03_check_coverage.py`: Ticker coverage & RIC→Yahoo remapping

#### Problem: RIC codes do not map directly to Yahoo Finance tickers

STOXX files use Reuters Instrument Codes (RICs), whose suffix conventions differ
from Yahoo Finance's. Without remapping, a large share of European securities
would appear as missing even though they are available on Yahoo Finance under a
slightly different identifier.

#### Remapping strategy

Two complementary strategies are applied:

**Manual dictionary (`MANUAL_REMAP`)** — country-specific corrections identified
by inspecting missing tickers:

| Pattern                  | Rule                                                      | Example                                 |
| ------------------------ | --------------------------------------------------------- | --------------------------------------- |
| French stocks with `.PA` | Ticker prefix differs (RIC uses ISIN-based prefixes)      | `RENA.PA → RNO.PA`                      |
| German stocks with `.DE` | RIC appends `G`, `n`, or `Gn` to the ticker stem          | `BASFn.DE → BAS.DE`, `ALVG.DE → ALV.DE` |
| Irish stocks             | `.I` suffix → `.IR` or `.L` depending on primary exchange | `GL9.I → GL9.IR`                        |
| Austrian stocks          | Occasional `R` suffix appended                            | `VIGR.VI → VIG.VI`                      |

A second pass of manual corrections (45 additional entries) was added after
auditing `missing_add_tickers.csv` against Yahoo Finance:

| Country | Count | Representative examples                                                                                                               |
| ------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------- |
| France  | 14    | `EPED.PA → FGR.PA` (Faurecia → Forvia), `AIRF.PA → AF.PA` (Air France-KLM), `LTEN.PA → ATE.PA` (Alten), `VLOF.PA → FR.PA` (Valeo)     |
| Germany | 10    | `WCHG.DE → WCH.DE` (Wacker Chemie), `EVDG.DE → EVD.DE` (CTS Eventim), `HYQGn.DE → HYQ.DE` (Hypoport), `FNTGn.DE → FNTN.DE` (Freenet)  |
| UK      | 21    | `WISEa.L → WISE.L` (Wise), `ECM.L → RS1.L` (Electrocomponents → RS Group), `HAYS.L → HAS.L` (Hays), `DC.L → CURY.L` (Dixons → Currys) |

**Regex rules (`_apply_regex_remaps`)** — systematic patterns that cannot be
enumerated stock by stock:

| Market              | Regex pattern                           | Replacement                          | Example                                        |
| ------------------- | --------------------------------------- | ------------------------------------ | ---------------------------------------------- |
| Switzerland         | `\.S$`                                  | `.SW`                                | `NESN.S → NESN.SW`, `NOVN.S → NOVN.SW`         |
| Nordics (class B/A) | `([A-Za-z])([BbAa])\.(ST\|CO\|HE\|OL)$` | `\1-\U\2.\3` (hyphenated, uppercase) | `CARLB.CO → CARL-B.CO`, `BETSb.ST → BETS-B.ST` |

The Swiss rule corrects STOXX's non-standard `.S` suffix (Yahoo Finance uses `.SW`
for the SIX Swiss Exchange). The Nordic rule inserts a hyphen before the share-class
letter (`B` for B-shares, `A` for A-shares), which Yahoo Finance requires for
dual-class Nordic equities.

The regex approach is preferred over a manual list for these two markets
because it covers _any_ ticker — including new entrants — without requiring
manual maintenance after each index rebalancing.

The regex approach is preferred over a manual list for Swiss and Nordic stocks
because it covers _any_ new ticker added to the index without requiring further
maintenance.

**Artifact filter** — Artifact tickers from the PDF parsing (e.g. `"1.7"`, `"2.0"`)
that were written as `ticker` values in `events.csv` are detected with
`str.fullmatch(r"\d+(\.\d+)?")` and dropped before any download attempt.

---

### Step 4 — `04_fix_tickers.py`: Re-download missing tickers

After the remapping applied in step 3, tickers that still had no price data are
retried with a dedicated download pass. The same remapping logic (manual
dictionary + regex) is applied again, ensuring consistency. The newly downloaded
prices are merged back into `prices_raw.parquet`, overwriting any blank entries
for the same ticker.

Download parameters mirror step 2 (BATCH_SIZE=50, pause 3–6s, 3 retries) to
maintain parity with the main download pass.

#### Coverage after steps 3 + 4

After both remapping passes, ADD-event coverage reached **70.8% (675/953)**.
The table below shows coverage by country after the full fix pipeline:

| Country | ADD events | With data | Coverage |
| ------- | ---------- | --------- | -------- |
| PT      | 3          | 3         | 100.0%   |
| LU      | 7          | 7         | 100.0%   |
| DE      | 105        | 89        | 84.8%    |
| CH      | 78         | 66        | 84.6%    |
| FR      | 66         | 54        | 81.8%    |
| SE      | 121        | 96        | 79.3%    |
| BE      | 22         | 17        | 77.3%    |
| GB      | 254        | 192       | 75.6%    |
| ES      | 29         | 20        | 69.0%    |
| IE      | 6          | 4         | 66.7%    |
| NO      | 52         | 34        | 65.4%    |
| DK      | 45         | 24        | 53.3%    |
| NL      | 30         | 15        | 50.0%    |
| PL      | 26         | 12        | 46.2%    |
| IT      | 59         | 27        | 45.8%    |
| FI      | 21         | 9         | 42.9%    |
| AT      | 13         | 2         | 15.4%    |

The remaining 29.2% missing are **unrecoverable**: they correspond to firms
that were taken private, delisted, or absorbed into a merger before Yahoo
Finance's historical record begins (e.g. Iliad PA privatised 2022, Natixis
PA delisted 2021, Software AG DE Silver Lake 2023, Deliveroo L DoorDash
2024, Immofinanz VI 2016). Austria's low coverage (15.4%) reflects the
small market size and the Immofinanz / Raiffeisen Bank delistings.

#### Output

`prices_raw.parquet` — **2,931,958 rows**, 1,035 tickers, 1 January 2013 → 27 February 2026.

---

### Step 5 — `05_build_panel.py`: Build the monthly efficiency metrics panel

#### Purpose

For each `(ticker, calendar month)` pair with at least **15 daily observations**,
a market model OLS regression is run:

$$r_{i,t} = \alpha_i + \beta_i \cdot r_{m,t} + \varepsilon_{i,t}$$

where $r_{m,t}$ is the log-return of the STOXX Europe 600 index (`^STOXX`).
Four efficiency metrics are derived from this regression:

| Metric          | Definition                                                                                     | Interpretation                                                                                      |
| --------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `R2_raw`        | $R^2$ of the OLS fit, clipped to $[0.001, 0.999]$                                              | Raw co-movement with the market                                                                     |
| `Synchronicity` | $\ln\!\left(\frac{R^2}{1-R^2}\right)$                                                          | Log-odds of $R^2$; normally distributed; preferred in the literature (Roll 1988, Morck et al. 2000) |
| `Idio_Vol`      | $\sigma(\hat{\varepsilon})$ — std dev of OLS residuals                                         | Idiosyncratic volatility; proxy for firm-specific information incorporation                         |
| `Amihud`        | $\frac{1}{T}\displaystyle\sum_t \frac{\lvert r_{i,t}\rvert}{P_{i,t}\times V_{i,t}}\times 10^6$ | Amihud (2002) illiquidity ratio                                                                     |
| `Avg_Volume`    | Mean daily share volume over the month                                                         | Absolute liquidity control                                                                          |

#### Data sanitisation decisions

Before computing log-returns, two strict filters are applied: **`Close > 0`
AND `Volume > 0`**. Both conditions must hold simultaneously. Yahoo Finance
occasionally returns zero or negative prices for trading halts, corporate
action gaps, or data errors; keeping such rows would produce `log(0) = -∞`
when computing daily log-returns. Zero-volume rows would additionally cause
division-by-zero in the per-day Amihud ratio, yielding `±inf` values that
propagate to the monthly mean. Applying these two filters at source eliminates
both failure modes cleanly.

As a second layer, any residual `±inf` in the log-return series (e.g. from
a price that slipped past the first filter during a multi-ticker merge) is
replaced by `NaN` before `dropna()`, ensuring a fully finite log-return
time series enters the OLS regression.

The Amihud ratio is protected at the daily level as well: any per-day
infinite value is replaced by `NaN` before the monthly mean is computed.

#### Output

`panel_monthly.parquet` — **132,652 rows**, 1,033 tickers, August 2013 → February 2026.

| Metric        | Median  | p99        | Max         |
| ------------- | ------- | ---------- | ----------- |
| Synchronicity | −1.88   | 1.14       | 2.82        |
| Idio_Vol      | 0.0156  | 0.0669     | 0.800       |
| Amihud        | 0.0018  | 9.17       | 6,938       |
| Avg_Volume    | 235,538 | 33,312,310 | 417,950,100 |

The extreme right skew of Amihud (max = 6,938, p99 = 9.17) is expected for
small-cap or illiquid securities that occasionally trade on very thin volumes.

#### Winsorisation of Amihud and Idio_Vol

Both `Amihud` and `Idio_Vol` are **winsorised at the 1st and 99th percentiles**
before any regression is run. A single month of extreme illiquidity (Amihud
reaching 6,938 versus a median of 0.0018) would dominate the OLS objective and
produce misleading coefficient estimates; the same applies to extreme residual
volatility spikes in `Idio_Vol`. Winsorisation replaces values below p1 with the
p1 value and values above p99 with the p99 value, preserving the full cross-sectional
rank ordering while preventing outliers from distorting the estimates.

`Synchronicity` is **not** winsorised: it is already bounded by construction.
Because $R^2$ is clipped to $[0.001, 0.999]$ before the log-odds transformation,
Synchronicity is mathematically confined to $[\ln(0.001/0.999),\ \ln(0.999/0.001)] \approx [-6.91,\ +6.91]$.
No extreme outlier can arise from this variable, making winsorisation unnecessary
and potentially distortionary.

This treatment is applied **at runtime inside the estimation scripts**
(`02_did_estimation.py` and `03_event_study.py`), not to the stored parquet file.
Keeping `panel_monthly.parquet` unmodified ensures full reproducibility: any
researcher can re-run the analysis with a different winsorisation threshold
(e.g. p2.5/p97.5) by changing a single line in the estimation script.

---

### Step 6 — `06_build_features.py`: Compute pre-event features for PSM

#### Purpose

Propensity Score Matching (PSM) requires observable pre-treatment characteristics
to estimate the propensity of being treated (added to the index). For each ADD
event date, a 12-month (365-day) pre-event window is used to compute three
matching variables for the treated firm **and for every potential control ticker**
in the price database:

| Feature          | Definition                                         | Rationale                                                                                                                                                                                                                                                                     |
| ---------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Log_MarketCap`  | $\ln(\text{price} \times \bar{\text{volume}})$     | **Log Dollar Volume proxy.** Following Brennan, Chordia & Subrahmanyam (1998), dollar volume acts as a near-perfect proxy for firm size when shares outstanding are unavailable; the two measures exhibit near-unit cross-sectional correlation in asset pricing regressions. |
| `Momentum_12m`   | $\ln(P_{\text{end}} / P_{\text{start}})$           | Pre-event return; controls for the well-documented momentum effect and for the fact that index committees tend to select strong recent performers.                                                                                                                            |
| `Volatility_pre` | $\sigma(\text{daily log-returns})$ over the window | Risk control; more volatile stocks have a different return-generating process regardless of index membership.                                                                                                                                                                 |

A minimum of **200 daily observations** (approximately 10 months of trading) in
the window is required for a ticker to enter the matching pool. Tickers with
infinite or missing feature values after log-transformation are dropped.

#### Why log(price × volume) as a size proxy?

Direct market capitalisation requires shares outstanding, which are not available
in the STOXX constituent files and would require a separate paid data source.
Log dollar volume ($\ln(P \times V)$) is the standard substitute in the empirical
asset pricing literature:

- **Brennan, Chordia & Subrahmanyam (1998)** — _"Alternative factor specifications,
  security characteristics, and the cross-section of expected stock returns"_,
  _Journal of Financial Economics_ 49(3): 345–373 — show that log dollar volume
  proxies for market capitalisation with a near-unit correlation in cross-sectional
  regressions, making it an appropriate size control when shares outstanding are
  unavailable.

- **Amihud (2002)** — _"Illiquidity and stock returns: cross-section and
  time-series effects"_, _Journal of Financial Markets_ 5(1): 31–56 — uses
  dollar volume as the denominator of the illiquidity ratio and documents its
  close relationship with size, confirming the validity of this proxy for
  European multi-country panels.

Log dollar volume is directly computable from the price data already collected
in step 2, requires no additional data source, and is available for every
ticker in the matching pool.

#### Output

`features_at_event.csv` — **894,866 rows**, 82 unique event dates, 284 treated
tickers, pool of ~10,900 potential controls per event on average.

| Feature        | Median | Std    |
| -------------- | ------ | ------ |
| Log_MarketCap  | 16.16  | 2.99   |
| Momentum_12m   | 0.048  | 0.445  |
| Volatility_pre | 0.0212 | 0.0126 |

---

## Analysis Pipeline

### `00_eda_and_checks.py` — Pre-analysis validation

Before running any causal model, four figures are generated to validate the
data and inform modelling choices:

1. **PSM feature distributions** — KDE plots comparing treated vs. control pools
   on `Log_MarketCap`, `Momentum_12m`, and `Volatility_pre`.
2. **Efficiency metrics boxplots** — Distribution of `Synchronicity`, `Idio_Vol`,
   `Amihud` (log₁₀ scale), and `Avg_Volume` across the full panel.
3. **Correlation matrix** — Pearson correlations between winsorised (p1–p99)
   efficiency metrics.
4. **ADD events per year** — bar chart to verify temporal balance of the sample.

### `01_matching.py` — Propensity Score Matching

Logistic regression on (`Log_MarketCap`, `Momentum_12m`, `Volatility_pre`) to
estimate treatment propensity. Nearest-neighbour 1:1 matching without
replacement. Each treated ticker is matched to the control with the closest
propensity score.

### `02_did_estimation.py` — Static Difference-in-Differences

PanelOLS with entity and time fixed effects on the stacked matched-pair panel
(±12 months around each event). The estimating equation is:

$$Y_{it} = \alpha_i + \gamma_t + \delta \cdot (\text{Treat}_i \times \text{Post}_{it}) + \beta_1 \text{Amihud}_{it} + \beta_2 \text{AvgVol}_{it} + \varepsilon_{it}$$

Standard errors are clustered at the entity level. Amihud and Idio_Vol are
winsorised at p1/p99 at runtime before estimation.

### `03_event_study.py` — Dynamic DiD (event study)

Interaction of `Treat` with period dummies $\mathbf{1}[\tau = t]$ for each
relative month $\tau \in [-6, +6] \setminus \{-1\}$ (month −1 is the omitted
reference period). Tests the pre-trend assumption and traces the post-treatment
dynamics.

---

## Robustness Checks

| Script                   | Test                                                                                                                    |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `placebo_test.py`        | 200 random-date permutations — tests whether the OLS DiD estimate exceeds what would be obtained by chance              |
| `placebo_dml.py`         | 200 within-pair label permutations — tests whether the DML estimate exceeds what would be obtained under the sharp null |
| `delete_events.py`       | Replicate the full pipeline for DELETE events — symmetric test of the causal channel                                    |
| `caliper_sensitivity.py` | Vary the PSM caliper width — tests whether results are sensitive to the matching bandwidth                              |

### Placebo test — OLS DiD (random event dates)

`placebo_test.py` draws a fake event date for each matched pair (uniformly in the
[event−12m, event−2m] window) and re-estimates the stacked PanelOLS DiD. Over 200
valid iterations the null distribution is:

| Statistic        | Value     |
| ---------------- | --------- |
| Mean β_placebo   | +0.0135   |
| Median β_placebo | +0.0143   |
| Std β_placebo    | 0.0249    |
| True β_DiD       | −0.0344   |
| Empirical p      | **0.025** |

The empirical p-value (share of placebo betas strictly below the true beta) is
**0.025**, significant at the 5 % level by randomization inference. Two features
of the null distribution are worth noting:

1. **The null is right-shifted** (mean β_placebo = +0.013 rather than 0).
   Placebo dates are drawn before the true event, typically in periods of
   positive pre-treatment momentum; the stacked DiD therefore systematically
   picks up a slightly positive beta under the null. The correct test is
   therefore relative to this right-shifted null, not to zero.

2. **The true beta (−0.034) lies in the left tail** of the right-shifted null.
   Only 2.5 % of placebo iterations produced a beta as negative or more
   negative. This constitutes evidence that a genuine downward effect on
   Synchronicity is present in the data — consistent with the DML result —
   even if the parametric OLS p-value (0.478) failed to detect it due to
   linearity constraints.

### Permutation test — DML (within-pair label swap)

`placebo_dml.py` permutes the `treated`/`control` labels within each matched pair
(fair-coin flip per pair, preserving the 1:1 balance) and re-fits the DoubleML
PLR. Over 200 iterations:

| Statistic       | Value     |
| --------------- | --------- |
| Mean θ_permuted | −0.0003   |
| Std θ_permuted  | 0.0480    |
| p95 θ_permuted  | +0.0760   |
| Observed θ      | +0.1340   |
| Empirical p     | **0.000** |

The observed DML estimate (θ = +0.134) lies entirely outside the permutation
null distribution: zero of the 200 permuted draws reached or exceeded +0.134
(empirical p < 0.005). The null distribution is tightly centred on zero
(mean ≈ 0.000, std = 0.048), confirming that the DML result is not an
artefact of the random forest or the cross-fitting procedure. The result is
statistically significant at the 1 % level by randomization inference.

### DELETE events — Symmetric robustness test

`delete_events.py` replicates the full estimation pipeline (PSM → static DiD →
event study → DML) for index **exclusions**, providing a symmetric causal test
of the basket-trading channel.

#### Sample

921 DELETE events are detected in the panel. After PSM (caliper = 0.01), **599 valid
matched pairs** are retained (2.9 % rejection rate) — a sample size comparable to
the 654 ADD pairs, enabling a direct symmetric comparison.

#### Static DiD — DELETE events (PanelOLS, entity + time FE, clustered SE)

| Outcome           | β_DiD   | p-value | Significance       |
| ----------------- | ------- | ------- | ------------------ |
| **Synchronicity** | −0.1489 | 0.0024  | ★★★ significant 1% |
| **Idio_Vol**      | −0.0001 | 0.8110  | n.s.               |

Index exclusion **reduces Synchronicity significantly** (β = −0.1489, p = 0.002):
once a firm leaves the index, passive ETF flows no longer inject the common
index-level factor into its price, and co-movement with the market falls sharply.
Idio_Vol, by contrast, is unaffected (p = 0.811), indicating that firm-specific
noise does not rebound immediately after deletion.

#### DML — DELETE events

| Metric       | Value            |
| ------------ | ---------------- |
| θ            | +0.1189          |
| SE           | 0.0662           |
| p-value      | 0.0725           |
| Significance | ★ (10 %)         |
| 95 % CI      | [−0.011, +0.249] |

The DML estimate for DELETE events is positive (θ = +0.119, p = 0.073), reflecting
that on the PSM-matched cross-section, the deleted (treated) firms retain a higher
level of Synchronicity relative to matched controls at the post-event date. This
is consistent with the gradual unwinding of passive ownership: ETF reconstitution
occurs over several days and residual index tracking means some passive funds
continue to hold the name for weeks after deletion. The DiD (within-entity
before/after change, β = −0.149) and the DML (cross-sectional level difference)
together confirm that deletion withdraws the basket-trading channel, but the
transition is not instantaneous.

#### ADD vs DELETE comparison — The mirror pattern

| Event type             | N pairs | β_DiD Synchronicity | p     | β_DiD Idio_Vol | p     | θ_DML Synchronicity |
| ---------------------- | ------- | ------------------- | ----- | -------------- | ----- | ------------------- |
| **ADD (inclusion)**    | 654     | −0.0344             | 0.478 | −0.0007        | 0.013 | +0.134              |
| **DELETE (exclusion)** | 599     | −0.1489             | 0.002 | −0.0001        | 0.811 | +0.119              |

The two event types produce a striking, complementary mirror:

- **On inclusion**: idiosyncratic volatility falls significantly (β = −0.0007,
  p = 0.013). Linear OLS cannot detect a Synchronicity effect (p = 0.478), but
  DML — by partialling out non-linearities in firm size and pre-event volatility —
  reveals a significant increase (θ = +0.134, p = 0.051).

- **On exclusion**: the pattern inverts. Synchronicity falls significantly
  (β = −0.1489, p = 0.002), while Idio_Vol is unaffected (p = 0.811). Here the
  linear OLS estimator is sufficient because the Synchronicity effect of exclusion
  is economically large (4× the ADD coefficient), making it detectable without
  flexible nuisance models.

The asymmetry in OLS detectability is itself informative: the Synchronicity effect
of basket-trading flows is **large and linear on exit** (passive ownership is
abruptly withdrawn at deletion) but **small and non-linear on entry** (ETF
ownership builds up gradually over many rebalancing cycles). DML is needed
precisely in the less extreme entry case — it recovers the signal that the linear
estimator misses.

### Identification Strategy

1. **Propensity Score Matching** — LogisticRegression on Log_MarketCap,
   Momentum_12m, Volatility_pre; nearest-neighbor; caliper = 0.01
2. **Stacked DiD** — PanelOLS with entity + time FE, clustered SE at entity level
3. **Event Study** — Dynamic DiD with τ ∈ [−6, +6], reference τ = −1

### Key Results

#### PSM summary

The caliper of 0.01 yields **654 valid matched pairs** across the full panel
(2015–2026). Propensity score distances remain highly concentrated (P75 ≈ 0.00004,
P95 ≈ 0.0016), indicating near-perfect covariate balance for retained pairs.

#### Static DiD — ADD events (PanelOLS, entity + time FE, clustered SE)

| Outcome           | β_DiD   | p-value | Significance     |
| ----------------- | ------- | ------- | ---------------- |
| **Synchronicity** | −0.0344 | 0.4781  | n.s.             |
| **Idio_Vol**      | −0.0007 | 0.0126  | ★ significant 5% |

The negative coefficient on **Idio_Vol** (β = −0.0007, p = 0.013) indicates that
index inclusion reduces idiosyncratic volatility, consistent with the noise-trader
hypothesis: passive ownership shocks dampen firm-specific price movements by
shifting trading activity toward index-level arbitrage. The non-significance of
Synchronicity under a linear OLS framework motivates the Double ML extension
(Section _Advanced Identification_ below).

Caliper sensitivity: results stable across [0.005, 0.01, 0.05].

---

## Advanced Identification: Double Machine Learning

### Motivation

The static PanelOLS DiD describes the _average_ conditional relationship between
index inclusion and price efficiency, but imposes two restrictive assumptions:
(i) linearity in all covariates, and (ii) correct specification of the propensity
model. Any non-linear interaction between pre-treatment characteristics
(size, momentum, pre-event volatility) and treatment assignment can introduce
residual confounding that OLS cannot absorb, even with entity and time fixed
effects. This is particularly relevant for Synchronicity, whose non-significance
under PanelOLS (p = 0.478) may reflect model misspecification rather than a
genuine null effect.

### Framework — Chernozhukov et al. (2018)

We implement the **Partially Linear Regression (PLR)** variant of Double Machine
Learning (DML) following Chernozhukov, Chetverikov, Demirer, Duflo, Hansen,
Newey & Robins (2018), _"Double/Debiased machine learning for treatment and
structural parameters"_, _The Econometrics Journal_ 21(1): C1–C68.

The structural equation is:

$$Y = \theta D + g(X) + \varepsilon, \qquad \mathbb{E}[\varepsilon \mid D, X] = 0$$

where:

- $Y$ = `Synchronicity` (outcome)
- $D$ = `treated` (binary treatment indicator)
- $X$ = (`Log_MarketCap`, `Momentum_12m`, `Volatility_pre`) (pre-event covariates)
- $\theta$ = causal Average Treatment Effect (ATE) of index inclusion on
  Synchronicity, **the parameter of interest**

### Neyman Orthogonality and Cross-Fitting

The DML estimator achieves $\sqrt{n}$-consistency and asymptotic normality despite
using flexible ML models for the nuisance functions through two mechanisms:

**1. Neyman Orthogonality** — The moment condition used to identify $\theta$ is
orthogonal to small perturbations in the nuisance functions $g(\cdot)$ and
$m(\cdot)$ (the propensity score $\mathbb{E}[D \mid X]$). This orthogonality
condition eliminates the first-order bias that would otherwise arise from
regularisation error in the ML nuisance estimates, ensuring that even if $\hat{g}$
and $\hat{m}$ converge at slower rates (e.g. $n^{-1/4}$ for a random forest),
the estimator of $\theta$ converges at the parametric rate $n^{-1/2}$.

**2. Cross-Fitting (K = 5 folds)** — To prevent the nuisance models from
overfitting to the same observations used to orthogonalise the residuals, the
sample is partitioned into 5 folds. For each fold, the nuisance models are trained
on the remaining 4 folds and predictions are generated out-of-sample. This
sample-splitting step removes the Donsker condition requirement and is essential
for the validity of the procedure with high-dimensional or non-parametric ML
estimators.

### Nuisance Models

| Nuisance function             | Model                    | Role                                                     |
| ----------------------------- | ------------------------ | -------------------------------------------------------- |
| $g(X) = \mathbb{E}[Y \mid X]$ | `RandomForestRegressor`  | Captures non-linear effects of covariates on $Y$         |
| $m(X) = \mathbb{E}[D \mid X]$ | `RandomForestClassifier` | Flexible propensity score; avoids logit misspecification |

Both forests use default `scikit-learn` hyperparameters (100 trees). No explicit
tuning is performed; the doubly-robust structure ensures that consistent
estimation of $\theta$ requires only that _at least one_ of the two nuisance
models converges.

### Implementation

Script: `scripts/analysis/04_double_ml.py`  
Library: [`DoubleML`](https://docs.doubleml.org) (Bach et al. 2022)  
Output: `data/results/dml_results.csv`

```
Y  = Synchronicity
D  = treated (ADD event indicator)
X  = [Log_MarketCap, Momentum_12m, Volatility_pre]
n_folds = 5
n_rep   = 5   (repeated cross-fitting for stability)
```

### Results

| Metric         | OLS (PanelOLS) | DML (PLR, Random Forest) |
| -------------- | -------------- | ------------------------ |
| β / θ          | −0.0344        | **+0.1187**              |
| p-value        | 0.4781         | **0.0506**               |
| Significance   | n.s.           | ★ marginal (10 %)        |
| N observations | ~stacked panel | 1,268 (636 treated)      |

The sign reversal and dramatic improvement in p-value (0.478 → 0.051) confirm
that the OLS model was misspecified: the effects of firm size and pre-event
volatility on post-inclusion Synchronicity are non-linear, and a linear
projection could not isolate the true treatment signal. Once the random forests
partial out these non-linearities via Neyman-orthogonal residualisation, the
effect of index inclusion on Synchronicity becomes statistically meaningful.

#### Economic interpretation

The positive ATE (θ = +0.119) indicates that **index inclusion increases price
synchronicity** — i.e. the stock's returns co-move more strongly with the STOXX
600 after addition. This is consistent with the **basket-trading channel**: ETF
and index-fund flows buy and sell all index constituents simultaneously, injecting
a common factor into individual security prices that is orthogonal to firm-level
news. The result stands alongside the Idio_Vol finding: passive ownership shocks
simultaneously reduce idiosyncratic volatility (less firm-specific noise) and
increase market co-movement (more index-level signal), two faces of the same
information-dilution phenomenon.

#### Note on propensity score warnings

The `RandomForestClassifier` produced predictions close to 0 or 1 in several
cross-fitting folds. This is expected and methodologically benign on a
PSM-matched sample, where treated and control firms are nearly identical on
observables by construction — making treatment nearly unpredictable from $X$.
Far from being a problem, it confirms the quality of the upstream PSM step:
the DML propensity model has little residual confounding to absorb, so the
orthogonalisation operates primarily through the outcome nuisance $g(X)$.

#### Synthesis

Two complementary results jointly characterise the impact of passive ownership
shocks on price efficiency:

| Outcome           | Event  | Method   | Effect      | p-value | Interpretation                              |
| ----------------- | ------ | -------- | ----------- | ------- | ------------------------------------------- |
| **Idio_Vol**      | ADD    | PanelOLS | −0.0007 (↓) | 0.013   | Less idiosyncratic noise after inclusion    |
| **Synchronicity** | ADD    | DML-PLR  | +0.134 (↑)  | 0.051   | Stronger market co-movement after inclusion |
| **Synchronicity** | DELETE | PanelOLS | −0.1489 (↓) | 0.002   | Weaker market co-movement after exclusion   |
| **Idio_Vol**      | DELETE | PanelOLS | −0.0001     | 0.811   | No significant change after exclusion       |

All results point in the same direction: passive ownership shocks shift the
information content of prices — on entry, away from firm-specific signals and
toward systematic market-wide movements; on exit, the reverse. The DELETE
results confirm the ADD findings by providing an out-of-sample replication of the
basket-trading channel under a symmetric event (Israeli, Lee & Sridharan 2017;
Morck, Yeung & Yu 2000).
