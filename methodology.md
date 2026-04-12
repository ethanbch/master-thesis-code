# Methodology

## 1. Data Sources and Sample

The empirical analysis draws on two primary data sources. Daily and monthly stock prices—including dividends-adjusted closing prices, trading volumes, and shares outstanding—are retrieved from Yahoo Finance via the `yfinance` API. The composition data for the EURO STOXX 600 (SXXP) universe is sourced from monthly index membership snapshots, which record the full ranked list of index constituents and their weight-based ranks at each rebalancing date. Twenty-six such snapshots are available, spanning January 2, 2024 through February 2, 2026, reflecting the index provider's periodic reconstitutions.

The benchmark return series used in the market model is approximated by the STOXX Europe 600 index (Yahoo Finance ticker: `^STOXX`). For the STOXX 50 sub-sample, the EURO STOXX 50 index (`^STOXX50E`) is used as the market benchmark.

---

## 2. Panel Construction

### 2.1 Event Identification

Index inclusion and deletion events are identified by comparing consecutive monthly membership snapshots. A firm is classified as an **addition** (ADD) if its rank in snapshot $t$ is $\leq K$ while its rank in snapshot $t-1$ was $> K$. Conversely, a firm is classified as a **deletion** (DELETE) if its rank in snapshot $t$ is $> K$ while its rank in snapshot $t-1$ was $\leq K$. Firms that do not appear in a given snapshot (i.e., whose rank is missing) are excluded from event detection; only explicit rank transitions between consecutive snapshots are retained. The rank threshold is set at $K = 600$ for the STOXX 600 analysis and $K = 50$ for the STOXX 50 robustness exercise.

A firm may contribute at most one event per direction per calendar year to prevent event-window overlaps distorting the panel estimates.

### 2.2 Price Data Collection

Daily price data are collected for all firms that appear in at least one membership snapshot of the STOXX 600. The collection window extends from January 2013 to February 2026, providing a sufficiently long history for the estimation of pre-event informational parameters. The raw daily series are cleaned by forward-filling short gaps (up to two consecutive missing trading days) and dropping tickers whose coverage is below a minimum threshold. The dataset for the STOXX 50 sub-sample is bootstrapped from the same underlying price database; only the benchmark ticker (`^STOXX50E`) is appended if not already present.

### 2.3 Monthly Market Model and Informational Efficiency Metrics

For each firm–month observation $(i, t)$, a market model OLS regression is estimated on the available daily return observations within that calendar month:

$$r_{i,d} = \alpha_i + \beta_i \, r_{m,d} + \varepsilon_{i,d}$$

where $r_{i,d}$ is the daily return of stock $i$ on day $d$, and $r_{m,d}$ is the contemporaneous market return. A minimum of **15 trading days** is required in a given month; firm–month observations with fewer observations are set to missing and excluded from subsequent analysis.

From this monthly regression, two informational efficiency metrics (outcome variables) and two liquidity-based control variables are derived:

**Price Synchronicity (Synchronicity).** Following Roll (1988) and Morck, Yeung & Yu (2000), price synchronicity is defined as the logistic transformation of the monthly $R^2$ from the market model:

$$\Psi_{i,t} = \ln\!\left(\frac{R^2_{i,t}}{1 - R^2_{i,t}}\right)$$

A higher value of $\Psi_{i,t}$ indicates that a larger fraction of daily return variation is explained by market-wide factors, which is interpreted as lower _stock-price informativeness_—that is, less firm-specific information is impounded into prices.

**Idiosyncratic Volatility (Idio_Vol).** The standard deviation of the OLS residuals $\hat{\varepsilon}_{i,d}$ within month $t$:

$$\sigma^{\text{idio}}_{i,t} = \text{std}(\hat{\varepsilon}_{i,d})_{d \in t}$$

Higher residual volatility may reflect noise-driven trading or genuine firm-specific information flow after an index event.

**Amihud Illiquidity Ratio (Amihud).** The Amihud (2002) price-impact measure is computed as the monthly average of the daily ratio:

$$\text{ILLIQ}_{i,t} = \overline{\left(\frac{|r_{i,d}|}{P_{i,d} \times V_{i,d}}\right)} \times 10^6$$

where $P_{i,d}$ is the closing price and $V_{i,d}$ is the trading volume (in shares). This ratio captures the price impact per unit of trading volume, and serves as a **control variable** for liquidity effects in the DiD regressions.

**Turnover.** Monthly share turnover is defined as:

$$\text{TO}_{i,t} = \frac{\sum_{d \in t} V_{i,d}}{\text{Shares Outstanding}_{i,t}}$$

Turnover controls for shifts in trading activity that might co-move with index membership changes independently of informational efficiency.

---

## 3. Propensity Score Matching (PSM)

### 3.1 Rationale

A naïve comparison of information efficiency before and after index inclusion would confound the index effect with pre-existing cross-sectional differences between treated (added) and control firms. Propensity Score Matching (PSM) selects, for each treated firm, a control firm from the **non-event universe** that closely resembles the treated firm on dimensions that predict index inclusion and that may independently affect informational efficiency.

### 3.2 Covariate Selection

Three pre-event characteristics are used as matching variables:

1. **Log Market Capitalisation** (`Log_MarketCap`): natural logarithm of the average market capitalisation computed over the 200 trading days preceding the event. Larger firms are systematically more likely to be included in broad indices and to exhibit lower synchronicity.
2. **12-Month Momentum** (`Momentum_12m`): the cumulative return over the 200 trading days prior to the event. Momentum captures price-trend effects that may influence both inclusion likelihood and subsequent return characteristics.
3. **Pre-event Volatility** (`Volatility_pre`): the standard deviation of daily returns over the 200 trading days prior to the event, which is closely related to idiosyncratic volatility.

The three variables are standardised (zero mean, unit variance) prior to propensity score estimation.

### 3.3 Estimation and Matching Procedure

A logistic regression without penalty (i.e., maximum-likelihood logit) is estimated to predict the binary treatment indicator (ADD = 1, control = 0) as a function of the three standardised covariates. The model uses the `lbfgs` solver with a maximum of 1,000 iterations.

Nearest-Neighbour (NN-1) matching without replacement is then performed: each treated firm is matched to the single control firm with the closest propensity score. A **caliper** of $c = 0.01$ (in propensity-score units, baseline specification) is imposed; treated firms for which no control falls within the caliper are discarded as unmatched, generating a minor sample reduction. Matches are assessed via the absolute **standardised mean difference (SMD)** on each covariate:

$$\text{SMD}_k = \frac{\bar{X}^T_k - \bar{X}^C_k}{\sqrt{\tfrac{1}{2}(s^{T2}_k + s^{C2}_k)}}$$

where $T$ and $C$ denote the treated and matched control samples respectively.

### 3.4 DELETE-Event Matching

A symmetric PSM procedure is applied to deletion events to support the asymmetric-effect robustness analysis (Section 5.2). The matching features and estimation protocol are identical to the ADD specification; the outcome of the DELETE matching constitutes an independent matched sample.

---

## 4. Stacked Difference-in-Differences (DiD)

### 4.1 Identification Strategy

The stacked DiD design (Cengiz et al., 2019; Baker et al., 2022) addresses heterogeneous treatment timing by constructing a separate "mini-panel" for each event-cohort and stacking them into a pooled dataset. Each mini-panel contains one treated firm and its PSM-matched control, observed over an event window of $[-12, +12]$ calendar months relative to the event date.

The **identifying assumption** is parallel trends: absent index inclusion, the treated and matched control firm would have evolved along parallel informational efficiency trajectories in the post-event period.

### 4.2 Regression Specification

The main estimating equation is:

$$Y_{i,t} = \alpha_i + \lambda_t + \beta \cdot \text{Treat}_{i} \times \text{Post}_{it} + \gamma_1 \, \text{ILLIQ}_{i,t} + \gamma_2 \, \text{TO}_{i,t} + \varepsilon_{i,t}$$

where:

- $Y_{i,t}$ is either **Synchronicity** ($\Psi_{i,t}$) or **Idiosyncratic Volatility** ($\sigma^{\text{idio}}_{i,t}$)
- $\alpha_i$ is a firm fixed effect absorbing all time-invariant firm-level heterogeneity
- $\lambda_t$ is a calendar-month fixed effect absorbing aggregate time trends common to all firms
- $\text{Treat}_i = 1$ if firm $i$ is the treated (added) firm in a given event-cohort
- $\text{Post}_{it} = 1$ for months $t \geq t_{\text{event}}$ within the event window
- $\beta$ is the DiD coefficient of interest: the average treatment effect of index inclusion on informational efficiency

The regression pools treated and matched control observations across all valid matched pairs, so standard errors are **clustered at the entity (firm) level** to account for serial correlation within firms over the event window (implemented via `linearmodels.PanelOLS` with `cov_type='clustered', cluster_entity=True`, `entity_effects=True`, `time_effects=True`, `drop_absorbed=True`).

### 4.3 Interpretation

- A **negative** and statistically significant $\hat{\beta}$ on $\Psi_{i,t}$ implies a reduction in synchronicity (i.e., prices become more informationally efficient) after index inclusion.
- A **positive** $\hat{\beta}$ implies the opposite: co-movement with the market index increases, consistent with noise trading or index arbitrage channels.
- For $\sigma^{\text{idio}}_{i,t}$, a positive coefficient may reflect greater firm-specific information flow or, alternatively, heightened noise.

---

## 5. Robustness Tests

### 5.1 Placebo Test (Random Treatment Assignment)

To assess whether the estimated $\hat{\beta}$ could arise by chance, a placebo permutation test is conducted. In each of **500 iterations**, the treatment/control label is randomly shuffled within each matched pair (i.e., the matched control is designated as "treated" and vice versa with probability 0.5), the DiD regression is re-estimated, and the placebo coefficient $\hat{\beta}^{\text{placebo}}_s$ is recorded. The empirical $p$-value is the share of placebo iterations in which the placebo coefficient is more extreme (lower, in the case of a negative true $\hat{\beta}$) than the actual estimate:

$$p^{\text{emp}} = \frac{1}{S}\sum_{s=1}^{S} \mathbf{1}\!\left[\hat{\beta}^{\text{placebo}}_s < \hat{\beta}\right], \quad S = 500$$

### 5.2 DELETE-Event Analysis

The ADD DiD specification is replicated on the sample of **index deletion** events. A symmetric directional mechanism would predict that deletions reverse the effect of additions: if inclusions reduce synchronicity, deletions should increase it. Comparing the sign and magnitude of $\hat{\beta}^{\text{ADD}}$ and $\hat{\beta}^{\text{DELETE}}$ provides evidence on whether the index mechanism is genuinely informational or driven by mechanical liquidity/price-pressure effects.

### 5.3 Caliper Sensitivity

The baseline PSM caliper of $c = 0.01$ is varied to $c \in \{0.005, 0.010, 0.050\}$. Stricter calipers (smaller values) trade off sample size against match quality; more permissive calipers include additional—potentially less well-matched—pairs. Stability of the DiD coefficient across caliper values indicates that the results are not an artifact of a particular matching tolerance.

---

## 6. STOXX 50 Sub-Sample (Index Specificity Analysis)

To assess whether the effects documented for the STOXX 600 are specific to broad, diversified indices or also apply to more concentrated, high-profile blue-chip indices, the **complete analysis pipeline** (panel construction, PSM, stacked DiD, robustness) is replicated on the EURO STOXX 50 universe, using $K = 50$ as the rank threshold and `^STOXX50E` as the market benchmark. The smaller rebalancing universe naturally yields a substantially reduced sample of events and matched pairs, implying reduced statistical power; findings should therefore be interpreted in conjunction with the STOXX 600 results.

---

## 7. Limitations

Several methodological caveats merit acknowledgment. First, the event dates span only January 2024 to February 2026, a relatively short and specific macroeconomic period that may not generalise to other regimes. Second, propensity score matching conditions on observable pre-event characteristics; unobserved differences between treated and control firms cannot be ruled out. Third, the parallel-trends assumption is difficult to test directly; the placebo exercise provides indirect evidence but does not constitute a formal pre-trend test. Fourth, the sample size—particularly for the STOXX 50 analysis—limits statistical power, so the absence of significant results should not be interpreted as evidence of no effect.

---

## References

- Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects. _Journal of Financial Markets_, 5(1), 31–56.
- Baker, A. C., Larcker, D. F., & Wang, C. C. Y. (2022). How much should we trust staggered difference-in-differences estimates? _Journal of Financial Economics_, 144(2), 370–395.
- Cengiz, D., Dube, A., Lindner, A., & Zipperer, B. (2019). The effect of minimum wages on low-wage jobs. _Quarterly Journal of Economics_, 134(3), 1405–1454.
- Morck, R., Yeung, B., & Yu, W. (2000). The information content of stock markets: why do emerging markets have synchronous stock price movements? _Journal of Financial Economics_, 58(1–2), 215–260.
- Roll, R. (1988). R². _Journal of Finance_, 43(3), 541–566.
