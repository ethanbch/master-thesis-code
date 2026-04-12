# Empirical Results

## 1. Sample Description and Descriptive Statistics

### 1.1 Panel Overview

The unbalanced panel covers the full STOXX Europe 600 universe and spans from January 2013 to February 2026. After applying the minimum monthly observation threshold of 15 trading days, the panel comprises **130,940 firm–month observations** across **981 unique tickers**.

**Table 1 — Descriptive Statistics of the Panel**

| Variable                 | N       | Mean   | Std Dev | P25    | Median | P75    |
| ------------------------ | ------- | ------ | ------- | ------ | ------ | ------ |
| Synchronicity ($\Psi$)   | 130,940 | −2.287 | 1.980   | −3.536 | −1.924 | −0.899 |
| Idiosyncratic Volatility | 130,940 | 0.0185 | 0.0165  | 0.0112 | 0.0154 | 0.0218 |
| Amihud Illiquidity Ratio | 130,720 | 1.868  | —       | 0.0002 | 0.0019 | 0.0203 |

_Notes: Synchronicity is defined as_ $\ln(R^2 / (1-R^2))$ _from a monthly OLS market model estimated on daily returns with a minimum of 15 observations. Idiosyncratic Volatility is the standard deviation of OLS residuals within each firm–month. The Amihud ratio measures price impact per unit of volume (monthly average of daily ratios scaled by 10⁶)._

The negative mean synchronicity (−2.287) is consistent with the literature: for most European equities, idiosyncratic variation accounts for the majority of daily return variation. The wide interquartile range (IQR ≈ 2.64 log-odds units) reflects substantial cross-sectional heterogeneity in firm-specific information environments.

### 1.2 Event Sample

STOXX 600 reconstitution events over the 2024–2026 period yield a total of **597 events** extracted from 26 consecutive monthly snapshots (January 2, 2024 to February 2, 2026).

**Table 2 — Event Sample Composition**

|                       | STOXX 600               | STOXX 50                |
| --------------------- | ----------------------- | ----------------------- |
| Rebalancing snapshots | 26                      | 26                      |
| ADD events            | 302                     | 36                      |
| DELETE events         | 295                     | 35                      |
| Total events          | 597                     | 71                      |
| Sample period         | 2024-01-02 → 2026-02-02 | 2024-01-02 → 2026-02-02 |

The STOXX 50 sub-sample is substantially smaller by construction: replacing the rank threshold $K = 600$ with $K = 50$ retains only blue-chip reconstitutions, yielding 71 events in total.

---

## 2. Propensity Score Matching Quality

### 2.1 STOXX 600 Matching

Of 201 ADD events with sufficient pre-event price history for feature construction, **196 are successfully matched** within the caliper of $c = 0.01$ (rejection rate: 2.5%). The remaining five events are discarded.

**Table 3 — PSM Match Quality (STOXX 600 ADD Events, Caliper = 0.01)**

| Metric                            | Value     |
| --------------------------------- | --------- |
| Events attempted                  | 201       |
| Valid matched pairs               | 196       |
| Rejection rate                    | 2.5%      |
| Mean propensity-score distance    | 0.000156  |
| Median propensity-score distance  | 0.0000061 |
| P75 propensity-score distance     | 0.0000270 |
| Maximum propensity-score distance | 0.004403  |

**Table 4 — Standardised Mean Differences (SMD) after Matching (STOXX 600)**

| Covariate                 | Mean  | SMD |     |
| ------------------------- | ----- | --- | --- |
| Log Market Capitalisation | 0.618 |
| 12-Month Momentum         | 0.514 |
| Pre-event Volatility      | 0.449 |

_Notes: SMD = (mean_treated − mean_control) / pooled SD. Lower values indicate better covariate balance. A conventional threshold of |SMD| < 0.25 is desirable; residual imbalance here reflects the constraint of matching within a relatively small cross-section of European large-cap equities._

The very low median propensity-score distance (0.0000061) indicates that the majority of pairs are nearly identical in predicted inclusion probability, with only a small number of outlier pairs stretching toward the caliper boundary.

### 2.2 STOXX 50 Matching

Given the smaller event universe, only 27 ADD events had sufficient historical data for PSM. Of these, **20 are matched** within the $c = 0.01$ caliper, corresponding to a rejection rate of 25.9%—considerably higher than for the STOXX 600, reflecting the more limited pool of plausible control firms at the blue-chip tier.

**Table 5 — PSM Match Quality (STOXX 50 ADD Events, Caliper = 0.01)**

| Metric                           | Value    |
| -------------------------------- | -------- |
| Events attempted                 | 27       |
| Valid matched pairs              | 20       |
| Rejection rate                   | 25.9%    |
| Mean propensity-score distance   | 0.001081 |
| Median propensity-score distance | 0.000088 |
| Max propensity-score distance    | 0.005243 |

---

## 3. Main DiD Results — STOXX 600

### 3.1 Price Synchronicity

**Table 6 — DiD Estimates: Effect of Index Inclusion on Price Synchronicity (STOXX 600)**

|                                     | Dependent variable: Synchronicity ($\Psi_{i,t}$) |
| ----------------------------------- | ------------------------------------------------ |
| **DiD coefficient** ($\hat{\beta}$) | **−0.1149**                                      |
| Standard error (clustered by firm)  | 0.0846                                           |
| _t_-statistic                       | −1.359                                           |
| _p_-value                           | 0.174                                            |
| 95% confidence interval             | [−0.281, +0.051]                                 |
|                                     |                                                  |
| Amihud (control)                    | −1.41 × 10⁻⁵                                     |
| Amihud _p_-value                    | 0.617                                            |
| Turnover (control)                  | −7.29 × 10⁻⁸                                     |
| Turnover _p_-value                  | **0.002**                                        |
|                                     |                                                  |
| Fixed effects                       | Entity + Time                                    |
| SE clustering                       | Entity                                           |
| Matched pairs (N)                   | 196                                              |
| Event window                        | ±12 months                                       |

_Notes: Estimation via_ `linearmodels.PanelOLS` _with two-way (entity + calendar-month) fixed effects and entity-clustered standard errors. Post = 1 for months on or after the event date within the ±12-month window. \*\* p < 0.01._

The coefficient $\hat{\beta} = -0.1149$ on the Treat × Post interaction is **negative**, consistent with the hypothesis that STOXX 600 inclusion reduces price synchronicity—interpreted as an improvement in firm-specific price informativeness. However, the coefficient fails to reach conventional significance levels ($p = 0.174$), and the 95% confidence interval comfortably spans zero. The point estimate implies a reduction of 0.115 log-odds units in synchronicity, which is economically modest relative to the cross-sectional standard deviation of 1.980.

Turnover enters significantly ($p = 0.002$), confirming that liquidity shifts co-vary with the index event and must be controlled for to isolate the informational channel.

### 3.2 Idiosyncratic Volatility

**Table 7 — DiD Estimates: Effect of Index Inclusion on Idiosyncratic Volatility (STOXX 600)**

|                                     | Dependent variable: Idio*Vol ($\sigma^{\text{idio}}*{i,t}$) |
| ----------------------------------- | ----------------------------------------------------------- |
| **DiD coefficient** ($\hat{\beta}$) | **+0.000263**                                               |
| Standard error                      | 0.000507                                                    |
| _t_-statistic                       | +0.518                                                      |
| _p_-value                           | 0.605                                                       |
| 95% confidence interval             | [−0.000732, +0.001257]                                      |
| Fixed effects                       | Entity + Time                                               |
| SE clustering                       | Entity                                                      |

The idiosyncratic volatility estimate is positive but statistically indistinguishable from zero ($p = 0.605$). The confidence interval is symmetric and narrow relative to the mean ($\mu = 0.0185$), suggesting that STOXX 600 additions do not systematically alter the dispersion of firm-specific return innovations within a 12-month post-event window.

---

## 4. Robustness Tests (STOXX 600)

### 4.1 Placebo Permutation Test

To assess whether the DiD estimate could be generated by chance under the null of no treatment effect, 500 permutation iterations were conducted by randomly reassigning the treatment label within each matched pair.

**Table 8 — Placebo Permutation Test Results (STOXX 600, Synchronicity)**

| Statistic                     | Value     |
| ----------------------------- | --------- |
| True $\hat{\beta}$            | −0.1149   |
| Placebo iterations (S)        | 500       |
| Valid iterations              | 500       |
| Mean placebo $\hat{\beta}$    | −0.0772   |
| Std dev placebo $\hat{\beta}$ | 0.0439    |
| P5 of placebo distribution    | −0.1519   |
| P95 of placebo distribution   | −0.0065   |
| **Empirical _p_-value**       | **0.200** |

_Notes: Empirical p-value = share of placebo draws with_ $\hat{\beta}^{\text{placebo}} < \hat{\beta}^{\text{true}}$. _A value of 0.200 means that 20% of random permutations produce a coefficient more negative than the actual estimate._

The empirical $p$-value of 0.200 is broadly consistent with the parametric $p$-value of 0.174, confirming that the main result is not an artifact of distributional assumptions. It also reveals that the true estimate lies squarely in the lower tail of the permutation distribution, though not at conventional significance thresholds.

### 4.2 DELETE-Event Analysis

The ADD DiD specification was replicated on matched deletion events (174 valid pairs) to test whether index membership changes have symmetric effects.

**Table 9 — DiD Estimates for DELETE Events (STOXX 600)**

| Outcome                  | $\hat{\beta}$ | SE     | _t_    | _p_-value |
| ------------------------ | ------------- | ------ | ------ | --------- |
| Synchronicity ($\Psi$)   | −0.1183       | 0.0915 | −1.293 | 0.196     |
| Idiosyncratic Volatility | −0.0023       | 0.0013 | −1.773 | 0.076†    |

_† p < 0.10._

The deletion effect on synchronicity ($\hat{\beta} = -0.118$, $p = 0.196$) is of similar sign and magnitude to the addition effect, which is surprising: if index inclusion increases informational efficiency, deletions should do the opposite. The lack of asymmetry is consistent with either (i) transient price-pressure effects around reconstitutions that affect both event types, or (ii) insufficient statistical power to distinguish the direction of the effect.

For idiosyncratic volatility, the DELETE coefficient is negative and marginally significant at the 10% level ($p = 0.076$), suggesting that deletions may marginally reduce firm-specific return volatility—the opposite of what a purely informational channel would predict.

### 4.3 Caliper Sensitivity

The DiD synchronicity estimate was re-estimated under two alternative PSM calipers.

**Table 10 — Caliper Sensitivity Analysis (STOXX 600, Synchronicity)**

| Caliper              | N Pairs | $\hat{\beta}$ | SE         | _t_        | _p_-value |
| -------------------- | ------- | ------------- | ---------- | ---------- | --------- |
| 0.005 (strict)       | 196     | −0.1149       | 0.0846     | −1.359     | 0.174     |
| **0.010 (baseline)** | **196** | **−0.1149**   | **0.0846** | **−1.359** | **0.174** |
| 0.050 (permissive)   | 197     | −0.1115       | 0.0842     | −1.324     | 0.185     |

The main estimates are strikingly stable across all three calipers: the coefficient varies by less than 0.003 units (3%) across specifications, and the $p$-value remains in the range [0.174, 0.185]. This stability indicates that the result is robust to the particular matching tolerance chosen and does not depend on a handful of borderline pairs admitted at the wider caliper.

---

## 5. STOXX 50 Sub-Sample Results

### 5.1 Motivation and Caveats

The STOXX 50 analysis provides an index-specificity check: the EURO STOXX 50 covers only the 50 largest euro-area blue-chip equities and is a high-profile, widely tracked benchmark. Inclusions into this index carry greater signalling value and potentially attract more index-tracking inflows, which could amplify either the price-pressure or the informational channel. The considerably smaller matched sample (20 ADD pairs, 23 DELETE pairs) substantially limits statistical power, so null results must be interpreted with caution.

### 5.2 Main DiD Results

**Table 11 — DiD Estimates: Effect of Index Inclusion on Informational Efficiency (STOXX 50)**

|                                     | Synchronicity    | Idiosyncratic Volatility |
| ----------------------------------- | ---------------- | ------------------------ |
| **DiD coefficient** ($\hat{\beta}$) | **+0.2552**      | **−0.000991**            |
| Standard error                      | 0.2355           | 0.000481                 |
| _t_-statistic                       | +1.083           | −2.059                   |
| _p_-value                           | 0.279            | **0.040**                |
| 95% CI                              | [−0.207, +0.717] | [−0.001936, −0.000046]   |
| Fixed effects                       | Entity + Time    | Entity + Time            |
| SE clustering                       | Entity           | Entity                   |
| Matched pairs (N)                   | 20               | 20                       |

The STOXX 50 synchronicity coefficient is **positive** ($\hat{\beta} = +0.2552$), the opposite sign to the STOXX 600 estimate, though statistically insignificant ($p = 0.279$). This reversal—if real—would suggest that additions to the STOXX 50 _increase_ co-movement with the market index, consistent with index arbitrage and passive-flow effects dominating over any informational improvement for the most liquid blue-chip stocks.

The idiosyncratic volatility estimate is negative and statistically significant at the 5% level ($\hat{\beta} = -0.000991$, $p = 0.040$): STOXX 50 additions are associated with a reduction in idiosyncratic volatility, which is consistent with the price-pressure channel (increased buying pressure reducing idiosyncratic noise) or with risk-sharing effects through increased index investor participation.

### 5.3 Placebo Test (STOXX 50)

| Statistic                          | Value     |
| ---------------------------------- | --------- |
| True $\hat{\beta}$ (Synchronicity) | +0.2552   |
| Placebo iterations                 | 500       |
| Mean placebo $\hat{\beta}$         | −0.0129   |
| Std dev placebo $\hat{\beta}$      | 0.1214    |
| **Empirical _p_-value**            | **0.980** |

The empirical $p$-value of 0.980 indicates that 98% of random permutations produce a coefficient _less positive_ than the actual estimate of +0.2552. This is strong evidence against the null for the STOXX 50: the positive synchronicity effect is unusual under random treatment assignment, in the direction opposite to the STOXX 600 result.

### 5.4 DELETE-Event Analysis (STOXX 50)

**Table 12 — DiD Estimates for DELETE Events (STOXX 50)**

| Outcome                  | $\hat{\beta}$ | SE       | _t_    | _p_-value | N pairs |
| ------------------------ | ------------- | -------- | ------ | --------- | ------- |
| Synchronicity            | +0.1359       | 0.2503   | +0.543 | 0.587     | 23      |
| Idiosyncratic Volatility | −0.000057     | 0.000731 | −0.078 | 0.938     | 23      |

DELETE effects are statistically indistinguishable from zero for both outcomes, consistent with limited power at this sample size.

### 5.5 Caliper Sensitivity (STOXX 50, Synchronicity)

| Caliper              | N Pairs | $\hat{\beta}$ | SE         | _t_        | _p_-value |
| -------------------- | ------- | ------------- | ---------- | ---------- | --------- |
| 0.005 (strict)       | 18      | +0.3204       | 0.2474     | +1.295     | 0.196     |
| **0.010 (baseline)** | **20**  | **+0.2552**   | **0.2355** | **+1.083** | **0.279** |
| 0.050 (permissive)   | 23      | +0.2676       | 0.2314     | +1.156     | 0.248     |

The positive sign is preserved across all three calipers, and estimates are numerically stable (range: +0.255 to +0.320), providing reassurance that the directional result is not driven by the matching tolerance.

---

## 6. Summary and Interpretation

**Table 13 — Summary of Main DiD Coefficients across Specifications**

| Specification    | Outcome       | $\hat{\beta}$ | _p_-value | Sign consistent with efficiency↑? |
| ---------------- | ------------- | ------------- | --------- | --------------------------------- |
| STOXX 600 ADD    | Synchronicity | −0.1149       | 0.174     | ✓ (negative = less co-movement)   |
| STOXX 600 ADD    | Idio_Vol      | +0.000263     | 0.605     | ✓ (positive = more firm-specific) |
| STOXX 600 DELETE | Synchronicity | −0.1183       | 0.196     | ✗ (expected positive)             |
| STOXX 600 DELETE | Idio_Vol      | −0.0023       | 0.076†    | ✗ (expected positive)             |
| STOXX 50 ADD     | Synchronicity | +0.2552       | 0.279     | ✗ (positive = more co-movement)   |
| STOXX 50 ADD     | Idio_Vol      | −0.000991     | 0.040\*   | ✗ (negative = less firm-specific) |

_† p < 0.10; _ p < 0.05.\*

The central finding is one of **statistical non-significance for the STOXX 600**, combined with a directional negative synchronicity coefficient that is _consistent_ with a mild improvement in price informativeness following index inclusion. The point estimate of −0.115 log-odds units translates to roughly 0.06 standard deviations of cross-sectional synchronicity variation—an economically small effect.

Several mechanisms may account for the weak results:

1. **Short event horizon.** The 26 snapshots span less than two years; index effects may require longer observation periods to materialise or may be partially expected by the market in advance.
2. **Counteracting forces.** Index inclusion simultaneously attracts passive institutional investors (price-pressure, increased co-movement) and may stimulate new analyst coverage (informational improvement). The two channels may partially offset.
3. **Sample composition.** The STOXX 600 already represents large, well-covered European equities; the marginal information environment effect of entering such an index may be smaller than for less-visible indices studied in earlier literature.

The **STOXX 50 idiosyncratic volatility result** ($\hat{\beta} = -0.000991$, $p = 0.040$) is the single entry that achieves conventional significance. Its negative sign is consistent with index-driven price stabilisation (passive flows reducing noise) rather than enhanced information processing—a finding that warrants further investigation with a larger event sample as more STOXX 50 reconstitutions accumulate over time.
