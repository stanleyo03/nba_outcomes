A project that we're interested in working on is developing a model to predict NBA players’ career outcomes. Specifically, given a player’s first few years in the NBA, what does their career trajectory look like? This is interesting because we can use past NBA outcomes to predict future outcomes of the young talent in the league. This is especially relevant for NBA scouts and teams so that they can identify the players that are looking to have a "break out" season. We can categorize career outcome in bins like “out of league”, “role player”, “starter”, “all-star”, etc. We can thus model this response using a multinomial logistic regression model or random forest with features relating to players’ season-level performance and accolades.

Most, if not all of the data required for the task described above can be from Basketball Reference. We can use the NBAloveR API: https://cran.rproject.org/web/packages/NBAloveR/index.html. The hoopR package contains valuable play-by-play data. 

---

## Modeling Strategy

All models predict the four-level career outcome factor (Out of League / Role Player / Starter / All-Star) from features averaged over each player's first 1–3 NBA seasons plus draft pick number.

**Features:** `draft_pick`, `avg_ppg`, `avg_rpg`, `avg_apg`, `avg_spg`, `avg_bpg`, `avg_fg_pct`, `avg_fg3_pct`, `avg_ft_pct`, `avg_min_pg`, `n_seasons`

**Train/test split:** 80/20, `set.seed(42)`.

### Model 1: Multinomial Logistic Regression (baseline)

**Package:** `nnet::multinom`

Fast, interpretable baseline. Coefficients show which early-career stats predict each outcome tier. Uncertainty via 500-resample bootstrap 95% CIs on coefficients and a confusion matrix on the held-out test set.

### Model 2: Random Forest

**Package:** `ranger`

Captures non-linear interactions (e.g., high PPG only matters if minutes are also high). OOB error provides a natural uncertainty estimate. Variable importance bar chart shows which stats matter most — readable for a sports audience. Minimal tuning: 500 trees, `mtry` chosen by CV between `floor(sqrt(p))` and `floor(p/3)`.

### Model 3: Bayesian Multinomial Logistic Regression

**Package:** `rstan` (hand-written Stan model)

Gives full posterior distributions over class probabilities for new players, directly satisfying the project's uncertainty-quantification requirement. Priors: `Normal(0, 2.5)` on all coefficients (weakly informative). Reduced predictor set (`draft_pick`, `avg_ppg`, `avg_rpg`, `avg_apg`, `avg_min_pg`, `avg_fg_pct`) to keep sampling fast. 4 chains × 2000 iterations (1000 warmup). Diagnostics: R-hat < 1.05, bulk-ESS. Output: `mcmc_intervals` coefficient plot and posterior class-probability distributions for illustrative player profiles (lottery pick vs. late second-round).

### Model 4: Ordinal Logistic Regression (fallback / complement)

**Package:** `MASS::polr`

Because the outcome tiers are ordered (Out of League < Role Player < Starter < All-Star), ordinal logistic regression is a natural fit. Simpler than multinomial logistic and leverages the ordinal structure for more efficient estimates. Serves as a fallback if the Bayesian model has sampling issues, and as a complement that tests whether the ordinal assumption holds. Uncertainty via standard errors and profile-likelihood 95% CIs.

### Model Comparison

Single summary table reporting test-set accuracy and macro-averaged F1 for all four models, plus (for Model 3) posterior expected calibration. Variable importance / coefficient directions should be consistent across models as a sanity check.
