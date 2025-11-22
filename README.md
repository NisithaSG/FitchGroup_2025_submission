# Team 1 – Predicting Scope 1 & Scope 2 Emissions

This project focuses on estimating company greenhouse gas emissions (Scope 1 and Scope 2) when they aren’t reported. These emissions numbers are important for things like climate-risk scoring, investment decisions, and sustainability reporting.  

Our goal was to build a model that can take in company information (like revenue, sector, geography, ESG data, and external activity indicators) and predict their emissions as accurately as possible.

---

## 1. Problem Understanding & Hypothesis

A lot of companies don’t report emissions, but organizations still need these values. So we wanted to create a method that works even when data is messy or incomplete.

### Our Main Hypotheses

- **Emissions grow with revenue, but not in a simple linear way.**  
- **Sector and country matter a lot** because energy sources and industry types vary.  
- **The data is extremely skewed**, so using `np.log1p` on revenue and the targets should help stabilize the model.  
- **Nonlinear models like XGBoost should outperform linear regression** because the relationships are complicated (revenue × sector × geography).

---

## 2. EDA – What We Found

During exploratory data analysis, we noticed a few big issues:

| Issue | What We Saw | Why It’s a Problem | Fix |
|-------|-------------|-------------------|-----|
| Extreme right-skew | Revenue & emissions vary by millions | Models behave badly | Log-transform |
| Missing ESG data | 10–25% missing | Random noise in model | Median impute |
| Geography gaps | Missing countries/regions | Weak regional patterns | Impute + one-hot encode |
| Messy external files | Duplicate entities | Data leakage risk | Aggregate by entity_id |
| Very sparse features | After encoding | Too many low-variance columns | VarianceThreshold |

---

## 3. Data Engineering

This part took the longest and had the biggest impact on the final model.

### 3.1 Log Transformations  
We applied `np.log1p()` to:

- Revenue  
- Scope 1 target  
- Scope 2 target  

This helped reduce skew, improve gradients, and prevent huge emitters from dominating the model.

### 3.2 Cleaning & Imputation  

- ESG scores → median  
- External dataset features → fill with **0** (meaning “no known activity”)  
- Removed rows missing essential fields

### 3.3 External Dataset Integration  
We combined three extra datasets:

- Sector revenue shares (converted to %)  
- Environmental activity totals per entity  
- SDG participation indicators (one-hot encoded)

### 3.4 Encoding & Scaling  

- One-hot encoding for country and region  
- Standard scaling for numeric features  
- VarianceThreshold to drop near-zero columns  
- Reindexed test set so it matches training columns exactly

---

## 4. Model Selection

We tested different models:

| Model | Result | Notes |
|-------|--------|-------|
| Linear Regression | ❌ Poor | Couldn’t handle nonlinear behavior |
| LightGBM | ⚠️ Mixed | Sometimes good, sometimes unstable |
| XGBoost | ✅ Best | Most consistent and strongest performance |

### Why We Picked XGBoost

- Good at nonlinear interactions  
- Works well with sparse one-hot encoded features  
- Strong regularization  
- Stable across folds  
- Handles outliers nicely (especially in log space)

We trained **two separate XGBoost models**: one for Scope 1 and one for Scope 2.

---

## 5. Hyperparameter Tuning

Here are the most important parameters:

| Parameter | Value | Why |
|----------|--------|------|
| `n_estimators` | 3000 | Allows depth + early stopping |
| `learning_rate` | 0.03 | Stable training |
| `max_depth` | 8 | Captures interactions |
| `subsample` | 0.9 | Regularization |
| `colsample_bytree` | 0.9 | Avoids feature dependence |
| `early_stopping_rounds` | 50 | Prevents overfitting |
| `eval_metric` | rmse | Matches our goal |

We tuned using manual search + 5-fold cross-validation.

---

## 6. Final Evaluation & Business Impact

### Cross-Validation Scores (Log Space)

| Target | Mean RMSE | Notes |
|--------|------------|-------|
| **Scope 1** | **1.9405** | Predictions are stable |
| **Scope 2** | **2.4309** | Harder target but still consistent |

### Why This Matters

Even though this is a machine learning project, it ties directly to real applications:

| Business Need | How Our Model Helps |
|----------------|----------------------|
| Fill missing emissions values | Predicts realistic non-negative numbers |
| Portfolio risk scoring | Low variance → stable results |
| Emissions comparisons | Log RMSE preserves scale relationships |
| Identifying large emitters | Good at ranking by magnitude |

### Big Picture Impact

- Helps financial institutions understand climate exposure  
- Supports regulatory reporting  
- Provides emissions estimates for companies that never disclosed them  


