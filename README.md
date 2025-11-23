# Team 1 – Predicting Scope 1 & Scope 2 Emissions

This project focuses on estimating company greenhouse gas emissions (Scope 1 and Scope 2) when they aren’t reported. These emissions numbers are important for things like climate-risk scoring, investment decisions, and sustainability reporting.  

Our goal was to build a model that can take in company information (like revenue, sector, geography, ESG data, and external activity indicators) and predict their emissions as accurately as possible.

---

## 1. Problem Understanding & Hypothesis

A lot of companies don’t report their own emissions, but organizations still need these values.

### Our Main Hypotheses

- **Emissions grow with revenue, but not in a simple linear way.**  
- **Sector and country matter a lot** because energy sources and industry types vary.  
- **The data is extremely skewed**, so using `np.log1p` on revenue and the targets should help stabilize the model.  
- **Nonlinear models like XGBoost should outperform linear regression** because the relationships are complicated (revenue × sector × geography).

---

## 2. EDA – What We Found

During our exploratory data analysis (EDA), we found several important issues with the dataset. Even though the data was relatively clean (no repeated companies and no missing region info), there were still a few challenges we had to fix before modeling.

| Issue | What We Saw | Why It’s a Problem | Fix |
|-------|-------------|-------------------|-----|
| Extreme right-skew | Revenue and emissions values were extremely large and spread out | Makes training unstable and hurts model accuracy | Apply `np.log1p` to reduce skew |
| Missing ESG data | Some ESG scores were missing | Missing values confuse the model | Fill with median values |
| High feature sparsity | One-hot encoding created a lot of columns, many barely used | Adds noise and slows training | Use `VarianceThreshold` to drop low-variance columns |
| External data inconsistencies | External datasets weren't aligned (different formats, multiple rows per entity, etc.) | Hard to merge with main dataset | Clean and aggregate by `entity_id` |
| Imbalanced feature scales | Some numeric features were huge compared to others | Can distort model behavior | Scale using `StandardScaler` |

---

## 3. Data Engineering

### Log Transformations  
We applied `np.log1p()` to:

- Revenue  
- Scope 1 target  
- Scope 2 target  

This helped reduce skew, improve gradients, and prevent huge emitters from dominating the model.

### Cleaning & Imputation  

- ESG scores → median  
- External dataset features → fill with **0** (meaning “no known activity”)  
- Removed rows missing essential fields

### External Dataset Integration  
We combined three extra datasets:

- Sector revenue shares (converted to %)  
- Environmental activity totals per entity  
- SDG participation indicators (one-hot encoded)

### Encoding & Scaling  

- One-hot encoding for country and region  
- Standard scaling for numeric features  
- VarianceThreshold to drop near-zero columns
  
---

## 4. Model Selection

We tested different models:

| Model | Result | Notes |
|-------|--------|-------|
| Linear Regression | Poor | Couldn’t handle nonlinear behavior |
| LightGBM | Poor | Sometimes good, required larger data |
| XGBoost | Best | Most consistent and strongest performance |

### Why We Picked XGBoost

- Good at nonlinear interactions  
- Works well with sparse one-hot encoded features  
- Strong regularization 
- Stable across folds  
- Handles outliers nicely 

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
- Provides emissions estimates for companies 


