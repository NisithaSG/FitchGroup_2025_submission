# Team 1 – Scope 1 & Scope 2 Emissions Prediction 
### *Predicting Corporate Greenhouse Gas Emissions Using XGBoost & Advanced Feature Engineering*

---

## 1. Problem Understanding & Hypothesis Setting

### **Business Problem**
Many companies worldwide do not disclose their Scope 1 (direct) or Scope 2 (purchased energy) greenhouse gas emissions.  
However, financial institutions, regulators, and climate-risk analysts require these values for:

- Climate risk scoring  
- Portfolio emissions accounting  
- Alignment with net-zero targets  
- Sustainability reporting (TCFD, ISSB, etc.)

### **Project Goal**
Develop a defensible, scalable, and accurate machine-learning model that predicts Scope 1 & Scope 2 emissions for companies with missing disclosures.

### **Core Hypotheses**

#### **1. Emissions scale *nonlinearly* with revenue**
Larger firms emit more, but not proportionally.

#### **2. Sector and geography strongly influence emissions**
Electric grid intensity, sector behavior, and industrial composition vary by country.

#### **3. Log-transforming targets is essential**
`np.log1p()` stabilizes variance and handles heavy right-skew.

#### **4. Nonlinear tree-based models outperform linear models**
Due to interaction effects:
- Revenue × Sector  
- Sector × Country  
- Country × ESG performance  

---

## 2. Exhaustive EDA – Identifying Key Pain Points

A full exploratory analysis revealed several structural challenges.

### **Key EDA Findings**

| Issue | What We Observed | Impact | Required Fix |
|-------|------------------|--------|--------------|
| Extreme right-skew | Revenue/emissions vary by 10⁶–10⁸× | Unstable gradients | Log-transform revenue & targets |
| Missing ESG scores | 10–25% missing | Model instability | Median imputation |
| Geography inconsistencies | Missing country/region codes | Weak geographic patterns | OHE + imputation |
| Messy external datasets | Duplicates, inconsistent formatting | Leakage & misalignment | Aggregate by entity_id |
| High feature sparsity | Thousands of OHE features | Dimensionality issues | VarianceThreshold |

---

## 3. Data Engineering & Handling Messy Data

Data engineering had the **largest impact** on final model performance.

### **3.1 Log Transformations**
Applied `np.log1p()` to:
- Revenue  
- Scope 1 target  
- Scope 2 target  

**Benefits:**
- Reduces skew  
- Improves gradient stability  
- Preserves order-of-magnitude relationships  
- Reduces influence of extreme emitters  

### **3.2 Cleaning & Imputation**
- ESG scores → **median imputation**  
- External features (SDGs, environmental indicators, sector splits) → **fill with 0** (“no activity known”)  
- Removed companies missing essential fields (e.g., revenue)

### **3.3 External Data Integration**
Integrated **three external datasets**:
1. **Sector revenue shares** → converted to % of total revenue  
2. **Environmental activity file** → summed adjustments by entity  
3. **SDG indicators** → one-hot encoded & aggregated  

### **3.4 Feature Encoding & Scaling**

| Step | Method |
|------|--------|
| Geographic Encoding | One-Hot Encoding (country, region) |
| Scaling | StandardScaler |
| Dimensionality Reduction | VarianceThreshold(0.05) |
| Train/Test Alignment | Reindex test to training columns |

---

## 4. Model Selection & Experimentation

### **Benchmarked Models**

| Model | Performance | Observed Behavior |
|-------|-------------|-------------------|
| Linear Regression | ❌ Poor | Underfit due to nonlinear interactions |
| LightGBM | ⚠️ Mixed | Good for very large datasets; unstable here |
| XGBoost | ✅ Best | Strong, stable, robust for sparse tabular data |

### **Why XGBoost Was Selected**
- Captures **nonlinear** revenue–sector–geography relationships  
- Handles **sparse OHE features** efficiently  
- Built-in **regularization**  
- Excellent performance across **5-fold CV**  
- Robust to outliers (log-space training)

**Final choice:** Train **two separate XGBoost regressors** (Scope 1 & Scope 2).

---

## 5. Model Tweaking & Hyperparameter Tuning

### **Core Hyperparameters**

| Parameter | Value | Purpose |
|-----------|--------|---------|
| n_estimators | 3000 | Allows deep learning with early stopping |
| early_stopping_rounds | 50 | Prevents overfitting |
| learning_rate | 0.03 | Stable convergence |
| max_depth | 8 | Captures complex interactions |
| subsample | 0.9 | Regularization |
| colsample_bytree | 0.9 | Prevent feature dependency |
| eval_metric | "rmse" | Optimized for log-space RMSE |

### **Tuning Strategy**
- Manual search informed by domain intuition  
- 5-fold CV for stability  
- Log-space training for variance control  

---

## 6. Final Evaluation & Business Tie-Back

### **Cross-Validation Results (Log Space)**

| Target | Mean RMSE | Interpretation |
|--------|-----------|----------------|
| **Scope 1** | **1.9405** | Predicts within ~2 logits |
| **Scope 2** | **2.4309** | Harder to model, but stable |

---

## Why This Matters to the Business

| Business Requirement | Model Contribution |
|----------------------|--------------------|
| Fill missing emissions | ✔ Predicts non-negative tCO₂e for all firms |
| Portfolio climate-risk scoring | ✔ Low CV variance → reliable predictions |
| Benchmarking | ✔ Log-space accuracy preserves magnitude |
| Prioritize large emitters | ✔ Correctly identifies high-impact outliers |

### **Business Impact**
- Enables **regulatory-grade emissions estimation**  
- Supports **financial climate stress testing**  
- Fills **global disclosure gaps at scale**  

---

