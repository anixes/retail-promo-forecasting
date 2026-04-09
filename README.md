# Retail Promo Impact & Demand Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-informational.svg)](https://www.statsmodels.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-informational.svg)](https://lightgbm.readthedocs.io/)
[![Prophet](https://img.shields.io/badge/Prophet-informational.svg)](https://facebook.github.io/prophet/)

---

## Overview

This project analyzes promotional effectiveness and demand forecasting using Dominick’s Finer Foods retail dataset.

The central finding is structural:

> In a high promotion-saturation environment (~95% of weeks on promo), demand becomes persistence-driven, causing a simple naive baseline to outperform machine learning models.

This project demonstrates that **data-generating conditions—not model complexity—determine predictive performance**.

---

## Key Findings

- **Causal Impact**
  - Promotions generate a statistically significant lift of ~1,900 units
  - Estimated using fixed-effects regression controlling for:
    - Store-level heterogeneity
    - Seasonality
    - Price variation

- **Heterogeneity**
  - No meaningful variation in promotional effectiveness across store segments

- **Forecasting Performance**
  - Naive baseline outperforms Prophet and LightGBM
  - LightGBM improves over Prophet but remains inferior to persistence

- **Root Cause**
  - ~95% promotion saturation removes useful variance
  - Time series becomes dominated by autocorrelation

- **Key Insight**
  - When signal collapses, **model sophistication cannot compensate**

---

## Problem Context

Retail promotions are not randomly assigned—they are typically scheduled during high-demand periods.

This creates **selection bias**, making naive comparisons (promo vs non-promo averages) unreliable.

Additionally, poor demand forecasts lead to:
- Stockouts → lost revenue
- Overstocking → increased holding costs

---

## Methodology

### 1. Data Integrity & Reproducibility (Phase 0)
Raw retail data is highly noisy. Before inference, I ran a strict Data Integrity Audit utilizing the dataset's `OK` flag to prune suspect collection periods. Built an orchestration pipeline (`--task prepare`) that applies standard data engineering principles: enforcing raw data immutability while caching cleaned, continuous chain-level metrics to `data/processed/` for downstream reproducibility.

### 2. Causal Inference (Fixed Effects Regression)

To estimate the *partial effect* of promotions:

- Model: `Sales = Promo + Price + Store FE + Time FE`
- Controls:
  - Store fixed effects → absorbs baseline demand differences
  - Time fixed effects → controls seasonality
  - Price → isolates true promo impact

This treats the dataset as **observational**, not experimental.

---

### 3. Forecasting Pipeline

Built a leakage-safe time series pipeline:

- **Train/Test Split:** Last 12 weeks held out (strictly out-of-sample).
- **Models:**
  - Naive (persistence baseline)
  - Prophet (additive model with regressors)
  - LightGBM (lag + rolling features)
- **Features (LightGBM):**
  - Lag features (t-1, t-52, etc.)
  - Rolling Promo Dynamics (Promo_Roll_4)

---

### 4. Core Diagnostic Insight

The primary result is not model performance—it is **model failure analysis**:

> In a high-saturation regime, promotional activity becomes nearly constant, collapsing useful variance. Under these conditions, persistence-based models establish a performance ceiling that ML models cannot surpass.

---

## Repository Structure

```bash
├── data/
│   ├── raw/                        # Immutable source files
│   └── processed/                  # Cached, cleaned analysis tables
├── notebooks/
│   ├── 01_Inference_EDA.ipynb             # Inference & Data Integrity Proofs
│   └── 02_Forecasting_Diagnostic_EDA.ipynb # ML Failure Diagnostics
├── src/
│   ├── data/                       # ETL & Aggregation logic
│   ├── features/                   # Time-series engineering
│   ├── regression/                 # Fixed Effects Econometrics
│   ├── forecasting/                # Baseline, Prophet, LightGBM Models
│   └── pipelines/                  # Orchestration routines
└── main.py                         # Single-entry CLI
```

---

## Execution

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Data Preparation (Audit & Caching)**
   ```bash
   python main.py --task prepare
   ```
3. **Run Regression Analysis**
   ```bash
   python main.py --task regression
   ```
4. **Run Forecasting Pipeline**
   ```bash
   python main.py --task forecasting
   ```

---

## Business Impact

- **Model Selection Strategy:** Avoid over-engineering models in low-variance regimes. Use baselines as strong benchmarks, not placeholders.
- **Promotional Strategy:** Recognize diminishing returns in saturated promo environments.

## What This Project Demonstrates

1. Causal inference using observational retail data
2. Fixed-effects modeling for unbiased estimation
3. Leakage-free time series forecasting
4. Feature engineering for temporal models
5. **Diagnosing when machine learning fails**
6. Translating statistical results into business decisions

---

## Reflection

This project highlights a non-obvious but critical lesson:

**The limiting factor in predictive systems is often the data, not the model.**

By identifying a saturation-driven regime where ML underperforms, this work emphasizes the importance of understanding the data-generating process before optimizing model complexity.
