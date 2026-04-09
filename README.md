# Dominick's Finer Foods: Retail Demand Analysis
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-informational.svg)](https://www.statsmodels.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-informational.svg)](https://lightgbm.readthedocs.io/)
[![Prophet](https://img.shields.io/badge/Prophet-informational.svg)](https://facebook.github.io/prophet/)

This project engineers an end-to-end analytical pipeline to quantify promotional lift and forecast future demand using the Dominick’s Finer Foods dataset. It bridges the gap between **Econometric Inference** (measuring truth) and **Machine Learning** (predicting the future).

---

## Technical Highlights
*   **Scale**: Processed and analyzed a **6M+ row longitudinal panel** of grocery transactions.
*   **Inference**: **Isolated marginal promotional lift** using high-dimensional Fixed Effects to account for unobserved store-level heterogeneity.
*   **Forecasting**: Benchmarked Gradient Boosting (LightGBM) and additive models (Prophet) against a persistence baseline, identifying a **"Saturation Regime"** (>95% promo intensity) that shifted the data-generating process.

---

## Project Structure
The repository is structured for modularity and reproducibility.

```bash
├── notebooks/
│   ├── 01_Inference_EDA.ipynb      # Proofs for Fixed Effects and Log-scale transformations
│   └── 02_Forecasting_EDA.ipynb    # Visual diagnostic of the "Promo Saturation" regime
├── src/
│   ├── data/                       # Optimized loaders and chain-level aggregators
│   ├── features/                   # Temporal engineering: seasonal lags ($Lag_{52}$) and rolling dynamics
│   ├── regression/                 # Econometric analysis: Fixed Effects and HTE models
│   │   └── analysis.py             # Consolidated econometric modeling logic
│   ├── forecasting/                # Predictive models: Baseline, Prophet, and LightGBM
│   │   ├── lgbm_model.py           # Bias-corrected Gradient Boosting
│   │   └── prophet_model.py        # Multiplicative Bayesian forecasting
│   └── pipelines/                  # Production pipeline orchestration
└── main.py                         # Single entry-point CLI
```

---

## Strategic Methodology

### 1. Data Integrity & Reproducibility (Phase 0)
Raw retail data is highly noisy. Before inference, I ran a **strict Data Integrity Audit** utilizing the dataset's `OK` flag to prune suspect collection periods. Built an orchestration pipeline (`--task prepare`) that applies standard data engineering principles: enforcing raw data immutability while caching cleaned, continuous chain-level metrics to `data/processed/` for downstream reproducibility.

### 2. Causal Impact Analysis (Phase 1)
Instead of relying on simple averages, I implemented **Fixed Effects Regression** to "absorb" the baseline volume variance across 80+ stores. By controlling for store-specific traits and seasonality, the model successfully **quantified a stable ~1,900 unit incremental lift** per promotion.

### 3. Forensic Forecasting (Phase 2)
To ensure forecast integrity, I engineered a **strict temporal validation framework** to mitigate data leakage. The key finding was not a "winner" model, but a **feature degeneracy diagnosis**: in periods of near-constant promotion, persistence models effectively establish a dominant ceiling for ML models unless more granular cross-sectional variance is reintroduced.

---

## Execution

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Pipeline Execution (Audit & Caching)**:
   ```bash
   python main.py --task prepare
   ```

3. **Run Causal Inference Pipeline**:
   ```bash
   python main.py --task regression
   ```

4. **Run Forecasting Showdown**:
   ```bash
   python main.py --task forecasting
   ```

---

### Reflection
> I developed this project to move beyond standard model-fitting. The primary challenge was translating structural data limitations—like promotional saturation—into actionable business insights regarding inventory risk and model selection.
